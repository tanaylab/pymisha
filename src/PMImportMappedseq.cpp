/*
 * PMImportMappedseq.cpp
 *
 * Port of R misha GenomeTrackImportMappedseq.cpp.
 * Task 3: per-byte FSM SAM/tab parser + gzip auto-detect.
 * Track writers come in Tasks 4-5.
 *
 * R parity: exact chrom-name match (no normalization), tab-only split.
 * pymisha extension: gzip auto-detect via zlib (0x1f 0x8b magic bytes).
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <sys/stat.h>
#include <sys/wait.h>

#include <zlib.h>

#include <Python.h>

#include "pymisha.h"
#include "PMDb.h"
#include "PMImportMappedseq.h"
#include "BufferedFile.h"
#include "GenomeTrack.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeTrackSparse.h"
#include "GInterval.h"
#include "IndexedTrackWriter.h"
#include "TrackIndex.h"

using namespace std;

namespace {

enum { SEQ_COL, CHROM_COL, COORD_COL, STRAND_COL, NUM_COLS };
static const char *COL_NAMES[NUM_COLS] = {
    "sequence", "chromosome", "coordinate", "strand"
};

static bool file_exists_pm(const string &path) {
    struct stat st; return stat(path.c_str(), &st) == 0;
}

static bool db_is_indexed_pm() {
    if (!g_pmdb || !g_pmdb->is_initialized()) return false;
    const std::string base = g_pmdb->groot() + "/seq/";
    return file_exists_pm(base + "genome.idx") && file_exists_pm(base + "genome.seq");
}

static void ensure_track_dir_pm(const std::string &track_dir) {
    namespace fs = std::filesystem;
    fs::path p(track_dir);
    fs::create_directories(p.parent_path());
    fs::create_directories(p);
}

// --------------- ByteSource abstraction (plain + gzip) -----------------------

class ByteSource {
public:
    virtual ~ByteSource() = default;
    virtual int getc() = 0;
    virtual bool error() const = 0;
};

class PlainSource : public ByteSource {
    BufferedFile buf_;
public:
    explicit PlainSource(const std::string &path) {
        if (buf_.open(path.c_str(), "r") != 0 || buf_.error())
            verror("Failed to open %s: %s", path.c_str(), strerror(errno));
    }
    int getc() override { return buf_.getc(); }
    bool error() const override { return buf_.error() != 0; }
};

class GzipSource : public ByteSource {
    gzFile gz_;
public:
    explicit GzipSource(const std::string &path) : gz_(nullptr) {
        gz_ = gzopen(path.c_str(), "rb");
        if (!gz_)
            verror("Failed to open gzipped %s: %s", path.c_str(), strerror(errno));
    }
    ~GzipSource() override { if (gz_) gzclose(gz_); }
    int getc() override { return gzgetc(gz_); }
    bool error() const override {
        if (!gz_) return true;
        int err = 0;
        gzerror(gz_, &err);
        return err != Z_OK && err != Z_STREAM_END;
    }
};

// FdSource: wraps an arbitrary file descriptor (used by the "-" / "fd:N"
// public paths for shell-pipeline composition). Ownership of the fd is
// transferred to FdSource; fclose() closes the fd which signals EOF to the
// writer. Callers that need the fd to stay open elsewhere must dup() first.
class FdSource : public ByteSource {
    FILE *fp_;
public:
    explicit FdSource(int fd) : fp_(nullptr) {
        fp_ = fdopen(fd, "rb");
        if (!fp_)
            verror("Failed to fdopen fd %d: %s", fd, strerror(errno));
    }
    ~FdSource() override {
        if (fp_) fclose(fp_);
    }
    int getc() override { return ::getc(fp_); }
    bool error() const override { return fp_ ? ferror(fp_) != 0 : true; }
};

// PipeSource: popen-based source. Used for BAM auto-detect (samtools view).
// finish() is called explicitly after the FSM has drained the pipe so the
// caller can inspect the child's exit status; pclose in the destructor is a
// fallback for the exception path.
class PipeSource : public ByteSource {
    FILE *fp_;
    std::string cmd_;
    bool finished_;
    int finish_status_;
public:
    explicit PipeSource(const std::string &cmd)
        : fp_(nullptr), cmd_(cmd), finished_(false), finish_status_(-1)
    {
        fp_ = popen(cmd.c_str(), "r");
        if (!fp_)
            verror("Failed to popen '%s': %s", cmd.c_str(), strerror(errno));
    }
    ~PipeSource() override {
        if (fp_) pclose(fp_);  // best-effort cleanup on exception paths
    }
    int getc() override { return ::getc(fp_); }
    bool error() const override { return fp_ ? ferror(fp_) != 0 : true; }
    // Drains the rest of the pipe and waits for the child. Returns the raw
    // status from pclose (use WEXITSTATUS to extract the exit code).
    int finish() {
        if (finished_) return finish_status_;
        finished_ = true;
        finish_status_ = pclose(fp_);
        fp_ = nullptr;
        return finish_status_;
    }
    const std::string &cmd() const { return cmd_; }
};

// Single-quote escape for paths handed to /bin/sh via popen: every ' becomes
// '\'' and the whole string is wrapped in '...'. Safe against shell
// metacharacters in filesystem paths.
static std::string shellquote_single(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') out.append("'\\''");
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

// Returns 0 if no magic bytes could be read, otherwise fills buf and returns
// number of bytes read (up to nbuf).
static size_t read_magic_bytes(const std::string &path, unsigned char *buf, size_t nbuf) {
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp) return 0;
    size_t n = fread(buf, 1, nbuf, fp);
    fclose(fp);
    return n;
}

static std::unique_ptr<ByteSource> open_source(const std::string &path) {
    // stdin shorthand
    if (path == "-")
        return std::unique_ptr<ByteSource>(new FdSource(0));
    // explicit fd: "fd:N"
    if (path.size() > 3 && path.compare(0, 3, "fd:") == 0) {
        char *endptr = nullptr;
        long fd = strtol(path.c_str() + 3, &endptr, 10);
        if (*endptr != '\0' || fd < 0)
            verror("Invalid fd path: %s (expected fd:N for non-negative integer N)", path.c_str());
        return std::unique_ptr<ByteSource>(new FdSource((int)fd));
    }
    // Regular path: peek magic bytes. BAM is bgzip = gzip with FLG=0x04
    // (FEXTRA - block-size subfield); plain gzip has FLG=0x00 or 0x08, so
    // byte 3 distinguishes BAM. Pipe BAM through samtools view; gzip and
    // plain stay as before.
    unsigned char magic[4] = {0, 0, 0, 0};
    size_t n = read_magic_bytes(path, magic, 4);
    if (n == 4 && magic[0] == 0x1f && magic[1] == 0x8b &&
                  magic[2] == 0x08 && magic[3] == 0x04) {
        const std::string cmd = "samtools view " + shellquote_single(path);
        return std::unique_ptr<ByteSource>(new PipeSource(cmd));
    }
    if (n >= 2 && magic[0] == 0x1f && magic[1] == 0x8b)
        return std::unique_ptr<ByteSource>(new GzipSource(path));
    return std::unique_ptr<ByteSource>(new PlainSource(path));
}

}  // namespace

PyObject *pm_import_mappedseq(PyObject *self, PyObject *args)
{
    (void)self;
    try {
        PyMisha pymisha(true);

        const char *track_dir_cstr = nullptr;
        const char *file_path_cstr = nullptr;
        long pileup = 0;
        long binsize = -1;
        PyObject *py_cols_order = nullptr;
        int remove_dups_int = 0;

        if (!PyArg_ParseTuple(args, "ssllOp",
                              &track_dir_cstr, &file_path_cstr,
                              &pileup, &binsize,
                              &py_cols_order, &remove_dups_int))
            verror("Invalid arguments to pm_import_mappedseq");

        const string track_dir(track_dir_cstr);
        const string file_path(file_path_cstr);

        // cols_order: None -> SAM defaults; else 4-int sequence (1-based).
        int cols_order[NUM_COLS];
        const bool is_sam_format = (py_cols_order == Py_None);
        if (is_sam_format) {
            cols_order[SEQ_COL]    = 10;
            cols_order[CHROM_COL]  = 3;
            cols_order[COORD_COL]  = 4;
            cols_order[STRAND_COL] = 2;
        } else {
            PyObject *seq = PySequence_Fast(py_cols_order,
                "cols_order must be a sequence of 4 ints or None");
            if (!seq)
                verror("cols_order must be a sequence of 4 ints or None");
            if (PySequence_Fast_GET_SIZE(seq) != NUM_COLS) {
                Py_DECREF(seq);
                verror("cols_order must have %d entries", NUM_COLS);
            }
            for (int i = 0; i < NUM_COLS; ++i) {
                PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                long v = PyLong_AsLong(item);
                if (PyErr_Occurred()) {
                    Py_DECREF(seq);
                    verror("cols_order[%d] is not an integer", i);
                }
                cols_order[i] = (int)v;
            }
            Py_DECREF(seq);
        }

        if (pileup < 0)
            verror("pileup cannot be negative");
        if (pileup == 0 && binsize >= 0)
            verror("For pileup=0 (sparse), binsize must be -1");
        if (pileup > 0 && binsize <= 0)
            verror("For pileup>0 (dense), binsize must be > 0");

        for (int i = 0; i < NUM_COLS; ++i) {
            if (cols_order[i] <= 0)
                verror("Invalid cols_order: %s index is %d (must be >= 1)",
                       COL_NAMES[i], cols_order[i]);
            for (int j = i + 1; j < NUM_COLS; ++j) {
                if (cols_order[i] == cols_order[j])
                    verror("Invalid cols_order: %s and %s share index %d",
                           COL_NAMES[i], COL_NAMES[j], cols_order[i]);
            }
        }

        const bool is_fd_path = (file_path == "-" ||
            (file_path.size() > 3 && file_path.compare(0, 3, "fd:") == 0));
        if (!is_fd_path && !file_exists_pm(file_path))
            verror("File not found: %s", file_path.c_str());

        if (!g_pmdb || !g_pmdb->is_initialized())
            verror("Database not initialized. Call gdb_init() first.");

        const auto &chromkey = g_pmdb->chromkey();
        const uint32_t num_chroms = (uint32_t)chromkey.get_num_chroms();

        // Build chrom name -> index map (R lines 96-104).
        std::unordered_map<std::string, int> str2chrom;
        str2chrom.reserve(num_chroms);
        std::vector<int64_t> chrom_ends(num_chroms);
        for (int i = 0; i < (int)num_chroms; ++i) {
            str2chrom[chromkey.id2chrom(i)] = i;
            chrom_ends[i] = (int64_t)chromkey.get_chrom_size(i);
        }

        // Per-chrom counters and coordinate storage.
        // coords[0..num_chroms) = plus strand, coords[num_chroms..2*num_chroms) = minus strand.
        std::vector<uint64_t> num_mapped(num_chroms, 0);
        std::vector<uint64_t> num_dups(num_chroms, 0);  // populated by writer tasks (4-5)
        std::vector<std::vector<int64_t>> coords(2 * num_chroms);

        auto src = open_source(file_path);
        int64_t total_unmapped = 0;

        // FSM state (mirrors R lines 119-225).
        int col = 1;
        int active_col_idx = -1;
        int pos = 0;
        std::string col_buf[NUM_COLS];

        // Prime active_col_idx for col=1.
        for (int i = 0; i < NUM_COLS; ++i) {
            if (cols_order[i] == 1) { active_col_idx = i; break; }
        }

        while (true) {
            int c = src->getc();

            // SAM '@' header skip - only at start of line, only in SAM mode.
            if (!pos && is_sam_format && c == '@') {
                while (true) {
                    c = src->getc();
                    if (c == '\n' || c == EOF) break;
                }
                if (c == EOF) break;
                continue;
            }
            ++pos;

            if (c == '\n' || c == EOF || c == '\t') {
                if (c == '\n' || c == EOF) {
                    int num_nonempty = 0;
                    bool mapped = false;
                    pos = 0;
                    for (int i = 0; i < NUM_COLS; ++i)
                        if (!col_buf[i].empty()) ++num_nonempty;

                    // R's one-shot block (mirrors `while (num_nonempty_strs == NUM_COLS) {...break;}`).
                    if (num_nonempty == NUM_COLS) {
                        auto it = str2chrom.find(col_buf[CHROM_COL]);
                        if (it != str2chrom.end()) {
                            int chrom_idx = it->second;
                            char *endptr = nullptr;
                            int64_t coord = strtoll(col_buf[COORD_COL].c_str(), &endptr, 10);
                            if (*endptr == '\0' && coord >= 0 && coord < chrom_ends[chrom_idx]) {
                                std::string strand;
                                bool strand_ok = true;
                                if (is_sam_format) {
                                    char *fendp = nullptr;
                                    uint64_t flag = (uint64_t)strtoll(col_buf[STRAND_COL].c_str(), &fendp, 0);
                                    if (*fendp != '\0') strand_ok = false;
                                    else strand = (flag & 0x10) ? "-" : "+";
                                } else {
                                    strand = col_buf[STRAND_COL];
                                }
                                if (strand_ok) {
                                    if (strand == "+" || strand == "F") {
                                        coords[chrom_idx].push_back(coord);
                                        ++num_mapped[chrom_idx];
                                        mapped = true;
                                    } else if (strand == "-" || strand == "R") {
                                        coords[num_chroms + chrom_idx].push_back(
                                            coord + (int64_t)col_buf[SEQ_COL].size());
                                        ++num_mapped[chrom_idx];
                                        mapped = true;
                                    }
                                }
                            }
                        }
                    }

                    if (!mapped && num_nonempty > 0)
                        ++total_unmapped;

                    if (c == EOF) break;

                    if (num_nonempty > 0) {
                        for (int i = 0; i < NUM_COLS; ++i) col_buf[i].clear();
                    }
                    col = 1;
                } else {
                    ++col;
                }

                active_col_idx = -1;
                for (int i = 0; i < NUM_COLS; ++i) {
                    if (cols_order[i] == col) { active_col_idx = i; break; }
                }
            } else if (active_col_idx >= 0) {
                col_buf[active_col_idx].push_back((char)c);
            }
        }

        if (src->error())
            verror("Error while reading %s", file_path.c_str());

        // If we routed through samtools (BAM auto-detect), drain the pipe and
        // surface a non-zero exit. 127 is "command not found" - tell the user
        // samtools isn't installed; other codes mean samtools itself failed.
        if (auto *pipe_src = dynamic_cast<PipeSource *>(src.get())) {
            int raw_status = pipe_src->finish();
            if (raw_status != 0) {
                int code = WIFEXITED(raw_status) ? WEXITSTATUS(raw_status) : -1;
                if (code == 127) {
                    verror("BAM input detected at %s but samtools is not on PATH. "
                           "Install samtools (e.g. `apt-get install samtools` or "
                           "`conda install -c bioconda samtools`) or pre-convert: "
                           "`samtools view %s > %s.sam`.",
                           file_path.c_str(), file_path.c_str(), file_path.c_str());
                } else {
                    verror("samtools view %s exited with code %d (raw status %d). "
                           "Run `samtools view %s | head` to see the underlying error.",
                           file_path.c_str(), code, raw_status, file_path.c_str());
                }
            }
        }

        // Write track files (Tasks 4-5).
        ensure_track_dir_pm(track_dir);

        const bool indexed_db = db_is_indexed_pm();
        std::unique_ptr<IndexedTrackWriter> indexed_writer;
        if (indexed_db) {
            MishaTrackType type = (pileup > 0) ? MishaTrackType::DENSE : MishaTrackType::SPARSE;
            indexed_writer.reset(new IndexedTrackWriter(track_dir, type, num_chroms));
        }

        if (pileup > 0) {
            const uint32_t bs_u32 = (uint32_t)binsize;
            const bool remove_dups = (remove_dups_int != 0);
            for (uint32_t chrom = 0; chrom < num_chroms; ++chrom) {
                int64_t chrom_end = chrom_ends[chrom];
                int64_t nbins = (int64_t)std::ceil((double)chrom_end / (double)binsize);
                if (nbins < 0) nbins = 0;

                GenomeTrackFixedBin gtrack;
                if (indexed_db) {
                    indexed_writer->begin_chrom((int)chrom);
                    indexed_writer->stream_bytes(&bs_u32, sizeof(bs_u32));
                } else {
                    std::string filename = track_dir + "/" + chromkey.id2chrom((int)chrom);
                    gtrack.init_write(filename.c_str(), bs_u32, (int)chrom);
                }

                std::vector<float> trackvals((size_t)nbins, 0.0f);

                for (int strand = 0; strand < 2; ++strand) {
                    std::vector<int64_t> &cur = coords[strand * num_chroms + chrom];
                    std::sort(cur.begin(), cur.end());

                    for (size_t k = 0; k < cur.size(); ++k) {
                        int64_t coord = cur[k];
                        if (remove_dups && k > 0 && coord == cur[k - 1]) {
                            ++num_dups[chrom];
                            continue;
                        }
                        int64_t from_coord = std::max<int64_t>(
                            strand ? coord - (int64_t)pileup : coord, (int64_t)0);
                        int64_t to_coord = std::min<int64_t>(
                            strand ? coord : coord + (int64_t)pileup, chrom_end);
                        int64_t from_bin = (int64_t)(from_coord / binsize);
                        int64_t to_bin = (int64_t)std::ceil((double)to_coord / (double)binsize) - 1;

                        if (from_bin >= to_bin) {
                            trackvals[from_bin] +=
                                (float)((double)(to_coord - from_coord) / (double)binsize);
                        } else {
                            trackvals[from_bin] +=
                                (float)((double)(from_bin + 1) - (double)from_coord / (double)binsize);
                            trackvals[to_bin]   +=
                                (float)((double)to_coord / (double)binsize - (double)to_bin);
                            for (int64_t b = from_bin + 1; b < to_bin; ++b)
                                trackvals[b] += 1.0f;
                        }
                    }
                }

                if (indexed_db) {
                    indexed_writer->stream_bytes(trackvals.data(),
                                                 trackvals.size() * sizeof(float));
                    indexed_writer->end_chrom();
                } else {
                    if (!trackvals.empty())
                        gtrack.write_next_bins(trackvals.data(), trackvals.size());
                }
            }
            if (indexed_db) indexed_writer->finish();
        } else {  // pileup == 0  ->  sparse
            const bool remove_dups = (remove_dups_int != 0);
            const int32_t sparse_sig = GenomeTrack::FORMAT_SIGNATURES[GenomeTrack::SPARSE];

            for (uint32_t chrom = 0; chrom < num_chroms; ++chrom) {
                GenomeTrackSparse gtrack;
                if (indexed_db) {
                    indexed_writer->begin_chrom((int)chrom);
                    indexed_writer->stream_bytes(&sparse_sig, sizeof(sparse_sig));
                } else {
                    std::string filename = track_dir + "/" + chromkey.id2chrom((int)chrom);
                    gtrack.init_write(filename.c_str(), (int)chrom);
                }

                std::vector<int64_t> &plus  = coords[chrom];
                std::vector<int64_t> &minus = coords[num_chroms + chrom];
                std::sort(plus.begin(),  plus.end());
                std::sort(minus.begin(), minus.end());

                size_t i = 0, j = 0;
                while (i < plus.size() || j < minus.size()) {
                    float val = 0.0f;
                    int64_t coord = -1;

                    // R lines 289-296: plus branch (when plus has items and either
                    // minus is exhausted or minus[j] >= plus[i]).
                    if (i < plus.size() && (j >= minus.size() || minus[j] >= plus[i])) {
                        val = std::max(val + (remove_dups ? 0.0f : 1.0f), 1.0f);
                        coord = plus[i++];
                        while (i < plus.size() && plus[i] == coord) {
                            ++num_dups[chrom];                  // R counts dups regardless of remove_dups
                            if (!remove_dups) val += 1.0f;
                            ++i;
                        }
                    }

                    // R lines 298-305: minus branch (when minus has items AND either
                    // plus didn't fire this iter (coord==-1) OR minus[j] == coord).
                    if (j < minus.size() && (coord == -1 || minus[j] == coord)) {
                        val = std::max(val + (remove_dups ? 0.0f : 1.0f), 1.0f);
                        coord = minus[j++];
                        while (j < minus.size() && minus[j] == coord) {
                            ++num_dups[chrom];
                            if (!remove_dups) val += 1.0f;
                            ++j;
                        }
                    }

                    if (coord < 0) continue;  // defensive (should be unreachable)

                    if (indexed_db) {
                        int64_t s = coord;
                        int64_t e = coord + 1;
                        indexed_writer->stream_bytes(&s,   sizeof(s));
                        indexed_writer->stream_bytes(&e,   sizeof(e));
                        indexed_writer->stream_bytes(&val, sizeof(val));
                    } else {
                        gtrack.write_next_interval(
                            GInterval((int)chrom, coord, coord + 1, (char)0), val);
                    }
                }

                if (indexed_db) {
                    indexed_writer->end_chrom();
                }
                // Per-chrom-file path: GenomeTrackSparse::init_write already wrote
                // the signature in init_write; if no intervals were added we leave
                // the file as just the 4-byte signature (matches R behavior).
            }
            if (indexed_db) indexed_writer->finish();
        }

        // Build the result dict with REAL counters.
        PyObject *total = PyDict_New();
        uint64_t tot_mapped = 0, tot_dups = 0;
        for (uint32_t i = 0; i < num_chroms; ++i) {
            tot_mapped += num_mapped[i];
            tot_dups   += num_dups[i];
        }
        PyDict_SetItemString(total, "total",
            PyFloat_FromDouble((double)(tot_mapped + (uint64_t)total_unmapped + tot_dups)));
        PyDict_SetItemString(total, "total.mapped",   PyFloat_FromDouble((double)tot_mapped));
        PyDict_SetItemString(total, "total.unmapped", PyFloat_FromDouble((double)total_unmapped));
        PyDict_SetItemString(total, "total.dups",     PyFloat_FromDouble((double)tot_dups));

        PyObject *chroms_l = PyList_New(num_chroms);
        PyObject *mapped_l = PyList_New(num_chroms);
        PyObject *dups_l   = PyList_New(num_chroms);
        for (uint32_t i = 0; i < num_chroms; ++i) {
            PyList_SET_ITEM(chroms_l, i,
                            PyUnicode_FromString(chromkey.id2chrom((int)i).c_str()));
            PyList_SET_ITEM(mapped_l, i, PyLong_FromUnsignedLongLong((unsigned long long)num_mapped[i]));
            PyList_SET_ITEM(dups_l,   i, PyLong_FromUnsignedLongLong((unsigned long long)num_dups[i]));
        }
        PyObject *chrom_stats = PyDict_New();
        PyDict_SetItemString(chrom_stats, "chrom",  chroms_l);
        PyDict_SetItemString(chrom_stats, "mapped", mapped_l);
        PyDict_SetItemString(chrom_stats, "dups",   dups_l);
        Py_DECREF(chroms_l); Py_DECREF(mapped_l); Py_DECREF(dups_l);

        PyObject *result = PyDict_New();
        PyDict_SetItemString(result, "total",       total);
        PyDict_SetItemString(result, "chrom_stats", chrom_stats);
        Py_DECREF(total); Py_DECREF(chrom_stats);

        return result;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}
