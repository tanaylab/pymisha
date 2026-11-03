/*
 * PMImportMappedseq.cpp
 *
 * Port of R misha GenomeTrackImportMappedseq.cpp (gtrackimport_mappedseq):
 * per-byte FSM SAM/tab parser, SAM flag / MAPQ filters, paired-end fragments,
 * plain / gzip / BAM (samtools) / named-pipe input.
 *
 * pymisha additions: "-" and "fd:N" read from a file descriptor, and an indexed
 * database is written through IndexedTrackWriter.
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

// The first NUM_USER_COLS columns are the ones cols_order positions; the rest are read only from SAM
// (MAPQ, CIGAR, PNEXT, TLEN) or from fragment files (END). A column with order 0 is not read.
enum { SEQ_COL, CHROM_COL, COORD_COL, STRAND_COL, NUM_USER_COLS, MAPQ_COL = NUM_USER_COLS, CIGAR_COL, PNEXT_COL, TLEN_COL, END_COL, NUM_COLS };
static const char *COL_NAMES[NUM_COLS] = {
    "sequence", "chromosome", "coordinate", "strand", "mapq", "cigar", "pnext", "tlen", "end"
};
// SAM flags: unmapped (0x4), secondary (0x100), QC fail (0x200) and supplementary (0x800) records are never imported
static const uint64_t SAM_SKIP_FLAGS = 0x4 | 0x100 | 0x200 | 0x800;

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
    std::string head_; // bytes read ahead by peek(), served before the rest of the stream
    size_t head_pos_ = 0;
public:
    explicit GzipSource(const std::string &path) : gz_(nullptr) {
        gz_ = gzopen(path.c_str(), "rb");
        if (!gz_)
            verror("Failed to open gzipped %s: %s", path.c_str(), strerror(errno));
    }
    ~GzipSource() override { if (gz_) gzclose(gz_); }
    int getc() override { return head_pos_ < head_.size() ? (unsigned char)head_[head_pos_++] : gzgetc(gz_); }
    // The first n (decompressed) bytes, without consuming them; works on a pipe.
    const std::string &peek(size_t n) {
        int c;
        while (head_.size() < n && (c = gzgetc(gz_)) != EOF)
            head_.push_back((char)c);
        return head_;
    }
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

// True if the decompressed stream starts with the BAM magic "BAM\1".
static bool has_bam_payload(const std::string &path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) return false;
    char magic[4];
    int n = gzread(gz, magic, 4);
    gzclose(gz);
    return n == 4 && !memcmp(magic, "BAM\1", 4);
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
    // A FIFO or other non-regular file cannot be sniffed: reading its first bytes consumes them and closes
    // the only reader. zlib reads plain or gzipped text from it as a stream.
    struct stat st;
    if (::stat(path.c_str(), &st) == 0 && !S_ISREG(st.st_mode)) {
        auto src = std::unique_ptr<GzipSource>(new GzipSource(path));
        if (src->peek(4) == std::string("BAM\1", 4))
            verror("%s is a pipe carrying BAM, which misha reads only from a regular file. "
                   "Send SAM through the pipe instead (samtools view -h).", path.c_str());
        return src;
    }

    // Regular path: peek magic bytes. bgzip uses gzip magic with FLG.FEXTRA (0x04) and method 0x08; that
    // distinguishes BAM/bgzip from plain gzip (which has FLG 0x00 or 0x08). A bgzipped text file (e.g. a
    // 10x fragments.tsv.gz) is read with zlib.
    unsigned char magic[4] = {0, 0, 0, 0};
    size_t n = read_magic_bytes(path, magic, 4);
    if (n == 4 && magic[0] == 0x1f && magic[1] == 0x8b &&
                  magic[2] == 0x08 && magic[3] == 0x04 && has_bam_payload(path)) {
        const std::string cmd = "samtools view " + shellquote_single(path);
        return std::unique_ptr<ByteSource>(new PipeSource(cmd));
    }
    if (n >= 2 && magic[0] == 0x1f && magic[1] == 0x8b)
        return std::unique_ptr<ByteSource>(new GzipSource(path));
    return std::unique_ptr<ByteSource>(new PlainSource(path));
}

// Reference length of a CIGAR (M, D, N, = and X operations); -1 for "*", a malformed CIGAR or an absurd length.
static int64_t cigar_ref_span(const std::string &cigar) {
    const int64_t MAX_LEN = (int64_t)1 << 40; // keeps len and span far from overflow
    int64_t span = 0, len = 0;
    bool has_len = false;
    for (char ch : cigar) {
        if (ch >= '0' && ch <= '9') {
            len = len * 10 + (ch - '0');
            if (len >= MAX_LEN)
                return -1;
            has_len = true;
            continue;
        }
        if (!has_len)
            return -1;
        if (ch == 'M' || ch == 'D' || ch == 'N' || ch == '=' || ch == 'X') {
            span += len;
            if (span >= MAX_LEN)
                return -1;
        }
        else if (ch != 'I' && ch != 'S' && ch != 'H' && ch != 'P')
            return -1;
        len = 0;
        has_len = false;
    }
    return has_len ? -1 : span;
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
        int paired_int = 0;
        long min_mapq = 0;
        long long max_fraglen = 2000;
        int one_based_int = 0;

        if (!PyArg_ParseTuple(args, "ssllOp|plLp",
                              &track_dir_cstr, &file_path_cstr,
                              &pileup, &binsize,
                              &py_cols_order, &remove_dups_int,
                              &paired_int, &min_mapq, &max_fraglen, &one_based_int))
            verror("Invalid arguments to pm_import_mappedseq");

        const string track_dir(track_dir_cstr);
        const string file_path(file_path_cstr);
        const bool remove_dups = (remove_dups_int != 0);
        const bool paired = (paired_int != 0);

        // cols_order: None -> SAM; else 4-int sequence (1-based).
        int cols_order[NUM_COLS] = { 0 };
        const bool is_sam_format = (py_cols_order == Py_None);
        // paired on a non-SAM file: tab-delimited fragments, 0-based half-open chrom/start/end in columns 1-3
        // (BED, 10x fragments.tsv.gz). Fragment files are taken as already deduplicated.
        const bool is_frag_format = paired && !is_sam_format;
        // SAM POS is 1-based by the spec; a tab-delimited coordinate is 1-based only when one_based is set
        const bool one_based = is_sam_format || one_based_int;
        // Tab-delimited files without one_based keep the historical placement: the coordinate is taken as is,
        // and a reverse read is recorded one base past its 5' end (its end-exclusive coordinate).
        const bool legacy = !is_sam_format && !is_frag_format && !one_based;
        const int64_t rev_end_off = legacy ? 0 : 1; // recorded reverse-read coordinate + rev_end_off = end-exclusive coordinate

        if (is_frag_format && one_based_int)
            verror("one_based is not used for fragment files: their coordinates are 0-based, half-open");

        if (is_sam_format) {
            cols_order[SEQ_COL]    = 10;
            cols_order[CHROM_COL]  = 3;
            cols_order[COORD_COL]  = 4;
            cols_order[STRAND_COL] = 2;
            cols_order[MAPQ_COL]   = 5;
            cols_order[CIGAR_COL]  = 6;
            cols_order[PNEXT_COL]  = 8;
            cols_order[TLEN_COL]   = 9;
        } else if (is_frag_format) {
            cols_order[CHROM_COL] = 1;
            cols_order[COORD_COL] = 2;
            cols_order[END_COL]   = 3;
        } else {
            PyObject *seq = PySequence_Fast(py_cols_order,
                "cols_order must be a sequence of 4 ints or None");
            if (!seq)
                verror("cols_order must be a sequence of 4 ints or None");
            if (PySequence_Fast_GET_SIZE(seq) != NUM_USER_COLS) {
                Py_DECREF(seq);
                verror("cols_order must have %d entries", NUM_USER_COLS);
            }
            for (int i = 0; i < NUM_USER_COLS; ++i) {
                PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                long v = PyLong_AsLong(item);
                if (PyErr_Occurred()) {
                    Py_DECREF(seq);
                    verror("cols_order[%d] is not an integer", i);
                }
                cols_order[i] = (int)v;
            }
            Py_DECREF(seq);
            for (int i = 0; i < NUM_USER_COLS; ++i) {
                if (cols_order[i] <= 0)
                    verror("Invalid cols_order: %s index is %d (must be >= 1)",
                           COL_NAMES[i], cols_order[i]);
            }
        }

        if (pileup < 0)
            verror("pileup cannot be negative");
        if (min_mapq < 0)
            verror("min_mapq cannot be negative");
        if (min_mapq > 0 && !is_sam_format)
            verror("min_mapq requires SAM or BAM input (a SAM file needs cols_order=None)");
        if (paired) {
            if (pileup)
                verror("pileup is not used with paired=True: each fragment covers its own span. Set pileup to 0.");
            if (binsize <= 0)
                verror("For paired=True (dense), binsize must be > 0");
            if (max_fraglen <= 0)
                verror("max_fraglen must be positive");
        } else {
            if (pileup == 0 && binsize >= 0)
                verror("For pileup=0 (sparse), binsize must be -1");
            if (pileup > 0 && binsize <= 0)
                verror("For pileup>0 (dense), binsize must be > 0");
        }

        int num_used_cols = 0;
        for (int i = 0; i < NUM_COLS; ++i) {
            if (!cols_order[i])
                continue;
            ++num_used_cols;
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

        // Chrom name -> index map; exact names, as R matches them
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
        std::vector<uint64_t> num_dups(num_chroms, 0);
        std::vector<std::vector<int64_t>> coords(2 * num_chroms);
        std::vector<std::vector<std::pair<int64_t, int64_t>>> frags(paired ? num_chroms : 0);
        // lines starting with this char are headers / comments
        const int comment_char = is_sam_format ? '@' : is_frag_format ? '#' : 0;

        auto src = open_source(file_path);
        int64_t total_unmapped = 0;
        int64_t total_filtered = 0;

        // FSM state (mirrors R's gtrackimport_mappedseq).
        int col = 1;
        int active_col_idx = -1;
        int pos = 0;
        std::string str[NUM_COLS];

        for (int i = 0; i < NUM_COLS; ++i) {
            if (cols_order[i] == 1) { active_col_idx = i; break; }
        }

        while (true) {
            int c = src->getc();

            // CRLF line endings: drop the CR
            if (c == '\r')
                continue;

            // skip SAM headers (@) and fragment file comments (#)
            if (!pos && comment_char && c == comment_char) {
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
                    int num_nonempty_strs = 0;
                    bool mapped = false;
                    bool filtered = false; // a valid record left out by a filter (flags, MAPQ, fragment length)
                    bool second_mate = false; // a paired SAM import counts each pair once, by its first mate

                    pos = 0;
                    for (int i = 0; i < NUM_COLS; ++i)
                        if (!str[i].empty()) ++num_nonempty_strs;

                    // one pass, left early with break
                    while (num_nonempty_strs == num_used_cols) {
                        char *endptr;
                        uint64_t flag = 0;
                        int64_t span = -1; // reference span of a SAM read, from its CIGAR

                        if (is_sam_format) {
                            flag = strtoull(str[STRAND_COL].c_str(), &endptr, 0);
                            if (*endptr)
                                break;
                            if (paired && (flag & 0x1) && (flag & 0x80)) {
                                second_mate = true;
                                break;
                            }
                        }

                        auto istr2chrom = str2chrom.find(str[CHROM_COL]);
                        if (istr2chrom == str2chrom.end())
                            break;
                        int chrom_idx = istr2chrom->second;
                        int64_t chrom_end = chrom_ends[chrom_idx];

                        int64_t coord = strtoll(str[COORD_COL].c_str(), &endptr, 10);
                        if (*endptr || coord < (one_based ? 1 : 0)) // SAM POS 0 means no position
                            break;
                        if (one_based)
                            --coord; // 0-based leftmost base
                        if (coord >= chrom_end)
                            break;

                        if (is_frag_format) {
                            int64_t end = strtoll(str[END_COL].c_str(), &endptr, 10);
                            if (*endptr || end <= coord)
                                break;
                            if (end - coord > max_fraglen) {
                                filtered = true;
                                break;
                            }
                            frags[chrom_idx].emplace_back(coord, std::min(end, chrom_end));
                            mapped = true;
                            ++num_mapped[chrom_idx];
                            break;
                        }

                        if (is_sam_format) {
                            if (flag & 0x4)
                                break;

                            if (flag & SAM_SKIP_FLAGS) {
                                filtered = true;
                                break;
                            }

                            if (min_mapq) {
                                long mapq = strtol(str[MAPQ_COL].c_str(), &endptr, 10);
                                if (*endptr)
                                    break;
                                if (mapq < min_mapq) {
                                    filtered = true;
                                    break;
                                }
                            }

                            // htslib (and so a BAM import) takes a mapped record without a CIGAR ('*') as unmapped;
                            // a malformed CIGAR, or one with no aligned reference base, is unusable too
                            if ((span = cigar_ref_span(str[CIGAR_COL])) <= 0)
                                break;

                            if (paired) {
                                // One fragment per proper pair, taken from the first mate (as MACS3 BAMPE does):
                                // [min(POS, PNEXT), + |TLEN|) in 1-based POS terms. Unpaired reads are filtered.
                                if ((flag & 0x3) != 0x3 || (flag & 0x8)) {
                                    filtered = true;
                                    break;
                                }
                                int64_t pnext = strtoll(str[PNEXT_COL].c_str(), &endptr, 10);
                                if (*endptr)
                                    break;
                                int64_t tlen = strtoll(str[TLEN_COL].c_str(), &endptr, 10);
                                if (*endptr)
                                    break;
                                tlen = llabs(tlen);
                                int64_t start = std::min(coord, pnext - 1);
                                if (!tlen || tlen > max_fraglen || start < 0) {
                                    filtered = true;
                                    break;
                                }
                                frags[chrom_idx].emplace_back(start, std::min(start + tlen, chrom_end));
                                mapped = true;
                                ++num_mapped[chrom_idx];
                                break;
                            }

                            str[STRAND_COL] = flag & 0x10 ? "-" : "+";
                        }

                        if (str[STRAND_COL] == "+" || str[STRAND_COL] == "F")
                            coords[chrom_idx].push_back(coord);
                        else if (str[STRAND_COL] == "-" || str[STRAND_COL] == "R") {
                            // the 5' end of a reverse read is its rightmost aligned base (one past it in legacy mode)
                            if (span <= 0) // a tab-delimited file: no CIGAR
                                span = str[SEQ_COL].size();
                            // the recorded point must lie on the chromosome; a legacy dense track clips the read instead
                            int64_t room = chrom_end - coord;
                            if (legacy ? !pileup && span >= room : span > room)
                                break;
                            coords[num_chroms + chrom_idx].push_back(coord + span - rev_end_off);
                        } else
                            break;

                        mapped = true;
                        ++num_mapped[chrom_idx];
                        break;
                    }

                    if (filtered)
                        ++total_filtered;
                    else if (!mapped && !second_mate && num_nonempty_strs)
                        ++total_unmapped;

                    if (c == EOF) break;

                    if (num_nonempty_strs > 0) {
                        for (int i = 0; i < NUM_COLS; ++i) str[i].clear();
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
                str[active_col_idx].push_back((char)c);
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

        // Write track files: dense with pileup or paired, sparse otherwise
        ensure_track_dir_pm(track_dir);
        const bool dense = pileup > 0 || paired;

        const bool indexed_db = db_is_indexed_pm();
        std::unique_ptr<IndexedTrackWriter> indexed_writer;
        if (indexed_db) {
            MishaTrackType type = dense ? MishaTrackType::DENSE : MishaTrackType::SPARSE;
            indexed_writer.reset(new IndexedTrackWriter(track_dir, type, num_chroms));
        }

        if (dense) {
            const uint32_t bs_u32 = (uint32_t)binsize;
            const double binsize_d = (double)binsize;  // R computes the bins in double
            for (uint32_t chrom = 0; chrom < num_chroms; ++chrom) {
                int64_t chrom_end = chrom_ends[chrom];
                int64_t nbins = (int64_t)std::ceil((double)chrom_end / binsize_d);
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

                // adds the fraction of each bin covered by [from_coord, to_coord); float += double, as in R
                auto add_coverage = [&](int64_t from_coord, int64_t to_coord) {
                    // a reverse read longer than pileup can end past the chromosome: nothing left to add
                    if (from_coord >= to_coord)
                        return;
                    int64_t from_bin = (int64_t)(from_coord / binsize_d);
                    int64_t to_bin = (int64_t)std::ceil(to_coord / binsize_d) - 1;

                    // the last, partial bin of a chromosome is also divided by binsize, as in R
                    if (from_bin >= to_bin)
                        trackvals[from_bin] += (to_coord - from_coord) / binsize_d;
                    else {
                        trackvals[from_bin] += from_bin + 1 - from_coord / binsize_d;
                        trackvals[to_bin] += to_coord / binsize_d - to_bin;
                        for (int64_t bin = from_bin + 1; bin < to_bin; ++bin)
                            trackvals[bin]++;
                    }
                };

                if (paired) {
                    std::vector<std::pair<int64_t, int64_t>> &cur_frags = frags[chrom];
                    std::sort(cur_frags.begin(), cur_frags.end());

                    for (size_t k = 0; k < cur_frags.size(); ++k) {
                        if (remove_dups && !is_frag_format && k > 0 && cur_frags[k] == cur_frags[k - 1]) {
                            ++num_dups[chrom];
                            continue;
                        }
                        add_coverage(cur_frags[k].first, cur_frags[k].second);
                    }
                    std::vector<std::pair<int64_t, int64_t>>().swap(cur_frags);
                }

                for (int strand = 0; strand < 2; ++strand) {
                    std::vector<int64_t> &cur = coords[strand * num_chroms + chrom];
                    std::sort(cur.begin(), cur.end());

                    for (size_t k = 0; k < cur.size(); ++k) {
                        if (remove_dups && k > 0 && cur[k] == cur[k - 1]) {
                            ++num_dups[chrom];
                            continue;
                        }
                        int64_t end_coord = strand ? cur[k] + rev_end_off : cur[k] + pileup;
                        add_coverage(std::max(end_coord - (int64_t)pileup, (int64_t)0), std::min(end_coord, chrom_end));
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
        } else {  // sparse
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
        // PyDict_SetItemString does NOT steal a reference, so each freshly
        // created float must be DECREF'd after insertion to avoid leaking it.
        auto set_float = [](PyObject *d, const char *key, double v) {
            PyObject *f = PyFloat_FromDouble(v);
            PyDict_SetItemString(d, key, f);
            Py_DECREF(f);
        };
        // tot_mapped includes the duplicates
        set_float(total, "total",
                  (double)(tot_mapped + (uint64_t)total_unmapped + (uint64_t)total_filtered));
        set_float(total, "total.mapped",   (double)tot_mapped);
        set_float(total, "total.unmapped", (double)total_unmapped);
        set_float(total, "total.dups",     (double)tot_dups);
        set_float(total, "total.filtered", (double)total_filtered);

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
