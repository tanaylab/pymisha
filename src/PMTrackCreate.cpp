/*
 * PMTrackCreate.cpp
 *
 * Track creation backends for pymisha (dense/sparse 1D).
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <climits>
#include <sys/stat.h>
#include <unistd.h>

#include <Python.h>

#include "pymisha.h"
#include "PMDataFrame.h"
#include "PMDb.h"
#include "CRC64.h"
#include "GenomeTrack.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeTrackSparse.h"
#include "PMTrackExpressionScanner.h"
#include "PMTrackExpressionIterator.h"
#include "TrackIndex.h"

using namespace std;

// Thread-local override for the directory pm_track_create_* should mkdir into.
// When non-empty, track_name_to_dir() returns this exact path instead of the
// path computed from the track name. Set/cleared by pm_set_create_dir_override
// and pm_clear_create_dir_override. Mirrors R misha .create_dir_override
// (R 5.6.30 81635130) and enables atomic gtrack.create via tmp dir + rename.
namespace pymisha_track_create {
    thread_local std::string g_create_dir_override;
}

namespace {

struct TrackRec {
    int chromid;
    int64_t start;
    int64_t end;
    float value;
};

static bool file_exists(const string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool db_is_indexed()
{
    if (!g_pmdb || !g_pmdb->is_initialized())
        return false;
    const string base = g_pmdb->groot() + "/seq/";
    return file_exists(base + "genome.idx") && file_exists(base + "genome.seq");
}

static string track_name_to_dir(const string &track_name)
{
    if (!pymisha_track_create::g_create_dir_override.empty())
        return pymisha_track_create::g_create_dir_override;

    if (!g_pmdb || !g_pmdb->is_initialized())
        TGLError("Database not initialized. Call gdb_init() first.");

    const string &root = g_pmdb->uroot().empty() ? g_pmdb->groot() : g_pmdb->uroot();
    string rel = track_name;
    replace(rel.begin(), rel.end(), '.', '/');
    return root + "/tracks/" + rel + ".track";
}

static void ensure_track_dir(const string &track_dir)
{
    namespace fs = std::filesystem;
    fs::path p(track_dir);
    fs::create_directories(p.parent_path());
    fs::create_directories(p);
}

static int col_idx(PMDataFrame &df, const char *name)
{
    for (size_t i = 0; i < df.num_cols(); ++i) {
        if (!strcmp(df.col_name(i), name))
            return (int)i;
    }
    TGLError("Input data frame is missing '%s' column", name);
    return -1;
}

// Write a 4-byte sparse-track signature to `path`. Used for empty per-chrom
// files in non-indexed DBs. Uses low-level posix syscalls (open/write/close)
// so we can dispatch many in parallel via std::thread without the libc
// stdio FILE* mutex contention that fopen() introduces.
static void write_empty_sparse_file_syscall(const string &path, int32_t sig)
{
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0664);
    if (fd < 0) {
        // Save errno across close() in case caller wants it. Throw outside
        // worker threads via a flag (callers check it post-join).
        return;  // failure surfaced via post-join check on file_exists/size
    }
    ssize_t n = ::write(fd, &sig, sizeof(sig));
    (void)n;
    ::close(fd);
}

// Create the empty per-chrom files for chroms not yet written. Dispatches
// the open/write/close syscalls across a small thread pool so the NFS
// server can pipeline create requests. On hg38 (455 chroms, 1 chrom with
// data) this drops ~1.0 s of wall time to ~0.5 s. Mirrors the sequential
// loop semantics: each chrom that was not touched by the main write loop
// gets a 4-byte signature file at its canonical 1d filename.
static void create_empty_sparse_files_parallel(const string &track_dir,
                                               const GenomeChromKey &chromkey,
                                               uint32_t num_chroms,
                                               const vector<bool> &created)
{
    // Collect target paths upfront so the workers do no chromkey lookup.
    const int32_t sig = GenomeTrack::FORMAT_SIGNATURES[GenomeTrack::SPARSE];
    vector<string> paths;
    paths.reserve(num_chroms);
    for (uint32_t cid = 0; cid < num_chroms; ++cid) {
        if (created[cid]) continue;
        paths.emplace_back(track_dir + "/" +
                           GenomeTrack::get_1d_filename(chromkey, (int)cid));
    }
    if (paths.empty()) return;

    // 4 workers saturates NFSv3 CREATE pipelining empirically; more threads
    // don't help and just add startup overhead.
    unsigned nworkers = std::min<unsigned>(4u, (unsigned)paths.size());
    if (nworkers <= 1) {
        for (const string &p : paths)
            write_empty_sparse_file_syscall(p, sig);
        return;
    }

    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= paths.size()) return;
            write_empty_sparse_file_syscall(paths[i], sig);
        }
    };

    vector<std::thread> threads;
    threads.reserve(nworkers - 1);
    for (unsigned t = 0; t < nworkers - 1; ++t)
        threads.emplace_back(worker);
    worker();  // current thread does its share
    for (auto &th : threads) th.join();
}

// Indexed 1D-track direct writer.
//
// Produces track.dat + track.idx with the SAME byte layout that the
// per-chrom + pm_track_convert_to_indexed pipeline would produce. The
// .dat contents are a chromid-ordered concatenation of the per-chrom
// file bytes (signature + payload). Chroms with no payload contribute
// 0 bytes and get length=0 in the index. The .idx file is the 36-byte
// header (TrackIdxHeader from PMTrackIndexedFormat.cpp) followed by
// num_chroms 24-byte contig entries; CRC64-ECMA is over (chrom_id,
// offset, length) per entry in chromid order. See PMTrackIndexedFormat.cpp.
struct IndexedTrackWriter {
    static constexpr size_t HEADER_BYTES = 36;          // matches TrackIdxHeader
    static constexpr size_t HEADER_TO_CHECKSUM = 28;    // bytes before checksum field
    static constexpr size_t ENTRY_BYTES = 24;

    string dat_path;
    string idx_path;
    string dat_tmp;
    string idx_tmp;
    FILE *dat_fp;
    FILE *idx_fp;
    vector<TrackContigEntry> entries;
    uint64_t current_offset;
    MishaTrackType track_type;
    uint32_t num_chroms;
    bool finished;

    IndexedTrackWriter(const string &track_dir, MishaTrackType type, uint32_t nchroms)
        : dat_path(track_dir + "/track.dat"),
          idx_path(track_dir + "/track.idx"),
          dat_tmp(track_dir + "/track.dat.tmp"),
          idx_tmp(track_dir + "/track.idx.tmp"),
          dat_fp(nullptr), idx_fp(nullptr),
          current_offset(0), track_type(type), num_chroms(nchroms),
          finished(false)
    {
        dat_fp = fopen(dat_tmp.c_str(), "wb");
        if (!dat_fp)
            TGLError<GenomeTrack>("Failed to create %s: %s", dat_tmp.c_str(), strerror(errno));
        idx_fp = fopen(idx_tmp.c_str(), "wb");
        if (!idx_fp) {
            fclose(dat_fp); dat_fp = nullptr;
            unlink(dat_tmp.c_str());
            TGLError<GenomeTrack>("Failed to create %s: %s", idx_tmp.c_str(), strerror(errno));
        }
        // Bump stdio buffer to 1 MiB so per-record fwrite()s coalesce into
        // few syscalls. Critical on networked filesystems where the default
        // ~4-8 KiB buffer means thousands of write() syscalls for a 1M-row
        // sparse track (-1.0 s wall on NFS). Best-effort: if setvbuf fails,
        // libc keeps using its default buffer.
        setvbuf(dat_fp, nullptr, _IOFBF, 1 << 20);
        setvbuf(idx_fp, nullptr, _IOFBF, 1 << 20);
        entries.reserve(num_chroms);
        write_initial_header();
    }

    ~IndexedTrackWriter()
    {
        // Cleanup on early destruction (exception path).
        if (!finished) {
            if (dat_fp) fclose(dat_fp);
            if (idx_fp) fclose(idx_fp);
            unlink(dat_tmp.c_str());
            unlink(idx_tmp.c_str());
        }
    }

    void write_initial_header()
    {
        // Matches TrackIdxHeader (PMTrackIndexedFormat.cpp): packed 36 bytes,
        // checksum zero for now and rewritten after entries are computed.
#pragma pack(push, 1)
        struct DiskHeader {
            char magic[8];
            uint32_t version;
            uint32_t track_type_raw;
            uint32_t num_contigs;
            uint64_t flags;
            uint64_t checksum;
        };
#pragma pack(pop)
        DiskHeader hdr;
        const char magic[8] = {'M','I','S','H','A','T','D','X'};
        memcpy(hdr.magic, magic, 8);
        hdr.version = 1;
        hdr.track_type_raw = static_cast<uint32_t>(track_type);
        hdr.num_contigs = num_chroms;
        hdr.flags = 0x01; // IS_LITTLE_ENDIAN
        hdr.checksum = 0;

        if (fwrite(&hdr, sizeof(hdr), 1, idx_fp) != 1)
            TGLError<GenomeTrack>("Failed to write index header to %s", idx_tmp.c_str());
    }

    // Append a chromosome chunk to track.dat. payload is the exact bytes
    // a per-chrom file would have contained (signature + records or
    // signature + binsize + bins). Pass length=0 to record an empty
    // chrom entry without writing any bytes.
    void append_chrom(int chromid, const void *payload, uint64_t length)
    {
        TrackContigEntry entry;
        entry.chrom_id = (uint32_t)chromid;
        entry.offset = current_offset;
        entry.length = length;
        entry.reserved = 0;

        if (length > 0) {
            if (fwrite(payload, 1, length, dat_fp) != length)
                TGLError<GenomeTrack>("Failed to write chrom payload to %s: %s",
                                      dat_tmp.c_str(), strerror(errno));
            current_offset += length;
        }
        entries.push_back(entry);

#pragma pack(push, 1)
        struct DiskContigEntry {
            uint32_t chrom_id;
            uint64_t offset;
            uint64_t length;
            uint32_t reserved;
        };
#pragma pack(pop)
        DiskContigEntry de;
        de.chrom_id = entry.chrom_id;
        de.offset = entry.offset;
        de.length = entry.length;
        de.reserved = entry.reserved;
        if (fwrite(&de, sizeof(de), 1, idx_fp) != 1)
            TGLError<GenomeTrack>("Failed to write index entry to %s", idx_tmp.c_str());
    }

    // Streaming chunk write (no temporary buffer). Used by dense path
    // which streams floats into track.dat directly. Caller must call
    // begin_chrom() and end_chrom() around the chunk writes.
    void begin_chrom(int chromid)
    {
        TrackContigEntry entry;
        entry.chrom_id = (uint32_t)chromid;
        entry.offset = current_offset;
        entry.length = 0;
        entry.reserved = 0;
        entries.push_back(entry);
    }

    void stream_bytes(const void *data, size_t n)
    {
        if (n == 0) return;
        if (fwrite(data, 1, n, dat_fp) != n)
            TGLError<GenomeTrack>("Failed to write chrom payload to %s: %s",
                                  dat_tmp.c_str(), strerror(errno));
        current_offset += n;
    }

    void end_chrom()
    {
        TrackContigEntry &entry = entries.back();
        entry.length = current_offset - entry.offset;

#pragma pack(push, 1)
        struct DiskContigEntry {
            uint32_t chrom_id;
            uint64_t offset;
            uint64_t length;
            uint32_t reserved;
        };
#pragma pack(pop)
        DiskContigEntry de;
        de.chrom_id = entry.chrom_id;
        de.offset = entry.offset;
        de.length = entry.length;
        de.reserved = entry.reserved;
        if (fwrite(&de, sizeof(de), 1, idx_fp) != 1)
            TGLError<GenomeTrack>("Failed to write index entry to %s", idx_tmp.c_str());
    }

    void finish()
    {
        if ((uint32_t)entries.size() != num_chroms)
            TGLError<GenomeTrack>("IndexedTrackWriter: wrote %u entries, expected %u",
                                  (unsigned)entries.size(), (unsigned)num_chroms);

        misha::CRC64 crc64;
        uint64_t checksum = crc64.init_incremental();
        for (const auto &entry : entries) {
            checksum = crc64.compute_incremental(checksum,
                (const unsigned char*)&entry.chrom_id, sizeof(entry.chrom_id));
            checksum = crc64.compute_incremental(checksum,
                (const unsigned char*)&entry.offset, sizeof(entry.offset));
            checksum = crc64.compute_incremental(checksum,
                (const unsigned char*)&entry.length, sizeof(entry.length));
        }
        checksum = crc64.finalize_incremental(checksum);

        if (fseek(idx_fp, (long)HEADER_TO_CHECKSUM, SEEK_SET) != 0)
            TGLError<GenomeTrack>("Failed to seek to checksum in %s", idx_tmp.c_str());
        if (fwrite(&checksum, sizeof(checksum), 1, idx_fp) != 1)
            TGLError<GenomeTrack>("Failed to write checksum to %s", idx_tmp.c_str());

        fflush(dat_fp);
        fflush(idx_fp);
        fsync(fileno(dat_fp));
        fsync(fileno(idx_fp));
        fclose(dat_fp); dat_fp = nullptr;
        fclose(idx_fp); idx_fp = nullptr;

        if (rename(dat_tmp.c_str(), dat_path.c_str()) != 0)
            TGLError<GenomeTrack>("Failed to rename %s -> %s: %s",
                                  dat_tmp.c_str(), dat_path.c_str(), strerror(errno));
        if (rename(idx_tmp.c_str(), idx_path.c_str()) != 0)
            TGLError<GenomeTrack>("Failed to rename %s -> %s: %s",
                                  idx_tmp.c_str(), idx_path.c_str(), strerror(errno));

        finished = true;
    }
};

static void parse_track_data(PyObject *py_df, vector<TrackRec> &out)
{
    PMPY pm_df(py_df, false);
    PMDataFrame df(pm_df, "track_data");

    int ichrom = col_idx(df, "chrom");
    int istart = col_idx(df, "start");
    int iend = col_idx(df, "end");
    int ivalue = col_idx(df, "value");

    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    out.clear();
    out.reserve(df.num_rows());

    for (size_t i = 0; i < df.num_rows(); ++i) {
        const char *chrom = df.val_str(i, ichrom);
        if (!chrom)
            continue;

        int chromid = chromkey.chrom2id(chrom);
        if (chromid < 0)
            continue;

        int64_t start = (int64_t)df.val_long(i, istart);
        int64_t end = (int64_t)df.val_long(i, iend);
        double value = df.val_double(i, ivalue);

        if (start < 0 || end <= start)
            continue;

        TrackRec rec;
        rec.chromid = chromid;
        rec.start = start;
        rec.end = end;
        rec.value = (float)value;
        out.push_back(rec);
    }

    sort(out.begin(), out.end(), [](const TrackRec &a, const TrackRec &b) {
        if (a.chromid != b.chromid) return a.chromid < b.chromid;
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });
}

static long parse_iterator_policy_local(PyObject *py_iterator, long default_policy, const char *context)
{
    if (!py_iterator || py_iterator == Py_None) {
        return default_policy;
    }

    if (PyLong_Check(py_iterator)) {
        long val = PyLong_AsLong(py_iterator);
        if (PyErr_Occurred()) {
            verror("%s: iterator value out of range", context);
        }
        return val;
    }

    if (PyFloat_Check(py_iterator)) {
        double val = PyFloat_AsDouble(py_iterator);
        if (!std::isfinite(val)) {
            verror("%s: iterator must be a finite number", context);
        }
        if (val > (double)LONG_MAX || val < (double)LONG_MIN) {
            verror("%s: iterator value out of range", context);
        }
        return (long)val;
    }

    if (PyNumber_Check(py_iterator)) {
        PMPY as_long(PyNumber_Long(py_iterator), true);
        if (!as_long) {
            verror("%s: iterator must be an integer or float", context);
        }
        long val = PyLong_AsLong((PyObject *)as_long);
        if (PyErr_Occurred()) {
            verror("%s: iterator value out of range", context);
        }
        return val;
    }

    verror("%s: iterator must be an integer or float", context);
    return default_policy;
}

static PyObject *get_progress_cb_local(PyObject *py_config)
{
    if (!py_config || py_config == Py_None || !PyDict_Check(py_config)) {
        return NULL;
    }

    PyObject *cb = PyDict_GetItemString(py_config, "_progress_cb");
    if (cb && cb != Py_None && !PyCallable_Check(cb)) {
        verror("progress callback must be callable");
    }
    return (cb && cb != Py_None) ? cb : NULL;
}

static void parse_intervals_from_df(PyObject *py_intervals, vector<GInterval> &intervals)
{
    PMPY pm_df(py_intervals, false);
    PMDataFrame df(pm_df, "intervals");

    int ichrom = -1, istart = -1, iend = -1;
    for (size_t i = 0; i < df.num_cols(); ++i) {
        if (!strcmp(df.col_name(i), "chrom")) ichrom = (int)i;
        else if (!strcmp(df.col_name(i), "start")) istart = (int)i;
        else if (!strcmp(df.col_name(i), "end")) iend = (int)i;
    }
    if (ichrom < 0 || istart < 0 || iend < 0)
        TGLError("intervals must contain 'chrom', 'start', and 'end' columns");

    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    intervals.clear();
    intervals.reserve(df.num_rows());

    for (size_t i = 0; i < df.num_rows(); ++i) {
        const char *chrom = df.val_str(i, ichrom);
        int64_t start = (int64_t)df.val_long(i, istart);
        int64_t end = (int64_t)df.val_long(i, iend);
        if (!chrom)
            continue;
        int chromid = chromkey.chrom2id(chrom);
        if (chromid < 0)
            TGLError("Unknown chromosome: %s", chrom);
        if (start < 0 || end <= start)
            continue;
        intervals.emplace_back(chromid, start, end);
    }
}

} // namespace

PyObject *pm_track_create_sparse(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        PyObject *py_data = nullptr;
        if (!PyArg_ParseTuple(args, "sO", &track, &py_data))
            verror("Invalid arguments to pm_track_create_sparse");

        string track_name(track);
        if (g_pmdb->track_exists(track_name))
            verror("Track '%s' already exists", track);

        string track_dir = track_name_to_dir(track_name);
        if (file_exists(track_dir))
            verror("Track directory already exists: %s", track_dir.c_str());
        ensure_track_dir(track_dir);

        vector<TrackRec> recs;
        parse_track_data(py_data, recs);

        for (size_t i = 1; i < recs.size(); ++i) {
            if (recs[i - 1].chromid == recs[i].chromid && recs[i].start < recs[i - 1].end) {
                verror("Sparse intervals must be non-overlapping within each chromosome");
            }
        }

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        const bool indexed_db = db_is_indexed();
        const uint32_t num_chroms = (uint32_t)chromkey.get_num_chroms();

        if (indexed_db) {
            // Direct write of track.dat + track.idx, bypassing per-chrom
            // files. Byte-identical to the per-chrom + convert pipeline:
            // for each chrom with data we emit [sparse signature (4 bytes,
            // little-endian int -8)] + N x [start(8) end(8) value(4)] in
            // chromid order; empty chroms contribute zero bytes. Mirrors
            // R misha 5.6.30 94a6446d.
            IndexedTrackWriter writer(track_dir, MishaTrackType::SPARSE, num_chroms);

            const int32_t sig = GenomeTrack::FORMAT_SIGNATURES[GenomeTrack::SPARSE];
            size_t cursor = 0;
            for (int chromid = 0; chromid < (int)num_chroms; ++chromid) {
                size_t begin = cursor;
                while (begin < recs.size() && recs[begin].chromid < chromid)
                    ++begin;
                size_t end = begin;
                while (end < recs.size() && recs[end].chromid == chromid)
                    ++end;
                cursor = end;

                if (begin == end) {
                    // Empty chrom: no per-chrom file would be created in
                    // indexed mode, so contribute zero bytes here.
                    writer.append_chrom(chromid, nullptr, 0);
                    continue;
                }

                writer.begin_chrom(chromid);
                writer.stream_bytes(&sig, sizeof(sig));
                for (size_t i = begin; i < end; ++i) {
                    const TrackRec &r = recs[i];
                    int64_t s = r.start;
                    int64_t e = r.end;
                    float v = r.value;
                    writer.stream_bytes(&s, sizeof(s));
                    writer.stream_bytes(&e, sizeof(e));
                    writer.stream_bytes(&v, sizeof(v));
                }
                writer.end_chrom();
            }
            writer.finish();
        } else {
            vector<bool> created(num_chroms, false);
            GenomeTrackSparse gtrack;
            int cur_chromid = -1;
            for (const auto &r : recs) {
                if (cur_chromid != r.chromid) {
                    cur_chromid = r.chromid;
                    string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, cur_chromid);
                    gtrack.init_write(path.c_str(), cur_chromid);
                    created[cur_chromid] = true;
                }
                GInterval interv(cur_chromid, r.start, r.end);
                gtrack.write_next_interval(interv, r.value);
            }
            // Empty per-chrom files: each is a 4-byte signature. For
            // genomes with many small alt contigs the create+close
            // roundtrip per file dominates on NFS (~2 ms / file -> -1.0 s
            // on hg38). Issue the open/write/close syscalls from a small
            // thread pool so the NFS server can pipeline create requests.
            // Empirically saturates at ~4 threads on NFSv3 (~2x speedup).
            create_empty_sparse_files_parallel(track_dir, chromkey, num_chroms, created);
        }

        return_none();
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}

// Per-bin aggregation knob added in R misha 5.6.32 (068a02a2, 5e69c2c8).
// WMEAN keeps the historical streaming-sum fast path; the others collect
// contributions into a small per-bin vector and reduce.
enum class DenseAggFunc {
    WMEAN,
    WSUM,
    MAX,
    MIN,
    WMEDIAN,
    COUNT,
    COVERAGE,
};

static DenseAggFunc parse_dense_agg_func(const char *s)
{
    if (!strcmp(s, "weighted.mean"))   return DenseAggFunc::WMEAN;
    if (!strcmp(s, "weighted.sum"))    return DenseAggFunc::WSUM;
    if (!strcmp(s, "max"))             return DenseAggFunc::MAX;
    if (!strcmp(s, "min"))             return DenseAggFunc::MIN;
    if (!strcmp(s, "weighted.median")) return DenseAggFunc::WMEDIAN;
    if (!strcmp(s, "count"))           return DenseAggFunc::COUNT;
    if (!strcmp(s, "coverage"))        return DenseAggFunc::COVERAGE;
    verror("Unknown func '%s' for gtrack_create_dense", s);
    return DenseAggFunc::WMEAN;  // unreachable
}

namespace {
struct Contribution {
    double value;
    double overlap;  // base-pair overlap with the bin
};
}  // namespace

static float reduce_bin(DenseAggFunc func, vector<Contribution> &contribs, double bin_width)
{
    if (contribs.empty()) {
        return (func == DenseAggFunc::COUNT)
                   ? 0.0f
                   : std::numeric_limits<float>::quiet_NaN();
    }

    switch (func) {
    case DenseAggFunc::WMEAN: {
        double sum = 0.0, w = 0.0;
        for (const auto &c : contribs) {
            sum += c.value * c.overlap;
            w   += c.overlap;
        }
        return w > 0.0 ? (float)(sum / w)
                       : std::numeric_limits<float>::quiet_NaN();
    }
    case DenseAggFunc::WSUM: {
        double sum = 0.0;
        for (const auto &c : contribs)
            sum += c.value * c.overlap;
        return (float)sum;
    }
    case DenseAggFunc::MAX: {
        double m = contribs[0].value;
        for (size_t i = 1; i < contribs.size(); ++i)
            if (contribs[i].value > m) m = contribs[i].value;
        return (float)m;
    }
    case DenseAggFunc::MIN: {
        double m = contribs[0].value;
        for (size_t i = 1; i < contribs.size(); ++i)
            if (contribs[i].value < m) m = contribs[i].value;
        return (float)m;
    }
    case DenseAggFunc::WMEDIAN: {
        // Lower weighted median: sort by value asc, accumulate overlap;
        // return first value whose running overlap reaches total/2.
        std::sort(contribs.begin(), contribs.end(),
                  [](const Contribution &a, const Contribution &b) {
                      return a.value < b.value;
                  });
        double total = 0.0;
        for (const auto &c : contribs) total += c.overlap;
        if (total <= 0.0)
            return std::numeric_limits<float>::quiet_NaN();
        double half = total / 2.0;
        double acc = 0.0;
        for (const auto &c : contribs) {
            acc += c.overlap;
            if (acc >= half) return (float)c.value;
        }
        return (float)contribs.back().value;  // defensive
    }
    case DenseAggFunc::COUNT:
        return (float)contribs.size();
    case DenseAggFunc::COVERAGE: {
        // sum(v_i * ov_i / bin_width). With v=1 and defval=0 this is the
        // average per-base signal in the bin (ChIP-seq pileup).
        if (bin_width <= 0.0)
            return std::numeric_limits<float>::quiet_NaN();
        double sum = 0.0;
        for (const auto &c : contribs)
            sum += c.value * (c.overlap / bin_width);
        return (float)sum;
    }
    }
    return std::numeric_limits<float>::quiet_NaN();  // unreachable
}

PyObject *pm_track_create_dense(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        PyObject *py_data = nullptr;
        unsigned binsize = 0;
        double defval = std::numeric_limits<double>::quiet_NaN();
        const char *func_str = nullptr;

        if (!PyArg_ParseTuple(args, "sOIds", &track, &py_data, &binsize, &defval, &func_str))
            verror("Invalid arguments to pm_track_create_dense");

        if (binsize == 0)
            verror("binsize must be positive");

        const DenseAggFunc func = parse_dense_agg_func(func_str);
        const bool is_wmean = (func == DenseAggFunc::WMEAN);

        string track_name(track);
        if (g_pmdb->track_exists(track_name))
            verror("Track '%s' already exists", track);

        string track_dir = track_name_to_dir(track_name);
        if (file_exists(track_dir))
            verror("Track directory already exists: %s", track_dir.c_str());
        ensure_track_dir(track_dir);

        vector<TrackRec> recs;
        parse_track_data(py_data, recs);

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        const bool indexed_db = db_is_indexed();
        const uint32_t num_chroms = (uint32_t)chromkey.get_num_chroms();
        size_t data_idx = 0;

        // Optional indexed writer (constructed only when indexed_db).
        unique_ptr<IndexedTrackWriter> writer;
        if (indexed_db) {
            // Direct write of track.dat + track.idx. Dense per-chrom files
            // contain [binsize uint32 (4 bytes)] + [chrom_size/binsize x
            // float32] with NO format signature byte (see
            // GenomeTrackFixedBin::init_write). We mirror that byte layout
            // chunk-for-chunk into track.dat. Mirrors R misha 5.6.30 b2ca08cc.
            writer.reset(new IndexedTrackWriter(track_dir, MishaTrackType::DENSE, num_chroms));
        }

        // Per-bin contribution buffer reused across bins (only allocated
        // when we leave the streaming weighted.mean fast path).
        vector<Contribution> contribs;
        if (!is_wmean)
            contribs.reserve(64);

        for (int chromid = 0; chromid < (int)num_chroms; ++chromid) {
            uint64_t chrom_size = chromkey.get_chrom_size(chromid);

            GenomeTrackFixedBin gtrack;
            if (indexed_db) {
                writer->begin_chrom(chromid);
                writer->stream_bytes(&binsize, sizeof(binsize));
            } else {
                string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);
                gtrack.init_write(path.c_str(), binsize, chromid);
            }

            while (data_idx < recs.size() && recs[data_idx].chromid < chromid)
                ++data_idx;
            size_t chrom_begin = data_idx;
            while (data_idx < recs.size() && recs[data_idx].chromid == chromid)
                ++data_idx;
            size_t chrom_end = data_idx;

            size_t cur_idx = chrom_begin;
            vector<float> batch;
            batch.reserve(10000);

            auto flush_batch = [&]() {
                if (batch.empty()) return;
                if (indexed_db) {
                    writer->stream_bytes(batch.data(), batch.size() * sizeof(float));
                } else {
                    gtrack.write_next_bins(batch.data(), batch.size());
                }
                batch.clear();
            };

            for (uint64_t start = 0; start < chrom_size; start += binsize) {
                uint64_t end = std::min(start + (uint64_t)binsize, chrom_size);
                while (cur_idx < chrom_end && (uint64_t)recs[cur_idx].end <= start)
                    ++cur_idx;

                uint64_t width = end - start;
                float out;

                if (is_wmean) {
                    // Historical streaming-sum fast path. Byte-identical
                    // to pre-5.6.32 output.
                    double sum = 0;
                    uint64_t covered = 0;
                    size_t j = cur_idx;
                    while (j < chrom_end && (uint64_t)recs[j].start < end) {
                        uint64_t ov_start = std::max(start, (uint64_t)recs[j].start);
                        uint64_t ov_end = std::min(end, (uint64_t)recs[j].end);
                        if (ov_end > ov_start && !std::isnan((double)recs[j].value)) {
                            uint64_t ov = ov_end - ov_start;
                            sum += (double)recs[j].value * (double)ov;
                            covered += ov;
                        }
                        ++j;
                    }
                    if (covered < width && !std::isnan(defval)) {
                        uint64_t unc = width - covered;
                        sum += defval * (double)unc;
                        covered += unc;
                    }
                    out = covered ? (float)(sum / (double)covered)
                                  : std::numeric_limits<float>::quiet_NaN();
                } else {
                    contribs.clear();
                    uint64_t covered = 0;
                    size_t j = cur_idx;
                    while (j < chrom_end && (uint64_t)recs[j].start < end) {
                        uint64_t ov_start = std::max(start, (uint64_t)recs[j].start);
                        uint64_t ov_end = std::min(end, (uint64_t)recs[j].end);
                        if (ov_end > ov_start && !std::isnan((double)recs[j].value)) {
                            uint64_t ov = ov_end - ov_start;
                            contribs.push_back({(double)recs[j].value, (double)ov});
                            covered += ov;
                        }
                        ++j;
                    }
                    // Synthetic uncovered contribution at defval, except for COUNT
                    // (which reports only real interval count).
                    if (covered < width && !std::isnan(defval) && func != DenseAggFunc::COUNT) {
                        uint64_t unc = width - covered;
                        contribs.push_back({defval, (double)unc});
                    }
                    out = reduce_bin(func, contribs, (double)width);
                }

                batch.push_back(out);
                if (batch.size() >= 10000)
                    flush_batch();
            }

            flush_batch();

            if (indexed_db)
                writer->end_chrom();
        }

        if (indexed_db)
            writer->finish();

        return_none();
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}

PyObject *pm_track_create_expr(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        const char *expr = nullptr;
        PyObject *py_intervals = nullptr;
        PyObject *py_iterator = nullptr;
        PyObject *py_config = nullptr;

        if (!PyArg_ParseTuple(args, "ssO|OO", &track, &expr, &py_intervals, &py_iterator, &py_config))
            verror("Invalid arguments to pm_track_create_expr");

        string track_name(track);
        if (g_pmdb->track_exists(track_name))
            verror("Track '%s' already exists", track);

        long iterator_policy = parse_iterator_policy_local(py_iterator, 0, "pm_track_create_expr");

        vector<GInterval> intervals;
        parse_intervals_from_df(py_intervals, intervals);
        if (intervals.empty())
            verror("intervals are empty");

        string track_dir = track_name_to_dir(track_name);
        if (file_exists(track_dir))
            verror("Track directory already exists: %s", track_dir.c_str());
        ensure_track_dir(track_dir);

        PMTrackExprScanner scanner;
        PyObject *progress_cb = get_progress_cb_local(py_config);
        if (progress_cb && !PyMisha::is_kid()) {
            scanner.set_progress_callback(progress_cb);
            scanner.report_progress(false);
        }

        vector<string> exprs = {string(expr)};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        const bool indexed_db = db_is_indexed();
        vector<bool> created(chromkey.get_num_chroms(), false);

        if (dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator())) {
            PMFixedBinIterator *fitr = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
            int64_t binsize = fitr ? fitr->get_bin_size() : 0;
            if (binsize <= 0)
                verror("Failed to infer dense iterator binsize");

            GenomeTrackFixedBin gtrack;
            int cur_chromid = -1;
            for (; !scanner.isend(); scanner.next()) {
                const GInterval &interv = scanner.last_interval();
                if (interv.chromid != cur_chromid) {
                    cur_chromid = interv.chromid;
                    string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, cur_chromid);
                    gtrack.init_write(path.c_str(), (unsigned)binsize, cur_chromid);
                    created[cur_chromid] = true;
                }
                float v = (float)scanner.vdouble();
                gtrack.write_next_bin(v);
            }

            if (!indexed_db) {
                for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
                    if (created[chromid])
                        continue;
                    string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);
                    gtrack.init_write(path.c_str(), (unsigned)binsize, chromid);
                }
            }
        } else {
            GenomeTrackSparse gtrack;
            int cur_chromid = -1;
            for (; !scanner.isend(); scanner.next()) {
                const GInterval &interv = scanner.last_interval();
                if (interv.chromid != cur_chromid) {
                    cur_chromid = interv.chromid;
                    string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, cur_chromid);
                    gtrack.init_write(path.c_str(), cur_chromid);
                    created[cur_chromid] = true;
                }
                float v = (float)scanner.vdouble();
                gtrack.write_next_interval(interv, v);
            }

            if (!indexed_db) {
                for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
                    if (created[chromid])
                        continue;
                    string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);
                    gtrack.init_write(path.c_str(), chromid);
                }
            }
        }

        return_none();
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}

PyObject *pm_modify(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        const char *expr = nullptr;
        PyObject *py_intervals = nullptr;
        long iterator_policy = 0;

        if (!PyArg_ParseTuple(args, "ssOl", &track, &expr, &py_intervals, &iterator_policy))
            verror("Invalid arguments to pm_modify");

        string track_name(track);
        if (!g_pmdb->track_exists(track_name))
            verror("Track '%s' does not exist", track);

        string track_dir = track_name_to_dir(track_name);

        vector<GInterval> intervals;
        parse_intervals_from_df(py_intervals, intervals);
        if (intervals.empty())
            verror("intervals are empty");

        PMTrackExprScanner scanner;
        vector<string> exprs = {string(expr)};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        PMFixedBinIterator *fitr = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
        if (!fitr)
            verror("gtrack_modify requires a fixed-bin (dense) iterator");

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        GenomeTrackFixedBin gtrack;
        int cur_chromid = -1;

        for (; !scanner.isend(); scanner.next()) {
            const GInterval &interv = scanner.last_interval();

            if (interv.chromid != cur_chromid) {
                cur_chromid = interv.chromid;
                string fname = GenomeTrack::find_existing_1d_filename(chromkey, track_dir, cur_chromid);
                string path = track_dir + "/" + fname;
                gtrack.init_update(path.c_str(), cur_chromid);
            }

            uint64_t bin_idx = interv.start / gtrack.get_bin_size();
            gtrack.goto_bin(bin_idx);
            float v = (float)scanner.vdouble();
            gtrack.write_next_bin(v);
        }

        return_none();
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}

PyObject *pm_smooth(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        const char *expr = nullptr;
        PyObject *py_intervals = nullptr;
        long iterator_policy = 0;
        double winsize = 0;
        double weight_thr = 0;
        int smooth_nans = 0;
        const char *alg = nullptr;

        if (!PyArg_ParseTuple(args, "ssOlddis", &track, &expr, &py_intervals, &iterator_policy, &winsize, &weight_thr, &smooth_nans, &alg))
            verror("Invalid arguments to pm_smooth");

        string track_name(track);
        if (g_pmdb->track_exists(track_name))
            verror("Track '%s' already exists", track);

        string alg_str(alg);
        bool use_linear_ramp = false;
        if (alg_str == "LINEAR_RAMP")
            use_linear_ramp = true;
        else if (alg_str == "MEAN")
            use_linear_ramp = false;
        else
            verror("Invalid smoothing algorithm '%s'. Use 'LINEAR_RAMP' or 'MEAN'.", alg);

        vector<GInterval> intervals;
        parse_intervals_from_df(py_intervals, intervals);
        if (intervals.empty())
            verror("intervals are empty");

        string track_dir = track_name_to_dir(track_name);
        if (file_exists(track_dir))
            verror("Track directory already exists: %s", track_dir.c_str());
        ensure_track_dir(track_dir);

        PMTrackExprScanner scanner;
        vector<string> exprs = {string(expr)};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        PMFixedBinIterator *fitr = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
        if (!fitr)
            verror("gtrack_smooth requires a fixed-bin (dense) iterator");
        int64_t binsize = fitr->get_bin_size();
        if (binsize <= 0)
            verror("Failed to infer dense iterator binsize");

        int num_samples_aside = (int)(0.5 * winsize / binsize + 0.5);
        if (num_samples_aside < 0)
            num_samples_aside = 0;
        int window_size = 2 * num_samples_aside + 1;

        double adjusted_weight_thr;
        if (use_linear_ramp)
            adjusted_weight_thr = weight_thr * (num_samples_aside + 1);
        else
            adjusted_weight_thr = weight_thr;

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        const bool indexed_db = db_is_indexed();
        vector<bool> created(chromkey.get_num_chroms(), false);

        GenomeTrackFixedBin out_track;
        int cur_chromid = -1;

        vector<float> buf(window_size, numeric_limits<float>::quiet_NaN());
        int buf_pos = 0;
        int64_t samples_fed = 0;

        auto compute_smoothed = [&]() -> float {
            int center_idx = (buf_pos + window_size - 1 - num_samples_aside) % window_size;
            float center_val = buf[center_idx];

            if (!smooth_nans && std::isnan(center_val))
                return numeric_limits<float>::quiet_NaN();

            double total_val = 0;
            double total_weight = 0;

            for (int offset = -num_samples_aside; offset <= num_samples_aside; offset++) {
                int idx = (center_idx + offset + window_size) % window_size;
                float v = buf[idx];
                if (!std::isnan(v)) {
                    double w;
                    if (use_linear_ramp)
                        w = (double)(num_samples_aside + 1 - abs(offset));
                    else
                        w = 1.0;
                    total_val += v * w;
                    total_weight += w;
                }
            }

            if (total_weight >= adjusted_weight_thr && total_weight > 0)
                return (float)(total_val / total_weight);
            else
                return numeric_limits<float>::quiet_NaN();
        };

        auto flush_remaining = [&](int count) {
            for (int i = 0; i < count; i++) {
                buf[buf_pos] = numeric_limits<float>::quiet_NaN();
                buf_pos = (buf_pos + 1) % window_size;
                samples_fed++;
                if (samples_fed > num_samples_aside)
                    out_track.write_next_bin(compute_smoothed());
            }
        };

        for (; !scanner.isend(); scanner.next()) {
            const GInterval &interv = scanner.last_interval();

            if (interv.chromid != cur_chromid) {
                if (cur_chromid >= 0 && num_samples_aside > 0)
                    flush_remaining(num_samples_aside);

                cur_chromid = interv.chromid;
                string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, cur_chromid);
                out_track.init_write(path.c_str(), (unsigned)binsize, cur_chromid);
                created[cur_chromid] = true;

                fill(buf.begin(), buf.end(), numeric_limits<float>::quiet_NaN());
                buf_pos = 0;
                samples_fed = 0;
            }

            float v = (float)scanner.vdouble();
            buf[buf_pos] = v;
            buf_pos = (buf_pos + 1) % window_size;
            samples_fed++;

            if (samples_fed <= num_samples_aside)
                continue;

            out_track.write_next_bin(compute_smoothed());
        }

        if (cur_chromid >= 0 && num_samples_aside > 0)
            flush_remaining(num_samples_aside);

        if (!indexed_db) {
            for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
                if (created[chromid])
                    continue;
                string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);
                out_track.init_write(path.c_str(), (unsigned)binsize, chromid);
            }
        }

        return_none();
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return_err();
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return_err();
    }
}

PyObject *pm_set_create_dir_override(PyObject *self, PyObject *args)
{
    (void)self;
    const char *path = nullptr;
    if (!PyArg_ParseTuple(args, "s", &path))
        return nullptr;
    pymisha_track_create::g_create_dir_override.assign(path);
    Py_RETURN_NONE;
}

PyObject *pm_clear_create_dir_override(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    pymisha_track_create::g_create_dir_override.clear();
    Py_RETURN_NONE;
}
