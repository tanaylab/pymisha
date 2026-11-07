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
#include <set>
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
#include "IndexedTrackWriter.h"
#include "TrackIndex.h"

using namespace std;

// Thread-local override for the directory pm_track_create_* should mkdir into.
// When non-empty, track_name_to_dir() returns this exact path instead of the
// path computed from the track name. Set/cleared by pm_set_create_dir_override
// and pm_clear_create_dir_override. Mirrors R misha .create_dir_override
// (R 5.6.30 81635130) and enables atomic gtrack.create via tmp dir + rename.
namespace pymisha_track_create {
    thread_local std::string g_create_dir_override;

    // Worker count for pm_track_create_sparse's empty-chrom signature-file
    // dispatch. Set by Python wrapper from pm.CONFIG (multitasking +
    // max_processes); 0 means "use default" (4); 1 forces sequential.
    // Capped to a sane upper bound in the C++ side so misconfiguration
    // can't spawn hundreds of threads per call.
    thread_local int g_create_empty_writers = 0;
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

// A track of that name is an error, unless the track is built in a staging directory
// (_atomic_track_create), whose caller checked the name as R's .gconfirmtrackcreate does: a
// loaded dataset's track of the name is allowed, and the new one then wins.
static void check_new_track_name(const string &track_name)
{
    if (pymisha_track_create::g_create_dir_override.empty() && g_pmdb->track_exists(track_name))
        verror("Track '%s' already exists", track_name.c_str());
}

static string track_name_to_dir(const string &track_name)
{
    if (!pymisha_track_create::g_create_dir_override.empty())
        return pymisha_track_create::g_create_dir_override;

    if (!g_pmdb || !g_pmdb->is_initialized())
        TGLError("Database not initialized. Call gdb_init() first.");

    return g_pmdb->new_track_path(track_name);
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
// stdio FILE* mutex contention that fopen() introduces. A failure is reported
// back via `first_errno` so the caller can raise after all workers finish
// (misha's write_empty_sparse_signature).
static void write_empty_sparse_file_syscall(const string &path, int32_t sig,
                                            std::atomic<int> &first_errno)
{
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0664);
    if (fd < 0) {
        int e = errno ? errno : EIO;
        int expected = 0;
        first_errno.compare_exchange_strong(expected, e);
        return;
    }
    ssize_t n = ::write(fd, &sig, sizeof(sig));
    int werr = (n != (ssize_t)sizeof(sig)) ? (errno ? errno : EIO) : 0;
    if (::close(fd) && !werr)   // NFS reports a failed write here
        werr = errno ? errno : EIO;
    if (werr) {
        int expected = 0;
        first_errno.compare_exchange_strong(expected, werr);
    }
}

// Create the empty per-chrom files for chroms not yet written. Dispatches
// the open/write/close syscalls across a small thread pool so the NFS
// server can pipeline create requests. On hg38 (455 chroms, 1 chrom with
// data) this drops ~1.0 s of wall time to ~0.5 s. Mirrors the sequential
// loop semantics: each chrom that was not touched by the main write loop
// gets a 4-byte signature file at its canonical 1d filename.
//
// Worker count is controlled by pymisha_track_create::g_create_empty_writers
// (set by the Python wrapper from pm.CONFIG -- multitasking off => 1,
// otherwise min(max_processes, hard cap)). Value 0 means "use default".
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

    // Default: 4 workers (empirically saturates NFSv3 CREATE pipelining
    // for the hg38-on-NFS case); user can override via pm.CONFIG. Hard cap
    // at 16 so a misconfigured max_processes can't spawn hundreds of
    // threads here -- empty-file creates don't benefit from more.
    int requested = pymisha_track_create::g_create_empty_writers;
    if (requested <= 0) requested = 4;
    if (requested > 16) requested = 16;
    unsigned nworkers = std::min<unsigned>((unsigned)requested,
                                           (unsigned)paths.size());
    std::atomic<int> first_errno{0};

    if (nworkers <= 1) {
        for (const string &p : paths)
            write_empty_sparse_file_syscall(p, sig, first_errno);
    } else {
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            for (;;) {
                size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= paths.size()) return;
                write_empty_sparse_file_syscall(paths[i], sig, first_errno);
            }
        };

        vector<std::thread> threads;
        threads.reserve(nworkers - 1);
        for (unsigned t = 0; t < nworkers - 1; ++t)
            threads.emplace_back(worker);
        worker();  // current thread does its share
        for (auto &th : threads) th.join();
    }

    int e = first_errno.load();
    if (e != 0)
        TGLError("Failed to create empty chromosome track file: %s", strerror(e));
}

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
        check_new_track_name(track_name);

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
            // init_write() flushes each chromosome as it moves to the next; the
            // last one has nobody to flush it but us. Without this a full disk
            // truncates it silently and the create reports success.
            gtrack.flush_writes();
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
        // gtrack.import's arithmetic: misha's import multiplies a record's float value by its
        // overlap in float before adding it to the double sum; gtrack.create_dense multiplies
        // in double. Only the weighted mean has an import counterpart.
        int float_products = 0;

        if (!PyArg_ParseTuple(args, "sOIds|p", &track, &py_data, &binsize, &defval, &func_str, &float_products))
            verror("Invalid arguments to pm_track_create_dense");

        if (binsize == 0)
            verror("binsize must be positive");

        const DenseAggFunc func = parse_dense_agg_func(func_str);
        const bool is_wmean = (func == DenseAggFunc::WMEAN);

        string track_name(track);
        check_new_track_name(track_name);

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
        vector<size_t> active;

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

            // The first interval that has not reached a bin yet, and the intervals that overlap the
            // current bin, in sorted order. An interval stays in active for every bin it overlaps, so
            // a long interval does not make the bins after it rescan the short intervals it spans.
            size_t next_idx = chrom_begin;
            active.clear();
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

                // Drop the intervals that ended before this bin, and add those that start in it
                size_t num_active = 0;
                for (size_t idx : active) {
                    if ((uint64_t)recs[idx].end > start)
                        active[num_active++] = idx;
                }
                active.resize(num_active);
                while (next_idx < chrom_end && (uint64_t)recs[next_idx].start < end)
                    active.push_back(next_idx++);

                uint64_t width = end - start;
                // Bases of the bin under at least one interval with a value. Intervals may overlap,
                // so this is the length of their union, not the sum of their overlaps; defval
                // weighs the rest of the bin.
                uint64_t covered = 0;
                uint64_t covered_end = start;
                auto add_covered = [&](uint64_t ov_start, uint64_t ov_end) {
                    // ov_start does not decrease along the sorted intervals
                    if (ov_end > covered_end) {
                        covered += ov_end - std::max(ov_start, covered_end);
                        covered_end = ov_end;
                    }
                };
                float out;

                if (is_wmean) {
                    // Historical streaming-sum fast path. Byte-identical
                    // to pre-5.6.32 output.
                    double sum = 0;
                    uint64_t weight = 0;
                    for (size_t j : active) {
                        uint64_t ov_start = std::max(start, (uint64_t)recs[j].start);
                        uint64_t ov_end = std::min(end, (uint64_t)recs[j].end);
                        if (ov_end > ov_start && !std::isnan((double)recs[j].value)) {
                            uint64_t ov = ov_end - ov_start;
                            sum += float_products ? (double)(recs[j].value * (float)ov)
                                                  : (double)recs[j].value * (double)ov;
                            weight += ov;
                            add_covered(ov_start, ov_end);
                        }
                    }
                    if (covered < width && !std::isnan(defval)) {
                        uint64_t unc = width - covered;
                        sum += defval * (double)unc;
                        weight += unc;
                    }
                    out = weight ? (float)(sum / (double)weight)
                                 : std::numeric_limits<float>::quiet_NaN();
                } else {
                    contribs.clear();
                    for (size_t j : active) {
                        uint64_t ov_start = std::max(start, (uint64_t)recs[j].start);
                        uint64_t ov_end = std::min(end, (uint64_t)recs[j].end);
                        if (ov_end > ov_start && !std::isnan((double)recs[j].value)) {
                            uint64_t ov = ov_end - ov_start;
                            contribs.push_back({(double)recs[j].value, (double)ov});
                            add_covered(ov_start, ov_end);
                        }
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
            else
                // gtrack is per-iteration here, so its file is closed by the
                // destructor below - where a deferred ENOSPC/EDQUOT has nobody
                // to report to. Flush while it still can.
                gtrack.flush_writes();
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
        PyObject *py_vtracks = nullptr;

        if (!PyArg_ParseTuple(args, "ssO|OOO", &track, &expr, &py_intervals, &py_iterator, &py_config, &py_vtracks))
            verror("Invalid arguments to pm_track_create_expr");
        if (py_vtracks == Py_None)
            py_vtracks = nullptr;

        string track_name(track);
        check_new_track_name(track_name);

        long iterator_policy = parse_iterator_policy_local(py_iterator, 0, "pm_track_create_expr");

        vector<GInterval> intervals;
        parse_intervals_from_df(py_intervals, intervals);
        // An intervals iterator (-1: the scope already cut to its intervals) of no intervals
        // makes an empty sparse track, as in misha
        if (intervals.empty() && iterator_policy != -1)
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
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy, py_vtracks);

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
            gtrack.flush_writes();   // the last file written; see pm_track_create_sparse
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

            // With no interval at all, an indexed database too gets a file per chromosome:
            // what tells the indexed conversion that the track is sparse
            if (!indexed_db || none_of(created.begin(), created.end(), [](bool c) { return c; })) {
                for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
                    if (created[chromid])
                        continue;
                    string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);
                    gtrack.init_write(path.c_str(), chromid);
                }
            }
            gtrack.flush_writes();   // the last file written; see pm_track_create_sparse
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


namespace {

// gtrack_modify was the only pymisha writer that edited a track in place ("rb+"
// straight into the live file), so an interrupt, a bad expression or a full disk
// left the track durably half-old/half-new under its real name, structurally
// valid, with nothing to mark it as damaged. Worse than misha's on an indexed
// database, where "rb+" opens the consolidated track.dat and rewrites bins inside
// the shared file, and track.idx's CRC64 covers only (chrom_id, offset, length),
// never content - so nothing detects it afterwards.
//
// It is now staged like every other writer: the data files the modification
// touches are copied into a staging directory, the new values are written there,
// and the result is committed by renaming the staged files back over the
// originals. Until the commit the live track is untouched. Ported from misha
// 5.11.22 (src/GenomeTrackModify.cpp), including 5.11.24's fsync.
//
// Whole-track atomicity is only as good as the number of files: on an indexed
// database there is exactly one data file, so the commit is a single rename and
// is atomic. On a per-chromosome track it is one rename per touched chromosome,
// back to back with no interruptible work in between; each file is still
// individually consistent.
class Stager {
public:
    Stager(const string &track_dir, const string &stage_dir) :
        m_track_dir(track_dir), m_stage_dir(stage_dir),
        m_indexed(GenomeTrack::get_track_index(track_dir) != nullptr) {}

    bool enabled() const { return !m_stage_dir.empty(); }

    // Returns the path to open for update for this chromosome, copying whatever
    // backing file the chromosome lives in on first use.
    string stage(const string &chrom_filename)
    {
        if (m_indexed) {
            // All chromosomes share track.dat; the index is needed alongside it
            // so the staged copy resolves the same offsets.
            stage_file("track.idx", false);
            stage_file("track.dat", true);
        } else
            stage_file(chrom_filename, true);

        return m_stage_dir + "/" + chrom_filename;
    }

    // Renames the staged data files over the originals. Nothing else may be
    // interposed here: this is the commit. Each staged file is fsynced first and
    // the track directory after - a rename is atomic against other processes but
    // not against a machine death, which could otherwise leave the live name
    // pointing at bytes that never reached stable storage.
    void commit()
    {
        for (vector<string>::const_iterator i = m_committable.begin(); i != m_committable.end(); ++i)
            fsync_path(m_stage_dir + "/" + *i, false);

        for (vector<string>::const_iterator i = m_committable.begin(); i != m_committable.end(); ++i) {
            string src = m_stage_dir + "/" + *i;
            string dst = m_track_dir + "/" + *i;
            if (rename(src.c_str(), dst.c_str()))
                verror("Failed to commit %s to %s: %s", src.c_str(), dst.c_str(), strerror(errno));
        }

        if (!m_committable.empty())
            fsync_path(m_track_dir, true);

        m_committable.clear();
    }

private:
    string         m_track_dir;
    string         m_stage_dir;
    bool           m_indexed;
    set<string>    m_staged;
    vector<string> m_committable;

    // A directory fsync answers EINVAL on filesystems that do not implement it;
    // that means "unsupported", not "lost data".
    static void fsync_path(const string &path, bool is_dir)
    {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
            verror("Failed to open %s for fsync: %s", path.c_str(), strerror(errno));

        int rc = fsync(fd);
        int err = errno;
        close(fd);

        if (rc && !(is_dir && err == EINVAL))
            verror("Failed to fsync %s: %s", path.c_str(), strerror(err));
    }

    void stage_file(const string &name, bool committable)
    {
        if (!m_staged.insert(name).second)
            return;
        copy_file(m_track_dir + "/" + name, m_stage_dir + "/" + name);
        if (committable)
            m_committable.push_back(name);
    }

    static void copy_file(const string &src, const string &dst)
    {
        int in = ::open(src.c_str(), O_RDONLY);
        if (in < 0)
            verror("Cannot open %s: %s", src.c_str(), strerror(errno));

        int out = ::open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (out < 0) {
            int err = errno;
            ::close(in);
            verror("Cannot create %s: %s", dst.c_str(), strerror(err));
        }

        // The staged file is renamed over the original, so it must carry the
        // original's permissions rather than whatever the umask happens to be.
        struct stat st;
        if (!::fstat(in, &st))
            (void)::fchmod(out, st.st_mode & 07777);

        vector<char> buf(1 << 20);
        while (1) {
            ssize_t nread = ::read(in, &buf[0], buf.size());
            if (nread < 0) {
                if (errno == EINTR)
                    continue;
                int err = errno;
                ::close(in);
                ::close(out);
                verror("Reading %s: %s", src.c_str(), strerror(err));
            }
            if (!nread)
                break;

            ssize_t written = 0;
            while (written < nread) {
                ssize_t n = ::write(out, &buf[written], nread - written);
                if (n < 0) {
                    if (errno == EINTR)
                        continue;
                    int err = errno;
                    ::close(in);
                    ::close(out);
                    verror("Writing %s: %s", dst.c_str(), strerror(err));
                }
                written += n;
            }

            // A multi-GB track.dat must stay interruptible.
            try {
                check_interrupt();
            } catch (...) {
                ::close(in);
                ::close(out);
                throw;
            }
        }

        ::close(in);
        // close() is where a deferred ENOSPC/EDQUOT surfaces on some filesystems.
        if (::close(out))
            verror("Writing %s: %s", dst.c_str(), strerror(errno));
    }
};

}

PyObject *pm_modify(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        const char *expr = nullptr;
        PyObject *py_intervals = nullptr;
        long iterator_policy = 0;
        PyObject *py_vtracks = nullptr;
        const char *stage_dir = nullptr;

        if (!PyArg_ParseTuple(args, "ssOl|Oz", &track, &expr, &py_intervals, &iterator_policy, &py_vtracks, &stage_dir))
            verror("Invalid arguments to pm_modify");
        if (py_vtracks == Py_None)
            py_vtracks = nullptr;

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
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy, py_vtracks);

        PMFixedBinIterator *fitr = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
        if (!fitr)
            verror("gtrack_modify requires a fixed-bin (dense) iterator");

        GenomeTrackFixedBin gtrack;
        int cur_chromid = -1;
        Stager stager(track_dir, stage_dir ? string(stage_dir) : string());
        const GenomeTrack::ChromFiles &files = g_pmdb->chrom_files(track_dir);
        std::set<int> modified;

        for (; !scanner.isend(); scanner.next()) {
            const GInterval &interv = scanner.last_interval();

            if (interv.chromid != cur_chromid) {
                cur_chromid = interv.chromid;
                const string &fname = files.filename(cur_chromid);
                string path = track_dir + "/" + fname;
                // The values go to the staging copy; the live track keeps the old
                // ones until commit() below.
                if (stager.enabled()) {
                    gtrack.init_update(stager.stage(fname).c_str(), cur_chromid);
                    modified.insert(cur_chromid);
                } else
                    gtrack.init_update(path.c_str(), cur_chromid);
            }

            uint64_t bin_idx = interv.start / gtrack.get_bin_size();
            gtrack.goto_bin(bin_idx);
            float v = (float)scanner.vdouble();
            gtrack.write_next_bin(v);
        }

        // Everything is written and reported before anything is published: a
        // short write must fail here, not silently at fclose() after commit.
        gtrack.flush_writes();
        // A chromosome's staged file is closed when the next one is opened, and a failure there
        // is only recorded (BufferedFile::write_error): it must stop the commit too.
        if (BufferedFile::write_error()[0])
            verror("%s", BufferedFile::write_error());
        stager.commit();

        // The commit replaced the file each modified chromosome was read from. Its other names (links to the
        // same file, copies with the same bytes, or files the read one hid) become names of the committed file,
        // so they do not differ from it and whatever links to them reads the new data (R's gtrack_modify).
        for (const GenomeTrack::ChromFiles::File &file : files.files()) {
            if (file.chromid < 0 || !modified.count(file.chromid))
                continue;
            const string &committed = files.filename(file.chromid);
            string name = track_dir + "/" + file.name;
            struct stat st1, st2;
            // not one that leads to the committed file already (a symlink to it)
            if (file.name != committed && (stat(name.c_str(), &st1) || stat((track_dir + "/" + committed).c_str(), &st2) ||
                                           st1.st_dev != st2.st_dev || st1.st_ino != st2.st_ino)) {
                string tmp = track_dir + "/." + file.name + ".link";  // hidden from the listing
                unlink(tmp.c_str());
                if (!link((track_dir + "/" + committed).c_str(), tmp.c_str()) && rename(tmp.c_str(), name.c_str()))
                    unlink(tmp.c_str());
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

namespace {

// Running-sum window smoothers, as R misha's GenomeTrackSmooth.cpp: each sample updates the
// window sums instead of rescanning the whole window. A sample is written once the samples to
// its right are read; finish() writes the last ones of the chromosome.
class Smoother {
public:
    Smoother(GenomeTrackFixedBin &gtrack, unsigned num_samples_aside, bool smooth_nans) :
        m_gtrack(gtrack), m_smooth_nans(smooth_nans), m_counter(0), m_num_read_samples(0),
        m_num_samples(2 * num_samples_aside + 1), m_num_samples_aside(num_samples_aside),
        m_left_idx(0), m_peak_idx(num_samples_aside)
    {
        m_vals.resize(m_num_samples, numeric_limits<double>::quiet_NaN());
    }
    virtual ~Smoother() {}

    virtual void set_next_sample(double sample) = 0;

    void finish()
    {
        for (unsigned i = 0; i <= m_num_samples_aside; ++i)
            set_next_sample(numeric_limits<double>::quiet_NaN());
    }

protected:
    GenomeTrackFixedBin &m_gtrack;
    bool           m_smooth_nans;
    int            m_counter;
    unsigned       m_num_read_samples;
    unsigned       m_num_samples;
    unsigned       m_num_samples_aside;
    vector<double> m_vals;
    unsigned       m_left_idx;
    unsigned       m_peak_idx;
};

class LinearRampSmoother : public Smoother {
public:
    LinearRampSmoother(GenomeTrackFixedBin &gtrack, unsigned num_samples_aside, double weight_thr, bool smooth_nans) :
        Smoother(gtrack, num_samples_aside, smooth_nans),
        m_weight_thr(weight_thr * (num_samples_aside + 1)),
        m_left_nans(num_samples_aside), m_right_nans(num_samples_aside + 1),
        m_left_weights_sum(0), m_right_weights_sum(0), m_left_vals_sum(0), m_right_vals_sum(0),
        m_left_weighted_vals_sum(0), m_right_weighted_vals_sum(0) {}

    void set_next_sample(double sample) override
    {
        if (m_num_read_samples > m_num_samples_aside) {
            float mean;
            double weight = m_left_weights_sum + m_right_weights_sum;

            if ((!m_smooth_nans && std::isnan(m_vals[m_peak_idx])) || !weight || weight < m_weight_thr)
                mean = numeric_limits<float>::quiet_NaN();
            else
                mean = (m_left_weighted_vals_sum + m_right_weighted_vals_sum) / weight;
            m_gtrack.write_next_bin(mean);
            m_counter++;
        } else
            m_num_read_samples++;

        // shift all elements left to the peak one position left
        m_left_weights_sum -= m_num_samples_aside - m_left_nans;
        m_left_weighted_vals_sum -= m_left_vals_sum;

        // release the leftmost element
        if (std::isnan(m_vals[m_left_idx]))
            m_left_nans--;
        else
            m_left_vals_sum -= m_vals[m_left_idx];

        // shift the peak element left
        if (std::isnan(m_vals[m_peak_idx])) {
            m_left_nans++;
            m_right_nans--;
        } else {
            m_left_weights_sum += m_num_samples_aside;
            m_left_vals_sum += m_vals[m_peak_idx];
            m_left_weighted_vals_sum += m_vals[m_peak_idx] * m_num_samples_aside;

            m_right_weights_sum -= m_num_samples_aside + 1;
            m_right_vals_sum -= m_vals[m_peak_idx];
            m_right_weighted_vals_sum -= m_vals[m_peak_idx] * (m_num_samples_aside + 1);
        }

        // shift all elements right to the peak one position left
        m_right_weights_sum += m_num_samples_aside - m_right_nans;
        m_right_weighted_vals_sum += m_right_vals_sum;

        // add the new element at the rightmost position
        if (std::isnan(sample))
            m_right_nans++;
        else {
            m_right_weights_sum++;
            m_right_vals_sum += sample;
            m_right_weighted_vals_sum += sample;
        }
        m_vals[m_left_idx] = sample;

        if (++m_left_idx >= m_num_samples) m_left_idx = 0;
        if (++m_peak_idx >= m_num_samples) m_peak_idx = 0;

        // The running sums lose precision over time: recalculate them every half window of
        // written samples (the weights sums hold integers and stay exact). Not while the window
        // fills, when m_counter stays 0.
        if (m_num_read_samples > m_num_samples_aside && m_counter % m_num_samples_aside == 0) {
            double val;

            m_left_vals_sum = m_right_vals_sum = 0;
            m_left_weighted_vals_sum = m_right_weighted_vals_sum = 0;

            for (unsigned i = 1; i <= m_num_samples_aside; ++i) {
                // left of the peak
                val = m_vals[(m_peak_idx + m_num_samples - i) % m_num_samples];
                if (!std::isnan(val))
                    m_left_vals_sum += val;
                m_left_weighted_vals_sum += m_left_vals_sum;

                // right of the peak
                val = m_vals[(m_peak_idx + i) % m_num_samples];
                if (!std::isnan(val))
                    m_right_vals_sum += val;
                m_right_weighted_vals_sum += m_right_vals_sum;
            }

            // the peak itself (it is counted at the right)
            val = m_vals[m_peak_idx];
            if (!std::isnan(val)) {
                m_right_weighted_vals_sum += val * (m_num_samples_aside + 1);
                m_right_vals_sum += val;
            }
        }
    }

private:
    double   m_weight_thr;
    // left = the samples left to the peak, right = the peak sample and the samples right to it
    unsigned m_left_nans;
    unsigned m_right_nans;
    double   m_left_weights_sum;
    double   m_right_weights_sum;
    double   m_left_vals_sum;
    double   m_right_vals_sum;
    double   m_left_weighted_vals_sum;
    double   m_right_weighted_vals_sum;
};

class MeanSmoother : public Smoother {
public:
    MeanSmoother(GenomeTrackFixedBin &gtrack, unsigned num_samples_aside, double weight_thr, bool smooth_nans) :
        Smoother(gtrack, num_samples_aside, smooth_nans), m_weight_thr(weight_thr), m_weights_sum(0), m_vals_sum(0) {}

    void set_next_sample(double sample) override
    {
        if (m_num_read_samples > m_num_samples_aside) {
            float mean = (!m_smooth_nans && std::isnan(m_vals[m_peak_idx])) || !m_weights_sum || m_weights_sum < m_weight_thr ?
                numeric_limits<float>::quiet_NaN() : m_vals_sum / m_weights_sum;
            m_gtrack.write_next_bin(mean);
            m_counter++;
        } else
            m_num_read_samples++;

        if (!std::isnan(m_vals[m_left_idx])) {
            m_weights_sum--;
            m_vals_sum -= m_vals[m_left_idx];
        }

        if (!std::isnan(sample)) {
            m_weights_sum++;
            m_vals_sum += sample;
        }

        m_vals[m_left_idx] = sample;

        if (++m_left_idx >= m_num_samples) m_left_idx = 0;
        if (++m_peak_idx >= m_num_samples) m_peak_idx = 0;

        // The running sum loses precision over time: recalculate it every half window of
        // written samples. Not while the window fills, when m_counter stays 0.
        if (m_num_read_samples > m_num_samples_aside && m_counter % m_num_samples_aside == 0) {
            m_vals_sum = 0;
            for (vector<double>::iterator i = m_vals.begin(); i != m_vals.end(); ++i) {
                if (!std::isnan(*i))
                    m_vals_sum += *i;
            }
        }
    }

private:
    double m_weight_thr;
    double m_weights_sum;
    double m_vals_sum;
};

}  // namespace

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
        PyObject *py_vtracks = nullptr;

        if (!PyArg_ParseTuple(args, "ssOlddis|O", &track, &expr, &py_intervals, &iterator_policy, &winsize, &weight_thr, &smooth_nans, &alg, &py_vtracks))
            verror("Invalid arguments to pm_smooth");
        if (py_vtracks == Py_None)
            py_vtracks = nullptr;

        string track_name(track);
        check_new_track_name(track_name);

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
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy, py_vtracks);

        PMFixedBinIterator *fitr = dynamic_cast<PMFixedBinIterator *>(scanner.get_iterator());
        if (!fitr)
            verror("gtrack_smooth requires a fixed-bin (dense) iterator");
        int64_t binsize = fitr->get_bin_size();
        if (binsize <= 0)
            verror("Failed to infer dense iterator binsize");

        // the window size is taken as a whole number, as R does
        unsigned num_samples_aside = (unsigned)(0.5 * std::floor(winsize) / binsize + 0.5);
        if (!num_samples_aside)
            verror("Smoothing window is narrow and does not cover any other values around the center");
        g_pymisha->verify_max_data_size(2 * (uint64_t)num_samples_aside + 1, "Smoothing window");

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        const bool indexed_db = db_is_indexed();
        vector<bool> created(chromkey.get_num_chroms(), false);

        GenomeTrackFixedBin out_track;
        int cur_chromid = -1;
        std::unique_ptr<Smoother> smoother;

        for (; !scanner.isend(); scanner.next()) {
            const GInterval &interv = scanner.last_interval();

            if (interv.chromid != cur_chromid) {
                // the previous chromosome's last bins, before init_write() moves to the next one
                if (smoother)
                    smoother->finish();

                cur_chromid = interv.chromid;
                string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, cur_chromid);
                out_track.init_write(path.c_str(), (unsigned)binsize, cur_chromid);
                created[cur_chromid] = true;

                if (use_linear_ramp)
                    smoother.reset(new LinearRampSmoother(out_track, num_samples_aside, weight_thr, smooth_nans));
                else
                    smoother.reset(new MeanSmoother(out_track, num_samples_aside, weight_thr, smooth_nans));
            }

            smoother->set_next_sample(scanner.vdouble());
        }

        if (smoother)
            smoother->finish();

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

// Worker count for the empty-chrom signature-file dispatch inside
// pm_track_create_sparse. 0 means "default" (4 workers); 1 forces
// sequential. Thread-local so concurrent callers don't fight over
// a global. Python wrapper sets it from pm.CONFIG (multitasking +
// max_processes) before each pm_track_create_sparse call.
PyObject *pm_set_create_parallel_writers(PyObject *self, PyObject *args)
{
    (void)self;
    int n = 0;
    if (!PyArg_ParseTuple(args, "i", &n))
        return nullptr;
    pymisha_track_create::g_create_empty_writers = n;
    Py_RETURN_NONE;
}
