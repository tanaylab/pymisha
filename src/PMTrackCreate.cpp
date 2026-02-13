/*
 * PMTrackCreate.cpp
 *
 * Track creation backends for pymisha (dense/sparse 1D).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#include <climits>
#include <sys/stat.h>

#include <Python.h>

#include "pymisha.h"
#include "PMDataFrame.h"
#include "PMDb.h"
#include "GenomeTrack.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeTrackSparse.h"
#include "PMTrackExpressionScanner.h"
#include "PMTrackExpressionIterator.h"

using namespace std;

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
        vector<bool> created(chromkey.get_num_chroms(), false);

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

        if (!indexed_db) {
            for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
                if (created[chromid])
                    continue;
                string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);
                gtrack.init_write(path.c_str(), chromid);
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

PyObject *pm_track_create_dense(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track = nullptr;
        PyObject *py_data = nullptr;
        unsigned binsize = 0;
        double defval = std::numeric_limits<double>::quiet_NaN();

        if (!PyArg_ParseTuple(args, "sOId", &track, &py_data, &binsize, &defval))
            verror("Invalid arguments to pm_track_create_dense");

        if (binsize == 0)
            verror("binsize must be positive");

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
        size_t data_idx = 0;

        for (int chromid = 0; chromid < (int)chromkey.get_num_chroms(); ++chromid) {
            uint64_t chrom_size = chromkey.get_chrom_size(chromid);
            string path = track_dir + "/" + GenomeTrack::get_1d_filename(chromkey, chromid);

            GenomeTrackFixedBin gtrack;
            gtrack.init_write(path.c_str(), binsize, chromid);

            while (data_idx < recs.size() && recs[data_idx].chromid < chromid)
                ++data_idx;
            size_t chrom_begin = data_idx;
            while (data_idx < recs.size() && recs[data_idx].chromid == chromid)
                ++data_idx;
            size_t chrom_end = data_idx;

            size_t cur_idx = chrom_begin;
            vector<float> batch;
            batch.reserve(10000);

            for (uint64_t start = 0; start < chrom_size; start += binsize) {
                uint64_t end = std::min(start + (uint64_t)binsize, chrom_size);
                while (cur_idx < chrom_end && (uint64_t)recs[cur_idx].end <= start)
                    ++cur_idx;

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

                uint64_t width = end - start;
                if (covered < width && !std::isnan(defval)) {
                    uint64_t unc = width - covered;
                    sum += defval * (double)unc;
                    covered += unc;
                }

                float out = covered ? (float)(sum / (double)covered)
                                    : std::numeric_limits<float>::quiet_NaN();
                batch.push_back(out);
                if (batch.size() >= 10000) {
                    gtrack.write_next_bins(batch.data(), batch.size());
                    batch.clear();
                }
            }

            if (!batch.empty())
                gtrack.write_next_bins(batch.data(), batch.size());
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
        bool use_linear_ramp;
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
