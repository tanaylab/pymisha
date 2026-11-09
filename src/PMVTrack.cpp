/*
 * PMVTrack.cpp
 *
 * Virtual track computation for pymisha (C++ backend).
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <unistd.h>

#include <string>
#include "pymisha.h"
#include "PMDb.h"
#include "GInterval.h"
#include "GenomeTrack.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeTrackSparse.h"
#include "GenomeTrackArray.h"
#include "GenomeTrackInMemory.h"
#include "PMTrackExpressionVars.h"
#include "GenomeSeqFetch.h"
#include "PWMScorer.h"
#include "PottsScorer.h"
#include "PMPottsParams.h"
#include "PWMEditDistanceScorer.h"
#include "PWMLseEditDistanceScorer.h"
#include "KmerCounter.h"
#include "MaskedBpCounter.h"
#include "SegmentFinder.h"
#include "Segment.h"
#include "pmutils.h"

extern PyObject *s_pm_err;

// Defined in PMStubs.cpp. Declared outside the anonymous namespace below so
// they keep external linkage.
extern void normalize_interval_columns(PMPY &py_chrom, PMPY &py_start, PMPY &py_end);
extern void verify_interval_coords(const GenomeChromKey &chromkey, int chromid,
                                   int64_t start, int64_t end);
extern std::string py_chrom_name(PyObject *val, Py_ssize_t i);

namespace {

bool str_to_bool(PyObject *obj, bool default_val) {
    if (!obj) return default_val;
    if (obj == Py_True) return true;
    if (obj == Py_False) return false;
    if (PyBool_Check(obj)) return obj == Py_True;
    if (PyNumber_Check(obj)) {
        PMPY tmp(PyNumber_Long(obj), true);
        if (!tmp) return default_val;
        long v = PyLong_AsLong(tmp);
        return v != 0;
    }
    return default_val;
}

int64_t obj_to_int64(PyObject *obj, int64_t default_val, bool *found = nullptr) {
    if (!obj || obj == Py_None) {
        if (found) *found = false;
        return default_val;
    }
    if (found) *found = true;
    PMPY tmp(PyNumber_Long(obj), true);
    if (!tmp) {
        PyErr_Clear();
        return default_val;
    }
    return PyLong_AsLongLong(tmp);
}

double obj_to_double(PyObject *obj, double default_val, bool *found = nullptr) {
    if (!obj || obj == Py_None) {
        if (found) *found = false;
        return default_val;
    }
    if (found) *found = true;
    PMPY tmp(PyNumber_Float(obj), true);
    if (!tmp) {
        PyErr_Clear();
        return default_val;
    }
    return PyFloat_AsDouble(tmp);
}

std::string obj_to_string(PyObject *obj, const std::string &default_val, bool *found = nullptr) {
    if (!obj || obj == Py_None) {
        if (found) *found = false;
        return default_val;
    }
    if (found) *found = true;
    if (!PyUnicode_Check(obj)) {
        return default_val;
    }
    const char *s = PyUnicode_AsUTF8(obj);
    if (!s) return default_val;
    return std::string(s);
}

PyObject *dict_get(PyObject *dict, const char *key) {
    if (!dict || !PyDict_Check(dict)) return nullptr;
    return PyDict_GetItemString(dict, key);
}

static char parse_strand(PyObject *obj) {
    if (!obj || obj == Py_None) return 0;
    if (PyUnicode_Check(obj)) {
        const char *s = PyUnicode_AsUTF8(obj);
        if (!s) return 0;
        if (s[0] == '+') return 1;
        if (s[0] == '-') return -1;
        return 0;
    }
    if (PyNumber_Check(obj)) {
        PMPY tmp(PyNumber_Long(obj), true);
        if (!tmp) return 0;
        long v = PyLong_AsLong(tmp);
        if (v > 0) return 1;
        if (v < 0) return -1;
        return 0;
    }
    return 0;
}

bool apply_shift(const GInterval &in, int64_t sshift, int64_t eshift,
                 const GenomeChromKey &chromkey, GInterval &out) {
    out = in;
    int64_t start = in.start + sshift;
    int64_t end = in.end + eshift;
    if (start < 0) start = 0;
    int64_t chrom_size = (int64_t)chromkey.get_chrom_size(in.chromid);
    if (end > chrom_size) end = chrom_size;
    out.start = start;
    out.end = end;
    return out.start < out.end;
}

// The scores the pwm and potts scorers keep from earlier reads are stale once
// the sequence changes under them; the other sequence scorers keep none.
inline void invalidate_scorer_cache(PWMScorer &s) { s.invalidate_cache(); }
inline void invalidate_scorer_cache(PottsScorer &s) { s.invalidate_cache(); }
inline void invalidate_scorer_cache(GenomeSeqScorer &) {}

// A pwm.max.pos / potts.max.pos scorer that answers each interval with the
// max score its position was chosen by (get_last_max_score()): the score R's
// filter aggregation ranks the unmasked parts by, as vtracks.py does here.
template <class Scorer>
struct LastMaxScore {
    Scorer &scorer;
    double score_interval(const GInterval &interval, const GenomeChromKey &chromkey) {
        scorer.score_interval(interval, chromkey);
        return scorer.get_last_max_score();
    }
};
template <class Scorer>
inline void invalidate_scorer_cache(LastMaxScore<Scorer> &s) { s.scorer.invalidate_cache(); }

// Per-interval substitutions for gextract_variants, from the spec's optional
// sub_pos / sub_base arrays (aligned to `intervals`; base 0..3 = A, C, G, T,
// -1 = none). Empty when the spec has none.
std::vector<SeqSubstitution> parse_substitutions(PyObject *spec, const std::vector<GInterval> &intervals) {
    std::vector<SeqSubstitution> subs;
    PyObject *py_pos = PyDict_GetItemString(spec, "sub_pos");
    PyObject *py_base = PyDict_GetItemString(spec, "sub_base");
    if (!py_pos || !py_base || py_pos == Py_None || py_base == Py_None)
        return subs;
    PMPY pos_arr(PyArray_FROM_OTF(py_pos, NPY_INT64, NPY_ARRAY_IN_ARRAY), true);
    PMPY base_arr(PyArray_FROM_OTF(py_base, NPY_INT64, NPY_ARRAY_IN_ARRAY), true);
    if (!pos_arr || !base_arr || PyArray_NDIM((PyArrayObject *)*pos_arr) != 1 ||
        PyArray_NDIM((PyArrayObject *)*base_arr) != 1 ||
        (size_t)PyArray_DIM((PyArrayObject *)*pos_arr, 0) != intervals.size() ||
        (size_t)PyArray_DIM((PyArrayObject *)*base_arr, 0) != intervals.size()) {
        PyErr_Clear();
        TGLError("sub_pos and sub_base must be integer arrays with one entry per interval");
    }
    const int64_t *pos = (const int64_t *)PyArray_DATA((PyArrayObject *)*pos_arr);
    const int64_t *base = (const int64_t *)PyArray_DATA((PyArrayObject *)*base_arr);
    static const char BASES[4] = {'A', 'C', 'G', 'T'};
    subs.resize(intervals.size());
    for (size_t i = 0; i < intervals.size(); ++i) {
        if (base[i] < -1 || base[i] > 3)
            TGLError("Invalid substituted base code %lld", (long long)base[i]);
        if (base[i] >= 0) {
            subs[i].chromid = intervals[i].chromid;
            subs[i].pos = pos[i];
            subs[i].base = BASES[base[i]];
        }
    }
    return subs;
}

// Score every interval, shifted, with subs[i] applied to the sequence while
// interval i is scored (when subs is not empty).
template <class Scorer>
void score_intervals(Scorer &scorer, GenomeSeqFetch &seq, const std::vector<GInterval> &intervals,
                     const std::vector<SeqSubstitution> &subs, int64_t sshift, int64_t eshift,
                     const GenomeChromKey &chromkey, double *out) {
    SeqSubstitution cur;
    for (size_t i = 0; i < intervals.size(); ++i) {
        if (!subs.empty() && !(subs[i] == cur)) {
            cur = subs[i];
            seq.set_substitution(cur);
            invalidate_scorer_cache(scorer);
        }
        GInterval eval;
        if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
            out[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        out[i] = scorer.score_interval(eval, chromkey);
    }
}

bool interval_cmp_start(const GInterval &a, const GInterval &b) {
    if (a.chromid != b.chromid) return a.chromid < b.chromid;
    if (a.start != b.start) return a.start < b.start;
    return a.end < b.end;
}

bool interval_cmp_end(const GInterval &a, const GInterval &b) {
    if (a.chromid != b.chromid) return a.chromid < b.chromid;
    if (a.end != b.end) return a.end < b.end;
    return a.start < b.start;
}

void unify_overlaps(std::vector<GInterval> &intervs) {
    if (intervs.empty()) return;
    std::sort(intervs.begin(), intervs.end(), interval_cmp_start);
    std::vector<GInterval> merged;
    merged.reserve(intervs.size());
    GInterval cur = intervs.front();
    for (size_t i = 1; i < intervs.size(); ++i) {
        const GInterval &next = intervs[i];
        if (next.chromid == cur.chromid && next.start <= cur.end) {
            if (next.end > cur.end) cur.end = next.end;
        } else {
            merged.push_back(cur);
            cur = next;
        }
    }
    merged.push_back(cur);
    intervs.swap(merged);
}

void convert_py_intervals_basic(PyObject *py_intervals, std::vector<GInterval> &intervals) {
    PMPY py_chrom;
    PMPY py_start;
    PMPY py_end;

    if (PyList_Check(py_intervals) && PyList_Size(py_intervals) >= 2) {
        PyObject *colnames = PyList_GetItem(py_intervals, 0);
        if (colnames && PyArray_Check(colnames)) {
            Py_ssize_t num_cols = PyArray_SIZE((PyArrayObject *)colnames);
            int chrom_idx = -1, start_idx = -1, end_idx = -1;

            for (Py_ssize_t i = 0; i < num_cols; ++i) {
                PyObject *name = PyArray_GETITEM((PyArrayObject *)colnames,
                    (const char *)PyArray_GETPTR1((PyArrayObject *)colnames, i));
                if (name && PyUnicode_Check(name)) {
                    const char *name_str = PyUnicode_AsUTF8(name);
                    if (strcmp(name_str, "chrom") == 0) chrom_idx = i;
                    else if (strcmp(name_str, "start") == 0) start_idx = i;
                    else if (strcmp(name_str, "end") == 0) end_idx = i;
                }
                Py_XDECREF(name);
            }

            if (chrom_idx >= 0 && start_idx >= 0 && end_idx >= 0) {
                py_chrom.assign(PyList_GetItem(py_intervals, chrom_idx + 1), false);
                py_start.assign(PyList_GetItem(py_intervals, start_idx + 1), false);
                py_end.assign(PyList_GetItem(py_intervals, end_idx + 1), false);
            }
        }
    }

    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        py_chrom.assign(PyObject_GetAttrString(py_intervals, "chrom"), true);
        py_start.assign(PyObject_GetAttrString(py_intervals, "start"), true);
        py_end.assign(PyObject_GetAttrString(py_intervals, "end"), true);
    }

    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        PMPY key_chrom(PyUnicode_FromString("chrom"), true);
        PMPY key_start(PyUnicode_FromString("start"), true);
        PMPY key_end(PyUnicode_FromString("end"), true);
        py_chrom.assign(PyObject_GetItem(py_intervals, key_chrom), true);
        py_start.assign(PyObject_GetItem(py_intervals, key_start), true);
        py_end.assign(PyObject_GetItem(py_intervals, key_end), true);
    }

    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        TGLError("intervals must have 'chrom', 'start', and 'end' columns");
    }

    normalize_interval_columns(py_chrom, py_start, py_end);

    Py_ssize_t len = PyObject_Length(py_chrom);
    if (len < 0) {
        PyErr_Clear();
        TGLError("Cannot determine length of intervals");
    }

    intervals.clear();
    intervals.reserve(len);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();

    for (Py_ssize_t i = 0; i < len; ++i) {
        PMPY chrom_val(PySequence_GetItem(py_chrom, i), true);
        PMPY start_val(PySequence_GetItem(py_start, i), true);
        PMPY end_val(PySequence_GetItem(py_end, i), true);

        if (!chrom_val || !start_val || !end_val) {
            PyErr_Clear();
            TGLError("Failed to get interval values at index %ld", (long)i);
        }

        int chromid = chromkey.chrom2id(py_chrom_name(chrom_val, i));

        int64_t start = PyLong_AsLongLong(start_val);
        int64_t end = PyLong_AsLongLong(end_val);

        // a missing (None) or non-integer coordinate, as R says of NA (1-based)
        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("Invalid format of interval at index %ld", (long)i + 1);
        }

        if (start < 0 || start >= end ||
            (uint64_t)end > chromkey.get_chrom_size(chromid))
            verify_interval_coords(chromkey, chromid, start, end);

        intervals.emplace_back(chromid, start, end);
    }
}

void convert_py_intervals_with_values(PyObject *py_intervals,
                                      std::vector<GInterval> &intervals,
                                      std::vector<double> &values,
                                      bool *has_values,
                                      bool *has_strand) {
    if (has_values) *has_values = false;
    if (has_strand) *has_strand = false;

    if (!PyList_Check(py_intervals) || PyList_Size(py_intervals) < 2) {
        convert_py_intervals_basic(py_intervals, intervals);
        values.clear();
        return;
    }

    PyObject *colnames = PyList_GetItem(py_intervals, 0);
    if (!colnames || !PyArray_Check(colnames)) {
        convert_py_intervals_basic(py_intervals, intervals);
        values.clear();
        return;
    }

    Py_ssize_t num_cols = PyArray_SIZE((PyArrayObject *)colnames);
    int chrom_idx = -1, start_idx = -1, end_idx = -1, strand_idx = -1, value_idx = -1;

    for (Py_ssize_t i = 0; i < num_cols; ++i) {
        PyObject *name = PyArray_GETITEM((PyArrayObject *)colnames,
            (const char *)PyArray_GETPTR1((PyArrayObject *)colnames, i));
        if (name && PyUnicode_Check(name)) {
            const char *name_str = PyUnicode_AsUTF8(name);
            if (strcmp(name_str, "chrom") == 0) chrom_idx = i;
            else if (strcmp(name_str, "start") == 0) start_idx = i;
            else if (strcmp(name_str, "end") == 0) end_idx = i;
            else if (strcmp(name_str, "strand") == 0) strand_idx = i;
            else if (strcmp(name_str, "intervalID") != 0 && value_idx == -1) value_idx = i;
        }
        Py_XDECREF(name);
    }

    if (chrom_idx < 0 || start_idx < 0 || end_idx < 0) {
        convert_py_intervals_basic(py_intervals, intervals);
        values.clear();
        return;
    }

    PMPY py_chrom(PyList_GetItem(py_intervals, chrom_idx + 1), false);
    PMPY py_start(PyList_GetItem(py_intervals, start_idx + 1), false);
    PMPY py_end(PyList_GetItem(py_intervals, end_idx + 1), false);
    PMPY py_strand;
    PMPY py_value;

    if (strand_idx >= 0) {
        py_strand.assign(PyList_GetItem(py_intervals, strand_idx + 1), false);
        if (has_strand) *has_strand = true;
    }
    if (value_idx >= 0) {
        py_value.assign(PyList_GetItem(py_intervals, value_idx + 1), false);
        if (has_values) *has_values = true;
    }

    normalize_interval_columns(py_chrom, py_start, py_end);

    Py_ssize_t len = PyObject_Length(py_chrom);
    if (len < 0) {
        PyErr_Clear();
        TGLError("Cannot determine length of intervals");
    }

    intervals.clear();
    intervals.reserve(len);
    values.clear();
    values.reserve(len);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();

    for (Py_ssize_t i = 0; i < len; ++i) {
        PMPY chrom_val(PySequence_GetItem(py_chrom, i), true);
        PMPY start_val(PySequence_GetItem(py_start, i), true);
        PMPY end_val(PySequence_GetItem(py_end, i), true);

        if (!chrom_val || !start_val || !end_val) {
            PyErr_Clear();
            TGLError("Failed to get interval values at index %ld", (long)i);
        }

        int chromid = chromkey.chrom2id(py_chrom_name(chrom_val, i));

        int64_t start = PyLong_AsLongLong(start_val);
        int64_t end = PyLong_AsLongLong(end_val);

        // a missing (None) or non-integer coordinate, as R says of NA (1-based)
        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("Invalid format of interval at index %ld", (long)i + 1);
        }

        if (start < 0 || start >= end ||
            (uint64_t)end > chromkey.get_chrom_size(chromid))
            verify_interval_coords(chromkey, chromid, start, end);

        char strand = 0;
        if (py_strand) {
            PMPY strand_val(PySequence_GetItem(py_strand, i), true);
            strand = parse_strand(strand_val);
        }

        intervals.emplace_back(chromid, start, end, strand);

        if (py_value) {
            PMPY val(PySequence_GetItem(py_value, i), true);
            double v = std::numeric_limits<double>::quiet_NaN();
            if (val && val != Py_None) {
                if (PyNumber_Check(val)) {
                    PMPY tmp(PyNumber_Float(val), true);
                    if (tmp) v = PyFloat_AsDouble(tmp);
                }
            }
            values.push_back(v);
        }
    }
}

double compute_quantile(std::vector<double> &vals, double percentile) {
    if (vals.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(vals.begin(), vals.end());
    if (percentile <= 0) return vals.front();
    if (percentile >= 1) return vals.back();
    double pos = percentile * (vals.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    size_t hi = (size_t)std::ceil(pos);
    if (lo == hi) return vals[lo];
    double frac = pos - lo;
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

bool parse_pssm(PyObject *obj, DnaPSSM &pssm, double prior) {
    if (!obj || obj == Py_None) return false;

    PMPY arr(PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
    if (!arr || !PyArray_Check((PyArrayObject *)*arr) || PyArray_NDIM((PyArrayObject *)*arr) != 2) {
        PyErr_Clear();
        return false;
    }

    npy_intp rows = PyArray_DIM((PyArrayObject *)*arr, 0);
    npy_intp cols = PyArray_DIM((PyArrayObject *)*arr, 1);
    if (rows <= 0 || cols <= 0) return false;

    npy_intp len = rows;
    bool transposed = false;
    if (cols == 4) {
        len = rows;
        transposed = false;
    } else if (rows == 4) {
        len = cols;
        transposed = true;
    } else {
        return false;
    }

    pssm.resize((int)len);
    for (npy_intp i = 0; i < len; ++i) {
        double pa, pc, pg, pt;
        if (!transposed) {
            pa = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 0);
            pc = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 1);
            pg = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 2);
            pt = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, i, 3);
        } else {
            pa = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 0, i);
            pc = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 1, i);
            pg = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 2, i);
            pt = *(double *)PyArray_GETPTR2((PyArrayObject *)*arr, 3, i);
        }
        pssm[i] = DnaProbVec((float)pa, (float)pc, (float)pg, (float)pt);
    }

    if (prior > 0) {
        pssm.add_dirichlet_prior((float)prior);
    }
    return true;
}

} // namespace


// ---- value-based vtracks (a DataFrame source), as R misha's TrackExpressionVars::Value_var ----

// The track of a value-based vtrack, as R misha builds it (add_vtrack_var_src_value): the
// intervals sorted by chromosome and start, each value held as a float, and the functions func
// reads registered.
static void value_var_track(GenomeTrackInMemory &track, const std::vector<GInterval> &intervals,
                            const std::vector<double> &values, const std::string &func)
{
    std::vector<std::pair<GInterval, float>> paired;
    paired.reserve(intervals.size());
    for (size_t i = 0; i < intervals.size(); ++i)
        paired.emplace_back(intervals[i], (float)values[i]);
    std::sort(paired.begin(), paired.end(), [](const std::pair<GInterval, float> &a, const std::pair<GInterval, float> &b) {
        return a.first < b.first;
    });
    GIntervals sorted_intervals;
    std::vector<float> sorted_values;
    sorted_intervals.reserve(paired.size());
    sorted_values.reserve(paired.size());
    for (const auto &p : paired) {
        sorted_intervals.push_back(p.first);
        sorted_values.push_back(p.second);
    }
    track.init_from_data(sorted_intervals, sorted_values, sorted_intervals.empty() ? 0 : sorted_intervals.front().chromid);

    if (func == "avg" || func == "mean") {
        track.register_function(GenomeTrack1D::AVG);
        track.register_function(GenomeTrack1D::SIZE);
    } else if (func == "min")
        track.register_function(GenomeTrack1D::MIN);
    else if (func == "max")
        track.register_function(GenomeTrack1D::MAX);
    else if (func == "sum")
        track.register_function(GenomeTrack1D::SUM);
    else if (func == "lse")
        track.register_function(GenomeTrack1D::LSE);
    else if (func == "stddev" || func == "std") {
        track.register_function(GenomeTrack1D::STDDEV);
        track.register_function(GenomeTrack1D::SIZE);
    } else if (func == "quantile")
        track.register_quantile(10000, 1000, 1000);
    else if (func == "nearest") {
        track.register_function(GenomeTrack1D::NEAREST);
        // with a filter the parts' means are merged through their value counts
        track.register_function(GenomeTrack1D::SIZE);
    } else if (func == "exists")
        track.register_function(GenomeTrack1D::EXISTS);
    else if (func == "size")
        track.register_function(GenomeTrack1D::SIZE);
    else if (func == "first")
        track.register_function(GenomeTrack1D::FIRST);
    else if (func == "last")
        track.register_function(GenomeTrack1D::LAST);
    else if (func == "sample") {
        track.register_function(GenomeTrack1D::SAMPLE);
        // with a filter one part's draw is kept with chance proportional to the part's value count
        track.register_function(GenomeTrack1D::SIZE);
    } else if (func == "first.pos.abs" || func == "first.pos.relative")
        track.register_function(GenomeTrack1D::FIRST_POS);
    else if (func == "last.pos.abs" || func == "last.pos.relative")
        track.register_function(GenomeTrack1D::LAST_POS);
    else if (func == "sample.pos.abs" || func == "sample.pos.relative") {
        track.register_function(GenomeTrack1D::SAMPLE_POS);
        track.register_function(GenomeTrack1D::SIZE);
    } else if (func == "min.pos.abs" || func == "min.pos.relative")
        track.register_function(GenomeTrack1D::MIN_POS);
    else if (func == "max.pos.abs" || func == "max.pos.relative")
        track.register_function(GenomeTrack1D::MAX_POS);
    else if (func == "weighted.sum")
        track.register_function(GenomeTrack1D::WEIGHTED_SUM);
    else
        TGLError("Unsupported value-based virtual track function %s", func.c_str());
}

// The value of a value-based vtrack over interval eval (R misha's ValueVarProcessor, no filter)
static double value_var_unfiltered(GenomeTrackInMemory &track, const std::string &func, const GInterval &eval,
                                   double percentile)
{
    track.read_interval(eval);
    if (func == "avg" || func == "mean") return track.last_avg();
    if (func == "min") return track.last_min();
    if (func == "max") return track.last_max();
    if (func == "sum") return track.last_sum();
    if (func == "weighted.sum") return track.last_weighted_sum();
    if (func == "lse") return track.last_lse();
    if (func == "stddev" || func == "std") return track.last_stddev();
    if (func == "quantile") return track.last_quantile(percentile);
    if (func == "nearest") return track.last_nearest();
    if (func == "exists") return track.last_exists();
    if (func == "size") return track.last_size();
    if (func == "first") return track.last_first();
    if (func == "last") return track.last_last();
    if (func == "sample") return track.last_sample();
    if (func == "first.pos.abs") return track.last_first_pos();
    if (func == "first.pos.relative") return track.last_first_pos() - eval.start;
    if (func == "last.pos.abs") return track.last_last_pos();
    if (func == "last.pos.relative") return track.last_last_pos() - eval.start;
    if (func == "sample.pos.abs") return track.last_sample_pos();
    if (func == "sample.pos.relative") return track.last_sample_pos() - eval.start;
    if (func == "min.pos.abs") return track.last_min_pos();
    if (func == "min.pos.relative") return track.last_min_pos() - eval.start;
    if (func == "max.pos.abs") return track.last_max_pos();
    if (func == "max.pos.relative") return track.last_max_pos() - eval.start;
    TGLError("Unsupported value-based virtual track function %s", func.c_str());
    return std::numeric_limits<double>::quiet_NaN();
}

// A value (pos = false) or its position (pos = true) drawn uniformly from the values of the
// parts, each value once (R misha's TrackVarProcessor::aggregate_sample_with_filter); the track
// computes SIZE
static double value_var_sample(GenomeTrack1D &track, const std::vector<GInterval> &parts, bool pos)
{
    std::vector<double> draws, counts;
    double total = 0;
    for (const auto &part : track.parts_reading_items_once(parts)) {
        track.read_interval(part);
        const double part_n = track.last_size();
        const double draw = pos ? track.last_sample_pos() : (double)track.last_sample();
        if (!(part_n > 0) || std::isnan(draw))
            continue;
        draws.push_back(draw);
        counts.push_back(part_n);
        total += part_n;
    }
    if (draws.empty())
        return std::numeric_limits<double>::quiet_NaN();
    double u = pm::pm_rnd_func() * total;
    for (size_t i = 0; i < draws.size(); ++i) {
        if (u < counts[i])
            return draws[i];
        u -= counts[i];
    }
    return draws.back();
}

// The value of a value-based vtrack over the unmasked parts of an interval whose start after the
// shifts is base_start (R misha's ValueVarProcessor with a filter). "nearest" without a value in
// the parts is NaN here: the caller looks for the nearest unmasked value outside them.
static double value_var_filtered(GenomeTrackInMemory &track, const std::string &func, std::vector<GInterval> parts,
                                 int64_t base_start, double percentile)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    if (func == "weighted.sum") {
        // the parts do not overlap, so each unmasked bp of an interval counts once
        double total = 0;
        bool has_value = false;
        for (const auto &part : parts) {
            track.read_interval(part);
            const double part_wsum = track.last_weighted_sum();
            if (!std::isnan(part_wsum)) {
                total += part_wsum;
                has_value = true;
            }
        }
        return has_value ? total : nan;
    }

    // The counting reductions read each value once even when the filter splits its interval;
    // the other functions (min, max, positions, ...) read the parts as they are
    if (func == "sum" || func == "size" || func == "lse" || func == "avg" || func == "mean" ||
        func == "stddev" || func == "std" || func == "quantile" || func == "nearest")
        parts = track.parts_reading_items_once(parts);

    if (parts.empty())
        return nan;

    double total_sum = 0, sd_mean = 0, sd_m2 = 0;
    int64_t total_size = 0;
    double min_val = nan, max_val = nan, first_val = nan, last_val = nan;
    double min_pos = nan, max_pos = nan, first_pos = nan, last_pos = nan;
    double lse_val = -std::numeric_limits<double>::infinity();
    bool exists = false, has_value = false;

    for (const auto &part : parts) {
        track.read_interval(part);
        const double part_avg = track.last_avg();
        const double part_sum = track.last_sum();
        if (std::isnan(part_avg) || std::isnan(part_sum))
            continue;

        const double part_size = track.last_size();
        const double part_stddev = track.last_stddev();
        if (part_size > 0) {
            // running mean and sum of squared deviations (Chan et al. merge)
            const double prev_n = total_size;
            const double part_m2 = part_size > 1 && !std::isnan(part_stddev) ? part_stddev * part_stddev * (part_size - 1) : 0;
            const double delta = part_avg - sd_mean;
            const double n = prev_n + part_size;
            sd_mean += delta * part_size / n;
            sd_m2 += part_m2 + delta * delta * prev_n * part_size / n;
        }

        total_sum += part_sum;
        total_size += static_cast<int64_t>(part_size);

        const double part_lse = track.last_lse();
        if (!std::isnan(part_lse))
            lse_accumulate(lse_val, part_lse);

        const double part_min = track.last_min();
        const double part_max = track.last_max();
        if (!std::isnan(part_min) && (std::isnan(min_val) || part_min < min_val)) {
            min_val = part_min;
            min_pos = track.last_min_pos();
        }
        if (!std::isnan(part_max) && (std::isnan(max_val) || part_max > max_val)) {
            max_val = part_max;
            max_pos = track.last_max_pos();
        }

        const double part_first = track.last_first();
        const double part_last = track.last_last();
        const double part_first_pos = track.last_first_pos();
        const double part_last_pos = track.last_last_pos();
        if (!std::isnan(part_first) && std::isnan(first_val))
            first_val = part_first;
        if (!std::isnan(part_last))
            last_val = part_last;
        if (!std::isnan(part_first_pos) && std::isnan(first_pos))
            first_pos = part_first_pos;
        if (!std::isnan(part_last_pos))
            last_pos = part_last_pos;

        const double part_exists = track.last_exists();
        exists = exists || (!std::isnan(part_exists) && part_exists != 0);
        has_value = true;
    }

    // No unmasked value: size and exists count none, the other functions have nothing to report
    if (!has_value)
        return func == "size" || func == "exists" ? 0 : nan;

    if (func == "quantile") {
        // as R misha's TrackVarProcessor::aggregate_quantile_with_filter
        StreamPercentiler<float> combined_sp;
        combined_sp.init(g_pymisha->max_data_size(), 100000, 100000);
        for (const auto &part : parts) {
            track.read_interval(part);
            const auto &sp = track.get_percentiler();
            for (const auto *vals : {&sp.lowest_vals(), &sp.samples(), &sp.highest_vals()}) {
                for (float val : *vals) {
                    if (!std::isnan(val))
                        combined_sp.add(val, pm::pm_rnd_func);
                }
            }
        }
        if (combined_sp.stream_size() > 0) {
            bool is_estimated;
            return combined_sp.get_percentile(percentile, is_estimated);
        }
        return nan;
    }

    auto rel = [&](double pos) { return std::isnan(pos) ? nan : pos - base_start; };
    if (func == "avg" || func == "mean" || func == "nearest")
        return total_size > 0 ? total_sum / total_size : nan;
    if (func == "stddev" || func == "std")
        return total_size > 1 ? std::sqrt(sd_m2 / (total_size - 1)) : nan;
    if (func == "sum") return total_sum;
    if (func == "lse") return std::isfinite(lse_val) ? lse_val : nan;
    if (func == "min") return min_val;
    if (func == "max") return max_val;
    if (func == "exists") return exists ? 1.0 : 0.0;
    if (func == "size") return (double)total_size;
    if (func == "first") return first_val;
    if (func == "last") return last_val;
    if (func == "sample") return value_var_sample(track, parts, false);
    if (func == "sample.pos.abs") return value_var_sample(track, parts, true);
    if (func == "sample.pos.relative") return value_var_sample(track, parts, true) - base_start;
    if (func == "first.pos.abs") return first_pos;
    if (func == "first.pos.relative") return rel(first_pos);
    if (func == "last.pos.abs") return last_pos;
    if (func == "last.pos.relative") return rel(last_pos);
    if (func == "min.pos.abs") return min_pos;
    if (func == "min.pos.relative") return rel(min_pos);
    if (func == "max.pos.abs") return max_pos;
    if (func == "max.pos.relative") return rel(max_pos);
    return nan;
}

PyObject *pm_vtrack_compute(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_spec = nullptr;
        PyObject *py_intervals = nullptr;
        PyObject *py_config = nullptr;

        if (!PyArg_ParseTuple(args, "OO|O", &py_spec, &py_intervals, &py_config)) {
            verror("Invalid arguments to pm_vtrack_compute");
        }

        if (!PyDict_Check(py_spec)) {
            TGLError("vtrack specification must be a dict");
        }

        std::vector<GInterval> intervals;
        convert_py_intervals_basic(py_intervals, intervals);

        npy_intp dims[1] = {(npy_intp)intervals.size()};
        PMPY py_vals(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        if (!py_vals) {
            TGLError("Failed to allocate vtrack result array");
        }
        double *out = (double *)PyArray_DATA((PyArrayObject *)*py_vals);
        // Initialize output with NaN so unset entries are NaN (not 0)
        std::fill(out, out + intervals.size(), std::numeric_limits<double>::quiet_NaN());

        // Defaults
        std::string func = obj_to_string(dict_get(py_spec, "func"), "avg");
        std::transform(func.begin(), func.end(), func.begin(), [](unsigned char c) { return std::tolower(c); });

        int64_t sshift = obj_to_int64(dict_get(py_spec, "sshift"), 0);
        int64_t eshift = obj_to_int64(dict_get(py_spec, "eshift"), 0);

        PyObject *py_src = dict_get(py_spec, "src");

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        auto set_nan = [&](size_t idx) {
            out[idx] = std::numeric_limits<double>::quiet_NaN();
        };

        // Sequence-based vtracks
        if (func == "pwm" || func == "pwm.max" || func == "pwm.max.pos" || func == "pwm.count" ||
            func == "pwm.edit_distance" || func == "pwm.edit_distance.pos" ||
            func == "pwm.max.edit_distance" || func == "pwm.n_mutations" ||
            func == "pwm.edit_distance.lse" || func == "pwm.edit_distance.lse.pos" ||
            func == "potts" || func == "potts.max" || func == "potts.max.pos" ||
            func == "potts.count" ||
            func == "kmer.count" || func == "kmer.frac" ||
            func == "masked.count" || func == "masked.frac") {

            if (py_src && py_src != Py_None) {
                TGLError("Virtual track function %s expects src=None (sequence-based)", func.c_str());
            }
            const std::vector<SeqSubstitution> subs = parse_substitutions(py_spec, intervals);

            if (func.rfind("pwm", 0) == 0) {
                PyObject *py_pssm = dict_get(py_spec, "pssm");
                double prior = obj_to_double(dict_get(py_spec, "prior"), 0.01);
                bool bidirect = str_to_bool(dict_get(py_spec, "bidirect"), true);
                bool extend = str_to_bool(dict_get(py_spec, "extend"), true);
                int strand_mode = (int)obj_to_int64(dict_get(py_spec, "strand"), 0);
                double score_thresh = obj_to_double(dict_get(py_spec, "score_thresh"), 0.0);
                if (PyObject *alt = dict_get(py_spec, "score.thresh")) {
                    score_thresh = obj_to_double(alt, score_thresh);
                }

                DnaPSSM pssm;
                if (!parse_pssm(py_pssm, pssm, prior)) {
                    TGLError("pwm functions require a numeric pssm matrix (shape Lx4)");
                }
                pssm.set_bidirect(bidirect);
                char strand = bidirect ? 0 : (char)strand_mode;

                // PWM edit distance functions
                bool is_edit_dist = (func == "pwm.edit_distance" || func == "pwm.edit_distance.pos" ||
                                     func == "pwm.max.edit_distance" || func == "pwm.n_mutations");
                bool is_lse_edit_dist = (func == "pwm.edit_distance.lse" || func == "pwm.edit_distance.lse.pos");

                if (is_edit_dist || is_lse_edit_dist) {
                    int max_edits = (int)obj_to_int64(dict_get(py_spec, "max_edits"), -1);
                    float score_min_f = std::numeric_limits<float>::quiet_NaN();
                    float score_max_f = std::numeric_limits<float>::quiet_NaN();
                    bool found_smin = false, found_smax = false;
                    double smin_d = obj_to_double(dict_get(py_spec, "score_min"), 0.0, &found_smin);
                    double smax_d = obj_to_double(dict_get(py_spec, "score_max"), 0.0, &found_smax);
                    if (found_smin) score_min_f = (float)smin_d;
                    if (found_smax) score_max_f = (float)smax_d;
                    // As PMTrackExpressionVars::build_vtrack_scorer(): the filtered
                    // edit-distance path scores its parts here.
                    const std::string dir_str = obj_to_string(dict_get(py_spec, "direction"), "above");
                    if (dir_str != "above" && dir_str != "below")
                        TGLError("direction parameter must be \"above\" or \"below\"");
                    const bool below = dir_str == "below";

                    GenomeSeqFetch shared_seq;
                    shared_seq.set_seqdir(g_pmdb->groot() + "/seq");

                    if (is_edit_dist) {
                        int max_indels = (int)obj_to_int64(dict_get(py_spec, "max_indels"), 0);

                        PWMEditDistanceScorer::Mode mode = PWMEditDistanceScorer::Mode::MIN_EDITS;
                        if (func == "pwm.edit_distance.pos") mode = PWMEditDistanceScorer::Mode::MIN_EDITS_POSITION;
                        else if (func == "pwm.max.edit_distance") mode = PWMEditDistanceScorer::Mode::PWM_MAX_EDITS;
                        else if (func == "pwm.n_mutations") mode = PWMEditDistanceScorer::Mode::N_MUTATIONS;

                        PWMEditDistanceScorer scorer(pssm, &shared_seq, (float)score_thresh,
                                                     max_edits, extend, strand, mode,
                                                     score_min_f, max_indels, score_max_f,
                                                     below ? PWMEditDistanceScorer::Direction::BELOW
                                                           : PWMEditDistanceScorer::Direction::ABOVE);

                        score_intervals(scorer, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                    } else {
                        PWMLseEditDistanceScorer::Mode mode = PWMLseEditDistanceScorer::Mode::LSE_EDIT_DISTANCE;
                        if (func == "pwm.edit_distance.lse.pos") mode = PWMLseEditDistanceScorer::Mode::LSE_EDIT_DISTANCE_POS;

                        PWMLseEditDistanceScorer scorer(pssm, &shared_seq, (float)score_thresh,
                                                        max_edits, extend, strand, mode,
                                                        score_min_f, score_max_f,
                                                        below ? PWMLseEditDistanceScorer::Direction::BELOW
                                                              : PWMLseEditDistanceScorer::Direction::ABOVE);

                        score_intervals(scorer, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                    }
                    return_py(py_vals);
                }

                // Standard PWM scoring functions (pwm, pwm.max, pwm.max.pos, pwm.count)
                std::vector<float> spat_factor;
                int spat_bin = (int)obj_to_int64(dict_get(py_spec, "spat_bin"), 1);
                if (PyObject *spat = dict_get(py_spec, "spat_factor")) {
                    PMPY arr(PyArray_FROM_OTF(spat, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
                    if (arr && PyArray_NDIM((PyArrayObject *)*arr) == 1) {
                        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
                        spat_factor.resize(n);
                        for (npy_intp i = 0; i < n; ++i) {
                            double v = *(double *)PyArray_GETPTR1((PyArrayObject *)*arr, i);
                            spat_factor[i] = (float)v;
                        }
                    }
                }

                // Same rule as PMTrackExpressionVars::build_vtrack_scorer().
                bool has_smin = false, has_smax = false;
                int64_t spat_min = obj_to_int64(dict_get(py_spec, "spat_min"), 0, &has_smin);
                int64_t spat_max = obj_to_int64(dict_get(py_spec, "spat_max"), DnaPSSM::NO_RANGE_CAP, &has_smax);
                if (has_smin || has_smax) {
                    pssm.set_range((int)spat_min, (int)spat_max);
                }

                PWMScorer::ScoringMode pwm_mode = PWMScorer::TOTAL_LIKELIHOOD;
                if (func == "pwm.max") pwm_mode = PWMScorer::MAX_LIKELIHOOD;
                else if (func == "pwm.max.pos") pwm_mode = PWMScorer::MAX_LIKELIHOOD_POS;
                else if (func == "pwm.count") pwm_mode = PWMScorer::MOTIF_COUNT;

                GenomeSeqFetch shared_seq;
                shared_seq.set_seqdir(g_pmdb->groot() + "/seq");
                PWMScorer scorer(pssm, &shared_seq, extend, pwm_mode, strand, spat_factor, spat_bin, (float)score_thresh);

                if (pwm_mode == PWMScorer::MAX_LIKELIHOOD_POS &&
                    obj_to_int64(dict_get(py_spec, "last_max_score"), 0)) {
                    LastMaxScore<PWMScorer> last_max{scorer};
                    score_intervals(last_max, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                } else {
                    score_intervals(scorer, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                }
                return_py(py_vals);
            }

            if (func.rfind("potts", 0) == 0) {
                // The same parser the expression path and pm_gseq_potts() use,
                // so no two entry points can disagree about what a model means.
                const PMPottsParams pp = PMPottsParams::parse(
                    py_spec, std::string("virtual track function '") + func + "'");

                PottsScorer::ScoringMode potts_mode = PottsScorer::TOTAL_LIKELIHOOD;
                if (func == "potts.max") potts_mode = PottsScorer::MAX_LIKELIHOOD;
                else if (func == "potts.max.pos") potts_mode = PottsScorer::MAX_LIKELIHOOD_POS;
                else if (func == "potts.count") potts_mode = PottsScorer::MOTIF_COUNT;

                GenomeSeqFetch shared_seq;
                shared_seq.set_seqdir(g_pmdb->groot() + "/seq");
                PottsScorer scorer(pp.model, &shared_seq, pp.extend_flag, potts_mode,
                                   pp.bidirect, pp.strand_mode, pp.score_thresh,
                                   pp.max_edits);

                if (potts_mode == PottsScorer::MAX_LIKELIHOOD_POS &&
                    obj_to_int64(dict_get(py_spec, "last_max_score"), 0)) {
                    LastMaxScore<PottsScorer> last_max{scorer};
                    score_intervals(last_max, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                } else {
                    score_intervals(scorer, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                }
                return_py(py_vals);
            }

            if (func == "kmer.count" || func == "kmer.frac") {
                std::string kmer = obj_to_string(dict_get(py_spec, "kmer"), "");
                if (kmer.empty()) {
                    PyObject *params = dict_get(py_spec, "params");
                    kmer = obj_to_string(params, "");
                }
                if (kmer.empty()) {
                    TGLError("kmer functions require 'kmer' parameter");
                }
                bool extend = str_to_bool(dict_get(py_spec, "extend"), true);
                char strand = (char)obj_to_int64(dict_get(py_spec, "strand"), 0);
                KmerCounter::CountMode mode = (func == "kmer.frac") ? KmerCounter::FRACTION : KmerCounter::SUM;
                GenomeSeqFetch shared_seq;
                shared_seq.set_seqdir(g_pmdb->groot() + "/seq");
                KmerCounter counter(kmer, &shared_seq, mode, extend, strand);
                score_intervals(counter, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                return_py(py_vals);
            }

            if (func == "masked.count" || func == "masked.frac") {
                MaskedBpCounter::CountMode mode = (func == "masked.frac") ? MaskedBpCounter::FRACTION : MaskedBpCounter::COUNT;
                GenomeSeqFetch shared_seq;
                shared_seq.set_seqdir(g_pmdb->groot() + "/seq");
                MaskedBpCounter counter(&shared_seq, mode);
                score_intervals(counter, shared_seq, intervals, subs, sshift, eshift, chromkey, out);
                return_py(py_vals);
            }
        }

        // Interval-based vtracks
        if (func == "distance" || func == "distance.center" || func == "distance.edge" ||
            func == "coverage" || func == "neighbor.count") {
            if (!py_src || py_src == Py_None || PyUnicode_Check(py_src)) {
                TGLError("Virtual track function %s expects intervals as source", func.c_str());
            }

            std::vector<GInterval> src_intervals;
            std::vector<double> src_values;
            bool has_values = false;
            bool has_strand = false;
            convert_py_intervals_with_values(py_src, src_intervals, src_values, &has_values, &has_strand);

            std::vector<GInterval> sintervs = src_intervals;
            std::sort(sintervs.begin(), sintervs.end(), interval_cmp_start);

            std::vector<GInterval> eintervs;

            if (func == "coverage") {
                unify_overlaps(sintervs);
            }

            double dist_param = obj_to_double(dict_get(py_spec, "params"), 0.0);
            if (func == "neighbor.count") {
                double dist = dist_param;
                if (dist < 0) TGLError("neighbor.count distance cannot be negative");
                eintervs.clear();
                eintervs.reserve(sintervs.size());
                for (const auto &interv : sintervs) {
                    double expanded_start = (double)interv.start - dist;
                    double expanded_end = (double)interv.end + dist;
                    int64_t new_start = expanded_start < 0.0 ? 0 : (int64_t)std::floor(expanded_start);
                    uint64_t chrom_size = chromkey.get_chrom_size(interv.chromid);
                    double chrom_size_d = (double)chrom_size;
                    if (expanded_end > chrom_size_d) expanded_end = chrom_size_d;
                    int64_t new_end = (int64_t)std::ceil(expanded_end);
                    if (new_end < new_start) new_end = new_start;
                    eintervs.emplace_back(interv.chromid, new_start, new_end, interv.strand);
                }
                std::sort(eintervs.begin(), eintervs.end(), interval_cmp_start);
            }

            double dist_margin = (func == "distance") ? (dist_param * 0.5) : 0.0;

            // Hybrid sequential/binary scan state for start-sorted source intervals
            size_t s_scan_idx = 0;
            int64_t s_last_key = std::numeric_limits<int64_t>::min();
            int s_last_chrom = -1;
            // Hybrid scan state for neighbor.count expanded intervals (start-sorted)
            size_t nc_scan_idx = 0;
            int64_t nc_last_key = std::numeric_limits<int64_t>::min();
            int nc_last_chrom = -1;

            // Adaptive lower_bound on start-sorted sintervs array: sequential when
            // queries advance monotonically, binary search on backward jump.
            auto adaptive_lb_start = [&](const GInterval &eval) -> size_t {
                if (eval.chromid == s_last_chrom && eval.start >= s_last_key) {
                    while (s_scan_idx < sintervs.size() &&
                           interval_cmp_start(sintervs[s_scan_idx], eval)) {
                        ++s_scan_idx;
                    }
                } else {
                    s_scan_idx = (size_t)(std::lower_bound(
                        sintervs.begin(), sintervs.end(), eval,
                        interval_cmp_start) - sintervs.begin());
                }
                s_last_key = eval.start;
                s_last_chrom = eval.chromid;
                return s_scan_idx;
            };

            // Adaptive lower_bound on neighbor.count's start-sorted expanded eintervs
            auto adaptive_lb_nc = [&](const GInterval &eval) -> size_t {
                if (eval.chromid == nc_last_chrom && eval.start >= nc_last_key) {
                    while (nc_scan_idx < eintervs.size() &&
                           interval_cmp_start(eintervs[nc_scan_idx], eval)) {
                        ++nc_scan_idx;
                    }
                } else {
                    nc_scan_idx = (size_t)(std::lower_bound(
                        eintervs.begin(), eintervs.end(), eval,
                        interval_cmp_start) - eintervs.begin());
                }
                nc_last_key = eval.start;
                nc_last_chrom = eval.chromid;
                return nc_scan_idx;
            };

            // Lazy per-chromosome nearest-neighbor index for distance /
            // distance.edge / distance.center. The 4-candidate lower_bound
            // heuristic that previously served these missed enclosing / nested
            // sources, returning a non-nearest interval (even a nonzero distance
            // for a query that overlaps a source). Routing through the same
            // SegmentFinder/NNIterator that gintervals_neighbors uses makes the
            // distance family agree with it exactly (R misha 5.9.1).
            SegmentFinder<GInterval> finder;
            SegmentFinder<GInterval>::NNIterator nn(&finder);
            int finder_chromid = -1;
            auto ensure_finder = [&](int chromid) {
                if (finder_chromid == chromid) return;
                finder_chromid = chromid;
                finder.reset(0, (int64_t)chromkey.get_chrom_size(chromid));
                // sintervs is sorted by (chromid, start, end): find this chrom's
                // contiguous run via lower_bound and insert it.
                GInterval lo_key(chromid, 0, 1, 0);
                auto lo = std::lower_bound(sintervs.begin(), sintervs.end(), lo_key, interval_cmp_start);
                for (auto it = lo; it != sintervs.end() && it->chromid == chromid; ++it)
                    finder.insert(*it);
            };

            for (size_t i = 0; i < intervals.size(); ++i) {
                GInterval eval;
                if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                    // A window shifted off the chromosome has no answer, for
                    // every function (misha 5.12.x), not a count of 0.
                    set_nan(i);
                    continue;
                }

                if (func == "distance") {
                    ensure_finder(eval.chromid);
                    int64_t coord = (eval.start + eval.end) / 2;
                    if (!nn.begin(Segment(coord, coord + 1))) {
                        set_nan(i);
                    } else {
                        // Nearest source to the bin center, measured at its edges
                        // (dist2coord returns 0 / a normalized fraction inside).
                        out[i] = nn->dist2coord(coord, dist_margin);
                    }
                    continue;
                }

                if (func == "distance.center") {
                    ensure_finder(eval.chromid);
                    int64_t coord = (eval.start + eval.end) / 2;
                    double best = std::numeric_limits<double>::quiet_NaN();
                    // distance.center is defined only when the bin center sits
                    // inside a source interval. NNIterator yields candidates by
                    // increasing proximity, so every interval containing the
                    // point precedes the first one that doesn't; among the
                    // containing intervals keep the nearest center. With
                    // non-overlapping sources there is at most one such interval
                    // (historical behavior); overlapping sources resolve to the
                    // nearest center instead of erroring.
                    for (bool ok = nn.begin(Segment(coord, coord + 1)); ok; ok = nn.next()) {
                        const GInterval &iv = *nn;
                        if (coord < iv.start || coord >= iv.end) break;
                        double d = iv.dist2center(coord);
                        if (std::isnan(best) || std::fabs(d) < std::fabs(best)) best = d;
                    }
                    out[i] = best;
                    continue;
                }

                if (func == "distance.edge") {
                    ensure_finder(eval.chromid);
                    if (!nn.begin(Segment(eval.start, eval.end))) {
                        set_nan(i);
                    } else {
                        // Nearest source by edge-to-edge distance; the first
                        // NN result is the closest, matching gintervals_neighbors.
                        out[i] = (double)eval.dist2interv(*nn);
                    }
                    continue;
                }

                if (func == "coverage") {
                    if (eval.range() <= 0) {
                        set_nan(i);
                        continue;
                    }
                    int64_t total_overlap = 0;
                    size_t idx = adaptive_lb_start(eval);
                    auto it = sintervs.begin() + idx;
                    if (it != sintervs.begin()) {
                        auto prev = it - 1;
                        if (prev->chromid == eval.chromid && prev->end > eval.start) {
                            int64_t ov_start = std::max(eval.start, prev->start);
                            int64_t ov_end = std::min(eval.end, prev->end);
                            total_overlap += (ov_end - ov_start);
                        }
                    }
                    for (; it != sintervs.end() && it->chromid == eval.chromid && it->start < eval.end; ++it) {
                        if (it->end > eval.start) {
                            int64_t ov_start = std::max(eval.start, it->start);
                            int64_t ov_end = std::min(eval.end, it->end);
                            total_overlap += (ov_end - ov_start);
                        }
                    }
                    out[i] = (double)total_overlap / (double)eval.range();
                    continue;
                }

                if (func == "neighbor.count") {
                    if (eintervs.empty()) {
                        out[i] = 0.0;
                        continue;
                    }
                    size_t count = 0;
                    size_t idx = adaptive_lb_nc(eval);
                    auto it = eintervs.begin() + idx;
                    if (it != eintervs.begin()) {
                        auto prev = it - 1;
                        while (true) {
                            if (prev->chromid != eval.chromid) break;
                            if (prev->end <= eval.start) break;
                            if (prev->start < eval.end && prev->end > eval.start) ++count;
                            if (prev == eintervs.begin()) break;
                            --prev;
                        }
                    }
                    for (; it != eintervs.end() && it->chromid == eval.chromid && it->start < eval.end; ++it) {
                        if (it->end > eval.start) ++count;
                    }
                    out[i] = (double)count;
                    continue;
                }
            }

            return_py(py_vals);
        }

        // Track-based or value-based vtracks
        if (py_src && py_src != Py_None && !PyUnicode_Check(py_src)) {
            // Value-based: intervals + values, read as R misha reads them (value_var_track)
            std::vector<GInterval> src_intervals;
            std::vector<double> src_values;
            bool has_values = false;
            bool has_strand = false;
            convert_py_intervals_with_values(py_src, src_intervals, src_values, &has_values, &has_strand);

            if (!has_values || src_values.empty()) {
                TGLError("Value-based virtual tracks require a numeric value column");
            }

            PyObject *params_obj = dict_get(py_spec, "params");
            if (params_obj && PyList_Check(params_obj) && PyList_Size(params_obj) > 0) {
                params_obj = PyList_GetItem(params_obj, 0);
            }
            const double percentile = obj_to_double(params_obj, std::numeric_limits<double>::quiet_NaN());

            GenomeTrackInMemory track;
            value_var_track(track, src_intervals, src_values, func);

            // With a filter the caller passes the unmasked parts of each interval, consecutive and
            // in order, with the interval each belongs to (part_owner) and its start after the
            // shifts (part_base); the result of an interval goes to its first part.
            PyObject *py_owner = dict_get(py_spec, "part_owner");
            if (py_owner && py_owner != Py_None) {
                PMPY owner(PyArray_FROM_OTF(py_owner, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
                PMPY base(PyArray_FROM_OTF(dict_get(py_spec, "part_base"), NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
                if (!owner || !base || PyArray_SIZE((PyArrayObject *)*owner) != (npy_intp)intervals.size() ||
                    PyArray_SIZE((PyArrayObject *)*base) != (npy_intp)intervals.size())
                    TGLError("pm_vtrack_compute: part_owner and part_base must hold one entry per part");
                const int64_t *o = (const int64_t *)PyArray_DATA((PyArrayObject *)*owner);
                const int64_t *b = (const int64_t *)PyArray_DATA((PyArrayObject *)*base);
                std::vector<GInterval> parts;
                for (size_t i = 0; i < intervals.size(); ) {
                    size_t j = i;
                    parts.clear();
                    for (; j < intervals.size() && o[j] == o[i]; ++j)
                        parts.push_back(intervals[j]);
                    out[i] = value_var_filtered(track, func, parts, b[i], percentile);
                    i = j;
                }
                return_py(py_vals);
            }

            for (size_t i = 0; i < intervals.size(); ++i) {
                GInterval eval;
                if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                    set_nan(i);
                    continue;
                }
                out[i] = value_var_unfiltered(track, func, eval, percentile);
            }

            return_py(py_vals);
        }

        if (!py_src || py_src == Py_None) {
            TGLError("Virtual track source is required for function %s", func.c_str());
        }

        std::string track_name = obj_to_string(py_src, "");
        if (track_name.empty()) {
            TGLError("Virtual track source must be a track name");
        }

        std::string track_path = g_pmdb->track_path(track_name);
        GenomeTrack::Type track_type = GenomeTrack::get_type(track_path.c_str(), chromkey, false);
        std::unique_ptr<GenomeTrack> track;
        GenomeTrackFixedBin *fixed_bin = nullptr;
        GenomeTrackSparse *sparse = nullptr;
        GenomeTrackArray *array = nullptr;

        if (track_type == GenomeTrack::FIXED_BIN) {
            track = std::make_unique<GenomeTrackFixedBin>();
            fixed_bin = static_cast<GenomeTrackFixedBin *>(track.get());
        } else if (track_type == GenomeTrack::SPARSE) {
            track = std::make_unique<GenomeTrackSparse>();
            sparse = static_cast<GenomeTrackSparse *>(track.get());
        } else if (track_type == GenomeTrack::ARRAYS) {
            // the value of an item is its columns reduced by the vtrack's slice (gvtrack.array.slice),
            // by default their average
            track = std::make_unique<GenomeTrackArray>();
            array = static_cast<GenomeTrackArray *>(track.get());
            PMTrackExpressionVars::configure_array_slice(*array, py_spec);
        } else {
            TGLError("Track type '%s' not supported for virtual track %s",
                     GenomeTrack::TYPE_NAMES[track_type], track_name.c_str());
        }

        GenomeTrack1D *track1d = dynamic_cast<GenomeTrack1D *>(track.get());
        if (!track1d) {
            TGLError("Virtual track %s is not a 1D track", track_name.c_str());
        }

        if (func == "stddev" || func == "std") track1d->register_function(GenomeTrack1D::STDDEV);
        if (func == "quantile") track1d->register_quantile(10000, 1000, 1000);
        if (func == "exists") track1d->register_function(GenomeTrack1D::EXISTS);
        if (func == "size") track1d->register_function(GenomeTrack1D::SIZE);
        if (func == "sample") track1d->register_function(GenomeTrack1D::SAMPLE);
        if (func == "sample.pos.abs" || func == "sample.pos.relative") track1d->register_function(GenomeTrack1D::SAMPLE_POS);
        if (func == "first") track1d->register_function(GenomeTrack1D::FIRST);
        if (func == "first.pos.abs" || func == "first.pos.relative") track1d->register_function(GenomeTrack1D::FIRST_POS);
        if (func == "last") track1d->register_function(GenomeTrack1D::LAST);
        if (func == "last.pos.abs" || func == "last.pos.relative") track1d->register_function(GenomeTrack1D::LAST_POS);
        if (func == "max.pos.abs" || func == "max.pos.relative") track1d->register_function(GenomeTrack1D::MAX_POS);
        if (func == "min.pos.abs" || func == "min.pos.relative") track1d->register_function(GenomeTrack1D::MIN_POS);
        if (func == "weighted.sum") track1d->register_function(GenomeTrack1D::WEIGHTED_SUM);
        // A sparse or array track computes lse in calc_vals, in float as R misha
        if ((array || sparse) && func == "lse") track1d->register_function(GenomeTrack1D::LSE);

        // Register the primary reducer on dense tracks so read_interval can take
        // the incremental sliding-window fast path (classify_fast_path_mode).
        // Dense-only: the sliding machinery lives in GenomeTrackFixedBin; sparse
        // keeps its existing scan path.
        if (fixed_bin) {
            if (func == "avg" || func == "mean") track1d->register_function(GenomeTrack1D::AVG);
            if (func == "sum") track1d->register_function(GenomeTrack1D::SUM);
            if (func == "nearest") track1d->register_function(GenomeTrack1D::NEAREST);
            if (func == "lse") track1d->register_function(GenomeTrack1D::LSE);
        }

        const bool sparse_fast_reduce =
            sparse &&
            (func == "avg" || func == "mean" || func == "sum" ||
             func == "min" || func == "max" || func == "size" || func == "exists");

        int cur_chromid = -1;
        bool cur_chromid_valid = false;
        const std::vector<GInterval> *sp_intervals_ptr = nullptr;
        const std::vector<float> *sp_vals_ptr = nullptr;
        size_t sp_scan_idx = 0;
        int64_t sp_last_start = std::numeric_limits<int64_t>::min();
        bool sp_scan_ready = false;

        for (size_t i = 0; i < intervals.size(); ++i) {
            GInterval eval;
            if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                set_nan(i);
                continue;
            }

            if (cur_chromid != eval.chromid) {
                std::string chrom_file = g_pmdb->chrom_files(track_path).filename(eval.chromid);
                std::string full_path = track_path + "/" + chrom_file;
                // Indexed tracks have no per-chrom file; init_read reads through
                // track.idx. Gating on the per-chrom file made value-based
                // vtracks over indexed source tracks return all-NaN.
                const bool indexed =
                    access((track_path + "/track.idx").c_str(), F_OK) == 0;
                if (!indexed && access(full_path.c_str(), F_OK) != 0) {
                    cur_chromid = eval.chromid;
                    cur_chromid_valid = false;
                } else {
                    if (fixed_bin) {
                        fixed_bin->init_read(full_path.c_str(), eval.chromid);
                    } else if (array) {
                        array->init_read(full_path.c_str(), eval.chromid);
                    } else if (sparse) {
                        sparse->init_read(full_path.c_str(), eval.chromid);
                        if (sparse_fast_reduce) {
                            sp_intervals_ptr = &sparse->get_intervals();
                            sp_vals_ptr = &sparse->get_vals();
                            sp_scan_idx = 0;
                            sp_last_start = std::numeric_limits<int64_t>::min();
                            sp_scan_ready = false;
                        }
                    }
                    cur_chromid = eval.chromid;
                    cur_chromid_valid = true;
                }
            }

            if (!cur_chromid_valid) {
                set_nan(i);
                continue;
            }

            if (sparse_fast_reduce) {
                const auto &sp_intervals = *sp_intervals_ptr;
                const auto &sp_vals = *sp_vals_ptr;
                if (sp_intervals.empty()) {
                    if (func == "size" || func == "exists")
                        out[i] = 0.0;
                    else
                        set_nan(i);
                    continue;
                }

                if (!sp_scan_ready || eval.start < sp_last_start) {
                    size_t lo = 0;
                    size_t hi = sp_intervals.size();
                    while (lo < hi) {
                        size_t mid = lo + (hi - lo) / 2;
                        if (sp_intervals[mid].end <= eval.start)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }
                    sp_scan_idx = lo;
                    sp_scan_ready = true;
                } else {
                    while (sp_scan_idx < sp_intervals.size() &&
                           sp_intervals[sp_scan_idx].end <= eval.start) {
                        ++sp_scan_idx;
                    }
                }
                sp_last_start = eval.start;

                double sum = 0.0;
                double mn = std::numeric_limits<double>::infinity();
                double mx = -std::numeric_limits<double>::infinity();
                size_t n_valid = 0;

                for (size_t j = sp_scan_idx; j < sp_intervals.size(); ++j) {
                    const GInterval &cur = sp_intervals[j];
                    if (cur.start >= eval.end)
                        break;
                    if (cur.end <= eval.start)
                        continue;
                    float v = sp_vals[j];
                    if (std::isnan(v))
                        continue;
                    const double dv = static_cast<double>(v);
                    sum += dv;
                    if (dv < mn) mn = dv;
                    if (dv > mx) mx = dv;
                    ++n_valid;
                }

                if (func == "exists") {
                    out[i] = n_valid > 0 ? 1.0 : 0.0;
                } else if (func == "size") {
                    out[i] = static_cast<double>(n_valid);
                } else if (n_valid == 0) {
                    set_nan(i);
                } else if (func == "avg" || func == "mean") {
                    out[i] = sum / static_cast<double>(n_valid);
                } else if (func == "sum") {
                    out[i] = sum;
                } else if (func == "min") {
                    out[i] = mn;
                } else if (func == "max") {
                    out[i] = mx;
                } else {
                    set_nan(i);
                }
                continue;
            }

            track1d->read_interval(eval);

            if (func == "avg" || func == "mean") {
                out[i] = track1d->last_avg();
            } else if (func == "sum") {
                out[i] = track1d->last_sum();
            } else if (func == "weighted.sum") {
                out[i] = track1d->last_weighted_sum();
            } else if (func == "lse") {
                out[i] = track1d->last_lse();
            } else if (func == "min") {
                out[i] = track1d->last_min();
            } else if (func == "max") {
                out[i] = track1d->last_max();
            } else if (func == "nearest") {
                out[i] = track1d->last_nearest();
            } else if (func == "stddev" || func == "std") {
                out[i] = track1d->last_stddev();
            } else if (func == "quantile") {
                PyObject *params_obj2 = dict_get(py_spec, "params");
                if (params_obj2 && PyList_Check(params_obj2) && PyList_Size(params_obj2) > 0) {
                    params_obj2 = PyList_GetItem(params_obj2, 0);
                }
                double percentile = obj_to_double(params_obj2, std::numeric_limits<double>::quiet_NaN());
                out[i] = track1d->last_quantile(percentile);
            } else if (func == "exists") {
                out[i] = track1d->last_exists();
            } else if (func == "size") {
                out[i] = track1d->last_size();
            } else if (func == "sample") {
                out[i] = track1d->last_sample();
            } else if (func == "sample.pos.abs") {
                out[i] = track1d->last_sample_pos();
            } else if (func == "sample.pos.relative") {
                out[i] = track1d->last_sample_pos() - eval.start;
            } else if (func == "first") {
                out[i] = track1d->last_first();
            } else if (func == "first.pos.abs") {
                out[i] = track1d->last_first_pos();
            } else if (func == "first.pos.relative") {
                out[i] = track1d->last_first_pos() - eval.start;
            } else if (func == "last") {
                out[i] = track1d->last_last();
            } else if (func == "last.pos.abs") {
                out[i] = track1d->last_last_pos();
            } else if (func == "last.pos.relative") {
                out[i] = track1d->last_last_pos() - eval.start;
            } else if (func == "max.pos.abs") {
                out[i] = track1d->last_max_pos();
            } else if (func == "max.pos.relative") {
                out[i] = track1d->last_max_pos() - eval.start;
            } else if (func == "min.pos.abs") {
                out[i] = track1d->last_min_pos();
            } else if (func == "min.pos.relative") {
                out[i] = track1d->last_min_pos() - eval.start;
            } else {
                TGLError("Unsupported virtual track function %s", func.c_str());
            }
        }

        return_py(py_vals);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}
