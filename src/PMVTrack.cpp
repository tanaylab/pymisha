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

#include "pymisha.h"
#include "PMDb.h"
#include "GInterval.h"
#include "GenomeTrack.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeTrackSparse.h"
#include "GenomeSeqFetch.h"
#include "PWMScorer.h"
#include "KmerCounter.h"
#include "MaskedBpCounter.h"
#include "pmutils.h"

extern PyObject *s_pm_err;

namespace {

struct IntervalValue {
    GInterval interval;
    double value;
};

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

        int chromid = -1;
        if (PyUnicode_Check(chrom_val)) {
            const char *chrom_name = PyUnicode_AsUTF8(chrom_val);
            chromid = chromkey.chrom2id(chrom_name);
            if (chromid < 0) {
                TGLError("Unknown chromosome: %s", chrom_name);
            }
        } else if (PyNumber_Check(chrom_val)) {
            PMPY py_long(PyNumber_Long(chrom_val), true);
            if (!py_long) {
                PyErr_Clear();
                TGLError("Failed to convert chromosome to integer at index %ld", (long)i);
            }
            long chrom_num = PyLong_AsLong(py_long);
            std::string chrom_str = std::to_string(chrom_num);
            chromid = chromkey.chrom2id(chrom_str.c_str());
            if (chromid < 0) {
                chrom_str = "chr" + std::to_string(chrom_num);
                chromid = chromkey.chrom2id(chrom_str.c_str());
            }
            if (chromid < 0) {
                TGLError("Unknown chromosome: %ld", chrom_num);
            }
        } else {
            TGLError("Invalid chromosome type at index %ld", (long)i);
        }

        int64_t start = PyLong_AsLongLong(start_val);
        int64_t end = PyLong_AsLongLong(end_val);

        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("Invalid start/end values at index %ld", (long)i);
        }

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

        int chromid = -1;
        if (PyUnicode_Check(chrom_val)) {
            const char *chrom_name = PyUnicode_AsUTF8(chrom_val);
            chromid = chromkey.chrom2id(chrom_name);
            if (chromid < 0) TGLError("Unknown chromosome: %s", chrom_name);
        } else if (PyNumber_Check(chrom_val)) {
            PMPY py_long(PyNumber_Long(chrom_val), true);
            long chrom_num = PyLong_AsLong(py_long);
            std::string chrom_str = std::to_string(chrom_num);
            chromid = chromkey.chrom2id(chrom_str.c_str());
            if (chromid < 0) {
                chrom_str = "chr" + std::to_string(chrom_num);
                chromid = chromkey.chrom2id(chrom_str.c_str());
            }
            if (chromid < 0) TGLError("Unknown chromosome: %ld", chrom_num);
        } else {
            TGLError("Invalid chromosome type at index %ld", (long)i);
        }

        int64_t start = PyLong_AsLongLong(start_val);
        int64_t end = PyLong_AsLongLong(end_val);

        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("Invalid start/end values at index %ld", (long)i);
        }

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
            func == "kmer.count" || func == "kmer.frac" ||
            func == "masked.count" || func == "masked.frac") {

            if (py_src && py_src != Py_None) {
                TGLError("Virtual track function %s expects src=None (sequence-based)", func.c_str());
            }

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

                std::vector<float> spat_factor;
                int spat_bin = (int)obj_to_int64(dict_get(py_spec, "spat_bin"), 1);
                bool has_spat = false;
                if (PyObject *spat = dict_get(py_spec, "spat_factor")) {
                    PMPY arr(PyArray_FROM_OTF(spat, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
                    if (arr && PyArray_NDIM((PyArrayObject *)*arr) == 1) {
                        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
                        spat_factor.resize(n);
                        for (npy_intp i = 0; i < n; ++i) {
                            double v = *(double *)PyArray_GETPTR1((PyArrayObject *)*arr, i);
                            spat_factor[i] = (float)v;
                        }
                        has_spat = true;
                    }
                }

                int64_t spat_min = obj_to_int64(dict_get(py_spec, "spat_min"), 0);
                int64_t spat_max = obj_to_int64(dict_get(py_spec, "spat_max"), 0);

                DnaPSSM pssm;
                if (!parse_pssm(py_pssm, pssm, prior)) {
                    TGLError("pwm functions require a numeric pssm matrix (shape Lx4)");
                }
                pssm.set_bidirect(bidirect);
                if (has_spat && spat_max > spat_min) {
                    pssm.set_range((int)spat_min, (int)spat_max);
                }

                PWMScorer::ScoringMode mode = PWMScorer::TOTAL_LIKELIHOOD;
                if (func == "pwm.max") mode = PWMScorer::MAX_LIKELIHOOD;
                else if (func == "pwm.max.pos") mode = PWMScorer::MAX_LIKELIHOOD_POS;
                else if (func == "pwm.count") mode = PWMScorer::MOTIF_COUNT;

                char strand = bidirect ? 0 : (char)strand_mode;
                GenomeSeqFetch shared_seq;
                shared_seq.set_seqdir(g_pmdb->groot() + "/seq");
                PWMScorer scorer(pssm, &shared_seq, extend, mode, strand, spat_factor, spat_bin, (float)score_thresh);

                for (size_t i = 0; i < intervals.size(); ++i) {
                    GInterval eval;
                    if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                        set_nan(i);
                        continue;
                    }
                    out[i] = scorer.score_interval(eval, chromkey);
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
                for (size_t i = 0; i < intervals.size(); ++i) {
                    GInterval eval;
                    if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                        set_nan(i);
                        continue;
                    }
                    out[i] = counter.score_interval(eval, chromkey);
                }
                return_py(py_vals);
            }

            if (func == "masked.count" || func == "masked.frac") {
                MaskedBpCounter::CountMode mode = (func == "masked.frac") ? MaskedBpCounter::FRACTION : MaskedBpCounter::COUNT;
                GenomeSeqFetch shared_seq;
                shared_seq.set_seqdir(g_pmdb->groot() + "/seq");
                MaskedBpCounter counter(&shared_seq, mode);
                for (size_t i = 0; i < intervals.size(); ++i) {
                    GInterval eval;
                    if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                        set_nan(i);
                        continue;
                    }
                    out[i] = counter.score_interval(eval, chromkey);
                }
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

            if (func == "distance" || func == "distance.edge") {
                eintervs = sintervs;
                std::sort(eintervs.begin(), eintervs.end(), interval_cmp_end);
            }

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

            for (size_t i = 0; i < intervals.size(); ++i) {
                GInterval eval;
                if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                    if (func == "neighbor.count" || func == "coverage") {
                        out[i] = 0.0;
                    } else {
                        set_nan(i);
                    }
                    continue;
                }

                if (func == "distance") {
                    int64_t coord = (eval.start + eval.end) / 2;
                    double min_dist = std::numeric_limits<double>::max();
                    auto it = std::lower_bound(sintervs.begin(), sintervs.end(), eval, interval_cmp_start);
                    if (it != sintervs.end() && it->chromid == eval.chromid) {
                        min_dist = it->dist2coord(coord, dist_margin);
                    }
                    if (it != sintervs.begin() && (it - 1)->chromid == eval.chromid) {
                        double d = (it - 1)->dist2coord(coord, dist_margin);
                        if (std::fabs(d) < std::fabs(min_dist)) min_dist = d;
                    }

                    if (min_dist == std::numeric_limits<double>::max()) {
                        set_nan(i);
                    } else {
                        auto it2 = std::lower_bound(eintervs.begin(), eintervs.end(), eval, interval_cmp_end);
                        if (it2 != eintervs.end() && it2->chromid == eval.chromid) {
                            double d = it2->dist2coord(coord, dist_margin);
                            if (std::fabs(d) < std::fabs(min_dist)) min_dist = d;
                        }
                        if (it2 != eintervs.begin() && (it2 - 1)->chromid == eval.chromid) {
                            double d = (it2 - 1)->dist2coord(coord, dist_margin);
                            if (std::fabs(d) < std::fabs(min_dist)) min_dist = d;
                        }
                        out[i] = min_dist;
                    }
                    continue;
                }

                if (func == "distance.center") {
                    int64_t coord = (eval.start + eval.end) / 2;
                    auto it = std::lower_bound(sintervs.begin(), sintervs.end(), eval, interval_cmp_start);
                    double dist = std::numeric_limits<double>::quiet_NaN();
                    if (it != sintervs.begin()) {
                        auto prev = it - 1;
                        if (prev->chromid == eval.chromid && coord >= prev->start && coord < prev->end) {
                            dist = prev->dist2center(coord);
                        }
                    }
                    if (std::isnan(dist) && it != sintervs.end() && it->chromid == eval.chromid) {
                        if (coord >= it->start && coord < it->end) {
                            dist = it->dist2center(coord);
                        }
                    }
                    out[i] = dist;
                    continue;
                }

                if (func == "distance.edge") {
                    int64_t best = std::numeric_limits<int64_t>::max();
                    auto it = std::lower_bound(sintervs.begin(), sintervs.end(), eval, interval_cmp_start);
                    if (it != sintervs.end() && it->chromid == eval.chromid) {
                        int64_t d = eval.dist2interv(*it);
                        if (llabs(d) < llabs(best) || best == std::numeric_limits<int64_t>::max())
                            best = d;
                    }
                    if (it != sintervs.begin() && (it - 1)->chromid == eval.chromid) {
                        int64_t d = eval.dist2interv(*(it - 1));
                        if (llabs(d) < llabs(best)) best = d;
                    }
                    auto it2 = std::lower_bound(eintervs.begin(), eintervs.end(), eval, interval_cmp_end);
                    if (it2 != eintervs.end() && it2->chromid == eval.chromid) {
                        int64_t d = eval.dist2interv(*it2);
                        if (llabs(d) < llabs(best)) best = d;
                    }
                    if (it2 != eintervs.begin() && (it2 - 1)->chromid == eval.chromid) {
                        int64_t d = eval.dist2interv(*(it2 - 1));
                        if (llabs(d) < llabs(best)) best = d;
                    }
                    if (best == std::numeric_limits<int64_t>::max()) {
                        set_nan(i);
                    } else {
                        out[i] = (double)best;
                    }
                    continue;
                }

                if (func == "coverage") {
                    if (eval.range() <= 0) {
                        set_nan(i);
                        continue;
                    }
                    int64_t total_overlap = 0;
                    auto it = std::lower_bound(sintervs.begin(), sintervs.end(), eval, interval_cmp_start);
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
                    auto it = std::lower_bound(eintervs.begin(), eintervs.end(), eval, interval_cmp_start);
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
            // Value-based: intervals + values
            std::vector<GInterval> src_intervals;
            std::vector<double> src_values;
            bool has_values = false;
            bool has_strand = false;
            convert_py_intervals_with_values(py_src, src_intervals, src_values, &has_values, &has_strand);

            if (!has_values || src_values.empty()) {
                TGLError("Value-based virtual tracks require a numeric value column");
            }

            std::vector<IntervalValue> entries;
            entries.reserve(src_intervals.size());
            for (size_t i = 0; i < src_intervals.size(); ++i) {
                entries.push_back({src_intervals[i], src_values[i]});
            }
            std::sort(entries.begin(), entries.end(), [](const IntervalValue &a, const IntervalValue &b) {
                if (a.interval.chromid != b.interval.chromid) return a.interval.chromid < b.interval.chromid;
                if (a.interval.start != b.interval.start) return a.interval.start < b.interval.start;
                return a.interval.end < b.interval.end;
            });

            PyObject *params_obj = dict_get(py_spec, "params");
            if (params_obj && PyList_Check(params_obj) && PyList_Size(params_obj) > 0) {
                params_obj = PyList_GetItem(params_obj, 0);
            }
            double percentile = obj_to_double(params_obj, std::numeric_limits<double>::quiet_NaN());

            for (size_t i = 0; i < intervals.size(); ++i) {
                GInterval eval;
                if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                    set_nan(i);
                    continue;
                }
                std::vector<double> vals;
                auto it = std::lower_bound(entries.begin(), entries.end(), eval, [](const IntervalValue &a, const GInterval &b) {
                    if (a.interval.chromid != b.chromid) return a.interval.chromid < b.chromid;
                    if (a.interval.start != b.start) return a.interval.start < b.start;
                    return a.interval.end < b.end;
                });
                if (it != entries.begin()) --it;
                for (; it != entries.end() && it->interval.chromid == eval.chromid && it->interval.start < eval.end; ++it) {
                    if (it->interval.end > eval.start) {
                        if (!std::isnan(it->value)) vals.push_back(it->value);
                    }
                }
                if (vals.empty()) {
                    set_nan(i);
                    continue;
                }

                if (func == "avg" || func == "mean") {
                    double sum = 0.0;
                    for (double v : vals) sum += v;
                    out[i] = sum / vals.size();
                } else if (func == "sum") {
                    double sum = 0.0;
                    for (double v : vals) sum += v;
                    out[i] = sum;
                } else if (func == "min") {
                    out[i] = *std::min_element(vals.begin(), vals.end());
                } else if (func == "max") {
                    out[i] = *std::max_element(vals.begin(), vals.end());
                } else if (func == "stddev" || func == "std") {
                    if (vals.size() < 2) {
                        out[i] = std::numeric_limits<double>::quiet_NaN();
                    } else {
                        double mean = 0.0;
                        for (double v : vals) mean += v;
                        mean /= vals.size();
                        double ss = 0.0;
                        for (double v : vals) ss += (v - mean) * (v - mean);
                        out[i] = std::sqrt(ss / (vals.size() - 1));
                    }
                } else if (func == "quantile") {
                    out[i] = compute_quantile(vals, percentile);
                } else if (func == "exists") {
                    out[i] = vals.empty() ? 0.0 : 1.0;
                } else if (func == "size") {
                    out[i] = (double)vals.size();
                } else if (func == "first") {
                    out[i] = vals.front();
                } else if (func == "last") {
                    out[i] = vals.back();
                } else if (func == "sample") {
                    size_t idx = (size_t)(pm::pm_rnd_func() * vals.size());
                    if (idx >= vals.size()) idx = vals.size() - 1;
                    out[i] = vals[idx];
                } else if (func == "lse") {
                    double m = *std::max_element(vals.begin(), vals.end());
                    if ((std::isinf(m) && m < 0)) {
                        out[i] = m;
                    } else {
                        double sum_exp = 0.0;
                        for (double v : vals) sum_exp += std::exp(v - m);
                        out[i] = m + std::log(sum_exp);
                    }
                } else {
                    TGLError("Unsupported value-based virtual track function %s", func.c_str());
                }
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

        if (track_type == GenomeTrack::FIXED_BIN) {
            track = std::make_unique<GenomeTrackFixedBin>();
            fixed_bin = static_cast<GenomeTrackFixedBin *>(track.get());
        } else if (track_type == GenomeTrack::SPARSE) {
            track = std::make_unique<GenomeTrackSparse>();
            sparse = static_cast<GenomeTrackSparse *>(track.get());
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

        // LSE (log-sum-exp) for physical tracks: read raw values and compute directly
        if (func == "lse") {
            int cur_chromid = -1;
            bool cur_chromid_valid = false;

            for (size_t i = 0; i < intervals.size(); ++i) {
                GInterval eval;
                if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                    set_nan(i);
                    continue;
                }

                if (cur_chromid != eval.chromid) {
                    std::string chrom_file = GenomeTrack::find_existing_1d_filename(chromkey, track_path, eval.chromid);
                    std::string full_path = track_path + "/" + chrom_file;
                    if (access(full_path.c_str(), F_OK) != 0) {
                        cur_chromid = eval.chromid;
                        cur_chromid_valid = false;
                    } else {
                        if (fixed_bin) {
                            fixed_bin->init_read(full_path.c_str(), eval.chromid);
                        } else if (sparse) {
                            sparse->init_read(full_path.c_str(), eval.chromid);
                        }
                        cur_chromid = eval.chromid;
                        cur_chromid_valid = true;
                    }
                }

                if (!cur_chromid_valid) {
                    set_nan(i);
                    continue;
                }

                // Collect non-NaN values from the interval
                std::vector<double> vals;
                if (fixed_bin) {
                    unsigned bin_size = fixed_bin->get_bin_size();
                    int64_t sbin = (int64_t)(eval.start / bin_size);
                    int64_t ebin = (int64_t)std::ceil(eval.end / (double)bin_size);
                    std::vector<float> bin_vals;
                    int64_t bins_read = fixed_bin->read_bins_bulk(sbin, ebin - sbin, bin_vals);
                    vals.reserve(bins_read);
                    for (int64_t j = 0; j < bins_read; ++j) {
                        if (!std::isnan(bin_vals[j])) vals.push_back((double)bin_vals[j]);
                    }
                } else if (sparse) {
                    const std::vector<GInterval> &sp_intervals = sparse->get_intervals();
                    const std::vector<float> &sp_vals = sparse->get_vals();
                    for (size_t j = 0; j < sp_intervals.size(); ++j) {
                        if (sp_intervals[j].start >= eval.end) break;
                        if (sp_intervals[j].end > eval.start && !std::isnan(sp_vals[j])) {
                            vals.push_back((double)sp_vals[j]);
                        }
                    }
                }

                if (vals.empty()) {
                    set_nan(i);
                } else {
                    double m = *std::max_element(vals.begin(), vals.end());
                    if ((std::isinf(m) && m < 0)) {
                        out[i] = m;
                    } else {
                        double sum_exp = 0.0;
                        for (double v : vals) sum_exp += std::exp(v - m);
                        out[i] = m + std::log(sum_exp);
                    }
                }
            }

            return_py(py_vals);
        }

        int cur_chromid = -1;
        bool cur_chromid_valid = false;

        for (size_t i = 0; i < intervals.size(); ++i) {
            GInterval eval;
            if (!apply_shift(intervals[i], sshift, eshift, chromkey, eval)) {
                set_nan(i);
                continue;
            }

            if (cur_chromid != eval.chromid) {
                std::string chrom_file = GenomeTrack::find_existing_1d_filename(chromkey, track_path, eval.chromid);
                std::string full_path = track_path + "/" + chrom_file;
                if (access(full_path.c_str(), F_OK) != 0) {
                    cur_chromid = eval.chromid;
                    cur_chromid_valid = false;
                } else {
                    if (fixed_bin) {
                        fixed_bin->init_read(full_path.c_str(), eval.chromid);
                    } else if (sparse) {
                        sparse->init_read(full_path.c_str(), eval.chromid);
                    }
                    cur_chromid = eval.chromid;
                    cur_chromid_valid = true;
                }
            }

            if (!cur_chromid_valid) {
                set_nan(i);
                continue;
            }

            track1d->read_interval(eval);

            if (func == "avg" || func == "mean") {
                out[i] = track1d->last_avg();
            } else if (func == "sum") {
                out[i] = track1d->last_sum();
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
