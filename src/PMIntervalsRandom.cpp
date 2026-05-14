/*
 * PMIntervalsRandom.cpp
 *
 * C++ port of R misha's C_grandom_genome (GenomeUtilsR.cpp) for pymisha.
 *
 * Generates `n` random non-overlapping interval samples of fixed `size` from a
 * (per-chrom) genome description, optionally avoiding a filter set.
 *
 * NOTE on RNG divergence: R uses unif_rand() (Mersenne Twister with R-specific
 * tempering). PyMisha's existing Python fallback uses numpy's default_rng
 * (PCG64). This C++ path uses std::mt19937_64. None of the three are
 * bit-identical to each other. All three are uniformly distributed; tests
 * compare statistical equivalence (per-chrom counts, mean positions) rather
 * than exact equality.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "pymisha.h"
#include "PMDb.h"
#include "GenomeChromKey.h"

using namespace std;

namespace {

struct ValidSegment {
    int     chromid;
    int64_t start;       // inclusive start position
    int64_t length;      // number of valid start positions = (seg_end - seg_start)
    double  cum_prob;    // cumulative probability up to and including this segment
};

// Sort filter pairs (start, end) by start ascending.
struct StartEnd {
    int64_t start;
    int64_t end;
};

static bool startend_less(const StartEnd &a, const StartEnd &b) {
    return a.start < b.start || (a.start == b.start && a.end < b.end);
}

// Parse a (chrom, start, end) pymisha-format DataFrame (the list returned by
// _df2pymisha). The chrom column may be a numpy object array (strings) or a
// categorical pair [categories, codes]. Returns vectors aligned by row.
//
// chrom names are mapped to chromid via the database's GenomeChromKey. Rows
// referencing unknown chromosomes throw.
static void parse_intervals_df(PyObject *py_df,
                               const GenomeChromKey &chromkey,
                               vector<int> &chromids,
                               vector<int64_t> &starts,
                               vector<int64_t> &ends,
                               bool skip_unknown_chroms,
                               const char *what)
{
    if (!py_df || py_df == Py_None) {
        return;
    }
    if (!PyList_Check(py_df) || PyList_Size(py_df) < 4) {
        TGLError("%s must be a pymisha DataFrame (list format)", what);
    }

    PyObject *colnames = PyList_GetItem(py_df, 0);
    if (!colnames || !PyArray_Check(colnames)) {
        TGLError("%s has no column names array", what);
    }

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

    if (chrom_idx < 0 || start_idx < 0 || end_idx < 0) {
        TGLError("%s must have 'chrom', 'start', 'end' columns", what);
    }

    PyObject *chrom_col = PyList_GetItem(py_df, chrom_idx + 1);
    PyObject *start_col = PyList_GetItem(py_df, start_idx + 1);
    PyObject *end_col = PyList_GetItem(py_df, end_idx + 1);

    // Decode chrom column. Either a 1D object/str numpy array, or a
    // [categories_array, codes_array] list (pandas Categorical).
    auto get_chrom_name = [&](Py_ssize_t i, string &out) -> bool {
        if (PyList_Check(chrom_col) && PyList_Size(chrom_col) == 2) {
            PyObject *cats = PyList_GetItem(chrom_col, 0);
            PyObject *codes = PyList_GetItem(chrom_col, 1);
            long code = *(long *)PyArray_GETPTR1((PyArrayObject *)codes, i);
            if (code < 0) {
                return false;  // NA chrom -> skip
            }
            PyObject *name = *(PyObject **)PyArray_GETPTR1((PyArrayObject *)cats, code);
            if (!name || !PyUnicode_Check(name)) {
                TGLError("%s: invalid chrom category at row %lld", what, (long long)i);
            }
            out = PyUnicode_AsUTF8(name);
            return true;
        }
        if (PyArray_Check(chrom_col)) {
            PyObject *name = *(PyObject **)PyArray_GETPTR1((PyArrayObject *)chrom_col, i);
            if (PyUnicode_Check(name)) {
                out = PyUnicode_AsUTF8(name);
                return true;
            }
            if (PyNumber_Check(name)) {
                PMPY py_long(PyNumber_Long(name), true);
                if (!py_long) {
                    PyErr_Clear();
                    TGLError("%s: bad chrom value at row %lld", what, (long long)i);
                }
                long v = PyLong_AsLong(py_long);
                out = std::to_string(v);
                return true;
            }
        }
        TGLError("%s: unsupported chrom column type at row %lld", what, (long long)i);
        return false;
    };

    Py_ssize_t nrows = PyObject_Length(start_col);
    if (nrows < 0) {
        PyErr_Clear();
        TGLError("%s: cannot determine number of rows", what);
    }

    chromids.reserve((size_t)nrows);
    starts.reserve((size_t)nrows);
    ends.reserve((size_t)nrows);

    for (Py_ssize_t i = 0; i < nrows; ++i) {
        string cname;
        if (!get_chrom_name(i, cname)) continue;

        int cid;
        try {
            cid = chromkey.chrom2id(cname);
        } catch (TGLException &) {
            if (skip_unknown_chroms) continue;
            throw;
        }
        if (cid < 0) {
            if (skip_unknown_chroms) continue;
            TGLError("%s: unknown chromosome '%s'", what, cname.c_str());
        }

        // start / end may be int64 or other integer numpy arrays. Read via
        // PyArray_GETITEM then convert.
        PyObject *s_obj = PyArray_GETITEM((PyArrayObject *)start_col,
            (const char *)PyArray_GETPTR1((PyArrayObject *)start_col, i));
        PyObject *e_obj = PyArray_GETITEM((PyArrayObject *)end_col,
            (const char *)PyArray_GETPTR1((PyArrayObject *)end_col, i));
        if (!s_obj || !e_obj) {
            Py_XDECREF(s_obj); Py_XDECREF(e_obj);
            PyErr_Clear();
            TGLError("%s: cannot read start/end at row %lld", what, (long long)i);
        }
        int64_t s_val = (int64_t)PyLong_AsLongLong(s_obj);
        int64_t e_val = (int64_t)PyLong_AsLongLong(e_obj);
        Py_DECREF(s_obj);
        Py_DECREF(e_obj);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("%s: bad start/end at row %lld", what, (long long)i);
        }

        chromids.push_back(cid);
        starts.push_back(s_val);
        ends.push_back(e_val);
    }
}

// Build the dict result {"chrom": np.ndarray[object], "start": np.ndarray[int64],
// "end": np.ndarray[int64]}.
static PyObject *build_result_dict(const vector<int> &out_chromids,
                                   const vector<int64_t> &out_starts,
                                   const vector<int64_t> &out_ends,
                                   const GenomeChromKey &chromkey)
{
    const size_t n = out_chromids.size();
    npy_intp dims[1] = { (npy_intp)n };

    PMPY py_chrom(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, dims, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, dims, NPY_INT64), true);
    if (!py_chrom || !py_start || !py_end) {
        verror("Failed to allocate result arrays");
    }

    int64_t *start_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_start);
    int64_t *end_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_end);

    // Cache chrom name PyObjects so we don't allocate one per row when chroms repeat.
    map<int, PyObject *> chrom_str_cache;
    for (size_t i = 0; i < n; ++i) {
        int cid = out_chromids[i];
        auto it = chrom_str_cache.find(cid);
        PyObject *name_obj;
        if (it == chrom_str_cache.end()) {
            name_obj = PyUnicode_FromString(chromkey.id2chrom(cid).c_str());
            if (!name_obj) verror("Failed to create chrom name string");
            chrom_str_cache[cid] = name_obj;
        } else {
            name_obj = it->second;
        }

        PyArrayObject *arr = (PyArrayObject *)(PyObject *)py_chrom;
        char *ptr = (char *)PyArray_GETPTR1(arr, i);
        Py_INCREF(name_obj);
        if (PyArray_SETITEM(arr, ptr, name_obj) < 0) {
            Py_DECREF(name_obj);
            for (auto &p : chrom_str_cache) Py_DECREF(p.second);
            verror("Failed to set chrom name in result array");
        }
        Py_DECREF(name_obj);  // SETITEM increfs, balance our incref above

        start_data[i] = out_starts[i];
        end_data[i] = out_ends[i];
    }
    for (auto &p : chrom_str_cache) Py_DECREF(p.second);

    PMPY py_result(PyDict_New(), true);
    if (!py_result) verror("Failed to create result dict");
    if (PyDict_SetItemString(py_result, "chrom", py_chrom) < 0 ||
        PyDict_SetItemString(py_result, "start", py_start) < 0 ||
        PyDict_SetItemString(py_result, "end", py_end) < 0) {
        verror("Failed to populate result dict");
    }

    py_result.to_be_stolen();
    return (PyObject *)py_result;
}

} // anonymous namespace


// pm_intervals_random(size, n, dist_from_edge, chrom_intervals, filter_intervals_or_none, seed)
//   size, n, dist_from_edge, seed: Python ints / floats
//   chrom_intervals: list-of-arrays produced by _df2pymisha (must have chrom/start/end)
//   filter_intervals_or_none: same format, or None
PyObject *pm_intervals_random(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        long long       size_in = 0;
        long long       n_in = 0;
        double          dist_from_edge = 0.0;
        PyObject       *py_chrom_df = nullptr;
        PyObject       *py_filter = nullptr;
        long long       seed_in = 0;

        if (!PyArg_ParseTuple(args, "LLdOOL",
                              &size_in, &n_in, &dist_from_edge,
                              &py_chrom_df, &py_filter, &seed_in)) {
            return_err();
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            TGLError("Database not initialized. Call gdb_init() first.");
        }

        if (size_in <= 0) TGLError("size must be positive");
        if (n_in <= 0) TGLError("n must be positive");
        if (dist_from_edge < 0) TGLError("dist_from_edge must be non-negative");

        const int64_t size = (int64_t)size_in;
        const int64_t n = (int64_t)n_in;
        const int64_t dfe = (int64_t)dist_from_edge;

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Parse chrom_intervals (canonical genome intervals).
        vector<int>     chrom_ids;
        vector<int64_t> chrom_starts;
        vector<int64_t> chrom_ends;
        parse_intervals_df(py_chrom_df, chromkey,
                           chrom_ids, chrom_starts, chrom_ends,
                           /*skip_unknown_chroms=*/false, "chrom_intervals");

        if (chrom_ids.empty()) {
            TGLError("No chromosomes provided");
        }

        // Parse filter (may be None or empty).
        vector<int>     filter_ids;
        vector<int64_t> filter_starts;
        vector<int64_t> filter_ends;
        if (py_filter && py_filter != Py_None) {
            parse_intervals_df(py_filter, chromkey,
                               filter_ids, filter_starts, filter_ends,
                               /*skip_unknown_chroms=*/true, "filter");
        }

        // Group filter intervals by chromid. Each filter interval is
        // left-expanded by (size - 1) so that any sample start in the
        // resulting segment cannot overlap the original interval.
        map<int, vector<StartEnd>> filter_by_chrom;
        for (size_t i = 0; i < filter_ids.size(); ++i) {
            StartEnd se;
            se.start = std::max<int64_t>(0, filter_starts[i] - size + 1);
            se.end = filter_ends[i];
            filter_by_chrom[filter_ids[i]].push_back(se);
        }
        // Sort & merge overlaps per chrom.
        for (auto &kv : filter_by_chrom) {
            auto &v = kv.second;
            std::sort(v.begin(), v.end(), startend_less);
            size_t out = 0;
            for (size_t i = 1; i < v.size(); ++i) {
                if (v[i].start <= v[out].end) {
                    if (v[i].end > v[out].end) v[out].end = v[i].end;
                } else {
                    v[++out] = v[i];
                }
            }
            v.resize(v.empty() ? 0 : out + 1);
        }

        // Build valid segments per chromosome.
        //
        // Without filter: a single segment per chrom of length
        // (chrom_end - dfe) - (chrom_start + dfe), where `length` counts
        // the number of valid start positions. Sampling draws a start in
        // [seg_start, seg_start + (length - size)] so we keep length as the
        // raw range and rely on the sampling step's `seg.length - size`.
        // This matches R 5.6.30's fix (commit 1b41bceb): chromosomes whose
        // length exactly equals `size + 2 * dfe` must yield exactly one
        // valid start position.
        vector<ValidSegment> segments;
        segments.reserve(chrom_ids.size());

        for (size_t i = 0; i < chrom_ids.size(); ++i) {
            int cid = chrom_ids[i];
            int64_t cstart = chrom_starts[i];
            int64_t cend = chrom_ends[i];

            int64_t valid_start = cstart + dfe;
            int64_t valid_end = cend - dfe;  // exclusive
            if (valid_end - valid_start < size) {
                continue;  // too short
            }

            auto it = filter_by_chrom.find(cid);
            if (it == filter_by_chrom.end() || it->second.empty()) {
                ValidSegment seg;
                seg.chromid = cid;
                seg.start = valid_start;
                seg.length = valid_end - valid_start;
                seg.cum_prob = 0.0;
                segments.push_back(seg);
                continue;
            }

            // Subtract sorted+merged filter from [valid_start, valid_end).
            int64_t cur = valid_start;
            for (const StartEnd &f : it->second) {
                int64_t f_start = std::max(f.start, valid_start);
                int64_t f_end = std::min(f.end, valid_end);
                if (f_end <= cur) continue;        // entirely before current sweep
                if (f_start >= valid_end) break;   // entirely after segment
                if (f_start > cur) {
                    int64_t seg_len = f_start - cur;
                    if (seg_len >= size) {
                        ValidSegment seg;
                        seg.chromid = cid;
                        seg.start = cur;
                        seg.length = seg_len;
                        seg.cum_prob = 0.0;
                        segments.push_back(seg);
                    }
                }
                if (f_end > cur) cur = f_end;
            }
            if (valid_end > cur) {
                int64_t seg_len = valid_end - cur;
                if (seg_len >= size) {
                    ValidSegment seg;
                    seg.chromid = cid;
                    seg.start = cur;
                    seg.length = seg_len;
                    seg.cum_prob = 0.0;
                    segments.push_back(seg);
                }
            }
        }

        if (segments.empty()) {
            TGLError("No valid genomic positions for intervals of size %lld with dist_from_edge %lld",
                     (long long)size, (long long)dfe);
        }

        // Compute cumulative probabilities (weight = segment length).
        double total_length = 0.0;
        for (const auto &s : segments) total_length += (double)s.length;
        if (!(total_length > 0.0)) {
            TGLError("No valid genomic positions for random intervals");
        }
        double running = 0.0;
        for (auto &s : segments) {
            running += (double)s.length / total_length;
            s.cum_prob = running;
        }
        // Anchor the last to exactly 1.0 to avoid floating-point miss.
        segments.back().cum_prob = 1.0;

        // Sample n intervals.
        std::mt19937_64 rng((uint64_t)seed_in);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        vector<int>     out_chromids((size_t)n);
        vector<int64_t> out_starts((size_t)n);
        vector<int64_t> out_ends((size_t)n);

        // Build cum_prob array for std::upper_bound lookup.
        vector<double> cum(segments.size());
        for (size_t i = 0; i < segments.size(); ++i) cum[i] = segments[i].cum_prob;

        for (int64_t i = 0; i < n; ++i) {
            double u = u01(rng);
            auto it = std::lower_bound(cum.begin(), cum.end(), u);
            size_t idx = (it == cum.end()) ? cum.size() - 1
                                            : (size_t)(it - cum.begin());
            const ValidSegment &seg = segments[idx];

            // Valid start range: [seg.start, seg.start + (seg.length - size)] inclusive.
            int64_t range = seg.length - size;  // >= 0 by construction
            double u2 = u01(rng);
            int64_t offset = (range == 0) ? 0 : (int64_t)(u2 * (double)(range + 1));
            if (offset > range) offset = range;  // guard for u2 == 1.0 edge

            int64_t s = seg.start + offset;
            int64_t e = s + size;

            out_chromids[(size_t)i] = seg.chromid;
            out_starts[(size_t)i] = s;
            out_ends[(size_t)i] = e;

            if ((i & 0xFFFF) == 0) check_interrupt();
        }

        return_py(build_result_dict(out_chromids, out_starts, out_ends, chromkey));

    } catch (TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return_err();
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return_err();
    }

    return_none();
}
