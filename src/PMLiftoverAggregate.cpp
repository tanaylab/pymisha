// pm_liftover_aggregate: C++ port of liftover._aggregate_overlapping.
// Segments overlapping intervals per chrom via sweep-line, applies the
// named aggregator at each segment, merges adjacent equal-value segments.
//
// Also exposes aggregate_overlapping_cpp and aggregate_per_bin_cpp helpers
// for use by the pm_liftover_track orchestrator (Task 6).

#include "PMLiftoverAggregate.h"
#include "pymisha.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Public: parse agg type string
// ---------------------------------------------------------------------------

AggType parse_agg_type_str(const std::string &s)
{
    if (s == "mean")   return AggType::MEAN;
    if (s == "median") return AggType::MEDIAN;
    if (s == "sum")    return AggType::SUM;
    if (s == "min")    return AggType::MIN;
    if (s == "max")    return AggType::MAX;
    if (s == "count")  return AggType::COUNT;
    if (s == "first")  return AggType::FIRST;
    if (s == "last")   return AggType::LAST;
    if (s == "nth")    return AggType::NTH;
    throw std::invalid_argument("Unknown agg_type: " + s);
}

namespace {

// Pull a key from df_dict, coerce to a contiguous numpy array of the requested
// dtype. Returns nullptr on failure (with Python error set).
PyObject *get_array(PyObject *df_dict, const char *key, int dtype)
{
    PyObject *item = PyDict_GetItemString(df_dict, key);
    if (!item) {
        PyErr_Format(PyExc_ValueError, "df_dict missing required key '%s'", key);
        return nullptr;
    }
    PyObject *arr = PyArray_FROM_OTF(item, dtype, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return nullptr;
    if (PyArray_NDIM((PyArrayObject *)arr) != 1) {
        PyErr_Format(PyExc_ValueError, "df_dict['%s'] must be 1-D", key);
        Py_DECREF(arr);
        return nullptr;
    }
    return arr;
}

// ---------------------------------------------------------------------------
// Per-bin aggregation internals
// ---------------------------------------------------------------------------

struct BinContribution {
    double value;
    double overlap_len;
    int64_t start;
    int64_t end;
    bool is_na;
    int64_t chain_id;
};

// Mirrors AggregationHelpers.h::aggregate_values for the 9 supported agg types.
// Contributions sharing the same chain_id are merged before the reduce step.
double aggregate_value_for_bin(
    const std::vector<BinContribution> &state,
    AggType agg, bool na_rm,
    int64_t min_n_or_negative, int64_t nth_index)
{
    if (state.empty()) {
        if (agg == AggType::COUNT) return 0.0;
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Step 1: merge contributions sharing the same chain_id.
    std::vector<BinContribution> merged;
    merged.reserve(state.size());
    for (const auto &c : state) {
        bool found = false;
        for (auto &m : merged) {
            if (m.chain_id == c.chain_id) {
                m.overlap_len += c.overlap_len;
                m.start = std::min(m.start, c.start);
                m.end   = std::max(m.end,   c.end);
                m.is_na = m.is_na || c.is_na;
                found = true;
                break;
            }
        }
        if (!found) merged.push_back(c);
    }
    const std::vector<BinContribution> &contribs = merged;

    // Step 2: collect valid (non-NA) contribs; respect na_rm.
    std::vector<const BinContribution *> valid;
    valid.reserve(contribs.size());
    for (const auto &c : contribs) {
        if (c.is_na) {
            if (!na_rm) return std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        valid.push_back(&c);
    }

    if (min_n_or_negative >= 0 && (int64_t)valid.size() < min_n_or_negative)
        return std::numeric_limits<double>::quiet_NaN();

    // Step 3: COUNT is special - defined even when no valid contribs.
    if (agg == AggType::COUNT)
        return (double)valid.size();
    if (valid.empty())
        return std::numeric_limits<double>::quiet_NaN();

    switch (agg) {
        case AggType::MEAN: {
            double s = 0;
            for (auto *c : valid) s += c->value;
            return s / (double)valid.size();
        }
        case AggType::SUM: {
            double s = 0;
            for (auto *c : valid) s += c->value;
            return s;
        }
        case AggType::MIN: {
            double r = valid.front()->value;
            for (auto *c : valid) if (c->value < r) r = c->value;
            return r;
        }
        case AggType::MAX: {
            double r = valid.front()->value;
            for (auto *c : valid) if (c->value > r) r = c->value;
            return r;
        }
        case AggType::MEDIAN: {
            std::vector<double> vs;
            vs.reserve(valid.size());
            for (auto *c : valid) vs.push_back(c->value);
            std::sort(vs.begin(), vs.end());
            const size_t mid = vs.size() / 2;
            return (vs.size() % 2 == 0) ? (vs[mid - 1] + vs[mid]) / 2.0 : vs[mid];
        }
        case AggType::FIRST: {
            const BinContribution *first = *std::min_element(
                valid.begin(), valid.end(),
                [](const BinContribution *a, const BinContribution *b) {
                    if (a->start != b->start) return a->start < b->start;
                    if (a->end   != b->end)   return a->end   < b->end;
                    return a->value > b->value;
                });
            return first->value;
        }
        case AggType::LAST: {
            const BinContribution *last = *std::max_element(
                valid.begin(), valid.end(),
                [](const BinContribution *a, const BinContribution *b) {
                    if (a->start != b->start) return a->start < b->start;
                    if (a->end   != b->end)   return a->end   < b->end;
                    return a->value > b->value;
                });
            return last->value;
        }
        case AggType::NTH: {
            std::vector<const BinContribution *> sorted_v = valid;
            std::sort(sorted_v.begin(), sorted_v.end(),
                [](const BinContribution *a, const BinContribution *b) {
                    if (a->start != b->start) return a->start < b->start;
                    if (a->end   != b->end)   return a->end   < b->end;
                    return a->value > b->value;
                });
            if (nth_index <= 0)
                return std::numeric_limits<double>::quiet_NaN();
            const size_t idx = (size_t)(nth_index - 1);
            if (idx >= sorted_v.size())
                return std::numeric_limits<double>::quiet_NaN();
            return sorted_v[idx]->value;
        }
        case AggType::COUNT:
            // handled above
            break;
    }
    return std::numeric_limits<double>::quiet_NaN();  // unreachable
}

} // namespace

// ---------------------------------------------------------------------------
// Public: aggregate_overlapping_cpp
// ---------------------------------------------------------------------------

void aggregate_overlapping_cpp(
    const std::vector<std::string> &in_chrom,
    const std::vector<std::int64_t> &in_start,
    const std::vector<std::int64_t> &in_end,
    const std::vector<double> &in_value,
    AggType agg,
    bool na_rm,
    std::int64_t min_n_or_negative,
    std::int64_t nth_index,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value)
{
    const size_t n = in_chrom.size();
    if (in_start.size() != n || in_end.size() != n || in_value.size() != n)
        throw std::invalid_argument("aggregate_overlapping_cpp: input vectors have mismatched lengths");

    // Group rows by chrom, preserving first-seen order.
    std::vector<std::string> chrom_order;
    std::unordered_map<std::string, std::vector<size_t>> chrom_rows;
    chrom_order.reserve(16);
    for (size_t i = 0; i < n; ++i) {
        const std::string &key = in_chrom[i];
        auto it = chrom_rows.find(key);
        if (it == chrom_rows.end()) {
            chrom_order.push_back(key);
            chrom_rows.emplace(key, std::vector<size_t>{i});
        } else {
            it->second.push_back(i);
        }
    }

    // Sort chrom_order alphabetically to mirror Python's groupby(sort=False)
    // after sort_values(["chrom", "start", "end"]).
    std::sort(chrom_order.begin(), chrom_order.end());

    for (const std::string &chrom : chrom_order) {
        std::vector<size_t> &row_ids = chrom_rows[chrom];

        // Sort by (start, end, original-row-order).
        std::sort(row_ids.begin(), row_ids.end(),
                  [&](size_t a, size_t b) {
                      if (in_start[a] != in_start[b]) return in_start[a] < in_start[b];
                      if (in_end[a] != in_end[b]) return in_end[a] < in_end[b];
                      return a < b;
                  });

        const size_t m = row_ids.size();
        if (m == 0) continue;

        // Local copies indexed [0..m) in sorted order.
        std::vector<int64_t> S(m), E(m);
        std::vector<double> V(m);
        for (size_t i = 0; i < m; ++i) {
            S[i] = in_start[row_ids[i]];
            E[i] = in_end[row_ids[i]];
            V[i] = in_value[row_ids[i]];
        }

        // Collect breakpoints: sorted unique union of starts and ends.
        std::vector<int64_t> points;
        points.reserve(2 * m);
        for (size_t i = 0; i < m; ++i) {
            points.push_back(S[i]);
            points.push_back(E[i]);
        }
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        if (points.size() < 2) continue;

        // Bucket rows by start and end coords.
        std::unordered_map<int64_t, std::vector<size_t>> starts_at, ends_at;
        for (size_t i = 0; i < m; ++i) {
            starts_at[S[i]].push_back(i);
            ends_at[E[i]].push_back(i);
        }

        // Active set kept ordered by row index (so iteration is row-order stable).
        std::set<size_t> active;

        // Used for adjacent-segment merge within this chrom run.
        const size_t out_chrom_anchor = out_chrom.size();

        for (size_t pi = 0; pi + 1 < points.size(); ++pi) {
            int64_t coord = points[pi];
            int64_t next_coord = points[pi + 1];

            auto it_end = ends_at.find(coord);
            if (it_end != ends_at.end()) {
                for (size_t idx : it_end->second) active.erase(idx);
            }
            auto it_start = starts_at.find(coord);
            if (it_start != starts_at.end()) {
                for (size_t idx : it_start->second) active.insert(idx);
            }

            if (next_coord <= coord || active.empty()) continue;

            // Gather active values in row order.
            std::vector<double> vals;
            vals.reserve(active.size());
            for (size_t idx : active) vals.push_back(V[idx]);

            // NaN handling.
            bool any_nan = false;
            std::vector<double> vals_clean;
            vals_clean.reserve(vals.size());
            for (double x : vals) {
                if (std::isnan(x)) any_nan = true;
                else vals_clean.push_back(x);
            }

            if (!na_rm && any_nan) continue;
            if (min_n_or_negative >= 0 && (int64_t)vals_clean.size() < min_n_or_negative) continue;
            if (vals_clean.empty()) continue;

            double seg_val;
            switch (agg) {
                case AggType::MEAN: {
                    double s = 0.0;
                    for (double x : vals_clean) s += x;
                    seg_val = s / (double)vals_clean.size();
                    break;
                }
                case AggType::MEDIAN: {
                    std::vector<double> cp = vals_clean;
                    std::sort(cp.begin(), cp.end());
                    size_t k = cp.size();
                    if (k % 2 == 1) seg_val = cp[k / 2];
                    else seg_val = 0.5 * (cp[k / 2 - 1] + cp[k / 2]);
                    break;
                }
                case AggType::SUM: {
                    double s = 0.0;
                    for (double x : vals_clean) s += x;
                    seg_val = s;
                    break;
                }
                case AggType::MIN: {
                    double mn = vals_clean[0];
                    for (size_t i = 1; i < vals_clean.size(); ++i)
                        if (vals_clean[i] < mn) mn = vals_clean[i];
                    seg_val = mn;
                    break;
                }
                case AggType::MAX: {
                    double mx = vals_clean[0];
                    for (size_t i = 1; i < vals_clean.size(); ++i)
                        if (vals_clean[i] > mx) mx = vals_clean[i];
                    seg_val = mx;
                    break;
                }
                case AggType::COUNT: {
                    seg_val = (double)vals_clean.size();
                    break;
                }
                case AggType::FIRST: {
                    bool found = false;
                    seg_val = 0.0;
                    for (double x : vals) {
                        if (!std::isnan(x)) { seg_val = x; found = true; break; }
                    }
                    if (!found) continue;
                    break;
                }
                case AggType::LAST: {
                    bool found = false;
                    seg_val = 0.0;
                    for (auto it = vals.rbegin(); it != vals.rend(); ++it) {
                        if (!std::isnan(*it)) { seg_val = *it; found = true; break; }
                    }
                    if (!found) continue;
                    break;
                }
                case AggType::NTH: {
                    if (nth_index < 1) continue;
                    int64_t k = 0;
                    bool found = false;
                    seg_val = 0.0;
                    for (double x : vals) {
                        if (!std::isnan(x)) {
                            ++k;
                            if (k == nth_index) { seg_val = x; found = true; break; }
                        }
                    }
                    if (!found) continue;
                    break;
                }
                default:
                    continue;
            }

            // Adjacent-segment merge: only within this chrom's run, and only if
            // the previous segment ends exactly at the current coord and shares
            // the same value (relative tolerance 1e-12 to mirror np.isclose).
            if (out_chrom.size() > out_chrom_anchor &&
                out_end.back() == coord &&
                std::fabs(out_value.back() - seg_val) <=
                    1e-12 * std::max(1.0, std::fabs(seg_val)))
            {
                out_end.back() = next_coord;
            } else {
                out_chrom.push_back(chrom);
                out_start.push_back(coord);
                out_end.push_back(next_coord);
                out_value.push_back(seg_val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public: aggregate_per_bin_cpp
// ---------------------------------------------------------------------------

void aggregate_per_bin_cpp(
    const std::vector<std::string> &in_chrom,
    const std::vector<std::int64_t> &in_start,
    const std::vector<std::int64_t> &in_end,
    const std::vector<double> &in_value,
    const std::vector<std::int64_t> &in_chain_id,
    const std::vector<std::pair<std::string, std::int64_t>> &tgt_chrom_sizes,
    std::int64_t bin_size,
    AggType agg,
    bool na_rm,
    std::int64_t min_n_or_negative,
    std::int64_t nth_index,
    std::vector<std::string> &out_chrom,
    std::vector<std::int64_t> &out_start,
    std::vector<std::int64_t> &out_end,
    std::vector<double> &out_value)
{
    if (bin_size <= 0)
        throw std::invalid_argument("aggregate_per_bin_cpp: bin_size must be positive");

    const size_t n = in_chrom.size();
    if (in_start.size() != n || in_end.size() != n ||
        in_value.size() != n || in_chain_id.size() != n)
        throw std::invalid_argument("aggregate_per_bin_cpp: input vectors have mismatched lengths");

    // 1. Group input row indices by chrom.
    std::unordered_map<std::string, std::vector<size_t>> by_chrom;
    for (size_t i = 0; i < n; ++i) by_chrom[in_chrom[i]].push_back(i);

    // 2. For each chrom in tgt_chrom_sizes (use vector order, which the caller
    //    builds from Python dict insertion order), sort input indices by
    //    (start, end), then iterate output bins.
    for (const auto &kv : tgt_chrom_sizes) {
        const std::string &chrom = kv.first;
        const int64_t chrom_size = kv.second;

        auto it = by_chrom.find(chrom);
        std::vector<size_t> idxs;
        if (it != by_chrom.end()) {
            idxs = it->second;
            std::sort(idxs.begin(), idxs.end(),
                [&](size_t a, size_t b) {
                    if (in_start[a] != in_start[b]) return in_start[a] < in_start[b];
                    return in_end[a] < in_end[b];
                });
        }

        const int64_t end_bin = (chrom_size + bin_size - 1) / bin_size;  // ceil
        // iinterv_val cursor: mirrors R's GTrackLiftover.cpp iinterv_val.
        // Advances monotonically across bins so we don't re-scan past intervals.
        size_t iinterv = 0;

        for (int64_t bin = 0; bin < end_bin; ++bin) {
            const int64_t bs = bin * bin_size;
            const int64_t be = std::min<int64_t>((bin + 1) * bin_size, chrom_size);

            // Collect contributions for this bin.
            // Mirrors R: iter starts at iinterv_val, advances until past bin.
            std::vector<BinContribution> state;
            bool had_intersect = false;
            size_t iter = iinterv;

            for (; iter < idxs.size(); ++iter) {
                const size_t i = idxs[iter];
                const int64_t ovl_s = std::max<int64_t>(bs, in_start[i]);
                const int64_t ovl_e = std::min<int64_t>(be, in_end[i]);
                if (ovl_s < ovl_e) {
                    // Genuine overlap.
                    had_intersect = true;
                    BinContribution c;
                    c.value      = in_value[i];
                    c.overlap_len = (double)(ovl_e - ovl_s);
                    c.start      = ovl_s;
                    c.end        = ovl_e;
                    c.is_na      = std::isnan(in_value[i]);
                    c.chain_id   = in_chain_id[i];
                    state.push_back(c);
                } else if (in_end[i] > bs) {
                    // interval ends after bin start but no genuine overlap:
                    // if we already had an intersection and this interval
                    // starts after bin end, back up one (like R's --iter + break).
                    if (had_intersect && in_start[i] > be) {
                        if (iter > iinterv) --iter;
                    }
                    break;
                }
                // else: interval is entirely before bs - advance iinterv.
                if (!had_intersect && iter == iinterv) iinterv = iter + 1;
            }
            // After inner loop, advance iinterv to iter (mirroring R's iinterv_val = iter).
            if (iter > iinterv) iinterv = iter;

            const double v = aggregate_value_for_bin(
                state, agg, na_rm, min_n_or_negative, nth_index);

            out_chrom.push_back(chrom);
            out_start.push_back(bs);
            out_end.push_back(be);
            out_value.push_back(v);  // NaN for empty bins.
        }
    }
}

// ---------------------------------------------------------------------------
// Python C-API: pm_liftover_aggregate (thin wrapper)
// ---------------------------------------------------------------------------

/*
 * pm_liftover_aggregate(df_dict, agg_type, na_rm, min_n, nth_index) -> dict
 *
 * df_dict:    dict with numpy arrays "chrom" (object/str), "start" (int64),
 *             "end" (int64), "value" (float64). All same length N.
 * agg_type:   str. One of mean, median, sum, min, max, count, first, last, nth.
 * na_rm:      bool. If False, any NaN in a segment's active set => NaN result.
 * min_n:      int. Minimum non-NaN count required; <0 means no minimum.
 * nth_index:  int. 1-based index for "nth" aggregation; ignored otherwise.
 *
 * Returns: dict with "chrom" (object/str), "start" (int64), "end" (int64),
 *          "value" (float64). NaN-valued segments are dropped, adjacent
 *          segments with equal values are merged.
 *
 * Raises ValueError on malformed input.
 */
PyObject *pm_liftover_aggregate(PyObject *self, PyObject *args)
{
    PyObject *df_dict;
    const char *agg_type_str;
    int na_rm_int;
    long long min_n_ll;
    long long nth_index_ll;
    if (!PyArg_ParseTuple(args, "OspLL", &df_dict, &agg_type_str,
                          &na_rm_int, &min_n_ll, &nth_index_ll)) {
        return nullptr;
    }

    AggType agg;
    try {
        agg = parse_agg_type_str(agg_type_str);
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }

    if (!PyDict_Check(df_dict)) {
        PyErr_SetString(PyExc_TypeError, "df_dict must be a dict");
        return nullptr;
    }

    try {

    PMPY arr_chrom(get_array(df_dict, "chrom", NPY_OBJECT), true);
    if (!arr_chrom) return_err();
    PMPY arr_start(get_array(df_dict, "start", NPY_INT64), true);
    if (!arr_start) return_err();
    PMPY arr_end(get_array(df_dict, "end", NPY_INT64), true);
    if (!arr_end) return_err();
    PMPY arr_value(get_array(df_dict, "value", NPY_DOUBLE), true);
    if (!arr_value) return_err();

    const npy_intp n_rows = PyArray_DIM((PyArrayObject *)*arr_chrom, 0);
    if (PyArray_DIM((PyArrayObject *)*arr_start, 0) != n_rows ||
        PyArray_DIM((PyArrayObject *)*arr_end, 0) != n_rows ||
        PyArray_DIM((PyArrayObject *)*arr_value, 0) != n_rows) {
        PyErr_SetString(PyExc_ValueError,
                        "df_dict columns have mismatched lengths");
        return_err();
    }

    PyObject **chrom_in = (PyObject **)PyArray_DATA((PyArrayObject *)*arr_chrom);
    const int64_t *start_in = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_start);
    const int64_t *end_in = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_end);
    const double *value_in = (const double *)PyArray_DATA((PyArrayObject *)*arr_value);

    // Extract numpy arrays into std::vectors.
    std::vector<std::string> v_chrom(n_rows);
    std::vector<int64_t> v_start(n_rows), v_end(n_rows);
    std::vector<double> v_value(n_rows);

    for (npy_intp i = 0; i < n_rows; ++i) {
        PyObject *co = chrom_in[i];
        if (!co || !PyUnicode_Check(co)) {
            PyErr_Format(PyExc_TypeError,
                         "df_dict['chrom'][%lld] is not a string",
                         (long long)i);
            return_err();
        }
        Py_ssize_t clen = 0;
        const char *cstr = PyUnicode_AsUTF8AndSize(co, &clen);
        if (!cstr) return_err();
        v_chrom[i].assign(cstr, (size_t)clen);
        v_start[i] = start_in[i];
        v_end[i]   = end_in[i];
        v_value[i] = value_in[i];
    }

    // Call the helper.
    std::vector<std::string> out_chrom;
    std::vector<int64_t> out_start, out_end;
    std::vector<double> out_value;

    try {
        aggregate_overlapping_cpp(
            v_chrom, v_start, v_end, v_value,
            agg, (bool)na_rm_int, (int64_t)min_n_ll, (int64_t)nth_index_ll,
            out_chrom, out_start, out_end, out_value);
    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return_err();
    }

    // Build output dict.
    npy_intp n_out = (npy_intp)out_chrom.size();
    PMPY py_chrom(PyArray_SimpleNew(1, &n_out, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, &n_out, NPY_INT64), true);
    PMPY py_value(PyArray_SimpleNew(1, &n_out, NPY_DOUBLE), true);
    if (!py_chrom || !py_start || !py_end || !py_value) return_err();

    PyObject **chrom_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_chrom);
    int64_t *start_out = (int64_t *)PyArray_DATA((PyArrayObject *)*py_start);
    int64_t *end_out = (int64_t *)PyArray_DATA((PyArrayObject *)*py_end);
    double *value_out = (double *)PyArray_DATA((PyArrayObject *)*py_value);

    // Pre-fill every object slot with a non-NULL placeholder.
    for (npy_intp i = 0; i < n_out; ++i) chrom_out[i] = nullptr;

    for (npy_intp i = 0; i < n_out; ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(out_chrom[i].data(),
                                                  out_chrom[i].size());
        if (!s) {
            for (npy_intp j = i; j < n_out; ++j) {
                Py_INCREF(Py_None);
                chrom_out[j] = Py_None;
            }
            return_err();
        }
        chrom_out[i] = s;
        start_out[i] = out_start[i];
        end_out[i]   = out_end[i];
        value_out[i] = out_value[i];
    }

    PMPY result(PyDict_New(), true);
    if (!result) return_err();
    PyDict_SetItemString(result, "chrom", py_chrom);
    PyDict_SetItemString(result, "start", py_start);
    PyDict_SetItemString(result, "end", py_end);
    PyDict_SetItemString(result, "value", py_value);

    return_py(result);

    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return_err();
    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return_err();
    } catch (TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return_err();
    }
}
