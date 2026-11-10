// The aggregation of gtrack_liftover, as R's GTrackLiftover.cpp: a sparse
// target cuts the lifted intervals at their boundaries and aggregates each
// segment over the intervals covering it (aggregate_overlapping_cpp), a dense
// target aggregates what lands in each bin (aggregate_per_bin_cpp), both through
// R's aggregate_values (aggregate_value_for_bin). pm_liftover_aggregate exposes
// the sparse one.

#include "PMLiftoverAggregate.h"
#include "pymisha.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <new>
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
    if (s == "max.coverage_len")  return AggType::MAX_COV_LEN;
    if (s == "min.coverage_len")  return AggType::MIN_COV_LEN;
    if (s == "max.coverage_frac") return AggType::MAX_COV_FRAC;
    if (s == "min.coverage_frac") return AggType::MIN_COV_FRAC;
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

} // namespace

// ---------------------------------------------------------------------------
// Per-bin aggregation
// ---------------------------------------------------------------------------

// R's contribution_order, the order of first, last and nth: where the lifted
// piece lies (start, end), then chain_id (pieces that coincide, as on an overlap
// "agg" cut out of both chains, or two source bins landing on one target bin),
// then value, larger first.
static bool contribution_order(const BinContribution *a, const BinContribution *b) {
    if (a->start != b->start) return a->start < b->start;
    if (a->end != b->end) return a->end < b->end;
    if (a->chain_id != b->chain_id) return a->chain_id < b->chain_id;
    return a->value > b->value;
}

// Mirrors AggregationHelpers.h::aggregate_values. Contributions sharing the same
// chain_id are merged before the reduce step.
double aggregate_value_for_bin(
    const std::vector<BinContribution> &state,
    AggType agg, bool na_rm,
    int64_t min_n_or_negative, int64_t nth_index)
{
    // Under na_rm = false, any NaN contribution makes the whole locus NaN
    // (checked on the raw contributions, before the per-chain merge). R 5.11.5.
    if (!na_rm) {
        for (const auto &c : state) {
            if (c.is_na) return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Step 1: merge contributions sharing the same chain key. NaN pieces are
    // dropped FIRST (na_rm == true here) so a chain mapping both a finite and a
    // NaN source bin into this locus keeps its finite value instead of being
    // discarded wholesale by an is_na carried over the whole chain. R 5.11.5.
    std::vector<BinContribution> merged;
    merged.reserve(state.size());
    for (const auto &c : state) {
        if (c.is_na) continue;  // na_rm == true here (na_rm == false already returned)
        bool found = false;
        for (auto &m : merged) {
            if (m.chain_id == c.chain_id) {
                m.overlap_len += c.overlap_len;
                m.coverage_frac += c.coverage_frac;
                m.start = std::min(m.start, c.start);
                m.end   = std::max(m.end,   c.end);
                found = true;
                break;
            }
        }
        if (!found) merged.push_back(c);
    }

    // Step 2: all merged contributions are non-NA.
    std::vector<const BinContribution *> valid;
    valid.reserve(merged.size());
    for (const auto &c : merged) valid.push_back(&c);

    if (min_n_or_negative >= 0 && (int64_t)valid.size() < min_n_or_negative)
        return std::numeric_limits<double>::quiet_NaN();

    // Step 3: COUNT is special - defined even when no valid contribs (but after
    // min_n, as in R: a bin with no value and min_n >= 1 is NaN, not 0).
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
                contribution_order);
            return first->value;
        }
        case AggType::LAST: {
            const BinContribution *last = *std::max_element(
                valid.begin(), valid.end(),
                contribution_order);
            return last->value;
        }
        case AggType::NTH: {
            std::vector<const BinContribution *> sorted_v = valid;
            std::sort(sorted_v.begin(), sorted_v.end(),
                contribution_order);
            if (nth_index <= 0)
                return std::numeric_limits<double>::quiet_NaN();
            const size_t idx = (size_t)(nth_index - 1);
            if (idx >= sorted_v.size())
                return std::numeric_limits<double>::quiet_NaN();
            return sorted_v[idx]->value;
        }
        case AggType::MAX_COV_LEN:
        case AggType::MIN_COV_LEN:
        case AggType::MAX_COV_FRAC:
        case AggType::MIN_COV_FRAC: {
            // The contribution covering the most (least) of the locus, by length or
            // by fraction; a tie goes to the higher value.
            const BinContribution *best = valid.front();
            for (size_t i = 1; i < valid.size(); ++i) {
                const BinContribution *curr = valid[i];
                bool better = false;
                switch (agg) {
                    case AggType::MAX_COV_LEN:
                        better = curr->overlap_len > best->overlap_len ||
                                 (curr->overlap_len == best->overlap_len && curr->value > best->value);
                        break;
                    case AggType::MIN_COV_LEN:
                        better = curr->overlap_len < best->overlap_len ||
                                 (curr->overlap_len == best->overlap_len && curr->value > best->value);
                        break;
                    case AggType::MAX_COV_FRAC:
                        better = curr->coverage_frac > best->coverage_frac ||
                                 (curr->coverage_frac == best->coverage_frac && curr->value > best->value);
                        break;
                    case AggType::MIN_COV_FRAC:
                        better = curr->coverage_frac < best->coverage_frac ||
                                 (curr->coverage_frac == best->coverage_frac && curr->value > best->value);
                        break;
                    default:
                        break;
                }
                if (better)
                    best = curr;
            }
            return best->value;
        }
        case AggType::COUNT:
            // handled above
            break;
    }
    return std::numeric_limits<double>::quiet_NaN();  // unreachable
}

// ---------------------------------------------------------------------------
// Public: aggregate_overlapping_cpp
// ---------------------------------------------------------------------------

namespace {

// R's GIntervalVal::operator<: start, end, key, value. Rows equal in the first
// three are pieces of one source interval through one chain, so of one value.
struct LiftedOrder {
    const std::vector<std::int64_t> &start, &end, &key;
    const std::vector<double> &value;
    bool operator()(size_t a, size_t b) const {
        if (start[a] != start[b]) return start[a] < start[b];
        if (end[a] != end[b]) return end[a] < end[b];
        if (key[a] != key[b]) return key[a] < key[b];
        return value[a] < value[b];
    }
};

// R's LiftedPiecesSweep: finds the lifted pieces (rows, sorted in R's order) that
// overlap each target locus, for loci visited left to right. active holds the
// pieces that overlap the current locus, in the order of rows, and next the
// first piece not yet reached, so each piece is looked at only while it
// overlaps a locus.
struct LiftedPiecesSweep {
    const std::vector<size_t> &rows;
    const std::vector<std::int64_t> &start, &end, &key;
    const std::vector<double> &value;
    size_t next = 0;
    std::vector<size_t> active;

    // Appends to state each piece that overlaps [s, e), weighed by its overlap,
    // with coverage_frac = overlap / locus_len. s and e must not decrease from
    // call to call.
    void add_overlapping(std::int64_t s, std::int64_t e, double locus_len, std::vector<BinContribution> &state) {
        while (next < rows.size() && start[rows[next]] < e)
            active.push_back(rows[next++]);
        active.erase(std::remove_if(active.begin(), active.end(), [&](size_t r) { return end[r] <= s; }),
                     active.end());
        for (size_t r : active) {
            const std::int64_t ovl_s = std::max(s, start[r]);
            const std::int64_t ovl_e = std::min(e, end[r]);
            if (ovl_s >= ovl_e)
                continue;
            BinContribution c;
            c.value = value[r];
            c.overlap_len = (double)(ovl_e - ovl_s);
            c.coverage_frac = locus_len > 0 ? c.overlap_len / locus_len : 0;
            c.start = start[r];  // the piece's own place, by which first/last/nth order
            c.end = end[r];
            c.is_na = std::isnan(value[r]);
            c.chain_id = key[r];
            state.push_back(c);
        }
    }
};

}  // namespace

void aggregate_overlapping_cpp(
    const std::vector<std::string> &in_chrom,
    const std::vector<std::int64_t> &in_start,
    const std::vector<std::int64_t> &in_end,
    const std::vector<double> &in_value,
    const std::vector<std::int64_t> &in_key,
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
    if (in_start.size() != n || in_end.size() != n || in_value.size() != n || in_key.size() != n)
        throw std::invalid_argument("aggregate_overlapping_cpp: input vectors have mismatched lengths");

    std::map<std::string, std::vector<size_t>> by_chrom;
    for (size_t i = 0; i < n; ++i) {
        if (in_start[i] < in_end[i])  // an empty interval overlaps nothing
            by_chrom[in_chrom[i]].push_back(i);
    }

    const LiftedOrder order{in_start, in_end, in_key, in_value};
    std::vector<BinContribution> state;
    for (auto &kv : by_chrom) {
        std::vector<size_t> &rows = kv.second;
        std::sort(rows.begin(), rows.end(), order);

        // GTrackLiftover.cpp, SPARSE: cut the intervals at their boundaries and give
        // each segment the aggregate of the intervals that cover it, as
        // gintervals.liftover(value_col =) does; a gap between intervals is skipped.
        std::vector<std::int64_t> breaks;
        breaks.reserve(2 * rows.size());
        for (size_t r : rows) {
            breaks.push_back(in_start[r]);
            breaks.push_back(in_end[r]);
        }
        std::sort(breaks.begin(), breaks.end());
        breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());

        LiftedPiecesSweep sweep{rows, in_start, in_end, in_key, in_value};
        for (size_t b = 0; b + 1 < breaks.size(); ++b) {
            state.clear();
            sweep.add_overlapping(breaks[b], breaks[b + 1], (double)(breaks[b + 1] - breaks[b]), state);
            if (state.empty())
                continue;
            out_chrom.push_back(kv.first);
            out_start.push_back(breaks[b]);
            out_end.push_back(breaks[b + 1]);
            out_value.push_back(aggregate_value_for_bin(state, agg, na_rm, min_n_or_negative, nth_index));
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
            // R's order (GIntervalVal::operator<), which is the order the bin's
            // contributions are summed in
            std::sort(idxs.begin(), idxs.end(), LiftedOrder{in_start, in_end, in_chain_id, in_value});
        }

        const int64_t end_bin = (chrom_size + bin_size - 1) / bin_size;  // ceil
        LiftedPiecesSweep sweep{idxs, in_start, in_end, in_chain_id, in_value};
        std::vector<BinContribution> state;

        for (int64_t bin = 0; bin < end_bin; ++bin) {
            const int64_t bs = bin * bin_size;
            const int64_t be = std::min<int64_t>((bin + 1) * bin_size, chrom_size);

            // every contribution overlapping [bs, be); the bin is clamped to the
            // chromosome end, as R's locus_len
            state.clear();
            sweep.add_overlapping(bs, be, (double)(be - bs), state);

            // a bin nothing lands in, on any chrom, gets what aggregate_values gives no
            // contributions: NaN, or 0 under "count" (NaN below min_n), as R 5.18.0
            const double v = aggregate_value_for_bin(state, agg, na_rm, min_n_or_negative, nth_index);

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
 *             "end" (int64), "value" (float64), and optionally "key" (int64):
 *             rows sharing a key count once (default: every row its own key).
 *             All same length N.
 * agg_type:   str. One of mean, median, sum, min, max, count, first, last, nth,
 *             max.coverage_len, min.coverage_len, max.coverage_frac, min.coverage_frac.
 * na_rm:      bool. If False, any NaN in a run => NaN result.
 * min_n:      int. Minimum non-NaN count required; <0 means no minimum.
 * nth_index:  int. 1-based index for "nth" aggregation; ignored otherwise.
 *
 * Returns: dict with "chrom" (object/str), "start" (int64), "end" (int64),
 *          "value" (float64): one row per segment some interval covers, as
 *          aggregate_overlapping_cpp (NaN values included).
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
    PMPY arr_key;
    if (PyDict_GetItemString(df_dict, "key")) {
        arr_key.assign(get_array(df_dict, "key", NPY_INT64), true);
        if (!arr_key) return_err();
    }

    const npy_intp n_rows = PyArray_DIM((PyArrayObject *)*arr_chrom, 0);
    if (PyArray_DIM((PyArrayObject *)*arr_start, 0) != n_rows ||
        PyArray_DIM((PyArrayObject *)*arr_end, 0) != n_rows ||
        PyArray_DIM((PyArrayObject *)*arr_value, 0) != n_rows ||
        (arr_key && PyArray_DIM((PyArrayObject *)*arr_key, 0) != n_rows)) {
        PyErr_SetString(PyExc_ValueError,
                        "df_dict columns have mismatched lengths");
        return_err();
    }

    PyObject **chrom_in = (PyObject **)PyArray_DATA((PyArrayObject *)*arr_chrom);
    const int64_t *start_in = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_start);
    const int64_t *end_in = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_end);
    const double *value_in = (const double *)PyArray_DATA((PyArrayObject *)*arr_value);
    const int64_t *key_in = arr_key ? (const int64_t *)PyArray_DATA((PyArrayObject *)*arr_key) : nullptr;

    // Extract numpy arrays into std::vectors.
    std::vector<std::string> v_chrom(n_rows);
    std::vector<int64_t> v_start(n_rows), v_end(n_rows), v_key(n_rows);
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
        v_key[i]   = key_in ? key_in[i] : (int64_t)i;
    }

    // Call the helper.
    std::vector<std::string> out_chrom;
    std::vector<int64_t> out_start, out_end;
    std::vector<double> out_value;

    try {
        aggregate_overlapping_cpp(
            v_chrom, v_start, v_end, v_value, v_key,
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
