/*
 * PMTrackExpression2DVars.cpp
 */

#include "PMTrackExpression2DVars.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "TGLException.h"

PMTrackExpression2DVars::PMTrackExpression2DVars()
    : m_rng(60427)  // project-wide RNG seed
{}

PMTrackExpression2DVars::~PMTrackExpression2DVars() = default;

PMTrackExpression2DVars::AggFunc
PMTrackExpression2DVars::parse_func(const std::string &name)
{
    if (name == "area")          return AGG_AREA;
    if (name == "weighted.sum")  return AGG_WEIGHTED_SUM;
    if (name == "min")           return AGG_MIN;
    if (name == "max")           return AGG_MAX;
    if (name == "avg" || name == "mean") return AGG_AVG;
    if (name == "exists")        return AGG_EXISTS;
    if (name == "size")          return AGG_SIZE;
    if (name == "first")         return AGG_FIRST;
    if (name == "last")          return AGG_LAST;
    if (name == "sample")        return AGG_SAMPLE;
    TGLError("2D scanner: unknown aggregation function '%s' "
             "(supported: area, weighted.sum, min, max, avg, "
             "exists, size, first, last, sample)",
             name.c_str());
    return AGG_AVG;  // unreachable
}

const char *PMTrackExpression2DVars::func_name(AggFunc f)
{
    switch (f) {
        case AGG_AREA:          return "area";
        case AGG_WEIGHTED_SUM:  return "weighted.sum";
        case AGG_MIN:           return "min";
        case AGG_MAX:           return "max";
        case AGG_AVG:           return "avg";
        case AGG_EXISTS:        return "exists";
        case AGG_SIZE:          return "size";
        case AGG_FIRST:         return "first";
        case AGG_LAST:          return "last";
        case AGG_SAMPLE:        return "sample";
    }
    return "?";
}

void PMTrackExpression2DVars::add_var(const std::string &track_name,
                                      const std::string &func_name_str,
                                      int64_t sshift1, int64_t eshift1,
                                      int64_t sshift2, int64_t eshift2)
{
    AggFunc func = parse_func(func_name_str);

    m_vars.emplace_back();
    TrackVar2D &v = m_vars.back();
    v.name = track_name;
    v.var_name = track_name;
    std::replace(v.var_name.begin(), v.var_name.end(), '.', '_');
    v.func = func;
    v.sshift1 = sshift1;
    v.eshift1 = eshift1;
    v.sshift2 = sshift2;
    v.eshift2 = eshift2;
    v.track = std::make_unique<PMGenomeTrack2D>();
    v.track->init(track_name);  // throws on bad track
}

void PMTrackExpression2DVars::define_py_vars(unsigned size, PMPY &ldict,
                                             bool use_python)
{
    npy_intp dims[1] = {static_cast<npy_intp>(size)};

    for (auto &v : m_vars) {
        if (use_python) {
            v.py_var.assign(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
            if (!v.py_var) {
                TGLError("Failed to allocate value array for var '%s'",
                         v.name.c_str());
            }
            v.values = static_cast<double *>(
                PyArray_DATA(reinterpret_cast<PyArrayObject *>(*v.py_var)));
            if (ldict) {
                PyDict_SetItemString(ldict, v.var_name.c_str(), v.py_var);
            }
        } else {
            v.cpp_values.resize(size);
            v.values = v.cpp_values.data();
        }
        std::fill_n(v.values, size, std::nan(""));
    }
}

void PMTrackExpression2DVars::set_vars_batch(
    const std::vector<GInterval2D> &intervals,
    unsigned start_idx, unsigned count,
    const quadtree::DiagonalBand *band)
{
    if (count == 0) return;

    // Pre-fill the affected slice with NaN - any chrom-pair with no data
    // (or any interval whose occupied_area is 0) must be NaN, and the
    // apply_agg loop skips those slots.
    for (auto &v : m_vars) {
        std::fill_n(v.values + start_idx, count, std::nan(""));
    }

    // Group buffer indices by (chromid1, chromid2). Using a vector of
    // (key -> indices) keeps allocation amortised across batches of
    // similar shape. (For Session 2 we don't optimise this; a hash map
    // is correct.)
    struct PairKey {
        int c1, c2;
        bool operator==(const PairKey &o) const { return c1 == o.c1 && c2 == o.c2; }
    };
    struct PairKeyHash {
        size_t operator()(const PairKey &k) const noexcept {
            return (static_cast<size_t>(static_cast<uint32_t>(k.c1)) << 32) |
                   static_cast<uint32_t>(k.c2);
        }
    };
    std::unordered_map<PairKey, std::vector<unsigned>, PairKeyHash> by_pair;
    by_pair.reserve(8);

    for (unsigned i = 0; i < count; ++i) {
        const GInterval2D &iv = intervals[start_idx + i];
        by_pair[{iv.chromid1, iv.chromid2}].push_back(start_idx + i);
    }

    // For each pair: build (optionally shifted) rects per var, run batch
    // stats, scatter results. Each var may carry independent 2D shifts.
    std::vector<int64_t> rects;
    for (auto &kv : by_pair) {
        const PairKey &key = kv.first;
        const std::vector<unsigned> &idxs = kv.second;

        for (auto &v : m_vars) {
            bool have_data = v.track->set_chrom_pair(key.c1, key.c2);
            if (!have_data) {
                // For exists/size: no data means 0, not NaN (R parity).
                // All other funcs (stats, first/last/sample) correctly stay NaN.
                if (v.func == AGG_EXISTS || v.func == AGG_SIZE) {
                    for (unsigned global_idx : idxs) {
                        v.values[global_idx] = 0.0;
                    }
                }
                continue;
            }

            if (is_object_func(v.func)) {
                // Object-level funcs: query_objects per cell (shifts applied inside).
                apply_agg_objects(v, v.func, intervals, idxs, band);
            } else {
                // Stats-level funcs: one batch query per (var, chrom-pair).
                // Build rects for this var, applying its 2D shifts.
                rects.clear();
                rects.reserve(idxs.size() * 4);
                for (unsigned global_idx : idxs) {
                    const GInterval2D &iv = intervals[global_idx];
                    rects.push_back(iv.start1 + v.sshift1);
                    rects.push_back(iv.start2 + v.sshift2);
                    rects.push_back(iv.end1   + v.eshift1);
                    rects.push_back(iv.end2   + v.eshift2);
                }
                quadtree::BatchQueryStats batch =
                    v.track->query_stats_batch(rects.data(), idxs.size(), band);
                apply_agg(v.func, batch, idxs, v.values);
            }
        }
    }
}

void PMTrackExpression2DVars::pad_tail_with_nan(unsigned start_idx,
                                                unsigned end_idx)
{
    if (end_idx <= start_idx) return;
    for (auto &v : m_vars) {
        std::fill(v.values + start_idx, v.values + end_idx, std::nan(""));
    }
}

void PMTrackExpression2DVars::apply_agg(AggFunc func,
                                        const quadtree::BatchQueryStats &batch,
                                        const std::vector<unsigned> &indices,
                                        double *values)
{
    const size_t n = indices.size();
    for (size_t j = 0; j < n; ++j) {
        int64_t occ = batch.occupied_area[j];
        if (occ == 0) continue;
        unsigned idx = indices[j];
        switch (func) {
            case AGG_AREA:         values[idx] = static_cast<double>(occ);             break;
            case AGG_WEIGHTED_SUM: values[idx] = batch.weighted_sum[j];                 break;
            case AGG_MIN:          values[idx] = batch.min_val[j];                      break;
            case AGG_MAX:          values[idx] = batch.max_val[j];                      break;
            case AGG_AVG:          values[idx] = batch.weighted_sum[j] / static_cast<double>(occ); break;
            // Object funcs handled separately in apply_agg_objects; cannot reach here.
            default: break;
        }
    }
}

void PMTrackExpression2DVars::apply_agg_objects(
    TrackVar2D &var,
    AggFunc func,
    const std::vector<GInterval2D> &intervals,
    const std::vector<unsigned> &indices,
    const quadtree::DiagonalBand *band)
{
    for (unsigned global_idx : indices) {
        const GInterval2D &iv = intervals[global_idx];
        quadtree::QueryObjects qobj =
            var.track->query_objects(iv.start1 + var.sshift1,
                                     iv.start2 + var.sshift2,
                                     iv.end1   + var.eshift1,
                                     iv.end2   + var.eshift2,
                                     band);

        const size_t m = qobj.ids.size();
        double result;

        switch (func) {
            case AGG_EXISTS:
                result = m > 0 ? 1.0 : 0.0;
                break;
            case AGG_SIZE:
                result = static_cast<double>(m);
                break;
            case AGG_FIRST:
                if (m == 0) continue;  // leave NaN
                result = static_cast<double>(qobj.vals[0]);
                break;
            case AGG_LAST:
                if (m == 0) continue;  // leave NaN
                result = static_cast<double>(qobj.vals[m - 1]);
                break;
            case AGG_SAMPLE:
                if (m == 0) continue;  // leave NaN
                {
                    std::uniform_int_distribution<size_t> dist(0, m - 1);
                    result = static_cast<double>(qobj.vals[dist(m_rng)]);
                }
                break;
            default:
                // Stats funcs never reach this helper; treat as NaN.
                continue;
        }

        var.values[global_idx] = result;
    }
}
