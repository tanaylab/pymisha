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

PMTrackExpression2DVars::PMTrackExpression2DVars() = default;

PMTrackExpression2DVars::~PMTrackExpression2DVars() = default;

PMTrackExpression2DVars::AggFunc
PMTrackExpression2DVars::parse_func(const std::string &name)
{
    if (name == "area")          return AGG_AREA;
    if (name == "weighted.sum")  return AGG_WEIGHTED_SUM;
    if (name == "min")           return AGG_MIN;
    if (name == "max")           return AGG_MAX;
    if (name == "avg" || name == "mean") return AGG_AVG;
    TGLError("2D scanner: unknown aggregation function '%s' "
             "(supported: area, weighted.sum, min, max, avg)",
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
    }
    return "?";
}

void PMTrackExpression2DVars::add_var(const std::string &track_name,
                                      const std::string &func_name_str)
{
    AggFunc func = parse_func(func_name_str);

    m_vars.emplace_back();
    TrackVar2D &v = m_vars.back();
    v.name = track_name;
    v.var_name = track_name;
    std::replace(v.var_name.begin(), v.var_name.end(), '.', '_');
    v.func = func;
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

    // For each pair: build rects, run batch stats per var, scatter results.
    std::vector<int64_t> rects;
    for (auto &kv : by_pair) {
        const PairKey &key = kv.first;
        const std::vector<unsigned> &idxs = kv.second;

        rects.clear();
        rects.reserve(idxs.size() * 4);
        for (unsigned global_idx : idxs) {
            const GInterval2D &iv = intervals[global_idx];
            rects.push_back(iv.start1);
            rects.push_back(iv.start2);
            rects.push_back(iv.end1);
            rects.push_back(iv.end2);
        }

        for (auto &v : m_vars) {
            bool have_data = v.track->set_chrom_pair(key.c1, key.c2);
            if (!have_data) {
                continue;  // slots stay NaN
            }
            quadtree::BatchQueryStats batch =
                v.track->query_stats_batch(rects.data(), idxs.size(), band);
            apply_agg(v.func, batch, idxs, v.values);
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
        }
    }
}
