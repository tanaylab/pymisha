// pm_liftover_track_2d: C++ port of liftover.gtrack_liftover 2D orchestrator.
// G1.P3.D - matches R GTrackLiftover.cpp:843-984 (RECTS / POINTS only).
//
// Reads a 2D source track via read_source_track_2d_cpp, then maps EVERY source
// rectangle through the chain on both dimensions in a SINGLE batched call per
// dimension (vs R's per-rect inner loop). Cross-product the per-rectangle
// per-dim mapped intervals to produce target rectangles. No aggregation.

#include "pymisha.h"
#include "PMSourceTrack2D.h"
#include "PMChainIntervals.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ChainColumnBuffers {
    std::vector<std::string> chrom;
    std::vector<int64_t> start;
    std::vector<int64_t> end;
    std::vector<int64_t> strand;
    std::vector<std::string> chromsrc;
    std::vector<int64_t> startsrc;
    std::vector<int64_t> endsrc;
    std::vector<int64_t> strandsrc;
    std::vector<int64_t> chain_id;
    std::vector<double>  score;
    int64_t n = 0;
};

PyObject *get_array_local(PyObject *df_dict, const char *key, int dtype)
{
    PyObject *item = PyDict_GetItemString(df_dict, key);
    if (!item) {
        PyErr_Format(PyExc_ValueError, "chain_df_dict missing required key '%s'", key);
        return nullptr;
    }
    PyObject *arr = PyArray_FROM_OTF(item, dtype, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr) return nullptr;
    if (PyArray_NDIM((PyArrayObject *)arr) != 1) {
        PyErr_Format(PyExc_ValueError, "chain_df_dict['%s'] must be 1-D", key);
        Py_DECREF(arr);
        return nullptr;
    }
    return arr;
}

bool extract_chain_dict(PyObject *chain_dict, ChainColumnBuffers &out)
{
    auto pull_int64 = [&](const char *key, std::vector<int64_t> &dest) -> bool {
        PMPY arr(get_array_local(chain_dict, key, NPY_INT64), true);
        if (!arr) return false;
        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
        const int64_t *p = (const int64_t *)PyArray_DATA((PyArrayObject *)*arr);
        dest.assign(p, p + n);
        return true;
    };
    auto pull_double = [&](const char *key, std::vector<double> &dest) -> bool {
        PMPY arr(get_array_local(chain_dict, key, NPY_DOUBLE), true);
        if (!arr) return false;
        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
        const double *p = (const double *)PyArray_DATA((PyArrayObject *)*arr);
        dest.assign(p, p + n);
        return true;
    };
    auto pull_string = [&](const char *key, std::vector<std::string> &dest) -> bool {
        PMPY arr(get_array_local(chain_dict, key, NPY_OBJECT), true);
        if (!arr) return false;
        npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
        PyObject **pp = (PyObject **)PyArray_DATA((PyArrayObject *)*arr);
        dest.clear();
        dest.reserve((size_t)n);
        for (npy_intp i = 0; i < n; ++i) {
            if (!pp[i] || !PyUnicode_Check(pp[i])) {
                PyErr_Format(PyExc_TypeError,
                             "chain_df_dict['%s'][%lld] must be a string",
                             key, (long long)i);
                return false;
            }
            Py_ssize_t slen = 0;
            const char *sstr = PyUnicode_AsUTF8AndSize(pp[i], &slen);
            if (!sstr) return false;
            dest.emplace_back(sstr, (size_t)slen);
        }
        return true;
    };

    if (!pull_string("chrom",     out.chrom))     return false;
    if (!pull_int64 ("start",     out.start))     return false;
    if (!pull_int64 ("end",       out.end))       return false;
    if (!pull_int64 ("strand",    out.strand))    return false;
    if (!pull_string("chromsrc",  out.chromsrc))  return false;
    if (!pull_int64 ("startsrc",  out.startsrc))  return false;
    if (!pull_int64 ("endsrc",    out.endsrc))    return false;
    if (!pull_int64 ("strandsrc", out.strandsrc)) return false;
    if (!pull_int64 ("chain_id",  out.chain_id))  return false;
    if (!pull_double("score",     out.score))     return false;

    out.n = (int64_t)out.chrom.size();
    if ((int64_t)out.start.size()     != out.n ||
        (int64_t)out.end.size()       != out.n ||
        (int64_t)out.strand.size()    != out.n ||
        (int64_t)out.chromsrc.size()  != out.n ||
        (int64_t)out.startsrc.size()  != out.n ||
        (int64_t)out.endsrc.size()    != out.n ||
        (int64_t)out.strandsrc.size() != out.n ||
        (int64_t)out.chain_id.size()  != out.n ||
        (int64_t)out.score.size()     != out.n) {
        PyErr_SetString(PyExc_ValueError,
                        "chain_df_dict columns have mismatched lengths");
        return false;
    }
    return true;
}

PyObject *build_2d_result(
    const std::vector<std::string> &c1,
    const std::vector<std::string> &c2,
    const std::vector<int64_t> &x1,
    const std::vector<int64_t> &y1,
    const std::vector<int64_t> &x2,
    const std::vector<int64_t> &y2,
    const std::vector<double>  &v,
    bool is_points)
{
    npy_intp n = (npy_intp)c1.size();
    PMPY py_c1(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_c2(PyArray_SimpleNew(1, &n, NPY_OBJECT), true);
    PMPY py_x1(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
    PMPY py_y1(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
    PMPY py_x2(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
    PMPY py_y2(PyArray_SimpleNew(1, &n, NPY_INT64),  true);
    PMPY py_v (PyArray_SimpleNew(1, &n, NPY_DOUBLE), true);
    if (!py_c1 || !py_c2 || !py_x1 || !py_y1 || !py_x2 || !py_y2 || !py_v) return nullptr;

    PyObject **c1_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_c1);
    PyObject **c2_out = (PyObject **)PyArray_DATA((PyArrayObject *)*py_c2);
    for (npy_intp i = 0; i < n; ++i) {
        Py_INCREF(Py_None); c1_out[i] = Py_None;
        Py_INCREF(Py_None); c2_out[i] = Py_None;
    }
    int64_t *x1o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_x1);
    int64_t *y1o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_y1);
    int64_t *x2o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_x2);
    int64_t *y2o = (int64_t *)PyArray_DATA((PyArrayObject *)*py_y2);
    double  *vo  = (double *) PyArray_DATA((PyArrayObject *)*py_v);

    for (npy_intp i = 0; i < n; ++i) {
        PyObject *s1 = PyUnicode_FromStringAndSize(c1[i].data(), (Py_ssize_t)c1[i].size());
        PyObject *s2 = PyUnicode_FromStringAndSize(c2[i].data(), (Py_ssize_t)c2[i].size());
        if (!s1 || !s2) { Py_XDECREF(s1); Py_XDECREF(s2); return nullptr; }
        Py_DECREF(c1_out[i]); c1_out[i] = s1;
        Py_DECREF(c2_out[i]); c2_out[i] = s2;
        x1o[i] = x1[i]; y1o[i] = y1[i];
        x2o[i] = x2[i]; y2o[i] = y2[i];
        vo[i]  = v[i];
    }

    PMPY result(PyDict_New(), true);
    if (!result) return nullptr;
    if (PyDict_SetItemString(*result, "chrom1", *py_c1) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "chrom2", *py_c2) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "x1",     *py_x1) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "y1",     *py_y1) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "x2",     *py_x2) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "y2",     *py_y2) < 0) return nullptr;
    if (PyDict_SetItemString(*result, "value",  *py_v)  < 0) return nullptr;
    PyObject *ip = PyBool_FromLong(is_points ? 1 : 0);
    if (PyDict_SetItemString(*result, "is_points", ip) < 0) { Py_DECREF(ip); return nullptr; }
    Py_DECREF(ip);

    result.to_be_stolen();
    return (PyObject *)*result;
}

PyObject *build_empty_2d_result(bool is_points)
{
    std::vector<std::string> es;
    std::vector<int64_t> ei;
    std::vector<double> ed;
    return build_2d_result(es, es, ei, ei, ei, ei, ed, is_points);
}

}  // namespace

// =========================================================================
// pm_liftover_track_2d(src_track_dir, chain_df_dict) -> dict
//
// Mirrors the structure of pm_liftover_track (1D) but for 2D RECTS / POINTS
// source tracks. No aggregation - cross-product of mapped x-dim and y-dim
// intervals per source rectangle.
// =========================================================================

PyObject *pm_liftover_track_2d(PyObject *self, PyObject *args)
{
    const char *src_track_dir;
    PyObject *chain_df_dict;
    if (!PyArg_ParseTuple(args, "sO", &src_track_dir, &chain_df_dict)) {
        return nullptr;
    }
    if (!PyDict_Check(chain_df_dict)) {
        PyErr_SetString(PyExc_TypeError, "chain_df_dict must be a dict");
        return nullptr;
    }

    try {
        // 1) Read source track.
        SourceTrack2DRows src;
        read_source_track_2d_cpp(src_track_dir, src);

        // 2) Extract chain into POD column buffers.
        ChainColumnBuffers chain_buf;
        if (!extract_chain_dict(chain_df_dict, chain_buf)) return nullptr;

        const int64_t N = (int64_t)src.chrom1.size();
        if (N == 0 || chain_buf.n == 0) {
            return build_empty_2d_result(src.is_points);
        }

        // 3) Two batched map_intervals_cpp calls - one per dimension. Each
        //    source rectangle becomes one input row per call. intervalID
        //    in the output preserves source-row provenance.
        ChainRowInput chain_in{
            chain_buf.chrom.data(),  chain_buf.start.data(),
            chain_buf.end.data(),    chain_buf.strand.data(),
            chain_buf.chromsrc.data(), chain_buf.startsrc.data(),
            chain_buf.endsrc.data(),   chain_buf.strandsrc.data(),
            chain_buf.chain_id.data(), chain_buf.score.data(),
            chain_buf.n,
        };

        // X-dim: source chrom = chrom1, start/end = x1/x2.
        SrcRowInput src_x{
            src.chrom1.data(), src.x1.data(), src.x2.data(),
            nullptr,  // no per-row value passed; we attach value externally.
            N,
        };
        // Y-dim: source chrom = chrom2, start/end = y1/y2.
        SrcRowInput src_y{
            src.chrom2.data(), src.y1.data(), src.y2.data(),
            nullptr,
            N,
        };

        MappedOutput mx, my;
        map_intervals_cpp(src_x, chain_in, /*include_metadata=*/false,
                          ClusterStrat::NONE, mx);
        map_intervals_cpp(src_y, chain_in, /*include_metadata=*/false,
                          ClusterStrat::NONE, my);

        // 4) Build per-source-rect index ranges using intervalID.
        //    map_intervals_cpp iterates source rows in input order, so
        //    intervalID is non-decreasing and rows for the same source rect
        //    are contiguous.
        auto build_ranges = [&](const MappedOutput &m,
                                std::vector<int64_t> &starts) {
            starts.assign((size_t)(N + 1), 0);
            for (size_t k = 0; k < m.intervalID.size(); ++k) {
                int64_t i = m.intervalID[k];
                if (i < 0 || i >= N) {
                    throw std::runtime_error("map_intervals_cpp returned out-of-range intervalID");
                }
                starts[(size_t)(i + 1)]++;
            }
            for (size_t i = 1; i <= (size_t)N; ++i) starts[i] += starts[i - 1];
        };

        std::vector<int64_t> mx_starts, my_starts;
        build_ranges(mx, mx_starts);
        build_ranges(my, my_starts);

        // 5) Cross-product per source rectangle.
        std::vector<std::string> out_c1, out_c2;
        std::vector<int64_t>     out_x1, out_y1, out_x2, out_y2;
        std::vector<double>      out_v;
        // Capacity hint - assume average 1 target per src per dim (most chain
        // entries don't split a single src interval).
        out_c1.reserve((size_t)N);
        out_c2.reserve((size_t)N);
        out_x1.reserve((size_t)N);
        out_y1.reserve((size_t)N);
        out_x2.reserve((size_t)N);
        out_y2.reserve((size_t)N);
        out_v .reserve((size_t)N);

        for (int64_t i = 0; i < N; ++i) {
            int64_t mx_lo = mx_starts[(size_t)i];
            int64_t mx_hi = mx_starts[(size_t)(i + 1)];
            int64_t my_lo = my_starts[(size_t)i];
            int64_t my_hi = my_starts[(size_t)(i + 1)];
            if (mx_lo == mx_hi || my_lo == my_hi) continue;
            double v = src.value[(size_t)i];

            for (int64_t a = mx_lo; a < mx_hi; ++a) {
                for (int64_t b = my_lo; b < my_hi; ++b) {
                    out_c1.push_back(mx.chrom[(size_t)a]);
                    out_c2.push_back(my.chrom[(size_t)b]);
                    out_x1.push_back(mx.start[(size_t)a]);
                    out_y1.push_back(my.start[(size_t)b]);
                    out_x2.push_back(mx.end[(size_t)a]);
                    out_y2.push_back(my.end[(size_t)b]);
                    out_v .push_back(v);
                }
            }
        }

        return build_2d_result(out_c1, out_c2, out_x1, out_y1, out_x2, out_y2,
                               out_v, src.is_points);

    } catch (const std::invalid_argument &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    } catch (const std::runtime_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return nullptr;
    }
}
