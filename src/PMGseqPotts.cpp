// pm_gseq_potts: score raw Python strings under a pairwise (Potts) energy
// model. No genome database is involved, so this is the sequence-level sibling
// of the potts virtual tracks - it shares their model, their parser and their
// kernel, and only the reduction geometry differs.

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "pymisha.h"
#include "PMPottsParams.h"
#include "PottsModel.h"
#include "util.h" // log_sum_log

namespace {
enum PottsMode { PM_LSE = 0, PM_MAX = 1, PM_POS = 2, PM_COUNT = 3 };
}

/*
 * pm_gseq_potts(seqs, params, mode)
 *
 * seqs:   list of str (DNA sequences, case-insensitive)
 * params: dict as built by _potts_params() - e, J, pairs, intercept,
 *         bidirect, extend, strand, score_thresh
 * mode:   str - "lse", "max", "pos" or "count"
 *
 * Returns: 1D numpy array of float64, one value per sequence.
 */
PyObject *pm_gseq_potts(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr;
    PyObject *py_params = nullptr;
    const char *mode_str = "lse";

    if (!PyArg_ParseTuple(args, "OOs", &py_seqs, &py_params, &mode_str))
        return nullptr;

    try {
        if (!PyList_Check(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs must be a list of strings");
            return nullptr;
        }

        int mode;
        if (strcmp(mode_str, "lse") == 0) mode = PM_LSE;
        else if (strcmp(mode_str, "max") == 0) mode = PM_MAX;
        else if (strcmp(mode_str, "pos") == 0) mode = PM_POS;
        else if (strcmp(mode_str, "count") == 0) mode = PM_COUNT;
        else {
            PyErr_Format(PyExc_ValueError,
                         "Unknown mode '%s'. Must be 'lse', 'max', 'pos' or 'count'", mode_str);
            return nullptr;
        }

        // The SAME parser the vtrack path uses, so the two cannot disagree
        // about what a model means.
        const PMPottsParams pp = PMPottsParams::parse(py_params, "gseq_potts");
        const PottsModel &model = pp.model;
        const PottsModel rc = model.rc();
        const int W = model.width();
        const double thresh = pp.score_thresh;

        // bidirect wins over strand, as gseq_pwm does. When bidirect is False
        // and strand is -1 only the reverse strand is read.
        const bool use_fwd = pp.bidirect || pp.strand_mode != -1;
        const bool use_rev = pp.bidirect || pp.strand_mode == -1;

        const Py_ssize_t n = PyList_Size(py_seqs);
        npy_intp dims[1] = {(npy_intp)n};
        PMPY py_result(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        if (!py_result) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate result array");
            return nullptr;
        }
        double *out = (double *)PyArray_DATA((PyArrayObject *)*py_result);

        std::vector<int8_t> codes, block_codes;
        std::vector<int32_t> nbad;

        for (Py_ssize_t r = 0; r < n; ++r) {
            // Same granularity as pm_pwm_score_strings: one check per sequence
            // is unmeasurable against a whole-sequence Potts scan, and it is
            // the only granularity at which a scan of millions of sequences
            // can be interrupted at all.
            check_interrupt();

            // None in, or a sequence too short to hold one window of the
            // model: no anchor exists at all, so there is nothing to count.
            // NaN, not 0, for every mode including count - a count of zero is
            // what an ambiguous-but-long-enough sequence gets below, where the
            // scan ran and found nothing above the threshold.
            PyObject *el = PyList_GetItem(py_seqs, r);
            if (!el || !PyUnicode_Check(el)) {
                out[r] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            Py_ssize_t seq_len = 0;
            const char *seq = py_seq_bytes(el, &seq_len);
            if (!seq)
                return nullptr;
            if ((int)seq_len < W) {
                out[r] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const std::string target(seq, seq_len);
            potts_encode(target, codes, nbad);
            potts_block_codes(codes, block_codes, 0, codes.size());
            const size_t n_anchor = target.size() - (size_t)W + 1;

            double acc_lse = -std::numeric_limits<double>::infinity();
            bool have_lse = false;
            double best = -std::numeric_limits<double>::infinity();
            long long best_i = -1;
            int best_dir = 1;
            int count = 0;
            bool any = false;

            for (size_t i = 0; i < n_anchor; ++i) {
                // A Potts has no prior, so an ambiguous base leaves the anchor
                // unscorable rather than charged a fallback energy.
                if (nbad[i + (size_t)W] - nbad[i] != 0)
                    continue;
                const int8_t *c = &codes[i];
                const int8_t *bc = &block_codes[i];
                double f = -std::numeric_limits<double>::infinity();
                double v = -std::numeric_limits<double>::infinity();
                if (use_fwd) f = model.score_codes(c, bc);
                if (use_rev) v = rc.score_codes(c, bc);

                // The strand union: log-sum-exp for lse/max/count, maximum for
                // pos, which has to name a strand. Inherited from the pwm
                // family on purpose.
                double u;
                int dir = 1;
                if (use_fwd && use_rev) {
                    if (mode == PM_POS) {
                        u = (v > f) ? v : f;
                        dir = (v > f) ? -1 : 1;
                    } else {
                        u = f;
                        log_sum_log(u, v);
                    }
                } else if (use_fwd) {
                    u = f;
                } else {
                    u = v;
                    dir = -1;
                }

                any = true;
                if (mode == PM_LSE) {
                    // Seeded from the first scorable anchor rather than -inf:
                    // util.h's double log_sum_log() has no isinf() guard,
                    // unlike the float overload, so log_sum_log(-inf, -inf)
                    // computes exp(NaN) = NaN. Every u here is finite by
                    // construction, but seeding this way means the accumulator
                    // can never touch that path.
                    if (!have_lse) {
                        acc_lse = u;
                        have_lse = true;
                    } else {
                        log_sum_log(acc_lse, u);
                    }
                }
                if (u > best) {
                    best = u;
                    best_i = (long long)i;
                    best_dir = dir;
                }
                if (mode == PM_COUNT && u >= thresh)
                    ++count;
            }

            if (mode == PM_COUNT)
                out[r] = (double)count;
            else if (!any)
                out[r] = std::numeric_limits<double>::quiet_NaN();
            else if (mode == PM_LSE)
                out[r] = acc_lse;
            else if (mode == PM_MAX)
                out[r] = best;
            else // PM_POS: 1-based, signed by strand when bidirect
                out[r] = (double)(best_i + 1) * (pp.bidirect ? best_dir : 1);
        }

        return_py(py_result);

    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/*
 * pm_potts_score_codes_cmp(params, codes)  -- TEST ONLY
 *
 * pymisha's counterpart of misha's C_potts_score_codes_cmp: scores each row of
 * `codes` (an (n, W) integer array of base codes 0..3, one window per row)
 * with BOTH PottsModel kernels and returns them side by side, so the tests can
 * check score_codes_naive() against score_codes_blocked() without a DNA
 * string. Also returns use_blocked: which kernel score_codes() runs for the
 * model and for its reverse complement. Not reachable from the pymisha API.
 *
 * Returns: dict with "naive" and "blocked" (float64 arrays of n),
 * "use_blocked" (a tuple of two bools), "delta_max" and "best_delta" (float64
 * arrays of n: the largest single-substitution change from delta_codes() one
 * (position, base) at a time and from best_delta()) and "score4_mismatch"
 * (how many score_codes4() values differ from score_codes() on four
 * consecutive anchors: window i followed by the first codes of window i + 1).
 */
PyObject *pm_potts_score_codes_cmp(PyObject *self, PyObject *args)
{
    PyObject *py_params = nullptr;
    PyObject *py_codes = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &py_params, &py_codes))
        return nullptr;

    try {
        const PMPottsParams pp = PMPottsParams::parse(py_params, "potts kernel equivalence check");
        const PottsModel &model = pp.model;
        const int W = model.width();

        PMPY codes_in(PyArray_FROM_O(py_codes), true);
        if (!codes_in || PyArray_NDIM((PyArrayObject *)*codes_in) != 2 ||
            !PyArray_ISINTEGER((PyArrayObject *)*codes_in)) {
            PyErr_Clear();
            TGLError("codes must be a 2D integer array, one window per row");
        }
        PMPY codes_arr(PyArray_FROM_OTF(codes_in, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
        if (!codes_arr) {
            PyErr_Clear();
            TGLError("codes must be a 2D integer array, one window per row");
        }
        const npy_intp n = PyArray_DIM((PyArrayObject *)*codes_arr, 0);
        if (PyArray_DIM((PyArrayObject *)*codes_arr, 1) != W)
            TGLError("codes has %ld columns but the model width is %d",
                     (long)PyArray_DIM((PyArrayObject *)*codes_arr, 1), W);
        // score_codes_blocked() requires it (false only for W > 256).
        if (!model.blocked_available())
            TGLError("potts kernel equivalence check: blocked kernel unavailable for W = %d", W);
        const int64_t *codes = (const int64_t *)PyArray_DATA((PyArrayObject *)*codes_arr);

        npy_intp dims[1] = {n};
        PMPY py_naive(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        PMPY py_blocked(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        PMPY py_delta_max(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        PMPY py_best_delta(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        double *naive = (double *)PyArray_DATA((PyArrayObject *)*py_naive);
        double *blocked = (double *)PyArray_DATA((PyArrayObject *)*py_blocked);
        double *delta_max = (double *)PyArray_DATA((PyArrayObject *)*py_delta_max);
        double *best_delta = (double *)PyArray_DATA((PyArrayObject *)*py_best_delta);

        // One base past the window, for the trailing block's second base.
        std::vector<int8_t> win((size_t)W + 1), win_block_codes;
        for (npy_intp i = 0; i < n; ++i) {
            check_interrupt();
            const int64_t *row = codes + (size_t)i * W;
            for (int j = 0; j < W; ++j) {
                // A code outside 0..3 is an out-of-bounds table read, not a
                // wrong answer.
                if (row[j] < 0 || row[j] > 3)
                    TGLError("pm_potts_score_codes_cmp: code at row %ld, column %d is %lld, outside 0..3",
                             (long)i, j, (long long)row[j]);
                win[j] = (int8_t)row[j];
            }
            // The base past the window, which the trailing block's tables must
            // ignore: win[0] varies it from window to window.
            win[W] = win[0];
            potts_block_codes(win, win_block_codes, 0, (size_t)W);
            naive[i] = model.score_codes_naive(win.data());
            blocked[i] = model.score_codes_blocked(win_block_codes.data());
            double dm = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < W; ++j)
                for (int b = 0; b < 4; ++b)
                    if (b != win[j]) {
                        const double d = model.delta_codes(win.data(), j, b);
                        if (d > dm)
                            dm = d;
                    }
            delta_max[i] = dm;
            best_delta[i] = model.best_delta(win.data());
        }

        // score_codes4() against score_codes() on four consecutive anchors:
        // window i followed by the first three codes of window i + 1.
        long n4_mismatch = 0;
        {
            std::vector<int8_t> seq((size_t)W + 4), seq_bc;
            for (npy_intp i = 0; i + 1 < n; ++i) {
                const int64_t *col = codes + (size_t)i * W;
                const int64_t *nxt = codes + (size_t)(i + 1) * W;
                for (int j = 0; j < W; ++j)
                    seq[j] = (int8_t)col[j];
                for (int j = 0; j < 4; ++j)
                    seq[W + j] = (int8_t)nxt[j % W];
                potts_block_codes(seq, seq_bc, 0, seq.size());
                double out[4];
                model.score_codes4(seq.data(), seq_bc.data(), out);
                for (int k = 0; k < 4; ++k)
                    if (out[k] != model.score_codes(seq.data() + k, seq_bc.data() + k))
                        ++n4_mismatch;
            }
        }

        PMPY res(Py_BuildValue("{s:O,s:O,s:(OO),s:O,s:O,s:l}", "naive", (PyObject *)py_naive,
                               "blocked", (PyObject *)py_blocked, "use_blocked",
                               model.use_blocked() ? Py_True : Py_False,
                               model.rc().use_blocked() ? Py_True : Py_False,
                               "delta_max", (PyObject *)py_delta_max,
                               "best_delta", (PyObject *)py_best_delta,
                               "score4_mismatch", n4_mismatch),
                 true);
        res.to_be_stolen();
        return (PyObject *)res;
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
