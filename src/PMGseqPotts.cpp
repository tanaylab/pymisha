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

        std::vector<int8_t> codes;
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
            const char *seq = PyUnicode_AsUTF8AndSize(el, &seq_len);
            if (!seq || (int)seq_len < W) {
                out[r] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const std::string target(seq, seq_len);
            potts_encode(target, codes, nbad);
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
                double f = -std::numeric_limits<double>::infinity();
                double v = -std::numeric_limits<double>::infinity();
                if (use_fwd) f = model.score_codes(c);
                if (use_rev) v = rc.score_codes(c);

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
