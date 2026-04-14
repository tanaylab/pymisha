// pm_pwm_score_strings: Score raw Python strings using the PWMScorer engine.
// This avoids the genome database dependency and works on arbitrary sequences.

#include "pymisha.h"
#include "PWMScorer.h"
#include "DnaPSSM.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <limits>

namespace {

// Parse PSSM from numpy array — same logic as PMVTrack.cpp parse_pssm
bool parse_pssm_strings(PyObject *obj, DnaPSSM &pssm, double prior) {
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

} // anonymous namespace


/*
 * pm_pwm_score_strings(seqs, pssm, prior, mode, bidirect, strand,
 *                      score_thresh, extend, spat_factor, spat_bin)
 *
 * seqs:         list of str (DNA sequences)
 * pssm:         numpy 2D array (Lx4 or 4xL)
 * prior:        float (default 0.01)
 * mode:         str — "lse", "max", "pos", or "count"
 * bidirect:     bool (default True)
 * strand:       int (default 0) — 0=both, 1=forward, -1=reverse
 * score_thresh: float (default 0.0)
 * extend:       bool (default False)
 * spat_factor:  numpy 1D array or None
 * spat_bin:     int (default 1)
 *
 * Returns: 1D numpy array of float64 scores
 */
PyObject *pm_pwm_score_strings(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr;
    PyObject *py_pssm = nullptr;
    double prior = 0.01;
    const char *mode_str = "lse";
    int bidirect = 1;
    int strand_mode = 0;
    double score_thresh = 0.0;
    int extend = 0;
    PyObject *py_spat_factor = nullptr;
    int spat_bin = 1;

    if (!PyArg_ParseTuple(args, "OO|dsiidiOi",
                          &py_seqs, &py_pssm,
                          &prior, &mode_str, &bidirect, &strand_mode,
                          &score_thresh, &extend, &py_spat_factor, &spat_bin)) {
        return nullptr;
    }

    try {
        // Parse sequences
        if (!PyList_Check(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs must be a list of strings");
            return nullptr;
        }
        Py_ssize_t n_seqs = PyList_Size(py_seqs);

        // Parse PSSM
        DnaPSSM pssm;
        if (!parse_pssm_strings(py_pssm, pssm, prior)) {
            PyErr_SetString(PyExc_ValueError, "pssm must be a numeric array with shape Lx4 or 4xL");
            return nullptr;
        }

        // Set bidirectional mode
        pssm.set_bidirect(bidirect != 0);
        char strand = (bidirect != 0) ? 0 : (char)strand_mode;

        // Map scoring mode
        PWMScorer::ScoringMode pwm_mode = PWMScorer::TOTAL_LIKELIHOOD;
        if (strcmp(mode_str, "lse") == 0) {
            pwm_mode = PWMScorer::TOTAL_LIKELIHOOD;
        } else if (strcmp(mode_str, "max") == 0) {
            pwm_mode = PWMScorer::MAX_LIKELIHOOD;
        } else if (strcmp(mode_str, "pos") == 0) {
            pwm_mode = PWMScorer::MAX_LIKELIHOOD_POS;
        } else if (strcmp(mode_str, "count") == 0) {
            pwm_mode = PWMScorer::MOTIF_COUNT;
        } else {
            PyErr_Format(PyExc_ValueError, "Unknown mode '%s'. Must be 'lse', 'max', 'pos', or 'count'", mode_str);
            return nullptr;
        }

        // Parse spatial factor if provided
        std::vector<float> spat_factor;
        if (py_spat_factor && py_spat_factor != Py_None) {
            PMPY arr(PyArray_FROM_OTF(py_spat_factor, NPY_DOUBLE, NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST), true);
            if (arr && PyArray_NDIM((PyArrayObject *)*arr) == 1) {
                npy_intp n = PyArray_DIM((PyArrayObject *)*arr, 0);
                spat_factor.resize(n);
                for (npy_intp i = 0; i < n; ++i) {
                    double v = *(double *)PyArray_GETPTR1((PyArrayObject *)*arr, i);
                    spat_factor[i] = (float)v;
                }
            }
        }

        // Create scorer (no genome database needed)
        PWMScorer scorer(pssm, pwm_mode, strand, spat_factor, spat_bin,
                         (float)score_thresh, extend != 0);

        // Allocate output array
        npy_intp dims[1] = {(npy_intp)n_seqs};
        PMPY py_result(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        if (!py_result) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate result array");
            return nullptr;
        }
        double *out = (double *)PyArray_DATA((PyArrayObject *)*py_result);

        // Score each sequence
        for (Py_ssize_t i = 0; i < n_seqs; ++i) {
            PyObject *py_seq = PyList_GetItem(py_seqs, i);
            if (!py_seq || !PyUnicode_Check(py_seq)) {
                out[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            Py_ssize_t seq_len = 0;
            const char *seq = PyUnicode_AsUTF8AndSize(py_seq, &seq_len);
            if (!seq || seq_len <= 0) {
                out[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            out[i] = (double)scorer.score_string(seq, (int)seq_len);
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
