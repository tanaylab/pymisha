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
            const char *seq = py_seq_bytes(py_seq, &seq_len);
            if (!seq)
                return nullptr;
            if (seq_len <= 0) {
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


// ---------------------------------------------------------------------------
// Hit positions (gseq_pwm_hits, gintervals_pwm_hits)
// ---------------------------------------------------------------------------

#include "PwmFastScan.h"
#include "PMDb.h"
#include "GenomeSeqFetch.h"
#include <cstdlib>

void convert_py_intervals(PyObject *py_intervals, std::vector<GInterval> &intervals);

namespace {

// A window-skipping scanner over the float weights the PSSM adds, or none
// under MISHA_DISABLE_PWM_SKIP=1 (for differential testing).
void pwm_logp_table(const DnaPSSM &pssm, std::vector<float> &logp)
{
    const int w = pssm.size();
    logp.assign((size_t)w * 4, 0.f);
    for (int c = 0; c < w; ++c)
        for (int b = 0; b < 4; ++b)
            logp[(size_t)c * 4 + b] = pssm[c].get_log_prob_from_code(b);
}

bool pwm_skip_disabled()
{
    const char *env = std::getenv("MISHA_DISABLE_PWM_SKIP");
    return env && std::string(env) == "1";
}

PyObject *hits_dict(const std::vector<std::pair<const char *, std::vector<int64_t>>> &icols,
                    const std::vector<int> &strands, const std::vector<double> &scores)
{
    PMPY dict(PyDict_New(), true);
    const npy_intp n = (npy_intp)strands.size();
    for (const auto &col : icols) {
        PMPY arr(PyArray_SimpleNew(1, (npy_intp *)&n, NPY_INT64), true);
        if (!arr) verror("Failed to allocate result array");
        if (n) std::memcpy(PyArray_DATA((PyArrayObject *)*arr), col.second.data(), n * sizeof(int64_t));
        PyDict_SetItemString(dict, col.first, arr);
    }
    PMPY rstrand(PyArray_SimpleNew(1, (npy_intp *)&n, NPY_INT32), true);
    PMPY rscore(PyArray_SimpleNew(1, (npy_intp *)&n, NPY_DOUBLE), true);
    if (!rstrand || !rscore) verror("Failed to allocate result array");
    if (n) {
        std::memcpy(PyArray_DATA((PyArrayObject *)*rstrand), strands.data(), n * sizeof(int));
        std::memcpy(PyArray_DATA((PyArrayObject *)*rscore), scores.data(), n * sizeof(double));
    }
    PyDict_SetItemString(dict, "strand", rstrand);
    PyDict_SetItemString(dict, "score", rscore);
    dict.to_be_stolen();
    return (PyObject *)dict;
}

} // anonymous namespace

/*
 * pm_pwm_seq_hits(seqs, pssm, prior, bidirect, strand, score_thresh, lo, hi)
 *
 * seqs:   list of str
 * lo, hi: lists of int, per sequence: the 0-based first and last window start
 *         (hi < lo for none)
 *
 * The windows R misha's gseq.pwm(mode = "count") counts (pwm_fast_scan::seq_hits()).
 * Returns a dict of numpy arrays: seq_idx (0-based), start (0-based), strand, score.
 */
PyObject *pm_pwm_seq_hits(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr, *py_pssm = nullptr, *py_lo = nullptr, *py_hi = nullptr;
    double prior = 0.01, score_thresh = 0.0;
    int bidirect = 1, strand = 1;

    if (!PyArg_ParseTuple(args, "OOdiidOO", &py_seqs, &py_pssm, &prior, &bidirect, &strand,
                          &score_thresh, &py_lo, &py_hi))
        return nullptr;

    try {
        if (!PyList_Check(py_seqs) || !PyList_Check(py_lo) || !PyList_Check(py_hi) ||
            PyList_Size(py_lo) != PyList_Size(py_seqs) || PyList_Size(py_hi) != PyList_Size(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs, lo and hi must be lists of the same length");
            return nullptr;
        }
        DnaPSSM pssm;
        if (!parse_pssm_strings(py_pssm, pssm, prior)) {
            PyErr_SetString(PyExc_ValueError, "pssm must be a numeric array with shape Lx4 or 4xL");
            return nullptr;
        }
        const int w = pssm.size();
        std::vector<float> logp;
        pwm_logp_table(pssm, logp);
        pwm_fast_scan::BoundScanner bound;
        if (!pwm_skip_disabled())
            bound.init(logp.data(), w);

        std::vector<int64_t> idx, starts;
        std::vector<int> strands;
        std::vector<double> scores;
        std::vector<pwm_fast_scan::SeqHit> hits;
        const Py_ssize_t n_seqs = PyList_Size(py_seqs);
        for (Py_ssize_t i = 0; i < n_seqs; ++i) {
            PyObject *py_seq = PyList_GetItem(py_seqs, i);
            if (!py_seq || !PyUnicode_Check(py_seq))
                continue;
            Py_ssize_t len = 0;
            const char *seq = py_seq_bytes(py_seq, &len);
            if (!seq)
                return nullptr;
            const long long lo = PyLong_AsLongLong(PyList_GetItem(py_lo, i));
            const long long hi = PyLong_AsLongLong(PyList_GetItem(py_hi, i));
            if (PyErr_Occurred())
                return nullptr;
            hits.clear();
            pwm_fast_scan::seq_hits(&bound, logp.data(), w, seq, (size_t)len, lo, hi,
                                    bidirect != 0, strand, score_thresh, hits);
            for (const pwm_fast_scan::SeqHit &h : hits) {
                idx.push_back(i);
                starts.push_back(h.start);
                strands.push_back(h.strand);
                scores.push_back(h.score);
            }
        }
        return hits_dict({{"seq_idx", idx}, {"start", starts}}, strands, scores);
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/*
 * pm_gintervals_pwm_hits(intervals, pssm, prior, bidirect, strand, score_thresh, extend)
 *
 * intervals: as _df2pymisha gives them
 *
 * The windows a pwm.count virtual track with the same parameters counts over
 * each interval (PWMScorer::collect_hits()). Returns a dict of numpy arrays:
 * interval_idx (0-based row), start, end (genomic), strand, score.
 */
PyObject *pm_gintervals_pwm_hits(PyObject *self, PyObject *args)
{
    PyObject *py_intervals = nullptr, *py_pssm = nullptr;
    double prior = 0.01, score_thresh = 0.0;
    int bidirect = 1, strand = 1, extend = 1;

    if (!PyArg_ParseTuple(args, "OOdiidi", &py_intervals, &py_pssm, &prior, &bidirect, &strand,
                          &score_thresh, &extend))
        return nullptr;

    try {
        PyMisha pymisha(true);

        DnaPSSM pssm;
        if (!parse_pssm_strings(py_pssm, pssm, prior))
            verror("pssm must be a numeric array with shape Lx4 or 4xL");
        // as the pwm virtual tracks build their scorer (PMTrackExpressionVars)
        pssm.set_bidirect(bidirect != 0);
        const char scorer_strand = bidirect ? 0 : (char)strand;

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        GenomeSeqFetch seqfetch;
        seqfetch.set_seqdir(g_pmdb->groot() + "/seq");
        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        PWMScorer scorer(pssm, &seqfetch, extend != 0, PWMScorer::MOTIF_COUNT, scorer_strand,
                         std::vector<float>(), 1, (float)score_thresh);

        // in genomic order, so the sequence is read sequentially
        std::vector<size_t> order(intervals.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const GInterval &x = intervals[a], &y = intervals[b];
            return x.chromid != y.chromid ? x.chromid < y.chromid : x.start < y.start;
        });

        std::vector<int64_t> ids, starts, ends;
        std::vector<int> strands;
        std::vector<double> scores;
        std::vector<PWMScorer::Hit> hits;
        for (size_t k : order) {
            check_interrupt();
            GInterval interval = intervals[k];
            interval.strand = 0;   // a strand column is ignored, as by the virtual tracks
            hits.clear();
            scorer.collect_hits(interval, chromkey, hits);
            for (const PWMScorer::Hit &h : hits) {
                ids.push_back((int64_t)k);
                starts.push_back(h.start);
                ends.push_back(h.end);
                strands.push_back(h.strand);
                scores.push_back(h.score);
            }
        }
        return hits_dict({{"interval_idx", ids}, {"start", starts}, {"end", ends}}, strands, scores);
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return nullptr;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return nullptr;
    }
}
