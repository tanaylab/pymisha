// pm_kmer_count_strings: k-mer counts on raw Python strings, gseq_kmer's
// engine. A port of the k-mer half of misha's GseqString.cpp (C_gseq_kmer,
// compute_bounds, score_kmer_over_range and its gap projection), so that
// gseq_kmer counts what gseq.kmer counts.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "pymisha.h"

namespace {

std::string revcomp(const std::string &s)
{
    std::string rc(s.rbegin(), s.rend());
    for (char &c : rc) {
        switch (c) {
        case 'A': c = 'T'; break;
        case 'C': c = 'G'; break;
        case 'G': c = 'C'; break;
        case 'T': c = 'A'; break;
        default: break;
        }
    }
    return rc;
}

// The window starts (0-based, inclusive) a k-mer of width w may take, for the
// 1-based ROI [roi_start, roi_end] extended by extend_val on each side.
void compute_bounds(int L, int w, int roi_start, int roi_end, int extend_val,
                    int &start_min0, int &start_max0)
{
    start_min0 = std::max(0, roi_start - 1 - extend_val);
    start_max0 = std::min(L - w, roi_end - 1 - w + 1 + extend_val);
}

// count positions in [start_min0, start_max0] where the k-mer matches on the
// forward strand (strand >= 0) or its reverse complement does (strand <= 0):
// a position counts once, even when both match. With skip_gaps the windows
// are taken over the sequence with the gap characters removed. frac divides
// by the number of window starts scanned.
double score_kmer_over_range(const std::string &seq, int start_min0, int start_max0,
                             const std::string &kmer, const std::string &kmer_rc,
                             int strand, bool frac, const bool *is_gap)
{
    const int w = (int)kmer.size();
    std::string view;
    int lo = start_min0, hi = start_max0;
    const std::string *s = &seq;
    if (is_gap) {
        // The non-gap bases of the physical range the windows can span.
        if (start_max0 < start_min0)
            return 0.0;
        for (int i = start_min0; i <= start_max0 + w - 1; ++i)
            if (!is_gap[(unsigned char)seq[i]])
                view.push_back(seq[i]);
        if ((int)view.size() < w)
            return 0.0;
        s = &view;
        lo = 0;
        hi = (int)view.size() - w;
    }
    if (hi < lo)
        return 0.0;

    int count = 0;
    for (int p = lo; p <= hi; ++p) {
        const bool fwd = strand >= 0 && s->compare(p, w, kmer) == 0;
        const bool rev = strand <= 0 && s->compare(p, w, kmer_rc) == 0;
        if (fwd || rev)
            ++count;
    }
    return frac ? (double)count / (double)(hi - lo + 1) : (double)count;
}

const int64_t *int64_data(PyObject *obj, Py_ssize_t n, PMPY &holder, const char *what)
{
    holder.assign(PyArray_FROM_OTF(obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
    if (!holder || PyArray_NDIM((PyArrayObject *)*holder) != 1 ||
        PyArray_DIM((PyArrayObject *)*holder, 0) != n) {
        PyErr_Clear();
        TGLError("%s must be an integer array with one entry per sequence", what);
    }
    return (const int64_t *)PyArray_DATA((PyArrayObject *)*holder);
}

} // namespace

/*
 * pm_kmer_count_strings(seqs, kmer, mode, strand, roi_start, roi_end,
 *                       extend, skip_gaps, gap_chars)
 *
 * seqs:      list of str (None gives NaN)
 * kmer:      str, A/C/G/T only (validated by gseq_kmer)
 * mode:      "count" or "frac"
 * strand:    0 = either strand, 1 = forward, -1 = reverse complement
 * roi_start, roi_end: int arrays, 1-based inclusive ROI per sequence
 * extend:    bases the window starts may extend past the ROI (>= 0)
 * skip_gaps: bool
 * gap_chars: str, one gap character per byte
 *
 * Returns: 1D numpy array of float64
 */
PyObject *pm_kmer_count_strings(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr;
    const char *kmer_str = nullptr;
    const char *mode_str = "count";
    int strand = 0;
    PyObject *py_roi_start = nullptr;
    PyObject *py_roi_end = nullptr;
    int extend_val = 0;
    int skip_gaps = 0;
    const char *gap_chars = "";

    if (!PyArg_ParseTuple(args, "OssiOOips", &py_seqs, &kmer_str, &mode_str, &strand,
                          &py_roi_start, &py_roi_end, &extend_val, &skip_gaps, &gap_chars))
        return nullptr;

    try {
        if (!PyList_Check(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs must be a list of strings");
            return nullptr;
        }
        const Py_ssize_t n = PyList_Size(py_seqs);

        std::string kmer(kmer_str);
        std::transform(kmer.begin(), kmer.end(), kmer.begin(), ::toupper);
        const std::string kmer_rc = revcomp(kmer);
        const int w = (int)kmer.size();
        const bool frac = std::string(mode_str) == "frac";

        PMPY start_holder, end_holder;
        const int64_t *roi_start = int64_data(py_roi_start, n, start_holder, "roi_start");
        const int64_t *roi_end = int64_data(py_roi_end, n, end_holder, "roi_end");

        bool is_gap[256] = {false};
        if (skip_gaps)
            for (const char *p = gap_chars; *p; ++p)
                is_gap[(unsigned char)*p] = true;

        npy_intp dims[1] = {(npy_intp)n};
        PMPY py_result(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        double *out = (double *)PyArray_DATA((PyArrayObject *)*py_result);

        for (Py_ssize_t i = 0; i < n; ++i) {
            check_interrupt();
            PyObject *py_seq = PyList_GetItem(py_seqs, i);
            if (!py_seq || !PyUnicode_Check(py_seq)) {
                out[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            Py_ssize_t len = 0;
            const char *raw = py_seq_bytes(py_seq, &len);
            if (!raw)
                return nullptr;
            std::string seq(raw, len);
            std::transform(seq.begin(), seq.end(), seq.begin(),
                           [](char c) { return (char)toupper((unsigned char)c); });
            const int L = (int)seq.size();

            const int rs = std::max<int64_t>(1, roi_start[i]);
            const int re = std::min<int64_t>(L, roi_end[i]);
            if (rs > re || L < w) {
                out[i] = 0.0;
                continue;
            }
            int start_min0, start_max0;
            compute_bounds(L, w, rs, re, extend_val, start_min0, start_max0);
            out[i] = score_kmer_over_range(seq, start_min0, start_max0, kmer, kmer_rc, strand,
                                           frac, skip_gaps ? is_gap : nullptr);
        }

        return_py(py_result);
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_ValueError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
