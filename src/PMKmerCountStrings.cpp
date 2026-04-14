// pm_kmer_count_strings: Count k-mer occurrences in raw Python strings.
// This avoids the genome database dependency and works on arbitrary sequences.

#include "pymisha.h"
#include "KmerCounter.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <limits>
#include <cctype>

/*
 * pm_kmer_count_strings(seqs, kmer, mode, strand)
 *
 * seqs:    list of str (DNA sequences)
 * kmer:    str (the k-mer pattern)
 * mode:    str — "count" or "frac"
 * strand:  int — 0=both, 1=forward, -1=reverse
 *
 * Returns: 1D numpy array of float64 scores
 */
PyObject *pm_kmer_count_strings(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr;
    const char *kmer_str = nullptr;
    const char *mode_str = "count";
    int strand_mode = 0;

    if (!PyArg_ParseTuple(args, "Os|si",
                          &py_seqs, &kmer_str,
                          &mode_str, &strand_mode)) {
        return nullptr;
    }

    try {
        // Validate seqs
        if (!PyList_Check(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs must be a list of strings");
            return nullptr;
        }
        Py_ssize_t n_seqs = PyList_Size(py_seqs);

        // Validate kmer
        if (!kmer_str || kmer_str[0] == '\0') {
            PyErr_SetString(PyExc_ValueError, "kmer must be non-empty");
            return nullptr;
        }
        for (const char *p = kmer_str; *p; ++p) {
            char c = (char)std::toupper((unsigned char)*p);
            if (c != 'A' && c != 'C' && c != 'G' && c != 'T') {
                PyErr_Format(PyExc_ValueError,
                    "kmer must contain only A, C, G, T characters, got '%c'", *p);
                return nullptr;
            }
        }

        // Map mode
        KmerCounter::CountMode mode = KmerCounter::SUM;
        if (strcmp(mode_str, "count") == 0) {
            mode = KmerCounter::SUM;
        } else if (strcmp(mode_str, "frac") == 0) {
            mode = KmerCounter::FRACTION;
        } else {
            PyErr_Format(PyExc_ValueError,
                "Unknown mode '%s'. Must be 'count' or 'frac'", mode_str);
            return nullptr;
        }

        // Validate strand
        if (strand_mode != 0 && strand_mode != 1 && strand_mode != -1) {
            PyErr_Format(PyExc_ValueError,
                "strand must be -1, 0, or 1, got %d", strand_mode);
            return nullptr;
        }

        // Create counter (no genome database needed)
        KmerCounter counter(std::string(kmer_str), mode, (char)strand_mode);

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
                out[i] = 0.0;
                continue;
            }

            out[i] = counter.count_string(seq, (int)seq_len);
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
