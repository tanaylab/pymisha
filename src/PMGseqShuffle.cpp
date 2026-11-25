// pm_gseq_shuffle: gseq_shuffle's engine. The shuffle itself is SeqShuffle.cpp,
// shared byte for byte with R misha's gseq.shuffle, so the two give the same
// sequences for the same seed.

#include <cstdint>
#include <string>

#include "SeqShuffle.h"
#include "pymisha.h"

/*
 * pm_gseq_shuffle(seqs, k, n, seed)
 *
 * seqs: list of str or None
 * k:    1..seqshuffle::MAX_K
 * n:    shuffles per sequence, >= 1
 * seed: int64, |seed| <= 2^53 (checked by gseq_shuffle)
 *
 * Returns: a list of len(seqs) * n str (None for a None sequence), the n
 * shuffles of the first sequence first.
 */
PyObject *pm_gseq_shuffle(PyObject *self, PyObject *args)
{
    PyObject *py_seqs = nullptr;
    int k = 2;
    int n = 1;
    long long seed = 0;

    if (!PyArg_ParseTuple(args, "OiiL", &py_seqs, &k, &n, &seed))
        return nullptr;

    try {
        if (!PyList_Check(py_seqs)) {
            PyErr_SetString(PyExc_TypeError, "seqs must be a list of strings");
            return nullptr;
        }
        if (k < 1 || k > seqshuffle::MAX_K) {
            PyErr_Format(PyExc_ValueError, "k must be between 1 and %d", seqshuffle::MAX_K);
            return nullptr;
        }
        if (n < 1) {
            PyErr_SetString(PyExc_ValueError, "n must be a positive integer");
            return nullptr;
        }

        const Py_ssize_t num_seqs = PyList_Size(py_seqs);
        PMPY res(PyList_New(num_seqs * n), true);
        if (!*res)
            return nullptr;
        std::string buf;

        for (Py_ssize_t i = 0; i < num_seqs; ++i) {
            check_interrupt();
            PyObject *py_seq = PyList_GetItem(py_seqs, i);
            if (!py_seq || !PyUnicode_Check(py_seq)) {
                for (int j = 0; j < n; ++j) {
                    Py_INCREF(Py_None);
                    PyList_SET_ITEM(*res, i * n + j, Py_None);
                }
                continue;
            }
            // Latin-1 bytes for a str of code points <= U+00FF, else UTF-8 (see
            // py_seq_bytes); every byte above 127 is left in place, so the result
            // decodes the same way
            const bool latin1 = PyUnicode_KIND(py_seq) == PyUnicode_1BYTE_KIND;
            Py_ssize_t len = 0;
            const char *raw = py_seq_bytes(py_seq, &len);
            if (!raw)
                return nullptr;
            if (len >= (Py_ssize_t)INT32_MAX) {
                PyErr_Format(PyExc_ValueError, "Sequence %zd is too long to shuffle (%zd bases)", i + 1, len);
                return nullptr;
            }

            seqshuffle::Shuffler shuffler(raw, (size_t)len, k);
            seqshuffle::Rng rng((int64_t)seed, (uint64_t)i);
            buf.resize((size_t)len);
            for (int j = 0; j < n; ++j) {
                if (j)
                    check_interrupt();
                shuffler.shuffle(rng, len ? &buf[0] : nullptr);
                PyObject *s = latin1 ? PyUnicode_DecodeLatin1(buf.data(), len, nullptr)
                                     : PyUnicode_DecodeUTF8(buf.data(), len, nullptr);
                if (!s)
                    return nullptr;
                PyList_SET_ITEM(*res, i * n + j, s);
            }
        }

        return_py(res);
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_ValueError, e.msg());
        return nullptr;
    } catch (const std::bad_alloc &) {
        PyErr_NoMemory();
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
