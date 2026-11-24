// pm_seq_onehot / pm_seq_onehot_strings: one-hot arrays of genomic windows, read straight from
// the sequence store, and of byte strings, for gseq_onehot and gextract_windows. The encoding
// itself is in SeqOneHot.h, shared verbatim with R misha (src/GenomeSeqOneHot.cpp there).

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "pymisha.h"
#include "PMDb.h"
#include "GenomeSeqFetch.h"
#include "SeqOneHot.h"

using namespace std;

namespace {

// out: a C-contiguous, writeable numpy array of shape (N, 4, L); every element is written.
PyArrayObject *check_out(PyObject *py_out, int64_t &num, int64_t &width)
{
	if (!PyArray_Check(py_out))
		verror("out must be a numpy array");
	PyArrayObject *out = (PyArrayObject *)py_out;
	if (PyArray_NDIM(out) != 3 || PyArray_DIM(out, 1) != 4)
		verror("out must have shape (N, 4, L)");
	if (!PyArray_IS_C_CONTIGUOUS(out) || !PyArray_ISWRITEABLE(out) || !PyArray_ISNOTSWAPPED(out))
		verror("out must be a writeable C-contiguous array in native byte order");
	num = PyArray_DIM(out, 0);
	width = PyArray_DIM(out, 2);
	return out;
}

// Calls f with a typed pointer to the data of out.
template <typename F>
void dispatch(PyArrayObject *out, F f)
{
	void *data = PyArray_DATA(out);
	switch (PyArray_TYPE(out)) {
	case NPY_BOOL:
	case NPY_UINT8: f((uint8_t *)data); break;
	case NPY_INT8: f((int8_t *)data); break;
	case NPY_INT16: f((int16_t *)data); break;
	case NPY_INT32: f((int32_t *)data); break;
	case NPY_INT64: f((int64_t *)data); break;
	case NPY_FLOAT32: f((float *)data); break;
	case NPY_FLOAT64: f((double *)data); break;
	default: verror("Unsupported dtype for the one-hot array");
	}
}

}  // namespace

// pm_seq_onehot(chrom_names, chrom_codes, starts, revcomp, out, pad)
//
// Window i is [starts[i], starts[i] + L) of chromosome chrom_names[chrom_codes[i]], reverse
// complemented if revcomp[i]; it is encoded into out[i] (see check_out). chrom_codes are int32,
// starts int64 and revcomp uint8 arrays of length N. With pad false, a window that runs past
// either end of its chromosome is an error; with pad true its outside positions are zero.
PyObject *pm_seq_onehot(PyObject *self, PyObject *args)
{
	try {
		PyMisha pymisha(true);
		PyObject *py_names, *py_codes, *py_starts, *py_revcomp, *py_out;
		int pad = 0;

		if (!PyArg_ParseTuple(args, "OOOOOp", &py_names, &py_codes, &py_starts, &py_revcomp, &py_out, &pad))
			verror("Invalid arguments to pm_seq_onehot");

		int64_t num, width;
		PyArrayObject *out = check_out(py_out, num, width);

		PMPY codes(PyArray_FROM_OTF(py_codes, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY starts(PyArray_FROM_OTF(py_starts, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY revcomp(PyArray_FROM_OTF(py_revcomp, NPY_UINT8, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY names(PySequence_Fast(py_names, "chrom_names must be a list of strings"), true);
		if (!codes || !starts || !revcomp || !names)
			verror("Invalid arguments to pm_seq_onehot");
		if (PyArray_SIZE((PyArrayObject *)*codes) != num || PyArray_SIZE((PyArrayObject *)*starts) != num ||
			PyArray_SIZE((PyArrayObject *)*revcomp) != num)
			verror("pm_seq_onehot: chrom_codes, starts and revcomp must have one element per window");

		const GenomeChromKey &chromkey = g_pmdb->chromkey();
		vector<int> name2id;
		for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(*names); ++i) {
			const char *s = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(*names, i));
			if (!s)
				verror("pm_seq_onehot: chrom_names must be a list of strings");
			name2id.push_back(chromkey.chrom2id(s));
		}

		const int32_t *code = (const int32_t *)PyArray_DATA((PyArrayObject *)*codes);
		const int64_t *start = (const int64_t *)PyArray_DATA((PyArrayObject *)*starts);
		const uint8_t *rc = (const uint8_t *)PyArray_DATA((PyArrayObject *)*revcomp);

		vector<int> chromids(num);
		for (int64_t i = 0; i < num; ++i) {
			if (code[i] < 0 || code[i] >= (int32_t)name2id.size())
				verror("pm_seq_onehot: chrom code out of range");
			chromids[i] = name2id[code[i]];
			if (!pad && (start[i] < 0 || start[i] + width > (int64_t)chromkey.get_chrom_size(chromids[i])))
				verror("Window (%s, %lld, %lld) runs past the end of its chromosome (use pad=True to fill with zeros)",
					   chromkey.id2chrom(chromids[i]).c_str(), (long long)start[i], (long long)(start[i] + width));
		}

		GenomeSeqFetch seqfetch;
		seqfetch.set_seqdir(g_pmdb->groot() + "/seq");

		seqonehot::Strides s = { 4 * width, width, 1 };
		auto interrupt = []() { check_interrupt(); };
		dispatch(out, [&](auto *data) {
			seqonehot::encode_windows(seqfetch, chromkey, chromids.data(), start, rc, num, width, 0, data, s, interrupt);
		});

		Py_RETURN_NONE;
	} catch (TGLException &e) {
		PyMisha::handle_error(e.msg());
		return NULL;
	} catch (const std::bad_alloc &e) {
		PyMisha::handle_error("Out of memory");
		return NULL;
	}
}

// pm_seq_onehot_strings(seqs, revcomp, out)
//
// seqs: a list of bytes (or str, taken as Latin-1) objects of length L each; revcomp: a bool.
PyObject *pm_seq_onehot_strings(PyObject *self, PyObject *args)
{
	try {
		PyMisha pymisha(false);
		PyObject *py_seqs, *py_out;
		int revcomp = 0;

		if (!PyArg_ParseTuple(args, "OpO", &py_seqs, &revcomp, &py_out))
			verror("Invalid arguments to pm_seq_onehot_strings");

		int64_t num, width;
		PyArrayObject *out = check_out(py_out, num, width);
		PMPY seqs(PySequence_Fast(py_seqs, "seqs must be a list of strings"), true);
		if (!seqs || PySequence_Fast_GET_SIZE(*seqs) != num)
			verror("pm_seq_onehot_strings: seqs must hold one sequence per row of out");

		seqonehot::Strides s = { 4 * width, width, 1 };
		string latin1;
		for (int64_t i = 0; i < num; ++i) {
			PyObject *item = PySequence_Fast_GET_ITEM(*seqs, i);
			const char *seq = NULL;
			Py_ssize_t len = 0;
			if (PyBytes_Check(item)) {
				seq = PyBytes_AS_STRING(item);
				len = PyBytes_GET_SIZE(item);
			} else if (PyUnicode_Check(item)) {
				// one byte per character, as gseq_extract returns the .seq bytes
				PMPY b(PyUnicode_AsLatin1String(item), true);
				if (!b) {
					PyErr_Clear();
					verror("Sequence %lld has characters outside Latin-1", (long long)(i + 1));
				}
				latin1.assign(PyBytes_AS_STRING(*b), PyBytes_GET_SIZE(*b));
				seq = latin1.data();
				len = (Py_ssize_t)latin1.size();
			} else
				verror("Sequence %lld is not a string", (long long)(i + 1));
			if (len != width)
				verror("Sequence %lld has %lld bases, expected %lld (all sequences must have the same length)",
					   (long long)(i + 1), (long long)len, (long long)width);
			dispatch(out, [&](auto *data) { seqonehot::encode(seq, width, revcomp != 0, i, data, s); });
			if (!(i % 4096))
				check_interrupt();
		}

		Py_RETURN_NONE;
	} catch (TGLException &e) {
		PyMisha::handle_error(e.msg());
		return NULL;
	} catch (const std::bad_alloc &e) {
		PyMisha::handle_error("Out of memory");
		return NULL;
	}
}
