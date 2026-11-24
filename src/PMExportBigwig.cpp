/*
 * PMExportBigwig.cpp
 *
 * gtrack_export_bigwig: the Python entry points of BigWigWriter (BigWigWriter.h, the same file as in
 * R misha). Python creates the file, appends the records of one batch of chromosomes at a time, as
 * gextract returns them, and finishes it; the writer's state is a bytes object that Python hands
 * from one call to the next.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "BigWigWriter.h"
#include "pymisha.h"

using namespace std;

namespace {

// The threads that compress blocks: at most CONFIG['max_processes'] and 8
unsigned num_threads(const PyMisha &pymisha)
{
	unsigned hw = std::thread::hardware_concurrency();

	return (unsigned)min<int64_t>({ hw ? (int64_t)hw : 1, max<int64_t>(pymisha.max_processes(), 1), 8 });
}

void state_arg(PyObject *py_state, vector<char> &state)
{
	char *buf;
	Py_ssize_t size;

	if (!PyBytes_Check(py_state) || PyBytes_AsStringAndSize(py_state, &buf, &size))
		verror("Invalid bigWig writer state");
	state.assign(buf, buf + size);
}

PyObject *state_value(const vector<char> &state)
{
	return PyBytes_FromStringAndSize(state.data(), state.size());
}

// A 1D array of n int64 (n < 0: any length, set to the array's)
const int64_t *int64_array(PMPY &arr, PyObject *obj, const char *name, npy_intp &n)
{
	arr.assign(PyArray_FROM_OTF(obj, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
	if (!arr || PyArray_NDIM((PyArrayObject *)*arr) != 1)
		verror("%s must be a 1D array of integers", name);
	if (n < 0)
		n = PyArray_SIZE((PyArrayObject *)*arr);
	else if (PyArray_SIZE((PyArrayObject *)*arr) != n)
		verror("chromids, starts, ends and values must have the same length");
	return (const int64_t *)PyArray_DATA((PyArrayObject *)*arr);
}

}

/*
 * pm_bigwig_create(path, names, sizes) -> bytes
 *
 * Creates the file: the chromosomes are names (a list of str, sorted byte by byte) and sizes.
 * Returns the writer's state.
 */
PyObject *pm_bigwig_create(PyObject *self, PyObject *args)
{
	try {
		PyMisha pymisha(false);
		const char *path;
		PyObject *py_names, *py_sizes;

		if (!PyArg_ParseTuple(args, "sOO", &path, &py_names, &py_sizes))
			verror("Invalid arguments to pm_bigwig_create");

		PMPY names(PySequence_Fast(py_names, "names must be a list of strings"), true);
		PMPY sizes;
		npy_intp n = -1;

		if (!names)
			verror("names must be a list of strings");

		const int64_t *size = int64_array(sizes, py_sizes, "sizes", n);

		if (PySequence_Fast_GET_SIZE(*names) != n)
			verror("names and sizes must have the same length");

		vector<BigWigWriter::Chrom> chroms(n);

		for (npy_intp i = 0; i < n; ++i) {
			Py_ssize_t len;
			const char *s = PyUnicode_AsUTF8AndSize(PySequence_Fast_GET_ITEM(*names, i), &len);

			if (!s)
				verror("names must be a list of strings");
			chroms[i].name.assign(s, len);
			chroms[i].size = size[i] < 0 ? 0 : (uint64_t)size[i];
		}

		vector<char> state;

		BigWigWriter::create(path, chroms, state);

		PMPY res(state_value(state), true);
		if (!res)
			verror("Out of memory");
		return_py(res);
	} catch (TGLException &e) {
		PyMisha::handle_error(e.msg());
	} catch (const bad_alloc &e) {
		PyMisha::handle_error("Out of memory");
	} catch (const exception &e) {
		PyMisha::handle_error(e.what());
	}
	return NULL;
}

/*
 * pm_bigwig_append(path, state, chromids, starts, ends, values) -> bytes
 *
 * Appends records: chromids (0-based, into the names of pm_bigwig_create), starts, ends and values,
 * in the order of the file. Returns the new state.
 */
PyObject *pm_bigwig_append(PyObject *self, PyObject *args)
{
	try {
		PyMisha pymisha(false);
		const char *path;
		PyObject *py_state, *py_chromids, *py_starts, *py_ends, *py_values;

		if (!PyArg_ParseTuple(args, "sOOOOO", &path, &py_state, &py_chromids, &py_starts, &py_ends, &py_values))
			verror("Invalid arguments to pm_bigwig_append");

		vector<char> state;
		PMPY chromids, starts, ends;
		npy_intp n = -1;

		state_arg(py_state, state);

		const int64_t *chromid = int64_array(chromids, py_chromids, "chromids", n);
		const int64_t *start = int64_array(starts, py_starts, "starts", n);
		const int64_t *end = int64_array(ends, py_ends, "ends", n);
		PMPY values(PyArray_FROM_OTF(py_values, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);

		if (!values || PyArray_NDIM((PyArrayObject *)*values) != 1)
			verror("values must be a 1D array of numbers");
		if (PyArray_SIZE((PyArrayObject *)*values) != n)
			verror("chromids, starts, ends and values must have the same length");

		BigWigWriter::append(path, state, n, chromid, start, end, (const double *)PyArray_DATA((PyArrayObject *)*values),
		                     num_threads(pymisha));

		PMPY res(state_value(state), true);
		if (!res)
			verror("Out of memory");
		return_py(res);
	} catch (TGLException &e) {
		PyMisha::handle_error(e.msg());
	} catch (const bad_alloc &e) {
		PyMisha::handle_error("Out of memory");
	} catch (const exception &e) {
		PyMisha::handle_error(e.what());
	}
	return NULL;
}

/*
 * pm_bigwig_finish(path, state) -> int
 *
 * Writes the index, the zoom levels, the summary and the header. Returns the number of records.
 */
PyObject *pm_bigwig_finish(PyObject *self, PyObject *args)
{
	try {
		PyMisha pymisha(false);
		const char *path;
		PyObject *py_state;

		if (!PyArg_ParseTuple(args, "sO", &path, &py_state))
			verror("Invalid arguments to pm_bigwig_finish");

		vector<char> state;

		state_arg(py_state, state);

		uint64_t num_items = BigWigWriter::finish(path, state, num_threads(pymisha));
		PMPY res(PyLong_FromUnsignedLongLong(num_items), true);

		if (!res)
			verror("Out of memory");
		return_py(res);
	} catch (TGLException &e) {
		PyMisha::handle_error(e.msg());
	} catch (const bad_alloc &e) {
		PyMisha::handle_error("Out of memory");
	} catch (const exception &e) {
		PyMisha::handle_error(e.what());
	}
	return NULL;
}
