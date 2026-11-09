/*
 * PMExportBedgraph.cpp
 *
 * gtrack_export_bedgraph: the data lines of a bedGraph file, formatted as R misha writes them
 * (its R code wrote paste(chrom, sprintf("%.0f", start), sprintf("%.0f", end), value, sep = "\t");
 * misha >= 5.17.0 builds the same lines in C++, GenomeTrackExportBedgraph.cpp), so that the files
 * are the same byte for byte. The value is formatted as R's as.character() formats a double with
 * the default options (scipen = 0, OutDec = "."): the formatting code is misha's port of R's
 * formatReal() and EncodeRealDrop0().
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include "pymisha.h"

using namespace std;

namespace {

// as.character() of a double, for a finite value: R's formatReal() for one value with
// R_print.digits = DBL_DIG, then EncodeRealDrop0() (src/main/format.c and printutils.c, the same
// in R 4.3 and 4.4). This is the code path of R built with a long double wider than double
// (capabilities("long.double")); where long double is double (arm64 macOS), R and this code compute
// in double alike.

const int R_DIGITS = 15;  // DBL_DIG, as coerceToString() sets R_print.digits
const int KP_MAX = 27;
const int NB = 1000;

// double literals, as in R: 1e23 and above are not exact powers of ten
const long double tbl[] = {
	1e00, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06, 1e07, 1e08, 1e09,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
	1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27
};

void scientific(double x, int *neg, int *kpower, int *nsig, bool *roundingwidens)
{
	double alpha, r;
	int kp;

	if (x == 0.0) {
		*kpower = 0;
		*nsig = 1;
		*neg = 0;
		*roundingwidens = false;
		return;
	}

	if (x < 0.0) {
		*neg = 1;
		r = -x;
	} else {
		*neg = 0;
		r = x;
	}

	kp = (int)floor(log10(r)) - R_DIGITS + 1;
	long double r_prec = r;
	if (abs(kp) <= KP_MAX) {
		if (kp > 0)
			r_prec /= tbl[kp];
		else if (kp < 0)
			r_prec *= tbl[-kp];
	} else
		r_prec /= powl(10.0, (long double)kp);
	if (r_prec < tbl[R_DIGITS - 1]) {
		r_prec *= 10.0;
		kp--;
	}
	alpha = (double)nearbyintl(r_prec);

	*nsig = R_DIGITS;
	for (int j = 1; j <= R_DIGITS; j++) {
		alpha /= 10.0;
		if (alpha == floor(alpha))
			(*nsig)--;
		else
			break;
	}
	if (*nsig == 0) {
		*nsig = 1;
		kp += 1;
	}
	*kpower = kp + R_DIGITS - 1;

	int rgt = R_DIGITS - *kpower;
	rgt = rgt < 0 ? 0 : rgt > KP_MAX ? KP_MAX : rgt;
	double fuzz = 0.5 / (double)tbl[rgt];
	*roundingwidens = *kpower > 0 && *kpower <= KP_MAX && r < tbl[*kpower] - fuzz;
}

// Appends x (finite) as as.character(x) writes it with scipen = 0 and OutDec = "."
void append_double(string &out, double x)
{
	const int scipen = 0;
	int neg, kpower, nsig;
	bool roundingwidens;

	scientific(x, &neg, &kpower, &nsig, &roundingwidens);

	int left = kpower + 1;
	if (roundingwidens)
		left--;
	int sleft = neg + ((left <= 0) ? 1 : left);
	int rgt = nsig - left;
	int mxsl = sleft;

	if (left < 0)
		mxsl = 1 + neg;
	if (rgt < 0)
		rgt = 0;
	int wF = mxsl + rgt + (rgt != 0);

	int e = (left > 100 || left <= -99) ? 2 : 1;
	int d = nsig - 1;
	int w = neg + (d > 0) + d + 4 + e;
	if (wF <= w + scipen) {
		e = 0;
		d = rgt;
		w = wF;
	}
	w = min(w, NB - 1);

	char buf[NB];
	if (x == 0.0)
		x = 0.0;  // no "-0"
	if (e)
		snprintf(buf, NB, d ? "%#*.*e" : "%*.*e", w, d, x);
	else
		snprintf(buf, NB, "%*.*f", w, d, x);
	buf[NB - 1] = '\0';

	// drop the trailing zeros of the fraction, as EncodeRealDrop0() does
	for (char *p = buf; *p; p++) {
		if (*p == '.') {
			char *replace = p++;
			while ('0' <= *p && *p <= '9') {
				if (*(p++) != '0')
					replace = p;
			}
			if (replace != p) {
				while ((*(replace++) = *(p++)))
					;
			}
			break;
		}
	}

	out += buf;
}

// as.character() of a double that is not NaN
void append_value(string &out, double x)
{
	if (std::isinf(x))
		out += x > 0 ? "Inf" : "-Inf";
	else
		append_double(out, x);
}

// sprintf("%.0f", x) of R for a whole number
void append_coord(string &out, int64_t v)
{
	char buf[32];
	char *p = buf + sizeof(buf);
	bool negative = v < 0;
	uint64_t u = negative ? -(uint64_t)v : (uint64_t)v;

	do {
		*--p = '0' + u % 10;
		u /= 10;
	} while (u);
	if (negative)
		*--p = '-';
	out.append(p, buf + sizeof(buf) - p);
}

}

// pm_bedgraph_lines(chrom_codes, chrom_names, starts, ends, values) -> bytes
//
// The bedGraph lines ("chrom\tstart\tend\tvalue\n") of the rows whose value is not NaN, in the
// order given. chrom_codes (int32) index chrom_names (a list of strings); starts and ends are
// int64 and values double arrays of the same length.
PyObject *pm_bedgraph_lines(PyObject *self, PyObject *args)
{
	try {
		PyMisha pymisha(false);
		PyObject *py_codes, *py_names, *py_starts, *py_ends, *py_values;

		if (!PyArg_ParseTuple(args, "OOOOO", &py_codes, &py_names, &py_starts, &py_ends, &py_values))
			verror("Invalid arguments to pm_bedgraph_lines");

		PMPY codes(PyArray_FROM_OTF(py_codes, NPY_INT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY starts(PyArray_FROM_OTF(py_starts, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY ends(PyArray_FROM_OTF(py_ends, NPY_INT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY values(PyArray_FROM_OTF(py_values, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST), true);
		PMPY names(PySequence_Fast(py_names, "chrom_names must be a list of strings"), true);

		if (!codes || !starts || !ends || !values || !names)
			verror("Invalid arguments to pm_bedgraph_lines");

		npy_intp n = PyArray_SIZE((PyArrayObject *)*values);
		if (PyArray_NDIM((PyArrayObject *)*codes) != 1 || PyArray_NDIM((PyArrayObject *)*starts) != 1 ||
		    PyArray_NDIM((PyArrayObject *)*ends) != 1 || PyArray_NDIM((PyArrayObject *)*values) != 1 ||
		    PyArray_SIZE((PyArrayObject *)*codes) != n || PyArray_SIZE((PyArrayObject *)*starts) != n ||
		    PyArray_SIZE((PyArrayObject *)*ends) != n)
			verror("pm_bedgraph_lines: chrom_codes, starts, ends and values must be 1D arrays of the same length");

		vector<string> chrom_names;
		for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(*names); ++i) {
			Py_ssize_t len;
			const char *s = PyUnicode_AsUTF8AndSize(PySequence_Fast_GET_ITEM(*names, i), &len);
			if (!s)
				verror("pm_bedgraph_lines: chrom_names must be a list of strings");
			chrom_names.push_back(string(s, len));
		}

		const int32_t *code = (const int32_t *)PyArray_DATA((PyArrayObject *)*codes);
		const int64_t *start = (const int64_t *)PyArray_DATA((PyArrayObject *)*starts);
		const int64_t *end = (const int64_t *)PyArray_DATA((PyArrayObject *)*ends);
		const double *value = (const double *)PyArray_DATA((PyArrayObject *)*values);
		string out;

		out.reserve(n * 40);
		for (npy_intp i = 0; i < n; ++i) {
			if (std::isnan(value[i]))
				continue;
			if (code[i] < 0 || code[i] >= (int32_t)chrom_names.size())
				verror("pm_bedgraph_lines: chromosome code %d out of range", code[i]);
			out += chrom_names[code[i]];
			out += '\t';
			append_coord(out, start[i]);
			out += '\t';
			append_coord(out, end[i]);
			out += '\t';
			append_value(out, value[i]);
			out += '\n';
			if (!(i % 100000))
				check_interrupt();
		}

		PMPY res(PyBytes_FromStringAndSize(out.data(), out.size()), true);
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
