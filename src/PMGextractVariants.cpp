// gextract_variants: which bases each sequence vtrack in an expression reads,
// and the expression's value with one base substituted per row. The pymisha
// counterpart of misha's GenomeTrackExtractVariants.cpp.

#include <cstdint>
#include <string>
#include <vector>

#include "pymisha.h"
#include "PMTrackExpressionScanner.h"
#include "PMTrackExpressionVars.h"

extern void convert_py_intervals(PyObject *py_intervals, std::vector<GInterval> &intervals);

/*
 * pm_gvariants_reach(expr, vtracks)
 *
 * Returns a list of (lo, hi), one per sequence vtrack of `expr`: a bin
 * [s, e) of that vtrack reads [s + lo, e + hi).
 */
PyObject *pm_gvariants_reach(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *expr = nullptr;
        PyObject *py_vtracks = nullptr;
        if (!PyArg_ParseTuple(args, "sO", &expr, &py_vtracks))
            verror("Invalid arguments to pm_gvariants_reach");

        PMTrackExpressionVars vars;
        std::vector<std::string> exprs4compile;
        vars.parse_exprs(std::vector<std::string>(1, expr), exprs4compile, py_vtracks);
        const std::vector<std::pair<int64_t, int64_t>> offs = vars.seq_read_offsets();

        PMPY out(PyList_New(offs.size()), true);
        for (size_t i = 0; i < offs.size(); ++i)
            PyList_SetItem(out, i, Py_BuildValue("(LL)", (long long)offs[i].first, (long long)offs[i].second));
        out.to_be_stolen();
        return (PyObject *)out;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

/*
 * pm_gextract_variants(expr, rows, sub_pos, sub_base, vtracks)
 *
 * The value of `expr` on each row [start, end) of `rows`, with base
 * sub_base[i] (0..3 = A, C, G, T; -1 = none) at sub_pos[i] while row i is
 * computed. Rows must be sorted by chromosome and start.
 */
PyObject *pm_gextract_variants(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *expr = nullptr;
        PyObject *py_rows = nullptr;
        PyObject *py_pos = nullptr;
        PyObject *py_base = nullptr;
        PyObject *py_vtracks = nullptr;
        if (!PyArg_ParseTuple(args, "sOOO|O", &expr, &py_rows, &py_pos, &py_base, &py_vtracks))
            verror("Invalid arguments to pm_gextract_variants");

        std::vector<GInterval> rows;
        convert_py_intervals(py_rows, rows);
        const size_t n = rows.size();

        PMPY pos_arr(PyArray_FROM_OTF(py_pos, NPY_INT64, NPY_ARRAY_IN_ARRAY), true);
        PMPY base_arr(PyArray_FROM_OTF(py_base, NPY_INT64, NPY_ARRAY_IN_ARRAY), true);
        if (!pos_arr || !base_arr || PyArray_NDIM((PyArrayObject *)*pos_arr) != 1 ||
            PyArray_NDIM((PyArrayObject *)*base_arr) != 1 ||
            (size_t)PyArray_DIM((PyArrayObject *)*pos_arr, 0) != n ||
            (size_t)PyArray_DIM((PyArrayObject *)*base_arr, 0) != n) {
            PyErr_Clear();
            verror("Invalid rows: chroms, starts, ends, positions and bases must have the same length");
        }
        const int64_t *pos = (const int64_t *)PyArray_DATA((PyArrayObject *)*pos_arr);
        const int64_t *base = (const int64_t *)PyArray_DATA((PyArrayObject *)*base_arr);

        static const char BASES[4] = {'A', 'C', 'G', 'T'};
        std::vector<SeqSubstitution> subs(n);
        for (size_t i = 0; i < n; ++i) {
            if (i && (rows[i].chromid < rows[i - 1].chromid ||
                      (rows[i].chromid == rows[i - 1].chromid && rows[i].start < rows[i - 1].start)))
                verror("Rows must be sorted by chromosome and start");
            const int64_t b = base[i];
            if (b < -1 || b > 3)
                verror("Invalid substituted base code %lld", (long long)b);
            if (b >= 0) {
                subs[i].chromid = rows[i].chromid;
                subs[i].pos = pos[i];
                subs[i].base = BASES[b];
            }
        }

        npy_intp dims[1] = {(npy_intp)n};
        PMPY out(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        double *vals = (double *)PyArray_DATA((PyArrayObject *)*out);
        size_t i = 0;
        if (n) {
            // Iterator policy -1 walks the rows as given, repeats included.
            PMTrackExprScanner scanner;
            scanner.report_progress(false);
            scanner.set_row_substitutions(&subs);
            for (scanner.begin(std::vector<std::string>(1, expr), PMTrackExprScanner::REAL_T, rows, -1, py_vtracks);
                 !scanner.isend(); scanner.next()) {
                if (i >= n)
                    verror("Internal error: more than %zu rows evaluated", n);
                vals[i++] = scanner.vdouble(0);
            }
        }
        if (i != n)
            verror("Internal error: %zu rows evaluated, %zu expected", i, n);

        out.to_be_stolen();
        return (PyObject *)out;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}
