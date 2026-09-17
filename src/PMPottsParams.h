#ifndef PM_POTTS_PARAMS_H_
#define PM_POTTS_PARAMS_H_

#include <string>
#include <vector>

#include <Python.h>

#include "PottsModel.h"
#include "pymisha.h"

// The one Python-dict -> PottsModel conversion, shared by pm_gseq_potts() and
// PMTrackExpressionVars' potts branch. Two copies of the index arithmetic below
// would be two chances to transpose a coupling block, and the symptom of that
// is a plausible wrong number rather than a crash.
//
// `e`, `J`, `pairs` and `intercept` arrive pre-validated and pre-shaped from
// _potts_params() in Python. This still checks types and shapes, because a
// dict handed across the C boundary is a trust boundary, but it does not
// re-derive anything.
//
// The layouts differ from misha's R-side parser, which reads column-major R
// matrices and transposes them here. numpy hands over C-order arrays, so both
// tables are already in the order PottsModel wants and are read straight
// through:
//   e     - (W, 4) double, row i = position i, columns A, C, G, T.
//   J     - (npair, 16) double, row k = J_k flattened in FORTRAN order, so
//           element [a][b] sits at b * 4 + a. _potts_params() does that
//           flattening; getting it wrong transposes every coupling.
//   pairs - (npair, 2) integer, 1-BASED, lower position first.
struct PMPottsParams {
    PottsModel model;
    bool bidirect = true;
    bool extend_flag = true;
    char strand_mode = 1;
    double score_thresh = 0.0;

    static PyObject *dict_get(PyObject *dict, const char *key) {
        if (!dict || !PyDict_Check(dict)) return nullptr;
        PyObject *v = PyDict_GetItemString(dict, key);
        return (v == Py_None) ? nullptr : v;
    }

    static PMPottsParams parse(PyObject *spec, const std::string &who) {
        PMPottsParams params;

        if (!spec || !PyDict_Check(spec))
            TGLError("%s: potts parameters must be a dict", who.c_str());

        PyObject *py_e = dict_get(spec, "e");
        if (!py_e)
            TGLError("%s: potts parameters require an 'e' matrix", who.c_str());

        PMPY e_arr(PyArray_FROM_OTF(py_e, NPY_DOUBLE,
                                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST),
                   true);
        if (!e_arr || PyArray_NDIM((PyArrayObject *)*e_arr) != 2) {
            PyErr_Clear();
            TGLError("%s: 'e' must be a 2-D numeric array", who.c_str());
        }
        const npy_intp W = PyArray_DIM((PyArrayObject *)*e_arr, 0);
        if (PyArray_DIM((PyArrayObject *)*e_arr, 1) != 4)
            TGLError("%s: 'e' must have 4 columns (A, C, G, T)", who.c_str());
        if (W < 1)
            TGLError("%s: 'e' has no rows", who.c_str());

        int npair = 0;
        std::vector<int> p1, p2;
        PyObject *py_pairs = dict_get(spec, "pairs");
        PMPY pairs_arr;
        if (py_pairs) {
            pairs_arr.assign(PyArray_FROM_OTF(py_pairs, NPY_INT64,
                                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST),
                             true);
            if (!pairs_arr || PyArray_NDIM((PyArrayObject *)*pairs_arr) != 2) {
                PyErr_Clear();
                TGLError("%s: 'pairs' must be a 2-D integer array", who.c_str());
            }
            if (PyArray_DIM((PyArrayObject *)*pairs_arr, 1) != 2)
                TGLError("%s: 'pairs' must have 2 columns", who.c_str());
            npair = (int)PyArray_DIM((PyArrayObject *)*pairs_arr, 0);
        }

        std::vector<double> j_row((size_t)npair * 16);
        if (npair) {
            PyObject *py_J = dict_get(spec, "J");
            if (!py_J)
                TGLError("%s: 'J' must be a %d x 16 numeric array", who.c_str(), npair);
            PMPY j_arr(PyArray_FROM_OTF(py_J, NPY_DOUBLE,
                                        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST),
                       true);
            if (!j_arr || PyArray_NDIM((PyArrayObject *)*j_arr) != 2 ||
                PyArray_DIM((PyArrayObject *)*j_arr, 0) != npair ||
                PyArray_DIM((PyArrayObject *)*j_arr, 1) != 16) {
                PyErr_Clear();
                TGLError("%s: 'J' must be a %d x 16 numeric array", who.c_str(), npair);
            }
            const double *jp = (const double *)PyArray_DATA((PyArrayObject *)*j_arr);
            j_row.assign(jp, jp + (size_t)npair * 16);

            const int64_t *pp = (const int64_t *)PyArray_DATA((PyArrayObject *)*pairs_arr);
            p1.resize(npair);
            p2.resize(npair);
            for (int k = 0; k < npair; ++k) {
                // Tested on the 1-BASED values, before the decrement, so the
                // range check sees what the caller wrote rather than a value
                // the decrement has already moved.
                const int64_t a = pp[(size_t)k * 2];     // column 1
                const int64_t b = pp[(size_t)k * 2 + 1]; // column 2
                if (a < 1 || b > (int64_t)W || a >= b)
                    TGLError("%s: 'pairs' row %d is out of 1..%d or not ascending",
                             who.c_str(), k + 1, (int)W);
                p1[k] = (int)a - 1; // 1-based -> 0-based
                p2[k] = (int)b - 1;
            }
        }

        double intercept = 0.0;
        if (PyObject *py_i = dict_get(spec, "intercept")) {
            PMPY tmp(PyNumber_Float(py_i), true);
            if (!tmp) {
                PyErr_Clear();
                TGLError("%s: 'intercept' must be a single number", who.c_str());
            }
            intercept = PyFloat_AsDouble(tmp);
        }

        // PottsModel's constructor never dereferences j_rowmajor when p1 is
        // empty, so j_row.data() is safe to pass unconditionally here even
        // though an empty vector's data() is not guaranteed non-null.
        params.model = PottsModel((int)W, intercept,
                                  (const double *)PyArray_DATA((PyArrayObject *)*e_arr),
                                  j_row.data(), p1, p2);

        if (PyObject *o = dict_get(spec, "bidirect"))
            params.bidirect = PyObject_IsTrue(o) == 1;
        if (PyObject *o = dict_get(spec, "extend"))
            params.extend_flag = PyObject_IsTrue(o) == 1;
        if (PyObject *o = dict_get(spec, "strand")) {
            PMPY tmp(PyNumber_Long(o), true);
            if (!tmp) {
                PyErr_Clear();
                TGLError("%s: 'strand' must be 1 or -1", who.c_str());
            }
            const long sv = PyLong_AsLong(tmp);
            if (sv != 1 && sv != -1)
                TGLError("%s: 'strand' must be 1 or -1", who.c_str());
            params.strand_mode = (char)sv;
        }
        // Python spells it score_thresh; the dotted R spelling is accepted too,
        // as it is everywhere else a vtrack spec crosses this boundary.
        PyObject *thr = dict_get(spec, "score_thresh");
        if (!thr) thr = dict_get(spec, "score.thresh");
        if (thr) {
            PMPY tmp(PyNumber_Float(thr), true);
            if (!tmp) {
                PyErr_Clear();
                TGLError("%s: 'score_thresh' must be a single number", who.c_str());
            }
            params.score_thresh = PyFloat_AsDouble(tmp);
        }

        return params;
    }
};

#endif // PM_POTTS_PARAMS_H_
