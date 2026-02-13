#ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE 199309
    #include <time.h>
    #undef _POSIX_C_SOURCE
#endif

// This must be undefined before "#include <numpy/arrayobject.h>" in the file that calls import_array
#ifdef NO_IMPORT_ARRAY
    #undef NO_IMPORT_ARRAY
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
    #define PY_ARRAY_UNIQUE_SYMBOL pymisha_ARRAY_API
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include "PMDb.h"

PyObject *g_module;
PyObject *s_pm_err;

static void pymisha_module_free(void *m)
{
    if (g_pmdb) {
        delete g_pmdb;
        g_pmdb = nullptr;
    }
    s_pm_err = nullptr;
}

// Forward declarations of functions to be implemented
PyObject *pm_dbinit(PyObject *self, PyObject *args);
PyObject *pm_dbreload(PyObject *self, PyObject *args);
PyObject *pm_dbunload(PyObject *self, PyObject *args);
PyObject *pm_dbsetdatasets(PyObject *self, PyObject *args);
PyObject *pm_dbgetdatasets(PyObject *self, PyObject *args);
PyObject *pm_extract(PyObject *self, PyObject *args);
PyObject *pm_screen(PyObject *self, PyObject *args);
PyObject *pm_summary(PyObject *self, PyObject *args);
PyObject *pm_quantiles(PyObject *self, PyObject *args);
PyObject *pm_intervals_summary(PyObject *self, PyObject *args);
PyObject *pm_intervals_quantiles(PyObject *self, PyObject *args);
PyObject *pm_track_names(PyObject *self, PyObject *args);
PyObject *pm_track_info(PyObject *self, PyObject *args);
PyObject *pm_track_path(PyObject *self, PyObject *args);
PyObject *pm_track_dataset(PyObject *self, PyObject *args);
PyObject *pm_normalize_chroms(PyObject *self, PyObject *args);
PyObject *pm_track_convert_to_indexed(PyObject *self, PyObject *args);
PyObject *pm_track_create_empty_indexed(PyObject *self, PyObject *args);
PyObject *pm_track_create_sparse(PyObject *self, PyObject *args);
PyObject *pm_track_create_dense(PyObject *self, PyObject *args);
PyObject *pm_track_create_expr(PyObject *self, PyObject *args);
PyObject *pm_intervals_all(PyObject *self, PyObject *args);
PyObject *pm_iterate(PyObject *self, PyObject *args);
PyObject *pm_seed(PyObject *self, PyObject *args);
PyObject *pm_test_df(PyObject *self, PyObject *args);
PyObject *pm_read_df(PyObject *self, PyObject *args);
PyObject *pm_vtrack_compute(PyObject *self, PyObject *args);
PyObject *pm_find_neighbors(PyObject *self, PyObject *args);
PyObject *pm_seq_extract(PyObject *self, PyObject *args);
PyObject *pm_partition(PyObject *self, PyObject *args);
PyObject *pm_dist(PyObject *self, PyObject *args);
PyObject *pm_intervals_union(PyObject *self, PyObject *args);
PyObject *pm_intervals_intersect(PyObject *self, PyObject *args);
PyObject *pm_intervals_diff(PyObject *self, PyObject *args);
PyObject *pm_intervals_canonic(PyObject *self, PyObject *args);
PyObject *pm_intervals_covered_bp(PyObject *self, PyObject *args);
PyObject *pm_sample(PyObject *self, PyObject *args);
PyObject *pm_cor(PyObject *self, PyObject *args);
PyObject *pm_lookup(PyObject *self, PyObject *args);
PyObject *pm_segment(PyObject *self, PyObject *args);
PyObject *pm_wilcox(PyObject *self, PyObject *args);
PyObject *pm_modify(PyObject *self, PyObject *args);
PyObject *pm_smooth(PyObject *self, PyObject *args);
PyObject *pm_gsynth_train(PyObject *self, PyObject *args);
PyObject *pm_gsynth_sample(PyObject *self, PyObject *args);
PyObject *pm_gsynth_replace_kmer(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"pm_dbinit", pm_dbinit, METH_VARARGS, "Initialize database connection"},
    {"pm_dbreload", pm_dbreload, METH_VARARGS, "Reload database"},
    {"pm_dbunload", pm_dbunload, METH_VARARGS, "Unload database"},
    {"pm_dbsetdatasets", pm_dbsetdatasets, METH_VARARGS, "Set loaded dataset roots"},
    {"pm_dbgetdatasets", pm_dbgetdatasets, METH_VARARGS, "Get loaded dataset roots"},
    {"pm_extract", pm_extract, METH_VARARGS, "Extract track values"},
    {"pm_screen", pm_screen, METH_VARARGS, "Screen intervals by expression"},
    {"pm_summary", pm_summary, METH_VARARGS, "Summarize expression values"},
    {"pm_quantiles", pm_quantiles, METH_VARARGS, "Compute expression quantiles"},
    {"pm_intervals_summary", pm_intervals_summary, METH_VARARGS, "Summarize expression values per interval"},
    {"pm_intervals_quantiles", pm_intervals_quantiles, METH_VARARGS, "Compute expression quantiles per interval"},
    {"pm_track_names", pm_track_names, METH_VARARGS, "Get track names"},
    {"pm_track_info", pm_track_info, METH_VARARGS, "Get track information"},
    {"pm_track_path", pm_track_path, METH_VARARGS, "Get track path on disk"},
    {"pm_track_dataset", pm_track_dataset, METH_VARARGS, "Get track dataset root"},
    {"pm_normalize_chroms", pm_normalize_chroms, METH_VARARGS, "Normalize chromosome names"},
    {"pm_track_convert_to_indexed", pm_track_convert_to_indexed, METH_VARARGS, "Convert track to indexed format"},
    {"pm_track_create_empty_indexed", pm_track_create_empty_indexed, METH_VARARGS, "Create empty indexed track"},
    {"pm_track_create_sparse", pm_track_create_sparse, METH_VARARGS, "Create sparse track from intervals+values"},
    {"pm_track_create_dense", pm_track_create_dense, METH_VARARGS, "Create dense track from intervals+values"},
    {"pm_track_create_expr", pm_track_create_expr, METH_VARARGS, "Create track from expression in streaming mode"},
    {"pm_intervals_all", pm_intervals_all, METH_VARARGS, "Get all genome intervals"},
    {"pm_iterate", pm_iterate, METH_VARARGS, "Iterate intervals with iterator policy"},
    {"pm_seed", pm_seed, METH_VARARGS, "Set random seed"},
    {"pm_vtrack_compute", pm_vtrack_compute, METH_VARARGS, "Compute virtual track values"},
    {"pm_find_neighbors", pm_find_neighbors, METH_VARARGS, "Find nearest neighbor intervals"},
    {"pm_seq_extract", pm_seq_extract, METH_VARARGS, "Extract DNA sequences for intervals"},
    {"pm_partition", pm_partition, METH_VARARGS, "Partition track values into bins"},
    {"pm_dist", pm_dist, METH_VARARGS, "Calculate distribution of track values over bins"},
    {"pm_lookup", pm_lookup, METH_VARARGS, "Lookup table transform on binned track values"},
    {"pm_intervals_union", pm_intervals_union, METH_VARARGS, "Union of two interval sets"},
    {"pm_intervals_intersect", pm_intervals_intersect, METH_VARARGS, "Intersection of two interval sets"},
    {"pm_intervals_diff", pm_intervals_diff, METH_VARARGS, "Difference of two interval sets"},
    {"pm_intervals_canonic", pm_intervals_canonic, METH_VARARGS, "Canonicalize intervals"},
    {"pm_intervals_covered_bp", pm_intervals_covered_bp, METH_VARARGS, "Count total covered basepairs"},
    {"pm_sample", pm_sample, METH_VARARGS, "Sample values from track expression"},
    {"pm_cor", pm_cor, METH_VARARGS, "Compute correlation between expression pairs"},
    {"pm_segment", pm_segment, METH_VARARGS, "Segment track expression using Wilcoxon test"},
    {"pm_wilcox", pm_wilcox, METH_VARARGS, "Sliding-window Wilcoxon test on track expression"},
    {"pm_modify", pm_modify, METH_VARARGS, "Modify dense track values in-place"},
    {"pm_smooth", pm_smooth, METH_VARARGS, "Create smoothed track from expression"},
    {"__pm_test_df", pm_test_df, METH_VARARGS, "Test DataFrame conversion"},
    {"__read_df", pm_read_df, METH_VARARGS, "Read DataFrame from internal format"},
    {"pm_gsynth_train", pm_gsynth_train, METH_VARARGS, "Train stratified Markov-5 model"},
    {"pm_gsynth_sample", pm_gsynth_sample, METH_VARARGS, "Sample synthetic genome"},
    {"pm_gsynth_replace_kmer", pm_gsynth_replace_kmer, METH_VARARGS, "Replace k-mers iteratively"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__pymisha(void)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_pymisha",
        "PyMisha genomics toolkit - C++ extension module",
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        pymisha_module_free
    };

    g_module = PyModule_Create(&moduledef);

    if (!g_module)
        return NULL;

    s_pm_err = PyErr_NewException("pymisha.error", NULL, NULL);
    if (!s_pm_err) {
        Py_DECREF(g_module);
        return NULL;
    }
    if (PyModule_AddObject(g_module, "error", s_pm_err) < 0) {
        Py_DECREF(s_pm_err);
        s_pm_err = NULL;
        Py_DECREF(g_module);
        return NULL;
    }

    import_array();

    struct timespec tm;
    clock_gettime(CLOCK_MONOTONIC, &tm);
    srand48(tm.tv_sec ^ tm.tv_nsec);

    return g_module;
}
