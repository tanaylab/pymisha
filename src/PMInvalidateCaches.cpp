/*
 * Drops the process-static per-directory index caches for a track or interval
 * set directory. Exposed to Python as pm_invalidate_dir_cache(paths) and
 * pm_clear_dir_caches().
 *
 * GenomeTrack::get_track_index memoises a TrackIndex per track directory and
 * TrackIndex2D does the same for 2D tracks. Neither had any invalidation, so a
 * directory that is removed, recreated, renamed or converted under the same path
 * left every later read routed through an index describing a layout that is no
 * longer there: "Cannot open .../track.dat" at best, silently wrong values at
 * worst. Unlike R misha there is no gdb_reload(rescan=True) escape hatch here, so
 * a poisoned session could only be fixed by restarting the interpreter.
 *
 * Negative lookups are never cached, so the only stale state is "cache says
 * indexed" - per-chrom -> indexed conversion is structurally safe.
 *
 * Mirrors misha's src/GdbInvalidateCaches.cpp (defect F, 5.11.20).
 */

#include <string>

#include "GenomeTrack.h"
#include "TrackIndex2D.h"
#include "pymisha.h"

PyObject *pm_invalidate_dir_cache(PyObject *self, PyObject *args)
{
    (void)self;
    try {
        PyObject *py_paths = nullptr;
        if (!PyArg_ParseTuple(args, "O", &py_paths))
            return NULL;

        // Accept a single str or any sequence of str.
        if (PyUnicode_Check(py_paths)) {
            const char *path = PyUnicode_AsUTF8(py_paths);
            if (!path)
                return NULL;
            if (*path) {
                GenomeTrack::invalidate_index_cache(path);
                TrackIndex2D::invalidate_cache(path);
            }
            Py_RETURN_NONE;
        }

        PMPY seq(PySequence_Fast(py_paths, "expected a str or a sequence of str"), true);
        if (!seq)
            return NULL;

        Py_ssize_t n = PySequence_Fast_GET_SIZE(*seq);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *item = PySequence_Fast_GET_ITEM(*seq, i);
            if (!PyUnicode_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "paths must be str");
                return NULL;
            }
            const char *path = PyUnicode_AsUTF8(item);
            if (!path)
                return NULL;
            if (*path) {
                GenomeTrack::invalidate_index_cache(path);
                TrackIndex2D::invalidate_cache(path);
            }
        }
        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    }
}

PyObject *pm_clear_dir_caches(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
    try {
        GenomeTrack::clear_index_cache();
        TrackIndex2D::clear_cache();
        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    }
}

