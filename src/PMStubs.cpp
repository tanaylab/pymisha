// Stub implementations for pymisha functions
// These will be replaced with full implementations as the project progresses

#include "pymisha.h"
#include "PMDataFrame.h"
#include "PMDirectAccumulator.h"
#include "PMDb.h"
#include "PMTrackExpressionScanner.h"
#include "PMTrackExpressionIterator.h"
#include "GenomeTrackFixedBin.h"
#include "GenomeSeqFetch.h"
#include "StreamPercentiler.h"
#include "StreamSampler.h"
#include "RandomShuffle.h"
#include "BinFinder.h"
#include "BinsManager.h"
#include "QuadTreeReader.h"
#include "MemQuadTree.h"
#include "GInterval2D.h"
#include "PMTrackExpression2DIterator.h"
#include "PMTrackExpression2DIteratorPolicy.h"
#include "PMTrackExpressionFixedRectIterator.h"
#include "PMTrackExpressionTrackRectsIterator.h"
#include "PMTrackExpressionCartesianGridIterator.h"
#include "PMGenomeTrack2D.h"
#include "PMTrackExpression2DScanner.h"
#include "PMTrackExpression2DVars.h"
#include "pmutils.h"
#include <new>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <sys/stat.h>

// Resolve a Python chrom value (str or int) to a chromid. Helper used by
// the vectorized parser below; not inlined into the hot loop so the loop
// stays compact.
static int resolve_chrom_id(PyObject *chrom_val, const GenomeChromKey &chromkey)
{
    if (PyUnicode_Check(chrom_val)) {
        const char *name = PyUnicode_AsUTF8(chrom_val);
        int id = chromkey.chrom2id(name);
        if (id < 0)
            TGLError("Unknown chromosome: %s", name);
        return id;
    }
    if (PyNumber_Check(chrom_val)) {
        PMPY py_long(PyNumber_Long(chrom_val), true);
        if (!py_long) {
            PyErr_Clear();
            TGLError("Failed to convert chromosome to integer");
        }
        long n = PyLong_AsLong((PyObject *)py_long);
        std::string s = std::to_string(n);
        int id = chromkey.chrom2id(s.c_str());
        if (id < 0) {
            s = "chr" + std::to_string(n);
            id = chromkey.chrom2id(s.c_str());
        }
        if (id < 0)
            TGLError("Unknown chromosome: %ld", n);
        return id;
    }
    TGLError("Invalid chromosome type");
    return -1;
}

// Bring the raw chrom / start / end columns into the shape the converters
// below expect: a chrom sequence of strings and int64 coordinate arrays.
void normalize_interval_columns(PMPY &py_chrom, PMPY &py_start, PMPY &py_end)
{
    // A categorical chrom column arrives from _df2pymisha as the 2-element
    // list [categories, codes]. Materialize it with numpy fancy indexing.
    if (PyList_Check((PyObject *)py_chrom) && PyList_Size((PyObject *)py_chrom) == 2) {
        PyObject *cats = PyList_GetItem((PyObject *)py_chrom, 0);
        PyObject *codes = PyList_GetItem((PyObject *)py_chrom, 1);
        if (cats && codes && PyArray_Check(cats) && PyArray_Check(codes)) {
            PyObject *expanded = PyObject_GetItem(cats, codes);
            if (expanded)
                py_chrom.assign(expanded, true);
            else
                PyErr_Clear();
        }
    }

    // Coerce start/end to int64. Float coordinates are routine (a BED read
    // with any blank field, or a non-matching merge, yields float64) and R
    // misha accepts them. FORCECAST truncates toward zero like R's
    // as.integer; a no-op incref when the array is already int64.
    for (PMPY *col : {&py_start, &py_end}) {
        PyObject *obj = (PyObject *)(*col);
        if (!obj)
            continue;
        if (PyArray_Check(obj) && PyArray_TYPE((PyArrayObject *)obj) == NPY_INT64 &&
            PyArray_NDIM((PyArrayObject *)obj) == 1 && PyArray_ISCARRAY_RO((PyArrayObject *)obj))
            continue;
        PyObject *casted = PyArray_FROM_OTF(obj, NPY_INT64,
                                            NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
        if (casted)
            col->assign(casted, true);
        else
            PyErr_Clear();  // leave it to the caller's slow path
    }
}

// R-parity: misha's IntervalConverter calls GInterval::verify() on every
// converted interval. Without it a negative start reaches
// GenomeTrackFixedBin::read_bins_bulk as a negative bin index and reads
// before the mmap (segfault, or silent garbage when it stays in-page), and
// coordinates past the end of a chromosome (a hg19 BED against an hg38 db)
// come back as plausible-looking NaN rows instead of an error.
void verify_interval_coords(const GenomeChromKey &chromkey, int chromid,
                                   int64_t start, int64_t end) {
    // NaN casts to INT64_MIN, which would otherwise be reported as a
    // nonsensical negative coordinate.
    if (start == INT64_MIN || end == INT64_MIN)
        TGLError("Interval on chromosome %s has a missing (NaN) coordinate",
                 chromkey.id2chrom(chromid).c_str());
    if (start < 0)
        TGLError("Interval (%s, %ld, %ld): start coordinate must be greater or equal than zero",
                 chromkey.id2chrom(chromid).c_str(), (long)start, (long)end);
    if (start >= end)
        TGLError("Interval (%s, %ld, %ld): start coordinate must be lesser than end coordinate",
                 chromkey.id2chrom(chromid).c_str(), (long)start, (long)end);
    if ((uint64_t)end > chromkey.get_chrom_size(chromid))
        TGLError("Interval (%s, %ld, %ld): end coordinate exceeds chromosome boundaries",
                 chromkey.id2chrom(chromid).c_str(), (long)start, (long)end);
}

// Convert Python intervals (DataFrame or _df2pymisha list) to a C++
// GInterval vector. Hot path: NPY_OBJECT chrom array of PyUnicode strings
// plus NPY_INT64 start/end arrays. The cold paths (DataFrame attribute /
// dict-like access, non-int64 numeric start/end, numeric chrom) fall back
// to PyArray_GETITEM-based reads.
//
// Optimizations:
//  - Direct PyArray_DATA buffer access for start/end when dtype == int64,
//    so the loop reads raw int64_t* (no Python C-API per row).
//  - Run-length cache for chrom -> chromid: typical input has long runs
//    of the same PyObject* (e.g. ['chr1']*N), and even when distinct
//    objects share the same interned UTF-8 we can compare pointers first.
//
// Expected impact: -200..-350 ms on a 1M-row input on hg38.
void convert_py_intervals(PyObject *py_intervals, std::vector<GInterval> &intervals) {
    PMPY py_chrom;
    PMPY py_start;
    PMPY py_end;

    // Check if it's the internal list format from _df2pymisha
    if (PyList_Check(py_intervals) && PyList_Size(py_intervals) >= 2) {
        // First element is column names array
        PyObject *colnames = PyList_GetItem(py_intervals, 0);
        if (colnames && PyArray_Check(colnames)) {
            // Find column indices
            Py_ssize_t num_cols = PyArray_SIZE((PyArrayObject *)colnames);
            int chrom_idx = -1, start_idx = -1, end_idx = -1;

            for (Py_ssize_t i = 0; i < num_cols; ++i) {
                PyObject *name = PyArray_GETITEM((PyArrayObject *)colnames,
                    (const char *)PyArray_GETPTR1((PyArrayObject *)colnames, i));
                if (name && PyUnicode_Check(name)) {
                    const char *name_str = PyUnicode_AsUTF8(name);
                    if (strcmp(name_str, "chrom") == 0) chrom_idx = i;
                    else if (strcmp(name_str, "start") == 0) start_idx = i;
                    else if (strcmp(name_str, "end") == 0) end_idx = i;
                }
                Py_XDECREF(name);
            }

            if (chrom_idx >= 0 && start_idx >= 0 && end_idx >= 0) {
                // Column values are at list indices 1, 2, 3, ... (offset by 1 from column index)
                py_chrom.assign(PyList_GetItem(py_intervals, chrom_idx + 1), false);
                py_start.assign(PyList_GetItem(py_intervals, start_idx + 1), false);
                py_end.assign(PyList_GetItem(py_intervals, end_idx + 1), false);
            }
        }
    }

    // Fallback: try DataFrame-like attribute access
    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        py_chrom.assign(PyObject_GetAttrString(py_intervals, "chrom"), true);
        py_start.assign(PyObject_GetAttrString(py_intervals, "start"), true);
        py_end.assign(PyObject_GetAttrString(py_intervals, "end"), true);
    }

    // Fallback: try dict-like access
    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        // Use PMPY to manage reference counting for temporary keys
        PMPY key_chrom(PyUnicode_FromString("chrom"), true);
        PMPY key_start(PyUnicode_FromString("start"), true);
        PMPY key_end(PyUnicode_FromString("end"), true);
        py_chrom.assign(PyObject_GetItem(py_intervals, key_chrom), true);
        py_start.assign(PyObject_GetItem(py_intervals, key_start), true);
        py_end.assign(PyObject_GetItem(py_intervals, key_end), true);
    }

    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        TGLError("intervals must have 'chrom', 'start', and 'end' columns");
    }

    normalize_interval_columns(py_chrom, py_start, py_end);

    // Get length
    Py_ssize_t len = PyObject_Length(py_chrom);
    if (len < 0) {
        PyErr_Clear();
        TGLError("Cannot determine length of intervals");
    }

    intervals.clear();
    intervals.reserve(len);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();

    // Fast-path: chrom is a 1D NPY_OBJECT array (most common), start/end
    // are 1D NPY_INT64 arrays (after _df2pymisha or _intervals_to_cpp).
    // Direct buffer access + run-length cache on chrom PyObject* pointer.
    PyArrayObject *chrom_arr = nullptr;
    PyArrayObject *start_arr = nullptr;
    PyArrayObject *end_arr   = nullptr;
    if (PyArray_Check((PyObject *)py_chrom)) chrom_arr = (PyArrayObject *)(PyObject *)py_chrom;
    if (PyArray_Check((PyObject *)py_start)) start_arr = (PyArrayObject *)(PyObject *)py_start;
    if (PyArray_Check((PyObject *)py_end))   end_arr   = (PyArrayObject *)(PyObject *)py_end;

    const bool fast_path =
        chrom_arr && start_arr && end_arr &&
        PyArray_NDIM(chrom_arr) == 1 && PyArray_NDIM(start_arr) == 1 && PyArray_NDIM(end_arr) == 1 &&
        PyArray_TYPE(chrom_arr) == NPY_OBJECT &&
        PyArray_TYPE(start_arr) == NPY_INT64 &&
        PyArray_TYPE(end_arr)   == NPY_INT64 &&
        PyArray_ISCARRAY_RO(chrom_arr) &&
        PyArray_ISCARRAY_RO(start_arr) &&
        PyArray_ISCARRAY_RO(end_arr);

    if (fast_path) {
        PyObject **chrom_data = (PyObject **)PyArray_DATA(chrom_arr);
        const int64_t *start_data = (const int64_t *)PyArray_DATA(start_arr);
        const int64_t *end_data   = (const int64_t *)PyArray_DATA(end_arr);

        // Single-entry cache: avoids re-resolving chromid when the
        // PyObject* (or its UTF-8 contents) match the previous row. Hg38
        // sparse inputs typically have one or a few runs over millions of
        // rows; the cached lookup is essentially free.
        PyObject *last_obj = nullptr;
        int last_id = -1;

        for (Py_ssize_t i = 0; i < len; ++i) {
            PyObject *obj = chrom_data[i];
            if (!obj) TGLError("Null chromosome at index %ld", (long)i);

            int id;
            if (obj == last_obj) {
                id = last_id;
            } else {
                id = resolve_chrom_id(obj, chromkey);
                last_obj = obj;
                last_id = id;
            }
            if (start_data[i] < 0 || start_data[i] >= end_data[i] ||
                (uint64_t)end_data[i] > chromkey.get_chrom_size(id))
                verify_interval_coords(chromkey, id, start_data[i], end_data[i]);
            intervals.emplace_back(id, start_data[i], end_data[i]);
        }
        return;
    }

    // Slow path: use PySequence_GetItem / PyArray_GETITEM-style reads for
    // arbitrary input layouts (non-int64 numeric, lists, integer chroms,
    // pandas Series with Arrow backing, etc.). Same semantics as the
    // original implementation.
    PyObject *last_obj_slow = nullptr;
    int last_id_slow = -1;
    for (Py_ssize_t i = 0; i < len; ++i) {
        PMPY chrom_val(PySequence_GetItem(py_chrom, i), true);
        PMPY start_val(PySequence_GetItem(py_start, i), true);
        PMPY end_val(PySequence_GetItem(py_end, i), true);

        if (!chrom_val || !start_val || !end_val) {
            PyErr_Clear();
            TGLError("Failed to get interval values at index %ld", (long)i);
        }

        int chromid;
        PyObject *cobj = (PyObject *)chrom_val;
        if (cobj == last_obj_slow) {
            chromid = last_id_slow;
        } else {
            chromid = resolve_chrom_id(cobj, chromkey);
            last_obj_slow = cobj;
            last_id_slow = chromid;
        }

        int64_t start = PyLong_AsLongLong(start_val);
        int64_t end = PyLong_AsLongLong(end_val);

        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("Invalid start/end values at index %ld", (long)i);
        }

        if (start < 0 || start >= end ||
            (uint64_t)end > chromkey.get_chrom_size(chromid))
            verify_interval_coords(chromkey, chromid, start, end);
        intervals.emplace_back(chromid, start, end);
    }
}

extern PyObject *s_pm_err;

static void parse_string_list(PyObject *obj, std::vector<std::string> &out, const char *what)
{
    out.clear();
    if (!obj || obj == Py_None) {
        return;
    }

    PMPY seq(PySequence_Fast(obj, what), true);
    if (!seq) {
        verror("%s", what);
    }

    PyObject *seq_obj = (PyObject *)seq;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq_obj);
    PyObject **items = PySequence_Fast_ITEMS(seq_obj);
    out.reserve((size_t)n);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item = items[i];
        if (!PyUnicode_Check(item)) {
            verror("Expected list of strings");
        }
        out.emplace_back(PyUnicode_AsUTF8(item));
    }
}

// Database initialization
PyObject *pm_dbinit(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(false);

        const char *groot = NULL;
        const char *uroot = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "s|sO", &groot, &uroot, &py_config)) {
            verror("Invalid arguments to pm_dbinit");
        }

        // Initialize the database
        if (!g_pmdb) {
            g_pmdb = new PMDb();
        }
        g_pmdb->init(groot ? groot : "", uroot ? uroot : "");

        // Also store in pymisha for backward compat
        g_pymisha->set_db_paths(groot ? groot : "", uroot ? uroot : "");

        vdebug("Database initialized: groot=%s, uroot=%s, chroms=%lu\n",
               groot ? groot : "(null)",
               uroot ? uroot : "(null)",
               (unsigned long)g_pmdb->chromkey().get_num_chroms());

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *pm_dbsetdatasets(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_datasets = NULL;
        if (!PyArg_ParseTuple(args, "O", &py_datasets)) {
            verror("Invalid arguments to pm_dbsetdatasets");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        std::vector<std::string> datasets;
        parse_string_list(py_datasets, datasets, "datasets must be a sequence of strings");

        g_pmdb->set_datasets(datasets);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *pm_dbgetdatasets(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        const auto &datasets = g_pmdb->datasets();
        PMPY list(PyList_New(datasets.size()), true);
        for (size_t i = 0; i < datasets.size(); ++i) {
            PyList_SetItem((PyObject *)list, i, PyUnicode_FromString(datasets[i].c_str()));
        }
        list.to_be_stolen();
        return (PyObject *)list;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

PyObject *pm_track_dataset(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_track = NULL;
        if (!PyArg_ParseTuple(args, "O", &py_track)) {
            verror("Invalid arguments to pm_track_dataset");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        if (!PyUnicode_Check(py_track)) {
            verror("Track name must be a string");
        }
        std::string track_name = PyUnicode_AsUTF8(py_track);

        if (!g_pmdb->track_exists(track_name)) {
            Py_RETURN_NONE;
        }

        std::string root = g_pmdb->track_dataset(track_name);
        PMPY result(PyUnicode_FromString(root.c_str()), true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

PyObject *pm_normalize_chroms(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_chroms = NULL;
        if (!PyArg_ParseTuple(args, "O", &py_chroms)) {
            verror("Invalid arguments to pm_normalize_chroms");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        PMPY seq(PySequence_Fast(py_chroms, "chroms must be a sequence"), true);
        if (!seq) {
            verror("chroms must be a sequence");
        }

        PyObject *seq_obj = (PyObject *)seq;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq_obj);
        PyObject **items = PySequence_Fast_ITEMS(seq_obj);
        PMPY out(PyList_New(n), true);

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *item = items[i];
            int chromid = -1;

            if (PyUnicode_Check(item)) {
                chromid = chromkey.chrom2id(PyUnicode_AsUTF8(item));
            } else if (PyLong_Check(item)) {
                long chrom_num = PyLong_AsLong(item);
                std::string chrom_str = std::to_string(chrom_num);
                chromid = chromkey.chrom2id(chrom_str.c_str());
                if (chromid < 0) {
                    chrom_str = "chr" + chrom_str;
                    chromid = chromkey.chrom2id(chrom_str.c_str());
                }
            } else {
                verror("Invalid chromosome type at index %ld", (long)i);
            }

            if (chromid < 0) {
                verror("Unknown chromosome at index %ld", (long)i);
            }

            PyList_SetItem((PyObject *)out, i,
                           PyUnicode_FromString(chromkey.id2chrom(chromid).c_str()));
        }

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

PyObject *pm_dbreload(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        if (g_pmdb && g_pmdb->is_initialized()) {
            g_pmdb->reload();
        }
        vdebug("Database reload requested\n");
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *pm_dbunload(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(false);

        if (g_pmdb) {
            g_pmdb->unload();
        }
        if (g_pymisha) {
            g_pymisha->set_db_paths("", "");
        }
        vdebug("Database unloaded\n");
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

// ---------------------------------------------------------------------------
//  Range-based interval splitting for intra-chromosome parallelization
// ---------------------------------------------------------------------------

struct SplitInterval {
    GInterval interval;
    uint64_t original_idx;  // 0-based index into original intervals vector
};

// Port of R misha's split_intervals_1d_by_range: walk intervals linearly,
// accumulate range per worker, split individual intervals at bin-aligned
// boundaries.
static void split_intervals_by_range(const std::vector<GInterval> &intervals,
                                     int desired_kids,
                                     int64_t split_align,
                                     std::vector<std::vector<SplitInterval>> &out_kids)
{
    out_kids.clear();
    if (intervals.empty())
        return;

    int64_t total_range = 0;
    for (const auto &iv : intervals)
        total_range += iv.end - iv.start;

    if (total_range <= 0)
        return;

    desired_kids = std::max(1, desired_kids);
    if ((int64_t)desired_kids > total_range)
        desired_kids = (int)total_range;

    int kids_left = desired_kids;
    int64_t range_left = total_range;
    int64_t kid_range = 0;

    out_kids.emplace_back();  // first worker bucket

    for (uint64_t idx = 0; idx < intervals.size(); ++idx) {
        const GInterval &iinterv = intervals[idx];
        int64_t seg_start = iinterv.start;
        const int64_t seg_end_total = iinterv.end;

        while (seg_start < seg_end_total) {
            if (kids_left <= 1) {
                // Last worker: absorb everything remaining
                SplitInterval si;
                si.interval = GInterval(iinterv.chromid, seg_start, seg_end_total, iinterv.strand, iinterv.udata);
                si.original_idx = idx;
                out_kids.back().push_back(si);

                int64_t seg_range = seg_end_total - seg_start;
                kid_range += seg_range;
                range_left -= seg_range;
                seg_start = seg_end_total;
                continue;
            }

            int64_t target_kid_range = (range_left + kids_left - 1) / kids_left;
            int64_t need = target_kid_range - kid_range;

            if (need <= 0 && !out_kids.back().empty()) {
                --kids_left;
                out_kids.emplace_back();
                kid_range = 0;
                continue;
            }

            int64_t chunk_end = seg_end_total;
            if (need > 0 && seg_start + need < seg_end_total)
                chunk_end = seg_start + need;

            if (split_align > 0 && chunk_end < seg_end_total) {
                int64_t aligned_end = (chunk_end / split_align) * split_align;
                if (aligned_end <= seg_start)
                    aligned_end = ((seg_start / split_align) + 1) * split_align;
                if (aligned_end < seg_end_total)
                    chunk_end = aligned_end;
            }

            if (chunk_end <= seg_start)
                chunk_end = seg_start + 1;

            SplitInterval si;
            si.interval = GInterval(iinterv.chromid, seg_start, chunk_end, iinterv.strand, iinterv.udata);
            si.original_idx = idx;
            out_kids.back().push_back(si);

            int64_t seg_range = chunk_end - seg_start;
            kid_range += seg_range;
            range_left -= seg_range;
            seg_start = chunk_end;

            if (kid_range >= target_kid_range && kids_left > 1) {
                --kids_left;
                out_kids.emplace_back();
                kid_range = 0;
            }
        }
    }

    // Trim trailing empty workers
    while (!out_kids.empty() && out_kids.back().empty())
        out_kids.pop_back();
}

static void extract_sub_intervals(
    const std::vector<SplitInterval> &split,
    std::vector<GInterval> &sub,
    std::vector<uint64_t> *id_remap = nullptr)
{
    sub.clear();
    sub.reserve(split.size());
    if (id_remap) {
        id_remap->clear();
        id_remap->reserve(split.size());
    }
    for (const auto &si : split) {
        sub.push_back(si.interval);
        if (id_remap)
            id_remap->push_back(si.original_idx);
    }
}

// Forward declaration (defined later)
static int choose_num_kids(uint64_t num_intervals, int64_t total_scope_bp = -1);

static int64_t total_intervals_scope(const std::vector<GInterval> &intervals) {
    int64_t total = 0;
    for (const auto &iv : intervals) total += iv.end - iv.start;
    return total;
}

static int choose_num_kids_for_range(size_t num_intervals, int64_t total_range, int64_t bin_size)
{
    if (!g_pymisha->multitasking_avail())
        return 0;

    int hw = std::thread::hardware_concurrency();
    int max_p = g_pymisha->max_processes();
    int min_p = g_pymisha->min_processes();
    int target = std::clamp(hw ? hw : max_p, min_p, max_p);

    // Range-based: at least 50000 bins per worker.
    int64_t total_bins = (total_range + bin_size - 1) / bin_size;
    int64_t min_bins_per_kid = 50000;
    int desired = std::min((int64_t)target, (total_bins + min_bins_per_kid - 1) / min_bins_per_kid);
    if (desired < 1) desired = 1;

    int count_based = choose_num_kids(num_intervals, total_range);

    int result = std::max(count_based, (int)desired);
    if (result < 2)
        return 0;

    // Scope floor: also require each kid sees min_scope4process bp on the
    // range-based path. Without this, dense scans on small scopes pay
    // fork+IPC cost that exceeds the parallel compute win.
    uint64_t min_scope = g_pymisha->min_scope4process();
    if (min_scope > 0 && total_range > 0) {
        uint64_t scope_per_kid = (uint64_t)total_range / (uint64_t)result;
        if (scope_per_kid < min_scope)
            return 0;
    }

    return std::min(result, target);
}

// ---------------------------------------------------------------------------

struct ExtractResult {
    std::vector<GInterval> intervals;
    std::vector<std::vector<double>> values;
    std::vector<uint64_t> interval_ids;
};

static PyObject *get_progress_cb(PyObject *py_config)
{
    if (!py_config || py_config == Py_None || !PyDict_Check(py_config)) {
        return NULL;
    }

    PyObject *cb = PyDict_GetItemString(py_config, "_progress_cb");
    if (cb && cb != Py_None && !PyCallable_Check(cb)) {
        verror("progress callback must be callable");
    }
    return (cb && cb != Py_None) ? cb : NULL;
}

static bool get_config_bool(PyObject *py_config, const char *key, bool default_value)
{
    if (!py_config || py_config == Py_None || !PyDict_Check(py_config)) {
        return default_value;
    }

    PyObject *value = PyDict_GetItemString(py_config, key);
    if (!value || value == Py_None) {
        return default_value;
    }

    int truth = PyObject_IsTrue(value);
    if (truth < 0) {
        verror("Config key '%s' must be truthy/falsy", key);
    }
    return truth != 0;
}

long parse_iterator_policy(PyObject *py_iterator, long default_policy, const char *context)
{
    if (!py_iterator || py_iterator == Py_None) {
        return default_policy;
    }

    if (PyLong_Check(py_iterator)) {
        long val = PyLong_AsLong(py_iterator);
        if (PyErr_Occurred()) {
            verror("%s: iterator value out of range", context);
        }
        return val;
    }

    if (PyFloat_Check(py_iterator)) {
        double val = PyFloat_AsDouble(py_iterator);
        if (!std::isfinite(val)) {
            verror("%s: iterator must be a finite number", context);
        }
        if (val > (double)LONG_MAX || val < (double)LONG_MIN) {
            verror("%s: iterator value out of range", context);
        }
        return (long)val;
    }

    if (PyNumber_Check(py_iterator)) {
        PMPY as_long(PyNumber_Long(py_iterator), true);
        if (!as_long) {
            verror("%s: iterator must be an integer or float", context);
        }
        long val = PyLong_AsLong((PyObject *)as_long);
        if (PyErr_Occurred()) {
            verror("%s: iterator value out of range", context);
        }
        return val;
    }

    verror("%s: iterator must be an integer or float", context);
    return default_policy;
}

static ExtractResult run_extract(const std::vector<std::string> &exprs,
                                 const std::vector<GInterval> &intervals,
                                 long iterator_policy,
                                 uint64_t interval_id_offset,
                                 PyObject *progress_cb,
                                 const std::vector<uint64_t> *id_remap = nullptr,
                                 PyObject *py_vtracks = nullptr)
{
    ExtractResult result;
    if (intervals.empty()) {
        return result;
    }

    PMTrackExprScanner scanner;
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy, py_vtracks);

    result.values.assign(exprs.size(), {});

    uint64_t est_size = scanner.get_iterator()->size();
    if (est_size > 0) {
        result.intervals.reserve(est_size);
        for (auto &v : result.values) {
            v.reserve(est_size);
        }
        result.interval_ids.reserve(est_size);
    }

    for (; !scanner.isend(); scanner.next()) {
        result.intervals.push_back(scanner.last_interval());

        for (size_t iexpr = 0; iexpr < exprs.size(); ++iexpr) {
            result.values[iexpr].push_back(scanner.vdouble(iexpr));
        }

        if (id_remap) {
            uint64_t local_id = scanner.last_interval_id();
            if (local_id >= 1 && local_id <= (uint64_t)id_remap->size()) {
                result.interval_ids.push_back((*id_remap)[local_id - 1] + 1);
            } else {
                result.interval_ids.push_back(scanner.last_interval_id() + interval_id_offset);
            }
        } else {
            result.interval_ids.push_back(scanner.last_interval_id() + interval_id_offset);
        }

        g_pymisha->verify_max_data_size(result.intervals.size(), "Result");
        check_interrupt();
    }

    return result;
}

// Direct-to-NumPy extract: scans directly into PMDirectAccumulator,
// bypassing intermediate std::vector storage and the PMDataFrame copy.
static void run_extract_direct(PMDirectAccumulator &acc,
                               const std::vector<std::string> &exprs,
                               const std::vector<std::string> &colnames,
                               const std::vector<GInterval> &intervals,
                               long iterator_policy,
                               uint64_t interval_id_offset,
                               PyObject *progress_cb,
                               const std::vector<uint64_t> *id_remap = nullptr,
                               PyObject *py_vtracks = nullptr)
{
    if (intervals.empty()) {
        return;
    }

    PMTrackExprScanner scanner;
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy, py_vtracks);

    uint64_t est_size = scanner.get_iterator()->size();
    acc.init(est_size, colnames);

    // Temporary buffer for per-row expression values
    std::vector<double> row_values(exprs.size());

    for (; !scanner.isend(); scanner.next()) {
        const GInterval &interval = scanner.last_interval();

        for (size_t iexpr = 0; iexpr < exprs.size(); ++iexpr) {
            row_values[iexpr] = scanner.vdouble(iexpr);
        }

        uint64_t interval_id;
        if (id_remap) {
            uint64_t local_id = scanner.last_interval_id();
            if (local_id >= 1 && local_id <= (uint64_t)id_remap->size()) {
                interval_id = (*id_remap)[local_id - 1] + 1;
            } else {
                interval_id = scanner.last_interval_id() + interval_id_offset;
            }
        } else {
            interval_id = scanner.last_interval_id() + interval_id_offset;
        }

        acc.write_row(interval.chromid, interval.start, interval.end,
                      row_values.data(), interval_id);

        g_pymisha->verify_max_data_size(acc.size(), "Result");
        check_interrupt();
    }
}

// Sort accumulator in-place by (intervalID, chromid, start, end)
static void sort_accumulator(PMDirectAccumulator &acc)
{
    const size_t n = acc.num_rows();
    if (n <= 1) return;

    int32_t *chromids = acc.chromids();
    int64_t *starts = acc.starts();
    int64_t *ends = acc.ends();
    long *ids = acc.interval_ids();
    size_t num_expr = acc.num_expr_cols();

    // Build sort order
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (ids[a] != ids[b]) return ids[a] < ids[b];
        if (chromids[a] != chromids[b]) return chromids[a] < chromids[b];
        if (starts[a] != starts[b]) return starts[a] < starts[b];
        return ends[a] < ends[b];
    });

    // Check if already sorted
    bool already_sorted = true;
    for (size_t i = 0; i < n; ++i) {
        if (order[i] != i) { already_sorted = false; break; }
    }
    if (already_sorted) return;

    // Apply permutation in-place using temporary buffers
    // Chromids
    {
        std::vector<int32_t> tmp(n);
        for (size_t i = 0; i < n; ++i) tmp[i] = chromids[order[i]];
        std::memcpy(chromids, tmp.data(), n * sizeof(int32_t));
    }
    // Starts
    {
        std::vector<int64_t> tmp(n);
        for (size_t i = 0; i < n; ++i) tmp[i] = starts[order[i]];
        std::memcpy(starts, tmp.data(), n * sizeof(int64_t));
    }
    // Ends
    {
        std::vector<int64_t> tmp(n);
        for (size_t i = 0; i < n; ++i) tmp[i] = ends[order[i]];
        std::memcpy(ends, tmp.data(), n * sizeof(int64_t));
    }
    // Values
    for (size_t iexpr = 0; iexpr < num_expr; ++iexpr) {
        double *vals = acc.values(iexpr);
        std::vector<double> tmp(n);
        for (size_t i = 0; i < n; ++i) tmp[i] = vals[order[i]];
        std::memcpy(vals, tmp.data(), n * sizeof(double));
    }
    // IntervalIDs
    {
        std::vector<long> tmp(n);
        for (size_t i = 0; i < n; ++i) tmp[i] = ids[order[i]];
        std::memcpy(ids, tmp.data(), n * sizeof(long));
    }
}

static PMPY build_extract_df(const std::vector<std::string> &exprs,
                             const std::vector<std::string> &colnames,
                             const ExtractResult &result)
{
    size_t num_cols = 3 + exprs.size() + 1;  // chrom, start, end + exprs + intervalID
    PMDataFrame df(result.intervals.size(), num_cols, "intervals");

    df.init_col(0, "chrom", PMDataFrame::STR);
    df.init_col(1, "start", PMDataFrame::LONG);
    df.init_col(2, "end", PMDataFrame::LONG);

    for (size_t i = 0; i < exprs.size(); ++i) {
        df.init_col(3 + i, colnames[i].c_str(), PMDataFrame::DOUBLE);
    }

    df.init_col(3 + exprs.size(), "intervalID", PMDataFrame::LONG);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    for (size_t i = 0; i < result.intervals.size(); ++i) {
        df.val_str(i, 0, chromkey.id2chrom(result.intervals[i].chromid).c_str());
        df.val_long(i, 1, result.intervals[i].start);
        df.val_long(i, 2, result.intervals[i].end);

        for (size_t iexpr = 0; iexpr < exprs.size(); ++iexpr) {
            df.val_double(i, 3 + iexpr, result.values[iexpr][i]);
        }

        df.val_long(i, 3 + exprs.size(), result.interval_ids[i]);
    }

    PMPY result_df = df.construct_py(true);
    result_df.to_be_stolen();
    return result_df;
}

static PMPY build_intervals_df(const std::vector<GInterval> &intervals,
                               const std::vector<uint64_t> &interval_ids)
{
    size_t num_cols = 4;  // chrom, start, end, intervalID
    PMDataFrame df(intervals.size(), num_cols, "intervals");

    df.init_col(0, "chrom", PMDataFrame::STR);
    df.init_col(1, "start", PMDataFrame::LONG);
    df.init_col(2, "end", PMDataFrame::LONG);
    df.init_col(3, "intervalID", PMDataFrame::LONG);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    for (size_t i = 0; i < intervals.size(); ++i) {
        df.val_str(i, 0, chromkey.id2chrom(intervals[i].chromid).c_str());
        df.val_long(i, 1, intervals[i].start);
        df.val_long(i, 2, intervals[i].end);
        if (i < interval_ids.size()) {
            df.val_long(i, 3, interval_ids[i]);
        } else {
            df.val_long(i, 3, (uint64_t)(i + 1));
        }
    }

    PMPY result_df = df.construct_py(true);
    result_df.to_be_stolen();
    return result_df;
}

static std::vector<GInterval> compute_screen(const std::string &expr,
                                             const std::vector<GInterval> &intervals,
                                             long iterator_policy,
                                             PyObject *progress_cb,
                                             PyObject *py_vtracks = nullptr)
{
    std::vector<GInterval> out_intervals;
    if (intervals.empty())
        return out_intervals;

    PMTrackExprScanner scanner;
    std::vector<std::string> exprs = {expr};
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::LOGICAL_T, intervals, iterator_policy, py_vtracks);

    GInterval prev_interval(-1, -1, -1);
    bool have_prev = false;

    for (; !scanner.isend(); scanner.next()) {
        bool keep = scanner.vlogical(0);

        if (keep) {
            const GInterval &cur_interval = scanner.last_interval();

            if (have_prev && prev_interval.chromid == cur_interval.chromid &&
                prev_interval.end == cur_interval.start) {
                prev_interval.end = cur_interval.end;
            } else {
                if (have_prev) {
                    out_intervals.push_back(prev_interval);
                    g_pymisha->verify_max_data_size(out_intervals.size(), "Result");
                }
                prev_interval = cur_interval;
                have_prev = true;
            }
        } else {
            if (have_prev) {
                out_intervals.push_back(prev_interval);
                g_pymisha->verify_max_data_size(out_intervals.size(), "Result");
                have_prev = false;
            }
        }

        check_interrupt();
    }

    if (have_prev) {
        out_intervals.push_back(prev_interval);
    }

    return out_intervals;
}

static void merge_adjacent(std::vector<GInterval> &intervals)
{
    if (intervals.empty())
        return;

    std::vector<GInterval> merged;
    merged.reserve(intervals.size());

    GInterval current = intervals.front();
    for (size_t i = 1; i < intervals.size(); ++i) {
        const GInterval &next = intervals[i];
        if (current.chromid == next.chromid && current.end == next.start) {
            current.end = next.end;
        } else {
            merged.push_back(current);
            current = next;
        }
    }
    merged.push_back(current);
    intervals.swap(merged);
}

static void sort_extract_result(ExtractResult &result)
{
    const size_t n = result.intervals.size();
    if (n <= 1) {
        return;
    }

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) {
        order[i] = i;
    }

    std::sort(order.begin(), order.end(), [&result](size_t a, size_t b) {
        const uint64_t ida = result.interval_ids[a];
        const uint64_t idb = result.interval_ids[b];
        if (ida != idb) {
            return ida < idb;
        }
        const GInterval &ia = result.intervals[a];
        const GInterval &ib = result.intervals[b];
        if (ia.chromid != ib.chromid) {
            return ia.chromid < ib.chromid;
        }
        if (ia.start != ib.start) {
            return ia.start < ib.start;
        }
        return ia.end < ib.end;
    });

    std::vector<GInterval> sorted_intervals;
    std::vector<uint64_t> sorted_interval_ids;
    sorted_intervals.reserve(n);
    sorted_interval_ids.reserve(n);

    std::vector<std::vector<double>> sorted_values(result.values.size());
    for (auto &vals : sorted_values) {
        vals.reserve(n);
    }

    for (size_t idx : order) {
        sorted_intervals.push_back(result.intervals[idx]);
        sorted_interval_ids.push_back(result.interval_ids[idx]);
        for (size_t iexpr = 0; iexpr < result.values.size(); ++iexpr) {
            sorted_values[iexpr].push_back(result.values[iexpr][idx]);
        }
    }

    result.intervals.swap(sorted_intervals);
    result.interval_ids.swap(sorted_interval_ids);
    result.values.swap(sorted_values);
}

// Track extraction - main function
// Called from Python as: pm_extract(exprs, intervals, iterator, config[, vtracks])
PyObject *pm_extract(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_exprs = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;
        PyObject *py_vtracks = NULL;

        if (!PyArg_ParseTuple(args, "OO|OOO", &py_exprs, &py_intervals, &py_iterator, &py_config, &py_vtracks)) {
            verror("Invalid arguments to pm_extract");
        }

        // Determine iterator policy
        long iterator_policy = 0;  // 0 = auto (use track bin size)
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_extract");

        // Convert expressions to vector of strings
        std::vector<std::string> exprs;
        if (PyUnicode_Check(py_exprs)) {
            exprs.push_back(PyUnicode_AsUTF8(py_exprs));
        } else if (PyList_Check(py_exprs)) {
            Py_ssize_t n = PyList_Size(py_exprs);
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject *item = PyList_GetItem(py_exprs, i);
                if (!PyUnicode_Check(item)) {
                    verror("Expression list must contain only strings");
                }
                exprs.push_back(PyUnicode_AsUTF8(item));
            }
        } else {
            verror("Expressions must be a string or list of strings");
        }

        // Column names are the (full) expression strings. Duplicate/colliding
        // names are de-duplicated downstream in _pymisha2df.
        std::vector<std::string> colnames = exprs;

        // Convert intervals
        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        if (intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        vdebug("pm_extract: %lu expressions, %lu intervals, iterator=%ld\n",
               exprs.size(), intervals.size(), iterator_policy);

        PMDirectAccumulator acc;

        // Range-based splitting for fixed-bin iterators
        bool use_range_split = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            struct ExtractHeader {
                uint64_t nrows;
                uint64_t nexprs;
            };

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    std::vector<uint64_t> id_remap;
                    uint64_t id_offset = 0;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub, &id_remap);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                        id_offset = start;
                    }
                    ExtractResult kid_result = run_extract(exprs, sub, iterator_policy, id_offset, progress_cb,
                                                           !id_remap.empty() ? &id_remap : nullptr, py_vtracks);

                    ExtractHeader header{kid_result.intervals.size(), exprs.size()};
                    std::vector<char> rowbuf(sizeof(int32_t) + sizeof(int64_t) * 2 + sizeof(uint64_t) +
                                             sizeof(double) * exprs.size());

                    PyMisha::lock_multitask_fifo();
                    try {
                        PyMisha::write_multitask_fifo_unlocked(&header, sizeof(header));

                        for (size_t i = 0; i < kid_result.intervals.size(); ++i) {
                            size_t offset = 0;
                            int32_t chromid = kid_result.intervals[i].chromid;
                            memcpy(rowbuf.data() + offset, &chromid, sizeof(chromid));
                            offset += sizeof(chromid);
                            memcpy(rowbuf.data() + offset, &kid_result.intervals[i].start, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            memcpy(rowbuf.data() + offset, &kid_result.intervals[i].end, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            memcpy(rowbuf.data() + offset, &kid_result.interval_ids[i], sizeof(uint64_t));
                            offset += sizeof(uint64_t);

                            for (size_t iexpr = 0; iexpr < exprs.size(); ++iexpr) {
                                double value = kid_result.values[iexpr][i];
                                memcpy(rowbuf.data() + offset, &value, sizeof(double));
                                offset += sizeof(double);
                            }

                            PyMisha::write_multitask_fifo_unlocked(rowbuf.data(), rowbuf.size());
                        }
                    } catch (...) {
                        PyMisha::unlock_multitask_fifo();
                        throw;
                    }
                    PyMisha::unlock_multitask_fifo();

                    _exit(0);
                }
            }

            // Parent: read FIFO data directly into accumulator.
            // Each kid writes [header][rows...] atomically under the FIFO lock,
            // so we read header+rows for each kid sequentially.
            // We don't know total rows upfront, so init with a reasonable estimate
            // and let the accumulator grow as needed.
            acc.init(intervals.size(), colnames);

            std::vector<double> row_values(exprs.size());
            std::vector<char> rowbuf(sizeof(int32_t) + sizeof(int64_t) * 2 + sizeof(uint64_t) +
                                     sizeof(double) * exprs.size());

            for (int kid = 0; kid < num_kids; ++kid) {
                ExtractHeader header{0, 0};
                PyMisha::read_multitask_fifo(&header, sizeof(header));

                if (header.nexprs != exprs.size()) {
                    verror("Multitask extract returned mismatched expression count");
                }

                for (uint64_t i = 0; i < header.nrows; ++i) {
                    PyMisha::read_multitask_fifo(rowbuf.data(), rowbuf.size());

                    size_t offset = 0;
                    int32_t chromid = -1;
                    int64_t start = 0;
                    int64_t end = 0;
                    uint64_t interval_id = 0;
                    memcpy(&chromid, rowbuf.data() + offset, sizeof(chromid));
                    offset += sizeof(chromid);
                    memcpy(&start, rowbuf.data() + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&end, rowbuf.data() + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&interval_id, rowbuf.data() + offset, sizeof(uint64_t));
                    offset += sizeof(uint64_t);

                    for (size_t iexpr = 0; iexpr < exprs.size(); ++iexpr) {
                        memcpy(&row_values[iexpr], rowbuf.data() + offset, sizeof(double));
                        offset += sizeof(double);
                    }

                    acc.write_row(chromid, start, end, row_values.data(), interval_id);
                }
            }

            sort_accumulator(acc);

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            // Single-process path: scan directly into NumPy arrays
            run_extract_direct(acc, exprs, colnames, intervals, iterator_policy, 0,
                               progress_cb, nullptr, py_vtracks);
        }

        if (acc.size() == 0) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        g_pymisha->verify_max_data_size(acc.size(), "Result");

        vdebug("pm_extract: collected %lu results\n", acc.size());

        PMPY result_df = acc.finalize(true);
        result_df.to_be_stolen();
        return (PyObject *)result_df;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Called from Python as: pm_iterate(intervals, iterator, config)
PyObject *pm_iterate(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "O|OO", &py_intervals, &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_iterate");
        }

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        long iterator_policy = -1;  // default: intervals iterator
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_iterate");
        bool interval_relative = get_config_bool(py_config, "interval_relative", false);

        if (iterator_policy == 0) {
            iterator_policy = -1;
        }

        std::unique_ptr<PMTrackExpressionIterator> itr;
        if (iterator_policy > 0) {
            itr = std::make_unique<PMFixedBinIterator>(intervals, iterator_policy, interval_relative);
        } else {
            itr = std::make_unique<PMIntervalsIterator>(intervals);
        }

        itr->begin();

        std::vector<GInterval> out_intervals;
        std::vector<uint64_t> out_interval_ids;

        uint64_t est_size = itr->size();
        if (est_size > 0) {
            out_intervals.reserve(est_size);
            out_interval_ids.reserve(est_size);
        }

        for (; !itr->isend(); itr->next()) {
            out_intervals.push_back(itr->last_interval());
            out_interval_ids.push_back(itr->original_interval_idx());
            g_pymisha->verify_max_data_size(out_intervals.size(), "Result");
            check_interrupt();
        }

        PMPY result_df = build_intervals_df(out_intervals, out_interval_ids);
        result_df.to_be_stolen();
        return (PyObject *)result_df;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Screen intervals by expression - keep intervals where expression evaluates to True
// Called from Python as: pm_screen(expr, intervals[, iterator, config, vtracks])
PyObject *pm_screen(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;
        PyObject *py_vtracks = NULL;

        if (!PyArg_ParseTuple(args, "OO|OOO", &py_expr, &py_intervals, &py_iterator, &py_config, &py_vtracks)) {
            verror("Invalid arguments to pm_screen");
        }

        // Check that expr is a single string
        if (!PyUnicode_Check(py_expr)) {
            verror("gscreen expression must be a string");
        }
        std::string expr = PyUnicode_AsUTF8(py_expr);

        // Determine iterator policy
        long iterator_policy = 0;  // 0 = auto (use track bin size)
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_screen");

        // Convert intervals
        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        if (intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        vdebug("pm_screen: expr='%s', %lu intervals, iterator=%ld\n",
               expr.c_str(), intervals.size(), iterator_policy);

        std::vector<GInterval> out_intervals;

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_screen = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split_screen) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            struct ScreenHeader {
                uint64_t nrows;
            };

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                    }
                    std::vector<GInterval> kid_intervals = compute_screen(expr, sub, iterator_policy, progress_cb, py_vtracks);

                    ScreenHeader header{kid_intervals.size()};
                    std::vector<char> rowbuf(sizeof(int32_t) + sizeof(int64_t) * 2);

                    PyMisha::lock_multitask_fifo();
                    try {
                        PyMisha::write_multitask_fifo_unlocked(&header, sizeof(header));

                        for (const auto &interval : kid_intervals) {
                            size_t offset = 0;
                            int32_t chromid = interval.chromid;
                            memcpy(rowbuf.data() + offset, &chromid, sizeof(chromid));
                            offset += sizeof(chromid);
                            memcpy(rowbuf.data() + offset, &interval.start, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            memcpy(rowbuf.data() + offset, &interval.end, sizeof(int64_t));
                            PyMisha::write_multitask_fifo_unlocked(rowbuf.data(), rowbuf.size());
                        }
                    } catch (...) {
                        PyMisha::unlock_multitask_fifo();
                        throw;
                    }
                    PyMisha::unlock_multitask_fifo();

                    _exit(0);
                }
            }

            for (int kid = 0; kid < num_kids; ++kid) {
                ScreenHeader header{0};
                PyMisha::read_multitask_fifo(&header, sizeof(header));

                std::vector<char> rowbuf(sizeof(int32_t) + sizeof(int64_t) * 2);
                for (uint64_t i = 0; i < header.nrows; ++i) {
                    PyMisha::read_multitask_fifo(rowbuf.data(), rowbuf.size());

                    size_t offset = 0;
                    int32_t chromid = -1;
                    int64_t start = 0;
                    int64_t end = 0;
                    memcpy(&chromid, rowbuf.data() + offset, sizeof(chromid));
                    offset += sizeof(chromid);
                    memcpy(&start, rowbuf.data() + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&end, rowbuf.data() + offset, sizeof(int64_t));

                    out_intervals.emplace_back(chromid, start, end);
                }
            }

            std::sort(out_intervals.begin(), out_intervals.end(),
                      [](const GInterval &a, const GInterval &b) {
                          if (a.chromid != b.chromid)
                              return a.chromid < b.chromid;
                          if (a.start != b.start)
                              return a.start < b.start;
                          return a.end < b.end;
                      });

            merge_adjacent(out_intervals);

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            out_intervals = compute_screen(expr, intervals, iterator_policy, progress_cb, py_vtracks);
        }

        if (out_intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        g_pymisha->verify_max_data_size(out_intervals.size(), "Result");

        vdebug("pm_screen: %lu intervals passed filter\n", out_intervals.size());

        // Build result DataFrame
        // Columns: chrom, start, end
        PMDataFrame df(out_intervals.size(), 3, "intervals");
        df.init_col(0, "chrom", PMDataFrame::STR);
        df.init_col(1, "start", PMDataFrame::LONG);
        df.init_col(2, "end", PMDataFrame::LONG);

        // Fill data
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        for (size_t i = 0; i < out_intervals.size(); ++i) {
            df.val_str(i, 0, chromkey.id2chrom(out_intervals[i].chromid).c_str());
            df.val_long(i, 1, out_intervals[i].start);
            df.val_long(i, 2, out_intervals[i].end);
        }

        PMPY result = df.construct_py(true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

struct SummaryStats {
    double num_bins{0};
    double num_non_nan{0};
    double total{0};
    double minval{std::numeric_limits<double>::max()};
    double maxval{-std::numeric_limits<double>::max()};
    double mean_square_sum{0};

    void update(double v) {
        ++num_bins;
        if (!std::isnan(v)) {
            ++num_non_nan;
            total += v;
            minval = std::min(minval, v);
            maxval = std::max(maxval, v);
            mean_square_sum += v * v;
        }
    }

    double mean() const { return total / num_non_nan; }

    double stdev() const {
        double mean_val = mean();
        // Clamp tiny negatives from catastrophic cancellation (the two terms
        // are close when variance is small relative to mean^2); return 0
        // instead of NaN on nearly-constant input.
        double var = mean_square_sum / (num_non_nan - 1) -
                     (mean_val * mean_val) * (num_non_nan / (num_non_nan - 1));
        return var > 0 ? std::sqrt(var) : 0;
    }

    void merge(const SummaryStats &other) {
        num_bins += other.num_bins;
        num_non_nan += other.num_non_nan;
        total += other.total;
        mean_square_sum += other.mean_square_sum;
        if (other.num_non_nan > 0) {
            minval = std::min(minval, other.minval);
            maxval = std::max(maxval, other.maxval);
        }
    }
};

static SummaryStats compute_summary(const std::string &expr,
                                    const std::vector<GInterval> &intervals,
                                    long iterator_policy,
                                    PyObject *progress_cb)
{
    SummaryStats summary;
    if (intervals.empty())
        return summary;

    PMTrackExprScanner scanner;
    std::vector<std::string> exprs = {expr};
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    for (; !scanner.isend(); scanner.next()) {
        summary.update(scanner.vdouble(0));
        check_interrupt();
    }

    return summary;
}

static const int SUMMARY_COLS = 7;

static void fill_summary_values(const std::vector<SummaryStats> &summaries,
                                std::vector<double> &out)
{
    out.assign(summaries.size() * SUMMARY_COLS, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < summaries.size(); ++i) {
        const SummaryStats &s = summaries[i];
        size_t base = i * SUMMARY_COLS;

        double total = s.num_bins;
        double nan = s.num_bins - s.num_non_nan;
        double min_val = std::numeric_limits<double>::quiet_NaN();
        double max_val = std::numeric_limits<double>::quiet_NaN();
        double sum_val = std::numeric_limits<double>::quiet_NaN();
        double mean_val = std::numeric_limits<double>::quiet_NaN();
        double stdev_val = std::numeric_limits<double>::quiet_NaN();

        if (s.num_non_nan > 0) {
            min_val = s.minval;
            max_val = s.maxval;
            sum_val = s.total;
            mean_val = s.mean();
            if (s.num_non_nan > 1) {
                stdev_val = s.stdev();
            }
        }

        out[base] = total;
        out[base + 1] = nan;
        out[base + 2] = min_val;
        out[base + 3] = max_val;
        out[base + 4] = sum_val;
        out[base + 5] = mean_val;
        out[base + 6] = stdev_val;
    }
}

static void compute_interval_summaries(const std::string &expr,
                                       const std::vector<GInterval> &intervals,
                                       long iterator_policy,
                                       std::vector<SummaryStats> &summaries,
                                       PyObject *progress_cb)
{
    summaries.assign(intervals.size(), SummaryStats());
    if (intervals.empty())
        return;

    PMTrackExprScanner scanner;
    std::vector<std::string> exprs = {expr};
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    for (; !scanner.isend(); scanner.next()) {
        uint64_t interval_id = scanner.last_interval_id();
        if (interval_id > 0 && interval_id <= intervals.size()) {
            summaries[interval_id - 1].update(scanner.vdouble(0));
        }
        check_interrupt();
    }
}

static int choose_num_kids(uint64_t num_intervals, int64_t total_scope_bp)
{
    if (!g_pymisha->multitasking_avail())
        return 0;

    unsigned int hw = std::thread::hardware_concurrency();
    int target = hw ? static_cast<int>(hw) : g_pymisha->max_processes();
    target = std::max(target, g_pymisha->min_processes());
    target = std::min(target, g_pymisha->max_processes());

    if (target < 2 || num_intervals < static_cast<uint64_t>(target))
        return 0;

    // Workload floors: fork-based multitasking pays an upfront cost
    // (fork + sem/mmap setup) plus per-row FIFO marshalling. Below these
    // thresholds, parallel is measured to lose by 20-200x vs serial on
    // hg38/Phylo447 sparse-track gextract/gquantiles/gsummary/gscreen.
    uint64_t min_intervs = g_pymisha->min_intervs4process();
    if (min_intervs > 0 && num_intervals < min_intervs)
        return 0;

    if (total_scope_bp >= 0) {
        uint64_t min_scope = g_pymisha->min_scope4process();
        if (min_scope > 0) {
            uint64_t scope_per_kid = (uint64_t)total_scope_bp / (uint64_t)target;
            if (scope_per_kid < min_scope)
                return 0;
        }
    }

    return target;
}

// Summarize expression values over intervals
PyObject *pm_summary(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OO|OO", &py_expr, &py_intervals, &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_summary");
        }

        if (!PyUnicode_Check(py_expr)) {
            verror("gsummary expression must be a string");
        }

        std::string expr = PyUnicode_AsUTF8(py_expr);

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_summary");

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        SummaryStats summary;

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_sum = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split_sum) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }
        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                    }
                    SummaryStats kid_summary = compute_summary(expr, sub, iterator_policy, progress_cb);
                    PyMisha::write_multitask_fifo(&kid_summary, sizeof(kid_summary));
                    _exit(0);
                }
            }

            for (int kid = 0; kid < num_kids; ++kid) {
                SummaryStats kid_summary;
                PyMisha::read_multitask_fifo(&kid_summary, sizeof(kid_summary));
                summary.merge(kid_summary);
            }

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            summary = compute_summary(expr, intervals, iterator_policy, progress_cb);
        }

        double total_bins = summary.num_bins;
        double nan_bins = summary.num_bins - summary.num_non_nan;
        double min_val = std::numeric_limits<double>::quiet_NaN();
        double max_val = std::numeric_limits<double>::quiet_NaN();
        double sum_val = std::numeric_limits<double>::quiet_NaN();
        double mean_val = std::numeric_limits<double>::quiet_NaN();
        double stdev_val = std::numeric_limits<double>::quiet_NaN();

        if (summary.num_non_nan > 0) {
            min_val = summary.minval;
            max_val = summary.maxval;
            sum_val = summary.total;
            mean_val = summary.mean();
            if (summary.num_non_nan > 1) {
                stdev_val = summary.stdev();
            }
        }

        PMPY result(PyDict_New(), true);
        auto set_double = [&](const char *key, double val) {
            PMPY py_val(PyFloat_FromDouble(val), true);
            if (!py_val || PyDict_SetItemString(result, key, py_val) < 0)
                verror("Failed to set summary result for key '%s'", key);
        };
        set_double("Total intervals", total_bins);
        set_double("NaN intervals", nan_bins);
        set_double("Min", min_val);
        set_double("Max", max_val);
        set_double("Sum", sum_val);
        set_double("Mean", mean_val);
        set_double("Std dev", stdev_val);

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

struct Percentile {
    double percentile{0};
    int64_t index{0};
    bool estimation{false};

    Percentile() {}
    Percentile(double _percentile, int64_t _index) : percentile(_percentile), index(_index) {}
    bool operator<(const Percentile &p) const { return percentile < p.percentile; }
};

static bool calc_quantiles(StreamPercentiler<double> &sp, std::vector<Percentile> &percentiles,
                           std::vector<double> &quantiles)
{
    bool estimated_results = false;

    // Fast path (port of R GenomeTrackQuantiles.cpp): the entire stream fits the
    // reservoir, so sp.samples() holds every non-NaN value with no sub-sampling.
    // Select the needed order statistics with nth_element-on-suffix (O(k*N)) and
    // interpolate, instead of the O(N log N) std::sort that get_percentile runs
    // on first call. Output is identical (same (N-1)*pct linear interpolation,
    // and all results are exact so the monotonicity repair below is a no-op).
    if (sp.stream_size() && sp.stream_size() <= sp.max_rnd_sampling_buf_size()) {
        std::vector<double> &samples = sp.samples_mutable();
        const uint64_t N = samples.size();

        std::vector<uint64_t> targets;
        targets.reserve(percentiles.size() * 2);
        for (const auto &p : percentiles) {
            double pct = std::max(0.0, std::min(1.0, p.percentile));
            double idx = (double)(N - 1) * pct;
            targets.push_back((uint64_t)std::floor(idx));
            targets.push_back((uint64_t)std::ceil(idx));
        }
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

        std::vector<double> values_at_targets(targets.size());
        const size_t log2N = (N <= 2) ? 1 : (size_t)std::ceil(std::log2((double)N));
        const size_t crossover = std::max<size_t>(64, 2 * log2N);

        if (targets.size() <= crossover) {
            // Ascending walk: each nth_element partitions only the unconsumed
            // suffix [prev_end, end); samples[0..prev_end) are already the
            // globally smallest, so samples[r] becomes the global r-th smallest.
            uint64_t prev_end = 0;
            for (size_t t = 0; t < targets.size(); ++t) {
                uint64_t r = targets[t];
                std::nth_element(samples.begin() + prev_end, samples.begin() + r, samples.end());
                values_at_targets[t] = samples[r];
                prev_end = r + 1;
            }
        } else {
            std::sort(samples.begin(), samples.end());
            for (size_t t = 0; t < targets.size(); ++t)
                values_at_targets[t] = samples[targets[t]];
        }

        for (auto &p : percentiles) {
            double pct = std::max(0.0, std::min(1.0, p.percentile));
            double idx = (double)(N - 1) * pct;
            uint64_t i1 = (uint64_t)std::floor(idx);
            uint64_t i2 = (uint64_t)std::ceil(idx);
            double w = idx - (double)i1;
            size_t pos1 = std::lower_bound(targets.begin(), targets.end(), i1) - targets.begin();
            size_t pos2 = std::lower_bound(targets.begin(), targets.end(), i2) - targets.begin();
            quantiles[p.index] = values_at_targets[pos1] * (1.0 - w) + values_at_targets[pos2] * w;
            p.estimation = false;
        }
        return false;
    }

    if (sp.stream_size()) {
        for (auto &p : percentiles) {
            bool estimated = false;
            quantiles[p.index] = sp.get_percentile(p.percentile, estimated);
            p.estimation = estimated;
            if (estimated) {
                estimated_results = true;
            }
        }

        for (size_t i = 1; i < percentiles.size(); ++i) {
            if (percentiles[i].estimation) {
                double prev = quantiles[percentiles[i - 1].index];
                double &cur = quantiles[percentiles[i].index];
                if (!std::isnan(prev) && (std::isnan(cur) || cur < prev)) {
                    cur = prev;
                }
            }
        }
        for (size_t i = percentiles.size(); i-- > 1;) {
            if (percentiles[i - 1].estimation) {
                double next = quantiles[percentiles[i].index];
                double &cur = quantiles[percentiles[i - 1].index];
                if (!std::isnan(next) && (std::isnan(cur) || cur > next)) {
                    cur = next;
                }
            }
        }
    } else {
        for (auto &p : percentiles) {
            quantiles[p.index] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return estimated_results;
}

static void fill_stream_percentiler(const std::string &expr,
                                    const std::vector<GInterval> &intervals,
                                    long iterator_policy,
                                    StreamPercentiler<double> &sp,
                                    PyObject *progress_cb)
{
    if (intervals.empty())
        return;

    PMTrackExprScanner scanner;
    std::vector<std::string> exprs = {expr};
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    for (; !scanner.isend(); scanner.next()) {
        double val = scanner.vdouble(0);
        if (!std::isnan(val))
            sp.add(val, pm::pm_rnd_func);
        check_interrupt();
    }
}

static bool compute_interval_quantiles(const std::string &expr,
                                        const std::vector<GInterval> &intervals,
                                        long iterator_policy,
                                        std::vector<Percentile> &percentiles,
                                        std::vector<double> &quantiles,
                                        PyObject *progress_cb)
{
    const size_t num_intervals = intervals.size();
    const size_t num_percentiles = percentiles.size();

    quantiles.assign(num_intervals * num_percentiles, std::numeric_limits<double>::quiet_NaN());
    if (intervals.empty())
        return false;

    PMTrackExprScanner scanner;
    std::vector<std::string> exprs = {expr};
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    StreamPercentiler<double> sp(g_pymisha->max_data_size(), 0, 0);
    std::vector<double> interval_quantiles(num_percentiles, std::numeric_limits<double>::quiet_NaN());
    uint64_t cur_interval_id = 0;
    bool estimated_any = false;

    for (; !scanner.isend(); scanner.next()) {
        uint64_t interval_id = scanner.last_interval_id();
        if (interval_id == 0 || interval_id > num_intervals) {
            check_interrupt();
            continue;
        }

        if (cur_interval_id == 0) {
            cur_interval_id = interval_id;
            sp.reset();
        } else if (interval_id != cur_interval_id) {
            bool estimated = calc_quantiles(sp, percentiles, interval_quantiles);
            estimated_any = estimated_any || estimated;

            size_t idx = cur_interval_id - 1;
            for (size_t j = 0; j < num_percentiles; ++j) {
                quantiles[idx * num_percentiles + j] = interval_quantiles[j];
            }

            sp.reset();
            cur_interval_id = interval_id;
        }

        double val = scanner.vdouble(0);
        if (!std::isnan(val))
            sp.add(val, pm::pm_rnd_func);

        check_interrupt();
    }

    if (cur_interval_id != 0) {
        bool estimated = calc_quantiles(sp, percentiles, interval_quantiles);
        estimated_any = estimated_any || estimated;

        size_t idx = cur_interval_id - 1;
        for (size_t j = 0; j < num_percentiles; ++j) {
            quantiles[idx * num_percentiles + j] = interval_quantiles[j];
        }
    }

    return estimated_any;
}

// Compute quantiles for expression values over intervals
PyObject *pm_quantiles(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_percentiles = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOO|OO", &py_expr, &py_percentiles, &py_intervals, &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_quantiles");
        }

        if (!PyUnicode_Check(py_expr)) {
            verror("gquantiles expression must be a string");
        }

        std::string expr = PyUnicode_AsUTF8(py_expr);

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_quantiles");

        PMPY percentiles_seq(PySequence_Fast(py_percentiles, "percentiles must be a sequence"), true);
        if (!percentiles_seq) {
            verror("percentiles must be a sequence of numbers");
        }

        Py_ssize_t n = PySequence_Fast_GET_SIZE((PyObject *)percentiles_seq);
        if (n <= 0) {
            verror("percentiles must contain at least one value");
        }

        std::vector<Percentile> percentiles;
        percentiles.reserve(n);

        PyObject *percentiles_obj = (PyObject *)percentiles_seq;
        PyObject **items = PySequence_Fast_ITEMS(percentiles_obj);
        for (Py_ssize_t i = 0; i < n; ++i) {
            double p = PyFloat_AsDouble(items[i]);
            if (PyErr_Occurred()) {
                PyErr_Clear();
                verror("percentiles must be numeric");
            }
            if (p < 0.0 || p > 1.0) {
                verror("Percentile (%g) is not in [0, 1] range", p);
            }
            percentiles.emplace_back(p, i);
        }

        std::sort(percentiles.begin(), percentiles.end());

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        std::vector<double> quantiles(n, std::numeric_limits<double>::quiet_NaN());

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_q = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split_q) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }
        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            uint64_t kid_sampling_buf_size = (uint64_t)std::ceil(g_pymisha->max_data_size() / (double)num_kids);

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                    }

                    StreamPercentiler<double> sp(kid_sampling_buf_size, 0, 0);
                    fill_stream_percentiler(expr, sub, iterator_policy, sp, progress_cb);

                    uint64_t stream_size = sp.stream_size();
                    uint64_t sample_size = sp.samples().size();

                    struct Header {
                        uint64_t stream_size;
                        uint64_t sample_size;
                    } header{stream_size, sample_size};

                    std::vector<char> buffer(sizeof(header) + sample_size * sizeof(double));
                    memcpy(buffer.data(), &header, sizeof(header));
                    if (sample_size > 0) {
                        memcpy(buffer.data() + sizeof(header), sp.samples().data(), sample_size * sizeof(double));
                    }

                    PyMisha::write_multitask_fifo(buffer.data(), buffer.size());
                    _exit(0);
                }
            }

            double min_sampling_rate = 1.0;
            uint64_t total_stream_size = 0;
            std::vector<std::vector<double>> kid_samples;
            std::vector<uint64_t> kid_stream_sizes;
            std::vector<uint64_t> kid_sample_sizes;
            kid_samples.reserve(num_kids);
            kid_stream_sizes.reserve(num_kids);
            kid_sample_sizes.reserve(num_kids);

            for (int kid = 0; kid < num_kids; ++kid) {
                struct Header {
                    uint64_t stream_size;
                    uint64_t sample_size;
                } header{0, 0};

                PyMisha::read_multitask_fifo(&header, sizeof(header));
                std::vector<double> samples(header.sample_size);
                if (header.sample_size > 0) {
                    PyMisha::read_multitask_fifo(samples.data(), header.sample_size * sizeof(double));
                }

                if (header.stream_size > 0) {
                    double rate = header.sample_size / (double)header.stream_size;
                    min_sampling_rate = std::min(min_sampling_rate, rate);
                    total_stream_size += header.stream_size;
                }
                kid_stream_sizes.push_back(header.stream_size);
                kid_sample_sizes.push_back(header.sample_size);
                kid_samples.push_back(std::move(samples));
            }

            std::vector<double> merged_samples;
            merged_samples.reserve(g_pymisha->max_data_size());

            for (int kid = 0; kid < num_kids; ++kid) {
                const auto &samples = kid_samples[kid];
                if (samples.empty() || kid_stream_sizes[kid] == 0) {
                    continue;
                }

                if (min_sampling_rate >= 1.0) {
                    merged_samples.insert(merged_samples.end(), samples.begin(), samples.end());
                    continue;
                }

                double kid_rate = kid_sample_sizes[kid] / (double)kid_stream_sizes[kid];
                double sampling_ratio = kid_rate > 0 ? (min_sampling_rate / kid_rate) : 0.0;
                for (double sample : samples) {
                    if (pm::pm_rnd_func() < sampling_ratio) {
                        merged_samples.push_back(sample);
                    }
                }
            }

            StreamPercentiler<double> sp;
            std::vector<double> empty_low;
            std::vector<double> empty_high;
            // Skip the O(N log N) sort of the merged reservoir: calc_quantiles
            // selects the needed order statistics with nth_element (exact path)
            // and falls back to a lazy sort inside get_percentile only for the
            // sub-sampling (approximate) case.
            sp.init_with_swap(total_stream_size, merged_samples, empty_low, empty_high, /*do_sort=*/false);

            bool estimated = calc_quantiles(sp, percentiles, quantiles);
            if (estimated || min_sampling_rate < 1.0) {
                PyErr_WarnEx(PyExc_RuntimeWarning,
                             "Data size exceeds the limit; quantiles are approximate. "
                             "Adjust CONFIG['max_data_size'] to increase the limit.",
                             1);
            }

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            if (!intervals.empty()) {
                uint64_t sampling_buf_size = g_pymisha->max_data_size();
                StreamPercentiler<double> sp(sampling_buf_size, 0, 0);
                fill_stream_percentiler(expr, intervals, iterator_policy, sp, progress_cb);

                bool estimated = calc_quantiles(sp, percentiles, quantiles);
                if (estimated) {
                    PyErr_WarnEx(PyExc_RuntimeWarning,
                                 "Data size exceeds the limit; quantiles are approximate. "
                                 "Adjust CONFIG['max_data_size'] to increase the limit.",
                                 1);
                }
            }
        }

        PMPY result(PyList_New(n), true);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyList_SET_ITEM((PyObject *)result, i, PyFloat_FromDouble(quantiles[i]));
        }

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Summarize expression values per interval (streaming)
PyObject *pm_intervals_summary(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OO|OO", &py_expr, &py_intervals, &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_intervals_summary");
        }

        if (!PyUnicode_Check(py_expr)) {
            verror("gintervals_summary expression must be a string");
        }

        std::string expr = PyUnicode_AsUTF8(py_expr);

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_intervals_summary");

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        g_pymisha->verify_max_data_size(intervals.size(), "Result");

        std::vector<double> values;
        int num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    size_t start = (intervals.size() * kid) / num_kids;
                    size_t end = (intervals.size() * (kid + 1)) / num_kids;
                    std::vector<GInterval> sub(intervals.begin() + start, intervals.begin() + end);

                    std::vector<SummaryStats> summaries;
                    compute_interval_summaries(expr, sub, iterator_policy, summaries, progress_cb);

                    std::vector<double> kid_values;
                    fill_summary_values(summaries, kid_values);

                    struct Header {
                        uint64_t start_idx;
                        uint64_t count;
                    } header{start, sub.size()};

                    PyMisha::write_multitask_fifo(&header, sizeof(header));
                    if (!kid_values.empty()) {
                        PyMisha::write_multitask_fifo(kid_values.data(),
                                                      kid_values.size() * sizeof(double));
                    }

                    _exit(0);
                }
            }

            values.assign(intervals.size() * SUMMARY_COLS, std::numeric_limits<double>::quiet_NaN());

            for (int kid = 0; kid < num_kids; ++kid) {
                struct Header {
                    uint64_t start_idx;
                    uint64_t count;
                } header{0, 0};

                PyMisha::read_multitask_fifo(&header, sizeof(header));
                std::vector<double> kid_values(header.count * SUMMARY_COLS);
                if (!kid_values.empty()) {
                    PyMisha::read_multitask_fifo(kid_values.data(),
                                                 kid_values.size() * sizeof(double));
                }

                for (size_t i = 0; i < header.count; ++i) {
                    size_t dst = (header.start_idx + i) * SUMMARY_COLS;
                    size_t src = i * SUMMARY_COLS;
                    for (int c = 0; c < SUMMARY_COLS; ++c) {
                        values[dst + c] = kid_values[src + c];
                    }
                }
            }

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            std::vector<SummaryStats> summaries;
            compute_interval_summaries(expr, intervals, iterator_policy, summaries, progress_cb);
            fill_summary_values(summaries, values);
        }

        PMDataFrame df(intervals.size(), 3 + SUMMARY_COLS, "intervals");
        df.init_col(0, "chrom", PMDataFrame::STR);
        df.init_col(1, "start", PMDataFrame::LONG);
        df.init_col(2, "end", PMDataFrame::LONG);
        df.init_col(3, "Total intervals", PMDataFrame::DOUBLE);
        df.init_col(4, "NaN intervals", PMDataFrame::DOUBLE);
        df.init_col(5, "Min", PMDataFrame::DOUBLE);
        df.init_col(6, "Max", PMDataFrame::DOUBLE);
        df.init_col(7, "Sum", PMDataFrame::DOUBLE);
        df.init_col(8, "Mean", PMDataFrame::DOUBLE);
        df.init_col(9, "Std dev", PMDataFrame::DOUBLE);

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        for (size_t i = 0; i < intervals.size(); ++i) {
            df.val_str(i, 0, chromkey.id2chrom(intervals[i].chromid).c_str());
            df.val_long(i, 1, intervals[i].start);
            df.val_long(i, 2, intervals[i].end);

            size_t base = i * SUMMARY_COLS;
            df.val_double(i, 3, values[base]);
            df.val_double(i, 4, values[base + 1]);
            df.val_double(i, 5, values[base + 2]);
            df.val_double(i, 6, values[base + 3]);
            df.val_double(i, 7, values[base + 4]);
            df.val_double(i, 8, values[base + 5]);
            df.val_double(i, 9, values[base + 6]);
        }

        PMPY result = df.construct_py(true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Shortest decimal string that round-trips `v`, so nearby percentiles don't
// collapse to the same column name the way a fixed "%g" (6 sig digits) would
// (e.g. 0.123456789 and 0.1234567891 both -> "0.123457"). PMDataFrame::init_col
// dedups any genuine duplicates by appending '_'.
static std::string format_percentile(double v)
{
    char buf[64];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v)
            return buf;
    }
    return buf;
}

// Compute quantiles per interval (streaming)
PyObject *pm_intervals_quantiles(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_percentiles = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOO|OO", &py_expr, &py_percentiles, &py_intervals, &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_intervals_quantiles");
        }

        if (!PyUnicode_Check(py_expr)) {
            verror("gintervals_quantiles expression must be a string");
        }

        std::string expr = PyUnicode_AsUTF8(py_expr);

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_intervals_quantiles");

        PMPY percentiles_seq(PySequence_Fast(py_percentiles, "percentiles must be a sequence"), true);
        if (!percentiles_seq) {
            verror("percentiles must be a sequence of numbers");
        }

        Py_ssize_t n = PySequence_Fast_GET_SIZE((PyObject *)percentiles_seq);
        if (n <= 0) {
            verror("percentiles must contain at least one value");
        }

        std::vector<double> pct_values(n);
        std::vector<Percentile> percentiles;
        percentiles.reserve(n);

        PyObject *percentiles_obj = (PyObject *)percentiles_seq;
        PyObject **items = PySequence_Fast_ITEMS(percentiles_obj);
        for (Py_ssize_t i = 0; i < n; ++i) {
            double p = PyFloat_AsDouble(items[i]);
            if (PyErr_Occurred()) {
                PyErr_Clear();
                verror("percentiles must be numeric");
            }
            if (p < 0.0 || p > 1.0) {
                verror("Percentile (%g) is not in [0, 1] range", p);
            }
            pct_values[i] = p;
            percentiles.emplace_back(p, i);
        }

        std::sort(percentiles.begin(), percentiles.end());

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        g_pymisha->verify_max_data_size(intervals.size(), "Result");

        std::vector<double> quantiles;
        bool estimated_any = false;

        int num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    size_t start = (intervals.size() * kid) / num_kids;
                    size_t end = (intervals.size() * (kid + 1)) / num_kids;
                    std::vector<GInterval> sub(intervals.begin() + start, intervals.begin() + end);

                    std::vector<double> kid_quantiles;
                    bool estimated = compute_interval_quantiles(expr, sub, iterator_policy,
                                                                percentiles, kid_quantiles, progress_cb);

                    struct Header {
                        uint64_t start_idx;
                        uint64_t count;
                        uint64_t estimated;
                    } header{start, sub.size(), estimated ? 1ULL : 0ULL};

                    PyMisha::write_multitask_fifo(&header, sizeof(header));
                    if (!kid_quantiles.empty()) {
                        PyMisha::write_multitask_fifo(kid_quantiles.data(),
                                                      kid_quantiles.size() * sizeof(double));
                    }

                    _exit(0);
                }
            }

            quantiles.assign(intervals.size() * static_cast<size_t>(n),
                             std::numeric_limits<double>::quiet_NaN());

            for (int kid = 0; kid < num_kids; ++kid) {
                struct Header {
                    uint64_t start_idx;
                    uint64_t count;
                    uint64_t estimated;
                } header{0, 0, 0};

                PyMisha::read_multitask_fifo(&header, sizeof(header));
                std::vector<double> kid_quantiles(header.count * static_cast<size_t>(n));
                if (!kid_quantiles.empty()) {
                    PyMisha::read_multitask_fifo(kid_quantiles.data(),
                                                 kid_quantiles.size() * sizeof(double));
                }

                if (header.estimated)
                    estimated_any = true;

                for (size_t i = 0; i < header.count; ++i) {
                    size_t dst = (header.start_idx + i) * static_cast<size_t>(n);
                    size_t src = i * static_cast<size_t>(n);
                    for (Py_ssize_t j = 0; j < n; ++j) {
                        quantiles[dst + static_cast<size_t>(j)] = kid_quantiles[src + static_cast<size_t>(j)];
                    }
                }
            }

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            estimated_any = compute_interval_quantiles(expr, intervals, iterator_policy,
                                                      percentiles, quantiles, progress_cb);
        }

        if (estimated_any) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Data size exceeds the limit; quantiles are approximate. "
                         "Adjust CONFIG['max_data_size'] to increase the limit.",
                         1);
        }

        PMDataFrame df(intervals.size(), 3 + n, "intervals");
        df.init_col(0, "chrom", PMDataFrame::STR);
        df.init_col(1, "start", PMDataFrame::LONG);
        df.init_col(2, "end", PMDataFrame::LONG);

        std::vector<std::string> colnames;
        colnames.reserve(n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            colnames.emplace_back(format_percentile(pct_values[i]));
            df.init_col(3 + i, colnames.back().c_str(), PMDataFrame::DOUBLE);
        }

        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        for (size_t i = 0; i < intervals.size(); ++i) {
            df.val_str(i, 0, chromkey.id2chrom(intervals[i].chromid).c_str());
            df.val_long(i, 1, intervals[i].start);
            df.val_long(i, 2, intervals[i].end);

            size_t base = i * static_cast<size_t>(n);
            for (Py_ssize_t j = 0; j < n; ++j) {
                df.val_double(i, 3 + j, quantiles[base + static_cast<size_t>(j)]);
            }
        }

        PMPY result = df.construct_py(true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Get track names
PyObject *pm_track_names(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        std::vector<std::string> tracks = g_pmdb->track_names();

        PMPY result(PyList_New(tracks.size()), true);
        for (size_t i = 0; i < tracks.size(); ++i) {
            PyList_SET_ITEM((PyObject *)result, i, PyUnicode_FromString(tracks[i].c_str()));
        }

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Get interval-set names (cached alongside the track scan).
PyObject *pm_interv_names(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        std::vector<std::string> names = g_pmdb->interv_names();

        PMPY result(PyList_New(names.size()), true);
        for (size_t i = 0; i < names.size(); ++i) {
            PyList_SET_ITEM((PyObject *)result, i, PyUnicode_FromString(names[i].c_str()));
        }

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Incrementally register an interval-set name in the PMDb cache.
PyObject *pm_interv_register(PyObject *self, PyObject *args)
{
    try {
        const char *name = nullptr;
        if (!PyArg_ParseTuple(args, "s", &name)) {
            verror("Invalid arguments to pm_interv_register");
        }
        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }
        g_pmdb->register_interv(name);
        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    }
}

// Incrementally remove an interval-set name from the PMDb cache.
PyObject *pm_interv_unregister(PyObject *self, PyObject *args)
{
    try {
        const char *name = nullptr;
        if (!PyArg_ParseTuple(args, "s", &name)) {
            verror("Invalid arguments to pm_interv_unregister");
        }
        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }
        g_pmdb->unregister_interv(name);
        Py_RETURN_NONE;
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    }
}

// Get track information
PyObject *pm_track_info(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track_name = NULL;
        if (!PyArg_ParseTuple(args, "s", &track_name)) {
            verror("Invalid arguments to pm_track_info");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        // Check track exists
        if (!g_pmdb->track_exists(track_name)) {
            verror("Track '%s' does not exist", track_name);
        }

        std::string track_path = g_pmdb->track_path(track_name);
        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Get track type
        GenomeTrack::Type track_type = GenomeTrack::get_type(track_path.c_str(), chromkey, false);

        // Map type to user-friendly name
        const char *type_name;
        switch (track_type) {
            case GenomeTrack::FIXED_BIN: type_name = "dense"; break;
            case GenomeTrack::SPARSE:    type_name = "sparse"; break;
            case GenomeTrack::ARRAYS:    type_name = "array"; break;
            case GenomeTrack::RECTS:     type_name = "rectangles"; break;
            case GenomeTrack::POINTS:    type_name = "points"; break;
            case GenomeTrack::COMPUTED:  type_name = "computed"; break;
            default:                     type_name = GenomeTrack::TYPE_NAMES[track_type]; break;
        }

        // Create result dictionary
        PMPY result(PyDict_New(), true);

        // type
        PMPY py_type(PyUnicode_FromString(type_name), true);
        PyDict_SetItemString(result, "type", py_type);

        // dimensions
        int dimensions = GenomeTrack::is_1d(track_type) ? 1 : 2;
        PMPY py_dims(PyLong_FromLong(dimensions), true);
        PyDict_SetItemString(result, "dimensions", py_dims);

        // Calculate size in bytes and format
        uint64_t total_size = 0;
        struct stat st;
        std::string idx_path = track_path + "/track.idx";
        std::string dat_path = track_path + "/track.dat";
        bool is_indexed = (stat(idx_path.c_str(), &st) == 0);

        if (is_indexed) {
            if (stat(idx_path.c_str(), &st) == 0) {
                total_size += st.st_size;
            }
            if (stat(dat_path.c_str(), &st) == 0) {
                total_size += st.st_size;
            }
        } else {
            for (uint64_t chromid = 0; chromid < chromkey.get_num_chroms(); ++chromid) {
                std::string chrom_file = GenomeTrack::find_existing_1d_filename(
                    chromkey, track_path, chromid);
                if (!chrom_file.empty()) {
                    std::string full_path = track_path + "/" + chrom_file;
                    if (stat(full_path.c_str(), &st) == 0) {
                        total_size += st.st_size;
                    }
                }
            }
        }

        PMPY py_size(PyLong_FromUnsignedLongLong(total_size), true);
        PyDict_SetItemString(result, "size_in_bytes", py_size);

        PMPY py_format(PyUnicode_FromString(is_indexed ? "indexed" : "per-chromosome"), true);
        PyDict_SetItemString(result, "format", py_format);

        // For dense tracks, get bin size (invariant across chromosomes).
        if (track_type == GenomeTrack::FIXED_BIN) {
            for (uint64_t chromid = 0; chromid < chromkey.get_num_chroms(); ++chromid) {
                std::string chrom_file = GenomeTrack::find_existing_1d_filename(
                    chromkey, track_path, chromid);
                if (chrom_file.empty())
                    continue;
                std::string full_path = track_path + "/" + chrom_file;
                // For per-chrom tracks, find_existing_1d_filename returns the
                // canonical base name even when no file exists for that chrom
                // (a legitimately empty leading scaffold, or a partial track).
                // Probing it with init_read would error "No such file or
                // directory". Skip to the first chrom that actually has a file.
                // Indexed tracks have no per-chrom files and read bin_size from
                // track.dat via the index inside init_read (which back-fills an
                // empty leading chrom), so probe the first chrom directly.
                if (!is_indexed && access(full_path.c_str(), F_OK) != 0)
                    continue;
                GenomeTrackFixedBin track;
                track.init_read(full_path.c_str(), chromid);
                int64_t bin_size = track.get_bin_size();
                PMPY py_binsize(PyLong_FromLongLong(bin_size), true);
                PyDict_SetItemString(result, "bin_size", py_binsize);
                break;
            }
        }

        // Load attributes from YAML if available
        std::string yaml_path = track_path + "/.attributes.yaml";
        struct stat yaml_stat;
        if (stat(yaml_path.c_str(), &yaml_stat) == 0) {
            // We'll add YAML parsing later or load via Python
            // For now, just store the path
            PMPY py_attrs_path(PyUnicode_FromString(yaml_path.c_str()), true);
            PyDict_SetItemString(result, "attributes_path", py_attrs_path);
        }

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Get track path on disk
PyObject *pm_track_path(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        const char *track_name = NULL;
        if (!PyArg_ParseTuple(args, "s", &track_name)) {
            verror("Invalid arguments to pm_track_path");
        }

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        // Check track exists
        if (!g_pmdb->track_exists(track_name)) {
            // Return None for non-existent tracks
            Py_RETURN_NONE;
        }

        std::string track_path = g_pmdb->track_path(track_name);

        PMPY result(PyUnicode_FromString(track_path.c_str()), true);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Get all genome intervals. Delegates to PMDb's cache of the chrom name/size
// vectors (E.1.2 - R 5.6.30 ce788e75): on million-contig databases the
// per-call rebuild of these strings is the dominant cost. We still construct
// a fresh PMDataFrame each call (so callers cannot mutate the cached state).
PyObject *pm_intervals_all(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        if (!g_pmdb || !g_pmdb->is_initialized()) {
            verror("Database not initialized. Call gdb_init() first.");
        }

        return g_pmdb->get_intervals_all_py();

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Set random seed
PyObject *pm_seed(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(false);

        long seed = 0;
        if (!PyArg_ParseTuple(args, "l", &seed))
            verror("Invalid seed value");

        srand48(seed);
    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

// Test DataFrame conversion
PyObject *pm_test_df(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(false);

        // Create a simple test DataFrame
        PMDataFrame df(3, 2, "test");
        df.init_col(0, "name", PMDataFrame::STR);
        df.init_col(1, "value", PMDataFrame::DOUBLE);

        df.val_str(0, 0, "a");
        df.val_str(1, 0, "b");
        df.val_str(2, 0, "c");

        df.val_double(0, 1, 1.0);
        df.val_double(1, 1, 2.0);
        df.val_double(2, 1, 3.0);

        PMPY result = df.construct_py(false);
        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Read DataFrame from internal format
PyObject *pm_read_df(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(false);

        PyObject *py_df = NULL;
        const char *df_name = "df";

        if (!PyArg_ParseTuple(args, "O|s", &py_df, &df_name)) {
            verror("Invalid arguments to __read_df");
        }

        PMPY df_arg(py_df, false);
        PMDataFrame df(df_arg, df_name);

        vdebug("Read DataFrame '%s' with %zu rows and %zu columns\n",
               df_name, df.num_rows(), df.num_cols());

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

// Helper to convert Python intervals with strand support
static void convert_py_intervals_with_strand(PyObject *py_intervals,
                                              std::vector<GInterval> &intervals) {
    PMPY py_chrom;
    PMPY py_start;
    PMPY py_end;
    PMPY py_strand;

    // Check if it's the internal list format from _df2pymisha
    if (PyList_Check(py_intervals) && PyList_Size(py_intervals) >= 2) {
        PyObject *colnames = PyList_GetItem(py_intervals, 0);
        if (colnames && PyArray_Check(colnames)) {
            Py_ssize_t num_cols = PyArray_SIZE((PyArrayObject *)colnames);
            int chrom_idx = -1, start_idx = -1, end_idx = -1, strand_idx = -1;

            for (Py_ssize_t i = 0; i < num_cols; ++i) {
                PyObject *name = PyArray_GETITEM((PyArrayObject *)colnames,
                    (const char *)PyArray_GETPTR1((PyArrayObject *)colnames, i));
                if (name && PyUnicode_Check(name)) {
                    const char *name_str = PyUnicode_AsUTF8(name);
                    if (strcmp(name_str, "chrom") == 0) chrom_idx = i;
                    else if (strcmp(name_str, "start") == 0) start_idx = i;
                    else if (strcmp(name_str, "end") == 0) end_idx = i;
                    else if (strcmp(name_str, "strand") == 0) strand_idx = i;
                }
                Py_XDECREF(name);
            }

            if (chrom_idx >= 0 && start_idx >= 0 && end_idx >= 0) {
                py_chrom.assign(PyList_GetItem(py_intervals, chrom_idx + 1), false);
                py_start.assign(PyList_GetItem(py_intervals, start_idx + 1), false);
                py_end.assign(PyList_GetItem(py_intervals, end_idx + 1), false);
                if (strand_idx >= 0) {
                    py_strand.assign(PyList_GetItem(py_intervals, strand_idx + 1), false);
                }
            }
        }
    }

    // Fallback: try DataFrame-like attribute access
    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        py_chrom.assign(PyObject_GetAttrString(py_intervals, "chrom"), true);
        py_start.assign(PyObject_GetAttrString(py_intervals, "start"), true);
        py_end.assign(PyObject_GetAttrString(py_intervals, "end"), true);
        PyErr_Clear();  // strand may not exist
        py_strand.assign(PyObject_GetAttrString(py_intervals, "strand"), true);
    }

    if (!py_chrom || !py_start || !py_end) {
        PyErr_Clear();
        TGLError("intervals must have 'chrom', 'start', and 'end' columns");
    }

    normalize_interval_columns(py_chrom, py_start, py_end);

    Py_ssize_t len = PyObject_Length(py_chrom);
    if (len < 0) {
        PyErr_Clear();
        TGLError("Cannot determine length of intervals");
    }

    intervals.clear();
    intervals.reserve(len);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();

    for (Py_ssize_t i = 0; i < len; ++i) {
        PMPY chrom_val(PySequence_GetItem(py_chrom, i), true);
        PMPY start_val(PySequence_GetItem(py_start, i), true);
        PMPY end_val(PySequence_GetItem(py_end, i), true);

        if (!chrom_val || !start_val || !end_val) {
            PyErr_Clear();
            TGLError("Failed to get interval values at index %ld", (long)i);
        }

        // Get chromosome id
        int chromid = -1;
        if (PyUnicode_Check(chrom_val)) {
            const char *chrom_name = PyUnicode_AsUTF8(chrom_val);
            chromid = chromkey.chrom2id(chrom_name);
            if (chromid < 0) {
                TGLError("Unknown chromosome: %s", chrom_name);
            }
        } else if (PyNumber_Check(chrom_val)) {
            PMPY py_long(PyNumber_Long(chrom_val), true);
            if (!py_long) {
                PyErr_Clear();
                TGLError("Failed to convert chromosome to integer at index %ld", (long)i);
            }
            long chrom_num = PyLong_AsLong(py_long);
            std::string chrom_str = std::to_string(chrom_num);
            chromid = chromkey.chrom2id(chrom_str.c_str());
            if (chromid < 0) {
                chrom_str = "chr" + std::to_string(chrom_num);
                chromid = chromkey.chrom2id(chrom_str.c_str());
            }
            if (chromid < 0) {
                TGLError("Unknown chromosome: %ld", chrom_num);
            }
        } else {
            TGLError("Invalid chromosome type at index %ld", (long)i);
        }

        int64_t start = PyLong_AsLongLong(start_val);
        int64_t end = PyLong_AsLongLong(end_val);

        if (PyErr_Occurred()) {
            PyErr_Clear();
            TGLError("Invalid start/end values at index %ld", (long)i);
        }

        // Get strand if available
        char strand = 0;  // 0 = no strand, 1 = forward, -1 = reverse
        if (py_strand) {
            PMPY strand_val(PySequence_GetItem(py_strand, i), true);
            if (strand_val && !PyErr_Occurred()) {
                if (PyNumber_Check(strand_val)) {
                    PMPY py_long(PyNumber_Long(strand_val), true);
                    if (py_long) {
                        long s = PyLong_AsLong(py_long);
                        if (s == -1 || s == 1) strand = (char)s;
                    }
                } else if (PyUnicode_Check(strand_val)) {
                    const char *s = PyUnicode_AsUTF8(strand_val);
                    if (s && s[0] == '+') strand = 1;
                    else if (s && s[0] == '-') strand = -1;
                }
            }
            PyErr_Clear();
        }

        if (start < 0 || start >= end ||
            (uint64_t)end > chromkey.get_chrom_size(chromid))
            verify_interval_coords(chromkey, chromid, start, end);

        GInterval interval(chromid, start, end);
        interval.strand = strand;
        intervals.push_back(interval);
    }
}

// Extract DNA sequences for intervals
PyObject *pm_seq_extract(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_intervals = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "O|O", &py_intervals, &py_config)) {
            verror("Invalid arguments to pm_seq_extract");
        }

        if (!py_intervals || py_intervals == Py_None) {
            verror("intervals argument is required for pm_seq_extract");
        }

        std::vector<GInterval> intervals;
        convert_py_intervals_with_strand(py_intervals, intervals);

        if (intervals.empty()) {
            // Return empty list
            return PyList_New(0);
        }

        // Set up sequence fetcher
        GenomeSeqFetch seqfetch;
        seqfetch.set_seqdir(g_pmdb->groot() + "/seq");

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Create result list
        PMPY result(PyList_New(intervals.size()), true);
        if (!result) {
            verror("Failed to create result list");
        }

        std::vector<char> seq_buf;

        for (size_t i = 0; i < intervals.size(); ++i) {
            seqfetch.read_interval(intervals[i], chromkey, seq_buf);

            // PyUnicode_FromStringAndSize copies the bytes once; the
            // intermediate std::string was a redundant copy of seq_buf.
            PyObject *py_seq = PyUnicode_FromStringAndSize(seq_buf.data(),
                                                          seq_buf.size());
            if (!py_seq) {
                verror("Failed to create sequence string");
            }

            // PyList_SET_ITEM steals reference, so don't decref
            PyList_SET_ITEM((PyObject *)result, i, py_seq);
        }

        result.to_be_stolen();
        return (PyObject *)result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Structure to hold partition result for a single interval
struct PartitionInterval {
    int chromid;
    int64_t start;
    int64_t end;
    int bin;  // 1-based bin index

    PartitionInterval(int c, int64_t s, int64_t e, int b)
        : chromid(c), start(s), end(e), bin(b) {}
};

// Helper to compute partition for a subset of intervals (used by workers)
static std::vector<PartitionInterval> compute_partition(
    const std::string &expr,
    const std::vector<GInterval> &intervals,
    const BinFinder &bin_finder,
    long iterator_policy,
    PyObject *progress_cb)
{
    std::vector<PartitionInterval> result;

    if (intervals.empty())
        return result;

    PMTrackExprScanner scanner;
    std::vector<std::string> exprs = {expr};
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    int last_bin = -1;
    GInterval merged_interval(-1, -1, -1);

    for (; !scanner.isend(); scanner.next()) {
        const GInterval &cur_interval = scanner.last_interval();
        double val = scanner.vdouble(0);
        int cur_bin = std::isnan(val) ? -1 : bin_finder.val2bin(val);

        // Check if we need to emit the merged interval
        // Emit when: bin changes, non-adjacent intervals, or different chromosomes
        if (last_bin >= 0 &&
            (cur_bin != last_bin ||
             cur_interval.start != merged_interval.end ||
             cur_interval.chromid != merged_interval.chromid)) {
            // Emit the previous merged interval with 1-based bin index
            result.emplace_back(merged_interval.chromid, merged_interval.start,
                                merged_interval.end, last_bin + 1);
        }

        // Update merged interval
        if (cur_bin < 0) {
            // NaN or out-of-range - reset merged interval
            merged_interval.start = -1;
        } else if (last_bin == cur_bin &&
                   cur_interval.start == merged_interval.end &&
                   cur_interval.chromid == merged_interval.chromid) {
            // Same bin and adjacent - extend the merged interval
            merged_interval.end = cur_interval.end;
        } else {
            // New interval (different bin or not adjacent)
            merged_interval = cur_interval;
        }

        last_bin = cur_bin;
        check_interrupt();
    }

    // Emit the last merged interval if valid
    if (merged_interval.start >= 0 && last_bin >= 0) {
        result.emplace_back(merged_interval.chromid, merged_interval.start,
                            merged_interval.end, last_bin + 1);
    }

    return result;
}

// Partition track expression values into bins and return corresponding intervals
PyObject *pm_partition(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_breaks = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        int include_lowest = 0;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOO|OpO", &py_expr, &py_breaks, &py_intervals,
                              &py_iterator, &include_lowest, &py_config)) {
            verror("Invalid arguments to pm_partition");
        }

        if (!PyUnicode_Check(py_expr)) {
            verror("gpartition expression must be a string");
        }
        std::string expr = PyUnicode_AsUTF8(py_expr);

        // Parse breaks
        if (!py_breaks || py_breaks == Py_None) {
            verror("gpartition requires breaks argument");
        }

        std::vector<double> breaks;
        if (PyList_Check(py_breaks)) {
            Py_ssize_t n = PyList_Size(py_breaks);
            breaks.reserve(n);
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject *item = PyList_GetItem(py_breaks, i);
                if (PyFloat_Check(item)) {
                    breaks.push_back(PyFloat_AsDouble(item));
                } else if (PyLong_Check(item)) {
                    breaks.push_back(PyLong_AsDouble(item));
                } else {
                    verror("gpartition breaks must be numeric values");
                }
            }
        } else if (PyArray_Check(py_breaks)) {
            PyArrayObject *arr = (PyArrayObject *)py_breaks;
            PMPY arr_float(PyArray_FROM_OTF(py_breaks, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY), true);
            if (!arr_float) {
                verror("gpartition breaks must be convertible to float array");
            }
            arr = (PyArrayObject *)(PyObject *)arr_float;
            Py_ssize_t n = PyArray_SIZE(arr);
            double *data = (double *)PyArray_DATA(arr);
            breaks.assign(data, data + n);
        } else {
            // Try to iterate
            PMPY iter(PyObject_GetIter(py_breaks), true);
            if (!iter) {
                PyErr_Clear();
                verror("gpartition breaks must be iterable");
            }
            PMPY item;
            while ((item.assign(PyIter_Next(iter), true))) {
                if (PyFloat_Check(item)) {
                    breaks.push_back(PyFloat_AsDouble(item));
                } else if (PyLong_Check(item)) {
                    breaks.push_back(PyLong_AsDouble(item));
                } else {
                    verror("gpartition breaks must be numeric values");
                }
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
                verror("Error iterating over breaks");
            }
        }

        if (breaks.size() < 2) {
            verror("gpartition requires at least 2 break values (for 1 bin)");
        }

        // Initialize BinFinder
        BinFinder bin_finder(breaks, include_lowest != 0, true);

        // Parse iterator policy
        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_partition");

        // Convert intervals
        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        if (intervals.empty()) {
            // Return None for empty intervals
            Py_INCREF(Py_None);
            return Py_None;
        }

        std::vector<PartitionInterval> result;

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_part = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split_part) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                    }
                    std::vector<PartitionInterval> kid_result =
                        compute_partition(expr, sub, bin_finder, iterator_policy, progress_cb);

                    // Write number of results first, then each result
                    size_t count = kid_result.size();
                    PyMisha::write_multitask_fifo(&count, sizeof(count));
                    for (const auto &pi : kid_result) {
                        PyMisha::write_multitask_fifo(&pi.chromid, sizeof(pi.chromid));
                        PyMisha::write_multitask_fifo(&pi.start, sizeof(pi.start));
                        PyMisha::write_multitask_fifo(&pi.end, sizeof(pi.end));
                        PyMisha::write_multitask_fifo(&pi.bin, sizeof(pi.bin));
                    }
                    _exit(0);
                }
            }

            // Read results from all children
            for (int kid = 0; kid < num_kids; ++kid) {
                size_t count;
                PyMisha::read_multitask_fifo(&count, sizeof(count));
                for (size_t i = 0; i < count; ++i) {
                    int chromid;
                    int64_t start, end;
                    int bin;
                    PyMisha::read_multitask_fifo(&chromid, sizeof(chromid));
                    PyMisha::read_multitask_fifo(&start, sizeof(start));
                    PyMisha::read_multitask_fifo(&end, sizeof(end));
                    PyMisha::read_multitask_fifo(&bin, sizeof(bin));
                    result.emplace_back(chromid, start, end, bin);
                }
            }

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            result = compute_partition(expr, intervals, bin_finder, iterator_policy, progress_cb);
        }

        if (result.empty()) {
            // Return None for no results
            Py_INCREF(Py_None);
            return Py_None;
        }

        // Build result DataFrame
        const GenomeChromKey &chromkey = g_pmdb->chromkey();
        size_t n = result.size();

        npy_intp dims[1] = {(npy_intp)n};

        // Create chrom array (object type for strings)
        PMPY py_chrom(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
        if (!py_chrom) verror("Failed to create chrom array");

        // Create numeric arrays
        PMPY py_start(PyArray_SimpleNew(1, dims, NPY_INT64), true);
        PMPY py_end(PyArray_SimpleNew(1, dims, NPY_INT64), true);
        PMPY py_bin(PyArray_SimpleNew(1, dims, NPY_INT64), true);
        if (!py_start || !py_end || !py_bin) verror("Failed to create result arrays");

        int64_t *start_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_start);
        int64_t *end_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_end);
        int64_t *bin_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_bin);
        PyObject **chrom_data = (PyObject **)PyArray_DATA((PyArrayObject *)(PyObject *)py_chrom);

        // Hot loop: reuse PyUnicode chrom string across same-chromid runs
        // (partition output is sorted by chrom). Matches intervals_to_py.
        int last_chromid = -1;
        PyObject *last_str = nullptr;
        for (size_t i = 0; i < n; ++i) {
            const PartitionInterval &pi = result[i];

            if (pi.chromid != last_chromid || !last_str) {
                last_str = PyUnicode_FromString(chromkey.id2chrom(pi.chromid).c_str());
                if (!last_str) verror("Failed to create chrom string");
                last_chromid = pi.chromid;
                chrom_data[i] = last_str;
            } else {
                Py_INCREF(last_str);
                chrom_data[i] = last_str;
            }

            start_data[i] = pi.start;
            end_data[i] = pi.end;
            bin_data[i] = pi.bin;
        }

        // Return as dict matching DataFrame format
        PMPY py_result(PyDict_New(), true);
        if (!py_result) verror("Failed to create result dict");

        if (PyDict_SetItemString(py_result, "chrom", py_chrom) < 0 ||
            PyDict_SetItemString(py_result, "start", py_start) < 0 ||
            PyDict_SetItemString(py_result, "end", py_end) < 0 ||
            PyDict_SetItemString(py_result, "bin", py_bin) < 0) {
            verror("Failed to set result dict items");
        }

        py_result.to_be_stolen();
        return (PyObject *)py_result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// ============================================================================
// Interval Set Operations - C++ Implementations
// ============================================================================

// Helper: convert C++ GInterval vector to Python dict (DataFrame-like)
static PyObject *intervals_to_py(const std::vector<GInterval> &intervals) {
    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    size_t n = intervals.size();

    // Create numpy arrays for each column
    npy_intp dims[1] = { (npy_intp)n };

    PMPY py_chrom(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
    PMPY py_start(PyArray_SimpleNew(1, dims, NPY_INT64), true);
    PMPY py_end(PyArray_SimpleNew(1, dims, NPY_INT64), true);

    if (!py_chrom || !py_start || !py_end) {
        verror("Failed to create result arrays");
    }

    int64_t *start_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_start);
    int64_t *end_data = (int64_t *)PyArray_DATA((PyArrayObject *)(PyObject *)py_end);
    PyObject **chrom_data = (PyObject **)PyArray_DATA((PyArrayObject *)(PyObject *)py_chrom);

    // Hot loop: copy start/end as raw int64 and reuse the PyUnicode object
    // across consecutive same-chromid rows. Sorted interval outputs (intersect,
    // union, diff, canonic, neighbors) have long runs of the same chromid,
    // so per-row PyUnicode_FromString collapses to a single allocation per
    // chrom plus Py_INCREF per row. Skips the PyArray_SETITEM overhead.
    int last_chromid = -1;
    PyObject *last_str = nullptr;
    for (size_t i = 0; i < n; ++i) {
        const GInterval &iv = intervals[i];
        if (iv.chromid != last_chromid || !last_str) {
            last_str = PyUnicode_FromString(chromkey.id2chrom(iv.chromid).c_str());
            if (!last_str) verror("Failed to allocate chrom string");
            last_chromid = iv.chromid;
            // The array steals the initial reference; subsequent same-id
            // rows take another reference.
            chrom_data[i] = last_str;
        } else {
            Py_INCREF(last_str);
            chrom_data[i] = last_str;
        }
        start_data[i] = iv.start;
        end_data[i] = iv.end;
    }

    // Return as dict
    PMPY py_result(PyDict_New(), true);
    if (!py_result) verror("Failed to create result dict");

    if (PyDict_SetItemString(py_result, "chrom", py_chrom) < 0 ||
        PyDict_SetItemString(py_result, "start", py_start) < 0 ||
        PyDict_SetItemString(py_result, "end", py_end) < 0) {
        verror("Failed to set interval result dict items");
    }

    py_result.to_be_stolen();
    return (PyObject *)py_result;
}

// Comparison function for sorting intervals by chrom and start
static bool intervals_compare_by_start(const GInterval &a, const GInterval &b) {
    return a.chromid < b.chromid || (a.chromid == b.chromid && a.start < b.start);
}

// Helper: sort intervals in place
static void intervals_sort(std::vector<GInterval> &intervals) {
    if (!intervals.empty() && !std::is_sorted(intervals.begin(), intervals.end(), intervals_compare_by_start)) {
        std::sort(intervals.begin(), intervals.end(), intervals_compare_by_start);
    }
}

// Helper: merge overlapping intervals in place (intervals must be sorted)
static void intervals_unify_overlaps(std::vector<GInterval> &intervals, bool merge_touching = true) {
    if (intervals.empty()) return;

    size_t cur_idx = 0;
    for (size_t i = 1; i < intervals.size(); i++) {
        GInterval &cur = intervals[cur_idx];
        const GInterval &next = intervals[i];

        // Check if we should merge
        bool different_chrom = cur.chromid != next.chromid;
        bool no_overlap = cur.end < next.start;
        bool touching_no_merge = !merge_touching && cur.end == next.start;

        if (different_chrom || no_overlap || touching_no_merge) {
            // Start a new interval
            intervals[++cur_idx] = next;
        } else if (cur.end < next.end) {
            // Extend the current interval
            cur.end = next.end;
        }
    }
    intervals.resize(cur_idx + 1);
}

// Union of two interval sets - O(n) merge algorithm
PyObject *pm_intervals_union(PyObject *self, PyObject *args) {
    try {
        PyMisha pymisha(true);

        PyObject *py_intervs1 = nullptr;
        PyObject *py_intervs2 = nullptr;

        if (!PyArg_ParseTuple(args, "OO", &py_intervs1, &py_intervs2)) {
            verror("Invalid arguments to pm_intervals_union");
        }

        std::vector<GInterval> intervs1, intervs2;
        convert_py_intervals(py_intervs1, intervs1);
        convert_py_intervals(py_intervs2, intervs2);

        // Both must be sorted
        intervals_sort(intervs1);
        intervals_sort(intervs2);

        // Merge-sort the two sets
        std::vector<GInterval> result;
        result.reserve(intervs1.size() + intervs2.size());

        size_t i1 = 0, i2 = 0;
        int last_chromid1 = -1, last_chromid2 = -1;
        int idx = 0;

        while (i1 < intervs1.size() && i2 < intervs2.size()) {
            const GInterval &a = intervs1[i1];
            const GInterval &b = intervs2[i2];

            if (a.chromid == b.chromid) {
                idx = a.start < b.start ? 0 : 1;
            } else if (last_chromid1 != a.chromid || last_chromid2 != b.chromid) {
                idx = intervals_compare_by_start(a, b) ? 0 : 1;
                last_chromid1 = a.chromid;
                last_chromid2 = b.chromid;
            }

            if (idx == 0) {
                result.push_back(a);
                ++i1;
            } else {
                result.push_back(b);
                ++i2;
            }
        }

        // Append remaining
        while (i1 < intervs1.size()) result.push_back(intervs1[i1++]);
        while (i2 < intervs2.size()) result.push_back(intervs2[i2++]);

        // Unify overlapping intervals
        intervals_unify_overlaps(result);

        return intervals_to_py(result);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Intersection of two interval sets - O(n) algorithm
PyObject *pm_intervals_intersect(PyObject *self, PyObject *args) {
    try {
        PyMisha pymisha(true);

        PyObject *py_intervs1 = nullptr;
        PyObject *py_intervs2 = nullptr;

        if (!PyArg_ParseTuple(args, "OO", &py_intervs1, &py_intervs2)) {
            verror("Invalid arguments to pm_intervals_intersect");
        }

        std::vector<GInterval> intervs1, intervs2;
        convert_py_intervals(py_intervs1, intervs1);
        convert_py_intervals(py_intervs2, intervs2);

        // Both must be sorted
        intervals_sort(intervs1);
        intervals_sort(intervs2);

        std::vector<GInterval> result;

        size_t i1 = 0, i2 = 0;
        // Virtual start positions - allows "consuming" part of an interval
        int64_t virt_start1 = 0, virt_start2 = 0;
        bool use_virt1 = false, use_virt2 = false;
        int last_chromid1 = -1, last_chromid2 = -1;
        int idx = 0;

        while (i1 < intervs1.size() && i2 < intervs2.size()) {
            const GInterval &a = intervs1[i1];
            const GInterval &b = intervs2[i2];
            int64_t eff_start1 = use_virt1 ? virt_start1 : a.start;
            int64_t eff_start2 = use_virt2 ? virt_start2 : b.start;

            if (a.chromid == b.chromid) {
                // Check for no overlap
                if (eff_start1 < eff_start2 && a.end <= eff_start2) {
                    ++i1;
                    use_virt1 = false;
                } else if (eff_start2 < eff_start1 && b.end <= eff_start1) {
                    ++i2;
                    use_virt2 = false;
                } else {
                    // Intervals intersect
                    int64_t start = std::max(eff_start1, eff_start2);
                    int64_t end = std::min(a.end, b.end);

                    result.emplace_back(a.chromid, start, end);

                    // Advance iterators
                    if (a.end == end) {
                        ++i1;
                        use_virt1 = false;
                    } else {
                        virt_start1 = end;
                        use_virt1 = true;
                    }

                    if (b.end == end) {
                        ++i2;
                        use_virt2 = false;
                    } else {
                        virt_start2 = end;
                        use_virt2 = true;
                    }
                }
            } else {
                // Different chromosomes - advance the smaller one
                if (last_chromid1 != a.chromid || last_chromid2 != b.chromid) {
                    idx = intervals_compare_by_start(a, b) ? 0 : 1;
                    last_chromid1 = a.chromid;
                    last_chromid2 = b.chromid;
                }
                if (idx == 0) {
                    ++i1;
                    use_virt1 = false;
                } else {
                    ++i2;
                    use_virt2 = false;
                }
            }
        }

        return intervals_to_py(result);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Difference of two interval sets (set1 - set2) - O(n) algorithm
PyObject *pm_intervals_diff(PyObject *self, PyObject *args) {
    try {
        PyMisha pymisha(true);

        PyObject *py_intervs1 = nullptr;
        PyObject *py_intervs2 = nullptr;

        if (!PyArg_ParseTuple(args, "OO", &py_intervs1, &py_intervs2)) {
            verror("Invalid arguments to pm_intervals_diff");
        }

        std::vector<GInterval> intervs1, intervs2;
        convert_py_intervals(py_intervs1, intervs1);
        convert_py_intervals(py_intervs2, intervs2);

        // Both must be sorted
        intervals_sort(intervs1);
        intervals_sort(intervs2);

        std::vector<GInterval> result;

        size_t i1 = 0, i2 = 0;
        int64_t virt_start1 = 0, virt_start2 = 0;
        bool use_virt1 = false, use_virt2 = false;
        int last_chromid1 = -1, last_chromid2 = -1;
        int idx = 0;

        while (i1 < intervs1.size() && i2 < intervs2.size()) {
            const GInterval &a = intervs1[i1];
            const GInterval &b = intervs2[i2];
            int64_t eff_start1 = use_virt1 ? virt_start1 : a.start;
            int64_t eff_start2 = use_virt2 ? virt_start2 : b.start;

            if (a.chromid == b.chromid) {
                // No overlap cases
                if (eff_start1 < eff_start2 && a.end <= eff_start2) {
                    // a is completely before b - add all of a
                    result.emplace_back(a.chromid, eff_start1, a.end);
                    ++i1;
                    use_virt1 = false;
                } else if (eff_start2 < eff_start1 && b.end <= eff_start1) {
                    // b is completely before a - skip b
                    ++i2;
                    use_virt2 = false;
                } else {
                    // Intervals intersect
                    int64_t intersect_start = std::max(eff_start1, eff_start2);
                    int64_t intersect_end = std::min(a.end, b.end);

                    // Add the part of a before the intersection
                    if (eff_start1 < intersect_start) {
                        result.emplace_back(a.chromid, eff_start1, intersect_start);
                    }

                    // Advance iterators
                    if (a.end == intersect_end) {
                        ++i1;
                        use_virt1 = false;
                    } else {
                        virt_start1 = intersect_end;
                        use_virt1 = true;
                    }

                    if (b.end == intersect_end) {
                        ++i2;
                        use_virt2 = false;
                    } else {
                        virt_start2 = intersect_end;
                        use_virt2 = true;
                    }
                }
            } else {
                // Different chromosomes
                if (last_chromid1 != a.chromid || last_chromid2 != b.chromid) {
                    idx = intervals_compare_by_start(a, b) ? 0 : 1;
                    last_chromid1 = a.chromid;
                    last_chromid2 = b.chromid;
                }
                if (idx == 0) {
                    // a's chromosome comes first - add all of a
                    int64_t eff_start = use_virt1 ? virt_start1 : a.start;
                    result.emplace_back(a.chromid, eff_start, a.end);
                    ++i1;
                    use_virt1 = false;
                } else {
                    ++i2;
                    use_virt2 = false;
                }
            }
        }

        // Append remaining intervals from set1
        if (i1 < intervs1.size()) {
            if (use_virt1) {
                result.emplace_back(intervs1[i1].chromid, virt_start1, intervs1[i1].end);
                ++i1;
            }
            while (i1 < intervs1.size()) {
                result.push_back(intervs1[i1++]);
            }
        }

        return intervals_to_py(result);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Helper: merge overlapping intervals in place and produce mapping
// intervals must be sorted and udata must contain original index
static void intervals_unify_overlaps_with_mapping(std::vector<GInterval> &intervals, std::vector<int64_t> &mapping, bool merge_touching = true) {
    if (intervals.empty()) return;

    size_t cur_idx = 0;
    
    // Process first interval
    int64_t orig_idx = (int64_t)(intptr_t)intervals[0].udata;
    if (orig_idx >= 0 && orig_idx < (int64_t)mapping.size()) mapping[orig_idx] = cur_idx;

    for (size_t i = 1; i < intervals.size(); i++) {
        GInterval &cur = intervals[cur_idx];
        const GInterval &next = intervals[i];
        int64_t next_orig_idx = (int64_t)(intptr_t)next.udata;

        // Check if we should merge
        bool different_chrom = cur.chromid != next.chromid;
        bool no_overlap = cur.end < next.start;
        bool touching_no_merge = !merge_touching && cur.end == next.start;

        if (different_chrom || no_overlap || touching_no_merge) {
            // Start a new interval
            intervals[++cur_idx] = next;
        } else if (cur.end < next.end) {
            // Extend the current interval
            cur.end = next.end;
        }
        
        if (next_orig_idx >= 0 && next_orig_idx < (int64_t)mapping.size()) {
            mapping[next_orig_idx] = cur_idx;
        }
    }
    intervals.resize(cur_idx + 1);
}

// Canonicalize intervals: sort and merge overlapping/touching
PyObject *pm_intervals_canonic(PyObject *self, PyObject *args) {
    try {
        PyMisha pymisha(true);

        PyObject *py_intervs = nullptr;
        int merge_touching = 1;  // Default: merge touching intervals

        if (!PyArg_ParseTuple(args, "O|p", &py_intervs, &merge_touching)) {
            verror("Invalid arguments to pm_intervals_canonic");
        }

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervs, intervals);

        // Store original indices
        size_t n = intervals.size();
        for (size_t i = 0; i < n; ++i) {
            intervals[i].udata = (void*)(intptr_t)i;
        }

        // Sort
        intervals_sort(intervals);

        // Merge overlaps and compute mapping
        std::vector<int64_t> mapping(n, -1);
        intervals_unify_overlaps_with_mapping(intervals, mapping, merge_touching != 0);

        // Convert intervals to Python dict
        PyObject *py_df = intervals_to_py(intervals);
        if (!py_df) return NULL;

        // Convert mapping to numpy array
        npy_intp dims[1] = { (npy_intp)n };
        PyObject *py_mapping = PyArray_SimpleNew(1, dims, NPY_INT64);
        if (!py_mapping) {
            Py_DECREF(py_df);
            verror("Failed to create mapping array");
        }

        int64_t *map_data = (int64_t *)PyArray_DATA((PyArrayObject *)py_mapping);
        std::copy(mapping.begin(), mapping.end(), map_data);

        // Return tuple (df, mapping)
        PyObject *ret = PyTuple_Pack(2, py_df, py_mapping);
        Py_DECREF(py_df);
        Py_DECREF(py_mapping);

        return ret;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// Compute total covered basepairs (intervals must be non-overlapping after canonic)
PyObject *pm_intervals_covered_bp(PyObject *self, PyObject *args) {
    try {
        PyMisha pymisha(true);

        PyObject *py_intervs = nullptr;

        if (!PyArg_ParseTuple(args, "O", &py_intervs)) {
            verror("Invalid arguments to pm_intervals_covered_bp");
        }

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervs, intervals);

        // Sort and merge to get non-overlapping intervals
        intervals_sort(intervals);
        intervals_unify_overlaps(intervals);

        // Sum up ranges
        int64_t total_bp = 0;
        for (const auto &iv : intervals) {
            total_bp += iv.end - iv.start;
        }

        return PyLong_FromLongLong(total_bp);

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}


// Helper to compute distribution for a subset of intervals (used by workers)
static std::vector<uint64_t> compute_dist(
    const std::vector<std::string> &exprs,
    const std::vector<GInterval> &intervals,
    const BinsManager &bins_manager,
    long iterator_policy,
    PyObject *progress_cb)
{
    uint64_t totalbins = bins_manager.get_total_bins();
    std::vector<uint64_t> distribution(totalbins, 0);

    if (intervals.empty() || totalbins == 0)
        return distribution;

    PMTrackExprScanner scanner;
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    unsigned num_exprs = exprs.size();
    std::vector<double> vals(num_exprs);

    for (; !scanner.isend(); scanner.next()) {
        // Collect values from all expressions
        for (unsigned i = 0; i < num_exprs; ++i) {
            vals[i] = scanner.vdouble(i);
        }

        // Convert to bin index
        int index = bins_manager.vals2idx(vals);

        if (index >= 0) {
            distribution[index]++;
        }
    }

    return distribution;
}


// Helper to parse a Python list of breaks into std::vector<double>
static std::vector<double> parse_breaks(PyObject *py_breaks, const char *context)
{
    std::vector<double> breaks;

    if (PyList_Check(py_breaks)) {
        Py_ssize_t n = PyList_Size(py_breaks);
        breaks.reserve(n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *item = PyList_GetItem(py_breaks, i);
            if (PyFloat_Check(item)) {
                breaks.push_back(PyFloat_AsDouble(item));
            } else if (PyLong_Check(item)) {
                breaks.push_back(PyLong_AsDouble(item));
            } else {
                verror("%s breaks must be numeric values", context);
            }
        }
    } else if (PyArray_Check(py_breaks)) {
        PMPY arr_float(PyArray_FROM_OTF(py_breaks, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY), true);
        if (!arr_float) {
            verror("%s breaks must be convertible to float array", context);
        }
        PyArrayObject *arr = (PyArrayObject *)(PyObject *)arr_float;
        Py_ssize_t n = PyArray_SIZE(arr);
        double *data = (double *)PyArray_DATA(arr);
        breaks.assign(data, data + n);
    } else {
        // Try to iterate
        PMPY iter(PyObject_GetIter(py_breaks), true);
        if (!iter) {
            PyErr_Clear();
            verror("%s breaks must be iterable", context);
        }
        PMPY item;
        while ((item.assign(PyIter_Next(iter), true))) {
            if (PyFloat_Check(item)) {
                breaks.push_back(PyFloat_AsDouble(item));
            } else if (PyLong_Check(item)) {
                breaks.push_back(PyLong_AsDouble(item));
            } else {
                verror("%s breaks must be numeric values", context);
            }
        }
        if (PyErr_Occurred()) {
            PyErr_Clear();
            verror("Error iterating over breaks");
        }
    }

    return breaks;
}


PyObject *pm_dist(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_exprs = NULL;          // List of expressions
        PyObject *py_breaks_list = NULL;    // List of break arrays
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        int include_lowest = 0;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOO|OpO", &py_exprs, &py_breaks_list, &py_intervals,
                              &py_iterator, &include_lowest, &py_config)) {
            verror("Invalid arguments to pm_dist");
        }

        // Parse expressions
        if (!PyList_Check(py_exprs)) {
            verror("gdist expressions must be a list");
        }
        Py_ssize_t num_exprs = PyList_Size(py_exprs);
        if (num_exprs < 1) {
            verror("gdist requires at least one expression");
        }

        std::vector<std::string> exprs;
        exprs.reserve(num_exprs);
        for (Py_ssize_t i = 0; i < num_exprs; ++i) {
            PyObject *py_expr = PyList_GetItem(py_exprs, i);
            if (!PyUnicode_Check(py_expr)) {
                verror("gdist expression %ld must be a string", i + 1);
            }
            exprs.push_back(PyUnicode_AsUTF8(py_expr));
        }

        // Parse breaks list
        if (!PyList_Check(py_breaks_list)) {
            verror("gdist breaks_list must be a list of break arrays");
        }
        Py_ssize_t num_breaks_sets = PyList_Size(py_breaks_list);
        if (num_breaks_sets != num_exprs) {
            verror("gdist requires same number of break sets as expressions");
        }

        std::vector<std::vector<double>> breaks_list;
        breaks_list.reserve(num_breaks_sets);
        for (Py_ssize_t i = 0; i < num_breaks_sets; ++i) {
            PyObject *py_breaks = PyList_GetItem(py_breaks_list, i);
            std::vector<double> breaks = parse_breaks(py_breaks, "gdist");
            if (breaks.size() < 2) {
                verror("gdist breaks[%ld] must have at least 2 elements", i);
            }
            breaks_list.push_back(std::move(breaks));
        }

        // Initialize BinsManager
        BinsManager bins_manager;
        bins_manager.init(breaks_list, include_lowest != 0);

        uint64_t totalbins = bins_manager.get_total_bins();
        g_pymisha->verify_max_data_size(totalbins, "Distribution result");

        // Parse iterator policy
        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_dist");

        // Convert intervals
        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        // Sort and unify overlaps (like R misha does)
        intervals_sort(intervals);
        intervals_unify_overlaps(intervals);

        if (intervals.empty()) {
            // Return zero-filled array for empty intervals
            // Create N-dimensional array with shape (n_bins_0, n_bins_1, ...)
            std::vector<npy_intp> dims;
            for (size_t i = 0; i < exprs.size(); ++i) {
                dims.push_back(bins_manager.get_bin_finder(i).get_numbins());
            }

            PMPY py_result(PyArray_ZEROS(dims.size(), dims.data(), NPY_DOUBLE, 0), true);
            if (!py_result) verror("Failed to create result array");

            py_result.to_be_stolen();
            return py_result;
        }

        std::vector<uint64_t> distribution;

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_dist = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split_dist) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                    }
                    std::vector<uint64_t> kid_dist =
                        compute_dist(exprs, sub, bins_manager, iterator_policy, progress_cb);

                    // Write distribution array
                    PyMisha::write_multitask_fifo(kid_dist.data(), totalbins * sizeof(uint64_t));
                    _exit(0);
                }
            }

            // Collect results from all children
            distribution.resize(totalbins, 0);
            std::vector<uint64_t> kid_dist(totalbins);

            for (int kid = 0; kid < num_kids; ++kid) {
                PyMisha::read_multitask_fifo(kid_dist.data(), totalbins * sizeof(uint64_t));
                for (uint64_t i = 0; i < totalbins; ++i) {
                    distribution[i] += kid_dist[i];
                }
            }

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            distribution = compute_dist(exprs, intervals, bins_manager, iterator_policy, progress_cb);
        }

        // Create N-dimensional result array
        std::vector<npy_intp> dims;
        for (size_t i = 0; i < exprs.size(); ++i) {
            dims.push_back(bins_manager.get_bin_finder(i).get_numbins());
        }

        PMPY py_result(PyArray_SimpleNew(dims.size(), dims.data(), NPY_DOUBLE), true);
        if (!py_result) verror("Failed to create result array");

        double *result_data = (double *)PyArray_DATA((PyArrayObject *)(PyObject *)py_result);
        for (uint64_t i = 0; i < totalbins; ++i) {
            result_data[i] = (double)distribution[i];
        }

        py_result.to_be_stolen();
        return py_result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// ---------------------------------------------------------------------------
//  glookup – streaming bin-transform lookup
// ---------------------------------------------------------------------------

struct LookupResult {
    std::vector<GInterval> intervals;
    std::vector<double> values;
    std::vector<uint64_t> interval_ids;
};

static LookupResult compute_lookup(
    const std::vector<std::string> &exprs,
    const std::vector<GInterval> &intervals,
    const BinsManager &bins_manager,
    const std::vector<double> &lookup_table,
    bool force_binning,
    long iterator_policy,
    uint64_t interval_id_offset,
    PyObject *progress_cb)
{
    LookupResult result;
    if (intervals.empty())
        return result;

    PMTrackExprScanner scanner;
    if (progress_cb && !PyMisha::is_kid()) {
        scanner.set_progress_callback(progress_cb);
        scanner.report_progress(false);
    }
    scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

    unsigned num_exprs = exprs.size();
    std::vector<double> vals(num_exprs);

    uint64_t est_size = scanner.get_iterator()->size();
    if (est_size > 0) {
        result.intervals.reserve(est_size);
        result.values.reserve(est_size);
        result.interval_ids.reserve(est_size);
    }

    for (; !scanner.isend(); scanner.next()) {
        bool nan = false;
        unsigned index = 0;

        for (unsigned i = 0; i < num_exprs; ++i) {
            vals[i] = scanner.vdouble(i);
        }

        // Compute flat bin index, handling force_binning
        for (unsigned i = 0; i < num_exprs; ++i) {
            double val = vals[i];

            if (std::isnan(val)) {
                nan = true;
                break;
            }

            int bin = bins_manager.get_bin_finder(i).val2bin(val);

            if (bin < 0 && force_binning) {
                const auto &breaks = bins_manager.get_bin_finder(i).get_breaks();
                bin = (val <= breaks.front()) ? 0 : (int)bins_manager.get_bin_finder(i).get_numbins() - 1;
            }

            if (bin >= 0) {
                // Use track_mult logic from BinsManager internals
                // BinsManager stores multipliers: m_track_mult[i]
                // We replicate: index += bin * mult[i]
                // Since BinsManager::vals2idx does this, but we need force_binning,
                // we compute manually using the same multiplier scheme.
                uint64_t mult = 1;
                for (unsigned j = 0; j < i; ++j) {
                    mult *= bins_manager.get_bin_finder(j).get_numbins();
                }
                index += bin * mult;
            } else {
                nan = true;
                break;
            }
        }

        double value;
        if (nan) {
            value = std::numeric_limits<double>::quiet_NaN();
        } else {
            if (index >= lookup_table.size()) {
                value = std::numeric_limits<double>::quiet_NaN();
            } else {
                value = lookup_table[index];
            }
        }

        result.intervals.push_back(scanner.last_interval());
        result.values.push_back(value);
        result.interval_ids.push_back(scanner.last_interval_id() + interval_id_offset);

        g_pymisha->verify_max_data_size(result.intervals.size(), "Result");
        check_interrupt();
    }

    return result;
}

static void sort_lookup_result(LookupResult &result)
{
    const size_t n = result.intervals.size();
    if (n <= 1)
        return;

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (result.intervals[a].chromid != result.intervals[b].chromid)
            return result.intervals[a].chromid < result.intervals[b].chromid;
        if (result.intervals[a].start != result.intervals[b].start)
            return result.intervals[a].start < result.intervals[b].start;
        return result.intervals[a].end < result.intervals[b].end;
    });

    // Check if already sorted
    bool sorted = true;
    for (size_t i = 0; i < n; ++i) {
        if (order[i] != i) { sorted = false; break; }
    }
    if (sorted) return;

    // Apply permutation
    std::vector<GInterval> tmp_intervals(n);
    std::vector<double> tmp_values(n);
    std::vector<uint64_t> tmp_ids(n);
    for (size_t i = 0; i < n; ++i) {
        tmp_intervals[i] = result.intervals[order[i]];
        tmp_values[i] = result.values[order[i]];
        tmp_ids[i] = result.interval_ids[order[i]];
    }
    result.intervals.swap(tmp_intervals);
    result.values.swap(tmp_values);
    result.interval_ids.swap(tmp_ids);
}

static PMPY build_lookup_df(const LookupResult &result)
{
    size_t num_cols = 5;  // chrom, start, end, value, intervalID
    PMDataFrame df(result.intervals.size(), num_cols, "intervals");

    df.init_col(0, "chrom", PMDataFrame::STR);
    df.init_col(1, "start", PMDataFrame::LONG);
    df.init_col(2, "end", PMDataFrame::LONG);
    df.init_col(3, "value", PMDataFrame::DOUBLE);
    df.init_col(4, "intervalID", PMDataFrame::LONG);

    const GenomeChromKey &chromkey = g_pmdb->chromkey();
    for (size_t i = 0; i < result.intervals.size(); ++i) {
        df.val_str(i, 0, chromkey.id2chrom(result.intervals[i].chromid).c_str());
        df.val_long(i, 1, result.intervals[i].start);
        df.val_long(i, 2, result.intervals[i].end);
        df.val_double(i, 3, result.values[i]);
        df.val_long(i, 4, result.interval_ids[i]);
    }

    PMPY result_df = df.construct_py(true);
    result_df.to_be_stolen();
    return result_df;
}

// pm_lookup(exprs, breaks_list, lookup_table, intervals, iterator, include_lowest, force_binning, config)
PyObject *pm_lookup(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_exprs = NULL;
        PyObject *py_breaks_list = NULL;
        PyObject *py_lookup_table = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        int include_lowest = 0;
        int force_binning = 1;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOOO|OppO", &py_exprs, &py_breaks_list,
                              &py_lookup_table, &py_intervals,
                              &py_iterator, &include_lowest, &force_binning,
                              &py_config)) {
            verror("Invalid arguments to pm_lookup");
        }

        // Parse expressions
        if (!PyList_Check(py_exprs)) {
            verror("glookup expressions must be a list");
        }
        Py_ssize_t num_exprs = PyList_Size(py_exprs);
        if (num_exprs < 1) {
            verror("glookup requires at least one expression");
        }

        std::vector<std::string> exprs;
        exprs.reserve(num_exprs);
        for (Py_ssize_t i = 0; i < num_exprs; ++i) {
            PyObject *py_expr = PyList_GetItem(py_exprs, i);
            if (!PyUnicode_Check(py_expr)) {
                verror("glookup expression %ld must be a string", i + 1);
            }
            exprs.push_back(PyUnicode_AsUTF8(py_expr));
        }

        // Parse breaks list
        if (!PyList_Check(py_breaks_list)) {
            verror("glookup breaks_list must be a list of break arrays");
        }
        Py_ssize_t num_breaks_sets = PyList_Size(py_breaks_list);
        if (num_breaks_sets != num_exprs) {
            verror("glookup requires same number of break sets as expressions");
        }

        std::vector<std::vector<double>> breaks_list;
        breaks_list.reserve(num_breaks_sets);
        for (Py_ssize_t i = 0; i < num_breaks_sets; ++i) {
            PyObject *py_breaks = PyList_GetItem(py_breaks_list, i);
            std::vector<double> breaks = parse_breaks(py_breaks, "glookup");
            if (breaks.size() < 2) {
                verror("glookup breaks[%ld] must have at least 2 elements", i);
            }
            breaks_list.push_back(std::move(breaks));
        }

        // Initialize BinsManager
        BinsManager bins_manager;
        bins_manager.init(breaks_list, include_lowest != 0);

        uint64_t totalbins = bins_manager.get_total_bins();

        // Parse lookup table from numpy array
        PMPY arr_float(PyArray_FROM_OTF(py_lookup_table, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY), true);
        if (!arr_float) {
            verror("glookup lookup_table must be convertible to float array");
        }
        PyArrayObject *arr = (PyArrayObject *)(PyObject *)arr_float;
        Py_ssize_t table_size = PyArray_SIZE(arr);
        double *table_data = (double *)PyArray_DATA(arr);

        if ((uint64_t)table_size != totalbins) {
            verror("glookup lookup_table size (%ld) must match total bins (%lu)",
                   table_size, (unsigned long)totalbins);
        }

        std::vector<double> lookup_table(table_data, table_data + table_size);

        // Parse iterator policy
        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_lookup");

        // Convert intervals
        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        intervals_sort(intervals);
        intervals_unify_overlaps(intervals);

        if (intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        LookupResult result;

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_lk = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (use_range_split_lk) {
            int64_t total_range = 0;
            for (const auto &iv : intervals) total_range += iv.end - iv.start;
            num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
            if (num_kids > 0) {
                split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                num_kids = (int)range_kids.size();
            }
        } else {
            num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
        }

        PyObject *progress_cb = get_progress_cb(py_config);
        if (num_kids > 0) {
            progress_cb = NULL;
        }

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            struct LookupHeader {
                uint64_t nrows;
            };

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    uint64_t lk_id_offset = 0;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                        lk_id_offset = start;
                    }
                    LookupResult kid_result =
                        compute_lookup(exprs, sub, bins_manager, lookup_table,
                                       force_binning != 0, iterator_policy, lk_id_offset, progress_cb);

                    LookupHeader header{kid_result.intervals.size()};

                    // Row: chromid(int32) + start(int64) + end(int64) + interval_id(uint64) + value(double)
                    std::vector<char> rowbuf(sizeof(int32_t) + sizeof(int64_t) * 2 +
                                             sizeof(uint64_t) + sizeof(double));

                    PyMisha::lock_multitask_fifo();
                    try {
                        PyMisha::write_multitask_fifo_unlocked(&header, sizeof(header));
                        for (size_t i = 0; i < kid_result.intervals.size(); ++i) {
                            size_t offset = 0;
                            int32_t chromid = kid_result.intervals[i].chromid;
                            memcpy(rowbuf.data() + offset, &chromid, sizeof(chromid));
                            offset += sizeof(chromid);
                            memcpy(rowbuf.data() + offset, &kid_result.intervals[i].start, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            memcpy(rowbuf.data() + offset, &kid_result.intervals[i].end, sizeof(int64_t));
                            offset += sizeof(int64_t);
                            memcpy(rowbuf.data() + offset, &kid_result.interval_ids[i], sizeof(uint64_t));
                            offset += sizeof(uint64_t);
                            double value = kid_result.values[i];
                            memcpy(rowbuf.data() + offset, &value, sizeof(double));

                            PyMisha::write_multitask_fifo_unlocked(rowbuf.data(), rowbuf.size());
                        }
                    } catch (...) {
                        PyMisha::unlock_multitask_fifo();
                        throw;
                    }
                    PyMisha::unlock_multitask_fifo();
                    _exit(0);
                }
            }

            // Collect results from children
            for (int kid = 0; kid < num_kids; ++kid) {
                LookupHeader header{0};
                PyMisha::read_multitask_fifo(&header, sizeof(header));

                result.intervals.reserve(result.intervals.size() + header.nrows);
                result.values.reserve(result.values.size() + header.nrows);
                result.interval_ids.reserve(result.interval_ids.size() + header.nrows);

                std::vector<char> rowbuf(sizeof(int32_t) + sizeof(int64_t) * 2 +
                                         sizeof(uint64_t) + sizeof(double));

                for (uint64_t i = 0; i < header.nrows; ++i) {
                    PyMisha::read_multitask_fifo(rowbuf.data(), rowbuf.size());

                    size_t offset = 0;
                    int32_t chromid = -1;
                    int64_t start = 0;
                    int64_t end = 0;
                    uint64_t interval_id = 0;
                    double value = 0.0;

                    memcpy(&chromid, rowbuf.data() + offset, sizeof(chromid));
                    offset += sizeof(chromid);
                    memcpy(&start, rowbuf.data() + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&end, rowbuf.data() + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    memcpy(&interval_id, rowbuf.data() + offset, sizeof(uint64_t));
                    offset += sizeof(uint64_t);
                    memcpy(&value, rowbuf.data() + offset, sizeof(double));

                    result.intervals.emplace_back(chromid, start, end);
                    result.values.push_back(value);
                    result.interval_ids.push_back(interval_id);
                }
            }

            sort_lookup_result(result);

            while (PyMisha::wait_for_kids(1))
                ;
        } else {
            result = compute_lookup(exprs, intervals, bins_manager, lookup_table,
                                    force_binning != 0, iterator_policy, 0, progress_cb);
        }

        if (result.intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        g_pymisha->verify_max_data_size(result.intervals.size(), "Result");

        PMPY result_df = build_lookup_df(result);
        result_df.to_be_stolen();
        return (PyObject *)result_df;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// ---------------------------------------------------------------------------
//  gsample – streaming reservoir sampling
// ---------------------------------------------------------------------------

PyObject *pm_sample(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_expr = NULL;
        PyObject *py_n = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OOO|OO", &py_expr, &py_n,
                              &py_intervals, &py_iterator, &py_config)) {
            verror("Invalid arguments to pm_sample");
        }

        if (!PyUnicode_Check(py_expr))
            verror("gsample expression must be a string");

        std::string expr = PyUnicode_AsUTF8(py_expr);

        long n = PyLong_AsLong(py_n);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            verror("Number of samples must be an integer");
        }
        if (n < 1)
            verror("Number of samples must be greater than zero");

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_sample");

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        if (intervals.empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        StreamSampler<double> sampler((uint64_t)n, true);

        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        for (; !scanner.isend(); scanner.next()) {
            double val = scanner.vdouble(0);
            if (!std::isnan(val)) {
                sampler.add(val, drand48);
            }
            check_interrupt();
        }

        if (sampler.samples().empty()) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        // Shuffle the samples (reservoir sampler doesn't guarantee random order)
        tgs_random_shuffle(sampler.samples().begin(), sampler.samples().end(), drand48);

        // Build NumPy result array
        npy_intp dims[1] = {(npy_intp)sampler.samples().size()};
        PMPY py_result(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        if (!py_result)
            verror("Failed to allocate result array");

        double *result_data = (double *)PyArray_DATA((PyArrayObject *)(PyObject *)py_result);
        for (size_t i = 0; i < sampler.samples().size(); ++i) {
            result_data[i] = sampler.samples()[i];
        }

        py_result.to_be_stolen();
        return py_result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// ---------------------------------------------------------------------------
//  gcor – correlation between expression pairs
// ---------------------------------------------------------------------------

enum class CorMethod {
    PEARSON,
    SPEARMAN,
    SPEARMAN_EXACT,
};

static CorMethod parse_cor_method(PyObject *py_method)
{
    if (!py_method || py_method == Py_None)
        return CorMethod::PEARSON;

    if (!PyUnicode_Check(py_method))
        verror("gcor method must be a string");

    std::string method = PyUnicode_AsUTF8(py_method);
    if (method == "pearson")
        return CorMethod::PEARSON;
    if (method == "spearman")
        return CorMethod::SPEARMAN;
    if (method == "spearman.exact")
        return CorMethod::SPEARMAN_EXACT;

    verror("Unsupported gcor method: %s", method.c_str());
    return CorMethod::PEARSON;
}

struct CorSummary {
    double n{0};
    double n_na{0};
    double sum1{0};
    double sum2{0};
    double sumsq1{0};
    double sumsq2{0};
    double cross{0};

    void update(double v1, double v2) {
        ++n;
        if (std::isnan(v1) || std::isnan(v2)) {
            ++n_na;
        } else {
            sum1 += v1;
            sum2 += v2;
            sumsq1 += v1 * v1;
            sumsq2 += v2 * v2;
            cross += v1 * v2;
        }
    }

    double n_valid() const { return n - n_na; }

    double mean1() const { return n_valid() > 0 ? sum1 / n_valid() : std::numeric_limits<double>::quiet_NaN(); }
    double mean2() const { return n_valid() > 0 ? sum2 / n_valid() : std::numeric_limits<double>::quiet_NaN(); }

    double var1() const {
        if (n_valid() < 2) return std::numeric_limits<double>::quiet_NaN();
        double m = mean1();
        return (sumsq1 - n_valid() * m * m) / (n_valid() - 1);
    }

    double var2() const {
        if (n_valid() < 2) return std::numeric_limits<double>::quiet_NaN();
        double m = mean2();
        return (sumsq2 - n_valid() * m * m) / (n_valid() - 1);
    }

    double sd1() const { return std::sqrt(var1()); }
    double sd2() const { return std::sqrt(var2()); }

    double cov() const {
        if (n_valid() < 2) return std::numeric_limits<double>::quiet_NaN();
        return (cross - n_valid() * mean1() * mean2()) / (n_valid() - 1);
    }

    double cor() const {
        double s1 = sd1();
        double s2 = sd2();
        if (std::isnan(s1) || std::isnan(s2) || s1 == 0 || s2 == 0)
            return std::numeric_limits<double>::quiet_NaN();
        return cov() / (s1 * s2);
    }

    void merge(const CorSummary &other) {
        // Parallel Welford-style merge for two-pass-free streaming correlation
        // This is a simplified merge that is exact when both summaries
        // are computed over non-overlapping partitions.
        double nv1 = n_valid();
        double nv2 = other.n_valid();
        n += other.n;
        n_na += other.n_na;
        sum1 += other.sum1;
        sum2 += other.sum2;
        sumsq1 += other.sumsq1;
        sumsq2 += other.sumsq2;
        cross += other.cross;
        (void)nv1;
        (void)nv2;
    }
};

struct ValuePair {
    double x;
    double y;
};

struct SpearmanSummary {
    double n{0};
    double n_na{0};
    double cor{std::numeric_limits<double>::quiet_NaN()};
};

// Compute ranks with average ties (same behavior as R rank(..., ties.method="average"))
static void compute_ranks(const std::vector<double> &values, std::vector<double> &ranks)
{
    uint64_t n = values.size();
    ranks.resize(n);

    std::vector<uint64_t> idx(n);
    for (uint64_t i = 0; i < n; ++i)
        idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&values](uint64_t a, uint64_t b) {
        return values[a] < values[b];
    });

    uint64_t i = 0;
    while (i < n) {
        uint64_t j = i + 1;
        while (j < n && values[idx[j]] == values[idx[i]])
            ++j;

        double avg_rank = (i + 1 + j) / 2.0;  // 1-based ranks
        for (uint64_t k = i; k < j; ++k)
            ranks[idx[k]] = avg_rank;
        i = j;
    }
}

static double pearson_from_values(const std::vector<double> &x, const std::vector<double> &y)
{
    uint64_t n = x.size();
    if (n < 2)
        return std::numeric_limits<double>::quiet_NaN();

    double sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0;
    for (uint64_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
        sum_xy += x[i] * y[i];
    }

    double var_x = (sum_x2 - (sum_x * sum_x) / n) / (n - 1);
    double var_y = (sum_y2 - (sum_y * sum_y) / n) / (n - 1);
    if (var_x <= 0 || var_y <= 0)
        return std::numeric_limits<double>::quiet_NaN();

    double cov = (sum_xy - (sum_x * sum_y) / n) / (n - 1);
    return cov / (std::sqrt(var_x) * std::sqrt(var_y));
}

static double estimate_rank(double value, const std::vector<double> &sorted_sample, uint64_t total_count)
{
    if (sorted_sample.empty())
        return std::numeric_limits<double>::quiet_NaN();

    auto it = std::lower_bound(sorted_sample.begin(), sorted_sample.end(), value);
    uint64_t pos = (uint64_t)(it - sorted_sample.begin());
    auto upper = std::upper_bound(sorted_sample.begin(), sorted_sample.end(), value);
    uint64_t ties = (uint64_t)(upper - it);
    double scale = (double)total_count / sorted_sample.size();

    if (ties > 0) {
        double low_rank = pos * scale + 1;
        double high_rank = (pos + ties) * scale;
        return (low_rank + high_rank) / 2.0;
    }
    return pos * scale + 0.5;
}

static CorSummary compute_cor_pearson(const std::string &expr1, const std::string &expr2,
                                      const std::vector<GInterval> &intervals,
                                      long iterator_policy)
{
    CorSummary summary;
    if (intervals.empty())
        return summary;

    // Try combined scanner first (works when both expressions use same track type)
    try {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr1, expr2};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);

        for (; !scanner.isend(); scanner.next()) {
            double v1 = scanner.vdouble(0);
            double v2 = scanner.vdouble(1);
            summary.update(v1, v2);
            check_interrupt();
        }

        return summary;
    } catch (TGLException &) {
        // Fall through to separate-scanner approach for mixed track types
    }

    // Separate scanners for mixed track types: extract both then correlate
    std::vector<double> vals1, vals2;
    {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr1};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            vals1.push_back(scanner.vdouble(0));
            check_interrupt();
        }
    }
    {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr2};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            vals2.push_back(scanner.vdouble(0));
            check_interrupt();
        }
    }

    if (vals1.size() != vals2.size()) {
        verror("Expressions '%s' and '%s' produce different numbers of values (%zu vs %zu). "
               "This can happen with mixed track types and different iterator alignments.",
               expr1.c_str(), expr2.c_str(), vals1.size(), vals2.size());
    }

    for (size_t i = 0; i < vals1.size(); ++i) {
        summary.update(vals1[i], vals2[i]);
    }

    return summary;
}

static SpearmanSummary compute_cor_spearman_exact(const std::string &expr1, const std::string &expr2,
                                                  const std::vector<GInterval> &intervals,
                                                  long iterator_policy)
{
    auto build_summary = [](const std::vector<ValuePair> &pairs, uint64_t total_bins) {
        SpearmanSummary s;
        s.n = total_bins;
        s.n_na = total_bins - pairs.size();
        if (pairs.size() < 2) {
            s.cor = std::numeric_limits<double>::quiet_NaN();
            return s;
        }
        std::vector<double> xvals(pairs.size()), yvals(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) {
            xvals[i] = pairs[i].x;
            yvals[i] = pairs[i].y;
        }
        std::vector<double> xranks, yranks;
        compute_ranks(xvals, xranks);
        compute_ranks(yvals, yranks);
        s.cor = pearson_from_values(xranks, yranks);
        return s;
    };

    SpearmanSummary summary;
    if (intervals.empty())
        return summary;

    std::vector<ValuePair> pairs;
    uint64_t total_bins = 0;

    auto add_pair = [&](double x, double y) {
        ++total_bins;
        if (!std::isnan(x) && !std::isnan(y)) {
            pairs.push_back({x, y});
            g_pymisha->verify_max_data_size(pairs.size(), "Result");
        }
    };

    // Try combined scanner first.
    try {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr1, expr2};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            add_pair(scanner.vdouble(0), scanner.vdouble(1));
            check_interrupt();
        }
    } catch (TGLException &) {
        // Mixed track types fallback.
        std::vector<double> vals1, vals2;
        {
            PMTrackExprScanner scanner;
            std::vector<std::string> exprs = {expr1};
            scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
            for (; !scanner.isend(); scanner.next()) {
                vals1.push_back(scanner.vdouble(0));
                check_interrupt();
            }
        }
        {
            PMTrackExprScanner scanner;
            std::vector<std::string> exprs = {expr2};
            scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
            for (; !scanner.isend(); scanner.next()) {
                vals2.push_back(scanner.vdouble(0));
                check_interrupt();
            }
        }

        if (vals1.size() != vals2.size()) {
            verror("Expressions '%s' and '%s' produce different numbers of values (%zu vs %zu). "
                   "This can happen with mixed track types and different iterator alignments.",
                   expr1.c_str(), expr2.c_str(), vals1.size(), vals2.size());
        }

        for (size_t i = 0; i < vals1.size(); ++i)
            add_pair(vals1[i], vals2[i]);
    }

    return build_summary(pairs, total_bins);
}

static void collect_cor_spearman_exact_pairs(const std::string &expr1, const std::string &expr2,
                                             const std::vector<GInterval> &intervals,
                                             long iterator_policy,
                                             std::vector<ValuePair> &pairs,
                                             uint64_t &total_bins)
{
    pairs.clear();
    total_bins = 0;

    auto add_pair = [&](double x, double y) {
        ++total_bins;
        if (!std::isnan(x) && !std::isnan(y)) {
            pairs.push_back({x, y});
            g_pymisha->verify_max_data_size(pairs.size(), "Result");
        }
    };

    // Try combined scanner first.
    try {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr1, expr2};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            add_pair(scanner.vdouble(0), scanner.vdouble(1));
            check_interrupt();
        }
        return;
    } catch (TGLException &) {
        // Mixed track types fallback.
    }

    std::vector<double> vals1, vals2;
    {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr1};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            vals1.push_back(scanner.vdouble(0));
            check_interrupt();
        }
    }
    {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr2};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            vals2.push_back(scanner.vdouble(0));
            check_interrupt();
        }
    }

    if (vals1.size() != vals2.size()) {
        verror("Expressions '%s' and '%s' produce different numbers of values (%zu vs %zu). "
               "This can happen with mixed track types and different iterator alignments.",
               expr1.c_str(), expr2.c_str(), vals1.size(), vals2.size());
    }

    for (size_t i = 0; i < vals1.size(); ++i)
        add_pair(vals1[i], vals2[i]);
}

static SpearmanSummary spearman_summary_from_pairs(const std::vector<ValuePair> &pairs, uint64_t total_bins)
{
    SpearmanSummary summary;
    summary.n = total_bins;
    summary.n_na = total_bins - pairs.size();
    if (pairs.size() < 2) {
        summary.cor = std::numeric_limits<double>::quiet_NaN();
        return summary;
    }

    std::vector<double> xvals(pairs.size()), yvals(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        xvals[i] = pairs[i].x;
        yvals[i] = pairs[i].y;
    }

    std::vector<double> xranks, yranks;
    compute_ranks(xvals, xranks);
    compute_ranks(yvals, yranks);
    summary.cor = pearson_from_values(xranks, yranks);
    return summary;
}

static SpearmanSummary compute_cor_spearman_approx(const std::string &expr1, const std::string &expr2,
                                                   const std::vector<GInterval> &intervals,
                                                   long iterator_policy,
                                                   uint64_t sample_size)
{
    SpearmanSummary summary;
    if (intervals.empty())
        return summary;
    if (sample_size == 0)
        sample_size = 1;

    StreamSampler<ValuePair> pair_sampler(sample_size, true);
    StreamSampler<double> x_sampler(sample_size, true);
    StreamSampler<double> y_sampler(sample_size, true);

    uint64_t non_nan = 0;

    auto add_pair = [&](double x, double y) {
        ++summary.n;
        if (std::isnan(x) || std::isnan(y))
            return;
        ++non_nan;
        pair_sampler.add({x, y}, drand48);
        x_sampler.add(x, drand48);
        y_sampler.add(y, drand48);
    };

    // Try combined scanner first.
    try {
        PMTrackExprScanner scanner;
        std::vector<std::string> exprs = {expr1, expr2};
        scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
        for (; !scanner.isend(); scanner.next()) {
            add_pair(scanner.vdouble(0), scanner.vdouble(1));
            check_interrupt();
        }
    } catch (TGLException &) {
        // Mixed track types fallback.
        std::vector<double> vals1, vals2;
        {
            PMTrackExprScanner scanner;
            std::vector<std::string> exprs = {expr1};
            scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
            for (; !scanner.isend(); scanner.next()) {
                vals1.push_back(scanner.vdouble(0));
                check_interrupt();
            }
        }
        {
            PMTrackExprScanner scanner;
            std::vector<std::string> exprs = {expr2};
            scanner.begin(exprs, PMTrackExprScanner::REAL_T, intervals, iterator_policy);
            for (; !scanner.isend(); scanner.next()) {
                vals2.push_back(scanner.vdouble(0));
                check_interrupt();
            }
        }

        if (vals1.size() != vals2.size()) {
            verror("Expressions '%s' and '%s' produce different numbers of values (%zu vs %zu). "
                   "This can happen with mixed track types and different iterator alignments.",
                   expr1.c_str(), expr2.c_str(), vals1.size(), vals2.size());
        }

        for (size_t i = 0; i < vals1.size(); ++i)
            add_pair(vals1[i], vals2[i]);
    }

    summary.n_na = summary.n - non_nan;
    if (non_nan < 2) {
        summary.cor = std::numeric_limits<double>::quiet_NaN();
        return summary;
    }

    std::vector<double> &xs = x_sampler.samples();
    std::vector<double> &ys = y_sampler.samples();
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());

    const std::vector<ValuePair> &pairs = pair_sampler.samples();
    std::vector<double> xranks(pairs.size()), yranks(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        xranks[i] = estimate_rank(pairs[i].x, xs, non_nan);
        yranks[i] = estimate_rank(pairs[i].y, ys, non_nan);
    }

    summary.cor = pearson_from_values(xranks, yranks);
    return summary;
}

PyObject *pm_cor(PyObject *self, PyObject *args)
{
    try {
        PyMisha pymisha(true);

        PyObject *py_exprs = NULL;
        PyObject *py_intervals = NULL;
        PyObject *py_iterator = NULL;
        PyObject *py_method = NULL;
        PyObject *py_config = NULL;

        if (!PyArg_ParseTuple(args, "OO|OOO", &py_exprs, &py_intervals,
                              &py_iterator, &py_method, &py_config)) {
            verror("Invalid arguments to pm_cor");
        }

        // Backward compatibility with old signature: (exprs, intervals, iterator, config)
        if (!py_config && py_method && PyDict_Check(py_method)) {
            py_config = py_method;
            py_method = NULL;
        }

        // Parse expression pairs from Python list
        if (!PyList_Check(py_exprs))
            verror("gcor expressions must be a list of strings");

        Py_ssize_t num_exprs = PyList_Size(py_exprs);
        if (num_exprs < 2 || num_exprs % 2 != 0)
            verror("gcor requires an even number of expressions (pairs)");

        std::vector<std::string> exprs;
        for (Py_ssize_t i = 0; i < num_exprs; ++i) {
            PyObject *item = PyList_GetItem(py_exprs, i);
            if (!PyUnicode_Check(item))
                verror("All expressions must be strings");
            exprs.push_back(PyUnicode_AsUTF8(item));
        }

        long iterator_policy = 0;
        iterator_policy = parse_iterator_policy(py_iterator, iterator_policy, "pm_cor");
        CorMethod method = parse_cor_method(py_method);

        std::vector<GInterval> intervals;
        convert_py_intervals(py_intervals, intervals);

        size_t num_pairs = exprs.size() / 2;
        std::vector<CorSummary> summaries(num_pairs);
        std::vector<SpearmanSummary> spearman_summaries(num_pairs);

        // Range-based splitting for fixed-bin iterators
        bool use_range_split_cor = (iterator_policy > 0);
        int num_kids;
        std::vector<std::vector<SplitInterval>> range_kids;

        if (method == CorMethod::PEARSON || method == CorMethod::SPEARMAN_EXACT) {
            if (use_range_split_cor) {
                int64_t total_range = 0;
                for (const auto &iv : intervals) total_range += iv.end - iv.start;
                num_kids = choose_num_kids_for_range(intervals.size(), total_range, iterator_policy);
                if (num_kids > 0) {
                    split_intervals_by_range(intervals, num_kids, iterator_policy, range_kids);
                    num_kids = (int)range_kids.size();
                }
            } else {
                num_kids = choose_num_kids(intervals.size(), total_intervals_scope(intervals));
            }
        } else {
            num_kids = 0;
        }

        uint64_t sample_size = g_pymisha->max_data_size();
        if (sample_size < 1)
            sample_size = 1;

        if (num_kids > 0) {
            PyMisha::prepare4multitasking();

            for (int kid = 0; kid < num_kids; ++kid) {
                pid_t pid = PyMisha::launch_process();
                if (pid == 0) {
                    std::vector<GInterval> sub;
                    if (!range_kids.empty()) {
                        extract_sub_intervals(range_kids[kid], sub);
                    } else {
                        size_t start = (intervals.size() * kid) / num_kids;
                        size_t end = (intervals.size() * (kid + 1)) / num_kids;
                        sub.assign(intervals.begin() + start, intervals.begin() + end);
                    }

                    for (size_t p = 0; p < num_pairs; ++p) {
                        if (method == CorMethod::PEARSON) {
                            CorSummary kid_summary = compute_cor_pearson(exprs[p * 2], exprs[p * 2 + 1],
                                                                         sub, iterator_policy);
                            PyMisha::write_multitask_fifo(&kid_summary, sizeof(kid_summary));
                        } else {
                            std::vector<ValuePair> kid_pairs;
                            uint64_t total_bins = 0;
                            collect_cor_spearman_exact_pairs(exprs[p * 2], exprs[p * 2 + 1],
                                                             sub, iterator_policy, kid_pairs, total_bins);

                            struct SpearmanHeader {
                                uint64_t total_bins;
                                uint64_t pair_count;
                            } header{total_bins, static_cast<uint64_t>(kid_pairs.size())};

                            PyMisha::lock_multitask_fifo();
                            try {
                                PyMisha::write_multitask_fifo_unlocked(&header, sizeof(header));
                                if (!kid_pairs.empty()) {
                                    PyMisha::write_multitask_fifo_unlocked(
                                        kid_pairs.data(),
                                        kid_pairs.size() * sizeof(ValuePair)
                                    );
                                }
                                PyMisha::unlock_multitask_fifo();
                            } catch (...) {
                                PyMisha::unlock_multitask_fifo();
                                throw;
                            }
                        }
                    }
                    _exit(0);
                }
            }

            std::vector<uint64_t> total_bins_per_pair(num_pairs, 0);
            std::vector<std::vector<ValuePair>> merged_pairs(num_pairs);

            for (int kid = 0; kid < num_kids; ++kid) {
                for (size_t p = 0; p < num_pairs; ++p) {
                    if (method == CorMethod::PEARSON) {
                        CorSummary kid_summary;
                        PyMisha::read_multitask_fifo(&kid_summary, sizeof(kid_summary));
                        summaries[p].merge(kid_summary);
                    } else {
                        struct SpearmanHeader {
                            uint64_t total_bins;
                            uint64_t pair_count;
                        } header{0, 0};

                        PyMisha::read_multitask_fifo(&header, sizeof(header));
                        total_bins_per_pair[p] += header.total_bins;
                        if (header.pair_count > 0) {
                            size_t old_size = merged_pairs[p].size();
                            merged_pairs[p].resize(old_size + static_cast<size_t>(header.pair_count));
                            PyMisha::read_multitask_fifo(
                                merged_pairs[p].data() + old_size,
                                static_cast<size_t>(header.pair_count) * sizeof(ValuePair)
                            );
                        }
                    }
                }
            }

            while (PyMisha::wait_for_kids(1))
                ;

            if (method == CorMethod::SPEARMAN_EXACT) {
                for (size_t p = 0; p < num_pairs; ++p) {
                    spearman_summaries[p] = spearman_summary_from_pairs(
                        merged_pairs[p], total_bins_per_pair[p]
                    );
                }
            }
        } else {
            for (size_t p = 0; p < num_pairs; ++p) {
                if (method == CorMethod::PEARSON) {
                    summaries[p] = compute_cor_pearson(exprs[p * 2], exprs[p * 2 + 1],
                                                       intervals, iterator_policy);
                } else if (method == CorMethod::SPEARMAN_EXACT) {
                    spearman_summaries[p] = compute_cor_spearman_exact(exprs[p * 2], exprs[p * 2 + 1],
                                                                        intervals, iterator_policy);
                } else {
                    spearman_summaries[p] = compute_cor_spearman_approx(exprs[p * 2], exprs[p * 2 + 1],
                                                                         intervals, iterator_policy, sample_size);
                }
            }
        }

        // Build result: list of dicts, one per pair.
        PMPY result(PyList_New(num_pairs), true);

        for (size_t p = 0; p < num_pairs; ++p) {
            PMPY d(PyDict_New(), true);

            auto set_d = [&](const char *key, double val) {
                PMPY v(PyFloat_FromDouble(val), true);
                PyDict_SetItemString(d, key, v);
            };

            if (method == CorMethod::PEARSON) {
                const CorSummary &s = summaries[p];
                set_d("cor", s.cor());
                set_d("cov", s.cov());
                set_d("mean1", s.mean1());
                set_d("mean2", s.mean2());
                set_d("sd1", s.sd1());
                set_d("sd2", s.sd2());
                set_d("n", s.n);
                set_d("n.na", s.n_na);
            } else {
                const SpearmanSummary &s = spearman_summaries[p];
                set_d("n", s.n);
                set_d("n.na", s.n_na);
                set_d("cor", s.cor);
            }

            d.to_be_stolen();
            PyList_SetItem(result, p, (PyObject *)d);
        }

        result.to_be_stolen();
        return result;

    } catch (TGLException &e) {
        PyMisha::handle_error(e.msg());
        return NULL;
    } catch (const std::bad_alloc &e) {
        PyMisha::handle_error("Out of memory");
        return NULL;
    }
}

// -------------------------------------------------------------------
// Quad-tree C++ reader functions
// -------------------------------------------------------------------

PyObject *pm_quadtree_query_stats(PyObject *self, PyObject *args)
{
    Py_buffer pybuf;
    long long qx1, qy1, qx2, qy2;
    int is_points_int;
    long long band_d1 = 0, band_d2 = 0;
    int has_band = 0;

    // Args: buffer, x1, y1, x2, y2, is_points, has_band, band_d1, band_d2
    if (!PyArg_ParseTuple(args, "y*LLLLi|iLL",
                          &pybuf, &qx1, &qy1, &qx2, &qy2,
                          &is_points_int, &has_band, &band_d1, &band_d2)) {
        return NULL;
    }

    const char *buf = static_cast<const char *>(pybuf.buf);
    size_t len = static_cast<size_t>(pybuf.len);
    bool is_points = (is_points_int != 0);

    PyObject *result = NULL;

    try {
        // Parse file header (signature at offset 0, then num_objs, root_chunk_fpos)
        uint64_t num_objs;
        int64_t root_chunk_fpos;
        quadtree::parse_header(buf, len, num_objs, root_chunk_fpos);

        quadtree::QueryStat stat;
        if (num_objs == 0) {
            stat.occupied_area = 0;
            stat.weighted_sum = std::numeric_limits<double>::quiet_NaN();
            stat.min_val = std::numeric_limits<double>::quiet_NaN();
            stat.max_val = std::numeric_limits<double>::quiet_NaN();
        } else {
            if (has_band) {
                quadtree::DiagonalBand band(band_d1, band_d2);
                stat = quadtree::query_stats(buf, len, is_points, root_chunk_fpos,
                                             qx1, qy1, qx2, qy2, &band);
            } else {
                stat = quadtree::query_stats(buf, len, is_points, root_chunk_fpos,
                                             qx1, qy1, qx2, qy2);
            }
        }

        // Build result dict
        result = PyDict_New();
        if (!result) goto cleanup;

        {
            PyObject *v;
            v = PyLong_FromLongLong(stat.occupied_area);
            PyDict_SetItemString(result, "occupied_area", v);
            Py_DECREF(v);

            v = PyFloat_FromDouble(stat.weighted_sum);
            PyDict_SetItemString(result, "weighted_sum", v);
            Py_DECREF(v);

            v = PyFloat_FromDouble(stat.min_val);
            PyDict_SetItemString(result, "min_val", v);
            Py_DECREF(v);

            v = PyFloat_FromDouble(stat.max_val);
            PyDict_SetItemString(result, "max_val", v);
            Py_DECREF(v);
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        result = NULL;
    }

cleanup:
    PyBuffer_Release(&pybuf);
    return result;
}

PyObject *pm_quadtree_query_objects(PyObject *self, PyObject *args)
{
    Py_buffer pybuf;
    long long qx1, qy1, qx2, qy2;
    int is_points_int;
    long long band_d1 = 0, band_d2 = 0;
    int has_band = 0;

    if (!PyArg_ParseTuple(args, "y*LLLLi|iLL",
                          &pybuf, &qx1, &qy1, &qx2, &qy2,
                          &is_points_int, &has_band, &band_d1, &band_d2)) {
        return NULL;
    }

    const char *buf = static_cast<const char *>(pybuf.buf);
    size_t len = static_cast<size_t>(pybuf.len);
    bool is_points = (is_points_int != 0);

    PyObject *result = NULL;

    try {
        uint64_t num_objs;
        int64_t root_chunk_fpos;
        quadtree::parse_header(buf, len, num_objs, root_chunk_fpos);

        quadtree::QueryObjects qresult;
        if (num_objs > 0) {
            if (has_band) {
                quadtree::DiagonalBand band(band_d1, band_d2);
                qresult = quadtree::query_objects(buf, len, is_points, root_chunk_fpos,
                                                  qx1, qy1, qx2, qy2, &band);
            } else {
                qresult = quadtree::query_objects(buf, len, is_points, root_chunk_fpos,
                                                  qx1, qy1, qx2, qy2);
            }
        }

        // Build result dict of numpy arrays
        size_t n = qresult.ids.size();
        npy_intp dims[1] = {static_cast<npy_intp>(n)};

        result = PyDict_New();
        if (!result) goto cleanup;

        {
            // id array (uint64)
            PyObject *arr_id = PyArray_SimpleNew(1, dims, NPY_UINT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_id), qresult.ids.data(), n * sizeof(uint64_t));
            PyDict_SetItemString(result, "id", arr_id);
            Py_DECREF(arr_id);

            // x1 array (int64)
            PyObject *arr_x1 = PyArray_SimpleNew(1, dims, NPY_INT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_x1), qresult.x1s.data(), n * sizeof(int64_t));
            PyDict_SetItemString(result, "x1", arr_x1);
            Py_DECREF(arr_x1);

            // y1 array (int64)
            PyObject *arr_y1 = PyArray_SimpleNew(1, dims, NPY_INT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_y1), qresult.y1s.data(), n * sizeof(int64_t));
            PyDict_SetItemString(result, "y1", arr_y1);
            Py_DECREF(arr_y1);

            // x2 array (int64)
            PyObject *arr_x2 = PyArray_SimpleNew(1, dims, NPY_INT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_x2), qresult.x2s.data(), n * sizeof(int64_t));
            PyDict_SetItemString(result, "x2", arr_x2);
            Py_DECREF(arr_x2);

            // y2 array (int64)
            PyObject *arr_y2 = PyArray_SimpleNew(1, dims, NPY_INT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_y2), qresult.y2s.data(), n * sizeof(int64_t));
            PyDict_SetItemString(result, "y2", arr_y2);
            Py_DECREF(arr_y2);

            // val array (float32)
            PyObject *arr_val = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_val), qresult.vals.data(), n * sizeof(float));
            PyDict_SetItemString(result, "val", arr_val);
            Py_DECREF(arr_val);
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        result = NULL;
    }

cleanup:
    PyBuffer_Release(&pybuf);
    return result;
}

PyObject *pm_quadtree_query_stats_batch(PyObject *self, PyObject *args)
{
    Py_buffer pybuf;
    PyArrayObject *rects_arr = NULL;
    int is_points_int;
    int has_band = 0;
    long long band_d1 = 0, band_d2 = 0;

    // Args: buffer, rects_numpy(N×4 int64), is_points, has_band, band_d1, band_d2
    if (!PyArg_ParseTuple(args, "y*O!i|iLL",
                          &pybuf,
                          &PyArray_Type, &rects_arr,
                          &is_points_int,
                          &has_band, &band_d1, &band_d2)) {
        return NULL;
    }

    const char *buf = static_cast<const char *>(pybuf.buf);
    size_t len = static_cast<size_t>(pybuf.len);
    bool is_points = (is_points_int != 0);

    // Validate rects array: must be contiguous int64 with shape (N, 4)
    if (PyArray_NDIM(rects_arr) != 2 || PyArray_DIM(rects_arr, 1) != 4 ||
        PyArray_TYPE(rects_arr) != NPY_INT64 ||
        !PyArray_IS_C_CONTIGUOUS(rects_arr)) {
        PyBuffer_Release(&pybuf);
        PyErr_SetString(PyExc_ValueError,
                        "rects must be a contiguous int64 array with shape (N, 4)");
        return NULL;
    }

    size_t n = static_cast<size_t>(PyArray_DIM(rects_arr, 0));
    const int64_t *rects_data = static_cast<const int64_t *>(PyArray_DATA(rects_arr));

    PyObject *result = NULL;

    try {
        uint64_t num_objs;
        int64_t root_chunk_fpos;
        quadtree::parse_header(buf, len, num_objs, root_chunk_fpos);

        quadtree::BatchQueryStats batch;
        if (num_objs == 0 || n == 0) {
            batch.resize(n);
        } else {
            if (has_band) {
                quadtree::DiagonalBand band_obj(band_d1, band_d2);
                batch = quadtree::query_stats_batch(buf, len, is_points,
                                                     root_chunk_fpos,
                                                     rects_data, n, &band_obj);
            } else {
                batch = quadtree::query_stats_batch(buf, len, is_points,
                                                     root_chunk_fpos,
                                                     rects_data, n);
            }
        }

        // Build result dict of numpy arrays
        npy_intp dims[1] = {static_cast<npy_intp>(n)};
        result = PyDict_New();
        if (!result) goto batch_cleanup;

        {
            PyObject *arr_occ = PyArray_SimpleNew(1, dims, NPY_INT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_occ),
                              batch.occupied_area.data(), n * sizeof(int64_t));
            PyDict_SetItemString(result, "occupied_area", arr_occ);
            Py_DECREF(arr_occ);

            PyObject *arr_ws = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_ws),
                              batch.weighted_sum.data(), n * sizeof(double));
            PyDict_SetItemString(result, "weighted_sum", arr_ws);
            Py_DECREF(arr_ws);

            PyObject *arr_min = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_min),
                              batch.min_val.data(), n * sizeof(double));
            PyDict_SetItemString(result, "min_val", arr_min);
            Py_DECREF(arr_min);

            PyObject *arr_max = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
            if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr_max),
                              batch.max_val.data(), n * sizeof(double));
            PyDict_SetItemString(result, "max_val", arr_max);
            Py_DECREF(arr_max);
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        result = NULL;
    }

batch_cleanup:
    PyBuffer_Release(&pybuf);
    return result;
}

//---------------------------------------------------------------------------
// pm_test_2d_iterator
//
// Test-only binding that constructs a PMTrackExpressionIntervals2DIterator
// over the supplied dict-of-arrays input and emits the intervals back as a
// dict of arrays. Used by tests/test_2d_scanner_iterator.py.
//---------------------------------------------------------------------------

static int _pm_test_2d_get_array(PyObject *dict, const char *key, PyArrayObject **out)
{
    PyObject *item = PyDict_GetItemString(dict, key);
    if (!item) {
        PyErr_Format(PyExc_KeyError, "missing key '%s' in intervals dict", key);
        return -1;
    }
    if (!PyArray_Check(item)) {
        PyErr_Format(PyExc_ValueError, "'%s' must be a numpy array", key);
        return -1;
    }
    *out = (PyArrayObject *)item;
    return 0;
}

PyObject *pm_test_2d_iterator(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *intervals_dict = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &intervals_dict)) {
        return NULL;
    }

    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(intervals_dict, "chrom1", &arr_c1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start1", &arr_s1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end1",   &arr_e1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "chrom2", &arr_c2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start2", &arr_s2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end2",   &arr_e2) < 0) return NULL;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "intervals arrays must all be the same length");
        return NULL;
    }

    // Materialize input as a std::vector<GInterval2D>.
    std::vector<GInterval2D> input;
    input.reserve(n);
    for (npy_intp i = 0; i < n; ++i) {
        int c1 = (int) (*(int32_t *)PyArray_GETPTR1(arr_c1, i));
        int c2 = (int) (*(int32_t *)PyArray_GETPTR1(arr_c2, i));
        int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, i);
        int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, i);
        int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, i);
        int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, i);
        input.emplace_back(c1, s1, e1, c2, s2, e2);
    }

    PMTrackExpressionIntervals2DIterator it(input);
    it.begin();

    std::vector<int>     out_c1, out_c2;
    std::vector<int64_t> out_s1, out_e1, out_s2, out_e2;
    std::vector<int64_t> out_id;
    out_c1.reserve(n); out_c2.reserve(n);
    out_s1.reserve(n); out_e1.reserve(n);
    out_s2.reserve(n); out_e2.reserve(n);
    out_id.reserve(n);

    while (!it.isend()) {
        const GInterval2D &iv = it.last_interval();
        out_c1.push_back(iv.chromid1);
        out_c2.push_back(iv.chromid2);
        out_s1.push_back(iv.start1);
        out_e1.push_back(iv.end1);
        out_s2.push_back(iv.start2);
        out_e2.push_back(iv.end2);
        out_id.push_back(static_cast<int64_t>(it.original_interval_idx()));
        it.next();
    }

    npy_intp m = static_cast<npy_intp>(out_c1.size());
    npy_intp dims[1] = {m};

    PyObject *result = PyDict_New();
    if (!result) return NULL;

    #define SET_INT32(name, vec)                                          \
        do {                                                              \
            PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT32);           \
            if (m > 0) {                                                  \
                int32_t *ptr = (int32_t *)PyArray_DATA((PyArrayObject *)a);\
                for (npy_intp i = 0; i < m; ++i) ptr[i] = (vec)[i];        \
            }                                                             \
            PyDict_SetItemString(result, name, a);                        \
            Py_DECREF(a);                                                 \
        } while (0)

    #define SET_INT64(name, vec)                                          \
        do {                                                              \
            PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT64);           \
            if (m > 0) memcpy(PyArray_DATA((PyArrayObject *)a),            \
                              (vec).data(), m * sizeof(int64_t));         \
            PyDict_SetItemString(result, name, a);                        \
            Py_DECREF(a);                                                 \
        } while (0)

    SET_INT32("chrom1", out_c1);
    SET_INT64("start1", out_s1);
    SET_INT64("end1",   out_e1);
    SET_INT32("chrom2", out_c2);
    SET_INT64("start2", out_s2);
    SET_INT64("end2",   out_e2);
    SET_INT64("interval_id", out_id);

    #undef SET_INT32
    #undef SET_INT64

    return result;
}

//---------------------------------------------------------------------------
// pm_test_fixed_rect_iterator
//
// Test-only binding that constructs a PMTrackExpressionFixedRectIterator and
// emits all grid cells back as a dict of arrays.
// Args: width (int), height (int), intervals_dict (dict of 6 numpy arrays),
//       band (None or (d1, d2) tuple).
// Returns: dict with keys chrom1/start1/end1/chrom2/start2/end2.
//---------------------------------------------------------------------------

PyObject *pm_test_fixed_rect_iterator(PyObject *self, PyObject *args)
{
    (void)self;
    long long width_l, height_l;
    PyObject *intervals_dict = NULL;
    PyObject *band_obj = NULL;
    if (!PyArg_ParseTuple(args, "LLOO",
                          &width_l, &height_l, &intervals_dict, &band_obj))
        return NULL;

    // Parse intervals dict.
    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(intervals_dict, "chrom1", &arr_c1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start1", &arr_s1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end1",   &arr_e1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "chrom2", &arr_c2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start2", &arr_s2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end2",   &arr_e2) < 0) return NULL;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "intervals arrays must all be the same length");
        return NULL;
    }

    std::vector<GInterval2D> scope;
    scope.reserve(n);
    for (npy_intp i = 0; i < n; ++i) {
        int     c1 = (int)(*(int32_t *)PyArray_GETPTR1(arr_c1, i));
        int     c2 = (int)(*(int32_t *)PyArray_GETPTR1(arr_c2, i));
        int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, i);
        int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, i);
        int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, i);
        int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, i);
        scope.emplace_back(c1, s1, e1, c2, s2, e2);
    }

    // Parse optional band.
    std::unique_ptr<quadtree::DiagonalBand> band_ptr;
    if (band_obj != Py_None) {
        long long d1, d2;
        if (!PyArg_ParseTuple(band_obj, "LL", &d1, &d2))
            return NULL;
        band_ptr = std::make_unique<quadtree::DiagonalBand>(
            static_cast<int64_t>(d1), static_cast<int64_t>(d2));
    }

    // Construct and walk the iterator.
    // begin() must be called explicitly after construction (Option A contract).
    // Walk pattern: read last_interval(), push, call next().
    std::vector<int>     out_c1, out_c2;
    std::vector<int64_t> out_s1, out_e1, out_s2, out_e2;

    try {
        PMTrackExpressionFixedRectIterator it(
            static_cast<int64_t>(width_l),
            static_cast<int64_t>(height_l),
            std::move(scope),
            band_ptr.get());
        it.begin();

        while (!it.isend()) {
            const GInterval2D &iv = it.last_interval();
            out_c1.push_back(iv.chromid1);
            out_c2.push_back(iv.chromid2);
            out_s1.push_back(iv.start1);
            out_e1.push_back(iv.end1);
            out_s2.push_back(iv.start2);
            out_e2.push_back(iv.end2);
            it.next();
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    npy_intp m = static_cast<npy_intp>(out_c1.size());
    npy_intp dims[1] = {m};

    PyObject *result = PyDict_New();
    if (!result) return NULL;

    #define SET_INT32_FR(name, vec)                                           \
        do {                                                                  \
            PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT32);              \
            if (m > 0) {                                                      \
                int32_t *ptr = (int32_t *)PyArray_DATA((PyArrayObject *)a);   \
                for (npy_intp i = 0; i < m; ++i) ptr[i] = (vec)[i];          \
            }                                                                 \
            PyDict_SetItemString(result, name, a);                            \
            Py_DECREF(a);                                                     \
        } while (0)

    #define SET_INT64_FR(name, vec)                                           \
        do {                                                                  \
            PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT64);              \
            if (m > 0) memcpy(PyArray_DATA((PyArrayObject *)a),               \
                              (vec).data(), m * sizeof(int64_t));             \
            PyDict_SetItemString(result, name, a);                            \
            Py_DECREF(a);                                                     \
        } while (0)

    SET_INT32_FR("chrom1", out_c1);
    SET_INT64_FR("start1", out_s1);
    SET_INT64_FR("end1",   out_e1);
    SET_INT32_FR("chrom2", out_c2);
    SET_INT64_FR("start2", out_s2);
    SET_INT64_FR("end2",   out_e2);

    #undef SET_INT32_FR
    #undef SET_INT64_FR

    return result;
}

//---------------------------------------------------------------------------
// pm_test_fixed_rect_scanner
//
// Test-only binding that drives PMTrackExpr2DScanner::run() with a
// FixedRect iterator policy over a single (track, func) var and a scope
// dict, returning a numpy double array of per-cell aggregated values.
//
// Args:
//   width        : int  - FixedRect cell width
//   height       : int  - FixedRect cell height
//   track_name   : str  - 2D track to aggregate
//   func_name    : str  - agg func ("area" / "weighted.sum" / "min" / "max" / "avg")
//   intervals    : dict of 6 numpy arrays (chrom1/start1/end1/chrom2/start2/end2)
//   band         : (int, int) | None
//
// Returns:
//   numpy double array of length num_emitted (grid cells).
//---------------------------------------------------------------------------

PyObject *pm_test_fixed_rect_scanner(PyObject *self, PyObject *args)
{
    (void)self;
    long long   width_l, height_l;
    const char *track_name, *func_name;
    PyObject   *intervals_dict = NULL;
    PyObject   *band_obj = Py_None;

    if (!PyArg_ParseTuple(args, "LLssOO",
                          &width_l, &height_l,
                          &track_name, &func_name,
                          &intervals_dict, &band_obj))
        return NULL;

    // Parse the intervals dict (same convention as pm_test_fixed_rect_iterator).
    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(intervals_dict, "chrom1", &arr_c1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start1", &arr_s1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end1",   &arr_e1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "chrom2", &arr_c2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start2", &arr_s2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end2",   &arr_e2) < 0) return NULL;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "intervals arrays must all be the same length");
        return NULL;
    }

    std::vector<GInterval2D> scope;
    scope.reserve(static_cast<size_t>(n));
    for (npy_intp i = 0; i < n; ++i) {
        int     c1 = static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c1, i));
        int     c2 = static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c2, i));
        int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, i);
        int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, i);
        int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, i);
        int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, i);
        scope.emplace_back(c1, s1, e1, c2, s2, e2);
    }

    // Parse optional band.
    std::unique_ptr<quadtree::DiagonalBand> band_ptr;
    if (band_obj != Py_None) {
        long long d1, d2;
        if (!PyArg_ParseTuple(band_obj, "LL", &d1, &d2))
            return NULL;
        band_ptr = std::make_unique<quadtree::DiagonalBand>(
            static_cast<int64_t>(d1), static_cast<int64_t>(d2));
    }

    try {
        PMTrackExpression2DIteratorPolicy policy;
        policy.kind   = PMTrackExpression2DIteratorPolicy::K_FIXED_RECT;
        policy.width  = static_cast<int64_t>(width_l);
        policy.height = static_cast<int64_t>(height_l);

        PMTrackExpr2DScanner scanner;
        std::vector<PMTrackExpr2DScanner::VarSpec> var_specs = {
            {track_name, func_name},
        };
        scanner.run(policy, var_specs, scope, band_ptr.get());

        size_t         emit = scanner.num_emitted();
        const double  *vals = scanner.values_for_var(0);
        npy_intp       dims[1] = {static_cast<npy_intp>(emit)};
        PyObject      *arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!arr) return NULL;
        if (emit > 0 && vals) {
            std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr)),
                        vals, emit * sizeof(double));
        }
        return arr;
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

//---------------------------------------------------------------------------
// pm_test_scanner_reuse
//
// Test-only binding that calls scanner.run() TWICE on the same
// PMTrackExpr2DScanner instance to verify that state (m_vars, emitted
// intervals, counts) is correctly reset between calls.
//
// Args:
//   width        : int  - FixedRect cell width (same for both runs)
//   height       : int  - FixedRect cell height (same for both runs)
//   track_name   : str  - 2D track to aggregate
//   func_name    : str  - agg func
//   intervals1   : dict - scope for first run
//   intervals2   : dict - scope for second run
//
// Returns:
//   tuple (n1, n2, values2_list) where:
//     n1          = num_emitted() after first run
//     n2          = num_emitted() after second run
//     values2_list = list of float64 values from second run's var(0)
//
// If m_vars accumulated across calls, values_for_var(0) after the second
// run would point to the ghost first-run buffer with stale values; the
// test detects this by comparing against a fresh scanner's result.
//---------------------------------------------------------------------------

static int _parse_scope_dict(PyObject *d,
                             std::vector<GInterval2D> &out)
{
    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(d, "chrom1", &arr_c1) < 0) return -1;
    if (_pm_test_2d_get_array(d, "start1", &arr_s1) < 0) return -1;
    if (_pm_test_2d_get_array(d, "end1",   &arr_e1) < 0) return -1;
    if (_pm_test_2d_get_array(d, "chrom2", &arr_c2) < 0) return -1;
    if (_pm_test_2d_get_array(d, "start2", &arr_s2) < 0) return -1;
    if (_pm_test_2d_get_array(d, "end2",   &arr_e2) < 0) return -1;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "intervals arrays must all be the same length");
        return -1;
    }
    out.reserve(static_cast<size_t>(n));
    for (npy_intp i = 0; i < n; ++i) {
        int     c1 = static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c1, i));
        int     c2 = static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c2, i));
        int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, i);
        int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, i);
        int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, i);
        int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, i);
        out.emplace_back(c1, s1, e1, c2, s2, e2);
    }
    return 0;
}

PyObject *pm_test_scanner_reuse(PyObject *self, PyObject *args)
{
    (void)self;
    long long   width_l, height_l;
    const char *track_name, *func_name;
    PyObject   *dict1 = NULL, *dict2 = NULL;

    if (!PyArg_ParseTuple(args, "LLssOO",
                          &width_l, &height_l,
                          &track_name, &func_name,
                          &dict1, &dict2))
        return NULL;

    std::vector<GInterval2D> scope1, scope2;
    if (_parse_scope_dict(dict1, scope1) < 0) return NULL;
    if (_parse_scope_dict(dict2, scope2) < 0) return NULL;

    try {
        PMTrackExpression2DIteratorPolicy policy;
        policy.kind   = PMTrackExpression2DIteratorPolicy::K_FIXED_RECT;
        policy.width  = static_cast<int64_t>(width_l);
        policy.height = static_cast<int64_t>(height_l);

        std::vector<PMTrackExpr2DScanner::VarSpec> var_specs = {
            {track_name, func_name},
        };

        // Reuse the same scanner instance for both runs.
        PMTrackExpr2DScanner scanner;
        scanner.run(policy, var_specs, scope1, nullptr);
        size_t n1 = scanner.num_emitted();

        scanner.run(policy, var_specs, scope2, nullptr);
        size_t n2 = scanner.num_emitted();

        // Build the return value: (n1, n2, numpy_array_of_run2_values)
        const double *vals2 = scanner.values_for_var(0);
        npy_intp dims[1] = {static_cast<npy_intp>(n2)};
        PyObject *arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!arr) return NULL;
        if (n2 > 0 && vals2) {
            std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr)),
                        vals2, n2 * sizeof(double));
        }

        PyObject *result = PyTuple_New(3);
        if (!result) { Py_DECREF(arr); return NULL; }
        PyTuple_SET_ITEM(result, 0, PyLong_FromSize_t(n1));
        PyTuple_SET_ITEM(result, 1, PyLong_FromSize_t(n2));
        PyTuple_SET_ITEM(result, 2, arr);
        return result;
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

//---------------------------------------------------------------------------
// pm_test_2d_scanner
//
// Test-only binding that drives PMTrackExpr2DScanner over a single track
// and a 2D intervals input, returning a numpy double array of per-row
// aggregated values. Used by tests/test_2d_scanner_vars.py to verify
// byte-for-byte parity with the pure-Python _gextract_2d_vtrack_agg.
//
// Args:
//   track_name : str
//   intervals  : dict of 6 numpy arrays (same shape as pm_test_2d_iterator)
//   func       : str   ("area" / "weighted.sum" / "min" / "max" / "avg")
//   band       : (int, int) | None
//
// Returns:
//   numpy double array of length n.
//---------------------------------------------------------------------------

PyObject *pm_test_2d_scanner(PyObject *self, PyObject *args)
{
    (void)self;
    const char *track_name = NULL;
    PyObject   *intervals_dict = NULL;
    const char *func_name = NULL;
    PyObject   *band_obj = Py_None;

    if (!PyArg_ParseTuple(args, "sO!sO",
                          &track_name,
                          &PyDict_Type, &intervals_dict,
                          &func_name,
                          &band_obj)) {
        return NULL;
    }

    // Parse band - must be either None or a length-2 tuple/list of ints.
    bool has_band = false;
    long long band_d1 = 0, band_d2 = 0;
    if (band_obj && band_obj != Py_None) {
        PyObject *seq = PySequence_Fast(band_obj, "band must be a 2-tuple or None");
        if (!seq) return NULL;
        if (PySequence_Fast_GET_SIZE(seq) != 2) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "band must have length 2");
            return NULL;
        }
        band_d1 = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, 0));
        band_d2 = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, 1));
        Py_DECREF(seq);
        if (PyErr_Occurred()) return NULL;
        if (band_d1 >= band_d2) {
            PyErr_SetString(PyExc_ValueError, "band d1 must be < d2");
            return NULL;
        }
        has_band = true;
    }

    // Pull the six arrays out of the intervals dict.
    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(intervals_dict, "chrom1", &arr_c1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start1", &arr_s1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end1",   &arr_e1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "chrom2", &arr_c2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start2", &arr_s2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end2",   &arr_e2) < 0) return NULL;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "intervals arrays must all be the same length");
        return NULL;
    }

    // Materialise the intervals vector.
    std::vector<GInterval2D> input;
    input.reserve(static_cast<size_t>(n));
    for (npy_intp i = 0; i < n; ++i) {
        int c1 = static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c1, i));
        int c2 = static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c2, i));
        int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, i);
        int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, i);
        int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, i);
        int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, i);
        input.emplace_back(c1, s1, e1, c2, s2, e2);
    }

    try {
        PMTrackExpr2DScanner scanner;
        quadtree::DiagonalBand band_obj_real;
        const quadtree::DiagonalBand *band_ptr = nullptr;
        if (has_band) {
            band_obj_real = quadtree::DiagonalBand(band_d1, band_d2);
            band_ptr = &band_obj_real;
        }
        scanner.run_single_var(track_name, func_name, input, band_ptr);

        npy_intp dims[1] = {n};
        PyObject *out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!out) return NULL;
        if (n > 0 && scanner.values()) {
            std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(out)),
                        scanner.values(),
                        static_cast<size_t>(n) * sizeof(double));
        }
        return out;
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

// Forward declaration: defined later in this file (pm_test_cartesian_grid_iterator section).
static int _parse_1d_intervals_to_centers(
    PyObject *d,
    std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> &out);

//---------------------------------------------------------------------------
// pm_extract_2d_scanner
//
// Production binding: drives PMTrackExpr2DScanner with a polymorphic
// iterator policy.
//
// Args:
//   policy_dict    : dict  - {"kind": str, ...kind-specific params...}
//                            kind "fixed_rect"  requires "width" (int) and "height" (int)
//                            kind "track_rects" requires "track_name" (str)
//                            kinds "intervals" and "cartesian_grid"
//                            raise RuntimeError (planned for later releases)
//   intervals_dict : dict  - 2D scope as 6 numpy arrays:
//                            chrom1 (int32), start1 (int64), end1 (int64),
//                            chrom2 (int32), start2 (int64), end2 (int64)
//   vars_list      : list  - [(track_name: str, func_name: str), ...]
//   colnames_list  : list  - [str, ...] one colname per var
//   band           : tuple | None  - (d1: int, d2: int) diagonal-band filter
//
// Returns:
//   dict with one entry per colname (float64 ndarray of length num_emitted),
//   plus six coord arrays keyed _chrom1 (int32), _start1 (int64), _end1 (int64),
//   _chrom2 (int32), _start2 (int64), _end2 (int64).
//---------------------------------------------------------------------------

PyObject *pm_extract_2d_scanner(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *policy_dict   = nullptr;
    PyObject *intervals_dict = nullptr;
    PyObject *vars_list     = nullptr;
    PyObject *colnames_list = nullptr;
    PyObject *band_obj      = nullptr;

    if (!PyArg_ParseTuple(args, "OOOOO",
                          &policy_dict, &intervals_dict, &vars_list,
                          &colnames_list, &band_obj))
        return nullptr;

    // Parse the iterator policy.
    PyObject *kind_obj = PyDict_GetItemString(policy_dict, "kind");
    if (!kind_obj) {
        PyErr_SetString(PyExc_RuntimeError, "iterator policy: missing 'kind'");
        return nullptr;
    }
    if (!PyUnicode_Check(kind_obj)) {
        PyErr_SetString(PyExc_RuntimeError, "iterator policy: 'kind' must be a string");
        return nullptr;
    }
    std::string kind = PyUnicode_AsUTF8(kind_obj);

    PMTrackExpression2DIteratorPolicy policy{};
    if (kind == "fixed_rect") {
        PyObject *w = PyDict_GetItemString(policy_dict, "width");
        PyObject *h = PyDict_GetItemString(policy_dict, "height");
        if (!w || !h) {
            PyErr_SetString(PyExc_RuntimeError,
                            "FixedRect policy: missing width or height");
            return nullptr;
        }
        policy.kind   = PMTrackExpression2DIteratorPolicy::K_FIXED_RECT;
        policy.width  = PyLong_AsLongLong(w);
        policy.height = PyLong_AsLongLong(h);
        if (PyErr_Occurred()) return nullptr;
    } else if (kind == "track_rects") {
        PyObject *tn = PyDict_GetItemString(policy_dict, "track_name");
        if (!tn) {
            PyErr_SetString(PyExc_RuntimeError,
                            "TrackRects policy: missing track_name");
            return nullptr;
        }
        if (!PyUnicode_Check(tn)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "TrackRects policy: track_name must be a string");
            return nullptr;
        }
        if (!g_pmdb) {
            PyErr_SetString(PyExc_RuntimeError, "Database not initialized");
            return nullptr;
        }
        policy.kind       = PMTrackExpression2DIteratorPolicy::K_TRACK_RECTS;
        policy.track_name = PyUnicode_AsUTF8(tn);
        policy.chromkey   = &g_pmdb->chromkey();
    } else if (kind == "cartesian_grid") {
        if (!g_pmdb) {
            PyErr_SetString(PyExc_RuntimeError, "Database not initialized");
            return nullptr;
        }
        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Parse intervals1 (required).
        PyObject *ivd1 = PyDict_GetItemString(policy_dict, "intervals1");
        if (!ivd1) {
            PyErr_SetString(PyExc_RuntimeError, "CartesianGrid policy: missing 'intervals1'");
            return nullptr;
        }
        std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> centers1;
        if (_parse_1d_intervals_to_centers(ivd1, centers1) < 0)
            return nullptr;

        // Parse expansion1 (required).
        PyObject *exp1_obj = PyDict_GetItemString(policy_dict, "expansion1");
        if (!exp1_obj || !PyArray_Check(exp1_obj)) {
            PyErr_SetString(PyExc_RuntimeError, "CartesianGrid policy: missing or invalid 'expansion1'");
            return nullptr;
        }
        PyArrayObject *exp1_arr = (PyArrayObject *)exp1_obj;
        npy_intp ne1 = PyArray_DIM(exp1_arr, 0);
        std::vector<int64_t> expansion1;
        expansion1.reserve(static_cast<size_t>(ne1));
        for (npy_intp i = 0; i < ne1; ++i)
            expansion1.push_back(*(int64_t *)PyArray_GETPTR1(exp1_arr, i));
        std::sort(expansion1.begin(), expansion1.end());
        expansion1.erase(std::unique(expansion1.begin(), expansion1.end()), expansion1.end());
        if (expansion1.size() < 2) {
            PyErr_SetString(PyExc_RuntimeError, "CartesianGrid policy: expansion1 must contain at least 2 unique values");
            return nullptr;
        }

        // Parse intervals2 (None => reuse intervals1).
        PyObject *ivd2 = PyDict_GetItemString(policy_dict, "intervals2");
        std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> centers2;
        if (ivd2 == nullptr || ivd2 == Py_None) {
            centers2 = centers1;
        } else {
            if (_parse_1d_intervals_to_centers(ivd2, centers2) < 0)
                return nullptr;
        }

        // Parse expansion2 (None => reuse expansion1).
        PyObject *exp2_obj = PyDict_GetItemString(policy_dict, "expansion2");
        std::vector<int64_t> expansion2;
        if (exp2_obj == nullptr || exp2_obj == Py_None) {
            expansion2 = expansion1;
        } else {
            if (!PyArray_Check(exp2_obj)) {
                PyErr_SetString(PyExc_RuntimeError, "CartesianGrid policy: expansion2 must be a numpy array or None");
                return nullptr;
            }
            PyArrayObject *exp2_arr = (PyArrayObject *)exp2_obj;
            npy_intp ne2 = PyArray_DIM(exp2_arr, 0);
            expansion2.reserve(static_cast<size_t>(ne2));
            for (npy_intp i = 0; i < ne2; ++i)
                expansion2.push_back(*(int64_t *)PyArray_GETPTR1(exp2_arr, i));
            std::sort(expansion2.begin(), expansion2.end());
            expansion2.erase(std::unique(expansion2.begin(), expansion2.end()), expansion2.end());
            if (expansion2.size() < 2) {
                PyErr_SetString(PyExc_RuntimeError, "CartesianGrid policy: expansion2 must contain at least 2 unique values");
                return nullptr;
            }
        }

        // Parse band_idx: min_band_idx and max_band_idx (both or neither).
        PyObject *min_bi_obj = PyDict_GetItemString(policy_dict, "min_band_idx");
        PyObject *max_bi_obj = PyDict_GetItemString(policy_dict, "max_band_idx");
        bool use_band_idx = false;
        int32_t min_band_idx = 0, max_band_idx = 0;
        bool min_bi_set = (min_bi_obj != nullptr && min_bi_obj != Py_None);
        bool max_bi_set = (max_bi_obj != nullptr && max_bi_obj != Py_None);
        if (min_bi_set || max_bi_set) {
            if (!min_bi_set || !max_bi_set) {
                PyErr_SetString(PyExc_RuntimeError,
                                "CartesianGrid policy: both min_band_idx and max_band_idx must be provided");
                return nullptr;
            }
            // R semantics: band_idx only valid when intervals2 is None.
            if (ivd2 != nullptr && ivd2 != Py_None) {
                PyErr_SetString(PyExc_RuntimeError,
                                "CartesianGrid policy: band_idx filter requires intervals2=None");
                return nullptr;
            }
            use_band_idx = true;
            min_band_idx = static_cast<int32_t>(PyLong_AsLong(min_bi_obj));
            max_band_idx = static_cast<int32_t>(PyLong_AsLong(max_bi_obj));
            if (PyErr_Occurred()) return nullptr;
        }

        // Build corrected grid points.
        auto gp_axis0 = PMTrackExpressionCartesianGridIterator::build_grid_points(
            centers1, expansion1, chromkey);
        auto gp_axis1 = PMTrackExpressionCartesianGridIterator::build_grid_points(
            centers2, expansion2, chromkey);

        policy.kind             = PMTrackExpression2DIteratorPolicy::K_CARTESIAN_GRID;
        policy.chromkey         = &chromkey;
        policy.grid_points_axis0 = std::move(gp_axis0);
        policy.grid_points_axis1 = std::move(gp_axis1);
        policy.expansion0       = expansion1;
        policy.expansion1       = expansion2;
        policy.use_band_idx     = use_band_idx;
        policy.min_band_idx     = min_band_idx;
        policy.max_band_idx     = max_band_idx;
    } else if (kind == "intervals") {
        // No additional fields needed — the scope dict IS the iteration source.
        policy.kind = PMTrackExpression2DIteratorPolicy::K_INTERVALS;
    } else {
        PyErr_SetString(PyExc_RuntimeError,
                        ("Unknown iterator kind: " + kind).c_str());
        return nullptr;
    }

    // Parse scope (uses the static helper added in T6).
    std::vector<GInterval2D> scope;
    if (_parse_scope_dict(intervals_dict, scope) < 0) return nullptr;

    // Parse vars list.
    // Accepted element formats:
    //   (track: str, func: str)                                          -- no shifts
    //   (track: str, func: str, ss1: int, es1: int, ss2: int, es2: int) -- with shifts
    if (!PyList_Check(vars_list)) {
        PyErr_SetString(PyExc_TypeError, "vars_list must be a list");
        return nullptr;
    }
    std::vector<PMTrackExpr2DScanner::VarSpec> vars;
    Py_ssize_t nv = PyList_Size(vars_list);
    for (Py_ssize_t i = 0; i < nv; ++i) {
        PyObject *t = PyList_GetItem(vars_list, i);
        if (!t || !PyTuple_Check(t)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "vars_list element must be a tuple");
            return nullptr;
        }
        Py_ssize_t tsz = PyTuple_Size(t);
        if (tsz != 2 && tsz != 6) {
            PyErr_SetString(PyExc_RuntimeError,
                            "vars_list element must be a 2-tuple (track, func) "
                            "or 6-tuple (track, func, ss1, es1, ss2, es2)");
            return nullptr;
        }
        PyObject *tn = PyTuple_GetItem(t, 0);
        PyObject *fn = PyTuple_GetItem(t, 1);
        if (!PyUnicode_Check(tn) || !PyUnicode_Check(fn)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "vars_list (track, func) entries must be strings");
            return nullptr;
        }
        PMTrackExpr2DScanner::VarSpec vs;
        vs.name = PyUnicode_AsUTF8(tn);
        vs.func = PyUnicode_AsUTF8(fn);
        if (tsz == 6) {
            vs.sshift1 = PyLong_AsLongLong(PyTuple_GetItem(t, 2));
            vs.eshift1 = PyLong_AsLongLong(PyTuple_GetItem(t, 3));
            vs.sshift2 = PyLong_AsLongLong(PyTuple_GetItem(t, 4));
            vs.eshift2 = PyLong_AsLongLong(PyTuple_GetItem(t, 5));
            if (PyErr_Occurred()) return nullptr;
        }
        vars.push_back(std::move(vs));
    }

    // Validate colnames length.
    if (!PyList_Check(colnames_list)) {
        PyErr_SetString(PyExc_TypeError, "colnames_list must be a list");
        return nullptr;
    }
    Py_ssize_t nc = PyList_Size(colnames_list);
    if (static_cast<size_t>(nc) != vars.size()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "colnames length does not match vars length");
        return nullptr;
    }

    // Guard against user colnames colliding with the internal coord keys:
    static const char *RESERVED_COLNAMES[] = {
        "_chrom1", "_start1", "_end1", "_chrom2", "_start2", "_end2"
    };
    for (Py_ssize_t i = 0; i < nc; ++i) {
        PyObject *name = PyList_GetItem(colnames_list, i);
        if (!PyUnicode_Check(name)) {
            PyErr_SetString(PyExc_RuntimeError, "colnames entries must be strings");
            return nullptr;
        }
        const char *name_cstr = PyUnicode_AsUTF8(name);
        for (const char *reserved : RESERVED_COLNAMES) {
            if (std::strcmp(name_cstr, reserved) == 0) {
                PyErr_SetString(PyExc_RuntimeError,
                                (std::string("colname '") + reserved +
                                 "' collides with an internal coord key; choose a different name")
                                    .c_str());
                return nullptr;
            }
        }
    }

    // Parse band.
    std::unique_ptr<quadtree::DiagonalBand> band;
    if (band_obj != Py_None) {
        long long d1, d2;
        if (!PyArg_ParseTuple(band_obj, "LL", &d1, &d2))
            return nullptr;
        band = std::make_unique<quadtree::DiagonalBand>(
            static_cast<int64_t>(d1), static_cast<int64_t>(d2));
    }

    try {
        PMTrackExpr2DScanner scanner;
        scanner.run(policy, vars, scope, band.get());

        size_t   n      = scanner.num_emitted();
        PyObject *result = PyDict_New();
        if (!result) return nullptr;

        // colname -> values arrays.
        for (Py_ssize_t i = 0; i < nc; ++i) {
            PyObject *name = PyList_GetItem(colnames_list, i);
            npy_intp dims[1] = {static_cast<npy_intp>(n)};
            PyObject *arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if (!arr) {
                Py_DECREF(result);
                return nullptr;
            }
            if (n > 0) {
                const double *src =
                    scanner.values_for_var(static_cast<unsigned>(i));
                // values_for_var returns nullptr only when i >= num_vars(), which we've
                // already validated cannot happen here. Guard is defense-in-depth.
                if (src) {
                    std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject *>(arr)),
                                src, n * sizeof(double));
                }
            }
            PyDict_SetItem(result, name, arr);
            Py_DECREF(arr);
        }

        // Coord arrays: _chrom1/_start1/_end1/_chrom2/_start2/_end2.
        const std::vector<GInterval2D> &emitted = scanner.emitted_intervals();
        npy_intp ed[1] = {static_cast<npy_intp>(n)};
        PyObject *c1 = PyArray_SimpleNew(1, ed, NPY_INT32);
        PyObject *s1 = PyArray_SimpleNew(1, ed, NPY_INT64);
        PyObject *e1 = PyArray_SimpleNew(1, ed, NPY_INT64);
        PyObject *c2 = PyArray_SimpleNew(1, ed, NPY_INT32);
        PyObject *s2 = PyArray_SimpleNew(1, ed, NPY_INT64);
        PyObject *e2 = PyArray_SimpleNew(1, ed, NPY_INT64);
        if (!c1 || !s1 || !e1 || !c2 || !s2 || !e2) {
            Py_XDECREF(c1); Py_XDECREF(s1); Py_XDECREF(e1);
            Py_XDECREF(c2); Py_XDECREF(s2); Py_XDECREF(e2);
            Py_DECREF(result);
            return nullptr;
        }
        for (size_t k = 0; k < n; ++k) {
            *((int32_t *)PyArray_GETPTR1((PyArrayObject *)c1, k)) =
                static_cast<int32_t>(emitted[k].chromid1);
            *((int64_t *)PyArray_GETPTR1((PyArrayObject *)s1, k)) = emitted[k].start1;
            *((int64_t *)PyArray_GETPTR1((PyArrayObject *)e1, k)) = emitted[k].end1;
            *((int32_t *)PyArray_GETPTR1((PyArrayObject *)c2, k)) =
                static_cast<int32_t>(emitted[k].chromid2);
            *((int64_t *)PyArray_GETPTR1((PyArrayObject *)s2, k)) = emitted[k].start2;
            *((int64_t *)PyArray_GETPTR1((PyArrayObject *)e2, k)) = emitted[k].end2;
        }
        PyDict_SetItemString(result, "_chrom1", c1); Py_DECREF(c1);
        PyDict_SetItemString(result, "_start1", s1); Py_DECREF(s1);
        PyDict_SetItemString(result, "_end1",   e1); Py_DECREF(e1);
        PyDict_SetItemString(result, "_chrom2", c2); Py_DECREF(c2);
        PyDict_SetItemString(result, "_start2", s2); Py_DECREF(s2);
        PyDict_SetItemString(result, "_end2",   e2); Py_DECREF(e2);

        return result;
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

//---------------------------------------------------------------------------
// pm_extract_2d
//
// Native object-enumeration path for `gextract(track, intervals)` on a 2D
// RECTS / POINTS track. Replaces the per-interval Python loop in
// pymisha.extract._gextract_2d_single.
//
// Args:
//   track_name : str           - physical 2D track name
//   intervals  : dict          - chrom1 / start1 / end1 / chrom2 / start2 / end2
//                                arrays (chromids as int32, coords as int64)
//   band       : tuple|None    - (d1, d2) diagonal-band filter, or None
//
// Returns:
//   dict-of-arrays with keys "chrom1" (int32), "start1" (int64), "end1" (int64),
//   "chrom2" (int32), "start2" (int64), "end2" (int64), "value" (float64),
//   "intervalID" (int64). Or Py_None if no rows.
//
// Python caller maps chromids back to names and assembles the DataFrame.
//---------------------------------------------------------------------------

PyObject *pm_extract_2d(PyObject *self, PyObject *args)
{
    (void)self;
    const char *track_name = NULL;
    PyObject   *intervals_dict = NULL;
    PyObject   *band_obj = Py_None;

    if (!PyArg_ParseTuple(args, "sO!O",
                          &track_name,
                          &PyDict_Type, &intervals_dict,
                          &band_obj)) {
        return NULL;
    }

    // Parse band.
    bool has_band = false;
    long long band_d1 = 0, band_d2 = 0;
    if (band_obj && band_obj != Py_None) {
        PyObject *seq = PySequence_Fast(band_obj, "band must be a 2-tuple or None");
        if (!seq) return NULL;
        if (PySequence_Fast_GET_SIZE(seq) != 2) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "band must have length 2");
            return NULL;
        }
        band_d1 = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, 0));
        band_d2 = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, 1));
        Py_DECREF(seq);
        if (PyErr_Occurred()) return NULL;
        if (band_d1 >= band_d2) {
            PyErr_SetString(PyExc_ValueError, "band d1 must be < d2");
            return NULL;
        }
        has_band = true;
    }

    // Pull intervals arrays.
    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(intervals_dict, "chrom1", &arr_c1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start1", &arr_s1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end1",   &arr_e1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "chrom2", &arr_c2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start2", &arr_s2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end2",   &arr_e2) < 0) return NULL;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "intervals arrays must all be the same length");
        return NULL;
    }

    // Group intervals by (chromid1, chromid2). For each group store the
    // original buffer index so we can stamp intervalID correctly later.
    struct PairKey {
        int c1, c2;
        bool operator==(const PairKey &o) const { return c1 == o.c1 && c2 == o.c2; }
    };
    struct PairKeyHash {
        size_t operator()(const PairKey &k) const noexcept {
            return (static_cast<size_t>(static_cast<uint32_t>(k.c1)) << 32) |
                   static_cast<uint32_t>(k.c2);
        }
    };

    std::unordered_map<PairKey, std::vector<int64_t>, PairKeyHash> by_pair;
    by_pair.reserve(32);
    for (npy_intp i = 0; i < n; ++i) {
        PairKey key{
            static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c1, i)),
            static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c2, i)),
        };
        by_pair[key].push_back(static_cast<int64_t>(i));
    }

    // Output accumulators.
    std::vector<int32_t> out_c1, out_c2;
    std::vector<int64_t> out_s1, out_e1, out_s2, out_e2;
    std::vector<double>  out_val;
    std::vector<int64_t> out_id;

    quadtree::DiagonalBand band_real;
    const quadtree::DiagonalBand *band_ptr = nullptr;
    if (has_band) {
        band_real = quadtree::DiagonalBand(band_d1, band_d2);
        band_ptr = &band_real;
    }

    try {
        PMGenomeTrack2D track;
        track.init(track_name);  // throws on bad / non-2D / non-RECTS-POINTS

        for (auto &kv : by_pair) {
            const PairKey &key = kv.first;
            const std::vector<int64_t> &idxs = kv.second;

            bool have_data = track.set_chrom_pair(key.c1, key.c2);
            if (!have_data) continue;

            const bool is_points = track.is_points();

            for (int64_t orig_idx : idxs) {
                int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, orig_idx);
                int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, orig_idx);
                int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, orig_idx);
                int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, orig_idx);

                quadtree::QueryObjects qresult =
                    track.query_objects(s1, s2, e1, e2, band_ptr);

                size_t m = qresult.ids.size();
                for (size_t j = 0; j < m; ++j) {
                    out_c1.push_back(static_cast<int32_t>(key.c1));
                    out_c2.push_back(static_cast<int32_t>(key.c2));
                    if (is_points) {
                        // POINTS: x1==x2-1==x, y1==y2-1==y (the Python path
                        // returns end = x+1 and end = y+1).
                        out_s1.push_back(qresult.x1s[j]);
                        out_e1.push_back(qresult.x1s[j] + 1);
                        out_s2.push_back(qresult.y1s[j]);
                        out_e2.push_back(qresult.y1s[j] + 1);
                    } else {
                        out_s1.push_back(qresult.x1s[j]);
                        out_e1.push_back(qresult.x2s[j]);
                        out_s2.push_back(qresult.y1s[j]);
                        out_e2.push_back(qresult.y2s[j]);
                    }
                    out_val.push_back(static_cast<double>(qresult.vals[j]));
                    out_id.push_back(orig_idx);
                }
            }
        }
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    if (out_c1.empty()) {
        Py_RETURN_NONE;
    }

    npy_intp out_n = static_cast<npy_intp>(out_c1.size());
    npy_intp dims[1] = {out_n};

    PyObject *result = PyDict_New();
    if (!result) return NULL;

    auto set_int32 = [&](const char *name, const std::vector<int32_t> &v) {
        PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT32);
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(), out_n * sizeof(int32_t));
        PyDict_SetItemString(result, name, a);
        Py_DECREF(a);
    };
    auto set_int64 = [&](const char *name, const std::vector<int64_t> &v) {
        PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT64);
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(), out_n * sizeof(int64_t));
        PyDict_SetItemString(result, name, a);
        Py_DECREF(a);
    };
    auto set_f64 = [&](const char *name, const std::vector<double> &v) {
        PyObject *a = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
        std::memcpy(PyArray_DATA((PyArrayObject *)a), v.data(), out_n * sizeof(double));
        PyDict_SetItemString(result, name, a);
        Py_DECREF(a);
    };

    set_int32("chrom1", out_c1);
    set_int64("start1", out_s1);
    set_int64("end1",   out_e1);
    set_int32("chrom2", out_c2);
    set_int64("start2", out_s2);
    set_int64("end2",   out_e2);
    set_f64  ("value",  out_val);
    set_int64("intervalID", out_id);

    return result;
}

//---------------------------------------------------------------------------
// pm_extract_2d_objects
//
// Native object-level reduction path for the 2D vtrack object funcs
// (exists / size / first / last / sample). One row per input interval.
// Replaces the per-interval Python loop in
// pymisha.extract._gextract_2d_vtrack_objects.
//
// Args:
//   track_name : str           - physical 2D track name (RECTS / POINTS)
//   intervals  : dict          - chrom1 / start1 / end1 / chrom2 / start2 / end2
//                                arrays (chromids as int32, coords as int64)
//   band       : tuple|None    - (d1, d2) diagonal-band filter, or None
//   func       : str           - "exists" | "size" | "first" | "last" | "sample"
//   seed       : int           - RNG seed for "sample" (ignored otherwise)
//
// Returns:
//   dict {"value": np.ndarray[n] of float64}
//
// Defaults when no object intersects an interval:
//   exists / size : 0.0
//   first / last / sample : NaN
//---------------------------------------------------------------------------

PyObject *pm_extract_2d_objects(PyObject *self, PyObject *args)
{
    (void)self;
    const char *track_name = NULL;
    PyObject   *intervals_dict = NULL;
    PyObject   *band_obj = Py_None;
    const char *func_str = NULL;
    long long   seed = 60427;

    if (!PyArg_ParseTuple(args, "sO!OsL",
                          &track_name,
                          &PyDict_Type, &intervals_dict,
                          &band_obj,
                          &func_str,
                          &seed)) {
        return NULL;
    }

    // Parse func.
    enum ObjFunc { F_EXISTS, F_SIZE, F_FIRST, F_LAST, F_SAMPLE };
    ObjFunc func;
    if      (strcmp(func_str, "exists") == 0) func = F_EXISTS;
    else if (strcmp(func_str, "size")   == 0) func = F_SIZE;
    else if (strcmp(func_str, "first")  == 0) func = F_FIRST;
    else if (strcmp(func_str, "last")   == 0) func = F_LAST;
    else if (strcmp(func_str, "sample") == 0) func = F_SAMPLE;
    else {
        PyErr_Format(PyExc_ValueError,
                     "pm_extract_2d_objects: unknown func '%s' (expected one of "
                     "exists/size/first/last/sample)", func_str);
        return NULL;
    }

    // Parse band.
    bool has_band = false;
    long long band_d1 = 0, band_d2 = 0;
    if (band_obj && band_obj != Py_None) {
        PyObject *seq = PySequence_Fast(band_obj, "band must be a 2-tuple or None");
        if (!seq) return NULL;
        if (PySequence_Fast_GET_SIZE(seq) != 2) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "band must have length 2");
            return NULL;
        }
        band_d1 = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, 0));
        band_d2 = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, 1));
        Py_DECREF(seq);
        if (PyErr_Occurred()) return NULL;
        if (band_d1 >= band_d2) {
            PyErr_SetString(PyExc_ValueError, "band d1 must be < d2");
            return NULL;
        }
        has_band = true;
    }

    // Pull intervals arrays.
    PyArrayObject *arr_c1, *arr_s1, *arr_e1, *arr_c2, *arr_s2, *arr_e2;
    if (_pm_test_2d_get_array(intervals_dict, "chrom1", &arr_c1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start1", &arr_s1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end1",   &arr_e1) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "chrom2", &arr_c2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "start2", &arr_s2) < 0) return NULL;
    if (_pm_test_2d_get_array(intervals_dict, "end2",   &arr_e2) < 0) return NULL;

    npy_intp n = PyArray_DIM(arr_c1, 0);
    if (PyArray_DIM(arr_s1, 0) != n || PyArray_DIM(arr_e1, 0) != n ||
        PyArray_DIM(arr_c2, 0) != n || PyArray_DIM(arr_s2, 0) != n ||
        PyArray_DIM(arr_e2, 0) != n) {
        PyErr_SetString(PyExc_ValueError,
                        "pm_extract_2d_objects: intervals arrays must all be the same length");
        return NULL;
    }

    // Default values per func (exists/size -> 0, first/last/sample -> NaN).
    double default_val = (func == F_EXISTS || func == F_SIZE)
        ? 0.0
        : std::numeric_limits<double>::quiet_NaN();

    // Output values, length n, initialized to default.
    npy_intp dims[1] = {n};
    PyObject *out_arr = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!out_arr) return NULL;
    double *out = (double *)PyArray_DATA((PyArrayObject *)out_arr);
    for (npy_intp i = 0; i < n; ++i) out[i] = default_val;

    // Empty input -> return empty result.
    if (n == 0) {
        PyObject *result = PyDict_New();
        if (!result) { Py_DECREF(out_arr); return NULL; }
        PyDict_SetItemString(result, "value", out_arr);
        Py_DECREF(out_arr);
        return result;
    }

    // Group intervals by (chromid1, chromid2).
    struct PairKey2 {
        int c1, c2;
        bool operator==(const PairKey2 &o) const { return c1 == o.c1 && c2 == o.c2; }
    };
    struct PairKey2Hash {
        size_t operator()(const PairKey2 &k) const noexcept {
            return (static_cast<size_t>(static_cast<uint32_t>(k.c1)) << 32) |
                   static_cast<uint32_t>(k.c2);
        }
    };

    std::unordered_map<PairKey2, std::vector<int64_t>, PairKey2Hash> by_pair;
    by_pair.reserve(32);
    for (npy_intp i = 0; i < n; ++i) {
        PairKey2 key{
            static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c1, i)),
            static_cast<int>(*(int32_t *)PyArray_GETPTR1(arr_c2, i)),
        };
        by_pair[key].push_back(static_cast<int64_t>(i));
    }

    quadtree::DiagonalBand band_real;
    const quadtree::DiagonalBand *band_ptr = nullptr;
    if (has_band) {
        band_real = quadtree::DiagonalBand(band_d1, band_d2);
        band_ptr = &band_real;
    }

    // Seeded RNG for sample.
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    try {
        PMGenomeTrack2D track;
        track.init(track_name);

        for (auto &kv : by_pair) {
            const PairKey2 &key = kv.first;
            const std::vector<int64_t> &idxs = kv.second;

            bool have_data = track.set_chrom_pair(key.c1, key.c2);
            if (!have_data) continue;

            for (int64_t orig_idx : idxs) {
                int64_t s1 = *(int64_t *)PyArray_GETPTR1(arr_s1, orig_idx);
                int64_t e1 = *(int64_t *)PyArray_GETPTR1(arr_e1, orig_idx);
                int64_t s2 = *(int64_t *)PyArray_GETPTR1(arr_s2, orig_idx);
                int64_t e2 = *(int64_t *)PyArray_GETPTR1(arr_e2, orig_idx);

                quadtree::QueryObjects qresult =
                    track.query_objects(s1, s2, e1, e2, band_ptr);

                size_t m = qresult.ids.size();

                switch (func) {
                case F_EXISTS:
                    out[orig_idx] = m > 0 ? 1.0 : 0.0;
                    break;
                case F_SIZE:
                    out[orig_idx] = static_cast<double>(m);
                    break;
                case F_FIRST:
                    if (m > 0) {
                        out[orig_idx] = static_cast<double>(qresult.vals[0]);
                    }
                    break;
                case F_LAST:
                    if (m > 0) {
                        out[orig_idx] = static_cast<double>(qresult.vals[m - 1]);
                    }
                    break;
                case F_SAMPLE:
                    if (m > 0) {
                        std::uniform_int_distribution<size_t> dist(0, m - 1);
                        size_t pick = dist(rng);
                        out[orig_idx] = static_cast<double>(qresult.vals[pick]);
                    }
                    break;
                }
            }
        }
    } catch (const TGLException &e) {
        Py_DECREF(out_arr);
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return NULL;
    } catch (const std::exception &e) {
        Py_DECREF(out_arr);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    PyObject *result = PyDict_New();
    if (!result) { Py_DECREF(out_arr); return NULL; }
    PyDict_SetItemString(result, "value", out_arr);
    Py_DECREF(out_arr);
    return result;
}

//---------------------------------------------------------------------------
// pm_test_track_rects_iterator
//
// Test-only binding that constructs a PMTrackExpressionTrackRectsIterator
// and collects all emitted (track_object x scope_rect) intersections.
//
// Args:
//   track_name   : str  - 2D RECTS or POINTS track name
//   intervals    : dict of 6 numpy arrays (chrom1/start1/end1/chrom2/start2/end2)
//   band         : (int, int) | None
//
// Returns: dict with keys chrom1/start1/end1/chrom2/start2/end2 (numpy arrays).
//---------------------------------------------------------------------------

PyObject *pm_test_track_rects_iterator(PyObject *self, PyObject *args)
{
    (void)self;
    const char *track_name = nullptr;
    PyObject   *intervals_dict = nullptr;
    PyObject   *band_obj = Py_None;

    if (!PyArg_ParseTuple(args, "sOO", &track_name, &intervals_dict, &band_obj))
        return nullptr;

    // Parse scope dict.
    std::vector<GInterval2D> scope;
    if (_parse_scope_dict(intervals_dict, scope) < 0) return nullptr;

    // Parse optional band.
    std::unique_ptr<quadtree::DiagonalBand> band_ptr;
    if (band_obj != Py_None) {
        long long d1, d2;
        if (!PyArg_ParseTuple(band_obj, "LL", &d1, &d2))
            return nullptr;
        band_ptr = std::make_unique<quadtree::DiagonalBand>(
            static_cast<int64_t>(d1), static_cast<int64_t>(d2));
    }

    std::vector<int>     out_c1, out_c2;
    std::vector<int64_t> out_s1, out_e1, out_s2, out_e2;

    try {
        if (!g_pmdb)
            throw std::runtime_error("Database not initialized");

        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        PMTrackExpressionTrackRectsIterator it(
            track_name,
            std::move(scope),
            chromkey,
            band_ptr.get());
        it.begin();

        while (!it.isend()) {
            const GInterval2D &iv = it.last_interval();
            out_c1.push_back(iv.chromid1);
            out_c2.push_back(iv.chromid2);
            out_s1.push_back(iv.start1);
            out_e1.push_back(iv.end1);
            out_s2.push_back(iv.start2);
            out_e2.push_back(iv.end2);
            it.next();
        }
    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    npy_intp m = static_cast<npy_intp>(out_c1.size());
    npy_intp dims[1] = {m};

    PyObject *result = PyDict_New();
    if (!result) return nullptr;

    #define SET_INT32_TRI(name, vec)                                          \
        do {                                                                  \
            PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT32);              \
            if (!a) { Py_DECREF(result); return nullptr; }                    \
            if (m > 0) {                                                      \
                int32_t *ptr = (int32_t *)PyArray_DATA((PyArrayObject *)a);   \
                for (npy_intp i = 0; i < m; ++i) ptr[i] = (vec)[i];          \
            }                                                                 \
            PyDict_SetItemString(result, name, a);                            \
            Py_DECREF(a);                                                     \
        } while (0)

    #define SET_INT64_TRI(name, vec)                                          \
        do {                                                                  \
            PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT64);              \
            if (!a) { Py_DECREF(result); return nullptr; }                    \
            if (m > 0)                                                        \
                memcpy(PyArray_DATA((PyArrayObject *)a),                      \
                       (vec).data(), m * sizeof(int64_t));                    \
            PyDict_SetItemString(result, name, a);                            \
            Py_DECREF(a);                                                     \
        } while (0)

    SET_INT32_TRI("chrom1", out_c1);
    SET_INT64_TRI("start1", out_s1);
    SET_INT64_TRI("end1",   out_e1);
    SET_INT32_TRI("chrom2", out_c2);
    SET_INT64_TRI("start2", out_s2);
    SET_INT64_TRI("end2",   out_e2);

    #undef SET_INT32_TRI
    #undef SET_INT64_TRI

    return result;
}

//---------------------------------------------------------------------------
// pm_test_cartesian_grid_iterator
//
// Test-only binding: construct a CartesianGridIterator and return all emitted
// 2D intervals as a 6-array dict.
//
// Args:
//   intervals1_dict  : dict {chrom: int32[N], start: int64[N], end: int64[N]}
//   expansion1       : int64 numpy array (at least 2 unique values)
//   intervals2_dict  : dict or None  (None => reuse intervals1)
//   expansion2       : int64 numpy array or None  (None => reuse expansion1)
//   band_idx         : (min_idx, max_idx) tuple of ints or None
//   scope_dict       : 6-array 2D-intervals dict (chrom1/start1/end1/chrom2/start2/end2)
//   band             : (d1, d2) tuple or None
//
// Returns: dict with keys chrom1/start1/end1/chrom2/start2/end2.
//---------------------------------------------------------------------------

// Helper: parse a 1D intervals dict {chrom: int32[], start: int64[], end: int64[]}
// into a vector of GridPoints (coord = floor((start+end)/2)).
static int _parse_1d_intervals_to_centers(
    PyObject *d,
    std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> &out)
{
    PyObject *key_chrom = PyUnicode_FromString("chrom");
    PyObject *key_start = PyUnicode_FromString("start");
    PyObject *key_end   = PyUnicode_FromString("end");
    PyObject *arr_c = PyDict_GetItem(d, key_chrom);
    PyObject *arr_s = PyDict_GetItem(d, key_start);
    PyObject *arr_e = PyDict_GetItem(d, key_end);
    Py_DECREF(key_chrom); Py_DECREF(key_start); Py_DECREF(key_end);

    if (!arr_c || !arr_s || !arr_e) {
        PyErr_SetString(PyExc_KeyError, "intervals dict must have 'chrom', 'start', 'end' keys");
        return -1;
    }
    if (!PyArray_Check(arr_c) || !PyArray_Check(arr_s) || !PyArray_Check(arr_e)) {
        PyErr_SetString(PyExc_ValueError, "'chrom', 'start', 'end' must be numpy arrays");
        return -1;
    }

    PyArrayObject *ac = (PyArrayObject *)arr_c;
    PyArrayObject *as_ = (PyArrayObject *)arr_s;
    PyArrayObject *ae = (PyArrayObject *)arr_e;

    npy_intp n = PyArray_DIM(ac, 0);
    if (PyArray_DIM(as_, 0) != n || PyArray_DIM(ae, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "chrom/start/end arrays must have same length");
        return -1;
    }

    out.reserve(static_cast<size_t>(n));
    for (npy_intp i = 0; i < n; ++i) {
        int32_t chromid = *(int32_t *)PyArray_GETPTR1(ac, i);
        int64_t start   = *(int64_t *)PyArray_GETPTR1(as_, i);
        int64_t end_    = *(int64_t *)PyArray_GETPTR1(ae, i);
        int64_t coord   = (start + end_) / 2;
        out.emplace_back(chromid, coord);
    }
    return 0;
}

PyObject *pm_test_cartesian_grid_iterator(PyObject *self, PyObject *args)
{
    (void)self;
    PyObject *intervals1_dict = nullptr;
    PyObject *expansion1_obj  = nullptr;
    PyObject *intervals2_dict = nullptr;   // may be None
    PyObject *expansion2_obj  = nullptr;   // may be None
    PyObject *band_idx_obj    = nullptr;   // may be None
    PyObject *scope_dict      = nullptr;
    PyObject *band_obj        = nullptr;   // may be None

    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &intervals1_dict, &expansion1_obj,
                          &intervals2_dict, &expansion2_obj,
                          &band_idx_obj, &scope_dict, &band_obj))
        return nullptr;

    try {
        if (!g_pmdb)
            throw std::runtime_error("Database not initialized");
        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        // Parse intervals1.
        std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> centers1;
        if (_parse_1d_intervals_to_centers(intervals1_dict, centers1) < 0)
            return nullptr;

        // Parse expansion1.
        if (!PyArray_Check(expansion1_obj)) {
            PyErr_SetString(PyExc_ValueError, "expansion1 must be a numpy array");
            return nullptr;
        }
        PyArrayObject *exp1_arr = (PyArrayObject *)expansion1_obj;
        npy_intp ne1 = PyArray_DIM(exp1_arr, 0);
        std::vector<int64_t> expansion1;
        expansion1.reserve(static_cast<size_t>(ne1));
        for (npy_intp i = 0; i < ne1; ++i)
            expansion1.push_back(*(int64_t *)PyArray_GETPTR1(exp1_arr, i));
        std::sort(expansion1.begin(), expansion1.end());
        expansion1.erase(std::unique(expansion1.begin(), expansion1.end()), expansion1.end());
        if (expansion1.size() < 2) {
            PyErr_SetString(PyExc_RuntimeError, "expansion must contain at least 2 unique values");
            return nullptr;
        }

        // Parse intervals2 (None => reuse centers1).
        std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> centers2;
        if (intervals2_dict == Py_None) {
            centers2 = centers1;
        } else {
            if (_parse_1d_intervals_to_centers(intervals2_dict, centers2) < 0)
                return nullptr;
        }

        // Parse expansion2 (None => reuse expansion1).
        std::vector<int64_t> expansion2;
        if (expansion2_obj == Py_None) {
            expansion2 = expansion1;
        } else {
            if (!PyArray_Check(expansion2_obj)) {
                PyErr_SetString(PyExc_ValueError, "expansion2 must be a numpy array or None");
                return nullptr;
            }
            PyArrayObject *exp2_arr = (PyArrayObject *)expansion2_obj;
            npy_intp ne2 = PyArray_DIM(exp2_arr, 0);
            expansion2.reserve(static_cast<size_t>(ne2));
            for (npy_intp i = 0; i < ne2; ++i)
                expansion2.push_back(*(int64_t *)PyArray_GETPTR1(exp2_arr, i));
            std::sort(expansion2.begin(), expansion2.end());
            expansion2.erase(std::unique(expansion2.begin(), expansion2.end()), expansion2.end());
            if (expansion2.size() < 2) {
                PyErr_SetString(PyExc_RuntimeError, "expansion2 must contain at least 2 unique values");
                return nullptr;
            }
        }

        // Build corrected grid points.
        auto gp_axis0 = PMTrackExpressionCartesianGridIterator::build_grid_points(
            centers1, expansion1, chromkey);
        auto gp_axis1 = PMTrackExpressionCartesianGridIterator::build_grid_points(
            centers2, expansion2, chromkey);

        // Parse band_idx (None or (min, max) tuple).
        bool    use_band_idx = false;
        int32_t min_band_idx = 0, max_band_idx = 0;
        if (band_idx_obj != Py_None) {
            long long bmin, bmax;
            if (!PyArg_ParseTuple(band_idx_obj, "LL", &bmin, &bmax))
                return nullptr;
            use_band_idx = true;
            min_band_idx = static_cast<int32_t>(bmin);
            max_band_idx = static_cast<int32_t>(bmax);
        }

        // Parse scope dict.
        std::vector<GInterval2D> scope;
        if (_parse_scope_dict(scope_dict, scope) < 0)
            return nullptr;

        // Parse band (None or (d1, d2) tuple).
        std::unique_ptr<quadtree::DiagonalBand> band_ptr;
        if (band_obj != Py_None) {
            long long d1, d2;
            if (!PyArg_ParseTuple(band_obj, "LL", &d1, &d2))
                return nullptr;
            band_ptr = std::make_unique<quadtree::DiagonalBand>(
                static_cast<int64_t>(d1), static_cast<int64_t>(d2));
        }

        // Construct and walk the iterator.
        std::vector<int>     out_c1, out_c2;
        std::vector<int64_t> out_s1, out_e1, out_s2, out_e2;

        PMTrackExpressionCartesianGridIterator it(
            std::move(gp_axis0),
            std::move(gp_axis1),
            expansion1,
            expansion2,
            use_band_idx,
            min_band_idx,
            max_band_idx,
            std::move(scope),
            band_ptr.get());
        it.begin();

        while (!it.isend()) {
            const GInterval2D &iv = it.last_interval();
            out_c1.push_back(iv.chromid1);
            out_c2.push_back(iv.chromid2);
            out_s1.push_back(iv.start1);
            out_e1.push_back(iv.end1);
            out_s2.push_back(iv.start2);
            out_e2.push_back(iv.end2);
            it.next();
        }

        npy_intp m = static_cast<npy_intp>(out_c1.size());
        npy_intp dims[1] = {m};

        PyObject *result = PyDict_New();
        if (!result) return nullptr;

        #define SET_INT32_CGI(name, vec)                                          \
            do {                                                                  \
                PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT32);              \
                if (!a) { Py_DECREF(result); return nullptr; }                    \
                if (m > 0) {                                                      \
                    int32_t *ptr = (int32_t *)PyArray_DATA((PyArrayObject *)a);   \
                    for (npy_intp i = 0; i < m; ++i) ptr[i] = (vec)[i];          \
                }                                                                 \
                PyDict_SetItemString(result, name, a);                            \
                Py_DECREF(a);                                                     \
            } while (0)

        #define SET_INT64_CGI(name, vec)                                          \
            do {                                                                  \
                PyObject *a = PyArray_SimpleNew(1, dims, NPY_INT64);              \
                if (!a) { Py_DECREF(result); return nullptr; }                    \
                if (m > 0)                                                        \
                    memcpy(PyArray_DATA((PyArrayObject *)a),                      \
                           (vec).data(), m * sizeof(int64_t));                    \
                PyDict_SetItemString(result, name, a);                            \
                Py_DECREF(a);                                                     \
            } while (0)

        SET_INT32_CGI("chrom1", out_c1);
        SET_INT64_CGI("start1", out_s1);
        SET_INT64_CGI("end1",   out_e1);
        SET_INT32_CGI("chrom2", out_c2);
        SET_INT64_CGI("start2", out_s2);
        SET_INT64_CGI("end2",   out_e2);

        #undef SET_INT32_CGI
        #undef SET_INT64_CGI

        return result;

    } catch (const TGLException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.msg());
        return nullptr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// =====================================================================
// pm_neighbors_2d — 2D NN iteration on one chrom-pair using MemQuadTree
// (Phase NN of the KICKOFF-5 deferred backlog). The caller pre-groups
// i1 / i2 by chrom-pair and invokes this per pair with int64 rect
// arrays.  Returns a dict of 4 int64 ndarrays:
//
//   q_local_idx : index into the query arrays of the matched row
//   t_local_idx : index into the target arrays of the matched row
//   dist1, dist2: per-axis unsigned gaps
//
// Rows are appended in the order the NN iterator surfaces them (per
// query, ascending Manhattan distance).  The Python wrapper re-sorts
// and joins back to global indices.
// =====================================================================

static PyObject *_emit_neighbors_2d_result(
        const std::vector<int64_t> &q_idx,
        const std::vector<int64_t> &t_idx,
        const std::vector<int64_t> &d1,
        const std::vector<int64_t> &d2) {
    npy_intp n = static_cast<npy_intp>(q_idx.size());
    PyObject *result = PyDict_New();
    if (!result) return NULL;

    auto build = [&](const std::vector<int64_t> &v, const char *key) -> bool {
        PyObject *arr = PyArray_SimpleNew(1, &n, NPY_INT64);
        if (!arr) return false;
        if (n > 0) memcpy(PyArray_DATA((PyArrayObject *)arr),
                          v.data(), n * sizeof(int64_t));
        PyDict_SetItemString(result, key, arr);
        Py_DECREF(arr);
        return true;
    };

    if (!build(q_idx, "q_local_idx") ||
        !build(t_idx, "t_local_idx") ||
        !build(d1,    "dist1") ||
        !build(d2,    "dist2")) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

PyObject *pm_neighbors_2d(PyObject *self, PyObject *args) {
    PyArrayObject *qx1_a, *qy1_a, *qx2_a, *qy2_a;
    PyArrayObject *tx1_a, *ty1_a, *tx2_a, *ty2_a;
    long long maxneighbors;
    long long mindist1, maxdist1, mindist2, maxdist2;

    if (!PyArg_ParseTuple(args, "OOOOOOOOLLLLL",
                          &qx1_a, &qy1_a, &qx2_a, &qy2_a,
                          &tx1_a, &ty1_a, &tx2_a, &ty2_a,
                          &maxneighbors,
                          &mindist1, &maxdist1, &mindist2, &maxdist2))
        return NULL;

    auto as_i64 = [](PyArrayObject *a) -> const int64_t * {
        return static_cast<const int64_t *>(PyArray_DATA(a));
    };
    auto count = [](PyArrayObject *a) -> npy_intp { return PyArray_DIM(a, 0); };

    npy_intp nq = count(qx1_a);
    npy_intp nt = count(tx1_a);

    if (nq == 0 || nt == 0) {
        std::vector<int64_t> empty;
        return _emit_neighbors_2d_result(empty, empty, empty, empty);
    }

    const int64_t *qx1 = as_i64(qx1_a), *qy1 = as_i64(qy1_a),
                  *qx2 = as_i64(qx2_a), *qy2 = as_i64(qy2_a);
    const int64_t *tx1 = as_i64(tx1_a), *ty1 = as_i64(ty1_a),
                  *tx2 = as_i64(tx2_a), *ty2 = as_i64(ty2_a);

    // Arena: bounding box of all i1 and i2 rects, plus a margin so all
    // objects are strictly inside (R's quadtree treats arena as
    // half-open).  +1 in each direction keeps insertions inside.
    int64_t x_lo = qx1[0], y_lo = qy1[0], x_hi = qx2[0], y_hi = qy2[0];
    for (npy_intp i = 0; i < nq; ++i) {
        if (qx1[i] < x_lo) x_lo = qx1[i];
        if (qy1[i] < y_lo) y_lo = qy1[i];
        if (qx2[i] > x_hi) x_hi = qx2[i];
        if (qy2[i] > y_hi) y_hi = qy2[i];
    }
    for (npy_intp i = 0; i < nt; ++i) {
        if (tx1[i] < x_lo) x_lo = tx1[i];
        if (ty1[i] < y_lo) y_lo = ty1[i];
        if (tx2[i] > x_hi) x_hi = tx2[i];
        if (ty2[i] > y_hi) y_hi = ty2[i];
    }
    x_hi += 1; y_hi += 1;

    memqt::MemQuadTree tree;
    tree.init(x_lo, y_lo, x_hi, y_hi);
    for (npy_intp i = 0; i < nt; ++i) {
        tree.insert(memqt::RectObj(
            memqt::Rectangle(tx1[i], ty1[i], tx2[i], ty2[i]),
            static_cast<int64_t>(i)));
    }

    std::vector<int64_t> out_q, out_t, out_d1, out_d2;
    int64_t per_q = maxneighbors > 16 ? 16 : maxneighbors;
    if (per_q < 1) per_q = 1;
    out_q.reserve(nq * per_q);
    out_t.reserve(nq * per_q);
    out_d1.reserve(nq * per_q);
    out_d2.reserve(nq * per_q);

    // Manhattan cutoff: if both maxdists are finite, the iterator can
    // stop once dist1 + dist2 exceeds the sum.  We treat "unbounded" as
    // any maxdist >= 1e9 (the Python `_BIG` sentinel).
    constexpr int64_t BIG = 1000000000LL;
    int64_t cutoff = INT64_MAX;
    if (maxdist1 < BIG && maxdist2 < BIG)
        cutoff = maxdist1 + maxdist2;

    for (npy_intp j = 0; j < nq; ++j) {
        memqt::MemQuadTree::NNIterator it(&tree);
        memqt::Rectangle qr(qx1[j], qy1[j], qx2[j], qy2[j]);
        if (!it.begin(qr)) continue;
        int kept = 0;
        do {
            const memqt::RectObj &t = *it;
            int64_t d1 = 0;
            if (t.rect.x1 >= qr.x2)      d1 = t.rect.x1 - qr.x2;
            else if (qr.x1 >= t.rect.x2) d1 = qr.x1 - t.rect.x2;
            int64_t d2 = 0;
            if (t.rect.y1 >= qr.y2)      d2 = t.rect.y1 - qr.y2;
            else if (qr.y1 >= t.rect.y2) d2 = qr.y1 - t.rect.y2;
            int64_t M = d1 + d2;
            if (M > cutoff) break;
            if (d1 >= mindist1 && d1 <= maxdist1 &&
                d2 >= mindist2 && d2 <= maxdist2) {
                out_q.push_back(static_cast<int64_t>(j));
                out_t.push_back(t.id);
                out_d1.push_back(d1);
                out_d2.push_back(d2);
                if (++kept >= static_cast<int>(maxneighbors)) break;
            }
        } while (it.next());
    }

    return _emit_neighbors_2d_result(out_q, out_t, out_d1, out_d2);
}
