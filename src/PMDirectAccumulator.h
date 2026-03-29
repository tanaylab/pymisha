/*
 * PMDirectAccumulator.h
 *
 * Pre-allocated NumPy array accumulator for gextract results.
 * Writes scan results directly into NumPy arrays, avoiding intermediate
 * std::vector storage and the subsequent copy into PMDataFrame.
 *
 * Usage:
 *   PMDirectAccumulator acc;
 *   acc.init(estimated_rows, num_expr_cols, colnames);
 *   // ... scan loop ...
 *   acc.write_row(chromid, start, end, values_ptr, interval_id);
 *   // ... end of scan ...
 *   return acc.finalize();   // returns PMPY list in pymisha internal format
 */

#ifndef PMDIRECTACCUMULATOR_H_INCLUDED
#define PMDIRECTACCUMULATOR_H_INCLUDED

#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "pymisha.h"
#include "PMObject.h"
#include "PMDb.h"
#include "TGLException.h"

class PMDirectAccumulator {
public:
    PMDirectAccumulator() = default;
    ~PMDirectAccumulator() = default;

    // Initialize with estimated row count and expression column names.
    // Columns: chrom (STR), start (LONG), end (LONG), [expr cols (DOUBLE)...], intervalID (LONG)
    void init(size_t estimated_rows, const std::vector<std::string> &expr_colnames)
    {
        m_num_expr_cols = expr_colnames.size();
        m_expr_colnames = expr_colnames;
        m_row = 0;
        m_alloc_size = estimated_rows > 0 ? estimated_rows : 1024;

        alloc_arrays(m_alloc_size);
    }

    // Write one row of data directly into the pre-allocated arrays.
    // values must point to m_num_expr_cols doubles.
    inline void write_row(int chromid, int64_t start, int64_t end,
                          const double *values, uint64_t interval_id)
    {
        if (m_row >= m_alloc_size) {
            grow();
        }

        // Chrom: store chromid as int32_t, we'll convert to strings in finalize()
        m_p_chromids[m_row] = static_cast<int32_t>(chromid);

        // Start, End
        m_p_starts[m_row] = start;
        m_p_ends[m_row] = end;

        // Expression values
        for (size_t i = 0; i < m_num_expr_cols; ++i) {
            m_p_values[i][m_row] = values[i];
        }

        // IntervalID
        m_p_interval_ids[m_row] = static_cast<long>(interval_id);

        ++m_row;
    }

    // Number of rows accumulated so far
    size_t size() const { return m_row; }

    // Finalize: truncate arrays to actual size, build chrom strings,
    // return PMPY in pymisha internal format (list of [colnames, col1, col2, ...]).
    // Returns Py_None if empty and none_if_empty is true.
    PMPY finalize(bool none_if_empty = true)
    {
        if (m_row == 0 && none_if_empty) {
            return PMPY(Py_None, true);
        }

        // Truncate arrays if over-allocated
        if (m_row < m_alloc_size) {
            truncate_arrays();
        }

        // Build chrom string column from chromid integers
        PMPY py_chrom_col = build_chrom_column();

        // Build the result list: [colnames, chrom, start, end, expr0, ..., exprN, intervalID]
        size_t total_cols = 3 + m_num_expr_cols + 1;
        PMPY py_answer(PyList_New(total_cols + 1), true);

        // Column names
        build_and_set_colnames(py_answer);

        // chrom (index 1)
        py_chrom_col.to_be_stolen();
        PyList_SetItem(py_answer, 1, py_chrom_col);

        // start (index 2)
        m_py_starts.to_be_stolen();
        PyList_SetItem(py_answer, 2, m_py_starts);

        // end (index 3)
        m_py_ends.to_be_stolen();
        PyList_SetItem(py_answer, 3, m_py_ends);

        // expression value columns (indices 4 .. 3+num_expr_cols)
        for (size_t i = 0; i < m_num_expr_cols; ++i) {
            m_py_values[i].to_be_stolen();
            PyList_SetItem(py_answer, 4 + i, m_py_values[i]);
        }

        // intervalID (last column)
        m_py_interval_ids.to_be_stolen();
        PyList_SetItem(py_answer, 4 + m_num_expr_cols, m_py_interval_ids);

        return py_answer;
    }

    // Access the raw data pointers (needed for sort_extract_result equivalent)
    size_t num_rows() const { return m_row; }
    size_t num_expr_cols() const { return m_num_expr_cols; }
    int32_t *chromids() { return m_p_chromids; }
    int64_t *starts() { return m_p_starts; }
    int64_t *ends() { return m_p_ends; }
    double  *values(size_t expr_idx) { return m_p_values[expr_idx]; }
    long    *interval_ids() { return m_p_interval_ids; }

private:
    size_t m_num_expr_cols{0};
    std::vector<std::string> m_expr_colnames;
    size_t m_row{0};
    size_t m_alloc_size{0};

    // Chromid storage: int32_t array (converted to strings in finalize())
    PMPY m_py_chromids;
    int32_t *m_p_chromids{nullptr};

    // Start/End: int64_t arrays (NPY_LONG on 64-bit Linux)
    PMPY m_py_starts;
    PMPY m_py_ends;
    int64_t *m_p_starts{nullptr};
    int64_t *m_p_ends{nullptr};

    // Expression values: double arrays
    std::vector<PMPY> m_py_values;
    std::vector<double *> m_p_values;

    // IntervalID: long array
    PMPY m_py_interval_ids;
    long *m_p_interval_ids{nullptr};

    void alloc_arrays(size_t n)
    {
        npy_intp dims[1] = {static_cast<npy_intp>(n)};

        // Chromids as int32 (temporary, converted to strings in finalize)
        m_py_chromids.assign(PyArray_SimpleNew(1, dims, NPY_INT32), true);
        m_p_chromids = static_cast<int32_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(m_py_chromids))));

        // Starts
        m_py_starts.assign(PyArray_SimpleNew(1, dims, NPY_LONG), true);
        m_p_starts = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(m_py_starts))));

        // Ends
        m_py_ends.assign(PyArray_SimpleNew(1, dims, NPY_LONG), true);
        m_p_ends = static_cast<int64_t *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(m_py_ends))));

        // Expression values
        m_py_values.resize(m_num_expr_cols);
        m_p_values.resize(m_num_expr_cols);
        for (size_t i = 0; i < m_num_expr_cols; ++i) {
            m_py_values[i].assign(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
            m_p_values[i] = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(m_py_values[i]))));
        }

        // IntervalID
        m_py_interval_ids.assign(PyArray_SimpleNew(1, dims, NPY_LONG), true);
        m_p_interval_ids = static_cast<long *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(m_py_interval_ids))));
    }

    void grow()
    {
        size_t new_size = m_alloc_size * 2;
        npy_intp dims[1] = {static_cast<npy_intp>(new_size)};

        // Helper: allocate new array, memcpy old data, update pointer
        auto grow_array = [&](PMPY &py_arr, void *&ptr, int npy_type, size_t elem_size) {
            PMPY py_new(PyArray_SimpleNew(1, dims, npy_type), true);
            void *new_ptr = PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(py_new)));
            std::memcpy(new_ptr, ptr, m_row * elem_size);
            py_arr = py_new;
            ptr = new_ptr;
        };

        // Grow chromids
        {
            void *vp = m_p_chromids;
            grow_array(m_py_chromids, vp, NPY_INT32, sizeof(int32_t));
            m_p_chromids = static_cast<int32_t *>(vp);
        }

        // Grow starts
        {
            void *vp = m_p_starts;
            grow_array(m_py_starts, vp, NPY_LONG, sizeof(int64_t));
            m_p_starts = static_cast<int64_t *>(vp);
        }

        // Grow ends
        {
            void *vp = m_p_ends;
            grow_array(m_py_ends, vp, NPY_LONG, sizeof(int64_t));
            m_p_ends = static_cast<int64_t *>(vp);
        }

        // Grow expression values
        for (size_t i = 0; i < m_num_expr_cols; ++i) {
            void *vp = m_p_values[i];
            grow_array(m_py_values[i], vp, NPY_DOUBLE, sizeof(double));
            m_p_values[i] = static_cast<double *>(vp);
        }

        // Grow intervalID
        {
            void *vp = m_p_interval_ids;
            grow_array(m_py_interval_ids, vp, NPY_LONG, sizeof(long));
            m_p_interval_ids = static_cast<long *>(vp);
        }

        m_alloc_size = new_size;
    }

    void truncate_arrays()
    {
        npy_intp dims[1] = {static_cast<npy_intp>(m_row)};

        auto truncate_array = [&](PMPY &py_arr, void *&ptr, int npy_type, size_t elem_size) {
            PMPY py_new(PyArray_SimpleNew(1, dims, npy_type), true);
            void *new_ptr = PyArray_DATA(reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(py_new)));
            std::memcpy(new_ptr, ptr, m_row * elem_size);
            py_arr = py_new;
            ptr = new_ptr;
        };

        // Truncate chromids
        {
            void *vp = m_p_chromids;
            truncate_array(m_py_chromids, vp, NPY_INT32, sizeof(int32_t));
            m_p_chromids = static_cast<int32_t *>(vp);
        }

        // Truncate starts
        {
            void *vp = m_p_starts;
            truncate_array(m_py_starts, vp, NPY_LONG, sizeof(int64_t));
            m_p_starts = static_cast<int64_t *>(vp);
        }

        // Truncate ends
        {
            void *vp = m_p_ends;
            truncate_array(m_py_ends, vp, NPY_LONG, sizeof(int64_t));
            m_p_ends = static_cast<int64_t *>(vp);
        }

        // Truncate expression values
        for (size_t i = 0; i < m_num_expr_cols; ++i) {
            void *vp = m_p_values[i];
            truncate_array(m_py_values[i], vp, NPY_DOUBLE, sizeof(double));
            m_p_values[i] = static_cast<double *>(vp);
        }

        // Truncate intervalID
        {
            void *vp = m_p_interval_ids;
            truncate_array(m_py_interval_ids, vp, NPY_LONG, sizeof(long));
            m_p_interval_ids = static_cast<long *>(vp);
        }

        m_alloc_size = m_row;
    }

    PMPY build_chrom_column()
    {
        npy_intp dims[1] = {static_cast<npy_intp>(m_row)};
        PMPY py_chrom(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
        PyObject **chrom_data = static_cast<PyObject **>(PyArray_DATA(
            reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(py_chrom))));

        // Cache chrom strings to avoid repeated PyUnicode_FromString
        std::unordered_map<int32_t, PyObject *> chrom_cache;
        const GenomeChromKey &chromkey = g_pmdb->chromkey();

        for (size_t i = 0; i < m_row; ++i) {
            int32_t cid = m_p_chromids[i];
            auto it = chrom_cache.find(cid);
            PyObject *chrom_str;
            if (it != chrom_cache.end()) {
                chrom_str = it->second;
            } else {
                chrom_str = PyUnicode_FromString(chromkey.id2chrom(cid).c_str());
                chrom_cache[cid] = chrom_str;
            }
            Py_INCREF(chrom_str);
            chrom_data[i] = chrom_str;
        }

        // Release the cache references
        for (auto &pair : chrom_cache) {
            Py_DECREF(pair.second);
        }

        return py_chrom;
    }

    void build_and_set_colnames(PMPY &py_answer)
    {
        size_t total_cols = 3 + m_num_expr_cols + 1;
        npy_intp dims[1] = {static_cast<npy_intp>(total_cols)};
        PMPY py_colnames(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);

        auto set_name = [&](size_t idx, const char *name) {
            PyObject *py_name = PyUnicode_FromString(name);
            PyObject **data = static_cast<PyObject **>(PyArray_DATA(
                reinterpret_cast<PyArrayObject *>(static_cast<PyObject *>(py_colnames))));
            Py_XDECREF(data[idx]);
            data[idx] = py_name;
        };

        set_name(0, "chrom");
        set_name(1, "start");
        set_name(2, "end");
        for (size_t i = 0; i < m_num_expr_cols; ++i) {
            set_name(3 + i, m_expr_colnames[i].c_str());
        }
        set_name(3 + m_num_expr_cols, "intervalID");

        py_colnames.to_be_stolen();
        PyList_SetItem(py_answer, 0, py_colnames);
    }
};

#endif // PMDIRECTACCUMULATOR_H_INCLUDED
