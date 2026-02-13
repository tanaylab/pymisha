#ifndef PMDATAFRAME_H_INCLUDED
#define PMDATAFRAME_H_INCLUDED

#include <cmath>
#include <Python.h>

#include "pymisha.h"
#include "PMObject.h"
#include "TGLException.h"

#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

class PMDataFrame {
public:
    enum Type { DOUBLE, LONG, BOOL, STR, CAT };

    PMDataFrame() {}
    PMDataFrame(size_t num_rows, size_t num_cols, const char *df_name) { init(num_rows, num_cols, df_name); }
    PMDataFrame(const PMPY &py_df, const char *df_name) { init(py_df, df_name); }

    // use this init() for constructing a new data frame
    void init(size_t num_rows, size_t num_cols, const char *df_name);

    // use this init() for accessing an existing data frame
    void init(const PMPY &py_df, const char *df_name);

    void clear();

    // used mainly for reading an existing data frame
    size_t      num_rows();
    size_t      num_cols() const { return m_num_cols; }
    const char *col_name(size_t col) const;
    Type        col_type(size_t col) const;  // detect column type from data
    double      val_double(size_t row, size_t col);
    long        val_long(size_t row, size_t col);
    bool        val_bool(size_t row, size_t col);
    const char *val_str(size_t row, size_t col);

    // used mainly for creation of a new data frame
    void init_col(size_t col, const char *name, Type type);
    void val_double(size_t row, size_t col, double val);
    void val_long(size_t row, size_t col, long val);
    void val_bool(size_t row, size_t col, bool val);
    void val_str(size_t row, size_t col, const char *val);
    void cats(size_t col, const vector<string> &cats);

    // Returns PMPY that represents the data frame.
    // The returned value should be passed to _pymisha2df function to construct pandas::DataFrame.
    PMPY construct_py(bool none_if_empty);

    // converts data frame in pandas format to pymisha's format which can be used then to construct PMDataFrame
    static void df2pymisha(PMPY &py_df, PMPY &py_res);

private:
    static constexpr size_t UNINITIALIZED_SIZE = static_cast<size_t>(-1);

    struct Column {
        int    type{-1};
        PMPY   py_col;
        PMPY   py_double;
        PMPY   py_long;
        PMPY   py_bool;
        PMPY   py_str;
        PMPY   py_cats;
        size_t num_cats{0};
    };

    string m_df_name;
    PMPY   m_py_df;
    PMPY   m_py_colnames;
    size_t m_num_rows{UNINITIALIZED_SIZE};
    size_t m_num_cols{UNINITIALIZED_SIZE};

    vector<Column> m_cols;

    void col2type(const PMPY &py_col, PMPY &py_vals, int type, const char *type_name, const char *col_name);
    string bad_df_msg() const { return m_df_name + " is not a valid pandas::DataFrame"; }
};


//------------------------------------ IMPLEMENTATION -----------------------------------------

inline size_t PMDataFrame::num_rows()
{
    if (m_num_rows == UNINITIALIZED_SIZE) {
        // Find the number of rows from any column
        // Columns can be either arrays (regular) or lists of [categories, codes] (categorical)
        for (const auto &col : m_cols) {
            if (PyArray_Check(col.py_col) && PyArray_NDIM((PyArrayObject *)*col.py_col) == 1) {
                // Regular column: use array length
                m_num_rows = PMLEN(col.py_col);
                break;
            } else if (PyList_Check(col.py_col) && PyList_Size(col.py_col) == 2) {
                // Categorical column: use codes array length
                PyObject *codes = PyList_GetItem(col.py_col, 1);
                if (PyArray_Check(codes) && PyArray_NDIM((PyArrayObject *)codes) == 1) {
                    m_num_rows = PyArray_DIM((PyArrayObject *)codes, 0);
                    break;
                }
            }
        }
        if (m_num_rows == UNINITIALIZED_SIZE)
            TGLError("%s: cannot determine number of rows", bad_df_msg().c_str());
    }
    return m_num_rows;
}

inline const char *PMDataFrame::col_name(size_t col) const
{
    if (!PyUnicode_Check(PMOBJ(m_py_colnames, col)))
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    return PyUnicode_AsUTF8(PMOBJ(m_py_colnames, col));
}

inline PMDataFrame::Type PMDataFrame::col_type(size_t col) const
{
    // Check if the column is categorical (list of [categories, codes])
    if (PyList_Check(m_cols[col].py_col) && PyList_Size(m_cols[col].py_col) == 2) {
        // Categorical column - typically string
        return STR;
    }

    // Check if it's an array
    if (PyArray_Check(m_cols[col].py_col)) {
        PyArray_Descr *dtype = PyArray_DTYPE((PyArrayObject *)*m_cols[col].py_col);
        int type_num = dtype->type_num;

        // String/object type
        if (type_num == NPY_OBJECT || PyDataType_ISSTRING(dtype)) {
            return STR;
        }
        // Boolean type
        if (type_num == NPY_BOOL) {
            return BOOL;
        }
        // Integer types
        if (PyDataType_ISINTEGER(dtype)) {
            return LONG;
        }
        // Float types
        if (PyDataType_ISFLOAT(dtype)) {
            return DOUBLE;
        }
    }

    // Default to DOUBLE for unknown types
    return DOUBLE;
}

inline double PMDataFrame::val_double(size_t row, size_t col)
{
    if (!m_cols[col].py_double)
        col2type(m_cols[col].py_col, m_cols[col].py_double, NPY_DOUBLE, "numeric", col_name(col));
    return PMDOUBLE(m_cols[col].py_double, row);
}

inline void PMDataFrame::val_double(size_t row, size_t col, double val)
{
    PMDOUBLE(m_cols[col].py_double, row) = val;
}

inline long PMDataFrame::val_long(size_t row, size_t col)
{
    if (!m_cols[col].py_long)
        col2type(m_cols[col].py_col, m_cols[col].py_long, NPY_LONG, "integer", col_name(col));
    return PMLONG(m_cols[col].py_long, row);
}

inline void PMDataFrame::val_long(size_t row, size_t col, long val)
{
    PMLONG(m_cols[col].py_long, row) = val;
}

inline bool PMDataFrame::val_bool(size_t row, size_t col)
{
    if (!m_cols[col].py_bool)
        col2type(m_cols[col].py_col, m_cols[col].py_bool, NPY_BOOL, "boolean", col_name(col));
    return PMBOOL(m_cols[col].py_bool, row);
}

inline void PMDataFrame::val_bool(size_t row, size_t col, bool val)
{
    PMBOOL(m_cols[col].py_bool, row) = val;
}

inline const char *PMDataFrame::val_str(size_t row, size_t col)
{
    if (!m_cols[col].py_str && !m_cols[col].py_cats) {
        if (PyArray_Check(m_cols[col].py_col))    // column == array of strings
            col2type(m_cols[col].py_col, m_cols[col].py_str, NPY_OBJECT, "string", col_name(col));
        else {   // column == categories (factor)
            if (!PyList_Check(m_cols[col].py_col) || PyList_Size(m_cols[col].py_col) != 2)
                TGLError("%s: column %s must consist of string values (%d)", m_df_name.c_str(), col_name(col), __LINE__);

            PyObject *py_cats = PyList_GetItem(m_cols[col].py_col, 0);
            PyObject *py_codes = PyList_GetItem(m_cols[col].py_col, 1);

            PyArray_Descr *cats_t = PyArray_DTYPE((PyArrayObject *)py_cats);
            PyArray_Descr *codes_t = PyArray_DTYPE((PyArrayObject *)py_codes);

            if ((!PyDataType_ISSTRING(cats_t) && !PyDataType_ISOBJECT(cats_t)) || !PyDataType_ISNUMBER(codes_t))
                TGLError("%s: column %s must consist of string values (%d)", m_df_name.c_str(), col_name(col), __LINE__);

            m_cols[col].py_cats.assign(PyArray_FROM_OTF(py_cats, NPY_OBJECT, 0), true);
            if (!m_cols[col].py_cats)
                TGLError("%s: column %s must consist of string values (%d)", m_df_name.c_str(), col_name(col), __LINE__);
            m_cols[col].num_cats = PMLEN(m_cols[col].py_cats);

            col2type(py_codes, m_cols[col].py_long, NPY_LONG, "string", col_name(col));
        }
    }

    if (*m_cols[col].py_str) {    // column == array of strings
        PyObject *py_val = PMOBJ(m_cols[col].py_str, row);
        if (PyUnicode_Check(py_val))
            return PyUnicode_AsUTF8(py_val);
        if (PyFloat_Check(py_val) && std::isnan(PyFloat_AsDouble(py_val)))
            return NULL;   // NA string
        TGLError("%s: column %s must consist of string values (%d)", m_df_name.c_str(), col_name(col), __LINE__);
    }

    // column == categories (factor)
    long code = PMLONG(m_cols[col].py_long, row);
    if (code == -1)
        return NULL;    // NA string

    if (code < 0 || code >= (long)m_cols[col].num_cats)
        TGLError("%s: column %s must consist of string values (%d)", m_df_name.c_str(), col_name(col), __LINE__);

    PyObject *py_val = PMOBJ(m_cols[col].py_cats, code);
    if (!PyUnicode_Check(py_val))
        TGLError("%s: column %s must consist of string values (%d)", m_df_name.c_str(), col_name(col), __LINE__);

    return PyUnicode_AsUTF8(py_val);
}

inline void PMDataFrame::val_str(size_t row, size_t col, const char *val)
{
    PyObject *py_val = val ? PyUnicode_FromString(val) : PyFloat_FromDouble(NPY_NAN);
    if (!py_val)
        verror("Failed to create Python object for string value");
    PyArrayObject *str_arr = (PyArrayObject *)(PyObject *)m_cols[col].py_str;
    char *ptr = (char *)PyArray_GETPTR1(str_arr, row);
    if (PyArray_SETITEM(str_arr, ptr, py_val) < 0) {
        Py_DECREF(py_val);
        verror("Failed to set string value in DataFrame");
    }
    Py_DECREF(py_val);
}

#endif
