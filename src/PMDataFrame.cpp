#include "PMDataFrame.h"

void PMDataFrame::init(size_t num_rows, size_t num_cols, const char *df_name)
{
    clear();
    m_num_rows = num_rows;
    m_num_cols = num_cols;
    m_df_name = df_name;
    m_cols.resize(m_num_cols);

    npy_intp dims[1] = {(npy_intp)m_num_cols};
    m_py_colnames.assign(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
}

void PMDataFrame::init(const PMPY &py_df, const char *df_name)
{
    clear();
    m_df_name = df_name;

    if (!PyList_Check(py_df))
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    size_t arg_list_size = PyList_Size(py_df);

    if (arg_list_size < 2)
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    m_py_colnames.assign(PyList_GetItem(py_df, 0));

    if (!PyArray_Check(m_py_colnames))
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    if (PyArray_NDIM((PyArrayObject *)*m_py_colnames) != 1)
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    m_num_cols = (int)PMLEN(m_py_colnames);

    if (!m_num_cols)
        m_num_rows = 0;

    if (arg_list_size != m_num_cols + 1)
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    PyArray_Descr *t = PyArray_DTYPE((PyArrayObject *)*m_py_colnames);

    if (!PyDataType_ISSTRING(t) && !PyDataType_ISOBJECT(t))
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    m_py_colnames.assign(PyArray_FROM_OTF(m_py_colnames, NPY_OBJECT, 0), true);
    if (!m_py_colnames)
        TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);

    m_cols.resize(m_num_cols);
    for (size_t i = 0; i < m_num_cols; ++i)
        m_cols[i].py_col.assign(PyList_GetItem(py_df, i + 1));
}

void PMDataFrame::clear()
{
    m_df_name = "";
    m_py_df.clear();
    m_py_colnames.clear();
    m_num_rows = UNINITIALIZED_SIZE;
    m_num_cols = UNINITIALIZED_SIZE;
    m_cols.clear();
}

void PMDataFrame::init_col(size_t col, const char *name, Type type)
{
    auto set_colname = [&](size_t idx, const std::string &colname) {
        PyObject *py_name = PyUnicode_FromString(colname.c_str());
        if (!py_name)
            TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);
        PyArrayObject *arr = (PyArrayObject *)(PyObject *)m_py_colnames;
        char *ptr = (char *)PyArray_GETPTR1(arr, idx);
        if (PyArray_SETITEM(arr, ptr, py_name) < 0) {
            Py_DECREF(py_name);
            TGLError("%s (%d)", bad_df_msg().c_str(), __LINE__);
        }
        Py_DECREF(py_name);
    };

    set_colname(col, name);

    // make sure that the column name is unique, otherwise add '_' to the end of the column name
    for (int i = 0; i < (int)m_num_cols; ++i) {
        if (m_cols[i].type != -1 && !strcmp(col_name(i), col_name(col))) {
            set_colname(col, string(col_name(col)) + '_');
            i = -1;   // the duplicate has been found and a new name was assigned => rewind the loop to the beginning and check the new name
        }
    }

    npy_intp dims[1] = {(npy_intp)m_num_rows};
    switch (type) {
    case DOUBLE:
        m_cols[col].py_double.assign(PyArray_SimpleNew(1, dims, NPY_DOUBLE), true);
        break;
    case LONG:
    case CAT:
        m_cols[col].py_long.assign(PyArray_SimpleNew(1, dims, NPY_LONG), true);
        break;
    case BOOL:
        m_cols[col].py_bool.assign(PyArray_SimpleNew(1, dims, NPY_BOOL), true);
        break;
    case STR:
        m_cols[col].py_str.assign(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
        break;
    }
    m_cols[col].type = type;
}

void PMDataFrame::cats(size_t col, const vector<string> &cats)
{
    if (!m_cols[col].py_cats) {
        npy_intp dims[1] = {(npy_intp)cats.size()};
        m_cols[col].py_cats.assign(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
        m_cols[col].num_cats = cats.size();
    }

    PyArrayObject *cats_arr = (PyArrayObject *)(PyObject *)m_cols[col].py_cats;
    for (size_t i = 0; i < cats.size(); ++i) {
        PyObject *py_cat = PyUnicode_FromString(cats[i].c_str());
        if (!py_cat)
            TGLError("%s: failed to create category value (%d)", m_df_name.c_str(), __LINE__);
        char *ptr = (char *)PyArray_GETPTR1(cats_arr, i);
        if (PyArray_SETITEM(cats_arr, ptr, py_cat) < 0) {
            Py_DECREF(py_cat);
            TGLError("%s: failed to set category value (%d)", m_df_name.c_str(), __LINE__);
        }
        Py_DECREF(py_cat);
    }
}

void PMDataFrame::col2type(const PMPY &py_col, PMPY &py_vals, int type, const char *type_name, const char *col_name)
{
    if (PMIS1D(py_col)) {
        py_vals.assign(PyArray_FROM_OTF(py_col, type, NPY_ARRAY_FORCECAST), true);
        if (!py_vals)
            TGLError("%s: column %s must consist of %s values (%d)", m_df_name.c_str(), col_name, type_name, __LINE__);

        size_t num_rows = PMLEN(py_vals);
        if (m_num_rows != UNINITIALIZED_SIZE && m_num_rows != num_rows)
            TGLError("%s is not a valid pandas::DataFrame (%d)", m_df_name.c_str(), __LINE__);
        m_num_rows = num_rows;
    } else
        TGLError("%s: column %s must consist of %s values (%d)", m_df_name.c_str(), col_name, type_name, __LINE__);
}

PMPY PMDataFrame::construct_py(bool none_if_empty)
{
    if (!num_rows() && none_if_empty)
        return PMPY(Py_None, true);

    PMPY py_answer(PyList_New(m_num_cols + 1), true);

    m_py_colnames.to_be_stolen();
    PyList_SetItem(py_answer, 0, m_py_colnames);

    for (size_t icol = 0; icol < m_cols.size(); ++icol) {
        PMPY py_obj;

        if (*m_cols[icol].py_cats) {
            py_obj.assign(PyList_New(2), true);
            m_cols[icol].py_cats.to_be_stolen();
            m_cols[icol].py_long.to_be_stolen();
            PyList_SetItem(py_obj, 0, m_cols[icol].py_cats);
            PyList_SetItem(py_obj, 1, m_cols[icol].py_long);
        } else if (*m_cols[icol].py_double)
            py_obj = m_cols[icol].py_double;
        else if (*m_cols[icol].py_long)
            py_obj = m_cols[icol].py_long;
        else if (*m_cols[icol].py_bool)
            py_obj = m_cols[icol].py_bool;
        else
            py_obj = m_cols[icol].py_str;

        py_obj.to_be_stolen();
        PyList_SetItem(py_answer, icol + 1, py_obj);
    }

    return py_answer;
}

void PMDataFrame::df2pymisha(PMPY &py_df, PMPY &py_res)
{
    PMPY py_df2pymisha(PyDict_GetItemString(g_pymisha->py_module_dict(), "_df2pymisha"));
    if (!py_df2pymisha)
        verror("Failed to access '_df2pymisha' from C");
    PMPY py_args(PyTuple_New(1), true);
    py_df.to_be_stolen();
    PyTuple_SetItem(py_args, 0, py_df);
    py_res.assign(PyObject_CallObject(py_df2pymisha, py_args), true);
    if (!py_res)
        verror("Failed to execute _df2pymisha");
}
