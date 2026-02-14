/*
 * PMTrackExpressionScanner.cpp
 *
 * Track expression scanner for pymisha with batched NumPy evaluation
 */

#include <sys/time.h>
#include <cmath>
#include <algorithm>

#include "PMTrackExpressionScanner.h"
#include "TGLException.h"

const int PMTrackExprScanner::INIT_REPORT_STEP = 10000;
const int PMTrackExprScanner::REPORT_INTERVAL = 3000;
const int PMTrackExprScanner::MIN_REPORT_INTERVAL = 1000;

static uint64_t get_cur_clock()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

PMTrackExprScanner::PMTrackExprScanner()
    : m_valtype(REAL_T),
      m_chrom_array(nullptr),
      m_start_array(nullptr),
      m_end_array(nullptr),
      m_eval_buf_idx(0),
      m_eval_buf_limit(0),
      m_eval_buf_size(0),
      m_do_report_progress(true),
      m_last_progress_reported(-1),
      m_num_evals(0),
      m_report_step(INIT_REPORT_STEP),
      m_last_report_clock(0),
      m_isend(true),
      m_use_python(true)
{
}

PMTrackExprScanner::~PMTrackExprScanner()
{
    for (auto &pair : m_chrom_str_cache) {
        Py_XDECREF(pair.second);
    }
    m_chrom_str_cache.clear();
    Py_XDECREF(m_empty_chrom_str);
}

std::unique_ptr<PMTrackExpressionIterator> PMTrackExprScanner::create_expr_iterator(
    const std::vector<GInterval> &intervals, int64_t iterator_policy)
{
    if (iterator_policy > 0) {
        // Positive integer = fixed bin size
        return std::make_unique<PMFixedBinIterator>(intervals, iterator_policy);
    } else if (iterator_policy == 0) {
        // 0 = infer iterator from track type
        if (m_expr_vars.has_common_track_type() &&
            m_expr_vars.get_common_track_type() == GenomeTrack::SPARSE) {
            return std::make_unique<PMSparseIterator>(intervals, m_expr_vars.first_track_path());
        }

        int64_t bin_size = m_expr_vars.get_bin_size();
        if (bin_size > 0) {
            return std::make_unique<PMFixedBinIterator>(intervals, bin_size);
        }

        return std::make_unique<PMIntervalsIterator>(intervals);
    } else {
        // -1 or less = intervals iterator
        return std::make_unique<PMIntervalsIterator>(intervals);
    }
}

void PMTrackExprScanner::check(const std::vector<std::string> &exprs,
                               const std::vector<GInterval> &intervals,
                               int64_t iterator_policy, ValType valtype)
{
    m_track_exprs = exprs;
    m_valtype = valtype;

    // Parse expressions to find track variables
    std::vector<std::string> exprs4compile;
    m_expr_vars.parse_exprs(m_track_exprs, exprs4compile);

    // Create the iterator
    m_itr = create_expr_iterator(intervals, iterator_policy);

    // Compile expressions
    m_py_compiled_exprs.resize(m_track_exprs.size());
    m_py_eval_bufs.resize(m_track_exprs.size());
    m_eval_doubles.resize(m_track_exprs.size(), nullptr);
    m_eval_bools.resize(m_track_exprs.size(), nullptr);

    for (size_t iexpr = 0; iexpr < m_track_exprs.size(); ++iexpr) {
        // Check if expression is just a track variable name
        const PMTrackExpressionVars::TrackVar *var = m_expr_vars.var(m_track_exprs[iexpr].c_str());
        if (!var) {
            // Need to compile: not a simple track name
            m_py_compiled_exprs[iexpr].assign(
                Py_CompileString(exprs4compile[iexpr].c_str(), "<string>", Py_eval_input), true);

            if (!m_py_compiled_exprs[iexpr]) {
                PyObject *py_type, *py_value, *py_traceback;
                PyErr_Fetch(&py_type, &py_value, &py_traceback);
                std::string err_msg = "Error while compiling expression '";
                err_msg += m_track_exprs[iexpr];
                err_msg += "'";
                if (py_value) {
                    PMPY str(PyObject_Str(py_value), true);
                    if (str) {
                        err_msg += ": ";
                        err_msg += PyUnicode_AsUTF8(str);
                    }
                }
                Py_XDECREF(py_type);
                Py_XDECREF(py_value);
                Py_XDECREF(py_traceback);
                TGLError("%s", err_msg.c_str());
            }
        }
    }

    // Check if we can bypass Python evaluation
    m_use_python = false;
    for (const auto &compiled : m_py_compiled_exprs) {
        if (compiled) {
            m_use_python = true;
            break;
        }
    }
    
    if (g_pymisha && g_pymisha->debug()) {
        vdebug("PMTrackExprScanner: use_python=%s\n", m_use_python ? "true" : "false");
    }
}

void PMTrackExprScanner::define_py_vars(unsigned eval_buf_limit)
{
    m_eval_buf_limit = eval_buf_limit;
    m_expr_itr_intervals.resize(eval_buf_limit);
    m_expr_itr_interval_ids.resize(eval_buf_limit, 0);

    if (m_use_python) {
        // Create local dictionary for evaluation
        m_py_ldict.assign(PyDict_New(), true);

        // Add numpy to local dict so expressions can use np.log2, np.sqrt, etc.
        PMPY np_module(PyImport_ImportModule("numpy"), true);
        if (np_module) {
            PyDict_SetItemString(m_py_ldict, "np", np_module);
            PyDict_SetItemString(m_py_ldict, "numpy", np_module);
        }
    }

    // Define track variables
    m_expr_vars.define_py_vars(eval_buf_limit, m_py_ldict, m_use_python);

    if (m_use_python) {
        // Create CHROM, START, END arrays
        npy_intp dims[1] = {(npy_intp)eval_buf_limit};

        m_py_chrom_array.assign(PyArray_SimpleNew(1, dims, NPY_OBJECT), true);
        m_py_start_array.assign(PyArray_SimpleNew(1, dims, NPY_INT64), true);
        m_py_end_array.assign(PyArray_SimpleNew(1, dims, NPY_INT64), true);

        m_chrom_array = (PyObject **)PyArray_DATA((PyArrayObject *)*m_py_chrom_array);
        m_start_array = (int64_t *)PyArray_DATA((PyArrayObject *)*m_py_start_array);
        m_end_array = (int64_t *)PyArray_DATA((PyArrayObject *)*m_py_end_array);

        // Initialize with safe defaults
        for (unsigned i = 0; i < eval_buf_limit; ++i) {
            Py_INCREF(Py_None);
            m_chrom_array[i] = Py_None;
            m_start_array[i] = 0;
            m_end_array[i] = 0;
        }

        PyDict_SetItemString(m_py_ldict, "CHROM", m_py_chrom_array);
        PyDict_SetItemString(m_py_ldict, "START", m_py_start_array);
        PyDict_SetItemString(m_py_ldict, "END", m_py_end_array);

        // Scan expressions to determine which coordinate arrays are actually referenced
        m_need_chrom = m_need_start = m_need_end = false;
        for (const auto &expr : m_track_exprs) {
            if (expr.find("CHROM") != std::string::npos) m_need_chrom = true;
            if (expr.find("START") != std::string::npos) m_need_start = true;
            if (expr.find("END") != std::string::npos) m_need_end = true;
        }
    }

    // For expressions that are just track variable references, point directly to the array
    // Note: Only do this for REAL_T mode. For LOGICAL_T, we need to go through
    // the compilation path to get proper boolean conversion.
    for (size_t iexpr = 0; iexpr < m_track_exprs.size(); ++iexpr) {
        const PMTrackExpressionVars::TrackVar *var = m_expr_vars.var(m_track_exprs[iexpr].c_str());
        if (var && m_valtype == REAL_T) {
            m_eval_doubles[iexpr] = var->values;
        }
    }
}

bool PMTrackExprScanner::begin(const std::vector<std::string> &exprs, ValType valtype,
                               const std::vector<GInterval> &intervals, int64_t iterator_policy)
{
    vdebug("PMTrackExprScanner::begin with %lu expressions, %lu intervals\n",
           exprs.size(), intervals.size());

    check(exprs, intervals, iterator_policy, valtype);

    // Define Python variables with buffer size from config
    unsigned buf_size = g_pymisha ? g_pymisha->eval_buf_size() : 1000;
    define_py_vars(buf_size);

    // Initialize iteration state
    m_num_evals = 0;
    m_last_progress_reported = -1;
    m_report_step = INIT_REPORT_STEP;
    m_last_report_clock = get_cur_clock();

    m_isend = false;
    m_eval_buf_idx = m_eval_buf_limit;  // Force refill on first next()
    m_eval_buf_size = 0;

    m_itr->begin();

    return next();
}

bool PMTrackExprScanner::next()
{
    if (isend())
        return false;

    if (eval_next())
        return true;

    // Report final progress
    if (m_last_progress_reported >= 0) {
        if (m_last_progress_reported != 100)
            vemsg("100%%\n");
        else
            vemsg("\n");
    }
    call_progress_callback(100, true);

    return false;
}

void PMTrackExprScanner::set_progress_callback(PyObject *cb)
{
    if (cb && cb != Py_None) {
        if (!PyCallable_Check(cb)) {
            TGLError("progress callback must be callable");
        }
        Py_INCREF(cb);
        m_progress_cb.assign(cb, true);
    } else {
        m_progress_cb.assign(NULL, false);
    }
}

bool PMTrackExprScanner::eval_next()
{
    m_eval_buf_idx++;

    // Need to refill the evaluation buffer?
    if (m_eval_buf_idx >= m_eval_buf_limit) {
        m_eval_buf_idx = 0;

        // Fill buffer with iterator values
        for (m_eval_buf_size = 0; m_eval_buf_size < m_eval_buf_limit; ++m_eval_buf_size) {
            if (m_itr->isend()) {
                // Pad remainder with NaN/sentinel values to prevent stale data
                // from affecting non-elementwise expressions (np.mean, np.sort, etc.)
                if (m_use_python) {
                    if (!m_empty_chrom_str)
                        m_empty_chrom_str = PyUnicode_FromString("");
                    for (unsigned i = m_eval_buf_size; i < m_eval_buf_limit; ++i) {
                        Py_DECREF(m_chrom_array[i]);
                        Py_INCREF(m_empty_chrom_str);
                        m_chrom_array[i] = m_empty_chrom_str;
                        m_start_array[i] = -1;
                        m_end_array[i] = -1;
                    }
                }
                // NaN-fill track variable arrays for tail slots
                m_expr_vars.pad_tail_with_nan(m_eval_buf_size, m_eval_buf_limit);
                break;
            }

            const GInterval &interval = m_itr->last_interval();
            m_expr_itr_intervals[m_eval_buf_size] = interval;
            m_expr_itr_interval_ids[m_eval_buf_size] = m_itr->original_interval_idx();

            if (m_use_python && (m_need_chrom || m_need_start || m_need_end)) {
                if (m_need_chrom) {
                    Py_DECREF(m_chrom_array[m_eval_buf_size]);
                    auto it = m_chrom_str_cache.find(interval.chromid);
                    if (it == m_chrom_str_cache.end()) {
                        const std::string &chrom = g_pmdb->chromkey().id2chrom(interval.chromid);
                        PyObject *chrom_str = PyUnicode_FromString(chrom.c_str());
                        m_chrom_str_cache[interval.chromid] = chrom_str;
                        it = m_chrom_str_cache.find(interval.chromid);
                    }
                    Py_INCREF(it->second);
                    m_chrom_array[m_eval_buf_size] = it->second;
                }
                if (m_need_start)
                    m_start_array[m_eval_buf_size] = interval.start;
                if (m_need_end)
                    m_end_array[m_eval_buf_size] = interval.end;
            }

            // Set track variable values
            m_expr_vars.set_vars(interval, m_eval_buf_size);

            m_itr->next();
        }

        check_interrupt();

        // Evaluate compiled expressions
        if (m_use_python) {
            for (size_t iexpr = 0; iexpr < m_track_exprs.size(); ++iexpr) {
                if (m_py_compiled_exprs[iexpr]) {
                    m_py_eval_bufs[iexpr].assign(
                        PyEval_EvalCode(m_py_compiled_exprs[iexpr], g_pymisha->gdict(), m_py_ldict), true);

                    if (PyErr_Occurred()) {
                        PyObject *py_type, *py_value, *py_traceback;
                        PyErr_Fetch(&py_type, &py_value, &py_traceback);
                        std::string err_msg = "Error while evaluating expression '";
                        err_msg += m_track_exprs[iexpr];
                        err_msg += "'";
                        if (py_value) {
                            PMPY str(PyObject_Str(py_value), true);
                            if (str) {
                                err_msg += ": ";
                                err_msg += PyUnicode_AsUTF8(str);
                            }
                        }
                        Py_XDECREF(py_type);
                        Py_XDECREF(py_value);
                        Py_XDECREF(py_traceback);
                        TGLError("%s", err_msg.c_str());
                    }

                    if (!PMIS1D(m_py_eval_bufs[iexpr])) {
                        TGLError("Evaluation of expression '%s' does not produce a numpy array",
                                 m_track_exprs[iexpr].c_str());
                    }

                    if (PMLEN(m_py_eval_bufs[iexpr]) != m_eval_buf_limit) {
                        TGLError("Evaluation of expression '%s' produces array of size %lu, expected %u",
                                 m_track_exprs[iexpr].c_str(),
                                 PMLEN(m_py_eval_bufs[iexpr]), m_eval_buf_limit);
                    }

                    PyArray_Descr *t = PyArray_DTYPE((PyArrayObject *)*m_py_eval_bufs[iexpr]);

                    if (PyTypeNum_ISBOOL(t->type_num)) {
                        // Boolean result - for LOGICAL_T, use directly as bool
                        // For REAL_T, convert to double (0.0/1.0)
                        if (m_valtype == LOGICAL_T) {
                            m_py_eval_bufs[iexpr].assign(
                                PyArray_FROM_OTF(m_py_eval_bufs[iexpr], NPY_BOOL, NPY_ARRAY_FORCECAST), true);
                            if (!m_py_eval_bufs[iexpr]) {
                                TGLError("Failed to convert result of expression '%s' to boolean array",
                                         m_track_exprs[iexpr].c_str());
                            }
                            m_eval_bools[iexpr] = (bool *)PyArray_DATA((PyArrayObject *)*m_py_eval_bufs[iexpr]);
                        } else {
                            // Convert bool to double for REAL_T
                            m_py_eval_bufs[iexpr].assign(
                                PyArray_FROM_OTF(m_py_eval_bufs[iexpr], NPY_DOUBLE, NPY_ARRAY_FORCECAST), true);
                            if (!m_py_eval_bufs[iexpr]) {
                                TGLError("Failed to convert result of expression '%s' to double array",
                                         m_track_exprs[iexpr].c_str());
                            }
                            m_eval_doubles[iexpr] = (double *)PyArray_DATA((PyArrayObject *)*m_py_eval_bufs[iexpr]);
                        }
                    } else if (PyDataType_ISNUMBER(t)) {
                        // Numeric result - for REAL_T, use as double
                        // For LOGICAL_T, convert to bool (0 = false, else = true, NaN = false)
                        if (m_valtype == LOGICAL_T) {
                            // First convert to double, then we'll interpret as boolean in vlogical()
                            m_py_eval_bufs[iexpr].assign(
                                PyArray_FROM_OTF(m_py_eval_bufs[iexpr], NPY_DOUBLE, NPY_ARRAY_FORCECAST), true);
                            if (!m_py_eval_bufs[iexpr]) {
                                TGLError("Failed to convert result of expression '%s' to double array",
                                         m_track_exprs[iexpr].c_str());
                            }
                            m_eval_doubles[iexpr] = (double *)PyArray_DATA((PyArrayObject *)*m_py_eval_bufs[iexpr]);
                        } else {
                            m_py_eval_bufs[iexpr].assign(
                                PyArray_FROM_OTF(m_py_eval_bufs[iexpr], NPY_DOUBLE, NPY_ARRAY_FORCECAST), true);
                            if (!m_py_eval_bufs[iexpr]) {
                                TGLError("Failed to convert result of expression '%s' to double array",
                                         m_track_exprs[iexpr].c_str());
                            }
                            m_eval_doubles[iexpr] = (double *)PyArray_DATA((PyArrayObject *)*m_py_eval_bufs[iexpr]);
                        }
                    } else {
                        TGLError("Evaluation of expression '%s' produces array of unsupported type %c",
                                 m_track_exprs[iexpr].c_str(), t->type);
                    }
                }
            }
        }

        report_progress_impl();
    }

    if (m_eval_buf_idx >= m_eval_buf_size) {
        m_eval_buf_idx = m_eval_buf_limit;
        m_isend = true;
    }

    return !m_isend;
}

void PMTrackExprScanner::report_progress_impl()
{
    m_num_evals += m_eval_buf_size;

    const bool want_text = m_do_report_progress;
    const bool want_cb = m_progress_cb && !PyMisha::is_kid();

    if (!want_text && !want_cb)
        return;

    if (m_num_evals > (size_t)m_report_step) {
        uint64_t curclock = get_cur_clock();
        double delta = curclock - m_last_report_clock;

        if (delta)
            m_report_step = (int)(m_report_step * (REPORT_INTERVAL / delta) + 0.5);
        else
            m_report_step *= 10;

        if (delta > MIN_REPORT_INTERVAL) {
            int progress = 0;

            if (m_itr->size()) {
                progress = (int)(m_itr->idx() * 100.0 / m_itr->size());
            }

            progress = std::max(progress, m_last_progress_reported);
            if (progress != 100) {
                if (want_text) {
                    if (progress != m_last_progress_reported)
                        vemsg("%d%%...", progress);
                    else
                        vemsg(".");
                }
                if (want_cb)
                    call_progress_callback(progress, false);
                m_last_progress_reported = progress;
            }
            m_num_evals = 0;
            m_last_report_clock = curclock;
        }
    }
}

void PMTrackExprScanner::call_progress_callback(int progress, bool final)
{
    if (!m_progress_cb || PyMisha::is_kid()) {
        return;
    }

    uint64_t total = 0;
    uint64_t done = 0;
    if (m_itr) {
        total = m_itr->size();
        done = (final && total) ? total : m_itr->idx();
    }

    PyObject *py_done = PyLong_FromUnsignedLongLong(done);
    PyObject *py_total = total ? PyLong_FromUnsignedLongLong(total) : Py_None;
    if (!total) {
        Py_INCREF(Py_None);
    }
    PyObject *py_pct = PyLong_FromLong(progress);

    if (!py_done || !py_total || !py_pct) {
        Py_XDECREF(py_done);
        Py_XDECREF(py_total);
        Py_XDECREF(py_pct);
        TGLError("Failed to allocate arguments for progress callback");
    }

    PyObject *res = PyObject_CallFunctionObjArgs(m_progress_cb, py_done, py_total, py_pct, NULL);
    Py_XDECREF(py_done);
    Py_XDECREF(py_total);
    Py_XDECREF(py_pct);

    if (!res || PyErr_Occurred()) {
        Py_XDECREF(res);
        TGLError("Progress callback failed");
    }
    Py_XDECREF(res);
}
