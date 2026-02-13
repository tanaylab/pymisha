/*
 * PMTrackExpressionScanner.h
 *
 * Track expression scanner for pymisha with batched NumPy evaluation
 */

#ifndef PMTRACKEXPRESSIONSCANNER_H_
#define PMTRACKEXPRESSIONSCANNER_H_

#include <vector>
#include <string>
#include <memory>
#include <cmath>

#include "pymisha.h"
#include "PMTrackExpressionIterator.h"
#include "PMTrackExpressionVars.h"
#include "GInterval.h"

class PMTrackExprScanner {
public:
    enum ValType { REAL_T, LOGICAL_T };

    PMTrackExprScanner();
    ~PMTrackExprScanner();

    // Begin scanning with given expressions and intervals
    bool begin(const std::vector<std::string> &exprs, ValType valtype,
               const std::vector<GInterval> &intervals, int64_t iterator_policy);

    // Advance to next value
    bool next();

    // Check if scanning is complete
    bool isend() const { return m_isend; }

    // Get the result of last evaluation as double
    double vdouble(int expr_idx = 0) const {
        return m_eval_doubles[expr_idx][m_eval_buf_idx];
    }

    // Get the result of last evaluation as boolean
    bool vbool(int expr_idx = 0) const {
        return m_eval_bools[expr_idx][m_eval_buf_idx];
    }

    // Get the result of last evaluation as logical (for gscreen)
    // Handles both boolean expressions (track1 > 0) and numeric expressions
    // NaN is treated as false (conservative filtering)
    bool vlogical(int expr_idx = 0) const {
        // Check if we have a bool array (from boolean expressions)
        if (m_eval_bools[expr_idx] != nullptr) {
            return m_eval_bools[expr_idx][m_eval_buf_idx];
        }
        // Otherwise use double array (from numeric expressions)
        double val = m_eval_doubles[expr_idx][m_eval_buf_idx];
        // NaN is falsy for filtering purposes (conservative)
        if (std::isnan(val)) return false;
        return val != 0.0;
    }

    // Get current interval
    const GInterval &last_interval() const {
        return m_expr_itr_intervals[m_eval_buf_idx];
    }

    // Get original interval ID (1-based) for current position
    uint64_t last_interval_id() const {
        return m_expr_itr_interval_ids[m_eval_buf_idx];
    }

    // Get iterator
    PMTrackExpressionIterator *get_iterator() const { return m_itr.get(); }

    // Get track expressions
    const std::vector<std::string> &get_track_exprs() const { return m_track_exprs; }

    // Get expression variables
    const PMTrackExpressionVars &get_expr_vars() const { return m_expr_vars; }

    // Enable/disable progress reporting
    void report_progress(bool do_report) { m_do_report_progress = do_report; }
    void set_progress_callback(PyObject *cb);

private:
    std::vector<std::string> m_track_exprs;
    PMTrackExpressionVars m_expr_vars;
    std::unique_ptr<PMTrackExpressionIterator> m_itr;
    ValType m_valtype;

    // Local dictionary for Python evaluation
    PMPY m_py_ldict;

    // Compiled expressions
    std::vector<PMPY> m_py_compiled_exprs;

    // Evaluation buffers (NumPy arrays)
    std::vector<PMPY> m_py_eval_bufs;
    std::vector<double *> m_eval_doubles;
    std::vector<bool *> m_eval_bools;

    // Iterator intervals buffer
    std::vector<GInterval> m_expr_itr_intervals;
    std::vector<uint64_t> m_expr_itr_interval_ids;  // 1-based original interval indices

    // CHROM, START, END arrays for Python
    PMPY m_py_chrom_array;
    PMPY m_py_start_array;
    PMPY m_py_end_array;
    PyObject **m_chrom_array;
    int64_t *m_start_array;
    int64_t *m_end_array;

    // Buffer management
    unsigned m_eval_buf_idx;
    unsigned m_eval_buf_limit;
    unsigned m_eval_buf_size;

    // Progress reporting
    bool m_do_report_progress;
    int m_last_progress_reported;
    size_t m_num_evals;
    int m_report_step;
    uint64_t m_last_report_clock;
    PMPY m_progress_cb;

    // State
    bool m_isend;
    bool m_use_python;

    // Constants
    static const int INIT_REPORT_STEP;
    static const int REPORT_INTERVAL;
    static const int MIN_REPORT_INTERVAL;

    // Internal methods
    void check(const std::vector<std::string> &exprs, const std::vector<GInterval> &intervals,
               int64_t iterator_policy, ValType valtype);
    void define_py_vars(unsigned eval_buf_limit);
    bool eval_next();
    void report_progress_impl();
    void call_progress_callback(int progress, bool final);

    // Create expression iterator based on policy
    std::unique_ptr<PMTrackExpressionIterator> create_expr_iterator(
        const std::vector<GInterval> &intervals, int64_t iterator_policy);
};

#endif /* PMTRACKEXPRESSIONSCANNER_H_ */
