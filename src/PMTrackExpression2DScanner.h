/*
 * PMTrackExpression2DScanner.h
 *
 * Skeleton 2D track-expression scanner. Parallel to PMTrackExprScanner
 * but specialised for 2D intervals.
 *
 * Session 2 scope: drive a PMTrackExpressionIntervals2DIterator over a
 * std::vector<GInterval2D>, batch-fill per-var values via
 * PMTrackExpression2DVars, and expose the resulting values arrays to the
 * caller. No Python expression compilation. The test-only binding
 * pm_test_2d_scanner uses this directly; Session 3 adds a full
 * pm_extract_2d binding that compiles expressions through Python.
 */

#ifndef PMTRACKEXPRESSION2DSCANNER_H_
#define PMTRACKEXPRESSION2DSCANNER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "GInterval2D.h"
#include "PMTrackExpression2DIterator.h"
#include "PMTrackExpression2DIteratorPolicy.h"
#include "PMTrackExpression2DVars.h"
#include "QuadTreeReader.h"

class PMTrackExpr2DScanner {
public:
    PMTrackExpr2DScanner();
    ~PMTrackExpr2DScanner();

    // Run the scanner end-to-end:
    //   - registers a single track variable (track_name, func_name)
    //   - iterates `intervals` in input order
    //   - fills per-row values into the var's value buffer
    //
    // `band` may be nullptr.
    //
    // After return, values() yields a const double * to the per-row
    // result array of length intervals.size().
    void run_single_var(const std::string &track_name,
                        const std::string &func_name,
                        const std::vector<GInterval2D> &intervals,
                        const quadtree::DiagonalBand *band);

    // Extended var spec: (track_name, func_name, sshift1, eshift1, sshift2, eshift2).
    // The shift fields default to zero (no shift) for backward compatibility.
    struct VarSpec {
        std::string name;
        std::string func;
        int64_t     sshift1{0};
        int64_t     eshift1{0};
        int64_t     sshift2{0};
        int64_t     eshift2{0};
    };

    // Run with a polymorphic iterator policy and a list of VarSpec entries.
    // Mirrors run_single_var but supports any iterator + any number of vars.
    // `scope` is the user's 2D intervals (drives the iterator). `band` may
    // be nullptr.
    //
    // Walk-loop note: the new run() uses read-then-advance so that it
    // works correctly with both PMTrackExpressionIntervals2DIterator
    // (begin() does NOT prime) and PMTrackExpressionFixedRectIterator
    // (constructor calls begin() -> next() to prime the first cell).
    // run_single_var uses a different loop (advance-then-batch-index);
    // both are correct for their respective callers.
    //
    // After return: values_for_var(i) gives a const double * to the
    // per-row result array; num_emitted() gives the row count.
    void run(const PMTrackExpression2DIteratorPolicy &policy,
             const std::vector<VarSpec> &vars,
             const std::vector<GInterval2D> &scope,
             const quadtree::DiagonalBand *band);

    // Access the result array for var i. Valid until the next run/run_single_var call.
    const double *values_for_var(unsigned i) const;
    size_t        num_emitted() const { return m_emitted; }
    // Read-only view of the emitted 2D intervals from the last run() call.
    // Valid until the next run() or run_single_var() call.
    const std::vector<GInterval2D> &emitted_intervals() const { return m_emitted_intervals; }

    // Access the result array of the (only) registered var. Valid until
    // the next run_* call.
    const double *values() const;
    size_t        num_values() const { return m_num_intervals; }

private:
    PMTrackExpression2DVars m_vars;
    std::unique_ptr<PMTrackExpression2DIterator> m_itr;
    size_t                  m_num_intervals{0};
    unsigned                m_eval_buf_limit{0};

    // State for run() / multi-var polymorphic path.
    size_t                  m_emitted{0};
    std::vector<GInterval2D> m_emitted_intervals;
};

#endif /* PMTRACKEXPRESSION2DSCANNER_H_ */
