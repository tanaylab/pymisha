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
#include <vector>

#include "GInterval2D.h"
#include "PMTrackExpression2DIterator.h"
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

    // Access the result array of the (only) registered var. Valid until
    // the next run_* call.
    const double *values() const;
    size_t        num_values() const { return m_num_intervals; }

private:
    PMTrackExpression2DVars m_vars;
    std::unique_ptr<PMTrackExpression2DIterator> m_itr;
    size_t                  m_num_intervals{0};
    unsigned                m_eval_buf_limit{0};
};

#endif /* PMTRACKEXPRESSION2DSCANNER_H_ */
