/*
 * PMTrackExpression2DIterator.h
 *
 * Iterator classes for 2D track expression evaluation in pymisha.
 * Parallel to PMTrackExpressionIterator.h (which is 1D-throughout).
 */

#ifndef PMTRACKEXPRESSION2DITERATOR_H_
#define PMTRACKEXPRESSION2DITERATOR_H_

#include <vector>
#include "GInterval2D.h"

// Base class for all 2D track expression iterators.
class PMTrackExpression2DIterator {
public:
    virtual ~PMTrackExpression2DIterator() {}

    virtual void begin() = 0;
    virtual void next() = 0;
    virtual bool isend() const = 0;

    virtual const GInterval2D &last_interval() const = 0;

    // Estimated number of intervals to be emitted (0 if unknown).
    virtual uint64_t size() const { return 0; }

    // 0-based index of the current emission (for progress reporting).
    virtual uint64_t idx() const { return 0; }

    // 1-based original-input-row index of the current emission.
    virtual uint64_t original_interval_idx() const { return 0; }

    // Tag method: always true for 2D iterators (mirrors 1D base's is_1d()).
    virtual bool is_2d() const { return true; }
};

// Iterator that emits each input 2D interval exactly once, in input order.
// Counterpart to PMIntervalsIterator (1D).
class PMTrackExpressionIntervals2DIterator : public PMTrackExpression2DIterator {
public:
    PMTrackExpressionIntervals2DIterator(const std::vector<GInterval2D> &intervals);
    virtual ~PMTrackExpressionIntervals2DIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval2D &last_interval() const override {
        return m_intervals[m_cur_idx];
    }

    virtual uint64_t size() const override { return m_intervals.size(); }
    virtual uint64_t idx() const override { return m_cur_idx; }
    virtual uint64_t original_interval_idx() const override { return m_cur_idx + 1; }

private:
    std::vector<GInterval2D> m_intervals;
    size_t                   m_cur_idx;
    bool                     m_isend;
};

#endif /* PMTRACKEXPRESSION2DITERATOR_H_ */
