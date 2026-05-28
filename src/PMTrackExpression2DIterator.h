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
#include "QuadTreeReader.h"

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
//
// R parity (TrackExpressionIntervals2DIterator + DiagonalBand):
//   - When a band is active and the input interval is on an *inter-chromosomal*
//     pair, the interval is skipped entirely (matches FixedRect's parity at
//     PMTrackExpressionFixedRectIterator.cpp:147-151).
//   - When a band is active and the pair is same-chrom but the band doesn't
//     intersect the rect at all, the interval is skipped.
//   - Surviving same-chrom intervals are emitted with their rect *shrunk to
//     the band-intersected bounding box* (DiagonalBand::shrink2intersected).
class PMTrackExpressionIntervals2DIterator : public PMTrackExpression2DIterator {
public:
    PMTrackExpressionIntervals2DIterator(
        const std::vector<GInterval2D> &intervals,
        const quadtree::DiagonalBand *band = nullptr);
    virtual ~PMTrackExpressionIntervals2DIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval2D &last_interval() const override {
        return m_emit;
    }

    virtual uint64_t size() const override { return m_intervals.size(); }
    virtual uint64_t idx() const override { return m_cur_idx; }
    virtual uint64_t original_interval_idx() const override { return m_cur_idx + 1; }

private:
    // Position m_cur_idx on the next interval (from m_scan_idx) that survives
    // the band, applying shrink2intersected; sets m_isend when exhausted.
    void advance_to_next_emit();

    std::vector<GInterval2D> m_intervals;
    size_t                   m_cur_idx;     // index of the current emission
    size_t                   m_scan_idx;    // next candidate to consider
    bool                     m_isend;
    bool                     m_use_band;
    quadtree::DiagonalBand   m_band;        // copy (safe under move-only sources)
    GInterval2D              m_emit;        // current (possibly shrunk) emission
};

#endif /* PMTRACKEXPRESSION2DITERATOR_H_ */
