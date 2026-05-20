/*
 * PMTrackExpressionFixedRectIterator
 *
 * Streaming 2D iterator that subdivides each 2D scope interval into a
 * fixed-size rect grid (width x height) and yields each grid cell.
 * Mirrors R misha's TrackExpressionFixedRectIterator.
 *
 * Walk order within a scope rect: y-outer, x-inner (row-major).
 * Cells at scope boundaries are clipped, never extended past the scope.
 * Band integration (DiagonalBand) is active when `band != nullptr && band->active`.
 */
#ifndef PMTRACKEXPRESSIONFIXEDRECTITERATOR_H_
#define PMTRACKEXPRESSIONFIXEDRECTITERATOR_H_

#include <cstdint>
#include <vector>

#include "GInterval2D.h"
#include "PMTrackExpression2DIterator.h"
#include "QuadTreeReader.h"

// Iterator that subdivides each scope rect into a fixed-size (width x height)
// grid and yields the cells row-major (y outer, x inner), clipped to the
// scope boundaries.  Mirrors R's TrackExpressionFixedRectIterator.
class PMTrackExpressionFixedRectIterator : public PMTrackExpression2DIterator {
public:
    // Construct and prime the iterator (calls begin() internally).
    // Throws std::invalid_argument if width or height <= 0.
    PMTrackExpressionFixedRectIterator(int64_t width,
                                       int64_t height,
                                       std::vector<GInterval2D> scope,
                                       const quadtree::DiagonalBand *band);
    virtual ~PMTrackExpressionFixedRectIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }
    virtual const GInterval2D &last_interval() const override { return m_last; }

    virtual uint64_t idx() const override { return m_emitted; }
    virtual uint64_t original_interval_idx() const override { return m_scope_idx + 1; }

private:
    int64_t                  m_width;
    int64_t                  m_height;
    std::vector<GInterval2D> m_scope;
    quadtree::DiagonalBand   m_band;   // active when m_use_band is true
    bool                     m_use_band;

    // Iterator state (mirrors R's member variables).
    size_t  m_scope_idx;       // index into m_scope
    int64_t m_cur_xbin;        // current x-bin index
    int64_t m_cur_ybin;        // current y-bin index
    int64_t m_start_xbin;      // first x-bin for the current scope rect
    int64_t m_end_xbin;        // one-past-last x-bin for the current scope rect
    int64_t m_end_ybin;        // one-past-last y-bin for the current scope rect
    int64_t m_minx, m_maxx;    // clipped x bounds of the current scope rect (may be
                               // band-shrunk per row; reset per row from m_scope_minx/maxx)
    int64_t m_miny, m_maxy;    // clipped y bounds of the current scope rect (scope-level)
    int64_t m_row_miny;        // current row's y-start (set per row by y-loop; x-loop resets to this)
    int64_t m_row_maxy;        // current row's y-end   (set per row by y-loop; x-loop resets to this)
    // Unshrunk x bounds of the scope rect (mirrors R's m_scope_interv.start1/end1).
    // Per-row band shrink must use these as the base, not the prior row's shrunken bounds.
    int64_t m_scope_minx;      // scope rect's original start1
    int64_t m_scope_maxx;      // scope rect's original end1
    bool    m_starting_iteration; // true until the first scope rect is consumed

    GInterval2D m_last;        // the most recently emitted interval
    uint64_t    m_emitted;     // count of emitted intervals
    bool        m_isend;
};

#endif /* PMTRACKEXPRESSIONFIXEDRECTITERATOR_H_ */
