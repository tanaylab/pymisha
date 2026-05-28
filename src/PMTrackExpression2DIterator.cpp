/*
 * PMTrackExpression2DIterator.cpp
 */

#include "PMTrackExpression2DIterator.h"

PMTrackExpressionIntervals2DIterator::PMTrackExpressionIntervals2DIterator(
    const std::vector<GInterval2D> &intervals,
    const quadtree::DiagonalBand *band)
    : m_intervals(intervals),
      m_cur_idx(0),
      m_scan_idx(0),
      m_isend(intervals.empty()),
      m_use_band(band != nullptr && band->active),
      m_band(band != nullptr ? *band : quadtree::DiagonalBand()),
      m_emit()
{
    advance_to_next_emit();
}

void PMTrackExpressionIntervals2DIterator::begin()
{
    m_scan_idx = 0;
    m_cur_idx = 0;
    m_isend = m_intervals.empty();
    advance_to_next_emit();
}

void PMTrackExpressionIntervals2DIterator::next()
{
    if (m_isend) return;
    ++m_scan_idx;
    advance_to_next_emit();
}

void PMTrackExpressionIntervals2DIterator::advance_to_next_emit()
{
    if (m_intervals.empty()) {
        m_isend = true;
        return;
    }

    if (!m_use_band) {
        // Fast path: no band, no filtering, emit as-is.
        if (m_scan_idx >= m_intervals.size()) {
            m_isend = true;
            return;
        }
        m_cur_idx = m_scan_idx;
        m_emit    = m_intervals[m_cur_idx];
        m_isend   = false;
        return;
    }

    // Band-active path: skip inter-chrom and non-intersecting same-chrom rects,
    // shrink survivors to the band-intersected bounding box (R parity).
    while (m_scan_idx < m_intervals.size()) {
        const GInterval2D &s = m_intervals[m_scan_idx];

        // R parity: under an active band, inter-chrom rects are skipped entirely
        // (mirrors PMTrackExpressionFixedRectIterator.cpp lines 147-151).
        if (s.chromid1 != s.chromid2) {
            ++m_scan_idx;
            continue;
        }

        quadtree::Rectangle r{s.start1, s.start2, s.end1, s.end2};
        if (!m_band.do_intersect(r)) {
            ++m_scan_idx;
            continue;
        }
        m_band.shrink2intersected(r);

        m_cur_idx = m_scan_idx;
        m_emit = GInterval2D(s.chromid1, r.x1, r.x2, s.chromid2, r.y1, r.y2);
        m_isend = false;
        return;
    }

    m_isend = true;
}
