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

PMTrackExpressionIntervals2DIterator::PMTrackExpressionIntervals2DIterator(
    const std::vector<GInterval2D> &intervals,
    const std::vector<GInterval2D> &scope,
    const GenomeChromKey &chromkey,
    const quadtree::DiagonalBand *band)
    : m_with_scope(true),
      m_cur_idx(0),
      m_scan_idx(0),
      m_isend(true),
      m_use_band(band != nullptr && band->active),
      m_band(band != nullptr ? *band : quadtree::DiagonalBand()),
      m_emit()
{
    // TrackExpressionIntervals2DIterator::next(), run to the end
    std::vector<size_t> iorder = r_2d_sort_order(intervals);
    std::vector<size_t> sorder = r_2d_sort_order(scope);
    RectsQuadTree qtree;
    Rectangles intersection;
    std::vector<uint64_t> indices;
    size_t scope_first = 0, scope_last = 0;  // the scope intervals (in sorder) of the current pair
    bool have_pair = false;

    for (size_t k = 0; k < iorder.size(); ++k) {
        const GInterval2D &iv = intervals[iorder[k]];

        if (m_use_band && iv.chromid1 != iv.chromid2)
            continue;

        if (!have_pair || !iv.is_same_chrom_pair(intervals[iorder[k - 1]])) {
            have_pair = true;
            scope_first = scope_last = 0;
            while (scope_first < sorder.size()) {
                const GInterval2D &s = scope[sorder[scope_first]];
                if (s.chromid1 > iv.chromid1 || (s.chromid1 == iv.chromid1 && s.chromid2 >= iv.chromid2))
                    break;
                ++scope_first;
            }
            scope_last = scope_first;
            while (scope_last < sorder.size() && scope[sorder[scope_last]].is_same_chrom_pair(iv))
                ++scope_last;
            if (scope_first < scope_last) {
                std::vector<GInterval2D> pair_scope;
                for (size_t i = scope_first; i < scope_last; ++i)
                    pair_scope.push_back(scope[sorder[i]]);
                build_scope_qtree(qtree, pair_scope, 0, pair_scope.size(), chromkey);
            }
        }

        if (scope_first == scope_last)
            continue;

        intersect_scope_qtree(qtree, Rectangle(iv.start1, iv.start2, iv.end1, iv.end2), band, intersection, indices);
        for (size_t j = 0; j < intersection.size(); ++j) {
            const Rectangle &r = intersection[j];
            m_emits.emplace_back(iv.chromid1, r.x1, r.x2, iv.chromid2, r.y1, r.y2);
            m_emit_scope.push_back(sorder[scope_first + indices[j]]);
        }
    }
    begin();
}

void PMTrackExpressionIntervals2DIterator::begin()
{
    m_scan_idx = 0;
    m_cur_idx = 0;
    if (m_with_scope) {
        m_isend = m_emits.empty();
        if (!m_isend)
            m_emit = m_emits[0];
        return;
    }
    m_isend = m_intervals.empty();
    advance_to_next_emit();
}

void PMTrackExpressionIntervals2DIterator::next()
{
    if (m_isend) return;
    if (m_with_scope) {
        if (++m_cur_idx >= m_emits.size())
            m_isend = true;
        else
            m_emit = m_emits[m_cur_idx];
        return;
    }
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
