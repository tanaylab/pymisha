#include "PMTrackExpressionFixedRectIterator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

PMTrackExpressionFixedRectIterator::PMTrackExpressionFixedRectIterator(
    int64_t width, int64_t height,
    std::vector<GInterval2D> scope,
    const quadtree::DiagonalBand *band)
    : m_width(width),
      m_height(height),
      m_scope(std::move(scope)),
      m_band(band ? *band : quadtree::DiagonalBand{}),
      m_use_band(band != nullptr && band->active),
      m_scope_idx(0),
      m_cur_xbin(0), m_cur_ybin(0),
      m_start_xbin(0), m_end_xbin(0), m_end_ybin(0),
      m_minx(0), m_maxx(0), m_miny(0), m_maxy(0),
      m_scope_minx(0), m_scope_maxx(0),
      m_starting_iteration(true),
      m_emitted(0),
      m_isend(false)
{
    if (width <= 0)
        throw std::invalid_argument(
            "FixedRect: bin sizes must be positive (width=" +
            std::to_string(width) + ")");
    if (height <= 0)
        throw std::invalid_argument(
            "FixedRect: bin sizes must be positive (height=" +
            std::to_string(height) + ")");
    // Caller must call begin() after construction; this matches the convention
    // of the other 2D iterators (e.g. PMTrackExpressionIntervals2DIterator).
}

void PMTrackExpressionFixedRectIterator::begin()
{
    m_last = GInterval2D{};
    m_scope_idx = 0;
    m_cur_xbin = m_start_xbin = m_end_xbin = 0;
    m_cur_ybin = m_end_ybin = 0;
    m_minx = m_maxx = m_miny = m_maxy = 0;
    m_row_miny = m_row_maxy = 0;
    m_scope_minx = m_scope_maxx = 0;
    m_starting_iteration = true;
    m_emitted = 0;

    if (m_scope.empty()) {
        m_isend = true;
        return;
    }
    m_isend = false;
    // Prime: advance to the first emitted interval.
    // m_starting_iteration=true means m_scope_idx is already pointing
    // at the first scope rect; the scope-advance section must NOT
    // increment m_scope_idx on its first pass. next() consults this flag.
    next();
}

void PMTrackExpressionFixedRectIterator::next()
{
    if (m_isend) return;

    // This loop mirrors R's TrackExpressionFixedRectIterator::next() exactly.
    while (true) {
        // --- x-loop: emit the next cell in the current row ---
        if (m_cur_xbin < m_end_xbin) {
            int64_t coord = m_cur_xbin * m_width;
            m_last.start1 = std::max(coord, m_minx);
            m_last.end1   = std::min(coord + m_width, m_maxx);

            // Band: shrink the x-cell boundaries if not fully inside the band.
            // Reset y-bounds to the row's y-range before per-cell shrink so each
            // cell is evaluated independently (R parity: lines 48-50 in R source).
            // Only applies on same-chrom pairs; inter-chrom pairs pass through.
            if (m_use_band && m_last.chromid1 == m_last.chromid2) {
                m_last.start2 = m_row_miny;
                m_last.end2   = m_row_maxy;
                quadtree::Rectangle rc{m_last.start1, m_last.start2,
                                       m_last.end1,   m_last.end2};
                if (!m_band.do_contain(rc)) {
                    m_band.shrink2intersected(rc);
                    m_last.start1 = rc.x1;
                    m_last.end1   = rc.x2;
                    m_last.start2 = rc.y1;
                    m_last.end2   = rc.y2;
                }
            }

            ++m_cur_xbin;
            ++m_emitted;
            return;
        }

        // --- y-loop: advance to the next row ---
        while (m_cur_ybin < m_end_ybin) {
            int64_t coord = m_cur_ybin * m_height;
            m_last.start2 = std::max(coord, m_miny);
            m_last.end2   = std::min(coord + m_height, m_maxy);
            ++m_cur_ybin;

            // Cache the row's y-bounds before any band shrink; the x-loop resets to
            // these per cell (R parity: mirrors m_last_interval.start2/end2 in R).
            m_row_miny = m_last.start2;
            m_row_maxy = m_last.end2;

            // Only apply per-row band shrink on same-chrom pairs.
            if (m_use_band && m_last.chromid1 == m_last.chromid2) {
                // Re-shrink x bounds from the original scope x-extents for this row.
                // Uses m_scope_minx/m_scope_maxx (not the prior row's shrunk bounds).
                quadtree::Rectangle r{m_scope_minx, m_last.start2,
                                      m_scope_maxx, m_last.end2};
                if (!m_band.do_intersect(r)) {
                    // This row has no intersection with the band; skip it.
                    continue;
                }
                m_band.shrink2intersected(r);
                m_minx = r.x1;
                m_maxx = r.x2;
                m_start_xbin = (int64_t)(m_minx / (double)m_width);
                m_end_xbin   = (int64_t)std::ceil(m_maxx / (double)m_width);
            }

            m_cur_xbin = m_start_xbin;
            break;
        }

        // Invariant: m_cur_xbin == m_end_xbin until a valid row is found and break executes.
        // If we exited the y-loop without finding one, scope-advance.
        if (m_cur_xbin < m_end_xbin) {
            // A valid row was set up; go back to x-loop.
            continue;
        }

        // --- scope-rect advance ---
        if (m_starting_iteration) {
            m_starting_iteration = false;
        } else {
            ++m_scope_idx;
        }

        // When band is active: inter-chrom scope rects are skipped entirely (R parity).
        // For same-chrom rects, skip those the band doesn't intersect at all.
        while (m_scope_idx < m_scope.size()) {
            const GInterval2D &s = m_scope[m_scope_idx];
            if (m_use_band) {
                if (s.chromid1 != s.chromid2) {
                    // R: inter-chrom + active band -> skip the entire scope rect.
                    ++m_scope_idx;
                    continue;
                }
                quadtree::Rectangle r{s.start1, s.start2, s.end1, s.end2};
                if (!m_band.do_intersect(r)) {
                    // Band doesn't touch this same-chrom scope rect at all; skip it.
                    ++m_scope_idx;
                    continue;
                }
            }
            break;
        }

        if (m_scope_idx >= m_scope.size()) {
            m_isend = true;
            return;
        }

        const GInterval2D &s = m_scope[m_scope_idx];

        // Set chromid fields (carried across all cells of this scope rect).
        m_last.chromid1 = s.chromid1;
        m_last.chromid2 = s.chromid2;

        m_scope_minx = s.start1;
        m_scope_maxx = s.end1;

        m_minx = s.start1; m_maxx = s.end1;
        m_miny = s.start2; m_maxy = s.end2;

        // When band is active on a same-chrom pair, shrink the scope rect's y-range
        // to the band intersection.  Per-row x-shrink is done in the y-loop section;
        // m_scope_minx/m_scope_maxx retain the original unshrunk x-extents for that.
        if (m_use_band && s.chromid1 == s.chromid2) {
            quadtree::Rectangle r{m_minx, m_miny, m_maxx, m_maxy};
            m_band.shrink2intersected(r);
            m_miny = r.y1;
            m_maxy = r.y2;
        }

        m_start_xbin = (int64_t)(m_minx / (double)m_width);
        m_end_xbin   = (int64_t)std::ceil(m_maxx / (double)m_width);
        m_cur_ybin   = (int64_t)(m_miny / (double)m_height);
        m_end_ybin   = (int64_t)std::ceil(m_maxy / (double)m_height);

        // R sets cur_xbin = end_xbin initially so the x-loop is exhausted
        // on the first pass and the y-loop picks up from cur_ybin.
        m_cur_xbin = m_end_xbin;

        // continue -> y-loop runs and sets start2/end2, then x-loop fires.
    }
}
