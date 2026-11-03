/*
 * PMTrackExpressionCartesianGridIterator.cpp
 *
 * Port of R misha's TrackExpressionCartesianGridIterator.
 *
 * State machine mirrors R's next() closely.  Key members:
 *   m_igp[0/1]       - current grid-point iterators
 *   m_igp_start[0/1] - anchors for the start of the current chromosome
 *   m_iexp[0/1]      - expansion pair upper-bound iterators
 *   m_cell_inter     - the pieces of the current cell still to emit (R's
 *                      m_intersection / m_iintersection), from a quad tree of
 *                      the pair's scope intervals as R builds it
 */

#include "PMTrackExpressionCartesianGridIterator.h"

#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// build_grid_points  (static)
// Mirrors R's grid-point construction loop (begin() lines 60-84).
// ---------------------------------------------------------------------------

std::vector<PMTrackExpressionCartesianGridIterator::GridPoint>
PMTrackExpressionCartesianGridIterator::build_grid_points(
    const std::vector<GridPoint>  &raw_centers,
    const std::vector<int64_t>    &expansion,
    const GenomeChromKey          &chromkey)
{
    if (expansion.size() < 2)
        throw std::runtime_error("Iterator grid expansion must contain at least 2 unique values");

    std::vector<GridPoint> gps = raw_centers;
    std::sort(gps.begin(), gps.end());
    gps.erase(std::unique(gps.begin(), gps.end()), gps.end());

    int64_t exp_front = expansion.front();
    int64_t exp_back  = expansion.back();

    for (size_t i = 0; i < gps.size(); ++i) {
        GridPoint &gp = gps[i];

        if (i > 0 && gps[i - 1].chromid == gp.chromid) {
            GridPoint &prev = gps[i - 1];

            // R line 64: do the maximal expansions overlap?
            if (gp.coord + exp_front < prev.coord + exp_back) {
                int64_t mid_coord = (int64_t)((gp.coord + prev.coord) * 0.5);

                if (gp.coord + exp_front < mid_coord) {
                    if (prev.coord + exp_back > mid_coord) {
                        // R lines 70-71: split at midpoint
                        prev.max_expansion = mid_coord - prev.coord;
                        gp.min_expansion   = mid_coord - gp.coord;
                    } else {
                        // R lines 73-74
                        prev.max_expansion = exp_back;
                        gp.min_expansion   = prev.coord - gp.coord + exp_back;
                    }
                } else {
                    // R lines 77-78
                    prev.max_expansion = gp.coord - prev.coord + exp_front;
                    gp.min_expansion   = exp_front;
                }
            } else {
                // R line 81: no overlap; extend to chrom start
                gp.min_expansion = -gp.coord;
            }
        } else {
            // First gpoint on this chromosome.
            gp.min_expansion = -gp.coord;
        }

        // R line 83: upper bound is chrom size.
        gp.max_expansion = static_cast<int64_t>(chromkey.get_chrom_size(gp.chromid)) - gp.coord;
    }

    return gps;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

PMTrackExpressionCartesianGridIterator::PMTrackExpressionCartesianGridIterator(
    std::vector<GridPoint>              gp_axis0,
    std::vector<GridPoint>              gp_axis1,
    std::vector<int64_t>                expansion0,
    std::vector<int64_t>                expansion1,
    bool                                use_band_idx,
    int32_t                             min_band_idx,
    int32_t                             max_band_idx,
    std::vector<GInterval2D>            scope,
    const GenomeChromKey               &chromkey,
    const quadtree::DiagonalBand       *band)
    : m_use_band_idx(use_band_idx),
      m_min_band_idx(min_band_idx),
      m_max_band_idx(max_band_idx),
      m_chromkey(chromkey),
      m_band(band ? *band : quadtree::DiagonalBand{}),
      m_use_band(band != nullptr && band->active),
      m_scope_first(0),
      m_cell_pos(0),
      m_scope_pos(0),
      m_emitted(0),
      m_gp0_idx(0),
      m_isend(false)
{
    m_gpoints[0] = std::move(gp_axis0);
    m_gpoints[1] = std::move(gp_axis1);
    m_expansion[0] = std::move(expansion0);
    m_expansion[1] = std::move(expansion1);

    m_cur_chromid[0] = m_cur_chromid[1] = -1;

    if (m_expansion[0].size() < 2)
        throw std::runtime_error("Iterator grid expansion must contain at least 2 unique values");
    if (m_expansion[1].size() < 2)
        throw std::runtime_error("Iterator grid expansion must contain at least 2 unique values");

    // R walks the scope sorted by chromosome pair (r_2d_sort_order).
    m_scope_orig = r_2d_sort_order(scope);
    for (size_t i : m_scope_orig)
        m_scope.push_back(scope[i]);

    // Build per-pair scope index.
    // When band is active, inter-chrom pairs are excluded entirely.
    for (size_t first = 0, last; first < m_scope.size(); first = last) {
        for (last = first; last < m_scope.size() && m_scope[last].is_same_chrom_pair(m_scope[first]); ++last)
            ;
        const GInterval2D &r = m_scope[first];
        if (m_use_band && r.chromid1 != r.chromid2)
            continue;
        PairKey key{static_cast<int32_t>(r.chromid1), static_cast<int32_t>(r.chromid2)};
        m_scope_index[key] = std::make_pair(first, last);
    }
}

// ---------------------------------------------------------------------------
// _load_scope_pair: set m_cur_scope_rects for (c1, c2).
// ---------------------------------------------------------------------------

bool PMTrackExpressionCartesianGridIterator::_load_scope_pair(int32_t c1, int32_t c2)
{
    auto it = m_scope_index.find(PairKey{c1, c2});
    if (it == m_scope_index.end())
        return false;
    m_scope_first = it->second.first;
    build_scope_qtree(m_scope_qtree, m_scope, it->second.first, it->second.second, m_chromkey);
    return true;
}

// ---------------------------------------------------------------------------
// begin()
// ---------------------------------------------------------------------------

void PMTrackExpressionCartesianGridIterator::begin()
{
    m_igp[0]         = m_gpoints[0].begin();
    m_igp[1]         = m_gpoints[1].begin();
    m_igp_start[0]   = m_gpoints[0].begin();
    m_igp_start[1]   = m_gpoints[1].begin();
    m_iexp[0]        = m_expansion[0].begin() + 1;
    m_iexp[1]        = m_expansion[1].begin() + 1;
    m_cell_inter.clear();
    m_cell_idx.clear();
    m_cell_pos       = 0;
    m_emitted        = 0;
    m_gp0_idx        = 0;
    m_cur_chromid[0] = m_cur_chromid[1] = -1;
    m_last           = GInterval2D{};

    if (m_gpoints[0].empty() || m_gpoints[1].empty() || m_scope_index.empty()) {
        m_isend = true;
        return;
    }

    m_isend = false;
    // Prime: advance to the first valid emission (mirrors R: begin() returns next()).
    next();
}

// ---------------------------------------------------------------------------
// next()
//
// State machine port of R's TrackExpressionCartesianGridIterator::next().
//
// The loop structure:
//   OUTER: iterate over (gp0, gp1) pairs grouped by chromosome
//     MIDDLE: iterate over expansion[0] values (m_iexp[0])
//       INNER: iterate over expansion[1] values (m_iexp[1])
//         SCOPE: the cell's pieces from the pair's scope quad tree
// ---------------------------------------------------------------------------

void PMTrackExpressionCartesianGridIterator::next()
{
    if (m_isend)
        return;

    while (true) {

        // ------------------------------------------------------------------
        // PHASE 1: emit the next piece of the current cell (R lines 103-113).
        // ------------------------------------------------------------------
        if (m_cell_pos < m_cell_inter.size()) {
            const Rectangle &r = m_cell_inter[m_cell_pos];
            m_last.chromid1 = m_igp[0]->chromid;
            m_last.chromid2 = m_igp[1]->chromid;
            m_last.start1   = r.x1;
            m_last.end1     = r.x2;
            m_last.start2   = r.y1;
            m_last.end2     = r.y2;
            m_scope_pos     = m_scope_first + m_cell_idx[m_cell_pos];
            m_gp0_idx       = static_cast<size_t>(m_igp[0] - m_gpoints[0].begin());
            ++m_cell_pos;
            if (m_cell_pos == m_cell_inter.size()) {
                m_cell_inter.clear();
                m_cell_pos = 0;
                ++m_iexp[1];
            }
            ++m_emitted;
            return;
        }

        // ------------------------------------------------------------------
        // PHASE 2: Advance gpoint iterators (mirrors R lines 115-133).
        // This fires when axis-1 has moved off its current chromosome.
        // ------------------------------------------------------------------
        if (m_igp[1] == m_gpoints[1].end() ||
            m_igp[1]->chromid != m_igp_start[1]->chromid)
        {
            ++m_igp[0];

            if (m_igp[0] != m_gpoints[0].end() &&
                m_igp[0]->chromid == m_igp_start[0]->chromid)
            {
                // Same chrom on axis-0: rewind axis-1 to start of its chrom.
                m_igp[1] = m_igp_start[1];
            } else {
                if (m_igp[1] == m_gpoints[1].end()) {
                    if (m_igp[0] == m_gpoints[0].end()) {
                        m_isend = true;
                        return;
                    }
                    // Axis-1 fully exhausted: restart from beginning.
                    m_igp_start[0] = m_igp[0];
                    m_igp[1] = m_igp_start[1] = m_gpoints[1].begin();
                } else {
                    // Axis-0 moved to a new chrom: advance axis-1's chrom anchor.
                    m_igp[0] = m_igp_start[0];
                    m_igp_start[1] = m_igp[1];
                }
            }

            // Reset inner state for new gp pair.
            m_iexp[0]        = m_expansion[0].begin() + 1;
            m_iexp[1]        = m_expansion[1].begin() + 1;
            continue;
        }

        // ------------------------------------------------------------------
        // PHASE 3: Band/band_idx filter (mirrors R lines 136-144).
        // Skip this (gp0, gp1) pair entirely.
        // ------------------------------------------------------------------
        {
            int32_t c1 = m_igp[0]->chromid;
            int32_t c2 = m_igp[1]->chromid;
            int delta_idx = static_cast<int>(m_igp[0] - m_gpoints[0].begin()) -
                            static_cast<int>(m_igp[1] - m_gpoints[1].begin());

            bool skip =
                (m_iexp[0] == m_expansion[0].end()) ||
                (c1 != c2 && (m_use_band || m_use_band_idx)) ||
                (m_use_band_idx && (delta_idx < m_min_band_idx || delta_idx > m_max_band_idx));

            if (skip) {
                m_iexp[0]        = m_expansion[0].begin() + 1;
                m_iexp[1]        = m_expansion[1].begin() + 1;
                ++m_igp[1];
                continue;
            }
        }

        // ------------------------------------------------------------------
        // PHASE 4: Compute the cell and intersect it with the scope.
        // ------------------------------------------------------------------
        {
            int32_t c1 = m_igp[0]->chromid;
            int32_t c2 = m_igp[1]->chromid;

            // Compute axis-0 window (mirrors R lines 147-153).
            int64_t start1 = m_igp[0]->coord +
                std::max(*(m_iexp[0] - 1), m_igp[0]->min_expansion);
            int64_t end1   = m_igp[0]->coord +
                std::min(*m_iexp[0], m_igp[0]->max_expansion);

            if (m_iexp[1] == m_expansion[1].end() || start1 == end1) {
                m_iexp[1]        = m_expansion[1].begin() + 1;
                ++m_iexp[0];
                continue;
            }

            // Compute axis-1 window (mirrors R lines 156-162).
            int64_t start2 = m_igp[1]->coord +
                std::max(*(m_iexp[1] - 1), m_igp[1]->min_expansion);
            int64_t end2   = m_igp[1]->coord +
                std::min(*m_iexp[1], m_igp[1]->max_expansion);

            if (start2 == end2) {
                ++m_iexp[1];
                continue;
            }

            // Rebuild the scope quad tree when the chrom-pair changes (mirrors R lines 164-200).
            if (c1 != m_cur_chromid[0] || c2 != m_cur_chromid[1]) {
                m_cur_chromid[0] = c1;
                m_cur_chromid[1] = c2;

                if (!_load_scope_pair(c1, c2)) {
                    // No scope rects for this pair: skip entire axis-1 chrom.
                    m_iexp[0]        = m_expansion[0].begin() + 1;
                    m_iexp[1]        = m_expansion[1].begin() + 1;
                    while (m_igp[1] != m_gpoints[1].end() && m_igp[1]->chromid == c2)
                        ++m_igp[1];
                    m_igp_start[1]   = m_igp[1];
                    m_cur_chromid[0] = m_cur_chromid[1] = -1;
                    continue;
                }
            }

            // R lines 203-215: the band only where it does not contain the cell.
            Rectangle r(start1, start2, end1, end2);
            quadtree::Rectangle qr{start1, start2, end1, end2};
            m_cell_inter.clear();
            m_cell_idx.clear();
            m_cell_pos = 0;
            if (!m_use_band || m_band.do_contain(qr))
                intersect_scope_qtree(m_scope_qtree, r, nullptr, m_cell_inter, m_cell_idx);
            else if (m_band.do_intersect(qr))
                intersect_scope_qtree(m_scope_qtree, r, &m_band, m_cell_inter, m_cell_idx);

            if (m_cell_inter.empty())
                ++m_iexp[1];
        }
    }
}
