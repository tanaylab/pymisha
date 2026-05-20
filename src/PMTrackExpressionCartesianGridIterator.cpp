/*
 * PMTrackExpressionCartesianGridIterator.cpp
 *
 * Port of R misha's TrackExpressionCartesianGridIterator.
 *
 * State machine mirrors R's next() closely.  Key members:
 *   m_igp[0/1]       - current grid-point iterators
 *   m_igp_start[0/1] - anchors for the start of the current chromosome
 *   m_iexp[0/1]      - expansion pair upper-bound iterators
 *   m_scope_rect_idx - index into the current pair's scope rect list
 *                      (replaces R's m_iintersection / qtree queue with a
 *                       linear scan; same semantics, O(S) per cell)
 *
 * The "intersection queue" in R processed multiple matches from a single
 * quadtree query.  Here we scan scope rects one-by-one; because we return
 * after each match, the caller just calls next() again to continue from
 * where we left off (m_scope_rect_idx already points past the last match).
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
    const quadtree::DiagonalBand       *band)
    : m_use_band_idx(use_band_idx),
      m_min_band_idx(min_band_idx),
      m_max_band_idx(max_band_idx),
      m_scope(std::move(scope)),
      m_band(band ? *band : quadtree::DiagonalBand{}),
      m_use_band(band != nullptr && band->active),
      m_scope_rect_idx(0),
      m_emitted(0),
      m_gp0_idx(0),
      m_isend(false),
      m_cur_scope_rects(nullptr)
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

    // Build per-pair scope index.
    // When band is active, inter-chrom pairs are excluded entirely.
    for (size_t i = 0; i < m_scope.size(); ++i) {
        const GInterval2D &r = m_scope[i];
        if (r.end1 <= r.start1 || r.end2 <= r.start2)
            continue;
        if (m_use_band && r.chromid1 != r.chromid2)
            continue;
        PairKey key{static_cast<int32_t>(r.chromid1), static_cast<int32_t>(r.chromid2)};
        m_scope_index[key].push_back(i);
    }
}

// ---------------------------------------------------------------------------
// _load_scope_pair: set m_cur_scope_rects for (c1, c2).
// ---------------------------------------------------------------------------

bool PMTrackExpressionCartesianGridIterator::_load_scope_pair(int32_t c1, int32_t c2)
{
    auto it = m_scope_index.find(PairKey{c1, c2});
    if (it == m_scope_index.end()) {
        m_cur_scope_rects = nullptr;
        return false;
    }
    m_cur_scope_rects = &it->second;
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
    m_scope_rect_idx = 0;
    m_emitted        = 0;
    m_gp0_idx        = 0;
    m_cur_chromid[0] = m_cur_chromid[1] = -1;
    m_cur_scope_rects = nullptr;
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
//         SCOPE: linear scan over scope rects for (c1, c2)
//
// On each entry after an emission, m_scope_rect_idx points to the next
// scope rect to try for the current (gp0, gp1, exp0, exp1).  Phase 4
// will continue from there; if exhausted it advances m_iexp[1].
//
// Differences from R:
//   - Linear scope scan vs. StatQuadTree.intersect()
//   - No separate m_intersection vector; we emit one at a time
// ---------------------------------------------------------------------------

void PMTrackExpressionCartesianGridIterator::next()
{
    if (m_isend)
        return;

    while (true) {

        // ------------------------------------------------------------------
        // PHASE 1: If we have a pending scope-rect scan (m_scope_rect_idx > 0
        // meaning we just returned from an emission and need to continue
        // scanning the same cell), skip directly to Phase 4.
        // We detect this by checking whether we should skip the gpoint-advance
        // and filter phases.  Since we always reset m_scope_rect_idx to 0
        // when advancing exp or gpoint iterators, a non-zero value means
        // we're mid-scan for the current cell.
        // ------------------------------------------------------------------
        // (Implemented inline: Phase 4 handles all scope scanning, and on
        //  re-entry after an emission we fall through Phases 2-3 without
        //  triggering their skip conditions, then Phase 4 continues scanning.)

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
            m_scope_rect_idx = 0;
            m_gp0_idx        = static_cast<size_t>(m_igp[0] - m_gpoints[0].begin());
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
                m_scope_rect_idx = 0;
                ++m_igp[1];
                continue;
            }
        }

        // ------------------------------------------------------------------
        // PHASE 4: Compute cell and scan scope rects.
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
                m_scope_rect_idx = 0;
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
                m_scope_rect_idx = 0;
                continue;
            }

            // Reload scope rects when chrom-pair changes (mirrors R lines 164-200).
            if (c1 != m_cur_chromid[0] || c2 != m_cur_chromid[1]) {
                m_cur_chromid[0] = c1;
                m_cur_chromid[1] = c2;
                m_scope_rect_idx = 0;

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

            // Linear scope scan (replaces R's qtree.intersect()).
            bool found = false;
            const std::vector<size_t> &sr_idx = *m_cur_scope_rects;

            while (m_scope_rect_idx < sr_idx.size()) {
                const GInterval2D &sr = m_scope[sr_idx[m_scope_rect_idx]];
                ++m_scope_rect_idx;

                int64_t ix1 = std::max(start1, sr.start1);
                int64_t ix2 = std::min(end1,   sr.end1);
                int64_t iy1 = std::max(start2, sr.start2);
                int64_t iy2 = std::min(end2,   sr.end2);

                if (ix1 >= ix2 || iy1 >= iy2)
                    continue;

                // Apply diagonal band if active and same-chrom (mirrors R lines 205-210).
                if (m_use_band && c1 == c2) {
                    quadtree::Rectangle rc{ix1, iy1, ix2, iy2};
                    if (!m_band.do_intersect(rc))
                        continue;
                    m_band.shrink2intersected(rc);
                    if (rc.x1 >= rc.x2 || rc.y1 >= rc.y2)
                        continue;
                    ix1 = rc.x1; iy1 = rc.y1;
                    ix2 = rc.x2; iy2 = rc.y2;
                }

                m_last.chromid1 = c1;
                m_last.chromid2 = c2;
                m_last.start1   = ix1;
                m_last.end1     = ix2;
                m_last.start2   = iy1;
                m_last.end2     = iy2;
                m_gp0_idx       = static_cast<size_t>(m_igp[0] - m_gpoints[0].begin());
                ++m_emitted;
                found = true;
                break;
            }

            if (found)
                return;

            // All scope rects exhausted for this cell: advance exp[1].
            m_scope_rect_idx = 0;
            ++m_iexp[1];
        }
    }
}
