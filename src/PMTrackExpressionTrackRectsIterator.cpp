#include "PMTrackExpressionTrackRectsIterator.h"

#include <algorithm>
#include <stdexcept>

#include <string>
// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

PMTrackExpressionTrackRectsIterator::PMTrackExpressionTrackRectsIterator(
    const std::string            &track_name,
    std::vector<GInterval2D>      scope,
    const GenomeChromKey         &chromkey,
    const quadtree::DiagonalBand *band)
    : m_chromkey(chromkey),
      m_band(band ? *band : quadtree::DiagonalBand{}),
      m_use_band(band != nullptr && band->active),
      m_obj_idx(0),
      m_scope_in_pair_idx(0),
      m_scope_global_idx(0),
      m_emitted(0),
      m_isend(false)
{
    // init() throws TGLException on unknown / non-2D / non-RECTS/POINTS track.
    // We let TGLException propagate; the binding converts it to RuntimeError.
    m_track.init(track_name);

    // Group scope rects by (chromid1, chromid2).
    // When band is active, inter-chrom pairs are skipped entirely (R semantics).
    for (size_t i = 0; i < scope.size(); ++i) {
        const GInterval2D &r = scope[i];
        // Skip zero-area rects.
        if (r.end1 <= r.start1 || r.end2 <= r.start2)
            continue;
        // Under active band, skip inter-chrom scope rects.
        if (m_use_band && r.chromid1 != r.chromid2)
            continue;
        PairKey key{r.chromid1, r.chromid2};
        m_pair_groups[key].emplace_back(i, r);
    }

    // ctor does NOT prime; caller calls begin().
}

// ---------------------------------------------------------------------------
// begin()
// ---------------------------------------------------------------------------

void PMTrackExpressionTrackRectsIterator::begin()
{
    m_last      = GInterval2D{};
    m_emitted   = 0;
    m_obj_idx   = 0;
    m_scope_in_pair_idx = 0;
    m_scope_global_idx  = 0;
    m_obj_x1.clear();
    m_obj_y1.clear();
    m_obj_x2.clear();
    m_obj_y2.clear();

    m_pair_it = m_pair_groups.begin();

    if (m_pair_groups.empty()) {
        m_isend = true;
        return;
    }

    m_isend = false;
    // Advance to the first pair that has track data.
    if (!advance_to_next_pair()) {
        m_isend = true;
        return;
    }
    // Find the first valid emission.
    if (!find_next_in_pair()) {
        // First pair has objects but none intersect the scope rects (or
        // advance_to_next_pair already walked past it); next() will continue.
        next();
    }
}

// ---------------------------------------------------------------------------
// next()
// ---------------------------------------------------------------------------

void PMTrackExpressionTrackRectsIterator::next()
{
    if (m_isend) return;

    // Try to advance within the current pair.
    // find_next_in_pair() starts from the current (m_obj_idx, m_scope_in_pair_idx)
    // but we need to move past the current emission first.
    // After an emission: increment m_scope_in_pair_idx, then search forward.
    ++m_scope_in_pair_idx;
    if (find_next_in_pair())
        return;

    // Current pair exhausted.  advance_to_next_pair() walks m_pair_it forward
    // past absent pairs itself, so we must NOT double-increment here.
    // Pattern: pre-increment once, then let advance_to_next_pair do the rest.
    ++m_pair_it;
    while (m_pair_it != m_pair_groups.end()) {
        if (advance_to_next_pair()) {
            // advance_to_next_pair left m_pair_it at a pair with data.
            if (find_next_in_pair())
                return;
            // Has objects but none intersect the scope rects for this pair.
            // Advance past this pair and keep looking.
            ++m_pair_it;
        }
        // advance_to_next_pair returned false -> it walked m_pair_it to end();
        // the while guard will exit on the next iteration check.
    }

    m_isend = true;
}

// ---------------------------------------------------------------------------
// advance_to_next_pair()
//
// Opens m_track for the pair at m_pair_it, materialises all objects.
// Resets m_obj_idx = 0, m_scope_in_pair_idx = 0.
// Returns true if the pair has at least one object.
// ---------------------------------------------------------------------------

bool PMTrackExpressionTrackRectsIterator::advance_to_next_pair()
{
    while (m_pair_it != m_pair_groups.end()) {
        int c1 = m_pair_it->first.first;
        int c2 = m_pair_it->first.second;

        bool has_data = m_track.set_chrom_pair(c1, c2);

        PMGenomeTrack2D::LookupState ls = m_track.lookup_state();
        if (ls == PMGenomeTrack2D::LOOKUP_OPEN_FAILED) {
            throw std::runtime_error(
                "PMTrackExpressionTrackRectsIterator: failed to open track pair ("
                + std::to_string(c1) + ", " + std::to_string(c2) + ")");
        }

        if (!has_data || ls == PMGenomeTrack2D::LOOKUP_ABSENT) {
            // Pair genuinely absent — skip.
            ++m_pair_it;
            continue;
        }

        // Enumerate all objects on this pair by querying with the full chrom extent.
        // Memory note: the result is materialised in a vector. For very large Hi-C
        // tracks this could be gigabytes — streaming enumeration is a deferred
        // optimisation (TODO).
        int64_t cs1 = static_cast<int64_t>(m_chromkey.get_chrom_size(c1));
        int64_t cs2 = static_cast<int64_t>(m_chromkey.get_chrom_size(c2));
        quadtree::QueryObjects objs = m_track.query_objects(0, 0, cs1, cs2, nullptr);

        size_t n = objs.x1s.size();
        m_obj_x1.assign(objs.x1s.begin(), objs.x1s.end());
        m_obj_y1.assign(objs.y1s.begin(), objs.y1s.end());
        m_obj_x2.assign(objs.x2s.begin(), objs.x2s.end());
        m_obj_y2.assign(objs.y2s.begin(), objs.y2s.end());

        m_obj_idx = 0;
        m_scope_in_pair_idx = 0;

        if (n == 0) {
            ++m_pair_it;
            continue;
        }

        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// find_next_in_pair()
//
// Searches from the current (m_obj_idx, m_scope_in_pair_idx) for the next
// non-empty intersection.  Updates m_last and m_scope_global_idx.
// Returns true on success, false when all objects x scope-rects are exhausted.
// ---------------------------------------------------------------------------

bool PMTrackExpressionTrackRectsIterator::find_next_in_pair()
{
    int c1 = m_pair_it->first.first;
    int c2 = m_pair_it->first.second;
    const auto &pair_scope = m_pair_it->second;  // vector<pair<size_t, GInterval2D>>
    size_t n_objs  = m_obj_x1.size();
    size_t n_scope = pair_scope.size();

    while (m_obj_idx < n_objs) {
        int64_t ox1 = m_obj_x1[m_obj_idx];
        int64_t oy1 = m_obj_y1[m_obj_idx];
        int64_t ox2 = m_obj_x2[m_obj_idx];
        int64_t oy2 = m_obj_y2[m_obj_idx];

        while (m_scope_in_pair_idx < n_scope) {
            const GInterval2D &sr = pair_scope[m_scope_in_pair_idx].second;

            // Compute intersection.
            int64_t ix1 = std::max(ox1, sr.start1);
            int64_t ix2 = std::min(ox2, sr.end1);
            int64_t iy1 = std::max(oy1, sr.start2);
            int64_t iy2 = std::min(oy2, sr.end2);

            if (ix1 < ix2 && iy1 < iy2) {
                // Non-empty intersection found; apply band if active.
                if (m_use_band && c1 == c2) {
                    quadtree::Rectangle rc{ix1, iy1, ix2, iy2};
                    if (!m_band.do_intersect(rc)) {
                        ++m_scope_in_pair_idx;
                        continue;
                    }
                    m_band.shrink2intersected(rc);
                    ix1 = rc.x1; iy1 = rc.y1;
                    ix2 = rc.x2; iy2 = rc.y2;
                    if (ix1 >= ix2 || iy1 >= iy2) {
                        ++m_scope_in_pair_idx;
                        continue;
                    }
                }

                m_last.chromid1 = c1;
                m_last.chromid2 = c2;
                m_last.start1   = ix1;
                m_last.end1     = ix2;
                m_last.start2   = iy1;
                m_last.end2     = iy2;
                m_scope_global_idx = pair_scope[m_scope_in_pair_idx].first;
                ++m_emitted;
                return true;
            }

            ++m_scope_in_pair_idx;
        }

        // Move to the next object; reset scope-rect pointer.
        ++m_obj_idx;
        m_scope_in_pair_idx = 0;
    }

    return false;
}
