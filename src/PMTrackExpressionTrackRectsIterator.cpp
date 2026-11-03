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
      m_pair_first(0),
      m_pair_last(0),
      m_emit_idx(0),
      m_emitted(0),
      m_isend(false)
{
    // init() throws TGLException on unknown / non-2D / non-RECTS/POINTS track.
    // We let TGLException propagate; the binding converts it to RuntimeError.
    m_track.init(track_name);

    // R walks the scope sorted by chromosome pair (r_2d_sort_order)
    std::vector<size_t> order = r_2d_sort_order(scope);
    for (size_t i : order) {
        m_scope.push_back(scope[i]);
        m_scope_orig.push_back(i);
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
    m_pair_first = m_pair_last = 0;
    m_emits.clear();
    m_emit_scope.clear();
    m_emit_idx  = 0;
    m_isend     = false;
    next_pair_emissions();
}

// ---------------------------------------------------------------------------
// next()
// ---------------------------------------------------------------------------

void PMTrackExpressionTrackRectsIterator::next()
{
    if (m_isend) return;
    ++m_emit_idx;
    if (m_emit_idx < m_emits.size()) {
        m_last = m_emits[m_emit_idx];
        ++m_emitted;
        return;
    }
    next_pair_emissions();
}

// ---------------------------------------------------------------------------
// next_pair_emissions()
//
// TrackExpressionTrackRectsIterator::next() for the next chromosome pair of the
// scope that gives anything: the track's objects in the order R's quad-tree
// iterator returns them, each cut by the scope intervals of the pair as R's
// scope quad tree returns them (intersect_scope_qtree). Under a band an
// inter-chromosomal pair is skipped. Sets m_isend when no pair is left.
// ---------------------------------------------------------------------------

void PMTrackExpressionTrackRectsIterator::next_pair_emissions()
{
    m_emits.clear();
    m_emit_scope.clear();
    m_emit_idx = 0;

    RectsQuadTree qtree;
    Rectangles intersection;
    std::vector<uint64_t> indices;

    while (m_pair_last < m_scope.size()) {
        m_pair_first = m_pair_last;
        while (m_pair_last < m_scope.size() && m_scope[m_pair_last].is_same_chrom_pair(m_scope[m_pair_first]))
            ++m_pair_last;

        int c1 = m_scope[m_pair_first].chromid1;
        int c2 = m_scope[m_pair_first].chromid2;

        if (m_use_band && c1 != c2)
            continue;

        bool has_data = m_track.set_chrom_pair(c1, c2);

        PMGenomeTrack2D::LookupState ls = m_track.lookup_state();
        if (ls == PMGenomeTrack2D::LOOKUP_OPEN_FAILED) {
            throw std::runtime_error(
                "PMTrackExpressionTrackRectsIterator: failed to open track pair ("
                + std::to_string(c1) + ", " + std::to_string(c2) + ")");
        }

        if (!has_data || ls == PMGenomeTrack2D::LOOKUP_ABSENT)
            continue;

        // A query over the whole pair returns the objects in the order of R's
        // StatQuadTreeCached::Iterator: depth first, quads in order, each object once.
        // Memory note: the objects of a pair are materialised in a vector.
        int64_t cs1 = static_cast<int64_t>(m_chromkey.get_chrom_size(c1));
        int64_t cs2 = static_cast<int64_t>(m_chromkey.get_chrom_size(c2));
        quadtree::QueryObjects objs = m_track.query_objects(0, 0, cs1, cs2, nullptr);
        if (objs.x1s.empty())
            continue;

        build_scope_qtree(qtree, m_scope, m_pair_first, m_pair_last, m_chromkey);

        for (size_t k = 0; k < objs.x1s.size(); ++k) {
            intersect_scope_qtree(qtree, Rectangle(objs.x1s[k], objs.y1s[k], objs.x2s[k], objs.y2s[k]),
                                  m_use_band ? &m_band : nullptr, intersection, indices);
            for (size_t j = 0; j < intersection.size(); ++j) {
                const Rectangle &r = intersection[j];
                m_emits.emplace_back(c1, r.x1, r.x2, c2, r.y1, r.y2);
                m_emit_scope.push_back(m_scope_orig[m_pair_first + indices[j]]);
            }
        }

        if (!m_emits.empty()) {
            m_last = m_emits[0];
            ++m_emitted;
            return;
        }
    }

    m_isend = true;
}
