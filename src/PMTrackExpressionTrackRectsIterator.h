/*
 * PMTrackExpressionTrackRectsIterator.h
 *
 * 2D iterator that walks stored track objects (RECTS or POINTS) and emits
 * one GInterval2D per (track_object, scope_rect) intersection.
 *
 * Semantics mirror R misha's TrackExpressionTrackRectsIterator:
 *   - Group scope rects by (chromid1, chromid2).
 *   - For each pair-group: open the track pair, enumerate all objects,
 *     linearly scan scope rects, clip intersections, emit non-empty ones.
 *   - When band is active, inter-chrom pairs in scope are skipped entirely.
 *   - Same-chrom intersections under an active band get band-shrunk.
 *
 * Implementation notes:
 *   - Per-pair object enumeration materialises all objects into a vector via
 *     PMGenomeTrack2D::query_objects() with a full-chrom-size rect. This is
 *     simple and correct; streaming enumeration is a deferred optimisation.
 *   - Linear scope scan per object is O(S) per object. A quadtree over scope
 *     (like R uses) is a deferred optimisation for large S.
 *   - Constructor does NOT prime the iterator; caller must call begin() first.
 */

#ifndef PMTRACKEXPRESSIONTRACKRECTSITERATOR_H_
#define PMTRACKEXPRESSIONTRACKRECTSITERATOR_H_

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "GInterval2D.h"
#include "GenomeChromKey.h"
#include "PMGenomeTrack2D.h"
#include "PMTrackExpression2DIterator.h"
#include "QuadTreeReader.h"

class PMTrackExpressionTrackRectsIterator : public PMTrackExpression2DIterator {
public:
    // track_name:  physical track name (must be a RECTS or POINTS 2D track).
    // scope:       user-supplied scope rectangles; may contain multiple chrom pairs.
    // chromkey:    for chrom-size lookup when enumerating all objects per pair.
    // band:        nullptr means no band; non-null activates diagonal filtering.
    //
    // Throws std::runtime_error if the track is unknown, not a 2D track, or
    // not of RECTS/POINTS type.
    PMTrackExpressionTrackRectsIterator(const std::string           &track_name,
                                        std::vector<GInterval2D>     scope,
                                        const GenomeChromKey        &chromkey,
                                        const quadtree::DiagonalBand *band);

    virtual ~PMTrackExpressionTrackRectsIterator() {}

    // Must be called before the first isend()/last_interval()/next() call.
    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }
    virtual const GInterval2D &last_interval() const override { return m_last; }

    virtual uint64_t idx() const override { return m_emitted; }
    // 1-based index of the scope rect that produced the current emission.
    virtual uint64_t original_interval_idx() const override { return m_scope_global_idx + 1; }

private:
    // Key type for pair-grouped scope map.
    using PairKey = std::pair<int, int>;

    PMGenomeTrack2D              m_track;      // opened once, pairs switched lazily
    const GenomeChromKey        &m_chromkey;
    quadtree::DiagonalBand       m_band;
    bool                         m_use_band;

    // Original scope kept in order so original_interval_idx() is meaningful.
    // Entries are (global_index_in_scope, GInterval2D).
    std::vector<std::pair<size_t, GInterval2D>>        m_scope_ordered;

    // Per-pair groups. Each entry is a list of (global_idx, GInterval2D).
    // Keys are iterated in insertion order (std::map orders by key).
    std::map<PairKey, std::vector<std::pair<size_t, GInterval2D>>> m_pair_groups;

    // Iterator state.
    std::map<PairKey, std::vector<std::pair<size_t, GInterval2D>>>::iterator m_pair_it;

    // Objects materialised from the current pair.
    std::vector<int64_t> m_obj_x1, m_obj_y1, m_obj_x2, m_obj_y2;
    size_t               m_obj_idx;    // index into m_obj_*

    // Per-object: index into the current pair's scope-rect list.
    size_t               m_scope_in_pair_idx;

    // Tracking for original_interval_idx().
    size_t               m_scope_global_idx;

    GInterval2D m_last;
    uint64_t    m_emitted;
    bool        m_isend;

    // Advance to the next non-empty pair (loads objects into m_obj_*).
    // Returns true if a valid pair was found, false if all pairs exhausted.
    bool advance_to_next_pair();

    // Attempt to find the next valid (object, scope-rect) intersection
    // starting from the current (m_obj_idx, m_scope_in_pair_idx).
    // Stores the result in m_last and returns true; returns false when the
    // current pair is exhausted.
    bool find_next_in_pair();
};

#endif /* PMTRACKEXPRESSIONTRACKRECTSITERATOR_H_ */
