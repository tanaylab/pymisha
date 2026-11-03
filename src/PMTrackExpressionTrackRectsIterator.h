/*
 * PMTrackExpressionTrackRectsIterator.h
 *
 * 2D iterator that walks stored track objects (RECTS or POINTS) and emits
 * one GInterval2D per (track_object, scope_rect) intersection.
 *
 * A port of R misha's TrackExpressionTrackRectsIterator:
 *   - The scope is walked sorted by chromosome pair, as R sorts it.
 *   - For each pair: the track's objects in the order of R's quad-tree
 *     iterator, each intersected with a quad tree of the pair's scope
 *     intervals, emitting the pieces in the order that tree returns them.
 *   - When band is active, inter-chrom pairs in scope are skipped entirely,
 *     and the pieces are shrunk to the band as R's StatQuadTree::intersect
 *     shrinks them.
 *
 * Implementation notes:
 *   - The emissions of a pair are computed when the walk reaches it and held
 *     in a vector, as the objects of the pair are.
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
    // 1-based index, into the scope as given, of the scope rect of the current emission.
    virtual uint64_t original_interval_idx() const override { return m_emit_scope[m_emit_idx] + 1; }

private:
    PMGenomeTrack2D              m_track;      // opened once, pairs switched lazily
    const GenomeChromKey        &m_chromkey;
    quadtree::DiagonalBand       m_band;
    bool                         m_use_band;

    // The scope sorted by chromosome pair, and the index of each interval in the scope as given.
    std::vector<GInterval2D>     m_scope;
    std::vector<size_t>          m_scope_orig;

    // The current pair: m_scope[m_pair_first, m_pair_last).
    size_t                       m_pair_first;
    size_t                       m_pair_last;

    // The emissions of the current pair, and the scope interval (as given) of each.
    std::vector<GInterval2D>     m_emits;
    std::vector<size_t>          m_emit_scope;
    size_t                       m_emit_idx;

    GInterval2D m_last;
    uint64_t    m_emitted;
    bool        m_isend;

    // Computes the emissions of the next pair that has any; sets m_isend if none is left.
    void next_pair_emissions();
};

#endif /* PMTRACKEXPRESSIONTRACKRECTSITERATOR_H_ */
