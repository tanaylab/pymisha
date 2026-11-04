/*
 * PMTrackExpression2DIterator.h
 *
 * Iterator classes for 2D track expression evaluation in pymisha.
 * Parallel to PMTrackExpressionIterator.h (which is 1D-throughout).
 */

#ifndef PMTRACKEXPRESSION2DITERATOR_H_
#define PMTRACKEXPRESSION2DITERATOR_H_

#include <algorithm>
#include <cstddef>
#include <vector>
#include "GInterval2D.h"
#include "GenomeChromKey.h"
#include "QuadTreeReader.h"
#include "StatQuadTree.h"

// The order in which R walks a 2D scope or a 2D intervals iterator: GIntervals2D::sort
// sorts by chromosome pair only (compare_for_sort), with std::sort, and only when the
// intervals are not sorted that way already. std::sort is not stable, so the intervals
// of a pair keep R's order only when they are sorted the same way: with the same
// comparator over the same sequence, which gives the same permutation. Returns the
// indices of `intervals` in that order.
inline std::vector<size_t> r_2d_sort_order(const std::vector<GInterval2D> &intervals)
{
    std::vector<size_t> order(intervals.size());
    for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    auto less = [&](size_t a, size_t b) {
        const GInterval2D &x = intervals[a], &y = intervals[b];
        return x.chromid1 < y.chromid1 || (x.chromid1 == y.chromid1 && x.chromid2 < y.chromid2);
    };
    for (size_t i = 1; i < order.size(); ++i) {
        if (less(order[i], order[i - 1])) {
            std::sort(order.begin(), order.end(), less);
            break;
        }
    }
    return order;
}

// The scope intervals [first, last) of one chromosome pair as R's 2D iterators hold them:
// a quad tree over the pair (TrackExpression2DIterator::IntervalsQuadTree), object i
// being interval first + i.
inline void build_scope_qtree(RectsQuadTree &qtree, const std::vector<GInterval2D> &scope, size_t first, size_t last,
                              const GenomeChromKey &chromkey)
{
    qtree.reset(0, 0, (int64_t)chromkey.get_chrom_size(scope[first].chromid1),
                (int64_t)chromkey.get_chrom_size(scope[first].chromid2));
    for (size_t i = first; i < last; ++i) {
        const GInterval2D &iv = scope[i];
        // R's GIntervals2D::verify_no_overlaps: an overlapping 2D scope is an
        // error (iterating one made the tree, and memory, grow without bound).
        Rectangle rect(iv.start1, iv.start2, iv.end1, iv.end2);
        if (qtree.do_intersect(rect)) {
            Rectangles hit;
            std::vector<uint64_t> hit_idx;
            qtree.intersect(rect, hit, hit_idx);
            const GInterval2D &other = scope[first + hit_idx[0]];
            TGLError("Intervals (%s, %ld, %ld, %s, %ld, %ld) and (%s, %ld, %ld, %s, %ld, %ld) overlap",
                     chromkey.id2chrom(iv.chromid1).c_str(), (long)iv.start1, (long)iv.end1,
                     chromkey.id2chrom(iv.chromid2).c_str(), (long)iv.start2, (long)iv.end2,
                     chromkey.id2chrom(other.chromid1).c_str(), (long)other.start1, (long)other.end1,
                     chromkey.id2chrom(other.chromid2).c_str(), (long)other.start2, (long)other.end2);
        }
        qtree.insert(RectsQuadTree::ValueType(iv.start1, iv.start2, iv.end1, iv.end2, 0));
    }
}

// R's m_scope_qtree.intersect(rect[, band], ...): the scope intervals that rect meets,
// cut to it (and shrunk to the band when there is one), in the order of the tree.
inline void intersect_scope_qtree(const RectsQuadTree &qtree, const Rectangle &rect, const quadtree::DiagonalBand *band,
                                  Rectangles &intersection, std::vector<uint64_t> &indices)
{
    if (band && band->active)
        qtree.intersect(rect, DiagonalBand(band->d1, band->d2), intersection, indices);
    else
        qtree.intersect(rect, intersection, indices);
}

// Base class for all 2D track expression iterators.
class PMTrackExpression2DIterator {
public:
    virtual ~PMTrackExpression2DIterator() {}

    virtual void begin() = 0;
    virtual void next() = 0;
    virtual bool isend() const = 0;

    virtual const GInterval2D &last_interval() const = 0;

    // Estimated number of intervals to be emitted (0 if unknown).
    virtual uint64_t size() const { return 0; }

    // 0-based index of the current emission (for progress reporting).
    virtual uint64_t idx() const { return 0; }

    // 1-based index, into the scope the iterator was given, of the scope interval
    // that the current emission comes from (R: the scanner's last scope interval).
    virtual uint64_t original_interval_idx() const { return 0; }

    // Tag method: always true for 2D iterators (mirrors 1D base's is_1d()).
    virtual bool is_2d() const { return true; }
};

// Iterator that emits each input 2D interval exactly once, in input order.
// Counterpart to PMIntervalsIterator (1D).
//
// R parity (TrackExpressionIntervals2DIterator + DiagonalBand):
//   - When a band is active and the input interval is on an *inter-chromosomal*
//     pair, the interval is skipped entirely (matches FixedRect's parity at
//     PMTrackExpressionFixedRectIterator.cpp:147-151).
//   - When a band is active and the pair is same-chrom but the band doesn't
//     intersect the rect at all, the interval is skipped.
//   - Surviving same-chrom intervals are emitted with their rect *shrunk to
//     the band-intersected bounding box* (DiagonalBand::shrink2intersected).
//
// With a scope, it is R's TrackExpressionIntervals2DIterator instead: the intervals
// (the iterator) in R's order (r_2d_sort_order), each cut by the scope intervals of its
// chromosome pair as R's scope quad tree returns them (intersect_scope_qtree), one
// emission per piece. Under a band an inter-chromosomal interval is skipped.
class PMTrackExpressionIntervals2DIterator : public PMTrackExpression2DIterator {
public:
    PMTrackExpressionIntervals2DIterator(
        const std::vector<GInterval2D> &intervals,
        const quadtree::DiagonalBand *band = nullptr);
    PMTrackExpressionIntervals2DIterator(
        const std::vector<GInterval2D> &intervals,
        const std::vector<GInterval2D> &scope,
        const GenomeChromKey &chromkey,
        const quadtree::DiagonalBand *band = nullptr);
    virtual ~PMTrackExpressionIntervals2DIterator() {}

    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }

    virtual const GInterval2D &last_interval() const override {
        return m_emit;
    }

    virtual uint64_t size() const override { return m_with_scope ? m_emits.size() : m_intervals.size(); }
    virtual uint64_t idx() const override { return m_cur_idx; }
    virtual uint64_t original_interval_idx() const override {
        return m_with_scope ? m_emit_scope[m_cur_idx] + 1 : m_cur_idx + 1;
    }

private:
    // with a scope: every emission and the scope interval it comes from
    bool                     m_with_scope{false};
    std::vector<GInterval2D> m_emits;
    std::vector<size_t>      m_emit_scope;

    // Position m_cur_idx on the next interval (from m_scan_idx) that survives
    // the band, applying shrink2intersected; sets m_isend when exhausted.
    void advance_to_next_emit();

    std::vector<GInterval2D> m_intervals;
    size_t                   m_cur_idx;     // index of the current emission
    size_t                   m_scan_idx;    // next candidate to consider
    bool                     m_isend;
    bool                     m_use_band;
    quadtree::DiagonalBand   m_band;        // copy (safe under move-only sources)
    GInterval2D              m_emit;        // current (possibly shrunk) emission
};

#endif /* PMTRACKEXPRESSION2DITERATOR_H_ */
