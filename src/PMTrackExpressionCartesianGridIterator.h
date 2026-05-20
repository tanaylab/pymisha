/*
 * PMTrackExpressionCartesianGridIterator.h
 *
 * Streaming 2D iterator that generates the cartesian product of 1D
 * windows around two sets of interval centers. For each pair of axis-0
 * and axis-1 grid points, generates one cell per (expansion1_pair,
 * expansion2_pair). Windows are clipped to chrom boundaries and adjusted
 * via R's overlap-correction to prevent neighbor-window overlap.
 *
 * Per chrom-pair: intersect each generated cell against the user's
 * 2D scope rect list (linear scan), emit clipped intersections.
 *
 * Optional band_idx filter: when both axes share the same intervals
 * (intervals2=None semantics), filter cells where |idx0 - idx1| is
 * outside [min_band_idx, max_band_idx].
 *
 * Optional band (diagonal): inter-chrom pairs skipped under active band.
 *
 * Mirrors R misha's TrackExpressionCartesianGridIterator.
 * Constructor does NOT prime; caller must call begin() first.
 */

#ifndef PMTRACKEXPRESSIONCARTESIANGRIDITERATOR_H_
#define PMTRACKEXPRESSIONCARTESIANGRIDITERATOR_H_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "GInterval2D.h"
#include "GenomeChromKey.h"
#include "PMTrackExpression2DIterator.h"
#include "QuadTreeReader.h"

class PMTrackExpressionCartesianGridIterator : public PMTrackExpression2DIterator {
public:
    struct GridPoint {
        int32_t chromid;
        int64_t coord;
        int64_t min_expansion;   // lower bound for the usable expansion range
        int64_t max_expansion;   // upper bound (exclusive) for the usable expansion range

        GridPoint() : chromid(-1), coord(-1), min_expansion(0), max_expansion(0) {}
        GridPoint(int32_t c, int64_t co) : chromid(c), coord(co), min_expansion(0), max_expansion(0) {}

        bool operator<(const GridPoint &o) const {
            return chromid < o.chromid || (chromid == o.chromid && coord < o.coord);
        }
        bool operator==(const GridPoint &o) const {
            return chromid == o.chromid && coord == o.coord;
        }
    };

    // Build corrected GridPoint vectors from raw (chromid, center) pairs,
    // applying R's overlap-correction and chrom-size clamping.
    // Exposed as a static helper so the test binding can call it directly.
    static std::vector<GridPoint> build_grid_points(
        const std::vector<GridPoint>     &raw_centers,
        const std::vector<int64_t>        &expansion,
        const GenomeChromKey              &chromkey);

    // Constructor receives PRE-BUILT grid points (overlap-corrected).
    // gp_axis1 may equal gp_axis0 (shared axis semantics) or differ.
    // Throws std::runtime_error on validation failure.
    PMTrackExpressionCartesianGridIterator(
        std::vector<GridPoint>             gp_axis0,
        std::vector<GridPoint>             gp_axis1,
        std::vector<int64_t>               expansion0,
        std::vector<int64_t>               expansion1,
        bool                               use_band_idx,
        int32_t                            min_band_idx,
        int32_t                            max_band_idx,
        std::vector<GInterval2D>           scope,
        const quadtree::DiagonalBand      *band);

    virtual ~PMTrackExpressionCartesianGridIterator() {}

    // Must be called before the first isend()/last_interval()/next() call.
    virtual void begin() override;
    virtual void next() override;
    virtual bool isend() const override { return m_isend; }
    virtual const GInterval2D &last_interval() const override { return m_last; }
    virtual uint64_t idx() const override { return m_emitted; }
    // 1-based index of the axis-0 grid point producing the current cell.
    virtual uint64_t original_interval_idx() const override { return m_gp0_idx + 1; }

private:
    using GPVec   = std::vector<GridPoint>;
    using GPIt    = GPVec::const_iterator;
    using ExpVec  = std::vector<int64_t>;
    using ExpIt   = ExpVec::const_iterator;

    // Inputs (immutable after construction).
    GPVec             m_gpoints[2];
    ExpVec            m_expansion[2];
    bool              m_use_band_idx;
    int32_t           m_min_band_idx;
    int32_t           m_max_band_idx;
    std::vector<GInterval2D>  m_scope;
    quadtree::DiagonalBand    m_band;
    bool                      m_use_band;

    // Per chrom-pair: indices into m_scope for that pair.
    // Built in constructor; key = (chromid1, chromid2).
    using PairKey = std::pair<int32_t, int32_t>;
    struct PairHash {
        size_t operator()(const PairKey &k) const noexcept {
            return std::hash<int64_t>()((int64_t)k.first << 32 | (uint32_t)k.second);
        }
    };
    std::unordered_map<PairKey, std::vector<size_t>, PairHash> m_scope_index;

    // Iterator state (reset in begin()).
    GPIt    m_igp[2];         // current grid point iterators
    GPIt    m_igp_start[2];   // anchor for start of current chromosome subset
    ExpIt   m_iexp[2];        // current expansion iterators (point past the end of the window being built)
    size_t  m_scope_rect_idx; // index into the current pair's scope rect list

    GInterval2D  m_last;
    uint64_t     m_emitted;
    size_t       m_gp0_idx;   // axis-0 gpoint index (for original_interval_idx)
    bool         m_isend;

    // Current active chrom-pair scope rects (pointer into m_scope_index).
    const std::vector<size_t> *m_cur_scope_rects;
    int32_t m_cur_chromid[2];

    // Internal helpers.
    void _reset_state();
    // Load scope rects for (c1, c2). Returns true if any scope rects exist.
    bool _load_scope_pair(int32_t c1, int32_t c2);
};

#endif /* PMTRACKEXPRESSIONCARTESIANGRIDITERATOR_H_ */
