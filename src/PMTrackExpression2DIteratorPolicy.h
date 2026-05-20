/*
 * PMTrackExpression2DIteratorPolicy
 *
 * Tagged sum type holding any of the four 2D iterator policies parsed
 * from a Python dict at the binding boundary. FixedRect and TrackRects
 * are implemented; CartesianGrid / Intervals2D are placeholders.
 *
 * The scanner uses make_iterator() to construct the concrete iterator
 * matching the active variant.
 */

#ifndef PMTRACKEXPRESSION2DITERATORPOLICY_H_
#define PMTRACKEXPRESSION2DITERATORPOLICY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "GInterval2D.h"
#include "GenomeChromKey.h"
#include "PMTrackExpression2DIterator.h"
#include "PMTrackExpressionCartesianGridIterator.h"
#include "QuadTreeReader.h"

class PMTrackExpression2DIteratorPolicy {
public:
    enum Kind { K_INTERVALS, K_FIXED_RECT, K_TRACK_RECTS, K_CARTESIAN_GRID };

    Kind kind;

    // FixedRect:
    int64_t width = 0;
    int64_t height = 0;

    // TrackRects:
    std::string track_name;
    // Non-owning pointer to the genome chrom key; must outlive make_iterator().
    // Reused by K_CARTESIAN_GRID.
    const GenomeChromKey *chromkey = nullptr;

    // CartesianGrid:
    std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> grid_points_axis0;
    std::vector<PMTrackExpressionCartesianGridIterator::GridPoint> grid_points_axis1;
    std::vector<int64_t> expansion0;
    std::vector<int64_t> expansion1;
    bool use_band_idx = false;
    int32_t min_band_idx = 0;
    int32_t max_band_idx = 0;

    // Build a streaming iterator for this policy. Scope is the 2D
    // intervals passed by the user. band may be nullptr.
    std::unique_ptr<PMTrackExpression2DIterator>
    make_iterator(std::vector<GInterval2D> scope,
                  const quadtree::DiagonalBand *band) const;
};

#endif /* PMTRACKEXPRESSION2DITERATORPOLICY_H_ */
