#include "PMTrackExpression2DIteratorPolicy.h"

#include <stdexcept>

#include "PMTrackExpression2DIterator.h"
#include "PMTrackExpressionCartesianGridIterator.h"
#include "PMTrackExpressionFixedRectIterator.h"
#include "PMTrackExpressionTrackRectsIterator.h"

std::unique_ptr<PMTrackExpression2DIterator>
PMTrackExpression2DIteratorPolicy::make_iterator(
    std::vector<GInterval2D> scope,
    const quadtree::DiagonalBand *band) const
{
    switch (kind) {
    case K_INTERVALS:
        // R's TrackExpressionIntervals2DIterator: the iterator intervals cut by
        // the scope (the scope itself when no intervals are given), shrunk to
        // the band, with inter-chrom intervals skipped under a band.
        if (!chromkey)
            throw std::runtime_error("Intervals policy: chromkey not set");
        return std::make_unique<PMTrackExpressionIntervals2DIterator>(
            has_intervals ? intervals : scope, scope, *chromkey, band);

    case K_FIXED_RECT:
        return std::make_unique<PMTrackExpressionFixedRectIterator>(
            width, height, std::move(scope), band);

    case K_TRACK_RECTS:
        if (!chromkey)
            throw std::runtime_error("TrackRects policy: chromkey not set");
        return std::make_unique<PMTrackExpressionTrackRectsIterator>(
            track_name, std::move(scope), *chromkey, band);

    case K_CARTESIAN_GRID:
        if (!chromkey)
            throw std::runtime_error("CartesianGrid policy: chromkey is null");
        return std::make_unique<PMTrackExpressionCartesianGridIterator>(
            grid_points_axis0, grid_points_axis1,
            expansion0, expansion1,
            use_band_idx, min_band_idx, max_band_idx,
            std::move(scope), *chromkey, band);
    }
    throw std::runtime_error("Unknown iterator policy kind");
}
