/*
 * Filter.h
 *
 * The mask of a filtered virtual track (gvtrack_filter), as R misha's Genome1DFilter: sorted,
 * disjoint intervals per chromosome. A virtual track with a filter is computed over the parts of
 * each iterator interval (after the shifts) that the mask leaves.
 */

#ifndef FILTER_H_
#define FILTER_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "GInterval.h"

struct Genome1DFilter {
    // per chromosome id, the mask intervals sorted by start, none overlapping or touching another
    std::vector<std::vector<GInterval>> per_chrom;

    // The parts of I outside the mask, in order, into out (nothing if the mask covers I)
    void subtract(const GInterval &I, std::vector<GInterval> &out) const
    {
        out.clear();
        if (I.chromid < 0 || (size_t)I.chromid >= per_chrom.size() || per_chrom[I.chromid].empty()) {
            out.push_back(I);
            return;
        }
        const std::vector<GInterval> &mask = per_chrom[I.chromid];
        // the first mask interval ending after I.start
        auto it = std::lower_bound(mask.begin(), mask.end(), I.start,
                                   [](const GInterval &m, int64_t pos) { return m.end <= pos; });
        int64_t cur = I.start;
        for (; it != mask.end() && it->start < I.end; ++it) {
            if (it->start > cur)
                out.emplace_back(I.chromid, cur, std::min<int64_t>(it->start, I.end), I.strand);
            cur = std::max<int64_t>(cur, it->end);
            if (cur >= I.end)
                break;
        }
        if (cur < I.end)
            out.emplace_back(I.chromid, cur, I.end, I.strand);
    }
};

// "nearest" of a filtered virtual track whose unmasked parts (sorted, of one chromosome) hold no
// value, as R misha's TrackVarProcessor::nearest_unmasked_value: the value of the item whose
// unmasked stretch is nearest to the parts, measured from the first part for the items on its left
// and from the last part for those on its right. An item inside the mask is never chosen, nor one
// without a value; the left one wins at the same distance; NaN if there is none. items[begin, end)
// are the chromosome's items (sorted, not overlapping each other), value(i) the value of item i.
template <class Value>
double nearest_unmasked_value(const std::vector<GInterval> &items, size_t begin, size_t end, Value value,
                              const Genome1DFilter &filter, const std::vector<GInterval> &parts)
{
    const int chromid = parts.front().chromid;
    const int64_t first_start = parts.front().start;
    const int64_t last_end = parts.back().end;
    std::vector<GInterval> stretches;

    // Left: the last value with an unmasked stretch before the first part
    bool has_left = false;
    double left_val = 0;
    int64_t left_dist = 0;
    auto ileft = std::upper_bound(items.begin() + begin, items.begin() + end, first_start,
                                  [](int64_t coord, const GInterval &item) { return coord < item.start; });
    for (size_t i = ileft - items.begin(); i-- > begin; ) {
        const GInterval &item = items[i];
        const float v = value(i);
        if (std::isnan(v))
            continue;
        filter.subtract(GInterval(chromid, item.start, std::min(item.end, first_start), 0), stretches);
        if (!stretches.empty()) {
            has_left = true;
            left_val = v;
            left_dist = first_start - stretches.back().end;
            break;
        }
    }

    // Right: the first value with an unmasked stretch after the last part
    bool has_right = false;
    double right_val = 0;
    int64_t right_dist = 0;
    auto iright = std::lower_bound(items.begin() + begin, items.begin() + end, last_end,
                                   [](const GInterval &item, int64_t coord) { return item.end <= coord; });
    for (size_t i = iright - items.begin(); i < end; ++i) {
        const GInterval &item = items[i];
        const float v = value(i);
        if (std::isnan(v))
            continue;
        filter.subtract(GInterval(chromid, std::max(item.start, last_end), item.end, 0), stretches);
        if (!stretches.empty()) {
            has_right = true;
            right_val = v;
            right_dist = stretches.front().start - last_end;
            break;
        }
    }

    if (has_left && (!has_right || left_dist <= right_dist))
        return left_val;
    if (has_right)
        return right_val;
    return std::numeric_limits<double>::quiet_NaN();
}

#endif /* FILTER_H_ */
