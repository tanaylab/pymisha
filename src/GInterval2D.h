/*
 * GInterval2D.h
 *
 * 2D interval data type for pymisha (used by the 2D track expression scanner).
 *
 * Distinct from quadtree::Rectangle: that type is a packed on-disk layout
 * (32 bytes, #pragma pack(8)) used by QuadTreeReader, whereas GInterval2D
 * is a plain in-memory type carrying chromids alongside the two 1D ranges.
 */

#ifndef GINTERVAL2D_H_
#define GINTERVAL2D_H_

#include <cstdint>
#include <inttypes.h>

struct GInterval2D {
    int     chromid1;
    int     chromid2;
    int64_t start1;
    int64_t end1;
    int64_t start2;
    int64_t end2;

    GInterval2D()
        : chromid1(-1), chromid2(-1), start1(-1), end1(-1), start2(-1), end2(-1) {}

    GInterval2D(int _chromid1, int64_t _start1, int64_t _end1,
                int _chromid2, int64_t _start2, int64_t _end2)
        : chromid1(_chromid1), chromid2(_chromid2),
          start1(_start1), end1(_end1), start2(_start2), end2(_end2) {}

    bool is_same_chrom_pair(const GInterval2D &o) const {
        return chromid1 == o.chromid1 && chromid2 == o.chromid2;
    }

    int64_t range1() const { return end1 - start1; }
    int64_t range2() const { return end2 - start2; }

    // 2D surface area; uses double to avoid int64 overflow on big rectangles.
    double surface() const {
        return static_cast<double>(range1()) * static_cast<double>(range2());
    }

    // Ordering: (chromid1, chromid2, start1, start2). Stable for grouping by chrom-pair.
    bool operator<(const GInterval2D &o) const {
        if (chromid1 != o.chromid1) return chromid1 < o.chromid1;
        if (chromid2 != o.chromid2) return chromid2 < o.chromid2;
        if (start1   != o.start1)   return start1   < o.start1;
        return start2 < o.start2;
    }
};

#endif /* GINTERVAL2D_H_ */
