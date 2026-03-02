/*
 * QuadTreeReader.h
 *
 * Fast C++ reader for misha StatQuadTreeCached binary format.
 * Replaces the pure-Python struct.unpack-based reader in pymisha/_quadtree.py.
 *
 * Operates on raw const byte buffers (from mmap or bytes) - zero-copy.
 * All structures match the on-disk binary layout with #pragma pack(8).
 */

#ifndef QUADTREEREADER_H_
#define QUADTREEREADER_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <unordered_set>

namespace quadtree {

// Format signatures from GenomeTrack.cpp
static constexpr int32_t SIGNATURE_RECTS  = -9;
static constexpr int32_t SIGNATURE_POINTS = -10;

// Quad indices: NW=0, NE=1, SE=2, SW=3
enum { NW = 0, NE = 1, SE = 2, SW = 3, NUM_QUADS = 4 };

// -------------------------------------------------------------------
// On-disk structures - must match R misha's #pragma pack(8) layout
// -------------------------------------------------------------------

#pragma pack(push, 8)

struct Stat {
    int64_t occupied_area;
    double  weighted_sum;
    double  min_val;
    double  max_val;

    void reset() {
        occupied_area = 0;
        weighted_sum = 0.0;
        min_val = std::numeric_limits<double>::max();
        max_val = -std::numeric_limits<double>::max();
    }
};

struct Rectangle {
    int64_t x1;
    int64_t y1;
    int64_t x2;
    int64_t y2;

    bool do_intersect(const Rectangle &r) const {
        return std::max(x1, r.x1) < std::min(x2, r.x2) &&
               std::max(y1, r.y1) < std::min(y2, r.y2);
    }

    bool is_inside(const Rectangle &r) const {
        return x1 >= r.x1 && y1 >= r.y1 && x2 <= r.x2 && y2 <= r.y2;
    }

    Rectangle intersect(const Rectangle &r) const {
        return {std::max(x1, r.x1), std::max(y1, r.y1),
                std::min(x2, r.x2), std::min(y2, r.y2)};
    }

    int64_t area() const { return (x2 - x1) * (y2 - y1); }
    bool is_non_empty_area() const { return x2 > x1 && y2 > y1; }
};

struct NodeBase {
    bool      is_leaf;    // 1 byte + 7 padding
    Stat      stat;       // 32 bytes
    Rectangle arena;      // 32 bytes
    // Total: 72 bytes
};

struct Node : public NodeBase {
    int64_t kid_ptr[NUM_QUADS]; // 32 bytes
    // Total: 104 bytes
};

struct Leaf : public NodeBase {
    uint32_t num_objs;    // 4 bytes + 4 padding
    // Total: 80 bytes
};

// Object types
struct RectObj {
    uint64_t id;
    int64_t  x1, y1, x2, y2;
    float    val;
    // 4 bytes padding to reach 48 bytes
};

struct PointObj {
    uint64_t id;
    int64_t  x, y;
    float    val;
    // 4 bytes padding to reach 32 bytes
};

#pragma pack(pop)

// Compile-time size verification
static_assert(sizeof(Stat) == 32, "Stat must be 32 bytes");
static_assert(sizeof(Rectangle) == 32, "Rectangle must be 32 bytes");
static_assert(sizeof(NodeBase) == 72, "NodeBase must be 72 bytes");
static_assert(sizeof(Node) == 104, "Node must be 104 bytes");
static_assert(sizeof(Leaf) == 80, "Leaf must be 80 bytes");
static_assert(sizeof(RectObj) == 48, "RectObj must be 48 bytes");
static_assert(sizeof(PointObj) == 32, "PointObj must be 32 bytes");

// -------------------------------------------------------------------
// Diagonal band filter
// -------------------------------------------------------------------

struct DiagonalBand {
    int64_t d1;
    int64_t d2;
    bool    active;

    DiagonalBand() : d1(0), d2(0), active(false) {}
    DiagonalBand(int64_t _d1, int64_t _d2) : d1(_d1), d2(_d2), active(true) {}

    // Rectangle intersects band?
    bool do_intersect_rect(int64_t rx1, int64_t ry1, int64_t rx2, int64_t ry2) const {
        return (rx2 - ry1 > d1) && (rx1 - ry2 + 1 < d2);
    }

    // Point inside band?
    bool contains_point(int64_t px, int64_t py) const {
        int64_t diff = px - py;
        return diff >= d1 && diff < d2;
    }

    // Rectangle fully contained in band?
    bool do_contain(const Rectangle &r) const {
        return (r.x1 - r.y2 + 1 >= d1) && (r.x2 - r.y1 <= d2);
    }

    // Rectangle intersects band?
    bool do_intersect(const Rectangle &r) const {
        return (r.x2 - r.y1 > d1) && (r.x1 - r.y2 + 1 < d2);
    }

    // Shrink rectangle to minimal area that still contains the intersection with the band
    void shrink2intersected(Rectangle &r) const {
        if (r.x1 - r.y1 < d1)
            r.x1 = r.y1 + d1;
        else if (r.x1 - r.y1 > d2)
            r.y1 = r.x1 - d2;

        if (r.x2 - r.y2 < d1)
            r.y2 = r.x2 - d1;
        else if (r.x2 - r.y2 > d2)
            r.x2 = r.y2 + d2;
    }

    // Compute intersected area of (already shrinked) rectangle with band
    int64_t intersected_area(const Rectangle &r) const {
        int64_t a = r.area();

        // subtract the area of the triangle above d1
        if (r.x1 - r.y2 + 1 < d1) {
            int64_t n = r.y1 + d1 - r.x1;
            a -= (n * n - n) >> 1;
        }

        // subtract the area of the triangle below d2
        if (r.x2 - r.y1 > d2) {
            int64_t n = r.x2 - (r.y1 + d2);
            a -= (n * n + n) >> 1;
        }

        return a;
    }
};

// -------------------------------------------------------------------
// Query result structures
// -------------------------------------------------------------------

struct QueryStat {
    int64_t occupied_area;
    double  weighted_sum;
    double  min_val;
    double  max_val;

    QueryStat() : occupied_area(0), weighted_sum(0.0),
                  min_val(std::numeric_limits<double>::max()),
                  max_val(-std::numeric_limits<double>::max()) {}
};

struct QueryObjects {
    std::vector<uint64_t> ids;
    std::vector<int64_t>  x1s, y1s, x2s, y2s;
    std::vector<float>    vals;
};

// -------------------------------------------------------------------
// Core query functions
// -------------------------------------------------------------------

// Parse the file header after the signature.
// Returns: num_objs, root_chunk_fpos
// The signature (int32_t at offset 0) should already be read by the caller.
// buffer starts at the signature.
inline void parse_header(const char *buf, size_t len,
                         uint64_t &num_objs, int64_t &root_chunk_fpos) {
    // After int32 signature (4 bytes): uint64 num_objs, int64 root_chunk_fpos
    memcpy(&num_objs, buf + 4, sizeof(uint64_t));
    if (num_objs > 0) {
        memcpy(&root_chunk_fpos, buf + 12, sizeof(int64_t));
    } else {
        root_chunk_fpos = 0;
    }
}

// Get the top_node_offset from a chunk header.
// Chunk format: [int64 chunk_size] [int64 top_node_offset] [data...]
inline int64_t get_chunk_top_node_offset(const char *buf, int64_t chunk_fpos) {
    int64_t top_node_offset;
    memcpy(&top_node_offset, buf + chunk_fpos + 8, sizeof(int64_t));
    return top_node_offset;
}

// -------------------------------------------------------------------
// Stat query (no band)
// -------------------------------------------------------------------
void query_stats_node(const char *buf, size_t len,
                      int64_t chunk_fpos, int64_t node_offset,
                      bool is_points,
                      const Rectangle &query,
                      QueryStat &stat,
                      int depth);

// Stat query (with band)
void query_stats_node_band(const char *buf, size_t len,
                           int64_t chunk_fpos, int64_t node_offset,
                           bool is_points,
                           const Rectangle &query,
                           const DiagonalBand &band,
                           QueryStat &stat,
                           int depth);

// Object query (no band)
void query_objects_node(const char *buf, size_t len,
                        int64_t chunk_fpos, int64_t node_offset,
                        bool is_points,
                        const Rectangle &query,
                        std::unordered_set<uint64_t> &seen_ids,
                        QueryObjects &result,
                        int depth);

// Object query (with band)
void query_objects_node_band(const char *buf, size_t len,
                             int64_t chunk_fpos, int64_t node_offset,
                             bool is_points,
                             const Rectangle &query,
                             const DiagonalBand &band,
                             std::unordered_set<uint64_t> &seen_ids,
                             QueryObjects &result,
                             int depth);

// -------------------------------------------------------------------
// Top-level query API
// -------------------------------------------------------------------

// Query stats for a rectangle. band can be nullptr for no band filtering.
QueryStat query_stats(const char *buf, size_t len,
                      bool is_points,
                      int64_t root_chunk_fpos,
                      int64_t qx1, int64_t qy1, int64_t qx2, int64_t qy2,
                      const DiagonalBand *band = nullptr);

// Query objects intersecting a rectangle. band can be nullptr.
QueryObjects query_objects(const char *buf, size_t len,
                           bool is_points,
                           int64_t root_chunk_fpos,
                           int64_t qx1, int64_t qy1, int64_t qx2, int64_t qy2,
                           const DiagonalBand *band = nullptr);

// -------------------------------------------------------------------
// Batch stats query — process N rectangles in a single C++ call
// -------------------------------------------------------------------

struct BatchQueryStats {
    std::vector<int64_t> occupied_area;
    std::vector<double>  weighted_sum;
    std::vector<double>  min_val;
    std::vector<double>  max_val;

    void resize(size_t n) {
        occupied_area.resize(n, 0);
        weighted_sum.resize(n, 0.0);
        min_val.resize(n, std::numeric_limits<double>::quiet_NaN());
        max_val.resize(n, std::numeric_limits<double>::quiet_NaN());
    }
};

// Query stats for N rectangles in batch.
// rects is an N×4 array of int64_t (qx1, qy1, qx2, qy2 per row, row-major).
// band can be nullptr for no band filtering.
BatchQueryStats query_stats_batch(const char *buf, size_t len,
                                  bool is_points,
                                  int64_t root_chunk_fpos,
                                  const int64_t *rects, size_t n,
                                  const DiagonalBand *band = nullptr);

} // namespace quadtree

#endif /* QUADTREEREADER_H_ */
