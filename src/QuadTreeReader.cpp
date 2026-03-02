/*
 * QuadTreeReader.cpp
 *
 * Fast C++ reader for misha StatQuadTreeCached binary format.
 * Implements the recursive quad-tree traversal for stat and object queries,
 * matching R misha's get_stat() algorithm.
 */

#include "QuadTreeReader.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace quadtree {

static constexpr int MAX_DEPTH = 100000;

// -------------------------------------------------------------------
// Helper: resolve kid pointer to (chunk_fpos, node_offset)
// -------------------------------------------------------------------
static inline void resolve_kid(const char *buf, int64_t chunk_fpos,
                                int64_t kid_ptr,
                                int64_t &out_chunk_fpos,
                                int64_t &out_node_offset) {
    if (kid_ptr >= 0) {
        // Positive: offset from chunk start (within same chunk)
        out_chunk_fpos = chunk_fpos;
        out_node_offset = kid_ptr;
    } else {
        // Negative: absolute file position of another chunk
        int64_t cross_chunk_fpos = -kid_ptr;
        out_chunk_fpos = cross_chunk_fpos;
        out_node_offset = get_chunk_top_node_offset(buf, cross_chunk_fpos);
    }
}

// -------------------------------------------------------------------
// Helper: read NodeBase at absolute offset
// -------------------------------------------------------------------
static inline const NodeBase *read_node_base(const char *buf, int64_t abs_offset) {
    return reinterpret_cast<const NodeBase *>(buf + abs_offset);
}

// -------------------------------------------------------------------
// Helper: update stat from a rectangle object intersection (no band)
// -------------------------------------------------------------------
static inline void update_stat_rect(float val, int64_t inter_area, QueryStat &stat) {
    double dval = static_cast<double>(val);
    stat.occupied_area += inter_area;
    stat.weighted_sum += dval * inter_area;
    if (dval < stat.min_val) stat.min_val = dval;
    if (dval > stat.max_val) stat.max_val = dval;
}

// -------------------------------------------------------------------
// Helper: update stat from a point object (no band)
// -------------------------------------------------------------------
static inline void update_stat_point(float val, QueryStat &stat) {
    double dval = static_cast<double>(val);
    stat.occupied_area += 1;
    stat.weighted_sum += dval;
    if (dval < stat.min_val) stat.min_val = dval;
    if (dval > stat.max_val) stat.max_val = dval;
}

// -------------------------------------------------------------------
// Helper: merge pre-computed node stat into query stat
// -------------------------------------------------------------------
static inline void merge_node_stat(const Stat &ns, QueryStat &stat) {
    if (ns.occupied_area > 0) {
        stat.occupied_area += ns.occupied_area;
        stat.weighted_sum += ns.weighted_sum;
        if (ns.min_val < stat.min_val) stat.min_val = ns.min_val;
        if (ns.max_val > stat.max_val) stat.max_val = ns.max_val;
    }
}

// -------------------------------------------------------------------
// Stat query: no band
// -------------------------------------------------------------------
void query_stats_node(const char *buf, size_t len,
                      int64_t chunk_fpos, int64_t node_offset,
                      bool is_points,
                      const Rectangle &query,
                      QueryStat &stat,
                      int depth) {
    if (depth > MAX_DEPTH) return;

    int64_t abs_offset = chunk_fpos + node_offset;
    const NodeBase *nb = read_node_base(buf, abs_offset);

    // Prune: no intersection with query
    if (!nb->arena.do_intersect(query)) return;

    if (nb->is_leaf) {
        const Leaf *leaf = reinterpret_cast<const Leaf *>(buf + abs_offset);
        const char *objs_ptr = buf + abs_offset + sizeof(Leaf);

        if (is_points) {
            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const PointObj *obj = reinterpret_cast<const PointObj *>(
                    objs_ptr + i * sizeof(PointObj));
                // Point intersection with query clamped to leaf arena
                int64_t eqx1 = std::max(query.x1, nb->arena.x1);
                int64_t eqy1 = std::max(query.y1, nb->arena.y1);
                int64_t eqx2 = std::min(query.x2, nb->arena.x2);
                int64_t eqy2 = std::min(query.y2, nb->arena.y2);
                if (obj->x >= eqx1 && obj->x < eqx2 &&
                    obj->y >= eqy1 && obj->y < eqy2) {
                    update_stat_point(obj->val, stat);
                }
            }
        } else {
            // Effective query rect clamped to leaf arena
            int64_t eqx1 = std::max(query.x1, nb->arena.x1);
            int64_t eqy1 = std::max(query.y1, nb->arena.y1);
            int64_t eqx2 = std::min(query.x2, nb->arena.x2);
            int64_t eqy2 = std::min(query.y2, nb->arena.y2);

            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const RectObj *obj = reinterpret_cast<const RectObj *>(
                    objs_ptr + i * sizeof(RectObj));
                // Intersection of obj with effective query (query intersect arena)
                int64_t ix1 = std::max(eqx1, obj->x1);
                int64_t iy1 = std::max(eqy1, obj->y1);
                int64_t ix2 = std::min(eqx2, obj->x2);
                int64_t iy2 = std::min(eqy2, obj->y2);
                if (ix1 < ix2 && iy1 < iy2) {
                    int64_t area = (ix2 - ix1) * (iy2 - iy1);
                    update_stat_rect(obj->val, area, stat);
                }
            }
        }
    } else {
        // Internal node: check each child quadrant
        const Node *node = reinterpret_cast<const Node *>(buf + abs_offset);
        for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
            int64_t kid_chunk, kid_offset;
            resolve_kid(buf, chunk_fpos, node->kid_ptr[iquad],
                        kid_chunk, kid_offset);

            int64_t kid_abs = kid_chunk + kid_offset;
            const NodeBase *kid_nb = read_node_base(buf, kid_abs);

            // Skip if child doesn't intersect query
            if (!kid_nb->arena.do_intersect(query)) continue;

            // Fast path: child fully inside query -> use pre-computed stats
            if (kid_nb->arena.is_inside(query)) {
                merge_node_stat(kid_nb->stat, stat);
            } else {
                // Partial overlap: recurse
                query_stats_node(buf, len, kid_chunk, kid_offset,
                                 is_points, query, stat, depth + 1);
            }
        }
    }
}

// -------------------------------------------------------------------
// Stat query: with band
// -------------------------------------------------------------------
void query_stats_node_band(const char *buf, size_t len,
                           int64_t chunk_fpos, int64_t node_offset,
                           bool is_points,
                           const Rectangle &query,
                           const DiagonalBand &band,
                           QueryStat &stat,
                           int depth) {
    if (depth > MAX_DEPTH) return;

    int64_t abs_offset = chunk_fpos + node_offset;
    const NodeBase *nb = read_node_base(buf, abs_offset);

    // Prune: no intersection with query
    if (!nb->arena.do_intersect(query)) return;

    if (nb->is_leaf) {
        const Leaf *leaf = reinterpret_cast<const Leaf *>(buf + abs_offset);
        const char *objs_ptr = buf + abs_offset + sizeof(Leaf);

        if (is_points) {
            int64_t eqx1 = std::max(query.x1, nb->arena.x1);
            int64_t eqy1 = std::max(query.y1, nb->arena.y1);
            int64_t eqx2 = std::min(query.x2, nb->arena.x2);
            int64_t eqy2 = std::min(query.y2, nb->arena.y2);

            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const PointObj *obj = reinterpret_cast<const PointObj *>(
                    objs_ptr + i * sizeof(PointObj));
                if (obj->x >= eqx1 && obj->x < eqx2 &&
                    obj->y >= eqy1 && obj->y < eqy2 &&
                    band.contains_point(obj->x, obj->y)) {
                    update_stat_point(obj->val, stat);
                }
            }
        } else {
            int64_t eqx1 = std::max(query.x1, nb->arena.x1);
            int64_t eqy1 = std::max(query.y1, nb->arena.y1);
            int64_t eqx2 = std::min(query.x2, nb->arena.x2);
            int64_t eqy2 = std::min(query.y2, nb->arena.y2);

            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const RectObj *obj = reinterpret_cast<const RectObj *>(
                    objs_ptr + i * sizeof(RectObj));
                // 3-way intersection: obj intersect query intersect arena
                Rectangle obj_rect = {obj->x1, obj->y1, obj->x2, obj->y2};
                Rectangle eff_query = {eqx1, eqy1, eqx2, eqy2};
                Rectangle intersection = obj_rect.intersect(eff_query);

                if (intersection.is_non_empty_area()) {
                    // Check band intersection
                    if (band.do_contain(intersection)) {
                        int64_t area = intersection.area();
                        update_stat_rect(obj->val, area, stat);
                    } else if (band.do_intersect(intersection)) {
                        band.shrink2intersected(intersection);
                        int64_t area = band.intersected_area(intersection);
                        if (area > 0) {
                            update_stat_rect(obj->val, area, stat);
                        }
                    }
                }
            }
        }
    } else {
        // Internal node
        const Node *node = reinterpret_cast<const Node *>(buf + abs_offset);
        for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
            int64_t kid_chunk, kid_offset;
            resolve_kid(buf, chunk_fpos, node->kid_ptr[iquad],
                        kid_chunk, kid_offset);

            int64_t kid_abs = kid_chunk + kid_offset;
            const NodeBase *kid_nb = read_node_base(buf, kid_abs);

            if (!kid_nb->arena.do_intersect(query)) continue;

            if (kid_nb->arena.is_inside(query)) {
                // Child fully inside query
                if (band.do_contain(kid_nb->arena)) {
                    // Also fully inside band: use pre-computed stats
                    merge_node_stat(kid_nb->stat, stat);
                } else if (band.do_intersect(kid_nb->arena)) {
                    // Partially overlaps band: shrink and recurse
                    Rectangle r = kid_nb->arena;
                    band.shrink2intersected(r);
                    query_stats_node_band(buf, len, kid_chunk, kid_offset,
                                          is_points, r, band, stat, depth + 1);
                }
            } else {
                // Partial overlap with query
                Rectangle intersection = kid_nb->arena.intersect(query);
                if (band.do_contain(intersection)) {
                    // Band fully contains the intersection: no band recursion needed
                    query_stats_node(buf, len, kid_chunk, kid_offset,
                                     is_points, query, stat, depth + 1);
                } else if (band.do_intersect(intersection)) {
                    Rectangle r = kid_nb->arena;
                    band.shrink2intersected(r);
                    query_stats_node_band(buf, len, kid_chunk, kid_offset,
                                          is_points, intersection, band, stat, depth + 1);
                }
            }
        }
    }
}

// -------------------------------------------------------------------
// Object query: no band
// -------------------------------------------------------------------
void query_objects_node(const char *buf, size_t len,
                        int64_t chunk_fpos, int64_t node_offset,
                        bool is_points,
                        const Rectangle &query,
                        std::unordered_set<uint64_t> &seen_ids,
                        QueryObjects &result,
                        int depth) {
    if (depth > MAX_DEPTH) return;

    int64_t abs_offset = chunk_fpos + node_offset;
    const NodeBase *nb = read_node_base(buf, abs_offset);

    // Prune: no intersection with query
    if (!nb->arena.do_intersect(query)) return;

    if (nb->is_leaf) {
        const Leaf *leaf = reinterpret_cast<const Leaf *>(buf + abs_offset);
        const char *objs_ptr = buf + abs_offset + sizeof(Leaf);

        if (is_points) {
            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const PointObj *obj = reinterpret_cast<const PointObj *>(
                    objs_ptr + i * sizeof(PointObj));
                if (seen_ids.count(obj->id)) continue;
                // Point occupies [x, x+1) x [y, y+1)
                if (obj->x < query.x2 && obj->x + 1 > query.x1 &&
                    obj->y < query.y2 && obj->y + 1 > query.y1) {
                    seen_ids.insert(obj->id);
                    result.ids.push_back(obj->id);
                    result.x1s.push_back(obj->x);
                    result.y1s.push_back(obj->y);
                    result.x2s.push_back(obj->x + 1);
                    result.y2s.push_back(obj->y + 1);
                    result.vals.push_back(obj->val);
                }
            }
        } else {
            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const RectObj *obj = reinterpret_cast<const RectObj *>(
                    objs_ptr + i * sizeof(RectObj));
                if (seen_ids.count(obj->id)) continue;
                if (obj->x1 < query.x2 && obj->x2 > query.x1 &&
                    obj->y1 < query.y2 && obj->y2 > query.y1) {
                    seen_ids.insert(obj->id);
                    result.ids.push_back(obj->id);
                    result.x1s.push_back(obj->x1);
                    result.y1s.push_back(obj->y1);
                    result.x2s.push_back(obj->x2);
                    result.y2s.push_back(obj->y2);
                    result.vals.push_back(obj->val);
                }
            }
        }
    } else {
        const Node *node = reinterpret_cast<const Node *>(buf + abs_offset);
        for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
            int64_t kid_chunk, kid_offset;
            resolve_kid(buf, chunk_fpos, node->kid_ptr[iquad],
                        kid_chunk, kid_offset);

            int64_t kid_abs = kid_chunk + kid_offset;
            const NodeBase *kid_nb = read_node_base(buf, kid_abs);

            if (kid_nb->stat.occupied_area > 0 && kid_nb->arena.do_intersect(query)) {
                query_objects_node(buf, len, kid_chunk, kid_offset,
                                   is_points, query, seen_ids, result, depth + 1);
            }
        }
    }
}

// -------------------------------------------------------------------
// Object query: with band
// -------------------------------------------------------------------
void query_objects_node_band(const char *buf, size_t len,
                             int64_t chunk_fpos, int64_t node_offset,
                             bool is_points,
                             const Rectangle &query,
                             const DiagonalBand &band,
                             std::unordered_set<uint64_t> &seen_ids,
                             QueryObjects &result,
                             int depth) {
    if (depth > MAX_DEPTH) return;

    int64_t abs_offset = chunk_fpos + node_offset;
    const NodeBase *nb = read_node_base(buf, abs_offset);

    if (!nb->arena.do_intersect(query)) return;

    if (nb->is_leaf) {
        const Leaf *leaf = reinterpret_cast<const Leaf *>(buf + abs_offset);
        const char *objs_ptr = buf + abs_offset + sizeof(Leaf);

        if (is_points) {
            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const PointObj *obj = reinterpret_cast<const PointObj *>(
                    objs_ptr + i * sizeof(PointObj));
                if (seen_ids.count(obj->id)) continue;
                if (obj->x < query.x2 && obj->x + 1 > query.x1 &&
                    obj->y < query.y2 && obj->y + 1 > query.y1 &&
                    band.contains_point(obj->x, obj->y)) {
                    seen_ids.insert(obj->id);
                    result.ids.push_back(obj->id);
                    result.x1s.push_back(obj->x);
                    result.y1s.push_back(obj->y);
                    result.x2s.push_back(obj->x + 1);
                    result.y2s.push_back(obj->y + 1);
                    result.vals.push_back(obj->val);
                }
            }
        } else {
            for (uint32_t i = 0; i < leaf->num_objs; ++i) {
                const RectObj *obj = reinterpret_cast<const RectObj *>(
                    objs_ptr + i * sizeof(RectObj));
                if (seen_ids.count(obj->id)) continue;
                if (obj->x1 < query.x2 && obj->x2 > query.x1 &&
                    obj->y1 < query.y2 && obj->y2 > query.y1 &&
                    band.do_intersect_rect(obj->x1, obj->y1, obj->x2, obj->y2)) {
                    seen_ids.insert(obj->id);
                    result.ids.push_back(obj->id);
                    result.x1s.push_back(obj->x1);
                    result.y1s.push_back(obj->y1);
                    result.x2s.push_back(obj->x2);
                    result.y2s.push_back(obj->y2);
                    result.vals.push_back(obj->val);
                }
            }
        }
    } else {
        const Node *node = reinterpret_cast<const Node *>(buf + abs_offset);
        for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
            int64_t kid_chunk, kid_offset;
            resolve_kid(buf, chunk_fpos, node->kid_ptr[iquad],
                        kid_chunk, kid_offset);

            int64_t kid_abs = kid_chunk + kid_offset;
            const NodeBase *kid_nb = read_node_base(buf, kid_abs);

            if (kid_nb->stat.occupied_area > 0 && kid_nb->arena.do_intersect(query)) {
                // Check if child arena intersects band at all
                Rectangle intersection = kid_nb->arena.intersect(query);
                if (band.do_intersect(intersection)) {
                    query_objects_node_band(buf, len, kid_chunk, kid_offset,
                                            is_points, query, band,
                                            seen_ids, result, depth + 1);
                }
            }
        }
    }
}

// -------------------------------------------------------------------
// Top-level API
// -------------------------------------------------------------------

QueryStat query_stats(const char *buf, size_t len,
                      bool is_points,
                      int64_t root_chunk_fpos,
                      int64_t qx1, int64_t qy1, int64_t qx2, int64_t qy2,
                      const DiagonalBand *band) {
    QueryStat stat;
    int64_t top_node_offset = get_chunk_top_node_offset(buf, root_chunk_fpos);

    Rectangle query = {qx1, qy1, qx2, qy2};

    if (band && band->active) {
        query_stats_node_band(buf, len, root_chunk_fpos, top_node_offset,
                              is_points, query, *band, stat, 0);
    } else {
        query_stats_node(buf, len, root_chunk_fpos, top_node_offset,
                         is_points, query, stat, 0);
    }

    // Match R misha: when no objects contribute, set all to NaN
    if (stat.occupied_area == 0) {
        stat.weighted_sum = std::numeric_limits<double>::quiet_NaN();
        stat.min_val = std::numeric_limits<double>::quiet_NaN();
        stat.max_val = std::numeric_limits<double>::quiet_NaN();
    }

    return stat;
}

QueryObjects query_objects(const char *buf, size_t len,
                           bool is_points,
                           int64_t root_chunk_fpos,
                           int64_t qx1, int64_t qy1, int64_t qx2, int64_t qy2,
                           const DiagonalBand *band) {
    QueryObjects result;
    std::unordered_set<uint64_t> seen_ids;
    int64_t top_node_offset = get_chunk_top_node_offset(buf, root_chunk_fpos);

    Rectangle query = {qx1, qy1, qx2, qy2};

    if (band && band->active) {
        query_objects_node_band(buf, len, root_chunk_fpos, top_node_offset,
                                is_points, query, *band,
                                seen_ids, result, 0);
    } else {
        query_objects_node(buf, len, root_chunk_fpos, top_node_offset,
                           is_points, query, seen_ids, result, 0);
    }

    return result;
}

// -------------------------------------------------------------------
// Batch stats query
// -------------------------------------------------------------------

BatchQueryStats query_stats_batch(const char *buf, size_t len,
                                  bool is_points,
                                  int64_t root_chunk_fpos,
                                  const int64_t *rects, size_t n,
                                  const DiagonalBand *band) {
    BatchQueryStats result;
    result.resize(n);

    int64_t top_node_offset = get_chunk_top_node_offset(buf, root_chunk_fpos);

    for (size_t i = 0; i < n; ++i) {
        int64_t qx1 = rects[i * 4 + 0];
        int64_t qy1 = rects[i * 4 + 1];
        int64_t qx2 = rects[i * 4 + 2];
        int64_t qy2 = rects[i * 4 + 3];

        Rectangle query = {qx1, qy1, qx2, qy2};
        QueryStat stat;

        if (band && band->active) {
            query_stats_node_band(buf, len, root_chunk_fpos, top_node_offset,
                                  is_points, query, *band, stat, 0);
        } else {
            query_stats_node(buf, len, root_chunk_fpos, top_node_offset,
                             is_points, query, stat, 0);
        }

        result.occupied_area[i] = stat.occupied_area;
        if (stat.occupied_area == 0) {
            result.weighted_sum[i] = std::numeric_limits<double>::quiet_NaN();
            result.min_val[i] = std::numeric_limits<double>::quiet_NaN();
            result.max_val[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            result.weighted_sum[i] = stat.weighted_sum;
            result.min_val[i] = stat.min_val;
            result.max_val[i] = stat.max_val;
        }
    }

    return result;
}

} // namespace quadtree
