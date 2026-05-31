// src/MemQuadTree.h
//
// In-memory quadtree of 2D rectangles with a best-first NN iterator.
// Used by pm_neighbors_2d to answer "for each query rect, give me the
// nearest k targets within a Manhattan distance window" on big inputs
// where the brute-force budget is exceeded.
//
// Faithful port of R misha's StatQuadTree<RectObj>::NNIterator
// (/home/aviezerl/src/misha/src/StatQuadTree.h:255-368). Manhattan
// distance with touch=true; per-axis unsigned gap (touching edges count
// as zero distance).
//
// This class is NOT used for any on-disk format; pymisha's on-disk 2D
// reader lives in QuadTreeReader.{h,cpp} and is unrelated. MemQuadTree
// builds in memory from int64 arrays the caller supplies.

#ifndef MEMQUADTREE_H_
#define MEMQUADTREE_H_

#include <cstdint>
#include <queue>
#include <vector>

namespace memqt {

constexpr int NUM_QUADS = 4;

struct Rectangle {
    int64_t x1, y1, x2, y2;

    Rectangle() : x1(0), y1(0), x2(-1), y2(-1) {}
    Rectangle(int64_t _x1, int64_t _y1, int64_t _x2, int64_t _y2)
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}

    // True for the sentinel "empty / no excluded area" rectangle
    // (x2 < x1 or y2 < y1). Matches R's Rectangle(0, 0, -1, -1).
    bool is_empty() const { return x2 < x1 || y2 < y1; }

    bool is_non_empty_area() const { return x2 > x1 && y2 > y1; }

    Rectangle intersect(const Rectangle &o) const {
        return Rectangle(
            x1 > o.x1 ? x1 : o.x1,
            y1 > o.y1 ? y1 : o.y1,
            x2 < o.x2 ? x2 : o.x2,
            y2 < o.y2 ? y2 : o.y2);
    }

    // True if this rectangle is fully inside outer; for the sentinel
    // empty outer (excluded_area==empty), nothing is inside it.
    bool is_inside(const Rectangle &outer) const {
        if (outer.is_empty()) return false;
        return x1 >= outer.x1 && y1 >= outer.y1
            && x2 <= outer.x2 && y2 <= outer.y2;
    }

    // Manhattan distance to query with touch=true semantics: per axis,
    // gap = max(0, max(self.x1 - q.x2, q.x1 - self.x2)). Touching edges
    // give 0.
    int64_t manhattan_dist(const Rectangle &q) const {
        int64_t gx = 0;
        if (x1 >= q.x2)       gx = x1 - q.x2;
        else if (q.x1 >= x2)  gx = q.x1 - x2;
        int64_t gy = 0;
        if (y1 >= q.y2)       gy = y1 - q.y2;
        else if (q.y1 >= y2)  gy = q.y1 - y2;
        return gx + gy;
    }
};

struct RectObj {
    Rectangle rect;
    int64_t   id;  // index back into the caller's array

    RectObj() : id(-1) {}
    RectObj(const Rectangle &r, int64_t _id) : rect(r), id(_id) {}

    bool do_intersect(const Rectangle &r) const {
        if (r.is_empty()) return false;
        return rect.x1 < r.x2 && rect.x2 > r.x1
            && rect.y1 < r.y2 && rect.y2 > r.y1;
    }

    int64_t manhattan_dist(const Rectangle &q) const {
        return rect.manhattan_dist(q);
    }
};

struct Leaf {
    uint32_t obj_ptr_start;
    uint32_t obj_ptr_end;  // half-open [start, end)
};

struct Node {
    Rectangle arena;
    int64_t   occupied_area;
    bool      is_leaf;
    Leaf      leaf;
    int64_t   kid_idx[NUM_QUADS];  // indices into the parent's m_nodes

    Node() : occupied_area(0), is_leaf(true) {
        leaf.obj_ptr_start = 0;
        leaf.obj_ptr_end = 0;
        for (int i = 0; i < NUM_QUADS; ++i) kid_idx[i] = -1;
    }
    explicit Node(const Rectangle &a) : arena(a), occupied_area(0), is_leaf(true) {
        leaf.obj_ptr_start = 0;
        leaf.obj_ptr_end = 0;
        for (int i = 0; i < NUM_QUADS; ++i) kid_idx[i] = -1;
    }
};

class MemQuadTree {
public:
    MemQuadTree();

    // Initialize the root arena and build parameters.
    void init(int64_t x1, int64_t y1, int64_t x2, int64_t y2,
              unsigned max_depth = 20, unsigned max_node_objs = 20);

    // Insert one rectangle. Each insert may trigger a split if the
    // target leaf overflows max_node_objs.
    void insert(const RectObj &obj);

    // Best-first nearest-neighbor iterator. Caller pattern:
    //
    //   NNIterator it(&tree);
    //   if (!it.begin(query)) return;     // nothing to iterate
    //   do {
    //       const RectObj &cur = *it;
    //       // ... evaluate window, emit / skip / break
    //   } while (it.next());
    class NNIterator {
    public:
        explicit NNIterator(MemQuadTree *parent);

        // Returns true iff the iterator is now pointing at an object
        // (i.e. there is at least one matching object).
        bool begin(const Rectangle &query,
                   const Rectangle &excluded = Rectangle());

        // Advance. Returns true iff still pointing at an object.
        bool next();

        const RectObj &operator*() const;
        bool is_end() const { return m_neighbors.empty(); }

    private:
        struct Neighbor {
            Node    *node;  // non-null iff this entry is a tree node
            RectObj *obj;   // non-null iff this entry is an object
            int64_t  dist;

            // Min-heap on dist via std::priority_queue<>: "greater dist
            // is lower priority". On ties, objects beat nodes (R's
            // exact comment: "Otherwise nearest neighbor query that
            // covers the whole arena will cause all the tree nodes to
            // be added to the heap").
            bool operator<(const Neighbor &o) const {
                return dist > o.dist || (dist == o.dist && node != nullptr);
            }
        };

        MemQuadTree                 *m_parent;
        Rectangle                    m_query, m_excluded;
        std::priority_queue<Neighbor> m_neighbors;
        std::vector<uint8_t>         m_used_objs;
    };

    // Accessors used by NNIterator.
    Node &node(int64_t i) { return m_nodes[i]; }
    RectObj &obj_at_ptr(uint32_t ptr_idx) {
        return m_objs[m_obj_ptrs[ptr_idx]];
    }
    size_t num_nodes() const { return m_nodes.size(); }
    size_t num_objs()  const { return m_objs.size(); }

private:
    // Insert obj_idx into node, recursing as needed. The intersection
    // is obj.rect ∩ node.arena (pre-computed by the caller); area is
    // used to update node.occupied_area.
    void insert_into_node(int64_t node_idx, const Rectangle &intersection,
                          unsigned depth, int64_t obj_idx);

    // Convert a leaf at node_idx to an internal node with 4 kids,
    // redistributing existing objects into them.
    void split_leaf(int64_t node_idx, unsigned depth);

    std::vector<Node>    m_nodes;
    std::vector<RectObj> m_objs;
    std::vector<uint32_t> m_obj_ptrs;  // leaf obj_ptr ranges index this
    unsigned             m_max_depth;
    unsigned             m_max_node_objs;
};

}  // namespace memqt

#endif  // MEMQUADTREE_H_
