// src/MemQuadTree.cpp
//
// Build + NN iterator for an in-memory 2D quadtree. Faithful port of R
// misha's StatQuadTree (/home/aviezerl/src/misha/src/StatQuadTree.h).
// Used by pm_neighbors_2d for large unbounded 2D nearest-neighbor
// queries.

#include "MemQuadTree.h"

#include <algorithm>
#include <cstdint>

namespace memqt {

MemQuadTree::MemQuadTree() : m_max_depth(20), m_max_node_objs(20) {}

void MemQuadTree::init(int64_t x1, int64_t y1, int64_t x2, int64_t y2,
                       unsigned max_depth, unsigned max_node_objs) {
    m_max_depth = max_depth;
    m_max_node_objs = max_node_objs;
    m_nodes.clear();
    m_objs.clear();
    m_obj_ptrs.clear();
    m_nodes.emplace_back(Rectangle(x1, y1, x2, y2));
}

void MemQuadTree::insert(const RectObj &obj) {
    m_objs.push_back(obj);
    int64_t obj_idx = static_cast<int64_t>(m_objs.size()) - 1;

    Rectangle intersection = obj.rect.intersect(m_nodes[0].arena);
    if (!intersection.is_non_empty_area())
        return;
    insert_into_node(0, intersection, 0, obj_idx);
}

void MemQuadTree::insert_into_node(int64_t node_idx,
                                   const Rectangle &intersection,
                                   unsigned depth,
                                   int64_t obj_idx) {
    Node &node = m_nodes[node_idx];

    int64_t area = (intersection.x2 - intersection.x1)
                 * (intersection.y2 - intersection.y1);
    node.occupied_area += area;

    if (node.is_leaf) {
        // Append the object's pointer to the leaf's range. The leaf's
        // obj_ptr range is contiguous within m_obj_ptrs; when we want
        // to grow it we push to the back and bump end. This means
        // splits that interleave can fragment ranges, so we rebuild
        // the leaf's range at split time.
        uint32_t cur_count = node.leaf.obj_ptr_end - node.leaf.obj_ptr_start;

        if (cur_count < m_max_node_objs || depth >= m_max_depth) {
            // Grow the leaf in place: copy existing ptrs to the back,
            // append the new ptr, update the range.
            uint32_t old_start = node.leaf.obj_ptr_start;
            uint32_t old_end   = node.leaf.obj_ptr_end;
            uint32_t new_start = static_cast<uint32_t>(m_obj_ptrs.size());
            for (uint32_t k = old_start; k < old_end; ++k)
                m_obj_ptrs.push_back(m_obj_ptrs[k]);
            m_obj_ptrs.push_back(static_cast<uint32_t>(obj_idx));
            node.leaf.obj_ptr_start = new_start;
            node.leaf.obj_ptr_end   = new_start + cur_count + 1;
            return;
        }

        // Otherwise split first, then re-enter as an internal node.
        split_leaf(node_idx, depth);
    }

    // Internal node: recurse into every kid the obj_rect intersects.
    Node &node_ref = m_nodes[node_idx];  // re-bind in case of vector reallocation
    Rectangle obj_rect = m_objs[obj_idx].rect;
    for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
        int64_t kid = node_ref.kid_idx[iquad];
        Rectangle kid_arena = m_nodes[kid].arena;
        Rectangle inter = obj_rect.intersect(kid_arena);
        if (inter.is_non_empty_area())
            insert_into_node(kid, inter, depth + 1, obj_idx);
    }
}

void MemQuadTree::split_leaf(int64_t node_idx, unsigned depth) {
    Node &node = m_nodes[node_idx];
    Rectangle a = node.arena;
    int64_t mx = (a.x1 + a.x2) / 2;
    int64_t my = (a.y1 + a.y2) / 2;

    // Save the leaf's obj range, then mark internal.
    uint32_t start = node.leaf.obj_ptr_start;
    uint32_t end   = node.leaf.obj_ptr_end;
    node.is_leaf = false;

    // Create 4 kids. Push them and remember their indices; they are
    // pushed in NW, NE, SE, SW order matching R / pymisha _quadtree.py:
    //   NW = (x1,   my,   mx,   y2)
    //   NE = (mx,   my,   x2,   y2)
    //   SE = (mx,   y1,   x2,   my)
    //   SW = (x1,   y1,   mx,   my)
    Rectangle kid_arenas[NUM_QUADS] = {
        Rectangle(a.x1, my,   mx,   a.y2),
        Rectangle(mx,   my,   a.x2, a.y2),
        Rectangle(mx,   a.y1, a.x2, my),
        Rectangle(a.x1, a.y1, mx,   my),
    };
    int64_t kid_indices[NUM_QUADS];
    for (int i = 0; i < NUM_QUADS; ++i) {
        m_nodes.emplace_back(kid_arenas[i]);
        kid_indices[i] = static_cast<int64_t>(m_nodes.size()) - 1;
    }
    // m_nodes vector may have reallocated; re-bind.
    Node &n2 = m_nodes[node_idx];
    for (int i = 0; i < NUM_QUADS; ++i) n2.kid_idx[i] = kid_indices[i];

    // Re-insert the saved objects into the new internal node. (We do
    // NOT undo the parent's occupied_area accumulation; the splits and
    // re-inserts will accumulate again into the kids, and the parent's
    // value remains the union area which is its invariant.)
    for (uint32_t k = start; k < end; ++k) {
        uint32_t obj_ptr = m_obj_ptrs[k];
        RectObj &child_obj = m_objs[obj_ptr];
        Rectangle obj_rect = child_obj.rect;
        for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
            int64_t kid = m_nodes[node_idx].kid_idx[iquad];
            Rectangle inter = obj_rect.intersect(m_nodes[kid].arena);
            if (inter.is_non_empty_area()) {
                // Append at the back of m_obj_ptrs for the kid leaf.
                int64_t prev_obj_idx = static_cast<int64_t>(obj_ptr);
                insert_into_node(kid, inter, depth + 1, prev_obj_idx);
            }
        }
    }
}

// --- NNIterator port of R's StatQuadTree<T>::NNIterator -----------------
//
// Best-first heap traversal: at any moment, the heap contains a mix of
// nodes (not yet expanded) and objects (candidates), all keyed by their
// Manhattan distance to the query. The heap's ordering puts the smallest
// distance on top; on a distance tie, objects pop before nodes (the
// "Otherwise nearest neighbor query that covers the whole arena will
// cause all the tree nodes to be added to the heap" comment in R).
//
// begin() seeds the heap with the root node and runs next() once so
// that operator*() points at the first object. next() pops the current
// (which was returned to the caller) and then expands nodes until the
// top is an object.

MemQuadTree::NNIterator::NNIterator(MemQuadTree *parent)
    : m_parent(parent) {}

bool MemQuadTree::NNIterator::begin(const Rectangle &query,
                                     const Rectangle &excluded) {
    m_query = query;
    m_excluded = excluded;
    m_used_objs.assign(m_parent->m_objs.size(), 0);
    while (!m_neighbors.empty()) m_neighbors.pop();

    if (m_parent->m_nodes.empty()) return false;

    Node &root = m_parent->m_nodes[0];
    if (!root.arena.is_inside(m_excluded)) {
        Neighbor n;
        n.node = &root;
        n.obj  = nullptr;
        n.dist = root.arena.manhattan_dist(m_query);
        m_neighbors.push(n);
    }
    return next();
}

bool MemQuadTree::NNIterator::next() {
    if (m_neighbors.empty()) return false;

    // If we are currently pointing at an object (top is obj), pop it
    // before searching for the next one.
    if (m_neighbors.top().obj != nullptr)
        m_neighbors.pop();

    while (!m_neighbors.empty()) {
        if (m_neighbors.top().obj != nullptr)
            return true;  // we are now pointing at the next object

        // Top is a node; expand it.
        Node *node = m_neighbors.top().node;
        m_neighbors.pop();

        if (node->is_leaf) {
            for (uint32_t k = node->leaf.obj_ptr_start;
                 k < node->leaf.obj_ptr_end; ++k) {
                uint32_t obj_idx = m_parent->m_obj_ptrs[k];
                if (m_used_objs[obj_idx]) continue;
                RectObj &obj = m_parent->m_objs[obj_idx];
                if (obj.do_intersect(m_excluded)) continue;
                Neighbor np;
                np.node = nullptr;
                np.obj  = &obj;
                np.dist = obj.manhattan_dist(m_query);
                m_neighbors.push(np);
                m_used_objs[obj_idx] = 1;
            }
        } else {
            for (int iquad = 0; iquad < NUM_QUADS; ++iquad) {
                int64_t kid_idx = node->kid_idx[iquad];
                Node &kid = m_parent->m_nodes[kid_idx];
                if (kid.occupied_area > 0 && !kid.arena.is_inside(m_excluded)) {
                    Neighbor np;
                    np.node = &kid;
                    np.obj  = nullptr;
                    np.dist = kid.arena.manhattan_dist(m_query);
                    m_neighbors.push(np);
                }
            }
        }
    }
    return false;
}

const RectObj &MemQuadTree::NNIterator::operator*() const {
    return *m_neighbors.top().obj;
}

}  // namespace memqt
