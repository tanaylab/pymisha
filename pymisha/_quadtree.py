"""
Pure Python quad-tree implementation for reading/writing misha-compatible 2D track files.

Binary format (StatQuadTreeCached):
  File: [int32 signature] [uint64 num_objs] [int64 root_chunk_fpos] [chunks...] [root_chunk]
  Chunk: [int64 chunk_size] [int64 top_node_offset] [nodes/leaves...]
  Node (pack(8)): [bool is_leaf + 7pad] [Stat:32] [arena:32] [4 x int64 kid_ptrs]
  Leaf (pack(8)): [bool is_leaf + 7pad] [Stat:32] [arena:32] [uint32 num_objs + 4pad]
  Obj<Rectangle_val<float>>: [uint64 id] [int64 x1,y1,x2,y2] [float v + 4pad] = 48 bytes
  Obj<Point_val<float>>: [uint64 id] [int64 x,y] [float v + 4pad] = 32 bytes

Struct Stat (pack(8)): [int64 occupied_area] [double weighted_sum] [double min_val] [double max_val]
"""

import mmap
import struct

import numpy as np

# Format signatures from GenomeTrack.cpp
SIGNATURE_RECTS = -9
SIGNATURE_POINTS = -10

# Quad indices: NW=0, NE=1, SE=2, SW=3
NW, NE, SE, SW = 0, 1, 2, 3

_MAX_DEPTH = 20
_MAX_NODE_OBJS = 20


def _pack_stat(occupied_area, weighted_sum, min_val, max_val):
    """Pack a Stat struct: int64 + 3 doubles = 32 bytes."""
    return struct.pack("<qddd", occupied_area, weighted_sum, min_val, max_val)


def _pack_arena(x1, y1, x2, y2):
    """Pack arena Rectangle: 4 x int64 = 32 bytes."""
    return struct.pack("<qqqq", x1, y1, x2, y2)


def _pack_node_base(is_leaf, stat_bytes, arena_bytes):
    """Pack NodeBase: bool(1) + pad(7) + stat(32) + arena(32) = 72 bytes."""
    return struct.pack("<B", 1 if is_leaf else 0) + b"\x00" * 7 + stat_bytes + arena_bytes


def _pack_leaf(is_leaf, stat_bytes, arena_bytes, num_objs):
    """Pack Leaf struct: NodeBase(72) + uint32(4) + pad(4) = 80 bytes."""
    return _pack_node_base(is_leaf, stat_bytes, arena_bytes) + struct.pack("<I", num_objs) + b"\x00" * 4


def _pack_node(stat_bytes, arena_bytes, kid_ptrs):
    """Pack Node struct: NodeBase(72) + 4 x int64(32) = 104 bytes."""
    return _pack_node_base(False, stat_bytes, arena_bytes) + struct.pack("<qqqq", *kid_ptrs)


def _pack_rect_obj(obj_id, x1, y1, x2, y2, value):
    """Pack Obj<Rectangle_val<float>>: uint64(8) + 4xint64(32) + float(4) + pad(4) = 48 bytes."""
    return struct.pack("<Qqqqq", obj_id, x1, y1, x2, y2) + struct.pack("<f", value) + b"\x00" * 4


def _pack_point_obj(obj_id, x, y, value):
    """Pack Obj<Point_val<float>>: uint64(8) + 2xint64(16) + float(4) + pad(4) = 32 bytes."""
    return struct.pack("<Qqq", obj_id, x, y) + struct.pack("<f", value) + b"\x00" * 4


class _QuadNode:
    """In-memory quad-tree node for building before serialization."""
    __slots__ = ("is_leaf", "arena", "stat", "kids", "obj_indices")

    def __init__(self, arena):
        self.is_leaf = True
        self.arena = arena  # (x1, y1, x2, y2)
        self.stat = {"occupied_area": 0, "weighted_sum": 0.0,
                     "min_val": float("inf"), "max_val": float("-inf")}
        self.kids = [None, None, None, None]  # NW, NE, SE, SW
        self.obj_indices = []


def _rect_intersect(r1, r2):
    """Return intersection of two rectangles, or None if empty."""
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    if x1 < x2 and y1 < y2:
        return (x1, y1, x2, y2)
    return None


def _rect_area(r):
    return (r[2] - r[0]) * (r[3] - r[1])


def _do_rects_overlap(r1, r2):
    """Check if two rectangles overlap (non-empty intersection)."""
    return (r1[0] < r2[2] and r2[0] < r1[2] and
            r1[1] < r2[3] and r2[1] < r1[3])


class QuadTree:
    """
    In-memory quad-tree that can be serialized to misha StatQuadTreeCached format.

    Supports Rectangle_val<float> (RECTS) and Point_val<float> (POINTS) objects.
    """

    def __init__(self, x1, y1, x2, y2, is_points=False,
                 max_depth=_MAX_DEPTH, max_node_objs=_MAX_NODE_OBJS):
        self.root = _QuadNode((x1, y1, x2, y2))
        self.is_points = is_points
        self.max_depth = max_depth
        self.max_node_objs = max_node_objs
        self.objs = []  # list of (x1, y1, x2, y2, value) for rects or (x, y, value) for points

    def insert(self, obj):
        """Insert an object. For rects: (x1,y1,x2,y2,value). For points: (x,y,value)."""
        if self.is_points:
            x, y, v = obj
            if np.isnan(v):
                return
            obj_rect = (x, y, x + 1, y + 1)
        else:
            if np.isnan(obj[4]):
                return
            obj_rect = (obj[0], obj[1], obj[2], obj[3])
        inter = _rect_intersect(obj_rect, self.root.arena)
        if inter is None:
            return
        obj_idx = len(self.objs)
        self.objs.append(obj)
        self._insert(self.root, inter, 0, obj_idx, obj_rect)

    def _get_value(self, obj_idx):
        if self.is_points:
            return self.objs[obj_idx][2]
        return self.objs[obj_idx][4]

    def _get_rect(self, obj_idx):
        if self.is_points:
            x, y, v = self.objs[obj_idx]
            return (x, y, x + 1, y + 1)
        return self.objs[obj_idx][:4]

    def _insert(self, node, intersection, depth, obj_idx, obj_rect):
        # Update stats
        area = _rect_area(intersection)
        val = self._get_value(obj_idx)
        node.stat["weighted_sum"] += val * area
        node.stat["min_val"] = min(val, node.stat["min_val"])
        node.stat["max_val"] = max(val, node.stat["max_val"])
        node.stat["occupied_area"] += area

        if node.is_leaf:
            arena = node.arena
            w = arena[2] - arena[0]
            h = arena[3] - arena[1]
            if (len(node.obj_indices) < self.max_node_objs or
                    depth >= self.max_depth or w < 4 or h < 4):
                node.obj_indices.append(obj_idx)
                return
            # Split leaf into node
            self._split_leaf(node, depth)

        # Insert into children
        for iquad in range(4):
            kid = node.kids[iquad]
            inter = _rect_intersect(obj_rect, kid.arena)
            if inter is not None:
                self._insert(kid, inter, depth + 1, obj_idx, obj_rect)

    def _split_leaf(self, node, depth):
        """Convert a leaf to an internal node with 4 children."""
        x1, y1, x2, y2 = node.arena
        split_x = (x1 + x2) // 2
        split_y = (y1 + y2) // 2
        node.is_leaf = False

        node.kids[NW] = _QuadNode((x1, split_y, split_x, y2))
        node.kids[NE] = _QuadNode((split_x, split_y, x2, y2))
        node.kids[SE] = _QuadNode((split_x, y1, x2, split_y))
        node.kids[SW] = _QuadNode((x1, y1, split_x, split_y))

        # Re-insert existing objects into children.
        # Keep node.stat intact: it already includes old objects and the
        # split-triggering object from _insert() caller.
        old_indices = node.obj_indices
        node.obj_indices = []

        for oi in old_indices:
            obj_rect = self._get_rect(oi)
            for iquad in range(4):
                kid = node.kids[iquad]
                inter = _rect_intersect(obj_rect, kid.arena)
                if inter is not None:
                    self._insert(kid, inter, depth + 1, oi, obj_rect)

    def serialize(self, f, chunk_size=0):
        """
        Serialize the quad-tree to a file-like object in StatQuadTreeCached format.

        This writes the portion AFTER the format signature (which is written by the caller).
        Format: [uint64 num_objs] [int64 root_chunk_fpos] [chunk data]
        """
        num_objs = len(self.objs)
        f.write(struct.pack("<Q", num_objs))

        if num_objs == 0:
            return

        # Placeholder for root_chunk_start_fpos
        root_fpos_pos = f.tell()
        f.write(struct.pack("<q", 0))

        # Serialize the tree as a single chunk (chunk_size=0 means no splitting)
        chunk_start = f.tell()

        # Chunk header: [int64 chunk_size] [int64 top_node_offset]
        f.write(struct.pack("<q", 0))   # placeholder for chunk_size
        f.write(struct.pack("<q", 0))   # placeholder for top_node_offset

        # Serialize nodes depth-first, leaves first (bottom-up to know kid offsets)
        top_node_offset = self._serialize_node(f, self.root, chunk_start)

        chunk_end = f.tell()
        chunk_total_size = chunk_end - chunk_start

        # Patch chunk header
        f.seek(chunk_start)
        f.write(struct.pack("<q", chunk_total_size))
        f.write(struct.pack("<q", top_node_offset))

        # Patch root_chunk_fpos
        f.seek(root_fpos_pos)
        f.write(struct.pack("<q", chunk_start))

        f.seek(chunk_end)

    def _serialize_node(self, f, node, chunk_start):
        """Serialize a node, return offset from chunk_start."""
        if node.is_leaf:
            return self._serialize_leaf(f, node, chunk_start)

        # Serialize children first to get their offsets
        kid_offsets = [0, 0, 0, 0]
        for iquad in range(4):
            kid_offsets[iquad] = self._serialize_node(f, node.kids[iquad], chunk_start)

        # Write node
        offset = f.tell() - chunk_start
        stat_bytes = _pack_stat(
            node.stat["occupied_area"],
            node.stat["weighted_sum"],
            node.stat["min_val"],
            node.stat["max_val"],
        )
        arena_bytes = _pack_arena(*node.arena)
        f.write(_pack_node(stat_bytes, arena_bytes, kid_offsets))
        return offset

    def _serialize_leaf(self, f, node, chunk_start):
        """Serialize a leaf node and its objects, return offset from chunk_start."""
        offset = f.tell() - chunk_start
        stat_bytes = _pack_stat(
            node.stat["occupied_area"],
            node.stat["weighted_sum"],
            node.stat["min_val"],
            node.stat["max_val"],
        )
        arena_bytes = _pack_arena(*node.arena)
        n = len(node.obj_indices)
        f.write(_pack_leaf(True, stat_bytes, arena_bytes, n))

        # Write objects
        for oi in node.obj_indices:
            if self.is_points:
                x, y, v = self.objs[oi]
                f.write(_pack_point_obj(oi, x, y, v))
            else:
                x1, y1, x2, y2, v = self.objs[oi]
                f.write(_pack_rect_obj(oi, x1, y1, x2, y2, v))

        return offset


def write_2d_track_file(filepath, objects, arena, is_points=False):
    """
    Write a misha-compatible 2D track file for one chromosome pair.

    Parameters
    ----------
    filepath : str
        Output file path.
    objects : list
        For RECTS: list of (x1, y1, x2, y2, value) tuples.
        For POINTS: list of (x, y, value) tuples.
    arena : tuple
        (x1, y1, x2, y2) bounding rectangle (typically (0, 0, chromsize1, chromsize2)).
    is_points : bool
        Whether objects are points (True) or rectangles (False).
    """
    signature = SIGNATURE_POINTS if is_points else SIGNATURE_RECTS
    qtree = QuadTree(*arena, is_points=is_points)

    for obj in objects:
        qtree.insert(obj)

    with open(filepath, "wb") as f:
        f.write(struct.pack("<i", signature))
        qtree.serialize(f)


def verify_no_overlaps_2d(rects):
    """
    Verify that no two 2D rectangles overlap.

    Parameters
    ----------
    rects : list of (x1, y1, x2, y2) tuples

    Raises
    ------
    ValueError
        If overlapping rectangles are found.
    """
    if not rects:
        return

    # Sweep-line by x1 with active intervals whose x2 exceeds current x1.
    indexed = list(enumerate(rects))
    indexed.sort(key=lambda x: (x[1][0], x[1][2], x[1][1], x[1][3]))
    active = []
    for _, rect in indexed:
        x1, y1, x2, y2 = rect
        active = [r for r in active if r[2] > x1]
        for a in active:
            if _do_rects_overlap(a, rect):
                raise ValueError(
                    f"Overlapping 2D intervals found: "
                    f"({a[0]},{a[1]},{a[2]},{a[3]}) and "
                    f"({rect[0]},{rect[1]},{rect[2]},{rect[3]})"
                )
        active.append(rect)


# ---------------------------------------------------------------------------
# Quad-tree binary reader / spatial query
# ---------------------------------------------------------------------------

# Struct sizes (pack(8) alignment)
_NODEBASE_SIZE = 72   # bool(1)+pad(7) + Stat(32) + arena(32)
_LEAF_SIZE = 80       # NodeBase(72) + uint32(4) + pad(4)
_NODE_SIZE = 104      # NodeBase(72) + 4*int64(32)
_RECT_OBJ_SIZE = 48   # uint64(8) + 4*int64(32) + float(4) + pad(4)
_POINT_OBJ_SIZE = 32  # uint64(8) + 2*int64(16) + float(4) + pad(4)


def _unpack_node_base(data, offset):
    """Unpack NodeBase from data at offset. Returns (is_leaf, arena, new_offset)."""
    is_leaf = struct.unpack_from("<B", data, offset)[0] != 0
    # Skip pad(7) + Stat(32) = 39 bytes to get to arena at offset+40
    arena_off = offset + 8 + 32  # bool+pad(8) + stat(32)
    x1, y1, x2, y2 = struct.unpack_from("<qqqq", data, arena_off)
    return is_leaf, (x1, y1, x2, y2)


def _read_leaf_objects(data, offset, num_objs, is_points):
    """Read objects following a leaf header. Returns list of tuples."""
    objs = []
    pos = offset
    if is_points:
        for _ in range(num_objs):
            obj_id, x, y = struct.unpack_from("<Qqq", data, pos)
            val = struct.unpack_from("<f", data, pos + 24)[0]
            objs.append((obj_id, x, y, val))
            pos += _POINT_OBJ_SIZE
    else:
        for _ in range(num_objs):
            obj_id, x1, y1, x2, y2 = struct.unpack_from("<Qqqqq", data, pos)
            val = struct.unpack_from("<f", data, pos + 40)[0]
            objs.append((obj_id, x1, y1, x2, y2, val))
            pos += _RECT_OBJ_SIZE
    return objs


def _collect_all_objects(data, chunk_data_offset, node_offset, is_points, _visited=None, _depth=0, _out=None):
    """Recursively collect all objects from a quad-tree node."""
    if _visited is None:
        _visited = set()
    if _out is None:
        _out = []
    key = (chunk_data_offset, node_offset)
    if key in _visited:
        raise RecursionError("Corrupt 2D track file: cyclic quad-tree node reference")
    if _depth > 100000:
        raise RecursionError("Corrupt 2D track file: excessive quad-tree recursion depth")
    _visited.add(key)

    abs_offset = chunk_data_offset + node_offset
    is_leaf, arena = _unpack_node_base(data, abs_offset)

    try:
        if is_leaf:
            num_objs = struct.unpack_from("<I", data, abs_offset + _NODEBASE_SIZE)[0]
            _out.extend(_read_leaf_objects(data, abs_offset + _LEAF_SIZE, num_objs, is_points))
            return _out

        # Internal node: read 4 kid offsets
        kid_offsets = struct.unpack_from("<qqqq", data, abs_offset + _NODEBASE_SIZE)
        for kid_off in kid_offsets:
            if kid_off >= 0:
                # Positive: offset from chunk start
                _collect_all_objects(
                    data, chunk_data_offset, kid_off, is_points, _visited, _depth + 1, _out
                )
            else:
                # Negative: absolute file position (cross-chunk reference)
                _collect_all_objects(data, 0, -kid_off, is_points, _visited, _depth + 1, _out)
        return _out
    finally:
        _visited.discard(key)


def _query_node(data, chunk_data_offset, node_offset, is_points, qx1, qy1, qx2, qy2, seen_ids, _visited=None, _depth=0):
    """Recursively query a quad-tree node for objects intersecting the query rectangle."""
    if _visited is None:
        _visited = set()
    key = (chunk_data_offset, node_offset)
    if key in _visited:
        raise RecursionError("Corrupt 2D track file: cyclic quad-tree node reference")
    if _depth > 100000:
        raise RecursionError("Corrupt 2D track file: excessive quad-tree recursion depth")
    _visited.add(key)

    abs_offset = chunk_data_offset + node_offset
    is_leaf, arena = _unpack_node_base(data, abs_offset)

    # Prune: if node arena doesn't intersect query, skip
    ax1, ay1, ax2, ay2 = arena
    if ax1 >= qx2 or ax2 <= qx1 or ay1 >= qy2 or ay2 <= qy1:
        _visited.discard(key)
        return []

    if is_leaf:
        num_objs = struct.unpack_from("<I", data, abs_offset + _NODEBASE_SIZE)[0]
        raw_objs = _read_leaf_objects(data, abs_offset + _LEAF_SIZE, num_objs, is_points)
        results = []
        for obj in raw_objs:
            obj_id = obj[0]
            if obj_id in seen_ids:
                continue
            if is_points:
                _, ox, oy, val = obj
                # Point occupies [x, x+1) x [y, y+1)
                if ox < qx2 and ox + 1 > qx1 and oy < qy2 and oy + 1 > qy1:
                    seen_ids.add(obj_id)
                    results.append(obj)
            else:
                _, ox1, oy1, ox2, oy2, val = obj
                if ox1 < qx2 and ox2 > qx1 and oy1 < qy2 and oy2 > qy1:
                    seen_ids.add(obj_id)
                    results.append(obj)
        return results

    # Internal node: recurse into children
    kid_offsets = struct.unpack_from("<qqqq", data, abs_offset + _NODEBASE_SIZE)
    results = []
    try:
        for kid_off in kid_offsets:
            if kid_off >= 0:
                results.extend(
                    _query_node(data, chunk_data_offset, kid_off, is_points,
                                qx1, qy1, qx2, qy2, seen_ids, _visited, _depth + 1)
                )
            else:
                results.extend(
                    _query_node(data, 0, -kid_off, is_points,
                                qx1, qy1, qx2, qy2, seen_ids, _visited, _depth + 1)
                )
        return results
    finally:
        _visited.discard(key)


def _read_file_header(filepath):
    """Read a 2D track file header. Returns (is_points, num_objs, data_bytes)."""
    with open(filepath, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        signature = struct.unpack_from("<i", data, 0)[0]
        if signature == SIGNATURE_RECTS:
            is_points = False
        elif signature == SIGNATURE_POINTS:
            is_points = True
        else:
            raise ValueError(f"Unknown 2D track signature: {signature}")

        num_objs = struct.unpack_from("<Q", data, 4)[0]
        return is_points, num_objs, data
    except Exception:
        data.close()
        raise


def read_2d_track_objects(filepath):
    """
    Read all objects from a misha 2D track file.

    Parameters
    ----------
    filepath : str
        Path to the binary 2D track file.

    Returns
    -------
    tuple of (is_points, objects)
        is_points : bool
        objects : list of tuples
            For RECTS: (x1, y1, x2, y2, value)
            For POINTS: (x, y, value)
    """
    is_points, num_objs, data = _read_file_header(filepath)
    try:
        if num_objs == 0:
            return is_points, []

        root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
        top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]

        raw_objs = _collect_all_objects(data, root_chunk_fpos, top_node_offset, is_points)

        # Deduplicate by obj_id and strip obj_id from output
        seen = set()
        result = []
        for obj in raw_objs:
            obj_id = obj[0]
            if obj_id not in seen:
                seen.add(obj_id)
                if is_points:
                    _, x, y, val = obj
                    result.append((x, y, val))
                else:
                    _, x1, y1, x2, y2, val = obj
                    result.append((x1, y1, x2, y2, val))

        return is_points, result
    finally:
        data.close()


def query_2d_track_objects(filepath, qx1, qy1, qx2, qy2):
    """
    Query a misha 2D track file for objects intersecting a rectangle.

    Parameters
    ----------
    filepath : str
        Path to the binary 2D track file.
    qx1, qy1, qx2, qy2 : int
        Query rectangle bounds.

    Returns
    -------
    list of tuples
        For RECTS: (x1, y1, x2, y2, value)
        For POINTS: (x, y, value)
    """
    is_points, num_objs, data = _read_file_header(filepath)
    try:
        if num_objs == 0:
            return []

        root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
        top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]

        seen_ids = set()
        raw_objs = _query_node(data, root_chunk_fpos, top_node_offset, is_points,
                               qx1, qy1, qx2, qy2, seen_ids)

        # Strip obj_id from output
        result = []
        for obj in raw_objs:
            if is_points:
                _, x, y, val = obj
                result.append((x, y, val))
            else:
                _, x1, y1, x2, y2, val = obj
                result.append((x1, y1, x2, y2, val))

        return result
    finally:
        data.close()
