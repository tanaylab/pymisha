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

from __future__ import annotations

import contextlib
import mmap
import os
import struct
from typing import Any

import numpy as np

try:
    import _pymisha
    _HAS_CPP_QUADTREE = True
except ImportError:
    _HAS_CPP_QUADTREE = False

# Format signatures from GenomeTrack.cpp::FORMAT_SIGNATURES.
SIGNATURE_RECTS = -9
SIGNATURE_POINTS = -10
SIGNATURE_COMPUTED = -11

# Quad indices: NW=0, NE=1, SE=2, SW=3
NW, NE, SE, SW = 0, 1, 2, 3

_MAX_DEPTH = 20
_MAX_NODE_OBJS = 20


def _pack_stat(occupied_area: int, weighted_sum: float, min_val: float, max_val: float) -> bytes:
    """Pack a Stat struct: int64 + 3 doubles = 32 bytes."""
    return struct.pack("<qddd", occupied_area, weighted_sum, min_val, max_val)


def _pack_arena(x1: int, y1: int, x2: int, y2: int) -> bytes:
    """Pack arena Rectangle: 4 x int64 = 32 bytes."""
    return struct.pack("<qqqq", x1, y1, x2, y2)


def _pack_node_base(is_leaf: bool, stat_bytes: bytes, arena_bytes: bytes) -> bytes:
    """Pack NodeBase: bool(1) + pad(7) + stat(32) + arena(32) = 72 bytes."""
    return struct.pack("<B", 1 if is_leaf else 0) + b"\x00" * 7 + stat_bytes + arena_bytes


def _pack_leaf(is_leaf: bool, stat_bytes: bytes, arena_bytes: bytes, num_objs: int) -> bytes:
    """Pack Leaf struct: NodeBase(72) + uint32(4) + pad(4) = 80 bytes."""
    return _pack_node_base(is_leaf, stat_bytes, arena_bytes) + struct.pack("<I", num_objs) + b"\x00" * 4


def _pack_node(stat_bytes: bytes, arena_bytes: bytes, kid_ptrs: list[int]) -> bytes:
    """Pack Node struct: NodeBase(72) + 4 x int64(32) = 104 bytes."""
    return _pack_node_base(False, stat_bytes, arena_bytes) + struct.pack("<qqqq", *kid_ptrs)


def _pack_rect_obj(obj_id: int, x1: int, y1: int, x2: int, y2: int, value: float) -> bytes:
    """Pack Obj<Rectangle_val<float>>: uint64(8) + 4xint64(32) + float(4) + pad(4) = 48 bytes."""
    return struct.pack("<Qqqqq", obj_id, x1, y1, x2, y2) + struct.pack("<f", value) + b"\x00" * 4


def _pack_point_obj(obj_id: int, x: int, y: int, value: float) -> bytes:
    """Pack Obj<Point_val<float>>: uint64(8) + 2xint64(16) + float(4) + pad(4) = 32 bytes."""
    return struct.pack("<Qqq", obj_id, x, y) + struct.pack("<f", value) + b"\x00" * 4


class _QuadNode:
    """In-memory quad-tree node for building before serialization."""
    __slots__ = ("is_leaf", "arena", "stat", "kids", "obj_indices")

    def __init__(self, arena: tuple[int, int, int, int]) -> None:
        self.is_leaf = True
        self.arena = arena  # (x1, y1, x2, y2)
        self.stat: dict[str, Any] = {"occupied_area": 0, "weighted_sum": 0.0,
                                     "min_val": float("inf"), "max_val": float("-inf")}
        self.kids: list[_QuadNode | None] = [None, None, None, None]  # NW, NE, SE, SW
        self.obj_indices: list[int] = []


def _rect_intersect(r1: tuple[int, int, int, int], r2: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Return intersection of two rectangles, or None if empty."""
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[2], r2[2])
    y2 = min(r1[3], r2[3])
    if x1 < x2 and y1 < y2:
        return (x1, y1, x2, y2)
    return None


def _rect_area(r: tuple[int, int, int, int]) -> int:
    return (r[2] - r[0]) * (r[3] - r[1])


def _do_rects_overlap(r1: tuple[int, int, int, int], r2: tuple[int, int, int, int]) -> bool:
    """Check if two rectangles overlap (non-empty intersection)."""
    return (r1[0] < r2[2] and r2[0] < r1[2] and
            r1[1] < r2[3] and r2[1] < r1[3])


class QuadTree:
    """
    In-memory quad-tree that can be serialized to misha StatQuadTreeCached format.

    Supports Rectangle_val<float> (RECTS) and Point_val<float> (POINTS) objects.
    """

    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        is_points: bool = False,
        max_depth: int = _MAX_DEPTH,
        max_node_objs: int = _MAX_NODE_OBJS,
    ) -> None:
        self.root = _QuadNode((x1, y1, x2, y2))
        self.is_points = is_points
        self.max_depth = max_depth
        self.max_node_objs = max_node_objs
        self.objs: list[tuple[Any, ...]] = []  # list of (x1, y1, x2, y2, value) for rects or (x, y, value) for points

    def insert(self, obj: tuple[Any, ...]) -> None:
        """Insert an object. For rects: (x1,y1,x2,y2,value). For points: (x,y,value).

        NaN-valued objects are stored in ``self.objs`` and spatially indexed
        (so coordinate queries return them) but are excluded from per-node
        stat aggregates (weighted_sum / min_val / max_val / occupied_area),
        matching R: 2D tracks may contain NaN rects (e.g. from
        ``gtrack.lookup`` with ``force_binning=False``), and stat queries
        like 2D vtrack avg/min/max/sum/weighted.sum must skip them.
        """
        if self.is_points:
            x, y, v = obj
            obj_rect = (x, y, x + 1, y + 1)
        else:
            v = obj[4]
            obj_rect = (obj[0], obj[1], obj[2], obj[3])
        inter = _rect_intersect(obj_rect, self.root.arena)
        if inter is None:
            return
        obj_idx = len(self.objs)
        self.objs.append(obj)
        self._insert(self.root, inter, 0, obj_idx, obj_rect, is_nan=bool(np.isnan(v)))

    def _get_value(self, obj_idx: int) -> float:
        if self.is_points:
            return float(self.objs[obj_idx][2])
        return float(self.objs[obj_idx][4])

    def _get_rect(self, obj_idx: int) -> tuple[Any, ...]:
        if self.is_points:
            x, y, v = self.objs[obj_idx]
            return (x, y, x + 1, y + 1)
        return self.objs[obj_idx][:4]

    def _insert(
        self,
        node: _QuadNode,
        intersection: tuple[int, int, int, int],
        depth: int,
        obj_idx: int,
        obj_rect: tuple[Any, ...],
        is_nan: bool = False,
    ) -> None:
        # Update stats only for finite values: NaN rects are spatially indexed
        # but excluded from value aggregates (matching R). The invariant
        # "node.stat aggregates contain only finite values" is preserved, so
        # existing 2D-vtrack stat queries are unchanged on non-NaN tracks.
        if not is_nan:
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
            assert kid is not None
            inter = _rect_intersect(obj_rect, kid.arena)
            if inter is not None:
                self._insert(kid, inter, depth + 1, obj_idx, obj_rect, is_nan=is_nan)

    def _split_leaf(self, node: _QuadNode, depth: int) -> None:
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
            # Preserve NaN-vs-finite status when re-inserting after a split
            # so the new children's stat aggregates remain finite-only.
            oi_is_nan = bool(np.isnan(self._get_value(oi)))
            for iquad in range(4):
                kid = node.kids[iquad]
                assert kid is not None
                inter = _rect_intersect(obj_rect, kid.arena)
                if inter is not None:
                    self._insert(kid, inter, depth + 1, oi, obj_rect, is_nan=oi_is_nan)

    def query(self, qx1: int, qy1: int, qx2: int, qy2: int) -> list[int]:
        """Return indices of inserted objects that strictly overlap the query rect.

        Mirrors R misha's ``StatQuadTree::intersect`` membership test: each
        object whose stored rectangle has a non-empty (strict ``<``)
        intersection with ``(qx1, qy1, qx2, qy2)`` is returned exactly once,
        regardless of how many leaves it spans.  Touching edges do not count.
        The returned indices are sorted ascending.
        """
        found: set[int] = set()
        self._query_node(self.root, int(qx1), int(qy1), int(qx2), int(qy2), found)
        return sorted(found)

    def _query_node(
        self,
        node: _QuadNode,
        qx1: int,
        qy1: int,
        qx2: int,
        qy2: int,
        found: set[int],
    ) -> None:
        ax1, ay1, ax2, ay2 = node.arena
        # Prune subtrees whose arena does not overlap the query.
        if not (max(ax1, qx1) < min(ax2, qx2) and max(ay1, qy1) < min(ay2, qy2)):
            return
        if node.is_leaf:
            for oi in node.obj_indices:
                if oi in found:
                    continue
                rx1, ry1, rx2, ry2 = self._get_rect(oi)
                if max(rx1, qx1) < min(rx2, qx2) and max(ry1, qy1) < min(ry2, qy2):
                    found.add(oi)
            return
        for kid in node.kids:
            if kid is not None:
                self._query_node(kid, qx1, qy1, qx2, qy2, found)

    def _count_subtree_bytes(self, node: _QuadNode) -> int:
        """Estimate serialized byte size of a subtree (excluding chunk header)."""
        obj_size = 32 if self.is_points else 48
        if node.is_leaf:
            return 80 + len(node.obj_indices) * obj_size
        size = 104  # Node struct
        for iquad in range(4):
            kid = node.kids[iquad]
            assert kid is not None
            size += self._count_subtree_bytes(kid)
        return size

    def serialize(self, f: Any, chunk_size: int = 0) -> None:
        """
        Serialize the quad-tree to a file-like object in StatQuadTreeCached format.

        This writes the portion AFTER the format signature (which is written by the caller).
        Format: [uint64 num_objs] [int64 root_chunk_fpos] [chunk data]

        Parameters
        ----------
        chunk_size : int
            Maximum chunk size in bytes. Subtrees exceeding this size are
            written as separate chunks with cross-chunk negative kid pointers.
            0 means unlimited (single chunk).
        """
        num_objs = len(self.objs)
        f.write(struct.pack("<Q", num_objs))

        if num_objs == 0:
            return

        # Placeholder for root_chunk_start_fpos
        root_fpos_pos = f.tell()
        f.write(struct.pack("<q", 0))

        # Map node id -> chunk file position (for cross-chunk references)
        # A node gets an entry here if it was serialized as a separate chunk.
        node_chunk_fpos: dict[int, int] = {}

        if chunk_size > 0:
            # Multi-chunk: analyze subtree sizes and write large subtrees
            # as separate chunks (bottom-up, matching R misha algorithm)
            self._analyze_and_serialize(f, self.root, chunk_size, node_chunk_fpos)

        # Serialize root as a chunk (always)
        chunk_start = f.tell()
        f.write(struct.pack("<q", 0))   # placeholder for chunk_size
        f.write(struct.pack("<q", 0))   # placeholder for top_node_offset

        top_node_offset = self._serialize_node(f, self.root, chunk_start, node_chunk_fpos)

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

    def _analyze_and_serialize(
        self,
        f: Any,
        node: _QuadNode,
        chunk_size: int,
        node_chunk_fpos: dict[int, int],
    ) -> int:
        """Recursively analyze subtree sizes and write large subtrees as separate chunks.

        Follows R misha's analyze_n_serialize_subtree algorithm: bottom-up
        traversal. If a subtree exceeds chunk_size bytes, write it as a
        separate chunk and record its file position.
        """
        if node.is_leaf:
            return self._count_subtree_bytes(node)

        size = 104  # Node struct
        for iquad in range(4):
            kid = node.kids[iquad]
            assert kid is not None
            subtree_size = self._analyze_and_serialize(f, kid, chunk_size, node_chunk_fpos)
            if subtree_size > 0:
                size += subtree_size

        # If this subtree exceeds chunk_size, write it as a separate chunk
        # (but not the root — the root is always written by serialize())
        if size > chunk_size and node is not self.root:
            chunk_start = f.tell()
            f.write(struct.pack("<q", 0))   # placeholder for chunk_size
            f.write(struct.pack("<q", 0))   # placeholder for top_node_offset

            top_node_offset = self._serialize_node(f, node, chunk_start, node_chunk_fpos)

            chunk_end = f.tell()
            chunk_total_size = chunk_end - chunk_start

            # Patch chunk header
            f.seek(chunk_start)
            f.write(struct.pack("<q", chunk_total_size))
            f.write(struct.pack("<q", top_node_offset))
            f.seek(chunk_end)

            node_chunk_fpos[id(node)] = chunk_start
            return 0  # subtree was written as chunk, size = 0 for parent

        return size

    def _serialize_node(
        self,
        f: Any,
        node: _QuadNode,
        chunk_start: int,
        node_chunk_fpos: dict[int, int],
    ) -> int:
        """Serialize a node, return offset from chunk_start."""
        if node.is_leaf:
            return self._serialize_leaf(f, node, chunk_start)

        # Serialize children first to get their offsets
        kid_offsets = [0, 0, 0, 0]
        for iquad in range(4):
            kid = node.kids[iquad]
            assert kid is not None
            kid_fpos = node_chunk_fpos.get(id(kid))
            if kid_fpos is not None:
                # Kid was written as a separate chunk — store negative file position
                kid_offsets[iquad] = -kid_fpos
            else:
                kid_offsets[iquad] = self._serialize_node(f, kid, chunk_start, node_chunk_fpos)

        # Write node
        offset: int = int(f.tell()) - chunk_start
        stat_bytes = _pack_stat(
            int(node.stat["occupied_area"]),
            float(node.stat["weighted_sum"]),
            float(node.stat["min_val"]),
            float(node.stat["max_val"]),
        )
        arena_bytes = _pack_arena(*node.arena)
        f.write(_pack_node(stat_bytes, arena_bytes, kid_offsets))
        return offset

    def _serialize_leaf(self, f: Any, node: _QuadNode, chunk_start: int) -> int:
        """Serialize a leaf node and its objects, return offset from chunk_start."""
        offset: int = int(f.tell()) - chunk_start
        stat_bytes = _pack_stat(
            int(node.stat["occupied_area"]),
            float(node.stat["weighted_sum"]),
            float(node.stat["min_val"]),
            float(node.stat["max_val"]),
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


def write_2d_track_file(
    filepath: str,
    objects: list[tuple[Any, ...]],
    arena: tuple[int, int, int, int],
    is_points: bool = False,
    chunk_size: int = 0,
) -> None:
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
    chunk_size : int
        Maximum chunk size in bytes for multi-chunk serialization.
        Subtrees exceeding this size are written as separate chunks with
        cross-chunk negative kid pointers. 0 means single chunk (default).
        Recommended: ~10MB (10_000_000) for large datasets.
    """
    signature = SIGNATURE_POINTS if is_points else SIGNATURE_RECTS
    qtree = QuadTree(*arena, is_points=is_points)

    for obj in objects:
        qtree.insert(obj)

    with open(filepath, "wb") as f:
        f.write(struct.pack("<i", signature))
        qtree.serialize(f, chunk_size=chunk_size)


def verify_no_overlaps_2d(rects: list[tuple[int, int, int, int]]) -> None:
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
    active: list[tuple[int, int, int, int]] = []
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


def _unpack_node_base(data: bytes | mmap.mmap, offset: int) -> tuple[bool, tuple[int, int, int, int]]:
    """Unpack NodeBase from data at offset. Returns (is_leaf, arena)."""
    is_leaf = struct.unpack_from("<B", data, offset)[0] != 0
    # Skip pad(7) + Stat(32) = 39 bytes to get to arena at offset+40
    arena_off = offset + 8 + 32  # bool+pad(8) + stat(32)
    x1, y1, x2, y2 = struct.unpack_from("<qqqq", data, arena_off)
    return is_leaf, (x1, y1, x2, y2)


def _unpack_stat(data: bytes | mmap.mmap, offset: int) -> tuple[int, float, float, float]:
    """Unpack Stat struct from a NodeBase at *offset*.

    Stat is at offset+8 (after bool(1)+pad(7)): int64 occupied_area,
    double weighted_sum, double min_val, double max_val = 32 bytes.

    Returns (occupied_area, weighted_sum, min_val, max_val).
    """
    stat_off = offset + 8  # bool+pad(8)
    return struct.unpack_from("<qddd", data, stat_off)


def _read_leaf_objects(data: bytes | mmap.mmap, offset: int, num_objs: int, is_points: bool) -> list[tuple[Any, ...]]:
    """Read objects following a leaf header. Returns list of tuples."""
    objs: list[tuple[Any, ...]] = []
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


def _resolve_chunk_node(data: bytes | mmap.mmap, chunk_fpos: int) -> tuple[int, int]:
    """Read a chunk header and return (chunk_fpos, top_node_offset).

    A chunk starts with [int64 chunk_size][int64 top_node_offset].
    The top_node_offset is relative to chunk_fpos.

    Parameters
    ----------
    data : mmap or bytes
        The raw file data.
    chunk_fpos : int
        Absolute file position where the chunk starts.

    Returns
    -------
    tuple of (int, int)
        (chunk_fpos, top_node_offset) so the caller can recurse with
        chunk_data_offset=chunk_fpos and node_offset=top_node_offset.
    """
    top_node_offset = struct.unpack_from("<q", data, chunk_fpos + 8)[0]
    return chunk_fpos, top_node_offset


def _collect_all_objects(
    data: bytes | mmap.mmap,
    chunk_data_offset: int,
    node_offset: int,
    is_points: bool,
    _visited: set[tuple[int, int]] | None = None,
    _depth: int = 0,
    _out: list[tuple[Any, ...]] | None = None,
) -> list[tuple[Any, ...]]:
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
                # Negative: absolute file position of another chunk.
                # Read the chunk header to find its top_node_offset.
                cross_chunk_fpos, cross_top_node = _resolve_chunk_node(data, -kid_off)
                _collect_all_objects(
                    data, cross_chunk_fpos, cross_top_node, is_points, _visited, _depth + 1, _out
                )
        return _out
    finally:
        _visited.discard(key)


def _query_node(
    data: bytes | mmap.mmap,
    chunk_data_offset: int,
    node_offset: int,
    is_points: bool,
    qx1: int,
    qy1: int,
    qx2: int,
    qy2: int,
    seen_ids: set[int],
    _visited: set[tuple[int, int]] | None = None,
    _depth: int = 0,
) -> list[tuple[Any, ...]]:
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
                # Negative: absolute file position of another chunk.
                # Read the chunk header to find its top_node_offset.
                cross_chunk_fpos, cross_top_node = _resolve_chunk_node(data, -kid_off)
                results.extend(
                    _query_node(data, cross_chunk_fpos, cross_top_node, is_points,
                                qx1, qy1, qx2, qy2, seen_ids, _visited, _depth + 1)
                )
        return results
    finally:
        _visited.discard(key)


def _read_file_header(filepath: str) -> tuple[bool, int, mmap.mmap]:
    """Read a 2D track file header.

    Returns
    -------
    tuple of (is_points, num_objs, data_bytes)
        is_points : bool
            ``True`` for the POINTS signature; ``False`` otherwise (RECTS
            or COMPUTED). COMPUTED tracks share the 48-byte ``Obj`` layout
            with RECTS, so the same per-rect unpacker is used downstream.
        num_objs : int
            Number of stored objects (after the optional Computer2D header).
        data_bytes : mmap.mmap
            Memory-mapped file bytes (caller must ``close()`` it).
    """
    with open(filepath, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        signature = struct.unpack_from("<i", data, 0)[0]
        if signature == SIGNATURE_RECTS:
            is_points = False
            payload_offset = 4
        elif signature == SIGNATURE_POINTS:
            is_points = True
            payload_offset = 4
        elif signature == SIGNATURE_COMPUTED:
            is_points = False
            from ._computer2d import skip_computer2d_header

            payload_offset = skip_computer2d_header(data, offset=4)
        else:
            raise ValueError(f"Unknown 2D track signature: {signature}")

        num_objs = struct.unpack_from("<Q", data, payload_offset)[0]
        return is_points, num_objs, data
    except Exception:
        data.close()
        raise


def _file_track_kind(filepath: str) -> str:
    """Return ``"RECTS"`` / ``"POINTS"`` / ``"COMPUTED"`` for a 2D track file.

    Lightweight signature-only sniff (no payload parsing).
    """
    with open(filepath, "rb") as fh:
        head = fh.read(4)
    if len(head) < 4:
        raise ValueError(f"Truncated 2D track file: {filepath}")
    sig = struct.unpack_from("<i", head, 0)[0]
    if sig == SIGNATURE_RECTS:
        return "RECTS"
    if sig == SIGNATURE_POINTS:
        return "POINTS"
    if sig == SIGNATURE_COMPUTED:
        return "COMPUTED"
    raise ValueError(f"Unknown 2D track signature: {sig}")


def _payload_offset(data: bytes | mmap.mmap) -> int:
    """Return the byte offset where the StatQuadTreeCached payload begins.

    For RECTS / POINTS this is 4 (just past the signature); for COMPUTED
    it skips the Computer2D header so the offset lands on ``num_objs``.
    """
    signature = struct.unpack_from("<i", data, 0)[0]
    if signature in (SIGNATURE_RECTS, SIGNATURE_POINTS):
        return 4
    if signature == SIGNATURE_COMPUTED:
        from ._computer2d import skip_computer2d_header

        return skip_computer2d_header(data, offset=4)
    raise ValueError(f"Unknown 2D track signature: {signature}")


def read_2d_track_objects(filepath: str) -> tuple[bool, list[tuple[Any, ...]]]:
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

        # RECTS / POINTS: payload starts at byte 4 (signature only), so the
        # root_chunk_fpos lives at byte 12 (= 4 sig + 8 num_objs).  For
        # COMPUTED the Computer2D header pushes the payload forward;
        # _payload_offset returns the byte position of num_objs there.
        payload_offset = _payload_offset(data)
        root_chunk_fpos = struct.unpack_from("<q", data, payload_offset + 8)[0]
        top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]

        raw_objs = _collect_all_objects(data, root_chunk_fpos, top_node_offset, is_points)

        # Deduplicate by obj_id and strip obj_id from output
        seen: set[Any] = set()
        result: list[tuple[Any, ...]] = []
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


def query_2d_track_opened(
    data: bytes | mmap.mmap,
    is_points: bool,
    num_objs: int,
    root_chunk_fpos: int,
    qx1: int,
    qy1: int,
    qx2: int,
    qy2: int,
    band: tuple[int, int] | None = None,
) -> list[tuple[Any, ...]]:
    """
    Query a pre-opened 2D track mmap for objects intersecting a rectangle.

    This avoids repeated file open/mmap/close when querying multiple intervals
    on the same chrom pair.

    Parameters
    ----------
    data : mmap or bytes
        Memory-mapped file data (from ``_read_file_header``).
    is_points : bool
        Whether the track stores points (True) or rectangles (False).
    num_objs : int
        Total number of objects in the file (from header).
    root_chunk_fpos : int
        File position of the root chunk (``struct.unpack_from("<q", data, 12)[0]``).
    qx1, qy1, qx2, qy2 : int
        Query rectangle bounds.
    band : tuple of (int, int) or None
        If not None, ``(d1, d2)`` diagonal band filter.

    Returns
    -------
    list of tuples
        For RECTS: (x1, y1, x2, y2, value)
        For POINTS: (x, y, value)
    """
    if num_objs == 0:
        return []

    # The C++ fast path assumes the RECTS / POINTS file header layout
    # (signature at 0, num_objs at 4, root_chunk_fpos at 12).  COMPUTED
    # files have an extra Computer2D header in front (sig at 0, type byte
    # at 4, then num_objs/root_chunk_fpos pushed forward); the C++ side
    # has no Computer2D dispatcher port, so route COMPUTED through the
    # pure-Python walker below (still mmap-fast for the test fixture).
    _sig = struct.unpack_from("<i", data, 0)[0]
    _is_computed = _sig == SIGNATURE_COMPUTED

    # C++ fast path
    if _HAS_CPP_QUADTREE and not _is_computed:
        try:
            has_band = 1 if band is not None else 0
            band_d1 = band[0] if band else 0
            band_d2 = band[1] if band else 0
            r = _pymisha.pm_quadtree_query_objects(
                data, int(qx1), int(qy1), int(qx2), int(qy2),
                1 if is_points else 0, has_band, int(band_d1), int(band_d2))
            n = len(r["id"])
            result: list[tuple[Any, ...]] = []
            if is_points:
                for i in range(n):
                    result.append((int(r["x1"][i]), int(r["y1"][i]), float(r["val"][i])))
            else:
                for i in range(n):
                    result.append((int(r["x1"][i]), int(r["y1"][i]),
                                   int(r["x2"][i]), int(r["y2"][i]),
                                   float(r["val"][i])))
            return result
        except Exception:
            pass  # Fall back to Python implementation

    top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]

    seen_ids: set[int] = set()
    raw_objs = _query_node(data, root_chunk_fpos, top_node_offset, is_points,
                           qx1, qy1, qx2, qy2, seen_ids)

    # Strip obj_id from output
    result2: list[tuple[Any, ...]] = []
    for obj in raw_objs:
        if is_points:
            _, x, y, val = obj
            result2.append((x, y, val))
        else:
            _, x1, y1, x2, y2, val = obj
            result2.append((x1, y1, x2, y2, val))
    result = result2

    # Apply band filter if needed (Python fallback doesn't handle band)
    if band is not None:
        d1, d2 = band
        if is_points:
            result = [(x, y, v) for x, y, v in result if d1 <= (x - y) < d2]
        else:
            result = [(x1, y1, x2, y2, v) for x1, y1, x2, y2, v in result
                      if (x2 - y1 > d1) and (x1 - y2 + 1 < d2)]

    return result


def _query_node_stats(data: bytes | mmap.mmap, chunk_data_offset: int, node_offset: int, is_points: bool,
                      qx1: int, qy1: int, qx2: int, qy2: int, stat: list[float | int],
                      _visited: set[tuple[int, int]] | None = None, _depth: int = 0) -> None:
    """Hybrid quad-tree stat traversal matching R misha's ``get_stat``.

    For internal nodes whose arena is *fully contained* by the query rectangle,
    the pre-computed node stats are used directly (O(1) per subtree).  For
    partially overlapping nodes, recursion continues.  Leaf nodes enumerate
    objects and compute intersection-based stats **clamped to the leaf's arena**
    so that each node only accounts for the portion of objects within its own
    bounding box (avoiding double-counting of objects that span siblings).

    This matches R misha's ``obj.intersect(rect, leaf->arena)`` 3-way
    intersection approach.

    *stat* is a 4-element list ``[occupied_area, weighted_sum, min_val, max_val]``
    that is updated in-place.
    """
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
    ax1, ay1, ax2, ay2 = arena

    # Prune: no intersection with query
    if ax1 >= qx2 or ax2 <= qx1 or ay1 >= qy2 or ay2 <= qy1:
        _visited.discard(key)
        return

    # Effective query rect clamped to this node's arena — ensures each node
    # only accounts for the portion of objects within its own bounding box.
    eqx1 = max(qx1, ax1)
    eqy1 = max(qy1, ay1)
    eqx2 = min(qx2, ax2)
    eqy2 = min(qy2, ay2)

    try:
        if is_leaf:
            # Enumerate objects and compute intersection stats, clamped to arena.
            num_objs = struct.unpack_from("<I", data, abs_offset + _NODEBASE_SIZE)[0]
            raw_objs = _read_leaf_objects(data, abs_offset + _LEAF_SIZE, num_objs, is_points)
            for obj in raw_objs:
                if is_points:
                    _, ox, oy, val = obj
                    # NaN-valued objects are stored on disk and returned by
                    # spatial queries, but must be skipped from stat aggregates
                    # (matching R: 2D tracks may contain NaN rects, and
                    # avg/min/max/sum/weighted.sum skip them).
                    if np.isnan(val):
                        continue
                    # Point inside both query and arena?
                    if (eqx1 <= ox < eqx2 and eqy1 <= oy < eqy2):
                        stat[0] += 1           # occupied_area
                        stat[1] += val         # weighted_sum
                        if val < stat[2]:
                            stat[2] = val      # min_val
                        if val > stat[3]:
                            stat[3] = val      # max_val
                else:
                    _, ox1, oy1, ox2, oy2, val = obj
                    if np.isnan(val):
                        continue  # NaN-rect: stored on disk, skipped from stats
                    # Intersection clamped to effective query (query ∩ arena)
                    inter = (max(0, min(eqx2, ox2) - max(eqx1, ox1))
                             * max(0, min(eqy2, oy2) - max(eqy1, oy1)))
                    if inter > 0:
                        stat[0] += inter           # occupied_area
                        stat[1] += val * inter     # weighted_sum
                        if val < stat[2]:
                            stat[2] = val
                        if val > stat[3]:
                            stat[3] = val
        else:
            # Internal node: check each child quadrant
            kid_offsets = struct.unpack_from("<qqqq", data, abs_offset + _NODEBASE_SIZE)
            for kid_off in kid_offsets:
                if kid_off >= 0:
                    child_abs = chunk_data_offset + kid_off
                    child_chunk = chunk_data_offset
                    child_node = kid_off
                else:
                    child_chunk, child_node = _resolve_chunk_node(data, -kid_off)
                    child_abs = child_chunk + child_node

                _, child_arena = _unpack_node_base(data, child_abs)
                cx1, cy1, cx2, cy2 = child_arena

                # Skip if child doesn't intersect query
                if cx1 >= qx2 or cx2 <= qx1 or cy1 >= qy2 or cy2 <= qy1:
                    continue

                # Fast path: child fully inside query -> use pre-computed stats
                if cx1 >= qx1 and cy1 >= qy1 and cx2 <= qx2 and cy2 <= qy2:
                    c_occ, c_ws, c_min, c_max = _unpack_stat(data, child_abs)
                    if c_occ > 0:
                        stat[0] += c_occ
                        stat[1] += c_ws
                        if c_min < stat[2]:
                            stat[2] = c_min
                        if c_max > stat[3]:
                            stat[3] = c_max
                    continue

                # Partial overlap: recurse
                _query_node_stats(data, child_chunk, child_node, is_points,
                                  qx1, qy1, qx2, qy2, stat,
                                  _visited, _depth + 1)
    finally:
        _visited.discard(key)


def query_2d_track_stats(data: bytes | mmap.mmap, is_points: bool, num_objs: int, root_chunk_fpos: int,
                         qx1: int, qy1: int, qx2: int, qy2: int, band: tuple[int, int] | None = None) -> dict[str, Any]:
    """
    Query a pre-opened 2D track mmap and return aggregated stats for a query rectangle.

    Uses a hybrid quad-tree traversal matching R misha's ``get_stat`` algorithm:
    for subtrees fully contained by the query, pre-computed node statistics are
    used directly (O(1) per subtree), avoiding object enumeration.

    Parameters
    ----------
    data : mmap or bytes
        Memory-mapped file data (from ``_read_file_header``).
    is_points : bool
        Whether the track stores points (True) or rectangles (False).
    num_objs : int
        Total number of objects in the file (from header).
    root_chunk_fpos : int
        File position of the root chunk.
    qx1, qy1, qx2, qy2 : int
        Query rectangle bounds.
    band : tuple of (int, int) or None
        If not None, ``(d1, d2)`` diagonal band filter. When a band is
        specified, the fast node-stats path cannot be used and a full
        object-level enumeration is performed instead.

    Returns
    -------
    dict
        ``{"occupied_area": int, "weighted_sum": float,
        "min_val": float, "max_val": float}``.
        When no objects contribute, *min_val* and *max_val* are ``float("nan")``.
    """
    if num_objs == 0:
        return {"occupied_area": 0, "weighted_sum": 0.0,
                "min_val": float("nan"), "max_val": float("nan")}

    # C++ fast path — gated on RECTS / POINTS signature (the binding
    # assumes the standard header layout and would seg-fault on COMPUTED).
    _sig = struct.unpack_from("<i", data, 0)[0]
    if _HAS_CPP_QUADTREE and _sig != SIGNATURE_COMPUTED:
        try:
            has_band = 1 if band is not None else 0
            band_d1 = band[0] if band else 0
            band_d2 = band[1] if band else 0
            return dict(_pymisha.pm_quadtree_query_stats(
                data, int(qx1), int(qy1), int(qx2), int(qy2),
                1 if is_points else 0, has_band, int(band_d1), int(band_d2)))
        except Exception:
            pass  # Fall back to Python implementation

    if band is not None:
        # Band filtering requires per-object inspection -- fall back to object
        # enumeration.  The node-level stats don't account for band constraints.
        return _query_2d_track_stats_with_band(
            data, is_points, num_objs, root_chunk_fpos,
            qx1, qy1, qx2, qy2, band)

    top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]

    # stat = [occupied_area, weighted_sum, min_val, max_val]
    stat = [0, 0.0, float("inf"), float("-inf")]

    _query_node_stats(data, root_chunk_fpos, top_node_offset, is_points,
                      qx1, qy1, qx2, qy2, stat)

    if stat[0] == 0:
        return {"occupied_area": 0, "weighted_sum": 0.0,
                "min_val": float("nan"), "max_val": float("nan")}

    return {"occupied_area": stat[0], "weighted_sum": stat[1],
            "min_val": stat[2], "max_val": stat[3]}


def query_2d_track_stats_batch(data: bytes | mmap.mmap, is_points: bool, num_objs: int, root_chunk_fpos: int,
                                rects: np.ndarray, band: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """
    Batch stats query: compute aggregated stats for N query rectangles in one call.

    Parameters
    ----------
    data : mmap or bytes
        Memory-mapped file data (from ``_read_file_header``).
    is_points : bool
        Whether the track stores points (True) or rectangles (False).
    num_objs : int
        Total number of objects in the file (from header).
    root_chunk_fpos : int
        File position of the root chunk.
    rects : numpy.ndarray
        (N, 4) int64 array of query rectangles ``[qx1, qy1, qx2, qy2]``.
    band : tuple of (int, int) or None
        If not None, ``(d1, d2)`` diagonal band filter.

    Returns
    -------
    dict
        ``{"occupied_area": ndarray(N, int64), "weighted_sum": ndarray(N, float64),
        "min_val": ndarray(N, float64), "max_val": ndarray(N, float64)}``.
    """
    n = len(rects) if rects is not None else 0
    if n == 0 or num_objs == 0:
        return {
            "occupied_area": np.zeros(n, dtype=np.int64),
            "weighted_sum": np.full(n, np.nan),
            "min_val": np.full(n, np.nan),
            "max_val": np.full(n, np.nan),
        }

    rects_arr = np.ascontiguousarray(rects, dtype=np.int64).reshape(-1, 4)

    # C++ fast path: gated on the file's signature being RECTS / POINTS
    # (the C++ binding assumes the standard 12-byte header; COMPUTED has
    # an extra Computer2D byte block that pushes num_objs / root_chunk
    # forward and would seg-fault the binding).
    _sig = struct.unpack_from("<i", data, 0)[0]
    _is_computed = _sig == SIGNATURE_COMPUTED

    if _HAS_CPP_QUADTREE and not _is_computed:
        try:
            has_band = 1 if band is not None else 0
            band_d1 = band[0] if band else 0
            band_d2 = band[1] if band else 0
            return dict(_pymisha.pm_quadtree_query_stats_batch(
                data, rects_arr,
                1 if is_points else 0,
                has_band, int(band_d1), int(band_d2)))
        except Exception:
            pass  # Fall back to Python per-rect loop

    # Python fallback: loop over rects
    occ = np.zeros(n, dtype=np.int64)
    ws = np.full(n, np.nan)
    mn = np.full(n, np.nan)
    mx = np.full(n, np.nan)
    for i in range(n):
        qx1, qy1, qx2, qy2 = int(rects_arr[i, 0]), int(rects_arr[i, 1]), int(rects_arr[i, 2]), int(rects_arr[i, 3])
        s = query_2d_track_stats(data, is_points, num_objs, root_chunk_fpos,
                                  qx1, qy1, qx2, qy2, band=band)
        occ[i] = s["occupied_area"]
        ws[i] = s["weighted_sum"]
        mn[i] = s["min_val"]
        mx[i] = s["max_val"]
    return {"occupied_area": occ, "weighted_sum": ws, "min_val": mn, "max_val": mx}


def _query_2d_track_stats_with_band(data: bytes | mmap.mmap, is_points: bool, num_objs: int, root_chunk_fpos: int,
                                     qx1: int, qy1: int, qx2: int, qy2: int, band: tuple[int, int]) -> dict[str, Any]:
    """Fall-back stat query when a diagonal band filter is active.

    Band filtering cannot leverage node-level stats (the stored stats don't
    account for band constraints), so we enumerate objects via
    ``query_2d_track_opened`` and compute stats per-object.
    """
    objs = query_2d_track_opened(data, is_points, num_objs, root_chunk_fpos,
                                 qx1, qy1, qx2, qy2)
    d1, d2 = band
    occupied_area = 0
    weighted_sum = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    if is_points:
        for x, y, val in objs:
            if np.isnan(val):
                continue  # NaN-rect: spatially returned, excluded from stats
            if not (d1 <= (x - y) < d2):
                continue
            occupied_area += 1
            weighted_sum += val
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    else:
        for ox1, oy1, ox2, oy2, val in objs:
            if np.isnan(val):
                continue  # NaN-rect: spatially returned, excluded from stats
            if not ((ox2 - oy1 > d1) and (ox1 - oy2 + 1 < d2)):
                continue
            inter = (max(0, min(qx2, ox2) - max(qx1, ox1))
                     * max(0, min(qy2, oy2) - max(qy1, oy1)))
            if inter > 0:
                occupied_area += inter
                weighted_sum += val * inter
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

    if occupied_area == 0 and min_val == float("inf"):
        return {"occupied_area": 0, "weighted_sum": 0.0,
                "min_val": float("nan"), "max_val": float("nan")}

    return {"occupied_area": occupied_area, "weighted_sum": weighted_sum,
            "min_val": min_val, "max_val": max_val}


def query_2d_track_opened_arrays(
    data: bytes | mmap.mmap,
    is_points: bool,
    num_objs: int,
    root_chunk_fpos: int,
    qx1: int,
    qy1: int,
    qx2: int,
    qy2: int,
    band: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Query a pre-opened 2D track mmap and return numpy arrays directly.

    Unlike ``query_2d_track_opened``, this returns raw numpy arrays without
    converting to Python tuples, which is much faster for vectorized consumers.

    Parameters
    ----------
    data : mmap or bytes
        Memory-mapped file data (from ``_read_file_header``).
    is_points : bool
        Whether the track stores points (True) or rectangles (False).
    num_objs : int
        Total number of objects in the file (from header).
    root_chunk_fpos : int
        File position of the root chunk.
    qx1, qy1, qx2, qy2 : int
        Query rectangle bounds.
    band : tuple of (int, int) or None
        If not None, ``(d1, d2)`` diagonal band filter.

    Returns
    -------
    dict
        ``{"x1": ndarray(int64), "y1": ndarray(int64),
        "x2": ndarray(int64), "y2": ndarray(int64),
        "val": ndarray(float32)}``.
        For POINTS, x2 = x1 + 1, y2 = y1 + 1.
        Empty arrays if no objects found.
    """
    _empty: dict[str, np.ndarray] = {
        "x1": np.empty(0, dtype=np.int64),
        "y1": np.empty(0, dtype=np.int64),
        "x2": np.empty(0, dtype=np.int64),
        "y2": np.empty(0, dtype=np.int64),
        "val": np.empty(0, dtype=np.float32),
    }

    if num_objs == 0:
        return _empty

    # C++ fast path — gated on RECTS / POINTS signature.
    _sig = struct.unpack_from("<i", data, 0)[0]
    if _HAS_CPP_QUADTREE and _sig != SIGNATURE_COMPUTED:
        try:
            has_band = 1 if band is not None else 0
            band_d1 = band[0] if band else 0
            band_d2 = band[1] if band else 0
            r = _pymisha.pm_quadtree_query_objects(
                data, int(qx1), int(qy1), int(qx2), int(qy2),
                1 if is_points else 0, has_band, int(band_d1), int(band_d2))
            n = len(r["id"])
            if n == 0:
                return _empty
            result = {
                "x1": r["x1"].astype(np.int64),
                "y1": r["y1"].astype(np.int64),
                "x2": r["x2"].astype(np.int64),
                "y2": r["y2"].astype(np.int64),
                "val": r["val"],
            }
            if is_points:
                result["x2"] = result["x1"] + 1
                result["y2"] = result["y1"] + 1
            return result
        except Exception:
            pass  # Fall back to Python implementation

    # Python fallback: query tuples, convert to arrays
    objs = query_2d_track_opened(data, is_points, num_objs, root_chunk_fpos,
                                  qx1, qy1, qx2, qy2, band=band)
    if not objs:
        return _empty

    if is_points:
        arr = np.array(objs, dtype=np.float64)
        return {
            "x1": arr[:, 0].astype(np.int64),
            "y1": arr[:, 1].astype(np.int64),
            "x2": arr[:, 0].astype(np.int64) + 1,
            "y2": arr[:, 1].astype(np.int64) + 1,
            "val": arr[:, 2].astype(np.float32),
        }

    arr = np.array(objs, dtype=np.float64)
    return {
        "x1": arr[:, 0].astype(np.int64),
        "y1": arr[:, 1].astype(np.int64),
        "x2": arr[:, 2].astype(np.int64),
        "y2": arr[:, 3].astype(np.int64),
        "val": arr[:, 4].astype(np.float32),
    }


def query_2d_track_objects(filepath: str, qx1: int, qy1: int, qx2: int, qy2: int) -> list[tuple[Any, ...]]:
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

        payload_offset = _payload_offset(data)
        root_chunk_fpos = struct.unpack_from("<q", data, payload_offset + 8)[0]
        return query_2d_track_opened(data, is_points, num_objs, root_chunk_fpos,
                                     qx1, qy1, qx2, qy2)
    finally:
        data.close()


# ---------------------------------------------------------------------------
# Indexed 2D track support (track.idx + track.dat)
# ---------------------------------------------------------------------------

# Cache of IndexedTrack2DReader instances keyed by track directory path
_indexed_2d_cache: dict[str, IndexedTrack2DReader] = {}


class IndexedTrack2DReader:
    """Reader for indexed 2D tracks (track.idx + track.dat format).

    An indexed 2D track stores all chromosome-pair data in a single
    concatenated file (``track.dat``), with a lookup index
    (``track.idx``) that records the offset and length of each pair's
    data within ``track.dat``.

    This class mmaps ``track.dat`` once and provides per-pair buffer
    slices that are compatible with the existing quad-tree query
    functions (``query_2d_track_opened``, ``query_2d_track_stats``,
    etc.).

    Parameters
    ----------
    track_dir : str
        Path to the track directory containing ``track.idx`` and
        ``track.dat``.
    """

    __slots__ = ("_track_dir", "_dat_mmap", "_dat_file", "_pair_map",
                 "_is_points", "_loaded")

    def __init__(self, track_dir: str) -> None:
        self._track_dir = track_dir
        self._dat_mmap: mmap.mmap | None = None
        self._dat_file: Any = None
        self._pair_map: dict[tuple[int, int], tuple[int, int]] = {}  # (chrom1_id, chrom2_id) -> (offset, length)
        self._is_points: bool = False
        self._loaded: bool = False
        self._load()

    def _load(self) -> None:
        """Load the index and mmap the data file."""
        idx_path = os.path.join(self._track_dir, "track.idx")
        dat_path = os.path.join(self._track_dir, "track.dat")

        if not os.path.exists(idx_path) or not os.path.exists(dat_path):
            return

        if not _HAS_CPP_QUADTREE:
            return

        try:
            info = _pymisha.pm_track2d_index_info(self._track_dir)
        except Exception:
            return

        if not info.get("loaded"):
            return

        self._is_points = info["track_type"] == "POINTS"

        for pair in info["pairs"]:
            key = (int(pair["chrom1_id"]), int(pair["chrom2_id"]))
            self._pair_map[key] = (int(pair["offset"]), int(pair["length"]))

        # mmap track.dat
        dat_size = os.path.getsize(dat_path)
        if dat_size == 0:
            return

        self._dat_file = open(dat_path, "rb")  # noqa: SIM115
        self._dat_mmap = mmap.mmap(
            self._dat_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self._loaded = True

    @property
    def loaded(self) -> bool:
        """Whether the indexed track was successfully loaded."""
        return self._loaded

    @property
    def is_points(self) -> bool:
        """Whether the track stores points (True) or rectangles (False)."""
        return self._is_points

    def get_pair_data(
        self, chrom1_id: int, chrom2_id: int,
    ) -> tuple[bool, int, bytes, int] | None:
        """Return a buffer slice for a chromosome pair.

        Parameters
        ----------
        chrom1_id, chrom2_id : int
            Numeric chromosome IDs (0-based, matching the genome database
            ordering).

        Returns
        -------
        tuple of (is_points, num_objs, data, root_chunk_fpos) or None
            Same format as what ``_read_file_header`` returns (plus
            ``root_chunk_fpos``), suitable for passing directly to
            ``query_2d_track_opened`` and friends.  Returns ``None`` if
            the pair is not present in the index.
        """
        if not self._loaded:
            return None

        key = (chrom1_id, chrom2_id)
        entry = self._pair_map.get(key)
        if entry is None:
            return None

        offset, length = entry
        if length == 0:
            return None

        # Create a bytes slice from the mmap. Using bytes() here
        # creates a copy, but the per-pair data is typically small
        # (KB to low MB) and this ensures the buffer is fully
        # independent — no lifecycle coupling with the mmap.
        assert self._dat_mmap is not None
        pair_data = bytes(self._dat_mmap[offset:offset + length])

        # Parse the signature + header from the pair data
        if len(pair_data) < 12:
            return None

        signature = struct.unpack_from("<i", pair_data, 0)[0]
        if signature == SIGNATURE_RECTS:
            is_points = False
            payload_offset = 4
        elif signature == SIGNATURE_POINTS:
            is_points = True
            payload_offset = 4
        elif signature == SIGNATURE_COMPUTED:
            is_points = False
            from ._computer2d import skip_computer2d_header

            payload_offset = skip_computer2d_header(pair_data, offset=4)
        else:
            return None

        if len(pair_data) < payload_offset + 8:
            return None

        num_objs = struct.unpack_from("<Q", pair_data, payload_offset)[0]
        if num_objs == 0:
            return (is_points, 0, pair_data, 0)

        root_chunk_fpos = struct.unpack_from("<q", pair_data, payload_offset + 8)[0]
        return (is_points, num_objs, pair_data, root_chunk_fpos)

    def close(self) -> None:
        """Release the mmap and file handle."""
        if self._dat_mmap is not None:
            with contextlib.suppress(Exception):
                self._dat_mmap.close()
            self._dat_mmap = None
        if self._dat_file is not None:
            with contextlib.suppress(Exception):
                self._dat_file.close()
            self._dat_file = None
        self._loaded = False

    def __del__(self) -> None:
        self.close()


def _get_indexed_reader(track_dir: str) -> IndexedTrack2DReader | None:
    """Get or create a cached IndexedTrack2DReader for a track directory.

    Returns ``None`` if the track directory does not have an indexed
    format (no ``track.idx``).
    """
    idx_path = os.path.join(track_dir, "track.idx")
    if not os.path.exists(idx_path):
        return None

    reader = _indexed_2d_cache.get(track_dir)
    if reader is not None and reader.loaded:
        return reader

    reader = IndexedTrack2DReader(track_dir)
    if reader.loaded:
        _indexed_2d_cache[track_dir] = reader
        return reader

    return None


def clear_indexed_2d_cache() -> None:
    """Clear all cached IndexedTrack2DReader instances."""
    for reader in _indexed_2d_cache.values():
        reader.close()
    _indexed_2d_cache.clear()


def open_2d_pair(track_path: str, c1: str, c2: str) -> tuple[bool, int, bytes | mmap.mmap, int, Any] | None:
    """Open a 2D track chromosome pair, supporting both per-pair files and
    indexed format.

    This is the unified entry point for opening 2D track data.  It
    checks for indexed format first (``track.idx`` + ``track.dat``),
    falling back to per-pair files.

    Parameters
    ----------
    track_path : str
        Path to the track directory (e.g., from ``pm_track_path``).
    c1, c2 : str
        Chromosome names (e.g., ``"1"``, ``"X"``).

    Returns
    -------
    tuple of (is_points, num_objs, data, root_chunk_fpos, close_fn) or None
        - *is_points*: whether the track stores points.
        - *num_objs*: number of objects in this pair.
        - *data*: buffer (mmap or bytes) for quad-tree queries.
        - *root_chunk_fpos*: file position of the root chunk.
        - *close_fn*: callable to release resources (call in a finally
          block).  For indexed format this is a no-op; for per-pair
          files it closes the mmap.
        Returns ``None`` if no data exists for this pair.
    """
    # Try indexed format first
    reader = _get_indexed_reader(track_path)
    if reader is not None:
        # Need chrom name -> ID mapping
        from .intervals import _chrom_id_lookup, _chrom_id_map

        cmap = _chrom_id_map()
        c1_id = _chrom_id_lookup(cmap, str(c1))
        c2_id = _chrom_id_lookup(cmap, str(c2))
        if c1_id is not None and c2_id is not None:
            result = reader.get_pair_data(c1_id, c2_id)
            if result is not None:
                is_points, num_objs, data, root_chunk_fpos = result
                return (is_points, num_objs, data, root_chunk_fpos, _noop_close)

    # Fall back to per-pair files
    from .extract import _find_2d_track_file

    filepath = _find_2d_track_file(track_path, c1, c2)
    if filepath is None:
        return None

    file_is_points, num_objs, mmap_data = _read_file_header(filepath)
    if num_objs == 0:
        root_chunk_fpos = 0
    else:
        payload_offset = _payload_offset(mmap_data)
        root_chunk_fpos = struct.unpack_from("<q", mmap_data, payload_offset + 8)[0]
    return (file_is_points, num_objs, mmap_data, root_chunk_fpos, mmap_data.close)


def _noop_close() -> None:
    """No-op close function for indexed pair data (bytes don't need closing)."""
