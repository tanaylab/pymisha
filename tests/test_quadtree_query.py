"""Tests for the in-memory ``QuadTree.query`` overlap primitive.

This is the scalable building block for the 2D rectangle intersection used by
``gintervals_2d_intersect`` and by the 2D-DataFrame-iterator routing in
``gextract`` / ``giterator_intervals``.  The query mirrors R misha's
``StatQuadTree::intersect`` semantics: each stored object that genuinely
overlaps the query rectangle is returned exactly once (deduped), strict overlap
(touching edges do not count).
"""
from __future__ import annotations

import random

from pymisha._quadtree import QuadTree


def _brute_overlaps(rects: list[tuple[int, int, int, int]], q: tuple[int, int, int, int]) -> list[int]:
    qx1, qy1, qx2, qy2 = q
    out = []
    for i, (x1, y1, x2, y2) in enumerate(rects):
        if max(x1, qx1) < min(x2, qx2) and max(y1, qy1) < min(y2, qy2):
            out.append(i)
    return sorted(out)


def _build(rects: list[tuple[int, int, int, int]], bound: int = 10_000) -> QuadTree:
    qt = QuadTree(0, 0, bound, bound, is_points=False)
    for (x1, y1, x2, y2) in rects:
        qt.insert((int(x1), int(y1), int(x2), int(y2), 0.0))
    return qt


class TestQuadTreeQuery:
    def test_returns_overlapping_object_indices(self):
        rects = [(0, 0, 100, 100), (50, 50, 150, 150), (200, 200, 300, 300)]
        qt = _build(rects)
        assert qt.query(40, 40, 60, 60) == [0, 1]
        assert qt.query(250, 250, 260, 260) == [2]
        assert qt.query(1000, 1000, 1100, 1100) == []

    def test_touching_not_overlapping(self):
        qt = _build([(0, 0, 100, 100)])
        # Touching at x == 100: no strict overlap.
        assert qt.query(100, 0, 200, 100) == []
        # Touching at y == 100: no strict overlap.
        assert qt.query(0, 100, 100, 200) == []

    def test_dedup_object_spanning_many_leaves(self):
        # A rect spanning the whole space lands in many leaves; it must be
        # reported exactly once for a query that overlaps it.
        rects = [(0, 0, 9999, 9999)]
        rects += [(i * 17, i * 17, i * 17 + 5, i * 17 + 5) for i in range(1, 400)]
        qt = _build(rects)
        res = qt.query(10, 10, 9990, 9990)
        assert len(res) == len(set(res)), "object reported more than once"
        assert res == sorted(res)
        assert res == _brute_overlaps(rects, (10, 10, 9990, 9990))

    def test_random_matches_brute_force(self):
        rng = random.Random(60427)
        rects = []
        for _ in range(2000):
            x1 = rng.randint(0, 9900)
            y1 = rng.randint(0, 9900)
            x2 = x1 + rng.randint(1, 100)
            y2 = y1 + rng.randint(1, 100)
            rects.append((x1, y1, x2, y2))
        qt = _build(rects)
        for _ in range(300):
            qx1 = rng.randint(0, 9900)
            qy1 = rng.randint(0, 9900)
            qx2 = qx1 + rng.randint(1, 500)
            qy2 = qy1 + rng.randint(1, 500)
            q = (qx1, qy1, qx2, qy2)
            assert qt.query(*q) == _brute_overlaps(rects, q)
