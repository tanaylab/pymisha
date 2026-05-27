"""Tests for the scalable, quadtree-backed 2D rectangle intersection helper.

``_intersect_2d_rects`` must produce exactly the same set of clipped
intersection rectangles as a brute-force O(n*m) broadcast, but without
materialising the full pairwise matrix (so it scales to the 10^5-rect screens
that OOM the naive approach).
"""
from __future__ import annotations

import random

import pandas as pd

from pymisha.intervals import _intersect_2d_rects

_COLS = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]


def _broadcast_ref(a: pd.DataFrame, b: pd.DataFrame) -> list[tuple]:
    rows = []
    for r1 in a.itertuples():
        for r2 in b.itertuples():
            if r1.chrom1 == r2.chrom1 and r1.chrom2 == r2.chrom2:
                s1, e1 = max(r1.start1, r2.start1), min(r1.end1, r2.end1)
                s2, e2 = max(r1.start2, r2.start2), min(r1.end2, r2.end2)
                if s1 < e1 and s2 < e2:
                    rows.append((str(r1.chrom1), s1, e1, str(r1.chrom2), s2, e2))
    return sorted(rows)


def _broadcast_ref_with_bidx(a: pd.DataFrame, b: pd.DataFrame) -> list[tuple]:
    rows = []
    b_recs = list(b.reset_index(drop=True).itertuples())  # positional index in Index
    for r1 in a.itertuples():
        for r2 in b_recs:
            if r1.chrom1 == r2.chrom1 and r1.chrom2 == r2.chrom2:
                s1, e1 = max(r1.start1, r2.start1), min(r1.end1, r2.end1)
                s2, e2 = max(r1.start2, r2.start2), min(r1.end2, r2.end2)
                if s1 < e1 and s2 < e2:
                    rows.append((str(r1.chrom1), s1, e1, str(r1.chrom2), s2, e2, int(r2.Index)))
    return sorted(rows)


def _rows(df: pd.DataFrame | None) -> list[tuple]:
    if df is None or len(df) == 0:
        return []
    return sorted(
        (str(r.chrom1), int(r.start1), int(r.end1), str(r.chrom2), int(r.start2), int(r.end2))
        for r in df.itertuples()
    )


def _rand_set(rng: random.Random, n: int, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    recs = []
    for _ in range(n):
        c1, c2 = rng.choice(pairs)
        x1 = rng.randint(0, 990)
        y1 = rng.randint(0, 990)
        recs.append((c1, x1, x1 + rng.randint(1, 60), c2, y1, y1 + rng.randint(1, 60)))
    return pd.DataFrame(recs, columns=_COLS)


class TestIntersect2DRects:
    def test_matches_broadcast_reference_random(self):
        rng = random.Random(60427)
        pairs = [("1", "1"), ("1", "2"), ("2", "2")]
        for _ in range(25):
            a = _rand_set(rng, rng.randint(0, 80), pairs)
            b = _rand_set(rng, rng.randint(0, 80), pairs)
            assert _rows(_intersect_2d_rects(a, b)) == _broadcast_ref(a, b)

    def test_duplicate_multiplicity(self):
        # Two identical rects in A, one overlapping rect in B -> two output rows.
        a = pd.DataFrame([("1", 0, 100, "1", 0, 100), ("1", 0, 100, "1", 0, 100)], columns=_COLS)
        b = pd.DataFrame([("1", 50, 150, "1", 50, 150)], columns=_COLS)
        assert _rows(_intersect_2d_rects(a, b)) == _broadcast_ref(a, b)
        assert len(_intersect_2d_rects(a, b)) == 2

    def test_one_to_many(self):
        # One big A rect tiled by several small B rects -> one row per tile.
        a = pd.DataFrame([("1", 0, 100, "1", 0, 100)], columns=_COLS)
        b = pd.DataFrame(
            [("1", 0, 50, "1", 0, 50), ("1", 50, 100, "1", 0, 50), ("1", 0, 50, "1", 50, 100)],
            columns=_COLS,
        )
        assert _rows(_intersect_2d_rects(a, b)) == _broadcast_ref(a, b)
        assert len(_intersect_2d_rects(a, b)) == 3

    def test_disjoint_chrom_pairs(self):
        a = pd.DataFrame([("1", 0, 100, "1", 0, 100)], columns=_COLS)
        b = pd.DataFrame([("1", 0, 100, "2", 0, 100)], columns=_COLS)  # different pair
        assert _rows(_intersect_2d_rects(a, b)) == []

    def test_empty_inputs(self):
        empty = pd.DataFrame(columns=_COLS)
        a = pd.DataFrame([("1", 0, 10, "1", 0, 10)], columns=_COLS)
        assert _rows(_intersect_2d_rects(a, empty)) == []
        assert _rows(_intersect_2d_rects(empty, a)) == []
        assert _rows(_intersect_2d_rects(empty, empty)) == []


class TestIntersect2DBIndex:
    """`return_b_index` attributes each intersection to its source `b` row."""

    def test_b_index_matches_broadcast_reference(self):
        rng = random.Random(60427)
        pairs = [("1", "1"), ("1", "2"), ("2", "2")]
        for _ in range(25):
            a = _rand_set(rng, rng.randint(0, 70), pairs)
            b = _rand_set(rng, rng.randint(0, 70), pairs)
            res, bidx = _intersect_2d_rects(a, b, return_b_index=True)
            assert len(bidx) == len(res)
            got = sorted(
                (str(r.chrom1), int(r.start1), int(r.end1),
                 str(r.chrom2), int(r.start2), int(r.end2), int(bi))
                for r, bi in zip(res.itertuples(), bidx)
            )
            assert got == _broadcast_ref_with_bidx(a, b)

    def test_b_index_points_into_b(self):
        # The reported b-index row must actually contain the clipped unit.
        a = pd.DataFrame([("1", 0, 100, "1", 0, 100)], columns=_COLS)
        b = pd.DataFrame(
            [("1", 200, 300, "1", 200, 300), ("1", 50, 150, "1", 50, 150)],
            columns=_COLS,
        )
        res, bidx = _intersect_2d_rects(a, b, return_b_index=True)
        assert len(res) == 1
        assert int(bidx[0]) == 1  # only the second b row overlaps a
        r = res.iloc[0]
        src = b.iloc[int(bidx[0])]
        assert src.start1 <= r.start1 and r.end1 <= src.end1
        assert src.start2 <= r.start2 and r.end2 <= src.end2
