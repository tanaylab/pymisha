"""Phase NN regression tests for the 2D nearest-neighbor iterator.

Geometry, window, maxneighbors, NA-if-notfound, and multi-chrom-pair
semantics of pymisha's _neighbors_2d (via the new C++ NN iterator)
against hand-computed expected values on the small testdb (chroms
1 / 2 / X). Mirrors R's gfind_neighbors 2D branch and the per-axis
unsigned-gap Manhattan-distance model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _rects(rows: list[tuple]) -> pd.DataFrame:
    """rows are (chrom1, start1, end1, chrom2, start2, end2) tuples."""
    return pd.DataFrame(
        rows,
        columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2"],
    )


@pytest.fixture
def i1_one() -> pd.DataFrame:
    return _rects([("1", 100, 200, "1", 100, 200)])


def test_single_nearest_neighbor_unbounded(i1_one):
    """One query, three targets at known Manhattan distances; the
    closest target is returned with its exact (dist1, dist2)."""
    i2 = _rects([
        ("1", 1000, 1100, "1", 100, 200),   # axis1 gap = 800, axis2 gap = 0
        ("1", 100, 200, "1", 5000, 5100),   # axis1 gap = 0, axis2 gap = 4800
        ("1", 50, 80, "1", 50, 80),         # axis1 gap = 20, axis2 gap = 20
    ])
    out = pm.gintervals_neighbors(i1_one, i2, maxneighbors=1)
    assert out is not None and len(out) == 1
    assert int(out["dist1"].iloc[0]) == 20
    assert int(out["dist2"].iloc[0]) == 20


def test_maxneighbors_cap_returns_two_closest(i1_one):
    """maxneighbors=2 returns the two closest in Manhattan order."""
    i2 = _rects([
        ("1", 10000, 11000, "1", 10000, 11000),  # very far
        ("1", 300, 400, "1", 100, 200),          # d1=100, d2=0, M=100
        ("1", 50, 80, "1", 50, 80),              # d1=20, d2=20, M=40
        ("1", 100, 200, "1", 500, 600),          # d1=0, d2=300, M=300
    ])
    out = pm.gintervals_neighbors(i1_one, i2, maxneighbors=2)
    assert out is not None and len(out) == 2
    sums = (out["dist1"] + out["dist2"]).astype(int).tolist()
    assert sums == [40, 100]


def test_bounded_window_excludes_outside(i1_one):
    """A bounded window filters out neighbors beyond mindist/maxdist."""
    i2 = _rects([
        ("1", 100, 200, "1", 100, 200),    # d1=0, d2=0 -> below mindist1=10
        ("1", 100, 200, "1", 1500, 1600),  # d1=0, d2=1300 -> in window
        ("1", 100, 200, "1", 5000, 5100),  # d1=0, d2=4800 -> beyond maxdist2
    ])
    out = pm.gintervals_neighbors(
        i1_one, i2, maxneighbors=10,
        mindist1=0, maxdist1=1_000_000,
        mindist2=10, maxdist2=2000,
    )
    assert out is not None and len(out) == 1
    assert int(out["dist2"].iloc[0]) == 1300


def test_unbounded_returns_all_in_manhattan_order(i1_one):
    """No window: every target on the same chrom-pair is returned in
    Manhattan-distance order."""
    i2 = _rects([
        ("1", 10000, 10100, "1", 100, 200),     # d1=9800
        ("1", 300, 400, "1", 300, 400),         # d1=100, d2=100, M=200
        ("1", 50, 80, "1", 50, 80),             # d1=20, d2=20, M=40
    ])
    out = pm.gintervals_neighbors(i1_one, i2, maxneighbors=10)
    assert out is not None and len(out) == 3
    sums = (out["dist1"] + out["dist2"]).astype(int).tolist()
    assert sums == sorted(sums)  # already in Manhattan-ascending order


def test_multi_chrom_pair_no_cross_match():
    """Queries on different chrom-pairs never match across pairs."""
    i1 = _rects([
        ("1", 100, 200, "1", 100, 200),
        ("2", 100, 200, "2", 100, 200),
    ])
    i2 = _rects([
        ("1", 300, 400, "1", 100, 200),   # only matches q0
        ("2", 500, 600, "2", 500, 600),   # only matches q1
    ])
    out = pm.gintervals_neighbors(i1, i2, maxneighbors=10)
    assert out is not None and len(out) == 2
    # Each output row must agree with its source pair: chrom1 == chrom11.
    assert (out["chrom1"].astype(str) == out["chrom11"].astype(str)).all()


def test_na_if_notfound_emits_na_row(i1_one):
    """A query with no candidates on its chrom-pair emits an NA row
    when na_if_notfound=True (R parity)."""
    i2 = _rects([("2", 100, 200, "2", 100, 200)])  # different chrom-pair
    out = pm.gintervals_neighbors(
        i1_one, i2, maxneighbors=1, na_if_notfound=True
    )
    assert out is not None and len(out) == 1
    assert pd.isna(out["dist1"].iloc[0])
    assert pd.isna(out["dist2"].iloc[0])


def test_na_if_notfound_false_drops_no_match_query(i1_one):
    """na_if_notfound=False drops queries with zero matches."""
    i2 = _rects([("2", 100, 200, "2", 100, 200)])
    out = pm.gintervals_neighbors(
        i1_one, i2, maxneighbors=1, na_if_notfound=False
    )
    assert out is None or len(out) == 0


def test_equal_distance_ties_both_returned_with_maxn_2(i1_one):
    """Two candidates at the same Manhattan distance; with maxn=2 both
    must come back (tie-break order is implementation-defined, but
    BOTH must be present)."""
    i2 = _rects([
        ("1", 300, 400, "1", 100, 200),   # d1=100, d2=0
        ("1", 100, 200, "1", 300, 400),   # d1=0, d2=100
    ])
    out = pm.gintervals_neighbors(i1_one, i2, maxneighbors=2)
    assert out is not None and len(out) == 2
    sums = (out["dist1"] + out["dist2"]).astype(int).tolist()
    assert sums == [100, 100]


def test_touching_edges_are_distance_zero(i1_one):
    """Touching edges (axis end == query start) count as distance 0,
    not 1, matching R's touch=true convention."""
    i2 = _rects([("1", 200, 300, "1", 100, 200)])  # axis1 abutting
    out = pm.gintervals_neighbors(i1_one, i2, maxneighbors=1)
    assert out is not None and len(out) == 1
    assert int(out["dist1"].iloc[0]) == 0


def test_falsification_moving_a_target_changes_nn(i1_one):
    """Perturb the closest target's coordinates; the NN must change.
    Falsification check that the test exercises real coordinates."""
    i2_a = _rects([
        ("1", 300, 400, "1", 300, 400),   # d1=100, d2=100, M=200
        ("1", 50, 80, "1", 50, 80),       # d1=20, d2=20, M=40
    ])
    i2_b = _rects([
        ("1", 300, 400, "1", 300, 400),
        ("1", 5000, 5080, "1", 5000, 5080),  # very far now
    ])
    out_a = pm.gintervals_neighbors(i1_one, i2_a, maxneighbors=1)
    out_b = pm.gintervals_neighbors(i1_one, i2_b, maxneighbors=1)
    sum_a = int(out_a["dist1"].iloc[0] + out_a["dist2"].iloc[0])
    sum_b = int(out_b["dist1"].iloc[0] + out_b["dist2"].iloc[0])
    assert sum_a < sum_b, "Moving target away must change NN distance"
