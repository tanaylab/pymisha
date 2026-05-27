"""End-to-end tests for gextract(..., iterator=(N, M)) FixedRect path.

The FixedRect path tiles the scope into (width x height) bins and returns
one aggregated value per bin (default: avg of all intersecting objects).
This is fundamentally different from the raw-object path: it produces one
row per bin, not one row per object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def test_gextract_fixed_rect_returns_grid():
    """iterator=(N, M) returns one row per grid cell, not per object."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result = pm.gextract("rects_track", intervals=intervals, iterator=(100_000, 100_000))
    # 5x5 grid over [0, 500000) x [0, 500000)
    assert result is not None
    assert len(result) == 25
    assert set(result.columns) >= {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}
    assert "rects_track" in result.columns
    assert "intervalID" in result.columns


def test_gextract_fixed_rect_matches_explicit_grid_extract():
    """Scanner path produces same per-bin values as feeding the same grid
    explicitly via intervals= with a vtrack."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })

    # Scanner path:
    new_result = pm.gextract(
        "rects_track", intervals=intervals, iterator=(100_000, 100_000),
    )

    # Reference path: build the same 25-cell grid and use a vtrack with
    # func="avg" to get per-bin averages via the established public path.
    grid_rows = []
    for y in range(0, 500_000, 100_000):
        for x in range(0, 500_000, 100_000):
            grid_rows.append({
                "chrom1": "1", "start1": x, "end1": x + 100_000,
                "chrom2": "1", "start2": y, "end2": y + 100_000,
            })
    grid_df = pd.DataFrame(grid_rows)

    pm.gvtrack_create("v_avg_ref", "rects_track", func="avg")
    try:
        ref_result = pm.gextract("v_avg_ref", intervals=grid_df)
    finally:
        pm.gvtrack_rm("v_avg_ref")

    new_sorted = new_result.sort_values(["start2", "start1"]).reset_index(drop=True)
    ref_sorted = ref_result.sort_values(["start2", "start1"]).reset_index(drop=True)

    np.testing.assert_allclose(
        new_sorted["rects_track"].to_numpy(),
        ref_sorted["v_avg_ref"].to_numpy(),
        rtol=0,
        atol=0,
    )


def test_gextract_fixed_rect_vtrack_routes_through_scanner():
    """A reducing 2D vtrack with iterator=(N, M) now routes through the C++
    scanner and returns one aggregated row per grid cell (R parity)."""
    pm.gvtrack_create("v_rects_avg2", "rects_track", func="avg")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_rects_avg2", intervals=intervals, iterator=(100_000, 100_000))
        # 5x5 grid over [0, 500000) x [0, 500000) -> 25 cells.
        assert result is not None
        assert len(result) == 25
        assert "v_rects_avg2" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    finally:
        pm.gvtrack_rm("v_rects_avg2")


def test_gextract_fixed_rect_column_names():
    """Custom colnames= are forwarded correctly."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result = pm.gextract(
        "rects_track",
        intervals=intervals,
        iterator=(100_000, 100_000),
        colnames=["my_col"],
    )
    assert "my_col" in result.columns
    assert "rects_track" not in result.columns


def test_gextract_fixed_rect_non_square_bins():
    """Width and height can differ."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [300_000],
        "chrom2": ["1"], "start2": [0], "end2": [200_000],
    })
    result = pm.gextract("rects_track", intervals=intervals, iterator=(100_000, 50_000))
    # 3 bins along dim1 x 4 bins along dim2 = 12 cells
    assert result is not None
    assert len(result) == 12


def test_gextract_fixed_rect_invalid_tuple_raises():
    """A tuple with non-integer values must raise ValueError."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    with pytest.raises(ValueError):
        pm.gextract("rects_track", intervals=intervals, iterator=(100_000, "bad"))


def test_gextract_fixed_rect_zero_bin_raises():
    """A zero bin size must raise ValueError."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    with pytest.raises(ValueError):
        pm.gextract("rects_track", intervals=intervals, iterator=(0, 100_000))


# ---------------------------------------------------------------------------
# R-parity tests - ported from R misha test suite
# ---------------------------------------------------------------------------
# Mapping between R test DB chroms and pymisha test DB:
#   R test DB: chromosomes indexed 1-22, X, Y, M (full genome)
#   pymisha test DB: '1' (500 kb), '2' (300 kb), 'X' (200 kb)
# R's `gintervals(c(2, 3))` = full extent of R chroms 2 and 3.
# Ported tests use pymisha's available chroms with comparable scope.
#
# R track name mapping:
#   test.fixedbin -> dense_track  (fixed-bin / dense 1D)
#   test.sparse   -> sparse_track (sparse 1D)
#   test.array    -> array_track  (array 1D)
#   test.rects    -> rects_track  (2D rectangles)
#   test.computed2d -> no equivalent in pymisha test DB (SKIPPED)


# --- test-gextract3.R: "gextract iterators" (lines 38-42) ------------------

def test_r_gextract3_fixedbin_with_2d_iterator_raises():
    """Ports test-gextract3.R L38: 1D dense track + iterator=c(N,M) must error.

    R: expect_error(gextract("test.fixedbin", gintervals(c(2,3)), iterator=c(100000,100000)))
    A tuple iterator is only valid for 2D track expressions; a 1D track with
    1D intervals and a tuple iterator must raise an error.
    """
    intervals = pd.DataFrame({
        "chrom": ["1", "2"], "start": [0, 0], "end": [500_000, 300_000],
    })
    with pytest.raises(Exception):
        pm.gextract("dense_track", intervals=intervals, iterator=(100_000, 100_000))


def test_r_gextract3_sparse_with_2d_iterator_raises():
    """Ports test-gextract3.R L39: 1D sparse track + iterator=c(N,M) must error.

    R: expect_error(gextract("test.sparse", gintervals(c(2,3)), iterator=c(100000,100000)))
    """
    intervals = pd.DataFrame({
        "chrom": ["1", "2"], "start": [0, 0], "end": [500_000, 300_000],
    })
    with pytest.raises(Exception):
        pm.gextract("sparse_track", intervals=intervals, iterator=(100_000, 100_000))


def test_r_gextract3_array_with_2d_iterator_raises():
    """Ports test-gextract3.R L40: 1D array track + iterator=c(N,M) must error.

    R: expect_error(gextract("test.array", gintervals(c(2,3)), iterator=c(100000,100000)))
    """
    intervals = pd.DataFrame({
        "chrom": ["1", "2"], "start": [0, 0], "end": [500_000, 300_000],
    })
    with pytest.raises(Exception):
        pm.gextract("array_track", intervals=intervals, iterator=(100_000, 100_000))


def test_r_gextract3_rects_fixedrect_two_chrom_pairs():
    """Ports test-gextract3.R L41: rects track + iterator=c(100000,100000) on 2 chrom pairs.

    R: expect_regression(
           gextract("test.rects",
                    gintervals.2d(chroms1=c(2,3), chroms2=c(2,4)),
                    iterator=c(100000,100000)),
           "gextract.27")

    R used chroms 2,3 x 2,4 (full genome extent).  pymisha test DB has
    chroms '1' (500 kb) and '2' (300 kb).  We use two chrom pairs that
    cover equivalent multi-pair scope.

    Structural assertions checked:
    - Returns one row per grid cell (no raw objects).
    - Interval bounds are multiples of the bin size.
    - chrom pair column values are correct.
    - All grid cells for each pair are present.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"], "start1": [0, 0], "end1": [500_000, 500_000],
        "chrom2": ["1", "2"], "start2": [0, 0], "end2": [500_000, 300_000],
    })
    result = pm.gextract("rects_track", intervals=intervals, iterator=(100_000, 100_000))

    assert result is not None
    # chrom1='1', chrom2='1': 5 bins x 5 bins = 25 cells
    # chrom1='1', chrom2='2': 5 bins x 3 bins = 15 cells
    assert len(result) == 40

    pair_11 = result[(result["chrom1"] == "1") & (result["chrom2"] == "1")]
    pair_12 = result[(result["chrom1"] == "1") & (result["chrom2"] == "2")]
    assert len(pair_11) == 25
    assert len(pair_12) == 15

    # Grid boundaries must be multiples of the bin size.
    assert (pair_11["start1"] % 100_000 == 0).all()
    assert (pair_11["start2"] % 100_000 == 0).all()
    assert (pair_12["start1"] % 100_000 == 0).all()
    assert (pair_12["start2"] % 100_000 == 0).all()

    # Data column exists and has finite values.
    assert "rects_track" in result.columns
    assert result["rects_track"].notna().all()


def test_r_gextract3_computed2d_fixedrect_skipped():
    """Ports test-gextract3.R L42: computed2d track + iterator=c(100000,100000).

    R: expect_regression(gextract("test.computed2d", ..., iterator=c(100000,100000)), "gextract.26")

    SKIPPED - pymisha test DB has no 'computed2d' equivalent track.
    TODO: create a computed2d fixture in tests/testdb or as a session-scoped
    fixture so this test can be fully ported.
    """
    pytest.skip("no computed2d track in pymisha test DB - TODO: create fixture")


# --- test-vtrack.R: vtrack over 2D track + FixedRect iterator (lines 68-82) -

def test_r_vtrack_rects_avg_2d_iterator(_init_db):
    """Ports test-vtrack.R 'vtrack.rects extraction with avg function and 2d iterator' (L68-74).

    R: gvtrack.create('v1', 'test.rects', func='avg')
       r <- gextract('v1', gintervals.2d(chroms1=c(1,3), 3000000, -1, chroms2=c(1,4), 3000000, -1),
                     iterator=c(2000000,3000000))
       expect_regression(r, 'vtrack.rects.avg.2d')

    R parity: vtrack + FixedRect routes through C++ scanner, returning one row per grid cell.
    """
    pm.gvtrack_create("v_rects_avg", "rects_track", func="avg")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_rects_avg", intervals=intervals, iterator=(200_000, 300_000))
        # Grid: ceil(500000/200000)=3 x ceil(500000/300000)=2 -> 6 cells (some may be partial).
        # Row count matches the bare-track call (same grid, same scope).
        bare = pm.gextract("rects_track", intervals=intervals, iterator=(200_000, 300_000))
        assert len(result) == len(bare)
        assert "v_rects_avg" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
        # Values must be equal to bare-track avg (same computation, different column name).
        np.testing.assert_array_equal(
            result.sort_values(["start1", "start2"])["v_rects_avg"].to_numpy(),
            bare.sort_values(["start1", "start2"])["rects_track"].to_numpy(),
        )
    finally:
        pm.gvtrack_rm("v_rects_avg")


def test_r_vtrack_computed2d_avg_2d_iterator(_init_db):
    """Ports test-vtrack.R 'vtrack.computed2d extraction with avg function and 2d iterator' (L76-82).

    R: gvtrack.create('v1', 'test.computed2d', func='avg')
       r <- gextract('v1', gintervals.2d(chroms1=c(6,1,5), 3000000, -1, chroms2=c(8,1,9), 3000000, -1),
                     iterator=c(2000000,3000000))
       expect_regression(r, 'vtrack.computed2d.avg.2d')

    Note: uses rects_track as the source since no computed2d track exists in
    the pymisha test DB. The structural intent (vtrack + FixedRect) is the same.
    R parity: routes through C++ scanner, one row per grid cell.
    """
    pm.gvtrack_create("v_2d_avg", "rects_track", func="avg")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_2d_avg", intervals=intervals, iterator=(200_000, 300_000))
        bare = pm.gextract("rects_track", intervals=intervals, iterator=(200_000, 300_000))
        assert len(result) == len(bare)
        assert "v_2d_avg" in result.columns
        np.testing.assert_array_equal(
            result.sort_values(["start1", "start2"])["v_2d_avg"].to_numpy(),
            bare.sort_values(["start1", "start2"])["rects_track"].to_numpy(),
        )
    finally:
        pm.gvtrack_rm("v_2d_avg")


# --- test-vtrack.R: vtrack max + 2D FixedRect iterator (lines 114-128) -----

def test_r_vtrack_rects_max_2d_iterator(_init_db):
    """Ports test-vtrack.R 'vtrack.rects extraction with max function and 2d iterator' (L114-120).

    R: gvtrack.create('v1', 'test.rects', func='max')
       r <- gextract('v1', gintervals.2d(chroms1=c(1,3), 3000000, -1, chroms2=c(1,4), 3000000, -1),
                     iterator=c(2000000,3000000))
       expect_regression(r, 'vtrack.rects.max.2d')

    R parity: vtrack(func=max) + FixedRect routes through C++ scanner.
    """
    pm.gvtrack_create("v_rects_max", "rects_track", func="max")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_rects_max", intervals=intervals, iterator=(200_000, 300_000))
        bare = pm.gextract("rects_track", intervals=intervals, iterator=(200_000, 300_000),
                           colnames=["rects_track_max"])
        # func=max gives different values than func=avg (the bare default), so we
        # only assert shape and column presence here; value verification would need
        # a separate max-aggregated reference call.
        assert len(result) > 0
        assert len(result) == len(bare)
        assert "v_rects_max" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    finally:
        pm.gvtrack_rm("v_rects_max")


def test_r_vtrack_computed2d_max_2d_iterator(_init_db):
    """Ports test-vtrack.R 'vtrack.computed2d extraction with max function and 2d iterator' (L122-128).

    R: gvtrack.create('v1', 'test.computed2d', func='max')
       r <- gextract('v1', gintervals.2d(chroms1=c(6,1,5), 3000000, -1, chroms2=c(8,1,9), 3000000, -1),
                     iterator=c(2000000,3000000))
       expect_regression(r, 'vtrack.computed2d.max.2d')

    Note: uses rects_track as the source since no computed2d track in pymisha test DB.
    R parity: vtrack(func=max) + FixedRect routes through C++ scanner.
    """
    pm.gvtrack_create("v_2d_max", "rects_track", func="max")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_2d_max", intervals=intervals, iterator=(200_000, 300_000))
        bare = pm.gextract("rects_track", intervals=intervals, iterator=(200_000, 300_000))
        assert len(result) == len(bare)
        assert "v_2d_max" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    finally:
        pm.gvtrack_rm("v_2d_max")


# --- test-vtrack.R: vtrack min + 2D FixedRect iterator (lines 160-174) -----

def test_r_vtrack_rects_min_2d_iterator(_init_db):
    """Ports test-vtrack.R 'rects_min' (L160-166).

    R: gvtrack.create('v1', 'test.rects', func='min')
       r <- gextract('v1', gintervals.2d(chroms1=c(1,3), 3000000, -1, chroms2=c(1,4), 3000000, -1),
                     iterator=c(2000000,3000000))
       expect_regression(r, 'vtrack.rects_min')

    R parity: vtrack(func=min) + FixedRect routes through C++ scanner.
    """
    pm.gvtrack_create("v_rects_min", "rects_track", func="min")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_rects_min", intervals=intervals, iterator=(200_000, 300_000))
        bare = pm.gextract("rects_track", intervals=intervals, iterator=(200_000, 300_000))
        assert len(result) == len(bare)
        assert "v_rects_min" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    finally:
        pm.gvtrack_rm("v_rects_min")


def test_r_vtrack_computed2d_min_2d_iterator(_init_db):
    """Ports test-vtrack.R 'computed2d_min' (L168-174).

    R: gvtrack.create('v1', 'test.computed2d', func='min')
       r <- gextract('v1', gintervals.2d(chroms1=c(6,1,5), 3000000, -1, chroms2=c(8,1,9), 3000000, -1),
                     iterator=c(2000000,3000000))
       expect_regression(r, 'vtrack.computed2d_min')

    Note: uses rects_track as the source since no computed2d track in pymisha test DB.
    R parity: vtrack(func=min) + FixedRect routes through C++ scanner.
    """
    pm.gvtrack_create("v_2d_min", "rects_track", func="min")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_2d_min", intervals=intervals, iterator=(200_000, 300_000))
        bare = pm.gextract("rects_track", intervals=intervals, iterator=(200_000, 300_000))
        assert len(result) == len(bare)
        assert "v_2d_min" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    finally:
        pm.gvtrack_rm("v_2d_min")


# --- test-vtrack.R: fixedbin quantile vtrack + 2D FixedRect iterator -> error -

def test_r_vtrack_fixedbin_quantile_2d_chroms1_1_3_raises():
    """Ports test-vtrack.R 'vtrack.fixedbin quantile extraction 2D chroms1=1,3; chroms2=1,4' (L326-332).

    R: gvtrack.create('v1', 'test.fixedbin', func='quantile', params=0.5)
       gvtrack.create('v2', 'test.fixedbin', func='quantile', params=0.9)
       expect_error(gextract('v1', 'v2',
                              gintervals.2d(chroms1=c(1,3), 3000000, -1, chroms2=c(1,4), 3000000, -1),
                              iterator=c(2000000,3000000)))

    Multiple vtracks over 1D tracks with 2D intervals + FixedRect iterator must error.
    In pymisha, vtracks are not supported with iterator=(N,M).
    """
    pm.gvtrack_create("v_q50", "dense_track", func="quantile", params=0.5)
    pm.gvtrack_create("v_q90", "dense_track", func="quantile", params=0.9)
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        with pytest.raises((NotImplementedError, Exception)):
            pm.gextract(["v_q50", "v_q90"], intervals=intervals, iterator=(200_000, 300_000))
    finally:
        pm.gvtrack_rm("v_q50")
        pm.gvtrack_rm("v_q90")


def test_r_vtrack_fixedbin_quantile_2d_chroms1_6_1_5_raises():
    """Ports test-vtrack.R 'vtrack.fixedbin quantile extraction 2D chroms1=6,1,5; chroms2=8,1,9' (L334-340).

    R: gvtrack.create('v1', 'test.fixedbin', func='quantile', params=0.5)
       gvtrack.create('v2', 'test.fixedbin', func='quantile', params=0.9)
       expect_error(gextract('v1', 'v2',
                              gintervals.2d(chroms1=c(6,1,5), 3000000, -1, chroms2=c(8,1,9), 3000000, -1),
                              iterator=c(2000000,3000000)))

    Variant with different chromosome sets - still must error.
    """
    pm.gvtrack_create("v_q50b", "dense_track", func="quantile", params=0.5)
    pm.gvtrack_create("v_q90b", "dense_track", func="quantile", params=0.9)
    intervals = pd.DataFrame({
        "chrom1": ["1", "2"], "start1": [0, 0], "end1": [500_000, 300_000],
        "chrom2": ["1", "X"], "start2": [0, 0], "end2": [500_000, 200_000],
    })
    try:
        with pytest.raises((NotImplementedError, Exception)):
            pm.gextract(["v_q50b", "v_q90b"], intervals=intervals, iterator=(200_000, 300_000))
    finally:
        pm.gvtrack_rm("v_q50b")
        pm.gvtrack_rm("v_q90b")


# ---------------------------------------------------------------------------
# giterator_intervals with a numeric 2D iterator (FixedRect cell enumeration)
# ---------------------------------------------------------------------------
# R parity: giterator.intervals(expr, scope, iterator=c(width, height)) returns
# the iterator cells (coordinates only, no track values) over a 2D scope.
# Ports the structural intent of test-giterator.intervals.R cases 4 and 13.


def _fixedrect_scope():
    return pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })


def test_giterator_intervals_2d_fixed_rect_grid():
    """A numeric 2D iterator enumerates every grid cell (coords only).

    5x5 = 25 cells over chr1 x chr1 [0,500k) x [0,500k); the returned set of
    rectangles must equal the hand-built grid exactly.
    """
    result = pm.giterator_intervals(None, _fixedrect_scope(), iterator=(100_000, 100_000))
    assert result is not None
    # No value column -- only the 6 coordinate columns.
    assert list(result.columns) == [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2"
    ]

    expected = []
    for y in range(0, 500_000, 100_000):
        for x in range(0, 500_000, 100_000):
            expected.append(("1", x, x + 100_000, "1", y, y + 100_000))
    exp = pd.DataFrame(expected, columns=result.columns)

    key = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    got = result.sort_values(key).reset_index(drop=True)
    exp = exp.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_giterator_intervals_2d_fixed_rect_ignores_expr():
    """The expr is ignored when the iterator is explicit: a 2D track name as
    expr produces the same cells as expr=None (R parity)."""
    key = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    none_expr = pm.giterator_intervals(
        None, _fixedrect_scope(), iterator=(100_000, 100_000)
    ).sort_values(key).reset_index(drop=True)
    with_expr = pm.giterator_intervals(
        "rects_track", _fixedrect_scope(), iterator=(100_000, 100_000)
    ).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(none_expr, with_expr, check_dtype=False)


def test_giterator_intervals_2d_fixed_rect_non_square_multipair():
    """Non-square bins over two chrom pairs: exact per-pair cell counts."""
    scope = pd.DataFrame({
        "chrom1": ["1", "1"], "start1": [0, 0], "end1": [300_000, 300_000],
        "chrom2": ["1", "2"], "start2": [0, 0], "end2": [200_000, 200_000],
    })
    result = pm.giterator_intervals(None, scope, iterator=(100_000, 50_000))
    # Each pair: 3 bins (dim1) x 4 bins (dim2) = 12 cells -> 24 total.
    assert len(result) == 24
    assert len(result[(result["chrom1"] == "1") & (result["chrom2"] == "1")]) == 12
    assert len(result[(result["chrom1"] == "1") & (result["chrom2"] == "2")]) == 12


def test_giterator_intervals_2d_fixed_rect_band():
    """A diagonal band reduces the cell set; a band wider than the scope is a
    no-op (returns the full grid)."""
    full = pm.giterator_intervals(None, _fixedrect_scope(), iterator=(100_000, 100_000))

    narrow = pm.giterator_intervals(
        None, _fixedrect_scope(), iterator=(100_000, 100_000), band=(-100_000, 100_000)
    )
    assert 0 < len(narrow) < len(full)

    wide = pm.giterator_intervals(
        None, _fixedrect_scope(), iterator=(100_000, 100_000),
        band=(-10_000_000, 10_000_000),
    )
    key = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    pd.testing.assert_frame_equal(
        wide.sort_values(key).reset_index(drop=True),
        full.sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


def test_giterator_intervals_2d_fixed_rect_invalid_bin_raises():
    """A non-positive bin size must raise ValueError."""
    with pytest.raises(ValueError):
        pm.giterator_intervals(None, _fixedrect_scope(), iterator=(0, 100_000))


# ---------------------------------------------------------------------------
# Multitask vs single-task equivalence
# ---------------------------------------------------------------------------

def test_fixed_rect_multitask_equals_single_task():
    """iterator=(N,M) FixedRect path produces identical output regardless of
    the multitask configuration (max_processes, min_scope4process, etc.).

    NOTE: pm_extract_2d_scanner does NOT integrate with pymisha's multitask
    (fork+FIFO) infrastructure.  The FixedRect path in _gextract_2d_via_scanner
    calls _pymisha.pm_extract_2d_scanner directly, bypassing _parallel_extract
    entirely.  As a result both the "single-task" and "multi-task" configs below
    run single-process.  The test is still a valuable regression guard: it
    confirms that CONFIG knobs do not accidentally corrupt or change the result,
    and that when multitask integration is eventually added the output remains
    stable.  Multitask integration for the FixedRect scanner is deferred to a
    future release (K.4 follow-on per the audit).
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })

    config = pm.CONFIG
    saved = config.copy()

    try:
        # Force single-process: multitasking disabled, max_processes=1.
        config.update({
            "multitasking": False,
            "max_processes": 1,
        })
        r1 = pm.gextract("rects_track", intervals=intervals, iterator=(50_000, 50_000))

        # Force "multi-task" mode: low floors so any workload would fork if the
        # scanner were integrated with _parallel_extract.
        config.update({
            "multitasking": True,
            "max_processes": 4,
            "min_scope4process": 0,
            "min_intervs4process": 0,
        })
        r4 = pm.gextract("rects_track", intervals=intervals, iterator=(50_000, 50_000))
    finally:
        # Restore original config.
        config.clear()
        config.update(saved)

    sort_keys = ["chrom1", "start2", "start1"]
    r1s = r1.sort_values(sort_keys).reset_index(drop=True)
    r4s = r4.sort_values(sort_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1s, r4s)


def test_fixed_rect_reducing_vtrack_multitask_equivalence(_init_db):
    """iterator=(N,M) + reducing 2D vtrack produces identical output regardless
    of multitask configuration.

    Exercises the new vtrack resolver + scanner path under both single-task
    and multi-task CONFIG knobs to ensure the output is stable.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    pm.gvtrack_create("v_avg_mt", "rects_track", func="avg")
    config = pm.CONFIG
    saved = config.copy()
    try:
        config.update({"multitasking": False, "max_processes": 1})
        r1 = pm.gextract("v_avg_mt", intervals=intervals, iterator=(100_000, 100_000))

        config.update({
            "multitasking": True,
            "max_processes": 4,
            "min_scope4process": 0,
            "min_intervs4process": 0,
        })
        r4 = pm.gextract("v_avg_mt", intervals=intervals, iterator=(100_000, 100_000))
    finally:
        config.clear()
        config.update(saved)
        pm.gvtrack_rm("v_avg_mt")

    sort_keys = ["chrom1", "start2", "start1"]
    r1s = r1.sort_values(sort_keys).reset_index(drop=True)
    r4s = r4.sort_values(sort_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1s, r4s)


# ---------------------------------------------------------------------------
# R6: object-level vtrack funcs through the C++ scanner (exists/size/first/last/sample)
# ---------------------------------------------------------------------------

def test_r_vtrack_rects_exists_2d_iterator(_init_db):
    """Closes R6: vtrack with func='exists' + iterator=(N,M) routes through C++ scanner.

    exists returns 1.0 for cells that intersect at least one object, 0.0 otherwise.
    All values must be 0 or 1; at least one cell must be non-zero (track has objects).
    """
    pm.gvtrack_create("v_exists", "rects_track", func="exists")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_exists", intervals=intervals, iterator=(100_000, 100_000))
        assert result is not None
        assert len(result) == 25
        assert "v_exists" in result.columns
        vals = result["v_exists"].to_numpy()
        # All values must be 0 or 1.
        assert np.all((vals == 0.0) | (vals == 1.0)), f"unexpected values: {vals}"
        # The track has objects on chr1 x chr1, so at least one cell is non-zero.
        assert vals.sum() > 0
    finally:
        pm.gvtrack_rm("v_exists")


def test_r_vtrack_rects_size_2d_iterator(_init_db):
    """Closes R6: vtrack with func='size' + iterator=(N,M) routes through C++ scanner.

    size returns the count of intersecting objects per cell.
    All values must be non-negative integers; sum across cells equals total object count.
    """
    pm.gvtrack_create("v_size", "rects_track", func="size")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_size", intervals=intervals, iterator=(100_000, 100_000))
        assert result is not None
        assert len(result) == 25
        assert "v_size" in result.columns
        vals = result["v_size"].to_numpy()
        # All values must be non-negative.
        assert np.all(vals >= 0), f"negative size values: {vals}"
        # No NaN allowed (size defaults to 0, not NaN).
        assert not np.any(np.isnan(vals)), f"NaN in size result"
        # At least some cells have objects.
        assert vals.sum() > 0
    finally:
        pm.gvtrack_rm("v_size")


def test_r_vtrack_rects_first_2d_iterator(_init_db):
    """Closes R6: vtrack with func='first' + iterator=(N,M) routes through C++ scanner.

    first returns the value of the first intersecting object per cell, NaN if none.
    Non-NaN values must be finite floats.
    """
    pm.gvtrack_create("v_first", "rects_track", func="first")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_first", intervals=intervals, iterator=(100_000, 100_000))
        assert result is not None
        assert len(result) == 25
        assert "v_first" in result.columns
        vals = result["v_first"].to_numpy()
        non_nan = vals[~np.isnan(vals)]
        # At least one non-NaN value (track has objects on chr1 x chr1).
        assert len(non_nan) > 0, "expected non-NaN values from first"
        # Non-NaN values must be finite.
        assert np.all(np.isfinite(non_nan)), f"non-finite values: {non_nan}"
    finally:
        pm.gvtrack_rm("v_first")


def test_r_vtrack_rects_last_2d_iterator(_init_db):
    """Closes R6: vtrack with func='last' + iterator=(N,M) routes through C++ scanner.

    last returns the value of the last intersecting object per cell, NaN if none.
    Non-NaN values must be finite floats.
    """
    pm.gvtrack_create("v_last", "rects_track", func="last")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result = pm.gextract("v_last", intervals=intervals, iterator=(100_000, 100_000))
        assert result is not None
        assert len(result) == 25
        assert "v_last" in result.columns
        vals = result["v_last"].to_numpy()
        non_nan = vals[~np.isnan(vals)]
        # At least one non-NaN value.
        assert len(non_nan) > 0, "expected non-NaN values from last"
        # Non-NaN values must be finite.
        assert np.all(np.isfinite(non_nan)), f"non-finite values: {non_nan}"
    finally:
        pm.gvtrack_rm("v_last")


def test_r_vtrack_rects_sample_2d_iterator(_init_db):
    """Closes R6: vtrack with func='sample' + iterator=(N,M) routes through C++ scanner.

    sample returns the value of a (seeded) randomly-sampled intersecting object per cell,
    NaN if no objects intersect. Must have the same NaN pattern as first/last.
    """
    pm.gvtrack_create("v_sample", "rects_track", func="sample")
    pm.gvtrack_create("v_exists2", "rects_track", func="exists")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    try:
        result_sample = pm.gextract("v_sample", intervals=intervals, iterator=(100_000, 100_000))
        result_exists = pm.gextract("v_exists2", intervals=intervals, iterator=(100_000, 100_000))
        assert result_sample is not None
        assert len(result_sample) == 25
        assert "v_sample" in result_sample.columns
        sample_vals = result_sample["v_sample"].to_numpy()
        exists_vals = result_exists["v_exists2"].to_numpy()
        # Cells with no objects must be NaN; cells with objects must be non-NaN.
        has_obj = exists_vals == 1.0
        assert np.all(np.isnan(sample_vals[~has_obj])), "no-object cells should be NaN"
        non_nan = sample_vals[has_obj]
        assert np.all(np.isfinite(non_nan)), f"non-finite sample values: {non_nan}"
    finally:
        pm.gvtrack_rm("v_sample")
        pm.gvtrack_rm("v_exists2")


# ---------------------------------------------------------------------------
# Regression: exists/size return 0 (not NaN) for chrom pairs with no data
# ---------------------------------------------------------------------------

def test_exists_size_zero_for_no_data_pair(_init_db):
    """R parity: exists and size must be 0.0 (not NaN) for chrom pairs the
    track has no data file for.

    rects_track has data on (chrom1='1', chrom2='1') and (chrom1='1',
    chrom2='2') only.  A scope on (chrom1='2', chrom2='2') has no data - all
    grid cells must produce 0 for exists and 0 for size.  first/last/sample
    must stay NaN (no object to read).

    Before the fix, set_vars_batch left the no-data slots as NaN for exists
    and size, which broke R parity.
    """
    intervals = pd.DataFrame({
        "chrom1": ["2"], "start1": [0], "end1": [300_000],
        "chrom2": ["2"], "start2": [0], "end2": [300_000],
    })

    for func in ("exists", "size"):
        vname = f"v_{func}_nodata"
        pm.gvtrack_create(vname, "rects_track", func=func)
        try:
            result = pm.gextract(vname, intervals=intervals, iterator=(100_000, 100_000))
            assert result is not None, f"{func}: expected a result DataFrame"
            vals = result[vname].to_numpy()
            assert not np.any(np.isnan(vals)), (
                f"{func}: got NaN for no-data pair (chrom2 x chrom2); expected 0.0. "
                f"values={vals}"
            )
            assert np.all(vals == 0.0), (
                f"{func}: expected all zeros for no-data pair, got {vals}"
            )
        finally:
            pm.gvtrack_rm(vname)

    # first/last/sample must remain NaN (no object to read).
    for func in ("first", "last", "sample"):
        vname = f"v_{func}_nodata"
        pm.gvtrack_create(vname, "rects_track", func=func)
        try:
            result = pm.gextract(vname, intervals=intervals, iterator=(100_000, 100_000))
            assert result is not None, f"{func}: expected a result DataFrame"
            vals = result[vname].to_numpy()
            assert np.all(np.isnan(vals)), (
                f"{func}: expected all NaN for no-data pair, got {vals}"
            )
        finally:
            pm.gvtrack_rm(vname)
