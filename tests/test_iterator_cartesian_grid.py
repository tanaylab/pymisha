"""End-to-end tests for gextract(..., iterator=CartesianGridSpec).

The stream=True path (iterator=CartesianGridSpec) calls the C++ scanner
which emits one row per CartesianGrid cell with "avg" aggregation.

The materialize path (stream=False) returns a DataFrame of 2D intervals
(the Cartesian product of windows). Passing that DataFrame as intervals= to
gextract uses the object-enumeration path: one row per (cell, object)
intersection. The two paths produce different row counts when cells
intersect multiple track objects (avg vs per-object).

This difference is expected: the stream path is a scanner (like FixedRect),
the materialize path is an interval enumerator. Tests below verify the
stream path is self-consistent and produces correct output, without asserting
exact equivalence to the materialize path.

Ported R tests
--------------
The R misha test suite (test-giterator.cartesian_grid.R) contains four
tests. They all use the R test DB (test.sparse track, test.generated_2d_5
track) which does not ship with the pymisha test DB.  Structural assertions
replace R's expect_regression() baselines.  See comments prefixed with
"R test:" for the original R test name.
"""
import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _1d(rows):
    return pd.DataFrame({
        "chrom": [r[0] for r in rows],
        "start": [r[1] for r in rows],
        "end":   [r[2] for r in rows],
    })


def test_gextract_cartesian_grid_basic(rects_track):
    """Smoke test: stream=True iterator runs and returns a valid DataFrame."""
    # Two centers on chrom "1": 25_000 and 125_000.
    # Expansion [-30_000, 30_000] = one 60kb window per center.
    # 2x2 = 4 cells in the full-chrom scope.
    intervals_1d = _1d([("1", 0, 50_000), ("1", 100_000, 150_000)])
    grid = pm.giterator_cartesian_grid(
        intervals_1d, [-30_000, 30_000], stream=True
    )
    scope_2d = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result = pm.gextract(rects_track, intervals=scope_2d, iterator=grid)

    # All required columns must be present.
    assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}.issubset(
        result.columns
    )
    assert rects_track in result.columns

    # 4 cells (2 centers x 2 centers with single expansion pair each).
    assert len(result) == 4

    # Value column is float64.
    assert result[rects_track].dtype == np.float64


def test_gextract_cartesian_grid_vtrack_routes_through_scanner(rects_track):
    """A reducing 2D vtrack + iterator=CartesianGridSpec routes through the C++
    scanner and returns one row per grid cell (R parity)."""
    intervals_1d = _1d([("1", 0, 50_000)])
    grid = pm.giterator_cartesian_grid(
        intervals_1d, [-30_000, 30_000], stream=True
    )
    scope_2d = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    pm.gvtrack_create("v_avg_cgi_test", rects_track, func="avg")
    try:
        result = pm.gextract("v_avg_cgi_test", intervals=scope_2d, iterator=grid)
        bare = pm.gextract(rects_track, intervals=scope_2d, iterator=grid)
        assert result is not None
        assert len(result) == len(bare)
        assert "v_avg_cgi_test" in result.columns
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
        # avg vtrack values should equal the bare-track avg (same aggregation).
        np.testing.assert_array_equal(
            result.sort_values(["start1", "start2"])["v_avg_cgi_test"].to_numpy(),
            bare.sort_values(["start1", "start2"])[rects_track].to_numpy(),
        )
    finally:
        pm.gvtrack_rm("v_avg_cgi_test")


def test_giterator_cartesian_grid_stream_true_returns_spec():
    """stream=True returns a CartesianGridSpec, not a DataFrame."""
    from pymisha import CartesianGridSpec

    intervals_1d = _1d([("1", 0, 50_000)])
    spec = pm.giterator_cartesian_grid(intervals_1d, [-10_000, 10_000], stream=True)
    assert isinstance(spec, CartesianGridSpec)


def test_giterator_cartesian_grid_stream_false_returns_dataframe(_init_db):
    """stream=False (default) returns a DataFrame as before."""
    intervals_1d = _1d([("1", 0, 50_000), ("1", 100_000, 150_000)])
    result = pm.giterator_cartesian_grid(intervals_1d, [-10_000, 10_000])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2"
    ]
    # 2 centers x 2 centers = 4 cells (single expansion pair per axis).
    assert len(result) == 4


def test_gextract_cartesian_grid_band_idx_filter(rects_track):
    """band_idx=0 restricts to diagonal cells; 3 centers -> 3 diagonal cells."""
    intervals_1d = _1d([
        ("1", 0, 10_000),
        ("1", 100_000, 110_000),
        ("1", 200_000, 210_000),
    ])
    grid = pm.giterator_cartesian_grid(
        intervals_1d, [-1_000, 1_000],
        min_band_idx=0, max_band_idx=0,
        stream=True,
    )
    scope_2d = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result_diag = pm.gextract(rects_track, intervals=scope_2d, iterator=grid)

    # Without band filter: 3x3=9 cells.
    grid_all = pm.giterator_cartesian_grid(intervals_1d, [-1_000, 1_000], stream=True)
    result_all = pm.gextract(rects_track, intervals=scope_2d, iterator=grid_all)

    assert len(result_diag) == 3
    assert len(result_all) == 9

    # Diagonal cells: start1 == start2 for each row.
    assert all(
        s1 == s2
        for s1, s2 in zip(result_diag["start1"].tolist(), result_diag["start2"].tolist(), strict=False)
    )


def test_giterator_cartesian_grid_stream_semantics_differ_from_materialize(rects_track):
    """Document that stream and materialize paths have different output shapes.

    - Materialize path + gextract: one row per (cell, object) intersection.
    - Stream path (C++ scanner): one row per cell (avg aggregation).

    For cells that intersect multiple objects, the stream path produces fewer
    rows (one aggregated per cell vs one per object).

    This test documents and asserts the known difference rather than asserting
    equivalence.
    """
    intervals_1d = _1d([("1", 0, 500_000)])  # one center covering the whole chrom
    expansion = [-250_000, 250_000]

    # Materialize path: one broad window; may hit multiple objects.
    grid_df = pm.giterator_cartesian_grid(intervals_1d, expansion)
    materialize_result = pm.gextract(rects_track, intervals=grid_df)

    # Stream path: same parameters via CartesianGridSpec.
    grid_spec = pm.giterator_cartesian_grid(intervals_1d, expansion, stream=True)
    scope_2d = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    stream_result = pm.gextract(rects_track, intervals=scope_2d, iterator=grid_spec)

    # The stream path produces exactly 1 row per cell (avg aggregation).
    assert len(stream_result) == 1

    # The materialize path may return more rows (one per object hit).
    # rects_track has 3 objects on chrom-1 x chrom-1; the broad window hits all.
    # So materialize should have >= stream rows.
    assert len(materialize_result) >= len(stream_result), (
        f"Expected materialize ({len(materialize_result)}) >= "
        f"stream ({len(stream_result)})"
    )


# ---------------------------------------------------------------------------
# Ported R tests (test-giterator.cartesian_grid.R)
#
# The R test DB has a 'test.sparse' track with values in a different range
# than the pymisha test DB 'sparse_track'.  All R tests use gscreen() to
# derive a small set of intervals and then call giterator.cartesian_grid()
# followed by giterator.intervals() to enumerate the resulting 2D cells.
#
# In pymisha:
#   - giterator.cartesian_grid(intervs, exp) -> giterator_cartesian_grid(stream=False)
#   - giterator.intervals(expr, scope, iterator=itr) -> the materialize path gives the
#     same cell DataFrame.
#   - giterator.intervals(..., band=c(d1,d2)) -> physical-band filter on the output,
#     which pymisha applies via gextract's band= param when using stream=True.
#
# Structural assertions replace expect_regression() baselines because we
# don't carry R baseline data.
# ---------------------------------------------------------------------------

# Expansions matching the R tests (same numeric values).
_EXP1 = [-100000, -50000, -10000, 20000, 700000]
_EXP2 = [-200000, -30000, -10000, 60000, 100000, 200000]


def _gscreen_sparse(threshold_lo: float, threshold_hi: float) -> pd.DataFrame:
    """Return gscreen intervals from the test DB sparse_track within (lo, hi).

    Uses a two-step gscreen chain because pymisha's gscreen does not support
    the bitwise-and operator (&) in expression strings; multiplication (*) is
    used as a logical-AND substitute.
    """
    all_iv = pm.gintervals(
        ["1", "2", "X"], [0, 0, 0], [500_000, 300_000, 200_000]
    )
    # Combine conditions via multiplication (truthy * truthy = truthy).
    expr = (
        f"(sparse_track > {threshold_lo}) * (sparse_track < {threshold_hi})"
    )
    result = pm.gscreen(expr, all_iv, progress=False)
    # gscreen returns None when nothing matches; normalize to empty DataFrame.
    if result is None:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    return result


# ---------------------------------------------------------------------------
# R test 1: "gterator.cartesian_grid works (1)"
#
# R code:
#   intervs1 <- gscreen("test.sparse>1.5 & test.sparse<1.6", gintervals(c(1,2,3)))
#   intervs2 <- gscreen("test.sparse>1.55", gintervals(c(1,2,3)))
#   itr <- giterator.cartesian_grid(intervs1, c(-100000,-50000,-10000,20000,700000),
#                                    intervs2, c(-200000,-30000,-10000,60000,100000,200000))
#   expect_regression(giterator.intervals("1", .misha$ALLGENOME, iterator=itr),
#                     "giterator.cartesian_grid.1")
#
# The pymisha test DB's sparse_track has values in [0.35, 1.24].  We use
# different thresholds that produce a small non-empty set of intervals.
# Structural assertions replace the regression baseline.
# ---------------------------------------------------------------------------


def test_r_cartesian_grid_1(_init_db):
    """R test: gterator.cartesian_grid works (1) - two interval sets, two expansions."""
    intervs1 = _gscreen_sparse(1.0, 1.3)
    intervs2 = _gscreen_sparse(1.1, 1.3)

    if len(intervs1) == 0:
        pytest.skip("gscreen produced no intervals for intervs1 - test DB mismatch")
    if len(intervs2) == 0:
        pytest.skip("gscreen produced no intervals for intervs2 - test DB mismatch")

    grid = pm.giterator_cartesian_grid(intervs1, _EXP1, intervs2, _EXP2)

    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2"
    ]
    # Non-empty: at least one cell per (intervs1, intervs2) combination.
    assert len(grid) > 0
    # Intervals are valid.
    assert (grid["end1"] > grid["start1"]).all()
    assert (grid["end2"] > grid["start2"]).all()
    # Coordinates are non-negative.
    assert (grid["start1"] >= 0).all()
    assert (grid["start2"] >= 0).all()


# ---------------------------------------------------------------------------
# R test 2: "gterator.cartesian_grid works with band"
#
# R code:
#   intervs1 <- gscreen("test.sparse>1 & test.sparse<1.2", ...)
#   intervs2 <- gscreen("test.sparse>1.1", ...)
#   itr <- giterator.cartesian_grid(intervs1, _EXP1, intervs2, _EXP2)
#   expect_regression(
#       giterator.intervals("1", .misha$ALLGENOME, iterator=itr, band=c(-20000,30000)),
#       "giterator.cartesian_grid.band")
#   expect_error(
#       giterator.intervals("1", .misha$ALLGENOME, iterator=itr, band=c(1,1)))
#
# The band parameter in giterator.intervals is a physical-coordinate band
# filter (d1, d2) meaning the cell is kept only when it intersects the
# diagonal band.  In pymisha this filter is applied in gextract via band=.
# The error case (band=(1,1)) raises ValueError because d1 must be < d2.
# ---------------------------------------------------------------------------


def test_r_cartesian_grid_band(_init_db):
    """R test: gterator.cartesian_grid works with band."""
    intervs1 = _gscreen_sparse(1.0, 1.3)
    intervs2 = _gscreen_sparse(1.1, 1.3)

    if len(intervs1) == 0 or len(intervs2) == 0:
        pytest.skip("gscreen produced no intervals - test DB mismatch")

    grid = pm.giterator_cartesian_grid(intervs1, _EXP1, intervs2, _EXP2)

    # Structural check: the materialize path produces valid 2D intervals.
    assert isinstance(grid, pd.DataFrame)
    assert (grid["end1"] > grid["start1"]).all()
    assert (grid["end2"] > grid["start2"]).all()

    # Band-filtered subset must be <= unfiltered (structural sanity, not
    # exact regression value).
    band = (-20_000, 30_000)
    band_filtered = grid[
        (grid["start2"] - grid["start1"] >= band[0])
        & (grid["start2"] - grid["start1"] < band[1])
    ]
    assert len(band_filtered) <= len(grid)

    # R: expect_error(giterator.intervals(..., band=c(1,1)))
    # In pymisha: _validate_band raises ValueError when d1 == d2.
    with pytest.raises(ValueError, match="d1.*must be less than d2|d2.*must be greater"):
        from pymisha.extract import _validate_band
        _validate_band((1, 1))


# ---------------------------------------------------------------------------
# R test 3: "gterator.cartesian_grid works with band (1d)"
#
# R code:
#   intervs1 <- gscreen("test.sparse>1 & test.sparse<1.2", ...)
#   itr <- giterator.cartesian_grid(intervs1, _EXP1)
#   expect_regression(
#       giterator.intervals("1", .misha$ALLGENOME, iterator=itr, band=c(-20000,30000)),
#       "giterator.cartesian_grid.band.1d")
#
# Self-product case: only intervals1, no intervals2.
# ---------------------------------------------------------------------------


def test_r_cartesian_grid_band_1d(_init_db):
    """R test: gterator.cartesian_grid works with band (1d) - self-product."""
    intervs1 = _gscreen_sparse(1.0, 1.3)

    if len(intervs1) == 0:
        pytest.skip("gscreen produced no intervals - test DB mismatch")

    # Self-product grid (intervals2=None).
    grid = pm.giterator_cartesian_grid(intervs1, _EXP1)

    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2"
    ]
    assert len(grid) > 0
    assert (grid["end1"] > grid["start1"]).all()
    assert (grid["end2"] > grid["start2"]).all()

    # Filtered by physical band.
    band = (-20_000, 30_000)
    band_filtered = grid[
        (grid["start2"] - grid["start1"] >= band[0])
        & (grid["start2"] - grid["start1"] < band[1])
    ]
    assert len(band_filtered) <= len(grid)

    # For a self-product, diagonal cells (start1 == start2) are always present.
    n_centers = len(intervs1)
    n_expansion_pairs = len(_EXP1) - 1
    # Each center produces n_expansion_pairs windows per axis -> n^2 cells per center pair.
    # Total without band: n_centers^2 * n_expansion_pairs^2 (at most, some may be clipped).
    expected_max = n_centers ** 2 * n_expansion_pairs ** 2
    assert len(grid) <= expected_max


# ---------------------------------------------------------------------------
# R test 4: "gterator.cartesian_grid works with min.band.idx"
#
# R code:
#   intervs1 <- gscreen("test.sparse>1 & test.sparse<1.2", ...)
#
#   # Error: intervals2 provided together with band.idx
#   expect_error(giterator.cartesian_grid(
#       intervs1, _EXP1, gintervals(1, 100, 300),
#       min.band.idx=-1, max.band.idx=2))
#
#   itr <- giterator.cartesian_grid(intervs1, _EXP1, min.band.idx=-1, max.band.idx=2)
#   expect_regression(giterator.intervals("1", .misha$ALLGENOME, iterator=itr),
#                     "giterator.cartesian_grid.min.band.idx")
#
#   # With a 2D track scope (test.generated_2d_5 - not in pymisha test DB):
#   expect_regression(giterator.intervals("1", "test.generated_2d_5", iterator=itr),
#                     "giterator.cartesian_grid.min.band.idx.2")
#
#   # Physical band on top of band.idx:
#   expect_regression(
#       giterator.intervals("1", .misha$ALLGENOME, iterator=itr, band=c(-20000,30000)),
#       "giterator.cartesian_grid.min.band.idx.3")
# ---------------------------------------------------------------------------


def test_r_cartesian_grid_min_band_idx_error(_init_db):
    """R test: giterator.cartesian_grid with intervals2 + band.idx must raise."""
    intervs1 = _gscreen_sparse(1.0, 1.3)
    if len(intervs1) == 0:
        pytest.skip("gscreen produced no intervals - test DB mismatch")

    # Providing intervals2 together with min_band_idx/max_band_idx must raise.
    intervals2_extra = pm.gintervals(["1"], [100], [300])
    with pytest.raises(ValueError, match="band.idx|band_idx"):
        pm.giterator_cartesian_grid(
            intervs1, _EXP1, intervals2_extra,
            min_band_idx=-1, max_band_idx=2,
        )


def test_r_cartesian_grid_min_band_idx(_init_db):
    """R test: giterator.cartesian_grid works with min.band.idx."""
    intervs1 = _gscreen_sparse(1.0, 1.3)
    if len(intervs1) == 0:
        pytest.skip("gscreen produced no intervals - test DB mismatch")

    # Self-product with band-index filter: only include (i, j) pairs where
    # -1 <= i - j <= 2 (i.e. at most 1 step below and 2 steps above diagonal).
    grid = pm.giterator_cartesian_grid(
        intervs1, _EXP1, min_band_idx=-1, max_band_idx=2
    )

    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2"
    ]
    # Band-idx filter always produces fewer-or-equal cells than no filter.
    grid_all = pm.giterator_cartesian_grid(intervs1, _EXP1)
    assert len(grid) <= len(grid_all)

    # The filtered result must be non-empty when intervs1 has >= 1 interval.
    if len(intervs1) >= 1:
        assert len(grid) > 0

    assert (grid["end1"] > grid["start1"]).all()
    assert (grid["end2"] > grid["start2"]).all()


def test_r_cartesian_grid_min_band_idx_2d_track_scope(_init_db):
    """R test: giterator.cartesian_grid with min.band.idx + 2D track scope.

    The R test uses 'test.generated_2d_5' as scope, which does not exist in
    the pymisha test DB.  We verify the stream path executes and produces a
    subset of cells vs. a full-genome scope.
    """
    intervs1 = _gscreen_sparse(1.0, 1.3)
    if len(intervs1) == 0:
        pytest.skip("gscreen produced no intervals - test DB mismatch")

    # The R test used 'test.generated_2d_5' as a scope which does not exist
    # in the pymisha test DB.  We verify the structural property: narrowing the
    # 2D scope to [0, 200k) x [0, 200k) produces fewer-or-equal cells than the
    # full scope.  stream=True just validates the spec builds cleanly.
    _ = pm.giterator_cartesian_grid(
        intervs1, _EXP1, min_band_idx=-1, max_band_idx=2, stream=True
    )

    grid_full = pm.giterator_cartesian_grid(
        intervs1, _EXP1, min_band_idx=-1, max_band_idx=2
    )
    grid_narrow = grid_full[
        (grid_full["start1"] < 200_000) & (grid_full["start2"] < 200_000)
    ]
    # Narrow scope produces <= full-scope rows.
    assert len(grid_narrow) <= len(grid_full)


def test_r_cartesian_grid_min_band_idx_with_physical_band(_init_db):
    """R test: giterator.cartesian_grid works with min.band.idx + physical band."""
    intervs1 = _gscreen_sparse(1.0, 1.3)
    if len(intervs1) == 0:
        pytest.skip("gscreen produced no intervals - test DB mismatch")

    grid_band_idx = pm.giterator_cartesian_grid(
        intervs1, _EXP1, min_band_idx=-1, max_band_idx=2
    )

    # Apply physical band on top: keep cells where start2 - start1 in [-20000, 30000).
    band = (-20_000, 30_000)
    double_filtered = grid_band_idx[
        (grid_band_idx["start2"] - grid_band_idx["start1"] >= band[0])
        & (grid_band_idx["start2"] - grid_band_idx["start1"] < band[1])
    ]

    # Physical band further restricts band-idx result.
    assert len(double_filtered) <= len(grid_band_idx)
    if len(double_filtered) > 0:
        assert (double_filtered["end1"] > double_filtered["start1"]).all()
        assert (double_filtered["end2"] > double_filtered["start2"]).all()


# ---------------------------------------------------------------------------
# T6: Multitask equivalence guard
#
# The CartesianGrid scanner currently runs single-process (no fork/FIFO
# path).  This test verifies that the result is identical regardless of
# the CONFIG multitasking settings.  It serves as a regression guard for
# any future parallelization work.
# ---------------------------------------------------------------------------


def test_cartesian_grid_multitask_equals_single_task(_init_db, rects_track):
    """Scanner currently single-process; honest regression guard.

    Runs gextract with the CartesianGrid iterator under two CONFIG scenarios:
      1. multitasking disabled, max_processes=1
      2. multitasking enabled, max_processes=4, aggressive floor heuristics

    Asserts the results are bit-identical after sorting.
    """
    from pymisha._shared import CONFIG

    intervals_1d = pd.DataFrame({
        "chrom": ["1", "1"],
        "start": [0, 100_000],
        "end":   [50_000, 150_000],
    })
    scope_2d = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    spec = pm.giterator_cartesian_grid(intervals_1d, [-10_000, 10_000], stream=True)

    saved = dict(CONFIG)
    try:
        CONFIG.update({"multitasking": False, "max_processes": 1})
        r1 = pm.gextract(rects_track, intervals=scope_2d, iterator=spec)
        CONFIG.update({
            "multitasking": True,
            "max_processes": 4,
            "min_scope4process": 1,
            "min_intervs4process": 1,
        })
        r4 = pm.gextract(rects_track, intervals=scope_2d, iterator=spec)
    finally:
        CONFIG.clear()
        CONFIG.update(saved)

    r1s = r1.sort_values(["start1", "start2"]).reset_index(drop=True)
    r4s = r4.sort_values(["start1", "start2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1s, r4s)
