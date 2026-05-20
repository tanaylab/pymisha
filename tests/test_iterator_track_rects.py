"""End-to-end tests for gextract(..., iterator="<2D track name>") TrackRects path.

R test file mapping
-------------------
test-gextract3.R        -> prefixed test_r_gextract3_*
test-vtrack.R           -> prefixed test_r_vtrack_*  (TrackRects-specific section)
test-gintervals.mapply.R -> prefixed test_r_mapply_*

Track name equivalences (R -> pymisha test DB):
  test.fixedbin  -> dense_track   (1D fixed-bin / dense)
  test.sparse    -> sparse_track  (1D sparse)
  test.array     -> array_track   (1D array)
  test.rects     -> rects_track   (2D rectangles, existing test-DB track)
  test.computed2d -> no equivalent; uses rects_track as closest proxy where noted
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_1x1_scope() -> pd.DataFrame:
    """Full chrom1=1 x chrom2=1 2D scope."""
    return pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_gextract_track_rects_returns_intersections():
    """iterator=<2D track name> returns one row per object in the iterator track
    that intersects the scope - not one row per scope rect."""
    intervals = _full_1x1_scope()
    result = pm.gextract("rects_track", intervals=intervals, iterator="rects_track")
    assert result is not None
    assert "rects_track" in result.columns
    assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    # The static rects_track has data on chrom1=1; at least one row must come back.
    assert len(result) > 0


def test_gextract_track_rects_fixture(rects_track):
    """Works with the fixture-based 2D track (test.track_rects_iter).

    The conftest fixture creates three rects on chrom-pair (1,1):
      R1: (100, 200, 300, 400)
      R2: (500, 600, 700, 800)
      R3: (50000, 60000, 150000, 160000)

    A scope that covers [0, 500000) x [0, 500000) must return all three.
    """
    intervals = _full_1x1_scope()
    result = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    assert result is not None
    assert len(result) == 3
    assert rects_track in result.columns


def test_gextract_track_rects_chrom_pair_filter(rects_track):
    """Scope restricted to chrom-pair (1, 2) returns the one object on that pair."""
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["2"], "start2": [0], "end2": [300_000],
    })
    result = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    assert result is not None
    assert len(result) == 1
    assert result.iloc[0]["chrom1"] == "1"
    assert result.iloc[0]["chrom2"] == "2"


def test_gextract_track_rects_two_scope_pairs(rects_track):
    """Two scope pairs sum to the expected object count."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"], "start1": [0, 0], "end1": [500_000, 500_000],
        "chrom2": ["1", "2"], "start2": [0, 0], "end2": [500_000, 300_000],
    })
    result = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    assert result is not None
    # 3 on (1,1) + 1 on (1,2) = 4 total
    assert len(result) == 4


def test_gextract_track_rects_has_intervalID(rects_track):
    """Output includes intervalID column.

    In TrackRects mode the intervalID tracks the index of each iterator object
    (track rect) across the full result, not the scope-interval index.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"], "start1": [0, 0], "end1": [500_000, 500_000],
        "chrom2": ["1", "2"], "start2": [0, 0], "end2": [500_000, 300_000],
    })
    result = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    assert "intervalID" in result.columns
    # There are 4 objects total (3 on 1-1, 1 on 1-2); intervalIDs are 0..3.
    assert set(result["intervalID"].unique()) == {0, 1, 2, 3}
    # Each object appears exactly once.
    assert len(result) == 4


def test_gextract_track_rects_values_match_direct_extract(rects_track):
    """Values returned via iterator=<track> equal those from a direct gextract
    over the same objects (no iterator)."""
    intervals = _full_1x1_scope()

    # TrackRects path:
    via_iter = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)

    # Reference: direct object extraction (no iterator) on the same scope.
    ref = pm.gextract(rects_track, intervals=intervals)

    via_iter_sorted = via_iter.sort_values(["start1", "start2"]).reset_index(drop=True)
    ref_sorted = ref.sort_values(["start1", "start2"]).reset_index(drop=True)

    np.testing.assert_array_equal(
        via_iter_sorted[rects_track].to_numpy(),
        ref_sorted[rects_track].to_numpy(),
    )


def test_gextract_track_rects_colnames(rects_track):
    """colnames= is forwarded correctly."""
    intervals = _full_1x1_scope()
    result = pm.gextract(
        rects_track, intervals=intervals, iterator=rects_track, colnames=["my_col"]
    )
    assert "my_col" in result.columns
    assert rects_track not in result.columns


# ---------------------------------------------------------------------------
# Error cases: 1D track or unknown name as iterator
# ---------------------------------------------------------------------------

def test_gextract_track_rects_1d_iterator_raises():
    """When iterator= is a 1D track (not a 2D rectangles/points track),
    gextract raises ValueError (R parity: R raises an error in this case)."""
    intervals = _full_1x1_scope()
    with pytest.raises(ValueError, match="1D track"):
        pm.gextract("rects_track", intervals=intervals, iterator="dense_track")


def test_gextract_track_rects_unknown_iterator_raises():
    """When iterator= is an unknown track name, gextract raises ValueError
    (R parity: R raises an error for unknown track names)."""
    intervals = _full_1x1_scope()
    with pytest.raises(ValueError, match="not a known track"):
        pm.gextract("rects_track", intervals=intervals, iterator="no_such_track_xyz")


# ---------------------------------------------------------------------------
# Vtrack expression with TrackRects iterator (legacy path)
# ---------------------------------------------------------------------------

def test_gextract_track_rects_vtrack_routes_through_scanner(rects_track):
    """A reducing 2D vtrack + iterator=<2D track> now routes through the C++
    scanner and returns one row per iterator object (R parity)."""
    pm.gvtrack_create("v_avg_tr", rects_track, func="avg")
    try:
        intervals = _full_1x1_scope()
        result = pm.gextract("v_avg_tr", intervals=intervals, iterator=rects_track)
        bare = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
        assert result is not None
        assert len(result) == len(bare)
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
        assert "v_avg_tr" in result.columns
        # avg vtrack values should equal the bare-track avg (same aggregation).
        np.testing.assert_array_equal(
            result.sort_values(["start1", "start2"])["v_avg_tr"].to_numpy(),
            bare.sort_values(["start1", "start2"])[rects_track].to_numpy(),
        )
    finally:
        pm.gvtrack_rm("v_avg_tr")


# ---------------------------------------------------------------------------
# R-parity gap: reducing 2D vtrack + iterator=<2D track> row count
# ---------------------------------------------------------------------------

def test_gextract_track_rects_reducing_vtrack_row_count_matches_r(_init_db, rects_track):
    """R-parity gap closed: reducing 2D vtrack + iterator=<2D track> now returns
    one row per iterator object, matching the bare-track call row count.

    Ported from R test-vtrack.R L567-572:
      gvtrack.create('v1', 'test.rects', func='avg')
      r <- gextract('v1', ..., iterator='test.rects')

    Previously xfail-strict (AssertionError) because pymisha returned 1 row
    (legacy one-row-per-scope path). Now routes through the C++ scanner.
    """
    pm.gvtrack_create("v_avg", rects_track, func="avg")
    try:
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [500_000],
            "chrom2": ["1"], "start2": [0], "end2": [500_000],
        })
        result_vtrack = pm.gextract("v_avg", intervals=intervals, iterator=rects_track)
        result_bare = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)

        assert len(result_vtrack) == len(result_bare), (
            f"vtrack returned {len(result_vtrack)} rows; "
            f"bare track returned {len(result_bare)} rows; "
            "they should match: one row per iterator object."
        )
        assert "v_avg" in result_vtrack.columns
    finally:
        pm.gvtrack_rm("v_avg")


# ===========================================================================
# R-parity tests: test-gextract3.R  (iterator = "test.rects" section)
# ===========================================================================
# Lines 21-27 of test-gextract3.R test gextract with two expressions when the
# iterator is a 2D track name.  The R test DB has test.rects and test.computed2d;
# pymisha test DB has rects_track (equivalent) and no computed2d (skipped).


def test_r_gextract3_rects_two_exprs_rects_iterator(rects_track):
    """Ports test-gextract3.R L24: two-expr gextract with 2D track iterator.

    R: expect_regression(
         gextract("test.rects", "test.rects * 3",
                  gintervals.2d(c(2,3), 10000000, 50000000, c(2,4), 30000000, 80000000),
                  iterator = "test.rects"),
         "gextract.44")

    Structural assertions: both expression columns present, finite values,
    second column = 3 * first column (expression correctness).
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result = pm.gextract(
        [rects_track, f"{rects_track} * 3"],
        intervals=intervals,
        iterator=rects_track,
    )
    assert result is not None
    assert len(result) > 0
    cols = list(result.columns)
    # Two data columns expected (plus coordinate and intervalID columns)
    data_cols = [c for c in cols if c not in {
        "chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID",
    }]
    assert len(data_cols) == 2
    v1 = result[data_cols[0]].to_numpy(dtype=float)
    v2 = result[data_cols[1]].to_numpy(dtype=float)
    np.testing.assert_allclose(v2, v1 * 3, rtol=1e-6)


def test_r_gextract3_computed2d_two_exprs_rects_iterator_skipped():
    """Ports test-gextract3.R L25-27 (computed2d variants with rects/computed2d iterator).

    R: expect_regression(gextract("test.computed2d", ..., iterator="test.computed2d"), ...)
       expect_regression(gextract("test.rects", ..., iterator="test.computed2d"), ...)
       expect_regression(gextract("test.computed2d", ..., iterator="test.rects"), ...)

    Skipped: no computed2d equivalent in pymisha test DB.
    TODO: add a computed2d fixture to complete this port.
    """
    pytest.skip("no computed2d track in pymisha test DB - TODO: create fixture")


# --- test-gextract3.R L46-57: 2D track + 1D track as iterator -> error ------

def test_r_gextract3_2d_rects_with_1d_fixedbin_iterator_raises(rects_track):
    """Ports test-gextract3.R L46: 2D rects track + 1D iterator name -> error.

    R: expect_error(gextract("test.rects", 2D_scope, iterator="test.fixedbin"))

    Using a 1D track name as iterator for a 2D scope must raise.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    with pytest.raises((ValueError, Exception)):
        pm.gextract(rects_track, intervals=intervals, iterator="dense_track")


def test_r_gextract3_2d_rects_with_1d_sparse_iterator_raises(rects_track):
    """Ports test-gextract3.R L51: 2D rects track + sparse 1D iterator -> error.

    R: expect_error(gextract("test.rects", 2D_scope, iterator="test.sparse"))
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    with pytest.raises((ValueError, Exception)):
        pm.gextract(rects_track, intervals=intervals, iterator="sparse_track")


def test_r_gextract3_2d_rects_with_1d_array_iterator_raises(rects_track):
    """Ports test-gextract3.R L56: 2D rects track + array 1D iterator -> error.

    R: expect_error(gextract("test.rects", 2D_scope, iterator="test.array"))
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    with pytest.raises((ValueError, Exception)):
        pm.gextract(rects_track, intervals=intervals, iterator="array_track")


# --- test-gextract3.R L58-60: 1D track + rects iterator -> error ------------

def test_r_gextract3_1d_fixedbin_with_rects_iterator_raises(rects_track):
    """Ports test-gextract3.R L58: 1D dense track + rects 2D iterator -> error.

    R: expect_error(gextract("test.fixedbin", 1D_scope, iterator="test.rects"))

    A 1D track with 1D intervals cannot use a 2D track name as iterator.
    """
    intervals = pd.DataFrame({
        "chrom": ["1"], "start": [0], "end": [500_000],
    })
    with pytest.raises((ValueError, Exception)):
        pm.gextract("dense_track", intervals=intervals, iterator=rects_track)


def test_r_gextract3_1d_sparse_with_rects_iterator_raises(rects_track):
    """Ports test-gextract3.R L59: 1D sparse track + rects 2D iterator -> error.

    R: expect_error(gextract("test.sparse", 1D_scope, iterator="test.rects"))
    """
    intervals = pd.DataFrame({
        "chrom": ["1"], "start": [0], "end": [500_000],
    })
    with pytest.raises((ValueError, Exception)):
        pm.gextract("sparse_track", intervals=intervals, iterator=rects_track)


def test_r_gextract3_1d_array_with_rects_iterator_raises(rects_track):
    """Ports test-gextract3.R L60: 1D array track + rects 2D iterator -> error.

    R: expect_error(gextract("test.array", 1D_scope, iterator="test.rects"))
    """
    intervals = pd.DataFrame({
        "chrom": ["1"], "start": [0], "end": [500_000],
    })
    with pytest.raises((ValueError, Exception)):
        pm.gextract("array_track", intervals=intervals, iterator=rects_track)


# --- test-gextract3.R L61-62: 2D track + rects iterator (success) -----------

def test_r_gextract3_2d_rects_with_rects_iterator_success(rects_track):
    """Ports test-gextract3.R L61: 2D rects track + same rects track as iterator.

    R: expect_regression(
         gextract("test.rects", gintervals.2d(chroms1=c(2,3), chroms2=c(2,4)),
                  iterator="test.rects"),
         "gextract.7")

    Result must be non-empty with correct 2D coordinate columns and data column.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"], "start1": [0, 0], "end1": [500_000, 500_000],
        "chrom2": ["1", "2"], "start2": [0, 0], "end2": [500_000, 300_000],
    })
    result = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    assert result is not None
    assert len(result) > 0
    assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
    assert rects_track in result.columns
    assert result[rects_track].notna().all()


def test_r_gextract3_computed2d_with_rects_iterator_skipped():
    """Ports test-gextract3.R L62: computed2d track with rects iterator.

    R: expect_regression(
         gextract("test.computed2d", gintervals.2d(...), iterator="test.rects"),
         "gextract.6")

    Skipped: no computed2d track in pymisha test DB.
    """
    pytest.skip("no computed2d track in pymisha test DB - TODO: create fixture")


# --- test-gextract3.R L63-67: 1D or 2D track + computed2d iterator ---------

def test_r_gextract3_1d_fixedbin_with_computed2d_iterator_raises_skipped():
    """Ports test-gextract3.R L63-65: 1D tracks + computed2d iterator -> error.

    Skipped: no computed2d track in pymisha test DB.
    The equivalent for rects is already covered by test_r_gextract3_1d_*_with_rects_iterator_raises.
    """
    pytest.skip("no computed2d track in pymisha test DB")


def test_r_gextract3_2d_with_computed2d_iterator_skipped():
    """Ports test-gextract3.R L66-67: 2D tracks with computed2d iterator -> success.

    Skipped: no computed2d track in pymisha test DB.
    """
    pytest.skip("no computed2d track in pymisha test DB")


# ===========================================================================
# R-parity tests: test-vtrack.R (TrackRects-iterator section)
# ===========================================================================


def test_r_vtrack_iterator_rects_raises_on_1d_vtrack(rects_track):
    """Ports test-vtrack.R L566-572: vtrack(1D-based) + 2D intervals + rects iterator -> error.

    R: gvtrack.create("v1", intervs, "distance.center")  # 1D intervals-based vtrack
       expect_error(gextract("v1",
                              gintervals.2d(chroms1=c(1,3), ..., chroms2=c(1,4), ...),
                              iterator="test.rects"))

    A vtrack whose source is 1D interval data cannot be extracted over 2D scope
    with a 2D track iterator (R errors out; pymisha must raise too).
    """
    intervals_src = pm.gscreen("dense_track > 0.5", pd.DataFrame({
        "chrom": ["1"], "start": [0], "end": [500_000],
    }))
    pm.gvtrack_create("v_dist_rects", intervals_src, "distance.center")
    try:
        intervals_2d = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [500_000],
            "chrom2": ["1"], "start2": [0], "end2": [500_000],
        })
        with pytest.raises(Exception):
            pm.gextract("v_dist_rects", intervals=intervals_2d, iterator=rects_track)
    finally:
        pm.gvtrack_rm("v_dist_rects")


def test_r_vtrack_iterator_computed2d_raises_on_1d_vtrack_skipped():
    """Ports test-vtrack.R L576-581: vtrack(1D-based) + computed2d iterator -> error.

    R: expect_error(gextract("v1", 2D_scope, iterator="test.computed2d"))

    Skipped: no computed2d track in pymisha test DB.
    The rects equivalent is covered by test_r_vtrack_iterator_rects_raises_on_1d_vtrack.
    """
    pytest.skip("no computed2d track in pymisha test DB")


def test_r_vtrack_fixedbin_dim1_rects_iterator(rects_track):
    """Ports test-vtrack.R L808-815: fixedbin vtrack with dim=1 + rects iterator.

    R: gvtrack.create("v1", "test.fixedbin")
       gvtrack.iterator("v1", dim=1)
       r <- gextract("v1",
                     gintervals.2d(chroms1=c(1,3), 3000000, -1, chroms2=c(2,4), 3000000, -1),
                     iterator="test.rects")
       expect_regression(r, "vtrack.tracktype_test.fixedbin_iterator_dim1_gintervals2d_testrects_regression")

    Structural assertions: non-empty result with 2D coordinates, vtrack column
    present, values finite.  dim=1 projects dim-1 position onto the 1D vtrack.
    """
    pm.gvtrack_create("v_fb_dim1", "dense_track")
    pm.gvtrack_iterator("v_fb_dim1", dim=1)
    try:
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [500_000],
            "chrom2": ["1"], "start2": [0], "end2": [500_000],
        })
        result = pm.gextract("v_fb_dim1", intervals=intervals, iterator=rects_track)
        assert result is not None
        assert len(result) > 0
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
        assert "v_fb_dim1" in result.columns
        assert np.isfinite(result["v_fb_dim1"].to_numpy(dtype=float)).any()
    finally:
        pm.gvtrack_rm("v_fb_dim1")


def test_r_vtrack_fixedbin_dim2_rects_iterator(rects_track):
    """Ports test-vtrack.R L818-825: fixedbin vtrack with dim=2 + rects iterator.

    R: gvtrack.create("v1", "test.fixedbin")
       gvtrack.iterator("v1", dim=2)
       r <- gextract("v1", ..., iterator="test.rects")
       expect_regression(r, "vtrack.fixedbin_dim2_gintervals2d_testrects_regression")

    dim=2 projects dim-2 position onto the 1D vtrack.
    """
    pm.gvtrack_create("v_fb_dim2", "dense_track")
    pm.gvtrack_iterator("v_fb_dim2", dim=2)
    try:
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [500_000],
            "chrom2": ["1"], "start2": [0], "end2": [500_000],
        })
        result = pm.gextract("v_fb_dim2", intervals=intervals, iterator=rects_track)
        assert result is not None
        assert len(result) > 0
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
        assert "v_fb_dim2" in result.columns
        assert np.isfinite(result["v_fb_dim2"].to_numpy(dtype=float)).any()
    finally:
        pm.gvtrack_rm("v_fb_dim2")


def test_r_vtrack_fixedbin_dim1_shifts_rects_iterator(rects_track):
    """Ports test-vtrack.R L828-835: fixedbin vtrack dim=1 with shifts + rects iterator.

    R: gvtrack.create("v1", "test.fixedbin")
       gvtrack.iterator("v1", dim=1, sshift=-130, eshift=224)
       r <- gextract("v1", ..., iterator="test.rects")
       expect_regression(r, "vtrack.fixedbin_dim1_shifts_gintervals2d_testrects_regression")

    Structural assertions same as dim=1 without shifts.
    """
    pm.gvtrack_create("v_fb_dim1s", "dense_track")
    pm.gvtrack_iterator("v_fb_dim1s", dim=1, sshift=-130, eshift=224)
    try:
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [500_000],
            "chrom2": ["1"], "start2": [0], "end2": [500_000],
        })
        result = pm.gextract("v_fb_dim1s", intervals=intervals, iterator=rects_track)
        assert result is not None
        assert len(result) > 0
        assert "v_fb_dim1s" in result.columns
    finally:
        pm.gvtrack_rm("v_fb_dim1s")


def test_r_vtrack_fixedbin_dim1_computed2d_skipped():
    """Ports test-vtrack.R L838-865: fixedbin vtrack dim=1/2 + computed2d iterator.

    R: gvtrack.create("v1", "test.fixedbin"); gvtrack.iterator("v1", dim=1)
       r <- gextract("v1", ..., iterator="test.computed2d")

    Skipped: no computed2d track in pymisha test DB.
    The rects equivalents are covered by test_r_vtrack_fixedbin_dim1_rects_iterator etc.
    """
    pytest.skip("no computed2d track in pymisha test DB")


def test_r_vtrack_rects_2d_iter_custom_shifts_rects_iterator(rects_track):
    """Ports test-vtrack.R L877-884: 2D rects vtrack with custom 2D iterator shifts.

    R: gvtrack.create("v1", "test.rects")
       gvtrack.iterator.2d("v1", sshift1=-1000000, eshift1=-500000, sshift2=2000000, eshift2=2800000)
       r <- gextract("v1",
                     gintervals.2d(chroms1=c(1,3), 3000000, -1, chroms2=c(2,4), 3000000, -1),
                     iterator="test.rects")
       expect_regression(r, "vtrack.rects_iterator2d_customShifts_gintervals2d_testrects_regression")

    The 2D iterator shifts widen/narrow the vtrack query window.  Structural
    assertions: result is non-empty, coordinate columns present, vtrack column
    has finite values.

    Note: large shifts may shift the window outside the test DB's 500kbp chrom
    extents, so we use smaller shifts that keep queries in-bounds.
    """
    pm.gvtrack_create("v_rects_2d", rects_track)
    pm.gvtrack_iterator_2d("v_rects_2d", sshift1=-100, eshift1=100, sshift2=200, eshift2=200)
    try:
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [500_000],
            "chrom2": ["1"], "start2": [0], "end2": [500_000],
        })
        result = pm.gextract("v_rects_2d", intervals=intervals, iterator=rects_track)
        assert result is not None
        assert len(result) > 0
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)
        assert "v_rects_2d" in result.columns
    finally:
        pm.gvtrack_rm("v_rects_2d")


def test_r_vtrack_computed2d_2d_iter_custom_shifts_skipped():
    """Ports test-vtrack.R L887-895: computed2d vtrack + 2D iterator shifts + computed2d iterator.

    R: gvtrack.create("v1", "test.computed2d")
       gvtrack.iterator.2d("v1", sshift1=-1000000, ...)
       r <- gextract("v1", ..., iterator="test.computed2d")

    Skipped: no computed2d track in pymisha test DB.
    """
    pytest.skip("no computed2d track in pymisha test DB")


# ===========================================================================
# R-parity tests: test-gintervals.mapply.R  (TrackRects-iterator section)
# ===========================================================================


def test_r_mapply_2d_rects_iterator_raises(rects_track):
    """Ports test-gintervals.mapply.R L26-29: mapply with 2D rects + rects iterator errors.

    R: expect_error(gintervals.mapply(function(x) {
         max(x + 2, na.rm=TRUE)
       }, "test.rects", .misha$ALLGENOME, iterator="test.rects"))

    R errors because mapply with a 2D track and a 2D track iterator is not
    supported.  Pymisha must raise as well.
    """
    # Use the full 2D scope
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    with pytest.raises(Exception):
        pm.gintervals_mapply(
            lambda x: float(x.max() + 2) if len(x) > 0 else float("nan"),
            rects_track,
            intervals=intervals,
            iterator=rects_track,
        )


# ===========================================================================
# Multitask vs single-task equivalence: TrackRects iterator
# ===========================================================================


def test_track_rects_multitask_equals_single_task(_init_db, rects_track):
    """Scanner currently single-process; honest regression guard for future.

    The TrackRects path in _gextract_2d_via_scanner calls
    pm_extract_2d_scanner directly, bypassing pymisha's fork+FIFO multitask
    infrastructure.  Both CONFIG settings below therefore run single-process.
    The test confirms CONFIG knobs do not corrupt results and will catch any
    future multitask integration that breaks output stability.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    saved = dict(pm.CONFIG)
    try:
        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        r1 = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
        pm.CONFIG.update({
            "multitasking": True, "max_processes": 4,
            "min_scope4process": 1, "min_intervs4process": 1,
        })
        r4 = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    finally:
        pm.CONFIG.clear()
        pm.CONFIG.update(saved)
    r1s = r1.sort_values(["start2", "start1"]).reset_index(drop=True)
    r4s = r4.sort_values(["start2", "start1"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1s, r4s)
