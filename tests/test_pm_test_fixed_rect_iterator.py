import numpy as np
import pytest

import _pymisha


def _scope(rects):
    """Build a scope dict (6 numpy arrays) from a list of rect tuples."""
    n = len(rects)
    return {
        "chrom1": np.array([r[0] for r in rects], dtype=np.int32),
        "start1": np.array([r[1] for r in rects], dtype=np.int64),
        "end1":   np.array([r[2] for r in rects], dtype=np.int64),
        "chrom2": np.array([r[3] for r in rects], dtype=np.int32),
        "start2": np.array([r[4] for r in rects], dtype=np.int64),
        "end2":   np.array([r[5] for r in rects], dtype=np.int64),
    }


def test_single_pair_no_band_yields_grid_cells():
    scope = _scope([(0, 0, 300, 0, 0, 200)])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    assert len(out["start1"]) == 6
    np.testing.assert_array_equal(out["start1"][:3], [0, 100, 200])
    np.testing.assert_array_equal(out["end1"][:3],   [100, 200, 300])
    np.testing.assert_array_equal(out["start2"][:3], [0, 0, 0])
    np.testing.assert_array_equal(out["end2"][:3],   [100, 100, 100])
    np.testing.assert_array_equal(out["start2"][3:], [100, 100, 100])
    np.testing.assert_array_equal(out["end2"][3:],   [200, 200, 200])


def test_clipping_at_scope_bounds():
    scope = _scope([(0, 0, 250, 0, 0, 150)])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    assert len(out["start1"]) == 6
    assert out["start1"][2] == 200 and out["end1"][2] == 250
    assert out["start2"][-1] == 100 and out["end2"][-1] == 150


def test_empty_scope_yields_nothing():
    scope = _scope([])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    assert len(out["start1"]) == 0


def test_zero_binsize_rejected():
    scope = _scope([(0, 0, 100, 0, 0, 100)])
    with pytest.raises(RuntimeError, match="positive"):
        _pymisha.pm_test_fixed_rect_iterator(0, 100, scope, None)
    with pytest.raises(RuntimeError, match="positive"):
        _pymisha.pm_test_fixed_rect_iterator(100, -1, scope, None)


def test_multi_pair_scope_walks_each_pair():
    # Two scope rects on different pairs: (0,0) and (1,1), each 200x200.
    # Binsize 100x100 -> 4 cells per pair, 8 total.
    scope = _scope([
        (0, 0, 200, 0, 0, 200),
        (1, 0, 200, 1, 0, 200),
    ])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    assert len(out["start1"]) == 8
    # First 4 cells on pair (0, 0):
    assert (out["chrom1"][:4] == 0).all()
    assert (out["chrom2"][:4] == 0).all()
    # Next 4 cells on pair (1, 1):
    assert (out["chrom1"][4:] == 1).all()
    assert (out["chrom2"][4:] == 1).all()


def test_scope_rect_smaller_than_binsize():
    # 50x50 scope at binsize (100, 100): one cell clipped to [0, 50) x [0, 50).
    scope = _scope([(0, 0, 50, 0, 0, 50)])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    assert len(out["start1"]) == 1
    assert out["start1"][0] == 0 and out["end1"][0] == 50
    assert out["start2"][0] == 0 and out["end2"][0] == 50


def test_scope_rect_zero_area_skipped():
    # start == end on one side: zero-area rect, must produce no cells.
    scope = _scope([
        (0, 100, 100, 0, 0, 100),  # zero width
        (0, 0, 100, 0, 0, 0),      # zero height
        (1, 0, 100, 1, 0, 100),    # normal
    ])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    assert len(out["start1"]) == 1
    assert out["chrom1"][0] == 1


def test_band_filters_out_of_band_cells():
    # 400x400 scope on (0,0) at binsize (100, 100) -> 16 cells without band.
    # With band (d1=-200, d2=200), only cells whose center diff |c1 - c2| <= 200
    # survive (R's DiagonalBand semantics).
    scope = _scope([(0, 0, 400, 0, 0, 400)])
    out_full = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    out_band = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, (-200, 200))
    assert len(out_full["start1"]) == 16
    assert len(out_band["start1"]) < 16  # narrower band drops cells
    # All band-surviving cells must satisfy the band constraint:
    # rect intersects band (d1, d2) iff s2 - e1 < d2 AND e2 - s1 > d1
    for s1, e1, s2, e2 in zip(out_band["start1"], out_band["end1"],
                               out_band["start2"], out_band["end2"]):
        assert (s2 - e1) < 200 and (e2 - s1) > -200


def test_band_skips_inter_chrom_pair():
    # R: when band is active and chromid1 != chromid2, the entire scope rect is
    # skipped. Inter-chrom pairs without band still yield a full grid.
    scope = _scope([(0, 0, 400, 1, 0, 400)])  # (chr 0, chr 1) - inter-chrom
    out_full = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, None)
    out_band = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, (-100, 100))
    assert len(out_full["start1"]) == 16   # no band: normal grid
    assert len(out_band["start1"]) == 0    # band active + inter-chrom: skipped


def test_band_skips_rows_fully_out_of_band():
    # Band (300, 400) on a 400x400 scope at binsize 100x100.
    # do_intersect requires x2-y1 > d1 (strict), so row y=[100,200): 400-100=300 > 300
    # is false. All rows except y=[0,100) are out of band and get skipped.
    # Only row y=[0,100) intersects: one cell at x=[300,400).
    scope = _scope([(0, 0, 400, 0, 0, 400)])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, (300, 400))
    assert len(out["start1"]) == 1
    assert out["start1"][0] == 300 and out["end1"][0] == 400
    assert out["start2"][0] == 0   and out["end2"][0] == 100


def test_band_per_cell_y_reset():
    # Band (50, 250) on 400x400 scope at binsize 100x100.
    # Row y=[0,100) has cells at x=[50,100), x=[100,200), x=[200,300), x=[300,350).
    # Cell x=[50,100): shrink2intersected sets end2=50 (x2-y2=100-100=0 < d1=50).
    # Without the per-cell y-reset bug fix, cell x=[100,200) inherits end2=50 and
    # do_contain reports "fully inside" (false positive), yielding wrong end2=50.
    # With the fix, each cell resets start2/end2 from the row bounds before shrink,
    # so cell x=[100,200) correctly has end2=100.
    scope = _scope([(0, 0, 400, 0, 0, 400)])
    out = _pymisha.pm_test_fixed_rect_iterator(100, 100, scope, (50, 250))
    s1 = np.asarray(out["start1"])
    e1 = np.asarray(out["end1"])
    s2 = np.asarray(out["start2"])
    e2 = np.asarray(out["end2"])
    # Find cell at x=[100,200) in row y=[0,100).
    mask = (s1 == 100) & (e1 == 200) & (s2 == 0)
    assert mask.sum() == 1, f"expected exactly one cell at x=[100,200) y=[0,?), got {mask.sum()}"
    assert e2[mask][0] == 100, (
        f"cell x=[100,200) y=[0,?) has end2={e2[mask][0]}, expected 100 "
        f"(end2=50 indicates the y-reset bug is present)"
    )
    # Find cell at x=[200,300) in row y=[0,100) - also must not inherit prior shrink.
    mask2 = (s1 == 200) & (e1 == 300) & (s2 == 0)
    assert mask2.sum() == 1
    assert e2[mask2][0] == 100, (
        f"cell x=[200,300) y=[0,?) has end2={e2[mask2][0]}, expected 100"
    )


# ---------------------------------------------------------------------------
# Scanner end-to-end test: FixedRect iterator + 2D scanner (T6)
# ---------------------------------------------------------------------------

def test_scanner_reuse_resets_state(_init_db):
    """Regression test for the m_vars accumulation bug (critical fix).

    Calls run() twice on the same PMTrackExpr2DScanner instance via
    pm_test_scanner_reuse.  The second call uses a different scope, so if
    m_vars is NOT reset between calls the second run would either crash
    (double-add) or return stale data from the first run's buffer.

    We verify:
      - n2 (second run cell count) matches a fresh scanner run over scope2.
      - values from the second run match a fresh scanner run over scope2.
    """
    track_name = "rects_track"
    chrom1_id = np.int32(0)

    # scope1: 3x3 grid (300k x 300k at 100k bins)
    scope1 = {
        "chrom1": np.array([chrom1_id], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([300_000], dtype=np.int64),
        "chrom2": np.array([chrom1_id], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([300_000], dtype=np.int64),
    }
    # scope2: 5x5 grid (500k x 500k at 100k bins)
    scope2 = {
        "chrom1": np.array([chrom1_id], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([500_000], dtype=np.int64),
        "chrom2": np.array([chrom1_id], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([500_000], dtype=np.int64),
    }
    width, height = 100_000, 100_000

    # Run the reuse binding (same scanner, two calls).
    n1, n2, vals_reuse = _pymisha.pm_test_scanner_reuse(
        width, height, track_name, "area", scope1, scope2
    )

    # Fresh reference scanner over scope2.
    vals_fresh = _pymisha.pm_test_fixed_rect_scanner(
        width, height, track_name, "area", scope2, None
    )

    assert n1 == 9,  f"first run over 300k x 300k at 100k bins: expected 9 cells, got {n1}"
    assert n2 == 25, f"second run over 500k x 500k at 100k bins: expected 25 cells, got {n2}"
    assert len(vals_reuse) == 25, f"reuse run2 values length: expected 25, got {len(vals_reuse)}"

    # If m_vars was not reset, values_for_var(0) would alias the first-run
    # buffer and the values would differ from a fresh run.
    np.testing.assert_array_equal(
        vals_reuse, vals_fresh,
        err_msg="scanner reuse: second run values differ from a fresh scanner run (m_vars reset bug)"
    )


def test_scanner_end_to_end_fixed_rect(_init_db):
    """Scanner + FixedRect iterator + RECTS track over the testdb rects_track.

    Uses the pre-existing rects_track fixture from the testdb (R-imported,
    chr1-chr1 naming).  Chrom '1' is chromid 0 in the test DB.

    A 100_000 x 100_000 grid over [0, 500_000) x [0, 500_000) yields a
    5x5 = 25-cell FixedRect grid.  We verify:
      - run() returns exactly 25 cells (no off-by-one from the FixedRect
        priming in begin()->next()).
      - The iterator-only walk (pm_test_fixed_rect_iterator) and the
        scanner walk emit the same number of cells (walk-loop parity).
      - No exception is raised and the result is a float64 array.
    """
    track_name = "rects_track"
    # chromid for '1' is 0 in the test DB.
    chrom1_id = np.int32(0)

    scope_dict = {
        "chrom1": np.array([chrom1_id], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([500_000], dtype=np.int64),
        "chrom2": np.array([chrom1_id], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([500_000], dtype=np.int64),
    }
    width, height = 100_000, 100_000

    # Check iterator emits the expected grid size.
    itr_out = _pymisha.pm_test_fixed_rect_iterator(width, height, scope_dict, None)
    expected_cells = len(itr_out["start1"])
    assert expected_cells == 25, (
        f"FixedRect iterator over 500k x 500k with 100k bins should yield 25 cells, "
        f"got {expected_cells}"
    )

    # Run the scanner and verify it matches.
    values = _pymisha.pm_test_fixed_rect_scanner(
        width, height, track_name, "area", scope_dict, None
    )
    assert values.dtype == np.float64
    assert values.shape == (25,), (
        f"scanner should return 25 values (5x5 grid), got shape {values.shape}"
    )
    # The iterator walk and scanner walk must agree on cell count.
    assert len(values) == expected_cells, (
        "scanner emitted a different number of cells than the bare iterator"
    )


# ---------------------------------------------------------------------------
# pm_extract_2d_scanner tests (T7)
# ---------------------------------------------------------------------------

def _scope_dict(chrom_id, start1, end1, start2, end2):
    """Build a single-rect scope dict."""
    return {
        "chrom1": np.array([chrom_id], dtype=np.int32),
        "start1": np.array([start1],   dtype=np.int64),
        "end1":   np.array([end1],     dtype=np.int64),
        "chrom2": np.array([chrom_id], dtype=np.int32),
        "start2": np.array([start2],   dtype=np.int64),
        "end2":   np.array([end2],     dtype=np.int64),
    }


def test_pm_extract_2d_scanner_fixed_rect_basic(_init_db):
    """Smoke test: pm_extract_2d_scanner returns same cell count as pm_test_fixed_rect_scanner."""
    # chromid 0 = chrom "1" in the test DB.
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    policy = {"kind": "fixed_rect", "width": 100_000, "height": 100_000}
    vars_list = [("rects_track", "area")]
    colnames = ["rects_track"]

    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)

    # Value array present and correct shape (5x5 = 25 cells).
    assert "rects_track" in result
    assert result["rects_track"].shape == (25,)
    assert result["rects_track"].dtype == np.float64

    # All six coord arrays present and correct shape.
    for key in ("_chrom1", "_start1", "_end1", "_chrom2", "_start2", "_end2"):
        assert key in result, f"missing coord array: {key}"
        assert result[key].shape == (25,), f"{key} shape mismatch"

    # All coords on chrom 0 (the only chrom in scope).
    assert (result["_chrom1"] == 0).all()
    assert (result["_chrom2"] == 0).all()

    # Values must match pm_test_fixed_rect_scanner (same underlying scanner path).
    ref_values = _pymisha.pm_test_fixed_rect_scanner(
        100_000, 100_000, "rects_track", "area", scope, None
    )
    np.testing.assert_array_equal(
        result["rects_track"], ref_values,
        err_msg="pm_extract_2d_scanner values differ from pm_test_fixed_rect_scanner"
    )


def test_pm_extract_2d_scanner_multi_var(_init_db):
    """Multi-var: two vars on the same track with different funcs."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    policy = {"kind": "fixed_rect", "width": 100_000, "height": 100_000}
    vars_list = [("rects_track", "area"), ("rects_track", "weighted.sum")]
    colnames = ["area_col", "wsum_col"]

    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)

    assert result["area_col"].shape == (25,)
    assert result["wsum_col"].shape == (25,)
    # area and weighted.sum differ in general (weighted.sum counts rect-pixel
    # overlap, area counts full-rect coverage).
    assert "_chrom1" in result


def test_pm_extract_2d_scanner_empty_scope(_init_db):
    """Empty scope dict yields zero-length arrays."""
    scope = {
        "chrom1": np.array([], dtype=np.int32),
        "start1": np.array([], dtype=np.int64),
        "end1":   np.array([], dtype=np.int64),
        "chrom2": np.array([], dtype=np.int32),
        "start2": np.array([], dtype=np.int64),
        "end2":   np.array([], dtype=np.int64),
    }
    policy = {"kind": "fixed_rect", "width": 100_000, "height": 100_000}
    result = _pymisha.pm_extract_2d_scanner(
        policy, scope, [("rects_track", "area")], ["col"], None
    )
    assert result["col"].shape == (0,)
    assert result["_chrom1"].shape == (0,)


def test_pm_extract_2d_scanner_rejects_unsupported_kinds(_init_db):
    """Unsupported iterator kinds raise RuntimeError with helpful message."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    vars_list = [("rects_track", "area")]
    colnames = ["col"]

    # Unknown kind.
    with pytest.raises(RuntimeError, match="Unknown iterator kind"):
        _pymisha.pm_extract_2d_scanner(
            {"kind": "garbage"}, scope, vars_list, colnames, None
        )


def test_pm_extract_2d_scanner_missing_kind(_init_db):
    """Policy dict without 'kind' raises RuntimeError."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    with pytest.raises(RuntimeError, match="missing 'kind'"):
        _pymisha.pm_extract_2d_scanner(
            {"width": 100_000, "height": 100_000},
            scope, [("rects_track", "area")], ["col"], None
        )


def test_pm_extract_2d_scanner_colnames_mismatch(_init_db):
    """Mismatched colnames vs vars raises RuntimeError."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    with pytest.raises(RuntimeError, match="colnames length"):
        _pymisha.pm_extract_2d_scanner(
            {"kind": "fixed_rect", "width": 100_000, "height": 100_000},
            scope,
            [("rects_track", "area")],
            ["col1", "col2"],   # two colnames but one var
            None,
        )


def test_pm_extract_2d_scanner_rejects_reserved_colname(_init_db):
    """Colnames that collide with internal coord keys are rejected."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    policy = {"kind": "fixed_rect", "width": 100_000, "height": 100_000}
    vars_list = [("rects_track", "area")]
    for bad in ("_chrom1", "_start1", "_end1", "_chrom2", "_start2", "_end2"):
        with pytest.raises(RuntimeError, match="collides with an internal coord key"):
            _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, [bad], None)


def test_pm_extract_2d_scanner_rejects_non_string_kind(_init_db):
    """Policy kind that is not a string raises RuntimeError with clear message."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    with pytest.raises(RuntimeError, match="'kind' must be a string"):
        _pymisha.pm_extract_2d_scanner(
            {"kind": 42}, scope, [("rects_track", "area")], ["rects_track"], None
        )


# ---------------------------------------------------------------------------
# pm_extract_2d_scanner with track_rects iterator (T3)
# ---------------------------------------------------------------------------

def test_pm_extract_2d_scanner_track_rects_smoke(rects_track):
    """Smoke test: pm_extract_2d_scanner with track_rects returns expected shape."""
    scope = {
        "chrom1": np.array([0], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([500_000], dtype=np.int64),
        "chrom2": np.array([0], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([500_000], dtype=np.int64),
    }
    policy = {"kind": "track_rects", "track_name": rects_track}
    vars_list = [(rects_track, "area")]
    colnames = ["rects_col"]
    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)

    assert "rects_col" in result
    assert "_chrom1" in result
    assert "_start1" in result
    assert "_end1" in result
    assert "_chrom2" in result
    assert "_start2" in result
    assert "_end2" in result
    # 3 objects on pair (0,0) x full-chrom scope => 3 emissions.
    assert result["rects_col"].shape == (3,)
    assert result["rects_col"].shape == result["_chrom1"].shape


def test_pm_extract_2d_scanner_track_rects_multi_pair(rects_track):
    """Two-pair scope: (0,0) has 3 objects, (0,1) has 1 => 4 total rows."""
    scope = {
        "chrom1": np.array([0, 0], dtype=np.int32),
        "start1": np.array([0, 0], dtype=np.int64),
        "end1":   np.array([500_000, 500_000], dtype=np.int64),
        "chrom2": np.array([0, 1], dtype=np.int32),
        "start2": np.array([0, 0], dtype=np.int64),
        "end2":   np.array([500_000, 300_000], dtype=np.int64),
    }
    policy = {"kind": "track_rects", "track_name": rects_track}
    vars_list = [(rects_track, "area")]
    colnames = ["rects_col"]
    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)

    assert result["rects_col"].shape == (4,)


def test_pm_extract_2d_scanner_track_rects_values_are_float64(rects_track):
    """Output value array is float64."""
    scope = {
        "chrom1": np.array([0], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([500_000], dtype=np.int64),
        "chrom2": np.array([0], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([500_000], dtype=np.int64),
    }
    policy = {"kind": "track_rects", "track_name": rects_track}
    result = _pymisha.pm_extract_2d_scanner(
        policy, scope, [(rects_track, "area")], ["col"], None
    )
    assert result["col"].dtype == np.float64


def test_pm_extract_2d_scanner_track_rects_missing_track_name(_init_db):
    """Missing track_name raises RuntimeError."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    with pytest.raises(RuntimeError, match="track_name"):
        _pymisha.pm_extract_2d_scanner(
            {"kind": "track_rects"}, scope,
            [("rects_track", "area")], ["col"], None,
        )


def test_pm_extract_2d_scanner_track_rects_non_string_track_name(_init_db):
    """Non-string track_name raises RuntimeError."""
    scope = _scope_dict(0, 0, 500_000, 0, 500_000)
    with pytest.raises(RuntimeError, match="must be a string"):
        _pymisha.pm_extract_2d_scanner(
            {"kind": "track_rects", "track_name": 42}, scope,
            [("rects_track", "area")], ["col"], None,
        )


def test_pm_extract_2d_scanner_intervals(_init_db, rects_track):
    """Smoke test: kind='intervals' uses the scope rects as iteration source."""
    import numpy as np
    import _pymisha

    # Two scope rects on chr1 x chr1.
    scope = {
        "chrom1": np.array([0, 0], dtype=np.int32),
        "start1": np.array([0, 200_000], dtype=np.int64),
        "end1":   np.array([100_000, 300_000], dtype=np.int64),
        "chrom2": np.array([0, 0], dtype=np.int32),
        "start2": np.array([0, 200_000], dtype=np.int64),
        "end2":   np.array([100_000, 300_000], dtype=np.int64),
    }
    policy = {"kind": "intervals"}
    vars_list = [(rects_track, "area")]
    colnames = [rects_track]
    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)
    # One row per scope interval (each scope rect becomes one output row).
    assert result[rects_track].shape == (2,)
    assert (np.asarray(result["_chrom1"]) == 0).all()
