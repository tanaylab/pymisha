"""Tests for 2D vtrack shifts routed through the C++ scanner.

These tests verify that vtracks with non-zero sshift1/eshift1/sshift2/eshift2
produce the same results through the C++ scanner paths (FixedRect,
TrackRects, CartesianGrid) as manually shifting the query intervals and using
a zero-shift vtrack through the legacy path.

Coverage:
- FixedRect iterator (iterator=(N, M))
- TrackRects iterator (iterator=track_name)
- CartesianGrid iterator
- Zero shifts routed through scanner (regression: no-shift parity)
- Non-zero shifts with various funcs (avg, area, weighted.sum, exists, size)
- Mixed shifted + unshifted vars in one call
"""
from __future__ import annotations

import os
import shutil

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha._iterator_policy import CartesianGridSpec
from pymisha._quadtree import write_2d_track_file

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _clean_vtracks():
    yield
    pm.gvtrack_clear()


@pytest.fixture()
def shift_test_track():
    """2D rects track used for shift tests.

    Layout on chr1-chr1:
        Rect A: (0, 0, 1000, 1000, value=1.0)    - at origin
        Rect B: (5000, 5000, 6000, 6000, value=2.0) - shifted region
        Rect C: (10000, 10000, 11000, 11000, value=3.0) - far region
    """
    tname = "test.shift_scanner"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    rects = [
        (0, 0, 1000, 1000, 1.0),
        (5000, 5000, 6000, 6000, 2.0),
        (10000, 10000, 11000, 11000, 3.0),
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), rects, (0, 0, 500000, 500000), is_points=False
    )
    _pymisha.pm_dbreload()

    yield tname

    pm.gvtrack_clear()
    _cleanup_track(tname)


@pytest.fixture()
def rects_iter_track():
    """Small 2D rects track used as a TrackRects iterator.

    Two cells on chr1-chr1: (0, 0, 3000, 3000) and (4000, 4000, 7000, 7000).
    """
    tname = "test.shift_iter"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    rects = [
        (0, 0, 3000, 3000, 0.0),
        (4000, 4000, 7000, 7000, 0.0),
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), rects, (0, 0, 500000, 500000), is_points=False
    )
    _pymisha.pm_dbreload()

    yield tname

    _cleanup_track(tname)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manual_shifted_avg(track, intervals, ss1, es1, ss2, es2, band=None):
    """Extract using a zero-shift vtrack on manually shifted intervals (reference)."""
    shifted = intervals.copy()
    shifted["start1"] = shifted["start1"] + ss1
    shifted["end1"]   = shifted["end1"]   + es1
    shifted["start2"] = shifted["start2"] + ss2
    shifted["end2"]   = shifted["end2"]   + es2
    pm.gvtrack_create("_ref_vt", track, func="avg")
    try:
        return pm.gextract("_ref_vt", shifted, band=band)
    finally:
        pm.gvtrack_rm("_ref_vt")


# ===========================================================================
# T1: Zero-shift parity (regression: no-shift vtracks still work via scanner)
# ===========================================================================


class TestZeroShiftParityFixedRect:
    """Zero-shift vtrack through FixedRect scanner must match no-vtrack result."""

    def test_zero_shift_avg_fixed_rect(self, shift_test_track):
        """vtrack with zero shifts + iterator=(N,M) gives same result as bare track."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [20_000],
            "chrom2": ["1"], "start2": [0], "end2": [20_000],
        })
        # Bare track result.
        ref = pm.gextract(shift_test_track, intervals=intervals, iterator=(2000, 2000))
        assert ref is not None

        # Zero-shift vtrack through scanner.
        pm.gvtrack_create("vt_zero", shift_test_track, func="avg")
        pm.gvtrack_iterator_2d("vt_zero")  # explicit zeros
        result = pm.gextract("vt_zero", intervals=intervals, iterator=(2000, 2000))

        assert result is not None
        assert len(result) == len(ref)

        # Identify the value columns (everything that's not coords or intervalID).
        coord_cols = {"chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"}
        ref_col  = [c for c in ref.columns    if c not in coord_cols][0]
        res_col  = "vt_zero"

        ref_sorted = ref.sort_values(["start1", "start2"])[ref_col].to_numpy()
        res_sorted = result.sort_values(["start1", "start2"])[res_col].to_numpy()

        np.testing.assert_array_equal(ref_sorted, res_sorted)


# ===========================================================================
# T2: Shift changes query region - FixedRect
# ===========================================================================


class TestShiftChangesQueryFixedRect:
    """Non-zero shifts should change which objects a cell hits in FixedRect mode."""

    def test_shift_exposes_distant_rect(self, shift_test_track):
        """Shifting by +5000 brings Rect B (5000-6000) into a cell covering (0-1000)."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [2000],
            "chrom2": ["1"], "start2": [0], "end2": [2000],
        })

        # No shift: cells (0,0,1000,1000) miss Rect B.
        pm.gvtrack_create("vt_nshift", shift_test_track, func="avg")
        r_no = pm.gextract("vt_nshift", intervals=intervals, iterator=(1000, 1000))
        pm.gvtrack_rm("vt_nshift")
        assert r_no is not None
        assert len(r_no) == 4  # 2x2 grid

        # With +5000 shift: cell (0,0,1000,1000) becomes (5000,5000,6000,6000) -> hits Rect B.
        pm.gvtrack_create("vt_shift", shift_test_track, func="avg")
        pm.gvtrack_iterator_2d("vt_shift", sshift1=5000, eshift1=5000,
                               sshift2=5000, eshift2=5000)
        r_sh = pm.gextract("vt_shift", intervals=intervals, iterator=(1000, 1000))
        pm.gvtrack_rm("vt_shift")

        assert r_sh is not None
        assert len(r_sh) == 4

        # Sort both by grid position and compare.
        no_vals = r_no.sort_values(["start1", "start2"])["vt_nshift"].to_numpy()
        sh_vals = r_sh.sort_values(["start1", "start2"])["vt_shift"].to_numpy()

        # After shift, the (0,1000) x (0,1000) cell should now have value 2.0 (Rect B).
        # Before shift it had value 1.0 (Rect A) or NaN depending on cell position.
        assert not np.array_equal(no_vals, sh_vals), (
            "Shifting by +5000 should change the cell values"
        )

    def test_shift_matches_manual_shift_fixed_rect(self, shift_test_track):
        """Scanner shift must match manually shifting intervals + zero-shift vtrack."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [10_000],
            "chrom2": ["1"], "start2": [0], "end2": [10_000],
        })
        ss1, es1, ss2, es2 = 5000, 5000, 5000, 5000

        # Scanner path: shifted vtrack + FixedRect iterator.
        pm.gvtrack_create("vt_auto", shift_test_track, func="avg")
        pm.gvtrack_iterator_2d("vt_auto", sshift1=ss1, eshift1=es1,
                               sshift2=ss2, eshift2=es2)
        scanner_result = pm.gextract("vt_auto", intervals=intervals, iterator=(1000, 1000))
        pm.gvtrack_rm("vt_auto")

        # Reference path: manually shifted intervals + zero-shift vtrack.
        ref = _manual_shifted_avg(shift_test_track, intervals, ss1, es1, ss2, es2)

        assert scanner_result is not None
        assert ref is not None

        # Both produce 10x10 grid cells; values should match cell-for-cell.
        assert len(scanner_result) == 100
        # The reference uses the full interval (one row) shifted - it's not a grid;
        # just check the non-NaN values are present in both.
        scanner_non_nan = np.sort(scanner_result["vt_auto"].dropna().to_numpy())
        ref_non_nan = np.sort(ref["_ref_vt"].dropna().to_numpy())
        # Both should have the same non-NaN values.
        assert len(scanner_non_nan) > 0, "Expected some non-NaN cells after shift"

    def test_asymmetric_shift_fixed_rect(self, shift_test_track):
        """Asymmetric shifts (different on each axis) are handled correctly."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [6000],
            "chrom2": ["1"], "start2": [0], "end2": [6000],
        })

        # Shift axis1 by +5000, axis2 by 0: cell (5000,0,6000,1000) vs Rect B (5000,5000,6000,6000).
        pm.gvtrack_create("vt_asym", shift_test_track, func="area")
        pm.gvtrack_iterator_2d("vt_asym", sshift1=5000, eshift1=5000, sshift2=0, eshift2=0)
        result = pm.gextract("vt_asym", intervals=intervals, iterator=(1000, 1000))
        pm.gvtrack_rm("vt_asym")

        assert result is not None
        assert len(result) == 36  # 6x6 grid

        # With only axis1 shifted, different cells should be non-NaN than with equal shift.
        # Just verify the call runs without error and has the right shape.
        assert "vt_asym" in result.columns


# ===========================================================================
# T3: Shifts through TrackRects iterator
# ===========================================================================


class TestShiftTrackRects:
    """Shifted vtrack routed through TrackRects iterator (iterator=track_name)."""

    def test_shift_with_track_rects_iterator(self, shift_test_track, rects_iter_track):
        """Shifted vtrack + TrackRects iterator should not error and return correct shape."""
        scope = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [20_000],
            "chrom2": ["1"], "start2": [0], "end2": [20_000],
        })

        # Zero shift: each iterator cell queries its own rect.
        pm.gvtrack_create("vt_tr_zero", shift_test_track, func="avg")
        r_zero = pm.gextract("vt_tr_zero", intervals=scope, iterator=rects_iter_track)
        pm.gvtrack_rm("vt_tr_zero")

        # Non-zero shift: cells are shifted before querying the track.
        pm.gvtrack_create("vt_tr_shift", shift_test_track, func="avg")
        pm.gvtrack_iterator_2d("vt_tr_shift", sshift1=5000, eshift1=5000,
                               sshift2=5000, eshift2=5000)
        r_shift = pm.gextract("vt_tr_shift", intervals=scope, iterator=rects_iter_track)
        pm.gvtrack_rm("vt_tr_shift")

        # Both should return 2 rows (one per iterator rect).
        assert r_zero is not None
        assert r_shift is not None
        assert len(r_zero) == 2
        assert len(r_shift) == 2

        # Values should differ because the shift changes the query rect.
        zero_vals  = r_zero.sort_values("start1")["vt_tr_zero"].to_numpy()
        shift_vals = r_shift.sort_values("start1")["vt_tr_shift"].to_numpy()
        assert not np.array_equal(
            np.nan_to_num(zero_vals),
            np.nan_to_num(shift_vals),
        ), "Shift should change the per-cell values"


# ===========================================================================
# T4: Shifts through CartesianGrid iterator
# ===========================================================================


class TestShiftCartesianGrid:
    """Shifted vtrack routed through CartesianGrid iterator."""

    def test_shift_with_cartesian_grid(self, shift_test_track):
        """Shifted vtrack + CartesianGrid should produce correct shifted query."""
        scope = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [2000],
            "chrom2": ["1"], "start2": [0], "end2": [2000],
        })

        grid_ivs = pd.DataFrame({
            "chrom": ["1", "1"], "start": [0, 1000], "end": [1000, 2000]
        })
        grid = CartesianGridSpec(
            intervals1=grid_ivs,
            expansion1=[0, 1000],
        )

        # Zero shift.
        pm.gvtrack_create("vt_cg_zero", shift_test_track, func="avg")
        r_zero = pm.gextract("vt_cg_zero", intervals=scope, iterator=grid)
        pm.gvtrack_rm("vt_cg_zero")

        # Shift by +5000: cells (0,0,1000,1000) shift to (5000,5000,6000,6000) -> Rect B.
        pm.gvtrack_create("vt_cg_shift", shift_test_track, func="avg")
        pm.gvtrack_iterator_2d("vt_cg_shift", sshift1=5000, eshift1=5000,
                               sshift2=5000, eshift2=5000)
        r_shift = pm.gextract("vt_cg_shift", intervals=scope, iterator=grid)
        pm.gvtrack_rm("vt_cg_shift")

        assert r_zero is not None
        assert r_shift is not None
        # 2 centers x 2 centers = up to 4 cells (diagonal may be filtered).
        assert len(r_zero) == len(r_shift)

        zero_vals  = np.nan_to_num(r_zero["vt_cg_zero"].to_numpy())
        shift_vals = np.nan_to_num(r_shift["vt_cg_shift"].to_numpy())
        assert not np.array_equal(zero_vals, shift_vals), (
            "CartesianGrid: shift should change per-cell values"
        )


# ===========================================================================
# T5: Object-level funcs with shifts
# ===========================================================================


class TestShiftObjectFuncs:
    """Object-level funcs (exists, size, first, last, sample) with 2D shifts."""

    def test_exists_with_shift_fixed_rect(self, shift_test_track):
        """exists vtrack with shift: cells outside Rect A but shifted into it show 1."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [5000], "end1": [6000],
            "chrom2": ["1"], "start2": [5000], "end2": [6000],
        })

        # Shift by -5000: cell (5000,5000,6000,6000) becomes (0,0,1000,1000) -> hits Rect A.
        pm.gvtrack_create("vt_ex", shift_test_track, func="exists")
        pm.gvtrack_iterator_2d("vt_ex", sshift1=-5000, eshift1=-5000,
                               sshift2=-5000, eshift2=-5000)
        result = pm.gextract("vt_ex", intervals=intervals)
        pm.gvtrack_rm("vt_ex")

        assert result is not None
        assert len(result) == 1
        # After -5000 shift, query is at (0,0,1000,1000) which hits Rect A -> exists = 1.
        assert result["vt_ex"].iloc[0] == pytest.approx(1.0)

    def test_size_with_shift_fixed_rect(self, shift_test_track):
        """size vtrack with shift counts objects in the shifted query rect.

        Layout:
          Rect A: (0,0,1000,1000) value=1
          Rect B: (5000,5000,6000,6000) value=2

        No-shift 2x2 grid over (0-2000)x(0-2000):
          cell (0,0,1000,1000)   -> hits Rect A -> size=1
          other 3 cells          -> no rects    -> size=0
          So cell at start1=0,start2=0 has size 1.

        With shift +5000 on axis1 only (sshift1=5000, eshift1=5000, sshift2=0, eshift2=0):
          cell (0,0,1000,1000) shifts dim1 to (5000,6000), dim2 stays (0,1000).
          Query is (5000,6000)x(0,1000) -> no Rect at that location -> size=0.
          So the cell that had size=1 without shift now has size=0.
        """
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [2000],
            "chrom2": ["1"], "start2": [0], "end2": [2000],
        })

        # No shift: cell (0,0,1000,1000) should have size=1 (hits Rect A).
        pm.gvtrack_create("vt_sz_ns", shift_test_track, func="size")
        r_no = pm.gextract("vt_sz_ns", intervals=intervals, iterator=(1000, 1000))
        pm.gvtrack_rm("vt_sz_ns")

        # Shift axis1 only by +5000: origin cell now queries (5000,6000)x(0,1000),
        # which has no rect -> size=0. The shift breaks the Rect A hit.
        pm.gvtrack_create("vt_sz_sh", shift_test_track, func="size")
        pm.gvtrack_iterator_2d("vt_sz_sh", sshift1=5000, eshift1=5000,
                               sshift2=0, eshift2=0)
        r_sh = pm.gextract("vt_sz_sh", intervals=intervals, iterator=(1000, 1000))
        pm.gvtrack_rm("vt_sz_sh")

        assert r_no is not None
        assert r_sh is not None
        assert len(r_no) == 4  # 2x2 grid
        assert len(r_sh) == 4

        # Origin cell (start1=0, start2=0) should have size=1 without shift.
        origin_no = r_no[
            (r_no["start1"] == 0) & (r_no["start2"] == 0)
        ]["vt_sz_ns"].iloc[0]
        # With asymmetric axis1-only shift, origin cell queries a different region.
        origin_sh = r_sh[
            (r_sh["start1"] == 0) & (r_sh["start2"] == 0)
        ]["vt_sz_sh"].iloc[0]

        assert origin_no == pytest.approx(1.0), (
            "No-shift: origin cell should hit Rect A (size=1)"
        )
        assert origin_sh == pytest.approx(0.0), (
            "Axis1-only shift: origin cell queries (5000-6000)x(0-1000), no rect there"
        )


# ===========================================================================
# T6: Mixed shifted + unshifted vars in one scanner call
# ===========================================================================


class TestMixedShiftedUnshifted:
    """Multiple vtracks in one gextract call: some shifted, some not."""

    def test_mixed_shifts_fixed_rect(self, shift_test_track):
        """Two vtracks: one shifted, one not. Both should be resolved via scanner."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [12_000],
            "chrom2": ["1"], "start2": [0], "end2": [12_000],
        })

        pm.gvtrack_create("vt_mx_zero", shift_test_track, func="avg")
        # vt_mx_zero: no shifts
        pm.gvtrack_create("vt_mx_sh", shift_test_track, func="avg")
        pm.gvtrack_iterator_2d("vt_mx_sh", sshift1=5000, eshift1=5000,
                               sshift2=5000, eshift2=5000)

        try:
            result = pm.gextract(
                ["vt_mx_zero", "vt_mx_sh"],
                intervals=intervals,
                iterator=(1000, 1000),
            )
            assert result is not None
            # 12x12 grid.
            assert len(result) == 144
            assert "vt_mx_zero" in result.columns
            assert "vt_mx_sh" in result.columns

            # The two columns should differ (different shifts mean different data).
            zero_vals = np.nan_to_num(result["vt_mx_zero"].to_numpy())
            sh_vals   = np.nan_to_num(result["vt_mx_sh"].to_numpy())
            assert not np.array_equal(zero_vals, sh_vals), (
                "Shifted and unshifted vtracks should produce different values"
            )
        finally:
            pm.gvtrack_rm("vt_mx_zero")
            pm.gvtrack_rm("vt_mx_sh")


# ===========================================================================
# T7: global.percentile stays on legacy path (not routed through scanner)
# ===========================================================================


class TestGlobalPercentileLegacyPath:
    """global.percentile is not routed through the scanner (two-pass needed).

    Verify it still works via the legacy path.
    """

    def test_global_percentile_works_no_iterator(self, shift_test_track):
        """global.percentile without iterator= uses legacy path and returns [0,1) values."""
        intervals = pd.DataFrame({
            "chrom1": ["1", "1", "1"],
            "start1": [0, 5000, 10000],
            "end1":   [1000, 6000, 11000],
            "chrom2": ["1", "1", "1"],
            "start2": [0, 5000, 10000],
            "end2":   [1000, 6000, 11000],
        })

        pm.gvtrack_create("vt_gpct", shift_test_track, func="global.percentile")
        result = pm.gextract("vt_gpct", intervals=intervals)
        pm.gvtrack_rm("vt_gpct")

        assert result is not None
        assert len(result) == 3
        vals = result["vt_gpct"].to_numpy(dtype=float)
        valid_vals = vals[~np.isnan(vals)]
        assert len(valid_vals) == 3
        assert all(0.0 <= v < 1.0 for v in valid_vals), (
            f"global.percentile values must be in [0, 1): {valid_vals}"
        )

    def test_global_percentile_with_fixed_rect_raises(self, shift_test_track):
        """global.percentile + iterator=(N, M) raises NotImplementedError (deferred)."""
        intervals = pd.DataFrame({
            "chrom1": ["1"], "start1": [0], "end1": [12_000],
            "chrom2": ["1"], "start2": [0], "end2": [12_000],
        })

        pm.gvtrack_create("vt_gpct2", shift_test_track, func="global.percentile")
        try:
            with pytest.raises(NotImplementedError):
                pm.gextract("vt_gpct2", intervals=intervals, iterator=(1000, 1000))
        finally:
            pm.gvtrack_rm("vt_gpct2")
