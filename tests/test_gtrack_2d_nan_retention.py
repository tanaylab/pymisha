"""Phase D no-NFS regression: 2D tracks must retain NaN-valued rectangles.

R's `gtrack.lookup` with `force.binning=FALSE` and equivalent pymisha paths
build 2D RECTS tracks containing both finite and NaN values; the NaN rects
must round-trip through `gtrack_2d_create` + `gextract`, and 2D virtual-track
stat aggregations (avg / min / max / weighted.sum / area) must count them as
R misha 5.12.1 does: a NaN rect's area counts and makes the weighted sum and
avg NaN, and min / max take the values after the last NaN met. The expected
values are R's, on the very file gtrack_2d_create writes here (R writes the
same bytes for these rects).
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import _pymisha
import pymisha as pm


from _dbpath import TESTDB_ROOT
TRACK_DIR = os.path.join(str(TESTDB_ROOT), "tracks")


def _track_dir(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name: str) -> None:
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _clear_vtracks():
    yield
    pm.gvtrack_clear()


def _build_mixed_track(name: str) -> None:
    """Create a 2D RECTS track on chrom 1 with 3 finite + 2 NaN rects.

    Rects are spaced at [100..550] on a chrom of size 500_000, so all five
    sit well inside the chrom-1 bounding box (no arena-clip surprises) and
    in different leaves of any reasonable quadtree depth.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"] * 5,
        "start1": [100, 200, 300, 400, 500],
        "end1":   [150, 250, 350, 450, 550],
        "chrom2": ["1"] * 5,
        "start2": [100, 200, 300, 400, 500],
        "end2":   [150, 250, 350, 450, 550],
    })
    # finite, NaN, finite, NaN, finite
    values = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
    pm.gtrack_2d_create(name, "Phase D mixed NaN/finite test", intervals, values)


def test_2d_track_with_nan_rects_round_trips_all_rows():
    """All 5 rects (including 2 with NaN values) must round-trip via gextract."""
    name = "test.test_2d_nan_roundtrip"
    _cleanup_track(name)
    try:
        _build_mixed_track(name)
        scope = pm.gintervals_2d([1], 0, 1000, [1], 0, 1000)
        result = pm.gextract(name, scope)
        assert result is not None
        assert len(result) == 5, (
            f"expected 5 rects (3 finite + 2 NaN), got {len(result)}"
        )
        finite_vals = result[result[name].notna()][name].to_numpy()
        nan_count = int(result[name].isna().sum())
        np.testing.assert_array_equal(sorted(finite_vals), [10.0, 30.0, 50.0])
        assert nan_count == 2, f"expected 2 NaN rects, got {nan_count}"
    finally:
        _cleanup_track(name)


def test_2d_vtrack_avg_is_nan_over_a_nan_rect():
    """avg over a window covering 3 finite + 2 NaN rects is NaN, as in R."""
    name = "test.test_2d_nan_avg"
    _cleanup_track(name)
    try:
        _build_mixed_track(name)
        pm.gvtrack_create("v_avg", name, "avg")
        scope = pm.gintervals_2d([1], 0, 1000, [1], 0, 1000)
        result = pm.gextract("v_avg", scope, iterator=scope)
        assert result is not None
        assert np.isnan(result["v_avg"].iloc[0])
    finally:
        _cleanup_track(name)


def test_2d_vtrack_min_max_sum_area_with_nan():
    """min / max / weighted.sum / area over a NaN-bearing window, as R gives them.

    The rects are 10, NaN, 30, NaN, 50, each 50x50 and read in that order: the
    area counts all five, the weighted sum is NaN, and min and max are both
    50, the only value after the last NaN.
    """
    name = "test.test_2d_nan_minmaxsum"
    _cleanup_track(name)
    try:
        _build_mixed_track(name)
        for func, expected in [
            ("min", 50.0),
            ("max", 50.0),
            ("weighted.sum", np.nan),
            ("area", 5 * 2500.0),
        ]:
            pm.gvtrack_clear()
            pm.gvtrack_create(f"v_{func}", name, func)
            scope = pm.gintervals_2d([1], 0, 1000, [1], 0, 1000)
            result = pm.gextract(f"v_{func}", scope, iterator=scope)
            assert result is not None, f"{func}: gextract returned None"
            np.testing.assert_array_equal(
                float(result[f"v_{func}"].iloc[0]), expected, err_msg=f"func={func}"
            )
    finally:
        _cleanup_track(name)


def test_2d_vtrack_window_with_only_nan_rects_returns_nan():
    """A window covering only a NaN rect gives NaN, and the rect's area, as R does."""
    name = "test.test_2d_nan_only"
    _cleanup_track(name)
    try:
        _build_mixed_track(name)
        pm.gvtrack_create("v_avg", name, "avg")
        pm.gvtrack_create("v_area", name, "area")
        # Window [180,260) x [180,260) covers only the (200,250) rect, which is NaN.
        scope = pm.gintervals_2d([1], 180, 260, [1], 180, 260)
        result = pm.gextract(["v_avg", "v_area"], scope, iterator=scope)
        assert len(result) == 1
        assert pd.isna(result["v_avg"].iloc[0])
        assert result["v_area"].iloc[0] == 2500
    finally:
        _cleanup_track(name)
