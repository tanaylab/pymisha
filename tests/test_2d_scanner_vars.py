"""Tests for the C++ 2D scanner skeleton + 2D vars class.

These tests drive `_pymisha.pm_test_2d_scanner`, a test-only binding that
runs the new PMTrackExpr2DScanner over a 2D track and a 2D intervals set,
producing one aggregated value per input interval. The expected values
come from the existing pure-Python `_gextract_2d_vtrack_agg` path, so
this is a byte-for-byte parity test against the path that gextract uses
today.

Coverage:
- RECTS track, all five agg funcs (area, weighted.sum, min, max, avg).
- POINTS track, all five agg funcs.
- Multi chrom-pair input (forces the per-pair grouping path in the vars
  class).
- Band-filter argument.
- Intervals that don't intersect any object (must yield NaN).
- Empty intervals input.
- Unknown agg func / unknown track / 1D track / mismatched array shapes
  must raise.
"""

from __future__ import annotations

import os
import shutil

import _pymisha
import numpy as np
import pandas as pd
import pytest

from pymisha._quadtree import write_2d_track_file

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name: str) -> None:
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture()
def rects_track():
    """Rects track on chrom-pair 1-1 + 1-2 with known geometry."""
    tname = "test.scan2d_rects"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    # chrom-pair 1-1
    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [
            (100, 200, 300, 400, 5.0),
            (200, 300, 500, 600, 10.0),
            (1000, 1000, 2000, 2000, 3.0),
        ],
        (0, 0, 500000, 500000),
        is_points=False,
    )
    # chrom-pair 1-2
    write_2d_track_file(
        os.path.join(tdir, "1-2"),
        [
            (50, 60, 150, 160, 7.0),
            (400, 500, 900, 1000, 2.0),
        ],
        (0, 0, 500000, 500000),
        is_points=False,
    )
    _pymisha.pm_dbreload()

    yield tname

    _cleanup_track(tname)


@pytest.fixture()
def points_track():
    """Points track on chrom-pair 1-1 with known positions."""
    tname = "test.scan2d_points"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=points\ndimensions=2\n")

    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [
            (100, 200, 5.0),
            (150, 250, 10.0),
            (5000, 5000, 3.0),
        ],
        (0, 0, 500000, 500000),
        is_points=True,
    )
    _pymisha.pm_dbreload()

    yield tname

    _cleanup_track(tname)


def _intervals_to_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a 2D intervals DataFrame to the chromid-keyed dict the
    scanner binding expects.

    `df` has chrom1/chrom2 as chromosome *names* (str); the binding takes
    them as chromids (int) per the iterator's contract.
    """
    from pymisha.intervals import _chrom_id_lookup, _chrom_id_map

    cmap = _chrom_id_map()  # name -> id
    return {
        "chrom1": np.array([_chrom_id_lookup(cmap, str(c)) for c in df["chrom1"]], dtype=np.int32),
        "start1": df["start1"].to_numpy(dtype=np.int64),
        "end1":   df["end1"].to_numpy(dtype=np.int64),
        "chrom2": np.array([_chrom_id_lookup(cmap, str(c)) for c in df["chrom2"]], dtype=np.int32),
        "start2": df["start2"].to_numpy(dtype=np.int64),
        "end2":   df["end2"].to_numpy(dtype=np.int64),
    }


def _ref_values(track: str, intervals: pd.DataFrame, func: str,
                band: tuple[int, int] | None = None) -> np.ndarray:
    """Run the pure-Python reference path and return its per-interval values."""
    from pymisha.extract import _gextract_2d_vtrack_agg

    out = _gextract_2d_vtrack_agg(track, "v", intervals, band, func)
    # out is sorted in input order (intervalID is 0..n-1)
    return out["v"].to_numpy()


@pytest.mark.parametrize("func", ["area", "weighted.sum", "min", "max", "avg"])
def test_rects_track_matches_python_ref(rects_track, func):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1", "1"],
        "start1": [50, 150, 900, 0],
        "end1":   [350, 250, 2500, 50],   # 4th row: empty area
        "chrom2": ["1", "1", "1", "1"],
        "start2": [150, 250, 900, 0],
        "end2":   [450, 350, 2500, 50],
    })
    expected = _ref_values(rects_track, intervals, func)
    actual = _pymisha.pm_test_2d_scanner(rects_track, _intervals_to_dict(intervals), func, None)
    assert actual.dtype == np.float64
    assert actual.shape == (len(intervals),)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=0, atol=0)


@pytest.mark.parametrize("func", ["area", "weighted.sum", "min", "max", "avg"])
def test_points_track_matches_python_ref(points_track, func):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [50, 100, 4000],
        "end1":   [200, 200, 6000],
        "chrom2": ["1", "1", "1"],
        "start2": [150, 300, 4000],
        "end2":   [300, 400, 6000],
    })
    expected = _ref_values(points_track, intervals, func)
    actual = _pymisha.pm_test_2d_scanner(points_track, _intervals_to_dict(intervals), func, None)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=0, atol=0)


def test_multi_chrom_pair(rects_track):
    """Force the per-pair grouping path: intervals on both 1-1 and 1-2."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [50, 100, 350],
        "end1":   [350, 200, 950],
        "chrom2": ["1", "2", "2"],
        "start2": [150, 50, 450],
        "end2":   [450, 200, 1050],
    })
    expected = _ref_values(rects_track, intervals, "avg")
    actual = _pymisha.pm_test_2d_scanner(rects_track, _intervals_to_dict(intervals), "avg", None)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=0, atol=0)


def test_band_filter(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [50],
        "end1":   [350],
        "chrom2": ["1"],
        "start2": [150],
        "end2":   [450],
    })
    expected = _ref_values(rects_track, intervals, "area", band=(-200, 200))
    actual = _pymisha.pm_test_2d_scanner(rects_track,
                                         _intervals_to_dict(intervals),
                                         "area", (-200, 200))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=0, atol=0)


def test_no_overlap_yields_nan(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [400000],
        "end1":   [400100],
        "chrom2": ["1"],
        "start2": [400000],
        "end2":   [400100],
    })
    actual = _pymisha.pm_test_2d_scanner(rects_track, _intervals_to_dict(intervals), "avg", None)
    assert np.isnan(actual[0])


def test_empty_intervals(rects_track):
    intervals = {
        "chrom1": np.array([], dtype=np.int32),
        "start1": np.array([], dtype=np.int64),
        "end1":   np.array([], dtype=np.int64),
        "chrom2": np.array([], dtype=np.int32),
        "start2": np.array([], dtype=np.int64),
        "end2":   np.array([], dtype=np.int64),
    }
    actual = _pymisha.pm_test_2d_scanner(rects_track, intervals, "avg", None)
    assert actual.dtype == np.float64
    assert actual.shape == (0,)


def test_pair_with_no_data_yields_nan(rects_track):
    """chrom-pair (1, X) has no data -> values must be NaN."""
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [1000],
        "chrom2": ["X"],
        "start2": [0],
        "end2":   [1000],
    })
    actual = _pymisha.pm_test_2d_scanner(rects_track, _intervals_to_dict(intervals), "avg", None)
    assert np.isnan(actual[0])


def test_unknown_track_raises():
    intervals = {
        "chrom1": np.array([0], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([100], dtype=np.int64),
        "chrom2": np.array([0], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([100], dtype=np.int64),
    }
    with pytest.raises((ValueError, RuntimeError)):
        _pymisha.pm_test_2d_scanner("does.not.exist", intervals, "avg", None)


def test_unknown_func_raises(rects_track):
    intervals = {
        "chrom1": np.array([0], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([100], dtype=np.int64),
        "chrom2": np.array([0], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([100], dtype=np.int64),
    }
    with pytest.raises((ValueError, RuntimeError)):
        _pymisha.pm_test_2d_scanner(rects_track, intervals, "no_such_func", None)


def test_one_d_track_raises():
    """Passing a 1D track must fail - scanner is 2D-only."""
    intervals = {
        "chrom1": np.array([0], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),
        "end1":   np.array([100], dtype=np.int64),
        "chrom2": np.array([0], dtype=np.int32),
        "start2": np.array([0], dtype=np.int64),
        "end2":   np.array([100], dtype=np.int64),
    }
    # `dense_track` is a 1D track created by conftest.
    with pytest.raises((ValueError, RuntimeError)):
        _pymisha.pm_test_2d_scanner("dense_track", intervals, "avg", None)


def test_mismatched_array_lengths_raises(rects_track):
    intervals = {
        "chrom1": np.array([0, 0], dtype=np.int32),
        "start1": np.array([0], dtype=np.int64),     # wrong length
        "end1":   np.array([100, 200], dtype=np.int64),
        "chrom2": np.array([0, 0], dtype=np.int32),
        "start2": np.array([0, 0], dtype=np.int64),
        "end2":   np.array([100, 200], dtype=np.int64),
    }
    with pytest.raises((ValueError, RuntimeError)):
        _pymisha.pm_test_2d_scanner(rects_track, intervals, "avg", None)
