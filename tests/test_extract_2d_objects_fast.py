"""Tests for the C++ pm_extract_2d_objects fast path.

Asserts byte-for-byte parity between:
- the fast path: pymisha.extract._gextract_2d_vtrack_objects (calls _pymisha.pm_extract_2d_objects)
- the renamed slow path: pymisha.extract._gextract_2d_vtrack_objects_python (the prior Python loop)

For exists / size / first / last: strict DataFrame equality.
For sample: assert the chosen value belongs to the intersecting object set,
since cross-language RNG alignment is not attempted.
"""

from __future__ import annotations

import os
import shutil

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm  # noqa: F401
from pymisha._quadtree import write_2d_track_file
from pymisha.extract import (
    _gextract_2d_vtrack_objects,
    _gextract_2d_vtrack_objects_python,
)

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
    tname = "test.fast2dobj_rects"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

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
    tname = "test.fast2dobj_points"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=points\ndimensions=2\n")

    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [(100, 200, 5.0), (150, 250, 10.0), (5000, 5000, 3.0)],
        (0, 0, 500000, 500000),
        is_points=True,
    )
    _pymisha.pm_dbreload()
    yield tname
    _cleanup_track(tname)


def _parity(track: str, intervals: pd.DataFrame, func: str, band=None) -> None:
    """Strict DataFrame equality for non-random funcs."""
    fast = _gextract_2d_vtrack_objects(track, "v", intervals, band, func)
    slow = _gextract_2d_vtrack_objects_python(track, "v", intervals, band, func)
    pd.testing.assert_frame_equal(fast, slow)


@pytest.mark.parametrize("func", ["exists", "size", "first", "last"])
def test_rects_basic(rects_track, func):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [50, 150, 900],
        "end1":   [350, 250, 2500],
        "chrom2": ["1", "1", "1"],
        "start2": [150, 250, 900],
        "end2":   [450, 350, 2500],
    })
    _parity(rects_track, intervals, func)


@pytest.mark.parametrize("func", ["exists", "size", "first", "last"])
def test_points_basic(points_track, func):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"],
        "start1": [50, 4000],
        "end1":   [200, 6000],
        "chrom2": ["1", "1"],
        "start2": [150, 4000],
        "end2":   [300, 6000],
    })
    _parity(points_track, intervals, func)


@pytest.mark.parametrize("func", ["exists", "size", "first", "last"])
def test_rects_multi_pair(rects_track, func):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [50, 100, 350],
        "end1":   [350, 200, 950],
        "chrom2": ["1", "2", "2"],
        "start2": [150, 50, 450],
        "end2":   [450, 200, 1050],
    })
    _parity(rects_track, intervals, func)


@pytest.mark.parametrize("func", ["exists", "size", "first", "last"])
def test_rects_band(rects_track, func):
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [50],
        "end1":   [350],
        "chrom2": ["1"],
        "start2": [150],
        "end2":   [450],
    })
    _parity(rects_track, intervals, func, band=(-200, 200))


def test_exists_no_overlap_is_zero(rects_track):
    """exists/size default to 0 when no objects intersect (not NaN)."""
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [400000],
        "end1":   [400100],
        "chrom2": ["1"],
        "start2": [400000],
        "end2":   [400100],
    })
    out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "exists")
    assert out["v"].iloc[0] == 0.0
    out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "size")
    assert out["v"].iloc[0] == 0.0


def test_first_no_overlap_is_nan(rects_track):
    """first/last/sample default to NaN when no objects intersect."""
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [400000],
        "end1":   [400100],
        "chrom2": ["1"],
        "start2": [400000],
        "end2":   [400100],
    })
    out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "first")
    assert np.isnan(out["v"].iloc[0])


def test_missing_chrom_pair_yields_defaults(rects_track):
    """chrom-pair (1, X) is missing - exists/size = 0, first/last = NaN."""
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [1000],
        "chrom2": ["X"],
        "start2": [0],
        "end2":   [1000],
    })
    for func, expected in [("exists", 0.0), ("size", 0.0)]:
        out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, func)
        assert out["v"].iloc[0] == expected
    for func in ["first", "last"]:
        out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, func)
        assert np.isnan(out["v"].iloc[0])


def test_empty_intervals(rects_track):
    intervals = pd.DataFrame({
        "chrom1": pd.Series([], dtype=str),
        "start1": pd.Series([], dtype="int64"),
        "end1":   pd.Series([], dtype="int64"),
        "chrom2": pd.Series([], dtype=str),
        "start2": pd.Series([], dtype="int64"),
        "end2":   pd.Series([], dtype="int64"),
    })
    out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "exists")
    assert len(out) == 0
    assert list(out.columns) == [
        "chrom1", "start1", "end1", "chrom2", "start2", "end2", "v", "intervalID"
    ]


def test_sample_picks_intersecting_value(rects_track):
    """For 'sample', assert the returned value is one of the intersecting
    objects' values, not strict parity with the Python oracle (Python's
    random.choice uses global state)."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"],
        "start1": [50, 150],
        "end1":   [350, 250],
        "chrom2": ["1", "1"],
        "start2": [150, 250],
        "end2":   [450, 350],
    })
    out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "sample")
    # First interval intersects rects with values 5.0 and 10.0.
    assert out["v"].iloc[0] in {5.0, 10.0}
    # Second interval intersects rects with values 5.0 and 10.0 (same).
    assert out["v"].iloc[1] in {5.0, 10.0}


def test_unknown_func_raises(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [100],
        "chrom2": ["1"], "start2": [0], "end2": [100],
    })
    with pytest.raises((ValueError, RuntimeError)):
        _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "median")


def test_output_intervalid_is_input_order(rects_track):
    """intervalID is 0..n-1 in the input order (not sorted)."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [1500, 50, 200],
        "end1":   [2200, 250, 350],
        "chrom2": ["1", "1", "1"],
        "start2": [1500, 150, 300],
        "end2":   [2200, 350, 450],
    })
    out = _gextract_2d_vtrack_objects(rects_track, "v", intervals, None, "exists")
    np.testing.assert_array_equal(out["intervalID"].to_numpy(), np.arange(3))
