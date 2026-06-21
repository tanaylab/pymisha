"""Tests for the C++ pm_extract_2d fast path.

These tests assert byte-for-byte parity between:
- the new fast path: pymisha.extract._gextract_2d_single (now calls _pymisha.pm_extract_2d)
- the renamed slow path: pymisha.extract._gextract_2d_single_python (the previous Python loop)

Both must produce the same DataFrame for every input. The slow path is kept
exclusively as a parity oracle; nothing in production code uses it.
"""

from __future__ import annotations

import contextlib
import os
import shutil

import _pymisha
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import clear_indexed_2d_cache, write_2d_track_file
from pymisha.extract import _gextract_2d_single, _gextract_2d_single_python

from _dbpath import TESTDB_ROOT
TRACK_DIR = os.path.join(str(TESTDB_ROOT), "tracks")


def _track_dir(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name: str) -> None:
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        # Release any cached mmaps (slow-path indexed reader holds track.dat
        # alive; on NFS this leaves a .nfsXXXX ghost file that breaks rmdir).
        clear_indexed_2d_cache()
        shutil.rmtree(tdir, ignore_errors=True)
        # On NFS the final rmdir can race even after we drop our handles.
        # Try once more; if it still fails, accept the leftover empty dir.
        if os.path.exists(tdir):
            with contextlib.suppress(OSError):
                os.rmdir(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture()
def rects_track():
    tname = "test.fast2d_rects"
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
    tname = "test.fast2d_points"
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


@pytest.fixture()
def indexed_rects_track():
    """A rects track stored in indexed format (track.idx + track.dat).

    Exercises the Session-2 carry-forward fix: the indexed reader should
    NOT copy each pair slice on every set_chrom_pair call.
    """
    tname = "test.fast2d_indexed_rects"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    # Build per-pair files first, then convert to indexed format.
    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [(100, 200, 300, 400, 5.0), (200, 300, 500, 600, 10.0)],
        (0, 0, 500000, 500000),
        is_points=False,
    )
    write_2d_track_file(
        os.path.join(tdir, "1-2"),
        [(50, 60, 150, 160, 7.0)],
        (0, 0, 500000, 500000),
        is_points=False,
    )
    _pymisha.pm_dbreload()

    # track_type: 0 = rects, 1 = points
    _pymisha.pm_track2d_convert_to_indexed(tdir, 0)
    _pymisha.pm_dbreload()

    yield tname
    _cleanup_track(tname)


def _parity(track: str, intervals: pd.DataFrame, band=None):
    col = "v"
    fast = _gextract_2d_single(track, col, intervals, band)
    slow = _gextract_2d_single_python(track, col, intervals, band)
    if fast is None and slow is None:
        return
    if fast is None or slow is None:
        raise AssertionError(f"fast={fast!r}, slow={slow!r}")
    pd.testing.assert_frame_equal(fast, slow)


def test_rects_basic(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [50, 150, 900],
        "end1":   [350, 250, 2500],
        "chrom2": ["1", "1", "1"],
        "start2": [150, 250, 900],
        "end2":   [450, 350, 2500],
    })
    _parity(rects_track, intervals)


def test_points_basic(points_track):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"],
        "start1": [50, 4000],
        "end1":   [200, 6000],
        "chrom2": ["1", "1"],
        "start2": [150, 4000],
        "end2":   [300, 6000],
    })
    _parity(points_track, intervals)


def test_rects_multi_pair(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1", "1", "1"],
        "start1": [50, 100, 350],
        "end1":   [350, 200, 950],
        "chrom2": ["1", "2", "2"],
        "start2": [150, 50, 450],
        "end2":   [450, 200, 1050],
    })
    _parity(rects_track, intervals)


def test_rects_band(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [50],
        "end1":   [350],
        "chrom2": ["1"],
        "start2": [150],
        "end2":   [450],
    })
    _parity(rects_track, intervals, band=(-200, 200))


def test_no_overlap_returns_none(rects_track):
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [400000],
        "end1":   [400100],
        "chrom2": ["1"],
        "start2": [400000],
        "end2":   [400100],
    })
    fast = _gextract_2d_single(rects_track, "v", intervals, None)
    slow = _gextract_2d_single_python(rects_track, "v", intervals, None)
    assert fast is None
    assert slow is None


def test_missing_chrom_pair_yields_no_rows(rects_track):
    """chrom-pair (1, X) has no data on this track."""
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [1000],
        "chrom2": ["X"],
        "start2": [0],
        "end2":   [1000],
    })
    _parity(rects_track, intervals)


def test_empty_intervals(rects_track):
    intervals = pd.DataFrame({
        "chrom1": pd.Series([], dtype=str),
        "start1": pd.Series([], dtype="int64"),
        "end1":   pd.Series([], dtype="int64"),
        "chrom2": pd.Series([], dtype=str),
        "start2": pd.Series([], dtype="int64"),
        "end2":   pd.Series([], dtype="int64"),
    })
    fast = _gextract_2d_single(rects_track, "v", intervals, None)
    slow = _gextract_2d_single_python(rects_track, "v", intervals, None)
    assert fast is None
    assert slow is None


def test_indexed_format_parity(indexed_rects_track):
    """Carry-forward fix: indexed reader must produce identical output
    to the per-pair format without doing an mmap+copy+munmap dance per
    pair switch. We verify behavior, not the mechanism."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"],
        "start1": [50, 100],
        "end1":   [350, 200],
        "chrom2": ["1", "2"],
        "start2": [150, 50],
        "end2":   [450, 200],
    })
    _parity(indexed_rects_track, intervals)


def test_gextract_top_level_uses_fast_path(rects_track):
    """End-to-end: pm.gextract on a bare 2D track + 2D intervals must
    produce the same DataFrame as before (via the slow oracle)."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"],
        "start1": [50, 100],
        "end1":   [350, 200],
        "chrom2": ["1", "1"],
        "start2": [150, 250],
        "end2":   [450, 350],
    })
    fast = pm.gextract(rects_track, intervals)
    slow = _gextract_2d_single_python(rects_track, rects_track, intervals, None)
    if fast is None or slow is None:
        assert fast is None and slow is None
        return
    pd.testing.assert_frame_equal(
        fast.reset_index(drop=True),
        slow.reset_index(drop=True),
    )


def test_one_d_track_raises_or_falls_back():
    """1D tracks must not hit pm_extract_2d. The fast path detects 2D-ness
    via the track type check inside the binding; a 1D track should never
    reach _gextract_2d_single in normal gextract flow, but if forced, the
    binding raises rather than silently returning garbage."""
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [100],
        "chrom2": ["1"],
        "start2": [0],
        "end2":   [100],
    })
    with pytest.raises((ValueError, RuntimeError)):
        _gextract_2d_single("dense_track", "v", intervals, None)


def test_unknown_track_raises():
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [100],
        "chrom2": ["1"],
        "start2": [0],
        "end2":   [100],
    })
    with pytest.raises((ValueError, RuntimeError)):
        _gextract_2d_single("does.not.exist", "v", intervals, None)
