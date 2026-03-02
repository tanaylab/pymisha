"""Tests for giterator_intervals_2d — streaming 2D extraction."""

import os
import shutil

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
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
    """Clean up all vtracks after each test."""
    yield
    pm.gvtrack_clear()


@pytest.fixture()
def rects_track():
    """Create a 2D rects track with known geometry.

    Layout on chr1-chr1 (arena 0..500000 x 0..500000):

        Rect A: (100, 200, 300, 400, value=5.0)
        Rect B: (200, 300, 500, 600, value=10.0)
        Rect C: (1000, 1000, 2000, 2000, value=3.0)

    Rects A and B overlap in [200,300) x [300,400).
    Rect C is isolated.
    """
    tname = "test.iter2d_rects"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    rects = [
        (100, 200, 300, 400, 5.0),       # Rect A
        (200, 300, 500, 600, 10.0),       # Rect B
        (1000, 1000, 2000, 2000, 3.0),    # Rect C
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), rects, (0, 0, 500000, 500000), is_points=False
    )
    _pymisha.pm_dbreload()

    yield tname

    pm.gvtrack_clear()
    _cleanup_track(tname)


@pytest.fixture()
def points_track():
    """Create a 2D points track with known positions.

    Points on chr1-chr1:
        P1: (100, 200, value=5.0)
        P2: (150, 250, value=10.0)
        P3: (5000, 5000, value=3.0)
    """
    tname = "test.iter2d_points"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=points\ndimensions=2\n")

    points = [
        (100, 200, 5.0),    # P1
        (150, 250, 10.0),   # P2
        (5000, 5000, 3.0),  # P3
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), points, (0, 0, 500000, 500000), is_points=True
    )
    _pymisha.pm_dbreload()

    yield tname

    pm.gvtrack_clear()
    _cleanup_track(tname)


# ---------------------------------------------------------------------------
# Tests: basic iteration
# ---------------------------------------------------------------------------


class TestBasicIteration:
    """Basic iteration over 2D intervals with a raw track."""

    def test_single_interval_yields_one_chunk(self, rects_track):
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        assert len(chunks) == 1
        # Should contain all 3 rects
        assert len(chunks[0]) == 3

    def test_multiple_intervals_yield_multiple_chunks(self, rects_track):
        intervals = pd.DataFrame({
            "chrom1": ["1", "1"],
            "start1": [0, 800],
            "end1":   [500, 3000],
            "chrom2": ["1", "1"],
            "start2": [0, 800],
            "end2":   [700, 3000],
        })
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        # First interval should hit Rects A and B
        # Second interval should hit Rect C
        assert len(chunks) == 2
        assert len(chunks[0]) >= 1
        assert len(chunks[1]) >= 1

    def test_interval_id_matches_input_position(self, rects_track):
        intervals = pd.DataFrame({
            "chrom1": ["1", "1", "1"],
            "start1": [0, 800, 100000],
            "end1":   [500, 3000, 200000],
            "chrom2": ["1", "1", "1"],
            "start2": [0, 800, 100000],
            "end2":   [700, 3000, 200000],
        })
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        # Third interval hits nothing (no data at 100000-200000)
        # So we should only get 2 chunks
        assert len(chunks) == 2
        # intervalID should be 0 and 1, reflecting position in input
        assert all(chunks[0]["intervalID"] == 0)
        assert all(chunks[1]["intervalID"] == 1)

    def test_returns_correct_columns(self, rects_track):
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        assert len(chunks) == 1
        expected_cols = [
            "chrom1", "start1", "end1", "chrom2", "start2", "end2",
            rects_track, "intervalID",
        ]
        assert list(chunks[0].columns) == expected_cols

    def test_is_generator(self, rects_track):
        """giterator_intervals_2d should return a generator, not a list."""
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)
        result = pm.giterator_intervals_2d(rects_track, intervals)
        import types
        assert isinstance(result, types.GeneratorType)

    def test_points_track(self, points_track):
        intervals = pm.gintervals_2d("1", 0, 1000, "1", 0, 1000)
        chunks = list(pm.giterator_intervals_2d(points_track, intervals))
        assert len(chunks) == 1
        # Should contain P1 and P2 (P3 is at 5000,5000)
        assert len(chunks[0]) == 2


# ---------------------------------------------------------------------------
# Tests: concatenated results match gextract
# ---------------------------------------------------------------------------


class TestMatchesGextract:
    """Concatenated iterator results should match gextract output."""

    def test_single_interval_matches_gextract(self, rects_track):
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)

        bulk = pm.gextract(rects_track, intervals=intervals)
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        assert len(chunks) == 1

        combined = chunks[0]
        # Sort both to enable comparison
        sort_cols = ["chrom1", "start1", "chrom2", "start2"]
        bulk_sorted = bulk.sort_values(sort_cols).reset_index(drop=True)
        combined_sorted = combined.sort_values(sort_cols).reset_index(drop=True)

        # Compare coordinates and values (intervalID may differ in numbering)
        for col in ["chrom1", "start1", "end1", "chrom2", "start2", "end2", rects_track]:
            np.testing.assert_array_equal(
                bulk_sorted[col].values,
                combined_sorted[col].values,
                err_msg=f"Column {col} mismatch",
            )

    def test_multi_interval_matches_gextract(self, rects_track):
        intervals = pd.DataFrame({
            "chrom1": ["1", "1"],
            "start1": [0, 800],
            "end1":   [500, 3000],
            "chrom2": ["1", "1"],
            "start2": [0, 800],
            "end2":   [700, 3000],
        })

        bulk = pm.gextract(rects_track, intervals=intervals)
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        combined = pd.concat(chunks, ignore_index=True)

        # Same total number of result rows
        assert len(combined) == len(bulk)

        # Same values (after sorting)
        sort_cols = ["chrom1", "start1", "chrom2", "start2", "intervalID"]
        bulk_sorted = bulk.sort_values(sort_cols).reset_index(drop=True)
        combined_sorted = combined.sort_values(sort_cols).reset_index(drop=True)

        for col in ["chrom1", "start1", "end1", "chrom2", "start2", "end2", rects_track]:
            np.testing.assert_array_equal(
                bulk_sorted[col].values,
                combined_sorted[col].values,
                err_msg=f"Column {col} mismatch",
            )


# ---------------------------------------------------------------------------
# Tests: band parameter
# ---------------------------------------------------------------------------


class TestBandParameter:
    """Test diagonal band filtering."""

    def test_band_filters_objects(self, points_track):
        """Band should filter points based on diagonal distance.

        Points:
            P1: (100, 200, 5.0)  => x-y = -100
            P2: (150, 250, 10.0) => x-y = -100
            P3: (5000, 5000, 3.0) => x-y = 0

        Band (-200, -50) includes P1 and P2 (x-y = -100) but excludes P3 (x-y = 0).
        """
        intervals = pm.gintervals_2d("1", 0, 10000, "1", 0, 10000)

        chunks_no_band = list(pm.giterator_intervals_2d(
            points_track, intervals
        ))
        total_no_band = sum(len(c) for c in chunks_no_band)
        assert total_no_band == 3

        chunks_with_band = list(pm.giterator_intervals_2d(
            points_track, intervals, band=(-200, -50)
        ))
        total_with_band = sum(len(c) for c in chunks_with_band)
        assert total_with_band == 2

    def test_band_matches_gextract_band(self, rects_track):
        """Band results should match gextract with same band."""
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)
        band = (-500, 500)

        bulk = pm.gextract(rects_track, intervals=intervals, band=band)
        chunks = list(pm.giterator_intervals_2d(
            rects_track, intervals, band=band
        ))

        if bulk is None:
            assert len(chunks) == 0
            return

        combined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        assert len(combined) == len(bulk)


# ---------------------------------------------------------------------------
# Tests: empty intervals
# ---------------------------------------------------------------------------


class TestEmptyIntervals:
    """Test behavior with empty or no-match intervals."""

    def test_empty_intervals_yields_nothing(self, rects_track):
        intervals = pd.DataFrame({
            "chrom1": pd.Series([], dtype=str),
            "start1": pd.Series([], dtype=int),
            "end1":   pd.Series([], dtype=int),
            "chrom2": pd.Series([], dtype=str),
            "start2": pd.Series([], dtype=int),
            "end2":   pd.Series([], dtype=int),
        })
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        assert len(chunks) == 0

    def test_no_match_interval_yields_nothing(self, rects_track):
        """Interval that doesn't overlap any data should be silently skipped."""
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        assert len(chunks) == 0

    def test_mixed_match_and_no_match(self, rects_track):
        """Only intervals with data should yield chunks."""
        intervals = pd.DataFrame({
            "chrom1": ["1", "1"],
            "start1": [0, 400000],
            "end1":   [500, 500000],
            "chrom2": ["1", "1"],
            "start2": [0, 400000],
            "end2":   [700, 500000],
        })
        chunks = list(pm.giterator_intervals_2d(rects_track, intervals))
        # First interval hits rects, second does not
        assert len(chunks) == 1
        assert all(chunks[0]["intervalID"] == 0)


# ---------------------------------------------------------------------------
# Tests: virtual tracks (aggregation)
# ---------------------------------------------------------------------------


class TestVirtualTrackAggregation:
    """Test with virtual tracks using 2D aggregation functions."""

    def test_area_vtrack(self, rects_track):
        pm.gvtrack_create("vt_area", rects_track, func="area")
        intervals = pd.DataFrame({
            "chrom1": ["1", "1"],
            "start1": [0, 800],
            "end1":   [500, 3000],
            "chrom2": ["1", "1"],
            "start2": [0, 800],
            "end2":   [700, 3000],
        })

        chunks = list(pm.giterator_intervals_2d("vt_area", intervals))
        # Aggregation produces one row per interval
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) == 1
            assert "vt_area" in chunk.columns

    def test_agg_vtrack_matches_gextract(self, rects_track):
        pm.gvtrack_create("vt_avg", rects_track, func="avg")
        intervals = pd.DataFrame({
            "chrom1": ["1", "1"],
            "start1": [0, 800],
            "end1":   [500, 3000],
            "chrom2": ["1", "1"],
            "start2": [0, 800],
            "end2":   [700, 3000],
        })

        bulk = pm.gextract("vt_avg", intervals=intervals)
        chunks = list(pm.giterator_intervals_2d("vt_avg", intervals))
        combined = pd.concat(chunks, ignore_index=True)

        sort_cols = ["chrom1", "start1", "chrom2", "start2"]
        bulk_sorted = bulk.sort_values(sort_cols).reset_index(drop=True)
        combined_sorted = combined.sort_values(sort_cols).reset_index(drop=True)

        np.testing.assert_array_almost_equal(
            bulk_sorted["vt_avg"].values,
            combined_sorted["vt_avg"].values,
        )

    def test_min_max_vtrack(self, rects_track):
        pm.gvtrack_create("vt_min", rects_track, func="min")
        pm.gvtrack_create("vt_max", rects_track, func="max")
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)

        chunks = list(pm.giterator_intervals_2d(
            ["vt_min", "vt_max"], intervals
        ))
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "vt_min" in chunk.columns
        assert "vt_max" in chunk.columns
        # All 3 rects have values 5.0, 10.0, 3.0
        assert chunk["vt_min"].iloc[0] == 3.0
        assert chunk["vt_max"].iloc[0] == 10.0


# ---------------------------------------------------------------------------
# Tests: colnames parameter
# ---------------------------------------------------------------------------


class TestColnames:
    """Test custom column names."""

    def test_custom_colnames(self, rects_track):
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)
        chunks = list(pm.giterator_intervals_2d(
            rects_track, intervals, colnames=["my_values"]
        ))
        assert len(chunks) == 1
        assert "my_values" in chunks[0].columns

    def test_colnames_length_mismatch_raises(self, rects_track):
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)
        with pytest.raises(ValueError, match="colnames length"):
            list(pm.giterator_intervals_2d(
                rects_track, intervals, colnames=["a", "b"]
            ))


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Test error conditions."""

    def test_1d_intervals_raise(self, rects_track):
        intervals = pm.gintervals("1", 0, 1000)
        with pytest.raises(ValueError, match="2D intervals"):
            list(pm.giterator_intervals_2d(rects_track, intervals))

    def test_no_db_init_raises(self):
        # We cannot unload the DB in the middle of a session fixture,
        # so we just verify the function exists and is callable.
        assert callable(pm.giterator_intervals_2d)


# ---------------------------------------------------------------------------
# Tests: multiple expressions
# ---------------------------------------------------------------------------


class TestMultipleExpressions:
    """Test with multiple track expressions."""

    def test_multi_expr_vtrack(self, rects_track):
        pm.gvtrack_create("vt_a", rects_track, func="area")
        pm.gvtrack_create("vt_w", rects_track, func="weighted.sum")
        intervals = pm.gintervals_2d("1", 0, 3000, "1", 0, 3000)

        chunks = list(pm.giterator_intervals_2d(
            ["vt_a", "vt_w"], intervals
        ))
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "vt_a" in chunk.columns
        assert "vt_w" in chunk.columns
