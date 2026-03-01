"""Tests for 2D virtual track aggregation functions.

These tests verify that 2D virtual tracks with aggregation functions
(area, weighted.sum, min, max, avg) return ONE ROW PER QUERY INTERVAL
with properly aggregated statistics, as opposed to the current alias-style
behavior that returns one row per intersecting object.

TDD approach: these tests define the expected behavior for features that
are not yet implemented.
"""

import os
import shutil

import _pymisha
import numpy as np
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
def rects_agg_track():
    """Create a 2D rects track with known geometry for precise aggregation tests.

    Layout on chr1-chr1 (arena 0..500000 x 0..500000):

        Rect A: (100, 200, 300, 400, value=5.0)
            width1=200, width2=200, area=40000
        Rect B: (200, 300, 500, 600, value=10.0)
            width1=300, width2=300, area=90000
        Rect C: (1000, 1000, 2000, 2000, value=3.0)
            width1=1000, width2=1000, area=1000000

    Rects A and B overlap in [200,300) x [300,400).
    Rect C is isolated.
    """
    tname = "test.agg_rects"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    rects = [
        (100, 200, 300, 400, 5.0),   # Rect A
        (200, 300, 500, 600, 10.0),   # Rect B
        (1000, 1000, 2000, 2000, 3.0),  # Rect C
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), rects, (0, 0, 500000, 500000), is_points=False
    )
    _pymisha.pm_dbreload()

    yield tname

    pm.gvtrack_clear()
    _cleanup_track(tname)


@pytest.fixture()
def points_agg_track():
    """Create a 2D points track with known positions for aggregation tests.

    Points on chr1-chr1:
        P1: (100, 200, value=5.0)    — area contribution = 1
        P2: (150, 250, value=10.0)   — area contribution = 1
        P3: (5000, 5000, value=3.0)  — isolated
    """
    tname = "test.agg_points"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)

    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=points\ndimensions=2\n")

    points = [
        (100, 200, 5.0),   # P1
        (150, 250, 10.0),  # P2
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
# Helper: compute expected aggregation values in pure Python
# ---------------------------------------------------------------------------


def _intersection_area_rect(qx1, qy1, qx2, qy2, ox1, oy1, ox2, oy2):
    """Intersection area between query rect and object rect."""
    ix1 = max(qx1, ox1)
    iy1 = max(qy1, oy1)
    ix2 = min(qx2, ox2)
    iy2 = min(qy2, oy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _expected_agg_rects(query, rects, func):
    """Compute expected aggregation for one query interval over a list of rects.

    Parameters
    ----------
    query : tuple (x1, y1, x2, y2)
    rects : list of (x1, y1, x2, y2, value)
    func : str — "area", "weighted.sum", "min", "max", "avg"
    """
    total_area = 0
    weighted_sum = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    qx1, qy1, qx2, qy2 = query
    for ox1, oy1, ox2, oy2, val in rects:
        a = _intersection_area_rect(qx1, qy1, qx2, qy2, ox1, oy1, ox2, oy2)
        if a > 0:
            total_area += a
            weighted_sum += val * a
            min_val = min(min_val, val)
            max_val = max(max_val, val)

    if total_area == 0:
        return np.nan

    if func == "area":
        return total_area
    if func == "weighted.sum":
        return weighted_sum
    if func == "min":
        return min_val
    if func == "max":
        return max_val
    if func == "avg":
        return weighted_sum / total_area
    raise ValueError(f"Unknown func: {func}")


def _expected_agg_points(query, points, func):
    """Compute expected aggregation for one query over a list of points.

    For points, each matching point contributes area=1.

    Parameters
    ----------
    query : tuple (x1, y1, x2, y2)
    points : list of (x, y, value)
    func : str
    """
    total_area = 0
    weighted_sum = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    qx1, qy1, qx2, qy2 = query
    for ox, oy, val in points:
        # Point occupies [x, x+1) x [y, y+1)
        if ox >= qx1 and ox < qx2 and oy >= qy1 and oy < qy2:
            total_area += 1
            weighted_sum += val * 1
            min_val = min(min_val, val)
            max_val = max(max_val, val)

    if total_area == 0:
        return np.nan

    if func == "area":
        return total_area
    if func == "weighted.sum":
        return weighted_sum
    if func == "min":
        return min_val
    if func == "max":
        return max_val
    if func == "avg":
        return weighted_sum / total_area
    raise ValueError(f"Unknown func: {func}")


# ---------------------------------------------------------------------------
# Test data constants
# ---------------------------------------------------------------------------

RECTS = [
    (100, 200, 300, 400, 5.0),
    (200, 300, 500, 600, 10.0),
    (1000, 1000, 2000, 2000, 3.0),
]

POINTS = [
    (100, 200, 5.0),
    (150, 250, 10.0),
    (5000, 5000, 3.0),
]


# ===========================================================================
# Tests
# ===========================================================================


class TestAreaBasic:
    """Test the 'area' aggregation function on 2D virtual tracks."""

    def test_area_full_overlap(self, rects_agg_track):
        """Area with a query that fully contains Rect A."""
        pm.gvtrack_create("vt_area", rects_agg_track, func="area")
        # Query fully contains Rect A (100,200,300,400) and partially overlaps Rect B
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_area", intervals)

        assert result is not None
        assert len(result) == 1, "Aggregation should return one row per query interval"

        # Expected: intersection with Rect A = (100,200,300,400) -> 200*200 = 40000
        #           intersection with Rect B = (200,300,500,600) clipped to (200,300,500,600)
        #               but query is (0,0,600,700) => clipped to (200,300,500,600) => 300*300 = 90000
        expected = _expected_agg_rects((0, 0, 600, 700), RECTS, "area")
        assert result["vt_area"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_area_partial_overlap(self, rects_agg_track):
        """Area with a query that partially overlaps Rect A."""
        pm.gvtrack_create("vt_area_p", rects_agg_track, func="area")
        # Query overlaps Rect A in [150,300) x [250,400) = 150*150 = 22500
        # Query also overlaps Rect B in [200,300) x [300,400) = 100*100 = 10000
        intervals = pm.gintervals_2d("1", 150, 300, "1", 250, 400)
        result = pm.gextract("vt_area_p", intervals)

        assert result is not None
        assert len(result) == 1

        expected = _expected_agg_rects((150, 250, 300, 400), RECTS, "area")
        assert result["vt_area_p"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_area_multiple_query_intervals(self, rects_agg_track):
        """Area aggregation returns one value per query interval."""
        pm.gvtrack_create("vt_area_m", rects_agg_track, func="area")
        # Two query intervals: one hits A+B, one hits C
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 900],
            ends1=[600, 2100],
            chroms2=["1", "1"],
            starts2=[0, 900],
            ends2=[700, 2100],
        )
        result = pm.gextract("vt_area_m", intervals)

        assert result is not None
        assert len(result) == 2

        exp0 = _expected_agg_rects((0, 0, 600, 700), RECTS, "area")
        exp1 = _expected_agg_rects((900, 900, 2100, 2100), RECTS, "area")

        # Sort by start1 to match expected ordering
        result = result.sort_values("start1").reset_index(drop=True)
        assert result["vt_area_m"].iloc[0] == pytest.approx(exp0, rel=1e-5)
        assert result["vt_area_m"].iloc[1] == pytest.approx(exp1, rel=1e-5)


class TestWeightedSumBasic:
    """Test the 'weighted.sum' aggregation function."""

    def test_weighted_sum_single_object(self, rects_agg_track):
        """Weighted sum with one intersecting object."""
        pm.gvtrack_create("vt_ws", rects_agg_track, func="weighted.sum")
        # Query hits only Rect C: (1000,1000,2000,2000) val=3.0
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_ws", intervals)

        assert result is not None
        assert len(result) == 1

        expected = _expected_agg_rects((900, 900, 2100, 2100), RECTS, "weighted.sum")
        assert result["vt_ws"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_weighted_sum_two_objects(self, rects_agg_track):
        """Weighted sum with two overlapping objects."""
        pm.gvtrack_create("vt_ws2", rects_agg_track, func="weighted.sum")
        # Query hits Rect A and Rect B
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_ws2", intervals)

        assert result is not None
        assert len(result) == 1

        expected = _expected_agg_rects((0, 0, 600, 700), RECTS, "weighted.sum")
        assert result["vt_ws2"].iloc[0] == pytest.approx(expected, rel=1e-5)


class TestMinBasic:
    """Test the 'min' aggregation function."""

    def test_min_single_object(self, rects_agg_track):
        """Min returns the value of the single matching object."""
        pm.gvtrack_create("vt_min", rects_agg_track, func="min")
        # Query hits only Rect C: val=3.0
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_min", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_min"].iloc[0] == pytest.approx(3.0, rel=1e-5)

    def test_min_multiple_objects(self, rects_agg_track):
        """Min returns the smallest value among matching objects."""
        pm.gvtrack_create("vt_min2", rects_agg_track, func="min")
        # Query hits Rect A (val=5) and Rect B (val=10)
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_min2", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_min2"].iloc[0] == pytest.approx(5.0, rel=1e-5)

    def test_min_all_three_objects(self, rects_agg_track):
        """Min over all three objects returns the global minimum."""
        pm.gvtrack_create("vt_min3", rects_agg_track, func="min")
        # Query hits all three rects
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_min3", intervals)

        assert result is not None
        assert len(result) == 1
        # min(5, 10, 3) = 3
        assert result["vt_min3"].iloc[0] == pytest.approx(3.0, rel=1e-5)


class TestMaxBasic:
    """Test the 'max' aggregation function."""

    def test_max_single_object(self, rects_agg_track):
        """Max returns the value of the single matching object."""
        pm.gvtrack_create("vt_max", rects_agg_track, func="max")
        # Query hits only Rect C: val=3.0
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_max", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_max"].iloc[0] == pytest.approx(3.0, rel=1e-5)

    def test_max_multiple_objects(self, rects_agg_track):
        """Max returns the largest value among matching objects."""
        pm.gvtrack_create("vt_max2", rects_agg_track, func="max")
        # Query hits Rect A (val=5) and Rect B (val=10)
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_max2", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_max2"].iloc[0] == pytest.approx(10.0, rel=1e-5)

    def test_max_all_three_objects(self, rects_agg_track):
        """Max over all three objects returns the global maximum."""
        pm.gvtrack_create("vt_max3", rects_agg_track, func="max")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_max3", intervals)

        assert result is not None
        assert len(result) == 1
        # max(5, 10, 3) = 10
        assert result["vt_max3"].iloc[0] == pytest.approx(10.0, rel=1e-5)


class TestAvgBasic:
    """Test the 'avg' aggregation function (weighted.sum / area)."""

    def test_avg_single_object(self, rects_agg_track):
        """Avg of a single object equals its value (uniform weight)."""
        pm.gvtrack_create("vt_avg", rects_agg_track, func="avg")
        # Query fully contains Rect C: val=3.0, area=1000000
        # avg = 3.0 * 1000000 / 1000000 = 3.0
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_avg", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_avg"].iloc[0] == pytest.approx(3.0, rel=1e-5)

    def test_avg_two_objects_weighted(self, rects_agg_track):
        """Avg with two objects is the area-weighted average."""
        pm.gvtrack_create("vt_avg2", rects_agg_track, func="avg")
        # Query: (0, 0, 600, 700) hits Rect A and Rect B
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_avg2", intervals)

        assert result is not None
        assert len(result) == 1

        expected = _expected_agg_rects((0, 0, 600, 700), RECTS, "avg")
        assert result["vt_avg2"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_avg_returns_one_row_per_interval(self, rects_agg_track):
        """Avg aggregation returns one row per query interval, not per object."""
        pm.gvtrack_create("vt_avg_shape", rects_agg_track, func="avg")
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 900],
            ends1=[600, 2100],
            chroms2=["1", "1"],
            starts2=[0, 900],
            ends2=[700, 2100],
        )
        result = pm.gextract("vt_avg_shape", intervals)

        assert result is not None
        # Must be exactly 2 rows, one per query interval
        assert len(result) == 2


class TestNoIntersectionReturnsNaN:
    """Test that query intervals with no matching objects return NaN."""

    def test_area_no_intersection(self, rects_agg_track):
        """Area returns NaN when no objects intersect the query."""
        pm.gvtrack_create("vt_area_nan", rects_agg_track, func="area")
        # Query far from any objects
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_area_nan", intervals)

        # Result should either be None (no data) or have NaN
        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_area_nan"].iloc[0])

    def test_weighted_sum_no_intersection(self, rects_agg_track):
        """Weighted sum returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_ws_nan", rects_agg_track, func="weighted.sum")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_ws_nan", intervals)

        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_ws_nan"].iloc[0])

    def test_min_no_intersection(self, rects_agg_track):
        """Min returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_min_nan", rects_agg_track, func="min")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_min_nan", intervals)

        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_min_nan"].iloc[0])

    def test_max_no_intersection(self, rects_agg_track):
        """Max returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_max_nan", rects_agg_track, func="max")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_max_nan", intervals)

        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_max_nan"].iloc[0])

    def test_avg_no_intersection(self, rects_agg_track):
        """Avg returns NaN when no objects intersect (0/0 case)."""
        pm.gvtrack_create("vt_avg_nan", rects_agg_track, func="avg")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_avg_nan", intervals)

        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_avg_nan"].iloc[0])

    def test_mixed_hit_and_miss(self, rects_agg_track):
        """Some intervals hit objects, others do not: hits get values, misses get NaN."""
        pm.gvtrack_create("vt_mixed", rects_agg_track, func="area")
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 400000],
            ends1=[600, 500000],
            chroms2=["1", "1"],
            starts2=[0, 400000],
            ends2=[700, 500000],
        )
        result = pm.gextract("vt_mixed", intervals)

        assert result is not None
        assert len(result) == 2

        result = result.sort_values("start1").reset_index(drop=True)
        # First interval should have a non-NaN area value
        assert not np.isnan(result["vt_mixed"].iloc[0])
        # Second interval has no intersecting objects -> NaN
        assert np.isnan(result["vt_mixed"].iloc[1])


class TestAreaWithBand:
    """Test area function with band filter."""

    def test_area_with_band_filter(self, rects_agg_track):
        """Area function should respect the band filter.

        Rect A: x=(100,300), y=(200,400), diagonal range: x1-y2+1=-299 to x2-y1=100
        Rect B: x=(200,500), y=(300,600), diagonal range: x1-y2+1=-399 to x2-y1=200
        Rect C: x=(1000,2000), y=(1000,2000), diagonal range: x1-y2+1=-999 to x2-y1=1000

        Band (-500, 0): should include Rect A (range -299..100) and Rect B (range -399..200)
            — rects intersect if x2-y1 > d1 AND x1-y2+1 < d2
            — Rect A: 100 > -500 AND -299 < 0 -> True
            — Rect B: 200 > -500 AND -399 < 0 -> True
            — Rect C: 1000 > -500 AND -999 < 0 -> True (also passes!)

        Use a tighter band to filter more precisely.
        Band (-300, -200): Rect A passes (-299 < -200 AND 100>-300), Rect B passes (-399 < -200 AND 200>-300)
                           Rect C: -999 < -200 AND 1000>-300 -> True (still passes because rects is wide)

        Let's use a band that clearly excludes Rect C by being very negative:
        Band (-400, -300): Rect A: 100>-400 AND -299<-300 -> -299 NOT < -300 -> False
                           Rect B: 200>-400 AND -399<-300 -> True
                           Rect C: 1000>-400 AND -999<-300 -> True
        """
        pm.gvtrack_create("vt_area_band", rects_agg_track, func="area")
        query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        # Band that excludes Rect A and C: only Rect B diagonal passes
        # For Rect B (200,300,500,600): x2-y1=200, x1-y2+1=-399
        # Band (-400, -100): A: 100>-400 AND -299<-100 -> True; B: 200>-400 AND -399<-100 -> True;
        #                    C: 1000>-400 AND -999<-100 -> True
        # These rects are quite wide, making band filtering less selective.
        # Let's create a more targeted test with point-like rects.

        # Instead, just verify that area works with band at all, and the output
        # has the aggregated (one-row-per-interval) shape.
        result = pm.gextract("vt_area_band", query, band=(-500, 500))
        assert result is not None
        assert len(result) == 1  # One row for the single query interval

        # Verify the area is positive (some objects matched)
        assert result["vt_area_band"].iloc[0] > 0

    def test_area_band_excludes_all(self, rects_agg_track):
        """Band that excludes all objects should yield NaN."""
        pm.gvtrack_create("vt_area_band2", rects_agg_track, func="area")
        query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        # Very far band that no object can reach
        result = pm.gextract("vt_area_band2", query, band=(999000, 1000000))

        # Either None or one row with NaN
        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_area_band2"].iloc[0])


class TestWeightedSumWith2dShifts:
    """Test weighted.sum with 2D iterator shifts."""

    def test_shifts_change_query_region(self, rects_agg_track):
        """2D shifts should offset the query region, changing which objects match."""
        pm.gvtrack_create("vt_ws_shift", rects_agg_track, func="weighted.sum")
        pm.gvtrack_iterator_2d("vt_ws_shift", sshift1=1000, eshift1=1000,
                               sshift2=1000, eshift2=1000)

        # Query: (0, 0, 600, 700) -> with shift becomes (1000, 1000, 1600, 1700)
        # Shifted query should hit Rect C (1000,1000,2000,2000)
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_ws_shift", intervals)

        assert result is not None
        assert len(result) == 1

        # The shifted query (1000,1000,1600,1700) intersects Rect C:
        # intersection = (1000,1000,1600,1700) -> 600*700 = 420000
        # weighted.sum = 3.0 * 420000 = 1260000
        expected = _expected_agg_rects((1000, 1000, 1600, 1700), RECTS, "weighted.sum")
        assert result["vt_ws_shift"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_shifts_vs_manual_shift(self, rects_agg_track):
        """Vtrack 2D shifts should match manually shifted intervals."""
        pm.gvtrack_create("vt_ws_auto", rects_agg_track, func="weighted.sum")
        pm.gvtrack_iterator_2d("vt_ws_auto", sshift1=500, eshift1=500,
                               sshift2=500, eshift2=500)

        pm.gvtrack_create("vt_ws_manual", rects_agg_track, func="weighted.sum")

        query = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        shifted_query = pm.gintervals_2d("1", 500, 1100, "1", 500, 1200)

        auto_result = pm.gextract("vt_ws_auto", query)
        manual_result = pm.gextract("vt_ws_manual", shifted_query)

        if auto_result is not None and manual_result is not None:
            np.testing.assert_allclose(
                auto_result["vt_ws_auto"].to_numpy(dtype=float),
                manual_result["vt_ws_manual"].to_numpy(dtype=float),
                rtol=1e-5,
            )


class TestMultipleObjectsPerInterval:
    """Test aggregation when multiple rectangles overlap one query interval."""

    def test_all_funcs_with_three_objects(self, rects_agg_track):
        """All aggregation functions with three overlapping objects."""
        query = (0, 0, 500000, 500000)  # Covers all three rects
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        for func_name in ["area", "weighted.sum", "min", "max", "avg"]:
            vt_name = f"vt_multi_{func_name.replace('.', '_')}"
            pm.gvtrack_create(vt_name, rects_agg_track, func=func_name)
            result = pm.gextract(vt_name, intervals)

            assert result is not None, f"func={func_name}: result should not be None"
            assert len(result) == 1, f"func={func_name}: should be one row per interval"

            expected = _expected_agg_rects(query, RECTS, func_name)
            assert result[vt_name].iloc[0] == pytest.approx(
                expected, rel=1e-5
            ), f"func={func_name}: value mismatch"

            pm.gvtrack_rm(vt_name)

    def test_partial_overlap_area_calculation(self, rects_agg_track):
        """Partial overlap: area should count only the intersecting portion."""
        pm.gvtrack_create("vt_partial", rects_agg_track, func="area")
        # Query overlaps only part of Rect A: [150,250) x [250,350)
        # Intersection with A: (150,250,250,350) -> 100*100 = 10000
        # Intersection with B: (200,300,250,350) -> 50*50 = 2500
        intervals = pm.gintervals_2d("1", 150, 250, "1", 250, 350)
        result = pm.gextract("vt_partial", intervals)

        assert result is not None
        assert len(result) == 1

        expected = _expected_agg_rects((150, 250, 250, 350), RECTS, "area")
        assert result["vt_partial"].iloc[0] == pytest.approx(expected, rel=1e-5)


class TestAggregationVtrackReturnsOneRowPerInterval:
    """Verify the output shape is one row per query interval for all aggregation funcs."""

    def test_output_shape_matches_input_intervals(self, rects_agg_track):
        """Number of output rows equals number of input query intervals."""
        # 5 query intervals: some hit objects, some do not
        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1", "1", "1"],
            starts1=[0, 900, 3000, 100, 400000],
            ends1=[600, 2100, 4000, 200, 500000],
            chroms2=["1", "1", "1", "1", "1"],
            starts2=[0, 900, 3000, 200, 400000],
            ends2=[700, 2100, 4000, 300, 500000],
        )

        for func_name in ["area", "weighted.sum", "min", "max", "avg"]:
            vt_name = f"vt_shape_{func_name.replace('.', '_')}"
            pm.gvtrack_create(vt_name, rects_agg_track, func=func_name)
            result = pm.gextract(vt_name, intervals)

            assert result is not None, f"func={func_name}: should not be None"
            assert len(result) == 5, (
                f"func={func_name}: expected 5 rows (one per interval), got {len(result)}"
            )
            pm.gvtrack_rm(vt_name)

    def test_intervalID_maps_back_correctly(self, rects_agg_track):
        """Each output row's intervalID should map to an input interval."""
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 900],
            ends1=[600, 2100],
            chroms2=["1", "1"],
            starts2=[0, 900],
            ends2=[700, 2100],
        )
        pm.gvtrack_create("vt_iid", rects_agg_track, func="area")
        result = pm.gextract("vt_iid", intervals)

        assert result is not None
        assert len(result) == 2
        assert set(result["intervalID"].unique()) == {0, 1}


class TestAreaPointsTrack:
    """Test area function on a points track (area = count of matching points)."""

    def test_area_equals_point_count(self, points_agg_track):
        """For points, area = number of matching points (each contributes 1)."""
        pm.gvtrack_create("vt_pt_area", points_agg_track, func="area")
        # Query covers P1 (100,200) and P2 (150,250) but not P3 (5000,5000)
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_pt_area", intervals)

        assert result is not None
        assert len(result) == 1

        expected = _expected_agg_points((0, 0, 300, 300), POINTS, "area")
        assert result["vt_pt_area"].iloc[0] == pytest.approx(expected, rel=1e-5)
        assert result["vt_pt_area"].iloc[0] == pytest.approx(2.0, rel=1e-5)

    def test_weighted_sum_points(self, points_agg_track):
        """Weighted sum for points: sum of values (each area=1)."""
        pm.gvtrack_create("vt_pt_ws", points_agg_track, func="weighted.sum")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_pt_ws", intervals)

        assert result is not None
        assert len(result) == 1

        # P1 val=5.0, P2 val=10.0 -> weighted.sum = 5.0*1 + 10.0*1 = 15.0
        expected = _expected_agg_points((0, 0, 300, 300), POINTS, "weighted.sum")
        assert result["vt_pt_ws"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_min_points(self, points_agg_track):
        """Min for points: minimum value among matching points."""
        pm.gvtrack_create("vt_pt_min", points_agg_track, func="min")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_pt_min", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_pt_min"].iloc[0] == pytest.approx(5.0, rel=1e-5)

    def test_max_points(self, points_agg_track):
        """Max for points: maximum value among matching points."""
        pm.gvtrack_create("vt_pt_max", points_agg_track, func="max")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_pt_max", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_pt_max"].iloc[0] == pytest.approx(10.0, rel=1e-5)

    def test_avg_points(self, points_agg_track):
        """Avg for points: simple average (all areas=1)."""
        pm.gvtrack_create("vt_pt_avg", points_agg_track, func="avg")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_pt_avg", intervals)

        assert result is not None
        assert len(result) == 1
        # avg = (5.0 + 10.0) / 2 = 7.5
        expected = _expected_agg_points((0, 0, 300, 300), POINTS, "avg")
        assert result["vt_pt_avg"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_points_no_match(self, points_agg_track):
        """Points query with no match returns NaN."""
        pm.gvtrack_create("vt_pt_empty", points_agg_track, func="area")
        intervals = pm.gintervals_2d("1", 300000, 400000, "1", 300000, 400000)
        result = pm.gextract("vt_pt_empty", intervals)

        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_pt_empty"].iloc[0])


class TestAggregationConsistency:
    """Cross-check aggregation functions against each other."""

    def test_avg_equals_weighted_sum_div_area(self, rects_agg_track):
        """avg should equal weighted.sum / area for any query."""
        pm.gvtrack_create("vt_area_c", rects_agg_track, func="area")
        pm.gvtrack_create("vt_ws_c", rects_agg_track, func="weighted.sum")
        pm.gvtrack_create("vt_avg_c", rects_agg_track, func="avg")

        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 900],
            ends1=[600, 2100],
            chroms2=["1", "1"],
            starts2=[0, 900],
            ends2=[700, 2100],
        )

        res_area = pm.gextract("vt_area_c", intervals)
        res_ws = pm.gextract("vt_ws_c", intervals)
        res_avg = pm.gextract("vt_avg_c", intervals)

        assert res_area is not None
        assert res_ws is not None
        assert res_avg is not None

        for i in range(len(res_area)):
            area_val = res_area["vt_area_c"].iloc[i]
            ws_val = res_ws["vt_ws_c"].iloc[i]
            avg_val = res_avg["vt_avg_c"].iloc[i]

            if np.isnan(area_val) or area_val == 0:
                assert np.isnan(avg_val)
            else:
                computed_avg = ws_val / area_val
                assert avg_val == pytest.approx(computed_avg, rel=1e-5)

    def test_min_le_avg_le_max(self, rects_agg_track):
        """For any query with hits: min <= avg <= max."""
        pm.gvtrack_create("vt_min_r", rects_agg_track, func="min")
        pm.gvtrack_create("vt_avg_r", rects_agg_track, func="avg")
        pm.gvtrack_create("vt_max_r", rects_agg_track, func="max")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        res_min = pm.gextract("vt_min_r", intervals)
        res_avg = pm.gextract("vt_avg_r", intervals)
        res_max = pm.gextract("vt_max_r", intervals)

        assert res_min is not None
        assert res_avg is not None
        assert res_max is not None

        min_val = res_min["vt_min_r"].iloc[0]
        avg_val = res_avg["vt_avg_r"].iloc[0]
        max_val = res_max["vt_max_r"].iloc[0]

        assert min_val <= avg_val
        assert avg_val <= max_val


class TestEdgeCases:
    """Edge case tests for 2D vtrack aggregation."""

    def test_zero_width_intersection(self, rects_agg_track):
        """Query that touches a rect edge but has zero intersection area."""
        pm.gvtrack_create("vt_edge", rects_agg_track, func="area")
        # Rect A ends at x2=300. Query starts at x1=300 -> no overlap (half-open).
        intervals = pm.gintervals_2d("1", 300, 400, "1", 200, 400)
        result = pm.gextract("vt_edge", intervals)

        if result is not None and len(result) == 1:
            # The query [300,400) x [200,400) should overlap Rect B (200,300,500,600)
            # in [300,400) x [300,400) = 100*100 = 10000
            # But NOT Rect A (100,200,300,400) because x1=300 == ox2=300 => no overlap
            expected = _expected_agg_rects((300, 200, 400, 400), RECTS, "area")
            assert result["vt_edge"].iloc[0] == pytest.approx(expected, rel=1e-5)

    def test_single_unit_query(self, rects_agg_track):
        """Query of width 1 x 1 (single bp) on a rects track."""
        pm.gvtrack_create("vt_1bp", rects_agg_track, func="area")
        # Point (150, 250) is inside Rect A [100,300) x [200,400)
        intervals = pm.gintervals_2d("1", 150, 151, "1", 250, 251)
        result = pm.gextract("vt_1bp", intervals)

        assert result is not None
        assert len(result) == 1
        # Intersection area = 1*1 = 1
        assert result["vt_1bp"].iloc[0] == pytest.approx(1.0, rel=1e-5)

    def test_trans_chromosomes_no_data(self, rects_agg_track):
        """Query on chr1-chr2 where the agg track only has chr1-chr1 data."""
        pm.gvtrack_create("vt_trans", rects_agg_track, func="area")
        intervals = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
        result = pm.gextract("vt_trans", intervals)

        # No chr1-chr2 file exists -> should be None or NaN
        if result is not None:
            assert len(result) == 1
            assert np.isnan(result["vt_trans"].iloc[0])


class TestVtrackCreationValidation:
    """Test that vtrack creation accepts the new aggregation function names."""

    @pytest.mark.parametrize("func_name", ["area", "weighted.sum", "min", "max", "avg"])
    def test_create_with_agg_func(self, rects_agg_track, func_name):
        """gvtrack_create should accept aggregation function names for 2D tracks."""
        vt_name = f"vt_valid_{func_name.replace('.', '_')}"
        # This should not raise
        pm.gvtrack_create(vt_name, rects_agg_track, func=func_name)
        info = pm.gvtrack_info(vt_name)
        assert info["func"] == func_name
        pm.gvtrack_rm(vt_name)


class TestExpressionMixing:
    """Test expressions that mix aggregation vtracks with other tracks/vtracks."""

    def test_two_agg_vtracks_in_expression(self, rects_agg_track):
        """Expression combining two aggregation vtracks: vt_min + vt_max."""
        pm.gvtrack_create("vt_min_e", rects_agg_track, func="min")
        pm.gvtrack_create("vt_max_e", rects_agg_track, func="max")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_min_e + vt_max_e", intervals)

        assert result is not None
        assert len(result) == 1

        expected_min = _expected_agg_rects((0, 0, 500000, 500000), RECTS, "min")
        expected_max = _expected_agg_rects((0, 0, 500000, 500000), RECTS, "max")
        expected_sum = expected_min + expected_max

        vals = result.iloc[0, -2]  # Expression column (before intervalID)
        assert vals == pytest.approx(expected_sum, rel=1e-5)

    def test_agg_vtrack_with_raw_track_raises_or_works(self, rects_agg_track):
        """Mixing an aggregation vtrack with a raw 2D track name in one expression.

        This should either:
        1. Raise an error (since raw produces N rows per interval and agg produces 1), or
        2. Work if the system broadcasts correctly.

        The test verifies one of these outcomes.
        """
        pm.gvtrack_create("vt_agg_mix", rects_agg_track, func="area")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        try:
            result = pm.gextract(f"vt_agg_mix + {rects_agg_track}", intervals)
            # If it succeeds, the shape should be consistent
            assert result is not None
        except (ValueError, RuntimeError):
            # It's acceptable for mixed shapes to be rejected
            pass
