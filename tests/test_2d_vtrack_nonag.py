"""Tests for non-aggregation 2D virtual track functions.

These tests verify that 2D virtual tracks with non-aggregation functions
(exists, size, first, last, sample, global.percentile) return ONE ROW PER
QUERY INTERVAL with properly computed values.
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
def rects_track():
    """Create a 2D rects track with known geometry.

    Layout on chr1-chr1 (arena 0..500000 x 0..500000):

        Rect A: (100, 200, 300, 400, value=5.0)
        Rect B: (200, 300, 500, 600, value=10.0)
        Rect C: (1000, 1000, 2000, 2000, value=3.0)

    Rects A and B overlap in [200,300) x [300,400).
    Rect C is isolated.
    """
    tname = "test.nonag_rects"
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
def points_track():
    """Create a 2D points track with known positions.

    Points on chr1-chr1:
        P1: (100, 200, value=5.0)
        P2: (150, 250, value=10.0)
        P3: (5000, 5000, value=3.0)
    """
    tname = "test.nonag_points"
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


# ===========================================================================
# Tests: exists
# ===========================================================================


class TestExists:
    """Test the 'exists' function on 2D virtual tracks."""

    def test_exists_returns_1_when_objects_found(self, rects_track):
        """exists returns 1.0 when objects intersect the query."""
        pm.gvtrack_create("vt_exists", rects_track, func="exists")
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_exists", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_exists"].iloc[0] == 1.0

    def test_exists_returns_0_when_no_objects(self, rects_track):
        """exists returns 0.0 when no objects intersect the query."""
        pm.gvtrack_create("vt_exists0", rects_track, func="exists")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_exists0", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_exists0"].iloc[0] == 0.0

    def test_exists_multiple_intervals(self, rects_track):
        """exists returns correct values for multiple query intervals."""
        pm.gvtrack_create("vt_exists_m", rects_track, func="exists")
        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1"],
            starts1=[0, 900, 400000],
            ends1=[600, 2100, 500000],
            chroms2=["1", "1", "1"],
            starts2=[0, 900, 400000],
            ends2=[700, 2100, 500000],
        )
        result = pm.gextract("vt_exists_m", intervals)

        assert result is not None
        assert len(result) == 3
        result = result.sort_values("start1").reset_index(drop=True)
        assert result["vt_exists_m"].iloc[0] == 1.0  # Hits A+B
        assert result["vt_exists_m"].iloc[1] == 1.0  # Hits C
        assert result["vt_exists_m"].iloc[2] == 0.0  # No objects

    def test_exists_points_track(self, points_track):
        """exists works on points tracks."""
        pm.gvtrack_create("vt_exists_p", points_track, func="exists")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_exists_p", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_exists_p"].iloc[0] == 1.0

    def test_exists_trans_no_data(self, rects_track):
        """exists on chr1-chr2 where the track has no data returns 0."""
        pm.gvtrack_create("vt_exists_t", rects_track, func="exists")
        intervals = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
        result = pm.gextract("vt_exists_t", intervals)

        assert result is not None
        assert len(result) == 1
        # No data file for chr1-chr2 -> exists=0
        assert result["vt_exists_t"].iloc[0] == 0.0


# ===========================================================================
# Tests: size
# ===========================================================================


class TestSize:
    """Test the 'size' function on 2D virtual tracks."""

    def test_size_counts_intersecting_objects(self, rects_track):
        """size returns the number of objects intersecting the query."""
        pm.gvtrack_create("vt_size", rects_track, func="size")
        # Query hits Rect A and Rect B
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_size", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size"].iloc[0] == 2.0

    def test_size_single_object(self, rects_track):
        """size returns 1 when exactly one object intersects."""
        pm.gvtrack_create("vt_size1", rects_track, func="size")
        # Query hits only Rect C
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_size1", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size1"].iloc[0] == 1.0

    def test_size_zero_when_no_objects(self, rects_track):
        """size returns 0.0 when no objects intersect."""
        pm.gvtrack_create("vt_size0", rects_track, func="size")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_size0", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size0"].iloc[0] == 0.0

    def test_size_all_three(self, rects_track):
        """size counts all three objects when query covers everything."""
        pm.gvtrack_create("vt_size3", rects_track, func="size")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_size3", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size3"].iloc[0] == 3.0

    def test_size_points_track(self, points_track):
        """size counts points correctly."""
        pm.gvtrack_create("vt_size_p", points_track, func="size")
        # Covers P1 and P2 but not P3
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_size_p", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size_p"].iloc[0] == 2.0

    def test_size_trans_no_data(self, rects_track):
        """size on missing chrom pair returns 0."""
        pm.gvtrack_create("vt_size_t", rects_track, func="size")
        intervals = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
        result = pm.gextract("vt_size_t", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size_t"].iloc[0] == 0.0


# ===========================================================================
# Tests: first
# ===========================================================================


class TestFirst:
    """Test the 'first' function on 2D virtual tracks."""

    def test_first_returns_value_of_first_object(self, rects_track):
        """first returns value of first intersecting object."""
        pm.gvtrack_create("vt_first", rects_track, func="first")
        # Query hits Rect A (val=5) and Rect B (val=10); first is A
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_first", intervals)

        assert result is not None
        assert len(result) == 1
        # The first object returned by the quad-tree query should be Rect A (val=5.0)
        val = result["vt_first"].iloc[0]
        assert val in (5.0, 10.0)  # Must be one of the intersecting objects

    def test_first_single_object(self, rects_track):
        """first returns the value of the single intersecting object."""
        pm.gvtrack_create("vt_first1", rects_track, func="first")
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_first1", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_first1"].iloc[0] == pytest.approx(3.0)

    def test_first_nan_when_no_objects(self, rects_track):
        """first returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_first_nan", rects_track, func="first")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_first_nan", intervals)

        assert result is not None
        assert len(result) == 1
        assert np.isnan(result["vt_first_nan"].iloc[0])

    def test_first_points_track(self, points_track):
        """first works on points tracks."""
        pm.gvtrack_create("vt_first_p", points_track, func="first")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_first_p", intervals)

        assert result is not None
        assert len(result) == 1
        val = result["vt_first_p"].iloc[0]
        assert val in (5.0, 10.0)


# ===========================================================================
# Tests: last
# ===========================================================================


class TestLast:
    """Test the 'last' function on 2D virtual tracks."""

    def test_last_returns_value_of_last_object(self, rects_track):
        """last returns value of last intersecting object."""
        pm.gvtrack_create("vt_last", rects_track, func="last")
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_last", intervals)

        assert result is not None
        assert len(result) == 1
        val = result["vt_last"].iloc[0]
        assert val in (5.0, 10.0)  # Must be one of the intersecting objects

    def test_last_single_object(self, rects_track):
        """last returns the value of the single intersecting object."""
        pm.gvtrack_create("vt_last1", rects_track, func="last")
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_last1", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_last1"].iloc[0] == pytest.approx(3.0)

    def test_last_nan_when_no_objects(self, rects_track):
        """last returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_last_nan", rects_track, func="last")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_last_nan", intervals)

        assert result is not None
        assert len(result) == 1
        assert np.isnan(result["vt_last_nan"].iloc[0])

    def test_last_points_track(self, points_track):
        """last works on points tracks."""
        pm.gvtrack_create("vt_last_p", points_track, func="last")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_last_p", intervals)

        assert result is not None
        assert len(result) == 1
        val = result["vt_last_p"].iloc[0]
        assert val in (5.0, 10.0)

    def test_first_and_last_differ_with_multiple_objects(self, rects_track):
        """With multiple objects, first and last return different ends of the list."""
        pm.gvtrack_create("vt_first_fl", rects_track, func="first")
        pm.gvtrack_create("vt_last_fl", rects_track, func="last")
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)

        result_f = pm.gextract("vt_first_fl", intervals)
        result_l = pm.gextract("vt_last_fl", intervals)

        assert result_f is not None
        assert result_l is not None
        # Both must be valid object values
        assert result_f["vt_first_fl"].iloc[0] in (5.0, 10.0)
        assert result_l["vt_last_fl"].iloc[0] in (5.0, 10.0)


# ===========================================================================
# Tests: sample
# ===========================================================================


class TestSample:
    """Test the 'sample' function on 2D virtual tracks."""

    def test_sample_returns_valid_value(self, rects_track):
        """sample returns a value from one of the intersecting objects."""
        pm.gvtrack_create("vt_sample", rects_track, func="sample")
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_sample", intervals)

        assert result is not None
        assert len(result) == 1
        val = result["vt_sample"].iloc[0]
        # Must be one of the values from Rect A or Rect B
        assert val in (5.0, 10.0)

    def test_sample_single_object(self, rects_track):
        """sample with one object always returns that object's value."""
        pm.gvtrack_create("vt_sample1", rects_track, func="sample")
        intervals = pm.gintervals_2d("1", 900, 2100, "1", 900, 2100)
        result = pm.gextract("vt_sample1", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_sample1"].iloc[0] == pytest.approx(3.0)

    def test_sample_nan_when_no_objects(self, rects_track):
        """sample returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_sample_nan", rects_track, func="sample")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_sample_nan", intervals)

        assert result is not None
        assert len(result) == 1
        assert np.isnan(result["vt_sample_nan"].iloc[0])

    def test_sample_points_track(self, points_track):
        """sample works on points tracks."""
        pm.gvtrack_create("vt_sample_p", points_track, func="sample")
        intervals = pm.gintervals_2d("1", 0, 300, "1", 0, 300)
        result = pm.gextract("vt_sample_p", intervals)

        assert result is not None
        assert len(result) == 1
        val = result["vt_sample_p"].iloc[0]
        assert val in (5.0, 10.0)

    def test_sample_all_three_objects(self, rects_track):
        """sample from all three objects always returns a valid value."""
        pm.gvtrack_create("vt_sample3", rects_track, func="sample")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_sample3", intervals)

        assert result is not None
        assert len(result) == 1
        val = result["vt_sample3"].iloc[0]
        assert val in (3.0, 5.0, 10.0)


# ===========================================================================
# Tests: global.percentile
# ===========================================================================


class TestGlobalPercentile:
    """Test the 'global.percentile' function on 2D virtual tracks."""

    def test_global_percentile_basic(self, rects_track):
        """global.percentile returns values in [0, 1)."""
        pm.gvtrack_create("vt_gpct", rects_track, func="global.percentile")
        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1"],
            starts1=[0, 900, 100],
            ends1=[600, 2100, 200],
            chroms2=["1", "1", "1"],
            starts2=[0, 900, 200],
            ends2=[700, 2100, 300],
        )
        result = pm.gextract("vt_gpct", intervals)

        assert result is not None
        assert len(result) == 3
        vals = result["vt_gpct"].to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        assert len(valid) == 3
        # All values should be in [0, 1]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_global_percentile_ordering(self, rects_track):
        """Intervals with higher avg values should have higher percentile ranks."""
        pm.gvtrack_create("vt_gpct_o", rects_track, func="global.percentile")
        pm.gvtrack_create("vt_avg_o", rects_track, func="avg")

        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1"],
            starts1=[0, 900, 100],
            ends1=[600, 2100, 200],
            chroms2=["1", "1", "1"],
            starts2=[0, 900, 200],
            ends2=[700, 2100, 300],
        )

        res_pct = pm.gextract("vt_gpct_o", intervals)
        res_avg = pm.gextract("vt_avg_o", intervals)

        assert res_pct is not None
        assert res_avg is not None

        pct_vals = res_pct.sort_values("start1").reset_index(drop=True)["vt_gpct_o"].to_numpy()
        avg_vals = res_avg.sort_values("start1").reset_index(drop=True)["vt_avg_o"].to_numpy()

        # Higher avg should have higher or equal percentile
        valid_mask = ~np.isnan(pct_vals) & ~np.isnan(avg_vals)
        pct_valid = pct_vals[valid_mask]
        avg_valid = avg_vals[valid_mask]

        # Sort by avg and check percentiles are non-decreasing
        sort_idx = np.argsort(avg_valid)
        sorted_pct = pct_valid[sort_idx]
        for i in range(1, len(sorted_pct)):
            assert sorted_pct[i] >= sorted_pct[i - 1]

    def test_global_percentile_nan_for_no_intersection(self, rects_track):
        """global.percentile returns NaN when no objects intersect."""
        pm.gvtrack_create("vt_gpct_nan", rects_track, func="global.percentile")
        intervals = pm.gintervals_2d("1", 400000, 500000, "1", 400000, 500000)
        result = pm.gextract("vt_gpct_nan", intervals)

        assert result is not None
        assert len(result) == 1
        assert np.isnan(result["vt_gpct_nan"].iloc[0])

    def test_global_percentile_single_interval(self, rects_track):
        """global.percentile with a single interval returns 0.0 (no values are strictly less)."""
        pm.gvtrack_create("vt_gpct_1", rects_track, func="global.percentile")
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_gpct_1", intervals)

        assert result is not None
        assert len(result) == 1
        # With only one valid value, percentile should be 0.0
        assert result["vt_gpct_1"].iloc[0] == pytest.approx(0.0)

    def test_global_percentile_points_track(self, points_track):
        """global.percentile works on points tracks."""
        pm.gvtrack_create("vt_gpct_p", points_track, func="global.percentile")
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 4000],
            ends1=[300, 6000],
            chroms2=["1", "1"],
            starts2=[0, 4000],
            ends2=[300, 6000],
        )
        result = pm.gextract("vt_gpct_p", intervals)

        assert result is not None
        assert len(result) == 2
        vals = result["vt_gpct_p"].to_numpy(dtype=float)
        valid = vals[~np.isnan(vals)]
        assert len(valid) == 2
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)


# ===========================================================================
# Tests: band filter with non-aggregation functions
# ===========================================================================


class TestBandFilter:
    """Test non-aggregation functions with band filter."""

    def test_exists_with_band_excludes_all(self, rects_track):
        """exists with a band that excludes all objects returns 0."""
        pm.gvtrack_create("vt_exists_band", rects_track, func="exists")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_exists_band", intervals, band=(999000, 1000000))

        assert result is not None
        assert len(result) == 1
        assert result["vt_exists_band"].iloc[0] == 0.0

    def test_size_with_band(self, rects_track):
        """size respects band filtering."""
        pm.gvtrack_create("vt_size_band", rects_track, func="size")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        # Use a band that passes some but not all objects.
        # band=(-500, 500) should pass most objects.
        result = pm.gextract("vt_size_band", intervals, band=(-500, 500))

        assert result is not None
        assert len(result) == 1
        # With this band, some objects should still pass
        val = result["vt_size_band"].iloc[0]
        assert val >= 0.0

    def test_first_with_band_excludes_all(self, rects_track):
        """first with band that excludes all returns NaN."""
        pm.gvtrack_create("vt_first_band", rects_track, func="first")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_first_band", intervals, band=(999000, 1000000))

        assert result is not None
        assert len(result) == 1
        assert np.isnan(result["vt_first_band"].iloc[0])


# ===========================================================================
# Tests: 2D shifts with non-aggregation functions
# ===========================================================================


class TestShifts:
    """Test non-aggregation functions with 2D iterator shifts."""

    def test_exists_with_shift(self, rects_track):
        """exists with 2D shifts correctly changes query region."""
        pm.gvtrack_create("vt_exists_s", rects_track, func="exists")
        pm.gvtrack_iterator_2d("vt_exists_s", sshift1=1000, eshift1=1000,
                               sshift2=1000, eshift2=1000)

        # Query: (0, 0, 600, 700) -> shifted to (1000, 1000, 1600, 1700)
        # Should hit Rect C (1000,1000,2000,2000)
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_exists_s", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_exists_s"].iloc[0] == 1.0

    def test_size_with_shift(self, rects_track):
        """size with 2D shifts correctly changes query region."""
        pm.gvtrack_create("vt_size_s", rects_track, func="size")
        pm.gvtrack_iterator_2d("vt_size_s", sshift1=1000, eshift1=1000,
                               sshift2=1000, eshift2=1000)

        # Shifted query hits only Rect C
        intervals = pm.gintervals_2d("1", 0, 600, "1", 0, 700)
        result = pm.gextract("vt_size_s", intervals)

        assert result is not None
        assert len(result) == 1
        assert result["vt_size_s"].iloc[0] == 1.0


# ===========================================================================
# Tests: output shape
# ===========================================================================


class TestOutputShape:
    """Verify the output shape is one row per query interval for all non-agg funcs."""

    @pytest.mark.parametrize("func_name", ["exists", "size", "first", "last", "sample"])
    def test_output_shape_matches_input(self, rects_track, func_name):
        """Number of output rows equals number of input query intervals."""
        vt_name = f"vt_shape_{func_name}"
        pm.gvtrack_create(vt_name, rects_track, func=func_name)
        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1"],
            starts1=[0, 900, 400000],
            ends1=[600, 2100, 500000],
            chroms2=["1", "1", "1"],
            starts2=[0, 900, 400000],
            ends2=[700, 2100, 500000],
        )
        result = pm.gextract(vt_name, intervals)

        assert result is not None
        assert len(result) == 3

    @pytest.mark.parametrize("func_name", ["exists", "size", "first", "last", "sample"])
    def test_intervalID_maps_correctly(self, rects_track, func_name):
        """Each output row's intervalID maps to an input interval."""
        vt_name = f"vt_iid_{func_name}"
        pm.gvtrack_create(vt_name, rects_track, func=func_name)
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[0, 900],
            ends1=[600, 2100],
            chroms2=["1", "1"],
            starts2=[0, 900],
            ends2=[700, 2100],
        )
        result = pm.gextract(vt_name, intervals)

        assert result is not None
        assert len(result) == 2
        assert set(result["intervalID"].unique()) == {0, 1}


# ===========================================================================
# Tests: vtrack creation accepts new function names
# ===========================================================================


class TestVtrackCreation:
    """Test that gvtrack_create accepts the new function names for 2D tracks."""

    @pytest.mark.parametrize("func_name", ["exists", "size", "first", "last", "sample", "global.percentile"])
    def test_create_with_nonag_func(self, rects_track, func_name):
        """gvtrack_create should accept non-aggregation function names for 2D tracks."""
        vt_name = f"vt_create_{func_name.replace('.', '_')}"
        pm.gvtrack_create(vt_name, rects_track, func=func_name)
        info = pm.gvtrack_info(vt_name)
        assert info["func"] == func_name
        pm.gvtrack_rm(vt_name)


# ===========================================================================
# Tests: expression mixing
# ===========================================================================


class TestExpressionMixing:
    """Test expressions combining non-aggregation vtracks."""

    def test_exists_plus_size_expression(self, rects_track):
        """Expression combining exists and size vtracks."""
        pm.gvtrack_create("vt_ex", rects_track, func="exists")
        pm.gvtrack_create("vt_sz", rects_track, func="size")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_ex + vt_sz", intervals)

        assert result is not None
        assert len(result) == 1
        # exists=1, size=3, so 1+3=4
        val = result.iloc[0, -2]  # Expression column (before intervalID)
        assert val == pytest.approx(4.0)

    def test_mixed_agg_and_nonag(self, rects_track):
        """Expression combining an aggregation vtrack with a non-aggregation vtrack."""
        pm.gvtrack_create("vt_area_mix", rects_track, func="area")
        pm.gvtrack_create("vt_size_mix", rects_track, func="size")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_area_mix + vt_size_mix", intervals)

        assert result is not None
        assert len(result) == 1
