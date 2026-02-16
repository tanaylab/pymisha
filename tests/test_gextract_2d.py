"""Tests for 2D extraction in gextract."""

import os
import shutil

import numpy as np
import pytest

import pymisha as pm

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        import _pymisha
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestGextract2dBasic:
    """Test basic 2D extraction from existing rects_track."""

    def test_extract_existing_rects_track(self):
        """Extract from rects_track with 2D intervals returns 2D columns."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("rects_track", intervals)
        assert result is not None
        assert len(result) > 0
        # Must have 2D columns
        assert "chrom1" in result.columns
        assert "start1" in result.columns
        assert "end1" in result.columns
        assert "chrom2" in result.columns
        assert "start2" in result.columns
        assert "end2" in result.columns
        assert "rects_track" in result.columns
        assert "intervalID" in result.columns

    def test_extract_rects_values_match_r(self):
        """Extracted values should match R misha output."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("rects_track", intervals)
        assert result is not None
        # R misha returns 76 rows for this query on the test DB
        assert len(result) == 76

    def test_extract_narrow_window(self):
        """Extract with narrow 2D interval returns intersecting objects."""
        intervals = pm.gintervals_2d("1", 0, 100000, "1", 300000, 400000)
        result = pm.gextract("rects_track", intervals)
        if result is not None and len(result) > 0:
            # All results must *intersect* the query rectangle
            # (not necessarily be fully contained)
            assert (result["start1"] < 100000).all()
            assert (result["end1"] > 0).all()
            assert (result["start2"] < 400000).all()
            assert (result["end2"] > 300000).all()

    def test_extract_non_overlapping_returns_empty(self):
        """Extract from region with no 2D track objects returns None."""
        # Far corner unlikely to have track data in the small test DB
        intervals = pm.gintervals_2d("X", 199000, 200000, "X", 199000, 200000)
        result = pm.gextract("rects_track", intervals)
        # May be None or empty DataFrame
        assert result is None or len(result) == 0

    def test_extract_intervalID_correct(self):
        """intervalID should correspond to the input interval index."""
        intervals = pm.gintervals_2d(
            ["1", "1"],
            [0, 200000],
            [100000, 300000],
            ["1", "1"],
            [0, 300000],
            [200000, 500000],
        )
        result = pm.gextract("rects_track", intervals)
        if result is not None and len(result) > 0:
            # intervalID should be 0 or 1
            assert set(result["intervalID"].unique()).issubset({0, 1})


class TestGextract2dCreatedTrack:
    """Test 2D extraction from a freshly created track."""

    def test_roundtrip_rects(self):
        """Create a rects track, extract values, verify roundtrip."""
        tname = "test.test_2d_extract_rt"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 1000],
                ends1=[200, 2000],
                chroms2=["1", "1"],
                starts2=[300, 3000],
                ends2=[400, 4000],
            )
            values = [42.0, 99.0]
            pm.gtrack_2d_create(tname, "extract test", intervals, values)

            # Extract over the full range
            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            result = pm.gextract(tname, query)

            assert result is not None
            assert len(result) == 2
            # Values should match (float32 precision)
            extracted_vals = sorted(result[tname].tolist())
            assert abs(extracted_vals[0] - 42.0) < 0.01
            assert abs(extracted_vals[1] - 99.0) < 0.01
        finally:
            _cleanup_track(tname)

    def test_roundtrip_points(self):
        """Create a points track, extract values, verify roundtrip."""
        tname = "test.test_2d_extract_pts"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[500],
                ends1=[501],
                chroms2=["1"],
                starts2=[1500],
                ends2=[1501],
            )
            values = [7.5]
            pm.gtrack_2d_create(tname, "points extract", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            result = pm.gextract(tname, query)

            assert result is not None
            assert len(result) == 1
            assert abs(result[tname].iloc[0] - 7.5) < 0.01
        finally:
            _cleanup_track(tname)

    def test_partial_overlap_filtering(self):
        """Only objects intersecting the query should be returned."""
        tname = "test.test_2d_extract_filter"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 400000],
                ends1=[200, 2000, 450000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 400000],
                ends2=[400, 4000, 450000],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "filter test", intervals, values)

            # Query only covers the first two objects
            query = pm.gintervals_2d("1", 0, 10000, "1", 0, 10000)
            result = pm.gextract(tname, query)

            assert result is not None
            assert len(result) == 2
            extracted_vals = sorted(result[tname].tolist())
            assert abs(extracted_vals[0] - 1.0) < 0.01
            assert abs(extracted_vals[1] - 2.0) < 0.01
        finally:
            _cleanup_track(tname)

    def test_multi_chrom_pairs(self):
        """Extract from track with multiple chromosome pairs."""
        tname = "test.test_2d_extract_multi"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 100],
                ends1=[200, 200],
                chroms2=["1", "2"],
                starts2=[300, 100],
                ends2=[400, 200],
            )
            values = [10.0, 20.0]
            pm.gtrack_2d_create(tname, "multi chrom", intervals, values)

            # Query for chrom1-chrom1
            query1 = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            result1 = pm.gextract(tname, query1)
            assert result1 is not None
            assert len(result1) == 1
            assert abs(result1[tname].iloc[0] - 10.0) < 0.01

            # Query for chrom1-chrom2
            query2 = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
            result2 = pm.gextract(tname, query2)
            assert result2 is not None
            assert len(result2) == 1
            assert abs(result2[tname].iloc[0] - 20.0) < 0.01
        finally:
            _cleanup_track(tname)


class TestGextract2dColnames:
    """Test colnames parameter with 2D extraction."""

    def test_custom_colnames(self):
        """Custom colnames should rename expression columns."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("rects_track", intervals, colnames=["my_values"])
        assert result is not None
        assert "my_values" in result.columns
        assert "rects_track" not in result.columns


class TestGextract2dBand:
    """Test band parameter with 2D extraction."""

    def test_band_filters_objects(self):
        """Band parameter should filter objects by diagonal distance."""
        tname = "test.test_2d_band_filter"
        _cleanup_track(tname)
        try:
            # Create objects at known diagonal distances:
            # obj1: x=100, y=200 → delta = 100 - 200 = -100
            # obj2: x=1000, y=500 → delta = 1000 - 500 = 500
            # obj3: x=2000, y=1800 → delta = 2000 - 1800 = 200
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 2000],
                ends1=[101, 1001, 2001],
                chroms2=["1", "1", "1"],
                starts2=[200, 500, 1800],
                ends2=[201, 501, 1801],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "band test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

            # Band (100, 600): captures obj2 (delta=500) and obj3 (delta=200)
            result = pm.gextract(tname, query, band=(100, 600))
            assert result is not None
            assert len(result) == 2
            vals = sorted(result[tname].tolist())
            assert abs(vals[0] - 2.0) < 0.01
            assert abs(vals[1] - 3.0) < 0.01

            # Band (-200, 0): captures obj1 (delta=-100)
            result2 = pm.gextract(tname, query, band=(-200, 0))
            assert result2 is not None
            assert len(result2) == 1
            assert abs(result2[tname].iloc[0] - 1.0) < 0.01

            # Band (1000, 2000): captures nothing
            result3 = pm.gextract(tname, query, band=(1000, 2000))
            assert result3 is None
        finally:
            _cleanup_track(tname)

    def test_band_with_rects(self):
        """Band filter works correctly with rectangle objects."""
        tname = "test.test_2d_band_rects"
        _cleanup_track(tname)
        try:
            # Rect obj: x1=100, y1=300, x2=200, y2=400
            # Diagonal range: x1-y2+1 to x2-y1 = 100-400+1=-299 to 200-300=-100
            # Band intersects if x2 - y1 > d1 AND x1 - y2 + 1 < d2
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            values = [42.0]
            pm.gtrack_2d_create(tname, "band rects", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

            # Band that includes this rect's diagonal range
            result = pm.gextract(tname, query, band=(-300, 0))
            assert result is not None
            assert len(result) == 1

            # Band that doesn't intersect
            result2 = pm.gextract(tname, query, band=(0, 100))
            assert result2 is None
        finally:
            _cleanup_track(tname)

    def test_band_on_existing_rects_track(self):
        """Band filter on the pre-existing rects_track."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        # No band → 76 objects
        all_result = pm.gextract("rects_track", intervals)
        assert all_result is not None
        assert len(all_result) == 76

        # Very narrow band should return fewer objects
        narrow = pm.gextract("rects_track", intervals, band=(-10, 10))
        if narrow is not None:
            assert len(narrow) < 76

    def test_band_validation(self):
        """Invalid band should raise ValueError."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        with pytest.raises(ValueError, match="d1.*must be less than d2"):
            pm.gextract("rects_track", intervals, band=(100, 50))

    def test_band_with_1d_raises(self):
        """Band with 1D intervals should raise ValueError."""
        intervals = pm.gintervals("1", 0, 1000)
        with pytest.raises(ValueError, match="only supported with 2D"):
            pm.gextract("dense_track", intervals, band=(0, 100))


class TestQuadTreeReader:
    """Test the quad-tree binary reader directly."""

    def test_read_existing_rects_file(self):
        """Read the existing rects_track chr1-chr1 file."""
        from pymisha._quadtree import read_2d_track_objects

        filepath = os.path.join(TRACK_DIR, "rects_track.track", "chr1-chr1")
        assert os.path.exists(filepath), f"Test file not found: {filepath}"

        is_points, objects = read_2d_track_objects(filepath)
        assert not is_points
        assert len(objects) > 0
        # Each object should be (x1, y1, x2, y2, value)
        for obj in objects:
            assert len(obj) == 5
            x1, y1, x2, y2, val = obj
            assert x2 > x1
            assert y2 > y1

    def test_read_roundtrip_rects(self, tmp_path):
        """Write then read a rects file."""
        from pymisha._quadtree import read_2d_track_objects, write_2d_track_file

        filepath = str(tmp_path / "test_rects")
        objects = [(100, 200, 300, 400, 5.5), (1000, 2000, 3000, 4000, 7.25)]
        write_2d_track_file(filepath, objects, (0, 0, 500000, 500000), is_points=False)

        is_points, read_objs = read_2d_track_objects(filepath)
        assert not is_points
        assert len(read_objs) == 2
        # Values should roundtrip (float32 precision)
        read_vals = sorted([o[4] for o in read_objs])
        assert abs(read_vals[0] - 5.5) < 0.01
        assert abs(read_vals[1] - 7.25) < 0.01

    def test_read_roundtrip_points(self, tmp_path):
        """Write then read a points file."""
        from pymisha._quadtree import read_2d_track_objects, write_2d_track_file

        filepath = str(tmp_path / "test_points")
        objects = [(100, 200, 5.5), (1000, 2000, 7.25)]
        write_2d_track_file(filepath, objects, (0, 0, 500000, 500000), is_points=True)

        is_points, read_objs = read_2d_track_objects(filepath)
        assert is_points
        assert len(read_objs) == 2

    def test_query_spatial_filter(self, tmp_path):
        """Spatial query returns only intersecting objects."""
        from pymisha._quadtree import query_2d_track_objects, write_2d_track_file

        filepath = str(tmp_path / "test_query")
        objects = [
            (100, 200, 300, 400, 1.0),
            (5000, 6000, 7000, 8000, 2.0),
            (100000, 200000, 300000, 400000, 3.0),
        ]
        write_2d_track_file(filepath, objects, (0, 0, 500000, 500000), is_points=False)

        # Query covering only the first object
        results = query_2d_track_objects(filepath, 0, 0, 1000, 1000)
        assert len(results) == 1
        assert abs(results[0][4] - 1.0) < 0.01

        # Query covering no objects
        results_empty = query_2d_track_objects(filepath, 400000, 0, 500000, 100)
        assert len(results_empty) == 0


class TestSummaryWithBand:
    """Test gsummary, gquantiles, gdist with band parameter."""

    def test_gsummary_with_band(self):
        """gsummary with band should compute stats on band-filtered 2D extraction."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        # Without band: 76 objects
        all_summary = pm.gsummary("rects_track", intervals)
        assert all_summary["Total intervals"] == 76

        # With band: fewer objects
        band_summary = pm.gsummary("rects_track", intervals, band=(-10, 10))
        assert band_summary["Total intervals"] <= 76
        # If there are values, stats should be non-NaN
        if band_summary["Total intervals"] > 0:
            assert not np.isnan(band_summary["Mean"])

    def test_gsummary_band_no_results(self):
        """gsummary with restrictive band returns empty stats."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        # Very large band offset unlikely to have any objects
        result = pm.gsummary("rects_track", intervals, band=(999999, 1000000))
        assert result["Total intervals"] == 0
        assert np.isnan(result["Mean"])

    def test_gquantiles_with_band(self):
        """gquantiles with band should compute quantiles on band-filtered extraction."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        # Full extraction quantiles
        all_q = pm.gquantiles("rects_track", [0.25, 0.5, 0.75], intervals)
        assert len(all_q) == 3
        assert not any(np.isnan(all_q))

        # Band-filtered quantiles
        band_q = pm.gquantiles("rects_track", 0.5, intervals, band=(-500000, 500000))
        # With a very wide band, should be similar to full
        assert not np.isnan(band_q.iloc[0])

    def test_gdist_with_band(self):
        """gdist with band should compute distribution on band-filtered values."""
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        # Get the value range from full extraction first
        all_result = pm.gextract("rects_track", intervals)
        all_vals = all_result["rects_track"].values
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        breaks = [vmin - 1, (vmin + vmax) / 2, vmax + 1]

        # Full distribution
        all_dist = pm.gdist("rects_track", breaks, intervals=intervals)
        assert all_dist.sum() == 76  # all 76 objects should be binned

        # Band-filtered distribution should have fewer or equal counts
        band_dist = pm.gdist("rects_track", breaks, intervals=intervals, band=(-10, 10))
        assert band_dist.sum() <= 76


# ---------------------------------------------------------------------------
# Tests ported from R test-2d-parity.R: 2D extraction with iterators, complex
# expressions, multiple tracks, virtual tracks, gscreen, giterator_intervals
# ---------------------------------------------------------------------------

class TestGextract2dIterator:
    """Test using 2D intervals / tracks as iterators in gextract (R parity)."""

    def test_saved_2d_intervals_as_iterator(self):
        """Saved 2D interval set can be used as iterator in gextract.

        Ported from R test-gintervals-2d-indexed.R: 'indexed 2d intervals
        can be used as iterator in gextract'.
        """
        tname = "test.test_2d_iter_save"
        iset_name = "test.test_2d_iter_iset"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 5000],
                ends1=[200, 2000, 6000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 7000],
                ends2=[400, 4000, 8000],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "iter test track", intervals, values)

            # Extract using the track as its own iterator
            result1 = pm.gextract(tname, intervals, iterator=tname)
            assert result1 is not None
            assert len(result1) == 3

            # Save intervals, reload, use as iterator
            pm.gintervals_save(intervals, iset_name)
            result2 = pm.gextract(tname, intervals, iterator=iset_name)
            assert result2 is not None
            assert len(result2) == 3

            # Results should match
            vals1 = sorted(result1[tname].tolist())
            vals2 = sorted(result2[tname].tolist())
            for v1, v2 in zip(vals1, vals2, strict=False):
                assert abs(v1 - v2) < 0.01
        finally:
            _cleanup_track(tname)
            pm.gintervals_rm(iset_name, force=True)

    def test_saved_2d_intervals_as_scope(self):
        """Saved 2D interval set can be used as intervals scope in gextract.

        Ported from R test-gintervals-2d-indexed.R: 'indexed 2d intervals
        can be used as intervals in gextract'.
        """
        tname = "test.test_2d_scope_save"
        iset_name = "test.test_2d_scope_iset"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 10000],
                ends1=[200, 20000],
                chroms2=["1", "1"],
                starts2=[300, 30000],
                ends2=[400, 40000],
            )
            values = [10.0, 20.0]
            pm.gtrack_2d_create(tname, "scope test", intervals, values)

            # Extract using DataFrame intervals
            result_df = pm.gextract(tname, intervals=intervals, iterator=tname)

            # Save intervals and extract using named set
            pm.gintervals_save(intervals, iset_name)
            result_named = pm.gextract(tname, intervals=iset_name, iterator=tname)

            assert result_df is not None
            assert result_named is not None
            assert len(result_df) == len(result_named)
        finally:
            _cleanup_track(tname)
            pm.gintervals_rm(iset_name, force=True)

    def test_2d_track_as_iterator(self):
        """A 2D track itself can serve as iterator.

        Ported from R test-2d-parity.R: '2D intervals work as iterators
        in gextract'.
        """
        tname = "test.test_2d_track_iter"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 1000, 50],
                ends1=[200, 2000, 150],
                chroms2=["1", "1", "2"],
                starts2=[300, 3000, 200],
                ends2=[400, 4000, 250],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "track iter test", intervals, values)

            # Use track as its own iterator
            query = pm.gintervals_2d_all()
            result = pm.gextract(tname, intervals=query, iterator=tname)
            assert result is not None
            assert len(result) == 3
            vals = sorted(result[tname].tolist())
            assert abs(vals[0] - 1.0) < 0.01
            assert abs(vals[1] - 2.0) < 0.01
            assert abs(vals[2] - 3.0) < 0.01
        finally:
            _cleanup_track(tname)


class TestGextract2dComplexExpressions:
    """Test complex expressions with 2D extraction (R parity)."""

    def test_arithmetic_expression_2d(self):
        """Arithmetic expression on 2D track values.

        Ported from R test-2d-parity.R: '2D intervals work with complex
        gextract expressions'.
        """
        tname = "test.test_2d_expr_arith"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 1000],
                ends1=[200, 2000],
                chroms2=["1", "1"],
                starts2=[300, 3000],
                ends2=[400, 4000],
            )
            values = [5.0, 10.0]
            pm.gtrack_2d_create(tname, "expr test", intervals, values)

            # Extract with arithmetic expression
            expr = f"{tname} * 2 + 5"
            result = pm.gextract(expr, intervals)
            assert result is not None
            assert len(result) == 2
            vals = sorted(result.iloc[:, -2].tolist())  # expression column
            assert abs(vals[0] - 15.0) < 0.1  # 5*2+5=15
            assert abs(vals[1] - 25.0) < 0.1  # 10*2+5=25
        finally:
            _cleanup_track(tname)

    def test_multiple_tracks_2d(self):
        """Extract multiple 2D tracks simultaneously.

        Ported from R test-2d-parity.R: '2D intervals work with multiple
        tracks in gextract'.
        """
        tname1 = "test.test_2d_multi1"
        tname2 = "test.test_2d_multi2"
        _cleanup_track(tname1)
        _cleanup_track(tname2)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 1000, 50],
                ends1=[200, 2000, 150],
                chroms2=["1", "1", "2"],
                starts2=[300, 3000, 200],
                ends2=[400, 4000, 250],
            )
            pm.gtrack_2d_create(tname1, "Track 1", intervals, [1.0, 2.0, 3.0])
            pm.gtrack_2d_create(tname2, "Track 2", intervals, [10.0, 20.0, 30.0])

            # Extract both tracks (need explicit iterator for multiple 2D tracks)
            result = pm.gextract(
                [tname1, tname2],
                intervals=intervals,
                iterator=tname1,
            )
            assert result is not None
            assert len(result) == 3
            assert tname1 in result.columns
            assert tname2 in result.columns

            # Values should correspond
            vals1 = sorted(result[tname1].tolist())
            vals2 = sorted(result[tname2].tolist())
            for i, (v1, v2) in enumerate(zip(vals1, vals2, strict=False)):
                assert abs(v1 - (i + 1)) < 0.01
                assert abs(v2 - (i + 1) * 10) < 0.01
        finally:
            _cleanup_track(tname1)
            _cleanup_track(tname2)

    def test_multiple_tracks_2d_aligns_to_iterator_track(self):
        """With multiple 2D tracks, rows are anchored to the iterator track."""
        tname1 = "test.test_2d_multi_align1"
        tname2 = "test.test_2d_multi_align2"
        _cleanup_track(tname1)
        _cleanup_track(tname2)
        try:
            intervals1 = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 1000],
                ends1=[200, 1100],
                chroms2=["1", "1"],
                starts2=[300, 1300],
                ends2=[400, 1400],
            )
            intervals2 = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 2000],
                ends1=[200, 2100],
                chroms2=["1", "1"],
                starts2=[300, 2300],
                ends2=[400, 2400],
            )

            pm.gtrack_2d_create(tname1, "Track 1", intervals1, [1.0, 2.0])
            pm.gtrack_2d_create(tname2, "Track 2", intervals2, [10.0, 20.0])

            query = pm.gintervals_2d("1", 0, 5000, "1", 0, 5000)
            result = pm.gextract([tname1, tname2], intervals=query, iterator=tname1)
            assert result is not None
            assert len(result) == 2
            assert tname1 in result.columns
            assert tname2 in result.columns

            assert abs(result[tname1].iloc[0] - 1.0) < 0.01
            assert abs(result[tname2].iloc[0] - 10.0) < 0.01
            assert abs(result[tname1].iloc[1] - 2.0) < 0.01
            assert np.isnan(result[tname2].iloc[1])
        finally:
            _cleanup_track(tname1)
            _cleanup_track(tname2)


class TestGextract2dBandIntersect:
    """Test gintervals_2d_band_intersect directly (R parity)."""

    def test_band_intersect_basic(self):
        """Band intersect filters cis intervals by diagonal distance.

        Ported from R test-2d-parity.R: '2D intervals with
        gintervals.2d.band_intersect'.
        """
        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1", "1", "1", "1"],
            starts1=[100, 200, 300, 400, 500, 600],
            ends1=[150, 250, 350, 450, 550, 650],
            chroms2=["1", "1", "1", "1", "1", "1"],
            starts2=[200, 300, 400, 500, 600, 700],
            ends2=[250, 350, 450, 550, 650, 750],
        )
        # All intervals have start2 - end1 = 50, so diagonal distance
        # x1-y2 to x2-y1 ranges around -200 to -50 depending on exact calc
        band = (-300, 0)
        result = pm.gintervals_2d_band_intersect(intervals, band)
        assert result is not None
        assert len(result) > 0
        assert len(result) <= len(intervals)

    def test_band_intersect_removes_trans(self):
        """Band intersect removes trans (different chromosome) intervals."""
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[100, 100],
            ends1=[200, 200],
            chroms2=["1", "2"],
            starts2=[200, 100],
            ends2=[300, 200],
        )
        band = (-300, 300)
        result = pm.gintervals_2d_band_intersect(intervals, band)
        assert result is not None
        # Only the cis interval (chr1-chr1) should remain
        assert len(result) == 1
        assert result.iloc[0]["chrom1"] == result.iloc[0]["chrom2"]

    def test_band_intersect_negative_band(self):
        """Negative band range filters upstream contacts.

        Ported from R test-2d-parity.R: '2D intervals with negative band
        range'.
        """
        intervals = pm.gintervals_2d(
            chroms1=["1"] * 6,
            starts1=[1000, 2000, 3000, 4000, 5000, 6000],
            ends1=[1050, 2050, 3050, 4050, 5050, 6050],
            chroms2=["1"] * 6,
            starts2=[2000, 3000, 4000, 5000, 6000, 7000],
            ends2=[2050, 3050, 4050, 5050, 6050, 7050],
        )
        # start1 - start2 = -1000 for all intervals
        band = (-5000, -500)
        result = pm.gintervals_2d_band_intersect(intervals, band)
        assert result is not None
        assert len(result) == 6  # all should pass this wide negative band

    def test_band_intersect_distance_filtering(self):
        """Band intersect correctly filters by distance range.

        Ported from R test-2d-parity.R: '2D intervals with distance
        calculations'.
        """
        intervals = pm.gintervals_2d(
            chroms1=["1"] * 3,
            starts1=[1000, 1000, 1000],
            ends1=[1100, 1100, 1100],
            chroms2=["1"] * 3,
            starts2=[1500, 3000, 10000],
            ends2=[1600, 3100, 10100],
        )
        # Diagonal (x - y) ranges are approximately -500, -2000, -9000.
        # This band should keep only the middle interval (~-2000).
        band = (-2500, -1500)
        result = pm.gintervals_2d_band_intersect(intervals, band)
        assert result is not None
        assert len(result) == 1
        # Verify all results satisfy the rectangle/diagonal intersection rule.
        for _, row in result.iterrows():
            assert row["end1"] - row["start2"] > band[0]
            assert row["start1"] - row["end2"] + 1 < band[1]

    def test_band_intersect_empty_result(self):
        """Band intersect with non-matching band returns empty DataFrame."""
        intervals = pm.gintervals_2d(
            chroms1=["1"],
            starts1=[100],
            ends1=[200],
            chroms2=["1"],
            starts2=[300],
            ends2=[400],
        )
        # Very narrow band far from the interval's diagonal position
        band = (100000, 200000)
        result = pm.gintervals_2d_band_intersect(intervals, band)
        assert result is not None
        assert len(result) == 0

    def test_band_intersect_upper_triangle(self):
        """Band intersect to select upper triangle (positive diagonal).

        Ported from R test-2d-parity.R: expand.grid pattern upper
        triangle filtering.
        """
        # Create all-vs-all grid for a few bins
        starts = [100, 200, 300]
        ends = [150, 250, 350]
        chroms1, s1, e1, chroms2, s2, e2 = [], [], [], [], [], []
        for i in range(3):
            for j in range(3):
                chroms1.append("1")
                s1.append(starts[i])
                e1.append(ends[i])
                chroms2.append("1")
                s2.append(starts[j])
                e2.append(ends[j])

        intervals = pm.gintervals_2d(
            chroms1=chroms1, starts1=s1, ends1=e1,
            chroms2=chroms2, starts2=s2, ends2=e2,
        )
        # Upper triangle: start1 > start2 (positive x-y distance)
        band = (1, int(1e9))
        result = pm.gintervals_2d_band_intersect(intervals, band)
        assert result is not None
        # Should have fewer intervals than the full grid (exclude diagonal + lower)
        assert len(result) < len(intervals)

    def test_band_intersect_validation(self):
        """Band intersect validates d1 < d2."""
        intervals = pm.gintervals_2d(
            chroms1=["1"], starts1=[100], ends1=[200],
            chroms2=["1"], starts2=[300], ends2=[400],
        )
        with pytest.raises(ValueError):
            pm.gintervals_2d_band_intersect(intervals, (100, 50))

    def test_band_intersect_empty_input(self):
        """Band intersect on empty intervals returns empty."""
        empty = pm.gintervals_2d(
            chroms1=[], starts1=[], ends1=[],
            chroms2=[], starts2=[], ends2=[],
        )
        result = pm.gintervals_2d_band_intersect(empty, (-100, 100))
        assert len(result) == 0


class TestGextract2dWithBandOnExtract:
    """Test band parameter in gextract with 2D extraction (R parity)."""

    def test_band_extract_with_created_track(self):
        """Band filter on gextract with a freshly created 2D track.

        Ported from R test-2d-parity.R: '2D intervals work with band
        parameter in gextract'.
        """
        tname = "test.test_2d_band_extr"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"] * 6,
                starts1=[100, 200, 300, 400, 500, 600],
                ends1=[150, 250, 350, 450, 550, 650],
                chroms2=["1"] * 6,
                starts2=[200, 300, 400, 500, 600, 700],
                ends2=[250, 350, 450, 550, 650, 750],
            )
            values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            pm.gtrack_2d_create(tname, "band extract", intervals, values)

            # Extract without band
            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            all_result = pm.gextract(tname, query)
            assert all_result is not None
            assert len(all_result) == 6

            # Extract with restrictive band
            band_result = pm.gextract(tname, query, band=(-200, -50))
            if band_result is not None:
                assert len(band_result) <= 6
        finally:
            _cleanup_track(tname)


class TestGscreen2d:
    """Test gscreen with 2D tracks and intervals (R parity)."""

    def test_gscreen_2d_basic(self):
        """gscreen filters 2D track values by expression.

        Ported from R test-2d-parity.R: '2D intervals work with gscreen'.
        """
        tname = "test.test_2d_screen"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"] * 6,
                starts1=[100, 200, 300, 400, 500, 600],
                ends1=[150, 250, 350, 450, 550, 650],
                chroms2=["1"] * 6,
                starts2=[200, 300, 400, 500, 600, 700],
                ends2=[250, 350, 450, 550, 650, 750],
            )
            # Values span from 0.1 to 1.2
            values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
            pm.gtrack_2d_create(tname, "screen test", intervals, values)

            # Screen for values > 0.5
            result = pm.gscreen(f"{tname} > 0.5", intervals=intervals)
            if result is not None:
                # Should filter to only intervals where value > 0.5
                assert len(result) > 0
                assert len(result) < len(intervals)
        finally:
            _cleanup_track(tname)


class TestGiterator2d:
    """Test giterator_intervals with 2D intervals/tracks."""

    def test_giterator_with_2d_track(self):
        """giterator_intervals returns iterator bins for a 2D track.

        Ported from R test-2d-parity.R: '2D intervals work with
        giterator.intervals'.
        """
        tname = "test.test_2d_giter"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 1000, 50],
                ends1=[200, 2000, 150],
                chroms2=["1", "1", "2"],
                starts2=[300, 3000, 200],
                ends2=[400, 4000, 250],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "giter test", intervals, values)

            # Get iterator intervals using the 2D track as iterator
            result = pm.giterator_intervals(iterator=tname)
            assert result is not None
            assert len(result) == 3
            # Should have 2D columns
            assert "chrom1" in result.columns
            assert "start1" in result.columns
        finally:
            _cleanup_track(tname)


class TestVirtualTracks2d:
    """Test virtual tracks with 2D extraction (R parity)."""

    def test_vtrack_on_2d_track(self):
        """Virtual track wrapping a 2D track can be used in gextract.

        Ported from R test-2d-parity.R: '2D intervals work with virtual
        tracks'.
        """
        tname1 = "test.test_2d_vt1"
        tname2 = "test.test_2d_vt2"
        vt_name = "test_vtrack_2d"
        _cleanup_track(tname1)
        _cleanup_track(tname2)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 1000],
                ends1=[200, 2000],
                chroms2=["1", "1"],
                starts2=[300, 3000],
                ends2=[400, 4000],
            )
            pm.gtrack_2d_create(tname1, "VT Track 1", intervals, [1.0, 2.0])
            pm.gtrack_2d_create(tname2, "VT Track 2", intervals, [10.0, 20.0])

            # Create virtual track
            pm.gvtrack_create(vt_name, tname1)

            # Extract using virtual track + physical track expression
            result = pm.gextract(
                f"{vt_name} + {tname2}",
                intervals=intervals,
                iterator=tname1,
            )
            assert result is not None
            assert len(result) == 2
            # Values should be sum: 1+10=11, 2+20=22
            vals = sorted(result.iloc[:, -2].tolist())
            assert abs(vals[0] - 11.0) < 0.5
            assert abs(vals[1] - 22.0) < 0.5
        finally:
            pm.gvtrack_rm(vt_name)
            _cleanup_track(tname1)
            _cleanup_track(tname2)


class TestIntervals2dSaveLoad:
    """Test save/load/update of 2D interval sets (R parity)."""

    def test_save_and_load_2d_intervals(self):
        """Save and load 2D intervals round-trips correctly.

        Ported from R test-2d-parity.R: basic save and load parity check.
        """
        iset = "test.test_2d_saveload"
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "X"],
                starts1=[1000, 2000],
                ends1=[2000, 3000],
                chroms2=["1", "2"],
                starts2=[5000, 100],
                ends2=[6000, 500],
            )
            pm.gintervals_save(intervals, iset)

            loaded = pm.gintervals_load(iset)
            assert loaded is not None
            assert len(loaded) == 2
            assert "chrom1" in loaded.columns
            assert "start1" in loaded.columns
            assert "end1" in loaded.columns
            assert "chrom2" in loaded.columns
            assert "start2" in loaded.columns
            assert "end2" in loaded.columns
        finally:
            pm.gintervals_rm(iset, force=True)

    def test_save_load_sparse_chrom_pairs(self):
        """2D intervals with sparse chromosome pairs maintain data.

        Ported from R test-2d-parity.R: '2D intervals with sparse
        chromosome pairs maintain parity'.
        """
        iset = "test.test_2d_sparse_pairs"
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1", "1", "2", "2", "2", "2",
                          "2", "2", "2", "2"],
                starts1=[100, 200, 300, 400, 100, 200, 300, 400,
                          500, 600, 700, 800],
                ends1=[150, 250, 350, 450, 150, 250, 350, 450,
                        550, 650, 750, 850],
                chroms2=["1", "1", "1", "1", "2", "2", "2", "2",
                          "2", "2", "2", "2"],
                starts2=[500, 600, 700, 800, 200, 300, 400, 500,
                          600, 700, 800, 900],
                ends2=[550, 650, 750, 850, 250, 350, 450, 550,
                        650, 750, 850, 950],
            )
            pm.gintervals_save(intervals, iset)

            # Load all
            loaded = pm.gintervals_load(iset)
            assert loaded is not None
            assert len(loaded) == 12

            # Load specific chromosome pair
            loaded_11 = pm.gintervals_load(iset, chrom1="1", chrom2="1")
            if loaded_11 is not None:
                assert len(loaded_11) == 4

            loaded_22 = pm.gintervals_load(iset, chrom1="2", chrom2="2")
            if loaded_22 is not None:
                assert len(loaded_22) == 8
        finally:
            pm.gintervals_rm(iset, force=True)

    def test_save_load_all_chrom_combinations(self):
        """2D intervals with all chromosome pair combinations.

        Ported from R test-2d-parity.R: '2D intervals with all
        chromosome combinations'.
        """
        iset = "test.test_2d_all_chroms"
        try:
            # Create intervals for all possible pairs: 1-1, 1-2, 1-X, 2-2, 2-X, X-X
            chroms1 = (["1"] * 5 + ["1"] * 5 + ["1"] * 5 +
                       ["2"] * 5 + ["2"] * 5 + ["X"] * 5)
            chroms2 = (["1"] * 5 + ["2"] * 5 + ["X"] * 5 +
                       ["2"] * 5 + ["X"] * 5 + ["X"] * 5)
            starts1 = list(range(100, 600, 100)) * 6
            ends1 = list(range(150, 650, 100)) * 6
            starts2 = list(range(200, 700, 100)) * 6
            ends2 = list(range(250, 750, 100)) * 6

            intervals = pm.gintervals_2d(
                chroms1=chroms1, starts1=starts1, ends1=ends1,
                chroms2=chroms2, starts2=starts2, ends2=ends2,
            )
            pm.gintervals_save(intervals, iset)

            loaded = pm.gintervals_load(iset)
            assert loaded is not None
            assert len(loaded) == 30  # 6 pairs * 5 intervals

            # Test loading specific pairs
            pairs = [("1", "1"), ("1", "2"), ("1", "X"),
                     ("2", "2"), ("2", "X"), ("X", "X")]
            for c1, c2 in pairs:
                pair_data = pm.gintervals_load(iset, chrom1=c1, chrom2=c2)
                if pair_data is not None:
                    assert len(pair_data) == 5, (
                        f"Expected 5 intervals for pair {c1}-{c2}, "
                        f"got {len(pair_data)}"
                    )
        finally:
            pm.gintervals_rm(iset, force=True)

    def test_single_interval_per_pair(self):
        """Edge case: single interval in the pair.

        Ported from R test-2d-parity.R: '2D intervals edge cases: single
        interval per pair'.
        """
        iset = "test.test_2d_single"
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[150],
                chroms2=["1"],
                starts2=[200],
                ends2=[250],
            )
            pm.gintervals_save(intervals, iset)
            loaded = pm.gintervals_load(iset)
            assert loaded is not None
            assert len(loaded) == 1
        finally:
            pm.gintervals_rm(iset, force=True)


class TestIntervals2dChromSizes:
    """Test gintervals_chrom_sizes with 2D intervals (R parity)."""

    def test_chrom_sizes_2d(self):
        """gintervals_chrom_sizes returns unique chromosomes from 2D intervals.

        Ported from R test-2d-parity.R: '2D intervals
        gintervals.chrom_sizes produces identical results'.
        """
        intervals = pm.gintervals_2d(
            chroms1=["1", "1", "1", "1", "1", "1", "1", "1",
                      "1", "1", "1", "1"],
            starts1=[100, 200, 300, 400] * 3,
            ends1=[150, 250, 350, 450] * 3,
            chroms2=["1", "1", "1", "1", "2", "2", "2", "2",
                      "2", "2", "2", "2"],
            starts2=[300, 100, 400, 500] * 3,
            ends2=[350, 150, 450, 550] * 3,
        )
        sizes = pm.gintervals_chrom_sizes(intervals)
        assert sizes is not None
        assert "chrom" in sizes.columns
        # Should have chroms 1 and 2
        chrom_set = set(sizes["chrom"].tolist())
        assert "1" in chrom_set or "chr1" in chrom_set


class TestLargeScale2d:
    """Test large-scale 2D operations (R parity)."""

    def test_large_2d_track_creation_and_extraction(self):
        """Create and extract from a larger 2D track.

        Ported from R test-2d-parity.R: '2D intervals large-scale
        operations'.
        """
        tname = "test.test_2d_large"
        _cleanup_track(tname)
        try:
            n = 50
            rng = np.random.RandomState(123)
            # Create grid of contacts on chr1
            starts1 = np.arange(1000, 1000 + n * 1000, 1000)
            starts2 = np.arange(1000, 1000 + n * 1000, 1000)

            # Create upper triangle contacts
            c1_list, s1_list, e1_list, c2_list, s2_list, e2_list, val_list = (
                [], [], [], [], [], [], [],
            )
            for i in range(n):
                for j in range(i, n):
                    c1_list.append("1")
                    s1_list.append(int(starts1[i]))
                    e1_list.append(int(starts1[i] + 100))
                    c2_list.append("1")
                    s2_list.append(int(starts2[j]))
                    e2_list.append(int(starts2[j] + 100))
                    val_list.append(float(rng.uniform(0, 10)))

            intervals = pm.gintervals_2d(
                chroms1=c1_list, starts1=s1_list, ends1=e1_list,
                chroms2=c2_list, starts2=s2_list, ends2=e2_list,
            )
            pm.gtrack_2d_create(tname, "large scale", intervals, val_list)

            # Extract all
            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            result = pm.gextract(tname, query)
            assert result is not None
            assert len(result) == len(val_list)

            # Extract with band
            band_result = pm.gextract(tname, query, band=(1000, 10000))
            if band_result is not None:
                assert len(band_result) < len(val_list)
                assert len(band_result) > 0
        finally:
            _cleanup_track(tname)

    def test_per_chrom_loading(self):
        """Load 2D intervals per chromosome pair.

        Ported from R test-2d-parity.R: '2D intervals with per-chromosome
        loading'.
        """
        tname = "test.test_2d_perchrom_load"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"] * 3 + ["2"] * 3,
                starts1=[100, 200, 300, 100, 200, 300],
                ends1=[150, 250, 350, 150, 250, 350],
                chroms2=["1"] * 3 + ["2"] * 3,
                starts2=[400, 500, 600, 400, 500, 600],
                ends2=[450, 550, 650, 450, 550, 650],
            )
            values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            pm.gtrack_2d_create(tname, "per chrom", intervals, values)

            # Extract only chr1-chr1
            query_1 = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            result_1 = pm.gextract(tname, query_1)
            assert result_1 is not None
            assert len(result_1) == 3

            # Extract only chr2-chr2
            query_2 = pm.gintervals_2d("2", 0, 300000, "2", 0, 300000)
            result_2 = pm.gextract(tname, query_2)
            assert result_2 is not None
            assert len(result_2) == 3
        finally:
            _cleanup_track(tname)
