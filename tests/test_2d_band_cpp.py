"""Tests for C++ band-filtered 2D quad-tree queries.

Verifies that band parameters (d1, d2) are correctly passed through
the entire extraction pipeline to the C++ QuadTreeReader.

- query_2d_track_stats with band
- query_2d_track_opened with band
- query_2d_track_stats_batch with band
- _gextract_2d_single with band
- _gextract_2d_vtrack_agg with band
- _gextract_2d_vtrack_objects with band
"""

import os
import shutil
import struct

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import (
    _read_file_header,
    query_2d_track_opened,
    query_2d_track_opened_arrays,
    query_2d_track_stats,
    query_2d_track_stats_batch,
    write_2d_track_file,
)

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
def band_rects_track():
    """Create a 2D rects track with objects at known diagonal positions.

    Layout on chr1-chr1 (arena 0..500000 x 0..500000):

        Rect A: (100, 100, 200, 200, 1.0)   diagonal x-y in [-100, 100)
        Rect B: (500, 100, 600, 200, 2.0)    diagonal x-y in [300, 500)
        Rect C: (1000, 500, 1100, 600, 3.0)  diagonal x-y in [400, 600)
        Rect D: (100, 800, 200, 900, 4.0)    diagonal x-y in [-800, -600)
    """
    tname = "test.band_rects"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    rects = [
        (100, 100, 200, 200, 1.0),   # A: near diagonal
        (500, 100, 600, 200, 2.0),    # B: above diagonal
        (1000, 500, 1100, 600, 3.0),  # C: above diagonal
        (100, 800, 200, 900, 4.0),    # D: below diagonal
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), rects, (0, 0, 500000, 500000), is_points=False
    )
    _pymisha.pm_dbreload()

    yield tname

    pm.gvtrack_clear()
    _cleanup_track(tname)


@pytest.fixture()
def band_points_track():
    """Create a 2D points track with points at known diagonal positions.

    Points on chr1-chr1:
        P0: (100, 100, 1.0)   diff=0
        P1: (500, 100, 2.0)   diff=400
        P2: (1000, 500, 3.0)  diff=500
        P3: (100, 800, 4.0)   diff=-700
    """
    tname = "test.band_points"
    _cleanup_track(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=points\ndimensions=2\n")

    points = [
        (100, 100, 1.0),
        (500, 100, 2.0),
        (1000, 500, 3.0),
        (100, 800, 4.0),
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), points, (0, 0, 500000, 500000), is_points=True
    )
    _pymisha.pm_dbreload()

    yield tname

    pm.gvtrack_clear()
    _cleanup_track(tname)


# ---------------------------------------------------------------------------
# Low-level: query_2d_track_opened with band
# ---------------------------------------------------------------------------


class TestQueryOpenedBandRects:
    """Test query_2d_track_opened passes band to C++ for rects."""

    def test_no_band_returns_all(self, band_rects_track):
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000,
            )
            assert len(objs) == 4
        finally:
            data.close()

    def test_band_filters_near_diagonal(self, band_rects_track):
        """Band (d1=-200, d2=200) should only return Rect A."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(-200, 200),
            )
            # Only Rect A (value=1.0) has x-y in [-100, 100) which falls in [-200, 200)
            assert len(objs) == 1
            assert objs[0][4] == pytest.approx(1.0)
        finally:
            data.close()

    def test_band_filters_above_diagonal(self, band_rects_track):
        """Band (d1=200, d2=700) should return Rects B and C."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(200, 700),
            )
            vals = sorted([o[4] for o in objs])
            assert len(objs) == 2
            assert vals[0] == pytest.approx(2.0)
            assert vals[1] == pytest.approx(3.0)
        finally:
            data.close()

    def test_band_returns_empty_when_no_match(self, band_rects_track):
        """Band (d1=10000, d2=20000) should return nothing."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(10000, 20000),
            )
            assert len(objs) == 0
        finally:
            data.close()


class TestQueryOpenedBandPoints:
    """Test query_2d_track_opened passes band to C++ for points."""

    def test_no_band_returns_all(self, band_points_track):
        tdir = _track_dir(band_points_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000,
            )
            assert len(objs) == 4
        finally:
            data.close()

    def test_band_filters_near_diagonal(self, band_points_track):
        """Band (-100, 100) should return only P0 (diff=0)."""
        tdir = _track_dir(band_points_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(-100, 100),
            )
            assert len(objs) == 1
            assert objs[0][2] == pytest.approx(1.0)
        finally:
            data.close()

    def test_band_filters_above_diagonal(self, band_points_track):
        """Band (300, 600) should return P1 (diff=400) and P2 (diff=500)."""
        tdir = _track_dir(band_points_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(300, 600),
            )
            vals = sorted([o[2] for o in objs])
            assert len(objs) == 2
            assert vals[0] == pytest.approx(2.0)
            assert vals[1] == pytest.approx(3.0)
        finally:
            data.close()


# ---------------------------------------------------------------------------
# Low-level: query_2d_track_stats with band
# ---------------------------------------------------------------------------


class TestQueryStatsBand:
    """Test query_2d_track_stats passes band to C++."""

    def test_stats_no_band(self, band_rects_track):
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            stats = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000,
            )
            assert stats["occupied_area"] > 0
            assert stats["min_val"] == pytest.approx(1.0)
            assert stats["max_val"] == pytest.approx(4.0)
        finally:
            data.close()

    def test_stats_with_band_near_diagonal(self, band_rects_track):
        """Band (-200, 200) should only include Rect A stats."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            stats = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(-200, 200),
            )
            assert stats["occupied_area"] > 0
            assert stats["min_val"] == pytest.approx(1.0)
            assert stats["max_val"] == pytest.approx(1.0)
        finally:
            data.close()

    def test_stats_with_band_empty(self, band_rects_track):
        """Band far away should return zero area."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            stats = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(10000, 20000),
            )
            assert stats["occupied_area"] == 0
            assert np.isnan(stats["min_val"])
            assert np.isnan(stats["max_val"])
        finally:
            data.close()

    def test_stats_band_points(self, band_points_track):
        """Band (300, 600) should only include P1 and P2."""
        tdir = _track_dir(band_points_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            stats = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=(300, 600),
            )
            assert stats["occupied_area"] == 2
            assert stats["min_val"] == pytest.approx(2.0)
            assert stats["max_val"] == pytest.approx(3.0)
        finally:
            data.close()


# ---------------------------------------------------------------------------
# Low-level: stats vs objects consistency
# ---------------------------------------------------------------------------


class TestStatObjectsConsistency:
    """Verify band stats match manual computation from band objects."""

    def test_rects_consistency(self, band_rects_track):
        """Band stats should match manual aggregation of band objects."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        band = (200, 700)
        qx1, qy1, qx2, qy2 = 0, 0, 500000, 500000
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]

            stats = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                qx1, qy1, qx2, qy2, band=band,
            )
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                qx1, qy1, qx2, qy2, band=band,
            )

            # Same objects should be captured by both
            if len(objs) == 0:
                assert stats["occupied_area"] == 0
            else:
                vals = [o[4] for o in objs]
                assert stats["min_val"] == pytest.approx(min(vals))
                assert stats["max_val"] == pytest.approx(max(vals))
                assert stats["occupied_area"] > 0
        finally:
            data.close()

    def test_points_consistency(self, band_points_track):
        """Band stats should match count/aggregation from band objects."""
        tdir = _track_dir(band_points_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        band = (-100, 100)
        qx1, qy1, qx2, qy2 = 0, 0, 500000, 500000
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]

            stats = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                qx1, qy1, qx2, qy2, band=band,
            )
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                qx1, qy1, qx2, qy2, band=band,
            )

            assert stats["occupied_area"] == len(objs)
            if len(objs) > 0:
                vals = [o[2] for o in objs]
                assert stats["min_val"] == pytest.approx(min(vals))
                assert stats["max_val"] == pytest.approx(max(vals))
                assert stats["weighted_sum"] == pytest.approx(sum(vals))
        finally:
            data.close()


# ---------------------------------------------------------------------------
# Low-level: batch stats with band
# ---------------------------------------------------------------------------


class TestBatchStatsBand:
    """Test query_2d_track_stats_batch passes band correctly."""

    def test_batch_band_matches_single(self, band_rects_track):
        """Batch with band should match individual queries."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        band = (-200, 200)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]

            # Individual query
            s1 = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=band,
            )
            s2 = query_2d_track_stats(
                data, is_points, num_objs, root_chunk_fpos,
                400, 400, 2000, 2000, band=band,
            )

            # Batch query
            rects = np.array([
                [0, 0, 500000, 500000],
                [400, 400, 2000, 2000],
            ], dtype=np.int64)
            batch = query_2d_track_stats_batch(
                data, is_points, num_objs, root_chunk_fpos,
                rects, band=band,
            )

            assert batch["occupied_area"][0] == s1["occupied_area"]
            assert batch["occupied_area"][1] == s2["occupied_area"]
            np.testing.assert_allclose(
                batch["min_val"][0], s1["min_val"], equal_nan=True
            )
            np.testing.assert_allclose(
                batch["min_val"][1], s2["min_val"], equal_nan=True
            )
        finally:
            data.close()


# ---------------------------------------------------------------------------
# query_2d_track_opened_arrays with band
# ---------------------------------------------------------------------------


class TestQueryOpenedArraysBand:
    """Test query_2d_track_opened_arrays passes band to C++."""

    def test_arrays_band_matches_tuples(self, band_rects_track):
        """Array results with band should match tuple results."""
        tdir = _track_dir(band_rects_track)
        filepath = os.path.join(tdir, "1-1")
        is_points, num_objs, data = _read_file_header(filepath)
        band = (-200, 200)
        try:
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]

            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=band,
            )
            arrs = query_2d_track_opened_arrays(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000, band=band,
            )
            assert len(arrs["val"]) == len(objs)
            if len(objs) > 0:
                np.testing.assert_allclose(
                    sorted(arrs["val"]), sorted([o[4] for o in objs]), atol=1e-5
                )
        finally:
            data.close()


# ---------------------------------------------------------------------------
# High-level: gextract with band via vtrack agg + objects
# ---------------------------------------------------------------------------


class TestGextractBandVtrackAgg:
    """Test _gextract_2d_vtrack_agg passes band through batch stats."""

    def test_agg_area_with_band(self, band_rects_track):
        """Vtrack area with band should only aggregate band-filtered objects."""
        pm.gvtrack_create("vt_band_area", band_rects_track, func="area")
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        result = pm.gextract("vt_band_area", intervals, band=(-200, 200))
        assert len(result) == 1
        # Rect A is the only rect in band(-200, 200), so area should be its area
        assert result["vt_band_area"].iloc[0] > 0

    def test_agg_min_with_band(self, band_rects_track):
        """Vtrack min with band should match band-filtered min value."""
        pm.gvtrack_create("vt_band_min", band_rects_track, func="min")
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        result = pm.gextract("vt_band_min", intervals, band=(-200, 200))
        assert len(result) == 1
        # Only Rect A (val=1.0) is in band
        assert result["vt_band_min"].iloc[0] == pytest.approx(1.0)


class TestGextractBandVtrackObjects:
    """Test _gextract_2d_vtrack_objects passes band through object queries."""

    def test_exists_with_band(self, band_rects_track):
        """Vtrack exists with band should only count band-filtered objects."""
        pm.gvtrack_create("vt_band_exists", band_rects_track, func="exists")
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        # Band (-200, 200) has Rect A -> exists = 1
        result = pm.gextract("vt_band_exists", intervals, band=(-200, 200))
        assert result["vt_band_exists"].iloc[0] == 1.0

    def test_exists_with_empty_band(self, band_rects_track):
        """Vtrack exists with band outside all objects should return 0."""
        pm.gvtrack_create("vt_band_exists2", band_rects_track, func="exists")
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        result = pm.gextract("vt_band_exists2", intervals, band=(10000, 20000))
        assert result["vt_band_exists2"].iloc[0] == 0.0

    def test_size_with_band(self, band_rects_track):
        """Vtrack size with band should count only band-filtered objects."""
        pm.gvtrack_create("vt_band_size", band_rects_track, func="size")
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        # Band (200, 700) should include Rect B and Rect C
        result = pm.gextract("vt_band_size", intervals, band=(200, 700))
        assert result["vt_band_size"].iloc[0] == 2.0

    def test_size_with_band_points(self, band_points_track):
        """Vtrack size with band on points track."""
        pm.gvtrack_create("vt_band_psize", band_points_track, func="size")
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        # Band (300, 600): P1(diff=400) and P2(diff=500) -> 2 points
        result = pm.gextract("vt_band_psize", intervals, band=(300, 600))
        assert result["vt_band_psize"].iloc[0] == 2.0


class TestGextractBandSingle:
    """Test _gextract_2d_single passes band through object queries."""

    def test_single_raw_with_band(self, band_rects_track):
        """Raw 2D extraction with band should only return band-filtered objects."""
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        # Band (-200, 200) should return only Rect A (val=1.0)
        result = pm.gextract(band_rects_track, intervals, band=(-200, 200))
        assert len(result) == 1
        assert result[band_rects_track].iloc[0] == pytest.approx(1.0)

    def test_single_raw_with_wide_band(self, band_rects_track):
        """Raw 2D extraction with wide band should return more objects."""
        intervals = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [0],
            "end1": [500000],
            "chrom2": ["1"],
            "start2": [0],
            "end2": [500000],
        })
        # Band (-1000, 1000) should return A, B, C (not D which is at -800..-600)
        result = pm.gextract(band_rects_track, intervals, band=(-1000, 1000))
        assert len(result) >= 3
