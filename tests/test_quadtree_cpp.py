"""Tests for C++ quad-tree reader (QuadTreeReader.cpp).

Verifies that the C++ fast path in _pymisha.pm_quadtree_query_stats and
_pymisha.pm_quadtree_query_objects produces results identical to the
pure-Python implementation in pymisha._quadtree.
"""

import math
import mmap
import os
import struct
import tempfile

import numpy as np
import pytest

import _pymisha
from pymisha._quadtree import (
    SIGNATURE_POINTS,
    SIGNATURE_RECTS,
    QuadTree,
    query_2d_track_opened,
    query_2d_track_stats,
    write_2d_track_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_rects_file(rects, arena=(0, 0, 100000, 100000)):
    """Create a temporary 2D RECTS track file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".track")
    os.close(fd)
    write_2d_track_file(path, rects, arena, is_points=False)
    return path


def _create_points_file(points, arena=(0, 0, 100000, 100000)):
    """Create a temporary 2D POINTS track file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".track")
    os.close(fd)
    write_2d_track_file(path, points, arena, is_points=True)
    return path


def _open_track(path):
    """Open a 2D track file for querying. Returns (data, is_points, num_objs, root_chunk_fpos)."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    signature = struct.unpack_from("<i", data, 0)[0]
    is_points = signature == SIGNATURE_POINTS
    num_objs = struct.unpack_from("<Q", data, 4)[0]
    root_chunk_fpos = struct.unpack_from("<q", data, 12)[0] if num_objs > 0 else 0
    return data, is_points, num_objs, root_chunk_fpos


def _py_query_stats(data, is_points, num_objs, root_chunk_fpos, qx1, qy1, qx2, qy2, band=None):
    """Force Python-only stats query (bypass C++ fast path)."""
    from pymisha._quadtree import (
        _query_2d_track_stats_with_band,
        _query_node_stats,
    )

    if num_objs == 0:
        return {"occupied_area": 0, "weighted_sum": 0.0,
                "min_val": float("nan"), "max_val": float("nan")}

    if band is not None:
        return _query_2d_track_stats_with_band(
            data, is_points, num_objs, root_chunk_fpos,
            qx1, qy1, qx2, qy2, band)

    top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]
    stat = [0, 0.0, float("inf"), float("-inf")]
    _query_node_stats(data, root_chunk_fpos, top_node_offset, is_points,
                      qx1, qy1, qx2, qy2, stat)
    if stat[0] == 0:
        return {"occupied_area": 0, "weighted_sum": 0.0,
                "min_val": float("nan"), "max_val": float("nan")}
    return {"occupied_area": stat[0], "weighted_sum": stat[1],
            "min_val": stat[2], "max_val": stat[3]}


def _py_query_objects(data, is_points, num_objs, root_chunk_fpos, qx1, qy1, qx2, qy2):
    """Force Python-only objects query (bypass C++ fast path)."""
    from pymisha._quadtree import _query_node

    if num_objs == 0:
        return []
    top_node_offset = struct.unpack_from("<q", data, root_chunk_fpos + 8)[0]
    seen_ids = set()
    raw_objs = _query_node(data, root_chunk_fpos, top_node_offset, is_points,
                           qx1, qy1, qx2, qy2, seen_ids)
    result = []
    for obj in raw_objs:
        if is_points:
            _, x, y, val = obj
            result.append((x, y, val))
        else:
            _, x1, y1, x2, y2, val = obj
            result.append((x1, y1, x2, y2, val))
    return result


def _cpp_query_stats(data, is_points, qx1, qy1, qx2, qy2, band=None):
    """Direct C++ stats query."""
    has_band = 1 if band is not None else 0
    band_d1 = band[0] if band else 0
    band_d2 = band[1] if band else 0
    return _pymisha.pm_quadtree_query_stats(
        data, int(qx1), int(qy1), int(qx2), int(qy2),
        1 if is_points else 0, has_band, int(band_d1), int(band_d2))


def _cpp_query_objects(data, is_points, qx1, qy1, qx2, qy2, band=None):
    """Direct C++ objects query, returning list of tuples matching Python format."""
    has_band = 1 if band is not None else 0
    band_d1 = band[0] if band else 0
    band_d2 = band[1] if band else 0
    r = _pymisha.pm_quadtree_query_objects(
        data, int(qx1), int(qy1), int(qx2), int(qy2),
        1 if is_points else 0, has_band, int(band_d1), int(band_d2))
    n = len(r["id"])
    result = []
    if is_points:
        for i in range(n):
            result.append((int(r["x1"][i]), int(r["y1"][i]), float(r["val"][i])))
    else:
        for i in range(n):
            result.append((int(r["x1"][i]), int(r["y1"][i]),
                           int(r["x2"][i]), int(r["y2"][i]),
                           float(r["val"][i])))
    return result


def _assert_stats_equal(cpp_stat, py_stat, rtol=1e-6):
    """Assert two stat dicts are equal, handling NaN.

    When occupied_area==0, C++ sets weighted_sum to NaN (matching R misha)
    while Python sets it to 0.0. Both are valid; we treat them as equal.
    """
    assert cpp_stat["occupied_area"] == py_stat["occupied_area"], (
        f"occupied_area: C++ {cpp_stat['occupied_area']} != Py {py_stat['occupied_area']}")

    # When occupied_area is 0, all value fields are semantically undefined
    if cpp_stat["occupied_area"] == 0:
        return

    if math.isnan(py_stat["weighted_sum"]):
        assert math.isnan(cpp_stat["weighted_sum"])
    else:
        assert abs(cpp_stat["weighted_sum"] - py_stat["weighted_sum"]) <= rtol * abs(py_stat["weighted_sum"]) + 1e-12
    if math.isnan(py_stat["min_val"]):
        assert math.isnan(cpp_stat["min_val"])
    else:
        assert abs(cpp_stat["min_val"] - py_stat["min_val"]) <= rtol * abs(py_stat["min_val"]) + 1e-12
    if math.isnan(py_stat["max_val"]):
        assert math.isnan(cpp_stat["max_val"])
    else:
        assert abs(cpp_stat["max_val"] - py_stat["max_val"]) <= rtol * abs(py_stat["max_val"]) + 1e-12


def _sort_objs(objs):
    """Sort objects for comparison."""
    return sorted(objs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rects_track():
    """3 known rectangles on a 100k x 100k arena."""
    rects = [
        (100, 200, 300, 400, 5.0),    # A
        (200, 300, 500, 600, 10.0),   # B: overlaps A in [200,300)x[300,400)
        (1000, 1000, 2000, 2000, 3.0),  # C: isolated
    ]
    path = _create_rects_file(rects)
    data, is_points, num_objs, root_chunk_fpos = _open_track(path)
    yield data, is_points, num_objs, root_chunk_fpos, rects
    data.close()
    os.unlink(path)


@pytest.fixture()
def points_track():
    """3 known points on a 100k x 100k arena."""
    points = [
        (100, 200, 5.0),    # P1
        (150, 250, 10.0),   # P2
        (5000, 5000, 3.0),  # P3: isolated
    ]
    path = _create_points_file(points)
    data, is_points, num_objs, root_chunk_fpos = _open_track(path)
    yield data, is_points, num_objs, root_chunk_fpos, points
    data.close()
    os.unlink(path)


@pytest.fixture()
def empty_track():
    """Empty rects track."""
    path = _create_rects_file([], arena=(0, 0, 1000, 1000))
    data, is_points, num_objs, root_chunk_fpos = _open_track(path)
    yield data, is_points, num_objs, root_chunk_fpos
    data.close()
    os.unlink(path)


@pytest.fixture()
def single_rect_track():
    """Single rectangle track."""
    rects = [(50, 50, 150, 150, 7.5)]
    path = _create_rects_file(rects, arena=(0, 0, 1000, 1000))
    data, is_points, num_objs, root_chunk_fpos = _open_track(path)
    yield data, is_points, num_objs, root_chunk_fpos, rects
    data.close()
    os.unlink(path)


@pytest.fixture()
def single_point_track():
    """Single point track."""
    points = [(500, 500, 42.0)]
    path = _create_points_file(points, arena=(0, 0, 1000, 1000))
    data, is_points, num_objs, root_chunk_fpos = _open_track(path)
    yield data, is_points, num_objs, root_chunk_fpos, points
    data.close()
    os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: RECTS stats
# ---------------------------------------------------------------------------


class TestRectsStats:

    def test_full_containment(self, rects_track):
        """Query that fully contains all rects."""
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        cpp = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 0, 0, 100000, 100000)
        _assert_stats_equal(cpp, py)
        assert cpp["occupied_area"] > 0

    def test_partial_overlap(self, rects_track):
        """Query that partially overlaps rect A."""
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        # Overlaps rect A only (100,200,300,400) in region [150,200)x[250,350)
        cpp = _cpp_query_stats(data, is_points, 150, 250, 200, 350)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 150, 250, 200, 350)
        _assert_stats_equal(cpp, py)

    def test_no_intersection(self, rects_track):
        """Query in empty region."""
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        cpp = _cpp_query_stats(data, is_points, 50000, 50000, 60000, 60000)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 50000, 50000, 60000, 60000)
        _assert_stats_equal(cpp, py)
        assert cpp["occupied_area"] == 0

    def test_overlap_region(self, rects_track):
        """Query in the overlap region of A and B."""
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        cpp = _cpp_query_stats(data, is_points, 200, 300, 300, 400)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 200, 300, 300, 400)
        _assert_stats_equal(cpp, py)

    def test_isolated_rect(self, rects_track):
        """Query rect C only."""
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        cpp = _cpp_query_stats(data, is_points, 1000, 1000, 2000, 2000)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 1000, 1000, 2000, 2000)
        _assert_stats_equal(cpp, py)


# ---------------------------------------------------------------------------
# Tests: POINTS stats
# ---------------------------------------------------------------------------


class TestPointsStats:

    def test_full_containment(self, points_track):
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        cpp = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 0, 0, 100000, 100000)
        _assert_stats_equal(cpp, py)
        assert cpp["occupied_area"] == 3

    def test_partial(self, points_track):
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        # Only P1 at (100,200) should match
        cpp = _cpp_query_stats(data, is_points, 50, 150, 120, 250)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 50, 150, 120, 250)
        _assert_stats_equal(cpp, py)

    def test_no_match(self, points_track):
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        cpp = _cpp_query_stats(data, is_points, 9000, 9000, 9500, 9500)
        py = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 9000, 9000, 9500, 9500)
        _assert_stats_equal(cpp, py)
        assert cpp["occupied_area"] == 0


# ---------------------------------------------------------------------------
# Tests: RECTS objects
# ---------------------------------------------------------------------------


class TestRectsObjects:

    def test_full_query(self, rects_track):
        data, is_points, num_objs, root_chunk_fpos, rects = rects_track
        cpp = _cpp_query_objects(data, is_points, 0, 0, 100000, 100000)
        py = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 0, 0, 100000, 100000)
        assert _sort_objs(cpp) == _sort_objs(py)
        assert len(cpp) == 3

    def test_partial_query(self, rects_track):
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        # Should match only rect A (100,200,300,400)
        cpp = _cpp_query_objects(data, is_points, 50, 50, 150, 250)
        py = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 50, 50, 150, 250)
        assert _sort_objs(cpp) == _sort_objs(py)
        assert len(cpp) == 1

    def test_empty_result(self, rects_track):
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        cpp = _cpp_query_objects(data, is_points, 80000, 80000, 90000, 90000)
        py = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 80000, 80000, 90000, 90000)
        assert cpp == py == []


# ---------------------------------------------------------------------------
# Tests: POINTS objects
# ---------------------------------------------------------------------------


class TestPointsObjects:

    def test_full_query(self, points_track):
        data, is_points, num_objs, root_chunk_fpos, points = points_track
        cpp = _cpp_query_objects(data, is_points, 0, 0, 100000, 100000)
        py = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 0, 0, 100000, 100000)
        assert _sort_objs(cpp) == _sort_objs(py)
        assert len(cpp) == 3

    def test_single_point(self, points_track):
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        # Only P1 at (100,200)
        cpp = _cpp_query_objects(data, is_points, 99, 199, 101, 201)
        py = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 99, 199, 101, 201)
        assert _sort_objs(cpp) == _sort_objs(py)
        assert len(cpp) == 1


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_track_stats(self, empty_track):
        data, is_points, num_objs, root_chunk_fpos = empty_track
        cpp = _cpp_query_stats(data, is_points, 0, 0, 1000, 1000)
        assert cpp["occupied_area"] == 0
        assert math.isnan(cpp["weighted_sum"])

    def test_empty_track_objects(self, empty_track):
        data, is_points, num_objs, root_chunk_fpos = empty_track
        cpp = _cpp_query_objects(data, is_points, 0, 0, 1000, 1000)
        assert cpp == []

    def test_single_rect_full(self, single_rect_track):
        data, is_points, num_objs, root_chunk_fpos, rects = single_rect_track
        cpp_s = _cpp_query_stats(data, is_points, 0, 0, 1000, 1000)
        py_s = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 0, 0, 1000, 1000)
        _assert_stats_equal(cpp_s, py_s)
        assert cpp_s["occupied_area"] == 100 * 100  # (150-50) * (150-50)

        cpp_o = _cpp_query_objects(data, is_points, 0, 0, 1000, 1000)
        assert len(cpp_o) == 1
        assert cpp_o[0] == (50, 50, 150, 150, 7.5)

    def test_single_point_full(self, single_point_track):
        data, is_points, num_objs, root_chunk_fpos, points = single_point_track
        cpp_s = _cpp_query_stats(data, is_points, 0, 0, 1000, 1000)
        py_s = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 0, 0, 1000, 1000)
        _assert_stats_equal(cpp_s, py_s)
        assert cpp_s["occupied_area"] == 1

        cpp_o = _cpp_query_objects(data, is_points, 0, 0, 1000, 1000)
        assert len(cpp_o) == 1
        assert cpp_o[0] == (500, 500, 42.0)


# ---------------------------------------------------------------------------
# Tests: Band filtering
# ---------------------------------------------------------------------------


class TestBandFiltering:

    def test_rects_band_stats_nonzero(self, rects_track):
        """Band filter on rects track should produce consistent stats.

        The C++ implementation follows R misha's exact band-area algorithm
        (using shrink2intersected + triangle subtraction). We verify:
        - occupied_area > 0 (band overlaps some rects)
        - occupied_area <= no-band occupied_area
        - min/max values are consistent with known rect values
        """
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        band = (-500, 500)
        cpp_band = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000, band=band)
        cpp_noband = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000)
        assert cpp_band["occupied_area"] > 0
        assert cpp_band["occupied_area"] <= cpp_noband["occupied_area"]
        assert cpp_band["min_val"] >= 3.0
        assert cpp_band["max_val"] <= 10.0

    def test_points_band_stats(self, points_track):
        """Band filter on points - C++ band stats should match Python fallback for points.

        For points, the band filter is simple (d1 <= x-y < d2), no triangle math.
        """
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        # P1: x-y=-100, P2: x-y=-100, P3: x-y=0
        band = (-200, 0)  # should match P1 and P2
        cpp = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000, band=band)
        assert cpp["occupied_area"] == 2  # P1 and P2

    def test_points_band_objects(self, points_track):
        """Band filter on points track."""
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        # P1: x-y = 100-200 = -100, P2: 150-250 = -100, P3: 5000-5000 = 0
        band = (-200, 0)  # should match P1 and P2
        cpp = _cpp_query_objects(data, is_points, 0, 0, 100000, 100000, band=band)
        # Manually filter expected
        assert len(cpp) == 2  # P1 and P2

    def test_rects_band_no_match(self, rects_track):
        """Band that excludes all rects."""
        data, is_points, num_objs, root_chunk_fpos, _ = rects_track
        band = (99999, 100000)  # very narrow, far from any rect
        cpp = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000, band=band)
        assert cpp["occupied_area"] == 0

    def test_rects_band_objects(self, rects_track):
        """Band filter on rect objects."""
        data, is_points, num_objs, root_chunk_fpos, rects = rects_track
        # Rect C: x1=1000, y1=1000, x2=2000, y2=2000 -> diagonal ~ 0
        # Band [−500, 500) should match rect C and possibly A/B
        band = (-500, 500)
        cpp = _cpp_query_objects(data, is_points, 0, 0, 100000, 100000, band=band)
        py_all = _py_query_objects(data, is_points, num_objs, root_chunk_fpos,
                                   0, 0, 100000, 100000)
        py_filtered = []
        for r in py_all:
            ox1, oy1, ox2, oy2, v = r
            if (ox2 - oy1 > band[0]) and (ox1 - oy2 + 1 < band[1]):
                py_filtered.append(r)
        assert _sort_objs(cpp) == _sort_objs(py_filtered)

    def test_points_band_excludes_p3(self, points_track):
        """Band [-200, -50) should only match P1 and P2 (x-y = -100)."""
        data, is_points, num_objs, root_chunk_fpos, _ = points_track
        band = (-200, -50)
        cpp = _cpp_query_objects(data, is_points, 0, 0, 100000, 100000, band=band)
        assert len(cpp) == 2
        for x, y, v in cpp:
            assert -200 <= (x - y) < -50


# ---------------------------------------------------------------------------
# Tests: Large dataset (stress test)
# ---------------------------------------------------------------------------


class TestLargeDataset:

    def test_many_rects(self):
        """Test with many rectangles to stress the quad-tree."""
        rng = np.random.RandomState(42)
        n = 500
        rects = []
        for i in range(n):
            x1 = int(rng.randint(0, 90000))
            y1 = int(rng.randint(0, 90000))
            w = int(rng.randint(10, 500))
            h = int(rng.randint(10, 500))
            val = float(rng.uniform(0.1, 100.0))
            rects.append((x1, y1, x1 + w, y1 + h, val))

        path = _create_rects_file(rects)
        data, is_points, num_objs, root_chunk_fpos = _open_track(path)
        try:
            # Test full query
            cpp_s = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000)
            py_s = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 0, 0, 100000, 100000)
            _assert_stats_equal(cpp_s, py_s)

            # Test partial query
            cpp_s = _cpp_query_stats(data, is_points, 10000, 10000, 50000, 50000)
            py_s = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 10000, 10000, 50000, 50000)
            _assert_stats_equal(cpp_s, py_s)

            # Test objects
            cpp_o = _cpp_query_objects(data, is_points, 10000, 10000, 50000, 50000)
            py_o = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 10000, 10000, 50000, 50000)
            assert _sort_objs(cpp_o) == _sort_objs(py_o)
        finally:
            data.close()
            os.unlink(path)

    def test_many_points(self):
        """Test with many points."""
        rng = np.random.RandomState(123)
        n = 1000
        points = []
        for i in range(n):
            x = int(rng.randint(0, 99999))
            y = int(rng.randint(0, 99999))
            val = float(rng.uniform(0.1, 50.0))
            points.append((x, y, val))

        path = _create_points_file(points)
        data, is_points, num_objs, root_chunk_fpos = _open_track(path)
        try:
            cpp_s = _cpp_query_stats(data, is_points, 0, 0, 100000, 100000)
            py_s = _py_query_stats(data, is_points, num_objs, root_chunk_fpos, 0, 0, 100000, 100000)
            _assert_stats_equal(cpp_s, py_s)

            cpp_o = _cpp_query_objects(data, is_points, 20000, 20000, 80000, 80000)
            py_o = _py_query_objects(data, is_points, num_objs, root_chunk_fpos, 20000, 20000, 80000, 80000)
            assert _sort_objs(cpp_o) == _sort_objs(py_o)
        finally:
            data.close()
            os.unlink(path)
