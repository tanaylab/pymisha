"""Tests for 2D track indexed format (TrackIndex2D)."""

import os
import shutil
import struct

import _pymisha
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import (
    IndexedTrack2DReader,
    clear_indexed_2d_cache,
    open_2d_pair,
)

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    """Get track directory path (dots become subdirectories)."""
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    """Remove a track and reload DB."""
    clear_indexed_2d_cache()
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


class TestTrack2DConvertToIndexed:
    """Tests for pm_track2d_convert_to_indexed."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_convert_rects_track(self):
        """Convert a RECTS 2D track to indexed format."""
        tname = "test.idx2d_rects"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 200000],
                ends1=[200, 300000],
                chroms2=["1", "1"],
                starts2=[300, 400000],
                ends2=[400, 499000],
            )
            values = [1.5, 2.5]
            pm.gtrack_2d_create(tname, "test rects indexed", intervals, values)

            tdir = _track_dir(tname)

            # Verify per-pair file exists before conversion
            pair_files_before = [
                f for f in os.listdir(tdir)
                if f not in {".attributes"} and "-" in f
            ]
            assert len(pair_files_before) > 0

            # Convert to indexed format (0 = RECTS)
            num_pairs = _pymisha.pm_track2d_convert_to_indexed(tdir, 0)

            assert num_pairs > 0
            assert os.path.exists(os.path.join(tdir, "track.idx"))
            assert os.path.exists(os.path.join(tdir, "track.dat"))

            # Per-pair files should be removed
            remaining = [
                f for f in os.listdir(tdir)
                if f not in {".attributes", "track.idx", "track.dat"}
            ]
            assert len(remaining) == 0

        finally:
            self._cleanup_track(tname)

    def test_convert_points_track(self):
        """Convert a POINTS 2D track to indexed format."""
        tname = "test.idx2d_points"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 200],
                ends1=[101, 201],
                chroms2=["1", "1"],
                starts2=[300, 400],
                ends2=[301, 401],
            )
            values = [10.0, 20.0]
            pm.gtrack_2d_create(tname, "test points indexed", intervals, values)

            tdir = _track_dir(tname)
            num_pairs = _pymisha.pm_track2d_convert_to_indexed(tdir, 1)

            assert num_pairs > 0
            assert os.path.exists(os.path.join(tdir, "track.idx"))
            assert os.path.exists(os.path.join(tdir, "track.dat"))

        finally:
            self._cleanup_track(tname)

    def test_convert_multi_chrom_pairs(self):
        """Convert a track with multiple chrom pairs."""
        tname = "test.idx2d_multi"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 100, 50],
                ends1=[200, 200, 150],
                chroms2=["1", "2", "2"],
                starts2=[300, 100, 200],
                ends2=[400, 200, 299000],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "multi pair indexed", intervals, values)

            tdir = _track_dir(tname)
            num_pairs = _pymisha.pm_track2d_convert_to_indexed(tdir, 0)

            # Should have at least 2 pairs (1-1, 1-2, and/or 2-2)
            assert num_pairs >= 2

        finally:
            self._cleanup_track(tname)

    def test_convert_already_indexed(self):
        """Converting an already-indexed track returns 0."""
        tname = "test.idx2d_already"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "test", intervals, [1.0])

            tdir = _track_dir(tname)
            num_pairs = _pymisha.pm_track2d_convert_to_indexed(tdir, 0)
            assert num_pairs > 0

            # Second call should return 0 (already indexed)
            num_pairs2 = _pymisha.pm_track2d_convert_to_indexed(tdir, 0)
            assert num_pairs2 == 0

        finally:
            self._cleanup_track(tname)

    def test_convert_invalid_track_type(self):
        """Invalid track_type raises error."""
        tname = "test.idx2d_badtype"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "test", intervals, [1.0])

            tdir = _track_dir(tname)
            with pytest.raises(Exception):
                _pymisha.pm_track2d_convert_to_indexed(tdir, 99)

        finally:
            self._cleanup_track(tname)


class TestTrack2DIndexInfo:
    """Tests for pm_track2d_index_info."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_info_rects(self):
        """Get index info for a RECTS track."""
        tname = "test.idx2d_info_rects"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 200000],
                ends1=[200, 300000],
                chroms2=["1", "1"],
                starts2=[300, 400000],
                ends2=[400, 499000],
            )
            pm.gtrack_2d_create(tname, "info test rects", intervals, [1.5, 2.5])

            tdir = _track_dir(tname)
            _pymisha.pm_track2d_convert_to_indexed(tdir, 0)

            info = _pymisha.pm_track2d_index_info(tdir)
            assert info["loaded"] is True
            assert info["track_type"] == "RECTS"
            assert info["num_pairs"] == 1  # single chrom pair: 1-1
            assert len(info["pairs"]) == 1

            pair = info["pairs"][0]
            assert "chrom1_id" in pair
            assert "chrom2_id" in pair
            assert "offset" in pair
            assert "length" in pair
            assert pair["offset"] == 0
            assert pair["length"] > 0

        finally:
            self._cleanup_track(tname)

    def test_info_points(self):
        """Get index info for a POINTS track."""
        tname = "test.idx2d_info_points"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 200],
                ends1=[101, 201],
                chroms2=["1", "1"],
                starts2=[300, 400],
                ends2=[301, 401],
            )
            pm.gtrack_2d_create(tname, "info test points", intervals, [10.0, 20.0])

            tdir = _track_dir(tname)
            _pymisha.pm_track2d_convert_to_indexed(tdir, 1)

            info = _pymisha.pm_track2d_index_info(tdir)
            assert info["loaded"] is True
            assert info["track_type"] == "POINTS"
            assert info["num_pairs"] == 1

        finally:
            self._cleanup_track(tname)

    def test_info_not_indexed(self):
        """Get info for a non-indexed track returns loaded=False."""
        tname = "test.idx2d_info_noindex"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "no index test", intervals, [1.0])

            tdir = _track_dir(tname)

            info = _pymisha.pm_track2d_index_info(tdir)
            assert info["loaded"] is False
            assert info["track_type"] is None
            assert info["num_pairs"] == 0
            assert len(info["pairs"]) == 0

        finally:
            self._cleanup_track(tname)

    def test_info_multi_pairs(self):
        """Info for a multi-pair indexed track."""
        tname = "test.idx2d_info_multi"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 100, 50],
                ends1=[200, 200, 150],
                chroms2=["1", "2", "2"],
                starts2=[300, 100, 200],
                ends2=[400, 200, 299000],
            )
            pm.gtrack_2d_create(tname, "multi info", intervals, [1.0, 2.0, 3.0])

            tdir = _track_dir(tname)
            num_pairs = _pymisha.pm_track2d_convert_to_indexed(tdir, 0)

            info = _pymisha.pm_track2d_index_info(tdir)
            assert info["loaded"] is True
            assert info["num_pairs"] == num_pairs
            assert len(info["pairs"]) == num_pairs

            # Pairs should be sorted by (chrom1_id, chrom2_id)
            pairs = info["pairs"]
            for i in range(1, len(pairs)):
                prev = (pairs[i - 1]["chrom1_id"], pairs[i - 1]["chrom2_id"])
                curr = (pairs[i]["chrom1_id"], pairs[i]["chrom2_id"])
                assert prev <= curr

            # Offsets should be contiguous
            for i in range(1, len(pairs)):
                expected_offset = pairs[i - 1]["offset"] + pairs[i - 1]["length"]
                assert pairs[i]["offset"] == expected_offset

        finally:
            self._cleanup_track(tname)


class TestTrack2DIndexBinaryFormat:
    """Tests verifying the binary format of the 2D index file."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_header_magic_and_version(self):
        """Verify the binary header has correct magic and version."""
        tname = "test.idx2d_binary"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "binary test", intervals, [1.0])

            tdir = _track_dir(tname)
            _pymisha.pm_track2d_convert_to_indexed(tdir, 0)

            idx_path = os.path.join(tdir, "track.idx")
            with open(idx_path, "rb") as f:
                # Magic: "MISHT2D\0" (8 bytes)
                magic = f.read(8)
                assert magic == b"MISHT2D\x00"

                # Version: uint32 = 1
                version = struct.unpack("<I", f.read(4))[0]
                assert version == 1

                # TrackType: uint32 = 0 (RECTS)
                track_type = struct.unpack("<I", f.read(4))[0]
                assert track_type == 0

                # NumPairs: uint32
                num_pairs = struct.unpack("<I", f.read(4))[0]
                assert num_pairs == 1

                # Flags: uint64 (bit 0 = little-endian)
                flags = struct.unpack("<Q", f.read(8))[0]
                assert flags & 0x01  # little-endian flag set

                # Checksum: uint64
                checksum = struct.unpack("<Q", f.read(8))[0]
                assert checksum != 0  # non-trivial checksum

                # Reserved: uint64
                reserved = struct.unpack("<Q", f.read(8))[0]
                assert reserved == 0

                # Per-pair entry (28 bytes)
                _chrom1_id = struct.unpack("<I", f.read(4))[0]  # noqa: F841
                _chrom2_id = struct.unpack("<I", f.read(4))[0]  # noqa: F841
                offset = struct.unpack("<Q", f.read(8))[0]
                length = struct.unpack("<Q", f.read(8))[0]
                entry_reserved = struct.unpack("<I", f.read(4))[0]

                assert offset == 0
                assert length > 0
                assert entry_reserved == 0

            # Verify track.dat size matches length in index
            dat_path = os.path.join(tdir, "track.dat")
            dat_size = os.path.getsize(dat_path)
            assert dat_size == length

        finally:
            self._cleanup_track(tname)

    def test_header_size_44_bytes(self):
        """Header should be exactly 44 bytes."""
        tname = "test.idx2d_hdrsize"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "hdr size test", intervals, [1.0])

            tdir = _track_dir(tname)
            _pymisha.pm_track2d_convert_to_indexed(tdir, 0)

            idx_path = os.path.join(tdir, "track.idx")
            file_size = os.path.getsize(idx_path)
            # Header (44) + 1 entry (28) = 72
            expected_size = 44 + 1 * 28
            assert file_size == expected_size

        finally:
            self._cleanup_track(tname)


# ---------------------------------------------------------------------------
# Phase 2: Roundtrip parity tests (indexed vs per-pair extraction)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _clean_vtracks():
    """Clean up all vtracks after each test."""
    yield
    pm.gvtrack_clear()
    clear_indexed_2d_cache()


def _create_rects_track(tname, intervals, values):
    """Create a RECTS 2D track and return its directory path."""
    pm.gtrack_2d_create(tname, "test rects", intervals, values)
    return _track_dir(tname)


def _create_points_track(tname, intervals, values):
    """Create a POINTS 2D track (unit-size intervals)."""
    pm.gtrack_2d_create(tname, "test points", intervals, values)
    return _track_dir(tname)


def _convert_to_indexed(tdir, is_points=False):
    """Convert a track to indexed format."""
    clear_indexed_2d_cache()
    track_type = 1 if is_points else 0
    n = _pymisha.pm_track2d_convert_to_indexed(tdir, track_type)
    assert n > 0, "Conversion should produce at least one pair"
    return n


class TestRoundtripRectsExtraction:
    """Verify gextract on RECTS tracks gives identical results before and after
    conversion to indexed format."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_rects_single_pair_roundtrip(self):
        """Extract RECTS track before and after indexed conversion — results
        must be identical."""
        tname = "test.idx_rt_rects1"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 200000],
                ends1=[200, 2000, 300000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 400000],
                ends2=[400, 4000, 499000],
            )
            values = [1.5, 2.5, 3.5]
            pm.gtrack_2d_create(tname, "roundtrip rects", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract(tname, intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)

    def test_rects_multi_pair_roundtrip(self):
        """RECTS track with multiple chromosome pairs."""
        tname = "test.idx_rt_rects_multi"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 100, 50],
                ends1=[200, 200, 150],
                chroms2=["1", "2", "2"],
                starts2=[300, 100, 200],
                ends2=[400, 200, 299000],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "multi pair rects", intervals, values)

            # Query all pairs
            q1 = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            q2 = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
            q3 = pm.gintervals_2d("2", 0, 300000, "2", 0, 300000)
            query = pd.concat([q1, q2, q3], ignore_index=True)

            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract(tname, intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)


class TestRoundtripPointsExtraction:
    """Verify gextract on POINTS tracks gives identical results before and
    after conversion to indexed format."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_points_single_pair_roundtrip(self):
        """Extract POINTS track before and after indexed conversion."""
        tname = "test.idx_rt_pts1"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 200000],
                ends1=[101, 1001, 200001],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 400000],
                ends2=[301, 3001, 400001],
            )
            values = [10.0, 20.0, 30.0]
            pm.gtrack_2d_create(tname, "roundtrip points", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=True)

            after = pm.gextract(tname, intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)

    def test_points_multi_pair_roundtrip(self):
        """POINTS track with multiple chromosome pairs."""
        tname = "test.idx_rt_pts_multi"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "2"],
                starts1=[500, 100],
                ends1=[501, 101],
                chroms2=["2", "2"],
                starts2=[100, 200],
                ends2=[101, 201],
            )
            values = [5.0, 15.0]
            pm.gtrack_2d_create(tname, "multi points", intervals, values)

            q1 = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
            q2 = pm.gintervals_2d("2", 0, 300000, "2", 0, 300000)
            query = pd.concat([q1, q2], ignore_index=True)

            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=True)

            after = pm.gextract(tname, intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)


class TestRoundtripBandFilter:
    """Verify band filtering works identically with indexed tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_band_filter_rects_roundtrip(self):
        """Band filtering on RECTS track before and after indexed conversion."""
        tname = "test.idx_rt_band"
        self._cleanup_track(tname)
        try:
            # Objects at different diagonal offsets
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 500, 10000],
                ends1=[200, 600, 11000],
                chroms2=["1", "1", "1"],
                starts2=[100, 200, 10000],
                ends2=[200, 300, 11000],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "band test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            band = (-500, 500)

            before = pm.gextract(tname, intervals=query, band=band)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract(tname, intervals=query, band=band)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)

    def test_band_filter_points_roundtrip(self):
        """Band filtering on POINTS track before and after indexed conversion."""
        tname = "test.idx_rt_band_pts"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 500, 10000],
                ends1=[101, 501, 10001],
                chroms2=["1", "1", "1"],
                starts2=[100, 200, 10000],
                ends2=[101, 201, 10001],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "band pts test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            band = (-50, 50)

            before = pm.gextract(tname, intervals=query, band=band)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=True)

            after = pm.gextract(tname, intervals=query, band=band)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)


class TestRoundtripVtrackAgg:
    """Verify virtual track aggregation on indexed tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_weighted_sum_roundtrip(self):
        """weighted.sum vtrack before and after indexed conversion."""
        tname = "test.idx_rt_ws"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "vtrack agg test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_ws", tname, func="weighted.sum")

            before = pm.gextract("vt_ws", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_ws", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_area_roundtrip(self):
        """area vtrack before and after indexed conversion."""
        tname = "test.idx_rt_area"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "area test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_area", tname, func="area")

            before = pm.gextract("vt_area", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_area", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_avg_roundtrip(self):
        """avg vtrack before and after indexed conversion."""
        tname = "test.idx_rt_avg"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "avg test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_avg", tname, func="avg")

            before = pm.gextract("vt_avg", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_avg", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


class TestRoundtripVtrackNonAgg:
    """Verify virtual track non-aggregation functions on indexed tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_exists_roundtrip(self):
        """exists vtrack before and after indexed conversion."""
        tname = "test.idx_rt_exists"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "exists test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_exists", tname, func="exists")

            before = pm.gextract("vt_exists", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_exists", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_size_roundtrip(self):
        """size vtrack before and after indexed conversion."""
        tname = "test.idx_rt_size"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "size test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_size", tname, func="size")

            before = pm.gextract("vt_size", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_size", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


class TestRoundtripGiteratorIntervals2D:
    """Verify giterator_intervals_2d works with indexed tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_giterator_2d_roundtrip(self):
        """giterator_intervals_2d produces same chunks before and after
        indexed conversion."""
        tname = "test.idx_rt_giter"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "giter test", intervals, values)

            query = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[0, 0],
                ends1=[500000, 500000],
                chroms2=["1", "1"],
                starts2=[0, 0],
                ends2=[500000, 500000],
            )

            before_chunks = list(
                pm.giterator_intervals_2d(tname, intervals=query)
            )

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after_chunks = list(
                pm.giterator_intervals_2d(tname, intervals=query)
            )

            assert len(before_chunks) == len(after_chunks)
            for b, a in zip(before_chunks, after_chunks, strict=False):
                pd.testing.assert_frame_equal(b, a, check_like=True)
        finally:
            self._cleanup_track(tname)


class TestRoundtripCppFastPath:
    """Verify the C++ quad-tree fast path works with indexed track data."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_cpp_fast_path_objects(self):
        """C++ pm_quadtree_query_objects on indexed pair data produces
        same results as on the original per-pair file."""
        tname = "test.idx_rt_cpp_obj"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 200000],
                ends1=[200, 2000, 300000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 400000],
                ends2=[400, 4000, 499000],
            )
            values = [1.5, 2.5, 3.5]
            pm.gtrack_2d_create(tname, "cpp test", intervals, values)

            # Extract before conversion
            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            # Use open_2d_pair to get the indexed data and verify it works
            pair = open_2d_pair(tdir, "1", "1")
            assert pair is not None
            is_points, num_objs, data, root_chunk_fpos, close_fn = pair
            assert not is_points
            assert num_objs == 3

            # Query via C++ (the data buffer should be valid)
            from pymisha._quadtree import query_2d_track_opened
            objs = query_2d_track_opened(
                data, is_points, num_objs, root_chunk_fpos,
                0, 0, 500000, 500000,
            )
            assert len(objs) == 3
            close_fn()

            # Full extraction should match
            after = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)

    def test_cpp_fast_path_stats(self):
        """C++ pm_quadtree_query_stats on indexed pair data produces
        same results as on the original per-pair file."""
        tname = "test.idx_rt_cpp_stats"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "cpp stats test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_ws_cpp", tname, func="weighted.sum")
            before = pm.gextract("vt_ws_cpp", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_ws_cpp", intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


class TestIndexedReaderUnit:
    """Unit tests for IndexedTrack2DReader class."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_reader_not_indexed(self):
        """Reader for a non-indexed directory returns loaded=False."""
        tname = "test.idx_unit_noindex"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "no index", intervals, [1.0])

            tdir = _track_dir(tname)
            reader = IndexedTrack2DReader(tdir)
            assert not reader.loaded
        finally:
            self._cleanup_track(tname)

    def test_reader_indexed(self):
        """Reader for an indexed directory loads correctly."""
        tname = "test.idx_unit_loaded"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "indexed", intervals, [1.0])

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            reader = IndexedTrack2DReader(tdir)
            assert reader.loaded
            assert not reader.is_points

            result = reader.get_pair_data(0, 0)  # chrom 1=id0, 1=id0
            assert result is not None
            is_pts, num_objs, data, rcf = result
            assert not is_pts
            assert num_objs == 1

            # Missing pair should return None
            result2 = reader.get_pair_data(0, 1)
            assert result2 is None

            reader.close()
        finally:
            self._cleanup_track(tname)

    def test_open_2d_pair_indexed(self):
        """open_2d_pair returns data from indexed format."""
        tname = "test.idx_unit_open2d"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "open 2d", intervals, [7.0])

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            pair = open_2d_pair(tdir, "1", "1")
            assert pair is not None
            is_pts, num_objs, data, rcf, close_fn = pair
            assert not is_pts
            assert num_objs == 1
            close_fn()

            # Non-existent pair
            pair2 = open_2d_pair(tdir, "1", "X")
            assert pair2 is None
        finally:
            clear_indexed_2d_cache()
            self._cleanup_track(tname)

    def test_open_2d_pair_falls_back_to_perpair(self):
        """open_2d_pair falls back to per-pair files for non-indexed tracks."""
        tname = "test.idx_unit_fallback"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "fallback", intervals, [9.0])

            tdir = _track_dir(tname)

            pair = open_2d_pair(tdir, "1", "1")
            assert pair is not None
            is_pts, num_objs, data, rcf, close_fn = pair
            assert not is_pts
            assert num_objs == 1
            close_fn()
        finally:
            self._cleanup_track(tname)


class TestRoundtripMixedVtrackBand:
    """Verify combined vtrack + band filter on indexed tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_agg_with_band_roundtrip(self):
        """Aggregation vtrack with band filter on indexed track matches
        pre-conversion results."""
        tname = "test.idx_rt_agg_band"
        self._cleanup_track(tname)
        try:
            # Objects near the diagonal
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 500, 10000],
                ends1=[200, 600, 11000],
                chroms2=["1", "1", "1"],
                starts2=[100, 200, 10000],
                ends2=[200, 300, 11000],
            )
            values = [1.0, 2.0, 3.0]
            pm.gtrack_2d_create(tname, "agg band test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            band = (-500, 500)
            pm.gvtrack_create("vt_area_band", tname, func="area")

            before = pm.gextract("vt_area_band", intervals=query, band=band)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_area_band", intervals=query, band=band)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


# ---------------------------------------------------------------------------
# Phase 3: Public API tests for gtrack_2d_convert_to_indexed
# ---------------------------------------------------------------------------


class TestGtrack2dConvertToIndexedAPI:
    """Tests for the public gtrack_2d_convert_to_indexed() function."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_convert_rects_track(self):
        """gtrack_2d_convert_to_indexed converts a RECTS track."""
        tname = "test.api_idx2d_rects"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 200000],
                ends1=[200, 300000],
                chroms2=["1", "1"],
                starts2=[300, 400000],
                ends2=[400, 499000],
            )
            pm.gtrack_2d_create(tname, "api rects test", intervals, [1.5, 2.5])

            tdir = _track_dir(tname)
            # Verify per-pair files exist before conversion
            assert any("-" in f for f in os.listdir(tdir) if f != ".attributes")

            pm.gtrack_2d_convert_to_indexed(tname)

            assert os.path.exists(os.path.join(tdir, "track.idx"))
            assert os.path.exists(os.path.join(tdir, "track.dat"))
        finally:
            self._cleanup_track(tname)

    def test_convert_points_track(self):
        """gtrack_2d_convert_to_indexed converts a POINTS track."""
        tname = "test.api_idx2d_points"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 200],
                ends1=[101, 201],
                chroms2=["1", "1"],
                starts2=[300, 400],
                ends2=[301, 401],
            )
            pm.gtrack_2d_create(tname, "api points test", intervals, [10.0, 20.0])

            pm.gtrack_2d_convert_to_indexed(tname)

            tdir = _track_dir(tname)
            assert os.path.exists(os.path.join(tdir, "track.idx"))
            assert os.path.exists(os.path.join(tdir, "track.dat"))
        finally:
            self._cleanup_track(tname)

    def test_skip_when_already_indexed(self):
        """gtrack_2d_convert_to_indexed skips if already indexed (no force)."""
        tname = "test.api_idx2d_skip"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "skip test", intervals, [1.0])

            pm.gtrack_2d_convert_to_indexed(tname)

            tdir = _track_dir(tname)
            idx_path = os.path.join(tdir, "track.idx")
            mtime_before = os.path.getmtime(idx_path)

            # Second call without force should be a no-op
            pm.gtrack_2d_convert_to_indexed(tname)
            mtime_after = os.path.getmtime(idx_path)
            assert mtime_before == mtime_after
        finally:
            self._cleanup_track(tname)

    def test_force_reconversion(self):
        """gtrack_2d_convert_to_indexed with force=True re-converts."""
        tname = "test.api_idx2d_force"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "force test", intervals, [1.0])

            pm.gtrack_2d_convert_to_indexed(tname)

            tdir = _track_dir(tname)
            idx_path = os.path.join(tdir, "track.idx")
            assert os.path.exists(idx_path)

            # Force should re-run (we can verify by checking it doesn't error)
            clear_indexed_2d_cache()
            pm.gtrack_2d_convert_to_indexed(tname, force=True)
            assert os.path.exists(idx_path)
        finally:
            self._cleanup_track(tname)

    def test_invalid_track_raises(self):
        """gtrack_2d_convert_to_indexed raises on non-existent track."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_2d_convert_to_indexed("no_such_track_xyz")

    def test_1d_track_raises(self):
        """gtrack_2d_convert_to_indexed raises on a 1D track."""
        with pytest.raises(ValueError, match="not a 2D track"):
            pm.gtrack_2d_convert_to_indexed("dense_track")

    def test_none_track_raises(self):
        """gtrack_2d_convert_to_indexed raises on None."""
        with pytest.raises(ValueError, match="cannot be None"):
            pm.gtrack_2d_convert_to_indexed(None)

    def test_extraction_after_api_convert(self):
        """Extraction works correctly after gtrack_2d_convert_to_indexed."""
        tname = "test.api_idx2d_extract"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 1000],
                ends1=[200, 2000],
                chroms2=["1", "1"],
                starts2=[300, 3000],
                ends2=[400, 4000],
            )
            values = [1.5, 2.5]
            pm.gtrack_2d_create(tname, "extract test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            before = pm.gextract(tname, intervals=query)

            pm.gtrack_2d_convert_to_indexed(tname)

            clear_indexed_2d_cache()
            after = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)


class TestGtrackConvertToIndexedDispatch:
    """Tests that gtrack_convert_to_indexed dispatches correctly for 2D tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_dispatch_2d_rects(self):
        """gtrack_convert_to_indexed on a 2D RECTS track dispatches to 2D conversion."""
        tname = "test.dispatch_2d_rects"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "dispatch test", intervals, [1.0])

            # Use the generic gtrack_convert_to_indexed
            pm.gtrack_convert_to_indexed(tname)

            tdir = _track_dir(tname)
            assert os.path.exists(os.path.join(tdir, "track.idx"))
            assert os.path.exists(os.path.join(tdir, "track.dat"))

            # Verify the index is 2D format (MISHT2D magic)
            idx_path = os.path.join(tdir, "track.idx")
            with open(idx_path, "rb") as f:
                magic = f.read(8)
            assert magic == b"MISHT2D\x00"
        finally:
            self._cleanup_track(tname)

    def test_dispatch_2d_points(self):
        """gtrack_convert_to_indexed on a 2D POINTS track dispatches to 2D conversion."""
        tname = "test.dispatch_2d_pts"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[101],
                chroms2=["1"],
                starts2=[300],
                ends2=[301],
            )
            pm.gtrack_2d_create(tname, "dispatch pts test", intervals, [5.0])

            pm.gtrack_convert_to_indexed(tname)

            tdir = _track_dir(tname)
            assert os.path.exists(os.path.join(tdir, "track.idx"))

            # Verify 2D magic
            idx_path = os.path.join(tdir, "track.idx")
            with open(idx_path, "rb") as f:
                magic = f.read(8)
            assert magic == b"MISHT2D\x00"
        finally:
            self._cleanup_track(tname)


# ---------------------------------------------------------------------------
# Phase 4: Additional coverage — vtrack functions, edge cases,
#           physical validation, multi-expression
# ---------------------------------------------------------------------------


class TestRoundtripVtrackMinMax:
    """Roundtrip tests for min/max aggregation vtracks on indexed 2D tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_min_roundtrip(self):
        """min vtrack before and after indexed conversion."""
        tname = "test.idx_rt_min"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 5000],
                ends1=[200, 2000, 6000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 7000],
                ends2=[400, 4000, 8000],
            )
            values = [5.0, -3.0, 10.0]
            pm.gtrack_2d_create(tname, "min test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_min", tname, func="min")

            before = pm.gextract("vt_min", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_min", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_max_roundtrip(self):
        """max vtrack before and after indexed conversion."""
        tname = "test.idx_rt_max"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 5000],
                ends1=[200, 2000, 6000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 7000],
                ends2=[400, 4000, 8000],
            )
            values = [5.0, -3.0, 10.0]
            pm.gtrack_2d_create(tname, "max test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_max", tname, func="max")

            before = pm.gextract("vt_max", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_max", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


class TestRoundtripVtrackFirstLastSample:
    """Roundtrip tests for first/last/sample object vtracks on indexed 2D tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_first_roundtrip(self):
        """first vtrack before and after indexed conversion."""
        tname = "test.idx_rt_first"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "first test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_first", tname, func="first")

            before = pm.gextract("vt_first", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_first", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_last_roundtrip(self):
        """last vtrack before and after indexed conversion."""
        tname = "test.idx_rt_last"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "last test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_last", tname, func="last")

            before = pm.gextract("vt_last", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_last", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_sample_roundtrip_single_object(self):
        """sample vtrack with exactly one object per query — deterministic."""
        tname = "test.idx_rt_sample"
        self._cleanup_track(tname)
        try:
            # One object — sample is deterministic when only one object
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            values = [42.0]
            pm.gtrack_2d_create(tname, "sample test", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_sample", tname, func="sample")

            before = pm.gextract("vt_sample", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_sample", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


class TestRoundtripVtrackGlobalPercentile:
    """Roundtrip tests for global.percentile vtrack on indexed 2D tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_global_percentile_roundtrip(self):
        """global.percentile vtrack before and after indexed conversion."""
        tname = "test.idx_rt_gpct"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "gpct test", intervals, values)

            # Use three separate query intervals to get a meaningful percentile
            query = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[0, 500, 4000],
                ends1=[500, 3000, 9000],
                chroms2=["1", "1", "1"],
                starts2=[0, 2000, 6000],
                ends2=[500, 5000, 9000],
            )
            pm.gvtrack_create("vt_gpct", tname, func="global.percentile")

            before = pm.gextract("vt_gpct", intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract("vt_gpct", intervals=query)

            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)


class TestEdgeCases:
    """Edge case tests for indexed 2D tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_single_chrom_pair_only(self):
        """Track with data only in one chrom pair (1-1), queried across
        multiple pairs — only 1-1 should have results."""
        tname = "test.idx_edge_single_pair"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 5000],
                ends1=[200, 6000],
                chroms2=["1", "1"],
                starts2=[300, 7000],
                ends2=[400, 8000],
            )
            values = [1.0, 2.0]
            pm.gtrack_2d_create(tname, "single pair edge", intervals, values)

            # Query all possible pairs (1-1, 1-2, 2-2, etc.)
            q1 = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            q2 = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
            q3 = pm.gintervals_2d("2", 0, 300000, "2", 0, 300000)
            query = pd.concat([q1, q2, q3], ignore_index=True)

            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)

            # Only 1-1 pair should have results
            assert all(after["chrom1"] == "1")
            assert all(after["chrom2"] == "1")
        finally:
            self._cleanup_track(tname)

    def test_all_chrom_pairs(self):
        """Track with data across all 3x3 chrom pair combinations."""
        tname = "test.idx_edge_all_pairs"
        self._cleanup_track(tname)
        try:
            chroms = ["1", "2", "X"]
            chrom_sizes = {"1": 500000, "2": 300000, "X": 200000}
            c1s, s1s, e1s, c2s, s2s, e2s, vals = [], [], [], [], [], [], []
            val = 1.0
            for ca in chroms:
                for cb in chroms:
                    # Ensure ca <= cb for misha canonical ordering
                    if chroms.index(ca) > chroms.index(cb):
                        continue
                    c1s.append(ca)
                    s1s.append(100)
                    e1s.append(200)
                    c2s.append(cb)
                    s2s.append(100)
                    e2s.append(200)
                    vals.append(val)
                    val += 1.0

            intervals = pm.gintervals_2d(
                chroms1=c1s, starts1=s1s, ends1=e1s,
                chroms2=c2s, starts2=s2s, ends2=e2s,
            )
            pm.gtrack_2d_create(tname, "all pairs edge", intervals, vals)

            # Query all pairs
            queries = []
            for ca in chroms:
                for cb in chroms:
                    if chroms.index(ca) > chroms.index(cb):
                        continue
                    queries.append(pm.gintervals_2d(
                        ca, 0, chrom_sizes[ca], cb, 0, chrom_sizes[cb]
                    ))
            query = pd.concat(queries, ignore_index=True)

            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            info = _pymisha.pm_track2d_index_info(tdir)
            assert info["num_pairs"] == len(vals)

            after = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)
        finally:
            self._cleanup_track(tname)

    def test_single_object_per_pair(self):
        """Track with exactly one object per chrom pair."""
        tname = "test.idx_edge_one_per_pair"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 50],
                ends1=[200, 150],
                chroms2=["1", "2"],
                starts2=[300, 100],
                ends2=[400, 200],
            )
            values = [7.0, 13.0]
            pm.gtrack_2d_create(tname, "one per pair", intervals, values)

            q1 = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            q2 = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
            query = pd.concat([q1, q2], ignore_index=True)

            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)
            assert len(after) == 2
        finally:
            self._cleanup_track(tname)

    def test_large_and_negative_values(self):
        """Track with extreme values: large positive and negative."""
        tname = "test.idx_edge_extreme_vals"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 5000],
                ends1=[200, 2000, 6000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 7000],
                ends2=[400, 4000, 8000],
            )
            values = [1e10, -5.0, -1e8]
            pm.gtrack_2d_create(tname, "extreme vals", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            before = pm.gextract(tname, intervals=query)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            after = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after, check_like=True)

            # Also verify vtrack aggregation handles extreme values
            pm.gvtrack_create("vt_min_ext", tname, func="min")
            pm.gvtrack_create("vt_max_ext", tname, func="max")

            min_result = pm.gextract("vt_min_ext", intervals=query)
            max_result = pm.gextract("vt_max_ext", intervals=query)

            assert min_result["vt_min_ext"].iloc[0] == -1e8
            assert max_result["vt_max_ext"].iloc[0] == 1e10
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_force_true_idempotent(self):
        """Convert twice with force=True — extraction still identical."""
        tname = "test.idx_edge_force_idem"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "force idempotent", intervals, values)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            before = pm.gextract(tname, intervals=query)

            # First conversion
            pm.gtrack_2d_convert_to_indexed(tname)
            after1 = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after1, check_like=True)

            # Second conversion with force
            clear_indexed_2d_cache()
            pm.gtrack_2d_convert_to_indexed(tname, force=True)
            after2 = pm.gextract(tname, intervals=query)
            pd.testing.assert_frame_equal(before, after2, check_like=True)
        finally:
            self._cleanup_track(tname)


class TestPhysicalStructureValidation:
    """Validate physical structure of indexed files after conversion."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_dat_size_matches_index_entries(self):
        """track.dat size equals the sum of all entry lengths from the index."""
        tname = "test.idx_phys_datsize"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 100, 50],
                ends1=[200, 200, 150],
                chroms2=["1", "2", "2"],
                starts2=[300, 100, 200],
                ends2=[400, 200, 299000],
            )
            pm.gtrack_2d_create(tname, "phys validation", intervals, [1.0, 2.0, 3.0])

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            info = _pymisha.pm_track2d_index_info(tdir)
            total_length = sum(p["length"] for p in info["pairs"])

            dat_path = os.path.join(tdir, "track.dat")
            dat_size = os.path.getsize(dat_path)
            assert dat_size == total_length
        finally:
            self._cleanup_track(tname)

    def test_perpair_files_removed_after_conversion(self):
        """After conversion, per-pair files (containing '-') are removed."""
        tname = "test.idx_phys_removed"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1"],
                starts1=[100, 50],
                ends1=[200, 150],
                chroms2=["1", "2"],
                starts2=[300, 100],
                ends2=[400, 200],
            )
            pm.gtrack_2d_create(tname, "remove old test", intervals, [1.0, 2.0])

            tdir = _track_dir(tname)

            # Before conversion: per-pair files exist
            pair_files_before = [
                f for f in os.listdir(tdir)
                if f not in {".attributes"} and "-" in f
            ]
            assert len(pair_files_before) >= 2

            _convert_to_indexed(tdir, is_points=False)

            # After conversion: per-pair files removed
            remaining = [
                f for f in os.listdir(tdir)
                if f not in {".attributes", "track.idx", "track.dat"}
            ]
            assert len(remaining) == 0

            # Index and data files exist
            assert os.path.exists(os.path.join(tdir, "track.idx"))
            assert os.path.exists(os.path.join(tdir, "track.dat"))
        finally:
            self._cleanup_track(tname)


class TestMultiExpression:
    """Tests for multi-expression extraction on indexed 2D tracks."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_two_vtrack_expressions(self):
        """Extract two different vtracks on same indexed track in one call."""
        tname = "test.idx_multi_two_vt"
        self._cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "1"],
                starts1=[100, 1000, 5000],
                ends1=[200, 2000, 6000],
                chroms2=["1", "1", "1"],
                starts2=[300, 3000, 7000],
                ends2=[400, 4000, 8000],
            )
            values = [5.0, -3.0, 10.0]
            pm.gtrack_2d_create(tname, "multi expr test", intervals, values)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_me_min", tname, func="min")
            pm.gvtrack_create("vt_me_max", tname, func="max")

            result = pm.gextract(["vt_me_min", "vt_me_max"], intervals=query)

            assert "vt_me_min" in result.columns
            assert "vt_me_max" in result.columns
            assert len(result) == 1
            assert result["vt_me_min"].iloc[0] == -3.0
            assert result["vt_me_max"].iloc[0] == 10.0
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)

    def test_arithmetic_on_indexed_vtrack(self):
        """Arithmetic expression referencing an indexed 2D vtrack."""
        tname = "test.idx_multi_arith"
        self._cleanup_track(tname)
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
            pm.gtrack_2d_create(tname, "arith test", intervals, values)

            tdir = _track_dir(tname)
            _convert_to_indexed(tdir, is_points=False)

            query = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            pm.gvtrack_create("vt_arith_ws", tname, func="weighted.sum")

            result = pm.gextract("vt_arith_ws * 2", intervals=query)

            # Also get the raw weighted.sum for comparison
            raw = pm.gextract("vt_arith_ws", intervals=query)
            assert abs(result["vt_arith_ws * 2"].iloc[0] - raw["vt_arith_ws"].iloc[0] * 2) < 1e-6
        finally:
            pm.gvtrack_clear()
            self._cleanup_track(tname)
