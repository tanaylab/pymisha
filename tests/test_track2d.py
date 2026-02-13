"""Tests for 2D track creation, import, and reading."""

import os
import shutil
import struct

import numpy as np
import pytest

import pymisha as pm

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    """Get track directory path (dots become subdirectories)."""
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    """Remove a track and reload DB."""
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        import _pymisha
        _pymisha.pm_dbreload()


def test_quadtree_split_preserves_parent_stats():
    from pymisha._quadtree import QuadTree

    qtree = QuadTree(0, 0, 16, 16, max_node_objs=1)
    qtree.insert((0, 0, 8, 8, 10.0))
    qtree.insert((8, 8, 16, 16, 20.0))

    root = qtree.root
    assert not root.is_leaf
    assert root.stat["occupied_area"] == 128
    assert root.stat["weighted_sum"] == pytest.approx(1920.0)
    assert root.stat["min_val"] == 10.0
    assert root.stat["max_val"] == 20.0


class TestGtrack2dCreate:
    """Tests for gtrack_2d_create."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_create_basic_rects(self):
        """Create a simple 2D rectangles track."""
        tname = "test.test_2d_rects"
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
            pm.gtrack_2d_create(tname, "test 2D track", intervals, values)

            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "rectangles"
        finally:
            self._cleanup_track(tname)

    def test_create_points_track(self):
        """Create a 2D points track (all intervals are 1bp)."""
        tname = "test.test_2d_points"
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
            pm.gtrack_2d_create(tname, "test points", intervals, values)

            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"
        finally:
            self._cleanup_track(tname)

    def test_create_multi_chrom_pairs(self):
        """Create 2D track spanning multiple chromosome pairs."""
        tname = "test.test_2d_multi"
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
            pm.gtrack_2d_create(tname, "multi chrom", intervals, values)

            assert pm.gtrack_exists(tname)
            # Check that per-chrom-pair files exist
            tdir = _track_dir(tname)
            # Files should be named with normalized chroms
            files = [f for f in os.listdir(tdir) if not f.startswith(".")]
            assert len(files) >= 2  # at least 1-1 and 1-2 (or 2-2)
        finally:
            self._cleanup_track(tname)

    def test_create_values_length_mismatch(self):
        """Values length must match intervals length."""
        intervals = pm.gintervals_2d(
            chroms1=["1"], starts1=[100], ends1=[200],
            chroms2=["1"], starts2=[300], ends2=[400],
        )
        with pytest.raises(ValueError, match="values"):
            pm.gtrack_2d_create("test.bad", "bad", intervals, [1.0, 2.0])

    def test_create_overlapping_intervals_error(self):
        """Overlapping 2D intervals should raise an error."""
        intervals = pm.gintervals_2d(
            chroms1=["1", "1"],
            starts1=[100, 150],
            ends1=[200, 250],
            chroms2=["1", "1"],
            starts2=[300, 350],
            ends2=[400, 450],
        )
        tname = "test.test_2d_overlap"
        self._cleanup_track(tname)
        try:
            with pytest.raises(ValueError, match="[Oo]verlap"):
                pm.gtrack_2d_create(tname, "overlap", intervals, [1.0, 2.0])
        finally:
            self._cleanup_track(tname)

    def test_roundtrip_read_existing_track(self):
        """Read the existing rects_track from test DB and verify we can parse it."""
        # The test DB has rects_track with chr1-chr1, chr1-chr2, chr1-chrX
        info = pm.gtrack_info("rects_track")
        assert info["type"] == "rectangles"


class TestGtrack2dImport:
    """Tests for gtrack_2d_import."""

    def _cleanup_track(self, name):
        _cleanup_track(name)

    def test_import_from_file(self, tmp_path):
        """Import 2D track from tab-delimited file."""
        tname = "test.test_2d_import"
        self._cleanup_track(tname)
        try:
            # Write a tab-delimited file
            tsv = tmp_path / "contacts.tsv"
            tsv.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t100\t200\t1\t300\t400\t5.5\n"
                "1\t1000\t2000\t2\t5000\t6000\t3.0\n"
            )
            pm.gtrack_2d_import(tname, "imported 2D", str(tsv))
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "rectangles"
        finally:
            self._cleanup_track(tname)

    def test_import_points_from_file(self, tmp_path):
        """Import 2D points track from tab-delimited file."""
        tname = "test.test_2d_import_pts"
        self._cleanup_track(tname)
        try:
            tsv = tmp_path / "points.tsv"
            tsv.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t100\t101\t1\t300\t301\t5.5\n"
                "1\t1000\t1001\t2\t5000\t5001\t3.0\n"
            )
            pm.gtrack_2d_import(tname, "imported points", str(tsv))
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"
        finally:
            self._cleanup_track(tname)


class TestQuadTreeBinaryCompat:
    """Test that our quad-tree writer produces files readable by the existing rects_track reader."""

    def test_create_and_verify_binary_format(self):
        """Create a track and verify the binary file has correct signature."""
        tname = "test.test_2d_bincompat"
        _cleanup_track(tname)
        tdir = _track_dir(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[200],
                chroms2=["1"],
                starts2=[300],
                ends2=[400],
            )
            pm.gtrack_2d_create(tname, "bincompat", intervals, [42.0])

            # Check binary format
            chrom_file = os.path.join(tdir, "1-1")
            assert os.path.exists(chrom_file), f"Expected 1-1 in {os.listdir(tdir)}"
            with open(chrom_file, "rb") as f:
                data = f.read()

            # Signature should be -9 (RECTS)
            sig = struct.unpack_from("<i", data, 0)[0]
            assert sig == -9

            # num_objs
            num_objs = struct.unpack_from("<Q", data, 4)[0]
            assert num_objs == 1

            # root_chunk_fpos
            root_fpos = struct.unpack_from("<q", data, 12)[0]
            assert root_fpos > 0

            # Read chunk header
            chunk_size, top_node_off = struct.unpack_from("<qq", data, root_fpos)
            assert chunk_size > 0
            assert top_node_off > 0

            # Read leaf
            node_start = root_fpos + top_node_off
            is_leaf = data[node_start]
            assert is_leaf == 1

            # Read stat
            off = node_start + 8  # skip is_leaf + padding
            occ_area = struct.unpack_from("<q", data, off)[0]
            assert occ_area == 100 * 100  # (200-100) * (400-300)

            # Read arena
            off += 32  # skip stat
            x1, y1, x2, y2 = struct.unpack_from("<qqqq", data, off)
            assert x1 == 0 and y1 == 0  # arena is full chrom range

            # Read num_objs in leaf
            off += 32
            n = struct.unpack_from("<I", data, off)[0]
            assert n == 1

            # Read object
            off += 8  # 4 bytes + 4 padding
            obj_id = struct.unpack_from("<Q", data, off)[0]
            assert obj_id == 0
            rx1, ry1, rx2, ry2 = struct.unpack_from("<qqqq", data, off + 8)
            assert (rx1, ry1, rx2, ry2) == (100, 300, 200, 400)
            rv = struct.unpack_from("<f", data, off + 40)[0]
            assert abs(rv - 42.0) < 0.01
        finally:
            _cleanup_track(tname)

    def test_points_binary_format(self):
        """Create a points track and verify signature is -10."""
        tname = "test.test_2d_pts_bin"
        _cleanup_track(tname)
        tdir = _track_dir(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[101],
                chroms2=["1"],
                starts2=[300],
                ends2=[301],
            )
            pm.gtrack_2d_create(tname, "pts bin", intervals, [7.0])

            chrom_file = os.path.join(tdir, "1-1")
            with open(chrom_file, "rb") as f:
                sig = struct.unpack_from("<i", f.read(4), 0)[0]
            assert sig == -10  # POINTS format
        finally:
            _cleanup_track(tname)

    def test_cpp_reads_created_rects_track(self):
        """Verify C++ type detection works on Python-created RECTS track."""
        tname = "test.test_2d_cpp_read"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1", "1", "2"],
                starts1=[100, 10000, 50],
                ends1=[200, 20000, 150],
                chroms2=["1", "1", "2"],
                starts2=[300, 30000, 200],
                ends2=[400, 40000, 250],
            )
            pm.gtrack_2d_create(tname, "cpp read test", intervals, [1.0, 2.0, 3.0])
            info = pm.gtrack_info(tname)
            assert info["type"] == "rectangles"
            assert info["dimensions"] == 2
        finally:
            _cleanup_track(tname)

    def test_cpp_reads_created_points_track(self):
        """Verify C++ type detection works on Python-created POINTS track."""
        tname = "test.test_2d_cpp_pts"
        _cleanup_track(tname)
        try:
            intervals = pm.gintervals_2d(
                chroms1=["1"],
                starts1=[100],
                ends1=[101],
                chroms2=["1"],
                starts2=[300],
                ends2=[301],
            )
            pm.gtrack_2d_create(tname, "cpp pts test", intervals, [5.0])
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"
            assert info["dimensions"] == 2
        finally:
            _cleanup_track(tname)

    def test_many_objects_trigger_splitting(self):
        """Create a track with enough objects to trigger quad-tree splitting."""
        tname = "test.test_2d_split"
        _cleanup_track(tname)
        try:
            n = 100
            rng = np.random.RandomState(42)
            starts1 = rng.randint(0, 400000, size=n)
            starts2 = rng.randint(0, 400000, size=n)
            # Make small non-overlapping rectangles
            ends1 = starts1 + 10
            ends2 = starts2 + 10
            vals = rng.uniform(1, 100, size=n).astype(np.float32)

            intervals = pm.gintervals_2d(
                chroms1=["1"] * n,
                starts1=starts1.tolist(),
                ends1=ends1.tolist(),
                chroms2=["1"] * n,
                starts2=starts2.tolist(),
                ends2=ends2.tolist(),
            )
            pm.gtrack_2d_create(tname, "split test", intervals, vals.tolist())
            info = pm.gtrack_info(tname)
            assert info["type"] == "rectangles"
            assert info["dimensions"] == 2
        finally:
            _cleanup_track(tname)
