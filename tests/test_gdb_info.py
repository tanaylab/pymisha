"""Tests for gdb_info (database inspection metadata)."""

import pandas as pd
import pytest

import pymisha as pm


def test_gdb_info_uses_active_db():
    """Without a path, gdb_info should inspect the active DB."""
    info = pm.gdb_info()
    assert info["is_db"] is True
    assert info["format"] in {"indexed", "per-chromosome"}
    assert info["num_chromosomes"] > 0
    assert info["genome_size"] > 0
    assert isinstance(info["chromosomes"], pd.DataFrame)
    assert list(info["chromosomes"].columns) == ["chrom", "size"]


def test_gdb_info_nonexistent_dir():
    """Nonexistent paths should return is_db=False with an error message."""
    info = pm.gdb_info("/tmp/does_not_exist_pymisha_12345")
    assert info["is_db"] is False
    assert "does not exist" in info["error"]


def test_gdb_info_not_misha_db(tmp_path):
    """Directories without chrom_sizes.txt are not considered misha DBs."""
    root = tmp_path / "notadb"
    root.mkdir()

    info = pm.gdb_info(str(root))
    assert info["is_db"] is False
    assert "chrom_sizes.txt" in info["error"]


def test_gdb_info_per_chromosome_format(tmp_path):
    """DB with chrom_sizes but without seq/genome.idx+seq is per-chromosome."""
    root = tmp_path / "db"
    (root / "seq").mkdir(parents=True)
    (root / "tracks").mkdir(parents=True)
    (root / "chrom_sizes.txt").write_text("chr1\t1000\nchr2\t2000\n", encoding="utf-8")

    info = pm.gdb_info(str(root))
    assert info["is_db"] is True
    assert info["format"] == "per-chromosome"
    assert info["num_chromosomes"] == 2
    assert info["genome_size"] == 3000


def test_gdb_info_indexed_format(tmp_path):
    """DB with seq/genome.idx and seq/genome.seq is indexed."""
    root = tmp_path / "db"
    seq_dir = root / "seq"
    seq_dir.mkdir(parents=True)
    (root / "tracks").mkdir(parents=True)
    (root / "chrom_sizes.txt").write_text("chr1\t1000\n", encoding="utf-8")
    (seq_dir / "genome.idx").write_bytes(b"idx")
    (seq_dir / "genome.seq").write_bytes(b"seq")

    info = pm.gdb_info(str(root))
    assert info["is_db"] is True
    assert info["format"] == "indexed"


def test_gdb_info_requires_active_db_when_path_not_given():
    """If no active DB and no path, function should raise."""
    old_root = pm._shared._GROOT
    try:
        pm._shared._GROOT = None
        with pytest.raises(ValueError, match="No database is currently active"):
            pm.gdb_info()
    finally:
        pm._shared._GROOT = old_root


# ============================================================================
# Tests for gtrack_info (ported from R test-gtrack.info.R)
# ============================================================================

class TestGtrackInfo:
    """Tests for gtrack_info function.

    Ported from R misha test-gtrack.info.R.
    Test DB tracks: dense_track, sparse_track, array_track, rects_track.
    """

    def test_gtrack_info_dense_track(self):
        """gtrack_info returns correct metadata for a dense track."""
        info = pm.gtrack_info("dense_track")
        assert info["type"] == "dense"
        assert info["dimensions"] == 1
        assert info["format"] == "per-chromosome"
        assert info["size_in_bytes"] > 0

    def test_gtrack_info_dense_track_bin_size(self):
        """Dense track has bin_size in gtrack_info result."""
        info = pm.gtrack_info("dense_track")
        assert "bin_size" in info
        assert info["bin_size"] == 50

    def test_gtrack_info_sparse_track(self):
        """gtrack_info returns correct metadata for a sparse track."""
        info = pm.gtrack_info("sparse_track")
        assert info["type"] == "sparse"
        assert info["dimensions"] == 1
        assert info["format"] == "per-chromosome"
        assert info["size_in_bytes"] > 0

    def test_gtrack_info_sparse_track_no_bin_size(self):
        """Sparse track does not have bin_size in gtrack_info result."""
        info = pm.gtrack_info("sparse_track")
        assert "bin_size" not in info

    def test_gtrack_info_array_track(self):
        """gtrack_info returns correct metadata for an array track."""
        info = pm.gtrack_info("array_track")
        assert info["type"] == "array"
        assert info["dimensions"] == 1
        assert info["format"] == "per-chromosome"
        assert info["size_in_bytes"] > 0

    def test_gtrack_info_2d_rects_track(self):
        """gtrack_info returns correct metadata for a 2D rectangles track."""
        info = pm.gtrack_info("rects_track")
        assert info["type"] == "rectangles"
        assert info["dimensions"] == 2
        assert info["format"] == "per-chromosome"

    def test_gtrack_info_dense_track2_in_subdir(self):
        """gtrack_info works for tracks in subdirectories."""
        info = pm.gtrack_info("subdir.dense_track2")
        assert info["type"] == "dense"
        assert info["dimensions"] == 1
        assert "bin_size" in info
        assert info["bin_size"] == 50

    def test_gtrack_info_returns_attributes(self):
        """gtrack_info includes track attributes."""
        info = pm.gtrack_info("dense_track")
        assert "attributes" in info
        assert isinstance(info["attributes"], dict)
        # Dense track should have created.by and created.date
        assert "created.by" in info["attributes"]
        assert "created.date" in info["attributes"]

    def test_gtrack_info_nonexistent_track_raises(self):
        """gtrack_info raises error for non-existent track."""
        with pytest.raises(Exception):
            pm.gtrack_info("nonexistent_track_xyz")
