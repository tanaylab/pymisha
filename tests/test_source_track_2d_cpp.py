"""Tests for pm_read_source_track_2d (G1.P3.D)."""
from __future__ import annotations

import _pymisha
import pytest

from pymisha._quadtree import write_2d_track_file


def _expect_arrays(result, n):
    """Sanity-check the dict structure and dimensionality of the result."""
    assert isinstance(result, dict)
    for k in ("chrom1", "chrom2", "x1", "y1", "x2", "y2", "value"):
        assert k in result, f"missing key {k}"
        assert len(result[k]) == n
    assert "is_points" in result


class TestReadSourceTrack2dArgs:
    def test_missing_directory_raises(self, tmp_path):
        nonexistent = tmp_path / "does-not-exist"
        with pytest.raises(RuntimeError):
            _pymisha.pm_read_source_track_2d(str(nonexistent))

    def test_file_instead_of_dir_raises(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("hi")
        with pytest.raises(RuntimeError):
            _pymisha.pm_read_source_track_2d(str(f))


class TestReadSourceTrack2dEmpty:
    def test_empty_directory(self, tmp_path):
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 0)
        # is_points defaults to False for empty dirs.
        assert r["is_points"] is False

    def test_only_dotfiles_ignored(self, tmp_path):
        (tmp_path / ".attributes").write_text("created.by: foo")
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 0)


class TestReadSourceTrack2dRects:
    def test_single_rects_pair(self, tmp_path):
        rects = [
            (10, 20, 30, 40, 1.5),
            (100, 200, 300, 400, 2.5),
        ]
        write_2d_track_file(
            str(tmp_path / "chr1-chr2"),
            rects,
            (0, 0, 1000, 1000),
            is_points=False,
        )
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 2)
        assert r["is_points"] is False
        got = sorted(zip(r["chrom1"], r["chrom2"], r["x1"], r["y1"], r["x2"], r["y2"], r["value"]))
        expected = sorted([
            ("chr1", "chr2", 10, 20, 30, 40, 1.5),
            ("chr1", "chr2", 100, 200, 300, 400, 2.5),
        ])
        assert got == expected

    def test_multi_pair_rects(self, tmp_path):
        write_2d_track_file(
            str(tmp_path / "1-2"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        write_2d_track_file(
            str(tmp_path / "3-3"),
            [(5, 5, 15, 15, 9.0)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 2)
        # Build set of unordered rows to compare against expected
        got = sorted(zip(r["chrom1"], r["chrom2"], r["x1"], r["y1"], r["x2"], r["y2"], r["value"]))
        assert got == sorted([
            ("1", "2", 10, 20, 30, 40, 1.5),
            ("3", "3", 5, 5, 15, 15, 9.0),
        ])

    def test_empty_pair_file_emits_no_rows(self, tmp_path):
        # Empty per-pair file still has a valid signature header but num_objs=0.
        write_2d_track_file(
            str(tmp_path / "1-1"),
            [],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 0)
        assert r["is_points"] is False


class TestReadSourceTrack2dPoints:
    def test_single_points_pair(self, tmp_path):
        points = [(10, 20, 1.5), (100, 200, 2.5)]
        write_2d_track_file(
            str(tmp_path / "1-1"),
            points,
            (0, 0, 1000, 1000),
            is_points=True,
        )
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 2)
        assert r["is_points"] is True
        # POINTS objects are returned as (x, x+1, y, y+1, val).
        got = sorted(zip(r["x1"], r["y1"], r["x2"], r["y2"], r["value"]))
        assert got == sorted([(10, 20, 11, 21, 1.5), (100, 200, 101, 201, 2.5)])


class TestReadSourceTrack2dMixedRejected:
    def test_mixed_rects_and_points_raises(self, tmp_path):
        write_2d_track_file(
            str(tmp_path / "1-1"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        write_2d_track_file(
            str(tmp_path / "1-2"),
            [(10, 20, 1.5)],
            (0, 0, 1000, 1000),
            is_points=True,
        )
        with pytest.raises(ValueError, match="Mixed"):
            _pymisha.pm_read_source_track_2d(str(tmp_path))


class TestReadSourceTrack2dCoexistingNon2DFiles:
    """Files that aren't 2D quadtree files should be silently skipped."""

    def test_dense_1d_file_skipped(self, tmp_path):
        import struct as _s

        # Write a non-2D file masquerading as a track file. signature > 0 means
        # dense 1D so it should be silently ignored by the 2D reader.
        with open(tmp_path / "irrelevant", "wb") as f:
            f.write(_s.pack("<i", 100))
            f.write(_s.pack("<f", 1.0))

        # Also write a real 2D pair so we have something to find.
        write_2d_track_file(
            str(tmp_path / "1-2"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        r = _pymisha.pm_read_source_track_2d(str(tmp_path))
        _expect_arrays(r, 1)
        assert r["chrom1"][0] == "1"
        assert r["chrom2"][0] == "2"
