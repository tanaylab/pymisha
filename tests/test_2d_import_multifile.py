"""Tests for gtrack_2d_import multi-file support (GAP-041)."""

import os
import shutil

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


class TestGtrack2dImportMultiFile:
    """Tests for gtrack_2d_import with single and multiple files."""

    def _cleanup(self, name):
        _cleanup_track(name)

    def test_single_file_string(self, tmp_path):
        """Single file (str) still works -- existing behaviour unchanged."""
        tname = "test.test_2d_imp_single"
        self._cleanup(tname)
        try:
            tsv = tmp_path / "data.tsv"
            tsv.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t100\t200\t1\t300\t400\t5.5\n"
                "1\t1000\t2000\t2\t5000\t6000\t3.0\n"
            )
            pm.gtrack_2d_import(tname, "single file", str(tsv))
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "rectangles"
        finally:
            self._cleanup(tname)

    def test_list_of_two_files(self, tmp_path):
        """Two files are concatenated into one track."""
        tname = "test.test_2d_imp_multi"
        self._cleanup(tname)
        try:
            tsv1 = tmp_path / "a.tsv"
            tsv1.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t100\t200\t1\t300\t400\t5.5\n"
            )
            tsv2 = tmp_path / "b.tsv"
            tsv2.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t1000\t2000\t2\t5000\t6000\t3.0\n"
            )
            pm.gtrack_2d_import(tname, "two files", [str(tsv1), str(tsv2)])
            assert pm.gtrack_exists(tname)

            # Extract and verify we have data from both files
            scope = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            df = pm.gextract(tname, scope)
            # At least 1 row from first file (1:100-200 x 1:300-400)
            assert len(df) >= 1

            # Also check the cross-chrom pair from second file
            scope2 = pm.gintervals_2d("1", 0, 500000, "2", 0, 300000)
            df2 = pm.gextract(tname, scope2)
            assert len(df2) >= 1
        finally:
            self._cleanup(tname)

    def test_list_of_three_files_points(self, tmp_path):
        """Three files with point intervals produce a points track."""
        tname = "test.test_2d_imp_multi3"
        self._cleanup(tname)
        try:
            files = []
            for i, (c1, s1, c2, s2, v) in enumerate([
                ("1", 100, "1", 300, 1.0),
                ("1", 2000, "1", 5000, 2.0),
                ("1", 10000, "1", 80000, 3.0),
            ]):
                tsv = tmp_path / f"f{i}.tsv"
                tsv.write_text(
                    "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                    f"{c1}\t{s1}\t{s1 + 1}\t{c2}\t{s2}\t{s2 + 1}\t{v}\n"
                )
                files.append(str(tsv))

            pm.gtrack_2d_import(tname, "three files", files)
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"

            scope = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            df = pm.gextract(tname, scope)
            assert len(df) == 3
        finally:
            self._cleanup(tname)

    def test_empty_list_raises(self):
        """Empty file list raises ValueError."""
        tname = "test.test_2d_imp_empty"
        self._cleanup(tname)
        try:
            with pytest.raises(ValueError, match="At least one file"):
                pm.gtrack_2d_import(tname, "empty", [])
        finally:
            self._cleanup(tname)

    def test_invalid_file_in_list_raises(self, tmp_path):
        """A non-existent file in the list raises ValueError."""
        tname = "test.test_2d_imp_badfile"
        self._cleanup(tname)
        try:
            good = tmp_path / "good.tsv"
            good.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t100\t200\t1\t300\t400\t5.5\n"
            )
            with pytest.raises(ValueError, match="File not found"):
                pm.gtrack_2d_import(
                    tname, "bad", [str(good), "/no/such/file.tsv"]
                )
        finally:
            self._cleanup(tname)

    def test_single_file_in_list(self, tmp_path):
        """A list with one element works the same as a plain string."""
        tname = "test.test_2d_imp_list1"
        self._cleanup(tname)
        try:
            tsv = tmp_path / "only.tsv"
            tsv.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t100\t200\t1\t300\t400\t5.5\n"
            )
            pm.gtrack_2d_import(tname, "list of one", [str(tsv)])
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "rectangles"
        finally:
            self._cleanup(tname)
