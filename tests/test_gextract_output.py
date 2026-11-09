"""Tests for gextract file and intervals_set_out parameters."""

import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


# ── 1D file output ───────────────────────────────────────────────────

class TestGextractFileOutput1D:
    """Test writing 1D extraction results to a TSV file."""

    def test_file_writes_tsv(self, tmp_path):
        """gextract(file=...) writes a readable TSV file."""
        out = tmp_path / "out.tsv"
        intervals = pm.gintervals("1", 0, 1000)
        result = pm.gextract(
            "dense_track", intervals, iterator=200, file=str(out), progress=False
        )
        assert result is None
        assert out.exists()
        df = pd.read_csv(out, sep="\t")
        assert "chrom" in df.columns
        assert "dense_track" in df.columns
        assert len(df) == 5

    def test_file_returns_none(self, tmp_path):
        """gextract(file=...) returns None even when data exists."""
        out = tmp_path / "out.tsv"
        intervals = pm.gintervals("1", 0, 500)
        result = pm.gextract(
            "dense_track", intervals, iterator=500, file=str(out), progress=False
        )
        assert result is None

    def test_file_content_matches_dataframe(self, tmp_path):
        """TSV file content matches the in-memory DataFrame's value columns.

        R parity: ``gextract(file=)`` writes only the iterator coordinate
        columns + expression columns -- no ``intervalID`` (see
        ``GenomeTrackExtract.cpp`` in R misha). The in-memory DataFrame
        still carries ``intervalID``.
        """
        out = tmp_path / "out.tsv"
        intervals = pm.gintervals("1", 0, 2000)

        df_expected = pm.gextract(
            "dense_track", intervals, iterator=500, progress=False
        )
        pm.gextract(
            "dense_track", intervals, iterator=500, file=str(out), progress=False
        )
        df_from_file = pd.read_csv(out, sep="\t")

        expected_cols = [c for c in df_expected.columns if c != "intervalID"]
        assert list(df_from_file.columns) == expected_cols
        assert len(df_expected) == len(df_from_file)
        np.testing.assert_array_almost_equal(
            df_expected["dense_track"].values,
            df_from_file["dense_track"].values,
        )

    def test_file_empty_extraction(self, tmp_path):
        """gextract(file=...) with empty result still returns None."""
        out = tmp_path / "out.tsv"
        empty_df = pd.DataFrame(
            {"chrom": pd.Series([], dtype=str),
             "start": pd.Series([], dtype=int),
             "end": pd.Series([], dtype=int)}
        )
        result = pm.gextract(
            "dense_track", empty_df, iterator=100, file=str(out), progress=False
        )
        assert result is None

    def test_file_mode(self, tmp_path):
        """R parity: R misha writes the file from C++ under umask 007, so a new file is
        0660 whatever the caller's umask, and a file it replaces is written in place and
        keeps its permission bits."""
        intervals = pm.gintervals("1", 0, 1000)
        new = tmp_path / "new.tsv"
        old_file = tmp_path / "old.tsv"
        old_file.write_text("existing\n")
        os.chmod(old_file, 0o604)
        prev = os.umask(0o022)
        try:
            pm.gextract("dense_track", intervals, iterator=200, file=str(new), progress=False)
            pm.gextract("dense_track", intervals, iterator=200, file=str(old_file), progress=False)
        finally:
            os.umask(prev)
        assert new.stat().st_mode & 0o777 == 0o660
        assert old_file.stat().st_mode & 0o777 == 0o604
        assert len(pd.read_csv(old_file, sep="\t")) == 5


# ── 1D intervals_set_out ─────────────────────────────────────────────

class TestGextractIntervalsSetOut1D:
    """Test saving 1D extraction coordinate columns as a named interval set."""

    def test_intervals_set_out_creates_loadable_set(self):
        """gextract(intervals_set_out=...) creates an interval set that gintervals_load can read."""
        iset_name = "test.gextract_iset_1d"
        try:
            intervals = pm.gintervals("1", 0, 1000)
            result = pm.gextract(
                "dense_track", intervals, iterator=200,
                intervals_set_out=iset_name, progress=False
            )
            # R parity: nothing is returned when the result goes to an interval set
            assert result is None

            loaded = pm.gintervals_load(iset_name)
            assert loaded is not None
            assert len(loaded) == 5
            assert "chrom" in loaded.columns
            assert "start" in loaded.columns
            assert "end" in loaded.columns
        finally:
            pm.gintervals_rm(iset_name, force=True)

    def test_intervals_set_out_coords_match(self):
        """Saved interval set coordinates must match extraction output."""
        iset_name = "test.gextract_iset_coords"
        try:
            intervals = pm.gintervals("1", 0, 2000)
            df = pm.gextract("dense_track", intervals, iterator=500, progress=False)
            assert pm.gextract(
                "dense_track", intervals, iterator=500,
                intervals_set_out=iset_name, progress=False
            ) is None
            loaded = pm.gintervals_load(iset_name)
            # The saved coords should be deduplicated extraction output coords
            expected_coords = (
                df[["chrom", "start", "end"]]
                .drop_duplicates()
                .sort_values(["chrom", "start", "end"])
                .reset_index(drop=True)
            )
            loaded_sorted = (
                loaded[["chrom", "start", "end"]]
                .sort_values(["chrom", "start", "end"])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(
                expected_coords, loaded_sorted, check_dtype=False
            )
        finally:
            pm.gintervals_rm(iset_name, force=True)


# ── file + intervals_set_out: an error, as R misha ───────────────────

class TestGextractBothParams:
    """R misha refuses file and intervals.set.out together, before any work."""

    @pytest.mark.parametrize("dim", [1, 2])
    def test_file_and_intervals_set_out_together_is_an_error(self, tmp_path, dim):
        out = tmp_path / "both.tsv"
        iset_name = "test.gextract_both"
        if dim == 1:
            args = ("dense_track", pm.gintervals("1", 0, 1000))
            kw = {"iterator": 200}
        else:
            args = ("rects_track", pm.gintervals_2d("1", 0, 500000, "1", 0, 500000))
            kw = {}
        with pytest.raises(ValueError, match="Cannot use both file and intervals_set_out arguments"):
            pm.gextract(*args, file=str(out), intervals_set_out=iset_name, progress=False, **kw)
        assert not out.exists()
        assert not pm.gintervals_exists(iset_name)


# ── 2D file output ───────────────────────────────────────────────────

class TestGextractFileOutput2D:
    """Test writing 2D extraction results to a TSV file."""

    def test_2d_file_writes_tsv(self, tmp_path):
        """gextract(file=...) writes a readable TSV file for 2D extraction."""
        out = tmp_path / "out2d.tsv"
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract(
            "rects_track", intervals, file=str(out), progress=False
        )
        assert result is None
        assert out.exists()
        df = pd.read_csv(out, sep="\t")
        assert "chrom1" in df.columns
        assert "start1" in df.columns
        assert "rects_track" in df.columns
        assert len(df) > 0

    def test_2d_file_content_matches_dataframe(self, tmp_path):
        """TSV file content for 2D extraction matches the DataFrame value columns.

        R parity: ``gextract(file=)`` omits ``intervalID`` (see
        ``GenomeTrackExtract.cpp`` in R misha).
        """
        out = tmp_path / "out2d.tsv"
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        df_expected = pm.gextract("rects_track", intervals, progress=False)
        pm.gextract("rects_track", intervals, file=str(out), progress=False)
        df_from_file = pd.read_csv(out, sep="\t")

        expected_cols = [c for c in df_expected.columns if c != "intervalID"]
        assert list(df_from_file.columns) == expected_cols
        assert len(df_expected) == len(df_from_file)


# ── 2D intervals_set_out ─────────────────────────────────────────────

class TestGextractIntervalsSetOut2D:
    """Test saving 2D extraction coordinate columns as a named interval set."""

    def test_2d_intervals_set_out_creates_loadable_set(self):
        """gextract(intervals_set_out=...) for 2D extraction creates loadable interval set."""
        iset_name = "test.gextract_iset_2d"
        try:
            intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            result = pm.gextract(
                "rects_track", intervals,
                intervals_set_out=iset_name, progress=False
            )
            assert result is None

            loaded = pm.gintervals_load(iset_name)
            assert loaded is not None
            assert len(loaded) > 0
            assert "chrom1" in loaded.columns
            assert "start1" in loaded.columns
            assert "end2" in loaded.columns
        finally:
            pm.gintervals_rm(iset_name, force=True)
