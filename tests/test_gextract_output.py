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
        """TSV file content must match the DataFrame that would have been returned."""
        out = tmp_path / "out.tsv"
        intervals = pm.gintervals("1", 0, 2000)

        df_expected = pm.gextract(
            "dense_track", intervals, iterator=500, progress=False
        )
        pm.gextract(
            "dense_track", intervals, iterator=500, file=str(out), progress=False
        )
        df_from_file = pd.read_csv(out, sep="\t")

        assert list(df_expected.columns) == list(df_from_file.columns)
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


# ── 1D intervals_set_out ─────────────────────────────────────────────

class TestGextractIntervalsSetOut1D:
    """Test saving 1D extraction coordinate columns as a named interval set."""

    def test_intervals_set_out_creates_loadable_set(self):
        """gextract(intervals_set_out=...) creates an interval set that gintervals_load can read."""
        iset_name = "test.gextract_iset_1d"
        try:
            intervals = pm.gintervals("1", 0, 1000)
            df = pm.gextract(
                "dense_track", intervals, iterator=200,
                intervals_set_out=iset_name, progress=False
            )
            # Should still return the DataFrame
            assert df is not None
            assert len(df) == 5

            loaded = pm.gintervals_load(iset_name)
            assert loaded is not None
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
            df = pm.gextract(
                "dense_track", intervals, iterator=500,
                intervals_set_out=iset_name, progress=False
            )
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


# ── 1D both file + intervals_set_out ─────────────────────────────────

class TestGextractBothParams1D:
    """Test combining file and intervals_set_out for 1D extraction."""

    def test_file_and_intervals_set_out_together(self, tmp_path):
        """Both file and intervals_set_out work simultaneously."""
        out = tmp_path / "both.tsv"
        iset_name = "test.gextract_both_1d"
        try:
            intervals = pm.gintervals("1", 0, 1000)
            result = pm.gextract(
                "dense_track", intervals, iterator=200,
                file=str(out), intervals_set_out=iset_name, progress=False
            )
            # file takes precedence -> returns None
            assert result is None
            # file was written
            assert out.exists()
            df_file = pd.read_csv(out, sep="\t")
            assert len(df_file) == 5
            # interval set was saved
            loaded = pm.gintervals_load(iset_name)
            assert loaded is not None
            assert len(loaded) > 0
        finally:
            pm.gintervals_rm(iset_name, force=True)


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
        """TSV file content for 2D extraction matches DataFrame result."""
        out = tmp_path / "out2d.tsv"
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        df_expected = pm.gextract("rects_track", intervals, progress=False)
        pm.gextract("rects_track", intervals, file=str(out), progress=False)
        df_from_file = pd.read_csv(out, sep="\t")

        assert list(df_expected.columns) == list(df_from_file.columns)
        assert len(df_expected) == len(df_from_file)


# ── 2D intervals_set_out ─────────────────────────────────────────────

class TestGextractIntervalsSetOut2D:
    """Test saving 2D extraction coordinate columns as a named interval set."""

    def test_2d_intervals_set_out_creates_loadable_set(self):
        """gextract(intervals_set_out=...) for 2D extraction creates loadable interval set."""
        iset_name = "test.gextract_iset_2d"
        try:
            intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
            df = pm.gextract(
                "rects_track", intervals,
                intervals_set_out=iset_name, progress=False
            )
            assert df is not None
            assert len(df) > 0

            loaded = pm.gintervals_load(iset_name)
            assert loaded is not None
            assert "chrom1" in loaded.columns
            assert "start1" in loaded.columns
            assert "end2" in loaded.columns
        finally:
            pm.gintervals_rm(iset_name, force=True)
