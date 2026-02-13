"""Tests for gintervals_load and gintervals_save functions."""


import numpy as np
import pandas as pd
import pytest

import pymisha as pm


class TestGintervalsLoad:
    """Test gintervals_load function."""

    def test_load_existing_interval_set(self):
        """Load an existing interval set returns correct DataFrame."""
        result = pm.gintervals_load("annotations")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns
        # annotations has 8 intervals
        assert len(result) == 8

    def test_load_returns_correct_columns(self):
        """Loaded interval set has expected columns including extra ones."""
        result = pm.gintervals_load("annotations")
        # annotations has strand and remark columns
        assert "strand" in result.columns
        assert "remark" in result.columns

    def test_load_values_match_expected(self):
        """Loaded interval set has correct values."""
        result = pm.gintervals_load("annotations")
        # First interval should be chr1:20-2000
        first = result.iloc[0]
        assert str(first["chrom"]) == "1"
        assert first["start"] == 20
        assert first["end"] == 2000
        assert first["strand"] == 1

    def test_load_nonexistent_interval_set_raises(self):
        """Loading non-existent interval set raises appropriate error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_load("nonexistent_intervals")

    def test_load_with_chrom_filter(self):
        """Load interval set filtered by chromosome."""
        result = pm.gintervals_load("annotations", chrom="1")
        assert result is not None
        # All intervals should be on chr1
        assert all(str(c) == "1" for c in result["chrom"])

    def test_load_with_chrom_filter_no_match(self):
        """Load with chromosome filter that matches nothing returns empty."""
        # annotations has chr1, chr2 - chrX should return empty
        result = pm.gintervals_load("annotations", chrom="chrX")
        # Should return empty DataFrame or None
        assert result is None or len(result) == 0


class TestGintervalsSave:
    """Test gintervals_save function."""

    def test_save_basic(self):
        """Save a simple interval set to the database."""
        # Create test intervals (using chroms "1", "2" matching test DB)
        intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])

        # Save to a new name
        pm.gintervals_save(intervals, "test_save_basic")

        # Verify it was saved
        assert pm.gintervals_exists("test_save_basic")

        # Clean up
        pm.gintervals_rm("test_save_basic")

    def test_save_and_load_roundtrip(self):
        """Saved intervals can be loaded back correctly."""
        # Create intervals with specific values (using chroms "1", "2" matching test DB)
        intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])

        # Save
        pm.gintervals_save(intervals, "test_roundtrip")

        # Load back
        loaded = pm.gintervals_load("test_roundtrip")

        # Verify
        assert loaded is not None
        assert len(loaded) == 2
        # Check values (sorted by chrom, start)
        assert loaded.iloc[0]["chrom"] == "1" or str(loaded.iloc[0]["chrom"]) == "1"
        assert loaded.iloc[0]["start"] == 100
        assert loaded.iloc[0]["end"] == 1000

        # Clean up
        pm.gintervals_rm("test_roundtrip")

    def test_save_with_extra_columns(self):
        """Save intervals with additional columns preserves them."""
        # Create intervals with extra column (using chroms "1", "2" matching test DB)
        df = pd.DataFrame({
            "chrom": ["1", "2"],
            "start": [100, 200],
            "end": [1000, 2000],
            "score": [1.5, 2.5],
            "name": ["gene1", "gene2"]
        })

        pm.gintervals_save(df, "test_extra_cols")
        loaded = pm.gintervals_load("test_extra_cols")

        assert "score" in loaded.columns
        assert "name" in loaded.columns
        assert loaded.iloc[0]["score"] == 1.5 or np.isclose(loaded.iloc[0]["score"], 1.5)

        pm.gintervals_rm("test_extra_cols")

    def test_save_existing_raises(self):
        """Saving to an existing interval set name raises error."""
        intervals = pm.gintervals("1", 100, 1000)

        # First save should work
        pm.gintervals_save(intervals, "test_dup")

        # Second save to same name should fail
        with pytest.raises(ValueError, match="already exists"):
            pm.gintervals_save(intervals, "test_dup")

        pm.gintervals_rm("test_dup")

    def test_save_invalid_name_raises(self):
        """Invalid interval set names are rejected."""
        intervals = pm.gintervals("1", 100, 1000)

        with pytest.raises(ValueError):
            pm.gintervals_save(intervals, "123invalid")  # starts with number

        with pytest.raises(ValueError):
            pm.gintervals_save(intervals, "has spaces")  # contains space


class TestGintervalsRm:
    """Test gintervals_rm function."""

    def test_rm_existing(self):
        """Remove an existing interval set."""
        # Create and save
        intervals = pm.gintervals("1", 100, 1000)
        pm.gintervals_save(intervals, "test_rm")
        assert pm.gintervals_exists("test_rm")

        # Remove
        pm.gintervals_rm("test_rm")
        assert not pm.gintervals_exists("test_rm")

    def test_rm_nonexistent_raises(self):
        """Removing non-existent interval set raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_rm("nonexistent_set")

    def test_rm_with_force_nonexistent(self):
        """Remove with force=True on non-existent set doesn't raise."""
        # Should not raise
        pm.gintervals_rm("nonexistent_set", force=True)


class TestGintervalsLoadGoldenMaster:
    """Golden master tests comparing with R misha output."""

    def test_load_annotations_matches_r(self):
        """gintervals_load returns same data as R gintervals.load."""
        result = pm.gintervals_load("annotations")

        # Expected values from R misha (verified with the interv file)
        expected_chroms = ["1", "1", "2", "2", "2", "2", "2", "2"]
        expected_starts = [20, 2500, 20, 3000, 9000, 12000, 13000, 15000]
        expected_ends = [2000, 2600, 2000, 8000, 11000, 12001, 14000, 15500]

        assert len(result) == len(expected_chroms)
        for i, (chrom, start, end) in enumerate(zip(expected_chroms, expected_starts, expected_ends, strict=False)):
            assert str(result.iloc[i]["chrom"]) == chrom
            assert result.iloc[i]["start"] == start
            assert result.iloc[i]["end"] == end
