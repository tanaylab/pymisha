"""Tests for gpartition function."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(scope="module", autouse=True)
def init_db():
    """Initialize the example database."""
    pm.gdb_init_examples()


def _check_partition_invariants(result, breaks, scope_intervals=None):
    """Verify structural invariants that all gpartition results must satisfy.

    Parameters
    ----------
    result : DataFrame
        gpartition output (must not be None).
    breaks : list
        Break points passed to gpartition.
    scope_intervals : DataFrame, optional
        The intervals scope passed to gpartition (used to verify containment).
    """
    # Correct columns in order
    assert list(result.columns) == ["chrom", "start", "end", "bin"]

    # Column types
    assert result["chrom"].dtype == object  # string
    assert result["start"].dtype == np.int64
    assert result["end"].dtype == np.int64
    assert result["bin"].dtype == np.int64

    # Every interval has start < end
    assert (result["start"] < result["end"]).all(), "All intervals must have start < end"

    # All starts >= 0
    assert (result["start"] >= 0).all(), "All starts must be >= 0"

    # Bin values are 1-based and within valid range
    num_bins = len(breaks) - 1
    assert (result["bin"] >= 1).all(), "Bin values must be >= 1"
    assert (result["bin"] <= num_bins).all(), f"Bin values must be <= {num_bins}"

    # Within each chromosome: non-overlapping and sorted
    for chrom, group in result.groupby("chrom"):
        starts = group["start"].values
        ends = group["end"].values
        # Sorted by start
        assert (np.diff(starts) >= 0).all(), f"Intervals on {chrom} must be sorted by start"
        # Non-overlapping: each end <= next start
        if len(starts) > 1:
            assert (ends[:-1] <= starts[1:]).all(), f"Intervals on {chrom} must be non-overlapping"
        # Adjacent intervals with the same bin must have been merged
        # (touching intervals with the same bin should not exist)
        bins = group["bin"].values
        for i in range(len(ends) - 1):
            if ends[i] == starts[i + 1]:
                assert bins[i] != bins[i + 1], (
                    f"Adjacent touching intervals on {chrom} at position {ends[i]} "
                    f"have the same bin {bins[i]} and should have been merged"
                )

    # If scope is given, verify containment
    if scope_intervals is not None and len(scope_intervals) > 0:
        scope_end = scope_intervals["end"].max()
        scope_start = scope_intervals["start"].min()
        assert (result["start"] >= scope_start).all(), "All starts must be >= scope start"
        assert (result["end"] <= scope_end).all(), "All ends must be <= scope end"


class TestGpartitionBasic:
    """Basic tests for gpartition."""

    def test_partition_returns_intervals_with_bin_column(self):
        """gpartition returns intervals with correct columns and non-trivial content."""
        breaks = [0, 0.05, 0.1, 0.15, 0.2]
        intervals = pm.gintervals("1", 0, 5000)
        result = pm.gpartition("dense_track", breaks, intervals)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "Result should have at least one interval"
        _check_partition_invariants(result, breaks, intervals)
        # All chroms should be from the requested scope
        assert set(result["chrom"].unique()) == {"1"}

    def test_partition_bin_values_are_1_indexed(self):
        """Bin indices are 1-based and cover the expected range for 4 bins."""
        breaks = [0, 0.05, 0.1, 0.15, 0.2]
        result = pm.gpartition("dense_track", breaks, pm.gintervals("1", 0, 5000))

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # With these breaks there are 4 bins; the test DB should produce values in all 4
        unique_bins = set(result["bin"].unique())
        assert unique_bins == {1, 2, 3, 4}, f"Expected all 4 bins, got {unique_bins}"

    def test_partition_merges_adjacent_same_bin(self):
        """Adjacent intervals with same bin value are merged into larger intervals."""
        # Use coarse breaks so most values fall in bin 1
        breaks = [0, 0.5, 1.0]
        intervals = pm.gintervals("1", 0, 10000)
        result = pm.gpartition("dense_track", breaks, intervals)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        _check_partition_invariants(result, breaks, intervals)
        # With these coarse breaks, all values are < 0.5 so everything goes to bin 1.
        # The merging should produce intervals with gaps (NaN regions excluded).
        assert (result["bin"] == 1).all(), "All values should fall in bin 1 for coarse breaks"
        # The merging means we get fewer intervals than the 10000/50=200 raw positions
        assert len(result) < 100, "Merged result should have far fewer than raw extraction rows"

    def test_partition_respects_include_lowest(self):
        """include_lowest=True includes the lowest break value, capturing more coverage."""
        breaks = [0, 0.1, 0.2]
        intervals = pm.gintervals("1", 0, 5000)

        result_no_lowest = pm.gpartition("dense_track", breaks, intervals, include_lowest=False)
        result_with_lowest = pm.gpartition("dense_track", breaks, intervals, include_lowest=True)

        # Both must return valid DataFrames
        assert isinstance(result_no_lowest, pd.DataFrame)
        assert isinstance(result_with_lowest, pd.DataFrame)
        assert len(result_no_lowest) > 0
        assert len(result_with_lowest) > 0
        _check_partition_invariants(result_no_lowest, breaks, intervals)
        _check_partition_invariants(result_with_lowest, breaks, intervals)

        # include_lowest=True uses [0, 0.1] for bin 1 instead of (0, 0.1],
        # so zero-valued positions are captured, giving more total coverage.
        coverage_no = (result_no_lowest["end"] - result_no_lowest["start"]).sum()
        coverage_yes = (result_with_lowest["end"] - result_with_lowest["start"]).sum()
        assert coverage_yes >= coverage_no, (
            f"include_lowest=True should capture at least as much coverage "
            f"({coverage_yes} vs {coverage_no})"
        )


class TestGpartitionEdgeCases:
    """Edge case tests for gpartition."""

    def test_partition_empty_intervals_returns_none(self):
        """gpartition with empty intervals returns None."""
        breaks = [0, 0.1, 0.2]
        # Use an empty DataFrame as intervals
        empty_intervals = pd.DataFrame(columns=["chrom", "start", "end"])
        result = pm.gpartition("dense_track", breaks, empty_intervals)

        # Should return None or empty DataFrame for no data
        assert result is None or len(result) == 0

    def test_partition_requires_breaks(self):
        """gpartition requires breaks argument."""
        with pytest.raises((ValueError, TypeError)):
            pm.gpartition("dense_track", None, pm.gintervals("1", 0, 5000))

    def test_partition_requires_at_least_two_breaks(self):
        """gpartition requires at least 2 break values (one bin)."""
        with pytest.raises(ValueError):
            pm.gpartition("dense_track", [0.5], pm.gintervals("1", 0, 5000))

    def test_partition_values_outside_breaks_excluded(self):
        """Values outside break range are excluded from result (returns None)."""
        # Dense track has values in ~[0, 0.2]; breaks [0.5, 0.6, 0.7] exclude everything
        breaks = [0.5, 0.6, 0.7]
        result = pm.gpartition("dense_track", breaks, pm.gintervals("1", 0, 5000))

        assert result is None, (
            "When all track values are outside the break range, gpartition should return None"
        )

    def test_partition_with_iterator(self):
        """Larger iterator steps produce fewer, coarser intervals."""
        breaks = [0, 0.1, 0.2]
        intervals = pm.gintervals("1", 0, 5000)

        result_default = pm.gpartition("dense_track", breaks, intervals)
        result_coarse = pm.gpartition("dense_track", breaks, intervals, iterator=500)

        assert isinstance(result_default, pd.DataFrame)
        assert isinstance(result_coarse, pd.DataFrame)
        assert len(result_default) > 0
        assert len(result_coarse) > 0
        _check_partition_invariants(result_default, breaks, intervals)
        _check_partition_invariants(result_coarse, breaks, intervals)

        # A coarser iterator should produce fewer partition intervals
        assert len(result_coarse) < len(result_default), (
            f"Coarse iterator should give fewer intervals "
            f"({len(result_coarse)} vs {len(result_default)})"
        )


class TestGpartitionMatchesR:
    """Golden master tests comparing to R implementation."""

    def test_partition_basic_matches_r(self):
        """Basic gpartition matches R output structure and expected values."""
        breaks = [0, 0.05, 0.1, 0.15, 0.2]
        intervals = pm.gintervals("1", 0, 5000)
        result = pm.gpartition("dense_track", breaks, intervals)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        _check_partition_invariants(result, breaks, intervals)

        # Bins should be int64 (C++ builds NPY_INT64 arrays)
        assert result["bin"].dtype == np.int64
        # All bin values should be valid (1 to num_bins)
        valid_bins = set(range(1, len(breaks)))
        assert set(result["bin"].unique()).issubset(valid_bins)
        # Exact row count from known test DB
        assert len(result) == 49, f"Expected 49 intervals, got {len(result)}"

    def test_partition_all_chroms(self):
        """gpartition over all chromosomes returns data from every chrom."""
        breaks = [0, 0.05, 0.1, 0.15, 0.2]
        result = pm.gpartition("dense_track", breaks)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        _check_partition_invariants(result, breaks)
        # Test DB has chroms 1, 2, X
        assert set(result["chrom"].unique()) == {"1", "2", "X"}
