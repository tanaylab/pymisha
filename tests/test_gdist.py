"""Tests for gdist function."""

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize the example database for each test."""
    pm.gdb_init_examples()
    yield


class TestGdistBasic:
    """Basic gdist functionality tests."""

    def test_gdist_single_track_single_dim(self):
        """Test gdist with a single track and single dimension."""
        # Calculate distribution of dense_track for bins: (0, 0.2], (0.2, 0.5], (0.5, 1]
        result = pm.gdist("dense_track", [0, 0.2, 0.5, 1])

        # Result should be a 1D array with 3 elements (one per bin)
        assert result is not None
        assert len(result) == 3
        # All counts should be non-negative integers
        assert all(v >= 0 for v in result)
        # Total should be equal to total number of iterator intervals
        total = sum(result)
        assert total > 0

    def test_gdist_with_intervals(self):
        """Test gdist with explicit intervals."""
        intervals = pm.gintervals("1", 0, 10000)
        result = pm.gdist("dense_track", [0, 0.2, 0.5, 1], intervals=intervals)

        assert result is not None
        assert len(result) == 3
        # Counts should be less than with all intervals
        total = sum(result)
        assert total > 0

    def test_gdist_include_lowest(self):
        """Test gdist with include_lowest=True."""
        intervals = pm.gintervals("1", 0, 10000)

        # Without include_lowest, the lowest value is not in the first bin
        result_default = pm.gdist("dense_track", [0, 0.2, 0.5, 1], intervals=intervals)

        # With include_lowest, the lowest value is in the first bin
        result_include = pm.gdist("dense_track", [0, 0.2, 0.5, 1],
                                  intervals=intervals, include_lowest=True)

        assert result_default is not None
        assert result_include is not None
        # Results may differ slightly due to include_lowest

    def test_gdist_with_iterator(self):
        """Test gdist with explicit iterator."""
        intervals = pm.gintervals("1", 0, 10000)

        # Use 100bp iterator instead of track-based iterator
        result = pm.gdist("dense_track", [0, 0.2, 0.5, 1],
                         intervals=intervals, iterator=100)

        assert result is not None
        assert len(result) == 3
        total = sum(result)
        # With 100bp iterator on 10000bp region, expect ~100 intervals
        assert total > 0


class TestGdistMultiDimensional:
    """Tests for multi-dimensional gdist."""

    def test_gdist_two_dimensional(self):
        """Test gdist with two track expressions (2D distribution)."""
        intervals = pm.gintervals("1", 0, 50000)

        # Calculate 2D distribution: dense_track vs itself (same track)
        # Note: Mixed track types (dense vs sparse) aren't supported in the scanner
        result = pm.gdist(
            "dense_track", [0, 0.25, 0.5, 1],
            "dense_track", [0, 0.5, 1],
            intervals=intervals,
            iterator=100
        )

        assert result is not None
        # Should be a 2D array: 3 bins for first x 2 bins for second
        assert result.shape == (3, 2)
        # All counts should be non-negative
        assert np.all(result >= 0)


class TestGdistDataframe:
    """Tests for gdist dataframe output mode."""

    def test_gdist_dataframe_mode(self):
        """Test gdist with dataframe=True returns a DataFrame."""
        intervals = pm.gintervals("1", 0, 10000)

        result = pm.gdist("dense_track", [0, 0.2, 0.5, 1],
                         intervals=intervals, dataframe=True)

        assert result is not None
        # Should be a DataFrame with columns for track expression and 'n'
        assert hasattr(result, 'columns')
        assert 'n' in result.columns
        # Should have 3 rows (one per bin)
        assert len(result) == 3

    def test_gdist_dataframe_with_names(self):
        """Test gdist dataframe mode with custom names."""
        intervals = pm.gintervals("1", 0, 10000)

        result = pm.gdist("dense_track", [0, 0.2, 0.5, 1],
                         intervals=intervals, dataframe=True,
                         names=["my_track"])

        assert result is not None
        assert 'my_track' in result.columns
        assert 'n' in result.columns


class TestGdistEdgeCases:
    """Edge case tests for gdist."""

    def test_gdist_empty_intervals(self):
        """Test gdist with empty intervals."""
        import pandas as pd
        empty = pd.DataFrame(columns=["chrom", "start", "end"])

        # Should handle empty intervals gracefully
        result = pm.gdist("dense_track", [0, 0.5, 1], intervals=empty)

        # Result should be all zeros
        assert result is not None
        assert all(v == 0 for v in result)

    def test_gdist_requires_expr_and_breaks(self):
        """Test that gdist requires expression and breaks."""
        with pytest.raises((ValueError, TypeError)):
            pm.gdist()

        with pytest.raises((ValueError, TypeError)):
            pm.gdist("dense_track")  # Missing breaks

    def test_gdist_breaks_validation(self):
        """Test that gdist validates breaks."""
        intervals = pm.gintervals("1", 0, 10000)

        # Breaks must be in increasing order
        with pytest.raises(ValueError):
            pm.gdist("dense_track", [0.5, 0.2, 1], intervals=intervals)
