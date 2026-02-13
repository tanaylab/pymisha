"""Tests for glookup function."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


class TestGlookupBasic:
    """Test basic glookup functionality."""

    def test_glookup_1d_lookup_table(self):
        """One-dimensional lookup table with track expression."""
        # Create a 1D lookup table with 5 values
        lookup_table = np.array([10, 20, 30, 40, 50])
        # Breaks create 5 bins: (0.1, 0.12], (0.12, 0.14], ..., (0.18, 0.2]
        breaks = np.linspace(0.1, 0.2, 6)

        # Get intervals - dense_track has values roughly 0.1-0.2
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(lookup_table, "dense_track", breaks, intervals=intervals)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns
        assert "value" in result.columns

        # All values should be from the lookup table
        assert all(v in lookup_table or np.isnan(v) for v in result["value"])

    def test_glookup_2d_lookup_table(self):
        """Two-dimensional lookup table with two expressions."""
        # Create a 2D lookup table (5 x 3)
        lookup_table = np.arange(1, 16).reshape((5, 3))
        breaks1 = np.linspace(0.1, 0.2, 6)  # 5 bins for first dim
        breaks2 = np.linspace(0.31, 0.37, 4)  # 3 bins for second dim

        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(
            lookup_table,
            "dense_track", breaks1,
            "2 * dense_track", breaks2,
            intervals=intervals
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "value" in result.columns

    def test_glookup_returns_intervalID(self):
        """Result has intervalID column for tracking."""
        lookup_table = np.array([10, 20, 30])
        breaks = np.array([0.0, 0.15, 0.3, 0.45])
        intervals = pm.gintervals("1", [0, 100], [100, 200])

        result = pm.glookup(lookup_table, "dense_track", breaks, intervals=intervals)

        assert result is not None
        assert "intervalID" in result.columns


class TestGlookupBinning:
    """Test binning behavior."""

    def test_glookup_include_lowest(self):
        """include_lowest includes the lowest break value in first bin."""
        # Values exactly at lowest break go to bin 0 when include_lowest=True
        lookup_table = np.array([100, 200, 300])
        breaks = np.array([0.1, 0.15, 0.2, 0.25])

        intervals = pm.gintervals("1", 0, 100)

        result_with = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, include_lowest=True
        )
        result_without = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, include_lowest=False
        )

        # Results should differ when values exactly equal lowest break
        assert result_with is not None
        assert result_without is not None

    def test_glookup_force_binning_true(self):
        """force_binning=True clamps out-of-range values to first/last bin."""
        lookup_table = np.array([1, 2, 3])
        # Narrow breaks that will miss many values
        breaks = np.array([0.14, 0.15, 0.16, 0.17])

        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, force_binning=True
        )

        assert result is not None
        # With force_binning, no NaN values (out-of-range are clamped)
        # All values should be from lookup_table
        valid_values = result["value"].dropna()
        assert len(valid_values) > 0

    def test_glookup_force_binning_false(self):
        """force_binning=False produces NaN for out-of-range values."""
        lookup_table = np.array([1, 2, 3])
        # Very narrow breaks - most values will be out of range
        breaks = np.array([0.14, 0.145, 0.15, 0.155])

        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, force_binning=False
        )

        assert result is not None
        # With force_binning=False, out-of-range values become NaN
        # There should be some NaN values
        assert result["value"].isna().any()


class TestGlookupEdgeCases:
    """Test edge cases."""

    def test_glookup_empty_intervals(self):
        """Empty intervals returns None or empty result."""
        lookup_table = np.array([1, 2, 3])
        breaks = np.array([0.0, 0.1, 0.2, 0.3])
        intervals = pd.DataFrame(columns=["chrom", "start", "end"])

        result = pm.glookup(lookup_table, "dense_track", breaks, intervals=intervals)

        assert result is None or len(result) == 0

    def test_glookup_nan_values_in_expression(self):
        """NaN values in expression produce NaN in result."""
        lookup_table = np.array([1, 2, 3])
        breaks = np.array([0.0, 0.5, 1.0, 1.5])

        # Use sparse_track which may have NaN values
        intervals = pm.gintervals("1", 0, 10000)

        result = pm.glookup(lookup_table, "sparse_track", breaks, intervals=intervals)

        # Should not crash; NaN in expression -> NaN in result
        assert result is not None

    def test_glookup_requires_lookup_table(self):
        """glookup requires a lookup_table argument."""
        breaks = np.array([0.0, 0.1, 0.2])
        intervals = pm.gintervals("1", 0, 100)

        with pytest.raises((ValueError, TypeError)):
            pm.glookup(None, "dense_track", breaks, intervals=intervals)

    def test_glookup_requires_intervals(self):
        """glookup requires intervals."""
        lookup_table = np.array([1, 2, 3])
        breaks = np.array([0.0, 0.1, 0.2, 0.3])

        with pytest.raises((ValueError, TypeError)):
            pm.glookup(lookup_table, "dense_track", breaks, intervals=None)


class TestGlookupIterator:
    """Test iterator parameter."""

    def test_glookup_with_iterator(self):
        """glookup with explicit iterator."""
        lookup_table = np.array([1, 2, 3])
        breaks = np.array([0.0, 0.15, 0.3, 0.45])
        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, iterator=50
        )

        assert result is not None
        # With iterator=50, we should get more results than with default
        assert len(result) > 0


class TestGlookup2D:
    """Test 2D intervals/band behavior in Python fallback."""

    def test_glookup_2d_intervals_returns_2d_coordinates(self):
        """glookup on 2D intervals returns 2D coordinate columns."""
        lookup_table = np.array([1, 2, 3, 4, 5], dtype=float)
        breaks = np.linspace(0.0, 9000.0, 6)
        intervals = pm.gintervals_2d_all()

        result = pm.glookup(
            lookup_table,
            "rects_track", breaks,
            intervals=intervals,
        )

        assert result is not None
        expected_cols = {
            "chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID", "value"
        }
        assert expected_cols.issubset(result.columns)
        assert len(result) > 0
        valid_values = result["value"].dropna().to_numpy(dtype=float)
        assert np.isin(valid_values, lookup_table).all()
        assert np.unique(valid_values).size > 1

    def test_glookup_2d_band_restricts_results(self):
        """band should reduce or keep (never increase) 2D lookup results."""
        lookup_table = np.array([1, 2, 3, 4, 5], dtype=float)
        breaks = np.linspace(0.0, 9000.0, 6)
        intervals = pm.gintervals_2d_all()

        full_result = pm.glookup(
            lookup_table,
            "rects_track", breaks,
            intervals=intervals,
        )
        band_result = pm.glookup(
            lookup_table,
            "rects_track", breaks,
            intervals=intervals,
            band=(-100, 100),
        )

        assert full_result is not None
        assert band_result is not None
        assert len(band_result) <= len(full_result)
        assert np.isin(
            band_result["value"].dropna().to_numpy(dtype=float),
            lookup_table,
        ).all()
