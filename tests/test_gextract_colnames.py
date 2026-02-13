"""Tests for gextract colnames parameter."""

import numpy as np
import pytest

import pymisha as pm


class TestGextractColnames:
    """Tests for the colnames parameter in gextract."""

    def test_single_expr_colnames(self):
        """Single expression with custom column name."""
        result = pm.gextract("dense_track", intervals=pm.gintervals_all(), iterator=100000, colnames=["my_col"])
        assert "my_col" in result.columns
        assert "dense_track" not in result.columns

    def test_multiple_expr_colnames(self):
        """Multiple expressions with custom column names."""
        result = pm.gextract(
            ["dense_track", "dense_track * 2"],
            intervals=pm.gintervals_all(),
            iterator=100000,
            colnames=["d_vals", "d_doubled"],
        )
        assert "d_vals" in result.columns
        assert "d_doubled" in result.columns
        assert "dense_track" not in result.columns

    def test_colnames_values_unchanged(self):
        """Column renaming should not alter values."""
        result_named = pm.gextract("dense_track", intervals=pm.gintervals_all(), iterator=100000, colnames=["my_col"])
        result_default = pm.gextract("dense_track", intervals=pm.gintervals_all(), iterator=100000)
        np.testing.assert_array_equal(result_named["my_col"].values, result_default["dense_track"].values)

    def test_colnames_wrong_length_raises(self):
        """colnames length must match number of expressions."""
        with pytest.raises(ValueError, match="colnames"):
            pm.gextract(
                ["dense_track", "sparse_track"],
                intervals=pm.gintervals_all(),
                iterator=100000,
                colnames=["only_one"],
            )

    def test_colnames_with_expression(self):
        """colnames work with computed expressions too."""
        result = pm.gextract(
            "dense_track * 2",
            intervals=pm.gintervals_all(),
            iterator=100000,
            colnames=["doubled"],
        )
        assert "doubled" in result.columns

    def test_colnames_none_uses_default(self):
        """colnames=None should use default behavior."""
        result = pm.gextract("dense_track", intervals=pm.gintervals_all(), iterator=100000, colnames=None)
        assert "dense_track" in result.columns
