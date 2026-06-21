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

    def test_long_names_not_truncated(self):
        """Issue #1: long column names must not be truncated to 40 chars.

        Two names sharing a long common prefix used to truncate to the same
        "<prefix>..." column and silently overwrite each other. Column names are
        now kept in full, so both columns survive with their full names. Covers
        both the C++ vtrack path and the pure-Python eval path.
        """
        # Two vtracks differing only in the final character, over *different*
        # sources so the values genuinely differ.
        base = "some_very_long.directory.path.that.exceeds.forty.characters.group_"
        names = [base + "1", base + "2"]
        for name, src in zip(names, ["dense_track", "sparse_track"], strict=True):
            pm.gvtrack_create(name, src, func="avg")
        try:
            intervals = pm.gintervals_all()
            # C++ vtrack path (clean aggregation vtracks, no user vars).
            cpp = pm.gextract(names, intervals=intervals, iterator=100000)
            assert list(cpp.columns) == ["chrom", "start", "end", names[0], names[1], "intervalID"]
            assert not np.array_equal(cpp[names[0]].values, cpp[names[1]].values)

            # Pure-Python eval path (user var forces evaluation off the C++ path).
            exprs = [f"k * np.nan_to_num({n})" for n in names]
            py = pm.gextract(exprs, intervals=intervals, iterator=100000, vars={"k": 1.0})
            assert list(py.columns) == ["chrom", "start", "end", exprs[0], exprs[1], "intervalID"]
            assert not np.array_equal(py[exprs[0]].values, py[exprs[1]].values)
        finally:
            for name in names:
                pm.gvtrack_rm(name)

    def test_identical_names_deduped(self):
        """The same expression listed twice keeps both columns (second gets '_')."""
        result = pm.gextract(
            ["dense_track", "dense_track"], intervals=pm.gintervals_all(), iterator=100000
        )
        assert list(result.columns) == ["chrom", "start", "end", "dense_track", "dense_track_", "intervalID"]
        np.testing.assert_array_equal(result["dense_track"].values, result["dense_track_"].values)
