"""Edge case tests for gextract (1D extraction)."""

import numpy as np
import pandas as pd

import pymisha as pm


class TestGextractEmptyIntervals:
    """Tests for gextract with empty or degenerate interval inputs."""

    def test_empty_dataframe_returns_none(self):
        """Empty intervals DataFrame should return None."""
        empty_df = pd.DataFrame(
            {"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int)}
        )
        result = pm.gextract("dense_track", empty_df, iterator=100)
        assert result is None

    def test_empty_dataframe_sparse_returns_none(self):
        """Empty intervals DataFrame with sparse track should return None."""
        empty_df = pd.DataFrame(
            {"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int)}
        )
        result = pm.gextract("sparse_track", empty_df, iterator=100)
        assert result is None

    def test_empty_dataframe_multiple_exprs_returns_none(self):
        """Empty intervals with multiple expressions should return None."""
        empty_df = pd.DataFrame(
            {"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int)}
        )
        result = pm.gextract(["dense_track", "dense_track * 2"], empty_df, iterator=100)
        assert result is None

    def test_single_interval_returns_one_row(self):
        """A single small interval should produce exactly one row."""
        intervals = pm.gintervals("1", 0, 100)
        result = pm.gextract("dense_track", intervals, iterator=100)
        assert result is not None
        assert len(result) == 1
        assert "dense_track" in result.columns
        assert "intervalID" in result.columns


class TestGextractAllNaN:
    """Tests for expressions where all values are NaN."""

    def test_sparse_track_region_all_nan(self):
        """Sparse track over a region with no data should return all NaN values."""
        # chrom X far end: sparse_track has no data here
        intervals = pm.gintervals("X", 190000, 200000)
        result = pm.gextract("sparse_track", intervals, iterator=1000)
        assert result is not None
        assert len(result) == 10
        assert result["sparse_track"].isna().all()

    def test_sparse_track_partial_nan(self):
        """Sparse track with partial coverage should have some NaN and some non-NaN."""
        # chrom 2 near end: sparse track has partial data
        intervals = pm.gintervals("2", 290000, 300000)
        result = pm.gextract("sparse_track", intervals, iterator=1000)
        assert result is not None
        assert len(result) == 10
        has_nan = result["sparse_track"].isna().any()
        has_data = result["sparse_track"].notna().any()
        assert has_nan, "Expected some NaN values in sparse region"
        assert has_data, "Expected some non-NaN values in this region"

    def test_dense_track_nan_region(self):
        """Dense track in a region with NaN should return all NaN values."""
        # Dense track on chrom X far end: likely all NaN
        intervals = pm.gintervals("X", 190000, 200000)
        result = pm.gextract("dense_track", intervals, iterator=1000)
        assert result is not None
        assert len(result) == 10
        assert result["dense_track"].isna().all()

    def test_expression_on_all_nan_is_all_nan(self):
        """Arithmetic on all-NaN track values should produce all NaN."""
        intervals = pm.gintervals("X", 190000, 200000)
        result = pm.gextract("dense_track * 2 + 1", intervals, iterator=1000)
        assert result is not None
        assert len(result) == 10
        col = [c for c in result.columns if c not in ("chrom", "start", "end", "intervalID")][0]
        assert result[col].isna().all()

    def test_multiple_expressions_all_nan(self):
        """Multiple expressions over all-NaN region should all be NaN."""
        intervals = pm.gintervals("X", 190000, 200000)
        result = pm.gextract(["dense_track", "dense_track * 2"], intervals, iterator=1000)
        assert result is not None
        assert len(result) == 10
        assert result["dense_track"].isna().all()
        assert result["dense_track * 2"].isna().all()


class TestGextractMixedPhysicalAndVTrack:
    """Tests for gextract with mixed physical track and virtual track expressions."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_list_of_physical_and_vtrack(self):
        """List of [physical_track, vtrack] should produce correct columns."""
        pm.gvtrack_create("vt_edge_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 1000)
        result = pm.gextract(["dense_track", "vt_edge_avg"], intervals, iterator=200)
        assert result is not None
        assert "dense_track" in result.columns
        assert "vt_edge_avg" in result.columns
        assert len(result) == 5  # 1000/200 = 5 bins
        # Both columns should have non-NaN values in this region
        assert result["dense_track"].notna().all()
        assert result["vt_edge_avg"].notna().all()

    def test_combined_expression_physical_plus_vtrack(self):
        """Expression combining physical track + vtrack should evaluate correctly."""
        pm.gvtrack_create("vt_edge_max", "sparse_track", func="max")
        intervals = pm.gintervals("1", 0, 500)
        result = pm.gextract("dense_track + vt_edge_max", intervals, iterator=100)
        assert result is not None
        assert len(result) == 5
        col = [c for c in result.columns if c not in ("chrom", "start", "end", "intervalID")][0]
        # Some bins may be NaN (where sparse has no data), some should have values
        # The first few bins on chr1 should have sparse data
        assert result[col].notna().any(), "Expected at least some non-NaN combined values"

    def test_vtrack_with_colnames(self):
        """colnames should work with mixed physical+vtrack expressions."""
        pm.gvtrack_create("vt_edge_sum", "dense_track", func="sum")
        intervals = pm.gintervals("1", 0, 600)
        result = pm.gextract(
            ["dense_track", "vt_edge_sum"],
            intervals,
            iterator=200,
            colnames=["phys_val", "vt_val"],
        )
        assert result is not None
        assert "phys_val" in result.columns
        assert "vt_val" in result.columns
        assert "dense_track" not in result.columns
        assert "vt_edge_sum" not in result.columns

    def test_vtrack_all_nan_region(self):
        """VTrack on sparse_track in a region with no data should return NaN."""
        pm.gvtrack_create("vt_edge_sparse_avg", "sparse_track", func="avg")
        intervals = pm.gintervals("X", 190000, 200000)
        result = pm.gextract("vt_edge_sparse_avg", intervals, iterator=1000)
        assert result is not None
        assert len(result) == 10
        assert result["vt_edge_sparse_avg"].isna().all()

    def test_mixed_expr_nan_propagation(self):
        """NaN from vtrack should propagate through arithmetic expressions."""
        pm.gvtrack_create("vt_edge_sp_max", "sparse_track", func="max")
        # Region where sparse track has no data (chrom 1, bins [300,500))
        intervals = pm.gintervals("1", 300, 500)
        result = pm.gextract("dense_track + vt_edge_sp_max", intervals, iterator=100)
        assert result is not None
        col = [c for c in result.columns if c not in ("chrom", "start", "end", "intervalID")][0]
        # Where vt_edge_sp_max is NaN, the sum should also be NaN
        # even though dense_track has data in this region
        assert result[col].isna().any(), "NaN from vtrack should propagate to combined expression"

    def test_vtrack_values_match_separate_extraction(self):
        """Values from mixed extraction should match separate extractions."""
        pm.gvtrack_create("vt_edge_check", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 1000)

        # Extract together
        combined = pm.gextract(["dense_track", "vt_edge_check"], intervals, iterator=200)

        # Extract separately
        phys_only = pm.gextract("dense_track", intervals, iterator=200)
        vt_only = pm.gextract("vt_edge_check", intervals, iterator=200)

        assert combined is not None
        assert phys_only is not None
        assert vt_only is not None

        np.testing.assert_array_equal(
            combined["dense_track"].values,
            phys_only["dense_track"].values,
        )
        np.testing.assert_allclose(
            combined["vt_edge_check"].values,
            vt_only["vt_edge_check"].values,
            rtol=1e-10,
        )


class TestGextractSmallIterator:
    """Tests for gextract with very small or unusual iterator sizes."""

    def test_iterator_smaller_than_interval(self):
        """Iterator smaller than interval should split into multiple bins."""
        intervals = pm.gintervals("1", 0, 50)
        result = pm.gextract("dense_track", intervals, iterator=10)
        assert result is not None
        assert len(result) == 5  # 50/10 = 5 bins
        # All bins should have the same intervalID
        assert (result["intervalID"] == result["intervalID"].iloc[0]).all()

    def test_iterator_equals_interval(self):
        """Iterator equal to interval size should produce one row."""
        intervals = pm.gintervals("1", 0, 100)
        result = pm.gextract("dense_track", intervals, iterator=100)
        assert result is not None
        assert len(result) == 1

    def test_iterator_larger_than_interval(self):
        """Iterator larger than interval should still produce one row per interval."""
        intervals = pm.gintervals("1", 0, 50)
        result = pm.gextract("dense_track", intervals, iterator=1000)
        assert result is not None
        assert len(result) == 1
