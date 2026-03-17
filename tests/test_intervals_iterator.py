"""TDD tests for DataFrame intervals as the iterator parameter.

R misha supports passing a data.frame as iterator to functions like gextract,
gscreen, gsummary, etc. When a data.frame is passed as iterator:
1. The intervals are sorted and overlaps are unified
2. Each iterator interval that overlaps a scope interval produces a bin
3. The output interval = intersection(scope_interval, iterator_interval)
4. intervalID maps back to the original scope interval index (1-based)

These tests define the expected behaviour BEFORE the feature is implemented
(TDD red phase).
"""

import numpy as np
import pandas as pd

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intervals(chroms, starts, ends):
    """Create a plain DataFrame with chrom/start/end columns."""
    return pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})


# ===================================================================
# Core functionality — gextract with DataFrame iterator
# ===================================================================

class TestGextractIntervalsIteratorCore:
    """Core behaviour of passing a DataFrame as iterator to gextract."""

    def test_gextract_intervals_iterator_basic(self):
        """Pass a DataFrame with chrom/start/end as iterator; verify result
        has correct interval coordinates and non-null track values."""
        scope = pm.gintervals("1", 0, 1000)
        # Iterator: two 400-bp bins inside the scope
        itr = _make_intervals(["1", "1"], [0, 400], [400, 800])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 2
        assert list(result["chrom"]) == ["1", "1"]
        assert list(result["start"]) == [0, 400]
        assert list(result["end"]) == [400, 800]
        assert "dense_track" in result.columns
        # dense_track has data in this region — values should be non-NaN
        assert result["dense_track"].notna().all()

    def test_gextract_intervals_iterator_clips_to_scope(self):
        """Iterator intervals extending beyond scope boundaries should be
        clipped to the scope boundaries."""
        scope = pm.gintervals("1", 200, 800)
        # Iterator extends beyond scope on both sides
        itr = _make_intervals(["1"], [0], [1000])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 1
        # The output interval should be the intersection: [200, 800)
        assert result["start"].iloc[0] == 200
        assert result["end"].iloc[0] == 800

    def test_gextract_intervals_iterator_outside_scope_ignored(self):
        """Iterator intervals completely outside the scope produce no rows."""
        scope = pm.gintervals("1", 0, 1000)
        # Iterator on chrom 2 — no overlap with scope on chrom 1
        itr = _make_intervals(["2"], [0], [1000])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        # No overlap → no output rows
        assert result is None or len(result) == 0

    def test_gextract_intervals_iterator_preserves_interval_id(self):
        """intervalID should map back to the 1-based index in the *scope*
        intervals (not the iterator intervals)."""
        scope = _make_intervals(["1", "1"], [0, 2000], [1000, 3000])
        # Iterator bins that each overlap a different scope interval
        itr = _make_intervals(["1", "1"], [0, 2000], [400, 2400])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert "intervalID" in result.columns
        assert len(result) == 2
        # First bin overlaps scope interval 1 → intervalID == 1
        # Second bin overlaps scope interval 2 → intervalID == 2
        assert result["intervalID"].iloc[0] == 1
        assert result["intervalID"].iloc[1] == 2

    def test_gextract_intervals_iterator_multiple_scope_intervals(self):
        """Multiple scope intervals, each overlapping different iterator
        intervals, should produce the correct set of output rows."""
        scope = _make_intervals(["1", "1"], [0, 2000], [1000, 3000])
        # Three iterator bins: two in the first scope interval, one in the second
        itr = _make_intervals(
            ["1", "1", "1"],
            [0, 400, 2000],
            [400, 800, 2400],
        )

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 3
        # First two rows from scope interval 1
        assert list(result["intervalID"].iloc[:2]) == [1, 1]
        # Third row from scope interval 2
        assert result["intervalID"].iloc[2] == 2

    def test_gextract_intervals_iterator_sorted(self):
        """Unsorted iterator intervals should still produce correct results
        (sorted internally before intersection)."""
        scope = pm.gintervals("1", 0, 1000)
        # Give iterator intervals in reverse order
        itr = _make_intervals(["1", "1"], [400, 0], [800, 400])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 2
        # Output should be sorted by genomic position
        assert result["start"].iloc[0] < result["start"].iloc[1]

    def test_gextract_intervals_iterator_overlapping(self):
        """Overlapping iterator intervals should be unified before
        intersection with scope, producing non-overlapping output bins."""
        scope = pm.gintervals("1", 0, 1000)
        # Two overlapping iterator intervals: [0,600) and [400,1000)
        itr = _make_intervals(["1", "1"], [0, 400], [600, 1000])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        # After unification: single interval [0, 1000)
        assert len(result) == 1
        assert result["start"].iloc[0] == 0
        assert result["end"].iloc[0] == 1000

    def test_gextract_intervals_iterator_matches_fixed_bin(self):
        """Uniformly-spaced iterator intervals matching a fixed bin size
        should produce the same result as an integer iterator."""
        scope = pm.gintervals("1", 0, 1000)
        bin_size = 200
        # Create iterator intervals equivalent to iterator=200
        n_bins = 1000 // bin_size
        itr = _make_intervals(
            ["1"] * n_bins,
            [i * bin_size for i in range(n_bins)],
            [(i + 1) * bin_size for i in range(n_bins)],
        )

        result_df_itr = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        result_int_itr = pm.gextract("dense_track", scope, iterator=bin_size, progress=False)

        assert result_df_itr is not None
        assert result_int_itr is not None
        assert len(result_df_itr) == len(result_int_itr)
        np.testing.assert_array_equal(result_df_itr["start"].values, result_int_itr["start"].values)
        np.testing.assert_array_equal(result_df_itr["end"].values, result_int_itr["end"].values)
        np.testing.assert_array_equal(
            result_df_itr["dense_track"].values,
            result_int_itr["dense_track"].values,
        )
        np.testing.assert_array_equal(
            result_df_itr["intervalID"].values,
            result_int_itr["intervalID"].values,
        )


# ===================================================================
# Cross-function tests
# ===================================================================

class TestIntervalsIteratorCrossFunctions:
    """DataFrame iterator support in functions other than gextract."""

    def test_gscreen_intervals_iterator(self):
        """gscreen should accept a DataFrame iterator and return intervals
        where the expression is True within the iterator bins."""
        scope = pm.gintervals("1", 0, 2000)
        itr = _make_intervals(["1", "1"], [0, 1000], [1000, 2000])

        result = pm.gscreen("dense_track > 0", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) > 0
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns

    def test_gsummary_intervals_iterator(self):
        """gsummary with a DataFrame iterator should return valid summary
        statistics."""
        scope = pm.gintervals("1", 0, 2000)
        itr = _make_intervals(["1", "1"], [0, 1000], [1000, 2000])

        result = pm.gsummary("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        # gsummary returns a Series with standard summary stats
        assert "Total intervals" in result.index
        total = result["Total intervals"]
        assert total == 2  # two iterator bins

    def test_gquantiles_intervals_iterator(self):
        """gquantiles with a DataFrame iterator should return quantile
        values."""
        scope = pm.gintervals("1", 0, 2000)
        itr = _make_intervals(["1", "1"], [0, 1000], [1000, 2000])

        result = pm.gquantiles("dense_track", percentiles=[0.25, 0.5, 0.75],
                               intervals=scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 3
        # Quantiles should be sorted
        assert result.iloc[0] <= result.iloc[1] <= result.iloc[2]

    def test_gdist_intervals_iterator(self):
        """gdist with a DataFrame iterator should produce a valid
        distribution array."""
        scope = pm.gintervals("1", 0, 2000)
        itr = _make_intervals(["1", "1"], [0, 1000], [1000, 2000])

        result = pm.gdist("dense_track", [0, 0.5, 1.0],
                          intervals=scope, iterator=itr, progress=False)
        assert result is not None
        # gdist returns an ndarray (or DataFrame); counts should be non-negative
        arr = np.asarray(result)
        assert arr.shape[0] == 2  # two break bins: (0, 0.5], (0.5, 1.0]
        assert (arr >= 0).all()

    def test_giterator_intervals_with_df_iterator(self):
        """giterator_intervals with a DataFrame iterator should return the
        expected grid of intervals."""
        scope = pm.gintervals("1", 0, 2000)
        itr = _make_intervals(["1", "1", "1"], [0, 600, 1200], [600, 1200, 1800])

        result = pm.giterator_intervals(intervals=scope, iterator=itr)
        assert result is not None
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns
        assert "intervalID" in result.columns
        assert len(result) == 3
        assert list(result["start"]) == [0, 600, 1200]
        assert list(result["end"]) == [600, 1200, 1800]


# ===================================================================
# Edge cases
# ===================================================================

class TestIntervalsIteratorEdgeCases:
    """Edge-case behaviour for DataFrame-as-iterator."""

    def test_gextract_intervals_iterator_empty(self):
        """An empty DataFrame iterator should return None or an empty
        DataFrame."""
        scope = pm.gintervals("1", 0, 1000)
        itr = pd.DataFrame({
            "chrom": pd.Series([], dtype=str),
            "start": pd.Series([], dtype=int),
            "end": pd.Series([], dtype=int),
        })

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is None or len(result) == 0

    def test_gextract_intervals_iterator_single_interval(self):
        """A single-interval DataFrame iterator should produce exactly one
        output row (assuming it overlaps the scope)."""
        scope = pm.gintervals("1", 0, 1000)
        itr = _make_intervals(["1"], [200], [600])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 1
        assert result["start"].iloc[0] == 200
        assert result["end"].iloc[0] == 600
        assert result["intervalID"].iloc[0] == 1

    def test_gextract_intervals_iterator_multi_chrom(self):
        """Iterator intervals spanning multiple chromosomes should each be
        intersected with the appropriate scope intervals."""
        scope = _make_intervals(["1", "2"], [0, 0], [1000, 1000])
        itr = _make_intervals(["1", "2"], [0, 0], [400, 400])

        result = pm.gextract("dense_track", scope, iterator=itr, progress=False)
        assert result is not None
        assert len(result) == 2
        chroms = list(result["chrom"])
        assert "1" in chroms
        assert "2" in chroms
        # intervalID should reference the correct scope intervals
        row_chr1 = result[result["chrom"] == "1"].iloc[0]
        row_chr2 = result[result["chrom"] == "2"].iloc[0]
        assert row_chr1["intervalID"] == 1
        assert row_chr2["intervalID"] == 2


class TestIntervalsIteratorStringName:
    """Tests for using a saved interval set name as iterator."""

    def test_gextract_intervals_iterator_set_name(self):
        """Passing a saved interval set name as iterator works like a DataFrame."""
        scope = pm.gintervals("1", 0, 2000)
        itr_df = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 400],
            "end": [200, 600],
        })
        pm.gintervals_save(itr_df, "test_itr_set")
        try:
            result_df = pm.gextract("dense_track", scope, iterator=itr_df, progress=False)
            result_str = pm.gextract("dense_track", scope, iterator="test_itr_set", progress=False)
            assert result_str is not None
            assert len(result_str) == len(result_df)
            pd.testing.assert_frame_equal(result_str, result_df)
        finally:
            pm.gintervals_rm("test_itr_set", force=True)

    def test_gsummary_intervals_iterator_set_name(self):
        """gsummary with a saved interval set name as iterator."""
        scope = pm.gintervals("1", 0, 5000)
        itr_df = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [0, 400, 800],
            "end": [200, 600, 1000],
        })
        pm.gintervals_save(itr_df, "test_itr_set2")
        try:
            result_df = pm.gsummary("dense_track", scope, iterator=itr_df, progress=False)
            result_str = pm.gsummary("dense_track", scope, iterator="test_itr_set2", progress=False)
            pd.testing.assert_series_equal(result_str, result_df)
        finally:
            pm.gintervals_rm("test_itr_set2", force=True)

    def test_gscreen_intervals_iterator_set_name(self):
        """gscreen with a saved interval set name as iterator."""
        scope = pm.gintervals("1", 0, 5000)
        itr_df = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 400],
            "end": [200, 600],
        })
        pm.gintervals_save(itr_df, "test_itr_set3")
        try:
            result_df = pm.gscreen("dense_track > 0", scope, iterator=itr_df, progress=False)
            result_str = pm.gscreen("dense_track > 0", scope, iterator="test_itr_set3", progress=False)
            if result_df is not None:
                assert result_str is not None
                pd.testing.assert_frame_equal(result_str, result_df)
            else:
                assert result_str is None
        finally:
            pm.gintervals_rm("test_itr_set3", force=True)
