"""
Tests for the LSE (log-sum-exp) virtual track function.

Ported from R misha: tests/testthat/test-vtrack-lse.R

LSE = log(sum(exp(x_i))), numerically stable: m + log(sum(exp(x_i - m)))
where m = max(x_i).
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lse(x):
    """Reference LSE implementation (numerically stable), ignoring NaN."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(logsumexp(x))


def _manual_lse_for_sparse(track_name, chrom, start, end):
    """Compute LSE for a sparse track by getting the per-interval values."""
    # For sparse tracks, we need to get one value per overlapping sparse
    # interval, not one per base. Use gextract with the track name to get
    # the track's value at each base, then get the unique contiguous runs.
    interval = pm.gintervals(chrom, start, end)
    raw = pm.gextract(track_name, interval, iterator=1)
    if raw is None or len(raw) == 0:
        return np.nan
    vals = raw[track_name].to_numpy(dtype=float)
    # Collect unique contiguous non-NaN runs (each sparse interval has one value)
    unique_vals = []
    prev_val = None
    for v in vals:
        if np.isnan(v):
            prev_val = None
            continue
        if prev_val is None or v != prev_val:
            unique_vals.append(v)
            prev_val = v
    if not unique_vals:
        return np.nan
    return _lse(unique_vals)


def _manual_lse(track_name, chrom, start, end):
    """Compute LSE manually from raw track values over an interval.

    For dense tracks: one value per bin.
    For sparse tracks: one value per overlapping sparse interval.
    """
    info = pm.gtrack_info(track_name)
    if info.get("type") == "sparse":
        return _manual_lse_for_sparse(track_name, chrom, start, end)

    # Dense track: use bin_size as iterator
    bin_size = info["bin_size"]
    interval = pm.gintervals(chrom, start, end)
    raw = pm.gextract(track_name, interval, iterator=bin_size)
    if raw is None or len(raw) == 0:
        return np.nan
    vals = raw[track_name].to_numpy(dtype=float)
    return _lse(vals)


def _manual_lse_vec(track_name, intervals_df):
    """Compute LSE for each row in an intervals DataFrame."""
    results = []
    for _, row in intervals_df.iterrows():
        results.append(_manual_lse(track_name, row["chrom"], row["start"], row["end"]))
    return np.array(results, dtype=float)


# ---------------------------------------------------------------------------
# TestLseBasic: basic LSE computation
# ---------------------------------------------------------------------------


class TestLseBasic:
    """Basic correctness tests for the LSE vtrack function."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_sparse_track(self):
        """lse vtrack function works on sparse track."""
        pm.gvtrack_create("vt_lse", "sparse_track", func="lse")
        pm.gvtrack_create("vt_sum", "sparse_track", func="sum")

        intervals = pm.gintervals_all()
        result_lse = pm.gextract("vt_lse", intervals=intervals, iterator=-1, colnames=["lse_value"])
        result_sum = pm.gextract("vt_sum", intervals=intervals, iterator=-1, colnames=["sum_value"])

        assert len(result_lse) == len(result_sum)
        # LSE is NaN iff sum is NaN
        np.testing.assert_array_equal(
            np.isnan(result_lse["lse_value"].to_numpy(dtype=float)),
            np.isnan(result_sum["sum_value"].to_numpy(dtype=float)),
        )

    def test_lse_known_values(self):
        """lse vtrack correctness with known values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [1.0, 2.0, 3.0],
        })

        pm.gvtrack_create("vt_lse_known", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_known", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_single_value_equals_identity(self):
        """lse of single value equals the value itself."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "score": [5.0],
        })

        pm.gvtrack_create("vt_lse_single", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 200)
        result = pm.gextract("vt_lse_single", intervals=iter_int, iterator=-1, colnames=["value"])

        np.testing.assert_allclose(result["value"].values[0], 5.0, atol=1e-6)

    def test_lse_returns_nan_for_no_data(self):
        """lse returns NaN for intervals with no data."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "score": [1.0],
        })

        pm.gvtrack_create("vt_lse_empty", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 500, 600)
        result = pm.gextract("vt_lse_empty", intervals=iter_int, iterator=-1, colnames=["value"])

        assert np.isnan(result["value"].values[0])

    def test_lse_with_negative_values(self):
        """lse with negative values (log-space)."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [-1.0, -2.0],
        })

        pm.gvtrack_create("vt_lse_neg", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract("vt_lse_neg", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(-1.0) + np.exp(-2.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestLseDenseTrack: LSE on dense track
# ---------------------------------------------------------------------------


class TestLseDenseTrack:
    """Tests for LSE on dense tracks."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_works_on_dense_track(self):
        """lse works on dense track."""
        pm.gvtrack_create("vt_lse_dense", src="dense_track", func="lse")

        intervals = pm.gintervals("1", 0, 1000)
        result = pm.gextract("vt_lse_dense", intervals=intervals, iterator=-1, colnames=["value"])

        raw = pm.gextract("dense_track", intervals=intervals, iterator=50)
        vals = raw["dense_track"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            expected = _lse(vals)
            np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-3)

    def test_lse_dense_matches_manual_across_multiple_intervals(self):
        """lse on dense track matches manual computation across multiple intervals."""
        pm.gvtrack_create("vt_lse_dense_multi", src="dense_track", func="lse")

        intervals = pd.concat([
            pm.gintervals("1", 0, 500),
            pm.gintervals("1", 500, 1000),
            pm.gintervals("1", 1000, 2000),
        ], ignore_index=True)

        result = pm.gextract("vt_lse_dense_multi", intervals=intervals, iterator=-1, colnames=["value"])
        manual = _manual_lse_vec("dense_track", intervals)

        np.testing.assert_allclose(result["value"].values, manual, rtol=1e-3)

    def test_lse_dense_with_numeric_iterator(self):
        """lse on dense track with numeric iterator."""
        pm.gvtrack_create("vt_lse_dense_iter", src="dense_track", func="lse")

        result = pm.gextract("vt_lse_dense_iter", pm.gintervals("1", 0, 500), iterator=100, colnames=["value"])

        # Each 100bp window should have its own LSE
        assert len(result) == 5

        # Verify first window manually
        raw = pm.gextract("dense_track", pm.gintervals("1", 0, 100), iterator=50)
        vals = raw["dense_track"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            np.testing.assert_allclose(result["value"].values[0], _lse(vals), rtol=1e-3)


# ---------------------------------------------------------------------------
# TestLseSparseTrack: LSE on sparse track
# ---------------------------------------------------------------------------


class TestLseSparseTrack:
    """Tests for LSE on sparse tracks with manual verification."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_sparse_matches_manual(self):
        """lse on sparse track matches manual computation."""
        pm.gvtrack_create("vt_lse_sparse_verify", src="sparse_track", func="lse")

        intervals = pd.concat([
            pm.gintervals("1", 0, 300),
            pm.gintervals("1", 600, 1000),
            pm.gintervals("1", 1200, 1500),
        ], ignore_index=True)

        result = pm.gextract("vt_lse_sparse_verify", intervals=intervals, iterator=-1, colnames=["value"])
        manual = _manual_lse_vec("sparse_track", intervals)

        result_vals = result["value"].to_numpy(dtype=float)
        # Both should have same NaN pattern
        np.testing.assert_array_equal(np.isnan(result_vals), np.isnan(manual))
        # Where both are non-NaN, values should match
        both_valid = ~np.isnan(result_vals) & ~np.isnan(manual)
        if np.any(both_valid):
            np.testing.assert_allclose(result_vals[both_valid], manual[both_valid], rtol=1e-3)

    def test_lse_sparse_with_large_iterator(self):
        """lse on sparse track with large iterator."""
        pm.gvtrack_create("vt_lse_sparse_large", src="sparse_track", func="lse")

        result = pm.gextract("vt_lse_sparse_large", pm.gintervals([1, 2]), iterator=10000, colnames=["value"])

        assert len(result) > 0
        vals = result["value"].to_numpy(dtype=float)
        assert np.any(~np.isnan(vals))

    def test_lse_sparse_full_verification(self):
        """lse on sparse track full verification with manual lse."""
        pm.gvtrack_create("vt_lse_full_sparse", src="sparse_track", func="lse")

        intervals = pd.concat([
            pm.gintervals("1", 0, 200),
            pm.gintervals("1", 200, 400),
            pm.gintervals("1", 400, 600),
            pm.gintervals("1", 600, 800),
            pm.gintervals("1", 800, 1000),
        ], ignore_index=True)

        result = pm.gextract("vt_lse_full_sparse", intervals, iterator=-1, colnames=["value"])
        manual = _manual_lse_vec("sparse_track", intervals)

        result_vals = result["value"].to_numpy(dtype=float)
        np.testing.assert_array_equal(np.isnan(result_vals), np.isnan(manual))
        both_valid = ~np.isnan(result_vals) & ~np.isnan(manual)
        if np.any(both_valid):
            np.testing.assert_allclose(result_vals[both_valid], manual[both_valid], rtol=1e-3)


# ---------------------------------------------------------------------------
# TestLseIteratorShifts: LSE with iterator shifts
# ---------------------------------------------------------------------------


class TestLseIteratorShifts:
    """Tests for LSE with iterator shifts."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_with_sshift_and_eshift(self):
        """lse with iterator shifts works."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [1.0, 2.0, 3.0],
        })

        pm.gvtrack_create("vt_lse_shift", src=intervals_df, func="lse")
        pm.gvtrack_iterator("vt_lse_shift", sshift=-100, eshift=100)

        # Iterator interval [200, 300] with shifts becomes [100, 400], covering all 3 values
        iter_int = pm.gintervals("1", 200, 300)
        result = pm.gextract("vt_lse_shift", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_with_sshift_only(self):
        """lse with sshift only narrows covered data."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [1.0, 2.0, 3.0],
        })

        pm.gvtrack_create("vt_lse_sshift", src=intervals_df, func="lse")
        pm.gvtrack_iterator("vt_lse_sshift", sshift=100)

        # Iterator [100, 400] with sshift=100 becomes [200, 400], covering values 2.0 and 3.0
        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_sshift", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(2.0) + np.exp(3.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_with_eshift_only(self):
        """lse with eshift only extends covered data."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [1.0, 2.0, 3.0],
        })

        pm.gvtrack_create("vt_lse_eshift", src=intervals_df, func="lse")
        pm.gvtrack_iterator("vt_lse_eshift", eshift=200)

        # Iterator [100, 200] with eshift=200 becomes [100, 400], covering all 3
        iter_int = pm.gintervals("1", 100, 200)
        result = pm.gextract("vt_lse_eshift", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_with_shifts_on_dense_track(self):
        """lse with shifts on dense track."""
        pm.gvtrack_create("vt_lse_dense_shift", src="dense_track", func="lse")
        pm.gvtrack_iterator("vt_lse_dense_shift", sshift=-50, eshift=50)

        iter_int = pm.gintervals("1", 200, 300)
        result = pm.gextract("vt_lse_dense_shift", intervals=iter_int, iterator=-1, colnames=["value"])

        # Shifted interval is [150, 350]
        raw = pm.gextract("dense_track", pm.gintervals("1", 150, 350), iterator=50)
        vals = raw["dense_track"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            np.testing.assert_allclose(result["value"].values[0], _lse(vals), rtol=1e-3)


# ---------------------------------------------------------------------------
# TestLseValueBased: value-based vtracks with LSE
# ---------------------------------------------------------------------------


class TestLseValueBased:
    """Tests for LSE on value-based (DataFrame source) vtracks."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_value_based(self):
        """lse on value-based track works."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 100, 200],
            "end": [100, 200, 300],
            "score": [0.5, 1.5, 2.5],
        })

        pm.gvtrack_create("vt_lse_val", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 0, 300)
        result = pm.gextract("vt_lse_val", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(0.5) + np.exp(1.5) + np.exp(2.5))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_value_based_multi_chromosomes(self):
        """lse on value-based track across multiple chromosomes."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr2", "chr2"],
            "start": [100, 300, 100, 300],
            "end": [200, 400, 200, 400],
            "score": [1.0, 2.0, 3.0, 4.0],
        })

        pm.gvtrack_create("vt_lse_mc", src=intervals_df, func="lse")

        # chr1: lse(1, 2)
        iter_chr1 = pm.gintervals("1", 100, 400)
        res1 = pm.gextract("vt_lse_mc", intervals=iter_chr1, iterator=-1, colnames=["value"])
        np.testing.assert_allclose(res1["value"].values[0], np.log(np.exp(1) + np.exp(2)), rtol=1e-5)

        # chr2: lse(3, 4)
        iter_chr2 = pm.gintervals("2", 100, 400)
        res2 = pm.gextract("vt_lse_mc", intervals=iter_chr2, iterator=-1, colnames=["value"])
        np.testing.assert_allclose(res2["value"].values[0], np.log(np.exp(3) + np.exp(4)), rtol=1e-5)

    def test_lse_value_based_partial_overlap(self):
        """lse on value-based track with partial overlap."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 300, 500],
            "end": [200, 400, 600],
            "score": [1.0, 2.0, 3.0],
        })

        pm.gvtrack_create("vt_lse_partial", src=intervals_df, func="lse")

        # Query only overlaps first two data intervals
        iter_int = pm.gintervals("1", 150, 350)
        result = pm.gextract("vt_lse_partial", intervals=iter_int, iterator=-1, colnames=["value"])
        np.testing.assert_allclose(result["value"].values[0], np.log(np.exp(1.0) + np.exp(2.0)), rtol=1e-5)

    def test_lse_value_based_handles_na(self):
        """lse on value-based track handles NaN values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [1.0, np.nan, 3.0],
        })

        pm.gvtrack_create("vt_lse_na", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_na", intervals=iter_int, iterator=-1, colnames=["value"])

        # NaN should be skipped; lse(1, 3)
        expected = np.log(np.exp(1.0) + np.exp(3.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_value_based_with_iterator_windows(self):
        """lse on value-based track with iterator windows."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"] * 5,
            "start": [0, 100, 200, 300, 400],
            "end": [100, 200, 300, 400, 500],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        pm.gvtrack_create("vt_lse_iter_window", src=intervals_df, func="lse")

        # 200bp windows
        result = pm.gextract("vt_lse_iter_window", pm.gintervals("1", 0, 400), iterator=200, colnames=["value"])

        vals = result["value"].values
        # Window [0,200): lse(1, 2)
        np.testing.assert_allclose(vals[0], np.log(np.exp(1) + np.exp(2)), rtol=1e-5)
        # Window [200,400): lse(3, 4)
        np.testing.assert_allclose(vals[1], np.log(np.exp(3) + np.exp(4)), rtol=1e-5)


# ---------------------------------------------------------------------------
# TestLseNumericalStability: numerical stability
# ---------------------------------------------------------------------------


class TestLseNumericalStability:
    """Tests for numerical stability of LSE computation."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_large_values(self):
        """lse numerical stability with large values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [100.0, 101.0],
        })

        pm.gvtrack_create("vt_lse_large", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract("vt_lse_large", intervals=iter_int, iterator=-1, colnames=["value"])

        # 101 + log(1 + exp(-1))
        expected = 101.0 + np.log(1 + np.exp(-1.0))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-4)

    def test_very_large_values(self):
        """lse numerical stability with very large values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [80.0, 85.0, 80.0],
        })

        pm.gvtrack_create("vt_lse_vlarge", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_vlarge", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = _lse([80, 85, 80])
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-3)

    def test_very_negative_values(self):
        """lse numerical stability with very negative values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [-80.0, -85.0, -80.0],
        })

        pm.gvtrack_create("vt_lse_vneg", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_vneg", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = _lse([-80, -85, -80])
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-3)

    def test_identical_values(self):
        """lse with identical values: lse(x, x, x) = x + log(3)."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [5.0, 5.0, 5.0],
        })

        pm.gvtrack_create("vt_lse_identical", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_identical", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = 5.0 + np.log(3)
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_zero_values(self):
        """lse with zero values: lse(0, 0, 0) = log(3)."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [0.0, 0.0, 0.0],
        })

        pm.gvtrack_create("vt_lse_zero", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_zero", intervals=iter_int, iterator=-1, colnames=["value"])

        np.testing.assert_allclose(result["value"].values[0], np.log(3), rtol=1e-5)

    def test_mixed_positive_negative(self):
        """lse with mixed positive and negative values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": [-2.0, 0.0, 2.0],
        })

        pm.gvtrack_create("vt_lse_mixed", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_mixed", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(-2) + np.exp(0) + np.exp(2))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_wide_value_range(self):
        """lse with wide value range: lse(-50, 50) ~ 50."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [-50.0, 50.0],
        })

        pm.gvtrack_create("vt_lse_wide", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract("vt_lse_wide", intervals=iter_int, iterator=-1, colnames=["value"])

        # When difference is huge, lse ~ max(values)
        np.testing.assert_allclose(result["value"].values[0], 50.0, atol=1e-4)


# ---------------------------------------------------------------------------
# TestLseProperties: mathematical properties of LSE
# ---------------------------------------------------------------------------


class TestLseProperties:
    """Tests for mathematical properties of LSE."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_geq_max(self):
        """lse is always >= max of input values."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"] * 4,
            "start": [100, 200, 300, 400],
            "end": [200, 300, 400, 500],
            "score": [-3.0, 1.0, -0.5, 2.0],
        })

        pm.gvtrack_create("vt_lse_gemax", src=intervals_df, func="lse")
        pm.gvtrack_create("vt_max_gemax", src=intervals_df, func="max")

        iter_int = pm.gintervals("1", 100, 500)
        lse_result = pm.gextract("vt_lse_gemax", intervals=iter_int, iterator=-1, colnames=["value"])
        max_result = pm.gextract("vt_max_gemax", intervals=iter_int, iterator=-1, colnames=["value"])

        assert lse_result["value"].values[0] >= max_result["value"].values[0]

    def test_lse_geq_max_dense_track(self):
        """lse >= max holds over multiple intervals on dense track."""
        pm.gvtrack_create("vt_lse_prop", src="dense_track", func="lse")
        pm.gvtrack_create("vt_max_prop", src="dense_track", func="max")

        intervals = pd.concat([
            pm.gintervals("1", 0, 500),
            pm.gintervals("1", 500, 1000),
            pm.gintervals("1", 1000, 2000),
        ], ignore_index=True)

        lse_res = pm.gextract("vt_lse_prop", intervals=intervals, iterator=-1, colnames=["value"])
        max_res = pm.gextract("vt_max_prop", intervals=intervals, iterator=-1, colnames=["value"])

        lse_vals = lse_res["value"].to_numpy(dtype=float)
        max_vals = max_res["value"].to_numpy(dtype=float)
        both_valid = ~np.isnan(lse_vals) & ~np.isnan(max_vals)
        if np.any(both_valid):
            assert np.all(lse_vals[both_valid] >= max_vals[both_valid])

    def test_lse_two_values_identity(self):
        """lse(a, b) = max(a,b) + log(1 + exp(-|a-b|))."""
        a, b = 3.0, 7.0
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [a, b],
        })

        pm.gvtrack_create("vt_lse_identity", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract("vt_lse_identity", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = max(a, b) + np.log(1 + np.exp(-abs(a - b)))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_n_equal_values(self):
        """lse of n equal values equals value + log(n)."""
        n = 10
        val = 4.0
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"] * n,
            "start": list(range(0, n * 100, 100)),
            "end": list(range(100, n * 100 + 100, 100)),
            "score": [val] * n,
        })

        pm.gvtrack_create("vt_lse_nequal", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 0, n * 100)
        result = pm.gextract("vt_lse_nequal", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = val + np.log(n)
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-4)

    def test_lse_formula_identity(self):
        """lse equals max + log(1 + sum(exp(xi - max))) for known values."""
        vals = [1.0, 2.0, 3.0]
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"] * 3,
            "start": [100, 200, 300],
            "end": [200, 300, 400],
            "score": vals,
        })

        pm.gvtrack_create("vt_lse_formula", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 400)
        result = pm.gextract("vt_lse_formula", intervals=iter_int, iterator=-1, colnames=["value"])

        m = max(vals)
        expected = m + np.log(sum(np.exp(v - m) for v in vals))
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_monotonically_nondecreasing(self):
        """lse is monotonically non-decreasing as more values are added."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"] * 4,
            "start": [100, 200, 300, 400],
            "end": [200, 300, 400, 500],
            "score": [1.0, 2.0, 3.0, 4.0],
        })

        pm.gvtrack_create("vt_lse_mono", src=intervals_df, func="lse")

        # Increasing window sizes
        lse_values = []
        for end_100 in range(2, 6):
            iter_int = pm.gintervals("1", 100, end_100 * 100)
            res = pm.gextract("vt_lse_mono", intervals=iter_int, iterator=-1, colnames=["value"])
            lse_values.append(res["value"].values[0])

        # Each value should be >= the previous one
        for i in range(1, len(lse_values)):
            assert lse_values[i] >= lse_values[i - 1], (
                f"LSE not monotonic: {lse_values[i]} < {lse_values[i - 1]}"
            )


# ---------------------------------------------------------------------------
# TestLseTrackExpressions: LSE in track expressions
# ---------------------------------------------------------------------------


class TestLseTrackExpressions:
    """Tests for LSE used in track expressions."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_in_expression(self):
        """lse works in track expressions."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [1.0, 2.0],
        })

        pm.gvtrack_create("vt_lse_expr", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract("vt_lse_expr * 2", intervals=iter_int, iterator=-1, colnames=["value"])

        expected = np.log(np.exp(1) + np.exp(2)) * 2
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)

    def test_lse_combined_with_other_vtracks(self):
        """lse combined with other vtracks in expression."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [1.0, 2.0],
        })

        pm.gvtrack_create("vt_lse_combo", src=intervals_df, func="lse")
        pm.gvtrack_create("vt_sum_combo", src=intervals_df, func="sum")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract("vt_lse_combo - vt_sum_combo", intervals=iter_int, iterator=-1, colnames=["value"])

        lse_val = np.log(np.exp(1) + np.exp(2))
        sum_val = 3.0
        np.testing.assert_allclose(result["value"].values[0], lse_val - sum_val, rtol=1e-5)

    def test_lse_in_conditional_expression(self):
        """lse in conditional expression (using Python ternary via np.where)."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [1.0, 2.0],
        })

        pm.gvtrack_create("vt_lse_cond", src=intervals_df, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        # Use arithmetic: multiply boolean by 1 to get 1 or 0
        result = pm.gextract("(vt_lse_cond > 2) * 1", intervals=iter_int, iterator=-1, colnames=["value"])

        lse_val = np.log(np.exp(1) + np.exp(2))
        expected = 1.0 if lse_val > 2 else 0.0
        np.testing.assert_allclose(result["value"].values[0], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestLseIntegration: gscreen / gsummary integration
# ---------------------------------------------------------------------------


class TestLseIntegration:
    """Tests for LSE integration with gscreen and gsummary."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_with_gscreen(self):
        """lse vtrack works with gscreen."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"] * 5,
            "start": [100, 200, 300, 500, 600],
            "end": [200, 300, 400, 600, 700],
            "score": [1.0, 2.0, 3.0, 0.1, 0.2],
        })

        pm.gvtrack_create("vt_lse_screen", src=intervals_df, func="lse")

        # Screen for intervals where lse > 2
        screened = pm.gscreen("vt_lse_screen > 2", pm.gintervals("1", 0, 800), iterator=200)
        assert len(screened) > 0

        # Verify: extract over screened intervals
        gextract_res = pm.gextract("vt_lse_screen", screened, iterator=-1, colnames=["value"])
        assert np.all(gextract_res["value"].values > 2)

    def test_lse_with_gsummary(self):
        """lse vtrack works with gsummary."""
        pm.gvtrack_create("vt_lse_summary", src="dense_track", func="lse")

        summary_result = pm.gsummary("vt_lse_summary", pm.gintervals([1, 2]), iterator=5000)

        assert summary_result is not None
        # gsummary returns a pandas Series with keys like 'Mean', 'Sum', etc.
        import pandas as pd
        if isinstance(summary_result, pd.Series):
            keys_lower = {k.lower() for k in summary_result.index}
            assert "mean" in keys_lower
        elif isinstance(summary_result, dict):
            keys_lower = {k.lower() for k in summary_result}
            assert "mean" in keys_lower
        else:
            assert np.isfinite(float(summary_result)) or np.isnan(float(summary_result))


# ---------------------------------------------------------------------------
# TestLseNaPattern: NA pattern consistency
# ---------------------------------------------------------------------------


class TestLseNaPattern:
    """Tests for NaN pattern consistency between LSE and other functions."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_na_pattern_matches_sum_dense(self):
        """lse NaN pattern matches sum on dense track."""
        pm.gvtrack_create("vt_lse_na_dense", src="dense_track", func="lse")
        pm.gvtrack_create("vt_sum_na_dense", src="dense_track", func="sum")

        intervals = pm.gintervals([1, 2])
        lse_res = pm.gextract("vt_lse_na_dense", intervals, iterator=2000, colnames=["value"])
        sum_res = pm.gextract("vt_sum_na_dense", intervals, iterator=2000, colnames=["value"])

        np.testing.assert_array_equal(
            np.isnan(lse_res["value"].to_numpy(dtype=float)),
            np.isnan(sum_res["value"].to_numpy(dtype=float)),
        )

    def test_na_pattern_matches_sum_sparse(self):
        """lse NaN pattern matches sum on sparse track with intervals iterator."""
        pm.gvtrack_create("vt_lse_na_sparse", src="sparse_track", func="lse")
        pm.gvtrack_create("vt_sum_na_sparse", src="sparse_track", func="sum")

        intervals = pd.concat([
            pm.gintervals("1", 0, 500),
            pm.gintervals("1", 500, 1000),
            pm.gintervals("1", 5000, 6000),
            pm.gintervals("1", 10000, 11000),
        ], ignore_index=True)

        lse_res = pm.gextract("vt_lse_na_sparse", intervals, iterator=-1, colnames=["value"])
        sum_res = pm.gextract("vt_sum_na_sparse", intervals, iterator=-1, colnames=["value"])

        np.testing.assert_array_equal(
            np.isnan(lse_res["value"].to_numpy(dtype=float)),
            np.isnan(sum_res["value"].to_numpy(dtype=float)),
        )


# ---------------------------------------------------------------------------
# TestLseFilters: LSE with vtrack filters
# ---------------------------------------------------------------------------


class TestLseFilters:
    """Tests for LSE with vtrack filters."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_filter_sparse(self):
        """lse respects vtrack filter on sparse track."""
        pm.gvtrack_create("vt_lse_filter_s", src="sparse_track", func="lse")
        pm.gvtrack_create("vt_lse_nofilt_s", src="sparse_track", func="lse")

        # Restrictive filter on chr1
        pm.gvtrack_filter("vt_lse_filter_s", mask=pm.gintervals("1", 100, 200))

        iter_int = pm.gintervals("1", 0, 500)

        filt_res = pm.gextract("vt_lse_filter_s", iter_int, iterator=-1, colnames=["value"])
        nofilt_res = pm.gextract("vt_lse_nofilt_s", iter_int, iterator=-1, colnames=["value"])

        filt_val = filt_res["value"].values[0]
        nofilt_val = nofilt_res["value"].values[0]

        # Filtered has fewer (or equal) values, so LSE should be <= unfiltered
        if not np.isnan(filt_val) and not np.isnan(nofilt_val):
            assert filt_val <= nofilt_val + 1e-10

    def test_lse_filter_dense(self):
        """lse respects vtrack filter on dense track."""
        pm.gvtrack_create("vt_lse_filter_dense", src="dense_track", func="lse")
        pm.gvtrack_create("vt_lse_nofilt_dense", src="dense_track", func="lse")

        # Restrictive filter
        pm.gvtrack_filter("vt_lse_filter_dense", mask=pm.gintervals("1", 100, 200))

        iter_int = pm.gintervals("1", 0, 500)

        filt_res = pm.gextract("vt_lse_filter_dense", iter_int, iterator=-1, colnames=["value"])
        nofilt_res = pm.gextract("vt_lse_nofilt_dense", iter_int, iterator=-1, colnames=["value"])

        filt_val = filt_res["value"].values[0]
        nofilt_val = nofilt_res["value"].values[0]

        # Filtered should include fewer values, so LSE should be <= unfiltered
        if not np.isnan(filt_val) and not np.isnan(nofilt_val):
            assert filt_val <= nofilt_val + 1e-10


# ---------------------------------------------------------------------------
# TestLseEdgeCases: edge cases
# ---------------------------------------------------------------------------


class TestLseEdgeCases:
    """Edge case tests for LSE."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_single_bin_dense(self):
        """lse with single bin on dense track: lse(x) = x."""
        pm.gvtrack_create("vt_lse_single_bin", src="dense_track", func="lse")

        # Use the track's own bin size as iterator to get single-bin results
        result = pm.gextract("vt_lse_single_bin", pm.gintervals("1", 0, 100), iterator=50, colnames=["value"])

        # For single bins, lse(x) = x
        raw = pm.gextract("dense_track", pm.gintervals("1", 0, 100), iterator=50)
        result_vals = result["value"].to_numpy(dtype=float)
        raw_vals = raw["dense_track"].to_numpy(dtype=float)
        both_valid = ~np.isnan(result_vals) & ~np.isnan(raw_vals)
        if np.any(both_valid):
            np.testing.assert_allclose(result_vals[both_valid], raw_vals[both_valid], atol=1e-6)

    def test_empty_query_returns_nan(self):
        """lse returns NaN for empty value-based track query."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "score": [1.0],
        })

        pm.gvtrack_create("vt_lse_empty_query", src=intervals_df, func="lse")

        # Query a different chromosome
        iter_int = pm.gintervals("2", 100, 200)
        result = pm.gextract("vt_lse_empty_query", intervals=iter_int, iterator=-1, colnames=["value"])

        assert np.isnan(result["value"].values[0])

    def test_many_na_intervals(self):
        """lse on many consecutive NaN intervals returns NaN for each."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
            "score": [1.0],
        })

        pm.gvtrack_create("vt_lse_multi_na", src=intervals_df, func="lse")

        # Multiple query intervals, all outside data range
        query_ints = pd.concat([
            pm.gintervals("1", 500, 600),
            pm.gintervals("1", 600, 700),
            pm.gintervals("1", 700, 800),
        ], ignore_index=True)

        result = pm.gextract("vt_lse_multi_na", intervals=query_ints, iterator=-1, colnames=["value"])

        assert len(result) == 3
        assert np.all(np.isnan(result["value"].to_numpy(dtype=float)))

    def test_alternating_data_and_nodata(self):
        """lse alternating data and no-data intervals."""
        intervals_df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 500],
            "end": [200, 600],
            "score": [2.0, 3.0],
        })

        pm.gvtrack_create("vt_lse_alt", src=intervals_df, func="lse")

        query_ints = pd.concat([
            pm.gintervals("1", 100, 200),  # has data
            pm.gintervals("1", 300, 400),  # no data
            pm.gintervals("1", 500, 600),  # has data
            pm.gintervals("1", 700, 800),  # no data
        ], ignore_index=True)

        result = pm.gextract("vt_lse_alt", intervals=query_ints, iterator=-1, colnames=["value"])

        vals = result["value"].to_numpy(dtype=float)
        np.testing.assert_allclose(vals[0], 2.0, atol=1e-6)  # lse(2) = 2
        assert np.isnan(vals[1])
        np.testing.assert_allclose(vals[2], 3.0, atol=1e-6)  # lse(3) = 3
        assert np.isnan(vals[3])


# ---------------------------------------------------------------------------
# TestLseMultipleVtracks: multiple vtracks simultaneously
# ---------------------------------------------------------------------------


class TestLseMultipleVtracks:
    """Tests for using multiple LSE vtracks simultaneously."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_multiple_lse_vtracks(self):
        """multiple lse vtracks can be used simultaneously."""
        df1 = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [1.0, 2.0],
        })
        df2 = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [3.0, 4.0],
        })

        pm.gvtrack_create("vt_lse_multi1", src=df1, func="lse")
        pm.gvtrack_create("vt_lse_multi2", src=df2, func="lse")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract(["vt_lse_multi1", "vt_lse_multi2"], intervals=iter_int, iterator=-1)

        np.testing.assert_allclose(result["vt_lse_multi1"].values[0], np.log(np.exp(1) + np.exp(2)), rtol=1e-5)
        np.testing.assert_allclose(result["vt_lse_multi2"].values[0], np.log(np.exp(3) + np.exp(4)), rtol=1e-5)

    def test_lse_and_other_functions_together(self):
        """lse and other functions work together."""
        df = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300],
            "score": [1.0, 2.0],
        })

        pm.gvtrack_create("vt_lse_together", src=df, func="lse")
        pm.gvtrack_create("vt_sum_together", src=df, func="sum")
        pm.gvtrack_create("vt_avg_together", src=df, func="avg")
        pm.gvtrack_create("vt_max_together", src=df, func="max")

        iter_int = pm.gintervals("1", 100, 300)
        result = pm.gextract(
            ["vt_lse_together", "vt_sum_together", "vt_avg_together", "vt_max_together"],
            intervals=iter_int, iterator=-1,
        )

        np.testing.assert_allclose(result["vt_lse_together"].values[0], np.log(np.exp(1) + np.exp(2)), rtol=1e-5)
        np.testing.assert_allclose(result["vt_sum_together"].values[0], 3.0, rtol=1e-5)
        np.testing.assert_allclose(result["vt_avg_together"].values[0], 1.5, rtol=1e-5)
        np.testing.assert_allclose(result["vt_max_together"].values[0], 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestLseDenseMultiChrom: dense track, multi-chromosome
# ---------------------------------------------------------------------------


class TestLseDenseMultiChrom:
    """Tests for LSE on dense track across multiple chromosomes."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_lse_dense_multi_chrom(self):
        """lse on dense track across multiple chromosomes."""
        pm.gvtrack_create("vt_lse_dense_mc", src="dense_track", func="lse")

        intervals = pd.concat([
            pm.gintervals("1", 0, 500),
            pm.gintervals("2", 0, 500),
        ], ignore_index=True)

        result = pm.gextract("vt_lse_dense_mc", intervals, iterator=-1, colnames=["value"])

        assert len(result) == 2
        # Verify per-chromosome against manual computation
        for i in range(len(result)):
            val = result["value"].values[i]
            if not np.isnan(val):
                manual = _manual_lse(
                    "dense_track",
                    str(intervals["chrom"].values[i]),
                    int(intervals["start"].values[i]),
                    int(intervals["end"].values[i]),
                )
                np.testing.assert_allclose(val, manual, rtol=1e-3)


# ---------------------------------------------------------------------------
# TestLseSlidingWindow: sliding window tests
# ---------------------------------------------------------------------------


class TestLseSlidingWindow:
    """Tests for sliding window LSE correctness."""

    def setup_method(self):
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_sliding_window_lse_dense_matches_manual(self):
        """sliding window LSE on dense track with small iterator matches manual."""
        pm.gvtrack_create("vt_lse_slide", src="dense_track", func="lse")
        pm.gvtrack_iterator("vt_lse_slide", sshift=-90, eshift=90)

        region = pm.gintervals("1", 100, 500)
        result = pm.gextract("vt_lse_slide", region, iterator=50, colnames=["value"])

        # Compute expected by extracting each window individually
        expected = []
        for _, row in result.iterrows():
            w_start = max(0, int(row["start"]) - 90)
            w_end = int(row["end"]) + 90
            raw = pm.gextract("dense_track", pm.gintervals("1", w_start, w_end), iterator=50)
            expected.append(_lse(raw["dense_track"].to_numpy(dtype=float)))
        expected = np.array(expected, dtype=float)

        result_vals = result["value"].to_numpy(dtype=float)
        both_valid = ~np.isnan(result_vals) & ~np.isnan(expected)
        if np.any(both_valid):
            np.testing.assert_allclose(result_vals[both_valid], expected[both_valid], rtol=1e-3)

    def test_sliding_window_nan_heavy_track(self):
        """sliding window LSE handles NaN-heavy tracks correctly."""
        src_df = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [0, 100, 400, 500],
            "end": [20, 120, 420, 520],
            "score": [1.0, 2.0, 3.0, 4.0],
        })

        pm.gvtrack_create("vt_lse_nan_slide", src=src_df, func="lse")

        # Sliding 200bp windows with 20bp step
        sliding_starts = list(range(0, 401, 20))
        sliding_ints = pd.DataFrame({
            "chrom": ["1"] * len(sliding_starts),
            "start": sliding_starts,
            "end": [s + 200 for s in sliding_starts],
        })

        result = pm.gextract("vt_lse_nan_slide", sliding_ints, iterator=-1, colnames=["value"])

        # Manually compute expected
        expected = []
        for _, row in sliding_ints.iterrows():
            w_start = row["start"]
            w_end = row["end"]
            overlap = (src_df["start"] < w_end) & (src_df["end"] > w_start)
            if overlap.any():
                expected.append(_lse(src_df.loc[overlap, "score"].values))
            else:
                expected.append(np.nan)
        expected = np.array(expected, dtype=float)

        result_vals = result["value"].to_numpy(dtype=float)
        np.testing.assert_array_equal(np.isnan(result_vals), np.isnan(expected))
        both_valid = ~np.isnan(result_vals) & ~np.isnan(expected)
        if np.any(both_valid):
            np.testing.assert_allclose(result_vals[both_valid], expected[both_valid], rtol=1e-3)

    def test_sliding_window_transition(self):
        """sliding window LSE transitions correctly from disjoint to overlapping."""
        pm.gvtrack_create("vt_lse_transition", src="dense_track", func="lse")

        # First, use disjoint intervals (non-sliding pattern)
        disjoint = pd.concat([
            pm.gintervals("1", 0, 200),
            pm.gintervals("1", 500, 700),
        ], ignore_index=True)
        pm.gextract("vt_lse_transition", disjoint, iterator=-1, colnames=["value"])

        # Then use consecutive sliding intervals
        sliding = pd.concat([
            pm.gintervals("1", 0, 200),
            pm.gintervals("1", 20, 220),
            pm.gintervals("1", 40, 240),
        ], ignore_index=True)
        res_sliding = pm.gextract("vt_lse_transition", sliding, iterator=-1, colnames=["value"])

        # Verify each individually against manual computation
        for i in range(len(res_sliding)):
            val = res_sliding["value"].values[i]
            if not np.isnan(val):
                manual = _manual_lse(
                    "dense_track",
                    str(res_sliding["chrom"].values[i]),
                    int(res_sliding["start"].values[i]),
                    int(res_sliding["end"].values[i]),
                )
                np.testing.assert_allclose(val, manual, rtol=1e-3,
                                           err_msg=f"sliding window {i}")

    def test_sliding_vs_nonsliding_dense(self):
        """sliding window LSE on dense track gives same result as non-sliding."""
        pm.gvtrack_create("vt_lse_slide_verify", src="dense_track", func="lse")

        region = pm.gintervals("1", 0, 1000)

        # Small iterator = many non-overlapping windows
        result_small = pm.gextract("vt_lse_slide_verify", region, iterator=100, colnames=["value"])
        manual = _manual_lse_vec("dense_track", result_small[["chrom", "start", "end"]])

        result_vals = result_small["value"].to_numpy(dtype=float)
        both_valid = ~np.isnan(result_vals) & ~np.isnan(manual)
        if np.any(both_valid):
            np.testing.assert_allclose(result_vals[both_valid], manual[both_valid], rtol=1e-3)
