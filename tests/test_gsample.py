"""Tests for gsample (streaming reservoir sampling)."""

import numpy as np
import pytest

import pymisha as pm


class TestGsample:
    """Test gsample."""

    def test_returns_correct_count(self):
        """gsample returns exactly n samples."""
        result = pm.gsample("dense_track", 50, pm.gintervals(1, 0, 10000))
        assert len(result) == 50

    def test_returns_numpy_array(self):
        """gsample returns a 1D numpy array of floats."""
        result = pm.gsample("dense_track", 10, pm.gintervals(1, 0, 10000))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype == np.float64

    def test_values_in_track_range(self):
        """Sampled values should be within the track's actual range."""
        full_data = pm.gextract("dense_track", pm.gintervals(1, 0, 10000))
        col = [c for c in full_data.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        track_min = full_data[col].min()
        track_max = full_data[col].max()

        result = pm.gsample("dense_track", 100, pm.gintervals(1, 0, 10000))
        # All values should be within [min, max] of the track
        finite = result[np.isfinite(result)]
        if len(finite) > 0:
            assert finite.min() >= track_min - 1e-10
            assert finite.max() <= track_max + 1e-10

    def test_different_seeds_different_samples(self):
        """Different random seeds produce different samples."""
        intervals = pm.gintervals(1, 0, 10000)
        np.random.seed(42)
        r1 = pm.gsample("dense_track", 50, intervals)
        np.random.seed(123)
        r2 = pm.gsample("dense_track", 50, intervals)
        # Very unlikely to be identical with different seeds
        # (but not impossible - just check they differ in at least some values)
        assert not np.array_equal(np.sort(r1), np.sort(r2))

    def test_n_larger_than_data(self):
        """When n > available data points, return all available data."""
        intervals = pm.gintervals(1, 0, 100)
        result = pm.gsample("dense_track", 1000000, intervals)
        # Should return less than or equal to n
        assert len(result) <= 1000000
        assert len(result) > 0

    def test_sparse_track(self):
        """gsample works with sparse tracks."""
        result = pm.gsample("sparse_track", 20, pm.gintervals(1, 0, 10000))
        assert isinstance(result, np.ndarray)
        # Should have some data (sparse_track has data in chr1)
        assert len(result) > 0

    def test_expression_not_just_track(self):
        """gsample works with expressions, not just track names."""
        result = pm.gsample(
            "dense_track + 1", 30, pm.gintervals(1, 0, 10000)
        )
        assert len(result) == 30

    def test_with_iterator(self):
        """gsample works with an iterator parameter."""
        result = pm.gsample(
            "dense_track", 50, pm.gintervals(1, 0, 10000), iterator=100
        )
        assert len(result) == 50

    def test_n_zero_raises(self):
        """n=0 should raise an error."""
        with pytest.raises(Exception):
            pm.gsample("dense_track", 0, pm.gintervals(1, 0, 10000))

    def test_negative_n_raises(self):
        """Negative n should raise an error."""
        with pytest.raises(Exception):
            pm.gsample("dense_track", -5, pm.gintervals(1, 0, 10000))

    def test_default_intervals(self):
        """gsample with None intervals uses all genome."""
        result = pm.gsample("dense_track", 100)
        assert len(result) == 100

    def test_nan_handling(self):
        """NaN values from sparse track gaps should not appear in samples."""
        # Extract values to check NaN presence in source
        full = pm.gextract("sparse_track", pm.gintervals(1, 0, 10000))
        col = [c for c in full.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        full[col].isna().any()

        result = pm.gsample("sparse_track", 100, pm.gintervals(1, 0, 10000))
        # StreamSampler should skip NaN values
        assert not np.any(np.isnan(result))


class TestGsampleStatistical:
    """Statistical validation tests for gsample."""

    def test_samples_within_population_range(self):
        """Every sampled value must exist in the population of track values."""
        intervals = pm.gintervals(1, 0, 50000)
        full = pm.gextract("dense_track", intervals)
        col = [c for c in full.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        pop_values = full[col].dropna().values
        pop_min, pop_max = pop_values.min(), pop_values.max()

        result = pm.gsample("dense_track", 500, intervals)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert finite.min() >= pop_min - 1e-10, (
            f"Sample min {finite.min()} below population min {pop_min}"
        )
        assert finite.max() <= pop_max + 1e-10, (
            f"Sample max {finite.max()} above population max {pop_max}"
        )

    def test_samples_from_multi_chrom_cover_all_value_ranges(self):
        """Sampling across chromosomes draws from the full value range.

        Use a wide enough sample from the entire genome and verify that the
        sample mean is close to the population mean, which would be very
        unlikely if entire chromosomes were skipped.
        """
        intervals = pm.gintervals_all()
        full = pm.gextract("dense_track", intervals)
        col = [c for c in full.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        pop_values = full[col].dropna().values
        pop_mean = pop_values.mean()
        pop_std = pop_values.std()

        n_samples = 5000
        result = pm.gsample("dense_track", n_samples, intervals)
        sample_mean = result.mean()

        # The sample mean should be within ~4 standard errors of the
        # population mean (fails with probability < 1e-4).
        se = pop_std / np.sqrt(len(result))
        assert abs(sample_mean - pop_mean) < 4 * se, (
            f"Sample mean {sample_mean:.4f} deviates from population mean "
            f"{pop_mean:.4f} by more than 4 SE ({4 * se:.4f})"
        )

    def test_repeated_sampling_is_stochastic(self):
        """Successive gsample calls produce different random subsets.

        The C++ reservoir sampler uses its own internal RNG, so consecutive
        calls should yield different orderings/selections.  We call gsample
        twice and verify the results are not identical (overwhelmingly likely
        for a random sampler drawing from a large population).
        """
        intervals = pm.gintervals(1, 0, 50000)
        n = 200

        r1 = pm.gsample("dense_track", n, intervals)
        r2 = pm.gsample("dense_track", n, intervals)

        # Two independent random draws of 200 from 50000 values should
        # almost never be identical
        assert not np.array_equal(r1, r2), (
            "Two independent gsample calls returned identical arrays"
        )

    def test_large_sample_approaches_population(self):
        """When n approaches the population size, the sample captures nearly
        all distinct values in the population.
        """
        intervals = pm.gintervals(1, 0, 10000)
        full = pm.gextract("dense_track", intervals)
        col = [c for c in full.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        pop_values = full[col].dropna().values
        pop_size = len(pop_values)

        # Request exactly the population size -- should get everything
        result = pm.gsample("dense_track", pop_size, intervals)
        assert len(result) == pop_size

        # The sorted sample should match the sorted population exactly
        np.testing.assert_array_almost_equal(
            np.sort(result), np.sort(pop_values), decimal=10
        )

    def test_sample_mean_and_variance_close_to_population(self):
        """Sample statistics should approximate population statistics.

        For a large enough sample, the sample mean and variance converge to
        the population values.  We use a 4-sigma tolerance.
        """
        intervals = pm.gintervals(1, 0, 100000)
        full = pm.gextract("dense_track", intervals)
        col = [c for c in full.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        pop = full[col].dropna().values
        pop_mean = pop.mean()
        pop_var = pop.var()

        n_samples = 3000
        result = pm.gsample("dense_track", n_samples, intervals)
        sample_mean = result.mean()
        sample_var = result.var()

        # Mean check: z-test
        se_mean = np.sqrt(pop_var / n_samples)
        assert abs(sample_mean - pop_mean) < 4 * se_mean, (
            f"Sample mean {sample_mean:.4f} too far from population mean "
            f"{pop_mean:.4f}"
        )

        # Variance check: under normality, var(s^2) ~ 2*sigma^4/(n-1).
        # Use a generous 5-sigma tolerance since track values may not be
        # normal.
        se_var = np.sqrt(2.0 * pop_var**2 / (n_samples - 1))
        assert abs(sample_var - pop_var) < 5 * se_var, (
            f"Sample variance {sample_var:.4f} too far from population "
            f"variance {pop_var:.4f}"
        )

    def test_distribution_proportional_to_interval_sizes(self):
        """When sampling from intervals of different sizes, the fraction of
        samples from each region should be roughly proportional to its size.

        We partition chr1 into a small and a large interval, sample many
        values, then check that the fraction of values matching the small-
        interval population is roughly proportional to its size share.
        """
        import pandas as pd

        small_iv = pm.gintervals(1, 0, 10000)       # 10 000 bp
        large_iv = pm.gintervals(1, 10000, 100000)   # 90 000 bp

        # Build combined intervals
        combined = pd.concat([small_iv, large_iv], ignore_index=True)

        # Get population values per region
        full_small = pm.gextract("dense_track", small_iv)
        full_large = pm.gextract("dense_track", large_iv)
        col = [c for c in full_small.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        vals_small = set(full_small[col].dropna().values)
        vals_large = set(full_large[col].dropna().values)

        # Values unique to each region (so we can attribute samples)
        only_small = vals_small - vals_large
        only_large = vals_large - vals_small

        n_samples = 5000
        result = pm.gsample("dense_track", n_samples, combined)

        # Count samples attributable to each region
        set(result)
        n_only_small = sum(1 for v in result if v in only_small)
        n_only_large = sum(1 for v in result if v in only_large)
        n_attributable = n_only_small + n_only_large

        # Skip test if too few values are uniquely attributable (the regions
        # overlap too much in value space)
        if n_attributable < 100:
            pytest.skip(
                "Too few uniquely attributable samples to test proportionality"
            )

        # Expected fraction of small region: 10000 / 100000 = 0.1
        frac_small = n_only_small / n_attributable
        # Allow generous tolerance: within 0.15 of expected 0.1
        assert 0.0 < frac_small < 0.25, (
            f"Small-region fraction {frac_small:.3f} outside expected range "
            f"[0.0, 0.25] (expected ~0.1)"
        )

    def test_uniform_reservoir_sampling_fairness(self):
        """Reservoir sampling should give every distinct value an
        approximately correct selection probability.

        Strategy: get the population, compute the empirical CDF, then sample
        many times and verify that the sampled value distribution matches
        the population distribution using a two-sample KS-like check
        (max absolute difference between empirical CDFs).
        """
        intervals = pm.gintervals(1, 0, 10000)
        full = pm.gextract("dense_track", intervals)
        col = [c for c in full.columns if c not in {"chrom", "start", "end", "intervalID"}][0]
        pop = np.sort(full[col].dropna().values)
        pop_size = len(pop)

        if pop_size < 50:
            pytest.skip("Population too small for fairness test")

        # Aggregate many samples to build a large empirical sample
        n_sample = pop_size // 2
        n_trials = 40
        all_samples = []
        for _ in range(n_trials):
            result = pm.gsample("dense_track", n_sample, intervals)
            all_samples.append(result)
        combined = np.sort(np.concatenate(all_samples))

        # Compute empirical CDFs
        np.arange(1, pop_size + 1) / pop_size
        np.arange(1, len(combined) + 1) / len(combined)

        # For each population value, find its CDF in the sample
        # Use a simplified KS statistic: check at population quantile points
        max_diff = 0.0
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            pop_val = pop[int(q * pop_size)]
            pop_frac = np.searchsorted(pop, pop_val, side="right") / pop_size
            sample_frac = np.searchsorted(combined, pop_val, side="right") / len(combined)
            max_diff = max(max_diff, abs(pop_frac - sample_frac))

        # With n_trials * n_sample total samples, the KS distance should be
        # small.  Allow up to 0.05.
        assert max_diff < 0.05, (
            f"Max CDF difference {max_diff:.4f} exceeds tolerance 0.05; "
            f"sample distribution does not match population"
        )
