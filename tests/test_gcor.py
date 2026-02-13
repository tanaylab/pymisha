"""Tests for gcor (Pearson + Spearman variants)."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _extract_paired(expr1, expr2, intervals, iterator):
    """Extract two expressions separately and align them for comparison."""
    v1 = pm.gextract(expr1, intervals=intervals, iterator=iterator)
    v2 = pm.gextract(expr2, intervals=intervals, iterator=iterator)
    # Get the data columns (not chrom/start/end/intervalID)
    meta = {"chrom", "start", "end", "intervalID"}
    col1 = [c for c in v1.columns if c not in meta][0]
    col2 = [c for c in v2.columns if c not in meta][0]
    # Merge on genomic position
    merged = v1[["chrom", "start", "end", col1]].merge(
        v2[["chrom", "start", "end", col2]],
        on=["chrom", "start", "end"],
    )
    return merged[col1].values, merged[col2].values


def _spearman_from_vectors(x, y):
    """Compute exact Spearman rho with average ranks for ties."""
    mask = np.isfinite(x) & np.isfinite(y)
    xc = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yc = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    if len(xc) < 2:
        return np.nan
    return np.corrcoef(xc, yc)[0, 1]


class TestGcor:
    """Test gcor."""

    def test_basic_pearson_correlation(self):
        """Pearson correlation between two tracks matches manual calculation."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        cor_result = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=iterator,
        )
        assert isinstance(cor_result, np.ndarray)
        assert len(cor_result) == 1

        # Verify against manual calculation from extracted values
        x, y = _extract_paired("dense_track", "sparse_track", intervals, iterator)
        mask = np.isfinite(x) & np.isfinite(y)
        expected = np.corrcoef(x[mask], y[mask])[0, 1]

        np.testing.assert_allclose(cor_result[0], expected, rtol=1e-10)

    def test_same_type_correlation(self):
        """Correlation between same-type tracks (both dense) uses fast path."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        cor_result = pm.gcor(
            "dense_track", "dense_track + 1",
            intervals=intervals, iterator=iterator,
        )
        # Adding a constant doesn't change correlation
        np.testing.assert_allclose(cor_result[0], 1.0, rtol=1e-10)

    def test_details_returns_dataframe(self):
        """details=True returns a DataFrame with full statistics."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        result = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=iterator,
            details=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert "cor" in result.columns
        assert "cov" in result.columns
        assert "mean1" in result.columns
        assert "mean2" in result.columns
        assert "sd1" in result.columns
        assert "sd2" in result.columns
        assert "n" in result.columns
        assert "n.na" in result.columns
        assert len(result) == 1

    def test_details_statistics_correct(self):
        """Detailed statistics match manual calculations."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        stats = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=iterator,
            details=True,
        )

        x, y = _extract_paired("dense_track", "sparse_track", intervals, iterator)
        mask = np.isfinite(x) & np.isfinite(y)
        xc = x[mask]
        yc = y[mask]

        np.testing.assert_allclose(stats["n"].iloc[0], len(x), rtol=1e-10)
        np.testing.assert_allclose(stats["n.na"].iloc[0], (~mask).sum(), rtol=1e-10)
        np.testing.assert_allclose(stats["mean1"].iloc[0], xc.mean(), rtol=1e-10)
        np.testing.assert_allclose(stats["mean2"].iloc[0], yc.mean(), rtol=1e-10)
        np.testing.assert_allclose(stats["sd1"].iloc[0], xc.std(ddof=1), rtol=1e-10)
        np.testing.assert_allclose(stats["sd2"].iloc[0], yc.std(ddof=1), rtol=1e-10)
        np.testing.assert_allclose(
            stats["cov"].iloc[0],
            np.cov(xc, yc)[0, 1],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            stats["cor"].iloc[0],
            np.corrcoef(xc, yc)[0, 1],
            rtol=1e-10,
        )

    def test_multiple_pairs(self):
        """gcor handles multiple expression pairs."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        result = pm.gcor(
            "dense_track", "sparse_track",
            "dense_track", "dense_track",
            intervals=intervals, iterator=iterator,
        )
        assert len(result) == 2
        # Self-correlation should be ~1.0 (dense_track with itself)
        np.testing.assert_allclose(result[1], 1.0, rtol=1e-10)

    def test_multiple_pairs_details(self):
        """details=True works with multiple pairs."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        result = pm.gcor(
            "dense_track", "sparse_track",
            "dense_track", "dense_track",
            intervals=intervals, iterator=iterator,
            details=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.index[0] == "dense_track~sparse_track"
        assert result.index[1] == "dense_track~dense_track"
        np.testing.assert_allclose(result.loc["dense_track~dense_track", "cor"], 1.0, rtol=1e-10)

    def test_custom_names(self):
        """Custom pair names appear in the result index."""
        intervals = pm.gintervals(1, 0, 10000)
        result = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=1000,
            details=True, names=["my_pair"],
        )
        assert result.index[0] == "my_pair"

    def test_default_intervals(self):
        """gcor works with default intervals (entire genome)."""
        result = pm.gcor(
            "dense_track", "sparse_track",
            iterator=1000,
        )
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_odd_number_of_expressions_raises(self):
        """Odd number of expressions raises ValueError."""
        with pytest.raises(ValueError, match="even number"):
            pm.gcor("dense_track", "sparse_track", "dense_track",
                     intervals=pm.gintervals(1, 0, 10000))

    def test_single_expression_raises(self):
        """Single expression raises ValueError."""
        with pytest.raises(ValueError, match="at least two"):
            pm.gcor("dense_track", intervals=pm.gintervals(1, 0, 10000))

    def test_expression_not_just_track(self):
        """gcor works with computed expressions, not just track names."""
        intervals = pm.gintervals(1, 0, 10000)
        result = pm.gcor(
            "dense_track", "dense_track + 1",
            intervals=intervals, iterator=1000,
        )
        # Adding a constant doesn't change correlation
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-10)

    def test_multitasking_matches_single(self):
        """Multitasking and single-process results match."""
        intervals = pm.gintervals_all()

        old = pm.CONFIG['multitasking']
        try:
            pm.CONFIG['multitasking'] = False
            r_single = pm.gcor(
                "dense_track", "sparse_track",
                intervals=intervals, iterator=1000,
            )

            pm.CONFIG['multitasking'] = True
            r_multi = pm.gcor(
                "dense_track", "sparse_track",
                intervals=intervals, iterator=1000,
            )

            np.testing.assert_allclose(r_single, r_multi, rtol=1e-10)
        finally:
            pm.CONFIG['multitasking'] = old

    def test_spearman_exact_multitasking_matches_single(self):
        """Spearman exact returns identical results with and without multitasking."""
        intervals = pm.gintervals_all()

        old_multi = pm.CONFIG["multitasking"]
        old_min = pm.CONFIG["min_processes"]
        old_max = pm.CONFIG["max_processes"]
        try:
            pm.CONFIG["min_processes"] = 2
            pm.CONFIG["max_processes"] = 2

            pm.CONFIG["multitasking"] = False
            r_single = pm.gcor(
                "dense_track", "sparse_track",
                intervals=intervals, iterator=1000,
                method="spearman.exact",
                details=True,
            )

            pm.CONFIG["multitasking"] = True
            r_multi = pm.gcor(
                "dense_track", "sparse_track",
                intervals=intervals, iterator=1000,
                method="spearman.exact",
                details=True,
            )

            np.testing.assert_allclose(
                r_single[["n", "n.na", "cor"]].to_numpy(dtype=float),
                r_multi[["n", "n.na", "cor"]].to_numpy(dtype=float),
                rtol=1e-12,
                atol=0.0,
                equal_nan=True,
            )
        finally:
            pm.CONFIG["multitasking"] = old_multi
            pm.CONFIG["min_processes"] = old_min
            pm.CONFIG["max_processes"] = old_max

    def test_spearman_exact_matches_manual_ranks(self):
        """Exact Spearman matches Pearson correlation of average ranks."""
        intervals = pm.gintervals(1, 0, 10000)
        iterator = 1000

        result = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=iterator,
            method="spearman.exact",
        )
        x, y = _extract_paired("dense_track", "sparse_track", intervals, iterator)
        expected = _spearman_from_vectors(x, y)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10, atol=1e-12)

    def test_spearman_exact_details_columns(self):
        """Spearman details include only (n, n.na, cor), matching R output."""
        result = pm.gcor(
            "dense_track", "sparse_track",
            intervals=pm.gintervals(1, 0, 10000), iterator=1000,
            method="spearman.exact",
            details=True,
        )
        assert list(result.columns) == ["n", "n.na", "cor"]
        assert len(result) == 1

    def test_spearman_approx_close_to_exact(self):
        """Approximate Spearman is close to exact on example DB."""
        intervals = pm.gintervals(1, 0, 20000)
        iterator = 1000

        exact = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=iterator,
            method="spearman.exact",
        )[0]
        approx = pm.gcor(
            "dense_track", "sparse_track",
            intervals=intervals, iterator=iterator,
            method="spearman",
        )[0]
        assert np.isfinite(approx)
        np.testing.assert_allclose(approx, exact, atol=0.15, rtol=0)

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            pm.gcor(
                "dense_track", "sparse_track",
                intervals=pm.gintervals(1, 0, 10000), iterator=1000,
                method="kendall",
            )
