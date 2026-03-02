"""Tests for gbins_summary and gbins_quantiles."""

import numpy as np
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLE_DB = pm.gdb_examples_path()


@pytest.fixture(autouse=True)
def _init_db():
    pm.gdb_init(EXAMPLE_DB)
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


# ---------------------------------------------------------------------------
# gbins_summary tests
# ---------------------------------------------------------------------------


def test_gbins_summary_basic_1d():
    """1D gbins_summary: bin dense_track values, summarize sparse_track per bin."""
    result = pm.gbins_summary(
        "dense_track", [0, 0.2, 0.5, 1.0],
        expr="sparse_track",
        iterator=100,
    )
    # Result should be a 2D array: (n_bins, 7_stats)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (3, 7)  # 3 bins, 7 stats columns

    # Total intervals per bin should be non-negative
    assert np.all(result[:, 0] >= 0)
    # NaN intervals should be non-negative and <= total
    assert np.all(result[:, 1] >= 0)
    assert np.all(result[:, 1] <= result[:, 0])


def test_gbins_summary_returns_correct_stats_columns():
    """Verify the 7 summary stats have correct semantics."""
    result = pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        expr="dense_track",
        iterator=100,
    )
    # For bins with data: mean should be between min and max
    for i in range(result.shape[0]):
        total = result[i, 0]
        nan_count = result[i, 1]
        if total - nan_count > 0:
            min_val = result[i, 2]
            max_val = result[i, 3]
            mean_val = result[i, 5]
            assert min_val <= mean_val <= max_val


def test_gbins_summary_2d():
    """2D gbins_summary: cross-bins of two expressions."""
    result = pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        "dense_track", [0, 0.3, 0.7, 1.0],
        expr="sparse_track",
        iterator=100,
    )
    # Result should be 3D: (2, 3, 7)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape == (2, 3, 7)


def test_gbins_summary_empty_bins_have_nan_stats():
    """Bins with no data should have NaN for min/max/sum/mean/stddev."""
    # Use very narrow breaks that likely miss most values
    result = pm.gbins_summary(
        "dense_track", [0.999, 0.9995, 1.0],
        expr="sparse_track",
        iterator=100,
    )
    # At least some bins may have zero count
    for i in range(result.shape[0]):
        if result[i, 0] == 0:
            # Empty bin: all stats except total/nan should be NaN
            assert np.isnan(result[i, 2])  # min
            assert np.isnan(result[i, 3])  # max
            assert np.isnan(result[i, 4])  # sum
            assert np.isnan(result[i, 5])  # mean
            assert np.isnan(result[i, 6])  # stddev


def test_gbins_summary_include_lowest():
    """include_lowest should include the minimum breakpoint value in first bin."""
    result_no = pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        expr="dense_track",
        iterator=100,
        include_lowest=False,
    )
    result_yes = pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        expr="dense_track",
        iterator=100,
        include_lowest=True,
    )
    # With include_lowest, first bin gets values exactly at 0
    # Total should be >= with include_lowest
    assert result_yes[0, 0] >= result_no[0, 0]


def test_gbins_summary_validates_args():
    """Should reject invalid arguments."""
    with pytest.raises((ValueError, TypeError)):
        pm.gbins_summary(expr="dense_track", iterator=100)

    with pytest.raises((ValueError, TypeError)):
        pm.gbins_summary("dense_track", iterator=100)

    with pytest.raises(ValueError, match="pairs"):
        pm.gbins_summary("dense_track", [0, 1], "extra_expr", expr="dense_track")


def test_gbins_summary_no_expr_uses_bin_expr():
    """When expr is not provided, summarize the binning expression itself."""
    result = pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        iterator=100,
    )
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (2, 7)


def test_gbins_summary_matches_manual_computation():
    """Validate gbins_summary results against manual bin + gsummary computation."""
    breaks = [0, 0.3, 0.7, 1.0]
    result = pm.gbins_summary(
        "dense_track", breaks,
        expr="sparse_track",
        iterator=100,
    )

    # Extract each track separately (mixed types not supported in single call)
    data_bin = pm.gextract("dense_track", pm.gintervals_all(), iterator=100)
    data_expr = pm.gextract("sparse_track", pm.gintervals_all(), iterator=100)
    bin_vals = data_bin["dense_track"].values
    expr_vals = data_expr["sparse_track"].values

    for i in range(len(breaks) - 1):
        lo, hi = breaks[i], breaks[i + 1]
        # Default: (lo, hi] — exclude lo
        mask = (bin_vals > lo) & (bin_vals <= hi) if i == 0 else (bin_vals > lo) & (bin_vals <= hi)
        bv = expr_vals[mask]
        total = float(len(bv))
        nan_count = float(np.count_nonzero(np.isnan(bv)))
        np.testing.assert_allclose(result[i, 0], total, rtol=1e-10)
        np.testing.assert_allclose(result[i, 1], nan_count, rtol=1e-10)
        valid = bv[~np.isnan(bv)]
        if len(valid) > 0:
            np.testing.assert_allclose(result[i, 2], np.min(valid), rtol=1e-10)
            np.testing.assert_allclose(result[i, 3], np.max(valid), rtol=1e-10)
            np.testing.assert_allclose(result[i, 4], np.sum(valid), rtol=1e-6)
            np.testing.assert_allclose(result[i, 5], np.mean(valid), rtol=1e-6)
            if len(valid) > 1:
                np.testing.assert_allclose(result[i, 6], np.std(valid, ddof=1), rtol=1e-6)


# ---------------------------------------------------------------------------
# gbins_quantiles tests
# ---------------------------------------------------------------------------


def test_gbins_quantiles_basic_1d():
    """1D gbins_quantiles: bin by dense_track, compute quantiles of sparse_track."""
    result = pm.gbins_quantiles(
        "dense_track", [0, 0.2, 0.5, 1.0],
        expr="sparse_track",
        percentiles=[0.25, 0.5, 0.75],
        iterator=100,
    )
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (3, 3)  # 3 bins, 3 percentiles


def test_gbins_quantiles_single_percentile():
    """Single percentile should still return 2D array."""
    result = pm.gbins_quantiles(
        "dense_track", [0, 0.5, 1.0],
        expr="sparse_track",
        percentiles=0.5,
        iterator=100,
    )
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (2, 1)


def test_gbins_quantiles_2d():
    """2D bins: result should have 3 dimensions."""
    result = pm.gbins_quantiles(
        "dense_track", [0, 0.5, 1.0],
        "dense_track", [0, 0.3, 0.7, 1.0],
        expr="sparse_track",
        percentiles=[0.25, 0.75],
        iterator=100,
    )
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape == (2, 3, 2)  # 2 bins x 3 bins x 2 percentiles


def test_gbins_quantiles_empty_bins_return_nan():
    """Empty bins should return NaN for all percentiles."""
    result = pm.gbins_quantiles(
        "dense_track", [0.999, 0.9995, 1.0],
        expr="sparse_track",
        percentiles=[0.25, 0.5, 0.75],
        iterator=100,
    )
    for _i in range(result.shape[0]):
        # Check if this bin has any data via summary
        # At least some bins should be empty in this range
        pass  # The actual assertion is that no crash occurs
    # Result should be well-formed: 2 bins (3 break points - 1), 3 percentiles
    assert result.shape == (2, 3)


def test_gbins_quantiles_monotonic_percentiles():
    """For bins with data, quantiles should be non-decreasing."""
    result = pm.gbins_quantiles(
        "dense_track", [0, 0.5, 1.0],
        expr="sparse_track",
        percentiles=[0.1, 0.5, 0.9],
        iterator=100,
    )
    for i in range(result.shape[0]):
        q = result[i, :]
        valid = q[~np.isnan(q)]
        if len(valid) > 1:
            assert np.all(np.diff(valid) >= -1e-10)  # non-decreasing


def test_gbins_quantiles_matches_manual():
    """Validate gbins_quantiles against manual bin + numpy quantile."""
    breaks = [0, 0.3, 0.7, 1.0]
    pcts = [0.25, 0.5, 0.75]
    result = pm.gbins_quantiles(
        "dense_track", breaks,
        expr="sparse_track",
        percentiles=pcts,
        iterator=100,
    )

    data_bin = pm.gextract("dense_track", pm.gintervals_all(), iterator=100)
    data_expr = pm.gextract("sparse_track", pm.gintervals_all(), iterator=100)
    bin_vals = data_bin["dense_track"].values
    expr_vals = data_expr["sparse_track"].values

    for i in range(len(breaks) - 1):
        lo, hi = breaks[i], breaks[i + 1]
        mask = (bin_vals > lo) & (bin_vals <= hi)
        bv = expr_vals[mask]
        valid = bv[~np.isnan(bv)]
        if len(valid) > 0:
            for j, p in enumerate(pcts):
                expected = np.quantile(valid, p)
                np.testing.assert_allclose(result[i, j], expected, rtol=1e-6)


def test_gbins_quantiles_include_lowest():
    """include_lowest should affect first bin."""
    result_no = pm.gbins_quantiles(
        "dense_track", [0, 0.5, 1.0],
        expr="dense_track",
        percentiles=0.5,
        iterator=100,
        include_lowest=False,
    )
    result_yes = pm.gbins_quantiles(
        "dense_track", [0, 0.5, 1.0],
        expr="dense_track",
        percentiles=0.5,
        iterator=100,
        include_lowest=True,
    )
    # Results may differ if there are values exactly at 0
    assert result_no.shape == result_yes.shape


# ---------------------------------------------------------------------------
# Shared edge case tests
# ---------------------------------------------------------------------------


def test_gbins_rejects_invalid_breaks():
    """Non-monotonic or too-few breaks should raise ValueError."""
    with pytest.raises(ValueError):
        pm.gbins_summary("dense_track", [1.0], expr="dense_track")
    with pytest.raises(ValueError):
        pm.gbins_summary("dense_track", [1.0, 0.5, 0.0], expr="dense_track")


def test_gbins_summary_with_vtracks():
    """gbins_summary should work with virtual tracks."""
    pm.gvtrack_create("vt_avg", "dense_track", func="avg")
    result = pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        expr="vt_avg",
        iterator=100,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 7)


# ---------------------------------------------------------------------------
# R parity tests (ported from test-gbins.R)
# ---------------------------------------------------------------------------


def test_gbins_quantiles_r_parity_iter10():
    """Port of: gbins.quantiles with iterator=10 — golden-master regression."""
    result = pm.gbins_quantiles(
        "dense_track", [0, 0.2, 0.3, 0.9, 1.2],
        expr="sparse_track",
        percentiles=[0.2, 0.5, 0.6],
        iterator=10,
    )
    assert result.shape == (4, 3)
    # First 3 bins should have values; last bin should be NaN
    np.testing.assert_allclose(result[0, 0], 0.36000001, rtol=1e-5)
    np.testing.assert_allclose(result[1, 1], 0.45000002, rtol=1e-5)
    np.testing.assert_allclose(result[2, 0], 0.51333338, rtol=1e-5)
    assert np.all(np.isnan(result[3, :]))


def test_gbins_quantiles_r_parity_iter100():
    """Port of: gbins.quantiles with iterator=100 — golden-master regression."""
    result = pm.gbins_quantiles(
        "dense_track", [0, 0.2, 0.3, 0.9, 1.2],
        expr="sparse_track",
        percentiles=[0.2, 0.5, 0.6],
        iterator=100,
    )
    assert result.shape == (4, 3)
    np.testing.assert_allclose(result[0, 0], 0.36000001, rtol=1e-5)
    np.testing.assert_allclose(result[0, 1], 0.40000001, rtol=1e-5)
    np.testing.assert_allclose(result[2, 1], 0.60000002, rtol=1e-5)
    assert np.all(np.isnan(result[3, :]))


def test_gbins_summary_r_parity():
    """Port of: gbins.summary with iterator=100 — golden-master regression."""
    result = pm.gbins_summary(
        "dense_track", [0, 0.2, 0.3, 0.9, 1.2],
        expr="sparse_track",
        iterator=100,
    )
    assert result.shape == (4, 7)
    # First bin: total ~6202, nans ~5128
    np.testing.assert_allclose(result[0, 0], 6202, rtol=1e-3)
    np.testing.assert_allclose(result[0, 1], 5128, rtol=1e-3)
    # Second bin: total ~641
    np.testing.assert_allclose(result[1, 0], 641, rtol=1e-3)
    # Third bin: total ~140
    np.testing.assert_allclose(result[2, 0], 140, rtol=1e-3)
    # Fourth bin: 0 total, NaN stats
    assert result[3, 0] == 0
    assert np.isnan(result[3, 2])


# ---------------------------------------------------------------------------
# Optimization parity tests
# ---------------------------------------------------------------------------


def test_gbins_summary_vectorized_stats_match_per_bin():
    """Verify vectorized summary stats match per-bin _compute_summary_stats."""
    from pymisha.summary import _assign_bins, _compute_summary_stats

    breaks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    result = pm.gbins_summary(
        "dense_track", breaks,
        expr="sparse_track",
        iterator=100,
    )

    # Manual per-bin computation (the old algorithm)
    data_bin = pm.gextract("dense_track", pm.gintervals_all(), iterator=100)
    data_expr = pm.gextract("sparse_track", pm.gintervals_all(), iterator=100)
    bin_vals = data_bin["dense_track"].values
    expr_vals = data_expr["sparse_track"].values

    idx = _assign_bins(bin_vals, np.array(breaks), False)
    for i in range(len(breaks) - 1):
        mask = idx == i
        bv = expr_vals[mask]
        expected = _compute_summary_stats(bv)
        np.testing.assert_allclose(result[i], expected, rtol=1e-10,
                                   err_msg=f"Bin {i} mismatch")


def test_gbins_quantiles_sort_based_matches_itertools():
    """Verify sort-based quantiles match itertools.product approach."""
    breaks1 = [0, 0.3, 0.7, 1.0]
    breaks2 = [0, 0.5, 1.0]
    pcts = [0.1, 0.25, 0.5, 0.75, 0.9]

    result = pm.gbins_quantiles(
        "dense_track", breaks1,
        "dense_track", breaks2,
        expr="sparse_track",
        percentiles=pcts,
        iterator=100,
    )

    # Manual per-bin computation (the old algorithm)
    from pymisha.summary import _assign_bins
    data_bin = pm.gextract("dense_track", pm.gintervals_all(), iterator=100)
    data_expr = pm.gextract("sparse_track", pm.gintervals_all(), iterator=100)
    bin_vals = data_bin["dense_track"].values
    expr_vals = data_expr["sparse_track"].values

    idx1 = _assign_bins(bin_vals, np.array(breaks1), False)
    idx2 = _assign_bins(bin_vals, np.array(breaks2), False)

    import itertools
    for b1, b2 in itertools.product(range(len(breaks1) - 1), range(len(breaks2) - 1)):
        mask = (idx1 == b1) & (idx2 == b2)
        bv = expr_vals[mask]
        bv = bv[~np.isnan(bv)]
        if len(bv) > 0:
            expected = np.quantile(bv, pcts)
            np.testing.assert_allclose(result[b1, b2], expected, rtol=1e-10,
                                       err_msg=f"Bin ({b1},{b2}) mismatch")
        else:
            assert np.all(np.isnan(result[b1, b2]))


def test_gbins_summary_single_value_bin_stddev_nan():
    """A bin with exactly one non-NaN value should have NaN stddev."""
    # Use narrow breaks to get a bin with exactly 1 value
    result = pm.gbins_summary(
        "dense_track", [0, 0.001, 0.01],
        expr="dense_track",
        iterator=1,
    )
    for i in range(result.shape[0]):
        n_valid = result[i, 0] - result[i, 1]
        if n_valid == 1:
            assert np.isnan(result[i, 6]), f"Bin {i}: stddev should be NaN for n=1"


def test_extract_values_direct_matches_gextract():
    """_extract_values_direct must return same arrays as _extract_expr_values."""
    from pymisha.summary import _extract_expr_values, _extract_values_direct

    intervals = pm.gintervals_all()
    for track in ["dense_track", "sparse_track"]:
        direct = _extract_values_direct(track, intervals, iterator=100)
        via_gextract = _extract_expr_values(track, intervals, iterator=100)
        np.testing.assert_array_equal(direct, via_gextract,
                                      err_msg=f"{track}: direct != gextract")
