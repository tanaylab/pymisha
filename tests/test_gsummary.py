import conftest
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
import pymisha.summary as pm_summary


def _expected_summary(values):
    total = float(values.size)
    if total == 0:
        return {
            "Total intervals": 0.0,
            "NaN intervals": 0.0,
            "Min": np.nan,
            "Max": np.nan,
            "Sum": np.nan,
            "Mean": np.nan,
            "Std dev": np.nan,
        }

    nan_mask = np.isnan(values)
    num_nan = float(np.count_nonzero(nan_mask))
    num_non_nan = values.size - num_nan
    if num_non_nan == 0:
        min_val = max_val = sum_val = mean_val = stdev_val = np.nan
    else:
        valid = values[~nan_mask]
        min_val = float(np.min(valid))
        max_val = float(np.max(valid))
        sum_val = float(np.sum(valid))
        mean_val = float(sum_val / num_non_nan)
        stdev_val = float(np.std(valid, ddof=1)) if num_non_nan > 1 else np.nan

    return {
        "Total intervals": total,
        "NaN intervals": num_nan,
        "Min": min_val,
        "Max": max_val,
        "Sum": sum_val,
        "Mean": mean_val,
        "Std dev": stdev_val,
    }


def _assert_summary_matches(summary, expected):
    for key, exp in expected.items():
        val = summary[key]
        if np.isnan(exp):
            assert np.isnan(val)
        else:
            assert val == pytest.approx(exp)


def test_gsummary_dense_track_all_intervals():
    intervals = pm.gintervals_all()
    values = conftest.extract_values("dense_track", intervals)
    expected = _expected_summary(values)

    summary = pm.gsummary("dense_track", intervals)
    _assert_summary_matches(summary, expected)


def test_gsummary_dense_track_filtered_intervals():
    intervals = pm.gscreen("dense_track > 0.2", pm.gintervals_all())
    values = conftest.extract_values("dense_track", intervals)
    expected = _expected_summary(values)

    summary = pm.gsummary("dense_track", intervals)
    _assert_summary_matches(summary, expected)


def test_gsummary_sparse_track_all_intervals():
    intervals = pm.gintervals_all()
    values = conftest.extract_values("sparse_track", intervals)
    expected = _expected_summary(values)

    summary = pm.gsummary("sparse_track", intervals)
    _assert_summary_matches(summary, expected)


def test_gsummary_vtrack_streaming_uses_stable_variance(monkeypatch):
    iter_df = pd.DataFrame(
        {
            "chrom": ["chr1"] * 10,
            "start": np.arange(10, dtype=int),
            "end": np.arange(1, 11, dtype=int),
            "intervalID": np.arange(1, 11, dtype=int),
        }
    )

    monkeypatch.setattr(
        pm_summary._shared,
        "_iterated_intervals",
        lambda intervals, iterator: iter_df,
    )

    result = pm_summary._gsummary_vtrack_streaming(
        "1000000000000.0 + (START % 2)",
        intervals=None,
        iterator=1,
    )

    expected_std = np.std(1000000000000.0 + (iter_df["start"].to_numpy(dtype=float) % 2.0), ddof=1)
    assert result["Std dev"] == pytest.approx(expected_std)
    assert result["Std dev"] > 0.0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_gsummary_empty_intervals():
    """gsummary on an empty intervals DataFrame returns all-NaN summary with zero counts."""
    empty_intervals = pd.DataFrame(
        {"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int), "end": pd.Series([], dtype=int)}
    )
    summary = pm.gsummary("dense_track", empty_intervals)
    expected = _expected_summary(np.array([], dtype=float))
    _assert_summary_matches(summary, expected)


def test_gsummary_single_bp_interval():
    """gsummary on a 1-bp interval returns correct stats for a single value."""
    # Use chrom 1, position 0-1 (1 bp)
    intervals = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [1]})
    values = conftest.extract_values("dense_track", intervals)
    expected = _expected_summary(values)

    summary = pm.gsummary("dense_track", intervals)
    _assert_summary_matches(summary, expected)

    # A single non-NaN value should have NaN std dev (ddof=1 with n=1)
    if values.size == 1 and not np.isnan(values[0]):
        assert np.isnan(summary["Std dev"])


def test_gsummary_single_bin_all_values():
    """gsummary on a small region produces stats matching the extracted values."""
    # Pick a small region that maps to a single iterator bin
    intervals = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    values = conftest.extract_values("dense_track", intervals)
    expected = _expected_summary(values)

    summary = pm.gsummary("dense_track", intervals)
    _assert_summary_matches(summary, expected)
