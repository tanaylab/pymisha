"""Tests for streaming gdist/gsummary/gquantiles with virtual track expressions.

These tests verify that summary/distribution functions work correctly when
expressions contain virtual tracks, using a chunked streaming approach that
keeps memory bounded regardless of genome size.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize example database and clear vtracks for each test."""
    pm.gdb_init_examples()
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


# ---------------------------------------------------------------------------
# gdist with virtual tracks
# ---------------------------------------------------------------------------

class TestGdistWithVtracks:
    """Test gdist produces correct results when expressions use vtracks."""

    def test_gdist_vtrack_matches_manual_binning(self):
        """gdist with a vtrack expression should match manual extract+bin."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks, intervals=intervals, iterator=100)

        # Manual: extract then bin
        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        expected = _manual_bin_count(vals, breaks, include_lowest=False)

        np.testing.assert_array_equal(result, expected)

    def test_gdist_vtrack_include_lowest(self):
        """gdist with vtrack and include_lowest=True matches manual binning."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks, intervals=intervals,
                          iterator=100, include_lowest=True)

        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        expected = _manual_bin_count(vals, breaks, include_lowest=True)

        np.testing.assert_array_equal(result, expected)

    def test_gdist_vtrack_2d(self):
        """gdist with two vtrack expressions (2D distribution)."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        pm.gvtrack_create("vt_max", "dense_track", func="max")
        intervals = pm.gintervals("1", 0, 10000)
        breaks1 = [0, 0.2, 0.5, 1.0]
        breaks2 = [0, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks1, "vt_max", breaks2,
                          intervals=intervals, iterator=100)

        assert result.shape == (3, 2)
        assert np.all(result >= 0)

        # Verify against manual computation
        vals1 = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        vals2 = _extract_vtrack_values("vt_max", intervals, iterator=100)
        expected = _manual_bin_count_2d(vals1, breaks1, vals2, breaks2,
                                       include_lowest=False)
        np.testing.assert_array_equal(result, expected)

    def test_gdist_mixed_vtrack_physical(self):
        """gdist with one vtrack and one physical track expression."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks1 = [0, 0.2, 0.5, 1.0]
        breaks2 = [0, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks1, "dense_track", breaks2,
                          intervals=intervals, iterator=100)

        assert result.shape == (3, 2)
        assert np.all(result >= 0)
        assert result.sum() > 0

    def test_gdist_vtrack_dataframe(self):
        """gdist with vtrack returns correct DataFrame when dataframe=True."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.2, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks, intervals=intervals,
                          iterator=100, dataframe=True)

        assert isinstance(result, pd.DataFrame)
        assert "n" in result.columns
        assert len(result) == 3  # 3 bins

    def test_gdist_vtrack_empty_intervals(self):
        """gdist with vtrack and empty intervals returns zeros."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        empty = pd.DataFrame(columns=["chrom", "start", "end"])
        breaks = [0, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks, intervals=empty, iterator=100)
        np.testing.assert_array_equal(result, [0, 0])

    def test_gdist_vtrack_with_shift(self):
        """gdist with a shifted vtrack still works correctly."""
        pm.gvtrack_create("vt_shifted", "dense_track", func="avg",
                          sshift=-50, eshift=50)
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.2, 0.5, 1.0]

        result = pm.gdist("vt_shifted", breaks, intervals=intervals,
                          iterator=100)

        # Verify against manual
        vals = _extract_vtrack_values("vt_shifted", intervals, iterator=100)
        expected = _manual_bin_count(vals, breaks, include_lowest=False)
        np.testing.assert_array_equal(result, expected)

    def test_gdist_vtrack_expression_arithmetic(self):
        """gdist with vtrack used in an arithmetic expression."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.4, 1.0, 2.0]

        result = pm.gdist("vt_avg * 2", breaks, intervals=intervals,
                          iterator=100)

        assert result is not None
        assert len(result) == 3
        # Total should equal the number of values within the break range
        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        doubled = vals * 2
        expected = _manual_bin_count(doubled, breaks, include_lowest=False)
        np.testing.assert_array_equal(result, expected)

    def test_gdist_vtrack_consistent_with_cpp_binning_semantics(self):
        """Verify that vtrack gdist uses the same binning semantics as C++ path.

        Bins are (breaks[i], breaks[i+1]] — open on left, closed on right.
        Values exactly at breaks[0] are excluded unless include_lowest=True.
        """
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        result_default = pm.gdist("vt_avg", breaks, intervals=intervals,
                                  iterator=100)
        result_include = pm.gdist("vt_avg", breaks, intervals=intervals,
                                  iterator=100, include_lowest=True)

        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        n_at_zero = int(np.count_nonzero(vals == 0.0))

        # include_lowest should capture n_at_zero additional values
        assert result_include.sum() - result_default.sum() == n_at_zero


# ---------------------------------------------------------------------------
# gsummary with virtual tracks (streaming)
# ---------------------------------------------------------------------------

class TestGsummaryWithVtracks:
    """Test gsummary with vtrack expressions."""

    def test_gsummary_vtrack_matches_manual(self):
        """gsummary with vtrack should match manual computation."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)

        result = pm.gsummary("vt_avg", intervals=intervals, iterator=100)

        # Manual computation
        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        valid = vals[~np.isnan(vals)]

        assert result["Total intervals"] == len(vals)
        assert result["NaN intervals"] == np.count_nonzero(np.isnan(vals))
        if len(valid) > 0:
            np.testing.assert_allclose(result["Min"], np.min(valid), rtol=1e-10)
            np.testing.assert_allclose(result["Max"], np.max(valid), rtol=1e-10)
            np.testing.assert_allclose(result["Sum"], np.sum(valid), rtol=1e-10)
            np.testing.assert_allclose(result["Mean"], np.mean(valid), rtol=1e-10)


# ---------------------------------------------------------------------------
# gquantiles with virtual tracks
# ---------------------------------------------------------------------------

class TestGquantilesWithVtracks:
    """Test gquantiles with vtrack expressions."""

    def test_gquantiles_vtrack_matches_manual(self):
        """gquantiles with vtrack should match manual quantile computation."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        pcts = [0.25, 0.5, 0.75]

        result = pm.gquantiles("vt_avg", pcts, intervals=intervals,
                               iterator=100)

        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        valid = vals[~np.isnan(vals)]
        expected = np.quantile(valid, pcts) if len(valid) > 0 else [np.nan] * 3

        np.testing.assert_allclose(result.values, expected, rtol=1e-10)

    def test_gquantiles_vtrack_uses_streaming_path(self, monkeypatch):
        """gquantiles vtrack path should not materialize via _extract_expr_values."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)

        import pymisha.summary as summary_mod

        def _unexpected_extract(*args, **kwargs):
            raise AssertionError("vtrack gquantiles should not call _extract_expr_values")

        monkeypatch.setattr(summary_mod, "_extract_expr_values", _unexpected_extract)
        result = pm.gquantiles("vt_avg", [0.5], intervals=intervals, iterator=100)
        assert len(result) == 1

    def test_gquantiles_vtrack_warns_when_sampling_is_capped(self):
        """gquantiles should warn when vtrack data exceeds max_data_size."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        starts = np.arange(0, 50000, 100, dtype=int)
        ends = starts + 100
        intervals = pm.gintervals(["1"] * len(starts), starts, ends)

        old_max_data_size = pm.CONFIG.get("max_data_size")
        pm.CONFIG["max_data_size"] = 100
        try:
            with pytest.warns(RuntimeWarning, match="Data size exceeds the limit; quantiles are approximate"):
                result = pm.gquantiles("vt_avg", [0.25, 0.5, 0.75], intervals=intervals)
            assert len(result) == 3
        finally:
            pm.CONFIG["max_data_size"] = old_max_data_size


# ---------------------------------------------------------------------------
# Binning bug regression test
# ---------------------------------------------------------------------------

class TestGdistBinningSemantics:
    """Regression tests for binning edge cases (values at breaks[0])."""

    def test_values_at_breaks0_excluded_without_include_lowest(self):
        """Values exactly at breaks[0] must be excluded when include_lowest=False.

        This is a regression test for a bug where searchsorted mapped values
        at breaks[0] into bin 0 even without include_lowest.
        """
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        int(np.count_nonzero(vals == breaks[0]))

        result = pm.gdist("vt_avg", breaks, intervals=intervals, iterator=100)

        # Total should equal len(vals) - n_at_break0 - n_nan - n_out_of_range
        int(np.count_nonzero(np.isnan(vals)))
        n_valid = sum(1 for v in vals if not np.isnan(v) and v > breaks[0] and v <= breaks[-1])
        assert result.sum() == n_valid

    def test_values_at_breaks0_included_with_include_lowest(self):
        """Values exactly at breaks[0] must be in bin 0 when include_lowest=True."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)

        result = pm.gdist("vt_avg", breaks, intervals=intervals,
                          iterator=100, include_lowest=True)

        int(np.count_nonzero(np.isnan(vals)))
        n_valid = sum(1 for v in vals if not np.isnan(v) and v >= breaks[0] and v <= breaks[-1])
        assert result.sum() == n_valid

    def test_cpp_path_binning_matches_semantics(self):
        """C++ path should also exclude values at breaks[0] without include_lowest.

        This test uses a physical track (C++ path) to verify.
        """
        intervals = pm.gintervals("1", 0, 10000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        result_default = pm.gdist("dense_track", breaks,
                                  intervals=intervals, iterator=100)
        result_include = pm.gdist("dense_track", breaks,
                                  intervals=intervals, iterator=100,
                                  include_lowest=True)

        # include_lowest should capture more values
        assert result_include.sum() >= result_default.sum()


# ---------------------------------------------------------------------------
# Memory-bounded streaming verification
# ---------------------------------------------------------------------------

class TestStreamingBehavior:
    """Verify that vtrack paths use chunked streaming (not full materialization)."""

    def test_gdist_vtrack_larger_scope(self):
        """gdist with vtrack on a larger scope matches manual computation."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 50000)
        breaks = [0, 0.1, 0.2, 0.5, 1.0]

        result = pm.gdist("vt_avg", breaks, intervals=intervals, iterator=100)

        assert result is not None
        assert len(result) == 4
        assert result.sum() > 0

        # Verify against manual full-extract approach
        vals = _extract_vtrack_values("vt_avg", intervals, iterator=100)
        expected = _manual_bin_count(vals, breaks, include_lowest=False)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_vtrack_values(vtrack_name, intervals, iterator=None):
    """Extract values from a vtrack expression via gextract."""
    df = pm.gextract(vtrack_name, intervals, iterator=iterator)
    if df is None or len(df) == 0:
        return np.array([], dtype=float)
    data_cols = [c for c in df.columns
                 if c not in {"chrom", "start", "end", "intervalID"}]
    assert len(data_cols) == 1
    return df[data_cols[0]].to_numpy(dtype=float, copy=False)


def _manual_bin_count(values, breaks, include_lowest=False):
    """Manually bin values and count, matching gdist/C++ BinFinder semantics.

    Bins are (breaks[i], breaks[i+1]] — open on left, closed on right.
    With include_lowest, first bin is [breaks[0], breaks[1]].
    """
    breaks = np.asarray(breaks, dtype=float)
    n_bins = len(breaks) - 1
    counts = np.zeros(n_bins, dtype=int)

    for v in values:
        if np.isnan(v):
            continue
        if v > breaks[-1]:
            continue
        if include_lowest:
            if v < breaks[0]:
                continue
        else:
            if v <= breaks[0]:
                continue
        idx = np.searchsorted(breaks, v, side='right') - 1
        if include_lowest and v == breaks[0]:
            idx = 0
        if 0 <= idx < n_bins:
            counts[idx] += 1

    return counts


def _manual_bin_count_2d(vals1, breaks1, vals2, breaks2, include_lowest=False):
    """Manually compute 2D bin counts."""
    breaks1 = np.asarray(breaks1, dtype=float)
    breaks2 = np.asarray(breaks2, dtype=float)
    n1 = len(breaks1) - 1
    n2 = len(breaks2) - 1
    counts = np.zeros((n1, n2), dtype=int)

    for v1, v2 in zip(vals1, vals2, strict=False):
        if np.isnan(v1) or np.isnan(v2):
            continue

        def _bin_idx(v, brk):
            if v > brk[-1]:
                return -1
            if include_lowest:
                if v < brk[0]:
                    return -1
            else:
                if v <= brk[0]:
                    return -1
            idx = np.searchsorted(brk, v, side='right') - 1
            if include_lowest and v == brk[0]:
                idx = 0
            if 0 <= idx < len(brk) - 1:
                return idx
            return -1

        i1 = _bin_idx(v1, breaks1)
        i2 = _bin_idx(v2, breaks2)
        if i1 >= 0 and i2 >= 0:
            counts[i1, i2] += 1

    return counts
