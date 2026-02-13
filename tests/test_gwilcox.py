"""Tests for gwilcox (sliding-window Wilcoxon test)."""

import contextlib

import pandas as pd
import pytest

import pymisha as pm


class TestGwilcoxBasic:
    """Basic gwilcox functionality."""

    def test_returns_dataframe_with_pval(self):
        """gwilcox returns a DataFrame with chrom/start/end/pval columns."""
        intervals = pm.gintervals([1, 2], 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.001, intervals=intervals
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"chrom", "start", "end", "pval"}

    def test_pval_column_below_threshold(self):
        """All returned pval values should be at or below maxpval."""
        intervals = pm.gintervals([1, 2], 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.001, intervals=intervals
        )
        if result is not None and len(result) > 0:
            assert (result["pval"] <= 0.001 + 1e-10).all()

    def test_intervals_sorted(self):
        """Returned intervals should be sorted by chrom then start."""
        intervals = pm.gintervals([1, 2], 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.01, intervals=intervals
        )
        if result is not None and len(result) > 1:
            for chrom in result["chrom"].unique():
                sub = result[result["chrom"] == chrom]
                assert (sub["start"].diff().iloc[1:] >= 0).all()

    def test_no_overlapping_intervals(self):
        """Returned intervals should not overlap."""
        intervals = pm.gintervals([1, 2], 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.01, intervals=intervals
        )
        if result is not None and len(result) > 1:
            for chrom in result["chrom"].unique():
                sub = result[result["chrom"] == chrom].reset_index(drop=True)
                for i in range(len(sub) - 1):
                    assert sub.loc[i, "end"] <= sub.loc[i + 1, "start"]


class TestGwilcoxParameters:
    """Test gwilcox parameter handling."""

    def test_what2find_highs(self):
        """what2find=1 searches for peaks only."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.01,
            what2find=1, intervals=intervals,
        )
        # Should return valid result (possibly empty)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_what2find_lows(self):
        """what2find=-1 searches for lows only."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.01,
            what2find=-1, intervals=intervals,
        )
        assert result is None or isinstance(result, pd.DataFrame)

    def test_what2find_both(self):
        """what2find=0 searches for both peaks and lows."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.01,
            what2find=0, intervals=intervals,
        )
        assert result is None or isinstance(result, pd.DataFrame)

    def test_stricter_pval_fewer_results(self):
        """A stricter (lower) maxpval should give fewer or equal results."""
        intervals = pm.gintervals([1, 2], 0, 200000)
        loose = pm.gwilcox("dense_track", 100000, 1000, maxpval=0.01, intervals=intervals)
        strict = pm.gwilcox("dense_track", 100000, 1000, maxpval=0.0001, intervals=intervals)
        n_loose = len(loose) if loose is not None else 0
        n_strict = len(strict) if strict is not None else 0
        assert n_strict <= n_loose

    def test_onetailed_parameter(self):
        """onetailed=False should produce a valid result."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.01,
            onetailed=False, intervals=intervals,
        )
        assert result is None or isinstance(result, pd.DataFrame)

    def test_explicit_iterator(self):
        """gwilcox with an explicit fixed-bin iterator."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gwilcox(
            "dense_track", 50000, 1000, maxpval=0.01,
            intervals=intervals, iterator=100,
        )
        assert result is None or isinstance(result, pd.DataFrame)

    def test_swapped_winsizes(self):
        """gwilcox should work regardless of which winsize is larger."""
        intervals = pm.gintervals(1, 0, 200000)
        r1 = pm.gwilcox("dense_track", 100000, 1000, maxpval=0.01, intervals=intervals)
        r2 = pm.gwilcox("dense_track", 1000, 100000, maxpval=0.01, intervals=intervals)
        # Should produce same results (winsize order is internally sorted)
        n1 = len(r1) if r1 is not None else 0
        n2 = len(r2) if r2 is not None else 0
        assert n1 == n2


class TestGwilcoxEdgeCases:
    """Edge cases for gwilcox."""

    def test_sparse_track_error(self):
        """gwilcox on a sparse track should raise an error (needs fixed-bin iterator)."""
        intervals = pm.gintervals(1, 0, 200000)
        with pytest.raises(Exception):
            pm.gwilcox("sparse_track", 100000, 1000, maxpval=0.01, intervals=intervals)

    def test_intervals_set_out(self):
        """gwilcox with intervals_set_out saves result and returns None."""
        intervals = pm.gintervals([1, 2], 0, 200000)
        set_name = "test.tmp_gwilcox_out"
        try:
            result = pm.gwilcox(
                "dense_track", 100000, 1000, maxpval=0.01,
                intervals=intervals,
                intervals_set_out=set_name,
            )
            assert result is None
            loaded = pm.gintervals_load(set_name)
            assert loaded is not None
            assert len(loaded) > 0
        finally:
            with contextlib.suppress(Exception):
                pm.gintervals_rm(set_name)

    def test_required_args(self):
        """gwilcox raises if required arguments are missing."""
        with pytest.raises((TypeError, ValueError)):
            pm.gwilcox("dense_track", 100000)  # missing winsize2

    def test_empty_intervals_returns_none(self):
        """gwilcox with empty intervals returns None."""
        empty_df = pd.DataFrame({"chrom": [], "start": [], "end": []})
        result = pm.gwilcox("dense_track", 100000, 1000, maxpval=0.01, intervals=empty_df)
        assert result is None


class TestGwilcoxRParity:
    """R misha parity tests ported from test-gwilcox.R."""

    def test_gwilcox_fixedbin_golden(self):
        """Port of: gwilcox on test.fixedbin — golden-master regression."""
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.000001,
            intervals=pm.gintervals([1, 2], 0, -1),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 16
        assert set(result.columns) >= {"chrom", "start", "end", "pval"}
        # Verify first and last rows
        assert result.iloc[0]["chrom"] == "1"
        assert result.iloc[0]["start"] == 32300
        assert result.iloc[0]["end"] == 35300
        assert result.iloc[-1]["chrom"] == "2"
        assert result.iloc[-1]["start"] == 288650
        assert result.iloc[-1]["end"] == 290500
        # All pvals below threshold
        assert (result["pval"] <= 0.000001 + 1e-10).all()

    def test_array_track_error(self):
        """Port of: gwilcox on test.array — expect error."""
        with pytest.raises(Exception):
            pm.gwilcox(
                "array_track", 100000, 1000, maxpval=0.000001,
                intervals=pm.gintervals([1, 2], 0, -1),
            )

    def test_rects_track_error(self):
        """Port of: gwilcox on test.rects — expect error."""
        with pytest.raises(Exception):
            pm.gwilcox(
                "rects_track", 100000, 1000, maxpval=0.000001,
                intervals=pm.gintervals([1, 2], 0, -1),
            )

    def test_gwilcox_screening_golden(self):
        """Port of: gwilcox on test.fixedbin with screening — golden-master."""
        intervs = pm.gscreen("dense_track < 0.2", pm.gintervals([1, 2], 0, -1))
        result = pm.gwilcox(
            "dense_track", 100000, 1000, maxpval=0.0001, intervals=intervs,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["chrom"] == "1"
        assert result.iloc[0]["start"] == 3150
        assert result.iloc[0]["end"] == 4550

    def test_gwilcox_intervals_set_out_golden(self):
        """Port of: gwilcox with interval setting and max data size — golden-master."""
        set_name = "test.tmp_gwilcox_parity"
        try:
            pm.gwilcox(
                "dense_track", 100000, 1000, maxpval=0.000001,
                intervals=pm.gintervals([1, 2], 0, -1),
                intervals_set_out=set_name,
            )
            loaded = pm.gintervals_load(set_name)
            assert len(loaded) == 16
            assert loaded.iloc[0]["start"] == 32300
            assert loaded.iloc[-1]["end"] == 290500
        finally:
            with contextlib.suppress(Exception):
                pm.gintervals_rm(set_name)
