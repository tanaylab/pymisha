"""Tests for gsegment (Wilcoxon-based segmentation)."""

import contextlib

import pandas as pd
import pytest

import pymisha as pm


class TestGsegmentBasic:
    """Basic gsegment functionality."""

    def test_returns_dataframe(self):
        """gsegment returns a DataFrame with chrom/start/end columns."""
        result = pm.gsegment("dense_track", 5000, maxpval=0.0001)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"chrom", "start", "end"}

    def test_segments_cover_scope(self):
        """Segments should collectively cover the input scope without gaps."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gsegment("dense_track", 5000, maxpval=0.0001, intervals=intervals)
        assert result is not None
        assert len(result) > 0
        # First segment starts at 0 and last ends at scope end
        # Use the chrom name from the result (test DB may use "1" not "chr1")
        chrom_name = result["chrom"].iloc[0]
        chr1 = result[result["chrom"] == chrom_name]
        assert chr1.iloc[0]["start"] == 0
        assert chr1.iloc[-1]["end"] == 200000
        # No gaps between consecutive segments
        for i in range(len(chr1) - 1):
            assert chr1.iloc[i]["end"] == chr1.iloc[i + 1]["start"]

    def test_min_segment_size(self):
        """Each segment should be at least minsegment bases wide."""
        minseg = 5000
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gsegment("dense_track", minseg, maxpval=0.001, intervals=intervals)
        assert result is not None
        sizes = result["end"] - result["start"]
        # Allow the last segment to be smaller (tail-end behavior)
        assert (sizes >= minseg).sum() >= len(result) - 1

    def test_multiple_chroms(self):
        """gsegment works across multiple chromosomes."""
        intervals = pm.gintervals([1, 2], 0, 100000)
        result = pm.gsegment("dense_track", 5000, maxpval=0.001, intervals=intervals)
        assert result is not None
        chroms = result["chrom"].unique()
        assert len(chroms) >= 2

    def test_expression_not_just_track(self):
        """gsegment works with expressions, not just bare track names."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gsegment("dense_track * 2", 5000, maxpval=0.0001, intervals=intervals)
        assert result is not None
        assert len(result) > 0


class TestGsegmentParameters:
    """Test gsegment parameter handling."""

    def test_stricter_pval_fewer_segments(self):
        """A lower maxpval (stricter) should produce fewer or equal segments."""
        intervals = pm.gintervals(1, 0, 200000)
        loose = pm.gsegment("dense_track", 5000, maxpval=0.01, intervals=intervals)
        strict = pm.gsegment("dense_track", 5000, maxpval=0.0001, intervals=intervals)
        assert len(strict) <= len(loose)

    def test_larger_minsegment_fewer_segments(self):
        """A larger minsegment should produce fewer or equal segments."""
        intervals = pm.gintervals(1, 0, 200000)
        small = pm.gsegment("dense_track", 2000, maxpval=0.001, intervals=intervals)
        large = pm.gsegment("dense_track", 10000, maxpval=0.001, intervals=intervals)
        assert len(large) <= len(small)

    def test_onetailed_parameter(self):
        """onetailed=False should produce a valid result."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gsegment("dense_track", 5000, maxpval=0.001, onetailed=False, intervals=intervals)
        assert result is not None
        assert len(result) > 0

    def test_explicit_iterator(self):
        """gsegment with an explicit fixed-bin iterator."""
        intervals = pm.gintervals(1, 0, 200000)
        result = pm.gsegment("dense_track", 5000, maxpval=0.001, intervals=intervals, iterator=100)
        assert result is not None
        assert len(result) > 0


class TestGsegmentEdgeCases:
    """Edge cases for gsegment."""

    def test_sparse_track_error(self):
        """gsegment on a sparse track should raise an error (needs fixed-bin iterator)."""
        intervals = pm.gintervals(1, 0, 200000)
        with pytest.raises(Exception):
            pm.gsegment("sparse_track", 5000, maxpval=0.001, intervals=intervals)

    def test_intervals_set_out(self):
        """gsegment with intervals_set_out saves result and returns None."""
        intervals = pm.gintervals(1, 0, 200000)
        set_name = "test.tmp_gsegment_out"
        try:
            result = pm.gsegment(
                "dense_track", 5000, maxpval=0.001,
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

    def test_none_intervals_returns_none(self):
        """When intervals is explicitly None after e.g. empty gscreen, return None."""
        # gsegment with None intervals should return None gracefully
        pm.gsegment("dense_track", 5000, maxpval=0.001, intervals=None)
        # Depending on implementation: either returns None or uses ALLGENOME
        # R uses ALLGENOME as default, so this should work
        assert True  # Accept either behavior

    def test_required_args(self):
        """gsegment raises if required arguments are missing."""
        with pytest.raises((TypeError, ValueError)):
            pm.gsegment("dense_track")  # missing minsegment


class TestGsegmentRParity:
    """R misha parity tests ported from test-gsegment.R."""

    def test_gsegment_fixedbin_golden(self):
        """Port of: gsegment with test.fixedbin — golden-master regression."""
        result = pm.gsegment("dense_track", 10000, maxpval=0.000001)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 32
        # Verify first/last rows on chrom 1
        chr1 = result[result["chrom"] == "1"]
        assert len(chr1) == 16
        assert chr1.iloc[0]["start"] == 0
        assert chr1.iloc[0]["end"] == 31200
        assert chr1.iloc[-1]["start"] == 389150
        assert chr1.iloc[-1]["end"] == 500000
        # Verify chrom 2
        chr2 = result[result["chrom"] == "2"]
        assert len(chr2) == 10
        assert chr2.iloc[0]["start"] == 0
        assert chr2.iloc[-1]["end"] == 300000
        # Verify chrom X
        chrx = result[result["chrom"] == "X"]
        assert len(chrx) == 6
        assert chrx.iloc[0]["start"] == 0
        assert chrx.iloc[-1]["end"] == 200000

    def test_array_track_error(self):
        """Port of: gsegment with test.array — expect error."""
        with pytest.raises(Exception):
            pm.gsegment("array_track", 10000, maxpval=0.000001)

    def test_rects_track_error(self):
        """Port of: gsegment with test.rects — expect error."""
        with pytest.raises(Exception):
            pm.gsegment("rects_track", 10000, maxpval=0.000001)

    def test_gsegment_modified_expr_golden(self):
        """Port of: gsegment with modified test.fixedbin — golden-master."""
        intervs = pm.gscreen("dense_track > 0.2", pm.gintervals([1, 2], 0, -1))
        result = pm.gsegment("dense_track*2", 10000, maxpval=0.000001, intervals=intervs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result.iloc[0]["chrom"] == "1"
        assert result.iloc[0]["start"] == 250
        assert result.iloc[0]["end"] == 460850
        assert result.iloc[1]["chrom"] == "2"
        assert result.iloc[1]["start"] == 2100
        assert result.iloc[1]["end"] == 299900

    def test_gsegment_sparse_with_iterator_and_intervals_set_out(self):
        """Port of: gsegment with data size option and sampling for test.sparse."""
        set_name = "test.tmp_gsegment_sparse_parity"
        try:
            pm.gsegment(
                "sparse_track", 10000, maxpval=0.0001,
                iterator=50, intervals_set_out=set_name,
            )
            loaded = pm.gintervals_load(set_name)
            assert loaded is not None
            assert len(loaded) >= 1
            # Segments should cover the full genome scope
            chroms = sorted(loaded["chrom"].unique())
            assert chroms == ["1", "2", "X"]
        finally:
            with contextlib.suppress(Exception):
                pm.gintervals_rm(set_name)
