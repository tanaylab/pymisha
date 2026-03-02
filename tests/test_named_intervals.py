"""Tests for transparent loading of named interval sets (including bigsets).

When a function receives a string interval set name, it should transparently
load the intervals via gintervals_load, so users don't need to call
gintervals_load manually.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _init_db():
    """Ensure the test database is initialized."""
    pm.gdb_init_examples()


@pytest.fixture()
def named_intervals():
    """Create a named interval set for testing and clean up afterwards."""
    name = "test_named_iset"
    intervals = pm.gintervals(["1", "2"], [0, 0], [5000, 5000])
    pm.gintervals_save(intervals, name)
    yield name, intervals
    pm.gintervals_rm(name, force=True)


@pytest.fixture()
def named_intervals_small():
    """Create a small named interval set for faster tests."""
    name = "test_named_small"
    intervals = pm.gintervals("1", 0, 1000)
    pm.gintervals_save(intervals, name)
    yield name, intervals
    pm.gintervals_rm(name, force=True)


class TestGextract:
    """gextract should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gextract("dense_track", intervals=name, iterator=200)
        result_df = pm.gextract("dense_track", intervals=intervals, iterator=200)
        assert result_name is not None
        assert len(result_name) == len(result_df)
        pd.testing.assert_frame_equal(result_name, result_df)


class TestGsummary:
    """gsummary should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gsummary("dense_track", intervals=name)
        result_df = pm.gsummary("dense_track", intervals=intervals)
        pd.testing.assert_series_equal(result_name, result_df)


class TestGquantiles:
    """gquantiles should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gquantiles("dense_track", intervals=name)
        result_df = pm.gquantiles("dense_track", intervals=intervals)
        pd.testing.assert_series_equal(result_name, result_df)


class TestGdist:
    """gdist should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        breaks = [0.0, 0.1, 0.2, 0.3, 0.5]
        result_name = pm.gdist("dense_track", breaks, intervals=name)
        result_df = pm.gdist("dense_track", breaks, intervals=intervals)
        np.testing.assert_array_equal(result_name, result_df)


class TestGscreen:
    """gscreen should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gscreen("dense_track > 0.2", intervals=name)
        result_df = pm.gscreen("dense_track > 0.2", intervals=intervals)
        if result_name is not None and result_df is not None:
            assert len(result_name) == len(result_df)
        else:
            assert result_name is None and result_df is None


class TestGintervalsSummary:
    """gintervals_summary should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gintervals_summary("dense_track", intervals=name)
        result_df = pm.gintervals_summary("dense_track", intervals=intervals)
        assert result_name is not None
        assert result_df is not None
        assert len(result_name) == len(result_df)
        pd.testing.assert_frame_equal(result_name, result_df)


class TestGintervalsQuantiles:
    """gintervals_quantiles should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gintervals_quantiles(
            "dense_track", percentiles=[0.25, 0.5, 0.75], intervals=name
        )
        result_df = pm.gintervals_quantiles(
            "dense_track", percentiles=[0.25, 0.5, 0.75], intervals=intervals
        )
        assert result_name is not None
        assert result_df is not None
        pd.testing.assert_frame_equal(result_name, result_df)


class TestGpartition:
    """gpartition should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gpartition(
            "dense_track", [0.0, 0.1, 0.2, 0.3], intervals=name
        )
        result_df = pm.gpartition(
            "dense_track", [0.0, 0.1, 0.2, 0.3], intervals=intervals
        )
        if result_name is not None and result_df is not None:
            assert len(result_name) == len(result_df)
        else:
            assert result_name is None and result_df is None


class TestGsample:
    """gsample should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, _intervals = named_intervals
        result = pm.gsample("dense_track", 50, intervals=name)
        assert len(result) == 50


class TestGcor:
    """gcor should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gcor(
            "dense_track", "sparse_track", intervals=name, iterator=500
        )
        result_df = pm.gcor(
            "dense_track", "sparse_track", intervals=intervals, iterator=500
        )
        assert result_name is not None
        np.testing.assert_allclose(result_name, result_df, atol=1e-10)


class TestGbinsSummary:
    """gbins_summary should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        breaks = [0.0, 0.1, 0.2, 0.3, 0.5]
        result_name = pm.gbins_summary(
            "dense_track", breaks, intervals=name, iterator=500
        )
        result_df = pm.gbins_summary(
            "dense_track", breaks, intervals=intervals, iterator=500
        )
        np.testing.assert_array_equal(result_name, result_df)


class TestGbinsQuantiles:
    """gbins_quantiles should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        breaks = [0.0, 0.1, 0.2, 0.3, 0.5]
        result_name = pm.gbins_quantiles(
            "dense_track", breaks, intervals=name, iterator=500
        )
        result_df = pm.gbins_quantiles(
            "dense_track", breaks, intervals=intervals, iterator=500
        )
        np.testing.assert_array_almost_equal(result_name, result_df)


class TestGlookup:
    """glookup should accept named interval set strings."""

    def test_string_intervals(self, named_intervals_small):
        name, intervals = named_intervals_small
        breaks = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
        table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result_name = pm.glookup(table, "dense_track", breaks, intervals=name)
        result_df = pm.glookup(table, "dense_track", breaks, intervals=intervals)
        if result_name is not None and result_df is not None:
            pd.testing.assert_frame_equal(result_name, result_df)
        else:
            assert result_name is None and result_df is None


class TestGsegment:
    """gsegment should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gsegment("dense_track", 1000, maxpval=0.1, intervals=name)
        result_df = pm.gsegment("dense_track", 1000, maxpval=0.1, intervals=intervals)
        if result_name is not None and result_df is not None:
            assert len(result_name) == len(result_df)
        else:
            assert result_name is None and result_df is None


class TestGwilcox:
    """gwilcox should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gwilcox(
            "dense_track", 2000, 500, maxpval=0.1, intervals=name
        )
        result_df = pm.gwilcox(
            "dense_track", 2000, 500, maxpval=0.1, intervals=intervals
        )
        if result_name is not None and result_df is not None:
            assert len(result_name) == len(result_df)
        else:
            assert result_name is None and result_df is None


class TestGseqExtract:
    """gseq_extract should accept named interval set strings."""

    def test_string_intervals(self, named_intervals_small):
        name, intervals = named_intervals_small
        result_name = pm.gseq_extract(name)
        result_df = pm.gseq_extract(intervals)
        assert result_name == result_df


class TestGseqKmerDist:
    """gseq_kmer_dist should accept named interval set strings."""

    def test_string_intervals(self, named_intervals_small):
        name, intervals = named_intervals_small
        result_name = pm.gseq_kmer_dist(name, k=2)
        result_df = pm.gseq_kmer_dist(intervals, k=2)
        # Sort both for comparison since order might differ
        result_name = result_name.sort_values("kmer").reset_index(drop=True)
        result_df = result_df.sort_values("kmer").reset_index(drop=True)
        pd.testing.assert_frame_equal(result_name, result_df)


class TestGiteratorIntervals:
    """giterator_intervals should accept named interval set strings (already works)."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.giterator_intervals(intervals=name, iterator=500)
        result_df = pm.giterator_intervals(intervals=intervals, iterator=500)
        assert result_name is not None
        assert result_df is not None
        assert len(result_name) == len(result_df)


class TestGintervalsMapply:
    """gintervals_mapply should accept named interval set strings."""

    def test_string_intervals(self, named_intervals):
        name, intervals = named_intervals
        result_name = pm.gintervals_mapply(
            np.mean, "dense_track", intervals=name, iterator=500
        )
        result_df = pm.gintervals_mapply(
            np.mean, "dense_track", intervals=intervals, iterator=500
        )
        assert result_name is not None
        assert result_df is not None
        assert len(result_name) == len(result_df)


class TestMaybeLoadIntervalsSet:
    """Test _maybe_load_intervals_set helper directly."""

    def test_nonstring_passthrough(self):
        """Non-string arguments should be returned unchanged."""
        from pymisha.extract import _maybe_load_intervals_set

        df = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
        assert _maybe_load_intervals_set(df) is df
        assert _maybe_load_intervals_set(None) is None

    def test_existing_named_set(self, named_intervals):
        """Known named interval set should be loaded."""
        from pymisha.extract import _maybe_load_intervals_set

        name, _intervals = named_intervals
        result = _maybe_load_intervals_set(name)
        assert isinstance(result, pd.DataFrame)
        assert "chrom" in result.columns

    def test_nonexistent_name_passthrough(self):
        """Non-existent named set should be returned as-is (string)."""
        from pymisha.extract import _maybe_load_intervals_set

        result = _maybe_load_intervals_set("no_such_intervals_set_xyz")
        assert result == "no_such_intervals_set_xyz"
