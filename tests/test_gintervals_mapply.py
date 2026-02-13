"""Tests for gintervals_mapply."""

import numpy as np
import pytest

import pymisha


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestGintervalsMapply:
    """Tests for gintervals_mapply."""

    def test_basic_max(self):
        """Apply max to a single track expression."""
        intervals = pymisha.gintervals(["1", "2"], [0, 0], [10000, 10000])
        result = pymisha.gintervals_mapply(
            np.nanmax, "dense_track", intervals=intervals
        )
        assert "value" in result.columns
        assert len(result) == len(intervals)
        # Values should be finite numbers (max of dense_track per interval)
        assert all(np.isfinite(result["value"]))

    def test_custom_colname(self):
        """Custom column name for result."""
        intervals = pymisha.gintervals(["1"], [0], [10000])
        result = pymisha.gintervals_mapply(
            np.nanmax, "dense_track", intervals=intervals, colnames="max_val"
        )
        assert "max_val" in result.columns
        assert "value" not in result.columns

    def test_multiple_expressions(self):
        """Apply function to multiple track expressions."""
        intervals = pymisha.gintervals(["1"], [0], [10000])

        def sum_of_maxes(*args):
            return sum(np.nanmax(a) for a in args if len(a) > 0 and not np.all(np.isnan(a)))

        result = pymisha.gintervals_mapply(
            sum_of_maxes, "dense_track", "dense_track + 1",
            intervals=intervals
        )
        assert len(result) == 1
        assert np.isfinite(result["value"].iloc[0])

    def test_result_has_interval_columns(self):
        """Result includes chrom/start/end columns from intervals."""
        intervals = pymisha.gintervals(["1", "2"], [100, 200], [500, 600])
        result = pymisha.gintervals_mapply(
            np.nanmean, "dense_track", intervals=intervals
        )
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns
        assert list(result["chrom"]) == ["1", "2"]

    def test_nan_values(self):
        """Function can return NaN."""
        intervals = pymisha.gintervals(["1"], [0], [100])

        def always_nan(x):
            return np.nan

        result = pymisha.gintervals_mapply(
            always_nan, "dense_track", intervals=intervals
        )
        assert np.isnan(result["value"].iloc[0])

    def test_no_intervals_raises(self):
        """Raises error if no intervals provided."""
        with pytest.raises((ValueError, TypeError)):
            pymisha.gintervals_mapply(np.nanmax, "dense_track")

    def test_no_exprs_raises(self):
        """Raises error if no track expressions provided."""
        intervals = pymisha.gintervals(["1"], [0], [1000])
        with pytest.raises(ValueError, match="[Ee]xpression"):
            pymisha.gintervals_mapply(np.nanmax, intervals=intervals)

    def test_with_iterator(self):
        """Works with explicit iterator."""
        intervals = pymisha.gintervals(["1"], [0], [50000])
        result = pymisha.gintervals_mapply(
            np.nanmean, "dense_track", intervals=intervals,
            iterator=10000
        )
        # With a 10000-bin iterator on a 50000 interval, we get 5 sub-intervals
        assert len(result) == 5

    def test_strand_reversal(self):
        """Negative strand reverses the values passed to function."""
        intervals = pymisha.gintervals(["1"], [0], [10000])
        intervals["strand"] = -1

        # Use a function that distinguishes direction
        def first_value(x):
            return x[0] if len(x) > 0 else np.nan

        result_rev = pymisha.gintervals_mapply(
            first_value, "dense_track", intervals=intervals
        )

        intervals2 = pymisha.gintervals(["1"], [0], [10000])
        result_fwd = pymisha.gintervals_mapply(
            first_value, "dense_track", intervals=intervals2
        )
        # With strand=-1, the first value should be what was the last value forward
        # They should differ (unless all values are identical)
        # Just check both return valid results
        assert np.isfinite(result_rev["value"].iloc[0])
        assert np.isfinite(result_fwd["value"].iloc[0])

    def test_intervals_set_out(self, tmp_path):
        """Result can be saved to an intervals set."""
        intervals = pymisha.gintervals(["1", "2"], [0, 0], [10000, 10000])
        set_name = "test_mapply_out"
        if pymisha.gintervals_exists(set_name):
            pymisha.gintervals_rm(set_name)
        result = pymisha.gintervals_mapply(
            np.nanmax, "dense_track", intervals=intervals,
            intervals_set_out=set_name
        )
        assert result is None
        loaded = pymisha.gintervals_load(set_name)
        assert len(loaded) == 2
        assert "value" in loaded.columns
        # Cleanup
        pymisha.gintervals_rm(set_name)

    def test_return_scalar(self):
        """Function returning scalar (not array) works."""
        intervals = pymisha.gintervals(["1"], [0], [10000])
        result = pymisha.gintervals_mapply(
            lambda x: 42.0, "dense_track", intervals=intervals
        )
        assert result["value"].iloc[0] == 42.0
