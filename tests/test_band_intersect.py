"""Tests for gintervals_2d_band_intersect."""

import contextlib

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestGintervals2dBandIntersect:
    """Tests for gintervals_2d_band_intersect."""

    def _make_2d(self, c1, s1, e1, c2, s2, e2):
        return pd.DataFrame({
            'chrom1': c1, 'start1': s1, 'end1': e1,
            'chrom2': c2, 'start2': s2, 'end2': e2,
        })

    def test_basic_cis_intersection(self):
        """Band intersect keeps intervals within diagonal band.

        Interval (300,100)-(400,200): x-y ranges from 300-200=100 to 400-100=300.
        Band (50, 500) captures this.
        """
        intervals = self._make_2d(["1"], [300], [400], ["1"], [100], [200])
        result = pm.gintervals_2d_band_intersect(intervals, band=(50, 500))
        assert len(result) == 1

    def test_band_excludes_out_of_range(self):
        """Intervals outside band are removed.

        Interval (300,100)-(400,200): x-y in [100, 300].
        Band (500, 1000) should exclude.
        """
        intervals = self._make_2d(["1"], [300], [400], ["1"], [100], [200])
        result = pm.gintervals_2d_band_intersect(intervals, band=(500, 1000))
        assert len(result) == 0

    def test_trans_intervals_removed(self):
        """Trans (different chromosome) intervals are removed by band intersect."""
        intervals = self._make_2d(
            ["1", "1"], [300, 300], [400, 400],
            ["1", "2"], [100, 100], [200, 200],
        )
        result = pm.gintervals_2d_band_intersect(intervals, band=(0, 1000))
        assert len(result) == 1
        assert result.iloc[0]['chrom2'] == '1'

    def test_band_shrinks_rectangle(self):
        """Band intersect shrinks intervals to the band region."""
        # Large square (0,0)-(1000,1000): x-y ranges from -1000 to 1000
        intervals = self._make_2d(["1"], [0], [1000], ["1"], [0], [1000])
        # Band (100, 200)
        result = pm.gintervals_2d_band_intersect(intervals, band=(100, 200))
        assert len(result) == 1
        r = result.iloc[0]
        # After shrinking, the rectangle should be contained within the band
        # x1 - y1 should be >= d1 or adjusted
        assert r['start1'] >= 0
        assert r['end1'] <= 1000
        assert r['start2'] >= 0
        assert r['end2'] <= 1000
        # The shrunk rectangle should be tighter
        assert r['start1'] > 0 or r['end1'] < 1000 or r['start2'] > 0 or r['end2'] < 1000

    def test_contained_interval_not_shrunk(self):
        """Interval fully within band is not modified.

        Interval (200,100)-(210,105): x-y in [95, 110].
        Band (50, 200) fully contains it.
        """
        intervals = self._make_2d(["1"], [200], [210], ["1"], [100], [105])
        result = pm.gintervals_2d_band_intersect(intervals, band=(50, 200))
        assert len(result) == 1
        r = result.iloc[0]
        assert r['start1'] == 200
        assert r['end1'] == 210
        assert r['start2'] == 100
        assert r['end2'] == 105

    def test_negative_band(self):
        """Band with negative distances works (below diagonal)."""
        # (100,300)-(200,400): x-y ranges from 100-400=-300 to 200-300=-100
        intervals = self._make_2d(["1"], [100], [200], ["1"], [300], [400])
        result = pm.gintervals_2d_band_intersect(intervals, band=(-400, -50))
        assert len(result) == 1

    def test_empty_intervals(self):
        """Empty input returns empty DataFrame with correct columns."""
        intervals = self._make_2d([], [], [], [], [], [])
        result = pm.gintervals_2d_band_intersect(intervals, band=(0, 100))
        assert len(result) == 0
        assert 'chrom1' in result.columns

    def test_band_must_be_length_2(self):
        """Band must be a pair (d1, d2)."""
        intervals = self._make_2d(["1"], [0], [100], ["1"], [0], [100])
        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_2d_band_intersect(intervals, band=(100,))

    def test_band_d1_less_than_d2(self):
        """Band d1 must be < d2."""
        intervals = self._make_2d(["1"], [0], [100], ["1"], [0], [100])
        with pytest.raises(ValueError):
            pm.gintervals_2d_band_intersect(intervals, band=(200, 100))

    def test_preserves_extra_columns(self):
        """Extra columns in input are preserved."""
        df = self._make_2d(["1"], [300], [400], ["1"], [100], [200])
        df['value'] = 42.0
        result = pm.gintervals_2d_band_intersect(df, band=(0, 1000))
        assert 'value' in result.columns
        assert result.iloc[0]['value'] == 42.0

    def test_multiple_intervals_mixed(self):
        """Some intervals kept, some removed."""
        intervals = self._make_2d(
            ["1", "1", "1"],
            [150, 600, 3000],
            [250, 700, 4000],
            ["1", "1", "1"],
            [0, 0, 0],
            [100, 100, 100],
        )
        # x-y ranges: [50,250], [500,700], [2900,4000]
        # Band (0, 300): first fits, second and third don't
        result = pm.gintervals_2d_band_intersect(intervals, band=(0, 300))
        assert len(result) == 1

    def test_intervals_set_out(self, tmp_path):
        """intervals_set_out writes result to disk and returns None."""
        intervals = self._make_2d(["1"], [300], [400], ["1"], [100], [200])
        set_name = "test.band_intersect_out"
        try:
            result = pm.gintervals_2d_band_intersect(
                intervals, band=(0, 1000), intervals_set_out=set_name
            )
            assert result is None
            loaded = pm.gintervals_load(set_name)
            assert len(loaded) == 1
        finally:
            with contextlib.suppress(Exception):
                pm.gintervals_rm(set_name)

    def test_shrink_coordinates_correct(self):
        """Verify shrink produces correct coordinates matching R DiagonalBand logic.

        Interval (0,0)-(100,100): x-y in [-100, 100].
        Band (20, 50):
        - Top-left (x1=0, y1=0): x1-y1=0 < d1=20, so x1 = y1 + d1 = 20
        - Bottom-right (x2=100, y2=100): x2-y2=0 < d1=20, so y2 = x2 - d1 = 80
        """
        intervals = self._make_2d(["1"], [0], [100], ["1"], [0], [100])
        result = pm.gintervals_2d_band_intersect(intervals, band=(20, 50))
        assert len(result) == 1
        r = result.iloc[0]
        assert r['start1'] == 20   # x1 adjusted
        assert r['start2'] == 0    # y1 unchanged
        assert r['end1'] == 100    # x2 unchanged
        assert r['end2'] == 80     # y2 adjusted
