"""Tests targeting uncovered C++ code paths identified by gcov analysis.

These tests exercise C++ code that was previously unreachable from the test suite,
focusing on:
- PWMScorer C++ class (spatial weighting via vtracks)
- masked.count / masked.frac virtual tracks
- pm_seed / pm_dbgetdatasets direct C++ calls
- Sparse-source vtrack aggregation paths
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha._shared import _pymisha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pssm(width: int = 6) -> np.ndarray:
    """Create a simple PSSM (uniform with one strong position)."""
    pssm = np.full((width, 4), 0.25)
    pssm[0] = [0.9, 0.03, 0.04, 0.03]  # strong A at position 0
    pssm[1] = [0.03, 0.9, 0.04, 0.03]  # strong C at position 1
    return pssm


def _extract_values(df: pd.DataFrame) -> np.ndarray:
    """Extract the single numeric result column from gextract output."""
    data_cols = [c for c in df.columns if c not in ("chrom", "start", "end", "intervalID")]
    assert len(data_cols) == 1, f"Expected 1 data column, got {data_cols}"
    return df[data_cols[0]].to_numpy()


# ===========================================================================
# 1. PWMScorer C++ via vtracks with spatial weighting
#    Exercises: PWMScorer constructor (spat_factor path), invalidate_cache,
#    count_motif_hits_no_spatial, count_motif_hits_with_spatial,
#    get_max_likelihood_pos_with_spatial, pos_value_with_dir, spat_log_at
# ===========================================================================

class TestPWMScorerSpatialVtrack:
    """PWM virtual tracks with spatial weighting — hits the C++ PWMScorer class."""

    @pytest.fixture(autouse=True)
    def _clear_vtracks(self):
        pm.gvtrack_clear()
        yield
        pm.gvtrack_clear()

    def test_pwm_lse_spatial_vtrack(self):
        """func='pwm' with spat_factor — exercises PWMScorer constructor + spatial LSE."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 1000)
        spat = [0.5, 1.0, 2.0, 1.0, 0.5]

        pm.gvtrack_create(
            "vt_pwm_spat", None, func="pwm",
            pssm=pssm, spat_factor=spat, spat_bin=20,
        )
        result = pm.gextract("vt_pwm_spat", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))

    def test_pwm_max_spatial_vtrack(self):
        """func='pwm.max' with spat_factor — spatial MAX mode in C++."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 1000)
        spat = [1.0, 2.0, 1.5]

        pm.gvtrack_create(
            "vt_pwm_max_spat", None, func="pwm.max",
            pssm=pssm, spat_factor=spat, spat_bin=50,
        )
        result = pm.gextract("vt_pwm_max_spat", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))

    def test_pwm_count_spatial_vtrack(self):
        """func='pwm.count' with spat_factor — spatial COUNT mode in C++."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 1000)
        spat = [0.5, 1.0, 2.0]

        pm.gvtrack_create(
            "vt_pwm_cnt_spat", None, func="pwm.count",
            pssm=pssm, score_thresh=-5.0,
            spat_factor=spat, spat_bin=30,
        )
        result = pm.gextract("vt_pwm_cnt_spat", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0
        # count should be non-negative integers (as float)
        assert np.all(vals >= 0)

    def test_pwm_pos_spatial_vtrack(self):
        """func='pwm.max.pos' with spat_factor — spatial POS mode in C++."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 1000)
        spat = [1.0, 1.5, 2.0, 1.5, 1.0]

        pm.gvtrack_create(
            "vt_pwm_pos_spat", None, func="pwm.max.pos",
            pssm=pssm, spat_factor=spat, spat_bin=20,
        )
        result = pm.gextract("vt_pwm_pos_spat", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0

    def test_pwm_count_no_spatial_vtrack(self):
        """func='pwm.count' without spatial — exercises count_motif_hits_no_spatial."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create(
            "vt_pwm_cnt", None, func="pwm.count",
            pssm=pssm, score_thresh=-5.0,
        )
        result = pm.gextract("vt_pwm_cnt", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0
        assert np.all(vals >= 0)

    def test_pwm_pos_no_spatial_vtrack(self):
        """func='pwm.max.pos' without spatial — exercises get_max_likelihood_pos."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create(
            "vt_pwm_pos", None, func="pwm.max.pos",
            pssm=pssm,
        )
        result = pm.gextract("vt_pwm_pos", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0

    def test_pwm_spatial_different_strandedness(self):
        """PWM vtrack with spatial + strand=1 (forward only)."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 500)
        spat = [1.0, 2.0, 0.5]

        pm.gvtrack_create(
            "vt_pwm_fwd_spat", None, func="pwm",
            pssm=pssm, strand=1,
            spat_factor=spat, spat_bin=30,
        )
        result = pm.gextract("vt_pwm_fwd_spat", intervals, iterator=100)
        vals = _extract_values(result)
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))

    def test_pwm_spatial_large_bin(self):
        """PWM spatial with large bin size — tests bin clamping at boundaries."""
        pssm = _make_pssm()
        intervals = pm.gintervals("1", 0, 500)
        # Only 2 spatial bins, large bin size means most positions map to bin 0 or 1
        spat = [0.5, 2.0]

        pm.gvtrack_create(
            "vt_pwm_bigbin", None, func="pwm",
            pssm=pssm, spat_factor=spat, spat_bin=200,
        )
        result = pm.gextract("vt_pwm_bigbin", intervals, iterator=500)
        vals = _extract_values(result)
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))


# ===========================================================================
# 2. masked.count / masked.frac virtual tracks
#    Exercises: MaskedBpCounter code path in PMVTrack.cpp
# ===========================================================================

class TestMaskedVtracks:
    """masked.count and masked.frac vtracks — previously untested C++ paths."""

    @pytest.fixture(autouse=True)
    def _clear_vtracks(self):
        pm.gvtrack_clear()
        yield
        pm.gvtrack_clear()

    def test_masked_count_vtrack(self):
        """func='masked.count' counts masked (lowercase/N) bases in sequence."""
        intervals = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_mask_cnt", None, func="masked.count")
        result = pm.gextract("vt_mask_cnt", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0
        # Count should be non-negative
        assert np.all(vals >= 0)

    def test_masked_frac_vtrack(self):
        """func='masked.frac' — fraction of masked bases in each window."""
        intervals = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_mask_frac", None, func="masked.frac")
        result = pm.gextract("vt_mask_frac", intervals, iterator=200)
        vals = _extract_values(result)
        assert len(vals) > 0
        # Fraction should be in [0, 1]
        assert np.all(vals >= 0)
        assert np.all(vals <= 1)


# ===========================================================================
# 3. pm_seed and pm_dbgetdatasets — direct C++ entry points
#    Exercises: pm_seed, pm_dbgetdatasets in PMStubs.cpp
# ===========================================================================

class TestDirectCppCalls:
    """Direct calls to C++ functions that are otherwise uncovered."""

    def test_pm_seed(self):
        """pm_seed sets the C++ random number generator seed."""
        _pymisha.pm_seed(60427)
        # No return value — just verify it doesn't crash
        _pymisha.pm_seed(0)
        _pymisha.pm_seed(2**31 - 1)

    def test_pm_dbgetdatasets(self):
        """pm_dbgetdatasets returns the list of loaded dataset roots."""
        result = _pymisha.pm_dbgetdatasets()
        assert isinstance(result, list)
        # With the test DB loaded, should return the dataset paths (may be empty)
        for item in result:
            assert isinstance(item, str)

    def test_pm_seed_affects_sampling(self):
        """Verify pm_seed actually affects randomness by checking gsample output."""
        intervals = pm.gintervals("1", 0, 100000)

        _pymisha.pm_seed(12345)
        result1 = pm.gsample("dense_track", 5, intervals)

        _pymisha.pm_seed(12345)
        result2 = pm.gsample("dense_track", 5, intervals)

        # Same seed should produce same sample
        np.testing.assert_array_equal(result1, result2)


# ===========================================================================
# 4. Sparse-source vtrack aggregation (intervals-as-source)
#    Exercises: the sparse interval enumeration path in PMVTrack.cpp
#    lines 1296-1325 (the sum/min/max/avg/exists/size branch)
# ===========================================================================

class TestSparseSourceVtrackAggregation:
    """Vtracks sourced from an intervals set with values — sparse aggregation path."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        pm.gvtrack_clear()
        yield
        pm.gvtrack_clear()

    def _make_source_intervals(self) -> pd.DataFrame:
        """Create a source DataFrame with chrom/start/end/value."""
        return pd.DataFrame({
            "chrom": ["1"] * 6,
            "start": [100, 200, 300, 500, 600, 700],
            "end":   [150, 250, 350, 550, 650, 750],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

    def test_avg_vtrack_from_intervals_source(self):
        """func='avg' with intervals source — sparse avg aggregation."""
        src = self._make_source_intervals()
        query = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_sp_avg", src, func="avg")
        result = pm.gextract("vt_sp_avg", query, iterator=500)
        vals = _extract_values(result)
        assert len(vals) > 0

    def test_sum_vtrack_from_intervals_source(self):
        """func='sum' with intervals source — sparse sum aggregation."""
        src = self._make_source_intervals()
        query = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_sp_sum", src, func="sum")
        result = pm.gextract("vt_sp_sum", query, iterator=500)
        vals = _extract_values(result)
        assert len(vals) > 0

    def test_min_max_vtrack_from_intervals_source(self):
        """func='min'/'max' with intervals source."""
        src = self._make_source_intervals()
        query = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_sp_min", src, func="min")
        pm.gvtrack_create("vt_sp_max", src, func="max")

        rmin = pm.gextract("vt_sp_min", query, iterator=500)
        rmax = pm.gextract("vt_sp_max", query, iterator=500)

        vmin = _extract_values(rmin)
        vmax = _extract_values(rmax)
        # max >= min where both are finite
        mask = np.isfinite(vmin) & np.isfinite(vmax)
        if mask.any():
            assert np.all(vmax[mask] >= vmin[mask])

    def test_exists_size_vtrack_from_intervals_source(self):
        """func='exists'/'size' with intervals source."""
        src = self._make_source_intervals()
        query = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_sp_exists", src, func="exists")
        pm.gvtrack_create("vt_sp_size", src, func="size")

        rexists = pm.gextract("vt_sp_exists", query, iterator=500)
        rsize = pm.gextract("vt_sp_size", query, iterator=500)

        vexists = _extract_values(rexists)
        vsize = _extract_values(rsize)

        # exists should be 0 or 1
        assert np.all((vexists == 0) | (vexists == 1))
        # size should be non-negative
        assert np.all(vsize >= 0)
        # where exists=1, size > 0
        assert np.all(vsize[vexists == 1] > 0)

    def test_first_last_vtrack_from_intervals_source(self):
        """func='first'/'last' with intervals source."""
        src = self._make_source_intervals()
        query = pm.gintervals("1", 0, 1000)

        pm.gvtrack_create("vt_sp_first", src, func="first")
        pm.gvtrack_create("vt_sp_last", src, func="last")

        rfirst = pm.gextract("vt_sp_first", query, iterator=500)
        rlast = pm.gextract("vt_sp_last", query, iterator=500)

        vfirst = _extract_values(rfirst)
        vlast = _extract_values(rlast)
        assert len(vfirst) > 0
        assert len(vlast) > 0
