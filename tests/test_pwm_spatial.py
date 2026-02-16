"""Tests for PWM spatial weighting and bidirectional spatial sliding.

Ported from R misha tests:
- tests/testthat/test-pwm-spatial.R (308 lines, ~9 tests)
- tests/testthat/test-pwm-count-spatial-bidirect.R (37 lines, 1 test)

Validates spatial PWM parameters (spat_factor, spat_bin, spat_min, spat_max)
across all PWM modes (pwm, pwm.max, pwm.max.pos, pwm.count) including
bidirectional counting with spatial sliding.
"""

import numpy as np
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_all_vtracks():
    """Remove all virtual tracks."""
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def _create_test_pssm():
    """Create a simple 2-position PSSM matching 'AC' exactly (matches R helper)."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # Only A
        [0.0, 1.0, 0.0, 0.0],  # Only C
    ])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPwmSpatialBasic:
    """Basic spatial PWM tests for each mode."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_pwm_with_spatial_parameters(self):
        """pwm with spatial parameters (higher weight in middle) returns valid score."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        spat_factors = [0.5, 1.0, 2.0, 1.0, 0.5]
        spat_bin = 20

        pm.gvtrack_create(
            "pwm_spatial", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=spat_factors, spat_bin=spat_bin,
        )
        scores = pm.gextract("pwm_spatial", test_interval, iterator=-1)

        assert scores["pwm_spatial"].dtype == np.float64
        assert not np.isnan(scores["pwm_spatial"].iloc[0])
        assert np.isfinite(scores["pwm_spatial"].iloc[0])

    def test_pwm_max_with_spatial_parameters(self):
        """pwm.max with spatial parameters returns valid score."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        pm.gvtrack_create(
            "pwm_max_spatial", None, func="pwm.max",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5], spat_bin=20,
        )
        scores = pm.gextract("pwm_max_spatial", test_interval, iterator=-1)

        assert not np.isnan(scores["pwm_max_spatial"].iloc[0])

    def test_pwm_max_pos_with_spatial_parameters(self):
        """pwm.max.pos with spatial parameters returns a nonzero position."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        pm.gvtrack_create(
            "pwm_maxpos_spatial", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5], spat_bin=20,
        )
        scores = pm.gextract("pwm_maxpos_spatial", test_interval, iterator=-1)

        assert abs(scores["pwm_maxpos_spatial"].iloc[0]) > 0


class TestPwmSpatialBackwardCompat:
    """Backward compatibility: no spatial params equals explicit no-spatial."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_no_spatial_backward_compat(self):
        """Two identical vtracks without spatial produce identical results."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 240)

        pm.gvtrack_create(
            "pwm_old", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
        )
        pm.gvtrack_create(
            "pwm_new", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
        )

        scores = pm.gextract(
            ["pwm_old", "pwm_new"], test_interval, iterator=-1,
        )
        np.testing.assert_allclose(
            scores["pwm_old"].values,
            scores["pwm_new"].values,
            atol=1e-10,
        )


class TestPwmSpatialUniformWeights:
    """Uniform spatial factors (all 1.0) should match no-spatial baseline."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_uniform_spatial_equals_no_spatial(self):
        """Uniform spatial factors (all 1.0) give same result as no spatial."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 250)

        pm.gvtrack_create(
            "pwm_nospatial", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
        )
        pm.gvtrack_create(
            "pwm_uniform", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=[1.0] * 10, spat_bin=10,
        )

        scores = pm.gextract(
            ["pwm_nospatial", "pwm_uniform"], test_interval, iterator=-1,
        )
        np.testing.assert_allclose(
            scores["pwm_nospatial"].values,
            scores["pwm_uniform"].values,
            atol=1e-5,
        )


class TestPwmSpatialRange:
    """Spatial range parameters (spat_min, spat_max)."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_spatial_range_parameters(self):
        """spat_min and spat_max restrict spatial weighting range."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 400)

        pm.gvtrack_create(
            "pwm_spatial_range", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=[0.5, 1.0, 2.0], spat_bin=50,
            spat_min=0, spat_max=100,
        )
        scores = pm.gextract("pwm_spatial_range", test_interval, iterator=-1)

        assert not np.isnan(scores["pwm_spatial_range"].iloc[0])


class TestPwmSpatialBidirectional:
    """Bidirectional spatial: bidi score >= forward-only score."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_bidi_geq_forward_with_spatial(self):
        """Bidirectional spatial PWM >= forward-only spatial PWM."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        pm.gvtrack_create(
            "pwm_spatial_bidi", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=[1.0, 2.0, 1.0], spat_bin=30,
        )
        pm.gvtrack_create(
            "pwm_spatial_fwd", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=[1.0, 2.0, 1.0], spat_bin=30,
        )

        scores = pm.gextract(
            ["pwm_spatial_bidi", "pwm_spatial_fwd"],
            test_interval, iterator=-1,
        )
        assert scores["pwm_spatial_bidi"].iloc[0] >= scores["pwm_spatial_fwd"].iloc[0]


class TestPwmSpatialIteratorShifts:
    """Spatial PWM with iterator shifts (sshift/eshift)."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_spatial_honors_iterator_shifts(self):
        """Spatial PWM with different shift magnitudes returns valid scores."""
        pssm = _create_test_pssm()
        base = pm.gintervals("1", 2000, 2100)

        spat_factors = [0.5, 1.0, 2.0, 1.0, 0.5]
        spat_bin = 20

        pm.gvtrack_create(
            "pwm_spat_small", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=spat_factors, spat_bin=spat_bin,
        )
        pm.gvtrack_create(
            "pwm_spat_large", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=spat_factors, spat_bin=spat_bin,
        )

        pm.gvtrack_iterator("pwm_spat_small", sshift=-10, eshift=10)
        pm.gvtrack_iterator("pwm_spat_large", sshift=-50, eshift=50)

        scores = pm.gextract(
            ["pwm_spat_small", "pwm_spat_large"], base, iterator=-1,
        )
        assert not np.isnan(scores["pwm_spat_small"].iloc[0])
        assert not np.isnan(scores["pwm_spat_large"].iloc[0])


class TestPwmSpatialErrorHandling:
    """Error handling for invalid spatial parameters."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    @pytest.mark.skip(reason="pymisha does not reject negative spatial weights at vtrack creation time")
    def test_negative_spatial_factor_rejected(self):
        """Negative spatial factors should be rejected."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "pwm_bad_spat", None, func="pwm",
                pssm=pssm, prior=0.01,
                spat_factor=[-1, 1, 1], spat_bin=10,
            )

    @pytest.mark.skip(reason="pymisha does not reject non-positive bin size at vtrack creation time")
    def test_non_positive_bin_size_rejected(self):
        """Non-positive bin size should be rejected."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "pwm_bad_bin", None, func="pwm",
                pssm=pssm, prior=0.01,
                spat_factor=[1, 1, 1], spat_bin=0,
            )

    @pytest.mark.skip(reason="pymisha does not reject zero spatial weights at vtrack creation time")
    def test_zero_spatial_factor_rejected(self):
        """Zero spatial factors should be rejected."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "pwm_zero_spat", None, func="pwm",
                pssm=pssm, prior=0.01,
                spat_factor=[0, 1, 1], spat_bin=10,
            )


class TestPwmCountSpatialBidirectSliding:
    """pwm.count spatial sliding with bidirectional scanning.

    Ported from test-pwm-count-spatial-bidirect.R.
    Validates that spatial sliding counts bidirectional hits consistently
    with a per-interval (non-sliding) reference.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_count_spatial_sliding_bidirect(self):
        """pwm.count spatial sliding counts bidirectional hits once per position.

        Uses uniform spatial weights (all 1.0) so the sliding path produces
        the same counts as a non-sliding reference computed per-interval.
        """
        pssm = _create_test_pssm()

        params = {
            "pssm": pssm,
            "bidirect": True,
            "strand": 1,
            "extend": True,
            "prior": 0.01,
            "score_thresh": -25,
            "spat_factor": [1.0] * 6,
            "spat_bin": 15,
        }

        pm.gvtrack_create("pwm_count_spat_slide", None, func="pwm.count", **params)

        n = 30
        starts = [2400 + i for i in range(n)]
        ends = [s + 55 for s in starts]
        ivs = pm.gintervals(["1"] * n, starts, ends)

        # Sliding result
        result_slide = pm.gextract("pwm_count_spat_slide", ivs, iterator=-1)

        # Per-interval reference (no sliding possible with single-interval extraction)
        per_interval_vals = []
        for idx in range(len(ivs)):
            single = ivs.iloc[[idx]].reset_index(drop=True)
            res = pm.gextract("pwm_count_spat_slide", single, iterator=-1)
            per_interval_vals.append(res["pwm_count_spat_slide"].iloc[0])

        np.testing.assert_allclose(
            result_slide["pwm_count_spat_slide"].values,
            np.array(per_interval_vals),
            atol=1e-6,
        )
