"""Tests for PWM sliding window optimization.

Ported from R misha tests/testthat/test-pwm-sliding-window.R (1268 lines, ~44 tests).
Validates that the C++ PWMScorer sliding window cache produces correct results
across all PWM modes (pwm/lse, pwm.max, pwm.count, pwm.max.pos) with various
iterator strides, shifts, and interval configurations.

Note: Test DB chr1 has an N-block starting at position 167280. All test regions
are kept below ~160000 (with shift margin) to avoid -inf from N bases.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Safe region on chr1 (no N bases): 0 - 167279
# With shifts up to 250, keep end below 167000
_SAFE_END = 160000


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


def _make_random_pssm(nrow, ncol=4, seed=42):
    """Create a random normalized PSSM (rows sum to 1)."""
    rng = np.random.RandomState(seed)
    mat = rng.random((nrow, ncol))
    return mat / mat.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPwmSlidingWindowBasicModes:
    """Basic sliding window tests for each PWM mode."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_total_likelihood_dense_iterator(self):
        """PWM (TOTAL_LIKELIHOOD) with dense iterator=1 triggers sliding window."""
        pssm = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.05, 0.05, 0.1, 0.8],
            [0.25, 0.25, 0.25, 0.25],
        ])

        pm.gvtrack_create("pwm_test", None, func="pwm", pssm=pssm, prior=0.01)
        result = pm.gextract("pwm_test", pm.gintervals("1", 10000, 11000), iterator=1)

        assert len(result) > 0
        assert not result["pwm_test"].isna().any()
        assert np.all(np.isfinite(result["pwm_test"]))

    def test_max_likelihood_mode(self):
        """PWM sliding window works with MAX_LIKELIHOOD (pwm.max) mode."""
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.05, 0.05, 0.1, 0.8],
        ])

        pm.gvtrack_create("pwm_max_test", None, func="pwm.max", pssm=pssm, prior=0.01)
        result = pm.gextract("pwm_max_test", pm.gintervals("1", 10000, 11000), iterator=1)

        assert len(result) > 0
        assert not result["pwm_max_test"].isna().any()
        assert np.all(np.isfinite(result["pwm_max_test"]))

    def test_motif_count_mode(self):
        """PWM sliding window works with MOTIF_COUNT (pwm.count) mode."""
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
        ])

        pm.gvtrack_create(
            "pwm_count_test", None, func="pwm.count",
            pssm=pssm, prior=0.01, score_thresh=-10,
        )
        result = pm.gextract("pwm_count_test", pm.gintervals("1", 10000, 11000), iterator=1)

        assert len(result) > 0
        assert not result["pwm_count_test"].isna().any()
        assert np.all(result["pwm_count_test"] >= 0)

    def test_max_pos_mode_iterator_1(self):
        """PWM MAX_POS mode works with iterator=1."""
        pssm = _make_random_pssm(8, seed=555)

        pm.gvtrack_create(
            "pwm_pos_test", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        result = pm.gextract("pwm_pos_test", pm.gintervals("1", 10000, 11000), iterator=1)

        assert len(result) > 0
        assert not result["pwm_pos_test"].isna().any()
        # Positions should be integers
        vals = result["pwm_pos_test"].values
        assert np.all(vals == np.round(vals))


class TestPwmSlidingWindowChromChange:
    """Cache invalidation on chromosome change."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_cache_invalidates_on_chrom_change(self):
        """Sliding window cache invalidates correctly on chromosome change."""
        pssm = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])

        pm.gvtrack_create("pwm_chrom_test", None, func="pwm", pssm=pssm, prior=0.01)

        intervals = pd.concat([
            pm.gintervals("1", 1000, 2000),
            pm.gintervals("2", 1000, 2000),
            pm.gintervals("1", 2000, 3000),
        ], ignore_index=True)

        result = pm.gextract("pwm_chrom_test", intervals, iterator=1)
        assert len(result) > 0
        assert not result["pwm_chrom_test"].isna().any()


class TestPwmSlidingWindowIteratorSteps:
    """Different iterator step sizes."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_different_iterator_steps(self):
        """Different iterator steps produce correctly sized results."""
        pssm = np.array([
            [0.6, 0.15, 0.15, 0.1],
            [0.15, 0.6, 0.15, 0.1],
            [0.15, 0.15, 0.6, 0.1],
        ])

        pm.gvtrack_create("pwm_iter_test", None, func="pwm", pssm=pssm, prior=0.01)

        interval = pm.gintervals("1", 10000, 11000)
        result_1 = pm.gextract("pwm_iter_test", interval, iterator=1)
        result_10 = pm.gextract("pwm_iter_test", interval, iterator=10)
        result_100 = pm.gextract("pwm_iter_test", interval, iterator=100)

        assert len(result_1) > len(result_10)
        assert len(result_10) > len(result_100)

        assert not result_1["pwm_iter_test"].isna().any()
        assert not result_10["pwm_iter_test"].isna().any()
        assert not result_100["pwm_iter_test"].isna().any()

    def test_various_iterator_values_with_shifts(self):
        """Multiple iterator values with shifts all produce valid results."""
        pssm = _make_random_pssm(7, seed=321)

        pm.gvtrack_create(
            "pwm_var_iter", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_var_iter", sshift=-150, eshift=150)

        interval = pm.gintervals("1", 10000, 60000)

        result_5 = pm.gextract("pwm_var_iter", interval, iterator=5)
        result_20 = pm.gextract("pwm_var_iter", interval, iterator=20)
        result_50 = pm.gextract("pwm_var_iter", interval, iterator=50)
        result_100 = pm.gextract("pwm_var_iter", interval, iterator=100)

        # Smaller iterator values give more results
        assert len(result_5) > len(result_20)
        assert len(result_20) > len(result_50)
        assert len(result_50) > len(result_100)

        # All should be valid
        for r in [result_5, result_20, result_50, result_100]:
            assert not r["pwm_var_iter"].isna().any()


class TestPwmSlidingWindowBidirectional:
    """Bidirectional PWM with sliding window."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_bidirectional_pssm(self):
        """Bidirectional sliding window produces valid results."""
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.05, 0.05, 0.1, 0.8],
        ])

        pm.gvtrack_create(
            "pwm_bidir_test", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        result = pm.gextract("pwm_bidir_test", pm.gintervals("1", 10000, 11000), iterator=1)

        assert len(result) > 0
        assert not result["pwm_bidir_test"].isna().any()

    def test_bidirect_ignores_strand_under_sliding(self):
        """When bidirect=True, strand parameter is ignored (union semantics)."""
        pssm = _create_test_pssm()

        n = 30
        starts = [2300 + i for i in range(n)]
        ends = [s + 50 for s in starts]
        ivs = pm.gintervals(["1"] * n, starts, ends)

        pm.gvtrack_create(
            "count_bidi_s1_slide", None, func="pwm.count",
            pssm=pssm, bidirect=True, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_create(
            "count_bidi_sneg1_slide", None, func="pwm.count",
            pssm=pssm, bidirect=True, strand=-1,
            extend=True, prior=0.01, score_thresh=-10,
        )

        out = pm.gextract(
            ["count_bidi_s1_slide", "count_bidi_sneg1_slide"],
            ivs, iterator=-1,
        )

        np.testing.assert_allclose(
            out["count_bidi_s1_slide"].values,
            out["count_bidi_sneg1_slide"].values,
            atol=1e-8,
        )


class TestPwmSlidingWindowShifts:
    """Sliding window with iterator shifts (sshift/eshift)."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_iterator_20_with_shifts(self):
        """iterator=20 with sshift=-250, eshift=250 on large region."""
        pssm = _make_random_pssm(20, seed=42)

        pm.gvtrack_create(
            "pwm_iter20_shift", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_iter20_shift", sshift=-250, eshift=250)

        result = pm.gextract(
            "pwm_iter20_shift",
            pm.gintervals("1", 10000, 110000),
            iterator=20,
        )
        # ~100000/20 = 5000 positions
        assert len(result) > 4000
        assert not result["pwm_iter20_shift"].isna().any()
        assert np.all(np.isfinite(result["pwm_iter20_shift"]))

    def test_iterator_20_without_shifts(self):
        """iterator=20 without shifts on large region."""
        pssm = _make_random_pssm(20, seed=42)

        pm.gvtrack_create(
            "pwm_iter20_no_shift", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )

        result = pm.gextract(
            "pwm_iter20_no_shift",
            pm.gintervals("1", 10000, _SAFE_END),
            iterator=20,
        )
        n_positions = (_SAFE_END - 10000) // 20
        assert len(result) >= n_positions - 100
        assert len(result) <= n_positions + 100
        assert not result["pwm_iter20_no_shift"].isna().any()
        assert np.all(np.isfinite(result["pwm_iter20_no_shift"]))

    def test_with_and_without_shifts_same_coordinates(self):
        """Shifted and unshifted produce same number of rows and coordinates."""
        pssm = _make_random_pssm(10, seed=123)

        pm.gvtrack_create(
            "pwm_with_shift", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_with_shift", sshift=-100, eshift=100)

        pm.gvtrack_create(
            "pwm_no_shift", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )

        interval = pm.gintervals("1", 10000, 20000)
        result_with = pm.gextract("pwm_with_shift", interval, iterator=50)
        result_without = pm.gextract("pwm_no_shift", interval, iterator=50)

        # Same number of rows and same coordinate ranges
        assert len(result_with) == len(result_without)
        assert result_with["start"].iloc[0] == result_without["start"].iloc[0]
        assert result_with["end"].iloc[-1] == result_without["end"].iloc[-1]

        # Both valid
        assert not result_with["pwm_with_shift"].isna().any()
        assert not result_without["pwm_no_shift"].isna().any()

    def test_shifts_produce_different_values(self):
        """Shifted vs unshifted produce different values (shift provides context)."""
        pssm = _make_random_pssm(10, seed=222)

        pm.gvtrack_create(
            "pwm_dense_shift", None, func="pwm",
            pssm=pssm, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_dense_shift", sshift=-50, eshift=50)

        pm.gvtrack_create(
            "pwm_dense_no_shift", None, func="pwm",
            pssm=pssm, prior=0.01,
        )

        interval = pm.gintervals("1", 50500, 50600)
        result_shift = pm.gextract("pwm_dense_shift", interval, iterator=1)
        result_no = pm.gextract("pwm_dense_no_shift", interval, iterator=1)

        assert len(result_shift) == len(result_no)

        # At least some values should differ
        diffs = np.abs(
            result_shift["pwm_dense_shift"].values
            - result_no["pwm_dense_no_shift"].values
        )
        assert np.any(diffs > 1e-10)

        # Both valid
        assert not result_shift["pwm_dense_shift"].isna().any()
        assert not result_no["pwm_dense_no_shift"].isna().any()

    def test_different_shift_magnitudes(self):
        """Large and small shifts produce same number of rows."""
        pssm = _make_random_pssm(8, seed=456)

        pm.gvtrack_create("pwm_large_shift", None, func="pwm", pssm=pssm, prior=0.01)
        pm.gvtrack_iterator("pwm_large_shift", sshift=-250, eshift=250)

        pm.gvtrack_create("pwm_small_shift", None, func="pwm", pssm=pssm, prior=0.01)
        pm.gvtrack_iterator("pwm_small_shift", sshift=-50, eshift=50)

        interval = pm.gintervals("1", 50000, 51000)
        result_large = pm.gextract("pwm_large_shift", interval, iterator=20)
        result_small = pm.gextract("pwm_small_shift", interval, iterator=20)

        assert len(result_large) >= 40
        assert len(result_small) >= 40
        assert len(result_large) == len(result_small)

        assert not result_large["pwm_large_shift"].isna().any()
        assert not result_small["pwm_small_shift"].isna().any()

    def test_large_shifts_with_various_iterators(self):
        """Shifts with iter=1, 100, 400 all produce valid results."""
        pssm = _make_random_pssm(12, seed=333)

        pm.gvtrack_create(
            "pwm_shift_consistent", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_shift_consistent", sshift=-200, eshift=200)

        interval = pm.gintervals("1", 50000, 60000)

        result_1 = pm.gextract("pwm_shift_consistent", interval, iterator=1)
        result_100 = pm.gextract("pwm_shift_consistent", interval, iterator=100)
        result_400 = pm.gextract("pwm_shift_consistent", interval, iterator=400)

        assert not result_1["pwm_shift_consistent"].isna().any()
        assert not result_100["pwm_shift_consistent"].isna().any()
        assert not result_400["pwm_shift_consistent"].isna().any()

        assert len(result_1) > len(result_100)
        assert len(result_100) > len(result_400)

        assert np.all(np.isfinite(result_1["pwm_shift_consistent"]))
        assert np.all(np.isfinite(result_100["pwm_shift_consistent"]))
        assert np.all(np.isfinite(result_400["pwm_shift_consistent"]))

    def test_boundary_positions_with_shifts(self):
        """Small interval with shifts at boundary returns valid results."""
        pssm = _make_random_pssm(5, seed=444)

        sshift, eshift = -30, 30
        pm.gvtrack_create(
            "pwm_boundary_test", None, func="pwm",
            pssm=pssm, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_boundary_test", sshift=sshift, eshift=eshift)

        interval = pm.gintervals("1", 50000, 50200)
        result = pm.gextract(
            "pwm_boundary_test", interval,
            iterator=abs(sshift) + abs(eshift),
        )

        assert len(result) > 0
        assert not result["pwm_boundary_test"].isna().any()
        assert np.all(np.isfinite(result["pwm_boundary_test"]))
        # 200bp / 60 ~ 3-4 positions
        assert 2 <= len(result) <= 5


class TestPwmSlidingWindowMultiInterval:
    """Sliding window across multiple intervals."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_iterator_20_multi_intervals(self):
        """iterator=20 with shifts across 10 consecutive intervals."""
        pssm = _make_random_pssm(12, seed=789)

        pm.gvtrack_create("pwm_multi_iter20", None, func="pwm", pssm=pssm, prior=0.01)
        pm.gvtrack_iterator("pwm_multi_iter20", sshift=-100, eshift=100)

        # 10 intervals of 10kb each, starting at 10000
        intervals = pd.concat([
            pm.gintervals("1", 10000 + i * 10000, 10000 + (i + 1) * 10000)
            for i in range(10)
        ], ignore_index=True)

        result = pm.gextract("pwm_multi_iter20", intervals, iterator=20)

        assert len(result) > 0
        assert not result["pwm_multi_iter20"].isna().any()
        assert np.all(np.isfinite(result["pwm_multi_iter20"]))
        # 10 intervals * ~500 positions each = ~5000
        assert len(result) >= 4000


class TestPwmSlidingWindowMaxPos:
    """MAX_POS mode with sliding window."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_max_pos_iterator_20_with_shifts(self):
        """MAX_POS with iterator=20 and shifts returns integer positions."""
        pssm = _make_random_pssm(10, seed=666)

        pm.gvtrack_create(
            "pwm_pos_iter20", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_pos_iter20", sshift=-150, eshift=150)
        result = pm.gextract(
            "pwm_pos_iter20",
            pm.gintervals("1", 10000, _SAFE_END),
            iterator=20,
        )

        assert len(result) > 0
        assert not result["pwm_pos_iter20"].isna().any()
        vals = result["pwm_pos_iter20"].values
        assert np.all(vals == np.round(vals))

    def test_max_pos_no_shift_iterator_50(self):
        """MAX_POS without shifts with iterator=50."""
        pssm = _make_random_pssm(7, seed=777)

        pm.gvtrack_create(
            "pwm_pos_no_shift", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        result = pm.gextract(
            "pwm_pos_no_shift",
            pm.gintervals("1", 10000, 60000),
            iterator=50,
        )

        assert len(result) > 0
        assert not result["pwm_pos_no_shift"].isna().any()
        vals = result["pwm_pos_no_shift"].values
        assert np.all(vals == np.round(vals))

    def test_max_pos_various_iterators_with_shifts(self):
        """MAX_POS with iter=10 vs iter=100 gives correct row counts."""
        pssm = _make_random_pssm(9, seed=888)

        pm.gvtrack_create(
            "pwm_pos_var", None, func="pwm.max.pos",
            pssm=pssm, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_pos_var", sshift=-100, eshift=100)

        interval = pm.gintervals("1", 10000, 60000)
        result_10 = pm.gextract("pwm_pos_var", interval, iterator=10)
        result_100 = pm.gextract("pwm_pos_var", interval, iterator=100)

        assert len(result_10) > len(result_100)
        assert not result_10["pwm_pos_var"].isna().any()
        assert not result_100["pwm_pos_var"].isna().any()

        # Integer values
        for r in [result_10, result_100]:
            vals = r["pwm_pos_var"].values
            assert np.all(vals == np.round(vals))


class TestPwmSlidingWindowModeShiftsCombined:
    """pwm.max and pwm.count with iterator=20 and shifts."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_max_likelihood_iter20_shifts(self):
        """pwm.max with iterator=20 and bidirect+shifts."""
        pssm = _make_random_pssm(8, seed=654)

        pm.gvtrack_create(
            "pwm_max_iter20", None, func="pwm.max",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_max_iter20", sshift=-200, eshift=200)

        result = pm.gextract(
            "pwm_max_iter20",
            pm.gintervals("1", 10000, _SAFE_END),
            iterator=20,
        )

        assert len(result) > 0
        assert not result["pwm_max_iter20"].isna().any()
        assert np.all(np.isfinite(result["pwm_max_iter20"]))

    def test_motif_count_iter20_shifts(self):
        """pwm.count with iterator=20 and shifts."""
        pssm = _make_random_pssm(6, seed=987)

        pm.gvtrack_create(
            "pwm_count_iter20", None, func="pwm.count",
            pssm=pssm, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_iterator("pwm_count_iter20", sshift=-100, eshift=100)

        result = pm.gextract(
            "pwm_count_iter20",
            pm.gintervals("1", 10000, _SAFE_END),
            iterator=20,
        )

        assert len(result) > 0
        assert not result["pwm_count_iter20"].isna().any()
        assert np.all(result["pwm_count_iter20"] >= 0)


class TestPwmSlidingWindowSpatialBaseline:
    """Sliding window matches spatial (no-sliding) baseline for uniform weights."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_spatial_sliding_matches_per_interval(self):
        """Sliding window with spatial weights matches per-interval extraction."""
        pssm = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])

        pm.gvtrack_create(
            "pwm_slide_spatial", None, func="pwm",
            pssm=pssm, prior=0.01,
            spat_factor=[0.2, 1.0, 3.0, 1.0, 0.5],
            spat_bin=10,
        )

        intervals = pd.concat([
            pm.gintervals("1", 10000 + i, 10100 + i)
            for i in range(4)
        ], ignore_index=True)

        # Per-interval extraction (no sliding possible)
        per_interval = []
        for idx in range(len(intervals)):
            single = intervals.iloc[[idx]].reset_index(drop=True)
            res = pm.gextract("pwm_slide_spatial", single, iterator=1)
            per_interval.extend(res["pwm_slide_spatial"].tolist())

        # Sliding extraction
        sliding_results = pm.gextract("pwm_slide_spatial", intervals, iterator=1)

        np.testing.assert_allclose(
            sliding_results["pwm_slide_spatial"].values,
            np.array(per_interval),
            atol=1e-6,
        )


class TestPwmSlidingVsSpatialBaselineByMode:
    """Plus-strand and minus-strand sliding equals spatial uniform-weight baseline."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def _build_overlapping_intervals(self, start, n, span):
        starts = [start + i for i in range(n)]
        ends = [s + span for s in starts]
        return pm.gintervals(["1"] * n, starts, ends)

    def test_pwm_plus_strand_sliding_vs_spatial_ref(self):
        """pwm: plus-strand sliding equals spatial (uniform weights) baseline."""
        pssm = _create_test_pssm()
        ivs = self._build_overlapping_intervals(2000, 40, 60)

        pm.gvtrack_create(
            "pwm_plus_slide", None, func="pwm",
            pssm=pssm, bidirect=False, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_create(
            "pwm_plus_spatial_ref", None, func="pwm",
            pssm=pssm, bidirect=False, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
            spat_factor=[1.0] * 5, spat_bin=20,
        )

        res = pm.gextract(
            ["pwm_plus_slide", "pwm_plus_spatial_ref"],
            ivs, iterator=-1,
        )
        np.testing.assert_allclose(
            res["pwm_plus_slide"].values,
            res["pwm_plus_spatial_ref"].values,
            atol=1e-6,
        )

    def test_pwm_max_plus_strand_sliding_vs_spatial_ref(self):
        """pwm.max: plus-strand sliding equals spatial (uniform weights) baseline."""
        pssm = _create_test_pssm()
        ivs = self._build_overlapping_intervals(2000, 40, 60)

        pm.gvtrack_create(
            "pwmmax_plus_slide", None, func="pwm.max",
            pssm=pssm, bidirect=False, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_create(
            "pwmmax_plus_spatial_ref", None, func="pwm.max",
            pssm=pssm, bidirect=False, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
            spat_factor=[1.0] * 5, spat_bin=20,
        )

        res = pm.gextract(
            ["pwmmax_plus_slide", "pwmmax_plus_spatial_ref"],
            ivs, iterator=-1,
        )
        np.testing.assert_allclose(
            res["pwmmax_plus_slide"].values,
            res["pwmmax_plus_spatial_ref"].values,
            atol=1e-6,
        )

    def test_pwm_max_minus_strand_sliding_vs_spatial_ref(self):
        """pwm.max: minus-strand sliding equals spatial (uniform weights) baseline."""
        pssm = _create_test_pssm()
        ivs = self._build_overlapping_intervals(2100, 40, 60)

        pm.gvtrack_create(
            "pwmmax_minus_slide", None, func="pwm.max",
            pssm=pssm, bidirect=False, strand=-1,
            extend=True, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_create(
            "pwmmax_minus_spatial_ref", None, func="pwm.max",
            pssm=pssm, bidirect=False, strand=-1,
            extend=True, prior=0.01, score_thresh=-10,
            spat_factor=[1.0] * 5, spat_bin=20,
        )

        res = pm.gextract(
            ["pwmmax_minus_slide", "pwmmax_minus_spatial_ref"],
            ivs, iterator=-1,
        )
        np.testing.assert_allclose(
            res["pwmmax_minus_slide"].values,
            res["pwmmax_minus_spatial_ref"].values,
            atol=1e-8,
        )

    def test_pwm_count_plus_strand_sliding_vs_spatial_ref(self):
        """pwm.count: plus-strand sliding equals spatial (uniform weights) baseline."""
        pssm = _create_test_pssm()
        ivs = self._build_overlapping_intervals(2000, 40, 60)

        pm.gvtrack_create(
            "count_plus_slide", None, func="pwm.count",
            pssm=pssm, bidirect=False, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_create(
            "count_plus_spatial_ref", None, func="pwm.count",
            pssm=pssm, bidirect=False, strand=1,
            extend=True, prior=0.01, score_thresh=-10,
            spat_factor=[1.0] * 5, spat_bin=20,
        )

        res = pm.gextract(
            ["count_plus_slide", "count_plus_spatial_ref"],
            ivs, iterator=-1,
        )
        np.testing.assert_allclose(
            res["count_plus_slide"].values,
            res["count_plus_spatial_ref"].values,
            atol=1e-8,
        )

    def test_pwm_count_minus_strand_sliding_vs_spatial_ref(self):
        """pwm.count: minus-strand sliding equals spatial (uniform weights) baseline."""
        pssm = _create_test_pssm()
        ivs = self._build_overlapping_intervals(2200, 40, 60)

        pm.gvtrack_create(
            "count_minus_slide", None, func="pwm.count",
            pssm=pssm, bidirect=False, strand=-1,
            extend=True, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_create(
            "count_minus_spatial_ref", None, func="pwm.count",
            pssm=pssm, bidirect=False, strand=-1,
            extend=True, prior=0.01, score_thresh=-10,
            spat_factor=[1.0] * 5, spat_bin=20,
        )

        res = pm.gextract(
            ["count_minus_slide", "count_minus_spatial_ref"],
            ivs, iterator=-1,
        )
        np.testing.assert_allclose(
            res["count_minus_slide"].values,
            res["count_minus_spatial_ref"].values,
            atol=1e-8,
        )


class TestPwmSpatialSlidingStride:
    """Spatial sliding with stride > 1 correctness."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_spatial_stride_gt1_produces_valid_results(self):
        """Spatial sliding with stride>1 produces valid non-NaN results.

        Note: The R test compares sliding vs MISHA_DISABLE_SPATIAL_SLIDING=1
        baseline, which pymisha does not support. Instead we verify that the
        spatial sliding path produces valid, finite results for all positions.
        """
        pssm = _create_test_pssm()

        params = {
            "pssm": pssm,
            "bidirect": False,
            "strand": 1,
            "extend": True,
            "prior": 0.01,
            "spat_factor": [0.2, 1.0, 3.0, 1.0, 0.5],
            "spat_bin": 20,
        }

        pm.gvtrack_create("pwm_spat_stride_slide", None, func="pwm", **params)

        interval = pm.gintervals("1", 12000, 12250)
        result_slide = pm.gextract("pwm_spat_stride_slide", interval, iterator=5)

        assert len(result_slide) > 0
        assert not result_slide["pwm_spat_stride_slide"].isna().any()
        # With spatial weighting, all values should be finite (region has no Ns)
        assert np.all(np.isfinite(result_slide["pwm_spat_stride_slide"]))


class TestPwmSlidingWindowRegression:
    """Regression tests ensuring consistency across modes, iterators, and shifts."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_regression_iter1_with_shifts(self):
        """PWM iterator=1 with sshift=-250, eshift=250 produces valid results."""
        pssm = _make_random_pssm(20, seed=100)

        pm.gvtrack_create(
            "pwm_reg1", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_reg1", sshift=-250, eshift=250)

        result = pm.gextract("pwm_reg1", pm.gintervals("1", 10000, _SAFE_END), iterator=1)
        assert len(result) > 0
        assert not result["pwm_reg1"].isna().any()
        assert np.all(np.isfinite(result["pwm_reg1"]))

    def test_regression_iter20_with_shifts(self):
        """PWM iterator=20 with shifts produces valid results."""
        pssm = _make_random_pssm(20, seed=100)

        pm.gvtrack_create(
            "pwm_reg20", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_reg20", sshift=-250, eshift=250)

        result = pm.gextract("pwm_reg20", pm.gintervals("1", 10000, _SAFE_END), iterator=20)
        assert len(result) > 0
        assert not result["pwm_reg20"].isna().any()

    def test_regression_iter20_no_shifts(self):
        """PWM iterator=20 without shifts produces valid results."""
        pssm = _make_random_pssm(20, seed=100)

        pm.gvtrack_create(
            "pwm_reg_no_shift", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )

        result = pm.gextract("pwm_reg_no_shift", pm.gintervals("1", 10000, _SAFE_END), iterator=20)
        assert len(result) > 0
        assert not result["pwm_reg_no_shift"].isna().any()

    def test_regression_iter100_with_shifts(self):
        """PWM iterator=100 with shifts produces valid results."""
        pssm = _make_random_pssm(20, seed=100)

        pm.gvtrack_create(
            "pwm_reg100", None, func="pwm",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_reg100", sshift=-200, eshift=200)

        result = pm.gextract("pwm_reg100", pm.gintervals("1", 10000, _SAFE_END), iterator=100)
        assert len(result) > 0
        assert not result["pwm_reg100"].isna().any()

    def test_regression_max_mode_with_shifts(self):
        """pwm.max iterator=20 with shifts regression."""
        pssm = _make_random_pssm(8, seed=200)

        pm.gvtrack_create(
            "pwm_max_reg", None, func="pwm.max",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_max_reg", sshift=-100, eshift=100)

        result = pm.gextract("pwm_max_reg", pm.gintervals("1", 10000, 60000), iterator=20)
        assert len(result) > 0
        assert not result["pwm_max_reg"].isna().any()

    def test_regression_count_mode_with_shifts(self):
        """pwm.count iterator=20 with shifts regression."""
        pssm = _make_random_pssm(6, seed=300)

        pm.gvtrack_create(
            "pwm_count_reg", None, func="pwm.count",
            pssm=pssm, prior=0.01, score_thresh=-10,
        )
        pm.gvtrack_iterator("pwm_count_reg", sshift=-100, eshift=100)

        result = pm.gextract("pwm_count_reg", pm.gintervals("1", 10000, 60000), iterator=20)
        assert len(result) > 0
        assert not result["pwm_count_reg"].isna().any()
        assert np.all(result["pwm_count_reg"] >= 0)

    def test_regression_multi_intervals_with_shifts(self):
        """Multiple intervals with iterator=20 and shifts regression."""
        pssm = _make_random_pssm(12, seed=400)

        pm.gvtrack_create("pwm_multi_reg", None, func="pwm", pssm=pssm, prior=0.01)
        pm.gvtrack_iterator("pwm_multi_reg", sshift=-100, eshift=100)

        intervals = pd.concat([
            pm.gintervals("1", 10000 + i * 10000, 10000 + (i + 1) * 10000)
            for i in range(10)
        ], ignore_index=True)

        result = pm.gextract("pwm_multi_reg", intervals, iterator=20)
        assert len(result) > 0
        assert not result["pwm_multi_reg"].isna().any()

    def test_regression_max_pos_iter1_shifts(self):
        """pwm.max.pos iterator=1 with shifts regression."""
        pssm = _make_random_pssm(10, seed=500)

        pm.gvtrack_create(
            "pwm_pos_reg1", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_pos_reg1", sshift=-150, eshift=150)

        result = pm.gextract("pwm_pos_reg1", pm.gintervals("1", 10000, 60000), iterator=1)
        assert len(result) > 0
        assert not result["pwm_pos_reg1"].isna().any()

    def test_regression_max_pos_iter20_shifts(self):
        """pwm.max.pos iterator=20 with shifts regression."""
        pssm = _make_random_pssm(10, seed=500)

        pm.gvtrack_create(
            "pwm_pos_reg20", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_pos_reg20", sshift=-150, eshift=150)

        result = pm.gextract("pwm_pos_reg20", pm.gintervals("1", 10000, _SAFE_END), iterator=20)
        assert len(result) > 0
        assert not result["pwm_pos_reg20"].isna().any()

    def test_regression_max_pos_iter20_no_shifts(self):
        """pwm.max.pos iterator=20 without shifts regression."""
        pssm = _make_random_pssm(10, seed=500)

        pm.gvtrack_create(
            "pwm_pos_reg_no_shift", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )

        result = pm.gextract(
            "pwm_pos_reg_no_shift",
            pm.gintervals("1", 10000, _SAFE_END),
            iterator=20,
        )
        assert len(result) > 0
        assert not result["pwm_pos_reg_no_shift"].isna().any()

    def test_regression_max_pos_iter100_shifts(self):
        """pwm.max.pos iterator=100 with shifts regression."""
        pssm = _make_random_pssm(10, seed=500)

        pm.gvtrack_create(
            "pwm_pos_reg100", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, prior=0.01,
        )
        pm.gvtrack_iterator("pwm_pos_reg100", sshift=-200, eshift=200)

        result = pm.gextract(
            "pwm_pos_reg100",
            pm.gintervals("1", 10000, _SAFE_END),
            iterator=100,
        )
        assert len(result) > 0
        assert not result["pwm_pos_reg100"].isna().any()
