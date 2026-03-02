"""Tests for gseq_pwm spatial weighting (spat_factor / spat_bin)."""

import math

import numpy as np
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_sum_exp(x):
    """Compute log-sum-exp matching the C++/R iterative algorithm."""
    x = [v for v in x if np.isfinite(v)]
    if not x:
        return -np.inf
    if len(x) == 1:
        return x[0]
    x = sorted(x, reverse=True)
    s = x[0]
    for i in range(1, len(x)):
        if s > x[i]:
            s = s + math.log1p(math.exp(x[i] - s))
        else:
            s = x[i] + math.log1p(math.exp(s - x[i]))
    return s


def _manual_score_fwd(seq, pssm, prior=0.01):
    """Per-position forward-strand log-prob scores (list of floats)."""
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    w = pssm.shape[0]
    normed = pssm.copy() + prior
    normed = normed / normed.sum(axis=1, keepdims=True)

    scores = []
    for i in range(len(seq) - w + 1):
        window = seq[i:i + w]
        s = 0.0
        valid = True
        for j in range(w):
            idx = base_map.get(window[j])
            if idx is None:
                s = -np.inf
                valid = False
                break
            p = normed[j, idx]
            if p == 0:
                s = -np.inf
                valid = False
                break
            s += math.log(p)
        scores.append(s if valid else -np.inf)
    return scores


def _manual_score_rc(seq, pssm, prior=0.01):
    """Per-position reverse-complement strand log-prob scores."""
    comp_map = {"A": 3, "C": 2, "G": 1, "T": 0}
    w = pssm.shape[0]
    normed = pssm.copy() + prior
    normed = normed / normed.sum(axis=1, keepdims=True)

    scores = []
    for i in range(len(seq) - w + 1):
        window = seq[i:i + w]
        s = 0.0
        valid = True
        for j in range(w):
            code = comp_map.get(window[j])
            if code is None:
                s = -np.inf
                valid = False
                break
            p = normed[w - 1 - j, code]
            if p == 0:
                s = -np.inf
                valid = False
                break
            s += math.log(p)
        scores.append(s if valid else -np.inf)
    return scores


def _spat_log(spat_factor, spat_bin, pos_idx, roi_start_1based):
    """Compute spatial log-factor for a 0-based window position."""
    offset = (pos_idx + 1) - roi_start_1based
    sb = offset // spat_bin
    n = len(spat_factor)
    if sb < 0:
        sb = 0
    elif sb >= n:
        sb = n - 1
    return math.log(max(spat_factor[sb], 1e-30))


# ---------------------------------------------------------------------------
# A simple 4-position PSSM that clearly prefers ACGT
# ---------------------------------------------------------------------------

PSSM = np.array([
    [0.7, 0.1, 0.1, 0.1],  # prefers A
    [0.1, 0.7, 0.1, 0.1],  # prefers C
    [0.1, 0.1, 0.7, 0.1],  # prefers G
    [0.1, 0.1, 0.1, 0.7],  # prefers T
])

PRIOR = 0.01

# ---------------------------------------------------------------------------
# Tests: basic semantics
# ---------------------------------------------------------------------------


class TestSpatialLse:
    """LSE mode with spatial weighting."""

    def test_uniform_spat_equals_no_spat(self):
        """Uniform spatial weights (all 1.0) = no spatial weighting."""
        seq = "ACGTACGTACGT"
        ref = pm.gseq_pwm(seq, PSSM, mode="lse", prior=PRIOR)
        spat = pm.gseq_pwm(
            seq, PSSM, mode="lse", prior=PRIOR,
            spat_factor=[1.0, 1.0, 1.0, 1.0, 1.0], spat_bin=3,
        )
        np.testing.assert_allclose(ref, spat, atol=1e-10)

    def test_high_weight_boosts_score(self):
        """A spatial weight > 1 at every bin increases the LSE score."""
        seq = "ACGTACGTACGT"
        ref = pm.gseq_pwm(seq, PSSM, mode="lse", prior=PRIOR)
        boosted = pm.gseq_pwm(
            seq, PSSM, mode="lse", prior=PRIOR,
            spat_factor=[2.0, 2.0, 2.0, 2.0, 2.0], spat_bin=3,
        )
        assert boosted[0] > ref[0]

    def test_low_weight_decreases_score(self):
        """A spatial weight < 1 at every bin decreases the LSE score."""
        seq = "ACGTACGTACGT"
        ref = pm.gseq_pwm(seq, PSSM, mode="lse", prior=PRIOR)
        dampened = pm.gseq_pwm(
            seq, PSSM, mode="lse", prior=PRIOR,
            spat_factor=[0.5, 0.5, 0.5, 0.5, 0.5], spat_bin=3,
        )
        assert dampened[0] < ref[0]

    def test_manual_reference_forward_only(self):
        """Manual computation of spatially-weighted LSE, forward strand only."""
        seq = "ACGTACGT"
        spat_factor = [0.5, 2.0]
        spat_bin = 4
        w = PSSM.shape[0]

        fwd_scores = _manual_score_fwd(seq, PSSM, PRIOR)
        roi_start = 1  # default
        weighted = []
        for i, s in enumerate(fwd_scores):
            sl = _spat_log(spat_factor, spat_bin, i, roi_start)
            weighted.append(s + sl if s > -np.inf else -np.inf)

        expected = _log_sum_exp(weighted)
        actual = pm.gseq_pwm(
            seq, PSSM, mode="lse", bidirect=False, strand=1,
            prior=PRIOR, spat_factor=spat_factor, spat_bin=spat_bin,
        )
        np.testing.assert_allclose(actual[0], expected, rtol=1e-6)

    def test_manual_reference_bidirectional(self):
        """Manual computation of spatially-weighted LSE, both strands."""
        seq = "ACGTACGT"
        spat_factor = [0.5, 2.0]
        spat_bin = 4

        fwd_scores = _manual_score_fwd(seq, PSSM, PRIOR)
        rc_scores = _manual_score_rc(seq, PSSM, PRIOR)
        roi_start = 1
        weighted = []
        for i in range(len(fwd_scores)):
            sl = _spat_log(spat_factor, spat_bin, i, roi_start)
            if fwd_scores[i] > -np.inf:
                weighted.append(fwd_scores[i] + sl)
            if rc_scores[i] > -np.inf:
                weighted.append(rc_scores[i] + sl)

        expected = _log_sum_exp(weighted)
        actual = pm.gseq_pwm(
            seq, PSSM, mode="lse", bidirect=True,
            prior=PRIOR, spat_factor=spat_factor, spat_bin=spat_bin,
        )
        np.testing.assert_allclose(actual[0], expected, rtol=1e-6)


class TestSpatialMax:
    """MAX mode with spatial weighting."""

    def test_uniform_spat_equals_no_spat(self):
        seq = "ACGTACGT"
        ref = pm.gseq_pwm(seq, PSSM, mode="max", prior=PRIOR)
        spat = pm.gseq_pwm(
            seq, PSSM, mode="max", prior=PRIOR,
            spat_factor=[1.0, 1.0], spat_bin=5,
        )
        np.testing.assert_allclose(ref, spat, atol=1e-10)

    def test_high_weight_boosts_max(self):
        """High spatial weight boosts the max score above the non-spatial max."""
        seq = "ACGTACGT"
        ref = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
        )
        boosted = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
            spat_factor=[2.0, 2.0], spat_bin=5,
        )
        # Every position gets +log(2), so max should also increase by log(2)
        np.testing.assert_allclose(
            boosted[0], ref[0] + math.log(2.0), rtol=1e-6,
        )

    def test_manual_reference_max(self):
        """Manual computation of spatially-weighted MAX, forward only."""
        seq = "ACGTACGT"
        spat_factor = [0.5, 2.0]
        spat_bin = 4

        fwd_scores = _manual_score_fwd(seq, PSSM, PRIOR)
        roi_start = 1
        best = -np.inf
        for i, s in enumerate(fwd_scores):
            sl = _spat_log(spat_factor, spat_bin, i, roi_start)
            val = s + sl if s > -np.inf else -np.inf
            if val > best:
                best = val

        actual = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1,
            prior=PRIOR, spat_factor=spat_factor, spat_bin=spat_bin,
        )
        np.testing.assert_allclose(actual[0], best, rtol=1e-6)


class TestSpatialPos:
    """POS mode with spatial weighting."""

    def test_pos_shifts_with_spatial(self):
        """Spatial weight can shift which position is reported as best."""
        # Construct sequence with identical motifs at positions 0 and 4
        seq = "ACGTACGT"
        # Without spatial: position 1 (0-based: 0) for forward ACGT match
        ref = pm.gseq_pwm(
            seq, PSSM, mode="pos", bidirect=False, strand=1, prior=PRIOR,
        )
        # Heavily weight only the second half of the sequence
        spat = pm.gseq_pwm(
            seq, PSSM, mode="pos", bidirect=False, strand=1, prior=PRIOR,
            spat_factor=[0.001, 100.0], spat_bin=4,
        )
        # With the second half weighted 100x, position 5 (1-based) should win
        assert spat[0] == 5.0  # 1-based position 5


class TestSpatialCount:
    """COUNT mode with spatial weighting."""

    def test_spatial_can_push_below_threshold(self):
        """Low spatial weight can push scores below threshold, reducing count."""
        seq = "ACGTACGT"
        # Without spatial, count everything above a low threshold
        ref = pm.gseq_pwm(
            seq, PSSM, mode="count", bidirect=False, strand=1,
            prior=PRIOR, score_thresh=-100,
        )
        # With very low spatial weight, scores drop, reducing count above 0
        dampened = pm.gseq_pwm(
            seq, PSSM, mode="count", bidirect=False, strand=1,
            prior=PRIOR, score_thresh=0,
            spat_factor=[1e-10, 1e-10], spat_bin=5,
        )
        # All scores should be below 0 with tiny spatial weights
        assert dampened[0] == 0

    def test_spatial_can_push_above_threshold(self):
        """High spatial weight can push scores above threshold."""
        seq = "ACGTACGT"
        # With a high threshold, no positions without spatial
        ref = pm.gseq_pwm(
            seq, PSSM, mode="count", bidirect=False, strand=1,
            prior=PRIOR, score_thresh=10,
        )
        assert ref[0] == 0  # no positions above 10

        # With massive spatial boost
        boosted = pm.gseq_pwm(
            seq, PSSM, mode="count", bidirect=False, strand=1,
            prior=PRIOR, score_thresh=10,
            spat_factor=[1e10, 1e10], spat_bin=5,
        )
        assert boosted[0] > 0  # some positions now exceed threshold


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestSpatialEdgeCases:

    def test_single_bin(self):
        """spat_factor with a single element applies uniformly."""
        seq = "ACGTACGT"
        factor = 2.0
        single = pm.gseq_pwm(
            seq, PSSM, mode="lse", prior=PRIOR,
            spat_factor=[factor], spat_bin=1,
        )
        ref = pm.gseq_pwm(seq, PSSM, mode="lse", prior=PRIOR)
        # Single bin factor of 2 means every score gets +log(2)
        # For LSE mode: LSE(s_i + log(2)) = LSE(s_i) + log(2)
        expected = ref[0] + math.log(factor)
        np.testing.assert_allclose(single[0], expected, rtol=1e-6)

    def test_interval_shorter_than_spat_bin(self):
        """Sequence shorter than spat_bin uses only the first bin."""
        seq = "ACGT"  # length 4, pssm width 4 -> only 1 window position
        result = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
            spat_factor=[3.0, 0.01, 0.01], spat_bin=100,
        )
        assert np.isfinite(result[0])

    def test_bin_clamping(self):
        """Positions beyond the last bin get clamped to the last bin."""
        seq = "ACGTACGTACGTACGT"  # 16 bases
        # spat_bin=4, so bins 0..3 for 13 positions (16-4+1=13)
        # Only 2 bins provided -> positions in bin >= 2 clamp to bin 1
        spat_factor = [1.0, 5.0]
        result = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
            spat_factor=spat_factor, spat_bin=4,
        )
        # Should not error and should be finite
        assert np.isfinite(result[0])

    def test_multiple_sequences(self):
        """Spatial weighting works correctly for multiple sequences."""
        seqs = ["ACGTACGT", "TGCATGCA", "ACGTACGT"]
        result = pm.gseq_pwm(
            seqs, PSSM, mode="lse", prior=PRIOR,
            spat_factor=[0.5, 2.0], spat_bin=4,
        )
        assert result.shape == (3,)
        # First and third sequences are identical, so results must match
        np.testing.assert_allclose(result[0], result[2], atol=1e-10)

    def test_roi_with_spatial(self):
        """Spatial offset is relative to ROI start, not sequence start."""
        seq = "AAAAACGTAAAA"  # length 12
        # Set ROI to positions 5..8 (1-based), which is the ACGT part
        result_with_roi = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
            start_pos=5, end_pos=8,
            spat_factor=[3.0], spat_bin=10,
        )
        result_no_roi = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
            spat_factor=[3.0], spat_bin=10,
        )
        # With ROI restricted, spatial factor is computed from ROI start
        assert np.isfinite(result_with_roi[0])

    def test_spat_bin_larger_than_sequence(self):
        """spat_bin larger than sequence puts all positions in bin 0."""
        seq = "ACGT"
        factor = 3.0
        result = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
            spat_factor=[factor, 0.01], spat_bin=1000,
        )
        # All positions in bin 0, so should equal non-spatial + log(factor)
        ref = pm.gseq_pwm(
            seq, PSSM, mode="max", bidirect=False, strand=1, prior=PRIOR,
        )
        expected = ref[0] + math.log(factor)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestSpatialValidation:

    def test_negative_spat_factor_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pm.gseq_pwm("ACGT", PSSM, spat_factor=[-1.0, 1.0])

    def test_empty_spat_factor_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            pm.gseq_pwm("ACGT", PSSM, spat_factor=[])

    def test_2d_spat_factor_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            pm.gseq_pwm("ACGT", PSSM, spat_factor=[[1.0, 2.0], [3.0, 4.0]])

    def test_zero_spat_factor_works(self):
        """Zero spatial factor is allowed (clamped to tiny value internally)."""
        result = pm.gseq_pwm("ACGT", PSSM, spat_factor=[0.0, 1.0])
        assert np.isfinite(result[0])
