"""Smoke tests for pm_pwm_score_strings C++ function."""

import numpy as np
import pytest

import _pymisha


def _make_simple_pssm():
    """Create a simple 4-position PSSM that strongly prefers ACGT."""
    # Each row = one motif position, columns = A, C, G, T
    pssm = np.array(
        [
            [0.97, 0.01, 0.01, 0.01],  # position 0: A
            [0.01, 0.97, 0.01, 0.01],  # position 1: C
            [0.01, 0.01, 0.97, 0.01],  # position 2: G
            [0.01, 0.01, 0.01, 0.97],  # position 3: T
        ],
        dtype=np.float64,
    )
    return pssm


class TestPwmScoreStringsBasic:
    """Basic smoke tests for pm_pwm_score_strings."""

    def test_returns_numpy_array(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGTACGT", "TTTTTTTT"]
        result = _pymisha.pm_pwm_score_strings(seqs, pssm)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (2,)

    def test_all_modes_return_finite(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGTACGT", "GGGGGGGG", "ACGT"]
        for mode in ["lse", "max", "pos", "count"]:
            result = _pymisha.pm_pwm_score_strings(seqs, pssm, 0.01, mode)
            assert result.shape == (3,), f"mode={mode}"
            assert all(np.isfinite(result)), f"mode={mode}: {result}"

    def test_empty_list(self):
        pssm = _make_simple_pssm()
        result = _pymisha.pm_pwm_score_strings([], pssm)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_short_sequence_returns_nan(self):
        pssm = _make_simple_pssm()  # 4 positions
        seqs = ["AC"]  # shorter than motif
        result = _pymisha.pm_pwm_score_strings(seqs, pssm)
        assert result.shape == (1,)
        assert np.isnan(result[0])

    def test_exact_length_sequence(self):
        pssm = _make_simple_pssm()  # 4 positions
        seqs = ["ACGT"]  # exactly motif length
        result = _pymisha.pm_pwm_score_strings(seqs, pssm)
        assert result.shape == (1,)
        assert np.isfinite(result[0])


class TestPwmScoreStringsScoring:
    """Tests for correctness of scoring."""

    def test_matching_sequence_scores_higher_lse(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGT", "TTTT"]
        result = _pymisha.pm_pwm_score_strings(seqs, pssm, 0.01, "lse")
        # ACGT should score higher than TTTT for an ACGT-preferring motif
        assert result[0] > result[1]

    def test_matching_sequence_scores_higher_max(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGT", "TTTT"]
        result = _pymisha.pm_pwm_score_strings(seqs, pssm, 0.01, "max")
        assert result[0] > result[1]

    def test_count_mode_with_threshold(self):
        pssm = _make_simple_pssm()
        # Use a very low threshold to count positions
        seqs = ["ACGTACGT"]
        result = _pymisha.pm_pwm_score_strings(
            seqs, pssm, 0.01, "count", True, 0, -1000.0
        )
        assert result[0] > 0  # should have some hits with low threshold

    def test_pos_mode_returns_position(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGT"]
        result = _pymisha.pm_pwm_score_strings(seqs, pssm, 0.01, "pos")
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_bidirect_false_forward(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGTACGT"]
        result_bidir = _pymisha.pm_pwm_score_strings(
            seqs, pssm, 0.01, "lse", True
        )
        result_fwd = _pymisha.pm_pwm_score_strings(
            seqs, pssm, 0.01, "lse", False, 1
        )
        # Bidirectional should be >= forward only (or very close)
        assert result_bidir[0] >= result_fwd[0] - 1e-6


class TestPwmScoreStringsEdgeCases:
    """Edge case tests."""

    def test_single_sequence(self):
        pssm = _make_simple_pssm()
        result = _pymisha.pm_pwm_score_strings(["ACGTACGT"], pssm)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_many_sequences(self):
        pssm = _make_simple_pssm()
        seqs = ["ACGTACGT"] * 100
        result = _pymisha.pm_pwm_score_strings(seqs, pssm)
        assert result.shape == (100,)
        # All identical sequences should get identical scores
        assert np.allclose(result, result[0])

    def test_n_characters(self):
        pssm = _make_simple_pssm()
        seqs = ["NNNNNNNN"]
        result = _pymisha.pm_pwm_score_strings(seqs, pssm, 0.01, "lse")
        assert result.shape == (1,)
        # N's should produce some score (not crash)
        assert np.isfinite(result[0])

    def test_lowercase_input(self):
        pssm = _make_simple_pssm()
        seqs_upper = ["ACGTACGT"]
        seqs_lower = ["acgtacgt"]
        result_upper = _pymisha.pm_pwm_score_strings(seqs_upper, pssm, 0.01, "lse")
        result_lower = _pymisha.pm_pwm_score_strings(seqs_lower, pssm, 0.01, "lse")
        np.testing.assert_allclose(result_upper, result_lower)

    def test_transposed_pssm(self):
        """Test that 4xL PSSM also works."""
        pssm = _make_simple_pssm().T  # 4xL
        assert pssm.shape == (4, 4)
        seqs = ["ACGT"]
        result = _pymisha.pm_pwm_score_strings(seqs, pssm, 0.01, "lse")
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_invalid_mode_raises(self):
        pssm = _make_simple_pssm()
        with pytest.raises(ValueError, match="Unknown mode"):
            _pymisha.pm_pwm_score_strings(["ACGT"], pssm, 0.01, "invalid_mode")

    def test_spat_factor(self):
        """Test with spatial weighting factor."""
        pssm = _make_simple_pssm()
        seqs = ["ACGTACGT"]
        spat = np.ones(10, dtype=np.float64)
        result = _pymisha.pm_pwm_score_strings(
            seqs, pssm, 0.01, "lse", True, 0, 0.0, False, spat, 1
        )
        assert result.shape == (1,)
        assert np.isfinite(result[0])
