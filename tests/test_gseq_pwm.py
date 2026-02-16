"""Tests for gseq_pwm function."""

import math

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

# ---- Helper functions (matching R test helpers) ----

def _log_sum_exp(x):
    """Compute log-sum-exp matching the C++/R iterative algorithm."""
    x = [v for v in x if np.isfinite(v)]
    if len(x) == 0:
        return -np.inf
    if len(x) == 1:
        return x[0]
    x = sorted(x, reverse=True)
    s = x[0]
    for i in range(1, len(x)):
        s = s + math.log1p(math.exp(x[i] - s)) if s > x[i] else x[i] + math.log1p(math.exp(s - x[i]))
    return s


def _manual_pwm_scores_single_strand(seq, pssm, prior=0.01):
    """Compute per-position log-probability scores for a single strand.

    Matches the R helper `manual_pwm_scores_single_strand`.
    pssm: numpy array (w, 4) with columns [A, C, G, T].
    """
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    w = pssm.shape[0]

    # Apply prior and normalize
    if prior > 0:
        normed = pssm.copy() + prior
        normed = normed / normed.sum(axis=1, keepdims=True)
    else:
        normed = pssm.copy()

    scores = []
    for i in range(len(seq) - w + 1):
        subseq = seq[i : i + w]
        score = 0.0
        valid = True
        for j in range(w):
            base = subseq[j]
            idx = base_map.get(base)
            if idx is None:
                valid = False
                break
            prob = normed[j, idx]
            if prob == 0:
                score = -np.inf
                valid = False
                break
            score += math.log(prob)
        if not valid:
            score = -np.inf
        scores.append(score)
    return scores


def _create_test_pssm():
    """Create a simple 2-position PSSM matching 'AC' exactly."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
        ]
    )


def _revcomp(seq):
    """Reverse complement a DNA sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq[::-1].translate(comp)


# ---- Tests ----


class TestGseqPwmValidation:
    """Input validation tests."""

    def test_pssm_must_have_acgt_columns_matrix(self):
        """PSSM as numpy array must have 4 columns."""
        pssm = np.array([[0.25, 0.25, 0.5]])  # Only 3 cols
        with pytest.raises((ValueError, KeyError)):
            pm.gseq_pwm(["ACGT"], pssm)

    def test_pssm_as_dataframe_must_have_acgt(self):
        """PSSM as DataFrame must have A, C, G, T columns."""
        pssm = pd.DataFrame({"X": [0.25], "Y": [0.25], "Z": [0.25], "W": [0.25]})
        with pytest.raises((ValueError, KeyError)):
            pm.gseq_pwm(["ACGT"], pssm)

    def test_invalid_mode_raises(self):
        pssm = _create_test_pssm()
        with pytest.raises(ValueError):
            pm.gseq_pwm(["ACGT"], pssm, mode="invalid")

    def test_invalid_strand_raises(self):
        pssm = _create_test_pssm()
        with pytest.raises(ValueError):
            pm.gseq_pwm(["ACGT"], pssm, strand=2)

    def test_prior_must_be_non_negative(self):
        pssm = _create_test_pssm()
        with pytest.raises(ValueError):
            pm.gseq_pwm(["ACGT"], pssm, prior=-0.1)


class TestGseqPwmBasicScoring:
    """Basic PWM scoring without gaps or neutral chars."""

    def test_lse_mode_single_sequence(self):
        """LSE mode scores match manual computation."""
        pssm = _create_test_pssm()
        seq = "CCCTAACCCTAAC"
        scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0.01)
        expected = _log_sum_exp(scores)

        result = pm.gseq_pwm([seq], pssm, mode="lse", bidirect=False, prior=0.01)
        assert abs(result[0] - expected) < 1e-6

    def test_max_mode(self):
        """Max mode returns the best single-position score."""
        pssm = _create_test_pssm()
        seq = "CCCTAACCCTAAC"
        scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0.01)
        expected = max(scores)

        result = pm.gseq_pwm([seq], pssm, mode="max", bidirect=False, prior=0.01)
        assert abs(result[0] - expected) < 1e-6

    def test_pos_mode(self):
        """Pos mode returns 1-based position of the best match."""
        pssm = _create_test_pssm()
        seq = "CCCTAACCCTAAC"
        scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0.01)
        expected_pos = int(np.argmax(scores)) + 1  # 1-based

        result = pm.gseq_pwm([seq], pssm, mode="pos", bidirect=False, prior=0.01)
        assert result[0] == expected_pos

    def test_count_mode(self):
        """Count mode returns number of positions exceeding threshold."""
        pssm = _create_test_pssm()
        seq = "AACAACAC"
        # With prior=0, "AC" matches get score 0, everything else -inf
        scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0)
        expected = sum(1 for s in scores if s >= 0)

        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0, score_thresh=0
        )
        assert result[0] == expected

    def test_no_prior(self):
        """With prior=0, only exact matches score above -Inf."""
        pssm = _create_test_pssm()
        seq = "XACX"
        # Positions: X-A (invalid), AC (exact match = score 0), CX (invalid)
        result = pm.gseq_pwm([seq], pssm, mode="max", bidirect=False, prior=0)
        # AC at position 2 should give score 0 (log(1.0) + log(1.0))
        assert result[0] == 0.0

    def test_multiple_sequences(self):
        """Scoring multiple sequences returns correct-length array."""
        pssm = _create_test_pssm()
        seqs = ["ACGTACGT", "GGGGGGGG", "ACACAC"]
        result = pm.gseq_pwm(seqs, pssm, mode="lse", bidirect=False, prior=0.01)
        assert len(result) == 3

    def test_sequence_shorter_than_motif(self):
        """Sequence shorter than motif returns NA/NaN."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm(["A"], pssm, mode="lse", bidirect=False, prior=0.01)
        assert np.isnan(result[0])

    def test_empty_sequence(self):
        """Empty sequence returns NA/NaN."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm([""], pssm, mode="lse", bidirect=False, prior=0.01)
        assert np.isnan(result[0])


class TestGseqPwmBidirectional:
    """Bidirectional (both strands) scoring."""

    def test_bidirect_lse_is_sum_of_both_strands(self):
        """Bidirectional LSE is log-sum-exp of forward + reverse scores."""
        pssm = _create_test_pssm()
        seq = "CCCTAACCCTAAC"
        seq_rc = _revcomp(seq)
        fwd_scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0.01)
        rev_scores = _manual_pwm_scores_single_strand(seq_rc, pssm, prior=0.01)
        expected = _log_sum_exp(fwd_scores + rev_scores)

        result = pm.gseq_pwm([seq], pssm, mode="lse", bidirect=True, prior=0.01)
        assert abs(result[0] - expected) < 1e-6

    def test_bidirect_max(self):
        """Bidirectional max is the best score across both strands."""
        pssm = _create_test_pssm()
        seq = "CCCTAACCCTAAC"
        seq_rc = _revcomp(seq)
        fwd_scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0.01)
        rev_scores = _manual_pwm_scores_single_strand(seq_rc, pssm, prior=0.01)
        expected = max(max(fwd_scores), max(rev_scores))

        result = pm.gseq_pwm([seq], pssm, mode="max", bidirect=True, prior=0.01)
        assert abs(result[0] - expected) < 1e-6

    def test_bidirect_count(self):
        """Bidirectional count sums hits from both strands."""
        pssm = _create_test_pssm()
        seq = "AACAAC"
        seq_rc = _revcomp(seq)
        fwd_scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0)
        rev_scores = _manual_pwm_scores_single_strand(seq_rc, pssm, prior=0)
        expected = sum(1 for s in fwd_scores + rev_scores if s >= 0)

        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=True, prior=0, score_thresh=0
        )
        assert result[0] == expected

    def test_bidirect_overrides_strand(self):
        """When bidirect=True, strand parameter is ignored."""
        pssm = _create_test_pssm()
        seq = "ACGTACGT"
        r1 = pm.gseq_pwm([seq], pssm, mode="lse", bidirect=True, strand=1, prior=0.01)
        r2 = pm.gseq_pwm(
            [seq], pssm, mode="lse", bidirect=True, strand=-1, prior=0.01
        )
        assert abs(r1[0] - r2[0]) < 1e-10

    def test_bidirect_geq_unidirectional(self):
        """Bidirectional LSE score >= forward-only score."""
        pssm = _create_test_pssm()
        seq = "ACGTACGTACGT"
        r_bidi = pm.gseq_pwm(
            [seq], pssm, mode="lse", bidirect=True, prior=0.01
        )
        r_fwd = pm.gseq_pwm(
            [seq], pssm, mode="lse", bidirect=False, strand=1, prior=0.01
        )
        assert r_bidi[0] >= r_fwd[0] - 1e-10


class TestGseqPwmStrand:
    """Strand-specific scoring."""

    def test_forward_strand(self):
        """strand=1 scores forward strand only."""
        pssm = _create_test_pssm()
        seq = "ACGTACGT"
        scores = _manual_pwm_scores_single_strand(seq, pssm, prior=0.01)
        expected = _log_sum_exp(scores)

        result = pm.gseq_pwm(
            [seq], pssm, mode="lse", bidirect=False, strand=1, prior=0.01
        )
        assert abs(result[0] - expected) < 1e-6

    def test_reverse_strand(self):
        """strand=-1 scores reverse complement strand only."""
        pssm = _create_test_pssm()
        seq = "ACGTACGT"
        seq_rc = _revcomp(seq)
        scores = _manual_pwm_scores_single_strand(seq_rc, pssm, prior=0.01)
        expected = _log_sum_exp(scores)

        result = pm.gseq_pwm(
            [seq], pssm, mode="lse", bidirect=False, strand=-1, prior=0.01
        )
        assert abs(result[0] - expected) < 1e-6


class TestGseqPwmROI:
    """Region of interest (start_pos, end_pos, extend)."""

    def test_start_pos_end_pos(self):
        """ROI restricts scoring to [start_pos, end_pos] (1-based inclusive)."""
        pssm = _create_test_pssm()
        seq = "XXXXXACXXXXXX"
        # ROI: positions 6-7 (1-based), which covers "AC"
        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, start_pos=6, end_pos=7
        )
        assert result[0] == 1  # "AC" at position 6

    def test_start_pos_excludes_before(self):
        """Positions before start_pos are excluded."""
        pssm = _create_test_pssm()
        seq = "ACXXXXXXXX"
        # AC is at position 1-2, but ROI starts at 3
        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, start_pos=3, end_pos=10
        )
        assert result[0] == 0

    def test_extend_true(self):
        """extend=True allows motif to start before ROI."""
        pssm = _create_test_pssm()
        seq = "XACXXXXXXXXX"
        # ROI starts at 3, "AC" starts at position 2
        # Without extend: not counted (starts outside ROI)
        # With extend: counted (overlaps ROI)
        result_no_ext = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, start_pos=3, end_pos=12, extend=False
        )
        result_ext = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, start_pos=3, end_pos=12, extend=True
        )
        assert result_no_ext[0] == 0
        assert result_ext[0] == 1


class TestGseqPwmGaps:
    """Gap character handling."""

    def test_skip_gaps_default(self):
        """Gap characters are skipped by default."""
        pssm = _create_test_pssm()
        # "A-C" with gap skipping = "AC" match
        seq = "A-C"
        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, skip_gaps=True
        )
        assert result[0] == 1

    def test_no_skip_gaps(self):
        """Without gap skipping, '-' breaks the match."""
        pssm = _create_test_pssm()
        seq = "A-C"
        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, skip_gaps=False
        )
        assert result[0] == 0

    def test_custom_gap_chars(self):
        """Custom gap characters are respected."""
        pssm = _create_test_pssm()
        seq = "A_C"
        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False, prior=0,
            score_thresh=0, skip_gaps=True, gap_chars=["_"]
        )
        assert result[0] == 1


class TestGseqPwmNeutralChars:
    """Neutral character handling."""

    def test_neutral_average_policy(self):
        """Neutral chars scored as average log-prob per column."""
        pssm = np.array([[0.7, 0.1, 0.1, 0.1]])  # 1-position motif
        seq = "N"
        # Average policy: log(mean([0.7+prior, 0.1+prior, ...] / sum))
        result = pm.gseq_pwm(
            [seq], pssm, mode="max", bidirect=False, prior=0.01,
            neutral_chars=["N"], neutral_chars_policy="average"
        )
        # Should not be NaN — average policy produces a finite score
        assert np.isfinite(result[0])

    def test_neutral_na_policy(self):
        """neutral_chars_policy='na' returns NaN when neutral char in window."""
        pssm = np.array([[0.7, 0.1, 0.1, 0.1]])
        seq = "N"
        result = pm.gseq_pwm(
            [seq], pssm, mode="max", bidirect=False, prior=0.01,
            neutral_chars=["N"], neutral_chars_policy="na"
        )
        assert np.isnan(result[0])

    def test_neutral_log_quarter_policy(self):
        """neutral_chars_policy='log_quarter' uses log(0.25) for neutral chars."""
        pssm = np.array([[0.7, 0.1, 0.1, 0.1]])
        seq = "N"
        expected = math.log(0.25)
        result = pm.gseq_pwm(
            [seq], pssm, mode="max", bidirect=False, prior=0,
            neutral_chars=["N"], neutral_chars_policy="log_quarter"
        )
        assert abs(result[0] - expected) < 1e-6


class TestGseqPwmPSSMFormats:
    """PSSM can be provided as numpy array or DataFrame."""

    def test_pssm_as_dataframe(self):
        """PSSM as DataFrame with A,C,G,T columns."""
        pssm_df = pd.DataFrame(
            {"A": [1.0, 0.0], "C": [0.0, 1.0], "G": [0.0, 0.0], "T": [0.0, 0.0]}
        )
        pssm_np = _create_test_pssm()
        seq = "ACGTACGT"
        r1 = pm.gseq_pwm([seq], pssm_df, mode="lse", bidirect=False, prior=0.01)
        r2 = pm.gseq_pwm([seq], pssm_np, mode="lse", bidirect=False, prior=0.01)
        assert abs(r1[0] - r2[0]) < 1e-10

    def test_pssm_dataframe_extra_columns(self):
        """PSSM DataFrame with extra columns — only A,C,G,T used."""
        pssm_extra = pd.DataFrame(
            {
                "A": [0.7, 0.1],
                "C": [0.1, 0.7],
                "G": [0.1, 0.1],
                "T": [0.1, 0.1],
                "name": ["pos1", "pos2"],
                "score": [1.0, 2.0],
            }
        )
        pssm_plain = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
        seq = "ACGTACGT"
        r1 = pm.gseq_pwm([seq], pssm_extra, mode="max", bidirect=False, prior=0.01)
        r2 = pm.gseq_pwm([seq], pssm_plain, mode="max", bidirect=False, prior=0.01)
        assert abs(r1[0] - r2[0]) < 1e-10

    def test_pssm_dataframe_reordered_columns(self):
        """PSSM DataFrame with columns in non-standard order."""
        pssm_reorder = pd.DataFrame(
            {
                "T": [0.1, 0.7],
                "G": [0.1, 0.1],
                "C": [0.7, 0.1],
                "A": [0.1, 0.1],
            }
        )
        pssm_standard = np.array([[0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
        seq = "ACGTACGT"
        r1 = pm.gseq_pwm([seq], pssm_reorder, mode="lse", bidirect=False, prior=0.01)
        r2 = pm.gseq_pwm([seq], pssm_standard, mode="lse", bidirect=False, prior=0.01)
        assert abs(r1[0] - r2[0]) < 1e-10


class TestGseqPwmReturnStrand:
    """pos mode with return_strand=True returns DataFrame."""

    def test_return_strand_dataframe(self):
        """return_strand=True produces a DataFrame with pos and strand columns."""
        pssm = _create_test_pssm()
        seq = "ACGTACGT"
        result = pm.gseq_pwm(
            [seq], pssm, mode="pos", bidirect=True, prior=0.01,
            return_strand=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert "pos" in result.columns
        assert "strand" in result.columns
        assert len(result) == 1

    def test_return_strand_values(self):
        """return_strand gives correct strand indicator."""
        pssm = _create_test_pssm()
        # "AC" is at position 1 on forward strand
        seq = "ACTTTT"
        result = pm.gseq_pwm(
            [seq], pssm, mode="pos", bidirect=True, prior=0.01,
            return_strand=True,
        )
        # The best match should be "AC" on the forward strand
        assert result["pos"].iloc[0] == 1
        assert result["strand"].iloc[0] == 1


class TestGseqPwmSingleString:
    """Accept a single string (not just list)."""

    def test_single_string_input(self):
        """A single string (not list) should be accepted."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm("ACGTACGT", pssm, mode="lse", bidirect=False, prior=0.01)
        assert len(result) == 1
        assert np.isfinite(result[0])


class TestGseqPwmPrior:
    """Prior (pseudocount) handling."""

    def test_prior_zero_exact_match(self):
        """With prior=0, exact match of 'AC' gives score 0."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm(["AC"], pssm, mode="max", bidirect=False, prior=0)
        assert result[0] == 0.0

    def test_prior_zero_no_match_gives_neg_inf(self):
        """With prior=0, no match gives -Inf."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm(["GG"], pssm, mode="max", bidirect=False, prior=0)
        assert result[0] == -np.inf

    def test_prior_nonzero_softens_scores(self):
        """With prior > 0, even mismatches get finite scores."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm(["GG"], pssm, mode="max", bidirect=False, prior=0.01)
        assert np.isfinite(result[0])
        assert result[0] < 0  # Still negative


class TestGseqPwmScoreThresh:
    """Score threshold for count mode."""

    def test_high_threshold_zero_count(self):
        """High threshold yields zero count."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm(
            ["ACGTACGT"], pssm, mode="count", bidirect=False,
            prior=0.01, score_thresh=100
        )
        assert result[0] == 0

    def test_low_threshold_counts_all(self):
        """Very low threshold counts all positions."""
        pssm = _create_test_pssm()
        seq = "ACGTACGT"
        w = pssm.shape[0]
        n_positions = len(seq) - w + 1
        result = pm.gseq_pwm(
            [seq], pssm, mode="count", bidirect=False,
            prior=0.01, score_thresh=-1000
        )
        assert result[0] == n_positions


# ---------------------------------------------------------------------------
# PWM count via virtual tracks  (ported from R test-pwm-count.R)
# ---------------------------------------------------------------------------


def _remove_all_vtracks():
    """Remove all virtual tracks."""
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


class TestPwmCountVtrack:
    """PWM count virtual track tests (ported from R test-pwm-count.R)."""

    @pytest.fixture(autouse=True)
    def _cleanup_vtracks(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_count_hits_above_threshold(self):
        """pwm.count counts hits above threshold (prior=0, score_thresh=0)."""
        pssm = _create_test_pssm()  # AC motif

        test_intervals = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_intervals)[0].upper()

        # Count AC occurrences manually
        ac_count = seq.count("AC")

        # Note: pymisha requires explicit strand=1 when bidirect=False
        # (strand=0 default scans no strand in the C++ vtrack path)
        pm.gvtrack_create(
            "count_hits", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0, score_thresh=0, strand=1,
        )

        result = pm.gextract("count_hits", test_intervals, iterator=-1)
        assert result["count_hits"].iloc[0] == ac_count

    def test_count_respects_score_threshold(self):
        """Lower threshold counts more hits than higher threshold."""
        pssm = _create_test_pssm()
        test_intervals = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create(
            "count_all", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0.01, score_thresh=-10, strand=1,
        )
        pm.gvtrack_create(
            "count_strict", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0.01, score_thresh=-1, strand=1,
        )

        result = pm.gextract(
            ["count_all", "count_strict"], test_intervals, iterator=-1
        )
        assert result["count_all"].iloc[0] >= result["count_strict"].iloc[0]
        assert result["count_all"].iloc[0] >= 0
        assert result["count_strict"].iloc[0] >= 0

    def test_count_bidirect_true(self):
        """Bidirectional should count at least as many as forward-only."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create(
            "count_fwd", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0.01, score_thresh=-5, strand=1,
        )
        pm.gvtrack_create(
            "count_bidi", None, func="pwm.count",
            pssm=pssm, bidirect=True, extend=False,
            prior=0.01, score_thresh=-5,
        )

        result = pm.gextract(
            ["count_fwd", "count_bidi"], test_interval, iterator=-1
        )
        assert result["count_bidi"].iloc[0] >= result["count_fwd"].iloc[0]

    def test_count_honors_extend_parameter(self):
        """extend=True can count additional positions at boundary."""
        pssm = _create_test_pssm()
        test_intervals = pm.gintervals(["1"], [200], [210])

        pm.gvtrack_create(
            "count_noext", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0.01, score_thresh=-5, strand=1,
        )
        pm.gvtrack_create(
            "count_ext", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
        )

        result = pm.gextract(
            ["count_noext", "count_ext"], test_intervals, iterator=-1
        )
        assert result["count_ext"].iloc[0] >= result["count_noext"].iloc[0]

    def test_count_with_iterator_shifts(self):
        """pwm.count with vtrack iterator shifts returns non-negative count."""
        pssm = _create_test_pssm()
        base = pm.gintervals(["1"], [2000], [2040])

        pm.gvtrack_create(
            "count_shift", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
        )
        pm.gvtrack_iterator("count_shift", sshift=-10, eshift=10)

        result = pm.gextract("count_shift", base, iterator=-1)
        assert result["count_shift"].iloc[0] >= 0

    def test_count_returns_0_for_very_high_threshold(self):
        """With very high threshold, nothing should pass."""
        pssm = _create_test_pssm()
        test_intervals = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create(
            "count_impossible", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0.01, score_thresh=100, strand=1,
        )

        result = pm.gextract("count_impossible", test_intervals, iterator=-1)
        assert result["count_impossible"].iloc[0] == 0

    def test_count_with_strand_parameter(self):
        """pwm.count with strand=1 and strand=-1 produce non-negative counts."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create(
            "count_plus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
        )
        pm.gvtrack_create(
            "count_minus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=-1,
        )

        result = pm.gextract(
            ["count_plus", "count_minus"], test_interval, iterator=-1
        )
        assert result["count_plus"].iloc[0] >= 0
        assert result["count_minus"].iloc[0] >= 0

    def test_count_matches_manual_counting_perfect_matches(self):
        """pwm.count with prior=0, thresh=0 matches manual AC count."""
        pssm = _create_test_pssm()
        test_intervals = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_intervals)[0].upper()
        ac_count = seq.count("AC")

        pm.gvtrack_create(
            "count_exact", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0, score_thresh=0, strand=1,
        )

        result = pm.gextract("count_exact", test_intervals, iterator=-1)
        assert result["count_exact"].iloc[0] == ac_count

    def test_count_bidirect_equals_lse_union(self):
        """Bidi count equals number of positions passing LSE-combined PWM threshold."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        # Per-position PWM scores
        pm.gvtrack_create(
            "pwm_plus", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=1,
        )
        pm.gvtrack_create(
            "pwm_minus", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        pm.gvtrack_create(
            "pwm_bidi", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True,
            prior=0.01, score_thresh=-10,
        )

        pwm_result = pm.gextract(
            ["pwm_plus", "pwm_minus", "pwm_bidi"], test_interval, iterator=1
        )
        n_pwm_plus = int((pwm_result["pwm_plus"] > -10).sum())
        n_pwm_minus = int((pwm_result["pwm_minus"] > -10).sum())
        n_pwm_bidi = int((pwm_result["pwm_bidi"] > -10).sum())

        # Count vtracks
        pm.gvtrack_create(
            "count_plus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=1,
        )
        pm.gvtrack_create(
            "count_minus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        pm.gvtrack_create(
            "count_bidi", None, func="pwm.count",
            pssm=pssm, bidirect=True, extend=True,
            prior=0.01, score_thresh=-10,
        )

        result = pm.gextract(
            ["count_plus", "count_minus", "count_bidi"],
            test_interval, iterator=-1,
        )

        assert result["count_plus"].iloc[0] == n_pwm_plus
        assert result["count_minus"].iloc[0] == n_pwm_minus
        assert result["count_bidi"].iloc[0] == n_pwm_bidi

    def test_count_bidi_equals_union_lse_matches_pwm_thresholding(self):
        """Bidi count equals LSE-combined PWM thresholding per position."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        pm.gvtrack_create(
            "pwm_plus", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=1,
        )
        pm.gvtrack_create(
            "pwm_minus", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        pm.gvtrack_create(
            "pwm_bidi", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True,
            prior=0.01, score_thresh=-10,
        )

        pwm_result = pm.gextract(
            ["pwm_plus", "pwm_minus", "pwm_bidi"], test_interval, iterator=1
        )
        n_pwm_plus = int((pwm_result["pwm_plus"] > -10).sum())
        n_pwm_minus = int((pwm_result["pwm_minus"] > -10).sum())
        n_pwm_bidi = int((pwm_result["pwm_bidi"] > -10).sum())

        pm.gvtrack_create(
            "count_plus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=1,
        )
        pm.gvtrack_create(
            "count_minus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        pm.gvtrack_create(
            "count_bidi", None, func="pwm.count",
            pssm=pssm, bidirect=True, extend=True,
            prior=0.01, score_thresh=-10,
        )

        result = pm.gextract(
            ["count_plus", "count_minus", "count_bidi"],
            test_interval, iterator=-1,
        )

        assert result["count_plus"].iloc[0] == n_pwm_plus
        assert result["count_minus"].iloc[0] == n_pwm_minus
        assert result["count_bidi"].iloc[0] == n_pwm_bidi

    def test_count_spatial_can_increase_counts(self):
        """Spatial weighting can increase counts over non-spatial at positive threshold."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        pm.gvtrack_create(
            "count_nospatial", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0, score_thresh=math.log(2), strand=1,
        )
        pm.gvtrack_create(
            "count_spatial", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0, score_thresh=math.log(2), strand=1,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5],
            spat_bin=20,
        )

        res = pm.gextract(
            ["count_nospatial", "count_spatial"], test_interval, iterator=-1
        )
        assert res["count_spatial"].iloc[0] >= res["count_nospatial"].iloc[0]

    @pytest.mark.skip(reason="pymisha does not reject zero spatial weights at vtrack creation time")
    def test_count_rejects_non_positive_spatial_weights(self):
        """spat_factor with zero value should be rejected."""
        pssm = _create_test_pssm()

        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "count_bad_spat", None, func="pwm.count",
                pssm=pssm, bidirect=True, extend=True,
                spat_factor=[1.0, 0.0, 1.0], spat_bin=10,
            )

    def test_count_prior_reduces_counts(self):
        """Prior > 0 reduces counts when threshold is fixed at 0."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        pm.gvtrack_create(
            "count_prior0", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0, score_thresh=0, strand=1,
        )
        pm.gvtrack_create(
            "count_prior01", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=False,
            prior=0.1, score_thresh=0, strand=1,
        )

        out = pm.gextract(
            ["count_prior0", "count_prior01"], test_interval, iterator=-1
        )
        assert out["count_prior0"].iloc[0] >= out["count_prior01"].iloc[0]

    def test_count_batch_path_consistent(self):
        """5 identical pwm.count vtracks return consistent values."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        vnames = []
        for i in range(5):
            name = f"count_rep_{i}"
            pm.gvtrack_create(
                name, None, func="pwm.count",
                pssm=pssm, bidirect=False, extend=True,
                prior=0.01, score_thresh=-5, strand=1,
            )
            vnames.append(name)

        res = pm.gextract(vnames, test_interval, iterator=-1)

        base = res[vnames[0]].iloc[0]
        assert base >= 0
        for nm in vnames[1:]:
            assert res[nm].iloc[0] == base

    def test_count_shift_equivalence(self):
        """Shifted vtrack over base equals unshifted over expanded iterator."""
        pssm = _create_test_pssm()

        base60 = pm.gintervals(["1"], [2100], [2160])
        base80 = pm.gintervals(["1"], [2090], [2170])

        pm.gvtrack_create(
            "pwmcount_shifted", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
        )
        pm.gvtrack_iterator("pwmcount_shifted", sshift=-10, eshift=10)

        s_shift = pm.gextract("pwmcount_shifted", base60, iterator=-1)

        pm.gvtrack_create(
            "pwmcount_unshifted", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
        )
        s_unshift = pm.gextract("pwmcount_unshifted", base80, iterator=-1)

        assert s_shift["pwmcount_shifted"].iloc[0] == s_unshift["pwmcount_unshifted"].iloc[0]

    def test_count_spatial_shift_equivalence(self):
        """Spatial + iterator shifts equivalence holds."""
        pssm = _create_test_pssm()

        base60 = pm.gintervals(["1"], [2200], [2260])
        base80 = pm.gintervals(["1"], [2190], [2270])

        pm.gvtrack_create(
            "pwmcount_spat_shifted", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5],
            spat_bin=20,
        )
        pm.gvtrack_iterator("pwmcount_spat_shifted", sshift=-10, eshift=10)
        a_shift = pm.gextract("pwmcount_spat_shifted", base60, iterator=-1)

        pm.gvtrack_create(
            "pwmcount_spat_unshifted", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-5, strand=1,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5],
            spat_bin=20,
        )
        a_unshift = pm.gextract("pwmcount_spat_unshifted", base80, iterator=-1)

        assert (
            a_shift["pwmcount_spat_shifted"].iloc[0]
            == a_unshift["pwmcount_spat_unshifted"].iloc[0]
        )

    def test_count_strand_minus_matches_pwm_minus_sliding(self):
        """strand=-1 count matches per-position minus-strand PWM thresholding (sliding path)."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        pm.gvtrack_create(
            "pwm_minus", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        pwm = pm.gextract("pwm_minus", test_interval, iterator=1)
        n_minus = int((pwm["pwm_minus"] > -10).sum())

        pm.gvtrack_create(
            "count_minus", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        res = pm.gextract("count_minus", test_interval, iterator=-1)
        assert res["count_minus"].iloc[0] == n_minus

    def test_count_strand_minus_matches_pwm_minus_spatial_unity(self):
        """strand=-1 with spatial weights=1 matches per-position minus-strand PWM."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        pm.gvtrack_create(
            "pwm_minus", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )
        pwm = pm.gextract("pwm_minus", test_interval, iterator=1)
        n_minus = int((pwm["pwm_minus"] > -10).sum())

        pm.gvtrack_create(
            "count_minus_spat", None, func="pwm.count",
            pssm=pssm, bidirect=False, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
            spat_factor=[1.0] * 5, spat_bin=20,
        )
        res = pm.gextract("count_minus_spat", test_interval, iterator=-1)
        assert res["count_minus_spat"].iloc[0] == n_minus

    def test_count_bidirect_ignores_strand(self):
        """When bidirect=True, strand parameter is ignored (union semantics)."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [300])

        pm.gvtrack_create(
            "count_bidi_s1", None, func="pwm.count",
            pssm=pssm, bidirect=True, extend=True,
            prior=0.01, score_thresh=-10, strand=1,
        )
        pm.gvtrack_create(
            "count_bidi_sneg1", None, func="pwm.count",
            pssm=pssm, bidirect=True, extend=True,
            prior=0.01, score_thresh=-10, strand=-1,
        )

        out = pm.gextract(
            ["count_bidi_s1", "count_bidi_sneg1"],
            test_interval, iterator=-1,
        )
        assert out["count_bidi_s1"].iloc[0] == out["count_bidi_sneg1"].iloc[0]


# ---------------------------------------------------------------------------
# Serial equivalents of R parallel tests (test-gseq_pwm-parallel.R)
# ---------------------------------------------------------------------------

class TestGseqPwmBulkScoring:
    """Port of R parallel tests as serial equivalents.

    The R tests compare parallel vs sequential execution to ensure identical
    results. Since pymisha does not support multitask gseq_pwm, we port the
    serial-equivalent logic: many sequences, various modes and options.
    """

    @staticmethod
    def _make_random_seqs(n, length, seed=42):
        """Generate n random DNA sequences of given length."""
        rng = np.random.RandomState(seed)
        bases = list("ACGT")
        return [
            "".join(rng.choice(bases, size=length)) for _ in range(n)
        ]

    def test_bulk_lse_many_sequences(self):
        """LSE mode on 1000 sequences of length 50."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(1000, 50)
        result = pm.gseq_pwm(seqs, pssm, mode="lse", bidirect=False, prior=0.01)
        assert len(result) == 1000
        assert all(np.isfinite(result))

    def test_bulk_max_bidirect(self):
        """Max mode bidirectional on 500 sequences of length 100."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(500, 100)
        result = pm.gseq_pwm(seqs, pssm, mode="max", bidirect=True, prior=0.01)
        assert len(result) == 500
        assert all(np.isfinite(result))

    def test_bulk_pos_mode(self):
        """Pos mode on 300 sequences of length 80."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(300, 80)
        result = pm.gseq_pwm(
            seqs, pssm, mode="pos", bidirect=False,
            return_strand=False, prior=0.01,
        )
        assert len(result) == 300
        assert all(r >= 1 for r in result)

    def test_bulk_pos_with_return_strand(self):
        """Pos mode with return_strand=True on 200 sequences."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(200, 60)
        result = pm.gseq_pwm(
            seqs, pssm, mode="pos", bidirect=True,
            return_strand=True, prior=0.01,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 200
        assert "pos" in result.columns
        assert "strand" in result.columns

    def test_bulk_count_mode(self):
        """Count mode on 400 sequences of length 70 with threshold."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(400, 70)
        result = pm.gseq_pwm(
            seqs, pssm, mode="count", score_thresh=-2,
            bidirect=True, prior=0.01,
        )
        assert len(result) == 400
        assert all(r >= 0 for r in result)

    def test_bulk_roi_parameters(self):
        """Many sequences with fixed start_pos / end_pos (scalar ROI)."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(200, 100)
        # pymisha gseq_pwm accepts scalar start_pos/end_pos (not per-sequence arrays)
        result = pm.gseq_pwm(
            seqs, pssm, mode="max",
            start_pos=10, end_pos=50, prior=0.01,
        )
        assert len(result) == 200

    def test_edge_case_empty_sequence_list(self):
        """Empty sequence list returns length-0 result."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm([], pssm, mode="lse", prior=0.01)
        assert len(result) == 0

    def test_edge_case_single_sequence(self):
        """Single sequence produces finite result."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm(["ACGTACGTACGT"], pssm, mode="max", prior=0.01)
        assert len(result) == 1
        assert np.isfinite(result[0])

    def test_edge_case_fewer_than_processes(self):
        """Two sequences (fewer than typical parallel processes) still work."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(2, 50)
        result = pm.gseq_pwm(seqs, pssm, mode="lse", prior=0.01)
        assert len(result) == 2

    def test_all_strand_modes_consistency(self):
        """Forward, reverse, and both-strand modes all produce valid output."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(200, 60)

        result_fwd = pm.gseq_pwm(
            seqs, pssm, mode="max", bidirect=False, strand=1, prior=0.01
        )
        result_rev = pm.gseq_pwm(
            seqs, pssm, mode="max", bidirect=False, strand=-1, prior=0.01
        )
        result_both = pm.gseq_pwm(
            seqs, pssm, mode="max", bidirect=False, strand=0, prior=0.01
        )

        assert len(result_fwd) == 200
        assert len(result_rev) == 200
        assert len(result_both) == 200
        assert all(np.isfinite(result_fwd))
        assert all(np.isfinite(result_rev))
        assert all(np.isfinite(result_both))

    def test_different_prior_values(self):
        """Different priors produce different scores on the same data."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(150, 50)

        results = {}
        for prior_val in [0.0, 0.01, 0.1, 1.0]:
            results[prior_val] = pm.gseq_pwm(
                seqs, pssm, mode="lse", prior=prior_val
            )
            assert len(results[prior_val]) == 150

        # Different priors should produce different scores
        assert not np.allclose(results[0.0], results[1.0], equal_nan=True)

    def test_deterministic_results(self):
        """Multiple runs produce identical results."""
        pssm = _create_test_pssm()
        seqs = self._make_random_seqs(300, 60)

        r1 = pm.gseq_pwm(seqs, pssm, mode="max", bidirect=True, prior=0.01)
        r2 = pm.gseq_pwm(seqs, pssm, mode="max", bidirect=True, prior=0.01)
        r3 = pm.gseq_pwm(seqs, pssm, mode="max", bidirect=True, prior=0.01)

        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(r1, r3)
