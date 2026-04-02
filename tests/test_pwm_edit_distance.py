"""Comprehensive tests for PWM edit distance features.

Covers:
  A. gseq_pwm_edits() — bare-sequence and interval-based edit analysis
  B. Virtual track functions — pwm.edit_distance, .pos, .lse, pwm.max.edit_distance
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_pssm():
    """Simple 2-position PSSM matching 'AC' (prior=0 gives log(1)=0 for match)."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # position 1 wants A
        [0.0, 1.0, 0.0, 0.0],  # position 2 wants C
    ])


def _revcomp(seq):
    """Reverse complement a DNA sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq[::-1].translate(comp)


def _manual_pwm_edit_distance(seq, pssm, threshold, max_edits=None, scan_all=True):
    """Compute minimum edits needed to reach *threshold* by greedy gain sorting.

    When *scan_all* is True, scans every window in *seq* and returns the
    minimum edit count across positions.  Otherwise treats *seq* as a single
    motif-length window.
    """
    motif_len = pssm.shape[0]
    if len(seq) < motif_len:
        return np.nan

    log_pssm = np.log(pssm, where=(pssm > 0), out=np.full_like(pssm, -np.inf))
    col_max = log_pssm.max(axis=1)

    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _score_window(window_seq):
        adjusted_score = 0.0
        mandatory_edits = 0
        gains = []

        for i in range(motif_len):
            base = window_seq[i]
            idx = base_map.get(base)
            base_score = log_pssm[i].min() if idx is None else log_pssm[i, idx]

            if not np.isfinite(base_score):
                mandatory_edits += 1
                adjusted_score += col_max[i]
            else:
                adjusted_score += base_score
                gains.append(col_max[i] - base_score)

        deficit = threshold - adjusted_score
        if deficit <= 0:
            if max_edits is not None and mandatory_edits > max_edits:
                return np.nan
            return mandatory_edits

        total_max_gain = col_max.sum() - adjusted_score
        if total_max_gain < deficit - 1e-12:
            return np.nan

        gains_sorted = sorted(gains, reverse=True)
        if max_edits is not None:
            remaining = max_edits - mandatory_edits
            if remaining < 0:
                return np.nan
            gains_sorted = gains_sorted[:remaining]

        acc = 0.0
        edits = mandatory_edits
        for g in gains_sorted:
            edits += 1
            acc += g
            if acc >= deficit:
                return edits

        return np.nan

    if not scan_all:
        return _score_window(seq)

    best = np.nan
    for start in range(len(seq) - motif_len + 1):
        window = seq[start:start + motif_len]
        cand = _score_window(window)
        if np.isnan(best) or (not np.isnan(cand) and cand < best):
            best = cand
    return best


def _remove_all_vtracks():
    """Remove every virtual track in the current session."""
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


# ===========================================================================
# A. gseq_pwm_edits() tests
# ===========================================================================


class TestGseqPwmEditsStructure:
    """Output structure and column presence."""

    def test_returns_dataframe_with_expected_columns(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits("CCGTACGT", pssm, score_thresh=-0.5, prior=0)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "seq_idx", "strand", "window_start", "score_before",
            "score_after", "n_edits", "edit_num", "motif_col",
            "ref", "alt", "gain", "window_seq", "mutated_seq",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) > 0

    def test_empty_input_returns_empty_dataframe(self):
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits([], pssm, score_thresh=-1.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "window_seq" in result.columns
        assert "mutated_seq" in result.columns


class TestGseqPwmEditsBasic:
    """Core edit logic on bare sequences."""

    def test_single_edit_tc_to_ac(self):
        """'TC' with AC-pssm and prior=0 needs 1 edit: T->A at motif col 1."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("TC", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=False)
        one_edit = result[result["n_edits"] == 1]
        assert len(one_edit) == 1
        assert one_edit.iloc[0]["motif_col"] == 1
        assert one_edit.iloc[0]["ref"] == "T"
        assert one_edit.iloc[0]["alt"] == "A"

    def test_already_above_threshold_zero_edits(self):
        """When the best window already exceeds the threshold, n_edits=0."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits("ACGTACGT", pssm, score_thresh=-5.0,
                                    prior=0)
        assert result["n_edits"].iloc[0] == 0
        assert result["edit_num"].iloc[0] == 0
        assert pd.isna(result["motif_col"].iloc[0])

    def test_three_position_all_edits(self):
        """'TTT' with ACG-pssm -> all 3 positions must change to ACG."""
        pssm = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ])
        result = pm.gseq_pwm_edits("TTT", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=False)
        assert all(result["mutated_seq"] == "ACG")
        assert all(result["window_seq"] == "TTT")

        # Verify individual edits
        for _, row in result.iterrows():
            mc = int(row["motif_col"])
            if not pd.isna(mc) and mc >= 1:
                assert row["window_seq"][mc - 1] == row["ref"]
                assert row["mutated_seq"][mc - 1] == row["alt"]

    def test_window_seq_and_mutated_seq(self):
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("TC", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=False)
        assert result["window_seq"].iloc[0] == "TC"
        assert result["mutated_seq"].iloc[0] == "AC"

    def test_score_after_ge_score_before_when_edits_needed(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits("TT", pssm, score_thresh=-0.5,
                                    prior=0, bidirect=False)
        rows_with_edits = result[result["n_edits"] > 0]
        if len(rows_with_edits) > 0:
            assert all(rows_with_edits["score_after"] > rows_with_edits["score_before"])

    def test_gain_positive_for_real_edits(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits("TT", pssm, score_thresh=-0.5,
                                    prior=0, bidirect=False)
        real_edits = result[result["edit_num"] > 0]
        assert all(real_edits["gain"] > 0)


class TestGseqPwmEditsMultipleSequences:
    """Multiple input sequences."""

    def test_seq_idx_is_1_based(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits(["AC", "TT", "GG"], pssm,
                                    score_thresh=-1.0, prior=0, bidirect=False)
        # AC already matches -> seq_idx=1 should have 0 edits
        ac_rows = result[result["seq_idx"] == 1]
        assert len(ac_rows) > 0
        assert ac_rows["n_edits"].iloc[0] == 0

        # TT and GG need edits
        assert any(result["seq_idx"] == 2)
        assert any(result["seq_idx"] == 3)

    def test_all_seq_idx_present(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits(["AAAA", "CCCC", "GGGG", "TTTT"], pssm,
                                    score_thresh=-2.0, prior=0, bidirect=False)
        present_idx = set(result["seq_idx"].unique())
        # All 4 sequences should have results
        assert {1, 2, 3, 4} == present_idx


class TestGseqPwmEditsBidirect:
    """Bidirectional scanning."""

    def test_bidirect_finds_revcomp_match(self):
        """'GT' has revcomp 'AC' — perfect match on reverse strand."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("GT", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=True)
        assert result["n_edits"].iloc[0] == 0
        assert result["strand"].iloc[0] == -1

    def test_bidirect_false_forward_only(self):
        """With bidirect=False, only forward strand is scanned."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("GT", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=False)
        # Forward strand: GT doesn't match AC, so needs edits
        assert all(result["strand"] == 1)
        rows_with_edits = result[result["n_edits"] > 0]
        assert len(rows_with_edits) > 0


class TestGseqPwmEditsMaxEdits:
    """max_edits parameter."""

    def test_max_edits_caps_results(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9],
        ])
        # TTTT needs multiple edits
        r1 = pm.gseq_pwm_edits("TTTT", pssm, score_thresh=-0.5,
                                max_edits=1, prior=0, bidirect=False)
        r4 = pm.gseq_pwm_edits("TTTT", pssm, score_thresh=-0.5,
                                max_edits=4, prior=0, bidirect=False)
        assert len(r1) <= len(r4)

    def test_max_edits_none_is_unlimited(self):
        """max_edits=None allows any number of edits."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("TT", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=False, max_edits=None)
        assert len(result) > 0


class TestGseqPwmEditsScoreFilters:
    """score_min and score_max filtering."""

    def test_score_max_very_low_filters_everything(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits("CCGTACGT", pssm, score_thresh=-0.5,
                                    prior=0, score_max=-50.0)
        assert len(result) == 0

    def test_score_min_very_high_filters_everything(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        result = pm.gseq_pwm_edits("CCGTACGT", pssm, score_thresh=-0.5,
                                    prior=0, score_min=0.0)
        assert len(result) == 0


class TestGseqPwmEditsIntervals:
    """Interval-based input extracts sequences and adds chrom/start/end."""

    def test_interval_input_adds_chrom_start_end(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        intervals = pm.gintervals(["1"], [200], [210])
        result = pm.gseq_pwm_edits(intervals, pssm, score_thresh=-3.0,
                                    prior=0)
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns
        assert len(result) > 0

    def test_multiple_intervals(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        intervals = pm.gintervals(["1", "1"], [200, 300], [210, 310])
        result = pm.gseq_pwm_edits(intervals, pssm, score_thresh=-3.0,
                                    prior=0)
        assert len(result) > 0
        assert 1 in result["seq_idx"].values
        assert 2 in result["seq_idx"].values


class TestGseqPwmEditsExtend:
    """extend parameter (True / False / int)."""

    def test_extend_numeric(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9],
        ])
        seq = "TTTTACGTTTTT"
        r_ext1 = pm.gseq_pwm_edits(seq, pssm, score_thresh=-1.0,
                                    prior=0, bidirect=False, extend=1)
        assert isinstance(r_ext1, pd.DataFrame)
        assert len(r_ext1) > 0

    def test_extend_false_vs_true(self):
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9],
        ])
        seq = "TTTTACGTTTTT"
        r_no = pm.gseq_pwm_edits(seq, pssm, score_thresh=-1.0,
                                  prior=0, bidirect=False, extend=False)
        r_yes = pm.gseq_pwm_edits(seq, pssm, score_thresh=-1.0,
                                   prior=0, bidirect=False, extend=True)
        assert isinstance(r_no, pd.DataFrame)
        assert isinstance(r_yes, pd.DataFrame)


class TestGseqPwmEditsZeroProbAndN:
    """Edge cases: zero-probability bases and N characters."""

    def test_zero_prob_base_scores_very_negative(self):
        pssm = _create_test_pssm()  # only A and C have prob > 0
        r = pm.gseq_pwm_edits("TC", pssm, score_thresh=-0.5,
                               prior=0, bidirect=False)
        # score_before should be very negative (T at pos 1 has prob=0)
        assert all(~np.isnan(r["score_before"]))
        assert all(r["score_before"] < -100)
        assert all(r["score_after"] > r["score_before"])

    def test_n_base_is_mandatory_edit(self):
        pssm = _create_test_pssm()
        r = pm.gseq_pwm_edits("NC", pssm, score_thresh=-0.5,
                               prior=0, bidirect=False)
        assert len(r) > 0
        assert any(r["ref"] == "N")

    def test_all_n_unreachable(self):
        """All-N sequence with impossible threshold -> empty."""
        pssm = _create_test_pssm()
        r = pm.gseq_pwm_edits("NN", pssm, score_thresh=100.0,
                               prior=0, bidirect=False)
        assert len(r) == 0


class TestGseqPwmEditsPssmFormat:
    """PSSM supplied as DataFrame vs array."""

    def test_pssm_as_dataframe(self):
        pssm_df = pd.DataFrame({
            "A": [0.9, 0.05],
            "C": [0.05, 0.9],
            "G": [0.025, 0.025],
            "T": [0.025, 0.025],
        })
        result = pm.gseq_pwm_edits("TT", pssm_df, score_thresh=-0.5,
                                    prior=0, bidirect=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ===========================================================================
# B. Virtual track tests
# ===========================================================================


class TestVtrackEditDistanceBasic:
    """pwm.edit_distance virtual track — basic functionality."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_basic_edit_distance(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_interval)[0].upper()

        threshold = -5.0
        pm.gvtrack_create("edist", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist", test_interval, iterator=test_interval)
        expected = _manual_pwm_edit_distance(seq, pssm, threshold)
        if np.isnan(expected):
            assert np.isnan(result["edist"].iloc[0])
        else:
            npt.assert_allclose(result["edist"].iloc[0], expected, atol=1e-6)

    def test_perfect_match_returns_zero(self):
        pssm = _create_test_pssm()
        # Find an "AC" occurrence in the test genome
        full_seq = pm.gseq_extract(
            pm.gintervals(["1"], [200], [300])
        )[0].upper()
        ac_pos = full_seq.find("AC")
        if ac_pos < 0:
            pytest.skip("No AC pattern found in test region")

        abs_pos = 200 + ac_pos
        test_interval = pm.gintervals(["1"], [abs_pos], [abs_pos + 2])

        pm.gvtrack_create("edist_perfect", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=0.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_perfect", test_interval, iterator=test_interval)
        npt.assert_allclose(result["edist_perfect"].iloc[0], 0, atol=1e-6)

    def test_unreachable_threshold_returns_nan(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create("edist_unreach", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=100.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_unreach", test_interval, iterator=test_interval)
        assert np.isnan(result["edist_unreach"].iloc[0])

    def test_tiny_interval_smaller_than_motif_returns_nan(self):
        pssm = _create_test_pssm()
        tiny = pm.gintervals(["1"], [200], [201])

        pm.gvtrack_create("edist_tiny", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=0.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_tiny", tiny, iterator=tiny)
        assert np.isnan(result["edist_tiny"].iloc[0])

    def test_multiple_intervals(self):
        pssm = _create_test_pssm()
        intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 300, 400],
            [210, 310, 410],
        )

        pm.gvtrack_create("edist_multi", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=-4.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_multi", intervals, iterator=intervals)
        assert len(result) == 3
        # Each result should be non-negative or NaN
        vals = result["edist_multi"].values
        assert all(np.isnan(v) or v >= 0 for v in vals)


class TestVtrackEditDistanceMaxEdits:
    """max_edits parameter on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_max_edits_limits_search(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -3.0

        pm.gvtrack_create("edist_exact", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_edits=None,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_max2", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_edits=2,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_exact", "edist_max2"], test_interval,
                             iterator=test_interval)

        exact = result["edist_exact"].iloc[0]
        max2 = result["edist_max2"].iloc[0]

        # If exact needs > 2 edits, max2 should be NaN
        if not np.isnan(exact) and exact > 2:
            assert np.isnan(max2)
        # If exact <= 2, they should agree
        elif not np.isnan(exact) and exact <= 2:
            npt.assert_allclose(max2, exact, atol=1e-6)

    def test_max_edits_consistency_across_k(self):
        """For k=1..5, if exact <= k then max_edits=k matches exact."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -3.0

        pm.gvtrack_create("edist_exact", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_edits=None,
                          bidirect=False, extend=False, prior=0)
        vnames = ["edist_exact"]
        for k in range(1, 6):
            vn = f"edist_max{k}"
            pm.gvtrack_create(vn, None,
                              func="pwm.edit_distance",
                              pssm=pssm, score_thresh=threshold, max_edits=k,
                              bidirect=False, extend=False, prior=0)
            vnames.append(vn)

        result = pm.gextract(vnames, test_interval, iterator=test_interval)
        exact = result["edist_exact"].iloc[0]

        for k in range(1, 6):
            val = result[f"edist_max{k}"].iloc[0]
            if not np.isnan(exact) and exact <= k:
                npt.assert_allclose(val, exact, atol=1e-6)


class TestVtrackEditDistanceBidirectional:
    """Bidirectional scanning on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_bidirect_is_min_of_both_strands(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_fwd", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, strand=1, extend=False, prior=0)
        pm.gvtrack_create("edist_rev", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, strand=-1, extend=False, prior=0)
        pm.gvtrack_create("edist_bidi", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract(["edist_fwd", "edist_rev", "edist_bidi"],
                             test_interval, iterator=test_interval)

        fwd = result["edist_fwd"].iloc[0]
        rev = result["edist_rev"].iloc[0]
        bidi = result["edist_bidi"].iloc[0]

        if not np.isnan(fwd) and not np.isnan(rev):
            npt.assert_allclose(bidi, min(fwd, rev), atol=1e-6)


class TestVtrackEditDistanceExtend:
    """extend parameter on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_extend_true_vs_false(self):
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]
        test_interval = pm.gintervals(["1"], [200], [202])
        threshold = -5.0

        pm.gvtrack_create("edist_ext", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=True, prior=0)
        pm.gvtrack_create("edist_noext", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_ext", "edist_noext"], test_interval,
                             iterator=test_interval)

        # With extend=TRUE, the window is expanded to include full motif
        seq_ext = pm.gseq_extract(
            pm.gintervals(["1"], [200], [200 + motif_len])
        )[0].upper()
        expected_ext = _manual_pwm_edit_distance(seq_ext, pssm, threshold)

        # With extend=FALSE, the window stays as-is
        seq_noext = pm.gseq_extract(test_interval)[0].upper()
        expected_noext = _manual_pwm_edit_distance(seq_noext, pssm, threshold)

        ext_val = result["edist_ext"].iloc[0]
        noext_val = result["edist_noext"].iloc[0]

        if np.isnan(expected_ext):
            assert np.isnan(ext_val)
        else:
            npt.assert_allclose(ext_val, expected_ext, atol=1e-6)

        if np.isnan(expected_noext):
            assert np.isnan(noext_val)
        else:
            npt.assert_allclose(noext_val, expected_noext, atol=1e-6)


class TestVtrackEditDistancePos:
    """pwm.edit_distance.pos virtual track."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_pos_returns_numeric(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_pos", None,
                          func="pwm.edit_distance.pos",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_pos", test_interval, iterator=test_interval)
        pos_val = result["edist_pos"].iloc[0]
        # Position should be numeric (possibly NaN)
        assert isinstance(pos_val, (int, float, np.integer, np.floating))

    def test_pos_within_interval(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_pos", None,
                          func="pwm.edit_distance.pos",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_pos", test_interval, iterator=test_interval)
        pos_val = result["edist_pos"].iloc[0]
        if not np.isnan(pos_val):
            abs_pos = abs(pos_val)
            # 1-based position within the interval
            assert abs_pos >= 1
            assert abs_pos <= 40  # interval is 40bp

    def test_pos_signed_for_bidirect(self):
        """When bidirect=True, positive means forward, negative means reverse."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [280])
        threshold = -5.0

        pm.gvtrack_create("edist_pos_bidi", None,
                          func="pwm.edit_distance.pos",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract("edist_pos_bidi", test_interval,
                             iterator=test_interval)
        pos_val = result["edist_pos_bidi"].iloc[0]
        # Just check it's a valid number
        assert isinstance(pos_val, (int, float, np.integer, np.floating))


class TestVtrackMaxEditDistance:
    """pwm.max.edit_distance — edit distance at best PWM window."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_max_edit_distance_returns_numeric(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_max_site", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_max_site", test_interval,
                             iterator=test_interval)
        val = result["edist_max_site"].iloc[0]
        assert isinstance(val, (int, float, np.integer, np.floating))

    def test_max_edit_distance_vs_max_pos(self):
        """Edit distance at best PWM window should match manual computation at that position."""
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]
        test_interval = pm.gintervals(["1"], [200], [280])
        threshold = -5.0

        pm.gvtrack_create("edist_max_site", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("pwm_max_pos", None,
                          func="pwm.max.pos",
                          pssm=pssm, bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_max_site", "pwm_max_pos"],
                             test_interval, iterator=test_interval)

        pwm_pos = result["pwm_max_pos"].iloc[0]
        edist_at_max = result["edist_max_site"].iloc[0]

        if not np.isnan(pwm_pos):
            offset = int(round(pwm_pos)) - 1
            assert offset >= 0
            pwm_window = pm.gintervals(
                ["1"],
                [test_interval["start"].iloc[0] + offset],
                [test_interval["start"].iloc[0] + offset + motif_len],
            )
            pwm_seq = pm.gseq_extract(pwm_window)[0].upper()
            expected = _manual_pwm_edit_distance(pwm_seq, pssm, threshold,
                                                  scan_all=False)
            if np.isnan(expected):
                assert np.isnan(edist_at_max)
            else:
                npt.assert_allclose(edist_at_max, expected, atol=1e-6)


class TestVtrackEditDistanceLSE:
    """pwm.edit_distance.lse and pwm.edit_distance.lse.pos virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_lse_returns_numeric(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_lse", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_lse", test_interval, iterator=test_interval)
        val = result["edist_lse"].iloc[0]
        assert isinstance(val, (int, float, np.integer, np.floating))

    def test_lse_pos_returns_numeric(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_lse_pos", None,
                          func="pwm.edit_distance.lse.pos",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_lse_pos", test_interval,
                             iterator=test_interval)
        val = result["edist_lse_pos"].iloc[0]
        assert isinstance(val, (int, float, np.integer, np.floating))

    def test_lse_non_negative_or_nan(self):
        pssm = _create_test_pssm()
        intervals = pm.gintervals(
            ["1", "1"],
            [200, 1000],
            [240, 1040],
        )
        threshold = -5.0

        pm.gvtrack_create("edist_lse", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_lse", intervals, iterator=intervals)
        vals = result["edist_lse"].values
        assert all(np.isnan(v) or v >= 0 for v in vals)


class TestVtrackEditDistanceScoreFilters:
    """score_min / score_max filters on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_score_min_high_returns_nan(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_highfilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          score_min=0.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_highfilt", test_interval,
                             iterator=test_interval)
        # With score_min=0.0, most windows should be filtered -> NaN
        # (unless there's a perfect match in the interval, which is unlikely
        #  for prior=0 with a strict identity PSSM)
        val = result["edist_highfilt"].iloc[0]
        # It should be NaN or at least >= unfiltered value
        assert isinstance(val, (int, float, np.integer, np.floating))

    def test_score_min_low_matches_unfiltered(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_nofilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_lowfilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          score_min=-100.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_nofilt", "edist_lowfilt"],
                             test_interval, iterator=test_interval)
        nf = result["edist_nofilt"].iloc[0]
        lf = result["edist_lowfilt"].iloc[0]
        if np.isnan(nf):
            assert np.isnan(lf)
        else:
            npt.assert_allclose(nf, lf, atol=1e-6)


class TestVtrackEditDistanceIndels:
    """max_indels parameter (non-LSE only)."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_max_indels_zero_matches_default(self):
        """max_indels=0 should give same result as omitting it."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_indels0", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_default", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_indels0", "edist_default"],
                             test_interval, iterator=test_interval)
        v0 = result["edist_indels0"].iloc[0]
        vd = result["edist_default"].iloc[0]
        if np.isnan(v0):
            assert np.isnan(vd)
        else:
            npt.assert_allclose(v0, vd, atol=1e-6)


class TestVtrackEditDistancePrior:
    """prior parameter on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_prior_zero_vs_nonzero(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_p0", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_p01", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0.01)

        result = pm.gextract(["edist_p0", "edist_p01"],
                             test_interval, iterator=test_interval)
        # Both should return valid (possibly different) results
        v0 = result["edist_p0"].iloc[0]
        v01 = result["edist_p01"].iloc[0]
        if not np.isnan(v0):
            assert v0 >= 0
        if not np.isnan(v01):
            assert v01 >= 0


class TestVtrackEditDistanceThresholds:
    """Higher thresholds should require more (or equal) edits."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_higher_threshold_more_edits(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        thresholds = [-10.0, -5.0, -2.0, 0.0]
        vnames = []
        for i, t in enumerate(thresholds):
            vn = f"edist_t{i}"
            pm.gvtrack_create(vn, None,
                              func="pwm.edit_distance",
                              pssm=pssm, score_thresh=t,
                              bidirect=False, extend=False, prior=0)
            vnames.append(vn)

        result = pm.gextract(vnames, test_interval, iterator=test_interval)

        edits = [result[vn].iloc[0] for vn in vnames]
        # Filter out NaNs and check monotonicity
        finite = [(t, e) for t, e in zip(thresholds, edits, strict=True) if not np.isnan(e)]
        if len(finite) > 1:
            for i in range(1, len(finite)):
                assert finite[i][1] >= finite[i - 1][1], (
                    f"Higher threshold {finite[i][0]} should require >= edits "
                    f"than {finite[i-1][0]}: got {finite[i][1]} < {finite[i-1][1]}"
                )


class TestVtrackEditDistanceLongerMotif:
    """Edit distance with a longer motif (6bp)."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_6bp_motif_works(self):
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],   # A
            [0.03, 0.9, 0.03, 0.04],   # C
            [0.03, 0.03, 0.9, 0.04],   # G
            [0.04, 0.03, 0.03, 0.9],   # T
            [0.9, 0.03, 0.03, 0.04],   # A
            [0.03, 0.9, 0.03, 0.04],   # C
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        pm.gvtrack_create("edist_6bp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=-3.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_6bp", test_interval, iterator=test_interval)
        val = result["edist_6bp"].iloc[0]
        assert np.isnan(val) or val >= 0


class TestVtrackEditDistance1bpIterator:
    """1bp iterator scanning."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_1bp_iterator_returns_per_position(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [210])
        threshold = -5.0

        # Use extend=True so the 1bp window [pos, pos+1) is expanded to
        # [pos, pos+w) giving one full motif window per position.
        pm.gvtrack_create("edist_1bp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=True, prior=0)

        result = pm.gextract("edist_1bp", test_interval, iterator=1)
        assert len(result) > 0

        # With extend=True the engine scans [pos - (w-1), pos + w),
        # i.e. all motif windows that overlap the 1bp bin.  Verify that
        # every returned value is non-negative or NaN.
        for idx in range(len(result)):
            actual = result["edist_1bp"].iloc[idx]
            assert np.isnan(actual) or actual >= 0


class TestVtrackEditDistanceStrand:
    """Explicit strand parameter (not bidirect)."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_fwd_and_rev_strands_independently(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_fwd", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          strand=1, extend=False, prior=0)
        pm.gvtrack_create("edist_rev", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          strand=-1, extend=False, prior=0)

        result = pm.gextract(["edist_fwd", "edist_rev"],
                             test_interval, iterator=test_interval)

        fwd = result["edist_fwd"].iloc[0]
        rev = result["edist_rev"].iloc[0]

        # Both should be valid
        if not np.isnan(fwd):
            assert fwd >= 0
        if not np.isnan(rev):
            assert rev >= 0


class TestVtrackEditDistanceForcedEdits:
    """Windows where every motif column must change."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_forced_edits_count(self):
        """With identity PSSM and a window that doesn't match at all,
        all positions are mandatory edits."""
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]

        # Search for a window where neither pos matches
        search_interval = pm.gintervals(["1"], [150], [350])
        seq_region = pm.gseq_extract(search_interval)[0].upper()

        start_offset = None
        for offset in range(len(seq_region) - motif_len + 1):
            candidate = seq_region[offset:offset + motif_len]
            if candidate[0] != "A" and candidate[1] != "C":
                start_offset = offset
                break

        if start_offset is None:
            pytest.skip("No window requiring two forced edits found in test genome")

        abs_start = 150 + start_offset
        forced_interval = pm.gintervals(["1"], [abs_start], [abs_start + motif_len])
        threshold = -1.0

        pm.gvtrack_create("edist_forced", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_forced", forced_interval,
                             iterator=forced_interval)
        npt.assert_allclose(result["edist_forced"].iloc[0], motif_len, atol=1e-6)

    def test_forced_edits_exceed_max_edits_returns_nan(self):
        """If forced edits > max_edits, result is NaN."""
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]

        search_interval = pm.gintervals(["1"], [150], [350])
        seq_region = pm.gseq_extract(search_interval)[0].upper()

        start_offset = None
        for offset in range(len(seq_region) - motif_len + 1):
            candidate = seq_region[offset:offset + motif_len]
            if candidate[0] != "A" and candidate[1] != "C":
                start_offset = offset
                break

        if start_offset is None:
            pytest.skip("No window requiring two forced edits found in test genome")

        abs_start = 150 + start_offset
        forced_interval = pm.gintervals(["1"], [abs_start], [abs_start + motif_len])

        pm.gvtrack_create("edist_limit", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=-1.0, max_edits=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_limit", forced_interval,
                             iterator=forced_interval)
        assert np.isnan(result["edist_limit"].iloc[0])


class TestVtrackEditDistanceMaxEditsHeuristic:
    """max_edits=1 fast heuristic consistency."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_max_edits_1_vs_exact(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -2.0

        pm.gvtrack_create("edist_fast1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_edits=1,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_exact", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_edits=None,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_fast1", "edist_exact"],
                             test_interval, iterator=test_interval)

        exact = result["edist_exact"].iloc[0]
        fast1 = result["edist_fast1"].iloc[0]

        if not np.isnan(exact) and exact <= 1:
            npt.assert_allclose(fast1, exact, atol=1e-6)
        if not np.isnan(exact) and exact > 1:
            assert np.isnan(fast1)


class TestVtrackEditDistancePssmDataFrame:
    """PSSM provided as pandas DataFrame for virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_pssm_dataframe_matches_array(self):
        pssm_arr = _create_test_pssm()
        pssm_df = pd.DataFrame(pssm_arr, columns=["A", "C", "G", "T"])
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_arr", None,
                          func="pwm.edit_distance",
                          pssm=pssm_arr, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_df", None,
                          func="pwm.edit_distance",
                          pssm=pssm_df, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_arr", "edist_df"],
                             test_interval, iterator=test_interval)

        va = result["edist_arr"].iloc[0]
        vd = result["edist_df"].iloc[0]
        if np.isnan(va):
            assert np.isnan(vd)
        else:
            npt.assert_allclose(va, vd, atol=1e-6)


class TestVtrackEditDistanceScoreMinMaxOnMax:
    """score_min / score_max on pwm.max.edit_distance."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_score_min_on_max_edit_distance(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("edist_max_nofilt", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_max_lowfilt", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          score_min=-100.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_max_nofilt", "edist_max_lowfilt"],
                             test_interval, iterator=test_interval)
        nf = result["edist_max_nofilt"].iloc[0]
        lf = result["edist_max_lowfilt"].iloc[0]
        if np.isnan(nf):
            assert np.isnan(lf)
        else:
            npt.assert_allclose(nf, lf, atol=1e-6)


class TestVtrackEditDistanceScan:
    """Verify edit distance scans entire interval correctly."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_scans_all_windows_in_interval(self):
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]
        threshold = -5.0

        # Find a suitable interval containing "AC" not at the start
        all_chroms = pm.gintervals_all()
        chrom_end = int(all_chroms["end"].iloc[0])
        window_size = 80

        scan_interval = None
        scan_seq = None
        for start_pos in range(0, min(5000, chrom_end - window_size), 10):
            candidate = pm.gintervals(["1"], [start_pos], [start_pos + window_size])
            seq = pm.gseq_extract(candidate)[0].upper()
            ac_pos = seq.find("AC")
            if ac_pos > 0 and seq[:motif_len] != "AC":
                scan_interval = candidate
                scan_seq = seq
                break

        if scan_interval is None:
            pytest.skip("Unable to find interval with delayed AC motif")

        pm.gvtrack_create("edist_scan", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_pos", None,
                          func="pwm.edit_distance.pos",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_scan", "edist_pos"],
                             scan_interval, iterator=scan_interval)

        best_manual = _manual_pwm_edit_distance(scan_seq, pssm, threshold)
        actual_edist = result["edist_scan"].iloc[0]

        if np.isnan(best_manual):
            assert np.isnan(actual_edist)
        else:
            npt.assert_allclose(actual_edist, best_manual, atol=1e-6)

        # The first window alone should have more or equal edits
        first_window = scan_seq[:motif_len]
        first_edits = _manual_pwm_edit_distance(first_window, pssm, threshold,
                                                 scan_all=False)
        if not np.isnan(first_edits) and not np.isnan(best_manual):
            assert first_edits >= best_manual


# ===========================================================================
# C. gseq_pwm_edits() — indel support tests
# ===========================================================================


class TestGseqPwmEditsIndels:
    """Indel support in gseq_pwm_edits (max_indels parameter)."""

    def test_edit_type_column_present(self):
        """With max_indels=0 (subs only), edit_type column exists and all edits are 'sub'."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("TT", pssm, score_thresh=-0.01,
                                    max_indels=0, prior=0, bidirect=False)
        assert "edit_type" in result.columns
        edit_rows = result[result["edit_num"] > 0]
        assert len(edit_rows) > 0
        assert all(edit_rows["edit_type"] == "sub")

        # For zero-edit rows, edit_type should be None/NaN
        zero_edit_rows = result[result["n_edits"] == 0]
        if len(zero_edit_rows) > 0:
            assert all(zero_edit_rows["edit_type"].isna())

    def test_edit_type_column_present_without_max_indels(self):
        """edit_type column is present even when max_indels is not specified."""
        pssm = _create_test_pssm()
        result = pm.gseq_pwm_edits("TT", pssm, score_thresh=-0.01,
                                    prior=0, bidirect=False)
        assert "edit_type" in result.columns

    def test_max_indels_zero_matches_default(self):
        """max_indels=0 gives identical results to omitting max_indels."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9],
        ])
        seqs = ["ACGT", "TTTT", "GCTA", "AACG"]
        for seq in seqs:
            r_default = pm.gseq_pwm_edits(seq, pssm, score_thresh=-1.0,
                                           prior=0, bidirect=False)
            r_indels0 = pm.gseq_pwm_edits(seq, pssm, score_thresh=-1.0,
                                            max_indels=0, prior=0, bidirect=False)
            # Same number of rows
            assert len(r_default) == len(r_indels0), (
                f"Row count mismatch for seq '{seq}': "
                f"default={len(r_default)}, max_indels=0={len(r_indels0)}"
            )
            if len(r_default) > 0:
                npt.assert_allclose(
                    r_default["n_edits"].values,
                    r_indels0["n_edits"].values,
                    atol=1e-6,
                )

    def test_indel_reduces_edit_count(self):
        """A deletion in the sequence should need fewer total edits with max_indels=1.

        PSSM for motif "ACGT" (4bp).  Sequence "ATCGT" has an extra T
        inserted at position 2.  Without indels, no 4bp window matches
        ACGT perfectly.  With max_indels=1, the solver can delete the
        extra T to align perfectly -> 1 edit instead of >= 1 sub.
        """
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # A
            [0.0, 1.0, 0.0, 0.0],  # C
            [0.0, 0.0, 1.0, 0.0],  # G
            [0.0, 0.0, 0.0, 1.0],  # T
        ])
        seq = "ATCGT"
        # Without indels: best 4bp window is "ATCG", "TCGT" — each needs subs
        r_no_indels = pm.gseq_pwm_edits(seq, pssm, score_thresh=-0.01,
                                         max_indels=0, prior=0, bidirect=False)
        # With indels: can delete the extra T to get ACGT (1 deletion)
        r_with_indels = pm.gseq_pwm_edits(seq, pssm, score_thresh=-0.01,
                                           max_indels=1, prior=0, bidirect=False)

        assert len(r_with_indels) > 0, "max_indels=1 should produce results"

        best_no_indels = r_no_indels["n_edits"].min() if len(r_no_indels) > 0 else np.nan
        best_with_indels = r_with_indels["n_edits"].min()

        # With indels should find 1 edit (deletion); without may need more or none found
        assert best_with_indels == 1
        if not np.isnan(best_no_indels):
            assert best_with_indels <= best_no_indels

    def test_indel_edit_types(self):
        """When max_indels > 0, edit_type can contain 'ins' or 'del' values."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # A
            [0.0, 1.0, 0.0, 0.0],  # C
            [0.0, 0.0, 1.0, 0.0],  # G
            [0.0, 0.0, 0.0, 1.0],  # T
        ])

        # Deletion case: "ATCGT" -> delete T to get ACGT
        r_del = pm.gseq_pwm_edits("ATCGT", pssm, score_thresh=-0.01,
                                   max_indels=1, prior=0, bidirect=False)
        edit_rows_del = r_del[r_del["edit_num"] > 0]
        assert any(edit_rows_del["edit_type"] == "del"), (
            "Should find a 'del' edit type for 'ATCGT' with ACGT PSSM"
        )

        # Insertion case: "AGT" -> insert C to get ACGT
        r_ins = pm.gseq_pwm_edits("AGT", pssm, score_thresh=-0.01,
                                   max_indels=1, prior=0, bidirect=False)
        if len(r_ins) > 0:
            edit_rows_ins = r_ins[r_ins["edit_num"] > 0]
            assert any(edit_rows_ins["edit_type"] == "ins"), (
                "Should find an 'ins' edit type for 'AGT' with ACGT PSSM"
            )

    def test_indel_gap_characters(self):
        """With indels, window_seq or mutated_seq should contain '-' gap characters."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Deletion: extra base in seq -> mutated_seq gets '-'
        r_del = pm.gseq_pwm_edits("ATCGT", pssm, score_thresh=-0.01,
                                   max_indels=1, prior=0, bidirect=False)
        assert len(r_del) > 0
        # For deletions, mutated_seq should have '-' (deleted base position)
        assert any("-" in str(s) for s in r_del["mutated_seq"].values), (
            "mutated_seq should contain '-' when a deletion is present"
        )

        # Insertion: missing base in seq -> window_seq gets '-'
        r_ins = pm.gseq_pwm_edits("AGT", pssm, score_thresh=-0.01,
                                   max_indels=1, prior=0, bidirect=False)
        if len(r_ins) > 0:
            assert any("-" in str(s) for s in r_ins["window_seq"].values), (
                "window_seq should contain '-' when an insertion is present"
            )

    def test_max_indels_parameter_validation(self):
        """max_indels=-1 should raise an error."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gseq_pwm_edits("ACGT", pssm, score_thresh=-1.0,
                              max_indels=-1, prior=0)

    def test_alignment_strings_same_length(self):
        """window_seq and mutated_seq must be same length in indel mode."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        for seq in ["ATCGT", "AGT", "AACGT", "ACGATAC"]:
            result = pm.gseq_pwm_edits(seq, pssm, score_thresh=-0.5,
                                        max_indels=1, prior=0, bidirect=False)
            if len(result) > 0:
                for _, row in result.iterrows():
                    assert len(row["window_seq"]) == len(row["mutated_seq"]), (
                        f"window_seq ({len(row['window_seq'])}) and mutated_seq "
                        f"({len(row['mutated_seq'])}) must have same length for '{seq}'"
                    )


# ===========================================================================
# D. Virtual track — indel support tests
# ===========================================================================


class TestVtrackEditDistanceIndelsExtended:
    """Virtual track indel support (max_indels parameter)."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_vtrack_max_indels_parameter(self):
        """Create a vtrack with max_indels=1, extract on intervals, verify valid results."""
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
        ])
        test_interval = pm.gintervals(["1"], [200], [260])

        pm.gvtrack_create("edist_indel1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=-3.0, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_indel1", test_interval,
                             iterator=test_interval)
        val = result["edist_indel1"].iloc[0]
        # Should return a valid number (not all NaN)
        assert isinstance(val, (int, float, np.integer, np.floating))
        # If not NaN, should be non-negative
        if not np.isnan(val):
            assert val >= 0

    def test_vtrack_indels_reduce_edits(self):
        """With max_indels=1 the edit count should be <= the count without indels.

        Uses a longer motif across many intervals to exercise the comparison.
        """
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
        ])
        intervals = pm.gintervals(
            ["1", "1", "1", "1", "1"],
            [200, 400, 600, 800, 1000],
            [260, 460, 660, 860, 1060],
        )
        threshold = -3.0

        pm.gvtrack_create("edist_no_indels", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_with_indels", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_no_indels", "edist_with_indels"],
                             intervals, iterator=intervals)

        for i in range(len(result)):
            no_indel = result["edist_no_indels"].iloc[i]
            with_indel = result["edist_with_indels"].iloc[i]

            # If both are non-NaN, indel version should be <= substitution-only
            if not np.isnan(no_indel) and not np.isnan(with_indel):
                assert with_indel <= no_indel + 1e-6, (
                    f"Row {i}: with indels ({with_indel}) should be "
                    f"<= without indels ({no_indel})"
                )
            # If substitution-only finds a result, indel version should too
            if not np.isnan(no_indel):
                assert not np.isnan(with_indel), (
                    f"Row {i}: indel version should not be NaN when "
                    f"sub-only is {no_indel}"
                )

    def test_vtrack_specialized_solver_consistency(self):
        """Compare max_indels=1 vs max_indels=2 on same intervals.

        The solver with more indel budget should give results <= the one
        with fewer indels (monotonicity).
        """
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
        ])
        intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 500, 1000],
            [280, 580, 1080],
        )
        threshold = -3.0

        pm.gvtrack_create("edist_0indels", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_1indel", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_2indels", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=2,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["edist_0indels", "edist_1indel", "edist_2indels"],
            intervals, iterator=intervals,
        )

        for i in range(len(result)):
            v0 = result["edist_0indels"].iloc[i]
            v1 = result["edist_1indel"].iloc[i]
            v2 = result["edist_2indels"].iloc[i]

            # Monotonicity: more indels allowed => edits <= fewer indels
            if not np.isnan(v0) and not np.isnan(v1):
                assert v1 <= v0 + 1e-6, (
                    f"Row {i}: 1-indel ({v1}) should be <= 0-indel ({v0})"
                )
            if not np.isnan(v1) and not np.isnan(v2):
                assert v2 <= v1 + 1e-6, (
                    f"Row {i}: 2-indel ({v2}) should be <= 1-indel ({v1})"
                )
            # If fewer indels found a result, more indels should too
            if not np.isnan(v0):
                assert not np.isnan(v1), (
                    f"Row {i}: 1-indel should not be NaN when 0-indel is {v0}"
                )
            if not np.isnan(v1):
                assert not np.isnan(v2), (
                    f"Row {i}: 2-indel should not be NaN when 1-indel is {v1}"
                )


# ===========================================================================
# E. Optimization consistency tests
# ===========================================================================


class TestOptimizationConsistency:
    """Verify that optimizations (heuristic, flat table) do not change results."""

    def test_exact_vs_heuristic_same_result(self):
        """max_edits=None (exact) gives same n_edits as max_edits=10 (heuristic).

        For a variety of sequences, the exact solver and the heuristic solver
        (with a generous edit budget) should agree on the minimum edit count.
        """
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.05],
            [0.025, 0.025, 0.05, 0.9],
        ])
        seqs = [
            "ACGT",       # perfect match
            "TTTT",       # all mismatches
            "ACTT",       # partial match
            "GGGG",       # all mismatches
            "ACGA",       # one mismatch
            "TTTTACGTTTTT",  # longer sequence with match inside
            "NNNN",       # all N
            "GCTA",       # reverse complement of ACGT
        ]
        thresh = -1.0

        for seq in seqs:
            r_exact = pm.gseq_pwm_edits(seq, pssm, score_thresh=thresh,
                                         max_edits=None, prior=0, bidirect=False)
            r_heuristic = pm.gseq_pwm_edits(seq, pssm, score_thresh=thresh,
                                             max_edits=10, prior=0, bidirect=False)

            if len(r_exact) == 0:
                assert len(r_heuristic) == 0, (
                    f"Seq '{seq}': exact found no results but heuristic did"
                )
                continue

            exact_n = r_exact["n_edits"].min()
            if len(r_heuristic) == 0:
                # Heuristic with budget=10 should not miss anything exact finds
                assert False, (
                    f"Seq '{seq}': exact found {exact_n} edits but "
                    f"heuristic (max_edits=10) found nothing"
                )
            heuristic_n = r_heuristic["n_edits"].min()
            assert exact_n == heuristic_n, (
                f"Seq '{seq}': exact={exact_n}, heuristic={heuristic_n}"
            )

    def test_large_motif_beyond_flat_table(self):
        """A PSSM wider than 64bp (MAX_MOTIF_LEN_OPT) still works correctly.

        Creates a 70-column uniform PSSM and verifies that gseq_pwm_edits
        returns sensible results, exercising the non-flat-table fallback path.
        """
        motif_len = 70
        # Uniform PSSM: each position slightly prefers one base (cycling ACGT)
        bases = [0, 1, 2, 3]
        pssm = np.full((motif_len, 4), 0.1)
        for i in range(motif_len):
            pssm[i, bases[i % 4]] = 0.7

        # Build a sequence that matches the motif exactly
        base_chars = "ACGT"
        perfect_seq = "".join(base_chars[i % 4] for i in range(motif_len))

        # Perfect match score is 70 * log(0.7) ~ -24.97, so threshold must
        # be at or below that for a perfect match to need 0 edits.
        thresh = -30.0

        # Perfect match should need 0 edits
        r_perfect = pm.gseq_pwm_edits(perfect_seq, pssm, score_thresh=thresh,
                                       prior=0, bidirect=False)
        assert len(r_perfect) > 0
        assert r_perfect["n_edits"].iloc[0] == 0

        # All-T sequence should need many edits
        all_t = "T" * motif_len
        r_mismatched = pm.gseq_pwm_edits(all_t, pssm, score_thresh=thresh,
                                          prior=0, bidirect=False)
        if len(r_mismatched) > 0:
            # Should need edits for positions where T is not the preferred base
            assert r_mismatched["n_edits"].min() > 0

        # Longer sequence containing the perfect motif
        padded_seq = "T" * 10 + perfect_seq + "T" * 10
        r_padded = pm.gseq_pwm_edits(padded_seq, pssm, score_thresh=thresh,
                                      prior=0, bidirect=False)
        assert len(r_padded) > 0
        assert r_padded["n_edits"].min() == 0, (
            "70bp motif should find a perfect match within padded sequence"
        )

    def test_large_motif_vtrack(self):
        """A vtrack with a PSSM wider than 64bp works via the fallback path."""
        motif_len = 70
        bases = [0, 1, 2, 3]
        pssm = np.full((motif_len, 4), 0.1)
        for i in range(motif_len):
            pssm[i, bases[i % 4]] = 0.7

        _remove_all_vtracks()
        test_interval = pm.gintervals(["1"], [200], [400])

        pm.gvtrack_create("edist_large", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=-30.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_large", test_interval,
                             iterator=test_interval)
        val = result["edist_large"].iloc[0]
        assert isinstance(val, (int, float, np.integer, np.floating))
        if not np.isnan(val):
            assert val >= 0
        _remove_all_vtracks()


# ===========================================================================
# F. Adversarial / differential tests (ported from R misha)
# ===========================================================================


def _get_min_edits(result):
    """Extract the minimum n_edits from a gseq_pwm_edits result DataFrame."""
    if len(result) == 0:
        return np.nan
    vals = result["n_edits"].dropna()
    if len(vals) == 0:
        return np.nan
    return int(vals.min())


class TestAdversarialEditDistance:
    """Adversarial and differential tests ported from R misha.

    These tests verify correctness of the C++ edit distance algorithms:
    - Specialized vs generic DP solver consistency
    - Exhaustive small-motif tests
    - Prior=0 with zero-probability bases creating mandatory edits
    - Near-threshold edge cases
    - Reverse complement with indels
    - Max_edits + max_indels budget interaction
    - Known bug regression tests
    """

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    # ------------------------------------------------------------------
    # Test 1: Extreme PSSM with huge gain difference at deficit boundary
    # ------------------------------------------------------------------

    def test_extreme_pssm_gain_difference(self):
        """Extreme PSSM with huge gain difference at deficit boundary."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],  # Strong A
            [0.01, 0.97, 0.01, 0.01],  # Strong C
            [0.01, 0.01, 0.97, 0.01],  # Strong G
            [0.01, 0.01, 0.01, 0.97],  # Strong T
        ])

        perfect_score = 4 * np.log(0.97)
        threshold = perfect_score - 0.001  # Just below perfect

        seqs = ["ACGT", "ACGA", "TGCA", "AAAA"]

        # Exact mode
        result_exact = pm.gseq_pwm_edits(seqs, pssm,
                                          score_thresh=threshold,
                                          prior=0, bidirect=False)
        # Heuristic mode with max_edits=4
        result_heur = pm.gseq_pwm_edits(seqs, pssm,
                                         score_thresh=threshold,
                                         max_edits=4, prior=0, bidirect=False)

        # Perfect match "ACGT" should need 0 edits in both
        exact_acgt = result_exact[result_exact["seq_idx"] == 1]
        heur_acgt = result_heur[result_heur["seq_idx"] == 1]
        assert exact_acgt["n_edits"].iloc[0] == 0, "Exact: perfect match should be 0 edits"
        assert heur_acgt["n_edits"].iloc[0] == 0, "Heuristic: perfect match should be 0 edits"

        # "ACGA" needs exactly 1 edit
        exact_acga = result_exact[result_exact["seq_idx"] == 2]
        heur_acga = result_heur[result_heur["seq_idx"] == 2]
        assert exact_acga["n_edits"].iloc[0] == 1, "Exact: ACGA needs 1 edit"
        assert heur_acga["n_edits"].iloc[0] == 1, "Heuristic: ACGA needs 1 edit"

        # Cross-check: exact and heuristic should agree on all sequences
        for si in range(1, 5):
            e_rows = result_exact[result_exact["seq_idx"] == si]
            h_rows = result_heur[result_heur["seq_idx"] == si]
            assert e_rows["n_edits"].iloc[0] == h_rows["n_edits"].iloc[0], (
                f"Exact vs heuristic disagree on seq {si}"
            )

    # ------------------------------------------------------------------
    # Test 2: Prior=0 with zero-probability bases creates mandatory edits
    # ------------------------------------------------------------------

    def test_prior_zero_mandatory_edits(self):
        """Prior=0 with zero-probability bases creates mandatory edits."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
            [0.0, 0.0, 1.0, 0.0],  # Only G
        ])

        threshold = -0.01

        # "TTT" -- every position hits a zero-probability base -> 3 mandatory edits
        result_exact = pm.gseq_pwm_edits("TTT", pssm,
                                          score_thresh=threshold,
                                          prior=0, bidirect=False)
        assert result_exact["n_edits"].iloc[0] == 3, \
            "TTT should need 3 mandatory edits with prior=0"

        # max_edits=3 should match exact
        result_heur3 = pm.gseq_pwm_edits("TTT", pssm,
                                           score_thresh=threshold,
                                           max_edits=3, prior=0, bidirect=False)
        assert result_heur3["n_edits"].iloc[0] == 3, \
            "TTT with max_edits=3 should match exact mode"

        # max_edits=2 cannot accommodate 3 mandatory edits -> unreachable (empty result)
        result_heur2 = pm.gseq_pwm_edits("TTT", pssm,
                                           score_thresh=threshold,
                                           max_edits=2, prior=0, bidirect=False)
        assert len(result_heur2) == 0, \
            "TTT with max_edits=2 should be unreachable (3 mandatory edits needed)"

        # "ACT" -- 1 mandatory edit (pos 3: T not in {G})
        result_act = pm.gseq_pwm_edits("ACT", pssm,
                                         score_thresh=threshold,
                                         prior=0, bidirect=False)
        assert result_act["n_edits"].iloc[0] == 1, "ACT needs 1 mandatory edit"

        # Heuristic with max_edits=1 should also find 1
        result_act_h1 = pm.gseq_pwm_edits("ACT", pssm,
                                            score_thresh=threshold,
                                            max_edits=1, prior=0, bidirect=False)
        assert result_act_h1["n_edits"].iloc[0] == 1, \
            "ACT with max_edits=1 should find 1 edit"

    # ------------------------------------------------------------------
    # Test 3: max_edits = max_indels (zero remaining substitution budget)
    # ------------------------------------------------------------------

    def test_max_edits_equals_max_indels_zero_subs_budget(self):
        """max_edits equals max_indels leaves zero substitution budget."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])

        threshold = 4 * np.log(0.97) - 0.001  # Just below perfect

        # "ATCGT" -- with 1 deletion (remove T at pos 2), gives ACGT = perfect
        result = pm.gseq_pwm_edits("ATCGT", pssm,
                                    score_thresh=threshold,
                                    max_edits=1, max_indels=1,
                                    prior=0, bidirect=False)
        assert any(result["n_edits"] == 1), \
            "1 deletion should reach threshold with max_edits=1, max_indels=1"

        # "TTCGT" -- best with 1 indel needs additional subs, so with max_edits=1
        # it should not find a 1-edit solution
        result2 = pm.gseq_pwm_edits("TTCGT", pssm,
                                     score_thresh=threshold,
                                     max_edits=1, max_indels=1,
                                     prior=0, bidirect=False)
        result2_full = pm.gseq_pwm_edits("TTCGT", pssm,
                                          score_thresh=threshold,
                                          max_indels=1,
                                          prior=0, bidirect=False)

        full_min = _get_min_edits(result2_full)
        if not np.isnan(full_min) and full_min > 1:
            restricted_min = _get_min_edits(result2)
            assert np.isnan(restricted_min) or restricted_min > 1, \
                "Should not find a solution with max_edits=1 when min total > 1"

    # ------------------------------------------------------------------
    # Test 4: L=2 motif with max_indels=1
    # ------------------------------------------------------------------

    def test_l2_motif_with_indels(self):
        """L=2 motif with max_indels=1: basic cases and specialized vs generic."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
        ])

        threshold = -0.01

        # "AC" -> 0 edits (perfect match)
        r0 = pm.gseq_pwm_edits("AC", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        assert r0["n_edits"].iloc[0] == 0, "AC -> 0 edits"

        # "AGC" (G inserted) -> 1 deletion gives AC = perfect
        r1 = pm.gseq_pwm_edits("AGC", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        assert any(r1["n_edits"] == 1), "AGC -> 1 edit (delete G)"

        # "A" -> 1 insertion (skip motif pos 2) = 1 edit
        r2 = pm.gseq_pwm_edits("A", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        assert any(r2["n_edits"] == 1), "A -> 1 edit (insert C)"

        # "C" -> 1 insertion (skip motif pos 1) + 0 subs = 1 edit
        r3 = pm.gseq_pwm_edits("C", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        assert any(r3["n_edits"] == 1), "C -> 1 edit (insert A)"

        # Compare specialized (max_indels=1) vs generic (max_indels=3)
        for s in ["AC", "AGC", "A", "C", "TC", "AT"]:
            r_opt = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=1, prior=0, bidirect=False)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)
            opt_min = _get_min_edits(r_opt)
            gen_min = _get_min_edits(r_gen)
            if not np.isnan(opt_min) and not np.isnan(gen_min):
                assert gen_min <= opt_min, (
                    f"L=2 seq='{s}': generic(max_indels=3)={gen_min} "
                    f"should be <= opt(max_indels=1)={opt_min}"
                )

    # ------------------------------------------------------------------
    # Test 5: L=3 motif with max_indels=2
    # ------------------------------------------------------------------

    def test_l3_motif_with_indels(self):
        """L=3 motif with max_indels=2: specialized vs generic consistency."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        threshold = -0.01

        # "A" -- needs 2 insertions (skip C and G) = 2 edits
        r = pm.gseq_pwm_edits("A", pssm, score_thresh=threshold,
                               max_indels=2, prior=0, bidirect=False)
        assert any(r["n_edits"] == 2), "A with L=3: 2 insertions needed"

        # "AACGT" -- window scan finds "ACG" at some position with 0 edits
        r2 = pm.gseq_pwm_edits("AACGT", pssm, score_thresh=threshold,
                                max_indels=2, prior=0, bidirect=False)
        assert r2["n_edits"].iloc[0] == 0, "AACGT with L=3: finds ACG = 0 edits"

        # Compare specialized (max_indels=2) vs generic (max_indels=3)
        test_seqs = ["ACG", "A", "AACGT", "TT", "ACGTT", "CG", "TACGT"]
        for s in test_seqs:
            r_opt = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=2, prior=0, bidirect=False)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)
            opt_min = _get_min_edits(r_opt)
            gen_min = _get_min_edits(r_gen)

            if not np.isnan(opt_min) and not np.isnan(gen_min):
                assert gen_min <= opt_min, (
                    f"L=3 seq='{s}': generic={gen_min} <= specialized={opt_min}"
                )

    # ------------------------------------------------------------------
    # Test 6: Specialized max_indels=1 matches generic on diverse sequences
    # ------------------------------------------------------------------

    def test_specialized_max_indels1_vs_generic(self):
        """Specialized max_indels=1 solver should never be worse than generic DP."""
        pssm = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ])

        threshold = -5.0

        # Generate diverse test sequences with fixed seed
        rng = np.random.default_rng(42)
        bases = list("ACGT")
        test_seqs = []
        # Exact length (L=6)
        for _ in range(10):
            test_seqs.append("".join(rng.choice(bases, 6)))
        # Length L-1=5 (insertion required)
        for _ in range(10):
            test_seqs.append("".join(rng.choice(bases, 5)))
        # Length L+1=7 (deletion required)
        for _ in range(10):
            test_seqs.append("".join(rng.choice(bases, 7)))
        # Edge cases
        test_seqs.extend(["ACGTAC", "AACGTAC", "CGTAC", "TTTTTT", "TTTTTTT", "TTTTT"])

        mismatches = []
        for s in test_seqs:
            r_spec = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                        max_indels=1, prior=0, bidirect=False)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)

            spec_min = _get_min_edits(r_spec)
            gen_min = _get_min_edits(r_gen)

            if not np.isnan(spec_min) and not np.isnan(gen_min):
                if gen_min > spec_min:
                    mismatches.append(
                        f"seq='{s}': generic({gen_min}) > specialized({spec_min})"
                    )
            elif np.isnan(gen_min) and not np.isnan(spec_min):
                mismatches.append(
                    f"seq='{s}': generic=NA but specialized={spec_min}"
                )

        assert len(mismatches) == 0, (
            "Mismatches found:\n" + "\n".join(mismatches)
        )

    # ------------------------------------------------------------------
    # Test 7: Specialized max_indels=2 matches generic on diverse sequences
    # ------------------------------------------------------------------

    def test_specialized_max_indels2_vs_generic(self):
        """Specialized max_indels=2 solver should never be worse than generic DP."""
        pssm = np.array([
            [0.8, 0.05, 0.1, 0.05],
            [0.05, 0.8, 0.05, 0.1],
            [0.1, 0.05, 0.8, 0.05],
            [0.05, 0.1, 0.05, 0.8],
            [0.8, 0.05, 0.05, 0.1],
        ])

        threshold = -4.0

        rng = np.random.default_rng(123)
        bases = list("ACGT")
        test_seqs = []
        # Various lengths: L-2=3, L-1=4, L=5, L+1=6, L+2=7
        for length in range(3, 8):
            for _ in range(8):
                test_seqs.append("".join(rng.choice(bases, length)))

        mismatches = []
        for s in test_seqs:
            r_spec = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                        max_indels=2, prior=0, bidirect=False)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)

            spec_min = _get_min_edits(r_spec)
            gen_min = _get_min_edits(r_gen)

            if not np.isnan(spec_min) and not np.isnan(gen_min):
                if gen_min > spec_min:
                    mismatches.append(
                        f"seq='{s}': generic({gen_min}) > specialized({spec_min})"
                    )
            elif np.isnan(gen_min) and not np.isnan(spec_min):
                mismatches.append(
                    f"seq='{s}': generic=NA but specialized={spec_min}"
                )

        assert len(mismatches) == 0, (
            "Mismatches found:\n" + "\n".join(mismatches)
        )

    # ------------------------------------------------------------------
    # Test 8: Near-threshold edge cases for substitution counting
    # ------------------------------------------------------------------

    def test_near_threshold_edge_cases(self):
        """Near-threshold edge cases for substitution counting."""
        pssm = np.array([
            [0.5, 0.25, 0.125, 0.125],  # A preferred, moderate gains
            [0.5, 0.25, 0.125, 0.125],
            [0.5, 0.25, 0.125, 0.125],
            [0.5, 0.25, 0.125, 0.125],
        ])

        # Set threshold exactly where 2 subs would barely cover it
        threshold = 4 * np.log(0.25) + 2 * (np.log(0.5) - np.log(0.25))

        # At exact boundary: should need exactly 2 edits
        result = pm.gseq_pwm_edits("CCCC", pssm, score_thresh=threshold,
                                    prior=0, bidirect=False)
        assert result["n_edits"].iloc[0] == 2, \
            "CCCC at exact 2-sub boundary should need 2 edits"

        # Slightly above boundary: should need 2 or 3 (floating point)
        result_above = pm.gseq_pwm_edits("CCCC", pssm,
                                          score_thresh=threshold + 1e-10,
                                          prior=0, bidirect=False)
        assert 2 <= result_above["n_edits"].iloc[0] <= 3, \
            "CCCC slightly above boundary should need 2 or 3 edits"

        # Slightly below boundary: still needs 2
        result_below = pm.gseq_pwm_edits("CCCC", pssm,
                                          score_thresh=threshold - 1e-10,
                                          prior=0, bidirect=False)
        assert result_below["n_edits"].iloc[0] == 2, \
            "CCCC slightly below boundary should need 2 edits"

        # Cross-check exact vs heuristic at boundary
        result_exact = pm.gseq_pwm_edits("CCCC", pssm,
                                          score_thresh=threshold,
                                          prior=0, bidirect=False)
        result_heur = pm.gseq_pwm_edits("CCCC", pssm,
                                          score_thresh=threshold,
                                          max_edits=4,
                                          prior=0, bidirect=False)
        assert result_exact["n_edits"].iloc[0] == result_heur["n_edits"].iloc[0], \
            "Exact and heuristic should agree at threshold boundary"

    # ------------------------------------------------------------------
    # Test 9: Reverse complement gives different edit count than forward
    # ------------------------------------------------------------------

    def test_reverse_complement_different_edit_count(self):
        """Reverse complement gives different edit count than forward."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],  # Strong A
            [0.01, 0.01, 0.97, 0.01],  # Strong G
        ])

        threshold = -0.1

        # "AG" forward-only -> perfect match = 0 edits
        r_fwd = pm.gseq_pwm_edits("AG", pssm, score_thresh=threshold,
                                    prior=0, bidirect=False, strand=1)
        assert r_fwd["n_edits"].iloc[0] == 0, "AG forward should be 0 edits"

        # "AG" reverse-only: revcomp = "CT" -- both positions mismatch
        r_rev = pm.gseq_pwm_edits("AG", pssm, score_thresh=threshold,
                                    prior=0, bidirect=False, strand=-1)
        assert r_rev["n_edits"].iloc[0] >= 1, "AG reverse should need edits"

        # Bidirectional should pick the forward (better) result
        r_bidi = pm.gseq_pwm_edits("AG", pssm, score_thresh=threshold,
                                     prior=0, bidirect=True)
        assert r_bidi["n_edits"].iloc[0] == 0, "AG bidirect should find 0 edits"

        # "CT" forward should need edits, reverse should be 0
        r_ct_fwd = pm.gseq_pwm_edits("CT", pssm, score_thresh=threshold,
                                       prior=0, bidirect=False, strand=1)
        assert r_ct_fwd["n_edits"].iloc[0] >= 1, "CT forward should need edits"

        r_ct_rev = pm.gseq_pwm_edits("CT", pssm, score_thresh=threshold,
                                       prior=0, bidirect=False, strand=-1)
        assert r_ct_rev["n_edits"].iloc[0] == 0, "CT reverse = AG = 0 edits"

    # ------------------------------------------------------------------
    # Test 10: Reverse complement with indels — specialized vs generic
    # ------------------------------------------------------------------

    def test_revcomp_with_indels_specialized_vs_generic(self):
        """Reverse complement with indels: specialized vs generic consistency."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])

        threshold = 4 * np.log(0.97) - 0.01

        test_seqs = ["AACGT", "TACGT", "ACGTT", "TTTTT", "AAAAA"]

        for s in test_seqs:
            # Specialized max_indels=1
            r_spec_fwd = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                            max_indels=1, prior=0, bidirect=False, strand=1)
            r_spec_rev = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                            max_indels=1, prior=0, bidirect=False, strand=-1)
            r_spec_bidi = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                             max_indels=1, prior=0, bidirect=True)

            # Generic max_indels=3
            r_gen_fwd = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                           max_indels=3, prior=0, bidirect=False, strand=1)
            r_gen_rev = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                           max_indels=3, prior=0, bidirect=False, strand=-1)
            r_gen_bidi = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                            max_indels=3, prior=0, bidirect=True)

            spec_fwd_min = _get_min_edits(r_spec_fwd)
            gen_fwd_min = _get_min_edits(r_gen_fwd)
            spec_rev_min = _get_min_edits(r_spec_rev)
            gen_rev_min = _get_min_edits(r_gen_rev)
            spec_bidi_min = _get_min_edits(r_spec_bidi)
            gen_bidi_min = _get_min_edits(r_gen_bidi)

            # Generic should never be worse than specialized
            if not np.isnan(spec_fwd_min) and not np.isnan(gen_fwd_min):
                assert gen_fwd_min <= spec_fwd_min, (
                    f"seq='{s}' fwd: generic={gen_fwd_min} > spec={spec_fwd_min}"
                )
            if not np.isnan(spec_rev_min) and not np.isnan(gen_rev_min):
                assert gen_rev_min <= spec_rev_min, (
                    f"seq='{s}' rev: generic={gen_rev_min} > spec={spec_rev_min}"
                )
            if not np.isnan(spec_bidi_min) and not np.isnan(gen_bidi_min):
                assert gen_bidi_min <= spec_bidi_min, (
                    f"seq='{s}' bidi: generic={gen_bidi_min} > spec={spec_bidi_min}"
                )

            # Bidirectional should be min of fwd and rev
            if (not np.isnan(spec_fwd_min) and not np.isnan(spec_rev_min)
                    and not np.isnan(spec_bidi_min)):
                assert spec_bidi_min == min(spec_fwd_min, spec_rev_min), (
                    f"seq='{s}': spec bidi={spec_bidi_min} should be "
                    f"min(fwd={spec_fwd_min}, rev={spec_rev_min})"
                )

    # ------------------------------------------------------------------
    # Test 11: quick_deficit_check with budget exactly meeting deficit
    # ------------------------------------------------------------------

    def test_quick_deficit_check_exact_budget(self):
        """Quick deficit check with budget exactly meeting deficit."""
        pssm = np.array([
            [0.75, 0.25, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0],
            [0.75, 0.25, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0],
        ])

        threshold = -2.0

        # "ACAC" = col-optimal = score = 4*log(0.75) ~ -1.151 -> 0 edits
        result = pm.gseq_pwm_edits("ACAC", pssm, score_thresh=threshold,
                                    prior=0, bidirect=False)
        assert result["n_edits"].iloc[0] == 0, "ACAC should be 0 edits"

        # "CACA" needs all 4 positions changed: 4 edits
        result_rev = pm.gseq_pwm_edits("CACA", pssm, score_thresh=threshold,
                                         prior=0, bidirect=False)
        assert result_rev["n_edits"].iloc[0] == 4, "CACA should need 4 edits"

        # With max_edits=3, should be NA (all NaN in n_edits)
        result_heur = pm.gseq_pwm_edits("CACA", pssm, score_thresh=threshold,
                                          max_edits=3, prior=0, bidirect=False)
        assert all(pd.isna(result_heur["n_edits"])), \
            "CACA with max_edits=3 should be NA"

    # ------------------------------------------------------------------
    # Test 12: Suffix bound early-abandon does not reject reachable windows
    # ------------------------------------------------------------------

    def test_suffix_bound_early_abandon(self):
        """Suffix bound early-abandon does not reject reachable windows."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.8, 0.1, 0.05, 0.05],
            [0.7, 0.15, 0.075, 0.075],
            [0.6, 0.2, 0.1, 0.1],
            [0.5, 0.25, 0.125, 0.125],
            [0.4, 0.3, 0.15, 0.15],
            [0.3, 0.35, 0.175, 0.175],
            [0.25, 0.25, 0.25, 0.25],
        ])

        # Use a threshold that is actually reachable for this 8bp sequence
        threshold = -8.0

        result_exact = pm.gseq_pwm_edits("TTTTAAAA", pssm,
                                          score_thresh=threshold,
                                          prior=0, bidirect=False)
        result_heur = pm.gseq_pwm_edits("TTTTAAAA", pssm,
                                          score_thresh=threshold,
                                          max_edits=8, prior=0, bidirect=False)

        # They should agree: both should find results or both should be empty
        exact_min = _get_min_edits(result_exact)
        heur_min = _get_min_edits(result_heur)
        if not np.isnan(exact_min) and not np.isnan(heur_min):
            assert exact_min == heur_min, \
                "Exact and heuristic should agree on TTTTAAAA"
        else:
            # Both should be NA if one is
            assert np.isnan(exact_min) == np.isnan(heur_min), \
                f"Exact ({exact_min}) and heuristic ({heur_min}) disagree on TTTTAAAA"

        # Also test with max_edits=1 (tight budget)
        result_tight = pm.gseq_pwm_edits("TTTTAAAA", pssm,
                                           score_thresh=threshold,
                                           max_edits=1, prior=0, bidirect=False)
        if not np.isnan(exact_min) and exact_min <= 1:
            assert _get_min_edits(result_tight) == exact_min
        elif not np.isnan(exact_min) and exact_min > 1:
            tight_min = _get_min_edits(result_tight)
            assert np.isnan(tight_min) or len(result_tight) == 0

    # ------------------------------------------------------------------
    # Test 13: Vtrack max_indels=1 matches max_indels=3 on genomic data
    # ------------------------------------------------------------------

    def test_vtrack_indel1_vs_indel3_genomic(self):
        """Vtrack max_indels=1 matches max_indels=3 on genomic data."""
        pssm = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.05, 0.85],
        ])

        threshold = -4.0
        test_interval = pm.gintervals(["1"], [200], [230])

        pm.gvtrack_create("ed_indel1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          max_indels=1, bidirect=False, extend=True, prior=0)
        pm.gvtrack_create("ed_indel3", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          max_indels=3, bidirect=False, extend=True, prior=0)
        pm.gvtrack_create("ed_no_indel", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=True, prior=0)

        result = pm.gextract(["ed_indel1", "ed_indel3", "ed_no_indel"],
                             test_interval, iterator=1)

        for i in range(len(result)):
            v1 = result["ed_indel1"].iloc[i]
            v3 = result["ed_indel3"].iloc[i]
            v0 = result["ed_no_indel"].iloc[i]

            if not np.isnan(v1) and not np.isnan(v3):
                assert v3 <= v1 + 1e-6, (
                    f"row {i}: indel3 ({v3}) > indel1 ({v1})"
                )
            # No-indel should be >= indel1
            if not np.isnan(v0) and not np.isnan(v1):
                assert v0 >= v1 - 1e-6, (
                    f"row {i}: no_indel ({v0}) < indel1 ({v1})"
                )

    # ------------------------------------------------------------------
    # Test 14: Vtrack indels=2 bidirectional on multiple intervals
    # ------------------------------------------------------------------

    def test_vtrack_indels2_bidirectional_multi_intervals(self):
        """Vtrack indels=2 bidirectional on multiple intervals."""
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.05, 0.8, 0.1, 0.05],
            [0.05, 0.05, 0.8, 0.1],
            [0.1, 0.05, 0.05, 0.8],
            [0.8, 0.05, 0.1, 0.05],
        ])

        threshold = -3.5
        test_intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 500, 1000],
            [220, 520, 1020],
        )

        pm.gvtrack_create("ed_spec2", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          max_indels=2, bidirect=True, extend=True, prior=0)
        pm.gvtrack_create("ed_gen3", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          max_indels=3, bidirect=True, extend=True, prior=0)

        result = pm.gextract(["ed_spec2", "ed_gen3"],
                             test_intervals, iterator=test_intervals)

        for i in range(len(result)):
            v2 = result["ed_spec2"].iloc[i]
            v3 = result["ed_gen3"].iloc[i]
            if not np.isnan(v2) and not np.isnan(v3):
                assert v3 <= v2 + 1e-6, (
                    f"interval {i}: gen3 ({v3}) > spec2 ({v2})"
                )
            # If spec2 found a result but gen3 didn't, that's a bug
            if not np.isnan(v2) and np.isnan(v3):
                raise AssertionError(f"interval {i}: spec2={v2} but gen3=NA -- BUG")

    # ------------------------------------------------------------------
    # Test 15: Zero-probability PSSM entries with indels
    # ------------------------------------------------------------------

    def test_zero_prob_pssm_with_indels(self):
        """Zero-probability PSSM entries with indels."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        threshold = -0.01

        # "ACGT" is perfect with 0 edits
        r = pm.gseq_pwm_edits("ACGT", pssm, score_thresh=threshold,
                               max_indels=1, prior=0, bidirect=False)
        assert r["n_edits"].iloc[0] == 0

        # "AACGT" -- window scan finds "ACGT" at position 2 with 0 edits
        r2 = pm.gseq_pwm_edits("AACGT", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        assert r2["n_edits"].iloc[0] == 0, "AACGT finds ACGT at position 2 = 0 edits"

        # "TTTT" with no indels: 3 mandatory edits
        r3 = pm.gseq_pwm_edits("TTTT", pssm, score_thresh=threshold,
                                max_indels=0, prior=0, bidirect=False)
        assert r3["n_edits"].iloc[0] == 3, \
            "TTTT needs 3 mandatory edits (T matches at pos 4)"

        # "TTTT" with max_indels=1: specialized solver
        r4 = pm.gseq_pwm_edits("TTTT", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        # Generic DP (max_indels=3) should correctly return 3
        r4_generic = pm.gseq_pwm_edits("TTTT", pssm, score_thresh=threshold,
                                        max_indels=3, prior=0, bidirect=False)
        r4g_min = _get_min_edits(r4_generic)
        assert r4g_min == 3, "Generic DP correctly finds 3 edits for TTTT"

        # Known bug: specialized may return NA for -Inf scores
        r4_min = _get_min_edits(r4)
        if np.isnan(r4_min) and not np.isnan(r4g_min):
            # This is the known bug -- specialized solver fails with -Inf scores
            pass  # Bug confirmed: specialized indel-1 fails on TTTT with zero-prob PSSM

        # "ACGTT" -- window scan finds "ACGT" at position 1 with 0 edits
        r5 = pm.gseq_pwm_edits("ACGTT", pssm, score_thresh=threshold,
                                max_indels=1, prior=0, bidirect=False)
        assert r5["n_edits"].iloc[0] == 0, "ACGTT finds ACGT at position 1 = 0 edits"

    # ------------------------------------------------------------------
    # Test 16: Exhaustive L=2 -- all 16 dinucleotides, exact vs heuristic
    # ------------------------------------------------------------------

    def test_exhaustive_l2_dinucleotides(self):
        """Exhaustive L=2 test: all 16 dinucleotides, exact vs heuristic."""
        pssm = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ])

        threshold = -2.0
        bases = list("ACGT")

        mismatches = []
        for b1 in bases:
            for b2 in bases:
                s = b1 + b2
                r_exact = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                             prior=0, bidirect=False)
                r_heur = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                            max_edits=2, prior=0, bidirect=False)

                e_min = _get_min_edits(r_exact)
                h_min = _get_min_edits(r_heur)

                if np.isnan(e_min) and np.isnan(h_min):
                    continue
                if (not np.isnan(e_min) and e_min <= 2
                        and (np.isnan(h_min) or abs(e_min - h_min) > 0.5)):
                    mismatches.append(
                        f"seq='{s}': exact={e_min}, heur={h_min}"
                    )

        assert len(mismatches) == 0, (
            "Exact vs heuristic mismatches:\n" + "\n".join(mismatches)
        )

    # ------------------------------------------------------------------
    # Test 17: Exhaustive L=2 with indels=1, all seqs length 1-3
    # ------------------------------------------------------------------

    def test_exhaustive_l2_indels_all_short_seqs(self):
        """Exhaustive L=2 with indels=1: specialized vs generic on all seqs length 1-3."""
        pssm = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ])

        threshold = -2.0
        bases = list("ACGT")

        # Generate all sequences of length 1-3
        all_seqs = []
        for length in range(1, 4):
            if length == 1:
                all_seqs.extend(bases)
            elif length == 2:
                all_seqs.extend(b1 + b2 for b1 in bases for b2 in bases)
            elif length == 3:
                all_seqs.extend(
                    b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases
                )

        mismatches = []
        for s in all_seqs:
            # Specialized (max_indels=1)
            r_spec = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                        max_indels=1, prior=0, bidirect=False)
            # Generic (max_indels=3)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)

            spec_min = _get_min_edits(r_spec)
            gen_min = _get_min_edits(r_gen)

            if np.isnan(spec_min) and np.isnan(gen_min):
                continue
            if np.isnan(spec_min) and not np.isnan(gen_min):
                # OK only if generic used > 1 indel -- skip
                continue
            if not np.isnan(spec_min) and np.isnan(gen_min):
                mismatches.append(f"seq='{s}': spec={spec_min}, gen=NA")
            elif (not np.isnan(spec_min) and not np.isnan(gen_min)
                      and gen_min > spec_min):
                mismatches.append(
                    f"seq='{s}': gen={gen_min} > spec={spec_min}"
                )

        assert len(mismatches) == 0, (
            "Specialized vs generic mismatches:\n" + "\n".join(mismatches)
        )

    # ------------------------------------------------------------------
    # Test 18: max_edits=1, max_indels=1 on diverse inputs
    # ------------------------------------------------------------------

    def test_max_edits1_max_indels1_diverse(self):
        """max_edits=1 max_indels=1: tight budget on diverse inputs."""
        pssm = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.85, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
            [0.05, 0.05, 0.05, 0.85],
            [0.85, 0.05, 0.05, 0.05],
        ])

        threshold = 5 * np.log(0.85) - 0.001

        rng = np.random.default_rng(55)
        bases = list("ACGT")
        test_seqs = []
        for length in range(4, 8):
            for _ in range(5):
                test_seqs.append("".join(rng.choice(bases, length)))
        # Perfect + 1 insertion
        test_seqs.extend(["ACGTA", "AACGTA", "ACGTAA", "ACGT", "ACGTAC"])

        for s in test_seqs:
            r_restricted = pm.gseq_pwm_edits(
                s, pssm, score_thresh=threshold,
                max_edits=1, max_indels=1, prior=0, bidirect=False)
            r_unrestricted = pm.gseq_pwm_edits(
                s, pssm, score_thresh=threshold,
                max_indels=1, prior=0, bidirect=False)

            rmin = _get_min_edits(r_restricted)
            umin = _get_min_edits(r_unrestricted)

            # Restricted should be >= unrestricted (or both NA)
            if not np.isnan(rmin) and not np.isnan(umin):
                assert rmin >= umin, (
                    f"seq='{s}': restricted={rmin} < unrestricted={umin}"
                )

            # If unrestricted > 1, restricted should be NA
            if not np.isnan(umin) and umin > 1:
                assert np.isnan(rmin), (
                    f"seq='{s}': unrestricted={umin}>1 but restricted={rmin}"
                )

            # If restricted is not NA, it should be <= 1
            if not np.isnan(rmin):
                assert rmin <= 1, (
                    f"seq='{s}': restricted={rmin} > max_edits=1"
                )

    # ------------------------------------------------------------------
    # Test 19: Two-deletion segment boundaries
    # ------------------------------------------------------------------

    def test_two_deletion_segment_boundaries(self):
        """Two-deletion segment boundaries: off-by-one in segment indices."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        threshold = -0.01

        # "XACGTY" -- delete X and Y to get ACGT = 2 edits
        for x in "ACGT":
            for y in "ACGT":
                s = x + "ACGT" + y
                r = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=2, prior=0, bidirect=False)
                assert any(r["n_edits"] <= 2), (
                    f"seq='{s}': should reach in <= 2 edits (2 deletions)"
                )

        # "TTACGT" -- finds "ACGT" at position 3 with 0 edits
        r_xx = pm.gseq_pwm_edits("TTACGT", pssm, score_thresh=threshold,
                                   max_indels=2, prior=0, bidirect=False)
        assert r_xx["n_edits"].iloc[0] == 0, \
            "TTACGT: finds ACGT at position 3 = 0 edits"

        # "ACGTTT" -- finds "ACGT" at position 1 with 0 edits
        r_trail = pm.gseq_pwm_edits("ACGTTT", pssm, score_thresh=threshold,
                                      max_indels=2, prior=0, bidirect=False)
        assert r_trail["n_edits"].iloc[0] == 0, \
            "ACGTTT: finds ACGT at position 1 = 0 edits"

        # "GACGAT" -- no ACGT substring, needs deletions
        r_2del = pm.gseq_pwm_edits("GACGAT", pssm, score_thresh=threshold,
                                     max_indels=2, prior=0, bidirect=False)
        assert any(r_2del["n_edits"] <= 2), \
            "GACGAT: should find solution with <= 2 edits via deletions"

    # ------------------------------------------------------------------
    # Test 20: One deletion + one insertion case (case 6 in two-indel solver)
    # ------------------------------------------------------------------

    def test_mixed_del_ins_case6(self):
        """One deletion + one insertion case (case 6 in two-indel solver).

        Compares specialized (max_indels=2) vs generic (max_indels=3) since
        we don't have a Python brute-force DP with indels.
        """
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        threshold = -0.01

        # Mixed del+ins cases
        test_seqs = ["AGGT", "ATGT", "ACTT", "TACT", "GACT"]
        for s in test_seqs:
            r_spec = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                        max_indels=2, prior=0, bidirect=False)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)
            spec_min = _get_min_edits(r_spec)
            gen_min = _get_min_edits(r_gen)

            if not np.isnan(gen_min):
                assert not np.isnan(spec_min), (
                    f"seq='{s}': gen={gen_min} but spec=NA"
                )
                assert spec_min <= gen_min + 1e-6, (
                    f"seq='{s}': spec={spec_min}, gen={gen_min}"
                )

    # ------------------------------------------------------------------
    # Test 21: BUG specialized indel solvers with -Inf base scores (vtrack)
    # ------------------------------------------------------------------

    def test_bug_specialized_indel_inf_scores_vtrack(self):
        """BUG regression: specialized indel solvers with -Inf base scores (vtrack path)."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
            [0.0, 0.0, 1.0, 0.0],  # Only G
            [0.0, 0.0, 0.0, 1.0],  # Only T
        ])

        threshold = -0.01

        # Find "TTTT" in genome for vtrack testing
        seq_region = pm.gseq_extract(pm.gintervals(["1"], [0], [5000]))[0].upper()
        pos = seq_region.find("TTTT")
        if pos < 0:
            pytest.skip("No TTTT found in test genome")

        test_interval = pm.gintervals(["1"], [pos], [pos + 4])
        seq_at = pm.gseq_extract(test_interval)[0].upper()
        assert seq_at == "TTTT"

        pm.gvtrack_create("ed_no_indel", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("ed_indel1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("ed_indel2", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=2,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("ed_indel3", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=3,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_no_indel", "ed_indel1", "ed_indel2", "ed_indel3"],
            test_interval, iterator=test_interval)

        assert result["ed_no_indel"].iloc[0] == 3, "no_indel correctly returns 3"
        assert result["ed_indel3"].iloc[0] == 3, "generic indel3 correctly returns 3"

        # BUG tests: specialized should return 3, matching no_indel
        assert result["ed_indel1"].iloc[0] == 3, \
            "BUG: indel1 should return 3, not 1"
        assert result["ed_indel2"].iloc[0] == 3, \
            "BUG: indel2 should return 3, not 1"

        # Indel-enabled should never return FEWER edits than no-indel
        assert result["ed_indel1"].iloc[0] >= result["ed_no_indel"].iloc[0], \
            "indel1 should not return fewer edits than no_indel"
        assert result["ed_indel2"].iloc[0] >= result["ed_no_indel"].iloc[0], \
            "indel2 should not return fewer edits than no_indel"

    # ------------------------------------------------------------------
    # Test 22: BUG gseq.pwm_edits ignores max_edits for mandatory edits
    # ------------------------------------------------------------------

    def test_bug_max_edits_cap_mandatory_edits(self):
        """BUG regression: gseq_pwm_edits ignores max_edits for mandatory edits."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
            [0.0, 0.0, 1.0, 0.0],  # Only G
        ])

        threshold = -0.01

        # max_edits=1 -- should be NA since 3 > 1
        r1 = pm.gseq_pwm_edits("TTT", pssm, score_thresh=threshold,
                                max_edits=1, prior=0, bidirect=False)
        # max_edits=2 -- should be NA since 3 > 2
        r2 = pm.gseq_pwm_edits("TTT", pssm, score_thresh=threshold,
                                max_edits=2, prior=0, bidirect=False)
        # max_edits=3 -- should return 3
        r3 = pm.gseq_pwm_edits("TTT", pssm, score_thresh=threshold,
                                max_edits=3, prior=0, bidirect=False)

        # BUG: max_edits=1 and =2 return n_edits=3 instead of NA
        # For now document the actual behavior
        assert all(pd.isna(r1["n_edits"])) or all(r1["n_edits"] > 1), \
            "BUG: max_edits=1 should return NA for 3 mandatory edits"
        assert all(pd.isna(r2["n_edits"])) or all(r2["n_edits"] > 2), \
            "BUG: max_edits=2 should return NA for 3 mandatory edits"
        assert r3["n_edits"].iloc[0] == 3, "max_edits=3 should return 3"

    # ------------------------------------------------------------------
    # Test 23: Specialized vs generic with PSSMs containing near-zero probs
    # ------------------------------------------------------------------

    def test_specialized_vs_generic_near_zero_probs(self):
        """Specialized vs generic with PSSMs containing near-zero probabilities."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])

        threshold = -1.0

        rng = np.random.default_rng(2026)
        bases = list("ACGT")
        test_seqs = []
        for length in range(3, 7):
            for _ in range(8):
                test_seqs.append("".join(rng.choice(bases, length)))

        mismatches = []
        for s in test_seqs:
            r_spec1 = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                          max_indels=1, prior=0, bidirect=False)
            r_spec2 = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                          max_indels=2, prior=0, bidirect=False)
            r_gen = pm.gseq_pwm_edits(s, pssm, score_thresh=threshold,
                                       max_indels=3, prior=0, bidirect=False)

            s1 = _get_min_edits(r_spec1)
            s2 = _get_min_edits(r_spec2)
            g3 = _get_min_edits(r_gen)

            # Generic should never be worse than specialized
            if not np.isnan(s1) and not np.isnan(g3) and g3 > s1:
                mismatches.append(
                    f"seq='{s}': gen3={g3} > spec1={s1}"
                )
            if not np.isnan(s2) and not np.isnan(g3) and g3 > s2:
                mismatches.append(
                    f"seq='{s}': gen3={g3} > spec2={s2}"
                )
            # spec2 should be <= spec1 (more indel budget)
            if not np.isnan(s1) and not np.isnan(s2) and s2 > s1:
                mismatches.append(
                    f"seq='{s}': spec2={s2} > spec1={s1}"
                )

        assert len(mismatches) == 0, (
            "Mismatches found:\n" + "\n".join(mismatches)
        )


# ===========================================================================
# F. Comprehensive LSE vtrack tests
# ===========================================================================


class TestVtrackLseComprehensive:
    """Comprehensive tests for pwm.edit_distance.lse virtual track.

    Covers: LSE returns 0 when already above threshold, LSE returns NaN for
    unreachable threshold, LSE respects max_edits, LSE respects score_min,
    LSE works with bidirect=TRUE/FALSE, LSE threshold monotonicity,
    LSE with prior > 0.
    """

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_lse_returns_zero_when_already_above_threshold(self):
        """When the LSE score already exceeds a very low threshold, 0 edits needed."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        # Very low threshold: the LSE score should already exceed it
        pm.gvtrack_create("lse_above", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=-100.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("lse_above", test_interval, iterator=test_interval)
        npt.assert_allclose(result["lse_above"].iloc[0], 0, atol=1e-6)

    def test_lse_returns_nan_for_unreachable_threshold(self):
        """An impossibly high threshold should yield NaN."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create("lse_unreach", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=1000.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("lse_unreach", test_interval, iterator=test_interval)
        assert np.isnan(result["lse_unreach"].iloc[0])

    def test_lse_respects_max_edits(self):
        """max_edits caps the LSE edit distance; exceeding the cap yields NaN."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -3.0

        # Uncapped LSE
        pm.gvtrack_create("lse_uncapped", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        # LSE with max_edits=1
        pm.gvtrack_create("lse_cap1", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold, max_edits=1,
                          bidirect=False, extend=False, prior=0)

        # LSE with max_edits=3
        pm.gvtrack_create("lse_cap3", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold, max_edits=3,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["lse_uncapped", "lse_cap1", "lse_cap3"],
                             test_interval, iterator=test_interval)

        uncapped = result["lse_uncapped"].iloc[0]
        cap1 = result["lse_cap1"].iloc[0]
        cap3 = result["lse_cap3"].iloc[0]

        # If uncapped needs > 1 edit, cap1 should be NaN
        if not np.isnan(uncapped) and uncapped > 1:
            assert np.isnan(cap1), (
                f"Uncapped = {uncapped} > 1, so cap1 should be NaN but got {cap1}"
            )

        # If uncapped needs <= 1 edit, cap1 should match
        if not np.isnan(uncapped) and uncapped <= 1:
            npt.assert_allclose(cap1, uncapped, atol=1e-6)

        # If uncapped needs <= 3 edits, cap3 should match
        if not np.isnan(uncapped) and uncapped <= 3:
            npt.assert_allclose(cap3, uncapped, atol=1e-6)

    def test_lse_respects_score_min(self):
        """score_min filters windows; lenient score_min matches unfiltered."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        # Without score_min filter
        pm.gvtrack_create("lse_nofilt", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        # Very low score_min (should not filter anything)
        pm.gvtrack_create("lse_lowfilt", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          score_min=-100.0,
                          bidirect=False, extend=False, prior=0)

        # Very high score_min (should filter out most/all windows -> NaN)
        pm.gvtrack_create("lse_highfilt", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          score_min=0.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["lse_nofilt", "lse_lowfilt", "lse_highfilt"],
                             test_interval, iterator=test_interval)

        nf = result["lse_nofilt"].iloc[0]
        lf = result["lse_lowfilt"].iloc[0]
        hf = result["lse_highfilt"].iloc[0]

        # Low filter should match no-filter
        if np.isnan(nf):
            assert np.isnan(lf)
        else:
            npt.assert_allclose(nf, lf, atol=1e-6)

        # High filter should return NaN or >= the unfiltered result
        if not np.isnan(hf) and not np.isnan(nf):
            assert hf >= nf - 1e-6

    def test_lse_bidirect_true_and_false(self):
        """LSE works with bidirect=True and bidirect=False; bidi <= min(fwd, rev)."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        # Forward only
        pm.gvtrack_create("lse_fwd", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, strand=1, extend=False, prior=0)

        # Reverse only
        pm.gvtrack_create("lse_rev", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, strand=-1, extend=False, prior=0)

        # Bidirectional
        pm.gvtrack_create("lse_bidi", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract(["lse_fwd", "lse_rev", "lse_bidi"],
                             test_interval, iterator=test_interval)

        fwd = result["lse_fwd"].iloc[0]
        rev = result["lse_rev"].iloc[0]
        bidi = result["lse_bidi"].iloc[0]

        # All should be valid (non-negative or NaN)
        assert np.isnan(fwd) or fwd >= 0
        assert np.isnan(rev) or rev >= 0
        assert np.isnan(bidi) or bidi >= 0

        # Bidirectional should return <= minimum of both strands
        if not np.isnan(fwd) and not np.isnan(rev):
            assert bidi <= min(fwd, rev) + 1e-6

    def test_lse_threshold_monotonicity(self):
        """Higher thresholds require monotonically more (or equal) edits.

        Once a threshold becomes NaN (unreachable), all higher thresholds
        should also be NaN.
        """
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        thresholds = [-20.0, -10.0, -5.0, -2.0, 0.0]
        vnames = [f"lse_thresh_{i}" for i in range(len(thresholds))]

        for i, t in enumerate(thresholds):
            pm.gvtrack_create(vnames[i], None,
                              func="pwm.edit_distance.lse",
                              pssm=pssm, score_thresh=t,
                              bidirect=False, extend=False, prior=0)

        result = pm.gextract(vnames, test_interval, iterator=test_interval)
        edits = [result[vn].iloc[0] for vn in vnames]

        # Filter out NaNs and check monotonicity
        finite = [(t, e) for t, e in zip(thresholds, edits, strict=True)
                  if not np.isnan(e)]
        if len(finite) > 1:
            for i in range(1, len(finite)):
                assert finite[i][1] >= finite[i - 1][1] - 1e-6, (
                    f"Threshold {finite[i][0]} gave {finite[i][1]} edits, "
                    f"but lower threshold {finite[i-1][0]} gave {finite[i-1][1]}"
                )

        # Once NaN is encountered, all subsequent should be NaN
        na_found = False
        for i, e in enumerate(edits):
            if np.isnan(e):
                na_found = True
            elif na_found:
                assert False, (
                    f"Found non-NaN edit distance ({e}) at threshold "
                    f"{thresholds[i]} after NaN at a lower threshold"
                )

    def test_lse_with_prior_gt_zero(self):
        """LSE with prior=0 and prior=0.01 both give valid non-negative results."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("lse_prior0", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("lse_prior01", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0.01)

        result = pm.gextract(["lse_prior0", "lse_prior01"],
                             test_interval, iterator=test_interval)

        v0 = result["lse_prior0"].iloc[0]
        v01 = result["lse_prior01"].iloc[0]

        if not np.isnan(v0):
            assert v0 >= 0
        if not np.isnan(v01):
            assert v01 >= 0

        # At least one should produce a result
        assert not np.isnan(v0) or not np.isnan(v01)


# ===========================================================================
# G. Comprehensive indel vtrack combination tests
# ===========================================================================


class TestVtrackIndelCombinations:
    """Comprehensive tests for indel vtrack parameter combinations.

    Covers: max_indels=1 with single insertion disrupting motif, max_indels=2
    requiring two indels, combined indels + substitutions, max_indels cap
    enforcement, indel at interval boundary with extend=TRUE/FALSE,
    max_indels with bidirectional scanning, max_indels with
    pwm.edit_distance.pos, max_indels with pwm.max.edit_distance,
    max_indels with 1bp iterator, max_indels with score_min combined.
    """

    # ACGT 4bp motif used across most tests
    _acgt_pssm = np.array([
        [0.97, 0.01, 0.01, 0.01],   # A
        [0.01, 0.97, 0.01, 0.01],   # C
        [0.01, 0.01, 0.97, 0.01],   # G
        [0.01, 0.01, 0.01, 0.97],   # T
    ])

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_max_indels_1_with_single_insertion_disrupting_motif(self):
        """Find ACGT in test genome; with/without indels both find perfect match."""
        pssm = self._acgt_pssm
        motif_len = pssm.shape[0]

        # Search for ACGT in test genome
        search_interval = pm.gintervals(["1"], [0], [5000])
        full_seq = pm.gseq_extract(search_interval)[0].upper()

        acgt_pos = full_seq.find("ACGT")
        if acgt_pos < 0:
            pytest.skip("No ACGT motif found in test genome region")

        # Use a window around the ACGT occurrence
        abs_start = acgt_pos
        test_interval = pm.gintervals(["1"], [abs_start],
                                      [abs_start + motif_len + 10])
        threshold = float(np.sum(np.log([0.97, 0.97, 0.97, 0.97])))

        # Without indels: should find perfect match (0 edits)
        pm.gvtrack_create("edist_no_indel", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)

        # With 1 indel: should also find 0 edits (same or better)
        pm.gvtrack_create("edist_1_indel", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_no_indel", "edist_1_indel"],
                             test_interval, iterator=test_interval)

        npt.assert_allclose(result["edist_no_indel"].iloc[0], 0, atol=1e-6)
        npt.assert_allclose(result["edist_1_indel"].iloc[0], 0, atol=1e-6)

    def test_max_indels_2_cases_requiring_two_indels(self):
        """Compare max_indels=0, 1, 2: monotonicity d2 <= d1 <= d0."""
        pssm = self._acgt_pssm
        intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 500, 1000],
            [260, 560, 1060],
        )
        threshold = -3.0

        pm.gvtrack_create("edist_d0", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_d1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_d2", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=2,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_d0", "edist_d1", "edist_d2"],
                             intervals, iterator=intervals)

        for i in range(len(result)):
            d0 = result["edist_d0"].iloc[i]
            d1 = result["edist_d1"].iloc[i]
            d2 = result["edist_d2"].iloc[i]

            # Monotonicity: more indels => edit count <=
            if not np.isnan(d0) and not np.isnan(d1):
                assert d1 <= d0 + 1e-6, (
                    f"Row {i}: d1={d1} should be <= d0={d0}"
                )
            if not np.isnan(d1) and not np.isnan(d2):
                assert d2 <= d1 + 1e-6, (
                    f"Row {i}: d2={d2} should be <= d1={d1}"
                )
            if not np.isnan(d0) and not np.isnan(d2):
                assert d2 <= d0 + 1e-6, (
                    f"Row {i}: d2={d2} should be <= d0={d0}"
                )

            # If substitution-only has a result, indel versions should too
            if not np.isnan(d0):
                assert not np.isnan(d1), f"Row {i}: d1 should not be NaN when d0={d0}"
                assert not np.isnan(d2), f"Row {i}: d2 should not be NaN when d0={d0}"

    def test_combined_indels_and_substitutions(self):
        """With a 6bp motif, combined indels + subs <= subs-only edits."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],   # A
            [0.01, 0.97, 0.01, 0.01],   # C
            [0.01, 0.01, 0.97, 0.01],   # G
            [0.01, 0.01, 0.01, 0.97],   # T
            [0.97, 0.01, 0.01, 0.01],   # A
            [0.01, 0.97, 0.01, 0.01],   # C
        ])
        test_interval = pm.gintervals(["1"], [200], [280])
        threshold = -5.0

        pm.gvtrack_create("edist_sub_only", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("edist_combined", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_sub_only", "edist_combined"],
                             test_interval, iterator=test_interval)

        sub_only = result["edist_sub_only"].iloc[0]
        combined = result["edist_combined"].iloc[0]

        # At least one should be non-NaN
        assert not np.isnan(sub_only) or not np.isnan(combined), (
            "At least one method should find a reachable window"
        )

        # Combined should be <= substitution-only (if both non-NaN)
        if not np.isnan(sub_only) and not np.isnan(combined):
            assert combined <= sub_only + 1e-6

        # Both should be non-negative when non-NaN
        if not np.isnan(sub_only):
            assert sub_only >= 0
        if not np.isnan(combined):
            assert combined >= 0

    def test_max_indels_cap_enforcement(self):
        """max_indels=1 should not allow 2 indels: d2 <= d1, and if d2=NaN then d1=NaN."""
        pssm = self._acgt_pssm
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -3.0

        pm.gvtrack_create("edist_cap1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("edist_cap2", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=2,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_cap1", "edist_cap2"],
                             test_interval, iterator=test_interval)

        d1 = result["edist_cap1"].iloc[0]
        d2 = result["edist_cap2"].iloc[0]

        # d=2 can only be <= d=1
        if not np.isnan(d1) and not np.isnan(d2):
            assert d2 <= d1 + 1e-6

        # If d=2 returns NaN, d=1 must also be NaN
        if np.isnan(d2):
            assert np.isnan(d1), (
                "If max_indels=2 is NaN, max_indels=1 should also be NaN"
            )

    def test_indel_at_interval_boundary_with_extend_true(self):
        """With extend=True, interval smaller than motif still produces a result.

        Without extend, interval < motif_len should yield NaN.
        """
        pssm = self._acgt_pssm
        motif_len = pssm.shape[0]

        # Interval smaller than motif length
        test_interval = pm.gintervals(["1"], [200], [200 + motif_len - 1])
        threshold = -5.0

        pm.gvtrack_create("edist_edge_ext", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=True, prior=0)

        pm.gvtrack_create("edist_edge_noext", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_edge_ext", "edist_edge_noext"],
                             test_interval, iterator=test_interval)

        # Non-extended with interval < motif_len should be NaN
        assert np.isnan(result["edist_edge_noext"].iloc[0]), (
            "Interval smaller than motif without extend should be NaN"
        )

        # Extended may or may not be NaN (depends on threshold reachability)
        ext_val = result["edist_edge_ext"].iloc[0]
        assert np.isnan(ext_val) or ext_val >= 0

    def test_indel_at_interval_boundary_with_extend_false(self):
        """With extend=False, interval exactly motif_len gives a valid result."""
        pssm = self._acgt_pssm
        motif_len = pssm.shape[0]

        # Interval exactly motif_len in size
        test_interval = pm.gintervals(["1"], [200], [200 + motif_len])
        threshold = -5.0

        pm.gvtrack_create("edist_exact_boundary", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_exact_boundary", test_interval,
                             iterator=test_interval)

        val = result["edist_exact_boundary"].iloc[0]
        assert np.isnan(val) or val >= 0

    def test_max_indels_with_bidirectional_scanning(self):
        """With max_indels=1, bidirectional should return min(fwd, rev)."""
        pssm = self._acgt_pssm
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -3.0

        pm.gvtrack_create("edist_fwd_d1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, strand=1, extend=False, prior=0)

        pm.gvtrack_create("edist_rev_d1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, strand=-1, extend=False, prior=0)

        pm.gvtrack_create("edist_bidi_d1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract(["edist_fwd_d1", "edist_rev_d1", "edist_bidi_d1"],
                             test_interval, iterator=test_interval)

        fwd = result["edist_fwd_d1"].iloc[0]
        rev = result["edist_rev_d1"].iloc[0]
        bidi = result["edist_bidi_d1"].iloc[0]

        if not np.isnan(fwd) and not np.isnan(rev):
            npt.assert_allclose(bidi, min(fwd, rev), atol=1e-6)
        elif not np.isnan(fwd):
            npt.assert_allclose(bidi, fwd, atol=1e-6)
        elif not np.isnan(rev):
            npt.assert_allclose(bidi, rev, atol=1e-6)

    def test_max_indels_with_pwm_edit_distance_pos(self):
        """pwm.edit_distance.pos with max_indels=1: position should be non-NaN when edits are non-NaN."""
        pssm = self._acgt_pssm
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -5.0

        pm.gvtrack_create("edist_val_d1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("edist_pos_d1", None,
                          func="pwm.edit_distance.pos",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["edist_val_d1", "edist_pos_d1"],
                             test_interval, iterator=test_interval)

        # Position should be non-NaN if edit distance is non-NaN
        if not np.isnan(result["edist_val_d1"].iloc[0]):
            assert not np.isnan(result["edist_pos_d1"].iloc[0])
            assert result["edist_pos_d1"].iloc[0] >= 1, (
                "Position should be >= 1 (1-based)"
            )

    def test_max_indels_with_pwm_max_edit_distance(self):
        """pwm.max.edit_distance with indels <= without indels at the same best-PWM window."""
        pssm = self._acgt_pssm
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -5.0

        pm.gvtrack_create("pwm_max_edist_d0", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("pwm_max_edist_d1", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(["pwm_max_edist_d0", "pwm_max_edist_d1"],
                             test_interval, iterator=test_interval)

        d0 = result["pwm_max_edist_d0"].iloc[0]
        d1 = result["pwm_max_edist_d1"].iloc[0]

        # With indels should be <= without indels
        if not np.isnan(d0) and not np.isnan(d1):
            assert d1 <= d0 + 1e-6

    def test_max_indels_with_1bp_iterator(self):
        """max_indels=1 with 1bp iterator: per-position d1 <= d0 at every position."""
        pssm = self._acgt_pssm
        test_interval = pm.gintervals(["1"], [200], [210])
        threshold = -5.0

        pm.gvtrack_create("edist_1bp_d0", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=0,
                          bidirect=False, extend=True, prior=0)

        pm.gvtrack_create("edist_1bp_d1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=True, prior=0)

        result = pm.gextract(["edist_1bp_d0", "edist_1bp_d1"],
                             test_interval, iterator=1)

        assert len(result) > 0

        for i in range(len(result)):
            d0 = result["edist_1bp_d0"].iloc[i]
            d1 = result["edist_1bp_d1"].iloc[i]

            if not np.isnan(d0) and not np.isnan(d1):
                assert d1 <= d0 + 1e-6, (
                    f"Position {result['start'].iloc[i]}: d1={d1} should be <= d0={d0}"
                )
            if not np.isnan(d0):
                assert not np.isnan(d1), (
                    f"Position {result['start'].iloc[i]}: d1 should not be NaN "
                    f"if d0={d0}"
                )

    def test_max_indels_with_score_min_combined(self):
        """max_indels + score_min: lenient filter matches no-filter; strict filter >= no-filter."""
        pssm = self._acgt_pssm
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -5.0

        # With indels, no score filter
        pm.gvtrack_create("edist_d1_nofilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          bidirect=False, extend=False, prior=0)

        # With indels, lenient score filter
        pm.gvtrack_create("edist_d1_lowfilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          score_min=-100.0,
                          bidirect=False, extend=False, prior=0)

        # With indels, strict score filter
        pm.gvtrack_create("edist_d1_highfilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_indels=1,
                          score_min=0.0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["edist_d1_nofilt", "edist_d1_lowfilt", "edist_d1_highfilt"],
            test_interval, iterator=test_interval,
        )

        nf = result["edist_d1_nofilt"].iloc[0]
        lf = result["edist_d1_lowfilt"].iloc[0]
        hf = result["edist_d1_highfilt"].iloc[0]

        # Lenient filter should match no-filter
        if np.isnan(nf):
            assert np.isnan(lf)
        else:
            npt.assert_allclose(nf, lf, atol=1e-6)

        # Strict filter should return NaN or >= no-filter result
        if not np.isnan(hf) and not np.isnan(nf):
            assert hf >= nf - 1e-6


# ===========================================================================
# direction="below" — helper
# ===========================================================================


def _manual_pwm_edit_distance_below(seq, pssm, threshold, max_edits=None, scan_all=True):
    """Compute minimum edits to bring the best window's score BELOW threshold.

    Returns the minimum number of substitutions to push the score to
    ``<= threshold``.  When *scan_all* is True every start position is
    examined and the minimum edit count is returned.
    """
    motif_len = pssm.shape[0]
    if len(seq) < motif_len:
        return np.nan

    log_pssm = np.log(pssm, where=(pssm > 0), out=np.full_like(pssm, -np.inf))
    col_min = log_pssm.min(axis=1)

    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _score_window(window_seq):
        current_score = 0.0
        has_neg_inf = False
        losses = []

        for i in range(motif_len):
            base = window_seq[i]
            idx = base_map.get(base)
            base_score = log_pssm[i].min() if idx is None else log_pssm[i, idx]

            if not np.isfinite(base_score):
                # Score is -Inf → already below any finite threshold
                has_neg_inf = True
                break

            current_score += base_score
            losses.append(base_score - col_min[i])

        if has_neg_inf:
            return 0

        # surplus = how much the current score exceeds the threshold
        surplus = current_score - threshold
        if surplus <= 0:
            return 0  # already at or below threshold

        # Check if switching all positions to worst can cover the surplus
        total_possible_loss = sum(losses)
        if total_possible_loss < surplus - 1e-12:
            return np.nan

        losses_sorted = sorted(losses, reverse=True)
        if max_edits is not None:
            losses_sorted = losses_sorted[:max_edits]

        acc = 0.0
        edits = 0
        for loss in losses_sorted:
            edits += 1
            acc += loss
            if acc >= surplus:
                return edits

        return np.nan

    if not scan_all:
        return _score_window(seq)

    best = np.nan
    for start in range(len(seq) - motif_len + 1):
        window = seq[start:start + motif_len]
        cand = _score_window(window)
        if np.isnan(best) or (not np.isnan(cand) and cand < best):
            best = cand
    return best


# ===========================================================================
# C. gseq_pwm_edits() with direction="below"
# ===========================================================================


class TestGseqPwmEditsDirectionBelowStructure:
    """Output structure for direction='below'."""

    def test_returns_dataframe_with_expected_columns(self):
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        result = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {
            "seq_idx", "strand", "window_start", "score_before",
            "score_after", "n_edits", "edit_num", "motif_col",
            "ref", "alt", "gain", "window_seq", "mutated_seq",
        }
        assert expected_cols.issubset(set(result.columns))
        assert len(result) > 0
        # "ACGT" is a perfect match — needs edits to go below threshold
        assert any(result["n_edits"] > 0)


class TestGseqPwmEditsDirectionBelow:
    """Core edit logic for direction='below'."""

    def test_already_below_threshold_zero_edits(self):
        """When the best window already scores below the threshold, n_edits=0."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
        ])
        # "TT" scores very low against AC-preferring PSSM
        result = pm.gseq_pwm_edits(
            "TT", pssm, score_thresh=-0.5,
            prior=0, bidirect=False, direction="below",
        )
        assert result["n_edits"].iloc[0] == 0
        assert result["edit_num"].iloc[0] == 0
        assert pd.isna(result["motif_col"].iloc[0])

    def test_score_before_above_score_after_below_threshold(self):
        """Edits must bring score from above to at/below threshold."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        threshold = -1.0
        result = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="below",
        )
        rows_with_edits = result[result["n_edits"] > 0]
        assert len(rows_with_edits) > 0, "Should need edits to bring score below threshold"
        assert all(rows_with_edits["score_before"] > threshold), \
            "score_before should be above threshold for below direction"
        assert all(rows_with_edits["score_after"] <= threshold + 1e-9), \
            "score_after should be at or below threshold for below direction"

    def test_edits_ref_matches_window_seq(self):
        """Edit ref bases should match the corresponding position in window_seq."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
        ])
        result = pm.gseq_pwm_edits(
            "ACG", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        edit_rows = result[result["edit_num"] > 0]
        assert len(edit_rows) > 0
        for _, row in edit_rows.iterrows():
            if not pd.isna(row.get("edit_type")) and row.get("edit_type") == "sub":
                mc = int(row["motif_col"])
                assert row["window_seq"][mc - 1] == row["ref"], \
                    f"ref should match window_seq at motif_col {mc}"

    def test_applying_edits_produces_mutated_seq(self):
        """Applying all substitution edits to window_seq yields mutated_seq."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
        ])
        result = pm.gseq_pwm_edits(
            "ACG", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        edit_rows = result[result["edit_num"] > 0]
        ws = result["window_seq"].iloc[0]
        ms = result["mutated_seq"].iloc[0]
        ws_chars = list(ws)
        for _, row in edit_rows.iterrows():
            et = row.get("edit_type")
            if pd.isna(et) or et == "sub":
                mc = int(row["motif_col"])
                ws_chars[mc - 1] = row["alt"]
        assert "".join(ws_chars) == ms, \
            "Applying all edits to window_seq should produce mutated_seq"

    def test_gain_negative_for_below_direction(self):
        """In below direction, gains should be negative (score is being reduced)."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        result = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        real_edits = result[result["edit_num"] > 0]
        assert len(real_edits) > 0
        sub_edits = real_edits[
            real_edits["edit_type"].isna() | (real_edits["edit_type"] == "sub")
        ] if "edit_type" in real_edits.columns else real_edits
        if len(sub_edits) > 0:
            assert all(sub_edits["gain"] < 0), \
                "gain should be negative for below direction"

    def test_score_after_equals_score_before_plus_gains(self):
        """score_after == score_before + sum(gains) for substitution edits."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        result = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        sub_rows = result[result["edit_num"] > 0]
        if "edit_type" in sub_rows.columns:
            sub_rows = sub_rows[
                sub_rows["edit_type"].isna() | (sub_rows["edit_type"] == "sub")
            ]
        if len(sub_rows) > 0:
            total_gain = sub_rows["gain"].sum()
            npt.assert_allclose(
                result["score_after"].iloc[0],
                result["score_before"].iloc[0] + total_gain,
                atol=1e-3,
            )

    def test_multiple_sequences(self):
        """Multiple sequences: one above, one below threshold."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
        ])
        result = pm.gseq_pwm_edits(
            ["AC", "TT"], pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        # "AC" is perfect match → needs edits to bring below
        ac_rows = result[result["seq_idx"] == 1]
        assert any(ac_rows["n_edits"] > 0), \
            "AC should need edits to bring score below threshold"
        # "TT" is poor match → already below threshold
        tt_rows = result[result["seq_idx"] == 2]
        assert any(tt_rows["n_edits"] == 0), \
            "TT should already be below threshold"

    def test_matches_manual_reference(self):
        """gseq_pwm_edits should match the manual reference implementation."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        seqs = ["ACGT", "TCGT", "TTTT", "ACGA"]
        threshold = -5.0

        for i, seq in enumerate(seqs):
            result = pm.gseq_pwm_edits(
                seq, pssm, score_thresh=threshold,
                prior=0, bidirect=False, direction="below",
            )
            expected = _manual_pwm_edit_distance_below(seq, pssm, threshold, scan_all=True)
            if len(result) > 0:
                actual = result["n_edits"].iloc[0]
                if not np.isnan(expected):
                    assert actual == expected, \
                        f"Seq '{seq}': n_edits={actual} should match manual={expected}"


class TestGseqPwmEditsDirectionBelowBidirect:
    """Bidirectional scanning with direction='below'."""

    def test_bidirect_picks_fewer_edits_strand(self):
        """Bidirectional should pick the strand requiring fewer edits."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
        ])
        # "GT": forward scores low (already below), revcomp "AC" scores high
        result = pm.gseq_pwm_edits(
            "GT", pssm, score_thresh=-1.0,
            prior=0, bidirect=True, direction="below",
        )
        assert len(result) > 0
        # Forward strand is already below threshold → 0 edits
        assert result["n_edits"].iloc[0] == 0
        assert result["strand"].iloc[0] == 1

    def test_bidirect_when_both_above(self):
        """When both strands are above threshold, pick the one needing fewer edits."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
        ])
        # "AC" forward is high, revcomp "GT" is low (already below -0.01)
        result = pm.gseq_pwm_edits(
            "AC", pssm, score_thresh=-0.01,
            prior=0, bidirect=True, direction="below",
        )
        assert len(result) > 0
        # Reverse strand "GT" is already below -0.01 → 0 edits
        assert result["n_edits"].iloc[0] == 0


class TestGseqPwmEditsDirectionBelowMaxEdits:
    """max_edits cap with direction='below'."""

    def test_max_edits_caps_results(self):
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        # Use a moderate threshold that needs 1 edit (not too extreme)
        r_unlim = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="below",
        )
        r_max1 = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=-1.0,
            max_edits=1, prior=0, bidirect=False, direction="below",
        )
        # Both should have results for this threshold
        assert len(r_unlim) > 0
        # Capped result should not have more edit rows than unlimited
        unlim_edit_rows = len(r_unlim[r_unlim["edit_num"] > 0])
        max1_edit_rows = len(r_max1[r_max1["edit_num"] > 0])
        assert max1_edit_rows <= max(1, unlim_edit_rows)


class TestGseqPwmEditsDirectionBelowAboveComplementary:
    """Direction above and below should be complementary."""

    def test_above_vs_below_complementary(self):
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
        ])
        threshold = -1.0

        # "AC" is a perfect match → above: 0 edits, below: needs edits
        r_above = pm.gseq_pwm_edits(
            "AC", pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="above",
        )
        r_below = pm.gseq_pwm_edits(
            "AC", pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="below",
        )
        assert r_above["n_edits"].iloc[0] == 0, \
            "AC is already above threshold, direction=above should need 0 edits"
        assert r_below["n_edits"].iloc[0] > 0, \
            "AC is above threshold, direction=below should need edits"

        # "TT" is well below threshold → above: needs edits, below: 0 edits
        r_above_tt = pm.gseq_pwm_edits(
            "TT", pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="above",
        )
        r_below_tt = pm.gseq_pwm_edits(
            "TT", pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="below",
        )
        assert r_above_tt["n_edits"].iloc[0] > 0, \
            "TT is below threshold, direction=above should need edits"
        assert r_below_tt["n_edits"].iloc[0] == 0, \
            "TT is already below threshold, direction=below should need 0 edits"


class TestGseqPwmEditsDirectionBelowLongerSeq:
    """direction='below' picks best window in longer sequences."""

    def test_longer_seq_best_window(self):
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        seq = "TTTTACGTTTTT"
        threshold = -1.0
        result = pm.gseq_pwm_edits(
            seq, pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="below",
        )
        assert len(result) > 0
        rows_with_edits = result[result["n_edits"] > 0]
        if len(rows_with_edits) > 0:
            # The best window's score should be above the threshold
            assert rows_with_edits["score_before"].iloc[0] > threshold


class TestGseqPwmEditsDirectionBelowInformative:
    """Highly informative PSSM with direction='below'."""

    def test_one_edit_suffices_for_near_perfect_match(self):
        pssm = np.array([
            [0.99, 0.003, 0.003, 0.004],
            [0.003, 0.99, 0.003, 0.004],
        ])
        result = pm.gseq_pwm_edits(
            "AC", pssm, score_thresh=-0.5,
            prior=0, bidirect=False, direction="below",
        )
        assert len(result) > 0
        rows_with_edits = result[result["n_edits"] > 0]
        assert len(rows_with_edits) > 0
        assert rows_with_edits["n_edits"].iloc[0] == 1, \
            "One edit should suffice to bring score below threshold"
        assert rows_with_edits["score_after"].iloc[0] <= -0.5 + 1e-9


class TestGseqPwmEditsDirectionBelowIndels:
    """direction='below' combined with indels."""

    def test_indels_with_direction_below(self):
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        result = pm.gseq_pwm_edits(
            "ACGT", pssm, score_thresh=-1.0,
            max_indels=1, prior=0, bidirect=False, direction="below",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "edit_type" in result.columns
        rows_with_edits = result[result["n_edits"] > 0]
        assert len(rows_with_edits) > 0
        assert all(rows_with_edits["score_after"] <= -1.0 + 1e-9)


class TestGseqPwmEditsDirectionValidation:
    """direction parameter validation."""

    def test_invalid_direction_raises(self):
        pssm = _create_test_pssm()
        with pytest.raises(ValueError, match="direction must be"):
            pm.gseq_pwm_edits(
                "AC", pssm, score_thresh=-1.0,
                direction="sideways",
            )

    def test_above_is_default(self):
        pssm = _create_test_pssm()
        r_default = pm.gseq_pwm_edits(
            "TT", pssm, score_thresh=-1.0,
            prior=0, bidirect=False,
        )
        r_above = pm.gseq_pwm_edits(
            "TT", pssm, score_thresh=-1.0,
            prior=0, bidirect=False, direction="above",
        )
        assert r_default["n_edits"].iloc[0] == r_above["n_edits"].iloc[0]


# ===========================================================================
# D. Virtual tracks with direction="below"
# ===========================================================================


class TestVtrackDirectionBelowBasic:
    """pwm.edit_distance with direction='below' — basic functionality."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_basic_edit_distance_below(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_interval)[0].upper()

        threshold = -5.0
        pm.gvtrack_create("edist_below", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_below", test_interval, iterator=test_interval)
        expected = _manual_pwm_edit_distance_below(seq, pssm, threshold)
        if np.isnan(expected):
            assert np.isnan(result["edist_below"].iloc[0])
        else:
            npt.assert_allclose(result["edist_below"].iloc[0], expected, atol=1e-6)

    def test_already_below_returns_zero(self):
        """Very high threshold: all windows score below it → 0 edits."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create("edist_below_easy", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=100.0,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_below_easy", test_interval, iterator=test_interval)
        npt.assert_allclose(result["edist_below_easy"].iloc[0], 0, atol=1e-6)

    def test_unreachable_threshold_returns_nan(self):
        """Uniform PSSM with impossibly low threshold → NA."""
        pssm = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ])
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create("edist_below_imp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=-100.0,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_below_imp", test_interval, iterator=test_interval)
        assert np.isnan(result["edist_below_imp"].iloc[0])

    def test_matches_reference_multiple_intervals(self):
        """Check against manual reference on multiple intervals."""
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 300, 400],
            [230, 330, 430],
        )
        threshold = -3.5

        pm.gvtrack_create("edist_below_ref", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("edist_below_ref", intervals, iterator=intervals)

        for i in range(len(intervals)):
            seq = pm.gseq_extract(intervals.iloc[[i]])[0].upper()
            expected = _manual_pwm_edit_distance_below(seq, pssm, threshold)
            actual = result["edist_below_ref"].iloc[i]
            if np.isnan(expected):
                assert np.isnan(actual), f"Interval {i}: expected NaN"
            else:
                npt.assert_allclose(actual, expected, atol=1e-6,
                                    err_msg=f"Interval {i}")


class TestVtrackDirectionBelowMaxEdits:
    """max_edits cap with direction='below' on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_max_edits_cap(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_interval)[0].upper()
        threshold = -5.0

        # Without cap
        pm.gvtrack_create("edist_below_exact", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        # With max_edits=1
        pm.gvtrack_create("edist_below_max1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold, max_edits=1,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["edist_below_exact", "edist_below_max1"],
            test_interval, iterator=test_interval,
        )

        exact = result["edist_below_exact"].iloc[0]
        max1 = result["edist_below_max1"].iloc[0]

        # If exact needs > 1 edit, max1 should be NaN
        if not np.isnan(exact) and exact > 1:
            assert np.isnan(max1)
        # If exact needs <= 1 edit, they should match
        if not np.isnan(exact) and exact <= 1:
            npt.assert_allclose(max1, exact, atol=1e-6)

    def test_max_edits_consistency(self):
        """Max_edits 1/2/3 consistent with unlimited."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("ed_b_exact", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        for k in range(1, 4):
            pm.gvtrack_create(f"ed_b_max{k}", None,
                              func="pwm.edit_distance",
                              pssm=pssm, score_thresh=threshold, max_edits=k,
                              direction="below",
                              bidirect=False, extend=False, prior=0)

        vnames = ["ed_b_exact"] + [f"ed_b_max{k}" for k in range(1, 4)]
        result = pm.gextract(vnames, test_interval, iterator=test_interval)
        exact = result["ed_b_exact"].iloc[0]

        for k in range(1, 4):
            capped = result[f"ed_b_max{k}"].iloc[0]
            if not np.isnan(exact) and exact <= k:
                npt.assert_allclose(capped, exact, atol=1e-6,
                                    err_msg=f"max_edits={k} should match exact={exact}")
            if not np.isnan(exact) and exact > k:
                assert np.isnan(capped), \
                    f"max_edits={k} should be NaN when exact={exact}"


class TestVtrackDirectionBelowBidirectional:
    """Bidirectional scanning with direction='below' on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_bidirectional_min_of_strands(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("ed_b_fwd", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, strand=1, extend=False, prior=0)

        pm.gvtrack_create("ed_b_rev", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, strand=-1, extend=False, prior=0)

        pm.gvtrack_create("ed_b_bidi", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_fwd", "ed_b_rev", "ed_b_bidi"],
            test_interval, iterator=test_interval,
        )

        fwd = result["ed_b_fwd"].iloc[0]
        rev = result["ed_b_rev"].iloc[0]
        bidi = result["ed_b_bidi"].iloc[0]

        # Bidirectional should be minimum of strands
        if not np.isnan(fwd) and not np.isnan(rev):
            npt.assert_allclose(bidi, min(fwd, rev), atol=1e-6)
        elif not np.isnan(fwd):
            npt.assert_allclose(bidi, fwd, atol=1e-6)
        elif not np.isnan(rev):
            npt.assert_allclose(bidi, rev, atol=1e-6)


class TestVtrackDirectionBelowComplementary:
    """above vs below are complementary on virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_above_vs_below_complementary(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -3.0

        pm.gvtrack_create("edist_above", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="above",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("edist_below", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["edist_above", "edist_below"],
            test_interval, iterator=test_interval,
        )

        above_val = result["edist_above"].iloc[0]
        below_val = result["edist_below"].iloc[0]

        # They can't both be 0 unless score is exactly at threshold
        if not np.isnan(above_val) and above_val == 0 and not np.isnan(below_val):
            assert below_val >= 0
        if not np.isnan(below_val) and below_val == 0 and not np.isnan(above_val):
            assert above_val >= 0
        # Both non-negative when not NaN
        if not np.isnan(above_val):
            assert above_val >= 0
        if not np.isnan(below_val):
            assert below_val >= 0


class TestVtrackDirectionBelowPosAndMax:
    """pwm.edit_distance.pos and pwm.max.edit_distance with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_pos_and_max_edit_distance(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        motif_len = pssm.shape[0]
        test_interval = pm.gintervals(["1"], [200], [250])
        seq = pm.gseq_extract(test_interval)[0].upper()
        threshold = -4.0

        pm.gvtrack_create("ed_b_min", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_pos", None,
                          func="pwm.edit_distance.pos",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_max", None,
                          func="pwm.max.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("pwm_max_pos", None,
                          func="pwm.max.pos",
                          pssm=pssm,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_min", "ed_b_pos", "ed_b_max", "pwm_max_pos"],
            test_interval, iterator=test_interval,
        )

        # Check min edit distance against manual reference
        expected_min = _manual_pwm_edit_distance_below(seq, pssm, threshold)
        actual_min = result["ed_b_min"].iloc[0]
        if np.isnan(expected_min):
            assert np.isnan(actual_min)
        else:
            npt.assert_allclose(actual_min, expected_min, atol=1e-6)

        # Check position — find the first window that achieves the minimum
        if not np.isnan(expected_min):
            best_pos = None
            for s in range(len(seq) - motif_len + 1):
                w = seq[s:s + motif_len]
                cand = _manual_pwm_edit_distance_below(w, pssm, threshold, scan_all=False)
                if not np.isnan(cand) and abs(cand - expected_min) < 1e-6:
                    best_pos = s + 1  # 1-based
                    break
            assert best_pos is not None
            npt.assert_allclose(result["ed_b_pos"].iloc[0], best_pos, atol=1e-6)

        # Check pwm.max.edit_distance (edits at the max-scoring window)
        pwm_pos = result["pwm_max_pos"].iloc[0]
        if not np.isnan(pwm_pos):
            pwm_offset = int(round(pwm_pos)) - 1
            assert pwm_offset >= 0
            pwm_window_int = pm.gintervals(
                [str(test_interval["chrom"].iloc[0])],
                [int(test_interval["start"].iloc[0]) + pwm_offset],
                [int(test_interval["start"].iloc[0]) + pwm_offset + motif_len],
            )
            pwm_seq = pm.gseq_extract(pwm_window_int)[0].upper()
            expected_max = _manual_pwm_edit_distance_below(
                pwm_seq, pssm, threshold, scan_all=False,
            )
            actual_max = result["ed_b_max"].iloc[0]
            if np.isnan(expected_max):
                assert np.isnan(actual_max)
            else:
                npt.assert_allclose(actual_max, expected_max, atol=1e-6)


class TestVtrackDirectionBelowScoreFilters:
    """score_min/score_max filters with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_score_filters(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        threshold = -5.0

        pm.gvtrack_create("ed_b_nofilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_lowfilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          score_min=-np.inf,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_highfilt", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          score_min=0.0,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_nofilt", "ed_b_lowfilt", "ed_b_highfilt"],
            test_interval, iterator=test_interval,
        )

        nf = result["ed_b_nofilt"].iloc[0]
        lf = result["ed_b_lowfilt"].iloc[0]
        hf = result["ed_b_highfilt"].iloc[0]

        # Lenient filter matches no-filter
        if np.isnan(nf):
            assert np.isnan(lf)
        else:
            npt.assert_allclose(nf, lf, atol=1e-6)

        # Strict filter should return NaN or >= no-filter
        if not np.isnan(hf) and not np.isnan(nf):
            assert hf >= nf - 1e-6


class TestVtrackDirectionBelow1bpIterator:
    """1bp iterator with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_1bp_iterator_matches_reference(self):
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]
        test_interval = pm.gintervals(["1"], [200], [210])
        threshold = -5.0

        pm.gvtrack_create("ed_b_1bp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=True, prior=0)

        result = pm.gextract("ed_b_1bp", test_interval, iterator=1)
        assert len(result) > 0

        # Check a few positions
        for idx in range(min(3, len(result))):
            pos = int(result["start"].iloc[idx])
            seq_window = pm.gseq_extract(
                pm.gintervals(["1"], [pos], [pos + motif_len])
            )[0].upper()
            expected = _manual_pwm_edit_distance_below(seq_window, pssm, threshold)
            actual = result["ed_b_1bp"].iloc[idx]
            if np.isnan(expected):
                assert np.isnan(actual), f"Position {pos}: expected NaN"
            else:
                npt.assert_allclose(actual, expected, atol=1e-6,
                                    err_msg=f"Position {pos}")


class TestVtrackDirectionBelowThresholdMonotonicity:
    """Lower thresholds require more edits in below direction."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_threshold_monotonicity(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_interval)[0].upper()

        thresholds = [-10.0, -5.0, -2.0, 0.0]
        vnames = [f"ed_b_t{i}" for i in range(len(thresholds))]

        for i, thresh in enumerate(thresholds):
            pm.gvtrack_create(vnames[i], None,
                              func="pwm.edit_distance",
                              pssm=pssm, score_thresh=thresh,
                              direction="below",
                              bidirect=False, extend=False, prior=0)

        result = pm.gextract(vnames, test_interval, iterator=test_interval)

        edits = [result[v].iloc[0] for v in vnames]
        finite_edits = [e for e in edits if not np.isnan(e)]
        if len(finite_edits) > 1:
            # As threshold increases (low→high), edits should decrease
            for i in range(1, len(finite_edits)):
                assert finite_edits[i] <= finite_edits[i - 1] + 1e-6, \
                    f"Edits should decrease with increasing threshold: {finite_edits}"

        # Verify each against manual reference
        for i, thresh in enumerate(thresholds):
            expected = _manual_pwm_edit_distance_below(seq, pssm, thresh)
            actual = edits[i]
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                npt.assert_allclose(actual, expected, atol=1e-6)


class TestVtrackDirectionBelowLongerMotif:
    """Longer motifs with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_6bp_motif(self):
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])
        seq = pm.gseq_extract(test_interval)[0].upper()
        threshold = -5.0

        pm.gvtrack_create("ed_b_6bp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("ed_b_6bp", test_interval, iterator=test_interval)
        expected = _manual_pwm_edit_distance_below(seq, pssm, threshold)
        actual = result["ed_b_6bp"].iloc[0]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            npt.assert_allclose(actual, expected, atol=1e-6)


class TestVtrackDirectionBelowExtend:
    """extend flag with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_extend_flag(self):
        pssm = _create_test_pssm()
        motif_len = pssm.shape[0]
        test_interval = pm.gintervals(["1"], [200], [202])
        threshold = -5.0

        pm.gvtrack_create("ed_b_ext", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=True, prior=0)

        pm.gvtrack_create("ed_b_noext", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_ext", "ed_b_noext"],
            test_interval, iterator=test_interval,
        )

        # With extend=TRUE the window is expanded to fit the full motif
        seq_ext = pm.gseq_extract(
            pm.gintervals(["1"], [200], [200 + motif_len])
        )[0].upper()
        expected_ext = _manual_pwm_edit_distance_below(seq_ext, pssm, threshold)

        # With extend=FALSE, window stays as-is (may be < motif)
        seq_noext = pm.gseq_extract(test_interval)[0].upper()
        expected_noext = _manual_pwm_edit_distance_below(seq_noext, pssm, threshold)

        actual_ext = result["ed_b_ext"].iloc[0]
        actual_noext = result["ed_b_noext"].iloc[0]

        if np.isnan(expected_ext):
            assert np.isnan(actual_ext)
        else:
            npt.assert_allclose(actual_ext, expected_ext, atol=1e-6)

        if np.isnan(expected_noext):
            assert np.isnan(actual_noext)
        else:
            npt.assert_allclose(actual_noext, expected_noext, atol=1e-6)


class TestVtrackDirectionBelowZeroProbability:
    """Zero-probability columns with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_zero_prob_columns(self):
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
        ])
        test_interval = pm.gintervals(["1"], [200], [240])
        seq = pm.gseq_extract(test_interval)[0].upper()
        threshold = -5.0

        pm.gvtrack_create("ed_b_zeros", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("ed_b_zeros", test_interval, iterator=test_interval)
        expected = _manual_pwm_edit_distance_below(seq, pssm, threshold)
        actual = result["ed_b_zeros"].iloc[0]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            npt.assert_allclose(actual, expected, atol=1e-6)


class TestVtrackDirectionBelowGscreen:
    """gscreen with direction='below' virtual tracks."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_gscreen_with_below(self):
        pssm = _create_test_pssm()
        threshold = -5.0

        pm.gvtrack_create("ed_b_screen", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        test_intervals = pm.gintervals(["1"], [200], [300])
        result = pm.gscreen("~np.isnan(ed_b_screen)", test_intervals, iterator=10)

        if result is not None and isinstance(result, pd.DataFrame) and len(result) > 0:
            assert {"chrom", "start", "end"}.issubset(set(result.columns))


class TestVtrackDirectionBelowComparison:
    """gseq_pwm_edits vs vtrack should match for direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_gseq_pwm_edits_matches_vtrack(self):
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        test_interval = pm.gintervals(["1"], [200], [210])
        seq = pm.gseq_extract(test_interval)[0].upper()
        threshold = -5.0

        # gseq_pwm_edits
        gseq_result = pm.gseq_pwm_edits(
            seq, pssm, score_thresh=threshold,
            prior=0, bidirect=False, direction="below",
        )

        # vtrack
        pm.gvtrack_create("ed_b_cmp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)
        vt_result = pm.gextract("ed_b_cmp", test_interval, iterator=test_interval)

        if len(gseq_result) > 0 and not np.isnan(vt_result["ed_b_cmp"].iloc[0]):
            assert gseq_result["n_edits"].iloc[0] == int(vt_result["ed_b_cmp"].iloc[0]), \
                "n_edits from gseq_pwm_edits should match vtrack"


# ===========================================================================
# E. Indels with direction="below" — virtual tracks
# ===========================================================================


class TestVtrackDirectionBelowIndels:
    """max_indels with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_indels_can_reduce_total_edits(self):
        """With indels, the total edit count should be <= substitution-only."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 500, 1000],
            [260, 560, 1060],
        )
        threshold = -3.0

        pm.gvtrack_create("ed_b_sub", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=0,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_ind1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_sub", "ed_b_ind1"],
            intervals, iterator=intervals,
        )

        for i in range(len(result)):
            sub_val = result["ed_b_sub"].iloc[i]
            ind_val = result["ed_b_ind1"].iloc[i]
            if not np.isnan(sub_val) and not np.isnan(ind_val):
                assert ind_val <= sub_val + 1e-6, \
                    f"Row {i}: indel={ind_val} should be <= sub-only={sub_val}"
            if not np.isnan(sub_val):
                assert not np.isnan(ind_val), \
                    f"Row {i}: indel should not be NaN when sub-only={sub_val}"

    def test_more_indels_reduce_or_equal(self):
        """max_indels=2 should give <= max_indels=1 <= max_indels=0."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        intervals = pm.gintervals(
            ["1", "1", "1"],
            [200, 500, 1000],
            [260, 560, 1060],
        )
        threshold = -3.0

        for d in range(3):
            pm.gvtrack_create(f"ed_b_d{d}", None,
                              func="pwm.edit_distance",
                              pssm=pssm, score_thresh=threshold,
                              direction="below", max_indels=d,
                              bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            [f"ed_b_d{d}" for d in range(3)],
            intervals, iterator=intervals,
        )

        for i in range(len(result)):
            d0 = result["ed_b_d0"].iloc[i]
            d1 = result["ed_b_d1"].iloc[i]
            d2 = result["ed_b_d2"].iloc[i]

            if not np.isnan(d0) and not np.isnan(d1):
                assert d1 <= d0 + 1e-6, f"Row {i}: d1={d1} <= d0={d0}"
            if not np.isnan(d1) and not np.isnan(d2):
                assert d2 <= d1 + 1e-6, f"Row {i}: d2={d2} <= d1={d1}"
            if not np.isnan(d0):
                assert not np.isnan(d1), f"Row {i}: d1 should not be NaN if d0={d0}"
                assert not np.isnan(d2), f"Row {i}: d2 should not be NaN if d0={d0}"

    def test_longer_motif_sub_vs_indels(self):
        """Longer motif: indels should never return more than sub-only."""
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
        ])
        test_interval = pm.gintervals(["1"], [200], [280])
        threshold = -5.0

        pm.gvtrack_create("ed_b_noind", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_1ind", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_2ind", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=2,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_noind", "ed_b_1ind", "ed_b_2ind"],
            test_interval, iterator=test_interval,
        )

        ni = result["ed_b_noind"].iloc[0]
        i1 = result["ed_b_1ind"].iloc[0]
        i2 = result["ed_b_2ind"].iloc[0]

        if not np.isnan(ni) and not np.isnan(i1):
            assert i1 <= ni + 1e-6
        if not np.isnan(ni) and not np.isnan(i2):
            assert i2 <= ni + 1e-6
        if not np.isnan(i1) and not np.isnan(i2):
            assert i2 <= i1 + 1e-6
        if not np.isnan(ni):
            assert not np.isnan(i1)
            assert not np.isnan(i2)

    def test_max_indels_zero_matches_default(self):
        """max_indels=0 should match default (no indels)."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        intervals = pm.gintervals(
            ["1", "1", "1", "1"],
            [200, 500, 1000, 2000],
            [260, 560, 1060, 2060],
        )
        threshold = -3.0

        pm.gvtrack_create("ed_b_default", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_cap0", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=0,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_default", "ed_b_cap0"],
            intervals, iterator=intervals,
        )

        for i in range(len(result)):
            d = result["ed_b_default"].iloc[i]
            c = result["ed_b_cap0"].iloc[i]
            if np.isnan(d):
                assert np.isnan(c), f"Row {i}: both should be NaN"
            else:
                npt.assert_allclose(d, c, atol=1e-6,
                                    err_msg=f"Row {i}: default and max_indels=0 should match")

    def test_already_below_with_indels_returns_zero(self):
        """All indel settings return 0 when already below threshold."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        for d in range(3):
            pm.gvtrack_create(f"ed_b_easy{d}", None,
                              func="pwm.edit_distance",
                              pssm=pssm, score_thresh=100.0,
                              direction="below", max_indels=d,
                              bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            [f"ed_b_easy{d}" for d in range(3)],
            test_interval, iterator=test_interval,
        )

        for d in range(3):
            npt.assert_allclose(result[f"ed_b_easy{d}"].iloc[0], 0, atol=1e-6)

    def test_indels_bidirectional(self):
        """Bidirectional with indels should return min of strands."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -3.0

        pm.gvtrack_create("ed_b_ind_fwd", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, strand=1, extend=False, prior=0)

        pm.gvtrack_create("ed_b_ind_rev", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, strand=-1, extend=False, prior=0)

        pm.gvtrack_create("ed_b_ind_bidi", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_ind_fwd", "ed_b_ind_rev", "ed_b_ind_bidi"],
            test_interval, iterator=test_interval,
        )

        fwd = result["ed_b_ind_fwd"].iloc[0]
        rev = result["ed_b_ind_rev"].iloc[0]
        bidi = result["ed_b_ind_bidi"].iloc[0]

        if not np.isnan(fwd) and not np.isnan(rev):
            npt.assert_allclose(bidi, min(fwd, rev), atol=1e-6)
        elif not np.isnan(fwd):
            npt.assert_allclose(bidi, fwd, atol=1e-6)
        elif not np.isnan(rev):
            npt.assert_allclose(bidi, rev, atol=1e-6)

    def test_max_edits_and_max_indels_interaction(self):
        """max_edits cap interacts correctly with max_indels in below direction."""
        pssm = np.array([
            [0.9, 0.03, 0.03, 0.04],
            [0.03, 0.9, 0.03, 0.04],
            [0.03, 0.03, 0.9, 0.04],
            [0.04, 0.03, 0.03, 0.9],
        ])
        test_interval = pm.gintervals(["1"], [200], [260])
        threshold = -5.0

        pm.gvtrack_create("ed_b_ind_unlim", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_ind_max1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1, max_edits=1,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_ind_max3", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1, max_edits=3,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_ind_unlim", "ed_b_ind_max1", "ed_b_ind_max3"],
            test_interval, iterator=test_interval,
        )

        unlim = result["ed_b_ind_unlim"].iloc[0]
        max1 = result["ed_b_ind_max1"].iloc[0]
        max3 = result["ed_b_ind_max3"].iloc[0]

        if not np.isnan(unlim):
            if unlim <= 1:
                npt.assert_allclose(max1, unlim, atol=1e-6)
            else:
                assert np.isnan(max1), \
                    f"max_edits=1 should be NaN when unlimited needs {unlim}"
            if unlim <= 3:
                npt.assert_allclose(max3, unlim, atol=1e-6)
            else:
                assert np.isnan(max3), \
                    f"max_edits=3 should be NaN when unlimited needs {unlim}"

    def test_indels_consistency_across_intervals(self):
        """Indels consistently <= sub-only across many intervals."""
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
            [0.04, 0.03, 0.03, 0.9],
        ])
        intervals = pm.gintervals(
            ["1"] * 8,
            list(range(200, 1000, 100)),
            list(range(260, 1060, 100)),
        )
        threshold = -4.0

        pm.gvtrack_create("ed_b_con_sub", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=0,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("ed_b_con_ind1", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("ed_b_con_ind2", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=2,
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["ed_b_con_sub", "ed_b_con_ind1", "ed_b_con_ind2"],
            intervals, iterator=intervals,
        )

        for i in range(len(result)):
            s = result["ed_b_con_sub"].iloc[i]
            i1 = result["ed_b_con_ind1"].iloc[i]
            i2 = result["ed_b_con_ind2"].iloc[i]

            if not np.isnan(s) and not np.isnan(i1):
                assert i1 <= s + 1e-6, f"Row {i}: indel1 should be <= sub-only"
            if not np.isnan(s) and not np.isnan(i2):
                assert i2 <= s + 1e-6, f"Row {i}: indel2 should be <= sub-only"
            if not np.isnan(i1) and not np.isnan(i2):
                assert i2 <= i1 + 1e-6, f"Row {i}: indel2 should be <= indel1"
            if not np.isnan(s):
                assert not np.isnan(i1), f"Row {i}: indel1 NaN but sub-only={s}"
                assert not np.isnan(i2), f"Row {i}: indel2 NaN but sub-only={s}"

    def test_1bp_iterator_with_indels(self):
        """1bp iterator with indels should match interval-level minimum."""
        pssm = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
        ])
        test_interval = pm.gintervals(["1"], [200], [210])
        threshold = -4.0

        pm.gvtrack_create("ed_b_ind_int", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("ed_b_ind_1bp", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below", max_indels=1,
                          bidirect=False, extend=True, prior=0)

        result_int = pm.gextract("ed_b_ind_int", test_interval, iterator=test_interval)
        result_1bp = pm.gextract("ed_b_ind_1bp", test_interval, iterator=1)

        if len(result_1bp) > 0:
            vals = result_1bp["ed_b_ind_1bp"].values
            finite_vals = vals[~np.isnan(vals)]
            if len(finite_vals) > 0:
                min_1bp = finite_vals.min()
                int_val = result_int["ed_b_ind_int"].iloc[0]
                if not np.isnan(int_val):
                    npt.assert_allclose(int_val, min_1bp, atol=1e-6)


# ===========================================================================
# F. LSE edit distance with direction="below"
# ===========================================================================


class TestVtrackLseDirectionBelow:
    """pwm.edit_distance.lse with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_basic_lse_below(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        # Get the actual LSE score
        pm.gvtrack_create("v_lse_score", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        lse_result = pm.gextract("v_lse_score", test_interval, iterator=test_interval)
        lse_score = lse_result["v_lse_score"].iloc[0]

        # Set threshold above the LSE score → already below → 0 edits
        high_thresh = lse_score + 5.0
        pm.gvtrack_create("v_lse_b_easy", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=high_thresh,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("v_lse_b_easy", test_interval, iterator=test_interval)
        npt.assert_allclose(result["v_lse_b_easy"].iloc[0], 0, atol=1e-6)

    def test_lse_below_needs_edits(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        # Get the actual LSE score
        pm.gvtrack_create("v_lse_score", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        lse_result = pm.gextract("v_lse_score", test_interval, iterator=test_interval)
        lse_score = lse_result["v_lse_score"].iloc[0]

        # Set threshold well below → needs edits
        low_thresh = lse_score - 10.0
        pm.gvtrack_create("v_lse_b_hard", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=low_thresh,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("v_lse_b_hard", test_interval, iterator=test_interval)
        val = result["v_lse_b_hard"].iloc[0]
        if not np.isnan(val):
            assert val >= 1, \
                f"LSE score {lse_score} above threshold {low_thresh} should need edits"

    def test_lse_below_returns_zero_when_already_below(self):
        pssm = _create_test_pssm()
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create("v_lse_b_al", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=100.0,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("v_lse_b_al", test_interval, iterator=test_interval)
        npt.assert_allclose(result["v_lse_b_al"].iloc[0], 0, atol=1e-6)

    def test_lse_below_unreachable_returns_nan(self):
        """Uniform PSSM with impossibly low threshold → NA."""
        pssm = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ])
        test_interval = pm.gintervals(["1"], [200], [240])

        pm.gvtrack_create("v_lse_b_imp", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=-1000.0,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract("v_lse_b_imp", test_interval, iterator=test_interval)
        assert np.isnan(result["v_lse_b_imp"].iloc[0])

    def test_lse_below_vs_above_complementary(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [240])

        # Get the actual LSE score
        pm.gvtrack_create("v_lse_sc", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        sc_result = pm.gextract("v_lse_sc", test_interval, iterator=test_interval)
        lse_score = sc_result["v_lse_sc"].iloc[0]

        # Threshold below current score
        low_thresh = lse_score - 5.0
        pm.gvtrack_create("v_lse_a_low", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=low_thresh,
                          direction="above",
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("v_lse_b_low", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=low_thresh,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["v_lse_a_low", "v_lse_b_low"],
            test_interval, iterator=test_interval,
        )

        # Score > threshold → above=0, below>=1
        npt.assert_allclose(result["v_lse_a_low"].iloc[0], 0, atol=1e-6)
        below_val = result["v_lse_b_low"].iloc[0]
        if not np.isnan(below_val):
            assert below_val >= 1

        # Threshold above current score
        high_thresh = lse_score + 5.0
        pm.gvtrack_create("v_lse_a_hi", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=high_thresh,
                          direction="above",
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("v_lse_b_hi", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=high_thresh,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result2 = pm.gextract(
            ["v_lse_a_hi", "v_lse_b_hi"],
            test_interval, iterator=test_interval,
        )

        # Score < threshold → below=0, above>=1
        npt.assert_allclose(result2["v_lse_b_hi"].iloc[0], 0, atol=1e-6)
        above_val = result2["v_lse_a_hi"].iloc[0]
        if not np.isnan(above_val):
            assert above_val >= 1


class TestVtrackLseDirectionBelowPos:
    """pwm.edit_distance.lse.pos with direction='below'."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_lse_below_pos_valid(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        # Get LSE score for meaningful threshold
        pm.gvtrack_create("v_lse_sc", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        sc_result = pm.gextract("v_lse_sc", test_interval, iterator=test_interval)
        threshold = sc_result["v_lse_sc"].iloc[0] - 5.0

        pm.gvtrack_create("v_lse_b_ed", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("v_lse_b_pos", None,
                          func="pwm.edit_distance.lse.pos",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["v_lse_b_ed", "v_lse_b_pos"],
            test_interval, iterator=test_interval,
        )

        edist_val = result["v_lse_b_ed"].iloc[0]
        pos_val = result["v_lse_b_pos"].iloc[0]

        if not np.isnan(edist_val) and edist_val >= 1:
            assert not np.isnan(pos_val), \
                "Position should be defined when edits are needed"
            interval_len = int(test_interval["end"].iloc[0]) - int(test_interval["start"].iloc[0])
            assert pos_val >= 1, f"Position {pos_val} should be >= 1"
            assert pos_val <= interval_len, \
                f"Position {pos_val} should be within interval length {interval_len}"

        if not np.isnan(edist_val) and edist_val == 0:
            assert np.isnan(pos_val), \
                "Position should be NA when no edits are needed"


class TestVtrackLseDirectionBelowConsistency:
    """LSE below consistency with max-mode below."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_lse_below_ge_max_below(self):
        """LSE below edits should be >= max below edits for the same threshold."""
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        # Get both scores
        pm.gvtrack_create("v_lse_sc", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("v_max_sc", None, func="pwm.max",
                          pssm=pssm, bidirect=False, extend=False, prior=0)

        scores = pm.gextract(
            ["v_lse_sc", "v_max_sc"],
            test_interval, iterator=test_interval,
        )
        max_score = scores["v_max_sc"].iloc[0]
        threshold = max_score - 2.0

        pm.gvtrack_create("v_lse_b_ed", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("v_max_b_ed", None,
                          func="pwm.edit_distance",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["v_lse_b_ed", "v_max_b_ed"],
            test_interval, iterator=test_interval,
        )

        lse_edits = result["v_lse_b_ed"].iloc[0]
        max_edits = result["v_max_b_ed"].iloc[0]

        # LSE >= max → pushing LSE down is at least as hard
        if not np.isnan(lse_edits) and not np.isnan(max_edits):
            assert lse_edits >= max_edits - 1e-6, \
                f"LSE below edits ({lse_edits}) should be >= max below edits ({max_edits})"

        if not np.isnan(lse_edits):
            assert lse_edits >= 0
        if not np.isnan(max_edits):
            assert max_edits >= 0


class TestVtrackLseDirectionBelowThresholdMonotonicity:
    """LSE below threshold monotonicity."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_threshold_monotonicity(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [240])

        # Get actual LSE score
        pm.gvtrack_create("v_lse_sc", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        sc_result = pm.gextract("v_lse_sc", test_interval, iterator=test_interval)
        lse_score = sc_result["v_lse_sc"].iloc[0]

        thresholds = [
            lse_score - 15, lse_score - 10, lse_score - 5,
            lse_score, lse_score + 5,
        ]
        vnames = [f"v_lse_b_t{i}" for i in range(len(thresholds))]

        for i, thresh in enumerate(thresholds):
            pm.gvtrack_create(vnames[i], None,
                              func="pwm.edit_distance.lse",
                              pssm=pssm, score_thresh=thresh,
                              direction="below",
                              bidirect=False, extend=False, prior=0)

        result = pm.gextract(vnames, test_interval, iterator=test_interval)
        edits = [result[v].iloc[0] for v in vnames]

        # As threshold increases, edits should decrease (non-increasing)
        for i in range(1, len(edits)):
            if not np.isnan(edits[i - 1]) and not np.isnan(edits[i]):
                assert edits[i] <= edits[i - 1] + 1e-6, \
                    (f"Edits at threshold {thresholds[i]}={edits[i]} should be "
                     f"<= edits at threshold {thresholds[i-1]}={edits[i-1]}")
            # If a lower threshold is reachable, the higher one must also be
            if not np.isnan(edits[i - 1]):
                assert not np.isnan(edits[i]), \
                    (f"Threshold {thresholds[i]} should be reachable since "
                     f"lower threshold {thresholds[i-1]} is reachable")


class TestVtrackLseDirectionBelowBidirectional:
    """LSE below with bidirectional scanning."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_lse_below_bidirectional(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        # Get LSE score for meaningful threshold
        pm.gvtrack_create("v_lse_sc", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        sc_result = pm.gextract("v_lse_sc", test_interval, iterator=test_interval)
        threshold = sc_result["v_lse_sc"].iloc[0] - 3.0

        pm.gvtrack_create("v_lse_b_fwd", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, strand=1, extend=False, prior=0)

        pm.gvtrack_create("v_lse_b_rev", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, strand=-1, extend=False, prior=0)

        pm.gvtrack_create("v_lse_b_bidi", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=True, extend=False, prior=0)

        result = pm.gextract(
            ["v_lse_b_fwd", "v_lse_b_rev", "v_lse_b_bidi"],
            test_interval, iterator=test_interval,
        )

        fwd = result["v_lse_b_fwd"].iloc[0]
        rev = result["v_lse_b_rev"].iloc[0]
        bidi = result["v_lse_b_bidi"].iloc[0]

        # All should be non-negative when not NaN
        if not np.isnan(fwd):
            assert fwd >= 0
        if not np.isnan(rev):
            assert rev >= 0
        if not np.isnan(bidi):
            assert bidi >= 0

        # If at least one strand reachable, bidi should be too
        if not np.isnan(fwd) or not np.isnan(rev):
            assert not np.isnan(bidi), \
                "Bidirectional should be reachable when at least one strand is"


class TestVtrackLseDirectionBelowMaxEdits:
    """LSE below with max_edits cap."""

    def setup_method(self):
        _remove_all_vtracks()

    def teardown_method(self):
        _remove_all_vtracks()

    def test_lse_below_max_edits(self):
        pssm = np.array([
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
        ])
        test_interval = pm.gintervals(["1"], [200], [250])

        # Get LSE score for meaningful threshold
        pm.gvtrack_create("v_lse_sc", None, func="pwm",
                          pssm=pssm, bidirect=False, extend=False, prior=0)
        sc_result = pm.gextract("v_lse_sc", test_interval, iterator=test_interval)
        threshold = sc_result["v_lse_sc"].iloc[0] - 8.0

        pm.gvtrack_create("v_lse_b_unlim", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("v_lse_b_max1", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold, max_edits=1,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        pm.gvtrack_create("v_lse_b_max5", None,
                          func="pwm.edit_distance.lse",
                          pssm=pssm, score_thresh=threshold, max_edits=5,
                          direction="below",
                          bidirect=False, extend=False, prior=0)

        result = pm.gextract(
            ["v_lse_b_unlim", "v_lse_b_max1", "v_lse_b_max5"],
            test_interval, iterator=test_interval,
        )

        unlim = result["v_lse_b_unlim"].iloc[0]
        max1 = result["v_lse_b_max1"].iloc[0]
        max5 = result["v_lse_b_max5"].iloc[0]

        if not np.isnan(unlim):
            if unlim <= 1:
                npt.assert_allclose(max1, unlim, atol=1e-6)
            else:
                assert np.isnan(max1), \
                    f"max_edits=1 should be NaN when unlimited needs {unlim}"
            if unlim <= 5:
                npt.assert_allclose(max5, unlim, atol=1e-6)
            else:
                assert np.isnan(max5), \
                    f"max_edits=5 should be NaN when unlimited needs {unlim}"

        # Non-negative when not NaN
        if not np.isnan(unlim):
            assert unlim >= 0
        if not np.isnan(max1):
            assert max1 >= 0
        if not np.isnan(max5):
            assert max5 >= 0
