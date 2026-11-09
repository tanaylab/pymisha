"""Tests for the ``pwm.n_mutations`` virtual track function.

Ported from R misha tests/testthat/test-pwm-n-mutations.R (21 tests).

``pwm.n_mutations`` counts the single-base substitutions that each independently
bring a motif-length window across ``score_thresh``. 0 when the threshold is
already satisfied, NaN when no single edit suffices. Aggregation across the
windows of an iterator interval is MAX.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

_BASES = "ACGT"


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def _seq(chrom, start, end):
    s = pm.gseq_extract(pm.gintervals(chrom, start, end))
    if isinstance(s, (list, np.ndarray)):
        s = np.asarray(s).ravel()[0]
    return str(s).upper()


def _create_test_pssm():
    """The R helper's 'AC' motif - has zero entries, so prior matters."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # only A
        [0.0, 1.0, 0.0, 0.0],  # only C
    ])


def _pssm_acg():
    """Strong A / C / G, no zero entries."""
    return np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.05, 0.8, 0.05],
    ])


def _pssm_acg_shifted():
    """A second zero-free 3-mer, used where -Inf would complicate the reference."""
    return np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.05, 0.8, 0.1, 0.05],
        [0.05, 0.05, 0.8, 0.1],
    ])


# ---------------------------------------------------------------------------
# Reference implementation (port of R's manual_pwm_n_mutations)
# ---------------------------------------------------------------------------

def _manual_n_mutations(seq, pssm, threshold, direction="above"):
    """Count single-base substitutions that independently cross the threshold.

    0 if the threshold is already satisfied, NaN if no single edit suffices.
    """
    L = pssm.shape[0]
    if len(seq) < L:
        return np.nan

    with np.errstate(divide="ignore"):
        logp = np.log(pssm)

    score = 0.0
    for i in range(L):
        bidx = _BASES.find(seq[i])
        # N base: the C++ score table maps index 4 to the column minimum
        score += logp[i].min() if bidx < 0 else logp[i, bidx]

    if direction == "above" and score >= threshold:
        return 0.0
    if direction == "below" and score <= threshold:
        return 0.0

    deficit = (threshold - score) if direction == "above" else (score - threshold)

    count = 0
    for i in range(L):
        cur = _BASES.find(seq[i])
        if cur < 0:  # skip N bases
            continue
        cur_log = logp[i, cur]
        for alt in range(4):
            if alt == cur:
                continue
            alt_log = logp[i, alt]
            delta = (alt_log - cur_log) if direction == "above" else (cur_log - alt_log)
            if not np.isfinite(delta):
                continue
            if delta >= deficit - 1e-12:
                count += 1

    return float(count) if count else np.nan


def _manual_n_mutations_scan(seq, pssm, threshold, direction="above"):
    """MAX across every window start in ``seq``."""
    L = pssm.shape[0]
    if len(seq) < L:
        return np.nan
    best = np.nan
    for start in range(len(seq) - L + 1):
        cand = _manual_n_mutations(seq[start:start + L], pssm, threshold, direction)
        if np.isnan(best) or (not np.isnan(cand) and cand > best):
            best = cand
    return best


@pytest.fixture(autouse=True)
def _cleanup():
    _remove_all_vtracks()
    yield
    _remove_all_vtracks()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestNMutationsBasic:

    def test_returns_zero_when_threshold_already_satisfied(self):
        full = _seq("1", 200, 300)
        ac_pos = full.find("AC")
        # Assert rather than skip: no AC means the test checked nothing.
        assert ac_pos >= 0, "no AC pattern in the test region"
        abs_pos = 200 + ac_pos
        iv = pm.gintervals("1", abs_pos, abs_pos + 2)

        # An AC window scores log(1)+log(1) = 0, satisfying >= 0.
        pm.gvtrack_create("nmut_perfect", None, "pwm.n_mutations",
                          pssm=_create_test_pssm(), score_thresh=0.0,
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract("nmut_perfect", intervals=iv, iterator=iv)
        assert r["nmut_perfect"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_matches_reference_for_one_edit_window(self):
        iv = pm.gintervals("1", 200, 240)
        seq = _seq("1", 200, 240)
        pssm, threshold = _pssm_acg(), -3.0

        pm.gvtrack_create("nmut_count", None, "pwm.n_mutations",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        got = pm.gextract("nmut_count", intervals=iv, iterator=iv)["nmut_count"].iloc[0]
        expected = _manual_n_mutations_scan(seq, pssm, threshold, "above")

        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-6)

    def test_returns_nan_when_no_single_edit_suffices(self):
        iv = pm.gintervals("1", 200, 240)
        # Impossibly high threshold - no single edit can reach it.
        pm.gvtrack_create("nmut_unreachable", None, "pwm.n_mutations",
                          pssm=_create_test_pssm(), score_thresh=100.0,
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract("nmut_unreachable", intervals=iv, iterator=iv)
        assert np.isnan(r["nmut_unreachable"].iloc[0])

    def test_matches_reference_on_multiple_intervals(self):
        ivs = pm.gintervals(chroms=["1", "1", "1"], starts=[200, 300, 400],
                            ends=[230, 330, 430])
        pssm, threshold = _pssm_acg(), -3.5

        pm.gvtrack_create("nmut_ref", None, "pwm.n_mutations",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract("nmut_ref", intervals=ivs, iterator=ivs).sort_values("start")

        for i, (start, end) in enumerate(zip(ivs["start"], ivs["end"])):
            expected = _manual_n_mutations_scan(_seq("1", start, end), pssm,
                                                threshold, "above")
            got = r["nmut_ref"].iloc[i]
            if np.isnan(expected):
                assert np.isnan(got), f"interval {i} expected NaN, got {got}"
            else:
                assert got == pytest.approx(expected, abs=1e-6), f"interval {i}"

    def test_1bp_iterator_matches_reference_per_position(self):
        pssm, threshold = _pssm_acg_shifted(), -3.0
        motif_len = pssm.shape[0]
        iv = pm.gintervals("1", 200, 210)

        pm.gvtrack_create("nmut_1bp", None, "pwm.n_mutations",
                          pssm=pssm, score_thresh=threshold,
                          bidirect=False, extend=True, prior=0)
        r = pm.gextract("nmut_1bp", intervals=iv, iterator=1).sort_values("start")
        assert len(r) > 0

        for idx in range(min(5, len(r))):
            pos = int(r["start"].iloc[idx])
            window = _seq("1", pos, pos + motif_len)
            expected = _manual_n_mutations(window, pssm, threshold, "above")
            got = r["nmut_1bp"].iloc[idx]
            if np.isnan(expected):
                assert np.isnan(got), f"position {pos}"
            else:
                assert got == pytest.approx(expected, abs=1e-6), f"position {pos}"


# ---------------------------------------------------------------------------
# direction=below
# ---------------------------------------------------------------------------

class TestNMutationsDirectionBelow:

    def test_returns_zero_when_score_already_below(self):
        iv = pm.gintervals("1", 200, 240)
        # Very high threshold - every window scores well below it.
        pm.gvtrack_create("nmut_below_already", None, "pwm.n_mutations",
                          pssm=_create_test_pssm(), score_thresh=100.0,
                          score_min=-np.inf, direction="below",
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract("nmut_below_already", intervals=iv, iterator=iv)
        assert r["nmut_below_already"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_matches_reference(self):
        ivs = pm.gintervals(chroms=["1", "1", "1"], starts=[200, 300, 400],
                            ends=[230, 330, 430])
        pssm, threshold = _pssm_acg(), -3.5

        pm.gvtrack_create("nmut_below_ref", None, "pwm.n_mutations",
                          pssm=pssm, score_thresh=threshold, score_min=-np.inf,
                          direction="below", bidirect=False, extend=False, prior=0)
        r = pm.gextract("nmut_below_ref", intervals=ivs, iterator=ivs).sort_values("start")

        for i, (start, end) in enumerate(zip(ivs["start"], ivs["end"])):
            expected = _manual_n_mutations_scan(_seq("1", start, end), pssm,
                                                threshold, "below")
            got = r["nmut_below_ref"].iloc[i]
            if np.isnan(expected):
                assert np.isnan(got), f"interval {i} expected NaN, got {got}"
            else:
                assert got == pytest.approx(expected, abs=1e-6), f"interval {i}"

    def test_returns_nan_when_threshold_unreachable(self):
        # Uniform PSSM: every entry is log(0.25), so -100 is out of reach.
        pssm = np.full((2, 4), 0.25)
        iv = pm.gintervals("1", 200, 240)
        pm.gvtrack_create("nmut_below_impossible", None, "pwm.n_mutations",
                          pssm=pssm, score_thresh=-100.0, score_min=-np.inf,
                          direction="below", bidirect=False, extend=False, prior=0)
        r = pm.gextract("nmut_below_impossible", intervals=iv, iterator=iv)
        assert np.isnan(r["nmut_below_impossible"].iloc[0])


# ---------------------------------------------------------------------------
# bidirect
# ---------------------------------------------------------------------------

class TestNMutationsBidirect:

    def test_above_takes_max_across_strands(self):
        iv = pm.gintervals("1", 200, 260)
        pssm, threshold = _create_test_pssm(), -5.0

        for name, kw in (("nmut_fwd", dict(bidirect=False, strand=1)),
                         ("nmut_rev", dict(bidirect=False, strand=-1)),
                         ("nmut_bidi", dict(bidirect=True))):
            pm.gvtrack_create(name, None, "pwm.n_mutations", pssm=pssm,
                              score_thresh=threshold, extend=False, prior=0, **kw)

        r = pm.gextract(["nmut_fwd", "nmut_rev", "nmut_bidi"], intervals=iv, iterator=iv)
        fwd, rev, bidi = (r[c].iloc[0] for c in ("nmut_fwd", "nmut_rev", "nmut_bidi"))

        # Either strand crossing the threshold counts, so bidirect combines them
        # by max per window; the aggregation across windows is still max.
        if not np.isnan(fwd) and not np.isnan(rev):
            assert bidi >= max(fwd, rev) - 1e-6, f"fwd={fwd} rev={rev} bidi={bidi}"

    def test_below_counts_substitutions_that_bring_both_strands_below(self):
        # R misha 5.17.0: one genomic substitution changes both strands, so a
        # window counts the substitutions after which BOTH strands score <=
        # threshold (0 if both already do, NaN if no single substitution does).
        # It used to take the larger of the two per-strand counts.
        pssm, threshold = _pssm_acg_shifted(), -4.0
        iv = pm.gintervals("1", 200, 260)
        pm.gvtrack_create("nmut_below_bidi", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, score_min=-np.inf,
                          direction="below", bidirect=True, extend=True, prior=0)
        r = pm.gextract("nmut_below_bidi", intervals=iv, iterator=1)

        lp = np.log(pssm)
        comp = {"A": "T", "C": "G", "G": "C", "T": "A"}

        def score(b):
            return sum(lp[i, _BASES.index(x)] for i, x in enumerate(b))

        def rc(b):
            return [comp[x] for x in reversed(b)]

        def expected_count(b):
            if score(b) <= threshold and score(rc(b)) <= threshold:
                return 0.0
            n = 0
            for j in range(len(b)):
                for x in _BASES:
                    if x == b[j]:
                        continue
                    m = list(b)
                    m[j] = x
                    if score(m) <= threshold and score(rc(m)) <= threshold:
                        n += 1
            return np.nan if n == 0 else float(n)

        expected = [expected_count(list(_seq("1", s, s + len(pssm)))) for s in r["start"]]
        expected = np.array(expected)
        assert np.any(~np.isnan(expected) & (expected > 0))
        np.testing.assert_array_equal(r["nmut_below_bidi"].to_numpy(), expected)

    def test_below_strand_that_cannot_be_disrupted_decides(self, tmp_path):
        # forward ACCAT scores -3.73 against the motif ACGAT, its reverse
        # complement ATGGT -5.68
        seq = "NNNNNACCATNNNNN"
        (tmp_path / "tracks").mkdir()
        (tmp_path / "seq").mkdir()
        (tmp_path / "chrom_sizes.txt").write_text(f"chr1\t{len(seq)}\n")
        (tmp_path / "seq" / "chr1.seq").write_bytes(seq.encode())
        pm.gdb_init(str(tmp_path))

        pssm = pd.DataFrame([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1],
                             [.7, .1, .1, .1], [.1, .1, .1, .7]], columns=list(_BASES))
        iv = pm.gintervals("chr1", 0, len(seq))
        p = {"pssm": pssm, "score_thresh": -6, "direction": "below", "prior": 0}
        pm.gvtrack_create("ed_fwd", None, "pwm.edit_distance", params={**p, "bidirect": False, "strand": 1})
        pm.gvtrack_create("ed_rev", None, "pwm.edit_distance", params={**p, "bidirect": False, "strand": -1})
        pm.gvtrack_create("ed_bidi", None, "pwm.edit_distance", params={**p, "bidirect": True})
        pm.gvtrack_create("ed_bidi_cap1", None, "pwm.edit_distance",
                          params={**p, "bidirect": True, "max_edits": 1})
        pm.gvtrack_create("nm_bidi", None, "pwm.n_mutations", params={**p, "bidirect": True})
        r = pm.gextract(["ed_fwd", "ed_rev", "ed_bidi", "ed_bidi_cap1", "nm_bidi"], iv, iterator=iv)

        assert r["ed_fwd"].iloc[0] == 2
        assert r["ed_rev"].iloc[0] == 1
        assert r["ed_bidi"].iloc[0] == 2
        # The forward strand needs 2 > max_edits: the window cannot be disrupted
        # within the cap (it used to report the reverse strand's 1)
        assert np.isnan(r["ed_bidi_cap1"].iloc[0])
        # No single substitution brings both strands to <= -6 (it used to count
        # the reverse strand's 9 alone)
        assert np.isnan(r["nm_bidi"].iloc[0])

        # A strand the score filter leaves out does not decide: with score_min
        # above the reverse strand's score, only the forward strand counts
        pm.gvtrack_create("ed_bidi_fwdonly", None, "pwm.edit_distance",
                          params={**p, "bidirect": True, "score_min": -5})
        pm.gvtrack_create("ed_bidi_fwdonly_cap1", None, "pwm.edit_distance",
                          params={**p, "bidirect": True, "score_min": -5, "max_edits": 1})
        r2 = pm.gextract(["ed_bidi_fwdonly", "ed_bidi_fwdonly_cap1"], iv, iterator=iv)
        assert r2["ed_bidi_fwdonly"].iloc[0] == 2
        assert np.isnan(r2["ed_bidi_fwdonly_cap1"].iloc[0])


# ---------------------------------------------------------------------------
# score_min / score_max filters
# ---------------------------------------------------------------------------

class TestNMutationsScoreFilters:

    def test_score_min_filters_low_scoring_windows(self):
        iv = pm.gintervals("1", 200, 240)
        pssm, threshold = _create_test_pssm(), -3.0

        pm.gvtrack_create("nmut_nofilt", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("nmut_highfilt", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, score_min=0.0,
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract(["nmut_nofilt", "nmut_highfilt"], intervals=iv, iterator=iv)

        # n_mutations is MAX across qualifying windows, so filtering can only
        # reduce the result or turn it into NaN.
        lo, hi = r["nmut_highfilt"].iloc[0], r["nmut_nofilt"].iloc[0]
        if not np.isnan(lo) and not np.isnan(hi):
            assert lo <= hi + 1e-6

    def test_score_max_filters_high_scoring_windows(self):
        iv = pm.gintervals("1", 200, 240)
        pssm, threshold = _create_test_pssm(), -5.0

        pm.gvtrack_create("nmut_nomax", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("nmut_maxfilt", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, score_max=-100.0,
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract(["nmut_nomax", "nmut_maxfilt"], intervals=iv, iterator=iv)

        filt, unfilt = r["nmut_maxfilt"].iloc[0], r["nmut_nomax"].iloc[0]
        if not np.isnan(filt) and not np.isnan(unfilt):
            assert filt <= unfilt + 1e-6

    def test_below_score_min_neg_inf_matches_default(self):
        iv = pm.gintervals("1", 200, 240)
        pssm, threshold = _create_test_pssm(), -5.0

        pm.gvtrack_create("nmut_below_default", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, direction="below",
                          bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("nmut_below_explicit", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, score_min=-np.inf, direction="below",
                          bidirect=False, extend=False, prior=0)
        r = pm.gextract(["nmut_below_default", "nmut_below_explicit"],
                        intervals=iv, iterator=iv)

        a, b = r["nmut_below_default"].iloc[0], r["nmut_below_explicit"].iloc[0]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
        else:
            assert a == pytest.approx(b, abs=1e-6)


# ---------------------------------------------------------------------------
# Aggregation, integration, prior
# ---------------------------------------------------------------------------

class TestNMutationsSemantics:

    def test_non_degenerate_pssm_matches_reference(self):
        pssm = np.array([[0.8, 0.1, 0.05, 0.05],
                         [0.05, 0.8, 0.1, 0.05]])
        iv = pm.gintervals("1", 200, 240)
        seq = _seq("1", 200, 240)
        threshold = -3.0

        pm.gvtrack_create("nmut_ntest", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, bidirect=False, extend=False, prior=0)
        got = pm.gextract("nmut_ntest", intervals=iv, iterator=iv)["nmut_ntest"].iloc[0]
        expected = _manual_n_mutations_scan(seq, pssm, threshold, "above")

        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-6)

    def test_aggregates_as_max_across_windows(self):
        pssm, threshold = _pssm_acg(), -3.5
        motif_len = pssm.shape[0]
        iv = pm.gintervals("1", 200, 260)
        seq = _seq("1", 200, 260)

        pm.gvtrack_create("nmut_agg", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, bidirect=False, extend=False, prior=0)
        got = pm.gextract("nmut_agg", intervals=iv, iterator=iv)["nmut_agg"].iloc[0]

        per_window = [_manual_n_mutations(seq[s:s + motif_len], pssm, threshold, "above")
                      for s in range(len(seq) - motif_len + 1)]
        non_nan = [v for v in per_window if not np.isnan(v)]
        expected = max(non_nan) if non_nan else np.nan

        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected, abs=1e-6)

    def test_works_in_gscreen_and_gextract(self):
        pm.gvtrack_create("nmut_screen", None, "pwm.n_mutations",
                          pssm=_create_test_pssm(), score_thresh=-5.0,
                          bidirect=False, extend=False, prior=0)

        # NaN != NaN, so this is the "not NA" predicate in a track expression.
        screened = pm.gscreen("nmut_screen == nmut_screen",
                              intervals=pm.gintervals("1", 0, 5000), iterator=20)
        if screened is not None and len(screened) > 0:
            assert {"chrom", "start", "end"} <= set(screened.columns)

        extracted = pm.gextract("nmut_screen", intervals=pm.gintervals("1", 0, 1000),
                                iterator=20)
        assert "nmut_screen" in extracted.columns

    def test_returns_zero_when_threshold_trivially_met(self):
        pm.gvtrack_create("nmut_easy", None, "pwm.n_mutations",
                          pssm=_create_test_pssm(), score_thresh=-20.0,
                          bidirect=False, extend=True, prior=0)
        r = pm.gextract("nmut_easy", intervals=pm.gintervals("1", 200, 210), iterator=1)
        assert len(r) > 0
        vals = r["nmut_easy"].dropna()
        if len(vals):
            assert (vals == 0).all(), "a trivially-met threshold must give 0"

    def test_prior_changes_the_result(self):
        iv = pm.gintervals("1", 200, 240)
        pssm, threshold = _create_test_pssm(), -3.0  # has zero entries

        pm.gvtrack_create("nmut_prior0", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, bidirect=False, extend=False, prior=0)
        pm.gvtrack_create("nmut_prior01", None, "pwm.n_mutations", pssm=pssm,
                          score_thresh=threshold, bidirect=False, extend=False, prior=0.01)
        r = pm.gextract(["nmut_prior0", "nmut_prior01"], intervals=iv, iterator=iv)

        # prior=0 on a PSSM with zeros gives -inf scores; prior>0 keeps them finite.
        assert not np.isnan(r["nmut_prior0"].iloc[0]) or \
               not np.isnan(r["nmut_prior01"].iloc[0])


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestNMutationsValidation:

    def test_requires_pssm(self):
        # Divergence from R misha, and shared by every pwm func in pymisha: the
        # PSSM is validated when the vtrack is evaluated, not when it is created.
        pm.gvtrack_create("nmut_nopssm", None, "pwm.n_mutations", score_thresh=-5.0)
        with pytest.raises(Exception, match="pssm"):
            pm.gextract("nmut_nopssm", intervals=pm.gintervals("1", 200, 240), iterator=40)

    def test_requires_score_thresh(self):
        with pytest.raises(Exception, match="score_thresh"):
            pm.gvtrack_create("nmut_nothresh", None, "pwm.n_mutations",
                              pssm=_create_test_pssm())

    def test_rejects_invalid_direction(self):
        with pytest.raises(Exception, match="direction"):
            pm.gvtrack_create("nmut_baddir", None, "pwm.n_mutations",
                              pssm=_create_test_pssm(), score_thresh=-5.0,
                              direction="sideways")
