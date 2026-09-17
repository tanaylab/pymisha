"""gseq_pwm against the pwm virtual tracks, ported from R misha's
tests/testthat/test-gseq-pwm-vtrack-parity.R.

The two are entry points into the same scorer, so under ``bidirect=True`` they
must give the same answer on the same anchors: every mode but ``"pos"``
combines the forward and reverse-complement readings of a window by log-sum-exp
before reducing over windows, and ``"pos"`` maximises across them because it has
to name a strand.

pymisha reaches gseq_pwm through three implementations - the C++ PWMScorer, a
vectorized numpy path and a per-sequence Python path - and picks between them on
the arguments, so every test here runs all three. The C++ path always took the
vtrack's union; the two Python paths took the maximum for ``"max"`` and
thresholded the strands apart for ``"count"``, so they answered differently from
the C++ path on the same input until this was fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

# The vtrack scores in float, gseq_pwm in double, so the two agree to float
# precision and not further.
_TOL = 1e-5

# How to force each of gseq_pwm's three implementations. The dispatch is in
# gseq_pwm(): naming neutral_chars leaves the C++ path, and a region of
# interest additionally leaves the vectorized one.
_PATHS = {
    "cpp": {},
    "vectorized": {"neutral_chars": ["N", "n", "*"]},
    "python": {"neutral_chars": ["N", "n", "*"], "start_pos": 1},
}


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


@pytest.fixture(autouse=True)
def _clean_vtracks():
    _remove_all_vtracks()
    yield
    _remove_all_vtracks()


def _parity_pssm():
    """A graded PSSM, not a consensus one.

    Every row keeps probability on all four bases, so a window's forward and
    reverse-complement scores are both finite and different, and the two strand
    unions (log-sum-exp vs maximum) are genuinely distinguishable. A
    deterministic matrix would push one strand to the prior floor and hide the
    difference.
    """
    return np.array([
        [0.70, 0.10, 0.15, 0.05],
        [0.05, 0.60, 0.25, 0.10],
        [0.10, 0.15, 0.65, 0.10],
        [0.20, 0.10, 0.10, 0.60],
        [0.55, 0.20, 0.15, 0.10],
    ])


def _parity_intervals():
    return pm.gintervals(
        ["1", "1", "1", "1", "2"],
        [200, 1000, 5000, 100000, 50000],
        [260, 1080, 5040, 100200, 50200],
    )


def _seqs(ivs):
    return [s.upper() for s in pm.gseq_extract(ivs)]


def _ordered(df, col):
    return df.sort_values("intervalID")[col].to_numpy(dtype=float)


def _count_threshold(seqs, pssm, prior=0.01):
    """A threshold sitting in a wide gap of the pooled per-anchor union scores.

    A threshold read off a quantile would land on an exact tie - a window and
    its reverse complement produce the same pair of numbers, so ties are common
    here - and then the vtrack's float against gseq_pwm's double would decide
    the result rather than the counting rule. Returns the threshold and the gap
    it sits in, so the caller can assert the gap is wider than any rounding.
    """
    mat = np.asarray(pssm, dtype=float) + prior
    mat = mat / mat.sum(axis=1, keepdims=True)
    logp = np.log(mat)
    w = logp.shape[0]
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    comp = {0: 3, 1: 2, 2: 1, 3: 0}

    pooled = []
    for s in seqs:
        for i in range(len(s) - w + 1):
            win = [code[b] for b in s[i:i + w]]
            f = sum(logp[j, win[j]] for j in range(w))
            r = sum(logp[j, comp[win[w - 1 - j]]] for j in range(w))
            pooled.append(float(np.logaddexp(f, r)))
    srt = np.sort(np.array(pooled))
    lo, hi = int(0.2 * len(srt)), int(0.8 * len(srt))
    idx = np.arange(lo, hi)
    j = int(idx[np.argmax(srt[idx + 1] - srt[idx])])
    return (srt[j] + srt[j + 1]) / 2, float(srt[j + 1] - srt[j])


@pytest.mark.parametrize("path", list(_PATHS))
def test_gseq_pwm_lse_equals_the_pwm_vtrack_under_bidirect(path):
    pssm = _parity_pssm()
    ivs = _parity_intervals()

    pm.gvtrack_create("p_lse", None, func="pwm", pssm=pssm,
                      bidirect=True, extend=False, prior=0.01)
    v = _ordered(pm.gextract("p_lse", ivs, iterator=ivs), "p_lse")

    g = pm.gseq_pwm(_seqs(ivs), pssm, mode="lse", bidirect=True,
                    extend=False, prior=0.01, **_PATHS[path])
    np.testing.assert_allclose(g, v, atol=_TOL)


@pytest.mark.parametrize("path", list(_PATHS))
def test_gseq_pwm_max_equals_the_pwm_max_vtrack_under_bidirect(path):
    pssm = _parity_pssm()
    ivs = _parity_intervals()

    pm.gvtrack_create("p_max", None, func="pwm.max", pssm=pssm,
                      bidirect=True, extend=False, prior=0.01)
    v = _ordered(pm.gextract("p_max", ivs, iterator=ivs), "p_max")

    g = pm.gseq_pwm(_seqs(ivs), pssm, mode="max", bidirect=True,
                    extend=False, prior=0.01, **_PATHS[path])
    np.testing.assert_allclose(g, v, atol=_TOL)


@pytest.mark.parametrize("path", list(_PATHS))
def test_gseq_pwm_count_equals_the_pwm_count_vtrack_under_bidirect(path):
    pssm = _parity_pssm()
    ivs = _parity_intervals()
    seqs = _seqs(ivs)
    thresh, gap = _count_threshold(seqs, pssm)
    assert gap > 1e-3

    pm.gvtrack_create("p_count", None, func="pwm.count", pssm=pssm,
                      score_thresh=thresh, bidirect=True, extend=False, prior=0.01)
    v = _ordered(pm.gextract("p_count", ivs, iterator=ivs), "p_count")

    g = pm.gseq_pwm(seqs, pssm, mode="count", score_thresh=thresh,
                    bidirect=True, extend=False, prior=0.01, **_PATHS[path])
    np.testing.assert_allclose(g, v, atol=_TOL)

    # Not saturated at either end, so the comparison above has something to say.
    n_anchors = np.array([len(s) - pssm.shape[0] + 1 for s in seqs])
    assert np.all((g > 0) & (g < n_anchors))


@pytest.mark.parametrize("path", list(_PATHS))
def test_a_count_is_one_hit_per_anchor_not_one_per_strand(path):
    pssm = _parity_pssm()
    ivs = _parity_intervals()
    seqs = _seqs(ivs)
    n_anchors = np.array([len(s) - pssm.shape[0] + 1 for s in seqs], dtype=float)

    # A threshold below every attainable score: with prior = 0.01 no score is
    # -inf, so every anchor passes on both strands. One count per anchor is
    # therefore exactly the number of anchors; one count per strand would be
    # twice that.
    thresh = -50.0

    pm.gvtrack_create("p_count", None, func="pwm.count", pssm=pssm,
                      score_thresh=thresh, bidirect=True, extend=False, prior=0.01)
    v = _ordered(pm.gextract("p_count", ivs, iterator=ivs), "p_count")

    g = pm.gseq_pwm(seqs, pssm, mode="count", score_thresh=thresh,
                    bidirect=True, extend=False, prior=0.01, **_PATHS[path])
    np.testing.assert_array_equal(g, n_anchors)
    np.testing.assert_allclose(g, v, atol=_TOL)


@pytest.mark.parametrize("path", list(_PATHS))
def test_gseq_pwm_equals_the_pwm_vtracks_with_extend(path):
    pssm = _parity_pssm()
    w = pssm.shape[0]
    ivs = _parity_intervals()

    # extend=True extends the END of the interval by w - 1, so the equivalent
    # sequence-level call is the end-extended sequence scanned without
    # extension. That adds w - 1 = 4 anchors per interval, 20 in all.
    ext = ivs.copy()
    ext["end"] = ext["end"] + w - 1
    seqs_ext = _seqs(ext)
    thresh, gap = _count_threshold(seqs_ext, pssm)
    assert gap > 1e-3

    pm.gvtrack_create("p_max", None, func="pwm.max", pssm=pssm,
                      bidirect=True, extend=True, prior=0.01)
    # mode="max" may not see the extension on these intervals - none of the 20
    # extra anchors need beat the best one already inside - so the count line is
    # what holds the extend geometry, and the max line is here for the strand
    # union at extend=True.
    pm.gvtrack_create("p_count", None, func="pwm.count", pssm=pssm,
                      score_thresh=thresh, bidirect=True, extend=True, prior=0.01)
    got = pm.gextract(["p_max", "p_count"], ivs, iterator=ivs)

    np.testing.assert_allclose(
        pm.gseq_pwm(seqs_ext, pssm, mode="max", bidirect=True, extend=False,
                    prior=0.01, **_PATHS[path]),
        _ordered(got, "p_max"), atol=_TOL,
    )
    np.testing.assert_allclose(
        pm.gseq_pwm(seqs_ext, pssm, mode="count", score_thresh=thresh,
                    bidirect=True, extend=False, prior=0.01, **_PATHS[path]),
        _ordered(got, "p_count"), atol=_TOL,
    )

    # And the extension is load-bearing: the plain interval's sequence is four
    # anchors short per interval and gives a different count.
    plain = pm.gseq_pwm(_seqs(ivs), pssm, mode="count", score_thresh=thresh,
                        bidirect=True, extend=False, prior=0.01, **_PATHS[path])
    assert not np.allclose(plain, _ordered(got, "p_count"))


@pytest.mark.parametrize("skip_gaps", [True, False])
def test_both_scan_paths_take_the_same_strand_union_as_the_vtracks(skip_gaps):
    pssm = _parity_pssm()
    ivs = _parity_intervals()
    seqs = _seqs(ivs)
    thresh, gap = _count_threshold(seqs, pssm)
    assert gap > 1e-3

    pm.gvtrack_create("p_max", None, func="pwm.max", pssm=pssm,
                      bidirect=True, extend=False, prior=0.01)
    pm.gvtrack_create("p_count", None, func="pwm.count", pssm=pssm,
                      score_thresh=thresh, bidirect=True, extend=False, prior=0.01)
    got = pm.gextract(["p_max", "p_count"], ivs, iterator=ivs)

    np.testing.assert_allclose(
        pm.gseq_pwm(seqs, pssm, mode="max", bidirect=True, extend=False,
                    prior=0.01, skip_gaps=skip_gaps),
        _ordered(got, "p_max"), atol=_TOL,
    )
    np.testing.assert_allclose(
        pm.gseq_pwm(seqs, pssm, mode="count", score_thresh=thresh, bidirect=True,
                    extend=False, prior=0.01, skip_gaps=skip_gaps),
        _ordered(got, "p_count"), atol=_TOL,
    )

    # A gap the scan has to step over gives the ungapped answer back, which is
    # what reaches the gap-handling path with its projection actually doing
    # something.
    with_gap = [s[:20] + "---" + s[20:] for s in seqs]
    np.testing.assert_allclose(
        pm.gseq_pwm(with_gap, pssm, mode="max", bidirect=True, extend=False,
                    prior=0.01),
        _ordered(got, "p_max"), atol=_TOL,
    )
    np.testing.assert_allclose(
        pm.gseq_pwm(with_gap, pssm, mode="count", score_thresh=thresh,
                    bidirect=True, extend=False, prior=0.01),
        _ordered(got, "p_count"), atol=_TOL,
    )


def test_gseq_pwm_pos_equals_the_pwm_max_pos_vtrack_under_bidirect():
    pssm = _parity_pssm()
    ivs = _parity_intervals()

    pm.gvtrack_create("p_pos", None, func="pwm.max.pos", pssm=pssm,
                      bidirect=True, extend=False, prior=0.01)
    v = _ordered(pm.gextract("p_pos", ivs, iterator=ivs), "p_pos")

    g = pm.gseq_pwm(_seqs(ivs), pssm, mode="pos", bidirect=True, extend=False,
                    prior=0.01, return_strand=True)

    # The vtrack signs the position by strand; gseq_pwm returns the two apart.
    np.testing.assert_array_equal(g["pos"].to_numpy(dtype=float), np.abs(v))
    np.testing.assert_array_equal(g["strand"].to_numpy(dtype=float), np.sign(v))


@pytest.mark.parametrize("path", list(_PATHS))
def test_a_palindromic_window_is_one_hit_scored_on_both_strands(path):
    # A two-base PSSM whose consensus AT is its own reverse complement, with
    # prior = 0: on the window "AT" the forward and reverse scores are both
    # log(1) = 0, so
    #   - the strand union is log(exp(0) + exp(0)) = log 2, not 0;
    #   - the anchor is one hit at score_thresh = 0, not two.
    pssm = np.array([[1.0, 0, 0, 0], [0, 0, 0, 1.0]])

    assert pm.gseq_pwm("AT", pssm, mode="max", bidirect=True, prior=0,
                       **_PATHS[path])[0] == pytest.approx(np.log(2))
    assert pm.gseq_pwm("AT", pssm, mode="count", score_thresh=0, bidirect=True,
                       prior=0, **_PATHS[path])[0] == 1

    # Single-strand scoring is untouched: still the plain forward score.
    assert pm.gseq_pwm("AT", pssm, mode="max", bidirect=False, strand=1, prior=0,
                       **_PATHS[path])[0] == pytest.approx(0.0)
    assert pm.gseq_pwm("AT", pssm, mode="count", score_thresh=0, bidirect=False,
                       strand=1, prior=0, **_PATHS[path])[0] == 1
