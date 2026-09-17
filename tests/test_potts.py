"""Tests for gseq_potts, ported from R misha tests/testthat/test-potts.R.

The reference throughout is the pure-Python oracle in tests/_potts_oracle.py,
which shares no logic with src/PottsModel.h.

One test differs from R's by design. R covers PottsModel's two kernels through
C_potts_score_codes_cmp, a test-only .Call that scores an integer code matrix
with score_codes_naive() and score_codes_blocked() side by side. pymisha has no
such entry point; the sweep at the bottom of this file drives both kernels
through the public gseq_potts() instead, relying on PottsModel's own dispatch
gate to pick each one - see the comment there for which case selects which.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _potts_oracle import (
    POTTS_BASES,
    log_sum_exp,
    potts_ref_anchors,
    potts_ref_model,
    potts_ref_rc,
    potts_ref_window,
    random_seqs,
    revcomp,
)

import pymisha as pm
from pymisha._shared import _potts_params


def test_oracle_rc_transform_is_self_consistent():
    for mode in ("full", "sparse", "none"):
        m = potts_ref_model(6, npair_mode=mode, seed=3)
        rc = potts_ref_rc(m)
        rng = np.random.default_rng(5)
        for s in random_seqs(50, 6, rng):
            assert potts_ref_window(s, rc) == pytest.approx(
                potts_ref_window(revcomp(s), m), abs=1e-12
            ), f"{mode} {s}"
        # and it is an involution
        back = potts_ref_rc(rc)
        assert np.allclose(back["e"], m["e"], atol=1e-12)
        assert back["intercept"] == m["intercept"]


def test_scores_a_single_window_like_the_oracle():
    for w in (4, 8, 20):
        for mode in ("full", "sparse", "none"):
            m = potts_ref_model(w, npair_mode=mode, seed=13)
            rng = np.random.default_rng(17)
            seqs = random_seqs(30, w, rng)
            ref = np.array([potts_ref_window(s, m) for s in seqs])

            # A single window: lse and max over one anchor are both that anchor.
            for reduce in ("max", "lse"):
                got = pm.gseq_potts(seqs, m, mode=reduce, bidirect=False, strand=1)
                assert np.allclose(got, ref, atol=1e-6), f"W {w} {mode} {reduce}"


def test_reverse_strand_equals_the_oracles_rc_twin():
    m = potts_ref_model(8, npair_mode="full", seed=23)
    rc = potts_ref_rc(m)
    rng = np.random.default_rng(29)
    seqs = random_seqs(40, 8, rng)

    # scoring the reverse strand of s == scoring the twin on s
    assert np.allclose(
        pm.gseq_potts(seqs, m, mode="max", bidirect=False, strand=-1),
        pm.gseq_potts(seqs, rc, mode="max", bidirect=False, strand=1),
        atol=1e-6,
    )
    # == scoring the original model on revcomp(s)
    assert np.allclose(
        pm.gseq_potts(seqs, m, mode="max", bidirect=False, strand=-1),
        pm.gseq_potts([revcomp(s) for s in seqs], m, mode="max", bidirect=False, strand=1),
        atol=1e-6,
    )


def test_bidirect_unions_the_two_strands():
    m = potts_ref_model(8, npair_mode="full", seed=31)
    rng = np.random.default_rng(37)
    seqs = random_seqs(40, 8, rng)

    fwd = pm.gseq_potts(seqs, m, mode="max", bidirect=False, strand=1)
    rev = pm.gseq_potts(seqs, m, mode="max", bidirect=False, strand=-1)
    want = np.array([log_sum_exp([f, r]) for f, r in zip(fwd, rev, strict=True)])

    # lse and max both lse-union the strands at one anchor
    assert np.allclose(pm.gseq_potts(seqs, m, mode="max", bidirect=True), want, atol=1e-6)
    assert np.allclose(pm.gseq_potts(seqs, m, mode="lse", bidirect=True), want, atol=1e-6)


def test_reduces_over_anchors_in_a_longer_sequence():
    m = potts_ref_model(6, npair_mode="full", seed=41)
    rng = np.random.default_rng(43)
    seqs = random_seqs(10, 40, rng)

    for bi in (True, False):
        anchors = [
            potts_ref_anchors(s, m, bidirect=bi, strand=1, union="lse") for s in seqs
        ]
        ref_lse = np.array([log_sum_exp(a) for a in anchors])
        ref_max = np.array([a.max() for a in anchors])

        assert np.allclose(
            pm.gseq_potts(seqs, m, mode="lse", bidirect=bi, strand=1), ref_lse, atol=1e-5
        ), f"bidirect {bi}"
        assert np.allclose(
            pm.gseq_potts(seqs, m, mode="max", bidirect=bi, strand=1), ref_max, atol=1e-5
        ), f"bidirect {bi}"


def test_refuses_to_score_a_window_with_a_non_acgt_base():
    m = potts_ref_model(4, npair_mode="full", seed=47)

    # the only window is unscorable -> NaN
    assert math.isnan(pm.gseq_potts("ACNT", m, mode="max", bidirect=False, strand=1)[0])
    # too short for one window -> NaN
    assert math.isnan(pm.gseq_potts("ACG", m, mode="max")[0])

    # an N excludes its own anchors but not the clean ones
    s = "ACGTACGTNACGTACGT"
    ref = potts_ref_anchors(s, m, bidirect=False, strand=1)
    good = ref[~np.isnan(ref)]
    assert pm.gseq_potts(s, m, mode="max", bidirect=False, strand=1)[0] == pytest.approx(
        good.max(), abs=1e-6
    )
    assert pm.gseq_potts(s, m, mode="lse", bidirect=False, strand=1)[0] == pytest.approx(
        log_sum_exp(good), abs=1e-6
    )


def test_mode_pos_reports_a_best_scoring_window():
    m = potts_ref_model(6, npair_mode="full", seed=61)
    rng = np.random.default_rng(67)
    seqs = random_seqs(20, 30, rng)
    w = len(m["e"])

    # Assert the SCORE at the reported position, never the index. Two anchors
    # can tie exactly - under bidirect, a window and its reverse complement
    # always do - and then the index depends on tie-break order, which the
    # kernel and the oracle resolve differently for reasons of floating-point
    # summation rather than of logic.
    got_f = pm.gseq_potts(seqs, m, mode="pos", bidirect=False, strand=1)
    for k, s in enumerate(seqs):
        a = potts_ref_anchors(s, m, bidirect=False, strand=1)
        i = int(got_f[k])
        assert 1 <= i <= len(a), s
        assert a[i - 1] == pytest.approx(a.max(), abs=1e-6), s

    # bidirect: the magnitude names a maximiser of the per-anchor strand
    # MAXIMUM (not the lse), and the sign names the winning strand there.
    got_b = pm.gseq_potts(seqs, m, mode="pos", bidirect=True)
    rc = potts_ref_rc(m)
    for k, s in enumerate(seqs):
        a = potts_ref_anchors(s, m, bidirect=True, strand=1, union="max")
        i = int(abs(got_b[k]))
        assert 1 <= i <= len(a), s
        assert a[i - 1] == pytest.approx(a.max(), abs=1e-6), s
        win = s[i - 1:i - 1 + w]
        f = potts_ref_window(win, m)
        r = potts_ref_window(win, rc)
        # Only assert the strand where the two strands actually differ; a
        # palindromic window ties and then either sign is correct.
        if abs(f - r) > 1e-9:
            assert np.sign(got_b[k]) == (-1 if r > f else 1), s


def test_mode_count_needs_a_threshold_and_counts_anchors():
    m = potts_ref_model(6, npair_mode="full", seed=71)
    rng = np.random.default_rng(60427)
    s = random_seqs(1, 200, rng)[0]

    with pytest.raises(ValueError, match="score_thresh"):
        pm.gseq_potts(s, m, mode="count")

    # Thresholds are placed midway BETWEEN two adjacent anchor scores rather
    # than at a quantile of them, as R's version does. The kernel and the
    # oracle sum a window in a different order, so they differ in the last bits
    # or two, and a threshold landing on an anchor makes the count depend on
    # which side of it that last bit falls - a real off-by-one against this
    # fixture at the 10% quantile, and nothing to do with the counting rule.
    a = potts_ref_anchors(s, m, bidirect=True, strand=1, union="lse")
    ordered = np.sort(a)
    for q in (0.1, 0.5, 0.9):
        i = int(q * (len(ordered) - 1))
        th = float((ordered[i] + ordered[i + 1]) / 2)
        assert pm.gseq_potts(s, m, mode="count", score_thresh=th)[0] == int(
            (a >= th).sum()
        ), f"thresh {th}"

    # "counted nothing" and "nothing to count" are different answers: a
    # sequence with windows but none of them scorable counts 0, a sequence too
    # short to hold one window has nothing to count and is NaN, for count as
    # for every other mode
    assert pm.gseq_potts("N" * 20, m, mode="count", score_thresh=0)[0] == 0
    assert math.isnan(pm.gseq_potts("ACG", m, mode="count", score_thresh=0)[0])


def test_pos_and_max_may_disagree_on_which_anchor_wins():
    # The strand union differs between the two - lse for max, maximum for pos -
    # so they can select different anchors. Inherited from the pwm family
    # (pwm.max combines the strands, pwm.max.pos does not) and pinned here so
    # neither half gets "corrected" alone.
    m = potts_ref_model(4, npair_mode="full", seed=79)
    rng = np.random.default_rng(83)
    disagreed = False
    for s in random_seqs(200, 20, rng):
        lse_a = potts_ref_anchors(s, m, bidirect=True, union="lse")
        max_a = potts_ref_anchors(s, m, bidirect=True, union="max")

        # max mode reduces the LSE-union over anchors
        assert pm.gseq_potts(s, m, mode="max", bidirect=True)[0] == pytest.approx(
            lse_a.max(), abs=1e-6
        ), s

        # pos mode names a maximiser of the MAX-union
        p = int(abs(pm.gseq_potts(s, m, mode="pos", bidirect=True)[0]))
        assert max_a[p - 1] == pytest.approx(max_a.max(), abs=1e-6), s

        # the wart itself: where the two unions rank anchors differently, the
        # anchor pos names scores strictly below the lse maximum, so max and
        # pos genuinely refer to different places.
        if lse_a[p - 1] < lse_a.max() - 1e-6:
            disagreed = True
    assert disagreed, "no disagreement in 200 draws - the fixture no longer exercises the wart"


def test_mode_pos_works_on_the_reverse_strand_alone():
    m = potts_ref_model(6, npair_mode="full", seed=149)
    rng = np.random.default_rng(151)
    seqs = random_seqs(30, 30, rng)

    # bidirect=False, strand=-1: the reported position is an unsigned 1-based
    # anchor index into the FORWARD sequence, and the anchor it names must be a
    # maximiser of the REVERSE-strand per-anchor scores. Asserting the score
    # rather than the index, for the tie reason established earlier.
    got = pm.gseq_potts(seqs, m, mode="pos", bidirect=False, strand=-1)

    differed = False
    for k, s in enumerate(seqs):
        rev_a = potts_ref_anchors(s, m, bidirect=False, strand=-1)
        fwd_a = potts_ref_anchors(s, m, bidirect=False, strand=1)
        p = int(got[k])
        assert 1 <= p <= len(rev_a), s
        assert rev_a[p - 1] == pytest.approx(rev_a.max(), abs=1e-6), s
        # The point of the test: a kernel that scored the forward strand here
        # would name a forward maximiser instead. Record whether the two
        # strands actually disagree about which anchor wins, so the assertion
        # above is known to be discriminating rather than accidentally
        # satisfied by a forward-scoring kernel.
        if fwd_a.max() - fwd_a[p - 1] > 1e-6:
            differed = True
    assert differed, (
        "the reverse-strand argmax never differed from the forward one - this "
        "fixture cannot distinguish the two strands"
    )


# PottsModel.h builds both kernels whenever W allows and picks between them per
# model, on lookup count: blocked costs ceil(W/2) + C(ceil(W/2), 2) lookups
# whatever the pairing, naive costs 1 + W + npair. So at W = 20 that is 55
# against 21 + npair, and at W = 21 it is 66 against 22 + npair - the dense and
# sparse models below select BLOCKED, the order-1 ones select NAIVE, and W = 21
# additionally exercises the trailing size-1 block an odd width leaves.
#
# R covers this through C_potts_score_codes_cmp, which calls the two kernels
# directly. pymisha exports no such entry point, so the sweep goes through
# gseq_potts() and compares against the oracle instead - which tests the
# production dispatch as well as the kernel it lands on, at the cost of not
# being able to reach the kernel the gate did not pick.
@pytest.mark.parametrize("w", [20, 21])
@pytest.mark.parametrize("npair_mode", ["full", "sparse", "none"])
def test_kernel_agrees_with_the_oracle_over_random_windows(w, npair_mode):
    m = potts_ref_model(w, npair_mode=npair_mode, seed=3)
    rng = np.random.default_rng(4)
    seqs = random_seqs(2000, w, rng)
    ref = np.array([potts_ref_window(s, m) for s in seqs])
    got = pm.gseq_potts(seqs, m, mode="max", bidirect=False, strand=1)
    max_diff = float(np.abs(got - ref).max())
    assert max_diff < 1e-9, (
        f"W={w} {npair_mode}: max |kernel - oracle| = {max_diff:.3e} over "
        f"{len(seqs)} windows"
    )


def test_rejects_a_model_whose_window_score_cannot_fit_a_float():
    # The scorer accumulates in double and reports in float, so a model whose
    # windows can leave the single-precision range would make its direct and
    # sliding paths answer differently. Bounded in the one validator both entry
    # points share.
    m = potts_ref_model(4, npair_mode="none", seed=5)
    m["e"] = m["e"] * 1e38
    with pytest.raises(ValueError, match="single-precision"):
        pm.gseq_potts("ACGT", m, mode="max")


def test_rejects_a_malformed_model():
    good = potts_ref_model(4, npair_mode="full", seed=7)

    with pytest.raises(ValueError, match="'e'"):
        pm.gseq_potts("ACGT", {"J": [], "pairs": np.zeros((0, 2))}, mode="max")

    bad_width = dict(good, width=5)
    with pytest.raises(ValueError, match="width"):
        pm.gseq_potts("ACGT", bad_width, mode="max")

    bad_pairs = dict(good, pairs=np.array([[2, 1]]), J=[np.zeros((4, 4))])
    with pytest.raises(ValueError, match="lower position first"):
        pm.gseq_potts("ACGT", bad_pairs, mode="max")

    dup = dict(good, pairs=np.array([[1, 2], [1, 2]]), J=[np.zeros((4, 4))] * 2)
    with pytest.raises(ValueError, match="twice"):
        pm.gseq_potts("ACGT", dup, mode="max")

    mismatched = dict(good, J=good["J"][:-1])
    with pytest.raises(ValueError, match="'J' holds"):
        pm.gseq_potts("ACGT", mismatched, mode="max")

    with pytest.raises(ValueError, match="mode must be"):
        pm.gseq_potts("ACGT", good, mode="nope")

    with pytest.raises(ValueError, match="strand must be"):
        pm.gseq_potts("ACGT", good, bidirect=False, strand=2)


def test_a_fitted_model_passes_through_verbatim():
    # width, pair_strength, attr and link are the extra fields a fitted Potts
    # model carries; they are accepted and ignored so it can be handed over as
    # it stands.
    m = potts_ref_model(4, npair_mode="sparse", seed=11)
    fitted = dict(
        m,
        width=4,
        pair_strength=np.ones(len(m["pairs"])),
        attr="whatever",
        link="logit",
    )
    assert pm.gseq_potts("ACGTACGT", fitted, mode="max") == pytest.approx(
        pm.gseq_potts("ACGTACGT", m, mode="max")
    )


def test_a_flat_j_matrix_is_accepted():
    m = potts_ref_model(5, npair_mode="full", seed=13)
    flat = np.vstack([np.asarray(jk).flatten(order="F") for jk in m["J"]])
    assert pm.gseq_potts("ACGTACGTA", dict(m, J=flat), mode="max") == pytest.approx(
        pm.gseq_potts("ACGTACGTA", m, mode="max")
    )


def test_an_e_dataframe_is_accepted_in_any_column_order():
    pd = pytest.importorskip("pandas")
    m = potts_ref_model(4, npair_mode="none", seed=17)
    df = pd.DataFrame(m["e"], columns=list(POTTS_BASES))
    shuffled = df[["T", "A", "G", "C"]]
    assert pm.gseq_potts("ACGT", dict(m, e=shuffled), mode="max") == pytest.approx(
        pm.gseq_potts("ACGT", m, mode="max")
    )


def test_a_repeated_pair_contributes_both_couplings():
    # _coerce_potts_model() rejects a duplicate (p1, p2), so this cannot be
    # reached through gseq_potts() or a vtrack with a validated model. The C++
    # parser is a trust boundary all the same, and build_blocked_tables() SUMS
    # a repeat rather than letting the last one win - which is what
    # score_codes_naive()'s loop over k does naturally. Without this test the
    # sum can be replaced by an overwrite and the whole suite still passes.
    #
    # R reaches both kernels through a test-only entry point that calls each
    # directly. pymisha goes through gseq_potts, so the kernel is whichever
    # PottsModel's gate picks: at W = 6 sparse that is naive, at W = 20 full it
    # is blocked (see the sweep above for the arithmetic).
    import _pymisha

    for w, npair_mode in ((6, "sparse"), (20, "full")):
        m = potts_ref_model(w, npair_mode=npair_mode, seed=11)
        single = _potts_params(m, bidirect=False, extend=False, strand=1,
                               score_thresh=0.0, what="duplicate pair test")
        dup = dict(single)
        dup["pairs"] = np.vstack([single["pairs"], single["pairs"][:1]])
        dup["J"] = np.vstack([single["J"], single["J"][:1]])

        rng = np.random.default_rng(12)
        seqs = random_seqs(500, w, rng)
        r1 = _pymisha.pm_gseq_potts(seqs, single, "max")
        r2 = _pymisha.pm_gseq_potts(seqs, dup, "max")

        # The duplicated block is counted twice, so the difference is exactly
        # that coupling's own contribution to each window - not zero, and not
        # double the whole score.
        j1 = single["J"][0].reshape(4, 4, order="F")
        p1, p2 = int(single["pairs"][0][0]), int(single["pairs"][0][1])
        extra = np.array([
            j1[POTTS_BASES.index(s[p1 - 1]), POTTS_BASES.index(s[p2 - 1])]
            for s in seqs
        ])
        np.testing.assert_allclose(r2 - r1, extra, atol=1e-9)
        assert not np.allclose(r2, r1)


def test_a_labelled_j_block_is_reordered_by_name_like_es_columns():
    # `e` is reordered by column NAME, so reading a labelled J block
    # positionally would score a relabelled copy of the same model
    # differently, with no warning.
    pd = pytest.importorskip("pandas")
    m = potts_ref_model(4, npair_mode="sparse", seed=21)
    s = ["ACGT", "TGCA", "GGTA"]
    want = pm.gseq_potts(s, m, mode="max", bidirect=False, strand=1)

    order = [3, 2, 1, 0]  # T, G, C, A
    names = [POTTS_BASES[i] for i in order]
    relabelled = dict(
        m,
        e=np.asarray(m["e"])[:, order],
        J=[pd.DataFrame(np.asarray(jk)[np.ix_(order, order)],
                        index=names, columns=names) for jk in m["J"]],
    )
    relabelled["e"] = pd.DataFrame(relabelled["e"], columns=names)
    np.testing.assert_allclose(
        pm.gseq_potts(s, relabelled, mode="max", bidirect=False, strand=1), want
    )

    # An unlabelled block stays positional, as before.
    bare = dict(m, J=[np.asarray(jk) for jk in m["J"]])
    np.testing.assert_allclose(
        pm.gseq_potts(s, bare, mode="max", bidirect=False, strand=1), want
    )

    # Labelled with anything but the four bases is an error, not a guess.
    bad = dict(m, J=[pd.DataFrame(np.asarray(jk), index=["A", "C", "G", "X"],
                                  columns=list(POTTS_BASES)) for jk in m["J"]])
    with pytest.raises(ValueError, match="named A, C, G, T"):
        pm.gseq_potts(s, bad, mode="max", bidirect=False, strand=1)


def test_a_fractional_pairs_index_is_rejected_not_truncated():
    # The integer cast truncates, and the range check that follows would then
    # validate the truncated value: 2.9 with W = 2 is both fractional AND out
    # of range, and would otherwise be accepted as 2.
    m = potts_ref_model(4, npair_mode="sparse", seed=31)
    m = dict(m, pairs=np.array([[1, 2]]), J=m["J"][:1])

    with pytest.raises(ValueError, match="whole number"):
        pm.gseq_potts("ACGT", dict(m, pairs=np.array([[1.0, 2.9]])),
                      mode="max", bidirect=False, strand=1)

    # and the whole-number equivalent still passes
    pm.gseq_potts("ACGT", dict(m, pairs=np.array([[1.0, 2.0]])),
                  mode="max", bidirect=False, strand=1)


def test_bidirect_false_requires_an_explicit_strand():
    # The signature default of 0 exists for parity with gseq_pwm. Under
    # bidirect=False it would otherwise mean the forward strand silently, which
    # hands half an answer to anyone carrying gseq_pwm's "0 = both strands"
    # reading over.
    m = potts_ref_model(4, npair_mode="sparse", seed=41)
    with pytest.raises(ValueError, match="explicit strand"):
        pm.gseq_potts("ACGT", m, mode="max", bidirect=False)
    pm.gseq_potts("ACGT", m, mode="max", bidirect=False, strand=1)
    pm.gseq_potts("ACGT", m, mode="max", bidirect=True)  # 0 is fine here
