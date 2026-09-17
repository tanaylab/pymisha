"""Tests for the potts virtual tracks, ported from R misha's
tests/testthat/test-potts-vtrack.R.

Fixture differences from the R suite, which uses a genome with chr20 and chr21:
pymisha's test DB has chroms 1 (500 kb), 2 (300 kb) and X (200 kb). chr1 carries
the same N block starting at 167280 the R suite uses; chrX's N run [34821,
84821) supplies the "sequence resumes after a gap" boundary that R takes from
chr20's leading gap, and an all-N interval; and chr2 ends in clean sequence, so
it plays the role R gives chr21's tail.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from _potts_oracle import (
    POTTS_BASES,
    log_sum_exp,
    potts_ref_anchors,
    potts_ref_model,
    potts_ref_rc,
    potts_ref_window,
)

import pymisha as pm
from pymisha._shared import _potts_params

# Fixture landmarks, asserted rather than assumed by the tests that use them.
_CHR1_N_START = 167280
_CHRX_N_END = 84821
_CHR2_END = 300000


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


@pytest.fixture(autouse=True)
def _clean_vtracks():
    _remove_all_vtracks()
    yield
    _remove_all_vtracks()


def _seq(chrom, start, end) -> str:
    # int() throughout: gintervals rejects a numpy scalar, and these bounds
    # routinely arrive as one out of a DataFrame column.
    return pm.gseq_extract(pm.gintervals(str(chrom), int(start), int(end)))[0].upper()


def _one(vtrack, iv):
    """The single value of ``vtrack`` over ``iv`` scored as one bin."""
    return float(pm.gextract(vtrack, iv, iterator=iv)[vtrack].iloc[0])


def _vt(name, func, model, **kwargs):
    pm.gvtrack_create(name, None, func, params=dict(model, **kwargs))


def _n_run_end(chrom, limit=300000):
    """The 0-based position just past the first N run of ``chrom``, or None."""
    all_iv = pm.gintervals_all()
    size = int(all_iv.loc[all_iv["chrom"].astype(str) == str(chrom), "end"].iloc[0])
    s = _seq(chrom, 0, min(limit, size))
    m = re.search("N+", s)
    return None if m is None else m.end()


# ---------------------------------------------------------------------------
# The parameter handler
# ---------------------------------------------------------------------------


def test_the_potts_params_handler_validates_and_normalises():
    m = potts_ref_model(5, npair_mode="sparse", seed=89)

    p = _potts_params(m, what="potts")
    assert p["e"].shape == (5, 4)
    assert p["J"].shape == (len(m["pairs"]), 16)
    assert p["pairs"].shape == np.asarray(m["pairs"]).shape
    assert np.issubdtype(p["pairs"].dtype, np.integer)
    assert p["intercept"] == m["intercept"]
    assert p["bidirect"] is True
    assert p["extend"] is True

    # the flat J is column-major per block, so J[k][a, b] lands at b * 4 + a
    assert p["J"][0][(3 - 1) * 4 + (2 - 1)] == m["J"][0][2 - 1][3 - 1]


def test_the_potts_params_handler_rejects_what_it_should():
    m = potts_ref_model(5, npair_mode="full", seed=101)

    with pytest.raises(ValueError, match="'e'"):
        pm.gvtrack_create("t", None, "potts", params={"J": m["J"]})

    with pytest.raises(ValueError, match="width"):
        pm.gvtrack_create("t", None, "potts", params=dict(m, width=7))

    bad_pairs = np.array(m["pairs"], copy=True)
    bad_pairs[0] = [3, 2]
    with pytest.raises(ValueError, match="lower position first"):
        pm.gvtrack_create("t", None, "potts", params=dict(m, pairs=bad_pairs))

    dup = np.array(m["pairs"], copy=True)
    dup[1] = dup[0]
    with pytest.raises(ValueError, match="twice"):
        pm.gvtrack_create("t", None, "potts", params=dict(m, pairs=dup))

    with pytest.raises(ValueError, match="'J' holds"):
        pm.gvtrack_create("t", None, "potts", params=dict(m, J=m["J"][1:]))

    # spatial parameters are not part of the potts family. Passed as a KEYWORD
    # argument they are rejected; inside `params` they are ignored, so a fitted
    # model with fields this family does not read goes through verbatim.
    with pytest.raises(ValueError, match="spat_factor"):
        pm.gvtrack_create("t", None, "potts", spat_factor=[1.0, 2.0], **m)

    # and potts.count has no default threshold
    with pytest.raises(ValueError, match="score_thresh"):
        pm.gvtrack_create("t", None, "potts.count", params=m)


def test_a_whole_fitted_model_goes_into_params_verbatim_but_a_typo_does_not():
    # A fitted model carries whatever fields its fit produced, far more than
    # the eight this family reads, so unknown names inside `params` are ignored
    # rather than rejected. A keyword argument is typed by hand, where an
    # unknown name is a typo worth catching - a misspelled `bidirect` would
    # silently score one strand.
    m = potts_ref_model(4, npair_mode="sparse", seed=191)
    fitted = dict(m, width=4, pair_strength=np.ones(len(m["pairs"])),
                  attr="whatever", link="logit", loglik=-123.4, converged=True)

    iv = pm.gintervals(1, 2000, 2100)
    pm.gvtrack_create("pv_fitted", None, "potts.max", params=fitted)
    pm.gvtrack_create("pv_plain", None, "potts.max", params=m)
    got = pm.gextract(["pv_fitted", "pv_plain"], iv, iterator=iv)
    assert got["pv_fitted"].iloc[0] == got["pv_plain"].iloc[0]

    with pytest.raises(ValueError, match="bidrect"):
        pm.gvtrack_create("pv_typo", None, "potts.max", bidrect=False, **m)


def test_a_model_whose_window_score_cannot_fit_a_float_is_rejected():
    # Every entry is finite, so the per-entry checks pass; what overflows is
    # the WINDOW. potts scores are reported in float, and the two scoring paths
    # then answer differently: a freshly seeded interval reads Inf where a slid
    # one reads NaN, because both running aggregators report a non-finite
    # maximum as -inf and slid_answer() maps that to NaN - which is exactly the
    # "nothing was scorable" answer it is supposed to mean.
    m = potts_ref_model(6, npair_mode="none", seed=167)

    over = dict(m, e=np.full_like(m["e"], 1e38))  # 6 * 1e38, past FLT_MAX
    with pytest.raises(ValueError, match="single-precision"):
        pm.gvtrack_create("t", None, "potts", params=over)
    # both entry points go through the one validator
    with pytest.raises(ValueError, match="single-precision"):
        pm.gseq_potts("ACGTACGTAC", over, mode="max")

    # J and the intercept count towards the same bound
    big = potts_ref_model(4, npair_mode="full", seed=167)
    big["J"] = [np.full((4, 4), 1e38) for _ in big["J"]]
    with pytest.raises(ValueError, match="single-precision"):
        pm.gvtrack_create("t", None, "potts", params=big)

    with pytest.raises(ValueError, match="single-precision"):
        pm.gvtrack_create("t", None, "potts", params=dict(m, intercept=1e39))

    # A large model that does fit is left alone, so the bound is not merely
    # "no big numbers": 6 * 1e37 is inside FLT_MAX.
    pm.gvtrack_create("t", None, "potts", params=dict(m, e=np.full_like(m["e"], 1e37)))


# ---------------------------------------------------------------------------
# The four funcs over a genome interval
# ---------------------------------------------------------------------------


def test_potts_and_potts_max_vtracks_equal_gseq_potts_on_the_same_sequence():
    m = potts_ref_model(6, npair_mode="full", seed=103)

    iv = pm.gintervals(1, 200, 300)
    # extend=True reads motif_len - 1 bases past the end, so the scored
    # sequence is longer than the interval. That is the sequence to compare on.
    seq_ext = _seq(1, 200, 300 + 6 - 1)

    _vt("p_lse", "potts", m, bidirect=True, extend=True)
    _vt("p_max", "potts.max", m, bidirect=True, extend=True)

    got = pm.gextract(["p_lse", "p_max"], iv, iterator=iv)
    assert got["p_lse"].iloc[0] == pytest.approx(
        pm.gseq_potts(seq_ext, m, mode="lse", bidirect=True)[0], abs=1e-5
    )
    assert got["p_max"].iloc[0] == pytest.approx(
        pm.gseq_potts(seq_ext, m, mode="max", bidirect=True)[0], abs=1e-5
    )


def test_an_order_1_potts_vtrack_reproduces_the_pwm_family():
    # A strictly positive PSSM: prior=0 on a zero entry gives -inf, and an
    # -inf comparison proves nothing.
    rng = np.random.default_rng(107)
    pssm = rng.uniform(0.05, 1.0, size=(8, 4))
    pssm = pssm / pssm.sum(axis=1, keepdims=True)

    # An order-1 Potts whose e is log(pssm) is exactly a PWM with prior = 0.
    m = {
        "e": np.log(pssm),
        "J": [],
        "pairs": np.zeros((0, 2), dtype=np.int64),
        "intercept": 0.0,
    }

    iv = pm.gintervals(1, 1000, 1400)
    for bi in (True, False):
        _remove_all_vtracks()
        pm.gvtrack_create("w_lse", None, "pwm", pssm=pssm, prior=0,
                          bidirect=bi, extend=True, strand=1)
        pm.gvtrack_create("w_max", None, "pwm.max", pssm=pssm, prior=0,
                          bidirect=bi, extend=True, strand=1)
        _vt("t_lse", "potts", m, bidirect=bi, extend=True, strand=1)
        _vt("t_max", "potts.max", m, bidirect=bi, extend=True, strand=1)

        got = pm.gextract(["w_lse", "w_max", "t_lse", "t_max"], iv, iterator=10)
        assert np.allclose(got["t_lse"], got["w_lse"], atol=1e-4), f"bidirect {bi}"
        assert np.allclose(got["t_max"], got["w_max"], atol=1e-4), f"bidirect {bi}"


def test_potts_vtracks_honour_extend():
    m = potts_ref_model(20, npair_mode="full", seed=109)

    _vt("t_no", "potts.max", m, extend=False)
    _vt("t_yes", "potts.max", m, extend=True)

    short_iv = pm.gintervals(1, 500, 510)  # 10 bp, narrower than W = 20
    got = pm.gextract(["t_no", "t_yes"], short_iv, iterator=short_iv)
    assert math.isnan(got["t_no"].iloc[0])
    assert not math.isnan(got["t_yes"].iloc[0])


def test_potts_max_pos_and_potts_count_work_on_a_genome_interval():
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=113)

    iv = pm.gintervals(1, 2000, 2200)
    seq_ext = _seq(1, 2000, 2200 + w - 1)
    a_max = potts_ref_anchors(seq_ext, m, bidirect=True, union="max")
    _vt("t_pos", "potts.max.pos", m, extend=True)

    got = _one("t_pos", iv)

    # Assert the SCORE at the reported anchor, not the argmax index: under
    # bidirect a window and its reverse complement give the same max-union, so
    # two anchors can tie exactly and either is a correct answer.
    p = int(abs(got))
    assert 1 <= p <= len(a_max)
    assert a_max[p - 1] == pytest.approx(a_max.max(), abs=1e-5)

    # The sign names the winning strand, but only where the two strands
    # differ: a palindromic window ties and then either sign is right.
    win = seq_ext[p - 1:p - 1 + w]
    f = potts_ref_window(win, m)
    r = potts_ref_window(win, potts_ref_rc(m))
    if abs(f - r) > 1e-9:
        assert np.sign(got) == (-1 if r > f else 1)


def _gapped_threshold(values, lo=0.8, hi=1.0):
    """A threshold in the widest gap of a slice of the sorted scores.

    Under bidirect a window and its reverse complement produce the same two
    per-strand numbers swapped, so their unions are mathematically identical -
    for the log-sum-exp union as much as for the maximum. The anchor scores
    therefore contain exact ties, C++ and Python round each one independently,
    and an anchor sitting ON the threshold lands either side of it. Threshold
    in a wide gap instead, and return the gap so the caller can assert it is
    wider than any rounding.
    """
    srt = np.sort(np.asarray(values, dtype=float))
    n = len(srt)
    idx = np.arange(int(lo * n), min(int(hi * n), n - 1))
    j = int(idx[np.argmax(srt[idx + 1] - srt[idx])])
    return (srt[j] + srt[j + 1]) / 2, float(srt[j + 1] - srt[j]), j, n


def test_potts_count_counts_the_anchors_at_or_above_its_threshold():
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=113)

    iv = pm.gintervals(1, 2000, 2200)
    seq_ext = _seq(1, 2000, 2200 + w - 1)
    a_lse = potts_ref_anchors(seq_ext, m, bidirect=True, union="lse")
    th, gap, j, n = _gapped_threshold(a_lse)
    assert gap > 1e-3

    # Below the minimum every scorable anchor counts, which pins the size of
    # the anchor set - the interval's length, not the extended fetch's.
    _vt("c_all", "potts.count", m, extend=True, score_thresh=float(a_lse.min() - 1))
    _vt("c_tail", "potts.count", m, extend=True, score_thresh=float(th))
    _vt("c_none", "potts.count", m, extend=True, score_thresh=float(a_lse.max() + 1))

    got = pm.gextract(["c_all", "c_tail", "c_none"], iv, iterator=iv)
    assert got["c_all"].iloc[0] == n
    assert got["c_tail"].iloc[0] == n - 1 - j
    assert got["c_none"].iloc[0] == 0


def test_a_potts_vtrack_over_an_all_n_interval_is_nan():
    m = potts_ref_model(6, npair_mode="full", seed=127)

    iv = pm.gintervals("X", 40000, 40200)
    assert re.fullmatch("N+", _seq("X", 40000, 40200 + 6 - 1))

    _vt("t_lse", "potts", m, extend=False)
    _vt("t_cnt", "potts.count", m, extend=False, score_thresh=0)
    got = pm.gextract(["t_lse", "t_cnt"], iv, iterator=iv)
    assert math.isnan(got["t_lse"].iloc[0])
    assert got["t_cnt"].iloc[0] == 0


def test_potts_max_pos_reports_the_argmax_anchor_when_one_strand_is_scored():
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=113)

    iv = pm.gintervals(1, 2000, 2200)
    seq_ext = _seq(1, 2000, 2200 + w - 1)
    a_fwd = potts_ref_anchors(seq_ext, m, bidirect=False, strand=1, union="max")

    # bidirect=False is what makes the argmax INDEX a testable quantity: with
    # only the forward strand scored there is no window/reverse-complement tie,
    # so the maximiser is unique and the position arithmetic is pinned exactly.
    assert int((a_fwd == a_fwd.max()).sum()) == 1

    _vt("t_pos", "potts.max.pos", m, bidirect=False, strand=1, extend=True)
    _vt("t_max", "potts.max", m, bidirect=False, strand=1, extend=True)

    got = pm.gextract(["t_pos", "t_max"], iv, iterator=iv)
    # unsigned: the sign is only applied when bidirect
    assert got["t_pos"].iloc[0] == float(int(np.argmax(a_fwd)) + 1)
    assert got["t_max"].iloc[0] == pytest.approx(a_fwd.max(), abs=1e-5)


def test_the_reverse_orientation_is_scored_and_reported_in_forward_coordinates():
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=113)

    # The twin of the test above, and the ONLY suite coverage of PottsScorer's
    # m_strand == -1 half: the model/rc swap that reads the original reverse
    # strand off an already reverse-complemented target, and the reverse ->
    # forward coordinate mapping that turns a target index back into a
    # forward-strand position. bidirect=True cannot reach either, because
    # _potts_params() clamps strand to 1 there, and gseq_potts cannot: it
    # scores a string, so it has no fetch, no reverse-complemented target and
    # no position mapping.
    iv = pm.gintervals(1, 2000, 2200)
    seq_ext = _seq(1, 2000, 2200 + w - 1)
    a_rev = potts_ref_anchors(seq_ext, m, bidirect=False, strand=-1)
    th, gap, j, n = _gapped_threshold(a_rev, lo=0.2, hi=0.8)
    assert gap > 1e-3

    p = {"bidirect": False, "strand": -1, "extend": True}
    _vt("r_lse", "potts", m, **p)
    _vt("r_max", "potts.max", m, **p)
    _vt("r_pos", "potts.max.pos", m, **p)
    _vt("r_cnt", "potts.count", m, score_thresh=float(th), **p)
    _vt("r_all", "potts.count", m, score_thresh=float(a_rev.min() - 1), **p)
    _vt("r_none", "potts.count", m, score_thresh=float(a_rev.max() + 1), **p)

    got = pm.gextract(
        ["r_lse", "r_max", "r_pos", "r_cnt", "r_all", "r_none"], iv, iterator=iv
    )

    assert got["r_lse"].iloc[0] == pytest.approx(log_sum_exp(a_rev), abs=1e-5)
    assert got["r_max"].iloc[0] == pytest.approx(a_rev.max(), abs=1e-5)
    assert got["r_cnt"].iloc[0] == int((a_rev >= th).sum())
    assert got["r_all"].iloc[0] == n
    assert got["r_none"].iloc[0] == 0

    # The position is reported in FORWARD-strand coordinates even though the
    # target was fetched reverse-complemented, so it indexes a_rev directly.
    # Asserted as the score at the reported anchor, not as an argmax index:
    # ties make the index itself undefined.
    pos = got["r_pos"].iloc[0]
    assert pos > 0  # unsigned: the sign is only applied when bidirect
    assert pos <= n
    assert a_rev[int(pos) - 1] == pytest.approx(a_rev.max(), abs=1e-5)


def test_a_potts_vtrack_honours_gvtrack_iterator_shifts():
    m = potts_ref_model(6, npair_mode="full", seed=113)

    _vt("s_lse", "potts", m, extend=True)
    _vt("p_lse", "potts", m, extend=True)
    pm.gvtrack_iterator("s_lse", sshift=-50, eshift=50)

    narrow = pm.gintervals(1, 2050, 2150)
    wide = pm.gintervals(1, 2000, 2200)
    shifted = pm.gextract(["s_lse", "p_lse"], narrow, iterator=narrow)
    plain = pm.gextract("p_lse", wide, iterator=wide)

    assert shifted["s_lse"].iloc[0] == pytest.approx(plain["p_lse"].iloc[0], abs=1e-6)
    assert shifted["s_lse"].iloc[0] != pytest.approx(shifted["p_lse"].iloc[0])


def test_a_bidirect_potts_vtrack_ignores_strand():
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=113)

    # Both intervals sit where the two scan orders diverge: chr1 opens with a
    # repeat, so many anchors tie to the last bit, and chr2's tail has the
    # end-only extension fully clipped by the chromosome end.
    #
    # `strand` cannot change WHICH anchors a bidirect scan covers - the range
    # is [0, tlen - W] in either orientation and maps onto the same genomic
    # windows - so potts, potts.max and potts.count agreed even before
    # _potts_params() clamped strand. potts.max.pos did not: strand=-1 fetches
    # the reverse-complemented target, walks the same anchors in the opposite
    # order, and first-wins tie-breaking then keeps a different, equally
    # maximal, one.
    ivs = {
        "chr1_start": pm.gintervals(1, 0, 60),
        "chr2_tail": pm.gintervals(2, _CHR2_END - 30, _CHR2_END),
    }
    funcs = ("potts", "potts.max", "potts.max.pos", "potts.count")

    for nm, iv in ivs.items():
        # A mid-range threshold, so potts.count cannot agree by saturating
        th = float(
            np.median(
                potts_ref_anchors(
                    _seq(iv["chrom"].iloc[0], iv["start"].iloc[0], iv["end"].iloc[0]),
                    m, bidirect=True, union="lse",
                )
            )
        )

        for ext in (True, False):
            _remove_all_vtracks()
            for f in funcs:
                key = f.replace(".", "_")
                _vt(f"p_{key}", f, m, bidirect=True, extend=ext,
                    score_thresh=th, strand=1)
                _vt(f"m_{key}", f, m, bidirect=True, extend=ext,
                    score_thresh=th, strand=-1)

            cols = [f"p_{f.replace('.', '_')}" for f in funcs] + [
                f"m_{f.replace('.', '_')}" for f in funcs
            ]
            got = pm.gextract(cols, iv, iterator=iv)
            for f in funcs:
                key = f.replace(".", "_")
                a = got[f"p_{key}"].iloc[0]
                b = got[f"m_{key}"].iloc[0]
                assert (a == b) or (math.isnan(a) and math.isnan(b)), (
                    f"{nm} extend {ext} {f}: {a} vs {b}"
                )
            # not vacuous: a real count, and a real position
            assert got["p_potts_count"].iloc[0] > 0
            assert got["p_potts_count"].iloc[0] < (
                iv["end"].iloc[0] - iv["start"].iloc[0]
            )
            assert abs(got["p_potts_max_pos"].iloc[0]) > 0


# ---------------------------------------------------------------------------
# The sliding-window cache.
#
# Every test below compares the SLID path against the RE-SEEDED path: the same
# kernel with the same tie-breaking on both sides, so they must agree, and a
# disagreement is a cache bug and never a rounding artefact. The re-seeded side
# is one gextract() per position, each of which builds a fresh scanner and
# therefore a fresh, cold PottsScorer.
#
# Note what potts.max.pos returns: a 1-based index into the FETCHED target, not
# a genomic coordinate. A window whose argmax anchor does not move therefore
# reports a position one lower at every step, which is why RunningMaxDeque keys
# on absolute genomic positions and the cache has to map back through the
# current expanded.start.
# ---------------------------------------------------------------------------


def _assert_potts_cache_agrees(fn, params, pad, iv, info, it=1, check_varies=True):
    """A contiguous scan of ``iv`` against the same bins fetched one at a time.

    ``pad`` is the gvtrack_iterator shift (None for none) and ``it`` the
    iterator width, which together set the window size and the stride.
    ``check_varies`` asks that the column actually move, so a constant answer
    cannot satisfy the comparison by accident.
    """
    _remove_all_vtracks()
    pm.gvtrack_create("t", None, fn, params=params)
    if pad is not None:
        pm.gvtrack_iterator("t", sshift=-pad, eshift=pad)

    slid = pm.gextract("t", iv, iterator=it)
    # Fixed bins are aligned to absolute multiples of the bin size, so the span
    # can open and close with a partial bin. Asserted rather than assumed: a
    # scan that silently returned nothing would satisfy every comparison below.
    start = int(iv["start"].iloc[0])
    end = int(iv["end"].iloc[0])
    assert len(slid) == (end - 1) // it - start // it + 1, info

    one_at_a_time = []
    for k in reversed(range(len(slid))):
        bin_iv = slid.iloc[[k]][["chrom", "start", "end"]].reset_index(drop=True)
        one_at_a_time.append(float(pm.gextract("t", bin_iv, iterator=bin_iv)["t"].iloc[0]))
    one_at_a_time = np.array(one_at_a_time[::-1])

    got = slid["t"].to_numpy(dtype=float)
    assert np.allclose(got, one_at_a_time, atol=1e-6, equal_nan=True), (
        f"{info}: {got} vs {one_at_a_time}"
    )
    if check_varies:
        assert len(np.unique(got[~np.isnan(got)])) > 1, info
    return got


def _potts_cache_thresh(m, iv, pad, bidirect, strand):
    """A mid-range threshold, so potts.count cannot saturate at 0 or the window."""
    w = len(m["e"])
    seq_ext = _seq(
        iv["chrom"].iloc[0],
        int(iv["start"].iloc[0]) - pad,
        int(iv["end"].iloc[0]) + pad + w - 1,
    )
    a = potts_ref_anchors(seq_ext, m, bidirect=bidirect, strand=strand, union="lse")
    return float(np.nanmedian(a))


_ORIENTS = {
    # The cache is keyed on strand_mode, and when the target is
    # reverse-complemented the anchor entering the window on a one-position
    # step arrives at the LOW target index, so the forward orientation alone
    # exercises only half of the slide arithmetic. bidirect=False is the only
    # way in: _potts_params() clamps strand to 1 whenever bidirect is True.
    "forward": {"bidirect": True, "strand": 1},
    "reverse": {"bidirect": False, "strand": -1},
}

# Four window/stride geometries:
#   pad 250, iterator 1 - the regime the cache exists for, a 501-anchor window
#     moving 1 bp. There the region's best anchor never leaves, so potts.max is
#     constant and only the other modes' comparisons say anything.
#   pad 40, iterator 1  - narrow enough that every mode's answer moves.
#   pad 40, iterator 7  - stride 7 into an 81-anchor window, so the multi-anchor
#     push loop runs with a stride well inside the window.
#   pad 40, iterator 60 - stride 60 into a 141-anchor window, so most of the
#     window is evicted and refilled on every step.
#   no shift, iterator 10 - consecutive windows share no anchor at all, so the
#     cache declines to keep one and every call answers directly. The comparison
#     is trivially satisfied there; it checks the un-populated path returns the
#     right number.
_GEOMS = [
    ("pad250/it1", 250, 1, False),
    ("pad40/it1", 40, 1, True),
    ("pad40/it7", 40, 7, True),
    ("pad40/it60", 40, 60, True),
    ("noshift/it10", None, 10, True),
]

_FUNCS = ("potts", "potts.max", "potts.max.pos", "potts.count")


@pytest.mark.parametrize("orient", list(_ORIENTS))
@pytest.mark.parametrize("geom", _GEOMS, ids=[g[0] for g in _GEOMS])
def test_the_sliding_cache_returns_what_the_seeded_path_returns(orient, geom):
    m = potts_ref_model(8, npair_mode="full", seed=131)
    iv = pm.gintervals(1, 5000, 5200)
    o = _ORIENTS[orient]
    nm, pad, it, varies = geom

    th = _potts_cache_thresh(m, iv, 0 if pad is None else pad, o["bidirect"], o["strand"])
    for fn in _FUNCS:
        _assert_potts_cache_agrees(
            fn, dict(m, **o, extend=True, score_thresh=th), pad, iv,
            info=f"{orient} {nm} {fn}", it=it, check_varies=varies,
        )


def test_the_potts_cache_re_seeds_where_the_chromosome_end_clips_the_fetch():
    m = potts_ref_model(12, npair_mode="full", seed=149)
    pad = 30

    # The end-only extension gets progressively clipped by the chromosome end,
    # so the fetched target shortens and i_max moves while the interval keeps
    # its width. The slide guard has to notice and re-seed; if it slid anyway,
    # the reverse orientation would key its anchors off the wrong target length.
    iv = pm.gintervals(2, _CHR2_END - 120, _CHR2_END)

    for o in _ORIENTS.values():
        for fn in _FUNCS:
            _assert_potts_cache_agrees(
                fn, dict(m, **o, extend=True, score_thresh=0), pad, iv,
                info=f"chr2 tail bidirect {o['bidirect']} {fn}", check_varies=False,
            )


def test_the_potts_cache_is_invalidated_on_a_chromosome_change():
    import pandas as pd

    m = potts_ref_model(8, npair_mode="full", seed=137)
    _vt("t", "potts.max", m, extend=True)

    iv = pd.concat(
        [pm.gintervals(1, 3000, 3020), pm.gintervals(2, 3000, 3020)],
        ignore_index=True,
    )

    # Interleaved across the boundary, versus one chromosome at a time. A cache
    # that survives the chromosome change returns the first chromosome's window
    # here.
    both = pm.gextract("t", iv, iterator=1)
    sep = pd.concat(
        [
            pm.gextract("t", iv.iloc[[0]].reset_index(drop=True), iterator=1),
            pm.gextract("t", iv.iloc[[1]].reset_index(drop=True), iterator=1),
        ],
        ignore_index=True,
    )
    assert np.allclose(both["t"], sep["t"], atol=1e-6, equal_nan=True)


def _potts_tied_model(w):
    """A model under which EVERY scorable anchor scores exactly the same.

    All-zero ``e`` and ``J``, zero intercept. Nothing else makes a tie-break
    testable: with a random model an exact tie is a fixture accident, so a
    comparison that only ever sees distinct scores says nothing about which of
    two equal maximisers a path picks - and picking differently is precisely
    how the cached and the uncached paths can disagree while both look right.
    """
    import itertools
    full = np.array(
        [[i + 1, j + 1] for i, j in itertools.combinations(range(w), 2)],
        dtype=np.int64,
    ).reshape(-1, 2)
    return {
        "e": np.zeros((w, 4)),
        "J": [np.zeros((4, 4)) for _ in range(len(full))],
        "pairs": full,
        "intercept": 0.0,
    }


_TIED_GEOMS = [
    ("pad40/it1", 40, 1),
    ("pad40/it7", 40, 7),
    ("noshift/it10", None, 10),
    ("noshift/it1", None, 1),
]
_TIED_ORIENTS = {
    "forward": {"bidirect": True, "strand": 1},
    "forward_uni": {"bidirect": False, "strand": 1},
    "reverse": {"bidirect": False, "strand": -1},
}


@pytest.mark.parametrize("orient", list(_TIED_ORIENTS))
@pytest.mark.parametrize("geom", _TIED_GEOMS, ids=[g[0] for g in _TIED_GEOMS])
def test_the_potts_cache_breaks_a_tie_the_same_way_whatever_the_scans_shape(orient, geom):
    # potts.max.pos is the only mode a tie-break can reach - the other three
    # reduce to a number that does not depend on WHICH maximiser is named - but
    # all four are compared, because a tied model is also the cheapest check
    # that nothing else in the cache depends on the scores being distinct.
    #
    # The geometries matter more than usual here. `noshift` is the case where
    # consecutive windows share no anchor, so the cache declines to keep one and
    # every call after the first answers from score_direct() - a SECOND code
    # path for the same question, and therefore a second chance to break a tie
    # differently from the deque. The one-interval-at-a-time reference always
    # takes the first call of a fresh scorer, so the two only agree if both
    # paths use one rule.
    m = _potts_tied_model(8)
    iv = pm.gintervals(1, 5000, 5200)
    nm, pad, it = geom

    for fn in _FUNCS:
        _assert_potts_cache_agrees(
            fn, dict(m, **_TIED_ORIENTS[orient], extend=True, score_thresh=-1),
            pad, iv, info=f"tied {orient} {nm} {fn}", it=it, check_varies=False,
        )


_STRADDLE_PADS = range(20, 81)


def _straddling_pad(bnd, it, pads=_STRADDLE_PADS):
    """The window shift that puts an N -> sequence boundary INSIDE one batch.

    Geometry: the anchors of iterator bin [s, s + it) are
    [s - pad, s + it - 1 + pad]; fixed bins are aligned to multiples of the bin
    size; an anchor is scorable only if it starts at or after ``bnd``; and a
    slide pushes the window's top ``it`` anchors in ascending genomic order. So
    with T the top anchor of the arriving bin the case needs T >= bnd (the batch
    brings in a scorable anchor), T - it < bnd (the window held nothing scorable
    before), and T - it + 1 < bnd (the batch pushes -inf before that anchor) -
    i.e. bnd <= T <= bnd + it - 2. That needs it >= 2; an iterator = 1 scan
    cannot reach it however far it runs, because its batch is one anchor and so
    cannot push -inf before a real value.
    """
    for pad in pads:
        top = np.arange(0, 2 * bnd, it) - 1 + pad
        if np.any((top >= bnd) & (top <= bnd + it - 2)):
            return int(pad)
    return None


def test_a_warm_potts_cache_slides_into_and_out_of_an_all_n_stretch():
    m = potts_ref_model(8, npair_mode="full", seed=139)

    # The one place the cached and the uncached paths can disagree SILENTLY. An
    # unscorable anchor enters the running structures as -inf, the identity for
    # both a log-sum-exp and a maximum; when the window's last finite anchor is
    # evicted the aggregator goes to -inf while the cache stays valid, and -inf
    # is not the answer - NaN is, or 0 for potts.count. -inf survives arithmetic
    # instead of poisoning it, so it is the harder failure to notice.
    #
    # Both directions, because they fail differently: sliding IN evicts the
    # window's last finite anchor, sliding OUT has to recover from a window that
    # is entirely -inf.
    bndx = _n_run_end("X")
    assert bndx == _CHRX_N_END
    spans = {
        "into_N": pm.gintervals(1, _CHR1_N_START - 40, _CHR1_N_START + 40),
        "out_of_N": pm.gintervals("X", bndx - 40, bndx + 40),
    }
    for nm, iv in spans.items():
        sq = _seq(
            iv["chrom"].iloc[0],
            int(iv["start"].iloc[0]) - 20,
            int(iv["end"].iloc[0]) + 20 + len(m["e"]) - 1,
        )
        assert "N" in sq and re.search("[ACGT]", sq), nm

        for fn in _FUNCS:
            v = _assert_potts_cache_agrees(
                fn, dict(m, extend=True, score_thresh=0), 20, iv,
                info=f"{nm} {fn}", it=1,
            )
            # The span really does cross the boundary, and the answer where no
            # anchor is scorable is NaN - 0 for the count - never -inf.
            if fn == "potts.count":
                assert (v == 0).any(), f"{nm} {fn}"
                assert (v > 0).any(), f"{nm} {fn}"
            else:
                assert np.isnan(v).any(), f"{nm} {fn}"
                assert np.isfinite(v).any(), f"{nm} {fn}"
                assert not np.isinf(v).any(), f"{nm} {fn}"


def test_a_potts_cache_batch_that_straddles_the_end_of_an_n_run_agrees_too():
    m = potts_ref_model(8, npair_mode="full", seed=151)
    it = 7

    # An iterator = 1 scan across the same boundary cannot reach this: the
    # arriving batch is one anchor, so it never pushes -inf and then a real
    # value into the SAME batch while the window holds nothing scorable. That
    # ordering is what reaches RunningLogSumExp::push(-INFINITY) with its
    # running maximum already -inf, which computes exp(-inf + inf) = NaN into
    # the accumulator - and the NaN then survives into the first finite push,
    # so the interval reports NaN although it has scorable anchors.
    bnd = _n_run_end("X")
    pad = _straddling_pad(bnd, it)
    assert pad is not None

    iv = pm.gintervals("X", bnd - 4 * it - pad, bnd + 4 * it)
    for fn in _FUNCS:
        v = _assert_potts_cache_agrees(
            fn, dict(m, extend=True, score_thresh=0), pad, iv,
            info=f"straddle {fn}", it=it, check_varies=False,
        )
        if fn == "potts.count":
            assert (v == 0).any(), fn
            assert (v > 0).any(), fn
        else:
            assert np.isnan(v).any(), fn
            assert np.isfinite(v).any(), fn


# ---------------------------------------------------------------------------
# gvtrack_filter aggregation
# ---------------------------------------------------------------------------


def test_potts_vtracks_aggregate_across_a_filters_unmasked_parts():
    m = potts_ref_model(6, npair_mode="full", seed=139)

    iv = pm.gintervals(1, 4000, 4300)
    mask = pm.gintervals(1, 4100, 4200)  # punches the middle out
    th = 0

    for fn in ("potts", "potts.max", "potts.count"):
        _remove_all_vtracks()
        _vt("t", fn, m, extend=True, score_thresh=th)
        pm.gvtrack_filter("t", filter=mask)
        # scores each unmasked part on its own, with the same func and params
        # and no filter - the reference the filtered result is aggregated from
        _vt("t_part", fn, m, extend=True, score_thresh=th)
        got = _one("t", iv)

        parts = [
            _one("t_part", pm.gintervals(1, 4000, 4100)),
            _one("t_part", pm.gintervals(1, 4200, 4300)),
        ]
        expected = {
            "potts": lambda: log_sum_exp(parts),
            "potts.max": lambda: max(parts),
            "potts.count": lambda: sum(parts),
        }[fn]()
        assert got == pytest.approx(expected, abs=1e-5), fn

    # a fully masked interval scores nothing - NaN for all four, potts.count
    # included. See the next test for why the count is not 0 here.
    for fn in _FUNCS:
        _remove_all_vtracks()
        _vt("t", fn, m, extend=True, score_thresh=th)
        pm.gvtrack_filter("t", filter=iv)
        assert math.isnan(_one("t", iv)), fn


def test_a_filtered_potts_count_separates_counted_nothing_from_nothing_to_count():
    # The count has two answers that look alike in a gextract() column and are
    # not the same claim:
    #
    #   the filter leaves parts, those parts hold anchors of the model, and
    #     none of the anchors is scorable - an assembly gap. The scan ran over a
    #     real set of candidates and none passed, so the count is 0.
    #   the filter leaves no part at all, or leaves only parts narrower than the
    #     model, so no window of it can be placed anywhere. There was no
    #     candidate to pass or fail. 0 would be indistinguishable from the row
    #     above, which folds "this bin was excluded" into "this bin had no hits"
    #     under any sum(), mean() or threshold downstream, and hides the
    #     excluded bins from isna(). NaN is the value kept for that, and it is
    #     what the other three funcs already report in the same situation.
    #
    # The third block is what stops this test from being satisfied by "NaN
    # everywhere": it pins that a genuine count of zero is still 0.
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=173)

    iv = pm.gintervals(1, 4000, 4300)
    narrow_mask = pm.gintervals(1, 4003, 4297)  # leaves 3 bp at each end, < W

    # chrX's N run supplies a stretch where both surviving parts hold anchors
    # and not one of them is scorable
    gap = pm.gintervals("X", 40000, 40300)
    assert re.fullmatch("N+", _seq("X", 40000, 40300 + w - 1))
    gap_mask = pm.gintervals("X", 40100, 40200)

    for fn in _FUNCS:
        # no part at all
        _remove_all_vtracks()
        _vt("t", fn, m, extend=True, score_thresh=0)
        pm.gvtrack_filter("t", filter=iv)
        assert math.isnan(_one("t", iv)), f"fully masked {fn}"

        # parts exist, none of them wide enough to hold a window. extend=False,
        # or the fetch would be padded to the model's width and the parts would
        # score.
        _remove_all_vtracks()
        _vt("t", fn, m, extend=False, score_thresh=0)
        pm.gvtrack_filter("t", filter=narrow_mask)
        assert math.isnan(_one("t", iv)), f"no anchor in any part {fn}"

        # parts exist and hold anchors, but the sequence is an assembly gap:
        # a real count of zero, which must stay 0
        _remove_all_vtracks()
        _vt("t", fn, m, extend=True, score_thresh=0)
        pm.gvtrack_filter("t", filter=gap_mask)
        got = _one("t", gap)
        if fn == "potts.count":
            assert got == 0, f"assembly gap {fn}"
        else:
            assert math.isnan(got), f"assembly gap {fn}"


def test_potts_max_pos_aggregates_across_a_filters_unmasked_parts_by_score():
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=139)

    iv = pm.gintervals(1, 4000, 4300)
    mask = pm.gintervals(1, 4100, 4200)  # same punched-out middle as above

    seq_ext = _seq(1, 4000, 4300 + w - 1)
    a_max = potts_ref_anchors(seq_ext, m, bidirect=True, union="max")

    # 1-based anchor indices (relative to iv start) that the filter leaves
    # unmasked - the two parts the filter's complement produces, expressed as
    # index ranges into a_max rather than as separate gextract calls, so this
    # does not depend on the aggregation's part-vs-part tie-break.
    unmasked_idx = list(range(1, 4100 - 4000 + 1)) + list(
        range(4200 - 4000 + 1, len(a_max) + 1)
    )
    expected_max = max(a_max[i - 1] for i in unmasked_idx)

    _vt("t_pos", "potts.max.pos", m, extend=True)
    pm.gvtrack_filter("t_pos", filter=mask)
    got = _one("t_pos", iv)

    # Assert the SCORE at the reported anchor, and that the anchor sits in an
    # unmasked part - never the argmax index. Under bidirect a window and its
    # reverse complement give the same max-union, so two anchors can tie
    # exactly, and either is a correct answer.
    p = int(abs(got))
    assert p in unmasked_idx, f"p = {p}"
    assert a_max[p - 1] == pytest.approx(expected_max, abs=1e-5)

    win = seq_ext[p - 1:p - 1 + w]
    f = potts_ref_window(win, m)
    r = potts_ref_window(win, potts_ref_rc(m))
    if abs(f - r) > 1e-9:
        assert np.sign(got) == (-1 if r > f else 1)


def test_a_single_unmasked_part_is_scored_like_the_unfiltered_survivor():
    # A mask flush with one edge of the iterator interval leaves exactly one
    # unmasked part rather than two - the only geometry that exercises the
    # aggregation's one-part case, which the earlier tests never reached (they
    # all mask out the MIDDLE). potts, potts.max and potts.count don't care: a
    # lone part is just scored directly. potts.max.pos does care: its answer is
    # a position, and a mask that clips the START of the interval moves the
    # surviving part's own start away from the interval's start, so the position
    # the part reports (relative to itself) has to be offset before it means
    # anything relative to the original interval. A mask that clips the END
    # leaves the surviving part flush with the interval's start (offset zero),
    # so that side alone could never catch a missing offset.
    m = potts_ref_model(6, npair_mode="full", seed=139)
    iv = pm.gintervals(1, 4000, 4300)
    th = 0

    clips = {
        "start": (pm.gintervals(1, 4000, 4100), pm.gintervals(1, 4100, 4300)),
        "end": (pm.gintervals(1, 4200, 4300), pm.gintervals(1, 4000, 4200)),
    }

    for nm, (mask, part) in clips.items():
        for fn in ("potts", "potts.max", "potts.count"):
            _remove_all_vtracks()
            _vt("t", fn, m, extend=True, score_thresh=th)
            pm.gvtrack_filter("t", filter=mask)
            _vt("t_part", fn, m, extend=True, score_thresh=th)

            assert _one("t", iv) == pytest.approx(_one("t_part", part), abs=1e-5), (
                f"{nm} {fn}"
            )

        _remove_all_vtracks()
        _vt("t_pos", "potts.max.pos", m, extend=True)
        pm.gvtrack_filter("t_pos", filter=mask)
        _vt("p_pos", "potts.max.pos", m, extend=True)

        got = _one("t_pos", iv)
        local_pos = _one("p_pos", part)
        offset = int(part["start"].iloc[0]) - int(iv["start"].iloc[0])
        if local_pos > 0:
            expected = local_pos + offset
        elif local_pos < 0:
            expected = local_pos - offset
        else:
            expected = local_pos
        assert got == pytest.approx(expected), f"{nm} potts.max.pos value"

        # The assertion that actually catches the bug this test guards against:
        # a position that survives the filter must resolve to a genomic
        # coordinate OUTSIDE the mask. A one-part fast path that returned the
        # part-relative position unadjusted would - for the "start" clip -
        # decode to a coordinate INSIDE the masked region.
        anchor_start = int(iv["start"].iloc[0]) + abs(got) - 1
        mask_start = int(mask["start"].iloc[0])
        mask_end = int(mask["end"].iloc[0])
        assert anchor_start < mask_start or anchor_start >= mask_end, (
            f"{nm} anchor at {anchor_start} mask {mask_start} {mask_end}"
        )


def test_a_potts_interval_with_no_anchor_at_all_is_nan_potts_count_included():
    # Two situations that look alike and are not:
    #
    #   anchors exist, none of them scorable (an all-N assembly gap) - the scan
    #     ran and found nothing, so potts.count is 0 and the other three NaN;
    #   no anchor exists at all - nothing to count, so all four are NaN.
    #
    # The second row is reached at every chromosome end with the DEFAULT
    # extend=True, not only through the narrow-interval-with-extend=False
    # route: extend pads the fetch by W - 1, that padding is clipped at the
    # contig boundary, and the target then comes back shorter than the model.
    w = 6
    m = potts_ref_model(w, npair_mode="full", seed=163)

    # 3 bp, so narrower than the model. At the chromosome end extend has
    # nowhere to pad and the fetch is short; 1 kb inside, the same 3 bp gets its
    # full W - 1 of padding and scores. The contrast is what shows the answer
    # comes from the clipped fetch and not merely from a narrow interval.
    edge = pm.gintervals(2, _CHR2_END - 3, _CHR2_END)
    inner = pm.gintervals(2, _CHR2_END - 1000, _CHR2_END - 997)
    assert re.fullmatch("[ACGT]+", _seq(2, _CHR2_END - 1000, _CHR2_END))

    gap = pm.gintervals("X", 40000, 40200)
    assert re.fullmatch("N+", _seq("X", 40000, 40200))

    for fn in _FUNCS:
        _remove_all_vtracks()
        _vt("t", fn, m, extend=True, score_thresh=0)

        # the 3 bp interval is scorable away from the boundary, so what makes
        # the edge unscorable is the clipped fetch
        assert not math.isnan(_one("t", inner)), f"inner {fn}"

        # no anchor at all -> NaN for all four
        assert math.isnan(_one("t", edge)), f"edge {fn}"

        # anchors exist, none scorable -> 0 for the count, NaN for the rest
        got_gap = _one("t", gap)
        if fn == "potts.count":
            assert got_gap == 0, f"gap {fn}"
        else:
            assert math.isnan(got_gap), f"gap {fn}"


def test_the_four_funcs_are_listed_as_sequence_functions():
    # A potts vtrack takes src=None; passing a track must be refused the same
    # way a pwm one is.
    m = potts_ref_model(4, npair_mode="none", seed=181)
    assert set(POTTS_BASES) == {"A", "C", "G", "T"}
    for fn in _FUNCS:
        _remove_all_vtracks()
        _vt("t", fn, m, extend=True, score_thresh=0)
        assert pm.gvtrack_info("t")["func"] == fn


def test_params_and_keyword_arguments_together_are_refused():
    # R misha lets `params` shadow the named arguments and ignores them, so a
    # `bidirect=False` passed alongside a model vanishes. pymisha refuses: that
    # is the same silent-default failure the parameter allowlist exists to
    # catch. `filter` is a pymisha addition accepted everywhere, so it is the
    # one keyword that may accompany a model.
    m = potts_ref_model(4, npair_mode="sparse", seed=193)
    with pytest.raises(ValueError, match="silently win"):
        pm.gvtrack_create("t", None, "potts.max", params=m, bidirect=False)

    mask = pm.gintervals(1, 2050, 2100)
    pm.gvtrack_create("t", None, "potts.max", params=m, filter=mask)
    assert pm.gvtrack_info("t")["filter"] is not None
