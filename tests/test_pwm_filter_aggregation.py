"""gvtrack_filter aggregation for the pwm family, ported from R misha's
tests/testthat/test-pwm-filter-aggregation.R.

A filter on a pwm vtrack scores the filter's unmasked parts separately and
reduces them. Every reference below is built from the same func and params with
NO filter, evaluated over each unmasked part on its own, so the reduction is
what is under test and not the per-part scoring.

Fixture note: R's version takes its assembly gap from chr20's leading N run;
pymisha's test DB has chrX's N run [34821, 84821) instead.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import pymisha as pm

_TH = -12.0
_LOG_SUM_EXP_TOL = 1e-5


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


@pytest.fixture(autouse=True)
def _clean_vtracks():
    _remove_all_vtracks()
    yield
    _remove_all_vtracks()


def _filter_agg_pssm():
    return np.array([
        [0.90, 0.04, 0.03, 0.03],
        [0.03, 0.85, 0.06, 0.06],
        [0.05, 0.05, 0.80, 0.10],
        [0.08, 0.08, 0.09, 0.75],
        [0.70, 0.10, 0.10, 0.10],
        [0.12, 0.65, 0.12, 0.11],
    ])


def _one(vtrack, iv):
    return float(pm.gextract(vtrack, iv, iterator=iv)[vtrack].iloc[0])


def _log_sum_exp(vals):
    arr = np.asarray(list(vals), dtype=float)
    m = float(arr.max())
    if math.isinf(m) and m < 0:
        return m
    return float(m + np.log(np.exp(arr - m).sum()))


def test_pwm_vtracks_aggregate_across_a_filters_unmasked_parts():
    pssm = _filter_agg_pssm()
    iv = pm.gintervals("1", 4000, 4300)
    mask = pm.gintervals("1", 4100, 4200)  # punches the middle out, leaving two parts
    part_a = pm.gintervals("1", 4000, 4100)
    part_b = pm.gintervals("1", 4200, 4300)

    for fn in ("pwm", "pwm.max", "pwm.count"):
        _remove_all_vtracks()
        params = {"pssm": pssm, "bidirect": False, "strand": 1, "extend": True,
                  "prior": 0.01, "score_thresh": _TH}
        pm.gvtrack_create("t", None, fn, **params)
        pm.gvtrack_filter("t", filter=mask)
        pm.gvtrack_create("t_part", None, fn, **params)

        got = _one("t", iv)
        parts = [_one("t_part", part_a), _one("t_part", part_b)]

        expected = {
            # log-sum-exp, not a sum: summing log-likelihoods multiplies
            # probabilities, which is not "the score over this interval"
            "pwm": lambda: _log_sum_exp(parts),
            "pwm.max": lambda: max(parts),
            "pwm.count": lambda: sum(parts),
        }[fn]()
        # the two parts must differ, or a max that picked the wrong one would
        # still agree, and a sum would be within tolerance of a log-sum-exp
        assert abs(parts[0] - parts[1]) > 1e-3
        assert got == pytest.approx(expected, abs=_LOG_SUM_EXP_TOL), fn


@pytest.mark.parametrize("spatial", [False, True])
def test_pwm_max_pos_aggregates_across_a_filters_parts_by_score_not_index(spatial):
    # The spatial scorer answers pwm.max.pos from its own code path, separately
    # from the non-spatial one, so both are exercised - with unit weights, so
    # the spatial factors leave the scores alone and only the code path differs.
    pssm = _filter_agg_pssm()
    base = 3000 if spatial else 4000
    iv = pm.gintervals("1", base, base + 300)
    mask = pm.gintervals("1", base + 100, base + 200)
    parts = [
        pm.gintervals("1", base, base + 100),
        pm.gintervals("1", base + 200, base + 300),
    ]
    params = {"pssm": pssm, "bidirect": False, "strand": 1, "extend": True,
              "prior": 0.01}
    if spatial:
        params.update(spat_factor=[1.0] * 5, spat_bin=20)

    pm.gvtrack_create("t_pos", None, "pwm.max.pos", **params)
    pm.gvtrack_filter("t_pos", filter=mask)
    pm.gvtrack_create("p_pos", None, "pwm.max.pos", **params)
    pm.gvtrack_create("p_max", None, "pwm.max", **params)

    got = _one("t_pos", iv)
    part_max = [_one("p_max", p) for p in parts]
    part_pos = [_one("p_pos", p) for p in parts]

    # unambiguous winner, so the expected part is not a coin toss
    assert abs(part_max[0] - part_max[1]) > 1e-3
    win = int(np.argmax(part_max))
    expected = part_pos[win] + (int(parts[win]["start"].iloc[0]) - base)

    assert got == pytest.approx(expected), "position, offset into the iterator interval"

    # An aggregation that compared the parts by their part-RELATIVE position
    # and reported the larger one with no offset would name an anchor in
    # whichever part carried the bigger local index, in that part's coordinate
    # frame. The invariants below pin the output's meaning.
    assert got > 0
    assert got <= 300
    anchor_start = base + got - 1
    assert anchor_start < base + 100 or anchor_start >= base + 200, (
        f"anchor at {anchor_start}"
    )


@pytest.mark.parametrize("strand", ["bidirect", "minus"])
def test_spatial_pwm_max_pos_ranks_the_parts_by_the_score_its_position_won_with(strand):
    # R ranks the parts by PWMScorer::get_last_max_score(): the best per-anchor
    # score INCLUDING its spatial weight, bins counted from the part's 5' start
    # on either strand (R misha 5.15 counts them so on the minus strand too).
    # pymisha used to rebuild that from two single-strand pwm.max scans, and
    # the reverse one read the reverse complement, whose bins count from the
    # 3' end - so with weights that are not symmetric over a part it named a
    # part R does not.
    pssm = _filter_agg_pssm()
    w = pssm.shape[0]
    spat, sbin = [0.5, 1.0, 2.0, 1.0, 0.5], 100
    params = {"pssm": pssm, "prior": 0.01, "extend": False,
              "spat_factor": spat, "spat_bin": sbin}
    params.update({"bidirect": True} if strand == "bidirect" else {"bidirect": False, "strand": -1})
    logp = np.log(pssm + 0.01) - np.log((pssm + 0.01).sum(axis=1, keepdims=True))
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    lspat = np.log(np.asarray(spat, dtype=np.float32))

    def part_best(start, end):
        """The best anchor score with its spatial weight, as the scorer ranks."""
        seq = pm.gseq_extract(pm.gintervals("1", start, end))[0].upper()
        if strand == "minus":   # scanned on the reverse complement
            seq = seq[::-1].translate(str.maketrans("ACGT", "TGCA"))
        best = -np.inf
        for i in range(len(seq) - w + 1):
            win = [code[b] for b in seq[i:i + w]]
            fwd_pos = (len(seq) - w - i) if strand == "minus" else i  # the anchor's offset from the 5' start
            sw = lspat[min(fwd_pos // sbin, len(spat) - 1)]
            rc = sum(logp[w - 1 - j, 3 - win[j]] for j in range(w))
            fwd = sum(logp[j, win[j]] for j in range(w))
            best = max(best, (max(fwd, rc) if strand == "bidirect" else fwd) + sw)
        return best

    starts = np.arange(10000, 40000, 1000)
    pm.gvtrack_create("t_pos", None, "pwm.max.pos", **params)
    pm.gvtrack_filter("t_pos", filter=pm.gintervals(["1"] * len(starts), starts + 200, starts + 250))
    pm.gvtrack_create("p_pos", None, "pwm.max.pos", **params)
    ivs = pm.gintervals(["1"] * len(starts), starts, starts + 500)
    got = pm.gextract("t_pos", ivs, iterator=ivs).sort_values("start")["t_pos"].to_numpy()

    checked = 0
    for s, g in zip(starts, got, strict=True):
        parts = [(int(s), int(s) + 200), (int(s) + 250, int(s) + 500)]
        best = [part_best(*p) for p in parts]
        if abs(best[0] - best[1]) < 1e-4:
            continue   # a near tie is decided by float rounding, not the rule
        a, b = parts[int(np.argmax(best))]
        pos = _one("p_pos", pm.gintervals("1", a, b))
        assert g == np.sign(pos) * (abs(pos) + (a - s)), int(s)
        checked += 1
    assert checked > 20


def test_a_single_unmasked_pwm_part_is_scored_like_the_unfiltered_survivor():
    # A mask flush with one edge of the iterator interval leaves exactly one
    # unmasked part rather than two. pwm, pwm.max and pwm.count don't care: a
    # lone part is just scored directly. pwm.max.pos does: a mask that clips
    # the START moves the surviving part's own start away from the interval's
    # start, so the position the part reports relative to itself has to be
    # offset before it means anything relative to the original interval. A mask
    # clipping the END leaves offset zero, so that side alone could never catch
    # a missing offset.
    pssm = _filter_agg_pssm()
    iv = pm.gintervals("1", 4000, 4300)

    # The start clip is deliberately WIDER than the part it leaves (200 bp of
    # mask, 100 bp of part). A part-relative position is then always in 1..100,
    # so reporting one without its offset necessarily decodes to a coordinate
    # inside the mask - which makes the "outside the mask" assertion below a
    # second, independent detector of a missing offset.
    clips = {
        "start": (pm.gintervals("1", 4000, 4200), pm.gintervals("1", 4200, 4300)),
        "end": (pm.gintervals("1", 4200, 4300), pm.gintervals("1", 4000, 4200)),
    }

    for nm, (mask, part) in clips.items():
        for fn in ("pwm", "pwm.max", "pwm.count"):
            _remove_all_vtracks()
            params = {"pssm": pssm, "bidirect": False, "strand": 1,
                      "extend": True, "prior": 0.01, "score_thresh": _TH}
            pm.gvtrack_create("t", None, fn, **params)
            pm.gvtrack_filter("t", filter=mask)
            pm.gvtrack_create("t_part", None, fn, **params)

            assert _one("t", iv) == pytest.approx(
                _one("t_part", part), abs=_LOG_SUM_EXP_TOL
            ), f"{nm} {fn}"

        _remove_all_vtracks()
        params = {"pssm": pssm, "bidirect": False, "strand": 1, "extend": True,
              "prior": 0.01}
        pm.gvtrack_create("t_pos", None, "pwm.max.pos", **params)
        pm.gvtrack_filter("t_pos", filter=mask)
        pm.gvtrack_create("p_pos", None, "pwm.max.pos", **params)

        got = _one("t_pos", iv)
        local_pos = _one("p_pos", part)
        expected = local_pos + (int(part["start"].iloc[0]) - 4000)
        assert got == pytest.approx(expected), f"{nm} pwm.max.pos value"

        anchor_start = 4000 + got - 1
        mask_start = int(mask["start"].iloc[0])
        mask_end = int(mask["end"].iloc[0])
        assert anchor_start < mask_start or anchor_start >= mask_end, (
            f"{nm} anchor at {anchor_start} mask {mask_start} {mask_end}"
        )


def test_a_fully_masked_pwm_interval_reports_nan():
    pssm = _filter_agg_pssm()
    iv = pm.gintervals("1", 4000, 4300)

    for fn in ("pwm", "pwm.max", "pwm.max.pos", "pwm.count"):
        _remove_all_vtracks()
        pm.gvtrack_create("t", None, fn, pssm=pssm, bidirect=False, strand=1,
                          extend=True, prior=0.01, score_thresh=_TH)
        pm.gvtrack_filter("t", filter=iv)
        assert math.isnan(_one("t", iv)), fn


def test_a_pwm_part_the_scorer_cannot_score_is_left_out_of_the_aggregation():
    # Not an assembly gap, although that is the case the potts side cares
    # about: pwm charges an N the mean of its PSSM column, so an all-N part
    # scores a real number and reduces correctly on its own. The part a pwm
    # scorer genuinely cannot score is one with no anchor to place at all:
    # narrower than the PSSM with extend=False, where the scorer returns NaN.
    #
    # The mask leaves that part FIRST, which is the ordering that exposes it: a
    # reduction seeded from the first part could never recover once that seed
    # was NaN.
    pssm = _filter_agg_pssm()  # 6 rows
    iv = pm.gintervals("1", 4000, 4300)
    mask = pm.gintervals("1", 4003, 4200)  # leaves a 3 bp first part, < 6
    part_a = pm.gintervals("1", 4000, 4003)
    part_b = pm.gintervals("1", 4200, 4300)

    params = {"pssm": pssm, "bidirect": False, "strand": 1, "extend": False,
              "prior": 0.01, "score_thresh": _TH}

    for fn in ("pwm", "pwm.max", "pwm.count", "pwm.max.pos"):
        _remove_all_vtracks()
        pm.gvtrack_create("t", None, fn, **params)
        pm.gvtrack_filter("t", filter=mask)
        pm.gvtrack_create("t_part", None, fn, **params)

        # the geometry has to actually produce one unscorable part and one
        # scorable one, or this test asserts nothing
        assert math.isnan(_one("t_part", part_a)), fn
        b = _one("t_part", part_b)
        assert not math.isnan(b), fn

        got = _one("t", iv)
        expected = b + (4200 - 4000) if fn == "pwm.max.pos" else b
        assert got == pytest.approx(expected, abs=_LOG_SUM_EXP_TOL), fn


def test_a_filtered_pwm_count_separates_counted_nothing_from_nothing_to_count():
    # The same two answers the potts side has to keep apart, and the same rule:
    #
    #   every part narrower than the PSSM, so no anchor can be placed in any of
    #     them - nothing to count, NaN. A 0 would be indistinguishable in the
    #     returned column from a bin that really was scanned and had no hits,
    #     which folds "excluded" into "no evidence" under any sum(), mean() or
    #     threshold downstream and hides the excluded bins from isna().
    #   parts that hold anchors none of which is a match - an assembly gap -
    #     really did count, and found none, so 0.
    #
    # The second block is what stops this from being satisfied by "NaN
    # everywhere".
    pssm = _filter_agg_pssm()  # 6 rows
    iv = pm.gintervals("1", 4000, 4300)
    narrow_mask = pm.gintervals("1", 4003, 4297)  # leaves 3 bp at each end

    gap = pm.gintervals("X", 40000, 40300)
    assert re.fullmatch(
        "N+", pm.gseq_extract(pm.gintervals("X", 40000, 40300 + 6 - 1))[0].upper()
    )
    gap_mask = pm.gintervals("X", 40100, 40200)

    for fn in ("pwm", "pwm.max", "pwm.max.pos", "pwm.count"):
        _remove_all_vtracks()
        pm.gvtrack_create("t", None, fn, pssm=pssm, bidirect=False, strand=1,
                          extend=False, prior=0.01, score_thresh=_TH)
        pm.gvtrack_filter("t", filter=narrow_mask)
        assert math.isnan(_one("t", iv)), f"no anchor in any part {fn}"

        # An N part holds anchors, none of them scorable (a window with an N is
        # not scored, R misha 5.15), so the count is a real 0; pwm and pwm.max
        # are -inf, and pwm.max.pos has no window to name.
        _remove_all_vtracks()
        pm.gvtrack_create("t", None, fn, pssm=pssm, bidirect=False, strand=1,
                          extend=True, prior=0.01, score_thresh=_TH)
        pm.gvtrack_filter("t", filter=gap_mask)
        got = _one("t", gap)
        if fn == "pwm.count":
            assert got == 0, f"assembly gap {fn}"
        elif fn == "pwm.max.pos":
            assert math.isnan(got), f"assembly gap {fn}"
        else:
            assert got == -math.inf, f"assembly gap {fn}"
