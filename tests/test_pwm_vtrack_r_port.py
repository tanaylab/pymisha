"""Ports of R misha's test-pwm.R vtrack blocks that pymisha had no counterpart
for: 3, 40, 483, 510, 533, 556, 574, 685, 714, 739, 764."""

import re

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from test_gseq_pwm import _log_sum_exp, _manual_pwm_scores_single_strand

AC = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # R's create_test_pssm()


def _revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


@pytest.fixture(autouse=True)
def _clear():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def test_pwm_vtrack_works():  # test-pwm.R:3
    iv = pm.gintervals("chr1", 200, 240)
    seq_ext = pm.gseq_extract(pm.gintervals("chr1", 200, 241))[0].upper()
    kw = dict(pssm=AC, bidirect=False, prior=0.01)
    pm.gvtrack_create("e", None, "pwm", extend=False, **kw)
    pm.gvtrack_create("e_max", None, "pwm.max", extend=False, **kw)
    pm.gvtrack_create("max_pos", None, "pwm.max.pos", extend=True, **kw)
    pm.gvtrack_create("e_ext", None, "pwm", extend=True, **kw)
    pm.gvtrack_create("e_ext_no_prior", None, "pwm", pssm=AC, bidirect=False, extend=True, prior=0)
    scores = pm.gextract(["e", "e_ext", "e_max", "max_pos", "e_ext_no_prior"], iv, iterator=iv)
    one_bp = pm.gextract(["e_ext", "e_ext_no_prior"], iv, iterator=1)
    man = np.array(_manual_pwm_scores_single_strand(seq_ext, AC, 0.01))
    man0 = np.array(_manual_pwm_scores_single_strand(seq_ext, AC, 0))

    np.testing.assert_allclose(one_bp["e_ext"], man, rtol=1e-6)
    assert scores["e"][0] == pytest.approx(_log_sum_exp(list(man[:-1])), rel=1e-6)
    assert scores["e_ext"][0] == pytest.approx(_log_sum_exp(list(man)), rel=1e-6)
    assert scores["e_max"][0] == pytest.approx(man.max(), rel=1e-5)
    assert scores["max_pos"][0] == np.argmax(man) + 1
    np.testing.assert_allclose(one_bp["e_ext_no_prior"], man0, rtol=1e-6)

    ac = [m.start() for m in re.finditer("AC", seq_ext)]
    v = one_bp["e_ext_no_prior"].to_numpy()
    assert (v[ac] == 0).all()
    assert (v[np.setdiff1d(np.arange(len(v)), ac)] == -np.inf).all()
    assert scores["max_pos"][0] == ac[0] + 1


def test_pwm_10bp_bins_sum_to_the_interval():  # test-pwm.R:40
    iv = pm.gintervals("chr1", 200, 240)
    pm.gvtrack_create("e", None, "pwm", pssm=AC, bidirect=False, extend=True, prior=0.01)
    s10 = pm.gextract("e", iv, iterator=10)["e"].tolist()
    s100 = pm.gextract("e", iv, iterator=100)["e"][0]
    assert _log_sum_exp(s10) == pytest.approx(s100, rel=1e-6)


def test_pwm_max_bidirect_is_max_of_both_strands():  # test-pwm.R:483
    base = pm.gintervals("chr1", 200, 240)
    pm.gvtrack_create("pmb", None, "pwm.max", pssm=AC, bidirect=True, extend=True, prior=0.01)
    pm.gvtrack_iterator("pmb", sshift=-200, eshift=200)
    got = pm.gextract("pmb", base, iterator=base)["pmb"][0]
    seq = pm.gseq_extract(pm.gintervals("chr1", 0, 240 + 200 + 1))[0].upper()
    expected = max(_manual_pwm_scores_single_strand(seq, AC, 0.01)
                   + _manual_pwm_scores_single_strand(_revcomp(seq), AC, 0.01))
    assert abs(got - expected) < 1e-3


def test_pwm_total_likelihood_grows_with_the_window():  # test-pwm.R:510
    base = pm.gintervals("chr1", 2000, 2040)
    names = []
    for s in (0, 10, 50, 200):
        n = f"pwm_total_shift_{s}"
        pm.gvtrack_create(n, None, "pwm", pssm=AC, bidirect=False, extend=True, prior=0.01)
        pm.gvtrack_iterator(n, sshift=-s, eshift=s)
        names.append(n)
    vals = pm.gextract(names, base, iterator=base)[names].iloc[0].to_numpy()
    assert (np.diff(vals) >= -1e-7).all()


@pytest.mark.parametrize("func,start,tol", [("pwm", 2000, 1e-6), ("pwm.max", 2100, 5e-6), ("pwm.max.pos", 2200, 0)])
def test_shift_equals_expanded_iterator(func, start, tol):  # test-pwm.R:533, 556, 574
    base60 = pm.gintervals("chr1", start, start + 60)
    base80 = pm.gintervals("chr1", start - 10, start + 70)
    pm.gvtrack_create("shifted", None, func, pssm=AC, bidirect=False, extend=True, prior=0.01)
    pm.gvtrack_iterator("shifted", sshift=-10, eshift=10)
    pm.gvtrack_create("unshifted", None, func, pssm=AC, bidirect=False, extend=True, prior=0.01)
    a = pm.gextract("shifted", base60, iterator=base60)["shifted"][0]
    b = pm.gextract("unshifted", base80, iterator=base80)["unshifted"][0]
    assert abs(a - b) <= tol


@pytest.mark.parametrize("func,extra,kw", [
    ("pwm", {"motif_name": "AC", "position": [1, 2], "score_threshold": [0.5, 0.8]},
     dict(bidirect=False, extend=True, prior=0.01)),
    ("pwm.max.pos", {"source": "JASPAR", "quality": "high"}, dict(bidirect=False, extend=True)),
])
def test_vtrack_pssm_with_extra_columns_ac(func, extra, kw):  # test-pwm.R:685, 739
    df = pd.DataFrame({"A": [1.0, 0.0], "C": [0.0, 1.0], "G": [0.0, 0.0], "T": [0.0, 0.0], **extra})
    iv = pm.gintervals("chr1", 200, 240)
    pm.gvtrack_create("vt_extra", None, func, pssm=df, **kw)
    pm.gvtrack_create("vt_regular", None, func, pssm=AC, **kw)
    s = pm.gextract(["vt_extra", "vt_regular"], iv, iterator=iv)
    assert s["vt_extra"][0] == pytest.approx(s["vt_regular"][0], rel=1e-6)


def test_vtrack_pwm_max_pssm_with_extra_columns():  # test-pwm.R:714
    df = pd.DataFrame({"A": [0.7, 0.1, 0.1], "C": [0.1, 0.7, 0.1], "G": [0.1, 0.1, 0.7], "T": [0.1, 0.1, 0.1],
                       "motif_id": ["pos1", "pos2", "pos3"], "conservation_score": [0.9, 0.85, 0.95]})
    iv = pm.gintervals("chr1", 200, 240)
    pm.gvtrack_create("vt_extra", None, "pwm.max", pssm=df, bidirect=True, prior=0.01)
    pm.gvtrack_create("vt_regular", None, "pwm.max", pssm=df[["A", "C", "G", "T"]].to_numpy(), bidirect=True, prior=0.01)
    s = pm.gextract(["vt_extra", "vt_regular"], iv, iterator=iv)
    assert s["vt_extra"][0] == pytest.approx(s["vt_regular"][0], rel=1e-6)


def test_vtrack_pwm_count_pssm_with_extra_columns():  # test-pwm.R:764
    df = pd.DataFrame({"A": [0.7, 0.1], "C": [0.1, 0.7], "G": [0.1, 0.1], "T": [0.1, 0.1],
                       "annotation": ["start", "end"], "weight": [1.0, 1.0]})
    iv = pm.gintervals("chr1", 200, 240)
    pm.gvtrack_create("vt_extra", None, "pwm.count", pssm=df, score_thresh=-1, bidirect=True)
    pm.gvtrack_create("vt_regular", None, "pwm.count", pssm=df[["A", "C", "G", "T"]].to_numpy(),
                      score_thresh=-1, bidirect=True)
    s = pm.gextract(["vt_extra", "vt_regular"], iv, iterator=iv)
    assert 0 < s["vt_regular"][0] < 40
    assert s["vt_extra"][0] == s["vt_regular"][0]


def _manual(func, base_start, base_end, shift, extend, bidirect, pssm=AC):
    L = pssm.shape[0]
    start = max(0, base_start - shift)
    end = base_end + shift + (L - 1 if extend else 0)
    seq = pm.gseq_extract(pm.gintervals("chr1", start, end))[0].upper()
    fwd = _manual_pwm_scores_single_strand(seq, pssm, 0.01)
    if func == "pwm":
        rev = _manual_pwm_scores_single_strand(_revcomp(seq), pssm, 0.01) if bidirect else []
        return _log_sum_exp(fwd + rev)
    if func == "pwm.max":
        return max(fwd)
    return int(np.argmax(fwd)) + 1  # pwm.max.pos: 1-based in the scan window


@pytest.mark.parametrize("func,base,extend,bidirect,shifts", [
    ("pwm", (2000, 2040), True, False, [0, 1, 5, 10, 25, 50, 100, 250, 1000]),  # :100, :224
    ("pwm", (2000, 2040), False, False, [0, 10, 50, 200]),                       # :262
    ("pwm", (200, 240), True, True, [10, 200]),                                   # :299
    ("pwm.max", (2100, 2140), True, False, [0, 10, 50, 200]),                     # :337
    ("pwm.max", (2100, 2140), False, False, [0, 10, 50, 200]),                    # :371
    ("pwm.max.pos", (2200, 2240), True, False, [0, 10, 50]),                      # :403
    ("pwm.max.pos", (2300, 2340), False, False, [50]),                            # :430
    ("pwm.max.pos", (2450, 2490), True, False, [25]),                             # :455
])
def test_pwm_honors_iterator_shifts(func, base, extend, bidirect, shifts):  # test-pwm.R:100-482
    iv = pm.gintervals("chr1", *base)
    names = []
    for s in shifts:
        n = f"v_shift_{s}"
        pm.gvtrack_create(n, None, func, pssm=AC, bidirect=bidirect, extend=extend, prior=0.01)
        pm.gvtrack_iterator(n, sshift=-s, eshift=s)
        names.append(n)
    got = pm.gextract(names, iv, iterator=iv)
    for s, n in zip(shifts, names, strict=True):
        want = _manual(func, *base, s, extend, bidirect)
        if func == "pwm.max.pos":
            assert got[n][0] == want, s
        else:
            assert got[n][0] == pytest.approx(want, rel=1e-4 if s >= 200 else 1e-6, abs=1e-6), s


def test_pwm_strands_against_manual():  # test-pwm.R:142
    pssm = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05], [0.8, 0.1, 0.05, 0.05]])
    iv = pm.gintervals("chr1", 200, 240)
    kw = dict(pssm=pssm, extend=True, prior=0.01)
    pm.gvtrack_create("pwm_fwd", None, "pwm", bidirect=False, **kw)
    pm.gvtrack_create("pwm_rev", None, "pwm", bidirect=False, strand=-1, **kw)
    pm.gvtrack_create("pwm_bidi", None, "pwm", bidirect=True, **kw)
    s = pm.gextract(["pwm_fwd", "pwm_rev", "pwm_bidi"], iv, iterator=iv)
    seq = pm.gseq_extract(pm.gintervals("chr1", 200, 242))[0].upper()
    fwd = _manual_pwm_scores_single_strand(seq, pssm, 0.01)
    rev = _manual_pwm_scores_single_strand(_revcomp(seq), pssm, 0.01)
    assert s["pwm_fwd"][0] == pytest.approx(_log_sum_exp(fwd), rel=1e-6)
    assert s["pwm_rev"][0] == pytest.approx(_log_sum_exp(rev[:40]), rel=1e-6)
    assert s["pwm_bidi"][0] == pytest.approx(_log_sum_exp(fwd + rev), rel=1e-6)
    assert s["pwm_bidi"][0] >= max(s["pwm_fwd"][0], s["pwm_rev"][0])


def test_minus_strand_sliding_lse_matches_manual():  # test-pwm-sliding-window.R:1020
    starts = [3000 + i for i in range(40)]
    ivs = pm.gintervals(["chr1"] * 40, starts, [s + 60 for s in starts])
    pm.gvtrack_create("pwm_minus", None, "pwm", pssm=AC, bidirect=False, strand=-1, extend=True, prior=0.01)
    got = pm.gextract("pwm_minus", ivs, iterator=ivs)["pwm_minus"].to_numpy()
    want = []
    for s in starts:
        seq = _revcomp(pm.gseq_extract(pm.gintervals("chr1", s, s + 61))[0].upper())
        want.append(_log_sum_exp(_manual_pwm_scores_single_strand(seq, AC, 0.01)[:60]))
    np.testing.assert_allclose(got, want, rtol=1e-6)
