"""Window skipping in the thresholded and maximum PWM scans (src/PwmFastScan.h),
and the hit-position functions built on it. Port of R misha's
tests/testthat/test-pwm-fast-scan.R.

The scanner only decides which windows are scored; the survivors are scored by
the unchanged code. So every result must be IDENTICAL to scoring every window,
which MISHA_DISABLE_PWM_SKIP=1 forces. The PSSMs stress the bound: zero and
near-zero probabilities, widths 1 to 30, N runs, lowercase, thresholds at and
around an exact window score, spatial weights and both strands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _both_ways(monkeypatch, fn):
    monkeypatch.setenv("MISHA_DISABLE_PWM_SKIP", "0")
    skip = fn()
    monkeypatch.setenv("MISHA_DISABLE_PWM_SKIP", "1")
    noskip = fn()
    monkeypatch.delenv("MISHA_DISABLE_PWM_SKIP")
    return skip, noskip


def _same(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif a is None or b is None:
        assert a is None and b is None
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _pssm(rng, w, kind):
    if kind == "sparse":
        m = rng.gamma(0.2, size=(w, 4))
    elif kind == "flat":
        m = rng.uniform(0.2, 1, size=(w, 4))
    elif kind == "tiny":
        m = rng.uniform(size=(w, 4))
        m[np.arange(w), rng.integers(0, 4, w)] = 1e-12
    elif kind == "zero":
        m = rng.uniform(size=(w, 4))
        m[np.arange(w), rng.integers(0, 4, w)] = 0
    else:  # onehot
        m = np.full((w, 4), 0.001)
        m[np.arange(w), rng.integers(0, 4, w)] = 1
    m[m.sum(axis=1) == 0, 0] = 1e-300
    return m


def _seqs(rng, n, length, with_n=True):
    letters = np.array(list("ACGTacgt"))
    p = np.r_[np.full(4, 3), np.ones(4)] / 16
    out = []
    for i in range(n):
        s = rng.choice(letters, length, p=p)
        if with_n and i % 3 == 2:
            at = rng.integers(0, length)
            s[at:at + rng.integers(0, 21)] = "N"
        out.append("".join(s))
    return out


@pytest.fixture(autouse=True)
def _clear():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def test_pwm_score_strings_identical_with_and_without_skipping(monkeypatch):
    """gseq_pwm's C++ path (PWMScorer::score_string) for count and max."""
    rng = np.random.default_rng(20261124)
    seqs = _seqs(rng, 10, 400, with_n=False)  # the C++ path takes no N
    kinds = ["sparse", "flat", "tiny", "zero", "onehot"]
    for it in range(40):
        w = it + 1 if it < 30 else int(rng.integers(1, 31))
        kind = kinds[it % len(kinds)]
        pssm = _pssm(rng, w, kind)
        prior = 0.0 if kind == "zero" else float(rng.choice([0.0, 0.01, 0.1]))
        kw = {"neutral_chars_policy": "log_quarter", "prior": prior, "bidirect": it % 4 != 0}
        if it % 7 == 0:
            kw.update(spat_factor=rng.uniform(0.1, 3, 5), spat_bin=37)
        mx = pm.gseq_pwm(seqs, pssm, mode="max", **kw)
        fin = mx[np.isfinite(mx)]
        for th in ([fin.max(), fin.min(), fin.max() - 1e-6, np.median(fin) - 3] if fin.size else [-1.0]):
            a, b = _both_ways(monkeypatch, lambda: pm.gseq_pwm(seqs, pssm, mode="count", score_thresh=th, **kw))
            _same(a, b)
        a, b = _both_ways(monkeypatch, lambda: pm.gseq_pwm(seqs, pssm, mode="max", **kw))
        _same(a, b)


def test_pwm_vtracks_identical_with_and_without_skipping(monkeypatch):
    rng = np.random.default_rng(3)
    allg = pm.gintervals_all()
    tiles = pm.giterator_intervals(intervals=allg, iterator=5000)
    first = allg.iloc[0]
    starts = np.arange(0, int(first["end"]) - 30000, 25000) + rng.integers(0, 1000)
    ragged = pd.DataFrame({
        "chrom": first["chrom"],
        "start": starts,
        "end": starts + rng.choice([10, 100, 2000, 23000], len(starts)),
    })
    kinds = ["sparse", "tiny", "onehot", "flat"]
    for it in range(8):
        w = int(rng.choice([1, 5, 9, 14, 20, 30]))
        pssm = _pssm(rng, w, kinds[it % 4])
        base = {"pssm": pssm, "bidirect": it % 3 != 0, "strand": -1 if it % 6 == 5 else 1, "prior": 0.01}
        if it % 5 == 4:
            base.update(spat_factor=rng.uniform(0.2, 2, 4), spat_bin=300)
        pm.gvtrack_create("fs_max", None, "pwm.max", **base)
        mx = pm.gextract("fs_max", allg, iterator=50000)["fs_max"].to_numpy()
        th = float(np.nanmax(mx[np.isfinite(mx)])) - [0.5, 3, 8][it % 3]
        pm.gvtrack_create("fs_count", None, "pwm.count", score_thresh=th, **base)
        pm.gvtrack_create("fs_pos", None, "pwm.max.pos", **base)
        pm.gvtrack_create("fs_count_sh", None, "pwm.count", score_thresh=th, **base)
        pm.gvtrack_iterator("fs_count_sh", sshift=-700, eshift=700)
        exprs = ["fs_count", "fs_max", "fs_pos", "fs_count_sh"]
        for iterator in (tiles, ragged, 1000):
            a, b = _both_ways(monkeypatch, lambda: pm.gextract(exprs, allg, iterator=iterator))
            _same(a, b)


def test_gseq_pwm_hits_are_the_windows_gseq_pwm_counts(monkeypatch):
    rng = np.random.default_rng(11)
    seqs = _seqs(rng, 10, 300)
    for w in (1, 4, 9, 17, 30):
        pssm = _pssm(rng, w, "sparse")
        mx = pm.gseq_pwm(seqs, pssm, mode="max")
        # off any window's exact score: gseq_pwm's Python path takes double
        # log-probabilities, the hits the PSSM's float ones (as R misha does)
        th = float(np.median(mx[np.isfinite(mx)])) - 0.37
        for bidirect, strand in ((True, 1), (False, 1), (False, -1)):
            cnt = pm.gseq_pwm(seqs, pssm, mode="count", score_thresh=th, bidirect=bidirect, strand=strand)
            a, b = _both_ways(monkeypatch, lambda: pm.gseq_pwm_hits(
                seqs, pssm, score_thresh=th, bidirect=bidirect, strand=strand))
            _same(a, b)
            h = a
            np.testing.assert_array_equal(np.bincount(h["seq_idx"] - 1, minlength=len(seqs)), cnt)
            assert (h["score"] >= th).all()
            assert ((h["end"] - h["start"] + 1) == w).all()
            if not bidirect:
                assert (h["strand"] == strand).all()
            k = min(len(h), 20)
            if k:
                rows = zip(h["seq_idx"][:k], h["start"][:k], h["end"][:k], strict=True)
                sub = [seqs[i - 1][s - 1:e] for i, s, e in rows]
                np.testing.assert_allclose(
                    pm.gseq_pwm(sub, pssm, mode="max", bidirect=bidirect, strand=strand),
                    h["score"][:k], rtol=1e-6)
    assert len(pm.gseq_pwm_hits("ACGT", _pssm(rng, 3, "flat"), score_thresh=100)) == 0
    with pytest.raises(ValueError, match="score_thresh"):
        pm.gseq_pwm_hits("ACGT", _pssm(rng, 3, "flat"))


def test_gseq_pwm_hits_match_r_misha():
    """The rows R misha's gseq.pwm_hits returns on the same input, to the last
    bit: both packages run the same C++ (pwm_fast_scan::seq_hits)."""
    pssm = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1],
                     [0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.7, 0.1]])
    seqs = ["TTACCGTT", "AACGGTAA", "GGGGGGGG", "accgNACCGG"]
    h = pm.gseq_pwm_hits(seqs, pssm, score_thresh=-4)
    # R: gseq.pwm_hits(seqs, pssm, score.thresh = -4)
    expected = pd.DataFrame({
        "seq_idx": np.array([1, 1, 1, 2, 2, 2, 4, 4, 4], dtype=np.int64),
        "start": np.array([3, 4, 5, 1, 2, 3, 1, 6, 7], dtype=np.int64),
        "end": np.array([6, 7, 8, 4, 5, 6, 4, 9, 10], dtype=np.int64),
        "strand": np.array([1, -1, -1, 1, 1, -1, 1, 1, 1], dtype=np.int64),
        "score": [-1.6773707465831633, -2.7212820216154241, -3.5391157588914117,
                  -3.5391157588914117, -2.7212820216154241, -1.6773707465831633,
                  -1.6773707465831633, -1.6773707465831633, -3.9166391510936496],
    })
    pd.testing.assert_frame_equal(h, expected, check_exact=True)
    np.testing.assert_array_equal(np.bincount(h["seq_idx"] - 1, minlength=4), [3, 3, 0, 3])


def test_gintervals_pwm_hits_are_the_windows_pwm_count_counts(monkeypatch):
    rng = np.random.default_rng(5)
    allg = pm.gintervals_all()
    c0, len0 = allg.iloc[0]["chrom"], int(allg.iloc[0]["end"])
    intervs = pd.DataFrame({
        "chrom": [c0, c0, c0],
        "start": [0, 20000, len0 - 1000],
        "end": [15000, 21000, len0],
    })
    for w in (1, 6, 12, 25):
        pssm = _pssm(rng, w, "sparse")
        for bidirect, strand in ((True, 1), (False, 1), (False, -1)):
            for extend in (True, False):
                kw = {"pssm": pssm, "bidirect": bidirect, "strand": strand, "extend": extend, "prior": 0.01}
                pm.gvtrack_create("fs_mx", None, "pwm.max", **kw)
                mx = pm.gextract("fs_mx", intervs, iterator=intervs)["fs_mx"].to_numpy()
                th = float(np.nanmax(mx[np.isfinite(mx)])) - 4
                pm.gvtrack_create("fs_cnt", None, "pwm.count", score_thresh=th, **kw)
                cnt = pm.gextract("fs_cnt", intervs, iterator=intervs)["fs_cnt"].to_numpy()
                a, b = _both_ways(monkeypatch, lambda: pm.gintervals_pwm_hits(
                    intervs, pssm, score_thresh=th, bidirect=bidirect, strand=strand, extend=extend))
                _same(a, b)
                n_hits = (np.zeros(len(intervs)) if a is None
                          else np.bincount(a["intervalID"] - 1, minlength=len(intervs)))
                np.testing.assert_array_equal(n_hits, np.nan_to_num(cnt))
                if a is not None:
                    assert (a["score"] >= th).all()
                    assert ((a["end"] - a["start"]) == w).all()
                    assert (a["start"].to_numpy() >= intervs["start"].to_numpy()[a["intervalID"] - 1]).all()
                    if not bidirect:
                        assert (a["strand"] == strand).all()
    assert pm.gintervals_pwm_hits(intervs, _pssm(rng, 5, "flat"), score_thresh=100) is None
    with pytest.raises(ValueError, match="score_thresh"):
        pm.gintervals_pwm_hits(intervs, _pssm(rng, 5, "flat"))
