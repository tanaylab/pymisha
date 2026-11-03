"""Tests for potts.max with max_edits, ported from R misha's
tests/testthat/test-potts-max-edits.R."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from _potts_oracle import POTTS_BASES, potts_ref_anchors, potts_ref_model

import pymisha as pm
from pymisha.extract import _can_vtracks_use_cpp


@pytest.fixture(autouse=True)
def _clean_vtracks():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def potts_one_edit_oracle(seq, model, bidirect, strand):
    """The best anchor score over the sequence and every sequence one
    substitution away, from the oracle in _potts_oracle.py. It substitutes over
    the whole read region and maximises over all anchors, so it does not rely
    on the per-anchor decomposition the kernel uses."""

    def score(s):
        v = potts_ref_anchors(s, model, bidirect=bidirect, strand=strand)
        return -math.inf if np.isnan(v).all() else float(np.nanmax(v))

    chars = list(seq.upper())
    best = score("".join(chars))
    for j, c in enumerate(chars):
        if c not in POTTS_BASES:
            continue
        for b in POTTS_BASES:
            if b == c:
                continue
            s2 = chars.copy()
            s2[j] = b
            best = max(best, score("".join(s2)))
    return best if math.isfinite(best) else math.nan


def test_potts_max_with_max_edits_1_matches_the_brute_force_oracle():
    cases = [
        (w, mode, bidirect, strand)
        for strand, bidirect, mode, w in itertools.product(
            (1, -1), (True, False), ("full", "sparse", "none"), (5, 7)
        )
        if not (bidirect and strand == -1)
    ]
    for r, (w, mode, bidirect, strand) in enumerate(cases, start=1):
        m = potts_ref_model(w, npair_mode=mode, seed=200 + r)
        pm.gvtrack_create("p1", None, "potts.max",
                          params=dict(m, bidirect=bidirect, strand=strand, max_edits=1))
        for ln in (1, 9):
            start = 5000 + 37 * r
            iv = pm.gintervals(1, start, start + ln)
            got = float(pm.gextract("p1", iv, iterator=iv)["p1"].iloc[0])
            seq = pm.gseq_extract(pm.gintervals(1, start, start + ln + w - 1))[0]
            want = potts_one_edit_oracle(seq, m, bidirect, strand)
            assert got == pytest.approx(want, rel=1e-5), f"case {r}, len {ln}"
        pm.gvtrack_rm("p1")


def test_max_edits_0_is_potts_max_and_max_edits_1_never_scores_below_it():
    m = potts_ref_model(6, npair_mode="full", seed=11)
    pm.gvtrack_create("p0", None, "potts.max", params=dict(m, bidirect=True))
    pm.gvtrack_create("p0e", None, "potts.max", params=dict(m, bidirect=True, max_edits=0))
    pm.gvtrack_create("p1", None, "potts.max", params=dict(m, bidirect=True, max_edits=1))
    x = pm.gextract(["p0", "p0e", "p1"], pm.gintervals(1, 10000, 12000), iterator=1)
    np.testing.assert_array_equal(x["p0e"], x["p0"])
    ok = ~np.isnan(x["p1"]) & ~np.isnan(x["p0"])
    assert (x["p1"][ok] >= x["p0"][ok] - 1e-5).all()
    assert (x["p1"][ok] > x["p0"][ok] + 1e-6).mean() > 0.5


def test_max_edits_is_validated():
    m = potts_ref_model(5, npair_mode="full", seed=5)
    for bad in (2, math.nan, True):
        with pytest.raises(ValueError, match="0 or 1"):
            pm.gvtrack_create("bad", None, "potts.max", params=dict(m, max_edits=bad))
    for f in ("potts", "potts.max.pos"):
        for me in (1, 0):
            with pytest.raises(ValueError, match="potts.max only"):
                pm.gvtrack_create("bad", None, f, params=dict(m, max_edits=me))
    for me in (0, 1):
        with pytest.raises(ValueError, match="potts.max only"):
            pm.gvtrack_create("bad", None, "potts.count",
                              params=dict(m, score_thresh=0, max_edits=me))


# Not in R's suite. A filtered vtrack is scored part by part through
# pm_vtrack_compute (src/PMVTrack.cpp), a second C++ path next to the
# expression scanner's; max_edits has to reach both.
def test_a_filtered_potts_max_with_max_edits_1_scores_like_the_unfiltered_one():
    m = potts_ref_model(6, npair_mode="full", seed=13)
    params = dict(m, bidirect=True, max_edits=1)
    pm.gvtrack_create("p1", None, "potts.max", params=params)
    pm.gvtrack_create("p1f", None, "potts.max", params=params)
    pm.gvtrack_create("p0", None, "potts.max", params=dict(m, bidirect=True))
    # A mask far from the bins leaves each one a single unmasked part.
    pm.gvtrack_filter("p1f", filter=pm.gintervals(1, 400000, 400100))
    assert not _can_vtracks_use_cpp(["p1f"])  # so it is scored part by part
    x = pm.gextract(["p1", "p1f", "p0"], pm.gintervals(1, 10000, 10200), iterator=20)
    np.testing.assert_allclose(x["p1f"], x["p1"], rtol=1e-6)
    assert (x["p1f"] > x["p0"] + 1e-6).all()
