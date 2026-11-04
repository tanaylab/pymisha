"""Ports of R misha's test-pwm-edit-distance.R blocks pymisha lacked:
191 (iterator shifts), 698 (PSSM with extra columns), 1769 (LSE <= max,
direction "above"), 1980 (LSE in gscreen), 2089 (LSE with iterator=500)."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from test_pwm_edit_distance import _manual_pwm_edit_distance

AC = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@pytest.fixture(autouse=True)
def _clear():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def test_edit_distance_iterator_shifts():  # :191
    base = pm.gintervals("chr1", 2000, 2040)
    pm.gvtrack_create("edist_shift", None, "pwm.edit_distance", pssm=AC, score_thresh=-3.0,
                      bidirect=False, extend=True, prior=0)
    pm.gvtrack_iterator("edist_shift", sshift=-10, eshift=10)
    got = pm.gextract("edist_shift", base, iterator=base)["edist_shift"][0]
    seq = pm.gseq_extract(pm.gintervals("chr1", 1990, 2040 + 10 + 1))[0].upper()
    assert got == pytest.approx(_manual_pwm_edit_distance(seq, AC, -3.0), abs=1e-6)


def test_edit_distance_pssm_with_extra_columns():  # :698
    df = pd.DataFrame({"A": [1.0, 0.0], "C": [0.0, 1.0], "G": [0.0, 0.0], "T": [0.0, 0.0],
                       "motif_name": "AC", "position": [1, 2], "conservation": [0.9, 0.95]})
    iv = pm.gintervals("chr1", 200, 240)
    kw = dict(score_thresh=-5.0, bidirect=False, extend=False, prior=0)
    pm.gvtrack_create("edist_extra", None, "pwm.edit_distance", pssm=df, **kw)
    pm.gvtrack_create("edist_regular", None, "pwm.edit_distance", pssm=AC, **kw)
    r = pm.gextract(["edist_extra", "edist_regular"], iv, iterator=iv)
    assert r["edist_extra"][0] == pytest.approx(r["edist_regular"][0])


def test_lse_edit_distance_at_most_max_edit_distance():  # :1769
    iv = pm.gintervals("chr1", 200, 240)
    kw = dict(pssm=AC, score_thresh=-3.0, bidirect=False, extend=False, prior=0)
    pm.gvtrack_create("lse_edist", None, "pwm.edit_distance.lse", **kw)
    pm.gvtrack_create("max_edist", None, "pwm.edit_distance", **kw)
    r = pm.gextract(["lse_edist", "max_edist"], iv, iterator=iv)
    assert not np.isnan(r["lse_edist"][0]) and not np.isnan(r["max_edist"][0])
    assert r["lse_edist"][0] <= r["max_edist"][0] + 1e-6


def test_lse_edit_distance_in_gscreen():  # :1980
    pm.gvtrack_create("lse_screen", None, "pwm.edit_distance.lse", pssm=AC, score_thresh=-5.0,
                      bidirect=False, extend=False, prior=0)
    screened = pm.gscreen("~np.isnan(lse_screen) & (lse_screen <= 1)", pm.gintervals("chr1", 0, 5000), iterator=20)
    assert screened is not None and len(screened) > 0
    verify = pm.gextract("lse_screen", screened, iterator=screened)
    assert verify["lse_screen"].notna().all() and (verify["lse_screen"] <= 1).all()


def test_lse_edit_distance_iterator_500():  # :2089
    pssm = np.array([[0.9, 0.03, 0.03, 0.04], [0.03, 0.9, 0.03, 0.04],
                     [0.03, 0.03, 0.9, 0.04], [0.04, 0.03, 0.03, 0.9]])
    pm.gvtrack_create("lse_wide", None, "pwm.edit_distance.lse", pssm=pssm, score_thresh=-5.0,
                      bidirect=False, extend=False, prior=0)
    r = pm.gextract("lse_wide", pm.gintervals("chr1", 0, 2000), iterator=500)
    assert len(r) == 4
    v = r["lse_wide"].to_numpy()
    assert (np.isnan(v) | (v >= 0)).all()
