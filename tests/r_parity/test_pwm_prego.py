"""Parity port of R misha ``test-pwm-prego-regression.R``.

Each R test cross-checks misha's PWM vtrack against the external ``prego`` R
package *and* freezes the misha result with ``expect_regression``. We port the
frozen misha side only (no ``prego`` needed): build the same ``pwm`` / ``pwm.max``
vtrack and extract over the test intervals.

All 10 match R exactly (basic, extend, iterator-shift, single- and
bidirectional spatial, max mode). The previously-failing single-strand spatial
cases were fixed: the spatial sliding-window seed (``PWMScorer::compute_motif_at``)
double-counted the reverse strand for a non-bidirectional PSSM when no strand was
explicitly selected; it now scores a single strand, matching R.
"""

from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

# PSSMs as (position x [A, C, G, T]) matrices, matching the R data frames.
_PSSM4 = np.array(
    [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
)
_PSSM2_GENOME = np.array([[0.9, 0.05, 0.025, 0.025], [0.05, 0.9, 0.025, 0.025]])
_PSSM2_BIN = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])


def _clear():
    for v in pm.gvtrack_ls():
        pm.gvtrack_rm(v)


def _iv3_100():
    return pm.gintervals([1, 1, 2], [10000, 20000, 15000], [10100, 20100, 15100])


def _iv3_280():
    return pm.gintervals([1, 1, 2], [10000, 20000, 15000], [10280, 20280, 15280])


def _iv2_300():
    return pm.gintervals([1, 2], [10000, 15000], [10300, 15300])


def _pwm(name, pssm, func, scope, *, extend, bidirect=True, prior=0.01,
         spat_factor=None, spat_bin=None, sshift=None, eshift=None, it=None):
    def f():
        _clear()
        kw = {"bidirect": bidirect, "extend": extend, "prior": prior}
        if spat_factor is not None:
            kw["spat_factor"] = spat_factor
            kw["spat_bin"] = spat_bin
        pm.gvtrack_create(name, None, func=func, pssm=pssm, **kw)
        if sshift is not None or eshift is not None:
            pm.gvtrack_iterator(name, sshift=sshift or 0, eshift=eshift or 0)
        sc = scope()
        return pm.gextract(name, sc, iterator=(it if it is not None else sc))

    return f


_CASES = {
    "pwm_basic_no_extend": (_pwm("pwm_test", _PSSM4, "pwm", _iv3_100, extend=False), None),
    "pwm_basic_with_extend": (_pwm("pwm_test_extend", _PSSM4, "pwm", _iv3_100, extend=True), None),
    "pwm_prego_regression_test_1": (
        _pwm("pwm_shift", _PSSM4, "pwm", lambda: pm.gintervals(1, 1000, 2000),
             extend=False, sshift=-100, eshift=100, it=200),
        None,
    ),
    "pwm_spatial_no_extend": (
        _pwm("pwm_spatial_test", _PSSM4, "pwm", _iv3_280, extend=False,
             spat_factor=[0.5, 1.0, 2.0, 2.5, 2.0, 1.0, 0.5], spat_bin=40),
        None,
    ),
    "pwm_spatial_with_extend": (
        _pwm("pwm_spatial_extend", _PSSM4, "pwm", _iv3_280, extend=True,
             spat_factor=[0.5, 1.0, 2.0, 2.5, 2.0, 1.0, 0.5, 0.5], spat_bin=40),
        None,
    ),
    "pwm_genome_nospatial": (
        _pwm("pwm_nospatial", _PSSM2_GENOME, "pwm", _iv3_280, extend=False, bidirect=False),
        None,
    ),
    "pwm_genome_spatial": (
        _pwm("pwm_spatial", _PSSM2_GENOME, "pwm", _iv3_280, extend=False, bidirect=False,
             spat_factor=[0.5, 1.0, 2.0, 2.5, 2.0, 1.0, 0.5], spat_bin=40),
        None,
    ),
    "pwm_max_no_extend": (_pwm("pwm_max_test", _PSSM4, "pwm.max", _iv3_100, extend=False), None),
    "pwm_max_with_extend": (_pwm("pwm_max_extend", _PSSM4, "pwm.max", _iv3_100, extend=True), None),
    "pwm_spatial_binning": (
        _pwm("pwm_bintest", _PSSM2_BIN, "pwm", _iv2_300, extend=False, bidirect=False,
             spat_factor=[10.0, 1.0, 1.0], spat_bin=100),
        None,
    ),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_pwm_prego(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
