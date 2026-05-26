"""Parity port of R misha ``test-gdist.R`` (6 regressions)."""
from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"


def _seq(a, b, by):
    n = int(round((b - a) / by))
    return a + np.arange(n + 1) * by


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


_CASES = {
    "gdist.1": (lambda: pm.gdist("test.fixedbin", _seq(0, 1, 0.001)), None),
    "gdist.2": (lambda: pm.gdist("test.fixedbin", _seq(0.2, 1, 0.001)), None),
    "gdist.3": (lambda: pm.gdist("test.fixedbin", _seq(0, 1, 0.01), "test.fixedbin+0.3", _seq(0, 1, 0.01)), None),
    "gdist_gscreen.1": (lambda: pm.gdist("test.fixedbin", _seq(0.2, 1, 0.001), intervals=pm.gscreen("test.fixedbin > 0.2", _i1([1, 2], 0, -1))), None),
    "gdist_gscreen_2d.1": (lambda: pm.gdist("test.rects", _seq(8, 10, 0.1), intervals=pm.gscreen("test.rects > 9")), None),
    "gdist_gscreen_2d.2": (lambda: pm.gdist("test.computed2d", _seq(9000000, 10000000, 100000), intervals=pm.gscreen("test.computed2d > 9500000")), GAP_COMPUTED),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gdist(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
