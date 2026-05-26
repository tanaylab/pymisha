"""Parity port of R misha ``test-gquantiles.R`` (4 regressions)."""
from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


_CASES = {
    "gquantiles_fixedbin_result": (lambda: pm.gquantiles("test.fixedbin+0.2", percentiles=[0.5, 0.3, 0.2, 0.9], intervals=pm.gscreen("test.fixedbin > 0.2", _i1([1, 2], 0, -1))), None),
    "gquantiles_rects_result": (lambda: pm.gquantiles("test.rects", percentiles=[0.5, 0.3, 0.2, 0.9, 0.999], intervals=_i2(chroms1=[2, 3], chroms2=[2, 4])), None),
    "gquantiles_computed2d_result": (lambda: pm.gquantiles("test.computed2d", percentiles=[0.5, 0.3, 0.2, 0.9, 0.999], intervals=_i2(chroms1=[6, 5], chroms2=[8, 9])), GAP_COMPUTED),
    "gquantiles_fixedbin_no_intervals_result": (lambda: pm.gquantiles("test.fixedbin+0.2", percentiles=[0.5, 0.999]), None),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gquantiles(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
