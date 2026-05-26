"""Parity port of R misha ``test-gsummary.R`` (11 regressions)."""
from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY = "array-track extract not supported"
GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _scr1d():
    return pm.gscreen("test.fixedbin > 0.2", _i1([1, 2], 0, -1))


_CASES = {
    "gsummary_fixedbin": (lambda: pm.gsummary("test.fixedbin"), None),
    "gsummary_sparse": (lambda: pm.gsummary("test.sparse"), None),
    "gsummary_array": (lambda: pm.gsummary("test.array"), GAP_ARRAY),
    "gsummary_rects": (lambda: pm.gsummary("test.rects"), None),
    "gsummary_computed2d": (lambda: pm.gsummary("test.computed2d"), GAP_COMPUTED),
    "gsummary_fixedbin_gscreen_filtered": (lambda: pm.gsummary("test.fixedbin", _scr1d()), None),
    "gsummary_sparse_gscreen_filtered": (lambda: pm.gsummary("test.sparse", _scr1d()), None),
    "gsummary_array_gscreen_filtered": (lambda: pm.gsummary("test.array", _scr1d()), GAP_ARRAY),
    "gsummary_rects_gscreen_filtered": (lambda: pm.gsummary("test.rects", pm.gscreen("test.rects > 40", _i2(chroms1=[2, 3], chroms2=[2, 4]))), None),
    "gsummary_computed2d_gscreen_filtered": (lambda: pm.gsummary("test.computed2d", pm.gscreen("test.computed2d > 4000000", _i2(chroms1=[6, 5], chroms2=[8, 9]))), GAP_COMPUTED),
    "gsummary_generated_1d_limited_data_size": (lambda: pm.gsummary("test.generated_1d_1", "test.generated_1d_2"), None),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gsummary(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
