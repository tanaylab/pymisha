"""Parity port of R misha ``test-gpartition.R`` (5 regressions).

The fixedbin partition is order-independent, so R's seeded row-shuffle of the
input is not reproduced. gpartition does not support 2D intervals, and the
data-size / intervals.set.out variants exercise a path not yet ported.
"""
from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D = "gpartition does not support 2D intervals"
GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"
GAP_DSIZE = "gpartition gmax.data.size + intervals.set.out path not ported"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _seq(a, b, by):
    n = int(round((b - a) / by))
    return a + np.arange(n + 1) * by


_CASES = {
    "gpartition_fixedbin_sampling_result": (lambda: pm.gpartition("test.fixedbin", _seq(0, 1, 0.1), intervals=pm.gscreen("test.fixedbin > 0.14", _i1([1, 2], 0, -1))), None),
    "gpartition_rects_result": (lambda: pm.gpartition("test.rects", _seq(50, 100, 1), intervals=_i2(chroms1=[2, 3], chroms2=[2, 4])), GAP_2D),
    "gpartition_computed2d_result": (lambda: pm.gpartition("test.computed2d", _seq(5000000, 10000000, 1000000), intervals=_i2(chroms1=[6, 5], chroms2=[8, 9])), GAP_2D),
    "gpartition_fixedbin_sampling_data_size_result": (lambda: None, GAP_DSIZE),
    "gpartition_rects_data_size_result": (lambda: None, GAP_DSIZE),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gpartition(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
