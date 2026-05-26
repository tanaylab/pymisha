"""Parity port of R misha ``test-gintervals.intersect.R`` (10 regressions)."""
from __future__ import annotations

import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D = "2D gintervals.intersect/union not supported"
GAP_NAMED_TRACK = "track name (not intervals set) as intersect operand not resolved"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _scr(*a, **k):
    return pm.gscreen(*a, **k)


def _rb(*dfs):
    return pd.concat(dfs, ignore_index=True)


_CASES = {
    "gintervals.intersect.1": (lambda: pm.gintervals_intersect(_scr("test.fixedbin > 0.1", _i1([1, 2], 0, -1)), _scr("test.fixedbin < 0.2", _i1([1, 2], 0, -1))), None),
    "gintervals.intersect.2": (lambda: pm.gintervals_intersect(_rb(_scr("test.fixedbin > 0.1 & test.fixedbin < 0.3", _i1([1, 2])), _scr("test.fixedbin > 0.2 & test.fixedbin < 0.4", _i1([1, 2]))), _scr("test.fixedbin > 0.34 & test.fixedbin < 0.5", _i1([1, 2]))), None),
    "gintervals.intersect.3": (lambda: pm.gintervals_intersect(_scr("test.rects > 40", _i2([1, 2, 5, 8], 0, -1)), _scr("test.rects < 60", _i2([2, 4, 5, 9], 0, -1))), GAP_2D),
    "gintervals.intersect.named.1": (lambda: pm.gintervals_intersect("test.bigintervs_1d_1", "test.bigintervs_1d_2"), None),
    "gintervals.intersect.named.2": (lambda: pm.gintervals_intersect("test.generated_1d_1", "test.generated_1d_2"), GAP_NAMED_TRACK),
    "gintervals.intersect.named.3": (lambda: pm.gintervals_intersect("test.bigintervs_2d_5", "test.bigintervs_2d_6"), GAP_2D),
    "gintervals.intersect.named.4": (lambda: pm.gintervals_intersect("test.generated_2d_5", "test.generated_2d_6"), GAP_2D),
    "gintervals.union.1": (lambda: pm.gintervals_union(_scr("test.fixedbin > 0.1 & test.fixedbin < 0.3", _i1([1, 2], 0, -1)), _scr("test.fixedbin < 0.2", _i1([1, 2], 0, -1))), None),
    "gintervals.union.2": (lambda: pm.gintervals_union(_rb(_scr("test.fixedbin > 0.1 & test.fixedbin < 0.3", _i1([1, 2])), _scr("test.fixedbin > 0.2 & test.fixedbin < 0.4", _i1([1, 2]))), _scr("test.fixedbin > 0.34 & test.fixedbin < 0.5", _i1([1, 2]))), None),
    "gintervals.union.3": (lambda: pm.gintervals_union("test.bigintervs_1d_1", "test.bigintervs_1d_2"), None),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gintervals_intersect(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
