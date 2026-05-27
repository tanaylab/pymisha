"""Parity port of R misha ``test-giterator.intervals.R`` (32 regressions).

1D iterator grids and 1D track/intervals-set scope+iterator resolution all match
R (the iterator-from-name feature). 2D scope/iterator/band combinations still
differ (GAP_2D_ITER); COMPUTED Hi-C is unsupported (GAP_COMPUTED).
"""
from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D_ITER = "2D scope/iterator/band combinations not at full R parity"
GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"

_gi = pm.giterator_intervals


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _ALL():
    return pm.gintervals_all()


# n -> (callable, xfail_reason_or_None)
_CASES = {
    "1": (lambda: _gi("test.fixedbin", _i1([1, 2], 0, 1000000)), None),
    "2": (lambda: _gi("test.fixedbin", _i1([2, 3])), None),
    "3": (lambda: _gi("test.fixedbin", _i1([2, 3]), iterator=120), None),
    "4": (lambda: _gi("test.rects", _i2([2, 3]), iterator=(100000, 200000)), None),
    "5": (lambda: _gi("test.computed2d", _i2(chroms1=[6, 1, 5], chroms2=[8, 1, 9])), GAP_COMPUTED),
    "7": (lambda: _gi(None, _i1([2, 3]), iterator=120), None),
    "8": (lambda: _gi(None, _ALL(), iterator="test.sparse"), None),
    "9": (lambda: _gi(None, _i1([2, 3]), iterator="test.sparse"), None),
    "10": (lambda: _gi(None, _i1([2, 3]), iterator="test.fixedbin"), None),
    "11": (lambda: _gi("test.sparse", _i1([1, 2], 0, 1000000)), None),
    "12": (lambda: _gi(None, iterator="test.rects_big_rects", band=(-187435, 234560)), None),
    "13": (lambda: _gi(None, _i2([2, 3]), iterator=(12345, 789), band=(-18743, 23456)), None),
    "14": (lambda: _gi("test.generated_1d_1", intervals=_gi("test.generated_1d_2"), iterator=_gi("test.generated_1d_1")), None),
    "15": (lambda: _gi("test.generated_1d_1", intervals=_gi("test.generated_1d_2"), iterator="test.bigintervs_1d_1"), None),
    "16": (lambda: _gi("test.generated_1d_1", intervals=_gi("test.generated_1d_2"), iterator="test.generated_1d_1"), None),
    "17": (lambda: _gi("test.generated_1d_1", intervals="test.bigintervs_1d_2", iterator=_gi("test.generated_1d_1")), None),
    "18": (lambda: _gi("test.generated_1d_1", intervals="test.bigintervs_1d_2", iterator="test.bigintervs_1d_1"), None),
    "19": (lambda: _gi("test.generated_1d_1", intervals="test.bigintervs_1d_2", iterator="test.generated_1d_1"), None),
    "20": (lambda: _gi("test.generated_1d_1", intervals="test.generated_1d_2", iterator=_gi("test.generated_1d_1")), None),
    "21": (lambda: _gi("test.generated_1d_1", intervals="test.generated_1d_2", iterator="test.bigintervs_1d_1"), None),
    "22": (lambda: _gi("test.generated_1d_1", intervals="test.generated_1d_2", iterator="test.generated_1d_1"), None),
    "23": (lambda: _gi("test.generated_2d_5", intervals=_gi("test.generated_2d_6"), iterator=_gi("test.generated_2d_5")), None),
    "24": (lambda: _gi("test.generated_2d_5", intervals=_gi("test.generated_2d_6"), iterator="test.bigintervs_2d_5"), None),
    "25": (lambda: _gi("test.generated_2d_5", intervals=_gi("test.generated_2d_6"), iterator="test.generated_2d_5"), None),
    "26": (lambda: _gi("test.generated_2d_5", intervals="test.bigintervs_2d_6", iterator=_gi("test.generated_2d_5")), None),
    "27": (lambda: _gi("test.generated_2d_5", intervals="test.bigintervs_2d_6", iterator="test.bigintervs_2d_5"), None),
    "28": (lambda: _gi("test.generated_2d_5", intervals="test.bigintervs_2d_6", iterator="test.generated_2d_5"), None),
    "29": (lambda: _gi("test.generated_2d_5", intervals="test.generated_2d_6", iterator=_gi("test.generated_2d_5")), None),
    "30": (lambda: _gi("test.generated_2d_5", intervals="test.generated_2d_6", iterator="test.bigintervs_2d_5"), None),
    "31": (lambda: _gi("test.generated_2d_5", intervals="test.generated_2d_6", iterator="test.generated_2d_5"), None),
    "32": (lambda: _gi("test.generated_2d_6", "test.bigintervs_2d_5"), None),
    "33": (lambda: _gi("test.generated_2d_6", "test.generated_2d_5"), None),
}


@pytest.mark.parametrize("n", list(_CASES))
def test_giterator_intervals(n, request):
    fn, reason = _CASES[n]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), f"giterator.intervals.{n}")
