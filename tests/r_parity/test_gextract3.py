"""Parity port of R misha ``test-gextract3.R`` (41 regressions).

Covers basic 1D/2D extraction, arithmetic expressions, multi-expression
extracts, band queries and the full matrix of numeric / track-name iterators.

Classification (vs R baselines):
* PASS: 1D dense/sparse + 2D rects basic / numeric-grid / rects-iterator.
* xfail GAP_ARRAY: any array-track expression or array-track iterator.
* xfail GAP_COMPUTED: any COMPUTED Hi-C (test.computed2d) expression or iterator.
* xfail GAP_BAND_59: rects band query returns a different coordinate set
  (single case; flagged for investigation).
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY = "array-track extract/iterator not supported"
GAP_COMPUTED = "COMPUTED Hi-C tracks (test.computed2d) not supported"
GAP_BAND_59 = "rects band query coordinate set differs from R (investigate)"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


# baseline_id -> (callable, xfail_reason_or_None)
_CASES = {
    "gextract.65": (lambda: pm.gextract("test.fixedbin", _i1([1, 2], 0, 1000000)), None),
    "gextract.64": (lambda: pm.gextract("test.sparse", _i1([1, 2], 0, 1000000)), None),
    "gextract.63": (lambda: pm.gextract("test.array", _i1([1, 2], 0, 1000000)), None),
    "gextract.62": (lambda: pm.gextract("test.rects", _i2([2, 3], 10000000, 50000000, [2, 4], 30000000, 80000000)), None),
    "gextract.59": (lambda: pm.gextract("test.rects_big_rects", _i2(list(range(1, 21))), band=(-1874356, 234560)), None),
    "gextract.58": (lambda: pm.gextract("test.computed2d", _i2([6, 8], 10000000, 50000000, [1, 3], 30000000, 80000000)), GAP_COMPUTED),
    "gextract.55": (lambda: pm.gextract("test.computed2d", _i2(list(range(1, 21))), band=(-1874356, 234560)), GAP_COMPUTED),
    "gextract.54": (lambda: pm.gextract("2 * test.fixedbin + 17", _i1([1, 2], 0, 1000000)), None),
    "gextract.53": (lambda: pm.gextract("2 * test.sparse + 17", _i1([1, 2], 0, 1000000)), None),
    "gextract.52": (lambda: pm.gextract("2 * test.array + 17", _i1([1, 2], 0, 1000000)), None),
    "gextract.51": (lambda: pm.gextract("2 * test.rects + 17", _i2([2, 3], 10000000, 50000000, [2, 4], 30000000, 80000000)), None),
    "gextract.50": (lambda: pm.gextract("2 * test.computed2d + 17", _i2([6, 8], 10000000, 50000000, [1, 3], 30000000, 80000000)), GAP_COMPUTED),
    "gextract.47": (lambda: pm.gextract(["test.fixedbin", "test.sparse"], _i1([1, 2], 0, 1000000), iterator="test.fixedbin"), None),
    "gextract.45": (lambda: pm.gextract(["test.fixedbin", "test.array"], _i1([1, 2], 0, 1000000), iterator="test.fixedbin"), None),
    "gextract.44": (lambda: pm.gextract(["test.rects", "test.rects * 3"], _i2([2, 3], 10000000, 50000000, [2, 4], 30000000, 80000000), iterator="test.rects"), None),
    "gextract.43": (lambda: pm.gextract(["test.computed2d", "test.computed2d * 3"], _i2([6, 5], 10000000, 50000000, [8, 9], 30000000, 80000000), iterator="test.computed2d"), GAP_COMPUTED),
    "gextract.42": (lambda: pm.gextract(["test.rects", "test.rects * 3"], _i2([6, 5], 10000000, 50000000, [8, 9], 30000000, 80000000), iterator="test.computed2d"), GAP_COMPUTED),
    "gextract.41": (lambda: pm.gextract(["test.computed2d", "test.computed2d * 3"], _i2([6, 5], 10000000, 50000000, [8, 9], 30000000, 80000000), iterator="test.rects"), GAP_COMPUTED),
    "gextract.40": (lambda: pm.gextract("test.fixedbin", _i1([2, 3])), None),
    "gextract.39": (lambda: pm.gextract("test.sparse", _i1([2, 3])), None),
    "gextract.38": (lambda: pm.gextract("test.array", _i1([2, 3])), None),
    "gextract.37": (lambda: pm.gextract("test.rects", _i2(chroms1=[2, 3], chroms2=[2, 4])), None),
    "gextract.36": (lambda: pm.gextract("test.computed2d", _i2(chroms1=[6, 5], chroms2=[8, 9])), GAP_COMPUTED),
    "gextract.35": (lambda: pm.gextract("test.fixedbin", _i1([2, 3]), iterator=120), None),
    "gextract.34": (lambda: pm.gextract("test.sparse", _i1([2, 3]), iterator=120), None),
    "gextract.33": (lambda: pm.gextract("test.array", _i1([2, 3]), iterator=120), None),
    "gextract.27": (lambda: pm.gextract("test.rects", _i2(chroms1=[2, 3], chroms2=[2, 4]), iterator=(100000, 100000)), None),
    "gextract.26": (lambda: pm.gextract("test.computed2d", _i2(chroms1=[6, 5], chroms2=[8, 9]), iterator=(100000, 100000)), GAP_COMPUTED),
    "gextract.25": (lambda: pm.gextract("test.fixedbin", _i1([2, 3]), iterator="test.fixedbin"), None),
    "gextract.24": (lambda: pm.gextract("test.sparse", _i1([2, 3]), iterator="test.fixedbin"), None),
    "gextract.23": (lambda: pm.gextract("test.array", _i1([2, 3]), iterator="test.fixedbin"), None),
    "gextract.20": (lambda: pm.gextract("test.fixedbin", _i1([2, 3]), iterator="test.sparse"), None),
    "gextract.19": (lambda: pm.gextract("test.sparse", _i1([2, 3]), iterator="test.sparse"), None),
    "gextract.18": (lambda: pm.gextract("test.array", _i1([2, 3]), iterator="test.sparse"), None),
    "gextract.15": (lambda: pm.gextract("test.fixedbin", _i1([2, 3]), iterator="test.array"), None),
    "gextract.14": (lambda: pm.gextract("test.sparse", _i1([2, 3]), iterator="test.array"), None),
    "gextract.13": (lambda: pm.gextract("test.array", _i1([2, 3]), iterator="test.array"), None),
    "gextract.7": (lambda: pm.gextract("test.rects", _i2(chroms1=[2, 3], chroms2=[2, 4]), iterator="test.rects"), None),
    "gextract.6": (lambda: pm.gextract("test.computed2d", _i2(chroms1=[6, 5], chroms2=[8, 9]), iterator="test.rects"), GAP_COMPUTED),
    "gextract.2": (lambda: pm.gextract("test.rects", _i2(chroms1=[6, 5], chroms2=[8, 9]), iterator="test.computed2d"), GAP_COMPUTED),
    "gextract.1": (lambda: pm.gextract("test.computed2d", _i2(chroms1=[6, 5], chroms2=[8, 9]), iterator="test.computed2d"), GAP_COMPUTED),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gextract3(bid, request):
    fn, xfail_reason = _CASES[bid]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), bid)
