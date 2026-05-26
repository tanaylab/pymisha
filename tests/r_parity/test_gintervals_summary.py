"""Parity port of R misha ``test-gintervals.summary.R``.

``gintervals.summary(expr, intervals, iterator)`` returns the *intervals* with
per-interval summary columns (Total intervals, NaN intervals, Min, Max, Sum,
Mean, Std dev). 1D whole-chromosome and gscreen-result inputs match R (the
result is order-independent, so the R `sample()` shuffle is omitted here).

Findings / open gaps (all ``xfail(strict=True)``):

* ``GAP_SUMMARY_ITER`` -- when the *iterator* is a track or interval-set name (a
  DataFrame iterator), pymisha intersects the scope with the iterator and emits
  one row per (iterator-bin ∩ scope) piece, dropping scope intervals that
  overlap no bin. R instead emits one row per *scope* interval, summarizing the
  iterator bins inside it (empty intervals get Total=0 / NaN stats). Root cause:
  ``_preprocess_intervals_iterator`` is shared with ``gextract`` (where the
  intersection is correct). A faithful fix should match the C++
  ``pm_intervals_summary`` Total/NaN-bin semantics (best done in C++ so dense
  NaN bins are still counted), so it is deferred.
* ``GAP_2D`` -- 2D ``gintervals.summary`` interval counts differ from R.
* ``GAP_2D_ITER`` -- 2D track/set/DataFrame iterator combinations differ from R.
* ``GAP_COMPUTED`` -- COMPUTED 2D Hi-C tracks (``test.computed2d``).
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_SUMMARY_ITER = "track-name/DataFrame iterator: scope intersected (empty scope intervals dropped) instead of one summary row per scope interval"
GAP_2D = "2D gintervals.summary interval count differs from R"
GAP_2D_ITER = "2D track/set/DataFrame iterator combinations differ from R"
GAP_COMPUTED = "COMPUTED 2D Hi-C tracks (test.computed2d) not supported"

S = pm.gintervals_summary


def _gi(name):
    return pm.giterator_intervals(name)


def _iv1():
    return pm.gintervals([1, 2], 0, -1)


def _rects_scope():
    return pm.gintervals_2d([2, 3], chroms2=[2, 4])


def _comp_scope():
    return pm.gintervals_2d([6, 1, 5], chroms2=[8, 1, 9])


# id -> (callable, xfail_reason_or_None)
_CASES: dict[str, tuple] = {
    "gintervals.summary_test.fixedbin": (lambda: S("test.fixedbin", _iv1()), None),
    "gintervals.summary_test.sparse": (lambda: S("test.sparse", _iv1()), None),
    "gintervals.summary_test.rects": (lambda: S("test.rects", _rects_scope()), GAP_2D),
    "gintervals.summary_test.computed2d": (lambda: S("test.computed2d", _comp_scope()), GAP_COMPUTED),
    "gintervals.summary_randomized_test.fixedbin": (lambda: S("test.fixedbin", pm.gscreen("test.fixedbin > 0.2", _iv1())), None),
    "gintervals.summary_filtered_test.sparse": (lambda: S("test.sparse", pm.gscreen("test.fixedbin > 0.2", _iv1())), None),
    "gintervals.summary_randomized_test.rects": (lambda: S("test.rects", pm.gscreen("test.rects > 40", _rects_scope())), GAP_2D),
    "gintervals.summary_filtered_test.computed2d": (lambda: S("test.computed2d", pm.gscreen("test.computed2d > 4000000", _comp_scope())), GAP_COMPUTED),
    "summary_test.generated_1d_1_case1": (lambda: S("test.generated_1d_1", _gi("test.generated_1d_2"), iterator=_gi("test.generated_1d_1")), None),
    "summary_test.generated_1d_1_case2": (lambda: S("test.generated_1d_1", _gi("test.generated_1d_2"), iterator="test.bigintervs_1d_1"), None),
    "summary_test.generated_1d_1_case3": (lambda: S("test.generated_1d_1", _gi("test.generated_1d_2"), iterator="test.generated_1d_1"), None),
    "summary_test.generated_1d_1_case4": (lambda: S("test.generated_1d_1", "test.bigintervs_1d_2", iterator=_gi("test.generated_1d_1")), None),
    "summary_test.generated_1d_1_case5": (lambda: S("test.generated_1d_1", "test.bigintervs_1d_2", iterator="test.bigintervs_1d_1"), None),
    "summary_test.generated_1d_1_case6": (lambda: S("test.generated_1d_1", "test.bigintervs_1d_2", iterator="test.generated_1d_1"), None),
    "summary_test.generated_1d_1_case7": (lambda: S("test.generated_1d_1", "test.generated_1d_2", iterator=_gi("test.generated_1d_1")), None),
    "summary_test.generated_1d_1_case8": (lambda: S("test.generated_1d_1", "test.generated_1d_2", iterator="test.bigintervs_1d_1"), None),
    "summary_test.generated_1d_1_case9": (lambda: S("test.generated_1d_1", "test.generated_1d_2", iterator="test.generated_1d_1"), None),
    "summary_test.generated_2d_5_case1": (lambda: S("test.generated_2d_5", _gi("test.generated_2d_6"), iterator=_gi("test.generated_2d_5")), GAP_2D_ITER),
    "summary_test.generated_2d_5_case2": (lambda: S("test.generated_2d_5", _gi("test.generated_2d_6"), iterator="test.bigintervs_2d_5"), GAP_2D_ITER),
    "summary_test.generated_2d_5_case3": (lambda: S("test.generated_2d_5", _gi("test.generated_2d_6"), iterator="test.generated_2d_5"), GAP_2D_ITER),
    "summary_test.generated_2d_5_case4": (lambda: S("test.generated_2d_5", "test.bigintervs_2d_6", iterator=_gi("test.generated_2d_5")), GAP_2D_ITER),
    "summary_test.generated_2d_5_case5": (lambda: S("test.generated_2d_5", "test.bigintervs_2d_6", iterator="test.bigintervs_2d_5"), GAP_2D_ITER),
    "summary_test.generated_2d_5_case6": (lambda: S("test.generated_2d_5", "test.bigintervs_2d_6", iterator="test.generated_2d_5"), GAP_2D_ITER),
    "summary_test.generated_2d_5_case7": (lambda: S("test.generated_2d_5", "test.generated_2d_6", iterator=_gi("test.generated_2d_5")), GAP_2D_ITER),
    "summary_test.generated_2d_5_case8": (lambda: S("test.generated_2d_5", "test.generated_2d_6", iterator="test.bigintervs_2d_5"), GAP_2D_ITER),
    "summary_test.generated_2d_5_case9": (lambda: S("test.generated_2d_5", "test.generated_2d_6", iterator="test.generated_2d_5"), GAP_2D_ITER),
    # intervals.set.out cases ported in-memory (the summary content is independent
    # of whether it is streamed to a track and re-loaded).
    "gintervals_fixedbin_result": (lambda: S(
        "test.fixedbin",
        intervals=pm.gscreen("test.fixedbin > 0.25 & test.fixedbin < 0.35", pm.gintervals([1, 2], 0, -1)),
        iterator=pm.gscreen("test.fixedbin > 0.2 & test.fixedbin < 0.3", pm.gintervals([1, 2, 3], 0, -1)),
    ), None),
    "gintervals_rects_result": (lambda: S(
        "test.rects",
        pm.gscreen("test.rects > 40", pm.gintervals_2d([1, 2, 5, 8], 0, -1)),
        iterator=[1, 1],
    ), GAP_2D),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gintervals_summary(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
