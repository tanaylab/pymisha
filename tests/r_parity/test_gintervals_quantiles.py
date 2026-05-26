"""Parity port of R misha ``test-gintervals.quantiles.R``.

``gintervals.quantiles`` with the default (per-scope-interval) iterator matches R
both whole-genome and over a screened scope.

Open gap marked ``xfail(strict=True)``:

* ``GAP_SUMMARY_ITER`` -- with an explicit track / intervals-set iterator, R emits
  one row per *scope* interval (NaN quantiles where no iterator bin falls inside),
  while pymisha emits one row per (iterator-bin ∩ scope) piece -- different
  coordinates and row count (e.g. 175706 vs 274493). Same root cause as the
  ``gintervals.summary`` iterator gap; the fix is the shared C++ change to emit
  one row per scope interval.

The ``intervals.set.out`` cases in the R file use ``expect_equal(load, recompute)``
rather than ``expect_regression`` -- not frozen baselines -- so they aren't ported.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_SUMMARY_ITER = (
    "gintervals.quantiles with an explicit iterator emits one row per (iterator-bin ∩ scope); "
    "R emits one row per scope interval (NaN when empty). Shared GAP_SUMMARY_ITER C++ fix."
)


def _q1():
    return pm.gintervals_quantiles("test.fixedbin+0.2", [0.5, 0.3, 0.2, 0.9], pm.gintervals_all())


def _q2():
    intervs = pm.gscreen("test.fixedbin > 0.2", pm.gintervals([1, 2], 0, -1))
    return pm.gintervals_quantiles("test.fixedbin+0.2", [0.5, 0.3, 0.2, 0.9], intervs)


def _q3():
    intervs1 = pm.gscreen("test.fixedbin > 0.2 & test.fixedbin < 0.3", pm.gintervals([1, 2, 3], 0, -1))
    intervs2 = pm.gscreen("test.fixedbin > 0.25 & test.fixedbin < 0.35", pm.gintervals([1, 2], 0, -1))
    return pm.gintervals_quantiles("test.fixedbin", [0.5, 0.2, 0.9], intervals=intervs2, iterator=intervs1)


def _q4():
    intervs = pm.gscreen("test.fixedbin > 0.2", pm.gintervals([1, 3], 0, -1))
    return pm.gintervals_quantiles("test.fixedbin+0.2", [0.5, 0.3, 0.2, 0.9], intervs, iterator="test.sparse")


_CASES = {
    "gintervals.quantiles.1": (_q1, None),
    "gintervals.quantiles.2": (_q2, None),
    "gintervals.quantiles.3": (_q3, GAP_SUMMARY_ITER),
    "gintervals.quantiles.4": (_q4, GAP_SUMMARY_ITER),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gintervals_quantiles(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
