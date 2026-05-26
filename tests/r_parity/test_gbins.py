"""Parity port of R misha ``test-gbins.R`` (binned summary / quantile tables).

``gbins.quantiles`` and ``gbins.summary`` compute per-bin statistics of a value
track (``expr``) stratified by a binning track (the first ``*args`` expression).
All three R regression cases reproduce exactly on R's own test DB.

R flattens the multi-dimensional result array column-major; pymisha returns it
row-major. The baseline comparator's vector path retries with column-major
order, so the orientation is handled transparently.

Pass: 3 / xfail: 0.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

# Bin breaks and percentiles shared by every R case.
_BREAKS = [0, 0.2, 0.3, 0.9, 1.2]
_PCT = [0.2, 0.5, 0.6]


def _quantiles(iterator):
    def f():
        return pm.gbins_quantiles(
            "test.fixedbin", _BREAKS, expr="test.sparse",
            percentiles=_PCT, iterator=iterator,
        )
    return f


def _summary(iterator):
    def f():
        return pm.gbins_summary(
            "test.fixedbin", _BREAKS, expr="test.sparse", iterator=iterator,
        )
    return f


# id -> (callable, xfail_reason_or_None)
_CASES: dict[str, tuple] = {
    "gbins.quantiles.1": (_quantiles(10), None),
    "gbins.quantiles.2": (_quantiles(100), None),
    "gbins.summary.1": (_summary(100), None),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gbins(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
