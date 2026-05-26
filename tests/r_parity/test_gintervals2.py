"""Parity port of R misha ``test-gintervals2.R`` (9 regressions).

Most of this file exercises save / remove / update / gintervals.ls snapshots,
which require writing to the database. R's test DB is read-only here, so those
cases are xfail; the two pure-compute cases (rbind, union) are reproduced
in-memory.
"""
from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_RO = "needs DB write (R test DB is read-only): save / update / gintervals.ls snapshot"
GAP_SETOUT = "intervals.set.out roundtrip (saved-set canonicalization)"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


_CASES = {
    "gintervals.rbind.1": (
        lambda: pm.gintervals_rbind(
            pm.gextract("test.fixedbin", _i1([1, 2], 1000, 4000)),
            pm.gextract("test.fixedbin", _i1([2, "X"], 2000, 5000)),
        ),
        None,
    ),
    "gscreen_and_gintervals.union.1": (
        lambda: pm.gintervals_union(
            pm.gscreen("test.fixedbin > 0.1 & test.fixedbin < 0.3", _i1([1, 2, 4, 8, 9], 0, -1)),
            pm.gscreen("test.fixedbin < 0.2", _i1([1, 2, 4, 7, 9], 0, -1)),
        ),
        None,
    ),
    "gintervals.save.1": (lambda: pm.gextract("test.fixedbin", pm.gscreen("test.fixedbin > 0.2", _i1([1, 2]))), GAP_SETOUT),
    "gintervals.rbind.2": (lambda: None, GAP_RO),
    "gintervals.ls.1": (lambda: None, GAP_RO),
    "gintervals.update.3": (lambda: None, GAP_RO),
    "gintervals.update.4": (lambda: None, GAP_RO),
    "gintervals.update.2d.3": (lambda: None, GAP_RO),
    "gintervals.update.2d.4": (lambda: None, GAP_RO),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gintervals2(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
