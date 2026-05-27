"""Parity port of R misha ``test-gintervals.neighbors.R``.

The four ``*.2d.2/.2d.3/.6/.7`` cases are commented out in the R source (their
ordering changed after a dist2segment fix) and are not ported. 2D
gintervals_neighbors is not yet implemented in pymisha.
"""
from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D = "2D gintervals_neighbors not implemented"
# maxneighbors=1 with two equidistant neighbors (one upstream, one downstream at
# the same gap): pymisha's SegmentFinder NNIterator pops the upstream one first,
# R the downstream one. Both are valid nearest neighbors at the same distance;
# the order among equal-distance objects is a priority_queue/tree tie-break
# artifact (the NN code is byte-identical to R). ~1% of rows differ in target.
GAP_NN_TIE = "maxn=1 equidistant tie-break differs from R (NNIterator pop order)"

_N = pm.gintervals_neighbors


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _scr(*a, **k):
    return pm.gscreen(*a, **k)


def _nb8():
    i1 = _scr("test.fixedbin > 0.2 & test.fixedbin < 0.3", _i1([1, 2], 0, -1))
    i2 = _scr("test.fixedbin > 0.25 & test.fixedbin < 0.35", _i1([1, 2], 0, -1)).copy()
    i2["usercol1"] = "aaa"
    i2["usercol2"] = [10 + k for k in range(1, len(i2) + 1)]
    return _N(i1, i2, 1)


def _nb_2d(maxn=1, **kw):
    i1 = _scr("test.rects > 95")
    i2 = _scr("test.rects < 97 & test.rects > 94").copy()
    i2["blabla"] = range(1, len(i2) + 1)
    return _N(i1, i2, maxn, **kw)


def _intervs():
    return _scr("test.fixedbin > 0.3")


_CASES = {
    "gintervals.neighbors.1": (lambda: _N("test.tss", _intervs(), 100, mindist=-10000, maxdist=10000), None),
    "gintervals.neighbors.2": (lambda: _N("test.tss", _intervs(), 100, mindist=2000, maxdist=10000), None),
    "gintervals.neighbors.3": (lambda: _N("test.tss", _intervs(), 100, mindist=-10000, maxdist=-2000), None),
    "gintervals.neighbors.4": (lambda: _N(_intervs(), "test.tss", 100, mindist=-10000, maxdist=-2000), None),
    "gintervals.neighbors.5": (lambda: _N(_i2(1), _i2(1)), GAP_2D),
    "gintervals.neighbors.2d.1": (lambda: _nb_2d(100, mindist1=10000, maxdist1=20000, mindist2=50000, maxdist2=70000), GAP_2D),
    "gintervals.neighbors.2d.4": (lambda: _N("test.bigintervs_2d_5", "test.bigintervs_2d_6"), GAP_2D),
    "gintervals.neighbors.2d.5": (lambda: _N("test.generated_2d_5", "test.generated_2d_6"), GAP_2D),
    "gintervals.neighbors.8": (_nb8, GAP_NN_TIE),
    "gintervals.neighbors.9": (_nb_2d, GAP_2D),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gintervals_neighbors(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
