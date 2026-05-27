"""Parity port of R misha ``test-gextract2.R`` (gscreen/giterator iterators).

1D iterator/scope resolution from track and intervals-set names now matches R
(see the iterator-from-name feature). The remaining 2D track/set/DataFrame
iterator combinations still differ from R (row counts / coordinates); those are
marked ``xfail`` with ``GAP_2D_ITER`` until 2D-scanner iterator parity lands.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D_ITER = "2D track/set/DataFrame iterator parity not complete (row counts/coords differ)"
GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"
GAP_ARRAY = "array-track gextract not supported"


# ── gscreen 1D scope, gscreen-result iterator ────────────────────────────────


def _screen_1d():
    i1 = pm.gscreen("test.fixedbin > 0.2", pm.gintervals([1, 2]))
    i2 = pm.gscreen("test.fixedbin < 0.4", pm.gintervals([1, 2]))
    return i1, i2


def test_gscreen_1d_fixedbin():
    i1, i2 = _screen_1d()
    assert_matches_baseline(pm.gextract("test.fixedbin", i1, iterator=i2), "gextract.gscreen.1d.fixedbin")


def test_gscreen_1d_sparse():
    i1, i2 = _screen_1d()
    assert_matches_baseline(pm.gextract("test.sparse", i1, iterator=i2), "gextract.gscreen.1d.sparse")


def test_gscreen_1d_array():
    i1, i2 = _screen_1d()
    assert_matches_baseline(pm.gextract("test.array", i1, iterator=i2), "gextract.gscreen.1d.array")


# ── gscreen 2D scope, gscreen-result iterator ────────────────────────────────


def _screen_2d():
    scope = pm.gintervals_2d([2, 3], chroms2=[2, 4])
    i1 = pm.gscreen("test.rects > 40", scope)
    i2 = pm.gscreen("test.rects < 50", scope)
    return i1, i2


def test_gscreen_2d_rects():
    i1, i2 = _screen_2d()
    assert_matches_baseline(pm.gextract("test.rects", i1, iterator=i2), "gextract.gscreen.2d.rects")


@pytest.mark.xfail(reason=GAP_COMPUTED, strict=True)
def test_gscreen_2d_computed2d():
    i1, i2 = _screen_2d()
    assert_matches_baseline(pm.gextract("test.computed2d", i1, iterator=i2), "gextract.gscreen.2d.computed2d")


# ── 2D ALLGENOME with explicit 2D iterator DataFrame ─────────────────────────


def test_2d_allgenome_rects():
    iv = pd.concat([
        pm.gintervals_2d(1, 10, 100, 1, 10, 100),
        pm.gintervals_2d(1, 400, 500, 1, 400, 500),
        pm.gintervals_2d(2, 600, 700, 2, 600, 700),
        pm.gintervals_2d(1, 200, 300, 2, 200, 300),
        pm.gintervals_2d(1, 7000, 9100, "X", 7000, 9100),
        pm.gintervals_2d(2, 9000, 18000, 2, 9000, 18000),
        pm.gintervals_2d(1, 30000, 31000, 1, 30000, 31000),
        pm.gintervals_2d(2, 1130, 15000, 1, 1130, 15000),
        pm.gintervals_2d(1, 1100, 1120, 1, 1100, 1120),
        pm.gintervals_2d(1, 1000, 1100, 2, 1000, 1100),
    ], ignore_index=True)
    assert_matches_baseline(
        pm.gextract("test.rects", pm.gintervals_2d_all(mode="full"), iterator=iv),
        "gextract.2d.ALLGENOME.rects",
    )


# ── giterator.intervals: track / intervals-set names as scope and iterator ───
# 1D (1-9) and the 2D string-track cases 15, 18 match R; the rest of the 2D
# combinations still differ (GAP_2D_ITER).

def _gi(name):
    return pm.giterator_intervals(name)


_GITER_CASES = {
    "1": lambda: pm.gextract("test.generated_1d_1", _gi("test.generated_1d_2"), iterator=_gi("test.generated_1d_1")),
    "2": lambda: pm.gextract("test.generated_1d_1", _gi("test.generated_1d_2"), iterator="test.bigintervs_1d_1"),
    "3": lambda: pm.gextract("test.generated_1d_1", _gi("test.generated_1d_2"), iterator="test.generated_1d_1"),
    "4": lambda: pm.gextract("test.generated_1d_1", "test.bigintervs_1d_2", iterator=_gi("test.generated_1d_1")),
    "5": lambda: pm.gextract("test.generated_1d_1", "test.bigintervs_1d_2", iterator="test.bigintervs_1d_1"),
    "6": lambda: pm.gextract("test.generated_1d_1", "test.bigintervs_1d_2", iterator="test.generated_1d_1"),
    "7": lambda: pm.gextract("test.generated_1d_1", "test.generated_1d_2", iterator=_gi("test.generated_1d_1")),
    "8": lambda: pm.gextract("test.generated_1d_1", "test.generated_1d_2", iterator="test.bigintervs_1d_1"),
    "9": lambda: pm.gextract("test.generated_1d_1", "test.generated_1d_2", iterator="test.generated_1d_1"),
    "10": lambda: pm.gextract("test.generated_2d_5", _gi("test.generated_2d_6"), iterator=_gi("test.generated_2d_5")),
    "11": lambda: pm.gextract("test.generated_2d_5", _gi("test.generated_2d_6"), iterator="test.bigintervs_2d_5"),
    "12": lambda: pm.gextract("test.generated_2d_5", _gi("test.generated_2d_6"), iterator="test.generated_2d_5"),
    "13": lambda: pm.gextract("test.generated_2d_5", "test.bigintervs_2d_6", iterator=_gi("test.generated_2d_5")),
    "14": lambda: pm.gextract("test.generated_2d_5", "test.bigintervs_2d_6", iterator="test.bigintervs_2d_5"),
    "15": lambda: pm.gextract("test.generated_2d_5", "test.bigintervs_2d_6", iterator="test.generated_2d_5"),
    "16": lambda: pm.gextract("test.generated_2d_5", "test.generated_2d_6", iterator=_gi("test.generated_2d_5")),
    "17": lambda: pm.gextract("test.generated_2d_5", "test.generated_2d_6", iterator="test.bigintervs_2d_5"),
    "18": lambda: pm.gextract("test.generated_2d_5", "test.generated_2d_6", iterator="test.generated_2d_5"),
}
_GITER_XFAIL = {"10", "11", "12", "13", "14", "16", "17"}


@pytest.mark.parametrize("n", list(_GITER_CASES))
def test_giterator_intervals(n, request):
    if n in _GITER_XFAIL:
        request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    assert_matches_baseline(_GITER_CASES[n](), f"gextract.giterator.intervals.{n}")
