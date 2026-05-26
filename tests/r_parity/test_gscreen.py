"""Parity port of R misha ``test-gscreen.R`` (6 regressions; all pass)."""
from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def test_gscreen_fixedbin_1():
    assert_matches_baseline(pm.gscreen("2 * test.fixedbin+0.2 > 0.4"), "gscreen.fixedbin.1")


def test_gscreen_fixedbin_2():
    assert_matches_baseline(pm.gscreen("test.fixedbin < -1"), "gscreen.fixedbin.2")


def test_gscreen_fixedbin_3():
    assert_matches_baseline(
        pm.gscreen("test.fixedbin > 0.2", _i1(1, [0, 2000000, 4000000], [1000000, 3000000, 5000000])),
        "gscreen.fixedbin.3",
    )


def test_gscreen_rects_1():
    assert_matches_baseline(pm.gscreen("2 * test.rects+1 > 100"), "gscreen.rects.1")


def test_gscreen_rects_2():
    assert_matches_baseline(pm.gscreen("test.rects < -1"), "gscreen.rects.2")


def test_gscreen_rects_3():
    assert_matches_baseline(
        pm.gscreen("test.rects > 40", _i2(chroms1=[2, 3], chroms2=[2, 4])), "gscreen.rects.3"
    )
