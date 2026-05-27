"""Parity port of R misha ``test-gintervals1.R`` (34 regressions).

Covers gintervals/gintervals.2d creation (with R vector recycling), 2d.all/all,
gintervals.chrom_sizes (per-chrom(-pair) counts), gintervals.diff, and
gintervals.load. ``intervals.set.out`` cases are reproduced in-memory where the
saved set equals the operation output; the rest are xfail (see reasons).
"""
from __future__ import annotations

import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY = "array-track extract not supported"
GAP_LOAD_TRACK = "gintervals_load does not accept a track name as the set"
GAP_SETOUT = "intervals.set.out roundtrip (R DB read-only; saved-set canonicalization)"
GAP_2D_ITER = "2D giterator/band-intersect parity not complete"
GAP_DIFF_OVERLAP = "gintervals_diff on overlapping (non-canonic) rbind input differs"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _scr(*a, **k):
    return pm.gscreen(*a, **k)


def _rb(*d):
    return pd.concat(d, ignore_index=True)


def _band1():
    iv = pm.giterator_intervals("test.rects_big_rects", _i2([2, 3]), iterator=(123450, 97891))
    return pm.gintervals_2d_band_intersect(iv, band=(-198743, 23456))


_CASES = {
    "gintervals.creation.1": (lambda: _i1([1, 2], [0, 50, 2000, 50, 10000, 1500], [100, 1300, 3000, 300, 11000, 2300]), None),
    "gintervals.creation.2": (lambda: _i1([1, 2], [0, 50, 2000, 50, 10000, 1500], [100, 1300, 3000, 300, 11000, 2300], -1), None),
    "gintervals.2d.creation.1": (lambda: _i2([1], [0, 50], [100, 200], [3], [0, 50], [400, 600]), None),
    "gintervals.2d.creation.2": (lambda: _i2([1, 2], [0, 1000, 2000, 50, 10000, 1500], [100, 1300, 3000, 300, 11000, 2300], [3, 4], [10, 1010, 2010, 60, 10010, 1510], [110, 1310, 3010, 310, 11010, 2310]), None),
    "gintervals.2d.all.1": (lambda: pm.gintervals_2d_all(mode="full"), None),
    "gintervals.all.1": (lambda: pm.gintervals_all(), None),
    "gintervals.2d.band_intersect.1": (_band1, None),
    "gintervals.2d.band_intersect.2": (_band1, None),
    "gintervals.chrom_sizes.1": (lambda: pm.gintervals_chrom_sizes("bigintervs1d"), None),
    "gintervals.chrom_sizes.2": (lambda: pm.gintervals_chrom_sizes("bigintervs2d"), None),
    "gintervals.chrom_sizes.3": (lambda: pm.gintervals_chrom_sizes("test.tss"), None),
    "gintervals.chrom_sizes.4": (lambda: pm.gintervals_chrom_sizes("test.array"), None),
    "gintervals.chrom_sizes.5": (lambda: pm.gintervals_chrom_sizes("test.sparse"), None),
    "gintervals.chrom_sizes.6": (lambda: pm.gintervals_chrom_sizes("test.rects"), None),
    "gintervals.diff.1": (lambda: pm.gintervals_diff(_scr("test.fixedbin > 0.2", _i1([1, 2], 0, -1)), _scr("test.fixedbin > 0.4", _i1([1, 2], 0, -1))), None),
    "gintervals.diff.2": (lambda: pm.gintervals_diff(_rb(_scr("test.fixedbin > 0.1 & test.fixedbin < 0.3", _i1([1, 2])), _scr("test.fixedbin > 0.2 & test.fixedbin < 0.4", _i1([1, 2]))), _scr("(test.fixedbin > 0.25 & test.fixedbin < 0.32) | test.fixedbin > 0.35", _i1([1, 2]))), None),
    "gintervals.diff.3": (lambda: pm.gintervals_diff(_scr("test.fixedbin > 0.2", _i1([1, 2, 4, 8, 9], 0, -1)), _scr("test.fixedbin > 0.4", _i1([1, 2, 4, 7, 9], 0, -1))), None),
    "gintervals.diff.binintervs": (lambda: pm.gintervals_diff("test.bigintervs_1d_1", "test.bigintervs_1d_2"), None),
    "gintervals.load.1": (lambda: pm.gintervals_load("test.foodgene"), None),
    "gintervals.load.2": (lambda: pm.gintervals_load("bigintervs1d"), None),
    "gintervals.load.3": (lambda: pm.gintervals_load("bigintervs1d", chrom=2), None),
    "gintervals.load.4": (lambda: pm.gintervals_load("bigintervs2d"), None),
    "gintervals.load.5": (lambda: pm.gintervals_load("bigintervs2d", chrom1=2, chrom2=2), None),
    "gintervals.load.6": (lambda: pm.gintervals_load("test.rects", chrom1=1, chrom2=2), GAP_LOAD_TRACK),
    "gintervals.load.7": (lambda: pm.gintervals_load("test.generated_1d_1", chrom=13), GAP_LOAD_TRACK),
    "gintervals.load.8": (lambda: pm.gintervals_load("test.generated_1d_1", chrom=12), GAP_LOAD_TRACK),
    "gintervals.load.9": (lambda: pm.gintervals_load("test.generated_2d_5", chrom1=1, chrom2=2), GAP_LOAD_TRACK),
    "gintervals.load.10": (lambda: pm.gintervals_load("test.generated_2d_5", chrom1=1, chrom2=3), GAP_LOAD_TRACK),
    "gintervals.1": (lambda: pm.gintervals_canonic(pm.gextract("test.fixedbin", _i1([1, 2]))[["chrom", "start", "end"]]), GAP_SETOUT),
    "gintervals.2": (lambda: pm.gintervals_canonic(pm.gextract("test.rects", _i2([1, 2]))), GAP_SETOUT),
    "gintervals.3": (lambda: None, GAP_SETOUT),
    "gintervals.4": (lambda: None, GAP_SETOUT),
    "gintervals.5": (lambda: _scr("2 * test.sparse+0.2 > 0.4"), None),
    "gintervals.6": (lambda: _scr("test.rects > 40"), None),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gintervals1(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)
