"""Parity port of R misha ``test-gpartition.R`` (5 regressions).

The fixedbin partition is order-independent, so R's seeded row-shuffle of the
input is not reproduced. gpartition does not support 2D intervals; the two
data-size / intervals.set.out cases roundtrip through gintervals.save and need
the writable overlay DB.
"""
from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D = "gpartition does not support 2D intervals"
GAP_COMPUTED = "COMPUTED Hi-C tracks not supported"


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


def _i2(*a, **k):
    return pm.gintervals_2d(*a, **k)


def _seq(a, b, by):
    n = int(round((b - a) / by))
    return a + np.arange(n + 1) * by


_CASES = {
    "gpartition_fixedbin_sampling_result": (lambda: pm.gpartition("test.fixedbin", _seq(0, 1, 0.1), intervals=pm.gscreen("test.fixedbin > 0.14", _i1([1, 2], 0, -1))), None),
    "gpartition_rects_result": (lambda: pm.gpartition("test.rects", _seq(50, 100, 1), intervals=_i2(chroms1=[2, 3], chroms2=[2, 4])), GAP_2D),
    "gpartition_computed2d_result": (lambda: pm.gpartition("test.computed2d", _seq(5000000, 10000000, 1000000), intervals=_i2(chroms1=[6, 5], chroms2=[8, 9])), GAP_2D),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gpartition(bid, request):
    fn, reason = _CASES[bid]
    if reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    assert_matches_baseline(fn(), bid)


def test_gpartition_fixedbin_sampling_data_size(overlay_db, track_namer):
    """R: gpartition(test.fixedbin, seq(0,1,0.1), intervals, intervals.set.out=temp)."""
    name = track_namer("test.tmptrack")
    intervs = pm.gscreen("test.fixedbin > 0.14", _i1([1, 2], 0, -1))
    pm.gpartition(
        "test.fixedbin",
        _seq(0, 1, 0.1),
        intervals=intervs,
        intervals_set_out=name,
    )
    assert_matches_baseline(
        pm.gintervals_load(name), "gpartition_fixedbin_sampling_data_size_result"
    )


@pytest.mark.xfail(reason=GAP_2D, strict=True)
def test_gpartition_rects_data_size(overlay_db, track_namer):
    """R: gpartition(test.rects, seq(0,100,1), 2D intervals, intervals.set.out=temp).

    Same 2D-intervals limitation as ``gpartition_rects_result``: pm_partition's
    C++ entry rejects 2D scope. Re-tagged from GAP_DSIZE because the saved-set
    roundtrip itself works (KICKOFF-9 E); only the underlying 2D-gpartition
    call is missing.
    """
    name = track_namer("test.tmptrack")
    pm.gpartition(
        "test.rects",
        _seq(0, 100, 1),
        intervals=_i2(chroms1=[6, 3], chroms2=[2, 4]),
        intervals_set_out=name,
    )
    assert_matches_baseline(
        pm.gintervals_load(name), "gpartition_rects_data_size_result"
    )
