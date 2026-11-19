"""The C++ entry points hand back a result the caller owns exactly once.

pm_iterate and pm_lookup took an extra reference to their result, so it was never freed:
_iterated_intervals (the iterator of every virtual-track expression that cannot be computed
in C++) and glookup kept ~80 MB per million rows for the life of the process. An empty result
is None, which the caller must own as well.
"""

import sys

import _pymisha
import numpy as np
import pytest

import pymisha as pm
from pymisha import _shared


def test_pm_iterate_result_has_no_extra_reference():
    intervals = _shared._df2pymisha(pm.gintervals(["1", "2"], 0, 1000))
    res = _pymisha.pm_iterate(intervals, 100, _shared.CONFIG)
    assert len(res[2]) == 20  # start (chrom is [categories, codes])
    control = []  # a local as res is, so interpreter-level borrowing affects both alike
    assert sys.getrefcount(res) == sys.getrefcount(control)


def test_pm_lookup_result_has_no_extra_reference():
    intervals = _shared._df2pymisha(pm.gintervals("1", 0, 1000))
    res = _pymisha.pm_lookup(
        ["dense_track"], [np.linspace(0, 1, 6).tolist()], np.arange(5, dtype=float),
        intervals, 100, False, True, _shared.CONFIG,
    )
    assert len(res[1]) == 10
    control = []  # a local as res is, so interpreter-level borrowing affects both alike
    assert sys.getrefcount(res) == sys.getrefcount(control)


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="None is immortal from Python 3.12")
def test_pm_iterate_empty_result_owns_its_none():
    empty = _shared._df2pymisha(pm.gintervals("1", 0, 100).iloc[:0])
    assert _pymisha.pm_iterate(empty, 10, _shared.CONFIG) is None
    before = sys.getrefcount(None)
    for _ in range(1000):
        _pymisha.pm_iterate(empty, 10, _shared.CONFIG)
    assert abs(sys.getrefcount(None) - before) < 100
