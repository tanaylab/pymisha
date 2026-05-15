"""Tests for the C++ 2D-intervals iterator (foundation of K.1+K.4 scanner port).

These tests drive ``_pymisha.pm_test_2d_iterator``, a test-only binding that
constructs a ``PMTrackExpressionIntervals2DIterator`` over the given
intervals and emits them back. They are a unit-test for the iterator state
machine before the scanner is wired up.
"""

import _pymisha
import numpy as np
import pytest


def _make_intervals(rows):
    """rows: list of ``(chromid1, start1, end1, chromid2, start2, end2)`` tuples.

    Returns a dict of 6 numpy arrays in the layout ``pm_test_2d_iterator``
    expects.
    """
    return {
        "chrom1": np.array([r[0] for r in rows], dtype=np.int32),
        "start1": np.array([r[1] for r in rows], dtype=np.int64),
        "end1":   np.array([r[2] for r in rows], dtype=np.int64),
        "chrom2": np.array([r[3] for r in rows], dtype=np.int32),
        "start2": np.array([r[4] for r in rows], dtype=np.int64),
        "end2":   np.array([r[5] for r in rows], dtype=np.int64),
    }


def test_iterator_emits_input_in_order():
    rows = [
        (0, 100, 200, 0, 300, 400),
        (0, 500, 600, 1, 700, 800),
        (2, 1000, 1100, 2, 1200, 1300),
    ]
    intervals = _make_intervals(rows)
    out = _pymisha.pm_test_2d_iterator(intervals)
    assert out["chrom1"].tolist() == [0, 0, 2]
    assert out["start1"].tolist() == [100, 500, 1000]
    assert out["end1"].tolist() == [200, 600, 1100]
    assert out["chrom2"].tolist() == [0, 1, 2]
    assert out["start2"].tolist() == [300, 700, 1200]
    assert out["end2"].tolist() == [400, 800, 1300]
    assert out["interval_id"].tolist() == [1, 2, 3]


def test_iterator_empty_input():
    intervals = _make_intervals([])
    out = _pymisha.pm_test_2d_iterator(intervals)
    assert len(out["chrom1"]) == 0
    assert len(out["interval_id"]) == 0


def test_iterator_single_interval():
    rows = [(0, 0, 100, 0, 0, 100)]
    intervals = _make_intervals(rows)
    out = _pymisha.pm_test_2d_iterator(intervals)
    assert out["chrom1"].tolist() == [0]
    assert out["interval_id"].tolist() == [1]


def test_iterator_preserves_dup_intervals():
    """Two identical 2D intervals must both be emitted (no dedup)."""
    rows = [
        (1, 100, 200, 1, 300, 400),
        (1, 100, 200, 1, 300, 400),
    ]
    intervals = _make_intervals(rows)
    out = _pymisha.pm_test_2d_iterator(intervals)
    assert len(out["chrom1"]) == 2
    assert out["interval_id"].tolist() == [1, 2]


def test_iterator_mixed_dtypes_int32_int64():
    """Caller may pass int32 arrays for chromid; iterator must still work."""
    rows = [(0, 100, 200, 1, 300, 400)]
    intervals = _make_intervals(rows)
    assert intervals["chrom1"].dtype == np.int32
    assert intervals["start1"].dtype == np.int64
    out = _pymisha.pm_test_2d_iterator(intervals)
    assert out["start1"].tolist() == [100]
    assert out["end1"].tolist() == [200]


def test_iterator_rejects_mismatched_array_lengths():
    intervals = {
        "chrom1": np.array([0, 0], dtype=np.int32),
        "start1": np.array([100], dtype=np.int64),       # wrong length
        "end1":   np.array([200, 300], dtype=np.int64),
        "chrom2": np.array([0, 0], dtype=np.int32),
        "start2": np.array([400, 500], dtype=np.int64),
        "end2":   np.array([600, 700], dtype=np.int64),
    }
    with pytest.raises((ValueError, RuntimeError)):
        _pymisha.pm_test_2d_iterator(intervals)


def test_iterator_rejects_missing_key():
    intervals = _make_intervals([(0, 100, 200, 0, 300, 400)])
    del intervals["end2"]
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        _pymisha.pm_test_2d_iterator(intervals)
