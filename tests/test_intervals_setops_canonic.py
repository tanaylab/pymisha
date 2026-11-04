"""Set operations canonicalize both inputs first, as R's gintervunion /
gintervintersect / gintervdiff do, so row order and overlaps don't change the answer."""

import pandas as pd
import pytest

import pymisha as pm


def _iv(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def _rows(df):
    return [tuple(r) for r in df[["chrom", "start", "end"]].astype({"chrom": str}).itertuples(index=False)]


A = _iv([("chr1", 0, 100), ("chr1", 0, 100)])


@pytest.mark.parametrize("b", [
    _iv([("chr1", 0, 50), ("chr1", 0, 100)]),
    _iv([("chr1", 0, 100), ("chr1", 0, 50)]),
])
def test_intersect_ignores_row_order_and_duplicates(b):
    assert _rows(pm.gintervals_intersect(A, b)) == [("chr1", 0, 100)]


def test_intersect_unifies_overlapping_rows():
    a = _iv([("chr1", 0, 100), ("chr1", 50, 150)])
    assert _rows(pm.gintervals_intersect(a, _iv([("chr1", 0, 200)]))) == [("chr1", 0, 150)]


def test_union_with_empty_side_is_canonic():
    a = _iv([("chr1", 50, 150), ("chr1", 0, 100)])
    assert _rows(pm.gintervals_union(a, _iv([]))) == [("chr1", 0, 150)]
