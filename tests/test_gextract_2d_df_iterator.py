"""No-NFS regression for the 2D-intervals-DataFrame iterator in gextract.

Mirrors the R-parity cases (gscreen.2d.rects, 2d.ALLGENOME.rects, the
gintervals.summary 2D iterator cases) on the small bundled test DB: a 2D
intervals DataFrame passed as ``iterator=`` makes gextract iterate
``iterator ∩ scope`` rectangles, with ``intervalID`` attributing each row to
its scope interval.
"""
from __future__ import annotations

import pandas as pd
import pytest

import pymisha as pm

_COLS6 = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def _all_rects() -> pd.DataFrame:
    return pm.gextract("rects_track", pm.gintervals_2d_all(mode="full"))


class TestGextract2dDataFrameIterator:
    def test_iterator_is_intersection_of_scope_and_iterator(self):
        from pymisha.intervals import _intersect_2d_rects, _sort_2d_intervals

        full = _all_rects()
        scope = full.iloc[:1000][_COLS6].reset_index(drop=True)
        iterator = full.iloc[600:][_COLS6].reset_index(drop=True)

        res = pm.gextract("rects_track", scope, iterator=iterator)
        units = _sort_2d_intervals(_intersect_2d_rects(iterator, scope)).reset_index(drop=True)

        got = res[_COLS6].sort_values(_COLS6).reset_index(drop=True)
        exp = units[_COLS6].sort_values(_COLS6).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, exp, check_dtype=False)
        assert res["rects_track"].notna().all()

    def test_iterator_ignored_no_more(self):
        # A DataFrame iterator must restrict the output (was previously ignored,
        # producing the whole object-enumeration of the scope).
        full = _all_rects()
        scope = pm.gintervals_2d_all(mode="full")
        iterator = full.iloc[:20][_COLS6].reset_index(drop=True)

        res = pm.gextract("rects_track", scope, iterator=iterator)
        # Whole-genome scope clips each iterator rect to itself -> one row each.
        assert len(res) == len(iterator)

    def test_intervalID_attributes_to_scope_interval(self):
        # intervalID must be the (1-based) scope-interval index, so callers like
        # gintervals_summary can group per scope interval.
        full = _all_rects()
        scope = full.iloc[:1000][_COLS6].reset_index(drop=True)
        iterator = full.iloc[600:][_COLS6].reset_index(drop=True)

        res = pm.gextract("rects_track", scope, iterator=iterator)
        ids = res["intervalID"].to_numpy()
        assert ids.min() >= 1
        assert ids.max() <= len(scope)

    def test_empty_intersection_returns_none(self):
        full = _all_rects()
        # Two disjoint halves of the rect set with no shared rectangles: the
        # first 50 and a synthetic rect on a non-covered region -- still
        # within chrom "1"'s 500000bp bounds (coordinates must validate),
        # just far from any rects_track object.
        scope = full.iloc[:50][_COLS6].reset_index(drop=True)
        iterator = pd.DataFrame(
            [("1", 499_000, 499_001, "1", 499_000, 499_001)], columns=_COLS6
        )
        res = pm.gextract("rects_track", scope, iterator=iterator)
        assert res is None or len(res) == 0


class TestGextract2dIntervalSetNameIterator:
    def test_interval_set_name_iterator_matches_dataframe(self):
        # A 2D interval-set *name* used as the iterator must route through the
        # same scalable intersect as the equivalent DataFrame.
        import contextlib

        full = _all_rects()
        scope = full.iloc[:1000][_COLS6].reset_index(drop=True)
        iterator_df = full.iloc[600:][_COLS6].reset_index(drop=True)

        set_name = "test.tmp_2d_iter_set"
        with contextlib.suppress(Exception):
            pm.gintervals_rm(set_name, force=True)
        pm.gintervals_save(iterator_df, set_name)
        try:
            by_name = pm.gextract("rects_track", scope, iterator=set_name)
            by_df = pm.gextract("rects_track", scope, iterator=iterator_df)
        finally:
            with contextlib.suppress(Exception):
                pm.gintervals_rm(set_name, force=True)

        a = by_name[_COLS6].sort_values(_COLS6).reset_index(drop=True)
        b = by_df[_COLS6].sort_values(_COLS6).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


class TestGintervalsSummary2dDataFrameIterator:
    def test_summary_one_row_per_scope_interval(self):
        full = _all_rects()
        scope = full.iloc[:1000][_COLS6].reset_index(drop=True)
        iterator = full.iloc[600:][_COLS6].reset_index(drop=True)

        summ = pm.gintervals_summary("rects_track", scope, iterator=iterator)
        # One row per scope interval (empties get Total intervals = 0).
        assert len(summ) == len(scope)
        assert {"Total intervals", "Min", "Max", "Sum", "Mean"}.issubset(summ.columns)

    def test_summary_totals_match_extract_grouping(self):
        # gintervals_summary's "Total intervals" per scope interval must equal
        # the number of gextract DataFrame-iterator rows attributed to it.
        full = _all_rects()
        scope = full.iloc[:1000][_COLS6].reset_index(drop=True)
        iterator = full.iloc[600:][_COLS6].reset_index(drop=True)

        ext = pm.gextract("rects_track", scope, iterator=iterator)
        summ = pm.gintervals_summary("rects_track", scope, iterator=iterator)

        # gextract intervalID is the 1-based scope index; summary rows are in
        # scope order (0-based row i == scope interval i+1).
        counts_by_id = ext.groupby("intervalID").size()
        expected_total = summ["Total intervals"].to_numpy()
        got_total = pd.Series(0, index=range(1, len(scope) + 1), dtype=int)
        got_total.loc[counts_by_id.index] = counts_by_id.to_numpy()
        assert list(got_total.to_numpy()) == list(expected_total.astype(int))
