"""Tests for intervals_set_out parameter on functions that gained it in GAP-042."""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def _cleanup_iset(name):
    """Remove an interval set if it exists."""
    if pm.gintervals_exists(name):
        pm.gintervals_rm(name, force=True)


class TestGscreenIntervalsSetOut:
    def test_saves_result(self):
        name = "test.gscreen_out"
        try:
            result = pm.gscreen(
                "dense_track > 0.2",
                intervals=pm.gintervals("1", 0, 10000),
                intervals_set_out=name,
                progress=False,
            )
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) > 0
            assert set(loaded.columns) >= {"chrom", "start", "end"}
        finally:
            _cleanup_iset(name)


class TestGpartitionIntervalsSetOut:
    def test_saves_result(self):
        name = "test.gpartition_out"
        try:
            result = pm.gpartition(
                "dense_track",
                [0, 0.05, 0.1, 0.15, 0.2],
                intervals=pm.gintervals("1", 0, 10000),
                intervals_set_out=name,
            )
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) > 0
        finally:
            _cleanup_iset(name)


class TestGlookupIntervalsSetOut:
    def test_saves_result(self):
        name = "test.glookup_out"
        try:
            result = pm.glookup(
                [10, 20, 30, 40, 50],
                "dense_track",
                [0.1, 0.12, 0.14, 0.16, 0.18, 0.2],
                intervals=pm.gintervals("1", 0, 5000),
                intervals_set_out=name,
            )
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) > 0
        finally:
            _cleanup_iset(name)


class TestGintervalsForceRangeIntervalsSetOut:
    def test_saves_result(self):
        name = "test.force_range_out"
        try:
            intervs = pd.DataFrame({
                "chrom": ["1", "1"],
                "start": [-100, 10000],
                "end": [200, 1300000],
            })
            result = pm.gintervals_force_range(intervs, intervals_set_out=name)
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == 2
        finally:
            _cleanup_iset(name)


class TestGintervalsUnionIntervalsSetOut:
    def test_saves_result(self):
        name = "test.union_out"
        try:
            i1 = pm.gintervals("1", [0, 500], [300, 800])
            i2 = pm.gintervals("1", [200, 700], [400, 900])
            result = pm.gintervals_union(i1, i2, intervals_set_out=name)
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) >= 1
        finally:
            _cleanup_iset(name)


class TestGintervalsIntersectIntervalsSetOut:
    def test_saves_result(self):
        name = "test.intersect_out"
        try:
            i1 = pm.gintervals("1", 0, 500)
            i2 = pm.gintervals("1", 300, 800)
            result = pm.gintervals_intersect(i1, i2, intervals_set_out=name)
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == 1
        finally:
            _cleanup_iset(name)


class TestGintervalsDiffIntervalsSetOut:
    def test_saves_result(self):
        name = "test.diff_out"
        try:
            i1 = pm.gintervals("1", 0, 500)
            i2 = pm.gintervals("1", 200, 300)
            result = pm.gintervals_diff(i1, i2, intervals_set_out=name)
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) >= 1
        finally:
            _cleanup_iset(name)


class TestGintervalsNormalizeIntervalsSetOut:
    def test_saves_result(self):
        name = "test.normalize_out"
        try:
            intervs = pm.gintervals("1", [1000, 5000], [2000, 6000])
            result = pm.gintervals_normalize(intervs, 500, intervals_set_out=name)
            assert result is None
            loaded = pm.gintervals_load(name)
            assert isinstance(loaded, pd.DataFrame)
            assert len(loaded) == 2
        finally:
            _cleanup_iset(name)
