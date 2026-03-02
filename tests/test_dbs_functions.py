"""Tests for gtrack_dbs and gintervals_dbs."""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestGtrackDbs:
    def test_single_track_found(self):
        """Known track should be found in the current DB root."""
        result = pm.gtrack_dbs("dense_track")
        assert isinstance(result, dict)
        assert "dense_track" in result
        assert len(result["dense_track"]) == 1

    def test_single_track_not_found(self):
        """Non-existent track should have empty list."""
        result = pm.gtrack_dbs("nonexistent_track_xyz")
        assert result["nonexistent_track_xyz"] == []

    def test_multiple_tracks(self):
        """Multiple tracks can be queried at once."""
        result = pm.gtrack_dbs(["dense_track", "sparse_track"])
        assert "dense_track" in result
        assert "sparse_track" in result
        assert len(result["dense_track"]) >= 1
        assert len(result["sparse_track"]) >= 1

    def test_dataframe_output(self):
        """dataframe=True returns a DataFrame."""
        result = pm.gtrack_dbs("dense_track", dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert "track" in result.columns
        assert "db" in result.columns
        assert len(result) >= 1
        assert result["track"].iloc[0] == "dense_track"

    def test_dataframe_multiple(self):
        """dataframe=True with multiple tracks."""
        result = pm.gtrack_dbs(["dense_track", "sparse_track"], dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 2

    def test_dataframe_not_found(self):
        """dataframe=True for non-existent track returns empty DataFrame."""
        result = pm.gtrack_dbs("nonexistent_xyz", dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestGintervalsDbs:
    def test_single_interval_set_found(self):
        """Known interval set should be found."""
        # Check what interval sets exist
        ls = pm.gintervals_ls()
        if not ls:
            pytest.skip("No interval sets in test DB")
        name = ls[0]
        result = pm.gintervals_dbs(name)
        assert isinstance(result, dict)
        assert name in result
        assert len(result[name]) >= 1

    def test_not_found(self):
        """Non-existent interval set should have empty list."""
        result = pm.gintervals_dbs("nonexistent_intervals_xyz")
        assert result["nonexistent_intervals_xyz"] == []

    def test_dataframe_output(self):
        """dataframe=True returns a DataFrame."""
        ls = pm.gintervals_ls()
        if not ls:
            pytest.skip("No interval sets in test DB")
        name = ls[0]
        result = pm.gintervals_dbs(name, dataframe=True)
        assert isinstance(result, pd.DataFrame)
        assert "intervals" in result.columns
        assert "db" in result.columns

    def test_multiple(self):
        """Multiple interval set names can be queried."""
        ls = pm.gintervals_ls()
        if not ls or len(ls) < 2:
            pytest.skip("Need at least 2 interval sets")
        result = pm.gintervals_dbs(ls[:2])
        assert len(result) == 2
