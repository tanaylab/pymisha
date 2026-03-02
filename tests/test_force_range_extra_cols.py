"""Tests for gintervals_force_range preserving extra columns."""

import pandas as pd
import pytest

import pymisha


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    """Ensure test DB is initialized."""


class TestForceRangeExtraColumns1D:
    """1D intervals: extra columns must survive clamping and filtering."""

    def test_extra_columns_preserved(self):
        """Extra columns like 'score' and 'name' survive force_range."""
        intervs = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "start": [100, -50, 1000],
            "end": [200, 300, 2000],
            "score": [1.5, 2.5, 3.5],
            "name": ["a", "b", "c"],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert "score" in result.columns
        assert "name" in result.columns
        assert len(result) == 3
        # First row: untouched
        assert result.iloc[0]["score"] == 1.5
        assert result.iloc[0]["name"] == "a"
        # Second row: start clamped to 0
        assert result.iloc[1]["start"] == 0
        assert result.iloc[1]["score"] == 2.5
        assert result.iloc[1]["name"] == "b"
        # Third row: untouched
        assert result.iloc[2]["score"] == 3.5
        assert result.iloc[2]["name"] == "c"

    def test_extra_columns_after_filtering(self):
        """Rows removed by force_range don't corrupt extra column alignment."""
        intervs = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [100, 600000, 200],
            "end": [200, 700000, 300],
            "tag": ["keep1", "drop", "keep2"],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert len(result) == 2
        assert list(result["tag"]) == ["keep1", "keep2"]

    def test_end_clamped_preserves_extras(self):
        """End clamped to chrom size; extra column values preserved."""
        intervs = pd.DataFrame({
            "chrom": ["1"],
            "start": [400000],
            "end": [999999],
            "value": [42],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert result.iloc[0]["end"] == 500000
        assert result.iloc[0]["value"] == 42

    def test_no_extra_columns_still_works(self):
        """Backward compat: no extra columns is fine."""
        intervs = pd.DataFrame({
            "chrom": ["1"],
            "start": [100],
            "end": [200],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert list(result.columns) == ["chrom", "start", "end"]

    def test_index_is_reset(self):
        """Result index is 0-based regardless of which rows survived."""
        intervs = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [-1, 600000, 100],
            "end": [50, 700000, 200],
            "x": [10, 20, 30],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert list(result.index) == [0, 1]
        assert list(result["x"]) == [10, 30]


class TestForceRangeExtraColumns2D:
    """2D intervals: extra columns must survive clamping and filtering."""

    def test_extra_columns_preserved_2d(self):
        """Extra columns survive 2D force_range."""
        intervs = pd.DataFrame({
            "chrom1": ["1", "1"],
            "start1": [-10, 100],
            "end1": [500, 200],
            "chrom2": ["2", "2"],
            "start2": [100, -50],
            "end2": [200, 300],
            "score": [1.1, 2.2],
            "label": ["alpha", "beta"],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert "score" in result.columns
        assert "label" in result.columns
        assert len(result) == 2
        # First row: start1 clamped
        assert result.iloc[0]["start1"] == 0
        assert result.iloc[0]["score"] == 1.1
        assert result.iloc[0]["label"] == "alpha"
        # Second row: start2 clamped
        assert result.iloc[1]["start2"] == 0
        assert result.iloc[1]["score"] == 2.2
        assert result.iloc[1]["label"] == "beta"

    def test_2d_filtering_preserves_extras(self):
        """Rows dropped in 2D don't break extra column alignment."""
        intervs = pd.DataFrame({
            "chrom1": ["1", "1", "1"],
            "start1": [100, 600000, 200],
            "end1": [200, 700000, 300],
            "chrom2": ["2", "2", "2"],
            "start2": [100, 100, 200],
            "end2": [200, 200, 300],
            "info": ["first", "gone", "third"],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert len(result) == 2
        assert list(result["info"]) == ["first", "third"]
        assert list(result.index) == [0, 1]

    def test_no_extra_columns_2d_still_works(self):
        """Backward compat: 2D without extra columns is fine."""
        intervs = pd.DataFrame({
            "chrom1": ["1"],
            "start1": [100],
            "end1": [200],
            "chrom2": ["2"],
            "start2": [100],
            "end2": [200],
        })
        result = pymisha.gintervals_force_range(intervs)
        assert result is not None
        assert list(result.columns) == [
            "chrom1", "start1", "end1", "chrom2", "start2", "end2",
        ]
