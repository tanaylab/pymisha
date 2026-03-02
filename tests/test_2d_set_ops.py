"""Tests for 2D interval set operations: gintervals_2d_intersect & gintervals_2d_union."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    """Ensure the test DB is initialised for every test."""
    pass


def _make_2d(chrom1, start1, end1, chrom2, start2, end2):
    """Helper to create a 2D intervals DataFrame from scalar or list args."""
    if isinstance(chrom1, str):
        chrom1, start1, end1 = [chrom1], [start1], [end1]
        chrom2, start2, end2 = [chrom2], [start2], [end2]
    return pd.DataFrame({
        'chrom1': chrom1,
        'start1': start1,
        'end1': end1,
        'chrom2': chrom2,
        'start2': start2,
        'end2': end2,
    })


# ──────────────────────────────────────────────────────────────────
# gintervals_2d_intersect
# ──────────────────────────────────────────────────────────────────

class TestIntersect2D:
    """Tests for gintervals_2d_intersect."""

    def test_basic_overlap(self):
        """Two overlapping rectangles produce one intersection."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = _make_2d("chr1", 500, 1500, "chr1", 500, 1500)
        result = pm.gintervals_2d_intersect(iv1, iv2)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['start1'] == 500
        assert result.iloc[0]['end1'] == 1000
        assert result.iloc[0]['start2'] == 500
        assert result.iloc[0]['end2'] == 1000

    def test_no_overlap(self):
        """Two non-overlapping rectangles yield None."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 200, 300, "chr1", 200, 300)
        result = pm.gintervals_2d_intersect(iv1, iv2)
        assert result is None

    def test_no_overlap_one_dim(self):
        """Overlap in dim1 but not dim2 yields None."""
        iv1 = _make_2d("chr1", 0, 500, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 100, 400, "chr1", 200, 300)
        result = pm.gintervals_2d_intersect(iv1, iv2)
        assert result is None

    def test_different_chroms(self):
        """Different chromosome pairs yield None."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = _make_2d("chr2", 0, 1000, "chr2", 0, 1000)
        result = pm.gintervals_2d_intersect(iv1, iv2)
        assert result is None

    def test_contained(self):
        """Inner rectangle is fully contained in outer."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = _make_2d("chr1", 200, 500, "chr1", 300, 700)
        result = pm.gintervals_2d_intersect(iv1, iv2)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['start1'] == 200
        assert result.iloc[0]['end1'] == 500
        assert result.iloc[0]['start2'] == 300
        assert result.iloc[0]['end2'] == 700

    def test_multiple_intersections(self):
        """One rectangle from set1 can intersect multiple from set2."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = _make_2d(
            ["chr1", "chr1"],
            [100, 500],
            [300, 800],
            ["chr1", "chr1"],
            [100, 500],
            [300, 800],
        )
        result = pm.gintervals_2d_intersect(iv1, iv2)

        assert result is not None
        assert len(result) == 2

    def test_pairwise_many_to_many(self):
        """2x2 => up to 4 intersections; verify count and coordinates."""
        iv1 = _make_2d(
            ["chr1", "chr1"],
            [0, 400],
            [600, 1000],
            ["chr1", "chr1"],
            [0, 400],
            [600, 1000],
        )
        iv2 = _make_2d(
            ["chr1", "chr1"],
            [200, 700],
            [800, 1200],
            ["chr1", "chr1"],
            [200, 700],
            [800, 1200],
        )
        result = pm.gintervals_2d_intersect(iv1, iv2)

        assert result is not None
        # (0,600) x (200,800) => (200,600) x (200,600)  valid
        # (0,600) x (700,1200) => (0,600) x (700,1200) => start1<end1 and start2>end2? no: (200,600) vs (700,800)
        # Let me recalculate:
        # iv1[0]: s1=0,e1=600, s2=0,e2=600
        # iv1[1]: s1=400,e1=1000, s2=400,e2=1000
        # iv2[0]: s1=200,e1=800, s2=200,e2=800
        # iv2[1]: s1=700,e1=1200, s2=700,e2=1200
        #
        # iv1[0] x iv2[0]: max(0,200)=200, min(600,800)=600, max(0,200)=200, min(600,800)=600 => valid
        # iv1[0] x iv2[1]: max(0,700)=700, min(600,1200)=600 => 700<600 = False => skip
        # iv1[1] x iv2[0]: max(400,200)=400, min(1000,800)=800, max(400,200)=400, min(1000,800)=800 => valid
        # iv1[1] x iv2[1]: max(400,700)=700, min(1000,1200)=1000, max(400,700)=700, min(1000,1200)=1000 => valid
        assert len(result) == 3

    def test_trans_chrom_pairs(self):
        """Intervals on different chrom pairs only match same pairs."""
        iv1 = _make_2d(
            ["chr1", "chr1"],
            [0, 0],
            [1000, 1000],
            ["chr1", "chr2"],
            [0, 0],
            [1000, 1000],
        )
        iv2 = _make_2d(
            ["chr1", "chr1"],
            [500, 500],
            [1500, 1500],
            ["chr1", "chr2"],
            [500, 500],
            [1500, 1500],
        )
        result = pm.gintervals_2d_intersect(iv1, iv2)

        assert result is not None
        assert len(result) == 2
        # Both chrom pairs should be present
        pairs = set(zip(result['chrom1'], result['chrom2']))
        assert pairs == {('chr1', 'chr1'), ('chr1', 'chr2')}

    def test_empty_input(self):
        """Empty input returns None."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        assert pm.gintervals_2d_intersect(iv1, iv2) is None
        assert pm.gintervals_2d_intersect(iv2, iv1) is None

    def test_none_input_raises(self):
        """None input raises ValueError."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        with pytest.raises(ValueError):
            pm.gintervals_2d_intersect(None, iv1)
        with pytest.raises(ValueError):
            pm.gintervals_2d_intersect(iv1, None)

    def test_missing_columns_raises(self):
        """Input missing required columns raises ValueError."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        bad = pd.DataFrame({'chrom1': ['chr1'], 'start1': [0], 'end1': [100]})
        with pytest.raises(ValueError, match="missing required 2D columns"):
            pm.gintervals_2d_intersect(iv1, bad)

    def test_result_sorted(self):
        """Result is sorted by (chrom1, start1, chrom2, start2)."""
        iv1 = _make_2d(
            ["chr2", "chr1"],
            [0, 0],
            [1000, 1000],
            ["chr2", "chr1"],
            [0, 0],
            [1000, 1000],
        )
        iv2 = _make_2d(
            ["chr1", "chr2"],
            [500, 500],
            [1500, 1500],
            ["chr1", "chr2"],
            [500, 500],
            [1500, 1500],
        )
        result = pm.gintervals_2d_intersect(iv1, iv2)
        assert result is not None
        assert result.iloc[0]['chrom1'] == 'chr1'
        assert result.iloc[1]['chrom1'] == 'chr2'

    def test_touching_not_intersecting(self):
        """Touching rectangles (end == start) do not produce an intersection."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 100, 200, "chr1", 100, 200)
        result = pm.gintervals_2d_intersect(iv1, iv2)
        assert result is None

    def test_result_columns(self):
        """Result has exactly the standard 2D columns."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = _make_2d("chr1", 500, 1500, "chr1", 500, 1500)
        result = pm.gintervals_2d_intersect(iv1, iv2)
        assert list(result.columns) == ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

    def test_with_gintervals_2d(self):
        """Works with output from gintervals_2d()."""
        iv1 = pm.gintervals_2d("1", 0, 100000, "1", 0, 100000)
        iv2 = pm.gintervals_2d("1", 50000, 200000, "1", 50000, 200000)
        result = pm.gintervals_2d_intersect(iv1, iv2)

        assert result is not None
        assert len(result) == 1
        row = result.iloc[0]
        assert row['start1'] == 50000
        assert row['end1'] == 100000
        assert row['start2'] == 50000
        assert row['end2'] == 100000


# ──────────────────────────────────────────────────────────────────
# gintervals_2d_union
# ──────────────────────────────────────────────────────────────────

class TestUnion2D:
    """Tests for gintervals_2d_union."""

    def test_basic_union(self):
        """Union of two non-overlapping sets concatenates them."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 200, 300, "chr1", 200, 300)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 2

    def test_overlapping_preserved(self):
        """Overlapping rectangles are both present (no merge)."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        iv2 = _make_2d("chr1", 500, 1500, "chr1", 500, 1500)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 2

    def test_empty_first(self):
        """Union with empty first set returns second."""
        iv1 = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        iv2 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 1

    def test_empty_second(self):
        """Union with empty second set returns first."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 1

    def test_both_empty(self):
        """Union of two empty sets returns None."""
        iv1 = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        iv2 = pd.DataFrame(columns=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'])
        result = pm.gintervals_2d_union(iv1, iv2)
        assert result is None

    def test_none_input_raises(self):
        """None input raises ValueError."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        with pytest.raises(ValueError):
            pm.gintervals_2d_union(None, iv1)
        with pytest.raises(ValueError):
            pm.gintervals_2d_union(iv1, None)

    def test_missing_columns_raises(self):
        """Input missing required columns raises ValueError."""
        iv1 = _make_2d("chr1", 0, 1000, "chr1", 0, 1000)
        bad = pd.DataFrame({'chrom1': ['chr1'], 'start1': [0], 'end1': [100]})
        with pytest.raises(ValueError, match="missing required 2D columns"):
            pm.gintervals_2d_union(iv1, bad)

    def test_result_sorted(self):
        """Result is sorted by (chrom1, start1, chrom2, start2)."""
        iv1 = _make_2d("chr2", 500, 1000, "chr2", 500, 1000)
        iv2 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert result.iloc[0]['chrom1'] == 'chr1'
        assert result.iloc[1]['chrom1'] == 'chr2'

    def test_result_columns(self):
        """Result has exactly the standard 2D columns."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 200, 300, "chr1", 200, 300)
        result = pm.gintervals_2d_union(iv1, iv2)
        assert list(result.columns) == ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

    def test_extra_columns_dropped(self):
        """Extra columns beyond the standard 6 are dropped."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv1['score'] = 42
        iv2 = _make_2d("chr1", 200, 300, "chr1", 200, 300)
        iv2['score'] = 99
        result = pm.gintervals_2d_union(iv1, iv2)

        assert 'score' not in result.columns
        assert len(result.columns) == 6

    def test_mixed_chrom_pairs(self):
        """Union of intervals on different chrom pairs works."""
        iv1 = _make_2d("chr1", 0, 100, "chr2", 0, 100)
        iv2 = _make_2d("chr2", 0, 100, "chr1", 0, 100)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 2

    def test_with_gintervals_2d(self):
        """Works with output from gintervals_2d()."""
        iv1 = pm.gintervals_2d("1", 0, 100000, "1", 0, 100000)
        iv2 = pm.gintervals_2d("2", 0, 100000, "2", 0, 100000)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 2

    def test_duplicate_intervals_preserved(self):
        """Identical intervals from both sets are both present."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        result = pm.gintervals_2d_union(iv1, iv2)

        assert result is not None
        assert len(result) == 2


# ──────────────────────────────────────────────────────────────────
# Combined / integration tests
# ──────────────────────────────────────────────────────────────────

class TestCombined2D:
    """Integration tests combining intersect and union."""

    def test_intersect_of_union(self):
        """Intersecting a union with one of its components works."""
        iv1 = _make_2d("chr1", 0, 500, "chr1", 0, 500)
        iv2 = _make_2d("chr1", 300, 800, "chr1", 300, 800)
        iv3 = _make_2d("chr1", 200, 600, "chr1", 200, 600)

        union = pm.gintervals_2d_union(iv1, iv2)
        result = pm.gintervals_2d_intersect(union, iv3)

        assert result is not None
        assert len(result) == 2

    def test_intersect_idempotent(self):
        """Intersecting an interval with itself returns the same interval."""
        iv = _make_2d("chr1", 100, 500, "chr1", 200, 700)
        result = pm.gintervals_2d_intersect(iv, iv)

        assert result is not None
        assert len(result) == 1
        row = result.iloc[0]
        assert row['start1'] == 100
        assert row['end1'] == 500
        assert row['start2'] == 200
        assert row['end2'] == 700

    def test_union_then_intersect_with_subset(self):
        """Union of A,B intersected with a small rect returns correct subset."""
        iv1 = _make_2d("chr1", 0, 100, "chr1", 0, 100)
        iv2 = _make_2d("chr1", 1000, 2000, "chr1", 1000, 2000)
        combined = pm.gintervals_2d_union(iv1, iv2)

        # Query that only hits the second rectangle
        query = _make_2d("chr1", 1200, 1800, "chr1", 1200, 1800)
        result = pm.gintervals_2d_intersect(combined, query)

        assert result is not None
        assert len(result) == 1
        row = result.iloc[0]
        assert row['start1'] == 1200
        assert row['end1'] == 1800
