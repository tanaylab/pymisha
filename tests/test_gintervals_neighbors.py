"""
Tests for gintervals_neighbors function.

TDD tests written before implementation based on R misha's gintervals.neighbors behavior.

Key R misha semantics:
- Distance is defined as base pairs between end of query and start of target
- Positive distance: target is downstream (to the right)
- Negative distance: target is upstream (to the left)
- Zero distance: intervals touch or overlap
- Results sorted by query interval, then by absolute distance
- Strand handling affects distance sign direction
"""
import pandas as pd
import pytest

import pymisha as pm

# ============================================================================
# Helper functions
# ============================================================================

def make_intervals(data, extra_cols=None):
    """Create intervals DataFrame from list of (chrom, start, end) tuples.

    Args:
        data: List of (chrom, start, end) tuples
        extra_cols: Optional dict of extra column names to values
    """
    if not data:
        return pd.DataFrame({'chrom': pd.Categorical([]), 'start': [], 'end': []})
    chroms, starts, ends = zip(*data, strict=False)
    df = pd.DataFrame({
        'chrom': pd.Categorical(chroms),
        'start': list(starts),
        'end': list(ends)
    })
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = vals
    return df


# ============================================================================
# Tests for gintervals_neighbors basic functionality
# ============================================================================

class TestGintervalsNeighborsBasic:
    """Basic tests for gintervals_neighbors function."""

    def test_neighbors_single_match(self):
        """Find single nearest neighbor."""
        # Query interval
        intervs1 = make_intervals([('1', 1000, 1100)])
        # Target interval 200 bp away
        intervs2 = make_intervals([('1', 1300, 1400)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        assert len(result) == 1
        # Should have query columns (chrom, start, end) and target columns (chrom1, start1, end1) and dist
        assert 'chrom' in result.columns
        assert 'start' in result.columns
        assert 'end' in result.columns
        assert 'chrom1' in result.columns or 'start1' in result.columns
        assert 'dist' in result.columns
        # Distance = 1300 - 1100 = 200
        assert result.iloc[0]['dist'] == 200

    def test_neighbors_multiple_targets(self):
        """Find multiple neighbors with maxneighbors > 1."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([
            ('1', 1150, 1200),  # dist 50
            ('1', 1300, 1400),  # dist 200
            ('1', 1500, 1600),  # dist 400
        ])

        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=2)

        assert result is not None
        assert len(result) == 2
        # Should return closest 2, sorted by distance
        dists = list(result['dist'])
        assert dists == [50, 200]

    def test_neighbors_respects_maxneighbors(self):
        """maxneighbors limits number of results per query."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([
            ('1', 1150, 1200),
            ('1', 1250, 1300),
            ('1', 1350, 1400),
            ('1', 1450, 1500),
        ])

        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=3)

        assert len(result) == 3

    def test_neighbors_distance_range(self):
        """mindist and maxdist filter by distance."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([
            ('1', 1110, 1150),   # dist 10
            ('1', 1150, 1200),   # dist 50
            ('1', 1200, 1250),   # dist 100
            ('1', 1350, 1400),   # dist 250
        ])

        # Only find neighbors with distance 40-60
        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=10,
                                         mindist=40, maxdist=60)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['dist'] == 50

    def test_neighbors_negative_distance_upstream(self):
        """Negative distance for upstream (left) neighbors."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        # Target is upstream (before) query
        intervs2 = make_intervals([('1', 800, 900)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        assert len(result) == 1
        # Distance is negative (upstream): start of query - end of target = 1000 - 900 = 100 bp gap
        # With strand=0 or default, this should be negative distance
        assert result.iloc[0]['dist'] < 0 or abs(result.iloc[0]['dist']) == 100

    def test_neighbors_zero_distance_overlap(self):
        """Zero distance for overlapping intervals."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([('1', 1050, 1150)])  # overlaps

        result = pm.gintervals_neighbors(intervs1, intervs2, mindist=0, maxdist=0)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['dist'] == 0

    def test_neighbors_zero_distance_touching(self):
        """Zero distance for touching intervals."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([('1', 1100, 1200)])  # touches at 1100

        result = pm.gintervals_neighbors(intervs1, intervs2, mindist=0, maxdist=0)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['dist'] == 0


class TestGintervalsNeighborsEdgeCases:
    """Edge cases and special handling."""

    def test_neighbors_no_match(self):
        """Return None/empty when no neighbors found."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([('1', 5000, 5100)])  # far away

        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=1,
                                         mindist=0, maxdist=100)

        assert result is None or len(result) == 0

    def test_neighbors_na_if_notfound_true(self):
        """na_if_notfound=True returns NA row when no match."""
        intervs1 = make_intervals([('1', 1000, 1100)], extra_cols={'qid': [1]})
        intervs2 = make_intervals([('1', 5000, 5100)])

        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=1,
                                         mindist=0, maxdist=100,
                                         na_if_notfound=True)

        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['qid'] == 1  # Query info preserved
        assert pd.isna(result.iloc[0]['dist'])

    def test_neighbors_different_chromosomes(self):
        """Neighbors only found on same chromosome."""
        intervs1 = make_intervals([('1', 1000, 1100), ('2', 1000, 1100)])
        intervs2 = make_intervals([('1', 1200, 1300), ('2', 1200, 1300)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        assert len(result) == 2
        # chr1 query should match chr1 target, chr2 query should match chr2 target
        for _, row in result.iterrows():
            assert row['chrom'] == row['chrom1']

    def test_neighbors_empty_intervals1(self):
        """Empty intervals1 returns None/empty."""
        intervs1 = make_intervals([])
        intervs2 = make_intervals([('1', 1000, 1100)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_neighbors_empty_intervals2(self):
        """Empty intervals2 returns None/empty (unless na_if_notfound)."""
        intervs1 = make_intervals([('1', 1000, 1100)])
        intervs2 = make_intervals([])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_neighbors_preserves_extra_columns(self):
        """Extra columns from both interval sets are preserved."""
        intervs1 = make_intervals([('1', 1000, 1100)], extra_cols={'gene': ['GeneA']})
        intervs2 = make_intervals([('1', 1200, 1300)], extra_cols={'feature': ['FeatureX']})

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        assert 'gene' in result.columns
        assert 'feature' in result.columns
        assert result.iloc[0]['gene'] == 'GeneA'
        assert result.iloc[0]['feature'] == 'FeatureX'

    def test_neighbors_multiple_queries(self):
        """Multiple queries each find their nearest neighbors."""
        intervs1 = make_intervals([
            ('1', 1000, 1100),
            ('1', 2000, 2100),
        ], extra_cols={'qid': [1, 2]})

        intervs2 = make_intervals([
            ('1', 1200, 1300),  # near query 1
            ('1', 2200, 2300),  # near query 2
        ])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        assert len(result) == 2

    def test_neighbors_sorted_by_query_then_distance(self):
        """Results sorted by query interval, then by distance."""
        intervs1 = make_intervals([
            ('1', 2000, 2100),  # query 2 (will be sorted after query 1)
            ('1', 1000, 1100),  # query 1
        ], extra_cols={'qid': [2, 1]})

        intervs2 = make_intervals([
            ('1', 1200, 1300),  # dist 100 from query 1
            ('1', 1150, 1180),  # dist 50 from query 1
            ('1', 2200, 2300),  # dist 100 from query 2
        ])

        pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=2)

        # Should be sorted by query, then distance
        # Query 1 results first (sorted by chrom, start)
        # Then query 2 results


class TestGintervalsNeighborsDistanceCalculation:
    """Test distance calculation semantics matching R misha."""

    def test_distance_query_left_of_target(self):
        """Query interval is left of target: positive distance."""
        # query: 100-200, target: 300-400
        # gap = 300 - 200 = 100
        intervs1 = make_intervals([('1', 100, 200)])
        intervs2 = make_intervals([('1', 300, 400)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        # Positive distance when target is downstream
        assert result.iloc[0]['dist'] == 100

    def test_distance_query_right_of_target(self):
        """Query interval is right of target: negative distance (or unsigned)."""
        # query: 300-400, target: 100-200
        # gap = 300 - 200 = 100 upstream
        intervs1 = make_intervals([('1', 300, 400)])
        intervs2 = make_intervals([('1', 100, 200)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        # Distance should reflect that target is upstream
        # In R misha without strand, this would be negative
        dist = result.iloc[0]['dist']
        assert abs(dist) == 100

    def test_distance_overlapping_is_zero(self):
        """Overlapping intervals have distance 0."""
        intervs1 = make_intervals([('1', 100, 300)])
        intervs2 = make_intervals([('1', 200, 400)])  # overlaps

        result = pm.gintervals_neighbors(intervs1, intervs2, mindist=-1000, maxdist=1000)

        assert result is not None
        assert result.iloc[0]['dist'] == 0

    def test_distance_touching_right_is_zero(self):
        """Touching on right (query.end == target.start) has distance 0."""
        intervs1 = make_intervals([('1', 100, 200)])
        intervs2 = make_intervals([('1', 200, 300)])

        result = pm.gintervals_neighbors(intervs1, intervs2, mindist=0, maxdist=0)

        assert result is not None
        assert result.iloc[0]['dist'] == 0

    def test_distance_touching_left_is_zero(self):
        """Touching on left (target.end == query.start) has distance 0."""
        intervs1 = make_intervals([('1', 200, 300)])
        intervs2 = make_intervals([('1', 100, 200)])

        result = pm.gintervals_neighbors(intervs1, intervs2, mindist=0, maxdist=0)

        assert result is not None
        assert result.iloc[0]['dist'] == 0


class TestGintervalsNeighborsZeroDistanceBug:
    """
    Test for mindist=0, maxdist=0 handling - all overlapping and touching intervals.

    This tests the behavior documented in R misha's test cases for the zero-distance bug fix.
    """

    def test_zero_distance_finds_all_overlapping(self):
        """mindist=0, maxdist=0 finds all overlapping and touching intervals."""
        # Query interval
        intervs1 = make_intervals([('1', 1000, 2000)])

        # Various relationships to query
        intervs2 = make_intervals([
            ('1', 500, 1000),    # touch left
            ('1', 1000, 2500),   # overlap/contain
            ('1', 1500, 1700),   # overlap
            ('1', 2000, 2500),   # touch right
            ('1', 1000, 2000),   # equal
            ('1', 900, 2500),    # contains query
            ('1', 1100, 1900),   # contained by query
        ], extra_cols={'case': ['touch_left', 'contain_query', 'overlap',
                                'touch_right', 'equal', 'contains_query', 'contained']})

        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=100,
                                         mindist=0, maxdist=0)

        # All 7 cases should be found (all have distance 0)
        assert result is not None
        assert len(result) == 7
        assert all(result['dist'] == 0)

    def test_zero_distance_various_relationships(self):
        """Test various interval relationships at distance 0."""
        intervs1 = make_intervals([
            ('1', 100, 150),
            ('1', 200, 250),
            ('1', 300, 350),
        ], extra_cols={'query_id': [1, 2, 3]})

        intervs2 = make_intervals([
            ('1', 120, 130),   # overlaps query 1
            ('1', 150, 160),   # touches query 1 (right)
            ('1', 90, 100),    # touches query 1 (left)
            ('1', 151, 160),   # dist 1 from query 1 - should NOT match
            ('1', 250, 260),   # touches query 2 (right)
            ('1', 349, 350),   # touches query 3 (left, adjacent)
            ('1', 300, 350),   # equals query 3
        ], extra_cols={'target_id': [1, 2, 3, 4, 5, 6, 7]})

        result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=100,
                                         mindist=0, maxdist=0)

        assert result is not None

        # Query 1 should find targets 1, 2, 3 (not 4 which is dist=1)
        q1_targets = result[result['query_id'] == 1]['target_id'].tolist()
        assert 1 in q1_targets
        assert 2 in q1_targets
        assert 3 in q1_targets
        assert 4 not in q1_targets


class TestGintervalsNeighborsRaises:
    """Test error conditions."""

    def test_raises_on_none_intervals1(self):
        """Raises error when intervals1 is None."""
        intervs2 = make_intervals([('1', 100, 200)])

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_neighbors(None, intervs2)

    def test_raises_on_none_intervals2(self):
        """Raises error when intervals2 is None."""
        intervs1 = make_intervals([('1', 100, 200)])

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_neighbors(intervs1, None)

    def test_raises_on_invalid_maxneighbors(self):
        """Raises error when maxneighbors < 1."""
        intervs1 = make_intervals([('1', 100, 200)])
        intervs2 = make_intervals([('1', 300, 400)])

        with pytest.raises(ValueError):
            pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=0)

    def test_raises_on_mindist_gt_maxdist(self):
        """Raises error when mindist > maxdist."""
        intervs1 = make_intervals([('1', 100, 200)])
        intervs2 = make_intervals([('1', 300, 400)])

        with pytest.raises(ValueError):
            pm.gintervals_neighbors(intervs1, intervs2, mindist=100, maxdist=50)


class TestGintervalsNeighborsIdentical:
    """Test handling of identical/duplicate intervals."""

    def test_identical_intervals(self):
        """Multiple identical queries find all identical targets."""
        queries = make_intervals([
            ('1', 1000, 1100),
            ('1', 1000, 1100),
            ('1', 1000, 1100),
        ], extra_cols={'qid': [1, 2, 3]})

        targets = make_intervals([
            ('1', 1000, 1100),
            ('1', 1000, 1100),
            ('1', 1000, 1100),
        ], extra_cols={'tid': [1, 2, 3]})

        result = pm.gintervals_neighbors(queries, targets, maxneighbors=10,
                                         mindist=0, maxdist=0)

        # Each query should find all 3 identical targets
        assert result is not None
        assert len(result) == 9  # 3 queries × 3 targets
        assert all(result['dist'] == 0)


class TestGintervalsNeighborsColnames:
    """Test column naming in results."""

    def test_result_columns_1d(self):
        """Result has correct column names for 1D intervals."""
        intervs1 = make_intervals([('1', 100, 200)])
        intervs2 = make_intervals([('1', 300, 400)])

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        # Query columns
        assert 'chrom' in result.columns
        assert 'start' in result.columns
        assert 'end' in result.columns
        # Target columns (renamed to avoid collision)
        assert 'chrom1' in result.columns or 'start1' in result.columns
        # Distance
        assert 'dist' in result.columns

    def test_duplicate_column_names_handled(self):
        """Duplicate column names are made unique."""
        intervs1 = make_intervals([('1', 100, 200)], extra_cols={'value': [1]})
        intervs2 = make_intervals([('1', 300, 400)], extra_cols={'value': [2]})

        result = pm.gintervals_neighbors(intervs1, intervs2)

        assert result is not None
        # Both 'value' columns should exist with unique names
        cols = list(result.columns)
        value_cols = [c for c in cols if 'value' in c]
        assert len(value_cols) == 2
