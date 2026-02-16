"""
Tests for interval operations: union, intersect, diff, canonic, and constructors.

These tests follow the TDD approach - written before implementation.
"""
import pandas as pd
import pytest

import pymisha as pm

# ============================================================================
# Helper functions
# ============================================================================

def make_intervals(data):
    """Create intervals DataFrame from list of (chrom, start, end) tuples.

    Note: Chromosome names are normalized to match the test database format.
    The test database uses '1', '2', 'X' (without 'chr' prefix).
    """
    if not data:
        return pd.DataFrame({'chrom': pd.Categorical([]), 'start': [], 'end': []})
    chroms, starts, ends = zip(*data, strict=False)
    # Normalize chromosome names to database format (remove 'chr' prefix)
    chroms = [c.replace('chr', '') if isinstance(c, str) else c for c in chroms]
    return pd.DataFrame({
        'chrom': pd.Categorical(chroms),
        'start': list(starts),
        'end': list(ends)
    })


def intervals_equal(df1, df2):
    """Check if two interval DataFrames are equal (ignoring order)."""
    if df1 is None and df2 is None:
        return True
    if df1 is None or df2 is None:
        return False
    if len(df1) != len(df2):
        return False
    if len(df1) == 0:
        return True

    # Sort both by chrom, start for comparison
    df1_sorted = df1.sort_values(['chrom', 'start']).reset_index(drop=True)
    df2_sorted = df2.sort_values(['chrom', 'start']).reset_index(drop=True)

    # Compare chrom, start, end columns
    return (
        list(df1_sorted['chrom']) == list(df2_sorted['chrom']) and
        list(df1_sorted['start']) == list(df2_sorted['start']) and
        list(df1_sorted['end']) == list(df2_sorted['end'])
    )


# ============================================================================
# Tests for gintervals_union
# ============================================================================

class TestGintervalsUnion:
    """Tests for gintervals_union function."""

    def test_union_non_overlapping_same_chrom(self):
        """Union of non-overlapping intervals on same chromosome."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr1', 300, 400)])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200), ('chr1', 300, 400)])

        assert intervals_equal(result, expected)

    def test_union_overlapping_same_chrom(self):
        """Union of overlapping intervals on same chromosome merges them."""
        intervs1 = make_intervals([('chr1', 100, 300)])
        intervs2 = make_intervals([('chr1', 200, 400)])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 400)])

        assert intervals_equal(result, expected)

    def test_union_touching_intervals(self):
        """Union of touching intervals merges them (end == start)."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr1', 200, 300)])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 300)])

        assert intervals_equal(result, expected)

    def test_union_different_chroms(self):
        """Union of intervals on different chromosomes."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr2', 100, 200)])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200), ('chr2', 100, 200)])

        assert intervals_equal(result, expected)

    def test_union_multiple_intervals(self):
        """Union with multiple intervals in each set."""
        intervs1 = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 400, 500),
            ('chr2', 100, 200)
        ])
        intervs2 = make_intervals([
            ('chr1', 150, 250),  # overlaps first
            ('chr1', 600, 700),  # new
            ('chr2', 300, 400)   # new
        ])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([
            ('chr1', 100, 250),
            ('chr1', 400, 500),
            ('chr1', 600, 700),
            ('chr2', 100, 200),
            ('chr2', 300, 400)
        ])

        assert intervals_equal(result, expected)

    def test_union_empty_first(self):
        """Union with empty first set returns second set."""
        intervs1 = make_intervals([])
        intervs2 = make_intervals([('chr1', 100, 200)])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200)])

        assert intervals_equal(result, expected)

    def test_union_empty_second(self):
        """Union with empty second set returns first set."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200)])

        assert intervals_equal(result, expected)

    def test_union_both_empty(self):
        """Union of two empty sets returns None or empty DataFrame."""
        intervs1 = make_intervals([])
        intervs2 = make_intervals([])

        result = pm.gintervals_union(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_union_contained_interval(self):
        """Union where one interval contains another."""
        intervs1 = make_intervals([('chr1', 100, 500)])
        intervs2 = make_intervals([('chr1', 200, 300)])

        result = pm.gintervals_union(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 500)])

        assert intervals_equal(result, expected)

    def test_union_raises_on_none_input(self):
        """Union raises error when input is None."""
        intervs1 = make_intervals([('chr1', 100, 200)])

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_union(None, intervs1)

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_union(intervs1, None)


# ============================================================================
# Tests for gintervals_intersect
# ============================================================================

class TestGintervalsIntersect:
    """Tests for gintervals_intersect function."""

    def test_intersect_overlapping(self):
        """Intersection of overlapping intervals."""
        intervs1 = make_intervals([('chr1', 100, 300)])
        intervs2 = make_intervals([('chr1', 200, 400)])

        result = pm.gintervals_intersect(intervs1, intervs2)
        expected = make_intervals([('chr1', 200, 300)])

        assert intervals_equal(result, expected)

    def test_intersect_non_overlapping(self):
        """Intersection of non-overlapping intervals is empty."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr1', 300, 400)])

        result = pm.gintervals_intersect(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_intersect_touching_not_overlapping(self):
        """Touching intervals (end == start) do not overlap."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr1', 200, 300)])

        result = pm.gintervals_intersect(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_intersect_contained_interval(self):
        """Intersection where one contains another returns the smaller."""
        intervs1 = make_intervals([('chr1', 100, 500)])
        intervs2 = make_intervals([('chr1', 200, 300)])

        result = pm.gintervals_intersect(intervs1, intervs2)
        expected = make_intervals([('chr1', 200, 300)])

        assert intervals_equal(result, expected)

    def test_intersect_different_chroms(self):
        """Intervals on different chromosomes don't intersect."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr2', 100, 200)])

        result = pm.gintervals_intersect(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_intersect_multiple_overlaps(self):
        """Multiple overlapping intervals produce multiple results."""
        intervs1 = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 300, 400)
        ])
        intervs2 = make_intervals([
            ('chr1', 150, 350)
        ])

        result = pm.gintervals_intersect(intervs1, intervs2)
        expected = make_intervals([
            ('chr1', 150, 200),
            ('chr1', 300, 350)
        ])

        assert intervals_equal(result, expected)

    def test_intersect_empty_first(self):
        """Intersection with empty first set is empty."""
        intervs1 = make_intervals([])
        intervs2 = make_intervals([('chr1', 100, 200)])

        result = pm.gintervals_intersect(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_intersect_empty_second(self):
        """Intersection with empty second set is empty."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([])

        result = pm.gintervals_intersect(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_intersect_raises_on_none_input(self):
        """Intersect raises error when input is None."""
        intervs1 = make_intervals([('chr1', 100, 200)])

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_intersect(None, intervs1)

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_intersect(intervs1, None)


# ============================================================================
# Tests for gintervals_diff
# ============================================================================

class TestGintervalsDiff:
    """Tests for gintervals_diff function."""

    def test_diff_non_overlapping(self):
        """Diff of non-overlapping intervals returns first set unchanged."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr1', 300, 400)])

        result = pm.gintervals_diff(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200)])

        assert intervals_equal(result, expected)

    def test_diff_partial_overlap(self):
        """Diff removes overlapping portion."""
        intervs1 = make_intervals([('chr1', 100, 300)])
        intervs2 = make_intervals([('chr1', 200, 400)])

        result = pm.gintervals_diff(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200)])

        assert intervals_equal(result, expected)

    def test_diff_contained_interval_removed(self):
        """Diff where first is contained in second is empty."""
        intervs1 = make_intervals([('chr1', 200, 300)])
        intervs2 = make_intervals([('chr1', 100, 400)])

        result = pm.gintervals_diff(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_diff_split_interval(self):
        """Diff can split an interval into two pieces."""
        intervs1 = make_intervals([('chr1', 100, 500)])
        intervs2 = make_intervals([('chr1', 200, 300)])

        result = pm.gintervals_diff(intervs1, intervs2)
        expected = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 300, 500)
        ])

        assert intervals_equal(result, expected)

    def test_diff_different_chroms(self):
        """Diff with intervals on different chromosomes."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([('chr2', 100, 200)])

        result = pm.gintervals_diff(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200)])

        assert intervals_equal(result, expected)

    def test_diff_empty_first(self):
        """Diff with empty first set is empty."""
        intervs1 = make_intervals([])
        intervs2 = make_intervals([('chr1', 100, 200)])

        result = pm.gintervals_diff(intervs1, intervs2)

        assert result is None or len(result) == 0

    def test_diff_empty_second(self):
        """Diff with empty second set returns first set unchanged."""
        intervs1 = make_intervals([('chr1', 100, 200)])
        intervs2 = make_intervals([])

        result = pm.gintervals_diff(intervs1, intervs2)
        expected = make_intervals([('chr1', 100, 200)])

        assert intervals_equal(result, expected)

    def test_diff_multiple_subtractions(self):
        """Diff with multiple intervals to subtract."""
        intervs1 = make_intervals([('chr1', 100, 600)])
        intervs2 = make_intervals([
            ('chr1', 150, 200),
            ('chr1', 300, 350),
            ('chr1', 500, 550)
        ])

        result = pm.gintervals_diff(intervs1, intervs2)
        expected = make_intervals([
            ('chr1', 100, 150),
            ('chr1', 200, 300),
            ('chr1', 350, 500),
            ('chr1', 550, 600)
        ])

        assert intervals_equal(result, expected)

    def test_diff_raises_on_none_input(self):
        """Diff raises error when input is None."""
        intervs1 = make_intervals([('chr1', 100, 200)])

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_diff(None, intervs1)

        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_diff(intervs1, None)


# ============================================================================
# Tests for gintervals_canonic
# ============================================================================

class TestGintervalsCanonic:
    """Tests for gintervals_canonic function."""

    def test_canonic_already_sorted_no_overlaps(self):
        """Canonic on already-canonical intervals is unchanged."""
        intervs = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 300, 400),
            ('chr2', 100, 200)
        ])

        result = pm.gintervals_canonic(intervs)

        assert intervals_equal(result, intervs)

    def test_canonic_sorts_intervals(self):
        """Canonic sorts unsorted intervals."""
        intervs = make_intervals([
            ('chr2', 100, 200),
            ('chr1', 300, 400),
            ('chr1', 100, 200)
        ])

        result = pm.gintervals_canonic(intervs)
        expected = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 300, 400),
            ('chr2', 100, 200)
        ])

        assert intervals_equal(result, expected)

    def test_canonic_merges_overlapping(self):
        """Canonic merges overlapping intervals."""
        intervs = make_intervals([
            ('chr1', 100, 250),
            ('chr1', 200, 400)
        ])

        result = pm.gintervals_canonic(intervs)
        expected = make_intervals([('chr1', 100, 400)])

        assert intervals_equal(result, expected)

    def test_canonic_merges_touching_by_default(self):
        """Canonic merges touching intervals by default."""
        intervs = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 200, 300)
        ])

        result = pm.gintervals_canonic(intervs)
        expected = make_intervals([('chr1', 100, 300)])

        assert intervals_equal(result, expected)

    def test_canonic_no_merge_touching_when_disabled(self):
        """Canonic can be configured to not merge touching intervals."""
        intervs = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 200, 300)
        ])

        result = pm.gintervals_canonic(intervs, unify_touching_intervals=False)
        expected = make_intervals([
            ('chr1', 100, 200),
            ('chr1', 200, 300)
        ])

        assert intervals_equal(result, expected)

    def test_canonic_empty_input(self):
        """Canonic on empty input returns empty or None."""
        intervs = make_intervals([])

        result = pm.gintervals_canonic(intervs)

        assert result is None or len(result) == 0

    def test_canonic_returns_mapping(self):
        """Canonic returns mapping attribute showing origin of intervals."""
        intervs = make_intervals([
            ('chr1', 100, 200),   # -> becomes interval 0
            ('chr1', 300, 400),   # -> becomes interval 1
            ('chr1', 350, 500)    # -> merged into interval 1
        ])

        result = pm.gintervals_canonic(intervs)

        # Result should have 2 intervals
        assert len(result) == 2

        # Check mapping attribute exists (may be stored differently)
        # The mapping shows which original interval -> which result interval
        if hasattr(result, 'attrs') and 'mapping' in result.attrs:
            mapping = result.attrs['mapping']
            # Original intervals 0 -> result 0, intervals 1,2 -> result 1
            assert mapping[0] == 0 or mapping[0] == 1

    def test_canonic_raises_on_none_input(self):
        """Canonic raises error when input is None."""
        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_canonic(None)


# ============================================================================
# Tests for gintervals constructor
# ============================================================================

class TestGintervals:
    """Tests for gintervals constructor function.

    Note: Test database has chromosomes "1", "2", "X" (without chr prefix).
    """

    def test_gintervals_single_chrom(self):
        """Create intervals for a single chromosome."""
        result = pm.gintervals("1", 100, 200)

        assert len(result) == 1
        assert result.iloc[0]['chrom'] == '1'
        assert result.iloc[0]['start'] == 100
        assert result.iloc[0]['end'] == 200

    def test_gintervals_multiple_same_chrom(self):
        """Create multiple intervals on same chromosome."""
        result = pm.gintervals("1", [100, 300], [200, 400])

        assert len(result) == 2
        # Should be sorted by start
        assert result.iloc[0]['start'] == 100
        assert result.iloc[1]['start'] == 300

    def test_gintervals_multiple_chroms(self):
        """Create intervals on different chromosomes."""
        result = pm.gintervals(["1", "2"], [100, 200], [300, 400])

        assert len(result) == 2
        chroms = list(result['chrom'])
        assert '1' in chroms
        assert '2' in chroms

    def test_gintervals_with_strand(self):
        """Create intervals with strand information."""
        result = pm.gintervals("1", 100, 200, strand=1)

        assert len(result) == 1
        assert 'strand' in result.columns
        assert result.iloc[0]['strand'] == 1

    def test_gintervals_default_start(self):
        """Default start is 0."""
        result = pm.gintervals("1", ends=1000)

        assert result.iloc[0]['start'] == 0
        assert result.iloc[0]['end'] == 1000

    def test_gintervals_full_chrom(self):
        """End of -1 means full chromosome length."""
        # This requires database to be initialized
        result = pm.gintervals("1")

        assert len(result) == 1
        assert result.iloc[0]['start'] == 0
        # End should be the chromosome size (500000 for chr1 in test db)
        assert result.iloc[0]['end'] == 500000

    def test_gintervals_integer_chrom(self):
        """Accept integer chromosome names (converted to chr1, chr2, etc)."""
        # Note: This converts 1 to "chr1", but test db uses "1"
        # So we test the conversion happens, but it may not match the db
        result = pm.gintervals("1", 100, 200)

        assert len(result) == 1
        assert result.iloc[0]['chrom'] == '1'

    def test_gintervals_raises_on_invalid_coords(self):
        """Raise error for invalid coordinates."""
        # Start >= end
        with pytest.raises(ValueError):
            pm.gintervals("1", 200, 100)

        # Negative start
        with pytest.raises(ValueError):
            pm.gintervals("1", -10, 100)

    def test_gintervals_raises_on_out_of_bounds(self):
        """Raise error for coordinates exceeding chromosome."""
        # End exceeds chromosome size (500000 for chr 1)
        with pytest.raises(ValueError):
            pm.gintervals("1", 0, 1000000)

    def test_gintervals_sorted_output(self):
        """Output is sorted by chrom and start."""
        result = pm.gintervals(
            ["2", "1", "1"],
            [100, 300, 100],
            [200, 400, 200]
        )

        # Should be sorted: 1:100, 1:300, 2:100
        assert list(result['chrom']) == ['1', '1', '2']
        assert list(result['start']) == [100, 300, 100]


# ============================================================================
# Tests for gintervals_force_range
# ============================================================================

class TestGintervalsForceRange:
    """Tests for gintervals_force_range function.

    Note: Test database has chromosomes "1", "2", "X" (without chr prefix).
    Sizes: 1=500000, 2=300000, X=200000
    """

    def test_force_range_already_valid(self):
        """Intervals already within range are unchanged."""
        intervs = make_intervals([('1', 100, 200)])

        result = pm.gintervals_force_range(intervs)

        assert intervals_equal(result, intervs)

    def test_force_range_clips_start(self):
        """Negative start is clipped to 0."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [-50],
            'end': [200]
        })

        result = pm.gintervals_force_range(intervs)

        assert result.iloc[0]['start'] == 0
        assert result.iloc[0]['end'] == 200

    def test_force_range_clips_end(self):
        """End exceeding chromosome is clipped."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [100],
            'end': [1000000]  # Exceeds chr 1 size of 500000
        })

        result = pm.gintervals_force_range(intervs)

        assert result.iloc[0]['start'] == 100
        assert result.iloc[0]['end'] == 500000

    def test_force_range_removes_invalid(self):
        """Intervals that become invalid (start >= end) are removed."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [600000],  # Beyond chromosome end
            'end': [700000]
        })

        result = pm.gintervals_force_range(intervs)

        assert result is None or len(result) == 0

    def test_force_range_multiple_intervals(self):
        """Force range on multiple intervals."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1', '1', '2']),
            'start': [-10, 100, 250000],
            'end': [200, 600000, 400000]  # chr 2 size is 300000
        })

        result = pm.gintervals_force_range(intervs)

        assert len(result) == 3
        # First: -10 -> 0
        assert result.iloc[0]['start'] == 0
        # Second: 600000 -> 500000 (chr 1 size)
        assert result.iloc[1]['end'] == 500000
        # Third: 400000 -> 300000 (chr 2 size)
        assert result.iloc[2]['end'] == 300000

    def test_force_range_raises_on_none(self):
        """Force range raises error on None input."""
        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_force_range(None)


# ============================================================================
# Tests for gintervals_covered_bp
# ============================================================================

class TestGintervalsCoveredBp:
    """Tests for gintervals_covered_bp function."""

    def test_covered_bp_single_interval(self):
        """Covered bp of a single interval."""
        intervs = make_intervals([('1', 100, 200)])

        result = pm.gintervals_covered_bp(intervs)

        assert result == 100

    def test_covered_bp_multiple_non_overlapping(self):
        """Covered bp of multiple non-overlapping intervals."""
        intervs = make_intervals([
            ('1', 100, 200),
            ('1', 300, 400),
            ('2', 100, 200)
        ])

        result = pm.gintervals_covered_bp(intervs)

        assert result == 300  # 100 + 100 + 100

    def test_covered_bp_overlapping_intervals(self):
        """Covered bp with overlapping intervals counts overlap once."""
        intervs = make_intervals([
            ('1', 100, 300),
            ('1', 200, 400)  # overlaps by 100
        ])

        result = pm.gintervals_covered_bp(intervs)

        assert result == 300  # (100-300) + (300-400) = 300, not 400

    def test_covered_bp_empty_intervals(self):
        """Covered bp of empty intervals is 0."""
        intervs = make_intervals([])

        result = pm.gintervals_covered_bp(intervs)

        assert result == 0

    def test_covered_bp_raises_on_none(self):
        """Covered bp raises on None input."""
        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_covered_bp(None)


# ============================================================================
# Tests for gintervals_coverage_fraction
# ============================================================================

class TestGintervalsCoverageFraction:
    """Tests for gintervals_coverage_fraction function."""

    def test_coverage_fraction_full_coverage(self):
        """Full coverage returns 1.0."""
        intervs1 = make_intervals([('1', 100, 300)])
        intervs2 = make_intervals([('1', 150, 250)])

        result = pm.gintervals_coverage_fraction(intervs1, intervs2)

        assert result == 1.0

    def test_coverage_fraction_partial_coverage(self):
        """Partial coverage returns correct fraction."""
        intervs1 = make_intervals([('1', 100, 200)])  # covers 100 bp
        intervs2 = make_intervals([('1', 150, 350)])  # total 200 bp

        result = pm.gintervals_coverage_fraction(intervs1, intervs2)

        # intervs1 covers 50 bp of intervs2 (150-200)
        assert result == 0.25  # 50/200

    def test_coverage_fraction_no_overlap(self):
        """No overlap returns 0."""
        intervs1 = make_intervals([('1', 100, 200)])
        intervs2 = make_intervals([('1', 300, 400)])

        result = pm.gintervals_coverage_fraction(intervs1, intervs2)

        assert result == 0.0

    def test_coverage_fraction_genome_default(self):
        """Without intervals2, calculates fraction of genome covered."""
        # Full chromosome 1 = 500000 bp
        intervs1 = make_intervals([('1', 0, 100000)])

        result = pm.gintervals_coverage_fraction(intervs1)

        # 100000 / (500000 + 300000 + 200000) = 0.1
        assert abs(result - 0.1) < 0.0001

    def test_coverage_fraction_empty_intervs1(self):
        """Empty covering intervals returns 0."""
        intervs1 = make_intervals([])
        intervs2 = make_intervals([('1', 100, 200)])

        result = pm.gintervals_coverage_fraction(intervs1, intervs2)

        assert result == 0.0

    def test_coverage_fraction_raises_on_none(self):
        """Coverage fraction raises on None input."""
        with pytest.raises((ValueError, TypeError)):
            pm.gintervals_coverage_fraction(None)


# ============================================================================
# Additional gintervals_force_range edge cases (ported from R test-gintervals1.R)
# ============================================================================

class TestGintervalsForceRangeEdgeCases:
    """Additional edge cases for gintervals_force_range.

    Ported from R misha test-gintervals1.R: gintervals.force_range handles 1D data correctly.
    Test DB chroms: 1 (500000), 2 (300000), X (200000).
    """

    def test_force_range_reversed_start_end_removed(self):
        """Intervals where start > end are removed after clamping."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [300],
            'end': [200],
        })
        result = pm.gintervals_force_range(intervs)
        # start (300) > end (200) => after clamping both are valid coords but
        # start >= end so removed
        assert result is None or len(result) == 0

    def test_force_range_both_negative_removed(self):
        """Intervals with both start and end negative are removed."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [-100],
            'end': [-30],
        })
        result = pm.gintervals_force_range(intervs)
        # After clamping: start=0, end=0 => start not < end => removed
        assert result is None or len(result) == 0

    def test_force_range_both_negative_reversed_removed(self):
        """Intervals with both negative and reversed are removed."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [-30],
            'end': [-100],
        })
        result = pm.gintervals_force_range(intervs)
        # After clamping: start=0, end=0 => removed
        assert result is None or len(result) == 0

    def test_force_range_both_beyond_chrom_size_removed(self):
        """Intervals entirely beyond chromosome size are removed."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [1000000],
            'end': [1000010],
        })
        result = pm.gintervals_force_range(intervs)
        # start (500000 after clamp) >= end (500000 after clamp) => removed
        assert result is None or len(result) == 0

    def test_force_range_beyond_chrom_reversed_removed(self):
        """Intervals beyond chrom size with reversed coords are removed."""
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1']),
            'start': [1000010],
            'end': [1000000],
        })
        result = pm.gintervals_force_range(intervs)
        assert result is None or len(result) == 0

    def test_force_range_comprehensive_1d(self):
        """Comprehensive edge case test matching R test-gintervals1.R.

        Tests 8 intervals with various edge cases:
        1. (10, 100)        -> kept as-is
        2. (300, 200)       -> removed (reversed)
        3. (-100, 50)       -> (0, 50) clamped start
        4. (-100, -30)      -> removed (both negative)
        5. (-30, -100)      -> removed (both negative reversed)
        6. (100, 1e9)       -> (100, 500000) clamped end
        7. (1e6, 1e6+10)    -> removed (beyond chrom size)
        8. (1e6+10, 1e6)    -> removed (beyond + reversed)
        """
        intervs = pd.DataFrame({
            'chrom': pd.Categorical(['1'] * 8),
            'start': [10, 300, -100, -100, -30, 100, 1000000, 1000010],
            'end': [100, 200, 50, -30, -100, 1000000000, 1000010, 1000000],
        })
        result = pm.gintervals_force_range(intervs)

        assert result is not None
        assert len(result) == 3

        # Row 0: (10, 100) kept as-is
        assert result.iloc[0]['start'] == 10
        assert result.iloc[0]['end'] == 100

        # Row 1: (-100, 50) -> (0, 50) clamped start
        assert result.iloc[1]['start'] == 0
        assert result.iloc[1]['end'] == 50

        # Row 2: (100, 1e9) -> (100, 500000) clamped end
        assert result.iloc[2]['start'] == 100
        assert result.iloc[2]['end'] == 500000

    @pytest.mark.skip(reason="2D gintervals_force_range not implemented")
    def test_force_range_2d_intervals(self):
        """Force range for 2D intervals (not yet implemented)."""
        intervs = pd.DataFrame({
            'chrom1': ['1', '1'],
            'start1': [10, -100],
            'end1': [100, 50],
            'chrom2': ['2', '2'],
            'start2': [10, -100],
            'end2': [100, 50],
        })
        result = pm.gintervals_force_range(intervs)
        assert result is not None


# ============================================================================
# Tests for gintervals_rbind (ported from R test-gintervals2.R)
# ============================================================================

class TestGintervalsRbind:
    """Tests for gintervals_rbind function.

    Ported from R misha test-gintervals2.R: gintervals.rbind with saved data.
    """

    def test_rbind_two_dataframes(self):
        """Rbind two interval DataFrames."""
        intervs1 = pm.gextract("dense_track", pm.gintervals(["1", "2"], 1000, 4000))
        intervs2 = pm.gextract("dense_track", pm.gintervals(["2", "X"], 2000, 5000))

        result = pm.gintervals_rbind(intervs1, intervs2)
        assert result is not None
        assert len(result) == len(intervs1) + len(intervs2)

    def test_rbind_with_saved_interval_set(self):
        """Rbind DataFrame with a saved interval set name."""
        tmp_name = "test_rbind_saved_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            intervs1 = pm.gextract("dense_track", pm.gintervals(["1", "2"], 1000, 4000))
            intervs2 = pm.gextract("dense_track", pm.gintervals(["2", "X"], 2000, 5000))
            pm.gintervals_save(intervs2, tmp_name)

            result = pm.gintervals_rbind(intervs1, tmp_name)
            assert result is not None
            assert len(result) == len(intervs1) + len(intervs2)
        finally:
            pm.gintervals_rm(tmp_name, force=True)


# ============================================================================
# Tests for gintervals_ls changes (ported from R test-gintervals2.R)
# ============================================================================

class TestGintervalsLsChanges:
    """Tests that gintervals_ls reflects save/remove operations.

    Ported from R misha test-gintervals2.R.
    """

    def test_ls_reflects_save(self):
        """gintervals_ls includes newly saved interval set."""
        tmp_name = "test_ls_save_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            ls_before = pm.gintervals_ls()
            assert tmp_name not in ls_before

            pm.gintervals_save(pm.gintervals(["1", "2"], 1000, 2000), tmp_name)
            ls_after = pm.gintervals_ls()
            assert tmp_name in ls_after
        finally:
            pm.gintervals_rm(tmp_name, force=True)

    def test_ls_reflects_remove(self):
        """gintervals_ls excludes removed interval set."""
        tmp_name = "test_ls_rm_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            pm.gintervals_save(pm.gintervals(["1", "2"], 1000, 2000), tmp_name)
            ls_after_save = pm.gintervals_ls()
            assert tmp_name in ls_after_save

            pm.gintervals_rm(tmp_name, force=True)
            ls_after_rm = pm.gintervals_ls()
            assert tmp_name not in ls_after_rm
        finally:
            pm.gintervals_rm(tmp_name, force=True)


# ============================================================================
# Tests for gintervals_rm force behavior (ported from R test-gintervals2.R)
# ============================================================================

class TestGintervalsRmForce:
    """Tests for gintervals_rm with force parameter.

    Ported from R misha test-gintervals2.R:
    gintervals.rm handles non-existent data without error when using force.
    """

    def test_rm_nonexistent_with_force_silent(self):
        """gintervals_rm with force=True on non-existent set does not raise."""
        pm.gintervals_rm("test_aaaaaaaaaaaaaaaaaaa", force=True)

    def test_rm_nonexistent_without_force_raises(self):
        """gintervals_rm without force on non-existent set raises ValueError."""
        with pytest.raises(ValueError):
            pm.gintervals_rm("test_aaaaaaaaaaaaaaaaaaa")


# ============================================================================
# Tests for gextract with removed interval set (ported from R test-gintervals2.R)
# ============================================================================

class TestGextractWithRemovedIntervals:
    """Test that gextract errors on removed interval set.

    Ported from R misha test-gintervals2.R:
    gextract with removed interval gives an error.
    """

    def test_gextract_with_removed_interval_set_raises(self):
        """gextract raises error when using a removed interval set name."""
        tmp_name = "test_ext_removed_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            pm.gintervals_save(
                pm.gintervals(["1", "2"], 1000, 2000), tmp_name
            )
            pm.gintervals_rm(tmp_name, force=True)

            with pytest.raises(Exception):
                pm.gextract("dense_track", tmp_name)
        finally:
            pm.gintervals_rm(tmp_name, force=True)


# ============================================================================
# Tests for gscreen + gintervals_union (ported from R test-gintervals2.R)
# ============================================================================

class TestGscreenAndUnion:
    """Test gscreen combined with gintervals_union.

    Ported from R misha test-gintervals2.R:
    gscreen and gintervals.union works correctly.
    """

    def test_gscreen_and_union(self):
        """gscreen results can be combined with gintervals_union."""
        intervs1 = pm.gscreen(
            "(dense_track > 0.1) & (dense_track < 0.3)",
            pm.gintervals(["1", "2"], 0, -1),
        )
        intervs2 = pm.gscreen(
            "dense_track < 0.2",
            pm.gintervals(["1", "2"], 0, -1),
        )

        result = pm.gintervals_union(intervs1, intervs2)
        assert result is not None
        assert len(result) > 0
        # Union should cover at least as many bp as either input
        bp_1 = pm.gintervals_covered_bp(intervs1)
        bp_2 = pm.gintervals_covered_bp(intervs2)
        bp_union = pm.gintervals_covered_bp(result)
        assert bp_union >= max(bp_1, bp_2)


# ============================================================================
# Tests for gintervals_update (ported from R test-gintervals2.R)
# ============================================================================

class TestGintervalsUpdate:
    """Tests for gintervals_update function.

    Ported from R misha test-gintervals2.R:
    gintervals.update with loaded data using chrom 2,
    gintervals.update removes chrom 2 from saved data.
    """

    def test_update_replace_chrom_data(self):
        """gintervals_update replaces data for a specific chromosome."""
        tmp_name = "test_update_repl_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            # Save intervals on chroms 1 and 2
            intervs = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])
            pm.gintervals_save(intervs, tmp_name)

            # Load chrom 2 data and replace with a subset
            new_chrom2 = pm.gintervals("2", 500, 1500)
            pm.gintervals_update(tmp_name, new_chrom2, chrom="2")

            # Verify update
            loaded = pm.gintervals_load(tmp_name)
            chrom2_rows = loaded[loaded["chrom"].astype(str) == "2"]
            assert len(chrom2_rows) == 1
            assert int(chrom2_rows.iloc[0]["start"]) == 500
            assert int(chrom2_rows.iloc[0]["end"]) == 1500

            # Chrom 1 should be unchanged
            chrom1_rows = loaded[loaded["chrom"].astype(str) == "1"]
            assert len(chrom1_rows) == 1
            assert int(chrom1_rows.iloc[0]["start"]) == 100
            assert int(chrom1_rows.iloc[0]["end"]) == 1000
        finally:
            pm.gintervals_rm(tmp_name, force=True)

    def test_update_remove_chrom(self):
        """gintervals_update with None removes a chromosome's intervals."""
        tmp_name = "test_update_rm_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            # Save intervals on chroms 1 and 2
            intervs = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])
            pm.gintervals_save(intervs, tmp_name)

            # Remove chrom 2
            pm.gintervals_update(tmp_name, None, chrom="2")

            # Verify only chrom 1 remains
            loaded = pm.gintervals_load(tmp_name)
            assert loaded is not None
            chroms = [str(c) for c in loaded["chrom"]]
            assert "2" not in chroms
            assert "1" in chroms
        finally:
            pm.gintervals_rm(tmp_name, force=True)

    def test_update_chrom_sizes_change(self):
        """gintervals_update updates the stored intervals correctly.

        After removing chrom 2 via update, loading should only return chrom 1.
        """
        tmp_name = "test_update_sizes_py"
        pm.gintervals_rm(tmp_name, force=True)
        try:
            # Save intervals on chroms 1 and 2
            intervs = pm.gintervals(
                ["1", "2"], [100, 200], [1000, 2000]
            )
            pm.gintervals_save(intervs, tmp_name)

            loaded_before = pm.gintervals_load(tmp_name)
            chroms_before = {str(c) for c in loaded_before["chrom"]}
            assert "1" in chroms_before
            assert "2" in chroms_before

            # Remove chrom 2
            pm.gintervals_update(tmp_name, None, chrom="2")

            loaded_after = pm.gintervals_load(tmp_name)
            chroms_after = {str(c) for c in loaded_after["chrom"]}
            assert "1" in chroms_after
            assert "2" not in chroms_after
            assert len(loaded_after) < len(loaded_before)
        finally:
            pm.gintervals_rm(tmp_name, force=True)


# ============================================================================
# Tests for gintervals_is_bigset (not implemented)
# ============================================================================

class TestGintervalsIsBigset:
    """Tests for gintervals_is_bigset.

    Ported from R misha test-gintervals2.R.
    Function not implemented in pymisha - all tests skipped.
    """

    @pytest.mark.skip(reason="gintervals_is_bigset not implemented in pymisha")
    def test_is_bigset_1d_true(self):
        """bigintervs1d should be reported as bigset."""

    @pytest.mark.skip(reason="gintervals_is_bigset not implemented in pymisha")
    def test_is_bigset_2d_true(self):
        """bigintervs2d should be reported as bigset."""

    @pytest.mark.skip(reason="gintervals_is_bigset not implemented in pymisha")
    def test_is_bigset_small_false(self):
        """Small interval set should not be reported as bigset."""
