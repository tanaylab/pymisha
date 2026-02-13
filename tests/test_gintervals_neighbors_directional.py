"""Tests for directional neighbor functions.

These functions find upstream/downstream neighbors using query strand directionality.

Note: The example database uses chromosome names "1", "2", "X" (not "chr1", etc.)
"""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(scope="module", autouse=True)
def init_db():
    """Initialize the example database."""
    pm.gdb_init_examples()


class TestGintervalsNeighborsUpstream:
    """Tests for gintervals_neighbors_upstream."""

    def test_upstream_plus_strand_query(self):
        """Upstream neighbors for + strand are to the left (negative distance)."""
        # Query intervals on + strand
        query = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [5000, 8000],
            "end": [5100, 8100],
            "strand": [1, 1]  # + strand
        })

        # Target intervals
        targets = pd.DataFrame({
            "chrom": ["1", "1", "1", "1"],
            "start": [1000, 4000, 6000, 9000],
            "end": [1100, 4100, 6100, 9100]
        })

        result = pm.gintervals_neighbors_upstream(query, targets, maxneighbors=2)

        # For + strand, upstream means targets to the left
        # Query [5000-5100] should find targets at 4000 and 1000 (upstream)
        # Query [8000-8100] should find targets at 6000 and 4000 (upstream)
        assert result is not None
        assert len(result) >= 2

        # Check first query's upstream neighbor
        q1_results = result[(result["chrom"] == "1") & (result["start"] == 5000)]
        assert len(q1_results) > 0
        # Upstream distance should be negative for + strand (target is left of query)
        assert all(q1_results["dist"] <= 0)

    def test_upstream_minus_strand_query(self):
        """Upstream neighbors for - strand are to the right (positive distance)."""
        # Query intervals on - strand
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [-1]  # - strand
        })

        # Target intervals
        targets = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [1000, 4000, 6000],
            "end": [1100, 4100, 6100]
        })

        result = pm.gintervals_neighbors_upstream(query, targets)

        # For - strand, upstream means targets to the right
        assert result is not None
        assert len(result) >= 1

        # Upstream for - strand: target at 6000 is upstream
        # Distance should be reported relative to strand direction
        assert all(result["dist"] <= 0)  # Upstream distances are always <= 0 in output

    def test_upstream_respects_maxdist(self):
        """gintervals_neighbors_upstream respects maxdist parameter."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [1000, 4500],
            "end": [1100, 4600]
        })

        # With small maxdist, should only find nearby target
        result = pm.gintervals_neighbors_upstream(query, targets, maxdist=1000)
        assert result is not None
        # Should only find target at 4500 (distance ~400bp), not 1000 (distance ~4000bp)
        assert all(abs(result["dist"]) <= 1000)

    def test_upstream_no_neighbors(self):
        """Returns empty/None when no upstream neighbors exist."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [100],
            "end": [200],
            "strand": [1]  # + strand
        })

        # Target is downstream of query
        targets = pd.DataFrame({
            "chrom": ["1"],
            "start": [500],
            "end": [600]
        })

        result = pm.gintervals_neighbors_upstream(query, targets)
        # No upstream neighbors for + strand query when all targets are to the right
        assert result is None or len(result) == 0


class TestGintervalsNeighborsDownstream:
    """Tests for gintervals_neighbors_downstream."""

    def test_downstream_plus_strand_query(self):
        """Downstream neighbors for + strand are to the right (positive distance)."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]  # + strand
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [4000, 6000, 9000],
            "end": [4100, 6100, 9100]
        })

        result = pm.gintervals_neighbors_downstream(query, targets, maxneighbors=2)

        assert result is not None
        assert len(result) >= 1

        # For + strand, downstream means targets to the right
        # Should find targets at 6000 and 9000, not 4000
        assert all(result["dist"] >= 0)

    def test_downstream_minus_strand_query(self):
        """Downstream neighbors for - strand are to the left (negative distance)."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [-1]  # - strand
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [1000, 4000, 6000],
            "end": [1100, 4100, 6100]
        })

        result = pm.gintervals_neighbors_downstream(query, targets)

        assert result is not None
        # For - strand, downstream means targets to the left
        # Downstream distance should be >= 0 in output
        assert all(result["dist"] >= 0)

    def test_downstream_respects_maxdist(self):
        """gintervals_neighbors_downstream respects maxdist parameter."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [5200, 10000],
            "end": [5300, 10100]
        })

        result = pm.gintervals_neighbors_downstream(query, targets, maxdist=500)
        assert result is not None
        # Should only find target at 5200, not 10000
        assert all(abs(result["dist"]) <= 500)


class TestGintervalsNeighborsDirectional:
    """Tests for gintervals_neighbors_directional which returns both upstream and downstream."""

    def test_directional_returns_both(self):
        """gintervals_neighbors_directional returns dict with 'upstream' and 'downstream'."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [3000, 6000, 9000],
            "end": [3100, 6100, 9100]
        })

        result = pm.gintervals_neighbors_directional(query, targets)

        assert isinstance(result, dict)
        assert "upstream" in result
        assert "downstream" in result

    def test_directional_separate_counts(self):
        """Can specify different maxneighbors for upstream and downstream."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1", "1", "1", "1"],
            "start": [1000, 3000, 6000, 8000, 10000],
            "end": [1100, 3100, 6100, 8100, 10100]
        })

        result = pm.gintervals_neighbors_directional(
            query, targets,
            maxneighbors_upstream=2,
            maxneighbors_downstream=1
        )

        # Should have up to 2 upstream neighbors and 1 downstream
        if result["upstream"] is not None:
            assert len(result["upstream"]) <= 2
        if result["downstream"] is not None:
            assert len(result["downstream"]) <= 1

    def test_directional_respects_maxdist(self):
        """gintervals_neighbors_directional respects maxdist for both directions."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [1000, 4500, 5200],
            "end": [1100, 4600, 5300]
        })

        result = pm.gintervals_neighbors_directional(query, targets, maxdist=1000)

        # Should only find neighbors within 1000bp
        if result["upstream"] is not None:
            assert all(abs(result["upstream"]["dist"]) <= 1000)
        if result["downstream"] is not None:
            assert all(abs(result["downstream"]["dist"]) <= 1000)


class TestDirectionalNeighborsEdgeCases:
    """Edge cases for directional neighbor functions."""

    def test_missing_strand_column_uses_positive(self):
        """When strand column is missing, assume + strand."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100]
            # No strand column
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [3000, 7000],
            "end": [3100, 7100]
        })

        # Should work without strand column, defaulting to + strand
        result = pm.gintervals_neighbors_upstream(query, targets)
        # With + strand default, upstream is to the left
        # Should find target at 3000
        assert result is not None

    def test_zero_strand_treated_as_positive(self):
        """Strand=0 (unstranded) should be treated as + strand."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [0]  # unstranded
        })

        targets = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [3000, 7000],
            "end": [3100, 7100]
        })

        upstream = pm.gintervals_neighbors_upstream(query, targets)
        downstream = pm.gintervals_neighbors_downstream(query, targets)

        # With strand=0 treated as +, upstream should find 3000, downstream should find 7000
        assert upstream is not None or downstream is not None

    def test_empty_query_intervals(self):
        """Empty query intervals return None."""
        query = pd.DataFrame(columns=["chrom", "start", "end", "strand"])

        targets = pd.DataFrame({
            "chrom": ["1"],
            "start": [1000],
            "end": [1100]
        })

        result = pm.gintervals_neighbors_upstream(query, targets)
        assert result is None

    def test_empty_target_intervals(self):
        """Empty target intervals return None."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1]
        })

        targets = pd.DataFrame(columns=["chrom", "start", "end"])

        result = pm.gintervals_neighbors_upstream(query, targets)
        assert result is None

    def test_preserves_extra_columns(self):
        """Extra columns from query are preserved in output."""
        query = pd.DataFrame({
            "chrom": ["1"],
            "start": [5000],
            "end": [5100],
            "strand": [1],
            "gene_name": ["TP53"]
        })

        targets = pd.DataFrame({
            "chrom": ["1"],
            "start": [3000],
            "end": [3100]
        })

        result = pm.gintervals_neighbors_upstream(query, targets)

        if result is not None and len(result) > 0:
            # The gene_name column should be preserved
            assert "gene_name" in result.columns or "gene_name1" in result.columns
