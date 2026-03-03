"""Tests for gintervals_annotate tie_method parameter."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _make_intervs(chrom, starts, ends, **extra_cols):
    """Build an intervals DataFrame without gintervals sorting."""
    df = pd.DataFrame({
        "chrom": [chrom] * len(starts),
        "start": starts,
        "end": ends,
    })
    for k, v in extra_cols.items():
        df[k] = v
    return df


class TestAnnotateTieMethod:
    """Test tie_method parameter in gintervals_annotate."""

    def test_default_tie_method_unchanged(self):
        """Default behavior (tie_method='first') matches pre-existing behavior."""
        intervs = pm.gintervals("1", [1000], [1100])
        ann = pm.gintervals("1", [900, 5400], [950, 5500])
        ann["label"] = ["near", "far"]

        result = pm.gintervals_annotate(
            intervs, ann, annotation_columns=["label"]
        )
        assert result is not None
        assert len(result) == 1
        assert result["label"].iloc[0] == "near"

    def test_tie_method_min_start_equidistant(self):
        """min.start picks the neighbor with smaller start when equidistant."""
        # Query: [5000, 5100).
        # Two neighbors both end at 4900: dist = -(5000 - 4900) = -100 for both
        # Neighbor at start=4800, Neighbor at start=4700
        # min.start => start=4700 first
        query = pm.gintervals("1", [5000], [5100])
        # Use DataFrame directly to control label assignment
        ann = _make_intervs("1", [4800, 4700], [4900, 4900], label=["B", "A"])

        result = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.start",
        )
        assert len(result) == 2
        labels = result["label"].tolist()
        # Both have dist=-100.  A has start=4700 < B start=4800.
        # min.start => A first.
        assert labels == ["A", "B"]

    def test_tie_method_min_start_reverses_default_order(self):
        """min.start reorders results when default order differs."""
        # Query: [5000, 5100).
        # Two overlapping neighbors (dist=0 for both):
        # Neighbor X: [4900, 5050) -> dist=0, start=4900
        # Neighbor Y: [4950, 5020) -> dist=0, start=4950
        # Default order from C++ may be arbitrary.
        # min.start => X (start=4900) before Y (start=4950)
        query = pm.gintervals("1", [5000], [5100])
        ann = _make_intervs("1", [4950, 4900], [5020, 5050], label=["Y", "X"])

        result = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.start",
        )
        assert len(result) == 2
        labels = result["label"].tolist()
        assert labels == ["X", "Y"]

    def test_tie_method_min_end_equidistant(self):
        """min.end picks the neighbor with smaller end when equidistant."""
        # Query: [5000, 5100).
        # Two overlapping neighbors (dist=0 for both):
        # Neighbor A: [4900, 5200) -> dist=0, end=5200
        # Neighbor B: [4900, 5150) -> dist=0, end=5150
        # min.end => B (end=5150) before A (end=5200)
        query = pm.gintervals("1", [5000], [5100])
        ann = _make_intervs("1", [4900, 4900], [5200, 5150], label=["A", "B"])

        result = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.end",
        )
        assert len(result) == 2
        labels = result["label"].tolist()
        assert labels == ["B", "A"]

    def test_tie_method_no_ties_same_result(self):
        """When no ties exist, tie_method should not change result content."""
        query = pm.gintervals("1", [5000], [5100])
        ann = pm.gintervals("1", [4900, 6000], [4950, 6100])
        ann["label"] = ["near", "far"]

        result_first = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="first",
        )
        result_min_start = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.start",
        )
        result_min_end = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.end",
        )

        # All should return "near" first (closer), then "far"
        assert result_first["label"].tolist() == ["near", "far"]
        assert result_min_start["label"].tolist() == ["near", "far"]
        assert result_min_end["label"].tolist() == ["near", "far"]

    def test_invalid_tie_method_raises(self):
        """Invalid tie_method value raises ValueError."""
        query = pm.gintervals("1", [5000], [5100])
        ann = pm.gintervals("1", [4900], [4950])
        ann["label"] = ["a"]

        with pytest.raises(ValueError, match="tie_method"):
            pm.gintervals_annotate(
                query, ann,
                annotation_columns=["label"],
                tie_method="invalid",
            )

    def test_tie_method_with_maxneighbors_1(self):
        """tie_method is accepted but has no effect when maxneighbors=1."""
        query = pm.gintervals("1", [5000], [5100])
        ann = _make_intervs("1", [4800, 4700], [4900, 4900], label=["B", "A"])

        result = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=1,
            tie_method="min.start",
        )
        # Only 1 neighbor returned regardless
        assert len(result) == 1

    def test_tie_method_min_start_multiple_queries(self):
        """min.start works correctly with multiple query intervals."""
        queries = pm.gintervals("1", [5000, 10000], [5100, 10100])

        # Two overlapping annotations per query:
        # Query 1 (5000-5100): overlap with [4900, 5050) start=4900 and [4950, 5020) start=4950
        # Query 2 (10000-10100): overlap with [9900, 10050) start=9900 and [9950, 10020) start=9950
        ann = _make_intervs(
            "1",
            [4950, 4900, 9950, 9900],
            [5020, 5050, 10020, 10050],
            label=["B", "A", "D", "C"],
        )

        result = pm.gintervals_annotate(
            queries, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.start",
        )
        assert len(result) == 4
        labels = result["label"].tolist()
        # Query 1: A (start=4900) before B (start=4950)
        # Query 2: C (start=9900) before D (start=9950)
        assert labels == ["A", "B", "C", "D"]

    def test_tie_method_with_dist_column(self):
        """Distance column values are preserved correctly with tie-breaking."""
        query = pm.gintervals("1", [5000], [5100])
        ann = _make_intervs("1", [4900, 4950], [5050, 5020], label=["X", "Y"])

        result = pm.gintervals_annotate(
            query, ann,
            annotation_columns=["label"],
            maxneighbors=2,
            tie_method="min.start",
            dist_column="d",
        )
        assert len(result) == 2
        # Both overlap -> dist=0
        np.testing.assert_array_equal(result["d"].values, [0, 0])
        # X first (start=4900 < 4950)
        assert result["label"].tolist() == ["X", "Y"]
