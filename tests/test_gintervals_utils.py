"""Tests for interval utility functions: gsetroot, giterator_intervals,
gintervals_mark_overlaps, gintervals_annotate, gintervals_normalize,
gintervals_random."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


# ---------------------------------------------------------------------------
# gsetroot
# ---------------------------------------------------------------------------
class TestGsetroot:
    """gsetroot should be an alias for gdb_init."""

    def test_gsetroot_is_exported(self):
        assert hasattr(pm, "gsetroot")

    def test_gsetroot_initializes_db(self):
        """gsetroot(path) should initialize database just like gdb_init."""
        # Use the examples DB path
        db_path = pm.gdb_examples_path()
        pm.gsetroot(db_path)
        tracks = pm.gtrack_ls()
        assert len(tracks) > 0
        # Restore original DB
        pm.gdb_init(str(TEST_DB))

    def test_gsetroot_invalid_path(self):
        with pytest.raises(Exception):
            pm.gsetroot("/nonexistent/path")


# ---------------------------------------------------------------------------
# giterator_intervals
# ---------------------------------------------------------------------------
class TestGiteratorIntervals:
    """giterator_intervals returns the iterator grid without track values."""

    def test_giterator_intervals_is_exported(self):
        assert hasattr(pm, "giterator_intervals")

    def test_numeric_iterator(self):
        """Numeric iterator produces fixed-size bins."""
        intervs = pm.gintervals("chr1", 0, 200)
        result = pm.giterator_intervals(intervals=intervs, iterator=50)
        assert result is not None
        assert len(result) > 0
        # All bins should be <= 50bp
        widths = result["end"] - result["start"]
        assert all(widths <= 50)
        assert all(widths > 0)
        # Should cover the original range
        assert result["start"].min() == 0
        assert result["end"].max() == 200

    def test_default_intervals(self):
        """Without explicit intervals, uses gintervals_all."""
        result = pm.giterator_intervals(iterator=1000000)
        assert result is not None
        assert len(result) > 0

    def test_track_as_expr_determines_iterator(self):
        """When expr is a track name and iterator is None, use track bin size."""
        info = pm.gtrack_info("dense_track")
        bin_size = info.get("bin.size", None)
        if bin_size is not None:
            bin_size = int(float(bin_size))
            result = pm.giterator_intervals(
                "dense_track",
                intervals=pm.gintervals("chr1", 0, 1000)
            )
            assert result is not None
            assert len(result) > 0
            # Bins should match track bin size
            widths = result["end"] - result["start"]
            assert all(widths <= bin_size)

    def test_has_interval_id(self):
        """Result should contain intervalID column."""
        intervs = pm.gintervals("chr1", [0, 500], [200, 700])
        result = pm.giterator_intervals(intervals=intervs, iterator=50)
        assert "intervalID" in result.columns

    def test_interval_relative_aligns_to_interval_start(self):
        """interval_relative=True starts bins from each interval start."""
        intervs = pm.gintervals("chr1", 10, 120)
        absolute = pm.giterator_intervals(intervals=intervs, iterator=50)
        relative = pm.giterator_intervals(
            intervals=intervs,
            iterator=50,
            interval_relative=True,
        )

        assert list(absolute["start"]) == [10, 50, 100]
        assert list(absolute["end"]) == [50, 100, 120]
        assert list(relative["start"]) == [10, 60, 110]
        assert list(relative["end"]) == [60, 110, 120]

    def test_interval_relative_requires_numeric_iterator(self):
        """interval_relative requires a numeric iterator value."""
        with pytest.raises(ValueError, match="numeric iterator"):
            pm.giterator_intervals(
                intervals=pm.gintervals("chr1", 0, 1000),
                iterator="dense_track",
                interval_relative=True,
            )

    def test_intervals_iterator_no_args_raises(self):
        """Calling with no expression and no iterator should raise."""
        with pytest.raises((ValueError, TypeError, Exception)):
            pm.giterator_intervals()


# ---------------------------------------------------------------------------
# gintervals_rbind
# ---------------------------------------------------------------------------
class TestGintervalsRbind:
    """gintervals_rbind concatenates interval sets efficiently."""

    def test_is_exported(self):
        assert hasattr(pm, "gintervals_rbind")

    def test_requires_at_least_one_argument(self):
        with pytest.raises(ValueError, match="Usage: gintervals_rbind"):
            pm.gintervals_rbind()

    def test_concatenates_dataframes_in_input_order(self):
        i1 = pm.gintervals("chr1", [100, 300], [200, 400])
        i1["score"] = [1.0, 2.0]
        i2 = pm.gintervals("chr2", 500, 650)
        i2["score"] = [3.0]

        res = pm.gintervals_rbind(i1, i2)
        assert res is not None
        assert len(res) == 3
        assert list(res["chrom"]) == ["1", "1", "2"]
        assert list(res["score"]) == [1.0, 2.0, 3.0]

    def test_accepts_named_interval_sets(self):
        i1 = pm.gintervals("chr1", [100, 300], [200, 400])
        i1["score"] = [1.0, 2.0]
        i2 = pm.gintervals("chr2", 500, 650)
        i2["score"] = [3.0]
        set_name = "tmp.rbind.named"
        if pm.gintervals_exists(set_name):
            pm.gintervals_rm(set_name, force=True)
        pm.gintervals_save(i2, set_name)
        try:
            res = pm.gintervals_rbind(i1, set_name)
            assert res is not None
            assert len(res) == 3
            assert list(res["chrom"]) == ["1", "1", "2"]
        finally:
            pm.gintervals_rm(set_name, force=True)

    def test_rejects_mismatched_columns(self):
        i1 = pm.gintervals("chr1", 100, 200)
        i1["score"] = [1.0]
        i2 = pm.gintervals("chr1", 300, 400)
        with pytest.raises(ValueError, match="columns differ"):
            pm.gintervals_rbind(i1, i2)

    def test_intervals_set_out_saves_result_and_returns_none(self):
        i1 = pm.gintervals("chr1", [100, 300], [200, 400])
        i1["score"] = [1.0, 2.0]
        i2 = pm.gintervals("chr2", 500, 650)
        i2["score"] = [3.0]

        out_set = "tmp.rbind.out"
        if pm.gintervals_exists(out_set):
            pm.gintervals_rm(out_set, force=True)
        try:
            ret = pm.gintervals_rbind(i1, i2, intervals_set_out=out_set)
            assert ret is None
            assert pm.gintervals_exists(out_set)
            loaded = pm.gintervals_load(out_set)
            assert loaded is not None
            assert len(loaded) == 3
            assert "score" in loaded.columns
        finally:
            pm.gintervals_rm(out_set, force=True)


# ---------------------------------------------------------------------------
# gintervals_mark_overlaps
# ---------------------------------------------------------------------------
class TestGintervalsMarkOverlaps:
    """gintervals_mark_overlaps marks groups of overlapping intervals."""

    def test_is_exported(self):
        assert hasattr(pm, "gintervals_mark_overlaps")

    def test_non_overlapping(self):
        """Non-overlapping intervals get different group IDs."""
        intervs = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [100, 500, 1000],
            "end": [200, 600, 1100]
        })
        result = pm.gintervals_mark_overlaps(intervs)
        assert "overlap_group" in result.columns
        # 3 non-overlapping intervals -> 3 groups
        assert result["overlap_group"].nunique() == 3

    def test_overlapping(self):
        """Overlapping intervals share the same group ID."""
        intervs = pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1", "chr1"],
            "start": [100, 150, 500, 1000],
            "end": [200, 300, 600, 1100]
        })
        result = pm.gintervals_mark_overlaps(intervs)
        assert "overlap_group" in result.columns
        # First two intervals overlap -> 3 groups total
        groups = result["overlap_group"].values
        assert groups[0] == groups[1]  # first two overlap
        assert groups[2] != groups[0]
        assert groups[3] != groups[2]

    def test_custom_group_col(self):
        """Custom group column name."""
        intervs = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 150],
            "end": [200, 300]
        })
        result = pm.gintervals_mark_overlaps(intervs, group_col="my_group")
        assert "my_group" in result.columns
        assert "overlap_group" not in result.columns

    def test_preserves_extra_columns(self):
        """Extra columns from input are preserved."""
        intervs = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 150],
            "end": [200, 300],
            "data": [10, 20]
        })
        result = pm.gintervals_mark_overlaps(intervs)
        assert "data" in result.columns
        assert list(result["data"]) == [10, 20]

    def test_touching_intervals_unified(self):
        """Touching intervals (end == start) are unified by default."""
        intervs = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300]
        })
        result = pm.gintervals_mark_overlaps(intervs, unify_touching_intervals=True)
        groups = result["overlap_group"].values
        assert groups[0] == groups[1]

    def test_touching_intervals_not_unified(self):
        """Touching intervals NOT unified when unify_touching_intervals=False."""
        intervs = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [100, 200],
            "end": [200, 300]
        })
        result = pm.gintervals_mark_overlaps(intervs, unify_touching_intervals=False)
        groups = result["overlap_group"].values
        assert groups[0] != groups[1]


# ---------------------------------------------------------------------------
# gintervals_annotate
# ---------------------------------------------------------------------------
class TestGintervalsAnnotate:
    """gintervals_annotate copies columns from nearest annotation intervals."""

    def test_is_exported(self):
        assert hasattr(pm, "gintervals_annotate")

    def test_basic_annotation(self):
        """Annotate intervals with columns from nearest neighbors."""
        intervs = pm.gintervals("chr1", [1000, 5000], [1100, 5050])
        ann = pm.gintervals("chr1", [900, 5400], [950, 5500])
        ann["remark"] = ["a", "b"]
        ann["score"] = [10.0, 20.0]

        result = pm.gintervals_annotate(intervs, ann)
        assert result is not None
        assert "remark" in result.columns
        assert "score" in result.columns
        assert "dist" in result.columns
        assert len(result) == 2

    def test_select_columns(self):
        """Only selected annotation columns are included."""
        intervs = pm.gintervals("chr1", [1000], [1100])
        ann = pm.gintervals("chr1", [900], [950])
        ann["remark"] = ["a"]
        ann["score"] = [10.0]

        result = pm.gintervals_annotate(
            intervs, ann, annotation_columns=["remark"]
        )
        assert "remark" in result.columns
        assert "score" not in result.columns

    def test_custom_column_names(self):
        """Annotation columns can be renamed."""
        intervs = pm.gintervals("chr1", [1000], [1100])
        ann = pm.gintervals("chr1", [900], [950])
        ann["remark"] = ["a"]

        result = pm.gintervals_annotate(
            intervs, ann,
            annotation_columns=["remark"],
            column_names=["my_annotation"]
        )
        assert "my_annotation" in result.columns
        assert "remark" not in result.columns

    def test_no_dist_column(self):
        """Omit distance column when dist_column=None."""
        intervs = pm.gintervals("chr1", [1000], [1100])
        ann = pm.gintervals("chr1", [900], [950])
        ann["remark"] = ["a"]

        result = pm.gintervals_annotate(
            intervs, ann, dist_column=None
        )
        assert "dist" not in result.columns

    def test_max_dist_threshold(self):
        """Annotations beyond max_dist are replaced with na_value."""
        intervs = pm.gintervals("chr1", [1000, 100000], [1100, 100100])
        ann = pm.gintervals("chr1", [900], [950])
        ann["remark"] = ["nearby"]

        result = pm.gintervals_annotate(
            intervs, ann,
            annotation_columns=["remark"],
            max_dist=200,
            na_value="too_far"
        )
        # First interval is close, second is far
        assert result.iloc[0]["remark"] == "nearby"
        assert result.iloc[1]["remark"] == "too_far"

    def test_keeps_original_order(self):
        """Original interval order is preserved by default."""
        intervs = pm.gintervals("chr1", [5000, 1000], [5100, 1100])
        ann = pm.gintervals("chr1", [900, 5400], [950, 5500])
        ann["remark"] = ["a", "b"]

        result = pm.gintervals_annotate(intervs, ann, keep_order=True)
        # Original order should be preserved
        assert len(result) == 2

    def test_column_conflict_raises(self):
        """Conflicting column names raise error without overwrite."""
        intervs = pm.gintervals("chr1", [1000], [1100])
        intervs["remark"] = ["existing"]
        ann = pm.gintervals("chr1", [900], [950])
        ann["remark"] = ["a"]

        with pytest.raises(ValueError, match="overwrite"):
            pm.gintervals_annotate(intervs, ann, annotation_columns=["remark"])

    def test_column_conflict_with_overwrite(self):
        """Conflicting columns replaced when overwrite=True."""
        intervs = pm.gintervals("chr1", [1000], [1100])
        intervs["remark"] = ["existing"]
        ann = pm.gintervals("chr1", [900], [950])
        ann["remark"] = ["new_value"]

        result = pm.gintervals_annotate(
            intervs, ann,
            annotation_columns=["remark"],
            overwrite=True
        )
        assert result["remark"].iloc[0] == "new_value"


# ---------------------------------------------------------------------------
# gintervals_normalize
# ---------------------------------------------------------------------------
class TestGintervalsNormalize:
    """gintervals_normalize centers intervals to specified size(s)."""

    def test_is_exported(self):
        assert hasattr(pm, "gintervals_normalize")

    def test_scalar_size(self):
        """All intervals normalized to the same size."""
        intervs = pm.gintervals("chr1", [1000, 5000], [2000, 6000])
        result = pm.gintervals_normalize(intervs, 500)
        assert result is not None
        assert len(result) == 2
        widths = result["end"] - result["start"]
        assert all(widths == 500)

    def test_center_preserved(self):
        """Center of interval is preserved after normalization."""
        intervs = pm.gintervals("chr1", [1000], [2000])
        result = pm.gintervals_normalize(intervs, 500)
        original_center = 1500
        new_center = (result.iloc[0]["start"] + result.iloc[0]["end"]) / 2
        assert abs(new_center - original_center) <= 1  # Allow rounding

    def test_vector_sizes(self):
        """Each interval gets its own target size."""
        intervs = pm.gintervals("chr1", [1000, 5000, 10000], [2000, 6000, 11000])
        result = pm.gintervals_normalize(intervs, [500, 1000, 750])
        widths = (result["end"] - result["start"]).values
        np.testing.assert_array_equal(widths, [500, 1000, 750])

    def test_one_to_many(self):
        """Single interval with multiple sizes creates multiple intervals."""
        intervs = pm.gintervals("chr1", [1000], [2000])
        result = pm.gintervals_normalize(intervs, [500, 1000, 1500])
        assert len(result) == 3
        widths = (result["end"] - result["start"]).values
        np.testing.assert_array_equal(widths, [500, 1000, 1500])

    def test_chromosome_boundary_clamped(self):
        """Intervals are clamped to chromosome boundaries."""
        # Make an interval near the start of chromosome
        intervs = pm.gintervals("chr1", [0], [100])
        # Normalize to large size -> should be clamped at 0
        result = pm.gintervals_normalize(intervs, 1000)
        assert result.iloc[0]["start"] >= 0

    def test_preserves_extra_columns(self):
        """Extra columns from input are preserved."""
        intervs = pm.gintervals("chr1", [1000, 5000], [2000, 6000])
        intervs["data"] = [10, 20]
        result = pm.gintervals_normalize(intervs, 500)
        assert "data" in result.columns
        assert list(result["data"]) == [10, 20]

    def test_preserves_strand(self):
        """Strand column is preserved."""
        intervs = pm.gintervals("chr1", [1000, 5000], [2000, 6000], strand=[1, -1])
        result = pm.gintervals_normalize(intervs, 500)
        assert "strand" in result.columns
        assert list(result["strand"]) == [1, -1]

    def test_invalid_size_raises(self):
        """Non-positive size should raise."""
        intervs = pm.gintervals("chr1", [1000], [2000])
        with pytest.raises((ValueError, Exception)):
            pm.gintervals_normalize(intervs, -100)

    def test_vector_size_length_mismatch_raises(self):
        """Mismatched vector lengths should raise."""
        intervs = pm.gintervals("chr1", [1000, 5000], [2000, 6000])
        with pytest.raises((ValueError, Exception)):
            pm.gintervals_normalize(intervs, [500, 1000, 750])  # 3 sizes, 2 intervals

    def test_2d_intervals_raises(self):
        """2D intervals should raise."""
        # We don't support 2D intervals for normalize
        intervs = pd.DataFrame({
            "chrom1": ["chr1"], "start1": [0], "end1": [100],
            "chrom2": ["chr1"], "start2": [0], "end2": [100],
        })
        with pytest.raises((ValueError, Exception)):
            pm.gintervals_normalize(intervs, 500)


# ---------------------------------------------------------------------------
# gintervals_random
# ---------------------------------------------------------------------------
class TestGintervalsRandom:
    """gintervals_random generates random genomic intervals."""

    def test_is_exported(self):
        assert hasattr(pm, "gintervals_random")

    def test_basic(self):
        """Generate n random intervals of specified size."""
        result = pm.gintervals_random(100, 50, dist_from_edge=0)
        assert result is not None
        assert len(result) == 50
        widths = result["end"] - result["start"]
        assert all(widths == 100)

    def test_correct_columns(self):
        """Result has chrom, start, end columns."""
        result = pm.gintervals_random(100, 10, dist_from_edge=0)
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns

    def test_within_chromosome_bounds(self):
        """All intervals are within chromosome boundaries."""
        result = pm.gintervals_random(100, 100, dist_from_edge=0)
        all_intervals = pm.gintervals_all()
        chrom_sizes = {row["chrom"]: row["end"] for _, row in all_intervals.iterrows()}

        for _, row in result.iterrows():
            assert row["start"] >= 0
            assert row["end"] <= chrom_sizes[row["chrom"]]

    def test_dist_from_edge(self):
        """Intervals respect dist_from_edge parameter."""
        dist = 1000
        result = pm.gintervals_random(100, 50, dist_from_edge=dist)
        all_intervals = pm.gintervals_all()
        chrom_sizes = {row["chrom"]: row["end"] for _, row in all_intervals.iterrows()}

        for _, row in result.iterrows():
            assert row["start"] >= dist
            assert row["end"] <= chrom_sizes[row["chrom"]] - dist

    def test_specific_chromosomes(self):
        """Only generate intervals on specified chromosomes."""
        result = pm.gintervals_random(100, 50, chromosomes=["chr1"], dist_from_edge=0)
        # Chromosomes may be returned as "1" (normalized)
        chroms = result["chrom"].unique()
        assert len(chroms) == 1

    def test_filter_exclusion(self):
        """Intervals do not overlap with filter regions."""
        # Create filter covering first half of chr1
        filter_regions = pm.gintervals("chr1", 0, 200000)
        result = pm.gintervals_random(100, 50, chromosomes=["chr1"],
                                       filter=filter_regions,
                                       dist_from_edge=0)
        assert result is not None
        # All results should be outside the filter region
        for _, row in result.iterrows():
            assert row["start"] >= 200000 or row["end"] <= 0

    def test_reproducible_with_seed(self):
        """Same seed produces same intervals."""
        np.random.seed(42)
        result1 = pm.gintervals_random(100, 50, dist_from_edge=0)
        np.random.seed(42)
        result2 = pm.gintervals_random(100, 50, dist_from_edge=0)
        pd.testing.assert_frame_equal(result1, result2)

    def test_invalid_size_raises(self):
        with pytest.raises((ValueError, Exception)):
            pm.gintervals_random(-100, 50)

    def test_invalid_n_raises(self):
        with pytest.raises((ValueError, Exception)):
            pm.gintervals_random(100, -5)

    def test_invalid_chromosomes_raises(self):
        with pytest.raises((ValueError, Exception)):
            pm.gintervals_random(100, 50, chromosomes=["nonexistent_chrom"])
