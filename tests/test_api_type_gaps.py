"""TDD red-phase tests for API type gaps between PyMisha and R misha.

Gap 1: Interval set operations should accept string interval set names.
Gap 2: giterator_intervals() missing partial_bins parameter.
Gap 4: gtrack_create() band parameter raises ValueError instead of working.
Gap 5: gintervals_covered_bp() missing optional src parameter.
"""

import contextlib

import pandas as pd
import pytest

import pymisha as pm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_interval_sets(*names):
    """Remove named interval sets, ignoring errors."""
    for name in names:
        with contextlib.suppress(Exception):
            pm.gintervals_rm(name, force=True)


def _cleanup_tracks(*names):
    """Remove tracks, ignoring errors."""
    for name in names:
        with contextlib.suppress(Exception):
            pm.gtrack_rm(name, force=True)


# ===========================================================================
# Gap 1 — String interval set names accepted by set operations
# ===========================================================================

class TestGintervalsIntersectStringName:
    """gintervals_intersect should accept string interval set names."""

    ISET1 = "test_api_gaps_intersect1"
    ISET2 = "test_api_gaps_intersect2"

    def test_gintervals_intersect_string_name(self):
        intervals1 = pm.gintervals("1", 0, 1000)
        intervals2 = pm.gintervals("1", 500, 1500)
        try:
            pm.gintervals_save(intervals1, self.ISET1)
            pm.gintervals_save(intervals2, self.ISET2)

            result_str = pm.gintervals_intersect(self.ISET1, self.ISET2)
            result_df = pm.gintervals_intersect(intervals1, intervals2)

            pd.testing.assert_frame_equal(
                result_str.reset_index(drop=True),
                result_df.reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            _cleanup_interval_sets(self.ISET1, self.ISET2)


class TestGintervalsUnionStringName:
    """gintervals_union should accept string interval set names."""

    ISET1 = "test_api_gaps_union1"
    ISET2 = "test_api_gaps_union2"

    def test_gintervals_union_string_name(self):
        intervals1 = pm.gintervals("1", 0, 1000)
        intervals2 = pm.gintervals("1", 500, 1500)
        try:
            pm.gintervals_save(intervals1, self.ISET1)
            pm.gintervals_save(intervals2, self.ISET2)

            result_str = pm.gintervals_union(self.ISET1, self.ISET2)
            result_df = pm.gintervals_union(intervals1, intervals2)

            pd.testing.assert_frame_equal(
                result_str.reset_index(drop=True),
                result_df.reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            _cleanup_interval_sets(self.ISET1, self.ISET2)


class TestGintervalsDiffStringName:
    """gintervals_diff should accept string interval set names."""

    ISET1 = "test_api_gaps_diff1"
    ISET2 = "test_api_gaps_diff2"

    def test_gintervals_diff_string_name(self):
        intervals1 = pm.gintervals("1", 0, 1000)
        intervals2 = pm.gintervals("1", 500, 1500)
        try:
            pm.gintervals_save(intervals1, self.ISET1)
            pm.gintervals_save(intervals2, self.ISET2)

            result_str = pm.gintervals_diff(self.ISET1, self.ISET2)
            result_df = pm.gintervals_diff(intervals1, intervals2)

            pd.testing.assert_frame_equal(
                result_str.reset_index(drop=True),
                result_df.reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            _cleanup_interval_sets(self.ISET1, self.ISET2)


class TestGintervalsCanonicStringName:
    """gintervals_canonic should accept a string interval set name."""

    ISET = "test_api_gaps_canonic"

    def test_gintervals_canonic_string_name(self):
        # Overlapping intervals to make canonic meaningful
        intervals = pm.gintervals("1", [0, 500], [700, 1200])
        try:
            pm.gintervals_save(intervals, self.ISET)

            result_str = pm.gintervals_canonic(self.ISET)
            result_df = pm.gintervals_canonic(intervals)

            pd.testing.assert_frame_equal(
                result_str.reset_index(drop=True),
                result_df.reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            _cleanup_interval_sets(self.ISET)


class TestGintervalsNeighborsStringName:
    """gintervals_neighbors should accept string interval set names."""

    ISET1 = "test_api_gaps_neighbors1"
    ISET2 = "test_api_gaps_neighbors2"

    def test_gintervals_neighbors_string_name(self):
        intervals1 = pm.gintervals("1", 0, 500)
        intervals2 = pm.gintervals("1", 1000, 1500)
        try:
            pm.gintervals_save(intervals1, self.ISET1)
            pm.gintervals_save(intervals2, self.ISET2)

            result_str = pm.gintervals_neighbors(self.ISET1, self.ISET2)
            result_df = pm.gintervals_neighbors(intervals1, intervals2)

            pd.testing.assert_frame_equal(
                result_str.reset_index(drop=True),
                result_df.reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            _cleanup_interval_sets(self.ISET1, self.ISET2)


class TestGintervalsCoveredBpStringName:
    """gintervals_covered_bp should accept a string interval set name."""

    ISET = "test_api_gaps_coveredbp"

    def test_gintervals_covered_bp_string_name(self):
        intervals = pm.gintervals("1", [0, 200], [300, 600])
        try:
            pm.gintervals_save(intervals, self.ISET)

            result_str = pm.gintervals_covered_bp(self.ISET)
            result_df = pm.gintervals_covered_bp(intervals)

            assert result_str == result_df
        finally:
            _cleanup_interval_sets(self.ISET)


class TestGintervalsCoverageFractionStringName:
    """gintervals_coverage_fraction should accept string interval set names."""

    ISET1 = "test_api_gaps_covfrac1"
    ISET2 = "test_api_gaps_covfrac2"

    def test_gintervals_coverage_fraction_string_name(self):
        intervals1 = pm.gintervals("1", 0, 1000)
        intervals2 = pm.gintervals("1", 0, 2000)
        try:
            pm.gintervals_save(intervals1, self.ISET1)
            pm.gintervals_save(intervals2, self.ISET2)

            result_str = pm.gintervals_coverage_fraction(self.ISET1, self.ISET2)
            result_df = pm.gintervals_coverage_fraction(intervals1, intervals2)

            assert result_str == pytest.approx(result_df)
        finally:
            _cleanup_interval_sets(self.ISET1, self.ISET2)


class TestGintervalsForceRangeStringName:
    """gintervals_force_range should accept a string interval set name."""

    ISET = "test_api_gaps_forcerange"

    def test_gintervals_force_range_string_name(self):
        # Use valid intervals (force_range is a no-op here but still tests the
        # string name dispatch path).
        intervals = pm.gintervals("1", 0, 1000)
        try:
            pm.gintervals_save(intervals, self.ISET)

            result_str = pm.gintervals_force_range(self.ISET)
            result_df = pm.gintervals_force_range(intervals)

            pd.testing.assert_frame_equal(
                result_str.reset_index(drop=True),
                result_df.reset_index(drop=True),
                check_dtype=False,
            )
        finally:
            _cleanup_interval_sets(self.ISET)


# ===========================================================================
# Gap 2 — giterator_intervals partial_bins parameter
# ===========================================================================

class TestGiteratorIntervalsPartialBins:
    """giterator_intervals should support partial_bins parameter."""

    def test_giterator_intervals_partial_bins_clip(self):
        """Default behavior (clip): last partial bin is truncated at interval boundary."""
        intervals = pm.gintervals("1", 0, 550)
        result = pm.giterator_intervals(
            intervals=intervals, iterator=200, partial_bins="clip",
        )
        assert result is not None
        assert len(result) == 3
        # Bins: [0,200), [200,400), [400,550)
        assert result["start"].iloc[0] == 0
        assert result["end"].iloc[0] == 200
        assert result["start"].iloc[1] == 200
        assert result["end"].iloc[1] == 400
        assert result["start"].iloc[2] == 400
        assert result["end"].iloc[2] == 550  # clipped

    def test_giterator_intervals_partial_bins_drop(self):
        """partial_bins='drop' should drop partial bins."""
        intervals = pm.gintervals("1", 0, 550)
        result = pm.giterator_intervals(
            intervals=intervals, iterator=200, partial_bins="drop",
        )
        assert result is not None
        # Only full bins: [0,200), [200,400) — partial [400,550) dropped
        assert len(result) == 2
        assert result["start"].iloc[0] == 0
        assert result["end"].iloc[0] == 200
        assert result["start"].iloc[1] == 200
        assert result["end"].iloc[1] == 400

    def test_giterator_intervals_partial_bins_exact(self):
        """partial_bins='exact' should behave like 'drop'."""
        intervals = pm.gintervals("1", 0, 550)
        result = pm.giterator_intervals(
            intervals=intervals, iterator=200, partial_bins="exact",
        )
        assert result is not None
        # Same as drop: only full bins
        assert len(result) == 2
        assert result["start"].iloc[0] == 0
        assert result["end"].iloc[0] == 200
        assert result["start"].iloc[1] == 200
        assert result["end"].iloc[1] == 400

    def test_giterator_intervals_partial_bins_default_is_clip(self):
        """Without partial_bins, default behavior should be clip (matches R)."""
        intervals = pm.gintervals("1", 0, 550)
        result_default = pm.giterator_intervals(
            intervals=intervals, iterator=200,
        )
        result_clip = pm.giterator_intervals(
            intervals=intervals, iterator=200, partial_bins="clip",
        )
        pd.testing.assert_frame_equal(
            result_default.reset_index(drop=True),
            result_clip.reset_index(drop=True),
            check_dtype=False,
        )

    def test_giterator_intervals_partial_bins_invalid(self):
        """Invalid partial_bins value should raise ValueError."""
        intervals = pm.gintervals("1", 0, 550)
        with pytest.raises(ValueError):
            pm.giterator_intervals(
                intervals=intervals, iterator=200, partial_bins="invalid",
            )

    def test_giterator_intervals_partial_bins_no_partial(self):
        """When interval divides evenly, all modes should produce the same result."""
        intervals = pm.gintervals("1", 0, 600)
        result_clip = pm.giterator_intervals(
            intervals=intervals, iterator=200, partial_bins="clip",
        )
        result_drop = pm.giterator_intervals(
            intervals=intervals, iterator=200, partial_bins="drop",
        )
        # 3 full bins, no partial
        assert len(result_clip) == 3
        pd.testing.assert_frame_equal(
            result_clip.reset_index(drop=True),
            result_drop.reset_index(drop=True),
            check_dtype=False,
        )


# ===========================================================================
# Gap 4 — gtrack_create band parameter should work
# ===========================================================================

class TestGtrackCreateBand:
    """gtrack_create should accept and handle the band parameter."""

    TRACK_NAME = "test_api_gaps_band_track"

    def test_gtrack_create_with_band_does_not_raise(self):
        """band parameter should not raise ValueError — it should be supported."""
        try:
            # This currently raises ValueError("band is not supported in pymisha gtrack_create yet")
            # After the fix, it should either succeed or fail for a legitimate reason
            # (e.g., missing 2D track), but NOT with "band is not supported".
            pm.gtrack_create(
                self.TRACK_NAME,
                "band test track",
                "dense_track",
                iterator=200,
                band=(0, 100000),
            )
        except ValueError as e:
            if "not supported" in str(e):
                pytest.fail(
                    f"band parameter still raises 'not supported' ValueError: {e}"
                )
            # Other ValueErrors (e.g., from C++ layer) are acceptable
            # since band with a 1D expression may not be valid
        finally:
            _cleanup_tracks(self.TRACK_NAME)

    def test_gtrack_create_band_none_still_works(self):
        """Passing band=None should still work as before (no regression)."""
        try:
            pm.gtrack_create(
                self.TRACK_NAME,
                "no band test",
                "dense_track",
                iterator=200,
                band=None,
            )
            info = pm.gtrack_info(self.TRACK_NAME)
            assert info is not None
        finally:
            _cleanup_tracks(self.TRACK_NAME)


# ===========================================================================
# Gap 5 — gintervals_covered_bp optional src parameter
# ===========================================================================

class TestGintervalsCoveredBpSrc:
    """gintervals_covered_bp should accept an optional src parameter."""

    def test_gintervals_covered_bp_with_src(self):
        """intervals=[0,1000), src=[500,1500) -> covered bp = 500."""
        intervals = pm.gintervals("1", 0, 1000)
        src = pm.gintervals("1", 500, 1500)
        result = pm.gintervals_covered_bp(intervals, src=src)
        assert result == 500

    def test_gintervals_covered_bp_with_src_no_overlap(self):
        """intervals=[0,100), src=[200,300) -> 0."""
        intervals = pm.gintervals("1", 0, 100)
        src = pm.gintervals("1", 200, 300)
        result = pm.gintervals_covered_bp(intervals, src=src)
        assert result == 0

    def test_gintervals_covered_bp_with_src_partial_overlap(self):
        """intervals=[0,500)+[600,1000), src=[300,700) -> 300 (300-500 + 600-700)."""
        intervals = pm.gintervals("1", [0, 600], [500, 1000])
        src = pm.gintervals("1", 300, 700)
        result = pm.gintervals_covered_bp(intervals, src=src)
        assert result == 300

    def test_gintervals_covered_bp_without_src(self):
        """Existing behavior without src should be preserved."""
        intervals = pm.gintervals("1", [0, 200], [300, 600])
        result = pm.gintervals_covered_bp(intervals)
        # [0,300) merged with [200,600) -> [0,600) = 600 bp
        assert result == 600

    def test_gintervals_covered_bp_src_string_name(self):
        """src can also be a string interval set name."""
        iset_name = "test_api_gaps_covbp_src"
        intervals = pm.gintervals("1", 0, 1000)
        src = pm.gintervals("1", 500, 1500)
        try:
            pm.gintervals_save(src, iset_name)
            result = pm.gintervals_covered_bp(intervals, src=iset_name)
            assert result == 500
        finally:
            _cleanup_interval_sets(iset_name)
