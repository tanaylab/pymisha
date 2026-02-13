"""Tests for gtrack_modify and gtrack_smooth."""

import shutil
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


# ─── gtrack_modify ───────────────────────────────────────────────────

class TestGtrackModify:
    """Tests for gtrack_modify: in-place dense track modification."""

    def test_modify_doubles_values(self, tmp_path):
        """Modify a dense track in-place with expression: track * 2."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            intervs = pm.gintervals("chr1", 0, 1000)
            before = pm.gextract("dense_track", intervs)
            vals_before = before["dense_track"].to_numpy().copy()

            pm.gtrack_modify("dense_track", "dense_track * 2", intervs)
            after = pm.gextract("dense_track", intervs)
            vals_after = after["dense_track"].to_numpy()

            np.testing.assert_allclose(vals_after, vals_before * 2, equal_nan=True)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_restores_values(self, tmp_path):
        """Modify then reverse: track * 2 then track / 2."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            intervs = pm.gintervals("chr1", 0, 500)
            before = pm.gextract("dense_track", intervs)
            vals_before = before["dense_track"].to_numpy().copy()

            pm.gtrack_modify("dense_track", "dense_track * 2", intervs)
            pm.gtrack_modify("dense_track", "dense_track / 2", intervs)
            after = pm.gextract("dense_track", intervs)
            vals_after = after["dense_track"].to_numpy()

            np.testing.assert_allclose(vals_after, vals_before, rtol=1e-5, equal_nan=True)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_subset_intervals(self, tmp_path):
        """Modify only a subset of intervals; rest stays unchanged."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            full_intervs = pm.gintervals("chr1", 0, 2000)
            before_full = pm.gextract("dense_track", full_intervs)
            vals_before = before_full["dense_track"].to_numpy().copy()

            modify_intervs = pm.gintervals("chr1", 300, 800)
            pm.gtrack_modify("dense_track", "dense_track + 100", modify_intervs)

            after_full = pm.gextract("dense_track", full_intervs)
            vals_after = after_full["dense_track"].to_numpy()

            # Identify modified bins
            info = pm.gtrack_info("dense_track")
            int(info["bin_size"])
            starts = before_full["start"].to_numpy()
            mask = (starts >= 300) & (starts < 800)

            # Modified region should have +100
            np.testing.assert_allclose(
                vals_after[mask], vals_before[mask] + 100, rtol=1e-5, equal_nan=True
            )
            # Unmodified region should be unchanged
            np.testing.assert_allclose(
                vals_after[~mask], vals_before[~mask], equal_nan=True
            )
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_with_constant_expression(self, tmp_path):
        """Set all values to a constant."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            intervs = pm.gintervals("chr1", 0, 500)
            pm.gtrack_modify("dense_track", "dense_track * 0 + 42", intervs)

            after = pm.gextract("dense_track", intervs)
            vals_after = after["dense_track"].to_numpy()
            np.testing.assert_allclose(vals_after, 42.0)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_allgenome_default(self, tmp_path):
        """Modify with no intervals (defaults to ALLGENOME)."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            intervs = pm.gintervals_all()
            before = pm.gextract("dense_track", intervs)
            vals_before = before["dense_track"].to_numpy().copy()

            pm.gtrack_modify("dense_track", "dense_track * 0")

            after = pm.gextract("dense_track", intervs)
            vals_after = after["dense_track"].to_numpy()
            # Non-NaN values should become 0; NaN stays NaN
            non_nan = ~np.isnan(vals_before)
            np.testing.assert_allclose(vals_after[non_nan], 0.0)
            assert np.all(np.isnan(vals_after[np.isnan(vals_before)]))
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_rejects_sparse_track(self, tmp_path):
        """gtrack_modify should reject non-dense tracks."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            with pytest.raises(Exception):
                pm.gtrack_modify("sparse_track", "sparse_track * 2")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_rejects_nonexistent_track(self, tmp_path):
        """gtrack_modify should error on nonexistent track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            with pytest.raises(Exception):
                pm.gtrack_modify("no_such_track", "no_such_track * 2")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_updates_created_by_attr(self, tmp_path):
        """gtrack_modify should append to created.by attribute."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            intervs = pm.gintervals("chr1", 0, 500)
            pm.gtrack_modify("dense_track", "dense_track * 2", intervs)
            created_by = pm.gtrack_attr_get("dense_track", "created.by")
            assert "gtrack.modify" in created_by
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_modify_multiple_chromosomes(self, tmp_path):
        """Modify across multiple chromosomes."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            intervs = pm.gintervals_all()
            before = pm.gextract("dense_track", intervs)
            vals_before = before["dense_track"].to_numpy().copy()

            pm.gtrack_modify("dense_track", "dense_track + 1", intervs)
            after = pm.gextract("dense_track", intervs)
            vals_after = after["dense_track"].to_numpy()

            np.testing.assert_allclose(vals_after, vals_before + 1, rtol=1e-5, equal_nan=True)
        finally:
            pm.gdb_init(str(TEST_DB))


# ─── gtrack_smooth ───────────────────────────────────────────────────

class TestGtrackSmooth:
    """Tests for gtrack_smooth: creates smoothed dense track."""

    def test_smooth_creates_track(self, tmp_path):
        """gtrack_smooth should create a new dense track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smoothed_track", "Smooth test", "dense_track", 500)
            assert pm.gtrack_exists("smoothed_track")
            info = pm.gtrack_info("smoothed_track")
            assert info["type"] == "dense"
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_mean_algorithm(self, tmp_path):
        """Smoothing with MEAN algorithm: simple window average."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smooth_mean", "Mean smooth", "dense_track", 500, alg="MEAN")
            assert pm.gtrack_exists("smooth_mean")

            # Smoothed values should be different from original (unless constant)
            intervs = pm.gintervals("chr1", 1000, 5000)
            orig = pm.gextract("dense_track", intervs)["dense_track"].to_numpy()
            smoothed = pm.gextract("smooth_mean", intervs)["smooth_mean"].to_numpy()
            # At least some values should differ
            assert not np.allclose(orig, smoothed, equal_nan=True)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_linear_ramp_algorithm(self, tmp_path):
        """Smoothing with LINEAR_RAMP algorithm: weighted average."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smooth_lr", "LR smooth", "dense_track", 500, alg="LINEAR_RAMP")
            assert pm.gtrack_exists("smooth_lr")

            intervs = pm.gintervals("chr1", 1000, 5000)
            orig = pm.gextract("dense_track", intervs)["dense_track"].to_numpy()
            smoothed = pm.gextract("smooth_lr", intervs)["smooth_lr"].to_numpy()
            assert not np.allclose(orig, smoothed, equal_nan=True)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_reduces_variance(self, tmp_path):
        """Smoothing should reduce variance of the signal."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smooth_var", "Var test", "dense_track", 1000, alg="MEAN")

            intervs = pm.gintervals("chr1", 2000, 50000)
            orig = pm.gextract("dense_track", intervs)["dense_track"].to_numpy()
            smoothed = pm.gextract("smooth_var", intervs)["smooth_var"].to_numpy()

            orig_var = np.nanvar(orig)
            smooth_var = np.nanvar(smoothed)
            assert smooth_var < orig_var
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_nans_false(self, tmp_path):
        """With smooth_nans=False, NaN center produces NaN output."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            # Create a track with some NaN values
            intervs = pm.gintervals("chr1", 0, 1000)
            pm.gtrack_create(
                "nan_track", "Track with NaN",
                "np.where(dense_track > 0.15, dense_track, np.nan)",
                iterator=pm.gtrack_info("dense_track")["bin_size"]
            )

            pm.gtrack_smooth("smooth_nonan", "No NaN smooth", "nan_track", 500, smooth_nans=False)

            # Where nan_track is NaN, smooth_nonan should also be NaN
            out = pm.gextract(["nan_track", "smooth_nonan"], intervs)
            nan_mask = np.isnan(out["nan_track"].to_numpy())
            smooth_vals = out["smooth_nonan"].to_numpy()
            assert np.all(np.isnan(smooth_vals[nan_mask]))
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_nans_true(self, tmp_path):
        """With smooth_nans=True, NaN center can produce non-NaN output."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            # Create a track with some NaN values in the middle (not edges)
            info = pm.gtrack_info("dense_track")
            binsize = int(info["bin_size"])
            pm.gtrack_create(
                "nan_track2", "NaN track",
                "np.where(dense_track > 0.15, dense_track, np.nan)",
                iterator=binsize
            )

            pm.gtrack_smooth("smooth_nan_yes", "Smooth NaN yes", "nan_track2", 500, smooth_nans=True)

            # Some originally-NaN positions should now have values (if surrounded by non-NaN)
            intervs = pm.gintervals("chr1", 2000, 10000)
            out = pm.gextract(["nan_track2", "smooth_nan_yes"], intervs)
            nan_mask = np.isnan(out["nan_track2"].to_numpy())
            smooth_vals = out["smooth_nan_yes"].to_numpy()
            # At least some NaN positions should now be non-NaN
            if nan_mask.any():
                assert not np.all(np.isnan(smooth_vals[nan_mask]))
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_weight_threshold(self, tmp_path):
        """Weight threshold should cause NaN at edges with insufficient data."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            info = pm.gtrack_info("dense_track")
            binsize = int(info["bin_size"])
            # Use MEAN algorithm where weight_thr is an absolute count of non-NaN bins required.
            # winsize=5000, binsize=50 → num_samples_aside=50, window=101 bins.
            # At the first bin, only ~51 bins are available. Threshold of 60 forces NaN.
            pm.gtrack_smooth("smooth_wt", "Weight thr test", "dense_track", 5000,
                             weight_thr=60, alg="MEAN")
            edge = pm.gintervals("chr1", 0, binsize * 2)
            out = pm.gextract("smooth_wt", edge)
            assert np.isnan(out["smooth_wt"].to_numpy()[0])
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_rejects_existing_track(self, tmp_path):
        """gtrack_smooth should reject creating over an existing track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            with pytest.raises(Exception):
                pm.gtrack_smooth("dense_track", "Overwrite attempt", "dense_track", 500)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_invalid_algorithm(self, tmp_path):
        """gtrack_smooth should reject invalid algorithm names."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            with pytest.raises(Exception):
                pm.gtrack_smooth("bad_alg", "Bad alg", "dense_track", 500, alg="INVALID")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_preserves_binsize(self, tmp_path):
        """Output track should have same bin size as input."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            orig_info = pm.gtrack_info("dense_track")
            pm.gtrack_smooth("smooth_bs", "Binsize test", "dense_track", 500)
            new_info = pm.gtrack_info("smooth_bs")
            assert int(new_info["bin_size"]) == int(orig_info["bin_size"])
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_sets_attributes(self, tmp_path):
        """gtrack_smooth should set description and created.by attributes."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smooth_attr", "My description", "dense_track", 500)
            desc = pm.gtrack_attr_get("smooth_attr", "description")
            created_by = pm.gtrack_attr_get("smooth_attr", "created.by")
            assert desc == "My description"
            assert "gtrack.smooth" in created_by
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_with_explicit_iterator(self, tmp_path):
        """gtrack_smooth with explicit iterator bin size."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            info = pm.gtrack_info("dense_track")
            binsize = int(info["bin_size"])
            pm.gtrack_smooth("smooth_iter", "Iter test", "dense_track", 500, iterator=binsize)
            assert pm.gtrack_exists("smooth_iter")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_mean_vs_linear_ramp_differ(self, tmp_path):
        """MEAN and LINEAR_RAMP should produce different results."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("s_mean", "Mean", "dense_track", 500, alg="MEAN")
            pm.gtrack_smooth("s_lr", "LR", "dense_track", 500, alg="LINEAR_RAMP")

            intervs = pm.gintervals("chr1", 2000, 10000)
            mean_vals = pm.gextract("s_mean", intervs)["s_mean"].to_numpy()
            lr_vals = pm.gextract("s_lr", intervs)["s_lr"].to_numpy()
            # They should produce different values (unless the input is constant)
            assert not np.allclose(mean_vals, lr_vals, equal_nan=True)
        finally:
            pm.gdb_init(str(TEST_DB))


class TestGtrackSmoothRParity:
    """R misha parity tests ported from test-gtrack.smooth.R."""

    def test_smooth_fixedbin_linear_ramp_golden(self, tmp_path):
        """Port of: gtrack.smooth with test.fixedbin using LINEAR_RAMP — golden-master."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smooth_lr", "", "dense_track", 10000, alg="LINEAR_RAMP")
            r = pm.gextract("smooth_lr", pm.gintervals([1, 2], 0, -1))
            assert len(r) == 16000
            vals = r["smooth_lr"].to_numpy()
            # Verify specific golden-master values
            np.testing.assert_allclose(vals[0], 0.05978947, rtol=1e-4)
            np.testing.assert_allclose(np.nanmean(vals), 0.094021, rtol=1e-3)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_fixedbin_mean_golden(self, tmp_path):
        """Port of: gtrack.smooth with test.fixedbin using MEAN — golden-master."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            pm.gtrack_smooth("smooth_mean", "", "dense_track", 10000, alg="MEAN")
            r = pm.gextract("smooth_mean", pm.gintervals([1, 2], 0, -1))
            assert len(r) == 16000
            vals = r["smooth_mean"].to_numpy()
            np.testing.assert_allclose(vals[0], 0.05086908, rtol=1e-4)
            np.testing.assert_allclose(np.nanmean(vals), 0.094017, rtol=1e-3)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_smooth_rects_track_error(self, tmp_path):
        """Port of: gtrack.smooth with test.rects using LINEAR_RAMP — expect error."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            with pytest.raises(Exception):
                pm.gtrack_smooth("smooth_rects", "", "rects_track", 10000, alg="LINEAR_RAMP")
        finally:
            pm.gdb_init(str(TEST_DB))
