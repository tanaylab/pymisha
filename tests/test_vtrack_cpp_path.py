"""Tests verifying that vtrack expressions go through the C++ pm_screen/pm_extract path.

The key strategy: compare results from the C++ parallel path (multitasking=True)
against the serial C++ path (multitasking=False). Both should produce identical
results when vtracks are routed through C++.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def _sort_df(df):
    """Sort DataFrame by chrom/start/end for deterministic comparison."""
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


class TestVtrackCppPath:
    """Verify that eligible vtracks go through the C++ scanner path."""

    def setup_method(self):
        self._saved_mt = pm.CONFIG["multitasking"]
        self._saved_mp = pm.CONFIG.get("max_processes", 20)
        self._saved_progress = pm.CONFIG.get("progress", True)
        pm.CONFIG["progress"] = False

    def teardown_method(self):
        pm.CONFIG["multitasking"] = self._saved_mt
        pm.CONFIG["max_processes"] = self._saved_mp
        pm.CONFIG["progress"] = self._saved_progress
        pm.gvtrack_clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_serial(self, func, *args, **kwargs):
        """Run function with multitasking disabled."""
        pm.CONFIG["multitasking"] = False
        pm.CONFIG["max_processes"] = 1
        return func(*args, **kwargs)

    def _run_parallel(self, func, *args, **kwargs):
        """Run function with multitasking enabled."""
        pm.CONFIG["multitasking"] = True
        pm.CONFIG["max_processes"] = 4
        return func(*args, **kwargs)

    def _assert_frames_equal(self, serial, parallel, msg=""):
        """Assert two DataFrames are equal after sorting."""
        assert serial is not None, f"serial result is None {msg}"
        assert parallel is not None, f"parallel result is None {msg}"
        assert len(serial) == len(parallel), (
            f"length mismatch: serial={len(serial)}, parallel={len(parallel)} {msg}"
        )
        s = _sort_df(serial)
        p = _sort_df(parallel)
        # Compare coordinates
        pd.testing.assert_frame_equal(
            s[["chrom", "start", "end"]],
            p[["chrom", "start", "end"]],
            check_dtype=False,
            obj=f"coordinates {msg}",
        )
        # Compare data columns with tolerance for floats
        data_cols = [
            c for c in s.columns if c not in {"chrom", "start", "end", "intervalID"}
        ]
        for col in data_cols:
            npt.assert_allclose(
                s[col].to_numpy(dtype=float),
                p[col].to_numpy(dtype=float),
                equal_nan=True,
                err_msg=f"column {col} {msg}",
            )

    # ------------------------------------------------------------------
    # 1. PWM vtrack: gscreen serial vs parallel
    # ------------------------------------------------------------------

    def test_gscreen_pwm_vtrack_matches_serial(self):
        """PWM vtrack through gscreen: parallel must match serial."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        pm.gvtrack_create("pwm_vt", None, func="pwm", pssm=pssm, bidirect=False, prior=0)

        intervals = pm.gintervals("1", 0, 100000)

        serial = self._run_serial(
            pm.gscreen, "pwm_vt > 0.5", intervals, iterator=200, progress=False
        )
        parallel = self._run_parallel(
            pm.gscreen, "pwm_vt > 0.5", intervals, iterator=200, progress=False
        )

        if serial is None:
            assert parallel is None, "parallel should be None when serial is None"
        else:
            self._assert_frames_equal(serial, parallel, msg="pwm gscreen")

    # ------------------------------------------------------------------
    # 2. PWM vtrack: gextract serial vs parallel
    # ------------------------------------------------------------------

    def test_gextract_pwm_vtrack_matches_serial(self):
        """PWM vtrack through gextract: parallel must match serial."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        pm.gvtrack_create("pwm_vt", None, func="pwm", pssm=pssm, bidirect=False, prior=0)

        intervals = pm.gintervals("1", 0, 100000)

        serial = self._run_serial(
            pm.gextract, "pwm_vt", intervals, iterator=200, progress=False
        )
        parallel = self._run_parallel(
            pm.gextract, "pwm_vt", intervals, iterator=200, progress=False
        )

        self._assert_frames_equal(serial, parallel, msg="pwm gextract")

    # ------------------------------------------------------------------
    # 3. Edit distance vtrack: gscreen serial vs parallel
    # ------------------------------------------------------------------

    def test_gscreen_edit_distance_vtrack(self):
        """Edit distance vtrack through gscreen: parallel must match serial."""
        pssm = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        pm.gvtrack_create(
            "ed_vt", None, func="pwm.edit_distance",
            pssm=pssm, score_thresh=-5.0, bidirect=False, prior=0,
        )

        intervals = pm.gintervals("1", 0, 50000)

        serial = self._run_serial(
            pm.gscreen, "ed_vt <= 1", intervals, iterator=200, progress=False
        )
        parallel = self._run_parallel(
            pm.gscreen, "ed_vt <= 1", intervals, iterator=200, progress=False
        )

        if serial is None:
            assert parallel is None, "parallel should be None when serial is None"
        else:
            self._assert_frames_equal(serial, parallel, msg="edit_distance gscreen")

    # ------------------------------------------------------------------
    # 4. Value-based vtrack (avg of dense_track): gscreen
    # ------------------------------------------------------------------

    def test_gscreen_value_based_vtrack(self):
        """Value-based vtrack (avg) through gscreen: parallel must match serial."""
        pm.gvtrack_create("avg_vt", "dense_track", func="avg")

        intervals = pm.gintervals_all()

        serial = self._run_serial(
            pm.gscreen, "avg_vt > 0.3", intervals, iterator=1000, progress=False
        )
        parallel = self._run_parallel(
            pm.gscreen, "avg_vt > 0.3", intervals, iterator=1000, progress=False
        )

        if serial is None:
            assert parallel is None, "parallel should be None when serial is None"
        else:
            self._assert_frames_equal(serial, parallel, msg="value-based gscreen")

    # ------------------------------------------------------------------
    # 5. Value-based vtrack (avg of dense_track): gextract
    # ------------------------------------------------------------------

    def test_gextract_value_based_vtrack(self):
        """Value-based vtrack (avg) through gextract: parallel must match serial."""
        pm.gvtrack_create("avg_vt", "dense_track", func="avg")

        intervals = pm.gintervals_all()

        serial = self._run_serial(
            pm.gextract, "avg_vt", intervals, iterator=1000, progress=False
        )
        parallel = self._run_parallel(
            pm.gextract, "avg_vt", intervals, iterator=1000, progress=False
        )

        self._assert_frames_equal(serial, parallel, msg="value-based gextract")

    # ------------------------------------------------------------------
    # 6. Mixed expression: vtrack + physical track
    # ------------------------------------------------------------------

    def test_mixed_vtrack_and_physical_track(self):
        """Expression with both vtrack and physical track: parallel must match serial."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        pm.gvtrack_create("pwm_vt", None, func="pwm", pssm=pssm, bidirect=False, prior=0)

        intervals = pm.gintervals("1", 0, 50000)
        expr = "(pwm_vt > 0.5) & (dense_track > 0.3)"

        serial = self._run_serial(
            pm.gscreen, expr, intervals, iterator=500, progress=False
        )
        parallel = self._run_parallel(
            pm.gscreen, expr, intervals, iterator=500, progress=False
        )

        if serial is None:
            assert parallel is None, "parallel should be None when serial is None"
        else:
            self._assert_frames_equal(serial, parallel, msg="mixed vtrack+physical gscreen")

    # ------------------------------------------------------------------
    # 7. Multiple vtracks in expression
    # ------------------------------------------------------------------

    def test_multiple_vtracks_in_expression(self):
        """Expression with two vtracks: parallel must match serial."""
        pssm = np.array([
            [0.9, 0.05, 0.025, 0.025],
            [0.05, 0.9, 0.025, 0.025],
        ])
        pm.gvtrack_create("pwm_vt", None, func="pwm", pssm=pssm, bidirect=False, prior=0)
        pm.gvtrack_create("avg_vt", "dense_track", func="avg")

        intervals = pm.gintervals("1", 0, 50000)
        expr = "(pwm_vt > 0.5) & (avg_vt > 0.3)"

        serial = self._run_serial(
            pm.gscreen, expr, intervals, iterator=500, progress=False
        )
        parallel = self._run_parallel(
            pm.gscreen, expr, intervals, iterator=500, progress=False
        )

        if serial is None:
            assert parallel is None, "parallel should be None when serial is None"
        else:
            self._assert_frames_equal(serial, parallel, msg="multi-vtrack gscreen")

    # ------------------------------------------------------------------
    # 8. Single chromosome with vtrack (intra-chrom splitting)
    # ------------------------------------------------------------------

    def test_vtrack_parallel_single_chrom(self):
        """Single-chrom vtrack gscreen: with intra-chrom splitting, results must match serial."""
        pm.gvtrack_create("avg_vt", "dense_track", func="avg")

        # Only chromosome 1 (500k bp)
        intervals = pm.gintervals("1")

        serial = self._run_serial(
            pm.gscreen, "avg_vt > 0.3", intervals, iterator=500, progress=False
        )
        parallel = self._run_parallel(
            pm.gscreen, "avg_vt > 0.3", intervals, iterator=500, progress=False
        )

        if serial is None:
            assert parallel is None, "parallel should be None when serial is None"
        else:
            self._assert_frames_equal(serial, parallel, msg="single-chrom vtrack gscreen")

    # ------------------------------------------------------------------
    # 9. Kmer vtrack through C++ path
    # ------------------------------------------------------------------

    def test_kmer_vtrack_through_cpp(self):
        """Kmer vtrack through gextract: parallel must match serial."""
        pm.gvtrack_create("kmer_vt", None, func="kmer.count", kmer="ACG")

        intervals = pm.gintervals("1", 0, 50000)

        serial = self._run_serial(
            pm.gextract, "kmer_vt", intervals, iterator=200, progress=False
        )
        parallel = self._run_parallel(
            pm.gextract, "kmer_vt", intervals, iterator=200, progress=False
        )

        self._assert_frames_equal(serial, parallel, msg="kmer gextract")

    # ------------------------------------------------------------------
    # 10. Filter vtrack falls through to Python (backward compat)
    # ------------------------------------------------------------------

    def test_filter_vtrack_falls_through_to_python(self):
        """Vtrack with filter must still work (goes through Python path)."""
        # Create filter intervals
        filt = pd.DataFrame({
            "chrom": ["chr1", "chr1"],
            "start": [0, 50000],
            "end": [30000, 80000],
        })
        pm.gvtrack_create("filt_vt", "dense_track", func="avg", filter=filt)

        intervals = pm.gintervals("1", 0, 100000)

        serial = self._run_serial(
            pm.gextract, "filt_vt", intervals, iterator=500, progress=False
        )
        parallel = self._run_parallel(
            pm.gextract, "filt_vt", intervals, iterator=500, progress=False
        )

        # Both must return results (filter path goes through Python eval)
        assert serial is not None, "filter vtrack serial returned None"
        assert parallel is not None, "filter vtrack parallel returned None"
        assert len(serial) == len(parallel), (
            f"filter vtrack length mismatch: serial={len(serial)}, parallel={len(parallel)}"
        )

        s = _sort_df(serial)
        p = _sort_df(parallel)
        npt.assert_allclose(
            s["filt_vt"].to_numpy(dtype=float),
            p["filt_vt"].to_numpy(dtype=float),
            equal_nan=True,
            err_msg="filter vtrack values should match",
        )
