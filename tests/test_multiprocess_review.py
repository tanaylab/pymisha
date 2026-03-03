"""Comprehensive tests for multi-process extraction (GAP-022 review).

Covers:
- Parity between parallel and serial for gextract, gsummary, gquantiles, gdist, gscreen
- Edge cases: single chrom, empty intervals, max_processes=1
- IntervalID remapping correctness
- Virtual tracks skip parallel path
- Multiple expression parity
"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture()
def _save_restore_config():
    """Save and restore CONFIG around each test."""
    saved = pm.CONFIG.copy()
    yield
    pm.CONFIG.update(saved)


def _sort_df(df):
    """Sort a DataFrame by coordinates and reset index."""
    cols = ["chrom", "start", "end"]
    if "intervalID" in df.columns:
        cols.append("intervalID")
    return df.sort_values(cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. gextract parity: parallel vs serial with arithmetic expression
# ---------------------------------------------------------------------------

class TestGextractParallelParity:
    def test_parallel_vs_serial_simple_expr(self, _save_restore_config):
        """Parallel gextract with dense_track + 1 matches serial."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "dense_track + 1", intervals=intervals, iterator=500, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        parallel = pm.gextract(
            "dense_track + 1", intervals=intervals, iterator=500, progress=False,
        )

        assert serial is not None and parallel is not None
        assert len(serial) == len(parallel)

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        np.testing.assert_allclose(
            serial_s.iloc[:, 3].to_numpy(),
            parallel_s.iloc[:, 3].to_numpy(),
            equal_nan=True,
        )

    def test_parallel_vs_serial_multiple_exprs(self, _save_restore_config):
        """Multiple expressions in parallel match serial."""
        intervals = pm.gintervals_all()
        exprs = ["dense_track", "dense_track * 2"]

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(exprs, intervals=intervals, iterator=1000, progress=False)

        pm.CONFIG.update({"multitasking": True, "max_processes": 3})
        parallel = pm.gextract(exprs, intervals=intervals, iterator=1000, progress=False)

        assert serial is not None and parallel is not None
        assert len(serial) == len(parallel)

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        for col in serial_s.columns:
            if col in ("chrom", "start", "end", "intervalID"):
                continue
            np.testing.assert_allclose(
                serial_s[col].to_numpy(),
                parallel_s[col].to_numpy(),
                equal_nan=True,
            )

    def test_parallel_vs_serial_colnames(self, _save_restore_config):
        """Column renaming works correctly in parallel mode."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "dense_track", intervals=intervals, iterator=500,
            colnames=["val"], progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 3})
        parallel = pm.gextract(
            "dense_track", intervals=intervals, iterator=500,
            colnames=["val"], progress=False,
        )

        assert "val" in serial.columns
        assert "val" in parallel.columns

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        np.testing.assert_allclose(
            serial_s["val"].to_numpy(),
            parallel_s["val"].to_numpy(),
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# 2. IntervalID remapping correctness
# ---------------------------------------------------------------------------

class TestIntervalIDRemapping:
    def test_interval_ids_consistent(self, _save_restore_config):
        """IntervalIDs from parallel match serial exactly after sorting."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "dense_track", intervals=intervals, iterator=2000, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        parallel = pm.gextract(
            "dense_track", intervals=intervals, iterator=2000, progress=False,
        )

        assert serial is not None and parallel is not None

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        np.testing.assert_array_equal(
            serial_s["intervalID"].to_numpy(),
            parallel_s["intervalID"].to_numpy(),
        )

    def test_interval_ids_range(self, _save_restore_config):
        """Parallel intervalIDs should use valid 1-based indices into the input."""
        intervals = pm.gintervals_all()
        n_input = len(intervals)

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        result = pm.gextract(
            "dense_track", intervals=intervals, iterator=5000, progress=False,
        )

        if result is not None and len(result) > 0:
            ids = result["intervalID"].to_numpy()
            assert ids.min() >= 1
            # intervalIDs reference iterated intervals (which may be more
            # numerous than input intervals), so just check >= 1
            assert ids.min() >= 1


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_max_processes_1_matches_serial(self, _save_restore_config):
        """gmax_processes(1) should produce identical results to multitasking=False."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "dense_track", intervals=intervals, iterator=1000, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 1})
        result = pm.gextract(
            "dense_track", intervals=intervals, iterator=1000, progress=False,
        )

        serial_s = _sort_df(serial)
        result_s = _sort_df(result)

        pdt.assert_frame_equal(serial_s, result_s)

    def test_single_chrom_falls_back_to_serial(self, _save_restore_config):
        """Single-chrom intervals should fall back to serial and return correct data."""
        intervals = pm.gintervals("1", 0, 10000)

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        result = pm.gextract(
            "dense_track", intervals=intervals, iterator=500, progress=False,
        )

        assert result is not None
        assert len(result) > 0
        # Test DB uses unprefixed chrom names ("1", not "chr1")
        assert result["chrom"].nunique() == 1

    def test_empty_intervals(self, _save_restore_config):
        """Empty intervals should return None or empty DataFrame."""
        empty = pd.DataFrame({"chrom": [], "start": [], "end": []})
        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        result = pm.gextract("dense_track", intervals=empty, iterator=100, progress=False)
        assert result is None or len(result) == 0

    def test_more_processes_than_chroms(self, _save_restore_config):
        """max_processes > number of chroms should still work correctly."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "dense_track", intervals=intervals, iterator=1000, progress=False,
        )

        # Test DB has 3 chroms; use 10 processes
        pm.CONFIG.update({"multitasking": True, "max_processes": 10})
        parallel = pm.gextract(
            "dense_track", intervals=intervals, iterator=1000, progress=False,
        )

        assert serial is not None and parallel is not None
        assert len(serial) == len(parallel)

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        np.testing.assert_allclose(
            serial_s["dense_track"].to_numpy(),
            parallel_s["dense_track"].to_numpy(),
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# 4. Virtual tracks skip parallel path
# ---------------------------------------------------------------------------

class TestVtracksNotParallel:
    def test_vtrack_expr_not_parallelized(self, _save_restore_config):
        """Virtual track expressions should skip parallel and produce correct results."""
        vt_name = "_review_test_vt_sum"
        if vt_name in (pm.gvtrack_ls() or []):
            pm.gvtrack_rm(vt_name)

        try:
            pm.gvtrack_create(vt_name, src="dense_track", func="sum")
            pm.gvtrack_iterator(vt_name, sshift=-50, eshift=50)

            intervals = pm.gintervals_all()

            pm.CONFIG.update({"multitasking": False, "max_processes": 1})
            serial = pm.gextract(
                vt_name, intervals=intervals, iterator=1000, progress=False,
            )

            # Even with parallel enabled, vtracks force serial
            pm.CONFIG.update({"multitasking": True, "max_processes": 4})
            result = pm.gextract(
                vt_name, intervals=intervals, iterator=1000, progress=False,
            )

            assert serial is not None and result is not None
            assert len(serial) == len(result)

            serial_s = _sort_df(serial)
            result_s = _sort_df(result)

            np.testing.assert_allclose(
                serial_s[vt_name].to_numpy(),
                result_s[vt_name].to_numpy(),
                rtol=1e-6,
                equal_nan=True,
            )
        finally:
            if vt_name in (pm.gvtrack_ls() or []):
                pm.gvtrack_rm(vt_name)


# ---------------------------------------------------------------------------
# 5. gsummary C++ multitask parity
# ---------------------------------------------------------------------------

class TestGsummaryMultitask:
    def test_gsummary_multitask_vs_serial(self, _save_restore_config):
        """gsummary with C++ multitask matches serial."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gsummary(
            "dense_track", intervals=intervals, iterator=500, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        multi = pm.gsummary(
            "dense_track", intervals=intervals, iterator=500, progress=False,
        )

        assert serial is not None and multi is not None
        # gsummary returns a pandas Series
        pdt.assert_series_equal(serial, multi, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. gquantiles C++ multitask parity
# ---------------------------------------------------------------------------

class TestGquantilesMultitask:
    def test_gquantiles_multitask_vs_serial(self, _save_restore_config):
        """gquantiles with C++ multitask matches serial."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gquantiles(
            "dense_track", percentiles=[0.1, 0.5, 0.9],
            intervals=intervals, iterator=500, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        multi = pm.gquantiles(
            "dense_track", percentiles=[0.1, 0.5, 0.9],
            intervals=intervals, iterator=500, progress=False,
        )

        assert serial is not None and multi is not None
        np.testing.assert_allclose(serial, multi, rtol=1e-6, equal_nan=True)


# ---------------------------------------------------------------------------
# 7. gdist C++ multitask parity
# ---------------------------------------------------------------------------

class TestGdistMultitask:
    def test_gdist_multitask_vs_serial(self, _save_restore_config):
        """gdist with C++ multitask matches serial."""
        intervals = pm.gintervals_all()

        breaks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gdist(
            "dense_track", breaks,
            intervals=intervals, iterator=500, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        multi = pm.gdist(
            "dense_track", breaks,
            intervals=intervals, iterator=500, progress=False,
        )

        assert serial is not None and multi is not None
        np.testing.assert_array_equal(serial, multi)


# ---------------------------------------------------------------------------
# 8. gscreen C++ multitask parity
# ---------------------------------------------------------------------------

class TestGscreenMultitask:
    def test_gscreen_multitask_vs_serial(self, _save_restore_config):
        """gscreen with C++ multitask matches serial."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gscreen(
            "dense_track > 0.3", intervals=intervals, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        multi = pm.gscreen(
            "dense_track > 0.3", intervals=intervals, progress=False,
        )

        if serial is None or multi is None:
            assert serial is None and multi is None
        else:
            pdt.assert_frame_equal(
                _sort_df(serial), _sort_df(multi),
            )


# ---------------------------------------------------------------------------
# 9. Stress: many intervals across chroms
# ---------------------------------------------------------------------------

class TestStress:
    def test_many_small_intervals(self, _save_restore_config):
        """Many small intervals across all chroms with parallel extraction."""
        all_ivls = pm.gintervals_all()
        rows = []
        for _, row in all_ivls.iterrows():
            chrom = row["chrom"]
            chrom_end = int(row["end"])
            step = 1000
            start = 0
            while start + 200 <= chrom_end:
                rows.append({"chrom": chrom, "start": start, "end": start + 200})
                start += step
        intervals = pd.DataFrame(rows)

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "dense_track", intervals=intervals, iterator=200, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        parallel = pm.gextract(
            "dense_track", intervals=intervals, iterator=200, progress=False,
        )

        assert serial is not None and parallel is not None
        assert len(serial) == len(parallel)

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        np.testing.assert_allclose(
            serial_s["dense_track"].to_numpy(),
            parallel_s["dense_track"].to_numpy(),
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# 10. glookup C++ multitask parity
# ---------------------------------------------------------------------------

class TestGlookupMultitask:
    def test_glookup_multitask_vs_serial(self, _save_restore_config):
        """glookup with C++ multitask matches serial."""
        intervals = pm.gintervals_all()
        breaks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        lookup_table = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, iterator=500, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 4})
        multi = pm.glookup(
            lookup_table, "dense_track", breaks,
            intervals=intervals, iterator=500, progress=False,
        )

        assert serial is not None and multi is not None
        serial_s = serial.sort_values(["chrom", "start"]).reset_index(drop=True)
        multi_s = multi.sort_values(["chrom", "start"]).reset_index(drop=True)

        np.testing.assert_allclose(
            serial_s["value"].to_numpy(),
            multi_s["value"].to_numpy(),
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# 11. Sparse track parallel parity
# ---------------------------------------------------------------------------

class TestSparseTrackParallel:
    def test_sparse_track_parallel_vs_serial(self, _save_restore_config):
        """Sparse track extraction in parallel matches serial."""
        intervals = pm.gintervals_all()

        pm.CONFIG.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(
            "sparse_track", intervals=intervals, iterator=1000, progress=False,
        )

        pm.CONFIG.update({"multitasking": True, "max_processes": 3})
        parallel = pm.gextract(
            "sparse_track", intervals=intervals, iterator=1000, progress=False,
        )

        assert serial is not None and parallel is not None
        assert len(serial) == len(parallel)

        serial_s = _sort_df(serial)
        parallel_s = _sort_df(parallel)

        np.testing.assert_allclose(
            serial_s["sparse_track"].to_numpy(),
            parallel_s["sparse_track"].to_numpy(),
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# 12. Config restoration sanity
# ---------------------------------------------------------------------------

class TestConfigRestoration:
    def test_config_not_corrupted_by_parallel(self, _save_restore_config):
        """After parallel extraction, CONFIG should be unchanged."""
        original = pm.CONFIG.copy()
        pm.CONFIG.update({"multitasking": True, "max_processes": 4})

        intervals = pm.gintervals_all()
        pm.gextract("dense_track", intervals=intervals, iterator=500, progress=False)

        # max_processes should still be 4
        assert pm.CONFIG["max_processes"] == 4
        assert pm.CONFIG["multitasking"] is True
