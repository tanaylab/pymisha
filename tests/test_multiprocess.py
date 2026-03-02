"""Tests for multi-process extraction (GAP-022)."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture()
def _save_restore_config():
    """Save and restore CONFIG max_processes around each test."""
    old = pm.CONFIG.get("max_processes", 1)
    old_mt = pm.CONFIG.get("multitasking", True)
    yield
    pm.CONFIG["max_processes"] = old
    pm.CONFIG["multitasking"] = old_mt


class TestGmaxProcesses:
    def test_get_default(self):
        """Default max_processes should be an integer."""
        n = pm.gmax_processes()
        assert isinstance(n, int)
        assert n >= 1

    def test_set_and_get(self, _save_restore_config):
        result = pm.gmax_processes(4)
        assert result == 4
        assert pm.gmax_processes() == 4

    def test_set_to_1(self, _save_restore_config):
        pm.gmax_processes(1)
        assert pm.gmax_processes() == 1

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            pm.gmax_processes(0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            pm.gmax_processes(-1)


class TestParallelExtract:
    """Test that parallel extraction matches serial extraction."""

    def test_parallel_matches_serial(self, _save_restore_config):
        """Parallel gextract produces same results as serial."""
        intervals = pm.gintervals_all()

        # Serial extraction
        pm.gmax_processes(1)
        pm.CONFIG["multitasking"] = False
        serial = pm.gextract(
            "dense_track", intervals=intervals, iterator=500, progress=False
        )

        # Parallel extraction
        pm.gmax_processes(3)
        pm.CONFIG["multitasking"] = True
        parallel = pm.gextract(
            "dense_track", intervals=intervals, iterator=500, progress=False
        )

        assert serial is not None
        assert parallel is not None
        assert len(serial) == len(parallel)

        # Sort both by chrom/start for comparison
        serial = serial.sort_values(["chrom", "start"]).reset_index(drop=True)
        parallel = parallel.sort_values(["chrom", "start"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            serial[["chrom", "start", "end", "dense_track"]],
            parallel[["chrom", "start", "end", "dense_track"]],
        )

    def test_parallel_multi_expr(self, _save_restore_config):
        """Parallel gextract with expression."""
        intervals = pm.gintervals_all()

        pm.gmax_processes(1)
        pm.CONFIG["multitasking"] = False
        serial = pm.gextract(
            "dense_track + 1",
            intervals=intervals,
            iterator=1000,
            progress=False,
        )

        pm.gmax_processes(2)
        pm.CONFIG["multitasking"] = True
        parallel = pm.gextract(
            "dense_track + 1",
            intervals=intervals,
            iterator=1000,
            progress=False,
        )

        assert serial is not None
        assert parallel is not None
        assert len(serial) == len(parallel)

        serial = serial.sort_values(["chrom", "start"]).reset_index(drop=True)
        parallel = parallel.sort_values(["chrom", "start"]).reset_index(drop=True)

        np.testing.assert_allclose(
            serial.iloc[:, 3].to_numpy(),
            parallel.iloc[:, 3].to_numpy(),
            equal_nan=True,
        )

    def test_single_chrom_no_parallel(self, _save_restore_config):
        """Single chromosome falls back to serial."""
        pm.gmax_processes(4)
        pm.CONFIG["multitasking"] = True
        result = pm.gextract(
            "dense_track",
            intervals=pm.gintervals("1", 0, 10000),
            iterator=500,
            progress=False,
        )
        assert result is not None
        assert len(result) > 0
