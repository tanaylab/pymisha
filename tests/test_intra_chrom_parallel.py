"""Tests for intra-chromosome range-based parallelization.

When multitasking is enabled and a fixed-bin iterator is used, large
chromosomes are split across multiple workers by genomic range.  These
tests verify that the parallel results are identical to the serial
baseline for gscreen, gextract, gsummary, etc.
"""

import pandas as pd
import pandas.testing as pdt
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def _sort_df(df):
    """Sort a DataFrame by chrom/start/end and reset index."""
    cols = ["chrom", "start", "end"]
    if "intervalID" in df.columns:
        cols.append("intervalID")
    return df.sort_values(cols).reset_index(drop=True)


class TestIntraChromParallel:
    """Compare multitasking=True vs False for intra-chrom splitting."""

    def setup_method(self):
        """Save CONFIG before each test."""
        self._saved = pm.CONFIG.copy()

    def teardown_method(self):
        """Restore CONFIG after each test."""
        pm.CONFIG.update(self._saved)

    # -- helpers ----------------------------------------------------------

    def _set_serial(self):
        pm.CONFIG.update({
            "multitasking": False,
            "min_processes": 2,
            "max_processes": 8,
        })

    def _set_parallel(self):
        pm.CONFIG.update({
            "multitasking": True,
            "min_processes": 2,
            "max_processes": 8,
        })

    # -- 1. gscreen single chrom ------------------------------------------

    def test_gscreen_single_chrom_matches_serial(self):
        """gscreen on a single chromosome must produce identical intervals
        regardless of multitasking mode."""
        intervals = pm.gintervals("1", 0, 500000)

        self._set_serial()
        serial = pm.gscreen(
            "dense_track > 0.2", intervals, iterator=200, progress=False,
        )

        self._set_parallel()
        multi = pm.gscreen(
            "dense_track > 0.2", intervals, iterator=200, progress=False,
        )

        if serial is None or multi is None:
            assert serial is None and multi is None
            return

        pdt.assert_frame_equal(
            _sort_df(serial), _sort_df(multi),
        )

    # -- 2. gextract single chrom -----------------------------------------

    def test_gextract_single_chrom_matches_serial(self):
        """gextract on chrom 1 with fixed-bin iterator must produce the
        same rows in serial and parallel modes."""
        intervals = pm.gintervals("1", 0, 500000)

        self._set_serial()
        serial = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        self._set_parallel()
        multi = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        assert serial is not None and multi is not None
        pdt.assert_frame_equal(
            _sort_df(serial), _sort_df(multi),
        )

    # -- 3. intervalID correctness ----------------------------------------

    def test_gextract_interval_ids_correct(self):
        """intervalID values must be 1-based and match the original interval
        indices when multitasking splits a single chromosome."""
        intervals = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [0, 100000, 200000],
            "end":   [50000, 150000, 250000],
        })

        self._set_serial()
        serial = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        self._set_parallel()
        multi = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        assert serial is not None and multi is not None

        # intervalIDs should be 1-based and span exactly {1, 2, 3}
        expected_ids = {1, 2, 3}
        assert set(serial["intervalID"].unique()) == expected_ids
        assert set(multi["intervalID"].unique()) == expected_ids

        # Full frame equality after sorting
        pdt.assert_frame_equal(
            _sort_df(serial), _sort_df(multi),
        )

    # -- 4. gsummary single chrom -----------------------------------------

    def test_gsummary_single_chrom_matches_serial(self):
        """gsummary on chrom 1 must produce identical statistics in serial
        and parallel modes."""
        intervals = pm.gintervals("1", 0, 500000)

        self._set_serial()
        serial = pm.gsummary(
            "dense_track", intervals, iterator=200,
        )

        self._set_parallel()
        multi = pm.gsummary(
            "dense_track", intervals, iterator=200,
        )

        pdt.assert_series_equal(serial, multi, rtol=1e-10)

    # -- 5. gscreen boundary merging --------------------------------------

    def test_gscreen_boundary_merging(self):
        """With intra-chrom splitting, abutting intervals at split
        boundaries must be merged so the result matches the serial
        baseline."""
        intervals = pm.gintervals("1", 0, 500000)

        self._set_serial()
        serial = pm.gscreen(
            "dense_track > 0", intervals, iterator=200, progress=False,
        )

        self._set_parallel()
        multi = pm.gscreen(
            "dense_track > 0", intervals, iterator=200, progress=False,
        )

        if serial is None or multi is None:
            assert serial is None and multi is None
            return

        pdt.assert_frame_equal(
            _sort_df(serial), _sort_df(multi),
        )

    # -- 6. max_processes=1 fallback --------------------------------------

    def test_multitasking_disabled_fallback(self):
        """With max_processes=1 the result must match the serial baseline
        (effective serial fallback)."""
        intervals = pm.gintervals("1", 0, 500000)

        self._set_serial()
        serial = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        # Enable multitasking but cap at 1 process
        pm.CONFIG.update({
            "multitasking": True,
            "min_processes": 1,
            "max_processes": 1,
        })
        fallback = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        assert serial is not None and fallback is not None
        pdt.assert_frame_equal(
            _sort_df(serial), _sort_df(fallback),
        )

    # -- 7. multi-chrom with range split ----------------------------------

    def test_multi_chrom_with_range_split(self):
        """gextract on all chromosomes exercises both per-chrom and
        intra-chrom splitting.  Results must match the serial baseline."""
        intervals = pm.gintervals_all()

        self._set_serial()
        serial = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        self._set_parallel()
        multi = pm.gextract(
            "dense_track", intervals, iterator=200, progress=False,
        )

        assert serial is not None and multi is not None
        pdt.assert_frame_equal(
            _sort_df(serial), _sort_df(multi),
        )
