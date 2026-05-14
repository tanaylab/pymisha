"""Tests for gtrack_create_dense(func=...) - R misha 5.6.32 parity.

Ports the R-side test blocks added in commits 068a02a2 and 5e69c2c8 plus
the wrap-fix regression sentinel from 1b4f5065.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _bin_values(name: str, chrom: str, start: int, end: int, binsize: int) -> np.ndarray:
    df = pm.gextract(
        name,
        pd.DataFrame({"chrom": [chrom], "start": [start], "end": [end]}),
        iterator=binsize,
    )
    return df[name].to_numpy(dtype=float, copy=False)


def test_default_func_weighted_mean_byte_identical(tmp_path):
    """Default no-kwarg path must be byte-equal to explicit func=weighted.mean."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame(
            {"chrom": ["1", "1"], "start": [0, 50], "end": [100, 150]}
        )
        pm.gtrack_create_dense(
            "wm_default", "d", intervals, [3.0, 7.0], binsize=20, defval=1.0
        )
        pm.gtrack_create_dense(
            "wm_explicit",
            "d",
            intervals,
            [3.0, 7.0],
            binsize=20,
            defval=1.0,
            func="weighted.mean",
        )
        default = _bin_values("wm_default", "1", 0, 200, 20)
        explicit = _bin_values("wm_explicit", "1", 0, 200, 20)
        np.testing.assert_array_equal(default, explicit)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_weighted_sum(tmp_path):
    """weighted.sum = sum(v_i * overlap_i) over intervals + defval-padding."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        # Two intervals each fully covering one bin.
        intervals = pd.DataFrame(
            {"chrom": ["1", "1"], "start": [0, 20], "end": [20, 40]}
        )
        pm.gtrack_create_dense(
            "wsum", "d", intervals, [3.0, 5.0], binsize=20, defval=0.0, func="weighted.sum"
        )
        # Bin [0,20): 3*20 = 60. Bin [20,40): 5*20 = 100. Bin [40,60): defval=0, 0*20=0.
        got = _bin_values("wsum", "1", 0, 60, 20)
        np.testing.assert_allclose(got, [60.0, 100.0, 0.0], rtol=1e-6)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_max(tmp_path):
    """max is unweighted over intervals touching the bin."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        # Three overlapping intervals in bin [0,20) with values 1, 7, 3.
        intervals = pd.DataFrame(
            {"chrom": ["1", "1", "1"], "start": [0, 5, 10], "end": [10, 15, 20]}
        )
        pm.gtrack_create_dense(
            "mx", "d", intervals, [1.0, 7.0, 3.0], binsize=20, defval=float("nan"), func="max"
        )
        got = _bin_values("mx", "1", 0, 20, 20)
        np.testing.assert_allclose(got, [7.0])
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_min(tmp_path):
    """min is unweighted over intervals touching the bin."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame(
            {"chrom": ["1", "1", "1"], "start": [0, 5, 10], "end": [10, 15, 20]}
        )
        pm.gtrack_create_dense(
            "mn", "d", intervals, [1.0, 7.0, 3.0], binsize=20, defval=float("nan"), func="min"
        )
        got = _bin_values("mn", "1", 0, 20, 20)
        np.testing.assert_allclose(got, [1.0])
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_median_lower(tmp_path):
    """Overlap-weighted lower median - intervals at [1.0, 3.0] equal overlap -> 1.0."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame(
            {"chrom": ["1", "1"], "start": [0, 10], "end": [10, 20]}
        )
        pm.gtrack_create_dense(
            "med",
            "d",
            intervals,
            [1.0, 3.0],
            binsize=20,
            defval=float("nan"),
            func="median",
        )
        got = _bin_values("med", "1", 0, 20, 20)
        # Total overlap = 20, half = 10. Sorted by value:
        #   v=1 acc=10 (>= 10) -> return 1.0 (lower median).
        np.testing.assert_allclose(got, [1.0])
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_count(tmp_path):
    """count = number of intervals touching each bin; defval does not contribute; empty = 0."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        # Bin [0,20): two intervals touch. Bin [20,40): one. Bin [40,60): zero.
        intervals = pd.DataFrame(
            {"chrom": ["1", "1", "1"], "start": [0, 5, 25], "end": [10, 15, 35]}
        )
        pm.gtrack_create_dense(
            "cnt", "d", intervals, [9.0, 9.0, 9.0], binsize=20, defval=1.0, func="count"
        )
        got = _bin_values("cnt", "1", 0, 60, 20)
        np.testing.assert_allclose(got, [2.0, 1.0, 0.0])
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_coverage_pileup(tmp_path):
    """coverage with v=1, defval=0 -> per-base average overlap count."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        # Bin [0,20): two intervals each covering 10 bp -> 20/20 = 1.0 mean depth.
        # Bin [20,40): one interval covering 10 bp -> 10/20 = 0.5.
        # Bin [40,60): empty + defval=0 -> 0.0.
        intervals = pd.DataFrame(
            {"chrom": ["1", "1", "1"], "start": [0, 0, 20], "end": [10, 10, 30]}
        )
        pm.gtrack_create_dense(
            "cov",
            "d",
            intervals,
            [1.0, 1.0, 1.0],
            binsize=20,
            defval=0.0,
            func="coverage",
        )
        got = _bin_values("cov", "1", 0, 60, 20)
        np.testing.assert_allclose(got, [1.0, 0.5, 0.0], rtol=1e-6)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_coverage_wrap_regression(tmp_path):
    """R 1b4f5065 regression sentinel: bin past an early-end interval must not wrap.

    Two intervals A=[0,200) B=[10,50). Bin [60,80) sees A only (B ended).
    Pre-fix in R, the unsigned subtraction wrapped to ~9e17 for B; pymisha
    has always had the `ov_end > ov_start` guard so this is just a sentinel.
    """
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame(
            {"chrom": ["1", "1"], "start": [0, 10], "end": [200, 50]}
        )
        pm.gtrack_create_dense(
            "wrap",
            "d",
            intervals,
            [1.0, 1.0],
            binsize=20,
            defval=0.0,
            func="coverage",
        )
        # Bin [60,80): A covers fully (20 bp), B ended at 50 -> contributes 0.
        # coverage = 1*20/20 + 0 = 1.0 (not ~9e17).
        got = _bin_values("wrap", "1", 60, 80, 20)
        np.testing.assert_allclose(got, [1.0], rtol=1e-6)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_invalid_raises_valueerror(tmp_path):
    """Unknown func string raises ValueError listing valid options."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
        with pytest.raises(ValueError, match="Invalid 'func'"):
            pm.gtrack_create_dense(
                "bad", "d", intervals, [1.0], binsize=20, func="bogus"
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_func_case_sensitive(tmp_path):
    """func is case-sensitive: COVERAGE != coverage."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
        with pytest.raises(ValueError, match="Invalid 'func'"):
            pm.gtrack_create_dense(
                "bad", "d", intervals, [1.0], binsize=20, func="COVERAGE"
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_created_by_attr_records_func(tmp_path):
    """created.by includes the non-default func; default omits it."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})

        pm.gtrack_create_dense(
            "cb_default", "d", intervals, [1.0], binsize=20, defval=0.0
        )
        cb_default = pm.gtrack_attr_get("cb_default", "created.by")
        assert "func=" not in cb_default

        pm.gtrack_create_dense(
            "cb_cov",
            "d",
            intervals,
            [1.0],
            binsize=20,
            defval=0.0,
            func="coverage",
        )
        cb_cov = pm.gtrack_attr_get("cb_cov", "created.by")
        assert 'func="coverage"' in cb_cov
    finally:
        pm.gdb_init(str(TEST_DB))
