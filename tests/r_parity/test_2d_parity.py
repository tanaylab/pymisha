"""Parity port of R misha ``test-2d-parity.R`` (2D legacy/indexed parity).

The file's two frozen baselines run on the *examples* DB (``gdb.init_examples``):
round-tripping a saved 2D interval set (``gintervals.save`` -> ``gintervals.load``)
and extracting a freshly created 2D track. Both match R exactly.

The remaining ``test_that`` blocks compare misha's own legacy vs indexed 2D
formats with ``expect_equal`` (no frozen baseline) and aren't ported.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import R_TESTDB, assert_matches_baseline


@pytest.fixture
def examples_db():
    """Switch to pymisha's writable examples DB; restore R's test DB after."""
    pm.gdb_init_examples()
    pm.CONFIG["max_data_size"] = int(1e9)
    try:
        yield
    finally:
        pm.gdb_init(R_TESTDB)
        pm.CONFIG["max_data_size"] = int(1e9)


def _intervs():
    return pm.gintervals_2d([1, "X"], [1000, 2000], [2000, 3000], [1, 2], [5000, 100], [6000, 500])


def test_gintervals_load_2d_basic(examples_db):
    pm.gintervals_save(_intervs(), "test.reg_2d_parity")
    try:
        assert_matches_baseline(pm.gintervals_load("test.reg_2d_parity"), "gintervals_load_2d_basic")
    finally:
        pm.gintervals_rm("test.reg_2d_parity", force=True)


def test_gextract_2d_basic(examples_db):
    pm.gtrack_2d_create("test.track_2d", "Test Track", _intervs(), [1.5, 2.5])
    try:
        assert_matches_baseline(pm.gextract("test.track_2d", _intervs()), "gextract_2d_basic")
    finally:
        pm.gtrack_rm("test.track_2d", force=True)
