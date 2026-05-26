"""Fixtures for the R misha baseline-parity suite.

These tests run pymisha's equivalent of each R ``expect_regression`` call on
R misha's *own* test DB and compare to R's frozen ``.rds`` baselines. Both live
on lab NFS and are absent in GitHub CI, so the whole package is skipped there.

Marker: ``@pytest.mark.r_parity`` (added automatically to every test here).
"""

from __future__ import annotations

import contextlib
import os
import random
import shutil
import tempfile
from pathlib import Path

import pytest

import pymisha as pm

from .baseline import R_SNAPSHOT_DIR, R_TESTDB
from .overlay import build_overlay

# Small test DB used by the rest of the suite (restored on teardown so r_parity
# tests don't leave GROOT pointing at R's DB for unrelated tests).
_SMALL_TEST_DB = Path(__file__).resolve().parents[1] / "testdb" / "trackdb" / "test"

_DB_AVAILABLE = os.path.isdir(R_TESTDB) and os.path.isdir(R_SNAPSHOT_DIR)
_SKIP_REASON = (
    f"R parity DB/snapshots not present (need {R_TESTDB} and {R_SNAPSHOT_DIR}); "
    "lab-only suite"
)


def pytest_collection_modifyitems(config, items):
    """Mark every test in this package ``r_parity`` and skip if the DB is absent."""
    skip = pytest.mark.skip(reason=_SKIP_REASON)
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.r_parity)
            if not _DB_AVAILABLE:
                item.add_marker(skip)


@pytest.fixture(scope="package", autouse=True)
def r_testdb():
    """Point GROOT at R misha's test DB for the duration of the package."""
    if not _DB_AVAILABLE:
        yield None
        return
    pm.gdb_init(R_TESTDB)
    # Many R regression tests set gmax.data.size = 1e9 (whole-genome extracts).
    # Mirror that for the whole parity package; restore the default afterwards.
    prev_max = pm.CONFIG.get("max_data_size")
    pm.CONFIG["max_data_size"] = int(1e9)
    yield R_TESTDB
    pm.CONFIG["max_data_size"] = prev_max
    # Restore the small test DB for any tests that run afterwards in the session.
    if _SMALL_TEST_DB.is_dir():
        pm.gdb_init(str(_SMALL_TEST_DB))


# --------------------------------------------------------------------------- #
# Writable-DB overlay (for track/interval-creating R tests)
# --------------------------------------------------------------------------- #
#
# R's track-creating regression tests run against an isolated *writable* copy of
# the test DB (``create_isolated_test_db``). We reproduce it with a symlink
# overlay (see ``overlay.py``): the large ``seq/`` is symlinked and every fixture
# track is symlinked into a fresh writable ``tracks/`` tree, so the only real
# disk used is the small ``intervs/``/``pssms/`` copies (~5 MB). That makes a
# temp-dir default safe regardless of host scratch policy.
_OVERLAY_BASE = os.environ.get("PYMISHA_R_OVERLAY_DIR", tempfile.gettempdir())


@pytest.fixture(scope="session")
def _overlay_path():
    """Build the writable overlay once per session; clean it up at the end."""
    if not _DB_AVAILABLE:
        yield None
        return
    os.makedirs(_OVERLAY_BASE, exist_ok=True)
    dest = tempfile.mkdtemp(prefix="pymisha_r_overlay_", dir=_OVERLAY_BASE)
    build_overlay(R_TESTDB, dest)
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


@pytest.fixture
def overlay_db(_overlay_path):
    """Point GROOT at the writable overlay for one test, then restore R's DB.

    Track-creating tests use this instead of the read-only ``r_testdb`` default.
    On teardown GROOT is switched back to R's read-only test DB so subsequent
    read-only parity tests are unaffected.
    """
    if _overlay_path is None:
        pytest.skip(_SKIP_REASON)
    pm.gdb_init(_overlay_path)
    pm.CONFIG["max_data_size"] = int(1e9)
    try:
        yield _overlay_path
    finally:
        pm.gdb_init(R_TESTDB)
        pm.CONFIG["max_data_size"] = int(1e9)


# hg19 snapshot DB used by test-2d-hic-analysis.R (real K562 Hi-C + synthetic
# test tracks). Read-only baselines run on it directly; the few set.out cases
# need the writable overlay below.
_HG19_DB = os.environ.get(
    "PYMISHA_R_HG19_DB", "/net/mraid20/export/tgdata/db/tgdb/misha_snapshot/hg19"
)
_HG19_AVAILABLE = os.path.isdir(_HG19_DB)


@pytest.fixture(scope="session")
def _hg19_overlay_path():
    """Build a writable overlay of the hg19 snapshot DB once per session."""
    if not (_DB_AVAILABLE and _HG19_AVAILABLE):
        yield None
        return
    os.makedirs(_OVERLAY_BASE, exist_ok=True)
    dest = tempfile.mkdtemp(prefix="pymisha_r_hg19_", dir=_OVERLAY_BASE)
    build_overlay(_HG19_DB, dest)
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


@pytest.fixture
def hg19_overlay(_hg19_overlay_path):
    """Point GROOT at the writable hg19 overlay for one test; restore R's DB."""
    if _hg19_overlay_path is None:
        pytest.skip("hg19 snapshot DB not present")
    pm.gdb_init(_hg19_overlay_path)
    pm.CONFIG["max_data_size"] = int(1e8)
    try:
        yield _hg19_overlay_path
    finally:
        pm.gdb_init(R_TESTDB)
        pm.CONFIG["max_data_size"] = int(1e9)


@pytest.fixture
def track_namer():
    """Yield a factory for unique scratch track/interval names; auto-remove them.

    Mirrors R's ``tmptrack <- paste0("test.tmptrack_", sample(1:1e9, 1))`` plus
    its ``withr::defer(gtrack.rm(...))`` cleanup, covering both tracks and
    interval sets (``intervals.set.out``) created during the test.
    """
    created: list[str] = []

    def make(prefix: str = "test.tmptrack") -> str:
        name = f"{prefix}_{random.randint(1, 10**9)}"
        created.append(name)
        return name

    yield make

    for name in created:
        with contextlib.suppress(Exception):
            pm.gtrack_rm(name, force=True)
        with contextlib.suppress(Exception):
            pm.gintervals_rm(name, force=True)
