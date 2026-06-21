import os
import shutil

import numpy as np
import pytest

import _pymisha
import pymisha as pm
from pymisha import _shared
from pymisha._quadtree import write_2d_track_file


def pytest_collection_modifyitems(config, items):
    """Auto-skip benchmark tests unless explicitly selected with ``-m benchmark``."""
    if config.getoption("-m") and "benchmark" in config.getoption("-m"):
        return  # user explicitly asked for benchmarks
    skip_bench = pytest.mark.skip(reason="benchmarks not selected (use -m benchmark)")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_bench)

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT
os.environ.setdefault("PYMISHA_EXAMPLES_DB", str(TEST_DB))

_TRACK_DIR = str(TEST_DB / "tracks")


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    pm.gdb_init(str(TEST_DB))
    yield
    pm.gdb_unload()


def _restore_groot(prev):
    if _shared._GROOT == prev:
        return
    if prev is None:
        pm.gdb_unload()
    else:
        pm.gdb_init(prev)


@pytest.fixture(scope="module", autouse=True)
def _isolate_db_state_module():
    """Undo GROOT changes made at *module* scope - e.g. a module-scoped
    ``gdb_init_examples`` fixture with no teardown - so they can't leak into the
    next test file on the same xdist worker. Snapshots before the module's own
    fixtures run (parent-conftest autouse fixtures init first) and restores
    after, so r_parity's package-scoped DB is preserved.
    """
    prev_groot = _shared._GROOT
    yield
    _restore_groot(prev_groot)


@pytest.fixture(autouse=True)
def _isolate_db_state():
    """Keep cross-test global state from leaking between tests on the same
    process (matters under xdist, where each worker is a long-lived process).

    1. Drop the cached ``pm_track_names`` result so a test sees tracks created
       by its own fixtures. Many fixtures write a track file directly and call
       ``_pymisha.pm_dbreload()`` (C++) without clearing the Python-side cache,
       which only ``gdb_init``/``gtrack_create`` normally do.
    2. Restore GROOT to whatever it was before the test, so a test that calls
       ``gdb_init_examples`` / ``gdb_init`` to another DB can't leak its root
       into the next test. Restoring (rather than forcing the test DB) keeps
       module/package-scoped DBs - incl. the r_parity suite - working.
    """
    prev_groot = _shared._GROOT
    _shared._clear_track_names_cache()
    yield
    _shared._clear_track_names_cache()
    _restore_groot(prev_groot)


def extract_values(expr, intervals, iterator=None):
    df = pm.gextract(expr, intervals, iterator=iterator)
    if df is None or len(df) == 0:
        return np.array([], dtype=float)

    data_cols = [c for c in df.columns if c not in {"chrom", "start", "end", "intervalID"}]
    assert len(data_cols) == 1
    return df[data_cols[0]].to_numpy(dtype=float, copy=False)


# ---------------------------------------------------------------------------
# Shared fixture: rects_track
#
# A small 2D rectangles track written into the test DB. Used by both the
# TrackRects iterator tests and the pm_extract_2d_scanner integration tests.
#
# Objects on pair (0, 0) -- chrom 1 x chrom 1:
#   R1: (100,  200, 300, 400)   area fully inside [0, 500000)^2
#   R2: (500,  600, 700, 800)
#   R3: (50_000, 60_000, 150_000, 160_000)
#
# Objects on pair (0, 1) -- chrom 1 x chrom 2:
#   R4: (100, 200, 300, 400)
# ---------------------------------------------------------------------------

def _track_dir_for(name: str) -> str:
    return os.path.join(_TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name: str) -> None:
    tdir = _track_dir_for(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture()
def rects_track(_init_db):
    """2D rects track with three objects on (0,0) and one on (0,1)."""
    # Re-initialize to the test DB in case a previous test left the global
    # GROOT pointing elsewhere (e.g. test_path_functions.py calls gdb_init_examples()).
    pm.gdb_init(str(TEST_DB))

    tname = "test.track_rects_iter"
    _cleanup_track(tname)

    tdir = _track_dir_for(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    # chrom-pair 1-1  (chromids 0-0)
    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [
            (100, 200, 300, 400, 1.0),
            (500, 600, 700, 800, 2.0),
            (50_000, 60_000, 150_000, 160_000, 3.0),
        ],
        (0, 0, 500_000, 500_000),
        is_points=False,
    )

    # chrom-pair 1-2  (chromids 0-1)
    write_2d_track_file(
        os.path.join(tdir, "1-2"),
        [
            (100, 200, 300, 400, 4.0),
        ],
        (0, 0, 500_000, 300_000),
        is_points=False,
    )

    _pymisha.pm_dbreload()

    yield tname

    _cleanup_track(tname)
