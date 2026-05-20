import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import _pymisha
import pymisha as pm
from pymisha._quadtree import write_2d_track_file


def pytest_collection_modifyitems(config, items):
    """Auto-skip benchmark tests unless explicitly selected with ``-m benchmark``."""
    if config.getoption("-m") and "benchmark" in config.getoption("-m"):
        return  # user explicitly asked for benchmarks
    skip_bench = pytest.mark.skip(reason="benchmarks not selected (use -m benchmark)")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_bench)

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"
os.environ.setdefault("PYMISHA_EXAMPLES_DB", str(TEST_DB))

_TRACK_DIR = str(TEST_DB / "tracks")


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    pm.gdb_init(str(TEST_DB))
    yield
    pm.gdb_unload()


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
