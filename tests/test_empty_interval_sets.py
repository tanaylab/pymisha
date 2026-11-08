"""Empty interval sets, as R misha: saved, loaded as frames of no rows, and written by
the functions that save their result to intervals_set_out. The fixtures are sets R
misha 5.16.1 saved (gintervals.save of a data frame of no rows, 1D and 2D)."""

import shutil
from pathlib import Path

import pandas as pd
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "empty_sets"
COLS_1D = ["chrom", "start", "end"]
COLS_2D = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
E1 = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]}).iloc[0:0]
E2 = pd.DataFrame({"chrom1": ["chr1"], "start1": [0], "end1": [10],
                   "chrom2": ["chr1"], "start2": [0], "end2": [10]}).iloc[0:0]


@pytest.fixture(scope="module", autouse=True)
def db(tmp_path_factory):
    root = tmp_path_factory.mktemp("empty_sets") / "db"
    shutil.copytree(TESTDB_ROOT, root)
    for name in ("r_empty_1d", "r_empty_2d"):
        shutil.copy(FIXTURES / f"{name}.interv", root / "tracks" / f"{name}.interv")
    pm.gdb_init(str(root))
    yield root
    pm.gdb_init(str(TESTDB_ROOT))


def empty_set(name, cols):
    res = pm.gintervals_load(name)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0
    assert list(res.columns) == cols
    return res


def test_an_empty_set_is_saved_and_loaded_as_a_frame_of_no_rows():
    pm.gintervals_save(E1, "e1")
    pm.gintervals_save(E2, "e2")
    assert pm.gintervals_exists("e1")
    res = empty_set("e1", COLS_1D)
    assert res["start"].dtype == "int64"
    assert res["chrom"].dtype == pm.gintervals("chr1", 0, 10)["chrom"].dtype
    empty_set("e2", COLS_2D)
    assert pm.gextract("dense_track", "e1") is None
    with pytest.raises(ValueError):
        pm.gintervals_save(None, "e_none")


def test_empty_sets_saved_by_r_are_read():
    empty_set("r_empty_1d", COLS_1D)
    empty_set("r_empty_2d", COLS_2D)


def test_a_chromosome_filter_that_leaves_nothing_gives_a_frame_of_no_rows():
    pm.gintervals_save(pm.gintervals("chr1", 0, 10), "one")
    res = pm.gintervals_load("one", chrom="chrX")
    assert len(res) == 0
    assert list(res.columns) == COLS_1D


def test_update_that_removes_everything_leaves_an_empty_set():
    pm.gintervals_save(pm.gintervals(["chr1", "chr2"], 0, 10), "u1")
    pm.gintervals_update("u1", None, chrom="chr1")
    pm.gintervals_update("u1", None, chrom="chr2")
    empty_set("u1", COLS_1D)


@pytest.mark.parametrize("expr, intervals, iterator, cols", [
    ("dense_track", E1, 100, COLS_1D),
    (None, E1, 100, COLS_1D),
    (None, E2, (100, 100), COLS_2D),
])
def test_giterator_intervals_of_nothing_saves_an_empty_set(expr, intervals, iterator, cols):
    pm.gintervals_rm("gi", force=True)
    assert pm.giterator_intervals(expr, intervals, iterator=iterator, intervals_set_out="gi") is None
    empty_set("gi", cols)


@pytest.mark.parametrize("expr, cols", [("dense_track > 1e9", COLS_1D), ("rects_track > 1e9", COLS_2D)])
def test_gscreen_with_no_hits_saves_an_empty_set(expr, cols):
    pm.gintervals_rm("gs", force=True)
    assert pm.gscreen(expr) is None
    assert pm.gscreen(expr, intervals_set_out="gs") is None
    empty_set("gs", cols)


def test_gscreen_with_hits_saves_them():
    pm.gintervals_rm("gs_hits", force=True)
    pm.gscreen("dense_track > 0.2", intervals_set_out="gs_hits")
    pd.testing.assert_frame_equal(pm.gintervals_load("gs_hits"), pm.gscreen("dense_track > 0.2"))


def test_mapply_and_gextract_over_no_intervals_create_no_set():
    assert pm.gintervals_mapply(lambda x: 1, "dense_track", intervals=E1, intervals_set_out="mp") is None
    assert not pm.gintervals_exists("mp")
    assert pm.gextract("dense_track", E1, intervals_set_out="gx") is None
    assert not pm.gintervals_exists("gx")
