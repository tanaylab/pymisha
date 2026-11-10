"""The 2D value summarizers skip NaN values, as R misha a3a27ed3 (test-2d-vtrack-nan.R).

avg, min, max, weighted.sum and a 2D track expression (avg) skip NaN cells, as
the 1D summarizers do, and give NaN when no value overlaps the interval; avg is
the mean over the area that has a value. area stays the area covered by the
track rectangles, NaN ones included. min and max used to depend on where a NaN
cell sat, and avg and weighted.sum were NaN. The statistics the quad-tree stores
per node are written with the NaN in them, so a query that covers such a node
must read it object by object.

The writer is unchanged, so the tracks created here carry the same node
statistics, NaN included, as tracks written by earlier versions.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import _pymisha
import pymisha as pm
from pymisha import _quadtree
from pymisha.extract import _gextract_2d_vtrack_agg

from _dbpath import TESTDB_ROOT

FUNCS = ["avg", "min", "max", "weighted.sum", "area"]
VALUE_FUNCS = ["expr", "avg", "min", "max", "weighted.sum"]
PATHS = ["scanner", "agg", "python"]


def _rm(name):
    tdir = os.path.join(str(TESTDB_ROOT), "tracks", name.replace(".", "/") + ".track")
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _clean():
    yield
    pm.gvtrack_clear()


def _extract(track, q, path, band=None, monkeypatch=None):
    """{func: values per query}: through gextract (the C++ scanner), through
    _gextract_2d_vtrack_agg, or through it with the pure-Python reader."""
    if path == "scanner":
        vts = [f"v_{f.replace('.', '_')}" for f in FUNCS]
        for v, f in zip(vts, FUNCS, strict=True):
            pm.gvtrack_create(v, track, f)
        # one query at a time: the queries overlap, which a 2D iterator does not allow
        rows = [pm.gextract([track, *vts], q.iloc[[i]], iterator=q.iloc[[i]], band=band,
                            colnames=["expr", *FUNCS]) for i in range(len(q))]
        res = pd.concat(rows, ignore_index=True)
        return {f: res[f].to_numpy(dtype=float) for f in ["expr", *FUNCS]}
    if path == "python":
        monkeypatch.setattr(_quadtree, "_HAS_CPP_QUADTREE", False)
    out = {f: _gextract_2d_vtrack_agg(track, "v", q, band, f)["v"].to_numpy(dtype=float) for f in FUNCS}
    out["expr"] = out["avg"]
    return out


def _expected(cells, vals, q):
    """The value summarizers over the cells that have a value, and the area of all the cells."""
    out = {f: [] for f in ["expr", *FUNCS]}
    for _, r in q.iterrows():
        a = (np.maximum(0, np.minimum(cells["end1"], r["end1"]) - np.maximum(cells["start1"], r["start1"]))
             * np.maximum(0, np.minimum(cells["end2"], r["end2"]) - np.maximum(cells["start2"], r["start2"])))
        a = a.to_numpy(dtype=float)
        keep = (a > 0) & ~np.isnan(vals)
        ws = float(np.sum(vals[keep] * a[keep])) if keep.any() else np.nan
        avg = ws / a[keep].sum() if keep.any() else np.nan
        for f, v in [("expr", avg), ("avg", avg), ("min", vals[keep].min() if keep.any() else np.nan),
                     ("max", vals[keep].max() if keep.any() else np.nan), ("weighted.sum", ws), ("area", a.sum())]:
            out[f].append(v)
    return {f: np.array(v) for f, v in out.items()}


@pytest.mark.parametrize("path", PATHS)
def test_a_nan_cell_is_skipped_wherever_it_sits(path, monkeypatch):
    name = "test.nan2d"
    rects = pm.gintervals_2d(["1", "1"], [0, 1000], [100, 1100], ["1", "1"], [0, 1000], [100, 1100])
    q = pm.gintervals_2d("1", 0, 2000, "1", 0, 2000)
    try:
        for vals in ([np.nan, 5.0], [5.0, np.nan]):
            _rm(name)
            pm.gtrack_2d_create(name, "", rects, vals)
            got = _extract(name, q, path, monkeypatch=monkeypatch)
            assert {f: float(v[0]) for f, v in got.items()} == {
                "expr": 5.0, "avg": 5.0, "min": 5.0, "max": 5.0, "weighted.sum": 5e4, "area": 2e4}, vals

        _rm(name)
        pm.gtrack_2d_create(name, "", rects, [np.nan, np.nan])
        got = _extract(name, q, path, monkeypatch=monkeypatch)
        assert got["area"][0] == 2e4
        assert all(np.isnan(got[f][0]) for f in VALUE_FUNCS)
    finally:
        _rm(name)


@pytest.fixture(scope="module")
def grid_tracks(tmp_path_factory):
    """900 cells on a grid, far more than the 20 objects of a quad-tree leaf, so the queries
    cover whole nodes and read their stored statistics. Each track three times: as is,
    without its NaN cells (_ref) and with 1 for NaN (_filled), as rects and as points."""
    rng = np.random.default_rng(17)
    i, j = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    i, j = i.ravel(), j.ravel()
    cells = pm.gintervals_2d(["1"] * i.size, i * 200, i * 200 + 100, ["1"] * i.size, j * 200, j * 200 + 100)
    vals = rng.integers(1, 51, size=len(cells)).astype(float)
    vals[rng.choice(len(cells), 120, replace=False)] = np.nan
    vals[-1] = np.nan   # the last cell written: the last value every node above it saw
    has_val = ~np.isnan(vals)
    versions = {"": (np.ones(len(cells), bool), vals), "_ref": (has_val, vals),
                "_filled": (np.ones(len(cells), bool), np.where(has_val, vals, 1.0))}
    names = []
    tmp = tmp_path_factory.mktemp("nan2d")
    for suffix, (keep, v) in versions.items():
        rname, pname = f"test.nan2d_rects{suffix}", f"test.nan2d_points{suffix}"
        for nm in (rname, pname):
            _rm(nm)
        pm.gtrack_2d_create(rname, "", cells[keep].reset_index(drop=True), v[keep])
        src = tmp / f"points{suffix}.txt"
        pd.DataFrame({"chrom1": "chr1", "start1": cells["start1"][keep], "end1": cells["start1"][keep] + 1,
                      "chrom2": "chr1", "start2": cells["start2"][keep], "end2": cells["start2"][keep] + 1,
                      "value": v[keep]}).to_csv(src, sep="\t", index=False, na_rep="NaN")
        pm.gtrack_2d_import(pname, "", str(src))
        names += [rname, pname]
    yield cells, vals
    for nm in names:
        _rm(nm)


QUERIES = pd.DataFrame({
    "chrom1": "chr1", "start1": [0, 0, 150, 1230, 2970], "end1": [6000, 3000, 5310, 4400, 3020],
    "chrom2": "chr1", "start2": [0, 3000, 40, 610, 0], "end2": [6000, 6000, 4890, 5990, 2900],
})


@pytest.mark.parametrize("path", PATHS)
def test_node_statistics_with_a_nan_are_not_used(grid_tracks, path, monkeypatch):
    cells, vals = grid_tracks
    assert pm.gtrack_info("test.nan2d_points")["type"] == "points"
    exp = _expected(cells, vals, QUERIES)
    got = _extract("test.nan2d_rects", QUERIES, path, monkeypatch=monkeypatch)
    for f in got:
        np.testing.assert_allclose(got[f], exp[f], rtol=1e-6, err_msg=f)


@pytest.mark.parametrize("band", [None, (-2500, 0), (-100000, -1300), (700, 100000)])
@pytest.mark.parametrize("kind", ["rects", "points"])
@pytest.mark.parametrize("path", PATHS)
def test_with_a_band_and_for_points(grid_tracks, band, kind, path, monkeypatch):
    """The values are those of the track without its NaN cells, the area that of the
    track with a value in every cell."""
    tr = f"test.nan2d_{kind}"
    got = _extract(tr, QUERIES, path, band, monkeypatch)
    ref = _extract(tr + "_ref", QUERIES, path, band, monkeypatch)
    filled = _extract(tr + "_filled", QUERIES, path, band, monkeypatch)
    for f in VALUE_FUNCS:
        np.testing.assert_allclose(got[f], ref[f], rtol=1e-6, err_msg=f)
    np.testing.assert_array_equal(got["area"], filled["area"])
