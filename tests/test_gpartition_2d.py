"""2D gpartition (R test-2d-parity.R:714): one row per iterator rectangle with its
1-based bin, the same from a per-pair and an indexed 2D interval set."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha


@pytest.fixture()
def part_track():
    iv = pm.gintervals_2d(
        chroms1=["chr1"] * 6 + ["chr2"] * 6, starts1=list(range(100, 700, 100)) * 2,
        ends1=list(range(150, 750, 100)) * 2,
        chroms2=["chr1"] * 6 + ["chr2"] * 6, starts2=list(range(200, 800, 100)) * 2,
        ends2=list(range(250, 850, 100)) * 2)
    vals = np.linspace(0.1, 1.2, len(iv))
    pm.gtrack_2d_create("test.track_2d_part", "Test Track", iv, vals)
    yield iv, vals
    for s in ("test.legacy_2d_part", "test.indexed_2d_part"):
        if pm.gintervals_exists(s):
            pm.gintervals_rm(s, force=True)
    pm.gtrack_rm("test.track_2d_part", force=True)
    _pymisha.pm_dbreload()


def test_gpartition_2d(part_track):
    iv, vals = part_track
    breaks = [0, 0.5, 1.0, 1.5]
    got = pm.gpartition("test.track_2d_part", breaks, intervals=iv)
    assert list(got.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "bin"]
    want = iv.copy()
    want["bin"] = np.digitize(vals.astype(np.float32).astype(float), breaks, right=True)  # stored as float
    pd.testing.assert_frame_equal(got.astype({"chrom1": str, "chrom2": str}),
                                  want.astype({"chrom1": str, "chrom2": str}), check_dtype=False)


def test_gpartition_2d_legacy_and_indexed_sets(part_track, monkeypatch):
    iv, _ = part_track
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 10)  # big sets, as R's gmax.data.size = 10
    pm.gintervals_save(iv, "test.legacy_2d_part")
    pm.gintervals_save(iv, "test.indexed_2d_part")
    pm.gintervals_2d_convert_to_indexed("test.indexed_2d_part", remove_old=True, force=True)
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 10_000_000)
    breaks = [0, 0.5, 1.0, 1.5]
    a = pm.gpartition("test.track_2d_part", breaks, intervals="test.legacy_2d_part")
    b = pm.gpartition("test.track_2d_part", breaks, intervals="test.indexed_2d_part")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(iv)


def test_gpartition_2d_drops_values_outside_the_breaks(part_track):
    iv, vals = part_track
    got = pm.gpartition("test.track_2d_part", [0.3, 0.7], intervals=iv)
    v = vals.astype(np.float32).astype(float)
    assert got["bin"].tolist() == [1] * int(((v > 0.3) & (v <= 0.7)).sum())
    assert pm.gpartition("test.track_2d_part", [5, 6], intervals=iv) is None
