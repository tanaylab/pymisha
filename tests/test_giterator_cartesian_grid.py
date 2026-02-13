import pandas as pd
import pytest

import pymisha as pm


def test_giterator_cartesian_grid_is_exported():
    assert hasattr(pm, "giterator_cartesian_grid")


def test_giterator_cartesian_grid_basic_self_product():
    intervs = pm.gintervals(["1", "1"], [100, 300], [200, 400])
    grid = pm.giterator_cartesian_grid(intervs, [-20, 20])

    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    assert len(grid) == 4
    assert (grid["end1"] > grid["start1"]).all()
    assert (grid["end2"] > grid["start2"]).all()


def test_giterator_cartesian_grid_uses_second_intervals_and_expansion():
    intervs1 = pm.gintervals(["1"], [100], [200])
    intervs2 = pm.gintervals(["2"], [1000], [1100])
    grid = pm.giterator_cartesian_grid(intervs1, [-10, 10], intervs2, [-20, 0, 20])

    assert len(grid) == 2
    assert (grid["chrom1"] == "1").all()
    assert (grid["chrom2"] == "2").all()


def test_giterator_cartesian_grid_band_idx_filters_pairs():
    intervs = pm.gintervals(["1", "1"], [100, 300], [200, 400])
    grid_diag = pm.giterator_cartesian_grid(intervs, [-10, 10], min_band_idx=0, max_band_idx=0)
    grid_upper = pm.giterator_cartesian_grid(intervs, [-10, 10], min_band_idx=-1, max_band_idx=0)

    assert len(grid_diag) == 2
    assert len(grid_upper) == 3


def test_giterator_cartesian_grid_invalid_band_usage_raises():
    intervs1 = pm.gintervals(["1"], [100], [200])
    intervs2 = pm.gintervals(["1"], [300], [400])

    with pytest.raises(ValueError, match="band.idx limit"):
        pm.giterator_cartesian_grid(
            intervs1,
            [-10, 10],
            intervs2,
            [-10, 10],
            min_band_idx=0,
            max_band_idx=0,
        )

    with pytest.raises(ValueError, match="Both min_band_idx and max_band_idx"):
        pm.giterator_cartesian_grid(intervs1, [-10, 10], min_band_idx=0)

    with pytest.raises(ValueError, match="exceeds"):
        pm.giterator_cartesian_grid(intervs1, [-10, 10], min_band_idx=1, max_band_idx=0)
