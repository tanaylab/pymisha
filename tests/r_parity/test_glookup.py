"""Parity port of R misha ``test-glookup.R`` (4 regressions)."""
from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline, has_baseline

M1 = np.arange(1, 16, dtype=float).reshape(3, 5).T  # R's matrix(1:15, nrow = 5, ncol = 3)

_CASES = {
    "glookup_fixedbin_sparse_result": lambda: pm.glookup(
        M1, "test.fixedbin", np.linspace(0.1, 0.2, 6), "test.sparse", np.linspace(0.25, 0.48, 4),
        intervals=pm.gintervals([1, 2]), iterator="test.fixedbin"),
    "glookup_fixedbin_sparse_no_force_result": lambda: pm.glookup(
        M1, "test.fixedbin", np.linspace(0.1, 0.2, 6), "test.sparse", np.linspace(0.25, 0.48, 4),
        intervals=pm.gintervals([1, 2]), force_binning=False, iterator="test.fixedbin"),
    "glookup_rects_no_force_result": lambda: pm.glookup(
        M1, "test.rects", np.linspace(50, 100, 6), "test.rects / 2", np.linspace(0, 40, 4),
        intervals=pm.gintervals_2d(chroms1=[2, 3], chroms2=[2, 4]), force_binning=False),
    "glookup_computed2d_no_force_result": lambda: pm.glookup(
        M1, "test.computed2d", np.linspace(5000000, 10000000, 6), "test.computed2d / 2", np.linspace(0, 4000000, 4),
        intervals=pm.gintervals_2d(chroms1=[6, 5], chroms2=[8, 9]), force_binning=False),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_glookup(bid):
    if not has_baseline(bid):
        pytest.skip(f"R baseline {bid} not in the snapshot")
    assert_matches_baseline(_CASES[bid](), bid)
