"""Parity port of R misha ``test-gintervals.mapply.R`` (3 regressions)."""
from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline


def test_mapply_fixedbin_allgenome():
    assert_matches_baseline(
        pm.gintervals_mapply(lambda x: np.nanmax(x + 2), "test.fixedbin", intervals=pm.gintervals_all()),
        "gintervals.mapply.fixedbin.ALLGENOME",
    )


def test_mapply_generated_1d():
    assert_matches_baseline(
        pm.gintervals_mapply(lambda x: np.nanmax(x + 2), "test.generated_1d_1", intervals="test.bigintervs_1d_1"),
        "gintervals.mapply.test.generated_1d_1.test.bigintervs_1d_1",
    )


@pytest.mark.xfail(reason="GAPPLY.INTERVID global differs (pymisha uses enable_gapply_intervals kwarg)", strict=True)
def test_mapply_intervid():
    assert_matches_baseline(
        pm.gintervals_mapply(lambda x: 0.0, "test.fixedbin", intervals=pm.gintervals_all()),
        "gintervals.mapply.fixedbin.ALLGENOME.INTERVID",
    )
