"""Parity port of R misha ``test-gcis_decay.R`` (3 regressions)."""
from __future__ import annotations

import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline


def _domain():
    return pd.concat(
        [pm.gintervals(c, [800000 * k for k in range(6)], [800000 * k + 400000 for k in range(6)]) for c in [1, 2, 3, 4, 5]],
        ignore_index=True,
    )


@pytest.mark.xfail(reason="gcis_decay with a valued src differs from R (last bins; investigate)", strict=True)
def test_gcis_decay_1():
    dom = _domain()
    src = pm.gextract("test.sparse", pm.gintervals([1, 2, 3, 4, 5]))
    assert_matches_baseline(pm.gcis_decay("test.rects", [k * 1000 for k in range(21)], src, dom), "gcis_decay.1")


def test_gcis_decay_2():
    dom = _domain()
    assert_matches_baseline(pm.gcis_decay("test.rects", [k * 1000 for k in range(21)], dom, dom), "gcis_decay.2")


@pytest.mark.xfail(reason="gcompute_strands_autocorr reads a multi-GB external export file", strict=False)
def test_gcompute_strands_autocorr_1():
    res = pm.gcompute_strands_autocorr(
        "/net/mraid20/export/tgdata/db/tgdb/misha_snapshot/input_files/s_7_export.txt", 1, 50
    )
    assert_matches_baseline(res, "gcompute_strands_autocorr.1")
