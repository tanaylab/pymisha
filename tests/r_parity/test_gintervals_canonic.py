"""Parity port of R misha ``test-gintervals.canonic.R`` (1 regression).

R seeds a row-shuffle of the input; canonic is order-independent so it is not
reproduced.
"""
from __future__ import annotations

import pandas as pd

import pymisha as pm

from .baseline import assert_matches_baseline


def test_gintervals_canonic_1():
    i1 = pm.gscreen("test.fixedbin>0.16 & test.fixedbin<0.19", pm.gintervals([1, 2]))
    i2 = pm.gscreen("test.fixedbin>0.13 & test.fixedbin<0.17", pm.gintervals([1, 2]))
    assert_matches_baseline(pm.gintervals_canonic(pd.concat([i1, i2], ignore_index=True)), "gintervals.canoic.1")
