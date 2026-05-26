"""Regression test for the ``distance.center`` virtual-track boundary fix.

``distance.center`` locates the source interval that contains the bin *center*.
The previous implementation searched by the bin *start* and only inspected the
first source interval starting at/after it, so when several source intervals
begin inside the bin it could miss the one actually containing the center and
return ``NaN``. It now searches by the center coordinate (matching R misha).
Runs on the bundled small test DB (no NFS).
"""

import numpy as np
import pandas as pd

import pymisha as pm


def test_distance_center_finds_containing_interval_when_not_first_in_bin():
    for v in pm.gvtrack_ls():
        pm.gvtrack_rm(v)
    # Single bin [0, 533), center 266. Two source intervals begin inside it:
    # (50, 100) does NOT contain the center, (250, 300) does (center 275).
    # The old start-keyed search stopped at (50, 100) and returned NaN.
    src = pd.DataFrame(
        {"chrom": ["1", "1"], "start": [50, 250], "end": [100, 300]}
    )
    pm.gvtrack_create("vt_dc", src, func="distance.center")
    df = pm.gextract("vt_dc", pm.gintervals(["1"], [0], [533]), iterator=533)
    assert len(df) == 1
    val = float(df["vt_dc"].to_numpy(dtype=float)[0])
    # center 266 is inside (250, 300); its center is 275 -> distance 9.
    assert val == 9.0, f"expected 9.0, got {val!r}"
    pm.gvtrack_rm("vt_dc")


def test_distance_center_nan_when_center_in_no_interval():
    for v in pm.gvtrack_ls():
        pm.gvtrack_rm(v)
    src = pd.DataFrame({"chrom": ["1"], "start": [50], "end": [100]})
    pm.gvtrack_create("vt_dc", src, func="distance.center")
    df = pm.gextract("vt_dc", pm.gintervals(["1"], [0], [533]), iterator=533)
    val = float(df["vt_dc"].to_numpy(dtype=float)[0])
    assert np.isnan(val), f"expected NaN, got {val!r}"
    pm.gvtrack_rm("vt_dc")
