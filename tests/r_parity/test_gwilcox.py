"""Parity port of R misha ``test-gwilcox.R`` (sliding Wilcoxon test).

``gwilcox`` on a dense track (whole-genome and over a screened scope) matches R,
as does the ``intervals.set.out`` round-trip.

Divergence (not asserted): as in ``test-gsegment``, R uses a tiny
``gmax.data.size`` to force the disk-streaming ``intervals.set.out`` path;
pymisha enforces ``CONFIG['max_data_size']`` there, so we run at the package
default (1e9). The written result matches R's baseline.
"""

from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline


def test_gwilcox_fixedbin(overlay_db):
    assert_matches_baseline(
        pm.gwilcox("test.fixedbin", 100000, 1000, maxpval=0.000001, intervals=pm.gintervals([1, 2], 0, -1)),
        "gwilcox.fixedbin",
    )


def test_gwilcox_fixedbin_screening(overlay_db):
    intervs = pm.gscreen("test.fixedbin < 0.2", pm.gintervals([1, 2], 0, -1))
    assert_matches_baseline(
        pm.gwilcox("test.fixedbin", 100000, 1000, maxpval=0.0001, intervals=intervs),
        "gwilcox.fixedbin_screening",
    )


def test_gwilcox_fixedbin_interval_setting(overlay_db, track_namer):
    t = track_namer()
    pm.gwilcox(
        "test.fixedbin", 100000, 1000, maxpval=0.000001,
        intervals=pm.gintervals([1, 2], 0, -1), intervals_set_out=t,
    )
    assert_matches_baseline(pm.gintervals_load(t), "gwilcox.fixedbin_interval_setting")
