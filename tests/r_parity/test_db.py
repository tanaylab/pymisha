"""Parity port of R misha ``test-db.R``.

The file's sole frozen baseline is ``gtrack.modify_and_extract_for_fixedbin``:
create a dense track from ``test.fixedbin``, modify it over a screened scope
(``> 0.17`` or NaN), and extract. It matches R exactly. The R ``is.na(x)`` idiom
maps to ``np.isnan(x)`` in pymisha track expressions.

The file's many other ``test_that`` blocks are behavioral
``expect_true``/``expect_equal`` checks of ``gdir.cd``/``gtrack.ls``/
``gtrack.exists``/``gtrack.var`` round-trips -- not frozen baselines -- so they
are not ported here.
"""

from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline


def test_gtrack_modify_and_extract_for_fixedbin(overlay_db, track_namer):
    t = track_namer()
    pm.gtrack_create(t, "", "test.fixedbin")
    intervs = pm.gscreen("test.fixedbin > 0.17 | np.isnan(test.fixedbin)", pm.gintervals([1, 7]))
    pm.gtrack_modify(t, "test.fixedbin + test.fixedbin", intervs)
    assert_matches_baseline(
        pm.gextract(t, pm.gintervals([1, 2]), colnames=[t]),
        "gtrack.modify_and_extract_for_fixedbin",
    )
