"""Parity port of R misha ``test-gtrack.import3.R`` (2D track import).

``gtrack.2d.import`` of a contacts file (``f4``) followed by a whole-genome 2D
extract matches R exactly. R runs it under a small ``gmax.data.size`` to force
chunked import; the imported result is identical, so we run at the package
default.

The file's second block checks ``gtrack.import(..., attrs=)`` with an
``expect_equal`` on a track attribute -- not a frozen baseline -- so it isn't
ported here.
"""

from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline

_F4 = "/net/mraid20/export/tgdata/db/tgdb/misha_snapshot/input_files/f4"


def test_gtrack_2d_import_gmax_option(overlay_db, track_namer):
    t = track_namer()
    pm.gtrack_2d_import(t, "aaa7", _F4)
    assert_matches_baseline(
        pm.gextract(t, pm.gintervals_2d_all(mode="full"), colnames=[t]),
        "track.import_gmax_option",
    )
