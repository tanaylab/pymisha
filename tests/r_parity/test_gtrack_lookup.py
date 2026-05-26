"""Parity port of R misha ``test-gtrack.lookup.R`` (lookup-table tracks).

Creating a 1D track from a 2D lookup table over two expressions
(``force.binning`` on and off) and extracting it matches R exactly.

Open gap marked ``xfail(strict=True)``:

* ``GAP_2D`` -- the 2D lookup case builds a *rectangles* track; pymisha's
  ``gtrack_lookup`` produces no readable 2D track (gextract returns ``None``),
  the same 2D-track-creation gap seen in ``gtrack.create``.
"""

from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D = "2D lookup track creation not supported (gextract returns None); same gap as 2D gtrack_create"

# R: m1 <- matrix(1:15, nrow = 5, ncol = 3) -- column-major fill.
_M1 = np.arange(1, 16).reshape((5, 3), order="F")


def _lookup_1d(namer, force_binning):
    t = namer()
    pm.gtrack_lookup(
        t,
        "",
        _M1,
        "test.fixedbin",
        np.linspace(0.1, 0.2, 6),
        "test.sparse",
        np.linspace(0.25, 0.48, 4),
        force_binning=force_binning,
        iterator="test.fixedbin",
    )
    return pm.gextract(t, pm.gintervals([1, 3]), colnames=[t])


def _lookup_2d(namer):
    t = namer()
    pm.gtrack_lookup(
        t,
        "",
        _M1,
        "test.rects",
        np.linspace(50, 100, 6),
        "test.rects / 2",
        np.linspace(0, 40, 4),
        force_binning=False,
    )
    return pm.gextract(t, pm.gintervals_2d([3, 5], chroms2=[4, 6]), colnames=[t])


_CASES = {
    "gtrack.lookup.default_binning": (lambda n: _lookup_1d(n, True), None),
    "gtrack.lookup.no_force_binning": (lambda n: _lookup_1d(n, False), None),
    "gtrack.lookup.2D_intervals": (_lookup_2d, GAP_2D),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gtrack_lookup(baseline_id, overlay_db, track_namer, request):
    builder, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(builder(track_namer), baseline_id)
