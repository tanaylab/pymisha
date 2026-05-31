"""Parity port of R misha ``test-gtrack.lookup.R`` (lookup-table tracks).

Creating a 1D track from a 2D lookup table over two expressions
(``force.binning`` on and off) and extracting it matches R exactly.

Open gap marked ``xfail(strict=True)``:

* ``GAP_2D`` -- the 2D lookup case builds a *rectangles* track with
  ``force_binning=False``, so out-of-range cells get a NaN value (baseline:
  9740 NaN + 4322 non-NaN = 14062). This is a MULTI-LAYER NaN-retention gap
  (the generic 2D-create path was fixed in gtrack_create); investigated and
  deferred as a deeper 2D-writer effort:
  (1) ``gtrack_lookup`` evaluates the 2D scope over ``gintervals_2d_all()``
  (diagonal) rather than ``mode="full"`` -> off-diagonal cells missing
  (returns ``None`` for the off-diagonal query without this);
  (2) ``QuadTree.insert`` (``_quadtree.py``) drops NaN-valued objects to keep
  the stat-tree aggregations NaN-free; storing them (but excluding from stats)
  recovers ~94% (4322 -> 13283 rows);
  (3) a further ~779 NaN rects are still dropped by another layer (NOT the C++
  indexed converter, which has no NaN filter - likely glookup's expression-NaN
  handling or a full-mode scope boundary). Completing this needs all three
  layers + validation that 2D-vtrack stat aggregations stay NaN-safe.
"""

from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_2D = "2D lookup builds a RECTS track with NaN values; QuadTree.insert drops NaN-valued rects (R retains them) + diagonal-vs-full scope"

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
    "gtrack.lookup.2D_intervals": (_lookup_2d, None),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gtrack_lookup(baseline_id, overlay_db, track_namer, request):
    builder, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(builder(track_namer), baseline_id)
