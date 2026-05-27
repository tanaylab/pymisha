"""Parity port of R misha ``test-gtrack.create.R`` (track creation).

These tests create a track in the writable :func:`overlay_db` (a symlink copy of
R's read-only test DB), extract it, and compare to R's frozen baseline. 1D dense
and sparse creation -- from an expression, with an integer / track-name /
intervals iterator, and via ``gtrack.create_sparse`` -- match R exactly.

Open gaps marked ``xfail(strict=True)``:

* ``GAP_ARRAY`` -- creating a track with an *array* iterator yields an array
  track, which ``gextract`` cannot read back yet.
* ``GAP_2D_CREATE`` -- creating a 2D track from a track *expression*
  (``test.rects+10``): the scanner can't iterate a rectangles track in the
  track-create path yet.
* ``GAP_2D_ITER`` -- ``gtrack.2d.create`` stores the supplied values in input
  row order; pymisha's 2D ``gscreen`` row order differs from R's, so the
  position-encoded ``1:nrow`` values land on different rectangles.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY = "array iterator yields an array track; gextract array-track read not supported"
GAP_2D_CREATE = "2D track creation from a track expression not supported (scanner can't iterate rects in create path)"
GAP_2D_ITER = "gtrack.2d.create values follow input row order; pymisha 2D gscreen order differs from R"


def _iv12():
    return pm.gintervals([1, 2], 0, 1000000)


def _create_extract(namer, expr, scope, iterator=None):
    """Create a track from ``expr`` with ``iterator``, extract over ``scope``."""
    t = namer()
    pm.gtrack_create(t, "", expr, iterator=iterator)
    return pm.gextract(t, scope, colnames=[t])


_CASES = {
    # R: gtrack.create(tmp, "", "test.fixedbin+1"); gextract over c(1,2) 0..1e6
    "gtrack.create.2": (lambda n: _create_extract(n, "test.fixedbin+1", _iv12()), None),
    # iterator = "test.sparse" (sparse output)
    "gtrack.create.3": (lambda n: _create_extract(n, "test.fixedbin+1", _iv12(), "test.sparse"), None),
    # iterator = "test.array" -> array track (unreadable)
    "gtrack.create.4": (lambda n: _create_extract(n, "test.fixedbin+1", _iv12(), "test.array"), None),
    # "test.rects+10": 2D track creation from an expression
    "gtrack.create.5": (
        lambda n: _create_extract(n, "test.rects+10", pm.gintervals_2d([2, 3], chroms2=[2, 4])),
        None,
    ),
    # iterator = a sparse-intervals DataFrame
    "gtrack.create.6": (
        lambda n: _create_extract(
            n, "test.fixedbin+1", _iv12(), pm.giterator_intervals("test.sparse", pm.gintervals([1, 3, 4]))
        ),
        None,
    ),
    # iterator = an array-intervals DataFrame -> array track (unreadable)
    "gtrack.create.7": (
        lambda n: _create_extract(
            n, "test.fixedbin+1", _iv12(), pm.giterator_intervals("test.array", pm.gintervals([1, 3, 4]))
        ),
        None,
    ),
    # "test.rects+10" with a 2D-intervals iterator
    "gtrack.create.8": (
        lambda n: _create_extract(
            n,
            "test.rects+10",
            pm.gintervals_2d([2, 3, 3], chroms2=[2, 3, 4]),
            pm.giterator_intervals("test.rects", pm.gintervals_2d([2, 3, 5], chroms2=[2, 4, 7])),
        ),
        None,
    ),
}


def _create_sparse_9(namer):
    intervs = pm.gscreen("test.fixedbin > 0.2", pm.gintervals([1, 2]))
    t = namer()
    pm.gtrack_create_sparse(t, "", intervs, np.arange(1, len(intervs) + 1))
    return pm.gextract(t, _iv12(), colnames=[t])


def _create_2d_11(namer):
    intervs = pm.gscreen("test.rects > 80", pm.gintervals_2d([1, 2]))
    t = namer()
    pm.gtrack_2d_create(t, "", intervs, np.arange(1, len(intervs) + 1))
    return pm.gextract(t, pm.gintervals_2d([1, 2, 3]), colnames=[t])


_CASES["gtrack.create.9"] = (_create_sparse_9, None)
_CASES["gtrack.create.11"] = (_create_2d_11, GAP_2D_ITER)


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gtrack_create(baseline_id, overlay_db, track_namer, request):
    builder, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(builder(track_namer), baseline_id)


def test_gtrack_create_1(overlay_db):
    """R: gdir.create("aaaaaaaaaaaaa"); gtrack.create("aaaaaaaaaaaaa.bbbbbbbbbbb", ...).

    Divergence (not asserted): R's ``gtrack.create`` *errors* when the parent
    group is missing; pymisha auto-creates it. R guards this with an
    ``expect_error`` before the ``gdir.create`` -- a behavioral difference, not a
    regression baseline, so it is not exercised here.
    """
    pm.gdir_create("aaaaaaaaaaaaa", show_warnings=False)
    try:
        pm.gtrack_create("aaaaaaaaaaaaa.bbbbbbbbbbb", "", "test.fixedbin")
        r = pm.gextract("aaaaaaaaaaaaa.bbbbbbbbbbb", _iv12())
        assert_matches_baseline(r, "gtrack.create.1")
    finally:
        with contextlib.suppress(Exception):
            pm.gtrack_rm("aaaaaaaaaaaaa.bbbbbbbbbbb", force=True)
        with contextlib.suppress(Exception):
            pm.gdir_rm("aaaaaaaaaaaaa", recursive=True, force=True)
