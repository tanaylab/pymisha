"""Parity port of R misha ``test-gtrack.smooth.R`` (track smoothing).

Smoothing a dense or sparse track with ``LINEAR_RAMP`` / ``MEAN`` over a
``winsize`` window, then extracting it, matches R exactly (run in the writable
:func:`overlay_db`).

Open gap marked ``xfail(strict=True)``:

* ``GAP_ARRAY`` -- smoothing an *array* track yields an array track, which
  ``gextract`` cannot read back yet.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY = "smoothing an array track yields an array track; gextract array-track read not supported"


def _smooth_extract(namer, src, alg, iterator=None):
    t = namer()
    pm.gtrack_smooth(t, "", src, 10000, alg=alg, iterator=iterator)
    return pm.gextract(t, pm.gintervals([1, 2], 0, 1000000), colnames=[t])


_CASES = {
    "gtrack.smooth.fixedbin_LINEAR_RAMP": (lambda n: _smooth_extract(n, "test.fixedbin", "LINEAR_RAMP"), None),
    "gtrack.smooth.fixedbin_MEAN": (lambda n: _smooth_extract(n, "test.fixedbin", "MEAN"), None),
    "gtrack.smooth.sparse_LINEAR_RAMP": (lambda n: _smooth_extract(n, "test.sparse", "LINEAR_RAMP", 1000), None),
    "gtrack.smooth.array_LINEAR_RAMP": (lambda n: _smooth_extract(n, "test.array", "LINEAR_RAMP", 1000), None),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gtrack_smooth(baseline_id, overlay_db, track_namer, request):
    builder, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(builder(track_namer), baseline_id)
