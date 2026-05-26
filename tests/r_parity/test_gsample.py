"""Parity port of R misha ``test-gsample.R`` (random value sampling).

Both cases ``xfail(strict=True)``:

* ``GAP_RNG`` -- ``gsample`` draws ``n`` random track values. R seeds R's RNG
  (``set.seed(60427)``); pymisha uses its own RNG. The drawn subsets differ, so
  the value vectors can't match -- this is reproducibility across two RNGs, not
  a correctness bug.
* The 2D (``test.rects``) case additionally isn't supported: pymisha's
  ``gsample`` requires 1D ``chrom/start/end`` intervals.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_RNG = "gsample draws differ across R's RNG and pymisha's RNG (not reproducible cross-impl)"
GAP_RNG_2D = "gsample over 2D intervals not supported (+ R/pymisha RNG mismatch)"

_CASES = {
    "gsample.sparse": (lambda: pm.gsample("test.sparse", 100, pm.gintervals([1, 2])), GAP_RNG),
    "gsample.rects": (lambda: pm.gsample("test.rects", 100, pm.gintervals_2d([1, 2])), GAP_RNG_2D),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_gsample(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
