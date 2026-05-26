"""Parity port of R misha ``test-gtrack.var.R`` (track variables).

The single regression reads the precomputed ``pv.percentiles`` track variable
off ``test.fixedbin``; it matches R exactly. The file's other assertions
(``gtrack.var.set``/``.ls``/``.rm`` round-trips, error paths) are behavioral
``expect_equal``/``expect_error`` checks, not frozen baselines.
"""

from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline


def test_gtrack_var_get_pv_percentiles():
    assert_matches_baseline(
        pm.gtrack_var_get("test.fixedbin", "pv.percentiles"), "test.fixedbin.pv.percentiles"
    )
