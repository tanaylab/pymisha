"""Parity guard for multitask gquantiles after the no-sort merge change.

The forked-kid path merges per-kid reservoirs and now skips the O(N log N) sort
of the merged reservoir, selecting order statistics via the nth_element fast
path (exact, no sub-sampling) or a lazy sort inside get_percentile (sub-sampled,
approximate). Forked output must equal serial output exactly when the whole
stream fits the reservoir.

choose_num_kids forks only when num_intervals >= target, so these tests use 8
intervals (with max_processes=4 + zeroed floor) to actually exercise the forked
parent-merge code rather than the single-process path.
"""
import numpy as np
import pandas as pd
import pytest

import pymisha as pm

PCTS = [i / 20 for i in range(21)]
_FORCE_MT = {"multitasking": True, "max_processes": 4,
             "min_intervs4process": 0, "min_scope4process": 0, "progress": False}

# 8 contiguous intervals tiling chrom 1 -> num_intervals (8) >= target (4) -> forks.
_IV8 = pd.DataFrame({
    "chrom": ["1"] * 8,
    "start": list(range(0, 400000, 50000)),
    "end": list(range(50000, 450000, 50000)),
})


def _gq(intervals, **cfg_over):
    saved = {k: pm.CONFIG.get(k) for k in cfg_over}
    pm.CONFIG.update(cfg_over)
    try:
        return pm.gquantiles("dense_track", PCTS, intervals).to_numpy(dtype=float)
    finally:
        for k, v in saved.items():
            if v is None:
                pm.CONFIG.pop(k, None)
            else:
                pm.CONFIG[k] = v


def test_multitask_matches_serial_exact():
    serial = _gq(_IV8, multitasking=False)
    parallel = _gq(_IV8, **_FORCE_MT)
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=0, equal_nan=True)


def test_multitask_matches_serial_more_workers():
    cfg = dict(_FORCE_MT)
    cfg["max_processes"] = 8
    serial = _gq(_IV8, multitasking=False)
    parallel = _gq(_IV8, **cfg)
    np.testing.assert_allclose(parallel, serial, rtol=0, atol=0, equal_nan=True)


def test_multitask_subsample_runs_monotonic():
    # Force sub-sampling (tiny reservoir) on the forked path: result is
    # approximate but must be finite and non-decreasing in percentile.
    with pytest.warns(RuntimeWarning):
        q = _gq(_IV8, max_data_size=200, **_FORCE_MT)
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q) >= -1e-9)
