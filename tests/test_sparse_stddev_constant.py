"""Regression test for the sparse-track ``stddev`` numerical-stability fix.

A bin whose overlapping sparse values are all identical must report a standard
deviation of exactly ``0``. The previous one-pass ``Sum(x^2) - n*mean^2``
formula suffered catastrophic cancellation for large-magnitude constant values
and returned a tiny nonzero stddev; the Welford online algorithm (matching R
misha) returns exactly ``0``. This runs on the bundled small test DB (no NFS).
"""

import numpy as np

import pymisha as pm


def _cleanup(track_name: str) -> None:
    if pm.gtrack_exists(track_name):
        pm.gtrack_rm(track_name, force=True)
        pm._pymisha.pm_dbreload()


def test_sparse_stddev_constant_bin_is_exactly_zero():
    track = "test.tmp_sparse_stddev_const"
    _cleanup(track)
    try:
        # Several identical, large-magnitude values inside one 5 kb bin: the
        # one-pass variance formula cancels to a tiny positive residual here.
        starts = [1000, 2000, 3000, 4000]
        intervals = pm.gintervals(["1"] * len(starts), starts, [s + 1 for s in starts])
        pm.gtrack_create_sparse(track, "tmp", intervals, [1_000_000.0] * len(starts))
        pm._pymisha.pm_dbreload()

        pm.gvtrack_create("vt_sd", track, func="stddev")
        df = pm.gextract("vt_sd", pm.gintervals(["1"], [0], [5000]), iterator=5000)
        vals = df["vt_sd"].to_numpy(dtype=float)

        assert vals.size == 1
        # Exact zero, not merely close: the cancellation residual was ~1e-4.
        assert vals[0] == 0.0, f"expected exact 0.0, got {vals[0]!r}"
    finally:
        pm.gvtrack_rm("vt_sd")
        _cleanup(track)


def test_sparse_stddev_matches_numpy_sample_std():
    """For non-constant values, sparse stddev matches the sample (n-1) std."""
    track = "test.tmp_sparse_stddev_var"
    _cleanup(track)
    try:
        starts = [1000, 2000, 3000, 4000, 5000]
        vals_in = [10.0, 12.0, 9.0, 15.0, 11.0]
        intervals = pm.gintervals(["1"] * len(starts), starts, [s + 1 for s in starts])
        pm.gtrack_create_sparse(track, "tmp", intervals, vals_in)
        pm._pymisha.pm_dbreload()

        pm.gvtrack_create("vt_sd", track, func="stddev")
        df = pm.gextract("vt_sd", pm.gintervals(["1"], [0], [6000]), iterator=6000)
        got = float(df["vt_sd"].to_numpy(dtype=float)[0])

        expected = float(np.std(vals_in, ddof=1))
        assert np.isclose(got, expected, rtol=1e-6, atol=1e-6)
    finally:
        pm.gvtrack_rm("vt_sd")
        _cleanup(track)
