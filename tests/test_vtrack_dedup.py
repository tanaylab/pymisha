"""Output-parity guard for dense vtrack-instance dedup.

When several dense (FIXED_BIN) vtracks share the same source track and the same
sshift/eshift, the scanner now reads the source once (primary) and lets the
others (followers) extract their own reducer. Extracting them together must give
exactly the same values as extracting each one alone.
"""
import numpy as np

import pymisha as pm

FUNCS = ["avg", "sum", "min", "max", "nearest", "stddev", "exists", "size",
         "lse", "first", "last", "min.pos.relative", "max.pos.relative"]


def _make(funcs, sshift, eshift, prefix="vt"):
    names = []
    for i, f in enumerate(funcs):
        n = f"{prefix}_{i}"
        pm.gvtrack_create(n, "dense_track", func=f)
        pm.gvtrack_iterator(n, sshift=sshift, eshift=eshift)
        names.append(n)
    return names


def setup_function(_):
    pm.gvtrack_clear()


def teardown_function(_):
    pm.gvtrack_clear()


def _alone(name, iv):
    return pm.gextract(name, iv, iterator=50)[name].to_numpy(dtype=float)


def test_dedup_same_source_shift_matches_individual():
    names = _make(FUNCS, -100, 100)
    iv = pm.gintervals(1, 0, 500000)
    together = pm.gextract(names, iv, iterator=50)  # one scan -> dedup
    for n, f in zip(names, FUNCS, strict=True):
        np.testing.assert_allclose(
            together[n].to_numpy(dtype=float), _alone(n, iv),
            rtol=1e-6, atol=1e-6, equal_nan=True,
            err_msg=f"dedup mismatch for func={f}",
        )


def test_dedup_duplicate_func_same_source():
    # Two vtracks, same source/shift/func -> should share and agree.
    names = _make(["avg", "avg", "sum"], -250, 250)
    iv = pm.gintervals(1, 0, 500000)
    together = pm.gextract(names, iv, iterator=50)
    for n in names:
        np.testing.assert_allclose(
            together[n].to_numpy(dtype=float), _alone(n, iv),
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )


def test_mixed_shifts_two_groups():
    # Two distinct shift groups must not cross-contaminate.
    g1 = _make(["avg", "sum"], -100, 100, prefix="a")
    g2 = _make(["avg", "max"], -300, 300, prefix="b")
    names = g1 + g2
    iv = pm.gintervals(1, 0, 500000)
    together = pm.gextract(names, iv, iterator=50)
    for n in names:
        np.testing.assert_allclose(
            together[n].to_numpy(dtype=float), _alone(n, iv),
            rtol=1e-6, atol=1e-6, equal_nan=True,
        )


def test_single_vtrack_unaffected():
    names = _make(["lse"], -500, 500)
    iv = pm.gintervals(1, 0, 500000)
    together = pm.gextract(names, iv, iterator=50)
    np.testing.assert_allclose(
        together["vt_0"].to_numpy(dtype=float), _alone("vt_0", iv),
        rtol=1e-6, atol=1e-6, equal_nan=True,
    )
