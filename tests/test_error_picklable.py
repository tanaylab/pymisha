"""pymisha.error must survive pickling so multiprocessing propagates it."""

import multiprocessing as mp
import pickle

import pymisha as pm


def _raise(_):
    raise pm.error("real misha message")


def test_error_pickles():
    assert pickle.loads(pickle.dumps(pm.error("boom"))).args == ("boom",)


def test_error_survives_pool():
    """Without pymisha.error being importable this is an opaque PicklingError."""
    with mp.Pool(1) as pool:
        try:
            pool.map(_raise, [1])
        except pm.error as exc:
            assert "real misha message" in str(exc)
        else:
            raise AssertionError("expected pm.error")
