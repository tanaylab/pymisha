"""A DataFrame iterator must not pin a many-track gextract to one process.

Regression for the cold-NFS case: `iterator=<DataFrame>` used to force
`multitasking=False` for the whole extraction, so a 50-track scan ran
serially and took ~25x longer than R misha (which forks one kid per
expression group). See _config_no_mt / _resolve_parallel_strategy.
"""
import contextlib
import os
import resource

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _shared
from pymisha.extract import _resolve_parallel_strategy


def test_dataframe_iterator_keeps_multitasking_for_track_parallel():
    """_config_no_mt(keep=True) leaves multitasking on."""
    prev = _shared.CONFIG.get("multitasking")
    _shared.CONFIG["multitasking"] = True
    try:
        id_map = np.array([1, 2, 3])
        with _shared._config_no_mt(id_map) as cfg:
            assert cfg["multitasking"] is False, "default still serialises"
        with _shared._config_no_mt(id_map, keep=True) as cfg:
            assert cfg["multitasking"] is True, "keep=True must not disable it"
        assert _shared.CONFIG["multitasking"] is True, "must restore"
    finally:
        if prev is None:
            _shared.CONFIG.pop("multitasking", None)
        else:
            _shared.CONFIG["multitasking"] = prev


def test_effective_max_procs_clamps_to_core_count(monkeypatch):
    """Worker count must not exceed the cores, mirroring C++ choose_num_kids."""
    from pymisha import extract as _extract

    monkeypatch.setattr(_extract.os, "cpu_count", lambda: 4)
    # max_processes far above the core count -> clamped to the cores.
    assert _extract._effective_max_procs({"max_processes": 89}) == 4
    # max_processes below the core count still wins.
    assert _extract._effective_max_procs({"max_processes": 2}) == 2


def test_max_processes_default_tracks_core_count():
    """Default mirrors R's gmax.processes: 70% of cores, at least 1 and at most 32
    (misha 5.11.20 - past that the per-worker cost outweighs the parallelism). The cap
    only binds above 46 cores. R has no floor on it (pymisha's min_processes is gone)."""
    expected = max(1, min(32, int((os.cpu_count() or 1) * 0.7)))
    assert _shared.CONFIG["max_processes"] == expected
    assert _shared.CONFIG["max_processes"] <= 32
    assert "min_processes" not in _shared.CONFIG


def test_auto_strategy_follows_r_misha():
    """R misha's .gmultitasking_strategy: "auto" picks track-parallel for at least 8
    expressions over at least 1000 intervals with an iterator of intervals (a DataFrame
    becomes -1 after preprocessing); an explicit strategy wins."""
    assert _resolve_parallel_strategy("auto", n_exprs=56, iterator=-1, n_intervals=17_000) == "tracks"
    assert _resolve_parallel_strategy("auto", n_exprs=7, iterator=-1, n_intervals=17_000) == "tiles"
    assert _resolve_parallel_strategy("auto", n_exprs=56, iterator=-1, n_intervals=999) == "tiles"
    # a bin size or an implicit iterator scans bins: the tiles split them over all the workers
    assert _resolve_parallel_strategy("auto", n_exprs=56, iterator=100, n_intervals=17_000) == "tiles"
    assert _resolve_parallel_strategy("auto", n_exprs=56, iterator=None, n_intervals=17_000) == "tiles"
    assert _resolve_parallel_strategy("tracks", n_exprs=2, iterator=100, n_intervals=5) == "tracks"
    assert _resolve_parallel_strategy("tiles", n_exprs=56, iterator=-1, n_intervals=17_000) == "tiles"


@contextlib.contextmanager
def _force(multitasking, **cfg):
    """Temporarily pin CONFIG so the fork decision is deterministic."""
    keys = ("multitasking", *cfg)
    prev = {k: _shared.CONFIG.get(k, _MISSING) for k in keys}
    _shared.CONFIG["multitasking"] = multitasking
    _shared.CONFIG.update(cfg)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is _MISSING:
                _shared.CONFIG.pop(k, None)
            else:
                _shared.CONFIG[k] = v


_MISSING = object()


def _both_ways(call, forks=True):
    """Run *call* serially and through the track-parallel path."""
    with _force(False):
        serial = call()
    with _force(True):
        faults = resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt
        parallel = call()
        assert (resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt > faults) == forks
    return serial, parallel


def _assert_same(serial, parallel, sort_cols=("chrom", "start")):
    assert list(serial.columns) == list(parallel.columns)
    assert serial.shape == parallel.shape
    pd.testing.assert_frame_equal(
        serial.sort_values(list(sort_cols)).reset_index(drop=True),
        parallel.sort_values(list(sort_cols)).reset_index(drop=True),
    )


def _grid(n=1000):
    return pm.gintervals("1", list(range(0, n * 200, 200)),
                         list(range(100, n * 200 + 100, 200)))


@pytest.mark.parametrize(
    "exprs",
    [
        pytest.param([f"dense_track + {i}" for i in range(10)], id="distinct"),
        # Repeated expressions must fall back, not mangle column names.
        pytest.param(["dense_track"] * 10, id="duplicate"),
    ],
)
def test_parallel_and_serial_agree_with_dataframe_iterator(exprs):
    """Values, coords and intervalIDs must be identical either way."""
    intervals = _grid()
    # The repeated expressions are extracted by the tiles, in one process: the scope is one
    # chromosome, and the iterator's intervals are not cut
    serial, parallel = _both_ways(
        lambda: pm.gextract(exprs, intervals=intervals, iterator=intervals,
                            progress=False),
        forks=len(set(exprs)) == len(exprs),
    )
    _assert_same(serial, parallel)


def test_parallel_respects_colnames():
    """colnames are applied positionally; the parallel column order must match."""
    exprs = [f"dense_track + {i}" for i in range(10)]
    names = [f"c{i}" for i in range(10)]
    intervals = _grid()
    serial, parallel = _both_ways(
        lambda: pm.gextract(exprs, intervals=intervals, iterator=intervals,
                            colnames=names, progress=False)
    )
    _assert_same(serial, parallel)
    assert [c for c in parallel.columns
            if c not in ("chrom", "start", "end", "intervalID")] == names


def test_parallel_respects_intervals_join():
    """intervals_join='intervals' attaches by intervalID - must survive forking."""
    exprs = [f"dense_track + {i}" for i in range(10)]
    intervals = _grid()
    intervals = intervals.assign(tag=[f"p{i}" for i in range(len(intervals))])
    serial, parallel = _both_ways(
        lambda: pm.gextract(exprs, intervals=intervals, iterator=intervals,
                            intervals_join="intervals", progress=False)
    )
    _assert_same(serial, parallel)
    assert "tag" in parallel.columns
