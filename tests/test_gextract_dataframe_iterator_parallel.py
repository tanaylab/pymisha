"""A DataFrame iterator must not pin a many-track gextract to one process.

Regression for the cold-NFS case: `iterator=<DataFrame>` used to force
`multitasking=False` for the whole extraction, so a 50-track scan ran
serially and took ~25x longer than R misha (which forks one kid per
expression group). See _config_no_mt / _tracks_workload_too_small_for_fork.
"""
import contextlib
import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _shared
from pymisha.extract import (
    _resolve_parallel_strategy,
    _tracks_workload_too_small_for_fork,
)


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
    assert _extract._effective_max_procs(
        {"min_processes": 1, "max_processes": 89}) == 4
    # max_processes below the core count still wins.
    assert _extract._effective_max_procs(
        {"min_processes": 1, "max_processes": 2}) == 2
    # min_processes floors it, as the C++ side does.
    assert _extract._effective_max_procs(
        {"min_processes": 8, "max_processes": 89}) == 8


def test_max_processes_default_tracks_core_count():
    """Default mirrors R's gmax.processes (70% of cores), floored at 4."""
    expected = max(4, int((os.cpu_count() or 1) * 0.7))
    assert _shared.CONFIG["max_processes"] == expected


def test_tracks_floor_counts_intervals_times_exprs():
    """The track-parallel floor scales with expressions, not base-pairs."""
    cfg = {"min_intervs4process": 250_000}
    many_tracks = pd.DataFrame({"chrom": ["1"] * 17_000, "start": range(17_000),
                                "end": range(1, 17_001)})
    # 17k intervals x 56 exprs = 952k visits -> worth forking, even though the
    # bp scope (17k x 1bp here, ~5Mbp in the real case) is far under the
    # 1e9 bp/worker floor that _workload_too_small_for_fork applies.
    assert not _tracks_workload_too_small_for_fork(many_tracks, 56, cfg)
    # Same intervals, one expression -> not worth forking.
    assert _tracks_workload_too_small_for_fork(many_tracks, 1, cfg)
    # Many expressions but a tiny scope -> not worth forking.
    tiny = many_tracks.head(5)
    assert _tracks_workload_too_small_for_fork(tiny, 56, cfg)


def test_dataframe_iterator_resolves_to_tracks_strategy():
    """After preprocessing, a DataFrame iterator becomes -1 (an int)."""
    assert _resolve_parallel_strategy("auto", n_exprs=56, iterator=-1) == "tracks"
    assert _resolve_parallel_strategy("auto", n_exprs=2, iterator=-1) == "tiles"


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


def _both_ways(call):
    """Run *call* serially and through the track-parallel path."""
    with _force(False):
        serial = call()
    # min_intervs4process=1 admits this small workload to the fork path.
    with _force(True, min_intervs4process=1):
        parallel = call()
    return serial, parallel


def _assert_same(serial, parallel, sort_cols=("chrom", "start")):
    assert list(serial.columns) == list(parallel.columns)
    assert serial.shape == parallel.shape
    pd.testing.assert_frame_equal(
        serial.sort_values(list(sort_cols)).reset_index(drop=True),
        parallel.sort_values(list(sort_cols)).reset_index(drop=True),
    )


def _grid(n=100):
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
    serial, parallel = _both_ways(
        lambda: pm.gextract(exprs, intervals=intervals, iterator=intervals,
                            progress=False)
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
