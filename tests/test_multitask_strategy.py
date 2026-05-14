"""Tests for `multitasking_strategy` in gextract (R 5.6.18 parity).

R 5.6.18 introduced `getOption("gmultitasking.strategy")` so users could
route many-track / few-interval workloads through a track-parallel mode
(each worker handles a subset of expressions across all intervals)
instead of the default tile-parallel mode (each worker handles a chunk
of intervals across all expressions).

These tests verify:
1. `multitasking_strategy="tracks"` produces output identical to serial.
2. `multitasking_strategy="tiles"` is the existing chrom-parallel path.
3. `multitasking_strategy="auto"` picks `tracks` when expr count is high.
"""
from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

import pymisha as pm


def _make_intervals():
    """A small per-chrom set so tile-parallel has work to do."""
    rows = []
    for chrom, end in [("1", 5000), ("2", 3000)]:
        for s in range(0, end, 500):
            rows.append({"chrom": chrom, "start": s, "end": s + 500})
    return pd.DataFrame(rows)


def _many_exprs(n=10):
    return [f"dense_track + {i}.0" for i in range(n)]


def _restore(config, saved):
    config.update(saved)
    for k in list(config.keys()):
        if k not in saved:
            del config[k]


def test_track_parallel_matches_serial():
    """gextract with multitasking_strategy='tracks' equals the serial result."""
    intervals = _make_intervals()
    exprs = _many_exprs(10)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "min_processes": 1, "max_processes": 1})
        serial = pm.gextract(exprs, intervals, iterator=100)

        config.update({
            "multitasking": True,
            "min_processes": 3, "max_processes": 3,
            "multitasking_strategy": "tracks",
        })
        parallel = pm.gextract(exprs, intervals, iterator=100)
    finally:
        _restore(config, saved)

    cols = ["chrom", "start", "end", "intervalID"]
    pdt.assert_frame_equal(
        serial.sort_values(cols).reset_index(drop=True),
        parallel.sort_values(cols).reset_index(drop=True),
    )


def test_track_parallel_matches_tile_parallel():
    """tracks-mode and tiles-mode produce equivalent rows."""
    intervals = _make_intervals()
    exprs = _many_exprs(10)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({
            "multitasking": True,
            "min_processes": 3, "max_processes": 3,
            "multitasking_strategy": "tiles",
        })
        tiles = pm.gextract(exprs, intervals, iterator=100)

        config.update({"multitasking_strategy": "tracks"})
        tracks = pm.gextract(exprs, intervals, iterator=100)
    finally:
        _restore(config, saved)

    cols = ["chrom", "start", "end", "intervalID"]
    pdt.assert_frame_equal(
        tiles.sort_values(cols).reset_index(drop=True),
        tracks.sort_values(cols).reset_index(drop=True),
    )


def test_strategy_auto_default_uses_tiles_for_few_exprs():
    """auto with <8 exprs sticks to tiles (current default behavior)."""
    intervals = _make_intervals()
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(["dense_track"], intervals, iterator=100)

        config.update({
            "multitasking": True,
            "min_processes": 3, "max_processes": 3,
            "multitasking_strategy": "auto",
        })
        auto = pm.gextract(["dense_track"], intervals, iterator=100)
    finally:
        _restore(config, saved)

    cols = ["chrom", "start", "end", "intervalID"]
    pdt.assert_frame_equal(
        serial.sort_values(cols).reset_index(drop=True),
        auto.sort_values(cols).reset_index(drop=True),
    )


def test_strategy_auto_uses_tracks_for_many_exprs():
    """auto with >=8 exprs equals the tracks-mode output."""
    intervals = _make_intervals()
    exprs = _many_exprs(10)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "max_processes": 1})
        serial = pm.gextract(exprs, intervals, iterator=100)

        config.update({
            "multitasking": True,
            "min_processes": 3, "max_processes": 3,
            "multitasking_strategy": "auto",
        })
        auto = pm.gextract(exprs, intervals, iterator=100)
    finally:
        _restore(config, saved)

    cols = ["chrom", "start", "end", "intervalID"]
    pdt.assert_frame_equal(
        serial.sort_values(cols).reset_index(drop=True),
        auto.sort_values(cols).reset_index(drop=True),
    )


def test_invalid_strategy_rejected():
    """Unknown strategy values are rejected."""
    intervals = _make_intervals()
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({
            "multitasking": True,
            "max_processes": 2,
            "multitasking_strategy": "bogus",
        })
        try:
            pm.gextract(["dense_track"], intervals, iterator=100)
        except (ValueError, KeyError) as exc:
            assert "bogus" in str(exc) or "strategy" in str(exc).lower()
            return
        raise AssertionError("Expected ValueError/KeyError for invalid strategy")
    finally:
        _restore(config, saved)


def test_strategy_tracks_routes_through_track_parallel(monkeypatch):
    """Confirm strategy='tracks' invokes the track-parallel orchestrator."""
    from pymisha import extract as _extract_mod

    calls = []
    orig = getattr(_extract_mod, "_parallel_extract_tracks", None)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return orig(*args, **kwargs) if orig is not None else None

    monkeypatch.setattr(_extract_mod, "_parallel_extract_tracks", spy, raising=False)

    intervals = _make_intervals()
    exprs = _many_exprs(10)
    config = pm.CONFIG
    saved = config.copy()
    try:
        config.update({
            "multitasking": True,
            "min_processes": 3, "max_processes": 3,
            "multitasking_strategy": "tracks",
        })
        pm.gextract(exprs, intervals, iterator=100)
    finally:
        _restore(config, saved)

    assert len(calls) == 1, "_parallel_extract_tracks should be invoked once"


def test_strategy_auto_with_many_exprs_routes_through_track_parallel(monkeypatch):
    """auto + >= 8 exprs should use the track-parallel orchestrator."""
    from pymisha import extract as _extract_mod

    calls = []
    orig = getattr(_extract_mod, "_parallel_extract_tracks", None)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return orig(*args, **kwargs) if orig is not None else None

    monkeypatch.setattr(_extract_mod, "_parallel_extract_tracks", spy, raising=False)

    intervals = _make_intervals()
    exprs = _many_exprs(10)
    config = pm.CONFIG
    saved = config.copy()
    try:
        config.update({
            "multitasking": True,
            "min_processes": 3, "max_processes": 3,
            "multitasking_strategy": "auto",
        })
        pm.gextract(exprs, intervals, iterator=100)
    finally:
        _restore(config, saved)

    assert len(calls) == 1, (
        "_parallel_extract_tracks should run for auto + many exprs; got 0 calls"
    )
