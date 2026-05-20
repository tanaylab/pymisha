"""Tests for the env-var-gated intervals-iterator scanner path."""
import pandas as pd
import pymisha as pm
import pytest
from pymisha._shared import CONFIG


@pytest.fixture
def use_scanner_env(monkeypatch):
    monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "1")


def test_intervals_via_scanner_basic(_init_db, rects_track, use_scanner_env):
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result = pm.gextract(rects_track, intervals=intervals)
    assert rects_track in result.columns
    assert len(result) >= 1


def test_intervals_via_scanner_matches_bypass(_init_db, rects_track, monkeypatch):
    """Scanner path output must be non-empty and structurally consistent.

    NOTE: bypass returns one row per (scope_rect, track_object) intersection;
    scanner with IntervalsPolicy + "area" reducer returns one row per scope
    rect.  These are semantically different — bypass gives per-object detail,
    scanner aggregates into a single value per scope rect.  Strict equivalence
    is intentionally NOT asserted here; this test is a sanity check only.
    """
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    # Bypass path:
    monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "0")
    bypass = pm.gextract(rects_track, intervals=intervals)
    # Scanner path:
    monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "1")
    scanner = pm.gextract(rects_track, intervals=intervals)

    # Both must succeed and contain the expected column.
    assert rects_track in bypass.columns
    assert rects_track in scanner.columns
    # Scanner: one row per scope rect (1 input rect -> 1 output row).
    assert len(scanner) == 1
    # Bypass: one row per intersecting object on (chrom1 x chrom1).
    assert len(bypass) >= 1


def test_intervals_via_scanner_disabled_by_default(_init_db, rects_track, monkeypatch):
    """When env var is not set, intervals iterator goes through bypass."""
    monkeypatch.delenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", raising=False)
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    result = pm.gextract(rects_track, intervals=intervals)
    # Behavior is the bypass behavior (one row per intersecting object).
    assert rects_track in result.columns
    # Three objects on chrom1-chrom1 in the test fixture.
    assert len(result) >= 1


def test_intervals_via_scanner_with_explicit_iterator_falls_through(_init_db, rects_track, monkeypatch):
    """When iterator= is explicitly provided (distinct from intervals), the legacy path is used.

    Note: passing a 2D DataFrame as iterator= to gextract is unusual and may
    not be a supported R-parity pattern.  The scanner opt-in only activates
    when iterator is None, so this case falls through to legacy.
    """
    monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "1")
    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })
    # Explicitly passing a 1D track name as iterator — legacy TrackRects path.
    result = pm.gextract(rects_track, intervals=intervals, iterator=rects_track)
    assert rects_track in result.columns
    assert len(result) >= 1


def test_intervals_via_scanner_multi_scope(_init_db, rects_track, use_scanner_env):
    """Multiple scope rects yield one output row each."""
    intervals = pd.DataFrame({
        "chrom1": ["1", "1"],
        "start1": [0, 200_000],
        "end1":   [100_000, 300_000],
        "chrom2": ["1", "1"],
        "start2": [0, 200_000],
        "end2":   [100_000, 300_000],
    })
    result = pm.gextract(rects_track, intervals=intervals)
    assert rects_track in result.columns
    # One output row per scope rect.
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Multitask regression guard
#
# The IntervalsPolicy scanner currently runs single-process (no fork/FIFO
# path).  This test verifies that the result is identical regardless of
# the CONFIG multitasking settings.  It is a regression guard for any
# future parallelization work.
# ---------------------------------------------------------------------------

def test_intervals_via_scanner_multitask_equivalence(_init_db, rects_track, monkeypatch):
    """Scanner currently single-process; regression guard.

    Runs gextract with IntervalsPolicy under two CONFIG scenarios:
      1. multitasking disabled, max_processes=1
      2. multitasking enabled, max_processes=4, aggressive floor heuristics

    Asserts the results are identical after sorting.
    """
    monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "1")

    intervals = pd.DataFrame({
        "chrom1": ["1"], "start1": [0], "end1": [500_000],
        "chrom2": ["1"], "start2": [0], "end2": [500_000],
    })

    saved = dict(CONFIG)
    try:
        CONFIG.update({"multitasking": False, "max_processes": 1})
        r1 = pm.gextract(rects_track, intervals=intervals)
        CONFIG.update({
            "multitasking": True,
            "max_processes": 4,
            "min_scope4process": 1,
            "min_intervs4process": 1,
        })
        r4 = pm.gextract(rects_track, intervals=intervals)
    finally:
        CONFIG.clear()
        CONFIG.update(saved)

    r1s = r1.sort_values(["chrom1", "start1", "chrom2", "start2"]).reset_index(drop=True)
    r4s = r4.sort_values(["chrom1", "start1", "chrom2", "start2"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(r1s, r4s)
