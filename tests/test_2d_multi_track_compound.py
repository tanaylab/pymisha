"""Tests for multi-track 2D compound expressions (Spec C).

R misha evaluates compound 2D expressions referencing multiple vtracks
(e.g. ``"v_a + v_b"``) or multiple bare 2D track names (``"track_a - track_b"``)
by computing each vtrack's per-rect value through the scanner, then
evaluating the compound expression in R. PyMisha previously fell back to
the legacy path or errored. This module asserts the scanner-routed path
returns numerically-correct results.

These tests use FixedRect, TrackRects, and CartesianGrid iterators - the
three scanner-routed iterators shipped in v0.1.75-v0.1.77. The Intervals
iterator path is exercised via iterator=None (legacy + opt-in scanner).
"""

import os
import shutil

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import write_2d_track_file

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _clean_vtracks():
    yield
    pm.gvtrack_clear()


@pytest.fixture()
def two_rects_tracks():
    """Create two 2D rects tracks on chr1-chr1 with known geometry."""
    ta = "test.compound_a"
    tb = "test.compound_b"
    for t in (ta, tb):
        _cleanup_track(t)
        tdir = _track_dir(t)
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, ".attributes"), "w") as f:
            f.write("type=rectangles\ndimensions=2\n")

    rects_a = [
        (1000, 2000, 3000, 4000, 5.0),
        (5000, 6000, 7000, 8000, 10.0),
    ]
    rects_b = [
        (1000, 2000, 3000, 4000, 1.0),
        (10000, 11000, 12000, 13000, 100.0),
    ]
    write_2d_track_file(
        os.path.join(_track_dir(ta), "1-1"),
        rects_a, (0, 0, 500000, 500000), is_points=False,
    )
    write_2d_track_file(
        os.path.join(_track_dir(tb), "1-1"),
        rects_b, (0, 0, 500000, 500000), is_points=False,
    )
    _pymisha.pm_dbreload()

    yield ta, tb

    for t in (ta, tb):
        _cleanup_track(t)


def _bare_2d_scope(chrom="1", x1=0, x2=500_000, y1=0, y2=500_000):
    return pd.DataFrame({
        "chrom1": [chrom], "start1": [x1], "end1": [x2],
        "chrom2": [chrom], "start2": [y1], "end2": [y2],
    })


# ---------------------------------------------------------------------------
# Multi-vtrack compound expressions
# ---------------------------------------------------------------------------


class TestCompoundVtrackExprFixedRect:
    """Two-vtrack compound expressions through the FixedRect scanner."""

    def test_sum_of_two_avg_vtracks(self, two_rects_tracks):
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")

        scope = _bare_2d_scope()
        compound = pm.gextract(
            ["v_a + v_b"], intervals=scope, iterator=(1000, 1000)
        )

        a = pm.gextract(["v_a"], intervals=scope, iterator=(1000, 1000))
        b = pm.gextract(["v_b"], intervals=scope, iterator=(1000, 1000))
        expected = a["v_a"].to_numpy() + b["v_b"].to_numpy()

        np.testing.assert_array_equal(compound.columns[6], "v_a + v_b")
        np.testing.assert_allclose(
            compound["v_a + v_b"].to_numpy(), expected, equal_nan=True
        )

    def test_difference_of_two_avg_vtracks(self, two_rects_tracks):
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")

        scope = _bare_2d_scope()
        compound = pm.gextract(
            ["v_a - v_b"], intervals=scope, iterator=(1000, 1000)
        )

        a = pm.gextract(["v_a"], intervals=scope, iterator=(1000, 1000))
        b = pm.gextract(["v_b"], intervals=scope, iterator=(1000, 1000))
        expected = a["v_a"].to_numpy() - b["v_b"].to_numpy()

        np.testing.assert_allclose(
            compound["v_a - v_b"].to_numpy(), expected, equal_nan=True
        )

    def test_compound_with_colname(self, two_rects_tracks):
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")

        scope = _bare_2d_scope()
        out = pm.gextract(
            ["v_a + v_b"], intervals=scope, iterator=(1000, 1000),
            colnames=["sum_ab"],
        )
        assert "sum_ab" in out.columns
        assert "v_a + v_b" not in out.columns

    def test_compound_with_two_min_max_vtracks(self, two_rects_tracks):
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="min")
        pm.gvtrack_create("v_b", tb, func="max")
        scope = _bare_2d_scope()
        compound = pm.gextract(
            ["v_a * v_b"], intervals=scope, iterator=(1000, 1000),
        )
        a = pm.gextract(["v_a"], intervals=scope, iterator=(1000, 1000))
        b = pm.gextract(["v_b"], intervals=scope, iterator=(1000, 1000))
        np.testing.assert_allclose(
            compound["v_a * v_b"].to_numpy(),
            a["v_a"].to_numpy() * b["v_b"].to_numpy(),
            equal_nan=True,
        )

    def test_compound_with_repeated_vtrack(self, two_rects_tracks):
        """Same vtrack referenced twice in a compound expression."""
        ta, _ = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        scope = _bare_2d_scope()
        compound = pm.gextract(
            ["v_a + v_a"], intervals=scope, iterator=(1000, 1000),
        )
        single = pm.gextract(["v_a"], intervals=scope, iterator=(1000, 1000))
        np.testing.assert_allclose(
            compound["v_a + v_a"].to_numpy(),
            2.0 * single["v_a"].to_numpy(),
            equal_nan=True,
        )

    def test_compound_bare_tracks(self, two_rects_tracks):
        """``"track_a - track_b"`` with no vtracks: bare 2D track refs."""
        ta, tb = two_rects_tracks
        scope = _bare_2d_scope()
        compound = pm.gextract(
            [f"{ta} - {tb}"], intervals=scope, iterator=(1000, 1000),
        )
        a = pm.gextract([ta], intervals=scope, iterator=(1000, 1000))
        b = pm.gextract([tb], intervals=scope, iterator=(1000, 1000))
        np.testing.assert_allclose(
            compound[f"{ta} - {tb}"].to_numpy(),
            a[ta].to_numpy() - b[tb].to_numpy(),
            equal_nan=True,
        )

    def test_compound_mixed_bare_and_vtrack(self, two_rects_tracks):
        """Compound expression mixing a bare track ref and a vtrack."""
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_b", tb, func="avg")
        scope = _bare_2d_scope()
        compound = pm.gextract(
            [f"{ta} + v_b"], intervals=scope, iterator=(1000, 1000),
        )
        a = pm.gextract([ta], intervals=scope, iterator=(1000, 1000))
        b = pm.gextract(["v_b"], intervals=scope, iterator=(1000, 1000))
        np.testing.assert_allclose(
            compound[f"{ta} + v_b"].to_numpy(),
            a[ta].to_numpy() + b["v_b"].to_numpy(),
            equal_nan=True,
        )

    def test_compound_multiple_exprs_same_iterator(self, two_rects_tracks):
        """Two user expressions, deduped vars under the hood."""
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")
        scope = _bare_2d_scope()
        out = pm.gextract(
            ["v_a + v_b", "v_a - v_b"],
            intervals=scope, iterator=(1000, 1000),
        )
        a = pm.gextract(["v_a"], intervals=scope, iterator=(1000, 1000))
        b = pm.gextract(["v_b"], intervals=scope, iterator=(1000, 1000))
        np.testing.assert_allclose(
            out["v_a + v_b"].to_numpy(),
            a["v_a"].to_numpy() + b["v_b"].to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            out["v_a - v_b"].to_numpy(),
            a["v_a"].to_numpy() - b["v_b"].to_numpy(),
            equal_nan=True,
        )


class TestCompoundVtrackExprTrackRects:
    """Two-vtrack compound expressions through the TrackRects scanner."""

    def test_sum_of_two_avg_vtracks(self, two_rects_tracks):
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")
        scope = _bare_2d_scope()
        compound = pm.gextract(
            ["v_a + v_b"], intervals=scope, iterator=ta,
        )
        a = pm.gextract(["v_a"], intervals=scope, iterator=ta)
        b = pm.gextract(["v_b"], intervals=scope, iterator=ta)
        np.testing.assert_allclose(
            compound["v_a + v_b"].to_numpy(),
            a["v_a"].to_numpy() + b["v_b"].to_numpy(),
            equal_nan=True,
        )


class TestCompoundVtrackExprIntervalsScanner:
    """Multi-vtrack compound via the opt-in Intervals-iterator scanner path."""

    def test_compound_intervals_scanner_optin(self, two_rects_tracks, monkeypatch):
        ta, tb = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")
        scope = _bare_2d_scope()

        monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "1")
        compound = pm.gextract(["v_a + v_b"], intervals=scope)

        monkeypatch.setenv("PYMISHA_USE_SCANNER_FOR_INTERVALS", "0")
        a = pm.gextract(["v_a"], intervals=scope)
        b = pm.gextract(["v_b"], intervals=scope)
        np.testing.assert_allclose(
            compound["v_a + v_b"].to_numpy(),
            a["v_a"].to_numpy() + b["v_b"].to_numpy(),
            equal_nan=True,
        )


class TestCompoundResolverRejects:
    """Sanity: compound resolver returns None for unsupported cases."""

    def test_returns_none_for_pure_constants(self):
        from pymisha.extract import _resolve_2d_compound_for_scanner
        assert _resolve_2d_compound_for_scanner(["1 + 2"]) is None

    def test_returns_none_for_1d_vtrack_in_compound(self, two_rects_tracks):
        from pymisha.extract import _resolve_2d_compound_for_scanner
        ta, tb = two_rects_tracks
        # v_1d wraps a 1D track (dense_track is 1D in the test DB).
        pm.gvtrack_create("v_1d", "dense_track", func="avg")
        pm.gvtrack_create("v_b", tb, func="avg")
        assert _resolve_2d_compound_for_scanner(["v_1d + v_b"]) is None

    def test_returns_none_for_unknown_symbol(self, two_rects_tracks):
        from pymisha.extract import _resolve_2d_compound_for_scanner
        ta, _ = two_rects_tracks
        pm.gvtrack_create("v_a", ta, func="avg")
        assert _resolve_2d_compound_for_scanner(["v_a + does_not_exist"]) is None
