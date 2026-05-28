"""R-parity regression tests for 2D band coord clipping.

R's ``StatQuadTree::intersect(..., band, ...)`` returns each object's coords
**clipped to the query rect and then shrunk to the band-intersected area**
(StatQuadTree.h:613-619 -> ``r = obj.intersect(rect); band.shrink2intersected(r);``).
Pymisha previously emitted the object's raw coords, diverging from R on:

* Raw 2D ``gextract`` with a band that cuts a contact rect (``query_objects_node_band``).
* Scanner-driven aggregations / per-rect emits over a 2D *intervals*
  iterator with a band (``PMTrackExpressionIntervals2DIterator``).

The tests below run entirely on the small in-repo test DB (no NFS); they pin
the clipped-rect behavior so future edits can't silently regress it.
"""

from __future__ import annotations

import os
import shutil

import _pymisha
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import write_2d_track_file

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup(name: str) -> None:
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture()
def diag_rects_track(_init_db):
    """Three rects on (chr1, chr1) that the near-diagonal band will clip.

    Each rect is 10kb x 10kb on the diagonal. With ``band=(-1024, 1024)`` only
    the strip ``|x - y| < 1024`` survives, so R reports each rect shrunk to::

        (start1, end1) -> (start1, end1)              # x unchanged here
        (start2, end2) -> (start1 + (-1024_at_x1=-y_at_x1...), end1 + 1024)

    Concretely, for a square rect ``(s, e) x (s, e)`` with ``band=(d1, d2)``
    where ``d1 < 0 < d2``, ``shrink2intersected`` yields::

        (s,       e + d1)   x   (s - d2,    e)
        clamped:  end2 = e + d2  -> e + 1024
                  start2 = s + d1 -> s - 1024 (negative d1 pushes start2 down)

    See ``DiagonalBand::shrink2intersected`` in src/QuadTreeReader.h:152.
    """
    tname = "test.band_coord_clip"
    _cleanup(tname)

    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    rects = [
        (10_000, 10_000, 20_000, 20_000, 1.0),
        (30_000, 30_000, 40_000, 40_000, 2.0),
        (50_000, 50_000, 60_000, 60_000, 3.0),
    ]
    write_2d_track_file(
        os.path.join(tdir, "1-1"), rects, (0, 0, 500_000, 500_000), is_points=False
    )
    _pymisha.pm_dbreload()

    yield tname

    _cleanup(tname)


# --------------------------------------------------------------------------- #
# Raw extract: query_objects_node{_band} must shrink coords to the band.
# --------------------------------------------------------------------------- #


class TestRawExtractBandClipsContactCoords:
    """gextract on a 2D track with a band must clip each contact rect to the
    band-intersected area (R parity: StatQuadTree::intersect(..., band, ...))."""

    def test_negative_band_clips_diagonal_rects(self, diag_rects_track):
        """Band ``(-2000, -100)`` (strictly below the diagonal) shrinks every
        on-diagonal square ``(s, e) x (s, e)`` whose corners have ``x-y == 0``.

        ``shrink2intersected`` (``DiagonalBand`` def):
          x1-y1=0 > d2=-100 -> y1 = x1 - d2 = x1 + 100
          x2-y2=0 > d2=-100 -> x2 = y2 + d2 = y2 - 100
        Output: ``(s, s+100, e-100, e)`` (axis1 shrinks at end1, axis2 at start2).
        """
        iv = pm.gintervals_2d("1", 0, 500_000, "1", 0, 500_000)
        r = pm.gextract(diag_rects_track, iv, band=(-2000, -100))

        assert r is not None and len(r) == 3
        r_sorted = r.sort_values("start1").reset_index(drop=True)
        for i, (s, e) in enumerate([(10_000, 20_000), (30_000, 40_000), (50_000, 60_000)]):
            assert int(r_sorted.loc[i, "start1"]) == s
            assert int(r_sorted.loc[i, "end1"]) == e - 100
            assert int(r_sorted.loc[i, "start2"]) == s + 100
            assert int(r_sorted.loc[i, "end2"]) == e

    def test_positive_band_clips_diagonal_rects(self, diag_rects_track):
        """Band ``(100, 2000)`` (strictly above the diagonal) shrinks the
        same on-diagonal squares the opposite way.

          x1-y1=0 < d1=100 -> x1 = y1 + d1 = y1 + 100
          x2-y2=0 < d1=100 -> y2 = x2 - d1 = x2 - 100
        Output: ``(s+100, s, e, e-100)`` (axis1 shrinks at start1, axis2 at end2).
        """
        iv = pm.gintervals_2d("1", 0, 500_000, "1", 0, 500_000)
        r = pm.gextract(diag_rects_track, iv, band=(100, 2000))

        assert r is not None and len(r) == 3
        r_sorted = r.sort_values("start1").reset_index(drop=True)
        for i, (s, e) in enumerate([(10_000, 20_000), (30_000, 40_000), (50_000, 60_000)]):
            assert int(r_sorted.loc[i, "start1"]) == s + 100
            assert int(r_sorted.loc[i, "end1"]) == e
            assert int(r_sorted.loc[i, "start2"]) == s
            assert int(r_sorted.loc[i, "end2"]) == e - 100

    def test_near_diagonal_band_does_not_clip_when_corners_inside_band(self, diag_rects_track):
        """Band ``(-1024, 1024)`` *contains* both corners of an on-diagonal
        square (x-y=0 is inside), so ``shrink2intersected`` is a no-op and the
        original coords come back. Pins this corner of the algorithm."""
        iv = pm.gintervals_2d("1", 0, 500_000, "1", 0, 500_000)
        r = pm.gextract(diag_rects_track, iv, band=(-1024, 1024))

        assert r is not None and len(r) == 3
        r_sorted = r.sort_values("start1").reset_index(drop=True)
        for i, (s, e) in enumerate([(10_000, 20_000), (30_000, 40_000), (50_000, 60_000)]):
            assert int(r_sorted.loc[i, "start1"]) == s
            assert int(r_sorted.loc[i, "end1"]) == e
            assert int(r_sorted.loc[i, "start2"]) == s
            assert int(r_sorted.loc[i, "end2"]) == e

    def test_no_band_returns_unclipped_object_coords(self, diag_rects_track):
        """Without a band, the object's stored rect is returned as-is."""
        iv = pm.gintervals_2d("1", 0, 500_000, "1", 0, 500_000)
        r = pm.gextract(diag_rects_track, iv)
        assert r is not None and len(r) == 3
        r_sorted = r.sort_values("start1").reset_index(drop=True)
        for i, (s, e) in enumerate([(10_000, 20_000), (30_000, 40_000), (50_000, 60_000)]):
            assert int(r_sorted.loc[i, "start1"]) == s
            assert int(r_sorted.loc[i, "end1"]) == e
            assert int(r_sorted.loc[i, "start2"]) == s
            assert int(r_sorted.loc[i, "end2"]) == e


# --------------------------------------------------------------------------- #
# Scanner / IntervalsPolicy: emitted scope rects must be band-shrunk.
# --------------------------------------------------------------------------- #


class TestScannerIntervalsBandShrinksScopeRect:
    """gextract with a value vtrack + explicit 2D iterator + band must emit
    each iterator rect shrunk to the band-intersected area (R parity)."""

    def test_scope_axis2_clamped_to_band(self, diag_rects_track):
        """For a scope rect (s1, e1) x (0, BIG) and band=(-d, +d), R clamps
        the emitted axis2 to ``[s1 - d, e1 + d]``.
        """
        pm.gvtrack_clear()
        pm.gvtrack_create("v", diag_rects_track, "weighted.sum")
        # Scope axis1 is small; axis2 spans the full arena. With band=(-1024, 1024)
        # R reports the emitted rect with axis2 clipped to start1-1024 .. end1+1024.
        scope = pd.DataFrame({
            "chrom1": ["1"], "start1": [30_000], "end1": [40_000],
            "chrom2": ["1"], "start2": [0],      "end2":   [500_000],
        })
        r = pm.gextract("v", scope, iterator=scope, band=(-1024, 1024))
        assert r is not None and len(r) == 1
        assert int(r["start1"].iloc[0]) == 30_000
        assert int(r["end1"].iloc[0]) == 40_000
        # axis2 must shrink to the band-intersected area.
        assert int(r["start2"].iloc[0]) == 30_000 - 1024
        assert int(r["end2"].iloc[0]) == 40_000 + 1024

    def test_no_band_axis2_unchanged(self, diag_rects_track):
        """Without a band, the emitted scope rect's axis2 stays as input."""
        pm.gvtrack_clear()
        pm.gvtrack_create("v", diag_rects_track, "weighted.sum")
        scope = pd.DataFrame({
            "chrom1": ["1"], "start1": [30_000], "end1": [40_000],
            "chrom2": ["1"], "start2": [0],      "end2":   [500_000],
        })
        r = pm.gextract("v", scope, iterator=scope)
        assert r is not None and len(r) == 1
        assert int(r["start2"].iloc[0]) == 0
        assert int(r["end2"].iloc[0]) == 500_000


# --------------------------------------------------------------------------- #
# weighted.sum area formula: pin R parity of DiagonalBand::intersected_area
# (previously off by ~3x on near-diagonal bins because the d1-triangle width
# was computed from y1 instead of y2 -- see QuadTreeReader.h::intersected_area).
# --------------------------------------------------------------------------- #


class TestWeightedSumBandIntersectedArea:
    """``weighted.sum`` over a band-shrunk rect must use R's
    ``intersected_area`` formula (= total - triangle(d1) - triangle(d2),
    triangles measured from y2/y1 of the *shrunk* rect)."""

    def test_one_diagonal_bin_inside_band(self, diag_rects_track):
        """A single 10kb diagonal rect (val=1.0), queried at its own extent
        with a symmetric near-diagonal band, contributes exactly the parallel-
        ogram-shaped band-intersected area to ``weighted.sum``.

        For ``rect = (s, s, s+W, s+W)`` and ``band = (-b, +b)`` with
        ``0 < b < W``, R's formula yields::

            area = W*W - 2 * (W - b) * (W - b - 1) / 2  -  (W - b)  =  W*W - (W-b)^2
                 = W^2 - (W-b)^2
        Actually using R's exact integer formula:
            n = W - b
            sub_top    = (n*n - n) / 2
            sub_bottom = (n*n + n) / 2
            area = W*W - sub_top - sub_bottom = W*W - n*n
        """
        pm.gvtrack_clear()
        pm.gvtrack_create("v", diag_rects_track, "weighted.sum")
        # Query each rect at its own coords with a symmetric band.
        # For (10000, 20000)^2 and band=(-1024, 1024): n = 10000 - 1024 = 8976
        # area = 10000^2 - 8976^2 = 100,000,000 - 80,568,576 = 19,431,424
        scope = pd.DataFrame({
            "chrom1": ["1"], "start1": [10_000], "end1": [20_000],
            "chrom2": ["1"], "start2": [10_000], "end2":   [20_000],
        })
        r = pm.gextract("v", scope, iterator=scope, band=(-1024, 1024))
        assert r is not None and len(r) == 1
        n = 10_000 - 1024
        expected_area = 10_000 * 10_000 - n * n
        assert float(r["v"].iloc[0]) == pytest.approx(float(expected_area))
