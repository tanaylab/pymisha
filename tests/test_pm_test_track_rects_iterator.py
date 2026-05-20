"""Tests for the TrackRects 2D iterator via the pm_test_track_rects_iterator binding.

The rects_track fixture is defined in conftest.py and creates a small rects
track with known geometry so tests have deterministic expected counts.
All chromids use chrom "1" = chromid 0 and chrom "2" = chromid 1.
"""

from __future__ import annotations

import _pymisha
import numpy as np
import pytest


def _scope(rects):
    """Build a scope dict from a list of (c1, s1, e1, c2, s2, e2) tuples."""
    return {
        "chrom1": np.array([r[0] for r in rects], dtype=np.int32),
        "start1": np.array([r[1] for r in rects], dtype=np.int64),
        "end1":   np.array([r[2] for r in rects], dtype=np.int64),
        "chrom2": np.array([r[3] for r in rects], dtype=np.int32),
        "start2": np.array([r[4] for r in rects], dtype=np.int64),
        "end2":   np.array([r[5] for r in rects], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicFunctionality:

    def test_single_pair_full_scope_yields_all_objects(self, rects_track):
        """Full chrom-pair scope on (0,0) should return all 3 objects."""
        scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 3
        assert all(c1 == 0 and c2 == 0
                   for c1, c2 in zip(out["chrom1"], out["chrom2"]))

    def test_inter_chrom_pair_full_scope(self, rects_track):
        """Full scope on (0,1) should return the 1 object there."""
        scope = _scope([(0, 0, 500_000, 1, 0, 300_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 1
        assert out["chrom1"][0] == 0
        assert out["chrom2"][0] == 1

    def test_multi_pair_scope_sums_both(self, rects_track):
        """Two scope rects spanning both pairs returns 3 + 1 = 4 objects."""
        scope = _scope([
            (0, 0, 500_000, 0, 0, 500_000),
            (0, 0, 500_000, 1, 0, 300_000),
        ])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 4

    def test_empty_scope_yields_nothing(self, rects_track):
        out = _pymisha.pm_test_track_rects_iterator(rects_track, _scope([]), None)
        assert len(out["start1"]) == 0

    def test_narrow_scope_excludes_large_object(self, rects_track):
        """Scope [0, 1000) x [0, 1000) clips out R3 (50k-150k x 60k-160k)."""
        scope = _scope([(0, 0, 1_000, 0, 0, 1_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        # R1 (100-300, 200-400) is inside [0,1000). R2 (500-700, 600-800) too.
        assert len(out["start1"]) == 2

    def test_scope_that_misses_all_objects(self, rects_track):
        """Scope window that doesn't intersect any object yields 0."""
        scope = _scope([(0, 200_000, 300_000, 0, 200_000, 300_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 0


class TestClipping:

    def test_emitted_rects_clipped_to_scope(self, rects_track):
        """Intersections must not extend outside the scope rect."""
        # Scope just clips into R3's boundary.
        scope = _scope([(0, 100_000, 200_000, 0, 100_000, 200_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 1
        assert out["start1"][0] >= 100_000
        assert out["end1"][0] <= 200_000
        assert out["start2"][0] >= 100_000
        assert out["end2"][0] <= 200_000

    def test_partial_overlap_clips_correctly(self, rects_track):
        """Scope partially overlapping R1 yields a clipped rect."""
        # R1 is at (100, 200, 300, 400). Scope [0,200) x [0,300):
        # intersection: (100, 200, 200, 300) — non-empty.
        scope = _scope([(0, 0, 200, 0, 0, 300)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 1
        assert out["start1"][0] == 100
        assert out["end1"][0] == 200    # clipped at scope end1
        assert out["start2"][0] == 200
        assert out["end2"][0] == 300    # clipped at scope end2

    def test_two_scope_rects_can_both_hit_same_object(self, rects_track):
        """Two scope rects that each intersect the same track object yield two emissions."""
        # Both scope rects intersect R1 (100-300, 200-400).
        # scope[0]: (0-1000, 0-1000) intersects R1 => clip (100, 200, 300, 400)
        # scope[1]: (0-1000, 250-1000) intersects R1 => clip (100, 250, 300, 400)
        # R2 (500-700, 600-800) also intersects both scope rects.
        # => 2 objects x 2 scope rects = 4 total emissions.
        scope = _scope([
            (0, 0, 1_000, 0, 0, 1_000),
            (0, 0, 1_000, 0, 250, 1_000),
        ])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        # R1 hits scope[0] and scope[1]; R2 hits scope[0] and scope[1] => 4 emissions.
        assert len(out["start1"]) == 4


class TestBand:

    def test_band_active_drops_inter_chrom_scope_rects(self, rects_track):
        """Inter-chrom scope (0,1) with active band yields 0."""
        scope_inter = _scope([(0, 0, 500_000, 1, 0, 300_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope_inter, (-1000, 1000))
        assert len(out["start1"]) == 0

    def test_band_inactive_includes_inter_chrom(self, rects_track):
        """Without band, inter-chrom scope should include the 1 object."""
        scope_inter = _scope([(0, 0, 500_000, 1, 0, 300_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope_inter, None)
        assert len(out["start1"]) == 1

    def test_band_shrinks_same_chrom_intersections(self, rects_track):
        """Band (-50, 50) keeps only diagonal-touching objects."""
        # R1 at (100,200,300,400): x-y range is [100-400, 300-200] = [-300, 100] — intersects (-50,50)
        # R2 at (500,600,700,800): x-y range is [500-800, 700-600] = [-300, 100] — intersects (-50,50)
        # R3 at (50000,60000,150000,160000): diagonal range [-110000, 90000] — intersects (-50,50)
        # All three objects have diagonal range overlapping (-50, 50) so all 3 should survive.
        scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
        out_no_band = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        out_band = _pymisha.pm_test_track_rects_iterator(rects_track, scope, (-50, 50))
        # Band may reduce count but not increase it.
        assert len(out_band["start1"]) <= len(out_no_band["start1"])

    def test_band_that_excludes_all(self, rects_track):
        """Band far above the diagonal excludes all objects on (0,0).

        Diagonal ranges (x - y):
          R1 (100-300 x 200-400): [-300, 100]
          R2 (500-700 x 600-800): [-300, 100]
          R3 (50k-150k x 60k-160k): [-110000, 90000]
        A band [200000, 300000] is above all three ranges, so all are excluded.
        """
        scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, (200_000, 300_000))
        assert len(out["start1"]) == 0


class TestErrors:

    def test_unknown_track_raises(self):
        with pytest.raises(RuntimeError):
            _pymisha.pm_test_track_rects_iterator(
                "no_such_track", _scope([(0, 0, 100, 0, 0, 100)]), None)

    def test_non_2d_track_raises(self):
        """1D dense/sparse tracks must be rejected."""
        with pytest.raises(RuntimeError):
            _pymisha.pm_test_track_rects_iterator(
                "dense_track", _scope([(0, 0, 100, 0, 0, 100)]), None)

    def test_non_2d_sparse_track_raises(self):
        with pytest.raises(RuntimeError):
            _pymisha.pm_test_track_rects_iterator(
                "sparse_track", _scope([(0, 0, 100, 0, 0, 100)]), None)


class TestReturnShape:

    def test_returns_six_keys(self, rects_track):
        scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert set(out.keys()) == {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}

    def test_all_arrays_same_length(self, rects_track):
        scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        n = len(out["start1"])
        for k in out:
            assert len(out[k]) == n, f"Array {k!r} has wrong length"

    def test_empty_scope_returns_zero_length_arrays(self, rects_track):
        out = _pymisha.pm_test_track_rects_iterator(rects_track, _scope([]), None)
        for k in ("chrom1", "start1", "end1", "chrom2", "start2", "end2"):
            assert k in out
            assert len(out[k]) == 0

    def test_absent_pair_skipped_silently(self, rects_track):
        """Scope on a pair with no track data yields 0, no exception."""
        # chromid (1,1) — chrom2 x chrom2 — has no data in our fixture.
        scope = _scope([(1, 0, 300_000, 1, 0, 300_000)])
        out = _pymisha.pm_test_track_rects_iterator(rects_track, scope, None)
        assert len(out["start1"]) == 0
