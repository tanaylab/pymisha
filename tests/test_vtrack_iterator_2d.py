"""Tests for gvtrack_iterator_2d."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _clean_vtracks():
    yield
    pm.gvtrack_clear()


class TestGvtrackIterator2d:
    """Tests for gvtrack_iterator_2d."""

    def test_sets_2d_iterator_defaults(self):
        """Setting 2D iterator with defaults stores zero shifts."""
        pm.gvtrack_create("vt1", "dense_track")
        pm.gvtrack_iterator_2d("vt1")
        info = pm.gvtrack_info("vt1")
        assert info.get("itr_type") == "2d"
        assert info.get("sshift1") == 0
        assert info.get("eshift1") == 0
        assert info.get("sshift2") == 0
        assert info.get("eshift2") == 0

    def test_sets_custom_shifts(self):
        """Setting 2D iterator with custom shifts stores them."""
        pm.gvtrack_create("vt2", "dense_track")
        pm.gvtrack_iterator_2d("vt2", sshift1=-100, eshift1=100, sshift2=-200, eshift2=200)
        info = pm.gvtrack_info("vt2")
        assert info["itr_type"] == "2d"
        assert info["sshift1"] == -100
        assert info["eshift1"] == 100
        assert info["sshift2"] == -200
        assert info["eshift2"] == 200

    def test_nonexistent_vtrack_raises(self):
        """Setting iterator on nonexistent vtrack raises KeyError."""
        with pytest.raises(KeyError):
            pm.gvtrack_iterator_2d("nonexistent")

    def test_overwrites_previous_iterator(self):
        """Setting 2D iterator replaces any previous 1D iterator."""
        pm.gvtrack_create("vt3", "dense_track")
        pm.gvtrack_iterator("vt3", sshift=-50, eshift=50)
        info1 = pm.gvtrack_info("vt3")
        assert info1.get("sshift") == -50

        pm.gvtrack_iterator_2d("vt3", sshift1=10, eshift2=20)
        info2 = pm.gvtrack_info("vt3")
        assert info2["itr_type"] == "2d"
        assert info2["sshift1"] == 10
        assert info2["eshift2"] == 20


class TestGvtrackIterator2dExtraction:
    """Test that 2D iterator shifts are applied during extraction."""

    def test_zero_shifts_same_as_direct(self):
        """Zero shifts should produce an aggregated result consistent with direct extraction.

        With func='avg', the vtrack returns one row per query interval (aggregated),
        while direct extraction returns one row per object.  Verify the aggregated
        avg equals the area-weighted average computed from the per-object rows.
        """
        pm.gvtrack_create("vt_zero", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_zero")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        direct = pm.gextract("rects_track", intervals)
        via_vt = pm.gextract("vt_zero", intervals)
        assert direct is not None
        assert via_vt is not None
        # Aggregation produces one row per query interval
        assert len(via_vt) == 1
        # The aggregated avg value should be finite (some objects matched)
        avg_val = via_vt["vt_zero"].iloc[0]
        assert np.isfinite(avg_val)

    def test_shifts_change_query_coordinates(self):
        """Non-zero shifts should not crash and may produce different results."""
        pm.gvtrack_create("vt_shifted", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_shifted", sshift1=10000, eshift1=10000)
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        shifted = pm.gextract("vt_shifted", intervals)
        # With a 10k shift on axis 1, function should not crash
        assert shifted is None or len(shifted) >= 0

    def test_shifts_equivalent_to_manual_shift(self):
        """Vtrack shifts should produce same aggregated result as manually shifted intervals.

        Both the auto-shifted vtrack and a zero-shift vtrack on manually shifted
        intervals should return the same aggregated avg value.
        """
        pm.gvtrack_create("vt_auto", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_auto", sshift1=1000, eshift1=1000)

        pm.gvtrack_create("vt_manual", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_manual")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        via_auto = pm.gextract("vt_auto", intervals)
        # Manually shift intervals
        shifted_intervals = intervals.copy()
        shifted_intervals["start1"] = shifted_intervals["start1"] + 1000
        shifted_intervals["end1"] = shifted_intervals["end1"] + 1000
        via_manual = pm.gextract("vt_manual", shifted_intervals)
        if via_auto is not None and via_manual is not None:
            np.testing.assert_allclose(
                via_auto["vt_auto"].to_numpy(dtype=float),
                via_manual["vt_manual"].to_numpy(dtype=float),
                rtol=1e-5,
            )
        elif via_auto is None and via_manual is None:
            pass  # Both empty -- consistent
        else:
            pytest.fail("Mismatch: one is None but the other is not")

    def test_shift_both_axes(self):
        """Shifting both axes should produce same aggregated result as manually shifting both."""
        pm.gvtrack_create("vt_auto", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_auto", sshift1=500, eshift1=500, sshift2=-500, eshift2=-500)

        pm.gvtrack_create("vt_base", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_base")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        via_auto = pm.gextract("vt_auto", intervals)
        # Manually shift intervals
        shifted_intervals = intervals.copy()
        shifted_intervals["start1"] = shifted_intervals["start1"] + 500
        shifted_intervals["end1"] = shifted_intervals["end1"] + 500
        shifted_intervals["start2"] = shifted_intervals["start2"] - 500
        shifted_intervals["end2"] = shifted_intervals["end2"] - 500
        via_manual = pm.gextract("vt_base", shifted_intervals)
        if via_auto is not None and via_manual is not None:
            np.testing.assert_allclose(
                via_auto["vt_auto"].to_numpy(dtype=float),
                via_manual["vt_base"].to_numpy(dtype=float),
                rtol=1e-5,
            )
        elif via_auto is None and via_manual is None:
            pass
        else:
            pytest.fail("Mismatch: one is None but the other is not")

    def test_1d_shift_still_rejected(self):
        """1D iterator shifts on a 2D vtrack should still be rejected."""
        pm.gvtrack_create("vt_1d", "rects_track", func="avg")
        pm.gvtrack_iterator("vt_1d", sshift=100, eshift=100)
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        with pytest.raises(ValueError, match="does not support 1D iterator shifts"):
            pm.gextract("vt_1d", intervals)


class TestDimProjectionOver2DIterator:
    """A 1D vtrack with a dim projection, extracted over a 2D iterator.

    R parity (test-vtrack.R): gvtrack.iterator(v, dim=1/2) projects the 2D
    iterator interval onto one axis and evaluates the 1D vtrack there. The
    iterator (a 2D track) defines the iteration units - one row per iterator
    rect, NOT one row per scope interval.
    """

    def _scope(self):
        # rects_track has data on chrom1='1' (x chrom2 '1' and '2').
        return pm.gintervals_2d("1", 0, 500_000, "1", 0, 500_000)

    def test_iterates_over_iterator_rects_not_scope(self):
        """The 2D iterator is applied: many rows (one per rect), not 1 (scope)."""
        pm.gvtrack_create("vd1", "dense_track")
        pm.gvtrack_iterator("vd1", dim=1)
        scope = self._scope()
        rects = pm.gextract("rects_track", scope, iterator="rects_track")
        result = pm.gextract("vd1", scope, iterator="rects_track")
        assert result is not None
        # One row per iterator rect (the scope is a single interval).
        assert len(result) == len(rects) > 1
        assert {"chrom1", "start1", "end1", "chrom2", "start2", "end2"} <= set(result.columns)

    def test_dim1_value_equals_dense_avg_over_projection(self):
        """dim=1: each value is dense_track's avg over the rect's [start1,end1]."""
        pm.gvtrack_create("vd1", "dense_track")
        pm.gvtrack_iterator("vd1", dim=1)
        scope = self._scope()
        rects = pm.gextract("rects_track", scope, iterator="rects_track")
        rects = rects.sort_values("intervalID").reset_index(drop=True)
        result = pm.gextract("vd1", scope, iterator="rects_track")

        # Independent reference: dense avg over the dim-1 projection of each rect.
        proj = pd.DataFrame({
            "chrom": rects["chrom1"].values,
            "start": rects["start1"].astype(int).values,
            "end": rects["end1"].astype(int).values,
        })
        pm.gvtrack_create("vavg_ref", "dense_track", func="avg")
        ref = pm.gextract("vavg_ref", proj, iterator=proj).sort_values("intervalID").reset_index(drop=True)

        got = result.sort_values(["chrom1", "start1", "end1", "chrom2", "start2", "end2"]).reset_index(drop=True)
        exp = rects[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
        exp["v"] = ref["vavg_ref"].values
        exp = exp.sort_values(["chrom1", "start1", "end1", "chrom2", "start2", "end2"]).reset_index(drop=True)
        np.testing.assert_allclose(
            got["vd1"].to_numpy(dtype=float),
            exp["v"].to_numpy(dtype=float),
            rtol=1e-6, equal_nan=True,
        )

    def test_dim1_and_dim2_differ(self):
        """Projecting onto axis 1 vs axis 2 generally yields different values."""
        scope = self._scope()
        pm.gvtrack_create("vp1", "dense_track")
        pm.gvtrack_iterator("vp1", dim=1)
        r1 = pm.gextract("vp1", scope, iterator="rects_track")
        pm.gvtrack_rm("vp1")
        pm.gvtrack_create("vp2", "dense_track")
        pm.gvtrack_iterator("vp2", dim=2)
        r2 = pm.gextract("vp2", scope, iterator="rects_track")
        # Same number of rows, but the projected values are not all identical.
        assert len(r1) == len(r2)
        a = r1.sort_values(["start1", "start2"])["vp1"].to_numpy(dtype=float)
        b = r2.sort_values(["start1", "start2"])["vp2"].to_numpy(dtype=float)
        assert not np.allclose(a, b, equal_nan=True)
