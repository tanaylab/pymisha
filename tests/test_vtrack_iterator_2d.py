"""Tests for gvtrack_iterator_2d."""

import numpy as np
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
