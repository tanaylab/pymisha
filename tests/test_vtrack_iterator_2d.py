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

        Explicit ``iterator=intervals`` keeps the one-row-per-scope-interval
        semantics this test originally relied on (since v0.8.3 a 2D-source
        vtrack with no iterator defaults to the source's rects, R parity).
        """
        pm.gvtrack_create("vt_zero", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_zero")
        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        direct = pm.gextract("rects_track", intervals)
        via_vt = pm.gextract("vt_zero", intervals, iterator=intervals)
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

        The shift crosses the chromosome boundary on purpose: +1000 pushes end1
        to 501000 on a 500000 bp chromosome, and the point is that this does not
        raise.

        What actually runs the shift here is the C++ scanner
        (``PMTrackExpression2DVars::compute``, src/PMTrackExpression2DVars.cpp),
        not ``_apply_2d_shifts`` - a reducing 2D vtrack with a DataFrame
        iterator routes through it. The scanner adds the shifts raw, with no
        clamp to [0, chrom_size], so what this pins is that the two sides agree
        *anyway*: the only region the unclamped query adds is past the end of
        the chromosome, where the track has nothing. The manual side has to
        clamp with ``gintervals_force_range`` for a different reason - a
        caller-supplied *scope* is validated, so 501000 would be rejected there.
        """
        pm.gvtrack_create("vt_auto", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_auto", sshift1=1000, eshift1=1000)

        pm.gvtrack_create("vt_manual", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_manual")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        # Explicit iterator=intervals: one-row-per-scope-interval aggregation
        # (the original test intent; the default-iterator path now expands to
        # source rects, R parity).
        via_auto = pm.gextract("vt_auto", intervals, iterator=intervals)
        shifted_intervals = intervals.copy()
        shifted_intervals["start1"] = shifted_intervals["start1"] + 1000
        shifted_intervals["end1"] = shifted_intervals["end1"] + 1000
        shifted_intervals = pm.gintervals_force_range(shifted_intervals)
        via_manual = pm.gextract("vt_manual", shifted_intervals, iterator=shifted_intervals)
        assert via_auto is not None and via_manual is not None
        auto_vals = via_auto["vt_auto"].to_numpy(dtype=float)
        assert np.isfinite(auto_vals).all(), "boundary-crossing shift must still produce values"
        np.testing.assert_allclose(
            auto_vals,
            via_manual["vt_manual"].to_numpy(dtype=float),
            rtol=1e-5,
        )

    def test_shift_both_axes(self):
        """Shifting both axes should produce same aggregated result as manually shifting both.

        Both directions run off the chromosome: +500 on axis1 pushes end1 past
        the 500000 bp end, -500 on axis2 pushes start2 to -500. As in
        ``test_shifts_equivalent_to_manual_shift``, the shift is applied by the
        C++ scanner and is *not* clamped there; the manual side is clamped by
        ``gintervals_force_range`` because a caller-supplied scope is validated.
        The two agree because neither out-of-range strip holds any data.
        """
        pm.gvtrack_create("vt_auto", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_auto", sshift1=500, eshift1=500, sshift2=-500, eshift2=-500)

        pm.gvtrack_create("vt_base", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_base")

        intervals = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        # Explicit iterator=intervals: keep one-row-per-scope semantics (see
        # test_shifts_equivalent_to_manual_shift for the rationale).
        via_auto = pm.gextract("vt_auto", intervals, iterator=intervals)
        shifted_intervals = intervals.copy()
        shifted_intervals["start1"] = shifted_intervals["start1"] + 500
        shifted_intervals["end1"] = shifted_intervals["end1"] + 500
        shifted_intervals["start2"] = shifted_intervals["start2"] - 500
        shifted_intervals["end2"] = shifted_intervals["end2"] - 500
        shifted_intervals = pm.gintervals_force_range(shifted_intervals)
        via_manual = pm.gextract("vt_base", shifted_intervals, iterator=shifted_intervals)
        assert via_auto is not None and via_manual is not None
        auto_vals = via_auto["vt_auto"].to_numpy(dtype=float)
        assert np.isfinite(auto_vals).all(), "boundary-crossing shift must still produce values"
        np.testing.assert_allclose(
            auto_vals,
            via_manual["vt_base"].to_numpy(dtype=float),
            rtol=1e-5,
        )

    def test_boundary_crossing_shift_on_raw_track_path(self):
        """A shift that runs off the chromosome must clamp, not raise.

        Regression: the raw/alias 2D path handed the *shifted* frame to
        _gextract_2d_single, which validated it - so an out-of-bounds shift
        raised "end coordinate exceeds chromosome boundaries", while the same
        query with an explicit iterator (the aggregation branch, which never
        validated) returned rows. R clamps in both cases.
        """
        pm.gvtrack_create("vt_shift_raw", "rects_track")
        pm.gvtrack_iterator_2d("vt_shift_raw", sshift1=1000, eshift1=1000)
        scope = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)

        via_default = pm.gextract("rects_track + vt_shift_raw", scope)
        via_iterator = pm.gextract("rects_track + vt_shift_raw", scope, iterator="rects_track")
        assert via_default is not None and via_iterator is not None
        assert len(via_default) == len(via_iterator)

    def test_shift_off_the_chromosome_yields_nan_not_error(self):
        """A shift that moves the whole rectangle off the chromosome gives NaN.

        Also routed through the C++ scanner, which does not clamp: the query
        lands entirely past the end of chromosome 1, finds nothing, and the
        value is NaN. What this pins is that the outcome is NaN rather than an
        error - the same outcome R reaches by a different route, where the
        clamp collapses the rectangle and ``out_of_range`` makes the variable
        NaN (misha/src/TrackVarProcessor.cpp).
        """
        pm.gvtrack_create("vt_off", "rects_track", func="avg")
        pm.gvtrack_iterator_2d("vt_off", sshift1=600000, eshift1=600000)
        scope = pm.gintervals_2d("1", 0, 500000, "1", 0, 500000)
        result = pm.gextract("vt_off", scope, iterator=scope)
        assert result is not None
        assert len(result) == 1
        assert np.isnan(result["vt_off"].to_numpy(dtype=float)).all()

    def test_collapsed_rectangle_keeps_values_on_the_right_scope_row(self, rects_track):
        """The raw/alias path restamps intervalID after a collapse.

        ``_apply_2d_shifts`` drops a rectangle the clamp collapsed, so the frame
        handed to the extractor is shorter than the scope and the intervalID it
        stamps is a position into *that* frame. ``_extract_shifted`` maps it back
        through ``kept``. The scope below is built so the collapsed rectangle
        comes first: without the remap the surviving row's value would be stamped
        intervalID 0 and land on the rectangle that produced nothing.

        (``test_boundary_crossing_shift_on_raw_track_path`` uses a shift that
        clamps without collapsing, so ``kept`` is None there and this branch
        never runs.)
        """
        pm.gvtrack_create("vt_collapse", rects_track)
        # +150 on start1 only: it inverts the 100..200 rectangle (250 >= 200)
        # and leaves the wide one alone (150 < 200000).
        pm.gvtrack_iterator_2d("vt_collapse", sshift1=150)

        # A bare track in the expression is what selects the raw/alias path;
        # a lone reducing vtrack would be aggregated elsewhere.
        scope = pd.DataFrame({
            "chrom1": ["1", "1"], "start1": [100, 0], "end1": [200, 200_000],
            "chrom2": ["1", "1"], "start2": [0, 0], "end2": [1000, 200_000],
        })
        result = pm.gextract(f"{rects_track} + vt_collapse", scope)
        assert result is not None

        # One row per object of the surviving rectangle, all pointing at scope
        # row 1. Without the remap they would carry intervalID 0 - the
        # rectangle the clamp collapsed, which returned nothing.
        assert len(result) == 3
        assert set(result["intervalID"].astype(int)) == {1}
        assert sorted(result["start1"].astype(int)) == [100, 500, 50_000]

    def test_apply_2d_shifts_clamps_like_r(self):
        """Unit-level pin on the clamp itself and on the row realignment.

        ``max(start + sshift, 0)`` / ``min(end + eshift, chrom_size)``, and a
        rectangle the clamp collapses is dropped from the query but keeps its
        own position in the output (NaN), never shifting later values up.
        """
        from pymisha.extract import _apply_2d_shifts, _scatter_shifted_values

        iv = pd.DataFrame({
            "chrom1": ["1", "1"], "start1": [0, 499000], "end1": [500000, 500000],
            "chrom2": ["1", "1"], "start2": [0, 0], "end2": [500000, 500000],
        })

        # Upper clamp (501000 -> 500000) plus a collapse: row 1's start moves
        # past the 500000 bp chromosome end.
        shifted, kept = _apply_2d_shifts(iv, 2000, 1000, 0, 0)
        assert list(kept) == [0]
        assert shifted["start1"].tolist() == [2000]
        assert shifted["end1"].tolist() == [500000]
        scattered = _scatter_shifted_values([7.0], kept, len(iv))
        assert scattered[0] == 7.0
        assert np.isnan(scattered[1])

        # Lower clamp: a negative start is pulled up to 0 and the row survives.
        shifted2, kept2 = _apply_2d_shifts(iv, -1000, 0, -1000, 0)
        assert kept2 is None
        assert shifted2["start1"].tolist() == [0, 498000]
        assert shifted2["start2"].tolist() == [0, 0]

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
