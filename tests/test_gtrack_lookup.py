"""Tests for gtrack_lookup: track creation via lookup table."""

import contextlib

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize example database for each test."""
    pm.gdb_init_examples()
    yield
    # Clean up any test tracks
    for t in pm.gtrack_ls():
        if t.startswith("test_lookup_"):
            with contextlib.suppress(Exception):
                pm.gtrack_rm(t, force=True)


class TestGtrackLookupBasic:
    """Basic gtrack_lookup functionality."""

    def test_creates_track_1d(self):
        """gtrack_lookup creates a 1D track from a lookup table."""
        breaks = [0, 0.05, 0.1, 0.15, 0.2]
        table = np.array([10.0, 20.0, 30.0, 40.0])

        pm.gtrack_lookup(
            "test_lookup_1d", "", table,
            "dense_track", breaks,
            iterator=100,
        )

        assert pm.gtrack_exists("test_lookup_1d")
        info = pm.gtrack_info("test_lookup_1d")
        assert info is not None

    def test_track_values_match_glookup(self):
        """Values in created track should match glookup output."""
        breaks = [0, 0.05, 0.1, 0.15, 0.2]
        table = np.array([10.0, 20.0, 30.0, 40.0])
        intervals = pm.gintervals("1", 0, 5000)

        pm.gtrack_lookup(
            "test_lookup_match", "", table,
            "dense_track", breaks,
            iterator=100,
        )

        # Extract values from the created track
        track_vals = pm.gextract("test_lookup_match", intervals, iterator=100)

        # Get values from glookup
        lookup_vals = pm.glookup(table, "dense_track", breaks,
                                 intervals=intervals, iterator=100)

        if track_vals is not None and lookup_vals is not None:
            # Compare the value columns
            track_data = track_vals.iloc[:, 3:-1].to_numpy(dtype=float).ravel()
            lookup_data = lookup_vals["value"].to_numpy(dtype=float)
            # They should match (NaN-equal comparison)
            np.testing.assert_array_equal(
                np.where(np.isnan(track_data), -999, track_data),
                np.where(np.isnan(lookup_data), -999, lookup_data),
            )

    def test_creates_track_2d_lookup(self):
        """gtrack_lookup with 2 expression-breaks pairs creates a track."""
        breaks1 = [0, 0.05, 0.1, 0.15, 0.2]
        breaks2 = [0, 0.1, 0.2]
        table = np.arange(1, 9, dtype=float).reshape((4, 2))

        pm.gtrack_lookup(
            "test_lookup_2d", "", table,
            "dense_track", breaks1,
            "dense_track", breaks2,
            iterator=100,
        )

        assert pm.gtrack_exists("test_lookup_2d")

    def test_include_lowest(self):
        """gtrack_lookup respects include_lowest parameter."""
        breaks = [0, 0.1, 0.2, 0.5]
        table = np.array([100.0, 200.0, 300.0])
        intervals = pm.gintervals("1", 0, 5000)

        pm.gtrack_lookup(
            "test_lookup_il_false", "", table,
            "dense_track", breaks,
            iterator=100,
            include_lowest=False,
        )
        pm.gtrack_lookup(
            "test_lookup_il_true", "", table,
            "dense_track", breaks,
            iterator=100,
            include_lowest=True,
        )

        vals_false = pm.gextract("test_lookup_il_false", intervals, iterator=100)
        vals_true = pm.gextract("test_lookup_il_true", intervals, iterator=100)

        if vals_false is not None and vals_true is not None:
            # With include_lowest=True, fewer NaN values (values at breaks[0]
            # get mapped to first bin instead of NaN)
            n_nan_false = vals_false.iloc[:, 3:-1].isna().sum().sum()
            n_nan_true = vals_true.iloc[:, 3:-1].isna().sum().sum()
            assert n_nan_true <= n_nan_false

    def test_force_binning_false(self):
        """gtrack_lookup with force_binning=False produces NaN for out-of-range."""
        breaks = [0.05, 0.1, 0.15]
        table = np.array([100.0, 200.0])
        intervals = pm.gintervals("1", 0, 5000)

        pm.gtrack_lookup(
            "test_lookup_nofb", "", table,
            "dense_track", breaks,
            iterator=100,
            force_binning=False,
        )

        vals = pm.gextract("test_lookup_nofb", intervals, iterator=100)
        if vals is not None:
            data = vals.iloc[:, 3:-1].to_numpy(dtype=float).ravel()
            # Some values should be NaN (out of range)
            assert np.any(np.isnan(data))
            # Valid values should only be from the table
            valid = data[~np.isnan(data)]
            assert all(v in table for v in valid)

    def test_force_binning_true(self):
        """gtrack_lookup with force_binning=True clamps out-of-range."""
        breaks = [0.05, 0.1, 0.15]
        table = np.array([100.0, 200.0])
        intervals = pm.gintervals("1", 0, 5000)

        pm.gtrack_lookup(
            "test_lookup_fb", "", table,
            "dense_track", breaks,
            iterator=100,
            force_binning=True,
        )

        vals = pm.gextract("test_lookup_fb", intervals, iterator=100)
        if vals is not None:
            data = vals.iloc[:, 3:-1].to_numpy(dtype=float).ravel()
            # With force_binning, only NaN should come from NaN source values
            valid = data[~np.isnan(data)]
            assert all(v in table for v in valid)

    def test_creates_2d_track_from_2d_expression(self):
        """2D source expressions should produce a 2D output track."""
        breaks = np.linspace(0.0, 9000.0, 6)
        table = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        pm.gtrack_lookup(
            "test_lookup_2d_rects", "", table,
            "rects_track", breaks,
        )

        assert pm.gtrack_exists("test_lookup_2d_rects")
        info = pm.gtrack_info("test_lookup_2d_rects")
        assert info.get("dimensions") == 2

        extracted = pm.gextract("test_lookup_2d_rects", pm.gintervals_2d_all())
        assert extracted is not None
        assert len(extracted) > 0

    def test_band_creates_filtered_2d_track(self):
        """band parameter should be accepted and applied for 2D lookup tracks."""
        breaks = np.linspace(0.0, 9000.0, 6)
        table = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        pm.gtrack_lookup(
            "test_lookup_2d_full", "", table,
            "rects_track", breaks,
        )
        pm.gtrack_lookup(
            "test_lookup_2d_band", "", table,
            "rects_track", breaks,
            band=(-100, 100),
        )

        full_res = pm.gextract("test_lookup_2d_full", pm.gintervals_2d_all())
        band_res = pm.gextract("test_lookup_2d_band", pm.gintervals_2d_all())

        assert full_res is not None
        assert band_res is not None
        assert len(band_res) <= len(full_res)


class TestGtrackLookupAttributes:
    """Test that gtrack_lookup sets proper track attributes."""

    def test_sets_description(self):
        """Created track should have the given description."""
        breaks = [0, 0.1, 0.2]
        table = np.array([1.0, 2.0])

        pm.gtrack_lookup(
            "test_lookup_desc", "my test description", table,
            "dense_track", breaks,
            iterator=100,
        )

        desc = pm.gtrack_attr_get("test_lookup_desc", "description")
        assert desc == "my test description"

    def test_sets_created_by(self):
        """Created track should have created.by attribute."""
        breaks = [0, 0.1, 0.2]
        table = np.array([1.0, 2.0])

        pm.gtrack_lookup(
            "test_lookup_cb", "", table,
            "dense_track", breaks,
            iterator=100,
        )

        created_by = pm.gtrack_attr_get("test_lookup_cb", "created.by")
        assert created_by is not None
        assert "gtrack.lookup" in created_by


class TestGtrackLookupValidation:
    """Test argument validation."""

    def test_rejects_existing_track(self):
        """Should reject creating a track that already exists."""
        breaks = [0, 0.1, 0.2]
        table = np.array([1.0, 2.0])

        with pytest.raises(Exception):
            pm.gtrack_lookup(
                "dense_track", "", table,
                "dense_track", breaks,
                iterator=100,
            )

    def test_rejects_bad_table_dimensions(self):
        """Should reject lookup table with wrong dimensions."""
        breaks1 = [0, 0.1, 0.2]
        breaks2 = [0, 0.5, 1.0]
        table = np.array([1.0, 2.0])  # 1D table for 2D lookup

        with pytest.raises(ValueError):
            pm.gtrack_lookup(
                "test_lookup_bad_dim", "", table,
                "dense_track", breaks1,
                "dense_track", breaks2,
                iterator=100,
            )

    def test_rejects_bad_table_shape(self):
        """Should reject lookup table with wrong shape."""
        breaks = [0, 0.1, 0.2, 0.3]  # 3 bins
        table = np.array([1.0, 2.0])  # 2 elements, need 3

        with pytest.raises(ValueError):
            pm.gtrack_lookup(
                "test_lookup_bad_shape", "", table,
                "dense_track", breaks,
                iterator=100,
            )

    def test_rejects_no_args(self):
        """Should reject missing expression arguments."""
        table = np.array([1.0])

        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_lookup("test_lookup_noarg", "", table)

    def test_rejects_odd_args(self):
        """Should reject odd number of expression arguments."""
        table = np.array([1.0, 2.0])

        with pytest.raises(ValueError):
            pm.gtrack_lookup(
                "test_lookup_odd", "", table,
                "dense_track",  # missing breaks
            )


class TestGtrackLookupRParity:
    """R misha parity tests ported from test-gtrack.lookup.R."""

    def test_lookup_default_binning_1d(self):
        """Port of: lookup and extract with default binning (1D dense)."""
        breaks = np.linspace(0, 0.3, 6).tolist()
        table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        trackname = "test_lookup_r_default"
        try:
            pm.gtrack_lookup(trackname, "", table,
                             "dense_track", breaks, iterator=50)
            r = pm.gextract(trackname, pm.gintervals([1, 2]))
            assert r is not None
            assert len(r) == 16000
            vals = r[trackname].values
            non_nan = vals[~np.isnan(vals)]
            # All non-NaN values should come from the table
            assert set(non_nan).issubset(set(table))
        finally:
            with contextlib.suppress(Exception):
                pm.gtrack_rm(trackname, force=True)

    def test_lookup_no_force_binning(self):
        """Port of: lookup and extract without force binning."""
        breaks = np.linspace(0.1, 0.2, 6).tolist()
        table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        trackname = "test_lookup_r_nofb"
        try:
            pm.gtrack_lookup(trackname, "", table,
                             "dense_track", breaks,
                             force_binning=False, iterator=50)
            r = pm.gextract(trackname, pm.gintervals([1, 2]))
            assert r is not None
            vals = r[trackname].values
            non_nan = vals[~np.isnan(vals)]
            nan_count = np.isnan(vals).sum()
            # Without force_binning, values outside range should be NaN
            assert nan_count > 0
            assert set(non_nan).issubset(set(table))
        finally:
            with contextlib.suppress(Exception):
                pm.gtrack_rm(trackname, force=True)

    def test_lookup_2d_dense_dimensions(self):
        """Port of: lookup with 2D expressions creates correct track."""
        m1 = np.arange(1, 16, dtype=float).reshape((5, 3), order='F')
        breaks1 = np.linspace(0.1, 0.2, 6).tolist()
        breaks2 = np.linspace(0.1, 0.5, 4).tolist()
        trackname = "test_lookup_r_2d"
        try:
            pm.gtrack_lookup(trackname, "", m1,
                             "dense_track", breaks1,
                             "dense_track", breaks2,
                             iterator=50)
            assert pm.gtrack_exists(trackname)
            r = pm.gextract(trackname, pm.gintervals([1, 2]))
            assert r is not None
            assert len(r) > 0
            vals = r[trackname].values
            non_nan = vals[~np.isnan(vals)]
            # Values should come from the matrix entries
            assert len(non_nan) > 0
            assert all(1 <= v <= 15 for v in non_nan)
        finally:
            with contextlib.suppress(Exception):
                pm.gtrack_rm(trackname, force=True)
