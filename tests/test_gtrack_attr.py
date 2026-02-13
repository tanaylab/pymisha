"""Tests for gtrack_attr_get and gtrack_attr_set functions."""

import pandas as pd
import pytest

import pymisha as pm


class TestGtrackAttrGet:
    """Test gtrack_attr_get function."""

    def test_get_existing_attribute(self):
        """Get an existing track attribute."""
        # dense_track should have created.by attribute
        result = pm.gtrack_attr_get("dense_track", "created.by")
        assert result is not None
        # Should be a string value
        assert isinstance(result, str)

    def test_get_nonexistent_attribute_returns_empty(self):
        """Getting a non-existent attribute returns empty string."""
        result = pm.gtrack_attr_get("dense_track", "nonexistent_attr_xyz")
        assert result == ""

    def test_get_requires_track(self):
        """gtrack_attr_get requires a track argument."""
        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_attr_get(None, "some_attr")

    def test_get_requires_attr(self):
        """gtrack_attr_get requires an attr argument."""
        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_attr_get("dense_track", None)

    def test_get_invalid_track_raises(self):
        """Getting attribute from non-existent track raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_attr_get("nonexistent_track", "some_attr")


class TestGtrackAttrSet:
    """Test gtrack_attr_set function."""

    def test_set_new_attribute(self):
        """Set a new track attribute."""
        # Set a new attribute
        pm.gtrack_attr_set("dense_track", "test_attr", "test_value")

        # Verify it was set
        result = pm.gtrack_attr_get("dense_track", "test_attr")
        assert result == "test_value"

        # Clean up - remove attribute by setting empty string
        pm.gtrack_attr_set("dense_track", "test_attr", "")

    def test_set_overwrites_existing(self):
        """Setting an attribute overwrites existing value."""
        # Set initial value
        pm.gtrack_attr_set("dense_track", "test_overwrite", "value1")

        # Overwrite
        pm.gtrack_attr_set("dense_track", "test_overwrite", "value2")

        result = pm.gtrack_attr_get("dense_track", "test_overwrite")
        assert result == "value2"

        # Clean up
        pm.gtrack_attr_set("dense_track", "test_overwrite", "")

    def test_set_empty_removes_attribute(self):
        """Setting attribute to empty string removes it."""
        # Create attribute
        pm.gtrack_attr_set("dense_track", "test_remove", "temp_value")

        # Verify it exists
        assert pm.gtrack_attr_get("dense_track", "test_remove") == "temp_value"

        # Remove by setting empty
        pm.gtrack_attr_set("dense_track", "test_remove", "")

        # Verify it's gone
        assert pm.gtrack_attr_get("dense_track", "test_remove") == ""

    def test_set_requires_track(self):
        """gtrack_attr_set requires a track argument."""
        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_attr_set(None, "attr", "value")

    def test_set_requires_attr(self):
        """gtrack_attr_set requires an attr argument."""
        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_attr_set("dense_track", None, "value")

    def test_set_invalid_track_raises(self):
        """Setting attribute on non-existent track raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_attr_set("nonexistent_track", "attr", "value")


class TestGtrackAttrExport:
    """Test gtrack_attr_export function."""

    def test_export_all_tracks(self):
        """Export attributes for all tracks."""
        result = pm.gtrack_attr_export()

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Should have rows for tracks
        assert len(result) > 0
        # Row names should be track names
        track_names = pm.gtrack_ls()
        for track in track_names:
            assert track in result.index

    def test_export_specific_tracks(self):
        """Export attributes for specific tracks."""
        result = pm.gtrack_attr_export(tracks=["sparse_track", "dense_track"])

        assert result is not None
        assert len(result) == 2
        assert "sparse_track" in result.index
        assert "dense_track" in result.index

    def test_export_specific_attrs(self):
        """Export specific attributes."""
        result = pm.gtrack_attr_export(attrs=["created.by"])

        assert result is not None
        assert "created.by" in result.columns

    def test_export_invalid_track_raises(self):
        """Export with invalid track raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_attr_export(tracks=["nonexistent_track"])
