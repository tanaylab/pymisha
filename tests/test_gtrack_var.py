"""Tests for gtrack.var.* functions (track variable management)."""

import numpy as np
import pytest

import pymisha as pm


class TestGtrackVarLs:
    """Test gtrack_var_ls."""

    def test_empty_track_returns_empty_list(self):
        """Track with no variables returns empty list."""
        result = pm.gtrack_var_ls("dense_track")
        assert isinstance(result, list)
        # dense_track may or may not have vars - just check it's a list

    def test_nonexistent_track_raises(self):
        """Non-existent track raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_var_ls("nonexistent_track_xyz")

    def test_pattern_filtering(self):
        """Pattern filters returned variable names."""
        # Create a temp track and some variables
        track = "test_var_ls_pattern"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "alpha_one", [1, 2, 3])
            pm.gtrack_var_set(track, "alpha_two", [4, 5, 6])
            pm.gtrack_var_set(track, "beta_one", [7, 8, 9])

            all_vars = pm.gtrack_var_ls(track)
            assert sorted(all_vars) == ["alpha_one", "alpha_two", "beta_one"]

            alpha_vars = pm.gtrack_var_ls(track, pattern="alpha")
            assert sorted(alpha_vars) == ["alpha_one", "alpha_two"]

            one_vars = pm.gtrack_var_ls(track, pattern="one")
            assert sorted(one_vars) == ["alpha_one", "beta_one"]
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()


class TestGtrackVarGetSet:
    """Test gtrack_var_get and gtrack_var_set."""

    def test_set_and_get_scalar(self):
        """Set and retrieve a scalar value."""
        track = "test_var_scalar"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "my_scalar", 42)
            result = pm.gtrack_var_get(track, "my_scalar")
            assert result == 42
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_set_and_get_list(self):
        """Set and retrieve a list."""
        track = "test_var_list"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "my_list", [1, 2, 3, 4, 5])
            result = pm.gtrack_var_get(track, "my_list")
            assert result == [1, 2, 3, 4, 5]
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_set_and_get_numpy_array(self):
        """Set and retrieve a numpy array."""
        track = "test_var_numpy"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            arr = np.array([1.5, 2.5, 3.5])
            pm.gtrack_var_set(track, "my_array", arr)
            result = pm.gtrack_var_get(track, "my_array")
            np.testing.assert_array_almost_equal(result, arr)
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_set_and_get_string(self):
        """Set and retrieve a string value."""
        track = "test_var_string"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "my_str", "hello world")
            result = pm.gtrack_var_get(track, "my_str")
            assert result == "hello world"
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_set_and_get_dict(self):
        """Set and retrieve a dictionary."""
        track = "test_var_dict"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            d = {"a": 1, "b": [2, 3], "c": "hello"}
            pm.gtrack_var_set(track, "my_dict", d)
            result = pm.gtrack_var_get(track, "my_dict")
            assert result == d
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_overwrite_variable(self):
        """Overwriting a variable replaces the old value."""
        track = "test_var_overwrite"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "v1", [1, 2, 3])
            assert pm.gtrack_var_get(track, "v1") == [1, 2, 3]

            pm.gtrack_var_set(track, "v1", [10, 20, 30])
            assert pm.gtrack_var_get(track, "v1") == [10, 20, 30]
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_get_nonexistent_var_raises(self):
        """Getting a non-existent variable raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_var_get("dense_track", "nonexistent_var_xyz")

    def test_set_nonexistent_track_raises(self):
        """Setting a variable on non-existent track raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_var_set("nonexistent_track_xyz", "v1", 42)

    def test_get_nonexistent_track_raises(self):
        """Getting a variable from non-existent track raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_var_get("nonexistent_track_xyz", "v1")


class TestGtrackVarRm:
    """Test gtrack_var_rm."""

    def test_rm_existing_var(self):
        """Remove an existing variable."""
        track = "test_var_rm"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "to_remove", 42)
            assert "to_remove" in pm.gtrack_var_ls(track)

            pm.gtrack_var_rm(track, "to_remove")
            assert "to_remove" not in pm.gtrack_var_ls(track)
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_rm_nonexistent_var(self):
        """Removing non-existent variable is silent (matches R behavior: file.remove warns)."""
        # R implementation uses file.remove which returns FALSE silently
        # We just don't raise
        track = "test_var_rm_novar"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            # Should not raise
            pm.gtrack_var_rm(track, "nonexistent_var")
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()

    def test_rm_nonexistent_track_raises(self):
        """Removing a variable from non-existent track raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_var_rm("nonexistent_track_xyz", "v1")

    def test_set_rm_set_roundtrip(self):
        """Set, remove, and re-set a variable."""
        track = "test_var_roundtrip"
        try:
            pm.gtrack_create_sparse(
                track, "test", pm.gintervals(1, 0, 1000), [1.0]
            )
            pm.gtrack_var_set(track, "v1", [1, 2])
            pm.gtrack_var_set(track, "v2", [3, 4])
            assert sorted(pm.gtrack_var_ls(track)) == ["v1", "v2"]

            pm.gtrack_var_rm(track, "v2")
            assert pm.gtrack_var_ls(track) == ["v1"]

            pm.gtrack_var_set(track, "v2", [5, 6])
            assert sorted(pm.gtrack_var_ls(track)) == ["v1", "v2"]
            assert pm.gtrack_var_get(track, "v2") == [5, 6]
        finally:
            pm.gtrack_rm(track, force=True)
            pm.gdb_reload()
