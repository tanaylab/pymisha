"""Tests for gtrack_path and gintervals_path convenience functions."""

import os

import pytest

import pymisha as pm


@pytest.fixture(scope="module", autouse=True)
def init_db():
    """Initialize the example database."""
    pm.gdb_init_examples()


class TestGtrackPath:
    """Tests for gtrack_path."""

    def test_returns_string(self):
        """gtrack_path returns a string."""
        result = pm.gtrack_path("dense_track")
        assert isinstance(result, str)

    def test_path_exists_on_disk(self):
        """Returned path is a real directory on disk."""
        path = pm.gtrack_path("dense_track")
        assert os.path.isdir(path)

    def test_path_ends_with_track_suffix(self):
        """Path ends with .track directory name."""
        path = pm.gtrack_path("dense_track")
        assert path.endswith("dense_track.track")

    def test_sparse_track(self):
        """Works for sparse tracks too."""
        path = pm.gtrack_path("sparse_track")
        assert os.path.isdir(path)
        assert path.endswith("sparse_track.track")

    def test_nonexistent_track_raises(self):
        """Raises ValueError for a track that does not exist."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_path("nonexistent_track_xyz")

    def test_none_raises(self):
        """Raises ValueError when track is None."""
        with pytest.raises(ValueError):
            pm.gtrack_path(None)

    def test_consistent_with_pm_track_path(self):
        """Returns the same value as the underlying C++ function."""
        path = pm.gtrack_path("dense_track")
        cpp_path = pm._pymisha.pm_track_path("dense_track")
        assert path == cpp_path


class TestGintervalsPath:
    """Tests for gintervals_path."""

    def test_returns_string(self):
        """gintervals_path returns a string."""
        result = pm.gintervals_path("annotations")
        assert isinstance(result, str)

    def test_path_exists_on_disk(self):
        """Returned path exists on disk (file or directory)."""
        path = pm.gintervals_path("annotations")
        assert os.path.exists(path)

    def test_path_ends_with_interv_suffix(self):
        """Path ends with .interv suffix."""
        path = pm.gintervals_path("annotations")
        assert path.endswith("annotations.interv")

    def test_nonexistent_set_raises(self):
        """Raises ValueError for an interval set that does not exist."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_path("nonexistent_intervals_xyz")

    def test_none_raises(self):
        """Raises ValueError when name is None."""
        with pytest.raises(ValueError):
            pm.gintervals_path(None)

    def test_consistent_with_dataset_root(self):
        """Path is under the dataset root returned by gintervals_dataset."""
        path = pm.gintervals_path("annotations")
        root = pm.gintervals_dataset("annotations")
        assert path.startswith(root)
