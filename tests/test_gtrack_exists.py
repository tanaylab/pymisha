"""Tests for gtrack_exists function."""

import pytest

import pymisha as pm


@pytest.fixture(scope="module", autouse=True)
def init_db():
    """Initialize the example database."""
    pm.gdb_init_examples()


class TestGtrackExists:
    """Tests for gtrack_exists."""

    def test_existing_track_returns_true(self):
        """gtrack_exists returns True for an existing track."""
        # dense_track exists in the example db
        assert pm.gtrack_exists("dense_track") is True

    def test_nonexistent_track_returns_false(self):
        """gtrack_exists returns False for a non-existent track."""
        assert pm.gtrack_exists("nonexistent_track_12345") is False

    def test_sparse_track_exists(self):
        """gtrack_exists works for sparse tracks."""
        assert pm.gtrack_exists("sparse_track") is True

    def test_empty_string_returns_false(self):
        """gtrack_exists returns False for empty string."""
        assert pm.gtrack_exists("") is False

    def test_requires_track_name(self):
        """gtrack_exists raises error when track name is None."""
        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_exists(None)

    def test_returns_bool(self):
        """gtrack_exists returns a boolean value."""
        result = pm.gtrack_exists("dense_track")
        assert isinstance(result, bool)

    def test_case_sensitive(self):
        """gtrack_exists is case-sensitive."""
        # dense_track exists, Dense_Track should not
        assert pm.gtrack_exists("dense_track") is True
        assert pm.gtrack_exists("Dense_Track") is False
