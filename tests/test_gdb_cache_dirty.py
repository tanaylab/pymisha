"""Tests for gdb_mark_cache_dirty (GAP-034)."""

import pytest

import pymisha as pm


def test_gdb_mark_cache_dirty_basic():
    """gdb_mark_cache_dirty should succeed when a database is loaded."""
    pm.gdb_mark_cache_dirty()


def test_gdb_mark_cache_dirty_returns_none():
    """gdb_mark_cache_dirty returns None (called for side effects)."""
    result = pm.gdb_mark_cache_dirty()
    assert result is None


def test_gdb_mark_cache_dirty_tracks_still_visible():
    """After mark_cache_dirty, existing tracks remain visible."""
    tracks_before = pm.gtrack_ls()
    pm.gdb_mark_cache_dirty()
    tracks_after = pm.gtrack_ls()
    assert tracks_before == tracks_after


def test_gdb_mark_cache_dirty_after_track_create():
    """Create a track, mark cache dirty, verify it is listed."""
    # Use a temp DB copy so we don't pollute the shared test DB
    db = pm.gdb_init_examples()
    try:
        tracks_before = set(pm.gtrack_ls())

        # Create a dense track from an expression
        pm.gtrack_create("test_dirty_track", "test track", "dense_track")
        pm.gdb_mark_cache_dirty()

        tracks_after = set(pm.gtrack_ls())
        assert "test_dirty_track" in tracks_after
        assert tracks_after - tracks_before == {"test_dirty_track"}
    finally:
        # Restore the original shared test DB
        pm.gdb_init(str(pm.gdb_examples_path()))


def test_gdb_mark_cache_dirty_no_db_raises():
    """gdb_mark_cache_dirty raises when no database is loaded."""
    old_root = pm._shared._GROOT
    try:
        pm._shared._GROOT = None
        with pytest.raises(RuntimeError, match="Database not set"):
            pm.gdb_mark_cache_dirty()
    finally:
        pm._shared._GROOT = old_root


def test_gdb_mark_cache_dirty_is_exported():
    """gdb_mark_cache_dirty is accessible from the pymisha namespace."""
    assert hasattr(pm, "gdb_mark_cache_dirty")
    assert callable(pm.gdb_mark_cache_dirty)
    assert "gdb_mark_cache_dirty" in pm.__all__
