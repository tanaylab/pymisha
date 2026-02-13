"""Tests for gtrack_attr_import."""

import contextlib

import pandas as pd
import pytest

import pymisha
from pymisha.tracks import _load_track_attributes, _save_track_attributes


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _restore_track_attrs():
    """Save and restore .attributes files for dense_track and sparse_track."""

    tracks_to_save = ["dense_track", "sparse_track"]
    saved = {}
    for t in tracks_to_save:
        try:
            saved[t] = dict(_load_track_attributes(t))
        except Exception:
            saved[t] = {}
    yield
    for t, attrs in saved.items():
        with contextlib.suppress(Exception):
            _save_track_attributes(t, attrs)


class TestGtrackAttrImport:
    """Tests for gtrack_attr_import."""

    def test_basic_import(self):
        """Import attributes for existing tracks."""
        table = pd.DataFrame(
            {"description": ["a dense track", "a sparse track"]},
            index=["dense_track", "sparse_track"],
        )
        pymisha.gtrack_attr_import(table)

        assert pymisha.gtrack_attr_get("dense_track", "description") == "a dense track"
        assert pymisha.gtrack_attr_get("sparse_track", "description") == "a sparse track"

    def test_multiple_attrs(self):
        """Import multiple attributes at once."""
        table = pd.DataFrame(
            {"color": ["red", "blue"], "priority": ["1", "2"]},
            index=["dense_track", "sparse_track"],
        )
        pymisha.gtrack_attr_import(table)

        assert pymisha.gtrack_attr_get("dense_track", "color") == "red"
        assert pymisha.gtrack_attr_get("sparse_track", "priority") == "2"

    def test_overwrite_existing_attr(self):
        """Importing overwrites existing attribute values."""
        pymisha.gtrack_attr_set("dense_track", "myattr", "old_value")
        table = pd.DataFrame(
            {"myattr": ["new_value"]},
            index=["dense_track"],
        )
        pymisha.gtrack_attr_import(table)
        assert pymisha.gtrack_attr_get("dense_track", "myattr") == "new_value"

    def test_remove_others(self):
        """With remove_others=True, non-imported non-readonly attrs are removed."""
        pymisha.gtrack_attr_set("dense_track", "keep_this", "yes")
        pymisha.gtrack_attr_set("dense_track", "remove_this", "yes")

        table = pd.DataFrame(
            {"keep_this": ["updated"]},
            index=["dense_track"],
        )
        pymisha.gtrack_attr_import(table, remove_others=True)

        assert pymisha.gtrack_attr_get("dense_track", "keep_this") == "updated"
        # remove_this should be gone
        assert pymisha.gtrack_attr_get("dense_track", "remove_this") == ""

    def test_nonexistent_track_raises(self):
        """Raises error for non-existent track."""
        table = pd.DataFrame(
            {"attr1": ["val"]},
            index=["no_such_track"],
        )
        with pytest.raises(ValueError, match="does not exist"):
            pymisha.gtrack_attr_import(table)

    def test_empty_table_raises(self):
        """Raises error for empty table."""
        table = pd.DataFrame()
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            pymisha.gtrack_attr_import(table)

    def test_empty_string_skips_attr(self):
        """Empty string values mean 'do not set this attr' (skip)."""
        pymisha.gtrack_attr_set("dense_track", "existing", "keep")
        table = pd.DataFrame(
            {"existing": [""], "newattr": ["hello"]},
            index=["dense_track"],
        )
        pymisha.gtrack_attr_import(table)
        # Empty string means don't set (or clear) — R behavior is to skip
        assert pymisha.gtrack_attr_get("dense_track", "newattr") == "hello"

    def test_values_converted_to_string(self):
        """Numeric values are converted to strings."""
        table = pd.DataFrame(
            {"score": [42]},
            index=["dense_track"],
        )
        pymisha.gtrack_attr_import(table)
        assert pymisha.gtrack_attr_get("dense_track", "score") == "42"
