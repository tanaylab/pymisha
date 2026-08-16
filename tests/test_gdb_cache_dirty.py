"""Tests for gdb_mark_cache_dirty (GAP-034) and the R `.db.cache.dirty` sentinel.

R misha decides whether its cached track/interval-set inventory
(``<groot>/.db.cache``) is stale purely by checking whether
``<groot>/.db.cache.dirty`` exists (misha's R/db-cache.R,
``.gdb.cache_is_dirty`` / ``.gdb.cache_mark_dirty``). pymisha used to mutate
the database without ever writing that sentinel, so a track or interval set
created in Python stayed invisible to a fresh R ``gsetroot()`` on the same
database until someone ran ``gdb.reload(rescan = TRUE)``.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _same_path(a, b):
    """Compare two database paths through symlinks.

    On macOS the per-user temp dir is reached as ``/var/folders/...`` but
    resolves to ``/private/var/folders/...``, and pymisha stores the two
    differently: ``gdb_init`` keeps the path it was handed, while
    ``gdataset_load`` records the resolved one. Comparing the strings
    directly passes on Linux and fails on macOS.
    """
    return os.path.realpath(a) == os.path.realpath(b)


def _path_in(path, paths):
    return any(_same_path(path, p) for p in (paths or []))


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


def _dirty_sentinel_path(groot: str) -> str:
    return os.path.join(groot, ".db.cache.dirty")


def _reset_dirty(dirty_path: str) -> None:
    """Remove the sentinel if present, so a test starts from a known state.

    ``gdb_init_examples`` copies the currently-active example DB tree, so a
    dirty flag left behind by an *unrelated* earlier test (e.g. one that
    calls ``gdb_mark_cache_dirty`` against the session DB) can be copied
    into a fresh temp DB too. Resetting explicitly, rather than asserting
    the sentinel is absent, keeps these tests independent of run order.
    """
    if os.path.exists(dirty_path):
        os.remove(dirty_path)


class TestDbCacheDirtySentinel:
    """R's `.db.cache.dirty` sentinel must be written by pymisha mutations.

    This is what a fresh, non-rescanning R `gsetroot()` on the same database
    actually consults (see misha's R/db-cache.R, `.gdb.cache_is_dirty`). Each
    test here uses a private temp copy of the example DB (`gdb_init_examples`)
    so it never touches the shared `tests/testdb` tree.
    """

    def test_gtrack_create_marks_db_cache_dirty(self):
        """Creating a track through pymisha writes the R sentinel."""
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            pm.gtrack_create("test_dirty_new_track", "test track", "dense_track")

            assert os.path.exists(dirty_path), (
                "gtrack_create did not write .db.cache.dirty; a sibling R "
                "session would not see the new track without "
                "gdb.reload(rescan = TRUE)"
            )
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gtrack_rm_marks_db_cache_dirty(self):
        """Removing a track through pymisha writes the R sentinel."""
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            pm.gtrack_create("test_dirty_rm_track", "test track", "dense_track")
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            pm.gtrack_rm("test_dirty_rm_track", force=True)

            assert os.path.exists(dirty_path)
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gintervals_save_marks_db_cache_dirty(self):
        """Saving an interval set through pymisha writes the R sentinel.

        This is the exact reproducer from the bug report:
        pm.gintervals_save(..., "py_set2") creates tracks/py_set2.interv on
        disk but must also leave the .db.cache.dirty sentinel behind.
        """
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])
            pm.gintervals_save(intervals, "py_set2")

            assert os.path.exists(dirty_path), (
                "gintervals_save did not write .db.cache.dirty; a sibling R "
                "session would not see the new interval set without "
                "gdb.reload(rescan = TRUE)"
            )
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gintervals_rm_marks_db_cache_dirty(self):
        """Removing an interval set through pymisha writes the R sentinel."""
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])
            pm.gintervals_save(intervals, "test_dirty_rm_intervs")
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            pm.gintervals_rm("test_dirty_rm_intervs")

            assert os.path.exists(dirty_path)
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gdb_mark_cache_dirty_writes_sentinel(self):
        """gdb_mark_cache_dirty() itself must also write the R sentinel.

        Its docstring already promises R's contract (writing
        `.db.cache.dirty`); this asserts the implementation matches it.
        """
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            pm.gdb_mark_cache_dirty()

            assert os.path.exists(dirty_path)
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))


class TestDbCacheDirtyGapFixes:
    """Coverage for the four mutation paths that bypass both of pymisha's
    existing choke points (`_pm_dbreload()` and `gintervals_save`/
    `gintervals_rm`'s register/unregister calls) and therefore had to be
    patched individually. These are exactly the sites a purely
    choke-point-based fix would have missed.
    """

    def test_gtrack_array_create_marks_db_cache_dirty(self):
        """gtrack_array_create bypasses _pm_dbreload() (calls the raw C++
        rescan directly) and must be touched explicitly."""
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            ivs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
            pm.gtrack_array_create(
                "test_dirty_array_create", "test array", ivs,
                np.array([[1.0, 2.0]]), ["a", "b"],
            )

            assert os.path.exists(dirty_path)
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gtrack_array_import_marks_db_cache_dirty(self):
        """gtrack_array_import bypasses _pm_dbreload() (calls the raw C++
        rescan directly) and must be touched explicitly."""
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            # "array_track" is a pre-existing array track in the example DB;
            # an existing array-track name is a valid gtrack_array_import source.
            pm.gtrack_array_import(
                "test_dirty_array_import", "test array import", "array_track",
            )

            assert os.path.exists(dirty_path)
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gdir_rm_recursive_marks_db_cache_dirty(self):
        """gdir_rm(recursive=True) can sweep away tracks/interval sets nested
        under the removed directory without going through gtrack_rm /
        gintervals_rm at all, and must mark the sentinel itself."""
        pm.gdb_init_examples()
        try:
            groot = pm._shared._GROOT
            pm.gtrack_create("dirty_subdir.nested_track", "nested", "dense_track")
            dirty_path = _dirty_sentinel_path(groot)
            _reset_dirty(dirty_path)

            pm.gdir_rm("dirty_subdir", recursive=True, force=True)

            assert os.path.exists(dirty_path)
            assert "dirty_subdir.nested_track" not in pm.gtrack_ls()
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gtrack_copy_unloaded_destination_marks_dest_db_dirty(self):
        """gtrack_copy(db=<destination>) must mark the *destination*
        database's sentinel even when that database is not loaded anywhere
        in the current pymisha session (so _pm_dbreload()'s "current
        session" rescan/cache-clear is deliberately skipped for it)."""
        src_db = pm.gdb_init_examples()
        dest_db = tempfile.mkdtemp(prefix="pymisha-copy-dest-")
        try:
            shutil.rmtree(dest_db)
            shutil.copytree(src_db, dest_db)
            # Clear pre-existing tracks so the copy below is a plain create,
            # not an overwrite (irrelevant to the dirty-flag behavior, but
            # keeps the test's intent obvious).
            dest_tracks = os.path.join(dest_db, "tracks")
            for name in os.listdir(dest_tracks):
                if name.endswith(".track"):
                    shutil.rmtree(os.path.join(dest_tracks, name))

            dirty_path = _dirty_sentinel_path(dest_db)
            _reset_dirty(dirty_path)
            assert not _same_path(dest_db, pm._shared._GROOT)
            assert not _path_in(dest_db, pm._shared._GDATASETS)

            pm.gtrack_copy("dense_track", dest=None, db=dest_db)

            assert os.path.exists(dirty_path)
        finally:
            shutil.rmtree(dest_db, ignore_errors=True)
            pm.gdb_init(str(pm.gdb_examples_path()))


class TestDbCacheDirtyCrossDatabase:
    """The wrong-target regression this task exists to close: a mutation
    against database B while `_GROOT` is database A must mark B's sentinel,
    not A's. Before this fix there was no dirty-marking at all (so no
    wrong-target failure mode existed to inherit); the fix itself had to be
    careful not to introduce one by defaulting every call site to the
    session `_GROOT`.
    """

    def test_gtrack_rm_with_explicit_db_marks_that_db_not_groot(self):
        """gtrack_rm(track, db=<other>) removes from <other> but, before
        this fix, marked the *unrelated, currently active* _GROOT dirty
        instead of <other> - and <other>/.db.cache.dirty was never written
        at all."""
        db_a = pm.gdb_init_examples()
        db_b = pm.gdb_init_examples()  # separate temp copy, never loaded
        pm.gdb_init(db_a)  # active _GROOT is A; B is just a directory on disk
        try:
            assert _same_path(pm._shared._GROOT, db_a)
            dirty_a = _dirty_sentinel_path(db_a)
            dirty_b = _dirty_sentinel_path(db_b)
            _reset_dirty(dirty_a)
            _reset_dirty(dirty_b)

            pm.gtrack_rm("dense_track", db=db_b, force=True)

            assert os.path.exists(dirty_b), "B (the database actually mutated) must be marked dirty"
            assert not os.path.exists(dirty_a), "A (GROOT, unrelated to this mutation) must NOT be marked dirty"
        finally:
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gtrack_mv_in_loaded_secondary_dataset_marks_that_dataset(self):
        """gtrack_mv resolves the track's home database via
        gtrack_dataset(src), which can be a loaded secondary dataset, not
        the primary _GROOT. Before this fix, the rename landed in that
        dataset but only the unrelated _GROOT got marked dirty."""
        db_a = pm.gdb_init_examples()  # will become the primary working DB

        # Build a second, independent DB with a uniquely-named track (so
        # loading it as a dataset alongside A has no name collision), then
        # load it as a secondary dataset under A.
        db_b = pm.gdb_init_examples()
        pm.gtrack_create("dataset_b_only_track", "b-only", "dense_track")
        pm.gdb_init(db_a)
        # force=True: B is a copy of the same example DB, so its other
        # tracks/intervals collide with A's; the working DB (A) wins those
        # per gdataset_load's documented behavior. The uniquely-named track
        # this test cares about has no collision and resolves to B either way.
        pm.gdataset_load(db_b, force=True)
        try:
            assert _same_path(pm._shared._GROOT, db_a)
            assert _path_in(db_b, pm._shared._GDATASETS)
            assert _same_path(pm.gtrack_dataset("dataset_b_only_track"), db_b)

            dirty_a = _dirty_sentinel_path(db_a)
            dirty_b = _dirty_sentinel_path(db_b)
            _reset_dirty(dirty_a)
            _reset_dirty(dirty_b)

            pm.gtrack_mv("dataset_b_only_track", "dataset_b_only_track_renamed")

            assert os.path.exists(dirty_b), "B (where the track actually lives) must be marked dirty"
            assert not os.path.exists(dirty_a), "A (GROOT, unrelated to this rename) must NOT be marked dirty"
        finally:
            pm.gdataset_unload(db_b, validate=False)
            pm.gdb_init(str(pm.gdb_examples_path()))

    def test_gtrack_create_with_uroot_marks_uroot_not_groot(self):
        """New tracks are written to _UROOT - a user-writable overlay root
        set via gdb_init(path, userpath=...), for working against a
        read-only shared database - not _GROOT, whenever _UROOT is set
        (see _target_root() in tracks.py). The sentinel must follow the
        actual write target, which every track-creation path resolves via
        _target_root() and was, before this fix, marking _GROOT dirty
        instead in every case where _UROOT differs from it."""
        groot = pm.gdb_init_examples()
        uroot = tempfile.mkdtemp(prefix="pymisha-uroot-")
        try:
            pm.gdb_init(groot, userpath=uroot)
            assert _same_path(pm._shared._UROOT, uroot)

            dirty_groot = _dirty_sentinel_path(groot)
            dirty_uroot = _dirty_sentinel_path(uroot)
            _reset_dirty(dirty_groot)
            _reset_dirty(dirty_uroot)

            pm.gtrack_create("test_dirty_uroot_track", "test", "dense_track")

            assert os.path.exists(dirty_uroot), "UROOT (the actual write target) must be marked dirty"
            assert not os.path.exists(dirty_groot), "GROOT (unrelated to this create) must NOT be marked dirty"
        finally:
            shutil.rmtree(uroot, ignore_errors=True)
            pm.gdb_init(str(pm.gdb_examples_path()))
