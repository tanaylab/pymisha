"""Tests for atomic gtrack_create_*: tmp dir + rename on success, trash on failure."""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import pymisha as pm


@pytest.fixture
def fresh_db():
    pm.gdb_init_examples()
    yield
    for tname in [
        "atomic_test_sparse", "atomic_test_fail", "atomic_test_interrupt",
        "atomic_test_concurrent_a", "atomic_test_concurrent_b",
    ]:
        with contextlib.suppress(Exception):
            pm.gtrack_rm(tname, force=True)


def _scrub_tmp_residue(parent: Path, track_basename: str):
    """Synchronously delete any .{track_basename}.tmp.* tmp dirs."""
    for entry in parent.glob(f".{track_basename}.tmp.*"):
        shutil.rmtree(entry, ignore_errors=True)


class TestAtomicCreateSuccessPath:
    def test_normal_create_leaves_final_dir(self, fresh_db):
        ivs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("atomic_test_sparse", "ok", ivs, [1.0])
        assert pm.gtrack_exists("atomic_test_sparse")
        track_dir = Path(pm.gtrack_path("atomic_test_sparse"))
        siblings = list(track_dir.parent.glob(f".{track_dir.name}.tmp.*"))
        assert siblings == []


class TestAtomicCreateFailurePath:
    def test_writer_failure_trashes_tmp(self, fresh_db):
        ivs = pm.gintervals("1", 0, 1000)
        with patch("pymisha._pymisha.pm_track_create_sparse",
                   side_effect=RuntimeError("simulated write failure")), \
             pytest.raises(RuntimeError, match="simulated write failure"):
            pm.gtrack_create_sparse("atomic_test_fail", "ok", ivs, [1.0])
        assert not pm.gtrack_exists("atomic_test_fail")
        # Tmp residue should be trashed (renamed to .trash.*) or already gone.
        from pymisha import _shared
        tracks_root = Path(_shared._GROOT) / "tracks"
        residue = list(tracks_root.glob(".atomic_test_fail.tmp.*"))
        # Either gone or in trash. Accept both - we just shouldn't have a
        # live tmp dir hanging around with the same name.
        assert residue == []

    def test_keyboard_interrupt_trashes_tmp(self, fresh_db):
        ivs = pm.gintervals("1", 0, 1000)
        with patch("pymisha._pymisha.pm_track_create_sparse",
                   side_effect=KeyboardInterrupt()), \
             pytest.raises(KeyboardInterrupt):
            pm.gtrack_create_sparse("atomic_test_interrupt", "ok", ivs, [1.0])
        assert not pm.gtrack_exists("atomic_test_interrupt")


class TestAtomicCreatePreCheck:
    def test_already_exists_blocks_before_tmp(self, fresh_db):
        ivs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("atomic_test_sparse", "first", ivs, [1.0])
        with pytest.raises(ValueError, match="already exists"):
            pm.gtrack_create_sparse("atomic_test_sparse", "second", ivs, [2.0])
        track_dir = Path(pm.gtrack_path("atomic_test_sparse"))
        siblings = list(track_dir.parent.glob(f".{track_dir.name}.tmp.*"))
        assert siblings == []


class TestAtomicCreateConcurrent:
    def test_concurrent_rescan_does_not_see_tmp(self, fresh_db, monkeypatch):
        observed = []
        ivs = pm.gintervals("1", 0, 1000)

        from pymisha import _pymisha as _pm
        original = _pm.pm_track_create_sparse

        def slow_writer(*args, **kwargs):
            observed.append("atomic_test_sparse" in (pm.gtrack_ls() or []))
            return original(*args, **kwargs)

        monkeypatch.setattr("pymisha._pymisha.pm_track_create_sparse", slow_writer)
        pm.gtrack_create_sparse("atomic_test_sparse", "ok", ivs, [1.0])
        assert observed == [False], (
            f"in-flight create should be invisible to gtrack_ls; got {observed}"
        )
