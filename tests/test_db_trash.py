"""Tests for _db_trash helpers - atomic-rename + async unlink."""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from unittest.mock import patch

import pytest

import pymisha as pm
from pymisha._db_trash import _gdb_trash, _gdb_trash_sweep_old


class TestGdbTrash:
    def test_trash_nonexistent_returns_false(self, tmp_path):
        target = tmp_path / "does_not_exist"
        assert _gdb_trash(target) is False

    def test_trash_atomic_rename_succeeds(self, tmp_path):
        target = tmp_path / "doomed"
        target.mkdir()
        (target / "x").write_text("hello")
        t0 = time.monotonic()
        assert _gdb_trash(target, async_unlink=False) is True
        elapsed = time.monotonic() - t0
        assert not target.exists()
        # Rename should be near-instant even for non-trivial dirs.
        assert elapsed < 0.5

    def test_trash_async_detaches(self, tmp_path):
        target = tmp_path / "async_doomed"
        target.mkdir()
        (target / "file.txt").write_text("data")
        assert _gdb_trash(target, async_unlink=True) is True
        assert not target.exists()
        # The background rm may not have finished yet but `target` is gone.
        # Wait briefly for the .trash.* sibling to be cleaned up by the
        # detached rm, with a generous bound.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            siblings = list(tmp_path.glob(".trash.*"))
            if not siblings:
                break
            time.sleep(0.05)
        assert list(tmp_path.glob(".trash.*")) == []

    def test_trash_cross_fs_fallback(self, tmp_path):
        target = tmp_path / "cross_fs"
        target.mkdir()
        (target / "f").write_text("x")
        with patch("os.rename", side_effect=OSError("EXDEV simulated")):
            assert _gdb_trash(target, async_unlink=False) is True
        assert not target.exists()

    def test_trash_returns_false_when_path_persists(self, tmp_path):
        target = tmp_path / "stubborn"
        target.mkdir()
        # Both rename and rmtree fail to clear it.
        with patch("os.rename", side_effect=OSError), \
             patch("shutil.rmtree"):
            result = _gdb_trash(target, async_unlink=False)
        assert result is False
        # Cleanup so pytest's tmp_path teardown works.
        shutil.rmtree(target, ignore_errors=True)


class TestGdbTrashSweepOld:
    def test_sweep_nonexistent_parent_returns_zero(self):
        assert _gdb_trash_sweep_old("/nonexistent/path") == 0

    def test_sweep_respects_cutoff(self, tmp_path):
        old = tmp_path / ".trash.old.123.abc"
        new = tmp_path / ".trash.new.456.def"
        old.mkdir()
        new.mkdir()
        # Age `old` to 48 hours ago; leave `new` fresh.
        ancient = time.time() - 48 * 3600
        os.utime(old, (ancient, ancient))
        swept = _gdb_trash_sweep_old(tmp_path, max_age_hours=24)
        assert swept == 1
        assert not old.exists()
        assert new.exists()

    def test_sweep_skips_unreadable_entries(self, tmp_path):
        bad = tmp_path / ".trash.unreadable.0.x"
        bad.mkdir()
        with patch("pathlib.Path.stat", side_effect=PermissionError):
            swept = _gdb_trash_sweep_old(tmp_path, max_age_hours=24)
        assert swept == 0
        assert bad.exists()


class TestRmUsesTrash:
    """gtrack_rm and gintervals_rm route through _gdb_trash."""

    def test_gtrack_rm_calls_trash(self, monkeypatch):
        pm.gdb_init_examples()
        # Build a throwaway track so we have something to delete.
        intervals = pm.gintervals("1", 0, 1000)
        values = [1.0]
        pm.gtrack_create_sparse("test_rm_via_trash", "test", intervals, values)
        try:
            calls = []

            from pymisha import _db_trash as _t
            _gdb_trash_real = _t._gdb_trash

            def spy(path, async_unlink=True):
                calls.append((str(path), async_unlink))
                # Delegate to the real helper for actual cleanup.
                return _gdb_trash_real(path, async_unlink=async_unlink)

            monkeypatch.setattr(_t, "_gdb_trash", spy)
            # Also monkey-patch the imported alias inside tracks.py if any.
            from pymisha import tracks as _tracks
            if hasattr(_tracks, "_gdb_trash"):
                monkeypatch.setattr(_tracks, "_gdb_trash", spy)

            pm.gtrack_rm("test_rm_via_trash", force=True)
            assert any("test_rm_via_trash.track" in p for p, _ in calls)
        finally:
            with contextlib.suppress(Exception):
                pm.gtrack_rm("test_rm_via_trash", force=True)

    def test_gtrack_rm_raises_when_trash_returns_false(self, monkeypatch):
        pm.gdb_init_examples()
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("test_rm_fail", "test", intervals, [1.0])
        try:
            from pymisha import tracks as _tracks
            monkeypatch.setattr(_tracks, "_gdb_trash", lambda *a, **k: False)
            with pytest.raises(RuntimeError, match="failed to remove"):
                pm.gtrack_rm("test_rm_fail", force=True)
        finally:
            # _gdb_trash returned False, so gtrack_rm raised before pm_dbreload.
            # Clear the phantom track from the C++ cache after manual cleanup.
            with contextlib.suppress(Exception):
                from pymisha import _db_trash as _t
                from pymisha import _pymisha
                track_dir = pm.gtrack_path("test_rm_fail")
                _t._gdb_trash(track_dir, async_unlink=False)
                _pymisha.pm_dbreload()

    def test_gintervals_rm_calls_trash(self, monkeypatch):
        pm.gdb_init_examples()
        # gintervals_save produces a file (small set), so the directory
        # branch we are wiring through _gdb_trash needs a manually-built
        # "bigset" interval directory to exercise.
        from pymisha import _shared
        groot = _shared._GROOT
        assert groot is not None
        interv_dir = os.path.join(groot, "tracks", "test_iv_rm_via_trash.interv")
        os.makedirs(interv_dir, exist_ok=True)
        # Drop a placeholder file so it looks like a populated bigset.
        with open(os.path.join(interv_dir, "placeholder"), "w") as f:
            f.write("x")
        try:
            calls = []
            from pymisha import _db_trash as _t
            real = _t._gdb_trash

            def spy(p, async_unlink=True):
                calls.append(p)
                return real(p, async_unlink=async_unlink)

            monkeypatch.setattr(_t, "_gdb_trash", spy)
            from pymisha import intervals as _iv
            if hasattr(_iv, "_gdb_trash"):
                monkeypatch.setattr(_iv, "_gdb_trash", spy)
            pm.gintervals_rm("test_iv_rm_via_trash", force=True)
            assert calls
        finally:
            with contextlib.suppress(Exception):
                if os.path.isdir(interv_dir):
                    shutil.rmtree(interv_dir, ignore_errors=True)


class TestGdbInitSweeps:
    def test_gdb_init_sweeps_stale_trash(self):
        # Use a real example DB copy to avoid clobbering shared state.
        from pathlib import Path
        groot = pm.gdb_init_examples()
        tracks_dir = Path(groot) / "tracks"
        stale = tracks_dir / ".trash.stale.0.deadbeef"
        stale.mkdir()
        ancient = time.time() - 48 * 3600
        os.utime(stale, (ancient, ancient))
        # Re-init: should sweep stale.
        pm.gdb_init(groot)
        assert not stale.exists()


class TestSweepTmpOrphans:
    def test_sweep_old_removes_stale_tmp_dirs(self, tmp_path):
        old_tmp = tmp_path / ".sometrack.tmp.123.deadbeef"
        new_tmp = tmp_path / ".other.tmp.456.cafebabe"
        old_tmp.mkdir()
        new_tmp.mkdir()
        ancient = time.time() - 48 * 3600
        os.utime(old_tmp, (ancient, ancient))
        swept = _gdb_trash_sweep_old(tmp_path, max_age_hours=24)
        assert swept >= 1
        assert not old_tmp.exists()
        assert new_tmp.exists()
