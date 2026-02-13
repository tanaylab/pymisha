"""Tests for gdir_* functions (database directory management)."""

import contextlib
import shutil
from pathlib import Path

import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


class TestGdirCwd:
    """Tests for gdir_cwd."""

    def test_cwd_returns_tracks_root(self):
        """After gdb_init, cwd should be the tracks directory."""
        cwd = pm.gdir_cwd()
        expected = str(TEST_DB / "tracks")
        assert cwd == expected

    def test_cwd_requires_db(self):
        """gdir_cwd raises if no database is initialized."""
        pm.gdb_unload()
        try:
            with pytest.raises(Exception):
                pm.gdir_cwd()
        finally:
            pm.gdb_init(str(TEST_DB))


class TestGdirCd:
    """Tests for gdir_cd."""

    def test_cd_to_subdir(self):
        """gdir_cd changes to a subdirectory and cwd reflects it."""
        pm.gdir_cd("subdir")
        try:
            cwd = pm.gdir_cwd()
            expected = str(TEST_DB / "tracks" / "subdir")
            assert cwd == expected
        finally:
            pm.gdir_cd("..")

    def test_cd_back_to_parent(self):
        """gdir_cd('..') returns to parent directory."""
        original = pm.gdir_cwd()
        pm.gdir_cd("subdir")
        pm.gdir_cd("..")
        assert pm.gdir_cwd() == original

    def test_cd_affects_track_listing(self):
        """After cd to subdir, track names should be relative to new cwd."""
        pm.gtrack_ls()
        pm.gdir_cd("subdir")
        try:
            subdir_tracks = pm.gtrack_ls()
            # In subdir, 'subdir.dense_track2' should appear as 'dense_track2'
            assert "dense_track2" in subdir_tracks
            # Top-level tracks like 'dense_track' should not be visible
            assert "dense_track" not in subdir_tracks
        finally:
            pm.gdir_cd("..")

    def test_cd_clears_vtracks(self):
        """gdir_cd should clear all virtual tracks."""
        pm.gvtrack_create("test_vt", "dense_track", func="avg")
        try:
            assert "test_vt" in pm.gvtrack_ls()
            pm.gdir_cd("subdir")
            assert "test_vt" not in pm.gvtrack_ls()
        finally:
            pm.gdir_cd("..")
            with contextlib.suppress(Exception):
                pm.gvtrack_rm("test_vt")

    def test_cd_nonexistent_dir_raises(self):
        """gdir_cd to a nonexistent directory raises."""
        with pytest.raises(Exception):
            pm.gdir_cd("this_does_not_exist_xyz")

    def test_cd_none_raises(self):
        """gdir_cd(None) raises."""
        with pytest.raises((TypeError, ValueError)):
            pm.gdir_cd(None)


class TestGdirCreate:
    """Tests for gdir_create."""

    def test_create_directory(self):
        """gdir_create creates a new directory in the tracks tree."""
        dirname = "test_gdir_newdir"
        full_path = TEST_DB / "tracks" / dirname
        try:
            pm.gdir_create(dirname)
            assert full_path.is_dir()
        finally:
            if full_path.exists():
                shutil.rmtree(full_path)

    def test_create_nested_fails(self):
        """gdir_create does not allow recursive creation."""
        with pytest.raises(Exception):
            pm.gdir_create("nonexistent_parent/child")

    def test_create_track_dir_fails(self):
        """gdir_create cannot create .track directories."""
        with pytest.raises(Exception):
            pm.gdir_create("bad_name.track")

    def test_create_relative_to_cwd(self):
        """gdir_create is relative to the current working directory."""
        dirname = "test_gdir_subdir_create"
        full_path = TEST_DB / "tracks" / "subdir" / dirname
        pm.gdir_cd("subdir")
        try:
            pm.gdir_create(dirname)
            assert full_path.is_dir()
        finally:
            pm.gdir_cd("..")
            if full_path.exists():
                shutil.rmtree(full_path)


class TestGdirRm:
    """Tests for gdir_rm."""

    def test_rm_empty_directory(self):
        """gdir_rm removes an empty directory."""
        dirname = "test_gdir_rm_empty"
        full_path = TEST_DB / "tracks" / dirname
        full_path.mkdir(exist_ok=True)
        try:
            pm.gdir_rm(dirname)
            assert not full_path.exists()
        finally:
            if full_path.exists():
                shutil.rmtree(full_path)

    def test_rm_nonempty_nonrecursive_fails(self):
        """gdir_rm without recursive=True fails on non-empty dir."""
        dirname = "test_gdir_rm_nonempty"
        full_path = TEST_DB / "tracks" / dirname
        full_path.mkdir(exist_ok=True)
        (full_path / "somefile.txt").write_text("hello")
        try:
            with pytest.raises(Exception):
                pm.gdir_rm(dirname)
        finally:
            if full_path.exists():
                shutil.rmtree(full_path)

    def test_rm_recursive(self):
        """gdir_rm with recursive=True removes non-empty directory."""
        dirname = "test_gdir_rm_recursive"
        full_path = TEST_DB / "tracks" / dirname
        full_path.mkdir(exist_ok=True)
        (full_path / "somefile.txt").write_text("hello")
        try:
            pm.gdir_rm(dirname, recursive=True, force=True)
            assert not full_path.exists()
        finally:
            if full_path.exists():
                shutil.rmtree(full_path)

    def test_rm_nonexistent_raises(self):
        """gdir_rm on nonexistent directory raises."""
        with pytest.raises(Exception):
            pm.gdir_rm("this_does_not_exist_xyz")

    def test_rm_nonexistent_force_silent(self):
        """gdir_rm on nonexistent dir with force=True is silent."""
        # Should not raise
        pm.gdir_rm("this_does_not_exist_xyz", force=True)

    def test_rm_relative_to_cwd(self):
        """gdir_rm is relative to the current working directory."""
        dirname = "test_gdir_rm_cwd"
        full_path = TEST_DB / "tracks" / "subdir" / dirname
        full_path.mkdir(exist_ok=True)
        pm.gdir_cd("subdir")
        try:
            pm.gdir_rm(dirname)
            assert not full_path.exists()
        finally:
            pm.gdir_cd("..")
            if full_path.exists():
                shutil.rmtree(full_path)


class TestGtrackCreateDirs:
    """Tests for gtrack_create_dirs."""

    def test_creates_namespace_dirs(self):
        """gtrack_create_dirs creates the directory hierarchy for a track name."""
        track_name = "test_ns1.test_ns2.my_track"
        dir1 = TEST_DB / "tracks" / "test_ns1"
        dir2 = dir1 / "test_ns2"
        try:
            pm.gtrack_create_dirs(track_name)
            assert dir1.is_dir()
            assert dir2.is_dir()
        finally:
            if dir1.exists():
                shutil.rmtree(dir1)

    def test_simple_track_name_no_dirs(self):
        """gtrack_create_dirs with no dots does not create directories."""
        # A simple track name like "my_track" has no namespace dirs
        pm.gtrack_create_dirs("simple_track")
        # Should not create anything (no namespace to create)

    def test_relative_to_cwd(self):
        """gtrack_create_dirs creates dirs relative to cwd."""
        pm.gdir_cd("subdir")
        dir_path = TEST_DB / "tracks" / "subdir" / "test_sub_ns"
        try:
            pm.gtrack_create_dirs("test_sub_ns.my_track")
            assert dir_path.is_dir()
        finally:
            pm.gdir_cd("..")
            if dir_path.exists():
                shutil.rmtree(dir_path)
