"""Interval-set names can't reach outside the DB, and a relative DB root
survives os.chdir()."""

import os
import shutil

import pytest

import pymisha as pm
from _dbpath import TESTDB_ROOT


def test_absolute_interval_set_name_never_touches_outside_files(tmp_path):
    victim = tmp_path / "important.interv"
    victim.write_text("keep me")
    name = str(tmp_path / "important")
    assert not pm.gintervals_exists(name)
    with pytest.raises(ValueError, match="Invalid intervals set name"):
        pm.gintervals_rm(name, force=True)
    assert victim.read_text() == "keep me"


def test_relative_root_survives_chdir(tmp_path, monkeypatch):
    shutil.copytree(TESTDB_ROOT, tmp_path / "db")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    try:
        pm.gdb_init("db")
        os.chdir(tmp_path / "elsewhere")
        assert len(pm.gextract("dense_track", pm.gintervals("chr1", 0, 1000))) > 0
        pm.gintervals_save(pm.gintervals("chr1", 0, 100), "reliv")
        assert (tmp_path / "db" / "tracks" / "reliv.interv").exists()
        assert not (tmp_path / "elsewhere" / "db").exists()
    finally:
        pm.gdb_init(str(TESTDB_ROOT))
