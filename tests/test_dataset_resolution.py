import shutil
from pathlib import Path

import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def test_gtrack_dataset_resolves_active_root():
    assert pm.gtrack_dataset("dense_track") == str(TEST_DB)


def test_gintervals_dataset_nonexistent_returns_none():
    assert pm.gintervals_dataset("no_such_intervals_set") is None


def test_resolution_prefers_active_root_on_collisions(tmp_path):
    dataset_root = _copy_db(tmp_path)

    pm.gdataset_load(str(dataset_root), force=True)
    try:
        # Working DB should win when names collide.
        assert pm.gtrack_dataset("dense_track") == str(TEST_DB)
        assert pm.gintervals_dataset("annotations") == str(TEST_DB)
    finally:
        pm.gdataset_unload(str(dataset_root), validate=True)


def test_gintervals_dataset_requires_name():
    with pytest.raises(ValueError, match="cannot be None"):
        pm.gintervals_dataset(None)
