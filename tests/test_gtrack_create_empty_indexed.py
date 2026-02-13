from pathlib import Path

import pytest

import pymisha as pm


def _cleanup_track(track_name):
    if pm.gtrack_exists(track_name):
        pm.gtrack_rm(track_name, force=True)


def test_gtrack_create_empty_indexed_creates_idx_and_dat():
    track_name = "test.tmp_empty_indexed"
    _cleanup_track(track_name)
    try:
        intervals = pm.gintervals(["1"], [100], [150])
        pm.gtrack_create_sparse(track_name, "tmp", intervals, [1.0])

        track_dir = Path(pm._pymisha.pm_track_path(track_name))
        idx_path = track_dir / "track.idx"
        dat_path = track_dir / "track.dat"

        if idx_path.exists():
            idx_path.unlink()
        if dat_path.exists():
            dat_path.unlink()

        pm.gtrack_create_empty_indexed(track_name)
        assert idx_path.exists()
        assert dat_path.exists()
        assert idx_path.stat().st_size > 0
        assert dat_path.stat().st_size == 0

        # Idempotent overwrite should keep valid files.
        pm.gtrack_create_empty_indexed(track_name)
        assert idx_path.exists()
        assert dat_path.exists()
        assert idx_path.stat().st_size > 0
    finally:
        _cleanup_track(track_name)


def test_gtrack_create_empty_indexed_nonexistent_track_raises():
    track_name = "test.no_such_track_for_empty_indexed"
    _cleanup_track(track_name)
    with pytest.raises(Exception):
        pm.gtrack_create_empty_indexed(track_name)
