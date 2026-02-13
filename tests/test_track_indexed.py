import shutil
from pathlib import Path

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def test_track_convert_to_indexed(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        track = "dense_track"
        pm.gtrack_convert_to_indexed(track, remove_old=False)

        track_dir = root / "tracks" / "dense_track.track"
        assert (track_dir / "track.idx").exists()
        assert (track_dir / "track.dat").exists()

        info = pm.gtrack_info(track)
        assert info.get("format") == "indexed"

        df = pm.gextract(track, pm.gintervals("1", 0, 1000), iterator=100)
        assert df is not None and len(df) > 0
    finally:
        pm.gdb_init(str(TEST_DB))
