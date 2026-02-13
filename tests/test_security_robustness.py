import os
import pickle
import shutil
import tarfile
from pathlib import Path

import pytest

import pymisha as pm
import pymisha.db_create as db_create
import pymisha.intervals as intervals_mod
import pymisha.liftover as liftover_mod
from pymisha.tracks import _validate_track_name

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _track_var_dir(track):
    root = pm.gtrack_dataset(track)
    track_path = Path(root) / "tracks" / f"{track.replace('.', '/')}.track"
    return track_path / "vars"


def test_gtrack_var_rejects_path_traversal_name(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        with pytest.raises(ValueError):
            pm.gtrack_var_set("dense_track", "../escape", 1)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_var_rejects_unsafe_pickle_payload(tmp_path):
    marker = tmp_path / "pwned"
    root = _copy_db(tmp_path)

    class _Exploit:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    try:
        pm.gdb_init(str(root))
        payload = pickle.dumps(_Exploit(), protocol=pickle.HIGHEST_PROTOCOL)
        var_dir = _track_var_dir("dense_track")
        var_dir.mkdir(parents=True, exist_ok=True)
        (var_dir / "malicious").write_bytes(payload)

        with pytest.raises(ValueError):
            pm.gtrack_var_get("dense_track", "malicious")
        assert not marker.exists()
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_var_set_rejects_unsupported_object_types(tmp_path):
    root = _copy_db(tmp_path)

    class _Custom:
        pass

    try:
        pm.gdb_init(str(root))
        with pytest.raises(TypeError):
            pm.gtrack_var_set("dense_track", "custom_obj", _Custom())
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gsynth_load_rejects_unsafe_pickle(tmp_path):
    marker = tmp_path / "gsynth_pwned"

    class _Exploit:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    payload_path = tmp_path / "model.pkl"
    payload_path.write_bytes(pickle.dumps(_Exploit(), protocol=pickle.HIGHEST_PROTOCOL))

    with pytest.raises(Exception):
        pm.gsynth_load(str(payload_path))
    assert not marker.exists()


def test_expression_eval_blocks_object_traversal():
    intervals = pm.gintervals(["1"], [0], [100])
    expr = "().__class__.__bases__[0].__subclasses__()"

    with pytest.raises(ValueError, match="Unsafe expression"):
        pm.gextract(expr, intervals=intervals, iterator=10)

    with pytest.raises(ValueError, match="Unsafe expression"):
        pm.gsummary(expr, intervals=intervals, iterator=10)


def test_gdir_rejects_escape_from_tracks_tree():
    with pytest.raises(ValueError):
        pm.gdir_create("../outside_dir")

    with pytest.raises(ValueError):
        pm.gdir_rm("../outside_dir", force=True)


def test_track_name_validation_rejects_empty_components():
    with pytest.raises(ValueError):
        _validate_track_name("a..b")
    with pytest.raises(ValueError):
        _validate_track_name("a.")


def test_gintervals_save_rejects_empty_dot_components():
    ivals = pm.gintervals(["1"], [0], [10])
    with pytest.raises(ValueError):
        pm.gintervals_save(ivals, "a..b")


def test_download_file_rejects_file_scheme(tmp_path):
    out = tmp_path / "out.bin"
    with pytest.raises(ValueError):
        db_create._download_file("file:///etc/passwd", out)


def test_safe_extract_tar_rejects_symlinks(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("bad_link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    with pytest.raises(ValueError, match="Unsupported tar member type"):
        db_create._safe_extract_tar(archive, tmp_path / "out")


def test_parse_chain_file_requires_regular_file(tmp_path):
    with pytest.raises(ValueError):
        liftover_mod._parse_chain_file(tmp_path, db_chrom_sizes={})


def test_decode_intervals_meta_reports_missing_rscript(monkeypatch, tmp_path):
    monkeypatch.setattr(intervals_mod.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Rscript is required"):
        intervals_mod._decode_intervals_meta(tmp_path / ".meta")
