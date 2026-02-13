"""Tests for DB-admin APIs: readonly attrs and gdb_create_genome."""

import contextlib
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import pytest

import pymisha as pm
import pymisha.db_create as db_create_mod
from pymisha import _shared
from pymisha import db_attrs as db_attrs_mod
from pymisha.tracks import _load_track_attributes, _save_track_attributes

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


@pytest.fixture(autouse=True)
def _restore_readonly_attrs_file():
    ro_path = TEST_DB / ".ro_attributes"
    backup = ro_path.read_bytes() if ro_path.exists() else None
    yield
    if backup is None:
        ro_path.unlink(missing_ok=True)
    else:
        ro_path.write_bytes(backup)
    pm.gdb_init(str(TEST_DB))


@pytest.fixture(autouse=True)
def _restore_track_attrs():
    tracks_to_save = ["dense_track", "sparse_track"]
    saved = {}
    for track in tracks_to_save:
        try:
            saved[track] = dict(_load_track_attributes(track))
        except Exception:
            saved[track] = {}
    yield
    for track, attrs in saved.items():
        with contextlib.suppress(Exception):
            _save_track_attributes(track, attrs)


class TestReadonlyAttrsApis:
    def test_readonly_attrs_path_requires_initialized_db(self, monkeypatch):
        monkeypatch.setattr(_shared, "_GROOT", None)
        with pytest.raises(Exception, match="Database not set"):
            db_attrs_mod._readonly_attrs_path()

    def test_get_existing_readonly_attrs(self):
        attrs = pm.gdb_get_readonly_attrs()
        assert attrs is not None
        assert "created.by" in attrs
        assert "created.date" in attrs

    def test_set_and_get_readonly_attrs(self):
        pm.gdb_set_readonly_attrs(["created.by", "test.attr"])
        assert pm.gdb_get_readonly_attrs() == ["created.by", "test.attr"]

    def test_set_none_clears_readonly_attrs(self):
        pm.gdb_set_readonly_attrs(["temp.attr"])
        pm.gdb_set_readonly_attrs(None)
        assert pm.gdb_get_readonly_attrs() is None
        assert not (TEST_DB / ".ro_attributes").exists()

    def test_rejects_duplicates_and_empty(self):
        with pytest.raises(ValueError, match="appears more than once"):
            pm.gdb_set_readonly_attrs(["dup", "dup"])
        with pytest.raises(ValueError, match="empty string"):
            pm.gdb_set_readonly_attrs(["valid", ""])

    def test_gtrack_attr_set_rejects_readonly_attr(self):
        pm.gdb_set_readonly_attrs(["blocked.attr"])
        with pytest.raises(ValueError, match="read-only"):
            pm.gtrack_attr_set("dense_track", "blocked.attr", "value")

    def test_gtrack_attr_import_rejects_readonly_attr(self):
        pm.gdb_set_readonly_attrs(["blocked.attr"])
        table = pd.DataFrame({"blocked.attr": ["x"]}, index=["dense_track"])
        with pytest.raises(ValueError, match="read-only"):
            pm.gtrack_attr_import(table)

    def test_remove_others_preserves_readonly_attrs(self):
        pm.gdb_set_readonly_attrs(["created.by"])
        created_by = pm.gtrack_attr_get("dense_track", "created.by")
        assert created_by != ""

        pm.gtrack_attr_set("dense_track", "temp.attr", "keep-for-now")
        table = pd.DataFrame({"new.attr": ["new-value"]}, index=["dense_track"])
        pm.gtrack_attr_import(table, remove_others=True)

        assert pm.gtrack_attr_get("dense_track", "temp.attr") == ""
        assert pm.gtrack_attr_get("dense_track", "new.attr") == "new-value"
        assert pm.gtrack_attr_get("dense_track", "created.by") == created_by


def _build_mock_genome_tar(tmp_path, genome):
    root = tmp_path / genome
    (root / "seq").mkdir(parents=True)
    (root / "tracks").mkdir()
    (root / "pssms").mkdir()
    (root / "seq" / "genome.seq").write_bytes(b"ACGT")
    (root / "seq" / "genome.idx").write_bytes(b"MISHAIDX")
    (root / "chrom_sizes.txt").write_text("chr1\t4\n", encoding="utf-8")

    tar_path = tmp_path / f"{genome}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(root, arcname=genome)
    return tar_path


class TestGdbCreateGenome:
    def test_download_extract_and_init(self, tmp_path, monkeypatch):
        genome = "mm10"
        tar_path = _build_mock_genome_tar(tmp_path, genome)
        downloaded_urls = []

        def _fake_download(url, dst_path):
            downloaded_urls.append(url)
            shutil.copyfile(tar_path, dst_path)

        def _fake_download_text(url):
            return db_create_mod._sha256_file(tar_path)

        monkeypatch.setattr(db_create_mod, "_download_file", _fake_download)
        monkeypatch.setattr(db_create_mod, "_download_text", _fake_download_text)

        out_dir = tmp_path / "out"
        try:
            pm.gdb_create_genome(genome, path=str(out_dir), tmpdir=str(tmp_path))
            extracted = out_dir / genome
            assert extracted.is_dir()
            assert (extracted / "seq" / "genome.seq").is_file()
            assert (extracted / "seq" / "genome.idx").is_file()
            assert pm.gdb_info()["path"] == str(extracted.resolve(strict=False))
            assert downloaded_urls == [
                f"https://misha-genome.s3.eu-west-1.amazonaws.com/{genome}.tar.gz"
            ]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_rejects_unsupported_genome(self, tmp_path):
        with pytest.raises(ValueError, match="not available yet"):
            pm.gdb_create_genome("unsupported_genome", path=str(tmp_path))
