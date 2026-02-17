"""Tests for gdb_export_fasta (database FASTA export)."""

from pathlib import Path

import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


@pytest.fixture(autouse=True)
def _restore_root():
    yield
    pm.gdb_init(str(TEST_DB))


def _create_db(tmp_path, name, fasta_text, *, db_format="indexed"):
    fasta_path = tmp_path / f"{name}.fa"
    db_path = tmp_path / name
    fasta_path.write_text(fasta_text, encoding="utf-8")
    pm.gdb_create(str(db_path), str(fasta_path), db_format=db_format)
    return db_path


def test_exports_current_database_with_wrapping_and_chunking(tmp_path):
    db_path = _create_db(
        tmp_path,
        "db_wrapping",
        ">chrA\nACTGACTG\n>chrB\nTTAA\n",
    )
    pm.gdb_init(str(db_path))

    out_fasta = tmp_path / "out.fa"
    result = pm.gdb_export_fasta(str(out_fasta), line_width=3, chunk_size=2)

    assert result == str(out_fasta)
    assert out_fasta.read_text(encoding="utf-8").splitlines() == [
        ">chrA",
        "ACT",
        "GAC",
        "TG",
        ">chrB",
        "TTA",
        "A",
    ]


def test_export_with_explicit_groot_restores_previous_root(tmp_path):
    db1 = _create_db(tmp_path, "db1", ">chr1\nAAAA\n")
    db2 = _create_db(tmp_path, "db2", ">chr2\nCCCC\n")
    pm.gdb_init(str(db1))
    original_root = Path(pm._shared._GROOT).resolve(strict=False)

    out_fasta = tmp_path / "explicit.fa"
    pm.gdb_export_fasta(
        str(out_fasta),
        groot=str(db2),
        line_width=2,
        chunk_size=2,
    )

    assert Path(pm._shared._GROOT).resolve(strict=False) == original_root
    assert out_fasta.read_text(encoding="utf-8").splitlines() == [
        ">chr2",
        "CC",
        "CC",
    ]


def test_overwrite_guard(tmp_path):
    db_path = _create_db(tmp_path, "db_overwrite", ">chr1\nACTG\n")
    pm.gdb_init(str(db_path))

    out_fasta = tmp_path / "overwrite.fa"
    out_fasta.write_text("placeholder\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pm.gdb_export_fasta(str(out_fasta))

    pm.gdb_export_fasta(str(out_fasta), overwrite=True)
    assert out_fasta.read_text(encoding="utf-8").splitlines() == [">chr1", "ACTG"]


def test_per_chromosome_export_supports_chr_prefix_fallback(tmp_path):
    db_path = tmp_path / "db_legacy"
    seq_dir = db_path / "seq"
    seq_dir.mkdir(parents=True)
    (db_path / "tracks").mkdir()
    (db_path / "pssms").mkdir()
    (db_path / "chrom_sizes.txt").write_text("chr1\t4\n", encoding="utf-8")
    (seq_dir / "1.seq").write_bytes(b"ACTG")

    pm.gdb_init(str(db_path))
    out_fasta = tmp_path / "legacy.fa"
    pm.gdb_export_fasta(str(out_fasta))

    assert out_fasta.read_text(encoding="utf-8").splitlines() == [">chr1", "ACTG"]
