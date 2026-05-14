"""Tests for the ``_build_seq`` dispatcher and manual/local/s3 backends.

These tests are network-free; the s3 backend is exercised via monkeypatching
``_gdb_create_genome_from_s3``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pymisha.db_create as db_create_mod
from pymisha.genome._build_seq import _build_seq

FASTA_TWO_CONTIGS = ">chr1\nACGTACGTACGT\n>chr2\nTTTTAAAA\n"


def test_build_seq_manual_writes_groot(tmp_path):
    groot = tmp_path / "g"
    _build_seq(
        {"source": "manual", "content": FASTA_TWO_CONTIGS},
        groot,
        format="indexed",
    )
    assert (groot / "chrom_sizes.txt").exists()
    assert (groot / "seq").is_dir()
    lines = (groot / "chrom_sizes.txt").read_text().splitlines()
    assert len(lines) == 2


def test_build_seq_manual_per_chromosome(tmp_path):
    groot = tmp_path / "g"
    _build_seq(
        {"source": "manual", "content": FASTA_TWO_CONTIGS},
        groot,
        format="per-chromosome",
    )
    assert (groot / "seq" / "chr1.seq").exists()
    assert (groot / "seq" / "chr2.seq").exists()


def test_build_seq_manual_bytes_content(tmp_path):
    groot = tmp_path / "g"
    _build_seq(
        {"source": "manual", "content": FASTA_TWO_CONTIGS.encode("utf-8")},
        groot,
        format="indexed",
    )
    assert (groot / "chrom_sizes.txt").exists()
    assert (groot / "seq").is_dir()
    lines = (groot / "chrom_sizes.txt").read_text().splitlines()
    assert len(lines) == 2


def _make_tiny_groot(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "chrom_sizes.txt").write_text("chr1\t10\n", encoding="utf-8")
    (path / "seq").mkdir()


def test_build_seq_local_copies_existing_groot(tmp_path):
    source = tmp_path / "src"
    _make_tiny_groot(source)
    dest = tmp_path / "dest"

    _build_seq({"source": "local", "path": str(source)}, dest)

    assert (dest / "chrom_sizes.txt").exists()
    assert (dest / "seq").is_dir()
    assert (dest / "chrom_sizes.txt").read_text() == (
        source / "chrom_sizes.txt"
    ).read_text()


def test_build_seq_local_rejects_non_groot(tmp_path):
    bad = tmp_path / "not-a-groot"
    bad.mkdir()
    (bad / "random.txt").write_text("hi", encoding="utf-8")

    with pytest.raises(ValueError, match="not a misha groot"):
        _build_seq({"source": "local", "path": str(bad)}, tmp_path / "out")


def test_build_seq_local_rejects_existing_target(tmp_path):
    source = tmp_path / "src"
    _make_tiny_groot(source)
    dest = tmp_path / "dest"
    dest.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _build_seq({"source": "local", "path": str(source)}, dest)


def test_build_seq_unknown_source_raises_notimplemented(tmp_path):
    with pytest.raises(NotImplementedError, match="ucsc"):
        _build_seq(
            {"source": "ucsc", "assembly": "hg38"},
            tmp_path / "out",
        )


def _make_recording_s3_stub(calls: list) -> callable:
    """Return an `_gdb_create_genome_from_s3` stub that records args + kwargs."""
    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": dict(kwargs)})
        # Replicate helper contract: extract <dest_dir>/<name>.
        name = args[0] if args else kwargs["name"]
        dest_dir = args[1] if len(args) > 1 else kwargs["dest_dir"]
        out = Path(dest_dir) / name
        out.mkdir(parents=True)
        (out / "chrom_sizes.txt").write_text("chr1\t10\n", encoding="utf-8")

    return fake


def test_build_seq_s3_dispatches_to_helper(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        db_create_mod, "_gdb_create_genome_from_s3",
        _make_recording_s3_stub(calls),
    )

    _build_seq({"source": "s3", "name": "hg38"}, tmp_path / "hg38")

    assert len(calls) == 1
    args, kwargs = calls[0]["args"], calls[0]["kwargs"]
    # Name + dest_dir forwarded (positional in current helper signature).
    assert args[0] == "hg38"
    assert args[1] == str(tmp_path)
    # Dispatcher default verbose=True must be forwarded.
    assert kwargs.get("verbose") is True
    # Default groot location matches `<parent>/<name>`, so no rename.
    assert (tmp_path / "hg38" / "chrom_sizes.txt").exists()


def test_build_seq_s3_renames_when_groot_differs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        db_create_mod, "_gdb_create_genome_from_s3",
        _make_recording_s3_stub(calls),
    )

    _build_seq({"source": "s3", "name": "hg38"}, tmp_path / "mydb")

    assert (tmp_path / "mydb" / "chrom_sizes.txt").exists()
    assert not (tmp_path / "hg38").exists()
    # Verbose default should still be forwarded even on the rename path.
    assert calls[0]["kwargs"].get("verbose") is True


def test_build_seq_s3_forwards_verbose_false(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        db_create_mod, "_gdb_create_genome_from_s3",
        _make_recording_s3_stub(calls),
    )

    _build_seq(
        {"source": "s3", "name": "hg38"},
        tmp_path / "hg38",
        verbose=False,
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"].get("verbose") is False
