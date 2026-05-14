"""Tests for the public ``gdb_build_genome`` skeleton (v0.1.43).

Network-free: ``gdb_init`` is monkeypatched in tests that would otherwise
bind a tmp groot into the active misha session (which would break the
session-scoped test DB fixture in ``conftest.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pymisha
from pymisha.genome import gdb_build_genome

FASTA_TWO_CONTIGS = ">chr1\nACGTACGTACGT\n>chr2\nTTTTAAAA\n"


def _write_manual_registry(yaml_path: Path, key: str = "myrecipe") -> None:
    yaml_path.write_text(
        "version: 1\n"
        "genome:\n"
        f"  {key}: {{source: manual, content: '{FASTA_TWO_CONTIGS}'}}\n",
        encoding="utf-8",
    )


def test_gdb_build_genome_manual_end_to_end(tmp_path, monkeypatch):
    """Build a manual recipe end-to-end with gdb_init stubbed out."""
    calls: list[str] = []
    monkeypatch.setattr(
        "pymisha.db.gdb_init",
        lambda p, *a, **kw: calls.append(str(p)),
    )

    registry = tmp_path / "registry.yaml"
    _write_manual_registry(registry)
    groot = tmp_path / "g"

    gdb_build_genome(
        "myrecipe",
        path=str(groot),
        registry=str(registry),
        format="indexed",
        verbose=False,
    )

    assert (groot / "chrom_sizes.txt").exists()
    assert (groot / "seq").is_dir()
    # gdb_init was invoked with the groot path.
    assert calls == [str(groot)]


def test_gdb_build_genome_unknown_genome_raises_keyerror(tmp_path):
    """A genome name absent from every registry layer must raise KeyError."""
    with pytest.raises(KeyError, match="not in any registry layer"):
        gdb_build_genome(
            "no_such_genome_xyz_abc",
            path=str(tmp_path / "g"),
            verbose=False,
        )


def test_gdb_build_genome_sets_with_manual_source_raises(tmp_path, monkeypatch):
    """Manual recipes have no asset fetcher; non-empty ``sets`` must raise."""
    monkeypatch.setattr(
        "pymisha.db.gdb_init",
        lambda p, *a, **kw: None,
    )
    registry = tmp_path / "registry.yaml"
    _write_manual_registry(registry)

    with pytest.raises(NotImplementedError, match="manual"):
        gdb_build_genome(
            "myrecipe",
            path=str(tmp_path / "g"),
            registry=str(registry),
            sets=("genes",),
            verbose=False,
        )


def test_gdb_build_genome_path_defaults_to_name(tmp_path, monkeypatch):
    """When ``path`` is None, the groot is created at ``cwd / name``."""
    calls: list[str] = []
    monkeypatch.setattr(
        "pymisha.db.gdb_init",
        lambda p, *a, **kw: calls.append(str(p)),
    )

    registry = tmp_path / "registry.yaml"
    _write_manual_registry(registry, key="defaultpath")
    monkeypatch.chdir(tmp_path)

    gdb_build_genome(
        "defaultpath",
        registry=str(registry),
        format="indexed",
        verbose=False,
    )

    assert (tmp_path / "defaultpath" / "chrom_sizes.txt").exists()
    assert calls == ["defaultpath"]


def test_gdb_build_genome_calls_gdb_init(tmp_path, monkeypatch):
    """``gdb_init`` must be called exactly once with the groot path after build."""
    init_calls: list[tuple] = []

    def fake_init(*args, **kwargs):
        init_calls.append((args, kwargs))

    monkeypatch.setattr("pymisha.db.gdb_init", fake_init)

    registry = tmp_path / "registry.yaml"
    _write_manual_registry(registry, key="initcheck")
    groot = tmp_path / "outg"

    gdb_build_genome(
        "initcheck",
        path=str(groot),
        registry=str(registry),
        verbose=False,
    )

    assert len(init_calls) == 1
    args, kwargs = init_calls[0]
    assert args == (str(groot),)
    assert kwargs == {}


def test_gdb_build_genome_exported_at_top_level():
    """``gdb_build_genome`` must be importable as ``pymisha.gdb_build_genome``."""
    assert hasattr(pymisha, "gdb_build_genome")
    assert pymisha.gdb_build_genome is gdb_build_genome
    assert "gdb_build_genome" in pymisha.__all__
