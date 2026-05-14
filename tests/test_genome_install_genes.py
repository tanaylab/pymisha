"""Tests for `_install_genes` (pymisha.genome._install_sets).

These tests monkeypatch `pymisha.gintervals_save` so no groot mutation
happens. We assert what *would* have been written and in which order.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import pymisha as pm
from pymisha.genome._install_sets import _install_genes

FIXTURE = Path(__file__).resolve().parent / "genome_fixtures" / "sample.gtf"


def _identity(chrom: str) -> str | None:
    return chrom


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[pd.DataFrame, str]]:
    """Replace gintervals_save with a recorder. Returns the call log list."""
    calls: list[tuple[pd.DataFrame, str]] = []

    def _fake_save(df: pd.DataFrame, name: str) -> None:
        calls.append((df.copy(), name))

    monkeypatch.setattr(pm, "gintervals_save", _fake_save)
    return calls


def test_install_genes_writes_four_sets(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_genes(FIXTURE.read_bytes(), _identity)
    names = [n for _, n in calls]
    assert set(names) == {"tss", "exons", "utr3", "utr5"}
    assert set(installed.keys()) == {"tss", "exons", "utr3", "utr5"}

    # Pull out the per-set DataFrames.
    by_name = {name: df for df, name in calls}

    # TSS: each row should be 1bp.
    tss = by_name["tss"]
    assert ((tss["end"] - tss["start"]) == 1).all()
    # 2 transcripts -> 2 TSS rows.
    assert len(tss) == 2

    # exons: 6 in fixture
    assert len(by_name["exons"]) == 6
    assert installed["exons"] == 6

    # utr5 and utr3: 1 each per transcript -> 2 each
    assert len(by_name["utr5"]) == 2
    assert len(by_name["utr3"]) == 2


def test_install_genes_skips_unmappable_chroms(monkeypatch):
    calls = _capture(monkeypatch)

    def _drop_chrom2(c: str) -> str | None:
        return None if c == "2" else c

    _install_genes(FIXTURE.read_bytes(), _drop_chrom2)
    for df, _name in calls:
        assert (df["chrom"] == "1").all()


def test_install_genes_respects_prefix(monkeypatch):
    calls = _capture(monkeypatch)
    _install_genes(FIXTURE.read_bytes(), _identity, prefix="ucsc.")
    names = {n for _, n in calls}
    assert names == {"ucsc.tss", "ucsc.exons", "ucsc.utr3", "ucsc.utr5"}


def test_install_genes_custom_gene_sets(monkeypatch):
    calls = _capture(monkeypatch)
    _install_genes(
        FIXTURE.read_bytes(),
        _identity,
        gene_sets={"tss": "promoters", "exons": "exonic"},
    )
    names = {n for _, n in calls}
    assert names == {"promoters", "exonic"}


def test_install_genes_tss_strand_aware(monkeypatch):
    """Verify exact TSS coordinates for + and - strand transcripts."""
    calls = _capture(monkeypatch)
    # Build two transcripts in-memory.
    gtf = (
        b'1\ttest\ttranscript\t100\t200\t.\t+\t.\tgene_id "GP"; transcript_id "TP";\n'
        b'1\ttest\ttranscript\t300\t400\t.\t-\t.\tgene_id "GM"; transcript_id "TM";\n'
    )
    _install_genes(gtf, _identity)
    by_name = {name: df for df, name in calls}
    tss = by_name["tss"].sort_values("start").reset_index(drop=True)
    # + strand: GTF start=100 -> 0-based 99; TSS is [99, 100)
    # - strand: GTF end=400; TSS is [end-1, end) -> [399, 400)
    assert list(tss["chrom"]) == ["1", "1"]
    assert list(tss["start"]) == [99, 399]
    assert list(tss["end"]) == [100, 400]


def test_install_genes_empty_after_filter(monkeypatch):
    calls = _capture(monkeypatch)
    # Only a CDS feature - not in the filter list.
    gtf = b'1\ttest\tCDS\t100\t200\t.\t+\t0\tgene_id "G"; transcript_id "T";\n'
    installed = _install_genes(gtf, _identity)
    assert calls == []
    assert installed == {}
