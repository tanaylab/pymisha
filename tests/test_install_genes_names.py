"""gene name / symbol columns on installed gene sets.

Ported from R misha tests/testthat/test-genome-build-installers.R (5.10.0):
the installed tss / exons / utr3 / utr5 sets carry a `name` column (transcript
accession) and a `geneName` column (gene symbol; gene_id fallback when the
source has no symbol; blank when neither). Overlapping features that unify to
one interval concatenate their distinct symbols with ";".
"""
from __future__ import annotations

import pandas as pd

import pymisha as pm
from pymisha.genome._install_sets import _install_genes


def _identity(chrom: str) -> str | None:
    return chrom


def _capture(monkeypatch):
    calls: list[tuple[pd.DataFrame, str]] = []

    def _fake_save(df, name):
        calls.append((df.copy(), name))

    monkeypatch.setattr(pm, "gintervals_save", _fake_save)
    return calls


def _gtf(lines: list[str]) -> bytes:
    return "\n".join(lines).encode("ascii")


def test_install_genes_attaches_name_and_geneName(monkeypatch):
    """The bundled fixture has gene_id but no gene_name -> gene_id fallback."""
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "genome_fixtures" / "sample.gtf"
    calls = _capture(monkeypatch)
    _install_genes(fixture.read_bytes(), _identity)
    by_name = {name: df for df, name in calls}

    tss = by_name["tss"]
    assert "name" in tss.columns
    assert "geneName" in tss.columns
    assert set(tss["name"]) == {"T1", "T2"}
    # No gene_name attribute -> geneName falls back to gene_id.
    assert set(tss["geneName"]) == {"G1", "G2"}


def test_install_genes_geneName_uses_gene_name_when_present(monkeypatch):
    gtf = _gtf(
        [
            '1\tt\ttranscript\t100\t2000\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; gene_name "Actb";',
            '1\tt\texon\t100\t500\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; gene_name "Actb";',
        ]
    )
    calls = _capture(monkeypatch)
    _install_genes(gtf, _identity)
    by_name = {name: df for df, name in calls}
    assert by_name["tss"]["geneName"].iloc[0] == "Actb"
    assert by_name["tss"]["name"].iloc[0] == "T1"


def test_install_genes_concatenates_symbols_on_overlapping_tss(monkeypatch):
    # Two transcripts of different genes with the SAME TSS coordinate unify to
    # one interval; their distinct symbols concatenate with ";".
    gtf = _gtf(
        [
            '1\tt\ttranscript\t100\t500\t.\t+\t.\tgene_id "G1"; transcript_id "txA"; gene_name "GeneA";',
            '1\tt\ttranscript\t100\t600\t.\t+\t.\tgene_id "G2"; transcript_id "txB"; gene_name "GeneB";',
        ]
    )
    calls = _capture(monkeypatch)
    _install_genes(gtf, _identity)
    by_name = {name: df for df, name in calls}
    tss = by_name["tss"]
    assert len(tss) == 1
    gn = tss["geneName"].iloc[0]
    assert "GeneA" in gn and "GeneB" in gn
    assert ";" in gn


def test_install_genes_blank_geneName_when_no_symbol(monkeypatch):
    # Neither gene_name nor gene_id -> blank geneName, row not dropped.
    gtf = _gtf(
        [
            '1\tt\ttranscript\t100\t2000\t.\t+\t.\ttranscript_id "T1";',
            '1\tt\texon\t100\t500\t.\t+\t.\ttranscript_id "T1";',
        ]
    )
    calls = _capture(monkeypatch)
    _install_genes(gtf, _identity)
    by_name = {name: df for df, name in calls}
    tss = by_name["tss"]
    assert len(tss) == 1
    assert tss["geneName"].iloc[0] == ""
    assert tss["name"].iloc[0] == "T1"
