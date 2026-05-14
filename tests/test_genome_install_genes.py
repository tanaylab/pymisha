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


# ---------------------------------------------------------------------------
# Real-shape backend inputs: NCBI GFF3 + UCSC GTF use feature names that
# differ from the Ensembl/GENCODE conventions of the sample.gtf fixture.
# Pre-fix, both backends silently dropped TSS and/or UTR sets.
# ---------------------------------------------------------------------------

# RefSeq GFF3 shape: mRNA, five_prime_UTR, three_prime_UTR; key=value attrs.
NCBI_GFF3 = b"""##gff-version 3
##sequence-region NC_000001.11 1 248956422
NC_000001.11\tBestRefSeq\tgene\t1000\t5000\t.\t+\t.\tID=gene-A;Name=A
NC_000001.11\tBestRefSeq\tmRNA\t1000\t5000\t.\t+\t.\tID=rna-A;Parent=gene-A
NC_000001.11\tBestRefSeq\texon\t1000\t1500\t.\t+\t.\tID=exon-A-1;Parent=rna-A
NC_000001.11\tBestRefSeq\texon\t3000\t5000\t.\t+\t.\tID=exon-A-2;Parent=rna-A
NC_000001.11\tBestRefSeq\tfive_prime_UTR\t1000\t1200\t.\t+\t.\tID=utr5-A;Parent=rna-A
NC_000001.11\tBestRefSeq\tthree_prime_UTR\t4800\t5000\t.\t+\t.\tID=utr3-A;Parent=rna-A
NC_000001.11\tBestRefSeq\tCDS\t1201\t4799\t.\t+\t0\tID=cds-A;Parent=rna-A
"""

# UCSC ncbiRefSeq.gtf.gz shape: transcript + exon + 5UTR/3UTR; key "value" attrs.
UCSC_GTF = (
    b'chrM\tncbiRefSeq\ttranscript\t100\t500\t.\t+\t.\tgene_id "G1"; transcript_id "rna-G1";\n'
    b'chrM\tncbiRefSeq\texon\t100\t250\t.\t+\t.\tgene_id "G1"; transcript_id "rna-G1"; exon_number "1";\n'
    b'chrM\tncbiRefSeq\texon\t350\t500\t.\t+\t.\tgene_id "G1"; transcript_id "rna-G1"; exon_number "2";\n'
    b'chrM\tncbiRefSeq\t5UTR\t100\t150\t.\t+\t.\tgene_id "G1"; transcript_id "rna-G1";\n'
    b'chrM\tncbiRefSeq\t3UTR\t450\t500\t.\t+\t.\tgene_id "G1"; transcript_id "rna-G1";\n'
    b'chrM\tncbiRefSeq\tCDS\t151\t449\t.\t+\t0\tgene_id "G1"; transcript_id "rna-G1";\n'
)


def test_install_genes_recognizes_ncbi_gff3_feature_names(monkeypatch):
    """NCBI RefSeq GFF3 uses mRNA + five_prime_UTR + three_prime_UTR (capital).

    Pre-fix: TSS, utr5, utr3 sets came out empty.
    """
    calls = _capture(monkeypatch)
    installed = _install_genes(NCBI_GFF3, lambda c: "1" if c == "NC_000001.11" else None)
    by_name = {name: df for df, name in calls}
    assert set(by_name) == {"tss", "exons", "utr3", "utr5"}
    assert len(by_name["tss"]) == 1
    assert ((by_name["tss"]["end"] - by_name["tss"]["start"]) == 1).all()
    assert len(by_name["exons"]) == 2
    assert len(by_name["utr5"]) == 1
    assert len(by_name["utr3"]) == 1
    assert installed["tss"] == 1
    assert installed["utr5"] == 1
    assert installed["utr3"] == 1


def test_install_genes_recognizes_ucsc_gtf_5utr_3utr(monkeypatch):
    """UCSC ncbiRefSeq.gtf.gz uses 5UTR / 3UTR (no underscore, no five_prime_*).

    Pre-fix: utr3 and utr5 sets came out empty.
    """
    calls = _capture(monkeypatch)
    installed = _install_genes(UCSC_GTF, lambda c: "M" if c == "chrM" else None)
    by_name = {name: df for df, name in calls}
    assert set(by_name) == {"tss", "exons", "utr3", "utr5"}
    assert len(by_name["tss"]) == 1
    assert len(by_name["exons"]) == 2
    assert len(by_name["utr5"]) == 1
    assert len(by_name["utr3"]) == 1
    assert installed["utr5"] == 1
    assert installed["utr3"] == 1
