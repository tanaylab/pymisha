"""Tests for the streaming GTF parser (pymisha.genome._gtf)."""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from pymisha.genome._gtf import (
    _gtf_to_dataframe,
    _iter_gtf_rows,
    _parse_attributes,
)

FIXTURE = Path(__file__).resolve().parent / "genome_fixtures" / "sample.gtf"


def _fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()


def test_parse_attributes_single_pair():
    assert _parse_attributes('gene_id "ABC"') == {"gene_id": "ABC"}


def test_parse_attributes_multiple():
    s = 'gene_id "ABC"; transcript_id "T1";'
    assert _parse_attributes(s) == {"gene_id": "ABC", "transcript_id": "T1"}


def test_parse_attributes_first_occurrence_wins():
    s = 'gene_id "ABC"; gene_id "XYZ";'
    assert _parse_attributes(s) == {"gene_id": "ABC"}


def test_iter_gtf_rows_skips_comments_and_blank():
    raw = (
        b"# this is a comment\n"
        b"\n"
        b'1\ttest\texon\t100\t200\t.\t+\t.\tgene_id "G";\n'
    )
    rows = list(_iter_gtf_rows(raw))
    assert len(rows) == 1
    assert rows[0]["chrom"] == "1"
    assert rows[0]["feature"] == "exon"


def test_iter_gtf_rows_converts_to_half_open():
    raw = b'1\ttest\texon\t100\t200\t.\t+\t.\tgene_id "G";\n'
    rows = list(_iter_gtf_rows(raw))
    assert len(rows) == 1
    assert rows[0]["start"] == 99
    assert rows[0]["end"] == 200


def test_iter_gtf_rows_skips_short_lines():
    raw = (
        b"1\ttest\texon\t100\t200\n"  # only 5 cols
        b'1\ttest\texon\t100\t200\t.\t+\t.\tgene_id "G";\n'
    )
    rows = list(_iter_gtf_rows(raw))
    assert len(rows) == 1


def test_iter_gtf_rows_handles_gzip():
    raw = _fixture_bytes()
    gz = gzip.compress(raw)
    rows = list(_iter_gtf_rows(gz))
    rows_plain = list(_iter_gtf_rows(raw))
    assert len(rows) == len(rows_plain) > 0
    assert rows[0] == rows_plain[0]


def test_gtf_to_dataframe_filters_features():
    raw = _fixture_bytes()
    df = _gtf_to_dataframe(raw, feature_filter=("exon",))
    assert isinstance(df, pd.DataFrame)
    assert set(df["feature"].unique()) == {"exon"}
    # 6 exons in the fixture (3 per transcript x 2 transcripts)
    assert len(df) == 6


def test_gtf_to_dataframe_empty_returns_empty_dataframe_with_columns():
    df = _gtf_to_dataframe(b"", feature_filter=("exon",))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "strand",
        "attributes",
    ]
