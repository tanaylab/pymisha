"""Tests for pymisha.genome._ucsc (URL helpers + _ucsc_fetch_assets).

All tests use unittest.mock.patch to stub network calls. No real network.
"""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pandas as pd
import pytest

from pymisha.genome import _ucsc
from pymisha.genome._ucsc import (
    _fetch_gtf_with_priority,
    _parse_chrom_alias_tsv,
    _ucsc_chrom_alias_url,
    _ucsc_fasta_url,
    _ucsc_fetch_assets,
    _ucsc_gtf_url,
)

# ---------------------------------------------------------------------------
# URL formatters
# ---------------------------------------------------------------------------


def test_ucsc_fasta_url_format():
    assert _ucsc_fasta_url("hg38") == (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
    )


def test_ucsc_chrom_alias_url_format():
    assert _ucsc_chrom_alias_url("hg38") == (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chromAlias.txt"
    )


def test_ucsc_gtf_url_format():
    assert _ucsc_gtf_url("hg38", "ncbiRefSeq") == (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/ncbiRefSeq.gtf.gz"
    )


# ---------------------------------------------------------------------------
# chromAlias parser
# ---------------------------------------------------------------------------


def test_parse_chrom_alias_tsv_pivot():
    """Long-format (3-col, no header) input -> pivoted wide DataFrame."""
    text = "1\tchr1\tensembl\nCM000663.2\tchr1\tgenbank\n2\tchr2\tensembl\n"
    df = _parse_chrom_alias_tsv(text)
    assert set(df.columns) == {"chrom", "ensembl", "genbank"}
    chr1_row = df[df["chrom"] == "chr1"].iloc[0]
    assert chr1_row["ensembl"] == "1"
    assert chr1_row["genbank"] == "CM000663.2"
    chr2_row = df[df["chrom"] == "chr2"].iloc[0]
    assert chr2_row["ensembl"] == "2"
    # chr2 has no genbank alias -> NaN
    assert pd.isna(chr2_row["genbank"])


def test_parse_chrom_alias_tsv_multiple_aliases_per_source():
    """If the same (chrom, source) appears twice, the first alias wins."""
    text = "1\tchr1\tensembl\n01\tchr1\tensembl\n"
    df = _parse_chrom_alias_tsv(text)
    assert len(df) == 1
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["ensembl"] == "1"


def test_parse_chrom_alias_tsv_wide_format_with_hash_header():
    """Wide format (bigZips/<asm>.chromAlias.txt) parsed via leading-# detection."""
    text = (
        "# sequenceName\talias names\tUCSC database: hg38\n"
        "chr1\t1\tCM000663.2\n"
        "chr2\t2\tCM000664.2\n"
    )
    df = _parse_chrom_alias_tsv(text)
    assert list(df.columns) == ["chrom", "alias_0", "alias_1"]
    assert df.iloc[0]["chrom"] == "chr1"
    assert df.iloc[0]["alias_0"] == "1"
    assert df.iloc[0]["alias_1"] == "CM000663.2"


# ---------------------------------------------------------------------------
# GTF priority fetcher
# ---------------------------------------------------------------------------


def test_fetch_gtf_with_priority_first_hit_wins():
    """If the first priority returns bytes, lower priorities are never queried."""
    seen_urls: list[str] = []

    def fake_open(url, **kw):
        seen_urls.append(url)
        return b"GTFDATA"

    with patch.object(_ucsc, "_open_url", side_effect=fake_open):
        raw, source = _fetch_gtf_with_priority("hg38", ("ncbiRefSeq", "refGene"))

    assert raw == b"GTFDATA"
    assert source == "ncbiRefSeq"
    assert len(seen_urls) == 1
    assert "ncbiRefSeq.gtf.gz" in seen_urls[0]


def test_fetch_gtf_with_priority_falls_through_on_404():
    """First priority raises URLError; second succeeds and is reported."""
    side_effects = [urllib.error.URLError("404"), b"BACKUP"]
    with patch.object(_ucsc, "_open_url", side_effect=side_effects):
        raw, source = _fetch_gtf_with_priority("hg38", ("ncbiRefSeq", "refGene"))
    assert raw == b"BACKUP"
    assert source == "refGene"


def test_fetch_gtf_with_priority_all_fail_raises():
    """If every priority fails, FileNotFoundError is raised."""
    err = urllib.error.URLError("nope")
    with (
        patch.object(_ucsc, "_open_url", side_effect=err),
        pytest.raises(FileNotFoundError, match="No GTF found"),
    ):
        _fetch_gtf_with_priority("zz1", ("ncbiRefSeq", "refGene"))


# ---------------------------------------------------------------------------
# _ucsc_fetch_assets
# ---------------------------------------------------------------------------


_ALIAS_LONG_TEXT = "1\tchr1\tensembl\nCM000663.2\tchr1\tgenbank\n"


def test_ucsc_fetch_assets_returns_full_shape():
    """All four sets requested -> dict has 6 keys, all populated."""

    def fake_open(url, **kw):
        # Map URL -> payload by suffix.
        if url.endswith(".gtf.gz"):
            return b"GTF"
        if url.endswith(".fa.out.gz"):
            return b"RMSK"
        if "cpgIslandExt" in url:
            return b"CGI"
        if "cytoBandIdeo" in url:
            return b"CYTO"
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_ucsc, "_open_url", side_effect=fake_open), \
            patch.object(_ucsc, "_read_url_text", return_value=_ALIAS_LONG_TEXT):
        out = _ucsc_fetch_assets(
            {"assembly": "hg38"},
            sets=("genes", "rmsk", "cgi", "cytoband"),
        )

    assert set(out.keys()) == {
        "chrom_alias", "genes", "genes_source", "rmsk", "cgi", "cytoband"
    }
    assert isinstance(out["chrom_alias"], pd.DataFrame)
    assert out["genes"] == b"GTF"
    assert out["genes_source"] == "ncbiRefSeq"  # first in default priority
    assert out["rmsk"] == b"RMSK"
    assert out["cgi"] == b"CGI"
    assert out["cytoband"] == b"CYTO"


def test_ucsc_fetch_assets_chrom_alias_missing_is_ok():
    """If the chromAlias URL 404s, chrom_alias is None and the rest still load."""

    def fake_open(url, **kw):
        if url.endswith(".gtf.gz"):
            return b"GTF"
        if url.endswith(".fa.out.gz"):
            return b"RMSK"
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_ucsc, "_open_url", side_effect=fake_open), \
            patch.object(
                _ucsc, "_read_url_text",
                side_effect=urllib.error.URLError("alias 404"),
            ):
        out = _ucsc_fetch_assets(
            {"assembly": "hg38"},
            sets=("genes", "rmsk"),
        )

    assert out["chrom_alias"] is None
    assert out["genes"] == b"GTF"
    assert out["rmsk"] == b"RMSK"
    assert out["cgi"] is None
    assert out["cytoband"] is None


def test_ucsc_fetch_assets_only_requested_sets():
    """Pass sets=('cgi',) -> only cgi (plus chrom_alias) is populated."""
    open_calls: list[str] = []

    def fake_open(url, **kw):
        open_calls.append(url)
        if "cpgIslandExt" in url:
            return b"CGI"
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_ucsc, "_open_url", side_effect=fake_open), \
            patch.object(_ucsc, "_read_url_text", return_value=_ALIAS_LONG_TEXT):
        out = _ucsc_fetch_assets({"assembly": "hg38"}, sets=("cgi",))

    assert out["genes"] is None
    assert out["genes_source"] is None
    assert out["rmsk"] is None
    assert out["cytoband"] is None
    assert out["cgi"] == b"CGI"
    assert isinstance(out["chrom_alias"], pd.DataFrame)
    # Only one network call against _open_url (the cgi one); no GTF/rmsk/cyto.
    assert len(open_calls) == 1
    assert "cpgIslandExt" in open_calls[0]
