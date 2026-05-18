"""Tests for pymisha.genome._ncbi (NCBI Datasets API + FTP fallback helpers).

All tests mock _open_url. No real network.
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import patch

import pytest

from pymisha.genome._ncbi import (
    NCBI_INCLUDE_VALUES,
    _datasets_report_url,
    _datasets_zip_url,
    _ncbi_assembly_name_from_ftp_listing,
    _ncbi_assembly_name_from_report,
    _ncbi_dataset_report,
    _ncbi_extract_sequence_report,
    _ncbi_ftp_assembly_dir,
    _ncbi_ftp_parent_dir,
    _ncbi_has_annotation,
    _ncbi_post_download,
    _ncbi_sequence_report,
    _validate_accession,
)

# ---------------------------------------------------------------------------
# Accession validation
# ---------------------------------------------------------------------------


def test_validate_accession_accepts_gcf_and_gca():
    _validate_accession("GCF_000001635.27")
    _validate_accession("GCA_009914755.4")


def test_validate_accession_rejects_short_digits():
    with pytest.raises(ValueError, match="Invalid NCBI accession"):
        _validate_accession("GCF_12345.1")


def test_validate_accession_rejects_missing_version():
    with pytest.raises(ValueError, match="Invalid NCBI accession"):
        _validate_accession("GCF_000001635")


# ---------------------------------------------------------------------------
# URL formatters
# ---------------------------------------------------------------------------


def test_datasets_zip_url_format():
    url = _datasets_zip_url("GCF_000001635.27", ["SEQUENCE_REPORT", "GENOME_GFF"])
    assert url == (
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        "GCF_000001635.27/download?include_annotation_type=SEQUENCE_REPORT,GENOME_GFF"
    )


def test_datasets_zip_url_rejects_unknown_include():
    with pytest.raises(ValueError, match="Invalid include values"):
        _datasets_zip_url("GCF_000001635.27", ["BOGUS"])


def test_datasets_zip_url_rejects_empty_include():
    with pytest.raises(ValueError, match="include must be non-empty"):
        _datasets_zip_url("GCF_000001635.27", [])


def test_datasets_report_url_format():
    url = _datasets_report_url("GCF_000001635.27")
    assert url == (
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        "GCF_000001635.27/dataset_report"
    )


# ---------------------------------------------------------------------------
# Network entrypoints (mocked)
# ---------------------------------------------------------------------------


def test_ncbi_post_download_returns_bytes():
    fake_zip = b"\x50\x4b\x03\x04" + b"...zip..."
    with patch("pymisha.genome._ncbi._open_url", return_value=fake_zip) as m:
        out = _ncbi_post_download("GCF_000001635.27", ["SEQUENCE_REPORT"])
    assert out == fake_zip
    # Sanity: the underlying URL is the Datasets zip URL.
    called_url = m.call_args[0][0]
    assert called_url.startswith(
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_000001635.27/download"
    )


def test_ncbi_dataset_report_parses_json():
    payload = {
        "reports": [
            {
                "assembly_info": {"assembly_name": "GRCm39"},
                "annotation_info": {"provider": "NCBI RefSeq"},
            }
        ]
    }
    with patch(
        "pymisha.genome._ncbi._open_url",
        return_value=json.dumps(payload).encode("utf-8"),
    ):
        out = _ncbi_dataset_report("GCF_000001635.27")
    assert out == payload


# ---------------------------------------------------------------------------
# Zip extraction
# ---------------------------------------------------------------------------


def _make_zip(jsonl_lines: list[dict] | None) -> bytes:
    """Build an in-memory NCBI-style zip; jsonl_lines=None means no sequence_report."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ncbi_dataset/data/README.md", "hello")
        if jsonl_lines is not None:
            body = "\n".join(json.dumps(r) for r in jsonl_lines) + "\n"
            zf.writestr(
                "ncbi_dataset/data/GCF_000001635.27/sequence_report.jsonl", body
            )
    return buf.getvalue()


def test_ncbi_extract_sequence_report_finds_jsonl():
    rows_in = [
        {"refseqAccession": "NC_000001.11", "chrName": "1", "length": 100},
        {"refseqAccession": "NC_000023.11", "chrName": "X", "length": 50},
    ]
    zb = _make_zip(rows_in)
    out = _ncbi_extract_sequence_report(zb)
    assert out == rows_in


def test_ncbi_extract_sequence_report_empty_when_absent():
    zb = _make_zip(None)
    assert _ncbi_extract_sequence_report(zb) == []


def test_ncbi_sequence_report_full_shape():
    rows_in = [
        {
            "refseqAccession": "NC_000001.11",
            "genbankAccession": "CM000663.2",
            "chrName": "1",
            "role": "assembled-molecule",
            "length": 100,
        },
        {
            "refseqAccession": "NC_000023.11",
            "genbankAccession": "CM000685.2",
            "chrName": "X",
            "role": "assembled-molecule",
            "length": 80,
        },
        {
            "refseqAccession": "",
            "genbankAccession": "KQ123.1",
            "chrName": "",
            "role": "unplaced-scaffold",
            "length": 50,
        },
    ]
    zb = _make_zip(rows_in)
    df = _ncbi_sequence_report(zb)
    assert list(df.columns) == [
        "refseq", "genbank", "sequence_name", "chr_name", "ucsc", "length"
    ]
    assert len(df) == 3
    chr1 = df.iloc[0]
    assert chr1["refseq"] == "NC_000001.11"
    assert chr1["genbank"] == "CM000663.2"
    assert chr1["sequence_name"] == "1"
    assert chr1["chr_name"] == "1"
    assert chr1["ucsc"] == "chr1"
    assert int(chr1["length"]) == 100

    chrX = df.iloc[1]
    assert chrX["ucsc"] == "chrX"

    scaffold = df.iloc[2]
    assert scaffold["refseq"] == ""
    assert scaffold["genbank"] == "KQ123.1"
    assert scaffold["sequence_name"] == ""
    assert scaffold["ucsc"] == "chrUn_KQ123v1"
    assert int(scaffold["length"]) == 50


def test_ncbi_sequence_report_empty_zip_returns_empty_df_with_columns():
    zb = _make_zip(None)
    df = _ncbi_sequence_report(zb)
    assert len(df) == 0
    assert list(df.columns) == [
        "refseq", "genbank", "sequence_name", "chr_name", "ucsc", "length"
    ]


# ---------------------------------------------------------------------------
# FTP fallback
# ---------------------------------------------------------------------------


def test_ncbi_ftp_assembly_dir_format():
    assert _ncbi_ftp_assembly_dir("GCF_000001635.27", "GRCm39") == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
        "GCF_000001635.27_GRCm39"
    )


def test_ncbi_ftp_assembly_dir_rejects_empty_asm():
    with pytest.raises(ValueError, match="assembly_name must be non-empty"):
        _ncbi_ftp_assembly_dir("GCF_000001635.27", "")


def test_ncbi_ftp_parent_dir_format():
    assert _ncbi_ftp_parent_dir("GCF_000001635.26") == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
    )
    assert _ncbi_ftp_parent_dir("GCA_009914755.4") == (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/009/914/755/"
    )


def test_ncbi_assembly_name_from_ftp_listing_finds_accession():
    listing = (
        b"GCF_000001635.26_GRCm38.p6/\n"
        b"GCF_000001635.27_GRCm39/\n"
    )
    assert _ncbi_assembly_name_from_ftp_listing("GCF_000001635.26", listing) == "GRCm38.p6"
    assert _ncbi_assembly_name_from_ftp_listing("GCF_000001635.27", listing) == "GRCm39"


def test_ncbi_assembly_name_from_ftp_listing_returns_empty_when_not_found():
    listing = b"GCF_000001635.27_GRCm39/\n"
    assert _ncbi_assembly_name_from_ftp_listing("GCF_000001635.26", listing) == ""


def test_ncbi_assembly_name_from_ftp_listing_handles_trailing_slash():
    # FTP listings may or may not have trailing slashes; both should work.
    listing_no_slash = b"GCF_000001635.26_GRCm38.p6\n"
    assert (
        _ncbi_assembly_name_from_ftp_listing("GCF_000001635.26", listing_no_slash)
        == "GRCm38.p6"
    )


# ---------------------------------------------------------------------------
# Report accessors
# ---------------------------------------------------------------------------


def test_ncbi_assembly_name_from_report():
    report = {
        "reports": [
            {"assembly_info": {"assembly_name": "GRCm39"}}
        ]
    }
    assert _ncbi_assembly_name_from_report(report) == "GRCm39"
    assert _ncbi_assembly_name_from_report({}) == ""
    assert _ncbi_assembly_name_from_report({"reports": [{}]}) == ""


def test_ncbi_has_annotation_true_when_provider_set():
    report = {
        "reports": [
            {"annotation_info": {"provider": "NCBI RefSeq"}}
        ]
    }
    assert _ncbi_has_annotation(report) is True


def test_ncbi_has_annotation_false_when_empty():
    assert _ncbi_has_annotation({}) is False
    assert _ncbi_has_annotation({"reports": [{}]}) is False
    assert _ncbi_has_annotation({"reports": [{"annotation_info": {"provider": ""}}]}) is False


# ---------------------------------------------------------------------------
# Misc sanity
# ---------------------------------------------------------------------------


def test_include_values_constant_has_expected_entries():
    assert "SEQUENCE_REPORT" in NCBI_INCLUDE_VALUES
    assert "GENOME_FASTA" in NCBI_INCLUDE_VALUES
    assert "GENOME_GFF" in NCBI_INCLUDE_VALUES
