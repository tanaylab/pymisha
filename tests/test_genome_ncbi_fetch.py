"""Tests for `pymisha.genome._ncbi._ncbi_fetch_assets` (R 5.6.30 parity).

All tests mock `_open_url` / `_ncbi_dataset_report` / `_ncbi_post_download`.
No network access.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest

from pymisha.genome._ncbi import (
    _extract_gff_from_zip,
    _ncbi_fetch_assets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACC = "GCF_000001635.27"
ASM = "GRCm39"

# Two assembled-molecule rows so _ncbi_sequence_report returns a non-empty df.
SEQREP_ROWS = [
    {
        "refseqAccession": "NC_000067.7",
        "genbankAccession": "CM000994.3",
        "chrName": "1",
        "role": "assembled-molecule",
        "length": 195154279,
    },
    {
        "refseqAccession": "NC_000086.8",
        "genbankAccession": "CM001013.3",
        "chrName": "X",
        "role": "assembled-molecule",
        "length": 169476592,
    },
]

GFF_BODY = (
    b"##gff-version 3\n"
    b"chr1\tRefSeq\texon\t1\t100\t.\t+\t.\tID=exon1\n"
)


def _make_zip(*, jsonl: list[dict] | None, gff: bytes | None = None,
              gff_gz: bool = False) -> bytes:
    """Build an in-memory NCBI-style zip.

    Parameters
    ----------
    jsonl : list of dict or None
        Rows for ``ncbi_dataset/data/<acc>/sequence_report.jsonl``. None -> omit.
    gff : bytes or None
        GFF body to include. None -> omit.
    gff_gz : bool
        Store the GFF as ``.gff.gz`` rather than plain ``.gff``.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ncbi_dataset/data/README.md", "hi")
        if jsonl is not None:
            body = "\n".join(json.dumps(r) for r in jsonl) + "\n"
            zf.writestr(
                f"ncbi_dataset/data/{ACC}/sequence_report.jsonl", body
            )
        if gff is not None:
            if gff_gz:
                zf.writestr(
                    f"ncbi_dataset/data/{ACC}/genomic.gff.gz",
                    gzip.compress(gff),
                )
            else:
                zf.writestr(
                    f"ncbi_dataset/data/{ACC}/genomic.gff", gff
                )
    return buf.getvalue()


def _fake_http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "fake", hdrs=None, fp=None)


# ---------------------------------------------------------------------------
# Datasets fast-path: include list
# ---------------------------------------------------------------------------


def test_ncbi_fetch_assets_install_path_only_includes_sequence_report_and_gff():
    """When 'genes' is requested, include is [SEQUENCE_REPORT, GENOME_GFF]."""
    captured = {}

    def _fake_post(acc, include):
        captured["include"] = list(include)
        return _make_zip(jsonl=SEQREP_ROWS, gff=GFF_BODY)

    with patch("pymisha.genome._ncbi._ncbi_post_download", side_effect=_fake_post):
        # No genes/rmsk requested -> no dataset_report call needed.
        out = _ncbi_fetch_assets(
            {"accession": ACC}, ("genes",),
        )

    assert captured["include"] == ["SEQUENCE_REPORT", "GENOME_GFF"]
    assert "GENOME_FASTA" not in captured["include"]
    assert out["genes"] == GFF_BODY
    assert out["genes_source"] == "RefSeq"


def test_ncbi_fetch_assets_install_path_skips_genome_fasta():
    """Without 'genes' in sets, include is just [SEQUENCE_REPORT]."""
    captured = {}

    def _fake_post(acc, include):
        captured["include"] = list(include)
        return _make_zip(jsonl=SEQREP_ROWS)

    with patch("pymisha.genome._ncbi._ncbi_post_download", side_effect=_fake_post):
        out = _ncbi_fetch_assets({"accession": ACC}, ())

    assert captured["include"] == ["SEQUENCE_REPORT"]
    assert "GENOME_FASTA" not in captured["include"]
    assert out["genes"] is None


def test_ncbi_fetch_assets_returns_chrom_alias_df():
    """SEQUENCE_REPORT -> 6-column chromAlias DataFrame."""
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS)
    with patch("pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes):
        out = _ncbi_fetch_assets({"accession": ACC}, ())
    df = out["chrom_alias"]
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "refseq", "genbank", "sequence_name", "chr_name", "ucsc", "length",
    ]
    assert len(df) == 2


# ---------------------------------------------------------------------------
# FTP fallback for GFF (R 5.6.30 d6cd6047)
# ---------------------------------------------------------------------------


def test_ncbi_fetch_assets_falls_back_to_ftp_on_empty_gff_in_zip():
    """Datasets zip has sequence_report but no GFF -> FTP fallback."""
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS, gff=None)
    report = {
        "reports": [
            {
                "assembly_info": {"assembly_name": ASM},
                "annotation_info": {"provider": "NCBI RefSeq"},
            }
        ]
    }
    ftp_body = b"FTP-GFF-BYTES"

    with patch(
        "pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes
    ), patch(
        "pymisha.genome._ncbi._ncbi_dataset_report", return_value=report
    ), patch(
        "pymisha.genome._ncbi._open_url", return_value=ftp_body
    ) as m_open:
        out = _ncbi_fetch_assets({"accession": ACC}, ("genes",))

    assert out["genes"] == ftp_body
    assert out["genes_source"] == "RefSeq-FTP"
    # FTP URL was the genomic.gff.gz path for the assembly.
    ftp_url = m_open.call_args[0][0]
    assert ftp_url.endswith(f"{ACC}_{ASM}_genomic.gff.gz")
    assert ftp_url.startswith("https://ftp.ncbi.nlm.nih.gov/genomes/all/")


def test_ncbi_fetch_assets_warns_on_genes_no_ftp_no_asm_name():
    """Empty zip + dataset_report fails -> UserWarning, genes None."""
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS, gff=None)

    def _boom(acc):
        raise RuntimeError("nope")

    with patch(
        "pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes
    ), patch(
        "pymisha.genome._ncbi._ncbi_dataset_report", side_effect=_boom
    ), pytest.warns(UserWarning, match="no assembly_name"):
        out = _ncbi_fetch_assets({"accession": ACC}, ("genes",))

    assert out["genes"] is None
    assert out["genes_source"] is None


# ---------------------------------------------------------------------------
# rmsk: FTP only
# ---------------------------------------------------------------------------


def test_ncbi_fetch_assets_rmsk_from_ftp():
    """rmsk fetched from FTP rm.out.gz."""
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS)
    report = {
        "reports": [
            {
                "assembly_info": {"assembly_name": ASM},
                "annotation_info": {"provider": "NCBI RefSeq"},
            }
        ]
    }
    rmsk_body = b"FTP-RMSK-BYTES"

    with patch(
        "pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes
    ), patch(
        "pymisha.genome._ncbi._ncbi_dataset_report", return_value=report
    ), patch(
        "pymisha.genome._ncbi._open_url", return_value=rmsk_body
    ) as m_open:
        out = _ncbi_fetch_assets({"accession": ACC}, ("rmsk",))

    assert out["rmsk"] == rmsk_body
    ftp_url = m_open.call_args[0][0]
    assert ftp_url.endswith(f"{ACC}_{ASM}_rm.out.gz")


def test_ncbi_fetch_assets_rmsk_warns_on_404():
    """FTP 404 -> warn, rmsk None."""
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS)
    report = {
        "reports": [
            {
                "assembly_info": {"assembly_name": ASM},
                "annotation_info": {"provider": "NCBI RefSeq"},
            }
        ]
    }

    def _open_404(url, **kw):
        raise _fake_http_error(url, 404)

    with patch(
        "pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes
    ), patch(
        "pymisha.genome._ncbi._ncbi_dataset_report", return_value=report
    ), patch(
        "pymisha.genome._ncbi._open_url", side_effect=_open_404
    ), pytest.warns(UserWarning, match="rm.out.gz"):
        out = _ncbi_fetch_assets({"accession": ACC}, ("rmsk",))

    assert out["rmsk"] is None


# ---------------------------------------------------------------------------
# cgi / cytoband: not available from NCBI
# ---------------------------------------------------------------------------


def test_ncbi_fetch_assets_cgi_warns_always():
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS)
    with patch(
        "pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes,
    ), pytest.warns(UserWarning, match="cgi.*not available"):
        out = _ncbi_fetch_assets({"accession": ACC}, ("cgi",))
    assert out["cgi"] is None


def test_ncbi_fetch_assets_cytoband_warns_always():
    zip_bytes = _make_zip(jsonl=SEQREP_ROWS)
    with patch(
        "pymisha.genome._ncbi._ncbi_post_download", return_value=zip_bytes,
    ), pytest.warns(UserWarning, match="cytoband.*not available"):
        out = _ncbi_fetch_assets({"accession": ACC}, ("cytoband",))
    assert out["cytoband"] is None


# ---------------------------------------------------------------------------
# _extract_gff_from_zip helper
# ---------------------------------------------------------------------------


def test_ncbi_fetch_assets_extract_gff_finds_gz():
    zip_bytes = _make_zip(jsonl=None, gff=GFF_BODY, gff_gz=True)
    assert _extract_gff_from_zip(zip_bytes) == GFF_BODY


def test_ncbi_fetch_assets_extract_gff_finds_plain():
    zip_bytes = _make_zip(jsonl=None, gff=GFF_BODY, gff_gz=False)
    assert _extract_gff_from_zip(zip_bytes) == GFF_BODY


def test_ncbi_fetch_assets_extract_gff_returns_none_when_absent():
    zip_bytes = _make_zip(jsonl=None, gff=None)
    assert _extract_gff_from_zip(zip_bytes) is None
