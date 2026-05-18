"""Regression tests: NCBI 'genes' falls back to FTP when Datasets zip has no GFF.

Ports R commit d6cd6047 (fix(install-intervals): FTP fallback for NCBI 'genes'
when Datasets zip empty).

The R fix addressed two bugs:
1. Preflight dropped 'genes' from sets when /dataset_report returned {} even
   though assembly might still have a GFF on FTP.
2. FTP fallback for GFF was missing; only rmsk had an FTP path.

pymisha never had bug (1) - it never had a preflight that dropped 'genes'.
Bug (2) was already fixed in pymisha's _ncbi_fetch_assets (lines 285-302).

However, the R fix also added a third fallback: when /dataset_report returns
{} (suppressed accession, assembly_name = ""), resolve assembly_name from the
parent FTP directory listing. pymisha currently lacks this third level of
fallback. The test below documents the gap and will fail until the FTP listing
fallback is added.

Accession used: GCF_000001635.26 (GRCm38.p6, replaced/suppressed in Datasets).
"""
from __future__ import annotations

import gzip
import io
import json
import urllib.error
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from pymisha.genome._ncbi import _ncbi_fetch_assets

ACC = "GCF_000001635.26"
ASM = "GRCm38.p6"

# Expected FTP base dir for GCF_000001635.26 / GRCm38.p6
_FTP_DIR = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
    f"{ACC}_{ASM}"
)
_GFF_FTP_URL = f"{_FTP_DIR}/{ACC}_{ASM}_genomic.gff.gz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip_no_gff() -> bytes:
    """Minimal Datasets zip with sequence_report but no GFF entry."""
    buf = io.BytesIO()
    rows = [
        {
            "refseqAccession": "NC_000001.10",
            "genbankAccession": "CM000663.1",
            "chrName": "1",
            "role": "assembled-molecule",
            "length": 195471971,
        }
    ]
    with zipfile.ZipFile(buf, "w") as zf:
        body = "\n".join(json.dumps(r) for r in rows) + "\n"
        zf.writestr(
            f"ncbi_dataset/data/{ACC}/sequence_report.jsonl", body
        )
    return buf.getvalue()


def _make_fake_gff_gz() -> bytes:
    """Minimal gzipped GFF bytes to simulate FTP payload."""
    content = b"##gff-version 3\n1\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=gene-1\n"
    return gzip.compress(content)


def _make_report_with_asm_name(asm_name: str = ASM, has_annotation: bool = False) -> bytes:
    """Serialize a dataset_report with assembly_name set, annotation_info empty."""
    payload: dict = {
        "reports": [
            {
                "assembly_info": {"assembly_name": asm_name},
            }
        ]
    }
    if has_annotation:
        payload["reports"][0]["annotation_info"] = {"provider": "NCBI RefSeq"}
    return json.dumps(payload).encode("utf-8")


def _make_empty_report() -> bytes:
    """Serialize a suppressed /dataset_report that returns no reports."""
    return json.dumps({}).encode("utf-8")


# ---------------------------------------------------------------------------
# Test 1: assembly_name in dataset_report, zip has no GFF -> FTP fallback
# ---------------------------------------------------------------------------


def test_genes_ftp_fallback_when_zip_empty_but_asm_name_known():
    """When Datasets zip has no GFF but assembly_name is in /dataset_report,
    _ncbi_fetch_assets must fetch the GFF from FTP and set genes_source to
    'RefSeq-FTP'.

    This is the core fix from R d6cd6047 part-2. pymisha already implements
    this path (lines 285-302 of _ncbi.py).
    """
    fake_zip = _make_zip_no_gff()
    fake_report = _make_report_with_asm_name(ASM, has_annotation=False)
    fake_gff_gz = _make_fake_gff_gz()

    def _fake_open_url(url: str) -> bytes:
        if "dataset_report" in url:
            return fake_report
        if url.endswith(".zip") or "download" in url:
            return fake_zip
        if url == _GFF_FTP_URL:
            return fake_gff_gz
        raise AssertionError(f"Unexpected URL: {url}")

    recipe = {"accession": ACC}
    with patch("pymisha.genome._ncbi._open_url", side_effect=_fake_open_url):
        result = _ncbi_fetch_assets(recipe, ("genes",))

    assert result["genes"] is not None, (
        "genes should be populated via FTP fallback when zip has no GFF "
        "but assembly_name is known from /dataset_report"
    )
    assert result["genes_source"] == "RefSeq-FTP", (
        f"genes_source should be 'RefSeq-FTP', got {result['genes_source']!r}"
    )
    # The returned bytes are gzipped (left for downstream to decompress).
    assert result["genes"] == fake_gff_gz


# ---------------------------------------------------------------------------
# Test 2: suppressed dataset_report ({}), assembly_name NOT in report,
#          but assembly_name IS resolvable from FTP directory listing.
#          R d6cd6047 added this path; pymisha currently lacks it.
# ---------------------------------------------------------------------------


def test_genes_ftp_fallback_when_report_empty_and_asm_name_from_ftp_listing():
    """When /dataset_report returns {} (suppressed accession) and the Datasets
    zip has no GFF, pymisha should fall back to resolving assembly_name from
    the parent FTP directory listing, then fetch the GFF via FTP.

    This is the R d6cd6047 third-level fallback. pymisha does NOT implement
    it yet - when assembly_name cannot be resolved from the report, it warns
    and skips. This test documents the gap and is expected to FAIL until
    the FTP listing fallback is added to _ncbi_fetch_assets.

    FTP listing URL pattern:
      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/
    The listing response contains one line per subdir, e.g.:
      GCF_000001635.26_GRCm38.p6/
    from which assembly_name = 'GRCm38.p6' can be extracted.
    """
    fake_zip = _make_zip_no_gff()
    fake_empty_report = _make_empty_report()
    fake_gff_gz = _make_fake_gff_gz()

    # FTP parent listing: one subdirectory per accession version.
    # The listing is the raw FTP index page; the directory name encodes asm.
    ftp_parent = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/"
    )
    ftp_listing_body = (
        f"GCF_000001635.26_{ASM}/\n"
        "GCF_000001635.27_GRCm39/\n"
    ).encode()

    def _fake_open_url(url: str) -> bytes:
        if "dataset_report" in url:
            return fake_empty_report
        if url.endswith(".zip") or "download" in url:
            return fake_zip
        if url == ftp_parent:
            return ftp_listing_body
        if url == _GFF_FTP_URL:
            return fake_gff_gz
        raise AssertionError(f"Unexpected URL: {url}")

    recipe = {"accession": ACC}
    with patch("pymisha.genome._ncbi._open_url", side_effect=_fake_open_url):
        result = _ncbi_fetch_assets(recipe, ("genes",))

    assert result["genes"] is not None, (
        "genes should be populated via FTP fallback when /dataset_report "
        "is suppressed ({}) but assembly_name is resolvable from FTP listing. "
        "pymisha lacks the FTP listing fallback added in R d6cd6047."
    )
    assert result["genes_source"] == "RefSeq-FTP"
    assert result["genes"] == fake_gff_gz


# ---------------------------------------------------------------------------
# Test 3: suppressed dataset_report + FTP listing also unavailable -> warn+skip
# ---------------------------------------------------------------------------


def test_genes_warn_and_skip_when_report_empty_and_no_ftp_listing():
    """When /dataset_report is suppressed ({}) AND FTP listing is unavailable,
    'genes' should be skipped with a warning (not raise).
    """
    fake_zip = _make_zip_no_gff()
    fake_empty_report = _make_empty_report()

    ftp_404 = urllib.error.HTTPError(
        url="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/635/",
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=None,
    )

    def _fake_open_url(url: str) -> bytes:
        if "dataset_report" in url:
            return fake_empty_report
        if url.endswith(".zip") or "download" in url:
            return fake_zip
        # FTP parent listing and any FTP GFF fetch -> 404
        raise ftp_404

    recipe = {"accession": ACC}
    with (
        patch("pymisha.genome._ncbi._open_url", side_effect=_fake_open_url),
        pytest.warns(UserWarning),
    ):
        result = _ncbi_fetch_assets(recipe, ("genes",))

    # genes should be skipped (None) rather than raising.
    assert result["genes"] is None


# ---------------------------------------------------------------------------
# Test 4: FTP 404 on GFF when assembly_name IS known -> warn + skip
# ---------------------------------------------------------------------------


def test_genes_ftp_404_warns_and_skips():
    """When Datasets zip has no GFF AND the FTP GFF URL returns 404,
    _ncbi_fetch_assets should warn and set genes=None (not raise).
    """
    fake_zip = _make_zip_no_gff()
    fake_report = _make_report_with_asm_name(ASM, has_annotation=False)

    ftp_404 = urllib.error.HTTPError(
        url=_GFF_FTP_URL,
        code=404,
        msg="Not Found",
        hdrs=MagicMock(),
        fp=None,
    )

    def _fake_open_url(url: str) -> bytes:
        if "dataset_report" in url:
            return fake_report
        if url.endswith(".zip") or "download" in url:
            return fake_zip
        if url == _GFF_FTP_URL:
            raise ftp_404
        raise AssertionError(f"Unexpected URL: {url}")

    recipe = {"accession": ACC}
    with (
        patch("pymisha.genome._ncbi._open_url", side_effect=_fake_open_url),
        pytest.warns(UserWarning, match="FTP 404"),
    ):
        result = _ncbi_fetch_assets(recipe, ("genes",))

    assert result["genes"] is None
    assert result["genes_source"] is None
