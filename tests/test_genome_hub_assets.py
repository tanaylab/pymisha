"""Tests for pymisha.genome._hub (URL helpers + _hub_fetch_assets).

All tests use unittest.mock.patch to stub network calls. No real network.
"""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pandas as pd
import pytest

from pymisha.genome import _hub
from pymisha.genome._hub import (
    _hub_cgi_url,
    _hub_chrom_alias_url,
    _hub_chrom_sizes_url,
    _hub_fasta_url,
    _hub_fetch_assets,
    _hub_gtf_url,
    _hub_rmsk_url,
    _hub_url_for,
    _parse_chrom_sizes,
    _parse_hub_chrom_alias,
    _try_fetch,
)

# ---------------------------------------------------------------------------
# URL formatters
# ---------------------------------------------------------------------------


def test_hub_url_for_gca():
    assert _hub_url_for("GCA_009914755.4") == (
        "https://hgdownload.soe.ucsc.edu/hubs/GCA/009/914/755/GCA_009914755.4/"
    )


def test_hub_url_for_gcf():
    assert _hub_url_for("GCF_000001635.27") == (
        "https://hgdownload.soe.ucsc.edu/hubs/GCF/000/001/635/GCF_000001635.27/"
    )


def test_hub_url_for_invalid_accession_raises():
    with pytest.raises(ValueError, match="Invalid accession"):
        _hub_url_for("not-an-accession")
    with pytest.raises(ValueError, match="Invalid accession"):
        _hub_url_for("GCA_12345.1")  # too few digits
    with pytest.raises(ValueError, match="Invalid accession"):
        _hub_url_for("GCX_000001635.27")  # wrong prefix


def test_hub_url_helpers_compose_correctly():
    acc = "GCA_009914755.4"
    base = (
        "https://hgdownload.soe.ucsc.edu/hubs/GCA/009/914/755/GCA_009914755.4/"
    )
    assert _hub_chrom_alias_url(acc) == f"{base}{acc}.chromAlias.txt"
    assert _hub_chrom_sizes_url(acc) == f"{base}{acc}.chrom.sizes.txt"
    assert _hub_fasta_url(acc) == f"{base}{acc}.fa.gz"
    assert _hub_rmsk_url(acc) == f"{base}{acc}.repeatMasker.out.gz"
    assert _hub_cgi_url(acc) == f"{base}{acc}.cpgIslandExt.txt.gz"
    assert _hub_gtf_url(acc, "ncbiRefSeq") == (
        f"{base}genes/{acc}.ncbiRefSeq.gtf.gz"
    )


# ---------------------------------------------------------------------------
# chromAlias parser
# ---------------------------------------------------------------------------


def test_parse_hub_chrom_alias_wide_format():
    text = (
        "# refseq\tucsc\tlength\n"
        "NC_001\tchr1\t100\n"
        "NC_002\tchr2\t200\n"
    )
    df = _parse_hub_chrom_alias(text)
    assert list(df.columns) == ["refseq", "ucsc", "length"]
    assert len(df) == 2
    assert df.iloc[0]["refseq"] == "NC_001"
    assert df.iloc[0]["ucsc"] == "chr1"
    assert df.iloc[0]["length"] == "100"


def test_parse_hub_chrom_alias_falls_back_to_long_format():
    """No leading `#` => long-format _ucsc.py parser is used."""
    text = "1\tchr1\tensembl\nCM000663.2\tchr1\tgenbank\n"
    df = _parse_hub_chrom_alias(text)
    # Long-format parser pivots to wide and exposes a `chrom` column.
    assert "chrom" in df.columns
    assert {"ensembl", "genbank"}.issubset(df.columns)


def test_parse_chrom_sizes():
    text = "chr1\t100\nchr2\t200\n"
    df = _parse_chrom_sizes(text)
    assert list(df.columns) == ["name", "length"]
    assert len(df) == 2
    assert df.iloc[0]["name"] == "chr1"
    assert df.iloc[0]["length"] == 100


# ---------------------------------------------------------------------------
# _try_fetch error handling
# ---------------------------------------------------------------------------


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=code, msg=str(code), hdrs=None, fp=None
    )


def test_try_fetch_returns_none_on_404():
    with patch.object(_hub, "_open_url", side_effect=_make_http_error(404)):
        assert _try_fetch("http://example/missing") is None


def test_try_fetch_reraises_on_500():
    with patch.object(_hub, "_open_url", side_effect=_make_http_error(500)):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _try_fetch("http://example/boom")
        assert excinfo.value.code == 500


# ---------------------------------------------------------------------------
# _hub_fetch_assets
# ---------------------------------------------------------------------------


_ALIAS_WIDE_TEXT = (
    "# refseq\tucsc\n"
    "NC_001\tchr1\n"
    "NC_002\tchr2\n"
)


def _route_text(url: str) -> str:
    if url.endswith(".chromAlias.txt"):
        return _ALIAS_WIDE_TEXT
    if url.endswith(".chrom.sizes.txt"):
        return "chr1\t100\nchr2\t200\n"
    raise AssertionError(f"unexpected _read_url_text call: {url}")


def test_hub_fetch_assets_full_shape():
    """All sets requested -> populated dict; cytoband always None."""

    def fake_open(url, **kw):
        if url.endswith(".gtf.gz"):
            return b"GTF"
        if url.endswith(".repeatMasker.out.gz"):
            return b"RMSK"
        if url.endswith(".cpgIslandExt.txt.gz"):
            return b"CGI"
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_hub, "_open_url", side_effect=fake_open), \
            patch.object(_hub, "_read_url_text", side_effect=_route_text):
        out = _hub_fetch_assets(
            {"accession": "GCA_009914755.4"},
            sets=("genes", "rmsk", "cgi", "cytoband"),
        )

    assert set(out.keys()) == {
        "chrom_alias", "genes", "genes_source", "rmsk", "cgi", "cytoband",
    }
    assert isinstance(out["chrom_alias"], pd.DataFrame)
    assert out["genes"] == b"GTF"
    assert out["genes_source"] == "ncbiRefSeq"
    assert out["rmsk"] == b"RMSK"
    assert out["cgi"] == b"CGI"
    # Hubs don't ship cytoband.
    assert out["cytoband"] is None


def test_hub_fetch_assets_merges_chrom_sizes_into_alias():
    """alias DF has no length col; chrom.sizes is merged on best-overlap col."""

    def fake_open(url, **kw):
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_hub, "_open_url", side_effect=fake_open), \
            patch.object(_hub, "_read_url_text", side_effect=_route_text):
        out = _hub_fetch_assets({"accession": "GCA_009914755.4"}, sets=())

    alias = out["chrom_alias"]
    assert isinstance(alias, pd.DataFrame)
    assert "length" in alias.columns
    # Merge happened on the `ucsc` column (overlaps with chrom.sizes names).
    chr1 = alias[alias["ucsc"] == "chr1"].iloc[0]
    assert chr1["length"] == 100


def test_hub_fetch_assets_no_alias_returns_none_for_alias():
    """alias URL 404 -> chrom_alias None (sizes still tried but unused)."""

    def fake_text(url, **kw):
        if url.endswith(".chromAlias.txt"):
            raise _make_http_error(404)
        if url.endswith(".chrom.sizes.txt"):
            return "chr1\t100\n"
        raise AssertionError(url)

    with patch.object(_hub, "_open_url", side_effect=AssertionError), \
            patch.object(_hub, "_read_url_text", side_effect=fake_text):
        out = _hub_fetch_assets({"accession": "GCA_009914755.4"}, sets=())
    assert out["chrom_alias"] is None


def test_hub_fetch_assets_gtf_priority_falls_through():
    """First gtf priority 404s; second succeeds -> genes_source is second."""
    open_calls: list[str] = []

    def fake_open(url, **kw):
        open_calls.append(url)
        if "ncbiRefSeq.gtf.gz" in url:
            raise _make_http_error(404)
        if "refGene.gtf.gz" in url:
            return b"REFGENE"
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_hub, "_open_url", side_effect=fake_open), \
            patch.object(_hub, "_read_url_text", side_effect=_route_text):
        out = _hub_fetch_assets(
            {"accession": "GCA_009914755.4"},
            sets=("genes",),
            gtf_priority=("ncbiRefSeq", "refGene"),
        )
    assert out["genes"] == b"REFGENE"
    assert out["genes_source"] == "refGene"
    # First attempt 404, second succeeded -> exactly 2 open calls.
    assert len(open_calls) == 2


def test_hub_fetch_assets_cytoband_always_none():
    """Even with 'cytoband' in sets, hub returns cytoband=None."""

    def fake_open(url, **kw):
        # Should never be called for cytoband; hubs don't ship it.
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_hub, "_open_url", side_effect=fake_open), \
            patch.object(_hub, "_read_url_text", side_effect=_route_text):
        out = _hub_fetch_assets(
            {"accession": "GCA_009914755.4"},
            sets=("cytoband",),
        )
    assert out["cytoband"] is None


def test_hub_fetch_assets_only_fetches_requested():
    """Pass sets=('cgi',) -> only cgi is fetched; genes/rmsk untouched."""
    open_calls: list[str] = []

    def fake_open(url, **kw):
        open_calls.append(url)
        if "cpgIslandExt" in url:
            return b"CGI"
        raise AssertionError(f"unexpected _open_url call: {url}")

    with patch.object(_hub, "_open_url", side_effect=fake_open), \
            patch.object(_hub, "_read_url_text", side_effect=_route_text):
        out = _hub_fetch_assets(
            {"accession": "GCA_009914755.4"},
            sets=("cgi",),
        )
    assert out["genes"] is None
    assert out["genes_source"] is None
    assert out["rmsk"] is None
    assert out["cgi"] == b"CGI"
    assert out["cytoband"] is None
    # Exactly one _open_url call (cgi).
    assert len(open_calls) == 1
    assert "cpgIslandExt" in open_calls[0]
