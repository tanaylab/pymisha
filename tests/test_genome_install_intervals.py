"""Tests for `gdb_install_intervals` (pymisha.genome).

Network-free: `_ucsc_fetch_assets` is monkeypatched to return canned
bytes from the fixtures under tests/genome_fixtures/. The `pm.gintervals_*`
helpers are monkeypatched so no real misha state is touched - in
particular, the session-scoped test DB initialized by conftest.py is left
intact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import pymisha as pm
from pymisha import genome as pgen
from pymisha.genome import gdb_install_intervals

FIXTURES = Path(__file__).resolve().parent / "genome_fixtures"
GTF_BYTES = (FIXTURES / "sample.gtf").read_bytes()
RMSK_BYTES = (FIXTURES / "sample_rmsk.out").read_bytes()
CGI_BYTES = (FIXTURES / "sample_cgi.txt").read_bytes()
CYTO_BYTES = (FIXTURES / "sample_cytoband.txt").read_bytes()

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


@pytest.fixture
def install_env(tmp_path, monkeypatch):
    """Stub `pm.gdb_init` + `pm.gintervals_*` and yield a (groot, record) pair.

    `record` collects calls so individual tests can assert what would have
    been written without touching the real misha DB. The autouse teardown
    re-binds the test DB so the rest of the session sees a clean state.
    """
    # Set up a minimal groot at tmp_path: chrom_sizes.txt + seq/ dir.
    groot = tmp_path / "groot"
    groot.mkdir()
    (groot / "seq").mkdir()
    (groot / "chrom_sizes.txt").write_text(
        "1\t5000\n2\t3000\n", encoding="utf-8"
    )

    saved: list[tuple[pd.DataFrame, str]] = []
    removed: list[str] = []
    existing: set[str] = set()

    def fake_init(p, *a, **kw):
        return None

    def fake_save(df, name):
        saved.append((df.copy(), name))

    def fake_rm(name):
        removed.append(name)
        existing.discard(name)

    def fake_exists(name):
        return name in existing

    monkeypatch.setattr(pm, "gdb_init", fake_init)
    monkeypatch.setattr(pm, "gintervals_save", fake_save)
    monkeypatch.setattr(pm, "gintervals_rm", fake_rm)
    monkeypatch.setattr(pm, "gintervals_exists", fake_exists)

    record = {"saved": saved, "removed": removed, "existing": existing}

    yield groot, record

    # Re-bind the conftest-initialized test DB so downstream tests are
    # unaffected. (gdb_init was stubbed, so no real bind happened.)
    pm.gdb_init(str(TEST_DB))


def _make_assets(
    *,
    chrom_alias: pd.DataFrame | None = None,
    genes: bytes | None = GTF_BYTES,
    rmsk: bytes | None = RMSK_BYTES,
    cgi: bytes | None = CGI_BYTES,
    cytoband: bytes | None = CYTO_BYTES,
    genes_source: str | None = "ncbiRefSeq",
) -> dict:
    """Build an asset dict identical in shape to `_ucsc_fetch_assets`'s return."""
    if chrom_alias is None:
        chrom_alias = pd.DataFrame({
            "ucsc": ["chr1", "chr2"],
            "ensembl": ["1", "2"],
        })
    return {
        "chrom_alias": chrom_alias,
        "genes": genes,
        "genes_source": genes_source,
        "rmsk": rmsk,
        "cgi": cgi,
        "cytoband": cytoband,
    }


# ---------------------------------------------------------------------------
# 1. End-to-end ucsc install.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_ucsc_end_to_end(install_env, monkeypatch):
    groot, record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ucsc._ucsc_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(),
    )

    out = gdb_install_intervals(
        str(groot),
        source={"source": "ucsc", "assembly": "hg38"},
        sets=("genes", "rmsk", "cgi", "cytoband"),
        verbose=False,
    )

    names = {n for _, n in record["saved"]}
    # All four headline sets present.
    assert {"tss", "exons", "utr3", "utr5"}.issubset(names)
    assert "rmsk" in names
    # Per-class subsets that the fixture supplies.
    assert {
        "rmsk_SINE", "rmsk_LINE", "rmsk_LTR",
        "rmsk_DNA", "rmsk_Simple_repeat", "rmsk_Low_complexity",
    }.issubset(names)
    assert "cgi" in names
    assert "cytoband" in names

    # Chrom names ended up in the groot convention ("1","2"), not ucsc.
    for df, _name in record["saved"]:
        assert set(df["chrom"]).issubset({"1", "2"})
        assert not any(c.startswith("chr") for c in df["chrom"].astype(str))

    # Return value structure.
    assert set(out) == {"installed", "skipped", "failed"}
    assert out["failed"] == []
    assert out["skipped"] == []
    assert out["installed"]["cgi"] == 6
    assert out["installed"]["cytoband"] == 6

    # Provenance file landed.
    prov = json.loads((groot / "tracks" / ".misha_install.json").read_text())
    assert prov["source"] == "ucsc"
    assert prov["recipe"] == {"source": "ucsc", "assembly": "hg38"}
    assert set(prov["sets_installed"]) == names
    assert prov["gtf_source"] == "ncbiRefSeq"


# ---------------------------------------------------------------------------
# 2. force= behaviour when a set is unavailable.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_force_warns_on_missing(install_env, monkeypatch):
    groot, _record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ucsc._ucsc_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(cgi=None),
    )

    # force=False -> ValueError.
    with pytest.raises(ValueError, match="does not provide sets"):
        gdb_install_intervals(
            str(groot),
            source={"source": "ucsc", "assembly": "hg38"},
            sets=("genes", "rmsk", "cgi", "cytoband"),
            verbose=False,
        )

    # force=True -> warns and installs the available subset.
    with pytest.warns(UserWarning, match="Skipping sets"):
        out = gdb_install_intervals(
            str(groot),
            source={"source": "ucsc", "assembly": "hg38"},
            sets=("genes", "rmsk", "cgi", "cytoband"),
            force=True,
            verbose=False,
        )
    assert "cgi" in out["skipped"]
    assert "cgi" not in out["installed"]


# ---------------------------------------------------------------------------
# 3. overwrite= deletes pre-existing sets.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_overwrite_deletes_existing(install_env, monkeypatch):
    groot, record = install_env
    # Mark "tss" and "rmsk" as pre-existing so overwrite triggers gintervals_rm.
    record["existing"].update({"tss", "rmsk"})

    monkeypatch.setattr(
        "pymisha.genome._ucsc._ucsc_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(),
    )
    gdb_install_intervals(
        str(groot),
        source={"source": "ucsc", "assembly": "hg38"},
        sets=("genes", "rmsk"),
        overwrite=True,
        verbose=False,
    )
    # Both pre-existing sets should have been removed before re-install.
    assert "tss" in record["removed"]
    assert "rmsk" in record["removed"]
    # Saves still happened afterward.
    saved_names = {n for _, n in record["saved"]}
    assert "tss" in saved_names
    assert "rmsk" in saved_names


# ---------------------------------------------------------------------------
# 4 + 5. local / s3 backends are rejected.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_rejects_local_source(install_env):
    groot, _ = install_env
    with pytest.raises(ValueError, match="does not provide assets"):
        gdb_install_intervals(
            str(groot),
            source={"source": "local", "path": str(groot)},
            verbose=False,
        )


def test_gdb_install_intervals_rejects_s3_source(install_env):
    groot, _ = install_env
    with pytest.raises(ValueError, match="does not provide assets"):
        gdb_install_intervals(
            str(groot),
            source={"source": "s3", "name": "hg38"},
            verbose=False,
        )


# ---------------------------------------------------------------------------
# 6. ncbi end-to-end (C.4.2): canned _ncbi_fetch_assets dict.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_ncbi_wired_up(install_env, monkeypatch):
    """ncbi source is wired into the orchestrator (no NotImplementedError)."""
    groot, record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ncbi._ncbi_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(
            # NCBI never ships cgi/cytoband; mimic the real fetcher.
            cgi=None,
            cytoband=None,
            genes_source="RefSeq",
        ),
    )

    out = gdb_install_intervals(
        str(groot),
        source={"source": "ncbi", "accession": "GCF_000001405.40"},
        sets=("genes", "rmsk"),
        verbose=False,
    )

    names = {n for _, n in record["saved"]}
    assert {"tss", "exons", "utr3", "utr5"}.issubset(names)
    assert "rmsk" in names
    assert out["failed"] == []
    assert out["skipped"] == []

    prov = json.loads((groot / "tracks" / ".misha_install.json").read_text())
    assert prov["source"] == "ncbi"
    assert prov["recipe"] == {
        "source": "ncbi",
        "accession": "GCF_000001405.40",
    }
    assert prov["gtf_source"] == "RefSeq"


# ---------------------------------------------------------------------------
# 7. ucsc-hub end-to-end (C.3.2): canned _hub_fetch_assets dict.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_hub_end_to_end(install_env, monkeypatch):
    groot, record = install_env
    monkeypatch.setattr(
        "pymisha.genome._hub._hub_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(cytoband=None),
    )

    out = gdb_install_intervals(
        str(groot),
        source={"source": "ucsc-hub", "accession": "GCA_009914755.4"},
        sets=("genes", "rmsk", "cgi"),
        verbose=False,
    )

    names = {n for _, n in record["saved"]}
    # Headline gene + rmsk + cgi sets present.
    assert {"tss", "exons", "utr3", "utr5"}.issubset(names)
    assert "rmsk" in names
    assert "cgi" in names
    # cytoband was not requested, so it should not have been installed.
    assert "cytoband" not in names

    # Chrom names ended up in the groot convention ("1","2"), not ucsc.
    for df, _name in record["saved"]:
        assert set(df["chrom"]).issubset({"1", "2"})

    assert out["failed"] == []
    assert out["skipped"] == []

    # Provenance reflects the ucsc-hub source.
    prov = json.loads((groot / "tracks" / ".misha_install.json").read_text())
    assert prov["source"] == "ucsc-hub"
    assert prov["recipe"] == {
        "source": "ucsc-hub",
        "accession": "GCA_009914755.4",
    }


# ---------------------------------------------------------------------------
# 8. gdb_build_genome with sets= invokes gdb_install_intervals.
# ---------------------------------------------------------------------------

def test_gdb_build_genome_with_sets_calls_install(tmp_path, monkeypatch):
    """Stub `_build_seq`, `gdb_init`, and `gdb_install_intervals` so we can
    verify the wiring in `gdb_build_genome` without touching real state.
    """
    monkeypatch.setattr("pymisha.genome._build_seq._build_seq", lambda *a, **kw: None)
    monkeypatch.setattr("pymisha.db.gdb_init", lambda *a, **kw: None)

    captured: list[dict] = []

    def fake_install(groot, source, **kwargs):
        captured.append({"groot": groot, "source": source, **kwargs})
        return {"installed": {}, "skipped": [], "failed": []}

    monkeypatch.setattr(pgen, "gdb_install_intervals", fake_install)

    # Build a minimal registry entry.
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "version: 1\ngenome:\n  myhg: {source: ucsc, assembly: hg38}\n",
        encoding="utf-8",
    )
    groot = tmp_path / "g"

    pgen.gdb_build_genome(
        "myhg",
        path=str(groot),
        registry=str(registry),
        sets=("genes",),
        verbose=False,
    )

    assert len(captured) == 1
    call = captured[0]
    assert call["groot"] == str(groot)
    assert call["source"]["source"] == "ucsc"
    assert call["source"]["assembly"] == "hg38"
    assert call["sets"] == ("genes",)


# ---------------------------------------------------------------------------
# 9. Provenance JSON shape.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_provenance_json_shape(install_env, monkeypatch):
    groot, _record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ucsc._ucsc_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(),
    )
    gdb_install_intervals(
        str(groot),
        source={"source": "ucsc", "assembly": "hg38"},
        sets=("genes",),
        verbose=False,
    )
    prov = json.loads((groot / "tracks" / ".misha_install.json").read_text())
    for key in ("source", "recipe", "sets_installed", "row_counts",
                "timestamp", "gtf_source"):
        assert key in prov, f"missing provenance key: {key}"
    # row_counts should be a positive-int dict.
    assert isinstance(prov["row_counts"], dict)
    for k, v in prov["row_counts"].items():
        assert isinstance(k, str)
        assert isinstance(v, int) and v > 0


# ---------------------------------------------------------------------------
# C.4.3 - force= parity across backends (R 5.6.29 968bf782).
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_force_ncbi_skips_cgi_and_cytoband(install_env, monkeypatch):
    """NCBI never ships cgi/cytoband. force=True installs the subset; force=False raises."""
    groot, record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ncbi._ncbi_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(
            cgi=None, cytoband=None, genes_source="RefSeq",
        ),
    )
    src = {"source": "ncbi", "accession": "GCF_000001405.40"}

    with pytest.raises(ValueError, match=r"does not provide sets.*\['cgi', 'cytoband'\]"):
        gdb_install_intervals(
            str(groot), source=src,
            sets=("genes", "rmsk", "cgi", "cytoband"),
            verbose=False,
        )

    with pytest.warns(UserWarning, match="Skipping sets"):
        out = gdb_install_intervals(
            str(groot), source=src,
            sets=("genes", "rmsk", "cgi", "cytoband"),
            force=True,
            verbose=False,
        )
    assert set(out["skipped"]) == {"cgi", "cytoband"}
    names = {n for _, n in record["saved"]}
    assert "rmsk" in names
    assert {"tss", "exons"}.issubset(names)
    assert "cgi" not in names
    assert "cytoband" not in names


def test_gdb_install_intervals_force_hub_skips_cytoband(install_env, monkeypatch):
    """UCSC hubs never ship cytoband. force=True installs the subset; force=False raises."""
    groot, record = install_env
    monkeypatch.setattr(
        "pymisha.genome._hub._hub_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(cytoband=None),
    )
    src = {"source": "ucsc-hub", "accession": "GCA_009914755.4"}

    with pytest.raises(ValueError, match=r"does not provide sets.*\['cytoband'\]"):
        gdb_install_intervals(
            str(groot), source=src,
            sets=("genes", "rmsk", "cgi", "cytoband"),
            verbose=False,
        )

    with pytest.warns(UserWarning, match="Skipping sets"):
        out = gdb_install_intervals(
            str(groot), source=src,
            sets=("genes", "rmsk", "cgi", "cytoband"),
            force=True,
            verbose=False,
        )
    assert set(out["skipped"]) == {"cytoband"}
    names = {n for _, n in record["saved"]}
    assert "rmsk" in names
    assert "cgi" in names
    assert "cytoband" not in names


def test_gdb_install_intervals_force_install_only_available_subset(install_env, monkeypatch):
    """Plan C.4.3 matrix: request only ('genes','cgi') from a backend lacking cgi."""
    groot, record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ucsc._ucsc_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(cgi=None),
    )
    src = {"source": "ucsc", "assembly": "hg38"}

    # force=False -> raises.
    with pytest.raises(ValueError, match=r"does not provide sets.*\['cgi'\]"):
        gdb_install_intervals(
            str(groot), source=src, sets=("genes", "cgi"), verbose=False,
        )

    # force=True -> warns, installs genes only.
    with pytest.warns(UserWarning, match="Skipping sets"):
        out = gdb_install_intervals(
            str(groot), source=src, sets=("genes", "cgi"),
            force=True, verbose=False,
        )
    names = {n for _, n in record["saved"]}
    assert {"tss", "exons", "utr3", "utr5"}.issubset(names)
    assert "cgi" not in names
    assert out["skipped"] == ["cgi"]


# ---------------------------------------------------------------------------
# 10. Atomic write: no .tmp.* files survive a successful run.
# ---------------------------------------------------------------------------

def test_gdb_install_intervals_writes_atomically(install_env, monkeypatch):
    groot, _record = install_env
    monkeypatch.setattr(
        "pymisha.genome._ucsc._ucsc_fetch_assets",
        lambda recipe, sets, gtf_priority=(): _make_assets(),
    )
    gdb_install_intervals(
        str(groot),
        source={"source": "ucsc", "assembly": "hg38"},
        sets=("genes",),
        verbose=False,
    )
    tracks_dir = groot / "tracks"
    leftover = list(tracks_dir.glob(".misha_install.json.tmp.*"))
    assert leftover == [], f"unexpected tmp files: {leftover}"
    assert (tracks_dir / ".misha_install.json").exists()
