"""Tests for `_install_rmsk`, `_install_cgi`, `_install_cytoband`.

Same pattern as `tests/test_genome_install_genes.py`: monkeypatch
`pymisha.gintervals_save` so no groot mutation happens. We assert what
*would* have been written.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

import pymisha as pm
from pymisha.genome._install_sets import (
    _install_cgi,
    _install_cytoband,
    _install_rmsk,
    _parse_rmsk_out,
)

FIXTURES = Path(__file__).resolve().parent / "genome_fixtures"
RMSK_FIXTURE = FIXTURES / "sample_rmsk.out"
CGI_FIXTURE = FIXTURES / "sample_cgi.txt"
CYTOBAND_FIXTURE = FIXTURES / "sample_cytoband.txt"


def _identity(chrom: str) -> str | None:
    return chrom


def _drop_chrom2(chrom: str) -> str | None:
    return None if chrom == "2" else chrom


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[pd.DataFrame, str]]:
    """Replace `gintervals_save` with a recorder; return the call log."""
    calls: list[tuple[pd.DataFrame, str]] = []

    def _fake_save(df: pd.DataFrame, name: str) -> None:
        calls.append((df.copy(), name))

    monkeypatch.setattr(pm, "gintervals_save", _fake_save)
    return calls


# ---------------------------------------------------------------------------
# _parse_rmsk_out
# ---------------------------------------------------------------------------

def test_parse_rmsk_out_skips_header_and_blanks():
    df = _parse_rmsk_out(RMSK_FIXTURE.read_bytes())
    # 7 data rows in the fixture, header / blank lines skipped.
    assert len(df) == 7
    assert list(df.columns) == [
        "chrom",
        "start",
        "end",
        "repeat_name",
        "repeat_class",
    ]
    assert set(df["chrom"]) == {"1", "2"}


def test_parse_rmsk_out_converts_to_half_open():
    df = _parse_rmsk_out(RMSK_FIXTURE.read_bytes())
    # First data row in the fixture: chrom 1, begin 1001, end 1200.
    # Half-open 0-based: start=1000, end=1200, width=200.
    first = df.iloc[0]
    assert first["chrom"] == "1"
    assert first["start"] == 1000
    assert first["end"] == 1200
    # All intervals positive width.
    assert ((df["end"] - df["start"]) > 0).all()


def test_parse_rmsk_out_handles_gzipped_input():
    raw = RMSK_FIXTURE.read_bytes()
    gzipped = gzip.compress(raw)
    df_plain = _parse_rmsk_out(raw)
    df_gz = _parse_rmsk_out(gzipped)
    pd.testing.assert_frame_equal(df_plain, df_gz)


# ---------------------------------------------------------------------------
# _install_rmsk
# ---------------------------------------------------------------------------

def test_install_rmsk_writes_whole_set(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_rmsk(RMSK_FIXTURE.read_bytes(), _identity)
    names = [n for _, n in calls]
    assert "rmsk" in names
    by_name = {name: df for df, name in calls}
    # 7 data rows in the fixture.
    assert len(by_name["rmsk"]) == 7
    assert installed["rmsk"] == 7
    # Only chrom/start/end columns get saved.
    assert list(by_name["rmsk"].columns) == ["chrom", "start", "end"]


def test_install_rmsk_writes_per_class_subsets(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_rmsk(RMSK_FIXTURE.read_bytes(), _identity)
    by_name = {name: df for df, name in calls}
    # Fixture has 2 SINE/Alu, 1 LINE/L1, 1 LTR/ERV1, 1 DNA/hAT-Charlie,
    # 1 Simple_repeat, 1 Low_complexity.
    assert installed["rmsk_SINE"] == 2
    assert installed["rmsk_LINE"] == 1
    assert installed["rmsk_LTR"] == 1
    assert installed["rmsk_DNA"] == 1
    assert installed["rmsk_Simple_repeat"] == 1
    assert installed["rmsk_Low_complexity"] == 1
    # Per-class DataFrames have only the canonical 3 columns.
    for cls in ("SINE", "LINE", "LTR", "DNA", "Simple_repeat", "Low_complexity"):
        sub = by_name[f"rmsk_{cls}"]
        assert list(sub.columns) == ["chrom", "start", "end"]


def test_install_rmsk_skips_unmappable_chroms(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_rmsk(RMSK_FIXTURE.read_bytes(), _drop_chrom2)
    # Only chrom "1" rows survive (3 of them: SINE/Alu, LINE/L1, LTR/ERV1).
    assert installed["rmsk"] == 3
    by_name = {name: df for df, name in calls}
    for df, _name in calls:
        assert (df["chrom"] == "1").all()
    # SINE on chrom 2 (AluY) and DNA/Simple_repeat/Low_complexity (all on
    # chrom 2) should be dropped.
    assert "rmsk_DNA" not in by_name
    assert "rmsk_Simple_repeat" not in by_name
    assert "rmsk_Low_complexity" not in by_name
    # SINE survives because there's an Alu on chrom 1.
    assert installed["rmsk_SINE"] == 1


def test_install_rmsk_respects_prefix(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_rmsk(
        RMSK_FIXTURE.read_bytes(), _identity, prefix="ucsc."
    )
    names = {n for _, n in calls}
    assert "ucsc.rmsk" in names
    assert "ucsc.rmsk_SINE" in names
    assert "ucsc.rmsk_LINE" in names
    assert all(n.startswith("ucsc.") for n in names)
    assert all(n.startswith("ucsc.") for n in installed)


def test_install_rmsk_returns_count_dict(monkeypatch):
    _capture(monkeypatch)
    installed = _install_rmsk(RMSK_FIXTURE.read_bytes(), _identity)
    # Total of the whole set must equal sum of per-class subsets (since
    # every fixture row falls in exactly one of the listed classes).
    per_class = sum(v for k, v in installed.items() if k != "rmsk")
    assert per_class == installed["rmsk"]


# ---------------------------------------------------------------------------
# _install_cgi
# ---------------------------------------------------------------------------

def test_install_cgi_writes_set(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_cgi(CGI_FIXTURE.read_bytes(), _identity)
    names = [n for _, n in calls]
    assert names == ["cgi"]
    df = calls[0][0]
    assert list(df.columns) == ["chrom", "start", "end"]
    assert len(df) == 6
    assert installed == {"cgi": 6}
    # First row in fixture: chrom 1, start 1000, end 1500.
    first = df.iloc[0]
    assert first["chrom"] == "1"
    assert int(first["start"]) == 1000
    assert int(first["end"]) == 1500


def test_install_cgi_skips_unmappable(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_cgi(CGI_FIXTURE.read_bytes(), _drop_chrom2)
    # Only chrom 1 rows (3 of them).
    assert installed == {"cgi": 3}
    df = calls[0][0]
    assert (df["chrom"] == "1").all()


def test_install_cgi_handles_gzipped_input(monkeypatch):
    calls = _capture(monkeypatch)
    raw = CGI_FIXTURE.read_bytes()
    _install_cgi(gzip.compress(raw), _identity)
    df_gz = calls[0][0]

    calls.clear()
    _install_cgi(raw, _identity)
    df_plain = calls[0][0]
    pd.testing.assert_frame_equal(df_plain, df_gz)


# ---------------------------------------------------------------------------
# _install_cytoband
# ---------------------------------------------------------------------------

def test_install_cytoband_writes_set(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_cytoband(CYTOBAND_FIXTURE.read_bytes(), _identity)
    names = [n for _, n in calls]
    assert names == ["cytoband"]
    df = calls[0][0]
    assert list(df.columns) == ["chrom", "start", "end"]
    assert len(df) == 6
    assert installed == {"cytoband": 6}
    # First row in fixture: chrom 1, start 0, end 2300000.
    first = df.iloc[0]
    assert first["chrom"] == "1"
    assert int(first["start"]) == 0
    assert int(first["end"]) == 2_300_000


def test_install_cytoband_skips_unmappable(monkeypatch):
    calls = _capture(monkeypatch)
    installed = _install_cytoband(CYTOBAND_FIXTURE.read_bytes(), _drop_chrom2)
    # Only chrom 1 rows (3 of them).
    assert installed == {"cytoband": 3}
    df = calls[0][0]
    assert (df["chrom"] == "1").all()


def test_install_cytoband_handles_gzipped_input(monkeypatch):
    calls = _capture(monkeypatch)
    raw = CYTOBAND_FIXTURE.read_bytes()
    _install_cytoband(gzip.compress(raw), _identity)
    df_gz = calls[0][0]

    calls.clear()
    _install_cytoband(raw, _identity)
    df_plain = calls[0][0]
    pd.testing.assert_frame_equal(df_plain, df_gz)
