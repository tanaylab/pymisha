"""Unit tests for _array_track._import_sources (pure-Python multi-source array merge).

These tests run against the small test DB (auto-initialised) and cover the
merge primitives in isolation. The R-parity test against the .rds baseline
lives in tests/r_parity/test_gtrack_array.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha._array_track import (
    _CSVSource,
    _TrackSource,
    _import_sources,
    extract_array,
    write_colnames,
)


@pytest.fixture(autouse=True)
def _init_db():
    pm.gdb_init_examples()


def _write_tsv(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    path = tmp_path / name
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return path


# ---------------------------------------------------------------------------
# _CSVSource
# ---------------------------------------------------------------------------

def test_csv_source_parses_header_and_intervals(tmp_path):
    path = _write_tsv(tmp_path, "src.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "a": 1.0, "b": 2.0},
        {"chrom": "1", "start": 100, "end": 200, "a": 3.0, "b": np.nan},
    ])
    src = _CSVSource(path)
    assert src.colnames == ["a", "b"]
    intervals, vals = src.read_chrom("1")
    assert intervals.tolist() == [[0, 100], [100, 200]]
    assert vals.shape == (2, 2)
    assert vals[0].tolist() == [1.0, 2.0]
    assert np.isnan(vals[1, 1])


def test_csv_source_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("chrom\tbegin\tend\ta\n1\t0\t10\t1.0\n")
    with pytest.raises(ValueError, match="invalid format"):
        _CSVSource(path)


def test_csv_source_rejects_overlap(tmp_path):
    path = _write_tsv(tmp_path, "ovl.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "a": 1.0},
        {"chrom": "1", "start": 50, "end": 200, "a": 2.0},
    ])
    src = _CSVSource(path)
    with pytest.raises(ValueError, match="overlap"):
        src.read_chrom("1")


def test_csv_source_rejects_zero_length(tmp_path):
    path = _write_tsv(tmp_path, "z.tsv", [
        {"chrom": "1", "start": 100, "end": 100, "a": 1.0},
    ])
    src = _CSVSource(path)
    with pytest.raises(ValueError, match="start coordinate exceeds"):
        src.read_chrom("1")


# ---------------------------------------------------------------------------
# _TrackSource
# ---------------------------------------------------------------------------

def test_track_source_reads_array_track():
    src = _TrackSource("array_track")
    assert isinstance(src.colnames, list) and len(src.colnames) > 0
    chroms = src.chroms()
    assert len(chroms) > 0
    intervals, vals = src.read_chrom(chroms[0])
    assert intervals.shape[1] == 2
    assert vals.shape[1] == len(src.colnames)


def test_track_source_rejects_non_array():
    with pytest.raises(ValueError, match="only array tracks"):
        _TrackSource("dense_track")


# ---------------------------------------------------------------------------
# _import_sources
# ---------------------------------------------------------------------------

def test_import_sources_single_csv(tmp_path):
    csv = _write_tsv(tmp_path, "one.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "a": 1.0, "b": 2.0},
        {"chrom": "1", "start": 100, "end": 200, "a": 3.0, "b": np.nan},
        {"chrom": "2", "start": 0, "end": 50, "a": 4.0, "b": 5.0},
    ])
    out_dir = tmp_path / "track"
    out_dir.mkdir()
    chrom_order = ["1", "2", "X"]
    colnames = _import_sources(out_dir, [_CSVSource(csv)], chrom_order)
    assert colnames == ["a", "b"]
    write_colnames(out_dir, colnames)
    query = pd.DataFrame({"chrom": ["1", "2"], "start": [0, 0], "end": [200, 50]})
    out = extract_array(out_dir, query, None, colnames, chrom_order=chrom_order)
    out = out.sort_values(["chrom", "start"]).reset_index(drop=True)
    assert out["a"].tolist() == [1.0, 3.0, 4.0]
    assert np.isnan(out["b"].iloc[1])


def test_import_sources_two_csv_disjoint(tmp_path):
    csv1 = _write_tsv(tmp_path, "s1.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "a": 1.0},
    ])
    csv2 = _write_tsv(tmp_path, "s2.tsv", [
        {"chrom": "1", "start": 100, "end": 200, "b": 2.0},
    ])
    out_dir = tmp_path / "track"
    out_dir.mkdir()
    cols = _import_sources(out_dir, [_CSVSource(csv1), _CSVSource(csv2)], ["1", "2", "X"])
    assert cols == ["a", "b"]
    write_colnames(out_dir, cols)
    query = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [200]})
    out = (
        extract_array(out_dir, query, None, cols, chrom_order=["1", "2", "X"])
        .sort_values(["start"]).reset_index(drop=True)
    )
    assert out["start"].tolist() == [0, 100]
    # row 0: a=1.0, b=NaN; row 1: a=NaN, b=2.0
    assert out["a"].iloc[0] == 1.0
    assert np.isnan(out["a"].iloc[1])
    assert np.isnan(out["b"].iloc[0])
    assert out["b"].iloc[1] == 2.0


def test_import_sources_two_csv_shared_interval(tmp_path):
    csv1 = _write_tsv(tmp_path, "s1.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "a": 1.0},
    ])
    csv2 = _write_tsv(tmp_path, "s2.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "b": 2.0},
    ])
    out_dir = tmp_path / "track"
    out_dir.mkdir()
    cols = _import_sources(out_dir, [_CSVSource(csv1), _CSVSource(csv2)], ["1", "2", "X"])
    assert cols == ["a", "b"]
    write_colnames(out_dir, cols)
    query = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    out = extract_array(out_dir, query, None, cols, chrom_order=["1", "2", "X"]).reset_index(drop=True)
    assert len(out) == 1
    assert out["a"].iloc[0] == 1.0
    assert out["b"].iloc[0] == 2.0


def test_import_sources_partial_overlap_raises(tmp_path):
    csv1 = _write_tsv(tmp_path, "s1.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "a": 1.0},
    ])
    csv2 = _write_tsv(tmp_path, "s2.tsv", [
        {"chrom": "1", "start": 50, "end": 150, "b": 2.0},
    ])
    out_dir = tmp_path / "track"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="overlaps interval"):
        _import_sources(out_dir, [_CSVSource(csv1), _CSVSource(csv2)], ["1", "2", "X"])


def test_import_sources_chain_consistency_error(tmp_path):
    csv1 = _write_tsv(tmp_path, "s1.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "shared": 1.0},
    ])
    csv2 = _write_tsv(tmp_path, "s2.tsv", [
        {"chrom": "1", "start": 0, "end": 100, "shared": 2.0},
    ])
    out_dir = tmp_path / "track"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="Non matching values"):
        _import_sources(out_dir, [_CSVSource(csv1), _CSVSource(csv2)], ["1", "2", "X"])


# ---------------------------------------------------------------------------
# gtrack_array_import (public API)
# ---------------------------------------------------------------------------

def test_gtrack_array_import_end_to_end(tmp_path):
    """End-to-end on the small test DB: build two CSVs, import, extract back."""
    csv1 = _write_tsv(tmp_path, "a.tsv", [
        {"chrom": "1", "start": 0, "end": 1000, "alpha": 0.5},
        {"chrom": "1", "start": 1000, "end": 2000, "alpha": 1.5},
    ])
    csv2 = _write_tsv(tmp_path, "b.tsv", [
        {"chrom": "1", "start": 0, "end": 1000, "beta": 10.0},
    ])
    name = "test.import_e2e"
    try:
        pm.gtrack_array_import(name, "round-trip e2e", str(csv1), str(csv2))
        cn = pm.gtrack_array_get_colnames(name)
        assert cn == ["alpha", "beta"]
        out = (
            pm.gtrack_array_extract(name, None, pm.gintervals(1))
            .sort_values(["start"]).reset_index(drop=True)
        )
        assert out["start"].tolist() == [0, 1000]
        assert out["alpha"].tolist() == [0.5, 1.5]
        assert out["beta"].iloc[0] == 10.0
        assert np.isnan(out["beta"].iloc[1])
    finally:
        import contextlib
        with contextlib.suppress(Exception):
            pm.gtrack_rm(name, force=True)


def test_gtrack_array_import_rejects_non_array_source():
    name = "test.import_reject"
    with pytest.raises(ValueError, match="only array tracks"):
        pm.gtrack_array_import(name, "", "dense_track")
    import contextlib
    with contextlib.suppress(Exception):
        pm.gtrack_rm(name, force=True)
