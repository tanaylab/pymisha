"""Tests for the C++ WIG/BedGraph parser fast path (pm_parse_wig_or_bedgraph)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import _pymisha
from pymisha.tracks import _parse_wig_or_bedgraph


def _read_via_cpp(path: str) -> dict:
    """Call the C++ parser. Returns dict with chrom/start/end/value arrays."""
    return _pymisha.pm_parse_wig_or_bedgraph(path)


def test_cpp_parser_fixedstep_simple(tmp_path: Path) -> None:
    wig = tmp_path / "fixed.wig"
    wig.write_text(
        "track type=wiggle_0 name=test\n"
        "fixedStep chrom=chr1 start=1 step=10 span=10\n"
        "2.0\n"
        "4.0\n"
        "6.0\n"
    )
    out = _read_via_cpp(str(wig))
    assert list(out["chrom"]) == ["chr1", "chr1", "chr1"]
    np.testing.assert_array_equal(out["start"], [0, 10, 20])
    np.testing.assert_array_equal(out["end"], [10, 20, 30])
    np.testing.assert_allclose(out["value"], [2.0, 4.0, 6.0])


def test_cpp_parser_variablestep(tmp_path: Path) -> None:
    wig = tmp_path / "var.wig"
    wig.write_text(
        "variableStep chrom=chr2 span=5\n"
        "10 1.5\n"
        "20 2.5\n"
        "100 -1.0\n"
    )
    out = _read_via_cpp(str(wig))
    assert list(out["chrom"]) == ["chr2", "chr2", "chr2"]
    np.testing.assert_array_equal(out["start"], [9, 19, 99])
    np.testing.assert_array_equal(out["end"], [14, 24, 104])
    np.testing.assert_allclose(out["value"], [1.5, 2.5, -1.0])


def test_cpp_parser_bedgraph(tmp_path: Path) -> None:
    bg = tmp_path / "in.bg"
    bg.write_text(
        "# comment\n"
        "track type=bedGraph\n"
        "chr1\t0\t10\t5.0\n"
        "chr1\t10\t20\t7.0\n"
        "chr2\t100\t200\t-3.5\n"
    )
    out = _read_via_cpp(str(bg))
    assert list(out["chrom"]) == ["chr1", "chr1", "chr2"]
    np.testing.assert_array_equal(out["start"], [0, 10, 100])
    np.testing.assert_array_equal(out["end"], [10, 20, 200])
    np.testing.assert_allclose(out["value"], [5.0, 7.0, -3.5])


def test_cpp_parser_skips_comments_and_browser(tmp_path: Path) -> None:
    bg = tmp_path / "x.bg"
    bg.write_text(
        "#some comment\n"
        "browser position chr1:1-1000\n"
        "track name=test color=255,0,0\n"
        "chr1\t0\t5\t1.0\n"
    )
    out = _read_via_cpp(str(bg))
    assert list(out["chrom"]) == ["chr1"]
    np.testing.assert_array_equal(out["start"], [0])
    np.testing.assert_array_equal(out["end"], [5])
    np.testing.assert_allclose(out["value"], [1.0])


def test_cpp_parser_fixedstep_default_step_and_span(tmp_path: Path) -> None:
    """fixedStep without step/span defaults both to 1."""
    wig = tmp_path / "min.wig"
    wig.write_text(
        "fixedStep chrom=chrX start=100\n"
        "1\n"
        "2\n"
        "3\n"
    )
    out = _read_via_cpp(str(wig))
    assert list(out["chrom"]) == ["chrX", "chrX", "chrX"]
    np.testing.assert_array_equal(out["start"], [99, 100, 101])
    np.testing.assert_array_equal(out["end"], [100, 101, 102])
    np.testing.assert_allclose(out["value"], [1.0, 2.0, 3.0])


def test_cpp_parser_variablestep_default_span(tmp_path: Path) -> None:
    """variableStep without span defaults span=1."""
    wig = tmp_path / "var2.wig"
    wig.write_text(
        "variableStep chrom=chr3\n"
        "5 0.5\n"
        "10 0.7\n"
    )
    out = _read_via_cpp(str(wig))
    np.testing.assert_array_equal(out["start"], [4, 9])
    np.testing.assert_array_equal(out["end"], [5, 10])


def test_cpp_parser_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.wig"
    p.write_text("# only comments\n")
    with pytest.raises(Exception, match="no intervals|empty|no data"):
        _read_via_cpp(str(p))


def test_cpp_parser_malformed_fixedstep_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.wig"
    p.write_text(
        "fixedStep chrom=chr1\n"  # missing start=
        "1\n"
    )
    with pytest.raises(Exception):
        _read_via_cpp(str(p))


def test_cpp_parser_gzipped_rejected(tmp_path: Path) -> None:
    """Gzipped paths are handled by the Python wrapper, not C++ directly.

    The C++ parser is plain-text only; the Python wrapper falls back when
    the path looks gzipped.
    """
    wig = tmp_path / "x.wig.gz"
    with gzip.open(str(wig), "wt") as fh:
        fh.write("fixedStep chrom=chr1 start=1\n1\n")
    # C++ parser should reject (or return garbage on) gz bytes
    with pytest.raises(Exception):
        _read_via_cpp(str(wig))


def test_cpp_parser_matches_python_fixedstep(tmp_path: Path) -> None:
    wig = tmp_path / "match.wig"
    wig.write_text(
        "track type=wiggle_0 name=match\n"
        "fixedStep chrom=chr1 start=1 step=5 span=5\n"
        "1.1\n2.2\n3.3\n"
        "fixedStep chrom=chr2 start=10 step=2 span=2\n"
        "10.0\n20.0\n"
    )
    cpp = _read_via_cpp(str(wig))
    py_df = _parse_wig_or_bedgraph(str(wig))

    assert list(cpp["chrom"]) == list(py_df["chrom"])
    np.testing.assert_array_equal(cpp["start"], py_df["start"].to_numpy())
    np.testing.assert_array_equal(cpp["end"], py_df["end"].to_numpy())
    np.testing.assert_allclose(cpp["value"], py_df["value"].to_numpy())


def test_cpp_parser_matches_python_bedgraph(tmp_path: Path) -> None:
    bg = tmp_path / "match.bg"
    bg.write_text(
        "track type=bedGraph\n"
        "chr1\t0\t10\t1.5\n"
        "chr1\t10\t20\t2.5\n"
        "chrX\t100\t150\t-7.0\n"
    )
    cpp = _read_via_cpp(str(bg))
    py_df = _parse_wig_or_bedgraph(str(bg))
    assert list(cpp["chrom"]) == list(py_df["chrom"])
    np.testing.assert_array_equal(cpp["start"], py_df["start"].to_numpy())
    np.testing.assert_array_equal(cpp["end"], py_df["end"].to_numpy())
    np.testing.assert_allclose(cpp["value"], py_df["value"].to_numpy())
