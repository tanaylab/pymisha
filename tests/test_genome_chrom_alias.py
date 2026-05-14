"""Tests for pymisha.genome._chrom_alias (single-pass C.2.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymisha.genome._chrom_alias import (
    _build_translator,
    _canonical_coverage,
    _detect_alias_column,
    _resolve_chrom_alias,
)

# ---------------------------------------------------------------------------
# _canonical_coverage
# ---------------------------------------------------------------------------

def test_canonical_coverage_empty_groot_returns_one():
    canon = pd.Series(["chr1", "chr2"])
    assert _canonical_coverage(canon, [], []) == 1.0


def test_canonical_coverage_bp_weighted():
    canon = pd.Series(["chr1", "chr3", "chrX"])
    groot = ["chr1", "chr2", "chr3"]
    lengths = [100, 50, 10]
    # mapped bp = 100 + 10 = 110; total = 160
    assert _canonical_coverage(canon, groot, lengths) == pytest.approx(110 / 160)


def test_canonical_coverage_empty_total_returns_zero():
    canon = pd.Series(["chr1", "chr2"])
    groot = ["chr1", "chr2"]
    lengths = [0, 0]
    assert _canonical_coverage(canon, groot, lengths) == 0.0


def test_canonical_coverage_drops_empty_strings_and_nan():
    canon = pd.Series(["chr1", "", None, "chr2"])
    groot = ["chr1", "chr2", ""]
    lengths = [100, 50, 25]
    # canon set after cleaning: {chr1, chr2}; groot "" is not in canon set
    # mapped bp = 100 + 50 = 150; total = 175
    assert _canonical_coverage(canon, groot, lengths) == pytest.approx(150 / 175)


def test_canonical_coverage_misaligned_raises():
    canon = pd.Series(["chr1"])
    with pytest.raises(ValueError, match="must align"):
        _canonical_coverage(canon, ["chr1", "chr2"], [100])


# ---------------------------------------------------------------------------
# _detect_alias_column
# ---------------------------------------------------------------------------

def test_detect_alias_column_picks_best_at_full_coverage():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2", "chr3"],
            "refseq": ["NC1", "NC2", "NC3"],
            "ensembl": ["1", "2", "3"],
        }
    )
    groot = ["1", "2", "3"]
    lengths = [100, 50, 10]
    picked, coverages = _detect_alias_column(
        alias_df, groot, lengths, min_coverage=1.0
    )
    assert picked == "ensembl"
    assert set(coverages.keys()) == {"ucsc", "refseq", "ensembl"}
    assert coverages["ensembl"] == 1.0
    assert coverages["ucsc"] == 0.0
    assert coverages["refseq"] == 0.0


def test_detect_alias_column_returns_none_below_min_coverage():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2"],
            "ensembl": ["1", "2"],
        }
    )
    groot = ["chrA", "chrB"]
    lengths = [100, 50]
    picked, coverages = _detect_alias_column(
        alias_df, groot, lengths, min_coverage=1.0
    )
    assert picked is None
    assert coverages == {"ucsc": 0.0, "ensembl": 0.0}


def test_detect_alias_column_ties_broken_by_max():
    # Two columns both at coverage=1.0: max picks the first to reach the max.
    alias_df = pd.DataFrame(
        {
            "first": ["1", "2"],
            "second": ["1", "2"],
        }
    )
    groot = ["1", "2"]
    lengths = [100, 50]
    picked, coverages = _detect_alias_column(
        alias_df, groot, lengths, min_coverage=1.0
    )
    assert coverages == {"first": 1.0, "second": 1.0}
    # Python's max on tied keys returns the first encountered.
    assert picked == "first"


# ---------------------------------------------------------------------------
# _build_translator
# ---------------------------------------------------------------------------

def test_build_translator_maps_across_columns():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1"],
            "refseq": ["NC1"],
            "ensembl": ["1"],
        }
    )
    tr = _build_translator(alias_df, canonical_col="ensembl")
    assert tr("chr1") == "1"
    assert tr("NC1") == "1"
    assert tr("1") == "1"


def test_build_translator_returns_none_for_unknown():
    alias_df = pd.DataFrame({"ucsc": ["chr1"], "ensembl": ["1"]})
    tr = _build_translator(alias_df, canonical_col="ensembl")
    assert tr("chrZZ") is None
    assert tr("") is None


def test_build_translator_skips_rows_with_empty_canonical():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2"],
            "ensembl": [np.nan, "2"],
        }
    )
    tr = _build_translator(alias_df, canonical_col="ensembl")
    # row 1 had canon=NaN, so chr1 isn't mapped
    assert tr("chr1") is None
    # row 2 mapped fine
    assert tr("chr2") == "2"
    assert tr("2") == "2"


def test_build_translator_first_row_wins_on_collision():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["dup", "dup"],
            "ensembl": ["first", "second"],
        }
    )
    tr = _build_translator(alias_df, canonical_col="ensembl")
    # First row's canonical wins for the duplicated upstream name.
    assert tr("dup") == "first"
    # Each canonical itself still resolves to itself via the canonical column.
    assert tr("first") == "first"
    assert tr("second") == "second"


# ---------------------------------------------------------------------------
# _resolve_chrom_alias
# ---------------------------------------------------------------------------

def test_resolve_chrom_alias_none_df_returns_identity_fallback():
    tr = _resolve_chrom_alias(None, ["1", "2"], [100, 50])
    assert tr("1") == "1"
    assert tr("2") == "2"
    assert tr("chrA") is None


def test_resolve_chrom_alias_empty_df_returns_identity_fallback():
    tr = _resolve_chrom_alias(pd.DataFrame(), ["1", "2"], [100, 50])
    assert tr("1") == "1"
    assert tr("chrZ") is None


def test_resolve_chrom_alias_picks_and_translates():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2", "chr3"],
            "refseq": ["NC1", "NC2", "NC3"],
            "ensembl": ["1", "2", "3"],
        }
    )
    groot = ["1", "2", "3"]
    lengths = [100, 50, 10]
    tr = _resolve_chrom_alias(alias_df, groot, lengths)
    # ensembl chosen as canonical; all aliases route to it.
    assert tr("chr1") == "1"
    assert tr("NC2") == "2"
    assert tr("3") == "3"
    assert tr("not-there") is None


def test_resolve_chrom_alias_raises_below_min_coverage():
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2"],
            "ensembl": ["1", "2"],
        }
    )
    groot = ["chrA", "chrB"]
    lengths = [100, 50]
    with pytest.raises(ValueError, match="no column with >="):
        _resolve_chrom_alias(alias_df, groot, lengths, min_coverage=1.0)


def test_resolve_chrom_alias_match_by_length_true_runs_rescue():
    # C.3.1: match_by_length=True now triggers the four-pass rescue and
    # returns a translator (instead of raising NotImplementedError).
    alias_df = pd.DataFrame({"ucsc": ["chr1"], "ensembl": ["1"]})
    tr = _resolve_chrom_alias(
        alias_df, ["1"], [100], match_by_length=True
    )
    assert tr("chr1") == "1"
    assert tr("1") == "1"
