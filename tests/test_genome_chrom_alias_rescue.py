"""Tests for the four-pass chromAlias rescue (C.3.1, match_by_length=True).

See `test_genome_chrom_alias.py` for single-pass coverage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pymisha.genome._chrom_alias import (
    _alias_row_lengths,
    _length_column,
    _length_fill,
    _length_override,
    _name_override,
    _resolve_chrom_alias,
    _synthesize_target_chroms,
)

# ---------------------------------------------------------------------------
# _length_column / _alias_row_lengths
# ---------------------------------------------------------------------------

def test_length_column_detection_case_insensitive():
    # First-match-wins: with multiple `length` variants, returns the first.
    df1 = pd.DataFrame({"ucsc": ["chr1"], "Length": [100]})
    assert _length_column(df1) == "Length"

    df2 = pd.DataFrame({"LENGTH": [100], "refseq": ["NC1"]})
    assert _length_column(df2) == "LENGTH"

    df3 = pd.DataFrame({"length": [100]})
    assert _length_column(df3) == "length"

    # `len` is NOT recognized - only exact (case-insensitive) "length".
    df4 = pd.DataFrame({"ucsc": ["chr1"], "len": [100]})
    assert _length_column(df4) is None


def test_alias_row_lengths_returns_none_when_no_length_col():
    df = pd.DataFrame({"ucsc": ["chr1", "chr2"], "ensembl": ["1", "2"]})
    assert _alias_row_lengths(df) is None


def test_alias_row_lengths_coerces_strings_to_numeric():
    df = pd.DataFrame({"length": ["100", "abc", "50"]})
    out = _alias_row_lengths(df)
    assert out is not None
    assert out.iloc[0] == 100
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == 50


# ---------------------------------------------------------------------------
# Pass 1: _length_fill
# ---------------------------------------------------------------------------

def test_length_fill_fills_missing_via_unique_length():
    canonical = pd.Series(["chr1", "", ""], dtype=object)
    lengths = pd.Series([100, 50, 30])
    groot_chroms = ["1", "2", "3"]
    groot_lengths = [100, 50, 30]

    out = _length_fill(canonical, lengths, groot_chroms, groot_lengths)
    assert out.tolist() == ["chr1", "2", "3"]


def test_length_fill_leaves_canonical_alone_when_not_missing():
    canonical = pd.Series(["chr1", "chr2"], dtype=object)
    lengths = pd.Series([100, 50])
    out = _length_fill(canonical, lengths, ["1", "2"], [100, 50])
    assert out.tolist() == ["chr1", "chr2"]


def test_length_fill_does_not_fill_when_length_not_unique_on_one_side():
    # alias row has unique length (50), but groot has TWO chroms at 50.
    canonical = pd.Series(["", ""], dtype=object)
    lengths = pd.Series([50, 30])
    groot_chroms = ["A", "B", "C"]
    groot_lengths = [50, 50, 30]
    out = _length_fill(canonical, lengths, groot_chroms, groot_lengths)
    # 50 is ambiguous on groot side - row 0 stays empty.
    # 30 is unique on both - row 1 gets filled.
    assert out.iloc[0] == ""
    assert out.iloc[1] == "C"


def test_length_fill_handles_nan_alias_length():
    canonical = pd.Series(["", ""], dtype=object)
    lengths = pd.Series([float("nan"), 50])
    out = _length_fill(canonical, lengths, ["1", "2"], [100, 50])
    assert out.iloc[0] == ""
    assert out.iloc[1] == "2"


def test_length_fill_no_length_series_returns_input():
    canonical = pd.Series(["", "chr2"], dtype=object)
    out = _length_fill(canonical, None, ["1", "2"], [100, 50])
    assert out.tolist() == ["", "chr2"]


# ---------------------------------------------------------------------------
# Pass 2: _length_override
# ---------------------------------------------------------------------------

def test_length_override_replaces_misnamed_when_unique_length_match():
    # canonical row 0 says "chrM" but the groot has "AY172581.1" at length 16k.
    canonical = pd.Series(["chrM", "1"], dtype=object)
    lengths = pd.Series([16_000, 100])
    groot_chroms = ["AY172581.1", "1"]
    groot_lengths = [16_000, 100]
    out = _length_override(canonical, lengths, groot_chroms, groot_lengths)
    assert out.tolist() == ["AY172581.1", "1"]


def test_length_override_skips_when_value_already_in_groot():
    canonical = pd.Series(["1", "2"], dtype=object)
    lengths = pd.Series([100, 50])
    out = _length_override(canonical, lengths, ["1", "2"], [100, 50])
    assert out.tolist() == ["1", "2"]


def test_length_override_skips_when_groot_chrom_already_taken():
    # Row 0 canonical="chrFoo" not in groot, length=100.
    # Row 1 already canonical="1" (groot chrom for length 100).
    # Override should NOT replace row 0's "chrFoo" with "1" (collision).
    canonical = pd.Series(["chrFoo", "1"], dtype=object)
    lengths = pd.Series([100, 50])
    groot_chroms = ["1", "2"]
    groot_lengths = [100, 50]
    out = _length_override(canonical, lengths, groot_chroms, groot_lengths)
    assert out.tolist() == ["chrFoo", "1"]


def test_length_override_skips_when_length_ambiguous():
    canonical = pd.Series(["chrX"], dtype=object)
    lengths = pd.Series([100])
    # groot side ambiguous on 100.
    out = _length_override(canonical, lengths, ["A", "B"], [100, 100])
    assert out.tolist() == ["chrX"]


# ---------------------------------------------------------------------------
# Pass 3: _name_override
# ---------------------------------------------------------------------------

def test_name_override_breaks_cross_row_collision():
    # Both rows say canonical="1" but only the row at length=100 is the real
    # match (groot has "1" at length 100). The other row is cleared.
    canonical = pd.Series(["1", "1"], dtype=object)
    lengths = pd.Series([100, 50])
    groot_chroms = ["1", "2"]
    groot_lengths = [100, 50]
    out = _name_override(canonical, lengths, groot_chroms, groot_lengths)
    # row 0 wins (length matches), row 1 cleared.
    assert out.iloc[0] == "1"
    assert out.iloc[1] is None


def test_name_override_leaves_alone_when_no_arbitration_possible():
    # Both colliding rows have the same length, can't decide -> leave alone.
    canonical = pd.Series(["1", "1"], dtype=object)
    lengths = pd.Series([100, 100])
    out = _name_override(canonical, lengths, ["1"], [100])
    assert out.tolist() == ["1", "1"]


def test_name_override_leaves_alone_when_no_groot_length_for_name():
    # canonical says "1" but groot doesn't have "1" at all -> no arbitration.
    canonical = pd.Series(["1", "1"], dtype=object)
    lengths = pd.Series([100, 50])
    out = _name_override(canonical, lengths, ["X"], [200])
    assert out.tolist() == ["1", "1"]


# ---------------------------------------------------------------------------
# Pass 4: _synthesize_target_chroms
# ---------------------------------------------------------------------------

def test_synthesize_target_chroms_appends_for_missing_lengths():
    alias_df = pd.DataFrame(
        {"ucsc": ["chr1"], "ensembl": ["1"], "length": [100]}
    )
    target_chroms = ["1", "2"]  # "1" already present, "2" is new
    target_lengths = [100, 50]
    groot_chroms = ["1", "2"]
    groot_lengths = [100, 50]

    out = _synthesize_target_chroms(
        alias_df, "ensembl", target_chroms, target_lengths,
        groot_chroms, groot_lengths,
    )
    # Expect one appended row carrying "2" as ensembl and 50 as length.
    assert len(out) == 2
    assert out.iloc[1]["ensembl"] == "2"
    assert int(out.iloc[1]["length"]) == 50
    # Other columns blanked.
    assert out.iloc[1]["ucsc"] == ""


def test_synthesize_target_chroms_skips_when_length_not_in_groot():
    alias_df = pd.DataFrame({"ensembl": ["1"], "length": [100]})
    out = _synthesize_target_chroms(
        alias_df, "ensembl", ["NEW"], [999], ["1"], [100],
    )
    # Length 999 isn't in groot - skip.
    assert len(out) == 1


def test_synthesize_target_chroms_requires_aligned_lengths():
    alias_df = pd.DataFrame({"ensembl": ["1"]})
    with pytest.raises(ValueError, match="aligned target_lengths"):
        _synthesize_target_chroms(
            alias_df, "ensembl", ["a", "b"], [100], ["1"], [100],
        )


def test_synthesize_target_chroms_no_op_when_empty():
    alias_df = pd.DataFrame({"ensembl": ["1"]})
    out = _synthesize_target_chroms(
        alias_df, "ensembl", [], [], ["1"], [100],
    )
    assert out.equals(alias_df)


# ---------------------------------------------------------------------------
# _resolve_chrom_alias (match_by_length=True end-to-end)
# ---------------------------------------------------------------------------

def test_resolve_match_by_length_true_recovers_99_to_100():
    # Pre-rescue: column "ensembl" covers 2/3 groot chroms by name. Row 2
    # has an empty ensembl but a unique length match. With match_by_length
    # = True the rescue lifts coverage to 100% and we pass min_coverage=1.0.
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2", "chr3"],
            "ensembl": ["1", "2", ""],
            "length": [100, 50, 30],
        }
    )
    groot_chroms = ["1", "2", "3"]
    groot_lengths = [100, 50, 30]

    # Single-pass: ensembl only covers 100 + 50 = 150 bp / 180 bp = 0.833.
    with pytest.raises(ValueError, match="no column with >="):
        _resolve_chrom_alias(
            alias_df, groot_chroms, groot_lengths,
            min_coverage=1.0, match_by_length=False,
        )

    # Multi-pass: filled and gate passes.
    tr = _resolve_chrom_alias(
        alias_df, groot_chroms, groot_lengths,
        min_coverage=1.0, match_by_length=True,
    )
    assert tr("chr1") == "1"
    assert tr("chr3") == "3"
    assert tr("3") == "3"


def test_resolve_match_by_length_with_target_chroms_synthesizes():
    # The alias frame is missing a chrom present in the groot; target_chroms
    # supplies it and the synthetic row gets through the gate.
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2"],
            "ensembl": ["1", "2"],
            "length": [100, 50],
        }
    )
    groot_chroms = ["1", "2", "3"]
    groot_lengths = [100, 50, 30]

    tr = _resolve_chrom_alias(
        alias_df, groot_chroms, groot_lengths,
        target_chroms=["3"], target_lengths=[30],
        min_coverage=1.0, match_by_length=True,
    )
    assert tr("3") == "3"
    assert tr("chr1") == "1"


def test_resolve_match_by_length_target_lengths_required():
    alias_df = pd.DataFrame({"ensembl": ["1"], "length": [100]})
    with pytest.raises(ValueError, match="target_chroms requires target_lengths"):
        _resolve_chrom_alias(
            alias_df, ["1"], [100],
            target_chroms=["3"], target_lengths=None,
            match_by_length=True,
        )


def test_resolve_match_by_length_target_chroms_required():
    alias_df = pd.DataFrame({"ensembl": ["1"], "length": [100]})
    with pytest.raises(ValueError, match="target_lengths requires target_chroms"):
        _resolve_chrom_alias(
            alias_df, ["1"], [100],
            target_chroms=None, target_lengths=[100],
            match_by_length=True,
        )


def test_resolve_match_by_length_post_rescue_gate_fails():
    # alias has only column "ensembl" matching 1/3 chroms. The unmapped rows
    # don't have a length-fillable match (length is ambiguous on groot side),
    # so post-rescue coverage stays below min_coverage and we raise.
    alias_df = pd.DataFrame(
        {
            "ensembl": ["1", "", ""],
            "length": [100, 50, 50],
        }
    )
    groot_chroms = ["1", "2", "3"]
    groot_lengths = [100, 50, 50]

    with pytest.raises(ValueError, match="post-rescue coverage"):
        _resolve_chrom_alias(
            alias_df, groot_chroms, groot_lengths,
            min_coverage=1.0, match_by_length=True,
        )


def test_resolve_no_alias_df_with_target_chroms_synthesizes_full_table():
    tr = _resolve_chrom_alias(
        None, ["1", "2"], [100, 50],
        target_chroms=["1", "2"], target_lengths=[100, 50],
        match_by_length=True,
    )
    assert tr("1") == "1"
    assert tr("2") == "2"
    assert tr("nope") is None


def test_resolve_match_by_length_no_length_column_falls_through():
    # No length column - rescue passes are no-ops. Coverage gate is still
    # enforced; here single-column ensembl with full coverage passes.
    alias_df = pd.DataFrame(
        {"ucsc": ["chr1", "chr2"], "ensembl": ["1", "2"]}
    )
    tr = _resolve_chrom_alias(
        alias_df, ["1", "2"], [100, 50],
        min_coverage=1.0, match_by_length=True,
    )
    assert tr("chr1") == "1"
    assert tr("2") == "2"


def test_resolve_match_by_length_no_length_column_gate_can_fail():
    alias_df = pd.DataFrame({"ensembl": ["1"]})  # only 1 of 2 groot chroms
    with pytest.raises(ValueError, match="post-rescue coverage"):
        _resolve_chrom_alias(
            alias_df, ["1", "2"], [100, 50],
            min_coverage=1.0, match_by_length=True,
        )


def test_resolve_match_by_length_no_alias_no_target_returns_identity():
    tr = _resolve_chrom_alias(
        None, ["1", "2"], [100, 50],
        match_by_length=True,
    )
    assert tr("1") == "1"
    assert tr("X") is None
