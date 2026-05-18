"""Regression: min_coverage gate must apply AFTER the four-pass rescue chain,
not against the pre-rescue best single-column score.

Ports R commit 537bfe29.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pymisha.genome._chrom_alias import _resolve_chrom_alias

# ---------------------------------------------------------------------------
# Positive: gate passes because final (post-rescue) coverage hits threshold
# ---------------------------------------------------------------------------

def test_min_coverage_applies_after_rescue():
    """Hybrid chromAlias: no single column covers >= 95% of groot bp.

    ucsc covers chr1 (1000 bp = 66.7%), genbank covers chr2 (500 bp = 33.3%).
    The length-fill rescue pass (Pass 1) fills the missing ucsc cell for chr2
    using the unique length pairing, pushing the final canonical coverage to
    100%.  With min_coverage=0.95 the gate must PASS on the post-rescue
    canonical, not the pre-rescue per-column score.

    Before R fix 537bfe29, the gate fired on the best pre-rescue column score
    (66.7%) and rejected this layout even though the rescue chain succeeds.
    """
    # alias_df must have a 'length' column so the length-based rescue fires.
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", None],
            "genbank": [None, "chr2"],
            "length": [1000, 500],
        }
    )
    groot_chroms = ["chr1", "chr2"]
    groot_lengths = [1000, 500]

    # min_coverage=0.95 should PASS (final coverage = 1.0 via length-fill).
    translator = _resolve_chrom_alias(
        alias_df,
        groot_chroms,
        groot_lengths,
        min_coverage=0.95,
        match_by_length=True,
    )
    assert translator is not None
    # The translator must resolve both groot chroms.
    assert translator("chr1") == "chr1"
    assert translator("chr2") == "chr2"


def test_min_coverage_hybrid_three_chroms():
    """Three-chrom hybrid: assembled contigs in 'ucsc', unplaced in 'genbank'.

    - chr1 (1000 bp): ucsc="chr1", genbank=None
    - chr2 (800 bp): ucsc="chr2", genbank=None
    - GL000001 (200 bp): ucsc=None, genbank="GL000001"

    Pre-rescue: ucsc covers 1800/2000 = 90%, genbank 200/2000 = 10%.
    At min_coverage=0.95, ucsc alone fails the gate (90% < 95%).
    After length-fill using the length column, canonical becomes 100%.
    Gate must pass on the post-rescue value.
    """
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2", None],
            "genbank": [None, None, "GL000001"],
            "length": [1000, 800, 200],
        }
    )
    groot_chroms = ["chr1", "chr2", "GL000001"]
    groot_lengths = [1000, 800, 200]

    translator = _resolve_chrom_alias(
        alias_df,
        groot_chroms,
        groot_lengths,
        min_coverage=0.95,
        match_by_length=True,
    )
    assert translator is not None
    assert translator("chr1") == "chr1"
    assert translator("chr2") == "chr2"
    assert translator("GL000001") == "GL000001"


# ---------------------------------------------------------------------------
# Negative: gate fires when final (post-rescue) coverage is still below
# threshold
# ---------------------------------------------------------------------------

def test_min_coverage_fails_when_final_coverage_below_threshold():
    """If the rescue chain cannot bridge to min_coverage, the post-rescue gate
    must still fire.

    alias_df covers chr1 and chr2 but not chr3 or chr4. The length column has
    no unique pairings that rescue chr3/chr4, so final coverage stays at
    2/4 contigs = (1000+800)/(1000+800+600+400) = 60%.  With
    min_coverage=0.95 the resolver must raise.
    """
    # chr3 and chr4 are absent from the alias entirely - no rescue can help.
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", "chr2"],
            "genbank": ["G1", "G2"],
            "length": [1000, 800],
        }
    )
    groot_chroms = ["chr1", "chr2", "chr3", "chr4"]
    groot_lengths = [1000, 800, 600, 400]

    with pytest.raises((ValueError, RuntimeError)):
        _resolve_chrom_alias(
            alias_df,
            groot_chroms,
            groot_lengths,
            min_coverage=0.95,
            match_by_length=True,
        )


def test_min_coverage_single_pass_still_gates_pre_rescue():
    """The strict single-column gate is preserved when match_by_length=False.

    With match_by_length=False (single-pass resolver), no rescue runs and
    the gate applies to the raw column scores. ucsc covers 66.7%, which is
    below 0.95, so the call must raise.
    """
    alias_df = pd.DataFrame(
        {
            "ucsc": ["chr1", None],
            "genbank": [None, "chr2"],
            "length": [1000, 500],
        }
    )
    groot_chroms = ["chr1", "chr2"]
    groot_lengths = [1000, 500]

    with pytest.raises((ValueError, RuntimeError)):
        _resolve_chrom_alias(
            alias_df,
            groot_chroms,
            groot_lengths,
            min_coverage=0.95,
            match_by_length=False,
        )
