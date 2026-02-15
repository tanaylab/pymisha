"""Tests for liftover chain workflow: load_chain, as_chain, liftover."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"
CHAIN_FILE = str(TEST_DB / "data" / "test.chain")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_chain(tmpdir, entries):
    """Write a chain file from a list of (header_dict, blocks) tuples.

    header_dict keys: score, src_chrom, src_size, src_strand, src_start, src_end,
                      tgt_chrom, tgt_size, tgt_strand, tgt_start, tgt_end, chain_id
    blocks: list of tuples (size,) or (size, dt, dq)
    """
    path = os.path.join(tmpdir, "test.chain")
    with open(path, "w") as f:
        for hdr, blocks in entries:
            f.write(
                f"chain {hdr['score']} "
                f"{hdr['src_chrom']} {hdr['src_size']} {hdr['src_strand']} {hdr['src_start']} {hdr['src_end']} "
                f"{hdr['tgt_chrom']} {hdr['tgt_size']} {hdr['tgt_strand']} {hdr['tgt_start']} {hdr['tgt_end']} "
                f"{hdr['chain_id']}\n"
            )
            for blk in blocks:
                if len(blk) == 3:
                    f.write(f"{blk[0]}\t{blk[1]}\t{blk[2]}\n")
                else:
                    f.write(f"{blk[0]}\n")
            f.write("\n")
    return path


# ===================================================================
# gintervals_load_chain
# ===================================================================

class TestLoadChain:
    """Tests for gintervals_load_chain."""

    def test_load_basic(self):
        """Load the example chain file and verify columns/types."""
        chain = pm.gintervals_load_chain(CHAIN_FILE)
        assert isinstance(chain, pd.DataFrame)
        expected_cols = {"chrom", "start", "end", "strand",
                         "chromsrc", "startsrc", "endsrc", "strandsrc",
                         "chain_id", "score"}
        assert expected_cols.issubset(set(chain.columns))
        assert len(chain) > 0

    def test_load_chain_values(self):
        """Verify parsed chain block coordinates match hand-computed values.

        test.chain has:
        chain 200000 chr25 100000 + 2000 8000 chr1 500000 + 12000 18500 1
        500  0  200
        800  300  600
        4400

        Blocks (all + strand, no reversal needed):
        Block 1: src=2000..2500, tgt=12000..12500  (size=500)
          gap: dt=0 => src advances 500+0=2500, dq=200 => tgt advances 500+200=12700
        Block 2: src=2500..3300, tgt=12700..13500  (size=800)
          gap: dt=300 => src advances 800+300=3600, dq=600 => tgt advances 800+600=14100
        Block 3: src=3600..8000, tgt=14100..18500  (size=4400)

        chain 200000 chr25 100000 + 10000 12000 chrX 200000 + 5000 7000 2
        2000
        Block: src=10000..12000, tgt=5000..7000
        """
        chain = pm.gintervals_load_chain(CHAIN_FILE)
        # Chain 1 has 3 blocks, chain 2 has 1 block => 4 rows total
        assert len(chain) == 4

        # Sort by target coordinates for stable comparison
        chain = chain.sort_values(["chrom", "start"]).reset_index(drop=True)

        # Chain 1 blocks (target = chr1)
        chr1_rows = chain[chain["chrom"] == "1"].reset_index(drop=True)
        assert len(chr1_rows) == 3

        # Block 1
        assert chr1_rows.loc[0, "start"] == 12000
        assert chr1_rows.loc[0, "end"] == 12500
        assert chr1_rows.loc[0, "chromsrc"] == "chr25"
        assert chr1_rows.loc[0, "startsrc"] == 2000
        assert chr1_rows.loc[0, "endsrc"] == 2500
        assert chr1_rows.loc[0, "chain_id"] == 1
        assert chr1_rows.loc[0, "score"] == 200000

        # Block 2
        assert chr1_rows.loc[1, "start"] == 12700
        assert chr1_rows.loc[1, "end"] == 13500

        # Block 3
        assert chr1_rows.loc[2, "start"] == 14100
        assert chr1_rows.loc[2, "end"] == 18500

        # Chain 2 (target = chrX)
        chrx_rows = chain[chain["chrom"] == "X"].reset_index(drop=True)
        assert len(chrx_rows) == 1
        assert chrx_rows.loc[0, "start"] == 5000
        assert chrx_rows.loc[0, "end"] == 7000
        assert chrx_rows.loc[0, "chain_id"] == 2

    def test_load_chain_policies_stored(self):
        """Chain DataFrame should carry overlap policy attributes."""
        chain = pm.gintervals_load_chain(CHAIN_FILE)
        assert chain.attrs.get("src_overlap_policy") == "error"
        assert chain.attrs.get("tgt_overlap_policy") == "auto_score"

    def test_load_chain_custom_policies(self):
        """Load with explicit overlap policies."""
        chain = pm.gintervals_load_chain(
            CHAIN_FILE,
            src_overlap_policy="keep",
            tgt_overlap_policy="keep",
        )
        assert chain.attrs.get("src_overlap_policy") == "keep"
        assert chain.attrs.get("tgt_overlap_policy") == "keep"

    def test_load_chain_min_score(self):
        """min_score filters out chains below threshold."""
        # Both chains have score 200000, so filtering at 300000 drops everything
        chain = pm.gintervals_load_chain(CHAIN_FILE, min_score=300000)
        assert len(chain) == 0

        # Filtering at 100000 keeps both
        chain = pm.gintervals_load_chain(CHAIN_FILE, min_score=100000)
        assert len(chain) == 4

    def test_load_chain_empty(self, tmp_path):
        """Loading an empty chain file returns empty DataFrame."""
        path = str(tmp_path / "empty.chain")
        with open(path, "w") as f:
            f.write("")
        chain = pm.gintervals_load_chain(path)
        assert len(chain) == 0
        expected_cols = {"chrom", "start", "end", "strand",
                         "chromsrc", "startsrc", "endsrc", "strandsrc",
                         "chain_id", "score"}
        assert expected_cols.issubset(set(chain.columns))

    def test_load_chain_comments_skipped(self, tmp_path):
        """Comment lines starting with # are skipped."""
        content = (
            "# This is a comment\n"
            "chain 1000 chr25 100000 + 0 100 chr1 500000 + 0 100 1\n"
            "100\n\n"
        )
        path = str(tmp_path / "comment.chain")
        with open(path, "w") as f:
            f.write(content)
        chain = pm.gintervals_load_chain(path)
        assert len(chain) == 1

    def test_load_chain_unknown_tgt_chrom_skipped(self, tmp_path):
        """Chains targeting chromosomes not in DB are silently skipped."""
        content = (
            "chain 1000 chr_other 50000 + 0 100 chr_unknown 50000 + 0 100 1\n"
            "100\n\n"
            "chain 1000 chr25 100000 + 0 500 chr1 500000 + 0 500 2\n"
            "500\n\n"
        )
        path = str(tmp_path / "unknown_tgt.chain")
        with open(path, "w") as f:
            f.write(content)
        chain = pm.gintervals_load_chain(path)
        assert len(chain) == 1
        assert chain.iloc[0]["chrom"] == "1"

    def test_load_chain_invalid_policy(self):
        """Invalid overlap policies raise ValueError."""
        with pytest.raises(ValueError):
            pm.gintervals_load_chain(CHAIN_FILE, src_overlap_policy="invalid")
        with pytest.raises(ValueError):
            pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="invalid")

    def test_load_chain_src_overlap_discard(self, tmp_path):
        """Source overlap discard removes overlapping source intervals."""
        # Two chains with overlapping source regions on same source chrom
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 600,
              "chain_id": 1},
             [(600,)]),
            ({"score": 2000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 500, "src_end": 1100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1600,
              "chain_id": 2},
             [(600,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="discard",
                                         tgt_overlap_policy="keep")
        # Both chains discarded because they overlap in source space
        assert len(chain) == 0

    def test_load_chain_src_overlap_discard_wide_interval(self, tmp_path):
        """Discard policy removes non-adjacent overlaps caused by a wide source interval."""
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 1000,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 1000,
              "chain_id": 1},
             [(1000,)]),
            ({"score": 900, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2100,
              "chain_id": 2},
             [(100,)]),
            ({"score": 800, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 300, "src_end": 400,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 3000, "tgt_end": 3100,
              "chain_id": 3},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="discard",
                                         tgt_overlap_policy="keep")
        assert len(chain) == 0

    def test_load_chain_src_overlap_error(self, tmp_path):
        """Source overlap with error policy raises."""
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 600,
              "chain_id": 1},
             [(600,)]),
            ({"score": 2000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 500, "src_end": 1100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1600,
              "chain_id": 2},
             [(600,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        with pytest.raises(Exception, match="[Oo]verlap|[Ss]ource"):
            pm.gintervals_load_chain(path, src_overlap_policy="error",
                                     tgt_overlap_policy="keep")

    def test_load_chain_tgt_overlap_discard(self, tmp_path):
        """Target overlap discard removes overlapping target intervals."""
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 600,
              "chain_id": 1},
             [(600,)]),
            ({"score": 2000, "src_chrom": "srcB", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 300, "tgt_end": 900,
              "chain_id": 2},
             [(600,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="discard")
        assert len(chain) == 0

    def test_load_chain_tgt_overlap_discard_wide_interval(self, tmp_path):
        """Discard policy removes non-adjacent overlaps caused by a wide target interval."""
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 1000,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 1000,
              "chain_id": 1},
             [(1000,)]),
            ({"score": 900, "src_chrom": "srcB", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
              "chain_id": 2},
             [(100,)]),
            ({"score": 800, "src_chrom": "srcC", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 300, "tgt_end": 400,
              "chain_id": 3},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="discard")
        assert len(chain) == 0

    def test_load_chain_negative_strand(self, tmp_path):
        """Negative strand coordinates are properly flipped."""
        # Chain on negative strand target
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "-", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")
        assert len(chain) == 1
        # Negative strand: start = chromsize - start - size = 500000 - 0 - 100 = 499900
        #                   end = chromsize - start = 500000 - 0 = 500000
        assert chain.iloc[0]["start"] == 499900
        assert chain.iloc[0]["end"] == 500000
        assert chain.iloc[0]["strand"] == 1


# ===================================================================
# gintervals_as_chain
# ===================================================================

class TestAsChain:
    """Tests for gintervals_as_chain."""

    def test_as_chain_basic(self):
        """Convert a DataFrame to chain format."""
        df = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [1000],
            "end": [2000],
            "strand": [0],
            "chromsrc": ["chr25"],
            "startsrc": [5000],
            "endsrc": [6000],
            "strandsrc": [0],
            "chain_id": [1],
            "score": [1000.0],
        })
        chain = pm.gintervals_as_chain(df)
        assert isinstance(chain, pd.DataFrame)
        assert chain.attrs.get("src_overlap_policy") == "error"
        assert chain.attrs.get("tgt_overlap_policy") == "auto_score"

    def test_as_chain_missing_columns(self):
        """Missing required columns raises ValueError."""
        df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        with pytest.raises(ValueError, match="[Mm]issing"):
            pm.gintervals_as_chain(df)

    def test_as_chain_custom_policies(self):
        """Custom policies are stored in attrs."""
        df = pd.DataFrame({
            "chrom": ["chr1"], "start": [0], "end": [100], "strand": [0],
            "chromsrc": ["chr25"], "startsrc": [0], "endsrc": [100],
            "strandsrc": [0], "chain_id": [1], "score": [100.0],
        })
        chain = pm.gintervals_as_chain(
            df, src_overlap_policy="keep", tgt_overlap_policy="keep"
        )
        assert chain.attrs.get("src_overlap_policy") == "keep"
        assert chain.attrs.get("tgt_overlap_policy") == "keep"

    def test_as_chain_not_dataframe(self):
        """Non-DataFrame input raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            pm.gintervals_as_chain("not a dataframe")


# ===================================================================
# gintervals_liftover
# ===================================================================

class TestLiftover:
    """Tests for gintervals_liftover."""

    def test_liftover_basic(self):
        """Basic liftover with pre-loaded chain."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        # Source interval that overlaps chain 1 source (chr25:2000-8000)
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2000],
            "end": [2500],
        })
        result = pm.gintervals_liftover(intervals, chain)
        assert isinstance(result, pd.DataFrame)
        assert "intervalID" in result.columns
        assert "chain_id" in result.columns
        assert len(result) > 0
        # The source interval chr25:2000-2500 maps to chain 1 block 1: chr1:12000-12500
        assert result.iloc[0]["chrom"] == "1"
        assert result.iloc[0]["start"] == 12000
        assert result.iloc[0]["end"] == 12500
        assert result.iloc[0]["intervalID"] == 0

    def test_liftover_from_file(self):
        """Liftover directly from chain file path."""
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2000],
            "end": [2500],
        })
        result = pm.gintervals_liftover(intervals, CHAIN_FILE,
                                        tgt_overlap_policy="keep")
        assert len(result) > 0
        assert result.iloc[0]["chrom"] == "1"

    def test_liftover_no_overlap(self):
        """Interval not overlapping any chain returns empty."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        intervals = pd.DataFrame({
            "chrom": ["chr99"],
            "start": [0],
            "end": [100],
        })
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 0

    def test_liftover_multiple_blocks(self):
        """Source interval spanning multiple chain blocks maps to multiple targets."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        # chr25:2000-8000 spans all 3 blocks of chain 1
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2000],
            "end": [8000],
        })
        result = pm.gintervals_liftover(intervals, chain)
        # Should produce 3 target intervals (one per chain block)
        chain1_results = result[result["chain_id"] == 1]
        assert len(chain1_results) == 3

    def test_liftover_partial_overlap(self):
        """Source interval partially overlapping a chain block maps only the overlap."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        # chr25:2100-2400 overlaps block 1 (src 2000-2500) partially
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2100],
            "end": [2400],
        })
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        # Offset from block start: 2100-2000 = 100, 2400-2000 = 400
        # Target: 12000+100=12100 to 12000+400=12400
        assert result.iloc[0]["start"] == 12100
        assert result.iloc[0]["end"] == 12400

    def test_liftover_canonic(self):
        """Canonic merges adjacent target blocks from same source and chain."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")

        # We need to construct a chain with adjacent target blocks (no gap in target)
        # The test.chain has gaps, so canonic won't merge those.
        # But we can test the non-merge case: canonic=True shouldn't merge
        # blocks that have gaps between them.
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2000],
            "end": [8000],
        })
        result_no_canonic = pm.gintervals_liftover(intervals, chain, canonic=False)
        result_canonic = pm.gintervals_liftover(intervals, chain, canonic=True)

        # Chain 1 blocks have gaps in target (12500..12700, 13500..14100),
        # so canonic should NOT merge them
        chain1_nc = result_no_canonic[result_no_canonic["chain_id"] == 1]
        chain1_c = result_canonic[result_canonic["chain_id"] == 1]
        assert len(chain1_nc) == len(chain1_c)

    def test_liftover_canonic_merges_adjacent(self, tmp_path):
        """Canonic merges adjacent target blocks (no gap in target)."""
        # Create a chain with two adjacent target blocks (gap only in source)
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 1200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 1000,
              "chain_id": 1},
             # Block 1: size=500, then gap dt=200 in source, dq=0 in target
             [(500, 200, 0),
              (500,)]),  # Block 2: size=500
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        intervals = pd.DataFrame({
            "chrom": ["srcA"],
            "start": [0],
            "end": [1200],
        })

        result_no_canonic = pm.gintervals_liftover(intervals, chain, canonic=False)
        result_canonic = pm.gintervals_liftover(intervals, chain, canonic=True)

        # Without canonic: 2 blocks (0-500 and 500-1000)
        assert len(result_no_canonic) == 2
        # With canonic: merged into 1 block (0-1000)
        assert len(result_canonic) == 1
        assert result_canonic.iloc[0]["start"] == 0
        assert result_canonic.iloc[0]["end"] == 1000

    def test_liftover_multiple_chains(self):
        """Source interval mapping through multiple chains."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        # chr25:2000-12000 overlaps both chains
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2000],
            "end": [12000],
        })
        result = pm.gintervals_liftover(intervals, chain)
        chain_ids = result["chain_id"].unique()
        assert 1 in chain_ids
        assert 2 in chain_ids

    def test_liftover_interval_id_tracks_source(self):
        """intervalID corresponds to the 0-based index of the source interval."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        intervals = pd.DataFrame({
            "chrom": ["chr25", "chr25"],
            "start": [2000, 10000],
            "end": [2500, 12000],
        })
        result = pm.gintervals_liftover(intervals, chain)
        # First interval maps via chain 1, second via chain 2
        ids = sorted(result["intervalID"].unique())
        assert ids == [0, 1]

    def test_liftover_include_metadata(self):
        """include_metadata adds score column."""
        chain = pm.gintervals_load_chain(CHAIN_FILE, tgt_overlap_policy="keep")
        intervals = pd.DataFrame({
            "chrom": ["chr25"],
            "start": [2000],
            "end": [2500],
        })
        result = pm.gintervals_liftover(intervals, chain, include_metadata=True)
        assert "score" in result.columns
        assert result.iloc[0]["score"] == 200000

    def test_liftover_value_col(self, tmp_path):
        """value_col preserves values through liftover."""
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 500,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 500,
              "chain_id": 1},
             [(500,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        intervals = pd.DataFrame({
            "chrom": ["srcA"],
            "start": [0],
            "end": [500],
            "value": [42.0],
        })
        result = pm.gintervals_liftover(intervals, chain, value_col="value")
        assert "value" in result.columns
        assert result.iloc[0]["value"] == 42.0

    def test_liftover_tgt_overlap_auto_score(self, tmp_path):
        """auto_score policy selects chain with highest score for overlapping targets."""
        entries = [
            ({"score": 500, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 1000,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 1000,
              "chain_id": 1},
             [(1000,)]),
            ({"score": 2000, "src_chrom": "srcB", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 1000,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 500, "tgt_end": 1500,
              "chain_id": 2},
             [(1000,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="auto_score")

        # After auto_score, overlap region 500-1000 should be assigned to chain 2 (higher score)
        # Chain 1 gets 0-500, chain 2 gets 500-1500
        intervals = pd.DataFrame({
            "chrom": ["srcA", "srcB"],
            "start": [0, 0],
            "end": [1000, 1000],
        })
        result = pm.gintervals_liftover(intervals, chain)
        # Chain 1 should be trimmed to 0-500 in target
        chain1_result = result[result["chain_id"] == 1]
        if len(chain1_result) > 0:
            assert chain1_result.iloc[0]["end"] <= 500


# ===================================================================
# Ported R parity tests — overlap policies, aggregation, canonic
# ===================================================================

class TestLiftoverOverlapPoliciesParity:
    """R parity tests for overlap policies during load_chain and liftover.

    Ported from test-liftover.R.
    """

    def test_src_overlap_keep_liftover_multi_target(self, tmp_path):
        """Source overlap keep policy produces multiple target mappings.

        R: 'gintervals.liftover works with keep source policy'
        Two chains overlap in source space on source1[10,20).
        Chain 1: source1[0-20] -> chr1[0-20]
        Chain 2: source1[10-26] -> chrX[0-16]
        With keep policy, liftover of source1[10,20) should map to both.
        """
        entries = [
            ({"score": 1000, "src_chrom": "source1", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
            ({"score": 2000, "src_chrom": "source1", "src_size": 100,
              "src_strand": "+", "src_start": 10, "src_end": 26,
              "tgt_chrom": "chrX", "tgt_size": 200000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 16,
              "chain_id": 2},
             [(16,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="keep")

        src = pd.DataFrame({"chrom": ["source1"], "start": [10], "end": [20]})
        result = pm.gintervals_liftover(src, chain)

        assert len(result) >= 2
        chroms = set(result["chrom"].tolist())
        assert "1" in chroms
        assert "X" in chroms

        # chr1 mapping: source[10,20) -> target[10,20) (offset within chain1)
        chr1_rows = result[result["chrom"] == "1"]
        assert chr1_rows.iloc[0]["start"] == 10
        assert chr1_rows.iloc[0]["end"] == 20

        # chrX mapping: source[10,20) overlaps chain2 source[10,26) -> target[0,10)
        chrx_rows = result[result["chrom"] == "X"]
        assert chrx_rows.iloc[0]["start"] == 0
        assert chrx_rows.iloc[0]["end"] == 10

    def test_tgt_overlap_error_raises(self, tmp_path):
        """Target overlap with error policy raises exception.

        R: 'gintervals.load_chain handles target overlaps with error policy'
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 600,
              "chain_id": 1},
             [(600,)]),
            ({"score": 2000, "src_chrom": "srcB", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 500,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 300, "tgt_end": 800,
              "chain_id": 2},
             [(500,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        with pytest.raises(Exception, match="[Oo]verlap|[Tt]arget"):
            pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                     tgt_overlap_policy="error")

    def test_tgt_overlap_discard_keeps_clean(self, tmp_path):
        """Target overlap discard removes overlapping pairs, keeps clean ones.

        R: 'gintervals.load_chain handles target overlaps with discard policy'
        Chains 1 and 2 overlap on chr1; chain 3 has no overlap on chrX.
        Discard should remove chr1 entries and keep chrX.
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 600,
              "chain_id": 1},
             [(600,)]),
            ({"score": 2000, "src_chrom": "srcB", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 500,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 300, "tgt_end": 800,
              "chain_id": 2},
             [(500,)]),
            ({"score": 500, "src_chrom": "srcC", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 200,
              "tgt_chrom": "chrX", "tgt_size": 200000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1200,
              "chain_id": 3},
             [(200,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="discard")
        assert len(chain) == 1
        assert chain.iloc[0]["chrom"] == "X"

    def test_auto_first_exact_truncation(self, tmp_path):
        """auto_first truncates second chain at overlap boundary.

        R: 'gintervals.load_chain exact truncation with auto_first policy'
        Chain 1: srcA[0-30] -> chr1[0-30], chain 2: srcB[0-20] -> chr1[20-40].
        After auto_first: chain 1 keeps [0,30), chain 2 trimmed to [30,40).
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 30,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
              "chain_id": 1},
             [(30,)]),
            ({"score": 500, "src_chrom": "srcB", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 20, "tgt_end": 40,
              "chain_id": 2},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="auto_first")
        chr1 = chain[chain["chrom"] == "1"].sort_values("start").reset_index(drop=True)
        assert len(chr1) == 2
        assert chr1.loc[0, "start"] == 0
        assert chr1.loc[0, "end"] == 30
        assert chr1.loc[1, "start"] == 30
        assert chr1.loc[1, "end"] == 40

    def test_auto_score_segmentation(self, tmp_path):
        """auto_score assigns overlap segments to highest-score chain.

        R: 'gintervals.load_chain segments overlaps for auto and agg policies'
        Three overlapping chains with scores 10, 20, 30.
        Segments: [10,15) from chain 1, [15,20) from chain 2, [20,40) from chain 3.
        """
        entries = [
            ({"score": 10, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 10, "tgt_end": 30,
              "chain_id": 1},
             [(20,)]),
            ({"score": 20, "src_chrom": "srcB", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 15, "tgt_end": 35,
              "chain_id": 2},
             [(20,)]),
            ({"score": 30, "src_chrom": "srcC", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 20, "tgt_end": 40,
              "chain_id": 3},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="auto_score")
        chr1 = chain[chain["chrom"] == "1"].sort_values("start").reset_index(drop=True)
        assert list(chr1["start"]) == [10, 15, 20]
        assert list(chr1["end"]) == [15, 20, 40]
        assert list(chr1["chain_id"]) == [1, 2, 3]

    def test_agg_segmentation_creates_duplicates(self, tmp_path):
        """agg policy creates one row per (chain_id, segment) pair.

        R: 'gintervals.load_chain segments overlaps for auto and agg policies'
        Three overlapping chains produce 9 segment rows.
        """
        entries = [
            ({"score": 10, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 10, "tgt_end": 30,
              "chain_id": 1},
             [(20,)]),
            ({"score": 20, "src_chrom": "srcB", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 15, "tgt_end": 35,
              "chain_id": 2},
             [(20,)]),
            ({"score": 30, "src_chrom": "srcC", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 20, "tgt_end": 40,
              "chain_id": 3},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="agg")
        chr1 = chain[chain["chrom"] == "1"].sort_values(
            ["start", "chain_id"]).reset_index(drop=True)
        # Expected 9 segments: each breakpoint pair x contributing chains
        assert len(chr1) == 9
        assert list(chr1["start"]) == [10, 15, 15, 20, 20, 20, 30, 30, 35]
        assert list(chr1["end"]) == [15, 20, 20, 30, 30, 30, 35, 35, 40]
        assert list(chr1["chain_id"]) == [1, 1, 2, 1, 2, 3, 2, 3, 3]

        # Verify startsrc shifts for chain 1 segments
        chain1_segs = chr1[chr1["chain_id"] == 1].reset_index(drop=True)
        assert list(chain1_segs["startsrc"]) == [0, 5, 10]
        # Verify startsrc shifts for chain 3 segments
        chain3_segs = chr1[chr1["chain_id"] == 3].reset_index(drop=True)
        assert list(chain3_segs["startsrc"]) == [0, 10, 15]


class TestLiftoverValueColParity:
    """R parity tests for value_col pass-through and agg chain behavior.

    Ported from test-gintervals.liftover-agg.R.
    In pymisha, gintervals_liftover passes value_col through without
    aggregation. Aggregation is done via agg policy in load_chain or
    in gtrack_liftover. These tests verify the pass-through and agg
    chain interaction.
    """

    def test_value_col_passthrough_basic(self, tmp_path):
        """value_col carries values through liftover mapping.

        R: 'gintervals.liftover multi-target aggregation policies' (basic)
        Source intervals with values map through 1:1 chain.
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 400,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA", "srcA"],
            "start": [0, 5],
            "end": [10, 15],
            "value": [1.0, 3.0],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
            value_col="value",
        )
        assert "value" in result.columns
        assert len(result) == 2
        result = result.sort_values("start").reset_index(drop=True)
        # Values are passed through from source intervals
        assert result.loc[0, "value"] == 1.0
        assert result.loc[1, "value"] == 3.0

    def test_value_col_nan_passthrough(self, tmp_path):
        """NaN values in value_col are passed through.

        R: 'gintervals.liftover multi-target aggregation policies' (NA handling)
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 400,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA", "srcA", "srcA"],
            "start": [0, 10, 20],
            "end": [10, 20, 30],
            "value": [1.0, float('nan'), 3.0],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
            value_col="value",
        )
        # All 3 intervals overlap source [0-20), but only first 2 are within
        # Actually src[20,30) does NOT overlap chain src[0,20) -> only 2 results
        assert "value" in result.columns
        result = result.sort_values("start").reset_index(drop=True)
        assert result.loc[0, "value"] == 1.0
        assert np.isnan(result.loc[1, "value"])

    def test_agg_chain_creates_segments(self, tmp_path):
        """agg chain policy creates segments at overlap breakpoints.

        R: 'gintervals.load_chain segments overlaps for auto and agg policies'
        Verify the chain-level segmentation is correct before liftover.
        """
        entries = [
            ({"score": 100, "src_chrom": "srcA", "src_size": 300,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 50, "tgt_end": 150,
              "chain_id": 1},
             [(100,)]),
            ({"score": 95, "src_chrom": "srcB", "src_size": 300,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 50, "tgt_end": 150,
              "chain_id": 2},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="keep",
                                         tgt_overlap_policy="agg")
        # agg creates duplicated entries (one per chain contributing to each segment)
        assert len(chain) == 2
        assert all(chain["chrom"] == "1")

    def test_value_col_multiple_chains(self, tmp_path):
        """value_col passes through when interval maps via multiple chains.

        R: 'gintervals.liftover aggregation preserves intervalID and chain_id'
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 10,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 110,
              "chain_id": 1},
             [(10,)]),
            ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 10, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 110, "tgt_end": 120,
              "chain_id": 2},
             [(10,)]),
            ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 20, "src_end": 30,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 120, "tgt_end": 130,
              "chain_id": 3},
             [(10,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA", "srcA", "srcA"],
            "start": [0, 10, 20],
            "end": [10, 20, 30],
            "score": [10.0, 20.0, 30.0],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
            value_col="score",
        )
        assert len(result) == 3
        assert "score" in result.columns
        assert "intervalID" in result.columns
        assert "chain_id" in result.columns
        assert sorted(result["score"].tolist()) == [10.0, 20.0, 30.0]
        assert sorted(result["intervalID"].tolist()) == [0, 1, 2]
        assert sorted(result["chain_id"].tolist()) == [1, 2, 3]

    def test_value_col_integer_type(self, tmp_path):
        """Integer value columns are preserved through liftover.

        R: 'gintervals.liftover aggregation with integer values'
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 30,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
              "chain_id": 1},
             [(30,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA", "srcA", "srcA"],
            "start": [0, 10, 20],
            "end": [10, 20, 30],
            "count": [5, 10, 15],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
            value_col="count",
        )
        assert "count" in result.columns
        assert sorted(result["count"].tolist()) == [5, 10, 15]

    def test_no_value_col_works_as_before(self, tmp_path):
        """Without value_col, liftover works as before (no value column added).

        R: 'gintervals.liftover without value_col works as before'
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA", "srcA"],
            "start": [0, 10],
            "end": [10, 20],
            "extra": ["a", "b"],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
        )
        assert "intervalID" in result.columns
        assert "chain_id" in result.columns
        assert "value" not in result.columns
        assert len(result) == 2

    def test_value_col_float_precision(self, tmp_path):
        """Float value columns maintain precision through liftover.

        R: 'gintervals.liftover aggregation handles multiple value types'
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 10,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 10,
              "chain_id": 1},
             [(10,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA"],
            "start": [0],
            "end": [10],
            "val": [3.14],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
            value_col="val",
        )
        assert "val" in result.columns
        np.testing.assert_allclose(result.iloc[0]["val"], 3.14, rtol=1e-6)


class TestLiftoverCanonicParity:
    """R parity tests for canonic merging behavior.

    Ported from test-gintervals.liftover-canonic.R.
    """

    def test_canonic_merges_adjacent_from_gap_chain(self, tmp_path):
        """Canonic merges adjacent target blocks from same chain with source gap.

        R: 'gintervals.liftover with canonic merges adjacent blocks from same interval'
        Chain with source gap but 0 target gap => adjacent target blocks [0,50) and [50,100).
        Without canonic: 2 rows. With canonic: merged to [0,100).
        """
        path = os.path.join(str(tmp_path), "test.chain")
        with open(path, "w") as f:
            # Chain: source1[0-200] -> chr1[0-100] with 100bp source gap, 0bp target gap
            f.write("chain 1000 source1 300 + 0 200 chr1 500000 + 0 100 1\n")
            f.write("50\t100\t0\n")  # block 50, gap: 100 in source, 0 in target
            f.write("50\n\n")

        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")
        src = pd.DataFrame({"chrom": ["source1"], "start": [0], "end": [200]})

        result_nc = pm.gintervals_liftover(src, chain, canonic=False)
        result_c = pm.gintervals_liftover(src, chain, canonic=True)

        # Without canonic: 2 adjacent intervals
        assert len(result_nc) == 2
        result_nc = result_nc.sort_values("start").reset_index(drop=True)
        assert list(result_nc["start"]) == [0, 50]
        assert list(result_nc["end"]) == [50, 100]
        assert result_nc["intervalID"].nunique() == 1  # same source

        # With canonic: 1 merged interval
        assert len(result_c) == 1
        assert result_c.iloc[0]["start"] == 0
        assert result_c.iloc[0]["end"] == 100

    def test_canonic_no_merge_different_interval_ids(self, tmp_path):
        """Canonic does not merge intervals from different source intervals.

        R: 'gintervals.liftover canonic does not merge intervals from different source intervals'
        Two source intervals map to adjacent targets via 1:1 chain.
        Canonic should keep them separate (different intervalIDs).
        """
        entries = [
            ({"score": 1000, "src_chrom": "source1", "src_size": 300,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        src = pd.DataFrame({
            "chrom": ["source1", "source1"],
            "start": [0, 50],
            "end": [50, 100],
        })

        result = pm.gintervals_liftover(src, chain, canonic=True)
        assert len(result) == 2  # NOT merged
        result = result.sort_values("start").reset_index(drop=True)
        assert list(result["start"]) == [0, 50]
        assert list(result["end"]) == [50, 100]

    def test_canonic_no_merge_non_adjacent(self, tmp_path):
        """Canonic does not merge non-adjacent target blocks.

        R: 'gintervals.liftover canonic handles non-adjacent blocks correctly'
        Chain with gaps in both source and target -> target blocks [0,50) and [100,150).
        These are NOT adjacent so canonic should not merge them.
        """
        path = os.path.join(str(tmp_path), "test.chain")
        with open(path, "w") as f:
            # Gap: 100 in source, 50 in target -> non-adjacent target blocks
            f.write("chain 1000 source1 300 + 0 200 chr1 500000 + 0 150 1\n")
            f.write("50\t100\t50\n")
            f.write("50\n\n")

        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")
        src = pd.DataFrame({"chrom": ["source1"], "start": [0], "end": [200]})

        result = pm.gintervals_liftover(src, chain, canonic=True)
        assert len(result) == 2  # NOT merged
        result = result.sort_values("start").reset_index(drop=True)
        assert list(result["start"]) == [0, 100]
        assert list(result["end"]) == [50, 150]

    def test_canonic_merges_three_adjacent_blocks(self, tmp_path):
        """Canonic merges three adjacent target blocks into one.

        R: 'gintervals.liftover canonic merges multiple adjacent blocks'
        Chain with 3 blocks, all adjacent in target.
        """
        path = os.path.join(str(tmp_path), "test.chain")
        with open(path, "w") as f:
            f.write("chain 1000 source1 500 + 0 350 chr1 500000 + 0 150 1\n")
            f.write("50\t100\t0\n")  # gap: 100 src, 0 tgt
            f.write("50\t100\t0\n")  # gap: 100 src, 0 tgt
            f.write("50\n\n")

        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")
        src = pd.DataFrame({"chrom": ["source1"], "start": [0], "end": [350]})

        result_nc = pm.gintervals_liftover(src, chain, canonic=False)
        result_c = pm.gintervals_liftover(src, chain, canonic=True)

        assert len(result_nc) == 3
        assert len(result_c) == 1
        assert result_c.iloc[0]["start"] == 0
        assert result_c.iloc[0]["end"] == 150


class TestLiftoverMappingParity:
    """R parity tests for exact mapping coordinates and edge cases.

    Ported from test-liftover.R.
    """

    def test_basic_1d_exact_coordinates(self, tmp_path):
        """Basic 1D mapping with exact coordinate verification.

        R: 'gintervals.liftover basic 1D mapping with exact coordinates'
        sourceA[0-20) -> chr1[10-30). Source [5,20) maps to [15,30).
        """
        entries = [
            ({"score": 1000, "src_chrom": "sourceA", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 10, "tgt_end": 30,
              "chain_id": 1},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({"chrom": ["sourceA"], "start": [5], "end": [20]})
        result = pm.gintervals_liftover(src, path, tgt_overlap_policy="keep")

        assert len(result) == 1
        assert result.iloc[0]["chrom"] == "1"
        assert result.iloc[0]["start"] == 15
        assert result.iloc[0]["end"] == 30
        assert result.iloc[0]["intervalID"] == 0

    def test_liftover_chain_with_gaps_splits_interval(self, tmp_path):
        """Interval spanning a chain gap produces 2 target intervals.

        R: 'gintervals.liftover matches liftOver binary - complex chain with gaps'
        Chain: source1[0-27] -> chr1[0-28] with gap (dt=2, dq=3).
        Block 1: 10bp (src[0-10) -> tgt[0-10))
        Gap: 2 in src, 3 in tgt
        Block 2: 15bp (src[12-27) -> tgt[13-28))
        Source [2,8) fits in block 1 -> [2,8). Source [15,23) fits in block 2 -> [18,26).
        """
        path = os.path.join(str(tmp_path), "test.chain")
        with open(path, "w") as f:
            f.write("chain 1000 source1 100 + 0 27 chr1 500000 + 0 28 1\n")
            f.write("10\t2\t3\n")
            f.write("15\n\n")

        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        src = pd.DataFrame({
            "chrom": ["source1", "source1"],
            "start": [2, 15],
            "end": [8, 23],
        })
        result = pm.gintervals_liftover(src, chain)
        result = result.sort_values("start").reset_index(drop=True)

        assert len(result) == 2
        # First interval in block 1: offset 2-8 from src_start=0, tgt_start=0
        assert result.loc[0, "start"] == 2
        assert result.loc[0, "end"] == 8
        # Second interval in block 2: src offset=15-12=3 from block2 src start,
        # tgt_start for block2=13, so 13+3=16 to 13+(23-12)=13+11=24
        assert result.loc[1, "start"] == 16
        assert result.loc[1, "end"] == 24

    def test_liftover_reverse_strand(self, tmp_path):
        """Negative strand target coordinates are properly flipped.

        R: 'gintervals.load_chain handles reverse strands correctly'
        + 'gintervals.load_chain handles mixed strand chains'
        """
        entries = [
            ({"score": 1000, "src_chrom": "source1", "src_size": 100,
              "src_strand": "-", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        assert len(chain) == 1
        # Target is forward -> strand=0 (forward in pymisha convention)
        assert chain.iloc[0]["strand"] == 0
        # Source is reverse -> strandsrc should indicate reverse
        assert chain.iloc[0]["strandsrc"] == 1  # 1 = reverse in pymisha

    def test_liftover_mixed_strands(self, tmp_path):
        """Mixed strand chains are parsed correctly.

        R: 'gintervals.load_chain handles mixed strand chains'
        Chain 1: +/+ (source forward, target forward)
        Chain 2: +/- (source forward, target reverse)
        """
        entries = [
            ({"score": 1000, "src_chrom": "source1", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
            ({"score": 1000, "src_chrom": "source2", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 16,
              "tgt_chrom": "chrX", "tgt_size": 200000,
              "tgt_strand": "-", "tgt_start": 0, "tgt_end": 16,
              "chain_id": 2},
             [(16,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        assert len(chain) == 2
        chr1_chain = chain[chain["chrom"] == "1"].iloc[0]
        chrx_chain = chain[chain["chrom"] == "X"].iloc[0]

        # chr1: +/+ both forward
        assert chr1_chain["strand"] == 0
        assert chr1_chain["strandsrc"] == 0

        # chrX: target is reverse strand -> flipped coordinates
        assert chrx_chain["strand"] == 1  # reverse

    def test_liftover_value_col_preserves_through_mapping(self, tmp_path):
        """Value column is preserved through liftover mapping.

        R: 'gintervals.liftover aggregation preserves intervalID and chain_id'
        Three intervals with score values map 1:1 to non-overlapping targets.
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 0, "src_end": 10,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 110,
              "chain_id": 1},
             [(10,)]),
            ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 10, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 110, "tgt_end": 120,
              "chain_id": 2},
             [(10,)]),
            ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
              "src_strand": "+", "src_start": 20, "src_end": 30,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 120, "tgt_end": 130,
              "chain_id": 3},
             [(10,)]),
        ]
        path = _write_chain(str(tmp_path), entries)

        src = pd.DataFrame({
            "chrom": ["srcA", "srcA", "srcA"],
            "start": [0, 10, 20],
            "end": [10, 20, 30],
            "score": [10.0, 20.0, 30.0],
        })

        result = pm.gintervals_liftover(
            src, path, src_overlap_policy="keep", tgt_overlap_policy="keep",
            value_col="score", multi_target_agg="sum",
        )
        assert len(result) == 3
        assert "score" in result.columns
        assert "intervalID" in result.columns
        assert "chain_id" in result.columns
        assert sorted(result["score"].tolist()) == [10.0, 20.0, 30.0]
        assert sorted(result["intervalID"].tolist()) == [0, 1, 2]
        assert sorted(result["chain_id"].tolist()) == [1, 2, 3]

    def test_liftover_discard_yields_empty(self, tmp_path):
        """Discard policy with all overlapping chains yields empty result.

        R: 'gintervals.liftover returns empty result when all intervals discarded'
        """
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 500,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 500,
              "chain_id": 1},
             [(500,)]),
            ({"score": 2000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 400, "src_end": 900,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 200, "tgt_end": 700,
              "chain_id": 2},
             [(500,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, src_overlap_policy="discard",
                                         tgt_overlap_policy="keep")
        assert len(chain) == 0

    def test_liftover_10_column_format(self, tmp_path):
        """Chain DataFrame has 10 columns with strand, chain_id, and score.

        R: 'gintervals.load_chain returns 10 columns with strand information, chain_id and score'
        """
        entries = [
            ({"score": 5000, "src_chrom": "source1", "src_size": 100,
              "src_strand": "+", "src_start": 0, "src_end": 20,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
              "chain_id": 1},
             [(20,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(path, tgt_overlap_policy="keep")

        expected_cols = ["chrom", "start", "end", "strand",
                         "chromsrc", "startsrc", "endsrc", "strandsrc",
                         "chain_id", "score"]
        assert list(chain.columns) == expected_cols
        assert len(chain) == 1
        assert chain.iloc[0]["startsrc"] == 0
        assert chain.iloc[0]["endsrc"] == 20
        assert chain.iloc[0]["score"] == 5000


# ===================================================================
# best_source_cluster / best_cluster_* policies
# Ported from R test-liftover-best_source_cluster.R
# ===================================================================

# The best_source_cluster family of policies clusters liftover mappings
# by source overlap:
# - If source chains overlap (duplication): keep all mappings in the cluster.
# - If source chains are disjoint (conflict): keep only the cluster with
#   the largest total target length (mass).
#
# The resolution happens *after* liftover mapping, not during chain loading.
# During loading these policies behave like "keep" -- all chains are retained.
# The cluster resolution must be applied on the liftover result.
#
# Cluster resolution is applied after liftover mapping. These tests verify
# the behavior for the alias and variant policies.


class TestBestSourceCluster:
    """Tests for best_source_cluster and related cluster policies.

    Ported from R test-liftover-best_source_cluster.R.
    """

    def test_best_source_cluster_policy_accepted(self, tmp_path):
        """Chain loading accepts best_source_cluster policy without error."""
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        assert chain.attrs.get("tgt_overlap_policy") == "best_source_cluster"
        assert len(chain) == 1

    def test_duplication_retains_both(self, tmp_path):
        """Overlapping source chains (duplication) retain all mappings.

        R: 'best_source_cluster policy retains duplications (overlapping source chains)'
        Chain A: src[0-100] -> chr1[0-100]
        Chain B: src[0-100] -> chr1[200-300] (same source = duplication)
        Both should be retained. (Works with current "keep" behavior.)
        """
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 100},
             [(100,)]),
            ({"score": 50, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 200, "tgt_end": 300,
              "chain_id": 50},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [100]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 2
        assert 0 in result["start"].values
        assert 200 in result["start"].values

    def test_disjoint_selects_larger_mass(self, tmp_path):
        """Disjoint source chains select cluster with largest target mass.

        R: 'best_source_cluster policy selects best cluster for disjoint source chains'
        Cluster 1: src[0-50] -> chr1[0-50] (mass 50)
        Cluster 2: src[100-200] -> chr1[200-300] (mass 100) - larger
        Should keep only cluster 2.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 50,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 50,
              "chain_id": 100},
             [(50,)]),
            ({"score": 500, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 200, "tgt_end": 300,
              "chain_id": 50},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [200]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["start"] == 200
        assert result.iloc[0]["end"] == 300

    def test_overlapping_cluster_beats_disjoint(self, tmp_path):
        """Cluster with overlapping sources (mass 200) beats single disjoint chain (mass 50).

        R: 'best_source_cluster policy handles multiple overlapping chains in winning cluster'
        Cluster 1: A src[0-100]->chr1[0-100] + B src[50-150]->chr1[200-300] (overlap, mass=200)
        Cluster 2: C src[500-550]->chr1[500-550] (mass=50)
        Cluster 1 wins and both A and B are retained.
        """
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 100},
             [(100,)]),
            ({"score": 90, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 50, "src_end": 150,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 200, "tgt_end": 300,
              "chain_id": 90},
             [(100,)]),
            ({"score": 80, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 500, "src_end": 550,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 500, "tgt_end": 550,
              "chain_id": 80},
             [(50,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [600]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 2
        assert set(result["start"].tolist()) == {0, 200}

    def test_single_mapping_passthrough(self, tmp_path):
        """Single chain works without error regardless of policy.

        R: 'best_source_cluster policy works with single mapping'
        """
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 100},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [100]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["start"] == 0
        assert result.iloc[0]["end"] == 100

    def test_adjacent_non_overlapping_separate_clusters(self, tmp_path):
        """Adjacent (touching) source chains are separate clusters, not merged.

        R: 'best_source_cluster policy handles adjacent but non-overlapping sources correctly'
        Chain A: src[0-100] -> chr1[0-100]
        Chain B: src[100-200] -> chr1[200-300] (starts where A ends -- not overlapping)
        Equal mass -> first cluster (by source position) wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 200, "tgt_end": 300,
              "chain_id": 2},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [200]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1

    def test_mass_tiebreaker_first_wins(self, tmp_path):
        """Equal mass: first cluster by source position wins.

        R: 'best_source_cluster policy correctly handles mass tie-breaker'
        Cluster 1: src[0-100] -> chr1[0-100] (mass 100)
        Cluster 2: src[500-600] -> chr1[500-600] (mass 100)
        First cluster wins.
        """
        entries = [
            ({"score": 200, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 500, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 500, "tgt_end": 600,
              "chain_id": 2},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [700]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["start"] == 0

    def test_partial_overlap_larger_mass_wins(self, tmp_path):
        """Partially overlapping intervals: cluster with larger mapped mass wins.

        R: 'best_source_cluster policy works with intervals that don't span full chain regions'
        Chain A: src[0-100] -> chr1[0-100]
        Chain B: src[200-400] -> chr1[300-500]
        Input src[50,300]: maps A -> [50,100) mass 50, maps B -> [300,400) mass 100.
        Cluster B wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 200, "src_end": 400,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 300, "tgt_end": 500,
              "chain_id": 2},
             [(200,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [50], "end": [300]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["start"] == 300
        assert result.iloc[0]["end"] == 400

    def test_duplication_different_chromosomes(self, tmp_path):
        """Same source maps to different target chromosomes (duplication).

        R: 'best_source_cluster scenario 1: duplication with same source to different chromosomes'
        src[100-200] -> chr1[100-200] and src[100-200] -> chrX[100-200].
        Overlapping source -> retain both. (Works with current "keep" behavior.)
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chrX", "tgt_size": 200000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
              "chain_id": 2},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [100], "end": [200]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 2
        chroms = set(result["chrom"].tolist())
        assert "1" in chroms
        assert "X" in chroms

    def test_adjacent_disjoint_higher_mass_wins(self, tmp_path):
        """Adjacent disjoint chains: higher mass cluster wins.

        R: 'best_source_cluster scenario 2: adjacent disjoint chains choose higher mass'
        src[100-130] -> chr1[100-130] (mass 30)
        src[130-200] -> chr2[130-200] (mass 70)
        Cluster 2 wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 100, "src_end": 130,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 130,
              "chain_id": 1},
             [(30,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 130, "src_end": 200,
              "tgt_chrom": "chr2", "tgt_size": 300000,
              "tgt_strand": "+", "tgt_start": 130, "tgt_end": 200,
              "chain_id": 2},
             [(70,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [100], "end": [200]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["chrom"] == "2"
        assert result.iloc[0]["start"] == 130
        assert result.iloc[0]["end"] == 200

    def test_overlapping_sources_retain_both(self, tmp_path):
        """Overlapping source chains retain both mappings.

        R: 'best_source_cluster scenario 3: overlapping source chains retain both'
        src[100-200] -> chr1[100-200]
        src[120-220] -> chrX[100-200] (overlaps in source 120-200)
        Both retained due to source overlap. (Works with current "keep" behavior.)
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 120, "src_end": 220,
              "tgt_chrom": "chrX", "tgt_size": 200000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
              "chain_id": 2},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [100], "end": [200]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 2
        assert any(result["chrom"] == "1")
        assert any(result["chrom"] == "X")

    def test_transitive_overlap_bridge(self, tmp_path):
        """Transitive overlap: A-B-C (bridge) outweighs disjoint D.

        R: 'best_source_cluster scenario 6: Transitive overlap (Bridge) A-B-C outweighs D'
        Chain A: src[100-200] -> chr1[1000-1100]
        Chain B: src[150-250] -> chr1[1200-1300] (overlaps A)
        Chain C: src[240-440] -> chr1[1400-1600] (overlaps B, not A)
        Chain D: src[500-750] -> chr1[2000-2250] (disjoint)
        Union(A-B-C) = 340 > D = 250 -> A-B-C wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 150, "src_end": 250,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1200, "tgt_end": 1300,
              "chain_id": 2},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 240, "src_end": 440,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1400, "tgt_end": 1600,
              "chain_id": 3},
             [(200,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 500, "src_end": 750,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2250,
              "chain_id": 4},
             [(250,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 3
        result_ids = sorted(result["chain_id"].tolist())
        assert result_ids == [1, 2, 3]
        assert 4 not in result["chain_id"].values

    def test_tie_first_source_position_wins(self, tmp_path):
        """Tie-breaking: first source position wins.

        R: 'best_source_cluster scenario 8: Tie-breaking (First source position wins)'
        Chain A: src[100-200] mass 100
        Chain B: src[500-600] mass 100 (same mass, different position)
        First cluster wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 2000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 2000,
              "src_strand": "+", "src_start": 500, "src_end": 600,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 500, "tgt_end": 600,
              "chain_id": 2},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["chain_id"] == 1

    def test_disjoint_cluster_outweighs_overlapping_pair(self, tmp_path):
        """Larger disjoint cluster beats smaller overlapping pair.

        R: 'best_source_cluster scenario 5: C (disjoint) outweighs A+B (overlapping)'
        Cluster 1: A src[100-200] + B src[150-250] (union 150)
        Cluster 2: C src[500-750] (mass 250)
        C wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 150, "src_end": 250,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1200, "tgt_end": 1300,
              "chain_id": 2},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 500, "src_end": 750,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2250,
              "chain_id": 3},
             [(250,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["chain_id"] == 3
        assert result.iloc[0]["start"] == 2000
        assert result.iloc[0]["end"] == 2250


class TestBestClusterVariants:
    """Tests for best_cluster_union, best_cluster_sum, best_cluster_max.

    Ported from R test-liftover-best_source_cluster.R (scenarios for the three
    clustering strategies).
    """

    def test_best_cluster_union_policy_accepted(self, tmp_path):
        """Chain loading accepts best_cluster_union policy."""
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_union",
        )
        assert chain.attrs.get("tgt_overlap_policy") == "best_cluster_union"

    def test_best_cluster_sum_policy_accepted(self, tmp_path):
        """Chain loading accepts best_cluster_sum policy."""
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_sum",
        )
        assert chain.attrs.get("tgt_overlap_policy") == "best_cluster_sum"

    def test_best_cluster_max_policy_accepted(self, tmp_path):
        """Chain loading accepts best_cluster_max policy."""
        entries = [
            ({"score": 100, "src_chrom": "src", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
              "chain_id": 1},
             [(100,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_max",
        )
        assert chain.attrs.get("tgt_overlap_policy") == "best_cluster_max"

    def test_sum_rewards_duplications(self, tmp_path):
        """SUM strategy: sum of member lengths rewards duplications.

        R: 'best_cluster_sum: Sum of lengths behavior (rewards duplications)'
        Cluster 1 (overlap): A src[100-200] + B src[150-250] -> SUM=200, UNION=150
        Cluster 2 (single): C src[500-680] -> SUM=180, UNION=180
        SUM: Cluster 1 (200) > Cluster 2 (180) -> A+B wins
        UNION: Cluster 2 (180) > Cluster 1 (150) -> C wins
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 150, "src_end": 250,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1200, "tgt_end": 1300,
              "chain_id": 2},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 500, "src_end": 680,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2180,
              "chain_id": 3},
             [(180,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_sum",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 2
        assert set(result["chain_id"].tolist()) == {1, 2}

    def test_union_vs_sum_different_winners(self, tmp_path):
        """UNION and SUM strategies produce different winners.

        R: 'best_cluster_union vs best_cluster_sum: Different winners'
        Same setup as above. Under UNION, C wins; under SUM, A+B wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 150, "src_end": 250,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1200, "tgt_end": 1300,
              "chain_id": 2},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 500, "src_end": 680,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2180,
              "chain_id": 3},
             [(180,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain_union = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_union",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})
        result_union = pm.gintervals_liftover(intervals, chain_union)
        assert len(result_union) == 1
        assert result_union.iloc[0]["chain_id"] == 3

    def test_max_single_large_beats_overlapping_cluster(self, tmp_path):
        """MAX strategy: single large chain beats overlapping cluster.

        R: 'best_cluster_max: Single large chain beats overlapping cluster'
        Cluster 1: A(100) + B(100) overlap -> MAX=100
        Cluster 2: C(120) -> MAX=120
        C wins.
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 150, "src_end": 250,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1200, "tgt_end": 1300,
              "chain_id": 2},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 500, "src_end": 620,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2120,
              "chain_id": 3},
             [(120,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        chain = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_max",
        )
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})
        result = pm.gintervals_liftover(intervals, chain)
        assert len(result) == 1
        assert result.iloc[0]["chain_id"] == 3

    def test_best_source_cluster_is_alias_for_union(self, tmp_path):
        """best_source_cluster should be an alias for best_cluster_union.

        R: 'best_source_cluster is alias for best_cluster_union'
        Both policies currently behave identically during loading (as keep).
        """
        entries = [
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 100, "src_end": 200,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 150, "src_end": 250,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1200, "tgt_end": 1300,
              "chain_id": 2},
             [(100,)]),
            ({"score": 1000, "src_chrom": "src", "src_size": 3000,
              "src_strand": "+", "src_start": 500, "src_end": 680,
              "tgt_chrom": "chr1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2180,
              "chain_id": 3},
             [(180,)]),
        ]
        path = _write_chain(str(tmp_path), entries)
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [1000]})

        chain_old = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_source_cluster",
        )
        chain_new = pm.gintervals_load_chain(
            path, src_overlap_policy="keep",
            tgt_overlap_policy="best_cluster_union",
        )
        result_old = pm.gintervals_liftover(intervals, chain_old)
        result_new = pm.gintervals_liftover(intervals, chain_new)

        assert len(result_old) == len(result_new)
        assert list(result_old["chain_id"].sort_values()) == list(result_new["chain_id"].sort_values())
        assert list(result_old["start"].sort_values()) == list(result_new["start"].sort_values())


# gtrack_liftover tests are in tests/test_track_liftover.py
