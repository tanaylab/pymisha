"""Tests for gseq_kmer and gseq_kmer_dist."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestGseqKmer:
    """Tests for gseq_kmer function."""

    def test_basic_count(self):
        """Count a simple k-mer on forward strand."""
        result = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=1)
        assert result[0] == 2  # ACG appears at positions 0 and 4

    def test_single_base_count(self):
        """Count single base occurrences."""
        result = pm.gseq_kmer(["AAACCC"], "A", mode="count", strand=1)
        assert result[0] == 3

    def test_forward_strand_only(self):
        """Count on forward strand only."""
        result = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=1)
        assert result[0] == 2

    def test_reverse_strand_only(self):
        """Count on reverse strand only."""
        # ACG reverse complement is CGT
        result = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=-1)
        # CGT appears at positions 2 and 6
        assert result[0] == 2

    def test_both_strands(self):
        """Count on both strands."""
        # R misha counts both strands even for palindromic kmers.
        result = pm.gseq_kmer(["AACGTT"], "CG", mode="count", strand=0)
        assert result[0] == 2

    def test_non_palindrome_both_strands(self):
        """Non-palindromic k-mer counted on both strands."""
        # ACG fwd at pos 0, CGT (revcomp ACG) at pos 1
        result = pm.gseq_kmer(["ACGT"], "ACG", mode="count", strand=0)
        assert result[0] == 2  # ACG forward + CGT (=revcomp ACG) at pos 1

    def test_fraction_mode(self):
        """Fraction mode returns count / possible positions."""
        # "ACGTACGT" has 8 chars, k=3, so 6 possible positions
        result = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="frac", strand=1)
        assert abs(result[0] - 2.0 / 6) < 1e-10

    def test_multiple_sequences(self):
        """Works with multiple input sequences."""
        result = pm.gseq_kmer(["AACG", "CCCC", "ACGA"], "CG", mode="count", strand=1)
        assert len(result) == 3
        assert result[0] == 1  # CG at pos 2
        assert result[1] == 0  # no CG
        assert result[2] == 1  # CG at pos 1

    def test_string_input(self):
        """Single string input works."""
        result = pm.gseq_kmer("ACGT", "ACG", mode="count", strand=1)
        assert len(result) == 1
        assert result[0] == 1

    def test_empty_sequence(self):
        """Empty sequence returns 0."""
        result = pm.gseq_kmer([""], "ACG", mode="count")
        assert result[0] == 0

    def test_kmer_longer_than_seq(self):
        """K-mer longer than sequence returns 0."""
        result = pm.gseq_kmer(["AC"], "ACGT", mode="count")
        assert result[0] == 0

    def test_start_end_pos(self):
        """ROI boundaries limit search."""
        # "ACGTACGT" (1-indexed: A=1, C=2, G=3, T=4, A=5, C=6, G=7, T=8)
        # ROI 5-8 = "ACGT", ACG at pos 0 → 1 match
        result = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=1,
                               start_pos=5, end_pos=8)
        assert result[0] == 1
        # Full sequence has 2 ACG matches
        result_full = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=1)
        assert result_full[0] == 2

    def test_invalid_kmer(self):
        """Invalid k-mer characters raise error."""
        with pytest.raises(ValueError, match="only A, C, G, T"):
            pm.gseq_kmer(["ACGT"], "AXG")

    def test_invalid_strand(self):
        """Invalid strand value raises error."""
        with pytest.raises(ValueError, match="strand must be"):
            pm.gseq_kmer(["ACGT"], "ACG", strand=2)

    def test_case_insensitive(self):
        """Case-insensitive matching."""
        result = pm.gseq_kmer(["acgtACGT"], "ACG", mode="count", strand=1)
        assert result[0] == 2

    def test_gap_skipping(self):
        """Gap characters are skipped."""
        # "AC-GT" with gaps removed becomes "ACGT"
        result = pm.gseq_kmer(["AC-GT"], "ACGT", mode="count", strand=1,
                               skip_gaps=True, gap_chars=["-"])
        assert result[0] == 1

    def test_returns_numpy_array(self):
        """Return type is numpy array."""
        result = pm.gseq_kmer(["ACGT"], "AC")
        assert isinstance(result, np.ndarray)


class TestGseqKmerDist:
    """Tests for gseq_kmer_dist function."""

    def test_basic_kmer_dist(self):
        """Count k-mer distribution in a genomic region."""
        intervals = pm.gintervals("1", 0, 1000)
        result = pm.gseq_kmer_dist(intervals, k=2)
        assert isinstance(result, pd.DataFrame)
        assert "kmer" in result.columns
        assert "count" in result.columns
        assert len(result) > 0
        # All k-mers should be of length 2
        assert all(len(k) == 2 for k in result["kmer"])
        # Counts should be positive
        assert (result["count"] > 0).all()

    def test_kmer_dist_sorted(self):
        """K-mers should be sorted alphabetically."""
        intervals = pm.gintervals("1", 0, 1000)
        result = pm.gseq_kmer_dist(intervals, k=2)
        kmers = result["kmer"].tolist()
        assert kmers == sorted(kmers)

    def test_kmer_dist_k1(self):
        """k=1 should return at most 4 k-mers (A, C, G, T)."""
        intervals = pm.gintervals("1", 0, 1000)
        result = pm.gseq_kmer_dist(intervals, k=1)
        assert len(result) <= 4
        assert set(result["kmer"]).issubset({"A", "C", "G", "T"})

    def test_kmer_dist_large_k(self):
        """k=6 returns many k-mers."""
        intervals = pm.gintervals("1", 0, 10000)
        result = pm.gseq_kmer_dist(intervals, k=6)
        assert len(result) > 10  # Should have many 6-mers in 10kb

    def test_kmer_dist_with_mask(self):
        """Mask intervals exclude regions from counting."""
        intervals = pm.gintervals("1", 0, 1000)
        mask = pm.gintervals("1", 0, 500)

        full_result = pm.gseq_kmer_dist(intervals, k=2)
        masked_result = pm.gseq_kmer_dist(intervals, k=2, mask=mask)

        # Masked result should have fewer total counts
        full_total = full_result["count"].sum()
        masked_total = masked_result["count"].sum()
        assert masked_total < full_total

    def test_kmer_dist_invalid_k(self):
        """Invalid k raises ValueError."""
        intervals = pm.gintervals("1", 0, 1000)
        with pytest.raises(ValueError, match="k must be"):
            pm.gseq_kmer_dist(intervals, k=0)
        with pytest.raises(ValueError, match="k must be"):
            pm.gseq_kmer_dist(intervals, k=11)

    def test_kmer_dist_total_matches_sequence(self):
        """Total k-mer count should match expected from sequence length."""
        intervals = pm.gintervals("1", 0, 100)
        result = pm.gseq_kmer_dist(intervals, k=1)
        total = result["count"].sum()
        # For k=1, total should equal sequence length (minus any N bases)
        seqs = pm.gseq_extract(intervals)
        expected = sum(1 for c in seqs[0].upper() if c in "ACGT")
        assert total == expected
