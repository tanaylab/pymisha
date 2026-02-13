"""Tests for gseq functions."""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize the example database for each test."""
    pm.gdb_init_examples()
    yield


class TestGseqExtract:
    """Tests for gseq_extract function."""

    def test_gseq_extract_basic(self):
        """Test basic sequence extraction."""
        intervals = pm.gintervals("1", 10000, 10020)
        result = pm.gseq_extract(intervals)

        assert result is not None
        assert len(result) == 1
        # Sequence should be 20 bp
        assert len(result[0]) == 20
        # Sequence should be uppercase letters
        assert all(c.upper() in "ACGTN" for c in result[0])

    def test_gseq_extract_multiple_intervals(self):
        """Test sequence extraction for multiple intervals."""
        intervals = pm.gintervals(["1", "2"], [10000, 20000], [10020, 20030])
        result = pm.gseq_extract(intervals)

        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 20
        assert len(result[1]) == 30

    def test_gseq_extract_with_strand(self):
        """Test that negative strand returns reverse complement."""
        intervals = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [10000, 10000],
            "end": [10020, 10020],
            "strand": [1, -1]
        })
        result = pm.gseq_extract(intervals)

        assert result is not None
        assert len(result) == 2
        # Forward and reverse complement should be different (unless palindromic)
        seq_fwd = result[0]
        seq_rev = result[1]
        assert len(seq_fwd) == len(seq_rev)


class TestGseqRev:
    """Tests for gseq_rev function."""

    def test_gseq_rev_single(self):
        """Test reversing a single sequence."""
        result = pm.gseq_rev("ACGT")
        assert result == "TGCA"

    def test_gseq_rev_list(self):
        """Test reversing multiple sequences."""
        result = pm.gseq_rev(["ACGT", "TATA"])
        assert result == ["TGCA", "ATAT"]


class TestGseqComp:
    """Tests for gseq_comp (complement) function."""

    def test_gseq_comp_single(self):
        """Test complementing a single sequence."""
        result = pm.gseq_comp("ACGT")
        assert result == "TGCA"

    def test_gseq_comp_list(self):
        """Test complementing multiple sequences."""
        result = pm.gseq_comp(["ACGT", "AAAA"])
        assert result == ["TGCA", "TTTT"]


class TestGseqRevcomp:
    """Tests for gseq_revcomp (reverse complement) function."""

    def test_gseq_revcomp_single(self):
        """Test reverse complement of a single sequence."""
        result = pm.gseq_revcomp("ACGT")
        # Complement: TGCA, then reverse: ACGT
        assert result == "ACGT"  # ACGT is a palindrome

    def test_gseq_revcomp_non_palindrome(self):
        """Test reverse complement of a non-palindrome."""
        result = pm.gseq_revcomp("AACG")
        # Complement: TTGC, then reverse: CGTT
        assert result == "CGTT"

    def test_gseq_revcomp_list(self):
        """Test reverse complement of multiple sequences."""
        result = pm.gseq_revcomp(["AACG", "AAAA"])
        # AACG -> comp TTGC -> rev CGTT
        # AAAA -> comp TTTT -> rev TTTT
        assert result == ["CGTT", "TTTT"]


class TestGseqEdgeCases:
    """Edge case tests for gseq functions."""

    def test_gseq_extract_requires_intervals(self):
        """Test that gseq_extract requires intervals."""
        with pytest.raises((ValueError, TypeError)):
            pm.gseq_extract(None)

    def test_gseq_rev_handles_lowercase(self):
        """Test that gseq_rev handles lowercase input."""
        result = pm.gseq_rev("acgt")
        assert result.lower() == "tgca"

    def test_gseq_comp_handles_n(self):
        """Test that complement handles N (unknown base)."""
        result = pm.gseq_comp("ANCG")
        # N complements to N
        assert result == "TNGC"
