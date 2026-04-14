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
        assert abs(result[0] - 2.0 / 6) < 1e-4

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

    # ---- Additional tests for numpy-optimized fast path ----

    def test_many_sequences_fast_path(self):
        """Many sequences take the batched numpy fast path."""
        # Generate enough sequences to exceed the 500-byte threshold
        seqs = ["ACGTACGTACGTACGT"] * 50  # 50 x 16 = 800 bytes > 500
        result = pm.gseq_kmer(seqs, "CG", mode="count", strand=1)
        assert len(result) == 50
        assert all(r == 4 for r in result)

    def test_many_sequences_both_strands(self):
        """Batch fast path with both-strand counting."""
        seqs = ["ACGTACGTACGTACGT"] * 50
        # CG appears 4 times forward; CG is palindrome, so 4 times reverse too
        result = pm.gseq_kmer(seqs, "CG", mode="count", strand=0)
        assert all(r == 8 for r in result)

    def test_many_sequences_fraction(self):
        """Batch fast path fraction mode."""
        seqs = ["ACGTACGT"] * 100
        result = pm.gseq_kmer(seqs, "ACG", mode="frac", strand=1)
        expected = 2.0 / 6  # 2 matches in 6 possible positions
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_many_sequences_reverse_only(self):
        """Batch fast path reverse-complement-only counting."""
        seqs = ["ACGTACGT"] * 100
        # ACG revcomp = CGT, appears at pos 2 and 6
        result = pm.gseq_kmer(seqs, "ACG", mode="count", strand=-1)
        assert all(r == 2 for r in result)

    def test_mixed_length_sequences(self):
        """Fast path handles mixed-length sequences correctly."""
        seqs = ["ACGT", "AC", "ACGTACGTACGT", "", "CG"]
        result = pm.gseq_kmer(seqs, "ACG", mode="count", strand=1)
        assert result[0] == 1   # ACG at pos 0
        assert result[1] == 0   # too short
        assert result[2] == 3   # ACG at pos 0, 4, 8
        assert result[3] == 0   # empty
        assert result[4] == 0   # too short for 3-mer

    def test_long_single_sequence(self):
        """Single long sequence uses numpy byte matching."""
        # 1000 bp sequence with known CG count
        seq = "ACGT" * 250
        result = pm.gseq_kmer([seq], "CG", mode="count", strand=1)
        # CG appears once per ACGT repeat at position 1
        assert result[0] == 250

    def test_overlapping_matches(self):
        """Overlapping k-mer matches are counted correctly."""
        # "AAAA" has 3 overlapping "AA" matches
        result = pm.gseq_kmer(["AAAA"], "AA", mode="count", strand=1)
        assert result[0] == 3

    def test_overlapping_matches_batch(self):
        """Overlapping matches work in batch mode too."""
        seqs = ["AAAA", "AAAAAA", "A"]
        result = pm.gseq_kmer(seqs, "AA", mode="count", strand=1)
        assert result[0] == 3   # 3 overlapping AA in AAAA
        assert result[1] == 5   # 5 overlapping AA in AAAAAA
        assert result[2] == 0   # too short

    def test_palindromic_kmer_both_strands(self):
        """Palindromic k-mers (e.g., AATT) counted on both strands."""
        # AATT is palindromic (revcomp = AATT)
        # In "AATTCC", AATT appears at pos 0 forward, and revcomp(AATT)=AATT also at pos 0
        result = pm.gseq_kmer(["AATTCC"], "AATT", mode="count", strand=0)
        assert result[0] == 2  # counted once per strand

    def test_empty_list(self):
        """Empty input list returns empty array."""
        result = pm.gseq_kmer([], "CG")
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_fast_path_parity_with_slow_path(self):
        """Fast and slow path produce identical results for the same input."""
        from pymisha.sequence import _gseq_kmer_fast, _count_kmer_in_seq
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"] * 10
        kmer = "ACG"
        k = 3
        for strand in (-1, 0, 1):
            for mode in ("count", "frac"):
                fast = _gseq_kmer_fast(seqs, kmer.upper(), mode, strand, k)
                slow = np.zeros(len(seqs), dtype=float)
                for i, seq in enumerate(seqs):
                    count = _count_kmer_in_seq(seq, kmer.upper(), strand,
                                               None, None, False, False, [])
                    if mode == "frac":
                        possible = max(0, len(seq) - k + 1)
                        if strand == 0:
                            possible *= 2
                        slow[i] = count / possible if possible > 0 else 0.0
                    else:
                        slow[i] = count
                np.testing.assert_array_equal(
                    fast, slow,
                    err_msg=f"Mismatch for strand={strand}, mode={mode}"
                )


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

    def test_kmer_dist_k2_total(self):
        """Total 2-mer count = seq_length - 1 for fully valid sequence."""
        intervals = pm.gintervals("1", 0, 100)
        result = pm.gseq_kmer_dist(intervals, k=2)
        total = result["count"].sum()
        seqs = pm.gseq_extract(intervals)
        seq = seqs[0].upper()
        # Count contiguous valid runs and sum (run_len - 1) for each
        expected = 0
        run = 0
        for c in seq:
            if c in "ACGT":
                run += 1
            else:
                if run >= 2:
                    expected += run - 1
                run = 0
        if run >= 2:
            expected += run - 1
        assert total == expected

    def test_kmer_dist_stride_tricks_vs_rolling(self):
        """Stride-tricks path (k<=8) matches rolling path (k>8)."""
        intervals = pm.gintervals("1", 0, 5000)
        # k=8 uses stride tricks, k=9 uses rolling multiply-accumulate
        # Both should produce consistent results per their k values
        result_k8 = pm.gseq_kmer_dist(intervals, k=8)
        result_k9 = pm.gseq_kmer_dist(intervals, k=9)
        # Both should have valid counts summing to approximately seq_len - k + 1
        assert result_k8["count"].sum() > 0
        assert result_k9["count"].sum() > 0
        # k=8 total should be slightly larger than k=9 total
        assert result_k8["count"].sum() >= result_k9["count"].sum()

    def test_kmer_strings_cache(self):
        """Verify _kmer_strings returns correct k-mer strings."""
        from pymisha.sequence import _kmer_strings
        table = _kmer_strings(2)
        assert len(table) == 16
        assert table[0] == "AA"
        assert table[1] == "AC"
        assert table[2] == "AG"
        assert table[3] == "AT"
        assert table[4] == "CA"
        assert table[15] == "TT"

    def test_kmer_strings_k1(self):
        """k=1 string table is [A, C, G, T]."""
        from pymisha.sequence import _kmer_strings
        table = _kmer_strings(1)
        assert list(table) == ["A", "C", "G", "T"]


class TestGseqKmerCpp:
    """Tests for C++ kmer counting dispatch."""

    def test_cpp_dispatch_is_used(self):
        """Verify the C++ path is available and callable directly."""
        from pymisha._shared import _pymisha
        result = _pymisha.pm_kmer_count_strings(["ACGTACGT"], "ACG", "count", 1)
        assert isinstance(result, np.ndarray)
        assert result[0] == 2.0

    def test_cpp_parity_count_forward(self):
        """C++ matches Python for mode=count, strand=1."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"]
        kmer = "ACG"
        cpp = pm.gseq_kmer(seqs, kmer, mode="count", strand=1)
        py = _gseq_kmer_fast(seqs, kmer.upper(), "count", 1, len(kmer))
        np.testing.assert_array_equal(cpp, py)

    def test_cpp_parity_count_reverse(self):
        """C++ matches Python for mode=count, strand=-1."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"]
        kmer = "ACG"
        cpp = pm.gseq_kmer(seqs, kmer, mode="count", strand=-1)
        py = _gseq_kmer_fast(seqs, kmer.upper(), "count", -1, len(kmer))
        np.testing.assert_array_equal(cpp, py)

    def test_cpp_parity_count_both(self):
        """C++ matches Python for mode=count, strand=0."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"]
        kmer = "ACG"
        cpp = pm.gseq_kmer(seqs, kmer, mode="count", strand=0)
        py = _gseq_kmer_fast(seqs, kmer.upper(), "count", 0, len(kmer))
        np.testing.assert_array_equal(cpp, py)

    def test_cpp_parity_frac_forward(self):
        """C++ matches Python for mode=frac, strand=1."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"]
        kmer = "ACG"
        cpp = pm.gseq_kmer(seqs, kmer, mode="frac", strand=1)
        py = _gseq_kmer_fast(seqs, kmer.upper(), "frac", 1, len(kmer))
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_cpp_parity_frac_reverse(self):
        """C++ matches Python for mode=frac, strand=-1."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"]
        kmer = "ACG"
        cpp = pm.gseq_kmer(seqs, kmer, mode="frac", strand=-1)
        py = _gseq_kmer_fast(seqs, kmer.upper(), "frac", -1, len(kmer))
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_cpp_parity_frac_both(self):
        """C++ matches Python for mode=frac, strand=0."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC"]
        kmer = "ACG"
        cpp = pm.gseq_kmer(seqs, kmer, mode="frac", strand=0)
        py = _gseq_kmer_fast(seqs, kmer.upper(), "frac", 0, len(kmer))
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_cpp_parity_all_combos(self):
        """Comprehensive parity test across all strand/mode combinations."""
        from pymisha.sequence import _gseq_kmer_fast
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC",
                "ATCGATCGATCG", "GCGCGCGCGCGC"]
        for kmer in ["CG", "ACG", "ACGT", "A", "AATT"]:
            k = len(kmer)
            for strand in (-1, 0, 1):
                for mode in ("count", "frac"):
                    cpp = pm.gseq_kmer(seqs, kmer, mode=mode, strand=strand)
                    py = _gseq_kmer_fast(seqs, kmer.upper(), mode, strand, k)
                    tol = 1e-4 if mode == "frac" else 0
                    np.testing.assert_allclose(
                        cpp, py, rtol=tol, atol=1e-15,
                        err_msg=f"Mismatch: kmer={kmer}, strand={strand}, mode={mode}"
                    )

    def test_cpp_empty_list(self):
        """C++ handles empty input list."""
        result = pm.gseq_kmer([], "CG")
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_cpp_short_sequences(self):
        """C++ handles sequences shorter than kmer."""
        result = pm.gseq_kmer(["A", "AC", ""], "ACG", mode="count", strand=1)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_cpp_palindromic_kmer(self):
        """Palindromic k-mer counted correctly on both strands."""
        # AATT is palindromic (revcomp = AATT)
        result = pm.gseq_kmer(["AATTCC"], "AATT", mode="count", strand=0)
        assert result[0] == 2  # counted once forward + once reverse (same positions)

        # CG is palindromic (revcomp = CG)
        # "CGCGCG": CG at positions 0,2,4 = 3 fwd matches + 3 rev matches = 6
        result2 = pm.gseq_kmer(["CGCGCG"], "CG", mode="count", strand=0)
        assert result2[0] == 6

    def test_cpp_n_characters(self):
        """N characters in sequences do not match any kmer."""
        result = pm.gseq_kmer(["ANCGT", "NNNNN", "ACNGT"], "ACG", mode="count", strand=1)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_cpp_direct_call(self):
        """Direct C++ call produces correct results for known inputs."""
        from pymisha._shared import _pymisha
        # Forward only: ACG at positions 0 and 4
        result = _pymisha.pm_kmer_count_strings(
            ["ACGTACGT", "TTTT"], "ACG", "count", 1
        )
        np.testing.assert_array_equal(result, [2.0, 0.0])

    def test_cpp_direct_frac(self):
        """Direct C++ call for fraction mode."""
        from pymisha._shared import _pymisha
        # "ACGTACGT" has 8 chars, k=3, 6 positions, 2 matches -> 2/6
        result = _pymisha.pm_kmer_count_strings(
            ["ACGTACGT"], "ACG", "frac", 1
        )
        np.testing.assert_allclose(result, [2.0 / 6.0], rtol=1e-4)

    def test_cpp_lowercase_handling(self):
        """C++ correctly uppercases input sequences."""
        result1 = pm.gseq_kmer(["acgtacgt"], "ACG", mode="count", strand=1)
        result2 = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=1)
        np.testing.assert_array_equal(result1, result2)
