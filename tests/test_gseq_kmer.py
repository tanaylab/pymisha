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
        # As in R misha, a position counts once when the k-mer matches on
        # either strand, so the palindrome CG counts once.
        result = pm.gseq_kmer(["AACGTT"], "CG", mode="count", strand=0)
        assert result[0] == 1

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
        # CG appears 4 times; a palindrome counts once per position
        result = pm.gseq_kmer(seqs, "CG", mode="count", strand=0)
        assert all(r == 4 for r in result)

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
        """A palindromic k-mer (AATT) counts once per position on both strands."""
        result = pm.gseq_kmer(["AATTCC"], "AATT", mode="count", strand=0)
        assert result[0] == 1

    def test_empty_list(self):
        """Empty input list returns empty array."""
        result = pm.gseq_kmer([], "CG")
        assert len(result) == 0
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


def _ref_gseq_kmer(seq, kmer, mode, strand):
    """R misha's gseq.kmer over a whole gap-free sequence, written out."""
    seq, kmer = seq.upper(), kmer.upper()
    rc = pm.grevcomp(kmer)
    starts = len(seq) - len(kmer) + 1
    if starts <= 0:
        return 0.0
    count = sum(
        1 for p in range(starts)
        if (strand >= 0 and seq[p:p + len(kmer)] == kmer)
        or (strand <= 0 and seq[p:p + len(kmer)] == rc)
    )
    return count / starts if mode == "frac" else float(count)


class TestGseqKmerCpp:
    """Tests for the C++ k-mer engine behind gseq_kmer."""

    def test_cpp_dispatch_is_used(self):
        """The C++ engine is callable directly."""
        from pymisha._shared import _pymisha
        result = _pymisha.pm_kmer_count_strings(
            ["ACGTACGT"], "ACG", "count", 1, np.array([1]), np.array([8]), 0, True, "-.")
        assert isinstance(result, np.ndarray)
        assert result[0] == 2.0

    def test_matches_the_written_out_rule_for_all_combos(self):
        """Every strand/mode combination against _ref_gseq_kmer."""
        seqs = ["ACGTACGT", "GGCCTTAA", "CGCGCGCG", "AAAAGGGG",
                "TTTTTTT", "ACACACAC", "GTGTGTGT", "CCCCCCCC",
                "ATCGATCGATCG", "GCGCGCGCGCGC"]
        for kmer in ["CG", "ACG", "ACGT", "A", "AATT"]:
            for strand in (-1, 0, 1):
                for mode in ("count", "frac"):
                    got = pm.gseq_kmer(seqs, kmer, mode=mode, strand=strand)
                    want = [_ref_gseq_kmer(x, kmer, mode, strand) for x in seqs]
                    np.testing.assert_allclose(
                        got, want, rtol=1e-12, atol=0,
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
        """A palindromic k-mer counts once per position on both strands."""
        result = pm.gseq_kmer(["AATTCC"], "AATT", mode="count", strand=0)
        assert result[0] == 1
        # "CGCGCG": CG at positions 0, 2, 4
        result2 = pm.gseq_kmer(["CGCGCG"], "CG", mode="count", strand=0)
        assert result2[0] == 3

    def test_cpp_n_characters(self):
        """N characters in sequences do not match any kmer."""
        result = pm.gseq_kmer(["ANCGT", "NNNNN", "ACNGT"], "ACG", mode="count", strand=1)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_cpp_direct_call(self):
        """Direct C++ call produces correct results for known inputs."""
        from pymisha._shared import _pymisha
        # Forward only: ACG at positions 0 and 4
        result = _pymisha.pm_kmer_count_strings(
            ["ACGTACGT", "TTTT"], "ACG", "count", 1, np.array([1, 1]), np.array([8, 4]), 0, False, ""
        )
        np.testing.assert_array_equal(result, [2.0, 0.0])

    def test_cpp_direct_frac(self):
        """Direct C++ call for fraction mode."""
        from pymisha._shared import _pymisha
        # "ACGTACGT" has 8 chars, k=3, 6 positions, 2 matches -> 2/6
        result = _pymisha.pm_kmer_count_strings(
            ["ACGTACGT"], "ACG", "frac", 1, np.array([1]), np.array([8]), 0, False, ""
        )
        np.testing.assert_allclose(result, [2.0 / 6.0], rtol=1e-4)

    def test_cpp_lowercase_handling(self):
        """C++ correctly uppercases input sequences."""
        result1 = pm.gseq_kmer(["acgtacgt"], "ACG", mode="count", strand=1)
        result2 = pm.gseq_kmer(["ACGTACGT"], "ACG", mode="count", strand=1)
        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Ported from R misha's tests/testthat/test-sequence.R: every test_that block
# that calls gseq.kmer or gseq.kmer.dist, in R's order.
# ---------------------------------------------------------------------------


class TestGseqKmerRParity:
    """gseq.kmer blocks of test-sequence.R."""

    def test_validates_inputs_correctly(self):
        seqs = "ACGTACGT"
        pm.gseq_kmer(seqs, "AC", mode="count")
        with pytest.raises(ValueError, match="single character"):
            pm.gseq_kmer(seqs, ["AC", "GT"], mode="count")
        with pytest.raises(ValueError, match="only A, C, G, T"):
            pm.gseq_kmer(seqs, "ACN", mode="count")
        with pytest.raises(ValueError, match="strand must be"):
            pm.gseq_kmer(seqs, "AC", mode="count", strand=5)
        with pytest.raises(ValueError, match="extend must be"):
            pm.gseq_kmer(seqs, "AC", mode="count", extend=-5)

    def test_mode_count_counts_exact_matches(self):
        seq = "CGCGCGCG"
        assert pm.gseq_kmer(seq, "CG", mode="count", start_pos=1, end_pos=8, extend=False)[0] == 4
        assert pm.gseq_kmer(seq, "GC", mode="count", start_pos=1, end_pos=8, extend=False)[0] == 3
        assert pm.gseq_kmer(seq, "AA", mode="count", start_pos=1, end_pos=8, extend=False)[0] == 0

    def test_mode_frac_computes_correct_fractions(self):
        seq = "AAACCCGGG"
        kw = {"mode": "frac", "strand": 1, "start_pos": 1, "end_pos": 9, "extend": False}
        assert pm.gseq_kmer(seq, "A", **kw)[0] == pytest.approx(3 / 9)
        assert pm.gseq_kmer(seq, "C", **kw)[0] == pytest.approx(3 / 9)
        assert pm.gseq_kmer(seq, "AA", **kw)[0] == pytest.approx(2 / 8)

    def test_handles_roi_bounds_correctly(self):
        seq = "GGGGACGGGG"
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=5, end_pos=6, extend=False)[0] == 1
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=1, end_pos=4, extend=False)[0] == 0
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=7, end_pos=10, extend=False)[0] == 0

    def test_extend_parameter_works_correctly(self):
        seq = "GGGGACGGGG"
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=6, end_pos=6, extend=False)[0] == 0
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=6, end_pos=6, extend=True)[0] == 1
        # extend=1 (explicit k - 1): same as extend=True
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=6, end_pos=6, extend=1)[0] == 1

    def test_strand_parameter_works(self):
        seq = "ACGTACGT"
        assert pm.gseq_kmer(seq, "AC", mode="count", strand=1, start_pos=1, end_pos=8)[0] == 2
        assert pm.gseq_kmer(seq, "AC", mode="count", strand=-1, start_pos=1, end_pos=8)[0] == 2
        assert pm.gseq_kmer(seq, "AC", mode="count", strand=0, start_pos=1, end_pos=8)[0] == 4

    def test_handles_edge_case_sequence_too_short(self):
        assert pm.gseq_kmer("AC", "ACGTA", mode="count")[0] == 0
        assert pm.gseq_kmer("AC", "ACGTA", mode="frac")[0] == 0

    def test_handles_edge_case_empty_roi(self):
        seq = "ACGTACGT"
        assert pm.gseq_kmer(seq, "AC", mode="count", start_pos=5, end_pos=3)[0] == 0
        assert pm.gseq_kmer(seq, "AC", mode="frac", start_pos=5, end_pos=3)[0] == 0

    def test_vectorization_works(self):
        result = pm.gseq_kmer(["ACACAC", "GGGGGG", "CGCGCG"], "AC", mode="count",
                              start_pos=1, end_pos=6, extend=False)
        assert len(result) == 3
        assert result.tolist() == [3, 0, 0]

    def test_handles_vectorized_start_pos_and_end_pos(self):
        result = pm.gseq_kmer(["ACACAC", "ACACAC"], "AC", mode="count",
                              start_pos=[1, 3], end_pos=[4, 6], extend=False)
        assert result[0] == 2  # AC at 1-2, 3-4
        assert result[1] == 2  # AC at 3-4, 5-6

    def test_frac_denominator_is_computed_correctly(self):
        seq = "ACGTACGT"
        assert pm.gseq_kmer(seq, "AC", mode="frac", strand=1, start_pos=1, end_pos=8,
                            extend=False)[0] == pytest.approx(2 / 7)
        assert pm.gseq_kmer(seq, "AC", mode="frac", strand=1, start_pos=1, end_pos=4,
                            extend=False)[0] == pytest.approx(1 / 3)

    # -- gseq.kmer against kmer virtual tracks --------------------------------

    @staticmethod
    def _seq_and_vtracks(specs):
        pm.gvtrack_clear()
        iv = pm.gintervals("chr1", 200, 240)
        seq = pm.gseq_extract(iv)[0].upper()
        for name, kw in specs.items():
            pm.gvtrack_create(name, None, kw.pop("func"), **kw)
        res = pm.gextract(list(specs), iv, iterator=iv)
        pm.gvtrack_clear()
        return seq, res

    def test_matches_gextract_kmer_count_vtrack(self):
        seq, res = self._seq_and_vtracks({"count_ta": {"func": "kmer.count", "kmer": "TA", "strand": 1}})
        got = pm.gseq_kmer(seq, "TA", mode="count", strand=1, start_pos=1, end_pos=len(seq), extend=False)
        assert got[0] == res["count_ta"].iloc[0]

    def test_matches_gextract_kmer_frac_vtrack(self):
        seq, res = self._seq_and_vtracks({"frac_ta": {"func": "kmer.frac", "kmer": "TA", "strand": 1, "extend": False}})
        got = pm.gseq_kmer(seq, "TA", mode="frac", strand=1, start_pos=1, end_pos=len(seq), extend=False)
        assert got[0] == pytest.approx(res["frac_ta"].iloc[0], rel=1e-6)

    def test_matches_gextract_kmer_vtrack_with_longer_kmer(self):
        seq, res = self._seq_and_vtracks({
            "count_ccc": {"func": "kmer.count", "kmer": "CCC", "strand": 1, "extend": False},
            "frac_ccc": {"func": "kmer.frac", "kmer": "CCC", "strand": 1, "extend": False},
        })
        kw = {"strand": 1, "start_pos": 1, "end_pos": len(seq), "extend": False}
        assert pm.gseq_kmer(seq, "CCC", mode="count", **kw)[0] == res["count_ccc"].iloc[0]
        assert pm.gseq_kmer(seq, "CCC", mode="frac", **kw)[0] == pytest.approx(
            res["frac_ccc"].iloc[0], rel=1e-6)

    def test_matches_gextract_kmer_vtrack_both_strands(self):
        seq, res = self._seq_and_vtracks({
            "count_both": {"func": "kmer.count", "kmer": "AC", "strand": 0},
            "count_fwd": {"func": "kmer.count", "kmer": "AC", "strand": 1},
            "count_rev": {"func": "kmer.count", "kmer": "AC", "strand": -1},
        })
        kw = {"mode": "count", "start_pos": 1, "end_pos": len(seq), "extend": False, "skip_gaps": False}
        assert pm.gseq_kmer(seq, "AC", strand=0, **kw)[0] == res["count_both"].iloc[0]
        assert pm.gseq_kmer(seq, "AC", strand=1, **kw)[0] == res["count_fwd"].iloc[0]
        assert pm.gseq_kmer(seq, "AC", strand=-1, **kw)[0] == res["count_rev"].iloc[0]

    # -- gaps ------------------------------------------------------------------

    def test_with_gaps_basic_matching(self):
        kw = {"mode": "count", "strand": 1, "start_pos": 1, "end_pos": 6, "extend": False}
        assert pm.gseq_kmer("A-CG-A", "ACG", skip_gaps=True, **kw)[0] == 1
        assert pm.gseq_kmer("A-CG-A", "ACG", skip_gaps=False, **kw)[0] == 0

    def test_with_gaps_fraction_mode(self):
        got = pm.gseq_kmer("A-A-A-GGG", "A", mode="frac", strand=1, start_pos=1, end_pos=9,
                           extend=False, skip_gaps=True)
        assert got[0] == pytest.approx(3 / 6)

    def test_with_gaps_fraction_mode_for_k_above_1(self):
        got = pm.gseq_kmer("A-AC-AC", "AC", mode="frac", strand=1, start_pos=1, end_pos=7,
                           extend=False, skip_gaps=True)
        assert got[0] == pytest.approx(2 / 4)

    def test_with_gaps_roi_handling(self):
        seq = "GGG-A-C-GGG"
        kw = {"mode": "count", "strand": 1, "extend": False, "skip_gaps": True}
        assert pm.gseq_kmer(seq, "AC", start_pos=5, end_pos=7, **kw)[0] == 1
        assert pm.gseq_kmer(seq, "AC", start_pos=1, end_pos=3, **kw)[0] == 0

    def test_with_gaps_reverse_strand(self):
        assert pm.gseq_kmer("G--T-GGG", "AC", mode="count", strand=-1, start_pos=1, end_pos=8,
                            extend=False, skip_gaps=True)[0] == 1

    def test_with_gaps_both_strands(self):
        assert pm.gseq_kmer("A-CG-G-T", "AC", mode="count", strand=0, start_pos=1, end_pos=8,
                            extend=False, skip_gaps=True)[0] == 2

    def test_with_gaps_empty_sequence(self):
        assert pm.gseq_kmer("-----", "AC", mode="count", skip_gaps=True)[0] == 0
        assert pm.gseq_kmer("-----", "AC", mode="frac", skip_gaps=True)[0] == 0

    def test_with_gaps_custom_gap_characters(self):
        got = pm.gseq_kmer("A_C.A*C", "AC", mode="count", strand=1, skip_gaps=True,
                           gap_chars=["-", ".", "_", "*"])
        assert got[0] == 2

    def test_skip_gaps_false_preserves_original_behavior(self):
        # R's block also checks gseq.pwm (covered in test_gseq_pwm*.py); this is
        # its k-mer half.
        seq = "ACGTACGT"
        assert pm.gseq_kmer(seq, "AC", mode="count", skip_gaps=False)[0] == \
            pm.gseq_kmer(seq, "AC", mode="count", skip_gaps=True)[0]


class TestGseqKmerDistRParity:
    """gseq.kmer.dist blocks of test-sequence.R. R runs them on
    gdb.init_examples(); pymisha's example database is the test database."""

    @pytest.fixture(autouse=True)
    def _examples(self):
        pm.gdb_init_examples()

    @staticmethod
    def _iv(chroms, starts, ends):
        return pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})

    def _one(self, start, end, chrom="chr1"):
        return self._iv([chrom], [start], [end])

    def test_counts_kmers_correctly(self):
        iv = self._one(0, 1000)
        r = pm.gseq_kmer_dist(iv, k=1)
        assert len(r) == 4
        assert r["count"].sum() == 1000
        assert {"A", "C", "G", "T"} <= set(r["kmer"])
        r = pm.gseq_kmer_dist(iv, k=2)
        assert len(r) == 16
        assert r["count"].sum() == 999
        assert pm.gseq_kmer_dist(iv, k=3)["count"].sum() == 998

    def test_validates_k_parameter(self):
        iv = self._one(0, 1000)
        for k in (0, 11, -1):
            with pytest.raises(ValueError, match="k must be an integer between 1 and 10"):
                pm.gseq_kmer_dist(iv, k=k)

    def test_mask_excludes_positions_correctly(self):
        iv = self._one(0, 1000)
        total = pm.gseq_kmer_dist(iv, k=3)["count"].sum()
        masked = pm.gseq_kmer_dist(iv, k=3, mask=self._one(500, 600))["count"].sum()
        # the k-mers with any base in the mask: starts 498-599, 102 positions
        assert total - masked == 102

    def test_handles_multiple_intervals(self):
        iv = self._iv(["chr1", "chr1"], [0, 2000], [1000, 3000])
        assert pm.gseq_kmer_dist(iv, k=2)["count"].sum() == 1998

    def test_handles_multiple_chromosomes(self):
        iv = self._iv(["chr1", "chr2"], [0, 0], [1000, 1000])
        assert pm.gseq_kmer_dist(iv, k=1)["count"].sum() == 2000

    def test_returns_correct_data_frame_structure(self):
        r = pm.gseq_kmer_dist(self._one(0, 1000), k=4)
        assert isinstance(r, pd.DataFrame)
        assert list(r.columns) == ["kmer", "count"]
        assert all(isinstance(x, str) for x in r["kmer"])
        assert pd.api.types.is_numeric_dtype(r["count"])
        assert (r["kmer"].str.len() == 4).all()

    def test_only_returns_non_zero_counts(self):
        r = pm.gseq_kmer_dist(self._one(0, 100), k=5)
        assert len(r) < 1024
        assert (r["count"] > 0).all()

    def test_handles_interval_shorter_than_k(self):
        r = pm.gseq_kmer_dist(self._one(0, 3), k=6)
        assert len(r) == 0
        assert r["count"].sum() == 0
        assert len(pm.gseq_kmer_dist(self._one(0, 5), k=6)) == 0
        assert pm.gseq_kmer_dist(self._one(0, 6), k=6)["count"].sum() == 1

    def test_matches_manual_counting_from_gseq_extract(self):
        iv = self._one(0, 500)
        seq = pm.gseq_extract(iv)[0].upper()
        manual: dict[str, int] = {}
        for i in range(len(seq) - 1):
            d = seq[i:i + 2]
            if "N" not in d:
                manual[d] = manual.get(d, 0) + 1
        r = pm.gseq_kmer_dist(iv, k=2)
        got = dict(zip(r["kmer"], r["count"], strict=True))
        for kmer, n in manual.items():
            assert got.get(kmer) == n, f"Mismatch for k-mer {kmer}"

    def test_handles_mask_at_interval_boundaries(self):
        iv = self._one(100, 200)
        start = pm.gseq_kmer_dist(iv, k=2, mask=self._one(100, 120))["count"].sum()
        end = pm.gseq_kmer_dist(iv, k=2, mask=self._one(180, 200))["count"].sum()
        none = pm.gseq_kmer_dist(iv, k=2)["count"].sum()
        assert none == 99
        assert none - start == 20
        # dinucleotides starting at 179-198 reach into [180, 200)
        assert none - end == 20

    def test_handles_overlapping_intervals_correctly(self):
        iv = self._iv(["chr1", "chr1"], [0, 500], [1000, 1500])
        assert pm.gseq_kmer_dist(iv, k=2)["count"].sum() == 1998

    def test_handles_multiple_mask_intervals(self):
        iv = self._one(0, 1000)
        mask = self._iv(["chr1"] * 3, [100, 300, 700], [150, 400, 800])
        diff = pm.gseq_kmer_dist(iv, k=3)["count"].sum() - pm.gseq_kmer_dist(iv, k=3, mask=mask)["count"].sum()
        # each mask also removes the 2 k-mers that start before it and reach into it
        assert diff == 52 + 102 + 102

    def test_with_k_10_works(self):
        r = pm.gseq_kmer_dist(self._one(0, 1000), k=10)
        assert r["count"].sum() == 991
        assert (r["kmer"].str.len() == 10).all()

    def test_with_gintervals_all_works(self):
        r = pm.gseq_kmer_dist(pm.gintervals_all(), k=2)
        assert len(r) == 16
        all_iv = pm.gintervals_all()
        max_possible = (all_iv["end"] - all_iv["start"] - 1).sum()
        assert r["count"].sum() > 0
        assert r["count"].sum() <= max_possible

    def test_kmer_strings_are_sorted_alphabetically(self):
        r = pm.gseq_kmer_dist(self._one(0, 10000), k=3)
        assert r["kmer"].tolist() == sorted(r["kmer"])

    def test_handles_mask_on_different_chromosome(self):
        iv = self._one(0, 1000)
        assert pm.gseq_kmer_dist(iv, k=2)["count"].sum() == \
            pm.gseq_kmer_dist(iv, k=2, mask=self._one(0, 500, "chr2"))["count"].sum()

    def test_is_deterministic(self):
        iv = self._one(0, 5000)
        pd.testing.assert_frame_equal(pm.gseq_kmer_dist(iv, k=4), pm.gseq_kmer_dist(iv, k=4))

    def test_handles_adjacent_intervals(self):
        adjacent = self._iv(["chr1", "chr1"], [0, 1000], [1000, 2000])
        continuous = self._one(0, 2000)
        diff = pm.gseq_kmer_dist(continuous, k=3)["count"].sum() - pm.gseq_kmer_dist(adjacent, k=3)["count"].sum()
        assert diff == 2

    def test_with_mask_covering_entire_interval_returns_empty(self):
        r = pm.gseq_kmer_dist(self._one(100, 200), k=2, mask=self._one(0, 500))
        assert len(r) == 0
        assert r["count"].sum() == 0

    def test_kmer_counts_are_consistent_across_k_values(self):
        iv = self._one(0, 1000)
        assert pm.gseq_kmer_dist(iv, k=1)["count"].sum() == 1000
        for k in range(2, 7):
            assert pm.gseq_kmer_dist(iv, k=k)["count"].sum() == 1000 - k + 1, f"Failed for k={k}"

    def test_frequencies_sum_correctly(self):
        r = pm.gseq_kmer_dist(self._one(0, 10000), k=3)
        assert (r["count"] / r["count"].sum()).sum() == pytest.approx(1, abs=1e-10)

    def test_handles_very_small_intervals(self):
        r = pm.gseq_kmer_dist(self._one(100, 101), k=2)
        assert len(r) == 0
        assert r["count"].sum() == 0

    def test_reverse_complement_kmers_are_counted_separately(self):
        r = pm.gseq_kmer_dist(self._one(0, 10000), k=2)
        assert r.loc[r["kmer"] == "AA", "count"].iloc[0] > 0
        assert r.loc[r["kmer"] == "TT", "count"].iloc[0] > 0

    def test_handles_unsorted_intervals(self):
        unsorted = self._iv(["chr1", "chr1"], [2000, 0], [3000, 1000])
        ordered = self._iv(["chr1", "chr1"], [0, 2000], [1000, 3000])
        assert pm.gseq_kmer_dist(unsorted, k=2)["count"].sum() == pm.gseq_kmer_dist(ordered, k=2)["count"].sum()

    def test_handles_mask_larger_than_interval(self):
        assert len(pm.gseq_kmer_dist(self._one(500, 600), k=2, mask=self._one(0, 1000))) == 0

    def test_handles_partial_mask_overlap(self):
        iv = self._one(0, 100)
        masked = pm.gseq_kmer_dist(iv, k=2, mask=self._one(50, 150))["count"].sum()
        diff = pm.gseq_kmer_dist(iv, k=2)["count"].sum() - masked
        # k-mers starting at 49-98 reach into the mask
        assert diff == 50

    def test_with_all_possible_k_values(self):
        iv = self._one(0, 100)
        for k in range(1, 11):
            r = pm.gseq_kmer_dist(iv, k=k)
            assert (r["kmer"].str.len() == k).all(), f"k={k} has wrong k-mer lengths"
            assert r["count"].sum() == max(0, 100 - k + 1), f"k={k} has wrong total count"

    def test_gc_content_matches_monomer_frequencies(self):
        iv = self._one(0, 10000)
        r = pm.gseq_kmer_dist(iv, k=1)
        gc = (r.loc[r["kmer"] == "G", "count"].iloc[0] + r.loc[r["kmer"] == "C", "count"].iloc[0]) / r["count"].sum()
        seq = pm.gseq_extract(iv)[0].upper()
        gc_from_seq = sum(1 for c in seq if c in "GC") / len(seq)
        assert gc == pytest.approx(gc_from_seq, abs=1e-10)
