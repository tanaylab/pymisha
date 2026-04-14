"""Tests verifying C++ PWM scoring path produces identical results to Python path.

Each test computes scores via both paths and asserts they match:
- C++ path: call _pymisha.pm_pwm_score_strings() directly
- Python path: call pm.gseq_pwm() with gap_chars=["-"] to force Python fallback
  (the "-" won't appear in the test sequences, so results are unchanged)

The C++ path uses float32 internally, so we allow rtol=1e-4 tolerance.

NOTE on bidirectional semantics:
The C++ and Python paths handle bidirectional scoring differently for max/count/pos
modes.  C++ combines forward+reverse scores per-position via LSE before
aggregating (matching R misha), while Python aggregates the concatenated
individual-strand scores.  These produce different results for bidirectional
max, count, and pos modes.  The lse mode is unaffected because LSE is
associative.  Therefore, bidirectional parity tests only cover lse mode,
and max/count/pos parity tests use single-strand (forward or reverse-only
with forward-strand position semantics).
"""

import time

import _pymisha
import numpy as np
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_dna_sequences(n, min_len=20, max_len=50, seed=42):
    """Generate n random DNA sequences with lengths in [min_len, max_len]."""
    rng = np.random.RandomState(seed)
    bases = list("ACGT")
    seqs = []
    for _ in range(n):
        length = rng.randint(min_len, max_len + 1)
        seqs.append("".join(rng.choice(bases, size=length)))
    return seqs


def _random_pssm(width=6, seed=42):
    """Generate a random PSSM (frequency matrix) of shape (width, 4)."""
    rng = np.random.RandomState(seed)
    return rng.dirichlet([1, 1, 1, 1], size=width)


def _cpp_score(seqs, pssm, prior=0.01, mode="lse", bidirect=True, strand=0,
               score_thresh=0.0, extend=False, spat_factor=None, spat_bin=1):
    """Score sequences via the C++ path directly."""
    cpp_strand = int(strand)
    if not bidirect and cpp_strand == 0:
        cpp_strand = 1
    return _pymisha.pm_pwm_score_strings(
        list(seqs),
        np.ascontiguousarray(pssm, dtype=float),
        float(prior),
        mode,
        bool(bidirect),
        cpp_strand,
        float(score_thresh),
        int(extend),
        np.asarray(spat_factor, dtype=float) if spat_factor is not None else None,
        int(spat_bin),
    )


def _py_score(seqs, pssm, prior=0.01, mode="lse", bidirect=True, strand=0,
              score_thresh=0.0, extend=False, spat_factor=None, spat_bin=1):
    """Score sequences via the Python path (forced by gap_chars=['-'])."""
    return pm.gseq_pwm(
        list(seqs),
        pssm,
        mode=mode,
        bidirect=bidirect,
        strand=strand,
        score_thresh=score_thresh,
        extend=extend,
        spat_factor=spat_factor,
        spat_bin=spat_bin,
        prior=prior,
        gap_chars=["-"],  # force Python path; "-" not in test sequences
    )


# ---------------------------------------------------------------------------
# 1. Mode parity tests (forward-only to avoid bidirectional semantic diff)
# ---------------------------------------------------------------------------


class TestModeParity:
    """Verify C++ and Python produce identical results for each scoring mode.

    Uses forward-only strand to avoid the bidirectional per-position combining
    semantic difference between C++ and Python.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pssm = _random_pssm(width=6, seed=42)
        self.seqs = _random_dna_sequences(10, min_len=20, max_len=50, seed=42)

    @pytest.mark.parametrize("mode", ["lse", "max", "pos", "count"])
    def test_forward_only_parity(self, mode):
        """All modes should match exactly for forward-only strand."""
        cpp = _cpp_score(self.seqs, self.pssm, mode=mode,
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, mode=mode,
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg=f"mode={mode} forward-only mismatch")

    @pytest.mark.parametrize("mode", ["lse", "max", "pos", "count"])
    def test_forward_only_wider_motif(self, mode):
        """Same forward-only test with a wider motif (10 positions)."""
        pssm = _random_pssm(width=10, seed=99)
        cpp = _cpp_score(self.seqs, pssm, mode=mode,
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, pssm, mode=mode,
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg=f"mode={mode}, width=10 fwd mismatch")

    @pytest.mark.parametrize("mode", ["lse", "max", "pos", "count"])
    def test_forward_only_different_prior(self, mode):
        """Forward-only parity with a different prior."""
        cpp = _cpp_score(self.seqs, self.pssm, prior=0.1, mode=mode,
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, prior=0.1, mode=mode,
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg=f"mode={mode}, prior=0.1 fwd mismatch")

    def test_bidirect_lse_parity(self):
        """LSE mode is associative, so bidirectional should match."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="lse", bidirect=True)
        py = _py_score(self.seqs, self.pssm, mode="lse", bidirect=True)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg="bidirectional lse mismatch")


# ---------------------------------------------------------------------------
# 2. Bidirectional vs single-strand
# ---------------------------------------------------------------------------


class TestStrandOptions:
    """Verify strand/bidirect options produce identical results."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pssm = _random_pssm(width=6, seed=77)
        self.seqs = _random_dna_sequences(10, min_len=20, max_len=50, seed=77)

    def test_bidirect_lse(self):
        """Bidirectional lse: C++ vs Python (associative, so matches)."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="lse", bidirect=True)
        py = _py_score(self.seqs, self.pssm, mode="lse", bidirect=True)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_forward_only_lse(self):
        """Forward-only lse: C++ vs Python."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="lse",
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, mode="lse",
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_forward_only_max(self):
        """Forward-only max: C++ vs Python."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="max",
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, mode="max",
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_forward_only_count(self):
        """Forward-only count: C++ vs Python."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="count",
                         bidirect=False, strand=1, score_thresh=-10.0)
        py = _py_score(self.seqs, self.pssm, mode="count",
                       bidirect=False, strand=1, score_thresh=-10.0)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_bidirect_max_cpp_gte_single(self):
        """Bidirectional max (C++) should be >= forward-only max (C++)."""
        bidir = _cpp_score(self.seqs, self.pssm, mode="max", bidirect=True)
        fwd = _cpp_score(self.seqs, self.pssm, mode="max",
                         bidirect=False, strand=1)
        # At each position, C++ does LSE(fwd, rev) >= fwd, so overall max >= fwd max
        assert np.all(bidir >= fwd - 1e-5)

    def test_bidirect_lse_gte_single(self):
        """Bidirectional lse should be >= forward-only lse."""
        bidir = _cpp_score(self.seqs, self.pssm, mode="lse", bidirect=True)
        fwd = _cpp_score(self.seqs, self.pssm, mode="lse",
                         bidirect=False, strand=1)
        assert np.all(bidir >= fwd - 1e-5)

    @pytest.mark.parametrize("mode", ["lse", "max", "pos", "count"])
    def test_forward_strand_parity_all_modes(self, mode):
        """Forward-only parity across both paths for all modes."""
        cpp = _cpp_score(self.seqs, self.pssm, bidirect=False, strand=1,
                         mode=mode, score_thresh=-10.0)
        py = _py_score(self.seqs, self.pssm, bidirect=False, strand=1,
                       mode=mode, score_thresh=-10.0)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg=f"fwd-only mode={mode}")


# ---------------------------------------------------------------------------
# 3. Spatial weighting
# ---------------------------------------------------------------------------


class TestSpatialWeighting:
    """Verify spatial weighting factors produce identical results."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pssm = _random_pssm(width=6, seed=55)
        self.seqs = _random_dna_sequences(10, min_len=20, max_len=50, seed=55)

    def test_spat_factor_uniform_matches_no_spat(self):
        """Uniform spat_factor=1.0 should match no spatial weighting."""
        spat = [1.0] * 50
        cpp_spat = _cpp_score(self.seqs, self.pssm, spat_factor=spat,
                              bidirect=False, strand=1)
        cpp_none = _cpp_score(self.seqs, self.pssm,
                              bidirect=False, strand=1)
        np.testing.assert_allclose(cpp_spat, cpp_none, rtol=1e-4)

    def test_spat_factor_parity_fwd(self):
        """Non-uniform spat_factor: C++ vs Python (forward-only)."""
        spat = [1.0, 0.5, 0.2, 0.1, 0.05]
        cpp = _cpp_score(self.seqs, self.pssm, spat_factor=spat,
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, spat_factor=spat,
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_spat_factor_parity_bidirect_lse(self):
        """Non-uniform spat_factor: C++ vs Python (bidirectional lse)."""
        spat = [1.0, 0.5, 0.2, 0.1, 0.05]
        cpp = _cpp_score(self.seqs, self.pssm, spat_factor=spat,
                         mode="lse", bidirect=True)
        py = _py_score(self.seqs, self.pssm, spat_factor=spat,
                       mode="lse", bidirect=True)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_spat_factor_with_spat_bin_2(self):
        """Spatial factor with spat_bin=2: C++ vs Python (forward-only)."""
        spat = [1.0, 0.5, 0.2, 0.1, 0.05]
        cpp = _cpp_score(self.seqs, self.pssm, spat_factor=spat, spat_bin=2,
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, spat_factor=spat, spat_bin=2,
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_spat_factor_with_spat_bin_3(self):
        """Spatial factor with spat_bin=3: C++ vs Python (forward-only)."""
        spat = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
        cpp = _cpp_score(self.seqs, self.pssm, spat_factor=spat, spat_bin=3,
                         bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, spat_factor=spat, spat_bin=3,
                       bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    @pytest.mark.parametrize("mode", ["lse", "max", "count"])
    def test_spat_factor_all_modes_fwd(self, mode):
        """Spatial factor parity for lse, max, count modes (forward-only)."""
        spat = [1.0, 0.5, 0.2]
        cpp = _cpp_score(self.seqs, self.pssm, spat_factor=spat, mode=mode,
                         bidirect=False, strand=1, score_thresh=-10.0)
        py = _py_score(self.seqs, self.pssm, spat_factor=spat, mode=mode,
                       bidirect=False, strand=1, score_thresh=-10.0)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg=f"spat fwd mode={mode}")


# ---------------------------------------------------------------------------
# 4. Score threshold (count mode)
# ---------------------------------------------------------------------------


class TestScoreThreshold:
    """Verify count mode with different score thresholds."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pssm = _random_pssm(width=6, seed=33)
        self.seqs = _random_dna_sequences(10, min_len=20, max_len=50, seed=33)

    @pytest.mark.parametrize("thresh", [0, -5, -10, -20])
    def test_count_threshold_parity_fwd(self, thresh):
        """Count mode threshold parity (forward-only)."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="count",
                         score_thresh=thresh, bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, mode="count",
                       score_thresh=thresh, bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4,
                                   err_msg=f"count thresh={thresh}")

    def test_high_threshold_gives_zero_counts(self):
        """A very high threshold should result in zero counts."""
        cpp = _cpp_score(self.seqs, self.pssm, mode="count",
                         score_thresh=1000.0, bidirect=False, strand=1)
        assert np.all(cpp == 0)

    def test_low_threshold_counts_all_positions(self):
        """A very low threshold should count all valid window positions."""
        w = self.pssm.shape[0]
        cpp = _cpp_score(self.seqs, self.pssm, mode="count",
                         score_thresh=-1e10, bidirect=False, strand=1)
        py = _py_score(self.seqs, self.pssm, mode="count",
                       score_thresh=-1e10, bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)
        # Each count should equal len(seq) - w + 1
        expected = np.array([len(s) - w + 1 for s in self.seqs], dtype=float)
        np.testing.assert_allclose(cpp, expected, rtol=1e-4)

    def test_increasing_threshold_decreases_count(self):
        """Higher thresholds should give fewer or equal counts."""
        thresholds = [-20, -15, -10, -5, 0]
        counts = []
        for t in thresholds:
            c = _cpp_score(self.seqs, self.pssm, mode="count",
                           score_thresh=t, bidirect=False, strand=1)
            counts.append(c)
        for i in range(len(thresholds) - 1):
            assert np.all(counts[i] >= counts[i + 1])


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: short sequences, N characters, empty lists, etc."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pssm = _random_pssm(width=6, seed=11)

    def test_exact_motif_length(self):
        """Sequence exactly as long as the motif (one window position)."""
        seqs = _random_dna_sequences(5, min_len=6, max_len=6, seed=11)
        cpp = _cpp_score(seqs, self.pssm, bidirect=False, strand=1)
        py = _py_score(seqs, self.pssm, bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_shorter_than_motif(self):
        """Sequence shorter than motif should give NaN."""
        seqs = ["ACGT"]  # length 4, motif width 6
        cpp = _cpp_score(seqs, self.pssm, bidirect=False, strand=1)
        py = _py_score(seqs, self.pssm, bidirect=False, strand=1)
        assert np.isnan(cpp[0])
        assert np.isnan(py[0])

    def test_sequence_with_n_chars_returns_finite(self):
        """Sequences containing N characters produce finite scores.

        C++ and Python handle N chars differently: C++ integrate_like uses
        log(0.25) while Python uses the PSSM column average log-prob.
        We only verify both paths return finite results (no crash/NaN).
        """
        seqs = ["ACGTNNACGTACGTNN", "NNNACGTACGTNNNNN", "ACGTACGTACGTACGT"]
        cpp = _cpp_score(seqs, self.pssm, bidirect=False, strand=1)
        py = _py_score(seqs, self.pssm, bidirect=False, strand=1)
        assert np.all(np.isfinite(cpp))
        assert np.all(np.isfinite(py))
        # Pure-DNA sequence (no N's) should still match between paths
        np.testing.assert_allclose(cpp[2], py[2], rtol=1e-4)

    def test_all_n_sequence_returns_finite(self):
        """Sequence of all N's produces finite scores (no crash)."""
        seqs = ["NNNNNNNNNN"]
        cpp = _cpp_score(seqs, self.pssm, bidirect=False, strand=1)
        py = _py_score(seqs, self.pssm, bidirect=False, strand=1)
        assert np.all(np.isfinite(cpp))
        assert np.all(np.isfinite(py))

    def test_empty_sequence_list(self):
        """Empty list of sequences."""
        cpp = _cpp_score([], self.pssm, bidirect=False, strand=1)
        py = _py_score([], self.pssm, bidirect=False, strand=1)
        assert len(cpp) == 0
        assert len(py) == 0

    def test_single_sequence_string(self):
        """Single string (not a list) via gseq_pwm API."""
        seq = "ACGTACGTACGTACGT"
        # gseq_pwm should handle single string
        py = pm.gseq_pwm(seq, self.pssm, gap_chars=["-"],
                         bidirect=False, strand=1)
        # C++ path needs a list
        cpp = _cpp_score([seq], self.pssm, bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_many_sequences(self):
        """Verify parity with a larger set of sequences (forward-only)."""
        seqs = _random_dna_sequences(100, min_len=30, max_len=80, seed=123)
        cpp = _cpp_score(seqs, self.pssm, bidirect=False, strand=1)
        py = _py_score(seqs, self.pssm, bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_motif_length_one(self):
        """PSSM with width=1 (single position)."""
        pssm = _random_pssm(width=1, seed=88)
        seqs = _random_dna_sequences(5, min_len=10, max_len=20, seed=88)
        cpp = _cpp_score(seqs, pssm, bidirect=False, strand=1)
        py = _py_score(seqs, pssm, bidirect=False, strand=1)
        np.testing.assert_allclose(cpp, py, rtol=1e-4)

    def test_n_chars_bidirect_lse_finite(self):
        """N characters with bidirectional lse produce finite scores.

        C++ integrate_like uses log(0.25) for N chars, Python uses PSSM column
        average.  We verify both paths return finite, and the pure-DNA seq matches.
        """
        seqs = ["ACGTNNACGTACGTNN", "NNNACGTACGTNNNNN", "ACGTACGTACGTACGT"]
        cpp = _cpp_score(seqs, self.pssm, mode="lse", bidirect=True)
        py = _py_score(seqs, self.pssm, mode="lse", bidirect=True)
        assert np.all(np.isfinite(cpp))
        assert np.all(np.isfinite(py))
        # Pure-DNA sequence matches
        np.testing.assert_allclose(cpp[2], py[2], rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. gseq_pwm dispatch test
# ---------------------------------------------------------------------------


class TestGseqPwmDispatch:
    """Verify gseq_pwm dispatches correctly to C++ vs Python.

    When called without Python-only features, gseq_pwm() should use the C++
    path internally.  We verify by comparing the default API call against
    direct _pymisha.pm_pwm_score_strings() calls.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.pssm = _random_pssm(width=6, seed=42)
        self.seqs = _random_dna_sequences(10, min_len=20, max_len=50, seed=42)

    def test_default_uses_cpp_lse(self):
        """gseq_pwm lse without Python-only features should match direct C++."""
        result_default = pm.gseq_pwm(self.seqs, self.pssm, mode="lse")
        result_cpp = _cpp_score(self.seqs, self.pssm, mode="lse")
        np.testing.assert_allclose(result_default, result_cpp, rtol=1e-6,
                                   err_msg="default dispatch should match C++")

    def test_default_uses_cpp_max(self):
        """gseq_pwm max without Python-only features should match direct C++."""
        result_default = pm.gseq_pwm(self.seqs, self.pssm, mode="max")
        result_cpp = _cpp_score(self.seqs, self.pssm, mode="max")
        np.testing.assert_allclose(result_default, result_cpp, rtol=1e-6)

    def test_default_uses_cpp_count(self):
        """gseq_pwm count without Python-only features should match direct C++."""
        result_default = pm.gseq_pwm(self.seqs, self.pssm, mode="count",
                                     score_thresh=-10.0)
        result_cpp = _cpp_score(self.seqs, self.pssm, mode="count",
                                score_thresh=-10.0)
        np.testing.assert_allclose(result_default, result_cpp, rtol=1e-6)

    def test_default_uses_cpp_pos(self):
        """gseq_pwm pos without Python-only features should match direct C++."""
        result_default = pm.gseq_pwm(self.seqs, self.pssm, mode="pos")
        result_cpp = _cpp_score(self.seqs, self.pssm, mode="pos")
        np.testing.assert_allclose(result_default, result_cpp, rtol=1e-6)

    def test_gap_chars_forces_python_lse(self):
        """gap_chars forces Python path; lse should still match C++."""
        result_py = pm.gseq_pwm(self.seqs, self.pssm, mode="lse",
                                gap_chars=["-"])
        result_cpp = _cpp_score(self.seqs, self.pssm, mode="lse")
        np.testing.assert_allclose(result_py, result_cpp, rtol=1e-4,
                                   err_msg="Python fallback lse should match C++")

    def test_both_paths_match_lse(self):
        """C++ default path and Python fallback produce same lse results."""
        result_default = pm.gseq_pwm(self.seqs, self.pssm, mode="lse")
        result_py = pm.gseq_pwm(self.seqs, self.pssm, mode="lse",
                                gap_chars=["-"])
        np.testing.assert_allclose(result_default, result_py, rtol=1e-4)

    def test_both_paths_match_fwd_all_modes(self):
        """Both paths match for all modes in forward-only."""
        for mode in ("lse", "max", "pos", "count"):
            result_default = pm.gseq_pwm(
                self.seqs, self.pssm, mode=mode,
                bidirect=False, strand=1, score_thresh=-10.0)
            result_py = pm.gseq_pwm(
                self.seqs, self.pssm, mode=mode,
                bidirect=False, strand=1, score_thresh=-10.0,
                gap_chars=["-"])
            np.testing.assert_allclose(result_default, result_py, rtol=1e-4,
                                       err_msg=f"dispatch fwd mode={mode}")

    def test_dispatch_with_spat_factor_lse(self):
        """Dispatch with spatial factor should produce identical lse results."""
        spat = [1.0, 0.5, 0.2, 0.1, 0.05]
        result_default = pm.gseq_pwm(self.seqs, self.pssm, mode="lse",
                                     spat_factor=spat)
        result_py = pm.gseq_pwm(self.seqs, self.pssm, mode="lse",
                                spat_factor=spat, gap_chars=["-"])
        np.testing.assert_allclose(result_default, result_py, rtol=1e-4)

    def test_dispatch_bidirect_false_fwd(self):
        """Dispatch with bidirect=False, strand=1."""
        result_default = pm.gseq_pwm(self.seqs, self.pssm,
                                     bidirect=False, strand=1)
        result_py = pm.gseq_pwm(self.seqs, self.pssm,
                                bidirect=False, strand=1, gap_chars=["-"])
        np.testing.assert_allclose(result_default, result_py, rtol=1e-4)


# ---------------------------------------------------------------------------
# 7. Performance sanity check
# ---------------------------------------------------------------------------


class TestPerformance:
    """Performance comparison (informational, no assertion on speed)."""

    def test_cpp_faster_than_python(self):
        """Time 1000 sequences through C++ vs Python, log speedup ratio."""
        pssm = _random_pssm(width=8, seed=42)
        seqs = _random_dna_sequences(1000, min_len=50, max_len=100, seed=42)

        # Warm up
        _cpp_score(seqs[:10], pssm, bidirect=False, strand=1)
        _py_score(seqs[:10], pssm, bidirect=False, strand=1)

        # Time C++
        t0 = time.perf_counter()
        cpp_result = _cpp_score(seqs, pssm, bidirect=False, strand=1)
        t_cpp = time.perf_counter() - t0

        # Time Python
        t0 = time.perf_counter()
        py_result = _py_score(seqs, pssm, bidirect=False, strand=1)
        t_py = time.perf_counter() - t0

        # Verify results match
        np.testing.assert_allclose(cpp_result, py_result, rtol=1e-4)

        speedup = t_py / t_cpp if t_cpp > 0 else float("inf")
        print(f"\nPerformance: C++={t_cpp:.4f}s, Python={t_py:.4f}s, "
              f"speedup={speedup:.1f}x")
