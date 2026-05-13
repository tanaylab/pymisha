"""Tests for gsynth_forbid_kmer. 1:1 port of R test-gsynth-forbid.R, plus seeding caveat coverage."""

from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(scope="module")
def small_model():
    """A small k=2 0D model for fast forbid tests."""
    pm.gdb_init_examples()
    yield pm.gsynth_train(
        intervals=pm.gintervals("1", 0, 50_000),
        iterator=200,
        k=2,
    )


class TestGsynthForbidKmer:
    def test_basic_forbid_zeroes_transitions(self, small_model):
        out = pm.gsynth_forbid_kmer(small_model, "CG", check=False)
        # Validate: for every bin, every (k+1)-mer containing 'CG' has count 0.
        for co in out.model_data["counts"]:
            if co is None:
                continue
            n_states = co.shape[0]
            k = int(round(np.log(n_states) / np.log(4)))
            for state in range(n_states):
                # Decompose state to bases.
                bases = []
                tmp = state
                for _ in range(k):
                    bases.insert(0, tmp % 4)
                    tmp //= 4
                for nb in range(4):
                    full = bases + [nb]
                    # Check if 'CG' substring (C=1, G=2) exists.
                    has_cg = any(full[i] == 1 and full[i + 1] == 2 for i in range(len(full) - 1))
                    if has_cg:
                        assert co[state, nb] == 0

    def test_does_not_mutate_input(self, small_model):
        counts_before = [c.copy() if c is not None else None for c in small_model.model_data["counts"]]
        cdf_before    = [c.copy() if c is not None else None for c in small_model.model_data["cdf"]]
        out = pm.gsynth_forbid_kmer(small_model, "CG", check=False)
        for before, after in zip(counts_before, small_model.model_data["counts"], strict=True):
            if before is None:
                assert after is None
            else:
                np.testing.assert_array_equal(before, after)
        for before, after in zip(cdf_before, small_model.model_data["cdf"], strict=True):
            if before is None:
                assert after is None
            else:
                np.testing.assert_array_equal(before, after)
        # Returned model has a distinct model_data dict.
        assert out.model_data is not small_model.model_data

    def test_longer_pattern_gcgc(self, small_model):
        # k=2 so k+1=3; "GCGC" len 4 should error.
        with pytest.raises(ValueError, match=r"exceeds model\.k \+ 1"):
            pm.gsynth_forbid_kmer(small_model, "GCGC")

    def test_invalid_pattern_raises(self, small_model):
        with pytest.raises(ValueError, match=r"non-empty DNA over ACGT"):
            pm.gsynth_forbid_kmer(small_model, "N")
        with pytest.raises(ValueError, match=r"non-empty DNA over ACGT"):
            pm.gsynth_forbid_kmer(small_model, "")
        with pytest.raises(ValueError, match=r"non-empty DNA over ACGT"):
            pm.gsynth_forbid_kmer(small_model, "ACGX")

    def test_check_false_suppresses_summary(self, small_model, capfd):
        _ = pm.gsynth_forbid_kmer(small_model, "CG", check=False)
        out, err = capfd.readouterr()
        assert "gsynth_forbid_kmer" not in (out + err)

    def test_check_true_prints_summary(self, small_model, capfd):
        _ = pm.gsynth_forbid_kmer(small_model, "CG", check=True)
        _, err = capfd.readouterr()
        assert "gsynth_forbid_kmer('CG')" in err
        assert "transitions" in err

    def test_integration_with_unaligned_intervals(self):
        """forbid_kmer survives unaligned interval starts (1D stratified path).

        The original R #94 scenario was a 0D model on unaligned starts. PyMisha's
        0D + unaligned path has a separate latent bug (INT64_MAX overflow in
        the C++ bin lookup - see task #14) that causes uniform random sampling
        regardless of CDF, so the 0D version of this test would always fail
        even with a working forbid_kmer. This 1D-stratified variant exercises
        the same code path (unaligned start through the bin-extract pipeline)
        without tripping the 0D-specific overflow.
        """
        pm.gdb_init_examples()
        pm.gvtrack_create("forbid_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "forbid_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 50_000),
                iterator=200,
                k=2,
            )
            forbid = pm.gsynth_forbid_kmer(model, "CG", check=False)
            # Unaligned start (not a multiple of iterator).
            ivs = pm.gintervals("1", 64, 64 + 2000)
            seqs = pm.gsynth_sample(
                forbid, intervals=ivs, iterator=200, seed=60427, output_format="vector",
            )
            assert seqs is not None
            # Approximation: most positions should respect the forbid (k=2 model, small).
            # We don't assert *zero* CG because the seed window can carry one through.
            combined = "".join(seqs)
            cg_rate = combined.count("CG") / max(1, len(combined))
            assert cg_rate < 0.005  # baseline expected ~1-5% CG in random DNA
        finally:
            pm.gvtrack_rm("forbid_vt")

    def test_trapped_state_uniform_fallback(self, small_model):
        """States whose k-mer already contains the pattern fall back to uniform 0.25."""
        out = pm.gsynth_forbid_kmer(small_model, "A", check=False)
        # k=2 model: state index 0 == "AA". Every extension contains an A, so
        # the state is "trapped" and must use the uniform fallback CDF.
        cdf_0 = out.model_data["cdf"][0]
        np.testing.assert_allclose(cdf_0[0], [0.25, 0.5, 0.75, 1.0], rtol=0, atol=1e-12)
