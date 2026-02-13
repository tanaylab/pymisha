"""Tests for parallel chunking support in gsynth_train and gsynth_sample.

Ports the R test-gsynth-parallel-helper.R test patterns to Python.
"""

import os
import tempfile

import numpy as np
import pytest

import pymisha as pm
from pymisha.gsynth import (
    _chunk_intervals,
    _compute_total_bases,
    _generate_chunk_seeds,
    _merge_train_results,
    _should_parallelize,
)


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize the example database for each test."""
    pm.gdb_init_examples()
    yield


# ============================================================================
# Helper function tests
# ============================================================================


class TestComputeTotalBases:
    """Tests for _compute_total_bases."""

    def test_single_interval(self):
        iv = pm.gintervals(["1"], [0], [1000])
        assert _compute_total_bases(iv) == 1000

    def test_multiple_intervals(self):
        iv = pm.gintervals(["1", "1", "2"], [0, 5000, 0], [1000, 6000, 500])
        assert _compute_total_bases(iv) == 2500


class TestShouldParallelize:
    """Tests for _should_parallelize."""

    def test_returns_false_when_disabled(self):
        iv = pm.gintervals_all()
        do_par, cores = _should_parallelize(iv, False, 4)
        assert do_par is False
        assert cores == 1

    def test_returns_false_for_small_genome(self):
        iv = pm.gintervals(["1"], [0], [1000])
        do_par, cores = _should_parallelize(iv, True, 4)
        assert do_par is False
        assert cores == 1

    def test_returns_false_for_single_row(self):
        """Even if total bases exceed threshold, a single row cannot be split."""
        iv = pm.gintervals(["1"], [0], [500000])
        do_par, cores = _should_parallelize(iv, True, 4, max_chunk_size=100)
        assert do_par is False
        assert cores == 1

    def test_returns_true_for_large_genome_many_rows(self):
        """Multiple rows exceeding threshold should parallelize."""
        iv = pm.gintervals(
            ["1", "1", "1"],
            [0, 100000, 200000],
            [100000, 200000, 300000],
        )
        do_par, cores = _should_parallelize(iv, True, 4, max_chunk_size=100)
        assert do_par is True
        assert cores == 3  # capped at number of rows

    def test_num_cores_capped_at_rows(self):
        iv = pm.gintervals(["1", "1"], [0, 1000], [1000, 2000])
        do_par, cores = _should_parallelize(iv, True, 10, max_chunk_size=100)
        assert do_par is True
        assert cores == 2

    def test_num_cores_defaults_to_cpu_count(self):
        """When num_cores=None, should use cpu_count (just verify it works)."""
        iv = pm.gintervals(
            ["1", "1", "1"],
            [0, 100000, 200000],
            [100000, 200000, 300000],
        )
        do_par, cores = _should_parallelize(iv, True, None, max_chunk_size=100)
        assert do_par is True
        assert cores >= 1


class TestChunkIntervals:
    """Tests for _chunk_intervals."""

    def test_equal_chunks(self):
        iv = pm.gintervals(
            ["1", "1", "1", "1"],
            [0, 100, 200, 300],
            [100, 200, 300, 400],
        )
        chunks = _chunk_intervals(iv, 2)
        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2

    def test_more_chunks_than_rows(self):
        iv = pm.gintervals(["1", "1"], [0, 100], [100, 200])
        chunks = _chunk_intervals(iv, 5)
        assert len(chunks) == 2
        assert len(chunks[0]) == 1
        assert len(chunks[1]) == 1

    def test_uneven_split(self):
        iv = pm.gintervals(
            ["1", "1", "1"],
            [0, 100, 200],
            [100, 200, 300],
        )
        chunks = _chunk_intervals(iv, 2)
        assert len(chunks) == 2
        # 3 rows, 2 chunks => 2+1 or 1+2
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 3

    def test_preserves_data(self):
        iv = pm.gintervals(
            ["1", "2"],
            [0, 0],
            [1000, 500],
        )
        chunks = _chunk_intervals(iv, 2)
        assert str(chunks[0].iloc[0]["chrom"]) == "1"
        assert str(chunks[1].iloc[0]["chrom"]) == "2"


class TestGenerateChunkSeeds:
    """Tests for _generate_chunk_seeds."""

    def test_none_seed_returns_nones(self):
        seeds = _generate_chunk_seeds(None, 5)
        assert len(seeds) == 5
        assert all(s is None for s in seeds)

    def test_deterministic(self):
        seeds1 = _generate_chunk_seeds(42, 5)
        seeds2 = _generate_chunk_seeds(42, 5)
        assert seeds1 == seeds2

    def test_different_seeds_differ(self):
        seeds1 = _generate_chunk_seeds(42, 5)
        seeds2 = _generate_chunk_seeds(99, 5)
        assert seeds1 != seeds2

    def test_correct_count(self):
        seeds = _generate_chunk_seeds(42, 3)
        assert len(seeds) == 3
        assert all(isinstance(s, int) for s in seeds)


class TestMergeTrainResults:
    """Tests for _merge_train_results."""

    def test_single_chunk_roundtrip(self):
        """Merging a single chunk should produce same counts."""
        counts = [np.ones((1024, 4), dtype=np.float64) * 10]
        result = _merge_train_results(
            [{
                "counts": counts,
                "total_kmers": 100,
                "per_bin_kmers": np.array([100.0]),
                "total_masked": 5,
                "total_n": 3,
            }],
            total_bins=1,
            pseudocount=1.0,
        )
        assert result["total_kmers"] == 100
        assert result["total_masked"] == 5
        assert result["total_n"] == 3
        np.testing.assert_array_equal(result["counts"][0], counts[0])

    def test_two_chunks_sum(self):
        """Merging two chunks should sum counts."""
        c1 = [np.ones((1024, 4), dtype=np.float64) * 5]
        c2 = [np.ones((1024, 4), dtype=np.float64) * 7]
        result = _merge_train_results(
            [
                {
                    "counts": c1,
                    "total_kmers": 50,
                    "per_bin_kmers": np.array([50.0]),
                    "total_masked": 2,
                    "total_n": 1,
                },
                {
                    "counts": c2,
                    "total_kmers": 70,
                    "per_bin_kmers": np.array([70.0]),
                    "total_masked": 3,
                    "total_n": 2,
                },
            ],
            total_bins=1,
            pseudocount=1.0,
        )
        assert result["total_kmers"] == 120
        assert result["total_masked"] == 5
        assert result["total_n"] == 3
        np.testing.assert_array_almost_equal(
            result["counts"][0], np.ones((1024, 4)) * 12
        )

    def test_cdf_valid(self):
        """CDF should be monotonically increasing, ending at 1.0."""
        c1 = [np.random.RandomState(42).rand(1024, 4) * 100]
        result = _merge_train_results(
            [{
                "counts": c1,
                "total_kmers": 1000,
                "per_bin_kmers": np.array([1000.0]),
                "total_masked": 0,
                "total_n": 0,
            }],
            total_bins=1,
            pseudocount=1.0,
        )
        cdf = result["cdf"][0]
        # Last column should be exactly 1.0
        np.testing.assert_array_almost_equal(cdf[:, -1], 1.0)
        # CDF should be monotonically increasing along columns
        for row in range(cdf.shape[0]):
            for col in range(1, cdf.shape[1]):
                assert cdf[row, col] >= cdf[row, col - 1]


# ============================================================================
# gsynth_train parallel tests
# ============================================================================


class TestGsynthTrainParallel:
    """Tests for parallel gsynth_train."""

    def test_parallel_disabled(self):
        """With allow_parallel=False, should work identically to serial."""
        model = pm.gsynth_train(allow_parallel=False)
        assert isinstance(model, pm.GsynthModel)
        assert model.total_kmers > 0

    def test_small_genome_no_parallel(self):
        """Small genome should not trigger parallel even with allow_parallel=True."""
        model = pm.gsynth_train(
            intervals=pm.gintervals(["1"], [0], [10000]),
            allow_parallel=True,
            num_cores=2,
        )
        assert isinstance(model, pm.GsynthModel)
        assert model.total_kmers > 0

    def test_forced_parallel_matches_serial(self):
        """Parallel train with low threshold should give similar results to serial.

        We compare total_kmers and per-bin counts rather than exact CDF values,
        since the parallel path re-computes CDFs from merged counts.
        """
        intervals = pm.gintervals_all()

        # Serial
        model_serial = pm.gsynth_train(
            intervals=intervals,
            allow_parallel=False,
        )

        # Parallel (force by setting max_chunk_size very low)
        model_parallel = pm.gsynth_train(
            intervals=intervals,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,  # Force chunking
        )

        assert model_serial.n_dims == model_parallel.n_dims
        assert model_serial.total_bins == model_parallel.total_bins
        assert model_serial.total_kmers == model_parallel.total_kmers
        np.testing.assert_array_almost_equal(
            model_serial.per_bin_kmers,
            model_parallel.per_bin_kmers,
        )

    def test_forced_parallel_1d(self):
        """1D parallel train with forced chunking should produce valid model."""
        pm.gvtrack_create("test_vt_par", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt_par", "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                intervals=pm.gintervals_all(),
                iterator=200,
                allow_parallel=True,
                num_cores=2,
                max_chunk_size=100,
            )

            assert isinstance(model, pm.GsynthModel)
            assert model.n_dims == 1
            assert model.total_bins == 5
            assert model.total_kmers > 0
        finally:
            pm.gvtrack_rm("test_vt_par")

    def test_forced_parallel_with_mask(self):
        """Parallel train with mask should work correctly."""
        # Create a small mask
        mask = pm.gintervals(["1"], [0], [1000])
        intervals = pm.gintervals_all()

        model_serial = pm.gsynth_train(
            mask=mask,
            intervals=intervals,
            allow_parallel=False,
        )

        model_parallel = pm.gsynth_train(
            mask=mask,
            intervals=intervals,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        assert model_serial.total_kmers == model_parallel.total_kmers
        assert model_serial.total_masked == model_parallel.total_masked

    def test_parallel_num_cores_1_is_serial(self):
        """num_cores=1 should never parallelize, even with low threshold."""
        model = pm.gsynth_train(
            intervals=pm.gintervals_all(),
            allow_parallel=True,
            num_cores=1,
            max_chunk_size=100,
        )
        assert isinstance(model, pm.GsynthModel)
        assert model.total_kmers > 0


# ============================================================================
# gsynth_sample parallel tests
# ============================================================================


class TestGsynthSampleParallel:
    """Tests for parallel gsynth_sample."""

    @pytest.fixture
    def trained_model(self):
        """Train a model for sampling tests."""
        return pm.gsynth_train()

    def test_parallel_disabled(self, trained_model):
        """With allow_parallel=False, sampling should work normally."""
        seqs = pm.gsynth_sample(
            trained_model,
            intervals=pm.gintervals(["1"], [0], [1000]),
            seed=42,
            allow_parallel=False,
        )
        assert len(seqs) == 1
        assert len(seqs[0]) == 1000

    def test_small_genome_no_parallel(self, trained_model):
        """Small genome should not trigger parallel."""
        seqs = pm.gsynth_sample(
            trained_model,
            intervals=pm.gintervals(["1"], [0], [1000]),
            seed=42,
            allow_parallel=True,
            num_cores=2,
        )
        assert len(seqs) == 1
        assert len(seqs[0]) == 1000

    def test_forced_parallel_vector_output(self, trained_model):
        """Parallel sample with vector output should concatenate results."""
        intervals = pm.gintervals_all()
        n_intervals = len(intervals)

        seqs = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        # Should have one sequence per interval
        assert len(seqs) == n_intervals

        # All sequences should be non-empty strings
        for seq in seqs:
            assert isinstance(seq, str)
            assert len(seq) > 0

    def test_forced_parallel_fasta_output(self, trained_model):
        """Parallel sample with FASTA output should write combined file."""
        intervals = pm.gintervals_all()
        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            fasta_path = f.name

        try:
            result = pm.gsynth_sample(
                trained_model,
                output=fasta_path,
                output_format="fasta",
                intervals=intervals,
                seed=42,
                allow_parallel=True,
                num_cores=2,
                max_chunk_size=100,
            )

            assert result is None
            assert os.path.exists(fasta_path)
            # Read and verify FASTA content
            with open(fasta_path) as f:
                content = f.read()
            assert ">" in content  # Has FASTA headers
            lines = content.strip().split("\n")
            header_count = sum(1 for line in lines if line.startswith(">"))
            assert header_count == len(intervals)
        finally:
            if os.path.exists(fasta_path):
                os.unlink(fasta_path)

    def test_forced_parallel_n_samples(self, trained_model):
        """Parallel sample with n_samples > 1 should return correct count."""
        intervals = pm.gintervals(
            ["1", "2"],
            [0, 0],
            [1000, 500],
        )

        seqs = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            n_samples=3,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        # 2 intervals * 3 samples = 6 sequences
        assert len(seqs) == 6

    def test_forced_parallel_reproducible_with_seed(self, trained_model):
        """Same seed should produce same sequences in parallel mode."""
        intervals = pm.gintervals_all()

        seqs1 = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        seqs2 = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        assert seqs1 == seqs2

    def test_forced_parallel_different_seeds_differ(self, trained_model):
        """Different seeds should produce different sequences."""
        intervals = pm.gintervals_all()

        seqs1 = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        seqs2 = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            seed=99,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        assert seqs1 != seqs2

    def test_parallel_with_mask_copy(self, trained_model):
        """Parallel sampling with mask_copy should work."""
        intervals = pm.gintervals_all()
        mask_copy = pm.gintervals(["1"], [100], [200])

        seqs = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            mask_copy=mask_copy,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        assert len(seqs) == len(intervals)

    def test_parallel_num_cores_1_is_serial(self, trained_model):
        """num_cores=1 should never parallelize."""
        seqs = pm.gsynth_sample(
            trained_model,
            intervals=pm.gintervals_all(),
            seed=42,
            allow_parallel=True,
            num_cores=1,
            max_chunk_size=100,
        )
        assert len(seqs) == len(pm.gintervals_all())


# ============================================================================
# Integration tests
# ============================================================================


class TestGsynthParallelIntegration:
    """Integration tests verifying parallel and serial paths produce
    equivalent results."""

    def test_train_and_sample_roundtrip_parallel(self):
        """Train in parallel, sample in parallel, verify sequences are valid."""
        intervals = pm.gintervals_all()

        model = pm.gsynth_train(
            intervals=intervals,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        seqs = pm.gsynth_sample(
            model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        assert len(seqs) == len(intervals)
        for seq in seqs:
            assert set(seq).issubset({"A", "C", "G", "T", "N"})

    def test_serial_parallel_sample_same_model(self):
        """Samples from same model via serial and parallel paths should be
        valid DNA.

        Note: We do NOT require exact equality since the parallel path
        splits intervals differently which can affect Markov context
        at chunk boundaries. We only verify both produce valid sequences.
        """
        model = pm.gsynth_train()
        intervals = pm.gintervals_all()

        seqs_serial = pm.gsynth_sample(
            model,
            intervals=intervals,
            seed=42,
            allow_parallel=False,
        )

        seqs_parallel = pm.gsynth_sample(
            model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        assert len(seqs_serial) == len(seqs_parallel)
        for s in seqs_serial:
            assert set(s).issubset({"A", "C", "G", "T", "N"})
        for s in seqs_parallel:
            assert set(s).issubset({"A", "C", "G", "T", "N"})

    def test_parallel_preserves_sequence_lengths(self):
        """Parallel path should produce sequences matching interval lengths."""
        intervals = pm.gintervals(
            ["1", "1", "2"],
            [0, 10000, 0],
            [5000, 15000, 3000],
        )

        model = pm.gsynth_train(
            intervals=intervals,
            allow_parallel=False,
        )

        seqs = pm.gsynth_sample(
            model,
            intervals=intervals,
            seed=42,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        expected_lengths = [5000, 5000, 3000]
        assert len(seqs) == 3
        for seq, expected_len in zip(seqs, expected_lengths, strict=False):
            assert len(seq) == expected_len
