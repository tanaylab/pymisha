"""Tests for parallel chunking support in gsynth_train and gsynth_sample.

Ports the R test-gsynth-parallel-helper.R test patterns to Python.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha.gsynth import (
    _chunk_intervals,
    _compute_total_bases,
    _generate_chunk_seeds,
    _max_chunk_bases,
    _merge_train_results,
    _n_workers,
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


class TestChunkIntervals:
    """Tests for _chunk_intervals: chunks of whole rows, in order, of at most max_bases bases."""

    def _sizes(self, chunks, iterator=None):
        it = iterator or 1
        return [int((((c["end"] - 1) // it - c["start"] // it + 1) * it).sum()) for c in chunks]

    def test_bound_respected_and_order_preserved(self):
        rng = np.random.RandomState(5)
        starts = np.sort(rng.randint(0, 400000, 300))
        iv = pm.gintervals(["1"] * 300, starts, starts + rng.randint(1, 3000, 300))
        for bound in (1, 2999, 3000, 10000, 123457, 10**9):
            chunks = _chunk_intervals(iv, bound)
            assert len(chunks) >= 1 and all(len(c) > 0 for c in chunks)
            joined = pd.concat(chunks, ignore_index=True)
            pd.testing.assert_frame_equal(joined, iv.reset_index(drop=True))
            for size, c in zip(self._sizes(chunks), chunks, strict=True):
                assert size <= bound or len(c) == 1
            # greedy: each chunk would pass the bound with the next row
            sizes = (iv["end"] - iv["start"]).tolist()
            pos = 0
            for c in chunks[:-1]:
                pos += len(c)
                assert sum(sizes[pos - len(c):pos + 1]) > bound

    def test_oversize_row_is_a_chunk_of_its_own(self):
        iv = pm.gintervals(["1"] * 5, [0, 5000, 5100, 5200, 100000], [5000, 5100, 5200, 90000, 100100])
        chunks = _chunk_intervals(iv, 1000)
        assert [c["start"].tolist() for c in chunks] == [[0], [5000, 5100], [5200], [100000]]

    def test_rows_count_the_iterator_bins_they_span(self):
        # [150, 250) spans 2 bins of 100: 200 bases at iterator 100, 100 without an iterator
        iv = pm.gintervals(["1"] * 4, [150, 1150, 2150, 3150], [250, 1250, 2250, 3250])
        assert self._sizes(_chunk_intervals(iv, 400, 100), 100) == [400, 400]
        assert len(_chunk_intervals(iv, 400)) == 1

    def test_one_chunk_is_the_intervals_as_given(self):
        iv = pm.gintervals(["1", "2"], [0, 0], [1000, 500])
        chunks = _chunk_intervals(iv, 10**6)
        assert len(chunks) == 1 and chunks[0] is iv

    def test_default_bound_is_max_data_size_times_iterator(self):
        old = pm.CONFIG.get("max_data_size")
        try:
            pm.CONFIG["max_data_size"] = 1234
            assert _max_chunk_bases(None, 200) == 1234 * 200
            assert _max_chunk_bases(None, None) == 1234
            assert _max_chunk_bases(5000, 200) == 5000
            with pytest.raises(ValueError, match="max_chunk_size"):
                _max_chunk_bases(0, 200)
        finally:
            pm.CONFIG["max_data_size"] = old

    def test_workers_do_not_change_the_chunks(self):
        assert _n_workers(False, 8, 10) == 1
        assert _n_workers(True, 8, 1) == 1
        assert _n_workers(True, 8, 3) == 3
        assert _n_workers(True, 2, 10) == 2


class TestGenerateChunkSeeds:
    """Tests for _generate_chunk_seeds."""

    def test_none_seed_draws_from_numpys_generator(self):
        np.random.seed(3)
        seeds1 = _generate_chunk_seeds(None, 5)
        np.random.seed(3)
        seeds2 = _generate_chunk_seeds(None, 5)
        assert seeds1 == seeds2
        assert all(isinstance(s, int) for s in seeds1)

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
        counts = [np.full((1024, 4), 10, dtype=np.uint64)]
        result = _merge_train_results(
            [{
                "counts": counts,
                "total_kmers": 100,
                "per_bin_kmers": np.array([100], dtype=np.uint64),
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
        c1 = [np.full((1024, 4), 5, dtype=np.uint64)]
        c2 = [np.full((1024, 4), 7, dtype=np.uint64)]
        result = _merge_train_results(
            [
                {
                    "counts": c1,
                    "total_kmers": 50,
                    "per_bin_kmers": np.array([50], dtype=np.uint64),
                    "total_masked": 2,
                    "total_n": 1,
                },
                {
                    "counts": c2,
                    "total_kmers": 70,
                    "per_bin_kmers": np.array([70], dtype=np.uint64),
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
        np.testing.assert_array_equal(result["counts"][0], np.full((1024, 4), 12, dtype=np.uint64))

    def test_cdf_valid(self):
        """CDF should be monotonically increasing, ending at 1.0."""
        c1 = [np.random.RandomState(60427).randint(0, 100, (1024, 4)).astype(np.uint64)]
        result = _merge_train_results(
            [{
                "counts": c1,
                "total_kmers": 1000,
                "per_bin_kmers": np.array([1000], dtype=np.uint64),
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
        """Parallel train in many chunks counts the k-mers of the serial one (every field is
        compared in TestChunksDoNotChangeResults).
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
            seed=60427,
            allow_parallel=False,
        )
        assert len(seqs) == 1
        assert len(seqs[0]) == 1000

    def test_small_genome_no_parallel(self, trained_model):
        """Small genome should not trigger parallel."""
        seqs = pm.gsynth_sample(
            trained_model,
            intervals=pm.gintervals(["1"], [0], [1000]),
            seed=60427,
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
            seed=60427,
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
                seed=60427,
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
            seed=60427,
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
            seed=60427,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        seqs2 = pm.gsynth_sample(
            trained_model,
            intervals=intervals,
            seed=60427,
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
            seed=60427,
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
            seed=60427,
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
            seed=60427,
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
            seed=60427,
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

        Note: the sequences differ: with several chunks each draws from a seed of its own
        (TestChunksDoNotChangeResults). We only verify both produce valid sequences.
        """
        model = pm.gsynth_train()
        intervals = pm.gintervals_all()

        seqs_serial = pm.gsynth_sample(
            model,
            intervals=intervals,
            seed=60427,
            allow_parallel=False,
        )

        seqs_parallel = pm.gsynth_sample(
            model,
            intervals=intervals,
            seed=60427,
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
            seed=60427,
            allow_parallel=True,
            num_cores=2,
            max_chunk_size=100,
        )

        expected_lengths = [5000, 5000, 3000]
        assert len(seqs) == 3
        for seq, expected_len in zip(seqs, expected_lengths, strict=False):
            assert len(seq) == expected_len


# ============================================================================
# Chunks do not change the results
# ============================================================================


MODEL_FIELDS = ("k", "n_dims", "dim_sizes", "dim_specs", "total_bins", "total_kmers", "per_bin_kmers",
                "total_masked", "total_n", "pseudocount", "min_obs", "iterator", "prior_mode",
                "prior_matrix", "marginal_fallbacks", "sparse_bins")


def _assert_same_model(a, b):
    for f in MODEL_FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, f
            np.testing.assert_array_equal(x, y, err_msg=f)
        else:
            assert x == y, f
    for key in ("counts", "cdf"):
        xs, ys = a.model_data[key], b.model_data[key]
        assert len(xs) == len(ys)
        for x, y in zip(xs, ys, strict=True):
            assert x.dtype == y.dtype, key
            np.testing.assert_array_equal(x, y, err_msg=key)  # NaN == NaN here


@pytest.fixture
def chunk_vtracks():
    pm.gvtrack_create("chunk_vt_d", "dense_track", "avg")
    pm.gvtrack_create("chunk_vt_g", None, "kmer.frac", kmer="G")
    yield
    pm.gvtrack_rm("chunk_vt_d")
    pm.gvtrack_rm("chunk_vt_g")


@pytest.fixture
def many_intervals():
    """300 intervals not aligned to the iterator, with overlaps and touching ones."""
    starts = list(range(0, 400000, 2000)) + list(range(0, 300000, 3000))
    ends = list(range(1500, 401500, 2000)) + list(range(2900, 302900, 3000))
    iv = pm.gintervals(["1"] * 200 + ["2"] * 100, starts, ends)
    extra = pm.gintervals(["1", "1", "X"], [1000, 9000, 0], [4000, 9500, 150000])
    return pd.concat([iv, extra], ignore_index=True)


DIM_D = {"expr": "chunk_vt_d", "breaks": [0, 0.1, 0.2, 0.4, 1]}
DIM_G = {"expr": "chunk_vt_g", "breaks": [0, 0.2, 0.25, 0.3, 1],
         "bin_merge": [{"from": 0.3, "to": [0.25, 0.3]}]}


class TestChunksDoNotChangeResults:

    @pytest.mark.parametrize("allow_parallel,num_cores", [(False, None), (True, 1), (True, 3), (True, 8)])
    @pytest.mark.parametrize("max_chunk_size", [1, 5000, 60000])
    def test_train_chunked_equals_unchunked(self, chunk_vtracks, many_intervals, allow_parallel,
                                            num_cores, max_chunk_size):
        kw = {"intervals": many_intervals, "iterator": 100, "k": 3, "min_obs": 2000,
              "mask": pm.gintervals(["1", "X"], [5000, 100], [20000, 150000])}
        with pytest.warns(UserWarning, match="fewer than 2000"):
            ref = pm.gsynth_train(DIM_D, DIM_G, allow_parallel=False, **kw)
        assert len(ref.sparse_bins) > 0
        with pytest.warns(UserWarning, match="fewer than 2000"):
            model = pm.gsynth_train(DIM_D, DIM_G, allow_parallel=allow_parallel, num_cores=num_cores,
                                    max_chunk_size=max_chunk_size, **kw)
        _assert_same_model(model, ref)

    @pytest.mark.parametrize("prior,pseudocount", [("marginal", 1.0), ("global", 0.5), ("uniform", 0.0),
                                                   ("explicit", 2.0)])
    def test_train_equals_the_cpp_model(self, chunk_vtracks, many_intervals, prior, pseudocount):
        """The CDF and prior gsynth_train computes from the summed counts of its chunks are those
        the C++ model computes (pm_gsynth_train), bit for bit."""
        from pymisha._shared import _df2pymisha, _pymisha
        from pymisha.gsynth import _canonic_keep_touching, _extract_bin_data

        if prior == "explicit":
            prior = np.random.RandomState(1).rand(16, 4) * np.array([1, 1, 0, 1])
            prior[3] = 0  # a row of zeros: uniform
        model = pm.gsynth_train(DIM_D, DIM_G, intervals=many_intervals, iterator=100, k=2, prior=prior,
                                pseudocount=pseudocount, max_chunk_size=20000, num_cores=3)
        iv = _canonic_keep_touching(many_intervals)
        bins, starts, chroms, breaks, bin_map, _, _, _ = _extract_bin_data(model.dim_specs, iv, 100)
        cpp = _pymisha.pm_gsynth_train(
            _df2pymisha(iv), bins, starts, chroms, breaks, bin_map, None, float(pseudocount), 2, 100,
            "explicit" if isinstance(prior, np.ndarray) else prior,
            np.ascontiguousarray(prior, dtype=np.float64) if isinstance(prior, np.ndarray) else None)
        np.testing.assert_array_equal(model.prior_matrix, cpp["prior"])
        assert model.per_bin_kmers.dtype == cpp["per_bin_kmers"].dtype
        np.testing.assert_array_equal(model.per_bin_kmers, cpp["per_bin_kmers"])
        for ours, theirs in zip(model.model_data["counts"], cpp["counts"], strict=True):
            assert ours.dtype == theirs.dtype
            np.testing.assert_array_equal(ours, theirs)
        for ours, theirs in zip(model.model_data["cdf"], cpp["cdf"], strict=True):
            np.testing.assert_array_equal(ours, theirs)
        if pseudocount == 0:
            assert any(np.isnan(c).any() for c in model.model_data["cdf"])

    def test_beyond_max_data_size_in_one_process(self, chunk_vtracks, many_intervals):
        """More iterator bins than CONFIG['max_data_size'] are processed in chunks, in this process
        too, with the results of no limit."""
        kw = {"intervals": many_intervals, "iterator": 1, "k": 1}
        ref = pm.gsynth_train(DIM_G, **kw)
        ref_seqs = pm.gsynth_sample(ref, intervals=many_intervals, seed=17, max_chunk_size=200000)
        old = pm.CONFIG.get("max_data_size")
        try:
            pm.CONFIG["max_data_size"] = 200000
            assert len(_chunk_intervals(many_intervals, 200000)) > 2
            for allow_parallel in (False, True):
                model = pm.gsynth_train(DIM_G, allow_parallel=allow_parallel, num_cores=2, **kw)
                _assert_same_model(model, ref)
                seqs = pm.gsynth_sample(model, intervals=many_intervals, seed=17,
                                        allow_parallel=allow_parallel, num_cores=2)
                assert seqs == ref_seqs
        finally:
            pm.CONFIG["max_data_size"] = old

    def test_sample_does_not_depend_on_the_workers(self, chunk_vtracks, many_intervals):
        model = pm.gsynth_train(DIM_D, intervals=many_intervals, iterator=100, k=3)
        kw = {"intervals": many_intervals, "max_chunk_size": 30000, "n_samples": 2,
              "mask_copy": pm.gintervals(["1"], [100], [20000])}
        runs = [(False, None), (True, 1), (True, 2), (True, 5)]
        seeded = [pm.gsynth_sample(model, seed=60427, allow_parallel=p, num_cores=c, **kw) for p, c in runs]
        assert all(s == seeded[0] for s in seeded[1:])
        unseeded = []
        for p, c in runs:
            np.random.seed(9)
            unseeded.append(pm.gsynth_sample(model, allow_parallel=p, num_cores=c, **kw))
        assert all(s == unseeded[0] for s in unseeded[1:])
        assert unseeded[0] != seeded[0]

    def test_each_chunk_draws_from_its_own_seed(self, chunk_vtracks, many_intervals):
        """Several chunks: the sequences of each are those of sampling it alone with its seed, drawn
        from *seed* (so they differ from one chunk's)."""
        from pymisha.gsynth import _genome_order

        model = pm.gsynth_train(DIM_D, intervals=many_intervals, iterator=100, k=3)
        chunks = _chunk_intervals(_genome_order(many_intervals), 30000, 100)
        assert len(chunks) > 5
        seqs = pm.gsynth_sample(model, intervals=many_intervals, seed=5, max_chunk_size=30000, num_cores=3)
        alone = [s for chunk, chunk_seed in zip(chunks, _generate_chunk_seeds(5, len(chunks)), strict=True)
                 for s in pm.gsynth_sample(model, intervals=chunk, seed=chunk_seed)]
        assert seqs == alone
        assert seqs != pm.gsynth_sample(model, intervals=many_intervals, seed=5)
