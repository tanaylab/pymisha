"""Tests for gsynth functions."""

import os
import tempfile

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize the example database for each test."""
    pm.gdb_init_examples()
    yield


# ============================================================================
# gsynth_bin_map
# ============================================================================


class TestGsynthBinMap:
    """Tests for gsynth_bin_map function."""

    def test_bin_map_identity(self):
        """With no merge ranges, bin_map is identity."""
        breaks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bm = pm.gsynth_bin_map(breaks, [])
        np.testing.assert_array_equal(bm, [0, 1, 2, 3, 4])

    def test_bin_map_single_merge(self):
        """Merge last bin into second-to-last."""
        breaks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bm = pm.gsynth_bin_map(breaks, [{"from": (0.8, 1.0), "to": (0.6, 0.8)}])
        np.testing.assert_array_equal(bm, [0, 1, 2, 3, 3])

    def test_bin_map_multiple_merges(self):
        """Merge first and last bins into middle."""
        breaks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bm = pm.gsynth_bin_map(breaks, [
            {"from": (0.0, 0.2), "to": (0.2, 0.4)},
            {"from": (0.8, 1.0), "to": (0.6, 0.8)},
        ])
        np.testing.assert_array_equal(bm, [1, 1, 2, 3, 3])

    def test_bin_map_too_few_breaks(self):
        """Breaks with fewer than 2 elements raises error."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            pm.gsynth_bin_map([0.5], [])


# ============================================================================
# gsynth_train
# ============================================================================


class TestGsynthTrain:
    """Tests for gsynth_train function."""

    def test_train_1d_basic(self):
        """Train a 1D model with a dense track."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            assert isinstance(model, pm.GsynthModel)
            assert model.n_dims == 1
            assert model.dim_sizes == [5]
            assert model.total_bins == 5
            assert model.total_kmers > 0
            assert len(model.model_data["cdf"]) == 5
            assert len(model.model_data["counts"]) == 5
            # CDF arrays should be 1024 x 4
            assert model.model_data["cdf"][0].shape == (1024, 4)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_train_2d(self):
        """Train a 2D model."""
        pm.gvtrack_create("test_vt1", "dense_track", "avg")
        pm.gvtrack_create("test_vt2", "dense_track", "min")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt1", "breaks": [0, 0.5, 1.0]},
                {"expr": "test_vt2", "breaks": [0, 0.3, 0.7, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            assert model.n_dims == 2
            assert model.dim_sizes == [2, 3]
            assert model.total_bins == 6
            assert len(model.model_data["cdf"]) == 6
        finally:
            pm.gvtrack_rm("test_vt1")
            pm.gvtrack_rm("test_vt2")

    def test_train_requires_expr(self):
        """dim_spec without 'expr' raises ValueError."""
        with pytest.raises(ValueError, match="expr"):
            pm.gsynth_train({"breaks": [0, 1]})

    def test_train_requires_breaks(self):
        """dim_spec without 'breaks' raises ValueError."""
        with pytest.raises(ValueError, match="breaks"):
            pm.gsynth_train({"expr": "dense_track"})

    def test_train_requires_dict(self):
        """Non-dict dim_spec raises TypeError."""
        with pytest.raises(TypeError, match="dict"):
            pm.gsynth_train("not_a_dict")

    def test_train_repr(self):
        """Model repr doesn't error."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )
            s = repr(model)
            assert "Markov" in s
            assert "Dimensions: 1" in s
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# gsynth_save / gsynth_load
# ============================================================================


class TestGsynthSaveLoad:
    """Tests for gsynth_save and gsynth_load functions."""

    def test_save_load_roundtrip(self):
        """Saved and loaded model should match."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                path = f.name
            try:
                pm.gsynth_save(model, path)
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0

                loaded = pm.gsynth_load(path)
                assert isinstance(loaded, pm.GsynthModel)
                assert loaded.n_dims == model.n_dims
                assert loaded.total_bins == model.total_bins
                assert loaded.total_kmers == model.total_kmers
                assert loaded.pseudocount == model.pseudocount
                # Compare CDFs
                for orig, load in zip(model.model_data["cdf"],
                                      loaded.model_data["cdf"], strict=False):
                    np.testing.assert_array_almost_equal(orig, load)
            finally:
                os.unlink(path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_save_non_model_raises(self):
        """Saving non-GsynthModel raises TypeError."""
        with pytest.raises(TypeError, match="GsynthModel"):
            pm.gsynth_save("not_a_model", "/tmp/test.pkl")

    def test_load_non_model_raises(self):
        """Loading non-GsynthModel file raises TypeError."""
        import pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a model"}, f)
            path = f.name
        try:
            with pytest.raises(TypeError, match="GsynthModel"):
                pm.gsynth_load(path)
        finally:
            os.unlink(path)


# ============================================================================
# gsynth_sample
# ============================================================================


class TestGsynthSample:
    """Tests for gsynth_sample function."""

    def test_sample_vector_mode(self):
        """Sample returns list of strings in vector mode."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            ivs = pm.gintervals("1", 0, 1000)
            result = pm.gsynth_sample(model, intervals=ivs, iterator=200)

            assert isinstance(result, list)
            assert len(result) >= 1
            # Check that sequences contain only valid DNA bases
            for seq in result:
                assert len(seq) > 0
                assert all(c in "ACGTacgt" for c in seq), \
                    f"Invalid chars in: {seq[:50]}..."
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_fasta_mode(self):
        """Sample writes FASTA file correctly."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                output_path = f.name
            try:
                ivs = pm.gintervals("1", 0, 1000)
                pm.gsynth_sample(
                    model, output=output_path,
                    output_format="fasta", intervals=ivs, iterator=200,
                )

                assert os.path.exists(output_path)
                with open(output_path) as f:
                    content = f.read()
                assert content.startswith(">")
                assert "\n" in content
            finally:
                os.unlink(output_path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_seed_reproducible(self):
        """Same seed produces identical output."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            ivs = pm.gintervals("1", 0, 500)
            r1 = pm.gsynth_sample(
                model, intervals=ivs, iterator=200, seed=42
            )
            r2 = pm.gsynth_sample(
                model, intervals=ivs, iterator=200, seed=42
            )
            assert r1 == r2
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_different_seeds(self):
        """Different seeds produce different output."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            ivs = pm.gintervals("1", 0, 500)
            r1 = pm.gsynth_sample(
                model, intervals=ivs, iterator=200, seed=42
            )
            r2 = pm.gsynth_sample(
                model, intervals=ivs, iterator=200, seed=123
            )
            assert r1 != r2
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# gsynth_random
# ============================================================================


class TestGsynthRandom:
    """Tests for gsynth_random function."""

    def test_random_vector_mode(self):
        """Random sequences returned as list of strings."""
        ivs = pm.gintervals("1", 0, 1000)
        result = pm.gsynth_random(intervals=ivs, seed=42)

        assert isinstance(result, list)
        assert len(result) >= 1
        for seq in result:
            assert len(seq) == 1000
            assert all(c in "ACGTacgt" for c in seq)

    def test_random_custom_probs(self):
        """Custom nucleotide probabilities affect composition."""
        ivs = pm.gintervals("1", 0, 10000)
        # Bias toward A and T
        result = pm.gsynth_random(
            intervals=ivs,
            nuc_probs={"A": 0.5, "C": 0.0, "G": 0.0, "T": 0.5},
            seed=42,
        )
        seq = result[0]
        # After the initial 5 seed bases (which use uniform random),
        # the rest should not contain C or G
        seq_after_seed = seq[5:]
        assert "C" not in seq_after_seed, \
            f"Found C in post-seed sequence (first 20: {seq_after_seed[:20]})"
        assert "G" not in seq_after_seed, \
            f"Found G in post-seed sequence (first 20: {seq_after_seed[:20]})"
        assert "A" in seq_after_seed
        assert "T" in seq_after_seed

    def test_random_seed_reproducible(self):
        """Same seed gives identical random output."""
        ivs = pm.gintervals("1", 0, 1000)
        r1 = pm.gsynth_random(intervals=ivs, seed=42)
        r2 = pm.gsynth_random(intervals=ivs, seed=42)
        assert r1 == r2


# ============================================================================
# gsynth_replace_kmer
# ============================================================================


class TestGsynthReplaceKmer:
    """Tests for gsynth_replace_kmer function."""

    def test_replace_kmer_basic(self):
        """Replace a k-mer and verify it's absent."""
        ivs = pm.gintervals("1", 10000, 11000)
        target = "CG"
        replacement = "GC"
        result = pm.gsynth_replace_kmer(
            target, replacement, intervals=ivs
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        for seq in result:
            assert target not in seq

    def test_replace_kmer_length_mismatch(self):
        """Different-length target and replacement raises error."""
        ivs = pm.gintervals("1", 10000, 11000)
        with pytest.raises(ValueError, match="same length"):
            pm.gsynth_replace_kmer("CG", "GCA", intervals=ivs)

    def test_replace_kmer_composition_check(self):
        """Different composition raises error with check_composition."""
        ivs = pm.gintervals("1", 10000, 11000)
        with pytest.raises(ValueError, match="composition"):
            pm.gsynth_replace_kmer(
                "CG", "AA", intervals=ivs, check_composition=True
            )

    def test_replace_kmer_no_composition_check(self):
        """No composition check allows different composition."""
        ivs = pm.gintervals("1", 10000, 11000)
        result = pm.gsynth_replace_kmer(
            "CG", "AA", intervals=ivs, check_composition=False
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_replace_kmer_empty_raises(self):
        """Empty target or replacement raises error."""
        ivs = pm.gintervals("1", 10000, 11000)
        with pytest.raises(ValueError, match="empty"):
            pm.gsynth_replace_kmer("", "", intervals=ivs)


# ============================================================================
# Multi-dimensional stratification stress tests
# ============================================================================


class TestGsynthMultiDimStress:
    """Multi-dimensional stratification stress tests (2D and 3D)."""

    def test_train_3d_stratification(self):
        """Train a 3D model with three kmer-frac virtual tracks."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        pm.gvtrack_create("gc_vt", "dense_track", "avg")
        try:
            track_range = pm.gsummary(
                "dense_track",
                intervals=pm.gintervals("1", 0, 50000),
            )
            model = pm.gsynth_train(
                {"expr": "g_frac", "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                {"expr": "c_frac", "breaks": [0, 0.125, 0.25, 0.375, 0.5]},
                {
                    "expr": "gc_vt",
                    "breaks": np.linspace(
                        track_range["Min"], track_range["Max"], 3
                    ).tolist(),
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert model.n_dims == 3
            assert model.dim_sizes == [5, 4, 2]
            assert model.total_bins == 5 * 4 * 2
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")
            pm.gvtrack_rm("gc_vt")

    def test_sample_from_3d_model(self):
        """Sample from a 3D model and verify valid DNA output."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        pm.gvtrack_create("a_frac", None, "kmer.frac", kmer="A")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac", "breaks": [0, 0.2, 0.4, 0.6]},
                {"expr": "c_frac", "breaks": [0, 0.2, 0.4, 0.6]},
                {"expr": "a_frac", "breaks": [0, 0.2, 0.4, 0.6]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert model.total_bins == 27  # 3 * 3 * 3

            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 5000),
                iterator=200,
                seed=60427,
            )
            assert len(seqs) == 1
            assert len(seqs[0]) == 5000
            assert all(c in "ACGT" for c in seqs[0])
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")
            pm.gvtrack_rm("a_frac")

    def test_flat_index_2d(self):
        """Flat index for 2D model: total = dim1 * dim2."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac", "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                {"expr": "c_frac", "breaks": [0, 0.125, 0.25, 0.375, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert model.dim_sizes == [5, 4]
            assert model.total_bins == 20
            assert len(model.model_data["cdf"]) == 20
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")

    def test_flat_index_3d(self):
        """Flat index for 3D model: total = dim1 * dim2 * dim3."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        pm.gvtrack_create("a_frac", None, "kmer.frac", kmer="A")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac", "breaks": [0, 0.2, 0.4]},
                {"expr": "c_frac", "breaks": [0, 0.15, 0.3, 0.45]},
                {"expr": "a_frac", "breaks": [0, 0.25, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert model.dim_sizes == [2, 3, 2]
            assert model.total_bins == 12
            assert len(model.model_data["cdf"]) == 12
            assert len(model.per_bin_kmers) == 12
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")
            pm.gvtrack_rm("a_frac")

    def test_multidim_sampling_reproducible(self):
        """Same seed -> identical output for multi-dimensional model."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac + c_frac", "breaks": np.linspace(0, 1, 11).tolist()},
                {"expr": "g_frac", "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

            ivs = pm.gintervals("1", 0, 5000)
            s1 = pm.gsynth_sample(model, intervals=ivs, iterator=200, seed=42)
            s2 = pm.gsynth_sample(model, intervals=ivs, iterator=200, seed=42)
            assert s1 == s2
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")

    def test_per_bin_kmers_sum_1d(self):
        """per_bin_kmers should sum to total_kmers (1D)."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac + c_frac", "breaks": np.linspace(0, 1, 11).tolist()},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert int(np.sum(model.per_bin_kmers)) == model.total_kmers
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")

    def test_per_bin_kmers_sum_2d(self):
        """per_bin_kmers should sum to total_kmers (2D)."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac + c_frac", "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                {"expr": "g_frac", "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert int(np.sum(model.per_bin_kmers)) == model.total_kmers
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")

    def test_2d_gc_cg_user_case(self):
        """2D GC+CG stratification with bin_merge (user use case)."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        pm.gvtrack_create("cg_frac", None, "kmer.frac", kmer="CG", strand=1)
        try:
            gc_breaks = np.linspace(0, 1, 41).tolist()  # 40 bins
            cg_breaks = [0, 0.01, 0.02, 0.03, 0.04, 0.2]  # 5 bins

            model = pm.gsynth_train(
                {
                    "expr": "g_frac + c_frac",
                    "breaks": gc_breaks,
                    "bin_merge": [{"from": (0.7, float("inf")), "to": (0.675, 0.7)}],
                },
                {
                    "expr": "cg_frac",
                    "breaks": cg_breaks,
                    "bin_merge": [{"from": (0.04, float("inf")), "to": (0.03, 0.04)}],
                },
                intervals=pm.gintervals("1", 0, 100000),
                iterator=200,
            )

            assert model.n_dims == 2
            assert model.dim_specs[0]["num_bins"] == 40
            assert model.dim_specs[1]["num_bins"] == 5
            assert model.total_bins == 200

            # Verify bin_map for GC dimension (bins 28..39 -> 27)
            gc_bm = model.dim_specs[0]["bin_map"]
            assert all(gc_bm[i] == 27 for i in range(28, 40))

            # Verify bin_map for CG dimension (bin 4 -> 3)
            cg_bm = model.dim_specs[1]["bin_map"]
            assert int(cg_bm[4]) == 3

            # Sample and verify valid output
            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
                seed=60427,
            )
            assert len(seqs[0]) == 10000
            assert all(c in "ACGT" for c in seqs[0])
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")
            pm.gvtrack_rm("cg_frac")


# ============================================================================
# Complex iterator / bin_merge edge cases
# ============================================================================


class TestGsynthBinMergeAdvanced:
    """bin_merge edge cases and advanced merging scenarios."""

    def test_bin_map_target_range_maps_to_nearest(self):
        """Target range that spans a bin boundary maps to the enclosing bin."""
        breaks = np.linspace(0, 1, 11).tolist()  # 10 bins: [0,0.1), [0.1,0.2), ...
        # (0.123, 0.456) maps to the bin enclosing 0.123, i.e. bin 1 (0.1, 0.2)
        bm = pm.gsynth_bin_map(
            breaks,
            [{"from": (0.5, 1.0), "to": (0.123, 0.456)}],
        )
        # Bins 5..9 should all map to the same target
        target = int(bm[5])
        assert all(int(bm[i]) == target for i in range(5, 10))

    def test_train_with_bin_merge_per_dim(self):
        """bin_merge during training affects each dimension independently."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac + c_frac",
                    "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "bin_merge": [{"from": (0.8, float("inf")), "to": (0.6, 0.8)}],
                },
                {
                    "expr": "g_frac",
                    "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "bin_merge": [{"from": (0.4, float("inf")), "to": (0.3, 0.4)}],
                },
                intervals=pm.gintervals("1", 0, 100000),
                iterator=200,
            )

            # Check first dimension: bin 4 -> 3
            bm1 = model.dim_specs[0]["bin_map"]
            assert int(bm1[4]) == 3
            assert int(bm1[3]) == 3

            # Check second dimension: bin 4 -> 3
            bm2 = model.dim_specs[1]["bin_map"]
            assert int(bm2[4]) == 3
            assert int(bm2[3]) == 3
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")

    def test_sample_from_model_with_aggressive_bin_merge(self):
        """Sample from model with aggressive bin merging still produces valid DNA."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac + c_frac",
                    "breaks": np.linspace(0, 1, 11).tolist(),
                    "bin_merge": [
                        {"from": (float("-inf"), 0.2), "to": (0.2, 0.3)},
                        {"from": (0.7, float("inf")), "to": (0.6, 0.7)},
                    ],
                },
                intervals=pm.gintervals("1", 0, 100000),
                iterator=200,
            )

            bm = model.dim_specs[0]["bin_map"]
            assert all(bm[i] == 2 for i in range(2))
            assert all(bm[i] == 6 for i in range(7, 10))

            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 5000),
                iterator=200,
                seed=12345,
            )
            assert len(seqs[0]) == 5000
            assert all(c in "ACGT" for c in seqs[0])
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")


# ============================================================================
# Model save/load round-trip verification
# ============================================================================


class TestGsynthSaveLoadAdvanced:
    """Advanced save/load round-trip tests."""

    def test_save_load_preserves_all_fields(self):
        """Save/load preserves all fields including dim_specs and bin_map."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac + c_frac",
                    "breaks": np.linspace(0, 1, 11).tolist(),
                    "bin_merge": [{"from": (0.8, float("inf")), "to": (0.7, 0.8)}],
                },
                {"expr": "g_frac", "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                path = f.name
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)

                assert loaded.n_dims == model.n_dims
                assert loaded.dim_sizes == model.dim_sizes
                assert loaded.total_bins == model.total_bins
                assert loaded.total_kmers == model.total_kmers
                assert loaded.total_masked == model.total_masked
                assert loaded.total_n == model.total_n
                np.testing.assert_array_equal(
                    loaded.per_bin_kmers, model.per_bin_kmers
                )

                # dim_specs
                for d in range(model.n_dims):
                    assert (
                        loaded.dim_specs[d]["expr"]
                        == model.dim_specs[d]["expr"]
                    )
                    np.testing.assert_array_almost_equal(
                        loaded.dim_specs[d]["breaks"],
                        model.dim_specs[d]["breaks"],
                    )
                    assert (
                        loaded.dim_specs[d]["num_bins"]
                        == model.dim_specs[d]["num_bins"]
                    )
                    np.testing.assert_array_equal(
                        loaded.dim_specs[d]["bin_map"],
                        model.dim_specs[d]["bin_map"],
                    )

                # model_data counts and cdf
                assert len(loaded.model_data["counts"]) == len(
                    model.model_data["counts"]
                )
                assert len(loaded.model_data["cdf"]) == len(
                    model.model_data["cdf"]
                )
            finally:
                os.unlink(path)
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")

    def test_save_load_3d_with_bin_merge(self):
        """Save/load preserves 3D model structure and bin_merge."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        pm.gvtrack_create("a_frac", None, "kmer.frac", kmer="A")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac",
                    "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "bin_merge": [{"from": (0.4, float("inf")), "to": (0.3, 0.4)}],
                },
                {"expr": "c_frac", "breaks": [0, 0.125, 0.25, 0.375, 0.5]},
                {"expr": "a_frac", "breaks": [0, 0.25, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                path = f.name
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)

                assert loaded.n_dims == 3
                assert loaded.dim_sizes == model.dim_sizes
                assert loaded.total_bins == model.total_bins

                for d in range(3):
                    assert (
                        loaded.dim_specs[d]["expr"]
                        == model.dim_specs[d]["expr"]
                    )
                    np.testing.assert_array_almost_equal(
                        loaded.dim_specs[d]["breaks"],
                        model.dim_specs[d]["breaks"],
                    )
                    assert (
                        loaded.dim_specs[d]["num_bins"]
                        == model.dim_specs[d]["num_bins"]
                    )
                    np.testing.assert_array_equal(
                        loaded.dim_specs[d]["bin_map"],
                        model.dim_specs[d]["bin_map"],
                    )
            finally:
                os.unlink(path)
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")
            pm.gvtrack_rm("a_frac")

    def test_save_load_sampling_identical(self):
        """Saved and loaded model should produce identical sampling output."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                path = f.name
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)

                ivs = pm.gintervals("1", 0, 1000)
                s1 = pm.gsynth_sample(model, intervals=ivs, iterator=200, seed=42)
                s2 = pm.gsynth_sample(loaded, intervals=ivs, iterator=200, seed=42)
                assert s1 == s2
            finally:
                os.unlink(path)
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# 0D model (unstratified)
# ============================================================================


class TestGsynth0D:
    """Tests for 0-dimensional (unstratified) models."""

    def test_train_0d(self):
        """Train 0D model without dimension specs."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 100000),
            iterator=1000,
        )
        assert isinstance(model, pm.GsynthModel)
        assert model.n_dims == 0
        assert model.total_bins == 1
        assert len(model.dim_specs) == 0
        # dim_sizes may be [1] (pymisha implementation detail)
        assert model.total_kmers > 0
        assert len(model.per_bin_kmers) == 1
        assert len(model.model_data["cdf"]) == 1

    def test_train_0d_cdf_valid(self):
        """0D model CDF structure is valid."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 100000),
            iterator=1000,
        )
        cdf_mat = model.model_data["cdf"][0]
        assert cdf_mat.shape == (1024, 4)
        assert np.all(cdf_mat >= 0)
        assert np.all(cdf_mat <= 1)
        # Last column should all be 1 (cumulative)
        np.testing.assert_allclose(cdf_mat[:, 3], 1.0, atol=1e-5)
        # Each row should be non-decreasing
        for ctx in range(10):
            assert np.all(np.diff(cdf_mat[ctx, :]) >= -1e-10)

    def test_sample_0d(self):
        """Sample from 0D model returns valid DNA."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 100000),
            iterator=1000,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("1", 0, 10000),
            seed=42,
        )
        assert len(seqs) == 1
        assert len(seqs[0]) == 10000
        assert all(c in "ACGT" for c in seqs[0])

    def test_save_load_0d(self):
        """0D model can be saved and loaded."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=1000,
        )
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            pm.gsynth_save(model, path)
            loaded = pm.gsynth_load(path)
            assert isinstance(loaded, pm.GsynthModel)
            assert loaded.n_dims == 0
            assert loaded.total_bins == 1
            assert loaded.total_kmers == model.total_kmers
            np.testing.assert_array_equal(
                loaded.per_bin_kmers, model.per_bin_kmers
            )
        finally:
            os.unlink(path)

    def test_0d_per_bin_kmers_equals_total(self):
        """Single bin should contain all k-mers."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=1000,
        )
        assert len(model.per_bin_kmers) == 1
        assert int(model.per_bin_kmers[0]) == model.total_kmers

    def test_0d_seed_reproducible(self):
        """0D model sampling reproducible with seed."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=1000,
        )
        ivs = pm.gintervals("1", 0, 5000)
        s1 = pm.gsynth_sample(model, intervals=ivs, seed=12345)
        s2 = pm.gsynth_sample(model, intervals=ivs, seed=12345)
        assert s1 == s2
        s3 = pm.gsynth_sample(model, intervals=ivs, seed=54321)
        assert s1 != s3

    def test_0d_multi_chrom(self):
        """0D model works with multiple chromosomes."""
        model = pm.gsynth_train(
            intervals=pm.gintervals_all(),
            iterator=1000,
        )
        assert model.n_dims == 0
        assert model.total_bins == 1

        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals(["1", "2"], [0, 0], [1000, 1000]),
            seed=60427,
        )
        assert len(seqs) == 2
        assert len(seqs[0]) == 1000
        assert len(seqs[1]) == 1000

    def test_0d_n_samples(self):
        """0D model with n_samples generates multiple sequences."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 100000),
            iterator=1000,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("1", 0, 1000),
            n_samples=5,
            seed=60427,
        )
        assert len(seqs) == 5
        for s in seqs:
            assert len(s) == 1000
            assert all(c in "ACGT" for c in s)
        # At least some should differ
        assert len(set(seqs)) > 1

    def test_0d_repr(self):
        """0D model repr does not error."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=1000,
        )
        s = repr(model)
        assert "Markov" in s
        assert "Dimensions: 0" in s
        assert "Total bins: 1" in s


# ============================================================================
# CDF structure validation
# ============================================================================


class TestGsynthCDFValidation:
    """CDF structure correctness tests."""

    def test_all_cdfs_valid_structure(self):
        """Every CDF bin should be 1024x4, values in [0,1], last col = 1, monotone."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            track_range = pm.gsummary(
                "dense_track",
                intervals=pm.gintervals("1", 0, 50000),
            )
            model = pm.gsynth_train(
                {
                    "expr": "test_vt",
                    "breaks": np.linspace(
                        track_range["Min"], track_range["Max"], 11
                    ).tolist(),
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

            for b in range(model.total_bins):
                cdf = model.model_data["cdf"][b]
                assert cdf.shape == (1024, 4)
                assert np.all(cdf >= 0)
                assert np.all(cdf <= 1)
                np.testing.assert_allclose(cdf[:, 3], 1.0, atol=1e-5)
                for ctx in range(1024):
                    assert np.all(np.diff(cdf[ctx, :]) >= -1e-10)
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# Train with mask
# ============================================================================


class TestGsynthTrainMask:
    """Tests for training with mask."""

    def test_mask_reduces_kmers(self):
        """Mask should reduce total k-mers and increase total_masked."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            ivs = pm.gintervals("1", 0, 100000)
            track_range = pm.gsummary("dense_track", intervals=ivs)
            breaks = np.linspace(
                track_range["Min"], track_range["Max"], 11
            ).tolist()

            model_no_mask = pm.gsynth_train(
                {"expr": "test_vt", "breaks": breaks},
                intervals=ivs,
                iterator=200,
            )
            mask = pm.gintervals("1", 0, 50000)
            model_with_mask = pm.gsynth_train(
                {"expr": "test_vt", "breaks": breaks},
                mask=mask,
                intervals=ivs,
                iterator=200,
            )

            assert model_with_mask.total_kmers < model_no_mask.total_kmers
            assert model_with_mask.total_masked > 0
        finally:
            pm.gvtrack_rm("test_vt")

    def test_0d_mask(self):
        """0D model with mask works correctly."""
        ivs = pm.gintervals("1", 0, 100000)
        mask = pm.gintervals("1", 0, 50000)

        model_no_mask = pm.gsynth_train(intervals=ivs, iterator=1000)
        model_with_mask = pm.gsynth_train(mask=mask, intervals=ivs, iterator=1000)

        assert model_with_mask.total_kmers < model_no_mask.total_kmers
        assert model_with_mask.total_masked > 0
        assert model_no_mask.total_masked == 0


# ============================================================================
# Train with different pseudocounts
# ============================================================================


class TestGsynthPseudocount:
    """Tests for pseudocount effect."""

    def test_different_pseudocounts(self):
        """Different pseudocounts yield same total_kmers but different CDFs."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            ivs = pm.gintervals("1", 0, 50000)
            track_range = pm.gsummary("dense_track", intervals=ivs)
            breaks = np.linspace(
                track_range["Min"], track_range["Max"], 11
            ).tolist()

            m1 = pm.gsynth_train(
                {"expr": "test_vt", "breaks": breaks},
                intervals=ivs,
                iterator=200,
                pseudocount=1,
            )
            m10 = pm.gsynth_train(
                {"expr": "test_vt", "breaks": breaks},
                intervals=ivs,
                iterator=200,
                pseudocount=10,
            )

            assert m1.total_kmers == m10.total_kmers
            # CDFs should differ due to pseudocount
            assert not np.array_equal(
                m1.model_data["cdf"][0], m10.model_data["cdf"][0]
            )
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# Empty bins
# ============================================================================


class TestGsynthEmptyBins:
    """Tests for empty bin handling."""

    def test_empty_bins_graceful(self):
        """Breaks beyond data range create empty bins; model remains usable."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            ivs = pm.gintervals("1", 0, 50000)
            track_range = pm.gsummary("dense_track", intervals=ivs)
            rng = track_range["Max"] - track_range["Min"]
            breaks = np.linspace(
                track_range["Min"] - rng,
                track_range["Max"] + rng,
                21,
            ).tolist()

            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": breaks},
                intervals=ivs,
                iterator=200,
            )
            assert isinstance(model, pm.GsynthModel)
            # Some bins should have 0 k-mers
            assert any(k == 0 for k in model.per_bin_kmers)

            # Should still be usable for sampling
            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 1000),
                iterator=200,
                seed=60427,
            )
            assert len(seqs[0]) == 1000
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# Sample advanced: mask_copy, multiple chroms, n_samples, FASTA multi-sample
# ============================================================================


class TestGsynthSampleAdvanced:
    """Advanced sampling tests."""

    def test_sample_mask_copy(self):
        """mask_copy preserves original sequence in masked regions."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            ivs = pm.gintervals("1", 0, 50000)
            track_range = pm.gsummary("dense_track", intervals=ivs)
            model = pm.gsynth_train(
                {
                    "expr": "test_vt",
                    "breaks": np.linspace(
                        track_range["Min"], track_range["Max"], 11
                    ).tolist(),
                },
                intervals=ivs,
                iterator=200,
            )

            mask_copy = pm.gintervals("1", 1000, 2000)
            sample_ivs = pm.gintervals("1", 0, 3000)

            with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
                fasta_path = f.name
            try:
                pm.gsynth_sample(
                    model,
                    output=fasta_path,
                    output_format="fasta",
                    intervals=sample_ivs,
                    iterator=200,
                    mask_copy=mask_copy,
                    seed=60427,
                )
                with open(fasta_path) as f:
                    lines = f.readlines()
                sampled_seq = "".join(
                    line.strip() for line in lines if not line.startswith(">")
                )

                # Get original sequence
                orig = pm.gseq_extract(mask_copy)[0]

                # Masked region (positions 1000..2000) should match original
                sampled_region = sampled_seq[1000:2000]
                assert sampled_region.upper() == orig.upper()
            finally:
                os.unlink(fasta_path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_multi_chrom(self):
        """Sampling from multiple chromosomes produces correct number of headers."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            ivs = pm.gintervals_all()
            track_range = pm.gsummary("dense_track", intervals=ivs)
            model = pm.gsynth_train(
                {
                    "expr": "test_vt",
                    "breaks": np.linspace(
                        track_range["Min"], track_range["Max"], 11
                    ).tolist(),
                },
                intervals=ivs,
                iterator=200,
            )

            sample_ivs = pm.gintervals(
                ["1", "2"], [0, 0], [1000, 1000]
            )
            with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
                fasta_path = f.name
            try:
                pm.gsynth_sample(
                    model,
                    output=fasta_path,
                    output_format="fasta",
                    intervals=sample_ivs,
                    iterator=200,
                    seed=60427,
                )
                with open(fasta_path) as f:
                    content = f.read()
                headers = [line for line in content.split("\n") if line.startswith(">")]
                assert len(headers) == 2
            finally:
                os.unlink(fasta_path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_n_samples_vector(self):
        """n_samples > 1 returns multiple sequences in vector mode."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 500),
                iterator=200,
                n_samples=5,
                seed=60427,
            )
            assert len(seqs) == 5
            for s in seqs:
                assert len(s) == 500
                assert all(c in "ACGT" for c in s)
            assert len(set(seqs)) > 1
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_n_samples_fasta(self):
        """n_samples > 1 writes multiple FASTA entries."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

            with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
                fasta_path = f.name
            try:
                pm.gsynth_sample(
                    model,
                    output=fasta_path,
                    output_format="fasta",
                    intervals=pm.gintervals("1", 0, 500),
                    iterator=200,
                    n_samples=3,
                    seed=60427,
                )
                with open(fasta_path) as f:
                    content = f.read()
                headers = [line for line in content.split("\n") if line.startswith(">")]
                assert len(headers) == 3
            finally:
                os.unlink(fasta_path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_n_samples_multi_intervals(self):
        """n_samples with multiple intervals produces n_intervals * n_samples sequences."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            sample_ivs = pm.gintervals("1", [0, 1000], [500, 1500])
            seqs = pm.gsynth_sample(
                model,
                intervals=sample_ivs,
                iterator=200,
                n_samples=3,
                seed=60427,
            )
            # 2 intervals * 3 samples = 6
            assert len(seqs) == 6
            for s in seqs:
                assert len(s) == 500
                assert all(c in "ACGT" for c in s)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_n_samples_seed_reproducible(self):
        """n_samples with same seed is reproducible."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            ivs = pm.gintervals("1", 0, 500)
            s1 = pm.gsynth_sample(model, intervals=ivs, iterator=200, n_samples=3, seed=12345)
            s2 = pm.gsynth_sample(model, intervals=ivs, iterator=200, n_samples=3, seed=12345)
            assert s1 == s2

            s3 = pm.gsynth_sample(model, intervals=ivs, iterator=200, n_samples=3, seed=54321)
            assert s1 != s3
        finally:
            pm.gvtrack_rm("test_vt")

    def test_sample_2d_model(self):
        """Sample from 2D model produces correct length sequence."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {"expr": "g_frac + c_frac", "breaks": np.linspace(0, 1, 11).tolist()},
                {"expr": "g_frac", "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
                seed=60427,
            )
            assert len(seqs[0]) == 10000
            assert all(c in "ACGT" for c in seqs[0])
        finally:
            pm.gvtrack_rm("g_frac")
            pm.gvtrack_rm("c_frac")


# ============================================================================
# Error handling
# ============================================================================


class TestGsynthErrorHandling:
    """Error handling tests for gsynth functions."""

    def test_train_0d_works(self):
        """0D model (no dim specs) should not error."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
        )
        assert model.n_dims == 0

    def test_train_non_dict_raises(self):
        """Passing a string instead of dict raises TypeError."""
        with pytest.raises(TypeError, match="dict"):
            pm.gsynth_train(
                "test_vt",
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

    def test_train_empty_dict_raises(self):
        """Empty dict (no expr) raises ValueError."""
        with pytest.raises(ValueError, match="expr"):
            pm.gsynth_train(
                {},
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

    def test_train_breaks_single_element(self):
        """Breaks with < 2 elements raises ValueError."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            with pytest.raises(ValueError, match="at least 2"):
                pm.gsynth_train(
                    {"expr": "test_vt", "breaks": [0.5]},
                    intervals=pm.gintervals("1", 0, 50000),
                    iterator=200,
                )
        finally:
            pm.gvtrack_rm("test_vt")


# ============================================================================
# gsynth_replace_kmer advanced
# ============================================================================


class TestGsynthReplaceKmerAdvanced:
    """Advanced gsynth_replace_kmer tests."""

    def test_replace_kmer_iterative(self):
        """CG->GC iterative replacement removes all CG from longer region."""
        seqs = pm.gsynth_replace_kmer(
            "CG", "GC",
            intervals=pm.gintervals("1", 0, 5000),
        )
        assert "CG" not in seqs[0]

    def test_replace_kmer_fasta_output(self):
        """Replace kmer writes valid FASTA."""
        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            path = f.name
        try:
            pm.gsynth_replace_kmer(
                "CG", "GC",
                intervals=pm.gintervals("1", 0, 500),
                output=path,
                output_format="fasta",
            )
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert lines[0].startswith(">")
            seq = "".join(line.strip() for line in lines if not line.startswith(">"))
            assert "CG" not in seq
        finally:
            os.unlink(path)

    def test_replace_kmer_multiple_intervals(self):
        """Replace kmer with multiple intervals returns one seq per interval."""
        ivs = pm.gintervals(
            ["1", "1", "2"], [0, 5000, 0], [1000, 6000, 1000]
        )
        result = pm.gsynth_replace_kmer("CG", "GC", intervals=ivs)
        assert len(result) == 3
        for seq in result:
            assert "CG" not in seq

    def test_replace_kmer_3mer(self):
        """Replace a 3-mer (ACG -> CAG) removes all ACG."""
        result = pm.gsynth_replace_kmer(
            "ACG", "CAG",
            intervals=pm.gintervals("1", 0, 1000),
            check_composition=True,
        )
        assert "ACG" not in result[0]

    def test_replace_kmer_4mer(self):
        """Replace a 4-mer (CGCG -> GCGC) removes all CGCG."""
        result = pm.gsynth_replace_kmer(
            "CGCG", "GCGC",
            intervals=pm.gintervals("1", 0, 1000),
            check_composition=True,
        )
        assert "CGCG" not in result[0]

    def test_replace_kmer_preserves_length(self):
        """Replacement preserves sequence length."""
        length = 2000
        result = pm.gsynth_replace_kmer(
            "CG", "GC",
            intervals=pm.gintervals("1", 0, length),
        )
        assert len(result[0]) == length

    def test_replace_kmer_preserves_composition(self):
        """CG->GC preserves base composition (same C+G counts)."""
        ivs = pm.gintervals("1", 0, 5000)
        orig = pm.gseq_extract(ivs)[0].upper()
        result = pm.gsynth_replace_kmer(
            "CG", "GC",
            intervals=ivs,
            check_composition=True,
        )
        replaced = result[0].upper()

        orig_c = orig.count("C")
        orig_g = orig.count("G")
        res_c = replaced.count("C")
        res_g = replaced.count("G")

        assert orig_c == res_c
        assert orig_g == res_g

    def test_replace_kmer_case_insensitive(self):
        """Lowercase target/replacement also works."""
        r_upper = pm.gsynth_replace_kmer(
            "CG", "GC",
            intervals=pm.gintervals("1", 0, 1000),
        )
        r_lower = pm.gsynth_replace_kmer(
            "cg", "gc",
            intervals=pm.gintervals("1", 0, 1000),
        )
        # Both should remove all CG
        assert "CG" not in r_upper[0].upper()
        assert "CG" not in r_lower[0].upper()

    def test_replace_kmer_identical_is_noop(self):
        """Identical target and replacement returns original sequence unchanged."""
        ivs = pm.gintervals("1", 0, 100)
        result = pm.gsynth_replace_kmer("CG", "CG", intervals=ivs)
        orig = pm.gseq_extract(ivs)[0]
        assert result[0].upper() == orig.upper()


# ============================================================================
# gsynth_random advanced
# ============================================================================


class TestGsynthRandomAdvanced:
    """Advanced gsynth_random tests."""

    def test_random_gc_rich(self):
        """GC-rich probabilities produce ~80% GC content."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 10000),
            nuc_probs={"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
            seed=42,
        )
        chars = list(seqs[0])
        gc = sum(1 for c in chars if c in "GC")
        gc_frac = gc / len(chars)
        assert 0.7 < gc_frac < 0.9

    def test_random_normalizes_probs(self):
        """Non-normalized probs (summing to 4) still work."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 1000),
            nuc_probs={"A": 1, "C": 1, "G": 1, "T": 1},
            seed=42,
        )
        assert len(seqs[0]) == 1000
        assert all(c in "ACGT" for c in seqs[0])

    def test_random_n_samples(self):
        """n_samples > 1 returns multiple random sequences."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 500),
            n_samples=5,
            seed=42,
        )
        assert len(seqs) == 5
        for s in seqs:
            assert len(s) == 500
            assert all(c in "ACGT" for c in s)
        assert len(set(seqs)) > 1

    def test_random_fasta_output(self):
        """Random FASTA output is valid."""
        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            path = f.name
        try:
            pm.gsynth_random(
                intervals=pm.gintervals("1", 0, 1000),
                output=path,
                output_format="fasta",
                seed=42,
            )
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert lines[0].startswith(">")
            seq = "".join(line.strip() for line in lines if not line.startswith(">"))
            assert len(seq) == 1000
            assert all(c in "ACGT" for c in seq)
        finally:
            os.unlink(path)

    def test_random_multi_intervals(self):
        """Random generation for multiple intervals returns correct number."""
        ivs = pm.gintervals(["1", "2"], [0, 0], [500, 500])
        seqs = pm.gsynth_random(intervals=ivs, seed=42)
        assert len(seqs) == 2
        assert len(seqs[0]) == 500
        assert len(seqs[1]) == 500

    def test_random_uniform_distribution(self):
        """Default probs produce roughly uniform base distribution."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 40000),
            seed=42,
        )
        chars = list(seqs[0])
        total = len(chars)
        for base in "ACGT":
            frac = sum(1 for c in chars if c == base) / total
            assert 0.22 < frac < 0.28, f"Base {base} fraction {frac} out of range"

    def test_random_partial_probs_defaults_missing_to_zero(self):
        """nuc_probs with only 2 keys defaults missing to 0."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 1000),
            nuc_probs={"A": 0.5, "C": 0.5},
            seed=42,
        )
        # Should still produce valid output
        assert len(seqs[0]) == 1000
        assert all(c in "ACGT" for c in seqs[0])

    def test_random_at_only_probs(self):
        """Probabilities with only A and T produce AT-rich sequence."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 10000),
            nuc_probs={"A": 0.5, "C": 0.0, "G": 0.0, "T": 0.5},
            seed=42,
        )
        chars = list(seqs[0][5:])  # skip initial 5 seed bases
        gc = sum(1 for c in chars if c in "GC")
        assert gc == 0, f"Found {gc} G/C bases with zero probability"

    def test_random_mask_copy(self):
        """mask_copy preserves original sequence in random generation."""
        mask_copy = pm.gintervals("1", 500, 700)
        sample_ivs = pm.gintervals("1", 0, 1000)

        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            fasta_path = f.name
        try:
            pm.gsynth_random(
                intervals=sample_ivs,
                output=fasta_path,
                output_format="fasta",
                mask_copy=mask_copy,
                seed=42,
            )
            with open(fasta_path) as f:
                lines = f.readlines()
            sampled = "".join(line.strip() for line in lines if not line.startswith(">"))

            orig = pm.gseq_extract(mask_copy)[0]
            assert sampled[500:700].upper() == orig.upper()
        finally:
            os.unlink(fasta_path)
