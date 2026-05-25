"""Tests for gsynth functions."""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

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
    """Tests for gsynth_save and gsynth_load functions (.gsm format)."""

    def _train_1d_model(self):
        """Helper: train a 1D model for save/load tests."""
        return pm.gsynth_train(
            {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
            intervals=pm.gintervals("1", 0, 10000),
            iterator=200,
        )

    def test_save_load_roundtrip_directory(self):
        """Saved and loaded model should match (directory mode)."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = self._train_1d_model()
            path = os.path.join(tempfile.mkdtemp(), "model.gsm")
            try:
                pm.gsynth_save(model, path)
                assert os.path.isdir(path)
                assert os.path.exists(os.path.join(path, "metadata.yaml"))
                assert os.path.exists(os.path.join(path, "counts.bin"))
                assert os.path.exists(os.path.join(path, "cdf.bin"))

                loaded = pm.gsynth_load(path)
                assert isinstance(loaded, pm.GsynthModel)
                assert loaded.n_dims == model.n_dims
                assert loaded.total_bins == model.total_bins
                assert loaded.total_kmers == model.total_kmers
                assert loaded.pseudocount == model.pseudocount
                assert loaded.min_obs == model.min_obs
                assert loaded.total_masked == model.total_masked
                assert loaded.total_n == model.total_n
                assert loaded.dim_sizes == model.dim_sizes
                # Compare counts
                for orig, load in zip(model.model_data["counts"],
                                      loaded.model_data["counts"], strict=False):
                    np.testing.assert_array_equal(orig, load)
                # Compare CDFs
                for orig, load in zip(model.model_data["cdf"],
                                      loaded.model_data["cdf"], strict=False):
                    np.testing.assert_array_almost_equal(orig, load)
                # Compare per_bin_kmers
                np.testing.assert_array_equal(model.per_bin_kmers, loaded.per_bin_kmers)
            finally:
                shutil.rmtree(path, ignore_errors=True)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_save_load_roundtrip_zip(self):
        """Saved and loaded model should match (ZIP/compress mode)."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = self._train_1d_model()
            with tempfile.NamedTemporaryFile(suffix=".gsm.zip", delete=False) as f:
                path = f.name
            try:
                pm.gsynth_save(model, path, compress=True)
                assert os.path.isfile(path)
                assert os.path.getsize(path) > 0

                loaded = pm.gsynth_load(path)
                assert isinstance(loaded, pm.GsynthModel)
                assert loaded.n_dims == model.n_dims
                assert loaded.total_bins == model.total_bins
                assert loaded.total_kmers == model.total_kmers
                assert loaded.pseudocount == model.pseudocount
                for orig, load in zip(model.model_data["cdf"],
                                      loaded.model_data["cdf"], strict=False):
                    np.testing.assert_array_almost_equal(orig, load)
                for orig, load in zip(model.model_data["counts"],
                                      loaded.model_data["counts"], strict=False):
                    np.testing.assert_array_equal(orig, load)
            finally:
                os.unlink(path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_legacy_pickle_backward_compat(self):
        """Loading a legacy pickle model should still work."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = self._train_1d_model()
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                path = f.name
            try:
                # Save with old pickle format directly
                with open(path, "wb") as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

                loaded = pm.gsynth_load(path)
                assert isinstance(loaded, pm.GsynthModel)
                assert loaded.n_dims == model.n_dims
                assert loaded.total_bins == model.total_bins
                assert loaded.total_kmers == model.total_kmers
            finally:
                os.unlink(path)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_save_non_model_raises(self):
        """Saving non-GsynthModel raises TypeError."""
        with pytest.raises(TypeError, match="GsynthModel"):
            pm.gsynth_save("not_a_model", "/tmp/test.gsm")

    def test_load_non_model_raises(self):
        """Loading non-GsynthModel pickle file raises TypeError."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a model"}, f)
            path = f.name
        try:
            with pytest.raises(TypeError, match="GsynthModel"):
                pm.gsynth_load(path)
        finally:
            os.unlink(path)

    def test_gsynth_convert(self):
        """gsynth_convert converts legacy pickle to .gsm format."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = self._train_1d_model()
            tmpdir = tempfile.mkdtemp()
            pkl_path = os.path.join(tmpdir, "model.pkl")
            gsm_path = os.path.join(tmpdir, "model.gsm")
            try:
                # Save as legacy pickle
                with open(pkl_path, "wb") as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

                pm.gsynth_convert(pkl_path, gsm_path)
                assert os.path.isdir(gsm_path)

                loaded = pm.gsynth_load(gsm_path)
                assert loaded.total_bins == model.total_bins
                assert loaded.total_kmers == model.total_kmers
                for orig, load in zip(model.model_data["cdf"],
                                      loaded.model_data["cdf"], strict=False):
                    np.testing.assert_array_almost_equal(orig, load)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_min_obs_preserved_roundtrip(self):
        """min_obs is preserved through save/load round-trip."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.5, 1.0]},
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
                min_obs=42,
            )
            assert model.min_obs == 42

            path = os.path.join(tempfile.mkdtemp(), "model.gsm")
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)
                assert loaded.min_obs == 42
            finally:
                shutil.rmtree(path, ignore_errors=True)
        finally:
            pm.gvtrack_rm("test_vt")

    def test_0d_model_roundtrip(self):
        """0-dim (unstratified) model saves and loads correctly."""
        model = pm.gsynth_train()
        assert model.n_dims == 0
        assert model.total_bins == 1
        assert model.dim_sizes == [1]

        path = os.path.join(tempfile.mkdtemp(), "model_0d.gsm")
        try:
            pm.gsynth_save(model, path)
            loaded = pm.gsynth_load(path)
            assert loaded.n_dims == 0
            assert loaded.total_bins == 1
            assert loaded.dim_sizes == [1]
            assert loaded.total_kmers == model.total_kmers
            np.testing.assert_array_equal(
                model.model_data["counts"][0], loaded.model_data["counts"][0]
            )
            np.testing.assert_array_almost_equal(
                model.model_data["cdf"][0], loaded.model_data["cdf"][0]
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_dim_specs_preserved(self):
        """dim_specs including bin_map are preserved through round-trip."""
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "test_vt",
                    "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "bin_merge": [{"from": (0.8, float("inf")), "to": (0.6, 0.8)}],
                },
                intervals=pm.gintervals("1", 0, 10000),
                iterator=200,
            )
            assert model.dim_specs[0]["bin_map"] is not None

            path = os.path.join(tempfile.mkdtemp(), "model.gsm")
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)
                assert loaded.dim_specs[0]["expr"] == model.dim_specs[0]["expr"]
                assert loaded.dim_specs[0]["num_bins"] == model.dim_specs[0]["num_bins"]
                assert loaded.dim_specs[0]["breaks"] == model.dim_specs[0]["breaks"]
                assert loaded.dim_specs[0]["bin_map"] == [int(x) for x in model.dim_specs[0]["bin_map"]]
            finally:
                shutil.rmtree(path, ignore_errors=True)
        finally:
            pm.gvtrack_rm("test_vt")


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
                model, intervals=ivs, iterator=200, seed=60427
            )
            r2 = pm.gsynth_sample(
                model, intervals=ivs, iterator=200, seed=60427
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
                model, intervals=ivs, iterator=200, seed=60427
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
        result = pm.gsynth_random(intervals=ivs, seed=60427)

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
            seed=60427,
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
        r1 = pm.gsynth_random(intervals=ivs, seed=60427)
        r2 = pm.gsynth_random(intervals=ivs, seed=60427)
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


class TestGsynthOutputFormatAliases:
    """R parity: "misha" is the default output_format; "seq" is a legacy alias.

    R signature is ``output_format = c("misha", "fasta", "vector")`` (first
    element is the default). Before this fix PyMisha defaulted to "fasta",
    which silently produced different output than the R API for identical
    invocations.
    """

    def test_misha_alias_writes_binary(self, tmp_path):
        ivs = pm.gintervals(["1"], [0], [200])
        model = pm.gsynth_train(intervals=ivs, k=2, iterator=200)
        out = tmp_path / "x.seq"
        pm.gsynth_sample(model, str(out), output_format="misha",
                         intervals=ivs, iterator=200, seed=60427)
        assert out.exists()
        # Binary format starts with raw bytes, not '>'
        with open(out, "rb") as fh:
            assert fh.read(1) != b">"

    def test_seq_alias_still_works(self, tmp_path):
        ivs = pm.gintervals(["1"], [0], [200])
        model = pm.gsynth_train(intervals=ivs, k=2, iterator=200)
        out = tmp_path / "x.seq"
        pm.gsynth_sample(model, str(out), output_format="seq",
                         intervals=ivs, iterator=200, seed=60427)
        assert out.exists()

    def test_default_is_misha_not_fasta(self, tmp_path):
        ivs = pm.gintervals(["1"], [0], [200])
        model = pm.gsynth_train(intervals=ivs, k=2, iterator=200)
        out = tmp_path / "x.bin"
        pm.gsynth_sample(model, str(out), intervals=ivs, iterator=200,
                         seed=60427)
        with open(out, "rb") as fh:
            assert fh.read(1) != b">", "default must be misha binary, not fasta"

    def test_invalid_format_raises(self):
        ivs = pm.gintervals(["1"], [0], [200])
        model = pm.gsynth_train(intervals=ivs, k=2, iterator=200)
        with pytest.raises(ValueError, match="Invalid output_format"):
            pm.gsynth_sample(model, output_format="bogus",
                             intervals=ivs, iterator=200)

    def test_random_accepts_iterator_param(self):
        ivs = pm.gintervals(["1"], [0], [500])
        # R: gsynth.random(..., iterator = 1). Must not raise.
        out = pm.gsynth_random(intervals=ivs, output_format="vector",
                               iterator=1, seed=60427)
        assert len(out) == 1
        assert len(out[0]) == 500

    def test_replace_kmer_default_is_misha(self, tmp_path):
        ivs = pm.gintervals(["1"], [0], [200])
        out = tmp_path / "x.bin"
        pm.gsynth_replace_kmer("CG", "GC", intervals=ivs, output=str(out))
        with open(out, "rb") as fh:
            assert fh.read(1) != b">"


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
            s1 = pm.gsynth_sample(model, intervals=ivs, iterator=200, seed=60427)
            s2 = pm.gsynth_sample(model, intervals=ivs, iterator=200, seed=60427)
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

    def test_sample_bin_merge_at_sampling_time(self):
        """bin_merge passed at sampling time overrides training-time bin_merge."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac",
                    "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )
            assert model.n_dims == 1
            # No training-time bin_merge was applied
            assert model.dim_specs[0].get("bin_merge") is None

            # Sample without bin_merge
            seqs_no_merge = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 2000),
                iterator=200,
                seed=60427,
            )
            assert len(seqs_no_merge[0]) == 2000

            # Sample with sampling-time bin_merge: fold last bin into second-to-last
            seqs_with_merge = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 2000),
                iterator=200,
                seed=60427,
                bin_merge=[[{"from": (0.4, 0.5), "to": (0.3, 0.4)}]],
            )
            assert len(seqs_with_merge[0]) == 2000
            assert all(c in "ACGT" for c in seqs_with_merge[0])

            # The sequences should differ because bin mapping changed
            # (not guaranteed but highly likely with different bin routing)
        finally:
            pm.gvtrack_rm("g_frac")

    def test_sample_bin_merge_validation(self):
        """bin_merge must match model dimensions."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000),
            iterator=200,
        )
        assert model.n_dims == 0

        # 0D model: bin_merge must be empty list
        with pytest.raises(ValueError, match="0 elements"):
            pm.gsynth_sample(model, bin_merge=[None],
                             intervals=pm.gintervals("1", 0, 1000))

    def test_sample_bin_merge_none_elements_use_training_default(self):
        """bin_merge=[None] uses training-time bin_merge for that dimension."""
        pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac",
                    "breaks": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "bin_merge": [{"from": (0.4, 0.5), "to": (0.3, 0.4)}],
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
            )

            # bin_merge=[None] should behave identically to no bin_merge at all
            seqs_default = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 2000),
                iterator=200,
                seed=60427,
            )
            seqs_none = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 2000),
                iterator=200,
                seed=60427,
                bin_merge=[None],
            )
            assert seqs_default == seqs_none
        finally:
            pm.gvtrack_rm("g_frac")


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

            path = os.path.join(tempfile.mkdtemp(), "model.gsm")
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
                shutil.rmtree(path, ignore_errors=True)
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

            path = os.path.join(tempfile.mkdtemp(), "model.gsm")
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
                shutil.rmtree(path, ignore_errors=True)
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

            path = os.path.join(tempfile.mkdtemp(), "model.gsm")
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)

                ivs = pm.gintervals("1", 0, 1000)
                s1 = pm.gsynth_sample(model, intervals=ivs, iterator=200, seed=60427)
                s2 = pm.gsynth_sample(loaded, intervals=ivs, iterator=200, seed=60427)
                assert s1 == s2
            finally:
                shutil.rmtree(path, ignore_errors=True)
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
            seed=60427,
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
        path = os.path.join(tempfile.mkdtemp(), "model_0d.gsm")
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
            shutil.rmtree(path, ignore_errors=True)

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

    def test_0d_non_first_chrom_only(self):
        """Regression: 0D training on intervals that exclude chromkey ID 0
        must still count k-mers. Previously ``_extract_bin_data`` hardcoded
        ``iter_chroms`` to zero in the 0D branch, so the C++ backend routed
        every entry to ``chrom_bins[0]`` and left other chroms' bins empty;
        intervals on any non-first chromosome were silently dropped."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("X", 0, 100000),
            iterator=1000,
        )
        assert model.total_kmers > 0
        # Single bin in 0D must hold all counted k-mers
        assert int(model.per_bin_kmers[0]) == model.total_kmers

    def test_0d_multi_chrom_counts_all_chroms(self):
        """Regression: 0D training on intervals spanning multiple chromosomes
        must count k-mers from every chromosome in the input, not just the
        one at chromkey ID 0. Training separately on each chrom and summing
        should equal training on the union."""
        chr1_intervs = pm.gintervals("1", 0, 100000)
        chrX_intervs = pm.gintervals("X", 0, 100000)
        multi = pd.concat([chr1_intervs, chrX_intervs], ignore_index=True)

        m_chr1 = pm.gsynth_train(intervals=chr1_intervs, iterator=1000)
        m_chrX = pm.gsynth_train(intervals=chrX_intervs, iterator=1000)
        m_multi = pm.gsynth_train(intervals=multi, iterator=1000)

        # Under the bug: m_chrX.total_kmers == 0 and
        # m_multi.total_kmers == m_chr1.total_kmers (chrX silently dropped)
        assert m_chrX.total_kmers > 0
        assert m_multi.total_kmers == m_chr1.total_kmers + m_chrX.total_kmers

        # Counts matrices for the single bin should sum across chroms
        sum_counts = (
            m_chr1.model_data["counts"][0].astype(np.int64)
            + m_chrX.model_data["counts"][0].astype(np.int64)
        )
        np.testing.assert_array_equal(
            m_multi.model_data["counts"][0].astype(np.int64),
            sum_counts,
        )

    def test_0d_sample_on_non_first_chrom_uses_trained_cdf(self):
        """Regression: sampling a 0D model on intervals from a chrom other
        than chromkey ID 0 must use the trained Markov CDF. Under the
        original bug, ``iter_chroms=0`` meant the sampler found no bin for
        non-first-chrom positions and fell back to ``drand48()*4`` uniform
        random instead of the trained CDF.

        Train a model whose CDF is strongly non-uniform by picking a seed
        genome where base frequencies in the training region deviate from
        25%/25%/25%/25%. We then sample many bases on chrX and check that
        the empirical base distribution matches the trained CDF (not
        uniform). With the test DB's chr1 the A/T/C/G composition is
        clearly unbalanced enough to detect.
        """
        # Train on chr1 (known to be AT-rich in the test DB).
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 100000),
            iterator=1000,
        )

        # Marginal base frequencies over the trained CDF (uniform context
        # weighting): average conditional next-base probability across
        # every k-mer context in the single bin.
        cdf_mat = model.model_data["cdf"][0]
        probs = np.empty_like(cdf_mat)
        probs[:, 0] = cdf_mat[:, 0]
        probs[:, 1:] = np.diff(cdf_mat, axis=1)
        train_marginal = probs.mean(axis=0)  # [A, C, G, T]

        # Only run the non-uniformity assertion if the trained model
        # really is non-uniform (skip on degenerate test fixtures).
        max_dev = float(np.max(np.abs(train_marginal - 0.25)))
        if max_dev < 0.02:
            pytest.skip(
                "Trained model marginal is too close to uniform "
                "({max_dev=:.3f}) to distinguish trained CDF from uniform "
                "fallback on this fixture."
            )

        # Sample a long stretch on chrX (a chromosome NOT in training and
        # not at chromkey ID 0).
        seq_len = 50000
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("X", 0, seq_len),
            seed=60427,
        )
        assert len(seqs) == 1 and len(seqs[0]) == seq_len

        base_counts = np.array(
            [seqs[0].count(b) for b in ("A", "C", "G", "T")],
            dtype=np.float64,
        )
        empirical = base_counts / base_counts.sum()

        # Under the bug, sampling on chrX falls back to uniform → empirical
        # should be close to 0.25 for every base. With the fix, empirical
        # should track the trained marginal.
        dev_from_train = float(np.max(np.abs(empirical - train_marginal)))
        dev_from_uniform = float(np.max(np.abs(empirical - 0.25)))
        assert dev_from_train < dev_from_uniform, (
            f"chrX sampled base frequencies {empirical.tolist()} are closer "
            f"to uniform (0.25) than to the trained marginal "
            f"{train_marginal.tolist()}; 0D chromid resolution is likely "
            "regressing to the pre-fix behaviour."
        )


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
            seed=60427,
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
            seed=60427,
        )
        assert len(seqs[0]) == 1000
        assert all(c in "ACGT" for c in seqs[0])

    def test_random_n_samples(self):
        """n_samples > 1 returns multiple random sequences."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 500),
            n_samples=5,
            seed=60427,
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
                seed=60427,
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
        seqs = pm.gsynth_random(intervals=ivs, seed=60427)
        assert len(seqs) == 2
        assert len(seqs[0]) == 500
        assert len(seqs[1]) == 500

    def test_random_uniform_distribution(self):
        """Default probs produce roughly uniform base distribution."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 40000),
            seed=60427,
        )
        chars = list(seqs[0])
        total = len(chars)
        for base in "ACGT":
            frac = sum(1 for c in chars if c == base) / total
            assert 0.22 < frac < 0.28, f"Base {base} fraction {frac} out of range"

    def test_random_rejects_partial_nuc_probs(self):
        """nuc_probs missing a nucleotide is rejected (no silent defaulting).

        Mirrors R misha 5.7.4: previously missing keys silently defaulted,
        producing a wrong distribution.
        """
        with pytest.raises(ValueError, match="A.*C.*G.*T"):
            pm.gsynth_random(
                intervals=pm.gintervals("1", 0, 1000),
                nuc_probs={"A": 0.5, "C": 0.5},
                seed=60427,
            )

    def test_random_rejects_extra_nuc_probs(self):
        """nuc_probs with an unexpected key is rejected."""
        with pytest.raises(ValueError, match="A.*C.*G.*T"):
            pm.gsynth_random(
                intervals=pm.gintervals("1", 0, 1000),
                nuc_probs={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25, "N": 0.0},
                seed=60427,
            )

    def test_random_rejects_duplicate_case_nuc_probs(self):
        """nuc_probs with case-duplicated keys (A and a) is rejected."""
        with pytest.raises(ValueError, match="A.*C.*G.*T"):
            pm.gsynth_random(
                intervals=pm.gintervals("1", 0, 1000),
                nuc_probs={"A": 0.25, "a": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
                seed=60427,
            )

    def test_random_accepts_lowercase_nuc_probs(self):
        """nuc_probs keys are case-insensitive (matches R toupper handling).

        Lowercase AT-only probs must actually be honoured (no G/C output);
        previously lowercase keys were ignored and silently defaulted.
        """
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 10000),
            nuc_probs={"a": 0.5, "c": 0.0, "g": 0.0, "t": 0.5},
            seed=60427,
        )
        chars = list(seqs[0][5:])  # skip initial 5 seed bases
        gc = sum(1 for c in chars if c in "GC")
        assert gc == 0, f"Found {gc} G/C bases with zero probability"

    def test_random_at_only_probs(self):
        """Probabilities with only A and T produce AT-rich sequence."""
        seqs = pm.gsynth_random(
            intervals=pm.gintervals("1", 0, 10000),
            nuc_probs={"A": 0.5, "C": 0.0, "G": 0.0, "T": 0.5},
            seed=60427,
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
                seed=60427,
            )
            with open(fasta_path) as f:
                lines = f.readlines()
            sampled = "".join(line.strip() for line in lines if not line.startswith(">"))

            orig = pm.gseq_extract(mask_copy)[0]
            assert sampled[500:700].upper() == orig.upper()
        finally:
            os.unlink(fasta_path)


# ============================================================================
# Variable Markov order k
# ============================================================================


class TestGsynthVariableK:
    """Tests for variable Markov order k parameter in gsynth functions."""

    # ------------------------------------------------------------------
    # k validation
    # ------------------------------------------------------------------

    def test_k_zero_raises(self):
        """k=0 is below the minimum (1) and should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[1, 10\]"):
            pm.gsynth_train(
                intervals=pm.gintervals("1", 0, 1000), iterator=200, k=0
            )

    def test_k_eleven_raises(self):
        """k=11 is above the maximum (10) and should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[1, 10\]"):
            pm.gsynth_train(
                intervals=pm.gintervals("1", 0, 1000), iterator=200, k=11
            )

    def test_k_negative_raises(self):
        """k=-1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"\[1, 10\]"):
            pm.gsynth_train(
                intervals=pm.gintervals("1", 0, 1000), iterator=200, k=-1
            )

    def test_k_float_rejected(self):
        """Non-integer k (e.g. 3.5) should raise ValueError."""
        with pytest.raises(ValueError, match="integer"):
            pm.gsynth_train(
                intervals=pm.gintervals("1", 0, 10000), iterator=200, k=3.5
            )

    def test_k_string_raises(self):
        """Non-numeric k (string) should raise."""
        with pytest.raises((ValueError, TypeError)):
            pm.gsynth_train(
                intervals=pm.gintervals("1", 0, 1000), iterator=200, k="abc"
            )

    def test_k_default_is_five(self):
        """Default k should be 5."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000), iterator=200,
        )
        assert model.k == 5
        assert model.num_kmers == 1024

    # ------------------------------------------------------------------
    # Training with various k values
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("k,expected_kmers", [
        (1, 4),
        (3, 64),
        (5, 1024),
        (7, 16384),
    ])
    def test_train_0d_various_k(self, k, expected_kmers):
        """Train 0D model with k={k} and verify attributes."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=k,
        )
        assert isinstance(model, pm.GsynthModel)
        assert model.k == k
        assert model.num_kmers == expected_kmers
        assert model.n_dims == 0
        assert model.total_bins == 1
        assert model.total_kmers > 0

        # CDF array shape
        cdf_mat = model.model_data["cdf"][0]
        assert cdf_mat.shape == (expected_kmers, 4)

        # Counts array shape
        counts_mat = model.model_data["counts"][0]
        assert counts_mat.shape == (expected_kmers, 4)

    def test_train_k10(self):
        """Train 0D model with k=10 (max)."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=10,
        )
        assert model.k == 10
        assert model.num_kmers == 4 ** 10  # 1048576
        assert model.model_data["cdf"][0].shape == (1048576, 4)

    @pytest.mark.parametrize("k", [1, 3, 5, 7])
    def test_cdf_validity_per_k(self, k):
        """CDF for k={k} should be monotonic with last column = 1.0."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=k,
        )
        cdf_mat = model.model_data["cdf"][0]
        n_kmers = 4 ** k
        assert cdf_mat.shape == (n_kmers, 4)

        # Values in [0, 1]
        assert np.all(cdf_mat >= 0)
        assert np.all(cdf_mat <= 1)

        # Last column should be 1.0
        np.testing.assert_allclose(cdf_mat[:, 3], 1.0, atol=1e-5)

        # Each row should be non-decreasing
        for ctx in range(min(n_kmers, 100)):  # spot-check first 100
            assert np.all(np.diff(cdf_mat[ctx, :]) >= -1e-10)

    # ------------------------------------------------------------------
    # k=5 explicit matches default
    # ------------------------------------------------------------------

    def test_k5_explicit_matches_default(self):
        """Training with explicit k=5 should produce identical results to default."""
        ivs = pm.gintervals("1", 0, 50000)
        model_k5 = pm.gsynth_train(intervals=ivs, iterator=200, k=5)
        model_default = pm.gsynth_train(intervals=ivs, iterator=200)

        assert model_k5.k == 5
        assert model_default.k == 5
        assert model_k5.num_kmers == 1024
        assert model_default.num_kmers == 1024

        # CDF matrices should be identical
        np.testing.assert_array_almost_equal(
            model_k5.model_data["cdf"][0],
            model_default.model_data["cdf"][0],
            decimal=15,
        )
        # Counts should be identical
        np.testing.assert_array_equal(
            model_k5.model_data["counts"][0],
            model_default.model_data["counts"][0],
        )

    # ------------------------------------------------------------------
    # Repr with variable k
    # ------------------------------------------------------------------

    def test_repr_shows_k(self):
        """Model repr should display the Markov order."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000), iterator=200, k=3
        )
        s = repr(model)
        assert "Markov" in s
        assert "3" in s


class TestGsynthVariableKSaveLoad:
    """Save/load round-trip tests for variable Markov order k."""

    # ------------------------------------------------------------------
    # Save/load round-trip with k != 5
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("k", [3, 7])
    def test_save_load_roundtrip_variable_k(self, k):
        """Train with k={k}, save to .gsm, load -> k and data preserved."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=k,
        )
        assert model.k == k
        n_kmers = 4 ** k

        path = os.path.join(tempfile.mkdtemp(), f"model_k{k}.gsm")
        try:
            pm.gsynth_save(model, path)
            loaded = pm.gsynth_load(path)

            assert isinstance(loaded, pm.GsynthModel)
            assert loaded.k == k
            assert loaded.num_kmers == n_kmers

            # CDF dimensions match
            assert loaded.model_data["cdf"][0].shape == (n_kmers, 4)

            # Counts match exactly
            np.testing.assert_array_equal(
                loaded.model_data["counts"][0],
                model.model_data["counts"][0],
            )
            # CDF match exactly
            np.testing.assert_array_almost_equal(
                loaded.model_data["cdf"][0],
                model.model_data["cdf"][0],
                decimal=15,
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_save_load_roundtrip_k1(self):
        """Train with k=1 (minimal), save, load -> preserved."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=1,
        )
        assert model.k == 1
        assert model.num_kmers == 4

        path = os.path.join(tempfile.mkdtemp(), "model_k1.gsm")
        try:
            pm.gsynth_save(model, path)
            loaded = pm.gsynth_load(path)

            assert loaded.k == 1
            assert loaded.num_kmers == 4
            assert loaded.model_data["cdf"][0].shape == (4, 4)
            np.testing.assert_array_equal(
                loaded.model_data["counts"][0],
                model.model_data["counts"][0],
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # .gsm metadata version: k=5 -> v1, k!=5 -> v2
    # ------------------------------------------------------------------

    def test_metadata_version_k5_is_v1(self):
        """k=5 model should save with version 1 for backward compatibility."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000),
            iterator=200,
            k=5,
        )
        path = os.path.join(tempfile.mkdtemp(), "model_k5.gsm")
        try:
            pm.gsynth_save(model, path)
            with open(os.path.join(path, "metadata.yaml")) as f:
                meta = yaml.safe_load(f)
            assert meta["version"] == 1
            assert meta["markov_order"] == 5
        finally:
            shutil.rmtree(path, ignore_errors=True)

    @pytest.mark.parametrize("k", [1, 3, 7])
    def test_metadata_version_k_nondefault_is_v2(self, k):
        """k!= 5 models should save with version 2."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=k,
        )
        path = os.path.join(tempfile.mkdtemp(), f"model_k{k}.gsm")
        try:
            pm.gsynth_save(model, path)
            with open(os.path.join(path, "metadata.yaml")) as f:
                meta = yaml.safe_load(f)
            assert meta["version"] == 2
            assert meta["markov_order"] == k
        finally:
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Backward compatibility: old v1 files load as k=5
    # ------------------------------------------------------------------

    def test_backward_compat_v1_loads_as_k5(self):
        """A v1 format .gsm file (no markov_order) should load with k=5."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000),
            iterator=200,
            k=5,
        )
        path = os.path.join(tempfile.mkdtemp(), "model_v1.gsm")
        try:
            pm.gsynth_save(model, path)

            # Manually strip markov_order from metadata to simulate old v1
            meta_path = os.path.join(path, "metadata.yaml")
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            assert meta["version"] == 1
            # Keep version=1 but remove markov_order
            meta.pop("markov_order", None)
            with open(meta_path, "w") as f:
                yaml.safe_dump(meta, f)

            loaded = pm.gsynth_load(path)
            assert loaded.k == 5
            assert loaded.num_kmers == 1024
        finally:
            shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Sampling identical from saved/loaded model with variable k
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("k", [3, 7])
    def test_sampling_identical_after_save_load(self, k):
        """Sampling from loaded model should match original with same seed."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=k,
        )
        path = os.path.join(tempfile.mkdtemp(), f"model_k{k}.gsm")
        try:
            pm.gsynth_save(model, path)
            loaded = pm.gsynth_load(path)

            ivs = pm.gintervals("1", 0, 5000)
            s1 = pm.gsynth_sample(model, intervals=ivs, seed=60427)
            s2 = pm.gsynth_sample(loaded, intervals=ivs, seed=60427)
            assert s1 == s2
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_save_load_zip_variable_k(self):
        """Compressed (.zip) round-trip preserves k."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000),
            iterator=200,
            k=3,
        )
        with tempfile.NamedTemporaryFile(suffix=".gsm.zip", delete=False) as f:
            path = f.name
        try:
            pm.gsynth_save(model, path, compress=True)
            loaded = pm.gsynth_load(path)
            assert loaded.k == 3
            assert loaded.num_kmers == 64
            assert loaded.model_data["cdf"][0].shape == (64, 4)
        finally:
            os.unlink(path)


class TestGsynthVariableKSampling:
    """Sampling with variable Markov order k."""

    @pytest.mark.parametrize("k", [1, 3, 5, 7])
    def test_sample_valid_dna_per_k(self, k):
        """Sampling from k={k} model should produce valid DNA sequences."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=k,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("1", 0, 5000),
            seed=60427,
        )
        assert len(seqs) == 1
        assert len(seqs[0]) == 5000
        assert all(c in "ACGT" for c in seqs[0])

    def test_sample_k1_correct_length(self):
        """k=1 model generates sequences of the correct length."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=1,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("1", 0, 10000),
            seed=60427,
        )
        assert len(seqs[0]) == 10000

    def test_sample_seed_reproducible_variable_k(self):
        """Same seed -> identical output for k=3 model."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=3,
        )
        ivs = pm.gintervals("1", 0, 5000)
        s1 = pm.gsynth_sample(model, intervals=ivs, seed=60427)
        s2 = pm.gsynth_sample(model, intervals=ivs, seed=60427)
        assert s1 == s2

    def test_sample_different_seeds_variable_k(self):
        """Different seeds -> different output for k=3 model."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=3,
        )
        ivs = pm.gintervals("1", 0, 5000)
        s1 = pm.gsynth_sample(model, intervals=ivs, seed=60427)
        s2 = pm.gsynth_sample(model, intervals=ivs, seed=123)
        assert s1 != s2

    def test_sample_fasta_output_variable_k(self):
        """FASTA output with k=3 model should be valid."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=3,
        )
        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            output_path = f.name
        try:
            pm.gsynth_sample(
                model,
                output=output_path,
                output_format="fasta",
                intervals=pm.gintervals("1", 0, 5000),
                seed=60427,
            )
            assert os.path.exists(output_path)
            with open(output_path) as f:
                lines = f.readlines()

            seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
            full_seq = "".join(seq_lines)
            assert len(full_seq) == 5000
            assert all(c in "ACGT" for c in full_seq)
        finally:
            os.unlink(output_path)

    def test_sample_multi_interval_variable_k(self):
        """k=3 model works with multiple chromosomes."""
        model = pm.gsynth_train(
            intervals=pm.gintervals_all(),
            iterator=200,
            k=3,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals(["1", "2"], [0, 0], [1000, 1000]),
            seed=60427,
        )
        assert len(seqs) == 2
        assert len(seqs[0]) == 1000
        assert len(seqs[1]) == 1000
        assert all(c in "ACGT" for c in seqs[0])
        assert all(c in "ACGT" for c in seqs[1])

    def test_sample_n_samples_variable_k(self):
        """n_samples with k=3 generates multiple sequences."""
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            k=3,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("1", 0, 1000),
            n_samples=3,
            seed=60427,
        )
        assert len(seqs) == 3
        for s in seqs:
            assert len(s) == 1000
            assert all(c in "ACGT" for c in s)
        # At least some should differ
        assert len(set(seqs)) > 1


class TestGsynthVariableK1DStratification:
    """1D stratification with variable Markov order k."""

    def test_train_1d_with_k3(self):
        """Train 1D model with k=3 and verify structure."""
        pm.gvtrack_create("g_frac_k", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac_k", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac_k + c_frac_k",
                    "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
                k=3,
            )
            assert model.k == 3
            assert model.num_kmers == 64
            assert model.n_dims == 1
            assert model.dim_sizes == [5]
            assert model.total_bins == 5

            # All CDF matrices should be 64 x 4
            for b in range(model.total_bins):
                assert model.model_data["cdf"][b].shape == (64, 4)
        finally:
            pm.gvtrack_rm("g_frac_k")
            pm.gvtrack_rm("c_frac_k")

    def test_sample_from_1d_k3_model(self):
        """Sample from 1D k=3 model and verify valid DNA."""
        pm.gvtrack_create("g_frac_k", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac_k", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac_k + c_frac_k",
                    "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
                k=3,
            )
            seqs = pm.gsynth_sample(
                model,
                intervals=pm.gintervals("1", 0, 5000),
                iterator=200,
                seed=60427,
            )
            assert len(seqs[0]) == 5000
            assert all(c in "ACGT" for c in seqs[0])
        finally:
            pm.gvtrack_rm("g_frac_k")
            pm.gvtrack_rm("c_frac_k")

    def test_1d_save_load_variable_k(self):
        """Save/load round-trip for 1D model with k=3."""
        pm.gvtrack_create("g_frac_k", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac_k", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac_k + c_frac_k",
                    "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
                k=3,
            )
            path = os.path.join(tempfile.mkdtemp(), "model_1d_k3.gsm")
            try:
                pm.gsynth_save(model, path)
                loaded = pm.gsynth_load(path)

                assert loaded.k == 3
                assert loaded.num_kmers == 64
                assert loaded.n_dims == 1
                assert loaded.total_bins == 5
                for b in range(5):
                    np.testing.assert_array_equal(
                        loaded.model_data["counts"][b],
                        model.model_data["counts"][b],
                    )
                    np.testing.assert_array_almost_equal(
                        loaded.model_data["cdf"][b],
                        model.model_data["cdf"][b],
                        decimal=15,
                    )
            finally:
                shutil.rmtree(path, ignore_errors=True)
        finally:
            pm.gvtrack_rm("g_frac_k")
            pm.gvtrack_rm("c_frac_k")

    def test_per_bin_kmers_sum_with_variable_k(self):
        """per_bin_kmers should sum to total_kmers with k=3."""
        pm.gvtrack_create("g_frac_k", None, "kmer.frac", kmer="G")
        pm.gvtrack_create("c_frac_k", None, "kmer.frac", kmer="C")
        try:
            model = pm.gsynth_train(
                {
                    "expr": "g_frac_k + c_frac_k",
                    "breaks": np.linspace(0, 1, 11).tolist(),
                },
                intervals=pm.gintervals("1", 0, 50000),
                iterator=200,
                k=3,
            )
            assert int(np.sum(model.per_bin_kmers)) == model.total_kmers
        finally:
            pm.gvtrack_rm("g_frac_k")
            pm.gvtrack_rm("c_frac_k")


class TestGsynthVariableKParallel:
    """Parallel training and sampling with variable Markov order k."""

    def test_parallel_train_k3_matches_serial(self):
        """Parallel train with k=3 should match serial."""
        ivs = pm.gintervals_all()
        model_serial = pm.gsynth_train(
            intervals=ivs, iterator=200, k=3, allow_parallel=False,
        )
        model_parallel = pm.gsynth_train(
            intervals=ivs, iterator=200, k=3, allow_parallel=True,
        )
        assert model_serial.k == model_parallel.k == 3
        assert model_serial.total_kmers == model_parallel.total_kmers

        np.testing.assert_array_equal(
            model_serial.model_data["counts"][0],
            model_parallel.model_data["counts"][0],
        )
        np.testing.assert_array_almost_equal(
            model_serial.model_data["cdf"][0],
            model_parallel.model_data["cdf"][0],
            decimal=15,
        )

    def test_parallel_sample_k3_valid(self):
        """Parallel sampling from k=3 model produces valid sequences."""
        model = pm.gsynth_train(
            intervals=pm.gintervals_all(),
            iterator=200,
            k=3,
            allow_parallel=True,
        )
        seqs = pm.gsynth_sample(
            model,
            intervals=pm.gintervals("1", 0, 10000),
            seed=60427,
            allow_parallel=True,
        )
        assert len(seqs) == 1
        assert len(seqs[0]) == 10000
        assert all(c in "ACGT" for c in seqs[0])


class TestGsynthIteratorAlignment:
    """Regression tests for the iter_size inference bug (R misha #94).

    gsynth_sample must honor the model's iterator value when intervals are
    not aligned to the iterator bin boundary, instead of silently inferring
    iter_size from the first same-chrom diff in iter_starts (which equals
    the partial first bin width and triggers uniform-random fallback for
    every position past that partial bin).
    """

    def test_sample_honors_iterator_on_unaligned_intervals(self):
        """Force CDF rows to always emit base A, sample on an interval whose
        start is not a multiple of the iterator (200), and assert that no
        post-seed positions fall through to uniform-random sampling.

        Before the fix: iter_size is inferred as iter_starts[1] - iter_starts[0]
        which for an interval starting at 64 with iterator=200 equals 136.
        Positions 200..335 only count as valid for offsets 0..135 within their
        bin, so 336..399, 536..599 fall through to uniform random and emit
        non-A bases. After the fix: zero non-A bases past seed.
        """
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            train_intervals = pm.gintervals("1", 0, 50000)
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                intervals=train_intervals,
                iterator=200,
            )

            # Force every CDF cell to 1.0: smallest base index (A=0) always
            # wins because the sampler picks the first b with rand() < cdf[b].
            for b in range(len(model.model_data["cdf"])):
                model.model_data["cdf"][b][:] = 1.0

            unaligned = pm.gintervals("1", 64, 664)
            seqs = pm.gsynth_sample(
                model,
                output_format="vector",
                intervals=unaligned,
                iterator=200,
                seed=60427,
            )

            assert len(seqs) == 1
            s = seqs[0]
            assert len(s) == 600

            # First k=5 bases are uniform random (seed init), so skip them.
            post_seed = s[5:]
            non_A = sum(1 for c in post_seed if c != "A")
            assert non_A == 0, (
                f"Expected all 595 post-seed bases to be A, got {non_A} non-A. "
                "iter_size was inferred from a partial first bin instead of "
                "the model's iterator value."
            )
        finally:
            pm.gvtrack_rm("test_vt")

    def test_aligned_and_unaligned_produce_matching_forbidden_kmer_stats(self):
        """Forbid C->G transitions in the CDF, sample aligned and unaligned
        intervals over the same region, and assert both have zero CG bigrams
        past the seeded prefix. Before the fix, the unaligned interval picks
        up CG bigrams from the uniform-fallback regions where iter_size
        truncates the bin coverage.
        """
        pm.gvtrack_create("test_vt", "dense_track", "avg")
        try:
            train_intervals = pm.gintervals("1", 0, 50000)
            model = pm.gsynth_train(
                {"expr": "test_vt", "breaks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                intervals=train_intervals,
                iterator=200,
            )

            # Forbid C -> G: zero cdf cells where context ends in C and next
            # base is G, then renormalise. State row r encodes a k-mer; its
            # last base is r % 4. Bases: A=0, C=1, G=2, T=3.
            num_kmers = 4 ** model.k
            state_ends_in_C = np.array(
                [(r % 4) == 1 for r in range(num_kmers)]
            )
            for b in range(len(model.model_data["cdf"])):
                cdf = model.model_data["cdf"][b].copy()
                # Recover per-row probabilities from cumulative.
                probs = np.zeros((num_kmers, 4))
                probs[:, 0] = cdf[:, 0]
                probs[:, 1] = cdf[:, 1] - cdf[:, 0]
                probs[:, 2] = cdf[:, 2] - cdf[:, 1]
                probs[:, 3] = 1.0 - cdf[:, 2]
                probs[state_ends_in_C, 2] = 0.0
                row_sums = probs.sum(axis=1)
                nz = row_sums > 0
                probs[nz] = probs[nz] / row_sums[nz, None]
                new_cdf = np.cumsum(probs, axis=1)
                new_cdf[:, 3] = 1.0
                model.model_data["cdf"][b] = new_cdf

            aligned = pm.gintervals("1", 0, 2000)
            unaligned = pm.gintervals("1", 64, 2064)

            seq_a = pm.gsynth_sample(
                model, output_format="vector",
                intervals=aligned, iterator=200, seed=60427,
            )[0]
            seq_u = pm.gsynth_sample(
                model, output_format="vector",
                intervals=unaligned, iterator=200, seed=60427,
            )[0]

            # Skip first k=5 seeded bases (uniform random).
            tail_a = seq_a[5:]
            tail_u = seq_u[5:]
            assert tail_a.count("CG") == 0
            assert tail_u.count("CG") == 0, (
                f"Unaligned interval has {tail_u.count('CG')} CG bigrams; "
                "expected 0 once iter_size honors the model's iterator."
            )
        finally:
            pm.gvtrack_rm("test_vt")


class TestGsynthScore:
    """gsynth_score: per-bp log-likelihood under a trained model
    aggregated to a misha dense track (R misha 5.6.21 ba88e197 + 3fba28c2)."""

    def setup_method(self):
        for t in ("ts_score", "ts_score_strat"):
            try:
                pm.gtrack_rm(t, force=True)
            except Exception:
                pass

    def teardown_method(self):
        for t in ("ts_score", "ts_score_strat"):
            try:
                pm.gtrack_rm(t, force=True)
            except Exception:
                pass

    def test_0d_score_writes_track(self):
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000), iterator=200
        )
        pm.gsynth_score(
            model, "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            resolution=200,
        )
        out = pm.gextract(
            "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            iterator=200,
        )
        assert len(out) == 10
        # Bin 0 includes positions 0..4 which lack k-upstream context →
        # NA. Subsequent bins should be valid log-probability sums (i.e.
        # finite negative numbers).
        assert np.isnan(out["ts_score"].iloc[0])
        assert out["ts_score"].iloc[1:].notna().all()
        # All per-bin values are negative log-probability sums.
        assert (out["ts_score"].iloc[1:] < 0).all()

    def test_score_resolution_default(self):
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000), iterator=200
        )
        pm.gsynth_score(
            model, "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
        )
        # Default resolution = model.iterator = 200
        out = pm.gextract(
            "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            iterator=200,
        )
        assert len(out) == 10

    def test_score_mask_poisons_bins(self):
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000), iterator=200
        )
        pm.gsynth_score(
            model, "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            mask=pm.gintervals("1", 600, 700),
            resolution=200,
        )
        out = pm.gextract(
            "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            iterator=200,
        )
        # Bin covering 600-800 should be NaN due to mask.
        masked_bin = out[(out["start"] >= 600) & (out["start"] < 800)]
        assert masked_bin["ts_score"].isna().all()

    def test_score_invalid_policies_raise(self):
        model = pm.gsynth_train(intervals=pm.gintervals("1", 0, 10000))
        with pytest.raises(ValueError, match="sparse_policy"):
            pm.gsynth_score(model, "ts_score", sparse_policy="bogus")
        with pytest.raises(ValueError, match="n_policy"):
            pm.gsynth_score(model, "ts_score", n_policy="bogus")
        with pytest.raises(ValueError, match="resolution"):
            pm.gsynth_score(model, "ts_score", resolution=-1)

    def test_score_overwrite(self):
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 10000), iterator=200
        )
        pm.gsynth_score(
            model, "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            resolution=200,
        )
        # Without overwrite, recreating should fail.
        with pytest.raises(Exception):
            pm.gsynth_score(
                model, "ts_score",
                intervals=pm.gintervals("1", 0, 2000),
                resolution=200,
            )
        # With overwrite, it should succeed.
        pm.gsynth_score(
            model, "ts_score",
            intervals=pm.gintervals("1", 0, 2000),
            resolution=200,
            overwrite=True,
        )

    def test_score_stratified_model(self):
        model = pm.gsynth_train(
            {"expr": "dense_track", "breaks": [0.0, 0.5, 1.0]},
            intervals=pm.gintervals("1", 0, 10000),
            iterator=200,
        )
        pm.gsynth_score(
            model, "ts_score_strat",
            intervals=pm.gintervals("1", 0, 5000),
            resolution=500,
        )
        out = pm.gextract(
            "ts_score_strat",
            intervals=pm.gintervals("1", 0, 5000),
            iterator=500,
        )
        # Some bins must be valid (those with full coverage and a known
        # stratum) and the values should be negative log probabilities.
        valid = out["ts_score_strat"].dropna()
        assert len(valid) >= 1
        assert (valid < 0).all()


class TestGsynthDirichletPrior:
    """Per-bin Dirichlet prior in gsynth_train (R misha 5.6.21)."""

    def test_default_prior_is_marginal(self):
        model = pm.gsynth_train(intervals=pm.gintervals("1", 0, 50000))
        assert model.prior_mode == "marginal"
        assert model.prior_matrix is not None
        assert model.prior_matrix.shape == (1, 4)
        # Rows sum to 1.
        assert np.allclose(model.prior_matrix.sum(axis=1), 1.0)
        # Marginal pi must reflect actual chr1 composition (non-uniform).
        assert not np.allclose(model.prior_matrix[0], 0.25, atol=0.01)

    def test_prior_uniform_recovers_legacy(self):
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000), prior="uniform"
        )
        assert model.prior_mode == "uniform"
        assert np.allclose(model.prior_matrix, 0.25)

    def test_prior_global_broadcasts(self):
        # Stratified model so n_dims > 0 and there is more than one bin.
        model = pm.gsynth_train(
            {"expr": "dense_track", "breaks": [0.0, 0.5, 1.0]},
            intervals=pm.gintervals("1", 0, 50000),
            iterator=200,
            prior="global",
        )
        assert model.prior_mode == "global"
        # Global broadcasts the same pi to every bin.
        assert model.prior_matrix.shape[0] == model.total_bins
        for b in range(model.total_bins):
            assert np.allclose(model.prior_matrix[b], model.prior_matrix[0])

    def test_prior_explicit_round_trip(self):
        custom = np.array([[0.4, 0.1, 0.1, 0.4]], dtype=float)
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50000), prior=custom
        )
        assert model.prior_mode == "explicit"
        assert np.allclose(model.prior_matrix, custom)

    def test_prior_explicit_wrong_shape_raises(self):
        bad = np.array([[0.4, 0.1, 0.1, 0.4], [0.25, 0.25, 0.25, 0.25]])
        with pytest.raises(ValueError, match="rows"):
            pm.gsynth_train(
                intervals=pm.gintervals("1", 0, 50000), prior=bad
            )

    def test_prior_round_trips_through_save_load(self, tmp_path):
        model = pm.gsynth_train(intervals=pm.gintervals("1", 0, 50000))
        path = tmp_path / "model.gsm"
        pm.gsynth_save(model, str(path))
        restored = pm.gsynth_load(str(path))
        assert restored.prior_mode == model.prior_mode
        assert np.allclose(restored.prior_matrix, model.prior_matrix)
        # Sampling should produce the same sequence under the same seed.
        seq1 = pm.gsynth_sample(
            model, intervals=pm.gintervals("1", 0, 1000),
            seed=7, output_format="vector",
        )[0]
        seq2 = pm.gsynth_sample(
            restored, intervals=pm.gintervals("1", 0, 1000),
            seed=7, output_format="vector",
        )[0]
        assert seq1 == seq2


class TestGsynthPreserveN:
    """preserve_n: positions whose reference is N stay N in the output."""

    def _find_n_interval(self):
        ref = pm.gseq_extract(pm.gintervals(["X"], [0], [200000]))[0]
        idx = ref.find("N")
        assert idx >= 0, "expected at least one N in chrX:0-200000 example db"
        start = max(0, idx - 50)
        end = min(200000, idx + 200)
        return start, end, ref[start:end]

    def test_default_preserves_n(self):
        start, end, ref_slice = self._find_n_interval()
        intervals = pm.gintervals(["X"], [start], [end])
        model = pm.gsynth_train(intervals=intervals)
        seq = pm.gsynth_sample(
            model, output_format="vector", intervals=intervals, seed=60427
        )[0]
        for i, c in enumerate(ref_slice):
            if c in "Nn":
                assert seq[i] == c, f"position {i} should be {c}, got {seq[i]}"

    def test_preserve_n_false_fabricates_acgt(self):
        start, end, ref_slice = self._find_n_interval()
        intervals = pm.gintervals(["X"], [start], [end])
        model = pm.gsynth_train(intervals=intervals)
        seq = pm.gsynth_sample(
            model, output_format="vector", intervals=intervals,
            preserve_n=False, seed=60427,
        )[0]
        n_count = sum(1 for c in seq if c in "Nn")
        ref_n = sum(1 for c in ref_slice if c in "Nn")
        assert ref_n > 0
        assert n_count == 0

    def test_random_preserves_n(self):
        start, end, ref_slice = self._find_n_interval()
        intervals = pm.gintervals(["X"], [start], [end])
        seq = pm.gsynth_random(
            intervals=intervals, output_format="vector", seed=60427
        )[0]
        for i, c in enumerate(ref_slice):
            if c in "Nn":
                assert seq[i] == c
