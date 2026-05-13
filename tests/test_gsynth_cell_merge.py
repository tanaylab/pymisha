"""Tests for gsynth_cell_merge resolver and gsynth_sample(cell_merge=)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha.gsynth import (
    _cell_merge_normalize,
    _cell_merge_sample_bin_maps,
    _compute_flat_indices,
)

# ---------------------------------------------------------------------------
# Task 1: _compute_flat_indices helper
# ---------------------------------------------------------------------------

class TestComputeFlatIndices:
    """Match R synth.R:5 .compute_flat_indices semantics: 1-based in, 1-based out, NA propagates."""

    def test_1d_identity(self):
        # Single dim, indices 1..5 -> flat 1..5
        per_dim = np.array([[1], [2], [3], [4], [5]], dtype=np.int64)
        flat = _compute_flat_indices(per_dim, [5])
        assert flat.tolist() == [1, 2, 3, 4, 5]

    def test_2d_column_major(self):
        # dim_sizes=[3,2]: cell (1,1)->1, (2,1)->2, (3,1)->3, (1,2)->4, (2,2)->5, (3,2)->6
        per_dim = np.array(
            [[1, 1], [2, 1], [3, 1], [1, 2], [2, 2], [3, 2]],
            dtype=np.int64,
        )
        flat = _compute_flat_indices(per_dim, [3, 2])
        assert flat.tolist() == [1, 2, 3, 4, 5, 6]

    def test_3d(self):
        # dim_sizes=[2,3,2]: stride = (1, 2, 6)
        per_dim = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=np.int64)
        flat = _compute_flat_indices(per_dim, [2, 3, 2])
        # flat_0based = (i1-1)*1 + (i2-1)*2 + (i3-1)*6
        # (1,1,1)->1, (2,1,1)->2, (1,2,1)->3, (1,1,2)->7
        assert flat.tolist() == [1, 2, 3, 7]

    def test_na_in_any_dim_propagates(self):
        # Pandas NA sentinel: we use -1 in pymisha (consistent with bin_idx -1 = invalid)
        per_dim = np.array([[1, 1], [-1, 2], [2, -1], [3, 2]], dtype=np.int64)
        flat = _compute_flat_indices(per_dim, [3, 2])
        # Valid: row 0 -> 1, row 3 -> 6. Invalid rows -> -1.
        assert flat.tolist() == [1, -1, -1, 6]

    def test_empty_input(self):
        flat = _compute_flat_indices(np.empty((0, 2), dtype=np.int64), [3, 2])
        assert flat.shape == (0,)
        assert flat.dtype == np.int64

    def test_all_rows_invalid(self):
        per_dim = np.array([[-1, 1], [1, -1]], dtype=np.int64)
        flat = _compute_flat_indices(per_dim, [3, 2])
        assert flat.tolist() == [-1, -1]


# ---------------------------------------------------------------------------
# Task 2: _cell_merge_normalize / _cell_merge_sample_bin_maps
# ---------------------------------------------------------------------------


class TestCellMergeNormalize:
    def test_empty_input_returns_empty_list(self):
        assert _cell_merge_normalize(None, n_dims=2) == []
        assert _cell_merge_normalize([], n_dims=2) == []

    def test_list_of_dicts_passthrough(self):
        out = _cell_merge_normalize(
            [{"from": [0.7, 0.05], "to": [0.6, 0.08]}], n_dims=2
        )
        assert len(out) == 1
        assert out[0]["from"] == [0.7, 0.05]
        assert out[0]["to"] == [0.6, 0.08]

    def test_dataframe_input_converts(self):
        df = pd.DataFrame({"from_1": [0.7], "to_1": [0.6], "from_2": [0.05], "to_2": [0.08]})
        out = _cell_merge_normalize(df, n_dims=2)
        assert len(out) == 1
        assert out[0]["from"] == [0.7, 0.05]
        assert out[0]["to"] == [0.6, 0.08]

    def test_dataframe_missing_cols_raises(self):
        df = pd.DataFrame({"from_1": [0.7], "to_1": [0.6]})  # missing from_2/to_2
        with pytest.raises(ValueError, match="missing required columns"):
            _cell_merge_normalize(df, n_dims=2)

    def test_entry_missing_to_raises(self):
        with pytest.raises(ValueError, match="must be a dict with 'from' and 'to'"):
            _cell_merge_normalize([{"from": [0.7, 0.05]}], n_dims=2)

    def test_entry_wrong_length_raises(self):
        with pytest.raises(ValueError, match=r"length 2 \(n_dims\)"):
            _cell_merge_normalize([{"from": [0.7], "to": [0.6, 0.08]}], n_dims=2)

    def test_entry_scalar_from_raises_clean_value_error(self):
        with pytest.raises(ValueError, match=r"length 2 \(n_dims\)"):
            _cell_merge_normalize([{"from": 0.7, "to": [0.6, 0.08]}], n_dims=2)


class TestCellMergeSampleBinMaps:
    def test_no_bin_merge_falls_back_to_training_bin_map(self):
        # Real GsynthModel stores bin_map as 0-based (gsynth_bin_map output).
        # The helper must convert to 1-based for downstream _compute_flat_indices.
        class FakeModel:
            n_dims = 2
            dim_specs = [
                {"breaks": [0.0, 0.3, 0.6, 1.0], "num_bins": 3, "bin_map": np.array([0, 1, 2])},
                {"breaks": [0.0, 0.5, 1.0],      "num_bins": 2, "bin_map": np.array([0, 1])},
            ]

        out = _cell_merge_sample_bin_maps(FakeModel(), bin_merge=None)
        assert len(out) == 2
        assert out[0].tolist() == [1, 2, 3]
        assert out[1].tolist() == [1, 2]

    def test_bin_merge_override_applies(self):
        class FakeModel:
            n_dims = 2
            dim_specs = [
                {"breaks": [0.0, 0.3, 0.6, 1.0], "num_bins": 3, "bin_map": np.array([0, 1, 2])},
                {"breaks": [0.0, 0.5, 1.0],      "num_bins": 2, "bin_map": np.array([0, 1])},
            ]

        # Merge dim 0 bin range (0.0, 0.3) into (0.3, 0.6).
        out = _cell_merge_sample_bin_maps(
            FakeModel(), bin_merge=[[{"from": (0.0, 0.3), "to": (0.3, 0.6)}], None]
        )
        # dim 0: bin 1 redirected to bin 2 (1-based); dim 1: unchanged 1-based identity.
        assert out[0].tolist() == [2, 2, 3]
        assert out[1].tolist() == [1, 2]

    def test_bin_merge_wrong_length_raises(self):
        class FakeModel:
            n_dims = 2
            dim_specs = [
                {"breaks": [0.0, 1.0], "num_bins": 1, "bin_map": np.array([0])},
                {"breaks": [0.0, 1.0], "num_bins": 1, "bin_map": np.array([0])},
            ]
        with pytest.raises(ValueError, match=r"bin_merge must be a list with 2 elements"):
            _cell_merge_sample_bin_maps(FakeModel(), bin_merge=[None])

    def test_realistic_int32_bin_map_storage(self):
        # Confirm dtype handling: real models may store int32; helper must cast safely.
        class FakeModel:
            n_dims = 1
            dim_specs = [
                {"breaks": [0.0, 0.5, 1.0], "num_bins": 2,
                 "bin_map": np.array([0, 1], dtype=np.int32)},
            ]
        out = _cell_merge_sample_bin_maps(FakeModel(), bin_merge=None)
        assert out[0].tolist() == [1, 2]
        assert out[0].dtype == np.int64

    def test_none_bin_map_uses_1based_identity(self):
        # gsynth_train stores bin_map=None when no training-time bin_merge is
        # given. The helper must produce a 1-based identity map of length num_bins.
        class FakeModel:
            n_dims = 1
            dim_specs = [{"breaks": [0.0, 0.5, 1.0], "num_bins": 2, "bin_map": None}]
        out = _cell_merge_sample_bin_maps(FakeModel(), bin_merge=None)
        assert out[0].tolist() == [1, 2]
        assert out[0].dtype == np.int64


# ---------------------------------------------------------------------------
# Task 3: gsynth_cell_merge public API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cell_merge_model_2d():
    """2D stratified gsynth model on chrom 1 of the example DB.

    Two dimensions: g_frac (5 bins over [0, 1]) and cg_frac (4 bins over [0, 0.1]).
    """
    pm.gdb_init_examples()
    # pymisha exposes per-kmer fractions via `kmer.frac` (the R-misha `dna.g_frac`
    # / `dna.cpg_frac` shortcuts are not surfaced as pymisha vtrack functions).
    pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
    pm.gvtrack_create("cg_frac", None, "kmer.frac", kmer="CG", strand=1)
    try:
        model = pm.gsynth_train(
            {"expr": "g_frac", "breaks": np.linspace(0, 1.0, 6).tolist()},
            {"expr": "cg_frac", "breaks": np.linspace(0, 0.1, 5).tolist()},
            intervals=pm.gintervals("1", 0, 100_000),
            iterator=200,
            k=2,
        )
        yield model
    finally:
        pm.gvtrack_rm("g_frac")
        pm.gvtrack_rm("cg_frac")


class TestGsynthCellMerge:
    @pytest.fixture(autouse=True)
    def _restore_vtracks(self, cell_merge_model_2d):
        """Re-register vtracks if a sibling test wiped them via gdb_init_examples."""
        import pymisha._shared as _shared
        if "g_frac" not in _shared._VTRACKS:
            pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        if "cg_frac" not in _shared._VTRACKS:
            pm.gvtrack_create("cg_frac", None, "kmer.frac", kmer="CG", strand=1)
        yield

    def test_resolves_value_space_to_flat_indices(self, cell_merge_model_2d):
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        # Midpoint of bin 1 in dim 0 and bin 2 in dim 1.
        from_vals = [
            (spec0["breaks"][0] + spec0["breaks"][1]) / 2,
            (spec1["breaks"][1] + spec1["breaks"][2]) / 2,
        ]
        to_vals = [
            (spec0["breaks"][1] + spec0["breaks"][2]) / 2,
            (spec1["breaks"][2] + spec1["breaks"][3]) / 2,
        ]
        resolved = pm.gsynth_cell_merge(
            cell_merge_model_2d,
            [{"from": from_vals, "to": to_vals}],
        )
        assert list(resolved.columns) == [
            "source_flat", "target_flat", "from_1", "to_1", "from_2", "to_2",
        ]
        assert len(resolved) == 1
        d0 = int(spec0["num_bins"])
        expected_src = 1 + (2 - 1) * d0
        expected_tgt = 2 + (3 - 1) * d0
        assert int(resolved["source_flat"].iloc[0]) == expected_src
        assert int(resolved["target_flat"].iloc[0]) == expected_tgt

    def test_accepts_dataframe_spec(self, cell_merge_model_2d):
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        df = pd.DataFrame({
            "from_1": [(spec0["breaks"][0] + spec0["breaks"][1]) / 2],
            "to_1": [(spec0["breaks"][1] + spec0["breaks"][2]) / 2],
            "from_2": [(spec1["breaks"][1] + spec1["breaks"][2]) / 2],
            "to_2": [(spec1["breaks"][2] + spec1["breaks"][3]) / 2],
        })
        list_form = [{
            "from": [df["from_1"].iloc[0], df["from_2"].iloc[0]],
            "to": [df["to_1"].iloc[0], df["to_2"].iloc[0]],
        }]
        a = pm.gsynth_cell_merge(cell_merge_model_2d, df).reset_index(drop=True)
        b = pm.gsynth_cell_merge(cell_merge_model_2d, list_form).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_out_of_range_value_raises(self, cell_merge_model_2d):
        with pytest.raises(ValueError, match=r"out of range in dimension"):
            pm.gsynth_cell_merge(
                cell_merge_model_2d,
                [{"from": [1.5, 0.05], "to": [0.5, 0.05]}],  # 1.5 > breaks[-1]=1.0
            )

    def test_empty_cell_merge_returns_empty_frame(self, cell_merge_model_2d):
        resolved = pm.gsynth_cell_merge(cell_merge_model_2d, [])
        assert len(resolved) == 0
        assert list(resolved.columns) == [
            "source_flat", "target_flat", "from_1", "to_1", "from_2", "to_2",
        ]

    def test_rejects_0d_model(self):
        # A 0D model has n_dims == 0; cell_merge requires n_dims >= 1.
        pm.gdb_init_examples()
        model = pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50_000),
            iterator=200,
            k=2,
        )
        with pytest.raises(ValueError, match=r"requires a stratified model"):
            pm.gsynth_cell_merge(model, [{"from": [], "to": []}])

    def test_honors_bin_merge(self, cell_merge_model_2d):
        # bin_merge on dim 0 collapses bin 1 into bin 2; cell_merge `from` value
        # inside bin 1 should resolve to flat index referencing bin 2.
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        from_vals = [
            (spec0["breaks"][0] + spec0["breaks"][1]) / 2,
            (spec1["breaks"][1] + spec1["breaks"][2]) / 2,
        ]
        to_vals = [
            (spec0["breaks"][1] + spec0["breaks"][2]) / 2,
            (spec1["breaks"][1] + spec1["breaks"][2]) / 2,
        ]
        bin_merge = [
            [{"from": (spec0["breaks"][0], spec0["breaks"][1]),
              "to": (spec0["breaks"][1], spec0["breaks"][2])}],
            None,
        ]
        resolved = pm.gsynth_cell_merge(
            cell_merge_model_2d,
            [{"from": from_vals, "to": to_vals}],
            bin_merge=bin_merge,
        )
        d0 = int(spec0["num_bins"])
        expected_src = 2 + (2 - 1) * d0
        assert int(resolved["source_flat"].iloc[0]) == expected_src

    def test_rejects_non_gsynth_model(self):
        with pytest.raises(TypeError, match="GsynthModel"):
            pm.gsynth_cell_merge("not a model", [])


# ---------------------------------------------------------------------------
# Task 4: gsynth_sample(cell_merge=) integration
# ---------------------------------------------------------------------------

class TestGsynthSampleCellMerge:
    """gsynth_sample(cell_merge=) integration tests.

    These tests call ``gsynth_sample``, which re-extracts the model's
    stratification expressions ("g_frac" / "cg_frac") via ``gextract``.
    Other tests in this module (e.g. ``test_rejects_0d_model``) call
    ``gdb_init_examples`` again and silently wipe ``_VTRACKS``; the
    autouse fixture below re-registers them automatically.
    """

    @pytest.fixture(autouse=True)
    def _restore_vtracks(self, cell_merge_model_2d):
        """Re-register vtracks if a sibling test wiped them via gdb_init_examples."""
        import pymisha._shared as _shared
        if "g_frac" not in _shared._VTRACKS:
            pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
        if "cg_frac" not in _shared._VTRACKS:
            pm.gvtrack_create("cg_frac", None, "kmer.frac", kmer="CG", strand=1)
        yield

    def test_identity_redirect_matches_no_cell_merge(self, cell_merge_model_2d):
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        # Identity: from == to (midpoint of bin 1 in each dim).
        vals = [(spec0["breaks"][0] + spec0["breaks"][1]) / 2,
                (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        ivs = pm.gintervals("1", 0, 2000)
        s_none = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
        )
        with pytest.warns(UserWarning, match="self-redirect"):
            s_id = pm.gsynth_sample(
                cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
                output_format="vector",
                cell_merge=[{"from": vals, "to": vals}],
            )
        assert s_id == s_none

    def test_redirect_changes_output(self, cell_merge_model_2d):
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        # Pick a from/to cell pair that is actually visited on chrom 1 at iter=200.
        # Bin (2, 1) covers low-moderate GC and low CpG - widely populated.
        from_vals = [(spec0["breaks"][1] + spec0["breaks"][2]) / 2,
                     (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        # Bin (5, 4) - high GC, high CpG - target's CDF should differ.
        to_vals = [(spec0["breaks"][-2] + spec0["breaks"][-1]) / 2,
                   (spec1["breaks"][-2] + spec1["breaks"][-1]) / 2]
        ivs = pm.gintervals("1", 0, 100_000)
        s_none = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
        )
        s_merged = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
            cell_merge=[{"from": from_vals, "to": to_vals}],
        )
        # Output differs in a non-trivial fraction of positions.
        assert len(s_none) == len(s_merged)
        differing = sum(
            1 for a, b in zip("".join(s_none), "".join(s_merged)) if a != b
        )
        total = sum(len(s) for s in s_none)
        # At least 1% of positions differ (typically much more if the redirect lands).
        assert differing / total > 0.01, (
            f"redirect produced only {differing}/{total} differing positions"
        )

    def test_warns_on_self_redirect(self, cell_merge_model_2d):
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        vals = [(spec0["breaks"][0] + spec0["breaks"][1]) / 2,
                (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        ivs = pm.gintervals("1", 0, 500)
        with pytest.warns(UserWarning, match="self-redirect"):
            pm.gsynth_sample(
                cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
                output_format="vector",
                cell_merge=[{"from": vals, "to": vals}],
            )

    def test_warns_on_duplicate_source(self, cell_merge_model_2d):
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        src = [(spec0["breaks"][0] + spec0["breaks"][1]) / 2,
               (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        tgt_a = [(spec0["breaks"][1] + spec0["breaks"][2]) / 2,
                 (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        tgt_b = [(spec0["breaks"][-2] + spec0["breaks"][-1]) / 2,
                 (spec1["breaks"][-2] + spec1["breaks"][-1]) / 2]
        ivs = pm.gintervals("1", 0, 500)
        with pytest.warns(UserWarning, match="duplicate source"):
            pm.gsynth_sample(
                cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
                output_format="vector",
                cell_merge=[
                    {"from": src, "to": tgt_a},
                    {"from": src, "to": tgt_b},
                ],
            )

    def test_dataframe_input_form_works(self, cell_merge_model_2d):
        """cell_merge=DataFrame should produce identical output to list-of-dicts."""
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        from_vals = [(spec0["breaks"][1] + spec0["breaks"][2]) / 2,
                     (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        to_vals = [(spec0["breaks"][-2] + spec0["breaks"][-1]) / 2,
                   (spec1["breaks"][-2] + spec1["breaks"][-1]) / 2]
        ivs = pm.gintervals("1", 0, 5000)
        s_list = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
            cell_merge=[{"from": from_vals, "to": to_vals}],
        )
        df = pd.DataFrame({
            "from_1": [from_vals[0]], "to_1": [to_vals[0]],
            "from_2": [from_vals[1]], "to_2": [to_vals[1]],
        })
        s_df = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
            cell_merge=df,
        )
        assert s_list == s_df

    def test_cell_merge_with_bin_merge_at_sample_time(self, cell_merge_model_2d):
        """cell_merge + bin_merge interact: cell_merge values resolve post-bin_merge."""
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        # bin_merge collapses dim 0 bin 1 into bin 2; cell_merge from-value in bin 1
        # should be redirected via the post-merge cell (bin 2 in dim 0).
        from_vals = [(spec0["breaks"][0] + spec0["breaks"][1]) / 2,
                     (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        to_vals = [(spec0["breaks"][-2] + spec0["breaks"][-1]) / 2,
                   (spec1["breaks"][-2] + spec1["breaks"][-1]) / 2]
        bin_merge = [
            [{"from": (spec0["breaks"][0], spec0["breaks"][1]),
              "to":   (spec0["breaks"][1], spec0["breaks"][2])}],
            None,
        ]
        ivs = pm.gintervals("1", 0, 5000)
        # Should not raise.
        s = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
            bin_merge=bin_merge,
            cell_merge=[{"from": from_vals, "to": to_vals}],
        )
        assert isinstance(s, list)
        # Output is valid DNA.
        for seq in s:
            assert all(c in "ACGTN" for c in seq)

    def test_cell_merge_works_in_parallel_path(self, cell_merge_model_2d):
        """Force parallel chunking; redirect must still apply across workers."""
        spec0 = cell_merge_model_2d.dim_specs[0]
        spec1 = cell_merge_model_2d.dim_specs[1]
        from_vals = [(spec0["breaks"][1] + spec0["breaks"][2]) / 2,
                     (spec1["breaks"][0] + spec1["breaks"][1]) / 2]
        to_vals = [(spec0["breaks"][-2] + spec0["breaks"][-1]) / 2,
                   (spec1["breaks"][-2] + spec1["breaks"][-1]) / 2]
        # Force chunking: low max_chunk_size, small interval is still enough
        # to trigger split given the threshold.
        ivs = pm.gintervals("1", 0, 50_000)
        s_serial = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
            allow_parallel=False,
            cell_merge=[{"from": from_vals, "to": to_vals}],
        )
        s_parallel = pm.gsynth_sample(
            cell_merge_model_2d, intervals=ivs, iterator=200, seed=60427,
            output_format="vector",
            allow_parallel=True, num_cores=2, max_chunk_size=10_000,
            cell_merge=[{"from": from_vals, "to": to_vals}],
        )
        # Note: parallel and serial may produce different exact sequences due to
        # per-chunk seeding, but both must succeed without error and produce
        # valid DNA of the right length.
        for s in (s_serial, s_parallel):
            assert sum(len(x) for x in s) == 50_000
            assert all(c in "ACGTN" for seq in s for c in seq)
