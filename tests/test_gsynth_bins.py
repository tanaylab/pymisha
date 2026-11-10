"""gsynth stratification bins as R misha: findInterval bin edges, flat index order, and
min_obs sparse bins (R tests from test-gsynth.R, test-gsynth-gsm.R, test-gsynth-score.R)."""

import os
import tempfile
import warnings
from collections import Counter

import numpy as np
import pytest
import yaml

import pymisha as pm

# dense_track and subdir.dense_track2 values (float32) at iterator 200 on the example DB; the
# interior breaks of B1 are values of 50 windows, and 0 is the value of 55.
B1 = [0, 0.099999994039535522, 0.185000002384185791, 0.430000007152557373]
B2 = [0, 0.28999999165534973, 0.86000001430511475]


@pytest.fixture(autouse=True)
def setup_db():
    pm.gdb_init_examples()
    yield


def _train(*specs, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.gsynth_train(*specs, **kw)


def _dense_breaks(n, extend):
    r = pm.gsummary("dense_track", intervals=pm.gintervals("1", 0, 50000))
    lo, hi = r["Min"], r["Max"]
    pad = (hi - lo) if extend else 0
    return np.linspace(lo - pad, hi + pad, n).tolist()


@pytest.fixture
def test_vt():
    pm.gvtrack_create("test_vt", "dense_track", "avg")
    yield "test_vt"
    pm.gvtrack_rm("test_vt")


@pytest.fixture
def gc_frac():
    pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G")
    pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C")
    yield
    pm.gvtrack_rm("g_frac")
    pm.gvtrack_rm("c_frac")


# ----------------------------------------------------------------------------
# Bin edges: findInterval(rightmost.closed = TRUE), [b_i, b_{i+1})
# ----------------------------------------------------------------------------


def test_values_on_first_break_are_counted():
    pm.gvtrack_create("cgf", None, "kmer.frac", kmer="CG", strand=1)
    try:
        m0 = _train(iterator=200, k=1)
        m1 = _train({"expr": "cgf", "breaks": [0, 0.01, 0.02, 0.05, 1]}, iterator=200, k=1)
        assert m1.total_kmers == m0.total_kmers
    finally:
        pm.gvtrack_rm("cgf")


def test_bin_counts_match_r():
    # R: gsynth.train(list(expr = "dense_track", breaks = B1), iterator = 200, k = 1)
    m1 = _train({"expr": "dense_track", "breaks": B1}, iterator=200, k=1)
    assert m1.total_kmers == 1565142
    assert list(m1.per_bin_kmers) == [973084, 508458, 83600]
    m2 = _train({"expr": "dense_track", "breaks": B1},
                {"expr": "subdir.dense_track2", "breaks": B2}, iterator=200, k=2)
    assert m2.total_kmers == 1565132
    assert list(m2.per_bin_kmers) == [973078, 368854, 0, 0, 139600, 83600]


def test_2d_score_matches_r():
    m2 = _train({"expr": "dense_track", "breaks": B1},
                {"expr": "subdir.dense_track2", "breaks": B2}, iterator=200, k=2)
    pm.gsynth_score(m2, "ts_bins_score", intervals=pm.gintervals("1", 0, 20000),
                    resolution=2000, overwrite=True)
    try:
        out = pm.gextract("ts_bins_score", pm.gintervals("1", 0, 20000), iterator=2000)
        r = [np.nan, -2651.3371582031, -2656.0737304688, -2650.1384277344, -2674.7160644531,
             -2653.9035644531, -2683.2404785156, -2662.8173828125, -2683.4096679688,
             -2783.5214843750]
        np.testing.assert_allclose(out["ts_bins_score"].to_numpy(), r, rtol=1e-6)
    finally:
        pm.gtrack_rm("ts_bins_score", force=True)


# ----------------------------------------------------------------------------
# min_obs sparse bins
# ----------------------------------------------------------------------------


def test_min_obs_identifies_sparse_bins(test_vt):
    with pytest.warns(UserWarning, match="bins have fewer than"):
        model = pm.gsynth_train({"expr": test_vt, "breaks": _dense_breaks(21, True)},
                                intervals=pm.gintervals("1", 0, 50000), iterator=200,
                                min_obs=1000)
    assert len(model.sparse_bins) > 0
    assert model.min_obs == 1000
    for b in model.sparse_bins:
        assert np.isnan(model.model_data["cdf"][b - 1]).all()
    non_sparse = sorted(set(range(1, model.total_bins + 1)) - set(model.sparse_bins))
    for b in non_sparse[:3]:
        assert not np.isnan(model.model_data["cdf"][b - 1]).any()


def test_min_obs_0_has_no_sparse_bins(test_vt):
    model = _train({"expr": test_vt, "breaks": _dense_breaks(11, False)},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=0)
    assert len(model.sparse_bins) == 0
    assert model.min_obs == 0
    assert model.sparse_coords is None


def test_sparse_bins_and_coords_match_r():
    # R: same 2D call as test_bin_counts_match_r, min_obs = 1e5
    m = _train({"expr": "dense_track", "breaks": B1},
               {"expr": "subdir.dense_track2", "breaks": B2}, iterator=200, k=2, min_obs=100000)
    assert list(m.sparse_bins) == [3, 4, 6]
    assert list(m.sparse_coords.columns) == ["dense_track", "subdir.dense_track2"]
    assert m.sparse_coords["dense_track"].tolist() == [3, 1, 3]
    assert m.sparse_coords["subdir.dense_track2"].tolist() == [1, 2, 2]


def test_sample_warns_when_using_sparse_bins(test_vt, tmp_path):
    model = _train({"expr": test_vt, "breaks": _dense_breaks(21, True)},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=1000)
    out = str(tmp_path / "s.fa")
    with pytest.warns(UserWarning, match="sparse bins"):
        pm.gsynth_sample(model, out, output_format="fasta",
                         intervals=pm.gintervals("1", 0, 50000), seed=60427)
    with open(out) as f:
        seq = "".join(line.strip() for line in f if not line.startswith(">"))
    assert set(seq) <= set("ACGTN")


def test_sample_sparse_bins_uniform(test_vt):
    model = _train({"expr": test_vt, "breaks": _dense_breaks(11, False)},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=10000000)
    assert len(model.sparse_bins) == model.total_bins
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq = pm.gsynth_sample(model, intervals=pm.gintervals("1", 0, 10000), seed=60427)[0]
    freq = Counter(seq)
    for base in "ACGT":
        assert 0.15 < freq[base] / len(seq) < 0.35


def test_repr_shows_sparse_bins(test_vt):
    model = _train({"expr": test_vt, "breaks": _dense_breaks(11, True)},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=5000)
    assert len(model.sparse_bins) > 0
    assert "Sparse bins" in repr(model)


@pytest.mark.parametrize("compress", [False, True])
def test_save_load_preserves_sparse_bins(test_vt, compress):
    model = _train({"expr": test_vt, "breaks": _dense_breaks(21, True)},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=1000)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gsm")
        pm.gsynth_save(model, path, compress=compress)
        loaded = pm.gsynth_load(path)
    np.testing.assert_array_equal(loaded.sparse_bins, model.sparse_bins)
    assert loaded.min_obs == model.min_obs
    for b in loaded.sparse_bins:
        assert np.isnan(loaded.model_data["cdf"][b - 1]).all()


def test_sample_bin_merge_handles_sparse_bins(test_vt):
    breaks = _dense_breaks(11, True)
    model = _train({"expr": test_vt, "breaks": breaks},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=5000)
    assert len(model.sparse_bins) > 0
    mid = (breaks[4], breaks[5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq = pm.gsynth_sample(
            model, intervals=pm.gintervals("1", 0, 5000), seed=60427,
            bin_merge=[[{"from": (-np.inf, breaks[2]), "to": mid},
                        {"from": (breaks[8], np.inf), "to": mid}]],
        )[0]
    assert len(seq) == 5000
    assert set(seq) <= set("ACGTN")


def test_gsm_roundtrip_2d_sparse_bins(gc_frac):
    model = _train({"expr": "g_frac + c_frac", "breaks": np.arange(0, 1.0001, 0.05).tolist()},
                   {"expr": "g_frac", "breaks": np.arange(0, 0.5001, 0.05).tolist()},
                   intervals=pm.gintervals("1", 0, 50000), iterator=200, min_obs=1000)
    assert len(model.sparse_bins) > 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gsm")
        pm.gsynth_save(model, path)
        loaded = pm.gsynth_load(path)
    np.testing.assert_array_equal(loaded.sparse_bins, model.sparse_bins)
    assert loaded.min_obs == model.min_obs


def test_score_sparse_policy(gc_frac):
    model = _train({"expr": "g_frac + c_frac", "breaks": np.arange(0, 1.0001, 0.05).tolist()},
                   intervals=pm.gintervals("1", 0, 30000), iterator=200, k=2, min_obs=1000000)
    assert len(model.sparse_bins) > 0
    iv = pm.gintervals("1", 1000, 5000)
    try:
        pm.gsynth_score(model, "ts_sparse_na", intervals=iv, resolution=1,
                        sparse_policy="NA", overwrite=True)
        pm.gsynth_score(model, "ts_sparse_uniform", intervals=iv, resolution=1,
                        sparse_policy="uniform", overwrite=True)
        na = pm.gextract("ts_sparse_na", iv, iterator=1)["ts_sparse_na"].to_numpy()
        unif = pm.gextract("ts_sparse_uniform", iv, iterator=1)["ts_sparse_uniform"].to_numpy()
    finally:
        for t in ("ts_sparse_na", "ts_sparse_uniform"):
            pm.gtrack_rm(t, force=True)
    assert np.isnan(na).mean() > 0.5
    sparse_pos = np.flatnonzero(np.isnan(na) & np.isfinite(unif))
    assert len(sparse_pos) > 0
    np.testing.assert_allclose(unif[sparse_pos[:10]], np.log(0.25), atol=1e-6)


def test_cell_merge_to_sparse_target_warns_and_is_uniform():
    # bin 3 (dense_track > 0.185) is sparse; redirect bin 1 to it
    model = _train({"expr": "dense_track", "breaks": B1}, iterator=200, k=1, min_obs=100000)
    assert list(model.sparse_bins) == [3]
    with pytest.warns(UserWarning, match="target cell is itself a sparse bin"):
        seq = pm.gsynth_sample(model, intervals=pm.gintervals("1", 0, 20000), seed=60427,
                               cell_merge=[{"from": [0.05], "to": [0.3]}])[0]
    # uniform, not R's: R keeps the NaN CDF of a sparse target no window hits, and samples all T
    freq = Counter(seq)
    for base in "ACGT":
        assert 0.15 < freq[base] / len(seq) < 0.35


def test_cell_merge_sparse_target_warns_even_when_overridden():
    # as R: checked before a later entry for the same source wins
    model = _train({"expr": "dense_track", "breaks": B1}, iterator=200, k=1, min_obs=100000)
    with pytest.warns(UserWarning, match="target cell is itself a sparse bin"):
        pm.gsynth_sample(model, intervals=pm.gintervals("1", 0, 2000), seed=60427,
                         cell_merge=[{"from": [0.05], "to": [0.3]},
                                     {"from": [0.05], "to": [0.15]}])


def test_min_obs_is_an_int():
    model = _train({"expr": "dense_track", "breaks": B1}, iterator=200, k=1, min_obs=1e5)
    assert model.min_obs == 100000 and isinstance(model.min_obs, int)
    assert "100000 obs" in repr(model)


def test_forbid_kmer_holds_in_sparse_bins():
    # forbid_kmer rebuilds the CDF of every bin, sparse ones too; R still samples bins of fewer
    # than min_obs k-mers uniformly and so emits the forbidden k-mer there
    model = _train({"expr": "dense_track", "breaks": B1}, iterator=200, k=1, min_obs=100000)
    assert list(model.sparse_bins) == [3]
    no_cg = pm.gsynth_forbid_kmer(model, "CG", check=False)
    iv = pm.gintervals("1", 0, 100000)
    out = pm.gextract("dense_track", iv, iterator=200)
    assert (out["dense_track"] > B1[2]).any()  # windows of the sparse bin are sampled
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq = pm.gsynth_sample(no_cg, intervals=iv, seed=60427)[0]
    assert "CG" not in seq[1:]  # the first base is a uniform seed


# ----------------------------------------------------------------------------
# bin_map in memory and in .gsm files
# ----------------------------------------------------------------------------

MERGE = [{"from": (B1[2], B1[3]), "to": (B1[0], B1[1])}]  # bin 3 into bin 1


@pytest.mark.parametrize("compress", [False, True])
def test_bin_merge_survives_save_load(compress, tmp_path):
    m = _train({"expr": "dense_track", "breaks": B1, "bin_merge": MERGE}, iterator=200, k=2)
    path = str(tmp_path / "m.gsm")
    pm.gsynth_save(m, path, compress=compress)
    loaded = pm.gsynth_load(path)
    assert list(loaded.dim_specs[0]["bin_map"]) == [0, 1, 0]
    iv = pm.gintervals("1", 0, 100000)
    assert pm.gsynth_sample(loaded, intervals=iv, seed=60427) == \
        pm.gsynth_sample(m, intervals=iv, seed=60427)


def _saved_meta(model, path):
    pm.gsynth_save(model, path)
    with open(os.path.join(path, "metadata.yaml")) as f:
        return yaml.safe_load(f)


def _rewrite_meta(path, meta):
    with open(os.path.join(path, "metadata.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


def test_gsm_bin_map_is_1_based_as_r(tmp_path):
    m = _train({"expr": "dense_track", "breaks": B1, "bin_merge": MERGE},
               {"expr": "subdir.dense_track2", "breaks": B2, "bin_merge": []}, iterator=200, k=1)
    meta = _saved_meta(m, str(tmp_path / "m.gsm"))
    assert meta["bin_map_base"] == 1
    assert meta["dim_specs"][0]["bin_map"] == [1, 2, 1]
    assert meta["dim_specs"][1]["bin_map"] is None  # identity, as R writes it


@pytest.mark.parametrize("writer, stored", [
    ("pymisha before bin_map_base", [0, 1, 0]),  # has marginal_fallbacks
    ("R", [1, 2, 1]),  # no marginal_fallbacks
    ("pymisha before marginal_fallbacks", [0, 1, 0]),  # a 0 in the map
])
def test_gsm_load_reads_each_writers_bin_map(writer, stored, tmp_path):
    m = _train({"expr": "dense_track", "breaks": B1, "bin_merge": MERGE}, iterator=200, k=1)
    path = str(tmp_path / "m.gsm")
    meta = _saved_meta(m, path)
    del meta["bin_map_base"]
    if writer != "pymisha before bin_map_base":
        del meta["marginal_fallbacks"]
    meta["dim_specs"][0]["bin_map"] = stored
    _rewrite_meta(path, meta)
    assert list(pm.gsynth_load(path).dim_specs[0]["bin_map"]) == [0, 1, 0]


def test_gsm_load_rejects_bad_bin_map(tmp_path):
    m = _train({"expr": "dense_track", "breaks": B1, "bin_merge": MERGE}, iterator=200, k=1)
    path = str(tmp_path / "m.gsm")
    meta = _saved_meta(m, path)
    meta["dim_specs"][0]["bin_map"] = [1, 2, 4]
    _rewrite_meta(path, meta)
    with pytest.raises(ValueError, match="Invalid bin_map"):
        pm.gsynth_load(path)
