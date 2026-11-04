"""Port of misha tests/testthat/test-gintervals.liftover-agg.R (value_col aggregation).

R's target chromosomes (chrA, chrB, ...) are chr1 of the test DB here, and
intervalIDs are 1-based, as in R.
"""

import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

SRC = "chrsource1"


def _chain(tmp_path, entries, src_size=400, name="c.chain"):
    """entries: (src_start, src_end, tgt_start, tgt_end, chain_id[, score]), one block each."""
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as f:
        for e in entries:
            s0, s1, t0, t1, cid = e[:5]
            score = e[5] if len(e) > 5 else 1000
            f.write(f"chain {score} {SRC} {src_size} + {s0} {s1} chr1 500000 + {t0} {t1} {cid}\n{s1 - s0}\n\n")
    return path


def _src(starts, ends, col="value", vals=None):
    return pd.DataFrame({"chrom": [SRC] * len(starts), "start": starts, "end": ends, col: vals})


@pytest.fixture
def overlap(tmp_path):
    """Two overlapping sources [0,10)=1 and [5,15)=3 through a 1:1 chain."""
    return _src([0, 5], [10, 15], vals=[1.0, 3.0]), _chain(tmp_path, [(0, 20, 0, 20, 1)])


def test_multi_target_aggregation_policies(tmp_path):
    src = _src([0, 10, 20], [10, 20, 30], vals=[1.0, 2.0, 3.0])
    src_na = _src([0, 10, 20], [10, 20, 30], vals=[1.0, np.nan, 3.0])
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1), (10, 20, 3, 13, 2), (20, 30, 7, 17, 3)])

    def lift(intervals=src, **kw):
        return pm.gintervals_liftover(intervals, chain, value_col="value", **kw)

    res = lift()
    assert "value" in res.columns
    assert len(res) == 3
    assert res["value"].isin([1, 2, 3]).all()
    assert lift(multi_target_agg="sum")["value"].isin([1, 2, 3]).all()
    for agg in ["min", "max", "median", "count", "first", "last",
                "max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac"]:
        assert len(lift(multi_target_agg=agg)) == 3, agg
    assert len(lift(multi_target_agg="nth", params=2)) == 3
    assert len(lift(multi_target_agg="nth", params={"n": 3})) == 3

    assert "value" in lift(src_na, na_rm=True).columns
    assert "value" in lift(src_na, na_rm=False).columns
    assert "value" in lift(src_na, na_rm=True, min_n=3).columns
    assert "value" in lift(src_na, na_rm=True, min_n=2).columns


@pytest.mark.parametrize(
    "agg,want",
    [("mean", [1, 2, 3]), ("sum", [1, 4, 3]), ("count", [1, 2, 1]), ("max.coverage_len", [1, 3, 3])],
)
def test_concrete_values_on_overlapping_sources(overlap, agg, want):
    src, chain = overlap
    res = pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg=agg)
    assert res["start"].tolist() == [0, 5, 10]
    assert res["end"].tolist() == [5, 10, 15]
    np.testing.assert_array_equal(res["value"], want)


def test_na_handling_on_overlapping_sources(overlap):
    src, chain = overlap
    src = src.assign(value=[np.nan, 3.0])
    res = pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg="mean", na_rm=True)
    assert np.isnan(res["value"][0])
    assert res["value"][1] == 3
    assert res["value"][2] == 3
    res = pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg="mean", na_rm=False)
    assert np.isnan(res["value"][1])


def test_min_n_gates_overlapping_sources(overlap):
    src, chain = overlap
    res = pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg="mean", min_n=2)
    assert np.isnan(res["value"][0]) and res["value"][1] == 2 and np.isnan(res["value"][2])


def test_aggregation_preserves_interval_id_and_chain_id(tmp_path):
    src = _src([0, 10, 20], [10, 20, 30], col="score", vals=[10.0, 20.0, 30.0])
    chain = _chain(tmp_path, [(0, 10, 100, 110, 1), (10, 20, 110, 120, 2), (20, 30, 120, 130, 3)], src_size=200)
    res = pm.gintervals_liftover(src, chain, value_col="score", multi_target_agg="sum")
    assert {"intervalID", "chain_id", "score"} <= set(res.columns)
    assert len(res) == 3
    assert sorted(res["intervalID"]) == [1, 2, 3]
    assert sorted(res["chain_id"]) == [1, 2, 3]
    assert sorted(res["score"]) == [10, 20, 30]


def test_aggregation_with_integer_values(tmp_path):
    src = _src([0, 10, 20], [10, 20, 30], col="count", vals=np.array([5, 10, 15], dtype=np.int32))
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1), (10, 20, 0, 10, 2), (20, 30, 0, 10, 3)], src_size=100)
    res = pm.gintervals_liftover(src, chain, value_col="count", multi_target_agg="mean")
    assert "count" in res.columns
    assert res["count"].isin([5, 10, 15]).all()


def test_without_value_col_works_as_before(tmp_path):
    src = pd.DataFrame({"chrom": [SRC] * 2, "start": [0, 10], "end": [10, 20], "extra_col": ["a", "b"]})
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1), (10, 20, 5, 15, 2)], src_size=100)
    res = pm.gintervals_liftover(src, chain)
    assert {"intervalID", "chain_id"} <= set(res.columns)
    assert "value" not in res.columns
    assert len(res) == 2


def test_validates_value_col(tmp_path):
    src = _src([0], [10], col="score", vals=[5.0])
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1)], src_size=100)
    with pytest.raises(ValueError, match="value_col 'nonexistent' not found in intervals"):
        pm.gintervals_liftover(src, chain, value_col="nonexistent")
    for bad in (123, ["score", "other"]):
        with pytest.raises(ValueError, match="value_col must be a single character string"):
            pm.gintervals_liftover(src, chain, value_col=bad)


def test_nth_validates_params(tmp_path):
    src = _src([0], [10], vals=[1.0])
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1)], src_size=100)
    with pytest.raises(ValueError, match="params must be supplied for 'nth' aggregation"):
        pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg="nth")
    with pytest.raises(ValueError, match="params list must contain an element 'n' for 'nth'"):
        pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg="nth", params={})


def test_all_na_values(tmp_path):
    src = _src([0, 10, 20], [10, 20, 30], vals=[np.nan] * 3)
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1), (10, 20, 0, 10, 2), (20, 30, 0, 10, 3)], src_size=100)
    for agg in ["mean", "sum", "min", "max", "median", "first", "last", "max.coverage_len"]:
        res = pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg=agg, na_rm=True)
        assert "value" in res.columns, agg
        assert res["value"].isna().all(), agg
    res = pm.gintervals_liftover(src, chain, value_col="value", multi_target_agg="count", na_rm=True)
    assert (res["value"].isna() | (res["value"] == 0)).all()


def test_canonic_mode_preserves_values(tmp_path):
    src = _src([0, 10], [10, 20], col="score", vals=[100.0, 100.0])
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1), (10, 20, 10, 20, 1)], src_size=200)
    res = pm.gintervals_liftover(src, chain, value_col="score", multi_target_agg="mean", canonic=True)
    assert "score" in res.columns
    assert (res["score"] == 100).all()


def test_multiple_value_types(tmp_path):
    src = _src([0], [10], col="val", vals=[3.14])
    chain = _chain(tmp_path, [(0, 10, 0, 10, 1)], src_size=100)
    res = pm.gintervals_liftover(src, chain, value_col="val", multi_target_agg="mean")
    assert res["val"][0] == pytest.approx(3.14, abs=1e-6)


@pytest.mark.parametrize("agg,want", [("max", 0.9), ("mean", 0.875), ("sum", 1.75), ("count", 2)])
def test_aggregates_across_chain_ids_mapping_to_same_target(tmp_path, agg, want):
    src = _src([0, 100], [100, 200], vals=[0.9, 0.85])
    chain = _chain(tmp_path, [(0, 100, 50, 150, 1, 100), (100, 200, 50, 150, 2, 95)], src_size=300)
    res = pm.gintervals_liftover(src, chain, src_overlap_policy="keep", tgt_overlap_policy="agg",
                                 value_col="value", multi_target_agg=agg)
    assert len(res) == 1
    assert res["value"][0] == pytest.approx(want, abs=1e-6)
    assert "intervalID" not in res.columns
    assert "chain_id" not in res.columns


def _raw_chain(tmp_path, lines):
    path = os.path.join(str(tmp_path), "raw.chain")
    with open(path, "w") as f:
        for hdr, size in lines:
            f.write(f"chain {hdr}\n{size}\n\n")
    return path


def test_carried_hint_does_not_skip_wider_earlier_source_chain(tmp_path):
    path = _raw_chain(tmp_path, [
        (f"1000 {SRC} 400 + 0 100 chr1 500000 + 0 100 1", 100),     # A (wide)
        (f"1000 {SRC} 400 + 10 20 chr1 500000 + 200 210 2", 10),    # B (decoy)
        (f"1000 {SRC} 400 + 30 40 chr1 500000 + 300 310 3", 10),    # C
    ])
    ch = pm.gintervals_load_chain(path, src_overlap_policy="keep", tgt_overlap_policy="keep")
    q = pd.DataFrame({"chrom": [SRC] * 2, "start": [5, 35], "end": [40, 38]})
    batch = pm.gintervals_liftover(q, ch, include_metadata=True)
    solo2 = pm.gintervals_liftover(q.iloc[[1]].reset_index(drop=True), ch, include_metadata=True)
    b2 = batch[batch["intervalID"] == 2]
    assert len(b2) == len(solo2) == 2
    assert set(b2["chain_id"]) == {1, 3}
    assert set(b2["start"]) == {35, 305}


def test_auto_score_truncated_minus_strand_chain_keeps_reversed_source_coords(tmp_path):
    # minus target 499300..499400 over chr1's 500000 bp is forward 600..700 (R: chrT of 1000 bp, 300..400)
    path = _raw_chain(tmp_path, [
        ("100 chrS 100000 + 100 200 chr1 500000 - 499300 499400 1", 100),
        ("900 chrS 100000 + 2000 2050 chr1 500000 + 650 700 2", 50),
    ])
    ch = pm.gintervals_load_chain(path, tgt_overlap_policy="auto_score")
    s1 = ch[ch["chain_id"] == 1]
    assert len(s1) == 1
    assert (s1["start"].item(), s1["end"].item()) == (600, 650)
    assert (s1["startsrc"].item(), s1["endsrc"].item()) == (150, 200)


def test_narrow_interval_at_its_leftmost_chain_not_dropped_after_wide_one(tmp_path):
    path = _raw_chain(tmp_path, [
        ("1000 src 5000 + 100 200 chr1 500000 + 1000 1100 1", 100),
        ("1000 src 5000 + 400 600 chr1 500000 + 2000 2200 2", 200),
    ])
    ch = pm.gintervals_load_chain(path)
    inp = pd.DataFrame({"chrom": ["src"] * 2, "start": [0, 100], "end": [1000, 200]})
    res = pm.gintervals_liftover(inp, ch, include_metadata=True)
    r2 = res[res["intervalID"] == 2]
    assert len(r2) == 1
    assert (r2["start"].item(), r2["end"].item()) == (1000, 1100)
