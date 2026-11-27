"""gintervals_promoters, gintervals_expand and gintervals_remove_overlaps, as R misha's
gintervals.promoters / .expand / .remove_overlaps (tests/testthat/test-gintervals.promoters.R)."""

import pandas as pd
import pytest

import pymisha as pm


def _chr1_end():
    allg = pm.gintervals_all()
    return int(allg.loc[allg["chrom"].astype(str).isin(["1", "chr1"]), "end"].iloc[0])


def test_promoters_window_from_5prime_base_per_strand():
    tss = pd.DataFrame({
        "chrom": "1", "start": [10000, 20000, 30000], "end": [10001, 20001, 30001],
        "strand": [1, -1, 0], "name": ["p", "m", "both"],
    })
    res = pm.gintervals_promoters(tss, upstream=500, downstream=50)
    # +: [tss - 500, tss + 50); -: 5' base 20000, so [20001 - 50, 20001 + 500); 0: both
    assert res["start"].tolist() == [9500, 19951, 29500]
    assert res["end"].tolist() == [10050, 20501, 30501]
    assert res["strand"].tolist() == [1, -1, 0]
    assert res["name"].tolist() == ["p", "m", "both"]


def test_promoters_use_5prime_end_of_longer_intervals():
    tx = pd.DataFrame({"chrom": "1", "start": [1000, 1000], "end": [5000, 5000], "strand": [1, -1]})
    res = pm.gintervals_promoters(tx, upstream=100, downstream=10)
    assert res["start"].tolist() == [900, 4990]
    assert res["end"].tolist() == [1010, 5100]


def test_promoters_clip_at_chromosome_ends():
    end = _chr1_end()
    tss = pd.DataFrame({"chrom": "1", "start": [100, end - 1], "end": [101, end], "strand": [1, -1]})
    res = pm.gintervals_promoters(tss, upstream=500, downstream=50)
    assert res["start"].tolist() == [0, end - 50]
    assert res["end"].tolist() == [150, end]


def test_promoters_require_strand_1_minus1_or_0():
    with pytest.raises(ValueError, match="strand column"):
        pm.gintervals_promoters(pm.gintervals("1", 1000, 1001))
    bad = pd.DataFrame({"chrom": "1", "start": [1000], "end": [1001], "strand": [2]})
    with pytest.raises(ValueError, match="strand must be"):
        pm.gintervals_promoters(bad)
    bad["strand"] = float("nan")
    with pytest.raises(ValueError, match="strand must be"):
        pm.gintervals_promoters(bad)


def test_expand_adds_bp_on_both_sides_and_clips():
    iv = pm.gintervals("1", [1000, 5000], [2000, 6000])
    res = pm.gintervals_expand(iv, 100)
    assert res["start"].tolist() == [900, 4900]
    assert res["end"].tolist() == [2100, 6100]

    res = pm.gintervals_expand(iv, [10, 2000])
    assert res["start"].tolist() == [990, 3000]
    assert res["end"].tolist() == [2010, 8000]

    end = _chr1_end()
    res = pm.gintervals_expand(pm.gintervals("1", [50, end - 50], [100, end]), 100)
    assert res["start"].tolist() == [0, end - 150]
    assert res["end"].tolist() == [200, end]


INTERVS = pd.DataFrame({
    "chrom": "1", "start": [11000, 100, 10000, 10500], "end": [12000, 200, 13000, 10600],
    "data": [10, 20, 30, 40],
})


def test_remove_overlaps_keeps_one_per_group_in_row_order():
    assert pm.gintervals_remove_overlaps(INTERVS)["data"].tolist() == [10, 20]
    assert pm.gintervals_remove_overlaps(INTERVS, select="last")["data"].tolist() == [20, 40]

    picked = [pm.gintervals_remove_overlaps(INTERVS, select="random", seed=s)["data"].tolist()
              for s in range(20)]
    assert all(len(p) == 2 and 20 in p for p in picked)
    assert {v for p in picked for v in p} == {10, 20, 30, 40}
    assert picked[3] == pm.gintervals_remove_overlaps(INTERVS, select="random", seed=3)["data"].tolist()

    with pytest.raises(ValueError, match="select"):
        pm.gintervals_remove_overlaps(INTERVS, select="middle")


def test_remove_overlaps_follows_unify_touching_intervals():
    touching = pd.DataFrame({"chrom": "1", "start": [0, 100], "end": [100, 200]})
    assert len(pm.gintervals_remove_overlaps(touching)) == 1
    assert len(pm.gintervals_remove_overlaps(touching, unify_touching_intervals=False)) == 2
