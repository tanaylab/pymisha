"""Tests for gseq_onehot and gextract_windows, as R misha's test-gextract-windows.R.

The test database is the example database: its sequence has lowercase runs and N runs.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _no_progress(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "progress", False)


def onehot_ref(seqs):
    """x[n, c, p] = 1 if upper(base p of string n) is channel c."""
    n, w = len(seqs), len(seqs[0])
    res = np.zeros((n, 4, w), dtype=np.uint8)
    for i, s in enumerate(seqs):
        for p, ch in enumerate(s.upper()):
            c = "ACGT".find(ch)
            if c >= 0:
                res[i, c, p] = 1
    return res


def windows_iv(num=60, width=300, seed=3):
    rng = np.random.default_rng(seed)
    sizes = pm.gintervals_all()
    chroms = rng.choice(sizes["chrom"].astype(str).to_numpy(), num)
    lens = sizes.set_index(sizes["chrom"].astype(str))["end"].loc[chroms].to_numpy()
    starts = np.floor(rng.random(num) * (lens - width)).astype(np.int64)
    return pd.DataFrame({"chrom": chroms, "start": starts, "end": starts + width})


def test_onehot_intervals_match_gseq_extract():
    iv = windows_iv()
    seqs = pm.gseq_extract(iv)
    assert any(any(c in s for c in "acgt") for s in seqs)
    assert any("N" in s for s in seqs)
    x = pm.gseq_onehot(iv)
    assert x.dtype == np.uint8
    assert x.shape == (len(iv), 4, 300)
    assert x.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(x, onehot_ref(seqs))
    is_n = np.array([[ch == "N" for ch in s] for s in seqs])
    np.testing.assert_array_equal(x.sum(axis=1) == 0, is_n)
    for dt in (np.float32, np.float64, np.int32, np.int8, bool):
        xd = pm.gseq_onehot(iv, dtype=dt)
        assert xd.dtype == np.dtype(dt)
        np.testing.assert_array_equal(xd, onehot_ref(seqs).astype(dt))


def test_onehot_strand_minus_is_reverse_complement():
    iv = windows_iv(40, 101)
    iv["strand"] = np.tile([1, -1], 20)
    seqs = pm.gseq_extract(iv)  # reverse complements the strand -1 rows
    x = pm.gseq_onehot(iv)
    np.testing.assert_array_equal(x, onehot_ref(seqs))
    fwd = pm.gseq_onehot(iv[["chrom", "start", "end"]])
    minus = iv["strand"].to_numpy() == -1
    np.testing.assert_array_equal(x[minus], fwd[minus][:, ::-1, ::-1])
    np.testing.assert_array_equal(x[~minus], fwd[~minus])
    iv["strand"] = np.where(minus, "-", "+")
    np.testing.assert_array_equal(pm.gseq_onehot(iv), x)


def test_onehot_strings():
    seqs = ["ACGTNRYacgtn-.", "tttTTTgggGGa*a"]
    np.testing.assert_array_equal(pm.gseq_onehot(seqs), onehot_ref(seqs))
    np.testing.assert_array_equal(pm.gseq_onehot([s.encode() for s in seqs]), onehot_ref(seqs))
    with pytest.raises(Exception, match="same length"):
        pm.gseq_onehot(["ACGT", "ACG"])


def test_onehot_pad():
    len1 = int(pm.gintervals_all()["end"].iloc[0])
    iv = pd.DataFrame(
        {
            "chrom": "chr1",
            "start": [-50, len1 - 70, -10, len1 + 5],
            "end": [50, len1 + 30, 90, len1 + 105],
            "strand": [1, -1, -1, 1],
        }
    )
    with pytest.raises(Exception, match="runs past"):
        pm.gseq_onehot(iv)
    np.testing.assert_array_equal(pm.gseq_onehot(iv, pad=True), onehot_ref(pm.gseq_extract(iv, pad=True)))


def test_onehot_checks():
    with pytest.raises(ValueError, match="same positive width"):
        pm.gseq_onehot(pm.gintervals(["1", "1"], [0, 100], [10, 120]))
    with pytest.raises(ValueError, match="dtype"):
        pm.gseq_onehot(pm.gintervals("1", 0, 10), dtype=np.complex64)
    with pytest.raises(Exception):
        pm.gseq_onehot(pd.DataFrame({"chrom": ["chrZ"], "start": [0], "end": [10]}))


def test_windows_centre_and_match_onehot_and_matrix():
    iv = windows_iv(50, 1, seed=11)
    rng = np.random.default_rng(1)
    iv["end"] = iv["start"] + rng.integers(1, 401, len(iv))
    iv["name"] = [f"r{i}" for i in range(len(iv))]
    w = pm.gextract_windows(
        iv, tracks={"dense_track": "dense_track", "sp": "sparse_track"}, width=301, signal_width=200, binsize=20
    )
    kept = w.intervals["index"].to_numpy()
    assert len(kept) > 30
    starts, ends = iv["start"].to_numpy()[kept], iv["end"].to_numpy()[kept]
    mids = starts + (ends - starts) // 2
    np.testing.assert_array_equal(w.intervals["start"], mids - 150)
    np.testing.assert_array_equal(w.intervals["end"], mids + 151)
    np.testing.assert_array_equal(w.intervals["signal_start"], mids - 100)
    np.testing.assert_array_equal(w.intervals["signal_end"], mids + 100)
    assert list(w.intervals["name"]) == list(iv["name"].to_numpy()[kept])
    assert sorted(list(kept) + list(w.dropped["index"])) == list(range(len(iv)))

    np.testing.assert_array_equal(w.x, pm.gseq_onehot(w.intervals[["chrom", "start", "end"]]))
    sig = pd.DataFrame(
        {"chrom": w.intervals["chrom"], "start": w.intervals["signal_start"], "end": w.intervals["signal_end"]}
    )
    m = pm.gextract_matrix(["dense_track", "sparse_track"], sig, iterator=20)
    assert w.y.shape == (len(kept), 2, 10)
    assert w.y.dtype == np.float32 and w.y.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(w.y, m.transpose(0, 2, 1).astype(np.float32))

    w64 = pm.gextract_windows(
        iv, tracks=["dense_track", "sparse_track"], width=301, signal_width=200, binsize=20, signal_dtype=np.float64
    )
    np.testing.assert_array_equal(w64.y, m.transpose(0, 2, 1))

    w2 = pm.gextract_windows(iv, width=301)
    assert w2.y is None
    np.testing.assert_array_equal(w2.x, pm.gseq_onehot(w2.intervals[["chrom", "start", "end"]]))
    assert pm.gextract_windows(iv, tracks="dense_track", width=301, sequence=False).x is None


def test_windows_off_chrom_and_jitter():
    len1 = int(pm.gintervals_all()["end"].iloc[0])
    s = np.array([100, 1000, len1 - 1000, len1 - 140, 5000])
    iv = pd.DataFrame({"chrom": "chr1", "start": s, "end": s + 1})
    w = pm.gextract_windows(iv, tracks="dense_track", width=200, signal_width=100, jitter=50)
    assert list(w.intervals["index"]) == [1, 2, 4]
    assert list(w.dropped["index"]) == [0, 3]
    assert list(w.dropped["reason"]) == ["off_chrom", "off_chrom"]
    assert w.x.shape == (3, 4, 300)
    assert w.y.shape == (3, 1, 200)
    np.testing.assert_array_equal(w.intervals["start"], s[[1, 2, 4]] - 150)
    np.testing.assert_array_equal(w.intervals["signal_start"], s[[1, 2, 4]] - 100)

    w = pm.gextract_windows(iv, tracks="dense_track", width=10, signal_width=300)
    assert list(w.dropped["index"]) == [0, 3]
    assert len(pm.gextract_windows(iv, width=10).dropped) == 0


def test_windows_strand():
    iv = windows_iv(30, 1, seed=5)
    iv["strand"] = np.tile([-1, 1, 0], 10)
    w = pm.gextract_windows(iv, tracks="dense_track", width=100, binsize=10)
    wf = pm.gextract_windows(iv, tracks="dense_track", width=100, binsize=10, use_strand=False)
    neg = w.intervals["strand"].to_numpy() == -1
    assert neg.any()
    np.testing.assert_array_equal(w.x[~neg], wf.x[~neg])
    np.testing.assert_array_equal(w.x[neg], wf.x[neg][:, ::-1, ::-1])
    np.testing.assert_array_equal(w.y[neg], wf.y[neg][:, :, ::-1])
    np.testing.assert_array_equal(w.y[~neg], wf.y[~neg])
    seqs = pm.gseq_extract(w.intervals[["chrom", "start", "end", "strand"]])
    np.testing.assert_array_equal(w.x, onehot_ref(seqs))


def test_windows_exclude():
    s = np.array([1000, 2000, 3000, 4000])
    iv = pd.DataFrame({"chrom": "chr1", "start": s, "end": s + 1})
    excl = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chrNotThere", "chr1"],
            "start": [1050, 2049, 2900, 3960, 3990],
            "end": [1100, 2060, 2950, 3970, 3990],
        }
    )
    w = pm.gextract_windows(iv, width=100, exclude=excl)
    assert list(w.dropped["index"]) == [1]
    assert list(w.dropped["reason"]) == ["excluded"]
    excl2 = pd.DataFrame({"chrom": "chr1", "start": [10, 500], "end": [3500, 600]})
    assert list(pm.gextract_windows(iv, width=100, exclude=excl2).dropped["index"]) == [0, 1, 2]
    iv2 = pd.concat([pd.DataFrame({"chrom": ["chr1"], "start": [10], "end": [11]}), iv], ignore_index=True)
    w = pm.gextract_windows(iv2, width=100, exclude=pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]}))
    assert list(w.dropped["reason"]) == ["off_chrom"]


def test_windows_count_filters():
    iv = windows_iv(40, 1, seed=8)
    tracks = {"d": "dense_track", "s": "sparse_track"}
    w = pm.gextract_windows(iv, tracks=tracks, width=200, binsize=10)
    counts = np.nansum(w.y[:, 1, :].astype(np.float64), axis=1)
    lo, hi = np.quantile(counts, 0.25), np.quantile(counts, 0.75)
    wf = pm.gextract_windows(iv, tracks=tracks, width=200, binsize=10, min_counts=lo, max_counts=hi, count_track="s")
    keep = (counts >= lo) & (counts <= hi)
    np.testing.assert_array_equal(wf.intervals["index"], w.intervals["index"].to_numpy()[keep])
    np.testing.assert_array_equal(wf.x, w.x[keep])
    np.testing.assert_array_equal(wf.y, w.y[keep])
    d = wf.dropped[wf.dropped["reason"] != "off_chrom"]
    np.testing.assert_array_equal(d["index"], w.intervals["index"].to_numpy()[~keep])
    assert list(d["reason"]) == list(np.where(counts[~keep] < lo, "min_counts", "max_counts"))
    w1 = pm.gextract_windows(iv, tracks=tracks, width=200, binsize=10, min_counts=lo, count_track=1)
    np.testing.assert_array_equal(w1.intervals["index"], w.intervals["index"].to_numpy()[counts >= lo])
    with pytest.raises(ValueError, match="tracks"):
        pm.gextract_windows(iv, width=200, min_counts=1)
    with pytest.raises(ValueError, match="count_track"):
        pm.gextract_windows(iv, tracks="dense_track", width=200, min_counts=1, count_track=1)


def test_windows_sets():
    pos = pd.DataFrame({"chrom": "chr1", "start": [1000, 2000], "end": [1001, 2001], "score": [1.0, 2.0]})
    neg = pd.DataFrame({"chrom": "chr2", "start": [5000, 10, 7000], "end": [5001, 11, 7001], "score": 0.0})
    w = pm.gextract_windows({"pos": pos, "neg": neg}, width=100)
    assert list(w.intervals["set"]) == ["pos", "pos", "neg", "neg"]
    assert list(w.intervals["index"]) == [0, 1, 2, 4]
    assert list(w.intervals["score"]) == [1.0, 2.0, 0.0, 0.0]
    assert list(w.dropped["index"]) == [3]
    assert list(pm.gextract_windows([pos, neg], width=100).intervals["set"]) == [0, 0, 1, 1]


def test_windows_checks_and_empty():
    iv = pm.gintervals("1", 1000, 1001)
    with pytest.raises(ValueError, match="Usage"):
        pm.gextract_windows(iv)
    with pytest.raises(ValueError, match="width"):
        pm.gextract_windows(iv, width=0)
    with pytest.raises(ValueError, match="binsize"):
        pm.gextract_windows(iv, tracks="dense_track", width=100, binsize=30)
    w = pm.gextract_windows(pm.gintervals("1", 10, 11), tracks="dense_track", width=100)
    assert w.x.shape == (0, 4, 100)
    assert w.y.shape == (0, 1, 100)
    assert len(w.intervals) == 0


def test_windows_vtracks_and_expressions():
    pm.gvtrack_create("dsum_w", "dense_track", func="sum")
    try:
        iv = windows_iv(20, 1, seed=2)
        w = pm.gextract_windows(iv, tracks=["dsum_w", "dense_track * 2"], width=100, binsize=25)
        m = pm.gextract_matrix(["dsum_w", "dense_track * 2"], w.intervals[["chrom", "start", "end"]], iterator=25)
        np.testing.assert_array_equal(w.y, m.transpose(0, 2, 1).astype(np.float32))
    finally:
        pm.gvtrack_rm("dsum_w")


def test_windows_expression_variables():
    scale = 3  # noqa: F841 - read by the track expression
    iv = windows_iv(10, 1, seed=4)
    w = pm.gextract_windows(iv, tracks="dense_track * scale", width=100, binsize=10)
    w1 = pm.gextract_windows(iv, tracks="dense_track", width=100, binsize=10)
    np.testing.assert_allclose(w.y, w1.y * 3, rtol=1e-6)
    w2 = pm.gextract_windows(iv, tracks="dense_track * k", width=100, binsize=10, vars={"k": 3})
    np.testing.assert_array_equal(w2.y, w.y)
