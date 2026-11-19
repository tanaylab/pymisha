"""Tests for gextract_matrix, ported from R misha's test-gextract-matrix.R."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _no_progress(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "progress", False)


def reference(exprs, intervals, iterator=1):
    """gextract output reshaped to intervals x bins x expressions, ordered by intervalID and start."""
    exprs = [exprs] if isinstance(exprs, str) else exprs
    names = [f"v{i}" for i in range(len(exprs))]
    r = pm.gextract(exprs, intervals, iterator=iterator, colnames=names)
    r = r.sort_values(["intervalID", "start"], kind="stable")
    num_bins = (int(intervals["end"].iloc[0]) - int(intervals["start"].iloc[0])) // iterator
    assert len(r) == len(intervals) * num_bins
    return np.stack([r[n].to_numpy().reshape(len(intervals), num_bins) for n in names], axis=2)


def windows(width, num=40, align=1):
    """Unsorted windows on several chromosomes, with duplicated and overlapping rows."""
    rng = np.random.default_rng(17)
    chroms = rng.choice(["chr1", "chr2", "chrX"], num)
    starts = rng.integers(0, 2001, num) * align
    intervs = pd.DataFrame({"chrom": chroms, "start": starts, "end": starts + width})
    dup = intervs.iloc[[2, 2, 6]]
    ovl = intervs.iloc[[4, 8]].copy()
    ovl["start"] += align
    ovl["end"] += align
    intervs = pd.concat([intervs, dup, ovl])
    return intervs.iloc[rng.permutation(len(intervs))].reset_index(drop=True)


def test_matches_gextract_on_unsorted_duplicated_overlapping_rows(monkeypatch):
    intervs = windows(200)
    ref = reference("dense_track", intervs)
    m = pm.gextract_matrix("dense_track", intervs)
    assert m.shape == (len(intervs), 200)
    assert m.dtype == np.float64
    np.testing.assert_array_equal(m, ref[:, :, 0])
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    np.testing.assert_array_equal(pm.gextract_matrix("dense_track", intervs), ref[:, :, 0])


def bin_reference(exprs, intervals, iterator):
    """Values of every bin of every window, from one gextract call per window with
    the window's bins, built here, as the iterator (the bins of one window touch
    but do not overlap, so gextract reports each one as it is)."""
    names = [f"v{i}" for i in range(len(exprs))]
    num_bins = (int(intervals["end"].iloc[0]) - int(intervals["start"].iloc[0])) // iterator
    res = np.full((len(intervals), num_bins, len(exprs)), np.nan)
    for i, (chrom, start) in enumerate(zip(intervals["chrom"], intervals["start"], strict=True)):
        bin_starts = start + np.arange(num_bins) * iterator
        bins = pd.DataFrame({"chrom": chrom, "start": bin_starts, "end": bin_starts + iterator})
        r = pm.gextract(exprs, intervals.iloc[[i]][["chrom", "start", "end"]], iterator=bins, colnames=names)
        r = r.sort_values("start")
        np.testing.assert_array_equal(r["start"].to_numpy(), bin_starts)
        np.testing.assert_array_equal(r["end"].to_numpy(), bin_starts + iterator)
        for j, n in enumerate(names):
            res[i, :, j] = r[n].to_numpy()
    return res


@pytest.mark.parametrize("k", [7, 10, 50])
def test_bins_counted_from_each_interval_start(monkeypatch, k):
    src = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2", "X"], 0, 3000))
    pm.gvtrack_create("gxmat_cov", src, func="coverage")
    try:
        exprs = ["dense_track", "sparse_track", "gxmat_cov"]
        intervs = windows(350)
        assert (intervs["start"] % k != 0).any()
        num_bins = 350 // k
        ref = bin_reference(exprs, intervs, k)
        a = pm.gextract_matrix(exprs, intervs, iterator=k)
        assert a.shape == (len(intervs), num_bins, 3)
        np.testing.assert_array_equal(a, ref)
        np.testing.assert_array_equal(pm.gextract_matrix("sparse_track", intervs, iterator=k), ref[:, :, 1])
        assert np.isnan(a[:, :, 1]).any() and (~np.isnan(a[:, :, 1])).any()
        cov = a[:, :, 2]
        assert ((cov >= 0) & (cov <= 1)).all() and ((cov > 0) & (cov < 1)).any()

        # a bin of the dense track gets the plain mean of the track bins it overlaps
        dense = {
            c: pm.gextract("dense_track", pm.gintervals(c, 0, 3000))["dense_track"].to_numpy()
            for c in ("chr1", "chr2", "chrX")
        }
        bin_starts = intervs["start"].to_numpy()[:, None] + np.arange(num_bins) * k
        chroms = np.repeat(intervs["chrom"].to_numpy()[:, None], num_bins, axis=1)
        direct = np.array([np.nanmean(dense[c][s // 50:(s + k - 1) // 50 + 1])
                           for c, s in zip(chroms.ravel(), bin_starts.ravel(), strict=True)]).reshape(bin_starts.shape)
        np.testing.assert_allclose(a[:, :, 0], direct, rtol=1e-6)

        # three intervals per gextract call
        with monkeypatch.context() as mp:
            mp.setitem(pm.CONFIG, "max_data_size", 3 * num_bins)
            np.testing.assert_array_equal(pm.gextract_matrix(exprs, intervs, iterator=k), ref)
        monkeypatch.setitem(pm.CONFIG, "multitasking", False)
        np.testing.assert_array_equal(pm.gextract_matrix(exprs, intervs, iterator=k), ref)
    finally:
        pm.gvtrack_rm("gxmat_cov")


@pytest.mark.parametrize("k", [10, 50])
def test_numeric_iterator_bins_when_starts_on_grid(k):
    exprs = ["dense_track", "sparse_track"]
    intervs = windows(500, align=k)
    ref = reference(exprs, intervs, iterator=k)
    a = pm.gextract_matrix(exprs, intervs, iterator=k)
    assert a.shape == (len(intervs), 500 // k, 2)
    np.testing.assert_array_equal(a, ref)
    # one interval off the grid: all rows go through the interval-relative bins
    mixed = pd.concat([intervs, pd.DataFrame({"chrom": ["chr1"], "start": [3], "end": [503]})], ignore_index=True)
    np.testing.assert_array_equal(pm.gextract_matrix(exprs, mixed, iterator=k)[: len(intervs)], ref)


def test_sparse_track_keeps_nan():
    intervs = windows(300)
    m = pm.gextract_matrix("sparse_track", intervs)
    assert np.isnan(m).any()
    assert (~np.isnan(m)).any()
    np.testing.assert_array_equal(m, reference("sparse_track", intervs)[:, :, 0])


def test_coverage_mask_from_intervals_vtrack():
    src = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2", "X"]))
    pm.gvtrack_create("gxmat_cov", src, func="coverage")
    try:
        intervs = windows(250)
        m = pm.gextract_matrix("gxmat_cov", intervs)
        assert set(np.unique(m)) == {0.0, 1.0}
        np.testing.assert_array_equal(m, reference("gxmat_cov", intervs)[:, :, 0])

        intervs10 = windows(250, align=10)
        m10 = pm.gextract_matrix("gxmat_cov", intervs10, iterator=10)
        np.testing.assert_array_equal(m10, reference("gxmat_cov", intervs10, iterator=10)[:, :, 0])
    finally:
        pm.gvtrack_rm("gxmat_cov")


def test_several_expressions(monkeypatch):
    exprs = ["dense_track", "sparse_track", "dense_track * 2"]
    intervs = windows(100)
    ref = reference(exprs, intervs)
    a = pm.gextract_matrix(exprs, intervs)
    assert a.shape == (len(intervs), 100, 3)
    np.testing.assert_array_equal(a, ref)
    np.testing.assert_array_equal(pm.gextract_matrix(exprs, intervs, colnames=["a", "b", "c"]), ref)
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    np.testing.assert_array_equal(pm.gextract_matrix(exprs, intervs), ref)


def test_same_result_in_many_chunks(monkeypatch):
    exprs = ["dense_track", "sparse_track"]
    intervs = windows(100)
    ref = reference(exprs, intervs)
    # 3 intervals per gextract call
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 350)
    np.testing.assert_array_equal(pm.gextract_matrix(exprs, intervs), ref)


def test_caller_variables():
    scale = 3.0
    intervs = windows(100)
    m = pm.gextract_matrix("dense_track * scale", intervs)
    np.testing.assert_array_equal(m, reference("dense_track", intervs)[:, :, 0] * scale)


def test_ignores_strand_and_accepts_set_name():
    intervs = windows(100)
    m = pm.gextract_matrix("dense_track", intervs)
    stranded = intervs.assign(strand=-1)
    np.testing.assert_array_equal(pm.gextract_matrix("dense_track", stranded), m)

    pm.gintervals_save(intervs, "test.gxmat_windows")
    try:
        saved = pm.gintervals_load("test.gxmat_windows")
        np.testing.assert_array_equal(
            pm.gextract_matrix("dense_track", "test.gxmat_windows"),
            pm.gextract_matrix("dense_track", saved),
        )
    finally:
        pm.gintervals_rm("test.gxmat_windows", force=True)


def test_rejects_invalid_input():
    intervs = pm.gintervals(["1", "2"], [100, 200], [200, 300])
    with pytest.raises(ValueError, match="same positive width"):
        pm.gextract_matrix("dense_track", pm.gintervals(["1", "2"], [100, 200], [200, 350]))
    with pytest.raises(ValueError, match="does not divide"):
        pm.gextract_matrix("dense_track", intervs, iterator=30)
    for bad in ("dense_track", 2.5, 0, True, intervs):
        with pytest.raises(ValueError, match="positive integer"):
            pm.gextract_matrix("dense_track", intervs, iterator=bad)
    with pytest.raises(ValueError, match="column names"):
        pm.gextract_matrix("dense_track", intervs, colnames=["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        pm.gextract_matrix("dense_track", intervs.iloc[:0])
    with pytest.raises(ValueError, match="1D"):
        pm.gextract_matrix("dense_track", pm.gintervals_2d("1", 0, 100, "1", 0, 100))
    chrom_end = int(pm.gintervals_all().set_index("chrom").loc["chr1", "end"])
    past_end = pd.DataFrame({"chrom": ["chr1"], "start": [chrom_end - 50], "end": [chrom_end + 50]})
    with pytest.raises(Exception, match="chromosome boundaries"):
        pm.gextract_matrix("dense_track", past_end)
    past_end_unaligned = pd.DataFrame({"chrom": ["chr1"], "start": [chrom_end - 53], "end": [chrom_end + 47]})
    with pytest.raises(Exception, match="chromosome boundaries"):
        pm.gextract_matrix("dense_track", past_end_unaligned, iterator=10)
