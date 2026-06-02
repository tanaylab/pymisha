"""Output-parity guard for the incremental sliding-window reducer.

Windowed vtrack reductions (avg/sum/lse/exists/size) must produce identical
results whether the reduction is recomputed per output bin or maintained
incrementally across a sliding window. Expected values are computed
independently with numpy, so this test is valid both BEFORE the C++ sliding
path lands (recompute) and AFTER (slide). It is the regression net.

Port reference: R misha GenomeTrackFixedBin read_interval_reducers_only.
"""
import numpy as np
import pytest

import pymisha as pm

CHROM = 1
CHROM_LEN = 500000  # chrom 1 in the test DB


def _lse(window_with_nan):
    v = np.asarray(window_with_nan, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan
    m = v.max()
    return float(m + np.log(np.sum(np.exp(v - m))))


@pytest.fixture()
def raw_bins():
    """Per-bin values of dense_track on chrom 1 (one row per native bin)."""
    pm.gvtrack_clear()
    info = pm.gtrack_info("dense_track")
    binsize = int(info["bin_size"])
    iv = pm.gintervals(CHROM, 0, CHROM_LEN)
    raw = pm.gextract("dense_track", iv, iterator=binsize)
    raw = raw.sort_values("start").reset_index(drop=True)
    vals = raw["dense_track"].to_numpy(dtype=float)
    yield binsize, vals
    pm.gvtrack_clear()


def _expected(func, vals, binsize, sshift, eshift):
    n = len(vals)
    exp = np.empty(n, dtype=float)
    for i in range(n):
        # window coordinates with chrom-boundary clamping (apply_shift)
        start = max(0, i * binsize + sshift)
        end = min(CHROM_LEN, (i + 1) * binsize + eshift)
        sbin = start // binsize
        ebin = -(-end // binsize)  # ceil
        w = vals[sbin:ebin]
        wv = w[~np.isnan(w)]
        if func == "avg":
            exp[i] = wv.mean() if wv.size else np.nan
        elif func == "sum":
            exp[i] = wv.sum() if wv.size else np.nan
        elif func == "size":
            exp[i] = float(wv.size)
        elif func == "exists":
            exp[i] = 1.0 if wv.size else 0.0
        elif func == "lse":
            exp[i] = _lse(w)
        else:
            raise ValueError(func)
    return exp


@pytest.mark.parametrize("func", ["avg", "sum", "exists", "size", "lse"])
@pytest.mark.parametrize("sshift,eshift", [(0, 0), (-200, 200), (-1000, 1000)])
def test_windowed_reducer_matches_numpy(raw_bins, func, sshift, eshift):
    binsize, vals = raw_bins
    pm.gvtrack_create("vt", "dense_track", func=func)
    pm.gvtrack_iterator("vt", sshift=sshift, eshift=eshift)
    iv = pm.gintervals(CHROM, 0, CHROM_LEN)
    out = pm.gextract("vt", iv, iterator=binsize).sort_values("start").reset_index(drop=True)
    got = out["vt"].to_numpy(dtype=float)

    exp = _expected(func, vals, binsize, sshift, eshift)
    assert got.shape == exp.shape
    if func in ("size", "exists"):
        np.testing.assert_array_equal(got, exp)
    else:
        np.testing.assert_allclose(got, exp, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_midchrom_start_then_slide(raw_bins):
    """First read seeds at a non-zero start (goto), then slides forward."""
    binsize, vals = raw_bins
    sshift, eshift = -500, 500
    pm.gvtrack_create("vt", "dense_track", func="lse")
    pm.gvtrack_iterator("vt", sshift=sshift, eshift=eshift)
    s, e = 100000, 120000
    iv = pm.gintervals(CHROM, s, e)
    out = pm.gextract("vt", iv, iterator=binsize).sort_values("start").reset_index(drop=True)
    got = out["vt"].to_numpy(dtype=float)

    exp_full = _expected("lse", vals, binsize, sshift, eshift)
    lo, hi = s // binsize, e // binsize
    np.testing.assert_allclose(got, exp_full[lo:hi], rtol=1e-5, atol=1e-5, equal_nan=True)


def test_gap_between_blocks_forces_full_reread(raw_bins):
    """Two forward, non-adjacent blocks in one extract: the jump across the gap
    must trigger a clean full re-read, not a corrupt slide."""
    binsize, vals = raw_bins
    sshift, eshift = -300, 300
    pm.gvtrack_create("vt", "dense_track", func="sum")
    pm.gvtrack_iterator("vt", sshift=sshift, eshift=eshift)
    iv = pm.gintervals([CHROM, CHROM], [100000, 300000], [110000, 310000])
    out = pm.gextract("vt", iv, iterator=binsize).sort_values("start").reset_index(drop=True)

    exp_full = _expected("sum", vals, binsize, sshift, eshift)
    for _, r in out.iterrows():
        i = int(r["start"]) // binsize
        np.testing.assert_allclose(
            float(r["vt"]), exp_full[i], rtol=1e-5, atol=1e-5, equal_nan=True
        )
