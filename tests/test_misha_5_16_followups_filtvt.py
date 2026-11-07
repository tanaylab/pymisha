"""Filtered virtual tracks read their source once for all the iterator intervals.

A filtered track vtrack reads the source over the union of the intervals' spans, a run of
intervals at a time, and gives each unmasked part the items it reads; a value-based vtrack
looks only at the source intervals near each interval. The values must be exactly those of
reading each interval on its own, which the reference functions here do.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import vtracks as vt

SEED = 60427
CHROMS = {"chr1": 500000, "chr2": 300000, "chrX": 200000}


def _rand_intervals(rng, n, maxlen, chroms=CHROMS):
    names = list(chroms)
    chrom = rng.choice(names, n)
    size = np.array([chroms[c] for c in chrom])
    start = (rng.random(n) * (size - 1)).astype(np.int64)
    end = np.minimum(start + rng.integers(1, maxlen + 1, n), size)
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end})


def _parts(start, end, mask):
    """[start, end) minus the mask intervals [(s, e), ...] (sorted, disjoint)."""
    parts = [(start, end)]
    for ms, me in mask:
        if me <= start:
            continue
        if ms >= end:
            break
        new = []
        for s, e in parts:
            if me <= s or ms >= e:
                new.append((s, e))
                continue
            if ms > s:
                new.append((s, ms))
            if me < e:
                new.append((me, e))
        parts = new
    return [(s, e) for s, e in parts if e > s]


def _ref_segments(intervals, mask_df, sshift, eshift):
    masks = {c: sorted(zip(g["start"].tolist(), g["end"].tolist(), strict=True))
             for c, g in mask_df.groupby("chrom")}
    rows = []
    for i, (c, s, e) in enumerate(zip(intervals["chrom"], intervals["start"], intervals["end"], strict=True)):
        s, e = max(int(s) + sshift, 0), min(int(e) + eshift, CHROMS[c])
        if e <= s:
            continue
        rows += [(c, ps, pe, i, pe - ps, s) for ps, pe in _parts(s, e, masks.get(c, []))]
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "orig_idx", "seg_len", "base_start"])


def _case(rng):
    J = _rand_intervals(rng, int(rng.integers(1, 120)), int(rng.choice([50, 3000, 40000])))
    if rng.random() < 0.5:
        J = J.sort_values(["chrom", "start"]).reset_index(drop=True)
    # short mask intervals split bins and track intervals between two unmasked parts
    mask = _rand_intervals(rng, int(rng.choice([1, 10, 300, 3000])), int(rng.choice([10, 60, 4000, 30000])))
    sshift = int(rng.choice([0, -1, -37, -2000]))
    eshift = int(rng.choice([0, 1, 37, 2000]))
    return J, mask, sshift, eshift


def test_build_unmasked_segments_subtracts_the_mask():
    rng = np.random.default_rng(SEED)
    for _ in range(40):
        J, mask, sshift, eshift = _case(rng)
        mask = vt._canonicalize_filter_df(mask)
        got = vt._build_unmasked_segments(J, {"sshift": sshift, "eshift": eshift}, mask)
        ref = _ref_segments(J, mask, sshift, eshift)
        if len(ref) == 0:
            assert got is None
            continue
        assert got[list(ref.columns)].astype(str).values.tolist() == ref.astype(str).values.tolist()


def test_seq_range_sums_add_in_order():
    rng = np.random.default_rng(SEED)
    vals = (rng.standard_normal(5000) * 10.0 ** rng.integers(-8, 9, 5000)).astype(np.float32).astype(float)
    vals[rng.random(5000) < 0.05] = -0.0
    lo = rng.integers(0, 5000, 400)
    n = rng.choice([0, 1, 2, 7, 9, 100, 1500], 400)
    hi = np.minimum(lo + n, 5000)
    lo[:3], hi[:3] = 0, 5000  # a few long ranges, summed on their own
    got = vt._seq_range_sums(vals, lo, hi)
    want = [float(np.cumsum(vals[a:b])[-1]) if b > a else -0.0 for a, b in zip(lo, hi, strict=True)]
    assert repr(got.tolist()) == repr(want)


def _ref_track_values(track, func, J, mask, sshift, eshift, q):
    """Each interval read on its own: its unmasked parts, the source's items over its span, each
    item with the first part ending after its start (if it overlaps that part)."""
    mask = vt._canonicalize_filter_df(mask)
    segs = _ref_segments(J, mask, sshift, eshift)
    out = [np.nan] * len(J)
    for o, g in segs.groupby("orig_idx"):
        ps, pe = g["start"].to_numpy(), g["end"].to_numpy()
        base = int(g["base_start"].iloc[0])
        r = pm.gextract(track, pd.DataFrame({"chrom": [g["chrom"].iloc[0]], "start": [ps[0]], "end": [pe[-1]]}))
        if r is None:
            vals = np.empty(0)
            pos = j = np.empty(0, dtype=np.int64)
        else:
            r = r[~np.isnan(r[track].to_numpy(dtype=float))]
            rs, re_ = r["start"].to_numpy(), r["end"].to_numpy()
            j = np.searchsorted(pe, rs, side="right")
            hit = j < len(ps)
            hit[hit] = ps[j[hit]] < re_[hit]
            vals, j = r[track].to_numpy(dtype=float)[hit], j[hit]
            pos = np.maximum(rs[hit], ps[np.minimum(j, len(ps) - 1)])
        if func == "size":
            out[o] = float(len(vals))
            continue
        if not len(vals):
            continue
        if func in ("avg", "sum"):
            bounds = np.flatnonzero(np.diff(j)) + 1
            ns = np.diff(np.r_[0, bounds, len(vals)]).astype(float)
            sums = np.asarray([np.cumsum(v)[-1] for v in np.split(vals, bounds)])
            if func == "sum":
                out[o] = float(np.cumsum(sums.astype(np.float32).astype(float))[-1])
            else:
                means = (sums / ns).astype(np.float32).astype(float)
                out[o] = float(np.cumsum(means * ns)[-1] / ns.sum())
        elif func == "stddev":
            if len(vals) >= 2:
                out[o] = float(np.std(vals.tolist(), ddof=1))
        elif func == "quantile":
            out[o] = float(np.quantile(vals.tolist(), q))
        elif func == "lse":
            out[o] = vt._logsumexp(vals.tolist())
        else:
            best = 0
            for k in range(1, len(vals)):
                if (vals[k] > vals[best]) if func.startswith("max") else (vals[k] < vals[best]):
                    best = k
            out[o] = float(pos[best] - base) if func.endswith("relative") else float(pos[best])
    return out


TRACK_FUNCS = ["avg", "sum", "size", "stddev", "quantile", "lse", "max.pos.abs", "max.pos.relative",
               "min.pos.abs", "min.pos.relative"]


@pytest.mark.parametrize(("track", "read_rows"), [
    ("dense_track", None), ("dense_track", 3), ("sparse_track", None), ("array_track", None)])
def test_filtered_track_vtrack_matches_reading_each_interval(track, read_rows, monkeypatch):
    if read_rows is not None:  # a dense track read in many runs of intervals
        monkeypatch.setattr(vt, "_UNMASKED_READ_ROWS", read_rows)
    rng = np.random.default_rng(SEED)
    for _ in range(6):
        J, mask, sshift, eshift = _case(rng)
        for func in TRACK_FUNCS:
            kw = {"params": 0.3} if func == "quantile" else {}
            pm.gvtrack_create("vt_filt", track, func, **kw)
            pm.gvtrack_filter("vt_filt", mask)
            pm.gvtrack_iterator("vt_filt", sshift=sshift, eshift=eshift)
            got = pm.gextract("vt_filt", J, iterator=J)
            got = got.sort_values("intervalID")["vt_filt"].to_numpy(dtype=float).tolist()
            want = _ref_track_values(track, func, J, mask, sshift, eshift, 0.3)
            assert repr(got) == repr(want), (track, func)


def test_value_source_finds_overlapping_and_nearest_intervals():
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        start = rng.integers(0, 2000, n)
        end = start + rng.integers(1, 300, n)
        value = rng.normal(size=n)
        value[rng.random(n) < 0.2] = np.nan
        src = vt._ValueSource(pd.DataFrame({"start": start, "end": end, "value": value}))
        s = int(rng.integers(0, 2300))
        e = s + int(rng.integers(1, 400))
        assert src.overlapping(s, e).tolist() == np.flatnonzero((start < e) & (end > s)).tolist()
        valid = ~np.isnan(value)
        dist = (np.maximum(s - end, 0) + np.maximum(start - e, 0))[valid]
        want = np.flatnonzero(dist == dist.min()).tolist() if valid.any() else []
        assert src.nearest(s, e).tolist() == want
