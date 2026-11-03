"""Dense import bins checked against a mean computed here from the records.

Port of misha's tests/testthat/test-gtrack.import-binning.R: partly covered
bins, defval fill, NaN records, records spanning bins, the last partial bin,
and a binsize larger than the chromosome.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from test_gtrack_import_bigwig import FIXTURES, make_db

import pymisha as pm


def expected_bins(recs, sizes, binsize, defval):
    out = []
    for chrom, size in sizes.items():
        r = recs[recs["chrom"] == chrom]
        starts = np.arange(0, size, binsize)
        ends = np.minimum(starts + binsize, size)
        ov = np.maximum(0, np.minimum(r["end"].to_numpy(), ends[:, None]) - np.maximum(r["start"].to_numpy(),
                                                                                      starts[:, None]))
        ov = np.where(np.isnan(r["value"].to_numpy()), 0, ov)
        s = (ov * np.nan_to_num(r["value"].to_numpy())).sum(axis=1)
        n = ov.sum(axis=1)
        if not np.isnan(defval):
            s = s + defval * (ends - starts - n)
            n = ends - starts
        with np.errstate(invalid="ignore"):
            v = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out.append(pd.DataFrame({"chrom": chrom, "start": starts, "end": ends, "v": v}))
    return pd.concat(out, ignore_index=True)


def assert_r_equal(x, want, tolerance, label):
    """testthat's expect_equal(tolerance=) on numbers (R's all.equal): NaN in the same places,
    and a mean relative difference within tolerance (absolute when the values are that small)."""
    x, want = np.asarray(x, dtype=float), np.asarray(want, dtype=float)
    np.testing.assert_array_equal(np.isnan(x), np.isnan(want), err_msg=label)
    ok = ~np.isnan(want)
    diff = np.abs(want[ok] - x[ok]).mean()
    scale = np.abs(want[ok]).mean()
    assert (diff / scale if scale > tolerance else diff) <= tolerance, label


def check_import_bins(file, recs, sizes):
    for binsize in (1, 10, 100, 2000):
        for defval in (np.nan, 0, 2.5):
            pm.gtrack_import("binned", "", str(file), binsize=binsize, defval=defval)
            x = pm.gextract("binned", pm.gintervals_all(), iterator=binsize, colnames=["v"])
            want = expected_bins(recs, sizes, binsize, defval)
            label = f"{Path(file).name}, binsize {binsize}, defval {defval}"
            assert x["chrom"].astype(str).tolist() == want["chrom"].tolist(), label
            np.testing.assert_array_equal(x["start"], want["start"], err_msg=label)
            np.testing.assert_array_equal(x["end"], want["end"], err_msg=label)
            assert_r_equal(x["v"], want["v"], 1e-6, label)
            pm.gtrack_rm("binned", force=True)


def test_dense_import_bins_are_defval_filled_mean(tmp_path):
    sizes = {"chr1": 1003, "chr2": 25}   # chr1 ends in a partial bin; chr2 is shorter than some binsizes
    make_db(tmp_path / "binning_db", sizes)
    recs = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1", "chr2"],
        "start": [0, 5, 8, 40, 995, 3],
        "end": [5, 6, 23, 41, 1003, 7],
        "value": [1, np.nan, 2.5, -1, 4, 1.5],
    })
    f = tmp_path / "recs.bedGraph"
    f.write_text("".join(f"{c}\t{s}\t{e}\t{'NaN' if np.isnan(v) else f'{v:g}'}\n"
                         for c, s, e, v in recs.itertuples(index=False)))
    check_import_bins(f, recs, sizes)


def test_bigwig_import_bins_are_defval_filled_mean(tmp_path):
    bg = pd.read_csv(FIXTURES / "bg.bedGraph", sep="\t", header=None, names=["chrom", "start", "end", "value"])
    sizes = {"chr1": 10000, "chr2": 10000}
    make_db(tmp_path / "binning_db", sizes)
    check_import_bins(FIXTURES / "bg.bw", bg[bg["chrom"].isin(sizes)], sizes)


# ---- pymisha addition: the bins are misha's to the bit -----------------------------------------


def misha_import_bins(recs, size, binsize, defval, float_products):
    """misha's import binner: sum of value * overlap (a float32 product when float_products,
    as in gtrack.import; double as in gtrack.create_dense) over the records in order, the defval
    fill, then the mean cast to float32."""
    starts = np.arange(0, size, binsize)
    ends = np.minimum(starts + binsize, size)
    s = np.zeros(len(starts))
    n = np.zeros(len(starts), dtype=np.int64)
    for st, en, v in zip(recs["start"], recs["end"], recs["value"], strict=True):
        ov = np.maximum(0, np.minimum(en, ends) - np.maximum(st, starts))
        if np.isnan(v):
            continue
        prod = (np.float32(v) * ov.astype(np.float32)).astype(np.float64) if float_products \
            else np.float64(np.float32(v)) * ov
        s += np.where(ov > 0, prod, 0)
        n += ov
    if not np.isnan(defval):
        s += defval * (ends - starts - n)
        n = ends - starts
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype(np.float32)


def test_dense_import_bins_match_misha_bit_for_bit(tmp_path):
    rng = np.random.default_rng(60427)
    size = 5003
    make_db(tmp_path / "db", {"chr1": size})
    lens = rng.integers(1, 30, 400)
    ends = np.cumsum(rng.integers(0, 10, 400) + lens)   # sorted, not overlapping
    vals = rng.normal(size=400) * 10.0 ** rng.integers(-3, 4, 400)
    vals[rng.random(400) < 0.03] = np.nan
    keep = ends <= size
    recs = pd.DataFrame({"start": (ends - lens)[keep], "end": ends[keep], "value": vals[keep]})
    f = tmp_path / "recs.bedGraph"
    f.write_text("".join(f"chr1\t{s}\t{e}\t{v:.9g}\n" for s, e, v in recs.itertuples(index=False)))
    recs = recs.assign(value=[float(f"{v:.9g}") for v in recs["value"]])
    for binsize in (7, 20, 1000):
        for defval in (np.nan, 2.5):
            pm.gtrack_import("binned", "", str(f), binsize=binsize, defval=defval)
            got = pm.gextract("binned", pm.gintervals_all(), iterator=binsize)["binned"].to_numpy(np.float32)
            want = misha_import_bins(recs, size, binsize, defval, True)
            np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                          err_msg=f"binsize {binsize}, defval {defval}")
            pm.gtrack_rm("binned", force=True)
    # the check can tell the two: double products give other bits in some bins
    assert not np.array_equal(misha_import_bins(recs, size, 20, 2.5, True).view(np.int32),
                              misha_import_bins(recs, size, 20, 2.5, False).view(np.int32))
