"""Dense import bins checked against a mean computed here from the records.

Port of misha's tests/testthat/test-gtrack.import-binning.R: partly covered
bins, defval fill, NaN records, records spanning bins, the last partial bin,
and a binsize larger than the chromosome.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    """misha's import binner: sum of value * overlap (a float32 product when float_products, as in
    gtrack.import before misha 5.17.0; double as in gtrack.create_dense and misha >= 5.17.0's
    gtrack.import) over the records in order, the defval fill, then the mean cast to float32."""
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
            want = misha_import_bins(recs, size, binsize, defval, False)
            np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                          err_msg=f"binsize {binsize}, defval {defval}")
            pm.gtrack_rm("binned", force=True)
    # the check can tell the two: double products give other bits in some bins
    assert not np.array_equal(misha_import_bins(recs, size, 20, 2.5, True).view(np.int32),
                              misha_import_bins(recs, size, 20, 2.5, False).view(np.int32))


# ---- misha 5.17.0 ------------------------------------------------------------------------------


def test_a_value_near_the_float_limit_survives_binning(tmp_path):
    make_db(tmp_path / "binning_db", {"chr1": 1000})
    f = tmp_path / "big.bedGraph"
    f.write_text("chr1\t0\t32\t3.4e38\nchr1\t32\t64\t1e37\nchr1\t64\t96\t1\n")
    # a float product value * 32 would overflow to Inf, which reads back as NaN
    for binsize in (1, 32):
        pm.gtrack_import("big", "", str(f), binsize=binsize)
        got = pm.gextract("big", pm.gintervals("chr1", 0, 96), iterator=32)["big"].to_numpy()
        np.testing.assert_array_equal(got, np.float32([3.4e38, 1e37, 1]).astype(np.float64),
                                      err_msg=f"binsize {binsize}")
        pm.gtrack_rm("big", force=True)


def past_end_records(sizes):
    """Records a model writes: 32 bp each, end = start + 32 also at the chromosome end; chr1 has
    one more that starts past its end."""
    recs = pd.concat([pd.DataFrame({"chrom": chrom, "start": np.arange(0, (1024 if chrom == "chr1" else 576) + 1, 32)})
                      for chrom in sizes], ignore_index=True)
    recs["end"] = recs["start"] + 32
    recs["value"] = np.arange(1, len(recs) + 1) / 2
    return recs


def collect_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught]


def test_records_past_the_chromosome_end_are_clipped_or_dropped_with_one_warning_in_every_text_format(tmp_path):
    sizes = {"chr1": 1000, "chr2": 600}
    make_db(tmp_path / "clip_db", sizes)
    recs = past_end_records(sizes)
    size = recs["chrom"].map(sizes)
    # chr1 992-1024 and chr2 576-608 are clipped, chr1 1024-1056 is dropped
    assert ((recs["end"] > size) & (recs["start"] < size)).sum() == 2
    assert (recs["start"] >= size).sum() == 1
    kept = recs[recs["start"] < size].copy()
    kept["end"] = np.minimum(kept["end"], kept["chrom"].map(sizes))

    lines = [f"{c}\t{s}\t{e}\t{v:g}" for c, s, e, v in recs.itertuples(index=False)]
    by_chrom = {chrom: recs[recs["chrom"] == chrom] for chrom in sizes}
    files = {
        "bedGraph": lines,
        "tab": ["chrom\tstart\tend\tvalue", *lines],
        "fixedStep": [x for chrom, r in by_chrom.items()
                      for x in [f"fixedStep chrom={chrom} start=1 step=32 span=32", *(f"{v:g}" for v in r["value"])]],
        "variableStep": [x for chrom, r in by_chrom.items()
                         for x in [f"variableStep chrom={chrom} span=32",
                                   *(f"{s + 1}\t{v:g}" for s, v in zip(r["start"], r["value"], strict=True))]],
    }
    first_line = {"bedGraph": 32, "tab": 33, "fixedStep": 33, "variableStep": 33}
    for fmt, text in files.items():
        f = tmp_path / f"{fmt}.txt"   # the format is found by reading the file
        f.write_text("\n".join(text) + "\n")
        for binsize in (0, 32, 10):
            label = f"{fmt} binsize {binsize}"
            warns = collect_warnings(lambda f=f, binsize=binsize: pm.gtrack_import("clipped", "", str(f),
                                                                                  binsize=binsize))
            assert warns == [
                f"File {f}: 2 record(s) running past the end of their chromosome were clipped to it, and 1 record(s) "
                f"starting at or past the chromosome end were dropped; the first: line {first_line[fmt]}: "
                "(chr1, 992, 1024), chromosome size 1000"
            ], label
            if binsize == 0:
                x = pm.gextract("clipped", pm.gintervals_all(), iterator="clipped", colnames=["v"])
                np.testing.assert_array_equal(x["end"], kept["end"], err_msg=label)
                np.testing.assert_array_equal(x["v"], kept["value"], err_msg=label)
            else:
                x = pm.gextract("clipped", pm.gintervals_all(), iterator=binsize, colnames=["v"])
                assert_r_equal(x["v"], expected_bins(kept, sizes, binsize, np.nan)["v"], 1e-6, label)
            pm.gtrack_rm("clipped", force=True)


def test_wig_records_at_coordinates_near_2_63_are_dropped_like_any_record_past_the_chromosome_end(tmp_path):
    make_db(tmp_path / "ovf_db", {"chr1": 1000})
    files = {
        # start + span would overflow
        "variableStep": ["variableStep chrom=chr1 span=10", "11\t7", "9223372036854775800\t1"],
        # start, and start + step, would overflow
        "fixedStep": ["fixedStep chrom=chr1 start=11 step=10 span=1", "7",
                      "fixedStep chrom=chr1 start=9223372036854775800 step=10", "1", "2", "3"],
    }
    dropped = {"variableStep": 1, "fixedStep": 3}
    for fmt, text in files.items():
        f = tmp_path / f"{fmt}.wig"
        f.write_text("\n".join(text) + "\n")
        for binsize in (0, 10):
            label = f"{fmt} binsize {binsize}"
            with pytest.warns(UserWarning, match=re.escape(
                "0 record(s) running past the end of their chromosome were clipped to it, and "
                f"{dropped[fmt]} record(s) starting at or past"
            )):
                pm.gtrack_import("ovf", "", str(f), binsize=binsize)
            x = pm.gextract("ovf", pm.gintervals_all(), iterator=binsize or "ovf", colnames=["v"])
            x = x[~np.isnan(x["v"])]
            assert x["start"].tolist() == [10], label
            assert x["v"].tolist() == [7], label
            pm.gtrack_rm("ovf", force=True)


def test_the_record_past_the_chromosome_end_that_the_warning_names_is_the_first_in_the_file(tmp_path):
    sizes = {"chr1": 1000, "chr2": 600}
    make_db(tmp_path / "first_db", sizes)
    # chr2 comes first in the files, after chr1 in the database
    files = {
        "bedGraph": ["chr2\t576\t608\t1", "chr1\t992\t1024\t2"],
        "tab": ["chrom\tstart\tend\tvalue", "chr2\t576\t608\t1", "chr1\t992\t1024\t2"],
        "fixedStep": ["fixedStep chrom=chr2 start=577 step=32 span=32", "1",
                      "fixedStep chrom=chr1 start=993 step=32 span=32", "2"],
    }
    first = {"bedGraph": "line 1: (chr2, 576, 608)", "tab": "line 2: (chr2, 576, 608)",
             "fixedStep": "line 2: (chr2, 576, 608)"}
    for fmt, text in files.items():
        f = tmp_path / f"{fmt}.txt"   # the format is found by reading the file
        f.write_text("\n".join(text) + "\n")
        with pytest.warns(UserWarning, match=re.escape(f"the first: {first[fmt]}")):
            pm.gtrack_import("first", "", str(f), binsize=32)
        pm.gtrack_rm("first", force=True)

    # an array import of several files: the first such record of the first file that has one
    a1, a2, a3 = (tmp_path / f"a{i}.tsv" for i in (1, 2, 3))
    a1.write_text("chrom\tstart\tend\tx\nchr1\t0\t10\t1\n")
    a2.write_text("chrom\tstart\tend\ty\nchr2\t576\t608\t1\nchr1\t992\t1024\t2\n")
    a3.write_text("chrom\tstart\tend\tz\nchr1\t992\t1024\t3\n")
    assert collect_warnings(lambda: pm.gtrack_array_import("first_arr", "", str(a1), str(a2))) == [
        f"File {a2}: 2 record(s) running past the end of their chromosome were clipped to it, and 0 record(s) starting "
        "at or past the chromosome end were dropped; the first: line 2: (chr2, 576, 608), chromosome size 600"
    ]
    assert collect_warnings(lambda: pm.gtrack_array_import("first_arr3", "", str(a1), str(a2), str(a3))) == [
        f"Files {a2}, {a3}: 3 record(s) running past the end of their chromosome were clipped to it, and 0 record(s) "
        f"starting at or past the chromosome end were dropped; the first: file {a2}, line 2: (chr2, 576, 608), "
        "chromosome size 600"
    ]


# ---- gtrack_import finds the format of a text file by reading it, as R misha does ---------------


@pytest.mark.parametrize(("name", "text", "want"), [
    # bedGraph and WIG whatever the name; a tab-delimited file named .wig
    ("bg.txt", "chr1\t0\t10\t1\nchr1\t20\t30\t2\n", [(0, 10, 1.0), (20, 30, 2.0)]),
    ("noext", "fixedStep chrom=chr1 start=1 step=10 span=5\n1\n2\n", [(0, 5, 1.0), (10, 15, 2.0)]),
    ("tab.wig", "chrom\tstart\tend\tv\nchr1\t0\t10\t1\nchr1\t20\t30\t2\n", [(0, 10, 1.0), (20, 30, 2.0)]),
    ("hdr.tsv", "track type=bedGraph\nchr1\t0\t10\t1.5\n", [(0, 10, 1.5)]),
    ("hdronly.txt", "chrom\tstart\tend\tv\n", []),
])
def test_the_format_of_a_text_file_is_found_by_reading_it(tmp_path, name, text, want):
    # checked against R misha 5.17.0 on the same files
    make_db(tmp_path / "db", {"chr1": 1000, "chr2": 500})
    f = tmp_path / name
    f.write_text(text)
    pm.gtrack_import("t", "", str(f), 0)
    x = pm.gextract("t", pm.gintervals_all())
    got = [] if x is None else list(zip(x["start"], x["end"], x["t"], strict=True))
    assert got == want


@pytest.mark.parametrize(("name", "text", "wig_error"), [
    ("unsorted.bedGraph", "chr1\t0\t10\t1\nchr2\t0\t10\t1\nchr1\t20\t30\t1\n", "not sorted by chromosomes"),
    ("three.txt", "chr1\t0\t10\nchr1\t20\t30\n", "Cannot parse WIG/BedGraph line"),
])
def test_a_file_that_is_neither_wig_nor_tab_delimited_gives_both_reasons(tmp_path, name, text, wig_error):
    make_db(tmp_path / "db", {"chr1": 1000, "chr2": 500})
    f = tmp_path / name
    f.write_text(text)
    with pytest.raises(ValueError, match=rf"Unrecognized format of file {re.escape(str(f))}\.\nWIG parser error: .*"
                                         rf"{wig_error}.*\nCSV parser error: .*'chrom' column"):
        pm.gtrack_import("t", "", str(f), 0)
    assert not pm.gtrack_exists("t")
    # a tab-delimited file with two value columns is one, and is refused as such
    two = tmp_path / "two.txt"
    two.write_text("chrom\tstart\tend\ta\tb\nchr1\t0\t10\t1\t2\n")
    with pytest.raises(ValueError, match="exactly one value column"):
        pm.gtrack_import("t", "", str(two), 0)
