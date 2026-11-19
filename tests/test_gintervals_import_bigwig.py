"""gintervals_import_bigwig reads the records of a bigWig over intervals.

Port of misha's tests/testthat/test-gintervals.import_bigwig.R. Fixtures as in
test_gtrack_import_bigwig.py: bg.bw has three records per data block
(-itemsPerSlot=3), so every chromosome spans many blocks and a query selects
some of them; step.bw holds fixedStep and variableStep sections.
"""

import logging
import struct
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

FIXTURES = Path(__file__).parent / "fixtures" / "bigwig"


def make_db(root, sizes):
    """misha's create_test_db: a per-chromosome database of all-A sequences, initialized."""
    (root / "tracks").mkdir(parents=True)
    (root / "seq").mkdir()
    (root / "chrom_sizes.txt").write_text("".join(f"{c}\t{n}\n" for c, n in sizes.items()))
    for c, n in sizes.items():
        (root / "seq" / f"{c}.seq").write_text("A" * n + "\n")
    pm.gdb_init(str(root))


@pytest.fixture
def bigwig_db(tmp_path):
    make_db(tmp_path / "bigwig_db", {"chr1": 10000, "chr2": 10000})   # scaffold_9 is not in it


def text_records(name):
    """The records of the text file, as gtrack_import reads them (clipped to the chromosomes)."""
    pm.gtrack_import("text_records", "", str(FIXTURES / name), binsize=0)
    try:
        x = pm.gextract("text_records", pm.gintervals_all(), iterator="text_records", colnames=["value"])
    finally:
        pm.gtrack_rm("text_records", force=True)
    return x[["chrom", "start", "end", "value"]].reset_index(drop=True)


def same_records(got, want):
    assert got["chrom"].astype(str).tolist() == want["chrom"].astype(str).tolist()
    np.testing.assert_array_equal(got["start"].to_numpy(np.int64), want["start"].to_numpy(np.int64))
    np.testing.assert_array_equal(got["end"].to_numpy(np.int64), want["end"].to_numpy(np.int64))
    np.testing.assert_allclose(got["value"].to_numpy(float), want["value"].to_numpy(float), rtol=1e-7)


def overlapping(records, query):
    """The records that overlap one of the intervals of query."""
    keep = [((query["chrom"] == c) & (query["start"] < e) & (query["end"] > s)).any()
            for c, s, e in zip(records["chrom"].astype(str), records["start"], records["end"], strict=True)]
    return records[np.array(keep, dtype=bool)].reset_index(drop=True)


def random_query(rng, n, width):
    start = rng.integers(0, 10000 - width + 1, n)
    return pd.DataFrame({"chrom": rng.choice(["chr1", "chr2"], n), "start": start,
                         "end": start + rng.integers(1, width + 1, n)})


@pytest.mark.parametrize("bw", ["bg.bw", "bg_unc.bw", "bg_libbigwig.bw"])
def test_all_records_are_those_of_the_text_file(bigwig_db, bw, caplog):
    want = text_records("bg.bedGraph")
    with caplog.at_level(logging.INFO, logger="pymisha"):
        got = pm.gintervals_import_bigwig(FIXTURES / bw)
    assert "scaffold_9" in caplog.text
    same_records(got, want)
    assert got["value"].dtype == np.float64


def test_all_step_records(bigwig_db):
    same_records(pm.gintervals_import_bigwig(FIXTURES / "step.bw"), text_records("step.wig"))


@pytest.mark.parametrize("bw", ["bg.bw", "bg_unc.bw", "step.bw"])
def test_query_returns_each_overlapping_record_once_and_whole(bigwig_db, bw):
    rng = np.random.default_rng(17)
    all_records = pm.gintervals_import_bigwig(FIXTURES / bw)
    # overlapping and touching query intervals, spanning several blocks or within one
    for width in (5, 60, 700):
        query = random_query(rng, 40, width)
        got = pm.gintervals_import_bigwig(FIXTURES / bw, query)
        same_records(got, overlapping(all_records, query))


def test_blocks_away_from_the_query_are_not_read(bigwig_db, tmp_path):
    src = FIXTURES / "bg.bw"
    data = bytearray(src.read_bytes())
    data_offset = struct.unpack_from("<Q", data, 16)[0]
    # a byte of the first data block (chr1 20-132, after the 8-byte block count) flipped: zlib's checksum fails
    data[data_offset + 18] ^= 0xFF
    f = tmp_path / "flipped.bw"
    f.write_bytes(bytes(data))
    with pytest.raises(Exception, match="failed to decompress"):
        pm.gintervals_import_bigwig(f)
    with pytest.raises(Exception, match="failed to decompress"):
        pm.gintervals_import_bigwig(f, pm.gintervals("chr1", 100, 110))
    q = pm.gintervals(["chr1", "chr2"], [5000, 5000], [5100, 5100])
    pd.testing.assert_frame_equal(pm.gintervals_import_bigwig(f, q), pm.gintervals_import_bigwig(src, q))


def test_records_serve_as_a_value_based_vtrack(bigwig_db):
    pm.gtrack_import("bg", "", str(FIXTURES / "bg.bw"), binsize=0)
    peaks = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr2"], "start": [100, 132, 1000, 9000],
                          "end": [400, 134, 3000, 9050]})
    src = pm.gintervals_import_bigwig(FIXTURES / "bg.bw", peaks)
    for func in ("sum", "weighted.sum", "avg", "max", "size", "exists"):
        pm.gvtrack_create("sig", src=src, func=func)
        pm.gvtrack_create("ref", src="bg", func=func)
        x = pm.gextract(["sig", "ref"], peaks, iterator=peaks)
        np.testing.assert_allclose(x["sig"].to_numpy(float), x["ref"].to_numpy(float), rtol=1e-6, err_msg=func)
        if func in ("size", "exists"):
            assert x["sig"].iloc[1] == 0   # no record in chr1 132-134
        else:
            assert np.isnan(x["sig"].iloc[1]), func


def test_intervals_no_record_overlaps_give_no_rows_that_serve_as_a_vtrack(bigwig_db):
    bw = FIXTURES / "bg.bw"
    empty = pm.gintervals_import_bigwig(bw, pm.gintervals("chr1", 132, 134))   # between the records 94-132 and 134-140
    pd.testing.assert_frame_equal(empty, pm.gintervals_import_bigwig(bw, pm.gintervals("chr1", 100, 110)).iloc[:0])
    pm.gvtrack_create("sig", src=empty, func="weighted.sum")
    iv = pm.gintervals("chr1", 0, 10000)
    assert np.isnan(pm.gextract("sig", iv, iterator=iv)["sig"]).all()


def test_records_past_the_chromosome_end_are_clipped_as_gtrack_import_clips_them(tmp_path):
    # chr2 ends inside the record 9870-9910, before the record 9933-9981
    make_db(tmp_path / "short_bg_db", {"chr1": 10000, "chr2": 9900})
    bw = FIXTURES / "bg.bw"
    msg = (r"1 record\(s\) running past the end of their chromosome were clipped to it, and 1 record\(s\) starting "
           r"at or past the chromosome end were dropped; the first: .*\(chr2, 9870, 9910\), chromosome size 9900")
    with pytest.warns(UserWarning, match=msg):
        x = pm.gintervals_import_bigwig(bw)
    assert x["end"].iloc[-1] == 9900
    # a query counts only the records it returns
    with pytest.warns(UserWarning, match=r"1 record\(s\) running past .* clipped to it, and 0 record\(s\)"):
        x = pm.gintervals_import_bigwig(bw, pm.gintervals("chr2", 9880, 9890))
    assert x["end"].tolist() == [9900]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.gintervals_import_bigwig(bw, pm.gintervals("chr2", 9000, 9100))


def test_chromosome_names_are_matched_through_the_database_aliases(tmp_path):
    bw = FIXTURES / "bg.bw"
    make_db(tmp_path / "bigwig_db", {"chr1": 10000, "chr2": 10000})
    want = pm.gintervals_import_bigwig(bw, pm.gintervals("chr2", 0, 500))
    make_db(tmp_path / "noprefix_db", {"1": 10000, "2": 10000})
    got = pm.gintervals_import_bigwig(bw, pm.gintervals("2", 0, 500))
    chroms = pm.gintervals_all()["chrom"].astype(str).tolist()
    assert set(got["chrom"]) == {chroms[1]}
    pd.testing.assert_frame_equal(got.drop(columns="chrom"), want.drop(columns="chrom"))


def test_bad_input_is_an_error(bigwig_db, tmp_path):
    bw = FIXTURES / "bg.bw"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pm.gintervals_import_bigwig(tmp_path / "no_such_file.bw")
    with pytest.raises(Exception, match="is not a bigWig file"):
        pm.gintervals_import_bigwig(FIXTURES / "bg.bedGraph")
    make_db(tmp_path / "other_db", {"contigA": 10000, "contigB": 10000})
    with pytest.raises(ValueError, match="None of the chromosome names"):
        pm.gintervals_import_bigwig(bw)
