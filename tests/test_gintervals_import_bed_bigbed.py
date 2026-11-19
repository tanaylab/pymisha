"""gintervals_import_bed reads bigBed files.

Port of misha's tests/testthat/test-gintervals.import_bed-bigbed.R. The fixtures,
copied from misha, were made by UCSC's bedToBigBed with -blockSize=2
-itemsPerSlot=4, so the records span many data blocks:
  typed.bb:  bedToBigBed -tab -type=bed9+7 -as=typed.as typed.bed chrom.sizes (extra
             fields of several autoSql types; three records on scaffold_9, which the
             test database lacks; chr1 5000-5100 before 5000-5050, so the file's order
             is not the sorted one; chr2 9870-9910 is the last record of chr2)
  bed12.bb:  bedToBigBed -type=bed12 bed12.bed chrom.sizes (UCSC's default BED autoSql)
"""

import logging
import shutil
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha.intervals import _sort_intervals_df

FIXTURES = Path(__file__).parent / "fixtures" / "bigbed"


def make_db(root, sizes):
    """misha's create_test_db: a per-chromosome database of all-A sequences, initialized."""
    (root / "tracks").mkdir(parents=True)
    (root / "seq").mkdir()
    (root / "chrom_sizes.txt").write_text("".join(f"{c}\t{n}\n" for c, n in sizes.items()))
    for c, n in sizes.items():
        (root / "seq" / f"{c}.seq").write_text("A" * n + "\n")
    pm.gdb_init(str(root))


@pytest.fixture
def bigbed_db(tmp_path):
    make_db(tmp_path / "bigbed_db", {"chr1": 10000, "chr2": 10000})


def patched(tmp_path, name, pos, fmt, value):
    """A copy of a fixture with the header field at pos (struct format fmt) set to value."""
    data = bytearray((FIXTURES / name).read_bytes())
    struct.pack_into(fmt, data, pos, value)
    f = tmp_path / f"patched_{name}"
    f.write_bytes(bytes(data))
    return f


def typed_text():
    x = pd.read_csv(FIXTURES / "typed.bed", sep="\t", header=None, dtype=str, keep_default_na=False)
    return x[x[0] != "scaffold_9"].reset_index(drop=True)


def test_fields_are_named_and_typed_by_the_autosql_in_file_order(bigbed_db, caplog):
    with caplog.at_level(logging.INFO, logger="pymisha"):
        x = pm.gintervals_import_bed(FIXTURES / "typed.bb", keep_order=True)
    assert "scaffold_9" in caplog.text
    assert list(x.columns) == ["chrom", "start", "end", "name", "score", "strand", "thickStart", "thickEnd",
                               "itemRgb", "count", "signal", "pval", "big", "note", "state", "triple"]
    text = typed_text()
    assert x["chrom"].tolist() == text[0].tolist()
    assert x["start"].tolist() == text[1].astype(int).tolist()
    assert x["end"].tolist() == text[2].astype(int).tolist()
    assert x["strand"].tolist() == text[5].map({"+": 1, "-": -1, ".": 0}).tolist()
    for col, i in {"score": 4, "thickStart": 6, "thickEnd": 7, "count": 9, "signal": 10, "pval": 11, "big": 12}.items():
        assert x[col].dtype == np.float64, col
        np.testing.assert_array_equal(x[col].to_numpy(), text[i].astype(float).to_numpy(), err_msg=col)
    # itemRgb is "R,G,B" text although autoSql declares it a uint
    for col, i in {"name": 3, "itemRgb": 8, "note": 13, "state": 14, "triple": 15}.items():
        assert x[col].tolist() == text[i].tolist(), col
    assert x["itemRgb"].str.contains(",").any()

    # sorted by default; the file has chr1 5000-5100 before 5000-5050
    by_pos = pm.gintervals_import_bed(FIXTURES / "typed.bb")
    pd.testing.assert_frame_equal(by_pos, _sort_intervals_df(x))
    same_start = ["chr1_same_start_0", "chr1_same_start_1"]
    assert x.loc[x["name"].isin(same_start), "end"].tolist() == [5100, 5050]
    assert by_pos.loc[by_pos["name"].isin(same_start), "end"].tolist() == [5050, 5100]


def test_name_score_and_strand_govern_the_fields_of_those_names(bigbed_db):
    full = pm.gintervals_import_bed(FIXTURES / "typed.bb")
    x = pm.gintervals_import_bed(FIXTURES / "typed.bb", name=False, score=False, strand=False)
    pd.testing.assert_frame_equal(x, full.drop(columns=["name", "score", "strand"]))


def test_a_bigbed_reads_as_the_text_bed_it_was_made_from(bigbed_db):
    x = pm.gintervals_import_bed(FIXTURES / "bed12.bb")
    text = pm.gintervals_import_bed(FIXTURES / "bed12.bed")
    assert list(x.columns) == ["chrom", "start", "end", "name", "score", "strand", "thickStart", "thickEnd",
                               "reserved", "blockCount", "blockSizes", "chromStarts"]
    pd.testing.assert_frame_equal(x[list(text.columns)], text, check_dtype=False)


def test_a_bigbed_without_autosql_gets_the_bed_standard_field_names(bigbed_db, tmp_path):
    with_as = pm.gintervals_import_bed(FIXTURES / "bed12.bb")
    # autoSqlOffset (8 bytes at 36) set to 0
    pd.testing.assert_frame_equal(pm.gintervals_import_bed(patched(tmp_path, "bed12.bb", 36, "<Q", 0)), with_as)
    with pytest.raises(Exception, match="has 7 fields past its 9 BED standard ones, and no autoSql to name them"):
        pm.gintervals_import_bed(patched(tmp_path, "typed.bb", 36, "<Q", 0))


def test_format_cannot_be_given_for_a_bigbed_whatever_its_name(bigbed_db, tmp_path):
    with pytest.raises(ValueError, match="a bigBed names its columns in its autoSql"):
        pm.gintervals_import_bed(FIXTURES / "bed12.bb", format="bed")
    f = tmp_path / "peaks.narrowPeak"
    shutil.copy(FIXTURES / "bed12.bb", f)
    pd.testing.assert_frame_equal(pm.gintervals_import_bed(f), pm.gintervals_import_bed(FIXTURES / "bed12.bb"))


def test_records_past_the_chromosome_end_are_clipped_as_gtrack_import_clips_them(tmp_path):
    make_db(tmp_path / "short_db", {"chr1": 10000, "chr2": 9900})
    with pytest.warns(UserWarning, match=r"1 record\(s\) running past the end of their chromosome were clipped to it, "
                                         r"and 0 record\(s\) .* the first: \(chr2, 9870, 9910\), chromosome size 9900"):
        x = pm.gintervals_import_bed(FIXTURES / "bed12.bb")
    assert x["end"].iloc[-1] == 9900


def test_a_damaged_bigbed_is_an_error(bigbed_db, tmp_path):
    data = (FIXTURES / "typed.bb").read_bytes()
    data_offset = struct.unpack_from("<Q", data, 16)[0]
    num_records = struct.unpack_from("<Q", data, data_offset)[0]
    with pytest.raises(Exception, match="records were read, but its header counts"):
        pm.gintervals_import_bed(patched(tmp_path, "typed.bb", data_offset, "<Q", num_records + 1))
    cut = tmp_path / "cut.bb"
    cut.write_bytes(data[:data_offset + 100])
    with pytest.raises(Exception, match="bigBed"):
        pm.gintervals_import_bed(cut)
    big_endian = tmp_path / "big_endian.bb"
    big_endian.write_bytes(data[3::-1] + data[4:])
    with pytest.raises(Exception, match="big-endian"):
        pm.gintervals_import_bed(big_endian)
