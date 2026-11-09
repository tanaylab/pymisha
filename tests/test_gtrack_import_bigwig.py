"""gtrack_import reads bigWig files natively (misha's BigWig.cpp).

Port of misha's tests/testthat/test-gtrack.import-bigwig.R. The fixtures, copied
from misha, were made from bg.bedGraph and step.wig by UCSC's bedGraphToBigWig
(bg.bw with -blockSize=2 -itemsPerSlot=3, so both index trees have inner nodes
and every chromosome spans many blocks; bg_unc.bw with -unc), wigToBigWig
(step.bw: fixedStep with span < step, variableStep with span > 1) and libBigWig
through pyBigWig (bg_libbigwig.bw, which also lists a chromosome that has no
data). edge.bw (bedGraphToBigWig): chr1 100-110 12345.678, a zero-length record
at 120, and 130-140 0.000123456789.
"""

import gzip
import logging
import shutil
import struct
import warnings
import zipfile
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm

FIXTURES = Path(__file__).parent / "fixtures" / "bigwig"


def make_db(root: Path, sizes: dict[str, int]) -> None:
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


@pytest.fixture
def short_db(tmp_path):
    # chr2 ends inside step.wig's last variableStep record (9982-9989)
    make_db(tmp_path / "short_db", {"chr1": 10000, "chr2": 9985})


def extract_all(track, iterator):
    return pm.gextract(track, pm.gintervals_all(), iterator=iterator, colnames=["v"])


def import_both(text, bw, binsize, defval=np.nan, warns=None):
    """Imports the text and the bigWig; with warns, each import must warn matching it."""
    for track, f in (("from_text", text), ("from_bw", bw)):
        if warns is None:
            pm.gtrack_import(track, "", str(FIXTURES / f), binsize=binsize, defval=defval)
        else:
            with pytest.warns(UserWarning, match=warns):
                pm.gtrack_import(track, "", str(FIXTURES / f), binsize=binsize, defval=defval)
    return extract_all("from_bw", binsize or "from_bw"), extract_all("from_text", binsize or "from_text")


def bigwig_offsets(path):
    """fullDataOffset (header byte 16) and fullIndexOffset (byte 24)."""
    return struct.unpack_from("<QQ", Path(path).read_bytes(), 16)


@pytest.mark.parametrize(("bw", "text"), [
    ("bg.bw", "bg.bedGraph"), ("bg_unc.bw", "bg.bedGraph"), ("bg_libbigwig.bw", "bg.bedGraph"),
    ("step.bw", "step.wig"),
])
def test_bigwig_imports_exactly_like_its_text(bigwig_db, bw, text):
    for binsize, defval in [(0, np.nan), (7, np.nan), (7, 0)]:
        got, want = import_both(text, bw, binsize, defval)
        pd.testing.assert_frame_equal(got, want, check_exact=True, obj=f"{bw}, binsize {binsize}, defval {defval}")
        pm.gtrack_rm("from_text", force=True)
        pm.gtrack_rm("from_bw", force=True)


def test_step_records_are_clipped_at_chromosome_end(short_db):
    for binsize in (0, 7):
        got, want = import_both("step.wig", "step.bw", binsize,
                                warns=r"1 record\(s\) running past the end of their chromosome were clipped to it")
        pd.testing.assert_frame_equal(got, want, check_exact=True)
        assert got.loc[got["chrom"] == "chr2", "end"].max() == 9985
        pm.gtrack_rm("from_text", force=True)
        pm.gtrack_rm("from_bw", force=True)


def test_bedgraph_records_past_the_chromosome_end_are_clipped_or_dropped_as_in_the_text_route(tmp_path, caplog):
    # chr2 ends inside the record 9870-9910, before the record 9933-9981
    make_db(tmp_path / "short_bg_db", {"chr1": 10000, "chr2": 9900})
    msg = (r"1 record\(s\) running past the end of their chromosome were clipped to it, and 1 record\(s\) starting "
           r"at or past the chromosome end were dropped; the first: .*\(chr2, 9870, 9910\), chromosome size 9900")
    for binsize in (0, 7):
        got, want = import_both("bg.bedGraph", "bg.bw", binsize, warns=msg)
        pd.testing.assert_frame_equal(got, want, check_exact=True)
        if binsize == 0:
            assert got.loc[got["chrom"] == "chr2", "end"].iloc[-1] == 9900
        pm.gtrack_rm("from_text", force=True)
        pm.gtrack_rm("from_bw", force=True)


def test_the_bigwig_record_past_the_chromosome_end_that_the_warning_names_is_the_first_in_the_file(tmp_path):
    # chr2 first in the database, chr1 first in the files; both end inside a record
    make_db(tmp_path / "rev_db", {"chr2": 9900, "chr1": 9900})
    if pm.gintervals_all()["chrom"].astype(str).tolist() != ["chr2", "chr1"]:
        pytest.skip("the database sorts its chromosomes")
    for f in ("bg.bw", "bg.bedGraph"):
        with pytest.warns(UserWarning, match=r"the first: (line [0-9]+: )?\(chr1, 9888, 9925\)"):
            pm.gtrack_import("first", "", str(FIXTURES / f), binsize=0)
        pm.gtrack_rm("first", force=True)


def test_full_float_precision_and_zero_length_records_skipped(bigwig_db):
    pm.gtrack_import("edge", "", str(FIXTURES / "edge.bw"), binsize=0)
    x = extract_all("edge", "edge")
    assert x["start"].tolist() == [100, 130]
    assert x["end"].tolist() == [110, 140]
    # bigWigToWig printed 6 significant digits: 12345.7, 0.000123457
    np.testing.assert_array_equal(x["v"].to_numpy(np.float32), np.float32([12345.678, 0.000123456789]))


def test_unknown_chromosomes_reported_empty_ones_not(bigwig_db, caplog):
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.INFO, logger="pymisha"):
        warnings.simplefilter("always")
        pm.gtrack_import("skipped", "", str(FIXTURES / "bg_libbigwig.bw"), binsize=10)
    assert not caught, [str(w.message) for w in caught]
    assert "scaffold_9" in caplog.text
    assert "chrEmpty" not in caplog.text
    assert pm.gtrack_exists("skipped")


def test_truncated_or_corrupt_bigwig_fails_and_leaves_no_track(bigwig_db, tmp_path):
    src = FIXTURES / "bg.bw"
    data, index = bigwig_offsets(src)
    raw = src.read_bytes()

    def flipped(pos, f):
        b = bytearray(raw)
        b[pos] = f(b[pos])
        return bytes(b)

    damaged = {
        # the full-resolution index comes after the data, so either cut loses it
        "cut_in_data": raw[:data + 20],
        "cut_in_index": raw[:index + 60],
        # inside the first compressed block (after the 8-byte block count): zlib's checksum fails
        "flipped_byte": flipped(data + 18, lambda x: x ^ 0xFF),
        # the 8-byte block count at fullDataOffset no longer matches the index
        "bad_block_count": flipped(data, lambda x: (x + 1) % 256),
        # named .bw but empty, or text: not imported as a WIG
        "empty": b"",
        "text": b"chr1\t0\t100\t1\n",
    }
    for what, content in damaged.items():
        f = tmp_path / f"{what}.bw"
        f.write_bytes(content)
        with pytest.raises(_pymisha.error, match="bigWig"):
            pm.gtrack_import("damaged", "", str(f), binsize=10)
        assert not pm.gtrack_exists("damaged"), what


# ---- pymisha additions --------------------------------------------------------


def test_bigwig_recognized_by_magic_number(bigwig_db, tmp_path):
    f = tmp_path / "bg.dat"
    shutil.copyfile(FIXTURES / "bg.bw", f)
    pm.gtrack_import("from_dat", "", str(f), binsize=7)
    pm.gtrack_import("from_bw", "", str(FIXTURES / "bg.bw"), binsize=7)
    pd.testing.assert_frame_equal(extract_all("from_dat", 7), extract_all("from_bw", 7), check_exact=True)


def test_two_bigwig_chromosomes_for_one_database_chromosome(bigwig_db, tmp_path):
    # rename chr2 to 1, an alias of chr1, in the chromosome tree (keys are null-padded to 10 bytes)
    raw = (FIXTURES / "bg.bw").read_bytes()
    assert raw.count(b"chr2\0") == 1
    f = tmp_path / "alias.bw"
    f.write_bytes(raw.replace(b"chr2\0", b"1\0\0\0\0"))
    with pytest.raises(_pymisha.error, match="same database chromosome chr1"):
        pm.gtrack_import("alias", "", str(f), binsize=10)
    assert not pm.gtrack_exists("alias")


@pytest.mark.parametrize(("text", "match"), [
    ("chr1\t0\t100\t1\nchr1\t50\t150\t2\n", "overlapping or unsorted"),
    ("chr1\t200\t300\t1\nchr1\t0\t100\t2\n", "overlapping or unsorted"),
    ("variableStep chrom=chr2 span=5\n10 1\n8 2\n", "overlapping or unsorted"),
    ("fixedStep chrom=chr1 start=1 step=5 span=10\n1\n2\n", "overlapping or unsorted"),   # span > step
    ("chr1\t0\t10\t1\nchr2\t0\t10\t1\nchr1\t20\t30\t1\n", "not sorted by chromosomes"),
    ("chr1\t0\t10\t1\nchr1\t20\t20\t1\n", "invalid record"),   # empty bedGraph record
])
def test_text_records_follow_misha_rules(bigwig_db, tmp_path, text, match):
    # misha refuses each of these files too (checked against misha 319f8020)
    f = tmp_path / "bad.wig"
    f.write_text(text)
    for binsize in (0, 10):
        with pytest.raises(ValueError, match=match):
            pm.gtrack_import("bad", "", str(f), binsize=binsize)
        assert not pm.gtrack_exists("bad")


# ---- gtrack_import as misha 5.14.0 does it: mixed files, compressed files, files without records ----


def records(track):
    x = pm.gextract(track, pm.gintervals_all(), iterator=track, colnames=["v"])
    if x is None:   # no rows, as misha's NULL
        return []
    return list(zip(x["chrom"].astype(str), x["start"], x["end"], x["v"], strict=True))


@pytest.mark.parametrize(("text", "want"), [
    ("fixedStep chrom=chr1 start=1 step=10 span=5\n1\n2\nchr2\t0\t10\t3\n",
     [("chr1", 0, 5, 1), ("chr1", 10, 15, 2), ("chr2", 0, 10, 3)]),
    ("fixedStep chrom=chr1 start=1 step=10 span=5\n1\n2\nchr1\t100\t110\t3\n",
     [("chr1", 0, 5, 1), ("chr1", 10, 15, 2), ("chr1", 100, 110, 3)]),
    ("chr1\t0\t10\t3\nfixedStep chrom=chr2 start=1 step=10 span=5\n1\n2\n",
     [("chr1", 0, 10, 3), ("chr2", 0, 5, 1), ("chr2", 10, 15, 2)]),
    ("chr1\t0\t10\t3\nvariableStep chrom=chr2 span=5\n11 1\n", [("chr1", 0, 10, 3), ("chr2", 10, 15, 1)]),
    ("variableStep chrom=chr1 span=5\n11 1\nchr2\t0\t10\t3\n", [("chr1", 10, 15, 1), ("chr2", 0, 10, 3)]),
])
def test_wig_and_bedgraph_lines_mix(bigwig_db, tmp_path, text, want):
    # misha reads a bedGraph line after a fixedStep / variableStep section, and the section after it
    f = tmp_path / "mix.wig"
    f.write_text(text)
    pm.gtrack_import("mix", "", str(f), binsize=0)
    assert records("mix") == want


@pytest.mark.parametrize(("text", "header"), [
    ("variableStep chrom=chr1 span=5\n11 1\nchr2\t0\t10\t3\n4\n", "fixedStep"),
    ("fixedStep chrom=chr1 start=1 step=10\n1\nchr2\t0\t10\t3\n20 4\n", "variableStep"),
])
def test_value_after_bedgraph_line_is_an_error(bigwig_db, tmp_path, text, header):
    # not a WIG file, nor a tab-delimited one: the error gives both reasons, as misha's
    f = tmp_path / "mix.wig"
    f.write_text(text)
    with pytest.raises(ValueError, match=f"(?s)Unrecognized format.*value is not preceded by {header} header"):
        pm.gtrack_import("mix", "", str(f), binsize=0)
    assert not pm.gtrack_exists("mix")


@pytest.mark.parametrize("name", ["edge.bw.gz", "edge.bigWig.gz", "edge.bw.zip", "edge.wig.gz", "edge.gz"])
def test_compressed_bigwig_is_unzipped_and_read(bigwig_db, tmp_path, name):
    # misha unzips first and finds the format on the plain file: by its name or its first bytes
    raw = (FIXTURES / "edge.bw").read_bytes()
    f = tmp_path / name
    if name.endswith(".zip"):
        with zipfile.ZipFile(f, "w") as z:
            z.writestr("edge.bw", raw)
    else:
        f.write_bytes(gzip.compress(raw))
    pm.gtrack_import("packed", "", str(f), binsize=0)
    pm.gtrack_import("plain", "", str(FIXTURES / "edge.bw"), binsize=0)
    assert records("packed") == records("plain")


@pytest.mark.parametrize(("name", "content", "match", "err"), [
    ("text.bw.gz", gzip.compress(b"chr1\t0\t10\t1\n", mtime=0), "bigWig", _pymisha.error),   # named a bigWig, is not
    ("bad.bw.gz", b"not gzip", "Failed to unzip file", ValueError),
    ("two.bw.zip", None, "Failed to unzip file", ValueError),   # gunzip, which misha uses, reads one-file zips
])
def test_bad_compressed_files(bigwig_db, tmp_path, name, content, match, err):
    f = tmp_path / name
    if content is None:
        with zipfile.ZipFile(f, "w") as z:
            z.writestr("readme.txt", "x")
            z.writestr("edge.bw", (FIXTURES / "edge.bw").read_bytes())
    else:
        f.write_bytes(content)
    with pytest.raises(err, match=match):
        pm.gtrack_import("packed", "", str(f), binsize=10)
    assert not pm.gtrack_exists("packed")


@pytest.mark.parametrize("name", ["bg.bedGraph.zip", "bg.wig.zip", "bg.bedGraph.gz"])
def test_compressed_text_imports_like_plain(bigwig_db, tmp_path, name):
    text = (FIXTURES / "bg.bedGraph").read_bytes()
    f = tmp_path / name
    if name.endswith(".zip"):
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(name[:-4], text)
    else:
        f.write_bytes(gzip.compress(text))
    for binsize in (0, 7):
        pm.gtrack_import("packed", "", str(f), binsize=binsize)
        pm.gtrack_import("plain", "", str(FIXTURES / "bg.bedGraph"), binsize=binsize)
        pd.testing.assert_frame_equal(extract_all("packed", binsize or "packed"),
                                      extract_all("plain", binsize or "plain"), check_exact=True)
        pm.gtrack_rm("packed", force=True)
        pm.gtrack_rm("plain", force=True)


@pytest.mark.parametrize(("name", "content"), [
    ("empty.wig", b""), ("empty.bedGraph", b""), ("comments.bedGraph", b"# nothing\n"),
    ("header.wig", b"track type=wiggle_0\n"), ("empty.tsv", b""), ("header.tsv", b"chrom\tstart\tend\tvalue\n"),
    ("empty.bedGraph.gz", gzip.compress(b"", mtime=0)),
])
def test_file_without_records_imports_an_empty_track(bigwig_db, tmp_path, name, content):
    f = tmp_path / name
    f.write_bytes(content)
    pm.gtrack_import("sparse", "", str(f), binsize=0)
    assert records("sparse") == []
    for defval in (np.nan, 2.5):
        pm.gtrack_import("dense", "", str(f), binsize=10, defval=defval)
        v = extract_all("dense", 10)["v"].to_numpy()
        assert len(v) == 2000
        np.testing.assert_array_equal(v, np.full(2000, defval))
        pm.gtrack_rm("dense", force=True)


def test_records_none_of_whose_chromosomes_exist_still_fail(bigwig_db, tmp_path):
    f = tmp_path / "unknown.bedGraph"
    f.write_text("chrZ\t0\t10\t1\n")
    with pytest.raises(ValueError, match="None of the chromosome names in .* exist in the genome database"):
        pm.gtrack_import("unknown", "", str(f), binsize=10)
    f = tmp_path / "empty.bed"   # an empty BED file fails in misha too
    f.write_text("")
    with pytest.raises(ValueError, match="contains no intervals"):
        pm.gtrack_import("unknown", "", str(f), binsize=10)
