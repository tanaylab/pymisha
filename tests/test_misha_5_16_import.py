"""R misha 5.16.0 track import and creation fixes (test-review-2026-11-import.R): fend id
checks of gtrack_2d_import_contacts, the report of unknown chromosomes and the errors when
nothing is imported (2D, BED, array imports), BED records clipped to the chromosome,
mappedseq chromosome aliases, WIG blocks that resume after an unknown chromosome, the
interval checks of gtrack_create_dense, the array import in the indexed format,
gtrack_export_bedgraph over overlapping intervals, and the array slice functions.

Each test runs on a small database of its own (chr1, chr2, chrX, chrM), made by R misha's
gdb.create as the R tests' import_test_db() makes it (tests/fixtures/misha_5_16_import).
"""

import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _shared

DF = pd.DataFrame
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "misha_5_16_import"
HEADER_2D = "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"


@pytest.fixture()
def import_db(tmp_path):
    """Make a copy of the R-made database in the given format the root, as import_test_db()."""
    def make(fmt="indexed"):
        db = tmp_path / f"db_{fmt}"
        shutil.copytree(_FIXTURES / fmt, db)
        (db / "tracks").mkdir()
        pm.gdb_init(str(db))
        return db
    prev = dict(_shared.CONFIG)
    yield make
    _shared.CONFIG.update({k: prev[k] for k in ("multitasking", "max_processes")})


@pytest.fixture()
def write_file(tmp_path):
    count = iter(range(1000))

    def write(text, ext):
        p = tmp_path / f"f{next(count)}{ext}"
        p.write_text(text)
        return str(p)
    return write


def _messages(caplog):
    return " ".join(r.getMessage() for r in caplog.records)


def _pairs(res, *cols):
    return [" ".join(str(v) for v in row) for row in res[list(cols)].itertuples(index=False)]


@pytest.mark.parametrize("fmt", ["indexed", "per-chromosome"])
def test_tab_delimited_file_without_trailing_newline_keeps_its_last_line(import_db, write_file, fmt):
    import_db(fmt)
    f = write_file("chrom\tstart\tend\tvalue\nchr1\t0\t10\t1\nchr1\t20\t30\t2\nchrX\t5\t9\t3", ".tsv")
    pm.gtrack_import("nonl", "", f, 0)
    res = pm.gextract("nonl", pm.gintervals_all())
    assert res["chrom"].tolist() == ["chr1", "chr1", "chrX"]
    assert res["start"].tolist() == [0, 20, 5]
    assert res["nonl"].tolist() == [1, 2, 3]

    f = write_file("chrom\tstart\tend\ta\tb\nchr1\t0\t10\t1\t2\nchr1\t20\t30\t3\t4", ".tsv")
    pm.gtrack_array_import("arr_nonl", "", f)
    res = pm.gtrack_array_extract("arr_nonl", None, pm.gintervals_all())
    assert res["start"].tolist() == [0, 20]
    assert res["a"].tolist() == [1, 3]
    assert res["b"].tolist() == [2, 4]


def test_import_contacts_reads_a_fends_file_not_sorted_by_fend_id(import_db, write_file):
    import_db()
    fends = write_file("fend\tchr\tcoord\n10\tchr1\t1000\n5\tchr1\t2000\n7\tchr2\t300\n", ".tsv")
    contacts = write_file("fend1\tfend2\tcount\n10\t5\t3\n10\t7\t4\n5\t7\t1\n", ".tsv")
    pm.gtrack_2d_import_contacts("unsorted_fends", "", contacts, fends)
    res = pm.gextract("unsorted_fends", pm.gintervals_2d_all())
    got = DF({"chrom1": res["chrom1"].astype(str), "start1": res["start1"], "chrom2": res["chrom2"].astype(str),
                  "start2": res["start2"], "value": res["unsorted_fends"]})
    got = got.sort_values(list(got.columns)).reset_index(drop=True)
    assert got.values.tolist() == [
        ["chr1", 1000, "chr1", 2000, 3], ["chr1", 1000, "chr2", 300, 4], ["chr1", 2000, "chr1", 1000, 3],
        ["chr1", 2000, "chr2", 300, 1], ["chr2", 300, "chr1", 1000, 4], ["chr2", 300, "chr1", 2000, 1]]

    neg = write_file("fend\tchr\tcoord\n-1\tchr1\t1000\n2\tchr1\t2000\n", ".tsv")
    with pytest.raises(ValueError, match="line 2: fend -1 is out of range \\(0 - 2147483647\\)"):
        pm.gtrack_2d_import_contacts("neg_fend", "", contacts, neg)
    assert not pm.gtrack_exists("neg_fend")


def test_create_dense_validates_the_intervals_as_create_sparse_does(import_db):
    import_db()
    with pytest.raises(ValueError, match="start coordinate must be greater or equal than zero"):
        pm.gtrack_create_dense("neg", "", DF({"chrom": "chr1", "start": [-10, 20], "end": [50, 60]}), [1, 3], 100)
    with pytest.raises(ValueError, match="start coordinate must be lesser than end coordinate"):
        pm.gtrack_create_dense("rev", "", DF({"chrom": "chr1", "start": [50, 20], "end": [10, 60]}), [1, 3], 100)
    with pytest.raises(ValueError, match="Invalid format of interval at index 1"):
        pm.gtrack_create_dense("na", "", DF({"chrom": "chr1", "start": [np.nan, 20], "end": [10, 60]}), [1, 3], 100)
    with pytest.raises(ValueError, match="end coordinate exceeds chromosome boundaries"):
        pm.gtrack_create_dense("past_end", "", DF({"chrom": ["chr1"], "start": [9990], "end": [10010]}), [1], 100)
    assert not any(pm.gtrack_exists(t) for t in ["neg", "rev", "na", "past_end"])

    # aliases still resolve
    pm.gtrack_create_dense("alias", "", DF({"chrom": ["1", "MT"], "start": [0, 0], "end": [50, 100]}), [2, 5], 100)
    assert pm.gextract("alias", pm.gintervals(["chr1", "chrM"], 0, 100))["alias"].tolist() == [2, 5]


@pytest.mark.parametrize("chrom,start,end,msg", [
    (["chrZZ", "chr1"], [0, 50], [10, 10], 'Chromosome "chrZZ" does not exist'),
    (["chr1", "chrZZ"], [0, 0], [20000, 10], "Interval \\(chr1, 0, 20000\\): end coordinate exceeds"),
    (["chr1", "chr1"], [50, np.nan], [10, 10], "Interval \\(chr1, 50, 10\\): start coordinate must be lesser"),
    (["chr1", "chrZZ"], [np.nan, 0], [10, 10], "Invalid format of interval at index 1"),
    (["chrZZ", "chr1"], [0, np.nan], [10, 10], 'Chromosome "chrZZ" does not exist'),
])
def test_creators_check_the_rows_in_turn(import_db, chrom, start, end, msg):
    """Not in the R test: misha checks the rows in turn, the chromosome of a row before its
    coordinates (R 5.16 gives these errors)."""
    import_db()
    iv = DF({"chrom": chrom, "start": start, "end": end})
    with pytest.raises(ValueError, match=msg):
        pm.gtrack_create_sparse("bad", "", iv, [1, 2])
    with pytest.raises(ValueError, match=msg):
        pm.gtrack_create_dense("bad", "", iv, [1, 2], 10)
    assert not pm.gtrack_exists("bad")

    # the coordinates are cast to integers before they are checked
    pm.gtrack_create_sparse("frac", "", DF({"chrom": ["chr1"], "start": [-0.5], "end": [10.7]}), [1])
    assert _pairs(pm.gextract("frac", pm.gintervals_all()), "chrom", "start", "end") == ["chr1 0 10"]


def test_import_mappedseq_resolves_chromosome_aliases(import_db, write_file):
    import_db()
    sam = write_file(
        "@HD\tVN:1.6\n"
        "r1\t0\t1\t101\t60\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
        "r2\t0\tMT\t11\t60\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
        "r3\t0\tchr1\t501\t60\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
        "r4\t0\tchrBogus\t1\t60\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n", ".sam")
    stats = pm.gtrack_import_mappedseq("ms_alias", "", sam, cols_order=None)
    assert stats["total"]["total.mapped"] == 3
    assert stats["total"]["total.unmapped"] == 1
    res = pm.gextract("ms_alias", pm.gintervals_all())
    assert res["chrom"].tolist() == ["chr1", "chr1", "chrM"]
    assert res["start"].tolist() == [100, 500, 10]
    assert res["ms_alias"].tolist() == [1, 1, 1]


def test_wig_block_resuming_after_an_unknown_chromosome_is_an_error(import_db, write_file, caplog):
    import_db()
    bg = write_file("chr1\t0\t10\t1\nchrZZ\t0\t10\t9\nchr1\t20\t30\t3\n", ".bedgraph")
    with pytest.raises(ValueError, match="not sorted by chromosomes"):
        pm.gtrack_import("resume_bg", "", bg, 0)
    assert not pm.gtrack_exists("resume_bg")

    wig = write_file("variableStep chrom=chr1\n1\t1\nvariableStep chrom=chrZZ\n1\t9\nvariableStep chrom=chr1\n21\t3\n",
                     ".wig")
    with pytest.raises(ValueError, match="not sorted by chromosomes"):
        pm.gtrack_import("resume_wig", "", wig, 0)

    # the sibling case: a known chromosome in the middle
    bg2 = write_file("chr1\t0\t10\t1\nchr2\t0\t10\t9\nchr1\t20\t30\t3\n", ".bedgraph")
    with pytest.raises(ValueError, match="not sorted by chromosomes"):
        pm.gtrack_import("resume_bg2", "", bg2, 0)

    # an unknown block between two different chromosomes is still skipped
    bg3 = write_file("chr1\t0\t10\t1\nchrZZ\t0\t10\t9\nchr2\t20\t30\t3\n", ".bedgraph")
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_import("skip_bg", "", bg3, 0)
    assert "chrZZ" in _messages(caplog)
    res = pm.gextract("skip_bg", pm.gintervals_all())
    assert res["chrom"].tolist() == ["chr1", "chr2"]
    assert res["skip_bg"].tolist() == [1, 3]


def test_bedgraph_and_wig_import_skip_ucsc_browser_lines(import_db, write_file):
    import_db()
    bg = write_file("browser position chr1:1-100\nbrowser hide all\ntrack type=bedGraph\nchr1\t0\t10\t1\n"
                    "chr1\t20\t30\t2\n", ".bedgraph")
    pm.gtrack_import("browser_bg", "", bg, 0)
    res = pm.gextract("browser_bg", pm.gintervals_all())
    assert res["start"].tolist() == [0, 20]
    assert res["browser_bg"].tolist() == [1, 2]

    wig = write_file("browser position chr1:1-100\ntrack type=wiggle_0\nfixedStep chrom=chr1 start=1 step=10 span=10\n"
                     "4\n5\n", ".wig")
    pm.gtrack_import("browser_wig", "", wig, 0)
    res = pm.gextract("browser_wig", pm.gintervals_all())
    assert res["start"].tolist() == [0, 10]
    assert res["browser_wig"].tolist() == [4, 5]


def test_bed_import_skips_and_reports_chromosomes_missing_from_the_database(import_db, write_file, caplog):
    import_db()
    bed = write_file("chr1\t0\t10\nchrUn_zzz\t0\t10\nchrX\t5\t9\n", ".bed")
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_import("bed_sparse", "", bed, 0)
    assert "skipped: chrUn_zzz" in _messages(caplog)
    res = pm.gextract("bed_sparse", pm.gintervals_all())
    assert res["chrom"].tolist() == ["chr1", "chrX"]
    assert res["start"].tolist() == [0, 5]
    assert res["bed_sparse"].tolist() == [1, 1]

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_import("bed_dense", "", bed, 10)
    assert "skipped: chrUn_zzz" in _messages(caplog)
    assert pm.gextract("bed_dense", pm.gintervals(["chr1", "chrX"], 0, 10))["bed_dense"].tolist() == [1, 1]

    # a primary-looking name gets the warning of the other formats
    bed7 = write_file("chr1\t0\t10\nchr7\t0\t10\n", ".bed")
    with pytest.warns(UserWarning, match="primary chromosome"):
        pm.gtrack_import("bed_chr7", "", bed7, 0)

    # nothing matched: an error, and no track
    none = write_file("chrUn_a\t0\t10\nchrUn_b\t0\t10\n", ".bed")
    with pytest.raises(ValueError, match="None of the chromosome names"):
        pm.gtrack_import("bed_none", "", none, 0)
    assert not pm.gtrack_exists("bed_none")


@pytest.mark.parametrize("fmt", ["indexed", "per-chromosome"])
def test_2d_imports_skip_and_report_unknown_chromosomes_and_fail_when_nothing_matches(import_db, write_file,
                                                                                      caplog, fmt):
    import_db(fmt)
    f = write_file(HEADER_2D + "1\t0\t10\tMT\t0\t10\t1\nchrBogus\t0\t10\tchr1\t0\t10\t2\n"
                   "chr2\t0\t10\tX\t0\t10\t3\n", ".tsv")
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_2d_import("rects_unknown", "", f)
    assert "skipped: chrBogus" in _messages(caplog)
    res = pm.gextract("rects_unknown", pm.gintervals_2d_all())
    assert _pairs(res, "chrom1", "chrom2") == ["chr1 chrM", "chr2 chrX"]
    assert res["rects_unknown"].tolist() == [1, 3]

    # the names of every reading process are reported
    f2 = write_file(HEADER_2D + "chr1\t20\t30\tchr2\t20\t30\t4\nchrUn_b\t0\t10\tchr1\t0\t10\t5\n", ".tsv")
    pm.CONFIG["multitasking"] = True
    pm.CONFIG["max_processes"] = 2
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_2d_import("rects_two", "", [f, f2])
    assert "chrBogus, chrUn_b" in _messages(caplog)
    assert len(pm.gextract("rects_two", pm.gintervals_2d_all())) == 3

    none = write_file(HEADER_2D + "chrBogus\t0\t10\tchrUn_q\t0\t10\t1\n", ".tsv")
    with pytest.raises(ValueError, match="None of the chromosome names"):
        pm.gtrack_2d_import("rects_none", "", none)
    assert not pm.gtrack_exists("rects_none")

    empty = write_file(HEADER_2D, ".tsv")
    with pytest.raises(ValueError, match="No intervals were read"):
        pm.gtrack_2d_import("rects_empty", "", empty)
    assert not pm.gtrack_exists("rects_empty")

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_2d_import_contacts("contacts_unknown", "", f)
    assert "skipped: chrBogus" in _messages(caplog)
    assert len(pm.gextract("contacts_unknown", pm.gintervals_2d_all())) == 4
    with pytest.raises(ValueError, match="None of the chromosome names"):
        pm.gtrack_2d_import_contacts("contacts_none", "", none)
    with pytest.raises(ValueError, match="No contacts were read"):
        pm.gtrack_2d_import_contacts("contacts_empty", "", empty)
    assert not pm.gtrack_exists("contacts_none")
    assert not pm.gtrack_exists("contacts_empty")

    # an unknown chromosome in a fends file: its fends are skipped
    fends = write_file("fend\tchr\tcoord\n1\tchr1\t1000\n2\tchrUn_x\t5\n3\tchr2\t300\n", ".tsv")
    contacts = write_file("fend1\tfend2\tcount\n1\t3\t4\n1\t2\t7\n", ".tsv")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_2d_import_contacts("fends_unknown", "", contacts, fends)
    assert "skipped: chrUn_x" in _messages(caplog)
    res = pm.gextract("fends_unknown", pm.gintervals_2d_all())
    assert _pairs(res, "chrom1", "start1", "chrom2", "start2") == ["chr1 1000 chr2 300", "chr2 300 chr1 1000"]
    assert res["fends_unknown"].tolist() == [4, 4]


def test_array_import_skips_and_reports_unknown_chromosomes_and_fails_when_nothing_matches(import_db, write_file,
                                                                                          caplog):
    import_db()
    f = write_file("chrom\tstart\tend\ta\nchrBogus\t0\t10\t1\nchr1\t0\t10\t2\n", ".tsv")
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_array_import("arr_unknown", "", f)
    assert "skipped: chrBogus" in _messages(caplog)
    res = pm.gtrack_array_extract("arr_unknown", None, pm.gintervals_all())
    assert res["chrom"].tolist() == ["chr1"]
    assert res["a"].tolist() == [2]

    none = write_file("chrom\tstart\tend\ta\nchrBogus\t0\t10\t1\n", ".tsv")
    with pytest.raises(ValueError, match="None of the chromosome names"):
        pm.gtrack_array_import("arr_none", "", none)
    assert not pm.gtrack_exists("arr_none")


def test_array_import_primary_chromosome_and_header_only_file(import_db, write_file):
    """Not in the R test: R 5.16 warns about a primary chromosome, and a file without records
    makes an empty array track."""
    import_db()
    f = write_file("chrom\tstart\tend\ta\nchr7\t0\t10\t1\nchr1\t0\t10\t2\n", ".tsv")
    with pytest.warns(UserWarning, match="among them primary chromosome\\(s\\): chr7"):
        pm.gtrack_array_import("arr_chr7", "", f)

    f = write_file("chrom\tstart\tend\ta\n", ".tsv")
    pm.gtrack_array_import("arr_empty", "", f)
    info = pm.gtrack_info("arr_empty")
    assert (info["type"], info["format"]) == ("array", "indexed")


def test_mappedseq_and_array_import_write_the_indexed_format_on_an_indexed_database(import_db, write_file):
    db = import_db()
    sam = write_file("r1\t0\tchr1\t101\t60\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
                     "r2\t0\tchr2\t11\t60\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n", ".sam")
    pm.gtrack_import_mappedseq("ms_indexed", "", sam, cols_order=None)
    assert pm.gtrack_info("ms_indexed")["format"] == "indexed"
    files = {p.name for p in (db / "tracks" / "ms_indexed.track").iterdir() if not p.name.startswith(".")}
    assert files == {"track.dat", "track.idx"}
    res = pm.gextract("ms_indexed", pm.gintervals_all())
    assert _pairs(res, "chrom", "start") == ["chr1 100", "chr2 10"]

    f = write_file("chrom\tstart\tend\ta\tb\nchr1\t0\t10\t1\t2\nchrX\t20\t30\t3\t4\n", ".tsv")
    pm.gtrack_array_import("arr_indexed", "", f)
    assert pm.gtrack_info("arr_indexed")["format"] == "indexed"
    files = {p.name for p in (db / "tracks" / "arr_indexed.track").iterdir()}
    assert not files & {"chr1", "chr2", "chrX", "chrM"}
    res = pd.concat([pm.gtrack_array_extract("arr_indexed", None, pm.gintervals("chr1")),
                     pm.gtrack_array_extract("arr_indexed", None, pm.gintervals("chrX"))])
    assert _pairs(res, "chrom", "start") == ["chr1 0", "chrX 20"]
    assert res["a"].tolist() == [1, 3]
    assert res["b"].tolist() == [2, 4]


def _read_bedgraph(path):
    return pd.read_csv(path, sep="\t", skiprows=1, header=None, names=["chrom", "start", "end", "value"])


def test_export_bedgraph_writes_each_base_once_when_the_intervals_overlap(import_db, tmp_path):
    import_db()
    pm.gtrack_create_dense("bins", "", DF({"chrom": "chr1", "start": np.arange(0, 100, 10),
                                              "end": np.arange(10, 110, 10)}), np.arange(1, 11), 10)
    out = str(tmp_path / "out.bedgraph")
    pm.gtrack_export_bedgraph("bins", out, intervals=pm.gintervals("chr1", [0, 30], [50, 80]))
    res = _read_bedgraph(out)
    assert res["start"].tolist() == list(range(0, 80, 10))
    assert res["end"].tolist() == list(range(10, 90, 10))
    assert res["value"].tolist() == list(range(1, 9))

    # with an iterator (R 5.16: 4 records)
    pm.gtrack_export_bedgraph("bins", out, intervals=pm.gintervals("chr1", [0, 30], [50, 80]), iterator=20)
    res = _read_bedgraph(out)
    assert res["start"].tolist() == [0, 20, 40, 60]
    assert res["value"].tolist() == [1.5, 3.5, 5.5, 7.5]


def test_import_contacts_rejects_fend_ids_too_large_to_index_and_fend_ids_listed_twice(import_db, write_file):
    import_db()
    contacts = write_file("fend1\tfend2\tcount\n1\t2\t3\n", ".tsv")

    # 2147483647 itself is valid but needs ~26 GB, so it is left out
    for fend_id in ["9223372036854775807", "99999999999999999999", "2147483648"]:
        fends = write_file(f"fend\tchr\tcoord\n{fend_id}\tchr1\t100\n2\tchr1\t200\n", ".tsv")
        with pytest.raises(ValueError, match=f"fend {fend_id} is out of range"):
            pm.gtrack_2d_import_contacts("big_fend", "", contacts, fends)
        assert not pm.gtrack_exists("big_fend")

    dup = write_file("fend\tchr\tcoord\n1\tchr1\t100\n2\tchr1\t200\n1\tchr2\t300\n", ".tsv")
    with pytest.raises(ValueError, match="line 4: fend 1 appears more than once"):
        pm.gtrack_2d_import_contacts("dup_fend", "", contacts, dup)
    assert not pm.gtrack_exists("dup_fend")

    # not in the R test: an empty id
    empty_id = write_file("fend\tchr\tcoord\n\tchr1\t100\n2\tchr1\t200\n", ".tsv")
    with pytest.raises(ValueError, match="line 2: invalid fend"):
        pm.gtrack_2d_import_contacts("empty_fend", "", contacts, empty_id)
    assert not pm.gtrack_exists("empty_fend")


@pytest.mark.parametrize("binsize", [0, 10])
def test_bed_import_clips_records_past_the_chromosome_end_and_drops_empty_ones(import_db, write_file, binsize):
    import_db()
    # chrM is 500 bp long
    bed = write_file("chr1\t0\t10\nchr1\t100\t100\nchrM\t490\t520\nchrM\t600\t700\nchrM\t500\t510\n", ".bed")
    with pytest.warns(UserWarning, match="1 record\\(s\\) running past the end of their chromosome were clipped to "
                                         "it, and 3 empty record\\(s\\)"):
        pm.gtrack_import("bed_fit", "", bed, binsize)
    res = pm.gextract("bed_fit", pm.gintervals(["chr1", "chrM"], [0, 490], [10, 500]))
    assert res["bed_fit"].tolist() == [1, 1]
    if binsize == 0:
        res = pm.gextract("bed_fit", pm.gintervals_all())
        assert _pairs(res, "chrom", "start", "end") == ["chr1 0 10", "chrM 490 500"]

    # a BED file that needs neither: no warning
    ok = write_file("chr1\t0\t10\n", ".bed")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pm.gtrack_import("bed_ok", "", ok, binsize)

    # the creators called directly stay strict
    with pytest.raises(ValueError, match="end coordinate exceeds chromosome boundaries"):
        pm.gtrack_create_sparse("strict", "", DF({"chrom": ["chrM"], "start": [490], "end": [520]}), [1])


def test_export_bedgraph_exports_over_a_big_intervals_set(import_db, tmp_path):
    db = import_db()
    pm.gtrack_create_dense("bins_big", "", DF({"chrom": "chr1", "start": np.arange(0, 100, 10),
                                                  "end": np.arange(10, 110, 10)}), np.arange(1, 11), 10)
    # the set R misha saved with gmax.data.size = 2: chr1 [0, 50), [30, 80) and chr2 [0, 20)
    shutil.copytree(_FIXTURES / "big_export_set.interv", db / "tracks" / "big_export_set.interv")
    pm.gdb_reload()
    assert pm.gintervals_is_bigset("big_export_set")

    out = str(tmp_path / "out.bedgraph")
    pm.gtrack_export_bedgraph("bins_big", out, intervals="big_export_set")
    res = _read_bedgraph(out)
    assert res["chrom"].tolist() == ["chr1"] * 8
    assert res["start"].tolist() == list(range(0, 80, 10))
    assert res["value"].tolist() == list(range(1, 9))


def test_2d_imports_fail_when_no_record_has_both_chromosomes_in_the_database(import_db, write_file):
    import_db()
    f = write_file(HEADER_2D + "chrBogus\t0\t10\tchr1\t0\t10\t1\n", ".tsv")
    with pytest.raises(ValueError, match="No interval of .* has both of its chromosomes in the genome database"):
        pm.gtrack_2d_import("one_sided", "", f)
    with pytest.raises(ValueError, match="No contact of .* has both of its chromosomes in the genome database"):
        pm.gtrack_2d_import_contacts("one_sided", "", f)
    assert not pm.gtrack_exists("one_sided")

    # fends on known chromosomes, but no contact between two listed fends
    fends = write_file("fend\tchr\tcoord\n1\tchr1\t100\n2\tchrUn_x\t5\n", ".tsv")
    contacts = write_file("fend1\tfend2\tcount\n1\t2\t3\n1\t7\t1\n", ".tsv")
    with pytest.raises(ValueError, match="No contact of .* could be placed"):
        pm.gtrack_2d_import_contacts("one_sided", "", contacts, fends)
    assert not pm.gtrack_exists("one_sided")


def test_bed_import_leaves_inverted_records_to_the_creator_and_fails_when_no_record_is_left(import_db, write_file):
    import_db()
    # chrM is 500 bp long: an inverted record past the end is an error, as an inverted one inside it
    inverted = write_file("chr1\t0\t10\nchrM\t600\t550\n", ".bed")
    with pytest.raises(ValueError, match="start coordinate must be lesser than end coordinate"):
        pm.gtrack_import("bed_inverted", "", inverted, 0)
    assert not pm.gtrack_exists("bed_inverted")

    all_empty = write_file("chr1\t5\t5\nchrM\t600\t700\n", ".bed")
    for binsize in [0, 10]:
        with pytest.raises(ValueError, match="all of its 2 record\\(s\\) are empty.*nothing to import"):
            pm.gtrack_import("bed_all_empty", "", all_empty, binsize)
        assert not pm.gtrack_exists("bed_all_empty")


def test_2d_imports_count_only_the_chromosomes_that_got_data_as_imported(import_db, write_file, caplog):
    import_db()
    f = write_file(HEADER_2D + "chr1\t0\t10\tchrUn_q\t0\t10\t1\nchr2\t0\t10\tchr2\t20\t30\t2\n", ".tsv")
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_2d_import("data_chroms", "", f)
    assert "remaining 1 chromosome(s)" in _messages(caplog)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pymisha"):
        pm.gtrack_2d_import_contacts("data_chroms_c", "", f)
    assert "remaining 1 chromosome(s)" in _messages(caplog)


def test_array_slice_takes_stddev_but_not_stdev(import_db, write_file):
    """R 5.16 knows the slice function only as "stddev"."""
    import_db()
    f = write_file("chrom\tstart\tend\ta\tb\tc\nchr1\t0\t10\t1\t2\t4\n", ".tsv")
    pm.gtrack_array_import("arr_sd", "", f)
    pm.gvtrack_create("v_sd", "arr_sd")
    try:
        pm.gvtrack_array_slice("v_sd", None, "stddev")
        assert pm.gextract("v_sd", pm.gintervals("chr1", 0, 10))["v_sd"].tolist() == pytest.approx([1.527525], abs=1e-6)
        with pytest.raises(ValueError, match="Virtual track v_sd: invalid function stdev used with a track"):
            pm.gvtrack_array_slice("v_sd", None, "stdev")
    finally:
        pm.gvtrack_rm("v_sd")
