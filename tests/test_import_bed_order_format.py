"""gintervals_import_bed(keep_order=, format=): the file's row order, and the
extra narrowPeak / broadPeak columns, as R misha. Expected values come from a
pandas.read_csv parse of the same lines, not from the importer."""

import gzip
import io
import re

import pandas as pd
import pytest

import pymisha as pm

# a narrowPeak body: unsorted, with a -1 (no summit) row and a strand column
NARROWPEAK_LINES = [
    "chr2\t5000\t5600\tp1\t310\t.\t7.25\t20.5\t18.125\t260",
    "chr1\t9000\t9400\tp2\t1000\t+\t11.5\t31.75\t29.5\t180",
    "chr1\t2000\t2300\tp3\t55\t-\t3.5\t4.25\t2.125\t-1",
    "chr1\t2000\t2200\tp4\t70\t.\t5\t6\t1.5\t0",
]


def _write(path, lines, gz=False):
    text = "\n".join(lines) + "\n"
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def _reference(lines):
    """The lines (header and comment lines skipped) parsed independently of pymisha."""
    data = [ln for ln in lines if not re.match(r"^(track|browser|#|$)", ln)]
    return pd.read_csv(io.StringIO("\n".join(data)), sep="\t", header=None, keep_default_na=False)


def test_keep_order_returns_file_order_header_lines_skipped(tmp_path):
    lines = [
        'track name="peaks"',
        "# a comment",
        "chr2\t500\t600\tb\t2\t+",
        "browser position chr1:1-1000",
        "chr1\t300\t400\tc\t3\t-",
        "chr1\t100\t200\ta\t1\t.",
    ]
    ref = _reference(lines)
    f = _write(tmp_path / "x.bed", lines)

    out = pm.gintervals_import_bed(f, keep_order=True)
    assert list(out["chrom"]) == list(ref[0])
    assert list(out["start"]) == list(ref[1])
    assert list(out["end"]) == list(ref[2])
    assert list(out["name"]) == list(ref[3])
    assert list(out["score"]) == list(ref[4].astype(float))
    assert list(out["strand"]) == [1, -1, 0]
    assert list(out.index) == list(range(len(ref)))

    # the default is unchanged: sorted by chrom, then start
    srt = pm.gintervals_import_bed(f)
    ref_sorted = ref.sort_values([0, 1], kind="mergesort")
    assert list(srt["name"]) == list(ref_sorted[3])
    assert list(srt.index) == list(range(len(ref)))


def test_keep_order_must_be_a_flag(tmp_path):
    f = _write(tmp_path / "x.bed", ["chr1\t100\t200"])
    with pytest.raises(ValueError, match="keep_order must be True or False"):
        pm.gintervals_import_bed(f, keep_order=float("nan"))


def test_narrowpeak_adds_columns_after_bed_columns(tmp_path):
    ref = _reference(NARROWPEAK_LINES)
    f = _write(tmp_path / "x.narrowPeak", NARROWPEAK_LINES)

    out = pm.gintervals_import_bed(f, keep_order=True)
    assert list(out.columns) == [
        "chrom", "start", "end", "strand", "name", "score", "signalValue", "pValue", "qValue", "peak",
    ]
    assert list(out["name"]) == list(ref[3])
    assert list(out["signalValue"]) == list(ref[6])
    assert list(out["pValue"]) == list(ref[7])
    assert list(out["qValue"]) == list(ref[8])
    assert out["peak"].dtype == "int64"
    assert list(out["peak"]) == list(ref[9])
    # -1 (no summit) is kept as the file has it
    assert out.loc[out["name"] == "p3", "peak"].item() == -1

    # sorted, the format columns move with their rows
    srt = pm.gintervals_import_bed(f)
    ref_sorted = ref.sort_values([0, 1, 2], kind="mergesort")
    assert list(srt["name"]) == list(ref_sorted[3])
    assert list(srt["peak"]) == list(ref_sorted[9])
    assert list(srt["qValue"]) == list(ref_sorted[8])

    # name/score/strand still control columns 4-6 only
    bare = pm.gintervals_import_bed(f, name=False, score=False, strand=False, keep_order=True)
    assert list(bare.columns) == ["chrom", "start", "end", "signalValue", "pValue", "qValue", "peak"]


def test_format_guessed_from_file_name(tmp_path):
    ref = _reference(NARROWPEAK_LINES)
    broad_lines = ["\t".join(ln.split("\t")[:9]) for ln in NARROWPEAK_LINES]
    np_gz = _write(tmp_path / "x.narrowPeak.gz", NARROWPEAK_LINES, gz=True)
    np_upper = _write(tmp_path / "y.NARROWPEAK", NARROWPEAK_LINES)
    bp = _write(tmp_path / "x.broadPeak", broad_lines)
    bp_gz = _write(tmp_path / "x.broadPeak.gz", broad_lines, gz=True)
    bed = _write(tmp_path / "x.bed", NARROWPEAK_LINES)
    bed3 = _write(tmp_path / "three.bed", ["chr1\t300\t400", "chr1\t100\t200"])

    # plain BED needs only 3 columns, sorted or not
    assert list(pm.gintervals_import_bed(bed3, keep_order=True)["start"]) == [300, 100]
    assert list(pm.gintervals_import_bed(bed3)["start"]) == [100, 300]

    assert list(pm.gintervals_import_bed(np_gz, keep_order=True)["peak"]) == list(ref[9])
    assert list(pm.gintervals_import_bed(np_upper, keep_order=True)["peak"]) == list(ref[9])

    for f in (bp, bp_gz):
        out = pm.gintervals_import_bed(f, keep_order=True)
        assert list(out.columns) == ["chrom", "start", "end", "strand", "name", "score", "signalValue", "pValue",
                                     "qValue"]
        assert list(out["signalValue"]) == list(ref[6])
        assert list(out["pValue"]) == list(ref[7])
        assert list(out["qValue"]) == list(ref[8])

    # a .bed name reads BED6 even with 10 columns, unless format says otherwise
    bed6 = ["chrom", "start", "end", "strand", "name", "score"]
    assert list(pm.gintervals_import_bed(bed).columns) == bed6
    assert list(pm.gintervals_import_bed(bed, format="narrowPeak", keep_order=True)["peak"]) == list(ref[9])
    assert list(pm.gintervals_import_bed(np_upper, format="bed").columns) == bed6


def test_format_errors(tmp_path):
    f6 = _write(tmp_path / "six.narrowPeak", ["chr1\t100\t200\tn\t1\t+"])
    f9 = _write(tmp_path / "nine.bed", ["chr1\t100\t200\tn\t1\t+\t1\t2\t3"])
    frac = _write(tmp_path / "frac.narrowPeak",
                  ["chr1\t100\t200\tn\t1\t+\t1\t2\t3\t10", "chr1\t300\t400\tm\t1\t+\t1\t2\t3\t12.5"])

    with pytest.raises(ValueError, match=re.escape('format must be one of "bed", "narrowPeak", "broadPeak"')):
        pm.gintervals_import_bed(f9, format="narrowpeak")
    with pytest.raises(ValueError, match=re.escape(f'BED file {f6} has 6 columns, but format "narrowPeak" needs 10')):
        pm.gintervals_import_bed(f6)
    with pytest.raises(ValueError, match=re.escape(f'BED file {f9} has 9 columns, but format "narrowPeak" needs 10')):
        pm.gintervals_import_bed(f9, format="narrowPeak")
    with pytest.raises(ValueError, match=re.escape(f'BED file {f6} has 6 columns, but format "broadPeak" needs 9')):
        pm.gintervals_import_bed(f6, format="broadPeak")
    with pytest.raises(ValueError, match=re.escape('got "12.5" at data row 2')):
        pm.gintervals_import_bed(frac)


def test_documented_summit_window_has_summit_at_centre(tmp_path):
    ref = _reference(NARROWPEAK_LINES)
    f = _write(tmp_path / "x.narrowPeak", NARROWPEAK_LINES)

    peaks = pm.gintervals_import_bed(f, keep_order=True)
    peaks = peaks[peaks["peak"] >= 0]
    summit = peaks["start"] + peaks["peak"]
    windows = peaks.assign(start=summit - 100, end=summit + 100, peak=100)

    has = ref[ref[9] >= 0]
    expected = list(has[1] + has[9])
    assert list((windows["start"] + windows["end"]) // 2) == expected
    assert list(windows["start"] + windows["peak"]) == expected
    assert list(windows["name"]) == list(has[3])
    # and the summit base is base 101 of the window's sequence
    summit_base = pm.gseq_extract(pd.DataFrame({"chrom": list(has[0]), "start": expected,
                                                "end": [s + 1 for s in expected]}))
    window_seqs = pm.gseq_extract(windows[["chrom", "start", "end"]])
    assert [s[100] for s in window_seqs] == list(summit_base)
