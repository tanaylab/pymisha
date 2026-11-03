"""Port of R misha tests/testthat/test-gtrack.import-paired.R.

Paired-end fragments and SAM flag / MAPQ filtering in gtrack_import_mappedseq.
R runs on its test DB (chrom "chr1"); here the small test DB, also "chr1".
"""

import gzip
import itertools
import math
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT

_BGZIP = shutil.which("bgzip")
_SAMTOOLS = shutil.which("samtools")
_names = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def _db(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired") / "trackdb" / "test"
    shutil.copytree(TESTDB_ROOT, root)
    pm.gdb_init(str(root))
    yield root


@pytest.fixture(autouse=True)
def _init(_db):
    pm.gdb_init(str(_db))


def _chr1_end():
    ga = pm.gintervals_all()
    return int(ga.loc[ga["chrom"].astype(str) == "chr1", "end"].iloc[0])


def sam_rec(name, flag, pos, mapq, pnext, tlen, rname="chr1"):
    return "\t".join(map(str, [name, flag, rname, pos, mapq, "50M", "=", pnext, tlen, "A" * 50, "*"]))


# Fragments (0-based, half-open):
#   p1, p2 [100, 300)    duplicates of each other
#   p3     [1000, 1150)  first mate on the reverse strand
#   p4     [2000, 2150)  first mate MAPQ 5
#   p5                   not a proper pair
#   p6                   secondary alignment
#   p7                   TLEN 5000
#   p8     [6000, 6050)  both mates start at the same position
#   u1                   unmapped mate placed at its partner's position
def paired_sam_text():
    return [
        "@HD\tVN:1.6",
        "@SQ\tSN:chr1\tLN:500000",
        sam_rec("p1", 99, 101, 40, 251, 200), sam_rec("p1", 147, 251, 40, 101, -200),
        sam_rec("p2", 99, 101, 40, 251, 200), sam_rec("p2", 147, 251, 40, 101, -200),
        sam_rec("p3", 163, 1001, 40, 1101, 150), sam_rec("p3", 83, 1101, 40, 1001, -150),
        sam_rec("p4", 99, 2001, 5, 2101, 150), sam_rec("p4", 147, 2101, 40, 2001, -150),
        sam_rec("p5", 65, 3001, 40, 3051, 100), sam_rec("p5", 129, 3051, 40, 3001, -100),
        sam_rec("p6", 355, 4001, 40, 4051, 100),
        sam_rec("p7", 99, 5001, 40, 9951, 5000), sam_rec("p7", 147, 9951, 40, 5001, -5000),
        sam_rec("p8", 99, 6001, 40, 6001, 50), sam_rec("p8", 147, 6001, 40, 6001, -50),
        "\t".join(map(str, ["u1", 4, "chr1", 7001, 0, "*", "=", 7001, 0, "*", "*"])),
    ]


def default_sam_text():
    return [
        "@HD\tVN:1.6",
        "@SQ\tSN:chr1\tLN:500000",
        "r1\t0\tchr1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*",
        "r2\t16\tchr1\t200\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*",
        "r3\t4\tunknown_chrom\t1\t0\t*\t*\t0\t0\t*\t*",
    ]


def write_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines))
    return path


def paired_sam_file(tmp_path, lines=None):
    # plain SAM, so that the tests do not need samtools
    return write_lines(tmp_path / f"paired_{next(_names)}.sam", paired_sam_text() if lines is None else lines)


def make_test_bam(tmp_path, lines):
    if _SAMTOOLS is None:
        pytest.skip("samtools not on PATH")
    sam = write_lines(tmp_path / f"bam_src_{next(_names)}.sam", lines)
    bam = tmp_path / f"in_{next(_names)}.bam"
    subprocess.run([_SAMTOOLS, "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    return bam


def bgzip(path):
    subprocess.run([_BGZIP, str(path)], check=True)
    return path.parent / (path.name + ".gz")


def bins_at(track, starts, binsize=50):
    starts = np.atleast_1d(starts)
    r = pm.gextract(track, pd.DataFrame({"chrom": "chr1", "start": starts, "end": starts + binsize}),
                    iterator=binsize, colnames=["v"])
    return r["v"].to_numpy(dtype=float)


def import_tmp(file, **kwargs):
    track = f"paired_test_{next(_names)}"
    stats = pm.gtrack_import_mappedseq(track, "paired test", str(file), **kwargs)
    return track, stats


def test_paired_sam_one_fragment_per_proper_pair_from_the_first_mate(tmp_path):
    track, stats = import_tmp(paired_sam_file(tmp_path), binsize=50, cols_order=None, paired=True)

    np.testing.assert_allclose(bins_at(track, [100, 150, 200, 250]), [1] * 4)
    np.testing.assert_allclose(bins_at(track, [1000, 1050, 1100]), [1] * 3)
    np.testing.assert_allclose(bins_at(track, [2000, 2050, 2100]), [1] * 3)
    np.testing.assert_allclose(bins_at(track, 6000), [1])
    # outside the fragments: improper, secondary, too long, unmapped
    np.testing.assert_allclose(bins_at(track, [50, 300, 3000, 3050, 4000, 5000, 7000, 9950]), [0] * 8)

    assert stats["total"]["total.mapped"] == 5
    assert stats["total"]["total.dups"] == 1
    assert stats["total"]["total.unmapped"] == 1
    # first mates of p5 (improper), p6 (secondary) and p7 (too long); second mates are not counted
    assert stats["total"]["total.filtered"] == 3


def test_paired_import_keeps_duplicates_when_asked_and_filters_by_mapq_and_length(tmp_path):
    sam = paired_sam_file(tmp_path)

    track, stats = import_tmp(sam, binsize=50, cols_order=None, paired=True, remove_dups=False)
    np.testing.assert_allclose(bins_at(track, [100, 250]), [2, 2])

    track, stats = import_tmp(sam, binsize=50, cols_order=None, paired=True, min_mapq=30)
    np.testing.assert_allclose(bins_at(track, [2000, 2100]), [0, 0])
    np.testing.assert_allclose(bins_at(track, 100), [1])
    assert stats["total"]["total.filtered"] == 4

    track, stats = import_tmp(sam, binsize=50, cols_order=None, paired=True, max_fraglen=6000)
    np.testing.assert_allclose(bins_at(track, [5000, 9900]), [1, 1])
    assert stats["total"]["total.filtered"] == 2


def test_paired_import_reads_a_plain_sam_file(tmp_path):
    sam = write_lines(tmp_path / "plain.sam", paired_sam_text())
    track, _ = import_tmp(sam, binsize=50, cols_order=None, paired=True)
    np.testing.assert_allclose(bins_at(track, [100, 1000, 6000]), [1, 1, 1])


def fragment_lines():
    return [
        "# comment line",
        "chr1\t100\t300\tAAAC-1\t3",
        "chr1\t100\t300\tAAAG-1\t1",  # same span, another cell: counted again
        "chr1\t1000\t1025\tAAAC-1\t1",
        "chr1\t4000\t9000\tAAAC-1\t1",  # longer than max_fraglen
        "chrUnknown\t1\t10\tAAAC-1\t1",
    ]


def test_paired_import_reads_fragment_files_plain_gzip_and_bgzip(tmp_path):
    plain = write_lines(tmp_path / "frags.tsv", fragment_lines())
    gz = tmp_path / "frags.tsv.gz"
    with gzip.open(gz, "wt") as fh:
        fh.write("".join(line + "\n" for line in fragment_lines()))
    files = {"plain": plain, "gz": gz}
    if _BGZIP is not None:
        files["bgzip"] = bgzip(write_lines(tmp_path / "frags_bgz.tsv", fragment_lines()))

    for kind, f in files.items():
        track, stats = import_tmp(f, binsize=50, paired=True)
        np.testing.assert_allclose(bins_at(track, [100, 250, 1000, 4000]), [2, 2, 0.5, 0], err_msg=kind)
        assert stats["total"]["total.mapped"] == 3, kind
        assert stats["total"]["total.dups"] == 0, kind
        assert stats["total"]["total.unmapped"] == 1, kind
        assert stats["total"]["total.filtered"] == 1, kind


def test_paired_import_rejects_arguments_that_do_not_apply(tmp_path):
    frag = write_lines(tmp_path / "frags.tsv", fragment_lines())
    track = "paired_rejects"

    with pytest.raises(Exception, match="pileup is not used"):
        pm.gtrack_import_mappedseq(track, "x", str(frag), pileup=100, binsize=50, paired=True)
    with pytest.raises(Exception, match="binsize"):
        pm.gtrack_import_mappedseq(track, "x", str(frag), paired=True)
    with pytest.raises(Exception, match="min_mapq requires SAM or BAM"):
        pm.gtrack_import_mappedseq(track, "x", str(frag), binsize=50, paired=True, min_mapq=30)
    with pytest.raises(Exception, match="cols_order is not used"):
        pm.gtrack_import_mappedseq(track, "x", str(frag), binsize=50, paired=True, cols_order=(1, 2, 3, 4))
    assert not pm.gtrack_exists(track)


def test_single_end_sam_import_skips_unmapped_secondary_and_low_mapq_records(tmp_path):
    sam = write_lines(tmp_path / "single.sam", [
        "@SQ\tSN:chr1\tLN:500000",
        "r1\t0\tchr1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*",
        "r2\t16\tchr1\t200\t10\t10M\t*\t0\t0\tAAAAAAAAAA\t*",
        "r3\t4\tchr1\t300\t0\t*\t*\t0\t0\tAAAAAAAAAA\t*",  # unmapped, placed on chr1
        "r4\t256\tchr1\t400\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*",  # secondary
        "r5\t2048\tchr1\t500\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*",  # supplementary
    ])

    track, stats = import_tmp(sam, cols_order=None)
    assert stats["total"]["total.mapped"] == 2
    assert stats["total"]["total.unmapped"] == 1
    assert stats["total"]["total.filtered"] == 2

    track, stats = import_tmp(sam, cols_order=None, min_mapq=20)
    assert stats["total"]["total.mapped"] == 1
    assert stats["total"]["total.filtered"] == 3
    assert "min.mapq=20" in pm.gtrack_attr_get(track, "created.by")


def test_paired_import_warns_when_a_sam_file_is_read_as_fragments(tmp_path):
    sam = write_lines(tmp_path / "as_frags.sam", paired_sam_text())
    with pytest.warns(UserWarning, match="cols_order=None"):
        import_tmp(sam, binsize=50, paired=True)


@pytest.mark.skipif(_BGZIP is None, reason="bgzip not on PATH")
def test_a_bgzipped_sam_is_read_as_sam_with_the_default_cols_order(tmp_path):
    sam = bgzip(write_lines(tmp_path / "bgz.sam", default_sam_text()))
    _, stats = import_tmp(sam)
    assert stats["total"]["total.mapped"] == 2
    assert stats["total"]["total.unmapped"] == 1


def test_a_large_max_fraglen_is_recorded_without_failing_the_import(tmp_path):
    track, _ = import_tmp(paired_sam_file(tmp_path), binsize=50, cols_order=None, paired=True, max_fraglen=1e10)
    np.testing.assert_allclose(bins_at(track, [5000, 9900]), [1, 1])
    assert "max.fraglen=1e+10" in pm.gtrack_attr_get(track, "created.by")


def test_fragment_files_with_crlf_line_endings_are_read(tmp_path):
    frag = tmp_path / "crlf.bed"
    frag.write_bytes(b"chr1\t100\t300\r\nchr1\t1000\t1050\r\n")
    track, stats = import_tmp(frag, binsize=50, paired=True)
    np.testing.assert_allclose(bins_at(track, [100, 1000]), [1, 1])
    assert stats["total"]["total.unmapped"] == 0


def test_total_counts_each_record_once(tmp_path):
    _, stats = import_tmp(paired_sam_file(tmp_path), binsize=50, cols_order=None, paired=True)
    s = stats["total"]
    # first mates p1-p8 and the unmapped u1
    assert s["total"] == 9
    assert s["total"] == s["total.mapped"] + s["total.unmapped"] + s["total.filtered"]


def test_a_reverse_read_past_the_chromosome_end_does_not_overflow_the_dense_track(tmp_path):
    chr1_end = _chr1_end()
    # tab-delimited (sequence, chromosome, coordinate, strand): the read is kept and ends past the chromosome
    tab = write_lines(tmp_path / "rev.tsv", ["\t".join(["A" * 151, "chr1", str(chr1_end - 10), "R"])])
    track, stats = import_tmp(tab, cols_order=(1, 2, 3, 4), pileup=100, binsize=20)
    assert stats["total"]["total.mapped"] == 1
    r = pm.gextract(track, pd.DataFrame({"chrom": ["chr1"], "start": [chr1_end - 200], "end": [chr1_end]}),
                    iterator=20, colnames=["v"])
    assert r["v"].sum() == 0

    # a SAM read whose 5' end is past the chromosome is not imported
    sam = write_lines(tmp_path / "rev.sam", ["\t".join(map(str, [
        "r1", 16, "chr1", chr1_end - 10, 30, "151M", "*", 0, 0, "A" * 151, "*"]))])
    with pytest.warns(UserWarning, match="No read was imported"):
        _, stats = import_tmp(sam, cols_order=None, pileup=100, binsize=20)
    assert stats["total"]["total.mapped"] == 0
    assert stats["total"]["total.unmapped"] == 1


def test_a_paired_bam_gives_the_same_track_as_the_sam(tmp_path):
    bam = make_test_bam(tmp_path, paired_sam_text())
    bam_track, bam_stats = import_tmp(bam, binsize=50, paired=True)
    sam_track, sam_stats = import_tmp(paired_sam_file(tmp_path), binsize=50, cols_order=None, paired=True)
    assert bam_stats["total"] == sam_stats["total"]
    pd.testing.assert_frame_equal(bam_stats["chrom"], sam_stats["chrom"])
    starts = np.arange(0, 10001, 50)
    np.testing.assert_array_equal(bins_at(bam_track, starts), bins_at(sam_track, starts))


def test_duplicates_are_fragments_with_the_same_start_and_end_not_the_same_start(tmp_path):
    sam = paired_sam_file(tmp_path, [
        sam_rec("a", 99, 101, 40, 251, 200), sam_rec("a", 147, 251, 40, 101, -200),  # [100, 300)
        sam_rec("b", 99, 101, 40, 151, 100), sam_rec("b", 147, 151, 40, 101, -100),  # [100, 200)
    ])
    track, stats = import_tmp(sam, binsize=50, cols_order=None, paired=True)
    assert stats["total"]["total.dups"] == 0
    np.testing.assert_allclose(bins_at(track, [100, 150, 200, 250]), [2, 2, 1, 1])


def test_a_pair_record_with_neither_mate_flag_is_taken_as_the_first_mate(tmp_path):
    # FLAG 3 = paired + proper pair, no 0x40 / 0x80 (MACS3 keeps such records too)
    track, _ = import_tmp(paired_sam_file(tmp_path, [sam_rec("x", 3, 101, 40, 151, 100)]),
                          binsize=50, cols_order=None, paired=True)
    np.testing.assert_allclose(bins_at(track, [100, 150]), [1, 1])


def test_a_sam_fragment_past_the_chromosome_end_is_clipped(tmp_path):
    chr1_end = _chr1_end()
    start = (chr1_end // 50) * 50 - 100  # two full bins before the chromosome's last bin
    sam = paired_sam_file(tmp_path, [
        sam_rec("e", 99, start + 1, 40, start + 61, 500), sam_rec("e", 147, start + 61, 40, start + 1, -500),
    ])
    track, stats = import_tmp(sam, binsize=50, cols_order=None, paired=True)
    assert stats["total"]["total.mapped"] == 1
    np.testing.assert_allclose(bins_at(track, [start, start + 50]), [1, 1])


def test_fragments_past_the_chromosome_end_are_clipped(tmp_path):
    chr1_end = _chr1_end()
    start = (chr1_end // 50) * 50 - 100  # two full bins before the chromosome's last bin
    frag = write_lines(tmp_path / "past.bed", [f"chr1\t{start}\t{chr1_end + 500}"])
    track, stats = import_tmp(frag, binsize=50, paired=True)
    assert stats["total"]["total.mapped"] == 1
    np.testing.assert_allclose(bins_at(track, [start, start + 50]), [1, 1])


def test_max_fraglen_inf_means_no_limit(tmp_path):
    track, _ = import_tmp(paired_sam_file(tmp_path), binsize=50, cols_order=None, paired=True, max_fraglen=math.inf)
    np.testing.assert_allclose(bins_at(track, [5000, 9900]), [1, 1])
    assert "max.fraglen=Inf" in pm.gtrack_attr_get(track, "created.by")


@pytest.mark.skipif(_BGZIP is None, reason="bgzip not on PATH")
def test_a_single_end_import_that_brings_in_nothing_warns(tmp_path):
    frag = bgzip(write_lines(tmp_path / "nothing.tsv", fragment_lines()))
    # a bgzipped fragment file without paired=True is read as SAM
    with pytest.warns(UserWarning, match="No read was imported"):
        import_tmp(frag)


def test_the_min_mapq_error_names_cols_order_none_for_sam_files(tmp_path):
    with pytest.raises(Exception, match="cols_order=None"):
        pm.gtrack_import_mappedseq("paired_mapq_err", "x", str(paired_sam_file(tmp_path)),
                                   binsize=50, paired=True, min_mapq=30)
