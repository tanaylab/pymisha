"""Port of R misha tests/testthat/test-gtrack.import-coords.R.

Read placement in gtrack_import_mappedseq: SAM POS is 1-based, reverse reads end at their CIGAR span,
and tab-delimited files are 0-based unless one_based=True. R runs on its test DB (chrom "chr1");
here the small test DB, also "chr1".
"""

import itertools
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT

_SAMTOOLS = shutil.which("samtools")
_names = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def _db(tmp_path_factory):
    root = tmp_path_factory.mktemp("coords") / "trackdb" / "test"
    shutil.copytree(TESTDB_ROOT, root)
    pm.gdb_init(str(root))
    yield root


@pytest.fixture(autouse=True)
def _init(_db):
    pm.gdb_init(str(_db))


def _chr1_end():
    ga = pm.gintervals_all()
    return int(ga.loc[ga["chrom"].astype(str) == "chr1", "end"].iloc[0])


def _iv(start, end):
    return pd.DataFrame({"chrom": ["chr1"], "start": [start], "end": [end]})


def write_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines))
    return path


def _track():
    return f"coords_test_{next(_names)}"


def import_points(file, **kwargs):
    track = _track()
    pm.gtrack_import_mappedseq(track, "coords test", str(file), remove_dups=False, **kwargs)
    r = pm.gextract(track, _iv(0, 100000), colnames=["v"])
    return dict(zip(r["start"].astype(int), r["v"].astype(float), strict=True))


def sam_line(name, flag, pos, cigar, seq_len):
    return "\t".join(map(str, [name, flag, "chr1", pos, 30, cigar, "*", 0, 0, "A" * seq_len, "*"]))


def test_sam_reads_are_placed_at_their_0_based_5_prime_base(tmp_path):
    sam = write_lines(tmp_path / "placed.sam", [
        sam_line("f1", 0, 100, "10M", 10),  # 5' base 99
        sam_line("f2", 0, 1001, "3S7M", 10),  # soft clip does not move POS: 1000
        sam_line("r1", 16, 200, "10M", 10),  # 199 + 10 - 1 = 208
        sam_line("r2", 16, 300, "3S7M", 10),  # 299 + 7 - 1 = 305
        sam_line("r3", 16, 400, "5M2D5M", 10),  # 399 + 12 - 1 = 410
        sam_line("r4", 16, 500, "5M2I3M", 10),  # 499 + 8 - 1 = 506
        sam_line("r5", 16, 600, "5M100N5M", 10),  # 599 + 110 - 1 = 708
        sam_line("r6", 16, 900, "*", 10),  # no CIGAR: unmapped, as in htslib
        sam_line("r7", 16, 1100, "2H3S5=2X1P2M", 10),  # H, S and P take no reference: 1099 + 9 - 1 = 1107
        sam_line("f3", 0, 1201, "*", 10),  # no CIGAR: unmapped
    ])
    p = import_points(sam, cols_order=None)
    assert sorted(p) == [99, 208, 305, 410, 506, 708, 1000, 1107]
    assert list(p.values()) == [1.0] * 8


def test_sam_dense_pileup_starts_at_the_5_prime_base(tmp_path):
    sam = write_lines(tmp_path / "dense.sam", [sam_line("f1", 0, 101, "10M", 10), sam_line("r1", 16, 291, "10M", 10)])
    track = _track()
    pm.gtrack_import_mappedseq(track, "dense", str(sam), cols_order=None, pileup=50, binsize=50)
    # forward [100, 150); reverse 5' base 299, so [250, 300)
    r = pm.gextract(track, _iv(0, 400), iterator=50, colnames=["v"])
    np.testing.assert_allclose(r["v"].to_numpy(dtype=float), [0, 0, 1, 0, 0, 1, 0, 0])


def test_a_read_on_the_last_base_of_a_chromosome_is_imported(tmp_path):
    chr1_end = _chr1_end()
    sam = write_lines(tmp_path / "last.sam", [sam_line("f1", 0, chr1_end, "1M", 1)])
    track = _track()
    stats = pm.gtrack_import_mappedseq(track, "last base", str(sam), cols_order=None)
    assert stats["total"]["total.mapped"] == 1
    assert pm.gextract(track, _iv(chr1_end - 10, chr1_end))["start"].tolist() == [chr1_end - 1]


def test_tab_delimited_coordinates_are_0_based_by_default_and_1_based_with_one_based(tmp_path):
    tab = write_lines(tmp_path / "coords.tsv", [
        "\t".join(["A" * 10, "chr1", "100", "F"]),
        "\t".join(["A" * 10, "chr1", "200", "R"]),
    ])
    # legacy placement: forward at the coordinate, reverse one past its 5' end (200 + 10)
    assert sorted(import_points(tab, cols_order=(1, 2, 3, 4))) == [100, 210]
    # 1-based: forward 99, reverse 199 + 10 - 1 = 208 (the same as SAM)
    assert sorted(import_points(tab, cols_order=(1, 2, 3, 4), one_based=True)) == [99, 208]


def test_one_based_is_rejected_for_fragment_files(tmp_path):
    frag = write_lines(tmp_path / "frag.bed", ["chr1\t100\t300"])
    track = _track()
    with pytest.raises(Exception, match="one_based is not used"):
        pm.gtrack_import_mappedseq(track, "x", str(frag), binsize=50, paired=True, one_based=True)
    assert not pm.gtrack_exists(track)


def test_sam_records_with_a_malformed_or_absurd_cigar_are_not_imported(tmp_path):
    cigars = ["10S", "5I5S", "0M", "10Q", "M10", "10M5", "10m", "9223372036854775807M"]
    sam = write_lines(tmp_path / "cigar.sam", [
        *[sam_line(f"r{i + 1}", 16, 100, cigar, 10) for i, cigar in enumerate(cigars)],
        sam_line("f1", 0, 200, "10Q", 10),
        sam_line("ok", 16, 300, "10M", 10),
    ])
    track = _track()
    stats = pm.gtrack_import_mappedseq(track, "bad cigar", str(sam), cols_order=None)
    assert stats["total"]["total.mapped"] == 1
    assert stats["total"]["total.unmapped"] == len(cigars) + 1
    assert pm.gextract(track, _iv(0, 1000))["start"].tolist() == [308]


def test_a_tab_delimited_reverse_read_ending_at_the_chromosome_end_is_not_written_past_it(tmp_path):
    chr1_end = _chr1_end()
    tab = write_lines(tmp_path / "end.tsv", ["\t".join(["A" * 10, "chr1", str(chr1_end - 10), "R"])])
    track = _track()
    # legacy sparse: the point would be chr1_end itself
    with pytest.warns(UserWarning, match="No read was imported"):
        stats = pm.gtrack_import_mappedseq(track, "end", str(tab), cols_order=(1, 2, 3, 4))
    assert stats["total"]["total.mapped"] == 0
    r = pm.gextract(track, _iv(chr1_end - 100, chr1_end))
    assert r is None or len(r) == 0


def test_reverse_reads_with_the_same_5_prime_end_are_duplicates_whatever_their_pos(tmp_path):
    sam = write_lines(tmp_path / "dups.sam", [
        sam_line("a", 16, 101, "10M", 10),  # 5' base 109
        sam_line("b", 16, 103, "8M", 8),  # 102 + 8 - 1 = 109
        sam_line("c", 16, 101, "8M2S", 10),  # soft clip at the 5' end: 100 + 8 - 1 = 107
    ])
    track = _track()
    stats = pm.gtrack_import_mappedseq(track, "dups", str(sam), cols_order=None, remove_dups=True)
    assert stats["total"]["total.dups"] == 1
    assert pm.gextract(track, _iv(0, 1000))["start"].tolist() == [107, 109]


@pytest.mark.skipif(_SAMTOOLS is None, reason="samtools not on PATH")
def test_sam_and_bam_place_reads_the_same_way(tmp_path):
    lines = [
        "@SQ\tSN:chr1\tLN:500000",
        sam_line("f1", 0, 100, "10M", 10), sam_line("r1", 16, 200, "3S7M", 10),
        sam_line("r2", 16, 400, "5M2D5M", 10), sam_line("r3", 16, 600, "5M100N5M", 10),
    ]
    sam = write_lines(tmp_path / "same.sam", lines)
    bam = tmp_path / "same.bam"
    subprocess.run([_SAMTOOLS, "view", "-b", "-o", str(bam), str(sam)], check=True, capture_output=True)
    assert import_points(bam) == import_points(sam, cols_order=None)


def test_paired_a_first_mate_on_the_last_base_is_imported_and_pos_0_is_unmapped(tmp_path):
    chr1_end = _chr1_end()

    def pair(name, pos, pnext, tlen, flag1=83, flag2=163):
        return [
            "\t".join(map(str, [name, flag1, "chr1", pos, 30, "1M", "=", pnext, -tlen, "A", "*"])),
            "\t".join(map(str, [name, flag2, "chr1", pnext, 30, "10M", "=", pos, tlen, "A" * 10, "*"])),
        ]

    sam = write_lines(tmp_path / "last_pair.sam", [
        *pair("last", chr1_end, chr1_end - 49, 50),  # fragment [chr1_end - 50, chr1_end)
        "\t".join(map(str, ["zero", 67, "chr1", 0, 30, "10M", "=", 100, 0, "A" * 10, "*"])),
    ])
    track = _track()
    stats = pm.gtrack_import_mappedseq(track, "last", str(sam), cols_order=None, paired=True, binsize=50)
    assert stats["total"]["total.mapped"] == 1
    assert stats["total"]["total.unmapped"] == 1
    last_bin = (chr1_end - 50) // 50 * 50
    v = pm.gextract(track, _iv(last_bin - 50, chr1_end), iterator=50, colnames=["v"])["v"].to_numpy(dtype=float)
    assert v.sum() * 50 == pytest.approx(50, rel=1e-5)
