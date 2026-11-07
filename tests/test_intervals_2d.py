"""Tests for 2D interval constructors."""

import warnings

import pandas as pd
import pytest

import pymisha


def _pairs(df):
    return list(zip(df["chrom1"], df["chrom2"], strict=True))


def _scratch_db(root, chroms, db_format="indexed", size=8):
    """Create a database of `size`-bp chromosomes under root and return its path."""
    root.mkdir(parents=True, exist_ok=True)
    if db_format == "indexed":
        fasta = root / "genome.fasta"
        fasta.write_text("".join(f">{c}\n{'A' * size}\n" for c in chroms))
        fastas = [str(fasta)]
    else:
        fastas = []
        for c in chroms:
            fasta = root / f"{c}.fasta"
            fasta.write_text(f">{c}\n{'A' * size}\n")
            fastas.append(str(fasta))
    db = root / "db"
    pymisha.gdb_create(str(db), fastas, db_format=db_format)
    return db


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    """Ensure test DB is initialized."""


class TestGintervals2d:
    """Tests for gintervals_2d."""

    def test_single_chrom_defaults(self):
        """Single chrom with default starts/ends gives full-chrom 2D interval."""
        result = pymisha.gintervals_2d("1")
        assert list(result.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        assert len(result) == 1
        assert result.iloc[0]["chrom1"] == "chr1"
        assert result.iloc[0]["chrom2"] == "chr1"
        assert result.iloc[0]["start1"] == 0
        assert result.iloc[0]["start2"] == 0
        # ends should be chrom size
        assert result.iloc[0]["end1"] == 500000
        assert result.iloc[0]["end2"] == 500000

    def test_two_different_chroms(self):
        """chrom1 != chrom2."""
        result = pymisha.gintervals_2d("1", 100, 200, "2", 300, 400)
        assert len(result) == 1
        assert result.iloc[0]["chrom1"] == "chr1"
        assert result.iloc[0]["start1"] == 100
        assert result.iloc[0]["end1"] == 200
        assert result.iloc[0]["chrom2"] == "chr2"
        assert result.iloc[0]["start2"] == 300
        assert result.iloc[0]["end2"] == 400

    def test_chroms2_defaults_to_chroms1(self):
        """When chroms2 is None, it defaults to chroms1."""
        result = pymisha.gintervals_2d("X", 10, 100)
        assert result.iloc[0]["chrom1"] == "chrX"
        assert result.iloc[0]["chrom2"] == "chrX"
        assert result.iloc[0]["start2"] == 0
        assert result.iloc[0]["end2"] == 200000

    def test_multiple_intervals(self):
        """Multiple intervals created from lists."""
        result = pymisha.gintervals_2d(
            ["1", "2"], [100, 200], [300, 400],
            ["X", "1"], [50, 60], [150, 160],
        )
        assert len(result) == 2

    def test_sorted_output(self):
        """Output is sorted by chrom1, start1, chrom2, start2."""
        result = pymisha.gintervals_2d(
            ["2", "1"], [100, 200], [300, 400],
            ["1", "X"], [50, 60], [150, 160],
        )
        # chrom1="1" should come before chrom1="2"
        assert result.iloc[0]["chrom1"] == "chr1"
        assert result.iloc[1]["chrom1"] == "chr2"

    def test_scalar_broadcast(self):
        """Scalar args broadcast to match list length."""
        result = pymisha.gintervals_2d(["1", "2"], 0, 100)
        assert len(result) == 2
        assert result.iloc[0]["start1"] == 0
        assert result.iloc[0]["end1"] == 100

    def test_end_minus1_means_full_chrom(self):
        """end=-1 means full chromosome length."""
        result = pymisha.gintervals_2d("2", 0, -1, "X", 0, -1)
        assert result.iloc[0]["end1"] == 300000
        assert result.iloc[0]["end2"] == 200000

    def test_invalid_chrom_raises(self):
        """Unknown chromosome raises ValueError or pymisha.error."""
        from pymisha import _pymisha
        with pytest.raises((ValueError, _pymisha.error)):
            pymisha.gintervals_2d("nonexistent")

    def test_bad_range_raises(self):
        """start >= end raises ValueError."""
        with pytest.raises(ValueError):
            pymisha.gintervals_2d("1", 200, 100)

    def test_chrom_normalization(self):
        """chr prefix is normalized away."""
        result = pymisha.gintervals_2d("chr1", 0, 100, "chrX", 0, 50)
        assert result.iloc[0]["chrom1"] == "chr1"
        assert result.iloc[0]["chrom2"] == "chrX"


class TestGintervals2dAll:
    """Tests for gintervals_2d_all."""

    def test_returns_2d_dataframe(self):
        """Returns DataFrame with 2D interval columns."""
        result = pymisha.gintervals_2d_all()
        assert list(result.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]

    def test_default_is_all_pairs_like_r(self):
        """The default is R's gintervals.2d.all(): every chromosome pair, chrom1 then
        chrom2 in the order of gintervals_all(), each pair covering both chromosomes."""
        result = pymisha.gintervals_2d_all()
        genome = pymisha.gintervals_all()
        chroms = list(genome["chrom"])
        assert chroms == ["chr1", "chr2", "chrX"]
        assert _pairs(result) == [(a, b) for a in chroms for b in chroms]
        sizes = dict(zip(genome["chrom"], genome["end"], strict=True))
        assert (result["start1"] == 0).all() and (result["start2"] == 0).all()
        assert list(result["end1"]) == [sizes[c] for c in result["chrom1"]]
        assert list(result["end2"]) == [sizes[c] for c in result["chrom2"]]
        assert result.index.equals(pd.RangeIndex(9))
        pd.testing.assert_frame_equal(result, pymisha.gintervals_2d_all(mode="full"))

    def test_diagonal_mode(self):
        """mode="diagonal" keeps only the chrom1 == chrom2 pairs, in genome order."""
        result = pymisha.gintervals_2d_all(mode="diagonal")
        assert _pairs(result) == [("chr1", "chr1"), ("chr2", "chr2"), ("chrX", "chrX")]

    def test_default_follows_the_threshold(self, monkeypatch):
        """Above CONFIG["multicontig_2d_threshold"] chromosomes the default is the
        intra-chromosomal pairs, as R's gintervals.2d.all() above
        gmulticontig.2d.threshold."""
        monkeypatch.setitem(pymisha.CONFIG, "multicontig_2d_threshold", 2)
        with pytest.warns(UserWarning, match="returns the cis chromosome pairs only"):
            assert _pairs(pymisha.gintervals_2d_all()) == [("chr1", "chr1"), ("chr2", "chr2"), ("chrX", "chrX")]
        monkeypatch.setitem(pymisha.CONFIG, "multicontig_2d_threshold", 3)
        assert len(pymisha.gintervals_2d_all()) == 9

    def test_full_mode_warns_above_the_threshold(self, monkeypatch):
        """mode="full" above the threshold warns with R's message, and only there."""
        monkeypatch.setitem(pymisha.CONFIG, "multicontig_2d_threshold", 2)
        with pytest.warns(UserWarning, match=r"Generating full 2D genome with 3 contigs \(9 pairs\)"):
            result = pymisha.gintervals_2d_all(mode="full")
        assert len(result) == 9
        monkeypatch.setitem(pymisha.CONFIG, "multicontig_2d_threshold", 3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pymisha.gintervals_2d_all(mode="full")
            pymisha.gintervals_2d_all()

    def test_giterator_intervals_2d_default_scope_stays_intra_chromosomal(self):
        """giterator_intervals_2d (no R counterpart) keeps its cis-only default scope."""
        got = pd.concat(list(pymisha.giterator_intervals_2d("rects_track")))
        expected = pymisha.gextract("rects_track", pymisha.gintervals_2d_all(mode="diagonal"))
        assert (got["chrom1"] == got["chrom2"]).all()
        assert len(got) == len(expected) > 0

    def test_diagonal_covers_full_chroms(self):
        """Each diagonal entry covers the full chromosome."""
        result = pymisha.gintervals_2d_all(mode="diagonal")
        all_intervals = pymisha.gintervals_all()
        # all_intervals has 'chrom', 'start', 'end' (where end is chrom size)

        for _, row in result.iterrows():
            expected_size = all_intervals[all_intervals["chrom"] == row["chrom1"]]["end"].iloc[0]
            assert row["start1"] == 0
            assert row["end1"] == expected_size
            assert row["start2"] == 0
            assert row["end2"] == expected_size

    def test_full_mode(self):
        """Full mode gives all NxN chromosome pairs."""
        result = pymisha.gintervals_2d_all(mode="full")
        # 3 chroms → 9 pairs
        assert len(result) == 9

    def test_full_mode_has_all_pairs(self):
        """Full mode contains cross-chromosome pairs."""
        result = pymisha.gintervals_2d_all(mode="full")
        pairs = set(zip(result["chrom1"], result["chrom2"], strict=False))
        chroms = ["chr1", "chr2", "chrX"]
        for c1 in chroms:
            for c2 in chroms:
                assert (c1, c2) in pairs

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            pymisha.gintervals_2d_all(mode="invalid")


class TestGintervals2dAllScratchDb:
    """gintervals_2d_all on databases other than the test DB."""

    @pytest.mark.parametrize("db_format", ["per-chromosome", "indexed"])
    def test_pairs_follow_chromosome_order_not_names(self, tmp_path, db_format):
        """Rows follow the database's chromosome order, as R's do, not the sorted names.

        R's gdb.create(per-chromosome) keeps the FASTA order in chrom_sizes.txt, and
        gdb.convert_to_indexed keeps that order: gintervals.all() is chr2, chr10, chr1,
        chrX and gintervals.2d.all() starts chr2-chr2, chr2-chr10, chr2-chr1.
        """
        order = ["chr2", "chr10", "chr1", "chrX"]
        db = _scratch_db(tmp_path, order, db_format="per-chromosome")
        # pymisha's gdb_create sorts the chromosomes; put back the FASTA order R keeps
        sizes = dict(line.split("\t") for line in (db / "chrom_sizes.txt").read_text().splitlines())
        (db / "chrom_sizes.txt").write_text("".join(f"{c}\t{sizes[c]}\n" for c in order))
        pymisha.gdb_init(str(db))
        if db_format == "indexed":
            pymisha.gdb_convert_to_indexed(force=True)
            pymisha.gdb_init(str(db))
            assert (db / "seq" / "genome.idx").exists()
        assert list(pymisha.gintervals_all()["chrom"]) == order
        assert _pairs(pymisha.gintervals_2d_all()) == [(a, b) for a in order for b in order]
        assert _pairs(pymisha.gintervals_2d_all(mode="diagonal")) == [(c, c) for c in order]

    def test_many_contig_db_like_r(self, tmp_path, monkeypatch):
        """101 chromosomes, above R's default gmulticontig.2d.threshold of 100: R's
        gintervals.2d.all() returns the 101 intra-chromosomal pairs, with a warning that
        says so, and warns before building all 10201. A 2D computation over the
        whole-genome scope stops unless CONFIG["multicontig_2d_mode"] is set, and is
        quiet in a mode that was set."""
        db = _scratch_db(tmp_path, [f"chrB{i}" for i in range(101)])
        pymisha.gdb_init(str(db))
        chroms = list(pymisha.gintervals_all()["chrom"])
        assert pymisha.CONFIG.get("multicontig_2d_threshold", 100) == 100
        assert pymisha.CONFIG.get("multicontig_2d_mode") is None
        with pytest.warns(UserWarning, match="returns the cis chromosome pairs only"):
            result = pymisha.gintervals_2d_all()
        assert _pairs(result) == [(c, c) for c in chroms]
        with pytest.warns(UserWarning, match=r"101 contigs \(10201 pairs\)"):
            full = pymisha.gintervals_2d_all(mode="full")
        assert _pairs(full) == [(a, b) for a in chroms for b in chroms]

        rects = pymisha.gintervals_2d(["chrB0", "chrB0"], 0, 4, ["chrB0", "chrB1"], 0, 4)
        pymisha.gtrack_2d_create("t2d", "x", rects, [1.0, 100.0])
        with pytest.raises(ValueError, match="multicontig_2d_mode"):
            pymisha.gsummary("t2d")
        monkeypatch.setitem(pymisha.CONFIG, "multicontig_2d_mode", "full")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = pymisha.gsummary("t2d")
        assert summary["Total intervals"] == 2
