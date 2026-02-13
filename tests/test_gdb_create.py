"""Tests for gdb_create (database creation from FASTA)."""

import gzip
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


@pytest.fixture
def tmpdir():
    """Create a temporary directory and clean up after."""
    d = tempfile.mkdtemp(prefix="pymisha-gdb-create-test-")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def simple_fasta(tmpdir):
    """Create a simple multi-FASTA file with two chromosomes."""
    fasta_path = tmpdir / "test.fa"
    fasta_path.write_text(
        ">chr1\n"
        "ACGTACGTACGTACGT\n"
        "NNNNNNNNNNNNNNNN\n"
        ">chr2\n"
        "TTTTAAAACCCCGGGG\n"
    )
    return fasta_path


@pytest.fixture
def gzipped_fasta(tmpdir):
    """Create a gzipped FASTA file."""
    fasta_path = tmpdir / "test.fa.gz"
    content = (
        ">chrX\n"
        "ACGTACGT\n"
        ">chrY\n"
        "TTTTAAAA\n"
    )
    with gzip.open(fasta_path, "wt") as f:
        f.write(content)
    return fasta_path


class TestGdbCreate:
    """Tests for gdb_create."""

    def test_creates_database_structure(self, tmpdir, simple_fasta):
        """gdb_create creates the expected directory structure."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))

        assert (groot / "seq").is_dir()
        assert (groot / "tracks").is_dir()
        assert (groot / "chrom_sizes.txt").is_file()

    def test_creates_indexed_format(self, tmpdir, simple_fasta):
        """gdb_create produces genome.seq and genome.idx in indexed format."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))

        assert (groot / "seq" / "genome.seq").is_file()
        assert (groot / "seq" / "genome.idx").is_file()

    def test_genome_idx_has_mishaidx_magic(self, tmpdir, simple_fasta):
        """genome.idx starts with MISHAIDX magic header."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))

        with open(groot / "seq" / "genome.idx", "rb") as f:
            magic = f.read(8)
        assert magic == b"MISHAIDX"

    def test_chrom_sizes_content(self, tmpdir, simple_fasta):
        """chrom_sizes.txt contains correct chromosome sizes."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))

        chrom_sizes = pd.read_csv(
            groot / "chrom_sizes.txt", sep="\t", header=None,
            names=["chrom", "size"],
        )
        # chr1: 32 bases (16 + 16), chr2: 16 bases
        assert len(chrom_sizes) == 2
        chr1 = chrom_sizes[chrom_sizes["chrom"] == "chr1"]
        chr2 = chrom_sizes[chrom_sizes["chrom"] == "chr2"]
        assert len(chr1) == 1
        assert len(chr2) == 1
        assert int(chr1["size"].iloc[0]) == 32
        assert int(chr2["size"].iloc[0]) == 16

    def test_genome_seq_content(self, tmpdir, simple_fasta):
        """genome.seq contains concatenated sequence data."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))

        seq = (groot / "seq" / "genome.seq").read_bytes()
        # Alphabetic chars and dashes only (no newlines)
        assert b"\n" not in seq
        # Total length = 32 + 16 = 48
        assert len(seq) == 48

    def test_db_is_loadable(self, tmpdir, simple_fasta):
        """Created database can be loaded with gdb_init."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))

        # Load the new database
        pm.gdb_init(str(groot))
        try:
            info = pm.gdb_info()
            assert info["is_db"]
            assert info["format"] == "indexed"
            assert info["num_chromosomes"] == 2
        finally:
            # Restore the test database
            pm.gdb_init(str(TEST_DB))

    def test_returns_chrom_info(self, tmpdir, simple_fasta):
        """gdb_create returns a DataFrame with contig names and sizes."""
        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(simple_fasta))

        assert isinstance(result, pd.DataFrame)
        assert "name" in result.columns
        assert "size" in result.columns
        assert len(result) == 2

    def test_gzipped_fasta(self, tmpdir, gzipped_fasta):
        """gdb_create supports gzipped FASTA files."""
        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(gzipped_fasta))

        assert (groot / "seq" / "genome.seq").is_file()
        assert (groot / "seq" / "genome.idx").is_file()
        assert len(result) == 2

    def test_existing_groot_raises(self, tmpdir, simple_fasta):
        """gdb_create raises if the target directory already exists."""
        groot = tmpdir / "newdb"
        groot.mkdir()
        with pytest.raises(Exception):
            pm.gdb_create(str(groot), str(simple_fasta))

    def test_missing_fasta_raises(self, tmpdir):
        """gdb_create raises if the FASTA file does not exist."""
        groot = tmpdir / "newdb"
        with pytest.raises(FileNotFoundError):
            pm.gdb_create(str(groot), "/nonexistent/file.fa")

    def test_empty_fasta_raises(self, tmpdir):
        """gdb_create raises on empty FASTA file."""
        fasta = tmpdir / "empty.fa"
        fasta.write_text("")
        groot = tmpdir / "newdb"
        with pytest.raises(ValueError):
            pm.gdb_create(str(groot), str(fasta))

    def test_chromosomes_sorted(self, tmpdir):
        """By default, chromosomes are sorted alphabetically."""
        fasta = tmpdir / "unsorted.fa"
        fasta.write_text(
            ">chrZ\nACGT\n"
            ">chrA\nTTTT\n"
            ">chrM\nGGGG\n"
        )
        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(fasta))

        names = result["name"].tolist()
        assert names == sorted(names)

    def test_fasta_header_sanitization(self, tmpdir):
        """FASTA headers with pipes and extra info are sanitized."""
        fasta = tmpdir / "complex_headers.fa"
        fasta.write_text(
            ">chr1 some description here\n"
            "ACGT\n"
            ">gi|12345|ref|NC_000002.1| Homo sapiens\n"
            "TTTT\n"
        )
        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(fasta))

        names = result["name"].tolist()
        # First should be just "chr1" (description stripped)
        assert "chr1" in names
        # Second should be sanitized from the pipe-delimited header
        # The exact result depends on sanitization logic, but it should not contain spaces or pipes
        for name in names:
            assert " " not in name
            assert "|" not in name

    def test_multiple_fasta_files(self, tmpdir):
        """gdb_create accepts a list of FASTA file paths."""
        fasta1 = tmpdir / "part1.fa"
        fasta1.write_text(">chr1\nACGTACGT\n")
        fasta2 = tmpdir / "part2.fa"
        fasta2.write_text(">chr2\nTTTTAAAA\n")

        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), [str(fasta1), str(fasta2)])

        assert len(result) == 2
        assert (groot / "seq" / "genome.seq").is_file()

    def test_pssms_dir_created(self, tmpdir, simple_fasta):
        """gdb_create creates the pssms directory."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta))
        assert (groot / "pssms").is_dir()


# ===========================================================================
# Indexed integration tests
# (ported from R test-indexed-integration.R)
# ===========================================================================


class TestIndexedIntegration:
    """Integration tests for indexed genome format: create DB, extract sequences.

    Ported from R test-indexed-integration.R.
    """

    def test_gseq_extract_all_strands(self, tmpdir):
        """Indexed format works with gseq_extract on all strands.

        Ported from R line 4-29.
        """
        fasta = tmpdir / "strand.fa"
        fasta.write_text(">test\nACTGACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            # Forward strand
            fwd = pm.gseq_extract(pm.gintervals(["test"], 0, 8, strand=1))
            assert fwd == ["ACTGACTG"]

            # Reverse strand
            rev = pm.gseq_extract(pm.gintervals(["test"], 0, 8, strand=-1))
            assert rev == ["CAGTCAGT"]

            # No strand specified (defaults to forward)
            nostrand = pm.gseq_extract(pm.gintervals(["test"], 0, 8))
            assert nostrand == ["ACTGACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_multiple_interval_extraction(self, tmpdir):
        """Indexed format works with multiple interval extraction.

        Ported from R line 32-56.
        """
        fasta = tmpdir / "multi.fa"
        fasta.write_text(">chr1\nAAAAAAAA\n>chr2\nCCCCCCCC\n>chr3\nGGGGGGGG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            intervals = pd.DataFrame({
                "chrom": ["chr1", "chr2", "chr3"],
                "start": [0, 0, 0],
                "end": [4, 4, 4],
            })
            seqs = pm.gseq_extract(intervals)
            assert seqs == ["AAAA", "CCCC", "GGGG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_gintervals_all_with_indexed(self, tmpdir):
        """Indexed format works with gintervals_all().

        Ported from R line 58-77.
        """
        fasta = tmpdir / "three.fa"
        fasta.write_text(">a\nACTG\n>b\nGGGG\n>c\nCCCC\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            all_intervals = pm.gintervals_all()
            assert len(all_intervals) == 3
            chroms = sorted(all_intervals["chrom"].tolist())
            assert chroms == ["a", "b", "c"]
            assert (all_intervals["start"] == 0).all()
            assert (all_intervals["end"] == 4).all()
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_boundary_conditions(self, tmpdir):
        """Indexed format handles boundary conditions.

        Ported from R line 152-180.
        """
        fasta = tmpdir / "boundary.fa"
        fasta.write_text(">test\nACTGACTGACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            # Extract single base from start
            start = pm.gseq_extract(pm.gintervals(["test"], 0, 1))
            assert start == ["A"]

            # Extract single base from end
            end = pm.gseq_extract(pm.gintervals(["test"], 11, 12))
            assert end == ["G"]

            # Extract full sequence
            full = pm.gseq_extract(pm.gintervals(["test"], 0, 12))
            assert full == ["ACTGACTGACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_persists_across_reloads(self, tmpdir):
        """Indexed format persists across database reloads.

        Ported from R line 182-207.
        """
        fasta = tmpdir / "persist.fa"
        fasta.write_text(">test\nACTGACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq1 = pm.gseq_extract(pm.gintervals(["test"], 0, 8))

            # Reload database
            pm.gdb_init(str(groot))

            seq2 = pm.gseq_extract(pm.gintervals(["test"], 0, 8))

            assert seq1 == seq2
            assert seq1 == ["ACTGACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_compatible_with_gdb_reload(self, tmpdir):
        """Indexed format is compatible with gdb_reload().

        Ported from R line 209-229.
        """
        fasta = tmpdir / "reload.fa"
        fasta.write_text(">test\nACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            pm.gdb_reload()
            seq = pm.gseq_extract(pm.gintervals(["test"], 0, 4))
            assert seq == ["ACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_chromosome_name_lookups(self, tmpdir):
        """Indexed format works with chromosome name lookups.

        Ported from R line 127-150.
        """
        fasta = tmpdir / "chrnames.fa"
        fasta.write_text(">chr1\nACTG\n>chr2\nGGGG\n>chrX\nCCCC\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            # All chromosome names are accessible
            s1 = pm.gseq_extract(pm.gintervals(["chr1"], 0, 4))
            assert s1 == ["ACTG"]
            s2 = pm.gseq_extract(pm.gintervals(["chr2"], 0, 4))
            assert s2 == ["GGGG"]
            sx = pm.gseq_extract(pm.gintervals(["chrX"], 0, 4))
            assert sx == ["CCCC"]

            # Invalid chromosome name raises
            with pytest.raises(Exception):
                pm.gseq_extract(pm.gintervals(["chrY"], 0, 4))
        finally:
            pm.gdb_init(str(TEST_DB))


# ===========================================================================
# Multi-FASTA import edge case tests
# (ported from R test-multifasta-import.R)
# ===========================================================================


class TestMultiFastaImport:
    """Tests for multi-FASTA import edge cases.

    Ported from R test-multifasta-import.R.
    """

    def test_extracts_sequences_correctly(self, tmpdir):
        """Multi-FASTA import extracts sequences correctly.

        Ported from R line 28-57.
        """
        fasta = tmpdir / "seqs.fa"
        fasta.write_text(">seq1\nACTGACTGACTG\n>seq2\nGGGGCCCC\n>seq3\nTATATA\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            assert pm.gseq_extract(pm.gintervals(["seq1"], 0, 12)) == ["ACTGACTGACTG"]
            assert pm.gseq_extract(pm.gintervals(["seq2"], 0, 8)) == ["GGGGCCCC"]
            assert pm.gseq_extract(pm.gintervals(["seq3"], 0, 6)) == ["TATATA"]

            # Partial extraction
            partial = pm.gseq_extract(pm.gintervals(["seq1"], 4, 8))
            assert partial == ["ACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_reverse_complement(self, tmpdir):
        """Multi-FASTA import handles reverse complement.

        Ported from R line 59-82.
        """
        fasta = tmpdir / "revcomp.fa"
        fasta.write_text(">test\nACTGACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            fwd = pm.gseq_extract(pm.gintervals(["test"], 0, 8, strand=1))
            assert fwd == ["ACTGACTG"]

            rev = pm.gseq_extract(pm.gintervals(["test"], 0, 8, strand=-1))
            assert rev == ["CAGTCAGT"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_multiline_sequences(self, tmpdir):
        """Multi-FASTA import handles multi-line sequences.

        Ported from R line 222-250.
        """
        fasta = tmpdir / "multiline.fa"
        fasta.write_text(
            ">multiline\n"
            "ACTGACTGACTGACTG\n"
            "GGGGCCCCAAAATTTT\n"
            "TATATATATATATATA\n"
        )
        expected = "ACTGACTGACTGACTGGGGGCCCCAAAATTTTTATATATATATATATA"

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            full = pm.gseq_extract(pm.gintervals(["multiline"], 0, 48))
            assert full == [expected]

            # Partial extraction across original line boundaries
            partial = pm.gseq_extract(pm.gintervals(["multiline"], 14, 34))
            assert partial == ["TGGGGGCCCCAAAATTTTTA"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_n_characters(self, tmpdir):
        """Multi-FASTA import handles N characters.

        Ported from R line 252-275.
        """
        fasta = tmpdir / "withN.fa"
        fasta.write_text(">withN\nACTGNNNNACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq_n = pm.gseq_extract(pm.gintervals(["withN"], 0, 12))
            assert seq_n == ["ACTGNNNNACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_validates_chromosome_sizes(self, tmpdir):
        """Multi-FASTA import validates chromosome sizes.

        Ported from R line 198-220.
        """
        fasta = tmpdir / "sizes.fa"
        fasta.write_text(
            ">chr1\nACTGACTGACTGACTG\n"
            ">chr2\nGG\n"
            ">chr3\nTATATATATATATATA\n"
        )

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            chroms = pm.gintervals_all()
            chr1_size = int(chroms[chroms["chrom"] == "chr1"]["end"].iloc[0])
            chr2_size = int(chroms[chroms["chrom"] == "chr2"]["end"].iloc[0])
            chr3_size = int(chroms[chroms["chrom"] == "chr3"]["end"].iloc[0])
            assert chr1_size == 16
            assert chr2_size == 2
            assert chr3_size == 16
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_preserves_sequence_boundaries(self, tmpdir):
        """Multi-FASTA import preserves exact sequence boundaries.

        Ported from R line 318-351.
        No cross-contamination between adjacent contigs.
        """
        fasta = tmpdir / "boundary.fa"
        fasta.write_text(
            ">boundary1\nAAAAAAAAAAAAAAAA\n"
            ">boundary2\nTTTTTTTTTTTTTTTT\n"
        )

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            start1 = pm.gseq_extract(pm.gintervals(["boundary1"], 0, 4))
            assert start1 == ["AAAA"]

            end1 = pm.gseq_extract(pm.gintervals(["boundary1"], 12, 16))
            assert end1 == ["AAAA"]

            # First base of second contig should be T, not A
            start2 = pm.gseq_extract(pm.gintervals(["boundary2"], 0, 1))
            assert start2 == ["T"]

            # Full sequences - no cross-contamination
            full1 = pm.gseq_extract(pm.gintervals(["boundary1"], 0, 16))
            full2 = pm.gseq_extract(pm.gintervals(["boundary2"], 0, 16))
            assert full1 == ["AAAAAAAAAAAAAAAA"]
            assert full2 == ["TTTTTTTTTTTTTTTT"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_various_contig_name_formats(self, tmpdir):
        """Multi-FASTA import works with various contig name formats.

        Ported from R line 387-422.
        """
        fasta = tmpdir / "names.fa"
        fasta.write_text(
            ">1\nACTG\n"        # Numeric (Ensembl style)
            ">chr2\nGGGG\n"     # chr prefix
            ">MT\nCCCC\n"       # Mitochondrial
            ">X\nTTTT\n"        # Sex chromosome
        )

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            chroms = pm.gintervals_all()
            chrom_names = chroms["chrom"].tolist()

            assert "1" in chrom_names
            assert "chr2" in chrom_names
            assert "MT" in chrom_names
            assert "X" in chrom_names

            # All sequences accessible
            assert pm.gseq_extract(pm.gintervals(["1"], 0, 4)) == ["ACTG"]
            assert pm.gseq_extract(pm.gintervals(["chr2"], 0, 4)) == ["GGGG"]
            assert pm.gseq_extract(pm.gintervals(["MT"], 0, 4)) == ["CCCC"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_large_contig(self, tmpdir):
        """Multi-FASTA import handles large contigs efficiently.

        Ported from R line 424-462.
        """
        large_seq = "ACTG" * 250  # 1KB
        fasta = tmpdir / "large.fa"
        # Write in 100-char lines to simulate real FASTA
        lines = [">large"]
        for i in range(0, 1000, 100):
            lines.append(large_seq[i : i + 100])
        fasta.write_text("\n".join(lines) + "\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            full = pm.gseq_extract(pm.gintervals(["large"], 0, 1000))
            assert len(full) == 1
            assert len(full[0]) == 1000
            assert full[0] == large_seq

            # Random access at different positions
            chunk1 = pm.gseq_extract(pm.gintervals(["large"], 0, 100))
            assert chunk1 == [large_seq[:100]]

            chunk2 = pm.gseq_extract(pm.gintervals(["large"], 500, 600))
            assert chunk2 == [large_seq[500:600]]

            chunk3 = pm.gseq_extract(pm.gintervals(["large"], 900, 1000))
            assert chunk3 == [large_seq[900:1000]]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_whitespace_in_header(self, tmpdir):
        """Multi-FASTA import handles whitespace in headers.

        Ported from R line 575-593.
        Name should be truncated at first space.
        """
        fasta = tmpdir / "spaces.fa"
        fasta.write_text(">contig with spaces in header\nACTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            chroms = pm.gintervals_all()
            # Name should be truncated at first space
            assert "contig" in chroms["chrom"].tolist()
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_chromosome_order(self, tmpdir):
        """Multi-FASTA import creates correct (sorted) chromosome order.

        Ported from R line 595-616.
        """
        fasta = tmpdir / "order.fa"
        fasta.write_text(">zebra\nACTG\n>apple\nGGGG\n>middle\nCCCC\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))

        chrom_sizes = pd.read_csv(
            groot / "chrom_sizes.txt", sep="\t", header=None, names=["chrom", "size"]
        )
        names = chrom_sizes["chrom"].tolist()
        assert names == sorted(names)

    def test_mixed_case_sequences(self, tmpdir):
        """Multi-FASTA import handles mixed case sequences.

        Ported from R line 555-573.
        """
        fasta = tmpdir / "mixcase.fa"
        fasta.write_text(">test\nActGacTG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq = pm.gseq_extract(pm.gintervals(["test"], 0, 8))
            assert seq[0].upper() == "ACTGACTG"
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_trailing_newlines(self, tmpdir):
        """Multi-FASTA import handles FASTA with trailing newlines.

        Ported from R line 704-723.
        """
        fasta = tmpdir / "trailing.fa"
        fasta.write_text(">test\nACTG\n\n\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq = pm.gseq_extract(pm.gintervals(["test"], 0, 4))
            assert seq == ["ACTG"]
            assert len(pm.gintervals_all()) == 1
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_windows_line_endings(self, tmpdir):
        """Multi-FASTA import handles Windows line endings.

        Ported from R line 725-744.
        """
        fasta = tmpdir / "crlf.fa"
        # Write with \r\n line endings
        fasta.write_bytes(b">test\r\nACTG\r\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq = pm.gseq_extract(pm.gintervals(["test"], 0, 4))
            assert seq == ["ACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_iupac_ambiguity_codes(self, tmpdir):
        """Multi-FASTA import handles ambiguous IUPAC codes.

        Ported from R line 618-637.
        R=A/G, Y=C/T, W=A/T, S=G/C, K=G/T, M=A/C
        """
        fasta = tmpdir / "iupac.fa"
        fasta.write_text(">test\nRYWSKM\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq = pm.gseq_extract(pm.gintervals(["test"], 0, 6))
            assert seq == ["RYWSKM"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_gzipped_fasta(self, tmpdir):
        """Multi-FASTA import handles gzipped files.

        Ported from R line 169-196.
        """
        fasta = tmpdir / "test.fa.gz"
        with gzip.open(fasta, "wt") as f:
            f.write(">test1\nACTGACTG\n>test2\nGGGGCCCC\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            seq1 = pm.gseq_extract(pm.gintervals(["test1"], 0, 8))
            assert seq1 == ["ACTGACTG"]

            seq2 = pm.gseq_extract(pm.gintervals(["test2"], 0, 8))
            assert seq2 == ["GGGGCCCC"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_index_file_structure(self, tmpdir):
        """Multi-FASTA import index file has correct structure.

        Ported from R line 659-681.
        """
        fasta = tmpdir / "idx.fa"
        fasta.write_text(">chr1\nACTG\n>chr2\nGGGG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))

        idx_file = groot / "seq" / "genome.idx"
        assert idx_file.is_file()
        # Index should be larger than just the header (24 bytes) + at least 2 entries
        assert idx_file.stat().st_size > 50

    def test_sequence_file_concatenation(self, tmpdir):
        """Multi-FASTA sequence file concatenates correctly.

        Ported from R line 683-702.
        """
        fasta = tmpdir / "concat.fa"
        fasta.write_text(">a\nAAAA\n>b\nCCCC\n>c\nGGGG\n")

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))

        seq_file = groot / "seq" / "genome.seq"
        seq_size = seq_file.stat().st_size
        assert seq_size == 12  # 4 + 4 + 4 bytes

    def test_read_only_source_file(self, tmpdir):
        """Multi-FASTA import works with read-only source file.

        Ported from R line 639-657.
        """
        fasta = tmpdir / "readonly.fa"
        fasta.write_text(">test\nACTG\n")
        os.chmod(str(fasta), 0o444)

        groot = tmpdir / "db"
        try:
            pm.gdb_create(str(groot), str(fasta))
            # Should complete without error
            assert (groot / "seq" / "genome.seq").is_file()
        finally:
            os.chmod(str(fasta), 0o644)

    def test_sanitizes_contig_names(self, tmpdir):
        """Multi-FASTA import sanitizes contig names with special prefixes.

        Ported from R line 84-115.
        """
        fasta = tmpdir / "sanitize.fa"
        fasta.write_text(
            ">simple\nACTG\n"
            ">chr1 description here\nGGGG\n"
        )

        groot = tmpdir / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        try:
            chroms = pm.gintervals_all()
            chrom_names = chroms["chrom"].tolist()
            assert "simple" in chrom_names
            assert "chr1" in chrom_names

            assert pm.gseq_extract(pm.gintervals(["simple"], 0, 4)) == ["ACTG"]
        finally:
            pm.gdb_init(str(TEST_DB))


# ===========================================================================
# Per-chromosome format tests
# ===========================================================================


class TestGdbCreatePerChromosome:
    """Tests for gdb_create with db_format='per-chromosome'."""

    def test_creates_per_chrom_seq_files(self, tmpdir, simple_fasta):
        """Per-chromosome format creates individual .seq files."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        assert (groot / "seq" / "chr1.seq").is_file()
        assert (groot / "seq" / "chr2.seq").is_file()
        # Should NOT have indexed files
        assert not (groot / "seq" / "genome.seq").exists()
        assert not (groot / "seq" / "genome.idx").exists()

    def test_creates_directory_structure(self, tmpdir, simple_fasta):
        """Per-chromosome format creates the expected directory structure."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        assert (groot / "seq").is_dir()
        assert (groot / "tracks").is_dir()
        assert (groot / "pssms").is_dir()
        assert (groot / "chrom_sizes.txt").is_file()

    def test_seq_file_content(self, tmpdir, simple_fasta):
        """Per-chromosome .seq files contain raw sequence bytes."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        # chr1: ACGTACGTACGTACGT + NNNNNNNNNNNNNNNN = 32 bytes
        chr1_seq = (groot / "seq" / "chr1.seq").read_bytes()
        assert len(chr1_seq) == 32
        assert chr1_seq == b"ACGTACGTACGTACGTNNNNNNNNNNNNNNNN"

        # chr2: TTTTAAAACCCCGGGG = 16 bytes
        chr2_seq = (groot / "seq" / "chr2.seq").read_bytes()
        assert len(chr2_seq) == 16
        assert chr2_seq == b"TTTTAAAACCCCGGGG"

    def test_chrom_sizes_content(self, tmpdir, simple_fasta):
        """Per-chromosome chrom_sizes.txt has correct sizes."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        chrom_sizes = pd.read_csv(
            groot / "chrom_sizes.txt", sep="\t", header=None,
            names=["chrom", "size"],
        )
        assert len(chrom_sizes) == 2
        chr1 = chrom_sizes[chrom_sizes["chrom"] == "chr1"]
        chr2 = chrom_sizes[chrom_sizes["chrom"] == "chr2"]
        assert int(chr1["size"].iloc[0]) == 32
        assert int(chr2["size"].iloc[0]) == 16

    def test_returns_chrom_info(self, tmpdir, simple_fasta):
        """Per-chromosome gdb_create returns a DataFrame."""
        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        assert isinstance(result, pd.DataFrame)
        assert "name" in result.columns
        assert "size" in result.columns
        assert len(result) == 2

    def test_db_is_loadable(self, tmpdir, simple_fasta):
        """Per-chromosome database can be loaded with gdb_init."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        pm.gdb_init(str(groot))
        try:
            info = pm.gdb_info()
            assert info["is_db"]
            assert info["format"] == "per-chromosome"
            assert info["num_chromosomes"] == 2
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_gseq_extract_works(self, tmpdir, simple_fasta):
        """Per-chromosome database supports gseq_extract."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        pm.gdb_init(str(groot))
        try:
            seq1 = pm.gseq_extract(pm.gintervals(["chr1"], 0, 16))
            assert seq1 == ["ACGTACGTACGTACGT"]

            seq2 = pm.gseq_extract(pm.gintervals(["chr2"], 0, 16))
            assert seq2 == ["TTTTAAAACCCCGGGG"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_gseq_extract_partial(self, tmpdir, simple_fasta):
        """Per-chromosome database supports partial extraction."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        pm.gdb_init(str(groot))
        try:
            partial = pm.gseq_extract(pm.gintervals(["chr1"], 4, 12))
            assert partial == ["ACGTACGT"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_gseq_extract_reverse_complement(self, tmpdir):
        """Per-chromosome database supports reverse complement extraction."""
        fasta = tmpdir / "revcomp.fa"
        fasta.write_text(">test\nACTGACTG\n")

        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(fasta), db_format="per-chromosome")

        pm.gdb_init(str(groot))
        try:
            fwd = pm.gseq_extract(pm.gintervals(["test"], 0, 8, strand=1))
            assert fwd == ["ACTGACTG"]

            rev = pm.gseq_extract(pm.gintervals(["test"], 0, 8, strand=-1))
            assert rev == ["CAGTCAGT"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_multiple_fasta_files(self, tmpdir):
        """Per-chromosome format works with multiple FASTA files."""
        fasta1 = tmpdir / "part1.fa"
        fasta1.write_text(">chr1\nACGTACGT\n")
        fasta2 = tmpdir / "part2.fa"
        fasta2.write_text(">chr2\nTTTTAAAA\n")

        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), [str(fasta1), str(fasta2)],
                               db_format="per-chromosome")

        assert len(result) == 2
        assert (groot / "seq" / "chr1.seq").is_file()
        assert (groot / "seq" / "chr2.seq").is_file()

    def test_gzipped_fasta(self, tmpdir, gzipped_fasta):
        """Per-chromosome format works with gzipped FASTA files."""
        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(gzipped_fasta),
                               db_format="per-chromosome")

        assert len(result) == 2
        assert (groot / "seq" / "chrX.seq").is_file()
        assert (groot / "seq" / "chrY.seq").is_file()

    def test_chromosomes_sorted(self, tmpdir):
        """Per-chromosome format sorts chromosomes alphabetically."""
        fasta = tmpdir / "unsorted.fa"
        fasta.write_text(">chrZ\nACGT\n>chrA\nTTTT\n>chrM\nGGGG\n")

        groot = tmpdir / "newdb"
        result = pm.gdb_create(str(groot), str(fasta), db_format="per-chromosome")

        names = result["name"].tolist()
        assert names == sorted(names)

    def test_gintervals_all_works(self, tmpdir):
        """Per-chromosome database works with gintervals_all."""
        fasta = tmpdir / "three.fa"
        fasta.write_text(">a\nACTG\n>b\nGGGG\n>c\nCCCC\n")

        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(fasta), db_format="per-chromosome")
        pm.gdb_init(str(groot))
        try:
            all_intervals = pm.gintervals_all()
            assert len(all_intervals) == 3
            chroms = sorted(all_intervals["chrom"].tolist())
            assert chroms == ["a", "b", "c"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_convertible_to_indexed(self, tmpdir, simple_fasta):
        """Per-chromosome database can be converted to indexed format."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), db_format="per-chromosome")

        # Should be per-chromosome initially
        info = pm.gdb_info(str(groot))
        assert info["format"] == "per-chromosome"

        # Convert to indexed
        pm.gdb_convert_to_indexed(groot=str(groot))

        # Should now be indexed
        info = pm.gdb_info(str(groot))
        assert info["format"] == "indexed"
        assert (groot / "seq" / "genome.seq").is_file()
        assert (groot / "seq" / "genome.idx").is_file()

    def test_format_alias_works(self, tmpdir, simple_fasta):
        """The 'format' kwarg alias works for per-chromosome."""
        groot = tmpdir / "newdb"
        pm.gdb_create(str(groot), str(simple_fasta), format="per-chromosome")

        assert (groot / "seq" / "chr1.seq").is_file()
        assert not (groot / "seq" / "genome.seq").exists()

    def test_invalid_format_raises(self, tmpdir, simple_fasta):
        """Invalid db_format value raises ValueError."""
        groot = tmpdir / "newdb"
        with pytest.raises(ValueError, match="db_format must be"):
            pm.gdb_create(str(groot), str(simple_fasta), db_format="invalid")

    def test_parity_with_indexed(self, tmpdir):
        """Per-chromosome and indexed produce identical sequences on extraction."""
        fasta = tmpdir / "parity.fa"
        fasta.write_text(
            ">chr1\nACTGACTGACTGACTG\n"
            ">chr2\nGGGGCCCCAAAATTTT\n"
            ">chrX\nNNNNACTGNNNN\n"
        )

        groot_idx = tmpdir / "db_indexed"
        groot_pc = tmpdir / "db_perchrom"

        pm.gdb_create(str(groot_idx), str(fasta), db_format="indexed")
        pm.gdb_create(str(groot_pc), str(fasta), db_format="per-chromosome")

        for groot_path, label in [(groot_idx, "indexed"), (groot_pc, "per-chromosome")]:
            pm.gdb_init(str(groot_path))
            try:
                info = pm.gdb_info()
                assert info["format"] == label

                s1 = pm.gseq_extract(pm.gintervals(["chr1"], 0, 16))
                assert s1 == ["ACTGACTGACTGACTG"], f"chr1 mismatch in {label}"

                s2 = pm.gseq_extract(pm.gintervals(["chr2"], 0, 16))
                assert s2 == ["GGGGCCCCAAAATTTT"], f"chr2 mismatch in {label}"

                sx = pm.gseq_extract(pm.gintervals(["chrX"], 0, 12))
                assert sx == ["NNNNACTGNNNN"], f"chrX mismatch in {label}"
            finally:
                pm.gdb_init(str(TEST_DB))
