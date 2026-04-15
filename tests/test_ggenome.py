"""Tests for ggenome_implant and ggenome_transplant."""

from pathlib import Path

import pandas as pd
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


@pytest.fixture(autouse=True)
def _restore_root():
    yield
    pm.gdb_init(str(TEST_DB))


def _create_db(tmp_path, name, fasta_text, *, db_format="indexed"):
    fasta_path = tmp_path / f"{name}.fa"
    db_path = tmp_path / name
    fasta_path.write_text(fasta_text, encoding="utf-8")
    pm.gdb_create(str(db_path), str(fasta_path), db_format=db_format)
    return db_path


def _read_fasta_seqs(path):
    """Read a FASTA file and return {chrom: sequence} dict."""
    chroms = {}
    current = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current is not None:
                    chroms[current] = "".join(chunks)
                current = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current is not None:
        chroms[current] = "".join(chunks)
    return chroms


class TestGgenomeImplant:

    def test_implant_literal_sequences(self, tmp_path):
        """Implant literal donor sequences into a FASTA."""
        ref_fasta = tmp_path / "ref.fa"
        # 20 A's and 10 C's
        ref_fasta.write_text(">chrA\n" + "A" * 20 + "\n>chrB\n" + "C" * 10 + "\n")

        intervals = pd.DataFrame({
            "chrom": ["chrA", "chrB"],
            "start": [5, 2],
            "end": [10, 6],
        })
        donors = ["TTTTT", "GGGG"]

        out = tmp_path / "out.fa"
        result = pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

        assert result == str(out)
        seqs = _read_fasta_seqs(out)
        assert seqs["chrA"] == "AAAAATTTTTAAAAAAAAAA"
        assert seqs["chrB"] == "CCGGGGCCCC"

    def test_implant_creates_fai(self, tmp_path):
        """Output FASTA should have a .fai index."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\nAAAAAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [4]})
        donors = ["TTTT"]

        out = tmp_path / "out.fa"
        pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

        fai = Path(str(out) + ".fai")
        assert fai.exists()
        lines = fai.read_text().strip().split("\n")
        assert len(lines) == 1
        parts = lines[0].split("\t")
        assert parts[0] == "chrA"
        assert parts[1] == "8"  # sequence length

    def test_implant_from_misha_donor(self, tmp_path):
        """Donor can be a misha root path — sequences extracted automatically."""
        # Create two databases with different sequences
        ref_db = _create_db(tmp_path, "ref_db", ">chrA\nAAAAAAAA\n>chrB\nCCCCCCCC\n")
        donor_db = _create_db(tmp_path, "donor_db", ">chrA\nTTTTTTTT\n>chrB\nGGGGGGGG\n")

        ref_fasta = tmp_path / "ref.fa"
        pm.gdb_init(str(ref_db))
        pm.gdb_export_fasta(str(ref_fasta))

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [2], "end": [6]})
        out = tmp_path / "out.fa"
        pm.ggenome_implant(
            intervals, str(donor_db), str(out), genome_fasta=str(ref_fasta)
        )

        seqs = _read_fasta_seqs(out)
        assert seqs["chrA"] == "AATTTTAA"
        assert seqs["chrB"] == "CCCCCCCC"  # untouched

    def test_implant_multiple_same_chrom(self, tmp_path):
        """Multiple perturbations on the same chromosome."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\n" + "A" * 20 + "\n")

        intervals = pd.DataFrame({
            "chrom": ["chrA", "chrA", "chrA"],
            "start": [0, 5, 15],
            "end": [3, 10, 20],
        })
        donors = ["TTT", "GGGGG", "CCCCC"]

        out = tmp_path / "out.fa"
        pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

        seqs = _read_fasta_seqs(out)
        assert seqs["chrA"] == "TTTAAGGGGGAAAAACCCCC"

    def test_implant_preserves_chrom_order(self, tmp_path):
        """Output FASTA preserves the chromosome order of the reference."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrB\nAAAA\n>chrA\nCCCC\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [2]})
        donors = ["TT"]

        out = tmp_path / "out.fa"
        pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

        seqs = _read_fasta_seqs(out)
        chroms_order = list(seqs.keys())
        assert chroms_order == ["chrB", "chrA"]
        assert seqs["chrA"] == "TTCC"

    def test_implant_creates_trackdb(self, tmp_path):
        """create_trackdb=True should produce a working misha database."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\nAAAAAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [4]})
        donors = ["TTTT"]

        out = tmp_path / "out.fa"
        tdb = tmp_path / "tdb"
        pm.ggenome_implant(
            intervals, donors, str(out),
            genome_fasta=str(ref_fasta),
            create_trackdb=True,
            trackdb_path=str(tdb),
        )

        assert tdb.exists()
        # Verify we can init and extract sequences from the trackdb
        pm.gdb_init(str(tdb))
        seqs = pm.gseq_extract(pd.DataFrame({
            "chrom": ["chrA"], "start": [0], "end": [8]
        }))
        assert seqs[0].upper() == "TTTTAAAA"

    def test_implant_from_current_db(self, tmp_path):
        """When genome_fasta is None, use the current misha database."""
        db = _create_db(tmp_path, "mydb", ">chrA\nAAAAAAAA\n")
        pm.gdb_init(str(db))

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [2], "end": [5]})
        donors = ["TTT"]

        out = tmp_path / "out.fa"
        pm.ggenome_implant(intervals, donors, str(out))

        seqs = _read_fasta_seqs(out)
        assert seqs["chrA"] == "AATTTAAA"

    def test_implant_error_on_length_mismatch(self, tmp_path):
        """Donor length must match interval length."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\nAAAAAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [4]})
        donors = ["TT"]  # length 2 != 4

        out = tmp_path / "out.fa"
        with pytest.raises(ValueError, match="does not match interval length"):
            pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

    def test_implant_error_on_missing_chrom(self, tmp_path):
        """Interval chromosome must exist in the reference."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\nAAAAAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrZ"], "start": [0], "end": [4]})
        donors = ["TTTT"]

        out = tmp_path / "out.fa"
        with pytest.raises(ValueError, match="not found in reference"):
            pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

    def test_implant_error_out_of_bounds(self, tmp_path):
        """Interval must not exceed chromosome length."""
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\nAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [2], "end": [6]})
        donors = ["TTTT"]

        out = tmp_path / "out.fa"
        with pytest.raises(ValueError, match="out of bounds"):
            pm.ggenome_implant(intervals, donors, str(out), genome_fasta=str(ref_fasta))

    def test_implant_no_overwrite_by_default(self, tmp_path):
        ref_fasta = tmp_path / "ref.fa"
        ref_fasta.write_text(">chrA\nAAAA\n")

        out = tmp_path / "out.fa"
        out.write_text("existing")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [2]})
        with pytest.raises(FileExistsError):
            pm.ggenome_implant(intervals, ["TT"], str(out), genome_fasta=str(ref_fasta))


class TestGgenomeTransplant:

    def test_transplant_from_misha_db(self, tmp_path):
        """Transplant sequences between two misha databases."""
        source_db = _create_db(tmp_path, "source", ">chrA\nTTTTTTTT\n")
        ref_fasta = tmp_path / "target.fa"
        ref_fasta.write_text(">chrA\nAAAAAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [2], "end": [6]})
        out = tmp_path / "out.fa"
        pm.ggenome_transplant(
            intervals, str(source_db), str(out),
            target_genome=str(ref_fasta),
        )

        seqs = _read_fasta_seqs(out)
        assert seqs["chrA"] == "AATTTTAA"

    def test_transplant_from_fasta(self, tmp_path):
        """Transplant sequences using a source FASTA file."""
        source_fasta = tmp_path / "source.fa"
        source_fasta.write_text(">chrA\nGGGGGGGG\n")

        ref_fasta = tmp_path / "target.fa"
        ref_fasta.write_text(">chrA\nAAAAAAAA\n")

        intervals = pd.DataFrame({"chrom": ["chrA"], "start": [0], "end": [3]})
        out = tmp_path / "out.fa"
        pm.ggenome_transplant(
            intervals, str(source_fasta), str(out),
            target_genome=str(ref_fasta),
        )

        seqs = _read_fasta_seqs(out)
        assert seqs["chrA"] == "GGGAAAAA"
