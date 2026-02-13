"""Tests for gintervals_import_genes."""

import gzip

import pandas as pd
import pytest

import pymisha as pm
from pymisha.intervals import (
    _parse_annots_file,
    _unify_intervals,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_gene_line(
    gene_id, chrom, strand, txstart, txend, cdsstart, cdsend,
    exon_starts, exon_ends, protein_id="", align_id=""
):
    """Build a single knownGene-format line."""
    exoncount = len(exon_starts)
    exon_starts_str = ",".join(str(s) for s in exon_starts) + ","
    exon_ends_str = ",".join(str(e) for e in exon_ends) + ","
    return "\t".join([
        gene_id, chrom, strand, str(txstart), str(txend),
        str(cdsstart), str(cdsend), str(exoncount),
        exon_starts_str, exon_ends_str, protein_id, align_id,
    ])


@pytest.fixture
def simple_genes_file(tmp_path):
    """Create a simple knownGene file with genes on chromosomes 1 and 2."""
    lines = [
        # Gene on chr1 (+ strand), one exon
        _make_gene_line(
            "gene1", "chr1", "+", 1000, 2000, 1100, 1900,
            [1100], [1900],
        ),
        # Gene on chr2 (- strand), two exons
        _make_gene_line(
            "gene2", "chr2", "-", 5000, 8000, 5200, 7800,
            [5200, 7000], [6000, 7800],
        ),
        # Gene on chrX (+ strand), one exon
        _make_gene_line(
            "gene3", "chrX", "+", 100, 500, 150, 450,
            [150], [450],
        ),
    ]
    p = tmp_path / "genes.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture
def simple_annots_file(tmp_path):
    """Create a simple annotation file."""
    lines = [
        "gene1\tProtA\tDescA",
        "gene2\tProtB\tDescB",
        "gene3\tProtC\tDescC",
    ]
    p = tmp_path / "annots.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture
def overlapping_genes_file(tmp_path):
    """Create genes with overlapping TSS regions on chromosome 1."""
    lines = [
        # Two genes with overlapping TSS on + strand
        _make_gene_line(
            "geneA", "chr1", "+", 1000, 3000, 1100, 2900,
            [1100], [2900],
        ),
        _make_gene_line(
            "geneB", "chr1", "+", 1000, 4000, 1100, 3900,
            [1100], [3900],
        ),
        # Gene with - strand overlapping TSS at same region
        _make_gene_line(
            "geneC", "chr1", "-", 999, 2000, 1000, 1900,
            [1000], [1900],
        ),
    ]
    p = tmp_path / "overlap_genes.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture
def gzipped_genes_file(tmp_path, simple_genes_file):
    """Create a gzipped version of the simple genes file."""
    gz_path = tmp_path / "genes.txt.gz"
    with open(simple_genes_file, "rb") as f_in, gzip.open(str(gz_path), "wb") as f_out:
        f_out.write(f_in.read())
    return str(gz_path)


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_none_genes_file_raises(self):
        with pytest.raises(ValueError, match="Usage"):
            pm.gintervals_import_genes(None)

    def test_annots_file_without_names_raises(self, simple_genes_file, simple_annots_file):
        with pytest.raises(ValueError, match="annots_names.*cannot be None"):
            pm.gintervals_import_genes(
                simple_genes_file, annots_file=simple_annots_file
            )

    def test_annots_names_not_list_raises(self, simple_genes_file, simple_annots_file):
        with pytest.raises(ValueError, match="annots_names.*must be a list"):
            pm.gintervals_import_genes(
                simple_genes_file,
                annots_file=simple_annots_file,
                annots_names="single_string",
            )


# ---------------------------------------------------------------------------
# Tests: Return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_returns_dict_with_four_keys(self, simple_genes_file):
        result = pm.gintervals_import_genes(simple_genes_file)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"tss", "exons", "utr3", "utr5"}

    def test_each_value_is_dataframe_or_none(self, simple_genes_file):
        result = pm.gintervals_import_genes(simple_genes_file)
        for key in ("tss", "exons", "utr3", "utr5"):
            val = result[key]
            assert val is None or isinstance(val, pd.DataFrame), (
                f"{key} should be DataFrame or None, got {type(val)}"
            )

    def test_dataframes_have_required_columns(self, simple_genes_file):
        result = pm.gintervals_import_genes(simple_genes_file)
        required_cols = {"chrom", "start", "end", "strand"}
        for key in ("tss", "exons", "utr3", "utr5"):
            df = result[key]
            if df is not None:
                assert required_cols.issubset(set(df.columns)), (
                    f"{key} missing columns: "
                    f"{required_cols - set(df.columns)}"
                )

    def test_start_end_are_float(self, simple_genes_file):
        result = pm.gintervals_import_genes(simple_genes_file)
        for key in ("tss", "exons", "utr3", "utr5"):
            df = result[key]
            if df is not None and len(df) > 0:
                assert df["start"].dtype == float
                assert df["end"].dtype == float


# ---------------------------------------------------------------------------
# Tests: TSS computation
# ---------------------------------------------------------------------------


class TestTSS:
    def test_plus_strand_tss(self, simple_genes_file):
        """TSS for + strand: start=txStart, end=txStart+1."""
        result = pm.gintervals_import_genes(simple_genes_file)
        tss = result["tss"]
        # gene1 is on chrom 1, + strand, txStart=1000
        gene1_tss = tss[tss["chrom"] == "1"]
        assert len(gene1_tss) == 1
        assert gene1_tss.iloc[0]["start"] == 1000.0
        assert gene1_tss.iloc[0]["end"] == 1001.0
        assert gene1_tss.iloc[0]["strand"] == 1.0

    def test_minus_strand_tss(self, simple_genes_file):
        """TSS for - strand: start=txEnd-1, end=txEnd."""
        result = pm.gintervals_import_genes(simple_genes_file)
        tss = result["tss"]
        # gene2 is on chrom 2, - strand, txEnd=8000
        gene2_tss = tss[tss["chrom"] == "2"]
        assert len(gene2_tss) == 1
        assert gene2_tss.iloc[0]["start"] == 7999.0
        assert gene2_tss.iloc[0]["end"] == 8000.0
        assert gene2_tss.iloc[0]["strand"] == -1.0


# ---------------------------------------------------------------------------
# Tests: Exons
# ---------------------------------------------------------------------------


class TestExons:
    def test_single_exon_gene(self, simple_genes_file):
        result = pm.gintervals_import_genes(simple_genes_file)
        exons = result["exons"]
        # gene1 has 1 exon: [1100, 1900)
        gene1_exons = exons[exons["chrom"] == "1"]
        assert len(gene1_exons) >= 1
        row = gene1_exons.iloc[0]
        assert row["start"] == 1100.0
        assert row["end"] == 1900.0

    def test_multi_exon_gene(self, simple_genes_file):
        result = pm.gintervals_import_genes(simple_genes_file)
        exons = result["exons"]
        # gene2 has 2 exons on chrom 2
        gene2_exons = exons[exons["chrom"] == "2"]
        assert len(gene2_exons) == 2


# ---------------------------------------------------------------------------
# Tests: UTR3 and UTR5
# ---------------------------------------------------------------------------


class TestUTR:
    def test_utr3_plus_strand(self, simple_genes_file):
        """UTR3 for + strand: start=lastExonEnd-1, end=txEnd."""
        result = pm.gintervals_import_genes(simple_genes_file)
        utr3 = result["utr3"]
        # gene1: + strand, lastExonEnd=1900, txEnd=2000
        g1_utr3 = utr3[utr3["chrom"] == "1"]
        assert len(g1_utr3) == 1
        assert g1_utr3.iloc[0]["start"] == 1899.0
        assert g1_utr3.iloc[0]["end"] == 2000.0

    def test_utr3_minus_strand(self, simple_genes_file):
        """UTR3 for - strand: start=txStart, end=firstExonStart+1."""
        result = pm.gintervals_import_genes(simple_genes_file)
        utr3 = result["utr3"]
        # gene2: - strand, txStart=5000, firstExonStart=5200
        g2_utr3 = utr3[utr3["chrom"] == "2"]
        assert len(g2_utr3) == 1
        assert g2_utr3.iloc[0]["start"] == 5000.0
        assert g2_utr3.iloc[0]["end"] == 5201.0

    def test_utr5_plus_strand(self, simple_genes_file):
        """UTR5 for + strand: start=txStart, end=firstExonStart+1."""
        result = pm.gintervals_import_genes(simple_genes_file)
        utr5 = result["utr5"]
        # gene1: + strand, txStart=1000, firstExonStart=1100
        g1_utr5 = utr5[utr5["chrom"] == "1"]
        assert len(g1_utr5) == 1
        assert g1_utr5.iloc[0]["start"] == 1000.0
        assert g1_utr5.iloc[0]["end"] == 1101.0

    def test_utr5_minus_strand(self, simple_genes_file):
        """UTR5 for - strand: start=lastExonEnd-1, end=txEnd."""
        result = pm.gintervals_import_genes(simple_genes_file)
        utr5 = result["utr5"]
        # gene2: - strand, lastExonEnd=7800, txEnd=8000
        g2_utr5 = utr5[utr5["chrom"] == "2"]
        assert len(g2_utr5) == 1
        assert g2_utr5.iloc[0]["start"] == 7799.0
        assert g2_utr5.iloc[0]["end"] == 8000.0


# ---------------------------------------------------------------------------
# Tests: Overlapping interval unification
# ---------------------------------------------------------------------------


class TestOverlapping:
    def test_overlapping_tss_same_strand_merged(self, overlapping_genes_file):
        """Overlapping intervals on the same strand should be merged."""
        result = pm.gintervals_import_genes(overlapping_genes_file)
        tss = result["tss"]
        # geneA TSS: [1000, 1001) +
        # geneB TSS: [1000, 1001) +
        # geneC TSS: [1999, 2000) -  (txEnd=2000, - strand)
        # geneA and geneB overlap => merge to [1000, 1001) but geneC at
        # [1999,2000) does not overlap
        assert len(tss[tss["chrom"] == "1"]) == 2

    def test_overlapping_different_strand_becomes_zero(self, overlapping_genes_file):
        """When overlapping intervals have different strands, merged strand is 0."""
        # Build a file where TSS intervals actually overlap with different strands
        result = pm.gintervals_import_genes(overlapping_genes_file)
        tss = result["tss"]
        # geneA and geneB at [1000,1001) both + strand, merged strand=1
        chr1_tss = tss[tss["chrom"] == "1"].sort_values("start").reset_index(drop=True)
        # First merged TSS: [1000, 1001) from geneA + geneB, both +1
        assert chr1_tss.iloc[0]["strand"] == 1.0
        # geneC at [1999, 2000), alone, -1
        assert chr1_tss.iloc[1]["strand"] == -1.0


class TestOverlappingMixedStrand:
    """Test the case where intervals with different strands actually overlap."""

    def test_mixed_strand_overlap(self, tmp_path):
        lines = [
            _make_gene_line(
                "g1", "chr1", "+", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
            _make_gene_line(
                "g2", "chr1", "-", 999, 1001, 999, 1001,
                [999], [1001],
            ),
        ]
        p = tmp_path / "mixed_strand.txt"
        p.write_text("\n".join(lines) + "\n")

        result = pm.gintervals_import_genes(str(p))
        tss = result["tss"]
        chr1_tss = tss[tss["chrom"] == "1"]
        # g1 TSS: [1000, 1001) +
        # g2 TSS: [1000, 1001) -  (txEnd=1001, - strand => [1000, 1001))
        # These overlap => merged strand should be 0
        assert len(chr1_tss) == 1
        assert chr1_tss.iloc[0]["strand"] == 0.0


# ---------------------------------------------------------------------------
# Tests: Annotations
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_annotations_attached(self, simple_genes_file, simple_annots_file):
        result = pm.gintervals_import_genes(
            simple_genes_file,
            annots_file=simple_annots_file,
            annots_names=["gene_id", "protein", "description"],
        )
        tss = result["tss"]
        assert "protein" in tss.columns
        assert "description" in tss.columns
        assert "gene_id" in tss.columns

    def test_annotation_values(self, simple_genes_file, simple_annots_file):
        result = pm.gintervals_import_genes(
            simple_genes_file,
            annots_file=simple_annots_file,
            annots_names=["gene_id", "protein", "description"],
        )
        tss = result["tss"]
        # gene1 on chrom 1
        g1_row = tss[tss["chrom"] == "1"].iloc[0]
        assert g1_row["protein"] == "ProtA"
        assert g1_row["description"] == "DescA"
        assert g1_row["gene_id"] == "gene1"

    def test_overlapping_annotations_concatenated(self, tmp_path):
        """When intervals overlap, annotations from each should be merged."""
        genes_lines = [
            _make_gene_line(
                "gA", "chr1", "+", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
            _make_gene_line(
                "gB", "chr1", "+", 1000, 3000, 1100, 2900,
                [1100], [2900],
            ),
        ]
        annots_lines = [
            "gA\tNameA",
            "gB\tNameB",
        ]
        gf = tmp_path / "genes_annot.txt"
        gf.write_text("\n".join(genes_lines) + "\n")
        af = tmp_path / "annots_annot.txt"
        af.write_text("\n".join(annots_lines) + "\n")

        result = pm.gintervals_import_genes(
            str(gf), annots_file=str(af), annots_names=["gene_id", "name"]
        )
        tss = result["tss"]
        chr1_tss = tss[tss["chrom"] == "1"]
        # Both TSS at [1000,1001) => merged, annotations concatenated
        assert len(chr1_tss) == 1
        name_val = chr1_tss.iloc[0]["name"]
        assert "NameA" in name_val
        assert "NameB" in name_val
        assert ";" in name_val

    def test_duplicate_annotations_deduplicated(self, tmp_path):
        """Identical annotation values in overlapping intervals should not repeat."""
        genes_lines = [
            _make_gene_line(
                "gA", "chr1", "+", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
            _make_gene_line(
                "gB", "chr1", "+", 1000, 3000, 1100, 2900,
                [1100], [2900],
            ),
        ]
        annots_lines = [
            "gA\tSameName",
            "gB\tSameName",
        ]
        gf = tmp_path / "genes_dup.txt"
        gf.write_text("\n".join(genes_lines) + "\n")
        af = tmp_path / "annots_dup.txt"
        af.write_text("\n".join(annots_lines) + "\n")

        result = pm.gintervals_import_genes(
            str(gf), annots_file=str(af), annots_names=["gene_id", "name"]
        )
        tss = result["tss"]
        chr1_tss = tss[tss["chrom"] == "1"]
        assert len(chr1_tss) == 1
        # Should be just "SameName", not "SameName;SameName"
        assert chr1_tss.iloc[0]["name"] == "SameName"


# ---------------------------------------------------------------------------
# Tests: Chromosome filtering
# ---------------------------------------------------------------------------


class TestChromosomeFiltering:
    def test_unknown_chrom_skipped(self, tmp_path):
        """Genes on chromosomes not in the DB should be silently skipped."""
        lines = [
            _make_gene_line(
                "gene_unknown", "chr99", "+", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
            _make_gene_line(
                "gene_known", "chr1", "+", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
        ]
        p = tmp_path / "genes_filter.txt"
        p.write_text("\n".join(lines) + "\n")

        result = pm.gintervals_import_genes(str(p))
        tss = result["tss"]
        # Only gene_known should survive
        assert len(tss) == 1
        assert tss.iloc[0]["chrom"] == "1"

    def test_chrx_normalized(self, simple_genes_file):
        """chrX in the gene file should be normalized to 'X'."""
        result = pm.gintervals_import_genes(simple_genes_file)
        tss = result["tss"]
        x_tss = tss[tss["chrom"] == "X"]
        assert len(x_tss) == 1


# ---------------------------------------------------------------------------
# Tests: Gzip support
# ---------------------------------------------------------------------------


class TestGzipSupport:
    def test_gzipped_file_parsed(self, gzipped_genes_file, simple_genes_file):
        """Gzipped files should produce the same result as plain files."""
        result_plain = pm.gintervals_import_genes(simple_genes_file)
        result_gz = pm.gintervals_import_genes(gzipped_genes_file)
        for key in ("tss", "exons", "utr3", "utr5"):
            if result_plain[key] is None:
                assert result_gz[key] is None
            else:
                pd.testing.assert_frame_equal(
                    result_plain[key].reset_index(drop=True),
                    result_gz[key].reset_index(drop=True),
                )


# ---------------------------------------------------------------------------
# Tests: Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_result_when_no_matching_chroms(self, tmp_path):
        """If no genes match any DB chromosome, all results should be None."""
        lines = [
            _make_gene_line(
                "gene_99", "chr99", "+", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
        ]
        p = tmp_path / "no_match.txt"
        p.write_text("\n".join(lines) + "\n")

        result = pm.gintervals_import_genes(str(p))
        for key in ("tss", "exons", "utr3", "utr5"):
            assert result[key] is None

    def test_empty_file(self, tmp_path):
        """An empty genes file should produce all-None results."""
        p = tmp_path / "empty.txt"
        p.write_text("")

        result = pm.gintervals_import_genes(str(p))
        for key in ("tss", "exons", "utr3", "utr5"):
            assert result[key] is None

    def test_invalid_strand_raises(self, tmp_path):
        """Invalid strand value should raise ValueError."""
        lines = [
            _make_gene_line(
                "gene_bad", "chr1", "?", 1000, 2000, 1100, 1900,
                [1100], [1900],
            ),
        ]
        p = tmp_path / "bad_strand.txt"
        p.write_text("\n".join(lines) + "\n")

        with pytest.raises(ValueError, match="invalid strand"):
            pm.gintervals_import_genes(str(p))

    def test_wrong_column_count_raises(self, tmp_path):
        """Lines with wrong number of columns should raise ValueError."""
        p = tmp_path / "bad_cols.txt"
        p.write_text("only\ttwo\tcolumns\n")

        with pytest.raises(ValueError, match="expected 12 columns"):
            pm.gintervals_import_genes(str(p))


# ---------------------------------------------------------------------------
# Tests: Internal parsing functions
# ---------------------------------------------------------------------------


class TestParseAnnotsFile:
    def test_basic_parse(self, tmp_path):
        p = tmp_path / "annots.txt"
        p.write_text("id1\tval1\tval2\nid2\tval3\tval4\n")
        with open(str(p)) as fh:
            result = _parse_annots_file(fh, 3)
        assert "id1" in result
        assert result["id1"] == ["id1", "val1", "val2"]

    def test_duplicate_id_raises(self, tmp_path):
        p = tmp_path / "dup.txt"
        p.write_text("id1\tval1\nid1\tval2\n")
        with open(str(p)) as fh, pytest.raises(ValueError, match="appears more than once"):
            _parse_annots_file(fh, 2)

    def test_wrong_column_count_raises(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("id1\tval1\tval2\n")
        with open(str(p)) as fh, pytest.raises(ValueError, match="number of annotation columns"):
            _parse_annots_file(fh, 2)


class TestUnifyIntervals:
    def test_no_overlap(self):
        records = [
            ("1", 100, 200, 1, None),
            ("1", 300, 400, 1, None),
        ]
        df = _unify_intervals(records, None)
        assert len(df) == 2
        assert df.iloc[0]["start"] == 100.0
        assert df.iloc[1]["start"] == 300.0

    def test_overlap_merged(self):
        records = [
            ("1", 100, 300, 1, None),
            ("1", 200, 400, 1, None),
        ]
        df = _unify_intervals(records, None)
        assert len(df) == 1
        assert df.iloc[0]["start"] == 100.0
        assert df.iloc[0]["end"] == 400.0

    def test_different_strand_merged_to_zero(self):
        records = [
            ("1", 100, 300, 1, None),
            ("1", 200, 400, -1, None),
        ]
        df = _unify_intervals(records, None)
        assert len(df) == 1
        assert df.iloc[0]["strand"] == 0.0

    def test_empty_records(self):
        assert _unify_intervals([], None) is None

    def test_cross_chrom_not_merged(self):
        records = [
            ("1", 100, 300, 1, None),
            ("2", 100, 300, 1, None),
        ]
        df = _unify_intervals(records, None)
        assert len(df) == 2

    def test_annotation_merging(self):
        records = [
            ("1", 100, 300, 1, ["id1", "A"]),
            ("1", 200, 400, 1, ["id2", "B"]),
        ]
        df = _unify_intervals(records, ["gene_id", "name"])
        assert len(df) == 1
        # Annotations should be semicolon-joined and sorted
        assert "A" in df.iloc[0]["name"] or "B" in df.iloc[0]["name"]
        assert ";" in df.iloc[0]["name"]
