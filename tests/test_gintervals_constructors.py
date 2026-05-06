import pandas as pd
import pytest

import pymisha as pm


def test_gintervals_from_tuples_supports_tuple_and_strand_override():
    rows = [("chr1", 10, 20), ("1", 30, 40)]
    result = pm.gintervals_from_tuples(rows, strand=[1, -1])

    assert list(result["chrom"]) == ["1", "1"]
    assert list(result["start"]) == [10, 30]
    assert list(result["end"]) == [20, 40]
    assert list(result["strand"]) == [1, -1]


def test_gintervals_from_tuples_supports_dict_rows():
    rows = [
        {"chrom": "chr2", "start": 5, "end": 15, "strand": -1},
        {"chrom": "2", "start": 20, "end": 30, "strand": 1},
    ]
    result = pm.gintervals_from_tuples(rows)

    assert list(result["chrom"]) == ["2", "2"]
    assert list(result["start"]) == [5, 20]
    assert list(result["end"]) == [15, 30]
    assert list(result["strand"]) == [-1, 1]


def test_gintervals_from_tuples_empty_returns_empty_dataframe():
    result = pm.gintervals_from_tuples([])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["chrom", "start", "end"]
    assert len(result) == 0


def test_gintervals_from_strings_parses_basic_and_strand_forms():
    result = pm.gintervals_from_strings(["chr1:10-20", "1:30-40:-"])
    assert list(result["chrom"]) == ["1", "1"]
    assert list(result["start"]) == [10, 30]
    assert list(result["end"]) == [20, 40]
    assert list(result["strand"]) == [0, -1]


def test_gintervals_from_strings_chrom_only_uses_full_chrom_extent():
    all_intervals = pm.gintervals_all()
    chr1_end = int(all_intervals.loc[all_intervals["chrom"] == "1", "end"].iloc[0])

    result = pm.gintervals_from_strings("chr1")
    assert len(result) == 1
    assert result.iloc[0]["chrom"] == "1"
    assert int(result.iloc[0]["start"]) == 0
    assert int(result.iloc[0]["end"]) == chr1_end


def test_gintervals_from_strings_invalid_raises():
    with pytest.raises(ValueError, match="Invalid interval string"):
        pm.gintervals_from_strings("chr1:10")


def test_gintervals_from_bed_with_and_without_strand(tmp_path):
    bed_path = tmp_path / "test.bed"
    bed_path.write_text(
        "# comment\n"
        "chr1\t10\t20\tname1\t0\t+\n"
        "1\t30\t40\tname2\t0\t-\n"
        "1 50 60\n",
        encoding="utf-8",
    )

    with_strand = pm.gintervals_from_bed(bed_path, has_strand=True)
    assert list(with_strand["chrom"]) == ["1", "1", "1"]
    assert list(with_strand["start"]) == [10, 30, 50]
    assert list(with_strand["end"]) == [20, 40, 60]
    assert list(with_strand["strand"]) == [1, -1, 0]

    no_strand = pm.gintervals_from_bed(bed_path, has_strand=False)
    assert list(no_strand.columns) == ["chrom", "start", "end"]
    assert len(no_strand) == 3


def test_gintervals_from_bed_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.bed"
    with pytest.raises(FileNotFoundError):
        pm.gintervals_from_bed(missing)


def test_gintervals_accepts_character_strand():
    df = pm.gintervals(
        ["1", "1", "2"],
        [10, 20, 30],
        [100, 200, 300],
        strand=["+", "-", "."],
    )
    assert list(df["strand"]) == [1, -1, 0]


def test_gintervals_character_strand_broadcasts_scalar():
    df = pm.gintervals(["1", "2"], [10, 20], [30, 40], strand="-")
    assert list(df["strand"]) == [-1, -1]


def test_gintervals_character_strand_extras():
    df = pm.gintervals(
        ["1", "1", "1"],
        [10, 20, 30],
        [40, 50, 60],
        strand=["*", "", "."],
    )
    assert list(df["strand"]) == [0, 0, 0]


def test_gintervals_invalid_strand_raises():
    with pytest.raises(ValueError, match="Invalid strand"):
        pm.gintervals(["1"], [10], [100], strand="X")


def test_gintervals_import_bed_with_metadata(tmp_path):
    bed = tmp_path / "ex.bed"
    bed.write_text(
        "track name=foo\n"
        "# comment\n"
        "1\t100\t200\tname1\t0.5\t+\n"
        "chr1\t300\t400\tname2\t0.7\t-\n"
        "2\t100\t150\tname3\t1.0\t.\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_bed(bed)
    assert list(df["chrom"]) == ["1", "1", "2"]
    assert list(df["start"]) == [100, 300, 100]
    assert list(df["end"]) == [200, 400, 150]
    assert list(df["strand"]) == [1, -1, 0]
    assert list(df["name"]) == ["name1", "name2", "name3"]
    assert list(df["score"]) == [0.5, 0.7, 1.0]


def test_gintervals_import_bed_drops_optional_columns(tmp_path):
    bed = tmp_path / "ex.bed"
    bed.write_text(
        "1\t100\t200\tname1\t0.5\t+\n"
        "1\t300\t400\tname2\t0.7\t-\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_bed(bed, name=False, score=False, strand=False)
    assert list(df.columns) == ["chrom", "start", "end"]


def test_gintervals_import_bed_three_column(tmp_path):
    bed = tmp_path / "ex.bed"
    bed.write_text(
        "1\t100\t200\n"
        "2\t50\t75\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_bed(bed)
    assert list(df.columns) == ["chrom", "start", "end"]
    assert list(df["start"]) == [100, 50]


def test_gintervals_import_bed_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pm.gintervals_import_bed(tmp_path / "missing.bed")


def test_gintervals_import_gff_offsets_start(tmp_path):
    gff = tmp_path / "ex.gff"
    gff.write_text(
        "##gff-version 3\n"
        "1\trefseq\texon\t101\t200\t.\t+\t.\tID=e1\n"
        "1\trefseq\tgene\t301\t400\t0.5\t-\t.\tID=g1\n"
        "2\trefseq\texon\t11\t20\t.\t.\t.\tID=e2\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_gff(gff)
    assert list(df["chrom"]) == ["1", "1", "2"]
    assert list(df["start"]) == [100, 300, 10]
    assert list(df["end"]) == [200, 400, 20]
    assert list(df["strand"]) == [1, -1, 0]
    assert list(df["type"]) == ["exon", "gene", "exon"]
    assert list(df["source"]) == ["refseq", "refseq", "refseq"]
    assert list(df["attrs"]) == ["ID=e1", "ID=g1", "ID=e2"]


def test_gintervals_import_gff_feature_filter(tmp_path):
    gff = tmp_path / "ex.gff"
    gff.write_text(
        "1\tref\texon\t101\t200\t.\t+\t.\t.\n"
        "1\tref\tgene\t301\t400\t.\t+\t.\t.\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_gff(gff, feature="exon")
    assert list(df["type"]) == ["exon"]


def test_gintervals_import_gff_feature_filter_missing_raises(tmp_path):
    gff = tmp_path / "ex.gff"
    gff.write_text(
        "1\tref\texon\t101\t200\t.\t+\t.\t.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No records of feature type"):
        pm.gintervals_import_gff(gff, feature="gene")


def test_gintervals_import_vcf_uses_ref_length(tmp_path):
    vcf = tmp_path / "ex.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t100\tv1\tA\tG\t30\tPASS\tDP=10\n"
        "1\t200\tv2\tACGT\tA\t40\t.\tDP=20\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_vcf(vcf)
    assert list(df["chrom"]) == ["1", "1"]
    assert list(df["start"]) == [99, 199]
    assert list(df["end"]) == [100, 203]
    assert list(df["id"]) == ["v1", "v2"]
    assert list(df["ref"]) == ["A", "ACGT"]
    assert list(df["alt"]) == ["G", "A"]
    assert list(df["qual"]) == [30.0, 40.0]
    assert list(df["filter"]) == ["PASS", "."]
    assert list(df["info"]) == ["DP=10", "DP=20"]


def test_gintervals_import_vcf_drops_info(tmp_path):
    vcf = tmp_path / "ex.vcf"
    vcf.write_text(
        "1\t100\tv1\tA\tG\t30\tPASS\tDP=10\n",
        encoding="utf-8",
    )

    df = pm.gintervals_import_vcf(vcf, info=False)
    assert "info" not in df.columns


def test_gintervals_window_broadcasts_scalars():
    result = pm.gintervals_window("chr1", [100, 200], half_width=25)
    assert list(result["chrom"]) == ["1", "1"]
    assert list(result["start"]) == [75, 175]
    assert list(result["end"]) == [125, 225]

    result2 = pm.gintervals_window(["1", "2"], 100, half_width=10)
    assert list(result2["chrom"]) == ["1", "2"]
    assert list(result2["start"]) == [90, 90]
    assert list(result2["end"]) == [110, 110]
