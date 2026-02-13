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


def test_gintervals_window_broadcasts_scalars():
    result = pm.gintervals_window("chr1", [100, 200], half_width=25)
    assert list(result["chrom"]) == ["1", "1"]
    assert list(result["start"]) == [75, 175]
    assert list(result["end"]) == [125, 225]

    result2 = pm.gintervals_window(["1", "2"], 100, half_width=10)
    assert list(result2["chrom"]) == ["1", "2"]
    assert list(result2["start"]) == [90, 90]
    assert list(result2["end"]) == [110, 110]
