"""Port of R tests/testthat/test-gintervals-to-mat.R (PR #120)."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def test_to_mat_round_trip_preserves_coords_and_values():
    df = pd.DataFrame({
        "chrom": ["1", "1", "2"],
        "start": [100, 500, 200],
        "end": [200, 700, 400],
        "t1": [1.5, 2.5, 3.5],
        "t2": [10.0, 20.0, 30.0],
    })
    mat = pm.gintervals_to_mat(df)
    out = pm.gintervals_from_mat(mat)

    assert list(out["chrom"]) == list(df["chrom"])
    assert list(out["start"]) == list(df["start"])
    assert list(out["end"]) == list(df["end"])
    np.testing.assert_array_equal(out["t1"].values, df["t1"].values)
    np.testing.assert_array_equal(out["t2"].values, df["t2"].values)
    assert pd.api.types.is_integer_dtype(out["start"])
    assert pd.api.types.is_integer_dtype(out["end"])


def test_to_mat_with_id_col():
    df = pd.DataFrame({
        "chrom": ["1", "2"],
        "start": [100, 200],
        "end": [200, 400],
        "gene": ["FOO", "BAR"],
        "t1": [1.0, 2.0],
    })
    mat = pm.gintervals_to_mat(df, id_col="gene", value_cols=["t1"])
    assert mat.index.names[0] == "gene"
    assert list(mat.index.get_level_values("gene")) == ["FOO", "BAR"]
    assert list(mat.columns) == ["t1"]

    out = pm.gintervals_from_mat(mat)
    assert list(out["chrom"]) == ["1", "2"]
    assert list(out["start"]) == [100, 200]
    assert list(out["end"]) == [200, 400]
    np.testing.assert_array_equal(out["t1"].values, df["t1"].values)


def test_to_mat_missing_id_col_raises():
    df = pd.DataFrame({"chrom": ["1"], "start": [100], "end": [200], "t1": [1.0]})
    with pytest.raises(ValueError, match="id_col"):
        pm.gintervals_to_mat(df, id_col="nope")


def test_to_mat_explicit_value_cols_subsets():
    df = pd.DataFrame({
        "chrom": ["1"], "start": [100], "end": [200],
        "t1": [1.0], "t2": [2.0], "t3": [3.0],
    })
    mat = pm.gintervals_to_mat(df, value_cols=["t1", "t3"])
    assert list(mat.columns) == ["t1", "t3"]
    np.testing.assert_array_equal(mat.iloc[0].values, [1.0, 3.0])


def test_to_mat_missing_value_cols_raises():
    df = pd.DataFrame({"chrom": ["1"], "start": [100], "end": [200], "t1": [1.0]})
    with pytest.raises(ValueError, match="value_cols"):
        pm.gintervals_to_mat(df, value_cols=["t1", "nope"])


def test_to_mat_auto_detect_rejects_non_numeric():
    df = pd.DataFrame({
        "chrom": ["1"], "start": [100], "end": [200],
        "gene": ["FOO"], "t1": [1.0],
    })
    with pytest.raises(TypeError, match="gene"):
        pm.gintervals_to_mat(df)


def test_to_mat_explicit_non_numeric_rejected():
    df = pd.DataFrame({
        "chrom": ["1"], "start": [100], "end": [200],
        "gene": ["FOO"], "t1": [1.0],
    })
    with pytest.raises(TypeError, match="gene"):
        pm.gintervals_to_mat(df, value_cols=["gene", "t1"])


def test_to_mat_labels_false_returns_range_index():
    df = pd.DataFrame({
        "chrom": ["1", "2"], "start": [100, 200], "end": [200, 400],
        "t1": [1.0, 2.0],
    })
    mat = pm.gintervals_to_mat(df, labels=False)
    assert isinstance(mat.index, pd.RangeIndex)
    assert list(mat.columns) == ["t1"]
    np.testing.assert_array_equal(mat["t1"].values, df["t1"].values)


def test_from_mat_on_labels_false_raises():
    df = pd.DataFrame({"chrom": ["1"], "start": [100], "end": [200], "t1": [1.0]})
    mat = pm.gintervals_to_mat(df, labels=False)
    with pytest.raises(ValueError, match="MultiIndex"):
        pm.gintervals_from_mat(mat)


def test_to_mat_underscore_chrom_round_trip():
    df = pd.DataFrame({
        "chrom": ["chr_unplaced_1", "chr_unplaced_2"],
        "start": [100, 500],
        "end": [200, 700],
        "t1": [1.5, 2.5],
    })
    mat = pm.gintervals_to_mat(df)
    out = pm.gintervals_from_mat(mat)
    assert list(out["chrom"]) == list(df["chrom"])
    assert list(out["start"]) == list(df["start"])
    assert list(out["end"]) == list(df["end"])


def test_iloc_row_subset_preserves_intervals():
    df = pd.DataFrame({
        "chrom": ["1", "1", "2"], "start": [100, 500, 200], "end": [200, 700, 400],
        "t1": [1.0, 2.0, 3.0], "t2": [10.0, 20.0, 30.0],
    })
    mat = pm.gintervals_to_mat(df)
    sub = mat.iloc[[0, 2]]
    out = pm.gintervals_from_mat(sub)
    assert list(out["chrom"]) == ["1", "2"]
    assert list(out["start"]) == [100, 200]
    assert list(out["end"]) == [200, 400]
    np.testing.assert_array_equal(out["t1"].values, [1.0, 3.0])


def test_column_only_subset_preserves_rows():
    df = pd.DataFrame({
        "chrom": ["1", "2"], "start": [100, 200], "end": [200, 400],
        "t1": [1.0, 2.0], "t2": [10.0, 20.0],
    })
    mat = pm.gintervals_to_mat(df)
    sub = mat[["t1"]]
    assert len(sub) == 2
    assert list(sub.columns) == ["t1"]
    out = pm.gintervals_from_mat(sub)
    assert list(out["chrom"]) == ["1", "2"]
    np.testing.assert_array_equal(out["t1"].values, [1.0, 2.0])


def test_single_row_iloc_returns_one_row_dataframe():
    df = pd.DataFrame({
        "chrom": ["1"], "start": [100], "end": [200], "t1": [1.0],
    })
    mat = pm.gintervals_to_mat(df)
    sub = mat.iloc[[0]]
    assert len(sub) == 1
    out = pm.gintervals_from_mat(sub)
    assert list(out["chrom"]) == ["1"]


def test_head_round_trip():
    df = pd.DataFrame({
        "chrom": ["1", "1", "2", "2"], "start": [100, 500, 200, 700],
        "end": [200, 700, 400, 900], "t1": [1.0, 2.0, 3.0, 4.0],
    })
    mat = pm.gintervals_to_mat(df)
    out = pm.gintervals_from_mat(mat.head(2))
    assert list(out["chrom"]) == ["1", "1"]
    assert list(out["start"]) == [100, 500]


def test_concat_preserves_intervals():
    a = pd.DataFrame({
        "chrom": ["1"], "start": [100], "end": [200], "t1": [1.0],
    })
    b = pd.DataFrame({
        "chrom": ["2"], "start": [300], "end": [400], "t1": [2.0],
    })
    mat_a = pm.gintervals_to_mat(a)
    mat_b = pm.gintervals_to_mat(b)
    combined = pd.concat([mat_a, mat_b])
    out = pm.gintervals_from_mat(combined)
    assert list(out["chrom"]) == ["1", "2"]
    assert list(out["start"]) == [100, 300]
    assert list(out["end"]) == [200, 400]
    np.testing.assert_array_equal(out["t1"].values, [1.0, 2.0])


# R parity: "intervalID is excluded from value_cols by default but kept in attribute"
def test_to_mat_excludes_intervalID_from_value_cols():
    """R parity: 'intervalID is excluded from value_cols by default but kept in attribute'.

    In pymisha, intervalID is excluded from value_cols auto-detection but the
    column is dropped (not preserved as a separate level) because pymisha's
    MultiIndex carries chrom/start/end only. Verify intervalID does not leak
    into mat.columns.
    """
    df = pd.DataFrame({
        "chrom": ["1", "1"], "start": [100, 200], "end": [200, 300],
        "intervalID": [1, 2],
        "t1": [1.0, 2.0],
    })
    mat = pm.gintervals_to_mat(df)
    assert "intervalID" not in mat.columns
    assert list(mat.columns) == ["t1"]


# R parity: "C++ helper handles factor chrom column"
def test_to_mat_categorical_chrom_round_trip():
    """R parity: 'C++ helper handles factor chrom column'.

    pymisha has no C++ helper, but should still round-trip when chrom is
    Categorical (pandas equivalent of R factor).
    """
    df = pd.DataFrame({
        "chrom": pd.Categorical(["1", "1", "2"]),
        "start": [100, 500, 200],
        "end": [200, 700, 400],
        "t1": [1.5, 2.5, 3.5],
    })
    mat = pm.gintervals_to_mat(df)
    out = pm.gintervals_from_mat(mat)
    assert list(out["chrom"]) == ["1", "1", "2"]
    assert list(out["start"]) == [100, 500, 200]
    np.testing.assert_array_equal(out["t1"].values, df["t1"].values)


# R parity: "id_col is validated even when labels = FALSE"
def test_to_mat_id_col_validated_when_labels_false():
    """R parity: 'id_col is validated even when labels = FALSE'."""
    df = pd.DataFrame({"chrom": ["1"], "start": [100], "end": [200], "t1": [1.0]})
    with pytest.raises(ValueError, match="id_col"):
        pm.gintervals_to_mat(df, id_col="nope", labels=False)


# R parity: "head() and tail() dispatch through [ and preserve intervs_mat"
def test_tail_round_trip():
    """R parity: tail() preserves intervals MultiIndex."""
    df = pd.DataFrame({
        "chrom": ["1", "1", "2", "2"], "start": [100, 500, 200, 700],
        "end": [200, 700, 400, 900], "t1": [1.0, 2.0, 3.0, 4.0],
    })
    mat = pm.gintervals_to_mat(df)
    out = pm.gintervals_from_mat(mat.tail(2))
    assert list(out["chrom"]) == ["2", "2"]
    assert list(out["start"]) == [200, 700]


# R parity: "from_mat without intervals on plain matrix errors clearly"
def test_from_mat_on_plain_dataframe_raises():
    """R parity: from_mat without the required structure errors clearly."""
    plain = pd.DataFrame({"t1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="MultiIndex"):
        pm.gintervals_from_mat(plain)


# Parity sanity check: wrong MultiIndex level names raises
def test_from_mat_wrong_index_levels_raises():
    """from_mat requires the last three index levels to be chrom/start/end."""
    bad_index = pd.MultiIndex.from_arrays(
        [["a"], ["b"], ["c"]],
        names=("foo", "bar", "baz"),
    )
    plain = pd.DataFrame({"t1": [1.0]}, index=bad_index)
    with pytest.raises(ValueError, match="chrom"):
        pm.gintervals_from_mat(plain)
