"""A 2D track name as the iterator of a 2D track expression, as R misha 5.16.0 takes it: the
iterator is the track's own objects (rectangles or points) cut by the scope, in every function
that takes an iterator, gtrack_create included (a Rectangles track holding the iterator
rectangles). Also R's errors for an iterator whose dimension is not that of the intervals or of
the tracks, and for the iterators gintervals_quantiles and gintervals_mapply do not take.

R misha 5.16.0 is the spec: the expected values are what it gives on the same database (a copy
of the test database with the tracks made below).
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm
from pymisha import _pymisha

DF = pd.DataFrame
NAN = np.nan
COLS_2D = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
SUMMARY_NAMES = ["Total intervals", "NaN intervals", "Min", "Max", "Sum", "Mean", "Std dev"]
ERRORS = (ValueError, _pymisha.error)  # the errors raised in Python and in C++
INDEXED_DB = Path(__file__).resolve().parent / "fixtures" / "misha_5_16_import" / "indexed"

RA = DF({"chrom1": ["chr1"] * 4, "start1": [100, 1000, 3000, 5000], "end1": [300, 2000, 3500, 6000],
         "chrom2": ["chr1", "chr1", "chr1", "chr2"], "start2": [200, 1000, 8000, 100], "end2": [400, 2000, 9000, 500]})
RB = DF({"chrom1": ["chr1"] * 4, "start1": [200, 250, 1500, 5500], "end1": [250, 600, 2500, 5600],
         "chrom2": ["chr1", "chr1", "chr1", "chr2"], "start2": [300, 100, 1500, 0], "end2": [350, 250, 2500, 1000]})
PTS = DF({"chrom1": ["chr1"] * 5, "start1": [150, 220, 1200, 3100, 5050], "end1": [151, 221, 1201, 3101, 5051],
          "chrom2": ["chr1", "chr1", "chr1", "chr1", "chr2"], "start2": [300, 320, 1800, 8500, 200],
          "end2": [301, 321, 1801, 8501, 201], "value": [1.0, 2, 3, 5, 4]})

# the iterator rectangles of rA and of pP over the whole genome
RA_ROWS = [("chr1", 100, 300, "chr1", 200, 400), ("chr1", 1000, 2000, "chr1", 1000, 2000),
           ("chr1", 3000, 3500, "chr1", 8000, 9000), ("chr1", 5000, 6000, "chr2", 100, 500)]
PP_ROWS = [("chr1", 150, 151, "chr1", 300, 301), ("chr1", 220, 221, "chr1", 320, 321),
           ("chr1", 1200, 1201, "chr1", 1800, 1801), ("chr1", 3100, 3101, "chr1", 8500, 8501),
           ("chr1", 5050, 5051, "chr2", 200, 201)]
# rA cut by the scope SCOPE2D below, and the scope intervals the pieces come from
RA_IN_SCOPE = [("chr1", 100, 300, "chr1", 250, 400), ("chr1", 1000, 1200, "chr1", 1000, 1500),
               ("chr1", 1200, 2000, "chr1", 1000, 2000), ("chr1", 3000, 3200, "chr1", 8000, 9000),
               ("chr1", 5000, 5800, "chr2", 100, 300)]


def scope2d():
    return pm.gintervals_2d(["chr1", "chr1", "chr1"], [0, 1200, 5000], [1200, 3200, 5800],
                            ["chr1", "chr1", "chr2"], [250, 0, 0], [1500, 9000, 300])


@pytest.fixture(scope="module", autouse=True)
def db(tmp_path_factory):
    """A copy of the test database with the 2D tracks rA, rB (rectangles) and pP (points) and
    the 2D intervals set is2d."""
    root = tmp_path_factory.mktemp("iter2d") / "db"
    shutil.copytree(TESTDB_ROOT, root)
    pm.gdb_init(str(root))
    pm.gtrack_2d_create("rA", "d", RA, [5, 3, 2, 7])
    pm.gtrack_2d_create("rB", "d", RB, [11, 13, 17, 19])
    pts = root.parent / "pts.tsv"
    PTS.to_csv(pts, sep="\t", index=False)
    pm.gtrack_2d_import("pP", "d", str(pts))
    pm.gintervals_save(scope2d(), "is2d")
    yield root
    pm.gvtrack_clear()
    pm.gdb_init(str(TESTDB_ROOT))


@pytest.fixture(autouse=True)
def _serial(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    yield
    pm.gvtrack_clear()


def A2():
    return pm.gintervals_2d_all()


def rows(df):
    return [tuple(None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
                  for v in r) for r in df.itertuples(index=False)]


def check(df, cols, expected):
    """df has the columns and the rows (NaN written as NAN) R 5.16 gives."""
    assert list(df.columns) == cols
    assert rows(df) == rows(DF(expected, columns=cols))


def with_values(coords, *values, ids=None):
    """Rows of coords followed by a value of each of values and, if given, the intervalID."""
    out = []
    for i, c in enumerate(coords):
        row = (*c, *(v[i] for v in values))
        out.append(row if ids is None else (*row, ids[i]))
    return out


def ex_cols(*names):
    return COLS_2D + list(names) + ["intervalID"]


# --- gextract ---


def test_gextract_another_track_over_the_rectangles_of_a_2d_track():
    check(pm.gextract("rB", A2(), iterator="rA"), ex_cols("rB"),
          with_values(RA_ROWS, [12, 17, NAN, 19], ids=[1, 1, 1, 2]))


def test_gextract_the_track_over_its_own_rectangles():
    check(pm.gextract("rA", A2(), iterator="rA"), ex_cols("rA"),
          with_values(RA_ROWS, [5, 3, 2, 7], ids=[1, 1, 1, 2]))
    check(pm.gextract("pP", A2(), iterator="pP"), ex_cols("pP"),
          with_values(PP_ROWS, [1, 2, 3, 5, 4], ids=[1, 1, 1, 1, 2]))


def test_gextract_points_and_rectangles_tracks_over_each_other():
    check(pm.gextract("pP", A2(), iterator="rA"), ex_cols("pP"),
          with_values(RA_ROWS, [1.5, 3, 5, 4], ids=[1, 1, 1, 2]))
    check(pm.gextract("rA", A2(), iterator="pP"), ex_cols("rA"),
          with_values(PP_ROWS, [5, 5, 3, 2, 7], ids=[1, 1, 1, 1, 2]))
    check(pm.gextract("rB", A2(), iterator="pP"), ex_cols("rB"),
          with_values(PP_ROWS, [NAN, 11, NAN, NAN, NAN], ids=[1, 1, 1, 1, 2]))


def test_gextract_expression_of_two_tracks_and_two_expressions():
    check(pm.gextract("rB * 2 + rA", A2(), iterator="rA"), ex_cols("rB * 2 + rA"),
          with_values(RA_ROWS, [29, 37, NAN, 45], ids=[1, 1, 1, 2]))
    rb_rows = [tuple(r) for r in RB[COLS_2D].itertuples(index=False)]
    check(pm.gextract(["rA", "rB"], A2(), iterator="rB", colnames=["a", "b"]), ex_cols("a", "b"),
          with_values(rb_rows, [5, 5, 3, 7], [11, 13, 17, 19], ids=[1, 1, 1, 2]))


def test_gextract_with_band():
    check(pm.gextract("rB", A2(), iterator="rA", band=(-300, 0)), ex_cols("rB"),
          with_values(RA_ROWS[:2], [11, 17], ids=[1, 1]))
    check(pm.gextract("rB", A2(), iterator="rA", band=(-150, 1000)), ex_cols("rB"),
          with_values(RA_ROWS[:2], [12, 17], ids=[1, 1]))
    # a rectangle the band crosses is cut by it
    check(pm.gextract("rA", A2(), iterator="rA", band=(-6000, -100)), ex_cols("rA"),
          [("chr1", 100, 300, "chr1", 200, 400, 5, 1), ("chr1", 1000, 1900, "chr1", 1100, 2000, 3, 1),
           ("chr1", 3000, 3500, "chr1", 8000, 9000, 2, 1)])
    assert pm.gextract("rB", A2(), iterator="rA", band=(10000, 20000)) is None


def test_gextract_over_2d_intervals_cuts_the_rectangles_by_the_scope():
    expected = with_values(RA_IN_SCOPE, [11, NAN, 17, NAN, 19], ids=[1, 1, 2, 2, 3])
    check(pm.gextract("rB", scope2d(), iterator="rA"), ex_cols("rB"), expected)
    check(pm.gextract("rB", "is2d", iterator="rA"), ex_cols("rB"), expected)
    # the rows follow the iterator; intervalID is the row of the scope interval
    rev = scope2d().iloc[[2, 0, 1]].reset_index(drop=True)
    check(pm.gextract("rB", rev, iterator="rA"), ex_cols("rB"),
          with_values(RA_IN_SCOPE, [11, NAN, 17, NAN, 19], ids=[2, 2, 3, 3, 1]))
    check(pm.gextract("pP", scope2d(), iterator="rA"), ex_cols("pP"),
          with_values(RA_IN_SCOPE, [1.5, NAN, 3, 5, 4], ids=[1, 1, 2, 2, 3]))
    check(pm.gextract("rA", scope2d(), iterator="pP"), ex_cols("rA"),
          with_values(PP_ROWS, [5, 5, 3, 2, 7], ids=[1, 1, 2, 2, 3]))
    check(pm.gextract("rB", scope2d(), iterator="rA", band=(-300, 0)), ex_cols("rB"),
          [("chr1", 100, 300, "chr1", 250, 400, 11, 1), ("chr1", 1000, 1200, "chr1", 1000, 1500, NAN, 1),
           ("chr1", 1200, 2000, "chr1", 1200, 2000, 17, 2)])


def test_gextract_without_intervals_is_over_the_whole_2d_genome():
    # pymisha's intervals=None is R's ALLGENOME, whose 2D part a 2D iterator takes
    check(pm.gextract("rA", iterator="rA"), ex_cols("rA"), with_values(RA_ROWS, [5, 3, 2, 7], ids=[1, 1, 1, 2]))


def test_gextract_with_a_track_of_the_test_database_as_iterator():
    res = pm.gextract("rA", A2(), iterator="rects_track")
    assert len(res) == 1600 and res["rA"].isna().all()
    assert res["intervalID"].value_counts().to_dict() == {3: 937, 2: 587, 1: 76}
    assert rows(res.head(2)) == [("chr1", 239835, 240759, "chr1", 371562, 374071, None, 1),
                                 ("chr1", 5786, 9222, "chr1", 305335, 305872, None, 1)]
    res = pm.gextract("rects_track", pm.gintervals_2d("chr1", 0, 100000, "chr2", 0, 100000), iterator="rects_track")
    assert len(res) == 29 and res["rects_track"].sum() == 979386
    assert rows(res.head(4)) == [("chr1", 83371, 83625, "chr2", 87908, 89220, 32821, 1),
                                 ("chr1", 75525, 76251, "chr2", 86507, 86552, 56431, 1),
                                 ("chr1", 82279, 82686, "chr2", 84198, 85866, 32943, 1),
                                 ("chr1", 77917, 78089, "chr2", 88960, 90002, 37414, 1)]


def test_gextract_file_and_intervals_set_out(tmp_path):
    f = tmp_path / "out.tsv"
    assert pm.gextract("rA", A2(), iterator="rA", file=str(f)) is None
    check(pd.read_csv(f, sep="\t"), COLS_2D + ["rA"], with_values(RA_ROWS, [5, 3, 2, 7]))
    pm.gextract("rB", A2(), iterator="rA", intervals_set_out="eout")
    check(pm.gintervals_load("eout"), COLS_2D + ["rB"], with_values(RA_ROWS, [12, 17, NAN, 19]))


# --- gscreen ---


def test_gscreen():
    check(pm.gscreen("rB > 12", A2(), iterator="rA"), COLS_2D, [RA_ROWS[1], RA_ROWS[3]])
    check(pm.gscreen("rA > 2", A2(), iterator="rA"), COLS_2D, [RA_ROWS[0], RA_ROWS[1], RA_ROWS[3]])
    check(pm.gscreen("pP > 1", A2(), iterator="rA"), COLS_2D, RA_ROWS)
    check(pm.gscreen("rB > 0", scope2d(), iterator="rA"), COLS_2D, [RA_IN_SCOPE[0], RA_IN_SCOPE[2], RA_IN_SCOPE[4]])
    check(pm.gscreen("rB > 0", A2(), iterator="rA", band=(-300, 0)), COLS_2D, RA_ROWS[:2])
    check(pm.gscreen("rA > 2", A2(), iterator="pP"), COLS_2D, [PP_ROWS[0], PP_ROWS[1], PP_ROWS[2], PP_ROWS[4]])
    pm.gscreen("rB > 0", A2(), iterator="rA", intervals_set_out="sout")
    check(pm.gintervals_load("sout"), COLS_2D, [RA_ROWS[0], RA_ROWS[1], RA_ROWS[3]])


# --- gsummary, gquantiles, gdist ---


@pytest.mark.parametrize("expr, kwargs, expected", [
    ("rB", {"iterator": "rA"}, [4, 1, 12, 19, 48, 16, 3.60555127546399]),
    ("rA", {"iterator": "rA"}, [4, 0, 2, 7, 17, 4.25, 2.21735578260835]),
    ("pP", {"iterator": "rA"}, [4, 0, 1.5, 5, 13.5, 3.375, 1.49303940559741]),
    ("rA", {"iterator": "pP"}, [5, 0, 2, 7, 22, 4.4, 1.94935886896179]),
    ("rB", {"iterator": "rA", "band": (10000, 20000)}, [0, 0, NAN, NAN, NAN, NAN, NAN]),
])
def test_gsummary(expr, kwargs, expected):
    res = pm.gsummary(expr, A2(), **kwargs)
    assert list(res.index) == SUMMARY_NAMES
    np.testing.assert_allclose(res.to_numpy(dtype=float), expected, rtol=1e-12)


def test_gsummary_over_2d_intervals_with_band_and_without_intervals():
    np.testing.assert_allclose(pm.gsummary("rB", scope2d(), iterator="rA", band=(-300, 0)).to_numpy(dtype=float),
                               [3, 1, 11, 17, 28, 14, 4.24264068711928], rtol=1e-12)
    np.testing.assert_allclose(pm.gsummary("rA", iterator="rA").to_numpy(dtype=float),
                               [4, 0, 2, 7, 17, 4.25, 2.21735578260835], rtol=1e-12)


def test_gquantiles():
    res = pm.gquantiles("rB", [0.1, 0.5, 0.9], A2(), iterator="rA")
    assert list(res.index) == [0.1, 0.5, 0.9]
    np.testing.assert_allclose(res.to_numpy(), [13, 17, 18.6], rtol=1e-12)
    np.testing.assert_allclose(pm.gquantiles("rA", 0.5, A2(), iterator="pP").to_numpy(), [5])
    np.testing.assert_allclose(pm.gquantiles("rA", 0.5, iterator="rA").to_numpy(), [4])
    np.testing.assert_allclose(pm.gquantiles("rB", [0.2, 0.8], "is2d", iterator="rA").to_numpy(), [13.4, 18.2],
                               rtol=1e-12)


def test_gdist():
    np.testing.assert_array_equal(pm.gdist("rB", [0, 12, 20], intervals=A2(), iterator="rA"), [1, 2])
    np.testing.assert_array_equal(pm.gdist("rA", [0, 4, 8], iterator="rA"), [2, 2])
    # two expressions: counts[i, j] of the iterator rectangles with rA in bin i and rB in bin j
    np.testing.assert_array_equal(pm.gdist("rA", [0, 4, 8], "rB", [0, 15, 20], intervals=A2(), iterator="rA"),
                                  [[0, 1], [1, 1]])
    res = pm.gdist("rB", [0, 12, 20], intervals=scope2d(), iterator="rA", dataframe=True)
    assert res["n"].tolist() == [1, 2]


# --- gintervals_summary, gintervals_quantiles, giterator_intervals ---


IS_COLS = COLS_2D + SUMMARY_NAMES
SCOPE_ROWS = [("chr1", 0, 1200, "chr1", 250, 1500), ("chr1", 1200, 3200, "chr1", 0, 9000),
              ("chr1", 5000, 5800, "chr2", 0, 300)]


def test_gintervals_summary():
    expected = with_values(SCOPE_ROWS, [2, 2, 1], [1, 1, 0], [11, 17, 19], [11, 17, 19], [11, 17, 19],
                           [11, 17, 19], [NAN, NAN, NAN])
    check(pm.gintervals_summary("rB", scope2d(), iterator="rA"), IS_COLS, expected)
    check(pm.gintervals_summary("rB", "is2d", iterator="rA"), IS_COLS, expected)
    rev = scope2d().iloc[[2, 0, 1]].reset_index(drop=True)
    check(pm.gintervals_summary("rB", rev, iterator="rA"), IS_COLS, [expected[2], expected[0], expected[1]])
    check(pm.gintervals_summary("pP", scope2d(), iterator="rA", band=(-300, 0)), IS_COLS,
          with_values(SCOPE_ROWS, [2, 1, 0], [1, 1, 0], [1.5, NAN, NAN], [1.5, NAN, NAN], [1.5, NAN, NAN],
                      [1.5, NAN, NAN], [NAN, NAN, NAN]))
    res = pm.gintervals_summary("rA", A2(), iterator="rA")
    assert res["Total intervals"].tolist() == [3, 1, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_allclose(res.iloc[0, 6:].to_numpy(dtype=float), [3, 0, 2, 5, 10, 10 / 3, 1.52752523165195],
                               rtol=1e-12)


def test_gintervals_quantiles_refuses_a_2d_track_or_2d_intervals_iterator():
    msg = "The type of iterator is currently not supported by the function"
    with pytest.raises(ValueError, match=msg):
        pm.gintervals_quantiles("rB", [0.5, 0.9], scope2d(), iterator="rA")
    with pytest.raises(ValueError, match=msg):
        pm.gintervals_quantiles("rB", 0.5, scope2d())  # the rectangles of rB
    with pytest.raises(ValueError, match=msg):
        pm.gintervals_quantiles("rB", 0.5, scope2d(), iterator=scope2d())
    with pytest.raises(ValueError, match="1D iterator is used along with 2D intervals"):
        pm.gintervals_quantiles("dense_track", 0.5, scope2d())
    with pytest.raises(ValueError, match="1D iterator is used along with 2D intervals"):
        pm.gintervals_quantiles("rA", 0.5, scope2d(), iterator=100)
    with pytest.raises(ValueError, match="2D iterator is used along with 1D intervals"):
        pm.gintervals_quantiles("rA", 0.5, pm.gintervals("chr1"), iterator="rA")
    # a rectangle size is taken
    check(pm.gintervals_quantiles("rA", 0.5, scope2d(), iterator=(1000, 1000)), COLS_2D + ["0.5"],
          with_values(SCOPE_ROWS, [4, 2.5, 7]))


def test_giterator_intervals():
    check(pm.giterator_intervals("rB", A2(), iterator="rA"), COLS_2D, RA_ROWS)
    check(pm.giterator_intervals(intervals=A2(), iterator="rA"), COLS_2D, RA_ROWS)
    check(pm.giterator_intervals(iterator="rA"), COLS_2D, RA_ROWS)
    check(pm.giterator_intervals(intervals=scope2d(), iterator="pP"), COLS_2D, PP_ROWS)
    check(pm.giterator_intervals(intervals=A2(), iterator="rA", band=(-300, 0)), COLS_2D, RA_ROWS[:2])
    check(pm.giterator_intervals(intervals="is2d", iterator="rA"), COLS_2D, RA_IN_SCOPE)
    rev = scope2d().iloc[[2, 0, 1]].reset_index(drop=True)
    check(pm.giterator_intervals(intervals=rev, iterator="rA"), COLS_2D, RA_IN_SCOPE)


# --- gtrack_create ---


def created(name):
    info = pm.gtrack_info(name)
    return info["type"], pm.gextract(name, A2())


def test_gtrack_create_over_the_rectangles_of_a_2d_track():
    pm.gtrack_create("c_rb10", "d", "rB + 10", iterator="rA")
    kind, res = created("c_rb10")
    assert kind == "rectangles"
    check(res, ex_cols("c_rb10"), with_values(RA_ROWS, [22, 27, NAN, 29], ids=[1, 1, 1, 2]))
    assert pm.gtrack_attr_get("c_rb10", "created.by") == \
        'gtrack.create("c_rb10", description, \'rB + 10\', iterator="rA")'

    pm.gtrack_create("c_ra", "d", "rA", iterator="rA")
    check(created("c_ra")[1], ex_cols("c_ra"), with_values(RA_ROWS, [5, 3, 2, 7], ids=[1, 1, 1, 2]))
    pm.gtrack_create("c_sum", "d", "rA + rB", iterator="rB")
    rb_rows = [tuple(r) for r in RB[COLS_2D].itertuples(index=False)]
    check(created("c_sum")[1], ex_cols("c_sum"), with_values(rb_rows, [16, 18, 20, 26], ids=[1, 1, 1, 2]))


def test_gtrack_create_with_points_makes_a_rectangles_track():
    pm.gtrack_create("c_pp", "d", "pP", iterator="rA")
    kind, res = created("c_pp")
    assert kind == "rectangles"
    check(res, ex_cols("c_pp"), with_values(RA_ROWS, [1.5, 3, 5, 4], ids=[1, 1, 1, 2]))
    pm.gtrack_create("c_itp", "d", "rA", iterator="pP")
    kind, res = created("c_itp")
    assert kind == "rectangles"
    check(res, ex_cols("c_itp"), with_values(PP_ROWS, [5, 5, 3, 2, 7], ids=[1, 1, 1, 1, 2]))


def test_gtrack_create_with_band():
    pm.gtrack_create("c_band", "d", "rB", iterator="rA", band=(-300, 0))
    check(created("c_band")[1], ex_cols("c_band"), with_values(RA_ROWS[:2], [11, 17], ids=[1, 1]))
    # the implicit iterator, the rectangles of rB, takes the band too
    pm.gtrack_create("c_band_noit", "d", "rB", band=(-300, 0))
    check(created("c_band_noit")[1], ex_cols("c_band_noit"),
          [("chr1", 200, 250, "chr1", 300, 350, 11, 1), ("chr1", 1500, 2500, "chr1", 1500, 2500, 17, 1)])
    pm.gtrack_create("c_df_band", "d", "rB", iterator=scope2d(), band=(-300, 0))
    check(created("c_df_band")[1], ex_cols("c_df_band"),
          [("chr1", 0, 1200, "chr1", 250, 1500, 11, 1), ("chr1", 1200, 3200, "chr1", 1200, 3500, 17, 1)])


def test_gtrack_create_with_2d_intervals_and_rectangle_size_iterators():
    pm.gtrack_create("c_df", "d", "rB", iterator=scope2d())
    check(created("c_df")[1], ex_cols("c_df"), with_values(SCOPE_ROWS, [11, 17, 19], ids=[1, 1, 2]))
    pm.gtrack_create("c_rect", "d", "rA", iterator=(100000, 100000))
    assert pm.gtrack_attr_get("c_rect", "created.by") == \
        'gtrack.create("c_rect", description, \'rA\', iterator=c(1e+05, 1e+05))'
    kind, res = created("c_rect")
    assert kind == "rectangles" and len(res) == 100
    res = res[res["c_rect"].notna()]
    assert rows(res[COLS_2D]) == [("chr1", 0, 100000, "chr1", 0, 100000), ("chr1", 0, 100000, "chr2", 0, 100000)]
    # the value is stored as a float, as R stores it
    np.testing.assert_allclose(res["c_rect"].to_numpy(), [2.72727274894714, 7], rtol=1e-14)


def test_gtrack_create_with_a_virtual_track_and_with_multitasking(monkeypatch):
    pm.gvtrack_create("vPws", "pP", "weighted.sum")
    pm.gtrack_create("c_vt", "d", "vPws", iterator="rA")
    check(created("c_vt")[1], ex_cols("c_vt"), with_values(RA_ROWS, [3, 3, 5, 4], ids=[1, 1, 1, 2]))
    monkeypatch.setitem(pm.CONFIG, "multitasking", True)
    pm.gtrack_create("c_mt", "d", "rB + 10", iterator="rA")
    check(created("c_mt")[1], ex_cols("c_mt"), with_values(RA_ROWS, [22, 27, NAN, 29], ids=[1, 1, 1, 2]))


def no_rectangle_creates_no_track():
    """R 5.16.1: gtrack.create and gtrack.lookup over a 2D iterator that gives no
    rectangle (none of rA is in this band) are errors and leave no track."""
    from pymisha import _shared

    for call in (lambda: pm.gtrack_create("e1", "d", "rB", iterator="rA", band=(10000, 20000)),
                 lambda: pm.gtrack_lookup("l1", "d", np.array([1.0, 2.0, 3.0]), "rB", [0, 1, 2, 1e9],
                                          iterator="rA", band=(10000, 20000))):
        with pytest.raises(ValueError, match="was not created: the iterator gave no rectangles"):
            call()
    for name in ("e1", "l1"):
        assert not (Path(_shared._GROOT) / "tracks" / f"{name}.track").exists()
        assert name not in pm.gtrack_ls()


@pytest.mark.parametrize("mt", [False, True])
def test_gtrack_create_and_lookup_from_no_iterator_rectangle_create_no_track(monkeypatch, mt):  # R followup :93
    monkeypatch.setitem(pm.CONFIG, "multitasking", mt)
    no_rectangle_creates_no_track()


def test_gtrack_create_with_a_2d_iterator_on_an_indexed_database(tmp_path):
    root = tmp_path / "db"
    shutil.copytree(INDEXED_DB, root)
    (root / "tracks").mkdir()
    try:
        pm.gdb_init(str(root))
        pm.gtrack_2d_create("rA", "d", RA, [5, 3, 2, 7])
        pm.gtrack_create("e2", "d", "rA + 1", iterator="rA")
        info = pm.gtrack_info("e2")
        assert (info["type"], info["format"]) == ("rectangles", "indexed")
        check(pm.gextract("e2", A2()), ex_cols("e2"), with_values(RA_ROWS, [6, 4, 3, 8], ids=[1, 1, 1, 2]))
        pm.gtrack_2d_create("rB", "d", RB, [1, 2, 3, 4])
        no_rectangle_creates_no_track()
    finally:
        pm.gdb_init(str(TESTDB_ROOT))


def test_gtrack_lookup():
    pm.gtrack_lookup("c_lk", "d", np.array([10.0, 20.0]), "rA", [0, 4, 8], iterator="rA")
    kind, res = created("c_lk")
    assert kind == "rectangles"
    check(res, ex_cols("c_lk"), with_values(RA_ROWS, [20, 10, 10, 20], ids=[1, 1, 1, 2]))


# --- 2D virtual tracks ---


def test_2d_virtual_tracks_over_the_rectangles_of_a_2d_track():
    pm.gvtrack_create("vB", "rB", "area")
    pm.gvtrack_create("vBmax", "rB", "max")
    pm.gvtrack_create("vBavg", "rB", "avg")
    pm.gvtrack_create("vPws", "pP", "weighted.sum")
    pm.gvtrack_create("vPmin", "pP", "min")
    check(pm.gextract("vB", A2(), iterator="rA"), ex_cols("vB"),
          with_values(RA_ROWS, [5000, 250000, 0, 40000], ids=[1, 1, 1, 2]))
    check(pm.gextract("vBmax", A2(), iterator="rA"), ex_cols("vBmax"),
          with_values(RA_ROWS, [13, 17, NAN, 19], ids=[1, 1, 1, 2]))
    check(pm.gextract("vBavg", A2(), iterator="rA"), ex_cols("vBavg"),
          with_values(RA_ROWS, [12, 17, NAN, 19], ids=[1, 1, 1, 2]))
    check(pm.gextract("vPws", A2(), iterator="rA"), ex_cols("vPws"),
          with_values(RA_ROWS, [3, 3, 5, 4], ids=[1, 1, 1, 2]))
    check(pm.gextract("vPmin", A2(), iterator="rA"), ex_cols("vPmin"),
          with_values(RA_ROWS, [1, 3, 5, 4], ids=[1, 1, 1, 2]))
    check(pm.gextract("vB", A2(), iterator="pP"), ex_cols("vB"),
          with_values(PP_ROWS, [0, 1, 0, 0, 0], ids=[1, 1, 1, 1, 2]))
    check(pm.gextract("vB", A2(), iterator="rA", band=(-300, 0)), ex_cols("vB"),
          with_values(RA_ROWS[:2], [2500, 104850], ids=[1, 1]))
    check(pm.gextract("vB", scope2d(), iterator="rA"), ex_cols("vB"),
          with_values(RA_IN_SCOPE, [2500, 0, 250000, 0, 20000], ids=[1, 1, 2, 2, 3]))
    check(pm.gextract("vB + rA", A2(), iterator="rA"), ex_cols("vB + rA"),
          with_values(RA_ROWS, [5005, 250003, 2, 40007], ids=[1, 1, 1, 2]))
    np.testing.assert_allclose(pm.gsummary("vBavg", A2(), iterator="rA").to_numpy(dtype=float),
                               [4, 1, 12, 19, 48, 16, 3.60555127546399], rtol=1e-12)
    check(pm.gscreen("vBmax > 12", A2(), iterator="rA"), COLS_2D, [RA_ROWS[0], RA_ROWS[1], RA_ROWS[3]])


def test_2d_virtual_track_with_shifts_over_the_rectangles_of_a_2d_track():
    pm.gvtrack_create("vBs", "rB", "area")
    pm.gvtrack_iterator_2d("vBs", sshift1=-100, eshift1=100, sshift2=-100, eshift2=100)
    check(pm.gextract("vBs", A2(), iterator="rA"), ex_cols("vBs"),
          with_values(RA_ROWS, [25000, 360000, 0, 60000], ids=[1, 1, 1, 2]))


# --- the other functions that take an iterator ---


def test_gcor_gbins_glookup_gpartition_gsample():
    np.testing.assert_allclose(pm.gcor("rA", "rB", intervals=A2(), iterator="rA"), [0.277350098112615], rtol=1e-12)
    res = pm.gcor("rA", "rB", intervals=A2(), iterator="rA", details=True)
    np.testing.assert_allclose(res.iloc[0][["n", "n.na", "mean1", "mean2", "sd1", "sd2", "cov", "cor"]].to_numpy(float),
                               [4, 1, 5, 16, 2, 3.60555127546399, 2, 0.277350098112615], rtol=1e-12)
    res = pm.gbins_summary("rA", [0, 4, 8], expr="rB", intervals=A2(), iterator="rA")
    np.testing.assert_allclose(res, np.array([2, 2, 1, 0, 17, 12, 17, 19, 17, 31, 17, 15.5, NAN, 4.94974746830583])
                               .reshape(2, 7, order="F"), rtol=1e-12)
    np.testing.assert_allclose(pm.gbins_quantiles("rA", [0, 4, 8], expr="rB", percentiles=0.5, intervals=A2(),
                                                  iterator="rA").ravel(), [17, 15.5])
    expected = with_values(RA_ROWS, [20, 10, 10, 20], ids=[1, 1, 1, 2])
    check(pm.glookup(np.array([10.0, 20.0]), "rA", [0, 4, 8], intervals=A2(), iterator="rA"), ex_cols("value"),
          expected)
    check(pm.glookup(np.array([10.0, 20.0]), "rA", [0, 4, 8], intervals=A2()), ex_cols("value"), expected)
    check(pm.gpartition("rB", [0, 12, 20], A2(), iterator="rA"), COLS_2D + ["bin"],
          with_values([RA_ROWS[0], RA_ROWS[1], RA_ROWS[3]], [1, 2, 2]))
    check(pm.gpartition("rB", [0, 12, 20], scope2d(), iterator="rA"), COLS_2D + ["bin"],
          with_values([RA_IN_SCOPE[0], RA_IN_SCOPE[2], RA_IN_SCOPE[4]], [1, 2, 2]))
    vals = pm.gsample("rB", 10, A2(), iterator="rA")
    assert len(vals) == 3 and sorted(vals) == [12, 17, 19]


def test_gintervals_mapply_takes_only_a_rectangle_size_of_the_2d_iterators():
    msg = "The type of iterator is currently not supported by the function"
    with pytest.raises(ValueError, match=msg):
        pm.gintervals_mapply(np.max, "rB", intervals=A2(), iterator="rA")
    with pytest.raises(ValueError, match=msg):
        pm.gintervals_mapply(np.max, "rB", intervals=scope2d())
    with pytest.raises(ValueError, match="1D iterator is used along with 2D intervals"):
        pm.gintervals_mapply(np.max, "dense_track", intervals=scope2d())
    with pytest.raises(ValueError, match="2D iterator is used along with 1D intervals"):
        pm.gintervals_mapply(np.max, "rA", intervals=pm.gintervals("chr1"), iterator="rA")
    check(pm.gintervals_mapply(np.max, "rA", intervals=scope2d(), iterator=(1000, 1000)), COLS_2D + ["value"],
          with_values(SCOPE_ROWS, [NAN, NAN, 7]))


# --- R's errors ---


@pytest.mark.parametrize("call, msg", [
    (lambda: pm.gextract("rA", pm.gintervals("chr1"), iterator="rA"), "2D iterator is used along with 1D intervals"),
    (lambda: pm.gextract("rA", pm.gintervals_all(), iterator="rA"), "2D iterator is used along with 1D intervals"),
    (lambda: pm.gextract("rA", "annotations", iterator="rA"), "2D iterator is used along with 1D intervals"),
    (lambda: pm.gextract("dense_track", pm.gintervals("chr1"), iterator="is2d"),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.gextract("dense_track", pm.gintervals("chr1"), iterator=pm.gintervals_2d("chr1")),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.gscreen("rA > 0", pm.gintervals("chr1"), iterator="rA"), "2D iterator is used along with 1D intervals"),
    (lambda: pm.gsummary("rA", pm.gintervals("chr1"), iterator="rA"), "2D iterator is used along with 1D intervals"),
    (lambda: pm.gquantiles("rA", 0.5, intervals=pm.gintervals("chr1"), iterator="rA"),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.gdist("rA", [0, 4, 8], intervals=pm.gintervals("chr1"), iterator="rA"),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.gintervals_summary("rA", pm.gintervals("chr1"), iterator="rA"),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.giterator_intervals(intervals=pm.gintervals("chr1"), iterator="rA"),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.giterator_intervals(intervals=pm.gintervals("chr1"), iterator=(1000, 1000)),
     "2D iterator is used along with 1D intervals"),
    (lambda: pm.gextract("rA", A2(), iterator="dense_track"), "1D iterator is used along with 2D intervals"),
    (lambda: pm.gextract("rA", A2(), iterator="sparse_track"), "1D iterator is used along with 2D intervals"),
    (lambda: pm.gextract("rA", A2(), iterator="array_track"), "1D iterator is used along with 2D intervals"),
    (lambda: pm.gextract("rA", A2(), iterator=100), "1D iterator is used along with 2D intervals"),
    (lambda: pm.gextract("rA", A2(), iterator=pm.gintervals("chr1", 0, 1000)),
     "1D iterator is used along with 2D intervals"),
    (lambda: pm.giterator_intervals(intervals=A2(), iterator="dense_track"),
     "1D iterator is used along with 2D intervals"),
    (lambda: pm.gextract("rA", pm.gintervals_all(), iterator=100), "1D iterator is applied to a 2D track rA"),
    (lambda: pm.gextract("rA", pm.gintervals_all(), iterator="dense_track"), "1D iterator is applied to a 2D track rA"),
    (lambda: pm.gsummary("rA", pm.gintervals_all(), iterator="dense_track"), "1D iterator is applied to a 2D track rA"),
    (lambda: pm.gextract("dense_track", A2(), iterator="rA"),
     "2D iterator is applied to a 1D track dense_track without explicit dimension projection"),
    (lambda: pm.gextract("rA + dense_track", A2(), iterator="rA"),
     "2D iterator is applied to a 1D track dense_track without explicit dimension projection"),
])
def test_iterator_of_the_wrong_dimension_gives_the_error_of_r(call, msg):
    with pytest.raises(ERRORS, match="^" + re.escape(msg) + "$"):
        call()


def test_virtual_track_under_an_iterator_of_the_wrong_dimension_gives_the_error_of_r():
    pm.gvtrack_create("vB", "rB", "area")
    with pytest.raises(ValueError, match=r"^Virtual track vB: 1D iterator is applied to a 2D track rB$"):
        pm.gextract("vB", pm.gintervals("chr1", 0, 1000), iterator=100)
    pm.gvtrack_create("vd", "dense_track", "avg")
    msg = r"^Virtual track vd: 2D iterator is applied to a 1D track dense_track without explicit dimension projection$"
    with pytest.raises(ValueError, match=msg):
        pm.gextract("vd", A2(), iterator="rA")
    with pytest.raises(ValueError, match=msg):
        pm.gtrack_create("c_vd", "d", "vd", iterator="rA")


@pytest.mark.parametrize("iterator", [100, "dense_track", "sparse_track", "intervals"])
def test_gtrack_create_of_a_2d_track_with_a_1d_iterator_gives_the_error_of_r(iterator):
    if iterator == "intervals":
        iterator = pm.gintervals("chr1", 0, 1000)
    with pytest.raises(ERRORS, match=r"^1D iterator is applied to a 2D track rA$"):
        pm.gtrack_create("c_bad", "d", "rA", iterator=iterator)
    assert not pm.gtrack_exists("c_bad")


def test_gtrack_create_of_a_1d_track_with_a_2d_iterator_gives_the_error_of_r():
    with pytest.raises(ValueError, match=r"^2D iterator is applied to a 1D track dense_track without explicit "
                                         r"dimension projection$"):
        pm.gtrack_create("c_bad", "d", "dense_track", iterator="rA")
    assert not pm.gtrack_exists("c_bad")
    assert not os.path.exists(os.path.join(pm.gdir_cwd(), "c_bad.track"))
