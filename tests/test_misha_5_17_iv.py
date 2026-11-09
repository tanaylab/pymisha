"""Interval and database behaviour of R misha 5.17.0 (misha PR #222) that pymisha follows, ported
from the R tests that PR added or changed (test-gintervals2.R, test-gintervals.canonic.R,
test-gintervals.liftover-canonic.R, test-giterator.intervals-relative.R, test-gtrack.import-paired.R,
test-gtrack.2d.import-parallel.R, test-review-2026-11-boundary.R, test-gintervals.normalize.R,
test-input-format-ergonomics.R, test-gintervals.neighbors.R, test-gintervals.annotate.R,
test-review-2026-11-dbstate.R, test-dataset.R, test-track-copy-crossdb.R) and checked against
R 5.17.0 on the same inputs."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _chromdb_helpers import create_test_db

import pymisha as pm
from pymisha import _pymisha

DF = pd.DataFrame
ERRORS = (ValueError, _pymisha.error)


def in_other_session(db, code):
    """Runs code in another Python process with db loaded: another session on the same database."""
    script = "import sys\nimport pymisha as pm\npm.CONFIG['multitasking'] = False\n" \
        "pm.gdb_init(sys.argv[1])\n" + textwrap.dedent(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    res = subprocess.run([sys.executable, "-c", script, str(db)], env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res


def _values(col):
    """A column's values as objects, a missing value as None."""
    return [None if pd.isna(v) else v for v in col.astype(object)]


# --- gintervals_rbind binds columns as R's rbind() does ---------------------------------------

def test_rbind_binds_categoricals_with_different_categories_by_their_values(tmp_path):
    a = DF({"chrom": pd.Categorical(["chr2"]), "start": [0.0], "end": [100.0]})
    b = DF({"chrom": pd.Categorical(["chr1"]), "start": [500.0], "end": [600.0]})
    r = pm.gintervals_rbind(a, b)
    assert r["chrom"].tolist() == ["chr2", "chr1"]
    # one categorical, its categories in the order met, as R's factor levels
    assert isinstance(r["chrom"].dtype, pd.CategoricalDtype)
    assert list(r["chrom"].cat.categories) == ["chr2", "chr1"]

    # an extra categorical column of saved sets
    a["name"] = pd.Categorical(["x"])
    b["name"] = pd.Categorical(["y"])
    pm.gdb_init(create_test_db(tmp_path / "db"))
    pm.gintervals_save(a, "tmprbind_a")
    pm.gintervals_save(b, "tmprbind_b")
    assert pm.gintervals_rbind("tmprbind_a", "tmprbind_b")["name"].tolist() == ["x", "y"]


def test_rbind_binds_a_categorical_with_strings_and_an_integer_with_a_float():
    fac = DF({"chrom": pd.Categorical(["chr1"]), "start": [0.0], "end": [10.0]})
    chr_ = DF({"chrom": ["chr1"], "start": [5], "end": [20]})
    r = pm.gintervals_rbind(fac, chr_)
    assert r["chrom"].tolist() == ["chr1", "chr1"]
    assert isinstance(r["chrom"].dtype, pd.CategoricalDtype)
    assert r["start"].dtype == np.float64 and r["start"].tolist() == [0.0, 5.0]
    # strings first: strings, as R's character column
    r = pm.gintervals_rbind(chr_, fac)
    assert not isinstance(r["chrom"].dtype, pd.CategoricalDtype)
    assert r["chrom"].tolist() == ["chr1", "chr1"]
    assert r["start"].tolist() == [5.0, 0.0]


def _random_frame(rng):
    chroms = ["chr1", "chr2", "chrX"]
    n = int(rng.integers(1, 5))
    d = DF({"chrom": rng.choice(chroms, n), "start": np.arange(1, n + 1) * 10.0, "end": np.arange(1, n + 1) * 10.0 + 5,
                "lab": pd.Series(rng.choice(np.array(["p", "q", "r", None], dtype=object), n), dtype=object)})
    if rng.random() < 0.7:
        d["chrom"] = pd.Categorical(d["chrom"], categories=list(rng.permutation(chroms)))
    if rng.random() < 0.5:
        d["lab"] = pd.Categorical(d["lab"], categories=list(rng.permutation(["p", "q", "r", "s"])))
    if rng.random() < 0.5:
        d["start"] = d["start"].astype(np.int64)
    return d


def _expected_rbind_col(cols):
    """The column R's rbind() makes, in pandas terms: categorical if the first is, with the
    categories met in order; else strings or numbers. A column of None only (R's logical NA)
    takes the type of the others."""
    typed = [c for c in cols if not (c.dtype == object and c.isna().all())] or cols
    first = typed[0]
    values = [v for c in cols for v in _values(c)]
    if isinstance(first.dtype, pd.CategoricalDtype):
        levels = []
        for c in cols:
            for v in (c.cat.categories if isinstance(c.dtype, pd.CategoricalDtype) else c.dropna()):
                if v not in levels:
                    levels.append(v)
        return values, levels
    return values, None


def test_rbind_matches_rs_rbind_on_categorical_and_string_columns_with_missing_values():
    rng = np.random.default_rng(60427)
    for i in range(200):
        objs = [_random_frame(rng) for _ in range(int(rng.integers(2, 5)))]
        r = pm.gintervals_rbind(*objs)
        for col in ("chrom", "lab"):
            values, levels = _expected_rbind_col([o[col] for o in objs])
            assert _values(r[col]) == values, (i, col)
            if levels is None:
                assert not isinstance(r[col].dtype, pd.CategoricalDtype), (i, col)
            else:
                assert list(r[col].cat.categories) == levels, (i, col)
        ints = all(o["start"].dtype == np.int64 for o in objs)
        assert r["start"].dtype == (np.int64 if ints else np.float64), i
        assert r["start"].tolist() == [v for o in objs for v in o["start"]], i


def test_rbind_refuses_columns_of_different_types():
    a = DF({"chrom": ["chr1"], "start": [0], "end": [10], "x": [1.0]})
    b = DF({"chrom": ["chr1"], "start": [5], "end": [20], "x": ["a"]})
    with pytest.raises(ValueError, match="column 'x' is numeric in object 1 and character in object 2"):
        pm.gintervals_rbind(a, b)


def test_rbind_keeps_ordered_and_binds_a_column_of_missing_values_as_rbind_does():
    ordered = pd.Categorical(["p"], categories=["q", "p"], ordered=True)
    x = DF({"chrom": ["chr1"], "start": [0], "end": [10], "v": ordered})
    y = DF({"chrom": ["chr1"], "start": [20], "end": [30], "v": pd.Categorical([None])})
    r = pm.gintervals_rbind(x, y)
    assert r["v"].cat.ordered
    assert list(r["v"].cat.categories) == ["q", "p"]
    assert _values(r["v"]) == ["p", None]

    # a column of missing values only (R's logical NA) takes the other column's type
    n = DF({"chrom": ["chr1"], "start": [40], "end": [50], "v": [None]})
    others = [pd.Categorical(["a"]), ["a"], [5], [2.5], pd.to_datetime(["2020-01-01"]),
              pd.Categorical(["p"], categories=["q", "p"], ordered=True)]
    for other in others:
        o = DF({"chrom": ["chr1"], "start": [0], "end": [10], "v": other})
        for objs, nas in (((o, n), [False, True]), ((n, o), [True, False])):
            r = pm.gintervals_rbind(*objs)
            assert [pd.isna(v) for v in r["v"]] == nas, other
            if isinstance(o["v"].dtype, pd.CategoricalDtype):
                assert isinstance(r["v"].dtype, pd.CategoricalDtype)
                assert r["v"].cat.ordered == o["v"].cat.ordered
            elif o["v"].dtype.kind == "M":
                assert r["v"].dtype == o["v"].dtype
            elif o["v"].dtype.kind in "if":
                assert r["v"].dtype == np.float64
            else:
                assert r["v"].dtype == o["v"].dtype
    r = pm.gintervals_rbind(n, DF({"chrom": ["chr1"], "start": [0], "end": [10], "v": pd.Categorical(["a"])}))
    assert _values(r["v"]) == [None, "a"]
    # a boolean column with values is not untyped
    l_ = DF({"chrom": ["chr1"], "start": [40], "end": [50], "v": [True]})
    with pytest.raises(ValueError, match="column 'v'"):
        pm.gintervals_rbind(DF({"chrom": ["chr1"], "start": [0], "end": [10], "v": ["a"]}), l_)


# --- gintervals_load(chrom=) and gintervals_update() with a categorical chrom -------------------

def test_load_and_update_take_a_categorical_chrom_with_some_of_the_categories(tmp_path):
    pm.gdb_init(create_test_db(tmp_path / "db", [("chr1", 10000), ("chr2", 10000)]))
    df = DF({"chrom": pd.Categorical(["chr1", "chr1"]), "start": [5, 50], "end": [10, 60]})
    assert pm.gintervals_load(df, chrom="chr1")["start"].tolist() == [5, 50]

    pm.gintervals_save(pm.gintervals(["chr1", "chr2"], 0, 100), "tmpupdate")
    pm.gintervals_update("tmpupdate", df, chrom="chr1")
    r = pm.gintervals_load("tmpupdate")
    assert r["chrom"].astype(str).tolist() == ["chr1", "chr1", "chr2"]
    assert r["start"].tolist() == [5, 50, 0]

    d2 = DF({"chrom1": pd.Categorical(["chr1"]), "start1": [1], "end1": [5], "chrom2": pd.Categorical(["chr2"]),
                 "start2": [1], "end2": [5]})
    pm.gintervals_save(pm.gintervals_2d(["chr1", "chr2"], 0, 100, ["chr2", "chr2"], 0, 50), "tmpupdate2")
    pm.gintervals_update("tmpupdate2", d2, chrom1="chr1", chrom2="chr2")
    assert pm.gintervals_load("tmpupdate2", chrom1="chr1", chrom2="chr2")["start1"].tolist() == [1]


# --- an NA flag is an error, not True ----------------------------------------------------------

def test_canonic_refuses_unify_touching_intervals_na():
    iv = pm.gintervals("chr1", [0, 100, 300], [100, 200, 400])
    assert len(pm.gintervals_canonic(iv, unify_touching_intervals=True)) == 2
    assert len(pm.gintervals_canonic(iv, unify_touching_intervals=False)) == 3
    for bad in (np.nan, pd.NA, [True, False]):
        with pytest.raises(ValueError, match="unify_touching_intervals argument must be True or False"):
            pm.gintervals_canonic(iv, unify_touching_intervals=bad)


def test_liftover_refuses_an_na_canonic_or_include_metadata(tmp_path):
    chain = tmp_path / "c.chain"
    chain.write_text("chain 1000 source1 300 + 0 200 chr1 500000 + 0 100 1\n50\t100\t0\n50\n\n")
    src = DF({"chrom": ["source1"], "start": [0], "end": [200]})
    # an NA was read as True
    for bad in (np.nan, pd.NA):
        with pytest.raises(ValueError, match="canonic must be True or False"):
            pm.gintervals_liftover(src, str(chain), canonic=bad)
        with pytest.raises(ValueError, match="include_metadata must be True or False"):
            pm.gintervals_liftover(src, str(chain), include_metadata=bad)
    assert len(pm.gintervals_liftover(src, str(chain), canonic=True)) == 1


def test_interval_relative_na_is_an_error():
    intervs = pm.gintervals("chr1", [100, 500], [300, 700])
    for bad in (np.nan, pd.NA, [True, False]):
        with pytest.raises(ValueError, match="interval_relative must be True or False"):
            pm.giterator_intervals(None, intervs, iterator=50, interval_relative=bad)


def test_mappedseq_remove_dups_na_is_an_error(tmp_path):
    pm.gdb_init(create_test_db(tmp_path / "db"))
    sam = tmp_path / "reads.sam"
    sam.write_text("")
    with pytest.raises(ValueError, match="remove_dups argument must be True or False"):
        pm.gtrack_import_mappedseq("na_dups", "x", str(sam), binsize=50, cols_order=None, paired=True,
                                   remove_dups=np.nan)
    assert not pm.gtrack_exists("na_dups")


def test_contacts_import_refuses_allow_duplicates_na(tmp_path):
    pm.gdb_init(create_test_db(tmp_path / "db"))
    contacts = tmp_path / "contacts.tsv"
    contacts.write_text("chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tobs\nchr1\t0\t10\tchr1\t20\t30\t1\n")
    with pytest.raises(ValueError, match="not a boolean"):
        pm.gtrack_2d_import_contacts("na_dup", "test", [str(contacts)], allow_duplicates=np.nan)
    assert not pm.gtrack_exists("na_dup")


# --- a missing, infinite or out-of-range coordinate is an invalid interval ----------------------

def _one_d(start, end, dtype=None):
    d = DF({"chrom": ["chr1"], "start": [start], "end": [end]})
    return d.astype({"start": dtype, "end": dtype}) if dtype else d


@pytest.mark.parametrize("iv", [_one_d(pd.NA, 10, "Int64"), _one_d(0, pd.NA, "Int64"), _one_d(np.nan, 10.0),
                                _one_d(None, 10), _one_d(-np.inf, 10.0), _one_d(0.0, np.inf), _one_d(0.0, 1e30),
                                _one_d(0.0, 9.3e18)])
def test_a_missing_or_out_of_range_coordinate_is_an_invalid_interval(iv):
    # NaN and infinite values cast to -9223372036854775808, which the error then named
    for f in (pm.gseq_extract, lambda x: pm.gextract("dense_track", x), pm.gintervals_canonic):
        with pytest.raises(ERRORS, match="Invalid format of interval at index 1"):
            f(iv)


@pytest.mark.parametrize("col", ["start1", "end1", "start2", "end2"])
def test_a_missing_or_infinite_2d_coordinate_is_an_invalid_interval(col):
    infinite = -np.inf if col.startswith("start") else np.inf
    for bad, dtype in ((pd.NA, "Int64"), (np.nan, "float64"), (infinite, "float64")):
        two_d = DF({"chrom1": ["chr1"], "start1": [0], "end1": [100], "chrom2": ["chr1"], "start2": [0], "end2": [100]})
        two_d = two_d.astype(dict.fromkeys(("start1", "end1", "start2", "end2"), dtype))
        two_d[col] = pd.array([bad], dtype=dtype)
        for f in (pm.gintervals_canonic, lambda x: pm.gextract("rects_track", x)):
            with pytest.raises(ERRORS, match="Invalid format of interval at index 1"):
                f(two_d)


def test_a_coordinate_within_range_past_the_chromosome_is_still_named():
    with pytest.raises(ERRORS, match="end coordinate exceeds chromosome boundaries"):
        pm.gextract("dense_track", _one_d(0.0, 9.1e18))


def test_gintervals_errors_print_coordinates_as_whole_numbers():
    chr1_end = int(pm.gintervals_all().set_index("chrom").loc["chr1", "end"])
    with pytest.raises(ValueError, match=rf"Invalid interval \(chr1, 400000, {chr1_end + 1}\): end"):
        pm.gintervals("chr1", 400000.0, float(chr1_end + 1))
    with pytest.raises(ValueError, match=r"Invalid interval \(chr1, -12345678, 10\): start"):
        pm.gintervals("chr1", -12345678.0, 10.0)
    with pytest.raises(ValueError, match=r"Invalid interval \(chr1, 22345, 12345\): start"):
        pm.gintervals("chr1", 22345.0, 12345.0)


# --- CONFIG['max_data_size'] -------------------------------------------------------------------

def test_max_data_size_none_means_no_limit_for_normalize_force_range_and_union(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "max_data_size", None)
    iv = pm.gintervals(["chr1", "chr2"], 0, 1000)
    assert len(pm.gintervals_normalize(iv, 10)) == 2
    assert len(pm.gintervals_force_range(iv)) == 2
    assert len(pm.gintervals_union(iv, pm.gintervals("chr1", 500, 2000))) == 2


def test_a_gextract_result_over_max_data_size_is_reported_in_rows(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 100)
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    # dense_track has 50 bp bins: 2000 rows
    with pytest.raises(_pymisha.error) as err:
        pm.gextract("dense_track", pm.gintervals("chr1", 0, 100000))
    msg = str(err.value)
    assert ("Result size (at least 101 rows) exceeded the maximum allowed (100 rows, set by "
            "CONFIG['max_data_size'])") in msg
    assert "CONFIG['max_data_size'] = <a larger number of rows>, or None for no limit" in msg
    # the first suggestion: a file output, which the limit does not bound
    assert msg.index('gextract(..., file="x.tsv") or file="x.parquet"') < msg.index("Raise the limit")
    assert not any(unit in msg for unit in ("MB", "GB", "bytes"))


def test_the_file_suggestion_is_gextracts_only(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 10)
    q2 = pm.gintervals_2d("chr1", 0, 20000, "chr1", 0, 20000)
    s = np.arange(0, 5000, 100)
    t2 = DF({"chrom1": "chr1", "start1": s, "end1": s + 5, "chrom2": "chr1", "start2": 0, "end2": 5})
    with pytest.raises(_pymisha.error) as err:
        pm.gintervals_neighbors(q2, t2, maxneighbors=20)
    assert "file=" not in str(err.value)


def test_gextract_still_reads_the_callers_variables():
    factor = 3.0  # noqa: F841 (read by the expression)
    iv = pm.gintervals("chr1", 0, 1000)

    def inner():
        scale = 2.0  # noqa: F841 (read by the expression)
        return pm.gextract("dense_track * scale * factor", iv, iterator=500)

    base = pm.gextract("dense_track", iv, iterator=500)["dense_track"].to_numpy()
    np.testing.assert_allclose(inner().iloc[:, 3].to_numpy(), base * 6)


def test_normalize_returns_shorter_windows_at_chromosome_ends():
    chr1_end = int(pm.gintervals_all().set_index("chrom").loc["chr1", "end"])
    r = pm.gintervals_normalize(pm.gintervals("chr1", [100, chr1_end - 100], [300, chr1_end]), 1000)
    assert r["start"].tolist() == [0, chr1_end - 550]
    assert r["end"].tolist() == [700, chr1_end]


# --- gintervals_neighbors: ties broken by the neighbor's coordinates --------------------------

def test_neighbor_ties_are_broken_by_coordinates_not_input_order():
    i1 = DF({"chrom": ["chr1"], "start": [1000], "end": [1100]})
    # all four at distance 100; a and b are the same interval
    i2 = DF({"chrom": "chr1", "start": [1200, 1200, 800, 1200], "end": [1300, 1300, 900, 1250], "id": list("abcd")})
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]):
        x = i2.iloc[order]
        assert pm.gintervals_neighbors(i1, x)["id"].tolist() == ["c"]
        assert pm.gintervals_neighbors(i1, x, maxneighbors=2)["id"].tolist() == ["c", "d"]
        assert pm.gintervals_neighbors(i1, x, maxneighbors=4)["id"].tolist()[:2] == ["c", "d"]
    # only the same intervals keep their input order
    assert pm.gintervals_neighbors(i1, i2, maxneighbors=4)["id"].tolist() == ["c", "d", "a", "b"]
    assert pm.gintervals_neighbors(i1, i2.iloc[::-1], maxneighbors=4)["id"].tolist() == ["c", "d", "b", "a"]


@pytest.mark.parametrize("bounded", [False, True])
def test_2d_neighbors_picks_the_nearest_by_dist1_plus_dist2_whatever_the_input_order(bounded):
    kw = {"maxdist1": 1000, "maxdist2": 1000} if bounded else {}
    q = DF({"chrom1": ["chr1"], "start1": [1000], "end1": [1100], "chrom2": ["chr1"], "start2": [1000], "end2": [1100]})
    # D: dist1 2, dist2 1 (3); E: dist1 4, dist2 0 (4); F: dist1 3, dist2 0 (3)
    n = DF({"chrom1": "chr1", "start1": [1102, 1104, 1103], "end1": [1200, 1200, 1150], "chrom2": "chr1",
                "start2": [1101, 1050, 1070], "end2": [1200, 1060, 1080], "id": list("DEF")})
    for order in ([0, 1], [1, 0]):
        assert pm.gintervals_neighbors(q, n.iloc[order], **kw)["id"].tolist() == ["D"]
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        res = pm.gintervals_neighbors(q, n.iloc[order], maxneighbors=2, **kw)
        assert res["id"].tolist() == ["D", "F"]
        assert (res["dist1"] + res["dist2"]).tolist() == [3, 3]


def test_tied_candidates_do_not_count_against_max_data_size(monkeypatch):
    # 1000 intervals overlap the query; one is kept
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 100)
    starts = np.arange(0, 10000, 10)
    t = DF({"chrom": "chr1", "start": starts, "end": starts + 5})
    r = pm.gintervals_neighbors(pm.gintervals("chr1", 0, 20000), t)
    assert len(r) == 1 and r["start1"].tolist() == [0]
    q2 = pm.gintervals_2d("chr1", 0, 20000, "chr1", 0, 20000)
    t2 = DF({"chrom1": "chr1", "start1": starts, "end1": starts + 5, "chrom2": "chr1", "start2": 0, "end2": 5})
    r2 = pm.gintervals_neighbors(q2, t2)
    assert len(r2) == 1 and r2["start11"].tolist() == [0]


def _gap(s1, e1, s2, e2):
    return np.where((s2 <= e1) & (e2 >= s1), 0, np.maximum(s2 - e1, s1 - e2))


def test_neighbors_at_distance_0_are_chosen_by_coordinates_when_a_query_overlaps_many():
    rng = np.random.default_rng(60427)
    s = rng.integers(0, 20001, 3000)
    t = DF({"chrom": "chr1", "start": s, "end": s + rng.choice([1, 10, 100, 1000], 3000)})
    t = pd.concat([t, t.iloc[:50]], ignore_index=True)  # the same intervals twice
    t["id"] = np.arange(1, len(t) + 1)
    qs = rng.choice(20001, 40, replace=False)
    q = DF({"chrom": "chr1", "start": qs, "end": qs + rng.choice([5, 500, 5000], 40)})
    q["qid"] = np.arange(1, 41)
    for k in (1, 3):
        res = pm.gintervals_neighbors(q, t, maxneighbors=k)
        for i in range(len(q)):
            d = _gap(q["start"][i], q["end"][i], t["start"].to_numpy(), t["end"].to_numpy())
            want = t["id"].to_numpy()[np.lexsort((t["id"], t["end"], t["start"], d))][:k]
            assert res.loc[res["qid"] == i + 1, "id"].tolist() == want.tolist(), (k, i)

    g = [(i, j) for j in range(60) for i in range(60)]
    gi = np.array([x[0] for x in g])
    gj = np.array([x[1] for x in g])
    t2 = DF({"chrom1": "chr1", "start1": gi * 100, "end1": gi * 100 + rng.choice([50, 100], len(g)),
                 "chrom2": "chr1", "start2": gj * 100, "end2": gj * 100 + 50})
    t2["id"] = np.arange(1, len(t2) + 1)
    s1 = rng.choice(5001, 40, replace=False)
    s2 = rng.choice(5001, 40, replace=False)
    q2 = DF({"chrom1": "chr1", "start1": s1, "end1": s1 + rng.choice([10, 300, 2000], 40),
             "chrom2": "chr1", "start2": s2,
                 "end2": s2 + 700})
    q2["qid"] = np.arange(1, 41)
    for k in (1, 3):
        res = pm.gintervals_neighbors(q2, t2, maxneighbors=k)
        for i in range(len(q2)):
            d = (_gap(q2["start1"][i], q2["end1"][i], t2["start1"].to_numpy(), t2["end1"].to_numpy())
                 + _gap(q2["start2"][i], q2["end2"][i], t2["start2"].to_numpy(), t2["end2"].to_numpy()))
            order = np.lexsort((t2["id"], t2["end2"], t2["start2"], t2["end1"], t2["start1"], d))
            want = t2["id"].to_numpy()[order][:k]
            assert res.loc[res["qid"] == i + 1, "id"].tolist() == want.tolist(), ("2D", k, i)


# --- 2D gintervals_annotate returns dist1 and dist2 --------------------------------------------

def test_annotate_returns_both_distances_of_a_2d_neighbor():
    q = pm.gintervals_2d("chr1", 5000, 5100, "chr1", 9000, 9100)
    a = pm.gintervals_2d("chr1", 7000, 7100, "chr1", 9500, 9600)
    a["label"] = "B"
    nb = pm.gintervals_neighbors(q, a)
    assert [nb["dist1"][0], nb["dist2"][0]] == [1900, 400]

    res = pm.gintervals_annotate(q, a)
    assert res["label"].tolist() == ["B"]
    assert [res["dist1"][0], res["dist2"][0]] == [1900, 400]
    assert "dist" not in res.columns

    res = pm.gintervals_annotate(q, a, dist_column="d")
    assert [res["d1"][0], res["d2"][0]] == [1900, 400]
    res = pm.gintervals_annotate(q, a, dist_column=None)
    assert not {"dist", "dist1", "dist2"} & set(res.columns)

    # no neighbor in range
    res = pm.gintervals_annotate(q, a, maxdist1=10)
    assert pd.isna(res["label"][0]) and pd.isna(res["dist1"][0]) and pd.isna(res["dist2"][0])

    # max_dist limits dist1 + dist2 (2300 here); dist1 alone (1900) is within 2000
    assert pm.gintervals_annotate(q, a, max_dist=2300)["label"].tolist() == ["B"]
    res = pm.gintervals_annotate(q, a, max_dist=2000)
    assert pd.isna(res["label"][0])
    assert [res["dist1"][0], res["dist2"][0]] == [1900, 400]

    q["dist2"] = 1
    with pytest.raises(ValueError, match="Distance column 'dist2' already exists"):
        pm.gintervals_annotate(q, a)


# --- database state ----------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:Generating full 2D genome")
@pytest.mark.parametrize("multitasking", [False, True])
def test_the_whole_genome_2d_scope_follows_a_change_of_multicontig_2d_mode(monkeypatch, multitasking):
    # pymisha builds the scope each time (R keeps it with gmulticontig.2d.cache), in the current mode
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_threshold", 2)
    monkeypatch.setitem(pm.CONFIG, "multitasking", multitasking)
    cis = len(pm.gextract("rects_track", pm.gintervals_2d_all(mode="diagonal")))
    every = len(pm.gextract("rects_track", pm.gintervals_2d_all(mode="full")))
    assert cis < every
    for mode, n, pairs in (("diagonal", cis, 3), ("full", every, 9), ("diagonal", cis, 3)):
        monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", mode)
        assert pm.gsummary("rects_track")["Total intervals"] == n, mode
        assert len(pm.gextract("rects_track")) == n, mode
        assert len(pm.gintervals_2d_all()) == pairs
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", None)
    with pytest.raises(ValueError, match="multicontig_2d_mode"):
        pm.gsummary("rects_track")


def test_a_refused_reload_of_a_loaded_dataset_leaves_it_loaded(tmp_path, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    work, ds1, ds2 = (create_test_db(tmp_path / n) for n in ("working_db", "dataset1", "dataset2"))
    pm.gdb_init(ds1)
    intervs = pm.gintervals("chr1", 0, 1000)
    pm.gtrack_create_sparse("ds_only", "", intervs, [1])
    pm.gtrack_create_sparse("shared", "", intervs, [10])
    pm.gdb_init(ds2)
    pm.gtrack_create_sparse("shared", "", intervs, [20])
    pm.gdb_init(work)
    pm.gintervals_save(pm.gintervals("chr1", 0, 100), "foo")
    pm.gdataset_load(ds2)
    pm.gdataset_load(ds1, force=True)
    loaded = pm.gdataset_ls()

    # without force, the re-load is refused for the name it took from dataset2
    with pytest.raises(ValueError, match="tracks already exist: \\['shared'\\]"):
        pm.gdataset_load(ds1)
    assert pm.gdataset_ls() == loaded
    assert pm.gtrack_dataset(["ds_only", "shared"]) == [ds1, ds1]
    assert pm.gextract("shared", intervs)["shared"].tolist() == [10]

    # another session adds to dataset1 a track named as an interval set of the working database
    in_other_session(ds1, "pm.gtrack_create_sparse('foo', '', pm.gintervals('chr1', 0, 1000), [2])")
    with pytest.raises(ValueError, match="'foo' would be both a track and an interval set"):
        pm.gdataset_load(ds1, force=True)
    assert pm.gdataset_ls() == loaded
    assert pm.gtrack_dataset(["ds_only", "shared"]) == [ds1, ds1]
    assert pm.gextract("ds_only", intervs)["ds_only"].tolist() == [1]
    assert "foo" not in pm.gtrack_ls()


def test_copy_with_db_of_a_name_another_loaded_database_holds_maps_it_as_reload_does(tmp_path, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    work, ds1, ds2 = (create_test_db(tmp_path / n) for n in ("workdb", "ds1", "ds2"))
    pm.gdb_init(ds2)
    intervs = pm.gintervals("chr1", 0, 1000)
    pm.gtrack_create_sparse("in_ds2", "", intervs, [2])
    pm.gdb_init(work)
    pm.gtrack_create_sparse("both", "", intervs, [1])
    pm.gtrack_create_sparse("src", "", intervs, [9])
    pm.gdataset_load(ds1)
    pm.gdataset_load(ds2)

    def values_of(tracks):
        return [pm.gextract(t, intervs)[t].iloc[0] for t in tracks]

    # the working database's name stays its own
    pm.gtrack_copy("src", "both", db=ds1)
    # a name of ds2, loaded after ds1, stays ds2's; copied into ds2, a name of ds1 becomes ds2's
    pm.gtrack_copy("src", "in_ds2", db=ds1)
    pm.gtrack_copy("src", "new_t", db=ds1)
    pm.gtrack_copy("src", "new_t", db=ds2)
    tracks = ["both", "in_ds2", "new_t"]
    assert pm.gtrack_dataset(tracks) == [work, ds2, ds2]
    assert values_of(tracks) == [1, 2, 9]

    pm.gdb_reload()
    assert pm.gtrack_dataset(tracks) == [work, ds2, ds2]
    # without the working database's track, the name is ds1's copy
    pm.gtrack_rm("both", force=True)
    assert pm.gtrack_dataset("both") == ds1
    assert values_of(["both"]) == [9]


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs file permissions")
def test_a_failed_track_conversion_to_indexed_leaves_no_file_behind_to_flush_at_exit(tmp_path):
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("pctrack", "", pm.gintervals(["chr1", "chr2"], 0, 10), [1, 2])
    # the second chromosome cannot be read: the conversion stops after writing the first
    unreadable = Path(db) / "tracks" / "pctrack.track" / "chr2"
    assert unreadable.exists()
    unreadable.chmod(0)
    try:
        # in a session of its own, which ends after the failed conversion with files it wrote
        # still open: a FILE of the conversion left unclosed would write its buffer into one of
        # them at exit
        user_files = [tmp_path / f"user{i}.txt" for i in range(1, 9)]
        code = f"""
            try:
                pm.gtrack_convert_to_indexed("pctrack")
            except Exception:
                pass
            else:
                raise SystemExit("the conversion did not fail")
            files = [open(f, "w") for f in {[str(f) for f in user_files]!r}]
            for f in files:
                f.write("user line\\n")
                f.flush()
            import os
            os._exit(0)
        """
        in_other_session(db, code)
    finally:
        unreadable.chmod(0o644)
    assert [f.stat().st_size for f in user_files] == [len("user line\n")] * 8



# --- round 2: gintervals_2d order, 2D neighbors and max_data_size, unknown iterators ------------

def test_gintervals_2d_sorts_by_chromosome_pair_and_keeps_the_order_within_a_pair():
    # R 5.17.0's gintervals.2d of the same arguments
    r = pm.gintervals_2d(["chr2", "chr1", "chr2", "chrX", "chr1"], [500, 300, 100, 0, 10], [600, 400, 200, 50, 20],
                         ["chr1", "chr2", "chr1", "chr1", "chr2"], [0, 10, 20, 30, 40], [5, 15, 25, 35, 45])
    assert [tuple(x) for x in r.itertuples(index=False)] == [
        ("chr1", 300, 400, "chr2", 10, 15), ("chr1", 10, 20, "chr2", 40, 45), ("chr2", 500, 600, "chr1", 0, 5),
        ("chr2", 100, 200, "chr1", 20, 25), ("chrX", 0, 50, "chr1", 30, 35)]
    n = pm.gintervals_2d("chr1", [1102, 1104, 1103], [1200, 1200, 1150], "chr1", [1101, 1050, 1070], [1200, 1060, 1080])
    assert n["start1"].tolist() == [1102, 1104, 1103]


@pytest.mark.parametrize("bounded", [False, True])
def test_2d_neighbors_check_max_data_size_as_r(monkeypatch, bounded):
    kw = {"maxdist1": 1e6, "maxdist2": 1e6} if bounded else {}
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 10)
    q2 = pm.gintervals_2d("chr1", 0, 20000, "chr1", 0, 20000)
    s = np.arange(0, 5000, 100)
    t2 = DF({"chrom1": "chr1", "start1": s, "end1": s + 5, "chrom2": "chr1", "start2": 0, "end2": 5})
    assert len(pm.gintervals_neighbors(q2, t2, maxneighbors=10, **kw)) == 10
    for k in (11, 20):
        with pytest.raises(_pymisha.error, match=r"Result size \(at least 11 records\) exceeded the maximum allowed "
                                                 r"\(10 records"):
            pm.gintervals_neighbors(q2, t2, maxneighbors=k, **kw)
    # the rows of queries with no neighbor count too
    q3 = pm.gintervals_2d("chr1", [0, 30000, 60000], [20000, 50000, 80000], "chr1", 0, 20000)
    for k, na, rows in ((3, False, 3), (3, True, 5), (4, True, 6)):
        assert len(pm.gintervals_neighbors(q3, t2, maxneighbors=k, na_if_notfound=na, maxdist1=0, maxdist2=0)) == rows
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 5)
    with pytest.raises(_pymisha.error, match=r"at least 6 records"):
        pm.gintervals_neighbors(q3, t2, maxneighbors=4, na_if_notfound=True, maxdist1=0, maxdist2=0)


_ITERATOR_USERS = {
    "gextract": lambda iv, it: pm.gextract("dense_track", iv, iterator=it),
    "gextract_vtrack_expr": lambda iv, it: pm.gextract("dense_track * 2", iv, iterator=it),
    "gscreen": lambda iv, it: pm.gscreen("dense_track > 0", iv, iterator=it),
    "gsummary": lambda iv, it: pm.gsummary("dense_track", iv, iterator=it),
    "gquantiles": lambda iv, it: pm.gquantiles("dense_track", 0.5, iv, iterator=it),
    "gdist": lambda iv, it: pm.gdist("dense_track", [0, 1], intervals=iv, iterator=it),
    "glookup": lambda iv, it: pm.glookup([1, 2], "dense_track", [0, 0.5, 1], intervals=iv, iterator=it),
    "gpartition": lambda iv, it: pm.gpartition("dense_track", [0, 0.5, 1], iv, iterator=it),
    "giterator_intervals": lambda iv, it: pm.giterator_intervals("dense_track", iv, iterator=it),
    "gsample": lambda iv, it: pm.gsample("dense_track", 10, iv, iterator=it),
    "gintervals_summary": lambda iv, it: pm.gintervals_summary("dense_track", iv, iterator=it),
    "gintervals_quantiles": lambda iv, it: pm.gintervals_quantiles("dense_track", 0.5, iv, iterator=it),
    "gbins_summary": lambda iv, it: pm.gbins_summary("dense_track", [0, 0.5, 1], expr="dense_track", intervals=iv,
                                                     iterator=it),
    "gcor": lambda iv, it: pm.gcor("dense_track", "dense_track", intervals=iv, iterator=it),
    "gwilcox": lambda iv, it: pm.gwilcox("dense_track", 1000, 100, intervals=iv, iterator=it),
    "gsegment": lambda iv, it: pm.gsegment("dense_track", 1000, intervals=iv, iterator=it),
    "gintervals_mapply": lambda iv, it: pm.gintervals_mapply(np.max, "dense_track", intervals=iv, iterator=it),
    "gtrack_create": lambda iv, it: pm.gtrack_create("tmp_nosuch_iterator", "", "dense_track", iterator=it),
    "gtrack_lookup": lambda iv, it: pm.gtrack_lookup("tmp_nosuch_iterator", "", [1, 2], "dense_track", [0, 0.5, 1],
                                                     iterator=it),
    "gextract_2d": lambda iv, it: pm.gextract("rects_track", pm.gintervals_2d("chr1", 0, 1000, "chr1", 0, 1000),
                                              iterator=it),
    "gscreen_2d": lambda iv, it: pm.gscreen("rects_track > 0", pm.gintervals_2d("chr1", 0, 1000, "chr1", 0, 1000),
                                            iterator=it),
}


@pytest.mark.parametrize("multitasking", [False, True])
@pytest.mark.parametrize("func", list(_ITERATOR_USERS))
def test_an_iterator_that_names_nothing_is_an_error_as_in_r(func, multitasking, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", multitasking)
    iv = pm.gintervals("chr1", 0, 10000)
    with pytest.raises(ERRORS, match="^Invalid iterator: nosuch is neither a name of a track nor a name of an "
                                     "intervals set$"):
        _ITERATOR_USERS[func](iv, "nosuch")
    assert not pm.gtrack_exists("tmp_nosuch_iterator")


@pytest.mark.parametrize("func", ["gextract", "gextract_2d"])
def test_an_iterator_name_with_other_characters_is_an_invalid_interval_name(func):
    with pytest.raises(ERRORS, match=r"Invalid interval name abc def\. Only alphanumeric characters and _ are allowed"):
        _ITERATOR_USERS[func](pm.gintervals("chr1", 0, 10000), "abc def")
