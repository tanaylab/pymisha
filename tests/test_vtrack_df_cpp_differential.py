"""A virtual track on a DataFrame source, evaluated in the C++ scanner, against the Python path.

The scanner reads the source once and evaluates each iterator interval (PMIntervalsVtrack). The
Python path, kept for what the scanner does not take (user variables, 2D iterators),
iterates the intervals first and computes the vtrack over all of them with pm_vtrack_compute; it
is forced here by making _can_vtracks_use_cpp refuse every vtrack. Both share the evaluator, so
the interval functions and the counting value functions are also checked against a direct
computation. Sources and scopes are random: several chromosomes, rows in no order, overlapping
intervals (where the function allows them), NaN values, empty sources, shifts that move a window
off its chromosome.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import pymisha as pm
from pymisha import extract as _extract

CHROMS = {"chr1": 500_000, "chr2": 300_000, "chrX": 200_000}

VALUE_FUNCS = [
    "avg", "mean", "sum", "weighted.sum", "min", "max", "first", "last", "size", "exists",
    "stddev", "std", "nearest", "lse",
    "first.pos.abs", "first.pos.relative", "last.pos.abs", "last.pos.relative",
    "min.pos.abs", "min.pos.relative", "max.pos.abs", "max.pos.relative",
]


@pytest.fixture(autouse=True)
def _clean(_init_db):
    saved = dict(pm.CONFIG)
    pm.CONFIG["progress"] = False
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()
    pm.CONFIG.clear()
    pm.CONFIG.update(saved)


@pytest.fixture
def python_path(monkeypatch):
    """Call fn with every vtrack sent to the Python path."""
    def run(fn, *args, **kwargs):
        with monkeypatch.context() as m:
            m.setattr(_extract, "_can_vtracks_use_cpp", lambda names: False)
            return fn(*args, **kwargs)
    return run


def _intervals(rng, n, min_len, max_len, chroms=CHROMS, overlap=True):
    """n random intervals over chroms, rows in random order; without overlap, none touch."""
    names = list(chroms)
    rows = []
    for _ in range(n):
        c = names[rng.integers(len(names))]
        ln = int(rng.integers(min_len, max_len + 1))
        s = int(rng.integers(0, chroms[c] - ln))
        rows.append((c, s, s + ln))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    if not overlap and len(df):
        df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
        keep = np.ones(len(df), dtype=bool)
        last_c, last_e = None, -1
        for i, (c, s, e) in enumerate(zip(df.chrom, df.start, df.end, strict=True)):
            if c == last_c and s <= last_e:
                keep[i] = False
            else:
                last_c, last_e = c, e
        df = df[keep].sample(frac=1, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    return df


def _scope(rng, n=12):
    # windows of a few kb, some on top of each other, in random row order
    return _intervals(rng, n, 200, 4000)


def _assert_same(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if col == "chrom":
            assert a[col].astype(str).tolist() == b[col].astype(str).tolist()
        else:
            npt.assert_array_equal(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), err_msg=col)


def _shifts(rng):
    # mostly small; sometimes far enough to move a window off its chromosome or invert it
    if rng.random() < 0.3:
        return 0, 0
    big = rng.random() < 0.2
    lim = 400_000 if big else 3000
    return int(rng.integers(-lim, lim)), int(rng.integers(-lim, lim))


@pytest.mark.parametrize("seed", range(6))
def test_value_funcs_cpp_matches_python(seed, python_path):
    rng = np.random.default_rng(seed)
    src = _intervals(rng, 400, 1, 300, overlap=False)
    src["score"] = rng.normal(size=len(src)).round(3)
    src.loc[rng.random(len(src)) < 0.1, "score"] = np.nan
    scope = _scope(rng)
    iterator = int(rng.choice([1, 7, 50, 333]))
    names = []
    for i, func in enumerate(VALUE_FUNCS):
        sshift, eshift = _shifts(rng)
        pm.gvtrack_create(f"v{i}", src, func=func, sshift=sshift, eshift=eshift)
        names.append(f"v{i}")
    pm.gvtrack_create("vq", src, func="quantile", params=0.3)
    names.append("vq")
    exprs = names + ["v0 + v2", "np.where(v8 > 0, v3, -1)"]

    cpp = pm.gextract(exprs, scope, iterator=iterator)
    py = python_path(pm.gextract, exprs, scope, iterator=iterator)
    _assert_same(cpp, py)


@pytest.mark.parametrize("func", ["sample", "sample.pos.abs", "sample.pos.relative"])
def test_sample_cpp_matches_python(func, python_path):
    # one vtrack draws the same values on both paths (with several, the scanner interleaves
    # their draws row by row, as R does, and the Python path draws one vtrack after another)
    rng = np.random.default_rng(11)
    src = _intervals(rng, 400, 1, 300, overlap=False)
    src["score"] = rng.normal(size=len(src)).round(3)
    pm.gvtrack_create("s", src, func=func, sshift=-100, eshift=100)
    scope = _scope(rng)
    pm.CONFIG["multitasking"] = False  # (a kid draws from its own stream)
    np.random.seed(5)
    cpp = pm.gextract("s", scope, iterator=50)
    np.random.seed(5)
    _assert_same(cpp, python_path(pm.gextract, "s", scope, iterator=50))


@pytest.mark.parametrize("seed", range(6))
def test_interval_funcs_cpp_matches_python(seed, python_path):
    rng = np.random.default_rng(100 + seed)
    src = _intervals(rng, 300, 1, 2000)  # overlapping, unsorted
    strand_src = src.assign(strand=rng.choice([-1, 0, 1], size=len(src)))
    disjoint = _intervals(rng, 200, 5, 500, overlap=False)
    scope = _scope(rng)
    iterator = int(rng.choice([1, 13, 100, 777]))
    specs = [
        ("coverage", src, None), ("neighbor.count", src, 0), ("neighbor.count", src, 250),
        ("distance", src, None), ("distance", disjoint, 40), ("distance.center", src, None),
        ("distance.edge", strand_src, None), ("distance", strand_src, None),
        ("distance.center", strand_src, None),
    ]
    names = []
    for i, (func, s, params) in enumerate(specs):
        sshift, eshift = _shifts(rng)
        pm.gvtrack_create(f"i{i}", s, func=func, params=params, sshift=sshift, eshift=eshift)
        names.append(f"i{i}")

    cpp = pm.gextract(names, scope, iterator=iterator)
    py = python_path(pm.gextract, names, scope, iterator=iterator)
    _assert_same(cpp, py)


def _direct(func, src, rows, param=0):
    """func over each row of rows, computed directly from src (no shifts)."""
    by_chrom = {c: (g.start.to_numpy(), g.end.to_numpy(),
                    g["score"].to_numpy(dtype=np.float32).astype(float) if "score" in g else None)
                for c, g in src.assign(chrom=src.chrom.astype(str)).groupby("chrom")}
    empty = (np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0))
    out = np.full(len(rows), np.nan)
    for k, (c, s, e) in enumerate(zip(rows.chrom.astype(str), rows.start, rows.end, strict=True)):
        st, en, sc = by_chrom.get(c, empty)
        if func == "neighbor.count":
            out[k] = np.sum((np.maximum(st - param, 0) < e) & (np.minimum(en + param, CHROMS[c]) > s))
            continue
        hit = (st < e) & (en > s)
        if func == "coverage":
            covered = np.zeros(e - s, dtype=bool)
            for a, b in zip(st[hit], en[hit], strict=True):
                covered[max(a, s) - s:min(b, e) - s] = True
            out[k] = covered.mean()
            continue
        vals = sc[hit]
        ok = ~np.isnan(vals)
        if func == "size":
            out[k] = ok.sum()
        elif func == "exists":
            out[k] = float(ok.any())
        elif ok.any():
            if func == "avg":
                out[k] = vals[ok].mean()
            elif func == "sum":
                out[k] = vals[ok].sum()
            elif func == "min":
                out[k] = vals[ok].min()
            elif func == "max":
                out[k] = vals[ok].max()
            elif func == "weighted.sum":
                bp = np.minimum(en[hit], e) - np.maximum(st[hit], s)
                out[k] = (vals[ok] * bp[ok]).sum()
    return out


@pytest.mark.parametrize("seed", range(4))
def test_cpp_matches_direct_computation(seed):
    rng = np.random.default_rng(200 + seed)
    ivs = _intervals(rng, 300, 1, 3000)
    # long intervals with short ones inside them, the case a scan by start alone gets wrong
    ivs = pd.concat([ivs, pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [1000, 1500, 0],
                                        "end": [400_000, 1600, 290_000]})], ignore_index=True)
    vals = _intervals(rng, 300, 1, 300, overlap=False)
    vals["score"] = rng.normal(size=len(vals)).round(3)
    vals.loc[rng.random(len(vals)) < 0.1, "score"] = np.nan
    scope = _scope(rng)
    iterator = int(rng.choice([3, 9, 120]))

    checks = [("coverage", ivs, None), ("neighbor.count", ivs, 0), ("neighbor.count", ivs, 300)]
    checks += [(f, vals, None) for f in ("avg", "sum", "min", "max", "size", "exists", "weighted.sum")]
    for i, (func, s, params) in enumerate(checks):
        pm.gvtrack_create(f"d{i}", s, func=func, params=params)
    res = pm.gextract([f"d{i}" for i in range(len(checks))], scope, iterator=iterator)
    for i, (func, s, params) in enumerate(checks):
        expected = _direct(func, s, res, params or 0)
        npt.assert_allclose(res[f"d{i}"].to_numpy(), expected, rtol=1e-5, atol=1e-6, err_msg=func)


def test_neighbor_count_after_scope_jumps_back():
    # a long source interval, then a short one that ends before the query: every window that
    # overlaps the long one counts it, also after an overlapping scope moves the scan back
    src = pd.DataFrame({"chrom": "1", "start": [0, 10, 1100], "end": [1000, 20, 1200]})
    pm.gvtrack_create("nc", src, func="neighbor.count", params=0)
    scope = pd.DataFrame({"chrom": "1", "start": [400, 500], "end": [1160, 520]})
    res = pm.gextract("nc", scope, iterator=10)
    assert (res.loc[res.start.isin([500, 510]), "nc"] == 1).all()
    assert res.loc[res.start == 1150, "nc"].tolist() == [1]


@pytest.mark.parametrize("func", ["avg", "size", "exists", "sum", "coverage", "distance", "neighbor.count"])
def test_empty_source(func, python_path):
    # as R: no value is NaN, a count is 0
    src = pd.DataFrame({"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int),
                        "end": pd.Series([], dtype=int), "score": pd.Series([], dtype=float)})
    pm.gvtrack_create("e", src, func=func)
    scope = pd.DataFrame({"chrom": ["1", "2"], "start": [0, 100], "end": [300, 250]})
    cpp = pm.gextract("e", scope, iterator=100)
    _assert_same(cpp, python_path(pm.gextract, "e", scope, iterator=100))
    expected = 0.0 if func in ("size", "exists", "coverage", "neighbor.count") else np.nan
    npt.assert_array_equal(cpp["e"].to_numpy(), np.full(len(cpp), expected))


def test_dataframe_iterator_and_screen(python_path):
    rng = np.random.default_rng(7)
    src = _intervals(rng, 200, 1, 400, overlap=False)
    src["score"] = rng.normal(size=len(src))
    pm.gvtrack_create("a", src, func="max", sshift=-50, eshift=50)
    pm.gvtrack_create("c", src[["chrom", "start", "end"]], func="coverage")
    scope = _scope(rng)
    itr = _intervals(rng, 30, 50, 3000)
    _assert_same(pm.gextract(["a", "c"], scope, iterator=itr),
                 python_path(pm.gextract, ["a", "c"], scope, iterator=itr))
    # (the Python path does not sort what it finds; the scanner, as R, does)
    _assert_same(pm.gscreen("c > 0.2 | a > 1", scope, iterator=25),
                 python_path(pm.gscreen, "c > 0.2 | a > 1", scope, iterator=25).sort_values(["chrom", "start"]))


@pytest.mark.parametrize("seed", range(3))
def test_reductions_cpp_matches_python(seed, python_path):
    rng = np.random.default_rng(300 + seed)
    src = _intervals(rng, 300, 1, 300, overlap=False)
    src["score"] = rng.normal(size=len(src)).round(3)
    pm.gvtrack_create("v", src, func="avg", sshift=-20, eshift=20)
    pm.gvtrack_create("c", src[["chrom", "start", "end"]], func="coverage")
    scope = _scope(rng)

    s_cpp = pm.gsummary("v * 2 + c", scope, iterator=10)
    s_py = python_path(pm.gsummary, "v * 2 + c", scope, iterator=10)
    npt.assert_allclose(s_cpp.to_numpy(), s_py.to_numpy(), rtol=1e-9)

    pct = [0, 0.1, 0.5, 0.97, 1]
    npt.assert_allclose(pm.gquantiles("c", pct, scope, iterator=10).to_numpy(),
                        python_path(pm.gquantiles, "c", pct, scope, iterator=10).to_numpy(), rtol=1e-12)

    breaks = [-3, -1, 0, 0.5, 1, 3]
    npt.assert_array_equal(pm.gdist("v", breaks, "c", [0, 0.5, 1], intervals=scope, iterator=10),
                           python_path(pm.gdist, "v", breaks, "c", [0, 0.5, 1], intervals=scope, iterator=10))


def test_no_iterator(python_path):
    # gextract and gscreen refuse to guess, as R; the reductions take the scope intervals as they are
    src = pd.DataFrame({"chrom": "1", "start": [100, 900], "end": [200, 1500]})
    pm.gvtrack_create("c", src, func="coverage")
    scope = pm.gintervals("1", [0, 800], [500, 2000])
    with pytest.raises(ValueError, match='track expression "c" does not contain any tracks'):
        pm.gextract("c", scope)
    with pytest.raises(ValueError, match="track expressions do not contain any tracks"):
        pm.gextract(["c", "c * 2"], scope)
    with pytest.raises(ValueError, match='track expression "c > 0" does not contain any tracks'):
        pm.gscreen("c > 0", scope)
    npt.assert_array_equal(pm.gsummary("c", scope).to_numpy(), python_path(pm.gsummary, "c", scope).to_numpy())


def test_filtered_vtrack_takes_the_scanner(monkeypatch):
    src = pd.DataFrame({"chrom": "1", "start": [100, 400], "end": [300, 600], "score": [1.0, 3.0]})
    pm.gvtrack_create("f", src, func="sum")
    pm.gvtrack_filter("f", pd.DataFrame({"chrom": ["1"], "start": [350], "end": [700]}))
    pm.gvtrack_create("g", src, func="sum")
    assert _extract._can_vtracks_use_cpp(["f", "g"])
    res = pm.gextract(["f", "g"], pm.gintervals("1", 0, 1000), iterator=500)
    npt.assert_array_equal(res["f"].to_numpy(), [1.0, np.nan])
    npt.assert_array_equal(res["g"].to_numpy(), [4.0, 3.0])


def test_gtrack_create_takes_a_dataframe_vtrack(tmp_path):
    # gtrack_create reads the expression with the scanner, which now takes a DataFrame source
    import shutil

    from _dbpath import TESTDB_ROOT

    root = tmp_path / "trackdb" / "test"
    shutil.copytree(TESTDB_ROOT, root)
    try:
        pm.gdb_init(str(root))
        pm.CONFIG["progress"] = False
        rng = np.random.default_rng(5)
        src = _intervals(rng, 200, 1, 400, overlap=False)
        src["score"] = rng.normal(size=len(src))
        pm.gvtrack_create("v", src, func="sum", sshift=-30, eshift=30)
        pm.gvtrack_create("c", src[["chrom", "start", "end"]], func="coverage")
        pm.gtrack_create("from_df_vtrack", "", "v + c", iterator=100)
        scope = pm.gintervals_all()
        got = pm.gextract("from_df_vtrack", scope, iterator=100)
        ref = pm.gextract("v + c", scope, iterator=100)
        npt.assert_allclose(got["from_df_vtrack"].to_numpy(), ref["v + c"].to_numpy(), rtol=1e-6)
    finally:
        pm.gvtrack_clear()
        pm.gdb_init(str(TESTDB_ROOT))
