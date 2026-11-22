"""A filtered virtual track (gvtrack_filter), evaluated in the C++ scanner, against the Python path.

The scanner subtracts the mask from each iterator interval (after the shifts) and computes the
function over the parts left (Genome1DFilter in PMTrackExpressionVars). The Python path, kept for
expressions with Python variables and for 2D iterators, builds the parts of all the intervals first
and computes the function over them in vtracks.py; it is forced here by making
_can_vtracks_use_cpp refuse every vtrack. Masks, sources and scopes are random: masks of short
gaps that split bins and track intervals, long blocks, intervals at the chromosome ends, masks
covering everything or nothing of the scope, and shifts that move a window off its chromosome.

The values are the same to the bit, except where the two paths add in another order (see
_TOLERANCE).
"""

import _pymisha
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import pymisha as pm
from pymisha import extract as _extract
from pymisha import variants as _variants

CHROMS = {"chr1": 500_000, "chr2": 300_000, "chrX": 200_000}

TRACK_FUNCS = [
    "avg", "mean", "sum", "weighted.sum", "min", "max", "first", "last", "size", "exists",
    "stddev", "std", "nearest", "lse",
    "first.pos.abs", "first.pos.relative", "last.pos.abs", "last.pos.relative",
    "min.pos.abs", "min.pos.relative", "max.pos.abs", "max.pos.relative",
]
SAMPLE_FUNCS = ["sample", "sample.pos.abs", "sample.pos.relative"]

# Where the paths round differently: lse of a track (the Python path takes the log-sum-exp of the
# values in double, the scanner merges the parts' log-sum-exps, each a float, as R misha)
_TOLERANCE = {"lse": 1e-6}


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
            m.setattr(_variants, "_can_vtracks_use_cpp", lambda names: False)
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


MASK_KINDS = ["gaps", "blocks", "edges", "mixed", "all", "outside", "empty"]


def _mask(rng, kind):
    """A random mask of the given kind (overlapping and touching intervals, in random order)."""
    if kind == "gaps":  # short, many: split dense bins and track intervals
        return _intervals(rng, 3000, 1, 120)
    if kind == "blocks":
        return _intervals(rng, 40, 1000, 30_000)
    if kind == "edges":  # touching the ends of each chromosome, and short gaps near them
        rows = []
        for c, size in CHROMS.items():
            a, b = int(rng.integers(1, 3000)), int(rng.integers(1, 3000))
            rows += [(c, 0, a), (c, size - b, size)]
            for _ in range(20):
                s = int(rng.integers(0, 6000))
                rows += [(c, s, s + int(rng.integers(1, 60))), (c, size - s - 70, size - s - 10)]
        return pd.DataFrame(rows, columns=["chrom", "start", "end"])
    if kind == "mixed":
        return pd.concat([_mask(rng, "gaps"), _mask(rng, "blocks"), _mask(rng, "edges")], ignore_index=True)
    if kind == "all":
        return pd.DataFrame({"chrom": list(CHROMS), "start": 0, "end": list(CHROMS.values())})
    if kind == "empty":  # no filter
        return pd.DataFrame({"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=int),
                             "end": pd.Series([], dtype=int)})
    # nothing of the scope (_scope keeps off chr2's last 50 kb)
    return pd.DataFrame({"chrom": ["chr2"], "start": [CHROMS["chr2"] - 50_000], "end": [CHROMS["chr2"]]})


def _scope(rng, n=12):
    # windows of a few kb, some on top of each other, in random row order; one at each chromosome
    # start, off chr2's last 50 kb
    chroms = dict(CHROMS, chr2=CHROMS["chr2"] - 50_000)
    edges = pd.DataFrame({"chrom": ["chr1", "chrX"], "start": [0, CHROMS["chrX"] - 3000],
                          "end": [3000, CHROMS["chrX"]]})
    return pd.concat([_intervals(rng, n, 200, 4000, chroms=chroms), edges], ignore_index=True)


def _shifts(rng):
    # mostly small; sometimes far enough to move a window off its chromosome or invert it
    if rng.random() < 0.3:
        return 0, 0
    big = rng.random() < 0.2
    lim = 400_000 if big else 3000
    return int(rng.integers(-lim, lim)), int(rng.integers(-lim, lim))


def _create(rng, name, src, func, mask, **kwargs):
    sshift, eshift = _shifts(rng)
    pm.gvtrack_create(name, src, func=func, sshift=sshift, eshift=eshift, **kwargs)
    pm.gvtrack_filter(name, filter=mask)


def _assert_same(cpp, py, funcs):
    """cpp and py (gextract results) hold the same rows and values; funcs: the function of each
    vtrack column."""
    assert (cpp is None) == (py is None)
    if cpp is None:
        return
    cpp = cpp.reset_index(drop=True)
    py = py.reset_index(drop=True)
    assert list(cpp.columns) == list(py.columns)
    assert cpp["chrom"].astype(str).tolist() == py["chrom"].astype(str).tolist()
    for col in ("start", "end", "intervalID"):
        npt.assert_array_equal(cpp[col].to_numpy(), py[col].to_numpy(), err_msg=col)
    for col, func in funcs.items():
        a, b = cpp[col].to_numpy(dtype=float), py[col].to_numpy(dtype=float)
        if func in _TOLERANCE:
            npt.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=f"{col} ({func})")
            npt.assert_allclose(a, b, rtol=_TOLERANCE[func], atol=0, err_msg=f"{col} ({func})")
        else:
            npt.assert_array_equal(a, b, err_msg=f"{col} ({func})")


def _track_sources():
    return ["dense_track", "sparse_track", "array_track", "array_slice"]


def _create_track_vtrack(rng, name, src, func, mask, **kwargs):
    if src == "array_slice":
        _create(rng, name, "array_track", func, mask, **kwargs)
        pm.gvtrack_array_slice(name, slice=[1, 4, 7], func=str(rng.choice(["max", "sum", "avg"])))
    else:
        _create(rng, name, src, func, mask, **kwargs)


@pytest.mark.parametrize("kind", MASK_KINDS)
@pytest.mark.parametrize("src", _track_sources())
def test_track_funcs(src, kind, python_path):
    rng = np.random.default_rng(10 * _track_sources().index(src) + MASK_KINDS.index(kind))
    mask = _mask(rng, kind)
    funcs = {}
    for i, func in enumerate(TRACK_FUNCS):
        _create_track_vtrack(rng, f"v{i}", src, func, mask)
        funcs[f"v{i}"] = func
    for q in (0, 0.3, 1):
        _create_track_vtrack(rng, f"q{q}", src, "quantile", mask, params=q)
        funcs[f"q{q}"] = "quantile"
    scope = _scope(rng)
    for iterator in (int(rng.choice([1, 7, 50, 333])), int(rng.choice([2000, 5000]))):
        cpp = pm.gextract(list(funcs), scope, iterator=iterator)
        py = python_path(pm.gextract, list(funcs), scope, iterator=iterator)
        _assert_same(cpp, py, funcs)


@pytest.mark.parametrize("seed", range(4))
def test_value_and_interval_funcs(seed, python_path):
    """A DataFrame source: the value functions of a value column, and the interval functions."""
    rng = np.random.default_rng(seed)
    vals = _intervals(rng, 400, 1, 300, overlap=False)
    vals["score"] = rng.normal(size=len(vals)).round(3)
    vals.loc[rng.random(len(vals)) < 0.1, "score"] = np.nan
    ivs = _intervals(rng, 300, 1, 2000)  # overlapping, unsorted
    mask = _mask(rng, MASK_KINDS[seed % 4])
    funcs = {}
    for i, func in enumerate(TRACK_FUNCS):
        _create(rng, f"v{i}", vals, func, mask)
        funcs[f"v{i}"] = func
    _create(rng, "vq", vals, "quantile", mask, params=0.3)
    funcs["vq"] = "quantile"
    for i, (func, params) in enumerate([("coverage", None), ("neighbor.count", 0), ("neighbor.count", 250),
                                        ("distance", None), ("distance.center", None), ("distance.edge", None)]):
        _create(rng, f"i{i}", ivs, func, mask, params=params)
        funcs[f"i{i}"] = func
    scope = _scope(rng)
    iterator = int(rng.choice([1, 13, 100, 777]))
    cpp = pm.gextract(list(funcs), scope, iterator=iterator)
    py = python_path(pm.gextract, list(funcs), scope, iterator=iterator)
    _assert_same(cpp, py, funcs)


@pytest.mark.parametrize("dist, expected", [(0, 3), (60, 4)])
def test_neighbor_count_counts_a_source_interval_once(dist, expected, python_path):
    # The mask leaves [0,400), [600,2000) and [2500,4000). [100,1000) is near two parts and
    # [300,3000) near three; each counts once, as R misha. [450,550) lies in the mask: it counts
    # only when the distance brings it to a part.
    src = pd.DataFrame({"chrom": "chr1", "start": [100, 450, 300, 3000], "end": [1000, 550, 3000, 3100]})
    pm.gvtrack_create("nc", src, func="neighbor.count", params=dist)
    pm.gvtrack_filter("nc", filter=pd.DataFrame({"chrom": "chr1", "start": [400, 2000], "end": [600, 2500]}))
    scope = pm.gintervals("chr1", 0, 4000)
    assert pm.gextract("nc", scope, iterator=scope)["nc"].tolist() == [expected]
    assert python_path(pm.gextract, "nc", scope, iterator=scope)["nc"].tolist() == [expected]

    pm.gvtrack_create("nc0", src.iloc[:0], func="neighbor.count", params=dist)
    pm.gvtrack_filter("nc0", filter=pd.DataFrame({"chrom": "chr1", "start": [400], "end": [600]}))
    assert pm.gextract("nc0", scope, iterator=scope)["nc0"].tolist() == [0]
    assert python_path(pm.gextract, "nc0", scope, iterator=scope)["nc0"].tolist() == [0]


_PSSM = np.array([
    [0.90, 0.04, 0.03, 0.03],
    [0.03, 0.85, 0.06, 0.06],
    [0.05, 0.05, 0.80, 0.10],
    [0.08, 0.08, 0.09, 0.75],
    [0.70, 0.10, 0.10, 0.10],
    [0.12, 0.65, 0.12, 0.11],
])


def _seq_specs():
    from _potts_oracle import potts_ref_model

    potts = potts_ref_model(5, npair_mode="sparse", seed=89)
    pwm = {"pssm": _PSSM, "prior": 0.01}
    ed = {"pssm": _PSSM, "score_thresh": -9.0}
    return [
        ("pwm", pwm), ("pwm.max", pwm), ("pwm.count", dict(pwm, score_thresh=-10.0)),
        ("pwm.max.pos", pwm), ("pwm.max.pos", dict(pwm, bidirect=False, strand=-1)),
        ("pwm.max.pos", dict(pwm, spat_factor=[1.0, 2.0, 0.5], spat_bin=40)),
        ("potts", {"params": potts}), ("potts.max", {"params": potts}),
        ("potts.max.pos", {"params": potts}), ("potts.count", {"params": dict(potts, score_thresh=-1.0)}),
        ("kmer.count", {"kmer": "CG"}), ("kmer.frac", {"kmer": "ACG"}),
        ("kmer.count", {"kmer": "TAG", "extend": False}), ("kmer.frac", {"kmer": "GA", "strand": 1}),
        ("kmer.frac", {"kmer": "ACGT", "extend": False}),
        ("masked.count", {}), ("masked.frac", {}),
        ("pwm.edit_distance", dict(ed, max_edits=2)), ("pwm.edit_distance.pos", dict(ed, max_edits=2)),
        ("pwm.max.edit_distance", dict(ed, max_edits=2)), ("pwm.n_mutations", ed),
        ("pwm.edit_distance", dict(ed, max_edits=1, direction="below", score_thresh=-6.0)),
        ("pwm.edit_distance.lse", dict(ed, max_edits=2)), ("pwm.edit_distance.lse.pos", dict(ed, max_edits=2)),
    ]


@pytest.mark.parametrize("kind", MASK_KINDS)
def test_sequence_funcs(kind, python_path):
    rng = np.random.default_rng(MASK_KINDS.index(kind) + 50)
    mask = _mask(rng, kind)
    funcs = {}
    for i, (func, kwargs) in enumerate(_seq_specs()):
        _create(rng, f"s{i}", None, func, mask, **kwargs)
        funcs[f"s{i}"] = func
    scope = _scope(rng, n=6)
    for iterator in (int(rng.choice([20, 97])), 1500):
        cpp = pm.gextract(list(funcs), scope, iterator=iterator)
        py = python_path(pm.gextract, list(funcs), scope, iterator=iterator)
        _assert_same(cpp, py, funcs)


@pytest.mark.parametrize("src", ["dense_track", "sparse_track", "array_track", "values"])
def test_sample_funcs_draw_unmasked_values(src):
    # the draws differ between the paths; each is one of the unmasked values (or its position)
    rng = np.random.default_rng(7)
    if src == "values":
        src = _intervals(rng, 400, 1, 300, overlap=False)
        src["score"] = rng.normal(size=len(src)).round(3)
    mask = _mask(rng, "mixed")
    for func in SAMPLE_FUNCS + ["min", "max", "first.pos.abs", "last.pos.abs", "first.pos.relative",
                                "last.pos.relative"]:
        pm.gvtrack_create(func, src, func=func, sshift=-60, eshift=40)
        pm.gvtrack_filter(func, filter=mask)
    scope = _scope(rng)
    res = pm.gextract(SAMPLE_FUNCS + ["min", "max", "first.pos.abs", "last.pos.abs", "first.pos.relative",
                                      "last.pos.relative"], scope, iterator=300)
    for lo, hi, col in [("min", "max", "sample"), ("first.pos.abs", "last.pos.abs", "sample.pos.abs"),
                        ("first.pos.relative", "last.pos.relative", "sample.pos.relative")]:
        v = res[col].to_numpy()
        npt.assert_array_equal(np.isnan(v), np.isnan(res[lo].to_numpy()), err_msg=col)
        ok = ~np.isnan(v)
        assert (v[ok] >= res[lo].to_numpy()[ok]).all() and (v[ok] <= res[hi].to_numpy()[ok]).all(), col


def test_reductions_and_screen(python_path):
    """gsummary, gquantiles, gdist and gscreen take a filtered vtrack through the scanner."""
    rng = np.random.default_rng(3)
    mask = _mask(rng, "mixed")
    pm.gvtrack_create("d", "dense_track", func="avg", sshift=-100, eshift=100)
    pm.gvtrack_filter("d", filter=mask)
    pm.gvtrack_create("s", "sparse_track", func="max")
    pm.gvtrack_filter("s", filter=mask)
    pm.gvtrack_create("k", None, func="kmer.frac", kmer="CG")
    pm.gvtrack_filter("k", filter=mask)
    scope = _scope(rng)

    # no iterator (the bins of the dense source), and intervals as the iterator
    funcs = {"d": "avg", "s": "max", "k": "kmer.frac"}
    _assert_same(pm.gextract("d", scope), python_path(pm.gextract, "d", scope), {"d": "avg"})
    itr = _intervals(rng, 40, 10, 3000)
    _assert_same(pm.gextract(list(funcs), scope, iterator=itr),
                 python_path(pm.gextract, list(funcs), scope, iterator=itr), funcs)

    expr = "d * 2 + np.nan_to_num(s) + k"
    npt.assert_allclose(pm.gsummary(expr, scope, iterator=50).to_numpy(),
                        python_path(pm.gsummary, expr, scope, iterator=50).to_numpy(), rtol=1e-9)
    pct = [0, 0.1, 0.5, 0.97, 1]
    npt.assert_array_equal(pm.gquantiles("d", pct, scope, iterator=50).to_numpy(),
                           python_path(pm.gquantiles, "d", pct, scope, iterator=50).to_numpy())
    breaks = [-1, 0, 0.1, 0.2, 0.5, 1, 100]
    npt.assert_array_equal(pm.gdist("d", breaks, "k", [0, 0.01, 0.05, 1], intervals=scope, iterator=50),
                           python_path(pm.gdist, "d", breaks, "k", [0, 0.01, 0.05, 1], intervals=scope,
                                       iterator=50))
    cpp = pm.gscreen("d > 0.1 & k > 0", scope, iterator=50)
    py = python_path(pm.gscreen, "d > 0.1 & k > 0", scope, iterator=50).sort_values(["chrom", "start"])
    pd.testing.assert_frame_equal(cpp.reset_index(drop=True), py.reset_index(drop=True))


def test_routing():
    # a filter no longer sends a vtrack to the Python path; Python variables still do
    src = pd.DataFrame({"chrom": "1", "start": [100, 400], "end": [300, 600], "score": [1.0, 3.0]})
    pm.gvtrack_create("f", src, func="sum")
    pm.gvtrack_filter("f", pd.DataFrame({"chrom": ["1"], "start": [350], "end": [700]}))
    pm.gvtrack_create("t", "dense_track", func="avg")
    pm.gvtrack_filter("t", pd.DataFrame({"chrom": ["1"], "start": [350], "end": [700]}))
    assert _extract._can_vtracks_use_cpp(["f", "t"])
    res = pm.gextract(["f", "t"], pm.gintervals("1", 0, 1000), iterator=500)
    npt.assert_array_equal(res["f"].to_numpy(), [1.0, np.nan])
    offset = 0.5  # noqa: F841 - read by the expression below
    res2 = pm.gextract("f + offset", pm.gintervals("1", 0, 1000), iterator=500)
    npt.assert_array_equal(res2["f + offset"].to_numpy(), [1.5, np.nan])


@pytest.fixture
def db_copy(tmp_path):
    """A copy of the test DB, for tests that create tracks."""
    import shutil

    from _dbpath import TESTDB_ROOT

    root = tmp_path / "trackdb" / "test"
    shutil.copytree(TESTDB_ROOT, root)
    pm.gdb_init(str(root))
    pm.CONFIG["progress"] = False
    yield
    pm.gvtrack_clear()
    pm.gdb_init(str(TESTDB_ROOT))


def test_gtrack_create_takes_a_filtered_vtrack(db_copy):
    rng = np.random.default_rng(5)
    pm.gvtrack_create("v", "dense_track", func="sum", sshift=-30, eshift=30)
    pm.gvtrack_filter("v", filter=_mask(rng, "mixed"))
    pm.gtrack_create("from_filtered_vtrack", "", "v", iterator=100)
    scope = pm.gintervals_all()
    got = pm.gextract("from_filtered_vtrack", scope, iterator=100)
    ref = pm.gextract("v", scope, iterator=100)
    # the track holds floats
    npt.assert_array_equal(got["from_filtered_vtrack"].to_numpy(),
                           ref["v"].to_numpy().astype(np.float32).astype(float))


def test_source_without_a_chromosome(db_copy, python_path):
    # a sparse track with no file for chr2 and chrX: no value there, a count of 0
    import os

    rng = np.random.default_rng(6)
    src = _intervals(rng, 300, 10, 200, chroms={"chr1": CHROMS["chr1"]}, overlap=False)
    pm.gtrack_create_sparse("chr1_only", "", src, rng.normal(size=len(src)))
    track_dir = _pymisha.pm_track_path("chr1_only")
    for chrom in ("chr2", "chrX"):
        os.remove(os.path.join(track_dir, chrom))
    _pymisha.pm_dbreload()
    mask = _mask(rng, "mixed")
    funcs = {f"v{i}": func for i, func in enumerate(TRACK_FUNCS)}
    for name, func in funcs.items():
        pm.gvtrack_create(name, "chr1_only", func=func)
        pm.gvtrack_filter(name, filter=mask)
    scope = _scope(rng)
    cpp = pm.gextract(list(funcs), scope, iterator=500)
    _assert_same(cpp, python_path(pm.gextract, list(funcs), scope, iterator=500), funcs)
    other = cpp["chrom"].astype(str) != "chr1"
    assert other.any()
    for name, func in funcs.items():
        if func in ("size", "exists"):
            assert set(cpp.loc[other, name].dropna()) == {0.0}, func


def test_gextract_variants(python_path):
    # a filtered sequence vtrack under substitutions: the scanner against the Python path
    rng = np.random.default_rng(9)
    pm.gvtrack_create("p", None, func="pwm.max", pssm=_PSSM, prior=0.01)
    pm.gvtrack_filter("p", filter=_mask(rng, "gaps"))
    pm.gvtrack_create("k", None, func="kmer.count", kmer="GC")
    pm.gvtrack_filter("k", filter=_mask(rng, "gaps"))
    scope = pm.gintervals(["1", "2"], [1000, 5000], [1300, 5200])
    cpp = pm.gextract_variants("p + k", scope, iterator=100)
    py = python_path(pm.gextract_variants, "p + k", scope, iterator=100)
    pd.testing.assert_frame_equal(cpp.reset_index(drop=True), py.reset_index(drop=True))
