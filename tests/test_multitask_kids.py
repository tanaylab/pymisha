"""Multitasking as R misha does it: the scope is split into a run of intervals per kid (whole
chromosomes, or ranges cut at bin boundaries with an explicit bin iterator), each kid hands its
result over in shared memory, and the parent puts the kids' results together in kid order. The
rows of gextract and the like are then the rows of one process, in the same order.

R misha's plan gives the 1 Mb, 3-chromosome test database at most 2 kids by range (500 kb each)
and 3 by chromosome, so these tests build a database of 12 chromosomes (2.9 Mb) of their own, in
both formats. Each multitasked call is checked to have forked: its kids' page faults count in
the parent's RUSAGE_CHILDREN once they are reaped.
"""
import resource
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm

SIZES = [350_000, 320_000, 300_000, 280_000, 260_000, 240_000, 230_000, 220_000, 200_000, 180_000, 160_000, 150_000]
CHROMS = [f"chr{i + 1}" for i in range(len(SIZES))]
PSSM = pd.DataFrame({"A": [0.7, 0.1, 0.1, 0.1], "C": [0.1, 0.7, 0.1, 0.1],
                     "G": [0.1, 0.1, 0.7, 0.1], "T": [0.1, 0.1, 0.1, 0.7]})


def _random_intervals(rng, n, min_len, max_len, spacing):
    rows = []
    for chrom, size in zip(CHROMS, SIZES, strict=True):
        starts = np.sort(rng.choice(size // spacing - 1, size=n, replace=False)) * spacing
        rows.append(pd.DataFrame({"chrom": chrom, "start": starts,
                                  "end": starts + rng.integers(min_len, max_len + 1, size=n)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module", params=["indexed", "per-chromosome"])
def db(request, tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp(f"mt_{request.param}")
    fasta = root / "genome.fa"
    with open(fasta, "w") as f:
        for chrom, size in zip(CHROMS, SIZES, strict=True):
            seq = rng.choice(list("ACGTN"), size=size, p=[0.28, 0.22, 0.22, 0.27, 0.01])
            f.write(f">{chrom}\n" + "".join(seq) + "\n")
    groot = root / "db"
    pm.gdb_create(str(groot), str(fasta), db_format=request.param, verbose=False)
    pm.gdb_init(str(groot))
    try:
        pm.CONFIG["progress"] = False
        dense = _random_intervals(rng, 250, 50, 400, 500)
        pm.gtrack_create_dense("dense", "dense", dense, rng.normal(size=len(dense)), 50)
        sparse = _random_intervals(rng, 200, 20, 300, 600)
        pm.gtrack_create_sparse("sparse", "sparse", sparse, rng.normal(size=len(sparse)))
        yield {"mask": _random_intervals(rng, 40, 100, 2000, 2500),
               "source": _random_intervals(rng, 100, 50, 500, 1000).assign(score=lambda d: rng.normal(size=len(d)))}
    finally:
        pm.gvtrack_clear()
        pm.gdb_init(str(TESTDB_ROOT))


def _vtracks(db):
    pm.gvtrack_create("v_avg", "dense", func="avg")
    pm.gvtrack_create("v_filt", "dense", func="avg")
    pm.gvtrack_filter("v_filt", db["mask"])
    pm.gvtrack_create("v_shift", "dense", func="max")
    pm.gvtrack_iterator("v_shift", sshift=-300, eshift=300)
    pm.gvtrack_create("v_sparse", "sparse", func="sum")
    pm.gvtrack_filter("v_sparse", db["mask"])
    pm.gvtrack_create("v_df", db["source"], func="avg")
    pm.gvtrack_create("v_kmer", None, func="kmer.count", kmer="CA")
    pm.gvtrack_create("v_pwm", None, func="pwm.max", pssm=PSSM)


def _run(call, multitask, **cfg):
    """call() with multitasking on (min_scope4process 0, up to 8 kids) or off, and whether it forked.
    max_data_size is one a small host can map: the default, sized from the RAM, makes gquantiles'
    kids' blocks of a sparse track (whose bins cannot be estimated) more than half the RAM of a
    16 GB host, which then runs the call in one process."""
    saved = dict(pm.CONFIG)
    cfg = {"max_data_size": 10_000_000, **cfg}
    pm.CONFIG.update(multitasking=multitask, max_processes=8, min_scope4process=0, progress=False, **cfg)
    try:
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt
        res = call()
        return res, resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt > before
    finally:
        pm.CONFIG.clear()
        pm.CONFIG.update(saved)


def _both(call, forks=True, **cfg):
    single, forked = _run(call, False, **cfg)
    assert not forked
    multi, forked = _run(call, True, **cfg)
    assert forked == forks, "the multitasked call should have forked" if forks else "it should not fork"
    return single, multi


def _scope_with_gaps():
    """Several intervals per chromosome, starts off the bin grid, in the database's order"""
    rows = []
    for chrom, size in zip(pm.gintervals_all()["chrom"].astype(str), pm.gintervals_all()["end"], strict=True):
        for s in range(1037, size - 30_000, 47_000):
            rows.append((chrom, s, s + 29_011))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


# R misha's plan gives a kid at least 50000 bins: 2.9 Mb in bins of 10 bp make 6 kids
EXTRACT_CASES = [
    ("dense", 10), ("dense", None), ("sparse", None), ("sparse", 10),
    ("v_avg", 10), ("v_filt", 10), ("v_shift", 7), ("v_sparse", 9), ("v_df", 10),
    ("v_kmer", 10), ("v_pwm", 10), (["dense", "v_filt", "sparse", "v_kmer"], 10),
]


@pytest.mark.parametrize(("expr", "iterator"), EXTRACT_CASES)
@pytest.mark.parametrize("scope", ["all", "gaps"])
def test_gextract_rows_are_those_of_one_process(db, expr, iterator, scope):
    _vtracks(db)
    intervals = pm.gintervals_all() if scope == "all" else _scope_with_gaps()
    single, multi = _both(lambda: pm.gextract(expr, intervals, iterator=iterator))
    assert len(single) > 1000
    pdt.assert_frame_equal(single, multi)


def test_gextract_dataframe_iterator_and_join(db):
    _vtracks(db)
    itr = _random_intervals(np.random.default_rng(3), 30, 100, 3000, 4000)
    scope = _scope_with_gaps().assign(tag=lambda d: [f"s{i}" for i in range(len(d))])
    for kw in ({}, {"intervals_join": "intervals"}, {"colnames": ["a", "b"]}):
        single, multi = _both(lambda kw=kw: pm.gextract(["v_filt", "dense"], scope, iterator=itr, **kw))
        pdt.assert_frame_equal(single, multi)


def test_unsorted_scope(db):
    _vtracks(db)
    scope = _scope_with_gaps().iloc[::-1].reset_index(drop=True)
    # cut into ranges along the scope as given
    single, multi = _both(lambda: pm.gextract(["dense", "v_kmer"], scope, iterator=10))
    pdt.assert_frame_equal(single, multi)
    # by chromosome: a sparse iterator reads the scope a chromosome at a time, so one process does
    single, multi = _both(lambda: pm.gextract("sparse", scope), forks=False)
    pdt.assert_frame_equal(single, multi)


def test_unsorted_scope_of_per_interval_results(db):
    # each interval's quantiles go to its row, whatever the order: the kids take runs of a
    # chromosome's intervals, the same chromosome in several runs
    scope = _scope_with_gaps().sample(frac=1, random_state=3).reset_index(drop=True)
    single, multi = _both(lambda: pm.gintervals_quantiles("dense", [0.2, 0.5, 0.9], scope, iterator=10))
    pdt.assert_frame_equal(single, multi)


@pytest.mark.parametrize("expr", [
    "dense * 2 + v_avg", "-dense / 3 - 1.5e-1", "(dense > 0.5) & (v_kmer >= 1) | (sparse != sparse)",
    "dense - v_shift * (v_df + 2)", ["dense + 1", "v_filt < 0", "dense"],
])
def test_an_expression_a_kid_evaluates(db, expr):
    # arithmetic and comparisons of numbers, which a kid evaluates in C++ as numpy does
    _vtracks(db)
    single, multi = _both(lambda: pm.gextract(expr, pm.gintervals_all(), iterator=10))
    pdt.assert_frame_equal(single, multi)


@pytest.mark.parametrize("expr", ["np.log2(dense + 2)", "dense ** 2", "START + dense", "(dense > 0) + 1", "abs(dense)"])
def test_an_expression_only_python_evaluates_runs_in_one_process(db, expr):
    # a kid does not call into Python
    _vtracks(db)
    single, multi = _both(lambda: pm.gextract(expr, pm.gintervals_all(), iterator=10), forks=False)
    pdt.assert_frame_equal(single, multi)


@pytest.mark.parametrize(("expr", "iterator"), [("dense", 10), ("dense", None), ("v_filt", 10), ("v_kmer", 10),
                                                 ("sparse", None), ("v_df", 7)])
def test_gsummary(db, expr, iterator):
    _vtracks(db)
    single, multi = _both(lambda: pm.gsummary(expr, pm.gintervals_all(), iterator=iterator))
    # counts, min and max as one process; the sums merged in kid order, as R misha
    np.testing.assert_array_equal(single.iloc[:4].to_numpy(), multi.iloc[:4].to_numpy())
    np.testing.assert_allclose(single.iloc[4:].to_numpy(), multi.iloc[4:].to_numpy(), rtol=1e-12)


@pytest.mark.parametrize(("expr", "iterator"), [("dense", 10), ("v_filt", 10), ("v_kmer", 10), ("sparse", None)])
def test_gquantiles(db, expr, iterator):
    _vtracks(db)
    p = [0, 0.1, 0.5, 0.77, 0.99, 1]
    state = np.random.get_state()[1].copy()
    single, multi = _both(lambda: pm.gquantiles(expr, p, pm.gintervals_all(), iterator=iterator))
    pdt.assert_series_equal(single, multi)
    # nothing sampled: numpy's generator is not drawn from
    np.testing.assert_array_equal(np.random.get_state()[1], state)


def test_gquantiles_sampled(db):
    # A kid samples max_data_size / kids values: the quantiles are estimates, the same for the same
    # numpy seed
    p = [0.1, 0.5, 0.9]

    def call():
        np.random.seed(5)
        with pytest.warns(RuntimeWarning, match="approximate"):
            return pm.gquantiles("dense", p, pm.gintervals_all(), iterator=10)

    exact, _ = _run(lambda: pm.gquantiles("dense", p, pm.gintervals_all(), iterator=10), False)
    a, forked = _run(call, True, max_data_size=20_000)
    b, _ = _run(call, True, max_data_size=20_000)
    assert forked
    pdt.assert_series_equal(a, b)
    np.testing.assert_allclose(a.to_numpy(), exact.to_numpy(), atol=0.1)


@pytest.mark.parametrize(("expr", "iterator"), [("dense > -100", 10), ("dense > 0.2", 10), ("v_filt <= 0", 10),
                                                 ("sparse > 0", None), ("(v_kmer > 1) & (dense < 1)", 10)])
def test_gscreen(db, expr, iterator):
    # "dense > -100" holds wherever dense has a value, in runs that cross the cuts between kids,
    # which one process merges
    _vtracks(db)
    single, multi = _both(lambda: pm.gscreen(expr, pm.gintervals_all(), iterator=iterator))
    pdt.assert_frame_equal(single, multi)


def test_gdist_glookup_gintervals_quantiles_gcor(db):
    _vtracks(db)
    scope = pm.gintervals_all()
    breaks = [-3, -1, 0, 0.5, 1, 3]
    calls = [
        lambda: pm.gdist("dense", breaks, intervals=scope, iterator=10),
        lambda: pm.gdist("dense", breaks, "v_kmer", [0, 1, 2, 5], intervals=scope, iterator=10),
        lambda: pm.glookup(np.arange(5.0), "dense", breaks, intervals=scope, iterator=10),
        lambda: pm.gintervals_quantiles("dense", [0.2, 0.5], _scope_with_gaps(), iterator=50),
        lambda: pm.gcor("dense", "v_filt", intervals=scope, iterator=10, method="spearman.exact"),
    ]
    for call in calls:
        single, multi = _both(call)
        if isinstance(single, pd.DataFrame):
            pdt.assert_frame_equal(single, multi)
        else:
            np.testing.assert_array_equal(single, multi)
    single, multi = _both(lambda: pm.gcor("dense", "v_filt", "dense", "sparse", intervals=scope, iterator=10))
    np.testing.assert_allclose(single, multi, rtol=1e-12)


# With a DataFrame iterator the scope is cut into the iterator's intervals; the kids take whole
# chromosomes of them, as R misha's take whole chromosomes of the scope
DF_ITERATOR_CALLS = {
    "gsummary": lambda itr: pm.gsummary("v_filt", pm.gintervals_all(), iterator=itr),
    "gquantiles": lambda itr: pm.gquantiles("dense", [0, 0.1, 0.5, 0.9, 1], pm.gintervals_all(), iterator=itr),
    "gdist": lambda itr: pm.gdist("dense", [-3, -1, 0, 0.5, 1, 3], intervals=pm.gintervals_all(), iterator=itr),
    "gscreen": lambda itr: pm.gscreen("v_kmer > 1", pm.gintervals_all(), iterator=itr),
    "glookup": lambda itr: pm.glookup(np.arange(4.0), "dense", [-3, -1, 0, 1, 3], intervals=pm.gintervals_all(),
                                      iterator=itr),
    "gintervals_quantiles": lambda itr: pm.gintervals_quantiles("dense", [0.2, 0.5], _scope_with_gaps(),
                                                                iterator=itr),
    "gintervals_summary": lambda itr: pm.gintervals_summary("v_filt", _scope_with_gaps(), iterator=itr),
    "gextract": lambda itr: pm.gextract(["dense", "v_df"], _scope_with_gaps(), iterator=itr),
}


@pytest.mark.parametrize("name", DF_ITERATOR_CALLS)
@pytest.mark.parametrize("order", ["sorted", "shuffled"])
def test_dataframe_iterator(db, name, order):
    _vtracks(db)
    itr = _random_intervals(np.random.default_rng(5), 40, 100, 3000, 3500)
    if order == "shuffled":
        itr = itr.sample(frac=1, random_state=1).reset_index(drop=True)
    single, multi = _both(lambda: DF_ITERATOR_CALLS[name](itr))
    if name == "gsummary":
        # sums merged in kid order, as R misha
        np.testing.assert_array_equal(single.iloc[:4].to_numpy(), multi.iloc[:4].to_numpy())
        np.testing.assert_allclose(single.iloc[4:].to_numpy(), multi.iloc[4:].to_numpy(), rtol=1e-12)
    elif isinstance(single, pd.DataFrame):
        pdt.assert_frame_equal(single, multi)
    elif isinstance(single, pd.Series):
        pdt.assert_series_equal(single, multi)
    else:
        np.testing.assert_array_equal(single, multi)


def _create_both(expr, iterator, forks=True):
    """The data files of the track gtrack_create makes in one process and multitasked (the
    attributes, which hold the date, aside)"""
    files = {}
    for multitask in (False, True):
        name = f"mt_create_{int(multitask)}"
        if pm.gtrack_exists(name):
            pm.gtrack_rm(name, force=True)
        _, forked = _run(lambda name=name: pm.gtrack_create(name, "", expr, iterator=iterator), multitask)
        assert forked == (multitask and forks)
        path = Path(_pymisha.pm_track_path(name))
        files[multitask] = {f.name: f.read_bytes() for f in sorted(path.iterdir()) if f.name != ".attributes"}
        pm.gtrack_rm(name, force=True)
    assert files[False], "no file"
    return files[False], files[True]


# As R misha's gtrack.create: the kids take whole chromosomes and write their files
@pytest.mark.parametrize(("expr", "iterator"), [("dense * 2", 10), ("dense", None), ("sparse", None), ("v_filt", 50),
                                                 ("v_kmer", 25), ("dense + v_df", 10), ("dense > 0", 10)])
def test_gtrack_create_writes_the_files_of_one_process(db, expr, iterator):
    _vtracks(db)
    single, multi = _create_both(expr, iterator)
    assert single == multi


@pytest.mark.parametrize("expr", ["dense", "v_filt", "sparse * 2"])
def test_gtrack_create_with_a_dataframe_iterator(db, expr):
    _vtracks(db)
    itr = _random_intervals(np.random.default_rng(5), 40, 100, 3000, 3500)
    single, multi = _create_both(expr, itr)
    assert single == multi


def test_gtrack_create_of_an_expression_only_python_evaluates_runs_in_one_process(db):
    single, multi = _create_both("np.log2(dense + 2)", 10, forks=False)
    assert single == multi


def test_max_data_size_counts_the_rows_of_all_the_kids(db):
    for multitask in (False, True):
        with pytest.raises(pm.error, match=r"Result size \(at least \d+ .*\) exceeded the maximum allowed"):
            _run(lambda: pm.gextract("dense", pm.gintervals_all(), iterator=10), multitask, max_data_size=20_000)
    # a limit that each kid's rows fit but not all of them together
    rows = len(pm.gextract("dense", pm.gintervals_all(), iterator=10))
    with pytest.raises(pm.error, match="exceeded the maximum allowed"):
        _run(lambda: pm.gextract("dense", pm.gintervals_all(), iterator=10), True, max_data_size=rows - 1)
    res, forked = _run(lambda: pm.gextract("dense", pm.gintervals_all(), iterator=10), True, max_data_size=rows)
    assert forked and len(res) == rows
