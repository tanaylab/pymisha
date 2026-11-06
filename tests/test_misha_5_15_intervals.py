"""R misha 5.15.0 interval fixes (test-review-2026-11-intervals.R, test-liftover.R:599),
against the values R 5.15.1 gives on this test DB: liftover uses the chain's source strand
and a pre-loaded chain's min_score, neighbors read strand labels, annotate takes its columns
by position, giterator_intervals keeps intervalID with interval_relative, summary Std dev is
accurate for large values, gintervals_random weighs every start the same, force_range keeps
alias chromosomes."""

import collections

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha
from pymisha import intervals as _iv

DF = pd.DataFrame
Q = DF(dict(chrom=["chr2"], start=[900], end=[910]))


def _as_chain(strand, strandsrc):
    return pm.gintervals_as_chain(DF(dict(
        chrom=["chr1"], start=[5000], end=[5100], strand=[strand], chromsrc=["chr2"], startsrc=[900],
        endsrc=[1000], strandsrc=[strandsrc], chain_id=[1], score=[1.0])))


def _lifted(chain, q=Q, **kw):
    r = pm.gintervals_liftover(q, chain, **kw)
    return list(zip(r["start"], r["end"], strict=True))


def _chain_file(tmp_path, name, ts, tstart, qs, qstart):
    p = tmp_path / f"{name}.chain"
    p.write_text(f"chain 1 chr2 300000 {ts} {tstart} {tstart + 100} "
                 f"chr1 500000 {qs} {qstart} {qstart + 100} 1\n100\n\n")
    return str(p)


@pytest.fixture()
def chain_files(tmp_path):
    return dict(ref_minus=_chain_file(tmp_path, "r", "-", 300000 - 1000, "+", 5000),
                query_minus=_chain_file(tmp_path, "q", "+", 900, "-", 500000 - 5100),
                both_minus=_chain_file(tmp_path, "b", "-", 300000 - 1000, "-", 500000 - 5100))


@pytest.mark.parametrize("strand,strandsrc,want", [  # R :217
    (1, 1, (5000, 5010)), (-1, -1, (5000, 5010)), (1, -1, (5090, 5100)), (-1, 1, (5090, 5100))])
def test_liftover_reads_the_source_strand_of_as_chain(strand, strandsrc, want):
    assert _lifted(_as_chain(strand, strandsrc)) == [want]
    assert _as_chain(strand, strandsrc)[["strand", "strandsrc"]].values.tolist() == [[strand, strandsrc]]


@pytest.mark.parametrize("name,want,strands", [  # R :217, test-liftover.R:599
    ("query_minus", (5090, 5100), [-1, 1]), ("ref_minus", (5090, 5100), [-1, 1]),
    ("both_minus", (5000, 5010), [1, 1])])
def test_chain_files_fold_the_source_strand(chain_files, name, want, strands):
    assert _lifted(chain_files[name]) == [want]
    ch = pm.gintervals_load_chain(chain_files[name])
    assert ch[["start", "end", "startsrc", "endsrc", "strand", "strandsrc"]].values.tolist() == \
        [[5000, 5100, 900, 1000, *strands]]


@pytest.fixture()
def src_track():
    names = []

    def make(start):
        name = f"r515_lift_src_{start}"
        if pm.gtrack_exists(name):
            pm.gtrack_rm(name, force=True)
        pm.gtrack_create_sparse(name, "", DF(dict(chrom=["chr2"], start=[start], end=[start + 10])), [7.0])
        names.append(name)
        return name
    yield make
    for n in [*names, "r515_lifted"]:
        if pm.gtrack_exists(n):
            pm.gtrack_rm(n, force=True)
    _pymisha.pm_dbreload()


def _glift(src, chain, iv, **kw):
    if pm.gtrack_exists("r515_lifted"):
        pm.gtrack_rm("r515_lifted", force=True)
    pm.gtrack_liftover("r515_lifted", "", f"{pm._shared._GROOT}/tracks/{src}.track", chain, **kw)
    r = pm.gextract("r515_lifted", gi(*iv))
    return r[["start", "end", "r515_lifted"]].values.tolist()


def gi(c, s, e):
    return pm.gintervals(c, s, e)


@pytest.mark.parametrize("pure", [False, True])
def test_gtrack_liftover_reads_the_source_strand(src_track, chain_files, pure):  # R :261
    src = src_track(900)
    kw = {"_force_pure_python": True} if pure else {}
    assert _glift(src, chain_files["ref_minus"], ("chr1", 4000, 6000), **kw) == [[5090, 5100, 7]]
    assert _glift(src, chain_files["both_minus"], ("chr1", 4000, 6000), **kw) == [[5000, 5010, 7]]
    assert _glift(src, _as_chain(1, -1), ("chr1", 4000, 6000), **kw) == [[5090, 5100, 7]]


TWO = DF(dict(chrom=["chr1"] * 2, start=[1000, 3000], end=[2000, 4000], strand=[1, 1], chromsrc=["chr2"] * 2,
              startsrc=[100] * 2, endsrc=[1100] * 2, strandsrc=[1, 1], chain_id=[1, 2], score=[10.0, 1000.0]))
KEEP = dict(src_overlap_policy="keep", tgt_overlap_policy="keep")


def test_strand_0_is_forward():  # R :76
    ch = pm.gintervals_as_chain(DF(dict(chrom=["chr1"], start=[1000], end=[2000], strand=[0], chromsrc=["chr1"],
                                        startsrc=[5000], endsrc=[6000], strandsrc=[0], chain_id=[1], score=[1000.0])))
    assert _lifted(ch, DF(dict(chrom=["chr1"], start=[5100], end=[5200]))) == [(1100, 1200)]


def test_min_score_of_a_pre_loaded_chain():  # R :76
    q = DF(dict(chrom=["chr2"], start=[200], end=[300]))
    assert _lifted(pm.gintervals_as_chain(TWO, **KEEP, min_score=100), q) == [(3100, 3200)]
    assert _lifted(TWO, q, **KEEP, min_score=100) == [(3100, 3200)]
    na = TWO.assign(score=[np.nan, 1000.0])
    assert _lifted(na, q, **KEEP, min_score=100) == [(3100, 3200)]
    assert _lifted(pm.gintervals_as_chain(na, **KEEP, min_score=100), q) == [(3100, 3200)]


@pytest.mark.parametrize("pure", [False, True])
def test_gtrack_liftover_applies_min_score(src_track, pure):  # R :318
    src = src_track(200)
    kw = {"_force_pure_python": True} if pure else {}
    iv = ("chr1", 0, 5000)
    assert _glift(src, pm.gintervals_as_chain(TWO, **KEEP, min_score=100), iv, **kw) == [[3100, 3110, 7]]
    assert _glift(src, TWO, iv, **KEEP, min_score=100, **kw) == [[3100, 3110, 7]]
    assert _glift(src, TWO.assign(score=[np.nan, 1000.0]), iv, **KEEP, min_score=100, **kw) == [[3100, 3110, 7]]


TSS = DF(dict(chrom=["chr1"] * 2, start=[1000, 5000], end=[1001, 5001], strand=pd.Categorical(["+", "-"])))
FEAT = DF(dict(chrom=["chr1"] * 2, start=[800, 4800], end=[810, 4810]))


@pytest.mark.parametrize("strand", [
    pd.Categorical(["+", "-"]), pd.Categorical(["+", "-"], categories=["-", "+"]), ["+", "-"],
    pd.Categorical([1, -1]), [1, -1], [True, False], ["1", "-1"]])
def test_neighbors_read_strand_labels(strand):  # R :5
    r = pm.gintervals_neighbors(TSS.assign(strand=strand), FEAT, use_intervals1_strand=True)
    assert r["dist"].tolist() == [-190, 190]
    assert list(r["strand"]) == list(strand)  # the query's own strand column


def test_upstream_downstream_with_a_categorical_strand():  # R :5
    up = pm.gintervals_neighbors_upstream(TSS, FEAT, maxdist=1000)
    assert up[["start", "dist"]].values.tolist() == [[1000, -190]]
    assert pm.gintervals_neighbors_downstream(TSS, FEAT, maxdist=1000)[["start", "dist"]].values.tolist() == \
        [[5000, 190]]


QA = DF(dict(chrom=["chr1"] * 2, start=[100, 5000], end=[110, 5010], gene=["Q1", "Q2"]))
ANN = DF(dict(chrom=["chr1"] * 2, start=[200, 5100], end=[210, 5110], gene=["A1", "A2"]))


def test_annotate_with_shared_column_names():  # R :23
    r = pm.gintervals_annotate(QA, ANN, annotation_columns=["gene"], column_names=["ann_gene"])
    assert r[["ann_gene", "gene", "dist"]].values.tolist() == [["A1", "Q1", 90], ["A2", "Q2", 90]]
    assert pm.gintervals_annotate(QA, ANN, annotation_columns=["gene"], overwrite=True)["gene"].tolist() == \
        ["A1", "A2"]
    q2 = DF(dict(chrom=["chr1"] * 2, start=[100, 5000], end=[110, 5010], dist=[-7, -8]))
    ann2 = ANN.rename(columns={"gene": "label"})
    r = pm.gintervals_annotate(q2, ann2, annotation_columns=["label"], dist_column="distance")
    assert r[["distance", "dist"]].values.tolist() == [[90, -7], [90, -8]]
    r = pm.gintervals_annotate(q2, ann2, annotation_columns=["label"], overwrite=True)
    assert r["dist"].tolist() == [90, 90] and list(r.columns).count("dist") == 1


def test_annotate_max_dist_blanks_the_annotation():  # R :55
    ann3 = DF(dict(chrom=["chr1"] * 2, start=[200, 5015], end=[210, 5025], gene=["A1", "A2"]))
    r = pm.gintervals_annotate(QA, ann3, annotation_columns=["gene"], overwrite=True, max_dist=50)
    assert r["dist"].tolist() == [90, 5] and pd.isna(r["gene"][0]) and r["gene"][1] == "A2"
    r = pm.gintervals_annotate(QA, ann3, annotation_columns=["gene"], column_names=["ann_gene"], max_dist=50)
    assert r["gene"].tolist() == ["Q1", "Q2"] and pd.isna(r["ann_gene"][0]) and r["ann_gene"][1] == "A2"


def test_annotate_tie_method():  # R :379
    qq = DF(dict(chrom=["chr1"], start=[1000], end=[1100]))
    annt = DF(dict(chrom=["chr1"] * 2, start=[1200, 800], end=[1300, 900], label=["right", "left"]))
    for keep_order in (True, False):
        r = pm.gintervals_annotate(qq, annt, annotation_columns=["label"], maxneighbors=2, tie_method="min.start",
                                   keep_order=keep_order)
        assert r["label"].tolist() == ["left", "right"]
        assert "_orig_order" not in r.columns
    anns = DF(dict(chrom=["chr1"] * 3, start=[1103, 700, 1390], end=[1110, 800, 1400], strand=[-1, 1, 1],
                   label=["near", "mid", "far"]))
    for tie in ("first", "min.start"):
        r = pm.gintervals_annotate(qq, anns, annotation_columns=["label"], maxneighbors=3, tie_method=tie)
        assert r[["label", "dist"]].values.tolist() == [["near", 3], ["mid", 200], ["far", -290]]


def test_giterator_intervals_interval_relative():  # R :67
    iv = DF(dict(chrom=["chr2", "chr1", "chr1"], start=[100, 500, 100], end=[200, 600, 180]))
    r = pm.giterator_intervals(None, iv, iterator=50, interval_relative=True)
    assert r.values.tolist() == [["chr1", 100, 150, 3], ["chr1", 150, 180, 3], ["chr1", 500, 550, 2],
                                 ["chr1", 550, 600, 2], ["chr2", 100, 150, 1], ["chr2", 150, 200, 1]]
    assert pm.giterator_intervals(None, iv, iterator=50).values.tolist() == [row[:3] for row in r.values.tolist()]


def test_summary_stddev_for_large_values(monkeypatch):  # R :164, :193
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    scope = DF(dict(chrom=["chr1", "chr2"], start=[0, 0], end=[200000, 200000]))
    expr = "dense_track + 1e8"
    vals = pm.gextract(expr, scope, iterator="dense_track", colnames=["v"])
    np.testing.assert_allclose(float(pm.gsummary(expr, scope, iterator="dense_track")["Std dev"]),
                               vals["v"].std(), rtol=1e-9)
    np.testing.assert_allclose(pm.gintervals_summary(expr, scope, iterator="dense_track")["Std dev"],
                               vals.groupby("intervalID")["v"].std(), rtol=1e-6)
    allv = pm.gextract(expr, pm.gintervals_all(), iterator=1000, colnames=["v"])["v"]
    want = allv.std()
    np.testing.assert_allclose(float(pm.gsummary(expr, pm.gintervals_all(), iterator=1000)["Std dev"]), want,
                               rtol=1e-9)
    monkeypatch.setitem(pm.CONFIG, "multitasking", True)
    monkeypatch.setitem(pm.CONFIG, "max_processes", 4)
    monkeypatch.setitem(pm.CONFIG, "min_processes", 4)
    monkeypatch.setitem(pm.CONFIG, "min_scope4process", 1)
    np.testing.assert_allclose(float(pm.gsummary(expr, pm.gintervals_all(), iterator=1000)["Std dev"]), want,
                               rtol=1e-9)


def test_gbins_summary_with_a_track_iterator():
    r = pm.gbins_summary("dense_track", [0, 0.1, 0.2, 1], expr="dense_track", intervals=gi("chr1", 0, 10000),
                         iterator="dense_track")
    assert np.asarray(r).shape == (3, 7)


def test_array_slice_stddev_at_a_large_offset(tmp_path):  # R :295
    vals = 1e8 + np.array([0, 8, 16, 24, 40.0])
    f = tmp_path / "arr.tsv"
    DF(dict(chrom=["chr1"], start=[100], end=[200], a=[vals[0]], b=[vals[1]], c=[vals[2]], d=[vals[3]],
            e=[vals[4]])).to_csv(f, sep="\t", index=False)
    name = "r515_arrsd"
    if pm.gtrack_exists(name):
        pm.gtrack_rm(name, force=True)
    pm.gtrack_array_import(name, "", str(f))
    try:
        pm.gvtrack_create("s1", name)
        pm.gvtrack_array_slice("s1", None, "stddev")
        pm.gvtrack_create("s2", name)
        pm.gvtrack_array_slice("s2", ["e", "a", "c"], "stddev")
        r = pm.gextract(["s1", "s2"], gi("chr1", 100, 200))
        np.testing.assert_allclose(r["s1"], [np.std(vals, ddof=1)], rtol=1e-6)
        np.testing.assert_allclose(r["s2"], [np.std(vals[[4, 0, 2]], ddof=1)], rtol=1e-6)
    finally:
        pm.gvtrack_clear()
        pm.gtrack_rm(name, force=True)
        _pymisha.pm_dbreload()



def _cpp(size, n, chroms, filt=None):
    ga = pm.gintervals_all()
    return _iv._gintervals_random_cpp(size, n, 0.0, ga[ga["chrom"].isin(chroms)], filt, 60427)


@pytest.mark.parametrize("path", ["python", "cpp"])
def test_gintervals_random_weighs_every_start_the_same(path):  # R :155, :354
    def draw(size, n, chroms, filt=None):
        if path == "cpp":
            return _cpp(size, n, chroms, filt)
        np.random.seed(60427)
        return pm.gintervals_random(size, n, dist_from_edge=0, chromosomes=chroms, filter=filt)

    filt = DF(dict(chrom=["chr1"] * 2, start=[0, 2150], end=[2000, 500000]))
    u = np.unique(draw(100, 3000, ["chr1"], filt)["start"])
    assert (len(u), u.min(), u.max()) == (51, 2000, 2050)  # a gap of 2 * size + 50 starts
    r = draw(10, 400000, ["chrX"])
    assert (r["start"].min(), r["start"].max()) == (0, 199990)  # the last start can be drawn
    r = draw(10, 2000, ["chrX"], DF(dict(chrom=["chrX"], start=[0], end=[199985])))
    assert sorted(set(r["start"])) == [199985, 199986, 199987, 199988, 199989, 199990]
    r = draw(1000, 2000, ["chr1", "chr2"], DF(dict(chrom=["chr1", "chr2"], start=[1000, 0], end=[500000, 299000])))
    assert set(collections.Counter(zip(r["chrom"], r["start"], strict=True))) == {("chr1", 0), ("chr2", 299000)}
    frac = draw(150000, 100000, ["chr1", "chr2", "chrX"])["chrom"].value_counts(normalize=True)
    starts = np.array([350001, 150001, 50001])  # weights by number of starts, not by length
    np.testing.assert_allclose(frac[["chr1", "chr2", "chrX"]], starts / starts.sum(), atol=0.01)


def test_force_range_keeps_alias_chromosomes():  # R :114
    r = pm.gintervals_force_range(DF(dict(chrom=["1", "chrZZ", "chr2"], start=[-10, 0, 10], end=[60, 5, 20])))
    assert r.values.tolist() == [["chr1", 0, 60], ["chr2", 10, 20]]
