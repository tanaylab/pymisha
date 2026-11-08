"""Behaviour of R misha 5.16.0 that pymisha follows, checked against R 5.16.0 on the test
database: gintervals_canonic of 2D intervals, missing strands, strand labels and boolean strand
columns, R's band error, R's errors for bad breaks
(gcis_decay, gdist, glookup, gtrack_lookup, gbins_summary, gbins_quantiles), gintervals_mapply's
gapply_intervals (R's GAPPLY.INTERVALS) and a categorical strand in gseq_extract."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha

DF = pd.DataFrame
COLS_2D = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]


def _rows(df):
    return [tuple(r) for r in df.itertuples(index=False)]


# --- gintervals_canonic of 2D intervals: sorted by chromosome pair, nothing merged ---

I2 = DF(dict(chrom1=["chr2", "chr1", "chr1"], start1=[100, 500, 100], end1=[200, 600, 200],
             chrom2=["chr1", "chr2", "chr1"], start2=[100, 300, 1000], end2=[200, 400, 2000]))
OVERLAPPING = DF(dict(chrom1=["chr1", "chr1"], start1=[100, 150], end1=[200, 250],
                      chrom2=["chr1", "chr1"], start2=[100, 150], end2=[200, 250]))


@pytest.mark.parametrize("unify", [True, False])
def test_canonic_2d_sorts_by_chromosome_pair_and_keeps_the_coordinates(unify):
    res = pm.gintervals_canonic(I2.assign(val=[1.0, 2.0, 3.0]), unify)
    assert list(res.columns) == COLS_2D
    assert _rows(res) == [("chr1", 100, 200, "chr1", 1000, 2000), ("chr1", 500, 600, "chr2", 300, 400),
                          ("chr2", 100, 200, "chr1", 100, 200)]
    assert list(res.attrs["mapping"]) == [3, 2, 1]


@pytest.mark.parametrize("unify", [True, False])
def test_canonic_2d_does_not_merge_touching_intervals(unify):
    touch = DF(dict(chrom1=["chr1", "chr1"], start1=[100, 200], end1=[200, 300],
                    chrom2=["chr1", "chr1"], start2=[100, 100], end2=[200, 200]))
    res = pm.gintervals_canonic(touch, unify)
    assert _rows(res) == _rows(touch)
    assert list(res.attrs["mapping"]) == [1, 2]


def test_canonic_2d_names_chromosomes_as_the_database():
    res = pm.gintervals_canonic(DF(dict(chrom1=["2", "1"], start1=[0, 0], end1=[10, 10],
                                        chrom2=["1", "1"], start2=[0, 0], end2=[10, 10])))
    assert _rows(res) == [("chr1", 0, 10, "chr1", 0, 10), ("chr2", 0, 10, "chr1", 0, 10)]
    assert list(res.attrs["mapping"]) == [2, 1]
    res = pm.gintervals_canonic(pm.gintervals_2d(["chr2", "chr1"], [10, 0], [20, 5], ["chr1", "chr1"], [0, 0], [5, 5]))
    assert _rows(res) == [("chr1", 0, 5, "chr1", 0, 5), ("chr2", 10, 20, "chr1", 0, 5)]


def test_canonic_2d_of_one_or_no_interval():
    res = pm.gintervals_canonic(I2.iloc[[0]])
    assert _rows(res) == [("chr2", 100, 200, "chr1", 100, 200)]
    assert list(res.attrs["mapping"]) == [1]
    assert pm.gintervals_canonic(I2.iloc[0:0]) is None


def test_canonic_2d_overlapping_intervals_are_an_error():
    with pytest.raises(_pymisha.error, match=r"^Intervals \(chr1, 150, 250, chr1, 150, 250\) and "
                                             r"\(chr1, 100, 200, chr1, 100, 200\) overlap$"):
        pm.gintervals_canonic(OVERLAPPING)
    with pytest.raises(_pymisha.error, match=r"^Intervals \(chr2, 100, 200, chr1, 100, 200\) and "
                                             r"\(chr2, 100, 200, chr1, 100, 200\) overlap$"):
        pm.gintervals_canonic(pd.concat([I2.iloc[[0]], I2.iloc[[0]]]))


def test_canonic_2d_invalid_interval_is_an_error():
    bad = DF(dict(chrom1=["chr1"], start1=[200], end1=[100], chrom2=["chr1"], start2=[0], end2=[10]))
    with pytest.raises(ValueError, match=r"Interval \(chr1, 200, 100, chr1, 0, 10\): start coordinate must be "
                                         "lesser than end coordinate"):
        pm.gintervals_canonic(bad)


def test_canonic_2d_of_many_intervals_is_sorted_by_chromosome_pair():
    rng = np.random.default_rng(60427)
    n = 300
    cell = rng.choice(100 * 100, n, replace=False)
    df = DF(dict(chrom1=rng.choice(["chr1", "chr2", "chrX"], n), start1=(cell // 100) * 1000,
                 end1=(cell // 100) * 1000 + 1000, chrom2=rng.choice(["chr1", "chr2", "chrX"], n),
                 start2=(cell % 100) * 1000, end2=(cell % 100) * 1000 + 1000))
    res = pm.gintervals_canonic(df)
    mapping = res.attrs["mapping"]
    assert sorted(mapping) == list(range(1, n + 1))
    assert _rows(res.iloc[mapping - 1]) == _rows(df)
    ids = {"chr1": 0, "chr2": 1, "chrX": 2}
    pairs = [(ids[a], ids[b]) for a, b in zip(res["chrom1"], res["chrom2"], strict=True)]
    assert pairs == sorted(pairs)


def test_mark_overlaps_of_2d_intervals():
    res = pm.gintervals_mark_overlaps(I2)
    assert _rows(res[COLS_2D]) == _rows(I2)
    assert list(res["overlap_group"]) == [3, 2, 1]
    with pytest.raises(_pymisha.error, match="overlap"):
        pm.gintervals_mark_overlaps(OVERLAPPING)


# --- a missing strand is an error in gintervals_canonic and gintervals_neighbors ---

BASE = DF(dict(chrom="chr1", start=[100, 300, 500], end=[200, 400, 600]))
TARGETS = DF(dict(chrom="chr1", start=[1000, 2000], end=[1100, 2100]))
NA_STRANDS = [["+", None, "-"], [1, np.nan, -1], pd.array([1, None, -1], dtype="Int64"),
              pd.Categorical(["+", None, "-"])]
NA_ERROR = r"^Invalid strand value \(NA\) at index 2$"
# a missing double is a bad interval, as R says of NA_real_ (5.16.1)
NA_FLOAT_ERROR = r"^Invalid format of interval at index 2$"


def na_error(strand):
    return NA_FLOAT_ERROR if isinstance(strand, list) and any(isinstance(x, float) for x in strand) else NA_ERROR


@pytest.mark.parametrize("strand", NA_STRANDS)
def test_canonic_missing_strand_is_an_error(strand):
    with pytest.raises(_pymisha.error, match=na_error(strand)):
        pm.gintervals_canonic(BASE.assign(strand=strand))


@pytest.mark.parametrize("strand", [["+", "-", "-"], [1, -1, 0]])
def test_canonic_strand_without_missing_values_is_dropped(strand):
    res = pm.gintervals_canonic(BASE.assign(strand=strand))
    assert _rows(res) == _rows(BASE)
    assert list(res.attrs["mapping"]) == [1, 2, 3]


@pytest.mark.parametrize("use_intervals1_strand", [False, True])
@pytest.mark.parametrize("strand", NA_STRANDS)
def test_neighbors_missing_strand_of_intervals1_is_an_error(strand, use_intervals1_strand):
    with pytest.raises(_pymisha.error, match=na_error(strand)):
        pm.gintervals_neighbors(BASE.assign(strand=strand), TARGETS, use_intervals1_strand=use_intervals1_strand,
                                warn_ignored_strand=False)


@pytest.mark.parametrize("strand", [["+", None], [1, np.nan], pd.array([1, None], dtype="Int64"),
                                    pd.Categorical(["+", None])])
def test_neighbors_missing_strand_of_intervals2_is_an_error(strand):
    with pytest.raises(_pymisha.error, match=na_error(strand)):
        pm.gintervals_neighbors(BASE, TARGETS.assign(strand=strand))


def test_neighbors_strand_of_intervals1_without_missing_values_still_works():
    res = pm.gintervals_neighbors(BASE.assign(strand=[".", "-", "+"]), TARGETS, use_intervals1_strand=True)
    assert list(res["dist"]) == [800, -600, 400]


# --- strand labels: "+", "-", ".", "*" or "", anything else is R's error ---

BAD_LABEL = 'Bad strand value "{}" (expected "+", "-", ".", "*" or "")'
B3 = BASE.iloc[:3]


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("bad", ["x", "1", "-1", "++"])
def test_canonic_bad_strand_label_is_an_error(bad, categorical):
    strand = ["+", bad, "-"]
    with pytest.raises(_pymisha.error) as err:
        pm.gintervals_canonic(B3.assign(strand=pd.Categorical(strand) if categorical else strand))
    assert str(err.value) == BAD_LABEL.format(bad)


@pytest.mark.parametrize("strand", [["+", ".", "*"], ["", "-", "+"], pd.Categorical(["+", ".", "*"]),
                                    pd.Categorical(["+", "-", "+"], categories=["+", "-", "x"])])
def test_canonic_valid_strand_labels(strand):
    assert _rows(pm.gintervals_canonic(B3.assign(strand=strand))) == _rows(B3)


NB_QUERY = DF(dict(chrom="chr1", start=[100, 300, 500, 5000], end=[200, 400, 600, 5100]))
NB_TARGETS = DF(dict(chrom="chr1", start=[1000, 2000, 3000], end=[1100, 2100, 3100]))


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("strand, dists", [
    (["+", "-", "."], [-800, -600, -400, 1900]),
    (["-", "+", "*"], [800, 600, 400, 1900]),
    (["", "-", "+"], [800, 600, 400, 1900]),
])
def test_neighbors_reads_strand_labels_of_intervals2(strand, dists, categorical):
    col = pd.Categorical(strand) if categorical else strand
    r = pm.gintervals_neighbors(NB_QUERY, NB_TARGETS.assign(strand=col))
    assert r["dist"].tolist() == dists
    assert list(r["strand"]) == [strand[0]] * 3 + [strand[2]]  # the target's own label
    assert isinstance(r["strand"].dtype, pd.CategoricalDtype) == categorical


def test_neighbors_strand_labels_of_intervals2_as_numbers():
    r = pm.gintervals_neighbors(NB_QUERY, NB_TARGETS.assign(strand=[1, -1, 0]))
    assert r["dist"].tolist() == [-800, -600, -400, 1900]


def test_neighbors_strand_labels_of_intervals2_with_several_neighbors():
    r = pm.gintervals_neighbors(NB_QUERY, NB_TARGETS.assign(strand=["-", "+", "."]), maxneighbors=2)
    assert r["start1"].tolist() == [1000, 2000, 1000, 2000, 1000, 2000, 3000, 2000]
    assert r["strand"].tolist() == ["-", "+", "-", "+", "-", "+", ".", "+"]
    assert r["dist"].tolist() == [800, -1800, 600, -1600, 400, -1400, 1900, 2900]


def test_neighbors_strand_labels_of_intervals2_with_a_query_without_neighbor():
    q = pd.concat([NB_QUERY, DF(dict(chrom=["chr2"], start=[0], end=[10]))], ignore_index=True)
    r = pm.gintervals_neighbors(q, NB_TARGETS.assign(strand=["+", "-", "."]), na_if_notfound=True)
    assert r["strand"].iloc[:4].tolist() == ["+", "+", "+", "."]
    assert r["dist"].iloc[:4].tolist() == [-800, -600, -400, 1900]
    assert r.iloc[4][["chrom1", "start1", "strand", "dist"]].isna().all()


@pytest.mark.parametrize("categorical", [False, True])
def test_neighbors_reads_strand_labels_of_intervals1_it_does_not_use(categorical):
    strand = ["+", "-", ".", "-"]
    q = NB_QUERY.assign(strand=pd.Categorical(strand) if categorical else strand)
    r = pm.gintervals_neighbors(q, NB_TARGETS, warn_ignored_strand=False)
    assert r["dist"].tolist() == [800, 600, 400, 1900]
    assert list(r["strand"]) == strand


@pytest.mark.parametrize("use_intervals1_strand, dists", [(False, [800, 600, 400, 1900]),
                                                          (True, [800, -600, 400, 1900])])
def test_neighbors_strand_labels_of_both_sets(use_intervals1_strand, dists):
    r = pm.gintervals_neighbors(NB_QUERY.assign(strand=["+", "-", ".", "-"]), NB_TARGETS.assign(strand=["-", "+", "."]),
                                use_intervals1_strand=use_intervals1_strand, warn_ignored_strand=False)
    assert list(r.columns) == ["chrom", "start", "end", "strand", "chrom1", "start1", "end1", "strand1", "dist"]
    assert r["strand"].tolist() == ["+", "-", ".", "-"]
    assert r["strand1"].tolist() == ["-", "-", "-", "."]
    assert r["dist"].tolist() == dists


def test_neighbors_upstream_with_strand_labels_of_both_sets():
    r = pm.gintervals_neighbors_upstream(NB_QUERY.assign(strand=["+", "-", "+", "-"]),
                                         NB_TARGETS.assign(strand=["-", "+", "."]))
    assert r[["start", "strand", "start1", "strand1", "dist"]].values.tolist() == [[300, "-", 1000, "-", -600]]


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("use_intervals1_strand", [False, True])
@pytest.mark.parametrize("bad", ["1", "-1", "x"])
def test_neighbors_bad_strand_label_of_intervals1_is_an_error(bad, use_intervals1_strand, categorical):
    strand = [bad, "-", "+", "+"]
    with pytest.raises(_pymisha.error) as err:
        pm.gintervals_neighbors(NB_QUERY.assign(strand=pd.Categorical(strand) if categorical else strand), NB_TARGETS,
                                use_intervals1_strand=use_intervals1_strand, warn_ignored_strand=False)
    assert str(err.value) == BAD_LABEL.format(bad)


@pytest.mark.parametrize("bad", ["x", "1"])
def test_neighbors_bad_strand_label_of_intervals2_is_an_error(bad):
    with pytest.raises(_pymisha.error) as err:
        pm.gintervals_neighbors(NB_QUERY, NB_TARGETS.assign(strand=["+", bad, "."]))
    assert str(err.value) == BAD_LABEL.format(bad)


# --- a boolean strand column is R's "Invalid format of intervals argument" ---

BOOL_IVS = DF(dict(chrom="chr1", start=[100, 1000, 3000], end=[500, 2000, 4000]))
BOOL_STRANDS = [[True, False, True], [True, None, True], np.array([True, False, True]),
                pd.array([True, None, True], dtype="boolean"), pd.Categorical([True, False, True])]


def _strand_readers():
    iv = BOOL_IVS
    return [
        lambda b: pm.gintervals_neighbors(b, iv, warn_ignored_strand=False),
        lambda b: pm.gintervals_neighbors(b, iv, use_intervals1_strand=True),
        lambda b: pm.gintervals_neighbors(iv, b),
        lambda b: pm.gintervals_neighbors_upstream(b, iv),
        lambda b: pm.gintervals_annotate(b, iv),
        pm.gintervals_canonic,
        pm.gintervals_mark_overlaps,
        pm.gseq_extract,
        lambda b: pm.gextract("dense_track", b),
        lambda b: pm.gscreen("dense_track > 0.1", b),
        lambda b: pm.gsummary("dense_track", b),
        lambda b: pm.gquantiles("dense_track", 0.5, b),
        lambda b: pm.gdist("dense_track", [0, 0.2, 1], intervals=b),
        lambda b: pm.glookup(np.array([1.0, 2.0]), "dense_track", [0, 0.2, 1], intervals=b),
        lambda b: pm.gsample("dense_track", 5, b),
        lambda b: pm.gcor("dense_track", "dense_track", intervals=b),
        lambda b: pm.gpartition("dense_track", [0, 0.2, 1], b),
        lambda b: pm.gintervals_summary("dense_track", b),
        lambda b: pm.gintervals_quantiles("dense_track", 0.5, b),
        lambda b: pm.gsegment("dense_track", 500, intervals=b),
        lambda b: pm.gwilcox("dense_track", 100, 1000, intervals=b),
        lambda b: pm.giterator_intervals("dense_track", b),
        lambda b: pm.gintervals_mapply(np.max, "dense_track", intervals=b),
    ]


@pytest.mark.parametrize("strand", BOOL_STRANDS)
def test_boolean_strand_column_is_an_error(strand):
    b = BOOL_IVS.assign(strand=strand)
    for call in _strand_readers():
        with pytest.raises(_pymisha.error, match=r"^Invalid format of intervals argument$"):
            call(b)


def test_integer_categorical_strand_is_read_by_its_values():
    b = BOOL_IVS.assign(strand=pd.Categorical([1, -1, 1]))
    assert len(pm.gintervals_canonic(b)) == 3
    as_numbers = BOOL_IVS.assign(strand=[1, -1, 1])
    for u in (False, True):
        r = pm.gintervals_neighbors(b, NB_TARGETS, use_intervals1_strand=u, warn_ignored_strand=False)
        r_num = pm.gintervals_neighbors(as_numbers, NB_TARGETS, use_intervals1_strand=u, warn_ignored_strand=False)
        assert r["dist"].tolist() == r_num["dist"].tolist()
        assert list(r["strand"]) == [1, -1, 1]


# --- every intervals argument has its strand column checked, as R reads it ---

BAD_STRANDS = [
    ([True, False, True], "Invalid format of intervals argument", True),
    (["+", None, "-"], "Invalid strand value (NA) at index 2", True),
    (["+", "x", "-"], 'Bad strand value "x" (expected "+", "-", ".", "*" or "")', False),
]


def _no_track(name):
    if pm.gtrack_exists(name):
        pm.gtrack_rm(name, force=True)


def _make_then_remove(name, make):
    try:
        return make()
    finally:
        _no_track(name)


def _modify(b):
    pm.gtrack_create("test_followups_iv_mod", "", "dense_track")
    return pm.gtrack_modify("test_followups_iv_mod", "dense_track * 2", b)


def _vtrack(b):
    try:
        pm.gvtrack_create("test_followups_iv_v", b, "distance")
    finally:
        if "test_followups_iv_v" in pm.gvtrack_ls():
            pm.gvtrack_rm("test_followups_iv_v")


def _intervals_readers():
    iv = BOOL_IVS
    chain = DF(dict(chrom=["chr1"], start=[0], end=[10000], strand=[0], chromsrc=["chr1"], startsrc=[0],
                    endsrc=[10000], strandsrc=[0], chain_id=[1], score=[1.0]))
    return [
        lambda b: pm.gintervals_intersect(b, iv),
        lambda b: pm.gintervals_intersect(iv, b),
        lambda b: pm.gintervals_union(b, iv),
        lambda b: pm.gintervals_union(iv, b),
        lambda b: pm.gintervals_diff(b, iv),
        lambda b: pm.gintervals_diff(iv, b),
        pm.gintervals_covered_bp,
        lambda b: pm.gintervals_coverage_fraction(b, iv),
        lambda b: pm.gintervals_coverage_fraction(iv, b),
        lambda b: pm.gintervals_normalize(b, 100),
        pm.gintervals_chrom_sizes,
        lambda b: pm.gintervals_save(b, "test_followups_iv_set"),
        lambda b: pm.gintervals_liftover(b, chain),
        lambda b: _make_then_remove("test_followups_iv_sp", lambda: pm.gtrack_create_sparse(
            "test_followups_iv_sp", "", b, [1.0, 2.0, 3.0])),
        lambda b: _make_then_remove("test_followups_iv_mod", lambda: _modify(b)),
        lambda b: pm.gtrack_array_extract("array_track", None, b),
        lambda b: pm.gseq_kmer_dist(iv, 2, mask=b),
        lambda b: pm.gbins_summary("dense_track", [0, 0.2, 1], expr="dense_track", intervals=b),
        lambda b: pm.gextract("dense_track", iv, iterator=b),
        lambda b: pm.gscreen("dense_track > 0", iv, iterator=b),
        lambda b: pm.gsummary("dense_track", iv, iterator=b),
        lambda b: pm.gintervals_mapply(np.max, "dense_track", intervals=iv, iterator=b),
        lambda b: _make_then_remove("test_followups_iv_cr", lambda: pm.gtrack_create(
            "test_followups_iv_cr", "", "dense_track", iterator=b)),
    ]


@pytest.mark.parametrize("strand, message, prefixed", BAD_STRANDS)
def test_intervals_arguments_check_the_strand(strand, message, prefixed):
    b = BOOL_IVS.assign(strand=strand)
    for call in _intervals_readers():
        with pytest.raises(_pymisha.error) as err:
            call(b)
        assert str(err.value) == message
    assert not pm.gintervals_exists("test_followups_iv_set")


@pytest.mark.parametrize("strand, message, prefixed", BAD_STRANDS)
def test_cis_decay_source_and_domain_check_the_strand(strand, message, prefixed):
    b = BOOL_IVS.assign(strand=strand)
    for prefix, src, domain in (("Source intervals: ", b, BOOL_IVS), ("Domain intervals: ", BOOL_IVS, b)):
        with pytest.raises(_pymisha.error) as err:
            pm.gcis_decay("rects_track", [0, 1e5], src, domain)
        assert str(err.value) == (prefix if prefixed else "") + message


@pytest.mark.parametrize("strand, message, prefixed", BAD_STRANDS)
def test_vtrack_source_checks_the_strand(strand, message, prefixed):
    with pytest.raises(_pymisha.error) as err:
        _vtrack(BOOL_IVS.assign(strand=strand))
    assert str(err.value) == ("Source of virtual track test_followups_iv_v was not recognized neither as a track "
                              f'nor as intervals. ("{message}")')


def test_first_bad_strand_row_is_reported():
    with pytest.raises(_pymisha.error, match=r'^Bad strand value "x"'):
        pm.gextract("dense_track", BOOL_IVS.assign(strand=["x", None, "+"]))
    with pytest.raises(_pymisha.error, match=r"^Invalid strand value \(NA\) at index 1$"):
        pm.gintervals_canonic(BOOL_IVS.assign(strand=[None, "x", "+"]))


def test_valid_strands_pass_the_checks():
    unused_x = pd.Categorical(["+", "-", "+"], categories=["+", "-", "x"])
    for strand in (["+", "-", "."], ["*", "", "-"], unused_x, [1, -1, 0]):
        b = BOOL_IVS.assign(strand=strand)
        assert pm.gintervals_covered_bp(b) == 2400
        assert len(pm.gextract("dense_track", b)) > 0
        assert pm.gintervals_intersect(b, BOOL_IVS) is not None


# --- gintervals_mapply reads the strand as R: labels, and the checks of a strand column ---

@pytest.mark.parametrize("strand, expected", [
    (["+", "-", "+"], [0.18, 0.0, 0.04]),
    (pd.Categorical(["-", "+", "-"]), [0.06, 0.1, 0.12]),
    ([1, -1, 1], [0.18, 0.0, 0.04]),
])
def test_mapply_reads_strand_labels(strand, expected):
    r = pm.gintervals_mapply(lambda x: x[0], "dense_track", intervals=BOOL_IVS.assign(strand=strand))
    assert r["value"].tolist() == pytest.approx(expected, abs=1e-6)
    assert list(r["strand"]) == list(strand)


@pytest.mark.parametrize("strand, message", [
    (["+", None, "-"], NA_ERROR),
    (pd.array([1, None, -1], dtype="Int64"), NA_ERROR),
    (["+", "x", "-"], r'^Bad strand value "x"'),
])
def test_mapply_bad_strand_is_an_error(strand, message):
    with pytest.raises(_pymisha.error, match=message):
        pm.gintervals_mapply(lambda x: x[0], "dense_track", intervals=BOOL_IVS.assign(strand=strand))


# --- band: R's error when distance1 is not below distance2 (after rounding) ---

BAND_ERROR = r"^Invalid band argument: distance1 exceeds distance2$"
SCOPE_2D = DF(dict(chrom1=["chr1"], start1=[0], end1=[500000], chrom2=["chr1"], start2=[0], end2=[500000]))
SRC = DF(dict(chrom=["chr1"], start=[0], end=[500000]))


@pytest.mark.parametrize("band", [(100, 50), (5, 5), (5.6, 5.4), (5.5, 6.4), (0, 0)])
def test_band_distance1_not_below_distance2_is_an_error(band):
    calls = [
        lambda: pm.gextract("rects_track", SCOPE_2D, band=band),
        lambda: pm.gintervals_2d_band_intersect(SCOPE_2D, band=band),
        lambda: pm.giterator_intervals("rects_track", SCOPE_2D, band=band),
        lambda: pm.gsummary("rects_track", SCOPE_2D, band=band),
        lambda: pm.gdist("rects_track", [0, 1, 1000], intervals=SCOPE_2D, band=band),
        lambda: pm.gscreen("rects_track > 0", SCOPE_2D, band=band),
        lambda: pm.gcis_decay("rects_track", [0, 1e5, 1e6], SRC, SRC, SCOPE_2D, band=band),
        lambda: pm.gquantiles("rects_track", 0.5, SCOPE_2D, band=band),
    ]
    for call in calls:
        with pytest.raises(ValueError, match=BAND_ERROR):
            call()


def test_band_rounded_apart_is_valid():
    assert len(pm.gextract("rects_track", SCOPE_2D, band=(5.4, 5.6))) == 1
    assert len(pm.gintervals_2d_band_intersect(SCOPE_2D, band=(5.4, 5.6))) == 1


# --- gcis_decay: the breaks are checked as R's BinFinder checks them ---

@pytest.mark.parametrize("breaks, message", [
    ([0, 2e5, 1e5], r"Breaks are not sorted (break[1]=200000, break[2]=100000)"),
    ([0, 1e5, 1e5, 2e5], r"Breaks are not unique (break[1]=break[2]=100000)"),
    ([1e5, 0], r"Breaks are not sorted (break[0]=100000, break[1]=0)"),
    ([5, 5], r"Breaks are not unique (break[0]=break[1]=5)"),
    ([0, 1e5, 2e5, 1.5e5, 1.5e5], r"Breaks are not sorted (break[2]=200000, break[3]=150000)"),
    ([0, 1e5, 1e5, 5e4], r"Breaks are not unique (break[1]=break[2]=100000)"),
    ([0.5, 0.25], r"Breaks are not sorted (break[0]=0.5, break[1]=0.25)"),
    ([0, 1e5, np.nan, 5], r"Breaks contain NA (break[2])"),
    ([0], r"Invalid number of breaks 1"),
    ([], r"Invalid number of breaks 0"),
    ([0, np.inf, np.inf], r"Breaks are not unique (break[1]=break[2]=inf)"),
    ([1e10, 1e9], r"Breaks are not sorted (break[0]=1e+10, break[1]=1e+09)"),
    ([0.1, 0.1], r"Breaks are not unique (break[0]=break[1]=0.1)"),
])
@pytest.mark.parametrize("include_lowest", [False, True])
def test_cis_decay_invalid_breaks_are_an_error(breaks, message, include_lowest):
    with pytest.raises(ValueError) as err:
        pm.gcis_decay("rects_track", breaks, SRC, SRC, include_lowest=include_lowest)
    assert str(err.value) == message


def test_cis_decay_valid_breaks():
    res = pm.gcis_decay("rects_track", [0, 1e5, 2e5], SRC, SRC)
    assert res.tolist() == [[25, 0], [20, 0]]
    res = pm.gcis_decay("rects_track", [-np.inf, 0, np.inf], SRC, SRC)
    assert res.tolist() == [[0, 0], [76, 0]]


BIN_SCOPE = DF(dict(chrom=["chr1"], start=[0], end=[10000]))
BAD_BREAKS = [
    ([0, 0.5, 0.2], "Breaks are not sorted (break[1]=0.5, break[2]=0.2)"),
    ([0, 0.2, 0.2, 0.5], "Breaks are not unique (break[1]=break[2]=0.2)"),
    ([0.5, 0], "Breaks are not sorted (break[0]=0.5, break[1]=0)"),
    ([0.2, 0.2], "Breaks are not unique (break[0]=break[1]=0.2)"),
    ([0, 0.2, np.nan], "Breaks contain NA (break[2])"),
    ([0.2], "Invalid number of breaks 1"),
    ([], "Invalid number of breaks 0"),
    ([0, np.inf, np.inf], "Breaks are not unique (break[1]=break[2]=inf)"),
    ([1e10, 1e9], "Breaks are not sorted (break[0]=1e+10, break[1]=1e+09)"),
]


def _lookup_track(table, *args):
    try:
        return pm.gtrack_lookup("test_followups_iv_lookup", "", table, *args, iterator=50)
    finally:
        if pm.gtrack_exists("test_followups_iv_lookup"):
            pm.gtrack_rm("test_followups_iv_lookup", force=True)


@pytest.mark.parametrize("breaks, message", BAD_BREAKS)
def test_binned_functions_invalid_breaks_are_an_error(breaks, message):
    nb = max(len(breaks) - 1, 1)
    table = np.arange(1, nb + 1, dtype=float)
    table2 = np.arange(1, 2 * nb + 1, dtype=float).reshape(2, nb, order="F")
    calls = [
        lambda: pm.gdist("dense_track", breaks, intervals=BIN_SCOPE),
        lambda: pm.gdist("dense_track", [0, 0.2, 0.5], "dense_track * 2", breaks, intervals=BIN_SCOPE),
        lambda: pm.gdist("dense_track", breaks, intervals=BIN_SCOPE, include_lowest=True),
        lambda: pm.glookup(table, "dense_track", breaks, intervals=BIN_SCOPE),
        lambda: pm.glookup(table2, "dense_track", [0, 1], "dense_track", breaks, intervals=BIN_SCOPE),
        lambda: _lookup_track(table, "dense_track", breaks),
        lambda: pm.gbins_summary("dense_track", breaks, expr="dense_track", intervals=BIN_SCOPE),
        lambda: pm.gbins_quantiles("dense_track", breaks, expr="dense_track", percentiles=0.5, intervals=BIN_SCOPE),
    ]
    for call in calls:
        with pytest.raises(ValueError) as err:
            call()
        assert str(err.value) == message


# --- gintervals_mapply: gapply_intervals holds the iterator bins of the interval ---

def _mapply_bins(*exprs, intervals, **kwargs):
    """The gapply_intervals and the values each call of FUN got, and the result."""
    calls = []

    def fun(*vals, gapply_intervals):
        calls.append((gapply_intervals, [list(v) for v in vals]))
        return float(len(gapply_intervals))

    res = pm.gintervals_mapply(fun, *exprs, intervals=intervals, enable_gapply_intervals=True, **kwargs)
    return calls, res


IVS = DF(dict(chrom=["chr1", "chr1", "chr2"], start=[0, 1050, 200], end=[230, 1300, 260]))


def test_mapply_gapply_intervals_default_iterator():
    calls, res = _mapply_bins("dense_track", intervals=IVS)
    bins = [_rows(gi) for gi, _ in calls]
    assert bins == [
        [("chr1", 0, 50), ("chr1", 50, 100), ("chr1", 100, 150), ("chr1", 150, 200), ("chr1", 200, 230)],
        [("chr1", 1050, 1100), ("chr1", 1100, 1150), ("chr1", 1150, 1200), ("chr1", 1200, 1250), ("chr1", 1250, 1300)],
        [("chr2", 200, 250), ("chr2", 250, 260)],
    ]
    for gi, _ in calls:
        assert list(gi.columns) == ["chrom", "start", "end"]
        assert list(gi.index) == list(range(len(gi)))
    assert list(res["value"]) == [5, 5, 2]


def test_mapply_gapply_intervals_fixed_bin_iterator():
    calls, res = _mapply_bins("dense_track", intervals=IVS, iterator=100)
    assert [_rows(gi) for gi, _ in calls] == [
        [("chr1", 0, 100), ("chr1", 100, 200), ("chr1", 200, 230)],
        [("chr1", 1050, 1100), ("chr1", 1100, 1200), ("chr1", 1200, 1300)],
        [("chr2", 200, 260)],
    ]
    assert list(res["value"]) == [3, 3, 1]


@pytest.mark.parametrize("expr, iterator", [("dense_track", "sparse_track"), ("sparse_track", None)])
def test_mapply_gapply_intervals_sparse_iterator(expr, iterator):
    ivs = DF(dict(chrom="chr1", start=[0, 5000], end=[5000, 10000]))
    calls, res = _mapply_bins(expr, intervals=ivs, iterator=iterator)
    assert [_rows(gi) for gi, _ in calls] == [
        [("chr1", 0, 50), ("chr1", 100, 150), ("chr1", 250, 300), ("chr1", 1400, 1450), ("chr1", 4750, 4800)],
        [("chr1", 6350, 6400), ("chr1", 9900, 9950)],
    ]
    assert list(res["value"]) == [5, 2]


def test_mapply_interval_without_bins_does_not_call_fun():
    ivs = DF(dict(chrom="chr1", start=[0, 499000], end=[5000, 499010]))
    calls, res = _mapply_bins("sparse_track", intervals=ivs)
    assert len(calls) == 1
    assert res["value"].iloc[0] == 5 and np.isnan(res["value"].iloc[1])


def test_mapply_gapply_intervals_on_minus_strand_are_in_genomic_order():
    ivs = DF(dict(chrom="chr1", start=[0, 1000], end=[130, 1130], strand=[-1, 1]))
    calls, _ = _mapply_bins("dense_track", intervals=ivs)
    (gi1, vals1), (gi2, vals2) = calls
    assert _rows(gi1) == [("chr1", 0, 50), ("chr1", 50, 100), ("chr1", 100, 130)]
    assert _rows(gi2) == [("chr1", 1000, 1050), ("chr1", 1050, 1100), ("chr1", 1100, 1130)]
    # the values, not the bins, are reversed
    plus = pm.gextract("dense_track", DF(dict(chrom=["chr1"], start=[0], end=[130])))["dense_track"].tolist()
    assert vals1[0] == pytest.approx(plus[::-1])


def test_mapply_gapply_intervals_with_two_expressions():
    calls, _ = _mapply_bins("dense_track", "dense_track * 2", intervals=DF(dict(chrom=["chr1"], start=[0], end=[120])))
    (gi, vals), = calls
    assert _rows(gi) == [("chr1", 0, 50), ("chr1", 50, 100), ("chr1", 100, 120)]
    assert vals[1] == pytest.approx([2 * v for v in vals[0]])


def test_mapply_gapply_intervals_with_multitasking():
    saved = pm.CONFIG["multitasking"]
    pm.CONFIG["multitasking"] = True
    try:
        res = pm.gintervals_mapply(lambda x, gapply_intervals: len(gapply_intervals) * 1000 + gapply_intervals.start[0],
                                   "dense_track", intervals=IVS, enable_gapply_intervals=True)
    finally:
        pm.CONFIG["multitasking"] = saved
    assert list(res["value"]) == [5000, 6050, 2200]


# --- gseq_extract: a categorical strand is read by its labels ---

SEQ_BASE = DF(dict(chrom="chr1", start=[100, 200, 300], end=[110, 210, 310]))
FWD = ["cctaacccaa", "ccctaaccct", "cccaacccca"]
REV = ["ttgggttagg", "agggttaggg", "tggggttggg"]


@pytest.mark.parametrize("strand, expected", [
    (pd.Categorical(["+", "-", "."]), [FWD[0], REV[1], FWD[2]]),
    (pd.Categorical(["+", "-", "-"], categories=["-", "+"]), [FWD[0], REV[1], REV[2]]),
    (pd.Categorical(["+", "-", "+"], categories=["+", "-", "x"]), [FWD[0], REV[1], FWD[2]]),
    (pd.Categorical(["*", "", "-"]), [FWD[0], FWD[1], REV[2]]),
    (["+", "-", "."], [FWD[0], REV[1], FWD[2]]),
])
def test_seq_extract_categorical_strand(strand, expected):
    assert pm.gseq_extract(SEQ_BASE.assign(strand=strand)) == expected


def test_seq_extract_unsorted_intervals_with_categorical_strand():
    df = DF(dict(chrom=["chr2", "chr1"], start=[100, 100], end=[110, 110], strand=pd.Categorical(["-", "+"])))
    assert pm.gseq_extract(df) == ["gtggtgtggg", "cctaacccaa"]


@pytest.mark.parametrize("strand, bad", [
    (pd.Categorical(["+", "x", "-"]), "x"),
    (pd.Categorical(["+", "++", "-"]), "++"),
    (pd.Categorical(["1", "-1", "1"]), "1"),
    (["+", "x", "-"], "x"),
    (["+", "-1", "-"], "-1"),
])
def test_seq_extract_invalid_strand_label_is_an_error(strand, bad):
    with pytest.raises(_pymisha.error) as err:
        pm.gseq_extract(SEQ_BASE.assign(strand=strand))
    assert str(err.value) == f'Bad strand value "{bad}" (expected "+", "-", ".", "*" or "")'


@pytest.mark.parametrize("strand", [pd.Categorical(["+", None, "-"]), ["+", None, "-"]])
def test_seq_extract_missing_strand_label_is_an_error(strand):
    with pytest.raises(_pymisha.error, match=NA_ERROR):
        pm.gseq_extract(SEQ_BASE.assign(strand=strand))
