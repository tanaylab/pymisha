"""R misha 5.16.0 argument and result fixes (test-review-2026-11-compute.R and
test-review-2026-11-boundary.R), on this test DB (dense_track for test.fixedbin):
gseq_kmer_dist(mask=) leaves out the k-mers that reach into the mask, gintervals_liftover
warns and gives an empty result when no chromosome is in the chain, 2D annotate rejects a
tie_method, band distances must be finite and within 1e18 (and are rounded), NA distances,
percentiles, shifts, breaks and logical flags are errors, a pv table with some NaN breaks is
rejected, gintervals_mapply gives a double per interval."""

import os
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha

DF = pd.DataFrame
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "pv_percentiles")
TRACKS = ["r516_pv_nan", "r516_pv_bad", "r516_pv_nan_r", "r516_smooth_i", "r516_smooth_d", "r516_lookup_na"]


def scope():
    return pm.gintervals("chr1", 0, 20000)


def _ab():
    a = DF(dict(chrom="chr1", start=[100, 5000], end=[200, 5100]))
    b = DF(dict(chrom="chr1", start=[1000, 5300], end=[1100, 5400]))
    return a, b


def _rm_tracks():
    for t in TRACKS:
        if pm.gtrack_exists(t):
            pm.gtrack_rm(t, force=True)
    _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    yield
    pm.gvtrack_clear()


@pytest.fixture()
def no_tracks():
    _rm_tracks()
    yield
    _rm_tracks()


# ---------------------------------------------------------------------------
# test-review-2026-11-compute.R
# ---------------------------------------------------------------------------

def test_kmer_dist_mask_leaves_out_kmers_that_reach_into_the_mask():
    iv = pm.gintervals("chr1", 0, 2000)
    mask = pm.gintervals("chr1", [150, 1000], [160, 1001])
    k = 3
    s = pm.gseq_extract(iv)[0].upper()
    masked = np.zeros(len(s), bool)
    masked[150:160] = True
    masked[1000] = True
    kmers = [s[p:p + k] for p in range(len(s) - k + 1)
             if not masked[p:p + k].any() and set(s[p:p + k]) <= set("ACGT")]
    expected = pd.Series(kmers).value_counts()

    r = pm.gseq_kmer_dist(iv, k=k, mask=mask)
    assert r["count"].sum() == len(kmers) == 1983  # R 5.16.0
    got = dict(zip(r["kmer"], r["count"], strict=True))
    assert {km: got.get(km) for km in expected.index} == expected.to_dict()


def _chain_ab():
    return pm.gintervals_as_chain(DF(dict(
        chrom=["chr1"], start=[1000], end=[2000], strand=[1], chromsrc=["chrA"], startsrc=[0], endsrc=[1000],
        strandsrc=[1], chain_id=[1], score=[10])))


NO_SRC_CHROM = ("None of the intervals' chromosomes is a source chromosome of the chain "
                "(for example chrA); the result is empty")


def test_liftover_is_empty_when_no_chromosome_is_in_the_chain():
    ch = _chain_ab()
    # no input chromosome in the chain (often a naming mismatch): empty, with a warning
    with pytest.warns(UserWarning, match=NO_SRC_CHROM.replace("(", r"\(").replace(")", r"\)")):
        r = pm.gintervals_liftover(DF(dict(chrom=["chrC"], start=[5], end=[60])), ch)
    assert r.empty and list(r.columns) == ["chrom", "start", "end", "intervalID", "chain_id"]
    with pytest.warns(UserWarning, match="source chromosome of the chain"):
        r = pm.gintervals_liftover(DF(dict(chrom=["chrC"], start=[5], end=[60], v=[1.0])), ch, value_col="v")
    assert r.empty and "v" in r.columns
    # as when the chromosome is in the chain but nothing overlaps it, which does not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pm.gintervals_liftover(DF(dict(chrom=["chrA"], start=[5000], end=[6000])), ch).empty
        r = pm.gintervals_liftover(DF(dict(chrom=["chrC", "chrA"], start=[5, 500], end=[60, 600])), ch)
    assert r[["start", "end", "intervalID"]].values.tolist() == [[1500, 1600, 2]]


def test_liftover_2d_and_empty_chain_warn_as_r():
    ch = _chain_ab()
    q2 = pm.gintervals_2d("chr1", 0, 100, "chr2", 0, 100)
    with pytest.warns(UserWarning, match=r"\(for example chrA\); the result is empty"):
        r = pm.gintervals_liftover(q2, ch)
    assert r.empty and list(r.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"]
    empty = ch.iloc[:0]
    for q in (DF(dict(chrom=["chrA"], start=[0], end=[10])), q2):
        with pytest.warns(UserWarning, match=r"\(the chain is empty\); the result is empty"):
            assert pm.gintervals_liftover(q, empty).empty
    # no intervals at all: nothing to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pm.gintervals_liftover(DF(dict(chrom=pd.Series([], dtype=str), start=pd.Series([], dtype=int),
                                              end=pd.Series([], dtype=int))), ch).empty


def test_2d_annotate_with_a_tie_method_and_maxneighbors_above_1_is_an_error():
    q = pm.gintervals_2d("chr1", 100, 200, "chr1", 300, 400)
    a = pm.gintervals_2d("chr1", [500, 1000], [600, 1100], "chr1", [700, 1200], [800, 1300])
    a["lab"] = ["x", "y"]
    for tm in ("min.start", "min.end"):
        with pytest.raises(ValueError, match="1D intervals only"):
            pm.gintervals_annotate(q, a, annotation_columns="lab", maxneighbors=2, tie_method=tm)
    assert pm.gintervals_annotate(q, a, annotation_columns="lab", maxneighbors=2)["lab"].tolist() == ["x", "y"]
    # one neighbor has no ties to break (R 5.16.0)
    r = pm.gintervals_annotate(q, a, annotation_columns="lab", maxneighbors=1, tie_method="min.start")
    assert r["lab"].tolist() == ["x"] and r["dist"].tolist() == [300]


def test_annotate_takes_a_single_column_name():
    q = pm.gintervals("chr1", 100, 200)
    a = pm.gintervals("chr1", 500, 600)
    a["lab"] = ["xy"]
    r = pm.gintervals_annotate(q, a, annotation_columns="lab", column_names="L")
    assert list(r.columns) == ["chrom", "start", "end", "L", "dist"]
    assert r["L"].tolist() == ["xy"]


def _square():
    return pm.gintervals_2d("chr1", 0, 100, "chr1", 0, 100)


def _band_rect(band):
    r = pm.gintervals_2d_band_intersect(_square(), band=band)
    return r[["start1", "end1", "start2", "end2"]].values.ravel().tolist()


def test_bands_wider_than_2_31_are_accepted():
    assert _band_rect((10, 3e9)) == [10, 100, 0, 90]
    assert _band_rect((-1e10, 1e10)) == [0, 100, 0, 100]


@pytest.mark.parametrize("band", [(0, 1e19), (np.nan, 0), (-np.inf, 0), (0, np.nan), (None, 0), (pd.NA, 0),
                                  (0, np.inf), (0, 10**19)])
def test_band_distances_must_be_finite_and_within_range(band):
    with pytest.raises(ValueError, match="finite numbers"):
        pm.gintervals_2d_band_intersect(_square(), band=band)


def test_band_distances_at_the_limit_and_rounded_as_r():
    assert _band_rect((-1e18, 1e18)) == [0, 100, 0, 100]
    # R rounds a band distance half away from zero: 10.6 is 11, -10.5 is -11
    assert _band_rect((10.6, 50)) == [11, 100, 0, 89]
    assert _band_rect((-50, -10.5)) == [0, 89, 11, 100]
    with pytest.raises(ValueError, match="finite numbers"):
        pm.gextract("rects_track", pm.gintervals_2d_all(), band=(-np.inf, 0))
    assert len(pm.gextract("rects_track", pm.gintervals_2d_all(), band=(10, 3e9))) == 42  # R 5.16.0


# ---------------------------------------------------------------------------
# test-review-2026-11-boundary.R
# ---------------------------------------------------------------------------

def _lookup_bin(vals, breaks):
    """The bin glookup() puts each value in: (b1, b2], ...; force_binning puts values outside
    the breaks into the first or the last bin. NaN for a NaN value."""
    b = np.clip(np.searchsorted(breaks, vals, side="left"), 1, len(breaks) - 1).astype(float)
    b[np.isnan(vals)] = np.nan
    return b


def test_integer_arguments_are_accepted_wherever_a_number_is_expected(no_tracks):
    ext = pm.gextract("dense_track", scope())
    vals = ext["dense_track"].to_numpy(dtype=float)

    # breaks
    ivals = np.round(vals * 10)
    expr = "np.round(dense_track * 10)"
    got = pm.glookup(np.arange(1, 8), expr, np.arange(8), intervals=scope())["value"].to_numpy()
    # the table 1..7 looked up by bin is the bin
    np.testing.assert_array_equal(got, _lookup_bin(ivals, np.arange(8)))
    # gpartition: runs of touching bins that fall in the same bin of (0, 1], ..., (6, 7]
    b = np.searchsorted(np.arange(8), ivals, side="left").astype(float)
    b[(b < 1) | (b > 7) | np.isnan(ivals)] = np.nan
    runs = []
    for s, e, x in zip(ext["start"], ext["end"], b, strict=True):
        if np.isnan(x):
            continue
        if runs and runs[-1][2] == x and runs[-1][1] == s:
            runs[-1][1] = e
        else:
            runs.append([s, e, x])
    part = pm.gpartition(expr, np.arange(8), scope())
    assert part[["start", "end", "bin"]].values.tolist() == runs

    # percentiles
    assert pm.gquantiles("dense_track", 1, scope()).tolist() == pytest.approx([np.nanmax(vals)])
    assert pm.gintervals_quantiles("dense_track", 1, scope())["1"].tolist() == pytest.approx([np.nanmax(vals)])

    # distances
    a, b2 = _ab()
    assert pm.gintervals_neighbors(a, b2, 1, -1000, 1000)["dist"].tolist() == [800, 200]
    assert pm.gintervals_neighbors(a, b2, mindist=0, maxdist=500)["dist"].tolist() == [200]
    assert pm.gintervals_neighbors(a, b2, np.int64(1), np.int64(-1000), np.int64(1000))["dist"].tolist() == [800, 200]

    # a FUN that returns an integer, and an expression that does: stored as a double
    r = pm.gintervals_mapply(lambda x: np.sum(x > 0.1), "dense_track", intervals=scope())
    assert r["value"].dtype == np.float64
    assert r["value"].tolist() == [np.sum(vals > 0.1)]
    with np.errstate(invalid="ignore"):  # the expression also sees NaN padding
        r = pm.gextract("np.int64(dense_track * 10)", scope())
    assert r.iloc[:, 3].dtype == np.float64
    assert r.iloc[:, 3].tolist() == (vals * 10).astype(np.int64).astype(float).tolist()

    # window sizes and thresholds: the same result as the double form
    big = pm.gintervals("chr1")
    pd.testing.assert_frame_equal(pm.gsegment("dense_track", 10000, 0.0001, intervals=big),
                                  pm.gsegment("dense_track", 10000.0, 0.0001, intervals=big))
    pd.testing.assert_frame_equal(pm.gwilcox("dense_track", 10000, 1000, maxpval=0.01, intervals=big),
                                  pm.gwilcox("dense_track", 10000.0, 1000.0, maxpval=0.01, intervals=big))
    domain = pm.gintervals("chr1", [0, 200000], [150000, 350000])
    src = pm.gextract("sparse_track", pm.gintervals("chr1"))
    np.testing.assert_array_equal(pm.gcis_decay("rects_track", np.arange(21) * 1000, src, domain),
                                  pm.gcis_decay("rects_track", np.arange(21) * 1000.0, src, domain))

    pm.gtrack_smooth("r516_smooth_i", "", "dense_track", 1000, weight_thr=0, alg="MEAN")
    pm.gtrack_smooth("r516_smooth_d", "", "dense_track", 1000.0, weight_thr=0.0, alg="MEAN")
    _pymisha.pm_dbreload()
    assert pm.gextract("r516_smooth_i", scope()).iloc[:, 3].tolist() == \
        pm.gextract("r516_smooth_d", scope()).iloc[:, 3].tolist()


def test_invalid_scalar_arguments_are_rejected_with_a_clear_error():
    a, b = _ab()
    # NA where a TRUE/FALSE is needed
    with pytest.raises(ValueError, match="include_lowest"):
        pm.gpartition("dense_track", [0, 0.1, 1], scope(), include_lowest=np.nan)
    with pytest.raises(ValueError, match="include_lowest"):
        pm.glookup([1, 2], "dense_track", [0, 0.1, 1], intervals=scope(), include_lowest=np.nan)
    with pytest.raises(ValueError, match="include_lowest"):
        pm.gdist("dense_track", [0, 0.1, 1], intervals=scope(), include_lowest=np.nan)

    # NA, NaN or a missing value where a number is needed
    with pytest.raises(ValueError, match="range"):
        pm.gquantiles("dense_track", np.nan, scope())
    with pytest.raises(ValueError, match="maxdist"):
        pm.gintervals_neighbors(a, b, maxdist=[])
    with pytest.raises(ValueError, match="maxdist"):
        pm.gintervals_neighbors(a, b, maxdist=np.nan)
    with pytest.raises(Exception, match="iterator"):
        pm.gextract("dense_track", scope(), iterator=[])
    pm.gvtrack_create("review_vt_shift", "dense_track", "avg")
    for shift in (np.nan, np.inf, None):
        with pytest.raises(ValueError, match="sshift"):
            pm.gvtrack_iterator("review_vt_shift", sshift=shift)
    pm.gvtrack_rm("review_vt_shift")
    with pytest.raises(ValueError, match="percentile"):
        pm.gvtrack_create("review_vt_q", "dense_track", "quantile", params=np.nan)
        pm.gextract("review_vt_q", scope())

    # an infinite distance limit is no limit
    assert pm.gintervals_neighbors(a, b, maxdist=np.inf)["dist"].tolist() == [800, 200]
    assert pm.gintervals_neighbors(a, b, mindist=-np.inf, maxdist=np.inf)["dist"].tolist() == [800, 200]


def test_na_distance_limits_and_maxneighbors_as_r():
    a, b = _ab()
    for kw, msg in [(dict(maxdist=np.nan), "maxdist argument is NA"), (dict(mindist=np.nan), "mindist argument is NA"),
                    (dict(maxdist=pd.NA), "maxdist argument is NA"), (dict(maxdist=None), "maxdist argument is NA"),
                    (dict(maxdist=np.array([])), "maxdist argument is not numeric"),
                    (dict(maxneighbors=np.inf), "maxneighbors must be >= 1"),
                    (dict(maxneighbors=np.nan), "maxneighbors must be >= 1")]:
        with pytest.raises(ValueError, match=msg):
            pm.gintervals_neighbors(a, b, **kw)
    assert pm.gintervals_neighbors(a, b, mindist=-np.inf)["dist"].tolist() == [800, 200]
    q = pm.gintervals_2d("chr1", 0, 100, "chr1", 0, 100)
    t = pm.gintervals_2d("chr1", 1000, 1100, "chr1", 1000, 1100)
    for k in ("mindist1", "maxdist1", "mindist2", "maxdist2"):
        with pytest.raises(ValueError, match=f"{k} argument is NA"):
            pm.gintervals_neighbors(q, t, **{k: np.nan})
    assert len(pm.gintervals_neighbors(q, t, mindist1=-np.inf, maxdist1=np.inf)) == 1


@pytest.mark.parametrize("pct", [np.nan, None, pd.NA, [0.5, np.nan]])
def test_na_percentiles_are_out_of_range(pct):
    for f in (lambda: pm.gquantiles("dense_track", pct, scope()),
              lambda: pm.gintervals_quantiles("dense_track", pct, scope()),
              lambda: pm.gbins_quantiles("dense_track", [0, 0.1, 1], expr="dense_track", percentiles=pct,
                                         intervals=scope())):
        with pytest.raises(ValueError, match=r"Percentile \(nan\) is not in \[0, 1\] range"):
            f()


def test_vtrack_shifts_must_be_finite_numbers():
    pm.gvtrack_create("review_vt_shift", "dense_track", "avg")
    for k in ("sshift", "eshift"):
        for v in (np.nan, np.inf, -np.inf, None, pd.NA):
            with pytest.raises(ValueError, match=f"Virtual track review_vt_shift: {k} must be an integer"):
                pm.gvtrack_iterator("review_vt_shift", **{k: v})
    # a fraction is truncated, as R does
    pm.gvtrack_iterator("review_vt_shift", sshift=-100.7, eshift=100.7)
    frac = pm.gextract("review_vt_shift", scope())["review_vt_shift"].tolist()
    pm.gvtrack_iterator("review_vt_shift", sshift=-100, eshift=100)
    assert pm.gextract("review_vt_shift", scope())["review_vt_shift"].tolist() == frac
    pm.gvtrack_create("review_vt_2d", "rects_track", "area")
    with pytest.raises(ValueError, match="Virtual track review_vt_2d: eshift2 must be an integer"):
        pm.gvtrack_iterator_2d("review_vt_2d", eshift2=np.nan)


def test_logical_flags_are_true_or_false_not_na(no_tracks):
    a, b = _ab()
    s = scope()
    domain = pm.gintervals("chr1", [0, 200000], [150000, 350000])
    src = pm.gextract("sparse_track", pm.gintervals("chr1"))
    for na in (np.nan, pd.NA):
        for name, f in [
            ("force_binning", lambda: pm.glookup([1, 2], "dense_track", [0, 0.1, 1], intervals=s, force_binning=na)),
            ("na_if_notfound", lambda: pm.gintervals_neighbors(a, b, na_if_notfound=na)),
            ("use_intervals1_strand", lambda: pm.gintervals_neighbors(a, b, use_intervals1_strand=na)),
            ("nans", lambda: pm.gtrack_smooth("r516_smooth_i", "", "dense_track", 1000, smooth_nans=na)),
            ("onetailed", lambda: pm.gsegment("dense_track", 10000, 0.0001, onetailed=na, intervals=s)),
            ("onetailed", lambda: pm.gwilcox("dense_track", 10000, 1000, onetailed=na, intervals=s)),
            ("include_lowest", lambda: pm.gbins_summary("dense_track", [0, 0.1, 1], expr="dense_track",
                                                        intervals=s, include_lowest=na)),
            ("include_lowest", lambda: pm.gbins_quantiles("dense_track", [0, 0.1, 1], expr="dense_track",
                                                          intervals=s, include_lowest=na)),
            ("include_lowest", lambda: pm.gtrack_lookup("r516_lookup_na", "", [1, 2], "dense_track", [0, 0.1, 1],
                                                        include_lowest=na)),
            ("force_binning", lambda: pm.gtrack_lookup("r516_lookup_na", "", [1, 2], "dense_track", [0, 0.1, 1],
                                                       force_binning=na)),
            ("include_lowest", lambda: pm.gcis_decay("rects_track", [0, 1000, 2000], src, domain,
                                                     include_lowest=na)),
        ]:
            with pytest.raises(ValueError, match=f"{name} argument is not logical"):
                f()
    assert not pm.gtrack_exists("r516_smooth_i") and not pm.gtrack_exists("r516_lookup_na")
    # True/False, numpy's too, and None still work
    assert len(pm.gpartition("dense_track", [0, 0.1, 1], s, include_lowest=np.True_)) == 108
    assert len(pm.gpartition("dense_track", [0, 0.1, 1], s, include_lowest=None)) == \
        len(pm.gpartition("dense_track", [0, 0.1, 1], s, include_lowest=False))


def test_na_breaks_are_rejected():
    s = scope()
    with pytest.raises(ValueError, match="NA"):
        pm.gdist("dense_track", [0, np.nan, 0.2, 1], intervals=s)
    with pytest.raises(ValueError, match="NA"):
        pm.gdist("dense_track", pd.array([0, pd.NA, 2], dtype="Int64"), intervals=s)
    with pytest.raises(ValueError, match="NA"):
        pm.gpartition("dense_track", [0, np.nan, 1], s)


def test_na_breaks_give_r_message_everywhere(no_tracks):
    s = scope()
    msg = r"Breaks contain NA \(break\[1\]\)"
    pm.gvtrack_create("review_vt_avg", "dense_track", "avg")
    domain = pm.gintervals("chr1", [0, 200000], [150000, 350000])
    src = pm.gextract("sparse_track", pm.gintervals("chr1"))
    br = [0, np.nan, 1]
    for f in (lambda: pm.gpartition("dense_track", [0, None, 1], s),
              lambda: pm.gpartition("review_vt_avg", br, s),
              lambda: pm.gpartition("rects_track", [0, np.nan, 1000], pm.gintervals_2d("chr1", 0, 500000, "chr1", 0,
                                                                                          500000)),
              lambda: pm.glookup([1, 2], "dense_track", br, intervals=s),
              lambda: pm.glookup([1, 2], "review_vt_avg", br, intervals=s),
              lambda: pm.gbins_summary("dense_track", br, expr="dense_track", intervals=s),
              lambda: pm.gbins_quantiles("dense_track", br, expr="dense_track", percentiles=0.5, intervals=s),
              lambda: pm.gtrack_lookup("r516_lookup_na", "", [1, 2], "dense_track", br),
              lambda: pm.gcis_decay("rects_track", [0, np.nan, 2000], src, domain)):
        with pytest.raises(ValueError, match=msg):
            f()
    assert not pm.gtrack_exists("r516_lookup_na")
    # infinite breaks are fine
    assert pm.gdist("dense_track", [-np.inf, 0.1, np.inf], intervals=s).tolist() == [307, 93]  # R 5.16.0


def test_global_percentile_on_a_track_with_no_values_gives_nan(no_tracks):
    iv = pm.gintervals("chr1", 0, 100000)
    pm.gtrack_create_dense("r516_pv_nan", "", iv, [np.nan], 10000, np.nan)
    _pymisha.pm_dbreload()
    pm.gvtrack_create("review_pv_nan", "r516_pv_nan", "global.percentile")
    assert np.isnan(pm.gextract("review_pv_nan", iv, iterator=10000)["review_pv_nan"]).all()


def _with_r_pv_table(track, fixture):
    """The track's pv.percentiles table as R 5.16.0 wrote it (tests/fixtures/pv_percentiles:
    nan_break is the table test-review-2026-11-boundary.R sets with .gtrack.var.set, no_values
    the one R computes for a track with no values)."""
    vars_dir = os.path.join(_pymisha.pm_track_path(track), "vars")
    os.makedirs(vars_dir, exist_ok=True)
    shutil.copyfile(os.path.join(FIXTURES, fixture), os.path.join(vars_dir, "pv.percentiles"))
    _pymisha.pm_dbreload()


def test_a_pv_table_with_some_nan_breaks_is_rejected(no_tracks):
    iv = pm.gintervals("chr1", 0, 100000)
    pm.gtrack_create_dense("r516_pv_bad", "", iv, [5.0], 10000, np.nan)
    _pymisha.pm_dbreload()
    _with_r_pv_table("r516_pv_bad", "nan_break")
    for func in ("global.percentile", "global.percentile.min", "global.percentile.max"):
        pm.gvtrack_create("review_pv_bad", "r516_pv_bad", func)
        with pytest.raises(ValueError, match=r"Breaks contain NA \(break\[1\]\)"):
            pm.gextract("review_pv_bad", iv, iterator=10000)
    # only the no-values table (all breaks NaN) gives NaN
    pm.gtrack_create_dense("r516_pv_nan_r", "", iv, [np.nan], 10000, np.nan)
    _pymisha.pm_dbreload()
    _with_r_pv_table("r516_pv_nan_r", "no_values")
    pm.gvtrack_create("review_pv_nan_r", "r516_pv_nan_r", "global.percentile")
    assert np.isnan(pm.gextract("review_pv_nan_r", iv, iterator=10000)["review_pv_nan_r"]).all()


def test_mapply_gives_one_double_per_interval_as_r():
    iv = DF(dict(chrom=["chr1"] * 3, start=[5000, 0, 250], end=[6000, 1000, 1000], lab=["a", "b", "c"],
                 strand=[1, 1, -1]))
    # one row per interval, in its order and with its columns; FUN gets its bins' values
    r = pm.gintervals_mapply(len, "dense_track", intervals=iv, iterator=500)
    assert list(r.columns) == ["chrom", "start", "end", "lab", "strand", "value"]
    assert r["value"].tolist() == [2.0, 2.0, 2.0]
    two = pm.gintervals("chr1", [0, 5000], [1000, 6000])
    assert pm.gintervals_mapply(len, "dense_track", intervals=two, iterator=50)["value"].tolist() == [20, 20]
    # reversed on strand -1 (R 5.16.0 values)
    first = pm.gintervals_mapply(lambda x: x[0], "dense_track", intervals=iv, iterator=500)["value"]
    assert first.tolist() == pytest.approx([0.046, 0.1577778, 0.022], abs=1e-6)
    # all the expressions on the same bins
    ov = pm.gintervals("chr1", [0, 500], [1000, 1500])
    assert pm.gintervals_mapply(lambda x, y: len(x) + len(y), "dense_track", "sparse_track", intervals=ov,
                                iterator=100)["value"].tolist() == [20, 20]
    # an interval with no iterator bin is NaN and FUN is not called
    calls = []
    r = pm.gintervals_mapply(lambda x: calls.append(1) or len(x), "sparse_track",
                             intervals=pm.gintervals("chr1", [0, 400], [300, 450]))
    assert r["value"].tolist()[0] == 3 and np.isnan(r["value"].tolist()[1]) and len(calls) == 1


@pytest.mark.parametrize("ret", [True, np.True_, None, "a", [1, 2]])
def test_mapply_fun_must_give_a_single_number(ret):
    with pytest.raises(ValueError, match="does not produce a single numeric value"):
        pm.gintervals_mapply(lambda x: ret, "dense_track", intervals=pm.gintervals("chr1", 0, 1000))
