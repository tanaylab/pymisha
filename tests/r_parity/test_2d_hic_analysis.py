"""Parity port of R misha ``test-2d-hic-analysis.R`` (real K562 Hi-C patterns).

Runs against the hg19 snapshot DB (``hg19_overlay`` fixture), which carries the
real ``hic.K562.*`` tracks plus the synthetic ``hic.test_*`` / ``domains.test`` /
``insulation.test`` fixtures the R file's ``setup_hic_test_data`` persisted.

13 cases match R: 1D insulation extract / screen, 2D ``gextract`` with a band,
``gintervals.2d`` construction (bins / all / domains), ``gintervals.2d.band_intersect``
(incl. positive/negative bands), and ``weighted.sum`` marginals (single and
multi-track, explicit 2D iterator).

The remaining cases ``xfail(strict)`` / ``skip`` on distinct gaps:

* ``GAP_2D_BAND`` -- a 2D ``band`` query near the diagonal returns a different
  contact set / coordinates than R (negative/near-diagonal bands).
* ``GAP_2D_ITER`` -- 2D iterator semantics differ: a value vtrack with no explicit
  iterator over a 2D scope, ``gvtrack.iterator.2d`` shifts, and a 1D-style 2D
  iterator span.
* ``GAP_BAND_GITER`` -- ``giterator_intervals`` has no ``band=`` argument (also
  blocks the 2D ``gintervals.neighbors`` case that builds its set that way).
* ``GAP_INSU_DOMS`` -- ``gtrack.2d.get_insu_doms`` is not implemented.
* ``GAP_R_POSTPROC`` -- the baseline is pure R post-processing
  (``cut``/``table``; or ``rpois`` with an R seed) not reproducible cross-impl.
* ``GAP_GCIS_DECAY`` -- ``gcis_decay`` over Hi-C differs (same family as the
  ``test-gcis_decay`` valued-source gap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline, assert_matches_list_baseline

GAP_2D_BAND = "2D band query near the diagonal returns a different contact set/coords than R"
GAP_2D_ITER = "2D iterator semantics differ (default 2D iterator / gvtrack.iterator.2d shifts / 2D span)"
GAP_BAND_GITER = "giterator_intervals has no band= argument"
GAP_2D_NEIGHBORS = "2D gintervals_neighbors not implemented"
GAP_INSU_DOMS = "gtrack_2d_get_insu_doms not implemented"
GAP_R_POSTPROC = "baseline is pure R post-processing (cut/table or rpois seed) not reproducible cross-impl"
GAP_GCIS_DECAY = "gcis_decay over Hi-C differs from R (valued-source gcis_decay gap)"


def _clear():
    for v in pm.gvtrack_ls():
        pm.gvtrack_rm(v)


# --------------------------------------------------------------------------- #
# Passing cases (single-DataFrame baselines)
# --------------------------------------------------------------------------- #


def test_real_k562_gextract_band(hg19_overlay):
    sc = pm.gintervals_2d("chr1", 0, 5e6, "chr1", 0, 5e6)
    r = pm.gextract("hic.K562.ela_k562", sc, band=(2e4, 1e6), colnames=["contacts"])
    assert_matches_baseline(r, "real_k562_gextract_band.hic.1")


def test_real_k562_decay(hg19_overlay):
    sc = pm.gintervals_2d("chr1", 0, 2e6, "chr1", 0, 2e6)
    breaks = [1e4, 5e4, 1e5, 5e5, 1e6]
    rows = []
    for i in range(len(breaks) - 1):
        c = pm.gextract("hic.K562.ela_k562", sc, band=(breaks[i], breaks[i + 1]), colnames=["contacts"])
        cnt = 0.0 if (c is None or len(c) == 0) else float(np.nansum(c["contacts"].to_numpy()))
        rows.append({"dist_bin": (breaks[i] + breaks[i + 1]) / 2, "count": cnt})
    assert_matches_baseline(pd.DataFrame(rows), "real_k562_decay.hic.1")


def test_real_k562_band_intersect(hg19_overlay):
    regions = pm.gintervals("chr1", [1e6, 2e6], [1.5e6, 2.5e6])
    pairs = [(i, j) for j in range(len(regions)) for i in range(len(regions))]
    iv = pm.gintervals_2d(
        [regions["chrom"].iloc[i] for i, _ in pairs], [regions["start"].iloc[i] for i, _ in pairs],
        [regions["end"].iloc[i] for i, _ in pairs],
        [regions["chrom"].iloc[j] for _, j in pairs], [regions["start"].iloc[j] for _, j in pairs],
        [regions["end"].iloc[j] for _, j in pairs],
    )
    r = pm.gintervals_2d_band_intersect(iv, band=(1e5, 1e6))
    assert_matches_baseline(r, "real_k562_band_intersect.hic.1")


def test_real_k562_insulation(hg19_overlay):
    r = pm.gextract(
        "hic.K562.ela_k562_ins_200kb", pm.gintervals("chr1", 0, 2e6),
        iterator=pm.gintervals("chr1", 0, 2e6), colnames=["insulation"],
    )
    assert_matches_baseline(r, "real_k562_insulation.hic.1")


def test_real_k562_boundaries(hg19_overlay):
    sc = pm.gintervals("chr1", 0, 5e6)
    r = pm.gscreen("hic.K562.ela_k562_ins_200kb < -2", sc, iterator=sc)
    assert_matches_baseline(r, "real_k562_boundaries.hic.1")


def test_real_k562_multi_track(hg19_overlay):
    sc = pm.gintervals_2d("chr1", 0, 1e6, "chr1", 0, 1e6)
    r = pm.gextract("hic.K562.ela_k562_score", sc, band=(2e4, 5e5), colnames=["score"])
    assert_matches_baseline(r, "real_k562_multi_track.hic.1")


def test_gintervals_2d_from_bins(hg19_overlay):
    bins = pm.gintervals("chr1", np.arange(0, 2e5 + 1, 1e4), np.arange(1e4, 2.1e5 + 1, 1e4))
    r = pm.gintervals_2d(bins["chrom"], bins["start"], bins["end"], bins["chrom"], bins["start"], bins["end"])
    assert_matches_baseline(r, "gintervals.2d.from_bins.1")


def test_gintervals_2d_all(hg19_overlay):
    # R's gintervals.2d.all() is the full N×N chrom-pair scope.
    assert_matches_baseline(pm.gintervals_2d_all(mode="full"), "gintervals.2d.all.hic.1")


def test_gintervals_2d_domains(hg19_overlay):
    d = pm.gintervals_load("domains.test")
    r = pm.gintervals_2d(d["chrom"], d["start"], d["end"], d["chrom"], d["start"], d["end"])
    assert_matches_baseline(r, "gintervals.2d.domains.1")


def test_expand_grid_2d(hg19_overlay):
    f1 = pm.gintervals("chr1", [5e5, 1e6, 1.5e6], [5.1e5, 1.1e6, 1.6e6])
    f2 = pm.gintervals("chr1", [7e5, 1.2e6, 1.8e6], [7.1e5, 1.3e6, 1.9e6])
    # expand.grid(1:n1, 1:n2): Var1 (f1 index) varies fastest.
    g = [(i, j) for j in range(len(f2)) for i in range(len(f1))]
    grid = pd.DataFrame({
        "chrom1": "chr1", "start1": [f1["start"].iloc[i] for i, _ in g], "end1": [f1["end"].iloc[i] for i, _ in g],
        "chrom2": "chr1", "start2": [f2["start"].iloc[j] for _, j in g], "end2": [f2["end"].iloc[j] for _, j in g],
    })
    assert_matches_baseline(grid, "expand.grid.2d.1")


def test_vtrack_weighted_sum(hg19_overlay):
    _clear()
    pm.gvtrack_create("v_test", "hic.test_basic", "weighted.sum")
    it = pm.gintervals_2d("chr1", 5e5, 1e6)
    assert_matches_baseline(pm.gextract("v_test", it, iterator=it), "vtrack.weighted_sum.hic.1")


def test_marginal_multi_track(hg19_overlay):
    _clear()
    tracks = ["hic.test_basic", "hic.test_shuffle"]
    for t in tracks:
        pm.gvtrack_create("v_" + t, t, "weighted.sum")
    it = pm.gintervals_2d("chr1", 5e5, 1e6)
    r = pm.gextract(["v_" + t for t in tracks], it, iterator=it, colnames=tracks)
    assert_matches_baseline(r, "marginal.multi_track.hic.1")


# --------------------------------------------------------------------------- #
# Passing cases (R named-list baselines)
# --------------------------------------------------------------------------- #


def test_gextract_band_ranges(hg19_overlay):
    iv = pm.gintervals_2d("chr1", 5e5, 1.5e6, "chr1", 5e5, 1.5e6)
    up = pm.gextract("hic.test_basic", iv, band=(-5e5, -1e4))
    down = pm.gextract("hic.test_basic", iv, band=(1e4, 5e5))
    assert_matches_list_baseline({"up": up, "down": down}, "gextract.band.ranges.hic.1")


def test_band_intersect_pos_neg(hg19_overlay):
    starts = np.arange(0, 1e6 + 1, 1e5)
    ends = starts + 1e5
    g = [(a, b) for b in range(len(starts)) for a in range(len(starts))]
    iv = pd.DataFrame({
        "chrom1": "chr1", "start1": [starts[a] for a, _ in g], "end1": [ends[a] for a, _ in g],
        "chrom2": "chr1", "start2": [starts[b] for _, b in g], "end2": [ends[b] for _, b in g],
    })
    pos = pm.gintervals_2d_band_intersect(iv, band=(2e5, 5e5))
    neg = pm.gintervals_2d_band_intersect(iv, band=(-5e5, -2e5))
    assert_matches_list_baseline({"pos": pos, "neg": neg}, "band_intersect.pos_neg.1")


# --------------------------------------------------------------------------- #
# Gaps -- xfail(strict) / skip
# --------------------------------------------------------------------------- #


def test_real_k562_vtrack_weighted_sum(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    _clear()
    pm.gvtrack_create("v_k562_score", "hic.K562.ela_k562_score", func="weighted.sum")
    sc = pm.gintervals_2d("chr1", 0, 1e6, "chr1", 0, 1e6)
    assert_matches_baseline(pm.gextract("v_k562_score", sc, band=(1e4, 5e5)), "real_k562_vtrack_weighted_sum.hic.1")


def test_gextract_band_exclude_diag(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_BAND, strict=True))
    iv = pm.gintervals_2d("chr1", 5e5, 1e6, "chr1", 5e5, 1e6)
    assert_matches_baseline(pm.gextract("hic.test_basic", iv, band=(-2e6, -1024)), "gextract.band.exclude_diag.hic.1")


def test_vtrack_band_filter(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_BAND, strict=True))
    _clear()
    pm.gvtrack_create("v_test", "hic.test_basic", "weighted.sum")
    it = pm.gintervals_2d("chr1", 5e5, 1e6)
    full = pm.gextract("v_test", it, iterator=it)
    diag = pm.gextract("v_test", it, iterator=it, band=(-1024, 1024))
    assert_matches_list_baseline({"full": full, "diag": diag}, "vtrack.band_filter.hic.1")


def test_vtrack_max_iterator2d(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    _clear()
    pm.gvtrack_create("max_score", "hic.test_score", "max")
    pm.gvtrack_iterator_2d("max_score", sshift1=-5e4, eshift1=5e4, sshift2=-5e4, eshift2=5e4)
    pts = pm.gintervals_2d("chr1", [5e5, 1e6], [5e5 + 1, 1e6 + 1], "chr1", [5e5, 1e6], [5e5 + 1, 1e6 + 1])
    assert_matches_baseline(pm.gextract("max_score", pts, iterator=pts), "vtrack.max.iterator2d.hic.1")


def test_vtrack_distance_dual(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    _clear()
    feats = pm.gintervals("chr1", [5e5, 1e6, 1.5e6], [5.1e5, 1.1e6, 1.6e6])
    pm.gvtrack_create("dist1", feats, "distance")
    pm.gvtrack_create("dist2", feats, "distance")
    pm.gvtrack_iterator("dist1", dim=1)
    pm.gvtrack_iterator("dist2", dim=2)
    sc = pm.gintervals_2d("chr1", 0, 2e6, "chr1", 0, 2e6)
    r = pm.gextract("dist1", "dist2", "hic.test_basic", sc, band=(-1e6, -1e4))
    assert_matches_baseline(r, "vtrack.distance.dual.hic.1")


def test_intra_vs_far_cis(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    _clear()
    pm.gvtrack_create("v_test", "hic.test_basic", "weighted.sum")
    d = pm.gintervals_load("domains.test")
    dom2d = pm.gintervals_2d(d["chrom"], d["start"], d["end"], d["chrom"], d["start"], d["end"])
    intra = pm.gextract("v_test", dom2d, iterator=dom2d, band=(-2e6, 0))
    it2d = pm.gintervals_2d(d["chrom"], d["start"], d["end"])
    far = pm.gextract("v_test", it2d, iterator=it2d, band=(-2e6, 0))
    assert_matches_list_baseline({"intra": intra, "far_cis": far}, "intra_vs_far_cis.hic.1")


def test_gscreen_2d_iterator2d(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    _clear()
    s1 = np.arange(5e5, 1e6 + 1, 1e5)
    s2 = np.arange(7e5, 1.2e6 + 1, 1e5)
    g = [(a, b) for b in s2 for a in s1]
    grid = pd.DataFrame({
        "chrom1": "chr1", "start1": [x[0] for x in g], "end1": [x[0] + 1e4 for x in g],
        "chrom2": "chr1", "start2": [x[1] for x in g], "end2": [x[1] + 1e4 for x in g],
    })
    pm.gvtrack_create("v_filter", "hic.test_score", "max")
    pm.gvtrack_iterator_2d("v_filter", sshift1=-5e4, eshift1=5e4, sshift2=-5e4, eshift2=5e4)
    r = pm.gscreen("v_filter > 50", grid, iterator=grid, band=(-1e6, -1e4))
    assert_matches_baseline(r, "gscreen.2d.iterator2d.1")


def test_gscreen_multi_criteria(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_ITER, strict=True))
    s = np.arange(5e5, 1e6 + 1, 5e4)
    e = s + 5e4
    grid = pm.gintervals_2d("chr1", s, e, "chr1", s, e)
    r = pm.gscreen("hic.test_score > 40 & hic.test_basic > 0", grid, iterator=grid)
    assert_matches_baseline(r, "gscreen.multi_criteria.1")


def test_marginal_diag_correct(hg19_overlay, request):
    request.node.add_marker(pytest.mark.xfail(reason=GAP_2D_BAND, strict=True))
    _clear()
    iv1 = pm.giterator_intervals(intervals=pm.gintervals("chr1", 0, 5e5), iterator=5e4)
    it2d = pm.gintervals_2d(iv1["chrom"], iv1["start"], iv1["end"])
    pm.gvtrack_create("v_marg", "hic.test_basic", "weighted.sum")
    marg = pm.gextract("v_marg", it2d, iterator=it2d, colnames=["marginal"])
    diag = pm.gextract("v_marg", it2d, iterator=it2d, band=(-1024, 1024), colnames=["diagonal"])
    assert_matches_list_baseline({"marginal": marg, "diagonal": diag}, "marginal.diag_correct.hic.1")


@pytest.mark.xfail(reason=GAP_GCIS_DECAY, strict=True)
def test_gcis_decay_basic(hg19_overlay):
    breaks = 2 ** np.arange(10, 20 + 1e-9, 0.5)
    r = pm.gcis_decay("hic.test_basic", breaks, pm.gintervals_all(), pm.gintervals_all(),
                      include_lowest=True, band=(-2e6, -1024))
    assert_matches_baseline(r, "gcis_decay.basic.hic.1")


@pytest.mark.xfail(reason=GAP_GCIS_DECAY, strict=True)
def test_gcis_decay_domains_symmetric(hg19_overlay):
    d = pm.gintervals_load("domains.test")
    breaks = 2 ** np.arange(10, 18 + 1e-9, 0.5)
    up = pm.gcis_decay("hic.test_basic", breaks, pm.gintervals_all(), d, band=(1e4, 5e5))
    down = pm.gcis_decay("hic.test_basic", breaks, pm.gintervals_all(), d, band=(-5e5, -1e4))
    combined = up.copy()
    combined.iloc[:, 0] = up.iloc[:, 0] + down.iloc[:, 0]
    combined.iloc[:, 1] = up.iloc[:, 1] + down.iloc[:, 1]
    assert_matches_baseline(combined, "gcis_decay.domains.symmetric.hic.1")


def test_giterator_intervals_band(hg19_overlay):
    pm.giterator_intervals("hic.test_basic", band=(-5e5, -1e4), intervals_set_out="test.band_set")
    try:
        assert_matches_baseline(pm.gintervals_load("test.band_set"), "giterator.intervals.band.hic.1")
    finally:
        import contextlib
        with contextlib.suppress(Exception):
            pm.gintervals_rm("test.band_set", force=True)


@pytest.mark.xfail(reason=GAP_2D_NEIGHBORS, strict=True)
def test_gintervals_neighbors_2d(hg19_overlay):
    # giterator_intervals(band=...) now works; the remaining blocker is 2D
    # gintervals_neighbors (unimplemented), not GAP_BAND_GITER.
    pm.giterator_intervals("hic.test_basic", band=(-5e5, -1e4), intervals_set_out="test.neigh_set")
    try:
        s1 = np.arange(5e5, 1e6 + 1, 1e5)
        s2 = np.arange(6e5, 1.1e6 + 1, 1e5)
        g = [(a, b) for b in s2 for a in s1]
        grid = pd.DataFrame({
            "chrom1": "chr1", "start1": [x[0] for x in g], "end1": [x[0] + 1e5 for x in g],
            "chrom2": "chr1", "start2": [x[1] for x in g], "end2": [x[1] + 1e5 for x in g],
        })
        r = pm.gintervals_neighbors("test.neigh_set", grid, mindist1=-5e4, maxdist1=5e4, mindist2=-5e4, maxdist2=5e4)
        assert_matches_baseline(r, "gintervals.neighbors.2d.hic.1")
    finally:
        import contextlib
        with contextlib.suppress(Exception):
            pm.gintervals_rm("test.neigh_set", force=True)


def test_spatial_max_grid(hg19_overlay):
    _clear()
    distance, resolution = 1e5, 5
    expansion = np.linspace(-distance, distance, resolution)
    vtracks = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v = f"max_{i}_{j}"
            pm.gvtrack_create(v, "hic.test_score", "max")
            pm.gvtrack_iterator_2d(v, sshift1=expansion[i], eshift1=expansion[i + 1],
                                   sshift2=expansion[j], eshift2=expansion[j + 1])
            vtracks.append(v)
    pts = pm.gintervals_2d("chr1", [1e6], [1e6 + 1])
    assert_matches_baseline(pm.gextract(vtracks, pts, iterator=pts), "spatial_max_grid.hic.1")


@pytest.mark.skip(reason=GAP_INSU_DOMS)
def test_get_insu_doms():
    pass


@pytest.mark.skip(reason=GAP_R_POSTPROC)
def test_contact_grid_binned():
    pass


@pytest.mark.skip(reason=GAP_R_POSTPROC)
def test_obs_exp_grid_norm():
    pass
