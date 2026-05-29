"""Regression for the greedy local-minimum scan bug in the distance vtrack family.

Ported from R misha tests/testthat/test-vtrack-distance-nearest-trap.R (5.9.1).

distance / distance.edge / distance.center used a candidate-limited scan that,
with overlapping / nested / widely-varying source intervals, returned a
non-nearest interval - in the extreme a nonzero distance for a query that
overlaps the source, where gintervals_neighbors correctly returns 0. The fix
routes all three through the same SegmentFinder/NNIterator nearest-neighbor
index that gintervals_neighbors uses.
"""
from __future__ import annotations

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _clean_vtracks():
    for v in list(pm.gvtrack_ls()):
        pm.gvtrack_rm(v)
    yield
    for v in list(pm.gvtrack_ls()):
        pm.gvtrack_rm(v)


# Trap geometry (single chromosome):
#   C  = [800, 900]    upstream of query
#   C2 = [850, 870]    nested inside C (sources overlap each other)
#   D  = [950, 1500]   OVERLAPS the query -> true nearest, distance 0
#   E  = [1250, 1300]  downstream, placed to also trap the end-sorted scan
# query = [1000, 1100], center 1050, sits inside D.
def _trap_src():
    return pm.gintervals("1", [800, 850, 950, 1250], [900, 870, 1500, 1300])


_TRAP_QUERY = None


def _trap_query():
    return pm.gintervals("1", 1000, 1100)


def test_distance_edge_zero_for_overlapping_query():
    pm.gvtrack_create("d_edge_trap", _trap_src(), "distance.edge")
    res = pm.gextract("d_edge_trap", _trap_query(), iterator=_trap_query())
    nb = pm.gintervals_neighbors(_trap_query(), _trap_src())
    assert res["d_edge_trap"].iloc[0] == 0
    assert res["d_edge_trap"].iloc[0] == nb["dist"].iloc[0]


def test_distance_zero_when_center_inside_overlapping_source():
    pm.gvtrack_create("d_plain_trap", _trap_src(), "distance")
    res = pm.gextract("d_plain_trap", _trap_query(), iterator=_trap_query())
    assert res["d_plain_trap"].iloc[0] == 0


def test_distance_edge_matches_neighbors_on_dense_overlapping_source():
    rng = np.random.default_rng(60427)
    n_src = 2000
    s = rng.integers(1, 490_000, n_src)
    # mix of narrow and very wide intervals -> heavy overlap / nesting
    narrow = rng.integers(10, 51, n_src)
    wide = rng.integers(500, 2001, n_src)
    w = np.where(rng.random(n_src) < 0.5, narrow, wide)
    src = pm.gintervals("1", s.tolist(), (s + w).tolist())
    pm.gvtrack_create("d_edge_stress", src, "distance.edge")

    n_q = 3000
    qs = rng.integers(1, 490_000, n_q)
    query = pm.gintervals("1", qs.tolist(), (qs + 30).tolist())
    query = pm.gintervals_canonic(query)

    res = pm.gextract("d_edge_stress", query, iterator=query).sort_values(
        "intervalID"
    )
    nb = pm.gintervals_neighbors(query, src)
    merged = res.merge(
        nb[["chrom", "start", "end", "dist"]], on=["chrom", "start", "end"], how="left"
    )
    assert (merged["d_edge_stress"].values == merged["dist"].values).all()


def test_distance_center_accepts_overlapping_source():
    # Previously raised "overlapping intervals ... not allowed for this function".
    pm.gvtrack_create("dc_overlap_src", _trap_src(), "distance.center")


def test_distance_center_distance_to_containing_center():
    pm.gvtrack_create("dc_single", _trap_src(), "distance.center")
    res = pm.gextract("dc_single", _trap_query(), iterator=_trap_query())
    # center 1050 inside D=[950,1500] only; D center 1225 -> |1050-1225| = 175.
    assert res["dc_single"].iloc[0] == 175


def test_distance_center_nearest_among_overlapping_containing():
    # C=[800,900] center 850, C2=[850,870] center 860 (C2 nested in C).
    src = pm.gintervals("1", [800, 850], [900, 870])
    pm.gvtrack_create("dc_multi", src, "distance.center")
    # query center 860 inside both C and C2; nearest center is C2 (860) -> 0.
    query = pm.gintervals("1", 855, 865)
    res = pm.gextract("dc_multi", query, iterator=query)
    assert res["dc_multi"].iloc[0] == 0
