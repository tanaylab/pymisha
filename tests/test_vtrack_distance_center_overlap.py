"""distance.center with overlapping input (query) regions.

Ported from R misha tests/testthat/test-vtrack-distance-center-overlap.R.

Overlapping query regions drive backward access in the iterator; the result of
extracting all regions at once must match extracting them one at a time.
"""
from __future__ import annotations

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _clean_vtracks():
    for v in list(pm.gvtrack_ls()):
        pm.gvtrack_rm(v)
    yield
    for v in list(pm.gvtrack_ls()):
        pm.gvtrack_rm(v)


def _concat(frames):
    return pd.concat(frames, ignore_index=True)


def test_distance_center_overlapping_input():
    src = _concat(
        [
            pm.gintervals("1", 100, 300),  # center 200
            pm.gintervals("1", 500, 700),  # center 600
            pm.gintervals("1", 1000, 1200),  # center 1100
        ]
    )
    pm.gvtrack_create("dc_overlap", src, "distance.center")

    regions = _concat(
        [
            pm.gintervals("1", 50, 350),
            pm.gintervals("1", 200, 750),
            pm.gintervals("1", 600, 1250),
        ]
    )
    res_overlap = pm.gextract("dc_overlap", regions, iterator=20)
    res_individual = _concat(
        [
            pm.gextract("dc_overlap", regions.iloc[[i]], iterator=20)
            for i in range(len(regions))
        ]
    )
    merged = res_overlap.merge(
        res_individual,
        on=["chrom", "start", "end"],
        suffixes=(".overlap", ".individual"),
    )
    assert merged["dc_overlap.overlap"].equals(merged["dc_overlap.individual"])


def test_distance_center_heavily_overlapping_input():
    src = _concat(
        [
            pm.gintervals("1", 0, 200),
            pm.gintervals("1", 400, 600),
            pm.gintervals("1", 800, 1000),
            pm.gintervals("1", 1200, 1400),
            pm.gintervals("1", 1800, 2000),
        ]
    )
    pm.gvtrack_create("dc_heavy", src, "distance.center")

    regions = _concat(
        [
            pm.gintervals("1", 0, 500),
            pm.gintervals("1", 100, 700),
            pm.gintervals("1", 300, 900),
            pm.gintervals("1", 500, 1100),
            pm.gintervals("1", 700, 1500),
            pm.gintervals("1", 1000, 2000),
        ]
    )
    res_overlap = pm.gextract("dc_heavy", regions, iterator=20)
    res_individual = _concat(
        [
            pm.gextract("dc_heavy", regions.iloc[[i]], iterator=20)
            for i in range(len(regions))
        ]
    )
    merged = res_overlap.merge(
        res_individual,
        on=["chrom", "start", "end"],
        suffixes=(".overlap", ".individual"),
    )
    assert merged["dc_heavy.overlap"].equals(merged["dc_heavy.individual"])


def test_distance_center_overlapping_input_multi_chrom():
    src = _concat(
        [
            pm.gintervals("1", 100, 300),
            pm.gintervals("1", 500, 700),
            pm.gintervals("1", 1000, 1200),
            pm.gintervals("2", 200, 400),
            pm.gintervals("2", 800, 1000),
        ]
    )
    pm.gvtrack_create("dc_mc", src, "distance.center")

    regions = _concat(
        [
            pm.gintervals("1", 50, 350),
            pm.gintervals("1", 200, 750),
            pm.gintervals("1", 600, 1250),
            pm.gintervals("2", 100, 500),
            pm.gintervals("2", 300, 1050),
        ]
    )
    res_overlap = pm.gextract("dc_mc", regions, iterator=20)
    res_individual = _concat(
        [
            pm.gextract("dc_mc", regions.iloc[[i]], iterator=20)
            for i in range(len(regions))
        ]
    )
    merged = res_overlap.merge(
        res_individual,
        on=["chrom", "start", "end"],
        suffixes=(".overlap", ".individual"),
    )
    assert merged["dc_mc.overlap"].equals(merged["dc_mc.individual"])
