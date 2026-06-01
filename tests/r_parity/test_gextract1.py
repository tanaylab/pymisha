"""Parity port of R misha ``test-gextract1.R``.

Each ``expect_regression(<expr>, "<id>")`` is run on R's own test DB and compared
to R's frozen ``.rds`` baseline via :func:`assert_matches_baseline`.

Feature gaps in pymisha (array-track extract, COMPUTED 2D Hi-C tracks, 2D
gscreen on rects) are marked ``xfail(strict=True)`` so they flip to failures -
and alert us to re-port - the moment the feature lands.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

# R: intervs <- gscreen("test.fixedbin > 0.2", gintervals(c(1, 2)))
# Shared scope used by the fixedbin/sparse/array extract tests.


def _screen_intervs():
    return pm.gscreen("test.fixedbin > 0.2", pm.gintervals([1, 2]))


def test_gextract_fixedbin():
    assert_matches_baseline(
        pm.gextract("test.fixedbin", _screen_intervs()), "gextract.fixedbin"
    )


def test_gextract_sparse():
    assert_matches_baseline(
        pm.gextract("test.sparse", _screen_intervs()), "gextract.sparse"
    )


def test_gextract_array():
    assert_matches_baseline(
        pm.gextract("test.array", _screen_intervs()), "gextract.array"
    )


def test_gextract_computed2d_1():
    # 2D gscreen on a rects track + COMPUTED Hi-C extract now match R (the
    # COMPUTED read path shipped in v0.8.4/v0.8.5; gscreen-on-rects works).
    intervs = pm.gscreen("test.rects > 9")
    assert_matches_baseline(
        pm.gextract("test.computed2d", intervs), "gextract.computed2d.1"
    )


@pytest.mark.xfail(reason="COMPUTED Hi-C tracks not supported", strict=True)
def test_gextract_computed2d_2():
    intervs = pm.gscreen("test.computed2d > 9000000")
    assert_matches_baseline(
        pm.gextract("test.computed2d", intervs), "gextract.computed2d.2"
    )


# R: test_that("gextract with .misha$ALLGENOME works", { withr gmax.data.size=1e9 ... })
# .misha$ALLGENOME -> gintervals_all() (1D) / gintervals_2d_all(mode="full") (2D).
# max_data_size is bumped to 1e9 by the r_testdb package fixture.


def test_gextract_allgenome_fixedbin():
    assert_matches_baseline(
        pm.gextract("test.fixedbin", pm.gintervals_all()), "gextract.allgenome.fixedbin"
    )


def test_gextract_allgenome_sparse():
    assert_matches_baseline(
        pm.gextract("test.sparse", pm.gintervals_all()), "gextract.allgenome.sparse"
    )


def test_gextract_allgenome_array():
    assert_matches_baseline(
        pm.gextract("test.array", pm.gintervals_all()), "gextract.allgenome.array"
    )


def test_gextract_allgenome_rects():
    assert_matches_baseline(
        pm.gextract("test.rects", pm.gintervals_2d_all(mode="full")),
        "gextract.allgenome.rects",
    )


@pytest.mark.xfail(reason="COMPUTED Hi-C tracks not supported", strict=True)
def test_gextract_allgenome_computed2d():
    assert_matches_baseline(
        pm.gextract("test.computed2d", pm.gintervals_2d_all(mode="full")),
        "gextract.allgenome.computed2d",
    )
