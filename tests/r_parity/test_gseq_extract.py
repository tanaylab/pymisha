"""Parity port of R misha ``test-gseq.extract.R`` (sequence extraction).

``gseq.extract`` over ``gscreen`` results -- forward and reverse strand --
matches R exactly. 2D intervals correctly raise.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline


def _screen():
    return pm.gscreen("test.fixedbin > 0.6", pm.gintervals([1, 2, 3]))


def test_gseq_extract_gscreen_fixedbin():
    assert_matches_baseline(pm.gseq_extract(_screen()), "gseq_extract_gscreen_fixedbin")


def test_gseq_extract_gscreen_fixedbin_modified_strand():
    intervs = _screen()
    intervs["strand"] = -1
    assert_matches_baseline(pm.gseq_extract(intervs), "gseq_extract_gscreen_fixedbin_modified_strand")


def test_gseq_extract_2d_errors():
    """R: ``expect_error(gseq.extract(gintervals.2d(...)))``."""
    with pytest.raises(Exception):
        pm.gseq_extract(pm.gintervals_2d([1], 10, 100, [2], 20, 300))
