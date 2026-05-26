"""Parity port of R misha ``test-gsegment.R`` (1D segmentation).

``gsegment`` on a dense track (whole-genome and over a screened, modified
expression) matches R, as does the ``intervals.set.out`` round-trip.

Divergence (not asserted): R runs the ``intervals.set.out`` case under a tiny
``gmax.data.size`` to force the chunked disk-write path, which streams to disk
regardless of the limit. pymisha's ``intervals_set_out`` is still bounded by
``CONFIG['max_data_size']`` (it raises "Result size exceeded" under the small
limit), so we run it at the package default (1e9); the written result is
identical to R's baseline.
"""

from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline


def test_gsegment_fixedbin(overlay_db):
    assert_matches_baseline(
        pm.gsegment("test.fixedbin", 10000, maxpval=0.000001), "gsegment_fixedbin"
    )


def test_gsegment_fixedbin_mod(overlay_db):
    intervs = pm.gscreen("test.fixedbin > 0.2", pm.gintervals([1, 2], 0, -1))
    assert_matches_baseline(
        pm.gsegment("test.fixedbin*2", 10000, maxpval=0.000001, intervals=intervs),
        "gsegment_fixedbin_mod",
    )


def test_gsegment_sparse_data_size(overlay_db, track_namer):
    t = track_namer()
    pm.gsegment("test.sparse", 10000, maxpval=0.0001, iterator=50, intervals_set_out=t)
    assert_matches_baseline(pm.gintervals_load(t), "gsegment_sparse_data_size")
