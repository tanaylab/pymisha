"""gextract / giterator_intervals: iterator given as a track or intervals-set name.

R misha lets the iterator be a track name (dense -> its bins, sparse -> its
intervals) or an intervals-set name, in addition to a numeric bin size or an
explicit intervals DataFrame. These tests pin that parity on the small test DB.
The oracle is the already-supported equivalent form (numeric bin size or
intervals DataFrame).
"""

import pandas as pd

import pymisha as pm


def _sorted(df):
    cols = [c for c in ("chrom", "start", "end") if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True)


def test_iterator_dense_track_name_equals_binsize():
    scope = pm.gintervals(["1", "2"])
    by_name = pm.gextract("dense_track", scope, iterator="dense_track")
    by_size = pm.gextract("dense_track", scope, iterator=50)  # dense_track bin=50
    assert by_name is not None and by_size is not None
    by_name = _sorted(by_name)
    by_size = _sorted(by_size)
    assert len(by_name) == len(by_size)
    assert list(by_name["start"]) == list(by_size["start"])
    assert list(by_name["end"]) == list(by_size["end"])
    pd.testing.assert_series_equal(
        by_name["dense_track"], by_size["dense_track"], check_names=False
    )


def test_iterator_sparse_track_name_equals_intervals_df():
    scope = pm.gintervals(["1", "2"])
    # Oracle: the sparse track's own intervals as an explicit DataFrame iterator.
    sparse_iv = pm.gextract("sparse_track", scope)[["chrom", "start", "end"]]
    by_name = pm.gextract("dense_track", scope, iterator="sparse_track")
    by_df = pm.gextract("dense_track", scope, iterator=sparse_iv)
    assert by_name is not None and by_df is not None
    assert len(by_name) == len(by_df)
    bn, bd = _sorted(by_name), _sorted(by_df)
    assert list(bn["start"]) == list(bd["start"])
    assert list(bn["end"]) == list(bd["end"])


def test_iterator_intervals_set_name():
    # An intervals-set name as iterator must behave like its loaded DataFrame.
    scope = pm.gintervals(["1", "2"])
    iset = pm.gintervals_load("annotations")
    by_name = pm.gextract("dense_track", scope, iterator="annotations")
    by_df = pm.gextract("dense_track", scope, iterator=iset)
    assert (by_name is None) == (by_df is None)
    if by_name is not None:
        assert len(_sorted(by_name)) == len(_sorted(by_df))


def test_giterator_intervals_sparse_track():
    # giterator_intervals on a sparse track must return that track's intervals.
    scope = pm.gintervals(["1", "2"])
    grid = pm.giterator_intervals("sparse_track", scope)
    oracle = pm.gextract("sparse_track", scope)[["chrom", "start", "end"]]
    assert grid is not None
    g, o = _sorted(grid), _sorted(oracle)
    assert len(g) == len(o)
    assert list(g["start"]) == list(o["start"])
    assert list(g["end"]) == list(o["end"])


def test_giterator_intervals_dense_track_still_works():
    scope = pm.gintervals(["1"])
    grid = pm.giterator_intervals("dense_track", scope)
    by_size = pm.giterator_intervals(intervals=scope, iterator=50)
    assert grid is not None and by_size is not None
    assert len(_sorted(grid)) == len(_sorted(by_size))
