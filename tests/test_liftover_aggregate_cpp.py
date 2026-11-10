"""pm_liftover_aggregate: the sparse target of gtrack_liftover, as R's GTrackLiftover.cpp.

Per chromosome the intervals are cut at their boundaries, and each segment some
interval covers gets R's aggregate_values over the intervals covering it, as
gintervals_liftover(value_col=) does (R misha 44ecd8d5): intervals sharing a
key count once, and a NaN result is kept. Gaps give no segment, and equal
neighbouring segments are not merged. The end-to-end comparison with R is in
test_liftover_1d_r_parity.py.
"""

import _pymisha
import numpy as np
import pandas as pd
import pytest

from pymisha.liftover import _TRACK_LIFTOVER_AGGS, _aggregate_overlapping


def _call(chrom, start, end, value, agg_type="mean", na_rm=True, min_n=-1, nth_index=0, key=None):
    df_dict = {
        "chrom": np.array(chrom, dtype=object),
        "start": np.array(start, dtype=np.int64),
        "end": np.array(end, dtype=np.int64),
        "value": np.array(value, dtype=np.float64),
    }
    if key is not None:
        df_dict["key"] = np.array(key, dtype=np.int64)
    res = _pymisha.pm_liftover_aggregate(df_dict, agg_type, na_rm, int(min_n), int(nth_index))
    return list(zip(res["chrom"], res["start"].tolist(), res["end"].tolist(), res["value"].tolist(), strict=True))


def _same(got, expected):
    assert len(got) == len(expected), (got, expected)
    for g, e in zip(got, expected, strict=True):
        assert g[:3] == e[:3], (got, expected)
        assert (np.isnan(g[3]) and np.isnan(e[3])) or g[3] == e[3], (got, expected)


def test_single_interval_passthrough():
    _same(_call(["chr1"], [100], [200], [5.0]), [("chr1", 100, 200, 5.0)])


@pytest.mark.parametrize("agg, value", [
    ("mean", 3.0), ("sum", 6.0), ("min", 2.0), ("max", 4.0), ("count", 2.0), ("median", 3.0),
])
def test_overlap_is_cut_into_segments(agg, value):
    # [100,200)=2 and [150,250)=4 overlap on [150,200) only
    one = 1.0 if agg == "count" else None
    _same(_call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type=agg),
          [("chr1", 100, 150, one or 2.0), ("chr1", 150, 200, value), ("chr1", 200, 250, one or 4.0)])


def test_contained_interval():
    _same(_call(["chr1", "chr1"], [100, 200], [500, 300], [2.0, 10.0]),
          [("chr1", 100, 200, 2.0), ("chr1", 200, 300, 6.0), ("chr1", 300, 500, 2.0)])


def test_touching_intervals_and_gaps():
    # [100,200) and [200,300) touch, so they are two segments; the second averages 2 and 3.
    # Nothing covers [300,400), and the equal segments around it stay apart.
    _same(_call(["chr1"] * 5, [100, 200, 200, 250, 400], [200, 300, 300, 300, 500], [1.0, 2.0, 3.0, 2.5, 2.5],
                agg_type="median"),
          [("chr1", 100, 200, 1.0), ("chr1", 200, 250, 2.5), ("chr1", 250, 300, 2.5), ("chr1", 400, 500, 2.5)])


def test_nth():
    # within a segment the pieces are in R's order: the one that starts first first
    _same(_call(["chr1", "chr1"], [100, 150], [200, 250], [10.0, 20.0], agg_type="nth", nth_index=2),
          [("chr1", 100, 150, np.nan), ("chr1", 150, 200, 20.0), ("chr1", 200, 250, np.nan)])
    _same(_call(["chr1", "chr1"], [100, 150], [200, 250], [20.0, 10.0], agg_type="nth", nth_index=2),
          [("chr1", 100, 150, np.nan), ("chr1", 150, 200, 10.0), ("chr1", 200, 250, np.nan)])


def test_nan_is_kept():
    # a segment of NaN values alone is a NaN interval
    nan = float("nan")
    args = (["chr1"] * 3, [100, 150, 400], [200, 250, 500], [2.0, nan, nan])
    _same(_call(*args),
          [("chr1", 100, 150, 2.0), ("chr1", 150, 200, 2.0), ("chr1", 200, 250, np.nan), ("chr1", 400, 500, np.nan)])
    _same(_call(*args, agg_type="count"),
          [("chr1", 100, 150, 1.0), ("chr1", 150, 200, 1.0), ("chr1", 200, 250, 0.0), ("chr1", 400, 500, 0.0)])
    _same(_call(*args, na_rm=False),
          [("chr1", 100, 150, 2.0), ("chr1", 150, 200, np.nan), ("chr1", 200, 250, np.nan), ("chr1", 400, 500, np.nan)])


def test_min_n():
    _same(_call(["chr1"] * 3, [100, 150, 400], [200, 250, 500], [2.0, 4.0, 1.0], min_n=2),
          [("chr1", 100, 150, np.nan), ("chr1", 150, 200, 3.0), ("chr1", 200, 250, np.nan),
           ("chr1", 400, 500, np.nan)])


def test_key_counts_once():
    # two pieces of key 1 and one of key 2 on [0,50)
    args = (["chr1"] * 3, [0, 0, 0], [50, 50, 50], [4.0, 4.0, 10.0])
    _same(_call(*args, agg_type="count", key=[1, 1, 2]), [("chr1", 0, 50, 2.0)])
    _same(_call(*args, agg_type="mean", key=[1, 1, 2]), [("chr1", 0, 50, 7.0)])
    _same(_call(*args, agg_type="count"), [("chr1", 0, 50, 3.0)])


@pytest.mark.parametrize("agg", ["max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac"])
def test_coverage(agg):
    # in [50,100) both intervals cover the whole segment, so the higher value wins
    _same(_call(["chr1", "chr1"], [0, 50], [100, 120], [1.0, 5.0], agg_type=agg),
          [("chr1", 0, 50, 1.0), ("chr1", 50, 100, 5.0), ("chr1", 100, 120, 5.0)])


def test_multi_chrom():
    _same(_call(["chr2", "chr1", "chr1"], [100, 100, 150], [200, 200, 250], [5.0, 2.0, 4.0]),
          [("chr1", 100, 150, 2.0), ("chr1", 150, 200, 3.0), ("chr1", 200, 250, 4.0), ("chr2", 100, 200, 5.0)])


def test_empty_input():
    assert _call([], [], [], []) == []


@pytest.mark.parametrize("agg_name", _TRACK_LIFTOVER_AGGS)
@pytest.mark.parametrize("na_rm", [True, False])
@pytest.mark.parametrize("min_n", [None, 2])
def test_cross_validate_python(agg_name, na_rm, min_n):
    """The C++ aggregation and its Python reference give the same intervals."""
    rng = np.random.default_rng(60427)
    n = 300
    starts = rng.integers(0, 20000, size=n).astype(np.int64)
    values = rng.integers(-8, 8, size=n) / 4.0
    values[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame({
        "chrom": rng.choice(["chrA", "chrB", "chrC"], size=n),
        "start": starts,
        "end": starts + rng.integers(1, 200, size=n),
        "value": values,
        "key": rng.integers(0, 40, size=n),
    })
    kw = {"agg_name": agg_name, "na_rm": na_rm, "min_n": min_n, "nth_index": 2 if agg_name == "nth" else 0}
    cpp = _aggregate_overlapping(df, **kw)
    py = _aggregate_overlapping(df, **kw, _force_pure_python=True)
    assert len(cpp) > 100  # many runs, some of several pieces
    pd.testing.assert_frame_equal(cpp, py, check_dtype=False)
