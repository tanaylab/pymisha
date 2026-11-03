"""pm_liftover_aggregate: the sparse target of gtrack_liftover, as R's GTrackLiftover.cpp.

Per chromosome every run of overlapping intervals becomes one interval, valued
by R's aggregate_values over the run: an interval counts its whole length,
intervals sharing a key count once, and a NaN result is kept. Touching
intervals are not merged. The end-to-end comparison with R is in
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
    ("first", 2.0), ("last", 4.0),
])
def test_overlapping_run_is_one_interval(agg, value):
    # [100,200)=2 and [150,250)=4 overlap: R writes [100,250)
    _same(_call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type=agg),
          [("chr1", 100, 250, value)])


def test_contained_interval():
    _same(_call(["chr1", "chr1"], [100, 200], [500, 300], [2.0, 10.0]), [("chr1", 100, 500, 6.0)])


def test_touching_intervals_are_not_merged():
    # [100,200) and [200,300) touch, so they are two runs; the second averages 2 and 3
    _same(_call(["chr1"] * 3, [100, 200, 200], [200, 300, 300], [1.0, 2.0, 3.0], agg_type="median"),
          [("chr1", 100, 200, 1.0), ("chr1", 200, 300, 2.5)])


def test_nth():
    # in (start, end) order: 10, 20
    _same(_call(["chr1", "chr1"], [100, 150], [200, 250], [10.0, 20.0], agg_type="nth", nth_index=2),
          [("chr1", 100, 250, 20.0)])
    _same(_call(["chr1", "chr1"], [100, 150], [200, 250], [10.0, 20.0], agg_type="nth", nth_index=3),
          [("chr1", 100, 250, np.nan)])


def test_nan_is_kept():
    # a NaN interval stretches the run, and a run of NaN alone is a NaN interval
    nan = float("nan")
    _same(_call(["chr1"] * 3, [100, 150, 400], [200, 250, 500], [2.0, nan, nan]),
          [("chr1", 100, 250, 2.0), ("chr1", 400, 500, np.nan)])
    _same(_call(["chr1"] * 3, [100, 150, 400], [200, 250, 500], [2.0, nan, nan], agg_type="count"),
          [("chr1", 100, 250, 1.0), ("chr1", 400, 500, 0.0)])
    _same(_call(["chr1"] * 3, [100, 150, 400], [200, 250, 500], [2.0, nan, nan], na_rm=False),
          [("chr1", 100, 250, np.nan), ("chr1", 400, 500, np.nan)])


def test_min_n():
    _same(_call(["chr1"] * 3, [100, 150, 400], [200, 250, 500], [2.0, 4.0, 1.0], min_n=2),
          [("chr1", 100, 250, 3.0), ("chr1", 400, 500, np.nan)])


def test_key_counts_once():
    # [0,50) and [50,100) are one source interval (key 1) and [20,80) another
    args = (["chr1"] * 3, [0, 20, 50], [50, 80, 100], [4.0, 10.0, 4.0])
    _same(_call(*args, agg_type="count", key=[1, 2, 1]), [("chr1", 0, 100, 2.0)])
    _same(_call(*args, agg_type="mean", key=[1, 2, 1]), [("chr1", 0, 100, 7.0)])
    _same(_call(*args, agg_type="count"), [("chr1", 0, 100, 3.0)])


@pytest.mark.parametrize("agg, value", [
    ("max.coverage_len", 1.0), ("min.coverage_len", 5.0),
    # every interval covers all of itself, so the fractions tie and the higher value wins
    ("max.coverage_frac", 5.0), ("min.coverage_frac", 5.0),
])
def test_coverage(agg, value):
    _same(_call(["chr1", "chr1"], [0, 50], [100, 120], [1.0, 5.0], agg_type=agg), [("chr1", 0, 120, value)])


def test_multi_chrom():
    _same(_call(["chr2", "chr1", "chr1"], [100, 100, 150], [200, 200, 250], [5.0, 2.0, 4.0]),
          [("chr1", 100, 250, 3.0), ("chr2", 100, 200, 5.0)])


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
