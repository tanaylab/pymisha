import numpy as np
import pytest
import _pymisha


def _call(chrom, start, end, value, agg_type="mean", na_rm=True, min_n=-1, nth_index=0):
    chrom_arr = np.array(chrom, dtype=object)
    start_arr = np.array(start, dtype=np.int64)
    end_arr = np.array(end, dtype=np.int64)
    value_arr = np.array(value, dtype=np.float64)
    df_dict = {"chrom": chrom_arr, "start": start_arr, "end": end_arr, "value": value_arr}
    return _pymisha.pm_liftover_aggregate(df_dict, agg_type, na_rm, int(min_n), int(nth_index))


def test_single_interval_passthrough():
    res = _call(["chr1"], [100], [200], [5.0])
    assert list(res["chrom"]) == ["chr1"]
    assert list(res["start"]) == [100]
    assert list(res["end"]) == [200]
    assert list(res["value"]) == [5.0]


def test_two_overlapping_mean():
    # intervals: [100,200)=2.0, [150,250)=4.0
    # segments: [100,150)=2.0; [150,200)=mean(2,4)=3.0; [200,250)=4.0
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type="mean")
    assert list(res["chrom"]) == ["chr1", "chr1", "chr1"]
    assert list(res["start"]) == [100, 150, 200]
    assert list(res["end"])   == [150, 200, 250]
    assert list(res["value"]) == [2.0, 3.0, 4.0]


def test_fully_contained_mean():
    # [100,500)=2.0, [200,300)=10.0
    # segments: [100,200)=2.0; [200,300)=6.0; [300,500)=2.0
    res = _call(["chr1", "chr1"], [100, 200], [500, 300], [2.0, 10.0], agg_type="mean")
    assert list(res["start"]) == [100, 200, 300]
    assert list(res["end"])   == [200, 300, 500]
    assert list(res["value"]) == [2.0, 6.0, 2.0]


def test_sum():
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type="sum")
    assert list(res["start"]) == [100, 150, 200]
    assert list(res["end"])   == [150, 200, 250]
    assert list(res["value"]) == [2.0, 6.0, 4.0]


def test_min():
    # [100,150)=2.0; [150,200)=min(2,4)=2.0; [200,250)=4.0 - first two merge
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type="min")
    assert list(res["start"]) == [100, 200]
    assert list(res["end"])   == [200, 250]
    assert list(res["value"]) == [2.0, 4.0]


def test_max():
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type="max")
    # [100,150)=2.0; [150,200)=4.0; [200,250)=4.0 - last two merge
    assert list(res["start"]) == [100, 150]
    assert list(res["end"])   == [150, 250]
    assert list(res["value"]) == [2.0, 4.0]


def test_count():
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0], agg_type="count")
    assert list(res["start"]) == [100, 150, 200]
    assert list(res["end"])   == [150, 200, 250]
    assert list(res["value"]) == [1.0, 2.0, 1.0]


def test_median():
    # [100,200)=1.0, [200,300)=2.0, [200,300)=3.0
    # [100,200) active={0} -> 1.0
    # [200,300) active={1,2} -> median(2,3) = 2.5
    res = _call(["chr1"]*3, [100, 200, 200], [200, 300, 300], [1.0, 2.0, 3.0], agg_type="median")
    assert list(res["start"]) == [100, 200]
    assert list(res["end"])   == [200, 300]
    assert list(res["value"]) == [1.0, 2.5]


def test_first():
    # [100,200)=10.0, [150,250)=20.0
    # [100,150) -> 10
    # [150,200) first=10 (row 0 before row 1)
    # [200,250) -> 20
    # adjacent [100,150)+[150,200) merge: same value 10.0
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [10.0, 20.0], agg_type="first")
    assert list(res["start"]) == [100, 200]
    assert list(res["end"])   == [200, 250]
    assert list(res["value"]) == [10.0, 20.0]


def test_last():
    # [100,150) -> 10; [150,200) last=20; [200,250) -> 20 - last two merge
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [10.0, 20.0], agg_type="last")
    assert list(res["start"]) == [100, 150]
    assert list(res["end"])   == [150, 250]
    assert list(res["value"]) == [10.0, 20.0]


def test_nth():
    # [100,150) active=[0] vals=[10] -> nth=2 NaN, drop
    # [150,200) active=[0,1] vals=[10,20] -> nth=2 = 20.0
    # [200,250) active=[1] vals=[20] -> nth=2 NaN, drop
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [10.0, 20.0],
                agg_type="nth", nth_index=2)
    assert list(res["start"]) == [150]
    assert list(res["end"])   == [200]
    assert list(res["value"]) == [20.0]


def test_na_rm_false_propagates_nan():
    # [100,200)=2.0, [150,250)=NaN
    # [100,150) vals=[2.0] (no NaN) -> 2.0
    # [150,200) vals=[2.0, NaN] -> NaN, drop
    # [200,250) vals=[NaN] -> NaN, drop
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, float("nan")],
                agg_type="mean", na_rm=False)
    assert list(res["start"]) == [100]
    assert list(res["end"])   == [150]
    assert list(res["value"]) == [2.0]


def test_min_n_drops_segments():
    # [100,150) n=1 < 2 -> drop
    # [150,200) n=2 -> 3.0
    # [200,250) n=1 -> drop
    res = _call(["chr1", "chr1"], [100, 150], [200, 250], [2.0, 4.0],
                agg_type="mean", min_n=2)
    assert list(res["start"]) == [150]
    assert list(res["end"])   == [200]
    assert list(res["value"]) == [3.0]


def test_multi_chrom():
    res = _call(["chr1", "chr2", "chr1"],
                [100, 100, 150],
                [200, 200, 250],
                [2.0, 5.0, 4.0],
                agg_type="mean")
    assert list(res["chrom"]) == ["chr1", "chr1", "chr1", "chr2"]
    assert list(res["start"]) == [100, 150, 200, 100]
    assert list(res["end"])   == [150, 200, 250, 200]
    assert list(res["value"]) == [2.0, 3.0, 4.0, 5.0]


def test_empty_input():
    res = _call([], [], [], [], agg_type="mean")
    assert len(list(res["chrom"])) == 0
    assert len(list(res["start"])) == 0


import pandas as pd
from pymisha.liftover import _aggregate_overlapping, _AGG_FUNCS


@pytest.mark.parametrize("agg_name", sorted(_AGG_FUNCS.keys()))
@pytest.mark.parametrize("na_rm", [True, False])
def test_cross_validate_python(agg_name, na_rm):
    rng = np.random.default_rng(60427)
    n = 200
    chroms = rng.choice(["chrA", "chrB", "chrC"], size=n)
    starts = rng.integers(0, 5000, size=n).astype(np.int64)
    lengths = rng.integers(50, 500, size=n).astype(np.int64)
    ends = starts + lengths
    values = rng.normal(size=n)
    nan_mask = rng.random(n) < 0.15
    values[nan_mask] = np.nan

    df = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends, "value": values})
    py_out = _aggregate_overlapping(df, _AGG_FUNCS[agg_name], na_rm=na_rm, min_n=None)

    chrom_arr = df["chrom"].to_numpy(dtype=object)
    start_arr = df["start"].to_numpy(dtype=np.int64)
    end_arr = df["end"].to_numpy(dtype=np.int64)
    value_arr = df["value"].to_numpy(dtype=np.float64)
    cpp_dict = _pymisha.pm_liftover_aggregate(
        {"chrom": chrom_arr, "start": start_arr, "end": end_arr, "value": value_arr},
        agg_name, bool(na_rm), -1, 0,
    )
    cpp_out = pd.DataFrame({
        "chrom": list(cpp_dict["chrom"]),
        "start": list(cpp_dict["start"]),
        "end":   list(cpp_dict["end"]),
        "value": list(cpp_dict["value"]),
    })
    # Sort both by (chrom, start, end) for comparison (avoids tripping on
    # multi-chrom ordering differences if any survive).
    py_sorted = py_out.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    cpp_sorted = cpp_out.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(py_sorted, cpp_sorted, atol=1e-9, rtol=1e-9, check_dtype=False)


def test_nth_cross_validate():
    """nth aggregator is C++-only (not in _AGG_FUNCS dict). Cross-validate
    against a Python reference inlined here."""
    rng = np.random.default_rng(60427)
    n = 200
    chroms = rng.choice(["chrA", "chrB", "chrC"], size=n)
    starts = rng.integers(0, 5000, size=n).astype(np.int64)
    lengths = rng.integers(50, 500, size=n).astype(np.int64)
    ends = starts + lengths
    values = rng.normal(size=n)
    values[rng.random(n) < 0.15] = np.nan
    df = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends, "value": values})

    def nth_at_3(v):
        v = np.asarray(v, dtype=np.float64)
        clean = v[~np.isnan(v)]
        if len(clean) < 3:
            return np.nan
        return float(clean[2])

    py_out = _aggregate_overlapping(df, nth_at_3, na_rm=True)
    cpp_dict = _pymisha.pm_liftover_aggregate(
        {"chrom": df["chrom"].to_numpy(dtype=object),
         "start": df["start"].to_numpy(dtype=np.int64),
         "end":   df["end"].to_numpy(dtype=np.int64),
         "value": df["value"].to_numpy(dtype=np.float64)},
        "nth", True, -1, 3,
    )
    cpp_out = pd.DataFrame({
        "chrom": list(cpp_dict["chrom"]),
        "start": list(cpp_dict["start"]),
        "end":   list(cpp_dict["end"]),
        "value": list(cpp_dict["value"]),
    })
    py_sorted = py_out.sort_values(["chrom","start","end"]).reset_index(drop=True)
    cpp_sorted = cpp_out.sort_values(["chrom","start","end"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(py_sorted, cpp_sorted, atol=1e-9, rtol=1e-9, check_dtype=False)
