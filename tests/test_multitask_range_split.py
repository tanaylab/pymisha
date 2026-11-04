"""Range-split multitasking must not cut an interval that lies inside one
iterator bin: the result may not depend on the number of processes."""

import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha
from pymisha._shared import _df2pymisha, _pymisha2df


@pytest.fixture
def sub_bin_intervals():
    rows = [(c, s + 13, s + 53) for c, size in [("chr1", 500000), ("chr2", 300000), ("chrX", 200000)]
            for s in range(0, size - 100, 100)]
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


@pytest.fixture
def _cfg(monkeypatch):
    # set on pm.CONFIG itself: the C++ side reads part of it from there
    def cfg(multitask):
        for k, v in {"progress": False, "multitasking": multitask,
                     "min_scope4process": 0 if multitask else 10**9,
                     "min_intervs4process": 0 if multitask else 10**9}.items():
            monkeypatch.setitem(pm.CONFIG, k, v)
        return pm.CONFIG
    return cfg


def test_extract_and_summary_same_with_and_without_multitasking(sub_bin_intervals, _cfg):
    iv = _df2pymisha(sub_bin_intervals)
    single = _pymisha2df(_pymisha.pm_extract(["dense_track"], iv, 100, _cfg(False)))
    multi = _pymisha2df(_pymisha.pm_extract(["dense_track"], iv, 100, _cfg(True)))
    assert len(single) == len(sub_bin_intervals)
    assert len(multi) == len(single)
    assert not multi["intervalID"].duplicated().any()

    a = _pymisha.pm_summary("dense_track", iv, 100, _cfg(False))
    b = _pymisha.pm_summary("dense_track", iv, 100, _cfg(True))
    assert a["Total intervals"] == b["Total intervals"] == len(sub_bin_intervals)
