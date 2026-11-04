"""Integer inputs given as numpy or pandas ints: chrom columns and gpartition breaks."""

import numpy as np
import pandas as pd

import pymisha as pm


def test_int_chrom_column_keeps_each_rows_chrom():
    iv = pd.DataFrame({"chrom": [1, 2, 1, 2], "start": [0, 0, 100, 100], "end": [50, 50, 150, 150]})
    got = pm.gextract("dense_track", iv)
    # genome order, as R; intervalID names the caller's row
    assert got["chrom"].astype(str).tolist() == ["chr1", "chr1", "chr2", "chr2"]
    assert got["intervalID"].tolist() == [1, 3, 2, 4]

    str_iv = iv.assign(chrom=["chr1", "chr2", "chr1", "chr2"])
    want = pm.gextract("dense_track", str_iv)
    np.testing.assert_array_equal(got["dense_track"].to_numpy(), want["dense_track"].to_numpy())


def test_gpartition_accepts_integer_numpy_breaks():
    scope = pm.gintervals(["chr1"], [0], [10000])
    want = pm.gpartition("dense_track * 100", [0, 10, 20, 50, 100], scope)
    got = pm.gpartition("dense_track * 100", np.array([0, 10, 20, 50, 100]), scope)
    pd.testing.assert_frame_equal(got, want)
