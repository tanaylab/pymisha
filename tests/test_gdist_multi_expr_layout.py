"""Multi-expression gdist: result[i, j] counts values in bin i of expr 1 and bin j of expr 2."""

import numpy as np

import pymisha as pm


def test_gdist_two_exprs_matches_histogram_of_extracted_values():
    b1 = [0, 0.2, 0.5, 1]
    b2 = [0, 0.1, 0.3, 0.6, 1]
    scope = pm.gintervals(["chr1", "chr2"], 0, 200000)
    got = pm.gdist("dense_track", b1, "dense_track * 0.8", b2, intervals=scope, iterator=100)

    vals = pm.gextract(["dense_track", "dense_track * 0.8"], scope, iterator=100)
    x = vals.iloc[:, 3].to_numpy(float)
    y = vals.iloc[:, 4].to_numpy(float)
    ix = np.searchsorted(b1, x, side="left") - 1
    iy = np.searchsorted(b2, y, side="left") - 1
    ok = (ix >= 0) & (ix < 3) & (iy >= 0) & (iy < 4) & ~np.isnan(x) & ~np.isnan(y)
    want = np.zeros((3, 4))
    np.add.at(want, (ix[ok], iy[ok]), 1)

    assert got.shape == (3, 4)
    assert want.sum() > 0
    np.testing.assert_array_equal(got, want)
