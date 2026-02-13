import conftest
import numpy as np
import pytest

import pymisha as pm


def test_gquantiles_dense_track_matches_numpy():
    intervals = pm.gintervals_all()
    percentiles = [0.1, 0.5, 0.9]

    values = conftest.extract_values("dense_track", intervals)
    expected = np.nanquantile(values, percentiles)

    result = pm.gquantiles("dense_track", percentiles, intervals)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_gquantiles_accepts_scalar_percentile():
    intervals = pm.gintervals_all()
    values = conftest.extract_values("dense_track", intervals)
    expected = np.nanquantile(values, [0.5])[0]

    result = pm.gquantiles("dense_track", 0.5, intervals)
    assert result.iloc[0] == pytest.approx(expected)


def test_gquantiles_rejects_out_of_range_percentiles():
    intervals = pm.gintervals_all()
    with pytest.raises(ValueError, match=r"percentiles must be within \[0, 1\]"):
        pm.gquantiles("dense_track", [-0.1, 1.1], intervals)


def test_gquantiles_sparse_track_matches_numpy():
    intervals = pm.gintervals_all()
    percentiles = [0.25, 0.5, 0.75]

    values = conftest.extract_values("sparse_track", intervals)
    expected = np.nanquantile(values, percentiles)

    result = pm.gquantiles("sparse_track", percentiles, intervals)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-12, atol=1e-12)
