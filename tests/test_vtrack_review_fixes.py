"""Regressions from the v0.15.3 review: vtrack name shadowing and filtered quantile 0."""

import numpy as np
import pytest

import pymisha as pm


def test_gvtrack_create_refuses_existing_track_name():
    with pytest.raises(ValueError, match="regular track named dense_track already exists"):
        pm.gvtrack_create("dense_track", "sparse_track", func="avg")
    assert "dense_track" not in pm.gvtrack_ls()


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_filtered_quantile_extremes_match_min_max(q):
    mask = pm.gintervals(["chr1"], [2000], [2600])
    scope = pm.gintervals(["chr1"], [0], [10000])
    pm.gvtrack_create("vq", "dense_track", func="quantile", params=q)
    pm.gvtrack_create("vref", "dense_track", func="min" if q == 0 else "max")
    try:
        pm.gvtrack_filter("vq", filter=mask)
        pm.gvtrack_filter("vref", filter=mask)
        got = pm.gextract(["vq", "vref"], scope, iterator=1000)
        assert got["vref"].notna().any()
        np.testing.assert_allclose(got["vq"], got["vref"], rtol=1e-6, equal_nan=True)
    finally:
        pm.gvtrack_rm("vq")
        pm.gvtrack_rm("vref")


@pytest.mark.parametrize("filt,want", [
    (None, 3.0),
    (("chr2", 0, 100), 3.0),      # masks nothing here
    (("chr1", 50, 60), 3.0),      # one unmasked part per source interval
    (("chr1", 5, 150), 3.0),      # cuts both source intervals
])
def test_dataframe_avg_is_unweighted_with_or_without_filter(filt, want):
    """R misha 5.14.2 gives 3 (the mean of 1 and 5) in every case."""
    import pandas as pd

    src = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 100], "end": [10, 300], "value": [1.0, 5.0]})
    q = pm.gintervals("chr1", 0, 400)
    pm.gvtrack_create("vdf", src, func="avg")
    try:
        if filt is not None:
            pm.gvtrack_filter("vdf", filter=pm.gintervals(*filt))
        assert pm.gextract("vdf", q, iterator=q)["vdf"][0] == pytest.approx(want)
    finally:
        pm.gvtrack_rm("vdf")
