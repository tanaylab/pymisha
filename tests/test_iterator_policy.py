import numpy as np

import pymisha as pm


def _extract_single(df):
    assert df is not None
    data_cols = [c for c in df.columns if c not in {"chrom", "start", "end", "intervalID"}]
    assert len(data_cols) == 1
    return df[data_cols[0]].to_numpy(dtype=float, copy=False)


def test_gextract_accepts_float_iterator_matches_int():
    intervals = pm.gintervals("1", 0, 10000)

    int_df = pm.gextract("dense_track", intervals, iterator=1000)
    float_df = pm.gextract("dense_track", intervals, iterator=1000.0)

    int_vals = _extract_single(int_df)
    float_vals = _extract_single(float_df)

    assert int_df[["chrom", "start", "end", "intervalID"]].equals(
        float_df[["chrom", "start", "end", "intervalID"]]
    )
    assert np.array_equal(int_vals, float_vals, equal_nan=True)
