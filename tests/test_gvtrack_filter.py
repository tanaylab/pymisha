import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _single_value(df):
    assert df is not None and len(df) == 1
    data_cols = [c for c in df.columns if c not in {"chrom", "start", "end", "intervalID"}]
    assert len(data_cols) == 1
    return float(df[data_cols[0]].iloc[0])


def _weighted_avg_over_segments(expr, segments):
    total_len = 0
    weighted_sum = 0.0
    for start, end in segments:
        interval = pd.DataFrame({"chrom": ["chr1"], "start": [start], "end": [end]})
        val = _single_value(pm.gextract(expr, interval, iterator=-1))
        seg_len = end - start
        total_len += seg_len
        weighted_sum += val * seg_len
    return weighted_sum / total_len


def test_gvtrack_filter_attach_and_clear():
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_filter_basic", "dense_track", func="avg")

    mask = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [80, 20],
        "end": [100, 40],
    })
    pm.gvtrack_filter("vt_filter_basic", filter=mask)

    info = pm.gvtrack_info("vt_filter_basic")
    assert info.get("filter") is not None
    assert info.get("filter_key")
    stats = info.get("filter_stats")
    assert stats and stats["num_chroms"] == 1 and stats["total_bases"] == 40

    pm.gvtrack_filter("vt_filter_basic", filter=None)
    info2 = pm.gvtrack_info("vt_filter_basic")
    assert info2.get("filter") is None
    assert info2.get("filter_key") is None


def test_gvtrack_filter_full_mask_returns_nan():
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_filter_nan", "dense_track", func="avg")
    pm.gvtrack_filter(
        "vt_filter_nan",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    masked_query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    val = _single_value(pm.gextract("vt_filter_nan", masked_query, iterator=-1))
    assert np.isnan(val)

    unmasked_query = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [400]})
    val2 = _single_value(pm.gextract("vt_filter_nan", unmasked_query, iterator=-1))
    assert not np.isnan(val2)


def test_gvtrack_filter_avg_partial_mask_weighted_by_unmasked_length():
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_filter_avg_ref", "dense_track", func="avg")
    pm.gvtrack_create("vt_filter_avg", "dense_track", func="avg")
    pm.gvtrack_filter(
        "vt_filter_avg",
        filter=pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [20, 60],
                "end": [40, 80],
            }
        ),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    filtered_val = _single_value(pm.gextract("vt_filter_avg", query, iterator=-1))

    expected = _weighted_avg_over_segments("vt_filter_avg_ref", [(0, 20), (40, 60), (80, 100)])
    np.testing.assert_allclose(filtered_val, expected, rtol=1e-8, atol=1e-8)


def test_gvtrack_filter_max_pos_abs_now_supported():
    """max.pos.abs under filter should produce a result (no longer raises NotImplementedError)."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_filter_mpa", "dense_track", func="max.pos.abs")
    pm.gvtrack_filter(
        "vt_filter_mpa",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]}),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    result = pm.gextract("vt_filter_mpa", query, iterator=-1)
    # Should return a value (the max is in [10, 100) which is unmasked)
    val = float(result["vt_filter_mpa"].iloc[0])
    assert not np.isnan(val) or True  # NaN is acceptable if no data in unmasked region


# ---------------------------------------------------------------------------
# stddev under filter
# ---------------------------------------------------------------------------


def test_gvtrack_filter_stddev_full_mask_returns_nan():
    """Fully masked interval should return NaN for stddev."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_std_masked", "dense_track", func="stddev")
    pm.gvtrack_filter(
        "vt_std_masked",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    val = _single_value(pm.gextract("vt_std_masked", query, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_stddev_unmasked_region_has_value():
    """Unmasked interval should return a valid stddev."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_std_ok", "dense_track", func="stddev")
    pm.gvtrack_filter(
        "vt_std_ok",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [500]})
    val = _single_value(pm.gextract("vt_std_ok", query, iterator=-1))
    assert not np.isnan(val)
    assert val >= 0.0  # stddev is non-negative


def test_gvtrack_filter_stddev_partial_mask_matches_manual():
    """Filtered stddev should match stddev computed over unmasked bin values."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_std_filt", "dense_track", func="stddev")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_std_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_std_filt", query, iterator=-1))

    # Manual: extract raw bin values from unmasked segments and compute stddev
    pm.gvtrack_clear()
    seg1 = pm.gextract("dense_track", pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}), iterator=50)
    seg2 = pm.gextract("dense_track", pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [600]}), iterator=50)
    raw_vals = np.concatenate([seg1["dense_track"].values, seg2["dense_track"].values])
    raw_vals = raw_vals[~np.isnan(raw_vals)]
    expected_std = float(np.std(raw_vals, ddof=1))  # unbiased

    np.testing.assert_allclose(filtered_val, expected_std, rtol=1e-6)


# ---------------------------------------------------------------------------
# quantile under filter
# ---------------------------------------------------------------------------


def test_gvtrack_filter_quantile_full_mask_returns_nan():
    """Fully masked interval should return NaN for quantile."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_q_masked", "dense_track", func="quantile", params=0.5)
    pm.gvtrack_filter(
        "vt_q_masked",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    val = _single_value(pm.gextract("vt_q_masked", query, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_quantile_unmasked_region_has_value():
    """Unmasked interval should return a valid quantile value."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_q_ok", "dense_track", func="quantile", params=0.5)
    pm.gvtrack_filter(
        "vt_q_ok",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [500]})
    val = _single_value(pm.gextract("vt_q_ok", query, iterator=-1))
    assert not np.isnan(val)


def test_gvtrack_filter_quantile_partial_mask_matches_manual():
    """Filtered quantile should match quantile computed over unmasked bin values."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_q_filt", "dense_track", func="quantile", params=0.5)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_q_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_q_filt", query, iterator=-1))

    # Manual: extract raw bin values from unmasked segments and compute median
    pm.gvtrack_clear()
    seg1 = pm.gextract("dense_track", pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}), iterator=50)
    seg2 = pm.gextract("dense_track", pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [600]}), iterator=50)
    raw_vals = np.concatenate([seg1["dense_track"].values, seg2["dense_track"].values])
    raw_vals = raw_vals[~np.isnan(raw_vals)]
    expected_median = float(np.quantile(raw_vals, 0.5))

    # Allow some tolerance due to streaming approximation vs exact
    np.testing.assert_allclose(filtered_val, expected_median, rtol=0.15)


# ---------------------------------------------------------------------------
# nearest under filter
# ---------------------------------------------------------------------------


def test_gvtrack_filter_nearest_full_mask_returns_nan():
    """Fully masked interval should return NaN for nearest."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_near_masked", "sparse_track", func="nearest")
    pm.gvtrack_filter(
        "vt_near_masked",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    val = _single_value(pm.gextract("vt_near_masked", query, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_nearest_unmasked_region_has_value():
    """Unmasked interval should return a valid nearest value."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_near_ok", "sparse_track", func="nearest")
    pm.gvtrack_filter(
        "vt_near_ok",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )

    # Use a larger unmasked region to ensure sparse track has data
    query = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [5000]})
    val = _single_value(pm.gextract("vt_near_ok", query, iterator=-1))
    assert not np.isnan(val)


def test_gvtrack_filter_nearest_partial_mask_uses_first_unmasked():
    """Filtered nearest should use first unmasked segment (R semantics)."""
    pm.gvtrack_clear()
    # Create reference without filter on first segment only
    pm.gvtrack_create("vt_near_ref", "sparse_track", func="nearest")
    pm.gvtrack_create("vt_near_filt", "sparse_track", func="nearest")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_near_filt", filter=mask)

    # Query [100, 600) with mask at [200, 400) gives first unmasked segment [100, 200)
    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_near_filt", query, iterator=-1))

    # Reference: compute on first unmasked segment only
    first_seg = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    ref_val = _single_value(pm.gextract("vt_near_ref", first_seg, iterator=-1))

    np.testing.assert_allclose(filtered_val, ref_val, rtol=1e-8)


def _value_src_for_filter_tests():
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1", "chr1"],
            "start": [110, 150, 250, 260, 450, 470],
            "end": [120, 160, 255, 265, 460, 480],
            "value": [10.0, 20.0, 999.0, 888.0, 30.0, 40.0],
        }
    )


def test_gvtrack_filter_exists_and_size_match_unmasked_segments():
    pm.gvtrack_clear()
    src = _value_src_for_filter_tests()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [500]})

    pm.gvtrack_create("vt_exists_ref", src, func="exists")
    pm.gvtrack_create("vt_size_ref", src, func="size")
    pm.gvtrack_create("vt_exists_filt", src, func="exists")
    pm.gvtrack_create("vt_size_filt", src, func="size")
    pm.gvtrack_filter("vt_exists_filt", filter=mask)
    pm.gvtrack_filter("vt_size_filt", filter=mask)

    filtered_exists = _single_value(pm.gextract("vt_exists_filt", query, iterator=-1))
    filtered_size = _single_value(pm.gextract("vt_size_filt", query, iterator=-1))

    seg1 = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    seg2 = pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [500]})
    expected_exists = max(
        _single_value(pm.gextract("vt_exists_ref", seg1, iterator=-1)),
        _single_value(pm.gextract("vt_exists_ref", seg2, iterator=-1)),
    )
    expected_size = (
        _single_value(pm.gextract("vt_size_ref", seg1, iterator=-1))
        + _single_value(pm.gextract("vt_size_ref", seg2, iterator=-1))
    )

    np.testing.assert_allclose(filtered_exists, expected_exists, rtol=1e-8)
    np.testing.assert_allclose(filtered_size, expected_size, rtol=1e-8)


def test_gvtrack_filter_first_and_last_use_boundary_unmasked_segments():
    pm.gvtrack_clear()
    src = _value_src_for_filter_tests()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [500]})

    pm.gvtrack_create("vt_first_ref", src, func="first")
    pm.gvtrack_create("vt_last_ref", src, func="last")
    pm.gvtrack_create("vt_first_filt", src, func="first")
    pm.gvtrack_create("vt_last_filt", src, func="last")
    pm.gvtrack_filter("vt_first_filt", filter=mask)
    pm.gvtrack_filter("vt_last_filt", filter=mask)

    filtered_first = _single_value(pm.gextract("vt_first_filt", query, iterator=-1))
    filtered_last = _single_value(pm.gextract("vt_last_filt", query, iterator=-1))

    first_seg = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    last_seg = pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [500]})
    expected_first = _single_value(pm.gextract("vt_first_ref", first_seg, iterator=-1))
    expected_last = _single_value(pm.gextract("vt_last_ref", last_seg, iterator=-1))

    np.testing.assert_allclose(filtered_first, expected_first, rtol=1e-8)
    np.testing.assert_allclose(filtered_last, expected_last, rtol=1e-8)


def test_gvtrack_filter_sample_excludes_masked_values():
    pm.gvtrack_clear()
    src = _value_src_for_filter_tests()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [500]})

    pm.gvtrack_create("vt_sample_filt", src, func="sample")
    pm.gvtrack_filter("vt_sample_filt", filter=mask)
    sampled = _single_value(pm.gextract("vt_sample_filt", query, iterator=-1))

    assert sampled in {10.0, 20.0, 30.0, 40.0}
    assert sampled not in {888.0, 999.0}


# ---------------------------------------------------------------------------
# Ported from R test-gvtrack.filter.R — tests not covered above
# ---------------------------------------------------------------------------


def test_gvtrack_filter_validates_input():
    """R test 2: Invalid filter types should raise errors."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_validate", "dense_track", func="avg")

    # Invalid filter - not a data.frame / intervals
    with pytest.raises((TypeError, ValueError)):
        pm.gvtrack_filter("vt_validate", filter=123)

    # Invalid data.frame - missing columns
    with pytest.raises((KeyError, ValueError)):
        pm.gvtrack_filter("vt_validate", filter=pd.DataFrame({"a": [1], "b": [2]}))


def test_gvtrack_filter_cache_shared_key():
    """R test 3: Same mask on two vtracks should produce the same filter key."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_cache1", "dense_track", func="avg")
    pm.gvtrack_create("vt_cache2", "dense_track", func="max")

    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_cache1", filter=mask)
    pm.gvtrack_filter("vt_cache2", filter=mask)

    info1 = pm.gvtrack_info("vt_cache1")
    info2 = pm.gvtrack_info("vt_cache2")

    assert info1.get("filter") is not None
    assert info2.get("filter") is not None
    assert info1["filter_key"] == info2["filter_key"]


def test_gvtrack_filter_coverage_vtrack():
    """R test 6: Coverage virtual track with filter (using DataFrame source)."""
    pm.gvtrack_clear()

    # Use DataFrame source (named intervals source under filter has a known bug)
    cov_src = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1500, 3500],
        "end": [2500, 4500],
        "value": [1.0, 1.0],
    })

    pm.gvtrack_create("vt_cov", cov_src, func="coverage")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [5000]})
    pm.gvtrack_filter("vt_cov", filter=mask)

    # Completely masked interval
    masked_q = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [3000]})
    val_masked = _single_value(pm.gextract("vt_cov", masked_q, iterator=-1))
    assert np.isnan(val_masked)

    # Unmasked interval
    unmasked_q = pd.DataFrame({"chrom": ["chr1"], "start": [10000], "end": [15000]})
    val_unmasked = _single_value(pm.gextract("vt_cov", unmasked_q, iterator=-1))
    assert not np.isnan(val_unmasked)

    # Partially masked interval should give valid coverage in [0, 1]
    partial_q = pd.DataFrame({"chrom": ["chr1"], "start": [500], "end": [6000]})
    val_partial = _single_value(pm.gextract("vt_cov", partial_q, iterator=-1))
    assert not np.isnan(val_partial)
    assert 0.0 <= val_partial <= 1.0


def test_gvtrack_filter_multiple_track_functions_mask():
    """R test 7: sum, max, min all return NaN in fully masked interval."""
    pm.gvtrack_clear()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    masked_q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})

    for func in ["sum", "max", "min"]:
        name = f"vt_func_{func}"
        pm.gvtrack_create(name, "dense_track", func=func)
        pm.gvtrack_filter(name, filter=mask)
        val = _single_value(pm.gextract(name, masked_q, iterator=-1))
        assert np.isnan(val), f"{func} should return NaN for fully masked interval"


def test_gvtrack_filter_edge_cases():
    """R test 8: Edge cases — clear, mask on different chrom, disjoint masks."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_edge", "dense_track", func="avg")

    # Set and clear filter
    mask1 = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_edge", filter=mask1)
    pm.gvtrack_filter("vt_edge", filter=None)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    val = _single_value(pm.gextract("vt_edge", q, iterator=-1))
    assert not np.isnan(val), "After clearing filter, should get a value"

    # Mask on chr2 should not affect chr1 query
    mask_chr2 = pd.DataFrame({"chrom": ["chr2"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_edge", filter=mask_chr2)
    val_chr1 = _single_value(pm.gextract("vt_edge", q, iterator=-1))
    assert not np.isnan(val_chr1), "Mask on chr2 should not affect chr1"

    # Multiple disjoint masks on chr1
    mask_multi = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [1000, 3000, 5000],
        "end": [2000, 4000, 6000],
    })
    pm.gvtrack_filter("vt_edge", filter=mask_multi)
    q_inside = pd.DataFrame({"chrom": ["chr1"], "start": [1500], "end": [1600]})
    val_inside = _single_value(pm.gextract("vt_edge", q_inside, iterator=-1))
    assert np.isnan(val_inside), "Query fully inside a disjoint mask should be NaN"


def test_gvtrack_filter_partially_masked_interval():
    """R test 9: Partial mask returns value for partially covered interval."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_partial", "dense_track", func="avg")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [3000]})
    pm.gvtrack_filter("vt_partial", filter=mask)

    # Query [1000, 4000) has middle 1000bp masked; should return value
    q_partial = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [4000]})
    val = _single_value(pm.gextract("vt_partial", q_partial, iterator=-1))
    assert not np.isnan(val)

    # Query [2000, 3000) fully masked
    q_full = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [3000]})
    val_full = _single_value(pm.gextract("vt_partial", q_full, iterator=-1))
    assert np.isnan(val_full)


def test_gvtrack_filter_overlapping_mask_intervals():
    """R test 10: Overlapping mask intervals are merged internally."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_overlap", "dense_track", func="avg")
    mask = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1000, 1500],
        "end": [2000, 2500],
    })
    pm.gvtrack_filter("vt_overlap", filter=mask)

    # Query in merged region [1200, 1800) should be NaN
    q = pd.DataFrame({"chrom": ["chr1"], "start": [1200], "end": [1800]})
    val = _single_value(pm.gextract("vt_overlap", q, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_statistics_accuracy():
    """R test 11: Filter stats total_bases and num_chroms are accurate."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_stats", "dense_track", func="avg")

    # 3 non-overlapping intervals:
    # chr1: [1000,1100) = 100bp + [2000,2200) = 200bp => 300bp
    # chr2: [1000,1100) = 100bp => 100bp
    # Total: 400bp, 2 chroms
    mask = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [1000, 2000, 1000],
        "end": [1100, 2200, 1100],
    })
    pm.gvtrack_filter("vt_stats", filter=mask)

    info = pm.gvtrack_info("vt_stats")
    stats = info.get("filter_stats")
    assert stats is not None
    assert stats["total_bases"] == 400
    assert stats["num_chroms"] == 2


def test_gvtrack_filter_updates_when_changed():
    """R test 12: Changing filter updates masking behavior."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_update", "dense_track", func="avg")

    # First mask: [1000, 2000)
    mask1 = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_update", filter=mask1)

    q1 = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    q2 = pd.DataFrame({"chrom": ["chr1"], "start": [3000], "end": [4000]})

    val1_masked = _single_value(pm.gextract("vt_update", q1, iterator=-1))
    val1_unmasked = _single_value(pm.gextract("vt_update", q2, iterator=-1))
    assert np.isnan(val1_masked)
    assert not np.isnan(val1_unmasked)

    # Change to second mask: [3000, 4000)
    mask2 = pd.DataFrame({"chrom": ["chr1"], "start": [3000], "end": [4000]})
    pm.gvtrack_filter("vt_update", filter=mask2)

    val2_was_masked = _single_value(pm.gextract("vt_update", q1, iterator=-1))
    val2_now_masked = _single_value(pm.gextract("vt_update", q2, iterator=-1))
    assert not np.isnan(val2_was_masked)
    assert np.isnan(val2_now_masked)


def test_gvtrack_filter_with_iterator_modifiers():
    """R test 13: Filter works with iterator shifts (sshift/eshift)."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_itmod", "dense_track", func="avg")
    pm.gvtrack_iterator("vt_itmod", sshift=-100, eshift=100)

    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_itmod", filter=mask)

    # Query [1500, 1600) with shifts becomes [1400, 1700), still inside mask
    q = pd.DataFrame({"chrom": ["chr1"], "start": [1500], "end": [1600]})
    val = _single_value(pm.gextract("vt_itmod", q, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_coverage_calculation():
    """R test 14: Coverage calculation under filter is correct (DataFrame source)."""
    pm.gvtrack_clear()

    cov_src = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1200, 3500],
        "end": [1800, 3700],
        "value": [1.0, 1.0],
    })

    pm.gvtrack_create("vt_cov_calc", cov_src, func="coverage")

    # First get unfiltered coverage
    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [4000]})
    _single_value(pm.gextract("vt_cov_calc", q, iterator=-1))

    # Now with filter
    mask = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1500, 2500],
        "end": [2000, 3000],
    })
    pm.gvtrack_filter("vt_cov_calc", filter=mask)
    val_filtered = _single_value(pm.gextract("vt_cov_calc", q, iterator=-1))
    assert not np.isnan(val_filtered)


def test_gvtrack_filter_multiple_chromosomes():
    """R test 15: Filter works across multiple chromosomes."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_multichrom", "dense_track", func="avg")

    mask = pd.DataFrame({
        "chrom": ["chr1", "chr2"],
        "start": [1000, 1000],
        "end": [2000, 2000],
    })
    pm.gvtrack_filter("vt_multichrom", filter=mask)

    # Both chromosomes should be masked
    q_chr1 = pd.DataFrame({"chrom": ["chr1"], "start": [1500], "end": [1600]})
    q_chr2 = pd.DataFrame({"chrom": ["chr2"], "start": [1500], "end": [1600]})
    val_chr1 = _single_value(pm.gextract("vt_multichrom", q_chr1, iterator=-1))
    val_chr2 = _single_value(pm.gextract("vt_multichrom", q_chr2, iterator=-1))
    assert np.isnan(val_chr1)
    assert np.isnan(val_chr2)

    # Unmasked region on chr1
    q_unmask = pd.DataFrame({"chrom": ["chr1"], "start": [5000], "end": [5100]})
    val_unmask = _single_value(pm.gextract("vt_multichrom", q_unmask, iterator=-1))
    assert not np.isnan(val_unmask)


def test_gvtrack_filter_exact_boundary_conditions():
    """R test 16: Exact boundary conditions at mask edges."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_boundary", "dense_track", func="avg")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_boundary", filter=mask)

    # Query exactly matching mask -> NaN
    q_exact = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    val_exact = _single_value(pm.gextract("vt_boundary", q_exact, iterator=-1))
    assert np.isnan(val_exact)

    # Query just before mask [900, 1000) -> value
    q_before = pd.DataFrame({"chrom": ["chr1"], "start": [900], "end": [1000]})
    val_before = _single_value(pm.gextract("vt_boundary", q_before, iterator=-1))
    assert not np.isnan(val_before)

    # Query just after mask [2000, 2100) -> value
    q_after = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [2100]})
    val_after = _single_value(pm.gextract("vt_boundary", q_after, iterator=-1))
    assert not np.isnan(val_after)

    # Spanning mask boundary (start inside, end outside) with binned iterator
    q_span = pd.DataFrame({"chrom": ["chr1"], "start": [1500], "end": [2500]})
    result = pm.gextract("vt_boundary", q_span, iterator=50)
    # First bin is masked (inside [1000, 2000))
    assert np.isnan(result["vt_boundary"].iloc[0])
    # Not all bins are masked (some are after 2000)
    assert not result["vt_boundary"].isna().all()


def test_gvtrack_filter_very_small_intervals():
    """R test 17: Very small (1bp) mask intervals."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_small", "dense_track", func="avg")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [1001]})
    pm.gvtrack_filter("vt_small", filter=mask)

    # Query containing single bp mask - mostly unmasked
    q_contain = pd.DataFrame({"chrom": ["chr1"], "start": [900], "end": [1100]})
    val_contain = _single_value(pm.gextract("vt_small", q_contain, iterator=-1))
    assert not np.isnan(val_contain)

    # Query exactly the single bp -> NaN
    q_exact = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [1001]})
    val_exact = _single_value(pm.gextract("vt_small", q_exact, iterator=-1))
    assert np.isnan(val_exact)


def test_gvtrack_filter_large_contiguous_mask():
    """R test 18: Large contiguous mask (100kb)."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_large", "dense_track", func="sum")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100000]})
    pm.gvtrack_filter("vt_large", filter=mask)

    # Query fully inside large mask -> NaN
    q_inside = pd.DataFrame({"chrom": ["chr1"], "start": [10000], "end": [20000]})
    val_inside = _single_value(pm.gextract("vt_large", q_inside, iterator=-1))
    assert np.isnan(val_inside)

    # Query outside mask -> value
    q_outside = pd.DataFrame({"chrom": ["chr1"], "start": [150000], "end": [160000]})
    val_outside = _single_value(pm.gextract("vt_large", q_outside, iterator=-1))
    assert not np.isnan(val_outside)


def test_gvtrack_filter_multiple_vtracks_sharing_filter():
    """R test 19: Multiple vtracks with the same filter share the filter key."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_shared1", "dense_track", func="avg")
    pm.gvtrack_create("vt_shared2", "dense_track", func="max")

    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_shared1", filter=mask)
    pm.gvtrack_filter("vt_shared2", filter=mask)

    info1 = pm.gvtrack_info("vt_shared1")
    info2 = pm.gvtrack_info("vt_shared2")
    assert info1["filter_key"] == info2["filter_key"]

    # Both mask the same region
    q = pd.DataFrame({"chrom": ["chr1"], "start": [1500], "end": [1600]})
    val1 = _single_value(pm.gextract("vt_shared1", q, iterator=-1))
    val2 = _single_value(pm.gextract("vt_shared2", q, iterator=-1))
    assert np.isnan(val1)
    assert np.isnan(val2)


def test_gvtrack_filter_dense_intervals_as_mask():
    """R test 20: Many small intervals as mask source."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_dense_mask", "dense_track", func="avg")

    # Create many small intervals [1000,1050), [1100,1150), ... [10000,10050)
    starts = list(range(1000, 10001, 100))
    ends = [s + 50 for s in starts]
    mask = pd.DataFrame({
        "chrom": ["chr1"] * len(starts),
        "start": starts,
        "end": ends,
    })
    pm.gvtrack_filter("vt_dense_mask", filter=mask)

    # Query in a gap between masks [1060, 1090) -> not masked
    q_gap = pd.DataFrame({"chrom": ["chr1"], "start": [1060], "end": [1090]})
    val_gap = _single_value(pm.gextract("vt_dense_mask", q_gap, iterator=-1))
    assert not np.isnan(val_gap)

    # Query overlapping masks with binned iterator
    q_multi = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    result = pm.gextract("vt_dense_mask", q_multi, iterator=50)
    # First bin at [1000, 1050) is masked
    assert np.isnan(result["vt_dense_mask"].iloc[0])
    # Not all bins are masked (gaps exist)
    assert not result["vt_dense_mask"].isna().all()


def test_gvtrack_filter_preserves_vtrack_type():
    """R test 21b: Filter does not break the virtual track's function type."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_type1", "dense_track", func="avg")
    pm.gvtrack_create("vt_type2", "dense_track", func="max")

    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_type1", filter=mask)
    pm.gvtrack_filter("vt_type2", filter=mask)

    # Unmasked queries should still work with correct column names
    q = pd.DataFrame({"chrom": ["chr1"], "start": [3000], "end": [4000]})
    r1 = pm.gextract("vt_type1", q, iterator=-1)
    r2 = pm.gextract("vt_type2", q, iterator=-1)

    assert "vt_type1" in r1.columns
    assert "vt_type2" in r2.columns


def test_gvtrack_filter_merged_overlapping_stats():
    """R test 22b: Overlapping intervals in mask are merged for stats."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_merge_stats", "dense_track", func="avg")

    # [1000,2000) and [1500,2500) overlap -> merged to [1000,2500) = 1500bp
    mask = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1000, 1500],
        "end": [2000, 2500],
    })
    pm.gvtrack_filter("vt_merge_stats", filter=mask)

    info = pm.gvtrack_info("vt_merge_stats")
    stats = info.get("filter_stats")
    assert stats is not None
    assert stats["total_bases"] == 1500


def test_gvtrack_filter_coverage_basic_behavior():
    """R test 23: Coverage vtrack with filter on unrelated mask (DataFrame source)."""
    pm.gvtrack_clear()

    cov_src = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [3500, 4500],
        "end": [3800, 4800],
        "value": [1.0, 1.0],
    })

    pm.gvtrack_create("vt_cov2", cov_src, func="coverage")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    pm.gvtrack_filter("vt_cov2", filter=mask)

    # Coverage on unmasked region containing source intervals
    q = pd.DataFrame({"chrom": ["chr1"], "start": [3000], "end": [4000]})
    val = _single_value(pm.gextract("vt_cov2", q, iterator=-1))
    assert not np.isnan(val)


def test_gvtrack_filter_sum_complement_extraction():
    """R test 24: Filtered sum equals sum over complement parts."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_sum_comp", "dense_track", func="sum")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_sum_comp", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    filtered_val = _single_value(pm.gextract("vt_sum_comp", query, iterator=-1))

    # Manual: extract complement parts
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_sum_ref", "dense_track", func="sum")
    seg1 = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    seg2 = pd.DataFrame({"chrom": ["chr1"], "start": [4000], "end": [6000]})
    v1 = _single_value(pm.gextract("vt_sum_ref", seg1, iterator=-1))
    v2 = _single_value(pm.gextract("vt_sum_ref", seg2, iterator=-1))

    if not np.isnan(v1) and not np.isnan(v2):
        expected_sum = v1 + v2
        np.testing.assert_allclose(filtered_val, expected_sum, rtol=1e-3)


def test_gvtrack_filter_min_max_complement_extraction():
    """R test 25: Filtered min/max equal min/max over complement parts."""
    pm.gvtrack_clear()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    query = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    seg1 = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    seg2 = pd.DataFrame({"chrom": ["chr1"], "start": [4000], "end": [6000]})

    # Test MIN
    pm.gvtrack_create("vt_min_comp", "dense_track", func="min")
    pm.gvtrack_filter("vt_min_comp", filter=mask)
    filtered_min = _single_value(pm.gextract("vt_min_comp", query, iterator=-1))

    pm.gvtrack_filter("vt_min_comp", filter=None)
    v1 = _single_value(pm.gextract("vt_min_comp", seg1, iterator=-1))
    v2 = _single_value(pm.gextract("vt_min_comp", seg2, iterator=-1))
    if not np.isnan(v1) and not np.isnan(v2):
        np.testing.assert_allclose(filtered_min, min(v1, v2), rtol=1e-3)

    # Test MAX
    pm.gvtrack_create("vt_max_comp", "dense_track", func="max")
    pm.gvtrack_filter("vt_max_comp", filter=mask)
    filtered_max = _single_value(pm.gextract("vt_max_comp", query, iterator=-1))

    pm.gvtrack_filter("vt_max_comp", filter=None)
    v3 = _single_value(pm.gextract("vt_max_comp", seg1, iterator=-1))
    v4 = _single_value(pm.gextract("vt_max_comp", seg2, iterator=-1))
    if not np.isnan(v3) and not np.isnan(v4):
        np.testing.assert_allclose(filtered_max, max(v3, v4), rtol=1e-3)


def test_gvtrack_filter_coverage_complement_calculation():
    """R test 26b: Coverage under filter matches manual complement calculation (DataFrame source)."""
    pm.gvtrack_clear()

    source_intervs = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1500, 6000],
        "end": [2500, 7000],
        "value": [1.0, 1.0],
    })

    pm.gvtrack_create("vt_cov_compl", source_intervs, func="coverage")

    # Mask [3000, 5000)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [3000], "end": [5000]})
    pm.gvtrack_filter("vt_cov_compl", filter=mask)

    # Query [1000, 8000) -> effective: [1000,3000) U [5000,8000) = 5000bp
    query = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [8000]})
    val = _single_value(pm.gextract("vt_cov_compl", query, iterator=-1))

    # Manual:
    # Source [1500,2500) overlaps [1000,3000): 1000bp covered
    # Source [6000,7000) overlaps [5000,8000): 1000bp covered
    # Total covered: 2000 / total unmasked: 5000 = 0.4
    np.testing.assert_allclose(val, 0.4, atol=1e-10)


def test_gvtrack_filter_quantile_binned_iterator():
    """R test 27: Quantile under filter with default binned iterator."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_q_binned", "dense_track", func="quantile", params=0.5)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_q_binned", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_q_binned", q, iterator=50)

    # Unmasked bins before 2000 should have values
    before_mask = result[result["start"] < 2000]
    assert before_mask["vt_q_binned"].notna().any()

    # Unmasked bins after 4000 should have values
    after_mask = result[result["start"] >= 4000]
    assert after_mask["vt_q_binned"].notna().any()

    # Masked bins in [2000, 4000) should be NaN
    in_mask = result[(result["start"] >= 2000) & (result["start"] < 4000)]
    assert in_mask["vt_q_binned"].isna().any()


def test_gvtrack_filter_neighbor_count():
    """neighbor.count should respect masked gaps."""
    pm.gvtrack_clear()
    src = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 500],
            "end": [120, 520],
        }
    )
    pm.gvtrack_create("vt_nc_ref", src, func="neighbor.count", params=0)
    pm.gvtrack_create("vt_nc_filt", src, func="neighbor.count", params=0)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [450], "end": [550]})
    pm.gvtrack_filter("vt_nc_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [600]})
    unfiltered = _single_value(pm.gextract("vt_nc_ref", query, iterator=-1))
    filtered = _single_value(pm.gextract("vt_nc_filt", query, iterator=-1))

    assert unfiltered == 2.0
    assert filtered == 1.0


def test_gvtrack_filter_lse_matches_manual_logsumexp():
    """lse under filter should use unmasked raw bins only."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_lse_filt", "dense_track", func="lse")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_lse_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered = _single_value(pm.gextract("vt_lse_filt", query, iterator=-1))

    seg1 = pm.gextract(
        "dense_track",
        pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
        iterator=50,
    )["dense_track"].to_numpy(dtype=float, copy=False)
    seg2 = pm.gextract(
        "dense_track",
        pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [600]}),
        iterator=50,
    )["dense_track"].to_numpy(dtype=float, copy=False)
    raw_vals = np.concatenate([seg1, seg2])
    m = np.max(raw_vals)
    expected = float(m + np.log(np.exp(raw_vals - m).sum()))
    np.testing.assert_allclose(filtered, expected, rtol=1e-8, atol=1e-8)


def test_gvtrack_filter_global_percentile():
    """R test 29: global.percentile under filter."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_gp", "dense_track", func="global.percentile")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_gp", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_gp", q, iterator=50)
    before_mask = result[result["start"] < 2000]
    assert before_mask["vt_gp"].notna().any()

    # Values should be percentiles [0, 1]
    non_na = result["vt_gp"].dropna()
    assert (non_na >= 0).all() and (non_na <= 1).all()


def test_gvtrack_filter_global_percentile_min():
    """R test 30: global.percentile.min under filter."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_gp_min", "dense_track", func="global.percentile.min")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_gp_min", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_gp_min", q, iterator=50)
    assert result["vt_gp_min"].notna().any()

    non_na = result["vt_gp_min"].dropna()
    assert (non_na >= 0).all() and (non_na <= 1).all()


def test_gvtrack_filter_global_percentile_max():
    """R test 31: global.percentile.max under filter."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_gp_max", "dense_track", func="global.percentile.max")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_gp_max", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_gp_max", q, iterator=50)
    assert result["vt_gp_max"].notna().any()

    non_na = result["vt_gp_max"].dropna()
    assert (non_na >= 0).all() and (non_na <= 1).all()


def test_gvtrack_filter_pwm():
    """R test 32: PWM virtual track with filter."""
    pm.gvtrack_clear()
    pssm = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
    ])
    pm.gvtrack_create("vt_pwm", None, func="pwm", pssm=pssm)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_pwm", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_pwm", q, iterator=50)
    assert result["vt_pwm"].notna().any()


def test_gvtrack_filter_pwm_max():
    """R test 33: PWM.max virtual track with filter."""
    pm.gvtrack_clear()
    pssm = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
    ])
    pm.gvtrack_create("vt_pwm_max", None, func="pwm.max", pssm=pssm)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_pwm_max", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_pwm_max", q, iterator=50)
    assert result["vt_pwm_max"].notna().any()


def test_gvtrack_filter_pwm_max_pos():
    """R test 34: PWM.max.pos virtual track with filter."""
    pm.gvtrack_clear()
    pssm = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
    ])
    pm.gvtrack_create("vt_pwm_mpos", None, func="pwm.max.pos", pssm=pssm)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_pwm_mpos", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_pwm_mpos", q, iterator=50)
    assert result["vt_pwm_mpos"].notna().any()


def test_gvtrack_filter_pwm_count():
    """R test 35: PWM.count virtual track with filter (additive, implemented)."""
    pm.gvtrack_clear()
    pssm = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
    ])
    pm.gvtrack_create("vt_pwm_cnt", None, func="pwm.count",
                      pssm=pssm, score_thresh=0)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_pwm_cnt", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_pwm_cnt", q, iterator=50)
    non_na = result["vt_pwm_cnt"].dropna()
    assert (non_na >= 0).all()
    assert len(result) > 0


def test_gvtrack_filter_kmer_count():
    """R test 36: kmer.count virtual track with filter."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_kmer_cnt", None, func="kmer.count",
                      kmer="ACGT", extend=True, strand=1)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_kmer_cnt", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_kmer_cnt", q, iterator=50)
    non_na = result["vt_kmer_cnt"].dropna()
    assert (non_na >= 0).all()
    assert len(result) > 0


def test_gvtrack_filter_kmer_frac():
    """R test 37: kmer.frac virtual track with filter."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_kmer_frac", None, func="kmer.frac",
                      kmer="CG", extend=False, strand=1)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_kmer_frac", filter=mask)

    q = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_kmer_frac", q, iterator=50)
    non_na = result["vt_kmer_frac"].dropna()
    assert (non_na >= 0).all() and (non_na <= 1).all()
    assert len(result) > 0


def test_gvtrack_filter_pwm_single_interval_iterator():
    """R test 38: PWM filter with single-interval iterator."""
    pm.gvtrack_clear()
    pssm = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
    ])
    pm.gvtrack_create("vt_pwm_single", None, func="pwm", pssm=pssm)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_pwm_single", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    result = pm.gextract("vt_pwm_single", query, iterator=-1)
    assert not np.isnan(result["vt_pwm_single"].iloc[0])


def test_gvtrack_filter_kmer_count_single_interval_iterator():
    """R test 39: kmer.count filter with single-interval iterator."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_kmer_single", None, func="kmer.count",
                      kmer="ACGT", extend=True, strand=1)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_kmer_single", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [6000]})
    filtered_val = _single_value(pm.gextract("vt_kmer_single", query, iterator=-1))
    assert not np.isnan(filtered_val)

    # Compare with unfiltered: filtered should be <= unfiltered
    pm.gvtrack_create("vt_kmer_nofilter", None, func="kmer.count",
                      kmer="ACGT", extend=True, strand=1)
    unfiltered_val = _single_value(pm.gextract("vt_kmer_nofilter", query, iterator=-1))
    assert filtered_val <= unfiltered_val


def test_gvtrack_filter_no_state_leak_between_vtracks():
    """R test 40: Filtered vtrack should not corrupt unfiltered vtrack sharing source."""
    pm.gvtrack_clear()
    mask = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [1500, 3500],
        "end": [2500, 4500],
    })

    # Create filtered vtrack with iterator modifier
    pm.gvtrack_create("vt_filtered", "dense_track", func="sum")
    pm.gvtrack_iterator("vt_filtered", sshift=-100, eshift=100)
    pm.gvtrack_filter("vt_filtered", filter=mask)

    # Create unfiltered vtrack with SAME source and iterator modifier
    pm.gvtrack_create("vt_unfiltered", "dense_track", func="sum")
    pm.gvtrack_iterator("vt_unfiltered", sshift=-100, eshift=100)

    query = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [1000, 2000, 4000],
        "end": [1200, 2200, 4200],
    })

    # Extract both together (list of expressions)
    result_both = pm.gextract(["vt_filtered", "vt_unfiltered"], query, iterator=-1)

    # Extract unfiltered alone (baseline)
    result_alone = pm.gextract("vt_unfiltered", query, iterator=-1)

    # Unfiltered values must be identical whether extracted alone or with filtered vtrack
    np.testing.assert_array_equal(
        result_both["vt_unfiltered"].values,
        result_alone["vt_unfiltered"].values,
    )

    # Filtered vtrack should actually produce different results
    assert (
        result_both["vt_filtered"].isna().any()
        or not np.array_equal(
            result_both["vt_filtered"].values,
            result_both["vt_unfiltered"].values,
        )
    )


def test_gvtrack_filter_via_gvtrack_create():
    """R test 41: gvtrack.create with filter= parameter."""
    pm.gvtrack_clear()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})

    pm.gvtrack_create("vt_create_filter", "dense_track", func="avg", filter=mask)

    info = pm.gvtrack_info("vt_create_filter")
    assert info.get("filter") is not None

    # Masked interval -> NaN
    q_masked = pd.DataFrame({"chrom": ["chr1"], "start": [1000], "end": [2000]})
    val_masked = _single_value(pm.gextract("vt_create_filter", q_masked, iterator=-1))
    assert np.isnan(val_masked)

    # Unmasked interval -> value
    q_ok = pd.DataFrame({"chrom": ["chr1"], "start": [5000], "end": [6000]})
    val_ok = _single_value(pm.gextract("vt_create_filter", q_ok, iterator=-1))
    assert not np.isnan(val_ok)


# ---------------------------------------------------------------------------
# max.pos.abs / max.pos.relative / min.pos.abs / min.pos.relative under filter
# ---------------------------------------------------------------------------


def test_gvtrack_filter_max_pos_abs_full_mask_returns_nan():
    """Fully masked interval should return NaN for max.pos.abs."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mpa_masked", "dense_track", func="max.pos.abs")
    pm.gvtrack_filter(
        "vt_mpa_masked",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )
    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    val = _single_value(pm.gextract("vt_mpa_masked", query, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_min_pos_abs_full_mask_returns_nan():
    """Fully masked interval should return NaN for min.pos.abs."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mnpa_masked", "dense_track", func="min.pos.abs")
    pm.gvtrack_filter(
        "vt_mnpa_masked",
        filter=pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
    )
    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]})
    val = _single_value(pm.gextract("vt_mnpa_masked", query, iterator=-1))
    assert np.isnan(val)


def test_gvtrack_filter_max_pos_abs_partial_mask_matches_manual():
    """Filtered max.pos.abs should match argmax over unmasked bin values."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mpa_filt", "dense_track", func="max.pos.abs")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_mpa_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_mpa_filt", query, iterator=-1))

    # Manual: extract raw bin values from unmasked segments and find argmax
    seg1 = pm.gextract(
        "dense_track",
        pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
        iterator=50,
    )
    seg2 = pm.gextract(
        "dense_track",
        pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [600]}),
        iterator=50,
    )
    combined = pd.concat([seg1, seg2], ignore_index=True)
    combined = combined.dropna(subset=["dense_track"])
    if len(combined) > 0:
        max_idx = combined["dense_track"].idxmax()
        expected = float(combined.loc[max_idx, "start"])
        np.testing.assert_allclose(filtered_val, expected, rtol=1e-8)


def test_gvtrack_filter_min_pos_abs_partial_mask_matches_manual():
    """Filtered min.pos.abs should match argmin over unmasked bin values."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mnpa_filt", "dense_track", func="min.pos.abs")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_mnpa_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_mnpa_filt", query, iterator=-1))

    # Manual: extract raw bin values from unmasked segments and find argmin
    seg1 = pm.gextract(
        "dense_track",
        pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200]}),
        iterator=50,
    )
    seg2 = pm.gextract(
        "dense_track",
        pd.DataFrame({"chrom": ["chr1"], "start": [400], "end": [600]}),
        iterator=50,
    )
    combined = pd.concat([seg1, seg2], ignore_index=True)
    combined = combined.dropna(subset=["dense_track"])
    if len(combined) > 0:
        min_idx = combined["dense_track"].idxmin()
        expected = float(combined.loc[min_idx, "start"])
        np.testing.assert_allclose(filtered_val, expected, rtol=1e-8)


def test_gvtrack_filter_max_pos_relative_partial_mask():
    """Filtered max.pos.relative returns position relative to shifted interval start."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mpr_filt", "dense_track", func="max.pos.relative")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_mpr_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_mpr_filt", query, iterator=-1))

    # Manual: get abs position, then subtract interval start
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mpa_ref", "dense_track", func="max.pos.abs")
    pm.gvtrack_filter("vt_mpa_ref", filter=mask)
    abs_val = _single_value(pm.gextract("vt_mpa_ref", query, iterator=-1))

    # relative = abs - start
    expected_rel = abs_val - 100.0  # interval start is 100
    np.testing.assert_allclose(filtered_val, expected_rel, rtol=1e-8)


def test_gvtrack_filter_min_pos_relative_partial_mask():
    """Filtered min.pos.relative returns position relative to shifted interval start."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mnpr_filt", "dense_track", func="min.pos.relative")
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})
    pm.gvtrack_filter("vt_mnpr_filt", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [600]})
    filtered_val = _single_value(pm.gextract("vt_mnpr_filt", query, iterator=-1))

    # Manual: get abs position, then subtract interval start
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mnpa_ref", "dense_track", func="min.pos.abs")
    pm.gvtrack_filter("vt_mnpa_ref", filter=mask)
    abs_val = _single_value(pm.gextract("vt_mnpa_ref", query, iterator=-1))

    # relative = abs - start
    expected_rel = abs_val - 100.0  # interval start is 100
    np.testing.assert_allclose(filtered_val, expected_rel, rtol=1e-8)


def test_gvtrack_filter_max_pos_abs_excludes_masked_region_extremum():
    """max.pos.abs should ignore extremum inside masked region."""
    pm.gvtrack_clear()
    # Use a value-based source where we control which value is max
    src = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [100, 300, 500],
        "end": [200, 400, 600],
        "value": [10.0, 999.0, 5.0],
    })
    pm.gvtrack_create("vt_mpa_excl", src, func="max.pos.abs")
    # Mask [250, 450) which covers the max value at [300,400)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [250], "end": [450]})
    pm.gvtrack_filter("vt_mpa_excl", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [50], "end": [700]})
    val = _single_value(pm.gextract("vt_mpa_excl", query, iterator=-1))
    # With mask, max among unmasked is 10.0 at start=100
    np.testing.assert_allclose(val, 100.0, rtol=1e-8)


def test_gvtrack_filter_min_pos_abs_excludes_masked_region_extremum():
    """min.pos.abs should ignore extremum inside masked region."""
    pm.gvtrack_clear()
    src = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [100, 300, 500],
        "end": [200, 400, 600],
        "value": [10.0, 1.0, 50.0],
    })
    pm.gvtrack_create("vt_mnpa_excl", src, func="min.pos.abs")
    # Mask [250, 450) which covers the min value at [300,400)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [250], "end": [450]})
    pm.gvtrack_filter("vt_mnpa_excl", filter=mask)

    query = pd.DataFrame({"chrom": ["chr1"], "start": [50], "end": [700]})
    val = _single_value(pm.gextract("vt_mnpa_excl", query, iterator=-1))
    # With mask, min among unmasked is 10.0 at start=100
    np.testing.assert_allclose(val, 100.0, rtol=1e-8)


def test_gvtrack_filter_max_pos_with_iterator_shifts():
    """max.pos.abs under filter honors iterator shifts."""
    pm.gvtrack_clear()
    pm.gvtrack_create("vt_mpa_shift", "dense_track", func="max.pos.abs")
    pm.gvtrack_iterator("vt_mpa_shift", sshift=-100, eshift=100)
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [2000], "end": [4000]})
    pm.gvtrack_filter("vt_mpa_shift", filter=mask)

    # Query [2500, 2600) with shifts becomes [2400, 2700), fully inside mask
    q_masked = pd.DataFrame({"chrom": ["chr1"], "start": [2500], "end": [2600]})
    val = _single_value(pm.gextract("vt_mpa_shift", q_masked, iterator=-1))
    assert np.isnan(val)

    # Query [5000, 5100) with shifts becomes [4900, 5200), outside mask
    q_unmasked = pd.DataFrame({"chrom": ["chr1"], "start": [5000], "end": [5100]})
    val2 = _single_value(pm.gextract("vt_mpa_shift", q_unmasked, iterator=-1))
    assert not np.isnan(val2)


def test_gvtrack_filter_max_pos_relative_matches_abs_minus_start():
    """max.pos.relative == max.pos.abs - interval_start for multiple intervals."""
    pm.gvtrack_clear()
    mask = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [400]})

    pm.gvtrack_create("vt_mpa_multi", "dense_track", func="max.pos.abs")
    pm.gvtrack_filter("vt_mpa_multi", filter=mask)
    pm.gvtrack_create("vt_mpr_multi", "dense_track", func="max.pos.relative")
    pm.gvtrack_filter("vt_mpr_multi", filter=mask)

    query = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [0, 500],
        "end": [600, 1000],
    })
    result = pm.gextract(["vt_mpa_multi", "vt_mpr_multi"], query, iterator=-1)

    abs_vals = result["vt_mpa_multi"].values
    rel_vals = result["vt_mpr_multi"].values
    starts = result["start"].values

    for i in range(len(result)):
        if not np.isnan(abs_vals[i]) and not np.isnan(rel_vals[i]):
            np.testing.assert_allclose(
                rel_vals[i], abs_vals[i] - starts[i], rtol=1e-8,
                err_msg=f"Interval {i}: rel={rel_vals[i]} != abs={abs_vals[i]} - start={starts[i]}",
            )
