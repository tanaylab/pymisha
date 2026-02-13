"""Tests for glookup C++ streaming backend (pm_lookup).

These tests verify that the C++ streaming path produces identical results
to the Python memory-resident path, and exercises force_binning, include_lowest,
multi-dimensional lookup, NaN handling, and multitasking.
"""

import numpy as np
import pandas as pd

import pymisha as pm
from pymisha._shared import _pymisha


class TestPmLookupExists:
    """Verify the C++ pm_lookup function is registered."""

    def test_pm_lookup_is_callable(self):
        """pm_lookup must exist in the C++ extension module."""
        assert hasattr(_pymisha, "pm_lookup"), "pm_lookup not found in _pymisha C++ module"
        assert callable(_pymisha.pm_lookup)


def _glookup_python(lookup_table, *args, intervals=None, include_lowest=False,
                    force_binning=True, iterator=None):
    """Force Python fallback path for glookup by using the pure-Python logic."""
    from pymisha.extract import gextract

    lookup_table = np.asarray(lookup_table)

    exprs = []
    breaks_list = []
    for i in range(0, len(args), 2):
        exprs.append(args[i])
        breaks_list.append(np.asarray(args[i + 1], dtype=float))

    # Extract values
    all_values = []
    extract_result = None
    for i, expr in enumerate(exprs):
        result = gextract(expr, intervals, iterator=iterator)
        if result is None or len(result) == 0:
            return None
        data_cols = [c for c in result.columns if c not in {"chrom", "start", "end", "intervalID"}]
        values = result[data_cols[0]].to_numpy(dtype=float, copy=False)
        all_values.append(values)
        if i == 0:
            extract_result = result

    n_values = len(all_values[0])

    # Bin each expression
    bin_indices = []
    for values, breaks in zip(all_values, breaks_list, strict=False):
        n_bin = len(breaks) - 1
        indices = np.searchsorted(breaks, values, side='right') - 1

        if include_lowest:
            at_lowest = values == breaks[0]
            indices[at_lowest] = 0

        below_min = values < breaks[0]
        if not include_lowest:
            at_min = values == breaks[0]
            below_min = below_min | at_min
        above_max = values > breaks[-1]

        if force_binning:
            indices[below_min] = 0
            indices[above_max] = n_bin - 1
        else:
            indices[below_min] = -1
            indices[above_max] = -1

        nan_mask = np.isnan(values)
        indices[nan_mask] = -1
        bin_indices.append(indices)

    # Lookup
    output_values = np.full(n_values, np.nan, dtype=float)
    valid = np.ones(n_values, dtype=bool)
    for indices in bin_indices:
        valid &= (indices >= 0)

    if valid.any():
        if len(exprs) == 1:
            output_values[valid] = lookup_table[bin_indices[0][valid]]
        else:
            for j in range(n_values):
                if not valid[j]:
                    continue
                coord = tuple(idx[j] for idx in bin_indices)
                output_values[j] = lookup_table[coord]

    result_df = extract_result[["chrom", "start", "end", "intervalID"]].copy()
    result_df["value"] = output_values
    return result_df


class TestGlookupStreamingParity:
    """Verify C++ streaming path matches Python reference for all cases."""

    def test_1d_basic_parity(self):
        """1D lookup: C++ path matches Python reference."""
        lookup_table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        breaks = np.linspace(0.1, 0.2, 6)
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals)
        expected = _glookup_python(lookup_table, "dense_track", breaks,
                                   intervals=intervals)

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        np.testing.assert_array_equal(result["chrom"].values, expected["chrom"].values)
        np.testing.assert_array_equal(result["start"].values, expected["start"].values)
        np.testing.assert_array_equal(result["end"].values, expected["end"].values)
        np.testing.assert_array_equal(result["intervalID"].values, expected["intervalID"].values)
        # Compare values (NaN == NaN for this purpose)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_1d_with_iterator_parity(self):
        """1D lookup with explicit iterator: C++ matches Python."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        breaks = np.array([0.0, 0.15, 0.3, 0.45])
        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals, iterator=50)
        expected = _glookup_python(lookup_table, "dense_track", breaks,
                                   intervals=intervals, iterator=50)

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_2d_parity(self):
        """2D lookup: C++ matches Python."""
        lookup_table = np.arange(1, 16, dtype=float).reshape((5, 3))
        breaks1 = np.linspace(0.1, 0.2, 6)
        breaks2 = np.linspace(0.31, 0.37, 4)
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(
            lookup_table, "dense_track", breaks1, "2 * dense_track", breaks2,
            intervals=intervals
        )
        expected = _glookup_python(
            lookup_table, "dense_track", breaks1, "2 * dense_track", breaks2,
            intervals=intervals
        )

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_force_binning_true_parity(self):
        """force_binning=True clamps out-of-range: C++ matches Python."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        breaks = np.array([0.14, 0.15, 0.16, 0.17])
        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals, force_binning=True)
        expected = _glookup_python(lookup_table, "dense_track", breaks,
                                   intervals=intervals, force_binning=True)

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_force_binning_false_parity(self):
        """force_binning=False produces NaN for out-of-range: C++ matches Python."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        breaks = np.array([0.14, 0.145, 0.15, 0.155])
        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals, force_binning=False)
        expected = _glookup_python(lookup_table, "dense_track", breaks,
                                   intervals=intervals, force_binning=False)

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_include_lowest_parity(self):
        """include_lowest=True: C++ matches Python."""
        lookup_table = np.array([100.0, 200.0, 300.0])
        breaks = np.array([0.1, 0.15, 0.2, 0.25])
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals, include_lowest=True)
        expected = _glookup_python(lookup_table, "dense_track", breaks,
                                   intervals=intervals, include_lowest=True)

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_nan_expression_values_parity(self):
        """NaN values in expression produce NaN: C++ matches Python."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        breaks = np.array([0.0, 0.5, 1.0, 1.5])
        intervals = pm.gintervals("1", 0, 10000)

        result = pm.glookup(lookup_table, "sparse_track", breaks,
                            intervals=intervals)
        expected = _glookup_python(lookup_table, "sparse_track", breaks,
                                   intervals=intervals)

        assert result is not None
        assert expected is not None
        assert len(result) == len(expected)
        r_vals = result["value"].values
        e_vals = expected["value"].values
        np.testing.assert_array_equal(np.isnan(r_vals), np.isnan(e_vals))
        valid = ~np.isnan(r_vals)
        if valid.any():
            np.testing.assert_allclose(r_vals[valid], e_vals[valid])

    def test_empty_intervals_returns_none(self):
        """Empty intervals returns None."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        breaks = np.array([0.0, 0.1, 0.2, 0.3])
        intervals = pd.DataFrame(columns=["chrom", "start", "end"])

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals)
        assert result is None

    def test_result_columns(self):
        """Result has expected columns: chrom, start, end, value, intervalID."""
        lookup_table = np.array([10.0, 20.0, 30.0])
        breaks = np.array([0.0, 0.15, 0.3, 0.45])
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals)

        assert result is not None
        assert set(result.columns) == {"chrom", "start", "end", "value", "intervalID"}

    def test_all_values_from_lookup_table(self):
        """All non-NaN values come from the lookup table."""
        lookup_table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        breaks = np.linspace(0.1, 0.2, 6)
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals)

        assert result is not None
        vals = result["value"].dropna().values
        for v in vals:
            assert v in lookup_table

    def test_force_binning_true_no_nan(self):
        """With force_binning=True, only NaN expression values produce NaN output."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        breaks = np.array([0.0, 0.15, 0.3, 0.45])
        intervals = pm.gintervals("1", 0, 200)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals, force_binning=True)

        assert result is not None
        # dense_track shouldn't have NaN on chr1:0-200, so no NaN expected
        assert not result["value"].isna().any()

    def test_force_binning_false_has_nan(self):
        """With force_binning=False, out-of-range values produce NaN."""
        lookup_table = np.array([1.0, 2.0, 3.0])
        # Very narrow breaks - most values will be out of range
        breaks = np.array([0.14, 0.145, 0.15, 0.155])
        intervals = pm.gintervals("1", 0, 500)

        result = pm.glookup(lookup_table, "dense_track", breaks,
                            intervals=intervals, force_binning=False)

        assert result is not None
        assert result["value"].isna().any()

    def test_multitask_parity(self):
        """Multitask results match single-task results."""
        lookup_table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        breaks = np.linspace(0.1, 0.2, 6)
        intervals = pm.gintervals_all()

        # Run with multitask disabled
        old_mt = pm.CONFIG.get("multitasking", True)
        try:
            pm.CONFIG["multitasking"] = False
            result_single = pm.glookup(lookup_table, "dense_track", breaks,
                                       intervals=intervals, iterator=100)

            pm.CONFIG["multitasking"] = True
            result_multi = pm.glookup(lookup_table, "dense_track", breaks,
                                      intervals=intervals, iterator=100)
        finally:
            pm.CONFIG["multitasking"] = old_mt

        assert result_single is not None
        assert result_multi is not None
        assert len(result_single) == len(result_multi)

        # Sort both by chrom/start for comparison
        r1 = result_single.sort_values(["chrom", "start"]).reset_index(drop=True)
        r2 = result_multi.sort_values(["chrom", "start"]).reset_index(drop=True)

        np.testing.assert_array_equal(r1["chrom"].values, r2["chrom"].values)
        np.testing.assert_array_equal(r1["start"].values, r2["start"].values)
        np.testing.assert_array_equal(r1["end"].values, r2["end"].values)

        v1 = r1["value"].values
        v2 = r2["value"].values
        np.testing.assert_array_equal(np.isnan(v1), np.isnan(v2))
        valid = ~np.isnan(v1)
        np.testing.assert_allclose(v1[valid], v2[valid])
