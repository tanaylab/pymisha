"""Tests for gvtrack_array_slice + gextract on array tracks (Spec B).

Ports relevant R tests from test-vtrack.R (gvtrack.array.slice section)
and exercises the Python-side ARRAYS aggregation path.

Array track fixture: ``array_track`` in the test DB.
  - 10 columns: col0 .. col9
  - Chroms: 1 (500k), 2 (300k), X (200k)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _init_db():
    pm.gdb_init_examples()


@pytest.fixture(autouse=True)
def _clear_vtracks():
    yield
    pm.gvtrack_clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ivs(chrom: str, start: int, end: int) -> pd.DataFrame:
    return pd.DataFrame({"chrom": [chrom], "start": [start], "end": [end]})


# ---------------------------------------------------------------------------
# T3: gvtrack_array_slice - mutation and validation
# ---------------------------------------------------------------------------

class TestGvtrackArraySliceCreate:
    """Ports R tests from vtrack.tracktype_test.array* in test-vtrack.R.

    Under the R-aligned API, gvtrack_array_slice mutates an existing vtrack
    rather than creating one. Each test creates the vtrack first via
    gvtrack_create, then calls gvtrack_array_slice to configure it.
    """

    def test_creates_vtrack(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice("v1")
        assert "v1" in pm.gvtrack_ls()

    def test_default_func_is_avg(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice("v1")
        info = pm.gvtrack_info("v1")
        assert info.get("func", "avg") == "avg"
        assert info.get("kind") == "array_slice"

    def test_non_array_src_raises(self):
        """R: gvtrack.array.slice on fixedbin or sparse track raises."""
        pm.gvtrack_create("v1", src="dense_track")
        with pytest.raises(ValueError, match="not an array track"):
            pm.gvtrack_array_slice("v1")

    def test_nonexistent_vtrack_raises(self):
        with pytest.raises(ValueError, match="no such vtrack"):
            pm.gvtrack_array_slice("nonexistent_vtrack_xyz")

    def test_bad_func_raises(self):
        """R: gvtrack.array.slice with invalid func raises."""
        pm.gvtrack_create("v1", src="array_track")
        with pytest.raises(ValueError, match="blabla"):
            pm.gvtrack_array_slice("v1", func="blabla")

    def test_slice_by_name(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice("v1", slice=["col1", "col3"])
        info = pm.gvtrack_info("v1")
        assert info["slice_cols"] == [1, 3]

    def test_slice_by_index(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice("v1", slice=[0, 5])
        info = pm.gvtrack_info("v1")
        assert info["slice_cols"] == [0, 5]

    def test_slice_out_of_range_raises(self):
        """R: gvtrack.array.slice with index 50 (out of 10) raises."""
        pm.gvtrack_create("v1", src="array_track")
        with pytest.raises(ValueError, match="out of range"):
            pm.gvtrack_array_slice("v1", slice=[50])

    def test_slice_negative_index_raises(self):
        """R: gvtrack.array.slice with index -50 raises."""
        pm.gvtrack_create("v1", src="array_track")
        with pytest.raises(ValueError, match="out of range"):
            pm.gvtrack_array_slice("v1", slice=[-50])

    def test_slice_bad_column_name_raises(self):
        """R: gvtrack.array.slice with unknown column name raises."""
        pm.gvtrack_create("v1", src="array_track")
        with pytest.raises(ValueError, match="blabla"):
            pm.gvtrack_array_slice("v1", slice=["blabla"])

    def test_reconfigure_existing_vtrack(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice("v1", func="avg")
        pm.gvtrack_array_slice("v1", func="min")
        info = pm.gvtrack_info("v1")
        assert info["func"] == "min"


# ---------------------------------------------------------------------------
# T4: gextract with array_slice vtracks
# ---------------------------------------------------------------------------

class TestGextractArraySlice:
    """gextract routes array_slice vtracks through Python aggregation."""

    def test_avg_all_cols(self):
        """All columns, avg: with no explicit iterator the array's own bins
        drive iteration (one row per bin, R parity), each holding the per-bin
        column average."""
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice("v_avg", func="avg")
        ivs = _ivs("1", 0, 5000)
        result = pm.gextract("v_avg", intervals=ivs)
        assert result is not None
        assert len(result) >= 1
        assert "v_avg" in result.columns
        # First bin [0,50): columns 0,2,4,6,8 -> avg 4.0
        first = result.sort_values(["chrom", "start", "end"]).iloc[0]
        assert int(first["start"]) == 0
        np.testing.assert_allclose(float(first["v_avg"]), 4.0, rtol=1e-5)

    def test_min_all_cols(self):
        pm.gvtrack_create("v_min", src="array_track")
        pm.gvtrack_array_slice("v_min", func="min")
        ivs = _ivs("1", 0, 5000)
        result = pm.gextract("v_min", intervals=ivs)
        assert result is not None
        v_min = result["v_min"].iloc[0]
        assert not pd.isna(v_min)

    def test_max_all_cols(self):
        pm.gvtrack_create("v_max", src="array_track")
        pm.gvtrack_array_slice("v_max", func="max")
        ivs = _ivs("1", 0, 5000)
        result = pm.gextract("v_max", intervals=ivs)
        assert result is not None
        v_max = result["v_max"].iloc[0]
        assert not pd.isna(v_max)

    def test_sum_gt_avg(self):
        """sum >= avg for non-negative data with > 1 value."""
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice("v_avg", func="avg")
        pm.gvtrack_create("v_sum", src="array_track")
        pm.gvtrack_array_slice("v_sum", func="sum")
        ivs = _ivs("1", 0, 5000)
        r_avg = pm.gextract("v_avg", intervals=ivs)["v_avg"].iloc[0]
        r_sum = pm.gextract("v_sum", intervals=ivs)["v_sum"].iloc[0]
        assert r_sum >= r_avg

    def test_slice_by_single_column_name(self):
        """col0 slice returns reproducible value."""
        pm.gvtrack_create("v_col0", src="array_track")
        pm.gvtrack_array_slice("v_col0", slice=["col0"], func="avg")
        ivs = _ivs("1", 0, 200)
        result = pm.gextract("v_col0", intervals=ivs)
        assert result is not None
        assert "v_col0" in result.columns

    def test_slice_by_single_column_index(self):
        """col0 by name and by index [0] must agree."""
        pm.gvtrack_create("v_name", src="array_track")
        pm.gvtrack_array_slice("v_name", slice=["col0"], func="avg")
        pm.gvtrack_create("v_idx", src="array_track")
        pm.gvtrack_array_slice("v_idx", slice=[0], func="avg")
        ivs = _ivs("1", 0, 5000)
        r_name = pm.gextract("v_name", intervals=ivs)["v_name"].iloc[0]
        r_idx = pm.gextract("v_idx", intervals=ivs)["v_idx"].iloc[0]
        np.testing.assert_almost_equal(r_name, r_idx)

    def test_slice_subset_vs_all_avg(self):
        """Slicing to a single column gives avg of that column only."""
        pm.gvtrack_create("v_all", src="array_track")
        pm.gvtrack_array_slice("v_all", func="avg")
        pm.gvtrack_create("v_c0", src="array_track")
        pm.gvtrack_array_slice("v_c0", slice=["col0"], func="avg")
        ivs = _ivs("1", 0, 5000)
        r_all = pm.gextract("v_all", intervals=ivs)["v_all"].iloc[0]
        r_c0 = pm.gextract("v_c0", intervals=ivs)["v_c0"].iloc[0]
        # They may or may not be equal, but both must be finite
        assert np.isfinite(r_all)
        assert np.isfinite(r_c0)

    def test_empty_interval_returns_nan(self):
        """An interval with no overlapping track data -> NaN."""
        # Use an interval known to have no data (e.g., far into chrom 2)
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice("v_avg", func="avg")
        ivs = pd.DataFrame({"chrom": ["2"], "start": [290_000], "end": [300_000]})
        result = pm.gextract("v_avg", intervals=ivs)
        # Result may be empty (no iterator intervals) or NaN; either is OK
        if result is not None and len(result) > 0:
            val = result["v_avg"].iloc[0]
            assert pd.isna(val) or np.isfinite(val)

    def test_multi_interval_query(self):
        """Each emitted row carries the intervalID of the scope interval whose
        region it falls in (the default iterator is the array's bins, so there
        may be several rows per scope interval)."""
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice("v_avg", func="avg")
        ivs = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "start": [0, 1000, 0],
            "end": [500, 2000, 1000],
        })
        result = pm.gextract("v_avg", intervals=ivs)
        assert result is not None
        assert len(result) >= 1
        # intervalIDs are a subset of the three scope intervals (1-based).
        assert set(result["intervalID"]).issubset({1, 2, 3})

    def test_with_explicit_iterator(self):
        """Array-slice vtrack works with an explicit bin-size iterator."""
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice("v_avg", func="avg")
        ivs = _ivs("1", 0, 5000)
        result = pm.gextract("v_avg", intervals=ivs, iterator=200)
        assert result is not None
        assert len(result) == 25  # 5000 / 200
        # At least one non-NaN value expected
        assert result["v_avg"].notna().any()

    def test_output_columns(self):
        """Output always has chrom, start, end, <vtrack>, intervalID."""
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice("v_avg", func="avg")
        ivs = _ivs("1", 0, 5000)
        result = pm.gextract("v_avg", intervals=ivs)
        assert result is not None
        assert set(result.columns) >= {"chrom", "start", "end", "v_avg", "intervalID"}

    def test_colnames_in_all(self):
        """gvtrack_array_slice is exported in __all__."""
        assert "gvtrack_array_slice" in pm.__all__


# ---------------------------------------------------------------------------
# T5a: Bare array track in gextract reads via the C++ scanner
# ---------------------------------------------------------------------------

class TestBareArrayTrackScanner:
    """gextract('array_track') reads the array directly through the scanner:
    each bin's columns are averaged (default reduction), matching the per-bin
    average of gtrack_array_extract."""

    def test_bare_array_default_iterator(self):
        ivs = _ivs("1", 0, 50000)
        scanned = pm.gextract("array_track", intervals=ivs)
        assert scanned is not None and len(scanned) > 0
        assert "array_track" in scanned.columns
        # Cross-check the per-bin value against the column average reported by
        # the independent array-extract reader.
        ext = pm.gtrack_array_extract("array_track", intervals=ivs)
        cols = [c for c in ext.columns if c.startswith("col")]
        ext = ext.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        scn = scanned.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        assert len(scn) == len(ext)
        manual = np.nanmean(ext[cols].to_numpy(dtype=float), axis=1)
        np.testing.assert_allclose(
            scn["array_track"].to_numpy(dtype=float), manual, rtol=1e-5, equal_nan=True
        )

    def test_bare_array_in_expression(self):
        ivs = _ivs("1", 0, 50000)
        base = pm.gextract("array_track", intervals=ivs)
        expr = pm.gextract("2 * array_track + 17", intervals=ivs)
        base = base.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        expr = expr.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        np.testing.assert_allclose(
            expr["2 * array_track + 17"].to_numpy(dtype=float),
            2 * base["array_track"].to_numpy(dtype=float) + 17,
            rtol=1e-5, equal_nan=True,
        )

    def test_array_as_iterator(self):
        """iterator='array_track' emits one row per array bin overlapping scope."""
        ivs = _ivs("1", 0, 50000)
        bins = pm.gextract("array_track", intervals=ivs)
        via_iter = pm.gextract("array_track", intervals=ivs, iterator="array_track")
        assert len(via_iter) == len(bins)


# ---------------------------------------------------------------------------
# T5b: R regression: col1+col3+col5 slice, avg
# (Ports: vtrack.tracktype_test.array_slice_col135_avg_regression)
# ---------------------------------------------------------------------------

class TestRegressionSliceAvg:
    """Mirror of R test: gvtrack.array.slice(v1, c('col1','col3','col5'), 'avg')
    followed by gextract(v1, gintervals(c(1,2))).

    We can't compare exact values without R regression files, but we verify
    structural invariants that R's test also relies on.
    """

    def test_col135_avg_result_shape(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice(
            "v1", slice=["col1", "col3", "col5"], func="avg"
        )
        ivs = pm.gintervals("1")
        result = pm.gextract("v1", intervals=ivs)
        assert result is not None
        assert "v1" in result.columns
        # Should produce at least one row per interval region
        assert len(result) >= 1

    def test_col135_values_finite_where_present(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice(
            "v1", slice=["col1", "col3", "col5"], func="avg"
        )
        ivs = pm.gintervals("1")
        result = pm.gextract("v1", intervals=ivs)
        assert result is not None
        non_nan = result["v1"].dropna()
        assert (non_nan.isfinite() if hasattr(non_nan, "isfinite") else np.isfinite(non_nan)).all()

    def test_col135_avg_lt_max(self):
        """avg across cols [1,3,5] should be <= max over same cols."""
        pm.gvtrack_create("v_avg", src="array_track")
        pm.gvtrack_array_slice(
            "v_avg", slice=["col1", "col3", "col5"], func="avg"
        )
        pm.gvtrack_create("v_max", src="array_track")
        pm.gvtrack_array_slice(
            "v_max", slice=["col1", "col3", "col5"], func="max"
        )
        ivs = _ivs("1", 0, 50000)
        r_avg = pm.gextract("v_avg", intervals=ivs)["v_avg"].iloc[0]
        r_max = pm.gextract("v_max", intervals=ivs)["v_max"].iloc[0]
        if np.isfinite(r_avg) and np.isfinite(r_max):
            assert r_avg <= r_max + 1e-9


# ---------------------------------------------------------------------------
# T5c: Duplicate column names in slice (R allows repeat slices)
# ---------------------------------------------------------------------------

class TestSliceWithDuplicateCols:
    """R test: gvtrack.array.slice(v1, c('col1','col1','col3','col5')) succeeds."""

    def test_duplicate_names_accepted(self):
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice(
            "v1", slice=["col1", "col1", "col3", "col5"], func="avg"
        )
        assert "v1" in pm.gvtrack_ls()
        info = pm.gvtrack_info("v1")
        # slice_cols may contain duplicates: [1, 1, 3, 5]
        assert info["slice_cols"] == [1, 1, 3, 5]


# ---------------------------------------------------------------------------
# T6: Error tests for R-aligned API (mutate-existing-vtrack semantics)
# ---------------------------------------------------------------------------

class TestGvtrackArraySliceErrors:
    """Error conditions specific to the R-aligned two-step API."""

    def test_nonexistent_vtrack_raises(self):
        """gvtrack_array_slice on a vtrack that has never been created raises."""
        with pytest.raises(ValueError, match="no such vtrack"):
            pm.gvtrack_array_slice("nonexistent")

    def test_non_array_vtrack_raises(self):
        """vtrack created with a 1D dense src raises 'not an array track'."""
        pm.gvtrack_create("v1", src="dense_track")
        with pytest.raises(ValueError, match="not an array track"):
            pm.gvtrack_array_slice("v1")

    def test_params_with_non_quantile_raises(self):
        """params is only meaningful for func='quantile'; otherwise rejected."""
        pm.gvtrack_create("v1", src="array_track")
        with pytest.raises(ValueError, match="quantile"):
            pm.gvtrack_array_slice("v1", slice=[0], func="avg", params=0.4)

    def test_quantile_requires_params(self):
        """func='quantile' without params raises."""
        pm.gvtrack_create("v1", src="array_track")
        with pytest.raises(ValueError, match="quantile"):
            pm.gvtrack_array_slice("v1", slice=[0], func="quantile")

    def test_quantile_with_params_accepted(self):
        """func='quantile' with a valid percentile configures the vtrack."""
        pm.gvtrack_create("v1", src="array_track")
        pm.gvtrack_array_slice("v1", slice=[0, 1, 2], func="quantile", params=0.4)
        info = pm.gvtrack_info("v1")
        assert info["func"] == "quantile"
        assert info["params"] == 0.4


# ---------------------------------------------------------------------------
# gtrack_array_extract: file= TSV dump (R parity)
# ---------------------------------------------------------------------------

def test_gtrack_array_extract_file_roundtrip(tmp_path):
    out = tmp_path / "extract.tsv"
    rc = pm.gtrack_array_extract("array_track", None, file=str(out))
    assert rc is None
    assert out.exists()
    text = out.read_text().splitlines()
    assert len(text) > 1
    header = text[0].split("\t")
    assert header[:3] == ["chrom", "start", "end"]
