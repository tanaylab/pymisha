import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pymisha as pm

TESTDB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _run_r_df(r_body):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as script_fd:
        script_path = script_fd.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as out_fd:
        out_path = out_fd.name

    r_script = f"""
library(misha)
gdb.init(\"{TESTDB}\")
{r_body}
names(df) <- gsub(\" \", \"_\", names(df))
write.table(df, file=\"{out_path}\", sep='\t', row.names=FALSE, quote=FALSE)
"""

    try:
        with open(script_path, "w") as handle:
            handle.write(r_script)

        result = subprocess.run(
            ["R", "--quiet", "--no-save", "-f", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R failed: {result.stderr}\n{result.stdout}")

        return pd.read_csv(out_path, sep="\t")
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


class TestVTrackFunctions:
    """Test virtual track function correctness against R misha."""

    def setup_method(self):
        pm.gdb_init(str(TESTDB))
        pm.gvtrack_clear()

    def teardown_method(self):
        pm.gvtrack_clear()

    def test_avg_function(self):
        """avg function matches R."""
        intervs = pm.gintervals("1", [0], [500])
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        py_res = pm.gextract("vt_avg", intervs, iterator=100)

        r_df = _run_r_df(
            """
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_avg", "dense_track", func="avg")
df <- gextract("vt_avg", intervs, iterator=100)
"""
        )

        np.testing.assert_allclose(
            py_res["vt_avg"].values, r_df["vt_avg"].values, rtol=1e-6
        )

    def test_max_function(self):
        """max function matches R."""
        intervs = pm.gintervals("1", [0], [500])
        pm.gvtrack_create("vt_max", "dense_track", func="max")
        py_res = pm.gextract("vt_max", intervs, iterator=100)

        r_df = _run_r_df(
            """
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_max", "dense_track", func="max")
df <- gextract("vt_max", intervs, iterator=100)
"""
        )

        np.testing.assert_allclose(
            py_res["vt_max"].values, r_df["vt_max"].values, rtol=1e-6
        )

    def test_min_function(self):
        """min function matches R."""
        intervs = pm.gintervals("1", [0], [500])
        pm.gvtrack_create("vt_min", "dense_track", func="min")
        py_res = pm.gextract("vt_min", intervs, iterator=100)

        r_df = _run_r_df(
            """
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_min", "dense_track", func="min")
df <- gextract("vt_min", intervs, iterator=100)
"""
        )

        np.testing.assert_allclose(
            py_res["vt_min"].values, r_df["vt_min"].values, rtol=1e-6
        )

    def test_sum_function(self):
        """sum function matches R."""
        intervs = pm.gintervals("1", [0], [500])
        pm.gvtrack_create("vt_sum", "dense_track", func="sum")
        py_res = pm.gextract("vt_sum", intervs, iterator=100)

        r_df = _run_r_df(
            """
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_sum", "dense_track", func="sum")
df <- gextract("vt_sum", intervs, iterator=100)
"""
        )

        np.testing.assert_allclose(
            py_res["vt_sum"].values, r_df["vt_sum"].values, rtol=1e-6
        )

    def test_stddev_function(self):
        """stddev function matches R."""
        intervs = pm.gintervals("1", [0], [500])
        pm.gvtrack_create("vt_std", "dense_track", func="stddev")
        py_res = pm.gextract("vt_std", intervs, iterator=100)

        r_df = _run_r_df(
            """
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_std", "dense_track", func="stddev")
df <- gextract("vt_std", intervs, iterator=100)
"""
        )

        # NaN positions must match
        py_nan = np.isnan(py_res["vt_std"].values)
        r_nan = np.isnan(r_df["vt_std"].values)
        np.testing.assert_array_equal(py_nan, r_nan)

        # stddev is sensitive to float32 track precision, so use relaxed
        # tolerances: rtol for relative accuracy, atol for near-zero values
        # where float rounding can produce 0.0 vs a tiny positive number.
        valid = ~py_nan
        np.testing.assert_allclose(
            py_res["vt_std"].values[valid], r_df["vt_std"].values[valid],
            rtol=1e-4, atol=1e-4
        )

    def test_quantile_function(self):
        """quantile function matches R."""
        intervs = pm.gintervals("1", [0], [500])
        pm.gvtrack_create("vt_q", "dense_track", func="quantile", params=[0.9])
        py_res = pm.gextract("vt_q", intervs, iterator=100)

        r_df = _run_r_df(
            """
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_q", "dense_track", func="quantile", params=0.9)
df <- gextract("vt_q", intervs, iterator=100)
"""
        )

        np.testing.assert_allclose(
            py_res["vt_q"].values, r_df["vt_q"].values, rtol=1e-6
        )

    def test_value_based_vtrack(self):
        """Value-based virtual track matches R."""
        intervs = pm.gintervals("1", [0], [500])

        # Create a dataframe for value-based track
        src_df = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [100, 200, 300],
            "end": [150, 250, 350],
            "score": [1.0, 5.0, 3.0]
        })

        pm.gvtrack_create("vt_val", src_df, func="max")
        py_res = pm.gextract("vt_val", intervs, iterator=100)

        r_df = _run_r_df(
            """
src <- data.frame(
    chrom = c("chr1", "chr1", "chr1"),
    start = c(100, 200, 300),
    end = c(150, 250, 350),
    score = c(1, 5, 3)
)
intervs <- gintervals("1", c(0), c(500))
gvtrack.create("vt_val", src, "max")
df <- gextract("vt_val", intervs, iterator=100)
"""
        )

        # NaNs match
        assert np.array_equal(np.isnan(py_res["vt_val"]), np.isnan(r_df["vt_val"]))

        # Values match
        valid = ~np.isnan(py_res["vt_val"])
        np.testing.assert_allclose(
            py_res.loc[valid, "vt_val"].values,
            r_df.loc[valid, "vt_val"].values,
            rtol=1e-6
        )

    def test_value_based_vtrack_overlapping(self):
        """Value-based virtual track with overlaps matches R."""
        # Overlapping intervals: [0-100, score=1], [200-300, score=5], [400-500, score=3]
        # Query: [0-250] (hits 1 and 2), [250-550] (hits 2 and 3)
        src_df = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [0, 200, 400],
            "end": [100, 300, 500],
            "score": [1.0, 5.0, 3.0]
        })

        pm.gvtrack_create("vt_val", src_df, func="max")
        # Use query intervals that overlap multiple source intervals
        query = pm.gintervals("1", [0, 250], [250, 550])
        py_res = pm.gextract("vt_val", query, iterator=0)

        r_df = _run_r_df(
            """
src <- data.frame(chrom = c("chr1", "chr1", "chr1"), start = c(0, 200, 400), end = c(100, 300, 500), score = c(1, 5, 3))
intervs <- gintervals("1", c(0, 250), c(250, 550))
gvtrack.create("vt_val", src, "max")
df <- gextract("vt_val", intervs, iterator=intervs)
"""
        )

        # NaNs match
        assert np.array_equal(np.isnan(py_res["vt_val"]), np.isnan(r_df["vt_val"]))

        # Values match
        valid = ~np.isnan(py_res["vt_val"])
        np.testing.assert_allclose(
            py_res.loc[valid, "vt_val"].values,
            r_df.loc[valid, "vt_val"].values,
            rtol=1e-6
        )
