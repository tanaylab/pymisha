"""Golden-master parity: BARE physical track extraction with a coarsening
numeric iterator, compared to live R misha.

This path (a plain track name in the expression, evaluated per output bin via
PMTrackExpressionVars) had NO value-level R-parity coverage: the existing
golden-master iterator tests all used *virtual* tracks (vt_avg/vt_max/...),
which take a different C++ read path. As a result a bug where the dense-track
read point-sampled the interval midpoint instead of averaging the covered
native bins went undetected. R's own suite covers this via
test-gextract3.R (`gextract("test.fixedbin", ..., iterator=120)` checked with
expect_regression), but that test was never ported.

These tests mirror R's behaviour directly so the bare-track averaging is
pinned against the reference implementation.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

if shutil.which("R") is None:
    pytest.skip("R not available; skipping golden-master iterator tests", allow_module_level=True)

TESTDB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _run_r_df(r_body):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as script_fd:
        script_path = script_fd.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as out_fd:
        out_path = out_fd.name
    r_script = f"""
library(misha)
gdb.init("{TESTDB}")
{r_body}
names(df) <- gsub(" ", "_", names(df))
write.table(df, file="{out_path}", sep='\t', row.names=FALSE, quote=FALSE)
"""
    try:
        with open(script_path, "w") as handle:
            handle.write(r_script)
        result = subprocess.run(
            ["R", "--quiet", "--no-save", "-f", script_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R failed: {result.stderr}\n{result.stdout}")
        import pandas as pd
        return pd.read_csv(out_path, sep="\t")
    finally:
        for p in (script_path, out_path):
            if os.path.exists(p):
                os.unlink(p)


class TestBareTrackCoarseningIterator:
    """gextract on a plain physical track with a numeric iterator coarser than
    the track's native bin size must AVERAGE the covered native bins (R parity)."""

    def setup_method(self):
        pm.gdb_init(str(TESTDB))

    @pytest.mark.parametrize("track,iterator", [
        ("dense_track", 100),     # native bin 50 -> 2 native bins per output bin
        ("dense_track", 137),     # unaligned coarsening
        ("dense_track", 500),     # 10 native bins per output bin
        ("sparse_track", 100),
        ("dense_track * 2 + 1", 100),  # bare-track expression
    ])
    def test_bare_track_iterator_matches_r(self, track, iterator):
        intervs = pm.gintervals("1", [0], [20000])
        py = pm.gextract(track, intervs, iterator=iterator).sort_values("start")
        r_df = _run_r_df(
            f"""
intervs <- gintervals("1", c(0), c(20000))
df <- gextract("{track}", intervs, iterator={iterator})
df <- df[order(df$start), ]
"""
        )
        # The value column name may be sanitized differently by R for an
        # expression, so select it positionally in both frames.
        def value_col(df):
            return [c for c in df.columns if c not in ("chrom", "start", "end", "intervalID")][0]
        np.testing.assert_array_equal(py["start"].to_numpy(int), r_df["start"].to_numpy(int))
        np.testing.assert_allclose(
            py[value_col(py)].to_numpy(float), r_df[value_col(r_df)].to_numpy(float),
            rtol=1e-5, atol=1e-7, equal_nan=True,
        )


class TestMixedDenseSparseExpression:
    """R allows mixing dense (fixed-bin) and sparse tracks in one expression,
    but ONLY with an explicit iterator; without one it errors. Mirror that."""

    def setup_method(self):
        pm.gdb_init(str(TESTDB))

    @pytest.mark.parametrize("iterator", [50, 100, 137])
    def test_mixed_with_explicit_iterator_matches_r(self, iterator):
        expr = "dense_track + sparse_track"
        intervs = pm.gintervals("1", [0], [20000])
        py = pm.gextract(expr, intervs, iterator=iterator).sort_values("start")
        r_df = _run_r_df(
            f"""
intervs <- gintervals("1", c(0), c(20000))
df <- gextract("dense_track + sparse_track", intervs, iterator={iterator})
df <- df[order(df$start), ]
"""
        )

        def value_col(df):
            return [c for c in df.columns if c not in ("chrom", "start", "end", "intervalID")][0]
        np.testing.assert_array_equal(py["start"].to_numpy(int), r_df["start"].to_numpy(int))
        np.testing.assert_allclose(
            py[value_col(py)].to_numpy(float), r_df[value_col(r_df)].to_numpy(float),
            rtol=1e-5, atol=1e-7, equal_nan=True,
        )

    def test_mixed_summary_with_iterator_matches_r(self):
        py = pm.gsummary("dense_track + sparse_track", pm.gintervals_all(), iterator=50)
        r_df = _run_r_df(
            """
df <- as.data.frame(t(gsummary("dense_track + sparse_track", gintervals.all(), iterator=50)))
"""
        )
        np.testing.assert_allclose(
            py.to_numpy(float), r_df.to_numpy(float).ravel(), rtol=1e-5, atol=1e-6, equal_nan=True,
        )

    def test_mixed_without_iterator_errors(self):
        # R: "Cannot implicitly determine iterator policy ... tracks in
        # different formats." pymisha should also refuse (not silently pick one).
        with pytest.raises(Exception, match="(?i)implicit|iterator|format"):
            pm.gextract("dense_track + sparse_track", pm.gintervals("1", 0, 20000))
