"""
Golden-master tests for general statistical functions (gsummary, gquantiles, gcor, etc.).

These tests compare pymisha output against R misha reference implementation.
"""
import contextlib
import os
import shutil
import subprocess
import tempfile
from io import StringIO

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

if shutil.which("R") is None:
    pytest.skip("R not available; skipping golden-master tests", allow_module_level=True)

# Path to test database
TESTDB = "tests/testdb/trackdb/test"


def run_r_code(code):
    """Run R code and return the output as string."""
    # Create temporary R script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            ['R', '--quiet', '--no-save', '-f', script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout, result.stderr
    finally:
        os.unlink(script_path)


class TestGoldenMasterStats:
    """Golden-master tests for statistical functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gsummary_matches_r(self):
        """gsummary matches R output."""
        intervals = pm.gintervals("1", 0, 5000)

        # Get pymisha result
        py_result = pm.gsummary("dense_track", intervals)

        # Get R result
        # We use gextract + manual summary in R to ensure na.rm=TRUE logic
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 5000)
# Extract all values
vals <- gextract("dense_track", intervals)
v <- vals$dense_track

# Compute stats ignoring NA
cat("min:", min(v, na.rm=TRUE), "\\n", sep="")
cat("max:", max(v, na.rm=TRUE), "\\n", sep="")
cat("mean:", mean(v, na.rm=TRUE), "\\n", sep="")
cat("sd:", sd(v, na.rm=TRUE), "\\n", sep="")
'''
        stdout, stderr = run_r_code(r_code)

        # Parse R output
        r_stats = {}
        for line in stdout.strip().splitlines():
            if line.startswith('>'):
                continue
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    r_stats[key] = val
                except ValueError:
                    pass

        if not r_stats:
             pytest.fail(f"Could not parse R output:\n{stdout}")

        # Compare
        assert np.isclose(py_result['Min'], r_stats['min']), (
            f"Min mismatch: {py_result['Min']} vs {r_stats['min']}"
        )
        assert np.isclose(py_result['Max'], r_stats['max']), (
            f"Max mismatch: {py_result['Max']} vs {r_stats['max']}"
        )
        assert np.isclose(py_result['Mean'], r_stats['mean']), (
            f"Mean mismatch: {py_result['Mean']} vs {r_stats['mean']}"
        )

        if 'sd' in r_stats and not np.isnan(r_stats['sd']):
             assert np.isclose(py_result['Std dev'], r_stats['sd']), (
                 f"Stdev mismatch: {py_result['Std dev']} vs {r_stats['sd']}"
             )


    def test_gquantiles_matches_r(self):
        """gquantiles matches R output."""
        intervals = pm.gintervals("1", 0, 100000)
        percentiles = [0.1, 0.5, 0.9]

        # Get pymisha result
        py_result = pm.gquantiles("dense_track", percentiles, intervals)

        # Get R result
        # Use gextract + quantile to ensure na.rm=TRUE
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 100000)
vals <- gextract("dense_track", intervals)
v <- vals$dense_track
q <- quantile(v, probs=c(0.1, 0.5, 0.9), na.rm=TRUE, type=7)

cat("P10:", q["10%"], "\\n", sep="")
cat("P50:", q["50%"], "\\n", sep="")
cat("P90:", q["90%"], "\\n", sep="")
'''
        stdout, stderr = run_r_code(r_code)

        # Parse R results
        r_quants = {}
        for line in stdout.strip().splitlines():
            if line.startswith('>'):
                continue
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    r_quants[key] = val
                except ValueError:
                    pass

        if not r_quants:
             pytest.fail(f"Could not parse R output:\n{stdout}")

        # Compare
        if hasattr(py_result, 'iloc'):
            assert np.isclose(py_result.iloc[0], r_quants['P10']), "10% mismatch"
            assert np.isclose(py_result.iloc[1], r_quants['P50']), "50% mismatch"
            assert np.isclose(py_result.iloc[2], r_quants['P90']), "90% mismatch"
        else:
            assert np.isclose(py_result[0], r_quants['P10']), "10% mismatch"
            assert np.isclose(py_result[1], r_quants['P50']), "50% mismatch"
            assert np.isclose(py_result[2], r_quants['P90']), "90% mismatch"

    def test_gcor_matches_r(self):
        """gcor matches R output."""
        intervals = pm.gintervals("1", 0, 10000)

        # Get pymisha result
        py_result = pm.gcor("dense_track", "dense_track * 2 + 0.1", intervals=intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 10000)
result <- gcor("dense_track", "dense_track * 2 + 0.1", intervals)
cat("Cor:", result, "\\n", sep="")
'''
        stdout, stderr = run_r_code(r_code)

        r_cor = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('>'):
                continue
            if line.startswith("Cor:"):
                with contextlib.suppress(ValueError):
                    r_cor = float(line.split(':')[1].strip())

        assert r_cor is not None
        py_val = py_result[0] if hasattr(py_result, '__getitem__') else py_result
        assert np.isclose(py_val, r_cor, atol=1e-6), f"Cor mismatch: {py_val} vs {r_cor}"

    def test_gbins_summary_matches_r(self):
        """gbins_summary matches R output."""
        intervals = pm.gintervals("1", [0, 5000], [1000, 6000])
        breaks = [0, 0.2, 0.5, 1.0]

        # Get pymisha result
        py_result = pm.gbins_summary("dense_track", breaks, intervals=intervals, track_stat="dense_track")

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", c(0, 5000), c(1000, 6000))
breaks <- c(0, 0.2, 0.5, 1.0)
result <- gbins.summary("dense_track", breaks, intervals, track.stat="dense_track")
write.table(result, sep="\\t", quote=FALSE, row.names=FALSE)
'''
        stdout, stderr = run_r_code(r_code)

        lines = [line for line in stdout.strip().splitlines() if not line.startswith('>')]
        header_line = -1
        for i, line in enumerate(lines):
            if "bin_min" in line or ("min" in line and "max" in line):
                header_line = i
                break

        if header_line >= 0:
            r_df = pd.read_csv(StringIO('\n'.join(lines[header_line:])), sep='\t')

            # Compare columns
            assert len(py_result) == len(r_df)
            assert np.allclose(py_result['bin_min'], r_df['bin_min'])
            assert np.allclose(py_result['bin_max'], r_df['bin_max'])
            assert np.allclose(py_result['mean'], r_df['mean'], equal_nan=True)

    def test_gsegment_matches_r(self):
        """gsegment matches R output."""
        intervals = pm.gintervals("1", 0, 50000)
        minsegment = 2000
        maxpval = 0.05

        # Get pymisha result
        py_result = pm.gsegment("dense_track", minsegment, maxpval=maxpval, intervals=intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 50000)
result <- gsegment("dense_track", {minsegment}, maxpval={maxpval}, intervals=intervals)
write.table(result, sep="\\t", quote=FALSE, row.names=FALSE)
'''
        stdout, stderr = run_r_code(r_code)

        lines = [line for line in stdout.strip().splitlines() if not line.startswith('>')]
        header_line = -1
        for i, line in enumerate(lines):
            if "chrom" in line and "start" in line:
                header_line = i
                break

        if header_line >= 0 and len(lines) > header_line + 1:
            r_df = pd.read_csv(StringIO('\n'.join(lines[header_line:])), sep='\t')

            if py_result is not None and not py_result.empty:
                assert len(py_result) == len(r_df)
                assert np.allclose(py_result['start'], r_df['start'])
                assert np.allclose(py_result['end'], r_df['end'])
        else:
             assert py_result is None or py_result.empty

    def test_gsample_matches_r(self):
        """gsample matches R output (statistical properties)."""
        intervals = pm.gintervals("1", 0, 10000)
        n_samples = 100

        # Get pymisha result
        py_result = pm.gsample("dense_track", n_samples, intervals=intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 10000)
result <- gsample("dense_track", {n_samples}, intervals)
cat("Mean:", mean(result, na.rm=TRUE), "\\n", sep="")
cat("SD:", sd(result, na.rm=TRUE), "\\n", sep="")
'''
        stdout, stderr = run_r_code(r_code)

        r_mean = None
        r_sd = None

        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith('>'):
                continue
            if line.startswith("Mean:"):
                r_mean = float(line.split(':')[1].strip())
            if line.startswith("SD:"):
                r_sd = float(line.split(':')[1].strip())

        assert r_mean is not None

        # Compare
        py_mean = np.mean(py_result)
        py_sd = np.std(py_result)

        assert len(py_result) == n_samples
        assert np.isclose(py_mean, r_mean, rtol=0.5), f"Mean mismatch: {py_mean} vs {r_mean}"
        assert np.isclose(py_sd, r_sd, rtol=0.5), f"SD mismatch: {py_sd} vs {r_sd}"
