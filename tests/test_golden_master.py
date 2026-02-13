"""
Golden-master tests comparing pymisha with R misha.

These tests generate reference outputs using R misha and compare them
against pymisha outputs to ensure compatibility.
"""
import os
import shutil
import subprocess
import tempfile

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


def r_df_to_pandas(r_output):
    """Parse R data frame output to pandas DataFrame.

    Handles the standard R print format for data frames.
    """
    lines = r_output.strip().split('\n')

    # Find the header line (starts with columns)
    header_idx = None
    for i, line in enumerate(lines):
        if 'chrom' in line and 'start' in line and 'end' in line:
            header_idx = i
            break

    if header_idx is None:
        return None

    # Parse header
    header = lines[header_idx].split()

    # Parse data rows
    data = []
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) >= len(header) + 1:  # +1 for row number
            # Skip the row number (first column)
            data.append(parts[1:len(header)+1])

    if not data:
        return None

    df = pd.DataFrame(data, columns=header)

    # Convert types
    if 'start' in df.columns:
        df['start'] = df['start'].astype(int)
    if 'end' in df.columns:
        df['end'] = df['end'].astype(int)

    return df


def normalize_chrom(chrom):
    """Strip 'chr' prefix for comparison purposes."""
    s = str(chrom)
    if s.startswith('chr'):
        return s[3:]
    return s


class TestGoldenMasterIntervals:
    """Golden-master tests for interval operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gintervals_all_matches_r(self):
        """gintervals_all matches R output."""
        # Get pymisha result
        py_result = pm.gintervals_all()

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
print(gintervals.all())
'''
        stdout, stderr = run_r_code(r_code)
        r_result = r_df_to_pandas(stdout)

        # Compare
        assert r_result is not None, f"Failed to parse R output: {stdout}"
        assert len(py_result) == len(r_result)

        # Sort both and compare
        py_sorted = py_result.sort_values(['chrom', 'start']).reset_index(drop=True)
        r_sorted = r_result.sort_values(['chrom', 'start']).reset_index(drop=True)

        # Compare chromosomes (normalize chr prefix)
        py_chroms = [normalize_chrom(c) for c in py_sorted['chrom']]
        r_chroms = [normalize_chrom(c) for c in r_sorted['chrom']]
        assert py_chroms == r_chroms
        assert list(py_sorted['start']) == list(r_sorted['start'])
        assert list(py_sorted['end']) == list(r_sorted['end'])

    def test_gintervals_union_matches_r(self):
        """gintervals_union matches R output."""
        # Create test intervals
        intervs1 = pm.gintervals("1", [100, 400], [200, 500])
        intervs2 = pm.gintervals("1", [150, 600], [350, 700])

        # Get pymisha result
        py_result = pm.gintervals_union(intervs1, intervs2)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs1 <- gintervals("1", c(100, 400), c(200, 500))
intervs2 <- gintervals("1", c(150, 600), c(350, 700))
print(gintervals.union(intervs1, intervs2))
'''
        stdout, stderr = run_r_code(r_code)
        r_result = r_df_to_pandas(stdout)

        # Compare
        assert r_result is not None, f"Failed to parse R output: {stdout}"
        assert len(py_result) == len(r_result)

        py_sorted = py_result.sort_values(['chrom', 'start']).reset_index(drop=True)
        r_sorted = r_result.sort_values(['chrom', 'start']).reset_index(drop=True)

        assert list(py_sorted['start']) == list(r_sorted['start'])
        assert list(py_sorted['end']) == list(r_sorted['end'])

    def test_gintervals_intersect_matches_r(self):
        """gintervals_intersect matches R output."""
        intervs1 = pm.gintervals("1", [100, 400], [300, 600])
        intervs2 = pm.gintervals("1", [200, 500], [400, 700])

        # Get pymisha result
        py_result = pm.gintervals_intersect(intervs1, intervs2)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs1 <- gintervals("1", c(100, 400), c(300, 600))
intervs2 <- gintervals("1", c(200, 500), c(400, 700))
print(gintervals.intersect(intervs1, intervs2))
'''
        stdout, stderr = run_r_code(r_code)
        r_result = r_df_to_pandas(stdout)

        # Compare
        assert r_result is not None, f"Failed to parse R output: {stdout}"
        assert len(py_result) == len(r_result)

        py_sorted = py_result.sort_values(['chrom', 'start']).reset_index(drop=True)
        r_sorted = r_result.sort_values(['chrom', 'start']).reset_index(drop=True)

        assert list(py_sorted['start']) == list(r_sorted['start'])
        assert list(py_sorted['end']) == list(r_sorted['end'])

    def test_gintervals_diff_matches_r(self):
        """gintervals_diff matches R output."""
        intervs1 = pm.gintervals("1", 100, 500)
        intervs2 = pm.gintervals("1", 200, 300)

        # Get pymisha result
        py_result = pm.gintervals_diff(intervs1, intervs2)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs1 <- gintervals("1", 100, 500)
intervs2 <- gintervals("1", 200, 300)
print(gintervals.diff(intervs1, intervs2))
'''
        stdout, stderr = run_r_code(r_code)
        r_result = r_df_to_pandas(stdout)

        # Compare
        assert r_result is not None, f"Failed to parse R output: {stdout}"
        assert len(py_result) == len(r_result)

        py_sorted = py_result.sort_values(['chrom', 'start']).reset_index(drop=True)
        r_sorted = r_result.sort_values(['chrom', 'start']).reset_index(drop=True)

        assert list(py_sorted['start']) == list(r_sorted['start'])
        assert list(py_sorted['end']) == list(r_sorted['end'])


class TestGoldenMasterGextract:
    """Golden-master tests for gextract."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gextract_dense_track_matches_r(self):
        """gextract on dense track matches R output."""
        # Use small intervals for comparison
        intervals = pm.gintervals("1", [0, 10000, 50000], [1000, 11000, 51000])

        # Get pymisha result
        py_result = pm.gextract("dense_track", intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", c(0, 10000, 50000), c(1000, 11000, 51000))
result <- gextract("dense_track", intervals)
print(result)
'''
        stdout, stderr = run_r_code(r_code)

        # Parse R output - columns are: row_num chrom start end dense_track intervalID
        lines = stdout.strip().split('\n')

        # Find the data lines after header
        r_values = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:  # row_num chrom start end value intervalID
                try:
                    # Dense track value is at index 4 (or -2 from end before intervalID)
                    val = float(parts[4])
                    r_values.append(val)
                except ValueError:
                    continue

        # Compare values (allow small floating point differences)
        py_values = py_result['dense_track'].values

        assert len(py_values) == len(r_values), f"Length mismatch: pymisha={len(py_values)}, R={len(r_values)}"

        for i, (py_val, r_val) in enumerate(zip(py_values, r_values, strict=False)):
            if np.isnan(py_val) and np.isnan(r_val):
                continue
            assert abs(py_val - r_val) < 1e-6, f"Value mismatch at index {i}: pymisha={py_val}, R={r_val}"

    def test_gextract_expression_matches_r(self):
        """gextract with expression matches R output."""
        intervals = pm.gintervals("1", [0, 10000], [1000, 11000])

        # Get pymisha result
        py_result = pm.gextract("dense_track * 2 + 1", intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", c(0, 10000), c(1000, 11000))
result <- gextract("dense_track * 2 + 1", intervals)
print(result)
'''
        stdout, stderr = run_r_code(r_code)

        # Parse values from R output - columns: row_num chrom start end value intervalID
        lines = stdout.strip().split('\n')
        r_values = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    val = float(parts[4])
                    r_values.append(val)
                except ValueError:
                    continue

        # R uses "dense_track * 2 + 1" as column name, pymisha uses same
        py_values = py_result['dense_track * 2 + 1'].values

        assert len(py_values) == len(r_values)
        for py_val, r_val in zip(py_values, r_values, strict=False):
            if np.isnan(py_val) and np.isnan(r_val):
                continue
            assert abs(py_val - r_val) < 1e-6


class TestGoldenMasterGscreen:
    """Golden-master tests for gscreen."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gscreen_simple_filter_matches_r(self):
        """gscreen with simple filter matches R output."""
        intervals = pm.gintervals("1", 0, 100000)

        # Get pymisha result
        py_result = pm.gscreen("dense_track > 0.5", intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 100000)
result <- gscreen("dense_track > 0.5", intervals)
cat("Rows:", nrow(result), "\\n")
'''
        stdout, stderr = run_r_code(r_code)

        # Parse row count from R
        r_count = None
        for line in stdout.split('\n'):
            if line.startswith('Rows:'):
                r_count = int(line.split(':')[1].strip())
                break

        assert r_count is not None, f"Failed to parse R output: {stdout}"

        py_count = len(py_result) if py_result is not None else 0

        # Row counts should match
        assert py_count == r_count, f"Row count mismatch: pymisha={py_count}, R={r_count}"

    def test_gscreen_pwm_matches_r(self):
        """gscreen with PWM virtual track matches R output."""
        # Define PSSM with R's order (A, C, T, G) to test reordering fix
        pssm = pd.DataFrame({
            'A': [0.4, 0.1, 0.1, 0.4],
            'C': [0.1, 0.4, 0.4, 0.1],
            'T': [0.1, 0.4, 0.1, 0.4],
            'G': [0.4, 0.1, 0.4, 0.1]
        })

        # Create virtual track in pymisha
        pm.gvtrack_create("ctcf_test", src=None, func="pwm", pssm=pssm)

        # Get pymisha result
        # Note: we use a small part of chromosome 1 for efficiency in testdb
        intervals = pm.gintervals("1", 0, 100000)

        # When using virtual tracks, pymisha's gscreen implementation merges adjacent
        # intervals that pass the filter. R's gscreen with iterator returns individual bins.
        # To compare, we'll check that all R hits are covered by Python results.
        py_result = pm.gscreen("ctcf_test > -5", intervals=intervals, iterator=4)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
pssm <- data.frame(
    A = c(0.4, 0.1, 0.1, 0.4),
    C = c(0.1, 0.4, 0.4, 0.1),
    T = c(0.1, 0.4, 0.1, 0.4),
    G = c(0.4, 0.1, 0.4, 0.1)
)
gvtrack.create("ctcf_test", pssm = pssm, func = "pwm")
intervals <- gintervals("1", 0, 100000)
result <- gscreen("ctcf_test > -5", intervals = intervals, iterator = 4)
if (!is.null(result) && nrow(result) > 0) {{
    cat("Starts:", paste(result$start, collapse=","), "\\n")
    cat("Ends:", paste(result$end, collapse=","), "\\n")
}}
'''
        stdout, stderr = run_r_code(r_code)

        # Parse results from R
        r_starts = []
        r_ends = []
        for line in stdout.split('\n'):
            if line.startswith('Starts:'):
                starts_str = line.split(':')[1].strip()
                if starts_str:
                    r_starts = [int(s) for s in starts_str.split(',')]
            if line.startswith('Ends:'):
                ends_str = line.split(':')[1].strip()
                if ends_str:
                    r_ends = [int(s) for s in ends_str.split(',')]

        if not r_starts:
             assert py_result is None
             return

        assert py_result is not None
        assert len(py_result) == len(r_starts), f"Row count mismatch: pymisha={len(py_result)}, R={len(r_starts)}"

        # Verify exact coordinates
        py_starts = py_result['start'].tolist()
        py_ends = py_result['end'].tolist()

        assert py_starts == r_starts, "Start coordinates do not match exactly"
        assert py_ends == r_ends, "End coordinates do not match exactly"


class TestGoldenMasterNeighbors:
    """Golden-master tests for gintervals_neighbors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gintervals_neighbors_basic_matches_r(self):
        """gintervals_neighbors basic case matches R output."""
        # Create test intervals
        intervs1 = pm.gintervals("1", [1000, 5000], [1100, 5100])
        intervs2 = pm.gintervals("1", [1200, 3000, 5200], [1300, 3100, 5300])

        # Get pymisha result
        py_result = pm.gintervals_neighbors(intervs1, intervs2)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs1 <- gintervals("1", c(1000, 5000), c(1100, 5100))
intervs2 <- gintervals("1", c(1200, 3000, 5200), c(1300, 3100, 5300))
result <- gintervals.neighbors(intervs1, intervs2)
print(result)
'''
        stdout, stderr = run_r_code(r_code)

        # Parse R output
        lines = stdout.strip().split('\n')

        # Find header line with 'dist'
        header_idx = None
        for i, line in enumerate(lines):
            if 'dist' in line and 'chrom' in line:
                header_idx = i
                break

        assert header_idx is not None, f"Failed to find header in R output: {stdout}"

        header = lines[header_idx].split()

        # Parse data rows
        r_dists = []
        for line in lines[header_idx + 1:]:
            parts = line.split()
            if len(parts) >= len(header) + 1:  # +1 for row number
                # Find dist column index in header
                dist_idx = header.index('dist') + 1  # +1 for row number offset
                try:
                    r_dists.append(int(parts[dist_idx]))
                except (ValueError, IndexError):
                    continue

        # Compare
        assert py_result is not None
        assert len(py_result) == len(r_dists), f"Row count mismatch: pymisha={len(py_result)}, R={len(r_dists)}"

        py_dists = py_result['dist'].tolist()
        for i, (py_dist, r_dist) in enumerate(zip(py_dists, r_dists, strict=False)):
            assert py_dist == r_dist, f"Distance mismatch at row {i}: pymisha={py_dist}, R={r_dist}"

    def test_gintervals_neighbors_multiple_matches_r(self):
        """gintervals_neighbors with maxneighbors > 1 matches R output."""
        intervs1 = pm.gintervals("1", 1000, 1100)
        intervs2 = pm.gintervals("1", [1200, 1300, 1400], [1250, 1350, 1450])

        # Get pymisha result
        py_result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=3)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs1 <- gintervals("1", 1000, 1100)
intervs2 <- gintervals("1", c(1200, 1300, 1400), c(1250, 1350, 1450))
result <- gintervals.neighbors(intervs1, intervs2, maxneighbors=3)
cat("Rows:", nrow(result), "\\n")
cat("Distances:", paste(result$dist, collapse=","), "\\n")
'''
        stdout, stderr = run_r_code(r_code)

        # Parse row count and distances from R
        r_count = None
        r_dists = None
        for line in stdout.split('\n'):
            if line.startswith('Rows:'):
                r_count = int(line.split(':')[1].strip())
            if line.startswith('Distances:'):
                dists_str = line.split(':')[1].strip()
                r_dists = [int(d) for d in dists_str.split(',')]

        assert r_count is not None, f"Failed to parse R output: {stdout}"
        assert r_dists is not None, f"Failed to parse distances from R output: {stdout}"

        assert py_result is not None
        assert len(py_result) == r_count
        py_dists = py_result['dist'].tolist()
        assert py_dists == r_dists, f"Distances mismatch: pymisha={py_dists}, R={r_dists}"

    def test_gintervals_neighbors_distance_range_matches_r(self):
        """gintervals_neighbors with distance range matches R output."""
        intervs1 = pm.gintervals("1", 1000, 1100)
        intervs2 = pm.gintervals("1", [1150, 1300, 1500], [1200, 1400, 1600])

        # Get pymisha result - only neighbors with dist between 50 and 250
        py_result = pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=10,
                                            mindist=50, maxdist=250)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs1 <- gintervals("1", 1000, 1100)
intervs2 <- gintervals("1", c(1150, 1300, 1500), c(1200, 1400, 1600))
result <- gintervals.neighbors(intervs1, intervs2, maxneighbors=10, mindist=50, maxdist=250)
cat("Rows:", nrow(result), "\\n")
if (!is.null(result) && nrow(result) > 0) {{
    cat("Distances:", paste(result$dist, collapse=","), "\\n")
}}
'''
        stdout, stderr = run_r_code(r_code)

        # Parse results
        r_count = None
        r_dists = []
        for line in stdout.split('\n'):
            if line.startswith('Rows:'):
                r_count = int(line.split(':')[1].strip())
            if line.startswith('Distances:'):
                dists_str = line.split(':')[1].strip()
                if dists_str:
                    r_dists = [int(d) for d in dists_str.split(',')]

        assert r_count is not None, f"Failed to parse R output: {stdout}"

        py_count = len(py_result) if py_result is not None else 0
        assert py_count == r_count, f"Row count mismatch: pymisha={py_count}, R={r_count}"

        if py_result is not None and len(py_result) > 0:
            py_dists = py_result['dist'].tolist()
            assert py_dists == r_dists, f"Distances mismatch: pymisha={py_dists}, R={r_dists}"


class TestGoldenMasterGpartition:
    """Golden-master tests for gpartition."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gpartition_basic_matches_r(self):
        """gpartition basic usage matches R output."""
        intervals = pm.gintervals("1", 0, 5000)
        breaks = [0.0, 0.05, 0.1, 0.15, 0.2]

        # Get pymisha result
        py_result = pm.gpartition("dense_track", breaks, intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 5000)
breaks <- c(0, 0.05, 0.1, 0.15, 0.2)
result <- gpartition("dense_track", breaks, intervals)
cat("Rows:", nrow(result), "\\n")
if (!is.null(result) && nrow(result) > 0) {{
    cat("Bins:", paste(result$bin, collapse=","), "\\n")
    cat("Starts:", paste(result$start, collapse=","), "\\n")
    cat("Ends:", paste(result$end, collapse=","), "\\n")
}}
'''
        stdout, stderr = run_r_code(r_code)

        # Parse results
        r_count = None
        r_bins = []
        r_starts = []
        r_ends = []
        for line in stdout.split('\n'):
            if line.startswith('Rows:'):
                r_count = int(line.split(':')[1].strip())
            if line.startswith('Bins:'):
                bins_str = line.split(':')[1].strip()
                if bins_str:
                    r_bins = [int(b) for b in bins_str.split(',')]
            if line.startswith('Starts:'):
                starts_str = line.split(':')[1].strip()
                if starts_str:
                    r_starts = [int(s) for s in starts_str.split(',')]
            if line.startswith('Ends:'):
                ends_str = line.split(':')[1].strip()
                if ends_str:
                    r_ends = [int(e) for e in ends_str.split(',')]

        assert r_count is not None, f"Failed to parse R output: {stdout}"

        py_count = len(py_result) if py_result is not None else 0
        assert py_count == r_count, f"Row count mismatch: pymisha={py_count}, R={r_count}"

        if py_result is not None and len(py_result) > 0:
            py_bins = py_result['bin'].tolist()
            py_starts = py_result['start'].tolist()
            py_ends = py_result['end'].tolist()
            assert py_bins == r_bins, f"Bins mismatch: pymisha={py_bins}, R={r_bins}"
            assert py_starts == r_starts, f"Starts mismatch: pymisha={py_starts}, R={r_starts}"
            assert py_ends == r_ends, f"Ends mismatch: pymisha={py_ends}, R={r_ends}"

    def test_gpartition_include_lowest_matches_r(self):
        """gpartition with include_lowest matches R output."""
        intervals = pm.gintervals("1", 0, 5000)
        breaks = [0.0, 0.05, 0.1, 0.15, 0.2]

        # Get pymisha result
        py_result = pm.gpartition("dense_track", breaks, intervals, include_lowest=True)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 5000)
breaks <- c(0, 0.05, 0.1, 0.15, 0.2)
result <- gpartition("dense_track", breaks, intervals, include.lowest=TRUE)
cat("Rows:", nrow(result), "\\n")
if (!is.null(result) && nrow(result) > 0) {{
    cat("Bins:", paste(result$bin, collapse=","), "\\n")
    cat("Starts:", paste(result$start, collapse=","), "\\n")
    cat("Ends:", paste(result$end, collapse=","), "\\n")
}}
'''
        stdout, stderr = run_r_code(r_code)

        # Parse results
        r_count = None
        r_bins = []
        r_starts = []
        r_ends = []
        for line in stdout.split('\n'):
            if line.startswith('Rows:'):
                r_count = int(line.split(':')[1].strip())
            if line.startswith('Bins:'):
                bins_str = line.split(':')[1].strip()
                if bins_str:
                    r_bins = [int(b) for b in bins_str.split(',')]
            if line.startswith('Starts:'):
                starts_str = line.split(':')[1].strip()
                if starts_str:
                    r_starts = [int(s) for s in starts_str.split(',')]
            if line.startswith('Ends:'):
                ends_str = line.split(':')[1].strip()
                if ends_str:
                    r_ends = [int(e) for e in ends_str.split(',')]

        assert r_count is not None, f"Failed to parse R output: {stdout}"

        py_count = len(py_result) if py_result is not None else 0
        assert py_count == r_count, f"Row count mismatch: pymisha={py_count}, R={r_count}"

        if py_result is not None and len(py_result) > 0:
            py_bins = py_result['bin'].tolist()
            py_starts = py_result['start'].tolist()
            py_ends = py_result['end'].tolist()
            assert py_bins == r_bins, f"Bins mismatch: pymisha={py_bins}, R={r_bins}"
            assert py_starts == r_starts, f"Starts mismatch: pymisha={py_starts}, R={r_starts}"
            assert py_ends == r_ends, f"Ends mismatch: pymisha={py_ends}, R={r_ends}"

    def test_gpartition_coarse_breaks_matches_r(self):
        """gpartition with coarse breaks matches R output."""
        intervals = pm.gintervals("1", 0, 5000)
        breaks = [0.0, 0.1, 0.2]

        # Get pymisha result
        py_result = pm.gpartition("dense_track", breaks, intervals)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", 0, 5000)
breaks <- c(0, 0.1, 0.2)
result <- gpartition("dense_track", breaks, intervals)
cat("Rows:", nrow(result), "\\n")
if (!is.null(result) && nrow(result) > 0) {{
    cat("Bins:", paste(result$bin, collapse=","), "\\n")
    cat("Starts:", paste(result$start, collapse=","), "\\n")
    cat("Ends:", paste(result$end, collapse=","), "\\n")
}}
'''
        stdout, stderr = run_r_code(r_code)

        # Parse results
        r_count = None
        r_bins = []
        r_starts = []
        r_ends = []
        for line in stdout.split('\n'):
            if line.startswith('Rows:'):
                r_count = int(line.split(':')[1].strip())
            if line.startswith('Bins:'):
                bins_str = line.split(':')[1].strip()
                if bins_str:
                    r_bins = [int(b) for b in bins_str.split(',')]
            if line.startswith('Starts:'):
                starts_str = line.split(':')[1].strip()
                if starts_str:
                    r_starts = [int(s) for s in starts_str.split(',')]
            if line.startswith('Ends:'):
                ends_str = line.split(':')[1].strip()
                if ends_str:
                    r_ends = [int(e) for e in ends_str.split(',')]

        assert r_count is not None, f"Failed to parse R output: {stdout}"

        py_count = len(py_result) if py_result is not None else 0
        assert py_count == r_count, f"Row count mismatch: pymisha={py_count}, R={r_count}"

        if py_result is not None and len(py_result) > 0:
            py_bins = py_result['bin'].tolist()
            py_starts = py_result['start'].tolist()
            py_ends = py_result['end'].tolist()
            assert py_bins == r_bins, f"Bins mismatch: pymisha={py_bins}, R={r_bins}"
            assert py_starts == r_starts, f"Starts mismatch: pymisha={py_starts}, R={r_starts}"
            assert py_ends == r_ends, f"Ends mismatch: pymisha={py_ends}, R={r_ends}"


class TestGoldenMasterGdist:
    """Golden-master tests for gdist."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gdist_1d_matches_r(self):
        """gdist 1D distribution matches R misha output."""
        # Get pymisha result
        py_result = pm.gdist("dense_track", [0, 0.2, 0.5, 1])

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
result <- gdist("dense_track", c(0, 0.2, 0.5, 1))
cat("Bin0:", result[1], "\\n")
cat("Bin1:", result[2], "\\n")
cat("Bin2:", result[3], "\\n")
cat("Total:", sum(result), "\\n")
'''
        stdout, stderr = run_r_code(r_code)

        # Parse R results
        r_bins = []
        r_total = 0
        for line in stdout.split('\n'):
            if line.startswith(('Bin0:', 'Bin1:', 'Bin2:')):
                r_bins.append(int(float(line.split(':')[1].strip())))
            elif line.startswith('Total:'):
                r_total = int(float(line.split(':')[1].strip()))

        # Compare
        py_bins = [int(v) for v in py_result.flatten()]
        assert len(py_bins) == len(r_bins), f"gdist bin count mismatch: pymisha={len(py_bins)}, R={len(r_bins)}"
        assert py_bins == r_bins, f"gdist bins mismatch: pymisha={py_bins}, R={r_bins}"
        assert int(py_result.sum()) == r_total, f"gdist total mismatch: pymisha={int(py_result.sum())}, R={r_total}"
