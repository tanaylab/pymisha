"""
Golden-master tests for advanced interval operations (2D, canonic).
"""
import os
import shutil
import subprocess
import tempfile
from io import StringIO

import pandas as pd
import pytest

import pymisha as pm

if shutil.which("R") is None:
    pytest.skip("R not available; skipping golden-master tests", allow_module_level=True)

TESTDB = "tests/testdb/trackdb/test"

def run_r_code(code):
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

def parse_r_table(stdout):
    lines = [line for line in stdout.strip().splitlines() if not line.startswith('>')]
    # Find header
    header_idx = -1
    for i, line in enumerate(lines):
        if "chrom" in line:
            header_idx = i
            break
    if header_idx == -1:
        return None
    # Use literal tab separator
    return pd.read_csv(StringIO('\n'.join(lines[header_idx:])), sep='\t')

class TestGoldenMasterAdvancedIntervals:
    @pytest.fixture(autouse=True)
    def setup(self):
        pm.gdb_init(TESTDB)

    def test_gintervals_canonic_matches_r(self):
        """gintervals_canonic matches R gintervals.canonic."""
        # Overlapping intervals
        intervs = pm.gintervals("1", [100, 150, 500], [200, 250, 600])
        # 100-200 and 150-250 overlap -> 100-250

        py_res = pm.gintervals_canonic(intervs)

        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs <- gintervals("1", c(100, 150, 500), c(200, 250, 600))
res <- gintervals.canonic(intervs)
write.table(res, sep="\\t", quote=FALSE, row.names=FALSE)
'''
        stdout, stderr = run_r_code(r_code)
        r_df = parse_r_table(stdout)

        assert len(py_res) == len(r_df)
        assert list(py_res['start']) == list(r_df['start'])
        assert list(py_res['end']) == list(r_df['end'])

    def test_gintervals_2d_matches_r(self):
        """gintervals_2d (parallel) matches R gintervals.2d."""
        # Parallel construction
        intervs1 = pm.gintervals("1", [100, 500], [200, 600])
        intervs2 = pm.gintervals("1", [300, 800], [400, 900])

        py_res = pm.gintervals_2d(
            intervs1['chrom'], intervs1['start'], intervs1['end'],
            intervs2['chrom'], intervs2['start'], intervs2['end']
        )

        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
# R gintervals.2d usually takes two interval sets and does Cartesian.
# To do parallel, we construct manually or use gintervals.2d with vectors?
# Check documentation: gintervals.2d(chroms1, starts1, ends1, chroms2, starts2, ends2)
res <- gintervals.2d(
    c(1, 1), c(100, 500), c(200, 600),
    c(1, 1), c(300, 800), c(400, 900)
)
write.table(res, sep="\\t", quote=FALSE, row.names=FALSE)
'''
        stdout, stderr = run_r_code(r_code)
        r_df = parse_r_table(stdout)

        # 2D intervals have chrom1, start1, end1, chrom2, start2, end2
        assert len(py_res) == len(r_df)
        assert list(py_res['start1']) == list(r_df['start1'])
        assert list(py_res['start2']) == list(r_df['start2'])

    def test_gintervals_covered_bp_matches_r(self):
        """gintervals_covered_bp matches R calculation."""
        intervs = pm.gintervals("1", [100, 150, 500], [200, 250, 600])
        # Union: 100-250 (150bp) + 500-600 (100bp) = 250bp

        py_res = pm.gintervals_covered_bp(intervs)

        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervs <- gintervals("1", c(100, 150, 500), c(200, 250, 600))
# Manual calculation via canonic
canon <- gintervals.canonic(intervs)
total <- sum(canon$end - canon$start)
cat("Covered:", total, "\\n", sep="")
'''
        stdout, stderr = run_r_code(r_code)
        r_covered = 0
        for line in stdout.splitlines():
            if line.startswith("Covered:"):
                r_covered = int(line.split(':')[1].strip())

        assert py_res == r_covered
