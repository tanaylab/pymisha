"""
Golden-master tests for liftover and lookup.
"""
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
    return pd.read_csv(StringIO(chr(10).join(lines[header_idx:])), sep=chr(9))

def _write_chain(path, entries):
    """Write a chain file."""
    NL = chr(10)
    TAB = chr(9)
    with open(path, "w") as f:
        for hdr, blocks in entries:
            f.write(
                f"chain {hdr['score']} "
                f"{hdr['src_chrom']} {hdr['src_size']} {hdr['src_strand']} {hdr['src_start']} {hdr['src_end']} "
                f"{hdr['tgt_chrom']} {hdr['tgt_size']} {hdr['tgt_strand']} {hdr['tgt_start']} {hdr['tgt_end']} "
                f"{hdr['chain_id']}{NL}"
            )
            for blk in blocks:
                if len(blk) == 3:
                    f.write(f"{blk[0]}{TAB}{blk[1]}{TAB}{blk[2]}{NL}")
                else:
                    f.write(f"{blk[0]}{NL}")
            f.write(f"{NL}")

class TestGoldenMasterLiftover:
    @pytest.fixture(autouse=True)
    def setup(self):
        pm.gdb_init(TESTDB)

    def test_glookup_matches_r(self):
        """glookup matches R glookup output (lookup table usage)."""
        intervals = pm.gintervals("1", [0, 100, 500], [100, 200, 600])

        # Define a lookup table based on dense_track values
        breaks = [0.0, 0.5, 1.0]
        table = [10, 20]

        # Python: default iterator (track)
        py_res = pm.glookup(table, "dense_track", breaks, intervals=intervals)

        # R: default iterator (track)
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", c(0, 100, 500), c(100, 200, 600))
breaks <- c(0.0, 0.5, 1.0)
table <- c(10, 20)
res <- glookup(table, "dense_track", breaks, intervals=intervals)
write.table(res, sep="\\t", quote=FALSE, row.names=FALSE)
'''
        stdout, stderr = run_r_code(r_code)
        r_df = parse_r_table(stdout)

        # Compare
        assert len(py_res) == len(r_df)
        np.testing.assert_allclose(py_res['value'], r_df['value'], rtol=1e-6)

    def test_gintervals_liftover_matches_r(self, tmp_path):
        """gintervals_liftover matches R output."""
        chain_path = os.path.join(str(tmp_path), "test.chain")
        entries = [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 1000,
              "src_strand": "+", "src_start": 0, "src_end": 100,
              "tgt_chrom": "1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
              "chain_id": 1},
             [(100,)]),
        ]
        _write_chain(chain_path, entries)

        # Use DataFrame directly
        src = pd.DataFrame({"chrom": ["srcA", "srcA"], "start": [0, 50], "end": [20, 70]})

        # Python
        chain_py = pm.gintervals_load_chain(chain_path)
        py_res = pm.gintervals_liftover(src, chain_py)

        # R
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
chain <- gintervals.load_chain("{chain_path}")
intervals <- data.frame(chrom=c("srcA", "srcA"), start=c(0, 50), end=c(20, 70))
res <- gintervals.liftover(intervals, chain)
write.table(res, sep="\\t", quote=FALSE, row.names=FALSE)
'''
        stdout, stderr = run_r_code(r_code)
        r_df = parse_r_table(stdout)

        assert len(py_res) == len(r_df)
        assert list(py_res['start']) == list(r_df['start'])
        assert list(py_res['end']) == list(r_df['end'])
