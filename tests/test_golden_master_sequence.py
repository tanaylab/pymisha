"""
Golden-master tests for sequence functions.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
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

def parse_r_cat_output(stdout):
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip() and not line.strip().startswith('>')]
    if not lines:
        return ""
    # Assuming cat output is on the last line(s) or the only non-echo lines
    return "".join(lines)

class TestGoldenMasterSequence:
    @pytest.fixture(autouse=True)
    def setup(self):
        pm.gdb_init(TESTDB)

    def test_gseq_extract_matches_r(self):
        """gseq_extract matches R gseq.extract output."""
        intervals = pm.gintervals("1", [0, 100], [20, 120])

        # Python result
        py_seqs = pm.gseq_extract(intervals)

        # R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", c(0, 100), c(20, 120))
seqs <- gseq.extract(intervals)
cat(paste(seqs, collapse=","), "\\n")
'''
        stdout, stderr = run_r_code(r_code)
        r_output = parse_r_cat_output(stdout)
        r_seqs = r_output.split(',')

        assert len(py_seqs) == len(r_seqs)
        for py_s, r_s in zip(py_seqs, r_seqs, strict=False):
            # Case-insensitive comparison just in case
            assert py_s.upper() == r_s.upper()

    def test_gseq_extract_revcomp_matches_r(self):
        """gseq_extract with strand=-1 matches R."""
        intervals = pm.gintervals("1", [0, 100], [20, 120])
        intervals['strand'] = -1

        py_seqs = pm.gseq_extract(intervals)

        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals("1", c(0, 100), c(20, 120))
intervals$strand <- -1
seqs <- gseq.extract(intervals)
cat(paste(seqs, collapse=","), "\\n")
'''
        stdout, stderr = run_r_code(r_code)
        r_output = parse_r_cat_output(stdout)
        r_seqs = r_output.split(',')

        assert len(py_seqs) == len(r_seqs)
        for py_s, r_s in zip(py_seqs, r_seqs, strict=False):
            assert py_s.upper() == r_s.upper()

    def test_gseq_pwm_matches_r_manual(self):
        """gseq_pwm matches manual R implementation."""
        # Simple PSSM
        # A C G T
        pssm = np.array([
            [0.1, 0.2, 0.3, 0.4], # Pos 1
            [0.4, 0.3, 0.2, 0.1]  # Pos 2
        ])
        # Log probabilities
        prior = 0.01
        pssm_norm = (pssm + prior) / (pssm + prior).sum(axis=1, keepdims=True)
        np.log(pssm_norm)

        seq = "ACGT"
        # PWM width 2.
        # AC: log(p[0,A]) + log(p[1,C])
        # CG: log(p[0,C]) + log(p[1,G])
        # GT: log(p[0,G]) + log(p[1,T])

        # Calculate in Python using gseq_pwm
        py_score = pm.gseq_pwm(seq, pssm, mode="max", bidirect=False, prior=prior)

        # Calculate in R manually to verify the math logic matches
        r_code = f'''
# PSSM
m <- matrix(c(0.1, 0.4, 0.2, 0.3, 0.3, 0.2, 0.4, 0.1), ncol=4, dimnames=list(NULL, c("A", "C", "G", "T")))
prior <- {prior}
m <- (m + prior) / rowSums(m + prior)
log_m <- log(m)

seq <- "ACGT"
chars <- strsplit(seq, "")[[1]]
scores <- numeric(0)

# Scan
for (i in 1:(length(chars)-1)) {{
    # Window i, i+1
    b1 <- chars[i]
    b2 <- chars[i+1]
    s <- log_m[1, b1] + log_m[2, b2]
    scores <- c(scores, s)
}}
cat("Max:", max(scores), "\\n", sep="")
'''
        stdout, stderr = run_r_code(r_code)
        r_max = None
        for line in stdout.splitlines():
            if line.startswith("Max:"):
                r_max = float(line.split(':')[1].strip())

        assert r_max is not None
        # py_score returns array of scores (one per sequence)
        assert np.isclose(py_score[0], r_max), f"PWM mismatch: {py_score[0]} vs {r_max}"
