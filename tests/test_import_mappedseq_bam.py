"""End-to-end BAM import tests for gtrack_import_mappedseq.

These tests require `samtools` on PATH. Skipped otherwise.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT
_SAMTOOLS = shutil.which("samtools")

pytestmark = pytest.mark.skipif(
    _SAMTOOLS is None, reason="samtools not on PATH"
)


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _make_bam(tmp_path: Path, sam_text: str) -> Path:
    sam = tmp_path / "in.sam"
    sam.write_text(sam_text)
    bam = tmp_path / "in.bam"
    subprocess.run([_SAMTOOLS, "view", "-b", "-o", str(bam), str(sam)],
                   check=True, capture_output=True)
    return bam


def test_import_mappedseq_bam_dense(tmp_path):
    root = _copy_db(tmp_path)
    bam = _make_bam(tmp_path,
        "@HD\tVN:1.6\n"
        "@SQ\tSN:1\tLN:500000\n"
        "r1\t0\t1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
        "r2\t16\t1\t200\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    )
    try:
        pm.gdb_init(str(root))
        stats = pm.gtrack_import_mappedseq(
            "bam_dense", "BAM dense", str(bam),
            pileup=10, binsize=10, cols_order=None, remove_dups=True,
        )
        assert stats["total"]["total.mapped"] == 2.0
        out = pm.gextract(
            "bam_dense",
            pd.DataFrame({"chrom": ["1", "1"], "start": [100, 200], "end": [110, 210]}),
            iterator=10,
        )
        np.testing.assert_allclose(out["bam_dense"].to_numpy(dtype=float), np.array([1.0, 1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_import_mappedseq_bam_sparse(tmp_path):
    root = _copy_db(tmp_path)
    bam = _make_bam(tmp_path,
        "@HD\tVN:1.6\n"
        "@SQ\tSN:1\tLN:500000\n"
        "r1\t0\t1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
        "r2\t0\t1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"  # dup
        "r3\t16\t1\t200\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    )
    try:
        pm.gdb_init(str(root))
        stats = pm.gtrack_import_mappedseq(
            "bam_sparse", "BAM sparse", str(bam),
            pileup=0, binsize=-1, cols_order=None, remove_dups=True,
        )
        # R parity: total.mapped = raw count (3), dups = 1
        assert stats["total"]["total.mapped"] == 3.0
        assert stats["total"]["total.dups"] == 1.0
    finally:
        pm.gdb_init(str(TEST_DB))


def test_import_mappedseq_bam_default_cols_order_switches_to_sam(tmp_path):
    """User passes BAM file with the legacy default cols_order=(9,11,13,14).
    The wrapper silently switches to SAM mode (cols_order=None) because BAM
    payload via `samtools view` is always SAM format."""
    root = _copy_db(tmp_path)
    bam = _make_bam(tmp_path,
        "@HD\tVN:1.6\n"
        "@SQ\tSN:1\tLN:500000\n"
        "r1\t0\t1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    )
    try:
        pm.gdb_init(str(root))
        stats = pm.gtrack_import_mappedseq(
            "bam_default", "BAM default", str(bam),
            pileup=10, binsize=10,  # no cols_order kwarg = default (9,11,13,14)
        )
        # Auto-switched to SAM mode -> parsed successfully -> 1 read mapped
        assert stats["total"]["total.mapped"] == 1.0
    finally:
        pm.gdb_init(str(TEST_DB))


def test_import_mappedseq_bam_no_samtools(tmp_path, monkeypatch):
    """If samtools is not on PATH, BAM input raises a clear error."""
    root = _copy_db(tmp_path)
    bam = _make_bam(tmp_path,
        "@HD\tVN:1.6\n"
        "@SQ\tSN:1\tLN:500000\n"
        "r1\t0\t1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    )
    monkeypatch.setenv("PATH", "/nonexistent")
    try:
        pm.gdb_init(str(root))
        # C++ side surfaces a pymisha.error (TGLException-derived) with the
        # actionable install hint. Match the hint substring so test stays
        # green if the prefix wording is later tweaked.
        with pytest.raises(Exception, match="samtools is not on PATH"):
            pm.gtrack_import_mappedseq(
                "bam_no_samtools", "BAM", str(bam),
                pileup=10, binsize=10, cols_order=None,
            )
    finally:
        pm.gdb_init(str(TEST_DB))
