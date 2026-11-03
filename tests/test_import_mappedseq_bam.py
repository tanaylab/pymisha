"""End-to-end BAM / gzip / named-pipe tests for gtrack_import_mappedseq.

Port of R misha tests/testthat/test-gtrack.import-bam.R (chrom "chr1" of the
small test DB, as in R). The BAM tests need `samtools` on PATH
and skip otherwise.
"""
import contextlib
import gzip
import itertools
import os
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT

_SAMTOOLS = shutil.which("samtools")
_MKFIFO = shutil.which("mkfifo")
_names = itertools.count()

needs_samtools = pytest.mark.skipif(_SAMTOOLS is None, reason="samtools not on PATH")
needs_mkfifo = pytest.mark.skipif(_MKFIFO is None or os.name != "posix", reason="mkfifo not on PATH")


@pytest.fixture(scope="module", autouse=True)
def _db(tmp_path_factory):
    root = tmp_path_factory.mktemp("bam") / "trackdb" / "test"
    shutil.copytree(TESTDB_ROOT, root)
    pm.gdb_init(str(root))
    yield root


@pytest.fixture(autouse=True)
def _init(_db):
    pm.gdb_init(str(_db))


def _track():
    return f"bam_test_{next(_names)}"


# SAM fixture for chrom "chr1". Two mapped reads on opposite strands and one
# unmapped record on a chromosome the database does not have.
DEFAULT_SAM = (
    "@HD\tVN:1.6\n"
    "@SQ\tSN:chr1\tLN:500000\n"
    "r1\t0\tchr1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    "r2\t16\tchr1\t200\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    "r3\t4\tunknown_chrom\t1\t0\t*\t*\t0\t0\t*\t*\n"
)


def _make_bam(tmp_path: Path, sam_text: str = DEFAULT_SAM) -> Path:
    sam = tmp_path / f"in_{next(_names)}.sam"
    sam.write_text(sam_text)
    bam = tmp_path / f"in_{next(_names)}.bam"
    subprocess.run([_SAMTOOLS, "view", "-b", "-o", str(bam), str(sam)],
                   check=True, capture_output=True)
    return bam


@needs_samtools
def test_import_mappedseq_imports_bam_as_sparse_track(tmp_path):
    stats = pm.gtrack_import_mappedseq(_track(), "BAM sparse", str(_make_bam(tmp_path)),
                                       cols_order=None, remove_dups=True)
    # 2 mapped records on chrom "chr1", 1 unmapped record (FLAG 4 -> chrom "*").
    assert stats["total"]["total.mapped"] == 2
    assert stats["total"]["total.unmapped"] == 1


@needs_samtools
def test_import_mappedseq_imports_bam_as_dense_pileup(tmp_path):
    track = _track()
    stats = pm.gtrack_import_mappedseq(track, "BAM dense", str(_make_bam(tmp_path)),
                                       pileup=10, binsize=10, cols_order=None, remove_dups=True)
    assert stats["total"]["total.mapped"] == 2
    out = pm.gextract(track, pd.DataFrame({"chrom": "chr1", "start": [90, 100, 190, 200], "end": [100, 110, 200, 210]}),
                      iterator=10, colnames=["v"])
    # POS 100 / 200 are 1-based: the forward read covers [99, 109), the reverse
    # read (5' end at 208) covers [199, 209).
    np.testing.assert_allclose(out["v"].to_numpy(dtype=float), [0.1, 0.9, 0.1, 0.9], rtol=1e-6)


@needs_samtools
def test_import_mappedseq_auto_switches_default_cols_order_for_bam(tmp_path):
    # No cols_order: the default (9, 11, 13, 14) is the tab-delimited layout; samtools view
    # emits SAM, so the wrapper switches to SAM, otherwise the import maps 0 reads.
    stats = pm.gtrack_import_mappedseq(_track(), "BAM default cols_order", str(_make_bam(tmp_path)),
                                       remove_dups=True)
    assert stats["total"]["total.mapped"] == 2


@needs_samtools
def test_import_mappedseq_errors_on_explicit_cols_order_with_bam(tmp_path):
    # R errors on any cols.order passed explicitly; pymisha cannot tell an explicit
    # (9, 11, 13, 14) from the default, so this uses another layout.
    track = _track()
    with pytest.raises(ValueError, match="BAM input forces SAM column layout"):
        pm.gtrack_import_mappedseq(track, "BAM explicit cols_order", str(_make_bam(tmp_path)),
                                   cols_order=(1, 2, 3, 4), remove_dups=True)
    assert not pm.gtrack_exists(track)


@needs_samtools
def test_import_mappedseq_surfaces_samtools_not_found_for_bam(tmp_path, monkeypatch):
    bam = _make_bam(tmp_path)
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(Exception, match="samtools is not on PATH") as err:
        pm.gtrack_import_mappedseq(_track(), "BAM no samtools", str(bam), cols_order=None, remove_dups=True)
    assert "conda install" in str(err.value)


def test_import_mappedseq_accepts_gzipped_sam(tmp_path):
    sam_gz = tmp_path / "in.sam.gz"
    with gzip.open(sam_gz, "wt") as fh:
        fh.write(DEFAULT_SAM)
    stats = pm.gtrack_import_mappedseq(_track(), "gzipped SAM", str(sam_gz), cols_order=None, remove_dups=True)
    assert stats["total"]["total.mapped"] == 2
    assert stats["total"]["total.unmapped"] == 1


def _start_background(cmd: str) -> subprocess.Popen:
    # its own process group, so that killing it kills what the shell started too
    return subprocess.Popen(["sh", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def _stop(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


class _Fifo:
    """A named pipe fed by `cat src > fifo` in the background. After 30 s a watchdog keeps opening the
    pipe for writing, so that an import stuck opening or reading it gets end-of-file and fails instead
    of hanging the test run."""

    def __init__(self, tmp_path: Path, src: Path):
        self.path = tmp_path / f"fifo_{next(_names)}"
        subprocess.run([_MKFIFO, str(self.path)], check=True)
        q = f"'{self.path}'"
        self.writer = _start_background(f"cat '{src}' > {q}")
        self.watchdog = _start_background(f"sleep 30; while [ -p {q} ]; do : > {q}; sleep 0.2; done")

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        _stop(self.writer)
        _stop(self.watchdog)
        self.path.unlink(missing_ok=True)


@needs_mkfifo
def test_import_mappedseq_reads_sam_from_a_named_pipe(tmp_path):
    sam = tmp_path / "pipe.sam"
    sam.write_text(DEFAULT_SAM)
    sam_gz = tmp_path / "pipe.sam.gz"
    with gzip.open(sam_gz, "wt") as fh:
        fh.write(DEFAULT_SAM)

    for gz in (False, True):
        with _Fifo(tmp_path, sam_gz if gz else sam) as fifo:
            track = _track()
            stats = pm.gtrack_import_mappedseq(track, "FIFO", str(fifo), cols_order=None)
            assert stats["total"]["total.mapped"] == 2, f"gzipped: {gz}"
            assert stats["total"]["total.unmapped"] == 1, f"gzipped: {gz}"
            assert pm.gtrack_exists(track)


@needs_mkfifo
@needs_samtools
def test_import_mappedseq_rejects_bam_sent_through_a_named_pipe(tmp_path):
    bam = _make_bam(tmp_path)
    with _Fifo(tmp_path, bam) as fifo:
        track = _track()
        with pytest.raises(Exception, match="pipe carrying BAM"):
            pm.gtrack_import_mappedseq(track, "FIFO", str(fifo), cols_order=None)
        assert not pm.gtrack_exists(track)


@needs_samtools
def test_import_mappedseq_bam_sparse_counts_duplicates(tmp_path):
    bam = _make_bam(tmp_path,
        "@HD\tVN:1.6\n"
        "@SQ\tSN:chr1\tLN:500000\n"
        "r1\t0\tchr1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
        "r2\t0\tchr1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"  # dup
        "r3\t16\tchr1\t200\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    )
    stats = pm.gtrack_import_mappedseq(_track(), "BAM sparse", str(bam),
                                       pileup=0, binsize=-1, cols_order=None, remove_dups=True)
    # R parity: total.mapped = raw count (3), dups = 1
    assert stats["total"]["total.mapped"] == 3.0
    assert stats["total"]["total.dups"] == 1.0
