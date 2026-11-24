"""End-to-end BAM / gzip / named-pipe tests for gtrack_import_mappedseq.

Port of R misha tests/testthat/test-gtrack.import-bam.R (chrom "chr1" of the
small test DB, as in R). pymisha reads BAM itself; none of these needs samtools.

The fixtures in fixtures/bam were made with samtools (see fixtures/bam/make-fixtures.sh), and are
the same files as R misha's. Each <name>.view.sam.gz there is `samtools view -h <name>.bam`: the SAM
text pymisha imported when it read BAM through samtools, so importing it gives the result a BAM
import must reproduce.
"""
import contextlib
import gzip
import itertools
import math
import os
import shutil
import signal
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT

_MKFIFO = shutil.which("mkfifo")
_names = itertools.count()
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bam"

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


def _fixture(name: str) -> str:
    return str(_FIXTURES / name)


# SAM fixture for chrom "chr1". Two mapped reads on opposite strands and one
# unmapped record on a chromosome the database does not have. default.bam holds
# the same records (samtools sets the unmapped record's RNAME to "*").
DEFAULT_SAM = (
    "@HD\tVN:1.6\n"
    "@SQ\tSN:chr1\tLN:500000\n"
    "r1\t0\tchr1\t100\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    "r2\t16\tchr1\t200\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\n"
    "r3\t4\tunknown_chrom\t1\t0\t*\t*\t0\t0\t*\t*\n"
)


def _import(path, _db_root, **kwargs):
    """Imports path into a new track; returns its statistics, warnings and the bytes of its files."""
    track = _track()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats = pm.gtrack_import_mappedseq(track, "BAM test", str(path), **kwargs)
    tdir = Path(_db_root) / "tracks" / f"{track.replace('.', '/')}.track"
    files = {f.name: f.read_bytes() for f in sorted(tdir.iterdir()) if f.name != ".attributes"}
    pm.gtrack_rm(track, force=True)
    return {"total": stats["total"], "chrom": stats["chrom"].to_dict("list"),
            "warnings": [str(w.message) for w in caught], "files": files}


_OPTION_SETS = {
    "sparse": {},
    "sparse_all_dups": {"remove_dups": False},
    "dense": {"pileup": 100, "binsize": 20},
    "dense_all_dups": {"pileup": 100, "binsize": 20, "remove_dups": False},
    "min_mapq": {"min_mapq": 30},
    "min_mapq_dense": {"min_mapq": 30, "pileup": 200, "binsize": 50},
    "paired": {"paired": True, "binsize": 50},
    "paired_all_dups": {"paired": True, "binsize": 50, "remove_dups": False},
    "paired_min_mapq": {"paired": True, "binsize": 10, "min_mapq": 30},
    "paired_max_fraglen": {"paired": True, "binsize": 50, "max_fraglen": 300},
    "paired_no_max_fraglen": {"paired": True, "binsize": 50, "max_fraglen": math.inf},
}

# split.bam holds single.bam's records in small BGZF blocks (records and header span blocks)
_FIXTURE_TEXT = {"default": "default", "single": "single", "paired": "paired", "split": "single",
                 "edge": "edge", "empty": "empty"}


@pytest.mark.parametrize("bam", list(_FIXTURE_TEXT))
@pytest.mark.parametrize("opt", list(_OPTION_SETS))
def test_a_bam_gives_the_track_and_statistics_of_its_samtools_view_text(bam, opt, _db):
    args = _OPTION_SETS[opt]
    from_bam = _import(_fixture(f"{bam}.bam"), _db, **args)
    from_sam = _import(_fixture(f"{_FIXTURE_TEXT[bam]}.view.sam.gz"), _db, cols_order=None, **args)
    assert from_bam == from_sam


def test_bam_import_statistics(_db):
    s = _import(_fixture("single.bam"), _db)
    assert s["total"] == {"total": 2555, "total.mapped": 1323, "total.unmapped": 808, "total.dups": 4,
                          "total.filtered": 424}
    assert s["chrom"]["mapped"] == [468, 408, 447]

    s = _import(_fixture("paired.bam"), _db, paired=True, binsize=50)
    assert s["total"] == {"total": 1029, "total.mapped": 466, "total.unmapped": 255, "total.dups": 1,
                          "total.filtered": 308}

    # a mapped record without a CIGAR or with an operation code SAM has no letter for, a record
    # placed without a position and a CIGAR spanning no reference base are unmapped
    s = _import(_fixture("edge.bam"), _db)
    assert s["total"] == {"total": 13, "total.mapped": 5, "total.unmapped": 8, "total.dups": 0,
                          "total.filtered": 0}

    s = _import(_fixture("empty.bam"), _db)
    assert s["total"]["total"] == 0
    assert any("No read was imported" in w for w in s["warnings"])


def test_the_threads_that_inflate_a_bam_do_not_change_the_result(_db, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "max_processes", 1)
    one = _import(_fixture("split.bam"), _db, pileup=100, binsize=20)
    monkeypatch.setitem(pm.CONFIG, "max_processes", 8)
    assert _import(_fixture("split.bam"), _db, pileup=100, binsize=20) == one


def test_import_mappedseq_imports_bam_as_sparse_track():
    stats = pm.gtrack_import_mappedseq(_track(), "BAM sparse", _fixture("default.bam"),
                                       cols_order=None, remove_dups=True)
    # 2 mapped records on chrom "chr1", 1 unmapped record (chrom "*" in the BAM).
    assert stats["total"]["total.mapped"] == 2
    assert stats["total"]["total.unmapped"] == 1


def test_import_mappedseq_imports_bam_as_dense_pileup():
    track = _track()
    stats = pm.gtrack_import_mappedseq(track, "BAM dense", _fixture("default.bam"),
                                       pileup=10, binsize=10, cols_order=None, remove_dups=True)
    assert stats["total"]["total.mapped"] == 2
    out = pm.gextract(track, pd.DataFrame({"chrom": "chr1", "start": [90, 100, 190, 200], "end": [100, 110, 200, 210]}),
                      iterator=10, colnames=["v"])
    # POS 100 / 200 are 1-based: the forward read covers [99, 109), the reverse
    # read (5' end at 208) covers [199, 209).
    np.testing.assert_allclose(out["v"].to_numpy(dtype=float), [0.1, 0.9, 0.1, 0.9], rtol=1e-6)


def test_import_mappedseq_auto_switches_default_cols_order_for_bam():
    # No cols_order: the default (9, 11, 13, 14) is the tab-delimited layout; BAM records are read
    # as SAM, so the wrapper switches to SAM, otherwise the import maps 0 reads.
    stats = pm.gtrack_import_mappedseq(_track(), "BAM default cols_order", _fixture("default.bam"),
                                       remove_dups=True)
    assert stats["total"]["total.mapped"] == 2


def test_import_mappedseq_errors_on_explicit_cols_order_with_bam():
    # R errors on any cols.order passed explicitly; pymisha cannot tell an explicit
    # (9, 11, 13, 14) from the default, so this uses another layout.
    track = _track()
    with pytest.raises(ValueError, match="BAM input forces SAM column layout"):
        pm.gtrack_import_mappedseq(track, "BAM explicit cols_order", _fixture("default.bam"),
                                   cols_order=(1, 2, 3, 4), remove_dups=True)
    assert not pm.gtrack_exists(track)


def test_bam_import_does_not_need_samtools(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    stats = pm.gtrack_import_mappedseq(_track(), "BAM no samtools", _fixture("default.bam"))
    assert stats["total"]["total.mapped"] == 2


def _block_size(data: bytes, at: int) -> int:
    return data[at + 16] + 256 * data[at + 17] + 1


def _corrupt_cases(tmp_path):
    bam = Path(_fixture("single.bam")).read_bytes()
    second = _block_size(bam, 0)
    bad_crc = bytearray(bam)
    bad_crc[second + _block_size(bam, second) - 8] ^= 0xFF

    def write(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return path

    half = len(bam) // 2
    return [
        (write("half.bam", bam[:half]), "does not end with the BGZF end-of-file block"),
        (write("no_eof.bam", bam[:-28]), "does not end with the BGZF end-of-file block"),
        (write("half_eof.bam", bam[:half] + bam[-28:]), "last BGZF block is incomplete"),
        (write("bad_crc.bam", bytes(bad_crc)), "CRC32 mismatch"),
        (_fixture("bad-truncated-record.bam"), "last record is incomplete"),
        (_fixture("bad-cigar-length.bam"), "CIGAR and query sequence lengths differ for bad"),
        (_fixture("bad-refid.bam"), "refers to a reference that its header does not list"),
    ]


def test_a_truncated_or_corrupt_bam_is_an_error_and_leaves_no_track(tmp_path):
    for path, msg in _corrupt_cases(tmp_path):
        track = _track()
        with pytest.raises(Exception, match=msg):
            pm.gtrack_import_mappedseq(track, "corrupt", str(path))
        assert not pm.gtrack_exists(track), str(path)


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
    # PermissionError: macOS refuses to signal a group left with only zombies
    with contextlib.suppress(ProcessLookupError, PermissionError):
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
def test_import_mappedseq_rejects_bam_sent_through_a_named_pipe(tmp_path):
    with _Fifo(tmp_path, Path(_fixture("default.bam"))) as fifo:
        track = _track()
        with pytest.raises(Exception, match="pipe carrying BAM"):
            pm.gtrack_import_mappedseq(track, "FIFO", str(fifo), cols_order=None)
        assert not pm.gtrack_exists(track)
