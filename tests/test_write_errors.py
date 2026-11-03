"""A file that is not written completely must fail the call and leave nothing behind.

Port of misha's tests/testthat/test-write-errors.R (misha PR #204). Writers used
to close their files without looking at what fclose() returned, and the last
buffer of a file is written there: a failure at that point (a full disk, a
quota, an NFS write error) left a truncated track behind a call that reported
success.

RLIMIT_FSIZE makes a write fail on demand: past the limit write(2) fails with
EFBIG (SIGXFSZ is ignored, so the process is not killed). Every track file here
is smaller than the stdio buffer, so none of it reaches the file before it is
flushed or closed. The limit is per process and inherited by forked children.
"""

import contextlib
import errno
import re
import resource
import signal
import subprocess
import sys
import warnings
from pathlib import Path

import _pymisha
import numpy as np
import pytest

import __main__
import pymisha as pm
from pymisha import db_attrs

_SAVED_LIMIT = resource.getrlimit(resource.RLIMIT_FSIZE)
_SAVED_SIGXFSZ = signal.getsignal(signal.SIGXFSZ)


def _cap(nbytes):
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (nbytes, _SAVED_LIMIT[1]))


def _lift():
    resource.setrlimit(resource.RLIMIT_FSIZE, _SAVED_LIMIT)
    signal.signal(signal.SIGXFSZ, _SAVED_SIGXFSZ)


@contextlib.contextmanager
def file_size_limit(nbytes):
    _cap(nbytes)
    try:
        yield
    finally:
        _lift()


@pytest.fixture(autouse=True)
def _always_lift(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    # A module first imported under the limit would leave a truncated .pyc behind: CPython does
    # not check the length of its bytecode write.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    yield
    _lift()


@pytest.fixture
def write_db(tmp_path):
    """setup_write_db(): a database whose chromosomes have the given sizes."""
    def make(sizes):
        root = tmp_path / "write_db"
        (root / "tracks").mkdir(parents=True)
        (root / "chrom_sizes.txt").write_text("".join(f"{c}\t{n}\n" for c, n in sizes.items()))
        pm.gdb_init(str(root))
        return root
    return make


@pytest.fixture
def fasta_db(tmp_path):
    """A database built by gdb_create, for the writers that need sequence."""
    def make(sizes, db_format):
        fasta = tmp_path / "genome.fa"
        fasta.write_text("".join(f">{c}\n{'A' * n}\n" for c, n in sizes.items()))
        root = tmp_path / "fasta_db"
        pm.gdb_create(str(root), str(fasta), db_format=db_format)
        pm.gdb_init(str(root))
        return root
    return make


@pytest.fixture
def in_main(monkeypatch):
    """Puts a function where a track expression looks names up (R's .GlobalEnv)."""
    def put(name, fn):
        monkeypatch.setattr(__main__, name, fn, raising=False)
    return put


ONES = "START * 0 + 1"  # R's "1": a constant is not an array in pymisha


# 400 bins of 1 bp: a 1604-byte dense file, over the 1000-byte limit; 100 bins: 404 bytes, under it
def expect_write_failure(write_db, sizes, create, failing_chrom):
    write_db(sizes)
    with pytest.raises(_pymisha.error, match=f"Failed to write .*/{failing_chrom}: "), file_size_limit(1000):
        create()
    assert not pm.gtrack_exists("t")
    create()  # and the same call succeeds without the limit
    assert pm.gtrack_exists("t")


def test_failed_write_of_last_chrom_file_fails_gtrack_create(write_db):
    expect_write_failure(write_db, {"chr1": 100, "chr2": 400},
                         lambda: pm.gtrack_create("t", "", ONES, iterator=1), "chr2")


def test_failed_write_of_file_closed_for_next_chrom_fails_gtrack_create(write_db):
    expect_write_failure(write_db, {"chr1": 400, "chr2": 100},
                         lambda: pm.gtrack_create("t", "", ONES, iterator=1), "chr1")


def test_failed_write_fails_gtrack_create_dense(write_db):
    expect_write_failure(
        write_db, {"chr1": 400},
        lambda: pm.gtrack_create_dense("t", "", pm.gintervals("chr1", 0, 400), [1], binsize=1, defval=0),
        "chr1",
    )


def test_failed_write_fails_gtrack_import(write_db, tmp_path):
    bg = tmp_path / "in.bedGraph"
    bg.write_text("chr1\t0\t400\t1\n")
    expect_write_failure(write_db, {"chr1": 400},
                         lambda: pm.gtrack_import("t", "", str(bg), binsize=1, defval=0), "chr1")


def test_failed_write_fails_gtrack_create_dense_on_indexed_db(fasta_db):
    # An indexed database writes track.dat directly (IndexedTrackWriter), not per-chromosome files
    fasta_db({"chr1": 400}, "indexed")
    with pytest.raises(_pymisha.error, match=r"Failed to write .*/track\.dat: "), file_size_limit(1000):
        pm.gtrack_create_dense("t", "", pm.gintervals("chr1", 0, 400), [1], binsize=1, defval=0)
    assert not pm.gtrack_exists("t")
    pm.gtrack_create_dense("t", "", pm.gintervals("chr1", 0, 400), [1], binsize=1, defval=0)
    assert (pm.gextract("t", pm.gintervals_all(), iterator=1)["t"] == 1).all()


def test_failed_write_in_forked_worker_is_reported_and_not_committed(fasta_db):
    # misha's case is a gtrack.create multitasking child. pymisha's C++ children only read;
    # the writers that run in forked children are gdb_convert_to_indexed's workers (threads > 1),
    # which report a failed track as a warning, as misha's do. Two tracks, so that it forks.
    fasta_db({"chr1": 400, "chr2": 400}, "per-chromosome")
    for t in ("t1", "t2"):
        pm.gtrack_create(t, "", ONES, iterator=1)
    with pytest.warns(UserWarning) as warned, file_size_limit(1000):
        pm.gdb_convert_to_indexed(convert_tracks=True, threads=2)
    pm.gdb_init(pm._shared._GROOT)
    for t in ("t1", "t2"):
        assert any(re.match(f"Failed to convert track '{t}': Failed to write .*/track.dat: ", str(w.message))
                   for w in warned)
        # the short track.dat was not published: the per-chromosome files still serve the track
        assert not (Path(_pymisha.pm_track_path(t)) / "track.idx").exists()
        assert (pm.gextract(t, pm.gintervals_all(), iterator=1)[t] == 1).all()


def _modify_case(write_db, in_main, nested):
    write_db({"chr1": 400, "chr2": 100})
    pm.gtrack_create("t", "", "START * 0", iterator=1)
    calls = [0]

    # the cap comes after chr1 has been staged (copied in full), so only its new values fail
    def g(x):
        calls[0] += 1
        if calls[0] == 5:
            _cap(1000)
        if nested:
            # evaluated while gtrack_modify holds its files; the inner call fails and is caught
            with contextlib.suppress(Exception):
                pm.gtrack_info("no_such_track")
        return x + 1

    in_main("g", g)
    with pytest.raises(_pymisha.error, match="Failed to write .*/chr1: "):
        pm.gtrack_modify("t", "g(t)", pm.gintervals_all())
    _lift()
    assert set(pm.gextract("t", pm.gintervals_all(), iterator=1)["t"]) == {0}


def test_call_nested_in_writer_neither_raises_nor_drops_its_write_error(write_db, in_main, monkeypatch):
    # misha nests the call in gtrack.create; pymisha's gtrack_create flushes each chromosome before
    # it closes it, so it raises at once and records nothing a nested call could take. gtrack_modify
    # only records the failure (the staged file is closed when the next chromosome's is opened).
    monkeypatch.setitem(pm.CONFIG, "eval_buf_size", 10)
    _modify_case(write_db, in_main, nested=True)


def test_gtrack_modify_does_not_commit_chrom_whose_staged_file_failed(write_db, in_main, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "eval_buf_size", 10)
    _modify_case(write_db, in_main, nested=False)


def test_failed_write_fails_intervals_set_out_and_gintervals_save(write_db):
    root = write_db({"chr1": 2400, "chr2": 240})
    starts = np.arange(0, 240, 2)
    pm.gtrack_create_sparse(
        "sp", "", pm.gintervals(["chr1"] * 120 + ["chr2"] * 120, np.tile(starts, 2), np.tile(starts, 2) + 1),
        np.ones(240),
    )
    with pytest.raises(OSError, match="File too large"), file_size_limit(1000):
        pm.gscreen("sp > 0", intervals_set_out="out")
    assert not pm.gintervals_exists("out")

    with pytest.raises(OSError, match="File too large"), file_size_limit(1000):
        pm.gintervals_save(pm.gintervals("chr1", starts, starts + 1), "s")
    assert not pm.gintervals_exists("s")
    pm.gintervals_save(pm.gintervals("chr1", starts, starts + 1), "s")
    assert len(pm.gintervals_load("s")) == 120

    # a set too large for the write buffer fails in write() rather than in close()
    many = pm.gintervals("chr1", np.arange(2400), np.arange(1, 2401))
    with pytest.raises(OSError, match="File too large"), file_size_limit(1000):
        pm.gintervals_save(many, "s2")
    assert not (root / "tracks" / "s2.interv").exists()


def test_failed_write_fails_gtrack_array_import(write_db, tmp_path):
    write_db({"chr1": 1000})
    af = tmp_path / "arr.txt"
    cols = [f"sample_column_{i:03d}" for i in range(1, 101)]  # .colnames is over 2000 bytes, the track data under it
    af.write_text("\t".join(["chrom", "start", "end", *cols]) + "\n"
                  + "\t".join(["chr1", "0", "10", *["1"] * 100]) + "\n")
    with pytest.raises(OSError, match=r"File too large: '.*\.colnames'"), file_size_limit(2000):
        pm.gtrack_array_import("arr", "", str(af))
    assert not pm.gtrack_exists("arr")
    pm.gtrack_array_import("arr", "", str(af))
    assert pm.gtrack_array_get_colnames("arr") == cols


def test_failed_gintervals_update_keeps_the_set(write_db):
    # misha drops the set: it rewrites the file in place, so a failed write had already
    # truncated the old one. pymisha writes the new set under a temporary name first, so the
    # old set survives intact and the temporary file is removed.
    root = write_db({"chr1": 1000, "chr2": 1000})
    old = pm.gintervals(["chr1", "chr2"], 0, 10)
    pm.gintervals_save(old, "sm")
    with pytest.raises(OSError, match="File too large"), file_size_limit(500):
        pm.gintervals_update("sm", pm.gintervals("chr1", np.arange(300), np.arange(1, 301)), chrom="chr1")
    assert sorted(p.name for p in (root / "tracks").iterdir()) == ["sm.interv"]
    pm.gdb_reload()
    assert pm.gintervals_ls() == ["sm"]
    got = pm.gintervals_load("sm")
    assert got[["start", "end"]].to_numpy().tolist() == old[["start", "end"]].to_numpy().tolist()


def test_track_variable_that_fails_to_write_midway_is_not_left_behind(write_db):
    write_db({"chr1": 100})
    pm.gtrack_create("t", "", ONES, iterator=1)
    # 16 KB: larger than the write buffer, so write() itself fails, not close()
    rng = np.random.default_rng(60427)
    with pytest.raises(OSError, match="File too large: '.*/vars/v'") as exc, file_size_limit(1000):
        pm.gtrack_var_set("t", "v", rng.random(2000))
    assert exc.value.errno == errno.EFBIG
    assert "v" not in pm.gtrack_var_ls("t")
    pm.gtrack_var_set("t", "v", [1, 2, 3])
    assert pm.gtrack_var_get("t", "v") == [1, 2, 3]


# Writers misha's change covers without a test of its own: a failed write raises and removes the file.
@pytest.mark.parametrize("what", ["track attribute", "intervals attribute",
                                  "readonly attributes (Rscript)", "readonly attributes (no Rscript)"])
def test_failed_attribute_write_removes_the_file(write_db, monkeypatch, tmp_path, what):
    root = write_db({"chr1": 100})
    pm.gtrack_create("t", "", ONES, iterator=1)
    pm.gintervals_save(pm.gintervals("chr1", 0, 10), "s")
    long = "x" * 2000
    ro_attrs = [f"attr_{i:04d}" for i in range(200)]  # serialized: over 2000 bytes
    if what == "readonly attributes (Rscript)":
        rscript = db_attrs._find_trusted_rscript()
        if rscript is None:
            pytest.skip("no trusted Rscript")
        # subprocess restores SIGXFSZ in the child, which would kill R before it writes anything
        wrapper = tmp_path / "Rscript"
        wrapper.write_text(f"#!/bin/sh\ntrap '' XFSZ\nexec '{rscript}' \"$@\"\n")
        wrapper.chmod(0o755)
        monkeypatch.setattr(db_attrs, "_find_trusted_rscript", lambda: str(wrapper))
    if what == "readonly attributes (no Rscript)":
        monkeypatch.setattr(db_attrs, "_find_trusted_rscript", lambda: None)
    write, path, raised = {
        "track attribute": (lambda: pm.gtrack_attr_set("t", "note", long),
                            root / "tracks" / "t.track" / ".attributes", OSError),
        "intervals attribute": (lambda: pm.gintervals_attr_set("s", "note", long),
                                root / "tracks" / "s.iattr", OSError),
        # close() in R only warns when writing the last buffer fails: Rscript exited 0
        "readonly attributes (Rscript)": (lambda: pm.gdb_set_readonly_attrs(ro_attrs),
                                          root / ".ro_attributes", subprocess.CalledProcessError),
        "readonly attributes (no Rscript)": (lambda: pm.gdb_set_readonly_attrs(ro_attrs),
                                             root / ".ro_attributes", OSError),
    }[what]
    with pytest.raises(raised), file_size_limit(1000), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Rscript is not available"
        write()
    assert not path.exists()
