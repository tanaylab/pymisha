import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm


def _make_intervals(num_per_chrom=20, width=100):
    all_intervals = pm.gintervals_all()
    rows = []
    for _, row in all_intervals.iterrows():
        chrom = row["chrom"]
        chrom_end = int(row["end"])
        step = max(width, chrom_end // (num_per_chrom + 1))
        start = 0
        for _ in range(num_per_chrom):
            if start + width > chrom_end:
                break
            rows.append({"chrom": chrom, "start": start, "end": start + width})
            start += step
    return pd.DataFrame(rows)


def test_gextract_multitask_matches_single():
    intervals = _make_intervals()
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "max_processes": 2})
        single = pm.gextract("dense_track", intervals)

        config.update({"multitasking": True, "max_processes": 2})
        multi = pm.gextract("dense_track", intervals)
    finally:
        config.update(saved)

    pdt.assert_frame_equal(single.reset_index(drop=True), multi.reset_index(drop=True))


def test_gscreen_multitask_matches_single():
    intervals = _make_intervals()
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "max_processes": 2})
        single = pm.gscreen("dense_track > 0.2", intervals)

        config.update({"multitasking": True, "max_processes": 2})
        multi = pm.gscreen("dense_track > 0.2", intervals)
    finally:
        config.update(saved)

    if single is None or multi is None:
        assert single is None and multi is None
        return

    pdt.assert_frame_equal(single.reset_index(drop=True), multi.reset_index(drop=True))


def test_gextract_multitask_large_output_matches_single():
    intervals = _make_intervals(num_per_chrom=50, width=10000)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "max_processes": 3})
        single = pm.gextract("dense_track", intervals, iterator=100)

        config.update({"multitasking": True, "max_processes": 3})
        multi = pm.gextract("dense_track", intervals, iterator=100)
    finally:
        config.update(saved)

    pdt.assert_frame_equal(single.reset_index(drop=True), multi.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Single-chrom multitask edge cases (ported from R test-gextract-single-chrom-multitask.R)
# ---------------------------------------------------------------------------


def _sort_df(df):
    """Sort a DataFrame by chrom/start/end/intervalID and reset index."""
    cols = ["chrom", "start", "end"]
    if "intervalID" in df.columns:
        cols.append("intervalID")
    return df.sort_values(cols).reset_index(drop=True)


def test_single_chrom_fixedbin_multitask_matches_serial():
    """Single-chrom fixed-bin multitask split matches serial output.

    Ported from R: test-gextract-single-chrom-multitask.R (line 3).
    When intervals span only a single chromosome, the multitask path should
    still produce exactly the same output as the serial path.
    """
    # Use only chromosome 1 (the largest in the test DB: 500000 bp)
    intervals = pm.gintervals(["1"], 0, 100000)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({
            "multitasking": False,
            "max_processes": 8,
        })
        serial = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])

        config.update({
            "multitasking": True,
            "max_processes": 8,
        })
        multi = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])
    finally:
        config.update(saved)

    pdt.assert_frame_equal(_sort_df(serial), _sort_df(multi))


def test_single_chrom_vtrack_shifted_multitask_matches_serial():
    """Single-chrom shifted vtrack extraction matches serial output.

    Ported from R: test-gextract-single-chrom-multitask.R (line 28).
    A virtual track with shifted windows on a single chromosome should
    produce identical results in serial and parallel modes.
    """
    vt_name = "test_mt_single_chrom_sum"
    # Clean up any pre-existing vtrack
    if vt_name in (pm.gvtrack_ls() or []):
        pm.gvtrack_rm(vt_name)

    try:
        pm.gvtrack_create(vt_name, src="dense_track", func="sum")
        pm.gvtrack_iterator(vt_name, sshift=-100, eshift=100)

        intervals = pm.gintervals(["1"], 0, 100000)
        config = pm.CONFIG
        saved = config.copy()

        try:
            config.update({
                "multitasking": False,
                "max_processes": 8,
            })
            serial = pm.gextract(vt_name, intervals, iterator=20)

            config.update({
                "multitasking": True,
                "max_processes": 8,
            })
            multi = pm.gextract(vt_name, intervals, iterator=20)
        finally:
            config.update(saved)

        serial_s = _sort_df(serial)
        multi_s = _sort_df(multi)

        pdt.assert_frame_equal(
            serial_s[["chrom", "start", "end", "intervalID"]],
            multi_s[["chrom", "start", "end", "intervalID"]],
        )
        pd.testing.assert_series_equal(
            serial_s[vt_name], multi_s[vt_name], rtol=1e-6
        )
    finally:
        if vt_name in (pm.gvtrack_ls() or []):
            pm.gvtrack_rm(vt_name)


def test_small_multichrom_multitask_matches_serial():
    """Small multi-chrom extraction still enters multitask path and matches serial.

    Ported from R: test-gextract-single-chrom-multitask.R (line 91).
    Even a tiny region spanning two chromosomes should produce correct results
    with multitasking enabled.
    """
    intervals = pm.gintervals(["1", "2"], 0, 1000)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({
            "multitasking": False,
            "max_processes": 8,
        })
        serial = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])

        config.update({
            "multitasking": True,
            "max_processes": 8,
        })
        multi = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])
    finally:
        config.update(saved)

    pdt.assert_frame_equal(_sort_df(serial), _sort_df(multi))


# ---------------------------------------------------------------------------
# A kid whose parent is killed ends with it
# ---------------------------------------------------------------------------

# Forks kids that run for minutes: 3 kids of the C++ layer (gquantiles splits the test genome by its
# 3 chromosomes), on a scan that calls no Python, or 4 Python workers (_run_forked) that then send a
# result larger than a pipe holds.
_KIDS_SCRIPT = textwrap.dedent("""
    import sys, time
    import pymisha as pm
    pm.gdb_init(sys.argv[1])
    if sys.argv[2] == "cpp":
        pm.CONFIG.update(multitasking=True, max_processes=4, min_scope4process=0, progress=False)
        # each bin counts the k-mer over 100 kb
        pm.gvtrack_create("slow_kmer", None, func="kmer.count", kmer="A")
        pm.gvtrack_iterator("slow_kmer", sshift=-50000, eshift=50000)
        try:
            pm.gquantiles("slow_kmer", [0.5], pm.gintervals_all(), iterator=1)
        except Exception as e:
            print("ERROR:", e, flush=True)
    else:
        from pymisha._shared import _run_forked
        def work(n):
            time.sleep(1)
            return b"x" * n
        _run_forked(work, [10**7] * 4)
""")
_NUM_KIDS = {"cpp": 3, "python": 4}


def _kids(pid):
    with open(f"/proc/{pid}/task/{pid}/children") as f:
        return [int(x) for x in f.read().split()]


def _runs(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):   # ended before the open, or before the read
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="kids die with their parent on Linux")
@pytest.mark.parametrize("kind", ["cpp", "python"])
def test_kids_end_when_their_parent_is_killed(kind):
    # The kids ran on, then waited forever: those of the C++ layer for the FIFO lock, which the kid
    # that took it first kept when its write to the dead parent failed; the Python workers to send
    # their result, as they hold the read end of their pipe too.
    parent = subprocess.Popen([sys.executable, "-c", _KIDS_SCRIPT, str(TESTDB_ROOT), kind],
                              cwd=str(Path(__file__).resolve().parent.parent))
    kids = []
    left = []
    try:
        deadline = time.monotonic() + 60
        while len(kids) < _NUM_KIDS[kind] and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
            kids = _kids(parent.pid)
        assert len(kids) == _NUM_KIDS[kind] and all(_runs(k) for k in kids), kids
        parent.kill()
        parent.wait()
        deadline = time.monotonic() + 20
        while (left := [k for k in kids if _runs(k)]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not left, f"kids {left} still run after their parent was killed"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for k in left:
            os.kill(k, signal.SIGKILL)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the kids from /proc")
def test_a_killed_kid_fails_the_call():
    # as the out-of-memory killer would: the call fails, it does not return the other kids' result
    parent = subprocess.Popen([sys.executable, "-c", _KIDS_SCRIPT, str(TESTDB_ROOT), "cpp"],
                              cwd=str(Path(__file__).resolve().parent.parent), stdout=subprocess.PIPE, text=True)
    try:
        kids = []
        deadline = time.monotonic() + 60
        while len(kids) < 3 and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
            kids = _kids(parent.pid)
        assert len(kids) == 3, kids
        os.kill(kids[1], signal.SIGKILL)
        out, _ = parent.communicate(timeout=60)
        assert f"ERROR: Child process {kids[1]} ended unexpectedly (signal 9)" in out, out
        assert not [k for k in kids if _runs(k)]
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


# A call whose kids' shared memory cannot be mapped runs in one process (R misha reports an error,
# except in gextract and gscreen). The address space is limited to what the session holds plus
# 256 MB, which the shared memory of these calls exceeds (it is sized by max_data_size, or half the
# RAM); a call whose kids hand over a few hundred bytes (gsummary, gdist, Pearson's gcor) still
# multitasks. With max_data_size at 50 million that is 0.4 to 2.2 GB, which a small host maps when
# the address space is not limited; the default, sized from the RAM, makes gquantiles' kid blocks
# (the implicit iterator's bins are not estimated) more than half the RAM of a 7 GB host.
_NO_SHM_SCRIPT = textwrap.dedent("""
    import resource, sys
    import numpy as np
    import pymisha as pm
    pm.gdb_init(sys.argv[1])
    pm.CONFIG.update(progress=False, max_processes=4, min_scope4process=0, max_data_size=50_000_000)
    scope = pm.gintervals_all()
    calls = {
        "gextract": lambda: pm.gextract(["dense_track", "dense_track + 1"], scope),
        "gscreen": lambda: pm.gscreen("dense_track > 0.2", scope),
        "glookup": lambda: pm.glookup(np.arange(3.0), "dense_track", [0, 0.1, 0.2, 1], intervals=scope),
        "gquantiles": lambda: pm.gquantiles("dense_track", [0.1, 0.5, 0.9], scope),
        "gintervals_quantiles": lambda: pm.gintervals_quantiles("dense_track", [0.1, 0.5], scope),
        "gcor": lambda: pm.gcor("dense_track", "dense_track * 2", intervals=scope, method="spearman.exact"),
    }

    def faults():
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_minflt

    pm.CONFIG["multitasking"] = False
    expected = {name: call() for name, call in calls.items()}
    pm.CONFIG["multitasking"] = True
    for name, call in calls.items():
        before = faults()
        call()
        print("FORKS", name, faults() > before)
    vm = int(open("/proc/self/status").read().split("VmSize:")[1].split()[0]) * 1024
    resource.setrlimit(resource.RLIMIT_AS, (vm + (256 << 20), resource.RLIM_INFINITY))
    for name, call in calls.items():
        before = faults()
        got = call()
        same = got.equals(expected[name]) if hasattr(got, "equals") else np.array_equal(got, expected[name])
        print("NO_SHM", name, faults() > before, same)
""")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="limits the address space and reads /proc")
def test_a_call_whose_shared_memory_cannot_be_mapped_runs_in_one_process():
    proc = subprocess.run([sys.executable, "-c", _NO_SHM_SCRIPT, str(TESTDB_ROOT)], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parent.parent), timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = [line.split() for line in proc.stdout.splitlines()]
    forks = {w[1]: w[2] for w in out if w[0] == "FORKS"}
    no_shm = {w[1]: w[2:] for w in out if w[0] == "NO_SHM"}
    assert len(forks) == len(no_shm) == 6, proc.stdout
    assert all(v == "True" for v in forks.values()), forks   # they multitask otherwise
    assert all(v == ["False", "True"] for v in no_shm.values()), no_shm   # one process, the same result
