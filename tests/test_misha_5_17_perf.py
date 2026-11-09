"""Cases of R misha 5.17.0's performance PR (#225) for the code pymisha shares with it.

R's test.fixedbin is dense_track in pymisha's test database, whose chromosomes are much
shorter than R's (500 kb and 300 kb), so the R tests' scopes are scaled down. R
expressions are written with numpy: ifelse(c, a, b) is np.where(c, a, b), keeping NaN
where R's ifelse keeps NA.
"""

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest
from _chromdb_helpers import create_test_db
from _dbpath import TESTDB_ROOT

import pymisha as pm
from pymisha import _shared

# ---------------------------------------------------------------------------
# Range split (test-gextract-single-chrom-multitask.R)
# ---------------------------------------------------------------------------


def test_range_split_gives_every_process_its_share_the_last_one_too():
    # 1000 intervals of 1 kb into 6 processes: 166.7 kb each. Dividing only what
    # was left after a process's own intervals made each stop at 1/7 of the rest
    # and left the last one 2/7 of the whole.
    shares = _pymisha.pm_split_by_range_shares([1000.0] * 1000, 6, 1)
    assert len(shares) == 6
    assert sum(shares) == 1e6
    assert max(shares) / np.mean(shares) <= 1.01
    # one interval, split within itself
    one = _pymisha.pm_split_by_range_shares([1e6], 6, 1)
    assert max(one) / np.mean(one) <= 1.01
    # contigs of unequal lengths into 16 processes
    rng = np.random.default_rng(60427)
    lens = np.round(rng.exponential(2e4, 5000)) + 1
    s = _pymisha.pm_split_by_range_shares(lens.tolist(), 16, 1)
    assert len(s) == 16
    assert sum(s) == lens.sum()
    assert max(s) / np.mean(s) <= 1.01


# ---------------------------------------------------------------------------
# Multitask gextract (test-gextract-direct-to-r.R). R's second case, the peak memory of
# R's gather from the kids' shared memory, has no counterpart: pymisha's kids stream their
# rows to the parent.
# ---------------------------------------------------------------------------


def test_multitask_gextract_returns_the_data_frame_of_the_serial_path_1d_and_2d():
    saved = dict(pm.CONFIG)
    pm.CONFIG.update(progress=False, max_processes=8, min_processes=8, min_scope4process=1,
                     min_intervs4process=0)
    sizes = {"1": 500000, "2": 300000, "X": 200000}
    # 22 intervals that do not touch, so that the scope is split between processes
    big = pd.concat([pm.gintervals(c, np.arange(0, n - 40000, 41000), np.arange(0, n - 40000, 41000) + 40000)
                     for c, n in sizes.items()], ignore_index=True)
    bigx = big.assign(label=[f"l{i}" for i in range(len(big))],
                      f=pd.Categorical(np.where(np.arange(len(big)) % 2 == 0, "x", "y")))
    calls = [
        lambda: pm.gextract(["dense_track", "dense_track * 2"], big, iterator=70, colnames=["a", "b"]),
        lambda: pm.gextract("dense_track", big, iterator=70, intervals_join="none"),
        lambda: pm.gextract("dense_track", bigx, iterator=70, intervals_join="intervals"),
        lambda: pm.gextract("rects_track", pm.gintervals_2d_all()),
        lambda: pm.gextract("dense_track", pm.gintervals(["1", "2"]),
                            iterator=pd.concat([pm.gintervals("1", 1000, 1100), pm.gintervals("2", 5000, 5100)])),
        lambda: pm.gextract("dense_track", pm.gintervals(["1", "2"]),
                            iterator=pd.concat([pm.gintervals("1", 1000, 1100), pm.gintervals("1", 3000, 3100),
                                                pm.gintervals("2", 5000, 5100)])),
    ]
    try:
        for f in calls:
            pm.CONFIG["multitasking"] = False
            serial = f()
            pm.CONFIG["multitasking"] = True
            pd.testing.assert_frame_equal(f(), serial, check_exact=True)
    finally:
        pm.CONFIG.clear()
        pm.CONFIG.update(saved)


# ---------------------------------------------------------------------------
# Track names in expressions (test-gextract3.R). R's backticks and $ have no Python
# counterpart.
# ---------------------------------------------------------------------------


def test_an_expression_uses_the_tracks_it_names_as_whole_identifiers(tmp_path):
    db = tmp_path / "test"
    shutil.copytree(TESTDB_ROOT, db)
    pm.gdb_init(str(db))
    iv = pm.gintervals("1", 0, 200)
    pm.gtrack_create_dense("dense_track_1bp", "x", pm.gintervals("1", np.arange(200), np.arange(1, 201)),
                           np.arange(200, dtype=float), 1)
    # dense_track is a prefix of dense_track_1bp: only the named track sets the iterator
    assert len(pm.gextract("dense_track_1bp", iv)) == 200
    assert len(pm.gextract("dense_track", iv)) == 4
    assert len(pm.gextract("dense_track_1bp*2", iv)) == 200
    base = pm.gextract("dense_track", iv, colnames=["v"])["v"].to_numpy()
    # in a string: a string
    got = pm.gextract("dense_track + ('dense_track' == 'dense' + '_track')", iv, colnames=["v"])["v"].to_numpy()
    np.testing.assert_array_equal(got, base + 1)
    got = pm.gextract("np.where(np.isnan(dense_track), -1, dense_track)", iv, colnames=["v"])["v"].to_numpy()
    np.testing.assert_array_equal(got, np.where(np.isnan(base), -1, base))
    # a name that runs on into more identifier characters is another name
    with pytest.raises(ValueError):
        pm.gextract("dense_trackx", iv)
    with pytest.raises(ValueError):
        pm.gextract("xdense_track", iv)

    try:
        pm.gvtrack_create("vt_avg", "dense_track", "avg")
        pm.gvtrack_create("vt_avg2", "dense_track", "avg")
        pm.gvtrack_iterator("vt_avg2", sshift=-100, eshift=100)
        v1 = pm.gextract("vt_avg", iv, iterator=50, colnames=["v"])["v"].to_numpy()
        np.testing.assert_array_equal(v1, base)
        both = pm.gextract(["vt_avg2 - vt_avg", "vt_avg"], iv, iterator=50, colnames=["d", "a"])
        np.testing.assert_array_equal(both["a"].to_numpy(), base)
        v2 = pm.gextract("vt_avg2", iv, iterator=50, colnames=["v"])["v"].to_numpy()
        np.testing.assert_array_equal(both["d"].to_numpy(), v2 - base)
        # a virtual track whose source names no track
        with pytest.raises(Exception, match="does not exist"):
            pm.gvtrack_create("vt_bad", "no.such.track", "avg")
            pm.gextract("vt_bad", iv, iterator=50)
    finally:
        pm.gvtrack_clear()
    # an iterator naming no track or interval set
    with pytest.raises(Exception):
        pm.gextract("dense_track", iv, iterator="no.such.track")


# ---------------------------------------------------------------------------
# A track.idx that appears behind the session's back (test-index-cache-invalidation.R)
# ---------------------------------------------------------------------------


def test_a_per_chromosome_track_that_another_session_converts_to_indexed_is_read_from_its_index(tmp_path):
    # "Not indexed" is cached too (a per-chromosome track is opened once per chromosome); the
    # cached answer must not outlive a track.idx that appears behind this session's back.
    db = create_test_db(tmp_path / "db", [("chr1", 10000), ("chr2", 10000), ("chr3", 10000)])
    pm.gdb_init(db)
    rng = np.random.default_rng(60427)
    starts = np.sort(rng.choice(np.arange(0, 9000, 10), 200, replace=False))
    ivs = (pd.DataFrame({"chrom": rng.choice(["chr1", "chr2", "chr3"], 200), "start": starts, "end": starts + 5})
           .sort_values(["chrom", "start"]).reset_index(drop=True))
    pm.gtrack_create_dense("pc_track", "test", ivs, np.ones(len(ivs)), 200, np.nan)
    pm.gtrack_create_dense("ix_track", "test", ivs, np.full(len(ivs), 2.0), 200, np.nan)
    pm.gtrack_convert_to_indexed("ix_track")
    iv = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 0], "end": [5000, 5000]})
    before = pm.gextract("pc_track", iv, iterator=200, colnames=["v"])
    assert set(before["v"].dropna()) == {1}

    # what another session's conversion leaves: track.idx + track.dat, no per-chromosome files
    pc_dir = Path(db) / "tracks" / "pc_track.track"
    ix_dir = Path(db) / "tracks" / "ix_track.track"
    for c in ("chr1", "chr2", "chr3"):
        (pc_dir / c).unlink()
    for f in ("track.idx", "track.dat"):
        shutil.copy(ix_dir / f, pc_dir / f)

    after = pm.gextract("pc_track", iv, iterator=200, colnames=["v"])
    assert len(after) == len(before)
    assert set(after["v"].dropna()) == {2}


# ---------------------------------------------------------------------------
# Zero quantiles (test-gquantiles.R, test-gintervals.quantiles.R, test-gbins.R)
# ---------------------------------------------------------------------------

_SIGNED_ZEROS = "np.where(np.isnan(dense_track), np.nan, np.where(dense_track > 0.1, -0.0, 0.0))"


@pytest.fixture
def config():
    saved = dict(pm.CONFIG)
    pm.CONFIG.update(max_data_size=int(1e9), progress=False)
    yield pm.CONFIG
    pm.CONFIG.clear()
    pm.CONFIG.update(saved)


def _serial_and_parallel(config, f, kids):
    config.update(multitasking=False)
    serial = f()
    config.update(multitasking=True, max_processes=kids, min_processes=kids,
                  min_scope4process=0, min_intervs4process=0)
    parallel = f()
    return serial, parallel


def _same_bits(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (
        a.shape == b.shape
        and np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(a[~np.isnan(a)].view(np.uint64), b[~np.isnan(b)].view(np.uint64))
    )


def _all_plus_zero(vals):
    vals = np.asarray(vals, dtype=float).ravel()
    zeros = vals[vals == 0]
    return len(zeros) > 0 and not np.signbit(zeros).any()


def test_multitask_gquantiles_matches_serial_with_ties_infinities_and_signed_zeros(config):
    # chromosomes 1 and 2 in 8 intervals that do not touch, so that 8 processes fork: the
    # whole chromosomes of the test database hold too few bins for that
    iv = pm.gintervals(["1"] * 5 + ["2"] * 3, [0, 100000, 200000, 300000, 400000, 0, 100000, 200000],
                       [99950, 199950, 299950, 399950, 499950, 99950, 199950, 299950])
    pcts = [0, 1e-7, 0.001, 0.1, 0.5, 0.75, 0.999, 1 - 1e-9, 1]
    exprs = [
        "np.round(dense_track * 4)",
        "np.where(dense_track > 0.15, np.inf, np.where(dense_track < 0.02, -np.inf, dense_track))",
        "np.where(dense_track > 0.1, -0.0, dense_track)",
        "dense_track * 0 + 2",
        "-dense_track",
        _SIGNED_ZEROS,
    ]
    for e in exprs:
        serial, parallel = _serial_and_parallel(config, lambda e=e: pm.gquantiles(e, pcts, iv).to_numpy(), 8)
        # bit for bit: == takes -0 and 0 as equal
        assert _same_bits(parallel, serial), e
        # a zero quantile is 0, not -0, whichever zeros the values hold
        assert not np.signbit(parallel[parallel == 0]).any(), e
    for e in ("np.where(dense_track > 0.1, -0.0, dense_track)", _SIGNED_ZEROS):
        assert _all_plus_zero(pm.gquantiles(e, pcts, iv)), e


def test_gintervals_quantiles_returns_plus_zero_single_process_and_multitask(config):
    intervs = pm.gintervals(["1", "1", "2", "2"], [0, 300000, 0, 150000], [200000, 500000, 100000, 300000])
    pcts = [0, 0.001, 0.1, 0.5, 0.75, 0.999, 1]
    for e in (_SIGNED_ZEROS, "np.where(dense_track > 0.1, -0.0, dense_track)"):
        serial, parallel = _serial_and_parallel(
            config, lambda e=e: pm.gintervals_quantiles(e, pcts, intervs), 4)
        vals_serial = serial.iloc[:, 3:].to_numpy(dtype=float)
        vals = parallel.iloc[:, 3:].to_numpy(dtype=float)
        assert _same_bits(vals, vals_serial), e
        assert _all_plus_zero(vals), e


def test_gintervals_quantiles_returns_plus_zero_on_the_extract_path(config):
    # a DataFrame iterator goes through gextract and pandas' quantiles
    intervs = pm.gintervals(["1", "2"], [0, 0], [200000, 100000])
    res = pm.gintervals_quantiles(_SIGNED_ZEROS, [0, 0.5, 1], intervs,
                                  iterator=pm.giterator_intervals(intervals=intervs, iterator=1000))
    assert _all_plus_zero(res.iloc[:, 3:].to_numpy(dtype=float))


def test_gbins_quantiles_returns_plus_zero():
    res = pm.gbins_quantiles("dense_track", [0, 0.1, 0.2, 0.5, 1], expr=_SIGNED_ZEROS,
                             percentiles=[0, 0.5, 1], intervals=pm.gintervals(["1", "2"]))
    assert _all_plus_zero(res)


# ---------------------------------------------------------------------------
# Fork-cost warning (test-multitask-fork-cost.R)
#
# Once per session means once per Python process, so each case runs in a fresh one.
# MISHA_TEST_FORK_4K_BYTES stands in for the memory of the session: "A" for A bytes in
# 4 KB pages, "A,H" for A bytes of which H are in huge pages.
# ---------------------------------------------------------------------------

_FORK_PRELUDE = textwrap.dedent("""
    import sys, warnings
    import pandas as pd
    import pymisha as pm
    pm.gdb_init(sys.argv[1])
    pm.CONFIG.update(multitasking=True, max_processes=4, min_processes=4, min_scope4process=1,
                     min_intervs4process=0, progress=False)
    # 8 intervals that do not touch (touching ones would be merged into one per chromosome):
    # as many processes as max_processes, up to 8
    SCOPE = pd.DataFrame({"chrom": ["1"] * 5 + ["2"] * 3,
                          "start": [0, 100000, 200000, 300000, 400000, 0, 100000, 200000],
                          "end": [90000, 190000, 290000, 390000, 490000, 90000, 190000, 290000]})

    def fork_warnings(f):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f()
        texts = [str(x.message) for x in w if "4 KB pages" in str(x.message)]
        for t in texts:
            print("TEXT", t)
        return len(texts)
""")


def _run_fresh(lines, fork_bytes=None):
    env = dict(os.environ)
    env.pop("MISHA_TEST_FORK_4K_BYTES", None)
    if fork_bytes is not None:
        env["MISHA_TEST_FORK_4K_BYTES"] = fork_bytes
    proc = subprocess.run(
        [sys.executable, "-c", _FORK_PRELUDE + textwrap.dedent(lines), str(TESTDB_ROOT)],
        capture_output=True, text=True, env=env, cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.splitlines()


def _counts(out, key):
    return [int(line.split()[1]) for line in out if line.startswith(key + " ")]


def test_session_holding_much_memory_in_4k_pages_is_warned_once():
    out = _run_fresh("""
        print("FIRST", fork_warnings(lambda: pm.gsummary("dense_track", SCOPE)))
        print("SECOND", fork_warnings(lambda: pm.gsummary("dense_track", SCOPE)))
        print("EXTRACT", fork_warnings(lambda: pm.gextract("dense_track", SCOPE, iterator=1000)))
    """, "64000000000")
    assert _counts(out, "FIRST") == [1], out
    assert _counts(out, "SECOND") == [0]
    assert _counts(out, "EXTRACT") == [0]
    text = " ".join(line for line in out if line.startswith("TEXT "))
    assert "This Python session holds 64.0 GB of memory in 4 KB pages" in text
    assert "Each of the 4 processes" in text
    assert "max_processes" in text
    assert "once per session" in text


def test_no_fork_cost_warning_for_a_small_session_few_gb_or_no_multitasking():
    call = 'print("N", fork_warnings(lambda: pm.gsummary("dense_track", SCOPE)))\n'
    # this session, as it is: small
    assert _counts(_run_fresh(call), "N") == [0]
    # 8 GB in 4 KB pages with 4 processes: forks of about a second, below the threshold
    assert _counts(_run_fresh(call, "8000000000"), "N") == [0]
    # nothing is forked
    assert _counts(_run_fresh("pm.CONFIG['multitasking'] = False\n" + call, "64000000000"), "N") == [0]


def test_fork_cost_warning_is_shown_by_a_call_that_then_fails_and_not_again():
    # R queues the warning with the call's diagnostics, which a failed call drops, so there
    # the next call warns. pymisha raises it before forking, so the failing call shows it.
    # The expression fails in every kid: the arrays it reads hold fewer than 10^9 values.
    out = _run_fresh("""
        def failing():
            try:
                pm.gsummary("dense_track[1000000000]", SCOPE)
            except Exception as e:
                print("FAILED", type(e).__name__)
                return
            print("FAILED none")
        print("FAILING", fork_warnings(failing))
        print("FIRST", fork_warnings(lambda: pm.gsummary("dense_track", SCOPE)))
    """, "64000000000")
    assert any(line.startswith("FAILED ") and line != "FAILED none" for line in out), out
    assert _counts(out, "FAILING") == [1], out
    assert _counts(out, "FIRST") == [0], out


def test_session_checked_with_few_processes_is_checked_again_with_more():
    # 50 GB of anonymous memory, 30 GB of it in huge pages: 20 GB in 4 KB pages, below the
    # threshold (GB x processes of 128) with 4 processes and above it with 8. The first call
    # reads how much is in huge pages; the second must not reuse its verdict.
    out = _run_fresh("""
        print("FEW", fork_warnings(lambda: pm.gsummary("dense_track", SCOPE)))
        pm.CONFIG.update(max_processes=8, min_processes=8)
        print("MORE", fork_warnings(lambda: pm.gsummary("dense_track", SCOPE)))
    """, "50000000000,30000000000")
    assert _counts(out, "FEW") == [0], out
    assert _counts(out, "MORE") == [1], out
    text = " ".join(line for line in out if line.startswith("TEXT "))
    assert "holds 20.0 GB of memory in 4 KB pages" in text
    assert "Each of the 8 processes" in text


# ---------------------------------------------------------------------------
# gtrack_create_dense (test-gtrack.create_dense.R)
# ---------------------------------------------------------------------------


def test_create_dense_on_the_bin_grid_gives_the_bytes_of_the_general_aggregation(tmp_path):
    # One interval per bin, sorted, skips the sort. An extra NaN interval off the grid adds
    # nothing to any bin but makes the intervals overlap: the files must be the same.
    sizes = [("chr1", 10007), ("chr2", 503), ("chr3", 7)]
    db = create_test_db(tmp_path / "db", sizes)
    pm.gdb_init(db)
    binsize = 10
    grid = pd.concat([
        pd.DataFrame({"chrom": chrom, "start": np.arange(0, size, binsize),
                      "end": np.minimum(np.arange(0, size, binsize) + binsize, size)})
        for chrom, size in sizes[:2]
    ], ignore_index=True)
    rng = np.random.default_rng(60427)
    vals = rng.standard_normal(len(grid)) * 10 ** rng.uniform(-3, 3, len(grid))
    vals[rng.choice(len(grid), 100, replace=False)] = np.nan
    vals[:3] = [-0.0, np.inf, -np.inf]
    keep = np.ones(len(grid), dtype=bool)
    keep[499:520] = False
    grid, vals = grid[keep].reset_index(drop=True), vals[keep]
    off = pd.concat([grid.assign(v=vals),
                     pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [3, 3], "end": [4, 4], "v": np.nan})])
    off = off.sort_values(["chrom", "start"], kind="mergesort").reset_index(drop=True)

    def files(track):
        d = Path(db) / "tracks" / f"{track}.track"
        return [(d / c).read_bytes() for c, _ in sizes]

    for func in ("weighted.mean", "min", "max", "median"):
        for defval in (np.nan, 0, 0.1, -0.0):
            pm.gtrack_create_dense("on_grid", "x", grid, vals, binsize, defval, func=func)
            pm.gtrack_create_dense("off_grid", "x", off[["chrom", "start", "end"]], off["v"].to_numpy(),
                                   binsize, defval, func=func)
            assert files("on_grid") == files("off_grid"), (func, defval)
            pm.gtrack_rm("on_grid", force=True)
            pm.gtrack_rm("off_grid", force=True)


# ---------------------------------------------------------------------------
# Threads (test-worker-threads.R)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux") or os.geteuid() == 0,
                    reason="needs Linux's RLIMIT_NPROC, which root ignores")
def test_create_sparse_works_when_no_writer_thread_can_start(tmp_path):
    # The empty chromosome files of a sparse track are written on a few threads. A thread
    # that cannot be started must leave its share to the calling thread, not fail the call.
    db = create_test_db(tmp_path / "db", [("chr1", 1000), ("chr2", 1000), ("chr3", 1000), ("chr4", 1000)])
    code = textwrap.dedent("""
        import os, resource, subprocess, sys
        import pandas as pd
        import pymisha as pm
        pm.gdb_init(sys.argv[1])
        pm.CONFIG.update(multitasking=True, max_processes=8)
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        try:
            subprocess.run(["true"], check=True)
            print("NO_LIMIT")
            sys.exit(0)
        except (OSError, subprocess.CalledProcessError):
            pass
        pm.gtrack_create_sparse("s", "x", pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]}), [1.0])
        files = sorted(f for f in os.listdir(sys.argv[1] + "/tracks/s.track") if not f.startswith("."))
        print("FILES", " ".join(files))
    """)
    proc = subprocess.run([sys.executable, "-c", code, db], capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parent.parent))
    if "NO_LIMIT" in proc.stdout:
        pytest.skip("the process limit did not take")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FILES chr1 chr2 chr3 chr4" in proc.stdout


# ---------------------------------------------------------------------------
# The track list, updated name by name (test-db-cache-provenance.R). A track creation,
# removal or move lists or drops just its names, as R's .gdb.add_track / .gdb.rm_track,
# instead of rescanning every track directory; the list is the one a rescan gives.
# ---------------------------------------------------------------------------


def _track_state():
    names = pm.gtrack_ls()
    return names, sorted(_shared._track_names_set()), {n: pm.gtrack_dataset(n) for n in names}


def _assert_state_is_a_rescans():
    before = _track_state()
    assert before[1] == sorted(before[0])
    pm.gdb_reload()
    assert _track_state() == before


@pytest.fixture
def counted_rescans(monkeypatch):
    calls = []
    rescan = _pymisha.pm_dbreload

    def counting(*args, **kwargs):
        calls.append(1)
        return rescan(*args, **kwargs)

    monkeypatch.setattr(_pymisha, "pm_dbreload", counting)
    return calls


def _sparse(track, value=1.0, chrom="chr1"):
    pm.gtrack_create_sparse(track, "t", pm.gintervals(chrom, 0, 10), [value])


def test_creating_a_track_lists_its_name_without_rescanning_the_database(tmp_path, counted_rescans):
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gdir_create("a1")
    for name in ("b1", "c_1", "a1.x", "zz", "B2"):
        _sparse(name)
    counted_rescans.clear()
    _sparse("m_new")
    assert counted_rescans == []
    assert set(pm.gtrack_ls()) == {"b1", "c_1", "a1.x", "zz", "B2", "m_new"}
    # the expression parser sees it at once
    assert pm.gextract("m_new * 2", pm.gintervals("chr1", 0, 10))["m_new * 2"].tolist() == [2.0]
    pm.gtrack_rm("m_new", force=True)
    assert counted_rescans == []
    assert "m_new" not in pm.gtrack_ls()
    _assert_state_is_a_rescans()


def test_the_track_list_after_creations_moves_copies_and_removals_is_a_rescans(tmp_path):
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gdir_create("ns")
    rng = np.random.default_rng(60427)
    first = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    chars = first + list("0123456789_")
    names = set()
    while len(names) < 30:
        names.add(str(rng.choice(first)) + "".join(rng.choice(chars, int(rng.integers(0, 6)))))
    names = sorted(names)
    for i, name in enumerate(names):
        _sparse(name if i % 5 else f"ns.{name}", float(i))
    names = [name if i % 5 else f"ns.{name}" for i, name in enumerate(names)]
    pm.gtrack_create_dense("dense1", "t", pm.gintervals("chr1", 0, 100), [1.0], 10)
    pm.gtrack_array_create("arr1", "t", pm.gintervals("chr1", 0, 10), np.array([[1.0, 2.0]]), ["c1", "c2"])
    pm.gtrack_2d_create("rects1", "t", pm.gintervals_2d("chr1", 0, 10, "chr1", 20, 30), [1.0])
    pm.gtrack_mv(names[1], "moved1")
    pm.gtrack_mv(names[2], "ns.moved2")
    pm.gtrack_copy(names[3], "copied1")
    for name in names[6:12]:
        pm.gtrack_rm(name, force=True)
    listed = pm.gtrack_ls()
    assert {"dense1", "arr1", "rects1", "moved1", "ns.moved2", "copied1", names[3]} <= set(listed)
    assert not ({names[1], names[2]} | set(names[6:12])) & set(listed)
    _assert_state_is_a_rescans()


def test_a_track_of_the_working_database_shadows_a_loaded_datasets_until_removed(tmp_path):
    ds = create_test_db(tmp_path / "ds")
    pm.gdb_init(ds)
    _sparse("shared", 2.0)
    _sparse("only_ds", 3.0)
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gdataset_load(ds)
    iv = pm.gintervals("chr1", 0, 10)

    def value(track):
        return pm.gextract(track, iv)[track].tolist()

    ds_root = pm.gtrack_dataset("shared")
    assert ds_root is not None and pm.gtrack_dataset("only_ds") == ds_root
    assert value("shared") == [2.0]
    # created in the working database: it takes precedence over the dataset's
    _sparse("shared", 1.0)
    assert pm.gtrack_dataset("shared") != ds_root
    assert value("shared") == [1.0]
    _assert_state_is_a_rescans()
    # removed from it: the dataset's is listed and read again
    pm.gtrack_rm("shared", force=True)
    assert "shared" in pm.gtrack_ls()
    assert pm.gtrack_dataset("shared") == ds_root
    assert value("shared") == [2.0]
    _assert_state_is_a_rescans()


def test_in_a_subdirectory_a_created_track_is_listed_relative_to_it(tmp_path):
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gdir_create("sub")
    pm.gdir_cd("sub")
    try:
        _sparse("inner")
        assert pm.gtrack_ls() == ["inner"]
        _assert_state_is_a_rescans()
        pm.gtrack_rm("inner", force=True)
        assert not pm.gtrack_ls()
        _sparse("inner2")
    finally:
        pm.gdir_cd("..")
    assert pm.gtrack_ls() == ["sub.inner2"]
    _assert_state_is_a_rescans()


def test_a_track_another_session_writes_is_listed_at_the_next_reload(tmp_path):
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    _sparse("mine")
    # what another session's track creation leaves
    shutil.copytree(Path(db) / "tracks" / "mine.track", Path(db) / "tracks" / "theirs.track")
    _sparse("mine2")
    assert pm.gtrack_ls() == ["mine", "mine2"]
    pm.gdb_reload()
    assert pm.gtrack_ls() == ["mine", "mine2", "theirs"]


# ---------------------------------------------------------------------------
# Opt-in timings (test-perf-regression.R; run with -m benchmark). R's sliding-window and
# dense bulk-read cases belong to the virtual-track code; its create_dense case times a
# path (on-grid chromosomes written without aggregation) that pymisha has no need for.
# ---------------------------------------------------------------------------


def _time_op(f, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


@pytest.mark.benchmark
def test_gquantiles_finds_a_high_percentile_as_fast_as_a_low_one(tmp_path):
    # Exact (unsampled) multitask path, 4 chromosomes, 20 bp bins: 10M values.
    sizes = [(f"chr{i}", 50_000_000) for i in range(1, 5)]
    db = create_test_db(tmp_path / "db", sizes)
    pm.gdb_init(db)
    saved = dict(pm.CONFIG)
    try:
        pm.CONFIG.update(progress=False, max_data_size=int(1e10), multitasking=True, min_scope4process=1,
                         min_intervs4process=0)
        rng = np.random.default_rng(60427)
        for chrom, size in sizes:
            starts = np.arange(0, size, 20)
            pm.gtrack_create_dense(f"q_{chrom}", "x", pd.DataFrame({"chrom": chrom, "start": starts,
                                                                    "end": starts + 20}),
                                   rng.standard_normal(len(starts)), 20)
        expr = " + ".join(f"np.nan_to_num(q_{c})" for c, _ in sizes)
        scope = pm.gintervals([c for c, _ in sizes])
        low = _time_op(lambda: pm.gquantiles(expr, [0.001], scope, iterator=20))
        high = _time_op(lambda: pm.gquantiles(expr, [0.999], scope, iterator=20))
        assert high / low < 1.3
    finally:
        pm.CONFIG.clear()
        pm.CONFIG.update(saved)


@pytest.mark.benchmark
def test_a_small_gextract_costs_about_the_same_with_50000_tracks_in_the_database(tmp_path):
    db = create_test_db(tmp_path / "db", [("chr1", 100000)])
    pm.gdb_init(db)
    pm.gtrack_create_dense("real", "x", pm.gintervals("chr1", 0, 100000), [1.0], 100, 0)
    iv = pm.gintervals("chr1", 0, 1000)

    def calls():
        for _ in range(50):
            pm.gextract("real", iv, progress=False)

    few = _time_op(calls)
    for d in range(50):
        for t in range(1000):
            (Path(db) / "tracks" / f"d{d:02d}" / f"t{t:04d}.track").mkdir(parents=True)
    pm.gdb_reload()
    assert len(pm.gtrack_ls()) >= 50000
    many = _time_op(calls)
    assert many / few < 1.5
