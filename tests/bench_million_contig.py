"""Bench + regression tests for million-contig perf items.

Benchmarks are marked ``@pytest.mark.benchmark`` and auto-skipped unless
``pytest -m benchmark`` is passed (see tests/conftest.py).
"""
from __future__ import annotations

import time
from pathlib import Path

import _pymisha
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _build_groot(path: Path, num_chroms: int, seq_len: int = 100) -> None:
    """Build a tiny groot with `num_chroms` contigs at `path`.

    Used by the cache tests; not a fixture because each test needs distinct
    groots and the singleton-DB nature means we cannot parameterize cleanly.
    """
    fasta = path.with_suffix(".fa")
    lines: list[str] = []
    seq = "ACGT" * (seq_len // 4)
    for i in range(num_chroms):
        lines.append(f">c{i}")
        lines.append(seq)
    fasta.write_text("\n".join(lines) + "\n")
    pm.gdb_create(str(path), str(fasta), db_format="indexed", verbose=False)


def _restore_test_db() -> None:
    """Reset the singleton DB back to the shared session fixture's groot."""
    if TEST_DB.exists():
        pm.gdb_init(str(TEST_DB))


@pytest.mark.benchmark
def test_intervals_all_is_cached(tmp_path):
    """After warm-up, repeated pm_intervals_all calls should be fast (<5ms each).

    Even with 100 contigs the cached path is orders of magnitude cheaper
    than the uncached rebuild (chrom-name strings + numpy alloc per call vs
    cache-hit + numpy alloc only). The threshold is generous to avoid CI flake.
    """
    try:
        groot = tmp_path / "g"
        _build_groot(groot, num_chroms=100)
        pm.gdb_init(str(groot))

        # Warm-up: builds the cache.
        _ = _pymisha.pm_intervals_all()

        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            _ = _pymisha.pm_intervals_all()
        elapsed_ms = (time.perf_counter() - start) * 1000 / n
        assert elapsed_ms < 5.0, (
            f"per-call cost {elapsed_ms:.3f}ms exceeds 5ms after caching"
        )
    finally:
        _restore_test_db()


@pytest.mark.benchmark
def test_intervals_all_cache_invalidates_on_reload(tmp_path):
    """gdb_init to a new groot must invalidate the cache."""
    try:
        groot_a = tmp_path / "a"
        _build_groot(groot_a, num_chroms=10)
        pm.gdb_init(str(groot_a))
        df_a = _pymisha.pm_intervals_all()
        from pymisha._shared import _pymisha2df

        df_a = _pymisha2df(df_a)
        assert len(df_a) == 10

        groot_b = tmp_path / "b"
        _build_groot(groot_b, num_chroms=20)
        pm.gdb_init(str(groot_b))
        df_b = _pymisha2df(_pymisha.pm_intervals_all())
        assert len(df_b) == 20
    finally:
        _restore_test_db()


def test_intervals_all_basic_shape(tmp_path):
    """Regression: pm_intervals_all returns chrom/start/end with right values."""
    try:
        from pymisha._shared import _pymisha2df

        fa = tmp_path / "g.fa"
        fa.write_text(">a\nACGT\n>b\nACGTACGT\n")
        groot = tmp_path / "g"
        pm.gdb_create(str(groot), str(fa), db_format="indexed", verbose=False)
        pm.gdb_init(str(groot))

        df = _pymisha2df(_pymisha.pm_intervals_all())
        assert list(df.columns) == ["chrom", "start", "end"]
        assert set(df["chrom"]) == {"a", "b"}
        assert int(df.loc[df["chrom"] == "a", "end"].iloc[0]) == 4
        assert int(df.loc[df["chrom"] == "b", "end"].iloc[0]) == 8
        assert int(df.loc[df["chrom"] == "a", "start"].iloc[0]) == 0
        assert int(df.loc[df["chrom"] == "b", "start"].iloc[0]) == 0
    finally:
        _restore_test_db()


def test_intervals_all_calls_return_independent_arrays(tmp_path):
    """Mutating the returned DataFrame must not corrupt subsequent calls.

    Guards the design choice of caching vectors (not the PyObject): two
    successive calls must yield independent numpy buffers.
    """
    try:
        from pymisha._shared import _pymisha2df

        fa = tmp_path / "g.fa"
        fa.write_text(">a\nACGT\n>b\nACGTACGT\n")
        groot = tmp_path / "g"
        pm.gdb_create(str(groot), str(fa), db_format="indexed", verbose=False)
        pm.gdb_init(str(groot))

        df1 = _pymisha2df(_pymisha.pm_intervals_all())
        df1.loc[:, "end"] = -1  # In-place poison

        df2 = _pymisha2df(_pymisha.pm_intervals_all())
        # df2 must reflect the true chrom sizes, not the poisoned values.
        assert int(df2.loc[df2["chrom"] == "a", "end"].iloc[0]) == 4
        assert int(df2.loc[df2["chrom"] == "b", "end"].iloc[0]) == 8
    finally:
        _restore_test_db()


@pytest.mark.benchmark
def test_find_existing_1d_filename_short_circuits_for_indexed(tmp_path):
    """Indexed-track reads should skip the per-chrom alias scan.

    Easier proxy than strace: time small gextract calls on a 100-contig
    indexed DB. After E.1.3 lands, per-call overhead drops because the
    alias loop (and its per-candidate access() syscalls) is short-circuited
    when the track has an index. Threshold is best-effort; the real safety
    net is the full regression suite.
    """
    try:
        import pandas as pd

        groot = tmp_path / "g"
        _build_groot(groot, num_chroms=100, seq_len=64)
        pm.gdb_init(str(groot))

        # Build a tiny dense track from a single interval on c0.
        intervs = pd.DataFrame({"chrom": ["c0"], "start": [0], "end": [64]})
        pm.gtrack_create_dense("t", "perf test track", intervs, [5.0], binsize=4)

        # Warm-up: prime any caches.
        c0 = pm.gintervals("c0", 0, 4)
        _ = pm.gextract("t", c0)

        n = 100
        start = time.perf_counter()
        for _ in range(n):
            _ = pm.gextract("t", c0)
        elapsed_ms = (time.perf_counter() - start) * 1000 / n
        assert elapsed_ms < 50, (
            f"per-call cost {elapsed_ms:.3f}ms exceeds 50ms after short-circuit"
        )
    finally:
        _restore_test_db()


@pytest.mark.benchmark
def test_gintervals_random_cpp_beats_python_on_large_genome(tmp_path):
    """C++ path should outperform Python on a many-contig genome.

    Builds a 5000-contig groot, then times the C++ and Python branches of
    ``gintervals_random`` directly. Asserts that C++ is at least 5x faster
    than Python at the same sample count. The asserted ratio is loose to
    avoid CI flake; in practice the speedup is much larger.
    """
    try:
        import numpy as np

        from pymisha.intervals import (
            _gintervals_random_cpp,
            _gintervals_random_python,
        )

        groot = tmp_path / "g"
        _build_groot(groot, num_chroms=5000, seq_len=400)
        pm.gdb_init(str(groot))

        genome = pm.gintervals_all()
        size = 50
        n = 1000

        # Warm-up.
        _ = _gintervals_random_cpp(size, n, 0.0, genome, None, 60427)
        np.random.seed(60427)
        _ = _gintervals_random_python(size, n, 0.0, genome, None)

        cpp_times = []
        for _ in range(5):
            t = time.perf_counter()
            _gintervals_random_cpp(size, n, 0.0, genome, None, 60427)
            cpp_times.append(time.perf_counter() - t)
        cpp_t = min(cpp_times)

        py_times = []
        for _ in range(5):
            np.random.seed(60427)
            t = time.perf_counter()
            _gintervals_random_python(size, n, 0.0, genome, None)
            py_times.append(time.perf_counter() - t)
        py_t = min(py_times)

        ratio = py_t / cpp_t if cpp_t > 0 else float("inf")
        print(f"\ngintervals_random (5000 contigs, n={n}): "
              f"cpp={cpp_t*1000:.2f}ms, py={py_t*1000:.2f}ms, speedup={ratio:.1f}x")
        assert ratio >= 5.0, (
            f"C++ path only {ratio:.1f}x faster than Python "
            f"(cpp={cpp_t*1000:.2f}ms, py={py_t*1000:.2f}ms); expected >= 5x"
        )
    finally:
        _restore_test_db()


@pytest.mark.benchmark
def test_init_read_no_redundant_stat(tmp_path):
    """E.1.4: stat(track.idx) call inside init_read is removed.

    Proxy for syscall reduction: time per-chrom transitions on a 100-contig
    indexed dense track. After E.1.4, every gextract that walks all chroms
    skips one stat(track.idx) per chrom, so the tight loop is cheaper.
    Threshold is best-effort; the real safety net is the full regression
    suite (zero regressions).
    """
    try:
        groot = tmp_path / "g"
        _build_groot(groot, num_chroms=100, seq_len=64)
        pm.gdb_init(str(groot))

        # One interval per chrom; gextract walks all of them in a single call.
        chrom_sizes = pm.gintervals_all()

        # Build a dense track whose binsize divides each chrom (seq_len=64,
        # binsize=4 => 16 bins per chrom). Reuse chrom_sizes as the intervals.
        values = [1.0] * len(chrom_sizes)
        pm.gtrack_create_dense("t", "perf test track", chrom_sizes, values, binsize=4)

        # Warm-up: prime any caches (including get_track_index).
        _ = pm.gextract("t", chrom_sizes)

        n = 10
        start = time.perf_counter()
        for _ in range(n):
            _ = pm.gextract("t", chrom_sizes)
        elapsed_ms = (time.perf_counter() - start) * 1000 / n
        assert elapsed_ms < 200, (
            f"per-call cost {elapsed_ms:.3f}ms exceeds 200ms after removing redundant stat()"
        )
    finally:
        _restore_test_db()
