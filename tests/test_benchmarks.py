"""
Benchmarks comparing pymisha vs R misha performance.

These tests measure execution time for key operations and compare
pymisha against R misha to ensure acceptable performance parity.

Run with: pytest tests/test_benchmarks.py -v -s

Note: Tests are marked with pytest.mark.benchmark and skipped by default.
Run with: pytest tests/test_benchmarks.py -v -s -m benchmark
"""
import gc
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

# Path to test database (use absolute path for consistency with conftest)
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

TESTDB = str(Path(__file__).resolve().parent / "testdb" / "trackdb" / "test")

# Large database for more reliable benchmarks - R misha test databases
# These have proper chromosome sizes and enough data for reliable timing
LARGEDB_CANDIDATES = [
    "/net/mraid20/ifs/wisdom/tanay_lab/tgdata/db/tgdb/misha_test_db_indexed/",
    "/net/mraid20/export/tgdata/db/tgdb/misha_test_db/",
]
LARGEDB_TRACK = "test.fixedbin"  # A dense track that exists in misha_test_db

# Number of repetitions for timing (more for stable results)
N_REPS = 10
N_WARMUP = 3

# Interval operation sizes (larger = more reliable timing)
INTERVAL_COUNT_SMALL = 1000      # For neighbor operations (O(n*k))
INTERVAL_COUNT_MEDIUM = 10000    # For set operations (O(n))
INTERVAL_COUNT_LARGE = 100000    # For simple operations


def _get_large_db():
    """Return large database path if available, else None."""
    for db_path in LARGEDB_CANDIDATES:
        if os.path.exists(db_path) and os.path.isdir(db_path):
            return db_path
    return None


def _has_large_db():
    """Check if large database is available."""
    return _get_large_db() is not None


@dataclass
class BenchmarkResult:
    """Holds benchmark timing results."""
    name: str
    python_time: float
    r_time: float
    python_std: float = 0.0
    r_std: float = 0.0
    speedup: float = field(init=False)

    def __post_init__(self):
        if self.r_time > 0:
            self.speedup = self.r_time / self.python_time
        else:
            self.speedup = float('inf')

    def __str__(self):
        if self.speedup >= 1:
            return (f"{self.name}: Python={self.python_time:.4f}s (±{self.python_std:.4f}), "
                    f"R={self.r_time:.4f}s (±{self.r_std:.4f}), speedup={self.speedup:.2f}x")
        return (f"{self.name}: Python={self.python_time:.4f}s (±{self.python_std:.4f}), "
                f"R={self.r_time:.4f}s (±{self.r_std:.4f}), slowdown={1/self.speedup:.2f}x")


def time_r_code(code: str, n_reps: int = N_REPS, n_warmup: int = N_WARMUP, testdb: str = None) -> tuple[float, float]:
    """Run R code and return (median, std) execution time in seconds."""
    if testdb is None:
        testdb = TESTDB
    # Wrap the code to measure time
    timed_code = f'''
library(misha)
gdb.init("{testdb}")
options(gmax.processes = 1)

# Warmup runs
for (i in 1:{n_warmup}) {{
    {code}
}}

# Timed runs
times <- numeric({n_reps})
for (i in 1:{n_reps}) {{
    gc()
    start <- Sys.time()
    {code}
    end <- Sys.time()
    times[i] <- as.numeric(end - start)
}}
cat("MEDIAN_TIME:", median(times), "\\n")
cat("STD_TIME:", sd(times), "\\n")
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(timed_code)
        script_path = f.name

    try:
        result = subprocess.run(
            ['R', '--quiet', '--no-save', '-f', script_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse timing from output
        median_time = None
        std_time = 0.0
        for line in result.stdout.split('\n'):
            if line.startswith('MEDIAN_TIME:'):
                median_time = float(line.split(':')[1].strip())
            elif line.startswith('STD_TIME:'):
                std_time = float(line.split(':')[1].strip())

        if median_time is None:
            raise ValueError(f"Failed to parse R timing output: {result.stdout}\n{result.stderr}")

        return median_time, std_time
    finally:
        os.unlink(script_path)


def time_python_code(func, n_reps: int = N_REPS, n_warmup: int = N_WARMUP) -> tuple[float, float]:
    """Run Python function and return (median, std) execution time in seconds."""
    # Warmup runs
    for _ in range(n_warmup):
        func()

    times = []
    for _ in range(n_reps):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return np.median(times), np.std(times)


# Store benchmark results for summary
_benchmark_results: list[BenchmarkResult] = []


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize database before each test."""
    pm.gdb_init(TESTDB)
    # Disable progress bars during benchmarks to avoid segfaults
    pm.CONFIG['progress'] = False
    pm.CONFIG['multitasking'] = False  # Test single-threaded first
    yield


# Mark all benchmark tests
pytestmark = pytest.mark.benchmark


class TestBenchmarkGintervals:
    """Benchmarks for interval operations.

    Implementation status:
    - gintervals_union/intersect/diff/covered_bp: C++ (fast O(n) merge algorithms)
    - gintervals_neighbors: C++ via pm_find_neighbors
    - gintervals_all: Python (builds DataFrame from chrom_sizes)
    - gintervals_canonic: Python (has mapping feature)
    """

    def test_benchmark_gintervals_all(self):
        """Benchmark gintervals_all (informational - Python implementation)."""
        # Python
        py_time, py_std = time_python_code(lambda: pm.gintervals_all())

        # R
        r_time, r_std = time_r_code("result <- gintervals.all()")

        result = BenchmarkResult("gintervals_all [Python impl]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")
        # No assertion - Python builds DataFrame from chrom_sizes (R uses C++)

    def test_benchmark_gintervals_union(self):
        """Benchmark gintervals_union (C++ O(n) merge algorithm)."""
        # Create intervals - use INTERVAL_COUNT_MEDIUM for reliable timing
        # Space them so they fit within chromosome 1 (size 500000)
        n = INTERVAL_COUNT_MEDIUM
        step = 40  # 40bp spacing to fit n intervals in ~400000bp
        starts1 = list(range(0, n * step, step))
        ends1 = [s + 20 for s in starts1]
        starts2 = [s + 10 for s in starts1]
        ends2 = [s + 30 for s in starts1]

        intervs1 = pm.gintervals("1", starts1, ends1)
        intervs2 = pm.gintervals("1", starts2, ends2)

        # Python
        py_time, py_std = time_python_code(lambda: pm.gintervals_union(intervs1, intervs2))

        # R
        r_code = f'''
starts1 <- seq(0, {(n-1) * step}, {step})
ends1 <- starts1 + 20
starts2 <- starts1 + 10
ends2 <- starts1 + 30
intervs1 <- gintervals("1", starts1, ends1)
intervs2 <- gintervals("1", starts2, ends2)
result <- gintervals.union(intervs1, intervs2)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_union ({n}) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 0.5, f"gintervals_union should be reasonably fast: {result}"

    def test_benchmark_gintervals_intersect(self):
        """Benchmark gintervals_intersect (C++ O(n) merge algorithm)."""
        n = INTERVAL_COUNT_MEDIUM
        step = 40
        starts1 = list(range(0, n * step, step))
        ends1 = [s + 25 for s in starts1]  # 25bp intervals
        starts2 = [s + 10 for s in starts1]  # offset by 10
        ends2 = [s + 35 for s in starts1]  # 25bp intervals, overlapping

        intervs1 = pm.gintervals("1", starts1, ends1)
        intervs2 = pm.gintervals("1", starts2, ends2)

        py_time, py_std = time_python_code(lambda: pm.gintervals_intersect(intervs1, intervs2))

        r_code = f'''
starts1 <- seq(0, {(n-1) * step}, {step})
ends1 <- starts1 + 25
starts2 <- starts1 + 10
ends2 <- starts1 + 35
intervs1 <- gintervals("1", starts1, ends1)
intervs2 <- gintervals("1", starts2, ends2)
result <- gintervals.intersect(intervs1, intervs2)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_intersect ({n}) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 0.5, f"gintervals_intersect should be reasonably fast: {result}"

    def test_benchmark_gintervals_neighbors(self):
        """Benchmark gintervals_neighbors (C++ via pm_find_neighbors)."""
        # Use smaller count for neighbors since it's O(n*k)
        n = INTERVAL_COUNT_SMALL
        step = 400  # larger spacing to fit in chromosome
        starts1 = list(range(0, n * step, step))
        ends1 = [s + 50 for s in starts1]
        starts2 = [s + 200 for s in starts1]
        ends2 = [s + 250 for s in starts1]

        intervs1 = pm.gintervals("1", starts1, ends1)
        intervs2 = pm.gintervals("1", starts2, ends2)

        py_time, py_std = time_python_code(
            lambda: pm.gintervals_neighbors(intervs1, intervs2, maxneighbors=3)
        )

        r_code = f'''
starts1 <- seq(0, {(n-1) * step}, {step})
ends1 <- starts1 + 50
starts2 <- starts1 + 200
ends2 <- starts1 + 250
intervs1 <- gintervals("1", starts1, ends1)
intervs2 <- gintervals("1", starts2, ends2)
result <- gintervals.neighbors(intervs1, intervs2, maxneighbors=3)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_neighbors ({n}, k=3) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 0.5, f"gintervals_neighbors should be reasonably fast: {result}"

    def test_benchmark_gintervals_diff(self):
        """Benchmark gintervals_diff (C++ O(n) merge algorithm)."""
        n = INTERVAL_COUNT_MEDIUM
        step = 40
        starts1 = list(range(0, n * step, step))
        ends1 = [s + 25 for s in starts1]
        starts2 = [s + 10 for s in starts1]
        ends2 = [s + 20 for s in starts1]  # smaller intervals to subtract

        intervs1 = pm.gintervals("1", starts1, ends1)
        intervs2 = pm.gintervals("1", starts2, ends2)

        py_time, py_std = time_python_code(lambda: pm.gintervals_diff(intervs1, intervs2))

        r_code = f'''
starts1 <- seq(0, {(n-1) * step}, {step})
ends1 <- starts1 + 25
starts2 <- starts1 + 10
ends2 <- starts1 + 20
intervs1 <- gintervals("1", starts1, ends1)
intervs2 <- gintervals("1", starts2, ends2)
result <- gintervals.diff(intervs1, intervs2)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_diff ({n}) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 0.5, f"gintervals_diff should be reasonably fast: {result}"

    def test_benchmark_gintervals_covered_bp(self):
        """Benchmark gintervals_covered_bp (C++ implementation)."""
        n = INTERVAL_COUNT_MEDIUM
        step = 40
        starts = list(range(0, n * step, step))
        ends = [s + 25 for s in starts]  # overlapping intervals

        intervs = pm.gintervals("1", starts, ends)

        py_time, py_std = time_python_code(lambda: pm.gintervals_covered_bp(intervs))

        # R doesn't have gintervals.covered.bp, so we compare to equivalent calculation
        r_code = f'''
starts <- seq(0, {(n-1) * step}, {step})
ends <- starts + 25
intervs <- gintervals("1", starts, ends)
canonical <- gintervals.canonic(intervs)
result <- sum(canonical$end - canonical$start)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_covered_bp ({n}) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        # Should be fast - no assertion needed since R doesn't have direct equivalent


class TestBenchmarkGextract:
    """Benchmarks for gextract operations.

    These use C++ streaming core and should be faster than R.
    """

    def test_benchmark_gextract_dense_track(self):
        """Benchmark gextract on dense track."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(lambda: pm.gextract("dense_track", intervals))

        r_code = '''
intervals <- gintervals.all()
result <- gextract("dense_track", intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gextract dense_track [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        # Should be at least as fast as R (ideally much faster)
        assert result.speedup >= 1.0, f"gextract should be at least as fast as R: {result}"

    def test_benchmark_gextract_expression(self):
        """Benchmark gextract with expression."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gextract("dense_track * 2 + 1", intervals)
        )

        r_code = '''
intervals <- gintervals.all()
result <- gextract("dense_track * 2 + 1", intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gextract expression [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gextract should be at least as fast as R: {result}"

    def test_benchmark_gextract_with_iterator(self):
        """Benchmark gextract with iterator."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gextract("dense_track", intervals, iterator=100)
        )

        r_code = '''
intervals <- gintervals.all()
result <- gextract("dense_track", intervals, iterator=100)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gextract iterator=100 [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gextract should be at least as fast as R: {result}"

    def test_benchmark_gextract_sparse_track(self):
        """Benchmark gextract on sparse track."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(lambda: pm.gextract("sparse_track", intervals))

        r_code = '''
intervals <- gintervals.all()
result <- gextract("sparse_track", intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gextract sparse_track [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gextract should be at least as fast as R: {result}"


class TestBenchmarkGscreen:
    """Benchmarks for gscreen operations (C++ streaming core)."""

    def test_benchmark_gscreen_simple(self):
        """Benchmark gscreen with simple filter."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gscreen("dense_track > 0.5", intervals)
        )

        r_code = '''
intervals <- gintervals.all()
result <- gscreen("dense_track > 0.5", intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gscreen [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gscreen should be at least as fast as R: {result}"


class TestBenchmarkGpartition:
    """Benchmarks for gpartition (C++ streaming with BinFinder)."""

    def test_benchmark_gpartition(self):
        """Benchmark gpartition (C++ streaming implementation)."""
        intervals = pm.gintervals_all()
        breaks = [0.0, 0.25, 0.5, 0.75, 1.0]

        py_time, py_std = time_python_code(
            lambda: pm.gpartition("dense_track", breaks, intervals)
        )

        r_code = '''
intervals <- gintervals.all()
breaks <- c(0.0, 0.25, 0.5, 0.75, 1.0)
result <- gpartition("dense_track", breaks, intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gpartition [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gpartition should be at least as fast as R: {result}"


class TestBenchmarkGsummary:
    """Benchmarks for gsummary/gquantiles operations (C++ streaming core)."""

    def test_benchmark_gsummary(self):
        """Benchmark gsummary."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(lambda: pm.gsummary("dense_track", intervals))

        r_code = '''
intervals <- gintervals.all()
result <- gsummary("dense_track", intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gsummary [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gsummary should be at least as fast as R: {result}"

    def test_benchmark_gquantiles(self):
        """Benchmark gquantiles."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gquantiles("dense_track", [0.25, 0.5, 0.75], intervals)
        )

        r_code = '''
intervals <- gintervals.all()
result <- gquantiles("dense_track", c(0.25, 0.5, 0.75), intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("gquantiles [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gquantiles should be at least as fast as R: {result}"

    def test_benchmark_gintervals_summary(self):
        """Benchmark gintervals_summary (C++ streaming per-interval stats)."""
        # Use INTERVAL_COUNT_SMALL since this computes stats per interval
        n = INTERVAL_COUNT_SMALL
        step = 400  # fit within chromosome
        starts = list(range(0, n * step, step))
        ends = [s + 300 for s in starts]

        intervals = pm.gintervals("1", starts, ends)

        py_time, py_std = time_python_code(
            lambda: pm.gintervals_summary("dense_track", intervals)
        )

        r_code = f'''
starts <- seq(0, {(n-1) * step}, {step})
ends <- starts + 300
intervals <- gintervals("1", starts, ends)
result <- gintervals.summary("dense_track", intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_summary ({n}) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        # Allow some variance - should be within 2x of R
        assert result.speedup >= 0.5, f"gintervals_summary should be reasonably fast: {result}"

    def test_benchmark_gintervals_quantiles(self):
        """Benchmark gintervals_quantiles (C++ streaming per-interval quantiles)."""
        # Use INTERVAL_COUNT_SMALL since this computes quantiles per interval
        n = INTERVAL_COUNT_SMALL
        step = 400
        starts = list(range(0, n * step, step))
        ends = [s + 300 for s in starts]

        intervals = pm.gintervals("1", starts, ends)

        py_time, py_std = time_python_code(
            lambda: pm.gintervals_quantiles("dense_track", [0.25, 0.5, 0.75], intervals)
        )

        r_code = f'''
starts <- seq(0, {(n-1) * step}, {step})
ends <- starts + 300
intervals <- gintervals("1", starts, ends)
result <- gintervals.quantiles("dense_track", c(0.25, 0.5, 0.75), intervals)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult(f"gintervals_quantiles ({n}) [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gintervals_quantiles should be at least as fast as R: {result}"


class TestBenchmarkVirtualTracks:
    """Benchmarks for virtual track operations (C++ core)."""

    def test_benchmark_vtrack_avg(self):
        """Benchmark virtual track with avg function."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        intervals = pm.gintervals("1", 0, 100000)

        py_time, py_std = time_python_code(
            lambda: pm.gextract("vt_avg", intervals, iterator=100)
        )

        r_code = '''
gvtrack.create("vt_avg", "dense_track", func="avg")
intervals <- gintervals("1", 0, 100000)
result <- gextract("vt_avg", intervals, iterator=100)
'''
        r_time, r_std = time_r_code(r_code)

        result = BenchmarkResult("vtrack avg [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"vtrack should be at least as fast as R: {result}"


class TestBenchmarkMultitasking:
    """Benchmarks comparing single vs multitasking performance."""

    def test_benchmark_gextract_multitask(self):
        """Benchmark gextract with multitasking."""
        intervals = pm.gintervals_all()

        # Single-threaded
        pm.CONFIG['multitasking'] = False
        py_single, py_single_std = time_python_code(
            lambda: pm.gextract("dense_track", intervals)
        )

        # Multi-threaded
        pm.CONFIG['multitasking'] = True
        py_multi, py_multi_std = time_python_code(
            lambda: pm.gextract("dense_track", intervals)
        )

        # Reset
        pm.CONFIG['multitasking'] = False

        print("\ngextract dense_track multitask comparison:")
        print(f"  Single: {py_single:.4f}s (±{py_single_std:.4f})")
        print(f"  Multi:  {py_multi:.4f}s (±{py_multi_std:.4f})")
        if py_multi > 0:
            print(f"  Speedup: {py_single/py_multi:.2f}x")

    def test_benchmark_gscreen_multitask(self):
        """Benchmark gscreen with multitasking."""
        intervals = pm.gintervals_all()

        # Single-threaded
        pm.CONFIG['multitasking'] = False
        py_single, py_single_std = time_python_code(
            lambda: pm.gscreen("dense_track > 0.5", intervals)
        )

        # Multi-threaded
        pm.CONFIG['multitasking'] = True
        py_multi, py_multi_std = time_python_code(
            lambda: pm.gscreen("dense_track > 0.5", intervals)
        )

        # Reset
        pm.CONFIG['multitasking'] = False

        print("\ngscreen multitask comparison:")
        print(f"  Single: {py_single:.4f}s (±{py_single_std:.4f})")
        print(f"  Multi:  {py_multi:.4f}s (±{py_multi_std:.4f})")
        if py_multi > 0:
            print(f"  Speedup: {py_single/py_multi:.2f}x")


@pytest.mark.skipif(not _has_large_db(), reason="Large database (misha_test_db) not available")
class TestBenchmarkLargeDatabase:
    """Benchmarks on large real database for reliable timing.

    These tests run on misha_test_db which has proper chromosome sizes.
    Results are much more reliable than the small test database.
    """

    @pytest.fixture(autouse=True)
    def setup_large_db(self):
        """Initialize large database."""
        self.largedb = _get_large_db()
        pm.gdb_init(self.largedb)
        pm.CONFIG['progress'] = False
        pm.CONFIG['multitasking'] = False
        yield
        # Restore small test db for other tests
        pm.gdb_init(TESTDB)

    def test_benchmark_large_gextract(self):
        """Benchmark gextract on large database."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gextract(LARGEDB_TRACK, intervals, iterator=10000)
        )

        r_code = f'''
intervals <- gintervals.all()
result <- gextract("{LARGEDB_TRACK}", intervals, iterator=10000)
'''
        r_time, r_std = time_r_code(r_code, testdb=self.largedb)

        result = BenchmarkResult("gextract large [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        assert result.speedup >= 1.0, f"gextract on large db should be fast: {result}"

    def test_benchmark_large_gsummary(self):
        """Benchmark gsummary on large database."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gsummary(LARGEDB_TRACK, intervals)
        )

        r_code = f'''
intervals <- gintervals.all()
result <- gsummary("{LARGEDB_TRACK}", intervals)
'''
        r_time, r_std = time_r_code(r_code, testdb=self.largedb)

        result = BenchmarkResult("gsummary large [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        # Allow some variance - should be within 2x of R
        assert result.speedup >= 0.5, f"gsummary on large db should be reasonably fast: {result}"

    def test_benchmark_large_gscreen(self):
        """Benchmark gscreen on large database."""
        intervals = pm.gintervals_all()

        py_time, py_std = time_python_code(
            lambda: pm.gscreen(f"{LARGEDB_TRACK} > 0", intervals)
        )

        r_code = f'''
intervals <- gintervals.all()
result <- gscreen("{LARGEDB_TRACK} > 0", intervals)
'''
        r_time, r_std = time_r_code(r_code, testdb=self.largedb)

        result = BenchmarkResult("gscreen large [C++ core]", py_time, r_time, py_std, r_std)
        _benchmark_results.append(result)
        print(f"\n{result}")

        # gscreen on large database may show variance depending on filter selectivity
        # Allow some slack - should be within 3x of R
        assert result.speedup >= 0.33, f"gscreen on large db should be reasonably fast: {result}"

    def test_benchmark_large_multitask(self):
        """Benchmark multitasking on large database (where it matters)."""
        intervals = pm.gintervals_all()

        # Single-threaded
        pm.CONFIG['multitasking'] = False
        py_single, py_single_std = time_python_code(
            lambda: pm.gextract(LARGEDB_TRACK, intervals, iterator=10000),
            n_reps=5, n_warmup=1
        )

        # Multi-threaded
        pm.CONFIG['multitasking'] = True
        py_multi, py_multi_std = time_python_code(
            lambda: pm.gextract(LARGEDB_TRACK, intervals, iterator=10000),
            n_reps=5, n_warmup=1
        )

        pm.CONFIG['multitasking'] = False

        print("\ngextract large multitask comparison:")
        print(f"  Single: {py_single:.4f}s (±{py_single_std:.4f})")
        print(f"  Multi:  {py_multi:.4f}s (±{py_multi_std:.4f})")
        if py_multi > 0:
            speedup = py_single / py_multi
            print(f"  Speedup: {speedup:.2f}x")

    def test_benchmark_large_gsummary_multitask(self):
        """Benchmark gsummary with single vs multiprocess execution."""
        intervals = pm.gintervals_all()

        # Single-process
        pm.CONFIG['multitasking'] = False
        py_single, py_single_std = time_python_code(
            lambda: pm.gsummary(LARGEDB_TRACK, intervals),
            n_reps=5, n_warmup=1
        )

        # Multiprocess
        pm.CONFIG['multitasking'] = True
        py_multi, py_multi_std = time_python_code(
            lambda: pm.gsummary(LARGEDB_TRACK, intervals),
            n_reps=5, n_warmup=1
        )

        pm.CONFIG['multitasking'] = False

        print("\ngsummary large multitask comparison:")
        print(f"  Single: {py_single:.4f}s (±{py_single_std:.4f})")
        print(f"  Multi:  {py_multi:.4f}s (±{py_multi_std:.4f})")
        if py_multi > 0:
            speedup = py_single / py_multi
            print(f"  Speedup: {speedup:.2f}x")


class TestBenchmarkSummary:
    """Print summary of all benchmarks."""

    def test_print_summary(self):
        """Print summary of all benchmark results."""
        if not _benchmark_results:
            pytest.skip("No benchmark results collected")

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY: pymisha vs R misha")
        print("=" * 80)

        # Separate C++ core vs Python impl results
        cpp_results = [r for r in _benchmark_results if "[C++ core]" in r.name]
        py_results = [r for r in _benchmark_results if "[Python impl]" in r.name]

        # C++ Core Operations
        if cpp_results:
            print("\n--- C++ Core Operations ---")
            print("(Track operations use C++ streaming; interval ops use C++ O(n) merge)")
            cpp_sorted = sorted(cpp_results, key=lambda x: x.speedup, reverse=True)
            for r in cpp_sorted:
                print(r)

            cpp_speedups = [r.speedup for r in cpp_sorted if r.speedup != float('inf')]
            if cpp_speedups:
                geomean_cpp = np.exp(np.mean(np.log(cpp_speedups)))
                print(f"\nC++ core geometric mean speedup: {geomean_cpp:.2f}x")

        # Python Implementation Operations
        if py_results:
            print("\n--- Python Implementation (pandas-based) ---")
            py_sorted = sorted(py_results, key=lambda x: x.speedup, reverse=True)
            for r in py_sorted:
                print(r)

            py_speedups = [r.speedup for r in py_sorted if r.speedup != float('inf')]
            if py_speedups:
                geomean_py = np.exp(np.mean(np.log(py_speedups)))
                print(f"\nPython impl geometric mean speedup: {geomean_py:.2f}x")
            print("(NOTE: These are utility functions; main workflows use C++)")

        # Overall summary
        all_results = sorted(_benchmark_results, key=lambda x: x.speedup)
        faster_count = sum(1 for r in all_results if r.speedup >= 1)
        slower_count = len(all_results) - faster_count

        print("\n" + "=" * 80)
        print(f"Total benchmarks: {len(all_results)}")
        print(f"Python faster/equal: {faster_count}")
        print(f"Python slower: {slower_count}")

        all_speedups = [r.speedup for r in all_results if r.speedup != float('inf')]
        if all_speedups:
            geomean = np.exp(np.mean(np.log(all_speedups)))
            print(f"Overall geometric mean speedup: {geomean:.2f}x")

        print("=" * 80)
