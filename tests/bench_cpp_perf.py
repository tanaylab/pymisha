"""
Focused C++ extraction path benchmarks for measuring optimization impact.

Measures wall time for:
1. Dense (fixed-bin) track extraction via gextract
2. Sparse track extraction via gextract
3. Expression evaluation with Python path (CHROM/START/END populated)
4. gscreen filtering on dense track
5. Multi-track extraction (2 dense + 1 sparse)

Uses the small test DB for fast iteration. Each benchmark runs
N_WARMUP warmup rounds then N_REPS timed rounds, reporting median +/- std.

Run: python tests/bench_cpp_perf.py [--json]
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pymisha as pm

TESTDB = str(Path(__file__).resolve().parent / "testdb" / "trackdb" / "test")
N_WARMUP = 3
N_REPS = 10


def bench(func, label, n_warmup=N_WARMUP, n_reps=N_REPS):
    """Run func with warmup + timed repetitions, return (label, median, std)."""
    for _ in range(n_warmup):
        func()
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    med = float(np.median(times))
    std = float(np.std(times))
    return {"label": label, "median_s": round(med, 6), "std_s": round(std, 6)}


def make_intervals(chrom, start, end, step):
    """Create a DataFrame of intervals with given step size."""
    starts = list(range(start, end, step))
    ends = [min(s + step, end) for s in starts]
    return pd.DataFrame({
        "chrom": [chrom] * len(starts),
        "start": starts,
        "end": ends,
    })


def run_benchmarks():
    pm.gdb_init(TESTDB)
    pm.CONFIG["progress"] = False
    pm.CONFIG["multitasking"] = False

    results = []

    # 100k intervals covering chrom 1 (500000 bp / 5 bp step = 100k intervals)
    intervals_100k = make_intervals("1", 0, 500000, 5)

    # 1. Dense extraction — 100k intervals
    results.append(bench(
        lambda: pm.gextract("dense_track", intervals=intervals_100k),
        "dense_extract_100k"
    ))

    # 2. Sparse extraction — 100k intervals
    results.append(bench(
        lambda: pm.gextract("sparse_track", intervals=intervals_100k),
        "sparse_extract_100k"
    ))

    # 3. Python expression path — forces CHROM/START/END population
    results.append(bench(
        lambda: pm.gextract("dense_track + 0", intervals=intervals_100k),
        "expr_dense_100k"
    ))

    # 4. gscreen with dense track
    results.append(bench(
        lambda: pm.gscreen("dense_track > 0.5", intervals=intervals_100k),
        "gscreen_dense_100k"
    ))

    # 5. Multi-track extraction (multiple dense track references)
    # This tests per-row overhead with multiple vars
    results.append(bench(
        lambda: pm.gextract(
            ["dense_track", "dense_track + 1", "dense_track * 2"],
            intervals=intervals_100k
        ),
        "multi_track_extract_100k"
    ))

    # 6. Dense extraction with wider intervals (tests multi-bin path)
    intervals_10k_wide = make_intervals("1", 0, 500000, 50)
    results.append(bench(
        lambda: pm.gextract("dense_track", intervals=intervals_10k_wide),
        "dense_extract_10k_wide"
    ))

    # 7. Expression with CHROM reference (ensures CHROM array is actually used)
    results.append(bench(
        lambda: pm.gextract("np.where(CHROM == '1', dense_track, 0)", intervals=intervals_100k),
        "expr_chrom_ref_100k"
    ))

    return results


if __name__ == "__main__":
    results = run_benchmarks()

    use_json = "--json" in sys.argv
    if use_json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'Benchmark':<30s} {'Median (s)':>12s} {'Std (s)':>10s}")
        print("-" * 54)
        for r in results:
            print(f"{r['label']:<30s} {r['median_s']:>12.6f} {r['std_s']:>10.6f}")
