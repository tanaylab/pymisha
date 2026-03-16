"""
Focused vtrack performance benchmarks for measuring optimization impact.

Run: python dev/benchmarks/bench_vtrack_perf.py [--json]
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pymisha as pm

TESTDB = str(Path(__file__).resolve().parent.parent.parent / "tests" / "testdb" / "trackdb" / "test")
N_WARMUP = 2
N_REPS = 5


def bench(func, label, n_warmup=N_WARMUP, n_reps=N_REPS):
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


def make_query_intervals(n, chrom="1", max_pos=500000, width=50):
    starts = np.sort(np.random.randint(0, max_pos - width, size=n))
    ends = starts + width
    return pd.DataFrame({"chrom": [chrom] * n, "start": starts, "end": ends})


def make_source_intervals(n, chrom="1", max_pos=500000, width=100):
    starts = np.sort(np.random.randint(0, max_pos - width, size=n))
    ends = starts + width
    return pd.DataFrame({"chrom": [chrom] * n, "start": starts, "end": ends})


def make_nonoverlapping_intervals(n, chrom="1", max_pos=500000, width=50, gap=10):
    """Create non-overlapping intervals for value-based vtracks that disallow overlaps."""
    max_n = max_pos // (width + gap)
    if n > max_n:
        n = max_n
    starts = np.arange(0, n * (width + gap), width + gap, dtype=np.int64)
    ends = starts + width
    return pd.DataFrame({"chrom": [chrom] * len(starts), "start": starts, "end": ends})


def make_source_with_values(n, chrom="1", max_pos=500000, width=50, gap=10):
    df = make_nonoverlapping_intervals(n, chrom, max_pos, width, gap)
    df["value"] = np.random.randn(len(df))
    return df


def run_benchmarks():
    pm.gdb_init(TESTDB)
    pm.CONFIG["progress"] = False
    pm.CONFIG["multitasking"] = False

    np.random.seed(42)
    results = []

    queries_100k = make_query_intervals(100000)
    src_10k = make_source_intervals(10000)
    src_5k = make_source_intervals(5000)
    src_values_10k = make_source_with_values(10000)
    queries_10k = make_query_intervals(10000)

    # 1. Distance vtrack (interval-source, 100K queries, 10K sources)
    pm.gvtrack_create("vt_dist_bench", src_10k, func="distance")
    results.append(bench(
        lambda: pm.gextract("vt_dist_bench", intervals=queries_100k, iterator=-1),
        "distance_100k_q_10k_src"
    ))

    # 2. Coverage vtrack
    pm.gvtrack_create("vt_cov_bench", src_10k, func="coverage")
    results.append(bench(
        lambda: pm.gextract("vt_cov_bench", intervals=queries_100k, iterator=-1),
        "coverage_100k_q_10k_src"
    ))

    # 3. Value-based avg
    pm.gvtrack_create("vt_avg_bench", src_values_10k, func="avg")
    results.append(bench(
        lambda: pm.gextract("vt_avg_bench", intervals=queries_100k, iterator=-1),
        "value_avg_100k_q_10k_src"
    ))

    # 4. LSE on sparse track (10K queries)
    pm.gvtrack_create("vt_lse_bench", "sparse_track", func="lse")
    results.append(bench(
        lambda: pm.gextract("vt_lse_bench", intervals=queries_10k, iterator=-1),
        "lse_sparse_10k_q"
    ))

    # 5. neighbor.count
    pm.gvtrack_create("vt_neigh_bench", src_5k, func="neighbor.count", params=1000)
    results.append(bench(
        lambda: pm.gextract("vt_neigh_bench", intervals=queries_100k, iterator=-1),
        "neighbor_count_100k_q_5k_src"
    ))

    # 6. distance.edge
    pm.gvtrack_create("vt_dedge_bench", src_10k, func="distance.edge")
    results.append(bench(
        lambda: pm.gextract("vt_dedge_bench", intervals=queries_100k, iterator=-1),
        "distance_edge_100k_q_10k_src"
    ))

    # Cleanup
    for name in ["vt_dist_bench", "vt_cov_bench", "vt_avg_bench",
                  "vt_lse_bench", "vt_neigh_bench", "vt_dedge_bench"]:
        try:
            pm.gvtrack_rm(name)
        except Exception:
            pass

    return results


if __name__ == "__main__":
    results = run_benchmarks()
    use_json = "--json" in sys.argv
    if use_json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'Benchmark':<40s} {'Median (s)':>12s} {'Std (s)':>10s}")
        print("-" * 64)
        for r in results:
            print(f"{r['label']:<40s} {r['median_s']:>12.6f} {r['std_s']:>10.6f}")
