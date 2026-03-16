"""Benchmark for gbins_summary and gbins_quantiles optimizations.

Run:
    python dev/benchmarks/bench_gbins_perf.py [--json]
"""

import argparse
import json
import sys
import time

import numpy as np

import pymisha as pm


def bench(func, n=20, warmup=3):
    """Time *func* n+warmup times and return (mean, min) of the last n."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pm.gdb_init(pm.gdb_examples_path())

    breaks4 = [0, 0.2, 0.3, 0.9, 1.2]
    breaks100 = np.linspace(0, 1.0, 101).tolist()
    pcts = [0.2, 0.5, 0.6]
    intervals = pm.gintervals_all()

    results = {}

    # gbins_summary — 4 bins, different expr
    m, mn = bench(lambda: pm.gbins_summary(
        "dense_track", breaks4, expr="sparse_track", iterator=100,
    ))
    results["gbins_summary_4bins_diff_expr"] = {"mean": m, "min": mn}

    # gbins_summary — 4 bins, same expr
    m, mn = bench(lambda: pm.gbins_summary(
        "dense_track", breaks4, iterator=100,
    ))
    results["gbins_summary_4bins_same_expr"] = {"mean": m, "min": mn}

    # gbins_summary — 2D
    m, mn = bench(lambda: pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0],
        "dense_track", [0, 0.3, 0.7, 1.0],
        expr="sparse_track", iterator=100,
    ))
    results["gbins_summary_2d"] = {"mean": m, "min": mn}

    # gbins_quantiles — 4 bins
    m, mn = bench(lambda: pm.gbins_quantiles(
        "dense_track", breaks4, expr="sparse_track",
        percentiles=pcts, iterator=100,
    ))
    results["gbins_quantiles_4bins"] = {"mean": m, "min": mn}

    # gbins_quantiles — 100 bins
    m, mn = bench(lambda: pm.gbins_quantiles(
        "dense_track", breaks100, expr="sparse_track",
        percentiles=pcts, iterator=100,
    ))
    results["gbins_quantiles_100bins"] = {"mean": m, "min": mn}

    # gbins_summary with vtrack
    pm.gvtrack_create("_bench_vt", "dense_track", func="avg")
    m, mn = bench(lambda: pm.gbins_summary(
        "dense_track", [0, 0.5, 1.0], expr="_bench_vt", iterator=100,
    ))
    results["gbins_summary_vtrack"] = {"mean": m, "min": mn}
    pm.gvtrack_clear()

    # Reference: bare gextract (lower bound)
    m, mn = bench(lambda: pm.gextract(
        "dense_track", intervals, iterator=100, progress=False,
    ))
    results["gextract_reference"] = {"mean": m, "min": mn}

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'Benchmark':<45} {'Mean (ms)':>10} {'Min (ms)':>10}")
        print("-" * 67)
        for name, vals in results.items():
            print(f"{name:<45} {vals['mean']*1000:>10.2f} {vals['min']*1000:>10.2f}")


if __name__ == "__main__":
    main()
