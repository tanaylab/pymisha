"""
PWM edit distance performance benchmark for pymisha.

Measures wall time for:
1. gseq_pwm_edits on synthetic sequences (varying lengths)
2. pwm.edit_distance virtual track scan on the test genome
3. pwm.edit_distance with indels (max_indels=1, 2)
4. gseq_pwm_edits on genomic intervals

Uses the small test DB (chroms 1: 500k, 2: 300k, X: 200k = 1Mb total).
Designed for before/after comparison of C++ optimization changes.

Run: python tests/bench_edit_distance.py [--json] [--reps N] [--compare FILE]

To do a before/after comparison:
  1. Build old version:  python tests/bench_edit_distance.py --json > baseline.json
  2. Build new version:  python tests/bench_edit_distance.py --json --compare baseline.json
"""
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pymisha as pm

TESTDB = str(Path(__file__).resolve().parent / "testdb" / "trackdb" / "test")

# CTCF-like PSSM: 12 columns (trimmed HOMER CTCF core motif)
# Realistic information content -- mix of high-IC and low-IC positions
CTCF_CORE_FREQS = np.array([
    [0.023, 0.378, 0.005, 0.593],  # high IC (T-dominant)
    [0.061, 0.005, 0.887, 0.047],  # high IC (G-dominant)
    [0.079, 0.905, 0.005, 0.010],  # very high IC (C-dominant)
    [0.002, 0.994, 0.001, 0.003],  # very high IC (C-dominant)
    [0.501, 0.475, 0.007, 0.016],  # moderate IC (A/C)
    [0.002, 0.527, 0.004, 0.467],  # moderate IC (C/T)
    [0.003, 0.995, 0.001, 0.001],  # very high IC (C-dominant)
    [0.030, 0.036, 0.004, 0.930],  # high IC (T-dominant)
    [0.382, 0.042, 0.446, 0.130],  # lower IC (A/G)
    [0.020, 0.273, 0.686, 0.021],  # high IC (G-dominant)
    [0.047, 0.039, 0.014, 0.900],  # high IC (T-dominant)
    [0.002, 0.001, 0.995, 0.002],  # very high IC (G-dominant)
], dtype=np.float64)

# Smaller 6-column PSSM for comparison
SHORT_PSSM = np.array([
    [0.9, 0.03, 0.04, 0.03],
    [0.02, 0.93, 0.02, 0.03],
    [0.03, 0.02, 0.92, 0.03],
    [0.03, 0.04, 0.03, 0.9],
    [0.9, 0.03, 0.03, 0.04],
    [0.02, 0.94, 0.02, 0.02],
], dtype=np.float64)


def bench(func, label, n_warmup=2, n_reps=5):
    """Run func with warmup + timed repetitions, return timing dict."""
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
    mn = float(np.min(times))
    return {"label": label, "median_s": round(med, 6), "std_s": round(std, 6),
            "min_s": round(mn, 6), "n_reps": n_reps}


def generate_random_seq(length, seed=42):
    """Generate a random DNA sequence of given length."""
    rng = np.random.RandomState(seed)
    bases = np.array(list("ACGT"))
    return "".join(bases[rng.randint(0, 4, size=length)])


def remove_all_vtracks():
    """Remove every virtual track in the current session."""
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def run_benchmarks(n_reps=5):
    pm.gdb_init(TESTDB)
    pm.CONFIG["progress"] = False
    pm.CONFIG["multitasking"] = False  # single-threaded for consistent measurement

    results = []

    # =========================================================================
    # 1. gseq_pwm_edits on synthetic sequences
    # =========================================================================
    print("--- gseq_pwm_edits: synthetic sequences ---")

    # 1a. 12-col PSSM on 10k sequence (many windows)
    seq_10k = generate_random_seq(10_000)
    results.append(bench(
        lambda: pm.gseq_pwm_edits(seq_10k, CTCF_CORE_FREQS,
                                   score_thresh=-8.0, max_edits=2,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_12col_10k_maxedit2",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # 1b. 12-col PSSM on 50k sequence
    seq_50k = generate_random_seq(50_000)
    results.append(bench(
        lambda: pm.gseq_pwm_edits(seq_50k, CTCF_CORE_FREQS,
                                   score_thresh=-8.0, max_edits=2,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_12col_50k_maxedit2",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # 1c. 12-col PSSM, no max_edits cap (exact mode)
    results.append(bench(
        lambda: pm.gseq_pwm_edits(seq_10k, CTCF_CORE_FREQS,
                                   score_thresh=-8.0,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_12col_10k_exact",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # 1d. 6-col short PSSM on 10k sequence
    results.append(bench(
        lambda: pm.gseq_pwm_edits(seq_10k, SHORT_PSSM,
                                   score_thresh=-3.0, max_edits=2,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_6col_10k_maxedit2",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # 1e. 12-col PSSM with indels on 10k sequence
    results.append(bench(
        lambda: pm.gseq_pwm_edits(seq_10k, CTCF_CORE_FREQS,
                                   score_thresh=-8.0, max_edits=2,
                                   max_indels=1,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_12col_10k_indel1",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # =========================================================================
    # 2. pwm.edit_distance virtual track scan on test genome
    # =========================================================================
    print("\n--- pwm.edit_distance vtrack: test genome scan ---")

    # Test genome: chrom 1 (500k), chrom 2 (300k), chrom X (200k) = 1Mb total
    all_intervals = pm.gintervals_all()
    chrom1 = all_intervals[all_intervals["chrom"] == "1"].reset_index(drop=True)
    total_bp_1 = int(chrom1["end"].iloc[0] - chrom1["start"].iloc[0])

    # 2a. 12-col PSSM, subs only, chrom 1
    remove_all_vtracks()
    pm.gvtrack_create("edist_bench", None,
                      func="pwm.edit_distance",
                      pssm=CTCF_CORE_FREQS,
                      score_thresh=-8.0,
                      max_edits=2,
                      bidirect=True,
                      prior=0.01)

    def scan_chrom1_subs():
        return pm.gscreen("edist_bench >= 0", intervals=chrom1, iterator=1)

    results.append(bench(scan_chrom1_subs,
                         f"vtrack_12col_chrom1_{total_bp_1 // 1000}k_subs",
                         n_reps=n_reps))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # Check hit count for sanity
    hits = scan_chrom1_subs()
    n_hits = len(hits) if hits is not None else 0
    print(f"    (hits: {n_hits})")

    remove_all_vtracks()

    # 2b. 12-col PSSM, with indels (D=1), chrom 1
    pm.gvtrack_create("edist_bench_d1", None,
                      func="pwm.edit_distance",
                      pssm=CTCF_CORE_FREQS,
                      score_thresh=-8.0,
                      max_edits=2,
                      max_indels=1,
                      bidirect=True,
                      prior=0.01)

    def scan_chrom1_indel1():
        return pm.gscreen("edist_bench_d1 >= 0", intervals=chrom1, iterator=1)

    results.append(bench(scan_chrom1_indel1,
                         f"vtrack_12col_chrom1_{total_bp_1 // 1000}k_indel1",
                         n_reps=n_reps))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")
    remove_all_vtracks()

    # 2c. 12-col PSSM, with indels (D=2), chrom 1
    pm.gvtrack_create("edist_bench_d2", None,
                      func="pwm.edit_distance",
                      pssm=CTCF_CORE_FREQS,
                      score_thresh=-8.0,
                      max_edits=2,
                      max_indels=2,
                      bidirect=True,
                      prior=0.01)

    def scan_chrom1_indel2():
        return pm.gscreen("edist_bench_d2 >= 0", intervals=chrom1, iterator=1)

    results.append(bench(scan_chrom1_indel2,
                         f"vtrack_12col_chrom1_{total_bp_1 // 1000}k_indel2",
                         n_reps=n_reps))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")
    remove_all_vtracks()

    # 2d. Full genome scan (all 3 chroms) with subs only
    pm.gvtrack_create("edist_all", None,
                      func="pwm.edit_distance",
                      pssm=CTCF_CORE_FREQS,
                      score_thresh=-8.0,
                      max_edits=2,
                      bidirect=True,
                      prior=0.01)

    total_bp_all = int((all_intervals["end"] - all_intervals["start"]).sum())

    def scan_all_subs():
        return pm.gscreen("edist_all >= 0", intervals=all_intervals, iterator=1)

    results.append(bench(scan_all_subs,
                         f"vtrack_12col_allchroms_{total_bp_all // 1000}k_subs",
                         n_reps=n_reps))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")
    remove_all_vtracks()

    # =========================================================================
    # 3. gseq_pwm_edits on genomic intervals
    # =========================================================================
    print("\n--- gseq_pwm_edits: genomic intervals ---")

    # 3a. 100 intervals of 1000bp each
    starts = list(range(0, 100_000, 1000))
    intervals_100 = pd.DataFrame({
        "chrom": ["1"] * len(starts),
        "start": starts,
        "end": [s + 1000 for s in starts],
    })
    results.append(bench(
        lambda: pm.gseq_pwm_edits(intervals_100, CTCF_CORE_FREQS,
                                   score_thresh=-8.0, max_edits=2,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_intervals_100x1k",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # 3b. 500 intervals of 200bp each
    starts_500 = list(range(0, 100_000, 200))
    intervals_500 = pd.DataFrame({
        "chrom": ["1"] * len(starts_500),
        "start": starts_500,
        "end": [s + 200 for s in starts_500],
    })
    results.append(bench(
        lambda: pm.gseq_pwm_edits(intervals_500, CTCF_CORE_FREQS,
                                   score_thresh=-8.0, max_edits=2,
                                   bidirect=True, prior=0.01),
        "gseq_pwm_edits_intervals_500x200",
        n_reps=n_reps,
    ))
    print(f"  {results[-1]['label']}: {results[-1]['median_s']:.4f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"{'Benchmark':<50} {'Median (s)':>10} {'Std':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<50} {r['median_s']:>10.4f} {r['std_s']:>10.4f}")
    print("=" * 70)

    return results


def load_baseline(path):
    """Load baseline results from a JSON file (or --json output with text)."""
    text = Path(path).read_text()
    # If the file has text before JSON, extract the JSON array
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)


def print_comparison(current, baseline):
    """Print side-by-side comparison with speedup factors."""
    base_map = {r["label"]: r for r in baseline}

    print("\n" + "=" * 82)
    print(f"{'Benchmark':<42} {'Old (s)':>9} {'New (s)':>9} {'Speedup':>9}")
    print("-" * 82)
    for r in current:
        label = r["label"]
        new_med = r["median_s"]
        if label in base_map:
            old_med = base_map[label]["median_s"]
            if new_med > 0 and old_med > 0:
                speedup = old_med / new_med
                marker = " ***" if speedup > 1.2 else (" (slow)" if speedup < 0.85 else "")
                print(f"{label:<42} {old_med:>9.4f} {new_med:>9.4f} {speedup:>8.2f}x{marker}")
            else:
                print(f"{label:<42} {old_med:>9.4f} {new_med:>9.4f}       N/A")
        else:
            print(f"{label:<42}       N/A {new_med:>9.4f}       N/A")
    print("=" * 82)


def main():
    import argparse
    p = argparse.ArgumentParser(description="PWM edit distance benchmark")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--reps", type=int, default=5, help="Number of timed repetitions")
    p.add_argument("--compare", type=str, default=None,
                   help="Path to baseline JSON for comparison")
    args = p.parse_args()

    results = run_benchmarks(n_reps=args.reps)

    if args.compare:
        baseline = load_baseline(args.compare)
        print_comparison(results, baseline)

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
