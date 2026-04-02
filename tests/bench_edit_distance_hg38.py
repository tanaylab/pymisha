"""
Benchmark edit distance performance on hg38 to validate optimization speedups.

Compares PWM edit distance scoring on full-genome scans using CTCF motif,
matching the R misha benchmark setup (CTCF K=2, substitution-only).

Usage:
    python tests/bench_edit_distance_hg38.py [--json] [--save FILE] [--compare FILE]
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pymisha as pm

HG38_ROOT = str(Path.home() / "hg38")

# CTCF-like 12bp PSSM (log-likelihood scores)
CTCF_PSSM = np.array([
    [-0.5, -1.5,  1.8, -1.2],
    [ 1.7, -1.5, -0.8, -1.2],
    [-1.5, -1.5,  1.9, -0.5],
    [-1.5,  1.8, -1.5, -0.5],
    [-1.5,  1.5, -1.5,  0.2],
    [-0.2, -1.5,  1.6, -1.2],
    [-1.5,  1.7, -1.5, -0.5],
    [-1.5,  0.5,  1.3, -1.5],
    [-1.5,  1.8, -1.5, -0.5],
    [ 0.5,  0.5, -1.5,  0.5],
    [-0.5, -0.5,  1.5, -1.5],
    [-0.5,  0.5, -1.5,  0.8],
], dtype=np.float64)


def bench(func, label, n_warmup=1, n_reps=3):
    """Run func with warmup + timed repetitions."""
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
    print(f"  {label}: {med:.3f}s +/- {std:.3f}s")
    return {"label": label, "median_s": round(med, 4), "std_s": round(std, 4)}


def main():
    do_json = "--json" in sys.argv
    save_file = None
    compare_file = None
    for i, arg in enumerate(sys.argv):
        if arg == "--save" and i + 1 < len(sys.argv):
            save_file = sys.argv[i + 1]
        if arg == "--compare" and i + 1 < len(sys.argv):
            compare_file = sys.argv[i + 1]

    # Init hg38
    pm.gdb_init(HG38_ROOT)

    results = []
    threshold = sum(max(row) for row in CTCF_PSSM) * 0.7  # ~70% of max score

    print(f"CTCF PSSM: {CTCF_PSSM.shape[0]} columns, threshold={threshold:.2f}")
    print()

    # Benchmark 1: vtrack edit distance, chr1 (249Mb), subs-only, max_edits=2
    print("Benchmark: vtrack pwm.edit_distance chr1 (249Mb), K=2, subs-only")
    def vtrack_chr1_subs():
        pm.gvtrack_create("v_ed", None, func="pwm.edit_distance",
                          pssm=CTCF_PSSM, score_thresh=threshold, max_edits=2)
        try:
            pm.gextract("v_ed", intervals=pm.gintervals("chr1"),
                        iterator=pm.gintervals("chr1"))
        finally:
            pm.gvtrack_rm("v_ed")
    results.append(bench(vtrack_chr1_subs, "vtrack chr1 249Mb K=2 subs"))

    # Benchmark 2: vtrack edit distance, chr22 (51Mb), subs-only, max_edits=2
    print("Benchmark: vtrack pwm.edit_distance chr22 (51Mb), K=2, subs-only")
    def vtrack_chr22_subs():
        pm.gvtrack_create("v_ed", None, func="pwm.edit_distance",
                          pssm=CTCF_PSSM, score_thresh=threshold, max_edits=2)
        try:
            pm.gextract("v_ed", intervals=pm.gintervals("chr22"),
                        iterator=pm.gintervals("chr22"))
        finally:
            pm.gvtrack_rm("v_ed")
    results.append(bench(vtrack_chr22_subs, "vtrack chr22 51Mb K=2 subs"))

    # Benchmark 3: vtrack edit distance, chr1, exact mode (no K cap)
    print("Benchmark: vtrack pwm.edit_distance chr22 (51Mb), exact, subs-only")
    def vtrack_chr22_exact():
        pm.gvtrack_create("v_ed", None, func="pwm.edit_distance",
                          pssm=CTCF_PSSM, score_thresh=threshold)
        try:
            pm.gextract("v_ed", intervals=pm.gintervals("chr22"),
                        iterator=pm.gintervals("chr22"))
        finally:
            pm.gvtrack_rm("v_ed")
    results.append(bench(vtrack_chr22_exact, "vtrack chr22 51Mb exact subs"))

    # Benchmark 4: gseq_pwm_edits on 1Mb genomic region
    print("Benchmark: gseq_pwm_edits 1Mb region, K=2, subs-only")
    def gseq_1mb():
        pm.gseq_pwm_edits(
            seqs=pm.gintervals("chr1", 0, 1_000_000),
            pssm=CTCF_PSSM, score_thresh=threshold, max_edits=2
        )
    results.append(bench(gseq_1mb, "gseq_pwm_edits 1Mb K=2 subs"))

    # Benchmark 5: vtrack with indels D=1, chr22
    print("Benchmark: vtrack pwm.edit_distance chr22, K=2, D=1")
    def vtrack_chr22_indel1():
        pm.gvtrack_create("v_ed", None, func="pwm.edit_distance",
                          pssm=CTCF_PSSM, score_thresh=threshold,
                          max_edits=2, max_indels=1)
        try:
            pm.gextract("v_ed", intervals=pm.gintervals("chr22"),
                        iterator=pm.gintervals("chr22"))
        finally:
            pm.gvtrack_rm("v_ed")
    results.append(bench(vtrack_chr22_indel1, "vtrack chr22 51Mb K=2 D=1"))

    print()
    print("=" * 60)
    if do_json:
        print(json.dumps(results, indent=2))

    if save_file:
        with open(save_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_file}")

    if compare_file:
        with open(compare_file) as f:
            old = json.load(f)
        old_map = {r["label"]: r for r in old}
        print()
        print("Comparison (old → new):")
        for r in results:
            if r["label"] in old_map:
                o = old_map[r["label"]]
                speedup = o["median_s"] / r["median_s"] if r["median_s"] > 0 else float("inf")
                print(f"  {r['label']}: {o['median_s']:.4f}s → {r['median_s']:.4f}s ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
