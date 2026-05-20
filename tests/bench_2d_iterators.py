"""Benchmark 2D FixedRect iterator: pymisha vs R misha 5.7.x.

Run:
    python tests/bench_2d_iterators.py
    python tests/bench_2d_iterators.py --json

Discovery notes
---------------
No 2D track was found in ~/hg38 (checked all 16035 tracks; the trackdb
contains 1D sparse/dense tracks only).  The benchmark therefore uses:

  - Workload A ("testdb_rects_track"): the existing testdb rects_track
    across all 3 chrom pairs at 100kbp x 100kbp resolution.  Small but
    shared with the test suite, so it exercises the real code path and
    enables a direct R comparison.

  - Workload B ("synth_large"): a synthetic single-pair 2D track with
    ~10k rectangular objects over a 30Mbp x 30Mbp space at 500kbp x 500kbp
    resolution (3600 bins).  Demonstrates scanner throughput at a scale
    comparable to a Hi-C contact map sub-region.

R comparison is run for Workload A only (testdb), because R misha can
read the same on-disk files.  Workload B uses a temp dir not registered
with a full R trackdb, so R is skipped there.

Requirements
------------
- R misha >= 5.7.0 installed (optional; benchmark degrades gracefully)
- testdb at tests/testdb/trackdb/test (ships with the repo)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure we import from THIS worktree, not a system install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pymisha

import pymisha as pm
from pymisha._quadtree import write_2d_track_file

TESTDB = str(Path(__file__).resolve().parent / "testdb" / "trackdb" / "test")

N_WARMUP = 2
N_REPS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(func, label, n_warmup=N_WARMUP, n_reps=N_REPS):
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


def _rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _r_misha_version() -> str | None:
    """Return installed R misha version string, or None."""
    if not _rscript_available():
        return None
    try:
        out = subprocess.check_output(
            ["Rscript", "-e", "cat(as.character(packageVersion('misha')))"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.decode().strip()
    except Exception:
        return None


def _run_r(script: str, timeout: int = 120) -> str | None:
    """Run an R script string, return stdout or None on failure."""
    if not _rscript_available():
        return None
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", "-"],
            input=script.encode(),
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  [R stderr]: {result.stderr.decode()[:300]}", file=sys.stderr)
            return None
        return result.stdout.decode()
    except Exception as e:
        print(f"  [R error]: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Workload A: testdb rects_track  (shared with test suite)
# ---------------------------------------------------------------------------

def _workload_a_pymisha() -> dict:
    """
    FixedRect extract on testdb rects_track, all 3 chrom pairs,
    100kbp x 100kbp bins.

    Chrom sizes: 1=500kbp, 2=300kbp, X=200kbp.
    Scope covers each pair as one large interval, then the scanner tiles it.
    """
    pm.gdb_init(TESTDB)
    _pymisha.pm_dbreload()

    # Full-genome 2D scope (all ordered pairs)
    intervals = pm.gintervals_2d_all(mode="full")

    def run():
        return pm.gextract("rects_track", intervals=intervals, iterator=(100_000, 100_000))

    r = _bench(run, "pymisha-testdb-rects_track-100k")
    result_df = run()
    r["rows"] = int(len(result_df))
    r["track"] = "rects_track"
    r["workload"] = "testdb_rects_track"
    r["binsize"] = "100kbp x 100kbp"
    r["scope"] = "all 9 chrom-pairs (1x1,1x2,1xX,2x1,...)"
    return r


def _workload_a_r() -> dict | None:
    """Same workload in R misha."""
    script = rf"""
library(misha)
gdb.init("{TESTDB}")
intervals <- gintervals.2d.all()

# Expand to all ordered pairs (mode="full" equivalent: row + col symmetry)
pairs <- expand.grid(
  chrom1 = c("chr1","chr2","chrX"),
  chrom2 = c("chr1","chr2","chrX"),
  stringsAsFactors=FALSE
)
csizes <- list(chr1=500000, chr2=300000, chrX=200000)
intervals2 <- data.frame(
  chrom1 = pairs$chrom1,
  start1 = 0L,
  end1   = unlist(csizes[pairs$chrom1]),
  chrom2 = pairs$chrom2,
  start2 = 0L,
  end2   = unlist(csizes[pairs$chrom2]),
  stringsAsFactors = FALSE
)

# Warmup
for (i in seq_len(2)) {{
  invisible(gextract("rects_track", intervals=intervals2, iterator=c(100000, 100000)))
}}

# Timed reps
times <- numeric(5)
nrows <- 0L
for (i in seq_len(5)) {{
  t0 <- proc.time()[["elapsed"]]
  res <- gextract("rects_track", intervals=intervals2, iterator=c(100000, 100000))
  times[i] <- proc.time()[["elapsed"]] - t0
  nrows <- nrow(res)
}}
med <- median(times)
std <- sd(times)
cat(sprintf(
  paste0(
    '{{"label":"r-testdb-rects_track-100k",',
    '"median_s":%.6f,"std_s":%.6f,"rows":%d,',
    '"workload":"testdb_rects_track","binsize":"100kbp x 100kbp"}}\n'
  ),
  med, std, nrows
))
"""
    out = _run_r(script)
    if out is None:
        return None
    # parse last JSON line
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip().startswith("{")]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Workload B: synthetic large track (~10k objects, 30Mbp x 30Mbp, 500kbp bins)
# ---------------------------------------------------------------------------

def _build_synth_track(tmpdir: str, n_objects: int = 10_000, chrom_size: int = 30_000_000) -> str:
    """
    Create a synthetic 2D rectangles track in tmpdir.

    The track has one chrom pair (chrom "1" x "1") with n_objects non-overlapping
    rectangles scattered over [0, chrom_size) x [0, chrom_size).
    Returns the track directory path.
    """
    rng = np.random.default_rng(60427)

    # Generate non-overlapping rectangles: tile the space roughly uniformly,
    # then jitter.  We use a grid of cells of ~300kbp x 300kbp and place one
    # object per cell for simplicity - this guarantees non-overlap and gives a
    # known distribution.
    cell = int(chrom_size / np.sqrt(n_objects))  # ~300kbp for n=10k, size=30Mbp
    positions = [(i * cell, j * cell) for i in range(int(chrom_size / cell))
                 for j in range(int(chrom_size / cell))]
    rng.shuffle(positions)
    positions = positions[:n_objects]

    obj_size = cell // 2  # object half-cell to avoid overlap
    objects = [
        (x, y, x + obj_size, y + obj_size, float(rng.random()))
        for x, y in positions
    ]

    track_dir = os.path.join(tmpdir, "bench_synth.track")
    os.makedirs(track_dir, exist_ok=True)
    with open(os.path.join(track_dir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    write_2d_track_file(
        os.path.join(track_dir, "1-1"),
        objects,
        (0, 0, chrom_size, chrom_size),
        is_points=False,
        chunk_size=10_000_000,
    )
    return track_dir


def _workload_b_pymisha(tmpdir: str) -> dict:
    """
    FixedRect extract on the synthetic track: full 30Mbp x 30Mbp scope
    at 500kbp x 500kbp bins -> 60x60 = 3600 bins.
    """
    # We need a minimal trackdb pointing at tmpdir so pymisha can init.
    # Create a minimal trackdb structure: tmpdir IS the trackdb root.
    db_root = tmpdir
    os.makedirs(os.path.join(db_root, "tracks"), exist_ok=True)
    # Move track dir under tracks/
    synth_src = os.path.join(db_root, "bench_synth.track")
    synth_dst = os.path.join(db_root, "tracks", "bench_synth.track")
    if not os.path.exists(synth_dst):
        shutil.move(synth_src, synth_dst)

    # Write chrom_sizes file
    chrom_size = 30_000_000
    cs_path = os.path.join(db_root, "chrom_sizes.txt")
    with open(cs_path, "w") as f:
        f.write(f"1\t{chrom_size}\n")

    # pymisha requires .db file to init
    db_file = os.path.join(db_root, ".db")
    with open(db_file, "w") as f:
        f.write("")

    pm.gdb_init(db_root)
    _pymisha.pm_dbreload()

    chrom_size = 30_000_000
    binsize = 500_000
    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [chrom_size],
        "chrom2": ["1"],
        "start2": [0],
        "end2":   [chrom_size],
    })

    def run():
        return pm.gextract("bench_synth", intervals=intervals, iterator=(binsize, binsize))

    r = _bench(run, f"pymisha-synth-{chrom_size//1_000_000}Mbp-500k")
    result_df = run()
    r["rows"] = int(len(result_df))
    r["track"] = "bench_synth (synthetic)"
    r["workload"] = "synth_large"
    r["binsize"] = f"{binsize//1000}kbp x {binsize//1000}kbp"
    r["scope"] = f"1 chrom-pair, {chrom_size//1_000_000}Mbp x {chrom_size//1_000_000}Mbp"
    r["n_objects"] = 10_000
    return r


# ---------------------------------------------------------------------------
# Workload C: TrackRects iterator  (rects_track, full chr1-chr1 scope)
# ---------------------------------------------------------------------------

def _workload_c_pymisha() -> dict:
    """
    TrackRects iterator on testdb rects_track: full chr1 x chr1 scope
    (500kbp x 500kbp).  Returns one row per intersecting object.
    """
    pm.gdb_init(TESTDB)
    _pymisha.pm_dbreload()

    intervals = pd.DataFrame({
        "chrom1": ["1"],
        "start1": [0],
        "end1":   [500_000],
        "chrom2": ["1"],
        "start2": [0],
        "end2":   [500_000],
    })

    def run():
        return pm.gextract("rects_track", intervals=intervals, iterator="rects_track")

    r = _bench(run, "pymisha-testdb-rects_track-track_rects_iter")
    result_df = run()
    r["rows"] = int(len(result_df))
    r["track"] = "rects_track"
    r["workload"] = "track_rects"
    r["iterator"] = "rects_track (TrackRects)"
    r["scope"] = "chr1 x chr1, [0, 500kbp)"
    return r


def _workload_c_r() -> dict | None:
    """Same workload in R misha using iterator='rects_track'."""
    script = rf"""
library(misha)
gdb.init("{TESTDB}")

intervals2 <- data.frame(
  chrom1 = "chr1",
  start1 = 0L,
  end1   = 500000L,
  chrom2 = "chr1",
  start2 = 0L,
  end2   = 500000L,
  stringsAsFactors = FALSE
)

# Warmup
for (i in seq_len(2)) {{
  invisible(gextract("rects_track", intervals=intervals2, iterator="rects_track"))
}}

# Timed reps
times <- numeric(5)
nrows <- 0L
for (i in seq_len(5)) {{
  t0 <- proc.time()[["elapsed"]]
  res <- gextract("rects_track", intervals=intervals2, iterator="rects_track")
  times[i] <- proc.time()[["elapsed"]] - t0
  nrows <- nrow(res)
}}
med <- median(times)
std <- sd(times)
cat(sprintf(
  paste0(
    '{{"label":"r-testdb-rects_track-track_rects_iter",',
    '"median_s":%.6f,"std_s":%.6f,"rows":%d,',
    '"workload":"track_rects","iterator":"rects_track (TrackRects)"}}\n'
  ),
  med, std, nrows
))
"""
    out = _run_r(script)
    if out is None:
        return None
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip().startswith("{")]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Workload D: CartesianGrid iterator (testdb rects_track, 5 centers)
#
# 5 centers on chr1 with a 4-breakpoint expansion (3 windows per axis).
# CartesianGrid produces 5*3 x 5*3 = 225 cells in the full chr1 x chr1 scope.
# The track has 3 objects on chr1-chr1, so each cell is scored (avg agg).
#
# Same testdb files read by both pymisha and R, enabling a direct comparison.
# ---------------------------------------------------------------------------

_CGI_SCOPE_2D = pd.DataFrame({
    "chrom1": ["1"],
    "start1": [0],
    "end1":   [500_000],
    "chrom2": ["1"],
    "start2": [0],
    "end2":   [500_000],
})
_CGI_CENTERS = pd.DataFrame({
    "chrom": ["1"] * 5,
    "start": [0,       100_000, 200_000, 300_000, 400_000],
    "end":   [50_000, 150_000, 250_000, 350_000, 450_000],
})
_CGI_EXPANSION = [-30_000, -10_000, 10_000, 30_000]


def _workload_d_pymisha() -> dict:
    """CartesianGrid iterator on testdb rects_track: 5 centers x 3 windows/axis."""
    pm.gdb_init(TESTDB)
    _pymisha.pm_dbreload()

    spec = pm.giterator_cartesian_grid(_CGI_CENTERS, _CGI_EXPANSION, stream=True)

    def run():
        return pm.gextract("rects_track", intervals=_CGI_SCOPE_2D, iterator=spec)

    r = _bench(run, "pymisha-testdb-cartesian_grid-5c-3w")
    result_df = run()
    r["rows"] = int(len(result_df))
    r["track"] = "rects_track"
    r["workload"] = "cartesian_grid"
    r["iterator"] = "CartesianGrid (5 centers, 3 windows/axis)"
    r["scope"] = "chr1 x chr1, [0, 500kbp)"
    return r


def _workload_d_r() -> dict | None:
    """Same CartesianGrid workload in R misha."""
    script = rf"""
library(misha)
gdb.init("{TESTDB}")

centers <- gintervals(rep(1, 5), c(0, 100000, 200000, 300000, 400000), c(50000, 150000, 250000, 350000, 450000))
expansion <- c(-30000, -10000, 10000, 30000)
itr <- giterator.cartesian_grid(centers, expansion)

intervals2 <- data.frame(
  chrom1 = "chr1", start1 = 0L, end1 = 500000L,
  chrom2 = "chr1", start2 = 0L, end2 = 500000L,
  stringsAsFactors = FALSE
)

# Warmup
for (i in seq_len({N_WARMUP})) {{
  invisible(gextract("rects_track", intervals=intervals2, iterator=itr))
}}

# Timed reps
times <- numeric({N_REPS})
nrows <- 0L
for (i in seq_len({N_REPS})) {{
  t0 <- proc.time()[["elapsed"]]
  res <- gextract("rects_track", intervals=intervals2, iterator=itr)
  times[i] <- proc.time()[["elapsed"]] - t0
  nrows <- nrow(res)
}}
med <- median(times)
std <- sd(times)
cat(sprintf(
  paste0(
    '{{"label":"r-testdb-cartesian_grid-5c-3w",',
    '"median_s":%.6f,"std_s":%.6f,"rows":%d,',
    '"workload":"cartesian_grid","iterator":"CartesianGrid (5 centers, 3 windows/axis)"}}\n'
  ),
  med, std, nrows
))
"""
    out = _run_r(script)
    if out is None:
        return None
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip().startswith("{")]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Workload E: IntervalsPolicy via scanner  (rects_track, 10 scope rects, chr1)
#
# Measures two paths on the same scope DataFrame:
#   - scanner path  (PYMISHA_USE_SCANNER_FOR_INTERVALS=1): one row per scope
#     rect; "area" reducer; per-rect aggregation.
#   - bypass path   (PYMISHA_USE_SCANNER_FOR_INTERVALS=0): one row per
#     (scope_rect, track-object) intersection; per-object enumeration.
#
# The two paths return different output shapes, so no cross-path correctness
# comparison is made here.  The bypass numbers are printed purely for
# reference (both run on the same testdb track and intervals).
#
# R comparison is omitted: R's gextract with intervals= uses the scanner
# aggregation shape (like the scanner path), but pymisha's bypass is a
# different code path — cross-language comparison is not meaningful here.
# ---------------------------------------------------------------------------

_WORKLOAD_E_SCOPE = pd.DataFrame({
    "chrom1": ["1"] * 10,
    "start1": [i * 50_000       for i in range(10)],
    "end1":   [(i + 1) * 50_000 for i in range(10)],
    "chrom2": ["1"] * 10,
    "start2": [i * 50_000       for i in range(10)],
    "end2":   [(i + 1) * 50_000 for i in range(10)],
})


def _workload_e_scanner() -> dict:
    """IntervalsPolicy (scanner) on testdb rects_track: 10 chr1 x chr1 scope rects."""
    pm.gdb_init(TESTDB)
    _pymisha.pm_dbreload()
    os.environ["PYMISHA_USE_SCANNER_FOR_INTERVALS"] = "1"
    try:
        def run():
            return pm.gextract("rects_track", intervals=_WORKLOAD_E_SCOPE)

        r = _bench(run, "pymisha-intervals-via-scanner-10rects")
        result_df = run()
        r["rows"] = int(len(result_df))
        r["track"] = "rects_track"
        r["workload"] = "intervals_via_scanner"
        r["path"] = "scanner (IntervalsPolicy)"
        r["scope"] = "10 chr1 x chr1 rects (50kbp each)"
        r["output_shape"] = "one row per scope rect (aggregated)"
    finally:
        os.environ.pop("PYMISHA_USE_SCANNER_FOR_INTERVALS", None)
    return r


def _workload_e_bypass() -> dict:
    """Bypass path on testdb rects_track: same 10 scope rects, per-object enumeration."""
    pm.gdb_init(TESTDB)
    _pymisha.pm_dbreload()
    os.environ["PYMISHA_USE_SCANNER_FOR_INTERVALS"] = "0"
    try:
        def run():
            return pm.gextract("rects_track", intervals=_WORKLOAD_E_SCOPE)

        r = _bench(run, "pymisha-intervals-via-bypass-10rects")
        result_df = run()
        r["rows"] = int(len(result_df))
        r["track"] = "rects_track"
        r["workload"] = "intervals_via_scanner"
        r["path"] = "bypass (per-object enumeration)"
        r["scope"] = "10 chr1 x chr1 rects (50kbp each)"
        r["output_shape"] = "one row per (scope_rect, track_object) intersection"
    finally:
        os.environ.pop("PYMISHA_USE_SCANNER_FOR_INTERVALS", None)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark 2D iterators (FixedRect + TrackRects + CartesianGrid + IntervalsViaScanner).")
    p.add_argument("--json", action="store_true", help="Emit results as JSON lines")
    p.add_argument(
        "--workload",
        choices=["testdb_rects_track", "synth_large", "track_rects", "cartesian_grid", "intervals_via_scanner", "all"],
        default="all",
        help="Which workload to run (default: all)",
    )
    args = p.parse_args()

    r_version = _r_misha_version()

    results = []
    verdicts = []

    # ---- Workload A -------------------------------------------------------
    if args.workload in ("testdb_rects_track", "all"):
        print("=== Workload A: testdb rects_track, 100kbp x 100kbp ===", flush=True)
        print("  Running pymisha...", flush=True)
        py_a = _workload_a_pymisha()
        print(
            f"  pymisha: median={py_a['median_s']:.4f}s  std={py_a['std_s']:.4f}s  rows={py_a['rows']}",
            flush=True,
        )
        results.append(py_a)

        if r_version is not None:
            print(f"  Running R misha {r_version}...", flush=True)
            r_a = _workload_a_r()
            if r_a is not None:
                print(
                    f"  R misha: median={r_a['median_s']:.4f}s  std={r_a['std_s']:.4f}s  rows={r_a['rows']}",
                    flush=True,
                )
                results.append(r_a)

                ratio = py_a["median_s"] / r_a["median_s"] if r_a["median_s"] > 0 else float("inf")
                verdict = "PASS (within 20%)" if ratio <= 1.20 else f"FAIL (pymisha {ratio:.2f}x R)"
                print(f"  Perf ratio: {ratio:.3f}x  -> {verdict}", flush=True)
                verdicts.append({"workload": "testdb_rects_track", "ratio": round(ratio, 4), "verdict": verdict})
            else:
                print("  R misha run failed (see stderr above).", flush=True)
                verdicts.append({"workload": "testdb_rects_track", "ratio": None, "verdict": "R_FAILED"})
        else:
            print("  R misha not available - skipping R comparison.", flush=True)
            verdicts.append({"workload": "testdb_rects_track", "ratio": None, "verdict": "R_NOT_AVAILABLE"})

    # ---- Workload B -------------------------------------------------------
    if args.workload in ("synth_large", "all"):
        print("\n=== Workload B: synthetic 10k-object 30Mbp track, 500kbp x 500kbp ===", flush=True)
        tmpdir = tempfile.mkdtemp(prefix="pymisha_bench_")
        try:
            print("  Building synthetic track...", flush=True)
            _build_synth_track(tmpdir)
            print("  Running pymisha...", flush=True)
            py_b = _workload_b_pymisha(tmpdir)
            print(
                f"  pymisha: median={py_b['median_s']:.4f}s  std={py_b['std_s']:.4f}s  rows={py_b['rows']}",
                flush=True,
            )
            results.append(py_b)
            verdicts.append({
                "workload": "synth_large",
                "ratio": None,
                "verdict": "R_NOT_RUN (no hg38 2D track; R comparison uses testdb only)",
            })
        finally:
            # Restore testdb init so tests still work after this script.
            pm.gdb_init(TESTDB)
            _pymisha.pm_dbreload()
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- Workload C: TrackRects iterator ----------------------------------
    if args.workload in ("track_rects", "all"):
        print(
            "\n=== Workload C: TrackRects iterator, rects_track chr1 x chr1 ===",
            flush=True,
        )
        print("  Running pymisha...", flush=True)
        py_c = _workload_c_pymisha()
        print(
            f"  pymisha: median={py_c['median_s']:.4f}s  std={py_c['std_s']:.4f}s  rows={py_c['rows']}",
            flush=True,
        )
        results.append(py_c)

        if r_version is not None:
            print(f"  Running R misha {r_version}...", flush=True)
            r_c = _workload_c_r()
            if r_c is not None:
                print(
                    f"  R misha: median={r_c['median_s']:.4f}s  std={r_c['std_s']:.4f}s  rows={r_c['rows']}",
                    flush=True,
                )
                results.append(r_c)
                ratio = py_c["median_s"] / r_c["median_s"] if r_c["median_s"] > 0 else float("inf")
                verdict = "PASS (within 20%)" if ratio <= 1.20 else f"WARN (pymisha {ratio:.2f}x R)"
                print(f"  Perf ratio: {ratio:.3f}x  -> {verdict}", flush=True)
                verdicts.append({
                    "workload": "track_rects",
                    "ratio": round(ratio, 4),
                    "verdict": verdict,
                })
            else:
                print("  R misha run failed (see stderr above).", flush=True)
                verdicts.append({"workload": "track_rects", "ratio": None, "verdict": "R_FAILED"})
        else:
            print("  R misha not available - skipping R comparison.", flush=True)
            verdicts.append({"workload": "track_rects", "ratio": None, "verdict": "R_NOT_AVAILABLE"})

    # ---- Workload D: CartesianGrid iterator ---------------------------------
    if args.workload in ("cartesian_grid", "all"):
        print(
            "\n=== Workload D: CartesianGrid iterator, rects_track chr1 x chr1 ===",
            flush=True,
        )
        print("  Running pymisha...", flush=True)
        py_d = _workload_d_pymisha()
        print(
            f"  pymisha: median={py_d['median_s']*1000:.1f}ms  std={py_d['std_s']*1000:.1f}ms"
            f"  rows={py_d['rows']}",
            flush=True,
        )
        results.append(py_d)

        if r_version is not None:
            print(f"  Running R misha {r_version}...", flush=True)
            r_d = _workload_d_r()
            if r_d is not None:
                print(
                    f"  R misha: median={r_d['median_s']*1000:.1f}ms  std={r_d['std_s']*1000:.1f}ms"
                    f"  rows={r_d['rows']}",
                    flush=True,
                )
                results.append(r_d)
                ratio = py_d["median_s"] / r_d["median_s"] if r_d["median_s"] > 0 else float("inf")
                verdict = "PASS (within 20%)" if ratio <= 1.20 else f"WARN (pymisha {ratio:.2f}x R)"
                print(f"  Perf ratio: {ratio:.3f}x  -> {verdict}", flush=True)
                verdicts.append({
                    "workload": "cartesian_grid",
                    "ratio": round(ratio, 4),
                    "verdict": verdict,
                })
            else:
                print("  R misha run failed (see stderr above).", flush=True)
                verdicts.append({"workload": "cartesian_grid", "ratio": None, "verdict": "R_FAILED"})
        else:
            print("  R misha not available - skipping R comparison.", flush=True)
            verdicts.append({"workload": "cartesian_grid", "ratio": None, "verdict": "R_NOT_AVAILABLE"})

    # ---- Workload E: IntervalsPolicy (scanner) vs bypass -------------------
    if args.workload in ("intervals_via_scanner", "all"):
        print(
            "\n=== Workload E: IntervalsPolicy via scanner vs bypass, rects_track 10 chr1 rects ===",
            flush=True,
        )
        print("  Running pymisha (scanner path)...", flush=True)
        py_e_scan = _workload_e_scanner()
        print(
            f"  pymisha (scanner, {py_e_scan['output_shape']}): "
            f"median={py_e_scan['median_s']*1000:.1f}ms  std={py_e_scan['std_s']*1000:.1f}ms"
            f"  rows={py_e_scan['rows']}",
            flush=True,
        )
        results.append(py_e_scan)

        print("  Running pymisha (bypass path)...", flush=True)
        py_e_bypass = _workload_e_bypass()
        print(
            f"  pymisha (bypass, {py_e_bypass['output_shape']}): "
            f"median={py_e_bypass['median_s']*1000:.1f}ms  std={py_e_bypass['std_s']*1000:.1f}ms"
            f"  rows={py_e_bypass['rows']}",
            flush=True,
        )
        results.append(py_e_bypass)

        print(
            "  Note: scanner and bypass return different output shapes; no R comparison run.",
            flush=True,
        )
        verdicts.append({
            "workload": "intervals_via_scanner",
            "ratio": None,
            "verdict": "NO_R_COMPARISON (different output shapes; scanner=aggregated, bypass=per-object)",
        })

    # ---- Summary ----------------------------------------------------------
    print("\n=== Summary ===", flush=True)
    print("  hg38 2D tracks: NONE found (all 16035 tracks are 1D)", flush=True)
    print(f"  R misha version: {r_version or 'not installed'}", flush=True)
    for v in verdicts:
        ratio_str = f"  ratio={v['ratio']:.3f}x" if v["ratio"] is not None else ""
        print(f"  [{v['workload']}]{ratio_str}  {v['verdict']}", flush=True)

    if args.json:
        print("\n--- JSON results ---")
        for r in results:
            print(json.dumps(r))
        for v in verdicts:
            print(json.dumps({"type": "verdict", **v}))


if __name__ == "__main__":
    main()
