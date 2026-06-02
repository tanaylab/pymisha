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
import os
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

    # 8. _aggregate_overlapping: pure-Python vs C++ fast path on 100k overlapping intervals.
    results.append(bench_liftover_aggregate())

    # 9. _parse_chain_file: pure-Python vs C++ fast path on a ~100k-block chain.
    results.append(bench_chain_parser())

    # 10. _read_source_track: pure-Python vs C++ fast path on 1M dense bins.
    results.append(bench_read_source_track())

    # 11. pm_chain_intervals_resolve: pure-Python vs C++ overlap-policy resolution.
    results += bench_chain_intervals_resolve()

    # 12. pm_map_intervals: pure-Python vs C++ map-intervals (100k src x 100k chain).
    results.append(bench_map_intervals())

    # 13. gtrack_liftover: end-to-end C++ orchestrator vs pure-Python on 1M-bin + 10k-chain.
    results.append(bench_gtrack_liftover())

    # 14. gtrack_import_mappedseq: C++ vs Python fallback on a 100k-read synthetic SAM.
    results.append(bench_import_mappedseq())

    results.append(bench_sliding_reducer())

    return results


def bench_liftover_aggregate():
    """Compare pure-Python vs C++ _aggregate_overlapping on ~100k overlapping
    intervals across ~22 chroms. Returns a single-summary dict with both
    timings and the speedup."""
    from pymisha.liftover import _aggregate_overlapping, _AGG_FUNCS

    rng = np.random.default_rng(60427)
    n = 100_000
    chrom_names = [f"chr{i}" for i in range(1, 23)]
    chroms = rng.choice(chrom_names, size=n)
    starts = rng.integers(0, 10_000_000, size=n).astype(np.int64)
    lengths = rng.integers(50, 500, size=n).astype(np.int64)
    ends = starts + lengths
    values = rng.normal(size=n)
    df = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends, "value": values})

    mean_fn = _AGG_FUNCS["mean"]

    # Warm both paths once.
    _aggregate_overlapping(df, mean_fn, na_rm=True)
    _aggregate_overlapping(df, mean_fn, na_rm=True, agg_name="mean")

    gc.collect()
    t0 = time.perf_counter()
    _aggregate_overlapping(df, mean_fn, na_rm=True)
    t1 = time.perf_counter()
    python_seconds = t1 - t0

    gc.collect()
    t0 = time.perf_counter()
    _aggregate_overlapping(df, mean_fn, na_rm=True, agg_name="mean")
    t1 = time.perf_counter()
    cpp_seconds = t1 - t0

    speedup = python_seconds / cpp_seconds if cpp_seconds > 0 else float("inf")
    return {
        "label": "liftover_aggregate_100k",
        "name": "liftover_aggregate",
        "n": n,
        "python_seconds": round(python_seconds, 6),
        "cpp_seconds": round(cpp_seconds, 6),
        "speedup": round(speedup, 3),
        # Keep these so the text formatter at the bottom still works.
        "median_s": round(cpp_seconds, 6),
        "std_s": 0.0,
    }


def bench_chain_parser():
    """Compare pure-Python vs C++ _parse_chain_file on a ~100k-block synthetic
    chain. Returns a single-summary dict with both timings and the speedup."""
    import os
    import tempfile

    from pymisha.liftover import _get_db_chrom_sizes, _parse_chain_file

    # 200 chains x 500 blocks = 100k blocks.
    n_chains = 200
    blocks_per_chain = 500

    with tempfile.NamedTemporaryFile(
        "w", suffix=".chain", delete=False, dir=tempfile.gettempdir()
    ) as fh:
        path = fh.name
        for cid in range(1, n_chains + 1):
            src_start = (cid - 1) * 300 + 100
            tgt_start = (cid - 1) * 500 + 100
            block = 1
            total_src = blocks_per_chain * block + (blocks_per_chain - 1) * 1
            total_tgt = blocks_per_chain * block + (blocks_per_chain - 1) * 1
            src_end = src_start + total_src
            tgt_end = tgt_start + total_tgt
            # Cap to chrom sizes (tgt chrom is "1" -> 500000 in the test DB).
            tgt_end = min(tgt_end, 499_000)
            score = 1000 + cid
            fh.write(
                f"chain {score} chr25 100000 + {src_start} {src_end} "
                f"1 500000 + {tgt_start} {tgt_end} {cid}\n"
            )
            for _ in range(blocks_per_chain - 1):
                fh.write(f"{block}\t1\t1\n")
            fh.write(f"{block}\n\n")

    sizes = _get_db_chrom_sizes()

    try:
        # Warm both paths once.
        _parse_chain_file(path, sizes, _force_pure_python=True)
        _parse_chain_file(path, sizes)

        gc.collect()
        t0 = time.perf_counter()
        py_out = _parse_chain_file(path, sizes, _force_pure_python=True)
        t1 = time.perf_counter()
        python_seconds = t1 - t0

        gc.collect()
        t0 = time.perf_counter()
        cpp_out = _parse_chain_file(path, sizes)
        t1 = time.perf_counter()
        cpp_seconds = t1 - t0
    finally:
        os.unlink(path)

    speedup = python_seconds / cpp_seconds if cpp_seconds > 0 else float("inf")
    return {
        "label": "chain_parser_100k_blocks",
        "name": "chain_parser",
        "n_chains": n_chains,
        "blocks_per_chain": blocks_per_chain,
        "n_rows_py": len(py_out["chrom"]) if py_out else 0,
        "n_rows_cpp": len(cpp_out["chrom"]) if cpp_out else 0,
        "python_seconds": round(python_seconds, 6),
        "cpp_seconds": round(cpp_seconds, 6),
        "speedup": round(speedup, 3),
        "median_s": round(cpp_seconds, 6),
        "std_s": 0.0,
    }


def bench_read_source_track():
    """Compare pure-Python vs C++ _read_source_track on ~1M valid dense bins
    across 3 chroms (bin_size=100, ~330k bins per chrom, ~10% NaN).
    Returns a single-summary dict with both timings and the speedup.

    The Python hot path being eliminated is the `list(zip(*.tolist()))` +
    `pd.DataFrame(rows)` materialization for the per-chrom dense decode.
    """
    import os
    import struct
    import tempfile

    import pymisha as pm  # noqa: F401  - imported for side-effect (registers _pymisha)
    from pymisha.liftover import _read_source_track, _read_source_track_python

    rng = np.random.default_rng(60427)
    bin_size = 100
    n_bins_per_chrom = 330_000
    chroms = ["1", "2", "X"]

    with tempfile.TemporaryDirectory(prefix="pymisha_bench_p3a_") as tmp:
        track_dir = os.path.join(tmp, "t")
        os.makedirs(track_dir)
        for chrom_name in chroms:
            vals = rng.normal(size=n_bins_per_chrom).astype(np.float32)
            mask = rng.random(n_bins_per_chrom) < 0.1
            vals[mask] = np.nan
            payload = struct.pack("<i", bin_size) + vals.tobytes(order="C")
            with open(os.path.join(track_dir, chrom_name), "wb") as fh:
                fh.write(payload)

        # Warm both paths once.
        _read_source_track(track_dir)
        _read_source_track_python(track_dir)

        gc.collect()
        t0 = time.perf_counter()
        cpp_type, cpp_df = _read_source_track(track_dir)
        t1 = time.perf_counter()
        cpp_seconds = t1 - t0

        gc.collect()
        t0 = time.perf_counter()
        py_type, py_df = _read_source_track_python(track_dir)
        t1 = time.perf_counter()
        python_seconds = t1 - t0

    assert cpp_type == py_type
    assert len(cpp_df) == len(py_df), (len(cpp_df), len(py_df))

    speedup = python_seconds / cpp_seconds if cpp_seconds > 0 else float("inf")
    return {
        "label": "read_source_track_1m_bins",
        "name": "read_source_track",
        "n_chroms": len(chroms),
        "bin_size": bin_size,
        "n_bins_per_chrom": n_bins_per_chrom,
        "n_rows": int(len(cpp_df)),
        "python_seconds": round(python_seconds, 6),
        "cpp_seconds": round(cpp_seconds, 6),
        "speedup": round(speedup, 3),
        "median_s": round(cpp_seconds, 6),
        "std_s": 0.0,
    }


def bench_chain_intervals_resolve():
    """Benchmark pm_chain_intervals_resolve vs the pure-Python pair on a
    100k-row synthetic chain with ~20% target-overlap density."""
    import _pymisha

    from pymisha.liftover import _resolve_chain_overlaps

    rng = np.random.RandomState(60427)
    n_rows = 100_000
    rows_per_chrom = n_rows // 4
    rows = []
    cid = 1
    for chrom_idx in range(4):
        chrom = f"chr{chrom_idx + 1}"
        chromsrc = f"chrA{chrom_idx + 1}"
        cursor = 0
        for _ in range(rows_per_chrom):
            start = cursor + int(rng.randint(0, 100))
            length = int(rng.randint(50, 300))
            end = start + length
            if rng.random() < 0.2:
                cursor = max(0, start - int(rng.randint(10, length)))
            else:
                cursor = end
            rows.append((chrom, start, end, 0, chromsrc,
                         start, end, 0, cid, float(rng.randint(100, 10000))))
            cid += 1

    cols = list(zip(*rows, strict=True))
    chain_dict = {
        "chrom":     np.array(cols[0], dtype=object),
        "start":     np.array(cols[1], dtype=np.int64),
        "end":       np.array(cols[2], dtype=np.int64),
        "strand":    np.array(cols[3], dtype=np.int64),
        "chromsrc":  np.array(cols[4], dtype=object),
        "startsrc":  np.array(cols[5], dtype=np.int64),
        "endsrc":    np.array(cols[6], dtype=np.int64),
        "strandsrc": np.array(cols[7], dtype=np.int64),
        "chain_id":  np.array(cols[8], dtype=np.int64),
        "score":     np.array(cols[9], dtype=np.float64),
    }

    def py_fn():
        return _resolve_chain_overlaps(chain_dict, "keep", "auto_score",
                                       _force_pure_python=True)

    def cpp_fn():
        return _pymisha.pm_chain_intervals_resolve(chain_dict, "keep", "auto_score")

    py_res = bench(py_fn, "chain_intervals_resolve (Python)", n_warmup=1, n_reps=3)
    cpp_res = bench(cpp_fn, "chain_intervals_resolve (C++)",  n_warmup=1, n_reps=3)
    speedup = py_res["median_s"] / max(cpp_res["median_s"], 1e-9)
    print(f"  speedup = {speedup:.2f}x")
    return [py_res, cpp_res, {"label": "chain_intervals_resolve (speedup)",
                              "median_s": round(speedup, 3), "std_s": 0.0}]


def bench_map_intervals(n_src: int = 100_000, n_chain: int = 100_000) -> dict:
    """Benchmark pm_map_intervals vs the pure-Python pair on synthetic 100k x 100k inputs."""
    import os as _os

    from pymisha import liftover as _liftover

    rng = np.random.default_rng(60427)

    # Synthetic chain: n_chain disjoint 100-bp blocks on one src chrom.
    starts_src = np.sort(rng.integers(0, 10_000_000, size=n_chain).astype(np.int64))
    # Enforce minimum gap of 100 between starts so blocks stay disjoint.
    starts_src = np.maximum.accumulate(starts_src + np.arange(n_chain) * 100)
    ends_src = starts_src + 100
    starts_tgt = starts_src.copy()
    ends_tgt = ends_src.copy()
    chain = pd.DataFrame({
        "chrom":     ["chr1"] * n_chain,
        "start":     starts_tgt,
        "end":       ends_tgt,
        "strand":    np.zeros(n_chain, dtype=np.int64),
        "chromsrc":  ["src"] * n_chain,
        "startsrc":  starts_src,
        "endsrc":    ends_src,
        "strandsrc": np.zeros(n_chain, dtype=np.int64),
        "chain_id":  np.arange(n_chain, dtype=np.int64) + 1,
        "score":     np.full(n_chain, 1000.0, dtype=np.float64),
    })

    # Synthetic src intervals: n_src 1000-bp intervals scattered over the same chrom.
    iv_starts = rng.integers(0, int(starts_src[-1]) + 100, size=n_src).astype(np.int64)
    iv_ends = iv_starts + 1000
    intervals = pd.DataFrame({
        "chrom": ["src"] * n_src,
        "start": iv_starts,
        "end":   iv_ends,
    })

    # Warm both paths once.
    _liftover._map_intervals_vectorized(intervals, chain, False, None)
    _liftover._map_intervals_vectorized(
        intervals, chain, False, None, _force_pure_python=True,
    )

    # Time the C++ path (env-var path, matching how users hit it).
    gc.collect()
    t0 = time.perf_counter()
    cpp_out = _liftover._map_intervals_vectorized(intervals, chain, False, None)
    t_cpp = time.perf_counter() - t0

    # Time the Python path.
    _os.environ["PYMISHA_FORCE_PY_MAP_INTERVALS"] = "1"
    try:
        gc.collect()
        t0 = time.perf_counter()
        py_out = _liftover._map_intervals_vectorized(intervals, chain, False, None)
        t_py = time.perf_counter() - t0
    finally:
        _os.environ.pop("PYMISHA_FORCE_PY_MAP_INTERVALS", None)

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")
    return {
        "label": "map_intervals_100k_x_100k",
        "name": "map_intervals (100k src x 100k chain)",
        "n_src": n_src,
        "n_chain": n_chain,
        "py_seconds": round(t_py, 6),
        "cpp_seconds": round(t_cpp, 6),
        "speedup": round(speedup, 3),
        "py_rows": int(len(py_out)),
        "cpp_rows": int(len(cpp_out)),
        # Keep these so the text formatter at the bottom still works.
        "median_s": round(t_cpp, 6),
        "std_s": 0.0,
    }


def bench_gtrack_liftover(n_bins: int = 1_000_000, n_chain: int = 10_000) -> dict:
    """Benchmark pm_liftover_track vs the pure-Python orchestrator on a
    1M-bin synthetic dense track + 10k-row synthetic chain."""
    import shutil
    import struct
    import tempfile

    import pymisha as pm

    rng = np.random.default_rng(60427)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "src.track")
        os.makedirs(src_dir, exist_ok=True)
        bin_size = 100
        vals = rng.random(n_bins).astype(np.float32)
        with open(os.path.join(src_dir, "srcA"), "wb") as f:
            f.write(struct.pack("I", bin_size))
            vals.tofile(f)

        # Synthetic chain: n_chain disjoint 1000-bp blocks on chr "1".
        starts_src = np.sort(rng.integers(0, n_bins * bin_size,
                                          size=n_chain).astype(np.int64))
        starts_src = np.maximum.accumulate(starts_src + np.arange(n_chain) * 1000)
        ends_src = starts_src + 1000
        starts_tgt = starts_src.copy()
        ends_tgt = ends_src.copy()
        chain = pd.DataFrame({
            "chrom":     ["1"] * n_chain,
            "start":     starts_tgt,
            "end":       ends_tgt,
            "strand":    np.zeros(n_chain, dtype=np.int64),
            "chromsrc":  ["srcA"] * n_chain,
            "startsrc":  starts_src,
            "endsrc":    ends_src,
            "strandsrc": np.zeros(n_chain, dtype=np.int64),
            "chain_id":  np.arange(n_chain, dtype=np.int64) + 1,
            "score":     np.full(n_chain, 1000.0, dtype=np.float64),
        })
        chain.attrs["tgt_overlap_policy"] = "keep"

        test_db = "tests/testdb/trackdb/test"
        if not os.path.isdir(test_db):
            return {"label": "gtrack_liftover_1m_x_10k", "skip": True,
                    "median_s": 0.0, "std_s": 0.0}
        db_root = os.path.join(tmpdir, "trackdb", "test")
        shutil.copytree(test_db, db_root)
        pm.gdb_init(db_root)

        # Time C++ path.
        gc.collect()
        t0 = time.perf_counter()
        pm.gtrack_liftover("bench_cpp", "bench", src_dir, chain,
                           multi_target_agg="mean")
        t_cpp = time.perf_counter() - t0

        # Time Python path.
        os.environ["PYMISHA_FORCE_PY_LIFTOVER_TRACK"] = "1"
        try:
            gc.collect()
            t0 = time.perf_counter()
            pm.gtrack_liftover("bench_py", "bench", src_dir, chain,
                               multi_target_agg="mean")
            t_py = time.perf_counter() - t0
        finally:
            os.environ.pop("PYMISHA_FORCE_PY_LIFTOVER_TRACK", None)

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")
    return {
        "label": "gtrack_liftover_1m_x_10k",
        "name": "gtrack_liftover (1M bins x 10k chain)",
        "n_bins": n_bins,
        "n_chain": n_chain,
        "py_seconds": round(t_py, 6),
        "cpp_seconds": round(t_cpp, 6),
        "speedup": round(speedup, 3),
        "median_s": round(t_cpp, 6),
        "std_s": 0.0,
    }


def bench_import_mappedseq(n_reads: int = 100_000) -> dict:
    """Benchmark pm_import_mappedseq vs the pure-Python fallback on a
    synthetic n_reads-line SAM.

    Uses the small test DB (chrom '1', 500k bp); reads are uniformly
    distributed within the chrom. Mix of + and - strand. Some duplicates
    expected on a 100k-read uniform sample over 500k positions.
    """
    import shutil
    import tempfile

    rng = np.random.default_rng(60427)

    test_db = "tests/testdb/trackdb/test"
    if not os.path.isdir(test_db):
        return {"label": "import_mappedseq_100k", "skip": True,
                "median_s": 0.0, "std_s": 0.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        db_root = os.path.join(tmpdir, "trackdb", "test")
        shutil.copytree(test_db, db_root)

        # Synthetic SAM: header + n_reads alignment lines.
        sam_path = os.path.join(tmpdir, "reads.sam")
        chrom = "1"
        chrom_size = 500_000
        seq_len = 50
        coords = rng.integers(0, chrom_size - seq_len, size=n_reads)
        flags = rng.choice([0, 16], size=n_reads)  # 0 = +, 16 = -
        seq = "A" * seq_len
        with open(sam_path, "w") as fh:
            fh.write("@HD\tVN:1.6\n")
            fh.write(f"@SQ\tSN:{chrom}\tLN:{chrom_size}\n")
            for i in range(n_reads):
                fh.write(
                    f"r{i}\t{flags[i]}\t{chrom}\t{coords[i]}\t30\t"
                    f"{seq_len}M\t*\t0\t0\t{seq}\t*\n"
                )

        pm.gdb_init(db_root)

        # Time C++ path.
        gc.collect()
        t0 = time.perf_counter()
        pm.gtrack_import_mappedseq(
            "bench_cpp_dense", "bench", sam_path,
            pileup=50, binsize=100,
            cols_order=None, remove_dups=True,
        )
        t_cpp_dense = time.perf_counter() - t0

        # Time Python fallback.
        os.environ["PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ"] = "1"
        try:
            gc.collect()
            t0 = time.perf_counter()
            pm.gtrack_import_mappedseq(
                "bench_py_dense", "bench", sam_path,
                pileup=50, binsize=100,
                cols_order=None, remove_dups=True,
            )
            t_py_dense = time.perf_counter() - t0
        finally:
            os.environ.pop("PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ", None)

        # Time C++ sparse.
        gc.collect()
        t0 = time.perf_counter()
        pm.gtrack_import_mappedseq(
            "bench_cpp_sparse", "bench", sam_path,
            pileup=0, binsize=-1,
            cols_order=None, remove_dups=True,
        )
        t_cpp_sparse = time.perf_counter() - t0

        os.environ["PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ"] = "1"
        try:
            gc.collect()
            t0 = time.perf_counter()
            pm.gtrack_import_mappedseq(
                "bench_py_sparse", "bench", sam_path,
                pileup=0, binsize=-1,
                cols_order=None, remove_dups=True,
            )
            t_py_sparse = time.perf_counter() - t0
        finally:
            os.environ.pop("PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ", None)

    speedup_dense = t_py_dense / t_cpp_dense if t_cpp_dense > 0 else float("inf")
    speedup_sparse = t_py_sparse / t_cpp_sparse if t_cpp_sparse > 0 else float("inf")
    return {
        "label": f"import_mappedseq_{n_reads // 1000}k",
        "name": f"gtrack_import_mappedseq ({n_reads} SAM reads)",
        "n_reads": n_reads,
        "py_dense_seconds": round(t_py_dense, 6),
        "cpp_dense_seconds": round(t_cpp_dense, 6),
        "dense_speedup": round(speedup_dense, 3),
        "py_sparse_seconds": round(t_py_sparse, 6),
        "cpp_sparse_seconds": round(t_cpp_sparse, 6),
        "sparse_speedup": round(speedup_sparse, 3),
        "median_s": round(t_cpp_dense, 6),
        "std_s": 0.0,
    }


def bench_sliding_reducer() -> dict:
    """Windowed-lse vtrack: incremental sliding window vs from-scratch recompute.

    Times a func="lse" vtrack with a 1000-bin (sshift/eshift) window scanned
    bin-by-bin over chrom 1 of the test DB, reduced by gextract. The sliding path
    pops/pushes the step bins per advance; PYMISHA_DISABLE_SLIDING_REDUCER forces
    the legacy per-bin recompute, so we can report the head-to-head speedup.
    """
    pm.gdb_init(TESTDB)
    pm.CONFIG["progress"] = False
    pm.CONFIG["multitasking"] = False
    pm.gvtrack_clear()
    pm.gvtrack_create("bench_slide_vt", "dense_track", func="lse")
    pm.gvtrack_iterator("bench_slide_vt", sshift=-25000, eshift=25000)
    iv = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})

    def _run():
        pm.gextract("bench_slide_vt", intervals=iv, iterator=50)

    os.environ.pop("PYMISHA_DISABLE_SLIDING_REDUCER", None)
    slide = bench(_run, "sliding_lse_window1000")
    os.environ["PYMISHA_DISABLE_SLIDING_REDUCER"] = "1"
    try:
        recompute = bench(_run, "recompute_lse_window1000")
    finally:
        os.environ.pop("PYMISHA_DISABLE_SLIDING_REDUCER", None)
    pm.gvtrack_rm("bench_slide_vt")

    speedup = (recompute["median_s"] / slide["median_s"]
               if slide["median_s"] > 0 else float("inf"))
    return {
        "label": "sliding_lse_vtrack_window1000",
        "name": "windowed lse vtrack (1000-bin window, chrom 1)",
        "slide_seconds": slide["median_s"],
        "recompute_seconds": recompute["median_s"],
        "speedup": round(speedup, 3),
        "median_s": slide["median_s"],
        "std_s": slide["std_s"],
    }


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
