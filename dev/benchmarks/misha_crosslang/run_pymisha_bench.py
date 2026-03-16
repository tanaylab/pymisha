#!/usr/bin/env python3
"""Benchmark suite for development PyMisha.

This script benchmarks gextract/gscreen/gsummary/gquantiles over virtual tracks:
sum, avg, global.percentile, pwm.

It is designed to be paired with run_rmisha_bench.R using identical case IDs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

import numpy as np

QUANTILES = [0.25, 0.5, 0.75]


@dataclass(frozen=True)
class VTrackSpec:
    base_name: str
    func: str
    src: str | None
    source_density: str
    threshold: float


@dataclass(frozen=True)
class ProfileSpec:
    case_suffix: str
    chroms: tuple[str, ...]
    start: int
    end: int
    iterator: int
    iterator_density: str
    chrom_mode: str
    size_label: str


@dataclass
class BenchRow:
    impl: str
    case_id: str
    operation: str
    vtrack_label: str
    vtrack_name: str
    vtrack_func: str
    source_track: str
    source_density: str
    profile: str
    chrom_mode: str
    size_label: str
    iterator: int
    iterator_density: str
    warmup: int
    reps: int
    status: str
    median_s: float | None
    std_s: float | None
    min_s: float | None
    max_s: float | None
    result_rows: int | None
    error: str
    timestamp_utc: str
    package_path: str


VTRACK_SPECS: list[VTrackSpec] = [
    VTrackSpec("vt_sum_dense", "sum", "dense_track", "dense", 0.30),
    VTrackSpec("vt_sum_sparse", "sum", "sparse_track", "sparse", 0.40),
    VTrackSpec("vt_avg_dense", "avg", "dense_track", "dense", 0.08),
    VTrackSpec("vt_avg_sparse", "avg", "sparse_track", "sparse", 0.40),
    VTrackSpec(
        "vt_global_percentile_dense",
        "global.percentile",
        "dense_track",
        "dense",
        0.50,
    ),
    VTrackSpec("vt_pwm", "pwm", None, "sequence", 2.00),
]

PROFILE_SPECS: list[ProfileSpec] = [
    ProfileSpec(
        "single_small_dense_iter",
        ("1",),
        0,
        50_000,
        100,
        "dense",
        "single",
        "small",
    ),
    ProfileSpec(
        "single_full_sparse_iter",
        ("1",),
        0,
        -1,
        5_000,
        "sparse",
        "single",
        "large",
    ),
    ProfileSpec(
        "multi_medium_dense_iter",
        ("1", "2", "X"),
        0,
        100_000,
        200,
        "dense",
        "multi",
        "medium",
    ),
    ProfileSpec(
        "multi_full_sparse_iter",
        ("1", "2", "X"),
        0,
        -1,
        10_000,
        "sparse",
        "multi",
        "large",
    ),
]

OPERATIONS = ("gextract", "gscreen", "gsummary", "gquantiles")


def _default_pymisha_src() -> Path:
    return Path(os.environ.get("PYMISHA_SRC", "~/src/pymisha")).expanduser().resolve()


def _default_db_root(pymisha_src: Path) -> Path:
    env_db = os.environ.get("MISHA_BENCH_DB")
    if env_db:
        return Path(env_db).expanduser().resolve()
    return (pymisha_src / "tests" / "testdb" / "trackdb" / "test").resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyMisha benchmark suite.")
    parser.add_argument("--pymisha-src", default=None, help="Path to development PyMisha source.")
    parser.add_argument("--db-root", default=None, help="Misha DB root to benchmark against.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per case.")
    parser.add_argument("--reps", type=int, default=5, help="Timed repetitions per case.")
    parser.add_argument("--name-prefix", default="bench", help="Prefix for temporary benchmark vtracks.")
    parser.add_argument("--output-csv", default=None, help="Path to write CSV results.")
    parser.add_argument("--output-json", default=None, help="Path to write JSON results.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    return parser.parse_args()


def _count_rows(result: Any) -> int | None:
    if result is None:
        return 0
    if hasattr(result, "shape"):
        shape = getattr(result, "shape")
        if isinstance(shape, tuple) and len(shape) >= 1:
            try:
                return int(shape[0])
            except Exception:
                pass
    try:
        return int(len(result))
    except Exception:
        return None


def _bench_callable(func: Callable[[], Any], warmup: int, reps: int) -> tuple[float, float, float, float, int | None]:
    for _ in range(warmup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = func()

    times: list[float] = []
    result_rows: int | None = None
    for idx in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if idx == 0:
            result_rows = _count_rows(result)

    arr = np.array(times, dtype=float)
    return (
        float(np.median(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        result_rows,
    )


def _build_intervals(pm: Any, profile: ProfileSpec) -> Any:
    chroms: str | list[str]
    if len(profile.chroms) == 1:
        chroms = profile.chroms[0]
    else:
        chroms = list(profile.chroms)
    return pm.gintervals(chroms, profile.start, profile.end)


def _build_case_callable(
    pm: Any,
    operation: str,
    vtrack_name: str,
    threshold: float,
    intervals: Any,
    iterator: int,
) -> Callable[[], Any]:
    if operation == "gextract":
        return lambda: pm.gextract(vtrack_name, intervals, iterator=iterator)
    if operation == "gscreen":
        expr = f"{vtrack_name} > {threshold:.8g}"
        return lambda: pm.gscreen(expr, intervals, iterator=iterator)
    if operation == "gsummary":
        return lambda: pm.gsummary(vtrack_name, intervals, iterator=iterator)
    if operation == "gquantiles":
        return lambda: pm.gquantiles(vtrack_name, QUANTILES, intervals, iterator=iterator)
    raise ValueError(f"Unsupported operation: {operation}")


def _write_csv(rows: list[BenchRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in rows]
    fieldnames = list(payload[0].keys()) if payload else list(asdict(BenchRow(  # type: ignore[arg-type]
        impl="",
        case_id="",
        operation="",
        vtrack_label="",
        vtrack_name="",
        vtrack_func="",
        source_track="",
        source_density="",
        profile="",
        chrom_mode="",
        size_label="",
        iterator=0,
        iterator_density="",
        warmup=0,
        reps=0,
        status="",
        median_s=None,
        std_s=None,
        min_s=None,
        max_s=None,
        result_rows=None,
        error="",
        timestamp_utc="",
        package_path="",
    )).keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def _write_json(rows: list[BenchRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in rows], handle, indent=2)


def main() -> int:
    args = _parse_args()

    pymisha_src = Path(args.pymisha_src).expanduser().resolve() if args.pymisha_src else _default_pymisha_src()
    db_root = Path(args.db_root).expanduser().resolve() if args.db_root else _default_db_root(pymisha_src)
    if not pymisha_src.exists():
        raise FileNotFoundError(f"PyMisha source path does not exist: {pymisha_src}")
    if not db_root.exists():
        raise FileNotFoundError(f"DB root does not exist: {db_root}")

    sys.path.insert(0, str(pymisha_src))
    import pymisha as pm  # pylint: disable=import-outside-toplevel

    package_path = str(Path(inspect.getfile(pm)).resolve())
    pm.gdb_init(str(db_root))
    pm.CONFIG["progress"] = False
    pm.CONFIG["multitasking"] = False

    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    pssm = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

    vtrack_runtime_names: dict[str, str] = {}
    vtrack_creation_error: dict[str, str] = {}
    for spec in VTRACK_SPECS:
        runtime_name = f"{args.name_prefix}_{spec.base_name}_{run_tag}"
        try:
            if spec.func == "pwm":
                pm.gvtrack_create(
                    runtime_name,
                    None,
                    func="pwm",
                    pssm=pssm,
                    bidirect=True,
                    prior=0.01,
                    extend=True,
                )
            else:
                pm.gvtrack_create(runtime_name, spec.src, func=spec.func)
            vtrack_runtime_names[spec.base_name] = runtime_name
        except Exception as exc:  # noqa: BLE001
            vtrack_creation_error[spec.base_name] = f"{type(exc).__name__}: {exc}"

    interval_cache: dict[str, Any] = {}
    for profile in PROFILE_SPECS:
        interval_cache[profile.case_suffix] = _build_intervals(pm, profile)

    rows: list[BenchRow] = []
    timestamp_utc = datetime.now(UTC).isoformat()

    total_cases = len(OPERATIONS) * len(VTRACK_SPECS) * len(PROFILE_SPECS)
    case_idx = 0
    for operation in OPERATIONS:
        for spec in VTRACK_SPECS:
            for profile in PROFILE_SPECS:
                case_idx += 1
                case_id = f"{operation}__{spec.base_name}__{profile.case_suffix}"
                runtime_name = vtrack_runtime_names.get(spec.base_name, "")
                status = "success"
                err_msg = ""
                median_s = std_s = min_s = max_s = None
                result_rows = None

                if spec.base_name in vtrack_creation_error:
                    status = "unsupported"
                    err_msg = vtrack_creation_error[spec.base_name]
                else:
                    try:
                        call = _build_case_callable(
                            pm,
                            operation,
                            runtime_name,
                            spec.threshold,
                            interval_cache[profile.case_suffix],
                            profile.iterator,
                        )
                        median_s, std_s, min_s, max_s, result_rows = _bench_callable(
                            call,
                            warmup=args.warmup,
                            reps=args.reps,
                        )
                    except Exception as exc:  # noqa: BLE001
                        err_msg = f"{type(exc).__name__}: {exc}"
                        if "Unsupported virtual track function" in str(exc):
                            status = "unsupported"
                        else:
                            status = "error"

                rows.append(
                    BenchRow(
                        impl="pymisha",
                        case_id=case_id,
                        operation=operation,
                        vtrack_label=spec.base_name,
                        vtrack_name=runtime_name,
                        vtrack_func=spec.func,
                        source_track=spec.src or "NULL",
                        source_density=spec.source_density,
                        profile=profile.case_suffix,
                        chrom_mode=profile.chrom_mode,
                        size_label=profile.size_label,
                        iterator=profile.iterator,
                        iterator_density=profile.iterator_density,
                        warmup=args.warmup,
                        reps=args.reps,
                        status=status,
                        median_s=median_s,
                        std_s=std_s,
                        min_s=min_s,
                        max_s=max_s,
                        result_rows=result_rows,
                        error=err_msg,
                        timestamp_utc=timestamp_utc,
                        package_path=package_path,
                    )
                )

                if not args.quiet:
                    print(f"[{case_idx:03d}/{total_cases}] {case_id}: {status}")

    if args.output_csv:
        _write_csv(rows, Path(args.output_csv).expanduser().resolve())
    if args.output_json:
        _write_json(rows, Path(args.output_json).expanduser().resolve())

    if not args.quiet:
        ok = sum(r.status == "success" for r in rows)
        unsupported = sum(r.status == "unsupported" for r in rows)
        errors = sum(r.status == "error" for r in rows)
        print(
            f"Completed PyMisha benchmark suite: total={len(rows)} "
            f"success={ok} unsupported={unsupported} errors={errors}"
        )
        print(f"PyMisha package path: {package_path}")
        print(f"DB root: {db_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
