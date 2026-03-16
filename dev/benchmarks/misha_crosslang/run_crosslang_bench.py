#!/usr/bin/env python3
"""Run and compare development PyMisha vs R misha benchmark suites."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path


def _default_pymisha_src() -> Path:
    return Path(os.environ.get("PYMISHA_SRC", "~/src/pymisha")).expanduser().resolve()


def _default_rmisha_src() -> Path:
    return Path(os.environ.get("RMISHA_SRC", "~/src/misha")).expanduser().resolve()


def _default_db_root(pymisha_src: Path) -> Path:
    env_db = os.environ.get("MISHA_BENCH_DB")
    if env_db:
        return Path(env_db).expanduser().resolve()
    return (pymisha_src / "tests" / "testdb" / "trackdb" / "test").resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-language misha benchmarks.")
    parser.add_argument("--pymisha-src", default=None, help="Path to development PyMisha source.")
    parser.add_argument("--rmisha-src", default=None, help="Path to development R misha source.")
    parser.add_argument("--db-root", default=None, help="Misha DB root.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per case.")
    parser.add_argument("--reps", type=int, default=5, help="Timed repetitions per case.")
    parser.add_argument("--name-prefix", default="deploybench", help="Prefix for temporary benchmark vtracks.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for outputs. Defaults to dev/benchmarks/results/<timestamp>/",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output.")
    return parser.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merge_rows(py_rows: list[dict[str, str]], r_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    py_map = {row["case_id"]: row for row in py_rows}
    r_map = {row["case_id"]: row for row in r_rows}
    all_case_ids = sorted(set(py_map) | set(r_map))

    merged: list[dict[str, object]] = []
    for case_id in all_case_ids:
        py = py_map.get(case_id)
        rr = r_map.get(case_id)

        py_status = py["status"] if py else "missing"
        r_status = rr["status"] if rr else "missing"
        py_median = _to_float(py["median_s"]) if py else None
        r_median = _to_float(rr["median_s"]) if rr else None
        speedup = None
        if py_status == "success" and r_status == "success" and py_median and py_median > 0 and r_median is not None:
            speedup = r_median / py_median

        merged.append(
            {
                "case_id": case_id,
                "operation": (py or rr or {}).get("operation", ""),
                "vtrack_label": (py or rr or {}).get("vtrack_label", ""),
                "profile": (py or rr or {}).get("profile", ""),
                "py_status": py_status,
                "r_status": r_status,
                "py_median_s": py_median,
                "r_median_s": r_median,
                "py_std_s": _to_float(py["std_s"]) if py else None,
                "r_std_s": _to_float(rr["std_s"]) if rr else None,
                "speedup_r_over_py": speedup,
                "py_error": (py or {}).get("error", ""),
                "r_error": (rr or {}).get("error", ""),
            }
        )

    return merged


def main() -> int:
    args = _parse_args()
    script_dir = Path(__file__).resolve().parent
    py_runner = script_dir / "run_pymisha_bench.py"
    r_runner = script_dir / "run_rmisha_bench.R"

    pymisha_src = Path(args.pymisha_src).expanduser().resolve() if args.pymisha_src else _default_pymisha_src()
    rmisha_src = Path(args.rmisha_src).expanduser().resolve() if args.rmisha_src else _default_rmisha_src()
    db_root = Path(args.db_root).expanduser().resolve() if args.db_root else _default_db_root(pymisha_src)

    if not py_runner.exists():
        raise FileNotFoundError(f"Missing Python benchmark runner: {py_runner}")
    if not r_runner.exists():
        raise FileNotFoundError(f"Missing R benchmark runner: {r_runner}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser().resolve()
    else:
        results_dir = (script_dir.parent / "results" / timestamp).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    py_csv = results_dir / "pymisha_bench.csv"
    py_json = results_dir / "pymisha_bench.json"
    r_csv = results_dir / "rmisha_bench.csv"
    merged_csv = results_dir / "crosslang_merged.csv"

    py_cmd = [
        sys.executable,
        str(py_runner),
        "--pymisha-src",
        str(pymisha_src),
        "--db-root",
        str(db_root),
        "--warmup",
        str(args.warmup),
        "--reps",
        str(args.reps),
        "--name-prefix",
        args.name_prefix,
        "--output-csv",
        str(py_csv),
        "--output-json",
        str(py_json),
    ]
    if args.quiet:
        py_cmd.append("--quiet")

    r_cmd = [
        "Rscript",
        str(r_runner),
        "--rmisha-src",
        str(rmisha_src),
        "--db-root",
        str(db_root),
        "--warmup",
        str(args.warmup),
        "--reps",
        str(args.reps),
        "--name-prefix",
        args.name_prefix,
        "--output-csv",
        str(r_csv),
    ]
    if args.quiet:
        r_cmd.append("--quiet")

    if not args.quiet:
        print("Running PyMisha benchmarks...")
    subprocess.run(py_cmd, check=True)

    if not args.quiet:
        print("Running R misha benchmarks...")
    subprocess.run(r_cmd, check=True)

    py_rows = _load_csv(py_csv)
    r_rows = _load_csv(r_csv)
    merged = _merge_rows(py_rows, r_rows)
    _write_csv(merged, merged_csv)

    if not args.quiet:
        both_success = [r for r in merged if r["py_status"] == "success" and r["r_status"] == "success"]
        unsupported = [r for r in merged if "unsupported" in (r["py_status"], r["r_status"])]
        print(f"Merged results written: {merged_csv}")
        print(f"Total cases: {len(merged)}")
        print(f"Both success: {len(both_success)}")
        print(f"Unsupported on at least one side: {len(unsupported)}")
        if both_success:
            speedups = [r["speedup_r_over_py"] for r in both_success if r["speedup_r_over_py"] is not None]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"Average speedup (R median / Py median): {avg_speedup:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
