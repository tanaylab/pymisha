# Cross-Language Misha Benchmark Suite

This suite benchmarks development versions of:
- R `misha` from `~/src/misha`
- Python `pymisha` from `~/src/pymisha`

It covers:
- APIs: `gextract`, `gscreen`, `gsummary`, `gquantiles`
- Virtual-track functions: `sum`, `avg`, `global.percentile`, `pwm`
- Source track density: dense + sparse tracks
- Iterator density: dense + sparse iterator settings
- Scope: single-chromosome + multi-chromosome
- Dataset sizes: small + medium + large interval profiles

## Scripts

- `dev/benchmarks/misha_crosslang/run_pymisha_bench.py`
- `dev/benchmarks/misha_crosslang/run_rmisha_bench.R`
- `dev/benchmarks/misha_crosslang/run_crosslang_bench.py`

## Quick Start

Run both suites and merge outputs:

```bash
python dev/benchmarks/misha_crosslang/run_crosslang_bench.py --warmup 1 --reps 5
```

Run Python suite only:

```bash
python dev/benchmarks/misha_crosslang/run_pymisha_bench.py \
  --pymisha-src ~/src/pymisha \
  --db-root ~/src/pymisha/tests/testdb/trackdb/test \
  --warmup 1 --reps 5 \
  --output-csv dev/benchmarks/results/pymisha_bench.csv
```

Run R suite only:

```bash
Rscript dev/benchmarks/misha_crosslang/run_rmisha_bench.R \
  --rmisha-src ~/src/misha \
  --db-root ~/src/pymisha/tests/testdb/trackdb/test \
  --warmup 1 --reps 5 \
  --output-csv dev/benchmarks/results/rmisha_bench.csv
```

## Output

`run_crosslang_bench.py` writes:
- `pymisha_bench.csv`
- `pymisha_bench.json`
- `rmisha_bench.csv`
- `crosslang_merged.csv`

under `dev/benchmarks/results/<timestamp>/` (or `--results-dir`).

## Notes

- R suite loads development code via `devtools::load_all("~/src/misha")`.
- Python suite imports from `~/src/pymisha` by prepending that path to `sys.path`.
- Current PyMisha may mark `global.percentile` as `unsupported`; this is recorded per-case instead of failing the entire run.
