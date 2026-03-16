---
name: pymisha-dev-cycle
description: Rebuild/test/benchmark cycle for PyMisha C++ and Python changes.
---

# PyMisha Dev Cycle

## Important context

- All paths, track names, and benchmark commands below are examples.
- Replace `"/home/aviezerl/mm10"` and example track IDs with the DB/tracks relevant to the current task.
- For routine local tests, the repository test DB (`tests/testdb/...`) is usually enough.

## Use when

- You changed `src/*.cpp` or C++-bound behavior.
- You need quick parity/performance checks.
- You need to debug multitask hangs in `gextract`/`gscreen`/summary paths.

## Workflow

1. Rebuild extension after C++ changes:
- `python -m pip install -e .`

2. Sanity check import + DB (example only; replace DB path):
- `python - <<'PY'`
- `import pymisha as pm`
- `pm.gdb_init("/home/aviezerl/mm10")`
- `print(pm.gtrack_ls()[:5])`
- `PY`

3. Run targeted tests:
- `pytest -k <name> -q`

4. Optional benchmark (Python, example DB + track):
- `python - <<'PY'`
- `import time, pymisha as pm`
- `pm.gdb_init("/home/aviezerl/mm10")`
- `track = "seq.IQ.pcg.flashzoi.mm10.rf524k_EB4_cnt"`
- `t0 = time.time()`
- `res = pm.gextract(track, pm.gintervals_all(), iterator=10000)`
- `print("rows", None if res is None else len(res))`
- `print("elapsed", time.time() - t0)`
- `print(res.head(3))`
- `PY`

5. Optional benchmark (R, example DB + track):
- `R -q -e 'library(misha); gsetroot("/home/aviezerl/mm10"); track <- "seq.IQ.pcg.flashzoi.mm10.rf524k_EB4_cnt"; print(system.time(gextract(track, gintervals.all(), iterator = 1e4)))'`

## Debug checklist for hangs/slowness

1. Force numeric iterator (`iterator=10000`).
2. Disable multitasking to isolate (`pm.CONFIG["multitasking"] = False`).
3. Enable progress (`pm.CONFIG["progress"] = "text"`).
4. If multitask-only issue: inspect FIFO/write serialization behavior.
