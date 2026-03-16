# PyMisha Build + Quality Modernization Plan (Low Risk)

Last updated: 2026-02-13

This plan focuses on modernizing tooling without destabilizing parity/performance delivery.

## Guiding principle

Do infrastructure migrations only when they reduce risk for core goals:
- R parity
- streaming performance
- DB interoperability

Avoid changing build backend and API behavior in the same PR.

## Current state

- Build backend: `setuptools` (`pyproject.toml`)
- C++ extension: custom `src/*.cpp` build path
- Tests: `pytest` + golden-master parity tests
- Typing: basic `mypy` config, package includes `py.typed`
- Linting: migrated config toward `ruff`
- Docs: Sphinx + MyST scaffold + docs CI exists

## Decision on scikit-build-core

Recommendation: defer migration until parity-critical gaps are reduced.

Migration gates:
1. `gvtrack_filter` implemented and tested.
2. `glookup`/`gdist` streaming gap reduced (or clear design frozen).
3. 2D core API implementation started (so build changes are validated by broader native surface).

When gates are met, do migration in an infra-only PR sequence (below).

## Phased plan

### Phase A: immediate (safe, high ROI)

1. Strengthen type checking in CI
- Keep `mypy` as CI gate; move gradually toward stricter settings per module.
- Start with non-extension Python modules (`dataset.py`, `db.py`, utility functions).

2. ~~Add extension stubs (`.pyi`) for `_pymisha`~~ **DONE** (2026-02-13)
- Created `pymisha/_pymisha.pyi` with 46 function signatures + error class.
- All signatures inferred from C++ method table and Python wrapper call patterns.

3. ~~Adopt `pre-commit`~~ **DONE** (2026-02-13)
- Created `.pre-commit-config.yaml` with ruff check/format, trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files.
- Ruff config converged: line-length=120, select=[E,F,I,UP,W,B,SIM,PIE,RET,C4], isort known-first-party, per-file-ignores for tests.

4. Add property-based tests (`hypothesis`) for interval algorithms
- Good targets: `gintervals_union/intersect/diff/canonic/normalize/random`.
- Invariants (e.g., union idempotence, diff disjointness, bounds safety).

5. Add benchmark regression checks
- Use `pytest-benchmark` on selected streaming APIs (`gextract`, `gscreen`, `gsummary`, `gquantiles`).
- Keep thresholds conservative initially to avoid flaky failures.

### Phase B: packaging/distribution hardening

6. Add binary wheel builds via `cibuildwheel`
- Build wheels for Linux/macOS first.
- Add Windows once expected native dependencies and runtime behavior are validated.
- Publish to test index first, then production PyPI.

7. CI matrix modernization
- Python 3.10-3.12 first; expand only after wheel pipeline is stable.
- Separate jobs: lint/type, unit/parity tests, docs, wheels.

### Phase C: build backend migration (`setuptools` -> `scikit-build-core`)

8. Introduce CMake build in parallel (no removal yet)
- Add `CMakeLists.txt` to produce the existing extension module with same name.
- Validate local editable installs and test suite equivalence.

9. Switch backend in dedicated PR
- Move `[build-system]` to `scikit-build-core`.
- Keep package metadata in `[project]` unchanged.
- Ensure output artifacts and import paths remain stable.

10. Remove legacy build path after soak period
- Keep a short rollback window (one release cycle).

## Risk controls

- No mixed PRs: infra-only vs feature-only.
- Every phase requires:
  - parity tests passing,
  - docs build passing,
  - no regression in benchmark sentinels.
- Keep lockstep changelog entries for tooling changes.

## Suggested next concrete tasks

1. Add `pre-commit` config and CI hook.
2. Add initial `_pymisha.pyi` stub with core function signatures.
3. Add first `hypothesis` tests for interval set invariants.
4. Add `pytest-benchmark` baseline tests for 3-4 core streaming APIs.

## What not to do yet

- Do not migrate to `scikit-build-core` immediately.
- Do not enforce fully strict mypy repository-wide in one shot.
- Do not gate CI on aggressive benchmark thresholds before baseline stabilization.
