---
name: pymisha-docs-parity
description: Keep README/plan/docs/tests parity artifacts synchronized with implementation status.
---

# PyMisha Docs + Parity Sync

## Use when

- API status changed.
- New parity gaps were closed or discovered.
- R tests were ported to Python.

## Required updates

1. Update implementation source-of-truth:
- `dev/notes/2026-01-27-pymisha-implementation-plan-v2.md`

2. Update user-facing status:
- `README.md`

3. Update R->Python test coverage tracker:
- `dev/notes/r-tests-porting-matrix.md`

4. If build/tooling policy changed:
- `dev/notes/2026-02-06-build-quality-modernization-plan.md`

5. Keep docs nav consistent:
- `docs/roadmap.md`

## Validation

1. Build docs:
- `sphinx-build -W --keep-going -b html docs docs/_build/html`

2. Link check:
- `sphinx-build -W --keep-going -b linkcheck docs docs/_build/linkcheck`

3. Run changed/new tests:
- `pytest -q <targeted tests>`

4. Lint changed Python files:
- `ruff check <changed files>`
