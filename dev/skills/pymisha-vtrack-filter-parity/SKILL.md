---
name: pymisha-vtrack-filter-parity
description: Implement or extend gvtrack_filter parity with focused tests and safe function-by-function rollout.
---

# PyMisha VTrack Filter Parity

## Use when

- You are adding or fixing `gvtrack_filter` behavior.
- A vtrack function should respect masks and currently does not.
- You need to port `test-gvtrack.filter.R` cases into Python.

## Core approach

1. Add or update targeted tests first in `tests/test_gvtrack_filter.py`.
2. Keep filter source resolution centralized in `pymisha/vtracks.py`:
- DataFrame with `chrom/start/end`
- intervals set name
- sparse/interval track name
- list/tuple union of any of the above
3. Canonicalize filter intervals before attaching:
- normalize chrom names
- validate coordinates
- merge overlaps/touching intervals
4. Apply filter semantics after iterator shift.
5. For unsupported function semantics, raise `NotImplementedError` with a precise function name.

## Validation checklist

1. Run focused tests:
- `pytest -q tests/test_gvtrack_filter.py`
2. Run adjacent virtual-track tests:
- `pytest -q tests/test_vtracks.py tests/test_golden_master_vtracks.py`
3. Update status docs in same change:
- `README.md`
- `dev/notes/2026-01-27-pymisha-implementation-plan-v2.md`
- `dev/notes/r-tests-porting-matrix.md`

## Notes

- Keep heavy scanning logic in C++ where possible; Python-side filtering should remain bounded and chunk-local.
- Preserve explicit `NaN` behavior for fully masked intervals.
- Track unsupported function gaps in the implementation plan so next steps remain obvious.
