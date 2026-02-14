# PyMisha Code Review: Verified Pending Issues

**Date:** 2026-02-13
**Status:** Verified against codebase.

This document lists issues that have been **verified** as still pending (unimplemented or untested) in the codebase.

---

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical | 0     |
| High     | 0     |
| Medium   | 1     |
| Low      | 1     |
| **Total** | **2** |

---

## Detailed Verified Pending Issues

### Critical (0)

No remaining critical issues.

---

### High (0)

No remaining high-severity issues.

---

### Medium (1)

#### R Parity
| # | Finding | File | Notes |
|---|---------|------|-------|
| 90 | **Track array suite** (gtrack.array.*) not implemented. | N/A | 5 functions missing. |

---

### Low (1)

#### R Parity & Test Quality
| # | Finding | File | Notes |
|---|---------|------|-------|
| 152 | **~7 R test files** have no Python port. | N/A | See R parity review for full list. |

---

## Resolved Issues (this session)

| # | Finding | Resolution |
|---|---------|------------|
| 87 | gextract tests missing edge cases | RESOLVED — 18 edge-case tests added. |
| 88 | gsummary tests missing edge cases | RESOLVED — 3 edge-case tests added. |
| 89 | gsample tests only check length/type | RESOLVED — 8 statistical tests added. |
| 91 | gcis_decay not implemented | RESOLVED — implemented in `pymisha/analysis.py`, 37 tests. |
| 92 | gintervals_import_genes not implemented | RESOLVED — implemented in `pymisha/intervals.py`, 38 tests. |
| 151 | gpartition tests mostly assert `is not None` | RESOLVED — 7 stronger assertion tests added. |
