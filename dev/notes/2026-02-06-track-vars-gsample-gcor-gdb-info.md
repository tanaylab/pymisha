# Track Variables, gsample, gcor, gdb.info Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add gtrack.var.* (track variable management), gsample (streaming reservoir sampling), gcor (Pearson/Spearman correlation), and gdb.info (database inspection) to pymisha.

**Architecture:**
- gtrack.var.* are pure Python (file I/O for R-serialized objects stored in `vars/` subdirectory of track directory)
- gsample uses C++ StreamSampler (already ported) via a new `pm_sample` C++ function
- gcor needs a new C++ `pm_cor` function for streaming Pearson correlation with optional Spearman support
- gdb.info is pure Python (filesystem inspection)

**Tech Stack:** Python, C++ (CPython C API), NumPy, pandas, R serialization (via pyreadr)

---

### Task 1: gtrack_var_ls - List track variables

**Files:**
- Modify: `pymisha/tracks.py`
- Modify: `pymisha/__init__.py`
- Test: `tests/test_gtrack_var.py`

### Task 2: gtrack_var_get / gtrack_var_set / gtrack_var_rm - Track variable CRUD

**Files:**
- Modify: `pymisha/tracks.py`
- Test: `tests/test_gtrack_var.py`

### Task 3: gsample - Streaming reservoir sampling

**Files:**
- Modify: `src/PMStubs.cpp` (add pm_sample C++ function)
- Modify: `src/pymisha_init.cpp` (register pm_sample)
- Modify: `pymisha/summary.py` (add Python wrapper)
- Modify: `pymisha/__init__.py` (export)
- Test: `tests/test_gsample.py`

### Task 4: gcor - Pearson correlation (streaming C++)

**Files:**
- Modify: `src/PMStubs.cpp` (add pm_cor C++ function)
- Modify: `src/pymisha_init.cpp` (register pm_cor)
- Modify: `pymisha/summary.py` (add Python wrapper)
- Modify: `pymisha/__init__.py` (export)
- Test: `tests/test_gcor.py`

### Task 5: gdb_info - Database information

**Files:**
- Modify: `pymisha/db.py`
- Modify: `pymisha/__init__.py` (export)
- Test: `tests/test_gdb_info.py`

### Task 6: Update README and plan
