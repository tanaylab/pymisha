# PyMisha Architectural Findings & Optimization Report

**Date:** 2026-02-07
**Auditor:** Gemini CLI Agent

This document records architectural inefficiencies and C++ backend limitations identified during the optimization review.

## 1. Eager Data Loading (High Impact)
**Component:** `GenomeTrackSparse` (C++)
**Location:** `src/GenomeTrackSparse.cpp`, function `read_file_into_mem`.

**Issue:**
The current implementation eagerly loads *all* intervals and values for an entire chromosome into memory whenever any part of that chromosome is accessed.
- **Impact:** High memory usage and slow initial access latency for large sparse tracks.
- **Recommendation:** Refactor to use memory mapping (`mmap`) or a chunked/indexed reading mechanism (e.g., using `BufferedFile` more effectively or a true database-like page cache) to load data on-demand.

## 2. Non-Streaming Iterators (High Impact)
**Component:** `PMSparseIterator` (C++)
**Location:** `src/PMTrackExpressionIterator.cpp`, method `begin()`.

**Issue:**
The `begin()` method materializes the entire result set of the iterator in memory before iteration starts.
- **Impact:** Negates the benefits of "streaming" APIs when overlaps are dense or numerous. High memory spike at startup.
- **Recommendation:** Refactor `PMSparseIterator` to be a true lazy iterator that finds the next interval/overlap only when `next()` is called.

## 3. Buffered I/O Overhead (Medium Impact)
**Component:** `BufferedFile` (C++)
**Location:** `src/BufferedFile.h`.

**Issue:**
The class implements custom buffering logic with prefetching.
- **Impact:** While intended to optimize I/O, it may introduce unnecessary memory copying compared to OS-level `mmap`.
- **Recommendation:** Evaluate replacing custom buffering with direct kernel-level memory mapping for read-only genomic data access.

## 4. Python-Side Iteration Bottlenecks (Fixed/Mitigated)
**Component:** `gintervals_summary`, `gintervals_quantiles` (Python)
**Location:** `pymisha/summary.py`.

**Issue:**
Original implementation iterated over unique interval IDs in a Python loop ($O(N)$ overhead).
- **Fix (2026-02-07):** Optimized using `pandas.DataFrame.groupby` and vectorized aggregation.
- **Performance:** Achieved ~30-40x speedup in benchmarks.
- **Remaining:** `gintervals_canonic` still calculates index mapping via a Python loop. This should be moved to the C++ backend (`pm_intervals_canonic`).

## 5. VTrack Materialization (Medium Impact)
**Component:** `gsummary`, `gquantiles` (Python)
**Location:** `pymisha/summary.py`.

**Issue:**
When virtual tracks are involved, these functions fully materialize extracted values into a numpy array (`_extract_expr_values`) instead of using a streaming pass.
- **Recommendation:** Implement a streaming chunked processing path (similar to the optimized `gdist` vtrack path) to keep memory usage bounded for large genomic scopes.
