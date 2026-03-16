# PyMisha Gap Analysis Report

**Date:** 2026-03-02
**Version analyzed:** 0.1.11 (dev branch)
**Last updated:** 2026-03-02 (post P0-P2 sprint — 16 gaps resolved total, 231 new tests)
**Methodology:** 8-agent parallel analysis covering function inventory, core DB/tracks, intervals, virtual tracks, statistics, sequence, 2D tracks, and performance

---

## Executive Summary

PyMisha implements **132 of 138 in-scope R misha exports (96%)**. Six functions are permanently excluded from scope (5 track array functions + `gcluster.run`); `gwget` remains deferred. *(Updated 2026-03-03: GAP-033 gtrack_dbs/gintervals_dbs marked DONE — already fully implemented with 10 tests in test_dbs_functions.py. GAP-035 gdataset_example_path implemented — creates example dataset on the fly, 8 tests. GAP-034 gdb_mark_cache_dirty implemented — delegates to gdb_reload, 6 tests. GAP-040 gsynth_sample bin_merge marked DONE — already fully implemented with 8+ tests in TestBinMerge; GAP-025 track arrays marked NOT PLANNED — permanently excluded from scope. Updated 2026-03-02: +16 total from 2D sprint + P0-P2 sprint: dim param fix, trans mirroring, 2D union/intersect, 2D vtrack non-agg functions, 2D iterator, path functions, R-serialization detection, gextract file output, PWM spatial weighting, force_range fix, bigset iteration, gbins/k-mer/liftover optimization.)* The project is functionally complete for the most common genomic analysis workflows -- track extraction, interval operations, virtual tracks, liftover, and sequence analysis all work.

The remaining gaps fall into three categories:

1. **Missing subsystems** (COMPUTED 2D tracks): Track arrays are permanently excluded from scope (2026-03-03 decision — no known user demand; use R misha for track array workflows). COMPUTED tracks are specific to HiC normalization pipelines (detection + informative error now implemented; full computation engine deferred). Neither blocks typical usage.

2. **Performance gaps in pure-Python code paths** (quad-tree I/O, liftover, 2D extraction, gcis_decay): These are the most impactful gaps for users with large datasets. The quad-tree reader is 20-100x slower than C++ would be, and liftover mapping loops hit similar slowdowns on genome-scale chain files.

3. **Parameter and behavioral divergences** (missing output modes, regex vs glob, silent parameter ignoring): These are minor correctness issues that affect edge cases.

**Honest assessment:** For a beta release at v0.1.11, the coverage is strong. The critical path for most misha users (extract, summary, intervals, vtracks) works well and is backed by C++. The performance gaps matter only for specific workloads (large 2D datasets, genome-wide liftover, cis-decay on Hi-C data). The missing functions are genuinely obscure -- `gcompute_strands_autocorr`, `gtrack.convert`, `gtrack.path` are used by a small minority of R misha users.

---

## Function Inventory

### Coverage by Category

| Category | R functions | Implemented | Coverage |
|----------|------------|-------------|----------|
| Database management | 12 | 12 | 100% |
| Track operations | 22 | 18 | 82% |
| Interval operations | 28 | 29 | 100%+ |
| Virtual tracks | 10 | 10 | 100% |
| Data extraction/summary | 14 | 14 | 100% |
| Sequence analysis | 8 | 8 | 100% |
| 2D tracks | 6 | 6 | 100% |
| Liftover | 4 | 4 | 100% |
| Genome synthesis | 8 | 8 | 100% |
| Analysis | 4 | 4 | 100% |
| Utility/path functions | 8 | 3 | 38% |
| Track arrays | 5 | 0 | NOT PLANNED (excluded from scope) |
| Cluster/network | 2 | 0 | 1 NOT PLANNED (`gcluster.run`), 1 deferred (`gwget`) |
| Dataset | 6 | 6 | 100% |
| Directory | 4 | 4 | 100% |
| **In-scope total** | **138** | **133** | **96%** |
| *(Out-of-scope: 6 — track arrays ×5, gcluster.run ×1)* | | | |

### Missing Functions

| R Function | Status | Priority | Notes |
|-----------|--------|----------|-------|
| `gtrack.array.extract` | NOT PLANNED | P2 | Track arrays: permanently out of scope (2026-03-03) |
| `gtrack.array.import` | NOT PLANNED | P2 | Track arrays: permanently out of scope (2026-03-03) |
| `gtrack.array.get_colnames` | NOT PLANNED | P3 | Track arrays: permanently out of scope (2026-03-03) |
| `gtrack.array.set_colnames` | NOT PLANNED | P3 | Track arrays: permanently out of scope (2026-03-03) |
| `gvtrack.array.slice` | NOT PLANNED | P2 | Track arrays: permanently out of scope (2026-03-03) |
| `gcluster.run` | NOT PLANNED | P3 | Cluster job submission: R-specific, not applicable to Python |
| `gwget` | Deferred | P3 | Download track from URL |
| `gcompute_strands_autocorr` | ✅ Done | P3 | Strand autocorrelation (implemented 2026-03-03) |
| `gtrack.convert` | Missing | P3 | Track format migration |
| `gtrack.path` | ✅ Done | P2 | Return filesystem path (implemented 2026-03-02) |
| `gintervals.path` | ✅ Done | P2 | Return filesystem path (implemented 2026-03-02) |
| `gtrack.dbs` | Missing | P3 | Multi-DB track listing |
| `gintervals.dbs` | Missing | P3 | Multi-DB interval listing |
| `gdb.mark_cache_dirty` | Missing | P3 | Cache invalidation |
| `gdataset.example_path` | ✅ Done | P3 | Example path accessor (implemented 2026-03-03) |
| `grevcomp` | ✅ Done | P3 | Reverse complement string (implemented 2026-03-03) |
| `gintervals_2d_union` | ✅ Done | P2 | 2D set union (implemented 2026-03-02) |
| `gintervals_2d_intersect` | ✅ Done | P2 | 2D set intersection (implemented 2026-03-02) |
| `gtrack.2d.import` multi-file | ✅ Done | P3 | Multiple input files (implemented 2026-03-03) |
| `gintervals_neighbors` 2D params | Missing | P2 | 2D neighbor parameters |
| `gextract` file output | ✅ Done | P2 | Write results to file (implemented 2026-03-02) |
| `gextract` intervals_set_out | ✅ Done | P2 | Write intervals to named set (implemented 2026-03-02) |
| `gcompute_strands_autocorr` | ✅ Done | P3 | Strand autocorrelation (implemented 2026-03-03) |

---

## Feature Gaps by Domain

### 1. Core DB and Track Management

**Track arrays (NOT PLANNED)**
The entire track array subsystem (`gtrack.array.extract`, `gtrack.array.import`, `gtrack.array.get_colnames`, `gtrack.array.set_colnames`, `gvtrack.array.slice`) is permanently excluded from scope. Decision made 2026-03-03: this is a rarely-used R misha feature with no known pymisha user demand. Users who need track array functionality should use R misha. Track arrays store multi-column data per genomic bin -- used in some specialized pipelines (e.g., allele-specific expression). The five functions are excluded from coverage calculations and will not be implemented.

**Track variable serialization — ✅ FIXED (P1)**
~~Track variables fail silently on cross-language reads.~~ Fixed 2026-03-02: `gtrack_var_get` now detects R serialization magic bytes and raises an informative error. 6 tests.

**Path and listing functions — ✅ FIXED (P2)**
~~`gtrack.path` and `gintervals.path` missing.~~ Fixed 2026-03-02: `gtrack_path` and `gintervals_path` implemented. 13 tests. `gtrack.dbs`/`gintervals.dbs` remain P3 (multi-DB listing, rare). `gdb.mark_cache_dirty` also P3 (PyMisha uses `gdb_reload()` instead).

**Pattern matching — ✅ Already correct (P2)**
Investigation 2026-03-02 showed `gtrack_ls` and `gintervals_ls` already use `re.search()` (regex), matching R misha. Gap was a false positive.

### 2. Interval Operations

**2D set operations — ✅ FIXED (P2)**
~~`gintervals_2d_union` and `gintervals_2d_intersect` are missing.~~ Fixed 2026-03-02: both implemented with vectorized numpy broadcasting. 31 tests.

**`gintervals_force_range` column preservation — ✅ FIXED (P2)**
~~Extra columns dropped during clipping.~~ Fixed 2026-03-02: preserves all columns for both 1D and 2D intervals. 8 tests.

**`gintervals_neighbors` 2D parameters (P2)**
The 2D-specific neighbor parameters are not implemented. 2D neighbor queries are rare.

**`gintervals_annotate` tie_method — DONE (P3)**
~~The `tie_method` parameter for breaking annotation ties is not implemented.~~ Implemented 2026-03-03: supports `"first"`, `"min.start"`, `"min.end"`. 9 tests.

**Bigset transparent iteration — ✅ FIXED (P2)**
~~Not transparently iterable in all contexts.~~ Fixed 2026-03-02: `_maybe_load_intervals_set()` added to 21 functions across 8 modules. 22 tests.

### 3. Virtual Tracks

**`dim` parameter in `gvtrack_iterator` — ✅ FIXED (P1)**
~~The `dim` parameter that projects 2D tracks to 1D (extracting one dimension) is silently ignored.~~ Fixed 2026-03-02: dim=1/dim=2 now properly projects 2D tracks to 1D. 19 tests.

**2D vtrack function coverage — ✅ FIXED (P2)**
~~2D vtracks support only aggregation functions.~~ Fixed 2026-03-02: added exists, size, first, last, sample, global.percentile. 2D vtracks now have 11 functions. 53 tests.

**Python fallback overhead (P2)**
All vtrack expressions go through the Python fallback path, which evaluates vtracks in a separate pass from C++. This means 2+ DataFrame passes for mixed expressions vs a single C++ pass. For simple expressions this overhead is negligible; for large datasets it adds up.

### 4. Statistics and Extraction

**`gextract` output modes — ✅ FIXED (P2)**
~~Always returns DataFrame.~~ Fixed 2026-03-02: added `file` param (streaming TSV) and `intervals_set_out` param. Works for 1D and 2D. 10 tests.

**`intervals_set_out` parameter coverage — ✅ COMPLETE (GAP-042)**
All applicable functions now support `intervals_set_out`: `gextract`, `gintervals_summary`, `gintervals_quantiles`, `gpartition`, `gscreen`, `glookup`, `gintervals_2d_band_intersect`, `gintervals_force_range`, `gintervals_union`, `gintervals_intersect`, `gintervals_diff`, `gintervals_rbind`, `gintervals_normalize`, `gintervals_annotate`, `gneighbors`. Not applicable to `gdist` (R misha's `gdist` also lacks this parameter since it returns aggregate bin counts, not intervals).

**`gbins_summary`/`gbins_quantiles` — ✅ Optimized (P2)**
~~2-5x overhead from Python binning wrapper.~~ Optimized 2026-03-02: vectorized with numpy.bincount and sort-based grouping. 1.4-1.5x speedup. 4 tests.

**`gcis_decay` pure Python (P1)**
`gcis_decay` is implemented entirely in Python with per-interval row iteration. On large Hi-C datasets this is 20-100x slower than C++ would be. This is a commonly used analysis function in Hi-C workflows.

**`gintervals_mapply` Python inner loop (P2)**
The mapply function applies a user function to each interval group in Python. The per-group dispatch is pure Python. This is inherent to the design (user-provided Python callables) but could be optimized for common patterns.

**`gcompute_strands_autocorr` (P3)**
Strand autocorrelation is a specialized analysis function. Missing entirely. Used mainly in nascent transcription analysis.

**Multitask support -- ✅ DONE (P2)**
~~R misha's `gmax.processes` option missing.~~ Implemented: two-level parallelism. Python-level `_parallel_extract()` splits by chromosome via `multiprocessing.Pool(fork)` for gextract. C++ level fork/FIFO multitask (`PyMisha::prepare4multitasking()`) for gextract, gsummary, gquantiles, gdist, gscreen, glookup, gcor. `gmax_processes()` API exposed. Code reviewed 2026-03-03 with 18 comprehensive parity and edge-case tests.

### 5. Sequence Analysis

**`gseq_pwm` spatial weighting — ✅ FIXED (P1)**
~~`spat_factor`/`spat_bin` raise NotImplementedError.~~ Fixed 2026-03-02: log-space spatial factor application matching R misha C++ semantics. 21 tests.

**Pure Python implementations (P2)**
`gseq_pwm` is pure Python (10-30x slower than C++). `gseq_kmer` and `gseq_kmer_dist` were optimized 2026-03-02 with numpy stride_tricks-based hashing (3.5x speedup, 15 tests). C++ wiring for PWM scoring (GAP-009) remains open.

**`gsynth_sample` bin_merge — ✅ DONE (P3)**
~~The `bin_merge` parameter is not implemented in genome synthesis sampling.~~ Implemented: `bin_merge` is fully supported in `pymisha/gsynth.py`. 8+ tests in `TestBinMerge` class in `tests/test_gsynth.py`.

**`grevcomp` — ✅ DONE (P3)**
~~Standalone reverse-complement string function. Trivial to add but not exposed.~~ Implemented: `grevcomp()` in `pymisha/sequence.py`, delegating to `gseq_revcomp`. Exported from `pymisha/__init__.py`. 10 tests in `tests/test_grevcomp.py`.

### 6. 2D Tracks

**COMPUTED track type — detection done, full implementation deferred (P1)**
R misha supports COMPUTED 2D tracks -- tracks whose values are computed on-the-fly from other tracks using normalization models (PotentialComputer2D, TechnicalComputer2D). These are used in Hi-C normalization pipelines (observed/expected, ICE normalization). PyMisha now detects COMPUTED tracks and raises an informative `NotImplementedError` early in gextract, gsummary, gquantiles, gdist, gscreen, glookup, and gcor, with a clear message directing users to R misha. Full computation engine implementation is deferred.

**Trans contact mirroring — ✅ FIXED (P2)**
~~Trans contacts only written in canonical direction.~~ Fixed 2026-03-02: `gtrack_2d_import_contacts` now writes both chrA-chrB and chrB-chrA files with swapped coordinates. 8 tests.

**Band-aware aggregation (P2)**
2D aggregation with band filters falls back to full object enumeration instead of using pre-computed node stats. This is correct but slow for large tracks with tight bands.

**Single-chunk quad-tree writer (P2)**
The Python quad-tree writer creates a single chunk in memory. For very large 2D tracks (>100M contacts) this can cause memory issues. R misha uses multi-chunk writing with streaming.

**2D iterator — ✅ FIXED (P2)**
~~No dedicated 2D iterator abstraction exists.~~ Fixed 2026-03-02: `giterator_intervals_2d` generator function yields one DataFrame per interval. Supports band filtering, virtual tracks, multiple expressions. 21 tests.

---

## Performance Analysis

### Critical Hotspots (20-100x slower than C++ baseline)

| Component | Current | Bottleneck | Impact |
|-----------|---------|-----------|--------|
| Quad-tree reader | Python struct.unpack | Per-node struct.unpack vs C++ pointer cast | All 2D extraction |
| Liftover mapping | Python iterrows | Per-interval Python loop | Genome-wide liftover |
| 2D extraction | Python mmap + parse | Full Python I/O pipeline | All 2D workflows |
| gcis_decay | Python inner loop | Per-bin distance calculation | Hi-C cis-decay analysis |

### High Impact (10-30x)

| Component | Current | Bottleneck | Impact |
|-----------|---------|-----------|--------|
| PWM scoring | Python per-base loop | Per-base Python scoring | gseq_pwm on large intervals |
| Quad-tree writer | Python struct.pack | Per-node serialization | gtrack_2d_create on large datasets |
| ~~Liftover overlap policies~~ | ✅ Vectorized | Vectorized numpy/pandas | Large chain files |
| VTrack iterrows | Python per-row eval | Row-by-row vtrack computation | Complex vtrack expressions |
| Band query fallback | Python object enumeration | Per-object band check | 2D aggregation with band filter |

### Moderate Impact (2-5x)

| Component | Current | Bottleneck | Impact |
|-----------|---------|-----------|--------|
| ~~K-mer counting~~ | ✅ Vectorized | numpy stride_tricks (3.5x speedup) | gseq_kmer on many intervals |
| VTrack chunked eval | Python DataFrame ops | Per-chunk vtrack recomputation | Large extractions with vtracks |
| ~~DataFrame construction~~ | ✅ Column-wise | numpy array construction | Very large result sets |

### Performance Recommendations

1. **Highest ROI:** Move quad-tree reading into C++ (or use mmap + ctypes struct overlay). This unlocks faster 2D extraction, aggregation, and band queries simultaneously.
2. **Second priority:** Move gcis_decay inner loop to C++ using the existing streaming infrastructure.
3. **Third priority:** PWM scoring in C++ (already has PWMScorer.cpp infrastructure). Spatial weighting now works in Python (✅ Done).
4. ~~**Diminishing returns:** K-mer counting, liftover vectorization.~~ Both optimized 2026-03-02 (numpy vectorization).

---

## Recommendations

### Near-term (next 2 releases)

1. ~~**Fix `dim` parameter silently ignored**~~ ✅ Done 2026-03-02.
2. ~~**Add `gtrack.path`/`gintervals.path`**~~ ✅ Done 2026-03-02.
3. ~~**Warn on cross-language track variable access**~~ ✅ Done 2026-03-02.
4. ~~**Implement `gextract` file output**~~ ✅ Done 2026-03-02.

### Medium-term (next 3-6 months)

5. **C++ quad-tree reader** -- the single highest-impact performance improvement. Moderate effort (the C++ quad-tree code exists in R misha's StatQuadTreeCached.h).
6. **C++ gcis_decay** -- port the inner loop. The streaming infrastructure already handles the iteration.
7. ~~**COMPUTED 2D track type**~~ Detection + informative error done 2026-03-03. Full computation engine deferred.
8. ~~**Trans contact mirroring**~~ ✅ Done 2026-03-02.
9. ~~**PWM spatial weighting**~~ ✅ Done 2026-03-02.

### Long-term (6+ months)

10. ~~**Track arrays**~~ -- permanently NOT PLANNED (2026-03-03). No known user demand. Use R misha for track array workflows.
11. ~~**Multi-process extraction**~~ ✅ Done 2026-03-03. Two-level parallelism with comprehensive test coverage.
12. **Pattern matching standardization** -- align glob/regex behavior with R misha or document divergence prominently.

---

## Appendix: Analysis Methodology

Eight parallel analysis agents examined:
1. Full function inventory against R misha NAMESPACE exports
2. Core DB and track management implementation
3. Interval operation completeness
4. Virtual track function coverage and behavior
5. Statistics and extraction pipeline
6. Sequence analysis functions
7. 2D track subsystem
8. Python vs C++ performance profiling and algorithm analysis

Findings were deduplicated and cross-referenced. Priority assignments reflect actual user impact based on typical misha usage patterns (extraction > summary > intervals > 2D > sequence > synthesis).
