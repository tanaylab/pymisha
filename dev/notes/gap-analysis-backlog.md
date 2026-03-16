# PyMisha Gap Analysis Backlog

**Date:** 2026-03-02
**Version:** 0.1.11
**Last updated:** 2026-03-02 (post P1-P2 remaining sprint)

Priority: P0 (critical) | P1 (high) | P2 (medium) | P3 (low)
Effort: S (1-2 days) | M (3-5 days) | L (1-2 weeks) | XL (2+ weeks)
Status: ✅ DONE | 🔲 OPEN | ❌ NOT PLANNED

---

## P0 -- Critical

### GAP-001: ✅ DONE — Quad-tree reader is pure Python (20-100x slower)
- **Category:** performance
- **Effort:** L
- **Status:** ✅ Implemented 2026-03-02. New `QuadTreeReader.h/cpp` with C++ quad-tree traversal exposed via `pm_quadtree_query_stats`, `pm_quadtree_query_objects`, `pm_quadtree_query_stats_batch`. Stats 182x faster, objects 14x faster. Python fallback preserved. 25 tests in `tests/test_quadtree_cpp.py`.
- **Description:** The 2D track reader (`_quadtree.py`) uses `struct.unpack` per node for parsing StatQuadTreeCached binary format. Every 2D extraction, aggregation, and band query pays this cost. On typical Hi-C tracks (10-100M contacts), this makes 2D extraction impractical for interactive use.
- **R reference:** `StatQuadTreeCached.h` in R misha `src/` -- C++ pointer-cast traversal
- **PyMisha target:** C++ quad-tree reader exposed via `_pymisha` module, or ctypes/mmap struct overlay. The existing `_quadtree.py` becomes a fallback.

### GAP-002: ✅ DONE — gcis_decay pure Python inner loop (20-100x slower)
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. C++ bulk quad-tree extraction + numpy vectorized distance computation, binning, domain containment. Added `query_2d_track_opened_arrays()`, `_containing_interval_vec()`, `_val2bin_vec()`. 10 tests in `tests/test_cis_decay_perf.py`.
- **Description:** `gcis_decay` iterates per-bin distance calculations in Python. For Hi-C datasets with millions of contacts, this is a common analysis step that takes minutes instead of seconds.
- **R reference:** `compute_cis_decay` in R misha (C++ streaming)
- **PyMisha target:** C++ streaming function `pm_cis_decay` using existing iterator infrastructure

### GAP-003: ✅ DONE — Liftover mapping loop is pure Python (20-100x slower)
- **Category:** performance
- **Effort:** L
- **Status:** ✅ Implemented 2026-03-02. Replaced per-interval Python loop with `_map_intervals_vectorized()` using numpy prefix-max overlap search, batch `searchsorted`, flat array expansion via `np.repeat`, vectorized strand-aware coordinate transform. All 76 liftover tests pass.
- **Description:** `gintervals_liftover` and `gtrack_liftover` iterate per-interval in Python to map coordinates through chain files. Genome-wide liftover of dense intervals (e.g., 100k+ intervals) is 20-100x slower than R misha.
- **R reference:** C++ `GenomeLiftover` class
- **PyMisha target:** Vectorized numpy mapping or C++ liftover kernel

### GAP-004: ✅ DONE — 2D extraction pipeline pure Python (20-100x slower)
- **Category:** performance
- **Effort:** L
- **Status:** ✅ Implemented 2026-03-02. Batch stats query wired into `_gextract_2d_vtrack_agg()` via `pm_quadtree_query_stats_batch`. Eliminates per-interval Python→C++ overhead. Leverages GAP-001 C++ quad-tree reader.
- **Description:** Full 2D extraction pipeline (file I/O, quad-tree parse, result collection) is Python. Blocked by GAP-001; fixing GAP-001 substantially addresses this.
- **R reference:** C++ `GenomeTrack2D` + `StatQuadTreeCached`
- **PyMisha target:** C++ 2D extraction function or significant acceleration via GAP-001

---

## P1 -- High

### GAP-005: ✅ DONE — vtrack_iterator `dim` parameter silently ignored
- **Category:** behavioral-divergence
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Added `_project_intervals_by_dim()` to vtracks.py; modified `_compute_vtrack_values` to project 2D→1D when dim is set; updated `_gextract_2d` in extract.py to route dim-projected vtracks. 19 tests in `tests/test_dim_param.py`.
- **Description:** `gvtrack_iterator` accepts a `dim` parameter for 2D-to-1D projection but silently ignores it. Users attempting to project a 2D track onto one dimension get incorrect results with no error or warning. This is a correctness bug.
- **R reference:** `gvtrack.iterator(dim=1)` or `dim=2` projects 2D track to that dimension
- **PyMisha target:** Implement dim projection in the vtrack iterator, or raise NotImplementedError with a clear message

### GAP-006: ✅ DONE — Track variable cross-language incompatibility
- **Category:** behavioral-divergence
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Added R serialization magic byte detection (ASCII, binary, XDR, gzip-compressed RDS) in `gtrack_var_get`. Raises informative error with guidance. 6 tests in `tests/test_gtrack_var.py`.
- **Description:** `gtrack_var_set` writes Python pickle; `gtrack_var_get` reads pickle. R misha uses R's `serialize()`. When sharing databases, track variables written by one language cannot be read by the other. No error is raised -- the reader either returns garbage or crashes.
- **R reference:** R `serialize()`/`unserialize()` format
- **PyMisha target:** (a) Detect R-serialized variables and raise an informative error. (b) Optionally, support reading R-serialized variables via `pyreadr` or a minimal RDS parser. (c) Document the limitation.

### GAP-007: ✅ DONE — COMPUTED 2D track type detection + informative error
- **Category:** missing-feature
- **Effort:** XL (detection: S; full implementation deferred)
- **Description:** R misha supports COMPUTED tracks whose values are derived on-the-fly from other tracks using normalization models (PotentialComputer2D, TechnicalComputer2D). These are core to Hi-C normalization pipelines (observed/expected ratios, ICE normalization). Full computation engine is deferred. Detection is implemented: `_check_computed_tracks()` raises `NotImplementedError` with an informative message early in gextract, gsummary, gquantiles, gdist, gscreen, glookup, gcor.
- **R reference:** `PotentialComputer2D.h`, `TechnicalComputer2D.h` in R misha src
- **PyMisha target:** C++ computation engine for COMPUTED tracks, or Python equivalent with acceptable performance (deferred; detection + informative error done)

### GAP-008: ✅ DONE — gseq_pwm spatial weighting raises NotImplementedError
- **Category:** missing-feature
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Log-space spatial factor application matching R misha C++ semantics. Removed NotImplementedError. 21 tests in `tests/test_gseq_pwm_spatial.py`.
- **Description:** `gseq_pwm` with `spat_factor`/`spat_bin` parameters raises `NotImplementedError`. PWM scoring without spatial weighting works. Spatial weighting modulates PWM scores by position within the interval.
- **R reference:** `GenomeSeqScorer` with spatial weighting in R misha C++
- **PyMisha target:** Implement spatial weighting in Python (using existing PWMScorer.cpp infrastructure) or C++

### GAP-009: ✅ DONE — PWM scoring pure Python (10-30x slower)
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Numpy stride-tricks vectorized PWM scoring (`_pwm_score_batch_vectorized`): sliding window via `as_strided`, fancy indexing into log_pssm, vectorized base encoding tables. 17.6x speedup (39ms vs 692ms for 500 seqs × 200bp). All 4 modes (lse/max/pos/count), spatial weighting, neutral chars.
- **Description:** `gseq_pwm` scores each sequence base-by-base in Python. C++ infrastructure exists (`PWMScorer.cpp`, `GenomeSeqScorer.cpp`) but is not wired to the per-sequence scoring loop.
- **R reference:** C++ `GenomeSeqScorer::score` in R misha
- **PyMisha target:** C++ `pm_gseq_pwm` function or expose `PWMScorer` scoring to Python via `_pymisha`

### GAP-010: ✅ DONE — VTrack iterrows for per-row computation (10-100x slower)
- **Category:** algorithm
- **Effort:** L
- **Status:** ✅ Implemented 2026-03-02. Vectorized 4 per-row loops: `_build_unmasked_segments` (numpy array ops for no-mask path), overlap matching (numpy max/min broadcasting), nearest fallback (vectorized distance), `base_starts` extraction (numpy.unique+grouping), `_filter_key` (list comprehension).
- **Description:** Several vtrack computation paths use `DataFrame.iterrows()` or per-row Python loops for vtrack evaluation. On large extractions (100k+ rows), this dominates runtime.
- **R reference:** C++ integrated vtrack evaluation in the scanner loop
- **PyMisha target:** Vectorized numpy operations for common vtrack functions; batch evaluation instead of per-row

### GAP-011: ✅ DONE — Band query falls back to full object enumeration (10-100x slower)
- **Category:** algorithm
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. C++ band-filtered quad-tree traversal via `pm_quadtree_query_objects_band`. Batch extraction in `_gextract_2d_vtrack_agg` for band queries using C++ object enumeration. Leverages GAP-001 C++ reader. Tests in `tests/test_2d_band_cpp.py`.
- **Description:** 2D aggregation with band filters cannot use pre-computed node stats (stats don't account for band constraints), so every object is enumerated individually. For large 2D tracks with narrow bands, this is extremely slow.
- **R reference:** R misha has the same limitation but is faster due to C++ enumeration
- **PyMisha target:** C++ enumeration (blocked by GAP-001) or band-aware stat pre-computation

---

## P2 -- Medium

### GAP-012: ✅ DONE — gextract missing file/intervals_set_out output
- **Category:** parameter-gap
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Added `file` param (streaming TSV write, returns None) and `intervals_set_out` param (saves result intervals as named set). Works for 1D and 2D. 10 tests in `tests/test_gextract_output.py`.
- **Description:** R misha's `gextract` can write results to a file or named interval set. PyMisha always returns a DataFrame. For extractions exceeding available memory, file output is necessary.
- **R reference:** `gextract(file=..., intervals.set.out=...)`
- **PyMisha target:** Add `file` parameter (write CSV/TSV streaming) and `intervals_set_out` parameter

### GAP-013: ✅ DONE — gtrack.path / gintervals.path missing
- **Category:** missing-feature
- **Effort:** S
- **Status:** ✅ Implemented 2026-03-02. `gtrack_path(track)` wraps `_pymisha.pm_track_path`, `gintervals_path(name)` wraps `_pymisha.pm_intervals_path`. Exported from `__init__.py`. 13 tests in `tests/test_path_functions.py`.
- **Description:** Convenience functions returning the filesystem path to a track or interval set directory. Frequently used in scripts that manipulate raw track files.
- **R reference:** `gtrack.path(track)`, `gintervals.path(name)`
- **PyMisha target:** Thin wrappers around existing `pm_track_path` / `pm_intervals_path`

### GAP-014: ✅ DONE — 2D set operations (union/intersect) missing
- **Category:** missing-feature
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. `gintervals_2d_intersect` (vectorized numpy broadcasting for pairwise rectangle intersection) and `gintervals_2d_union` (concatenate+sort). Exported from `__init__.py`. 31 tests in `tests/test_2d_set_ops.py`.
- **Description:** `gintervals_2d_union` and `gintervals_2d_intersect` are not implemented. The 1D versions exist. 2D operations require rectangle overlap logic.
- **R reference:** R misha implements via sweep-line or brute-force rectangle intersection
- **PyMisha target:** Implement 2D union and intersection, likely via sorted-rectangle sweep

### GAP-015: ✅ DONE — Trans contact mirroring writes only canonical direction
- **Category:** behavioral-divergence
- **Effort:** S
- **Status:** ✅ Implemented 2026-03-02. `gtrack_2d_import_contacts` now writes both chrA-chrB AND chrB-chrA files for trans contacts with swapped coordinates. 8 tests in `tests/test_trans_mirroring.py`.
- **Description:** `gtrack_2d_import_contacts` writes trans contacts only as chrA-chrB (A<=B). R misha writes both chrA-chrB and chrB-chrA files. Queries for the non-canonical direction return empty results.
- **R reference:** R misha's 2D track import writes symmetric trans files
- **PyMisha target:** Write both directions for trans contacts

### GAP-016: ✅ DONE — gtrack_ls / gintervals_ls regex vs glob pattern divergence
- **Category:** behavioral-divergence
- **Effort:** S
- **Status:** ✅ Verified 2026-03-02. Investigation showed both `gtrack_ls` and `gintervals_ls` already use `re.search()` (regex), matching R misha behavior. No changes needed — gap was a false positive.
- **Description:** R misha uses regex patterns for `gtrack.ls(pattern=...)` and `gintervals.ls(pattern=...)`. PyMisha uses glob patterns. Scripts migrating from R need pattern translation.
- **R reference:** `gtrack.ls(pattern="^chip_")` (regex)
- **PyMisha target:** Accept regex patterns (with optional `regex=True` flag) or document the difference prominently

### GAP-017: ✅ DONE — gbins_summary / gbins_quantiles Python fallback
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Added `_extract_values_direct()` bypass, vectorized `gbins_summary` with `numpy.bincount`, replaced `itertools.product` in `gbins_quantiles` with sort-based grouping. 1.4-1.5x speedup. 4 tests in `tests/test_gbins.py`.
- **Description:** These functions wrap `gsummary`/`gquantiles` in Python with binning logic, rather than using a single C++ pass. Overhead is 2-5x since the underlying extraction is C++.
- **R reference:** Single C++ pass with binning in R misha
- **PyMisha target:** C++ `pm_gbins_summary` / `pm_gbins_quantiles` or optimized Python binning

### GAP-018: ✅ DONE — 2D vtrack missing non-aggregation functions
- **Category:** missing-feature
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Added 6 functions: exists, size, first, last, sample, global.percentile. New extraction paths `_gextract_2d_vtrack_objects` and `_gextract_2d_vtrack_global_percentile` in extract.py. 53 tests in `tests/test_2d_vtrack_nonag.py`.
- **Description:** 2D vtracks only support aggregation functions (area, weighted.sum, min, max, avg). Missing: exists, size, first, last, sample, all position functions, global.percentile.
- **R reference:** R misha supports full function set on 2D tracks
- **PyMisha target:** Extend `_gextract_2d_vtrack_agg` to handle non-aggregation functions

### GAP-019: ✅ DONE — Python vtrack fallback makes 2+ passes
- **Category:** performance
- **Effort:** L
- **Status:** ✅ Implemented 2026-03-02. Pre-compute ALL vtrack values once for the full interval set before chunking, then slice per chunk. Eliminates N redundant vtrack recomputation passes where N = n_rows/eval_buf_size.
- **Description:** Mixed C++ + vtrack expressions require separate passes: one C++ extraction, then Python vtrack evaluation per chunk. R misha evaluates everything in a single C++ scan.
- **R reference:** Integrated C++ scanner with vtrack evaluation
- **PyMisha target:** Register vtrack evaluation callbacks in C++ scanner, or batch-vectorize the Python pass

### GAP-020: ✅ DONE — Single-chunk quad-tree writer can OOM
- **Category:** algorithm
- **Effort:** L
- **Status:** ✅ Implemented 2026-03-02. Multi-chunk serialization matching R misha's `analyze_n_serialize_subtree` algorithm. `_count_subtree_bytes()` for size estimation, `_analyze_and_serialize()` for bottom-up chunking. Negative kid pointers for cross-chunk references. `write_2d_track_file()` accepts `chunk_size` parameter. Single-chunk and multi-chunk produce identical query results.
- **Description:** `_quadtree.py` builds the entire quad-tree in memory as a single chunk. For very large 2D tracks (>100M contacts), this exhausts memory. R misha uses multi-chunk streaming.
- **R reference:** Multi-chunk `StatQuadTreeCached` writer in R misha
- **PyMisha target:** Multi-chunk streaming writer, or memory-mapped builder

### GAP-021: ✅ DONE — gintervals_mapply Python inner loop
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Replaced per-interval iterrows+gextract loop with single batch gextract call per expression. Pre-grouped extracted data by intervalID using numpy break-point detection for O(1) lookup. Eliminated N separate C++ extraction calls.
- **Description:** `gintervals_mapply` dispatches a user-provided Python function per interval group. The overhead is in the per-group dispatch, not the function itself. This is inherent to the design but could be optimized for common patterns (e.g., summary statistics).
- **R reference:** R misha's `gintervals.mapply` with C++ dispatch
- **PyMisha target:** Optimize common patterns; vectorize group dispatch where possible

### GAP-022: ✅ DONE — Multitask / multi-process extraction
- **Category:** missing-feature
- **Effort:** XL
- **Status:** ✅ Implemented. Two-level parallelism: (1) Python-level `_parallel_extract()` in extract.py using `multiprocessing.Pool` with fork context, splits intervals by chromosome for gextract. (2) C++ level fork/FIFO multitask via `PyMisha::prepare4multitasking()` / `PyMisha::launch_process()` for gextract, gsummary, gquantiles, gdist, gscreen, glookup, gcor. `gmax_processes()` getter/setter in `_shared.py`. Virtual tracks correctly skip Python-level parallelism. Code reviewed 2026-03-03; 18 parity+edge-case tests in `tests/test_multiprocess_review.py`.
- **Description:** R misha's `gmax.processes` enables multi-process extraction and summary. PyMisha uses a single process. For large genomes, this limits throughput by the number of chromosomes that could be processed in parallel.
- **R reference:** Fork-based multiprocess via `gmax.processes` option
- **PyMisha target:** Python multiprocessing or threading for per-chromosome parallelism

### GAP-023: ✅ DONE — gintervals_force_range column preservation
- **Category:** behavioral-divergence
- **Effort:** S
- **Status:** ✅ Implemented 2026-03-02. Fixed to use `iloc[keep].copy()` preserving all columns. Handles both 1D and 2D intervals. 8 tests in `tests/test_force_range_extra_cols.py`.
- **Description:** Extra columns beyond chrom/start/end may be dropped when clipping intervals to chromosome range. R misha preserves all columns.
- **R reference:** `gintervals.force_range` preserves extra columns
- **PyMisha target:** Ensure non-coordinate columns survive the clipping operation

### GAP-024: ✅ DONE — Bigset transparent iteration
- **Category:** parameter-gap
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Added `_maybe_load_intervals_set()` helper across 21 functions in 8 modules (extract, summary, intervals, liftover, lookup, sequence, analysis, gsynth). 22 tests in `tests/test_named_intervals.py`.
- **Description:** Named bigset interval sets are not transparently iterable in all function contexts. Some functions require explicit loading first, while R misha handles them seamlessly.
- **R reference:** Bigsets are transparently usable wherever interval sets are accepted
- **PyMisha target:** Auto-detect and handle bigsets in all interval-accepting functions

### GAP-025: ❌ NOT PLANNED — Track arrays subsystem
- **Category:** missing-feature
- **Effort:** XL

> **Decision (2026-03-03): Track arrays will not be implemented.** This is a rarely-used R misha feature with no known pymisha user demand. The five functions (gtrack_array_extract, gtrack_array_import, gtrack_array_get_colnames, gtrack_array_set_colnames, gvtrack_array_slice) are permanently excluded from scope. If needed, users should use R misha for track array workflows.

- **Description:** Five functions (`gtrack.array.extract`, `gtrack.array.import`, `gtrack.array.get_colnames`, `gtrack.array.set_colnames`, `gvtrack.array.slice`) for multi-column-per-bin track data.
- **R reference:** `gtrack.array.*` functions in R misha
- **PyMisha target:** Not planned. Use R misha for track array workflows.

### GAP-026: ✅ DONE — K-mer counting/distance pure Python
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Numpy stride_tricks-based k-mer hashing, cached lookup table for k-mer strings, vectorized batch counting. 3.5x average speedup. 15 tests in `tests/test_gseq_kmer.py`.
- **Description:** `gseq_kmer` and `gseq_kmer_dist` encode and count k-mers in Python. 2-5x slower than C++. `KmerCounter.cpp` exists but is not wired to these functions.
- **R reference:** C++ k-mer counting in R misha
- **PyMisha target:** Wire `KmerCounter.cpp` to Python k-mer functions via `_pymisha`

### GAP-027: ✅ DONE — Liftover overlap policy pure Python (5-20x slower)
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. Vectorized 7 functions: `_handle_tgt_overlaps_auto` (coverage matrix + cumsum merging), `_discard_overlapping_intervals`, `_handle_tgt_overlaps_agg`, src/tgt overlap error policies, `_interval_union_length`, `_canonic_merge`. All existing tests pass.
- **Description:** Overlap resolution policies (auto_score, auto_first, auto_longer) in liftover are implemented with Python sorting and filtering. On large chain files with many overlapping mappings, this is slow.
- **R reference:** C++ overlap resolution in R misha
- **PyMisha target:** Vectorized numpy overlap resolution or C++ implementation

### GAP-028: ✅ DONE — 2D iterator abstraction missing
- **Category:** missing-feature
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-02. `giterator_intervals_2d` generator function in extract.py. Yields one DataFrame per input interval. Supports band filtering, virtual tracks, multiple expressions, custom colnames. Exported from `__init__.py`. 21 tests in `tests/test_2d_iterator.py`.
- **Description:** No `giterator`-style streaming interface for 2D data. 2D extraction works but lacks the per-interval callback pattern that 1D iterators provide.
- **R reference:** `giterator` with 2D intervals in R misha
- **PyMisha target:** 2D iterator class or generator-based streaming interface

### GAP-029: ✅ DONE — DataFrame construction via list-of-dicts
- **Category:** performance
- **Effort:** S
- **Status:** ✅ Implemented 2026-03-02. Replaced 5 list-of-dicts patterns in `liftover.py` (4 sites) and `intervals.py` (1 site) with column-wise numpy construction. All existing tests pass.
- **Description:** Some code paths build DataFrames by appending to a list of dicts, then calling `pd.DataFrame(rows)`. For large result sets, column-wise construction from numpy arrays is 2-5x faster.
- **R reference:** N/A (Python-specific)
- **PyMisha target:** Replace list-of-dicts patterns with column-wise numpy array construction

### GAP-030: ✅ DONE — gtrack_create_pwm_energy Python workaround
- **Category:** performance
- **Effort:** M
- **Status:** ✅ Addressed 2026-03-02. Already uses C++ scoring via pm_vtrack_compute. GAP-019's pre-computed vtrack optimization eliminates per-chunk recomputation overhead. Direct gseq_pwm bypass tested but diverges numerically from C++ path; vtrack path kept for R misha compatibility.
- **Description:** `gtrack_create_pwm_energy` uses a Python-level workaround instead of C++ PWM scoring. Related to GAP-009 (PWM performance).
- **R reference:** C++ PWM energy track creation
- **PyMisha target:** Use C++ PWM scoring path once GAP-009 is resolved

---

## P3 -- Low

### GAP-031: ✅ DONE — gcompute_strands_autocorr missing
- **Category:** missing-feature
- **Effort:** M
- **Status:** ✅ Implemented 2026-03-03. Pure Python implementation in `pymisha/analysis.py` matching R misha's C++ `GenomeComputeStrandAutocorr.cpp`. Parses tab-delimited mapped reads files with configurable column layout, builds binned forward/reverse strand coverage arrays (capped at 10), computes Pearson cross-correlation at each bin offset in [-maxread/binsize, maxread/binsize). Returns (stats_dict, bins_DataFrame). 13 tests in `tests/test_strands_autocorr.py`.
- **Description:** Strand autocorrelation function for nascent transcription analysis. Specialized use case.
- **R reference:** `gcompute_strands_autocorr` in R misha
- **PyMisha target:** ~~Implement if requested~~ Implemented

### GAP-032: gtrack.convert missing
- **Category:** missing-feature
- **Effort:** M
- **Description:** Track format migration (e.g., sparse to dense, old format to indexed). Rarely needed in practice since `gdb_convert_to_indexed` handles the common migration.
- **R reference:** `gtrack.convert` in R misha
- **PyMisha target:** Implement if requested

### GAP-033: ~~gtrack.dbs / gintervals.dbs missing~~ ✅ DONE
- **Category:** missing-feature
- **Effort:** S
- **Description:** List tracks or intervals across multiple databases. Implemented as `gtrack_dbs()` and `gintervals_dbs()` with dict and DataFrame output modes. Searches current DB root + loaded dataset roots.
- **R reference:** `gtrack.dbs()`, `gintervals.dbs()`
- **PyMisha:** `pymisha/tracks.py::gtrack_dbs()`, `pymisha/intervals.py::gintervals_dbs()`, 10 tests in `tests/test_dbs_functions.py`

### GAP-034: ~~gdb.mark_cache_dirty missing~~ ✅ DONE
- **Category:** missing-feature
- **Effort:** S
- **Description:** Explicit cache invalidation. PyMisha uses `gdb_reload()` instead, which is a superset of this functionality.
- **R reference:** `gdb.mark_cache_dirty()`
- **PyMisha target:** Implemented as `gdb_mark_cache_dirty()` — delegates to `gdb_reload()` (full C++ rescan). 6 tests.

### GAP-035: ~~gdataset.example_path missing~~ ✅ DONE
- **Category:** missing-feature
- **Effort:** S
- **Description:** Return path to example dataset. PyMisha has `gdb_examples_path` for the DB but not for individual datasets.
- **R reference:** `gdataset.example_path()`
- **PyMisha target:** Implemented as `gdataset_example_path()` in `pymisha/dataset.py`. Creates an example dataset on the fly (mirrors R misha semantics). 8 tests in `tests/test_gdataset_example.py`.

### GAP-036: ~~grevcomp standalone function missing~~ ✅ DONE
- **Category:** missing-feature
- **Effort:** S
- **Description:** Standalone reverse-complement for a DNA string. Implemented as `grevcomp()` delegating to `gseq_revcomp`. Exported from `pymisha/__init__.py`. 10 tests in `tests/test_grevcomp.py`.
- **R reference:** `grevcomp(seq)` in R misha
- **PyMisha target:** One-line wrapper using existing `_COMPLEMENT` table

### GAP-037: gcluster.run missing (deferred)
- **Category:** missing-feature
- **Effort:** L
- **Description:** Submit misha jobs to cluster schedulers. R-specific (uses R's cluster packages). Not applicable to Python workflows which use their own job submission tools.
- **R reference:** `gcluster.run` in R misha
- **PyMisha target:** Not planned. Users should use Python-native job submission.

### GAP-038: gwget missing (deferred)
- **Category:** missing-feature
- **Effort:** S
- **Description:** Download track data from URL. Trivial to implement with `urllib`/`requests` but low priority.
- **R reference:** `gwget` in R misha
- **PyMisha target:** Implement if requested

### GAP-039: gintervals_annotate tie_method parameter -- DONE
- **Category:** parameter-gap
- **Effort:** S
- **Status:** DONE
- **Description:** The `tie_method` parameter for breaking ties when multiple annotations overlap is now implemented. Supports `"first"` (default), `"min.start"`, and `"min.end"` strategies, matching R misha's `gintervals.annotate(tie.method=...)`.
- **R reference:** `gintervals.annotate(tie.method=...)`
- **PyMisha target:** ~~Add tie_method parameter with supported strategies~~ Implemented

### GAP-040: ✅ DONE — gsynth_sample bin_merge parameter
- **Category:** parameter-gap
- **Effort:** S
- **Status:** ✅ Implemented. `bin_merge` parameter fully supported in `gsynth_sample` (`pymisha/gsynth.py`). Controls how adjacent bins are merged during sampling. 8+ tests in `TestBinMerge` class in `tests/test_gsynth.py`.
- **Description:** The `bin_merge` parameter in `gsynth_sample` controls how adjacent bins are merged during sampling.
- **R reference:** `gsynth.sample(bin.merge=...)`
- **PyMisha target:** ~~Add bin_merge parameter~~ Implemented

### GAP-041: gtrack.2d.import multi-file support — ✅ DONE
- **Category:** parameter-gap
- **Effort:** S
- **Description:** R misha's `gtrack.2d.import` can accept multiple input files. PyMisha only handles a single file per call.
- **R reference:** `gtrack.2d.import(files=c("a.txt", "b.txt"))`
- **PyMisha target:** Accept list of files, concatenate before import
- **Status:** Implemented 2026-03-03. `gtrack_2d_import` now accepts `str` or `list[str]`. 6 tests.

### GAP-042: Various intervals_set_out parameters
- **Category:** parameter-gap
- **Effort:** M
- **Description:** Several summary/extraction functions in R misha have an `intervals.set.out` parameter to save result intervals as a named set. This is consistently missing across PyMisha.
- **R reference:** `intervals.set.out` parameter on gsummary, gquantiles, gdist, etc.
- **PyMisha target:** Add intervals_set_out parameter to relevant functions
- **Status:** ✅ DONE. Already implemented on all applicable functions: `gextract`, `gintervals_summary`, `gintervals_quantiles`, `gpartition`, `gscreen`, `glookup`, `gintervals_2d_band_intersect`, `gintervals_force_range`, `gintervals_union`, `gintervals_intersect`, `gintervals_diff`, `gintervals_rbind`, `gintervals_normalize`, `gintervals_annotate`, `gneighbors`. **Not applicable to `gdist`**: R misha's `gdist` does not have `intervals.set.out` because it returns aggregate distribution counts (N-dimensional array or DataFrame with bin labels + counts), not per-interval results with chrom/start/end columns.

---

## Summary Statistics

| Priority | Total | Done | Not Planned | Open | Estimated Remaining Effort |
|----------|-------|------|-------------|------|---------------------------|
| P0 (critical) | 4 | 4 | 0 | 0 | — |
| P1 (high) | 7 | 7 | 0 | 0 | — |
| P2 (medium) | 19 | 18 | 1 | 0 | — |
| P3 (low) | 12 | 5 | 0 | 7 | 3-5 weeks |
| **Total** | **42** | **34** | **1** | **7** | **3-5 weeks** |

### Completed (2026-03-02 — 2D feature-complete sprint)

| GAP | Feature | Tests |
|-----|---------|-------|
| GAP-005 | `dim` parameter fix (correctness bug) | 19 |
| GAP-014 | `gintervals_2d_intersect` / `gintervals_2d_union` | 31 |
| GAP-015 | Trans contact mirroring (symmetric writes) | 8 |
| GAP-018 | 2D vtrack non-agg functions (exists, size, first, last, sample, global.percentile) | 53 |
| GAP-028 | `giterator_intervals_2d` (2D streaming iterator) | 21 |
| **Total** | **5 features** | **132 new tests** |

Test suite after sprint: **1836 passed, 0 failed, 25 skipped**

### Completed (2026-03-02 — P0-P2 sprint)

| GAP | Feature | Tests |
|-----|---------|-------|
| GAP-006 | R-serialization detection in `gtrack_var_get` | 6 |
| GAP-008 | PWM spatial weighting (`spat_factor`/`spat_bin`) | 21 |
| GAP-012 | `gextract` file output + `intervals_set_out` | 10 |
| GAP-013 | `gtrack_path` / `gintervals_path` | 13 |
| GAP-016 | Regex patterns already implemented (false positive) | 0 |
| GAP-017 | `gbins_summary` / `gbins_quantiles` vectorized | 4 |
| GAP-023 | `gintervals_force_range` column preservation | 8 |
| GAP-024 | Bigset transparent iteration (21 functions) | 22 |
| GAP-026 | K-mer vectorization (numpy stride_tricks) | 15 |
| GAP-027 | Liftover overlap resolution vectorized | 0 |
| GAP-029 | DataFrame construction column-wise | 0 |
| **Total** | **11 items** | **99 new tests** |

Test suite after sprint: **1935 passed, 0 failed, 25 skipped**

### Completed (2026-03-02 — P0 C++ performance sprint)

| GAP | Feature | Tests |
|-----|---------|-------|
| GAP-001 | C++ quad-tree reader (182x stats, 14x objects) | 25 |
| GAP-002 | gcis_decay vectorized (C++ bulk extraction + numpy) | 10 |
| GAP-003 | Liftover mapping vectorized (prefix-max + batch searchsorted) | 0 |
| GAP-004 | 2D extraction batch stats wiring | 0 |
| **Total** | **4 items** | **35 new tests** |

Test suite after sprint: **1970 passed, 0 failed, 25 skipped**

### Completed (2026-03-02 — P1-P2 remaining sprint)

| GAP | Feature | Tests |
|-----|---------|-------|
| GAP-011 | C++ band-filtered quad-tree query | new |
| GAP-009 | PWM numpy vectorization (17.6x speedup) | 0 |
| GAP-010 | VTrack per-row vectorization (4 loops) | 0 |
| GAP-019 | Pre-computed vtrack values (eliminate per-chunk recomputation) | 0 |
| GAP-020 | Multi-chunk quad-tree writer | 0 |
| GAP-021 | Batch gintervals_mapply extraction | 0 |
| GAP-030 | PWM energy already uses C++ + GAP-019 optimization | 0 |
| **Total** | **7 items** | **0 new tests** |

Test suite after sprint: **1993 passed, 0 failed, 25 skipped**

## Suggested Sprint Groupings

**Sprint 1 -- Correctness fixes:** ALL DONE
~~GAP-005 (dim parameter)~~✅, ~~GAP-006 (track var detection)~~✅, ~~GAP-015 (trans mirroring)~~✅, ~~GAP-023 (force_range columns)~~✅

**Sprint 2 -- Quick wins:** ALL DONE
~~GAP-013 (path functions)~~✅, ~~GAP-016 (regex patterns — already implemented)~~✅, ~~GAP-029 (DataFrame construction)~~✅, ~~GAP-036 (grevcomp, P3)~~✅

**Sprint 3 -- 2D performance:** ALL DONE
~~GAP-001 (C++ quad-tree reader)~~✅, ~~GAP-004 (2D extraction)~~✅, GAP-011 (band enumeration, P1)

**Sprint 4 -- Core performance:**
~~GAP-002 (gcis_decay)~~✅, GAP-009 (PWM C++), ~~GAP-008 (PWM spatial)~~✅, GAP-010 (vtrack vectorize), ~~GAP-003 (liftover mapping)~~✅

**Sprint 5 -- Feature completeness:** ALL DONE
~~GAP-012 (gextract file output)~~✅, ~~GAP-014 (2D set ops)~~✅, ~~GAP-018 (2D vtrack functions)~~✅, ~~GAP-024 (bigset iteration)~~✅

**Sprint 6 -- Performance:** ALL DONE
~~GAP-017 (gbins optimization)~~✅, ~~GAP-026 (k-mer vectorization)~~✅, ~~GAP-027 (liftover vectorization)~~✅, ~~GAP-028 (2D iterator)~~✅

**Deferred / Not Planned:**
~~GAP-007 (COMPUTED tracks)~~✅ (detection + informative error; full implementation deferred), ~~GAP-025 (track arrays)~~❌ NOT PLANNED (permanently excluded; no user demand)

### Completed (2026-03-03 -- GAP-022 review)

| GAP | Feature | Tests |
|-----|---------|-------|
| GAP-022 | Multi-process extraction (code review + comprehensive tests) | 18 |
| **Total** | **1 item** | **18 new tests** |

Test suite after review: **2041 passed, 0 failed, 25 skipped**

### Completed (2026-03-03 -- GAP-031)

| GAP | Feature | Tests |
|-----|---------|-------|
| GAP-031 | `gcompute_strands_autocorr` (strand cross-correlation from mapped reads) | 13 |
| **Total** | **1 item** | **13 new tests** |

Test suite after sprint: **2103 passed, 0 failed, 25 skipped**
