# Changelog

## v0.1.25 (2026-04-12)

### Maintenance
- **Public API cleanup:** Removed 10 underscore-prefixed internal symbols from `__all__`; added `noqa: F401` for the C++ bridge imports.
- **C++ memory safety:** Replaced `new char[]`/`delete[]` with `std::vector<char>` in indexed format writers. Replaced manual `new`/`delete` with `std::unique_ptr` in PMWilcox, GenomeSeqFetch, PMTrackExpressionVars. Added `snprintf` bounds checking for shared-memory error buffer.
- **Compiler warnings:** Removed `-Wno-switch` and `-Wno-strict-aliasing` suppressions from setup.py. Fixed misleading indentation in GenomeTrackFixedBin.cpp and uninitialized variable in PMTrackCreate.cpp. Zero warnings from project sources.
- **Coverage reporting:** Added `[tool.coverage.run]`/`[tool.coverage.report]` to pyproject.toml; added `--cov=pymisha --cov-report=term-missing` to Linux CI.

### Documentation
- **Thread safety:** Added concurrency constraints section to README.md and quickstart guide (single-threaded, one DB per process, global CONFIG).
- **`_PMLOCALS` comment:** Expanded explanation of the C++ namespace bridge at the end of `__init__.py`.

## v0.1.24 (2026-04-12)

### Features
- **Comprehensive type annotations:** Added inline type annotations to all function signatures across all 26 modules (~500 functions). Created `py.typed` marker (PEP 561) and `pymisha/_types.py` with shared type aliases (`Intervals`, `PMDataFrame`, `Iterator`, `TrackExpr`, `Chroms`). Full mypy pass with zero errors.

### Maintenance
- **Dynamic `__version__`:** Replaced hardcoded version string with `importlib.metadata.version()` to stay in sync with `pyproject.toml` automatically.
- **mypy in CI:** Added type checking step to GitHub Actions workflow.
- **mypy config:** Enabled `check_untyped_defs` and `warn_unused_ignores` in `pyproject.toml`.

## v0.1.23 (2026-04-04)

### Performance
- **LUT-based DNA encoding:** Replaced all switch-statement character encoding in DnaPSSM with static 256-entry lookup tables (BASE_ENCODE, COMPLEMENT_ENCODE, NEUTRAL_CHAR), eliminating branch misprediction overhead in PWM scoring hot loops. Fixed latent `case 'h'` bug in `integrate_energy` reverse-complement path. Ported from R misha PR #83.
- **DP buffer reuse:** PWMEditDistanceScorer `compute_with_indels()` now reuses a persistent buffer instead of per-window heap allocation, reducing malloc/free overhead for indel-mode edit distance scoring.

## v0.1.22 (2026-04-03)

### Features
- **Direction parameter for edit distance:** New `direction="above"/"below"` parameter in `gseq_pwm_edits()`, virtual tracks (`pwm.edit_distance`, `pwm.edit_distance.lse`), and all edit distance scorers. `"above"` (default) finds min edits to reach threshold; `"below"` finds min edits to disrupt score below threshold. Ported from R misha.
- **Variable Markov order k:** `gsynth_train()` now accepts `k=1..10` (default 5) for configurable Markov order. Includes format v2 for `.gsm` files with v1 backward compatibility, non-integer k rejection, and full propagation through parallel train/sample.

### Performance
- **Edit distance 1.5x speedup** (subs-only genome scans on hg38): IC-ordered column processing, score-aware pigeonhole viable tables, sliding-window N-count skip, persistent allocation in heuristic, inline pigeonhole prefilter. Benchmarked on hg38 chr1 (249Mb): 10.6s → 7.0s.

### Testing
- 98 new tests (2551 total): 54 direction=below tests (subs, indels, LSE, vtracks, edge cases), 44 variable-k tests (k=1/3/5/7/10 training, validation, save/load, parallel).

## v0.1.21 (2026-03-29)

### Performance
- **C++ data structure optimizations (ported from R misha):** Bitmask replacement for `m_functions` (`vector<bool>` → `uint32_t`), `vector<uint8_t>` in PWMEditDistanceScorer, golden-ratio multiplicative hashing, StreamPercentiler template comparators for compiler inlining.
- **C++ I/O optimizations:** MmapFile RAII utility, coalesced packed-struct I/O in TrackIndex2D/TrackIndex/GenomeIndex/PMTrackIndexedFormat, mmap-backed fixed-bin reads in GenomeTrackFixedBin.
- **C++ safety:** SegmentFinder overflow-safe midpoint and iterative destructor.
- **PyMisha-specific C++ optimizations:** CHROM string interning at scanner init, NumPy array reuse across evaluation batches, sparse interval vector adaptive pre-allocation.
- **Direct-to-NumPy accumulation:** New PMDirectAccumulator writes gextract scan results directly into pre-allocated NumPy arrays, bypassing intermediate C++ vector storage.
- **Python vectorization:** Replaced `iterrows()`/`itertuples()` with vectorized groupby in analysis.py/extract.py/vtracks.py, NumPy advanced indexing in `_compute_value_df_vtrack()`, removed 7 unnecessary DataFrame `.copy()` calls, batch filter resolution with `pd.concat`.
- Real hg38 benchmarks: **gsummary +87%**, sparse extract +10%, dense extract +2-4%.

## v0.1.20 (2026-03-25)

### Features
- **PWM edit distance optimizations:** Ported all R misha performance enhancements — flat PSSM lookup tables, precomputed base index arrays, pigeonhole pre-filter, suffix-bound early-abandon, quick deficit check, and specialized exact solvers for `max_indels=1` and `max_indels=2`. Mandatory edit handling fixes for log-zero PSSM entries.
- **`gseq_pwm_edits()` indel support:** New `max_indels` parameter enables insertion/deletion detection via banded 3D Needleman-Wunsch DP with alignment traceback. Output includes `edit_type` column (`"sub"`, `"ins"`, `"del"`) and gap characters in `window_seq`/`mutated_seq`.
- **Intra-chromosome parallelization:** Large chromosomes are now split across multiple workers by genomic range with bin-aligned boundaries (ported from R misha's `split_intervals_1d_by_range`). Affects `gscreen`, `gextract`, `gsummary`, `gquantiles`, `gdist`, `gpartition`, `gcor`. Minimum 50,000 bins per worker.
- **Vtrack C++ scanner integration:** Virtual track expressions (PWM, edit_distance, kmer, masked, and value-based aggregations like avg/sum/min/max) now evaluate inline in the C++ scanner loop instead of falling back to a serial Python eval path. This enables full fork/FIFO parallelism and intra-chromosome splitting for vtrack-heavy workloads.

### Performance
- Edit distance genome scans: **19x speedup** with parallelism (76.5s → 4.0s serial → parallel on chr1:0-10Mb, CTCF D=2 K=2).
- Single-threaded edit distance throughput matches R misha (8.78 vs 7.32 sec/Mb for D=2 K=2).
- Hit counts match R misha exactly across all benchmark configurations.

### Testing
- 72 new tests (2453 total): 55 edit distance tests (adversarial, indel, optimization consistency), 7 intra-chromosome parallelization tests, 10 vtrack C++ path tests.
- Adversarial test suite validates specialized vs generic DP solvers, exhaustive L=2 enumeration, mandatory edit handling, and known bug regressions.

## v0.1.19 (2026-03-17)

### Features
- **DataFrame intervals as iterator:** `gextract()`, `gscreen()`, `gsummary()`, `gquantiles()`, `gdist()`, `gpartition()`, `giterator_intervals()`, `gtrack_create()`, `gtrack_smooth()`, and `glookup()` now accept a pandas DataFrame of intervals as the `iterator` parameter, matching R misha behavior. Iterator intervals are intersected with the scope in Python and passed to C++ with `iterator=-1`.
- **Interval set names as iterator:** All iterator-accepting functions now also accept a string naming a saved interval set as the `iterator` parameter.
- **String interval set names in set operations:** `gintervals_intersect()`, `gintervals_union()`, `gintervals_diff()`, `gintervals_canonic()`, `gintervals_neighbors()`, `gintervals_covered_bp()`, `gintervals_coverage_fraction()`, `gintervals_force_range()`, `gintervals_mark_overlaps()`, and `gintervals_annotate()` now accept string interval set names in addition to DataFrames.
- **`partial_bins` parameter:** `giterator_intervals()` gained a `partial_bins` parameter (`"clip"`, `"drop"`, or `"exact"`) controlling how bins that don't fit entirely within an interval are handled.
- **`gintervals_covered_bp()` src parameter:** Optional `src` parameter restricts counting to the intersection of `intervals` with `src`.
- **`gtrack_create()` band support:** The `band` parameter is now wired through for 2D track creation instead of raising `ValueError`.

### Testing
- 21 new tests for API type gap fixes (string names, partial_bins, band, covered_bp src).
- Tests for DataFrame and interval-set-name iterators across all affected functions.

## v0.1.18 (2026-03-16)

### Features
- **User variables in expressions:** Python variables from the caller's scope can now be used in expression strings passed to `gextract()`, `gscreen()`, `gdist()`, `gsummary()`, and `gquantiles()`. Variables are resolved via frame introspection at the API boundary. An optional `vars=` parameter allows explicit control. Track names and coordinates (`CHROM`, `START`, `END`) take priority over user variables. The AST-validated security sandbox is preserved.

### Testing
- 16 new tests covering module-level variables, function locals, closures, explicit `vars=`, virtual track integration, priority/shadowing, error handling, and numpy operations.

## v0.1.17 (2026-03-16)

### Features
- **Motif format import:** `gseq_read_meme()`, `gseq_read_jaspar()`, and `gseq_read_homer()` for reading MEME, JASPAR PFM, and HOMER motif formats. Returns `dict[str, pd.DataFrame]` with A/C/G/T probability columns directly usable with `gseq_pwm()`. All parsers are native (no new dependencies).
- **Track export:** `gtrack_export_bedgraph()` and `gtrack_export_bigwig()` for exporting tracks and track expressions to standard bedGraph and BigWig formats. Supports gzip compression, virtual tracks, track expressions, and custom iterators.

### Testing
- 69 motif import tests covering MEME, JASPAR (header + simple PFM), HOMER parsing, error handling, and integration with `gseq_pwm()`.
- 14 track export tests covering bedGraph format, gzip compression, NaN exclusion, sparse tracks, track expressions, and BigWig conversion.
- Cross-validated with R misha: all 7 parsed matrices identical (max diff: 3.3e-16).

## v0.1.16 (2026-03-13)

### Features
- **Cross-platform `.gsm` format for gsynth models:** `gsynth_save()` and `gsynth_load()` now use a YAML metadata + binary arrays format readable by both Python and R misha. Legacy pickle files are still supported via automatic format detection.
- Added `compress` parameter to `gsynth_save()` for optional ZIP archive output.
- Added `gsynth_convert()` to migrate legacy pickle model files to `.gsm` format.
- Added `min_obs` field to `GsynthModel` dataclass.
- Added `pyyaml` as a dependency.

### Testing
- 9 new tests covering directory/ZIP round-trip, legacy pickle backward compatibility, conversion, and min_obs preservation.
- Cross-platform compatibility verified with R misha (Python saves, R loads and vice versa).

## v0.1.15 (2026-03-09)

### Features
- **Interval set attributes:** `gintervals_attr_get()`, `gintervals_attr_set()`, `gintervals_attr_export()`, `gintervals_attr_import()` for managing interval set attributes stored as `.iattr` binary files next to `.interv` files. Matches R misha `gintervals.attr.*` API.
- `gintervals_rm()` now cleans up companion `.iattr` files when deleting interval sets.

### Testing
- 24 new tests covering basic get/set, export/import, cleanup on deletion, bulk operations, and edge cases.

## v0.1.14 (2026-03-04)

### Features
- **Indexed 2D track support:** `gtrack_2d_convert_to_indexed()` converts per-chromosome-pair 2D tracks to single-file indexed format (`track.dat` + `track.idx`), matching R misha 5.5.0.
- Auto-conversion of 2D tracks to indexed format when database is indexed via `gdb_convert_to_indexed()`.
- `gdb_convert_to_indexed()` now includes 2D tracks (rectangles and points) in batch conversion.
- `gtrack_convert_to_indexed()` auto-dispatches to 2D conversion for 2D tracks.
- 2D tracks created via `gtrack_2d_create()` and `gtrack_2d_import_contacts()` are automatically converted to indexed format when the database is in indexed mode.

### Performance
- Indexed 2D tracks reduce file descriptor usage from O(N^2) to O(1) per track.
- Single mmap for indexed track.dat eliminates per-pair file open/close overhead.

## v0.1.13 (2026-03-03)

### Features

- **`gcompute_strands_autocorr`:** Strand autocorrelation for nascent transcription analysis, matching R misha's C++ `GenomeComputeStrandAutocorr` algorithm. Parses mapped reads files, builds binned strand coverage, computes Pearson cross-correlation at distance offsets.
- **`gintervals_annotate` tie_method:** Added `tie_method` parameter (`"first"`, `"min.start"`, `"min.end"`) for controlling tie-breaking when multiple annotations are equidistant.
- **`gtrack_2d_import` multi-file:** Accepts a list of file paths, reads and concatenates all before building the quad-tree.
- **`grevcomp`:** Standalone reverse complement function for DNA strings.
- **`gdb_mark_cache_dirty`:** Cache invalidation function (delegates to `gdb_reload`).
- **`gdataset_example_path`:** Returns filesystem path to a bundled example dataset.
- **COMPUTED track detection:** COMPUTED 2D tracks (Hi-C normalization) now raise an informative `NotImplementedError` in 7 API functions instead of failing silently.

### Testing

- **Multi-process hardening:** 18 new edge-case tests for `gmax_processes` / `_parallel_extract` covering parity, intervalID remapping, single-chrom, empty intervals, virtual track bypass, and stress scenarios.
- **Test suite:** 2103 passed, 25 skipped (up from 1993).

### Documentation

- **Track arrays explicitly excluded:** GAP-025 marked NOT PLANNED — 5 `gtrack.array.*` functions permanently out of scope.
- **Gap coverage:** 136/138 in-scope R misha functions (98.6%), up from 130/144 (90%).

## v0.1.12 (2026-03-02)

### Features

- **2D vtrack non-aggregation functions:** Added `exists`, `size`, `first`, `last`, `sample`, and `global.percentile` for 2D virtual tracks. All return one row per query interval.
- **2D set operations:** `gintervals_2d_intersect` (vectorized numpy pairwise rectangle intersection) and `gintervals_2d_union` (concatenate + sort).
- **2D iterator:** `giterator_intervals_2d` generator yields one DataFrame per input 2D interval. Supports band filtering, virtual tracks, multiple expressions.
- **Trans contact mirroring:** `gtrack_2d_import_contacts` now writes both chrA-chrB and chrB-chrA files for trans contacts, matching R misha symmetric behavior.
- **Path functions:** Added `gtrack_path(track)` and `gintervals_path(name)` convenience functions returning filesystem paths to track/interval set directories.
- **R-serialization detection:** `gtrack_var_get` now detects R-serialized track variables (RDS/serialize format) and raises an informative error instead of returning garbage or crashing.
- **gextract file output:** Added `file` parameter (streaming TSV write) and `intervals_set_out` parameter (save result intervals as named set) to `gextract`.
- **PWM spatial weighting:** Implemented `spat_factor`/`spat_bin` parameters for `gseq_pwm`, matching R misha's log-space spatial weight modulation. Removed NotImplementedError.
- **Bigset transparent iteration:** Named bigset interval sets are now transparently loaded in 21 functions across all modules (extract, summary, intervals, liftover, lookup, sequence, analysis, gsynth).
- **`gtrack_dbs` / `gintervals_dbs`:** Return the dataset that provides each track or interval set, matching R misha `gtrack.dbs` / `gintervals.dbs`.
- **`intervals_set_out` parameter:** Added to 8 functions (`gscreen`, `gpartition`, `glookup`, `gintervals_force_range`, `gintervals_union`, `gintervals_intersect`, `gintervals_diff`, `gintervals_normalize`) for saving results as named interval sets.
- **`gsynth_sample` bin_merge override:** Added `bin_merge` parameter for sampling-time bin merge overrides without modifying the model.
- **Parallel extraction (`gmax_processes`):** Multi-process `gextract` splits work by chromosome across forked workers. Configurable via `gmax_processes(n)`.

### Bug fixes

- **`dim` parameter in `gvtrack_iterator`:** Fixed correctness bug where `dim=1`/`dim=2` was silently ignored. 2D tracks can now be projected to 1D for extraction over 1D intervals.
- **`gintervals_force_range` column preservation:** Extra columns beyond chrom/start/end are now preserved when clipping intervals to chromosome boundaries, matching R misha behavior.

### Performance

- **C++ quad-tree reader:** Replaced pure-Python `struct.unpack` quad-tree traversal with C++ implementation (`QuadTreeReader.h/cpp`). Stats queries 182x faster, object queries 14x faster. Batch stats API (`pm_quadtree_query_stats_batch`) eliminates per-interval Python→C++ overhead for 2D vtrack aggregation.
- **gcis_decay vectorization:** C++ bulk quad-tree object extraction + numpy vectorized distance computation, binning (`np.searchsorted` + `np.bincount`), and domain containment checks. Eliminates per-object Python loop.
- **Liftover mapping vectorization:** Replaced per-interval Python mapping loop with numpy prefix-max overlap search, batch `searchsorted`, and vectorized strand-aware coordinate transformation.
- **DataFrame construction:** Replaced list-of-dicts `pd.DataFrame(rows)` patterns with column-wise numpy array construction in liftover.py and intervals.py (5 sites, 2-5x faster for large results).
- **gbins optimization:** Vectorized `gbins_summary` with `numpy.bincount` and optimized `gbins_quantiles` with sort-based grouping. 1.4-1.5x speedup.
- **K-mer vectorization:** Numpy stride_tricks-based k-mer hashing in `gseq_kmer` and `gseq_kmer_dist`. 3.5x average speedup over per-sequence Python loops.
- **Liftover overlap resolution:** Vectorized 7 overlap resolution functions using pandas groupby, numpy cumsum merging, and vectorized interval operations.
- **PWM scoring vectorization:** Numpy stride-tricks vectorized PWM scoring in `gseq_pwm` — sliding window via `as_strided`, fancy indexing into log_pssm, vectorized base encoding. 17.6x speedup for batch scoring.
- **VTrack per-row vectorization:** Replaced 4 iterrows/per-row loops in vtracks.py with numpy operations: `_build_unmasked_segments` no-mask path, overlap matching, nearest fallback, `base_starts` extraction.
- **Pre-computed vtrack values:** Eliminated per-chunk vtrack recomputation in mixed C++/vtrack extraction. Vtracks are now computed once for the full interval set and sliced per chunk.
- **Multi-chunk quad-tree writer:** `_quadtree.py` now supports multi-chunk serialization matching R misha's `StatQuadTreeCached` format. Prevents OOM on very large 2D tracks.
- **Batch gintervals_mapply:** Replaced per-interval `gextract` calls with single batch extraction + intervalID grouping. Eliminates N separate C++ calls.
- **C++ band-filtered query:** Added `pm_quadtree_query_objects_band` for C++ band-filtered quad-tree object enumeration, replacing pure-Python band filtering.

## v0.1.11 (2026-03-01)

### Features

- **2D virtual track aggregation:** All five 2D vtrack functions (`area`, `weighted.sum`, `min`, `max`, `avg`) are now supported, matching R misha feature parity. Previously only alias-style vtracks (`avg`/`mean`) were allowed in 2D extraction.
- **Hybrid quad-tree stat traversal:** 2D aggregation uses R misha's `get_stat` algorithm — O(1) for fully-contained subtrees via pre-computed node stats, O(K) enumeration only at partially-overlapping leaves. Arena-clamped 3-way intersection prevents double-counting across sibling nodes.
- **Band filter support:** 2D aggregation vtracks work with band filters (falls back to per-object enumeration since node-level stats don't account for diagonal band constraints).

### Bug fixes

- **pandas 3.0 compatibility:** Fixed C++ extension and Python codebase for pandas 3.0 (DataFrame construction, deprecated APIs).

## v0.1.10 (2026-02-27)

### Documentation

- Fixed API reference: added `docstring_style: numpy` to mkdocstrings config so Parameters, Returns, Examples, and See Also sections render correctly instead of as plain text.
- Split monolithic API page (856KB, 136 functions) into 10 per-section pages: Database, Datasets, Tracks, Virtual Tracks, Intervals, Data Operations, Liftover, Sequence Analysis, Genome Synthesis.
- Disabled inline source code display (`show_source: false`) to reduce page bloat.
- Added signature annotations and separate signature rendering for better readability.
- Limited TOC depth to prevent sub-sections (Parameters, Returns) from cluttering the sidebar.

## v0.1.9 (2026-02-27)

### Bug fixes

- Fixed multi-chunk quad-tree reader: cross-chunk references (negative kid offsets) now correctly read the target chunk header instead of treating the file position as a node offset.
- Fixed `gintervals_summary` and `gintervals_quantiles` for 2D intervals: replaced hardcoded 1D column names with dynamic coordinate column selection.
- Added `_maybe_load_2d_intervals_set` calls to `gsummary`, `gquantiles`, `gdist` so string-named 2D interval sets are auto-detected.

### Features

- **2D vtrack iterator shifts:** `gvtrack_iterator_2d` shifts (`sshift1`/`eshift1`/`sshift2`/`eshift2`) are now applied during 2D extraction.

### Performance

- Cache file mmap per chrom pair in 2D extraction — opens each file once instead of per-interval.
- Replace `iterrows()` with vectorized numpy extraction in `gtrack_2d_create` and `gtrack_2d_import_contacts`.

## v0.1.8 (2026-02-26)

### Bug fixes

- Fixed `GInterval::dist2coord` treating `coord == end` as inside the interval, inconsistent with the half-open `[start, end)` convention used throughout the codebase. This could affect distance calculations for coordinates that fall exactly on an interval boundary.

## v0.1.7 (2026-02-23)

### Performance

- **Batch chromosome normalization:** `_canonicalize_known_chroms` now normalizes only unique chromosome names (one C++ call per unique name instead of per row), then applies the mapping vectorially via `Series.map`. ~50× faster on large interval sets.
- **Vectorized dense pileup in `gtrack_import_mappedseq`:** Replace per-coordinate Python loops with NumPy-based duplicate detection, vectorized bin assignment, and `np.add.at` accumulation. Replace per-bin `dict.append` row building with `np.arange`/`np.concatenate`. ~22× faster row building.
- **Cached chromosome normalization during SAM parsing:** Per-read `pm_normalize_chroms` calls are now cached so each unique chromosome string is normalized only once.
- **Removed redundant DataFrame copy** in `gtrack_create_dense`.

### Features

- **`gtrack_create_dense_direct`:** New function that writes Misha dense track binary files directly, bypassing the C++ bridge. Supports `reload=False` for batch creation (call `gdb_reload()` once after many tracks). Inspired by borzoi_finetune's ~100× faster direct-write approach for multi-track workloads.

## v0.1.6 (2026-02-17)

### Documentation

- Replace the docs favicon with an icon-only transparent asset (no `pymisha` wordmark text) and configure MkDocs to use it.
- Re-export the docs logo with transparent background and cleaner edges while reducing file size from 5.2MB to ~3.6MB.

## v0.1.5 (2026-02-17)

### Features

- Add `gdb_export_fasta` for efficient full-database genome export to FASTA with streaming I/O for indexed and per-chromosome database formats, line wrapping, chunked reads, overwrite guard, and optional temporary `groot` switching.

### Bug fixes

- Fix `gtrack_liftover` indexed-source detection to ignore non-file entries (for example `vars/`), ensuring indexed-only source tracks are parsed from `track.idx`/`track.dat` correctly.

### Tests and benchmarks

- Add tests for `gdb_export_fasta` covering chunking/wrapping parity, overwrite behavior, root restoration, and per-chromosome `chr` prefix fallback.
- Make Python-vs-R benchmark comparison fair by forcing single-process R timing in benchmark helper (`options(gmax.processes = 1)`), and add a new large-database multiprocess benchmark for `gsummary`.

## v0.1.4 (2026-02-16)

### Documentation

- Add a concise "Misha Basics (Short Guide)" tutorial focused on core concepts: tracks, intervals, iterator policies, virtual tracks (including `sshift`/`eshift`), and PWM basics with examples from the bundled example DB.
- Add the new basics tutorial to MkDocs navigation under Tutorials.

## v0.1.3 (2026-02-15)

### Features

- **2D extraction parity:** Added 2D support for arithmetic expressions, virtual-track expressions, named 2D interval-set scopes in extraction/screening, and 2D iterator intervals from track-name iterators.
- **Intervals utilities:** Added `gintervals_is_bigset` API and exported it from the public package namespace.

### Bug fixes

- **Value-based virtual tracks:** Fixed DataFrame-source handling for interval-only functions, multi-chrom behavior, overlap validation by function class, and Python fallback parity for `nearest` and position reducers.
- **Filtered value semantics:** Fixed filtered value-based `avg` to use overlap-length weighting and aligned empty-bin behavior for reductions.
- **PWM spatial validation:** Enforced positive finite `spat_factor` and positive integer `spat_bin` at vtrack creation.
- **2D range clipping:** Added 2D support in `gintervals_force_range`.

## v0.1.2 (2026-02-15)

### Features

- **global.percentile vtracks:** Python-side support for `global.percentile`, `global.percentile.min`, and `global.percentile.max` virtual track functions.
- **Sparse vtrack C++ fast path:** Forward-scan cursor for `avg`/`sum`/`min`/`max`/`size`/`exists` reducers on sparse tracks, replacing per-interval generic reducer flow.

### Infrastructure

- **Conda packaging:** Automated conda package builds on release (Python 3.10–3.12 × NumPy 1.26/2.0/2.1 × Linux/macOS). Install via `conda install -c aviezerl pymisha`.

### Bug fixes

- Fix vtrack cache key to include DB root, avoiding cross-DB cache collisions.

## v0.1.1 (2026-02-14)

### Performance

- **Phase 1 optimizations:** Reduce BufferedFile default buffer (2MB → 128KB) for multitask workloads, cache per-chromosome CHROM strings to avoid per-row `PyUnicode_FromString`, skip `fseek` for sequential fixed-bin reads, add reducer fast-path in fixed-bin to skip unused function bookkeeping, stream sparse overlaps lazily instead of materializing all upfront.
- **Phase 2 optimizations:** Add basic-only sparse fast path in `calc_vals` (tight loop for avg/sum/min/max when no position/stddev/sample needed), replace `dynamic_cast` with `static_cast` in per-row hot loop, skip CHROM/START/END array population when expressions don't reference them, reuse scratch buffers in fixed-bin multi-bin path, eliminate extra copy in sparse track loading.
- Combined effect: 13–21% speedup across extraction workloads.

### Documentation

- Migrate docs from Sphinx/Furo to MkDocs Material.
- Port R misha vignettes to pymisha docs.
- Add pymisha logo and favicon.

## v0.1.0 (2026-02-13)

Initial public release.

### Core functionality

- **Track operations:** `gextract`, `gscreen`, `gsummary`, `gquantiles`, `gdist`, `glookup`, `gpartition`, `gsample`, `gcor` with C++ streaming backends.
- **Track creation:** `gtrack_create`, `gtrack_create_dense`, `gtrack_create_sparse`, `gtrack_modify`, `gtrack_smooth`, `gtrack_lookup`, `gtrack_create_pwm_energy`.
- **2D tracks:** `gtrack_2d_create`, `gtrack_2d_import`, `gtrack_2d_import_contacts`, 2D extraction, `gintervals_2d_band_intersect`.
- **Interval operations:** Union, intersection, difference, canonicalization, neighbors (k-nearest, directional), annotation, normalization, random generation, mark overlaps, mapply, import genes.
- **Virtual tracks:** 30+ aggregation functions, filtering with mask support, iterator shifts, 2D iterators.
- **Statistical analysis:** `gsegment` (Wilcoxon-based segmentation), `gwilcox` (sliding-window Wilcoxon), `gbins_summary`, `gbins_quantiles`, `gcis_decay`.
- **Liftover:** `gintervals_load_chain`, `gintervals_as_chain`, `gintervals_liftover`, `gtrack_liftover` with full overlap policy support.
- **Sequence analysis:** `gseq_extract`, `gseq_kmer`, `gseq_kmer_dist`, `gseq_pwm`.
- **Genome synthesis:** `gsynth_train`, `gsynth_sample`, `gsynth_random`, `gsynth_replace_kmer`, `gsynth_bin_map`, `gsynth_save`, `gsynth_load`.
- **Database management:** `gdb_init`, `gdb_create`, `gdb_create_genome`, `gdb_create_linked`, `gdb_convert_to_indexed`, `gdb_info`, `gdb_reload`, dataset and directory management.
- **Track management:** List, info, attributes, variables, import (BED, WIG, BigWig, TSV), copy, move, remove.

### R misha compatibility

- 123 of 145 R misha exports covered with compatible on-disk formats.
- Full database interoperability: tracks and interval sets created by either R misha or PyMisha are readable by both.

### Not yet implemented

- Track arrays (`gtrack.array.*`, `gvtrack.array.slice`).
- Legacy 2D format conversion (`gtrack.convert`).
