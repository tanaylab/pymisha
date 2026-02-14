# PyMisha Implementation Plan (Audited)

Last audited: 2026-02-13

This document is the current execution plan. It replaces older narrative sections and keeps only verified status + prioritized next work.

## Scope and constraints

- Goal: parity with R misha for core 1D workflows, with compatible on-disk formats and competitive performance.
- Keep heavy genomic operations streaming in C++ (`pm_*`) whenever R uses C++ streaming.
- Python layer should orchestrate/parsing/validation; avoid materializing genome-scale data when avoidable.
- Endgame hardening order (guideline now, full pass later):
  - First: comprehensive API docs derived from R roxygen docs, with examples adapted to the PyMisha example DB.
  - Second: port the R test suite coverage to Python (golden-master/parity where feasible).
- Do not perform global file descriptor sweeps in teardown; close only descriptors owned by PyMisha.
- Track create/import compatibility:
  - Write misha-compatible per-chrom files first.
  - Auto-convert to `track.idx`/`track.dat` only when DB is indexed (`seq/genome.idx` + `seq/genome.seq`).
- Normalize chromosome names before Python/C++ merge comparisons (`_normalize_chroms()`), because C++ returns normalized names.

## What is implemented (verified in code)

### Public API footprint

- PyMisha exports 134 public symbols in `pymisha/__init__.py` (`__all__`). Test suite: ~1658 tests (1633 passed, 24 skipped, 15 xfailed) + 133 doctests.
- `misha` exports 145 symbols in `~/src/misha/NAMESPACE`.
- Direct name-mapped coverage (`.` -> `_`) is 123/145 exports.

### Core C++ streaming (implemented)

- Extraction/screening: `gextract` (with `colnames` support), `gscreen`
- Summary/statistics: `gsummary`, `gquantiles`, `gintervals_summary`, `gintervals_quantiles`
- Distribution/partition/sampling/correlation/lookup:
  - `gdist` (C++ path for non-vtrack expressions; streaming chunked Python path for vtracks with correct BinFinder binning semantics)
  - `glookup` (C++ streaming via `pm_lookup`; Python fallback for vtracks/2D/band; supports `force_binning`, `include_lowest`, multitasking)
  - `gpartition`
  - `gsample`
  - `gcor` (`pearson`, `spearman`, `spearman.exact`; exact supports multitask merge)
- Binned analysis (Python orchestration, multi-dim bins, mixed track types):
  - `gbins_summary`, `gbins_quantiles`
- Segmentation/statistical tests:
  - `gsegment` (Wilcoxon-based segmentation via `pm_segment` C++ backend)
  - `gwilcox` (sliding-window Wilcoxon test via `pm_wilcox` C++ backend)
- Interval kernels in C++:
  - `gintervals_union`, `gintervals_intersect`, `gintervals_diff`, `gintervals_canonic`, `gintervals_covered_bp`
  - `gintervals_neighbors` (base neighbor kernel)
- Track creation/modification C++ backends:
  - `gtrack_create` (`pm_track_create_expr` streaming)
  - `gtrack_create_dense`, `gtrack_create_sparse`
  - `gtrack_modify` (`pm_modify` in-place dense update with `find_existing_1d_filename` for chr prefix compat)
  - `gtrack_smooth` (`pm_smooth` circular-buffer LINEAR_RAMP/MEAN smoothing)
- Sequence extraction kernel: `gseq_extract`
- Sequence analysis (pure Python): `gseq_kmer`, `gseq_kmer_dist`, `gseq_pwm`
- Genome synthesis (C++ backends + Python orchestration): `gsynth_train`, `gsynth_sample`, `gsynth_random`, `gsynth_replace_kmer`, `gsynth_bin_map`, `gsynth_save`, `gsynth_load`
- PWM track creation: `gtrack_create_pwm_energy` (Python orchestration via temporary PWM vtrack + extract + dense track)

- Track creation via lookup:
  - `gtrack_lookup` (Python orchestration over `glookup` + `gtrack_create_dense`/`gtrack_create_sparse`/`gtrack_2d_create`; supports multi-dimensional lookup, force_binning, include_lowest, band)

- 2D track creation (pure Python quad-tree writer):
  - `gtrack_2d_create` (creates RECTS or POINTS tracks from 2D intervals + values; auto-detects format; Python quad-tree serializer produces misha-compatible binary files)
  - `gtrack_2d_import` (imports from tab-delimited file)

### Python APIs implemented (orchestration/utility)

- DB/datasets: `gdb_init`, `gdb_reload`, `gdb_unload`, `gdb_info`, `gsetroot`, `gdb_create`, `gdb_create_genome`, `gdb_create_linked`, `gdb_convert_to_indexed`, `gdb_get_readonly_attrs`, `gdb_set_readonly_attrs`, `gdataset_load/ls/unload/save/info`
- Directory management: `gdir_cwd`, `gdir_cd`, `gdir_create`, `gdir_rm`, `gtrack_create_dirs`
- Track management: `gtrack_ls/info/exists/dataset/rm/mv/copy`, attrs (`get/set/export/import`), vars (`ls/get/set/rm`), import APIs
- Indexed conversion helpers: `gtrack_convert_to_indexed`, `gtrack_create_empty_indexed`
- Intervals: constructors/load/save/update/list/remove/dataset/chrom_sizes, 1D/2D indexed conversion helpers, `gintervals_rbind`, `gintervals_mark_overlaps`, `gintervals_annotate`, `gintervals_normalize`, `gintervals_random`, `giterator_intervals`, `gintervals_mapply`, `gintervals_import_genes`
- Analysis: `gcis_decay` (cis-decay analysis for 2D tracks with domain-aware binning)
- Vtracks: create/list/info/iterator/remove/clear
- Sequence helpers: `gseq_rev`, `gseq_comp`, `gseq_revcomp`
- Liftover/chain: `gintervals_load_chain`, `gintervals_as_chain`, `gintervals_liftover`, `gtrack_liftover` (pure Python, complete workflow)

### Explicitly not implemented / partial

- `gvtrack_filter` is implemented in Python orchestration with broad function coverage:
  - supports canonicalized mask sources (DataFrame, intervals set name, sparse/interval track name, or list unions),
  - applies mask after iterator shift and returns `NaN` for fully masked intervals,
  - supported with filtering: `avg/mean/sum/min/max/stddev/quantile/nearest/coverage`, `exists/size/first/last/sample`, `first/last/sample.pos.(abs|relative)`, `distance*`, `neighbor.count`, `kmer/masked`, `pwm/pwm.max/pwm.max.pos/pwm.count`, `lse`, and `global.percentile*`,
  - `stddev` and `quantile` use raw-bin-value extraction from unmasked segments for exact results,
  - `nearest` uses first-unmasked-segment semantics matching R behavior,
  - `max.pos.*` and `min.pos.*` now supported under filtering (position-of-max/min with masked-segment exclusion).
- `band` parameter supported in: `gextract` (2D), `gsummary`, `gquantiles`, `gdist`, `gintervals_summary`, `gintervals_quantiles`, `gbins_summary`, `gbins_quantiles`, `glookup` (all via extract-then-compute path for 2D/band). `gtrack_lookup` inherits band support from `glookup`.
- `gtrack_create` correctly rejects band (not applicable to whole-genome track creation).
- `gdist` uses streaming chunked Python path when expressions include virtual tracks (memory-bounded, correct binning).
- `gquantiles` uses streaming chunked Python path when expressions include virtual tracks (bounded reservoir sampling via `CONFIG['max_data_size']`; warns when quantiles are approximate).
- `glookup` falls back to Python when expressions include virtual tracks or when running 2D/band paths.
- 2D track creation (`gtrack_2d_create`, `gtrack_2d_import`, `gtrack_2d_import_contacts`) done. `gintervals_2d_band_intersect` done. `gvtrack_iterator_2d` done. `gextract` supports 2D extraction (quad-tree spatial query), including multi-expression extraction for simple 2D track names.

## Missing vs R misha (high-level)

Large missing groups:

1. 2D tracks/intervals (partially done)
- ~~`gintervals.2d`, `gintervals.2d.all`~~ **DONE**
- ~~`gtrack.2d.create`, `gtrack.2d.import`~~ **DONE** — pure Python quad-tree writer with misha-compatible binary format. Auto-detects RECTS vs POINTS. 13 tests.
- ~~`gtrack.2d.import_contacts`~~ **DONE** — pure Python: intervals-value and fends-value formats, midpoint conversion, canonical ordering, cis mirroring, duplicate handling. 9 tests.
- ~~`gintervals.2d.band_intersect`~~ **DONE** — diagonal band intersection with rectangle shrinking. `intervals_set_out` supported via 2D-aware `gintervals_save`. 13 tests.
- ~~`gvtrack.iterator.2d`~~ **DONE** — stores 2D shift parameters (sshift1/eshift1/sshift2/eshift2). 4 tests.
- ~~2D extraction in `gextract`~~ **DONE** — Python-side quad-tree spatial query for 2D extraction. Supports both R-created (`chr`-prefixed) and PyMisha-created file naming, including multi-expression extraction for simple 2D track names.

2. ~~High-impact analysis/statistical APIs~~ **DONE**
- ~~`gbins.summary`, `gbins.quantiles`~~ implemented as `gbins_summary`, `gbins_quantiles`

3. ~~Liftover and chain APIs~~ **DONE**
- ~~`gintervals.load_chain`, `gintervals.as_chain`, `gintervals.liftover`~~ implemented as pure Python with chain parsing, overlap handling (error/keep/discard/auto_score/auto_first/auto_longer/agg), canonic merging, value_col tracking, and 29 tests.
- ~~`gtrack.liftover`~~ **DONE** — pure Python: reads source track (dense/sparse) from per-chrom files, lifts intervals via `gintervals_liftover`, aggregates overlapping target values (mean/median/sum/min/max/count/first/last with na_rm/min_n), creates sparse target track. Supports pre-loaded chain or chain file path. 13 tests.

4. Advanced track tooling
- ~~`gtrack.modify`, `gtrack.smooth`~~ **DONE** — `gtrack_modify` (in-place dense track modification via C++ `pm_modify` with scanner-based expression evaluation, `find_existing_1d_filename` for chr prefix compat) and `gtrack_smooth` (new dense track creation with LINEAR_RAMP/MEAN smoothing via C++ `pm_smooth` circular buffer). 22 tests.
- ~~`gtrack.lookup`~~ **DONE** — Python orchestration over `glookup` + `gtrack_create_dense`/`gtrack_create_sparse`/`gtrack_2d_create`. Supports multi-dimensional lookup, `force_binning`, `include_lowest`, and `band`. Output type follows lookup scope (1D dense/sparse or 2D). 15 tests.
- ~~`gtrack.create_pwm_energy`~~ **DONE** — Python orchestration: loads PSSM from `GROOT/pssms/`, creates temporary PWM virtual track (LSE scoring), extracts values at iterator resolution via `gextract`, writes dense track via `gtrack_create_dense`. 17 tests.
- Remaining: `gtrack.convert` (legacy format migration, low priority — only relevant for old 2D formats)

5. Track-array
- `gtrack.array.*`, `gvtrack.array.slice`

6. ~~DB/directory admin APIs~~ **DONE**
- ~~`gdb.create`~~ **DONE** (indexed format; per-chromosome format not yet)
- ~~`gdb.create_genome`~~ **DONE** (download pre-built genomes: `mm9`, `mm10`, `mm39`, `hg19`, `hg38`)
- ~~`gdb.create_linked`~~ **DONE** (linked DB root with symlinks to parent `seq/` and `chrom_sizes.txt`)
- ~~`gdb.convert_to_indexed`~~ **DONE** (pure Python per-chromosome -> `genome.seq`/`genome.idx`, chr-prefix reconciliation, optional track/interval conversion)
- ~~`gdir.*`~~ **DONE** (`gdir_cwd/cd/create/rm`, `gtrack_create_dirs`)
- ~~readonly-attrs APIs~~ **DONE** (`gdb_get_readonly_attrs`, `gdb_set_readonly_attrs`, readonly enforcement in `gtrack_attr_set`/`gtrack_attr_import`)

7. ~~Sequence synthesis / motif APIs~~ **DONE**
- ~~`gsynth.*`~~ **DONE** — `gsynth_train`, `gsynth_sample`, `gsynth_random`, `gsynth_replace_kmer`, `gsynth_bin_map`, `gsynth_save`, `gsynth_load`. Stratified Markov-5 model with C++ training/sampling backends. Pure Python orchestration for bin mapping and model persistence.
- ~~`gseq.kmer`, `gseq.kmer.dist`, `gseq.pwm`~~ **DONE**

## Prioritized roadmap

### Priority 0 (next sprint): close correctness/performance gaps in existing surface

1. ~~Extend `gvtrack_filter` parity~~ **DONE** — All filter functions supported: `stddev`, `quantile`, `nearest`, `exists/size/first/last/sample`, `first/last/sample.pos.(abs|relative)`, `max.pos.*`, `min.pos.*`, `neighbor.count`, `lse`, `global.percentile*`, and non-count PWM functions.

2. ~~Remove Python fallback in `gdist` for vtracks~~ **DONE** — replaced full-materialization fallback with streaming chunked Python path (`_gdist_vtrack_streaming`). Values are extracted and binned per chunk, accumulating counts without materializing all data simultaneously. Fixed binning bug where values at `breaks[0]` were incorrectly included when `include_lowest=False`. 15 tests.

3. ~~Add streaming `pm_lookup` backend for `glookup`~~ **DONE** — C++ streaming `pm_lookup` backend with `BinsManager`-based binning, `force_binning` clamping, `include_lowest`, multitask support via FIFO. Python fallback retained for vtrack expressions. 14 parity tests.

4. ~~`gsegment`, `gwilcox`~~ **DONE** — C++ streaming backends (`pm_segment`, `pm_wilcox`) with `IncrementalWilcox` incremental Wilcoxon test, Python wrappers, and 28 tests.
5. ~~DB/directory admin (`gdb.create*`, `gdir.*`)~~ **DONE** — `gdb_create`, `gdb_create_genome`, `gdb_create_linked`, `gdb_convert_to_indexed`, `gdb_get_readonly_attrs`, `gdb_set_readonly_attrs`, `gdir_cwd/cd/create/rm`, `gtrack_create_dirs`, `gdataset_*` management. 51+ tests.

6. ~~Optimize Core Interval Kernels (C++ backends)~~ **ASSESSED** — Investigation (2026-02-10) shows `gintervals_canonic` is already fully optimized in C++ (mapping built inline during merge, no Python loop overhead). `gintervals_summary` and `gintervals_quantiles` C++ backends are well-optimized with multitask support. The vtrack fallback paths use pandas groupby which is acceptable for the typical use case. No further optimization needed for P0.

7. ~~colnames argument in gextract~~ **DONE** — `gextract` now accepts `colnames` parameter for custom column naming on both C++ and Python vtrack paths. 6 tests.

8. ~~Remove Python full-materialization in `gquantiles` for vtracks~~ **DONE** — replaced `_extract_expr_values` fallback with streaming chunked evaluation (`_gquantiles_vtrack_streaming`) and bounded reservoir sampling controlled by `CONFIG['max_data_size']`. Emits `RuntimeWarning` when sampling is required. Added tests for streaming path selection and approximation warning.

### Priority 1: core parity expansion

6. ~~Implement 2D foundational APIs~~ **DONE**
- ~~`gintervals_2d`, `gintervals_2d_all`~~ **DONE** (pure Python)
- ~~`gtrack_2d_create`, `gtrack_2d_import`~~ **DONE** — pure Python quad-tree writer producing misha-compatible StatQuadTreeCached binary format. RECTS/POINTS auto-detection. Overlap checking. C++ type detection verified. 13 tests.
- ~~`gtrack_2d_import_contacts`~~ **DONE** — pure Python: intervals-value and fends-value formats, midpoint conversion, canonical ordering, cis mirroring, duplicate handling. 9 tests.
- ~~`gintervals_2d_band_intersect`~~ **DONE** — diagonal band intersection. 13 tests.
- ~~`gvtrack_iterator_2d`~~ **DONE** — 2D shift storage. 4 tests.
- ~~2D extraction in `gextract`~~ **DONE** — Python quad-tree spatial query with multi-expression support for simple 2D track names.

7. ~~Implement `band` support~~ — **DONE**
- **DONE**: `gextract` (2D diagonal band filter), `gsummary`, `gquantiles`, `gdist` — extract-then-compute path via Python-side band filtering. 4 tests.
- **Also DONE**: 2D intervals auto-detected and routed to Python extraction path in `gsummary`, `gquantiles`, `gdist`, `gintervals_summary`, `gintervals_quantiles`, `gbins_summary`, `gbins_quantiles`, `glookup`.
- `gtrack_lookup` inherits band support from `glookup`. `gtrack_create` correctly rejects band (not applicable to whole-genome track creation).

8. ~~Implement `gbins_summary` and `gbins_quantiles`~~ **DONE** — Pure Python orchestration over `gextract` + numpy binning/stats. Supports multi-dimensional bins, include_lowest, mixed track types. 17 tests.

9. ~~Implement liftover chain workflow~~ **DONE** — `gintervals_load_chain` (UCSC chain parser with chrom normalization, overlap handling, min_score filtering), `gintervals_as_chain` (DataFrame validation + attrs), `gintervals_liftover` (interval mapping, canonic merging, value_col, include_metadata), `gtrack_liftover` (track-level liftover with aggregation). Pure Python implementation. 42 tests total.

### Priority 2: ecosystem completeness — **ALL DONE**

10. ~~sequence synthesis/motif stack~~ **DONE**
- ~~`gseq_kmer`~~ **DONE** — Pure Python k-mer counting in sequences. Supports forward/reverse/both strands, ROI boundaries, gap skipping, count/fraction modes. 17 tests.
- ~~`gseq_kmer_dist`~~ **DONE** — Counts all k-mers of size k within genomic intervals. Supports masking. 7 tests.
- ~~`gseq_pwm`~~ **DONE** — Pure Python PWM/PSSM scoring of arbitrary DNA sequences. Supports 4 modes (lse/max/pos/count), bidirectional/strand-specific scoring, ROI with extend, gap skipping, neutral character policies (average/log_quarter/na), prior pseudocount, DataFrame and array PSSM inputs. 40 tests.
- ~~`gsynth.*`~~ **DONE** — `gsynth_train`, `gsynth_sample`, `gsynth_random`, `gsynth_replace_kmer`, `gsynth_bin_map`, `gsynth_save`, `gsynth_load`. C++ training/sampling backends with Python orchestration.
- ~~`gtrack.create_pwm_energy`~~ **DONE** — Python orchestration via temporary PWM vtrack + extract + dense track creation. 17 tests.
11. Track arrays (`gtrack.array.*`, `gvtrack.array.slice`) — **NOT YET SUPPORTED** (not implemented in v0.1.0; may be added in a future release)


### Priority 3: endgame docs and parity hardening

12. Full documentation pass from R roxygen2 — **DONE**
- All 133 docstring examples across 16 modules are now runnable doctests using `gdb_init_examples()`.
- Doctest collection enabled in pytest via `addopts = "--doctest-modules"` in `pyproject.toml`.
- State-mutating examples use `# doctest: +SKIP`; variable output uses `+ELLIPSIS`/`+NORMALIZE_WHITESPACE`.

13. ~~Port R tests to Python~~ **MOSTLY DONE** — 89/101 R test files ported (89 PORTED, 1 TODO, 11 SKIPPED). Full suite: ~1658 tests (1633 passed, 24 skipped, 15 xfailed). Parallel gsynth is now done. Remaining TODO: track arrays (not yet supported). See `dev/notes/r-tests-porting-matrix.md` for details.

14. Bug fixes: Major progress. Critical (0/5 pending), High (0/16 pending). See `dev/notes/code-review-report.md` and `dev/notes/code-review-pending.md` for details on pending items.

15. Tool modernization — **DONE** (Phase A)
- `.pre-commit-config.yaml` added (ruff check/format, trailing-whitespace, end-of-file-fixer, check-yaml).
- Ruff config converged: line-length=120, expanded rule sets (E, F, I, UP, W, B, SIM, PIE, RET, C4), isort known-first-party, per-file-ignores for tests.
- `pymisha/_pymisha.pyi` type stub created: 46 function signatures + error class, matching full C++ method table.

## Documentation stack recommendation (2026)

- Recommended for PyMisha: `Sphinx + MyST + Furo` as the primary API docs stack.
  - Reason: stronger cross-referencing and better fit for mixed Python/C++ internals and parity notes.
- Optional companion for landing/guide UX: `MkDocs + Material` for a lightweight user guide site.
- If maintaining one stack only, prefer Sphinx+MyST for this repository.

## Tooling modernization (staged)

- `uv`: adopt for local/CI environment and command execution (`uv run pytest`, `uv sync`) without forcing immediate packaging backend changes.
- `ruff`: converge lint/format workflow to Ruff and reduce separate `black`/`isort` maintenance.
- Keep `setuptools` build backend for now because of the C++ extension build path; revisit backend migration only after parity-critical work.
- Keep `pyproject.toml` as the single configuration source.

## Testing and parity policy

For every new API in Priority 0/1:

- Add Py tests + golden-master comparisons versus R misha where feasible.
- Validate streaming behavior (memory bounded for genome-scale input).
- Validate DB interoperability (same dataset readable by both R misha and PyMisha).
- Add examples on example DB mirroring R roxygen usage where applicable.

## Documentation policy

- Keep README short and user-facing (installation, quick start, current gaps).
- Keep this plan as source of truth for implementation status and priorities.
- Update both in same PR whenever API status changes.
