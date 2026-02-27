# Changelog

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
