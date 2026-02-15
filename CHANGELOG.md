# Changelog

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
