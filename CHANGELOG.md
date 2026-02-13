# Changelog

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
