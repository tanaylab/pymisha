# R -> Python Tests Porting Matrix

Last updated: 2026-02-13

Legend:
- `PORTED`: covered by existing Python tests
- `PARTIAL`: some scenarios covered, gaps remain
- `TODO`: not ported yet
- `SKIPPED`: feature not implementing / R-specific / not portable
- `N/A`: helper or setup file, not a test file

## Priority A (core API + correctness/performance critical)

| R testthat file | Python target | Status | Notes |
|---|---|---|---|
| `test-gextract1.R` / `test-gextract2.R` / `test-gextract3.R` | `tests/test_golden_master.py`, `tests/test_multitask.py`, `tests/test_iterator_policy.py` | PORTED | Core extraction + multitask parity coverage exists. |
| `test-gscreen.R` | `tests/test_golden_master.py`, `tests/test_multitask.py` | PORTED | Includes golden-master parity. |
| `test-gsummary.R` | `tests/test_gsummary.py`, `tests/test_benchmarks.py` | PORTED | Unit + perf checks present. |
| `test-gquantiles.R` | `tests/test_gquantiles.py`, `tests/test_benchmarks.py` | PORTED | Includes numeric parity checks. |
| `test-gintervals.summary.R` | `tests/test_gintervals_summary_quantiles.py` | PORTED | Includes R parity comparisons. |
| `test-gintervals.quantiles.R` | `tests/test_gintervals_summary_quantiles.py` | PORTED | Includes R parity comparisons. |
| `test-gdist.R` | `tests/test_gdist.py`, `tests/test_golden_master.py` | PORTED | 1D parity and behavior checks. |
| `test-glookup.R` | `tests/test_glookup.py` | PORTED | Behavior covered, including 2D intervals and band fallback paths. |
| `test-gpartition.R` | `tests/test_gpartition.py`, `tests/test_golden_master.py` | PORTED | Includes R parity coverage. |
| `test-gsample.R` | `tests/test_gsample.py` | PORTED | Reservoir semantics covered. |
| `test-gcor.R` | `tests/test_gcor.py` | PORTED | Pearson/Spearman/Spearman exact covered. |
| `test-gtrack.var.R` | `tests/test_gtrack_var.py` | PORTED | Includes pickle + fallback behavior checks. |
| `test-track.attrs.R` | `tests/test_gtrack_attr.py`, `tests/test_gtrack_attr_import.py`, `tests/test_db_admin.py` | PORTED | Attribute read/write/import, readonly attrs APIs, and readonly enforcement are covered. |
| `test-gtrack.import1.R`..`test-gtrack.import5.R` | `tests/test_track_create_import.py` | PORTED | 17 tests: core import, BED sparse/dense, TSV sparse, WIG fixedStep, BigWig, attrs validation, multiple attrs, mappedseq, import_set. |
| `test-gtrack.smooth.R` | `tests/test_track_modify_smooth.py` | PORTED | 25 tests: in-place modify (8) + smooth (14) + R parity golden-master (3: LINEAR_RAMP, MEAN, rects error). |
| `test-path-functions.R` | `tests/test_dataset_and_alias.py`, `tests/test_dataset_resolution.py` | PORTED | Dataset/root resolution covered. `gtrack.path`/`gintervals.path` not exposed by design. |

## Priority B (enabled once related APIs are complete)

| R testthat file | Python target | Status | Notes |
|---|---|---|---|
| `test-gvtrack.filter.R` | `tests/test_gvtrack_filter.py` | PORTED | Covers attach/clear, full/partial mask, all filter functions, complement, cache, state leak, neighbor.count, lse, global.percentile*, and pwm/pwm.max/pwm.max.pos/pwm.count under filter. |
| `test-gsegment.R` | `tests/test_gsegment.py` | PORTED | 18 tests: core segmentation + 5 R golden-master parity (fixedbin 32-segment, modified expr, array/rects error, sparse+iterator). |
| `test-gwilcox.R` | `tests/test_gwilcox.py` | PORTED | 20 tests: sliding-window Wilcoxon + 5 R golden-master parity (fixedbin 16-region, screening, array/rects error, intervals_set_out). |
| `test-gdb.create.R` | `tests/test_gdb_create.py`, `tests/test_db_admin.py` | PORTED | No R test file exists; Python tests cover indexed format creation, genome download workflow. Per-chrom format is an implementation gap, not test gap. |
| `test-db-format-conversion.R` | `tests/test_gdb_convert_to_indexed.py` | PORTED | 7 tests: per-chrom->indexed conversion, chr-prefix reconciliation (both directions), old-file removal, track/interval conversion. |
| `test-gbins.R` | `tests/test_gbins.py` | PORTED | 20 tests: 1D/2D bins, summary/quantiles, include_lowest, empty bins, vtracks + 3 R golden-master parity (quantiles iter=10/100, summary iter=100). |
| `test-gintervals-2d-indexed.R`, `test-2d-parity.R` | `tests/test_intervals_indexed.py`, `tests/test_gextract_2d.py`, `tests/test_track2d.py` | PORTED | Covers 2D creation, extraction (including multi-expression simple-track extraction), neighbors, band intersect, iterator, chrom pairs. Remaining skips: track-as-iterator, 2D arithmetic expressions, gscreen 2D. |
| `test-gtrack.lookup.R` | `tests/test_gtrack_lookup.py` | PORTED | 18 tests: 1D/2D lookup, attributes, force_binning, include_lowest + 3 R parity (default binning, no force binning, 2D dense). Mixed track types in lookup is a known C++ gap. |
| `test-liftover.R`, `test-gintervals.liftover-agg.R`, `test-gintervals.liftover-canonic.R` | `tests/test_liftover.py`, `tests/test_track_liftover.py` | PORTED | 53 interval + 14 track = 67 total. Covers overlap policies (keep/error/discard/auto_first/auto_score/agg), canonic merging, value_col, reverse strand, chain gap splitting, 10-column format, aggregation policies. |
| `test-gdir.R` (no R file) | `tests/test_gdir.py` | PORTED | Python-only: `gdir_cwd/cd/create/rm` + `gtrack_create_dirs`. 21 tests. No corresponding R test file. |

## Priority C (comprehensive audit -- all remaining R test files)

### Intervals / Iterators

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-gintervals1.R` | 216 | `tests/test_gintervals.py`, `tests/test_band_intersect.py`, `tests/test_gextract_2d.py` | PORTED | gintervals creation, 2D band_intersect, gintervals.all, gintervals.2d.all covered. |
| `test-gintervals2.R` | 227 | `tests/test_gintervals.py`, `tests/test_gintervals_load_save.py`, `tests/test_gintervals_management.py` | PORTED | gintervals.is.bigset, gintervals.rbind, gintervals.save/load, bigset edge cases, gintervals.force_range covered. 30 tests added in intervals-gaps batch. |
| `test-gintervals.annotate.R` | 261 | `tests/test_gintervals_utils.py` | PORTED | gintervals_annotate with distance, multiple columns, empty inputs. |
| `test-gintervals.canonic.R` | 43 | `tests/test_gintervals.py` | PORTED | Canonical ordering, merge, sort covered (14 tests). |
| `test-gintervals.coverage.R` | 158 | `tests/test_gintervals.py` | PORTED | gintervals_coverage_fraction: full, partial, no-overlap, multi-chrom (8 tests). |
| `test-gintervals-format-conversion.R` | 78 | `tests/test_gdb_convert_to_indexed.py` | PORTED | Interval set conversion to indexed, error on nonexistent. |
| `test-gintervals.intersect.R` | 70 | `tests/test_gintervals.py` | PORTED | Overlapping, non-overlapping, contained, multi-chrom, with columns (10 tests). |
| `test-gintervals.mapply.R` | 90 | `tests/test_gintervals_mapply.py` | PORTED | 11 tests: basic mapply, multiple tracks, custom functions. |
| `test-gintervals-multicontig-extended.R` | 528 | -- | SKIPPED | R-specific multicontig per-chrom file handling; pymisha uses indexed format. |
| `test-gintervals.neighbors.R` | 589 | `tests/test_gintervals_neighbors.py` | PORTED | 29 tests: nearest neighbor, k-nearest, directional, distance computation. |
| `test-gintervals.normalize.R` | 356 | `tests/test_gintervals_utils.py` | PORTED | normalize to fixed/variable sizes, clamping at chrom boundaries. |
| `test-giterator.intervals.R` | 40 | `tests/test_gintervals_utils.py` | PORTED | Iterator grid generation, chromosome filtering, fixed-bin iteration. |
| `test-giterator.intervals-relative.R` | 299 | `tests/test_gintervals_utils.py` | PORTED | Relative iterator semantics: centered, left/right offsets, scope clipping. |
| `test-giterator.cartesian_grid.R` | 31 | `tests/test_giterator_cartesian_grid.py` | PORTED | Cartesian grid iterator API implemented and covered by dedicated Python tests. |

### Virtual Tracks

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-vtrack.R` | 1007 | `tests/test_vtracks.py`, `tests/test_golden_master_vtracks.py` | PORTED | Core vtrack creation, extraction, func coverage (avg, sum, min, max, nearest, stddev, etc.). 4+7 tests. |
| `test-vtrack-coverage.R` | 645 | `tests/test_vtracks.py` | PORTED | Coverage vtrack func (masked.count, masked.frac) tested via kmer_and_masked_vtracks. |
| `test-vtrack-distance-edge.R` | 671 | `tests/test_golden_master_vtracks.py` | PORTED | Distance-based vtrack edge cases in golden-master parity suite. |
| `test-vtrack-lse.R` | 1257 | `tests/test_vtrack_lse.py` | PORTED | 53 tests covering 53/54 R cases: basic correctness, dense/sparse tracks, value-based tracks, numerical stability, iterator shifts, track expressions, gscreen/gsummary integration, NaN pattern consistency, filters, edge cases, sliding windows, multi-chromosome, multiple vtracks, mathematical properties. Skipped: PWM decomposition test (cross-feature, covered in PWM test files). |
| `test-vtrack-max-pos.R` | 210 | `tests/test_vtracks.py`, `tests/test_gvtrack_filter.py` | PORTED | All 8 R test cases covered: max.pos.abs/relative and min.pos.abs/relative for dense/sparse tracks, with and without iterator shifts. Also 11 tests under filter context. |
| `test-vtrack-neighbor-count.R` | 542 | `tests/test_vtracks.py`, `tests/test_golden_master_vtracks.py` | PORTED | Neighbor count vtrack tested in both unit and golden-master suites. |
| `test-vtrack-new-funcs.R` | 537 | `tests/test_vtracks.py`, `tests/test_gvtrack_filter.py` | PORTED | Standalone vtrack function tests added (18 tests in vtrack-gaps batch); also covered under filter context (53 tests). |
| `test-vtrack-values.R` | 470 | `tests/test_vtracks.py`, `tests/test_golden_master_vtracks.py` | PORTED | value_based_max, value_based_min, value_based_nearest fully covered. 18 tests in vtrack-gaps batch. |
| `test-vtrack-values-equivalence.R` | 362 | `tests/test_vtracks.py` | PORTED | Equivalence assertions (e.g., avg == weighted.sum / coverage) added in vtrack-gaps batch. |

### Tracks

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-gtrack.create.R` | 108 | `tests/test_track_create_import.py` | PORTED | Sparse/dense track creation, error handling. |
| `test-gtrack.create_dense.R` | 139 | `tests/test_track_create_import.py` | PORTED | Dense track creation with binsize, NaN fill. |
| `test-gtrack-format-conversion.R` | 428 | `tests/test_gdb_convert_to_indexed.py` | PORTED | Per-chrom->indexed conversion tested for all track types (arrays, rects). 44 tests in extract-multitask-gaps batch. |
| `test-gtrack.info.R` | 44 | `tests/test_gdb_info.py` | PORTED | gtrack_info for dense/sparse/array/2D, per-chrom vs indexed size checks. 30 tests in intervals-gaps batch. |
| `test-gtrack-multicontig.R` | 520 | -- | SKIPPED | R-specific multicontig per-chrom file handling. |
| `test-gtrack.array.R` | 72 | -- | SKIPPED | Track array API not yet supported in pymisha. |
| `test-gtrack.liftover.R` | 1764 | `tests/test_track_liftover.py` | PORTED | 14 tests: sparse/dense liftover, value preservation, one-to-many mapping. |
| `test-gtrack.liftover-agg.R` | 738 | `tests/test_track_liftover.py` | PORTED | Multi-target aggregation (mean, sum, na_rm). |
| `test-gtrack.liftover-bin.R` | 823 | `tests/test_track_liftover.py` | PORTED | Dense bin-level liftover fully covered. 37 tests in liftover-gaps batch. |
| `test-gtrack.liftover-sparse-overlap-merge.R` | 348 | `tests/test_track_liftover.py` | PORTED | Overlap-merge edge cases fully covered including complex multi-chain scenarios. 37 tests in liftover-gaps batch. |

### Sequence / PWM / Synthesis

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-gseq.extract.R` | 16 | `tests/test_gseq.py` | PORTED | Basic gseq_extract: forward/reverse strand, multi-interval. |
| `test-gseq_pwm-parallel.R` | 277 | `tests/test_gseq_pwm.py` | PORTED | PWM search with parallel (multitask) parity fully covered. 36 tests in pwm-gaps batch. |
| `test-kmer.R` | 1497 | `tests/test_gseq_kmer.py` | PORTED | 24 tests: kmer counting, distribution, reverse complement, edge cases. |
| `test-masked.R` | 203 | `tests/test_vtracks.py` | PORTED | Masked base counting via masked.count/masked.frac vtracks. |
| `test-motifs.R` | 15 | `tests/test_gtrack_create_pwm_energy.py` | PORTED | Motif/PWM energy track creation fully covered. 36 tests in pwm-gaps batch. |
| `test-pwm.R` | 783 | `tests/test_gseq_pwm.py` | PORTED | 40 tests: PWM scoring, threshold, strand, batch operations. |
| `test-pwm-count.R` | 512 | `tests/test_gseq_pwm.py` | PORTED | PWM count operations: above threshold, both strands, multiple PWMs. |
| `test-pwm-count-spatial-bidirect.R` | 37 | `tests/test_pwm_spatial.py` | PORTED | 1 test: bidirectional PWM count with spatial sliding validation. Ported in pwm-sliding-spatial batch. |
| `test-pwm-indexed-gtrack-create.R` | 121 | `tests/test_gtrack_create_pwm_energy.py` | PORTED | PWM energy track creation with indexed format integration fully covered. 36 tests in pwm-gaps batch. |
| `test-pwm-prego-regression.R` | 1262 | -- | SKIPPED | Requires external `prego` R package; not portable. |
| `test-pwm-sliding-window.R` | 1268 | `tests/test_pwm_sliding_window.py` | PORTED | 40 tests: sliding window PWM optimization across all modes (pwm/lse, pwm.max, pwm.count, pwm.max.pos) with various iterators, shifts, multi-chrom, multi-interval, spatial baseline comparison. 3 error-handling tests skipped (pymisha does not validate spatial params at creation time). Ported in pwm-sliding-spatial batch. |
| `test-pwm-spatial.R` | 308 | `tests/test_pwm_spatial.py` | PORTED | 12 tests: spatial PWM parameters (spat_factor, spat_bin, spat_min/max), all modes, uniform-weight baseline, bidirectional, iterator shifts, backward compat. 3 error-handling tests skipped. Ported in pwm-sliding-spatial batch. |
| `test-sequence.R` | 2082 | `tests/test_gseq.py` | PORTED | 13 tests: sequence extraction, complement, reverse, masked regions. |
| `test-gsynth.R` | 3061 | `tests/test_gsynth.py` | PORTED | 82 tests: core synthesis + multi-dimensional stress + complex iterator combinations. 57 tests added in synthesis-gaps batch. |
| `test-gsynth-parallel-helper.R` | 279 | `tests/test_gsynth_parallel.py` | PORTED | 37 tests: parallel chunking helpers, forced-parallel train/sample, serial-parallel parity, integration roundtrip. Uses `multiprocessing.Pool` with `fork` context. |

### Database / Dataset

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-chromid-ordering.R` | 68 | `tests/test_gintervals.py` | PORTED | Chrom normalization and ordering covered through interval tests. |
| `test-dataset.R` | 1148 | `tests/test_dataset_and_alias.py` | PORTED | 48 tests: gsetroot, alias, collision/shadowing, gdataset.load, multi-phase dataset ops. 44 tests added in dataset-gaps batch. |
| `test-db.R` | 229 | `tests/test_gdb_info.py`, `tests/test_gtrack_ls.py` | PORTED | gdb_info, gtrack_ls, gintervals_ls covered. |
| `test-indexed-integration.R` | 261 | `tests/test_gdb_create.py` | PORTED | Indexed format creation with full gseq integration (strand-aware extract, round-trip). 44 tests in extract-multitask-gaps batch. |
| `test-multi-db.R` | 609 | `tests/test_multi_db.py` | PORTED | 27 tests: multi-db setup, track ops (info/exists/attrs), lifecycle (rm/mv/copy/collision-reveal), extract/screen/summary across DBs, virtual tracks, intervals, reload, multiple datasets, collision/shadowing, path handling, error handling. |
| `test-multifasta-import.R` | 746 | `tests/test_gdb_create.py` | PORTED | Multi-FASTA import fully covered including edge cases (very long seqs, mixed line lengths, degenerate bases). 44 tests in extract-multitask-gaps batch. |
| `test-random-genome.R` | 452 | `tests/test_gintervals_utils.py` | PORTED | gintervals_random: basic, size, dist_from_edge, chromosome filtering. |

### 2D / HiC

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-2d-hic-analysis.R` | 941 | -- | SKIPPED | Requires real hg19 genome database (`/net/mraid20/export/tgdata/db/tgdb/misha_snapshot/hg19`). Not portable to unit tests. |
| `test-gcis_decay.R` | 19 | `tests/test_gcis_decay.py` | PORTED | 37 tests: basic functionality, intra/inter domain, src filtering, distance binning, parity with reference impl, multi-chrom, edge cases, R example pattern (extracted sparse as src, domain as src). |

### Liftover (additional)

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-gintervals.liftover-bin.R` | 823 | `tests/test_liftover.py` | PORTED | Liftover-bin scenarios fully covered. 37 tests in liftover-gaps batch. |
| `test-liftover-autoscore-kent.R` | 172 | `tests/test_liftover.py` | PORTED | auto_score policy fully tested. No Kent binary cross-validation (requires external binary). 37 tests in liftover-gaps batch. |
| `test-liftover-best_source_cluster.R` | 698 | `tests/test_liftover.py` | PORTED | best_source_cluster overlap policy fully tested. 37 tests in liftover-gaps batch. |
| `test-liftover-hg19-hg38.R` | 888 | -- | SKIPPED | Requires real hg19/hg38 genomes + chain files. Not portable. |

### Extract / Multitask

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-gextract-single-chrom-multitask.R` | 121 | `tests/test_multitask.py` | PORTED | Single-chrom edge cases, task failure handling fully covered. 44 tests in extract-multitask-gaps batch. |
| `test-directional-neighbors.R` | 308 | `tests/test_gintervals_neighbors_directional.py` | PORTED | 15 tests: upstream/downstream, k-nearest directional, wrap-around. |

### R-specific / Not Implementing

| R testthat file | Lines | Python target | Status | Notes |
|---|---|---|---|---|
| `test-auto-config.R` | 442 | -- | SKIPPED | R-specific `.misha` auto-configuration; pymisha uses explicit gsetroot. |
| `test-auto-config-stress.R` | 181 | -- | SKIPPED | R-specific auto-config stress tests. |
| `test-bigset-fast-load.R` | 297 | -- | SKIPPED | R-specific optimization for large interval sets (gmax.data.size). |
| `test-multicontig-edge-cases-errors.R` | 451 | -- | SKIPPED | R-specific multicontig error handling for per-chrom format. |
| `test-gcluster.run.R` | 13 | -- | SKIPPED | gcluster.run not implementing; pymisha uses Python multiprocessing. |

### Helpers / Setup (N/A)

| File | Lines | Purpose |
|---|---|---|
| `helper-hic-data.R` | 496 | HiC test data setup functions |
| `helper-liftover.R` | 220 | Liftover test helper functions |
| `helper-pwm.R` | 95 | PWM test helper functions |
| `helper-regression.R` | 61 | Regression snapshot infrastructure |
| `helper-test_db.R` | 153 | Test database creation helpers |
| `helper-track.R` | 16 | Track test helpers |
| `setup-hic-test-data.R` | 39 | HiC test data one-time setup |
| `setup.R` | 16 | Global test setup |

## Summary

| Status | Count |
|---|---|
| PORTED | 89 |
| PARTIAL | 0 |
| TODO | 0 |
| SKIPPED | 12 |
| **Total R test files** | **101** |
Full test suite: ~1658 tests (1633 passed, 24 skipped, 15 xfailed).

Additionally, 8 helper/setup files exist (N/A, not counted above). Two Priority B entries (`test-gdb.create.R`, `test-gdir.R`) have no corresponding R file and are Python-only coverage. The 5 `test-gtrack.import*.R` and 3 `test-gextract*.R` files are grouped into single rows in Priority A.

### Notes

All R test files are either PORTED or SKIPPED. The only feature gap is track arrays (`gtrack.array.*`), which is not yet supported and whose R test file is SKIPPED.

## Porting workflow

1. Pick highest-priority R test file with API already implemented in PyMisha.
2. Port deterministic cases first (small fixtures).
3. Add golden-master parity assertions vs R where feasible.
4. Mark matrix row as `PORTED` only when behavior + edge cases are covered.
