# PyMisha Implementation Notes

Detailed per-function implementation notes for agent context. Referenced from `CLAUDE.md`.

Last updated: 2026-02-12

## C++ Streaming Backends

- `giterator_intervals` wraps `pm_iterate`, supports numeric iterators; track expressions resolve bin size via `gtrack_info`. `interval_relative=True` is numeric-iterator only.
- `gsample` is C++ (`pm_sample`) with NaN skipping and shuffled reservoir output.
- `gcor` is C++ (`pm_cor`) for Pearson, Spearman (approx), and Spearman exact; multitask merge exists for Pearson + Spearman exact.
- `gsegment` and `gwilcox` are C++ streaming (`pm_segment`, `pm_wilcox`) using `IncrementalWilcox` (ported from R misha). Python layer converts `maxpval` to z-score via `_pval_to_zscore()` before calling C++. Both require fixed-bin iterators.
- `glookup` is C++ streaming (`pm_lookup`) with `BinsManager`-based N-dimensional binning, `force_binning` clamping, `include_lowest`, and multitask support. Falls back to Python when expressions contain virtual tracks, `band` is provided, or intervals are 2D. Python fallback preserves 1D/2D coordinate schema and handles `value == breaks[-1]` as the last bin. Lookup table is flattened in Fortran (column-major) order to match `BinsManager::vals2idx` index convention.
- `gtrack_modify` is C++ streaming (`pm_modify`): in-place dense track modification using scanner-based expression evaluation. Uses `find_existing_1d_filename` to handle legacy `chr`-prefixed per-chromosome filenames. Only dense tracks supported; binsize auto-inferred from track info.
- `gtrack_smooth` is C++ streaming (`pm_smooth`): creates a new dense track with smoothed values from a track expression. Supports `LINEAR_RAMP` (distance-weighted) and `MEAN` (uniform window) algorithms. Uses circular buffer for O(1) per-bin smoothing. `weight_thr` semantics differ by algorithm: scaled by `(num_samples_aside + 1)` for LINEAR_RAMP, absolute count for MEAN.

## Python Orchestration APIs

- `gintervals_annotate` is Python-layer over `gintervals_neighbors`; large annotation payloads may need C++ fusion later.
- `gintervals_summary/quantiles` support `intervals_set_out` and return `None` when writing set output.
- `gintervals_rbind` requires exact column parity across inputs.
- `gtrack_import_mappedseq` follows misha coordinate/strand semantics and returns per-chrom/total stats.
- `gtrack_var_*` uses pickle storage with R-serialized fallback reads.
- `gdb_info` mirrors R `gdb.info` semantics.
- `gdataset_save/info` uses flat scalar `misha.yaml` schema.
- `gdb_create` is pure Python: parses FASTA (including gzip), writes MISHAIDX-format `genome.idx` with CRC64-ECMA checksums, and `genome.seq`. Supports sorted chromosomes and FASTA header sanitization matching C++ logic. Only indexed format is supported.
- `gdb_create_genome` is pure Python orchestration: downloads supported prebuilt DB archives (`mm9`, `mm10`, `mm39`, `hg19`, `hg38`) from the misha S3 bucket, extracts with path-traversal checks, then initializes the extracted DB root.
- `gdb_get_readonly_attrs` / `gdb_set_readonly_attrs` manage `.ro_attributes` at DB root. Reader supports R-serialized files (via `pyreadr`) and a PyMisha fallback format. Writer uses `Rscript` serialization when available for R compatibility.
- `gdir_cd` changes `_GWD` (global working directory), clears vtracks, reloads DB. Track/interval names are rebased via `_apply_gwd_to_names()` in Python since C++ always scans from `GROOT/tracks`.
- `gtrack_create_dirs` creates the dot-separated namespace directory hierarchy for a track name.
- `gbins_summary` and `gbins_quantiles` are pure Python orchestration over `gextract` + numpy. Support multi-dimensional bins, `include_lowest`, and mixed track types (each expression extracted independently). Not yet C++ streaming — adequate for typical bin counts but could be slow for very fine-grained bins on large genomes.
- `gtrack_lookup` is Python orchestration over `glookup` + track creators. For 1D results it uses `gtrack_create_dense`/`gtrack_create_sparse` (dense if iterator is integer, otherwise sparse). For 2D results it uses `gtrack_2d_create`. Supports multi-dimensional lookup, `force_binning`, `include_lowest`, and `band` (via `glookup`). Not streaming — materializes values via `glookup` then creates the track.
- `gintervals_mapply` is pure Python: iterates over intervals, extracts track values at native resolution via `gextract` per interval, applies user function. Supports multiple expressions, strand reversal, `intervals_set_out`, custom column names. Not batched — calls `gextract` per interval, adequate for moderate interval counts.
- `gintervals_update` is pure Python: loads existing intervals set, replaces all intervals for a target chromosome, re-saves. Supports adding new chromosomes and deleting intervals (pass None).
- `gtrack_attr_import` is pure Python: bulk imports track attributes from a DataFrame. Supports `remove_others` to clear non-imported attributes while preserving read-only attrs, validates duplicate track/attribute names, and rejects writes to read-only attrs.
- `gtrack_attr_set` rejects writes/removals for read-only attributes.
- `gextract` supports `colnames` parameter for custom column naming on both C++ and Python vtrack paths.

## Liftover / Chain

- `gintervals_load_chain` is pure Python: parses UCSC chain files, normalizes target chroms via `_normalize_chrom()` (C++ chromkey), handles source/target overlap policies (error/keep/discard/auto_score/auto_first/auto_longer/agg), negative strand coordinate flipping, min_score filtering, and stores policies in DataFrame attrs.
- `gintervals_as_chain` validates DataFrame columns and sets overlap policy attrs.
- `gintervals_liftover` maps source intervals through chain blocks: computes overlap, proportional coordinate mapping (+ and - strand), canonic merging of adjacent target blocks from same intervalID/chain_id, value_col tracking, and include_metadata for score output.
- UCSC chain format terminology: UCSC "target/reference" (tName) = misha "source" (chromsrc); UCSC "query" (qName) = misha "target" (chrom). This reversal is a common source of confusion.
- `gtrack_liftover` is pure Python: reads source track (dense/sparse) per-chrom files directly via `_read_source_track()`, lifts intervals through `gintervals_liftover`, aggregates overlapping target values with configurable `multi_target_agg` (mean/median/sum/min/max/count/first/last) and `na_rm`/`min_n` parameters, then creates a sparse target track. Does not require `gdb_init` on the source DB.

## Virtual Track Filtering

- `gvtrack_filter` supports broad function coverage: `avg/mean/sum/min/max/stddev/quantile/nearest/coverage`, `exists/size/first/last/sample`, `first/last/sample.pos.(abs|relative)`, `distance*` passthrough, `neighbor.count`, `kmer/masked` functions, `pwm/pwm.max/pwm.max.pos/pwm.count`, `lse`, and `global.percentile*`.
- `stddev` and `quantile` use raw-bin extraction from unmasked segments for exact results.
- `nearest` uses first-unmasked-segment semantics.
- Remaining unsupported under filtering: `max.pos.*` and `min.pos.*` variants. Unsupported funcs raise `NotImplementedError`.

## Distribution / Binning

- `gdist` vtrack path uses streaming chunked extraction+binning (`_gdist_vtrack_streaming`): parses expressions to find physical tracks and vtracks, extracts values per chunk via `pm_extract` (physical) + `_compute_vtrack_values` (vtracks), bins with `_bin_values` (correct BinFinder semantics), and accumulates counts. Memory is bounded by chunk size (`eval_buf_size`).
- `gquantiles` vtrack path uses streaming chunked evaluation with bounded reservoir sampling (`_gquantiles_vtrack_streaming`): avoids full-value materialization and bounds memory by `CONFIG['max_data_size']`; emits `RuntimeWarning` when quantiles are approximate due to sampling.
- Binning helper `_bin_values` in `summary.py` implements C++ `BinFinder` semantics: bins are `(breaks[i], breaks[i+1]]` (open-left, closed-right); `include_lowest` makes first bin `[breaks[0], breaks[1]]`; NaN and out-of-range values get -1.

## 2D Tracks

- `gtrack_2d_create` builds an in-memory quad-tree, serializes to misha-compatible `StatQuadTreeCached` binary format. Auto-detects RECTS (signature -9) vs POINTS (signature -10) based on interval sizes (all 1bp -> POINTS). Writes per-chromosome-pair files (e.g., `1-2`). Uses `#pragma pack(8)` struct layout: Leaf=80 bytes, Node=104 bytes, Obj<Rectangle_val<float>>=48 bytes, Obj<Point_val<float>>=32 bytes.
- `gtrack_2d_import` reads tab-delimited file (7+ columns), delegates to `gtrack_2d_create`.
- `gtrack_2d_import_contacts` reads contacts in intervals-value (7-col) or fends-value format. Converts intervals to midpoints (POINTS track). Handles canonical chrom ordering, mirrors cis contacts around diagonal, sums duplicate contacts when `allow_duplicates=True`.
- `_quadtree.py` contains the quad-tree implementation: `QuadTree` class, `write_2d_track_file`, `verify_no_overlaps_2d`, `read_2d_track_objects`, `query_2d_track_objects`. Single root chunk (no multi-chunk splitting).
- `gextract` 2D extraction: detects 2D intervals (has `chrom1` column), queries per-chrom-pair binary files via `query_2d_track_objects`. Handles both PyMisha-created (`c1-c2`) and R-created (`chrc1-chrc2`) file naming. Supports multi-expression extraction for simple 2D track names (rows anchored to iterator track when provided), and supports `band=(d1, d2)` diagonal band filter.
- `gsummary`, `gquantiles`, `gdist`, `gintervals_summary`, `gintervals_quantiles`, `gbins_summary`, `gbins_quantiles`, `glookup` auto-detect 2D intervals and support `band` parameter, routing to extract-then-compute path. Not C++ streaming — acceptable for typical 2D track sizes.

## PWM Energy Track Creation

- `gtrack_create_pwm_energy` is Python orchestration: loads PSSM from `GROOT/pssms/{pssmset}.key` + `.data`, creates a temporary PWM virtual track with LSE scoring (`func='pwm'`), extracts values genome-wide at the given iterator resolution via `gextract` (Python vtrack path), then writes a dense track via `gtrack_create_dense`. The temporary vtrack is cleaned up in a finally block. This differs from R misha which uses a dedicated C++ `gcreate_pwm_energy` function, but produces equivalent results.
- `_load_pssm_from_db` reads the tab-delimited `.key` and `.data` files. Key file maps PSSM id to consensus string. Data file has rows: `id\tposition\tA\tC\tG\tT`. Missing positions filled with uniform (0.25 each).

## Genome Synthesis

- `gsynth_train` extracts track values for stratification dimensions via `gextract`, computes flat bin indices, then calls `pm_gsynth_train` C++ backend for Markov-5 context counting. Returns a `GsynthModel` dataclass with counts, CDFs, and per-bin statistics.
- `gsynth_sample` and `gsynth_random` call `pm_gsynth_sample` C++ backend with pre-computed CDFs and bin indices. Support FASTA, binary seq, and vector output formats. `gsynth_random` creates a single-bin uniform CDF model.
- `gsynth_replace_kmer` calls `pm_gsynth_replace_kmer` C++ backend for iterative k-mer replacement in genome sequences.
- `gsynth_bin_map` is pure Python: computes bin remapping for merging sparse bins, used to consolidate low-count bins before training.
- `gsynth_save`/`gsynth_load` use pickle serialization for model persistence.

## Sequence Analysis

- `gseq_kmer` is pure Python: sliding window k-mer counting with forward/reverse/both strand support, ROI boundaries (1-based), gap character skipping, count/fraction modes. Palindromic k-mers on strand=0 are counted once (forward only).
- `gseq_kmer_dist` extracts sequences from genomic intervals via `gseq_extract`, counts all k-mers of size k (1-10), returns sorted DataFrame of (kmer, count). Supports mask intervals. Skips N-containing k-mers.
- `gseq_pwm` is pure Python: PWM/PSSM scoring of arbitrary DNA sequences. Supports 4 modes (`lse`/`max`/`pos`/`count`). Bidirectional or strand-specific. ROI via `start_pos`/`end_pos` (1-based inclusive) with `extend`. Gap skipping with logical-to-physical coordinate mapping. Neutral character policies: `average`, `log_quarter`, `na`. Prior pseudocount applied before normalization. PSSM accepted as numpy array `(w, 4)` or DataFrame with A/C/G/T columns. Spatial weighting (`spat_factor`) accepted but not yet implemented. The C++ `DnaPSSM`/`PWMScorer` backends exist for genomic track scoring (virtual track functions `pwm.*`), but `gseq_pwm` operates on arbitrary strings and doesn't require `gdb_init`.

## Per-Chromosome File Naming

Tracks created by R misha may use `chr`-prefixed filenames (e.g., `chr1`), while pymisha-created tracks use normalized names (e.g., `1`). Use `GenomeTrack::find_existing_1d_filename()` in C++ when opening existing per-chrom track files for update/read.

## Known Future Gaps

- `glookup` vtrack fallback could similarly be converted to a streaming chunked path.
- `gdir_cd` track filtering is Python-side only; for full parity the C++ scan root should respect GWD (not critical for correctness, only performance with very large track trees).
- `gtrack.convert` (legacy 2D format migration only — low priority) is not yet implemented.
