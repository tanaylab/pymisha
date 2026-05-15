# Changelog

## v0.1.62 (2026-05-15)

### Performance
- **WIG/BedGraph import is ~7x faster.** `gtrack_import` from plain (non-gzipped) WIG and BedGraph files now streams through a C++ parser (`pm_parse_wig_or_bedgraph`) instead of building Python lists line-by-line. Bench: 50 MB / 3M-row WIG variableStep parses in 0.7s vs 5.6s; 91 MB / 3M-row BedGraph in 0.9s vs 5.9s. Gzipped inputs still use the pure-Python streamer. Closes part of Group M.2 of the 2026-05-15 parity audit.

## v0.1.61 (2026-05-15)

### Fixes
- **Docs build:** removed a `../../dev/notes/` link from `docs/guides/parity.md` that broke `mkdocs --strict` on main (dev/ is excluded from the shipped tree).

## v0.1.60 (2026-05-15)

### Fixes
- **`_read_r_readonly_format`** now uses the native R-serialize reader (`pymisha._r_serialize`) instead of `pyreadr`. The runtime no longer needs `pyreadr` to load a database with R-written read-only-attribute files. `pyreadr` remains a soft dependency only for `gintervals_save` (RDS writer).

### Docs
- Refreshed `docs/guides/parity.md` to reflect Groups G/H/I/J shipped in v0.1.55-v0.1.59.

## v0.1.59 (2026-05-15)

### Features
- **`gtrack_array_create(track, description, intervals, values, colnames)`** writes a new array track from a DataFrame + 2-D matrix in memory. NaN cells are stored sparsely, matching R's array-track invariant. The on-disk format is byte-compatible with R misha - `.colnames` written by PyMisha is readable via `unserialize()` in R. Complements `gtrack_array_extract` / `gtrack_array_get_colnames` shipped in v0.1.57.

## v0.1.58 (2026-05-15)

### Fixes
- **CI mypy errors from v0.1.55-v0.1.57 are resolved.** mypy is clean across `pymisha/_r_serialize.py`, `pymisha/genome/registry.py`, `pymisha/intervals.py`, and `pymisha/tracks.py`. No runtime behaviour change.

### Improvements
- **Obsolete 2D track formats now produce an actionable error message** instead of "requires conversion". Lists the obsolete format name, points to R misha's `gtrack.convert` as the upgrade path, and notes that a PyMisha in-process converter is tracked under Group J of the 2026-05-15 parity audit. The legacy formats (`OLD_RECTS1`, `OLD_RECTS2`, `OLD_COMPUTED1`, `OLD_COMPUTED2`, `OLD_COMPUTED3`) have not been written by misha for years; the converter port is deferred.

## v0.1.57 (2026-05-15)

### Features
- **Array track read support.** Closes B1 of the 2026-05-15 parity audit. Three new public functions mirror R's `gtrack.array.*`:
  - `gtrack_array_get_colnames(track)` reads the column names from `<track>/.colnames` via the new R-serialize reader.
  - `gtrack_array_set_colnames(track, names)` writes the file in a format both R misha and pymisha can read.
  - `gtrack_array_extract(track, slice=, intervals=)` returns a DataFrame with one row per overlapping track interval and one column per requested array column (NaN where the track has no value at that index). `slice` accepts either column names or 0-based indices.
- **Improved `gextract` error message on array tracks.** Previously raised `Track type 'array' not yet supported`; now points the user at `gtrack_array_extract`. Full C++ scanner integration for array tracks is deferred to Group K.

## v0.1.56 (2026-05-15)

### Features
- **Native R-serialize reader (drops the Rscript hard dependency).** `pymisha._r_serialize.read()` decodes R's XDR (binary) format plus gzip-compressed RDS, with support for character / integer / numeric / logical vectors, NULL, raw bytes, named lists, data frames, and the common ALTREP encodings (`compact_intseq`, `compact_realseq`, `wrap_*`, `deferred_string`). Used by:
  - **`gintervals_load(legacy_bigset)`**: legacy `.meta` files no longer need `Rscript` at runtime - the previous `"Rscript is required to load legacy intervals metadata"` error is gone.
  - **`gtrack_var_get`**: variables written by R misha (XDR or gzipped XDR) are now readable from Python. The ASCII variants (`A\n`, `B\n`) are still rejected with a clear message pointing to `serialize(..., ascii=FALSE)` as the workaround.

## v0.1.55 (2026-05-15)

### Features
- **`gdb_list_genomes()` and `gdb_genome_info(name)`** are now public APIs (R parity for `gdb.list_genomes` / `gdb.genome_info`). They walk the registry chain (explicit arg -> `PYMISHA_GENOME_REGISTRY` -> `./misha.yaml` -> bundled `recipes.yaml`) and return a DataFrame / recipe dict.
- **`gintervals_neighbors(intervals_set_out=, warn_ignored_strand=, mindist1/maxdist1/mindist2/maxdist2=)`** now accept the same parameter surface as R `gintervals.neighbors`. The 2D-distance params are accepted as no-ops for 1D inputs (matching R); 2D inputs raise `NotImplementedError` (deferred to the 2D C++ scanner work).
- **`gintervals_mapply(enable_gapply_intervals=, band=)`** R parity. `enable_gapply_intervals=True` passes the current iterator interval as a `gapply_intervals` kwarg to `func` (PyMisha analogue of R's `GAPPLY.INTERVALS`).
- **`gcis_decay`** now accepts compound 2D expressions that reference exactly one 2D track (e.g., `"track + 0"`). Distance is computed from coordinates so the expression value is unused.

### Fixes
- **`gsynth_sample` / `gsynth_random` / `gsynth_replace_kmer` default `output_format` is now `"misha"`** to match R. `"seq"` remains accepted as a legacy alias. Unknown values now raise `ValueError` instead of silently falling back to `"fasta"`.
- **`gsynth_random` accepts an `iterator=1` parameter** for R API parity (no-op in PyMisha because sampling is per-position).
- **`gintervals_neighbors` no longer ignores `strand` columns silently** - emits a warning by default when `intervals1` has a `strand` column and `use_intervals1_strand=False`. The directional helpers (`_upstream`, `_downstream`) suppress the warning.

## v0.1.54 (2026-05-14)

### Fixes
- **`gsynth_sample` on 0D models with unaligned intervals no longer falls back to uniform-random output.** A signed-int64 overflow when adding `iter_size = INT64_MAX` (the no-constraint sentinel for 0D models) to a positive bin start position made the bin-bounds check fail, returning `bin_idx = -1` and falling through to uniform-random base selection. `gsynth_forbid_kmer` on a 0D model now correctly produces samples that respect the forbidden pattern regardless of interval alignment. Aligned-interval samples are byte-identical to v0.1.53. (Roadmap follow-up #1.)

## v0.1.53 (2026-05-14)

### Features
- **`gtrack_create_dense(func=)` knob for per-bin aggregation.** Seven reductions: `weighted.mean` (default, byte-identical to prior output), `weighted.sum`, `max`, `min`, `median`, `count`, `coverage`. The `coverage` mode with `values=[1]*N` and `defval=0` produces a one-call ChIP-seq-style pileup track from BED inputs. `defval` acts as a synthetic uncovered contribution for every func except `count`. (R misha 5.6.32 `068a02a2`, `5e69c2c8`.)

### Notes
- **R 5.6.32 `1b4f5065` (unsigned-wrap guard) is N/A in pymisha:** the `ov_end > ov_start` guard was already present at `src/PMTrackCreate.cpp:637`; the bug never existed here. Wrap-regression test added as a sentinel.

## v0.1.52 (2026-05-14)

### Fixes
- **`gdb_install_intervals` now produces the full TSS/UTR sets on NCBI and UCSC backends.** `_install_genes` previously filtered only Ensembl/GENCODE feature names (`transcript`, `five_prime_utr`, `three_prime_utr`); production NCBI GFF3 uses `mRNA` + `five_prime_UTR` + `three_prime_UTR`, and UCSC's `ncbiRefSeq.gtf.gz` uses `5UTR` + `3UTR`. The NCBI path was producing empty TSS+UTR sets; UCSC was producing empty UTR sets. The synthetic test fixture used GENCODE naming and hid the bug.
- **`pwm.edit_distance` family: `direction="below"` + `bidirect=True` now takes MAX across strands (was MIN).** A genomic substitution affects both strands, so disrupting a motif site needs both strands below threshold; the harder strand bounds the answer. (R misha 5.6.10 `19c51158`.)
- **`pwm.edit_distance` family: removed hidden `score_min = score_thresh` default for `direction="below"`.** The hidden default was a footgun: users pre-filtering for strong matches and then calling edit distance got unexpected NAs. `score_min` now defaults to no filter for both directions; pass it explicitly when needed. (R misha 5.6.10 `88e49b62`.)

### Performance
- **`gextract` gained a `multitasking_strategy` config knob (`"auto"` | `"tracks"` | `"tiles"`).** `auto` (default) picks track-parallel for >= 8 expressions with a non-streaming iterator and tile-parallel (legacy chrom-parallel) otherwise. Track-parallel runs each worker over a subset of expressions across the full interval set, which substantially outperforms the legacy chrom-parallel path on many-track / few-interval workloads (e.g., scoring thousands of motif vtracks across a peak set). (R misha 5.6.18 `gmultitasking.strategy`.)

### Breaking
- `direction="below"` PWM edit distance results change for `bidirect=True` callers (now MAX across strands) and for any caller relying on the implicit `score_min = score_thresh` default. Set `score_min` explicitly to recover the prior filter, or `score_min=-inf` to keep no filter.

## v0.1.51 (2026-05-14)

### Performance
- **`gintervals_random` auto-routes to a C++ implementation for large genomes.** Genomes with > 1000 contigs OR > 10M total bp get the new `pm_intervals_random` C++ path; smaller genomes keep the pure-Python implementation. 5000-contig synthetic benchmark: 6.3x speedup (~9 ms -> ~1.4 ms). C++ path uses `std::mt19937_64`; output is statistically equivalent to the Python path but NOT bit-identical (different RNG family). Edge case from R `1b41bceb` (contig length exactly `size + 2*dist_from_edge`) handled.

## v0.1.50 (2026-05-14)

### Performance
- **`gtrack_create_sparse` and `gtrack_create_dense` stream directly to `track.dat` + `track.idx` on indexed databases.** Removed the per-chromosome intermediate files + post-create `gtrack_convert_to_indexed` step. Saves ~2.5M filesystem operations per track on million-contig genomes. Byte-identical regression tests guard the on-disk layout. `gtrack_liftover` inherits the optimization (it calls `gtrack_create_sparse`). Non-indexed databases keep the previous per-chromosome write path. (R misha 5.6.30 `94a6446d`, `b2ca08cc`, `18985be4`.)

### Notes
- **R 5.6.30 `747b0076` (TrackIndexWriter refactor) is N/A in pymisha:** the abstraction never existed here; the new direct-write path adds an `IndexedTrackWriter` helper inside `PMTrackCreate.cpp` rather than refactoring shared code.

## v0.1.49 (2026-05-14)

### Fixes
- **`gdb_reload` now clears the Python-side dataset scan cache.** Track and interval-set names created externally between two `gdb_reload` calls become immediately visible. The C++ side was already rescanning correctly; this closes the Python-cache gap. (R misha 5.6.30 `c82b01f0`.)

### Notes
- **R 5.6.30 `6dc476a8` (meta short-circuit per-chrom probe) is N/A in pymisha:** `pm_track_info` already takes the indexed fast path (two `stat()` calls on `track.idx` + `track.dat`) when an index is present and only falls back to per-chromosome stats for per-chrom-format tracks - matching R's post-`6dc476a8` behavior.

## v0.1.48 (2026-05-14)

### Performance
- **`pm_intervals_all` caches its result on `PMDb`.** Repeat calls return in microseconds instead of rebuilding the chrom-intervals DataFrame each time. Cache invalidated on `gdb_init` / `gdb_reload`. (R misha 5.6.30 `ce788e75`.)
- **`find_existing_1d_filename` short-circuits for indexed tracks.** Skips the O(N_aliases) chromAlias scan + per-candidate `access()` syscalls when the track has an index. Hot read paths on indexed million-contig databases no longer pay the alias-search cost per chromosome transition. (R misha 5.6.30 `b340ccfa`.)
- **Eliminated redundant `stat(track.idx)` calls inside `init_read`.** Both `GenomeTrackFixedBin::init_read` and `GenomeTrackSparse::init_read` now ask `get_track_index()` directly (which caches its own stat result) instead of stat-then-load. Removes ~2 stat syscalls per chromosome transition. (R misha 5.6.30 `1db467d1`.)

### Notes
- **R 5.6.30 `5a9828e6` (GenomeChromKey caching) and `327bcbb2` (scanner index-aware iterator setup) are N/A in pymisha:** the singleton `PMDb` already caches the chrom-key for the database lifetime, and the scanner's bin-size discovery loop already breaks after the first non-empty chromosome, making it constant work. Verified by measurement.

## v0.1.47 (2026-05-14)

### Fixes
- **mypy compliance** for the `pymisha/genome/` modules added in v0.1.43-v0.1.46. Restores green CI on `main`. No behavior change.

## v0.1.46 (2026-05-14)

### Features
- **`gdb_build_genome(..., source={"source": "ncbi", "accession": "GCF_..."})`.** NCBI Datasets API v2 backend with FTP fallback. Install path fetches `SEQUENCE_REPORT` + (optional) `GENOME_GFF` only (full `GENOME_FASTA` skipped, ~900 MB saved per call on a human accession). FTP fallback at `https://ftp.ncbi.nlm.nih.gov/genomes/all/<GCx>/<NNN>/<NNN>/<NNN>/<acc>_<asm>/` covers empty-zip GFF and rmsk (`<acc>_<asm>_genomic.gff.gz`, `<acc>_<asm>_rm.out.gz`). cgi/cytoband are not provided by NCBI; requesting them warns and skips. (R misha 5.6.30 `409c235e` `d6cd6047`.)
- **`gdb_install_intervals(..., force=True)`** parity confirmed across all three install-capable backends (ucsc/ucsc-hub/ncbi). With `force=False` (default), requesting a set the backend does not provide raises `ValueError`. With `force=True`, the orchestrator emits a `UserWarning` and installs only the available subset. (R misha 5.6.29 `968bf782`.)

## v0.1.45 (2026-05-14)

### Features
- **`gdb_build_genome(..., source={"source": "ucsc-hub", "accession": "GCA_..."})`.** UCSC mammal-hub backend. Probes known hub filename conventions (no HTML scraping); 404 on a per-asset URL is treated as "not available". Hub directories provide chromAlias + chrom.sizes + FASTA + RepeatMasker + cpgIslandExt + GTFs under `genes/`. Cytoband is never available from hubs. (R misha 5.6.16.)
- **Multi-pass chromAlias rescue (`match_by_length=True`, default).** Four-pass algorithm fills missing canonical entries via unique-length matches, overrides misnamed rows when a length pair resolves them, breaks cross-row name collisions by length, and synthesizes `target_chroms` rows when the upstream sources don't carry them. Post-rescue `min_coverage` gate (R 5.6.30 `537bfe29`). Single-pass mode (`match_by_length=False`) retained for callers that want the strict pre-rescue behavior.

## v0.1.44 (2026-05-14)

### Features
- **`gdb_install_intervals(groot, source, sets=...)`.** Install UCSC intervals sets (`genes`, `rmsk`, `cgi`, `cytoband`) into an existing groot. Parses gzipped GTF + RepeatMasker .out + UCSC database TSVs; bp-weighted chromAlias detection picks the canonical chrom column. Writes per-rmsk-class subsets for the major classes (SINE/LINE/LTR/DNA/Simple_repeat/Low_complexity). Provenance written to `<groot>/tracks/.misha_install.json`. (R misha 5.6.16 48c8a700 partial port; ucsc-hub and ncbi backends land in v0.1.45-v0.1.46.)
- **`gdb_build_genome(..., sets=...)` re-enabled.** When `sets` is non-empty, `gdb_build_genome` now invokes `gdb_install_intervals` after the sequence build. Same set of backends (`ucsc` only in v0.1.44).

## v0.1.43 (2026-05-14)

### Features
- **`gdb_build_genome(name, ...)` skeleton** for the `manual`, `local`, and `s3` backends. New entry point with a bundled registry of 11 genomes (hg19, hg38, mm9, mm10, mm39, rn6, rn7, dm6, ce11, sacCer3, danRer11). Resolution chain: explicit `registry=` arg, then `$PYMISHA_GENOME_REGISTRY`, then `./misha.yaml`, then the bundled `recipes.yaml`. The `ucsc`, `ucsc-hub`, `ncbi` backends and `gdb_install_intervals` land in later releases. (R misha 5.6.16 partial port.)

## v0.1.42 (2026-05-14)

### Robustness
- **`gtrack_rm` and `gintervals_rm` return in microseconds even on directories with millions of files.** The doomed directory is renamed to a hidden `.trash.<base>.<pid>.<rand>` sibling and the actual filesystem cleanup runs in a detached background process. Falls back to synchronous unlink when atomic rename is not available (cross-filesystem). The cross-database overwrite path in `gtrack_copy` uses the same mechanism. Stale `.trash.*` and `.<name>.tmp.*` entries are swept on `gdb_init` (24h cutoff). (R misha 5.6.30.)
- **`gtrack_create_*` is now atomic.** An interrupted create no longer leaves a partial track directory that blocks re-creation. Writers mkdir into a hidden `.<base>.tmp.<pid>.<rand>` directory and `os.rename` to the final name on success; on failure the tmp dir is trashed. Concurrent rescans (`gtrack_ls`) skip in-flight tmp dirs. Applies to `gtrack_create_sparse`, `gtrack_create_dense`, `gtrack_create_dense_direct`, `gtrack_smooth`, `gtrack_create`, `gtrack_2d_create`, and `gtrack_2d_import_contacts`. Functions that delegate (`gtrack_import`, `gtrack_import_set`, `gtrack_import_mappedseq`, `gtrack_create_pwm_energy`, `gtrack_2d_import`) inherit atomicity from the wrapped writers they call. (R misha 5.6.30.)
- **`gdb_convert_to_indexed(threads=N)` runs per-track conversions in parallel.** Default `threads` is `min(os.cpu_count(), 8)`. Per-track failures surface as warnings without aborting the batch. Falls back to serial on Windows and when `threads == 1`. (R misha 5.6.30.)

## v0.1.41 (2026-05-13)

### Testing
- **`gintervals_random` regression tests** for the R 5.6.30 `1b41bceb` edge case: contigs of length exactly `size + 2*dist_from_edge` (a single valid start position is available) and contigs of length exactly equal to `size` with `dist_from_edge=0`. PyMisha's pure-Python implementation already handles these correctly - the C++ bug never made it to the Python port. (R misha 5.6.30.)

## v0.1.40 (2026-05-13)

### Features
- **`gsynth_forbid_kmer(model, pattern)`** returns a new model whose samples avoid `pattern` as a substring (subject to a seeding caveat for the first `k` bp of each interval). Useful for CpG-null or motif-null synthetic backgrounds. Pattern length is capped at `model.k + 1`. (R misha 5.6.16.)
- **`gsynth_cell_merge()` + `gsynth_sample(cell_merge=)`.** Per-joint-cell CDF redirects let you reassign under-trained joint cells to a nearest-sufficient neighbor at sample time without retraining. Accepts a list of `{"from": [...], "to": [...]}` dicts or a DataFrame with `from_<d>` / `to_<d>` columns. Warns on self-redirects (dropped) and duplicate sources (later entry wins). (R misha 5.6.16.)
- **`gsynth_sample(output_format="fasta")` and `gsynth_random(output_format="fasta")` now write a samtools-compatible `.fai`** alongside the FASTA (`<output>.fai`). Removes the need to call `samtools faidx` by hand. The `gsynth_random` `.fai` is a pymisha-only extension (R only added it to `gsynth.sample()`). (R misha 5.6.16.)

### Testing
- Project-wide RNG seed convention: every random seed in production code, tests, benchmarks, and doctest examples now uses `60427` (was a mix of `42` and `60427`).

## v0.1.39 (2026-05-13)

### Fixes
- **`gpartition`, `gquantiles`, `gscreen`, `gintervals_quantiles` with `±inf` breaks routed every value to the last bin.** `BinFinder` used a uniform-binsize fast path that hit `Inf/Inf = NaN` on breaks like `c(-inf, x, inf)` and silently misrouted output. The binary-search path is now used whenever the binsize is non-finite. (R misha #110.)

### Features
- **`gtrack_copy` now supports cross-database copy.** New optional arguments:
  - `db=` — destination database root. Accepts the active `GROOT`, any member of `GDATASETS`, or a valid unloaded misha root (`chrom_sizes.txt` + `tracks/`). Cross-genome destinations work.
  - `overwrite=` — replace an existing destination track.
  - `src` accepts a single track name or an iterable. With an iterable, `dest=` becomes a namespace prefix (`"ns"` -> `ns.track1`, `ns.track2`, ...); `dest=None` keeps each track's name.

  Format mismatches between source and destination (per-chromosome vs indexed) are converted on the fly. Chromosome-order differences are remapped per file, with chr-prefix canonicalization. Chromosomes present in source but not destination are dropped with a warning; the copy refuses to create an empty track. 2D tracks (`rectangles`, `points`) require identical chromosome order. `gtrack_copy` now returns the list of created destination track names. (R misha 5.6.28.)

### Performance
- **`gdb_init` on fragmented assemblies (>1000 contigs, no `chr` prefix, no mito chrom) no longer creates a `chr<name>` alias for every contig.** Ensembl-style mammalian or insect genomes are unaffected; only genomes like Phylo447 (2.4M `scaffold_*` contigs) avoid the alias blowup. (R misha #112.)

## v0.1.38 (2026-05-07)

### CI fixes
- **conda publish: install `anaconda-client` in the activated build env, not base.** v0.1.36 moved `conda-build` to `-n base` so the `conda build` subcommand resolves; that surfaced the symmetric issue on the upload step, where the workflow's `defaults.run.shell: bash -el {0}` activates the `build` env, and `anaconda` (provided by `anaconda-client`) wasn't on its PATH. With v0.1.36/v0.1.37 conda builds, packages built fine but the upload step bailed with `anaconda: command not found`. Split the install: `conda-build` in base (next to the conda CLI), `anaconda-client` in the activated env.

## v0.1.37 (2026-05-07)

### CI fixes
- **mypy under numpy 2.2.x in CI.** `_numpy.where(...)` returns a generic-shape `ndarray[tuple[int, ...], ...]` that mypy (with the typing tightening that landed in numpy 2.2) refuses to assign back to a variable previously bound to a 1-D `ndarray[tuple[int], ...]`. Two `_numpy.where(...)` results in `gsynth_score` (the per-flat-bin index in `_extract_bin_data` and the NaN-masked `raw` log-p) are now wrapped in `cast(ndarray, ...)`. Newer numpy (≥2.4) didn't reproduce this locally, hence the slip.

## v0.1.36 (2026-05-07)

### CI fixes
- **mypy: clean type errors introduced in v0.1.34/v0.1.35.** Type-correct fixes only; no functional changes.
  - `gintervals_from_tuples` cast its `list[int]` strands to the wider `list[int | str]` accepted by `gintervals` after the character-strand widening in v0.1.34.
  - Wrap a couple of fancy-indexed `_DNA_BASE_CODE[seq_bytes]` and `arr[arr[:, 0].argsort()]` returns in `_numpy.asarray` so mypy recognises the ndarray result.
  - `_extract_bin_data`'s string lookup branch in `gsynth_score` now narrows the result of `_maybe_load_intervals_set` (which can return a name string) to `DataFrame` before calling `reset_index`.
  - Replace `unique(..., return_index=True) -> list(...)` reassignment with a fresh `edge_list: list[int]` to avoid the array→list type clash.
  - Use `cast(dict[str, Any], metadata["data"])` to type-check the prior-bin nested dict assignment in `gsynth_save`.
- **conda publish: install `conda-build` in the `base` env, not the activated `build` env.** The recent `setup-miniconda@v3` change leaves the activated env without the `conda build` subcommand if `conda-build` is installed there, which was causing every `conda build` step to fail with "argument COMMAND: invalid choice: 'build'" since v0.1.34. Install with `-n base` so the subcommand is registered against the conda CLI.

## v0.1.35 (2026-05-06)

### Features
- **Per-bin Dirichlet prior in `gsynth_train`.** New `prior=` argument selects how the per-bin prior `pi(b)` is resolved for the Bayesian posterior `P(a|c,b) = (N + alpha * pi_a(b)) / (sum_a N + alpha)` (with `alpha = pseudocount`):
  - `"marginal"` (default) — per-bin empirical base composition computed on post-merge counts. Bins with zero observations fall back to uniform.
  - `"global"` — pooled empirical base composition broadcast to every bin.
  - `"uniform"` — `1/4` per base for every bin (legacy symmetric Laplace smoothing).
  - array-like `(total_bins, 4)` — explicit per-bin pi.

  The resolved prior is exposed as `model.prior_mode`, `model.prior_matrix`, and `model.marginal_fallbacks`, and is round-tripped through `gsynth_save`/`gsynth_load`. Legacy pickles backfill to `prior_mode='uniform'`. Matches R misha 5.6.21 (commits `1a49d803`..`e89b1738`).
- **`gsynth_score()`** -- score reference sequence under a trained model and write per-bp summed log-probability into a misha dense track. Supports `mask=` (NA-poisons output bins overlapping mask intervals), `resolution=` (default `model.iterator`), `n_policy={"NA","uniform"}`, `sparse_policy={"NA","uniform"}`, `overwrite=`, and stratified or 0D models. Bin lookup is aligned to `pos - k` (training convention). Matches R misha 5.6.21 commits `ba88e197` and `3fba28c2`.

### Fixes
- **`gsynth_sample` on 0D models past `model.iterator` bp.** The 0D `_extract_bin_data` path emits a single iter entry per input interval (covering `iter_size` bp), so positions past the first `iter_size` bp on a sample interval fell back to uniform random sampling instead of the trained CDF. The sample path now passes the `INT64_MAX` `iter_size` sentinel for 0D models (matching `gsynth_random`), so the single bin always resolves correctly. The previously-skipped regression test `test_0d_sample_on_non_first_chrom_uses_trained_cdf` now exercises this path under the marginal prior and passes.

### Notes
- **gquantiles perf fixes (R `625438a7`, `9970dbf5`) not ported.** PyMisha's `pm_quantiles` uses `StreamPercentiler::get_percentile` (sort-once on the reservoir, then index by position), so it never had the O(k * N) `nth_element` suffix walk that R 5.6.20 introduced and 5.6.21 reverted. Same reasoning for the per-kid sort + parent k-way merge optimisation -- the existing implementation already sorts per-kid at access time.
- **gintervals chrom-factor normalisation (R `ba88e197` part 1) not ported.** `pandas` interval frames use string `chrom` columns, so the R-only factor-level mismatch on bigset save/load doesn't have a pymisha analogue.

## v0.1.34 (2026-05-06)

### Fixes
- **`gsynth_sample` stratum bin lookup off by `k` bp at every iter-window boundary.** `pm_gsynth_sample` queried `bin_at(pos)` at the predicted-base position, but training attributes each `(k+1)`-mer event to `bin_at(pos - k)` (the leftmost base of the context window). At iter-window boundaries the sampler therefore picked up the *next* bin's CDF for the last `k` predicted positions, so cross-bin context dependencies were silently miscounted. Realigned the sample-time bin lookup to `pos - k` to match the convention used by `gsynth_train` and the cached `.gsm` model. Cached models stay valid; downstream tracks built from `gsynth_sample` should be regenerated. Matches R misha commit `3fba28c2`.

### Features
- **`gsynth_sample` and `gsynth_random` now preserve reference `N` positions by default (`preserve_n=True`).** Centromeres, telomeric Ns and other gap regions are written verbatim into the output instead of being filled with a fabricated ACGT base. `mask_copy` regions still take precedence. Pass `preserve_n=False` to recover the previous behaviour. Matches R misha #109 (commit `e90314be`).
- **`gintervals()` accepts BED-style character strand input.** Strand values can now be `"+"`, `"-"`, `"."`, `"*"`, or `""` (mapped to `1`/`-1`/`0`/`0`/`0`) in addition to numeric `-1`/`0`/`1`. Output remains numeric. Matches R misha #104 (commit `1de5131e`).
- **`gintervals_import_bed()`, `gintervals_import_gff()`, `gintervals_import_vcf()`.** Three new file-format importers that preserve common metadata columns and normalise chromosome names through the active database's chromosome-alias mechanism:
  - `gintervals_import_bed(file, name=True, score=True, strand=True)` -- BED is already 0-based half-open, coordinates are passed through. Optionally keeps the BED4/BED5/BED6 metadata columns.
  - `gintervals_import_gff(file, feature=None, strand=True, attrs=True)` -- GFF/GTF is 1-based closed; converts to 0-based half-open by subtracting 1 from `start`. Optional feature-type filter; keeps `source`, `type`, `score`, optional raw `attrs`.
  - `gintervals_import_vcf(file, info=True)` -- sets `start = POS - 1` and `end = POS - 1 + len(REF)`. Keeps `id`, `ref`, `alt`, `qual`, `filter`, optional raw `info`.

  Matches R misha #105 (commit `da592845`).

## v0.1.33 (2026-04-26)

### Fixes
- **`gsynth_sample` silently fell back to uniform-random sampling when intervals were not aligned to the iterator bin boundary.** `pm_gsynth_train` / `pm_gsynth_sample` inferred `iter_size` from the first same-chrom diff in `iter_starts`. For an interval whose start was not a multiple of the iterator (e.g. `iterator=200`, interval start at 64), `gextract` emits a partial first bin, so the inferred `iter_size` equalled the partial width (e.g. 136) instead of the true iterator. Every position past the partial bin then fell through `bin_idx = -1` and was sampled uniformly at random — including for k-mer constraints the caller explicitly tried to enforce. Matches R misha #94 (commit `02f7ad2f`). The fix:
  - `pm_gsynth_train` / `pm_gsynth_sample` now require the iterator as an explicit positional argument; non-positive values raise.
  - `GsynthModel` stores the training-time iterator (`model.iterator`) and `gsynth_sample` defaults to it when the caller does not pass `iterator=`.
  - `.gsm` save/load and legacy pickle load preserve / backfill `model.iterator`.

### Performance
- **Drop `MAP_POPULATE` from `MmapFile`.** `MmapFile::open()` previously requested `MAP_POPULATE` on Linux, which forces synchronous page-in of the entire mapped file on every `mmap` call. Per-chromosome track files trigger a fresh `mmap` on every chrom transition during expression evaluation, so multi-track multi-chrom queries paid full page-walk cost per track per chrom. `MADV_SEQUENTIAL` (still set) is sufficient to drive read-ahead for the bin-scan access pattern. Matches R misha #96 (commit `eb30be95`); R measured ~10× speedups on realistic motif-extract workloads.

## v0.1.32 (2026-04-19)

### Fixes
- **`gsynth_train` / `gsynth_sample` silently dropped non-first-chromosome intervals in 0D (unstratified) models.** `_extract_bin_data` hardcoded `iter_chroms` to zero for the 0D branch, so the C++ backend routed every iterator entry to `chrom_bins[0]` and left bins empty for every other chromosome. In training, k-mers on any chromosome other than chromkey ID 0 were silently uncounted (`chrX`-only training produced `total_kmers == 0`); in sampling, positions on such chromosomes fell back to `drand48() * 4` uniform random instead of the trained Markov CDF. Multi-dimensional models, `gsynth_random`, and `gsynth_replace_kmer` were not affected. Users who trained 0D (unstratified) models on intervals spanning multiple chromosomes should re-train and re-sample with this version.

## v0.1.31 (2026-04-16)

### Fixes
- **`pwm.max.pos` strand sign:** `DnaPSSM::max_like_match()` used to update `best_dir` at every iteration, so the returned strand reflected the last scanned position rather than the best-scoring one. Fixed to match R misha commit b7d469a6 (bug present since R misha v4.3.0). Added golden-master regression test against R misha for the signed position output.

## v0.1.30 (2026-04-15)

### Performance
- **ggenome_implant C++ fast path:** FASTA read/write/perturb loop now runs in C++ with 4 MB I/O buffers, matching the misha R package implementation.

## v0.1.29 (2026-04-15)

### Features
- **Genome editing:** Added `ggenome_implant()` and `ggenome_transplant()` for replacing intervals in a reference genome with donor sequences. Supports literal donor strings or extraction from a misha database, with optional trackdb creation and `.fai` index generation.

## v0.1.28 (2026-04-14)

### Fixes
- **mypy CI green:** Resolved 3 `no-any-return` errors from C++ bridge calls in `_crc64.py` and `sequence.py`.

## v0.1.27 (2026-04-14)

### Fixes
- **Example DB dotfiles in wheel:** Added `examples/**/.*` glob to package-data so `.attributes`, `.colnames`, and `.ro_attributes` files are included in wheels. Without these the bundled example DB was non-functional.

## v0.1.26 (2026-04-14)

### Features
- **Bundled example database:** `gdb_init_examples()` now works on any machine after `pip install pymisha`. The example trackdb is shipped inside the wheel under `pymisha/examples/`.

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
