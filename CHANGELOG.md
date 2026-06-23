# Changelog

## v0.8.18 (2026-06-23)

- **Listing interval sets no longer errors when a track is being removed concurrently.** `gintervals_ls` and dataset loading skip misha's transient `.trash.*` removal directories instead of failing with `FileNotFoundError` if one is unlinked mid-scan.

## v0.8.16 (2026-06-23)

- **`gintervals_quantiles` no longer collapses nearby percentile columns.** Percentile column names now use the shortest decimal that round-trips the value, so close percentiles (e.g. `0.123456789` and `0.1234567891`) keep distinct columns instead of both rounding to `0.123457` and silently dropping one. Common percentiles are unchanged (`0.5`, `0.95`, ...).

## v0.8.14 (2026-06-21)

- **`gextract` no longer truncates column names to 40 characters.** Auto-generated column names now use the full expression (e.g. the complete track name) instead of `<first 37 chars>...`. Long track names that previously collapsed to the same truncated column - silently dropping one - are now kept distinct. Pass `colnames=` to set names explicitly.

## v0.8.13 (2026-06-05)

Performance (all results unchanged):

- **BED and tabular track import are faster.** The common tab-delimited case is parsed with a vectorized reader instead of a per-row Python loop (~2.5x BED, ~3.2x tabular on multi-million-row inputs); space-delimited and non-standard files fall back to the original parser.
- **`gintervals_force_range` and `gintervals_normalize` are vectorized** - the per-row chromosome-boundary clamp is now a single numpy pass.
- **Loading indexed ("bigset") interval sets avoids a temporary file per chromosome.** Each chromosome shard is decoded directly from memory instead of being written to and re-read from a temp file (notably faster on networked filesystems).
- **`gsynth_score` is ~2x faster** - the per-base-pair log-probability lookup is a single vectorized gather instead of a Python loop.
- **2D track creation is slightly faster** - the quadtree writer uses a cheaper per-object NaN test.

## v0.8.12 (2026-06-03)

Performance (all results unchanged):

- **`gquantiles` / `gintervals_quantiles` are faster when the data fits in memory.** Percentiles are selected with `nth_element` over the reservoir instead of a full sort (~3x for typical percentile counts), and the multitasked path no longer re-sorts the merged reservoir on the parent.
- **Multi-chromosome `gextract` parallelizes by balanced base-pair ranges.** Large extracts (e.g. genome-wide over `gintervals_all()`) split the scope into bp-balanced chunks across workers - splitting a single large chromosome when needed - instead of one chunk per chromosome bounded by the largest chromosome. ~4x on a 3-large-chromosome extract with 6 processes (more on cold cache).
- **Multiple dense virtual tracks over the same source are read once.** When several virtual tracks share the same source track and `gvtrack_iterator` shift, the source is read once per output bin and shared across them instead of re-read per vtrack. ~1.7x when extracting 8 such vtracks together.

## v0.8.11 (2026-06-02)

- **Windowed virtual-track reductions over dense tracks are now incremental (R 5.10.2 parity).** A `sum`/`avg`/`nearest`/`lse`/`exists`/`size` virtual track with a sliding window (`gvtrack.iterator(sshift=, eshift=)`) scanned bin-by-bin now maintains a running window instead of recomputing the reduction from scratch at every step. The motif-energy workload (windowed `lse` reduced by `gquantiles`/`gscreen`) is ~8x faster on a 1000-bin window (hg38); results are unchanged. Set `PYMISHA_DISABLE_SLIDING_REDUCER=1` to force the legacy per-bin recompute.
- **`global.percentile` virtual tracks now match R bit-for-bit** - the windowed mean no longer differs from R by ~1 float32 ULP at quantile-break boundaries.
- **`exists` / `size` virtual tracks over a single all-NaN dense bin now return 0** (matching R) instead of 1.

## v0.8.10 (2026-06-01)

- **Documentation corrections (no code changes).** The R-parity guide and several capability notes were out of date. Fixed: COMPUTED 2D tracks are now documented as readable for `AreaComputer2D`/`TestComputer2D` (Hi-C `Potential`/`Technical` computers and creation remain unsupported); array tracks and the 2D iterator family work through the C++ scanner; array and 2D tracks support the indexed format; gene annotations install via `gdb_install_intervals`/`gdb_build_genome`. Removed stale "not implemented" notes for features that have shipped (`gvtrack.array.slice`, liftover, SAM import). Added a reproducibility note (NumPy vs R RNG; nearest-neighbor tie ordering).

## v0.8.9 (2026-06-01)

- **Per-chromosome dense tracks whose first chromosome has no data no longer error in `gtrack_info` / `gtrack_copy`.** A per-chrom dense track may legitimately lack a file for the genome's first chromosome (an empty scaffold, or a partial track). `gtrack_info` probed that chromosome to read the bin size and failed with "No such file or directory"; since `gtrack_copy` reads the source's info, copying such a track also failed. It now reads the bin size from the first chromosome that actually has data. (Per-chromosome companion to the v0.8.0 indexed-track fix.)

## v0.8.8 (2026-06-01)

- **`gtrack_2d_get_insu_doms` and `gtrack_2d_get_insu_borders`**: thin `gscreen` wrappers that extract TAD-style domains / borders from a 1D insulation track. R parity for the same-named test-helper idioms in misha (now public API in pymisha).
- **`gtrack_array_import(track, description, *srcs)`**: create an array track by merging one or more sources. Each source is either an existing array-track name or a tab-separated file with header `chrom\tstart\tend\t<col1>...`. Sources sharing an identical interval merge cell-wise; sources sharing a column name chain through a consistency check; partial overlaps raise. R parity for `gtrack.array.import`.
- **`gtrack_array_extract` gains `file=` parameter**: writes the result as a tab-separated table without the `intervalID` bookkeeping column, mirroring R's behaviour. The TSV is consumable by `gtrack_array_import`.
- **`gextract(file=)` no longer writes the `intervalID` column** (R parity). The in-memory `DataFrame` is unchanged; only the on-disk TSV is affected.

## v0.8.7 (2026-05-31)

- **`intervals_set_out=` on `gextract` / `glookup` / `gpartition` now stores value columns alongside the geometry.** The saved interval set is the full extraction frame minus the bookkeeping `intervalID`, matching R's `C_gextract` writer. Previously pymisha sliced to `chrom`/`start`/`end` (and deduplicated for `gextract`), so the loaded set lost `dense_track`, `bin`, lookup `value`, etc. - a round-trip drop that R never did.
- **`gintervals_load` now accepts a *track* name and returns its underlying geometry**, mirroring R's `.gintervals.load_file` track branch. 1D tracks yield per-bin `(chrom, start, end)`; 2D RECTS tracks yield per-rect `(chrom1, start1, end1, chrom2, start2, end2)`. `gintervals_intersect` / `_union` / `_diff` therefore now accept track names as either operand. Previously these calls raised "Intervals set does not exist".
- **`gintervals_update` now supports 2D interval sets via `chrom1=` / `chrom2=`.** Previously the 2D arms of R's `gintervals.update` (bigintervs2d, multi-chrom-pair updates) had no Python equivalent.

## v0.8.6 (2026-05-31)

- **CI is green again on `main`.** Three patches needed since v0.8.3 left the suite red:
  - **Reverted v0.8.3's "2D-source value vtrack iterator defaults to source rects".** That flip won one r_parity case (`test_real_k562_vtrack_weighted_sum`, now re-xfailed) but silently changed the output shape of every 2D-source vtrack (`area`, `weighted.sum`, `exists`, `size`, `first`, `last`, `sample`, `min`, `max`, `avg`) - too breaking with only a single r_parity test to anchor the right per-func behaviour. The pre-v0.8.3 contract is restored: `gextract("v", scope_2d)` over a 2D-source vtrack returns one row per scope interval with the aggregated value. Pass an explicit `iterator=` to opt in to per-rect iteration.
  - **`_read_file_header` API reverted to returning `(is_points: bool, num_objs, data)`.** v0.8.4 widened it to `(kind: str, ...)` which broke external test callers that pass `is_points` straight through to `query_2d_track_opened` (the C++ binding does a truthy check). COMPUTED tracks share the 48-byte RectObj layout with RECTS, so `is_points = False` is the right answer there. A new lightweight `_file_track_kind(path)` helper sniffs the signature for callers that specifically need `"COMPUTED"`.
  - **`test_computed_tracks.py` stub now writes `CT2_POTENTIAL`** (=1) instead of the all-zeros byte (which is `CT2_AREA` - supported since v0.8.4). The file's intent - "COMPUTED tracks with an unsupported computer must raise NotImplementedError" - is preserved.

## v0.8.5 (2026-05-31)

- **COMPUTED 2D tracks now clip stored rects to the query and recompute the cached value when needed.** R's `Computed_val<T>::val(rect, band)` returns the cached `v` only when `rect` exactly matches the stored coords (and the band contains it); otherwise it recomputes via `Computer2D::compute(rect, band)`. Pymisha's 2D extract now clips each emitted stored rect to (query ∩ band) via the R-parity `DiagonalBand::shrink2intersected` and routes mismatching cases through `recompute_or_cached`, mirroring `update_stat`. `TestComputer2D` mod operator now uses C-style truncation (truncation toward zero, sign of dividend) - matters when band-shrinking produces negative coordinates. `DiagonalBand.do_contain` + `intersected_area` corrected to mirror R's source exactly. Flips r_parity gextract.50 (`2 * test.computed2d + 17`), gextract.55 (band query), gextract.58 (scope-clipping at chrom boundaries), gextract.43 (iterator=COMPUTED + multi-expression).

## v0.8.4 (2026-05-31)

- **COMPUTED 2D tracks backed by `AreaComputer2D` / `TestComputer2D` are now readable.** `gextract`, `gsummary`, `gquantiles`, and related routines used to raise `NotImplementedError` on any track of type `"computed"`; pymisha now parses the COMPUTED file format (signature -11), dispatches the embedded Computer2D header (`CT2_AREA` / `CT2_TEST`), and reads the StatQuadTreeCached payload via the existing RECTS machinery (Computed_val<float> on-disk layout matches RectObj). Aggregation-only paths (`gsummary`, `gquantiles`, scope-level `gscreen`) consume cached node stats and return R-parity values for these tracks. `PotentialComputer2D` / `TechnicalComputer2D` (Hi-C normalisation) and the per-rect compute-on-mismatch fallback (used when an iterator emits clipped or grid-binned rects whose coords don't exactly match a stored rect) remain deferred.

## v0.8.3 (2026-05-31)

- **A value-based 2D vtrack with no explicit iterator now iterates the source 2D track's rects.** `pm.gvtrack_create("v", "hic.track", "weighted.sum"); pm.gextract("v", scope_2d)` (no `iterator=`) used to fall through to one-row-per-scope-interval aggregation; it now defaults the iterator to the source 2D track's rectangles (one row per source rect in the scope), matching R's per-vtrack default iterator. Pass `iterator=scope_2d` to keep the prior one-row-per-scope-interval behavior. 1D iterator shifts (`gvtrack_iterator(sshift=, eshift=)`) set on a 2D-source vtrack are now rejected by the scanner path too (the legacy path already rejected them).
- **Mixing a 1D dim-projected vtrack with a bare 2D track under a 2D scope now projects each rect.** `pm.gextract(["dist1_dim1", "dist2_dim2", "hic.track"], scope_2d)` used to compute the dim-projected vtracks once for the whole scope and broadcast that single value to every rect row; it now defaults the iterator to the lone 2D track in the expression and projects each rect's `[start1,end1]` / `[start2,end2]` independently per row, matching R's implicit-iterator behaviour for a 2D-track-bearing expression.
- **`gintervals_neighbors` on 2D intervals now normalizes chrom names on both sides.** Passing two 2D-interval DataFrames whose `chrom1` / `chrom2` columns disagreed on the `chr` prefix (one carried it, the other didn't) silently produced `None` because the per-chrom-pair grouping is a literal string compare; both inputs are now normalized through the same loader path as the 1D `gintervals_neighbors`, so a query of e.g. `("1", ...)` matches targets of e.g. `("chr1", ...)`.

## v0.8.2 (2026-05-31)

- **`gintervals_neighbors` on 2D interval sets now handles large unbounded inputs.** A nearest-neighbor query without a `maxdist*` window on intervals beyond a few million rects previously raised `NotImplementedError`; it now runs an in-memory quadtree NN iterator in C++ (port of R misha's `StatQuadTree::NNIterator`), matching R's per-axis-unsigned-gap geometry. R's `priority_queue` tie-break order on equidistant `maxn=1` neighbors is not portable across STL implementations and is not replicated; query rows that hit such ties may pick a different (but equidistant) neighbor. 2D track names passed as `intervals1` / `intervals2` are now materialized via `gextract` over the 2D ALLGENOME(full) scope (R parity).

## v0.8.1 (2026-05-31)

- **2D tracks built with NaN-valued rectangles now retain the NaN rects, matching R.** `gtrack_lookup` with `force_binning=False` on a 2D source, and any other path constructing a 2D RECTS track from NaN-bearing values, previously silently dropped NaN rects so the resulting track was incomplete (an off-diagonal `gtrack_lookup` query returned `None`); they now write the NaN rects to disk, exclude them from per-bin stat aggregations (avg / min / max / sum / weighted.sum), and `gextract` returns all rects including NaN values. `gtrack_lookup` also now evaluates the 2D scope under `mode="full"` so off-diagonal chrompair queries return data.

## v0.8.0 (2026-05-29)

- **`distance` / `distance.edge` / `distance.center` virtual tracks now return the true nearest source interval on overlapping or nested sources.** The previous scan could return a non-nearest interval, even a nonzero distance for a query that overlaps a source (e.g. against `rmsk`-style nested intervals). All three now use the same nearest-neighbor index as `gintervals_neighbors`, so `distance.edge` matches it exactly. `distance.center` also no longer errors when the source intervals overlap: a bin center inside several intervals resolves to the nearest center.
- **`gdb_install_intervals` / `gdb_build_genome` now attach `name` and `geneName` columns to the installed `tss` / `exons` / `utr3` / `utr5` sets.** `name` is the transcript accession; `geneName` is the gene symbol (from the `gene_name` attribute, falling back to `gene_id`, blank when the source has neither). Overlapping features that unify to one interval concatenate their distinct symbols with `;`.
- **Indexed dense tracks whose first chromosome has no data no longer report `bin_size = 0` or crash on read.** A packed/converted indexed track whose leading chrom has an empty index entry left the bin size uninitialized, so `gtrack_info` reported `bin_size = 0` and subsequent reads divided by zero. The bin size is now back-filled from the first non-empty chromosome.

## v0.7.1 (2026-05-28)

- **2D `gextract` with a diagonal `band` now clips emitted rectangle coords to the band-intersected area.** Each contact rectangle returned by a 2D extract (raw track or scanner-driven aggregation over an intervals iterator) is now shrunk to the smallest bounding box containing its intersection with the band, matching R's `DiagonalBand::shrink2intersected` (so e.g. a 10kb-bin contact on the diagonal with `band=(-1024, 1024)` keeps its on-diagonal slice instead of returning the full bin). Previously pymisha returned the raw stored coords. Inter-chrom rectangles under an active band are now skipped, matching R.
- **`weighted.sum` (and other 2D stats) over a near-diagonal band now agree with R.** The band-intersected area formula used a wrong corner (`y1` instead of `y2`) in the d1-triangle subtraction, over-reporting area by ~3x on contact bins that straddle the diagonal. Fixed to match R's `DiagonalBand::intersected_area`.
- **2D `gextract` over a 2D-intervals scope/iterator now normalizes chromosome names (`chr1 ↔ 1`) before looking them up.** Previously the C++ scanner path used a raw chrom-name lookup, so passing a DataFrame whose `chrom1`/`chrom2` carried a `chr` prefix against a DB that stores chroms without it (or vice versa) produced `chromid=-1` and combined with a registered virtual track could crash with `std::bad_alloc`. The 1D `gextract` path already normalized; the 2D scanner path now does too.
- **`gintervals_2d` recycles a length-1 axis against a length-N axis.** `gintervals_2d("chr1", starts, ends)` (axis2 omitted) now produces N rows of `(chr1, starts[i], ends[i]) × (chr1, 0, chrom_size)`, matching R's `data.frame`-style recycling. Previously this raised `ValueError("chroms1 and chroms2 must produce the same number of intervals")` when axis1 was a vector and axis2 fell back to chromosome-wide defaults. Equal-length 1D-vector arguments continue to pair positionally.

## v0.7.0 (2026-05-28)

- **`gextract` now honors a 2D-intervals iterator (DataFrame or interval-set name).** `gextract("rects_track", scope_2d, iterator=other_2d_intervals)` (or `iterator="my_2d_set"`) iterates the rectangles of `iterator ∩ scope` and evaluates the expression on each (with `intervalID` attributing each row to its scope interval), matching R's 2D intervals iterator. Previously the 2D iterator was ignored and the whole scope was object-enumerated.
- **`giterator_intervals` over a bare 2D track now visits all chrom pairs.** `giterator_intervals("rects_track")` (no scope) now enumerates every rectangle of the track, not just the intra-chromosomal (diagonal) ones, matching R's whole-genome 2D scope. It also accepts a 2D-intervals DataFrame or interval-set name as the `iterator` (returning `iterator ∩ scope` cells), and a 2D-track name as the scope.
- **`gintervals_summary` now supports a 2D-intervals iterator (DataFrame, interval-set name, or 2D-track name).** `gintervals_summary("rects_track", scope_2d, iterator=other_2d_intervals)` summarizes the expression over the iterator cells within each scope interval (one row per scope interval), matching R.
- **`gintervals_neighbors` now supports 2D intervals.** For two 2D-interval sets it finds, per query rectangle, the nearest target rectangles on the same chrom-pair within a per-axis distance window (`mindist1`/`maxdist1`/`mindist2`/`maxdist2`), ordered by Manhattan distance, returning `dist1`/`dist2` columns - matching R. Previously any 2D input raised `NotImplementedError`. (A nearest-neighbour search over very large *unbounded* sets still needs a scalable quadtree NN iterator and raises a clear error.)
- **`gintervals_2d_intersect` is now scalable.** The pairwise rectangle intersection is computed per chrom-pair with an in-memory quadtree instead of an `O(n1·n2)` broadcast, so intersecting large 2D screens (e.g. 10^5 × 10^5 rectangles on one chrom-pair) no longer exhausts memory. Results are unchanged.

## v0.6.0 (2026-05-27)

- **Array tracks now work in the track-expression scanner.** A `array` track can be used directly in a track expression (`gextract("my.array", ...)`, `gextract("2 * my.array + 17", ...)`), mixed with dense/sparse tracks under an explicit iterator, as the iterator itself (`iterator="my.array"`, one bin per row), and as the source of a value-based virtual track (`gvtrack_create("v", "my.array", "min")`). Each bin's columns are reduced to a scalar (default: average over all columns) which is then aggregated over the iterator interval, matching R. Previously any array track in an expression or as an iterator raised "scanner does not support array tracks".
- **`gvtrack_array_slice` virtual tracks now evaluate through the fast C++ scanner and support a column quantile.** Selecting a column subset and reduction (`gvtrack_array_slice("v", ["col1", "col3"], func="max")`) is now computed in C++ and, with no explicit iterator, iterates the array's native bins (one value per bin), matching R exactly - including R's float32 standard-deviation accumulation. `func="quantile"` with `params=<percentile>` is now accepted (previously raised `NotImplementedError`).

## v0.5.2 (2026-05-27)

- **`giterator_intervals` now accepts a numeric 2D iterator.** `giterator_intervals(expr, scope, iterator=(width, height), band=...)` over a 2D scope enumerates the fixed-size grid cells (clipped to the scope and the optional diagonal `band`), matching R's `giterator.intervals(expr, scope, iterator=c(width, height))`. Previously a two-element numeric iterator over a 2D scope raised "intervals must have 'chrom', 'start', and 'end' columns".
- **A dim-projected 1D virtual track now honors a 2D iterator.** `gextract("v", scope_2d, iterator="rects_track")` for a 1D vtrack with `gvtrack_iterator(dim=1/2)` now iterates the iterator's 2D cells - projecting each onto the chosen axis and evaluating the vtrack there (one row per iterator cell), matching R. Previously the iterator was ignored and the vtrack was evaluated once per 2D scope interval.

## v0.5.1 (2026-05-27)

- **`gtrack_create` can now build a 2D track from a 2D track expression.** `gtrack_create("t", desc, "rects_track + 10")` (and with a 2D-intervals `iterator`) evaluates the expression over the source track's rectangles and writes a 2D RECTS/POINTS track, matching R's `gtrack.create` on a 2D expression. Previously the 1D scanner could not iterate a rectangles track and no usable track was produced.
- **`giterator_intervals` now accepts a cartesian-grid iterator.** `giterator_intervals(expr, scope, iterator=giterator_cartesian_grid(..., stream=True), band=...)` enumerates the 2D grid cells over the scope, matching R's `giterator.intervals(expr, scope, iterator=cartesian_grid)`. The cells are built with grid-point (center) de-duplication and adjacent-center midpoint clipping, intersected with the 2D scope, and filtered by the optional diagonal `band`. A 2D-track name or `None` (the whole 2D genome) is accepted as the scope.
- **Single-strand spatial PWM scores no longer double-count the reverse strand.** For a non-bidirectional PWM virtual track (`bidirect=False`) with spatial weighting (`spat_factor`/`spat_bin`), the spatial sliding-window optimization scored both strands when no strand was explicitly selected, inflating `pwm`/`pwm.max` energies (only at the first interval of each scan segment). It now scores a single strand, matching R. Bidirectional and non-spatial scoring were unaffected.
- **`global.percentile.min` / `global.percentile.max` virtual tracks now match R exactly.** They map each per-bin statistic through the source track's precomputed `vars/pv.percentiles` binned quantile table (the same file R uses) instead of an exact empirical CDF, so the returned percentile is bit-for-bit identical to R for R-created tracks. Tracks without that file (e.g. pymisha-created) fall back to the empirical CDF as before.
- **`gintervals_neighbors` now signs the distance by the target's strand.** When the second (target) interval set has a `strand` column, the reported distance is negative for upstream neighbors and positive for downstream ones (as in R); previously the target's strand was ignored and every distance was unsigned (positive), so a signed distance window such as `mindist=-10000, maxdist=-2000` matched nothing and returned `None`. Target sets without a `strand` column are unchanged (unsigned distance).
- **Values landing exactly on a bin break are now assigned to the lower bin.** Binning is right-closed `(breaks[i], breaks[i+1]]` as in R, so a value equal to a break belongs to the bin ending at it, not the one starting at it. Previously such values went one bin too high. This was invisible for continuous track data but mis-binned integer values (e.g. `gcis_decay` distances against multiple-of-1000 breaks), and also affected `gbins_summary`/`gbins_quantiles`/`gdist` over virtual tracks.

## v0.5.0 (2026-05-26)

- **A value-based virtual track with no explicit iterator now uses its source track's native iterator.** `gextract("vt", intervals)` for a vtrack over a dense track previously evaluated one value over each whole input interval; it now iterates the source track's native bins (applying any `gvtrack_iterator` shifts), matching R.
- **`gintervals_summary` and `gintervals_quantiles` now return one row per scope interval when the iterator is a track or interval-set.** Previously, passing a track/interval-set name (or a DataFrame) as the `iterator` produced one row per (iterator-bin ∩ scope) piece and silently dropped scope intervals that contained no bin. They now emit exactly one row per input scope interval - empty intervals get `Total intervals = 0` and `NaN` statistics - matching R. The per-interval standard deviation also uses R's one-pass formula for exact parity.
- **`gintervals_diff` now canonicalizes its inputs first.** When either operand contained overlapping intervals (e.g. an `rbind`/`concat` of two screen results), the difference double-counted the overlaps and returned too much genomic space. Both operands are now sorted and overlap-unified before the difference, matching R.
- **`giterator_intervals` gains `band` and `intervals_set_out` arguments.** A diagonal `band=(d1, d2)` now restricts a 2D iterator to rectangles whose offset from the diagonal falls within the band (as in `gextract`), and `intervals_set_out="name"` saves the resulting iterator intervals as a named set (returning `None`), matching R's `giterator.intervals(..., band=, intervals.set.out=)`.
- **`distance.center` virtual tracks no longer return `NaN` for some bins whose center lies inside a source interval.** When several source intervals began within a single iterator bin, the search inspected only the first one and could miss the interval actually containing the bin center, returning `NaN` instead of the true distance. It now locates the containing interval by the center coordinate, matching R misha.
- **Sparse-track `stddev` is now numerically stable.** Virtual tracks and extractions using `func="stddev"` over a sparse track previously used a one-pass formula that produced a tiny nonzero value (e.g. `1e-4`) for a bin of identical large-magnitude values, where the true standard deviation is `0`. They now use the Welford online algorithm (matching R misha and the dense-track path), returning exactly `0` for constant bins.
- **`gvtrack_create` with an intervals-set source now defaults to `func="distance"`.** When `func` is omitted and the source is an intervals set (a DataFrame with no value column, or an intervals-set name), the virtual track now measures distance to the nearest interval, matching R. Previously it defaulted to `"avg"` and raised "DataFrame source must include one value column". Track and value-bearing sources still default to `"avg"`.

## v0.4.0 (2026-05-26)

- **Track expressions now follow R operator precedence for `&` and `|`.** Expressions such as `gscreen("track > 0.1 & track < 0.3")` previously raised a bitwise-operator error because `&`/`|` bound tighter than the comparisons. They now bind looser, as in R, so `a > x & b < y` means `(a > x) & (b < y)` (and `&` binds tighter than `|`). Affects `gscreen`, `gextract`, `gsummary`, `gdist`, `gquantiles`, `gpartition` and any other track-expression evaluation.
- **Indexed sparse tracks no longer read back empty.** `gextract`, `gsummary` and value-based virtual tracks returned no data (and `gsummary` reported 0 intervals) for sparse tracks stored in the indexed format; dense tracks were unaffected. They now read correctly.
- **The iterator and scope may be a track or interval-set name.** `gextract(expr, intervals, iterator="some.track")` iterates over that track's bins (dense) or intervals (sparse); a track or interval-set name passed as `intervals` uses its intervals/rectangles as the scope; `giterator_intervals` infers the grid from a sparse track. Previously only a numeric bin size or an explicit intervals DataFrame was accepted.
- **Bare 2D expressions no longer require explicit intervals.** `gscreen("rects_track > 10")`, `gsummary("rects_track")` and similar now default to the whole 2D genome (as in R) instead of raising "rectangles not yet supported".
- **`gintervals` / `gintervals_2d` recycle shorter argument vectors.** `gintervals([1, 2], starts, ends)` with longer `starts`/`ends` now repeats the chromosome list to match (as in R), instead of raising "chroms, starts, and ends must have the same length". Lengths must still be multiples.
- **`gintervals_chrom_sizes` now returns interval counts per chromosome.** It returns `chrom`/`size` for 1D intervals and `chrom1`/`chrom2`/`size` for 2D (the number of intervals on each chromosome or chromosome pair), matching R. Previously it returned only the unique chromosome names with no counts. It also accepts an interval-set or track name directly.
- **`global.percentile` virtual tracks no longer read back empty on indexed databases.** `gvtrack.create(..., func="global.percentile" / "global.percentile.min" / "global.percentile.max")` returned all-`NaN` for source tracks stored in the indexed format; these functions use a read path whose per-chromosome-file gate skipped indexed tracks (which keep their data in `track.dat`). They now read correctly. Per-chromosome databases were unaffected.
- **`gtrack_array_extract` no longer returns an empty result for array tracks stored in the indexed format.** The array reader only knew the legacy per-chromosome files, so on an indexed database it found no data and returned zero rows (column names still read correctly). It now also reads each chromosome's block from `track.dat`/`track.idx`. Per-chromosome databases were unaffected.

## v0.3.0 (2026-05-25)

- **Fixed a correctness bug in coarsening-iterator extraction.** When a numeric `iterator` larger than a dense track's native bin size was used, `gextract`, `gsummary`, `gquantiles`, `gbins_summary` and `gbins_quantiles` sampled the value at each output bin's midpoint instead of averaging the native bins it covers, returning wrong values. They now average, matching R misha. Extraction with the default (native) iterator was unaffected.
- **Mixed dense and sparse tracks in one expression are now supported.** Expressions such as `gextract("dense_track + sparse_track", intervals, iterator=50)` work with an explicit iterator, and (as in R) raise when no iterator can be inferred. Previously any dense+sparse mix raised "Mixed track types in expression are not supported".
- `gintervals_canonic`, `gintervals_intersect`, `gintervals_union` and `gintervals_diff` now reject intervals with `start >= end` or `start < 0` instead of silently dropping zero-width intervals or passing inverted intervals through unchanged.
- `gintervals_neighbors(..., na_if_notfound=True)` returns `NaN` for the start/end coordinates of rows with no neighbor, matching R (previously a `-1` sentinel).
- Fixed two small reference leaks (result-dict float objects in `gtrack_import_mappedseq` and the strands-autocorrelation routine).
- `gsummary` and `gintervals_summary` no longer return `NaN` for the standard deviation of near-constant, large-magnitude input; the tiny negative variance produced by catastrophic cancellation is clamped to 0 (matching R misha 5.7.4).
- `gsynth_random` now validates `nuc_probs`: a missing, extra, or duplicated nucleotide name is rejected with a clear error instead of being silently mishandled.

## v0.2.4 (2026-05-25)

- `gtrack_import()` gains a `func` argument selecting the per-bin reduction when importing with a `binsize` (Dense track): one of `"weighted.mean"` (default), `"weighted.sum"`, `"max"`, `"min"`, `"median"`, `"count"`, `"coverage"`. For example `gtrack_import("pileup", desc, "reads.bed", binsize=20, func="coverage")` builds a ChIP-style pileup track in one call. Works for every input format (BED/WIG/bedGraph/BigWig/tab). Passing a non-default `func` without a `binsize` raises an error.

## v0.2.3 (2026-05-21)

- **CI green again on main**. 15 mypy errors in `pymisha/liftover.py` were silently introduced during the G1 liftover C++ port (v0.1.87-v0.1.94) and only surfaced once dev shipped to main as v0.2.0. All fixed: `# type: ignore[no-any-return]` on the two C++-entry returns (`pm_parse_chain_file`, `pm_chain_intervals_resolve`), explicit `float()` / `int()` conversions on the aggregate reducers, and an inner-loop variable rename (`m` -> `mc`) that was shadowing the outer median-index `m`.
- **Conda package builds again**. `conda-recipe/meta.yaml` now lists `zlib` in `host` and `run` requirements (broken since v0.2.0 added zlib for SAM gzip support).

## v0.2.2 (2026-05-21)

- Internal: BAM auto-detect for `gtrack_import_mappedseq` moved from the Python wrapper into the C++ entry. `samtools view` is now spawned via `popen()` inside `pm_import_mappedseq`; the Python side no longer manages a subprocess or dups file descriptors. User-visible behavior is unchanged. This makes the architecture symmetric with the upcoming R misha port (R can't easily share fds with C, but both languages can let C drive `popen`).
- The C++ side surfaces a clear `pymisha.error` when `samtools` is missing (exit code 127) or when `samtools view` exits non-zero. The exception type changed from `RuntimeError` to `pymisha.error` - both are catchable as `Exception`.

## v0.2.1 (2026-05-21)

- `gtrack_import_mappedseq` now accepts BAM files directly: bgzip magic bytes (`1f 8b 08 04`) are auto-detected and the file is streamed through `samtools view` into the existing C++ FSM. Requires `samtools` on PATH; a clear `RuntimeError` is raised otherwise. The legacy default `cols_order=(9, 11, 13, 14)` is silently switched to SAM mode (`None`) for BAM inputs since `samtools view` always emits SAM-format payload.
- New stdin / fd source in the C++ FSM: `file="-"` reads from stdin and `file="fd:N"` reads from an arbitrary file descriptor. Lets users compose pipelines such as `samtools view -q 30 reads.bam | python -c "pm.gtrack_import_mappedseq(...)"` or pre-filter via `samtools markdup` / region restriction.

## v0.2.0 (2026-05-21)

Bundle release covering v0.1.75 through v0.1.95. Major themes:

- **Liftover ported to C++ end-to-end** (G1, v0.1.87-v0.1.94). `_aggregate_overlapping`, chain-file parser, source-track reader (1D dense / sparse / indexed + 2D RECTS / POINTS), overlap-policy resolution, `_map_intervals`, the `gtrack_liftover` orchestrator, and the dispatcher all run in C++ with R-parity semantics. ARRAYS source tracks still wait on G3. Env-var fallbacks: `PYMISHA_FORCE_PY_AGGREGATE_OVERLAPPING`, `_PARSE_CHAIN_FILE`, `_READ_SOURCE_TRACK`, `_CHAIN_INTERVALS_RESOLVE`, `_MAP_INTERVALS`, `_LIFTOVER_TRACK`.
- **SAM import ported to C++** (G2, v0.1.95). `pm_import_mappedseq` per-byte FSM + direct dense / sparse writers. 3-5x on 100k synthetic SAM. R-parity: exact chrom-name match, tab-only split. Gzip auto-detect retained. `PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ=1` keeps the Python path.
- **2D scanner + virtual-track R-parity** (v0.1.75-v0.1.85). FixedRect, TrackRects, CartesianGrid iterators. Opt-in scanner routing for the intervals iterator. Reducing 2D vtracks, scanner-side `exists` / `size` / `first` / `last` / `sample` object functions. ARRAYS branch in the C++ scanner + `gvtrack_array_slice`. 2D vtrack shifts through the scanner. Multi-track 2D compound expressions (Spec C).
- **LLM agent guides** (`agent-guides/`) ported from the R misha guides. Excluded from the sdist; designed for raw-github-URL fetch by downstream agents.
- **Misc**: `gtrack_create / _modify / _smooth` now resolve vtracks in expressions (v0.1.84); various R-parity corrections for 2D vtracks (`exists` / `size` NaN handling, v0.1.81); CI green on main (mypy + test isolation, v0.1.86).

R parity is the spec wherever the prior Python implementation diverged. See individual v0.1.x entries below for per-release details.

## v0.1.95 (2026-05-21)

- `gtrack_import_mappedseq` ported to C++ (`pm_import_mappedseq`): per-byte
  FSM SAM/tab parser + direct dense / sparse track writers. Dense speedup
  3.34x and sparse speedup 5.41x on a 100k-read synthetic SAM (measured
  vs. the prior pure-Python implementation). R-parity: chromosome names
  must match the DB exactly (no normalization), tab is the only field
  separator. Gzip auto-detect via the magic bytes is retained.
  `PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ=1` selects the Python fallback.

## v0.1.94 (2026-05-21)

- `gtrack_liftover` now supports 2D source tracks (RECTS and POINTS) end-to-end in C++ via the new `pm_liftover_track_2d` entry point. The dispatcher auto-detects 2D sources by quadtree signature and routes them through the dedicated 2D path. No aggregation is performed on the 2D side, matching R behavior. ARRAYS source tracks are still out of scope.

## v0.1.93 (2026-05-21)

- `gtrack_liftover` now preserves source track type: dense (FIXED_BIN) source produces a dense target track; sparse source produces sparse. Previously always produced sparse. This is a breaking change for code relying on the always-sparse output; aggregation semantics now match R for both paths.
- `gtrack_liftover` runs end-to-end in C++ via the new `pm_liftover_track` entry point (~3.56x faster on a 1M-bin + 10k-chain workload). Set `PYMISHA_FORCE_PY_LIFTOVER_TRACK=1` to fall back to the pure-Python path.

## v0.1.92 (2026-05-21)

- `gintervals_liftover`: ~1.7x end-to-end speedup on 100k src x 100k chain.
- `best_cluster_*` policies now union by `chain_id` AND source overlap,
  matching R behavior. Fixes a divergence where multi-block chains were
  incorrectly split into multiple clusters.
- Set `PYMISHA_FORCE_PY_MAP_INTERVALS=1` to fall back to the pure-Python
  liftover path.

## v0.1.91 (2026-05-20)

### Fixes (R-parity)
- `gintervals_load_chain` now matches R `misha` behavior on two edge cases that v0.1.90 (and earlier) handled differently:
  - `src_overlap_policy="discard"` uses a pair-only scan after sort by source coordinates (matching R `rdbinterval.cpp` `handle_src_overlaps`). Previously this was a whole-cluster discard that also dropped non-overlapping intervals nested inside a wider overlap. A nested interval separated by a gap from the prior overlap pair is now kept.
  - `tgt_overlap_policy="auto_score"`/`"auto_first"`/`"auto_longer"`/`"agg"` no longer collapse the split pieces of a negative-strand chain back into a single row via `min(startsrc)` / `max(endsrc)`. The merge step now requires `prev.endsrc == slice.startsrc` (matching R `append_slice`), so negative-strand chains that were split by an overlapping target chain stay as N separate rows with their reversed source coordinates intact.

## v0.1.90 (2026-05-20)

### Performance
- Chain overlap-policy resolution ported to C++ (~22x faster on a 100k-row synthetic chain under `auto_score`). `gintervals_load_chain` picks up the speedup automatically for all `src_overlap_policy` / `tgt_overlap_policy` combinations. Set `PYMISHA_FORCE_PY_CHAIN_INTERVALS_RESOLVE=1` (or pass `_force_pure_python=True`) to fall back to the pure-Python implementation.

### Fixes
- C++ tgt-overlap sweep now skips zero-length intervals (matching the Python `np.unique`/coverage path) instead of emitting a phantom slice spanning to the next event.

## v0.1.89 (2026-05-20)

### Performance
- `gtrack.liftover` source-track reader (`_read_source_track`) ported to C++. Per-chrom dense, per-chrom sparse (32-bit + 64-bit record layouts), and indexed `track.idx`/`track.dat` formats all decode directly into numpy arrays. ~6x speedup on a 1M-bin synthetic dense track. Set `PYMISHA_FORCE_PY_READ_SOURCE_TRACK=1` (or pass `_force_pure_python=True`) to fall back to the pure-Python reader.

### Fixes
- Indexed source tracks with sibling subdirectories (e.g. the `vars/` directory created by `gtrack.convert_to_indexed`) were silently being read as empty sparse tracks. Subdirectories are now filtered out of the per-chrom file list, so the indexed branch runs as intended.

## v0.1.88 (2026-05-20)

### Performance
- `gintervals_load_chain` is ~5x faster on large chain files. Set `PYMISHA_FORCE_PY_CHAIN_PARSER=1` to fall back to the pure-Python parser.

## v0.1.87 (2026-05-20)

### Performance
- `gtrack.liftover` overlap aggregation ported to C++. The per-chrom sweep-line in `_aggregate_overlapping` now runs through `_pymisha.pm_liftover_aggregate`. Pure-Python fallback retained for custom aggregator callables.

### Fixes
- `_aggregate_overlapping` now iterates the active set in row order (was hash order, non-deterministic by value). Affects `first`/`last`/`nth` aggregators.

## v0.1.86 (2026-05-20)

### Fixes
- **CI green.** Mypy now passes (`pymisha/extract.py` policy-dict and CartesianGrid resolver were inferred as `dict[str, str]` and `object`; both annotated explicitly. `pymisha/vtracks.py::gvtrack_array_slice` slice-list narrowing was tightened). The `test_computed_tracks` module fixture re-inits the canonical test db before writing the COMPUTED stub so the on-disk file matches the active db pointer regardless of which prior test last called `gdb_init_examples()`. No runtime behavior change.

## v0.1.85 (2026-05-20)

### Features
- **Multi-track 2D compound expressions (R parity).** `gextract("v_a + v_b", ...)`, `gextract("track_a - track_b", ...)`, and any compound 2D expression mixing bare 2D tracks and reducing vtracks now route through the C++ scanner. Each symbol is computed once per rectangle, then the compound expression is evaluated over the per-symbol arrays. Works with `iterator=(N, M)`, `iterator="<2D track>"`, `iterator=CartesianGridSpec(...)`, and the opt-in `PYMISHA_USE_SCANNER_FOR_INTERVALS=1` intervals-iterator path. Previously raised `NotImplementedError` or silently fell through.

## v0.1.84 (2026-05-20)

### Fixes
- **`gtrack_create`, `gtrack_modify`, `gtrack_smooth` now resolve virtual tracks in their `expr` argument.** Previously these handed the raw expression string to the C++ engine without forwarding the Python-side vtrack registry, so any expression referencing a `gvtrack_create`-registered name failed with `name '<vtrack>' is not defined`. The canonical R misha pattern `gvtrack.create(...) ; gtrack.create("smoothed", ..., expr="log2(vt_sum + 1)", iterator=20)` now works in pymisha.

## v0.1.83 (2026-05-20)

### Features
- **2D vtrack shifts now work through all scanner paths (R parity).** `gvtrack_iterator_2d(name, sshift1=, eshift1=, sshift2=, eshift2=)` shifts are now applied per-var when querying the underlying 2D track, for `gextract` calls using `iterator=(N, M)`, `iterator=track_name`, or `iterator=CartesianGridSpec(...)`. Previously any vtrack with non-zero shifts fell back to the slow legacy path or raised `NotImplementedError` with FixedRect/CartesianGrid iterators.

### Limitations
- `global.percentile` is not routed through the scanner (deferred - requires a two-pass population accumulation). It continues to work via the legacy path for plain `gextract` calls, but raises `NotImplementedError` with FixedRect/CartesianGrid iterators.

## v0.1.82 (2026-05-20)

### Features
- **`gvtrack_array_slice(vtrack, slice, func)` - R-aligned API for `gvtrack.array.slice`.** Mutates an existing vtrack (created via `gvtrack_create(name, src=array_track)`) rather than creating one. Matches R's two-step pattern: `gvtrack_create` then `gvtrack_array_slice`. Supports all R functions: `avg`, `min`, `max`, `sum`, `stddev`. `gextract("vtrack_name", intervals=...)` on such a vtrack returns one scalar per iterator interval.
- **Array-slice vtracks work in `gextract` without an explicit iterator.** When all expressions are array-slice vtracks and no physical track is present, the query intervals serve as the iterator (one row per input interval).

### Limitations
- Bare `gextract("array_track", ...)` (without a vtrack) still raises, pointing to `gtrack_array_extract`. Use `gvtrack_array_slice` to aggregate first.
- `quantile` func not supported (requires a different data path); deferred.

## v0.1.81 (2026-05-20)

### Fixes
- **`exists` and `size` 2D vtracks now return 0 (not NaN) for chrom pairs with no track data.** R parity. Affects `gextract(vtrack, intervals=scope, iterator=...)` where the iterator emits cells on pairs the track does not cover.

## v0.1.80 (2026-05-20)

### Features
- **Scanner supports object-level reducer funcs (R parity).** `PMTrackExpression2DVars` now handles `exists`, `size`, `first`, `last`, `sample` for use with 2D iterators. `gextract(vtrack_name, intervals=scope, iterator=(N,M))` (or CartesianGrid iterator) now works when the vtrack has any of these five object-level funcs. Closes the remaining R-parity gap from v0.1.79 for reducing 2D vtracks + 2D iterators.

### Limitations
- `global.percentile` not yet supported through the scanner path (deferred - requires a precomputed population).
- 2D vtracks with shifts still raise (deferred).

## v0.1.79 (2026-05-20)

### Features
- **Reducing 2D vtracks now supported with 2D iterators (R parity).** `gextract(vtrack_name, intervals=scope, iterator=(N,M))` (or `iterator="<2D track>"` or `iterator=CartesianGridSpec(...)`) now computes one aggregated value per emitted cell when the vtrack has a 2D reducer func (`area`, `avg`, `min`, `max`, `weighted.sum`). Previously raised `NotImplementedError` or fell through to the wrong path. Closes 8 xfail-strict R-parity gaps from v0.1.75-v0.1.77.

### Limitations
- Vtracks with 2D shifts (`sshift1`/`sshift2`) still raise - deferred.
- Multi-vtrack compound expressions still raise - deferred.
- `exists`, `size`, `first`, `last`, `sample`, `global.percentile` funcs still raise - C++ scanner does not support object-level funcs; deferred.

## v0.1.78 (2026-05-20)

### Features
- **Intervals iterator via scanner (opt-in).** `PYMISHA_USE_SCANNER_FOR_INTERVALS=1` routes the implicit intervals-iterator case (`gextract(track, intervals=scope_df)`) through the new C++ scanner path. The scanner returns one row per scope interval (per-rect aggregation via the var's func, e.g. `avg`). This complements the existing default path which returns one row per (scope_rect, track_object) intersection (per-object enumeration). The two paths solve different problems; both remain available.
- Bench (testdb, 10 scope rects on chr1 x chr1): scanner path ~4.9 ms (10 rows, aggregated); existing per-object enumeration path ~7.3 ms (7 rows, per-object). Different output shapes; the numbers are informational only.

### Internal
- New `IntervalsPolicy` dataclass + parser support in `pymisha/_iterator_policy.py`.
- `pm_extract_2d_scanner` accepts `kind="intervals"`.
- The bypass paths `pm_extract_2d` and `pm_extract_2d_objects` are intentionally retained as the default - they provide per-object enumeration semantics that the scanner aggregation cannot directly replace.
- Multitask regression guard added: `test_intervals_via_scanner_multitask_equivalence` confirms scanner result is identical under single-process and multi-process CONFIG.

### Limitations
- Same as v0.1.77.
- `PYMISHA_USE_SCANNER_FOR_INTERVALS=1` only affects the default no-iterator case for bare physical tracks. vtracks and explicit-DataFrame iterators continue to use the legacy path.
- Scanner is single-process; multitasking deferred.

## v0.1.77 (2026-05-20)

### Features
- **CartesianGrid 2D iterator (`giterator_cartesian_grid(..., stream=True)` + `gextract(iterator=spec)`).** New C++ streaming iterator generates the cartesian product of 1D windows around two sets of interval centers. Optional `band_idx` filtering keeps only (center-i, center-j) pairs within a specified index-delta range. Bench (testdb rects_track, 5 centers, 3 windows/axis, 225 cells, chr1 x chr1): ~5.5 ms pymisha vs ~8.0 ms R misha 5.7.2 (1.45x faster).

### Internal
- `giterator_cartesian_grid` accepts a new `stream=True` kwarg that returns a `CartesianGridSpec` instead of a materialized DataFrame. The materialize path (`stream=False`) remains the default for backward compatibility. Stream path produces one row per cell (avg aggregation); materialize path produces one row per (cell, track-object) intersection.
- `bench_2d_iterators.py` extended with Workload D (CartesianGrid) including an R misha comparison.

### Limitations
- Same as v0.1.76 (vtrack + iterator policies, multitasking, memory materialization deferred).
- Reducing-vtrack regression-guard xfail remains.

## v0.1.76 (2026-05-20)

### Features
- **TrackRects 2D iterator (`gextract(..., iterator="<2D track name>")`).** New C++ streaming iterator yields the rects from a 2D rects/points track that intersect the user's 2D scope. Bench (testdb rects_track, chr1 x chr1 scope, 76 rows): 6.3 ms pymisha vs 8.0 ms R misha 5.7.2 (1.27x faster).

### Fixes
- **`gextract` with `iterator=<1D track>` or `iterator=<unknown track>`** now raises a clear `ValueError`. Previously the argument was silently ignored.

### Internal
- Iterator constructor contract: 2D iterators no longer prime in the constructor; callers must call `begin()` explicitly. Affects FixedRect (v0.1.75-shipped) and the new TrackRects. Externally visible only via the test bindings.

### Limitations
- Same as v0.1.75 (vtrack + iterator policies, multitasking, memory materialization deferred).
- Reducing 2D vtracks (e.g. `func="avg"`) with `iterator=<2D track>` still routes through the legacy one-row-per-scope path - gap documented with xfail-strict regression test.

## v0.1.75 (2026-05-20)

### Features
- **FixedRect 2D iterator (`gextract(..., iterator=(N1, N2))`).** New C++ streaming iterator subdivides each 2D scope rect into an `N1 x N2` grid, yielding cells in row-major order with diagonal-band shrinking. Replaces the Python materialize-then-iterate path for this case. Bench (testdb, 100-bin scope): ~5.5 ms in pymisha vs ~106 ms in R misha (~19x speedup). First step of Group K parity.

### Limitations
- `iterator=(N, M)` with virtual tracks raises `NotImplementedError` until vtrack support lands in a later release. Use a bare physical 2D track name.
- The FixedRect path runs single-process. Multitask integration is deferred to a later release.
- Large grids (N1 × N2 cells across many chrom pairs) materialize the full rect set in memory before aggregation; for genome-wide Hi-C resolution grids this can exceed available RAM. Streaming aggregation is planned for a later release.

## v0.1.74 (2026-05-19)

### Fixes
- **R factor decode in legacy `.interv` files.** A factor `chrom` column was decoded as bare 1-based integer codes (`"1"`..`"22"`) instead of the level labels (`"chr1"`..`"chrY"`), producing misleading `Chromosome "20" does not exist` errors on databases with fewer chromosomes than factor levels. Factors now decode to `pandas.Categorical`, including ordered factors and NA codes.
- **R `NA_LOGICAL` and `NA_integer_` preserved.** Atomic logical NAs no longer silently become `True`; integer NAs no longer surface as the `-INT_MAX` sentinel. Vectors with NAs decode to `pandas.arrays.BooleanArray` / `IntegerArray`; NA-free vectors keep the existing `bool` / `int32` ndarray return type.

### Internal
- Remove unused `_write_factor_column`; the writer dispatch flattens Categorical to character on disk (unchanged).

## v0.1.73 (2026-05-18)

### Fixes
- **`gdb_install_intervals` resolves assembly_name from the FTP listing when the NCBI Datasets `/dataset_report` is empty or suppressed.** Previously, accessions whose Datasets API record was suppressed (e.g. `GCF_000001635.26` GRCm38.p6, replaced by GRCm39 in the active index) left `assembly_name` empty and the GFF was silently skipped even though the FTP directory still hosted `<acc>_<asm>_genomic.gff.gz`. Now falls back to parsing the parent FTP listing for the assembly subdirectory. Ports R commit `d6cd6047`.

### Internal
- Regression tests added for: cache invalidation on `gtrack_rm` + `gtrack_create_*` and `gintervals_rm` + `gintervals_save` on indexed DBs (R commit `4c3803b0`); `min_coverage` gate position in chrom-alias resolution (R commit `537bfe29`). Both pass on current pymisha; tests guard against drift.

## v0.1.72 (2026-05-18)

### Features
- **`gintervals_to_mat` / `gintervals_from_mat`.** Pivot an intervals + values DataFrame into a matrix-shaped DataFrame indexed by intervals (3-level MultiIndex of `chrom/start/end`, or 4-level when `id_col` is given). Inverse via `gintervals_from_mat`. Pandas-native `iloc` slicing and `pd.concat` preserve the intervals correspondence. Ports R PR #120.

### Fixes
- **mypy CI:** narrow the type of `df` after `_apply_intervals_join` in `_apply_extract_output` so the post-processing block type-checks cleanly. No behavior change.

## v0.1.71 (2026-05-18)

### Features
- **`gextract` gains `intervals_join` argument.** Replaces the `misha.ext::gextract.left_join` workflow with a single built-in call. Modes: `"id"` (default; appends `intervalID`), `"intervals"` (drops `intervalID`, attaches every column of the input intervals DataFrame to each output row, suffixing conflicting names with `"1"`), `"none"` (drops `intervalID`, attaches nothing). Supported attach dtypes: numeric, bool, string, category. `file=` and `intervals_set_out=` are rejected with `intervals_join="intervals"`. Ports R PR #124.

## v0.1.70 (2026-05-15)

### Performance
- **2D object-level vtrack functions are ~5x faster.** The `exists`, `size`, `first`, `last`, and `sample` reductions over a 2D RECTS/POINTS track route through the new `pm_extract_2d_objects` C++ binding instead of the per-interval Python loop. Synthetic 10M-rect track, 100k queries, warm cache: 6.8 s -> 1.26 s.
- Vectorized chromid name lookups in `_gextract_2d_single` (matters at multi-million-row outputs).

## v0.1.69 (2026-05-15)

### Performance
- **`gquantiles` / `gsummary` / `gscreen` / `gpartition` 20-30x faster on 100k-1M interval inputs.** Multitasking no longer fires for sparse-track workloads where fork + IPC overhead exceeds the serial scan. Phylo447 / 100k 500-bp intervals: gquantiles 190 -> 7 ms, gsummary 188 -> 7 ms, gscreen 188 -> 8 ms, gpartition 186 -> 6 ms.
- **`gextract` 20-90x faster on small-to-medium workloads.** Phylo447 / 100k 500-bp intervals: 491 -> 7 ms (single track), 634 -> 7 ms (3 tracks).
- **Child-process wakeup polling reduced from 100 ms to 1 ms** in all multitask call sites.
- **`gintervals_neighbors`** does a single pass over the sorted target set per chrom run instead of rescanning it on every query-chrom transition.
- **`gseq_extract`** drops a redundant per-interval `std::string` copy.
- **`gpartition`** reuses the chrom `PyUnicode` across same-chrom output runs (matches the v0.1.64 `intervals_to_py` pattern).

### Configuration
- **`pm.CONFIG["min_scope4process"]`** (default `1_000_000_000` bp): minimum scope per worker required to enable multitasking. Set to `0` to fork unconditionally.
- **`pm.CONFIG["min_intervs4process"]`** (default `250_000`): minimum total intervals required before any fork.

## v0.1.68 (2026-05-15)

### Performance
- **`gextract` on raw 2D RECTS/POINTS tracks is ~2x faster.** Replaces the per-interval Python loop in `_gextract_2d_single` with a native C++ object-enumeration path (`pm_extract_2d`). Synthetic 3M-rect track, 100k queries, 2.4M output rows: 6.18 s -> 2.61 s (2.37x). Vtrack-aggregated paths (`avg` / `area` / `weighted.sum` / `min` / `max`) and object-level vtracks are unchanged in this release.

## v0.1.67 (2026-05-15)

### Fixes
- **mypy CI:** annotate the `pm.CONFIG["track_create_parallel_writers"]` cast in `_apply_create_parallel_writers_from_config` (CONFIG values are typed as `object`). No behavior change.

## v0.1.66 (2026-05-15)

### Configuration
- **`pm.CONFIG["track_create_parallel_writers"]`** controls how many threads `gtrack_create_sparse` uses for empty per-chrom signature file dispatch on non-indexed DBs. Default 4. Set to 1 to force sequential, higher to push parallelism. `multitasking=False` also forces 1 worker. Was hardcoded to 4 in v0.1.65.

## v0.1.65 (2026-05-15)

### Performance
- **`gintervals_intersect` / `union` / `diff` / `canonic` 1.4-1.9x faster on million-row inputs.** Hg38, 1M-row 1D ops: intersect 522 -> 331 ms, union 391 -> 212 ms, diff 447 -> 260 ms, canonic 241 -> 145 ms.
- **`gtrack_create_sparse` ~1.5x faster on million-row inputs.** Hg38 (non-indexed, 455 chroms): 1.88 s -> 1.24 s. Indexed DBs (Phylo447, 194 chroms): 599 -> 511 ms.
- **`gtrack_create_dense` -215 ms per call** on databases with thousands of tracks.
- **`gextract` with `iter=intvar` ~1.4x faster on dense tracks.** Hg38, 5 Mb / 200-bp bins: 30.3 -> 21.5 ms.

## v0.1.64 (2026-05-15)

### Fixes
- **mypy CI:** expression helpers (`_parse_expr_vars`, `_validate_expr_security`, and friends) now accept `collections.abc.Set[str]` for `track_names` / `vtrack_names`, so the cached `frozenset` returned by the v0.1.63 `pm_track_names` cache type-checks at every call site. No behavior change.

## v0.1.63 (2026-05-15)

### Performance
- **`gintervals_ls()` is now O(1) on warm databases (was O(N_files)).** PMDb caches interval-set names alongside tracks during the existing `gdb_init` scan; the old Python `Path.rglob("*.interv*")` walked every per-chrom file under `tracks/`. On hg38 (15k tracks, 20 interval sets): ~38 s -> ~0.04 ms. `gintervals_save` / `gintervals_rm` incrementally register / unregister names without paying a full `pm_dbreload` rescan.
- **`gextract` / `gsummary` / `gscreen` / `gdist` / `glookup` with `iterator=intervals` is 100-1000x faster on million-row inputs.** Vectorized the per-chromosome two-pointer sweep that intersects a scope DataFrame with an iterator DataFrame; the old O(K*J) Python loop ate ~95% of total time on real workloads (hg38 phyloP447, 10k 500-bp intervals: 9.4 s -> 77 ms).
- **`gintervals_save` is ~50x faster on million-row data frames.** Replaced `pyreadr.write_rds()` with a native R-serialize XDR writer (`pymisha._r_serialize.write_dataframe`); the writer streams numeric/string columns to disk in bulk `numpy.tobytes()` ops instead of round-tripping through the librdata C++ binding row by row (1M rows, 24 unique chroms: 16.8 s -> 0.34 s).
- **`gintervals_load` is ~2x faster on million-row interval sets.** The R-serialize reader's STRSXP path now inlines CHARSXP parsing instead of recursing through `_read_item` for every string (1M rows: 1.0 s -> 0.55 s).
- **`gextract` / `gsummary` / `gscreen` tight loops are 5-8x faster.** Cached `_check_computed_tracks` (per-(exprs, vtracks) and per-track results) plus a Python-side cache for `pm_track_names()` shave ~12 ms of fixed Python overhead off every call. Caches are cleared on `gdb_init` / `gdb_reload` / `gdb_unload`; a `_pymisha.pm_dbreload` monkey-patch keeps the cache in sync even for callers that hit the C extension directly. Hg38, 1000 intervals, tight `iter=intervals` loop: ~77 ms -> ~10 ms per call.

### Internal
- `_intervals_to_cpp` now constructs the pymisha-internal list-of-arrays format directly, skipping a `DataFrame.copy()` + per-column `iloc` round-trip.
- New C++ API: `pm_interv_names`, `pm_interv_register`, `pm_interv_unregister`. New PMDb members: `m_interv_cache`, `interv_names()`, `register_interv()`, `unregister_interv()`.

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
