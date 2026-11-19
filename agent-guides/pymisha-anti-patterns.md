# pymisha - anti-patterns

The silent footguns - code runs, output looks plausible, downstream conclusions are subtly wrong. Inline `Avoid:` callouts in [`pymisha-core.md`](pymisha-core.md) and [`pymisha-advanced.md`](pymisha-advanced.md) reference entries here.

Entry ids are shared with R misha's `misha-anti-patterns.md`: the same id is the same pitfall in both. An id missing here (A1) applies only to R; A9c, A14f and A16 apply only to pymisha.

## Contents

- [Configuration](#configuration)
- [Virtual track lifecycle](#virtual-track-lifecycle)
- [Intervals](#intervals)
- [Extraction & expressions](#extraction--expressions)
- [Quantiles / thresholds](#quantiles--thresholds)
- [Aggregator choice](#aggregator-choice)

## Configuration

**A2. Mid-script `gsetroot` swap.** Switching genomes while in-memory intervals reference the previous genome silently corrupts coordinates - every subsequent `gintervals_load`, vtrack lookup, or sequence extract resolves against the new reference. If you must swap, wrap with `try / finally` so a partial run restores cleanly:

```python
prev = pm.gdb_info()["path"]
try:
    pm.gsetroot("/path/to/other_db")
    # ... analysis ...
finally:
    pm.gsetroot(prev)
```

See pymisha-advanced.md §6.

**A3. `CONFIG["multitasking"] = True` inside `multiprocessing` / `joblib.Parallel` / `concurrent.futures` workers.** pymisha's multitasking forks worker processes to parallelize across chromosomes. When the pymisha call is *itself* inside a forked context (`multiprocessing.Pool` body, `joblib.Parallel(prefer="processes")` body, Dask worker), the nested forks deadlock or oversubscribe cores. Fix: set `pm.CONFIG["multitasking"] = False` inside the inner body and let the outer layer own the parallelism. Symptom: hung workers that never make progress and don't error.

**A4. High `max_data_size` combined with `multitasking = True` → OOM.** `max_data_size` caps the result size of *each* `gextract`; with multitasking, several extractions hold that much simultaneously. On a heavy genome-wide pull (`max_data_size = 10_000_000_000` × multiple worker copies) the resident set exceeds available RAM. Three fixes, pick by what's tightest: lower `max_data_size` to what one extract actually needs, set `multitasking = False` for the heavy call, or partition the query by chromosome and process the partitions serially.

## Virtual track lifecycle

**A5. Defensive `gvtrack_rm` before every `gvtrack_create`.** Boilerplate of the form `if name in pm.gvtrack_ls(): pm.gvtrack_rm(name)` (or `try: pm.gvtrack_rm(name); except: pass`) before every `gvtrack_create` is dead weight - `gvtrack_create` silently overwrites on name clash with another vtrack. The error only fires when the name clashes with a *regular track* or *intervals set*, which the defensive `gvtrack_rm` doesn't fix anyway.

## Intervals

**A6. Iterating `gintervals_normalize` + `gintervals_canonic` in a `while` loop until "stable".** Old code does `while True: x = pm.gintervals_canonic(pm.gintervals_normalize(x, W)); if ...: break` to "settle" overlapping peaks. A single pass suffices by construction: `gintervals_normalize` writes width-`W` intervals centered on the input midpoints, and `gintervals_canonic` then merges overlaps in one sort+merge. The second iteration changes nothing because the input is already fixed-width and sorted.

**A7. `(start + end) / 2` for the midpoint.** Float division yields a non-integer center; downstream operations round inconsistently (sometimes via `astype(int)`, sometimes via `np.floor`, sometimes via `round`). Always integer-divide: `(start + end) // 2`.

**A8. Reflexive `gintervals_canonic` after every `pd.concat`.** Canonicalizing destroys per-row identity - paired matched-control rows, per-sample replicate identity, parallel rows for downstream joins all collapse into the merged set. Run `gintervals_canonic` *only* when you actually want overlaps merged (typical: union-of-peak-sets across conditions). When the rows mean "distinct things at the same location", skip canonic.

**A9. Base `pd.DataFrame({"chrom": ..., "start": ..., "end": ...})` instead of `pm.gintervals(...)`.** `pm.gintervals(chrom, start, end)` checks the intervals against the active genome when you build them and normalizes integer types; a plain DataFrame is checked only when a query reads it, which then stops with an error (unknown chromosome, negative start, end past the chromosome) far from the code that built it. Exception: when the row order matters (rows aligned with another table, or file row numbers used as ids), derive the new frame from one that is already validated, e.g. `peaks.assign(start=..., end=...)` on the output of `pm.gintervals_import_bed(..., keep_order=True)`, because `pm.gintervals()` sorts (A14b).

**A9b. Joining pymisha output with an external table by the chrom string.** pymisha resolves chromosome aliases on input (`chr1`, `1`, `NC_000001.11`, ...), but a `pd.merge` / `.isin()` between pymisha output and a table read from another source compares strings: `"1"` never equals `"chr1"`, and the join comes back silently empty. Bring the external table to the database's names first (`pm.gintervals_force_range(ext)` returns them and keeps row order and extra columns), or do the join itself with `pm.gintervals_neighbors` (overlaps: `mindist=0, maxdist=0`).

**A9c. Chromosome names written by pymisha 0.14 or earlier.** In a database whose `chrom_sizes.txt` names lack `chr` (per-chromosome databases such as the lab's hg38 and mm10, and the example database), pymisha up to 0.14 named the chromosomes `"1"`, `"2"`, ...; from 0.15.0 they are `"chr1"`, `"chr2"`, ..., as in R misha. Tables, caches and files saved by the older versions carry the old names, so matching them against current pymisha output or a `chr`-prefixed file gives the A9b empty join. Recompute them, or bring them to the database's names with `pm.gintervals_force_range`.

## Extraction & expressions

**A10. Looping `gextract` over track names.** Calling `gextract` once per track and joining on `chrom/start/end` is N× slower than one `pm.gextract(["track1", "track2", ...], ...)`. The C++ engine processes all expressions jointly in one genome pass; the merge step is unnecessary work and a source of join bugs (NaN-vs-missing-row).

**A11. Post-extract `prof.fillna(0)` instead of inside the expression.** Sparse tracks (and many dense tracks) return NaN at unmeasured bins. Filling at extract time via `pm.gextract("np.where(np.isnan(track), 0, track)", ...)` keeps the NaN handling inside the expression; the post-hoc pandas fill works but is slower and easier to forget.

**A12. `iterator = peaks` when you meant per-bin resolution.** With `iterator=<intervals>`, `gextract` collapses to one row per interval (with vtrack aggregation). For a per-bin profile inside the same intervals, use `iterator=20` (or another bin size). For interval-relative bin indices use `pm.giterator_intervals(... interval_relative=True)`.

**A13. `iterator = 20` and assuming bin offsets are interval-relative.** Integer iterators are *chromosome-relative* - two peaks at different chromosome offsets get *misaligned* bin grids. If you need bin indices anchored to each peak's start, use `pm.giterator_intervals(... interval_relative=True)` (with the caveat that dense source tracks still align to their stored bin grid).

**A14. Skipping `.sort_values("intervalID")` (or equivalent) on gextract output.** With `multitasking = True`, `gextract` returns rows in chunked-chromosome order, not input order. If the downstream code indexes by row position into the original peaks frame, the result is silently shuffled. Always `.sort_values("intervalID").drop(columns="intervalID")` or equivalent. If you want the input intervals' columns on every output row (the most common reason to need `intervalID`), use `pm.gextract(..., intervals_join="intervals")` instead - the C++ side attaches them inline, keyed by `intervalID`, with no Python-side merge.

**A14b. Sorting by `intervalID` back into a frame you built with `pm.gintervals()`.** `pm.gintervals()` sorts its input, so `intervalID` indexes the *sorted* frame, not the one you handed it. A9 and A14 therefore interact: build an intervals frame in a meaningful non-genomic order (sampled rows, paired controls, an order you intend to index positionally), pass it through `pm.gintervals()`, then assign `res.sort_values("intervalID")["value"].to_numpy()` back onto your original frame, and you get a silent row shuffle. A plain DataFrame round-trips (its `intervalID`s come back permuted and sorting by them undoes it); the `pm.gintervals()` one does not. Either sort your own frame first so the two orders agree, or carry your own id column and join on it - never index positionally by `intervalID` into a frame you did not sort yourself.

**A14c. Rebuilding a frame with `pm.gintervals()` before `gseq_extract`.** `pm.gseq_extract` on a plain DataFrame returns one sequence per row, in the input order, and reverse-complements rows with `strand == -1`. `pm.gintervals(chroms, starts, ends)` sorts the rows and, without `strand=`, drops the strand, so sequences extracted from its result no longer line up with the original rows and all come from the + strand. Pass the DataFrame itself (with its `strand` column) to `gseq_extract`.

**A14d. One-hot encoding `gseq_extract` output as is.** Sequences keep the genome's soft mask: in a database built from a soft-masked FASTA (UCSC hg38, for one) repeat bases come back lowercase. An encoder that only knows `ACGT` turns them into zeros or `N` without a warning. Call `.upper()` on each sequence first, unless you want the mask.

**A14e. `(a + b) / 2` for per-element averaging across two tracks.** `(a + b) / 2` propagates NaN whenever *either* input is NaN, so any position where one track has data and the other doesn't comes out as NaN instead of the value you have. Spell out the NaN cases: `"np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))"`.

**A14f. Functions that reduce the whole vector inside an expression.** An expression is evaluated on a buffer of bins at a time, with each track name bound to a NumPy array. `min(track)`, `max(track)`, `np.mean(track)`, `np.nanmin(track)` and the like reduce that buffer to one number, which NumPy then broadcasts back over the buffer's bins - so every bin gets the same value, and it is not even a genome-wide statistic, just one per buffer. (Bare function names other than `abs`, `min`, `max`, `round`, `int`, `float` and `bool` - `mean`, `sum`, `log`, `pmin` - are refused as not allowed.) Use element-wise functions: `np.minimum`, `np.maximum`, `np.fmin` / `np.fmax` (ignore NaN), `np.where`, arithmetic, `np.log2`, `abs` - or compute the scalar separately and put its value into the expression string with an f-string.

## Quantiles / thresholds

**A15. Hardcoded threshold values.** `pm.gscreen("track > 1870", ...)` makes the analysis uninterpretable across datasets and silently breaks on retraining or track replacement. Derive the threshold from `pm.gquantiles(track, percentiles=p, ...)` so the rule is "top p-percentile of the empirical distribution" - re-evaluable on any new track.

**A16. `pm.gquantiles(x, probs=0.9, ...)` instead of `percentiles=`.** `probs` (NumPy / pandas convention) and `q` are not the right kwargs; pymisha uses `percentiles=`. Unknown kwargs get swallowed by `**kwargs` and `gquantiles` returns the default `percentiles=0.5` (median) regardless of what you passed. The threshold is wrong; the downstream screen call returns the wrong intervals; nothing errors.

**A17. `gquantiles` on a track with many NaNs assumed to include zeros.** `gquantiles` silently drops NaN positions when computing percentiles. For "top 1% of covered positions" that's right; for "top 1% of all positions including zero-coverage", wrap with `"np.where(np.isnan(track), 0, track)"` first.

## Aggregator choice

**A18. `func = "avg"` for count-like tracks.** ATAC reads, ChIP coverage, contact counts have meaningful magnitudes - `avg` divides by bin count and discards the count. For enrichment analyses you almost always want `"sum"` (1D) or `"weighted.sum"` (2D / sparse).

**A19. Aggregating a stored `pwm` (LSE) track with `func = "sum"`.** LSE does not compose under summation. Re-aggregating per-bin LSE values needs `func = "lse"` so the result is the LSE over the region's windows. `sum` over LSE values produces a number that has no probabilistic interpretation.

**A20. `bidirect = True` when motif orientation matters.** Bidirectional PWM scoring combines the two strands per position with a log-sum-exp (not a max), and discards which strand won. For oriented-motif analyses (CTCF orientation, stranded peak calling), pair two vtracks with `strand = +1` and `strand = -1`, `bidirect = False`, so the orientation survives.

**A21. Verifying a `pwm.max` hit by re-scoring a span that is not the vtrack's window set.** Since pymisha 0.14.0 `pm.gseq_pwm(mode="max")` and `pwm.max` take the same strand union - LSE per position, then max over positions - so they do agree, but only window-for-window. The vtrack's `extend=True` (the default) pads the interval's **END** by `len(pssm) - 1`, so `pm.gseq_extract(iv)` on its own is short by that much and silently drops the last `len(pssm) - 1` anchors. Extract `pm.gintervals(chrom, start, end + len(pssm) - 1)` and score that with `extend=False` (`gseq_pwm`'s default), or re-score with the same vtrack under `extend=False`. Same rule for `gseq_pwm(mode="count")` against `pwm.count`. Before 0.14.0 `gseq_pwm` took the max of the two strands where `pwm.max` log-sum-exps them, and `mode="count"` counted a window matching on both strands twice, on the pure-Python scoring paths (`neutral_chars`, `gap_chars`, `start_pos` / `end_pos`, `extend`, `return_strand`).
