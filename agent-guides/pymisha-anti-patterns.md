# pymisha - anti-patterns

The silent footguns - code runs, output looks plausible, downstream conclusions are subtly wrong. Inline `Avoid:` callouts in [`pymisha-core.md`](pymisha-core.md) and [`pymisha-advanced.md`](pymisha-advanced.md) reference entries here.

## Contents

- [Configuration](#configuration)
- [Virtual track lifecycle](#virtual-track-lifecycle)
- [Intervals](#intervals)
- [Extraction & expressions](#extraction--expressions)
- [Quantiles / thresholds](#quantiles--thresholds)
- [Aggregator choice](#aggregator-choice)

## Configuration

**A1. Mid-script `gsetroot` swap.** Switching genomes while in-memory intervals reference the previous genome silently corrupts coordinates - every subsequent `gintervals_load`, vtrack lookup, or sequence extract resolves against the new reference. If you must swap, wrap with `try / finally` so a partial run restores cleanly:

```python
prev = pm.gdb_info()["path"]
try:
    pm.gsetroot("/path/to/other_db")
    # ... analysis ...
finally:
    pm.gsetroot(prev)
```

See pymisha-advanced.md §6.

**A2. `CONFIG["multitasking"] = True` inside `multiprocessing` / `joblib.Parallel` / `concurrent.futures` workers.** pymisha's multitasking forks worker processes to parallelize across chromosomes. When the pymisha call is *itself* inside a forked context (`multiprocessing.Pool` body, `joblib.Parallel(prefer="processes")` body, Dask worker), the nested forks deadlock or oversubscribe cores. Fix: set `pm.CONFIG["multitasking"] = False` inside the inner body and let the outer layer own the parallelism. Symptom: hung workers that never make progress and don't error.

**A3. High `max_data_size` combined with `multitasking = True` → OOM.** `max_data_size` caps the result size of *each* `gextract`; with multitasking, several extractions hold that much simultaneously. On a heavy genome-wide pull (`max_data_size = 10_000_000_000` × multiple worker copies) the resident set exceeds available RAM. Three fixes, pick by what's tightest: lower `max_data_size` to what one extract actually needs, set `multitasking = False` for the heavy call, or partition the query by chromosome and process the partitions serially.

## Virtual track lifecycle

**A4. Defensive `gvtrack_rm` before every `gvtrack_create`.** Boilerplate of the form `if name in pm.gvtrack_ls(): pm.gvtrack_rm(name)` (or `try: pm.gvtrack_rm(name); except: pass`) before every `gvtrack_create` is dead weight - `gvtrack_create` silently overwrites on name clash with another vtrack. The error only fires when the name clashes with a *regular track* or *intervals set*, which the defensive `gvtrack_rm` doesn't fix anyway.

## Intervals

**A5. Iterating `gintervals_normalize` + `gintervals_canonic` in a `while` loop until "stable".** Old code does `while True: x = pm.gintervals_canonic(pm.gintervals_normalize(x, W)); if ...: break` to "settle" overlapping peaks. A single pass suffices by construction: `gintervals_normalize` writes width-`W` intervals centered on the input midpoints, and `gintervals_canonic` then merges overlaps in one sort+merge. The second iteration changes nothing because the input is already fixed-width and sorted.

**A6. `(start + end) / 2` for the midpoint.** Float division yields a non-integer center; downstream operations round inconsistently (sometimes via `astype(int)`, sometimes via `np.floor`, sometimes via `round`). Always integer-divide: `(start + end) // 2`.

**A7. Reflexive `gintervals_canonic` after every `pd.concat`.** Canonicalizing destroys per-row identity - paired matched-control rows, per-sample replicate identity, parallel rows for downstream joins all collapse into the merged set. Run `gintervals_canonic` *only* when you actually want overlaps merged (typical: union-of-peak-sets across conditions). When the rows mean "distinct things at the same location", skip canonic.

**A8. Base `pd.DataFrame({"chrom": ..., "start": ..., "end": ...})` instead of `pm.gintervals(...)`.** `pd.DataFrame` skips coordinate validation against the active genome - typos in chrom names, ends past chromosome length, or negative starts pass through silently. Downstream queries either silently drop the bad rows or raise opaque C++-level errors. Always pass through `pm.gintervals(chrom, start, end)` which validates against the active genome and normalizes integer types.

## Extraction & expressions

**A9. Looping `gextract` over track names.** Calling `gextract` once per track and joining on `chrom/start/end` is N× slower than one `pm.gextract(["track1", "track2", ...], ...)`. The C++ engine processes all expressions jointly in one genome pass; the merge step is unnecessary work and a source of join bugs (NaN-vs-missing-row).

**A10. Post-extract `prof.fillna(0)` instead of inside the expression.** Sparse tracks (and many dense tracks) return NaN at unmeasured bins. Filling at extract time via `pm.gextract("ifelse(is.na(track), 0, track)", ...)` keeps the NaN handling inside the C++ engine; the post-hoc pandas fill works but is slower and easier to forget.

**A11. `iterator = peaks` when you meant per-bin resolution.** With `iterator=<intervals>`, `gextract` collapses to one row per interval (with vtrack aggregation). For a per-bin profile inside the same intervals, use `iterator=20` (or another bin size). For interval-relative bin indices use `pm.giterator_intervals(... interval_relative=True)`.

**A12. `iterator = 20` and assuming bin offsets are interval-relative.** Integer iterators are *chromosome-relative* - two peaks at different chromosome offsets get *misaligned* bin grids. If you need bin indices anchored to each peak's start, use `pm.giterator_intervals(... interval_relative=True)` (with the caveat that dense source tracks still align to their stored bin grid).

**A13. Skipping `.sort_values("intervalID")` (or equivalent) on gextract output.** With `multitasking = True`, `gextract` returns rows in chunked-chromosome order, not input order. If the downstream code indexes by row position into the original peaks frame, the result is silently shuffled. Always `.sort_values("intervalID").drop(columns="intervalID")` or equivalent. If you want the input intervals' columns on every output row (the most common reason to need `intervalID`), use `pm.gextract(..., intervals_join="intervals")` instead - the C++ side attaches them inline, keyed by `intervalID`, with no Python-side merge.

**A14. Scalar-collapse functions inside an expression.** `mean(track)`, `min(track)`, `sum(track)` and similar inside an expression string collapse the per-bin vector to one scalar that the engine then recycles across every bin. Expression functions must be vectorized; use `pmin`, `pmax`, `ifelse`, arithmetic, `log` / `exp`, `abs` - or compute the scalar separately and substitute its value into the expression string via f-string.

**A15. Hand-rolled `(a + b) / 2` for per-element averaging across two tracks.** `(a + b) / 2` propagates NaN whenever *either* input is NaN, so any position where one track has data and the other doesn't comes out as NaN instead of the value you have. There's no `pmean` in the expression language; hand-roll: `"ifelse(is.na(a), b, ifelse(is.na(b), a, (a + b) / 2))"`. Same trap as R's `tgutil::pmean(na.rm=TRUE)`, no built-in shortcut on the Python side.

## Quantiles / thresholds

**A16. Hardcoded threshold values.** `pm.gscreen("track > 1870", ...)` makes the analysis uninterpretable across datasets and silently breaks on retraining or track replacement. Derive the threshold from `pm.gquantiles(track, percentiles=p, ...)` so the rule is "top p-percentile of the empirical distribution" - re-evaluable on any new track.

**A17. `pm.gquantiles(x, probs=0.9, ...)` instead of `percentiles=`.** `probs` (NumPy / pandas convention) and `q` are not the right kwargs; pymisha uses `percentiles=`. Unknown kwargs get swallowed by `**kwargs` and `gquantiles` returns the default `percentiles=0.5` (median) regardless of what you passed. The threshold is wrong; the downstream screen call returns the wrong intervals; nothing errors.

**A18. `gquantiles` on a track with many NaNs assumed to include zeros.** `gquantiles` silently drops NaN positions when computing percentiles. For "top 1% of covered positions" that's right; for "top 1% of all positions including zero-coverage", wrap with `"ifelse(is.na(track), 0, track)"` first.

## Aggregator choice

**A19. `func = "avg"` for count-like tracks.** ATAC reads, ChIP coverage, contact counts have meaningful magnitudes - `avg` divides by bin count and discards the count. For enrichment analyses you almost always want `"sum"` (1D) or `"weighted.sum"` (2D / sparse).

**A20. Aggregating a stored `pwm` (LSE) track with `func = "sum"`.** LSE does not compose under summation. Re-aggregating per-bin LSE values needs `func = "lse"` so the result is the LSE over the region's windows. `sum` over LSE values produces a number that has no probabilistic interpretation.

**A21. `bidirect = True` when motif orientation matters.** Bidirectional PWM scoring returns the better-strand score per position, discarding the strand. For oriented-motif analyses (CTCF orientation, stranded peak calling), pair two vtracks with `strand = +1` and `strand = -1`, `bidirect = False`, so the orientation survives.
