# pymisha - core operations

Compact reference for the everyday pymisha workflow. Concepts first, then chooser tables, then one recipe per common task. Each recipe ends with short `Avoid:` callouts; the full anti-pattern catalogue is in [`pymisha-anti-patterns.md`](pymisha-anti-patterns.md). Advanced topics (2D / Hi-C, PWM, import/export, new genomes) live in [`pymisha-advanced.md`](pymisha-advanced.md).

pymisha is a Python port of the R `misha` package. The on-disk track DB format is shared - the same trackdb is read interchangeably from both. Function names map dot-to-underscore: `gtrack.create` (R) becomes `pm.gtrack_create` (Python), `gvtrack.iterator` becomes `pm.gvtrack_iterator`, etc. Return types are pandas DataFrames (1D / 2D intervals) and NumPy arrays (histograms, quantiles).

## Contents

- [1. Concepts](#1-concepts)
- [2. Session bootstrap](#2-session-bootstrap)
- [3. Function chooser tables](#3-function-chooser-tables)
- [4. Recipes](#4-recipes)
  - [4.1 Build and manipulate intervals](#41-build-and-manipulate-intervals)
  - [4.2 Annotate intervals with nearby features](#42-annotate-intervals-with-nearby-features)
  - [4.3 Distance flavors - picking the right primitive](#43-distance-flavors)
  - [4.4 Extract per-region values](#44-extract-per-region-values)
  - [4.5 Aggregate signal in a flanking window](#45-aggregate-signal-in-a-flanking-window)
  - [4.6 Call regions of interest (gscreen + thresholding)](#46-call-regions-of-interest)
  - [4.7 Joint distributions, summaries, correlations](#47-joint-distributions-summaries-correlations)
  - [4.8 Materialize derived tracks](#48-materialize-derived-tracks)

## 1. Concepts

**Mental model.** Think of every pymisha query as a *walk across the genome*: an iterator visits one position (or one interval, or one pair of positions) at a time, in genomic order, gathering values from any tracks or virtual tracks you ask for at each stop. You don't load tracks into Python and slice them; you describe *where to walk* (`intervals=`), *what step size* (`iterator=`), and *what to collect at each step* (the expression / vtracks). The engine streams the genome once, top to bottom, in C++. Almost everything else in this guide is a variation on that idea.

### 1.1 Track database

A *track DB* is a directory tree on disk under `<GROOT>`. Each subdirectory is a namespace; each leaf is a **track**. Track names use dots, mirroring the layout: `epi.k27me3.es` lives at `<GROOT>/epi/k27me3/es/`. Attach with `pm.gsetroot(<path>)` (or the alias `pm.gdb_init(<path>)`). Tracks are not loaded into Python; every query streams off disk in C++. The common project-portability convention is a relative symlink at the project root (`db -> /full/path/to/groot`) so `pm.gsetroot("db")` works from any check-out. Project-local intervals can live in a separate **dataset** attached with `pm.gdataset_load(<path>)`.

### 1.2 Tracks

Three storage shapes:

- **Dense** - value per bin (typical 20bp). `pm.gtrack_create`, `pm.gtrack_create_dense`. Continuous signal.
- **Sparse** - values at named intervals, NaN elsewhere. `pm.gtrack_create_sparse`. Peaks, per-CpG methylation.
- **2D** - value per rectangle in `(chrom1×coord1) × (chrom2×coord2)`. Hi-C / capture-C. Covered in [pymisha-advanced.md](pymisha-advanced.md).

`pm.gtrack_ls("pattern")` lists tracks; `pm.gtrack_exists(name)` is the idempotency guard. If a track you know exists on disk doesn't appear in `gtrack_ls` (for example, someone else added it after your session attached the DB, or you just dropped a file in by hand), run `pm.gdb_reload()` to rescan and refresh the cached track / intervals index.

**On-disk format.** Older trackdbs store one file per chromosome inside each track's directory. Newer trackdbs use an **indexed** format - `track.dat` + `track.idx` per track - which scales to thousands of contigs without dragging the filesystem. The pymisha API hides this entirely: `gextract`, `gscreen`, etc. work on both formats. **Do not reason from per-chromosome files on disk; always go through the API.** Convert an existing trackdb (or a single track) to the indexed form with `pm.gdb_convert_to_indexed(convert_tracks=True, convert_intervals=True)` (both flags default to `False`, so the bare call is a no-op), `pm.gtrack_convert_to_indexed(track)`, or `pm.gtrack_2d_convert_to_indexed(track, ...)` - important for fragmented assemblies (Zoonomia-style multi-contig genomes) where the per-chromosome layout is unworkable.

### 1.3 Intervals

A 1D intervals frame has columns `chrom, start, end` (+ optional `strand` and user columns); 2D has `chrom1, start1, end1, chrom2, start2, end2`. Build with `pm.gintervals(...)` / `pm.gintervals_2d(...)`. Genome-wide scopes: `pm.gintervals_all()` (1D), `pm.gintervals_2d_all()` (2D); whole-chrom 2D shorthand is `pm.gintervals_2d(chrom, 0, -1, chrom, 0, -1)`.

Canonical sets shipped per genome:
- **Annotation**: `intervs.global.tss`, `intervs.global.exon`, `intervs.global.tad_names`, `intervs.global.rmsk_<class>` (LINE, SINE, LTR, ...).
- **Sequence content** (extract via vtracks for GC / CpG features): `seq.GC_500_mean`, `seq.CG_500_mean`, `seq.GC500_bin20`, `seq.G_or_C`, `seq.CG`. The `_<W>_mean` form is a windowed average; `_<W>_bin20` is binned for fast threshold queries.

### 1.4 Virtual tracks (vtracks)

A vtrack is a parameterized view onto a track. One call composes aggregator and window:

```python
pm.gvtrack_create("k27_flank", src="epi.k27me3.es", func="sum",
                  sshift=-2000, eshift=2000)
```

Common aggregators (`func=`): `"sum"`, `"area"`, `"avg"`, `"max"`, `"min"`, `"lse"` (log-sum-exp), `"stdev"`, `"quantile"`, `"global.percentile.max"`, `"weighted.sum"`. Non-track sources: `"distance"` / `"distance.center"` / `"distance.edge"` (three distance-to-nearest-interval flavors - see [§4.3](#43-distance-flavors)), `"pwm"` / `"pwm.max"` / `"pwm.edit_distance"` (PSSM energies - see [pymisha-advanced.md](pymisha-advanced.md)), `"kmer.frac"` (sequence content).

For 2D: window args go through a paired call: `pm.gvtrack_iterator_2d(name, sshift1=..., eshift1=..., sshift2=..., eshift2=...)`. To bind a 1D vtrack to one axis of a 2D iteration: `pm.gvtrack_iterator(name, dim=1)` or `dim=2`.

Re-creating a vtrack name silently overwrites - no `gvtrack_rm` needed first.

### 1.5 Iterators

Every query takes `iterator=` defining what one row of the answer is:

- `iterator=20` - tile the scope at 20bp. One row per bin. *Chromosome-relative*, not interval-relative.
- `iterator="some.track"` - use the track's stored bin grid (one row per stored position; common with CpG tracks).
- `iterator=<intervals frame>` - one row per interval. With `intervals=<same frame>`, this is the universal "one row per peak" recipe.
- `iterator=pm.giterator_cartesian_grid(...)` - 2D pair grid (see [pymisha-advanced.md](pymisha-advanced.md)).

`intervals=` is the *scope*; `iterator=` is the *resolution*. They are independent.

**`intervals=` is mandatory for most query functions.** `gextract`, `gscreen`, `gsummary`, `gdist`, `gintervals_neighbors`, `gintervals_annotate`, etc. don't default to "whole genome" - if you want the walk to cover everything, pass `intervals=pm.gintervals_all()` (1D) or `intervals=pm.gintervals_2d_all()` (2D) explicitly. The same applies to `iterator=` when there's no natural per-row resolution.

### 1.6 Two rules that apply everywhere

**Rule 1 - prefer a virtual track over a hand-rolled computation.** Before round-tripping through Python for any per-bin or per-region computation (sums, percentiles, distances, kmer counts, PWM scores, log-sum-exp aggregates), check whether a vtrack `func=` already does it. Virtual tracks evaluate in C++ at the engine level and stay inside the single-genome-pass model that makes pymisha fast.

**Rule 2 - unify into one `gextract` call.** When you need several values per region (or per bin), pass *all* the expressions to a single `gextract` as a `["expr1", "expr2", ...]` list. The engine evaluates them jointly in one genome pass; N separate calls are N× slower and require a manual join on chrom/start/end (or `intervalID`) afterwards.

## 2. Session bootstrap

```python
import pymisha as pm

pm.gsetroot("db")                       # or absolute path; symlink "db" from project root
pm.gdataset_load("db_extra")            # optional, project-local dataset

pm.CONFIG["max_data_size"] = 10_000_000_000      # raise from default for large gextract returns
pm.CONFIG["multitasking"] = True                 # parallel across chroms (default; explicit)
pm.CONFIG["multitasking_strategy"] = "auto"      # or "tracks" for many-track scans
```

Attach the genome you need once at the top and stay on it. Switching `gsetroot` mid-script while intervals from the previous genome are still in scope is the most reliable way to produce silent coordinate drift.

**`multitasking` and external parallelism.** pymisha's `CONFIG["multitasking"] = True` forks worker processes internally. When you then wrap pymisha calls in `multiprocessing.Pool` (or `joblib.Parallel` with `prefer="processes"`, or `concurrent.futures.ProcessPoolExecutor`), the nested forks can deadlock or oversubscribe cores. Set `pm.CONFIG["multitasking"] = False` inside the worker body and let the outer layer do the parallelism.

**Memory balance.** `max_data_size` caps the *result size* of any single `gextract`. With `multitasking = True`, several extractions run in parallel and each holds up to that much in memory. A high `max_data_size` combined with `multitasking = True` on a heavy `gextract` can blow past available RAM and surface as `MemoryError` or OOM kill. If you hit it: lower `max_data_size`, drop `multitasking`, or partition by chromosome.

**Many-track strategy.** `pm.CONFIG["multitasking_strategy"] = "auto" | "tracks" | "tiles"` controls how the engine parallelizes a multi-track `gextract`. `"tiles"` (the historical default) parallelizes across genome tiles within one expression at a time - good for a few heavy tracks. `"tracks"` parallelizes across the *expressions* in `["...", "..."]` instead - dramatically faster for thousands of motif / feature tracks where each is cheap individually. `"auto"` picks per call. For motif-scan workloads (hundreds of PSSMs over the genome) set `"tracks"` explicitly.

## 3. Function chooser tables

Pick the verb by *what's returned*, not by alphabetical proximity.

### Workhorse verbs

| Verb | Returns | When |
|---|---|---|
| `gextract` | DataFrame | Values to look at, plot, or join. |
| `gscreen`  | intervals frame | Filter to a region set (peaks above threshold). |
| `gquantiles` | quantile cutoffs | Pick a threshold, normalize. Argument is `percentiles=` (0..1), **not** `probs=`. |
| `gdist`    | N-d count array (or DataFrame with `dataframe=True`) | Binned joint distribution (signal × distance, etc.). |
| `gsummary` | per-scope summary Series | One-shot min/max/mean/sum/Nbin (often before deciding `gdist` breaks). |
| `gintervals_summary` | per-*interval* min/max/mean/sum/sd/nbin | One row per input interval; one-call alternative to multi-expression `gextract`. |
| `gcor`     | scalar correlation (or matrix) | Pearson / Spearman correlation between two+ expressions. |
| `gintervals_neighbors` | intervals + nearest-feature cols | Nearest-feature annotation; supports `maxdist` / `mindist`. |
| `gintervals_annotate` | input rows + selected annot cols + `dist` | Column-attach without changing row count or order. |
| `gintervals_canonic` | merged intervals frame | After `pd.concat` / union when you want overlaps merged. |
| `gintervals_to_mat` / `gintervals_from_mat` | DataFrame indexed by intervals / data.frame | Round-trip intervals + values to a numeric matrix for clustering / heatmaps. |
| `gtrack_create` / `gtrack_smooth` | (side effect) | Materialize a derived track. |

### Distance - which primitive?

| Use | Returns | When |
|---|---|---|
| `gvtrack_create(_, src, "distance")` / `"distance.center"` / `"distance.edge"` | per-bin signed numeric column | You're sweeping the genome at fixed resolution and want distance as a feature alongside other tracks. |
| `gintervals_neighbors(a, b)` | paired-row frame (a + b + signed `dist`) | k-NN, distance bands via `mindist` / `maxdist`, asymmetric upstream/downstream. |
| `gintervals_annotate(intervs, annot, annotation_columns)` | input rows + selected annot cols + `dist` | Attach gene symbol + distance to a fixed query frame without changing row count. |

### Aggregators (`func=` on `gvtrack_create`)

| `func` | Use when |
|---|---|
| `"sum"`  | Count signals (ATAC, ChIP, contacts). Adds across bins in the window. |
| `"area"` | Width-aware sum for sparse / 2D sources where bin widths vary. |
| `"weighted.sum"` | Sparse / 2D contact aggregation against rectangles. |
| `"avg"`  | Continuous signal where you want a per-bin mean. **Do not use for count tracks.** |
| `"max"`  | Peak detection (PWM energies, percentile maxima). |
| `"min"`  | Local-min insulation border calling, diagnostic windows. |
| `"lse"`  | log-sum-exp - for summing motif energies (log-space). Composes correctly under nested aggregation. |
| `"global.percentile.max"` | Genome-wide percentile of max-in-window. Comparable across tracks. |

### Argument-name gotchas (silent footguns)

| Wrong | Right | Why it bites |
|---|---|---|
| `gquantiles(x, probs=0.9)` | `gquantiles(x, percentiles=0.9)` | `probs` is silently swallowed by `**kwargs`; default `percentiles=0.5` (median) is used. |
| `gextract("track > 5", ..., intervals_set_out="...")` and assume `intervalID` is dropped | Use `intervals_join="intervals"` to drop `intervalID` and attach input columns | Default `intervals_join="id"` keeps the `intervalID` column; downstream joins that match on `chrom/start/end` get the right shape but lose row identity if multiple iterator bins map to one input. |

## 4. Recipes

### 4.1 Build and manipulate intervals

**Build.**

```python
# 1D - vectorized; chroms can be a scalar or per-row. Argument names are PLURAL.
peaks = pm.gintervals(chroms=["chr1", "chr2"],
                      starts=[100,    200],
                      ends  =[500,    700])

# 2D - three canonical forms.
pm.gintervals_2d(chroms1, starts1, ends1, chroms2, starts2, ends2)   # explicit rectangles
pm.gintervals_2d(chroms1=chrs, chroms2=chrs)                          # whole-chrom cartesian
pm.gintervals_2d(chroms1=chrom, chroms2=chrom)                        # one whole chrom (cis)
```

**Load from file.** Dedicated readers for the three common annotation formats - output is a validated pymisha intervals frame:

```python
peaks = pm.gintervals_import_bed("peaks.bed", name=True, score=True, strand=True)
genes = pm.gintervals_import_gff("refseq.gff", feature="exon", attrs=True)
snps  = pm.gintervals_import_vcf("snps.vcf",  info=True)
```

**Sort and merge overlaps (`gintervals_canonic`).** When you want a non-overlapping, sorted set - typical after `pd.concat`'ing per-condition `gscreen` results. To fold per-row metadata into the merged set, use `pm.gintervals_mark_overlaps` (tags each source row with its merge-group ID) and aggregate per group:

```python
import pandas as pd

merged = (
    pm.gintervals_mark_overlaps(pd.concat([set_a, set_b], ignore_index=True))
    .groupby("overlap_group", as_index=False)
    .agg(chrom=("chrom", "first"),
         start=("start", "min"),
         end=("end", "max"),
         strand=("strand", lambda s: s.iloc[0] if s.nunique() == 1 else 0))
    .drop(columns="overlap_group")
)
```

`gintervals_mark_overlaps` is `gintervals_canonic` plus a join-friendly group column; reach for it when you need custom per-merged-region aggregation. Bare `gintervals_canonic(...)` is fine when you only want the merged intervals frame and don't need to carry source metadata.

If you want to *keep* duplicate / overlapping rows (paired matches, parallel rows per sample), skip canonic.

**Set operations.**

```python
pm.gintervals_union(a, b)         # union
pm.gintervals_intersect(a, b)     # intersection
pm.gintervals_diff(a, b)          # a minus b
```

**Fixed-width peaks (center ± W).**

```python
peaks_2k = pm.gintervals_normalize(peaks, 2000)   # 2kb-wide, centered on the original midpoint
```

For asymmetric expansion, do it manually with integer arithmetic - *never* float division:

```python
mid = (peaks["start"] + peaks["end"]) // 2
peaks = pm.gintervals(chroms=peaks["chrom"],
                      starts=mid - W_up,
                      ends  =mid + W_down + 1)
peaks = pm.gintervals_canonic(peaks)
```

**Clip to genome bounds.** `pm.gintervals_force_range(intervs)` clamps `start >= 0` and `end <= chrom_len`. Use after any expansion that may push past chromosome ends.

**Symmetric expansion.** No one-call helper - the manual form is short:

```python
expanded = peaks.assign(start=peaks["start"] - 100, end=peaks["end"] + 100)
expanded = pm.gintervals_force_range(expanded)
```

**Persist.**

```python
pm.gintervals_save("intervs.my.peaks", peaks)
my_peaks = pm.gintervals_load("intervs.my.peaks")
```

**Avoid:**
- `(start + end) / 2` - float division yields non-integer centers. Always `(start + end) // 2`.
- `gintervals_canonic` reflexively after every `pd.concat` - only run it when you actually want overlaps merged.
- Base `pd.DataFrame({"chrom": ..., "start": ..., "end": ...})` skips coordinate validation against the active genome. Pass through `pm.gintervals(...)`; it validates chrom names and clamps types.
- Hand-rolling fixed-width peaks via `assign` when `pm.gintervals_normalize(intervs, size)` does the same in one call.

### 4.2 Annotate intervals with nearby features

Two primitives, different return shapes:

- **`gintervals_neighbors(query, target, ...)`** - paired-row output, one input row × matched neighbor; signed `dist` column. Good for k-NN, distance-banded matching, asymmetric upstream/downstream.
- **`gintervals_annotate(intervals, annotation_intervals, annotation_columns=, ...)`** - column-attach output, preserves row count and order. Good for "add gene symbol + signed distance".

```python
tss = pm.gintervals_load("intervs.global.tss")

# Nearest neighbor with signed distance, dropping unmatched rows:
near = pm.gintervals_neighbors(peaks, tss)

# Distance-banded - keep all peaks (na_if_notfound is the key flag):
near = pm.gintervals_neighbors(peaks, tss, maxdist=50_000, na_if_notfound=True)

# k nearest neighbors:
knn = pm.gintervals_neighbors(peaks, tss, maxneighbors=5)

# Strand-aware variants:
pm.gintervals_neighbors_upstream(peaks, tss, maxdist=100_000)
pm.gintervals_neighbors_downstream(peaks, tss, maxdist=100_000)
pm.gintervals_neighbors_directional(peaks, tss,
    maxneighbors_upstream=1, maxneighbors_downstream=1)

# Column attach - keeps row count and order:
peaks = pm.gintervals_annotate(peaks, tss,
                               annotation_columns=["geneSymbol"],
                               dist_column="tss_dist",
                               max_dist=100_000,
                               na_value=float("nan"))
```

Signed-distance convention: positive `dist` means the target is downstream of the query (in genome coordinates if query has no strand; in transcription direction if `use_intervals1_strand=True`).

**Promoters from TSS.** Manual one-step recipe:

```python
import numpy as np

tss = pm.gintervals_load("intervs.global.tss")
# upstream=500, downstream=50, strand-aware:
upstream, downstream = 500, 50
fwd = tss["strand"].fillna(1) >= 0
prom = tss.assign(
    start=lambda d: np.where(fwd, d["start"] - upstream, d["end"] - downstream),
    end  =lambda d: np.where(fwd, d["start"] + downstream, d["end"] + upstream),
)
prom = pm.gintervals_force_range(prom)
```

Reuse this as the source for `intervs.global.tss` derived analyses (overlap with peaks, distance-banded screens, etc.).

**Snap intervals to nearest landmark.** Different shape from `gintervals_annotate`: instead of attaching a neighbor column, *replace* each interval's coordinates with its nearest match in another set when within a distance band. Manual form:

```python
import numpy as np

near = pm.gintervals_neighbors(peaks, ctcf_motifs,
                               mindist=-100, maxdist=100,
                               na_if_notfound=True)
# Replace peaks coords with the motif coords when a match exists:
snapped = peaks.copy()
match = near["chrom1"].notna()
snapped.loc[match, ["chrom", "start", "end"]] = (
    near.loc[match, ["chrom1", "start1", "end1"]].to_numpy()
)
```

Same row count and order as the input; coordinates are rewritten only for rows that found a match. Use when downstream code needs canonical landmark coordinates (anchor-pair pile-ups, motif-centered meta-profiles); use `gintervals_annotate` when you just want to attach the landmark's distance/identity without changing the query's coordinates.

**Avoid:**
- `gintervals_neighbors` without `na_if_notfound=True` when annotating a fixed set - rows with no neighbor in range are dropped silently.
- `gintervals_neighbors` when you only want to *attach* columns to an existing frame - `gintervals_annotate` is the right tool.
- Forgetting that `dist` is *signed*. Filter with `abs(dist) < W`; bare `dist < W` accepts arbitrary upstream distances.

### 4.3 Distance flavors

`gvtrack_create` accepts three `func` values for distance, with different semantics:

| `func` | Measures | Returns 0 when | Notes |
|---|---|---|---|
| `"distance"` | Iterator-bin *center* → nearest source *edge* (outside); *normalized* fractional position when inside. | Bin center exactly on a source edge | Mixed semantics - fine for genome-wide profiles, surprising if you assumed pure edge-to-edge. |
| `"distance.center"` | Iterator-bin *center* → nearest source *center*. | Bin center coincides with a source center | Use for anchor-to-anchor offsets in meta-profiles. |
| `"distance.edge"` | *Edge-to-edge*, same as `gintervals_neighbors`. | Iterator interval overlaps the source | Use when you specifically want `gintervals_neighbors` semantics inside a vtrack. |

All three return *signed* distance when the source has a `strand` column (sign = direction relative to strand); unsigned otherwise. All three return NaN when the chromosome has no source intervals.

```python
pm.gvtrack_create("d_tss", src="intervs.global.tss", func="distance")

# Per-bin distance - one column alongside other tracks:
pm.gextract(["epi.atac.es", "d_tss"], intervals=peaks, iterator=20)

# Distance-banded screen:
pm.gscreen("epi.atac.es > 5 & abs(d_tss) > 5000",
           intervals=pm.gintervals_all(), iterator=20)
```

For 2D, bind a 1D `distance` vtrack to one axis of a 2D iteration with `pm.gvtrack_iterator(name, dim=1)` or `dim=2`.

**Avoid:**
- Treating distance vtrack values as unsigned - `abs(d_tss) < W` is almost always what you want.
- Forgetting NaN on chromosomes with no source intervals. Wrap with `ifelse(is.na(d_x), 1e18, d_x)` inside the expression if you want "no neighbor = infinitely far".

### 4.4 Extract per-region values

Signature: `gextract(expr, intervals, iterator, colnames, band, vars, intervals_join, file, intervals_set_out)`. `expr` is a string or list of strings - bare track names, vtrack names, or arithmetic involving them.

**Many tracks, one call - the central concept.** A single `gextract` over a `["expr1", "expr2", ...]` list makes ONE pass over the genome:

```python
profs = pm.gextract(["epi.k27me3.es", "epi.k4me3.es", "atac.es"],
                    intervals=peaks,
                    iterator=peaks,
                    colnames=["k27", "k4", "atac"])
```

**`colnames=` is how you name the output columns.** Without it, each column is named by the literal expression string - fine for bare track names, ugly for anything with arithmetic. Always pass `colnames` when expressions are not bare track names.

**One row per region.** Pass the same intervals frame as both `intervals=` and `iterator=`:

```python
pm.gvtrack_create("k27", "epi.k27me3.es", "sum", sshift=-2000, eshift=2000)
pm.gvtrack_create("atac", "atac.es",       "sum", sshift=-250,  eshift=250)

per_peak = (pm.gextract(["k27", "atac"],
                        intervals=peaks, iterator=peaks,
                        colnames=["k27_flank", "atac_summit"])
            .sort_values("intervalID")
            .drop(columns="intervalID"))
```

`.sort_values("intervalID").drop(columns="intervalID")` recovers input row order and drops the helper column.

**Attach the input intervals' columns directly (no Python join).** When you want the input `peaks` columns - coords, gene names, scores - on every output row, pass `intervals_join="intervals"`:

```python
per_peak = pm.gextract(["k27", "atac"],
                       intervals=peaks, iterator=peaks,
                       colnames=["k27_flank", "atac_summit"],
                       intervals_join="intervals")
# per_peak has chrom/start/end + k27_flank + atac_summit + chrom1/start1/end1
# + every other column of `peaks` (e.g. gene_id, score, ...)
```

`intervalID` is dropped; conflicting names from `peaks` get a `"1"` suffix (`chrom` -> `chrom1`). Output rows are in genomic-sort order, not original `peaks` order. The C++ side copies the input columns by `intervalID` index, so this avoids the Python-side `pd.merge` round-trip and is the recommended path. Use `intervals_join="none"` to just drop `intervalID` without attaching anything. Only available when returning to memory; combining with `file=` or `intervals_set_out=` raises an error.

**NaN handling - crucial gotcha.** Sparse tracks return NaN at unmeasured bins; many dense tracks also have NaNs. Fill at extract time, inside the expression string:

```python
pm.gextract(["ifelse(is.na(epi.atac.es), 0, epi.atac.es)",
             "ifelse(is.na(cpgs.meth),  0, cpgs.meth)"],
            intervals=peaks, iterator=peaks,
            colnames=["atac", "meth"])
```

Treat this as a default, not an edge case. Expression strings stay R-like (`ifelse`, `is.na`) regardless of whether you call them from R or Python - they're parsed by the same misha expression engine in C++.

**Expression functions must be vectorized.** The C++ engine passes track *vectors* to your expression and expects a same-length vector back. Standard vectorized expression-language ops all work (`+`, `-`, `*`, `/`, `<`, `>`, `&`, `|`, `ifelse`, `pmin`, `pmax`, `log`, `log2`, `exp`, `abs`). The pitfall is scalar-result functions like `mean(track)`, `min(track)`, `sum(track)`: those collapse the vector to one number, which the engine then recycles across every bin. There's no `pmean`; for per-element averaging across two tracks, hand-roll: `"ifelse(is.na(a), b, ifelse(is.na(b), a, (a + b) / 2))"`. (`(a + b) / 2` alone is wrong: it propagates NaN whenever *either* input is NaN, so any position where one track has data and the other doesn't comes out as NaN instead of the value you have.)

**Tiled bins inside intervals.** For sub-region resolution (per-bin profile), pass an integer iterator:

```python
prof = pm.gextract("epi.k27me3.es",
                   intervals=regions,
                   iterator=20)        # one row per 20bp bin within regions
```

**Iterator coordinate gotcha.** `iterator=20` tiles bins relative to **chromosome start**, not to each interval's start - so two peaks at different chromosome offsets get *misaligned* bin grids. For interval-relative bin indices use `pm.giterator_intervals(intervals=peaks, iterator=20, interval_relative=True)`.

**One row per stored position (`iterator=<track>`).** For sparse-aligned data - per-CpG methylation, per-fragment coverage:

```python
m = pm.gextract("cpgs.meth", intervals=pm.gintervals_all(), iterator="cpgs.cov")
```

**Per-interval descriptive stats in one call.** When you want `nbins / n_nan / min / max / sum / mean / sd` per input region, `gintervals_summary(expr, intervals, iterator)` returns one row per interval with all of them appended:

```python
stats = pm.gintervals_summary("epi.k27me3.es", intervals=peaks, iterator=20)
# stats has chrom/start/end + nbins, nbins.nan, min, max, sum, mean, stdev
```

This subsumes a multi-expression `gextract` for the standard descriptive-stats case. Reach for the multi-expression form only when you need non-stat aggregates (custom expressions, NA-fills, ratios) at the same time.

**Persist as a named intervals set.**

```python
pm.gextract("score > 5", intervals=pm.gintervals_all(), iterator=20,
            intervals_set_out="intervs.high_score")
```

**Reshape to a matrix for downstream analysis.** After `gextract`, the typical next step for clustering / heatmaps is a numeric matrix indexed by interval. `gintervals_to_mat` and its inverse `gintervals_from_mat` round-trip cleanly:

```python
mat = pm.gintervals_to_mat(profs)               # N peaks x K tracks; MultiIndex (chrom, start, end)
# or with explicit row IDs:
mat = pm.gintervals_to_mat(profs, id_col="gene")
out = pm.gintervals_from_mat(mat.iloc[order])   # back to intervals + values
```

Pass `labels=False` to skip MultiIndex construction (faster on million-row inputs when you don't need to look at the labels). Pass `value_cols=["t1", "t2"]` to subset which numeric columns go into the matrix. Round-trip is lossless on chrom names containing underscores (`chrUn_KI270442v1`, scaffolds) - the intervals are carried in `df.attrs["intervals"]`, not parsed back from labels.

**Avoid:**
- Looping `gextract` over a list of tracks. Always one call with `["track1", "track2", ...]`.
- Post-extract `prof.fillna(0)` when you can write `"ifelse(is.na(track), 0, track)"` inside the expression.
- Scalar-result functions (`mean(track)`, `min(track)`, `sum(track)`) inside an expression - they collapse per-bin vectors.
- `iterator=peaks` when you want sub-region resolution. Use `iterator=20`.
- Forgetting that `iterator=20` is chromosome-relative, not interval-relative.
- Skipping `.sort_values("intervalID")` when downstream code assumes input row order.

### 4.5 Aggregate signal in a flanking window

One `gvtrack_create` per aggregator + window combination, then `gextract`:

```python
# Window-summed ChIP signal in ±2kb:
pm.gvtrack_create("k27_flank", src="epi.k27me3.es", func="sum",
                  sshift=-2000, eshift=2000)

# Sum ATAC counts in the summit (-250 / +250bp):
pm.gvtrack_create("atac_summit", src="atac.es", func="sum",
                  sshift=-250, eshift=250)

# Max PWM energy across ±100bp:
pm.gvtrack_create("ctcf_max", src="motifs.ctcf", func="max",
                  sshift=-100, eshift=100)

per_peak = pm.gextract(["k27_flank", "atac_summit", "ctcf_max"],
                       intervals=peaks, iterator=peaks)
```

**Window placement.** Symmetric `sshift=-W, eshift=W` is the default. For asymmetric (upstream-only DI, strand-bias diagnostics) use `sshift=-W, eshift=0` or `sshift=0, eshift=W`. Point-sample at the iterator bin: omit shift args. 2D rectangle: use `pm.gvtrack_iterator_2d(name, sshift1=, eshift1=, sshift2=, eshift2=)` after `gvtrack_create`.

**Avoid:**
- `gvtrack_create(...); gvtrack_iterator(name, sshift=, eshift=)` as two calls - current pymisha takes `sshift` / `eshift` directly in `gvtrack_create` for the 1D case.
- Defensive `if name in pm.gvtrack_ls(): pm.gvtrack_rm(name)` before every `gvtrack_create` - current pymisha silently overwrites on re-create.
- Picking `"avg"` when you mean `"sum"` - for count tracks, `avg` divides by bin count and discards magnitude.
- Forgetting that `sshift` / `eshift` are in *base pairs*, not bin counts.

### 4.6 Call regions of interest

`gscreen(expr, intervals, iterator, intervals_set_out)`. Bins where the expression evaluates to non-zero / non-NaN become intervals (adjacent bins auto-merged).

**Pick threshold from a quantile, then screen.** Almost never hardcode a threshold - derive it from the empirical distribution:

```python
thr = pm.gquantiles("epi.k27me3.es",
                    percentiles=0.99,
                    intervals=pm.gintervals_all())

peaks = pm.gscreen(f"epi.k27me3.es > {thr:g}",
                   intervals=pm.gintervals_all(),
                   iterator=20)
```

`gquantiles` argument is **`percentiles=`** (0..1), not NumPy/pandas's `q=`/`probs=`. For *per-region* quantiles (one row per interval), reach for `pm.gintervals_quantiles` instead - same arg name, dedicated to the per-interval case.

**Multi-track OR.** Build a `" | "`-joined expression so one `gscreen` returns the union of per-track hits:

```python
expr = " | ".join(f"({t} > {thr:g})" for t, thr in zip(track_names, thrs))
peaks = pm.gscreen(expr, intervals=pm.gintervals_all(), iterator=20)
peaks = pm.gintervals_canonic(peaks)
```

**Refine peak centers by argmax.**

```python
summits = (pm.gextract("epi.k27me3.es", intervals=peaks, iterator=20)
           .loc[lambda d: d.groupby("intervalID")["epi.k27me3.es"].idxmax()]
           .loc[:, ["chrom", "start", "end"]]
           .pipe(pm.gintervals_normalize, 280))
```

**Persist as a named set.**

```python
pm.gscreen("epi.k27me3.es > 5",
           intervals=pm.gintervals_all(),
           iterator=20,
           intervals_set_out="intervs.k27.peaks")
```

**Distance-band screen.**

```python
pm.gvtrack_create("d_tss",  "intervs.global.tss",         "distance")
pm.gvtrack_create("d_rmsk", "intervs.global.rmsk_LINE",   "distance")
pm.gscreen("abs(d_tss) > 5000 & abs(d_rmsk) > 100",
           intervals=peaks, iterator=20,
           intervals_set_out="intervs.peaks.intergenic")
```

**Avoid:**
- Hardcoded literal thresholds (`gscreen("track > 1870", ...)`) - uninterpretable across datasets; derive from `gquantiles`.
- `gquantiles(x, probs=0.9)` - wrong argument name. Always `percentiles=`.
- `gquantiles` on a track with many NaNs assumed to include zeros. NaN positions are silently dropped. Wrap with `"ifelse(is.na(track), 0, track)"` if zeros should count.

### 4.7 Joint distributions, summaries, correlations

Four primitives:

- **`gdist(track1, breaks1, track2, breaks2, ..., intervals, iterator)`** - N-d count histogram. Pass `dataframe=True` for a long-format frame with one column per axis + count column `n`.
- **`gbins_summary(strat_track, breaks, expr=value_track, ...)`** - per-bin n/mean/sum/var/min/max of `value_track` stratified by `strat_track` bins.
- **`gintervals_quantiles(expr, percentiles, intervals, iterator)`** - per-region quantile vector.
- **`gcor(expr1, expr2, ..., method)`** - Pearson / Spearman correlation between two+ expressions.

```python
import numpy as np

# Joint signal × distance histogram:
pm.gvtrack_create("d_tss", "intervs.global.tss", "distance")
h = pm.gdist("epi.k27me3.es", np.r_[-np.inf, np.arange(0, 10.5, 0.5), np.inf],
             "d_tss",          [-np.inf, -5e4, -1e4, -1e3, 0, 1e3, 1e4, 5e4, np.inf],
             intervals=pm.gintervals_all(), iterator=20,
             dataframe=True, names=["signal", "dist_tss"])

# Stratified mean signal per distance bin:
summ = pm.gbins_summary("d_tss", np.arange(-1e6, 1e6 + 1e4, 1e4),
                        expr="epi.k27me3.es",
                        intervals=pm.gintervals_all(), iterator=20)

# Per-region quantiles (one row per peak):
q = pm.gintervals_quantiles("epi.atac.es",
                            percentiles=[0.5, 0.9, 0.99],
                            intervals=peaks, iterator=20)

# Correlation between two tracks (Spearman, genome-wide):
pm.gcor("epi.k27me3.es", "epi.atac.es",
        intervals=pm.gintervals_all(), iterator=20,
        method="spearman")
```

**Picking `gdist` breaks.** Continuous tracks: `np.linspace(0, 1, 21)`. Count tracks: zero-vs-positive split + log spacing. Always `gsummary` first to pick range; never hardcode upper bound.

**Avoid:**
- Reaching for `gdist` when one axis is continuous and you want a per-bin mean - `gbins_summary` is the right tool.
- Hardcoded `breaks` without `gsummary` first - produces overflow / underflow bins silently.
- Mistaking `gdist`'s output for a probability - it's *counts*, not a density.

### 4.8 Materialize derived tracks

Four creation primitives:

- **`gtrack_create(track, description, expr, iterator)`** - evaluate an expression genome-wide and write the result as a dense track.
- **`gtrack_create_sparse(track, description, intervals, values)`** - values at irregular positions; NaN elsewhere.
- **`gtrack_create_dense(track, description, intervals, values, binsize, defval, func)`** - fully dense fixed-bin track from interval/value pairs. `func` ∈ {`"weighted.mean"` (default), `"weighted.sum"`, `"coverage"`, `"max"`, `"min"`, `"median"`, `"count"`}.
- **`gtrack_smooth(track, description, expr, winsize, alg)`** - windowed-mean track. `alg="LINEAR_RAMP"` (triangular, default) or `"MEAN"` (boxcar).

```python
pm.gtrack_create("epi.k27me3.log",
                 description="log2(K27me3 + 1)",
                 expr="log2(epi.k27me3.es + 1)",
                 iterator=20)
```

**Indicator from an intervals frame - no materialization.** Add a `value` column and pass the frame as `src` of a vtrack:

```python
peaks_ind = peaks.assign(value=1)
pm.gvtrack_create("peak_ind", src=peaks_ind, func="max")
pm.gextract("ifelse(is.na(peak_ind), 0, peak_ind)",
            intervals=pm.gintervals_all(), iterator=20)
```

Materialize an on-disk indicator track only when the indicator is reused across sessions or a downstream consumer specifically needs a stored track.

**Smooth an existing track.**

```python
pm.gtrack_smooth("epi.k27me3.smooth",
                 description="K27me3 ±20kb LINEAR_RAMP",
                 expr="epi.k27me3.es",
                 winsize=20000)
```

**Namespace via directories.** Long names with dots map to directories on disk:

```python
pm.gtrack_create_dirs("epi.derived")
pm.gtrack_create("epi.derived.k27_over_atac",
                 description="K27me3 over ATAC ratio",
                 expr="epi.k27me3.es / (atac.es + 1)",
                 iterator=20)
```

**Idempotency guard.**

```python
if pm.gtrack_exists("derived"):
    pm.gtrack_rm("derived", force=True)
pm.gtrack_create("derived", "...", expr="...", iterator=20)
```

**Persist track metadata.** `pm.gtrack_attr_set(track, key, value)` / `pm.gtrack_attr_get(track, key)` stores per-track scalar metadata (training parameters, source paths, dates). The intervals-set parallel is `pm.gintervals_attr_set(set, key, value)` / `pm.gintervals_attr_get(set, key)` for annotation metadata on named intervals sets (source build, filter version, etc.) - same convention.

**Avoid:**
- Materializing a "track" used once or twice - for ephemeral signals, a vtrack (in-memory) is the right tool.
- `gtrack_create(... expr="a / b")` without NaN-safe divisor - wrap as `"a / (b + 1e-6)"` at creation time.
- `gtrack_rm("name")` without `force=True` in a script - the interactive confirmation prompt hangs unattended runs.
- Aggregating a stored `pwm` (LSE) track with `func="sum"` - LSE doesn't compose under summation. Use `func="lse"`.
