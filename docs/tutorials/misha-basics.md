# Misha Basics (Short Guide)

This page gives a compact mental model for misha/PyMisha.  
Use it as the "first 20 minutes" before the full [User Manual](manual.md).

## The Core Idea

Most analyses follow the same pattern:

1. Choose **where** to look (intervals / scope).
2. Choose **how** to walk through it (iterator).
3. Evaluate a **track expression** over those iterator intervals.

In PyMisha, this is usually one call to `pm.gextract()`, `pm.gscreen()`, or `pm.gsummary()`.

You are not limited to raw track names. You can pass full expressions, for example
`np.log(dense_track + 1)`, `dense_track / (chip.sum + 1e-6)`, or
`np.minimum(dense_track, 2.0)`.

All examples below use the bundled example database:

```python
import pymisha as pm

pm.gdb_init_examples()
```

## Four Concepts You Need First

### 1) Track

A **track** is genomic signal organized over coordinates.

- Dense track: value for each fixed-size bin (for example `dense_track` in the examples DB).
- Sparse track: values on intervals (for example peaks).
- 2D track: values on genomic rectangles (for example contact matrices).

Useful starter commands:

```python
pm.gtrack_ls()                    # list tracks in the examples DB
pm.gtrack_info("dense_track")     # inspect type/metadata
pm.gtrack_info("sparse_track")
```

For intuition, you can think of `dense_track` as a ChIP-seq-like coverage signal.

### 2) Intervals

An **interval set** defines genomic regions (`chrom`, `start`, `end`) where you want to work.

- Intervals can come from files, annotations, peak calls, or be generated in code.
- Intervals often act as a **scope**: "analyze only here."

```python
regions = pm.gintervals_from_strings(["chr1:0-100000", "chr1:250000-260000"])
```

### 3) Iterator

The **iterator** is the stepping policy inside the scope.

- `iterator=100` -> fixed 100 bp bins
- `iterator="some_sparse_track"` -> iterate over that track's intervals
- `iterator=some_intervals_df` -> iterate over explicit regions
- `iterator="my_intervals_set"` -> iterate directly over an intervals set

Think of it as: scope says *where*, iterator says *in what chunks*.

```python
out = pm.gextract("dense_track", regions, iterator=100)
log_out = pm.gextract("np.log(dense_track + 1)", regions, iterator=100)
```

### 4) Virtual Track

A **virtual track** is a named on-the-fly transformation, not stored as a physical track file.

Examples:
- Local sum of a source track
- Distance to nearest annotation interval
- Quantile-like or nearest-neighbor summaries

```python
pm.gvtrack_create("chip.sum", "dense_track", "sum")
out = pm.gextract("chip.sum", regions, iterator=200)
```

You can also shift the iterator window used by the virtual track:

```python
pm.gvtrack_create("chip.shifted", "dense_track", "sum", sshift=-100, eshift=100)
out = pm.gextract("chip.shifted", regions, iterator=200)
```

Here, each iterator interval is expanded by 100 bp on both sides before evaluating `dense_track`.

Virtual tracks are session objects (easy to iterate with, easy to delete with `pm.gvtrack_rm()`).

## Minimal Workflow

```python
import pymisha as pm

pm.gdb_init_examples()

# 1) pick scope
regions = pm.gintervals_from_strings(["chr1:0-50000"])

# 2) inspect available tracks
print(pm.gtrack_ls())

# 3) extract signal with a chosen iterator
chip = pm.gextract("dense_track", regions, iterator=100)

# 4) screen high-signal bins (as a simple peak-like filter)
hi_chip = pm.gscreen("dense_track > 0.6", regions, iterator=100)

# 5) summarize distribution/coverage
stats = pm.gsummary("dense_track", regions, iterator=100)
```

## Where Tracks Usually Come From

- Existing tracks in the DB (`pm.gtrack_ls()`), including shared/reference tracks.
- Imported tracks from bedGraph/BED-like files (`pm.gtrack_import()` / `pm.gtrack_import_set()`).
- New tracks derived from expressions (`pm.gtrack_create()`).
- Non-persistent virtual tracks (`pm.gvtrack_create()`).


