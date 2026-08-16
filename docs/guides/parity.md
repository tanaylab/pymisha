# Parity Notes

PyMisha targets full functional parity with R misha. Nearly all of R's public
API is covered, with C++ backends for the heavy paths (track extraction, the 2D
quadtree scanner and its iterators, liftover, SAM import, array tracks, virtual
tracks). This page lists the **remaining divergences** - everything not on it
should behave as in R; if you find a difference that isn't documented here,
please file an issue.

## Argument shape: `gextract` does not take R's `...` call shape

R misha's `gextract` is defined as `gextract(..., intervals = NULL, iterator =
NULL, ...)`: any number of leading *unnamed* arguments are track expressions,
and `intervals`/`iterator` are matched afterward. `gdist` is defined the same
way (`gdist(..., intervals = NULL, ...)`) - these are the only two
track-expression functions in R misha that are actually variadic. `gscreen`,
`gsummary`, `gquantiles` and `giterator.intervals` are **not** variadic in R:
each takes a single fixed `expr` argument, same as their pymisha
counterparts, so the trap below does not apply to them. R's own docs use the
variadic `gextract` shape directly, e.g.
`gextract("dense_track", "sparse_track", intervals)` for two expressions.

PyMisha's `gextract` instead takes a single fixed `expr` argument (a string or
a list of strings) followed by fixed `intervals=` / `iterator=` arguments:
`gextract(expr, intervals=None, iterator=None, ...)`. Porting R's shape
literally does **not** work:

```python
# WRONG - looks like R, compiles, runs, returns a result with no error:
pm.gextract("dense_track", "sparse_track", my_intervals)
```

This binds `"sparse_track"` to pymisha's `intervals` argument and `my_intervals`
to `iterator`, instead of evaluating two expressions over `my_intervals`. It
does not raise, because a bare track (or virtual-track) name is itself a
**valid, deliberate** value for `intervals` in pymisha: it means "use that
track's own defined intervals as the scope" (R parity - see the "iterator and
scope may be a track or interval-set name" entry in `CHANGELOG.md` v0.4.0).
So the call above silently drops the second expression and silently rescopes
to `"sparse_track"`'s own domain instead. There is no way to catch this from
the argument value alone - a deliberate track-name-as-scope call looks
identical to the accidental case - so this is a real porting trap, not a bug
that a stricter check can fix.

The correct pymisha form passes multiple expressions as a **list**, with the
scope given by keyword:

```python
# RIGHT:
pm.gextract(["dense_track", "sparse_track"], intervals=my_intervals)
```

This affects any function with a fixed `expr: str | list[str]` argument
followed by a fixed `intervals` argument (currently `gextract` and
`giterator_intervals_2d`). `gscreen`, `gsummary`, `gquantiles` and
`giterator_intervals` have the same argument shape - a fixed `expr`
followed by a fixed positional `intervals` - but the trap cannot arise for
them, because their R counterparts are not variadic either: there is no
R call shape with a second expression to mis-bind in the first place.
`gdist` is variadic in R, and pymisha's `gdist` uses Python's own `*args`
for the same purpose, so it ports directly.

## Partially covered

- **COMPUTED 2D tracks** -- PyMisha **reads** COMPUTED tracks backed by
  `AreaComputer2D` / `TestComputer2D` (`gextract`, `gsummary`, `gquantiles`,
  `gscreen`), parsing the COMPUTED file format and recomputing the per-rectangle
  value on a query/band mismatch as R does. **Not** supported: the Hi-C
  normalization computers `PotentialComputer2D` / `TechnicalComputer2D`, and
  *creating* COMPUTED tracks (R exposes no public creation API either - the
  [shaman](https://github.com/tanaylab/shaman) Hi-C tool uses plain 2D tracks).

- **`gtrack.convert`** (legacy 2D format upgrade). Reading or upgrading the
  obsolete `OLD_RECTS1/2` / `OLD_COMPUTED1/2/3` trackdb formats is not
  implemented; the error message directs you to R misha's `gtrack.convert`. No
  misha version has written these formats in years.

## Not yet implemented

- **C++ `gtrack.import` for WIG / BedGraph / BigWig / BED / tab.** These formats
  are parsed in pure Python today; results match R but the throughput gap shows
  on multi-GB inputs. (Liftover, SAM `gtrack.import_mappedseq`, 2D extraction
  and the array/virtual-track paths already run in C++.)

- **R `gtrack.var` ASCII serialize variants** (`A\n`, `B\n`). PyMisha reads R's
  XDR binary and gzip-RDS variable formats via its native reader; the rare ASCII
  format is not decoded. Workaround: re-write with
  `serialize(value, con, ascii = FALSE)` in R.

- **`pwm.n_mutations`** virtual-track function (R misha's `Mode::N_MUTATIONS`).
  Not implemented in PyMisha; the other `pwm.edit_distance*` functions are.

## Data-shape differences

- **`chrom` column dtype.** PyMisha writes the `chrom` / `chrom1` / `chrom2`
  column of a saved interval set as **character**; R misha writes a **factor**
  whose levels are the genome's full `ALLGENOME` chromosome set. R reads the
  character form fine on every path we exercise, but code that relies on
  `levels()` or on factor codes will see the difference.

## Numerical reproducibility

These are not missing features - the functions work and match R's semantics -
but results are not bit-identical to R:

- **Randomized functions** (`gintervals_random`, `gsample`, `gsynth_random`, ...)
  draw from NumPy's RNG, not R's. A given seed produces a valid, correctly
  distributed result, but not the same draws as R for the same seed. Set
  `numpy.random.seed(...)` to make PyMisha runs reproducible.
- **Tie-breaking in nearest-neighbor queries** (`gintervals_neighbors`, distance
  virtual tracks): distances match R exactly, but when several neighbors are
  equidistant the order in which ties are returned can differ.

## Not planned (R-specific or supplanted)

- **`gcluster.run`** -- R-specific SGE/PBS wrapper. Python users drive their own
  schedulers (snakemake, nextflow, dask, ...).

- **`gwget`** -- R wget shim. PyMisha downloads via Python's stdlib HTTP, so the
  shim is unnecessary.

- **`gdb.install_gff3_converter` / `gdb.install_gtf_converter`** -- these install
  UCSC's `gff3ToGenePred` / `gtfToGenePred` binaries. PyMisha parses GFF/GTF
  natively (`pymisha/genome/_gtf.py`), so the converters are not needed.

## How parity is verified

The `tests/r_parity` suite is what actually checks this page's claims: it runs
PyMisha's equivalent of each R `expect_regression` call against R misha's own test
database and compares to R's frozen `.rds` baselines.

**It runs only on lab infrastructure, and CI does not run it.** It needs R misha's
test database and its baseline store (about 24 GB), both on lab NFS, and skips
itself when they are absent - which is every GitHub CI job. A green CI run means
the unit tests, linting and type checks passed; it is not evidence that parity
holds. Run the suite before releasing:

```bash
pytest tests/r_parity -q          # lab hosts only
```

Set `PYMISHA_R_TESTDB` and `PYMISHA_R_SNAPSHOT_DIR` if either lives somewhere
other than its default path.
