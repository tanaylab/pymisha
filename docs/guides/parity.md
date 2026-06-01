# Parity Notes

PyMisha targets full functional parity with R misha. Nearly all of R's public
API is covered, with C++ backends for the heavy paths (track extraction, the 2D
quadtree scanner and its iterators, liftover, SAM import, array tracks, virtual
tracks). This page lists the **remaining divergences** - everything not on it
should behave as in R; if you find a difference that isn't documented here,
please file an issue.

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
