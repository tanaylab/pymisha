# Parity Notes

PyMisha targets full functional parity with R misha. The remaining gaps are
documented here. The per-feature roadmap lives in
`dev/notes/2026-05-15-functional-parity-audit.md` on the development
branch of the source repo (not bundled with the published site).

## Newly covered (v0.1.55 - v0.1.59)

- **Array tracks** -- `gtrack_array_extract`, `gtrack_array_get_colnames`,
  `gtrack_array_set_colnames`, `gtrack_array_create`. Read and in-memory write
  paths are byte-compatible with R misha (verified via cross-language
  `.colnames` roundtrip).
- **R-serialize reader** -- `pymisha._r_serialize.read()` decodes R's XDR
  binary and gzip-RDS files natively. Drops the runtime `Rscript`
  dependency for legacy bigset metadata; lets `gtrack_var_get` read
  R-written variables.
- **Genome registry helpers** -- `gdb_list_genomes`, `gdb_genome_info`.
- **`gintervals_neighbors`** -- `intervals_set_out`, `warn_ignored_strand`,
  `mindist1/maxdist1/mindist2/maxdist2`.
- **`gsynth_*` defaults** -- `output_format` defaults to `"misha"` (matching R);
  `"seq"` accepted as a legacy alias.
- **`gintervals_random`** -- accepts `filter=` as an alias for `mask=`.
- **`gintervals_mapply`** -- `enable_gapply_intervals=`, `band=`.
- **`gcis_decay`** -- accepts compound 2D expressions referencing one 2D track.

## Partially covered

- **`gtrack.convert`** (legacy 2D format upgrade). The error message now
  directs the user at R misha's `gtrack.convert` to upgrade obsolete
  `OLD_RECTS1/2` / `OLD_COMPUTED1/2/3` format trackdbs. An in-process
  PyMisha converter is deferred; the legacy formats have not been written
  by any misha version in years.

## Not yet implemented

- **`gvtrack.array.slice`** -- the array-track read API is shipped, but the
  vtrack mechanism for slicing array columns inside an expression requires
  C++ scanner integration. Workaround: call `gtrack_array_extract` with
  `slice=` directly and aggregate in Python.

- **C++ 2D iterator family** (FixedRect, Intervals2D, CartesianGrid,
  TrackRects). PyMisha currently does 2D extraction in Python via
  `_quadtree`; correctness is parity-good but throughput is below R for
  scanner-heavy HiC workflows. Tracked as Group K of the 2026-05-15
  parity roadmap.

- **C++ ports of `gintervals.liftover` / `gtrack.liftover` / `gtrack.import`
  (WIG, BedGraph, BigWig) / `gtrack.import_mappedseq`.** PyMisha runs these
  pure-Python paths today; correctness matches R but the perf gap shows on
  multi-GB inputs. Tracked as Group M of the 2026-05-15 parity roadmap.

## Not planned (R-specific or supplanted)

- **`gcluster.run`** -- R-specific SGE/PBS wrapper. Python users use their
  own job schedulers (snakemake, nextflow, dask, etc.).

- **`gwget`** -- R wget shim. PyMisha's `gintervals_import_genes` downloads
  via Python's stdlib HTTP, so this shim is unnecessary.

- **`gdb.install_gff3_converter` / `gdb.install_gtf_converter`** -- they
  install UCSC's `gff3ToGenePred` / `gtfToGenePred` binaries. PyMisha
  parses GFF/GTF natively (`pymisha/genome/_gtf.py`), so these helpers are
  unneeded.

- **COMPUTED 2D Tracks** -- R has internal `Computer2D` classes for
  on-the-fly Hi-C normalization but no public API to create them. The
  [shaman](https://github.com/tanaylab/shaman) Hi-C tool uses plain 2D
  tracks. With no creation API and no known consumer, COMPUTED 2D will
  not be implemented in PyMisha.

- **R `gtrack.var` ASCII serialize variants** (`A\n`, `B\n`). PyMisha
  reads R's XDR binary and gzip RDS via the native reader; ASCII format
  is rare in practice. Workaround: re-write with
  `serialize(value, con, ascii=FALSE)`.
