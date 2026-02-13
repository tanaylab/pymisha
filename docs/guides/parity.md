# Parity Notes

PyMisha targets parity with R misha for core 1D and 2D workflows while preserving database interoperability.

## What is implemented

- **`gvtrack_filter`** is fully implemented with broad function coverage, including `avg`, `sum`, `min`, `max`, `stddev`, `quantile`, `nearest`, `exists`, `size`, `first`, `last`, `sample`, position functions (`first.pos.*`, `last.pos.*`, `sample.pos.*`, `max.pos.*`, `min.pos.*`), `distance*`, `neighbor.count`, `kmer`, `masked`, `pwm/pwm.max/pwm.max.pos/pwm.count`, `lse`, and `global.percentile*`.
- **`band`** is supported in: `gextract` (2D), `gsummary`, `gquantiles`, `gdist`, `gintervals_summary`, `gintervals_quantiles`, `gbins_summary`, `gbins_quantiles`, `glookup`, and `gtrack_lookup`.
- **`glookup`** has a streaming C++ backend (`pm_lookup`) with `BinsManager`-based binning, `force_binning`, `include_lowest`, and multitask support. Python fallback is retained only for virtual track expressions, 2D, and band paths.
- **Core 2D APIs** are implemented: `gtrack_2d_create`, `gtrack_2d_import`, `gtrack_2d_import_contacts`, `gintervals_2d_band_intersect`, `gvtrack_iterator_2d`, and 2D extraction in `gextract`.
- **Liftover/chain workflow** is complete: `gintervals_load_chain`, `gintervals_as_chain`, `gintervals_liftover`, `gtrack_liftover`.
- **Sequence synthesis** is complete: `gsynth_train`, `gsynth_sample`, `gsynth_random`, `gsynth_replace_kmer`, `gsynth_bin_map`, `gsynth_save`, `gsynth_load`.

## Remaining gaps

- **Track Arrays:** `gtrack.array.*` and `gvtrack.array.slice` are not implemented.
- **Legacy Conversion:** `gtrack.convert` (for migrating old 2D formats) is not implemented.

## References

Authoritative status and roadmap live in:
- `dev/notes/2026-01-27-pymisha-implementation-plan-v2.md`
- `dev/notes/r-tests-porting-matrix.md`
