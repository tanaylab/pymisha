---
name: importing-tracks
description: Use when importing external data (WIG / bigWig / BED / TSV / BAM / 2D contacts / array matrices) into a pymisha track DB, when picking which `gtrack_*import*` / `gtrack_create_*` / `gtrack_liftover` / `gtrack_copy` variant to call, or when a track import fails opaquely or produces an all-zero / all-NaN track.
---

# Importing tracks into pymisha

## Overview

A "track import" is any operation that materializes a persistent on-disk track from an external source - a single file, a directory or URL of files, a BAM, an in-Python DataFrame, a track in another misha DB, or a track in another genome assembly. There are multiple entry points; picking the wrong one is the most common ingestion problem after silent chrom-name mismatches.

Two failure modes worth pre-empting:

1. **Picking the wrong variant** for the source shape.
2. **Silent partial imports.** WIG / bigWig importers run with `ignore_unknown_chroms = True` - chroms not in `pm.gintervals_all()["chrom"]` are dropped without error. Mismatched naming (`1` vs `chr1`) yields an "all NaN" track, not a failure.

## Precondition: chrom-name compatibility (mandatory)

Before any file-based import, verify the source chroms resolve in the active gdb:

```python
import pymisha as pm

pm.gdb_init("/path/to/trackdb")
set(input_df["chrom"].unique()) - set(pm.gintervals_all()["chrom"])
# MUST be empty for chroms you care about
```

For huge WIG / bigWig inputs you can't load into Python, extract chroms from the source:

```bash
grep -oE 'chrom=[^ ]+' file.wig | sort -u | sed 's/chrom=//'   # WIG
bigWigInfo -chroms file.bw                                     # bigWig
```

Non-empty diff → wrong gdb, naming-convention mismatch, or a gdb missing scaffolds. Fix before importing. Silent partial imports are the #1 cause of "the track is all zeros".

## Format chooser

| Source | Function | Key args / caveats |
|---|---|---|
| In-memory intervals + values (DataFrame) | `gtrack_create_dense(name, desc, intervals, values, binsize, defval, func)` (`func` ∈ `"weighted.mean"` / `"weighted.sum"` / `"coverage"` / `"max"` / `"min"` / `"median"` / `"count"`) or `gtrack_create_sparse(name, desc, intervals, values)` | preferred whenever data is already in Python - no TSV round-trip needed |
| Single WIG / bigWig file | `gtrack_import(name, desc, file, binsize, defval=float("nan"))` | auto-dispatch on `.wig` / `.bw` / `.bigWig`; pass `defval=float("nan")` (not `0`) for continuous signal so uncovered bins read as NaN |
| Single BED file (or 4-col TSV) | same `gtrack_import` - pass `binsize` for a dense track, omit it for a sparse track | path goes through the same C++ importer as R misha |
| Python-computed values via TSV (chrom/start/end/value) | `gtrack_import(name, desc, file, binsize)` | pandas writes ints in decimal by default, no `scipen` equivalent needed; verify with `head` before launching the import on a large file |
| Directory or URL-with-wildcards of WIG / bigWig files | `gtrack_import_set(description, path, binsize, track_prefix=None, defval=float("nan"))` | bulk import; `path` may be a glob (`/data/*.bw`) or an `ftp://` URL; continues on per-file errors and returns a dict of successes / failures |
| ChIP / ATAC pileup from read intervals (DataFrame in Python) | `gtrack_create_dense(name, desc, intervals=reads, values=np.ones(len(reads)), binsize=20, defval=0, func="coverage")` | one-call replacement for the `gtrack_import_mappedseq` route when reads are already in Python |
| Mapped reads (TSV from `samtools view`, or pre-staged file) | `gtrack_import_mappedseq(name, desc, file, pileup=N, binsize=B, cols_order=(9, 11, 13, 14), remove_dups=True)` | for BAM input, pipe through `samtools view` first; the default `remove_dups=True` is correct for ChIP / ATAC / CUT&RUN. **C++ fast-path since v0.1.95** (3-5x over the legacy Python loop) with **R-parity semantics**: chromosome names in the SAM / TSV must exact-match the DB's chromkey (no `chr1` → `1` auto-fix) and only tabs split fields. Gzip is still auto-detected. Set `PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ=1` to fall back to the pure-Python loop |
| 2D Hi-C / capture-C contacts (per-contact adj + fends) | `gtrack_2d_import_contacts(name, desc, contacts, fends="<redb>/<RE>.fends", allow_duplicates=False)` | `contacts=` accepts a list (shaman per-rect scores, scHi-C per-core stagings - see "scHi-C pooling" below); `allow_duplicates=True` for adj files that already count duplicates. **pymisha defaults `allow_duplicates=True` - the safe default for pooled / non-bulk inputs is `False`; pass it explicitly.** |
| Restriction-enzyme fragment + fend prerequisites (any new 4C / Hi-C bootstrap) | `gtrack_import("redb.<RE>_flen", flen_file, 0)` + `gtrack_import("redb.<RE>_gc", gc_file, 0)` + `gtrack_create("redb.<RE>_map", mapab_track, iterator="redb.<RE>_flen")` | per-fragment iterator promotes a per-base mapability track to per-fragment mean. See "RE fragment-track bootstrap" below |
| Pooled multi-cell scHi-C track | `gtrack_2d_import_contacts(pool_name, desc, contacts=[stage_1, ..., stage_N], fends="<redb>/<RE>.fends")` | parallel-stage per-cell `gextract`s to N TSVs, then a single pool call. See "scHi-C pooling" below |
| Per-coordinate 2D matrix (already binned, no fends) | `gtrack_2d_import(name, desc, file)` | lower-level than `_contacts`; takes a coord-resolution TSV directly. Used for 4DN-style 1kb contacts. Skip when you have adj + fends |
| 2D track from a per-rectangle DataFrame | `gtrack_2d_create(name, desc, intervals, values)` | rectangles must be non-overlapping |
| Deep-learning model predictions (Borzoi / IceQream / equivalent) | `gtrack_import(track_name, desc, per_bin_tsv, binsize=res)` where the TSV is the model's per-bin output | lab convention: predicted tracks live under `seq.IQ.<model>.<genome>.<config>_<target>` (e.g. `seq.IQ.pcg.flashzoi.mm10.rf524k_EB4_cnt`). See "Deep-learning bridge" below |
| Per-CpG methylation from bismark `.cov.gz` (per sample) | Read TSV with pandas, build the 4-track quartet (`.cov` / `.meth` / `.unmeth` / `.avg`) via `gtrack_create_sparse`. See "Methylation tracks" below | bismark is 1-based, misha is 0-based half-open - convert. Always CG-validate against the genome before importing |
| 450k / EPIC array methylation matrix (TCGA-style wide DataFrame) | Load wide TSV with pandas, then `gtrack_array_create(name, desc, intervals, values, colnames)` + `gtrack_var_set` for per-sample metadata | one wide TSV: chrom / start / end + N value columns (one per sample); use `gtrack_array_extract` to query later. **Divergence from R:** pymisha has no file-import wrapper (R's `gtrack.array.import`); you load to pandas first |
| Wide feature matrix (non-methylation, e.g. signature scores) | same as above: `gtrack_array_create` from a pandas DataFrame + `gtrack_var_set` for per-feature metadata | one wide DataFrame: chrom / start / end + N value columns |
| Track from another misha DB (same assembly) | `gtrack_copy(src, dest, db="/path/to/other/trackdb", overwrite=False)` | format conversion (per-chrom ↔ indexed) and chrom-order remap handled automatically; list `src` supported |
| Intervals from another assembly (liftover via chain) | `gintervals_liftover(intervs, chain="<srcToTgt>.over.chain")` after `gsetroot(target_assembly)` | lifted intervals may extend past chrom ends and may overlap each other - always follow with `gintervals_force_range()` and (if you need disjoint output) `gintervals_canonic()`. See "Liftover" below |
| Track from another assembly (liftover via chain) | `gtrack_liftover(track, desc, src_track_dir, chain, multi_target_agg="mean")` | requires a chain (`gintervals_load_chain`); aggregation policy controls how multiple source bins folded into one target bin combine - pick deliberately. Much less used in the lab than the intervals-side liftover above |
| Intervals (BED / GFF / GTF / VCF) as an interval set, not a track | `gintervals_import_bed` / `gintervals_import_genes` / `gintervals_import_gff` / `gintervals_import_vcf` | for a sparse-track route instead, build the DataFrame and pass to `gtrack_create_sparse` |

## Pileup tracks: prefer the in-Python path

For ChIP / ATAC / CUT&Tag pileup tracks, `gtrack_create_dense(..., func="coverage")` remains the simplest path when reads are already in a Python DataFrame: with `values=np.ones(len(reads))`, bin value = `sum(overlap_i) / binsize` = average per-base read count - exactly the ChIP-seq pileup definition, in one C++ pass over the DataFrame. Use `gtrack_import_mappedseq` when you start from a SAM / TSV file (now also C++; see the import table) or when you specifically want the `pileup=` read-extension feature.

`gtrack_create_dense` `func` choices: `"weighted.mean"` (default), `"weighted.sum"`, `"coverage"`, `"max"`, `"min"`, `"median"`, `"count"`.

## Methylation tracks: the 4-track quartet convention

Lab-wide convention (high-frequency, cross-user - 95+ files / 23+ projects in the R corpus, same convention in Python): every per-sample methylation dataset materialises as **four sparse tracks** sharing a common stem, not one:

| Track | Value per CpG |
|---|---|
| `<sample>.cov`    | total reads at the CpG (`meth + unmeth`) |
| `<sample>.meth`   | methylated read count |
| `<sample>.unmeth` | unmethylated read count |
| `<sample>.avg`    | methylation level `meth / cov` (often materialised; sometimes recomputed as an expression-derived vtrack at the CpG iterator) |

Downstream tooling (lab helpers, `gpatterns` package, methylation-aware wrappers) walks the four suffixes by convention. Materialising only `.avg` and skipping the count pair breaks coverage-aware aggregations and re-binning.

### Importing from a bismark `.cov.gz` (per sample)

Bismark coverage files are 1-based, six columns: `chrom, start, end, avg, meth_count, unmeth_count` (`start == end`). misha (R and Python) is 0-based half-open. **Always CG-validate against the reference** before importing - bismark coordinates can sit on either strand and ambiguous calls land on non-CG positions:

```python
import pandas as pd
import pymisha as pm

df = (pd.read_csv(filename, sep="\t", compression="infer",
                  names=["chrom", "start", "end", "avg", "meth", "unmeth"])
        .drop(columns="avg")
        .query("chrom in @pm.gintervals_all()['chrom'].tolist()")
        .assign(start=lambda d: d["start"] - 1,                # 1-based -> 0-based
                end  =lambda d: d["start"] + 1))               # half-open

# Anchor to the C of the CpG: if the reference base at start is 'C' keep it; if it's
# 'G' (bismark called the minus strand) shift back by one so start points at the C.
ref = pd.Series([s.upper() for s in pm.gseq_extract(df)])
df.loc[ref == "G", "start"] -= 1
df["end"] = df["start"] + 1

# Validate that the 2bp dinucleotide at the anchored position really is CG;
# drop everything else (strand-ambiguous, soft-masked, sequencing errors).
df["end"] = df["start"] + 2
ref2 = pd.Series([s.upper() for s in pm.gseq_extract(df)])
df = df.loc[ref2 == "CG"].copy()
df["end"] = df["start"] + 1

# Deduplicate (the same CpG can appear twice after anchoring) by summing counts.
df = (df.groupby(["chrom", "start", "end"], as_index=False)
        .agg(meth=("meth", "sum"), unmeth=("unmeth", "sum"))
        .assign(cov=lambda d: d["meth"] + d["unmeth"],
                avg=lambda d: d["meth"] / d["cov"])
        .query("cov > 0"))

# Materialise the quartet
for suffix in ("cov", "meth", "unmeth", "avg"):
    tr = f"{track_stem}.{suffix}"
    pm.gtrack_create_sparse(track=tr,
                            description=f"{description} {suffix}",
                            intervals=df[["chrom", "start", "end"]],
                            values=df[suffix].to_numpy())
```

Three points the convention emphasises and that bite if skipped:

1. **`start = bismark_pos - 1`.** Bismark uses 1-based coordinates; misha is 0-based. Off-by-one silently shifts every CpG one base.
2. **The `gseq_extract` CG validation pair.** First call anchors to the C of the CpG; second call confirms the 2bp dinucleotide really is CG. Drops strand-ambiguous and miscalled rows. Skipping this leaks non-CG noise into the track.
3. **Group-summarise before creating.** The same CpG can appear twice in a bismark file (plus / minus calls both pointing to the same C after anchoring); deduplicate by summing counts.

### From bismark BAM

When the upstream is BAM (not `.cov`), parallelise across samples with `multiprocessing.Pool` (remembering to set `pm.CONFIG["multitasking"] = False` inside each worker). The output is the same `.cov / .meth / .unmeth / .avg` quartet, plus any optional pattern-derived suffixes (`n, n0, n1, nx, nc, ncpg, pat_meth, fid, epipoly`) that extend the convention.

### Population matrices: 450k / EPIC / PBAT cohorts

When you have a per-position × per-sample matrix (TCGA 450k arrays, PBAT cohort summaries, EWAS outputs), don't create N quartets - use the array form. **Divergence from R:** pymisha has no file-import wrapper (R's `gtrack.array.import` takes a TSV directly). Load the wide TSV with pandas first:

```python
# Wide TSV: chrom, start, end, sample1, sample2, ..., sampleN
wide = pd.read_csv("/path/to/wide.tsv", sep="\t")
intervs = wide[["chrom", "start", "end"]]
sample_cols = [c for c in wide.columns if c not in ("chrom", "start", "end")]

pm.gtrack_array_create("meth.tcga_brca",
                       description="TCGA BRCA 450k beta values",
                       intervals=intervs,
                       values=wide[sample_cols].to_numpy(),
                       colnames=sample_cols)
pm.gtrack_var_set("meth.tcga_brca", "donors", donors_df)     # per-sample metadata
```

Query with `pm.gtrack_array_extract(track, slice=names_or_indices, intervals=...)` - pulls a sample-subset slice in one call. Build the wide TSV from N per-sample sparse `.avg` tracks via `pm.gextract([samples], intervals=pm.gintervals_all(), file=tmp.tsv)` + pandas load + `gtrack_array_create`.

## RE fragment-track bootstrap (4C / Hi-C prerequisites)

Before any 2D contact import, the destination DB needs the restriction-enzyme fragment family for the RE used in library prep. Standard layout: `redb.<RE>_{flen, gc, map}` per-fragment 1D tracks plus a `fe<RE>_*` family produced from them.

```python
# Per-fragment tracks (binsize=0 -> sparse, one value per RE fragment).
# flen_file / gc_file are produced by an external perl pipeline
# (e.g. map3c/TG3C/gen_re_frags.pl); not part of pymisha proper.
pm.gtrack_import("redb.DpnII_flen", "DpnII fragment lengths", flen_file, binsize=0)
pm.gtrack_import("redb.DpnII_gc",   "DpnII fragment GC",      gc_file,   binsize=0)

# Per-fragment mapability: aggregate a per-base mapability track using
# the flen track as iterator - each output bin is one RE fragment.
pm.gtrack_create("redb.DpnII_map", "DpnII fragment mapability",
                 expr=mapab_per_base_track,
                 iterator="redb.DpnII_flen")

# Export back to text so re_frags_to_fends.pl can build the fends file
# consumed by gtrack_2d_import_contacts later.
pm.gextract("redb.DpnII_map", intervals=pm.gintervals_all(),
            iterator="redb.DpnII_map", file=fragmap_file)
```

The `<redb>/<RE>.fends` file produced downstream (`GATC.fends` for DpnII, etc.) is the `fends=` argument of every `gtrack_2d_import_contacts` call that follows.

## scHi-C: pool many single-cell tracks into one 2D track

Universal pattern: partition the cell list across cores, each core `gextract`s its cells' contacts into one TSV stage, then a single `gtrack_2d_import_contacts` call with a **list** of stage files pools everything. Avoids both N separate import calls and concatenating gigabytes through Python.

```python
import os, tempfile
from multiprocessing import Pool

tmp_dir = tempfile.mkdtemp()
num_cores = 16
core_assignment = [i % num_cores for i in range(len(cells))]

def stage_core(cur_core):
    pm.CONFIG["multitasking"] = False     # nested-fork safety
    parts = []
    for i, cell in enumerate(cells):
        if core_assignment[i] != cur_core:
            continue
        contacts = pm.gextract(cell, intervals=pm.gintervals_2d_all()).iloc[:, :6]
        parts.append(contacts)
    all_conts = pd.concat(parts, ignore_index=True)
    all_conts["value"] = 1
    out = os.path.join(tmp_dir, str(cur_core))
    all_conts.to_csv(out, sep="\t", index=False)
    return out

with Pool(num_cores) as p:
    stage_files = p.map(stage_core, range(num_cores))

pm.gtrack_2d_import_contacts(pool_track,
                             description="Pooled scHi-C contacts",
                             contacts=stage_files,
                             fends=os.path.join(redb_dir, "GATC.fends"),
                             allow_duplicates=False)
```

Key points: `value=1` per-contact (counts get aggregated by the importer); pandas writes ints in decimal by default - no equivalent of R's `format(..., scientific=FALSE)` needed, but verify on the first chunk if you ever override the float format. `allow_duplicates=True` (pymisha default) is appropriate only when the *same fend pair* should be counted more than once (e.g. raw bulk Hi-C); for pooled scHi-C, pass `allow_duplicates=False` explicitly. (R defaults the other way; pymisha agreed with the older convention. Always be explicit.)

## Deep-learning bridge: model predictions → pymisha tracks

Borzoi / IceQream / similar deep-learning genome models round-trip through pymisha: the training half is `gextract` (often via `intervals_join="intervals"`) to assemble per-bin training data; the inference half writes per-bin predictions back as a track. By lab convention, predicted tracks live under `seq.IQ.<model>.<genome>.<config>_<target>` (e.g. `seq.IQ.pcg.flashzoi.mm10.rf524k_EB4_cnt`).

Training-side extraction:

```python
import numpy as np

borzoi_data = pm.gextract(borzoi_vt, intervals=regs, iterator=32,
                          intervals_join="intervals")
for c in borzoi_vt:
    borzoi_data[c] = borzoi_data[c].fillna(0)
# Per-chromosome norm01(log2(1 + x)):
borzoi_data[borzoi_vt] = (borzoi_data
                          .groupby("chrom")[borzoi_vt]
                          .transform(lambda x: (np.log2(1 + x)
                                                - np.log2(1 + x).min())
                                               / (np.log2(1 + x).max()
                                                  - np.log2(1 + x).min())))
borzoi_data.to_csv("output/data-for-borzoi/borzoi_data.tsv",
                   sep="\t", index=False)
```

Prediction-side import (after the external model run produces per-bin TSVs):

```python
pm.gtrack_import("seq.IQ.pcg.flashzoi.mm10.rf524k_EB4_cnt",
                 description="Flashzoi mm10 rf524k EB4_cnt predictions",
                 file=pred_tsv,
                 binsize=32)         # bin width must match the model's output resolution
assert pm.gtrack_exists("seq.IQ.pcg.flashzoi.mm10.rf524k_EB4_cnt")
```

The DL-prediction track behaves like any other dense track downstream - `gcor`, `gscreen`, peak overlap, vtrack composition all work. The `binsize` must match the model's per-bin resolution exactly (typically 1, 20, 32, or 128); a binsize mismatch silently re-bins predictions onto the wrong grid.

## Liftover: intervals vs tracks

Two related operations. The intervals form dominates by frequency. `gintervals_liftover` has a wide parameter surface - picking the right policy combo is the load-bearing decision.

### Canonical idiom: lift then recenter

```python
pm.gsetroot("/path/to/mm9")   # target assembly
lifted = pm.gintervals_liftover(intervs_mm10, chain="mm10ToMm9.over.chain")

# Recenter to a fixed 300bp window so stretched/shrunk intervals don't confound downstream extracts:
mid = ((lifted["start"] + lifted["end"]) // 2).astype(int)
lifted = lifted.assign(start=mid - 150, end=mid + 150)
lifted = pm.gintervals_force_range(lifted)        # clamp to chrom bounds
# If overlap-free output is required (e.g. for sparse-track creation):
# lifted = pm.gintervals_canonic(lifted)
```

Two non-negotiable post-liftover steps: `gintervals_force_range` (lifted intervals can extend past chrom ends - `gintervals()` then rejects them with `end coordinate exceeds chromosome boundaries`) and, if disjoint output is required, `gintervals_canonic` (lifted intervals may overlap).

Output always carries `intervalID` (index into the input) and `chain_id` (which chain produced the mapping). The `chain_id` is essential when a source interval lifted via multiple chains (duplications): two rows with the same `intervalID` but different `chain_id` are distinct mappings.

### Policy chooser

**`src_overlap_policy` - source position mapped by multiple chains (paralogs / duplications):**

| Value | When to use |
|---|---|
| `"error"` (default) | Strict: fail if input contains source duplications. Right for curated chains and 1:1 lift between close assemblies (mm9↔mm10, hg19↔hg38) |
| `"keep"` | Cross-species / paralog-rich (e.g. cross-primate). One source → many output rows; distinguish by `chain_id` |
| `"discard"` | Drop any source that has duplications. Use when downstream tooling can't tolerate ambiguity |

**`tgt_overlap_policy` - multiple chains converging on overlapping target loci:**

| Value | When to use |
|---|---|
| `"auto"` / `"auto_score"` (default) | Segment target overlaps and pick the highest-scoring chain per segment. Right default for almost everything |
| `"auto_longer"` | Prefer the longer chain per segment. Useful when alignment scores are noisy or uniform across the chain file |
| `"auto_first"` | Prefer the lowest `chain_id` per segment (= original chain-file order). Use when the chain file encodes ranked preference |
| `"keep"` | Leave target overlaps untouched (downstream must handle them) |
| `"discard"` | Drop any chain involved in a target overlap |
| `"agg"` | Segment into disjoint regions and retain all contributors per region. **Pair with `value_col` + `multi_target_agg`** for proper value aggregation |
| `"best_source_cluster"` | Cluster chains by source-side overlap: keep every "true duplication" cluster (chains whose source intervals overlap) but pick only the largest "conflicting alternative" cluster (disjoint source intervals). For paralog-aware lifts where you want duplications but not alternative competing mappings |
| `"best_cluster_union"` / `"best_cluster_sum"` / `"best_cluster_max"` | Cluster-best variants for specialised cases (cluster scoring by union / sum / max). Most users want `"best_source_cluster"` or `"auto_score"` |

**`min_score=N`** - filter out chains with alignment score below N. Useful with public chain files that include low-confidence alignments (e.g. liftOver's chains from older UCSC builds).

### Lifting a value column

When source intervals carry a value (peak score, methylation level, deep-learning prediction, ...) and you need that value in the output, pass `value_col`. Pair with `tgt_overlap_policy="agg"` so target-side conflicts are segmented and contributors aggregated:

```python
pm.gintervals_liftover(intervs_with_score, chain,
                       value_col="score",
                       tgt_overlap_policy="agg",
                       multi_target_agg="mean",      # or sum / max / max.coverage_len / ...
                       na_rm=True,
                       min_n=1)                       # require >=1 non-NA contributor per row
```

`multi_target_agg` choices: `"mean"` (default) / `"median"` / `"sum"` / `"min"` / `"max"` / `"count"` / `"first"` / `"last"` / `"nth"` (set `params=N`) / `"max.coverage_len"` / `"min.coverage_len"` / `"max.coverage_frac"` / `"min.coverage_frac"`. Pick deliberately - the default is `"mean"`, which is wrong for count-like or coverage-weighted values. The `na_rm` and `min_n` knobs are only consulted when `value_col` is set.

### Other flags

- `include_metadata=True` adds a `score` column to the output (from the winning chain per row). Only meaningful with `tgt_overlap_policy="auto"` / `"auto_score"`. Use when you want to threshold or weight lifted intervals by alignment quality post-lift.
- `canonic=True` merges adjacent target intervals that share the same `(intervalID, chain_id)` - collapses chain-gap splits into one row per source / chain. Cheaper than running `gintervals_canonic` after the fact when that's the only canonicalisation you need.

### Pre-loaded chains + hand-built chains

`pm.gintervals_load_chain(file, src_overlap_policy=..., tgt_overlap_policy=..., src_groot=..., min_score=...)` parses a chain file once and stamps the policies as attributes on the returned DataFrame. Reusing it across many `gintervals_liftover` calls is much faster than re-parsing per call. Once policies are baked in this way, `gintervals_liftover` rejects attempts to override them - set them at load time.

`src_groot` is a path to the source-assembly groot; when provided, `gintervals_load_chain` validates that chain source chroms and coordinates exist there. Cheap safety net before a long batch.

`pm.gintervals_as_chain(intervals, src_overlap_policy=..., tgt_overlap_policy=..., min_score=...)` promotes an arbitrary DataFrame with the chain columns (`chrom, start, end, strand, chromsrc, startsrc, endsrc, strandsrc, chain_id, score`) into a chain - for synthesised or hand-edited chains.

### gtrack_liftover

Whole-track form. Shares the `multi_target_agg` enum with `gintervals_liftover` (no `value_col` argument - the track's value IS the value column). Same caveat: the default `"mean"` can hide signal when many source bins fold to one target locus; for coverage / count tracks prefer `"sum"` or one of `max.coverage_*` / `min.coverage_*`.

**2D source tracks (since v0.1.94).** `gtrack_liftover` auto-detects 2D source tracks by quadtree signature and routes them through the dedicated C++ 2D path. **RECTS** and **POINTS** source tracks lift end-to-end in C++ today; no aggregation is performed on the 2D side (matches R). **ARRAYS** source tracks are not yet supported (they wait on the ARRAYS native C++ port). 1D dense / sparse source tracks were ported in v0.1.93 (`pm_liftover_track` C++ orchestrator, ~3.5x end-to-end on a 1M-bin + 10k-chain bench); set `PYMISHA_FORCE_PY_LIFTOVER_TRACK=1` to fall back to the pure-Python orchestrator.

## Validating a multi-file concatenated source before import

If the input is a `cat`-style concatenation of pipeline chunks, validate before launching `gtrack_import` (on large inputs it can take 10+ minutes before erroring):

**Trailing newlines on every chunk.** A missing `\n` glues a value line to the next chunk's `fixedStep` header → invalid WIG. Detect and normalize:

```bash
# detect chunks missing terminal newline
for f in chunk_*.wig; do
  [ "$(tail -c1 "$f" | od -An -c | tr -d ' ')" != '\n' ] && echo "MISSING NL: $f"
done

# concatenate with numeric chunk order + newline-normalization in one pass
ls chunk_*.wig | sort -V | xargs awk 1 > combined.wig
```

`awk 1` re-emits every line with a terminating `\n`, so missing newlines are fixed in-flight; `sort -V` keeps `chunk_0, chunk_1, ..., chunk_10` in numeric order (not the lexical `0, 1, 10, 11, ..., 2` that plain `ls` or shell globbing gives).

**No malformed lines.**

```bash
LC_ALL=C grep -cvE '^(fixedStep|variableStep|track|#|-?[0-9.eE+-]+$|$)' combined.wig   # should print 0
```

**No chrom revisits.** Each known chrom must appear contiguously across the concat - the WIG parser rejects with `not sorted by chromosomes` otherwise. Sanity-check by walking the headers (`grep -nE '^(fixed|variable)Step' | awk ...`).

**No chunk overlaps within a chrom.** For consecutive same-chrom chunks, require `next_start_1based - 1 >= prev_start_1based - 1 + prev_value_count`. Overlaps surface at `gextract` time (not import), with `overlapping intervals` from the WIG `get_data` path.

## Post-import sanity

Sample multiple chromosomes - a single read at `chr1:0-1000` can be legitimately zero in continuous-signal tracks and hides chrom-name mismatches:

```python
import numpy as np

assert pm.gtrack_exists(name)
print(pm.gtrack_info(name))              # check bin.size / dimensions / size.in.bytes vs expectation

# Sample ~1000 random positions on each of the first few chroms in the gdb.
# Driving chrom selection from gintervals_all() keeps the check portable to any genome.
all_chr = pm.gintervals_all()
for i in range(min(5, len(all_chr))):
    c = all_chr["chrom"].iloc[i]
    cend = all_chr["end"].iloc[i]
    starts = np.sort(np.random.randint(0, max(cend - 100, 1),
                                       size=min(1000, max(cend - 100, 1))))
    pts = pm.gintervals(chroms=[c] * len(starts),
                        starts=starts.tolist(),
                        ends=(starts + 100).tolist())
    v = pm.gextract(name, intervals=pts)[name]
    print(f"{c:<12} nz={int((v != 0).sum())}/{len(v)} "
          f"mean={v.mean():.4f} range=[{v.min():.3f}, {v.max():.3f}]")
```

All-zero across every chrom you expect signal on → chrom-name mismatch (revisit the precondition). Non-zero with a sane range → import succeeded.

## Failure modes

| Symptom / error | Cause | Fix |
|---|---|---|
| `Invalid format of WIG file ..., line N` | Malformed line N - often a value glued to a `fixedStep` header from a cat-without-newline concat | Re-concat with `awk 1` (see Validation §1) |
| `not sorted by chromosomes` | Concat order interleaves chroms across files | `sort -V` chunk files before `cat`; verify with a header pass |
| `overlapping intervals` (at `gextract`, not import) | Same-chrom chunks overlap | Validation §4 |
| Import succeeds, `gextract` returns all 0 or all NaN | Chrom names didn't match gdb (silent skip); or `defval=0` with uncovered bases on a continuous-signal source | Precondition; use `defval=float("nan")` for continuous signal |
| `gtrack_import_mappedseq` track gives 5-20× expected counts at peaks | Legacy call site explicitly passed `remove_dups=False` (current default is `True`) | Drop the explicit `False`; rely on the safe default |
| Re-imports under the same name behave inconsistently | Importer doesn't fully overwrite | `gtrack_rm(name, force=True)` before re-import |
| `gtrack_liftover` produces NaN-heavy track | Many source bins mapped to overlapping target loci and `multi_target_agg` discarded most | Try `multi_target_agg="mean"` / `"max.coverage_len"` deliberately; inspect the chain coverage |
| `gtrack_copy(..., db=X)` errors on 2D | Chrom orders differ between source and destination | Reorder the destination DB, or stage via Python (`gextract` + `gtrack_2d_create`) |
| `BAD_CHROM` (paths that don't ignore unknown chroms) | Source has chroms the gdb doesn't know | Extend the gdb (see `gdb_install_intervals` / `gdb_build_genome`), add aliases, or filter input |
| `gtrack_2d_import_contacts` 5-20× over-counting | Pooled / scHi-C input + `allow_duplicates=True` (pymisha default) | Pass `allow_duplicates=False` explicitly |

## Common mistakes

- Round-tripping data already in Python through a temp TSV. `gtrack_create_dense` / `gtrack_create_sparse` take the DataFrame directly - faster and avoids parser-edge cases.
- Using `gtrack_import_mappedseq` for reads already loaded in Python, instead of `gtrack_create_dense(..., func="coverage")`. The latter is the modern one-call path.
- Using `defval=0` for continuous signal (phyloP, normalized ratios). Downstream `.isna()` filters become useless; everything looks like "data".
- Skipping post-import sampling because the function returned without error. The most common silent failure is chrom-name mismatch → all-NaN track.
- Treating `gtrack_copy` as a same-DB-only operation. It supports cross-DB copy via `db="/other/groot"`, including format conversion.
- Assuming `gtrack_2d_import_contacts` defaults `allow_duplicates=False` like R does. **pymisha defaults `True`.** Pass it explicitly for any pooled / non-bulk input.
- Assuming a one-call file-import variant `gtrack_array_import` exists. It does not in pymisha; load to pandas and use `gtrack_array_create`.

## Cross-references

- `agent-guides/pymisha-advanced.md` §5 - short recipe view of the same task. This skill is the full reference.
- `agent-guides/pymisha-core.md` §1.1-1.2 - what a trackdb / track is on disk.
- Implementation: `pymisha/tracks.py` (track creation / import / copy / liftover), `pymisha/liftover.py` (intervals-side lift), `pymisha/_array_track.py` (array tracks).
