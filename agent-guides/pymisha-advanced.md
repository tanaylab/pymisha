# pymisha - advanced topics

Specialized recipes that aren't part of the everyday flow. Read [`pymisha-core.md`](pymisha-core.md) first for concepts and the common workhorse recipes.

## Contents

- [1. Meta-profile around peak centers](#1-meta-profile-around-peak-centers)
- [2. 2D contact pile-ups](#2-2d-contact-pile-ups)
- [3. Insulation, directionality index, domain borders](#3-insulation-directionality-index-domain-borders)
- [4. Sequence and PWM (motif) tracks](#4-sequence-and-pwm-motif-tracks)
- [5. Bulk import and export](#5-bulk-import-and-export)
- [6. Side topics (pointers)](#6-side-topics-pointers)

## 1. Meta-profile around peak centers

**Goal.** For a peak set, get a `(peak × offset)` signal matrix you can heatmap, row-cluster, or average into a smooth meta-profile.

```python
import pandas as pd
import numpy as np

# 1. Anchor: per-peak center as a 1bp interval frame.
center = (peaks["start"] + peaks["end"]) // 2
center_1bp = pm.gintervals(chroms=peaks["chrom"], starts=center, ends=center + 1)

# 2. Distance vtrack -> signed offset at each iterator bin.
pm.gvtrack_create("d_center", src=center_1bp, func="distance.center")

# 3. Expand region symmetrically and extract one or more tracks at fine bins.
region = peaks.assign(start=center - W, end=center + W + 20)
spat = pm.gextract(["d_center", *track_names],
                   intervals=region,
                   iterator=20,
                   colnames=["d_center", *track_names])

# 4. Reshape long -> wide (peak x offset bin) for one track at a time.
mat = (spat
       .assign(bin=lambda d: np.floor(d["d_center"] / 20).astype(int))
       .query("abs(bin) <= @W / 20")
       .pivot_table(index="intervalID",
                    columns="bin",
                    values=track_names[0],
                    aggfunc="mean",
                    fill_value=0)
       .sort_index())
```

`distance.center` is the right flavor here - anchor → anchor.

**Variants.**
- **Multi-track in one pass.** Pass the whole list to `gextract`; pivot once per track from the same long-format frame.
- **Pre-aggregated sum (no offset axis).** If you don't need the offset axis, skip `d_center` and build a window vtrack: `pm.gvtrack_create("v", track, "sum", sshift=-W, eshift=W)`, then `pm.gextract("v", intervals=peaks, iterator=peaks)`.
- **Argmax-recenter first.** When peak centers are coarse, refine via the `groupby + idxmax` recipe in pymisha-core §4.6 before building `center_1bp`.

**Avoid:**
- `(peaks["start"] + peaks["end"]) / 2` - float division yields non-integer centers. Always `// 2`.
- Forgetting to wrap the signal column with `"ifelse(is.na(...), 0, ...)"` inside the extract expression.
- Using `func="distance"` instead of `func="distance.center"` - the default `distance` mixes edge and normalized-in-interval semantics, which is *not* what an anchor-relative offset matrix needs.

## 2. 2D contact pile-ups

**Goal.** From a 2D contact track (Hi-C / capture-HiC), build a feature-pair contact density matrix - e.g. CTCF × CTCF, enhancer × promoter - stratified by genomic distance.

**Build a 2D pair iterator.** `pm.giterator_cartesian_grid(intervals1, expansion1, intervals2, expansion2, min_band_idx, max_band_idx)` produces a 2D iterator over the cartesian product of two 1D intervals sets, each row expanded by `expansionN`.

Self-pair (CTCF × CTCF) restricted to a cis off-diagonal band - the common case. `intervals2=None` makes it a self-pair, which is the form that supports `min_band_idx` / `max_band_idx`:

```python
sites = pm.gintervals_load("intervs.ctcf_peaks")

it = pm.giterator_cartesian_grid(
    intervals1=sites,  expansion1=1000,
    intervals2=None,   expansion2=None,
    min_band_idx=1, max_band_idx=None)         # cis-only, off-diagonal
```

`min_band_idx` / `max_band_idx` restrict to a diagonal band - `1, None` is "off-diagonal cis only", `1, 1` is "near-diagonal", `0, 0` is "diagonal only". **Constraint:** band-index filtering requires `intervals2=None`. For an explicit pair of two different intervals sets (`sites_x` × `sites_y`), pass `intervals2=sites_y` and omit `min_band_idx` / `max_band_idx` - the band filter is unavailable in that mode.

**Aggregate contacts in the 2D window per pair.**

```python
pm.gvtrack_create("obs", src="hic.es.score", func="area")

pair_mat = pm.gextract("obs", intervals=it, iterator=it,
                       band=(-100_000_000, -1000))   # exclude near-diagonal trivial contacts
```

`band=(-1e8, -1e3)` keeps only contacts at least 1kb off-diagonal. The `band=` arg is a 2D-specific feature of `gextract` / `gdist`.

**Per-axis distance binning** (asymmetric feature pile-up). Use the explicit-pair iterator (two different intervals sets, no band filter):

```python
sites_x = pm.gintervals_load("intervs.ctcf_peaks")
sites_y = pm.gintervals_load("intervs.global.tss")
it_pair = pm.giterator_cartesian_grid(
    intervals1=sites_x, expansion1=1000,
    intervals2=sites_y, expansion2=1000)        # no band_idx in explicit-pair mode

pm.gvtrack_create("dx", src=sites_x, func="distance.center")
pm.gvtrack_iterator("dx", dim=1)
pm.gvtrack_create("dy", src=sites_y, func="distance.center")
pm.gvtrack_iterator("dy", dim=2)

h = pm.gdist("dx", dist_breaks,
             "dy", dist_breaks,
             "obs", obs_breaks,
             intervals=pm.gintervals_2d_all(), iterator=it_pair,
             include_lowest=True)
# Collapse the obs-value axis to (dx, dy) mean obs, weighting each obs-bin by
# its midpoint (not its lower edge) and dividing by the total count per cell:
mids = (np.asarray(obs_breaks[:-1]) + np.asarray(obs_breaks[1:])) / 2
mat = np.apply_along_axis(lambda w: np.sum(w * mids) / max(np.sum(w), 1),
                          axis=2, arr=h)
```

**Materialize a 2D track from a contacts text file.**

```python
pm.gtrack_2d_import_contacts("hic.cell_a", "Cell A Hi-C contacts",
                             contacts="/path/to/contacts.txt",
                             fends="/path/to/redb.fends",
                             allow_duplicates=False)
```

For pre-scored rectangle outputs (shaman `.score` files), `contacts=` accepts a list of file paths.

**`gtrack_2d_create` for sparse rectangles from Python.**

```python
rects_with_value = rects.assign(value=rects["obs"])
pm.gtrack_2d_create("hic.derived", "Per-rectangle scores",
                    intervals=rects_with_value, values=rects_with_value["value"])
```

**Avoid:**
- 2D iterator without `min_band_idx` / `max_band_idx` when you only care about cis - the implicit "everything" scope is much slower.
- `gtrack_2d_create` with overlapping rectangles - 2D tracks expect non-overlapping cells.
- Forgetting `band=(-X, -Y)` on cis extracts - the near-diagonal dominates the signal numerically.
- `gtrack_2d_import_contacts(... allow_duplicates=True)` (the default in pymisha) for pooled scHi-C or other workflows where duplicate fend pairs are not real duplicate contacts - inflates counts. Pass `allow_duplicates=False` deliberately. (Note: this default differs from R misha, which defaults `allow.duplicates=FALSE`.)

## 3. Insulation, directionality index, domain borders

**Goal.** From a 2D contact track, build 1D per-bin scores capturing local TAD structure - *insulation* (low across a TAD border, high inside) and *directionality index* (asymmetry between upstream and downstream contact counts) - then call borders from local minima.

**Insulation via paired 2D vtracks.** Define a square 2D window on the diagonal:

```python
W = 100_000
pm.gvtrack_create("obs_ins", src="hic.es.score", func="weighted.sum")
pm.gvtrack_iterator_2d("obs_ins",
                       sshift1=-W, eshift1=W,
                       sshift2=-W, eshift2=W)

ins = pm.gextract("obs_ins",
                  intervals=pm.gintervals_all(),
                  iterator=20_000)            # 20kb diagonal iterator
```

The engine sweeps the diagonal when you pass an integer `iterator` against a 2D track.

**Directionality index** - same shape, two asymmetric windows:

```python
pm.gvtrack_create("obs_up", src="hic.es.score", func="weighted.sum")
pm.gvtrack_iterator_2d("obs_up",
                       sshift1=-W, eshift1=0,
                       sshift2=0,  eshift2=W)
pm.gvtrack_create("obs_dn", src="hic.es.score", func="weighted.sum")
pm.gvtrack_iterator_2d("obs_dn",
                       sshift1=0,  eshift1=W,
                       sshift2=-W, eshift2=0)

di = pm.gextract("(obs_up - obs_dn) / (obs_up + obs_dn + 1)",
                 intervals=pm.gintervals_all(), iterator=20_000,
                 colnames=["di"])
```

**Persist as a track.**

```python
pm.gtrack_create("hic.es.ins_1e5",
                 description="Insulation, 100kb window, 20kb diagonal",
                 expr="obs_ins",
                 iterator=20_000)
```

**Call borders from local-min of insulation.**

```python
pm.gvtrack_create("ins_min",
                  src="hic.es.ins_1e5",
                  func="min",
                  sshift=-W, eshift=W)

borders = pm.gscreen("hic.es.ins_1e5 == ins_min & hic.es.ins_1e5 < threshold",
                     intervals=pm.gintervals_all(),
                     iterator=20_000)
```

Pick `threshold` with `pm.gquantiles("hic.es.ins_1e5", percentiles=0.1, ...)`.

**Multi-scale insulation.** Loop the same recipe over a list of window sizes:

```python
for W in (50_000, 100_000, 200_000, 500_000):
    track_nm = f"hic.es.ins_{W:g}"
    if pm.gtrack_exists(track_nm):
        continue
    pm.gvtrack_create("obs_ins", src="hic.es.score", func="weighted.sum")
    pm.gvtrack_iterator_2d("obs_ins",
                           sshift1=-W, eshift1=W,
                           sshift2=-W, eshift2=W)
    pm.gtrack_create(track_nm, f"Insulation, {W}bp window",
                     expr="obs_ins", iterator=20_000)
```

**Avoid:**
- A single hardcoded `W = 1e5` window - TAD sizes vary; multi-scale is the lab norm.
- Picking `func="sum"` over `"weighted.sum"` - for 2D contact tracks where rectangles vary in width, `weighted.sum` is correct.
- Computing DI without normalizing by `(up + dn + 1)` - raw difference scales with chromosome-arm coverage.
- Calling borders directly from `ins < threshold` without the local-min constraint - every bin in a deep valley qualifies, inflating border counts.

## 4. Sequence and PWM (motif) tracks

**Goal.** Extract genome sequence under intervals, score PSSMs on the fly via vtracks, and persist motif energies as tracks when reuse warrants it.

**Before writing string-level sequence code, check `gseq_*`.** pymisha ships a sequence-manipulation family - `gseq_extract`, `gseq_rev`, `gseq_comp`, `gseq_revcomp` (reverse / complement / both), `gseq_kmer` / `gseq_kmer_dist` (kmer counting / distance), `gseq_pwm` / `gseq_pwm_edits` (direct PWM scoring on character strings), and motif file readers (see "Motif file readers" below). Reach for these before hand-rolling reverse complement / kmer scans / PWM scoring in plain Python.

**Extract sequence.**

```python
seqs = pm.gseq_extract(intervals)
# List of strings, one DNA string per row. Reverse-complements when
# intervals["strand"] == -1; positive strand if `strand` is absent.
```

For 2D intervals frames, returns paired sequences (one per axis).

**Sequence-content vtracks.** Built-in `seq.GC500_bin20`, `seq.CG_500_mean`, etc. For custom windows, build `kmer.frac` / `kmer.count` vtracks. The `kmer` argument is a **single** k-mer string (e.g. `"G"`, `"CG"`, `"GATC"`); to count two or more, build one vtrack per k-mer and combine in the expression:

```python
pm.gvtrack_create("g_frac", None, "kmer.frac", kmer="G", sshift=-250, eshift=250)
pm.gvtrack_create("c_frac", None, "kmer.frac", kmer="C", sshift=-250, eshift=250)
pm.gextract("g_frac + c_frac", intervals=peaks, iterator=peaks, colnames=["gc_w"])

# Palindromic / strand-explicit k-mer: pass strand=1 (or -1) to avoid double-counting.
pm.gvtrack_create("cg_frac", None, "kmer.frac", kmer="CG", strand=1)
```

`src=None` selects the genome sequence as the source. Optional kwargs: `strand` (0 = both, default; 1 / -1 = stranded), `extend` (True by default, scan past iterator bin edges so a k-mer straddling the boundary still gets counted).

For ephemeral analyses, the two-vtrack-plus-expression form above is fine; for tracks you'll reuse, materialize via `gtrack_create` over a kmer vtrack.

**Motif file readers.** pymisha can read PSSMs from HOMER, JASPAR, and MEME motif files - useful when you don't have a separate motif-database package installed. See `pm.motif_import` for the reader entry points. The resulting matrix plugs straight into the `pssm=` kwarg of `gvtrack_create`.

**PWM kwargs - direct, not nested.** Unlike R misha's `params = list(pssm=..., prior=..., bidirect=..., strand=...)`, pymisha takes PWM / kmer parameters as **direct keyword arguments** on `gvtrack_create`: `pssm=`, `prior=`, `bidirect=`, `strand=`, `extend=`, `score_thresh=`, `max_edits=`, `max_indels=`, `direction=`, `score_min=`, `score_max=`, `kmer=`. The `params=` arg on `gvtrack_create` is reserved for the scalar parameter of a few non-PWM funcs (e.g. `quantile` percentile, `neighbor.count` maxdist).

**PWM vtrack semantics.** A PWM vtrack scans the source sequence with sliding windows of width `nrow(pssm)`. Per-window score = log-likelihood under the PSSM relative to a uniform background (modulated by `prior`). Different funcs aggregate the per-window scores differently:

| `func` | What it returns per iterator bin | Aggregation across windows |
|---|---|---|
| `"pwm"`     | log-sum-exp of all window scores in the iterator interval | LSE (soft sum) |
| `"pwm.max"` | the maximum window score in the iterator interval | best-window |
| `"pwm.max.pos"` | the **1-based** position of the best-scoring window's first base, measured in bp from the start of the scanned region. **Sign carries strand** when `bidirect=True`: `+pos` = best window is on the forward strand, `-pos` = reverse strand. With `bidirect=False` the value is always positive. | argmax |
| `"pwm.count"` | number of windows with score ≥ `score.thresh` | count above threshold |
| `"pwm.edit_distance"` | min #edits to raise (`direction="above"`) or disrupt (`direction="below"`) the best-window score across `score.thresh` | search over edit budget |
| `"pwm.edit_distance.lse"` | same but LSE of windows, not the max | LSE-based |
| `"pwm.edit_distance.pos"` / `"pwm.edit_distance.lse.pos"` | 1-based position of an edit in the optimal edit set; exactly "the most impactful single edit" only when one edit suffices. Same sign-by-strand convention as `pwm.max.pos` when `bidirect=True` | argmax over edit positions |
| `"pwm.n_mutations"` | number of single-base substitutions that each *independently* bring a window across `score_thresh`; `0` if already met, `NaN` if no single edit suffices | max across windows |

**Edit-distance knobs.** `direction="above"` (default) finds the minimum edits to *raise* the score across `score_thresh` - "how close is this site to becoming a hit". `direction="below"` finds the minimum edits to *disrupt* an existing hit - "how robust is this site". `max_edits` caps the total budget (exhaustive search for `max_edits <= 2`, greedy heuristic for `max_edits >= 3`). `max_indels` allows insertions / deletions in addition to substitutions. `score_min` / `score_max` clip the per-window score range before the edit search to keep the optimization bounded. With `bidirect=True`, an edit's effect is evaluated against both strands and the better orientation is kept.

The iterator-bin width and the **PSSM width** interact. With `iterator=20` and a 12bp PSSM, every 20bp bin contains 9 candidate windows (offsets 0..8 within the bin). With `iterator=1`, each bin contains a single window.

**`extend`** (bool, default `True`): when `True`, the scan window is extended by `nrow(pssm) - 1` bp past the *end* of the iterator interval, so a motif whose anchor sits just before the boundary still gets scored without being clipped. (No extension on the start side; the engine only walks anchors forward.) Set `False` only if you specifically want to drop boundary-straddling motifs.

**`bidirect` and `strand` - which strand(s) get scanned.**

The two params interact, and which one wins depends on `bidirect`:

| `bidirect` | `strand` | What the engine actually scans |
|---|---|---|
| `True` *(default)* | *(ignored)* | Both strands at every window position; the per-window score is the LSE of forward + reverse-complement matches (for `pwm` / `pwm.count`) or the max of the two (for `pwm.max` / `*.pos`). |
| `False` | `+1` *(default)* | Forward strand only. |
| `False` | `-1` | Reverse-complement strand only. (Positions are remapped back to the original strand's coordinates, so values stay positive and comparable to the forward-strand walk.) |

So `strand` is a no-op while `bidirect=True`. To get strand-resolved hits, you must set `bidirect=False` explicitly. Note also that there is **no `strand=0`** for PWM (that's the kmer convention); the PWM equivalent of "both strands" is `bidirect=True`.

```python
# bidirect=True (default): both strands at every position, keep the better.
# The .pos value carries strand info: +pos = forward hit, -pos = reverse hit.
pm.gvtrack_create("ctcf_max", src=None, func="pwm.max",
                  pssm=ctcf_pssm, prior=0.01, bidirect=True)

# Strand-specific: pair two vtracks, strand=+1 / -1, bidirect=False.
# Each .pos is always positive; you know the strand from which vtrack it came from.
pm.gvtrack_create("ctcf_fwd", src=None, func="pwm.max",
                  pssm=ctcf_pssm, prior=0.01, strand=+1, bidirect=False)
pm.gvtrack_create("ctcf_rev", src=None, func="pwm.max",
                  pssm=ctcf_pssm, prior=0.01, strand=-1, bidirect=False)
```

**Position semantics (`pwm.max.pos`, `pwm.edit_distance.pos`).** All `*.pos` values are **1-based** bp offsets *relative to the iterator interval* (after any `gvtrack_iterator()` `sshift` / `eshift` shifts), pointing at the first base of the best window **in forward-strand orientation** - on either strand and under either `bidirect` setting. `pos = 1` means the best window starts at the very first base of the (possibly shifted) iterator interval; the motif occupies `pos .. pos + nrow(pssm) - 1` in forward coordinates in every case. The sign convention above only applies under `bidirect=True`, where it says *which strand won*, not which end you are pointed at; under `bidirect=False` the value is always positive regardless of `strand`.

Ties break by scan order, which is strand-dependent: `strand=1` (and `bidirect=True`) keeps the most 5' tied anchor and the forward strand wins a tie at the same coordinate, but `bidirect=False, strand=-1` scans a reverse-complemented buffer and so keeps the most 3' tied anchor in forward coordinates.

> **Version caveat.** That uniform convention holds from pymisha 0.12.0. Earlier builds returned a position one less than the 1-based forward start for `strand=-1`, reverse-complemented target coordinates from `pwm.edit_distance.pos`, and a mirrored position from `pwm.edit_distance.lse.pos`. Reading positions off an older build: verify the offset on a planted motif before building spans from it.

`prior` controls the strength of the uniform-background regularizer (smaller = sharper PWM; typical range 0.001-0.05). Tune by AUC on labelled data.

**Materialize a PWM energy track when reuse is warranted.** Preferred path: feed a `pwm` vtrack into `gtrack_create` - composes cleanly with any aggregator and lets you control `iterator`, `prior`, `bidirect`, etc. in one place:

```python
pm.gvtrack_create("ctcf_lse", src=None, func="pwm",
                  pssm=ctcf_pssm, prior=0.01)

pm.gtrack_create("motifs.ctcf",
                 description="CTCF PWM energy (mm10)",
                 expr="ctcf_lse",
                 iterator=20)
```

Standard recipe: **materialize a dense PWM track at 20bp with the `pwm` (LSE) func**, then aggregate over arbitrary regions with `func="lse"` vtracks on top - log-sum-exp composes correctly across bins. There's also a one-shot `pm.gtrack_create_pwm_energy(track, description, pssmset, pssmid, prior, iterator)` for PSSMs stored under `<GROOT>/pssms/`, but the vtrack-then-`gtrack_create` route is more flexible (custom PSSM in `params`, control of `bidirect` / `strand` / `prior`, no `pssms/` directory required) and is the recommended form.

```python
# Re-aggregate a stored LSE track over arbitrary windows:
pm.gvtrack_create("ctcf_region",
                  src="motifs.ctcf",
                  func="lse",
                  sshift=-250, eshift=250)
```

**Background calibration.** `pm.gintervals_random(size, n, chromosomes=..., mask=..., seed=...)` draws `n` fixed-width random intervals - feed to `gseq_extract` for de-novo PSSM training, or score with a PWM vtrack for empirical-quantile calibration.

**Avoid:**
- Aggregating a stored `pwm` (LSE) track with `func="sum"` - LSE doesn't compose under summation. Use `func="lse"`.
- `bidirect=True` when downstream code cares about motif orientation - use paired `strand=+1` / `strand=-1` vtracks.
- Manually reverse-complementing strings from `gseq_extract` to score the opposite strand - the PWM engine handles strand internally.
- Picking `iterator=1` "to be safe" - for most motifs `iterator=10` or `20` with `extend=True` gives the same hits with 10-20× less storage.

## 5. Bulk import and export

**Goal.** Get external data into the pymisha track DB and round-trip tracks back out to standard formats for browser viewing, deposition, sharing.

For the deep version (format chooser across all import paths, pre-import validation for concatenated inputs, post-import sanity protocol, failure-mode lookup) read [`skills/importing-tracks/SKILL.md`](skills/importing-tracks/SKILL.md). The recipes below are the everyday shorthand.

**bigWig / WIG / TSV → dense fixed-bin track.**

```python
pm.gtrack_import("epi.k27me3.es",
                 description="K27me3 ES bigWig (ENCODE ENCSR...)",
                 file="k27me3_es.bw",
                 binsize=20,
                 defval=float("nan"))
```

`gtrack_import` auto-dispatches on extension (`.bw`, `.wig`, `.bed`, `.txt` / `.tsv`).

**BED-style read intervals (Python DataFrame) → pileup track in one call.** `gtrack_create_dense(..., func="coverage")`: bin value = `sum(value_i * overlap_i / binsize)` = average per-base signal in the bin. With `values=np.ones(len(reads))`, this is a ChIP-seq-style pileup in a single call:

```python
import numpy as np

pm.gtrack_create_dense("atac.es",
                       description="ATAC ES pileup",
                       intervals=read_intervals,
                       values=np.ones(len(read_intervals)),
                       binsize=20,
                       defval=0,
                       func="coverage")
```

Preferred path for any read-intervals-in-Python source.

**Mapped reads from a BAM, SAM, or tab-delimited file.**

```python
pm.gtrack_import_mappedseq("atac.es",
                           description="ATAC ES read coverage",
                           file="atac_es.bam",   # BAM auto-detected; SAM / .txt.gz also accepted
                           pileup=200,
                           binsize=20,
                           remove_dups=True)
```

BAM files are auto-detected by bgzip magic bytes and piped through `samtools view` automatically - `samtools` must be on PATH. For SAM or gzipped SAM inputs, pass `cols_order=None` explicitly. For tab-delimited inputs, pass `cols_order=(seq_col, chrom_col, coord_col, strand_col)` (1-based). C++ fast-path since v0.1.95 (3-5x over the legacy Python loop). R-parity: chromosome names must match the DB's chromkey verbatim (no `chr1` → `1` rewrite). `PYMISHA_FORCE_PY_IMPORT_MAPPEDSEQ=1` selects the pure-Python fallback if you need it for debugging.

**Per-fragment 2D contacts → 2D track.**

```python
pm.gtrack_2d_import_contacts("hic.cell_a",
                             description="Cell A Hi-C",
                             contacts="contacts.txt",
                             fends="redb.fends",
                             allow_duplicates=False)
```

**Export a track to bigWig / bedGraph.** For UCSC browser uploads, GEO depositions, sharing with collaborators on non-misha stacks:

```python
pm.gtrack_export_bigwig("epi.k27me3.es",
                        file="k27me3_es.bw",
                        intervals=pm.gintervals_all(),
                        iterator=20)

pm.gtrack_export_bedgraph("epi.k27me3.es",
                          file="k27me3_es.bg",
                          intervals=peaks,                       # subset to a region set
                          iterator=20,
                          name="K27me3 ES (peaks only)")         # track header line
```

**Persist provenance at import.** Use `pm.gtrack_attr_set(track, "source", ...)` / `pm.gtrack_attr_set(track, "date", ...)` post-import so the track survives someone trying to reproduce your analysis a year later.

**Copy a track between databases.** `pm.gtrack_copy(src, dest=None, db=None, overwrite=False)` copies a track from the *active* trackdb to `db` (another trackdb on disk), preserving indexed-format files, attributes, and metadata. `dest=None` keeps the same name in the destination; pass a list of `src` names with `dest` as a namespace prefix to copy many at once. Useful for staging tracks from a personal trackdb to a shared one, or for promoting a derived track from a project-local trackdb (attached via `gdataset_load`) into the main `<GROOT>`. Distinct from `gsetroot` (which only switches which DB is *active*) and `gdataset_load` (which *attaches* a secondary DB read-only).

**Avoid:**
- `gtrack_import` without `binsize` for a continuous-signal source - resulting bin grid may not align with the rest of your trackdb.
- `gtrack_import_mappedseq` without `remove_dups=True` for ChIP / ATAC - PCR duplicates inflate per-bin counts.
- Pre-converting BAM to SAM manually before calling `gtrack_import_mappedseq` - BAM is now auto-detected and piped through `samtools view` internally.
- Per-chromosome `multiprocessing` + `pd.concat` + `gtrack_create_sparse` to build a coverage track from a BAM - `gtrack_create_dense(..., func="coverage")` does it in one call.

## 6. Side topics (pointers)

These have full conventions outside this guide.

**Methylation tracks (WGBS / PBAT / RRBS).** The lab convention is a four-suffix family per sample - `<sample>.cov`, `.meth`, `.unmeth`, `.avg`. For per-region methylation, the rule is `sum(meth) / sum(cov)` from the *count* tracks - **never** `mean(.avg)`, which is biased by low-coverage CpGs and undefined where `cov = 0`. The `.avg` track is for per-CpG views (heatmaps, scatter plots), not region-level aggregation.

The hand-rolled equivalent is one `func="sum"` vtrack per `.meth` and `.cov`, then `pm.gextract(["meth_v / cov_v", "cov_v"], ...)` and an explicit NaN-fill on `.cov`. The full bismark `.cov.gz` → quartet recipe is documented in [`skills/importing-tracks/SKILL.md`](skills/importing-tracks/SKILL.md).

**New genome bootstrap.** Three layers, from low-level to convenience:

- `pm.gdb_create(groot, fasta, genes_file=None, annots_file=None, annots_names=None)` - write directory structure, index chromosomes from a FASTA, optionally seed annotation tracks from files you provide.
- `pm.gdb_install_intervals(groot, source, sets=..., prefix=..., ...)` - install / refresh `intervs.global.*` annotation sets into an existing groot from an upstream source (UCSC, UCSC track hub, NCBI, local files, S3 backend). `sets=` is a whitelist (default `("genes", "rmsk", "cgi", "cytoband")`); `prefix` lets you namespace alternative versions.
- `pm.gdb_build_genome(name, path=..., registry=..., sets=..., ...)` - top-level convenience: builds an assembly end-to-end from a registry entry. Looks up the FASTA + annotation sources via `registry` and runs `gdb_create` + `gdb_install_intervals` in one call.

Registry / discovery helpers: `pm.gdb_list_genomes(registry)` shows what's available; `pm.gdb_genome_info("hg38")` returns the registry entry (FASTA URL, annotation sources, alias rules). The default registry chain points at canonical UCSC / NCBI sources; override with a project-local YAML registry for custom assemblies.

For multi-contig / fragmented assemblies (Zoonomia primates, draft genomes), use chrom-alias matching to map source-track chromosome names onto the new genome's contigs: `target_chroms`, `match_by_length`, `min_coverage` arguments on `gdb_install_intervals` / `gdb_build_genome` control the heuristic. Pair with `pm.gdb_convert_to_indexed()` after build so the resulting trackdb scales.

Parallelize over many species with `multiprocessing.Pool` (set `pm.CONFIG["multitasking"] = False` inside the worker - see §2 in pymisha-core).

**Cross-assembly liftover.** `pm.gintervals_load_chain(file)` + `pm.gintervals_liftover(intervs, chain)`; always follow with `pm.gintervals_force_range()` (lifted intervals can extend past chromosome ends) and `pm.gintervals_canonic()` (lifted intervals may overlap).

**Reentrant genome swap.** When a script must temporarily switch genomes:

```python
prev = pm.gdb_info()["path"]
try:
    pm.gsetroot("/path/to/other_db")
    # ... analysis ...
finally:
    pm.gsetroot(prev)
```

Without the `finally` restore, a partially-completed run leaves the session pointing at the wrong DB and subsequent `gintervals_load` calls silently load against the wrong reference.

**Synthetic genomes via `gsynth`.** Train a k-th-order Markov sequence model on a real genome (optionally stratified by tracks such as GC content / CpG fraction), then sample synthetic sequence with matched per-base composition. Useful for ML background-distribution work.

- `pm.gsynth_train(*dim_specs, mask=None, intervals=None, iterator=None, k=5, prior="marginal", ...)` - fit a k-th-order Markov model. `*dim_specs` takes zero or more stratification spec dicts (each `{"expr": ..., "breaks": ..., "bin_merge": ...}` over an existing vtrack) so the model has per-stratum transition tables. With no specs, fits a single global model. `mask=` is an intervals frame whose positions are skipped (e.g. repeats).
- `pm.gsynth_save(model, path, compress=False)` / `pm.gsynth_load(path)` - canonical persistence (writes a `.gsm` directory, or a zip with `compress=True`).
- `pm.gsynth_score(model, track, description=None, intervals=None, mask=None, resolution=None, ...)` - materialize the model's per-bin log-probability as a dense pymisha track (`track=` is the new track name; `resolution=` defaults to the model's training iterator). Always pass `description=` / `intervals=` by name - they're kw-only.
- `pm.gsynth_cell_merge(model, cell_merge, bin_merge=None)` - merge specified stratification cells *within* a single stratified model (not a multi-model combiner). `cell_merge` is a list of `{"from": [...], "to": [...]}` mappings.
- `pm.gsynth_forbid_kmer(model, pattern, check=True)` - zero out transitions that would produce `pattern` (e.g. `"CG"` for a CpG-depleted background). `pattern` must be at most `k + 1` bases.
- `pm.gsynth_sample(model, output, output_format="misha", intervals=None, n_samples=1, preserve_n=True, seed=None, ...)` - emit synthetic sequence. Length is determined by `intervals` (defaults to the whole genome). `output_format="fasta"` writes FASTA + `.fai` at `output`; `"misha"` writes a `.seq` binary at `output` (it does NOT create a named track in the current trackdb - load it separately if you need one); `"vector"` returns a list of strings.

**Synthetic perturbation genomes (`ggenome_*`).** Build a derived FASTA by editing the source genome at chosen intervals, then optionally bootstrap a new trackdb on top of it for downstream extraction.

- `pm.ggenome_implant(intervals, donor, output, *, create_trackdb=False, trackdb_path=None, overwrite=False)` - replace `intervals` in the source genome with the corresponding donor sequences (one per row). Used for CRE-destroy / motif-shuffle ablations.
- `pm.ggenome_transplant(intervals, source_genome, output, *, target_genome=None, create_trackdb=False, ...)` - splice intervals from one assembly into another at matched positions (cross-species transplants).

Typical workflow: load the source genome, write the derived FASTA via `ggenome_implant`, point `gdb_create` or `gdb_build_genome` at it to make a new trackdb, then run your usual extract / vtrack analysis against the perturbed genome.
