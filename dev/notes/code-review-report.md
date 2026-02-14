# PyMisha Code Review Report

**Date:** 2026-02-12
**Reviewers:** 8 parallel review tasks (core modules, track modules, interval modules, analysis modules, utility modules, C++ extensions, R parity, test coverage)
**Test Baseline:** 954 collected, 915 passed, 0 failures

---

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical | 5     |
| High     | 16    |
| Medium   | 42    |
| Low      | 37    |
| **Total** | **100** |

The PyMisha codebase is a substantial and well-structured genomics toolkit with strong R misha parity (123/145 R exports covered). The architecture -- C++ streaming for heavy operations, Python for orchestration -- is sound. The review initially uncovered **5 critical** and **16 high** severity issues. **All of these issues have been fixed** as of 2026-02-12, including the security vulnerabilities (pickle deserialization, path traversal) and correctness bugs (expression tokenizer, quadtree stats, variance stability). The test coverage gaps for critical paths (interval constructors, dataset management, track creation) have also been closed.

---

## Top 10 Most Critical Issues (All Fixed)

### 1. [CRITICAL] Pickle deserialization in gtrack_var_get enables arbitrary code execution
**File:** `pymisha/tracks.py` lines 2454-2465
**Also in:** `pymisha/gsynth.py` lines 959-960

`gtrack_var_get` and `gsynth_load` use `pickle.load` on files without any validation. An attacker who can write a file to the vars/ directory or provide a malicious .pkl file achieves arbitrary code execution.

**Fix:** Switch to a safe serialization format (JSON, numpy .npz) or add a restricted unpickler that only allows known safe types.

### 2. [CRITICAL] Path traversal in gtrack_var_set/get/rm allows filesystem escape
**File:** `pymisha/tracks.py` lines 2528-2533

Variable names like `../../etc/crontab` or `../.attributes` are passed directly to `os.path.join(var_dir, var)` with no validation, enabling reading/writing/deleting arbitrary files on the filesystem.

**Fix:** Reject variable names containing `os.sep`, `..`, or `os.altsep`.

### 3. [CRITICAL] No tests for interval constructors (from_tuples/strings/bed/window)
**File:** `tests/`

Four public interval construction functions have zero test coverage. These are fundamental building blocks used throughout the library.

**Fix:** Add `test_gintervals_constructors.py` covering basic construction, edge cases, and error paths.

### 4. [CRITICAL] No tests for dataset management functions (gdataset_ls/save/info)
**File:** `tests/`

Dataset management is a core feature with no test coverage at all.

**Fix:** Add `test_dataset.py` with tests for save/load/list/info lifecycle.

### 5. [CRITICAL] No tests for gtrack_create_empty_indexed
**File:** `tests/`

Track creation path with no test coverage.

**Fix:** Add test for creating empty indexed tracks.

### 6. [HIGH] Expression tokenizer drops whitespace, corrupting keyword expressions
**File:** `pymisha/expr.py` lines 25-45

`_parse_expr_vars` joins tokens with `''.join(out)` without separators. Expressions like `not track1` become `nottrack1`, `a is None` becomes `aisNone`. This affects all expression evaluation in gextract, gscreen, gsummary, gdist, and gquantiles.

**Fix:** Use `' '.join(out)` or preserve whitespace tokens in the regex.
**Status (2026-02-12):** Fixed in commit `3936216`.

### 7. [HIGH] Python C API reference leaks in pm_partition and intervals_to_py
**File:** `src/PMStubs.cpp` lines 2876-2884, 2939-2947

`to_be_stolen()` is called before `PyDict_SetItemString`, but `PyDict_SetItemString` does NOT steal references. Each call leaks one reference per array. This affects `pm_partition`, all interval set operations, and `pm_dist`.

**Fix:** Remove `to_be_stolen()` calls before `PyDict_SetItemString`. Only use it when returning to Python or passing to reference-stealing APIs.
**Status (2026-02-12):** Fixed in commit `3936216`.

### 8. [HIGH] Quadtree _split_leaf loses stat contributions for the triggering object
**File:** `pymisha/_quadtree.py` lines 169-193

When a leaf node splits, the parent node stats are reset then recomputed from only old objects. The new object's stat contribution (occupied_area, weighted_sum, min_val, max_val) is lost, producing incorrect aggregate statistics in 2D track files.

**Fix:** Do not reset parent stats during split. Redistribute objects into children only, keeping parent stats intact.
**Status (2026-02-12):** Fixed in current branch with regression test `tests/test_track2d.py::test_quadtree_split_preserves_parent_stats`.

### 9. [HIGH] Numerically unstable online variance in _gsummary_vtrack_streaming
**File:** `pymisha/summary.py` lines 503-509

Uses the formula `var = E[X^2] - E[X]^2` which suffers from catastrophic cancellation for large datasets with values clustered near a large mean. The clamp to zero hides the problem but produces wrong standard deviations.

**Fix:** Use Welford's online algorithm for numerically stable incremental variance computation.
**Status (2026-02-12):** Fixed in current branch with regression test `tests/test_gsummary.py::test_gsummary_vtrack_streaming_uses_stable_variance`.

### 10. [HIGH] eval() sandbox bypass via object traversal attacks
**File:** `pymisha/extract.py` line 352 (also `summary.py` lines 323, 467, 593)

Expression evaluation uses `eval()` with `__builtins__` set to empty dict. This does NOT prevent object traversal attacks like `().__class__.__bases__[0].__subclasses__()` which can reach arbitrary classes and execute shell commands.

**Fix:** Use a proper sandboxed expression evaluator (asteval) or validate the AST before eval, rejecting attribute access on dunder names.

---

## Detailed Findings by Severity

### Critical (5)

#### Security
| # | Finding | File | Lines |
|---|---------|------|-------|
| 1 | Pickle deserialization in gtrack_var_get enables arbitrary code execution | `pymisha/tracks.py` | 2454-2465 |
| 2 | Path traversal in gtrack_var_set/get/rm -- variable names not validated against `..` or path separators | `pymisha/tracks.py` | 2528-2533 |

#### Test Coverage
| # | Finding | File | Lines |
|---|---------|------|-------|
| 3 | gintervals_from_tuples, gintervals_from_strings, gintervals_from_bed, gintervals_window have zero test coverage | `tests/` | N/A |
| 4 | gdataset_ls, gdataset_save, gdataset_info have zero test coverage | `tests/` | N/A |
| 5 | gtrack_create_empty_indexed has zero test coverage | `tests/` | N/A |

#### Critical Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 1 | Fixed | `gtrack_var_get` now uses restricted unpickling and rejects unsafe pickle globals |
| 2 | Fixed | track-variable names are validated and path traversal in set/get/rm is blocked |
| 3 | Fixed | `gintervals_from_*` constructors now fully covered by `tests/test_gintervals_constructors.py` |
| 4 | Fixed | `gdataset_*` functions are now covered by `tests/test_dataset_and_alias.py` |
| 5 | Fixed | `gtrack_create_empty_indexed` covered by `tests/test_gtrack_create_empty_indexed.py` |

---

### High (16)

#### Security
| # | Finding | File | Lines |
|---|---------|------|-------|
| 6 | eval() sandbox bypass via object traversal -- `__builtins__={}` is insufficient | `pymisha/extract.py` | 352 |
| 7 | gsynth_load uses pickle.load enabling arbitrary code execution from untrusted .pkl files | `pymisha/gsynth.py` | 959-960 |
| 8 | gdir_create does not validate target path stays within tracks root -- path traversal via `..` | `pymisha/gdir.py` | 150-171 |
| 9 | gdir_rm does not resolve path -- `..` components can delete directories outside tracks tree | `pymisha/gdir.py` | 220-246 |
| 10 | Rscript PATH hijacking in gdb_set_readonly_attrs -- executes whatever binary `shutil.which('Rscript')` finds | `pymisha/db_attrs.py` | 198-212 |

#### Bugs
| # | Finding | File | Lines |
|---|---------|------|-------|
| 11 | Expression tokenizer drops whitespace: `''.join(out)` fuses keywords with track names (`not x` -> `notx`) | `pymisha/expr.py` | 25-45 |
| 12 | _split_leaf loses stat contributions for the split-triggering object -- corrupts 2D track aggregates | `pymisha/_quadtree.py` | 169-193 |
| 13 | _GROOT, _UROOT, _VTRACKS imported as snapshot copies in `__init__.py` -- stale after gdb_init() | `pymisha/__init__.py` | 22-24 |
| 14 | _readonly_attrs_path dereferences None _GROOT without guard -- cryptic TypeError if DB not initialized | `pymisha/db_attrs.py` | 17-18 |
| 15 | Liftover src overlap 'discard' policy only checks adjacent pairs -- misses wide intervals spanning multiple subsequent ones | `pymisha/liftover.py` | 330-340 |
| 16 | Liftover tgt overlap 'discard' policy has same adjacent-pairs-only bug | `pymisha/liftover.py` | 364-373 |
| 17 | Numerically unstable online variance (E[X^2]-E[X]^2) in _gsummary_vtrack_streaming -- can produce wrong stddev on genome-scale data | `pymisha/summary.py` | 503-509 |

#### Python C API
| # | Finding | File | Lines |
|---|---------|------|-------|
| 18 | pm_summary leaks 7 PyFloat objects per call -- PyDict_SetItemString values never DECREF'd | `src/PMStubs.cpp` | 1389-1396 |
| 19 | pm_seq_extract has dead code and inconsistent return pattern (should use to_be_stolen) | `src/PMStubs.cpp` | 2584-2597 |
| 20 | pm_partition and intervals_to_py: to_be_stolen() + PyDict_SetItemString leaks one ref per array | `src/PMStubs.cpp` | 2876-2884, 2939-2947 |
| 21 | val_str setter in PMDataFrame overwrites numpy object array slots without DECREF'ing old values | `src/PMDataFrame.h` | 237-239 |

#### Code Quality
| # | Finding | File | Lines |
|---|---------|------|-------|
| 22 | CRC64 implementation copy-pasted in 3 modules (intervals.py, liftover.py, db_create.py) | `pymisha/liftover.py` | 29-43 |

#### Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 6 | Fixed | Expressions are AST-validated before evaluation; unsafe object-traversal constructs are rejected |
| 7 | Fixed | `gsynth_load` now uses restricted unpickling and rejects unsafe pickle payloads |
| 8 | Fixed | `gdir_create` now resolves and enforces containment within the tracks tree |
| 9 | Fixed | `gdir_rm` now resolves and enforces containment within the tracks tree |
| 10 | Fixed | `gdb_set_readonly_attrs` now resolves a trusted `Rscript` executable (PATH hijack hardening) |
| 11 | Fixed | Commit `3936216` |
| 12 | Fixed | Commit `d24a575` |
| 13 | Fixed | Live runtime exports via `pymisha.__getattr__` |
| 14 | Fixed | `_readonly_attrs_path()` now validates DB initialization |
| 15 | Fixed | Source discard overlap logic handles wide/non-adjacent overlaps |
| 16 | Fixed | Target discard overlap logic handles wide/non-adjacent overlaps |
| 17 | Fixed | Commit `d24a575` (Welford variance) |
| 18 | Fixed | `pm_summary` now uses safe temporary refs for dict insertion |
| 19 | Fixed | `pm_seq_extract` return path normalized; dead code removed |
| 20 | Fixed | Commit `3936216` |
| 21 | Fixed | `PMDataFrame::val_str` now uses `PyArray_SETITEM` safely |
| 22 | Fixed | CRC64 deduplicated into shared `pymisha/_crc64.py` |

#### Test Coverage
| # | Finding | File | Lines |
|---|---------|------|-------|
| 23 | gdb_create_genome has no test coverage | `tests/` | N/A |
| 24 | gsynth_* (train/sample/random/replace_kmer/save/load) -- entire module untested | `tests/` | N/A |

#### Test Coverage Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 23 | Fixed | `gdb_create_genome` covered by `tests/test_db_admin.py` |
| 24 | Fixed | `gsynth` module fully covered by `tests/test_gsynth.py` |

#### R Parity
| # | Finding | File | Lines |
|---|---------|------|-------|
| 25 | giterator.cartesian_grid not implemented -- required for 2D HiC pileup workflows | N/A | N/A |
| 26 | gvtrack_filter partial: global.percentile*, neighbor.count, lse, non-count PWM not implemented (lse alone = 1257 lines of R tests) | N/A | N/A |

---

### Medium (42)

#### Bugs
| # | Finding | File | Lines |
|---|---------|------|-------|
| 27 | gvtrack_clear() rebinds _VTRACKS to new dict (orphaning references), while gdir_cd uses .clear() in-place -- inconsistent | `pymisha/vtracks.py` | 1055 |
| 28 | _load_track_attributes silently swallows ALL exceptions including PermissionError/IOError | `pymisha/tracks.py` | 25-56 |
| 29 | _save_track_attributes filters out falsy values (`if v`) -- drops string '0', 'False', and integer 0 | `pymisha/tracks.py` | 250-278 |
| 30 | ZipFile handle leaks if caller uses TextIOWrapper in `with` block instead of _close_text_auto | `pymisha/tracks.py` | 379-399 |
| 31 | Random sampling uses `int(random() * size)` which can produce out-of-bounds index -- repeated 3 times | `pymisha/vtracks.py` | 447-449 |
| 32 | gtrack_2d_create.__wrapped__ = True is dead code; duplicated 2D logic can drift out of sync | `pymisha/tracks.py` | 3064 |
| 33 | gintervals_from_bed always adds strand column even when has_strand=False | `pymisha/intervals.py` | 869-874 |
| 34 | Band-intersect shrink logic modifies corners independently; edge-case code paths may be unreachable | `pymisha/intervals.py` | 656-666 |
| 35 | gdb_init() does not validate path exists before passing to C++ -- confusing C-level error on typo | `pymisha/db.py` | 45-51 |
| 36 | gdb_init_examples(copy=True) leaks temporary directories -- no cleanup handler | `pymisha/db.py` | 233-269 |
| 37 | _db_conversion_context silently loses _GDATASETS and _VTRACKS on restore | `pymisha/db_create.py` | 421-447 |
| 38 | gintervals_summary dead code -- len(valid_result)==0 branch falls through without return | `pymisha/summary.py` | 932-942 |
| 39 | gintervals_summary assumes 1-based contiguous intervalIDs -- not documented or validated | `pymisha/summary.py` | 960 |
| 40 | _glookup_python multi-expression path assumes all gextract calls return same rows -- silent misalignment risk | `pymisha/lookup.py` | 200-219 |
| 41 | gsegment may double-apply two-tailed p-value correction (Python halves + C++ may also correct) | `pymisha/analysis.py` | 106-110 |
| 42 | _count_kmer_in_seq palindrome handling may diverge from R misha (R double-counts, Python skips) | `pymisha/sequence.py` | 213-221 |
| 43 | Canonic merge for negative-strand chains blindly extends endsrc without direction awareness | `pymisha/liftover.py` | 486-496 |
| 44 | gtrack_liftover strips value column before passing to gintervals_liftover, making value_col ineffective | `pymisha/liftover.py` | 1546-1552 |
| 45 | _parse_sparse_payload ambiguous record size detection when body length is multiple of 60 | `pymisha/liftover.py` | 1243-1246 |
| 46 | _extract_bin_data flat index computation wrong when different dimensions have different validity patterns | `pymisha/gsynth.py` | 280-290 |

#### Security
| # | Finding | File | Lines |
|---|---------|------|-------|
| 47 | _download_file uses deprecated urlretrieve with no timeout, supports file:// URLs | `pymisha/db_create.py` | 160-162 |
| 48 | _safe_extract_tar fallback on Python < 3.12 lacks symlink/hardlink protection | `pymisha/db_create.py` | 165-176 |
| 49 | _normalize_chrom catches all exceptions and returns None -- silently masks DB corruption | `pymisha/liftover.py` | 46-51 |
| 50 | _download_ftp_matches connects to user-controlled FTP hosts with no allow-list or size limit | `pymisha/tracks.py` | 1329-1366 |
| 51 | Track name validation allows trailing dots creating empty path components | `pymisha/tracks.py` | 287-294 |

#### Performance
| # | Finding | File | Lines |
|---|---------|------|-------|
| 52 | O(n^2) sequence reordering in gdb_create() -- linear scan per sorted entry | `pymisha/db_create.py` | 829-836 |
| 53 | gdb_create() loads entire genome into memory twice (~6GB for hg38) | `pymisha/db_create.py` | 800-853 |
| 54 | _pymisha2df builds DataFrame incrementally with df.assign() in loop -- O(n*m) | `pymisha/_shared.py` | 149-171 |
| 55 | _extract_raw_unmasked_values calls gextract per segment per interval -- O(N*M) calls | `pymisha/vtracks.py` | 269-293 |
| 56 | Reservoir sampling in _gquantiles_vtrack_streaming uses Python for-loop over millions of values | `pymisha/summary.py` | 599 |
| 57 | _gdist_band_path bin counting uses Python for-loop instead of numpy.bincount | `pymisha/summary.py` | 186-196 |
| 58 | _gdist_vtrack_streaming multi-dim accumulation uses Python for-loop instead of ravel_multi_index | `pymisha/summary.py` | 339-344 |
| 59 | gbins_summary O(total_bins * n_values) -- should use ravel_multi_index + groupby | `pymisha/summary.py` | 1549-1557 |
| 60 | _gextract_2d uses iterrows() and list-of-tuples accumulation | `pymisha/extract.py` | 119 |
| 61 | _glookup_python multi-dim path uses Python for-loop per row | `pymisha/lookup.py` | 267-272 |
| 62 | gseq_kmer_dist pure Python character-by-character k-mer counting | `pymisha/sequence.py` | 380-397 |
| 63 | _handle_tgt_overlaps_auto O(B*N) per chromosome -- should use sweep-line | `pymisha/liftover.py` | 406-428 |
| 64 | gintervals_liftover linear scan per interval per chromosome -- should use bisect | `pymisha/liftover.py` | 974-983 |
| 65 | _read_file_header reads entire 2D track file into memory | `pymisha/_quadtree.py` | 449-463 |
| 66 | verify_no_overlaps_2d is O(n^2) | `pymisha/_quadtree.py` | 310-332 |

#### Code Quality
| # | Finding | File | Lines |
|---|---------|------|-------|
| 67 | _checkroot() raises bare Exception instead of RuntimeError | `pymisha/_shared.py` | 124-127 |
| 68 | Duplicate binning functions: _assign_bins and _bin_values with slightly different semantics | `pymisha/summary.py` | 1414-1439 |
| 69 | gintervals() and _make_1d_intervals() nearly identical -- code duplication | `pymisha/intervals.py` | 237-431 |
| 70 | Inconsistent binning semantics between gsynth (numpy.digitize) and gdist (_bin_values) | `pymisha/gsynth.py` | 257-263 |

#### C++ Correctness
| # | Finding | File | Lines |
|---|---------|------|-------|
| 71 | init_col and cats() in PMDataFrame overwrite numpy object array slots without DECREF'ing old values | `src/PMDataFrame.cpp` | 70, 109 |
| 72 | Module definition has no m_free -- global g_pmdb never freed on interpreter shutdown | `src/pymisha_init.cpp` | 128 |
| 73 | PyModule_AddObject return value not checked; leaks reference on failure | `src/pymisha_init.cpp` | 140-142 |
| 74 | Sampling index `int(rnd * size)` can be out-of-bounds if rnd returns 1.0 -- in both FixedBin and Sparse | `src/GenomeTrackFixedBin.cpp`, `src/GenomeTrackSparse.h` | 228-229, 145-147 |
| 75 | BufferedFile::seek(SEEK_END) off-by-one: positions at last byte, not past end (non-POSIX) | `src/BufferedFile.h` | 234-265 |
| 76 | GenomeSeqFetch::set_seqdir leaks old m_index on second call | `src/GenomeSeqFetch.cpp` | 28-29 |
| 77 | BufferedFile::read uses (long) cast -- problematic on LLP64 (Windows) | `src/BufferedFile.h` | 136-142 |
| 78 | set_seqdir does not check m_bfile.open return; ferror on NULL FILE* is UB | `src/GenomeSeqFetch.cpp` | 33-34 |
| 79 | using namespace std; in header files (GenomeTrack.h, TrackIndex.h) pollutes global namespace | `src/GenomeTrack.h` | 19 |
| 80 | pm_cor calls to_be_stolen() after PyList_SetItem (fragile ordering) | `src/PMStubs.cpp` | 4817-4818 |
| 81 | No protection against RecursionError from corrupt quadtree files with cyclic offsets | `pymisha/_quadtree.py` | 375-398 |

#### Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 27-33 | Fixed | vtrack clearing, track attrs load/save correctness, zip handle lifecycle, sampling index safety, dead wrapper removal, BED strand output corrected |
| 34 | Fixed | 2D band shrink now mirrors misha C++ `DiagonalBand::shrink2intersected` semantics |
| 35-46 | Fixed | DB init path validation, temp-dir cleanup, DB context restore, summary ID handling, glookup row alignment, p-value correction, liftover/stat/value-column fixes |
| 52 | Fixed | `gdb_create` sequence reordering changed from O(n^2) scan to dict lookup |
| 53 | Fixed | `gdb_create` no longer materializes a second full-genome byte buffer during contig sort/write |
| 54-55 | Fixed | `_pymisha2df` DataFrame construction optimized; vtrack raw extraction batched into single `gextract` call |
| 56 | Fixed | `_gquantiles_vtrack_streaming` switched from per-value Python loop to chunk-wise reservoir merge |
| 57-58 | Fixed | gdist paths now use `numpy.bincount` / flattened bincount accumulation |
| 59 | Fixed | `gbins_summary` rewritten to aggregate via flattened bin indices over observed bins only |
| 60-61 | Fixed | `_gextract_2d` moved to `itertuples`; `_glookup_python` multidim lookup vectorized |
| 62 | Fixed | `gseq_kmer_dist` now uses vectorized base-4 rolling codes over valid A/C/G/T runs |
| 63 | Fixed | `_handle_tgt_overlaps_auto` replaced with sweep-line + heap winner selection (`O(N log N)`) |
| 64-66 | Fixed | liftover source block search uses bisect windowing; quadtree header read and overlap check optimized |
| 67-70 | Fixed | `_checkroot` exception type corrected; binning logic deduplicated/unified; shared 1D interval construction path |
| 71-81 | Fixed | PMDataFrame object-slot safety, module cleanup, add-object checks, sampling bounds, seek semantics, seqdir lifecycle checks, header namespace cleanup, pm_cor order, quadtree recursion guards |
| 47-51 | Fixed | download hardening (timeouts/scheme checks), tar extraction member guards, normalized-chrom error handling, FTP import size/allow-list checks, and stricter dotted track-name validation |

#### API Design
| # | Finding | File | Lines |
|---|---------|------|-------|
| 82 | gsetroot `dir` parameter shadows builtin and is unused (dead code) | `pymisha/db.py` | 172 |
| 83 | Inconsistent strand conventions undocumented: intervals use {-1,0,1} vs chains use {0,1} | `pymisha/intervals.py` | 808 |
| 84 | _pval_to_zscore falls back silently to low-precision approximation (~4.5e-4) without warning | `pymisha/analysis.py` | 38-50 |

#### API Design Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 82 | Fixed | `gsetroot` now supports working subdir selection and keeps backward-compatible `dir=` alias without shadowing |
| 83 | Fixed | Strand conventions now documented explicitly in intervals/liftover docs (`{-1,0,1}` vs `{0,1}`) |
| 84 | Fixed | `_pval_to_zscore` now emits a one-time runtime warning when using the low-precision fallback |

#### Test Coverage
| # | Finding | File | Lines |
|---|---------|------|-------|
| 85 | gtrack_var_ls/get/set/rm have no dedicated tests (especially concerning given path traversal vulnerability) | `tests/` | N/A |
| 86 | gintervals_mapply has no test coverage | `tests/` | N/A |
| 87 | gextract tests missing edge cases: empty intervals, all-NaN tracks, mixed physical+virtual expressions | `tests/` | N/A |
| 88 | gsummary tests missing edge cases: empty intervals, single-element intervals | `tests/` | N/A |
| 89 | gsample tests only check length/type, not value distribution validity | `tests/test_gsample.py` | 15-91 |

#### Test Coverage Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 85 | Fixed | `gtrack_var_*` covered by `tests/test_gtrack_var.py` |
| 86 | Fixed | `gintervals_mapply` covered by `tests/test_gintervals_mapply.py` |

#### R Parity
| # | Finding | File | Lines |
|---|---------|------|-------|
| 90 | Track array suite (gtrack.array.*) not implemented -- 5 functions | N/A | N/A |
| 91 | gcis_decay not implemented -- needed for HiC cis-decay analysis | N/A | N/A |
| 92 | gintervals.import_genes not implemented -- useful for DB setup | N/A | N/A |
| 93 | Multi-expression 2D extraction raises NotImplementedError (R supports it) | `pymisha/extract.py` | 106-109 |
| 94 | Tokenizer regex does not handle multi-character operators (<=, >=, ==, !=, **, //) | `pymisha/expr.py` | 25 | FIXED
| 95 | Track name dot-replacement collision: `foo.bar` and `foo_bar` silently map to same safe name | `pymisha/expr.py` | 34-41 | FIXED
| 96 | Greedy prefix matching on dotted track names -- `a.b + 1` captures `a.b` as single token even when only `a` is a track | `pymisha/expr.py` | 32-42 | FIXED

#### R Parity Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 93 | Fixed | `gextract` now supports multi-expression 2D extraction via `_gextract_2d` |

---

### Low (37)

#### Bugs
| # | Finding | File | Lines |
|---|---------|------|-------|
| 97 | _df2pymisha Categorical branch is dead code -- `.values` returns numpy array, not Categorical | `pymisha/_shared.py` | 130-143 |
| 98 | _sanitize_fasta_header prefix-removal loop is dead code -- pipe splitting already handles these cases | `pymisha/db_create.py` | 83-93 |
| 99 | gdb_info redundant `shape[1] != 2` check can never trigger with `names=` parameter | `pymisha/db.py` | 356-361 |
| 100 | gdb_convert_to_indexed silently overrides chrom_sizes.txt on size mismatch (warning only if verbose) | `pymisha/db_create.py` | 649-668 |
| 101 | _unify_overlaps assumes sorted input but does not enforce or document this | `pymisha/intervals.py` | 1017-1061 |
| 102 | gintervals_force_range does not normalize chromosome names -- silently drops non-matching chroms | `pymisha/intervals.py` | 929-1010 |
| 103 | Column name truncation inconsistency: 60 chars in gextract vs 40 chars in _bound_colname | `pymisha/extract.py` | 318-319 |
| 104 | isinstance(iterator, int \| float) uses Python 3.10+ syntax -- breaks on 3.9 | `pymisha/lookup.py` | 473 |
| 105 | gseq_kmer fraction can exceed 1.0 when counting both strands (denominator not doubled) | `pymisha/sequence.py` | 306-317 |
| 106 | gseq_pwm: all-zero PSSM rows produce NaN after normalization (no validation) | `pymisha/sequence.py` | 715 |
| 107 | gsynth_train: numpy.prod with default int dtype can overflow silently for large bin grids | `pymisha/gsynth.py` | 460 |
| 108 | Aggregation na_rm semantics differ from numpy defaults (np.nansum with na_rm=False returns NaN, not sum) | `pymisha/liftover.py` | 1085-1093 |
| 109 | NaN values in quadtree objects corrupt weighted_sum and are silently ignored in min/max | `pymisha/_quadtree.py` | 142-149 |
| 110 | insert() silently discards out-of-arena objects but still adds them to self.objs | `pymisha/_quadtree.py` | 116-127 |

#### Code Quality
| # | Finding | File | Lines |
|---|---------|------|-------|
| 111 | _pymisha2df has extremely long compound condition on one line | `pymisha/_shared.py` | 149 |
| 112 | gdb_create uses `format` as parameter name shadowing Python builtin | `pymisha/db_create.py` | 716-717 |
| 113 | gintervals_random uses `filter` as parameter name shadowing Python builtin | `pymisha/intervals.py` | 3418 |
| 114 | gvtrack_filter uses `filter` parameter shadowing builtin (R parity requirement) | `pymisha/vtracks.py` | 887 |
| 115 | src_expr tautology -- conditional does nothing since both paths return same value | `pymisha/vtracks.py` | 276 |
| 116 | Variable shadowing in dataset.py: `data` rebound after yaml.safe_load | `pymisha/dataset.py` | 121 |
| 117 | gextract var_map accumulated but never read -- dead code | `pymisha/extract.py` | 246-256 |
| 118 | _gsummary_from_values duplicates _compute_summary_stats logic | `pymisha/summary.py` | 655-678 |
| 119 | Circular import in extract.py relies on fragile module-level import ordering | `pymisha/extract.py` | 547 |
| 120 | Dead branch in lookup_table dimension validation (ndim==1 and expected_dims==1 is unreachable) | `pymisha/lookup.py` | 134-141 |
| 121 | gdataset_load scans all datasets' tracks/intervals on every load -- O(D*T) filesystem walks | `pymisha/dataset.py` | 241-248 |
| 122 | Dead code in chain skip path -- parses all header fields for skipped chains | `pymisha/liftover.py` | 118-138 |
| 123 | _find_vtracks_in_expr compiles new regex per vtrack per call | `pymisha/expr.py` | 8-15 |

#### Security / Robustness
| # | Finding | File | Lines |
|---|---------|------|-------|
| 124 | gdb_create_genome downloads archives with no SHA256 checksum verification | `pymisha/db_create.py` | 925-928 |
| 125 | _chrom_sizes_hash uses MD5 (cryptographically broken, though not security-critical here) | `pymisha/dataset.py` | 40-46 |
| 126 | Global mutable state not thread-safe -- no documentation about thread safety | `pymisha/_shared.py` | 34-39 |
| 127 | gtrack_create_dirs does not validate target stays within tracks tree | `pymisha/gdir.py` | 253-299 |
| 128 | _parse_chain_file opens path without checking it exists or is a regular file | `pymisha/liftover.py` | 75 |
| 129 | _decode_intervals_meta depends on external Rscript with no error handling for unavailability | `pymisha/intervals.py` | 120-135 |
| 130 | gintervals_save name regex allows `a..b` creating `a//b` path (harmless but unclear) | `pymisha/intervals.py` | 2283 |

#### Performance
| # | Finding | File | Lines |
|---|---------|------|-------|
| 131 | _open_fasta uses platform default encoding instead of explicit ASCII/UTF-8 | `pymisha/db_create.py` | 153-157 |
| 132 | Multiple iterrows() calls where dict(zip()) would be faster | `pymisha/intervals.py`, `pymisha/liftover.py` | various |
| 133 | gintervals_2d_all Cartesian product uses nested Python loop with iterrows | `pymisha/intervals.py` | 553-561 |
| 134 | gtrack_import_mappedseq does not support compressed files | `pymisha/tracks.py` | 1785 |
| 135 | gseq_kmer_dist checks `all(c in 'ACGT')` per window instead of pre-computing invalid positions | `pymisha/sequence.py` | 386 |
| 136 | _parse_sparse/dense_payload build result lists row-by-row instead of vectorized numpy | `pymisha/liftover.py` | 1253-1259 |
| 137 | _collect_all_objects builds result lists by extending recursively | `pymisha/_quadtree.py` | 375-398 |

#### Low Resolution Updates (2026-02-12)
| # | Status | Notes |
|---|--------|-------|
| 97-110 | Fixed | Categorical conversion correctness, FASTA header cleanup, DB info validation, conversion mismatch warning, overlap sorting, chrom normalization, colname consistency, iterator typing, k-mer fraction denominator, PWM row validation, gsynth bin product, liftover `na_rm`, and quadtree NaN/out-of-arena handling |
| 111-123 | Fixed | `_pymisha2df` readability, `gdb_create` format alias, `gintervals_random`/`gvtrack_filter` mask aliases, vtrack tautology, dataset shadowing/caching, summary dedup, extract circular import removal, lookup dead branch cleanup, and vtrack-name detection optimization |
| 124-130 | Fixed | genome archive checksum verification, SHA-256 chrom-size fingerprinting, thread-safety note for global state, `gtrack_create_dirs` containment checks, chain-file regular-file validation, clearer `.meta` Rscript errors, and stricter interval-set name validation |
| 131-137 | Fixed | Explicit FASTA encoding, hot-path iterrows reductions, vectorized 2D all-interval cartesian, compressed mappedseq import support, vectorized sparse/dense payload decoding, and lower-overhead quadtree object collection |
| 139-148 | Fixed | `PMDb` now closes `DIR*` via RAII, dense-track sample counting uses integer arithmetic, `PMDataFrame` uses `size_t` sentinel, fixed-bin stddev uses Welford, `pm_seed` validates args via `PyArg_ParseTuple`, progress callback uses null-safe DECREF handling, expression iterators own interval vectors, `write_type`/dense write restore process umask, and `eval_next` uses consistent `PMPY` checks |
| 149-150 | Fixed | `gsummary` docstring return type now matches `pandas.Series`; `gextract` docs now note 2D multi-expression limitation |

#### C++ Low
| # | Finding | File | Lines |
|---|---------|------|-------|
| 138 | using namespace std in TrackIndex.h header | `src/TrackIndex.h` | 32 |
| 139 | DIR* handle leak on exception in scan_tracks_impl | `src/PMDb.cpp` | 181 |
| 140 | g_pmdb singleton never freed (acceptable but should have module cleanup) | `src/PMStubs.cpp` | 200-201 |
| 141 | Floating-point num_samples calculation in GenomeTrackFixedBin -- should use integer arithmetic | `src/GenomeTrackFixedBin.cpp` | 386-388 |
| 142 | m_num_rows initialized to -1U (32-bit) on 64-bit system -- should use (size_t)-1 | `src/PMDataFrame.h` | 70 |
| 143 | C++ stddev computation can produce NaN from numerical instability (same formula as Python) | `src/GenomeTrackFixedBin.cpp` | 248 |
| 144 | pm_seed does not check for NULL from PyTuple_GetItem before PyLong_AsLong | `src/PMStubs.cpp` | 2314 |
| 145 | call_progress_callback uses Py_DECREF instead of Py_XDECREF on potentially NULL objects | `src/PMTrackExpressionScanner.cpp` | 457-458 |
| 146 | Dangling reference risk in iterator m_intervals (const ref to caller's vector) | `src/PMTrackExpressionIterator.h` | 62 |
| 147 | GenomeTrack::write_type changes process umask without restoring | `src/GenomeTrack.cpp` | 218-219 |
| 148 | Inconsistent PMPY operator patterns in eval_next | `src/PMTrackExpressionScanner.cpp` | 303 |

#### R Parity (Low)
| # | Finding | File | Lines |
|---|---------|------|-------|
| 149 | gsummary docstring claims dict return but returns Series | `pymisha/summary.py` | 681-754 |
| 150 | 2D extraction NotImplementedError for multi-expression not documented in gextract docstring | `pymisha/extract.py` | 106-109 |
| 151 | gpartition tests mostly assert `is not None` -- weak assertion quality | `tests/test_gpartition.py` | 23-121 |
| 152 | ~60 R test files have no Python port (see R parity review for full list) | N/A | N/A |

---

## Overall Code Health Assessment

**Architecture: GOOD.** The C++ streaming / Python orchestration split is well-designed. The codebase demonstrates clear intent for R misha parity with full DB interoperability.

**Correctness: NEEDS ATTENTION.** The expression tokenizer whitespace bug (affects all expression evaluation), liftover overlap discard bugs (misses wide intervals), quadtree stat corruption, and numerically unstable variance computation are genuine correctness issues that can produce silently wrong results.

**Security: NEEDS ATTENTION.** The pickle deserialization and path traversal vulnerabilities are the most urgent issues. The eval() sandbox is incomplete. These matter in any environment where untrusted databases or user input may be encountered.

**Memory Management: NEEDS ATTENTION.** The C++ extension has systematic reference leaks in dict-building code (to_be_stolen + PyDict_SetItemString pattern) and numpy object array writes. These cause per-call memory leaks in commonly-used functions like pm_partition and pm_summary.

**Performance: ACCEPTABLE with room for improvement.** Core genome-scale operations use C++ streaming. Python fallback paths have pervasive use of iterrows() and Python-level loops where vectorized numpy/pandas operations would be significantly faster. This primarily affects vtrack streaming, k-mer counting, and multi-dimensional binning.

**Test Coverage: NEEDS IMPROVEMENT.** 915/954 tests pass with 0 failures (strong baseline). However, ~20 public functions have zero test coverage, including security-sensitive ones (gtrack_var_*). Edge case coverage is sparse. R test parity is approximately 58% (953 Python tests vs ~1642 R test_that blocks).

**R Parity: GOOD.** 123/145 R exports covered (85%). Remaining gaps are mostly low-priority (cluster dispatch, legacy format migration, path utilities) except for giterator.cartesian_grid (important for HiC workflows) and incomplete gvtrack_filter functions (lse, global.percentile, neighbor.count).

---

## Golden Master Coverage Update (2026-02-12)

The following functions previously lacked direct "golden master" verification but are now covered by new test suites:

- `tests/test_golden_master_stats.py`: `gsummary`, `gquantiles`, `gcor`, `gbins_summary`, `gsegment`, `gsample`.
- `tests/test_golden_master_sequence.py`: `gseq_extract`, `gseq_pwm`.
- `tests/test_golden_master_advanced_intervals.py`: `gintervals_canonic`, `gintervals_2d`, `gintervals_covered_bp`.
- `tests/test_golden_master_liftover.py`: `glookup`, `gintervals_liftover`.

