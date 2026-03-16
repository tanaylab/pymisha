# PyMisha GPU Acceleration Design

Date: 2026-02-15  
Scope: `pymisha` + `_pymisha` C++ backend  
Reference pattern used: `~/src/MotifBlaze` backend/fallback/build approach

## Executive Summary

This document proposes adding optional GPU acceleration to pymisha for
sequence-heavy workloads. The primary motivation is accelerating PWM/PSSM
scoring and k-mer operations where compute dominates I/O.

**Realistic scope.** After careful analysis, the only operation with a clear
GPU advantage today is `gseq_pwm` in non-spatial mode, which uses pure Python
loops and has no sliding window optimization. The C++ vtrack PWM path already
uses `SlideCache` with `RunningLogSumExp` and `RunningMaxDeque`, achieving
amortized O(stride) per interval for contiguous scans. GPU batch recomputation
loses this sliding window amortization, so GPU only wins for `gseq_pwm` or
non-contiguous vtrack access patterns.

**Recommendation.** Implement Phase 0 (infrastructure) and Phase 1 (`gseq_pwm`
non-spatial, PyTorch) only. Gate all subsequent phases on profiling evidence
that GPU throughput exceeds the existing CPU sliding window baseline.

**Key risk.** PyTorch adds ~2-5 GB to the install footprint. PyTorch dispatch
overhead (~0.5-2 ms per call) may negate gains for small batches. The existing
CPU code is already highly optimized for the contiguous-interval case.

**Cost-benefit assessment.** The honest case for GPU acceleration is narrow:
`gseq_pwm` with many sequences and large windows, or whole-genome PWM scans
with non-contiguous intervals. For typical vtrack workflows (contiguous iterator,
stride 1), the CPU sliding window is already near-optimal and GPU transfer
overhead makes acceleration unlikely to help.

---

## Table of Contents

- [1. Goals](#1-goals)
- [2. Non-Goals](#2-non-goals)
- [3. Current Hotspots and GPU Candidacy](#3-current-hotspots-and-gpu-candidacy)
  - [Existing CPU Optimizations](#existing-cpu-optimizations)
  - [High-value candidates](#high-value-candidates)
  - [Low-value / defer](#low-value--defer)
- [4. Architecture](#4-architecture)
  - [Why PyTorch Over Alternatives](#why-pytorch-over-alternatives)
  - [GPU Memory Budget and OOM Handling](#gpu-memory-budget-and-oom-handling)
  - [Error Handling and Graceful Degradation](#error-handling-and-graceful-degradation)
  - [Version Constraints](#version-constraints)
  - [Proposed components](#proposed-components)
  - [Consolidated Config Reference](#consolidated-config-reference)
- [5. Data Pipeline Design](#5-data-pipeline-design)
  - [5.1 Batch packing](#51-batch-packing)
  - [5.2 Fallback threshold](#52-fallback-threshold)
- [6. PWM/PSSM GPU Design (Phase 1)](#6-pwmpssm-gpu-design-phase-1-pytorch-first)
  - [6.1 Semantic parity requirements](#61-semantic-parity-requirements)
  - [6.2 Initial implementation plan](#62-initial-implementation-plan-torch)
  - [6.3 Integration points](#63-integration-points)
- [7. k-mer GPU Design (Phase 3)](#7-k-mer-gpu-design-phase-3-pytorch-first)
- [8. gsynth_train GPU Design (Phase 4)](#8-gsynth_train-gpu-design-phase-4-pytorch-first)
- [9. Build and Packaging Strategy](#9-build-and-packaging-strategy)
- [10. Multitasking and GPU](#10-multitasking-and-gpu)
- [11. Correctness and Numerical Policy](#11-correctness-and-numerical-policy)
- [12. Performance Benchmarks](#12-performance-benchmarks)
- [13. Phased Delivery Plan](#13-phased-delivery-plan)
  - [Recommended Initial Scope](#recommended-initial-scope)
- [14. Risks and Mitigations](#14-risks-and-mitigations)
- [15. Expected Outcomes](#15-expected-outcomes)
- [16. Concrete Plan for gextract / gscreen GPU Fast Path](#16-concrete-plan-for-gextract--gscreen-gpu-fast-path)
- [17. Multi-GPU Cluster Plan](#17-multi-gpu-cluster-plan)
- [18. API Surface Changes](#18-api-surface-changes)
- [19. Profiling and Observability](#19-profiling-and-observability)
- [20. GPU Test Infrastructure](#20-gpu-test-infrastructure)
- [21. User Migration Guide](#21-user-migration-guide)
- [Glossary](#glossary)

---

## 1. Goals

1. Accelerate sequence-heavy workloads where compute dominates I/O.
2. Keep full API compatibility and CPU fallback behavior.
3. Preserve parity with current semantics (NaN handling, strand, iterator shifts, position conventions).
4. Avoid regressions for CPU-only users and existing wheels.

## 2. Non-Goals

1. Rewriting all track/interval operators for GPU.
2. Forcing CUDA as a hard dependency.
3. Accelerating clearly I/O-bound paths (`gextract`/`gscreen` track reads) before sequence kernels.

## 3. Current Hotspots and GPU Candidacy

### Existing CPU Optimizations

Before evaluating GPU candidacy, it is critical to understand the existing CPU
optimizations that set the baseline:

1. **SlideCache** (non-spatial PWM vtracks): For contiguous intervals with
   stride `s`, the sliding window maintains:
   - `RunningLogSumExp`: O(1) amortized push/pop with max-trick rescaling and
     monotonic deque for current max. Uses `double` accumulation with periodic
     refresh every 50K steps to prevent numerical drift.
   - `RunningMaxDeque`: O(1) amortized sliding-window max via monotonic deque.
   - `hits` deque with running count for `pwm.count` mode.
   - Net cost: O(stride) per interval for contiguous scans. For stride=1,
     each interval costs O(1) amortized work.

2. **SpatSlideCache** (spatial PWM vtracks): For spatially-weighted scoring,
   maintains per-bin ring-buffer delta updates:
   - Per-bin log-sum-exp accumulators (`bin_anchor`, `bin_sum_fwd`, `bin_sum_rc`)
     with dirty-bit lazy recomputation.
   - Per-bin max tracking (`BinMax` structs) for `pwm.max`/`pwm.max.pos`.
   - Per-bin hit counts for `pwm.count`.
   - Delta updates only touch bins affected by the slide, not the full window.

**Implication for GPU candidacy:** The claim of "massive window-level
parallelism" in the table below must be qualified. GPU batch recomputation
loses sliding window amortization entirely. For the common case of contiguous
intervals with stride=1, the CPU processes each interval in O(1) amortized
time. GPU acceleration only wins when:
- Intervals are non-contiguous (no sliding window reuse possible).
- Window sizes are very large (>10K positions) where per-window parallelism
  matters more than amortization.
- The entire genome is processed in a single batch (e.g., `gseq_pwm`).
- The Python loop overhead itself is the bottleneck (as in `gseq_pwm`).

### High-value candidates

| Area | Current path | Why GPU helps | Planned phase |
|---|---|---|---|
| PWM/PSSM scoring for vtracks (`pwm`, `pwm.max`, `pwm.max.pos`, `pwm.count`) | `src/PMVTrack.cpp` -> `PWMScorer`/`DnaPSSM` loops | Massive window-level parallelism, arithmetic-heavy | Phase 2 (profile-gated) |
| `gseq_pwm` (non-spatial only) | `pymisha/sequence.py` pure Python loops | Very high Python overhead, embarrassingly parallel. Note: `spat_factor` raises `NotImplementedError`, so Phase 1 scope is non-spatial only. | Phase 1 (recommended) |
| `gextract`/`gscreen` (restricted fast path, including GPU-native vtracks) | `PMTrackExprScanner` + per-interval CPU eval | Large vector expression + predicate workloads can be offloaded | Phase E0-E6 (future) |
| k-mer counting (`kmer.count`, `kmer.frac`) | `src/KmerCounter.cpp` per-position `memcmp` loops | Sliding-window hashing maps well to GPU threads | Phase 3 (future) |
| `gseq_kmer_dist` | `pymisha/sequence.py` NumPy vectorized rolling hash + bincount (per-seq iteration, but inner loops are vectorized) | Histogram kernel on GPU; marginal benefit for k<=6 (4^6=4096 bins fits in L1 cache) | Phase 3 (future) |
| `gsynth_train` counting phase | `src/PMGsynth.cpp` per-base context counting | Large-scale parallel counting with atomics/reduction | Phase 4 (future) |

### Low-value / defer

| Area | Reason |
|---|---|
| `gextract`/`gscreen` full general path | Mixed track types, sparse/interval iterators, non-native vtracks, arbitrary Python expressions, and 2D/band logic are too branchy for first GPU version |
| `gwilcox`/`gsegment` | Stateful incremental algorithms with branch-heavy updates |
| interval set algebra (`union/intersect/diff`) | Already efficient C++ merge kernels; limited GPU upside |

## 4. Architecture

Use a backend-dispatch model, but make PyTorch the first GPU implementation layer:

1. Keep existing CPU implementations as baseline and fallback.
2. Add optional GPU backend selected at runtime:
   - `auto`: pick GPU only when workload is large enough
   - `cpu`: force existing behavior
   - `gpu`: force GPU or raise clear error if unavailable
3. Implement first GPU paths with PyTorch tensor ops and `torch.distributed` orchestration.
4. Maintain a shared semantic contract between CPU and GPU paths.
5. Introduce custom CUDA kernels only for profiled hotspots where PyTorch is insufficient.

### Why PyTorch Over Alternatives

| Alternative | Install Size | Pros | Cons |
|---|---|---|---|
| **CuPy** | ~300 MB | Lightweight, NumPy-like API | No `torch.compile`, limited ecosystem, no distributed |
| **Numba CUDA** | ~50 MB | Minimal footprint, JIT kernels | Manual memory management, no tensor abstractions |
| **JAX** | ~1-2 GB | Functional transforms, XLA compilation | Less mature on non-Google hardware, different programming model |
| **Raw CUDA/C++** | ~0 (bundled) | Maximum control | High development cost, platform-specific builds |
| **PyTorch** | ~2-5 GB | Rich ecosystem, `torch.compile`, distributed, large community | Heavy dependency, dispatch overhead (~0.5-2 ms per op) |

PyTorch is chosen because:
1. Users in genomics/ML often already have PyTorch installed, reducing effective cost.
2. `torch.compile` can fuse element-wise operations without custom CUDA kernels.
3. `torch.distributed` provides production-grade multi-GPU support.
4. Fallback to CPU tensors enables testing without a GPU.

**Caveats to acknowledge:**
- PyTorch dispatch overhead (~0.5-2 ms) vs typical operation time (1.6-5 ms for
  a single `gseq_pwm` batch) means GPU is only worthwhile for large batches.
- First-call JIT compilation latency can add 2-10 seconds on initial use.
- `torch.compile` interactions with custom ops need careful testing.

### GPU Memory Budget and OOM Handling

1. **Memory budget calculation:** Before launching a batch, query available
   memory with `torch.cuda.mem_get_info()` and reserve 80% of free memory.
   Compute maximum batch size as `free_bytes * 0.8 / per_interval_bytes`.

2. **Adaptive batch sizing:** Start with `gpu_max_batch_bases` from config.
   If a batch fails with `OutOfMemoryError`, halve the batch size and retry.
   After two consecutive OOM failures, fall back to CPU for the remainder.

3. **OOM recovery protocol:**
   ```
   try:
       result = gpu_score_batch(batch)
   except torch.cuda.OutOfMemoryError:
       torch.cuda.empty_cache()
       half = len(batch) // 2
       if half < min_batch:
           warn("GPU OOM, falling back to CPU")
           result = cpu_score_batch(batch)
       else:
           result = concat(gpu_score_batch(batch[:half]),
                           gpu_score_batch(batch[half:]))
   ```

4. **Jupyter/long-session memory management:** Call `torch.cuda.empty_cache()`
   between major operations. Document that users should restart the kernel if
   GPU memory becomes fragmented after many iterations.

5. **Cleanup between batches:** Delete intermediate tensors and call
   `del` explicitly for large buffers. Use `torch.cuda.synchronize()` before
   memory queries to ensure accurate free-memory readings.

### Error Handling and Graceful Degradation

1. **CUDA init failures:** If `torch.cuda.is_available()` returns `False` or
   device initialization raises an exception, silently fall back to CPU with
   a one-time `logging.info` message. No user-facing error in `auto` mode.

2. **Mid-computation errors:** If a GPU kernel fails after partial results,
   discard partial output, log the error, and retry the entire batch on CPU.

3. **Partial failure reporting:** When `gpu_verify` mode detects mismatches,
   log the batch index, interval coordinates, and observed vs expected values
   at `logging.warning` level. Do not raise unless the deviation exceeds the
   hard tolerance (10x the soft tolerance).

4. **Logging and diagnostics:** Expose `pm.gpu_info()` returning a dict with:
   `available`, `device_name`, `driver_version`, `pytorch_version`,
   `cuda_version`, `free_memory_mb`, `total_memory_mb`.

### Version Constraints

| Component | Minimum | Tested | Notes |
|---|---|---|---|
| PyTorch | >= 2.0 | 2.2, 2.3, 2.4 | Required for `torch.compile` support |
| CUDA Toolkit | 11.8 | 11.8, 12.1, 12.4 | Must match PyTorch build |
| NVIDIA Driver | >= 525.60 | 535.x, 545.x | For CUDA 12.x support |
| Python | 3.10-3.12 | 3.10, 3.11, 3.12 | Matches existing wheel matrix |

**Version compatibility strategy:** PyTorch is an optional runtime dependency,
not a build dependency. The GPU modules import torch lazily at first use. If
the installed PyTorch version is below 2.0, raise `ImportError` with a clear
message specifying the minimum version required.

### Proposed Components

1. Python GPU runtime:
   - `pymisha/gpu_runtime.py`: device discovery, dtype policy, stream helpers.
   - `pymisha/gpu_dist.py`: rank/world setup (`torchrun` env + safe defaults).
2. Python GPU compute modules:
   - `pymisha/gpu_pwm.py`: `pwm`, `pwm.max`, `pwm.max.pos`, `pwm.count` via torch ops.
   - `pymisha/gpu_kmer.py`: k-mer count/fraction/distribution via torch ops.
   - `pymisha/gpu_extract.py`: restricted `gextract`/`gscreen` fast path on torch tensors.
3. Existing C++ `_pymisha` remains primary CPU engine and fallback path.
4. Optional later C++/CUDA module (deferred):
   - only for proven hotspots after profiling.
5. Config extensions in `pymisha/_shared.py`:
   - `compute_backend`: `"auto" | "cpu" | "gpu"`
   - `gpu_device`: int
   - `gpu_min_windows`: int
   - `gpu_max_batch_bases`: int
   - `gpu_dist_mode`: `"off" | "torchrun_shard" | "spawn_shard"`
   - `gpu_verify`: bool (dual-run spot checks)

### Consolidated Config Reference

All GPU-related configuration keys from Sections 4, 16, and 17, unified in one table:

| Key | Type | Default | Valid Range | Description | Phase |
|---|---|---|---|---|---|
| `compute_backend` | str | `"auto"` | `"auto"`, `"cpu"`, `"gpu"` | Global backend selection | P0 |
| `gpu_device` | int | `0` | 0-7 | CUDA device index | P0 |
| `gpu_min_windows` | int | `100000` | >= 1 | Min scored windows to trigger GPU | P0 |
| `gpu_max_batch_bases` | int | `50000000` | >= 1000 | Max bases per GPU batch | P0 |
| `gpu_verify` | bool | `False` | -- | Dual-run CPU/GPU spot checks | P0 |
| `extract_backend` | str | `"auto"` | `"auto"`, `"cpu"`, `"gpu"` | Backend for gextract/gscreen | E0 |
| `gpu_extract_min_work` | int | `500000` | >= 1 | rows * complexity threshold | E0 |
| `gpu_extract_max_batch_rows` | int | `100000` | >= 100 | Max rows per extract batch | E1 |
| `gpu_extract_verify` | bool | `False` | -- | Dual-run for extract path | E1 |
| `gpu_extract_allow_vtracks` | bool | `True` | -- | Allow GPU-native vtracks | E2 |
| `gpu_vtrack_seq_min_windows` | int | `50000` | >= 1 | Min windows for seq vtrack GPU | E2 |
| `gpu_vtrack_track_min_bins` | int | `10000` | >= 1 | Min bins for track vtrack GPU | E4 |
| `gpu_dist_mode` | str | `"off"` | `"off"`, `"torchrun_shard"`, `"spawn_shard"` | Multi-GPU distribution mode | D0 |
| `gpu_dist_launcher` | str | `"torchrun"` | `"torchrun"`, `"spawn"` | Launcher for spawn mode | D0 |
| `gpu_dist_merge_format` | str | `"parquet"` | `"parquet"` | Rank-local output format | D1 |
| `gpu_dist_timeout_sec` | int | `300` | >= 10 | Distributed barrier timeout | D0 |
| `gpu_nccl_profile` | str | `"auto"` | `"auto"`, `"l40s"`, `"blackwell"`, `"custom"` | NCCL transport tuning profile | D0 |

## 5. Data Pipeline Design

### Pipeline Flow

```
Intervals (DataFrame)
    |
    v
[1. Build eval intervals]  (CPU: sshift/eshift, strand, chrom bounds)
    |
    v
[2. Fetch sequences]       (CPU: batch by chrom, group by length)
    |
    v
[3. Pack buffers]           (CPU: seq_data, offsets, lengths, meta -> pinned memory)
    |
    v
[4. H2D transfer]          (async copy to GPU via CUDA stream)
    |
    v
[5. GPU compute]           (PWM scoring / k-mer hashing / expression eval)
    |
    v
[6. D2H transfer]          (one value per interval + optional position/strand)
    |
    v
[7. Collect results]       (CPU: assemble into DataFrame / ndarray)
```

### 5.1 Batch packing

For sequence-scoring kernels:

1. Build eval intervals exactly as today (including `sshift`/`eshift`, strand, bounds).
2. Fetch sequences on CPU in batches (group by chromosome and similar length).
3. Pack into contiguous buffers:
   - `seq_data`: flattened `uint8` encoded bases
   - `seq_offsets`: start offset per interval
   - `seq_lengths`
   - `interval_meta`: strand, ROI info, extension policy
4. Use pinned host memory and async H2D copies.
5. Execute compute on torch CUDA streams.
6. Return one value per interval (plus position/strand payload where needed).

**Dtype policy:**

| Buffer | Dtype | Rationale |
|---|---|---|
| `seq_data` | `uint8` | 2-bit encoding (A=0, C=1, G=2, T=3), packed as bytes |
| `seq_offsets`, `seq_lengths` | `int64` | Supports genome-scale coordinates |
| PWM log-probabilities (PSSM matrix) | `float32` | C++ `DnaPSSM` uses `float` for per-position scoring |
| LSE accumulation | `float64` | C++ `RunningLogSumExp` uses `double` for `sum_scaled` and `M`; match for parity |
| `pwm` (LSE) output | `float64` | Final `M + log(sum_scaled)` must be double to match C++ |
| `pwm.max` output | `float32` | Single max score, float32 sufficient |
| `pwm.max.pos` output | `int32` | Position index within interval |
| `pwm.count` output | `int32` | Integer count of threshold exceedances |
| k-mer codes | `int32` | Base-4 encoding, max 4^10 = 1M fits int32 |
| k-mer counts | `int64` | Genome-wide counts can exceed 2^31 |

### 5.2 Fallback threshold

GPU only when both are true:

1. Device available and compatible.
2. Workload exceeds threshold (`total_windows >= gpu_min_windows`).

Small workloads stay on CPU to avoid transfer overhead.

## 6. PWM/PSSM GPU Design (Phase 1, PyTorch-first)

### 6.1 Semantic parity requirements

Must match current behavior from `PWMScorer`/`DnaPSSM`:

1. modes: `pwm` (LSE), `pwm.max`, `pwm.max.pos`, `pwm.count`
2. bidirectional/strand semantics
3. `prior`, `spat_factor`, `spat_bin`, `spat_min/max`
4. neutral chars and invalid-base behavior (see details below)
5. signed position conventions for bidirectional `max.pos`
6. tie-breaking rules for equal max scores:
   - use strict `>` comparisons (never `>=`) so equal values do not replace an existing best.
   - evaluate forward then reverse at each position; equal forward/reverse scores keep forward.
   - for equal best scores across positions, keep the earliest encountered position.

**Neutral character handling per method:**

| Method | C++ behavior | GPU must match |
|---|---|---|
| `DnaPSSM::calc_like` | Returns `-inf` for any window containing a neutral/invalid base | Output `-inf` (torch: `-float('inf')`) |
| `integrate_like` (LSE mode) | Uses `avg_log_prob` (average of valid-position log probs) for neutral positions | Replace neutral positions with `avg_log_prob` before LSE reduction |
| Python `gseq_pwm` | Three policies: `"average"` (use avg log prob), `"log_quarter"` (use log(0.25)), `"na"` (skip windows with neutrals) | Implement all three policies; default `"average"` |

**LSE algorithm difference:**

The C++ `RunningLogSumExp` uses iterative max-trick: maintain running `M` (max)
and `sum_scaled = sum(exp(x_i - M))`, with incremental rescaling when `M`
changes. The GPU implementation should use `torch.logsumexp` which internally
uses the max-subtract trick (`M + log(sum(exp(x - M)))`) but computes it in
one pass over the full window rather than incrementally. Both approaches are
numerically stable, but may produce slightly different results due to
floating-point ordering.

**Tolerance specification:**
- `pwm` (LSE): `atol=1e-4`, `rtol=1e-3`
- `pwm.max`: `atol=1e-6`, `rtol=0`
- `pwm.max.pos` and `pwm.count`: exact integer equality
- These tolerances apply to `gpu_verify` mode comparisons and are the same as Section 11.

### 6.2 Initial implementation plan (torch)

0. **Profile CPU sliding window baseline first.** Before writing any GPU code,
   measure the existing C++ `PWMScorer` with `SlideCache` on representative
   workloads. Record scored-positions/sec for contiguous and non-contiguous
   interval patterns. This establishes the bar that GPU must beat.

1. Batch intervals by similar sequence lengths and pack to torch tensors.
2. Compute forward/reverse PWM scores with torch tensor ops.
3. Use torch reductions for:
   - `lse`: stable log-sum-exp
   - `max`: argmax
   - `count`: threshold predicate sum
4. Return score/position/strand outputs per interval.
5. If this path is too slow on profiling, replace only hot reductions with custom CUDA kernels.

**Crossover analysis:**

| Metric | CPU (SlideCache, stride=1) | CPU (`gseq_pwm` Python) | GPU (estimated) |
|---|---|---|---|
| Throughput | ~1M scored-positions/sec | ~50K scored-positions/sec | ~100M scored-positions/sec |
| Fixed overhead | ~0 | ~0 | 1-5 ms (H2D + kernel launch + D2H) |
| Break-even | N/A (always on) | N/A | ~100K-1M total positions per batch |

For `gseq_pwm` (pure Python loops), GPU wins at modest batch sizes (~1K
sequences of length 1K). For C++ vtrack PWM with `SlideCache` on contiguous
intervals, GPU only wins at very large batch sizes (~1M+ positions) or when
intervals are non-contiguous. The `auto` mode threshold (`gpu_min_windows`)
should be calibrated from this crossover analysis.

### 6.3 Integration points

1. `gseq_pwm`: route to torch batch implementation first; retain Python path as fallback for debugging.
2. `vtracks.py`: defer sequence-based torch dispatch to Phase 2 (after Phase 1 profiling gate).

## 7. k-mer GPU Design (Phase 3, PyTorch-first)

### 7.1 `kmer.count` / `kmer.frac`

1. Encode k-mer to base-4 integer.
2. Use torch rolling/hash-like tensor operations over windows.
3. Compare hash to target (and reverse-complement hash when needed).
4. Reduce to count; derive fraction on device or host.

### 7.2 `gseq_kmer_dist`

**Current implementation note:** `gseq_kmer_dist` in `pymisha/sequence.py`
already uses NumPy vectorized operations: `numpy.frombuffer` for encoding,
lookup-table base conversion, `numpy.diff`-based run detection for valid
regions, vectorized rolling hash via `numpy` array slicing, and `numpy`
accumulation into the count array. The per-sequence loop is a thin
orchestration layer; the inner computation is vectorized. GPU benefit is
marginal for k<=6 (4^6 = 4096 histogram bins fit in L1 cache).

GPU path (if justified by profiling):

1. For each sequence segment, compute rolling k-mer codes on `torch` tensors.
2. Build histogram with `torch` ops and reduce across batches.
3. Transfer final 4^k vector to host and convert to DataFrame.
4. Consider custom CUDA histogram only if `torch` histogram proves to be the bottleneck.

## 8. `gsynth_train` GPU Design (Phase 4, PyTorch-first)

Current `pm_gsynth_train` loop does:

1. mask checks
2. N checks
3. 5-mer context + next-base encoding
4. forward + reverse-complement count increments

Initial GPU version:

1. Precompute per-position bin index + mask flags on CPU (or simple GPU prepass).
2. Torch updates `counts[bin][context][base]` via batched indexing/scatter-add.
3. Use chunked accumulation to reduce contention and memory pressure.
4. Keep normalization/CDF generation on CPU first; optional GPU reduction later.

Sampling (`pm_gsynth_sample`) is less attractive initially due to sequential dependency per generated base.

> **Note:** This section is speculative and deferred. The count table
> (4^5 contexts * 4 bases = 4096 entries) fits entirely in L1 cache (~32-64 KB),
> so the CPU counting loop is already memory-efficient. GPU acceleration for
> this operation should only be pursued if profiling shows it as a bottleneck
> in real workflows, which is unlikely given the small table size.

## 9. Build and Packaging Strategy

### 9.1 Build system

Current build is setuptools + `src/*.cpp`. Keep this unchanged for Phase 1.

1. Phase 1 (preferred): PyTorch-only GPU path in Python; no custom CUDA build requirement.
2. Phase 2 (optional): add custom CUDA extension behind build flag only if profiling justifies it.
3. If Phase 2 is needed, migrate extension build to CMake/scikit-build-core for cleaner optional components.

### 9.2 Distribution

1. CPU wheels remain default.
2. GPU path requires PyTorch at runtime (documented optional dependency).
3. Runtime reports clear capability status (`pm.gpu_info()`).
4. Custom CUDA wheels are explicitly deferred.

## 10. Multitasking and GPU

> **WARNING: fork-CUDA interaction is a critical correctness hazard.**
> The C++ backend uses `fork()` for 10+ operations (`gextract`, `gscreen`,
> `gwilcox`, etc. with `num_kids > 0`). CUDA contexts are NOT fork-safe:
> forking a process after CUDA initialization causes undefined behavior
> (hangs, silent corruption, segfaults). This is a fundamental constraint,
> not a bug to fix.

Current multitasking uses forked processes and FIFO merge. For GPU mode:

1. Disable post-init `fork` by default (`num_kids=0`) for GPU mode.
2. Single GPU: single process + torch CUDA streams.
3. Multi-GPU: explicit `torchrun` (or Python `spawn`) rank-per-device launch; no in-process forking.
4. Synchronization via `torch.distributed` primitives (`barrier`, optional gather), not custom marker files.
5. Merge outputs by `intervalID` order to preserve exact API semantics.

**Fork-CUDA mitigations:**

1. **Guard at dispatch time:** Before routing to GPU path, check
   `torch.cuda.is_initialized()`. If CUDA is already initialized and
   `num_kids > 0`, raise `RuntimeError` with a clear message explaining the
   incompatibility.
2. **Enforce mutual exclusivity:** When `compute_backend="gpu"`, automatically
   set `num_kids=0` and log a warning if the user had configured `num_kids > 0`.
3. **Detect pre-fork CUDA init:** In the fork path (`_pymisha` C++ side),
   add a Python callback that checks `torch.cuda.is_initialized()` before
   `fork()`. If True, abort with a diagnostic message instead of risking
   undefined behavior.
4. **Documentation:** Prominently document that GPU mode and `num_kids > 0`
   are mutually exclusive.

## 11. Correctness and Numerical Policy

1. CPU is source of truth.
2. Add `gpu_verify` mode:
   - sample batches run on both backends
   - assert tolerance per function/mode (see table below)
3. Deterministic reductions where possible; document tolerated nondeterminism for large LSE reductions.

**Concrete tolerance values:**

| Function/Mode | atol | rtol | Notes |
|---|---|---|---|
| `pwm` (LSE) | 1e-4 | 1e-3 | float32 scoring + float64 LSE accumulation recommended |
| `pwm.max` | 1e-6 | 0 | Single max score comparison |
| `pwm.max.pos` | 0 (exact) | 0 | Integer position, must match exactly |
| `pwm.count` | 0 (exact) | 0 | Integer count, must match exactly |
| `kmer.count` | 0 (exact) | 0 | Integer count |
| `kmer.frac` | 1e-10 | 0 | Derived from exact integer counts |

**Recommendation:** Use `float32` for per-position PSSM scoring (matching C++
`DnaPSSM` which uses `float`) but `float64` for LSE accumulation (matching C++
`RunningLogSumExp` which uses `double` for `M` and `sum_scaled`). This
combination provides the best trade-off between GPU throughput and numerical
parity with the CPU path.

4. Reuse existing tests; add GPU-marked parity suites:
   - `tests/test_gseq_pwm_gpu.py`
   - `tests/test_vtrack_pwm_gpu.py`
   - `tests/test_gseq_kmer_gpu.py`
   - `tests/test_gsynth_train_gpu.py`

## 12. Performance Benchmarks

### Baseline CPU measurements (to be collected before GPU work)

These baselines must be measured before GPU implementation begins, using
representative workloads on the target hardware:

| Operation | Current Implementation | Expected Baseline | Notes |
|---|---|---|---|
| `gseq_pwm` (Python loop) | `pymisha/sequence.py` | ~50K scored-positions/sec | Pure Python, no sliding window |
| vtrack PWM with SlideCache | `PWMScorer` C++ | ~1M scored-positions/sec | Amortized O(stride), stride=1 |
| vtrack PWM with SpatSlideCache | `PWMScorer` C++ | ~500K scored-positions/sec | Per-bin delta updates |
| `gseq_kmer_dist` (NumPy) | `pymisha/sequence.py` | ~10M bases/sec | Vectorized rolling hash + bincount |

**Estimated GPU throughput (to validate):**

| Operation | Estimated GPU | Estimated speedup vs CPU | Break-even batch |
|---|---|---|---|
| `gseq_pwm` batch | ~100M scored-positions/sec | ~2000x vs Python, ~100x vs hypothetical C++ | ~10K positions |
| PWM batch (non-contiguous) | ~100M scored-positions/sec | ~100x vs recompute-per-interval C++ | ~100K positions |
| k-mer histogram | ~500M bases/sec | ~50x vs NumPy | ~1M bases |

Add dedicated benchmarks with CPU/GPU crossover curves:

1. PWM: windows/sec by motif length and interval count.
2. k-mer count/dist: throughput by `k`, sequence length, batch size.
3. gsynth_train: bases/sec by bins and mask density.
4. End-to-end: `gextract` on PWM vtracks over genome-scale intervals.

Report:

1. speedup vs current CPU implementation
2. transfer vs compute breakdown
3. memory footprint and occupancy

## 13. Phased Delivery Plan

### Unified Timeline

| Phase | Scope | Duration | Dependencies | Status |
|---|---|---|---|---|
| **Phase 0** | Backend config, capability detection, dispatch scaffolding | 1-2 weeks | None | **Recommended** |
| **Phase 1** | `gseq_pwm` non-spatial GPU path (PyTorch) | 2-4 weeks | Phase 0 | **Recommended** |
| Phase 2 | vtrack PWM integration in `vtracks.py` | 1-2 weeks | Phase 1 | Future: gate on profiling |
| Phase 3 | k-mer acceleration (`kmer.count`, `gseq_kmer_dist`) | 2-3 weeks | Phase 0 | Future: gate on profiling |
| Phase 4 | `gsynth_train` counting phase | 2-4 weeks | Phase 0 | Future: gate on profiling |
| Phase 5 | Custom CUDA kernels (only if needed) | TBD | Phase 1-4 profiling | Future: speculative |
| Phase E0-E6 | `gextract`/`gscreen` GPU fast path | 8-16 weeks total | Phase 1+2 | Future: see Section 16 |
| Phase D0-D4 | Multi-GPU cluster support | 6-12 weeks total | Phase E1+ | Future: see Section 17 |

### Recommended Initial Scope

**Implement Phase 0 + Phase 1 only.** This is the narrowest scope that
delivers measurable value:

1. **Phase 0** establishes the backend dispatch infrastructure, config keys,
   and capability detection. This is necessary for any GPU work and has no
   risk of performance regression.

2. **Phase 1** accelerates `gseq_pwm` (non-spatial only), which is the
   clearest win: pure Python loops with no existing sliding window
   optimization. GPU provides ~2000x speedup over the Python path.

**Why not more initially:**
- Phase 2 (vtrack PWM) competes with `SlideCache` which is already O(stride).
  GPU only wins for non-contiguous access patterns, which need profiling to
  quantify.
- Phase 3 (k-mer) competes with NumPy vectorized code. Marginal for k<=6.
- Phases E0-E6 and D0-D4 are large architectural investments that should be
  gated on demonstrated demand and profiling evidence from Phase 1.

**Gate criteria for subsequent phases:** Phase 2+ should proceed only after
Phase 1 benchmarks show GPU throughput exceeding CPU baseline by >10x on
representative workloads, AND user demand for the specific operation is
documented.

### Phase 0: groundwork (1-2 weeks)

1. Backend config, capability detection, dispatch scaffolding.
2. Add PyTorch runtime checks and baseline benchmark capture.

### Phase 1: PWM core (2-4 weeks)

1. Torch implementation for `gseq_pwm` (non-spatial only).
2. Hook into `gseq_pwm` with `compute_backend` dispatch.
3. Parity tests + crossover heuristic.

### Phase 2: vtrack PWM integration (1-2 weeks)

1. Torch-backed sequence branch dispatch in `vtracks.py`.
2. Support `pwm`, `pwm.max`, `pwm.max.pos`, `pwm.count`.

### Phase 3: k-mer acceleration (2-3 weeks)

1. Torch path for `kmer.count`/`kmer.frac`.
2. Torch path for `gseq_kmer_dist`.

### Phase 4: gsynth_train counting (2-4 weeks)

1. Torch implementation of counting phase with bin-stratified reductions.
2. Validation on real datasets.

### Phase 5: optional custom CUDA specialization (only if needed)

1. Profile Phase 1-4 workloads to identify true hotspots.
2. Add targeted custom kernels only for those hotspots.
3. Keep PyTorch and CPU paths as reference/fallback.

## 14. Risks and Mitigations

| # | Severity | Risk | Mitigation |
|---|---|---|---|
| 1 | **CRITICAL** | **fork-CUDA interaction:** C++ uses `fork()` for 10+ operations. CUDA contexts are not fork-safe. Forking after CUDA init causes undefined behavior. | Enforce `num_kids=0` when GPU active. Add `torch.cuda.is_initialized()` guard before fork. See Section 10. |
| 2 | **HIGH** | **No profiling evidence:** No baseline measurements exist yet to confirm GPU wins over optimized CPU sliding window. | Mandate Phase 0 profiling before Phase 1 implementation. Gate all phases on measured crossover. |
| 3 | **HIGH** | **PyTorch 2-5 GB dependency:** Significant install footprint for a library that currently has minimal dependencies. | Keep PyTorch as optional runtime dep. Document install size. Consider CuPy as lighter alternative if PyTorch proves too heavy. |
| 4 | **HIGH** | **CUDA driver fragmentation:** Different cluster nodes may have different driver versions, causing silent failures or incompatibilities. | Pin minimum driver version (525.60). Add runtime driver version check in `gpu_info()`. Provide clear error messages. |
| 5 | **HIGH** | **CUDA/GIL interaction:** PyTorch CUDA operations release the GIL, but Python callback registration and error handling may re-acquire it at unexpected times. | Test under multithreaded workloads. Document thread-safety guarantees. |
| 6 | **MEDIUM** | **GPU memory fragmentation:** Long Jupyter sessions accumulate fragmented GPU memory, reducing effective batch sizes. | Call `torch.cuda.empty_cache()` between operations. Document kernel restart recommendation. |
| 7 | **MEDIUM** | **No GPU CI:** GitHub Actions does not provide GPU runners. GPU-specific bugs may ship undetected. | Use `@pytest.mark.gpu` markers. Run GPU tests on local cluster before release. Consider self-hosted runner. |
| 8 | **MEDIUM** | **PyTorch first-call overhead:** First PyTorch CUDA operation triggers JIT compilation (2-10 seconds). | Warm up in `gpu_runtime.py` init. Document expected first-call latency. |
| 9 | MEDIUM | Build complexity | Optional component, clear CPU fallback. |
| 10 | MEDIUM | Transfer overhead for small jobs | Strict workload thresholding (`gpu_min_windows`). |
| 11 | LOW | Numeric drift | Dual-run verification mode and per-function tolerances (Section 11). |
| 12 | LOW | GPU memory pressure | Bounded batching (`gpu_max_batch_bases`) and adaptive OOM recovery (Section 4). |
| 13 | LOW | Existing optimized CPU sliding window regressions | Keep CPU path for small/stepwise scans where it wins. Never auto-select GPU for contiguous stride-1 vtracks without profiling evidence. |

## 15. Expected Outcomes

1. Immediate speedups on `gseq_pwm` non-spatial workloads; additional PWM/k-mer gains are deferred to later, profile-gated phases.
2. No behavior change for CPU users.
3. A backend architecture that can later host non-CUDA accelerators (e.g., Halide/OpenCL) with the same dispatch contract.

## 16. Concrete Plan for `gextract` / `gscreen` GPU Fast Path

This is feasible, including virtual tracks, but only as an explicit fast path with strict eligibility checks.

### 16.1 Why it can work

`pm_extract`/`pm_screen` currently spend time in three phases:

1. iterator setup and interval generation (building the scan plan)
2. materializing per-track arrays in `PMTrackExprScanner` batches (I/O + vtrack compute)
3. evaluating expressions / boolean predicates over those materialized arrays

The third phase is SIMD-friendly and can be GPU-accelerated when expressions
are element-wise and track inputs are dense and aligned. The second phase
can also be partially accelerated when vtracks use GPU-native operations
(PWM, k-mer). The first phase remains CPU-only.

### 16.2 Fast-path eligibility (must all hold)

1. 1D only (no 2D intervals, no `band` logic).
2. Track type: dense fixed-bin only.
3. Iterator: fixed-bin policy only.
4. Expression subset only:
   - arithmetic: `+ - * /`
   - comparisons: `< <= > >= == !=`
   - boolean: `& | ~`
   - selected elementwise funcs: `np.minimum`, `np.maximum`, `np.where`, `np.abs`, `np.log`, `np.exp`, `np.isnan`
5. Virtual tracks are allowed only if every referenced vtrack compiles to a GPU-native node:
   - sequence source (`src=None`): `pwm`, `pwm.max`, `pwm.max.pos`, `pwm.count`, `kmer.count`, `kmer.frac`, `masked.count`, `masked.frac`
   - fixed-bin track source (`src=<track>`): `avg`/`mean`, `sum`, `min`, `max`, `exists`, `size`, `lse` (position-returning variants in later phase)
   - `sshift`/`eshift` are integers and become part of the node plan
   - no `filter`, no DataFrame source, no intervals-set source, no random/sample functions, no quantile/stddev/nearest
6. No unsupported Python control flow / object dtype ops.
7. Work size above threshold (`rows * expr_complexity >= gpu_extract_min_work`).

Otherwise fallback to current CPU path unchanged.

### 16.3 Virtual Track IR and Shift Semantics

Compile each used vtrack into a compact node:

1. `name`: expression identifier.
2. `src_kind`: `sequence` or `fixedbin_track`.
3. `src_track`: track name for fixed-bin nodes (empty for sequence).
4. `func`: GPU-native function enum.
5. `params`: normalized numeric payload (PSSM, kmer code, thresholds, strand).
6. `sshift` / `eshift`: per-node interval shifts.

Shift rule (must match current semantics):

1. For iterator interval `I=[s,e)`, evaluate node on `I'=[s+sshift,e+eshift)`.
2. If `I'` is invalid/out-of-chrom, output `NaN` for that row.
3. Relative-position outputs (later phase) are computed from shifted start (`abs_pos - I'.start`).

### 16.4 GPU execution model (with vtracks, PyTorch-first)

1. Reuse existing iterator and interval generation.
2. At each eval buffer:
   - materialize required physical track columns (current dense path)
   - execute GPU-native vtrack nodes for this batch using torch ops
   - sequence nodes: pack shifted intervals and run torch PWM/kmer/masked ops
   - fixed-bin reducer nodes: map shifted intervals to bin spans and run torch segmented reductions
   - execute restricted expression plan over torch tensors
   - for `gscreen`, run predicate + compact mask path in torch
3. Transfer back:
   - `gextract`: selected output columns only
   - `gscreen`: compacted keep-mask or kept interval IDs only
4. Merge adjacent intervals for `gscreen` on CPU (existing logic).

### 16.5 Expression engine plan

**Existing infrastructure:** `pymisha/_safe_eval.py` already provides
`compile_safe_expression(expr, allowed_names)` which validates an expression
AST against an allowlist and compiles it. The current implementation
(`validate_expression_ast`) checks for forbidden node types and restricts
callable functions.

This existing validator can be adapted for GPU expression planning:

1. Extend `validate_expression_ast` to classify each node as GPU-native or
   CPU-only, rather than just accept/reject. The existing allowlist provides
   the starting point for the GPU-compatible operator set.
2. If all nodes are GPU-native, compile a torch execution plan instead of
   a Python `eval` code object.
3. If any node is CPU-only, fall back to the existing `compile_safe_expression`
   path unchanged.
4. Cache compiled plans by normalized expression string (same approach as
   current `compile` caching).
5. Execute plan on torch tensors per batch.
6. Only add custom fused CUDA kernels later if profiling proves torch execution
   is the bottleneck.

This avoids building a custom AST/bytecode engine from scratch and reuses the
existing expression safety infrastructure.

### 16.6 Integration points

1. Python `pymisha/extract.py`:
   - replace current hard fallback on vtracks with plan-based dispatch
   - build `GPUExprPlan` from parsed expressions + referenced `_shared._VTRACKS`
   - call torch fast path when all nodes are GPU-native
2. `PMTrackExprScanner`:
   - keep current C++ CPU path unchanged for fallback.
3. new Python modules:
   - `pymisha/gpu_extract.py`
   - `pymisha/gpu_pwm.py`
   - `pymisha/gpu_kmer.py`
   - `pymisha/gpu_dist.py`
4. optional later C++/CUDA modules:
   - added only for profiled hotspots that need lower-level kernels.

### 16.7 Config surface

Extend `CONFIG`:

1. `extract_backend`: `"auto" | "cpu" | "gpu"`
2. `gpu_extract_min_work`: int
3. `gpu_extract_max_batch_rows`: int
4. `gpu_extract_verify`: bool
5. `gpu_extract_allow_vtracks`: bool
6. `gpu_vtrack_seq_min_windows`: int
7. `gpu_vtrack_track_min_bins`: int

### 16.8 Validation strategy

1. Golden parity tests CPU vs GPU for eligible expressions and vtracks:
   - single-track and multi-track arithmetic
   - mixed expressions (`track + vtrack`, `np.where(vtrack > t, ...)`)
   - predicate-heavy `gscreen`
   - NaN-heavy edge cases
2. Shift parity matrix for each supported vtrack function:
   - `sshift/eshift` in-range, clipped, and out-of-range cases
   - boundary behavior at chromosome starts/ends
3. Run randomized expression fuzzing within allowlist.
4. Verification mode in CI for sampled batches (`gpu_extract_verify=1`).

### 16.9 Rollout phases for `gextract`/`gscreen`

#### Phase E0 (instrumentation)

1. Add per-phase timers in scanner (`load`, `eval`, `pack`, `build_df`).
2. Add planner/classifier that labels each vtrack as GPU-native or fallback.
3. Capture real workloads to set thresholds.

#### Phase E1 (torch `gscreen`, physical tracks only)

1. Implement torch predicate path.
2. Keep interval merge on CPU.
3. Ship behind opt-in config.

#### Phase E2 (torch `gscreen` with sequence-based vtracks)

1. Add torch sequence-node execution (`pwm*`, `kmer.*`, `masked.*`) with shifts.
2. Keep unsupported vtracks on existing CPU fallback path.
3. Add parity tests against current Python vtrack route.

#### Phase E3 (torch `gextract` with sequence-based vtracks)

1. Implement restricted torch expression plan for one expression.
2. Support mixed `track + vtrack` expressions.
3. Add column transfer and parity tests.

#### Phase E4 (fixed-bin source vtracks on torch path)

1. Implement segmented reductions for `avg/sum/min/max/exists/size/lse` over shifted bin spans.
2. Add support in both `gextract` and `gscreen`.
3. Keep position-returning variants behind a separate gate.

#### Phase E5 (multi-expression + wider op set)

1. Multi-expression evaluation and cache reuse.
2. Add `np.where`/min/max/isnan support.

#### Phase E6 (optional low-level specialization)

1. Add custom CUDA kernels only for profiled bottlenecks.
2. Dense bulk reader + pinned buffers for better H2D throughput.
3. Re-evaluate sparse support separately.

## 17. Multi-GPU Cluster Plan

> **Note:** This entire section is speculative future work. Multi-GPU support
> should only be pursued after single-GPU acceleration (Phase 0 + Phase 1) is
> validated and user demand is confirmed. `torchrun` is designed for
> minute-scale training jobs; pymisha operations typically complete in seconds,
> making the overhead of distributed setup potentially dominant.

The distributed pattern below is based on standard `torchrun` conventions.

### 17.1 Observed pattern to reuse

The standard `torchrun` distributed pattern used in PyTorch training pipelines:

1. GPU list comes from `CUDA_VISIBLE_DEVICES`; process count is derived from it.
2. Launcher sets `MASTER_PORT`, `OMP_NUM_THREADS`, and NCCL flags:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   export MASTER_PORT=29500
   export OMP_NUM_THREADS=4
   # Hardware-tuned NCCL transport flags:
   export NCCL_P2P_DISABLE=1    # if P2P not supported
   export NCCL_IB_DISABLE=1     # if InfiniBand not available
   ```
3. NCCL transport is hardware-tuned by GPU type.
4. Runtime gets `RANK`/`WORLD_SIZE`/`LOCAL_RANK`, binds each process to one GPU.

Work sharding follows a size-aware round-robin pattern:

```python
# Sort intervals by length (descending), assign round-robin to ranks
intervals_sorted = sorted(intervals, key=lambda x: x.length, reverse=True)
for i, interval in enumerate(intervals_sorted):
    rank_assignment[i % world_size].append(interval)
```

Each rank writes rank-local output; rank 0 performs final merge.

### 17.2 PyMisha multi-GPU execution modes

1. `torchrun_shard` (first target): one rank per GPU, launched externally.
2. `spawn_shard` (single-node fallback): parent process spawns one worker per GPU.

First implementation target is `torchrun_shard` because it is standard and robust on cluster infrastructure.

### 17.3 Concrete design for `gextract` / `gscreen`

1. Keep public API unchanged (`gextract`, `gscreen`); distributed mode is internal and opt-in.
2. Rank discovery from env:
   - `RANK`, `WORLD_SIZE`, `LOCAL_RANK` (default `0/1/0`).
3. Device binding:
   - map `LOCAL_RANK` to visible device index and call `torch.cuda.set_device(local_rank)` before init.
4. Sharding policy:
   - compute per-interval weight = `interval_len * expr_complexity`.
   - sort descending and assign in round-robin across ranks (same strategy as Borzoi).
5. Rank-local compute:
   - each rank runs torch GPU fast path on its shard only.
   - unsupported expressions/vtracks still fallback locally to CPU path.
6. Rank-local output:
   - **Preferred (single-node):** Use `torch.distributed` gather to collect
     results in-memory on rank 0. This avoids filesystem I/O and is
     appropriate for single-node multi-GPU where all ranks share memory.
   - **Fallback (multi-node):** Write `.rank_<r>.extract.parquet` /
     `.rank_<r>.screen.parquet` with `intervalID` for cross-node merging.
     Parquet is justified for multi-node because ranks may be on different
     machines without shared memory.
7. Synchronization and merge:
   - use `dist.barrier()` for synchronization.
   - rank 0 merges outputs, sorts by `intervalID`, applies final `merge_adjacent` for `gscreen`, writes final result.
   - non-zero ranks exit after writing shard output.
8. Failure handling:
   - propagate rank exceptions via process exit status / torch elastic failure handling.
   - if any rank fails, abort merge and surface rank-specific error summary.

### 17.4 Config and CLI surface

Add config keys:

1. `gpu_dist_mode`: `"off" | "torchrun_shard" | "spawn_shard"`
2. `gpu_dist_launcher`: `"torchrun" | "spawn"`
3. `gpu_dist_merge_format`: `"parquet"`
4. `gpu_dist_timeout_sec`: int
5. `gpu_nccl_profile`: `"auto" | "l40s" | "blackwell" | "custom"`

Add helper scripts:

1. `scripts/run_gpu_extract_mcluster11.sh`
2. `scripts/run_gpu_screen_mcluster11.sh`

These scripts should mirror Borzoi launch behavior:

1. derive process count from `CUDA_VISIBLE_DEVICES`
2. set `MASTER_PORT`
3. set NCCL flags by detected GPU type
4. launch one process per GPU with `torchrun`

### 17.5 C++ / Python integration points

1. `pymisha/extract.py`:
   - add distributed shard planner and rank-local runner using torch runtime.
   - keep existing `gextract`/`gscreen` API unchanged; distributed mode is opt-in.
2. Python GPU modules:
   - `pymisha/gpu_dist.py` for rank/init helpers.
   - `pymisha/gpu_extract.py` for rank-local compute + parquet IO.
3. `_pymisha` C++:
   - keep `run_extract` / `compute_screen` unchanged for CPU baseline/fallback.
4. Telemetry:
   - expose per-rank `rows_processed`, `gpu_ms`, `h2d_ms`, `compute_ms`, `d2h_ms`.

### 17.6 Rollout phases for multi-GPU

#### Phase D0 (launcher + env contract)

1. Add `torchrun` rank/env parsing and device binding helpers.
2. Add mcluster11 helper scripts with Borzoi-like NCCL toggles.

#### Phase D1 (`gextract` distributed merge)

1. Implement shard planner + rank-local extract outputs.
2. `dist.barrier()` then rank 0 merge by `intervalID` and return DataFrame.

#### Phase D2 (`gscreen` distributed merge)

1. Rank-local screen outputs.
2. `dist.barrier()` then rank 0 global sort + `merge_adjacent` for final semantics.

#### Phase D3 (robustness hardening)

1. Rank failure propagation and timeout policy.
2. Retry and partial-output cleanup policy.

#### Phase D4 (performance tuning)

1. Validate sharding balance on real interval length distributions.
2. Tune batch size and overlap H2D/compute per rank.

## 18. API Surface Changes

No existing function signatures change. GPU acceleration is additive and opt-in,
and new config keys default to CPU-equivalent behavior in `auto` mode.

### New public functions

| Function | Module | Description | Phase |
|---|---|---|---|
| `pm.gpu_info()` | `pymisha/gpu_runtime.py` | Returns dict with GPU availability, device name, driver version, memory | Phase 0 |

### New config keys

See [Consolidated Config Reference](#consolidated-config-reference) in Section 4
for the complete list of all new config keys, their types, defaults, and valid
ranges.

### New internal modules

| Module | Purpose | Phase |
|---|---|---|
| `pymisha/gpu_runtime.py` | Device discovery, dtype policy, stream helpers | Phase 0 |
| `pymisha/gpu_pwm.py` | GPU PWM/PSSM scoring | Phase 1 |
| `pymisha/gpu_kmer.py` | GPU k-mer operations | Phase 3 |
| `pymisha/gpu_extract.py` | GPU gextract/gscreen fast path | Phase E1 |
| `pymisha/gpu_dist.py` | Distributed rank/world setup | Phase D0 |

### Backward compatibility guarantees

1. No existing function signatures change.
2. Existing config keys keep their current defaults; new GPU-specific keys are additive.
3. `compute_backend` defaults to `"auto"`, which selects CPU unless GPU is
   available AND workload exceeds threshold. In practice, this means existing
   code behaves identically.
4. PyTorch is not imported until a GPU path is actually triggered.
5. ImportError for missing PyTorch is caught and logged, never raised to user
   in `auto` mode.

## 19. Profiling and Observability

### CPU baseline profiling methodology

Before implementing any GPU path, collect baseline measurements:

1. **Micro-benchmarks:** Time individual operations (`gseq_pwm`, vtrack PWM
   with SlideCache, `gseq_kmer_dist`) using `time.perf_counter_ns()` with
   10+ repetitions and warm-up runs.
2. **Workload characterization:** For each candidate operation, measure:
   - Total scored positions or bases processed
   - Wall-clock time and CPU time
   - Memory allocation (via `tracemalloc`)
   - Cache miss rate (via `perf stat` on Linux)
3. **Representative datasets:** Use the test database for unit-level
   benchmarks. Use full-genome intervals for realistic scaling measurements.

### GPU profiling with `torch.cuda.Event`

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... GPU operation ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

### Per-operation timing breakdown

Expose timing via `pm.gpu_info()` extended dict when `gpu_verify` is enabled:

| Metric | Description |
|---|---|
| `h2d_ms` | Host-to-device transfer time |
| `compute_ms` | GPU kernel execution time |
| `d2h_ms` | Device-to-host transfer time |
| `total_gpu_ms` | Total GPU wall-clock time |
| `cpu_baseline_ms` | CPU comparison time (if `gpu_verify` enabled) |
| `speedup` | `cpu_baseline_ms / total_gpu_ms` |

### Benchmark suite

Add `tests/bench_gpu_perf.py` with:
1. PWM scoring: sweep motif length (8, 12, 20), interval count (100, 1K, 10K, 100K).
2. k-mer distribution: sweep k (4, 6, 8), total bases (1M, 10M, 100M).
3. Report CPU vs GPU throughput and crossover point.
4. Output JSON for automated regression tracking.

## 20. GPU Test Infrastructure

### Running GPU tests without a GPU

1. All GPU test files use `@pytest.mark.gpu` marker.
2. `conftest.py` auto-skips GPU tests when `torch.cuda.is_available()` returns
   `False`.
3. GPU parity logic can be tested with CPU tensors by setting
   `compute_backend="gpu"` with `torch` CPU fallback (tests correctness of
   data pipeline and reductions, not GPU-specific behavior).

### CI matrix

| Environment | GPU | What it tests |
|---|---|---|
| GitHub Actions (Linux x86_64) | No | CPU paths, data pipeline, parity logic on CPU tensors |
| GitHub Actions (macOS arm64) | No | CPU paths, import guards, graceful degradation |
| Local cluster (L40S / A100) | Yes | Full GPU parity, performance benchmarks, OOM handling |

### Parity test framework

Each GPU test follows this pattern:

```python
@pytest.mark.gpu
def test_gseq_pwm_parity():
    cpu_result = gseq_pwm(seqs, pssm, compute_backend="cpu")
    gpu_result = gseq_pwm(seqs, pssm, compute_backend="gpu")
    np.testing.assert_allclose(cpu_result, gpu_result, atol=1e-4, rtol=1e-3)
```

### Regression detection

1. Benchmark results are stored as JSON with git commit hash.
2. CI compares against previous baseline and flags >10% regressions.
3. GPU memory usage is tracked per test to detect leaks.

### Test data requirements

1. Reuse existing test database (`tests/testdb/trackdb/test`) for unit tests.
2. Add synthetic large-interval datasets for crossover benchmarks.
3. Add edge-case sequences: all-N, single base, maximum length, mixed strands.

## 21. User Migration Guide

### Installation

GPU acceleration requires PyTorch as an optional dependency:

```bash
# Install pymisha (no GPU support)
pip install pymisha

# Add GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

PyTorch is ~2-5 GB depending on platform and CUDA version.

### Auto mode decision logic

When `compute_backend="auto"` (the default):

```
Is torch installed?
  No  -> CPU path (silent)
  Yes -> Is CUDA available?
    No  -> CPU path (silent)
    Yes -> Is workload >= gpu_min_windows?
      No  -> CPU path (no overhead)
      Yes -> GPU path
```

No user action is required. The default behavior is identical to CPU-only
pymisha unless PyTorch with CUDA is installed AND the workload is large enough.

### GPU verification

```python
import pymisha as pm

# Check GPU status
info = pm.gpu_info()
print(info)
# {'available': True, 'device_name': 'NVIDIA L40S', 'driver_version': '535.129.03',
#  'pytorch_version': '2.3.0', 'cuda_version': '12.1', 'free_memory_mb': 45000,
#  'total_memory_mb': 48000}
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `pm.gpu_info()` shows `available: False` | PyTorch not installed or no CUDA GPU | Install PyTorch with CUDA support |
| GPU path not triggered | Workload below `gpu_min_windows` threshold | Lower threshold or use `compute_backend="gpu"` |
| `RuntimeError: CUDA out of memory` | Batch too large for GPU memory | Reduce `gpu_max_batch_bases` |
| `RuntimeError: GPU mode incompatible with num_kids > 0` | fork-CUDA conflict | Set `num_kids=0` or use `compute_backend="cpu"` |
| First call is slow (2-10 seconds) | PyTorch JIT compilation | Normal on first use; subsequent calls are fast |

### Forcing CPU

To disable GPU acceleration entirely:

```python
pm.gsetparam("compute_backend", "cpu")
```

Or set environment variable:

```bash
export PYMISHA_COMPUTE_BACKEND=cpu
```

---

## Glossary

| Term | Definition |
|---|---|
| **AST** | Abstract syntax tree; tree representation of parsed expression code |
| **CDF** | Cumulative distribution function |
| **D2H** | Device-to-host transfer; copying data from GPU memory to CPU memory |
| **FIFO** | First in, first out; queue ordering used in current multitasking merge |
| **H2D** | Host-to-device transfer; copying data from CPU memory to GPU memory |
| **IR** | Intermediate representation; compiled form of vtrack/expression plans |
| **LSE** | Log-sum-exp; numerically stable way to compute log(sum(exp(x_i))) |
| **NCCL** | NVIDIA Collective Communications Library; used for multi-GPU communication |
| **Phase labels** | `Phase 0..5` = core GPU rollout, `Phase E*` = `gextract`/`gscreen` fast-path rollout, `Phase D*` = distributed multi-GPU rollout |
| **PSSM** | Position-specific scoring matrix; the weight matrix used in PWM scoring |
| **ROI** | Region of interest; the genomic interval being evaluated |
| **SIMD** | Single instruction, multiple data; parallel processing of data vectors |
