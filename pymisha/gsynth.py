"""Genome synthesis functions (gsynth.*)."""

from __future__ import annotations

import logging as _logging
import multiprocessing as _multiprocessing
import os as _os
import zipfile as _zipfile
from dataclasses import dataclass as _dataclass
from dataclasses import field as _field
from typing import Any as _Any

import numpy as _numpy
import pandas as _pd
import yaml as _yaml  # type: ignore[import-untyped]

from ._safe_pickle import restricted_load
from ._shared import _checkroot, _df2pymisha, _pymisha
from .extract import _maybe_load_intervals_set, gextract
from .intervals import gintervals_all

_logger = _logging.getLogger(__name__)

# Default chunk size threshold for parallel processing (1 billion bases)
GSYNTH_MAX_CHUNK_SIZE = int(1e9)

# ---------------------------------------------------------------------------
# Model dataclass
# ---------------------------------------------------------------------------

@_dataclass
class GsynthModel:
    """Trained stratified Markov model for genome synthesis.

    Stores the transition probabilities (as CDFs) for a k-th order Markov
    chain, optionally stratified by one or more genomic track dimensions.
    Each of the ``4^k`` possible k-mer contexts maps to a probability
    distribution over the four nucleotides (A, C, G, T), independently for
    every flat bin in the stratification grid.  The model is created by
    :func:`gsynth_train` and consumed by :func:`gsynth_sample`.

    Attributes
    ----------
    k : int
        Markov order (context length).  Default 5 for backward compatibility.
    n_dims : int
        Number of stratification dimensions.
    dim_sizes : list of int
        Number of bins per dimension.
    dim_specs : list of dict
        Per-dimension specification (expr, breaks, num_bins, bin_map).
    total_bins : int
        Product of all dim_sizes (total flat bins).
    model_data : dict
        Contains ``'counts'`` (list of 2-D arrays, one per flat bin, shape
        ``(4^k, 4)``) and ``'cdf'`` (same layout, cumulative probabilities).
    total_kmers : int
        Total k-mers counted during training.
    per_bin_kmers : numpy.ndarray
        K-mers per flat bin.
    total_masked : int
        Positions skipped due to mask.
    total_n : int
        Positions skipped due to N bases.
    pseudocount : float
        Pseudocount used for CDF normalization.

    See Also
    --------
    gsynth_train : Create a ``GsynthModel`` from genome sequences.
    gsynth_sample : Generate synthetic sequences from a model.
    gsynth_save : Persist a model to disk.
    gsynth_load : Restore a model from disk.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> model = pm.gsynth_train()
    >>> model.n_dims
    0
    >>> model.total_bins
    1
    """

    k: int = 5
    n_dims: int = 0
    dim_sizes: list[int] = _field(default_factory=list)
    dim_specs: list[dict[str, _Any]] = _field(default_factory=list)
    total_bins: int = 0
    model_data: dict[str, _Any] = _field(default_factory=dict)
    total_kmers: int = 0
    per_bin_kmers: _Any = None  # numpy array
    total_masked: int = 0
    total_n: int = 0
    pseudocount: float = 1.0
    min_obs: int = 0
    iterator: int | None = None

    @property
    def num_kmers(self) -> int:
        """Number of k-mer contexts (4^k)."""
        return int(4 ** self.k)

    def __repr__(self) -> str:
        lines = [
            f"Synthetic Genome Markov-{self.k} Model",
            f"  Markov order: {self.k}",
            f"  Dimensions: {self.n_dims}",
            f"  Total bins: {self.total_bins}",
            f"  Dim sizes:  {self.dim_sizes}",
            f"  Total k-mers: {self.total_kmers:,}",
            f"  Masked positions: {self.total_masked:,}",
            f"  N positions: {self.total_n:,}",
            f"  Pseudocount: {self.pseudocount}",
        ]
        if self.min_obs > 0:
            lines.append(f"  Min observations: {self.min_obs}")
        for i, spec in enumerate(self.dim_specs):
            lines.append(f"  Dim {i + 1}: expr='{spec.get('expr', '')}', "
                         f"bins={spec.get('num_bins', '?')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# gsynth_bin_map — pure Python
# ---------------------------------------------------------------------------

def gsynth_bin_map(breaks: list[float] | _numpy.ndarray, merge_ranges: list[dict[str, _Any]]) -> _numpy.ndarray:
    """Compute bin mapping for merging sparse bins.

    Converts value-based merge specifications into an integer index array that
    redirects source bins to target bins. This is useful when certain
    stratification bins have too few observations to learn reliable transition
    probabilities -- their counts can be folded into a neighbouring,
    better-populated bin.

    Parameters
    ----------
    breaks : array-like
        Sorted bin boundaries (length = ``num_bins + 1``).
    merge_ranges : list of dict
        Each dict has:

        - ``"from"`` : float or tuple of (lo, hi) -- source value or range to
          remap.  A scalar ``v`` is shorthand for ``(v, inf)``.
        - ``"to"`` : tuple of (lo, hi) -- target value range whose bin receives
          the merged counts.  Must overlap exactly one bin defined by *breaks*.

    Returns
    -------
    numpy.ndarray
        Integer array of length ``num_bins``, where ``bin_map[i]`` is the
        0-based target bin index for source bin ``i``.  Unmapped bins map to
        themselves (identity).

    Raises
    ------
    ValueError
        If *breaks* has fewer than 2 elements, or if a ``"to"`` range does not
        match any bin in *breaks*.

    See Also
    --------
    gsynth_train : Accepts ``bin_merge`` specifications per dimension.

    Examples
    --------
    >>> from pymisha import gsynth_bin_map
    >>> breaks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    >>> gsynth_bin_map(breaks, [{"from": (0.4, 0.5), "to": (0.3, 0.4)}])
    array([0, 1, 2, 3, 3], dtype=int32)

    Multiple merges -- fold both tails into the centre:

    >>> breaks = [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> gsynth_bin_map(breaks, [
    ...     {"from": (0.0, 0.25), "to": (0.25, 0.5)},
    ...     {"from": (0.75, 1.0), "to": (0.5, 0.75)},
    ... ])
    array([1, 1, 2, 2], dtype=int32)
    """
    breaks = _numpy.asarray(breaks, dtype=float)
    num_bins = len(breaks) - 1
    if num_bins < 1:
        raise ValueError("breaks must have at least 2 elements")

    bin_map = _numpy.arange(num_bins, dtype=_numpy.int32)

    for spec in merge_ranges:
        from_val = spec.get("from")
        to_val = spec.get("to")

        if to_val is None:
            raise ValueError("merge_ranges entry must have a 'to' key")

        # Determine target bin
        to_lo, to_hi = (to_val, to_val) if _numpy.isscalar(to_val) else to_val
        to_mid = (float(to_lo) + float(to_hi)) / 2.0  # type: ignore[arg-type]
        target_bin = _numpy.searchsorted(breaks[:-1], to_mid, side="right") - 1
        target_bin = int(_numpy.clip(target_bin, 0, num_bins - 1))  # type: ignore[assignment]

        # Verify target bin matches the specified range
        if not (breaks[target_bin] <= to_hi and breaks[target_bin + 1] >= to_lo):
            raise ValueError(
                f"Target range ({to_lo!r}, {to_hi!r}) does not match any bin in breaks"
            )

        # Determine source bins
        if from_val is None:
            continue

        if _numpy.isscalar(from_val):
            from_lo = float(from_val)  # type: ignore[arg-type]
            from_hi = float("inf")
        else:
            from_lo, from_hi = float(from_val[0]), float(from_val[1])

        # Map source bins to target
        for i in range(num_bins):
            bin_lo = breaks[i]
            bin_hi = breaks[i + 1]
            bin_mid = (bin_lo + bin_hi) / 2.0
            if from_lo <= bin_mid <= from_hi:
                bin_map[i] = target_bin

    return bin_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Sentinel passed to the C++ backend when no iterator constraint applies
# (0D models, gsynth_random, legacy models). Equals INT64_MAX, which makes
# `pos < bin_start + iter_size` true for every reachable position.
_ITER_SIZE_NO_CONSTRAINT = 9223372036854775807


def _resolve_iter_size(iterator: int | None) -> int:
    """Resolve an iterator into a positive int suitable for the C backend.

    The C side now requires the caller to pass the iterator value explicitly
    instead of inferring it from ``iter_starts`` diffs (which is wrong when
    intervals are not aligned to the iterator bin boundary). ``None`` means
    "no constraint" and maps to the INT64_MAX sentinel.
    """
    if iterator is None:
        return _ITER_SIZE_NO_CONSTRAINT
    iter_int = int(iterator)
    if iter_int <= 0:
        raise ValueError(f"iterator must be a positive integer, got {iter_int}")
    return iter_int


def _extract_bin_data(
    dim_specs: list[dict[str, _Any]],
    intervals: _pd.DataFrame | None,
    iterator: int | None,
    sample_bin_merge: list[_Any] | None = None,
) -> tuple[_numpy.ndarray, _numpy.ndarray, _numpy.ndarray, list[float], _numpy.ndarray | None, list[int]]:
    """Extract track values and compute flat bin indices.

    Parameters
    ----------
    dim_specs : list of dict
        Per-dimension specification from the model.
    intervals : DataFrame
        Genomic intervals.
    iterator : int or None
        Iterator bin size.
    sample_bin_merge : list or None
        Optional sampling-time bin merge overrides, one per dimension.
        Each element is either ``None`` (use training-time bin_merge) or
        a list of merge-range dicts passed to :func:`gsynth_bin_map`.

    Returns
    -------
    bin_indices : numpy.ndarray (int32)
        Flat bin index per iterator position.
    iter_starts : numpy.ndarray (int64)
        Start position per iterator position.
    iter_chroms : numpy.ndarray (int32)
        Chrom index per iterator position.
    breaks : list of float
        Flat bin boundaries for the C++ layer (num_bins+1).
    bin_map : numpy.ndarray or None
        Combined bin mapping for the C++ layer, or None.
    dim_sizes : list of int
        Number of bins per dimension.
    """
    if not dim_specs:
        # 0D model: single bin
        _checkroot()
        all_intervals = intervals if intervals is not None else gintervals_all()
        bin_indices = _numpy.zeros(len(all_intervals), dtype=_numpy.int32)
        iter_starts = all_intervals["start"].to_numpy(dtype=_numpy.int64)

        # Resolve chromids via misha's internal chromkey order (same approach as
        # the multi-D branch below). Previously iter_chroms was hardcoded to
        # zero, so the C++ backend (PMGsynth.cpp) would route every iterator
        # entry to chrom_bins[0] and leave chrom_bins[c] empty for every other
        # chromosome. Intervals on any chromosome other than chromkey ID 0 were
        # then silently dropped in training (no k-mers counted) and fell back
        # to uniform random sampling instead of using the trained Markov CDF.
        all_chroms = gintervals_all()
        chrom_to_id = {str(name): i for i, name in enumerate(all_chroms["chrom"])}
        iter_chroms = _numpy.array(
            [chrom_to_id.get(str(c), -1) for c in all_intervals["chrom"]],
            dtype=_numpy.int32,
        )
        return bin_indices, iter_starts, iter_chroms, [0.0, 1.0], None, [1]

    # Extract track values for each dimension
    from .summary import _bin_values

    dim_data = []
    dim_sizes = []

    for d_idx, spec in enumerate(dim_specs):
        expr = spec["expr"]
        breaks = _numpy.asarray(spec["breaks"], dtype=float)
        n_bins = len(breaks) - 1
        dim_sizes.append(n_bins)

        # Extract track values using the iterator
        df = gextract(expr, intervals=intervals, iterator=iterator)
        if df is None or len(df) == 0:
            raise ValueError(f"No data extracted for expression '{expr}'")

        # Get values column (last non-interval column)
        val_cols = [c for c in df.columns if c not in {"chrom", "start", "end", "intervalID"}]
        if not val_cols:
            raise ValueError(f"No value column in extraction for '{expr}'")
        values = df[val_cols[0]].to_numpy(dtype=float)

        # Keep binning semantics consistent with gdist/gbins.
        bin_idx = _bin_values(values, breaks, include_lowest=False)

        # Determine effective bin_merge for this dimension:
        # sample_bin_merge overrides training-time bin_merge when provided.
        if sample_bin_merge is not None and d_idx < len(sample_bin_merge) and sample_bin_merge[d_idx] is not None:
            bm_spec = sample_bin_merge[d_idx]
        else:
            bm_spec = spec.get("bin_merge")
        if bm_spec:
            bm = gsynth_bin_map(breaks, bm_spec)
            valid = bin_idx >= 0
            bin_idx[valid] = bm[bin_idx[valid]]

        dim_data.append({
            "bin_idx": bin_idx,
            "n_bins": n_bins,
            "starts": df["start"].to_numpy(dtype=_numpy.int64),
            "chroms": df["chrom"].to_numpy(),
        })

    # Compute flat bin index: idx = d0 + d0_size * (d1 + d1_size * d2 ...)
    n = len(dim_data[0]["bin_idx"])
    flat_idx = _numpy.zeros(n, dtype=_numpy.int64)
    global_valid = _numpy.ones(n, dtype=bool)
    total_bins = 1
    for _i, dd in enumerate(dim_data):
        if len(dd["bin_idx"]) != n:
            raise ValueError("All dimensions must extract the same number of positions")
        idx_arr = dd["bin_idx"].astype(_numpy.int64, copy=False)
        valid = idx_arr >= 0
        global_valid &= valid
        flat_idx += _numpy.where(valid, idx_arr, 0) * total_bins
        total_bins *= dd["n_bins"]
    flat_idx = flat_idx.astype(_numpy.int32, copy=False)
    flat_idx[~global_valid] = -1

    # Convert chrom strings to integer IDs
    iter_starts = dim_data[0]["starts"]
    chrom_strs = dim_data[0]["chroms"]

    # Build chrom name -> id mapping
    _checkroot()
    all_chroms = gintervals_all()
    chrom_to_id = {name: i for i, name in enumerate(all_chroms["chrom"])}
    iter_chroms = _numpy.array(  # type: ignore[assignment]
        [chrom_to_id.get(str(c), -1) for c in chrom_strs], dtype=_numpy.int32
    )

    # Create flat breaks for C++: just [0, 1, 2, ..., total_bins]
    flat_breaks: list[float] = [float(x) for x in range(total_bins + 1)]

    return flat_idx, iter_starts, iter_chroms, flat_breaks, None, dim_sizes


# ---------------------------------------------------------------------------
# Parallel processing helpers
# ---------------------------------------------------------------------------

def _compute_total_bases(intervals: _pd.DataFrame) -> int:
    """Compute total bases covered by intervals DataFrame."""
    return int((intervals["end"] - intervals["start"]).sum())


def _should_parallelize(intervals: _pd.DataFrame, allow_parallel: bool, num_cores: int | None,
                        max_chunk_size: int | None = None) -> tuple[bool, int]:
    """Determine whether parallel processing should be used.

    Returns (do_parallel, effective_cores) tuple.
    """
    if not allow_parallel:
        return False, 1

    if max_chunk_size is None:
        max_chunk_size = GSYNTH_MAX_CHUNK_SIZE

    total_bases = _compute_total_bases(intervals)
    if total_bases <= max_chunk_size:
        return False, 1

    n_rows = len(intervals)
    if n_rows <= 1:
        return False, 1

    if num_cores is None:
        num_cores = _multiprocessing.cpu_count()
    effective_cores = max(1, min(int(num_cores), n_rows))
    if effective_cores <= 1:
        return False, 1

    return True, effective_cores


def _chunk_intervals(intervals: _pd.DataFrame, n_chunks: int) -> list[_pd.DataFrame]:
    """Split intervals DataFrame into approximately equal chunks by row.

    Each chunk is a contiguous slice of the intervals DataFrame.
    Returns a list of DataFrames.
    """
    n_rows = len(intervals)
    if n_chunks >= n_rows:
        # One row per chunk
        return [intervals.iloc[[i]].reset_index(drop=True)
                for i in range(n_rows)]

    chunk_size = n_rows // n_chunks
    remainder = n_rows % n_chunks
    chunks = []
    start = 0
    for i in range(n_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(intervals.iloc[start:end].reset_index(drop=True))
        start = end
    return chunks


def _generate_chunk_seeds(seed: int | None, n_chunks: int) -> list[int | None]:
    """Generate reproducible per-chunk seeds from a master seed.

    If seed is None, returns a list of Nones.
    """
    if seed is None:
        return [None] * n_chunks
    rng = _numpy.random.RandomState(seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(n_chunks)]


def _worker_train_chunk(args: tuple[_Any, ...]) -> dict[str, _Any]:
    """Worker function for parallel gsynth_train.

    Runs in a forked subprocess. The child inherits the parent's
    fully-initialized C++ and Python state via fork(), so no
    re-initialization is needed.

    Parameters
    ----------
    args : tuple
        (chunk_intervals_dict, dim_specs_dicts, mask_dict,
         iterator, pseudocount, total_bins, parsed_specs, k)

    Returns
    -------
    dict
        With keys 'counts', 'total_kmers', 'per_bin_kmers',
        'total_masked', 'total_n'.
    """
    (chunk_intervals_dict, dim_specs_dicts, mask_dict,
     iterator, pseudocount, total_bins, parsed_specs_data, k) = args

    import pandas as pd
    chunk_intervals = pd.DataFrame(chunk_intervals_dict)

    mask = pd.DataFrame(mask_dict) if mask_dict is not None else None

    # Reconstruct parsed_specs
    parsed_specs = []
    for sp in parsed_specs_data:
        parsed_specs.append({
            "expr": sp["expr"],
            "breaks": sp["breaks"],
            "num_bins": sp["num_bins"],
            "bin_merge": sp.get("bin_merge"),
            "bin_map": sp.get("bin_map"),
        })

    # Extract bin data for this chunk
    bin_indices, iter_starts, iter_chroms, flat_breaks, bin_map, dim_sizes = \
        _extract_bin_data(parsed_specs, chunk_intervals, iterator)

    # Call C++ backend
    py_mask = _df2pymisha(mask) if mask is not None else None
    result = _pymisha.pm_gsynth_train(
        _df2pymisha(chunk_intervals),
        bin_indices,
        iter_starts,
        iter_chroms,
        flat_breaks,
        bin_map,
        py_mask,
        float(pseudocount),
        int(k),
        _resolve_iter_size(iterator),
    )

    # Return only what we need for merging (numpy arrays + scalars)
    return {
        "counts": result["counts"],
        "total_kmers": int(result["total_kmers"]),
        "per_bin_kmers": result["per_bin_kmers"].copy(),
        "total_masked": int(result["total_masked"]),
        "total_n": int(result["total_n"]),
    }


def _merge_train_results(
    chunk_results: list[dict[str, _Any]],
    total_bins: int,
    pseudocount: float,
    k: int = 5,
) -> dict[str, _Any]:
    """Merge training results from multiple chunks.

    Sums count arrays across chunks and recomputes CDFs.

    Parameters
    ----------
    chunk_results : list of dict
        Each dict has 'counts', 'total_kmers', 'per_bin_kmers',
        'total_masked', 'total_n'.
    total_bins : int
        Total number of flat bins.
    pseudocount : float
        Pseudocount for CDF computation.
    k : int
        Markov order.

    Returns
    -------
    dict
        Merged result with 'counts', 'cdf', 'total_kmers',
        'per_bin_kmers', 'total_masked', 'total_n'.
    """
    num_kmers = 4 ** k

    # Initialize with zeros
    merged_counts = [_numpy.zeros((num_kmers, 4), dtype=_numpy.float64)
                     for _ in range(total_bins)]
    merged_total_kmers = 0
    merged_per_bin_kmers = _numpy.zeros(total_bins, dtype=_numpy.float64)
    merged_total_masked = 0
    merged_total_n = 0

    for cr in chunk_results:
        merged_total_kmers += cr["total_kmers"]
        merged_per_bin_kmers += cr["per_bin_kmers"]
        merged_total_masked += cr["total_masked"]
        merged_total_n += cr["total_n"]
        for b in range(total_bins):
            merged_counts[b] += cr["counts"][b]

    # Recompute CDFs from merged counts
    merged_cdf = []
    for b in range(total_bins):
        counts_b = merged_counts[b] + pseudocount
        row_sums = counts_b.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        probs = counts_b / row_sums
        cdf = _numpy.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0  # ensure last column is exactly 1
        merged_cdf.append(cdf)

    return {
        "counts": merged_counts,
        "cdf": merged_cdf,
        "total_kmers": merged_total_kmers,
        "per_bin_kmers": merged_per_bin_kmers,
        "total_masked": merged_total_masked,
        "total_n": merged_total_n,
    }


def _worker_sample_chunk(args: tuple[_Any, ...]) -> list[str]:
    """Worker function for parallel gsynth_sample.

    Runs in a forked subprocess. The child inherits the parent's
    fully-initialized C++ and Python state via fork(), so no
    re-initialization is needed.

    Parameters
    ----------
    args : tuple
        (chunk_intervals_dict, dim_specs_list, cdf_list,
         iterator, mask_copy_dict, n_samples, chunk_seed, bin_merge, k)

    Returns
    -------
    list of str
        Sampled sequences for this chunk.
    """
    (chunk_intervals_dict, dim_specs_list, cdf_list_data,
     iterator, mask_copy_dict, n_samples, chunk_seed,
     sample_bin_merge, k, preserve_n) = args

    import pandas as pd
    chunk_intervals = pd.DataFrame(chunk_intervals_dict)
    mask_copy = pd.DataFrame(mask_copy_dict) if mask_copy_dict is not None else None

    # Reconstruct dim_specs
    dim_specs = []
    for sp in dim_specs_list:
        dim_specs.append({
            "expr": sp["expr"],
            "breaks": sp["breaks"],
            "num_bins": sp["num_bins"],
            "bin_merge": sp.get("bin_merge"),
            "bin_map": sp.get("bin_map"),
        })

    # Reconstruct CDF list (already numpy arrays)
    cdf_list = cdf_list_data

    # Extract bin data for this chunk
    bin_indices, iter_starts, iter_chroms, flat_breaks, _, _ = \
        _extract_bin_data(dim_specs, chunk_intervals, iterator,
                          sample_bin_merge=sample_bin_merge)

    # Prepare mask_copy
    py_mask_copy = _df2pymisha(mask_copy) if mask_copy is not None else None

    # Call C++ backend (vector mode)
    return list(_pymisha.pm_gsynth_sample(
        cdf_list,
        flat_breaks,
        bin_indices,
        iter_starts,
        iter_chroms,
        _df2pymisha(chunk_intervals),
        py_mask_copy,
        "",   # output_path empty -> vector mode
        2,    # fmt_code = vector
        int(n_samples),
        chunk_seed,
        int(k),
        _resolve_iter_size(iterator),
        bool(preserve_n),
    ))


# ---------------------------------------------------------------------------
# gsynth_train
# ---------------------------------------------------------------------------

def gsynth_train(
    *dim_specs: dict[str, _Any],
    mask: _pd.DataFrame | None = None,
    intervals: _pd.DataFrame | None = None,
    iterator: int | None = None,
    pseudocount: float = 1.0,
    min_obs: int = 0,
    k: int = 5,
    allow_parallel: bool = True,
    num_cores: int | None = None,
    max_chunk_size: int | None = None,
) -> GsynthModel:
    """Train a stratified Markov model from genome sequences.

    Computes a k-th order Markov model optionally stratified by bins of one or
    more track expressions (e.g., GC content and CpG dinucleotide frequency).
    The resulting :class:`GsynthModel` can be used with :func:`gsynth_sample`
    to generate synthetic genomes that preserve the k-mer statistics of the
    original genome within each stratification bin.

    Both the forward-strand (k+1)-mer and its reverse complement are counted
    for every valid position, ensuring strand-symmetric transition
    probabilities.  Positions containing N bases are skipped and counted
    separately in the returned model's ``total_n`` attribute.

    When called with no dimension specifications, trains a single unstratified
    (0-D) model.

    For large genomes (total bases > threshold), intervals can be split into
    chunks and processed in parallel using multiple cores. Each chunk trains
    independently, and the resulting k-mer count arrays are merged before
    computing the final CDF. This matches the R ``misha`` parallel gsynth
    behavior.

    Parameters
    ----------
    *dim_specs : dict
        Each positional argument is a dict specifying a stratification
        dimension with the following keys:

        - ``"expr"`` (str): Track expression for this dimension (required).
        - ``"breaks"`` (array-like): Sorted bin boundaries (required).
          Length must be at least 2.
        - ``"bin_merge"`` (list of dict, optional): Merge specifications for
          sparse bins, in the format accepted by :func:`gsynth_bin_map`.
    mask : DataFrame, optional
        Intervals to exclude from training.  Regions in the mask do not
        contribute to k-mer counts but are tallied in ``total_masked``.
    intervals : DataFrame, optional
        Genomic intervals to train on.  If ``None``, uses all chromosomes.
    iterator : int, optional
        Iterator bin size for track extraction.  Determines the resolution
        at which track values are evaluated.
    pseudocount : float, default 1.0
        Pseudocount added to all k-mer counts to avoid zero probabilities
        in the CDF.
    min_obs : int, default 0
        Minimum number of (k+1)-mer observations required per bin.  Reserved
        for future use.
    k : int, default 5
        Markov order (context length).  Must be in ``[1, 10]``.  The model
        stores ``4^k`` context states; higher values capture longer-range
        dependencies but require more training data and memory.
    allow_parallel : bool, default True
        Whether to enable parallel chunking for large genomes.  When
        ``True`` and the total bases exceed *max_chunk_size*, intervals
        are split across multiple processes.  When ``False``, always
        runs single-threaded.
    num_cores : int, optional
        Number of worker processes.  If ``None``, defaults to
        ``multiprocessing.cpu_count()``.  Capped at the number of
        interval rows.
    max_chunk_size : int, optional
        Total-base threshold above which parallel processing triggers.
        Defaults to ``GSYNTH_MAX_CHUNK_SIZE`` (1 billion).

    Returns
    -------
    GsynthModel
        Trained model containing transition CDFs, dimension metadata, and
        training statistics.

    Raises
    ------
    TypeError
        If a dimension spec is not a dict.
    ValueError
        If a dimension spec is missing ``"expr"`` or ``"breaks"``, or if
        *breaks* has fewer than 2 elements, or if no data is extracted for
        a given expression, or if *k* is not in ``[1, 10]``.

    See Also
    --------
    gsynth_sample : Sample synthetic sequences from a trained model.
    gsynth_random : Generate random sequences without a model.
    gsynth_save : Persist a trained model to disk.
    gsynth_load : Restore a model from disk.
    gsynth_bin_map : Compute bin-merge mappings for sparse bins.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Train an unstratified (0-D) model over the whole genome:

    >>> model_0d = pm.gsynth_train()
    >>> model_0d.n_dims
    0
    """
    _checkroot()

    # Validate k
    if not isinstance(k, int) and isinstance(k, float) and k != int(k):
        raise ValueError(f"k must be an integer, got {k}")
    k = int(k)
    if k < 1 or k > 10:
        raise ValueError(f"Markov order k must be in [1, 10], got {k}")

    # Validate dimension specs
    parsed_specs = []
    for i, spec in enumerate(dim_specs):
        if not isinstance(spec, dict):
            raise TypeError(f"Dimension spec {i + 1} must be a dict")
        if "expr" not in spec:
            raise ValueError(f"Dimension spec {i + 1} must have an 'expr' element")
        if "breaks" not in spec:
            raise ValueError(f"Dimension spec {i + 1} must have a 'breaks' element")
        breaks = _numpy.asarray(spec["breaks"], dtype=float)
        if len(breaks) < 2:
            raise ValueError(
                f"Dimension spec {i + 1} breaks must have at least 2 elements"
            )
        parsed_specs.append({
            "expr": spec["expr"],
            "breaks": breaks.tolist(),
            "num_bins": len(breaks) - 1,
            "bin_merge": spec.get("bin_merge"),
            "bin_map": None,
        })

    # Extract bin data
    if intervals is None:
        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)

    # Compute per-dimension bin_map for storage
    for _i, spec in enumerate(parsed_specs):
        if spec["bin_merge"]:
            spec["bin_map"] = gsynth_bin_map(
                spec["breaks"], spec["bin_merge"]
            ).tolist()

    # Compute total_bins and dim_sizes early (needed for parallel path)
    dim_sizes = [sp["num_bins"] for sp in parsed_specs] if parsed_specs else [1]
    total_bins = 1
    for ds in dim_sizes:
        total_bins *= int(ds)

    # Check whether to parallelize
    do_parallel, effective_cores = _should_parallelize(
        intervals, allow_parallel, num_cores, max_chunk_size
    )

    if do_parallel:
        _logger.info(
            "Large genome detected (%s bases). "
            "Processing %d chunks across %d cores...",
            f"{_compute_total_bases(intervals):,}",
            len(intervals),
            effective_cores,
        )

        chunks = _chunk_intervals(intervals, effective_cores)
        mask_dict = mask.to_dict(orient="list") if mask is not None else None
        parsed_specs_data = [
            dict(sp.items()) for sp in parsed_specs
        ]

        worker_args = [
            (chunk.to_dict(orient="list"), parsed_specs_data,
             mask_dict, iterator, pseudocount, total_bins,
             parsed_specs_data, k)
            for chunk in chunks
        ]

        ctx = _multiprocessing.get_context("fork")
        with ctx.Pool(processes=effective_cores) as pool:
            chunk_results = pool.map(_worker_train_chunk, worker_args)

        # Merge results from all chunks
        result = _merge_train_results(chunk_results, total_bins, pseudocount, k=k)

        return GsynthModel(
            k=k,
            n_dims=len(parsed_specs),
            dim_sizes=dim_sizes,
            dim_specs=parsed_specs,
            total_bins=total_bins,
            model_data={
                "counts": result["counts"],
                "cdf": result["cdf"],
            },
            total_kmers=int(result["total_kmers"]),
            per_bin_kmers=result["per_bin_kmers"],
            total_masked=int(result["total_masked"]),
            total_n=int(result["total_n"]),
            pseudocount=pseudocount,
            min_obs=min_obs,
            iterator=iterator,
        )

    # --- Single-process path (original logic) ---

    bin_indices, iter_starts, iter_chroms, flat_breaks, bin_map, dim_sizes = \
        _extract_bin_data(parsed_specs, intervals, iterator)

    # Call C++ backend
    py_mask = _df2pymisha(mask) if mask is not None else None
    result = _pymisha.pm_gsynth_train(
        _df2pymisha(intervals),
        bin_indices,
        iter_starts,
        iter_chroms,
        flat_breaks,
        bin_map,
        py_mask,
        float(pseudocount),
        int(k),
        _resolve_iter_size(iterator),
    )

    # Build model
    total_bins = 1
    for dim_size in dim_sizes:
        total_bins *= int(dim_size)

    return GsynthModel(
        k=k,
        n_dims=len(parsed_specs),
        dim_sizes=dim_sizes,
        dim_specs=parsed_specs,
        total_bins=total_bins,
        model_data={
            "counts": result["counts"],
            "cdf": result["cdf"],
        },
        total_kmers=int(result["total_kmers"]),
        per_bin_kmers=result["per_bin_kmers"],
        total_masked=int(result["total_masked"]),
        total_n=int(result["total_n"]),
        pseudocount=pseudocount,
        min_obs=min_obs,
        iterator=iterator,
    )



# ---------------------------------------------------------------------------
# gsynth_sample
# ---------------------------------------------------------------------------

def gsynth_sample(
    model: GsynthModel,
    output: str | None = None,
    *,
    output_format: str = "fasta",
    intervals: _pd.DataFrame | None = None,
    iterator: int | None = None,
    mask_copy: _pd.DataFrame | None = None,
    preserve_n: bool = True,
    n_samples: int = 1,
    seed: int | None = None,
    bin_merge: list[_Any] | None = None,
    allow_parallel: bool = True,
    num_cores: int | None = None,
    max_chunk_size: int | None = None,
) -> list[str] | None:
    """Sample synthetic genome sequences from a trained model.

    Generates a synthetic genome by sampling from a trained stratified
    Markov model.  For each genomic position the sampler looks up the
    current k-mer context and the position's stratification bin, then draws
    the next nucleotide from the corresponding CDF.  The result preserves
    the k-mer statistics of the original genome within each bin.

    When the sampler needs to initialise the first k-mer context and
    encounters regions with only N bases, it falls back to uniform random
    base selection until a valid context is established.

    For large genomes (total bases > threshold), intervals can be split into
    chunks and processed in parallel using multiple cores. Each chunk samples
    independently and the resulting sequences are concatenated. For file
    output modes (``"fasta"`` or ``"seq"``), the parallel path first samples
    to in-memory vectors and then writes the combined result.

    Parameters
    ----------
    model : GsynthModel
        Trained Markov model from :func:`gsynth_train`.
    output : str, optional
        Output file path.  If ``None``, sequences are returned in memory
        (equivalent to ``output_format="vector"``).
    output_format : {"fasta", "seq", "vector"}, default "fasta"
        Output format:

        - ``"fasta"`` -- FASTA text format.
        - ``"seq"`` -- misha binary ``.seq`` format.
        - ``"vector"`` -- return sequences as a Python list of strings
          (does not write to file).
    intervals : DataFrame, optional
        Genomic intervals to synthesise.  If ``None``, uses all chromosomes.
    iterator : int, optional
        Iterator bin size for track extraction during bin-index computation.
    mask_copy : DataFrame, optional
        Intervals where the original reference sequence is preserved
        verbatim instead of being sampled.  Useful for keeping repetitive
        or regulatory regions intact.  Should be non-overlapping and sorted
        by start position within each chromosome.
    preserve_n : bool, default True
        When ``True`` (default), positions whose original reference is
        ``N`` (or lowercase ``n``) are written to the output verbatim
        rather than filled with a random ACGT base. Case is preserved.
        ``mask_copy`` regions take precedence: inside a ``mask_copy``
        interval the original byte is copied regardless. Set to
        ``False`` to recover the pre-0.1.34 behaviour of fabricating
        ACGT at every position.
    n_samples : int, default 1
        Number of independent samples to generate per interval.  When
        ``n_samples > 1`` and ``output_format="fasta"``, headers include a
        ``_sampleN`` suffix.  When ``output_format="vector"``, the returned
        list has length ``n_intervals * n_samples``.
    seed : int, optional
        Random seed for reproducibility.  If ``None``, uses the current
        random state.
    bin_merge : list, optional
        Sampling-time bin merge overrides, one element per model dimension.
        Each element is either ``None`` (use the training-time bin_merge)
        or a list of merge-range dicts as accepted by
        :func:`gsynth_bin_map`.  This allows redirecting sparse bins to
        better-populated ones at sampling time without retraining the
        model.
    allow_parallel : bool, default True
        Whether to enable parallel chunking for large genomes.  When
        ``True`` and the total bases exceed *max_chunk_size*, intervals
        are split across multiple processes.  When ``False``, always
        runs single-threaded.
    num_cores : int, optional
        Number of worker processes.  If ``None``, defaults to
        ``multiprocessing.cpu_count()``.  Capped at the number of
        interval rows.
    max_chunk_size : int, optional
        Total-base threshold above which parallel processing triggers.
        Defaults to ``GSYNTH_MAX_CHUNK_SIZE`` (1 billion).

    Returns
    -------
    list of str or None
        List of nucleotide strings when *output* is ``None`` or
        *output_format* is ``"vector"``.  ``None`` otherwise (output is
        written to file).

    Raises
    ------
    TypeError
        If *model* is not a :class:`GsynthModel`.

    See Also
    --------
    gsynth_train : Train the model consumed by this function.
    gsynth_random : Generate random sequences without a model.
    gsynth_save : Persist a model for later sampling.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> model = pm.gsynth_train()
    >>> seqs = pm.gsynth_sample(
    ...     model,
    ...     intervals=pm.gintervals(["1"], [0], [1000]),
    ...     seed=42,
    ... )
    >>> len(seqs[0])
    1000
    """
    _checkroot()

    if not isinstance(model, GsynthModel):
        raise TypeError("model must be a GsynthModel")

    if bin_merge is not None and (
        not isinstance(bin_merge, list) or len(bin_merge) != model.n_dims
    ):
        raise ValueError(
            f"bin_merge must be a list with {model.n_dims} elements "
            "(one per dimension)"
        )

    model_k = model.k

    # Default iterator to the value used during training (R parity:
    # model$iterator). Without this, the C backend would error or use the
    # INT64_MAX sentinel, and bin extraction in _extract_bin_data would emit
    # rows at unexpected positions.
    if iterator is None:
        iterator = model.iterator

    if intervals is None:
        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)

    # Check whether to parallelize
    do_parallel, effective_cores = _should_parallelize(
        intervals, allow_parallel, num_cores, max_chunk_size
    )

    if do_parallel:
        _logger.info(
            "Large genome detected (%s bases). "
            "Sampling %d chunks across %d cores...",
            f"{_compute_total_bases(intervals):,}",
            len(intervals),
            effective_cores,
        )

        chunks = _chunk_intervals(intervals, effective_cores)
        chunk_seeds = _generate_chunk_seeds(seed, len(chunks))

        mask_copy_dict = (mask_copy.to_dict(orient="list")
                          if mask_copy is not None else None)
        dim_specs_list = [
            dict(sp.items()) for sp in model.dim_specs
        ]
        cdf_list = model.model_data["cdf"]

        worker_args = [
            (chunk.to_dict(orient="list"), dim_specs_list,
             cdf_list, iterator, mask_copy_dict, n_samples,
             chunk_seeds[i], bin_merge, model_k, preserve_n)
            for i, chunk in enumerate(chunks)
        ]

        ctx = _multiprocessing.get_context("fork")
        with ctx.Pool(processes=effective_cores) as pool:
            chunk_results = pool.map(_worker_sample_chunk, worker_args)

        # Concatenate all sequence lists
        all_seqs = []
        for cr in chunk_results:
            if cr is not None:
                all_seqs.extend(cr)

        # Handle output mode
        if output is None:
            return all_seqs

        # Write to file
        output_path = str(output)
        parent = _os.path.dirname(output_path)
        if parent:
            _os.makedirs(parent, exist_ok=True)

        if output_format == "fasta":
            _write_fasta(output_path, intervals, all_seqs, n_samples)
        else:
            # For seq/binary format, fall back to single-process
            # (binary format requires C++ writer)
            pass  # fall through to single-process path below

        if output_format == "fasta":
            return None

        # If we fell through for non-fasta file output,
        # re-run single-process below
        do_parallel = False

    # Determine output format code
    fmt_map = {"seq": 0, "fasta": 1, "vector": 2}
    if output is None:
        fmt_code = 2  # vector mode
        output_path = ""
    else:
        fmt_code = fmt_map.get(output_format, 1)
        output_path = str(output)
        # Ensure parent directory exists
        parent = _os.path.dirname(output_path)
        if parent:
            _os.makedirs(parent, exist_ok=True)

    # Extract bin data for model dimensions
    bin_indices, iter_starts, iter_chroms, flat_breaks, _, _ = \
        _extract_bin_data(model.dim_specs, intervals, iterator,
                          sample_bin_merge=bin_merge)

    # Get CDF list from model
    cdf_list = model.model_data["cdf"]

    # Prepare mask_copy
    py_mask_copy = _df2pymisha(mask_copy) if mask_copy is not None else None

    # Call C++ backend
    _raw = _pymisha.pm_gsynth_sample(
        cdf_list,
        flat_breaks,
        bin_indices,
        iter_starts,
        iter_chroms,
        _df2pymisha(intervals),
        py_mask_copy,
        output_path,
        fmt_code,
        int(n_samples),
        seed,
        int(model_k),
        _resolve_iter_size(iterator),
        bool(preserve_n),
    )
    return list(_raw) if _raw is not None else None


def _write_fasta(output_path: str, intervals: _pd.DataFrame, sequences: list[str], n_samples: int) -> None:
    """Write sequences to a FASTA file.

    Parameters
    ----------
    output_path : str
        Path to the output file.
    intervals : DataFrame
        Genomic intervals.
    sequences : list of str
        Sequences to write (length = n_intervals * n_samples).
    n_samples : int
        Number of samples per interval.
    """
    with open(output_path, "w") as f:
        seq_idx = 0
        for i in range(len(intervals)):
            row = intervals.iloc[i]
            chrom = str(row["chrom"])
            start = int(row["start"])
            end = int(row["end"])
            for s in range(n_samples):
                header = f">{chrom}:{start}-{end}_sample{s + 1}" if n_samples > 1 else f">{chrom}:{start}-{end}"
                f.write(header + "\n")
                if seq_idx < len(sequences):
                    f.write(sequences[seq_idx] + "\n")
                seq_idx += 1



# ---------------------------------------------------------------------------
# gsynth_random
# ---------------------------------------------------------------------------

def gsynth_random(
    *,
    intervals: _pd.DataFrame | None = None,
    nuc_probs: dict[str, float] | None = None,
    output: str | None = None,
    output_format: str = "fasta",
    mask_copy: _pd.DataFrame | None = None,
    preserve_n: bool = True,
    n_samples: int = 1,
    seed: int | None = None,
) -> list[str] | None:
    """Generate random genome sequences without a trained model.

    Produces random DNA sequences where each nucleotide is sampled
    independently according to the specified probabilities.  Unlike
    :func:`gsynth_sample`, no Markov context is used -- consecutive bases
    are statistically independent.  This is useful for generating baseline
    random sequences or sequences with a specific GC content.

    Parameters
    ----------
    intervals : DataFrame, optional
        Genomic intervals to generate.  If ``None``, uses all chromosomes.
    nuc_probs : dict, optional
        Nucleotide probabilities keyed by ``'A'``, ``'C'``, ``'G'``,
        ``'T'``.  Values are automatically normalised to sum to 1.
        Default is uniform (0.25 each).
    output : str, optional
        Output file path.  If ``None``, sequences are returned in memory.
    output_format : {"fasta", "seq", "vector"}, default "fasta"
        Output format:

        - ``"fasta"`` -- FASTA text format.
        - ``"seq"`` -- misha binary ``.seq`` format.
        - ``"vector"`` -- return sequences as a Python list of strings.
    mask_copy : DataFrame, optional
        Intervals where the original reference sequence is preserved
        instead of being randomly generated.
    preserve_n : bool, default True
        When ``True`` (default), positions whose original reference is
        ``N`` (or lowercase ``n``) are written to the output verbatim
        rather than filled with a random ACGT base. ``mask_copy``
        intervals take precedence. Set to ``False`` to recover the
        pre-0.1.34 behaviour.
    n_samples : int, default 1
        Number of independent samples to generate per interval.
    seed : int, optional
        Random seed for reproducibility.  If ``None``, uses the current
        random state.

    Returns
    -------
    list of str or None
        List of nucleotide strings when *output* is ``None`` or
        *output_format* is ``"vector"``.  ``None`` otherwise (output is
        written to file).

    See Also
    --------
    gsynth_sample : Sample from a trained Markov model.
    gsynth_train : Train a Markov model for context-dependent sampling.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Uniform random sequence:

    >>> seqs = pm.gsynth_random(
    ...     intervals=pm.gintervals(["1"], [0], [1000]),
    ...     seed=42,
    ... )
    >>> len(seqs[0])
    1000
    """
    _checkroot()

    if intervals is None:
        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)

    # gsynth_random always uses k=5 (context doesn't matter since all
    # contexts share the same distribution, but the CDF matrix size and
    # the C++ context window must agree).
    random_k = 5
    num_kmers = 4 ** random_k  # 1024

    # Build uniform/custom CDF for a single bin
    if nuc_probs is None:
        probs = _numpy.array([0.25, 0.25, 0.25, 0.25])
    else:
        probs = _numpy.array([
            nuc_probs.get("A", 0.25),
            nuc_probs.get("C", 0.25),
            nuc_probs.get("G", 0.25),
            nuc_probs.get("T", 0.25),
        ], dtype=float)
        probs = probs / probs.sum()

    cdf = _numpy.cumsum(probs)
    cdf[-1] = 1.0  # Ensure exact 1.0

    # Create CDF matrix for all contexts (same CDF for all)
    cdf_mat = _numpy.tile(cdf, (num_kmers, 1))  # num_kmers x 4
    cdf_list = [cdf_mat]  # Single bin

    assert isinstance(intervals, _pd.DataFrame)
    # Single bin: all positions map to bin 0
    n_positions = len(intervals)
    bin_indices = _numpy.zeros(n_positions, dtype=_numpy.int32)
    iter_starts = intervals["start"].to_numpy(dtype=_numpy.int64)

    # Convert chroms to IDs
    all_chroms = gintervals_all()
    chrom_to_id = {name: i for i, name in enumerate(all_chroms["chrom"])}
    iter_chroms = _numpy.array(
        [chrom_to_id.get(str(c), -1) for c in intervals["chrom"]],
        dtype=_numpy.int32
    )

    flat_breaks = [0.0, 1.0]  # Single bin

    # Output setup
    fmt_map = {"seq": 0, "fasta": 1, "vector": 2}
    if output is None:
        fmt_code = 2
        output_path = ""
    else:
        fmt_code = fmt_map.get(output_format, 1)
        output_path = str(output)
        parent = _os.path.dirname(output_path)
        if parent:
            _os.makedirs(parent, exist_ok=True)

    py_mask_copy = _df2pymisha(mask_copy) if mask_copy is not None else None

    # gsynth_random uses a single bin (no stratification), so no iterator
    # constraint applies; the INT64_MAX sentinel disables the per-bin extent
    # check in the C backend.
    _raw2 = _pymisha.pm_gsynth_sample(
        cdf_list,
        flat_breaks,
        bin_indices,
        iter_starts,
        iter_chroms,
        _df2pymisha(intervals),
        py_mask_copy,
        output_path,
        fmt_code,
        int(n_samples),
        seed,
        int(random_k),
        _ITER_SIZE_NO_CONSTRAINT,
        bool(preserve_n),
    )
    return list(_raw2) if _raw2 is not None else None



# ---------------------------------------------------------------------------
# gsynth_replace_kmer
# ---------------------------------------------------------------------------

def gsynth_replace_kmer(
    target: str,
    replacement: str,
    *,
    intervals: _pd.DataFrame | None = None,
    output: str | None = None,
    output_format: str = "fasta",
    check_composition: bool = True,
) -> list[str] | None:
    """Iteratively replace a k-mer in genome sequences.

    Scans each sequence and replaces every occurrence of *target* with
    *replacement*.  If a replacement creates a new instance of *target*
    (e.g., replacing ``"CG"`` with ``"GC"`` in the sequence ``"CCG"``
    produces ``"CGC"``), the new instance is also replaced.  The scan
    repeats until the sequence is completely free of *target*.

    When *target* and *replacement* are permutations of each other (e.g.,
    ``"CG"`` and ``"GC"``), the operation acts as a local "bubble sort" of
    nucleotides, preserving the total base counts and GC content of the
    genome.

    Parameters
    ----------
    target : str
        K-mer to remove.  Case-insensitive (converted to uppercase
        internally).
    replacement : str
        Replacement sequence.  Must be the same length as *target*.
    intervals : DataFrame, optional
        Genomic intervals to process.  If ``None``, uses all chromosomes.
    output : str, optional
        Output file path.  If ``None``, sequences are returned in memory.
    output_format : {"fasta", "seq", "vector"}, default "fasta"
        Output format:

        - ``"fasta"`` -- FASTA text format.
        - ``"seq"`` -- misha binary ``.seq`` format.
        - ``"vector"`` -- return sequences as a Python list of strings.
    check_composition : bool, default True
        If ``True``, verify that *target* and *replacement* contain the
        same nucleotides (i.e., are anagrams).  Set to ``False`` to allow
        replacements that change base composition.

    Returns
    -------
    list of str or None
        List of modified nucleotide strings when *output* is ``None`` or
        *output_format* is ``"vector"``.  ``None`` otherwise (output is
        written to file).

    Raises
    ------
    ValueError
        If *target* or *replacement* is empty, if they differ in length,
        or if ``check_composition=True`` and their nucleotide compositions
        differ.

    See Also
    --------
    gsynth_sample : Markov-model-based genome synthesis.
    gsynth_random : Independent random nucleotide generation.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Remove all CpG dinucleotides while preserving GC content:

    >>> seqs = pm.gsynth_replace_kmer(
    ...     "CG", "GC",
    ...     intervals=pm.gintervals(["1"], [0], [1000]),
    ... )
    >>> "CG" not in seqs[0]
    True
    """
    _checkroot()

    if not target or not replacement:
        raise ValueError("target and replacement cannot be empty")
    if len(target) != len(replacement):
        raise ValueError("target and replacement must have the same length")

    target = target.upper()
    replacement = replacement.upper()

    if check_composition and sorted(target) != sorted(replacement):
        raise ValueError(
            "target and replacement must have the same nucleotide composition "
            "when check_composition=True"
        )

    if intervals is None:
        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)

    # Output setup
    fmt_map = {"seq": 0, "fasta": 1, "vector": 2}
    if output is None:
        fmt_code = 2
        output_path = ""
    else:
        fmt_code = fmt_map.get(output_format, 1)
        output_path = str(output)
        parent = _os.path.dirname(output_path)
        if parent:
            _os.makedirs(parent, exist_ok=True)

    _raw3 = _pymisha.pm_gsynth_replace_kmer(
        target,
        replacement,
        _df2pymisha(intervals),
        output_path,
        fmt_code,
    )
    return list(_raw3) if _raw3 is not None else None



# ---------------------------------------------------------------------------
# gsynth_save / gsynth_load
# ---------------------------------------------------------------------------

def gsynth_save(model: GsynthModel, path: str, *, compress: bool = False) -> None:
    """Save a trained model to disk in .gsm format.

    Serialises a :class:`GsynthModel` to a cross-platform ``.gsm`` directory
    (or ZIP archive when *compress=True*) containing YAML metadata and raw
    binary arrays.  The file can later be restored with :func:`gsynth_load`.

    The ``.gsm`` format stores counts and CDFs as raw float64 arrays in
    row-major (C) order, making them readable from both Python and R without
    any language-specific serialisation quirks.

    Parameters
    ----------
    model : GsynthModel
        Trained model to save.
    path : str
        Destination path.  When *compress* is ``False`` (default), a directory
        is created at this path.  When ``True``, a ZIP archive is written.
        Parent directories are **not** created automatically.
    compress : bool, default False
        If ``True``, write a ZIP archive instead of a directory.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *model* is not a :class:`GsynthModel`.

    See Also
    --------
    gsynth_load : Restore a model saved by this function.
    gsynth_train : Create a model.
    gsynth_convert : Convert legacy pickle models to ``.gsm`` format.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> model = pm.gsynth_train()
    >>> import tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "model.gsm")
    >>> pm.gsynth_save(model, path)
    """
    if not isinstance(model, GsynthModel):
        raise TypeError("model must be a GsynthModel")

    model_k = model.k
    num_kmers = 4 ** model_k

    # Build metadata dict
    total_bins = model.total_bins
    per_bin_kmers = model.per_bin_kmers
    per_bin_kmers_list = [int(x) for x in per_bin_kmers] if per_bin_kmers is not None else []

    dim_specs_out = []
    for spec in model.dim_specs:
        ds = {
            "expr": str(spec.get("expr", "")),
            "breaks": [float(b) for b in spec.get("breaks", [])],
            "num_bins": int(spec.get("num_bins", 0)),
            "bin_map": (
                [int(x) for x in spec["bin_map"]]
                if spec.get("bin_map") is not None
                else None
            ),
        }
        dim_specs_out.append(ds)

    # Use version 2 when k != 5 (non-default); version 1 for backward compat
    file_version = 2 if model_k != 5 else 1

    metadata = {
        "format": "gsynth_model",
        "version": file_version,
        "markov_order": int(model_k),
        "n_dims": int(model.n_dims),
        "dim_sizes": [int(x) for x in model.dim_sizes],
        "total_bins": int(total_bins),
        "pseudocount": float(model.pseudocount),
        "min_obs": int(model.min_obs),
        "iterator": (int(model.iterator) if model.iterator is not None else None),
        "total_kmers": int(model.total_kmers),
        "total_masked": int(model.total_masked),
        "total_n": int(model.total_n),
        "per_bin_kmers": per_bin_kmers_list,
        "dim_specs": dim_specs_out,
        "data": {
            "counts": {
                "dtype": "float64",
                "shape": [int(total_bins), int(num_kmers), 4],
                "order": "C",
                "file": "counts.bin",
            },
            "cdf": {
                "dtype": "float64",
                "shape": [int(total_bins), int(num_kmers), 4],
                "order": "C",
                "file": "cdf.bin",
            },
        },
    }

    # Stack arrays into contiguous float64
    counts_arr = _numpy.stack(model.model_data["counts"]).astype(_numpy.float64)
    cdf_arr = _numpy.stack(model.model_data["cdf"]).astype(_numpy.float64)

    metadata_bytes = _yaml.dump(metadata, default_flow_style=False, sort_keys=False).encode("utf-8")
    counts_bytes = counts_arr.tobytes()  # C order by default
    cdf_bytes = cdf_arr.tobytes()

    if compress:
        with _zipfile.ZipFile(path, "w", compression=_zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.yaml", metadata_bytes)
            zf.writestr("counts.bin", counts_bytes)
            zf.writestr("cdf.bin", cdf_bytes)
    else:
        _os.makedirs(path, exist_ok=True)
        with open(_os.path.join(path, "metadata.yaml"), "wb") as f:
            f.write(metadata_bytes)
        with open(_os.path.join(path, "counts.bin"), "wb") as f:
            f.write(counts_bytes)
        with open(_os.path.join(path, "cdf.bin"), "wb") as f:
            f.write(cdf_bytes)


def _load_legacy_pickle(path: str) -> GsynthModel:
    """Load a legacy pickle-format GsynthModel.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    GsynthModel
        The deserialised model.

    Raises
    ------
    TypeError
        If the deserialised object is not a :class:`GsynthModel`.
    """
    with open(path, "rb") as f:
        model = restricted_load(
            f, extra_allowed_globals={("pymisha.gsynth", "GsynthModel")}
        )
    if not isinstance(model, GsynthModel):
        raise TypeError("Loaded object is not a GsynthModel")
    # Backfill min_obs for models created before it was added
    if not hasattr(model, "min_obs"):
        model.min_obs = 0
    # Backfill k for models created before variable k was added
    if not hasattr(model, "k"):
        model.k = 5
    # Backfill iterator for models saved before it was tracked.
    if not hasattr(model, "iterator"):
        model.iterator = None
    return model


def _load_gsm_from_meta_and_files(metadata: dict[str, _Any], read_file: _Any) -> GsynthModel:
    """Build a GsynthModel from parsed metadata and a file-reader callable.

    Parameters
    ----------
    metadata : dict
        Parsed metadata.yaml content.
    read_file : callable
        ``read_file(name)`` returns bytes for the named file
        (``"counts.bin"`` or ``"cdf.bin"``).

    Returns
    -------
    GsynthModel
    """
    fmt = metadata.get("format")
    version = metadata.get("version")
    if fmt != "gsynth_model":
        raise ValueError(f"Unknown format: {fmt!r}")
    if version not in (1, 2):
        raise ValueError(f"Unsupported version: {version}")

    # Read k from metadata (default 5 for version 1 / legacy)
    model_k = int(metadata.get("markov_order", 5))
    if model_k < 1 or model_k > 10:
        raise ValueError(f"Invalid markov_order: {model_k}")
    num_kmers = 4 ** model_k

    total_bins = int(metadata["total_bins"])
    shape = (total_bins, num_kmers, 4)

    counts_raw = _numpy.frombuffer(read_file("counts.bin"), dtype=_numpy.float64).reshape(shape)
    cdf_raw = _numpy.frombuffer(read_file("cdf.bin"), dtype=_numpy.float64).reshape(shape)

    # Split into per-bin arrays; counts back to uint64
    counts_list = [counts_raw[i].astype(_numpy.uint64) for i in range(total_bins)]
    cdf_list = [cdf_raw[i].copy() for i in range(total_bins)]

    # Reconstruct dim_specs
    dim_specs = []
    for ds in metadata.get("dim_specs", []):
        spec = {
            "expr": ds["expr"],
            "breaks": ds["breaks"],
            "num_bins": ds["num_bins"],
            "bin_map": ds.get("bin_map"),
        }
        dim_specs.append(spec)

    per_bin_kmers_raw = metadata.get("per_bin_kmers", [])
    if per_bin_kmers_raw is not None and not isinstance(per_bin_kmers_raw, list):
        per_bin_kmers_raw = [per_bin_kmers_raw]  # YAML scalar -> list
    per_bin_kmers = (
        _numpy.atleast_1d(_numpy.array(per_bin_kmers_raw, dtype=_numpy.int64))
        if per_bin_kmers_raw
        else None
    )

    return GsynthModel(
        k=model_k,
        n_dims=int(metadata.get("n_dims", 0)),
        dim_sizes=[int(x) for x in metadata.get("dim_sizes", [])],
        dim_specs=dim_specs,
        total_bins=total_bins,
        model_data={"counts": counts_list, "cdf": cdf_list},
        total_kmers=int(metadata.get("total_kmers", 0)),
        per_bin_kmers=per_bin_kmers,
        total_masked=int(metadata.get("total_masked", 0)),
        total_n=int(metadata.get("total_n", 0)),
        pseudocount=float(metadata.get("pseudocount", 1.0)),
        min_obs=int(metadata.get("min_obs", 0)),
        iterator=(
            int(metadata["iterator"])
            if metadata.get("iterator") is not None
            else None
        ),
    )


def gsynth_load(path: str) -> GsynthModel:
    """Load a trained model from disk.

    Auto-detects the format: ``.gsm`` directory, ``.gsm`` ZIP archive, or
    legacy pickle.

    Parameters
    ----------
    path : str
        Path to the saved model.  Can be a ``.gsm`` directory, a ZIP file,
        or a legacy pickle file.

    Returns
    -------
    GsynthModel
        The deserialised model, ready for use with :func:`gsynth_sample`.

    Raises
    ------
    TypeError
        If the deserialised object is not a :class:`GsynthModel`.
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the format or version is unrecognised.

    See Also
    --------
    gsynth_save : Save a model to disk.
    gsynth_train : Create a new model from scratch.
    gsynth_sample : Sample sequences from the loaded model.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> model = pm.gsynth_train()
    >>> import tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "model.gsm")
    >>> pm.gsynth_save(model, path)
    >>> restored = pm.gsynth_load(path)
    >>> restored.total_bins == model.total_bins
    True
    """
    # Directory-based .gsm
    if _os.path.isdir(path):
        meta_path = _os.path.join(path, "metadata.yaml")
        if not _os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.yaml not found in {path}")
        with open(meta_path) as f:
            metadata = _yaml.safe_load(f)

        def read_file(name: str) -> bytes:
            with open(_os.path.join(path, name), "rb") as fh:
                return fh.read()

        return _load_gsm_from_meta_and_files(metadata, read_file)

    # File-based: try ZIP first, then legacy pickle
    if _os.path.isfile(path):
        if _zipfile.is_zipfile(path):
            with _zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if "metadata.yaml" in names:
                    metadata = _yaml.safe_load(zf.read("metadata.yaml"))
                    return _load_gsm_from_meta_and_files(metadata, zf.read)
        # Fall back to legacy pickle
        return _load_legacy_pickle(path)

    raise FileNotFoundError(f"Path not found: {path}")


def gsynth_convert(input_path: str, output_path: str, *, compress: bool = False) -> None:
    """Convert a legacy pickle model to ``.gsm`` format.

    Reads a model from *input_path* (any supported format, including legacy
    pickle) and writes it to *output_path* in the cross-platform ``.gsm``
    format.

    Parameters
    ----------
    input_path : str
        Path to the source model (pickle, ``.gsm`` directory, or ZIP).
    output_path : str
        Destination path for the ``.gsm`` output.
    compress : bool, default False
        If ``True``, write a ZIP archive instead of a directory.

    Returns
    -------
    None

    See Also
    --------
    gsynth_save : Save a model in ``.gsm`` format.
    gsynth_load : Load a model from any supported format.
    """
    model = gsynth_load(input_path)
    gsynth_save(model, output_path, compress=compress)
