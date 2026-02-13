"""Type stubs for the _pymisha C++ extension module.

This module provides the low-level C++ bindings for PyMisha genomics
operations.  All functions use METH_VARARGS and accept positional
arguments only.

The ``intervals`` parameters typically receive the output of
``_df2pymisha(df)`` -- a list whose first element is a numpy object
array of column names followed by one numpy array per column.

Return types annotated as ``dict[str, numpy.ndarray]`` represent the
internal "PMDataFrame" wire format that is converted to a pandas
DataFrame by ``_pymisha2df()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# -- Exception class exported by the module -----------------------------------

class error(Exception): ...

# -- Database Management ------------------------------------------------------

def pm_dbinit(
    groot: str,
    uroot: str = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> None:
    """Initialize database connection.

    C++ signature: ``PyArg_ParseTuple(args, "s|sO", &groot, &uroot, &py_config)``
    """

def pm_dbreload() -> None:
    """Reload database (re-scan tracks on disk)."""

def pm_dbunload() -> None:
    """Unload database and release resources."""

def pm_dbsetdatasets(datasets: list[str], /) -> None:
    """Set loaded dataset roots.

    C++ signature: ``PyArg_ParseTuple(args, "O", &py_datasets)``
    """

def pm_dbgetdatasets() -> list[str]:
    """Get loaded dataset roots."""

# -- Track Data Extraction & Analysis -----------------------------------------

def pm_extract(
    exprs: str | list[str],
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Extract track values over intervals.

    Returns a PMDataFrame dict (chrom, start, end, <expr_cols>, intervalID)
    or None when the iterator produces no intervals.

    C++ signature: ``PyArg_ParseTuple(args, "OO|OO", ...)``
    """

def pm_screen(
    expr: str,
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Screen intervals by logical expression.

    Returns a PMDataFrame dict (chrom, start, end) or None.

    C++ signature: ``PyArg_ParseTuple(args, "OO|OO", ...)``
    """

def pm_summary(
    expr: str,
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, float]:
    """Summarize expression values over intervals.

    Returns dict with keys: 'Total intervals', 'NaN intervals', 'Min',
    'Max', 'Sum', 'Mean', 'Std dev'.

    C++ signature: ``PyArg_ParseTuple(args, "OO|OO", ...)``
    """

def pm_quantiles(
    expr: str,
    percentiles: Sequence[float],
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> list[float]:
    """Compute expression quantiles.

    Returns a list of quantile values.

    C++ signature: ``PyArg_ParseTuple(args, "OOO|OO", ...)``
    """

def pm_intervals_summary(
    expr: str,
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Summarize expression values per interval.

    Returns a PMDataFrame dict (chrom, start, end, Total intervals,
    NaN intervals, Min, Max, Sum, Mean, Std dev) or None.

    C++ signature: ``PyArg_ParseTuple(args, "OO|OO", ...)``
    """

def pm_intervals_quantiles(
    expr: str,
    percentiles: Sequence[float],
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Compute expression quantiles per interval.

    Returns a PMDataFrame dict (chrom, start, end, <percentile_cols>)
    or None.

    C++ signature: ``PyArg_ParseTuple(args, "OOO|OO", ...)``
    """

# -- Track Metadata & Management ----------------------------------------------

def pm_track_names() -> list[str]:
    """Get all track names in the database."""

def pm_track_info(track: str, /) -> dict[str, Any]:
    """Get track information (type, dimensions, bin_size, etc.).

    C++ signature: ``PyArg_ParseTuple(args, "s", &track_name)``
    """

def pm_track_path(track: str, /) -> str | None:
    """Get track directory path on disk.  Returns None if track does not exist.

    C++ signature: ``PyArg_ParseTuple(args, "s", &track_name)``
    """

def pm_track_dataset(track: str, /) -> str | None:
    """Get dataset root for a track.  Returns None if track does not exist.

    C++ signature: ``PyArg_ParseTuple(args, "O", &py_track)``
    """

def pm_normalize_chroms(chroms: Sequence[str | int], /) -> list[str]:
    """Normalize chromosome names to canonical DB names.

    C++ signature: ``PyArg_ParseTuple(args, "O", &py_chroms)``
    """

# -- Track Creation & Modification --------------------------------------------

def pm_track_convert_to_indexed(
    track: str, remove_old: bool = ..., /
) -> None:
    """Convert track to indexed format (track.idx + track.dat).

    C++ signature: ``PyArg_ParseTuple(args, "s|p", &track, &remove_old)``
    """

def pm_track_create_empty_indexed(track: str, /) -> None:
    """Create empty indexed track (empty track.idx + track.dat).

    C++ signature: ``PyArg_ParseTuple(args, "s", &track)``
    """

def pm_track_create_sparse(track: str, data: Any, /) -> None:
    """Create sparse track from intervals+values.

    ``data`` is the PMDataFrame wire format from ``_df2pymisha()``.

    C++ signature: ``PyArg_ParseTuple(args, "sO", &track, &py_data)``
    """

def pm_track_create_dense(
    track: str, data: Any, binsize: int, defval: float, /
) -> None:
    """Create dense track from intervals+values.

    C++ signature: ``PyArg_ParseTuple(args, "sOId", &track, &py_data, &binsize, &defval)``
    """

def pm_track_create_expr(
    track: str,
    expr: str,
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> None:
    """Create track from expression in streaming mode.

    C++ signature: ``PyArg_ParseTuple(args, "ssO|OO", ...)``
    """

def pm_modify(
    track: str,
    expr: str,
    intervals: Any,
    iterator_policy: int,
    /,
) -> None:
    """Modify dense track values in-place.

    C++ signature: ``PyArg_ParseTuple(args, "ssOl", &track, &expr, &py_intervals, &iterator_policy)``
    """

def pm_smooth(
    track: str,
    expr: str,
    intervals: Any,
    iterator_policy: int,
    winsize: float,
    weight_thr: float,
    smooth_nans: int,
    alg: str,
    /,
) -> None:
    """Create smoothed track from expression.

    C++ signature: ``PyArg_ParseTuple(args, "ssOlddis", ...)``
    """

# -- Intervals & Iteration ----------------------------------------------------

def pm_intervals_all() -> dict[str, np.ndarray]:
    """Get all genome intervals (one per chromosome).

    Returns PMDataFrame dict (chrom, start, end).
    """

def pm_iterate(
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray]:
    """Iterate intervals with iterator policy.

    Returns PMDataFrame dict (chrom, start, end, intervalID).

    C++ signature: ``PyArg_ParseTuple(args, "O|OO", ...)``
    """

def pm_seed(seed: int, /) -> None:
    """Set random seed.

    C++ signature: ``PyArg_ParseTuple(args, "l", &seed)``
    """

# -- Virtual Tracks & Neighbors -----------------------------------------------

def pm_vtrack_compute(
    spec: dict[str, Any],
    intervals: Any,
    config: dict[str, Any] | None = ...,
    /,
) -> npt.NDArray[np.float64]:
    """Compute virtual track values.

    Returns a 1-D float64 numpy array with one value per interval.

    C++ signature: ``PyArg_ParseTuple(args, "OO|O", &py_spec, &py_intervals, &py_config)``
    """

def pm_find_neighbors(
    intervals1: Any,
    intervals2: Any,
    maxneighbors: int,
    mindist: float,
    maxdist: float,
    na_if_notfound: int,
    use_intervals1_strand: int = ...,
    /,
) -> Any:
    """Find nearest neighbor intervals.

    C++ signature: ``PyArg_ParseTuple(args, "OOiddi|i", ...)``
    """

# -- Sequence ------------------------------------------------------------------

def pm_seq_extract(
    intervals: Any, config: dict[str, Any] | None = ..., /
) -> list[str]:
    """Extract DNA sequences for intervals.

    Returns a list of sequence strings.

    C++ signature: ``PyArg_ParseTuple(args, "O|O", ...)``
    """

# -- Analysis / Stats ----------------------------------------------------------

def pm_partition(
    expr: str,
    breaks: list[float],
    intervals: Any,
    iterator: int | float | None = ...,
    include_lowest: bool = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Partition track values into bins.

    Returns dict with 'chrom', 'start', 'end', 'bin' arrays, or None
    if no values fall within bins.

    C++ signature: ``PyArg_ParseTuple(args, "OOO|OpO", ...)``
    """

def pm_dist(
    exprs: list[str],
    breaks_list: list[list[float]],
    intervals: Any,
    iterator: int | float | None = ...,
    include_lowest: bool = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> npt.NDArray[np.float64]:
    """Calculate distribution of track values over bins.

    Returns an N-dimensional numpy array of counts.

    C++ signature: ``PyArg_ParseTuple(args, "OOO|OpO", ...)``
    """

def pm_lookup(
    exprs: list[str],
    breaks_list: list[list[float]],
    lookup_table: npt.ArrayLike,
    intervals: Any,
    iterator: int | float | None = ...,
    include_lowest: bool = ...,
    force_binning: bool = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Lookup table transform on binned track values.

    Returns PMDataFrame dict (chrom, start, end, value, intervalID)
    or None.

    C++ signature: ``PyArg_ParseTuple(args, "OOOO|OppO", ...)``
    """

def pm_sample(
    expr: str,
    n: int,
    intervals: Any,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> npt.NDArray[np.float64] | None:
    """Sample values from track expression.

    Returns a 1-D float64 numpy array of sampled values, or None if
    no non-NaN data points exist.

    C++ signature: ``PyArg_ParseTuple(args, "OOO|OO", ...)``
    """

def pm_cor(
    exprs: list[str],
    intervals: Any,
    iterator: int | float | None = ...,
    method: str | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> list[dict[str, float]] | None:
    """Compute correlation between expression pairs.

    Returns a list of dicts (one per pair), each containing at least
    'cor', 'n', 'n.na' and optionally 'cov', 'mean1', 'mean2', 'sd1',
    'sd2' for Pearson method. Returns None on error.

    C++ signature: ``PyArg_ParseTuple(args, "OO|OOO", ...)``
    """

def pm_segment(
    expr: str,
    intervals: Any,
    minsegment: float,
    maxpval: float,
    onetailed: int,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Segment track expression using Wilcoxon test.

    Returns PMDataFrame dict or None.

    C++ signature: ``PyArg_ParseTuple(args, "OOddi|OO", ...)``
    """

def pm_wilcox(
    expr: str,
    intervals: Any,
    winsize1: float,
    winsize2: float,
    maxpval: float,
    onetailed: int,
    what2find: int,
    iterator: int | float | None = ...,
    config: dict[str, Any] | None = ...,
    /,
) -> dict[str, np.ndarray] | None:
    """Sliding-window Wilcoxon test on track expression.

    Returns PMDataFrame dict or None.

    C++ signature: ``PyArg_ParseTuple(args, "OOdddii|OO", ...)``
    """

# -- Interval Set Operations ---------------------------------------------------

def pm_intervals_union(
    intervals1: Any, intervals2: Any, /
) -> dict[str, np.ndarray]:
    """Union of two interval sets.

    Returns dict with 'chrom', 'start', 'end' arrays.

    C++ signature: ``PyArg_ParseTuple(args, "OO", ...)``
    """

def pm_intervals_intersect(
    intervals1: Any, intervals2: Any, /
) -> dict[str, np.ndarray]:
    """Intersection of two interval sets.

    Returns dict with 'chrom', 'start', 'end' arrays.

    C++ signature: ``PyArg_ParseTuple(args, "OO", ...)``
    """

def pm_intervals_diff(
    intervals1: Any, intervals2: Any, /
) -> dict[str, np.ndarray]:
    """Difference of two interval sets (set1 - set2).

    Returns dict with 'chrom', 'start', 'end' arrays.

    C++ signature: ``PyArg_ParseTuple(args, "OO", ...)``
    """

def pm_intervals_canonic(
    intervals: Any, merge_touching: bool = ..., /
) -> tuple[dict[str, np.ndarray], npt.NDArray[np.int64]]:
    """Canonicalize intervals (sort + merge overlapping/touching).

    Returns a tuple of (intervals_dict, mapping_array) where
    mapping_array maps original indices to canonical indices.

    C++ signature: ``PyArg_ParseTuple(args, "O|p", ...)``
    """

def pm_intervals_covered_bp(intervals: Any, /) -> int:
    """Count total covered basepairs.

    C++ signature: ``PyArg_ParseTuple(args, "O", ...)``
    """

# -- Genome Synthesis ----------------------------------------------------------

def pm_gsynth_train(
    intervals: Any,
    bin_indices: Any,
    iter_starts: Any,
    iter_chroms: Any,
    breaks: list[float],
    bin_map: Any,
    mask: Any,
    pseudocount: float,
    /,
) -> Any:
    """Train stratified Markov-5 model.

    C++ signature: ``PyArg_ParseTuple(args, "OOOOOOOd", ...)``
    """

def pm_gsynth_sample(
    cdf_list: Any,
    breaks: Any,
    bin_indices: Any,
    iter_starts: Any,
    iter_chroms: Any,
    intervals: Any,
    mask_copy: Any,
    output_path: str,
    output_format: int,
    n_samples: int,
    seed: int | None,
    /,
) -> Any:
    """Sample synthetic genome.

    C++ signature: ``PyArg_ParseTuple(args, "OOOOOOOsiiO", ...)``
    """

def pm_gsynth_replace_kmer(
    target: str,
    replacement: str,
    intervals: Any,
    output_path: str,
    output_format: int,
    /,
) -> Any:
    """Replace k-mers iteratively.

    C++ signature: ``PyArg_ParseTuple(args, "ssOsi", ...)``
    """

# -- Private Helpers -----------------------------------------------------------

def __pm_test_df() -> dict[str, np.ndarray]:
    """Test DataFrame conversion (internal)."""

def __read_df(df: Any, name: str = ..., /) -> None:
    """Read DataFrame from internal format (internal).

    C++ signature: ``PyArg_ParseTuple(args, "O|s", &py_df, &df_name)``
    """
