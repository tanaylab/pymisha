"""
Shared globals and utilities for PyMisha modules.

Thread-safety note:
The module-level mutable state (`CONFIG`, `_GROOT`, `_GWD`, `_GDATASETS`,
`_VTRACKS`) is process-global and not synchronized for concurrent mutation.
Use PyMisha APIs from a single controlling thread or add external locking.
"""

from __future__ import annotations

import contextlib as _contextlib
import sys as _sys
from contextlib import contextmanager
from typing import Any

import numpy as _numpy
import pandas as _pandas

try:
    import _pymisha
except ImportError as e:
    raise ImportError(
        "Failed to import _pymisha C extension. "
        "Please ensure the package is properly installed:\n"
        f"  pip install -e .\n"
        f"Original error: {e}"
    ) from e

# Configuration dictionary (following pynaryn pattern)
CONFIG = {
    'multitasking': True,           # Allow parallel processing
    'multitasking_stdout': False,   # Debug output from children
    'multitasking_strategy': 'auto',  # 'auto' | 'tracks' | 'tiles' (R 5.6.18 parity)
    'min_processes': 4,             # Min workers for multitasking
    'max_processes': 20,            # Max workers for multitasking
    'max_data_size': 10000000,      # Max rows in memory
    # Workload floors for fork-based multitasking. Fork+IPC costs ~150ms on
    # this stack; below these floors serial is faster (measured 20-200x on
    # hg38/Phylo447 sparse-track ops). Set to 0 to fork unconditionally.
    'min_scope4process': 1_000_000_000,   # min total scope (bp) per worker
    'min_intervs4process': 250_000,       # min total intervals before any fork
    'eval_buf_size': 1000,          # Batch size for expression eval
    'debug': False,                 # Debug prints
    'progress': True,              # False, True, 'tqdm', 'rich', 'text', or callable
    'progress_style': 'rich',       # Default when progress=True
    # Parallel writers for the empty-chrom signature-file dispatch inside
    # gtrack_create_sparse on non-indexed DBs. None = follow multitasking
    # + max_processes (capped at 8). Set explicitly (1..16) to tune for
    # your storage; 1 forces sequential. Only relevant on NFS-like
    # filesystems with many alt contigs.
    'track_create_parallel_writers': None,
}

# Global state
_GROOT: str | None = None    # Global database root
_UROOT: str | None = None    # User database root
_GWD: str | None = None      # Global working directory (tracks root or subdir)
_GDATASETS: list[str] = []  # Loaded dataset roots (in load order)
_VTRACKS: dict[str, dict[str, Any]] = {}    # Virtual tracks


def gmax_processes(n=None):
    """Get or set the maximum number of worker processes for parallel operations.

    When called without arguments, returns the current value.
    When called with a positive integer, sets the value and returns it.

    Parameters
    ----------
    n : int, optional
        Maximum number of worker processes.  Must be >= 1.
        ``1`` means single-process (no parallelism).

    Returns
    -------
    int
        Current (or newly set) maximum process count.
    """
    if n is None:
        return int(CONFIG.get("max_processes", 1))
    n = int(n)
    if n < 1:
        raise ValueError("gmax_processes must be >= 1")
    CONFIG["max_processes"] = n
    return n


def _make_progress_callback(progress, total=None, desc=None):
    if progress is None:
        progress = CONFIG.get('progress', "rich")
    if not progress:
        return None, None

    if callable(progress):
        return progress, None

    style = progress
    if style is True:
        style = CONFIG.get('progress_style', 'rich')

    if style in ('tqdm', 'auto'):
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total, desc=desc)

            def cb(done, total, pct):
                if total is not None and pbar.total != total:
                    pbar.total = total
                pbar.n = int(done)
                pbar.refresh()

            return cb, pbar.close
        except Exception:
            style = 'text'

    if style == 'rich':
        try:
            from rich.progress import Progress
            progress_obj = Progress()
            progress_obj.start()
            task_id = progress_obj.add_task(desc or "working", total=total)

            def cb(done, total, pct):
                if total is not None:
                    progress_obj.update(task_id, total=total)
                progress_obj.update(task_id, completed=done)

            def close():
                progress_obj.stop()

            return cb, close
        except Exception:
            style = 'text'

    if style == 'text':
        last = {'pct': -1}
        label = desc or "progress"

        def cb(done, total, pct):
            if pct != last['pct']:
                _sys.stderr.write(f"\r{label}: {pct}%")
                if pct >= 100:
                    _sys.stderr.write("\n")
                _sys.stderr.flush()
                last['pct'] = pct

        return cb, None

    return None, None


@contextmanager
def _progress_context(progress=None, total=None, desc=None):
    cb, close = _make_progress_callback(progress, total=total, desc=desc)
    prev = CONFIG.get('_progress_cb')
    if cb:
        CONFIG['_progress_cb'] = cb
    try:
        yield cb
    finally:
        if cb:
            if prev is None:
                CONFIG.pop('_progress_cb', None)
            else:
                CONFIG['_progress_cb'] = prev
        if close:
            close()


def _checkroot():
    """Verify database is initialized."""
    if _GROOT is None:
        raise RuntimeError('Database not set. Call gdb_init() first.')


# Python-side cache for `_pymisha.pm_track_names()`.  The C++ side already
# keeps a `m_track_cache` of track names; converting that vector to a
# Python list of 15k PyUnicode objects (and then `set(...)`-ing it) still
# costs ~3 ms on hg38, and several hot-path Python helpers call it on
# every gextract / gsummary / gscreen invocation.
#
# Both the list and the set are cached.  Invalidation is wired to
# gdb_init / gdb_reload / gdb_unload via :func:`_clear_track_names_cache`.
_TRACK_NAMES_LIST: list[str] | None = None
_TRACK_NAMES_SET: frozenset[str] | None = None


def _clear_track_names_cache() -> None:
    """Drop the cached pm_track_names result.  Called on db transitions."""
    global _TRACK_NAMES_LIST, _TRACK_NAMES_SET
    _TRACK_NAMES_LIST = None
    _TRACK_NAMES_SET = None


def _track_names_list() -> list[str]:
    """Return the cached pm_track_names() list (no copy)."""
    global _TRACK_NAMES_LIST
    if _TRACK_NAMES_LIST is None:
        _TRACK_NAMES_LIST = _pymisha.pm_track_names()
    return _TRACK_NAMES_LIST


def _track_names_set() -> frozenset[str]:
    """Return the cached pm_track_names() result as a frozenset."""
    global _TRACK_NAMES_SET
    if _TRACK_NAMES_SET is None:
        _TRACK_NAMES_SET = frozenset(_track_names_list())
    return _TRACK_NAMES_SET


def _pm_dbreload() -> None:
    """Reload the C++ track scan + clear all Python-side caches.

    Every internal track-mutation site (gtrack_rm, gtrack_create_*,
    gintervals_save, gdir / gdataset changes...) used to call
    ``_pymisha.pm_dbreload()`` directly, which bypasses the Python
    caches (track names, computed-track types, expression-validation
    set).  Routing all those callers through this helper keeps the
    cached views in sync with the C++ side.
    """
    _pymisha.pm_dbreload()
    _clear_track_names_cache()
    from .tracks import _clear_computed_track_cache
    _clear_computed_track_cache()


def _df2pymisha(arg):
    """Convert DataFrame to internal format (following pynaryn pattern)."""
    if isinstance(arg, _pandas.DataFrame):
        colnames = arg.columns.to_numpy()
        cols = [colnames]

        for i in range(colnames.size):
            series = arg.iloc[:, i]
            if isinstance(series.dtype, _pandas.CategoricalDtype):
                cat = series.astype("category")
                cols.append(
                    [
                        cat.cat.categories.to_numpy(),
                        cat.cat.codes.to_numpy(),
                    ]
                )
            else:
                cols.append(series.to_numpy())
        return cols
    return arg


def _pymisha2df(arg):
    """Convert internal format to DataFrame."""
    if (
        arg is None
        or not isinstance(arg, list)
        or len(arg) < 2
        or not isinstance(arg[0], _numpy.ndarray)
        or len(arg) != len(arg[0]) + 1
    ):
        return arg

    colnames = arg[0]
    numrows = -1
    data = {}

    for i in range(colnames.size):
        colname = colnames[i]
        col = arg[i + 1]
        if isinstance(col, _numpy.ndarray):
            if numrows != -1 and col.size != numrows:
                return arg
            numrows = col.size
            data[colname] = col
        else:
            if (
                not isinstance(col, list)
                or len(col) != 2
                or not isinstance(col[0], _numpy.ndarray)
                or not isinstance(col[1], _numpy.ndarray)
                or (numrows != -1 and len(col[1]) != numrows)
            ):
                return arg
            numrows = len(col[1])
            cats = _pandas.Categorical.from_codes(col[1], col[0])
            data[colname] = cats

    return _pandas.DataFrame(data)


def _itr2pymisha(itr):
    """Convert iterator to internal format."""
    return [itr[0], _df2pymisha(itr[1])] if isinstance(itr, list) and len(itr) == 2 else _df2pymisha(itr)


def _intersect_scope_with_iterator(scope, iterator_intervals):
    """Intersect scope intervals with iterator intervals.

    Both inputs are DataFrames with chrom/start/end columns.  The iterator
    intervals are first sorted and unified (overlapping intervals merged).
    A vectorized per-chromosome sweep then computes all pairwise
    intersections between scope and unified-iterator intervals.

    Returns
    -------
    tuple of (DataFrame, numpy.ndarray)
        - result_df : DataFrame with chrom/start/end of each intersection
        - id_map : 1-D int array, one entry per result row, holding the
          1-based index of the originating scope interval.
    """
    if len(scope) == 0 or len(iterator_intervals) == 0:
        empty = _pandas.DataFrame({"chrom": _pandas.Series([], dtype=str),
                                    "start": _pandas.Series([], dtype=int),
                                    "end": _pandas.Series([], dtype=int)})
        return empty, _numpy.array([], dtype=int)

    # --- iterator: sort by (chrom, start), merge overlapping within chrom ---
    itr_chroms = iterator_intervals["chrom"].to_numpy()
    itr_starts = iterator_intervals["start"].to_numpy(dtype=_numpy.int64, copy=False)
    itr_ends = iterator_intervals["end"].to_numpy(dtype=_numpy.int64, copy=False)
    itr_order = _numpy.lexsort((itr_starts, itr_chroms))
    itr_chroms = itr_chroms[itr_order]
    itr_starts = itr_starts[itr_order]
    itr_ends = itr_ends[itr_order]

    n_iter = len(itr_chroms)
    # Per-chrom running max of end (reset on chrom boundary) - vectorized
    # per slice via np.maximum.accumulate.
    chrom_boundary = _numpy.empty(n_iter, dtype=bool)
    chrom_boundary[0] = True
    if n_iter > 1:
        chrom_boundary[1:] = itr_chroms[1:] != itr_chroms[:-1]
    chrom_start_idx = _numpy.flatnonzero(chrom_boundary)
    chrom_end_idx = _numpy.r_[chrom_start_idx[1:], n_iter]

    running_max_end = _numpy.empty(n_iter, dtype=_numpy.int64)
    for s, e in zip(chrom_start_idx, chrom_end_idx, strict=False):
        running_max_end[s:e] = _numpy.maximum.accumulate(itr_ends[s:e])

    new_group = _numpy.empty(n_iter, dtype=bool)
    new_group[0] = True
    if n_iter > 1:
        new_group[1:] = chrom_boundary[1:] | (itr_starts[1:] >= running_max_end[:-1])

    group_id = _numpy.cumsum(new_group) - 1
    n_groups = int(group_id[-1] + 1)
    merged_chroms = itr_chroms[new_group]
    merged_starts = itr_starts[new_group]
    merged_ends = _numpy.zeros(n_groups, dtype=_numpy.int64)
    _numpy.maximum.at(merged_ends, group_id, itr_ends)

    # --- scope: sort by (chrom, start), keep original 1-based ids ---
    sc_chroms = scope["chrom"].to_numpy()
    sc_starts = scope["start"].to_numpy(dtype=_numpy.int64, copy=False)
    sc_ends = scope["end"].to_numpy(dtype=_numpy.int64, copy=False)
    sc_order = _numpy.lexsort((sc_starts, sc_chroms))
    sc_chroms_sorted = sc_chroms[sc_order]
    sc_starts_sorted = sc_starts[sc_order]
    sc_ends_sorted = sc_ends[sc_order]
    sc_orig_ids = sc_order.astype(_numpy.int64, copy=False) + 1

    # Per-chrom processing - all ops inside each chrom are vectorized.
    n_merged = len(merged_chroms)
    if n_merged == 0:
        empty = _pandas.DataFrame({"chrom": _pandas.Series([], dtype=str),
                                    "start": _pandas.Series([], dtype=int),
                                    "end": _pandas.Series([], dtype=int)})
        return empty, _numpy.array([], dtype=int)
    merged_chrom_boundary = _numpy.empty(n_merged, dtype=bool)
    merged_chrom_boundary[0] = True
    if n_merged > 1:
        merged_chrom_boundary[1:] = merged_chroms[1:] != merged_chroms[:-1]
    merged_chrom_starts = _numpy.flatnonzero(merged_chrom_boundary)
    merged_chrom_ends = _numpy.r_[merged_chrom_starts[1:], n_merged]
    iter_chrom_map = {
        merged_chroms[s]: (int(s), int(e))
        for s, e in zip(merged_chrom_starts, merged_chrom_ends, strict=False)
    }

    sc_chrom_boundary = _numpy.empty(len(sc_chroms_sorted), dtype=bool)
    sc_chrom_boundary[0] = True
    if len(sc_chroms_sorted) > 1:
        sc_chrom_boundary[1:] = sc_chroms_sorted[1:] != sc_chroms_sorted[:-1]
    sc_chrom_starts = _numpy.flatnonzero(sc_chrom_boundary)
    sc_chrom_ends = _numpy.r_[sc_chrom_starts[1:], len(sc_chroms_sorted)]

    out_chroms_parts = []
    out_starts_parts = []
    out_ends_parts = []
    out_ids_parts = []

    for sc_s, sc_e in zip(sc_chrom_starts, sc_chrom_ends, strict=False):
        chrom = sc_chroms_sorted[sc_s]
        rng = iter_chrom_map.get(chrom)
        if rng is None:
            continue
        it_lo, it_hi = rng
        it_s = merged_starts[it_lo:it_hi]
        it_e = merged_ends[it_lo:it_hi]
        sc_s_arr = sc_starts_sorted[sc_s:sc_e]
        sc_e_arr = sc_ends_sorted[sc_s:sc_e]
        sc_ids_arr = sc_orig_ids[sc_s:sc_e]

        # For each scope[i], iter range [lo[i], hi[i]) of merged iter that overlap.
        # lo[i] = first j with it_e[j] > sc_s_arr[i]
        # hi[i] = first j with it_s[j] >= sc_e_arr[i]
        lo = _numpy.searchsorted(it_e, sc_s_arr, side='right')
        hi = _numpy.searchsorted(it_s, sc_e_arr, side='left')
        sizes = _numpy.maximum(hi - lo, 0)
        total = int(sizes.sum())
        if total == 0:
            continue

        sc_rep_idx = _numpy.repeat(_numpy.arange(len(sc_s_arr)), sizes)
        offsets = _numpy.repeat(lo, sizes)
        cum_sizes = _numpy.r_[0, _numpy.cumsum(sizes[:-1])]
        within_group = _numpy.arange(total) - _numpy.repeat(cum_sizes, sizes)
        iter_idx = offsets + within_group

        sc_s_rep = sc_s_arr[sc_rep_idx]
        sc_e_rep = sc_e_arr[sc_rep_idx]
        ov_s = _numpy.maximum(sc_s_rep, it_s[iter_idx])
        ov_e = _numpy.minimum(sc_e_rep, it_e[iter_idx])
        valid = ov_s < ov_e
        if not valid.any():
            continue

        n_valid = int(valid.sum())
        out_chroms_parts.append(_numpy.full(n_valid, chrom, dtype=object))
        out_starts_parts.append(ov_s[valid])
        out_ends_parts.append(ov_e[valid])
        out_ids_parts.append(sc_ids_arr[sc_rep_idx][valid])

    if not out_chroms_parts:
        empty = _pandas.DataFrame({"chrom": _pandas.Series([], dtype=str),
                                    "start": _pandas.Series([], dtype=int),
                                    "end": _pandas.Series([], dtype=int)})
        return empty, _numpy.array([], dtype=int)

    out_chroms = _numpy.concatenate(out_chroms_parts)
    out_starts = _numpy.concatenate(out_starts_parts)
    out_ends = _numpy.concatenate(out_ends_parts)
    out_ids = _numpy.concatenate(out_ids_parts)

    result_df = _pandas.DataFrame({
        "chrom": out_chroms,
        "start": out_starts,
        "end": out_ends,
    })
    return result_df, out_ids


def _preprocess_intervals_iterator(intervals, iterator):
    """Handle DataFrame-as-iterator by intersecting with scope intervals.

    Parameters
    ----------
    intervals : DataFrame
        Scope intervals (chrom/start/end).
    iterator : any
        The iterator parameter. If a DataFrame with chrom/start/end columns,
        it is treated as an intervals-based iterator.

    Returns
    -------
    tuple of (DataFrame, iterator, id_map_or_None)
        - new_intervals : DataFrame to pass to C++ (intersected intervals
          when iterator was a DataFrame, otherwise original intervals)
        - new_iterator : iterator to pass to C++ (-1 when iterator was a
          DataFrame, otherwise the original iterator)
        - id_map : numpy int array mapping sequential 1-based index back to
          original scope intervalID (1-based), or None if iterator was not
          a DataFrame
    """
    if isinstance(iterator, str):
        # Resolve 1D interval set names to DataFrames (like R misha's .giterator).
        # 2D interval sets are left as strings for downstream 2D handling.
        from .intervals import gintervals_exists, gintervals_load

        if gintervals_exists(iterator):
            loaded = gintervals_load(iterator)
            if isinstance(loaded, _pandas.DataFrame) and "chrom1" in loaded.columns:
                # 2D interval set — leave as string for 2D iterator path
                return intervals, iterator, None
            iterator = loaded
        else:
            # Not an interval set — let downstream handle it (track name, etc.)
            return intervals, iterator, None

    if not isinstance(iterator, _pandas.DataFrame):
        return intervals, iterator, None

    if len(iterator) == 0 or len(intervals) == 0:
        empty = _pandas.DataFrame({"chrom": _pandas.Series([], dtype=str),
                                    "start": _pandas.Series([], dtype=int),
                                    "end": _pandas.Series([], dtype=int)})
        return empty, -1, _numpy.array([], dtype=int)

    required = {"chrom", "start", "end"}
    if not required.issubset(iterator.columns):
        raise ValueError(
            "DataFrame iterator must have 'chrom', 'start', 'end' columns"
        )

    new_intervals, id_map = _intersect_scope_with_iterator(intervals, iterator)
    if len(new_intervals) == 0:
        empty = _pandas.DataFrame({"chrom": _pandas.Series([], dtype=str),
                                    "start": _pandas.Series([], dtype=int),
                                    "end": _pandas.Series([], dtype=int)})
        return empty, -1, _numpy.array([], dtype=int)
    return new_intervals, -1, id_map


@_contextlib.contextmanager
def _config_no_mt(id_map):
    """Context manager that disables C++ multitasking when id_map is set.

    When a DataFrame iterator was preprocessed into a small set of
    intersected intervals, the C++ fork/FIFO multitasking overhead
    dominates the actual work.  The C++ side reads multitasking from
    the module-level ``CONFIG`` global, so we temporarily mutate it.
    """
    if id_map is None:
        yield CONFIG
        return
    prev = CONFIG.get("multitasking")
    CONFIG["multitasking"] = False
    try:
        yield CONFIG
    finally:
        if prev is None:
            CONFIG.pop("multitasking", None)
        else:
            CONFIG["multitasking"] = prev


def _remap_interval_ids(df, id_map):
    """Remap intervalID column using id_map from _preprocess_intervals_iterator.

    Parameters
    ----------
    df : DataFrame or None
        Result DataFrame that may contain an 'intervalID' column.
    id_map : numpy.ndarray or None
        Mapping array from _preprocess_intervals_iterator.
    """
    if id_map is None or df is None or not isinstance(df, _pandas.DataFrame):
        return df
    if "intervalID" not in df.columns or len(id_map) == 0:
        return df
    ids = df["intervalID"].to_numpy()
    # ids are 1-based sequential indices into the intersected intervals
    # id_map holds the original scope 1-based intervalID for each
    valid = (ids >= 1) & (ids <= len(id_map))
    new_ids = _numpy.zeros_like(ids)
    new_ids[valid] = id_map[ids[valid] - 1]
    df = df.copy()
    df["intervalID"] = new_ids
    return df


def _iterated_intervals(intervals, iterator):
    """Return intervals after applying iterator policy (includes intervalID)."""
    if isinstance(iterator, _pandas.DataFrame):
        new_intervals, new_iterator, id_map = _preprocess_intervals_iterator(
            intervals, iterator
        )
        if len(new_intervals) == 0:
            return _pandas.DataFrame({
                "chrom": _pandas.Series([], dtype=str),
                "start": _pandas.Series([], dtype=int),
                "end": _pandas.Series([], dtype=int),
                "intervalID": _pandas.Series([], dtype=int),
            })
        result = _pymisha.pm_iterate(
            _df2pymisha(new_intervals), new_iterator, CONFIG
        )
        out = _pymisha2df(result)
        return _remap_interval_ids(out, id_map)
    if iterator is None:
        out = intervals.copy()
        out["intervalID"] = _numpy.arange(1, len(out) + 1)
        return out
    result = _pymisha.pm_iterate(_df2pymisha(intervals), iterator, CONFIG)
    return _pymisha2df(result)


def _chunk_slices(n, chunk_size):
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def _bound_colname(expr: str, maxlen: int = 40) -> str:
    if len(expr) > maxlen:
        return expr[: maxlen - 3] + '...'
    return expr


def _gwd_prefix():
    """Return the dotted prefix for the current working directory.

    If GWD is ``{GROOT}/tracks/subdir``, returns ``"subdir."``.
    If GWD is the tracks root, returns ``""``.
    """
    if _GROOT is None or _GWD is None:
        return ""
    import os
    tracks_root = os.path.join(_GROOT, "tracks")
    if tracks_root == _GWD:
        return ""
    relpath = os.path.relpath(_GWD, tracks_root)
    return relpath.replace(os.sep, ".") + "."


def _apply_gwd_to_names(names):
    """Filter and rebase names by current working directory prefix.

    Given a list of dotted names (tracks or intervals), keeps only those
    under the current GWD prefix and strips that prefix.
    """
    prefix = _gwd_prefix()
    if not prefix:
        return names
    return [n[len(prefix):] for n in names if n.startswith(prefix)]
