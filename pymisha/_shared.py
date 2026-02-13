"""
Shared globals and utilities for PyMisha modules.

Thread-safety note:
The module-level mutable state (`CONFIG`, `_GROOT`, `_GWD`, `_GDATASETS`,
`_VTRACKS`) is process-global and not synchronized for concurrent mutation.
Use PyMisha APIs from a single controlling thread or add external locking.
"""

import sys as _sys
from contextlib import contextmanager

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
    'min_processes': 4,             # Min workers for multitasking
    'max_processes': 20,            # Max workers for multitasking
    'max_data_size': 10000000,      # Max rows in memory
    'eval_buf_size': 1000,          # Batch size for expression eval
    'debug': False,                 # Debug prints
    'progress': True,              # False, True, 'tqdm', 'rich', 'text', or callable
    'progress_style': 'rich'        # Default when progress=True
}

# Global state
_GROOT = None    # Global database root
_UROOT = None    # User database root
_GWD = None      # Global working directory (tracks root or subdir)
_GDATASETS = []  # Loaded dataset roots (in load order)
_VTRACKS = {}    # Virtual tracks


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


def _df2pymisha(arg):
    """Convert DataFrame to internal format (following pynaryn pattern)."""
    if isinstance(arg, _pandas.DataFrame):
        colnames = arg.columns.values
        cols = [colnames]

        for i in range(colnames.size):
            series = arg.iloc[:, i]
            if isinstance(series.dtype, _pandas.CategoricalDtype):
                cat = series.astype("category")
                cols.append(
                    [
                        cat.cat.categories.to_numpy(copy=False),
                        cat.cat.codes.to_numpy(copy=False),
                    ]
                )
            else:
                cols.append(series.to_numpy(copy=False))
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


def _iterated_intervals(intervals, iterator):
    """Return intervals after applying iterator policy (includes intervalID)."""
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
