"""
Shared globals and utilities for PyMisha modules.

Thread-safety note:
The module-level mutable state (`CONFIG`, `_GROOT`, `_GWD`, `_GDATASETS`,
`_VTRACKS`) is process-global and not synchronized for concurrent mutation.
Use PyMisha APIs from a single controlling thread or add external locking.
"""

from __future__ import annotations

import contextlib as _contextlib
import datetime as _datetime
import os as _os
import sys as _sys
import warnings as _warnings
from contextlib import contextmanager
from typing import Any

import numpy as _numpy
import pandas as _pandas

from ._log import PymishaWarning, get_logger, user_stacklevel

_logger = get_logger(__name__)

try:
    import _pymisha
except ImportError as e:
    raise ImportError(
        "Failed to import _pymisha C extension. "
        "Please ensure the package is properly installed:\n"
        f"  pip install -e .\n"
        f"Original error: {e}"
    ) from e

def _default_max_mem_usage_kb() -> int:
    """R misha's gmax.mem.usage default: 80% of the system RAM in KB, at least 10 GB."""
    try:
        sys_mem = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        sys_mem = 0
    return max(10_000_000, int(sys_mem * 0.8 / 1024))


# Configuration dictionary (following pynaryn pattern)
CONFIG = {
    'multitasking': True,           # Allow parallel processing
    'multitasking_stdout': False,   # Debug output from children
    'multitasking_strategy': 'auto',  # 'auto' | 'tracks' | 'tiles' (R 5.6.18 parity)
    'min_processes': 4,             # Min workers for multitasking
    # Max workers for multitasking. 70% of cores capped at 32, as R misha's
    # gmax.processes auto-calculates it (5.11.20); the old flat 20 left
    # large-core machines idle (a 56-track extraction ran ~1.5x slower at 20
    # workers than at 40). Past ~32 the per-worker cost - fork, shared memory,
    # result merge - outweighs the extra parallelism: misha measured 89 workers
    # on a 128-core host as slower than 32 on every workload tried, and
    # cold-NFS whole-genome scans saturate at ~32 too. The cap only binds above
    # 46 cores. Both the C++ scanner and the Python fork paths clamp to the
    # actual core count, so a small machine is unaffected, and setting
    # CONFIG['max_processes'] by hand still overrides it.
    'max_processes': max(4, min(32, int((_os.cpu_count() or 1) * 0.7))),
    'max_data_size': 10000000,      # Max rows in memory
    # Memory budget in KB, R misha's gmax.mem.usage: 80% of the RAM, at least
    # 10 GB. The parallel 2D imports start a chromosome pair only when its
    # estimated memory fits into it with the pairs being written.
    'max_mem_usage': _default_max_mem_usage_kb(),
    # gintervals_2d_all() returns every chromosome pair on a genome with up to
    # this many chromosomes and only the intra-chromosomal pairs above it, as
    # R misha's gmulticontig.2d.threshold does.
    'multicontig_2d_threshold': 100,
    # Above that many chromosomes, the whole-genome 2D scope a 2D computation
    # runs over without explicit intervals: "full" (all chromosome pairs) or
    # "diagonal" (the cis pairs only). Unset (None), such a computation is an
    # error, and gintervals_2d_all() warns that it returns the cis pairs only.
    # R misha's gmulticontig.2d.mode.
    'multicontig_2d_mode': None,
    # Umask held around every write to the database. "0007" gives 660 files and
    # 770 directories - group rwx, no world access - which is what pymisha's own
    # C++ layer already sets (PyMisha's constructor) and what the shared lab
    # databases look like on disk. The Python write sites used to leave the
    # process umask alone, so a track directory written from C++ was 0770 inside
    # a namespace directory written from Python at 0755, and a .interv or a
    # .db.cache.dirty saved from Python was 0644 - which a colleague could not
    # then overwrite from either package. Set to None to leave the process umask
    # alone. Mirrors R misha's gpermissions.umask (5.11.21).
    'permissions_umask': 0o007,
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

# CONFIG['max_data_size'] or CONFIG['max_mem_usage'] of None, inf or any value of at least
# this (R misha's MAX_DATA_SIZE_NO_LIMIT) is no limit
_NO_LIMIT = 2**53
# their values when the key is absent
_LIMIT_DEFAULTS = {"max_data_size": 10_000_000, "max_mem_usage": _NO_LIMIT}


def _config_limit(name: str, config: dict | None = None) -> int:
    """CONFIG[name] for "max_data_size" (R's gmax.data.size) or "max_mem_usage" (gmax.mem.usage,
    in KB), as R misha reads them: None, inf or any value of 2**53 or more is no limit (2**53),
    NaN or a negative value is an error, and a max_data_size below 1 counts as 1. The C++ side
    reads them through this too (PyMisha::load_options)."""
    cfg = CONFIG if config is None else config
    if name not in cfg:
        return _LIMIT_DEFAULTS[name]
    value = cfg[name]
    if value is None:
        return _NO_LIMIT
    if value is _pandas.NA or (isinstance(value, (float, _numpy.floating)) and _numpy.isnan(value)):
        raise ValueError(f"CONFIG['{name}'] must be a number, not NA")
    if isinstance(value, (bool, _numpy.bool_)) or not isinstance(
            value, (int, float, _numpy.integer, _numpy.floating)):
        raise ValueError(f"CONFIG['{name}'] must be a number, not {value!r}")
    if value < 0:
        raise ValueError(f"CONFIG['{name}'] must not be negative (got {float(value):g})")
    if value >= _NO_LIMIT:
        return _NO_LIMIT
    return max(int(value), 1) if name == "max_data_size" else int(value)


def _config_multitasking(config: dict | None = None) -> bool:
    """CONFIG['multitasking'] (R's gmultitasking): True or False (or 1 or 0), and None or an
    absent key is False. Any other value, NaN included, is an error, so that the C++ side
    (which reads it through this, in PyMisha::load_options) and the Python fork paths agree."""
    value = (CONFIG if config is None else config).get("multitasking")
    if value is None:
        return False
    if isinstance(value, (bool, _numpy.bool_)):
        return bool(value)
    if isinstance(value, (int, _numpy.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"CONFIG['multitasking'] must be True or False, not {value!r}")


# Global state
_GROOT: str | None = None    # Global database root
_UROOT: str | None = None    # User database root
_GWD: str | None = None      # Global working directory (tracks root or subdir)
_GDATASETS: list[str] = []  # Loaded dataset roots (in load order)
# The chromosome names by chrom id of each loaded dataset that numbers them differently
# from the working database (R's GDATASET_CHROMS), by its root; see gdataset_load.
_GDATASET_CHROMS: dict[str, list[str]] = {}
_VTRACKS: dict[str, dict[str, Any]] = {}    # Virtual tracks of the working directory
# The virtual tracks of the other working directories they were created in, by directory
# (R's GVTRACKS keeps them by GWD): gdir_cd puts them back on return
_GVTRACKS: dict[str, dict[str, dict[str, Any]]] = {}


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
        except ImportError:
            _logger.debug("tqdm is not installed; falling back to text progress", exc_info=True)
            style = 'text'
        except Exception:
            _logger.warning("tqdm progress bar failed; falling back to text progress", exc_info=True)
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
        except ImportError:
            _logger.debug("rich is not installed; falling back to text progress", exc_info=True)
            style = 'text'
        except Exception:
            _logger.warning("rich progress bar failed; falling back to text progress", exc_info=True)
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


def _touch_db_cache_dirty(groot: str | None = None) -> None:
    """Write R misha's ``.db.cache.dirty`` sentinel for *groot*.

    R misha caches its track/interval-set inventory in
    ``<groot>/.db.cache`` and decides whether that cache is stale purely
    by checking whether ``<groot>/.db.cache.dirty`` exists (see misha's
    ``R/db-cache.R``, ``.gdb.cache_is_dirty`` / ``.gdb.cache_mark_dirty``).
    pymisha writes directly to the database through the C++ layer and
    never wrote that sentinel, so a track or interval set created in
    Python stayed invisible to a fresh R ``gsetroot()`` on the same
    database until someone ran ``gdb.reload(rescan = TRUE)``.

    Call this from every pymisha path that creates, removes, or renames
    a track or interval set. Best-effort: a failure to write the
    sentinel (e.g. a read-only database) does not raise, mirroring R's
    own tolerant behavior in ``.gdb.cache_mark_dirty`` - but it does warn,
    as R does, because the consequence is a sibling R session serving a
    stale inventory with no other sign that anything went wrong.
    """
    if groot is None:
        groot = _GROOT
    if not groot:
        return
    dirty_path = _os.path.join(groot, ".db.cache.dirty")
    try:
        with _with_umask(), open(dirty_path, "w") as fh:
            fh.write(_datetime.datetime.now().isoformat())
    except OSError as exc:
        _warnings.warn(
            f"could not write {dirty_path}: {exc}. An R misha session on this database "
            "may keep serving a stale track/interval-set list until it runs "
            "gdb.reload(rescan = TRUE).",
            PymishaWarning,
            stacklevel=user_stacklevel(),
        )


@_contextlib.contextmanager
def _with_umask():
    """Hold ``CONFIG["permissions_umask"]`` for the shortest possible span.

    The umask is process-global state, so it is taken immediately before the
    write and restored in a ``finally`` that runs on a normal return, on an
    error and on a KeyboardInterrupt alike. Never set it anywhere that outlives
    a call: R misha used to set it once in ``.onLoad()`` and never restore it,
    which silently changed the permissions of everything the session wrote,
    misha or not.

    ``None`` means "leave my umask alone". Mirrors R misha's ``.gwith_umask``.
    """
    um = CONFIG.get("permissions_umask")
    if um is None:
        yield
        return
    old = _os.umask(um)
    try:
        yield
    finally:
        _os.umask(old)


@_contextlib.contextmanager
def _with_cpp_umask():
    """Hold the umask 007 that every C++ call runs under (PyMisha, as R misha's
    RdbInitializer), for a file R misha writes from C++ and pymisha from Python: a new file
    is 0660 whatever the caller's umask, and one written in place keeps its permission bits.
    """
    old = _os.umask(0o007)
    try:
        yield
    finally:
        _os.umask(old)


@_contextlib.contextmanager
def _write_file(path):
    """``open(path, "wb")`` for a file the database keeps.

    A write that fails midway, fails when ``close()`` writes the last buffer
    (a full disk, a quota, an NFS error reported at close) or is interrupted
    removes the file and re-raises, naming it, instead of leaving a truncated
    file behind for the next scan to find. Mirrors R misha's
    ``.gserialize_file``.
    """
    # Opened outside the try: a file that could not even be opened was not truncated, so it stays.
    fh = open(path, "wb")  # noqa: SIM115
    try:
        with fh:
            yield fh
    except BaseException as exc:
        with _contextlib.suppress(OSError):
            _os.unlink(path)
        if isinstance(exc, OSError) and exc.filename is None:
            exc.filename = _os.fspath(path)
        raise


def _invalidate_dir_cache(*paths: object) -> None:
    """Drop the C++ memoised track index for one or more track directories.

    ``GenomeTrack::get_track_index`` and ``TrackIndex2D`` memoise a parsed index
    per track directory, keyed by absolute path, for the life of the process.
    Nothing invalidated them, so any path that removes, recreates, renames or
    converts a track directory left later reads routed through an index
    describing a layout that is no longer on disk: ``Cannot open
    .../track.dat``, or silently wrong values. Unlike R misha there is no
    ``gdb.reload(rescan=True)`` here, so a poisoned session could only be fixed
    by restarting the interpreter.

    Call this from every path that changes what lives at a track directory.
    Negative lookups are never cached, so per-chrom -> indexed conversion is
    already safe; the stale case is "cache says indexed" and the index was then
    removed or replaced under the same path.

    Best-effort: a cache miss is not fatal, so failures are swallowed. Mirrors
    misha's ``.gdb.invalidate_dir_cache`` (defect F, 5.11.20).
    """
    args = [str(p) for p in paths if p]
    if not args:
        return
    with _contextlib.suppress(Exception):
        _pymisha.pm_invalidate_dir_cache(args)


def _file_stamp(path: Any) -> tuple[int, ...] | None:
    """What tells a file or directory from another that another session put at its path
    since: its inode, size, mtime and ctime (a rename or a write changes the ctime); None
    if it is missing."""
    try:
        st = _os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _track_stamp(track_path: str) -> tuple[Any, ...]:
    """The stamp of a track directory and its indexed files (_file_stamp). A value cached
    from a track is stale once another session replaced the track (gtrack_rm then a
    creation, gtrack_copy over it, a conversion), which nothing in this session clears."""
    return tuple(_file_stamp(_os.path.join(track_path, f)) for f in ("", "track.idx", "track.dat"))


def _pm_dbreload(groot: str | None = None) -> None:
    """Reload the C++ track scan + clear all Python-side caches.

    Every internal track-mutation site (gtrack_rm, gtrack_create_*,
    gintervals_save, gdir / gdataset changes...) used to call
    ``_pymisha.pm_dbreload()`` directly, which bypasses the Python
    caches (track names, computed-track types, expression-validation
    set).  Routing all those callers through this helper keeps the
    cached views in sync with the C++ side.

    *groot*: the database root the triggering mutation actually wrote
    to, when that can differ from the current ``_GROOT`` - a user-root
    overlay (``_UROOT``, see ``_target_root()`` in tracks.py), or an
    explicit ``db=`` argument on a track op that targets another loaded
    database. Defaults to the current ``_GROOT`` (matching the previous,
    implicit behavior) when the caller doesn't know any better.
    """
    _pymisha.pm_dbreload()
    _clear_track_names_cache()
    from .tracks import _clear_computed_track_cache
    _clear_computed_track_cache()
    # Also tell a sibling R session sharing this database that its
    # .db.cache is stale (R never sees pymisha's mutations otherwise).
    _touch_db_cache_dirty(groot)


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
            elif colnames[i] == "strand" and _pandas.api.types.is_integer_dtype(series.dtype) and series.hasnans:
                # (pandas would hand a nullable integer column with NA over as floats; R tells
                # a missing integer strand from a missing double one)
                cols.append(series.to_numpy(dtype=object))
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
    # Dedup colliding column names (C++ PMDataFrame convention: append '_').
    # Two expressions can resolve to the same name (e.g. the same track listed
    # twice); without this they would silently overwrite each other in this dict.
    seen: set = set()

    for i in range(colnames.size):
        colname = colnames[i]
        while colname in seen:
            colname += "_"
        seen.add(colname)
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


_CHROM_COLS = ("chrom", "chrom1", "chrom2")


def _chrom_names(chroms: Any, col: str = "chrom", arg: str = "intervals") -> _pandas.Series:
    """The chromosome names of a chrom column, read as R misha reads one: a string is the name,
    and a number is printed as as.character() prints a whole number (1 and 1.0 are "1", -0 is
    "0"), for the chromosome aliases to resolve as gintervals(1, ...) does. A missing value or a
    fraction is R's "Invalid format of interval at index i" (1-based); a bool or another type is
    an error too (col and arg name the column and the argument in it). A Series that holds only
    strings - object, string or categorical - is returned as it is."""
    s = chroms if isinstance(chroms, _pandas.Series) else _pandas.Series(chroms)
    what = "columns chrom1 and chrom2" if col in ("chrom1", "chrom2") else f"column {col}"
    bad_type = ValueError(f"Invalid format of {arg} argument: {what} must be character, factor or numeric")

    def bad_value(i: int) -> ValueError:
        return ValueError(f"Invalid format of interval at index {i + 1}")

    def first(mask: Any) -> int:
        return int(_numpy.argmax(mask))

    dtype = s.dtype
    if isinstance(dtype, _pandas.CategoricalDtype):
        missing = s.cat.codes.to_numpy() < 0
        if missing.any():
            raise bad_value(first(missing))
        if _pandas.api.types.infer_dtype(dtype.categories, skipna=False) == "string":
            return s
        return _chrom_names(s.astype(object), col, arg)
    if _pandas.api.types.is_bool_dtype(dtype):
        raise bad_type
    if _pandas.api.types.is_integer_dtype(dtype):
        missing = s.isna().to_numpy()
        if missing.any():
            raise bad_value(first(missing))
        return s.astype(str)
    if _pandas.api.types.is_float_dtype(dtype):
        v = s.to_numpy(dtype=_numpy.float64, na_value=_numpy.nan)
        bad = ~_numpy.isfinite(v) | (v != _numpy.floor(v)) | (_numpy.abs(v) > 1e15)
        if bad.any():
            raise bad_value(first(bad))
        return _pandas.Series(v.astype(_numpy.int64).astype(str), index=s.index, dtype=object)
    if _pandas.api.types.is_object_dtype(dtype):
        if _pandas.api.types.infer_dtype(s, skipna=False) == "string":
            return s
        names = []
        for i, v in enumerate(s.to_numpy()):
            if isinstance(v, str):
                names.append(v)
            elif isinstance(v, (bool, _numpy.bool_)):
                raise bad_type
            elif isinstance(v, (int, _numpy.integer)):
                names.append(str(int(v)))
            elif isinstance(v, (float, _numpy.floating)):
                f = float(v)
                if not (_numpy.isfinite(f) and f == _numpy.floor(f) and abs(f) <= 1e15):
                    raise bad_value(i)
                names.append(str(int(f)))
            elif v is None or v is _pandas.NA:
                raise bad_value(i)
            else:
                raise bad_type
        return _pandas.Series(names, index=s.index, dtype=object)
    if _pandas.api.types.is_string_dtype(dtype):
        missing = s.isna().to_numpy()
        if missing.any():
            raise bad_value(first(missing))
        return s
    raise bad_type


def _with_chrom_names(df: _pandas.DataFrame) -> _pandas.DataFrame:
    """df with its chrom, chrom1 and chrom2 columns as chromosome names (_chrom_names); df
    itself when they hold names already."""
    out = df
    for col in _CHROM_COLS:
        if col in df.columns:
            values = df[col]
            names = _chrom_names(values, col)
            if names is not values:
                if out is df:
                    out = df.copy()
                out[col] = names
    return out


def _with_canonical_chroms(df: _pandas.DataFrame) -> _pandas.DataFrame:
    """df with its chromosome columns holding the database's names for the
    chromosomes (an alias, such as "1" for chr1, replaced; a number is a name,
    as in _chrom_names). A name the database does not know is kept, and all of
    them with no database."""
    out = df
    if _GROOT is None:
        return _with_chrom_names(df)
    for col in _CHROM_COLS:
        if col not in df.columns:
            continue
        raw = df[col]
        names = _chrom_names(raw, col)
        values = names.astype(str)
        uniq = values.unique().tolist()
        try:
            canonical = dict(zip(uniq, _pymisha.pm_normalize_chroms(uniq), strict=True))
        except _pymisha.error:
            canonical = {}
            for name in uniq:
                try:
                    canonical[name] = _pymisha.pm_normalize_chroms([name])[0]
                except _pymisha.error:
                    canonical[name] = name
        if names is raw and all(k == v for k, v in canonical.items()):
            continue
        if out is df:
            out = df.copy()
        out[col] = values.map(canonical)
    return out


def _chrom_sort_keys(chroms: Any) -> tuple[_numpy.ndarray, _numpy.ndarray]:
    """The keys that order chromosome names as misha orders them, for ``numpy.lexsort``
    (the primary key last): the chrom id, and the name for names the database does not
    know, which come after all its chromosomes. An alias (such as "1" for chr1) has its
    chromosome's id. In a per-chromosome database the ids follow R's collation of the
    names, not the order of the strings (chr1_KI270706v1_random sorts before chr10)."""
    codes, uniques = _pandas.factorize(_numpy.asarray(chroms, dtype=object).astype(str))
    uniques = _numpy.asarray(uniques, dtype=str)
    ids: _numpy.ndarray = _numpy.full(len(uniques), -1, dtype=_numpy.intp)
    if _GROOT is not None and len(uniques):
        ids = _numpy.asarray(_pymisha.pm_chrom_ids(uniques.tolist()), dtype=_numpy.intp)
    unknown = ids < 0
    name_rank = _numpy.zeros(len(uniques), dtype=_numpy.intp)
    order = _numpy.argsort(uniques[unknown], kind="stable")
    name_rank[_numpy.flatnonzero(unknown)[order]] = _numpy.arange(1, len(order) + 1)
    rank = _numpy.where(unknown, _numpy.iinfo(_numpy.intp).max, ids)
    return name_rank[codes], rank[codes]


def _chrom_sizes_names_at(root: Any) -> list[str]:
    """The chromosome names of chrom_sizes.txt in the database at root, as written."""
    with open(_os.path.join(str(root), "chrom_sizes.txt"), encoding="utf-8") as fh:
        return [line.split()[0] for line in fh if line.strip()]


def _gdb_check_genome_idx(groot: str, names: list[str] | None = None,
                          sizes: list[int] | None = None) -> None:
    """R misha's ``.gdb.check_genome_idx``: raises where gsetroot stops for the
    database's seq/genome.idx, and warns as it warns (``pm_gdb_check_genome_idx``)."""
    warning = _pymisha.pm_gdb_check_genome_idx(str(groot), names, sizes)
    if warning:
        _warnings.warn(warning, stacklevel=user_stacklevel())


def _has_own_sequence(groot: str, first_chrom: str) -> bool:
    """Whether the seq/ of the database at groot holds sequence: genome.idx and
    genome.seq, or the first chromosome's .seq file, with or without "chr"."""
    seq_dir = _os.path.join(groot, "seq")
    if all(_os.path.exists(_os.path.join(seq_dir, f)) for f in ("genome.idx", "genome.seq")):
        return True
    stripped = first_chrom[3:] if first_chrom.startswith("chr") else first_chrom
    return any(_os.path.exists(_os.path.join(seq_dir, f"{name}.seq"))
               for name in (first_chrom, f"chr{first_chrom}", stripped))


def _gdb_chrom_names_at(groot: str, from_disk: bool = False) -> list[str]:
    """R misha's ``.gdb.chrom_names_at``: the chromosome names of the database at
    groot by chrom id, as gdb_init gives them. An indexed track of the database is
    keyed by these ids. The loaded database and its loaded datasets take the session's
    order (a dataset loaded with an order of its own keeps it, ``_GDATASET_CHROMS``),
    unless ``from_disk``; any other database is read from disk without loading it, and
    one whose seq/ holds no sequence, so that its order cannot be told, is an error."""
    real = _os.path.realpath(str(groot))
    if not from_disk and _GROOT is not None:
        loaded = [_GROOT] + list(_GDATASETS)
        if real in {_os.path.realpath(p) for p in loaded}:
            own = _GDATASET_CHROMS.get(real)
            if own is not None:
                return list(own)
            return [str(c) for c in _pymisha.pm_intervals_all()[1]]
    cs = _os.path.join(str(groot), "chrom_sizes.txt")
    if not _os.path.exists(cs):
        raise ValueError(f"chrom_sizes.txt missing in {groot}")
    names, sizes, file_names, per_chromosome = _pymisha.pm_gdb_chrom_order(str(groot))
    # Names mostly without the "chr" prefix are sorted in a per-chromosome database and
    # kept in chrom_sizes.txt order in an indexed one; a seq/ with no sequence cannot tell
    # which. (file_names is by chrom id; chrom_sizes.txt's first name is the one probed.)
    with open(cs, encoding="utf-8") as fh:
        first = next((line.split()[0] for line in fh if line.strip()), None)
    if (not per_chromosome and file_names and first is not None
            and not _has_own_sequence(str(groot), first)
            and sum(not n.startswith("chr") for n in file_names) >= 0.8 * len(file_names)):
        raise ValueError(
            f"The chromosome order of {groot} cannot be told: its chrom_sizes.txt names lack "
            "the \"chr\" prefix, so its indexed tracks are keyed by the sorted names if it is a "
            "per-chromosome database and by the chrom_sizes.txt order if it is indexed, and its "
            "seq/ has no sequence to show which (as in a copy of a database's tracks without its "
            "seq/). Put back its seq/ (a link to the original database's seq/ will do), or load "
            "it with gdataset_load() as a dataset of its database."
        )
    # as gdb_init checks it: an indexed seq/ linked to a database converted since gives the
    # order of chrom_sizes.txt, not of the index its tracks would be read with
    _gdb_check_genome_idx(str(groot), list(names), list(sizes))
    return list(names)


def _refuse_dataset_order(what: str, name: str, db_root: str | None) -> None:
    """Stops writing a track or interval set of a dataset loaded with an order of its own
    (``_GDATASET_CHROMS``) in the indexed format, keyed by the session's chrom ids, as R's
    gtrack.convert_to_indexed and gintervals.convert_to_indexed do."""
    if db_root and _os.path.realpath(db_root) in _GDATASET_CHROMS:
        raise ValueError(
            f"{what} {name} is in a dataset that numbers the chromosomes differently from the "
            "working database, so it cannot be written in the indexed format in this session; "
            "load that database with gsetroot() to do it there."
        )


def _sort_by_chrom(df: _pandas.DataFrame, by: list[str]) -> _pandas.DataFrame:
    """df sorted (stably) by the columns ``by``, a chromosome column as misha orders
    chromosomes (``_chrom_sort_keys``)."""
    if len(df) < 2:
        return df.reset_index(drop=True)
    keys: list[Any] = []
    for col in reversed(by):
        if col in _CHROM_COLS:
            keys += list(_chrom_sort_keys(df[col]))
        else:
            keys.append(df[col].to_numpy())
    return df.iloc[_numpy.lexsort(keys)].reset_index(drop=True)


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
    # misha's order: by chrom id, not by the name (see _chrom_sort_keys)
    itr_order = _numpy.lexsort((itr_starts, *_chrom_sort_keys(itr_chroms)))
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
    sc_order = _numpy.lexsort((sc_starts, *_chrom_sort_keys(sc_chroms)))
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


def _canonic_scope(intervals):
    """Sort and merge a 1D scope, preserving the caller's chromosome labels.

    Same semantics as misha's ``GIntervals::unify_overlaps(true)``: overlapping
    AND touching intervals merge. Only ``chrom``/``start``/``end`` survive - a
    scope is a coordinate mask, and any extra column would be ambiguous once two
    rows become one.
    """
    df = _sort_by_chrom(intervals[["chrom", "start", "end"]], ["chrom", "start", "end"])
    out = []
    for chrom, grp in df.groupby("chrom", sort=False):
        cur_start, cur_end = None, None
        for start, end in zip(grp["start"].to_numpy(), grp["end"].to_numpy(), strict=True):
            if cur_end is not None and start <= cur_end:
                if end > cur_end:
                    cur_end = end
                continue
            if cur_end is not None:
                out.append((chrom, cur_start, cur_end))
            cur_start, cur_end = start, end
        if cur_end is not None:
            out.append((chrom, cur_start, cur_end))
    return _pandas.DataFrame(out, columns=["chrom", "start", "end"]).astype(
        {"start": intervals["start"].dtype, "end": intervals["end"].dtype}
    )


def _warn_overlapping_iterator(iterator, rows):
    """Warn when overlapping iterator intervals were merged into a returned row
    (R misha's warn_merged_iterator, TrackExpressionScanner.cpp).

    *rows* are the rows actually scanned: the scope cut by the merged iterator
    blocks. As in R, a merge is reported only when the caller can see it: a row
    no single iterator interval contains (two intervals became one row), or an
    interval the scope reaches that contains no row (swallowed by a bigger one).
    So iterator=ivs with intervals=ivs, where the scope clips every row back to
    its interval, stays silent, and exact duplicates do too.
    """
    if len(iterator) < 2 or len(rows) == 0:
        return
    itr = iterator[["chrom", "start", "end"]]
    n_blocks = 0
    found = None  # (row, block chrom, block rows)
    for chrom, g in itr.groupby("chrom", sort=False, observed=True):
        s = g["start"].to_numpy(_numpy.int64)
        e = g["end"].to_numpy(_numpy.int64)
        o = _numpy.lexsort((e, s))
        s, e = s[o], e[o]
        pmax = _numpy.maximum.accumulate(e)
        n_blocks += 1 + int(_numpy.sum(s[1:] >= pmax[:-1]))  # touching intervals stay apart
        if found is not None:
            continue
        r = rows[rows["chrom"] == chrom]
        if len(r) == 0:
            continue
        rs = r["start"].to_numpy(_numpy.int64)
        re_ = r["end"].to_numpy(_numpy.int64)
        ro = _numpy.argsort(rs, kind="stable")
        rs, re_ = rs[ro], re_[ro]
        # (a) a row no single interval contains
        k = _numpy.searchsorted(s, rs, side="right") - 1
        covered = (k >= 0) & (pmax[_numpy.maximum(k, 0)] >= re_)
        if not covered.all():
            j = int(_numpy.flatnonzero(~covered)[0])
            found = (chrom, int(rs[j]), int(re_[j]))
            continue
        # (b) an interval the scope reaches (it overlaps a row) that contains no row
        rmax = _numpy.maximum.accumulate(re_)
        rmin_suffix = _numpy.minimum.accumulate(re_[::-1])[::-1]
        before = _numpy.searchsorted(rs, e, side="left") - 1
        reached = (before >= 0) & (rmax[_numpy.maximum(before, 0)] > s)
        first = _numpy.searchsorted(rs, s, side="left")
        verbatim = (first < len(rs)) & (rmin_suffix[_numpy.minimum(first, len(rs) - 1)] <= e)
        bad = _numpy.flatnonzero(reached & ~verbatim)
        if bad.size:
            i = int(bad[0])
            # the row that swallowed it: the first one it overlaps
            j = int(_numpy.flatnonzero((rs < e[i]) & (re_ > s[i]))[0])
            found = (chrom, int(rs[j]), int(re_[j]))

    if found is None:
        return
    chrom, r0, r1 = found
    g = itr[itr["chrom"] == chrom]
    s = g["start"].to_numpy(_numpy.int64)
    e = g["end"].to_numpy(_numpy.int64)
    o = _numpy.lexsort((e, s))
    s, e = s[o], e[o]
    # the first overlapping pair among the intervals that reach the row
    pair = None
    imax = None
    for i in _numpy.flatnonzero((s < r1) & (e > r0)):
        if imax is not None and s[i] < e[imax]:
            pair = (imax, i)
            break
        if imax is None or e[i] > e[imax]:
            imax = i
    if pair is None:
        return
    a, b = pair
    _warnings.warn(
        f"Overlapping intervals were used as an iterator: {len(itr)} intervals "
        f"were merged into {n_blocks} non-overlapping "
        f"block{'' if n_blocks == 1 else 's'}. For example {chrom} {s[a]}-{e[a]} and "
        f"{chrom} {s[b]}-{e[b]} are reported as a single row {chrom} {r0}-{r1}, so the "
        f"results do not correspond one-to-one to the iterator intervals. To get one value "
        f"per interval, call gextract() with the same intervals as the scope (its "
        f"'intervals' argument) and map the rows back with the intervalID column.",
        stacklevel=4,
    )


def _verify_iterator_dim(intervals, iterator_is_2d):
    """Refuse an iterator whose dimension is not that of the intervals, with
    R misha's messages (TrackExprScanner::verify_1d_iter / verify_2d_iter).
    Empty intervals are not checked, as in R."""
    if not isinstance(intervals, _pandas.DataFrame) or len(intervals) == 0:
        return
    intervals_are_2d = "chrom1" in intervals.columns
    if iterator_is_2d and not intervals_are_2d:
        raise ValueError("2D iterator is used along with 1D intervals")
    if not iterator_is_2d and intervals_are_2d:
        raise ValueError("1D iterator is used along with 2D intervals")


def _refuse_unsupported_2d_iterator(exprs, intervals, iterator):
    """gintervals_quantiles and gintervals_mapply take, of the 2D iterators,
    only a rectangle size: R refuses a 2D track, 2D intervals or a cartesian
    grid (GenomeTrackQuantiles.cpp, GenomeTrackApply.cpp), after checking
    that the iterator and the intervals have the same dimension."""
    from .expr import _expr_is_2d

    if not isinstance(intervals, _pandas.DataFrame) or len(intervals) == 0:
        return
    iterator_is_2d = _expr_is_2d(exprs, iterator)
    _verify_iterator_dim(intervals, iterator_is_2d)
    if iterator_is_2d and not (isinstance(iterator, (tuple, list)) and len(iterator) == 2):
        raise ValueError("The type of iterator is currently not supported by the function")


def _preprocess_intervals_iterator(intervals, iterator, canonic_scope=False):
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
                _verify_iterator_dim(intervals, True)
                return intervals, iterator, None
            iterator = loaded
        else:
            # Maybe a track name used as the iterator (R parity): a 1D dense
            # track iterates over its fixed bins, a 1D sparse/array track over
            # its own intervals. 2D tracks are left as a string for the 2D
            # iterator path downstream.
            from .tracks import gtrack_exists, gtrack_info

            if gtrack_exists(iterator):
                info = gtrack_info(iterator)
                track_is_2d = int(info.get("dimensions", 1) or 1) == 2
                _verify_iterator_dim(intervals, track_is_2d)
                if track_is_2d:
                    return intervals, iterator, None  # 2D track iterator path
                bin_size = info.get("bin_size") or info.get("bin.size")
                if bin_size is not None:
                    # Dense track -> iterate over its fixed bins.
                    return intervals, int(float(bin_size)), None
                # Sparse / array track -> iterate over its own intervals.
                from .extract import gextract

                track_iv = gextract(iterator, intervals)
                if track_iv is None or len(track_iv) == 0:
                    empty = _pandas.DataFrame(
                        {"chrom": _pandas.Series([], dtype=str),
                         "start": _pandas.Series([], dtype=int),
                         "end": _pandas.Series([], dtype=int)})
                    return empty, -1, _numpy.array([], dtype=int)
                iterator = track_iv[["chrom", "start", "end"]]
            else:
                # Not an interval set or track — let downstream handle it.
                return intervals, iterator, None

    if not isinstance(iterator, _pandas.DataFrame):
        from ._iterator_policy import CartesianGridSpec

        if isinstance(iterator, CartesianGridSpec) or (
            isinstance(iterator, (tuple, list)) and len(iterator) == 2
        ):
            _verify_iterator_dim(intervals, True)  # a rectangle size or a cartesian grid
        elif isinstance(iterator, (int, float, _numpy.integer, _numpy.floating)) and not isinstance(iterator, bool):
            _verify_iterator_dim(intervals, False)  # a bin size
        return intervals, iterator, None

    # 2D iterator DataFrame: leave for the 2D extraction path to handle.
    _verify_iterator_dim(intervals, "chrom1" in iterator.columns)
    if "chrom1" in iterator.columns:
        return intervals, iterator, None
    _check_intervals_strand(iterator)   # read as intervals, as R reads them


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

    # The intersection matches the chromosomes by name: name both by the
    # database's names, as misha does (an alias such as "1" for chr1 is accepted).
    iterator = _with_canonical_chroms(iterator)
    if isinstance(intervals, _pandas.DataFrame) and "chrom" in intervals.columns:
        intervals = _with_canonical_chroms(intervals)

    # Canonicalise the SCOPE before intersecting, never the iterator. The C++
    # entry points skip their own unify when they receive the intersection
    # (iterator_policy == -1), because by then these are bins, not a scope -
    # so this is the only place an overlapping scope gets merged on this path.
    # R parity: an overlapping or duplicated scope with a two-bin touching
    # iterator gives 2 in misha, exactly as a canonical scope does.
    #
    # Merged here in pandas rather than by gintervals_canonic(), which round-trips
    # through the C++ chromkey.
    # Opt-in, because it is only correct for the aggregating callers. misha
    # canonicalises the scope in gsummary/gquantiles/gcor/gscreen/gsample/
    # gsegment/gwilcox/gpartition/gdist/giterator.intervals and NOT in gextract,
    # gintervals.summary or gintervals.quantiles, whose output is per scope
    # interval - merging there would drop rows the caller asked for. gextract
    # shares this preprocessing, so defaulting to True silently changed it: a
    # duplicated scope returned 2 rows where misha returns 4.
    if canonic_scope and len(intervals) > 1:
        intervals = _canonic_scope(intervals)

    new_intervals, id_map = _intersect_scope_with_iterator(intervals, iterator)
    _warn_overlapping_iterator(iterator, new_intervals)
    if len(new_intervals) == 0:
        empty = _pandas.DataFrame({"chrom": _pandas.Series([], dtype=str),
                                    "start": _pandas.Series([], dtype=int),
                                    "end": _pandas.Series([], dtype=int)})
        return empty, -1, _numpy.array([], dtype=int)
    return new_intervals, -1, id_map


@_contextlib.contextmanager
def _config_no_mt(id_map, keep=False):
    """Context manager that disables C++ multitasking when id_map is set.

    When a DataFrame iterator was preprocessed into a small set of
    intersected intervals, the C++ fork/FIFO multitasking overhead
    dominates the actual work.  The C++ side reads multitasking from
    the module-level ``CONFIG`` global, so we temporarily mutate it.

    ``keep=True`` leaves multitasking enabled: the caller has already
    established that the workload is big enough to fork (the
    track-parallel gextract path).  Without this escape hatch a
    DataFrame iterator forced *every* extraction serial, which on cold
    NFS costs ~25x on many-track scans - one process servicing page
    faults where R misha forks one kid per expression group.
    """
    if id_map is None or keep:
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


def _iterated_intervals(intervals, iterator, *, unlimited=False):
    """Return intervals after applying iterator policy (includes intervalID).

    *unlimited*: for the streaming summaries, which read these coordinates in
    chunks; max_data_size limits results, not their working set."""
    if unlimited:
        # the C++ side reads max_data_size from the global CONFIG
        absent = "max_data_size" not in CONFIG
        prev = CONFIG.get("max_data_size")
        CONFIG["max_data_size"] = None
        try:
            return _iterated_intervals(intervals, iterator)
        finally:
            if absent:
                CONFIG.pop("max_data_size", None)
            else:
                CONFIG["max_data_size"] = prev
    cfg = CONFIG
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
            _df2pymisha(new_intervals), new_iterator, cfg
        )
        out = _pymisha2df(result)
        return _remap_interval_ids(out, id_map)
    if iterator is None:
        out = intervals.copy()
        out["intervalID"] = _numpy.arange(1, len(out) + 1)
        return out
    result = _pymisha.pm_iterate(_df2pymisha(intervals), iterator, cfg)
    return _pymisha2df(result)


def _chunk_slices(n, chunk_size):
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    return [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]


def _gwd_is_subdir() -> bool:
    """Whether the working directory (gdir_cd) is a subdirectory: not the tracks/ of the
    working database, the user root or a loaded dataset. Names are then relative to it, and
    only those under it are listed (R misha's ``!is.null(.gdb.gwd_db())``)."""
    return bool(_pymisha.pm_gwd_is_subdir())


def _path_db(path: Any) -> str | None:
    """R misha's ``.gdb.path_db``: the loaded database a path is in, the one whose tracks/
    holds it, the most specific one when several do (comparing the directories with links
    resolved), or None when none does or when a database root (a directory with
    chrom_sizes.txt and tracks/, loaded or not) lies between the path and that tracks/."""
    groots = [g for g in [_GROOT, *_GDATASETS] if g]
    tracks_dirs = [_os.path.realpath(_os.path.join(g, "tracks")) for g in groots]
    # links are followed in the part of the path that exists (a new track's directory does not yet)
    path = _os.path.abspath(str(path))
    rest: list[str] = []
    while not _os.path.exists(path) and _os.path.dirname(path) != path:
        rest.insert(0, _os.path.basename(path))
        path = _os.path.dirname(path)
    path = _os.path.join(_os.path.realpath(path), *rest)
    within = [i for i, t in enumerate(tracks_dirs) if path == t or path.startswith(t + "/")]
    if not within:
        return None
    best = max(within, key=lambda i: len(tracks_dirs[i]))
    while len(path) > len(tracks_dirs[best]):
        if _os.path.exists(_os.path.join(path, "chrom_sizes.txt")) and _os.path.isdir(_os.path.join(path, "tracks")):
            return None
        path = _os.path.dirname(path)
    return groots[best]


def _gwd_db() -> str | None:
    """R misha's ``.gdb.gwd_db``, for a working directory that is a subdirectory
    (``_gwd_is_subdir``): the loaded database it is in (``_path_db``), None if none."""
    return _path_db(_GWD)


def _same_order(db: str | None) -> bool:
    """R misha's ``.gdb.same_order``: whether the session's chrom ids are those of loaded
    database db. The working database's are, and a dataset's only when its order can be
    read now and is the same (as gdataset_load compares them); None or an unreadable order
    is not."""
    if db is None:
        return False
    if db == _GROOT:
        return True
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            chroms = _gdb_chrom_names_at(db, from_disk=True)
    except Exception:
        return False

    def strip(names: list[str]) -> list[str]:
        return [n[3:] if n.startswith("chr") else n for n in names]

    assert _GROOT is not None
    return strip(chroms) == strip(_gdb_chrom_names_at(_GROOT))


def _names_dir(db: str | None = None) -> str:
    """R misha's ``.gdb.names_dir``: the directory the names of database db are relative
    to: the working directory for None, and for the database it is in when it is a
    subdirectory (gdir_cd); db's tracks/ otherwise."""
    assert _GWD is not None
    if db is None or (_gwd_is_subdir() and _same_path(db, _gwd_db())):
        return _GWD
    return _os.path.join(db, "tracks")


def _same_path(a: Any, b: Any) -> bool:
    """Whether two paths are the same directory, links resolved (None is no path)."""
    return a is not None and b is not None and _os.path.realpath(str(a)) == _os.path.realpath(str(b))


def _check_flag(value: Any, name: str) -> Any:
    """A TRUE/FALSE argument: a NaN or pd.NA is R's NA, an error as in R. Returned
    unchanged otherwise (None included)."""
    if value is _pandas.NA or (isinstance(value, (float, _numpy.floating)) and _numpy.isnan(value)):
        raise ValueError(f"{name} argument is not logical")
    return value


_STR2STRAND = {"+": 1, "-": -1, ".": 0, "*": 0, "": 0}


def _bad_strand_label(col: _pandas.Series) -> tuple[int, str] | None:
    """The first row (0-based) of a strand column of labels (strings or a Categorical,
    read by its labels) whose label R's GInterval::str2strand rejects, and the label."""
    if not (isinstance(col.dtype, _pandas.CategoricalDtype) or col.dtype == object
            or _pandas.api.types.is_string_dtype(col)):
        return None
    values = col.astype(object).to_numpy()
    codes, uniques = _pandas.factorize(values)
    bad = [i for i, v in enumerate(uniques) if isinstance(v, str) and v not in _STR2STRAND]
    if not bad:
        return None
    row = int(_numpy.flatnonzero(_numpy.isin(codes, bad))[0])
    return row, values[row]


def _check_strand_column(col: _pandas.Series, prefix: str = "") -> None:
    """R's checks of a strand column, as the C++ interval converters make them
    (check_strand_column): a boolean column (a categorical of booleans too) is not a
    strand column, and the first row with a missing strand (None, NaN, pandas.NA), a
    number other than -1, 0 or 1, or a label other than "+", "-", ".", "*" or "" is an
    error. *prefix* starts the messages R prefixes (gcis_decay's "Source intervals: ")."""
    if _pandas.api.types.is_bool_dtype(col) or (
            col.dtype == object and any(isinstance(v, (bool, _numpy.bool_)) for v in col.to_numpy())):
        raise _pymisha.error(f"{prefix}Invalid format of intervals argument")
    na = col.isna().to_numpy()
    if _pandas.api.types.is_numeric_dtype(col):
        # a number is -1, 0 or 1; a missing float is a bad interval, as R says of NA_real_
        v = col.to_numpy(dtype=float, na_value=_numpy.nan)
        bad_rows = na | ~_numpy.isin(v, (-1.0, 0.0, 1.0))
        if bad_rows.any():
            i = int(_numpy.argmax(bad_rows))
            if not na[i]:
                raise _pymisha.error(f"{prefix}Invalid strand value {v[i]:g} at index {i + 1}")
            if _pandas.api.types.is_float_dtype(col):
                raise _pymisha.error(f"{prefix}Invalid format of interval at index {i + 1}")
            raise _pymisha.error(f"{prefix}Invalid strand value (NA) at index {i + 1}")
        return
    first_na = int(_numpy.argmax(na)) if na.any() else len(col)
    bad = _bad_strand_label(col)
    if col.dtype == object:
        # a number among the labels is -1, 0 or 1 too
        for i, x in enumerate(col.to_numpy()[:first_na]):
            if bad is not None and i == bad[0]:
                break
            if isinstance(x, (int, float, _numpy.number)) and x not in (-1, 0, 1):
                raise _pymisha.error(f"{prefix}Invalid strand value {float(x):g} at index {i + 1}")
    if bad is not None and bad[0] < first_na:
        raise _pymisha.error(f'Bad strand value "{bad[1]}" (expected "+", "-", ".", "*" or "")')
    if first_na < len(col):
        raise _pymisha.error(f"{prefix}Invalid strand value (NA) at index {first_na + 1}")


def _check_intervals_strand(intervals: Any, prefix: str = "") -> None:
    """_check_strand_column on the strand column of 1D intervals, as R reads every
    intervals argument; anything else is left alone."""
    if (isinstance(intervals, _pandas.DataFrame) and "strand" in intervals.columns
            and "chrom1" not in intervals.columns):
        _check_strand_column(intervals["strand"], prefix)


def _str2strand_values(col: _pandas.Series) -> Any:
    """A strand column of labels (strings or a Categorical, read by its labels) as R
    reads it (GInterval::str2strand): "+" is 1, "-" is -1, ".", "*" and "" are 0, any
    other label is an error. A number among the labels is kept. Returns None for a
    column of numbers."""
    if not (isinstance(col.dtype, _pandas.CategoricalDtype) or col.dtype == object
            or _pandas.api.types.is_string_dtype(col)):
        return None
    bad = _bad_strand_label(col)
    if bad is not None:
        raise _pymisha.error(f'Bad strand value "{bad[1]}" (expected "+", "-", ".", "*" or "")')
    codes, uniques = _pandas.factorize(col.astype(object).to_numpy())
    values = [_STR2STRAND[v] if isinstance(v, str) else int(v) for v in uniques]
    return _numpy.asarray(values, dtype=_numpy.int64)[codes]


def _check_breaks_na(breaks: _numpy.ndarray) -> None:
    """Breaks with a NaN (an NA in R) are an error, as R's BinFinder makes them."""
    nan = _numpy.flatnonzero(_numpy.isnan(breaks))
    if nan.size:
        raise ValueError(f"Breaks contain NA (break[{nan[0]}])")


def _check_breaks(breaks: _numpy.ndarray) -> None:
    """R's BinFinder checks of breaks, with its errors: at least 2 of them, none NA,
    strictly increasing."""
    if len(breaks) < 2:
        raise ValueError(f"Invalid number of breaks {len(breaks)}")
    _check_breaks_na(breaks)
    bad = _numpy.flatnonzero(breaks[1:] <= breaks[:-1])
    if bad.size:
        i = int(bad[0]) + 1
        if breaks[i] == breaks[i - 1]:
            raise ValueError(f"Breaks are not unique (break[{i - 1}]=break[{i}]={breaks[i]:g})")
        raise ValueError(f"Breaks are not sorted (break[{i - 1}]={breaks[i - 1]:g}, break[{i}]={breaks[i]:g})")


def _check_spat_bounds(
    spat_min: Any, spat_max: Any, motif_len: int
) -> tuple[float | None, float | None]:
    """R misha's checks on spat_min/spat_max, for the pwm virtual tracks and gseq_pwm.

    Each is a single 1-based position >= 1. A window spat_min..spat_max shorter
    than the motif can never hold it, so it is an error too, with spat_min taken
    as 1 when only spat_max is given. Returns the two as floats (None if not
    given); the caller converts them as R does.
    """
    out: list[float | None] = []
    for key, v in (("spat_min", spat_min), ("spat_max", spat_max)):
        if v is None:
            out.append(None)
            continue
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float, _numpy.number))
            or _numpy.isnan(v)
            or v < 1
        ):
            raise ValueError(f"{key} must be a single 1-based position (>= 1)")
        # R's as.integer() turns these into NA with a warning
        if v > 2**31 - 1:
            raise ValueError(f"{key} is outside the 32-bit integer range")
        out.append(float(v))
    smin, smax = out
    if smax is not None and smax - (1 if smin is None else smin) + 1 < motif_len:
        raise ValueError(
            f"the spat_min..spat_max window is shorter than the motif ({motif_len} bp), "
            "so no position can be scored"
        )
    return smin, smax


def _coerce_score_thresh(value: Any) -> float:
    """Port of R misha's ``.coerce_score_thresh`` (``R/pssm_utils.R``).

    One number, or a string spelling of one. Anything else - a sequence, a
    bool, an uncoercible string, NaN - is an error rather than a silent 0.0,
    because every ``>= nan`` comparison is false and a defaulted threshold is
    indistinguishable from one that is simply out of reach.
    """
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        try:
            out = float(text)
        except ValueError:
            raise ValueError(
                "score_thresh must be a single number, or something coercible "
                f"to one; {value!r} is not."
            ) from None
    elif isinstance(value, bool):
        raise ValueError(
            "score_thresh must be a single number, or a character spelling of "
            "one, not a bool."
        )
    elif isinstance(value, (int, float, _numpy.number)):
        out = float(value)
    else:
        try:
            items = list(value)
        except TypeError:
            raise ValueError(
                "score_thresh must be a single number, or a character spelling "
                f"of one, not a {type(value).__name__}."
            ) from None
        if len(items) == 0:
            raise ValueError(
                "score_thresh must be a single value, and this one is empty."
            )
        if len(items) != 1:
            raise ValueError(
                f"score_thresh must be a single value, got a sequence of length "
                f"{len(items)}. Only one threshold is used, so pass the single "
                "value you mean."
            )
        return _coerce_score_thresh(items[0])

    if out != out:  # NaN
        raise ValueError(
            "score_thresh must be a single number, or something coercible to "
            "one; NaN is not."
        )
    return out


_POTTS_BASES = ("A", "C", "G", "T")
# FLT_MAX, spelled out: the scorer reports in single precision.
_FLT_MAX = 3.4028234663852886e38


def _potts_e_matrix(e: Any, what: str) -> _numpy.ndarray:
    """``e`` as a (W, 4) float array with columns in A, C, G, T order."""
    if isinstance(e, _pandas.DataFrame):
        cols = {str(c).upper(): c for c in e.columns}
        if not all(b in cols for b in _POTTS_BASES):
            raise ValueError(f"{what}: 'e' must have columns named A, C, G, T")
        e = e[[cols[b] for b in _POTTS_BASES]].to_numpy(dtype=float)
    arr = _numpy.asarray(e, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"{what}: 'e' must be a numeric matrix with 4 columns (A, C, G, T)"
        )
    return _numpy.ascontiguousarray(arr)


def _orient_potts_J(jk: Any, k: int, what: str) -> _numpy.ndarray:
    """Reorder a labelled 4x4 J block into A, C, G, T on both axes.

    A J block's rows are indexed by the lower position's base and its columns
    by the higher position's, both in A, C, G, T order. ``e`` is reordered by
    column NAME, so a labelled J block has to be reordered too: reading a
    relabelled block positionally is a wrong score with no warning, and a user
    who has just seen ``e`` accept labelled columns has every reason to label J
    as well. An unlabelled block is positional, as before.
    """
    if not isinstance(jk, _pandas.DataFrame):
        return _numpy.asarray(jk, dtype=float)
    rows = [str(x).upper() for x in jk.index]
    cols = [str(x).upper() for x in jk.columns]
    if set(rows) != set(_POTTS_BASES) or set(cols) != set(_POTTS_BASES):
        raise ValueError(
            f"{what}: 'J' table {k} is labelled, so both its rows and its "
            "columns must be named A, C, G, T in some order"
        )
    out = jk.copy()
    out.index = rows
    out.columns = cols
    ordered: _numpy.ndarray = out.loc[
        list(_POTTS_BASES), list(_POTTS_BASES)
    ].to_numpy(dtype=float)
    return ordered


def _coerce_potts_model(model: Any, what: str) -> dict[str, Any]:
    """Port of R misha's ``.coerce_potts_model`` (``R/pssm_utils.R``).

    One validator for both Potts entry points - ``gseq_potts`` and the potts
    virtual tracks. It returns the model in the exact shape the C++ parser
    takes: ``e`` a (W, 4) float array, ``J`` an (npair, 16) float array whose
    rows are the 4x4 blocks flattened in Fortran order, ``pairs`` an (npair, 2)
    int array (1-based), ``intercept`` a float.

    It is deliberately tolerant of the extra fields a fitted Potts model
    carries (``width``, ``pair_strength``, ``attr``, ``link``) so that a fitted
    model can be passed verbatim, and it cross-checks ``width`` against the row
    count of ``e`` when present, which catches a hand-edited model.
    """
    if not isinstance(model, dict):
        raise ValueError(
            f"{what}: `model` must be a dict with 'e', 'J', 'pairs' and "
            f"'intercept'; it is a {type(model).__name__}"
        )
    if "e" not in model:
        raise ValueError(
            f"{what}: `model` requires an 'e' matrix - one row per position, "
            "columns A, C, G, T"
        )

    e = _potts_e_matrix(model["e"], what)
    if e.shape[0] < 1:
        raise ValueError(f"{what}: 'e' has no rows, so the model covers no positions")
    if not _numpy.isfinite(e).all():
        n_bad = int((~_numpy.isfinite(e)).sum())
        raise ValueError(f"{what}: 'e' has {n_bad} non-finite entries")
    w = int(e.shape[0])

    width = model.get("width")
    if width is not None:
        try:
            width_ok = int(width) == w
        except (TypeError, ValueError):
            width_ok = False
        if not width_ok:
            raise ValueError(f"{what}: 'width' says {width} but 'e' has {w} rows")

    pairs_in = model.get("pairs")
    # Declared unparameterised on purpose. The two branches below produce
    # different shape types - zeros((0, 2)) is ndarray[tuple[int, int], ...]
    # while ascontiguousarray() is ndarray[tuple[int, ...], ...] - and how
    # precisely numpy's stubs express that varies by numpy version, so the
    # 3.10 CI leg rejected the assignment while 3.11 and 3.12 accepted it.
    # Binding the name once, to any ndarray, is stub-agnostic.
    pairs: _numpy.ndarray
    if pairs_in is None:
        pairs = _numpy.zeros((0, 2), dtype=_numpy.int64)
    else:
        if isinstance(pairs_in, _pandas.DataFrame):
            pairs_in = pairs_in.to_numpy()
        pairs_arr = _numpy.asarray(pairs_in)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(
                f"{what}: 'pairs' must be a numeric matrix with 2 columns, one "
                "row per coupling"
            )
        # Before the integer cast, which truncates: 2.9 with W = 2 is both
        # fractional AND out of range, and the range check below would validate
        # the truncated 2 and pass it. Every other index here is strict.
        as_float = _numpy.asarray(pairs_arr, dtype=float)
        if as_float.size and (
            not _numpy.isfinite(as_float).all()
            or not (as_float == _numpy.trunc(as_float)).all()
        ):
            raise ValueError(f"{what}: every 'pairs' index must be a whole number")
        pairs = _numpy.ascontiguousarray(pairs_arr.astype(_numpy.int64))
    npair = int(pairs.shape[0])

    if npair:
        if pairs.min() < 1 or pairs.max() > w:
            raise ValueError(f"{what}: every 'pairs' index must be in 1..{w}")
        bad = _numpy.flatnonzero(pairs[:, 0] >= pairs[:, 1])
        if bad.size:
            raise ValueError(
                f"{what}: every row of 'pairs' must have the lower position "
                f"first (i < j); row {int(bad[0]) + 1} does not"
            )
        seen = {(int(a), int(b)) for a, b in pairs}
        if len(seen) != npair:
            raise ValueError(
                f"{what}: 'pairs' names the same pair of positions twice; a "
                "pair carries one coupling table"
            )

    j_in = model.get("J")
    if j_in is None:
        j_in = []
    if isinstance(j_in, (list, tuple)):
        if len(j_in) != npair:
            raise ValueError(
                f"{what}: 'J' holds {len(j_in)} tables and 'pairs' names {npair}"
            )
        rows = []
        for k, jk in enumerate(j_in):
            block = _orient_potts_J(jk, k + 1, what)
            if block.shape != (4, 4):
                raise ValueError(
                    f"{what}: 'J' table {k + 1} must be a 4 x 4 numeric matrix"
                )
            if not _numpy.isfinite(block).all():
                raise ValueError(f"{what}: 'J' table {k + 1} has non-finite entries")
            # Fortran-order flatten puts element [a, b] at b * 4 + a, which is
            # what the C++ kernel indexes.
            rows.append(block.flatten(order="F"))
        j_flat = (
            _numpy.vstack(rows) if rows else _numpy.zeros((0, 16), dtype=float)
        )
    else:
        j_flat = _numpy.asarray(j_in, dtype=float)
        # every shape checked, also with no pairs (R's ncol test short-circuited there)
        if j_flat.ndim != 2 or j_flat.shape[0] != npair or j_flat.shape[1] != 16:
            raise ValueError(f"{what}: a flat 'J' must be an {npair} x 16 numeric matrix")
        if not _numpy.isfinite(j_flat).all():
            raise ValueError(f"{what}: 'J' has non-finite entries")
        j_flat = j_flat.reshape(npair, 16)
    j_flat = _numpy.ascontiguousarray(j_flat, dtype=float)

    intercept_in = model.get("intercept")
    intercept = 0.0 if intercept_in is None else intercept_in
    if isinstance(intercept, bool) or not isinstance(
        intercept, (int, float, _numpy.number)
    ):
        raise ValueError(f"{what}: 'intercept' must be a single finite number")
    intercept = float(intercept)
    if not _numpy.isfinite(intercept):
        raise ValueError(f"{what}: 'intercept' must be a single finite number")

    # Per-entry finiteness is not enough. The scorer accumulates a window in
    # double but reports in float - PottsScorer::score_interval() returns
    # float, and its sliding aggregators hold float - so a model whose windows
    # can leave the single-precision range makes its two scoring paths answer
    # differently: the direct one returns an infinity, the sliding one maps a
    # non-finite aggregate to NaN, and the whole point of having two paths is
    # that they agree.
    #
    # Bounded here rather than checked per window in C++, so that
    # PottsScorer::slid_answer() and anchor_value() can go on reading -inf as
    # "no anchor was scorable" instead of "the arithmetic overflowed", and so
    # that both entry points get it from the one validator they share.
    # The tight bound, as R: a window takes one energy per position and one
    # coupling per pair, so sum each row's largest magnitude (W * max|e|
    # overstated it by orders of magnitude and refused usable models).
    worst = float(_numpy.abs(e).max(axis=1).sum()) + abs(intercept)
    if npair:
        worst += float(_numpy.abs(j_flat).max(axis=1).sum())
    if not _numpy.isfinite(worst) or worst > _FLT_MAX:
        raise ValueError(
            f"{what}: the model's largest possible window score is {worst:.3g}, "
            "which overflows the single-precision range potts scores are "
            "reported in (about 3.4e38) - a scored window would come back as "
            "Inf. Scale 'e', 'J' and 'intercept' down by a common factor, and "
            "'score_thresh' with them, or drop the entries that make the bound "
            "this large."
        )

    return {"e": e, "J": j_flat, "pairs": pairs, "intercept": intercept}


def _potts_params(
    model: Any,
    bidirect: bool = True,
    extend: bool = True,
    strand: int = 1,
    score_thresh: float = 0.0,
    what: str = "potts",
) -> dict[str, Any]:
    """The C++-ready parameter dict, built one way for both entry points.

    Its shape is what ``PMPottsParams::parse()`` reads, so a field renamed here
    must be renamed there; that is the price of having one parser instead of
    two.
    """
    p = _coerce_potts_model(model, what)

    if not isinstance(bidirect, (bool, _numpy.bool_)):
        raise ValueError(f"{what}: bidirect must be True or False")
    if not isinstance(extend, (bool, _numpy.bool_)):
        raise ValueError(f"{what}: extend must be True or False")
    if isinstance(strand, bool) or not isinstance(strand, (int, float, _numpy.number)):
        raise ValueError(f"{what}: strand must be 1 or -1")
    strand = int(strand)
    if strand not in (1, -1):
        raise ValueError(f"{what}: strand must be 1 or -1")
    if isinstance(score_thresh, bool) or not isinstance(
        score_thresh, (int, float, _numpy.number)
    ):
        raise ValueError(f"{what}: score_thresh must be a single number")
    score_thresh = float(score_thresh)
    if score_thresh != score_thresh:  # NaN
        raise ValueError(f"{what}: score_thresh must be a single number")

    # Under bidirect both strands are scored at every anchor, so `strand`
    # cannot change which anchors are scored or what any of them scores. All it
    # changes is the orientation the scan target is fetched in, and hence the
    # ORDER the anchors are visited in - and the *.pos functions break an exact
    # tie by keeping the first anchor they see. So an unclamped strand = -1
    # reports a different, equally maximal, position for the same interval
    # wherever two anchors tie, which a repeat region supplies readily. Clamped
    # here rather than at either entry point so both get it: the pwm family
    # clamps the same combination, for the same reason.
    if bidirect:
        strand = 1

    return {
        "e": p["e"],
        "J": p["J"],
        "pairs": p["pairs"],
        "intercept": p["intercept"],
        "bidirect": bool(bidirect),
        "extend": bool(extend),
        "strand": strand,
        "score_thresh": score_thresh,
    }
