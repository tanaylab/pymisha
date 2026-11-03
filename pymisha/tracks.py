"""Track listing, metadata, and creation/import helpers."""

from __future__ import annotations

import bz2
import contextlib
import datetime as _datetime
import errno
import fnmatch
import ftplib
import getpass as _getpass
import glob
import gzip
import io
import lzma
import math
import numbers
import os
import re
import secrets
import shutil
import struct
import tempfile
import warnings
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import _shared
from ._db_trash import _gdb_trash, _gdb_trash_sweep_old
from ._log import PymishaWarning, get_logger, user_stacklevel
from ._name_validation import validate_dotted_name
from ._safe_pickle import restricted_load, restricted_loads
from ._shared import (
    _checkroot,
    _config_no_mt,
    _df2pymisha,
    _pm_dbreload,
    _preprocess_intervals_iterator,
    _pymisha,
    _track_names_set,
)
from ._types import Intervals, NumpyArray

_logger = get_logger(__name__)

# bgzip magic: gzip (1f 8b) with FLG byte 0x04 (FEXTRA set - bgzip block-size
# subfield). Plain gzip files have FLG=0x00 or 0x08, so byte 4 reliably
# distinguishes BAM from other gzip-compressed files.
_BGZF_MAGIC = b"\x1f\x8b\x08\x04"


def _is_bam_file(path: str, payload: bool = False) -> bool:
    """Return True if path begins with bgzip magic bytes (BAM detection).

    Works even when the file is not named .bam. With ``payload=True`` the
    decompressed stream must also start with ``BAM\\1``, which tells a BAM
    from bgzipped text (SAM, 10x fragments.tsv.gz). An empty file or a FIFO
    is not read (reading a FIFO would consume its stream). Returns False on
    any OSError (missing file, permission denied) so callers do not need to
    guard.
    """
    try:
        if os.stat(path).st_size == 0:
            return False
        with open(path, "rb") as fh:
            head = fh.read(4)
        if head != _BGZF_MAGIC:
            return False
        if not payload:
            return True
        with gzip.open(path, "rb") as gz:
            return gz.read(4) == b"BAM\x01"
    except (OSError, EOFError):
        return False


def _load_track_attributes(track_name: str) -> dict[str, str]:
    """
    Load track attributes from .attributes file (binary) or .attributes.yaml.
    """
    track_path = _pymisha.pm_track_path(track_name)
    if not track_path:
        return {}

    attrs = {}

    bin_path = os.path.join(track_path, ".attributes")
    if os.path.exists(bin_path):
        try:
            with open(bin_path, 'rb') as f:
                data = f.read()
            raw_parts = data.split(b'\x00')
            str_parts: list[str] = [p.decode('utf-8', errors='replace') for p in raw_parts if p]
            for i in range(0, len(str_parts) - 1, 2):
                attrs[str_parts[i]] = str_parts[i + 1]
            return attrs
        except (UnicodeDecodeError, ValueError):
            pass

    yaml_path = os.path.join(track_path, ".attributes.yaml")
    if os.path.exists(yaml_path):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            return attrs
        try:
            with open(yaml_path) as f:
                attrs = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            pass

    return attrs


# Per-track "is computed?" cache, populated lazily by _check_computed_tracks.
# Maps track-name -> bool.  Invalidated by gdb_init / gdb_reload via
# _clear_computed_track_cache(); cheap enough to live as a module-global
# dict.  Even one cached lookup beats pm_track_info (~18 ms on hg38).
_COMPUTED_TRACK_CACHE: dict[str, bool] = {}

# Per-track "computer type supported?" cache for COMPUTED tracks (CT2_AREA /
# CT2_TEST -> True, CT2_POTENTIAL / CT2_TECHNICAL -> False).  Read once from
# the first per-chrom-pair file's Computer2D header.
_COMPUTED_TYPE_OK_CACHE: dict[str, bool] = {}

# Cache of "this expr tuple has already been validated as clean" — keyed by
# (frozenset(exprs), frozenset(vtrack_names)).  Avoids re-parsing the
# expression and re-touching the track_names set on every gextract/gscreen
# call inside a loop.
_CHECK_EXPRS_CACHE: set[tuple] = set()


def _clear_computed_track_cache() -> None:
    """Drop the _check_computed_tracks caches.  Called on db reload/unload."""
    _COMPUTED_TRACK_CACHE.clear()
    _COMPUTED_TYPE_OK_CACHE.clear()
    _CHECK_EXPRS_CACHE.clear()


def _is_computed_track_supported(track: str) -> bool:
    """Whether *track*'s embedded Computer2D type is one we can read.

    Reads the first per-chrom-pair file (or any chunk of the indexed
    track.dat) and inspects the Computer2D header byte at offset 4.
    Returns True for ``CT2_AREA`` / ``CT2_TEST`` (the ones the framework
    actually evaluates today), False otherwise.
    """
    cached = _COMPUTED_TYPE_OK_CACHE.get(track)
    if cached is not None:
        return cached
    import glob
    import os
    import struct

    from ._computer2d import _SUPPORTED_TYPES
    from ._quadtree import SIGNATURE_COMPUTED

    try:
        path = _pymisha.pm_track_path(track)
    except _pymisha.error:
        # "not a track" is the expected answer here; anything else (a bad
        # argument, a memory error) is not this probe's business.
        _logger.debug("no track path for %r; treating it as unsupported", track, exc_info=True)
        _COMPUTED_TYPE_OK_CACHE[track] = False
        return False

    # Indexed track: read the leading bytes of the first per-chunk record
    # in track.dat (each per-pair chunk starts with the signature header).
    candidates: list[str] = []
    indexed_dat = os.path.join(path, "track.dat")
    if os.path.exists(indexed_dat):
        candidates.append(indexed_dat)
    else:
        for entry in glob.glob(os.path.join(path, "*")):
            base = os.path.basename(entry)
            if base.startswith(".") or base.endswith(".idx"):
                continue
            if os.path.isfile(entry):
                candidates.append(entry)
    candidates.sort(key=os.path.getsize)
    for cand in candidates:
        try:
            with open(cand, "rb") as fh:
                head = fh.read(8)
        except OSError:
            continue
        if len(head) < 8:
            continue
        sig = struct.unpack_from("<i", head, 0)[0]
        if sig != SIGNATURE_COMPUTED:
            continue
        ct_type = struct.unpack_from("<i", head, 4)[0]
        ok = ct_type in _SUPPORTED_TYPES
        _COMPUTED_TYPE_OK_CACHE[track] = ok
        return ok

    _COMPUTED_TYPE_OK_CACHE[track] = False
    return False


def _check_computed_tracks(exprs: str | list[str]) -> None:
    """Check whether any track referenced in *exprs* is a COMPUTED track.

    COMPUTED tracks are a Hi-C normalization feature (PotentialComputer2D,
    TechnicalComputer2D) that is not yet implemented in pymisha.  This
    function parses one or more track expressions, resolves the physical
    track names they contain, and raises ``NotImplementedError`` if any of
    them has type ``"computed"``.

    Two caches keep tight loops cheap:
      1. :data:`_CHECK_EXPRS_CACHE`: once a particular (exprs, vtracks)
         pair has been validated clean, subsequent calls return immediately
         without touching pm_track_names or pm_track_info.
      2. :data:`_COMPUTED_TRACK_CACHE`: per-track is-computed result, so
         distinct expressions that mention the same track reuse the result.

    Both caches are cleared on ``gdb_init`` / ``gdb_reload`` /
    ``gdb_unload`` so they stay correct across db transitions.

    Parameters
    ----------
    exprs : str or list of str
        One or more track expressions to inspect.

    Raises
    ------
    NotImplementedError
        If any referenced track is of type ``"computed"``.
    """
    if isinstance(exprs, str):
        exprs = [exprs]

    vtrack_names = frozenset(_shared._VTRACKS.keys())
    cache_key = (tuple(exprs), vtrack_names)
    if cache_key in _CHECK_EXPRS_CACHE:
        return

    from .expr import _parse_expr_vars

    track_names = _track_names_set()

    all_tracks: set[str] = set()
    for expr in exprs:
        _, expr_tracks, _, _ = _parse_expr_vars(expr, track_names, vtrack_names)
        all_tracks.update(expr_tracks)

    for tname in sorted(all_tracks):
        cached = _COMPUTED_TRACK_CACHE.get(tname)
        if cached is None:
            try:
                info = _pymisha.pm_track_info(tname)
            except _pymisha.error:
                # The token is not a track name (a vtrack, a numpy call, ...).
                _logger.debug("no track info for %r; not a COMPUTED track", tname, exc_info=True)
                _COMPUTED_TRACK_CACHE[tname] = False
                continue
            cached = info.get("type") == "computed"
            _COMPUTED_TRACK_CACHE[tname] = cached
        # COMPUTED tracks backed by CT2_AREA / CT2_TEST are readable in pymisha
        # now; CT2_POTENTIAL / CT2_TECHNICAL still need their C++ port
        # (deferred).  Consult the on-disk Computer2D header to pick the
        # right behaviour.
        if cached and not _is_computed_track_supported(tname):
            raise NotImplementedError(
                f"COMPUTED track '{tname}' uses an unsupported Computer2D "
                "type (PotentialComputer2D / TechnicalComputer2D not yet "
                "ported). Consider using R misha for this workflow."
            )

    # All tracks clean — remember this expression set so the next call short-circuits.
    _CHECK_EXPRS_CACHE.add(cache_key)


def _resolve_vtracks_for_cpp_expr(expr: str, caller: str) -> dict | None:
    """Build the vtracks spec dict that the C++ track-create scanner needs.

    Track-creating C++ entry points (``pm_track_create_expr``, ``pm_modify``,
    ``pm_smooth``) accept an optional ``vtracks`` argument that mirrors the
    one ``pm_extract`` already uses.  This helper inspects *expr*, finds the
    virtual tracks it references, verifies they are all C++-eligible, and
    returns the spec dict to forward.  Returns ``None`` when the expression
    does not reference any vtracks (fast path: no extra work in C++).

    Parameters
    ----------
    expr : str
        Track expression about to be handed to the C++ scanner.
    caller : str
        Calling function name (``"gtrack_create"`` etc.) used in the error
        message so users can see which entry point complained.

    Raises
    ------
    NotImplementedError
        If *expr* references a vtrack that the C++ scanner cannot evaluate
        (filter-bearing, array-slice, DataFrame-backed, or with a non-C++
        function).  These vtracks still work through ``gextract``; users can
        materialise the values and write them via ``gtrack_create_sparse``
        / ``gtrack_create_dense``.
    """
    from .expr import _parse_expr_vars
    from .extract import _build_vtracks_dict, _can_vtracks_use_cpp

    vtrack_names = frozenset(_shared._VTRACKS.keys())
    if not vtrack_names:
        return None

    track_names = _track_names_set()
    _, _, used_vtracks, _ = _parse_expr_vars(expr, track_names, vtrack_names)
    if not used_vtracks:
        return None

    if not _can_vtracks_use_cpp(used_vtracks):
        bad = sorted(used_vtracks)
        raise NotImplementedError(
            f"{caller}: expression references virtual track(s) {bad} that "
            "use features not yet supported in the track-creating C++ path "
            "(filters, array slices, DataFrame sources, or non-aggregation "
            "functions). Workaround: materialise the values with gextract "
            "and write them via gtrack_create_sparse / gtrack_create_dense."
        )

    return _build_vtracks_dict(used_vtracks)


def gtrack_ls(*patterns: str, ignore_case: bool = False, **attr_filters: str) -> list[str] | None:
    """
    Return a list of track names in the Genomic Database.

    Returns track names that match all supplied patterns. Name patterns are
    applied as regex searches against track names. Attribute patterns are
    matched against the corresponding track attribute values. Multiple
    patterns are applied conjunctively (all must match).

    Parameters
    ----------
    *patterns : str
        Regex patterns to filter track names. Each pattern is applied
        sequentially; only tracks matching all patterns are returned.
    ignore_case : bool, default False
        If True, pattern matching is case-insensitive.
    **attr_filters : str
        Keyword arguments of the form ``attribute_name=pattern`` where
        underscores in the keyword are converted to dots for the attribute
        lookup (e.g., ``created_by="sparse"`` matches attribute
        ``created.by``).

    Returns
    -------
    list of str or None
        Sorted list of matching track names, or None if no tracks match.

    Raises
    ------
    ValueError
        If a regex pattern is invalid.

    See Also
    --------
    gtrack_exists : Test whether a single track exists.
    gtrack_info : Get metadata for a track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_ls()  # doctest: +SKIP
    ['array_track', 'dense_track', 'rects_track', 'sparse_track', 'subdir.dense_track2']
    >>> pm.gtrack_ls("dense")  # doctest: +SKIP
    ['dense_track', 'subdir.dense_track2']
    >>> pm.gtrack_ls(created_by="create_sparse")  # doctest: +SKIP
    ['sparse_track']
    """
    _checkroot()
    tracks = _pymisha.pm_track_names()

    if tracks is None or len(tracks) == 0:
        return None

    flags = re.IGNORECASE if ignore_case else 0

    for pattern in patterns:
        try:
            regex = re.compile(pattern, flags)
            tracks = [t for t in tracks if regex.search(t)]
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    if not tracks:
        return None

    if attr_filters:
        converted_filters = {key.replace('_', '.'): pattern for key, pattern in attr_filters.items()}
        filtered_tracks: list[str] = []
        for track in tracks:
            attrs = _load_track_attributes(track)

            all_match = True
            for attr_name, pattern in converted_filters.items():
                attr_value = attrs.get(attr_name, "")
                try:
                    regex = re.compile(pattern, flags)
                    if not regex.search(str(attr_value)):
                        all_match = False
                        break
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{pattern}' for attribute '{attr_name}': {e}") from e

            if all_match:
                filtered_tracks.append(track)

        tracks = filtered_tracks

    if not tracks:
        return None

    return list(tracks)


def gtrack_dbs(track: str | list[str], dataframe: bool = False) -> dict[str, list[str]] | pd.DataFrame:
    """
    Return database root(s) containing the given track(s).

    For each track name, searches the current database root and all
    loaded dataset roots to find which databases contain the track.

    Parameters
    ----------
    track : str or list of str
        Track name(s) to look up.
    dataframe : bool, default False
        If True, return a DataFrame with columns ``track`` and ``db``.
        If False, return a dict mapping track names to lists of database paths.

    Returns
    -------
    dict or DataFrame
        If *dataframe* is False, a dict ``{track_name: [db_path, ...]}``.
        If *dataframe* is True, a DataFrame with columns ``track`` and ``db``.

    See Also
    --------
    gtrack_ls : List available tracks.
    gtrack_exists : Test whether a track exists.
    gintervals_dbs : Same for interval sets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_dbs("dense_track")  # doctest: +SKIP
    {'dense_track': ['/path/to/db']}
    """
    _checkroot()

    tracks = [track] if isinstance(track, str) else list(track)

    assert _shared._GROOT is not None
    all_dbs = [_shared._GROOT] + list(_shared._GDATASETS)
    # in a subdirectory (gdir_cd), a name is the one under it, in the database holding it
    if _shared._gwd_is_subdir():
        all_dbs = [db for db in all_dbs if db == _shared._gwd_db()]

    result = {}
    for t in tracks:
        dbs = [db for db in all_dbs if _track_dir(t, db).exists()]
        result[t] = dbs

    if dataframe:
        rows_track = []
        rows_db = []
        for t, dbs in result.items():
            for db in dbs:
                rows_track.append(t)
                rows_db.append(db)
        import pandas as pd
        return pd.DataFrame({"track": rows_track, "db": rows_db})

    return result


def gtrack_info(track: str) -> dict[str, Any]:
    """
    Return metadata about a track.

    Returns a dictionary containing track properties such as type,
    dimensions, bin size, total size in bytes, and any user-defined
    attributes. The fields vary depending on the track type (Dense,
    Sparse, Rectangles, Points).

    Parameters
    ----------
    track : str
        Track name.

    Returns
    -------
    dict
        Dictionary of track properties. Common keys include ``"type"``
        (``"dense"``, ``"sparse"``, ``"rectangles"``, ``"points"``),
        ``"bin_size"`` (for dense tracks), ``"total_size"``, and
        ``"attributes"`` (dict of user-set attributes, if any).

    Raises
    ------
    ValueError
        If the track does not exist.

    See Also
    --------
    gtrack_exists : Test whether a track exists.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_info("dense_track")  # doctest: +SKIP
    {'type': 'dense', 'dimensions': 1, ...}
    >>> pm.gtrack_info("sparse_track")  # doctest: +SKIP
    {'type': 'sparse', 'dimensions': 1, ...}
    """
    _checkroot()
    info: dict[str, Any] = dict(_pymisha.pm_track_info(track))

    if 'attributes_path' in info:
        del info['attributes_path']

    # Check for 2D indexed format: the C++ engine may misidentify a 2D
    # indexed track as 1D sparse because it doesn't know about the 2D
    # index magic.  Read the track.idx header to detect this and override.
    if info.get('format') == 'indexed':
        track_path = _pymisha.pm_track_path(track)
        idx_path = os.path.join(track_path, 'track.idx')
        if os.path.exists(idx_path):
            try:
                with open(idx_path, 'rb') as f:
                    magic = f.read(8)
                if magic == b'MISHT2D\x00':
                    # This is a 2D indexed track.  Read type from header.
                    # NB: `struct` must stay the module-level import (tracks.py:20).
                    # A function-local `import struct` here makes the name local to
                    # gtrack_info, so the `except (OSError, struct.error)` below
                    # cannot even be evaluated when open() fails before this line.
                    with open(idx_path, 'rb') as f:
                        f.seek(12)  # skip magic(8) + version(4)
                        track_type_int = struct.unpack('<I', f.read(4))[0]
                    info['type'] = 'points' if track_type_int == 1 else 'rectangles'
                    info['dimensions'] = 2
            except (OSError, struct.error):
                # A truncated or unreadable track.idx: report the track as the
                # C++ engine sees it (1D sparse), but say so - the misreport is
                # otherwise indistinguishable from a genuinely 1D track.
                _logger.warning("could not read the 2D header of %s; reporting %r as the engine "
                                "sees it", idx_path, track, exc_info=True)

    attrs = _load_track_attributes(track)
    if attrs:
        info['attributes'] = attrs

    return info


def gtrack_dataset(track: str) -> str | None:
    """
    Return the database root path that contains a track.

    When multiple databases are connected, this identifies which database
    a track belongs to by returning the filesystem path of that database
    root.

    Parameters
    ----------
    track : str
        Track name.

    Returns
    -------
    str
        Absolute filesystem path of the database root containing the track.

    Raises
    ------
    ValueError
        If track is None or the track does not exist.

    See Also
    --------
    gtrack_info : Get full metadata for a track.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_dataset("dense_track")  # doctest: +SKIP
    '.../trackdb/test'
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()
    result: str | None = _pymisha.pm_track_dataset(track)
    # in a subdirectory (gdir_cd), the names under it are the database's holding it
    if result is not None and _shared._gwd_is_subdir():
        return _shared._gwd_db()
    return result


@_shared._with_umask()   # database writes carry misha's permissions
def _save_track_attributes(track_name: str, attrs: dict[str, str]) -> None:
    """
    Save track attributes to the .attributes file (binary format).
    """
    track_path = _pymisha.pm_track_path(track_name)
    if not track_path:
        raise ValueError(f"Track '{track_name}' does not exist")

    bin_path = os.path.join(track_path, ".attributes")

    # Filter out empty values (empty means delete attribute).
    # Keep falsy-but-valid values like 0 and False.
    attrs = {k: v for k, v in attrs.items() if v is not None and v != ""}

    if not attrs:
        # If no attributes, remove the file if it exists
        if os.path.exists(bin_path):
            os.remove(bin_path)
        return

    # Build binary format: key\0value\0key\0value\0...
    parts = []
    for key, value in sorted(attrs.items()):
        parts.append(key.encode('utf-8'))
        parts.append(b'\x00')
        parts.append(str(value).encode('utf-8'))
        parts.append(b'\x00')

    with _shared._write_file(bin_path) as f:
        f.write(b''.join(parts))


def _track_exists(track_name: str) -> bool:
    """Check if a track exists."""
    track_path = _pymisha.pm_track_path(track_name)
    return track_path is not None and track_path != ""


def _validate_track_name(track: str) -> None:
    validate_dotted_name(track, "track name")


def _validate_track_var_name(var: str) -> None:
    if not isinstance(var, str) or not var:
        raise ValueError("var must be a non-empty string")
    if "\x00" in var:
        raise ValueError("var must not contain NUL bytes")
    if os.path.isabs(var):
        raise ValueError("var must be a relative name")
    if var in {".", ".."}:
        raise ValueError("var cannot be '.' or '..'")
    if ".." in var.split("/"):
        raise ValueError("var cannot contain path traversal components")
    if os.sep in var or (os.altsep and os.altsep in var):
        raise ValueError("var must not contain path separators")


def _target_root() -> str | None:
    """The database a new track is written in: the user root when one is set and the
    working directory is the working database's tracks/, the database the working directory
    is in otherwise (the working database when it is in none)."""
    if _shared._GROOT is None:
        return None
    if _shared._gwd_is_subdir():
        return _shared._gwd_db() or _shared._GROOT
    # the working directory is the tracks/ of the working database, the user root or a dataset
    root = _shared._path_db(_shared._GWD) or _shared._GROOT
    return _shared._UROOT if _shared._UROOT and root == _shared._GROOT else root


def _track_dir_for_create(track: str) -> Path:
    """The directory a new track is created in: under the working directory (R misha's GWD),
    or the user root's tracks/ when one is set and the working directory is the working
    database's tracks/."""
    if _shared._GROOT is None:
        raise ValueError("Database not initialized. Call gdb_init() first.")
    track_dir = Path(_pymisha.pm_new_track_path(track))
    # In a subdirectory (gdir_cd), a track of an indexed working database is written in the
    # session's chromosome ids, so only into a loaded database that uses them: not through a
    # link or into a database kept below the working directory, nor into a dataset of unknown
    # order (R's .gconfirmtrackcreate)
    if (_shared._gwd_is_subdir() and _db_is_indexed(_shared._GROOT)
            and not _shared._same_order(_shared._path_db(track_dir))):
        raise ValueError(
            f"Track {track} would be written into a database that may number the chromosomes "
            "differently from this session; create it in a session of that database."
        )
    return track_dir


def _track_dir(track: str, db: str | None = None) -> Path:
    """R misha's ``.track_dir``: the directory of a track in database db (default: the one
    it is listed in), by a name relative to the database's tracks/, or to the working
    directory when that is a subdirectory of it (gdir_cd). A name not listed is under the
    working directory."""
    if db is None and not _shared._gwd_is_subdir():
        db = gtrack_dataset(track)
    return Path(_shared._names_dir(db)) / f"{track.replace('.', '/')}.track"


def _db_is_indexed(root: str | None) -> bool:
    if not root:
        return False
    seq_dir = Path(root) / "seq"
    return (seq_dir / "genome.idx").exists() and (seq_dir / "genome.seq").exists()


def _fsync_tree(path: Path) -> None:
    """Force a staged directory tree to stable storage, depth first.

    os.rename() is atomic against other *processes* - a killed writer leaves
    either the old track or nothing - but it says nothing about stable storage.
    The file contents and the directory entry naming them are two independent
    sets of dirty pages with no ordering between them, so after a power loss or a
    kernel panic a committed track name can point at data that never left the
    page cache: the exact half-written state the staging exists to prevent, with
    no error and no marker.

    Files before the directory naming them, children before parents, so a
    surviving directory entry always points at synced data.

    Cost, measured on misha's identical writers 2026-08-24: nil on the lab NFS
    (close-to-open consistency already forces the client to flush at close(), so
    this only names a cost that was already being paid), nil on tmpfs, ~0.35s per
    GB on local ext4 - under 1% of a track creation either way. Hence
    unconditional. See misha 5.11.24.

    Failures are not swallowed: EIO and ENOSPC from fsync are precisely the case
    where completing the rename would publish a damaged track under a live name.
    """
    for entry in sorted(path.iterdir()):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            _fsync_tree(entry)
        elif entry.is_file():
            fd = os.open(entry, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    _fsync_dir(path)


def _fsync_dir(path: Path) -> None:
    """fsync a directory entry. EINVAL means the filesystem does not implement
    it, which is "unsupported", not "lost data"."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno != errno.EINVAL:
            raise
    finally:
        os.close(fd)


@contextlib.contextmanager
def _atomic_track_create(track: str):
    """Run a track-create body atomically: tmp dir + rename on success.

    Yields the tmp directory the writer should mkdir into. While the
    context is active, the C++ create_dir_override slot is set so
    pm_track_create_* writers target the tmp path; Python-side writers
    must use the yielded path explicitly. On clean exit, the tmp dir is
    renamed to the final track dir (so concurrent gdb scans never see a
    partial track). On exception (including KeyboardInterrupt) the tmp
    dir is trashed and the original exception is re-raised. Pre-checks
    that the track does not already exist before doing any work.
    Mirrors R misha .gtrack.create_atomic (R 5.6.30 a7f6bb95, 81635130).
    """
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")

    track_dir = _track_dir_for_create(track)
    parent = track_dir.parent
    with _shared._with_umask():
        parent.mkdir(parents=True, exist_ok=True)
    rand = secrets.token_hex(4)
    tmp_dir = parent / f".{track_dir.name}.tmp.{os.getpid()}.{rand}"

    _pymisha.pm_set_create_dir_override(str(tmp_dir))
    try:
        yield tmp_dir
        if not tmp_dir.exists():
            raise RuntimeError(
                f"track writer did not produce expected directory: {tmp_dir}"
            )
        # Durability, not just atomicity: sync the staged tree before the
        # rename and the parent directory after it, so nothing is published
        # before what it references is on stable storage.
        _fsync_tree(tmp_dir)
        os.rename(tmp_dir, track_dir)
        _fsync_dir(parent)
        # Drop any index cached for this path in a previous track lifecycle:
        # "rm <track>; create <track>" otherwise routes reads through the old
        # layout (e.g. a track.dat that no longer exists).
        _shared._invalidate_dir_cache(track_dir)
    except BaseException:
        # Deliberately BaseException, not Exception: a Ctrl-C in the middle of
        # a track write must still take the half-written directory with it.
        # Nothing is swallowed - the exception is re-raised unchanged.
        _gdb_trash(tmp_dir, async_unlink=True)
        raise
    finally:
        _pymisha.pm_clear_create_dir_override()


def _normalize_intervals_df(intervals: Any) -> pd.DataFrame:
    if intervals is None:
        raise ValueError("intervals cannot be None")
    if not isinstance(intervals, pd.DataFrame):
        intervals = pd.DataFrame(intervals)
    required = {"chrom", "start", "end"}
    if not required.issubset(intervals.columns):
        raise ValueError("intervals must contain columns: chrom, start, end")

    out = intervals.copy()
    out = out.loc[:, ["chrom", "start", "end"]]
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["chrom", "start", "end"])
    if len(out) == 0:
        raise ValueError("intervals is empty after dropping invalid rows")

    out["start"] = out["start"].astype(np.int64)
    out["end"] = out["end"].astype(np.int64)
    out = out[(out["start"] >= 0) & (out["end"] > out["start"])].copy()
    if len(out) == 0:
        raise ValueError("intervals must contain at least one valid row with 0 <= start < end")
    return out.reset_index(drop=True)


_MAX_REPORTED_CHROMS = 20


def _is_primary_chrom_name(name: str) -> bool:
    """"chr7", "7", "X" - the names a genome database is expected to have.

    Scaffolds, patches and unplaced contigs carry extra tokens and do not match;
    neither does a mitochondrial name. Mirrors misha's
    ``UnknownChroms::is_primary_chrom_name`` (5.11.18).
    """
    core = name[3:] if len(name) > 3 and name[:3].lower() == "chr" else name
    if not core or len(core) > 2:
        return False
    if core.isdigit():
        return True
    return len(core) == 1 and core.upper() in "XYZW"


def _report_skipped_chroms(parsed: pd.DataFrame, file_path: str) -> None:
    """Report the chromosome names an import dropped (misha 5.11.18).

    Which channel depends on what was dropped. A scaffold or unplaced contig is
    what every whole-genome bigWig looks like against a primary-only database:
    a log message. A primary chromosome means the naming is probably wrong and
    real data is being lost silently: a warning.

    Only names present in the file that did not resolve count - not database
    chromosomes the file happens not to cover. A chr1-only bedGraph dropped
    nothing.
    """
    from .intervals import gintervals_all

    known = set(gintervals_all()["chrom"].astype(str))
    unknown: list[str] = []
    n_matched = 0
    for raw in pd.unique(parsed["chrom"].astype(str)):
        try:
            canon = _pymisha.pm_normalize_chroms([raw])[0]
        except (_pymisha.error, IndexError):
            canon = None
        if canon is not None and canon in known:
            n_matched += 1
        else:
            unknown.append(raw)

    _report_unknown_chroms(unknown, len(unknown), [c for c in unknown if _is_primary_chrom_name(c)],
                           n_matched, file_path)


def _report_unknown_chroms(unknown: list[str], num_unknown: int, primary: list[str], n_matched: int,
                           file_path: str) -> None:
    """The report of _report_skipped_chroms, given the names: *unknown* (or the first of them, of
    *num_unknown* in all), those of them that look like a primary chromosome, and the number of
    chromosomes that matched."""
    # Nothing dropped, or nothing matched at all - the latter is the caller's
    # "No intervals map to known chromosomes" error, not a partial import.
    if not num_unknown or not n_matched:
        return

    shown = unknown[:_MAX_REPORTED_CHROMS]
    more = "+" if num_unknown > len(shown) else ""
    names = ", ".join(shown) + (", ..." if more else "")

    if primary:
        warnings.warn(
            f"{num_unknown}{more} chromosome name(s) in {file_path} do not exist "
            f"in the genome database and were skipped, among them primary "
            f"chromosome(s): {', '.join(primary[:_MAX_REPORTED_CHROMS])}. Data for "
            f"the remaining {n_matched} chromosome(s) was imported.",
            stacklevel=user_stacklevel(),
        )
    else:
        _logger.info(
            "%d%s chromosome name(s) in %s do not exist in the genome database "
            "and were skipped: %s. Data for the remaining %d chromosome(s) was "
            "imported.", num_unknown, more, file_path, names, n_matched,
        )


def _canonicalize_known_chroms(df: pd.DataFrame) -> pd.DataFrame:
    from .intervals import gintervals_all

    chrom_sizes = gintervals_all()
    known = set(chrom_sizes["chrom"].astype(str).tolist())

    # Batch-normalize: collect unique chromosome names, normalize once
    raw_chroms = df["chrom"].astype(str)
    unique_raw = raw_chroms.unique()

    # Build mapping: raw -> canonical (or None if normalization fails)
    canon_map = {}
    # Batch call for all unique chroms at once
    for raw in unique_raw:
        try:
            c = _pymisha.pm_normalize_chroms([raw])[0]
            canon_map[raw] = c if c in known else None
        except (_pymisha.error, IndexError):
            _logger.debug("chromosome %r does not normalize to a known name; dropping its rows",
                          raw, exc_info=True)
            canon_map[raw] = None

    # Vectorized apply via map
    canonical = raw_chroms.map(canon_map)
    mask = canonical.notna()
    out = df.loc[mask.to_numpy()].copy()
    out["chrom"] = canonical.loc[mask.to_numpy()].to_numpy()
    return out.reset_index(drop=True)


def _set_created_attrs(track: str, description: str, created_by: str, attrs: dict[str, str] | None = None) -> None:
    # Bypass readonly check for internal track creation attrs
    existing_attrs = _load_track_attributes(track)
    existing_attrs["created.by"] = created_by
    existing_attrs["created.date"] = _datetime.datetime.now().ctime()
    existing_attrs["created.user"] = _getpass.getuser()
    existing_attrs["description"] = str(description)
    if attrs is not None:
        if not isinstance(attrs, dict):
            raise ValueError("attrs must be a dict of attribute name -> value")
        for k, v in attrs.items():
            if not isinstance(k, str) or not k:
                raise ValueError("attrs keys must be non-empty strings")
            existing_attrs[k] = "" if v is None else str(v)
    _save_track_attributes(track, existing_attrs)


@_shared._with_umask()   # database writes carry misha's permissions
def _write_created_attrs_at_path(
    track_dir: str | Path,
    description: str,
    created_by: str,
    attrs: dict[str, str] | None = None,
) -> None:
    """Write fresh track creation attributes directly to ``track_dir``.

    Used by gtrack_create_* to set initial attrs (description, created.by,
    created.date, created.user) before the C++ track_cache has been
    refreshed. Skipping the dbreload-before-attrs pattern saves one full
    track scan (~215 ms on hg38) per create. Mirrors _save_track_attributes
    but takes the path explicitly so we don't go through pm_track_path.
    """
    track_path = str(track_dir)
    payload: dict[str, str] = {
        "created.by": created_by,
        "created.date": _datetime.datetime.now().ctime(),
        "created.user": _getpass.getuser(),
        "description": str(description),
    }
    if attrs is not None:
        if not isinstance(attrs, dict):
            raise ValueError("attrs must be a dict of attribute name -> value")
        for k, v in attrs.items():
            if not isinstance(k, str) or not k:
                raise ValueError("attrs keys must be non-empty strings")
            payload[k] = "" if v is None else str(v)
    payload = {k: v for k, v in payload.items() if v is not None and v != ""}
    if not payload:
        return
    parts: list[bytes] = []
    for key, value in sorted(payload.items()):
        parts.append(key.encode("utf-8"))
        parts.append(b"\x00")
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\x00")
    bin_path = os.path.join(track_path, ".attributes")
    with open(bin_path, "wb") as f:
        f.write(b"".join(parts))


def _open_text_auto(path: str | Path) -> Any:
    lower = str(path).lower()
    if lower.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith((".xz", ".lzma")):
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) == 0:
            zf.close()
            raise ValueError(f"Zip file '{path}' does not contain a regular file")
        stream = io.TextIOWrapper(zf.open(names[0], "r"), encoding="utf-8", errors="replace")
        base_close = stream.close

        def _close_with_zip(*args, **kwargs):
            try:
                return base_close(*args, **kwargs)
            finally:
                zf.close()

        stream.close = _close_with_zip  # type: ignore[method-assign]
        return stream
    return open(path, encoding="utf-8", errors="replace")


def _close_text_auto(stream: Any) -> None:
    stream.close()


def _filter_track_lines(path: str, skip_prefixes: tuple[str, ...]) -> list[str]:
    """Read *path* and return non-blank lines not starting with *skip_prefixes*.

    Cheap line-level pass (strip + startswith only, no field parsing), shared
    by the BED and tabular parsers so the expensive splitting/numeric coercion
    can be vectorized downstream.
    """
    stream = _open_text_auto(path)
    try:
        return [
            raw for raw in stream
            if (s := raw.strip()) and not s.startswith(skip_prefixes)
        ]
    finally:
        _close_text_auto(stream)


def _parse_bed(path: str) -> pd.DataFrame:
    kept = _filter_track_lines(path, ("#", "track", "browser"))
    if not kept:
        raise ValueError(f"BED file '{path}' contains no intervals")

    # Fast path for tab-delimited BED (the UCSC standard): the C parser with
    # per-column `names` pads ragged BED3..BED12 rows and the numeric coercion
    # is vectorized (~1.8x vs the per-row Python loop, which did int(float(...))
    # per field). Space-delimited / non-numeric-coord files return None and
    # fall through to the general parser below.
    if "\t" in kept[0]:
        df = _parse_bed_fast_tab(kept)
        if df is not None:
            return df
    return _parse_bed_generic(kept)


def _parse_bed_fast_tab(kept: list[str]) -> pd.DataFrame | None:
    maxc = max(line.count("\t") for line in kept) + 1
    if maxc < 3:
        return None
    # Read only the needed columns with inferred dtypes (the C parser converts
    # coordinates directly - far cheaper than reading str + pd.to_numeric).
    # chrom is forced to str so a bare numeric chrom ("1") stays "1".
    usecols = [0, 1, 2, 4] if maxc >= 5 else [0, 1, 2]
    df = pd.read_csv(
        io.StringIO("".join(kept)), sep="\t", header=None,
        names=range(maxc), usecols=usecols, dtype={0: str}, engine="c",
    )
    start, end = df[1], df[2]
    # Non-numeric / NaN coordinate -> not clean tab-delimited BED (e.g.
    # space-delimited, or a row with < 3 fields); defer to the generic parser,
    # which parses it or raises.
    if not (pd.api.types.is_numeric_dtype(start) and pd.api.types.is_numeric_dtype(end)):
        return None
    if start.isna().any() or end.isna().any():
        return None
    # BED score is column 4 (0-based); unparseable / absent -> 1.0 (R parity).
    value = (
        pd.to_numeric(df[4], errors="coerce").fillna(1.0).to_numpy()
        if maxc >= 5
        else np.ones(len(df), dtype=float)
    )
    return pd.DataFrame({
        "chrom": df[0].astype(str).to_numpy(),
        "start": start.to_numpy().astype(np.int64),
        "end": end.to_numpy().astype(np.int64),
        "value": value,
    })


def _parse_bed_generic(kept: list[str]) -> pd.DataFrame:
    chrom: list[str] = []
    start: list[int] = []
    end: list[int] = []
    value: list[float] = []
    for raw in kept:
        line = raw.strip()
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(f"Malformed BED line: {line}")
        chrom.append(fields[0])
        start.append(int(float(fields[1])))
        end.append(int(float(fields[2])))
        v = 1.0
        if len(fields) >= 5:
            try:
                v = float(fields[4])
            except ValueError:
                v = 1.0
        value.append(v)
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})


def _unzip_import_file(path: str, tmpdir: str) -> str:
    """misha's gtrack.import unzips a .gz / .zip file into a temporary file named like it
    without that suffix, and imports that. Returns the unzipped file."""
    out = os.path.join(tmpdir, re.sub(r"\.(gz|zip)$", "", os.path.basename(path), flags=re.IGNORECASE))
    try:
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if len(names) != 1:   # gunzip, which misha uses, reads a zip of one file only
                    raise zipfile.BadZipFile(f"{len(names)} files in the archive")
                with zf.open(names[0]) as fin, open(out, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        else:
            with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Failed to unzip file {path}: {exc}") from exc
    return out


def _parse_wig_or_bedgraph(path: str) -> pd.DataFrame:
    """The records of a plain-text WIG / bedGraph file (gtrack_import unzips first), in C++."""
    cols = _pymisha.pm_parse_wig_or_bedgraph(str(path))
    return pd.DataFrame({c: cols[c] for c in ("chrom", "start", "end", "value", "step")})


def _check_wig_records(parsed: pd.DataFrame, path: str) -> pd.DataFrame:
    """misha's rules for WIG / bedGraph records, which a bigWig's records follow too.

    A fixedStep / variableStep record that crosses the chromosome end is clipped
    to it; one that starts past it is an error, as is a bedGraph record that is
    empty or does not fit the chromosome. A chromosome's records must be together,
    sorted and not overlapping. Returns the records of the database chromosomes.
    """
    from .intervals import gintervals_all

    if len(parsed) == 0:   # no records at all: an empty track, as in misha
        return parsed.drop(columns="step")
    df = _canonicalize_known_chroms(parsed)
    if len(df) == 0:
        raise ValueError("No intervals map to known chromosomes")
    sizes = gintervals_all().set_index("chrom")["end"]
    chrom = df["chrom"].to_numpy()
    size = df["chrom"].map(sizes).to_numpy(dtype=np.int64)
    step = df.pop("step").to_numpy(dtype=bool)
    start = df["start"].to_numpy(dtype=np.int64)
    end = df["end"].to_numpy(dtype=np.int64)

    bad = np.flatnonzero(np.where(step, start >= size, (end > size) | (start >= end)))
    if bad.size:
        i = bad[0]
        raise ValueError(f"WIG file {path}: invalid record {chrom[i]}:{start[i]}-{end[i]} "
                         f"(chromosome size {size[i]})")
    same = chrom[1:] == chrom[:-1]
    runs = chrom[np.r_[True, ~same]]
    if len(set(runs)) < len(runs):
        raise ValueError(f"WIG file {path} is not sorted by chromosomes")
    end = np.where(step, np.minimum(end, size), end)
    bad = np.flatnonzero(same & (start[1:] < end[:-1]))
    if bad.size:
        i = bad[0] + 1
        raise ValueError(f"WIG file {path}: records of {chrom[i]} are overlapping or unsorted at {start[i]}")
    df["end"] = end
    return df


def _parse_tabular_track(path: str) -> pd.DataFrame:
    kept = _filter_track_lines(path, ("#",))
    if not kept:   # no records at all: an empty track, as in misha
        return _empty_records()

    # Fast path for tab-delimited tables (the common case): one vectorized
    # C-engine read + numeric coercion instead of a per-row Python loop. Any
    # parser hiccup (e.g. rows wider than the header) defers to the general
    # parser, which preserves the original lenient per-row behavior.
    if "\t" in kept[0]:
        try:
            # Inferred dtypes: the C parser converts start/end/value directly
            # (reading as str + pd.to_numeric is markedly slower).
            df = pd.read_csv(io.StringIO("".join(kept)), sep="\t", header=0, engine="c")
        except ValueError:
            # pandas' ParserError / EmptyDataError / dtype coercion all derive
            # from ValueError; the lenient per-row parser takes it from here.
            _logger.debug("the fast tab-delimited parse of %s failed; using the general parser",
                          path, exc_info=True)
            df = None
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            return _tabular_finish(df, path)
    return _parse_tabular_generic(kept, path)


def _tabular_finish(df: pd.DataFrame, path: str) -> pd.DataFrame:
    req = ["chrom", "start", "end"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Tabular track file must contain '{c}' column")
    val_cols = [c for c in df.columns if c not in req]
    if len(val_cols) != 1:
        raise ValueError("Tabular track file must contain exactly one value column besides chrom/start/end")
    c_val = val_cols[0]

    # Drop short rows (a missing field reads as NaN) - mirrors the original
    # `len(fields) < len(cols)` skip - then coerce, raising on a non-numeric
    # field in a full row (mirrors the original int()/float() raise).
    sel = df[[*req, c_val]]
    keep = sel.notna().all(axis=1)
    if len(sel) and not keep.any():
        raise ValueError(f"File '{path}' contains no data rows")
    sel = sel[keep]
    start = pd.to_numeric(sel["start"], errors="coerce")
    end = pd.to_numeric(sel["end"], errors="coerce")
    value = pd.to_numeric(sel[c_val], errors="coerce")
    if start.isna().any() or end.isna().any() or value.isna().any():
        raise ValueError(f"Malformed numeric field in tabular track '{path}'")
    return pd.DataFrame({
        "chrom": sel["chrom"].astype(str).to_numpy(),
        "start": start.to_numpy().astype(np.int64),
        "end": end.to_numpy().astype(np.int64),
        "value": value.to_numpy().astype(float),
    })


def _parse_tabular_generic(kept: list[str], path: str) -> pd.DataFrame:
    header_line = kept[0].strip()
    header = header_line.split("\t") if "\t" in header_line else header_line.split()
    cols = [c.strip() for c in header]
    req = ["chrom", "start", "end"]
    for c in req:
        if c not in cols:
            raise ValueError(f"Tabular track file must contain '{c}' column")
    val_cols = [c for c in cols if c not in req]
    if len(val_cols) != 1:
        raise ValueError("Tabular track file must contain exactly one value column besides chrom/start/end")
    idx = {c: i for i, c in enumerate(cols)}
    c_val = val_cols[0]
    out: dict[str, list[Any]] = {"chrom": [], "start": [], "end": [], "value": []}
    for raw in kept[1:]:
        line = raw.strip()
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) < len(cols):
            continue
        out["chrom"].append(fields[idx["chrom"]])
        out["start"].append(int(float(fields[idx["start"]])))
        out["end"].append(int(float(fields[idx["end"]])))
        out["value"].append(float(fields[idx[c_val]]))
    if len(out["chrom"]) == 0:
        if len(kept) > 1:
            raise ValueError(f"File '{path}' contains no data rows")
        return _empty_records()
    return pd.DataFrame(out)


def _parse_bigwig(path: str, name: str) -> pd.DataFrame:
    """Read a bigWig's full-resolution records with misha's reader; zoom levels are not used.

    Records come back sorted, non-overlapping and within their chromosome, as
    _check_wig_records makes the text ones. A truncated or corrupt file is an
    error, as is one whose chromosomes are all unknown to the database. Reports
    the chromosomes it skipped, as _report_skipped_chroms does; messages name
    *name*, the file the user gave (path may be its unzipped copy).
    """
    res = _pymisha.pm_parse_bigwig(path)
    if res["num_unknown"] and not res["num_matched"]:
        raise ValueError(f"BigWig file '{name}' contains no intervals for known chromosomes")
    _report_unknown_chroms(res["unknown"], res["num_unknown"], res["unknown_primary"],
                           res["num_matched"], name)
    return pd.DataFrame({c: res[c] for c in ("chrom", "start", "end", "value")})


def _ensure_track_absent(track: str) -> None:
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")


def gtrack_create_sparse(
    track: str, description: str, intervals: Intervals, values: Any = None
) -> None:
    """
    Create a Sparse track from intervals and values.

    Creates a new Sparse track where each interval carries an associated
    numeric value. Intervals must be non-overlapping within each
    chromosome. Chromosome names are normalized and filtered to those
    present in the current genome database. The description is stored
    as a track attribute.

    *values* is matched to *intervals* row by row, in the order the intervals
    are passed; *intervals* need not be sorted. Note however that
    :func:`gintervals` returns its result sorted, so building *intervals* with
    ``gintervals()`` while keeping *values* in the original order will bind
    values to the wrong intervals. Keep the values in a ``value`` column of
    *intervals* and omit *values* to make such a mismatch impossible.

    Parameters
    ----------
    track : str
        Name for the new track. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.
    description : str
        Human-readable description stored as a track attribute.
    intervals : pandas.DataFrame
        One-dimensional intervals with columns ``chrom``, ``start``,
        ``end``.
    values : array-like of float, optional
        Numeric values, one per interval, in the same order as the rows
        of *intervals*. Length must match the number of rows in
        *intervals*. If ``None``, the ``value`` column of *intervals* is
        used.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, intervals overlap, values length
        does not match intervals, or no intervals map to known
        chromosomes.

    See Also
    --------
    gtrack_create_dense : Create a Dense (fixed-bin) track.
    gtrack_create : Create a track from a track expression.
    gtrack_import : Import a track from a file.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    >>> pm.gtrack_create_sparse("test_sp", "Test", intervs, [1.0])  # doctest: +SKIP
    >>> pm.gtrack_rm("test_sp", force=True)  # doctest: +SKIP
    """
    _create_sparse(track, description, intervals, values)


def _empty_records() -> pd.DataFrame:
    return pd.DataFrame({"chrom": pd.Series([], dtype=object), "start": pd.Series([], dtype=np.int64),
                         "end": pd.Series([], dtype=np.int64), "value": pd.Series([], dtype=np.float64)})


def _create_sparse(track: str, description: str, intervals: Intervals, values: Any = None,
                   r_import: bool = False) -> None:
    """gtrack_create_sparse. With *r_import* (gtrack_import), no intervals at all make an empty track."""
    _checkroot()
    _validate_track_name(track)

    if values is None:
        if not isinstance(intervals, pd.DataFrame) or "value" not in intervals.columns:
            raise ValueError('values is missing and intervals has no "value" column')
        values = intervals["value"].to_numpy()

    if r_import and len(intervals) == 0:
        data = _empty_records()
    else:
        data = _normalize_intervals_df(intervals)
        vals = np.asarray(values, dtype=np.float64)
        if len(vals) != len(data):
            raise ValueError("Length of values must match number of intervals")
        data = data.copy()
        data["value"] = vals
        data = _canonicalize_known_chroms(data)

        if len(data) == 0:
            raise ValueError("No intervals map to known chromosomes")

        data = data.sort_values(["chrom", "start", "end"], kind="mergesort").reset_index(drop=True)
        prev = data.groupby("chrom")["end"].shift(1)
        overlap = data["start"] < prev
        if bool(overlap.fillna(False).any()):
            raise ValueError("Sparse intervals must be sorted and non-overlapping per chromosome")

    with _atomic_track_create(track):
        _apply_create_parallel_writers_from_config()
        _pymisha.pm_track_create_sparse(track, _df2pymisha(data))

    # On indexed DBs the C++ writer (pm_track_create_sparse) already
    # produced track.dat + track.idx directly, so we skip the post-create
    # convert step. Byte-identical to the per-chrom + convert pipeline.
    #
    # Avoid the dbreload-then-set-attrs-then-dbreload sequence: we know
    # the final track_dir from `_track_dir_for_create`, so we can write
    # the .attributes file directly and trigger a single dbreload that
    # registers both the track and its attrs in one pass. Saves ~215 ms
    # per call on hg38 (15k tracks).
    track_dir = _track_dir_for_create(track)
    try:
        _write_created_attrs_at_path(
            track_dir, description,
            f'gtrack.create_sparse("{track}", description, intervals, values)',
        )
        _pm_dbreload(_target_root())
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=user_stacklevel(),
        )
        raise


_DEFAULT_TRACK_CREATE_PARALLEL_WRITERS = 4  # empirical NFSv3 CREATE saturation


def _apply_create_parallel_writers_from_config() -> None:
    """Push the configured worker count for empty-chrom file dispatch
    into C++ before a track-create call.

    Resolution order:
    1. `pm.CONFIG["track_create_parallel_writers"]` if explicitly set.
    2. `pm.CONFIG["multitasking"] == False` -> 1 worker (sequential).
    3. Default (4): empirically saturates NFSv3 CREATE pipelining; more
       workers don't help and add thread-startup overhead.
    """
    from . import _shared
    config = _shared.CONFIG
    override = config.get("track_create_parallel_writers")
    if override is not None:
        n = int(override)  # type: ignore[call-overload]
    elif not bool(config.get("multitasking", True)):
        n = 1
    else:
        n = _DEFAULT_TRACK_CREATE_PARALLEL_WRITERS
    if n < 1:
        n = 1
    _pymisha.pm_set_create_parallel_writers(n)


_CREATE_DENSE_FUNCS = (
    "weighted.mean",
    "weighted.sum",
    "max",
    "min",
    "median",
    "count",
    "coverage",
)


def gtrack_create_dense(
    track: str,
    description: str,
    intervals: Intervals,
    values: Any,
    binsize: int,
    defval: float = np.nan,
    func: str = "weighted.mean",
) -> None:
    """
    Create a Dense (fixed-bin) track from intervals and values.

    Creates a new Dense track whose genome is tiled into fixed-size bins.
    Each bin stores a single numeric value reduced from the intervals
    overlapping that bin, plus an optional synthetic uncovered
    contribution at value *defval*. The description is stored as a track
    attribute.

    Parameters
    ----------
    track : str
        Name for the new track. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.
    description : str
        Human-readable description stored as a track attribute.
    intervals : pandas.DataFrame
        One-dimensional intervals with columns ``chrom``, ``start``,
        ``end``.
    values : array-like of float
        Numeric values, one per interval. Length must match the number
        of rows in *intervals*.
    binsize : int
        Bin size in base pairs. Must be a positive integer.
    defval : float, default numpy.nan
        Default value for bins not covered by any interval. Acts as a
        synthetic contribution with value=defval and
        overlap=uncovered_bases for every ``func`` except ``count``.
    func : str, default "weighted.mean"
        Per-bin reduction over the intervals overlapping each bin. One
        of:

        ``"weighted.mean"``
            ``sum(v_i * ov_i) / sum(ov_i)`` (default; byte-identical to
            the historical behavior).
        ``"weighted.sum"``
            ``sum(v_i * ov_i)`` - coverage-weighted integral.
        ``"max"`` / ``"min"``
            Unweighted reduction over interval values touching the bin.
        ``"median"``
            Overlap-weighted (lower) median by coverage mass.
        ``"count"``
            Number of intervals touching the bin. Empty bin = 0.
            ``defval`` does not contribute.
        ``"coverage"``
            ``sum(v_i * ov_i / binsize)`` - per-base average signal in
            the bin. With ``values=[1]*N`` and ``defval=0`` this
            produces a ChIP-seq-style pileup track in one call.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, binsize is not positive, values
        length does not match intervals, ``func`` is not one of the
        seven supported reductions, or no intervals map to known
        chromosomes.

    See Also
    --------
    gtrack_create_sparse : Create a Sparse track.
    gtrack_create : Create a track from a track expression.
    gtrack_modify : Modify values of an existing Dense track.
    gtrack_import : Import a track from a file.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    >>> pm.gtrack_create_dense("test_dn", "Test", intervs, [5.0], 50)  # doctest: +SKIP
    >>> pm.gtrack_rm("test_dn", force=True)  # doctest: +SKIP
    """
    _create_dense(track, description, intervals, values, binsize, defval, func)


def _create_dense(track: str, description: str, intervals: Intervals, values: Any, binsize: int,
                  defval: float = np.nan, func: str = "weighted.mean", r_import: bool = False) -> None:
    """gtrack_create_dense. With *r_import* (gtrack_import), no intervals at all make an empty
    track, and the weighted mean multiplies each value by its overlap in float32, as misha's
    import does (gtrack.create_dense multiplies in double). The other funcs have no counterpart
    in misha's import and keep gtrack_create_dense's arithmetic."""
    _checkroot()
    _validate_track_name(track)

    if not isinstance(func, str) or func not in _CREATE_DENSE_FUNCS:
        raise ValueError(
            f"Invalid 'func': must be one of {_CREATE_DENSE_FUNCS}, got {func!r}"
        )
    func_canonical = "weighted.median" if func == "median" else func

    binsize = int(binsize)
    if binsize <= 0:
        raise ValueError("binsize must be a positive integer")
    defval = float(defval)

    if r_import and len(intervals) == 0:
        data = _empty_records()
    else:
        data = _normalize_intervals_df(intervals)
        vals = np.asarray(values, dtype=np.float64)
        if len(vals) != len(data):
            raise ValueError("Length of values must match number of intervals")
        # _normalize_intervals_df already returns a copy; no need to copy again
        data["value"] = vals
        data = _canonicalize_known_chroms(data)
        if len(data) == 0:
            raise ValueError("No intervals map to known chromosomes")

    with _atomic_track_create(track):
        _pymisha.pm_track_create_dense(
            track, _df2pymisha(data), int(binsize), float(defval), func_canonical, r_import
        )

    # On indexed DBs the C++ writer (pm_track_create_dense) already
    # produced track.dat + track.idx directly, so we skip the post-create
    # convert step. Byte-identical to the per-chrom + convert pipeline.
    if func == "weighted.mean":
        created_by = (
            f'gtrack.create_dense("{track}", description, intervals, values, '
            f"{binsize}, {defval:g})"
        )
    else:
        created_by = (
            f'gtrack.create_dense("{track}", description, intervals, values, '
            f'{binsize}, {defval:g}, func="{func}")'
        )
    track_dir = _track_dir_for_create(track)
    try:
        # Write created.* + type/binsize in one go, then a single dbreload
        # registers track + attrs together. Saves ~215 ms vs the
        # dbreload-attrs-dbreload pattern on hg38 (15k tracks).
        _write_created_attrs_at_path(
            track_dir,
            description,
            created_by,
            {"type": "dense", "binsize": str(binsize)},
        )
        _pm_dbreload(_target_root())
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=user_stacklevel(),
        )
        raise


@_shared._with_umask()   # database writes carry misha's permissions
def gtrack_create_dense_direct(
    track: str,
    description: str,
    intervals: Intervals,
    values: Any,
    binsize: int,
    defval: float = np.nan,
    reload: bool = True,
) -> None:
    """
    Create a Dense track by writing binary files directly.

    This is functionally equivalent to :func:`gtrack_create_dense` but
    bypasses the C++ bridge, writing per-chromosome binary files and the
    ``.attributes`` metadata file directly to the track directory.
    This avoids the overhead of the Python/C++ bridge and intermediate
    DataFrame copies, making it significantly faster for large datasets.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    intervals : pandas.DataFrame
        One-dimensional intervals with columns ``chrom``, ``start``,
        ``end``.
    values : array-like of float
        Numeric values, one per interval. Length must match the number
        of rows in *intervals*.
    binsize : int
        Bin size in base pairs. Must be a positive integer.
    defval : float, default numpy.nan
        Default value for bins not covered by any interval.
    reload : bool, default True
        If True, reload the genome database after writing. Set to False
        when creating many tracks in a batch and call
        ``gdb_reload()`` once at the end.

    Returns
    -------
    None

    See Also
    --------
    gtrack_create_dense : Create a Dense track via the C++ bridge.
    """
    _checkroot()
    _validate_track_name(track)

    binsize = int(binsize)
    if binsize <= 0:
        raise ValueError("binsize must be a positive integer")
    defval = float(defval)

    data = _normalize_intervals_df(intervals)
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) != len(data):
        raise ValueError("Length of values must match number of intervals")

    data["value"] = vals
    data = _canonicalize_known_chroms(data)
    if len(data) == 0:
        raise ValueError("No intervals map to known chromosomes")

    # Get chromosome sizes
    from .intervals import gintervals_all

    chrom_sizes_df = gintervals_all()
    chrom_sizes = {
        c: int(e)
        for c, e in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)
    }

    with _atomic_track_create(track) as track_dir:
        track_dir.mkdir(parents=True, exist_ok=True)
        vars_dir = track_dir / "vars"
        vars_dir.mkdir(exist_ok=True)

        # Group intervals by chromosome
        grouped = data.groupby("chrom", sort=False)

        for chrom, chrom_size in chrom_sizes.items():
            num_bins = math.ceil(chrom_size / binsize)
            dense = np.full(num_bins, defval, dtype=np.float32)

            if chrom in grouped.groups:
                grp = grouped.get_group(chrom)
                starts = grp["start"].values.astype(np.int64)
                ends = grp["end"].values.astype(np.int64)
                grp_vals = grp["value"].values.astype(np.float64)

                # Use an accumulator in float64 to collect weighted contributions.
                # Initialize to 0.0 - we merge with defval after all intervals.
                accum = np.zeros(num_bins, dtype=np.float64)
                touched = np.zeros(num_bins, dtype=np.bool_)

                # Vectorized bin assignment for all intervals at once
                fb = np.maximum(starts // binsize, 0).astype(np.intp)
                tb = np.minimum(
                    np.ceil(ends / binsize).astype(np.intp) - 1,
                    num_bins - 1,
                )

                # Single-bin intervals (fb == tb): one fractional contribution
                single = fb >= tb
                if np.any(single):
                    s_fb = fb[single]
                    fracs = (ends[single] - starts[single]) / binsize
                    np.add.at(accum, s_fb, grp_vals[single] * fracs)
                    touched[s_fb] = True

                # Multi-bin intervals: first-bin, last-bin, and full bins
                multi = ~single
                if np.any(multi):
                    m_fb = fb[multi]
                    m_tb = tb[multi]
                    m_s = starts[multi]
                    m_e = ends[multi]
                    m_v = grp_vals[multi]

                    # First-bin fraction
                    f_frac = ((m_fb + 1) * binsize - m_s) / binsize
                    np.add.at(accum, m_fb, m_v * f_frac)
                    touched[m_fb] = True

                    # Last-bin fraction
                    l_frac = (m_e - m_tb * binsize) / binsize
                    np.add.at(accum, m_tb, m_v * l_frac)
                    touched[m_tb] = True

                    # Full bins in between (value * 1.0 for each)
                    for j in range(len(m_fb)):
                        if m_tb[j] > m_fb[j] + 1:
                            accum[m_fb[j] + 1 : m_tb[j]] += m_v[j]
                            touched[m_fb[j] + 1 : m_tb[j]] = True

                # Merge: touched bins get accumulated value, rest keep defval
                dense[touched] = accum[touched].astype(np.float32)

            # Write binary: [uint32 binsize][float32 x num_bins]
            out_path = track_dir / chrom
            with open(out_path, "wb") as fout:
                fout.write(struct.pack("<I", binsize))
                dense.tofile(fout)

        # Write .attributes (null-separated binary format)
        now_str = _datetime.datetime.now().ctime()
        user = _getpass.getuser()
        created_by = (
            f'gtrack.create_dense("{track}", description, intervals, '
            f"values, {binsize}, {defval:g})"
        )
        attrs = {
            "created.by": created_by,
            "created.date": now_str,
            "created.user": user,
            "description": str(description),
            "type": "dense",
            "binsize": str(binsize),
        }
        parts = []
        for key, value in sorted(attrs.items()):
            parts.append(key.encode("utf-8"))
            parts.append(b"\x00")
            parts.append(str(value).encode("utf-8"))
            parts.append(b"\x00")
        with open(track_dir / ".attributes", "wb") as f:
            f.write(b"".join(parts))

    if reload:
        try:
            _pm_dbreload(_target_root())
            if _db_is_indexed(_shared._GROOT):
                _index_created_track(track, remove_old=False)
                _pm_dbreload(_target_root())
        except Exception as exc:
            warnings.warn(
                f"post-create steps failed for track '{track}': {exc}; "
                "the track was created but may have incomplete attributes",
                stacklevel=2,
            )
            raise


def gtrack_modify(track: str, expr: str, intervals: Intervals | None = None) -> None:
    """
    Modify a Dense track's values in-place by evaluating an expression.

    Overwrites the values of an existing Dense track with the result of
    evaluating *expr*. The iterator policy is automatically set to the
    track's bin size. Only Dense (fixed-bin) tracks are supported.

    Parameters
    ----------
    track : str
        Name of the dense track to modify.
    expr : str
        Track expression to evaluate (may reference the track itself).
    intervals : pandas.DataFrame or None, optional
        Genomic scope for modification. If None, the entire genome
        (ALLGENOME) is used.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist, is not a Dense track, or *expr* is
        None.

    See Also
    --------
    gtrack_create : Create a new track from a track expression.
    gtrack_smooth : Create a smoothed copy of a track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_modify("dense_track", "dense_track * 2")  # doctest: +SKIP
    >>> pm.gtrack_modify("dense_track", "dense_track / 2")  # doctest: +SKIP
    """
    from .intervals import gintervals_all

    _checkroot()
    if not isinstance(track, str) or not track:
        raise ValueError("track must be a non-empty string")
    if expr is None:
        raise ValueError("expr cannot be None")
    if not _pymisha.pm_track_info(track):
        raise ValueError(f"Track '{track}' does not exist")

    info = gtrack_info(track)
    if info.get("type") != "dense":
        raise ValueError(f"gtrack_modify only supports dense tracks, got '{info.get('type')}'")

    binsize = int(info["bin_size"])

    if intervals is None:
        intervals = gintervals_all()

    from .extract import _maybe_load_intervals_set
    intervals = _maybe_load_intervals_set(intervals)

    vtracks_dict = _resolve_vtracks_for_cpp_expr(expr, "gtrack_modify")

    # Staged, like every other writer: the C++ copies the data files the
    # modification touches into stage_dir, writes the new values there, and
    # renames them back over the originals only once the whole modification has
    # succeeded. Until then the track on disk is the old one, so an interrupt, a
    # bad expression or a full disk leaves it intact instead of durably
    # half-old/half-new under its real name. This was the only pymisha writer
    # that edited in place.
    track_dir = Path(_pymisha.pm_track_path(track))
    # in a subdirectory (gdir_cd), not an indexed track of no loaded database (reached through
    # a link or in a database kept below the working directory), whose chromosome order the
    # session may not know
    if _shared._gwd_is_subdir() and (track_dir / "track.idx").exists() and _shared._path_db(track_dir) is None:
        raise ValueError(f"Track {track} is in no loaded database; modify it in a session of the database it is in.")
    parent = track_dir.parent
    _gdb_trash_sweep_old(parent)
    stage_dir = parent / f".{track_dir.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}"
    with _shared._with_umask():
        stage_dir.mkdir(parents=True)
    try:
        _pymisha.pm_modify(
            track, str(expr), _df2pymisha(intervals), binsize, vtracks_dict, str(stage_dir)
        )
    finally:
        _gdb_trash(stage_dir, async_unlink=True)
        # The staged copy was opened through the track-index cache under its own
        # path; drop the entry so nothing later resolves that dead directory.
        _shared._invalidate_dir_cache(stage_dir, track_dir)

    # Update created.by attribute (bypass readonly check for internal update)
    modify_str = f'gtrack.modify({track}, {str(expr)}, intervs)'
    attrs = _load_track_attributes(track)
    existing = attrs.get("created.by", "")
    if existing:
        attrs["created.by"] = existing + "\n" + modify_str
    else:
        attrs["created.by"] = modify_str
    _save_track_attributes(track, attrs)


def gtrack_smooth(
    track: str,
    description: str,
    expr: str,
    winsize: float,
    weight_thr: float = 0,
    smooth_nans: bool = False,
    alg: str = "LINEAR_RAMP",
    iterator: int | None = None,
) -> None:
    """
    Create a new Dense track with smoothed values from a track expression.

    Each output bin at coordinate C is computed by smoothing the non-NaN
    values of *expr* within a window of size *winsize* (in coordinate
    units) around C. The smoothing algorithm and handling of NaN /
    edge-of-chromosome gaps are controlled by the remaining parameters.

    Parameters
    ----------
    track : str
        Name of the new track to create.
    description : str
        Human-readable description stored as a track attribute.
    expr : str
        Track expression whose values are smoothed.
    winsize : float
        Smoothing window size in coordinate units. Defines the total
        region considered on both sides of the central point.
    weight_thr : float, default 0
        Weight sum threshold below which the smoothed value is NaN
        instead of a partial-window estimate.
    smooth_nans : bool, default False
        If False, output NaN whenever the central window value is NaN,
        regardless of *weight_thr*. If True, NaN center values are
        filled from surrounding non-NaN values.
    alg : str, default ``"LINEAR_RAMP"``
        Smoothing algorithm. ``"LINEAR_RAMP"`` uses a weighted average
        with linearly decreasing weights. ``"MEAN"`` uses a simple
        arithmetic average.
    iterator : int or None, optional
        Fixed-bin iterator bin size for the new track. If None, the bin
        size is inferred from the track expression.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, *expr* is None, *winsize* is not
        positive, or *alg* is not one of the supported algorithms.

    See Also
    --------
    gtrack_create : Create a track from a track expression.
    gtrack_modify : Modify an existing Dense track in-place.
    gtrack_create_sparse : Create a Sparse track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_smooth("smoothed", "Test", "dense_track", 500)  # doctest: +SKIP
    >>> pm.gtrack_rm("smoothed", force=True)  # doctest: +SKIP
    """
    from .intervals import gintervals_all

    _checkroot()
    if expr is None:
        raise ValueError("expr cannot be None")
    if winsize is None or winsize <= 0:
        raise ValueError("winsize must be a positive number")
    if alg not in ("LINEAR_RAMP", "MEAN"):
        raise ValueError(f"Invalid algorithm '{alg}'. Use 'LINEAR_RAMP' or 'MEAN'.")

    _validate_track_name(track)

    all_intervs = gintervals_all()

    # Determine iterator: if None, infer from the expression (use track name as iterator)
    iter_val = iterator
    if iter_val is None:
        # Try to infer from expression - use expr as iterator policy
        # The C++ scanner will resolve the track's bin size
        iter_val = 0  # Let C++ determine from expression

    # Handle DataFrame-as-iterator
    all_intervs, iter_val, _itr_id_map = _preprocess_intervals_iterator(all_intervs, iter_val)

    vtracks_dict = _resolve_vtracks_for_cpp_expr(expr, "gtrack_smooth")

    with _atomic_track_create(track):
        _pymisha.pm_smooth(
            track, str(expr), _df2pymisha(all_intervs),
            iter_val, float(winsize), float(weight_thr),
            int(bool(smooth_nans)), alg, vtracks_dict,
        )

    try:
        _pm_dbreload(_target_root())
        created_by = (
            f'gtrack.smooth({track}, description, {str(expr)}, {winsize}, '
            f'{weight_thr}, {smooth_nans}, {alg})'
        )
        _set_created_attrs(track, description, created_by)
        new_info = gtrack_info(track)
        if new_info.get("type") == "dense":
            gtrack_attr_set(track, "type", "dense")
            if "bin_size" in new_info:
                gtrack_attr_set(track, "binsize", str(int(new_info["bin_size"])))
        if _db_is_indexed(_shared._GROOT):
            _index_created_track(track, remove_old=False)
        _pm_dbreload(_target_root())
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=2,
        )
        raise


def _gtrack_create_2d_from_expr(
    track: str,
    description: str,
    expr: str,
    iterator: Any = None,
) -> None:
    """Create a 2D RECTS/POINTS track from a 2D track expression.

    Evaluates *expr* over the 2D scope (the source track's native rectangles
    when *iterator* is ``None``, or an explicit 2D-intervals iterator) and
    writes one rectangle per iterated cell with the expression's value -
    R's ``gtrack.create(track, desc, "<2d expr>"[, iterator])``.
    """
    from .extract import gextract
    from .intervals import _gintervals_2d_all

    if iterator is None:
        scope: Any = _gintervals_2d_all("full")
        res = gextract(expr, scope, colnames=["value"])
    elif isinstance(iterator, pd.DataFrame) and "chrom1" in iterator.columns:
        # Explicit 2D iterator set: its rectangles are the cells to evaluate.
        res = gextract(expr, iterator, iterator=iterator, colnames=["value"])
    else:
        raise ValueError(
            "A 2D track expression requires either no iterator (use the source "
            "track's rectangles) or a 2D-intervals iterator"
        )

    coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    if res is None or len(res) == 0:
        raise ValueError(
            f"2D track expression {expr!r} produced no intervals to create a track"
        )

    intervals_2d = res[coord_cols].copy()
    values = res["value"].to_numpy()

    # gtrack_2d_create handles its own (atomic) track-dir creation, attribute
    # writing, dbreload and indexed-format conversion - it must NOT run inside
    # _atomic_track_create (which is for the C++ pm_track_create_* writers that
    # honour the dir override).
    gtrack_2d_create(track, description, intervals_2d, values)


def gtrack_create(
    track: str,
    description: str,
    expr: str,
    iterator: int | None = None,
    band: tuple[int, int] | None = None,
) -> None:
    """
    Create a track from a track expression.

    Creates a new track whose values are determined by evaluating *expr*
    over the entire genome. The type of the new track (Dense, Sparse, or
    Rectangles) is determined by the iterator policy. The description is
    stored as a track attribute.

    Parameters
    ----------
    track : str
        Name for the new track. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.
    description : str
        Human-readable description stored as a track attribute.
    expr : str
        Numeric track expression to evaluate.
    iterator : int or None, optional
        Fixed-bin iterator bin size. If None, the iterator is determined
        implicitly from the track expression.
    band : tuple or None, optional
        Diagonal band ``(d1, d2)`` for 2D track creation.  When provided,
        only contacts where ``d1 <= (x - y) < d2`` are stored.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, *expr* is None, or *band* is not
        None.

    See Also
    --------
    gtrack_create_sparse : Create a Sparse track from intervals/values.
    gtrack_create_dense : Create a Dense track from intervals/values.
    gtrack_2d_create : Create a 2D track.
    gtrack_smooth : Create a smoothed track.
    gtrack_modify : Modify an existing Dense track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_create("mixed", "Test", "dense_track * 2", iterator=70)  # doctest: +SKIP
    >>> pm.gtrack_info("mixed")  # doctest: +SKIP
    >>> pm.gtrack_rm("mixed", force=True)  # doctest: +SKIP
    """
    from .intervals import _gintervals_2d_all, gintervals_all

    _checkroot()
    if expr is None:
        raise ValueError("expr cannot be None")
    _validate_track_name(track)

    # 2D track expression (e.g. "test.rects+10"): the 1D scanner cannot iterate
    # a rectangles track, so evaluate the expression over the 2D scope and write
    # a 2D RECTS/POINTS track (R's gtrack.create on a 2D expression).
    from .expr import _expr_is_2d
    if _expr_is_2d(expr):
        _gtrack_create_2d_from_expr(track, description, str(expr), iterator)
        return

    if band is not None:
        from .extract import _validate_band
        band = _validate_band(band)
        # A band keeps only intra-chromosomal pairs, so build only those.
        all_intervs = _gintervals_2d_all("diagonal")
        if all_intervs is None or len(all_intervs) == 0:
            raise ValueError(
                "band requires a genome with 2D intervals "
                "(no 2D intervals available)"
            )
        from .intervals import gintervals_2d_band_intersect
        banded = gintervals_2d_band_intersect(all_intervs, band)
        if banded is None or len(banded) == 0:
            raise ValueError("band filter produced no 2D intervals")
        import pandas as _pd
        axis1 = banded[["chrom1", "start1", "end1"]].rename(
            columns={"chrom1": "chrom", "start1": "start", "end1": "end"}
        )
        axis2 = banded[["chrom2", "start2", "end2"]].rename(
            columns={"chrom2": "chrom", "start2": "start", "end2": "end"}
        )
        from .intervals import gintervals_canonic
        all_intervs = gintervals_canonic(
            _pd.concat([axis1, axis2], ignore_index=True)
        )
    else:
        all_intervs = gintervals_all()

    # Handle DataFrame-as-iterator
    all_intervs, iterator, _itr_id_map = _preprocess_intervals_iterator(all_intervs, iterator)

    vtracks_dict = _resolve_vtracks_for_cpp_expr(expr, "gtrack_create")

    with _config_no_mt(_itr_id_map) as _cfg, _atomic_track_create(track):
        _pymisha.pm_track_create_expr(
            track, str(expr), _df2pymisha(all_intervs), iterator, _cfg, vtracks_dict
        )

    try:
        _pm_dbreload(_target_root())
        _set_created_attrs(
            track,
            description,
            f'gtrack.create("{track}", description, {str(expr)!r}, iterator={iterator!r})',
        )
        info = gtrack_info(track)
        if info.get("type") == "dense":
            gtrack_attr_set(track, "type", "dense")
            if "bin_size" in info:
                gtrack_attr_set(track, "binsize", str(int(info["bin_size"])))
        if _db_is_indexed(_shared._GROOT):
            _index_created_track(track, remove_old=False)
        _pm_dbreload(_target_root())
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=2,
        )
        raise


def _load_pssm_from_db(pssmset: str, pssmid: int) -> NumpyArray:
    """Load a PSSM matrix from the database's pssms/ directory.

    Reads ``GROOT/pssms/{pssmset}.key`` and ``GROOT/pssms/{pssmset}.data``.

    Parameters
    ----------
    pssmset : str
        Name of the PSSM set (file basename without extension).
    pssmid : int
        Numeric ID of the PSSM within the set.

    Returns
    -------
    numpy.ndarray
        PSSM matrix of shape (L, 4) with columns A, C, G, T.
    """
    import numpy as np

    groot = _shared._GROOT
    if groot is None:
        raise ValueError("No genome database is initialized")

    pssm_dir = Path(groot) / "pssms"
    key_file = pssm_dir / f"{pssmset}.key"
    data_file = pssm_dir / f"{pssmset}.data"

    if not key_file.exists():
        raise FileNotFoundError(f"PSSM key file not found: {key_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"PSSM data file not found: {data_file}")

    # Verify the pssmid exists in the key file
    pssmid = int(pssmid)
    found = False
    with open(key_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 1 and int(parts[0]) == pssmid:
                found = True
                break
    if not found:
        raise ValueError(f"PSSM id {pssmid} not found in {key_file}")

    # Read the data file and extract rows for our pssmid
    positions = {}
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            row_id = int(parts[0])
            if row_id != pssmid:
                continue
            pos = int(parts[1])
            a, c, g, t = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            positions[pos] = (a, c, g, t)

    if not positions:
        raise ValueError(f"No data found for PSSM id {pssmid} in {data_file}")

    max_pos = max(positions.keys())
    pssm = np.full((max_pos + 1, 4), 0.25)
    for pos, (a, c, g, t) in positions.items():
        pssm[pos] = [a, c, g, t]

    return pssm


def gtrack_create_pwm_energy(
    track: str,
    description: str,
    pssmset: str,
    pssmid: int,
    prior: float,
    iterator: int,
) -> None:
    """
    Create a track from a PSSM energy function.

    Creates a new Dense track with values of a PSSM energy function
    (log-sum-exp scoring). PSSM parameters are read from
    ``{pssmset}.key`` and ``{pssmset}.data`` files in ``GROOT/pssms/``.
    Internally creates a temporary PWM virtual track, extracts values at
    the given iterator resolution, and writes them to a new Dense track.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    pssmset : str
        Name of PSSM set. Files ``{pssmset}.key`` and ``{pssmset}.data``
        must exist in ``GROOT/pssms/``.
    pssmid : int
        PSSM id within the set.
    prior : float
        Dirichlet prior for the PSSM.
    iterator : int
        Fixed-bin iterator bin size for the new track. Must be a
        positive integer.

    Raises
    ------
    ValueError
        If the track already exists, any required argument is None,
        *iterator* is not positive, or the PSSM set/id is not found.
    FileNotFoundError
        If the PSSM key or data file does not exist.

    Returns
    -------
    None

    See Also
    --------
    gtrack_create : Create a track from a general track expression.
    gtrack_create_dense : Create a Dense track from intervals/values.
    gtrack_smooth : Create a smoothed track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_create_pwm_energy(  # doctest: +SKIP
    ...     "pwm_track", "Test", "pssm", 3, 0.01, iterator=100
    ... )
    >>> pm.gtrack_rm("pwm_track", force=True)  # doctest: +SKIP
    """
    from .intervals import gintervals_all

    _checkroot()

    if track is None or description is None or pssmset is None or pssmid is None or prior is None or iterator is None:
        raise ValueError(
            "Usage: gtrack_create_pwm_energy(track, description, pssmset, pssmid, prior, iterator)"
        )

    _validate_track_name(track)
    _ensure_track_absent(track)

    iterator = int(iterator)
    if iterator <= 0:
        raise ValueError("iterator must be a positive integer")

    pssm = _load_pssm_from_db(pssmset, int(pssmid))

    # Create a temporary PWM virtual track and extract values.
    # The vtrack path uses C++ pm_vtrack_compute for scoring (already fast)
    # and GAP-019's pre-computed vtrack optimization avoids per-chunk recomputation.
    from .extract import gextract
    from .vtracks import gvtrack_create, gvtrack_rm

    vtrack_name = f"_pm_pwm_tmp_{track}"
    try:
        gvtrack_create(vtrack_name, None, func="pwm", pssm=pssm, prior=float(prior))
        all_intervs = gintervals_all()
        df = gextract(vtrack_name, intervals=all_intervs, iterator=iterator)
        if df is None or len(df) == 0:
            raise ValueError("No values extracted for PWM energy track")

        gtrack_create_dense(
            track, description, df[["chrom", "start", "end"]],
            df[vtrack_name].values, iterator, defval=np.nan,
        )

        # Overwrite the created.by attribute to match R's format
        created_by = (
            f'gtrack.create_pwm_energy("{track}", description, '
            f'"{pssmset}", {int(pssmid)}, {float(prior)}, {iterator})'
        )
        # Bypass readonly check for internal overwrite
        attrs = _load_track_attributes(track)
        attrs["created.by"] = created_by
        _save_track_attributes(track, attrs)
    except Exception:
        # Clean up if track was partially created. Stays broad: this runs while
        # another exception is in flight and must not replace it.
        try:
            gtrack_rm(track, force=True)
        except Exception:
            _logger.warning("could not remove the partially created track %r", track, exc_info=True)
        raise
    finally:
        try:
            gvtrack_rm(vtrack_name)
        except Exception:
            _logger.warning("could not remove the temporary virtual track %r", vtrack_name,
                            exc_info=True)


def gtrack_import(
    track: str,
    description: str,
    file: str,
    binsize: int | None = None,
    defval: float = np.nan,
    attrs: dict[str, str] | None = None,
    func: str = "weighted.mean",
) -> None:
    """
    Create a track from a WIG, BigWig, BedGraph, BED, or tab-delimited file.

    Parses the input file and creates either a Sparse or Dense track
    depending on *binsize*. File format is detected from the extension; a
    bigWig is also recognized by its first bytes, whatever its name.
    Compressed files (``.gz``, or a ``.zip`` holding one file) are unzipped
    to a temporary file first, and the format is detected on it, so a
    compressed bigWig is read too. Tab-delimited files must have a header
    with columns ``chrom``, ``start``, ``end``, and exactly one value column.
    A WIG / bedGraph / tab-delimited file with no records at all makes an
    empty track; one whose records are all on chromosomes the database
    lacks is an error.

    BigWig files are read directly at full resolution and full float
    precision (zoom levels are not used). A truncated or corrupt bigWig is
    an error. A bigWig imports bin for bin like the WIG / bedGraph text it
    was made from, and both follow the same rules: the records of a
    chromosome must be together, sorted and must not overlap, a bedGraph
    record must fit its chromosome, and a fixedStep / variableStep record
    that crosses the chromosome end is clipped to it. An empty bedGraph
    record is an error in text and skipped in a bigWig. Two bigWig
    chromosomes that name the same database chromosome (``chr1`` and ``1``)
    are an error. A WIG file may mix fixedStep / variableStep sections and
    bedGraph lines; a bedGraph line ends the section before it.

    Dense bins are computed as R misha's gtrack.import computes them, which
    rounds a little differently from :func:`gtrack_create_dense` for the
    default ``func``.

    A Parquet file (recognized by its first bytes, whatever its name) is
    imported when it has exactly one value column besides ``chrom``,
    ``start`` and ``end``; it always makes a Dense track, so *binsize* must
    be > 0. Use :func:`gtrack_import_parquet` for several columns.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    file : str
        Path to the input file. Supported extensions: ``.wig``,
        ``.bedgraph``, ``.bed``, ``.bw`` / ``.bigwig``, or tab-delimited
        (any other extension). May include ``.gz`` or ``.zip`` suffix.
        A Parquet file is recognized by its content.
    binsize : int or None, optional
        Bin size for a Dense track. If None or 0, a Sparse track is
        created. If positive, a Dense track with the given bin size is
        created.
    defval : float, default numpy.nan
        Default value for Dense track bins not covered by any interval.
        Ignored when creating Sparse tracks.
    attrs : dict or None, optional
        Additional attributes to set on the track after import, as a
        dict mapping attribute names to string values.
    func : str, default "weighted.mean"
        Per-bin reduction applied when creating a Dense track (i.e. when
        *binsize* is positive). Forwarded to :func:`gtrack_create_dense`;
        one of ``"weighted.mean"``, ``"weighted.sum"``, ``"max"``,
        ``"min"``, ``"median"``, ``"count"``, ``"coverage"``. Passing
        ``func="coverage"`` with ``values`` of 1 yields a ChIP-style
        pileup track in one call. Must be left at the default when no
        *binsize* is given (the Sparse path has no per-bin reduction).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, *file* is None, the file
        contains no valid intervals, or WIG / bedGraph records break
        the rules above.
    pymisha.error
        If a bigWig is truncated, corrupt or breaks the rules above.

    See Also
    --------
    gtrack_import_set : Batch-import multiple files into tracks.
    gtrack_import_parquet : Import the value columns of Parquet files.
    gtrack_create_sparse : Create a Sparse track programmatically.
    gtrack_create_dense : Create a Dense track programmatically.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_import("wig_track", "From WIG", "data.wig", binsize=10)
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)
    if file is None:
        raise ValueError("file cannot be None")
    file_path = str(file)
    if _is_parquet_file(file_path):
        _gtrack_import_parquet_file(track, description, file_path, binsize, defval, attrs, func)
        return

    binsize = int(binsize or 0)
    if binsize <= 0 and func != "weighted.mean":
        raise ValueError(
            f"func={func!r} requires a positive binsize; the sparse import path has no per-bin reduction"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # As in misha, a .gz / .zip file is unzipped first and its format found on the
        # plain copy, so a compressed bigWig reads like any other.
        lower = file_path.lower()
        src = _unzip_import_file(file_path, tmpdir) if lower.endswith((".gz", ".zip")) else file_path
        if lower.endswith((".bed", ".bed.gz", ".bed.zip")):
            parsed = _parse_bed(src)
            _report_skipped_chroms(parsed, file_path)
        elif _pymisha.pm_is_bigwig(src):
            parsed = _parse_bigwig(src, file_path)   # reports its skipped chromosomes itself
        elif src.lower().endswith((".wig", ".bedgraph")):
            parsed = _parse_wig_or_bedgraph(src)
            _report_skipped_chroms(parsed, file_path)
            parsed = _check_wig_records(parsed, file_path)
        else:
            parsed = _parse_tabular_track(src)
            _report_skipped_chroms(parsed, file_path)

    # r_import: misha's import arithmetic, and a file without records makes an empty track
    if binsize > 0:
        # func validation (allowed values) is delegated to _create_dense.
        _create_dense(track, description, parsed[["chrom", "start", "end"]], parsed["value"], binsize, defval,
                      func=func, r_import=True)
    else:
        _create_sparse(track, description, parsed[["chrom", "start", "end"]], parsed["value"], r_import=True)

    func_suffix = "" if func == "weighted.mean" else f', func="{func}"'
    created_by = (
        f'gtrack.import("{track}", description, "{file_path}", '
        f"{binsize}, {float(defval):g}, attrs{func_suffix})"
    )
    _set_created_attrs(track, description, created_by, attrs=attrs)


def _download_ftp_matches(path_pattern: str, tmpdir: str) -> list[str]:
    parsed = urlparse(path_pattern)
    if parsed.scheme.lower() != "ftp":
        raise ValueError("Only ftp:// URLs are supported for remote import sets")
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid FTP URL")
    allowed_hosts_env = os.environ.get("PYMISHA_ALLOWED_FTP_HOSTS", "").strip()
    if allowed_hosts_env:
        allowed_hosts = {h.strip() for h in allowed_hosts_env.split(",") if h.strip()}
        if host not in allowed_hosts:
            raise ValueError(
                f"FTP host '{host}' is not in PYMISHA_ALLOWED_FTP_HOSTS allow-list"
            )
    max_file_bytes = int(os.environ.get("PYMISHA_MAX_FTP_FILE_BYTES", str(512 * 1024 * 1024)))
    remote_path = parsed.path or "/"
    if "/" not in remote_path.strip("/"):
        remote_dir = "/"
        pattern = remote_path.lstrip("/")
    else:
        remote_dir = remote_path.rsplit("/", 1)[0] or "/"
        pattern = remote_path.rsplit("/", 1)[1]

    ftp = ftplib.FTP()
    ftp.connect(host, parsed.port or 21, timeout=30)
    ftp.login(parsed.username or "anonymous", parsed.password or "")
    try:
        names = ftp.nlst(remote_dir)
        matched = []
        for n in names:
            bn = os.path.basename(n)
            if fnmatch.fnmatch(bn, pattern):
                matched.append(n)

        out = []
        for remote in matched:
            remote_size = ftp.size(remote)
            if remote_size is None:
                raise ValueError(f"Could not determine FTP file size for '{remote}'")
            if remote_size > max_file_bytes:
                raise ValueError(
                    f"FTP file '{remote}' is too large ({remote_size} bytes > {max_file_bytes})"
                )
            local = Path(tmpdir) / os.path.basename(remote)
            with open(local, "wb") as f:
                ftp.retrbinary(f"RETR {remote}", f.write)
            out.append(str(local))
        return out
    finally:
        try:
            ftp.quit()
        except (ftplib.Error, OSError, EOFError):
            # ftplib.all_errors: a server that drops the control connection
            # rather than answering QUIT is routine.
            _logger.debug("FTP QUIT failed; closing the connection", exc_info=True)
            ftp.close()


def gtrack_import_set(
    description: str,
    path: str,
    binsize: int,
    track_prefix: str | None = None,
    defval: float = np.nan,
) -> dict[str, list[str]]:
    """
    Create one or more tracks from multiple WIG/BedGraph/BigWig/tab files.

    Similar to `gtrack_import` but operates on multiple files at once.
    Files can be specified by a local glob pattern or an FTP URL with
    wildcards. Each file produces one track named
    ``{track_prefix}{filestem}``. Existing tracks are skipped. The
    function continues importing even if individual files fail.

    Parameters
    ----------
    description : str
        Human-readable description stored as a track attribute on
        every imported track.
    path : str
        Local file glob pattern (e.g., ``"/data/*.wig"``) or FTP URL
        (e.g., ``"ftp://host/path/*.wig.gz"``).
    binsize : int
        Bin size for Dense tracks. If 0, Sparse tracks are created.
    track_prefix : str or None, optional
        Prefix prepended to each track name derived from the filename
        stem. If None, no prefix is used.
    defval : float, default numpy.nan
        Default value for Dense track bins not covered by any interval.

    Returns
    -------
    dict
        Dictionary with keys ``"files_imported"`` (list of successfully
        imported filenames) and/or ``"files_failed"`` (list of filenames
        that failed to import).

    Raises
    ------
    ValueError
        If *description*, *path*, or *binsize* is None, or no files
        match the pattern.

    See Also
    --------
    gtrack_import : Import a single file into a track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_import_set("Batch", "/data/*.wig", binsize=100, track_prefix="wigs.")
    """
    _checkroot()
    if description is None:
        raise ValueError("description cannot be None")
    if path is None:
        raise ValueError("path cannot be None")
    if binsize is None:
        raise ValueError("binsize cannot be None")

    track_prefix = "" if track_prefix is None else str(track_prefix)
    binsize = int(binsize)

    tmpdir = None
    files = []
    path_str = str(path)
    if path_str.lower().startswith("ftp://"):
        assert _shared._GROOT is not None
        downloads_root = Path(_shared._GROOT) / "downloads"
        downloads_root.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="pymisha-import-set-", dir=str(downloads_root))
        files = _download_ftp_matches(path_str, tmpdir)
    else:
        files = glob.glob(path_str)

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError("No files to import")

    imported = []
    failed = []
    try:
        for file in files:
            file_base = os.path.basename(file)
            stem = file_base.split(".", 1)[0]
            track_name = f"{track_prefix}{stem}"
            try:
                gtrack_import(track_name, description, file, binsize=binsize, defval=defval)
                imported.append(file_base)
            except Exception as exc:
                # R's gtrack.import_set reports each failed file's reason with
                # message(), which the user sees without configuring anything.
                # A per-file failure inside a bulk import is exactly the case a
                # silent log would hide, so warn to match.
                _logger.warning("could not import %s as track %r", file, track_name, exc_info=True)
                warnings.warn(
                    f"could not import {file} as track {track_name!r}: {exc}",
                    PymishaWarning,
                    stacklevel=user_stacklevel(),
                )
                failed.append(file_base)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    result = {}
    if failed:
        result["files_failed"] = failed
    if imported:
        result["files_imported"] = imported
    return result


_PARQUET_COORDS = ("chrom", "start", "end")
_PARQUET_COMPRESSED = (".gz", ".bz2", ".xz", ".lzma", ".zip")


def _open_binary_auto(path: str) -> Any:
    """*path* as a binary stream, decompressed as :func:`_open_text_auto` decompresses it."""
    lower = path.lower()
    if lower.endswith(".gz"):
        return gzip.open(path, "rb")
    if lower.endswith(".bz2"):
        return bz2.open(path, "rb")
    if lower.endswith((".xz", ".lzma")):
        return lzma.open(path, "rb")
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise ValueError(f"Zip file '{path}' does not contain a regular file")
            return zf.open(names[0], "r")  # keeps the archive's file open until it is closed
    return open(path, "rb")


def _is_parquet_file(path: str) -> bool:
    """Whether *path* (decompressed, if its name says so) starts with the Parquet magic "PAR1"."""
    try:
        with _open_binary_auto(path) as fh:
            return bool(fh.read(4) == b"PAR1")
    except (OSError, EOFError, ValueError, zipfile.BadZipFile, lzma.LZMAError):
        return False  # the parser of the file's format reports what is wrong with it


def _gtrack_import_parquet_file(
    track: str,
    description: str,
    file_path: str,
    binsize: int | None,
    defval: float,
    attrs: dict[str, str] | None,
    func: str,
) -> None:
    """gtrack_import of a Parquet file: its one value column as a Dense track."""
    if binsize is None or binsize == 0 or (isinstance(binsize, float) and math.isnan(binsize)):
        raise ValueError("Parquet import creates dense tracks; binsize must be > 0")
    if func != "weighted.mean":
        raise ValueError(f"func={func!r} is not supported for a Parquet file, which is binned by weighted.mean")

    with tempfile.TemporaryDirectory() as tmp:
        src = file_path
        if file_path.lower().endswith(_PARQUET_COMPRESSED):
            src = os.path.join(tmp, "unzipped.parquet")
            with _open_binary_auto(file_path) as fin, open(src, "wb") as fout:
                shutil.copyfileobj(fin, fout)

        columns = [n for n, _ in _pymisha.pm_parquet_columns(src, file_path) if n not in _PARQUET_COORDS]
        if len(columns) != 1:
            raise ValueError(
                f"gtrack_import needs a Parquet file with exactly one value column, but {file_path} has "
                f"{len(columns)} ({', '.join(columns)}); use gtrack_import_parquet to import several columns"
            )
        created_by = (
            f'gtrack.import("{track}", description, "{file_path}", '
            f"{int(binsize)}, {float(defval):g}, attrs)"
        )
        _import_parquet_tracks([track], columns, [src], [file_path], binsize, defval, description, [created_by], attrs)


def _import_parquet_tracks(
    tracks: list[str],
    columns: list[str],
    files: list[str],
    names: list[str],
    binsize: float,
    defval: float,
    description: str,
    created_by: list[str],
    attrs: dict[str, str] | None = None,
) -> None:
    """Import columns[i] of the files as Dense track tracks[i], all or nothing.

    Each track is written into a hidden staging directory and renamed into place only once
    all of them are complete; if anything fails, none of the tracks is left. names[i] is
    what messages call files[i].
    """
    root = _target_root()
    finals = [_track_dir_for_create(t) for t in tracks]
    staging = [f.parent / f".{f.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}" for f in finals]
    created: list[tuple[str, Path]] = []
    try:
        for parent in dict.fromkeys(f.parent for f in finals):
            with _shared._with_umask():
                parent.mkdir(parents=True, exist_ok=True)
            _gdb_trash_sweep_old(parent)
        for stage_dir in staging:
            with _shared._with_umask():
                stage_dir.mkdir()

        chrom_names, unknown_files = _pymisha.pm_parquet_import(
            [str(d) for d in staging], files, names, columns, float(binsize), float(defval)
        )

        for track, stage_dir, final, cb in zip(tracks, staging, finals, created_by, strict=True):
            _fsync_tree(stage_dir)
            if final.exists():
                raise ValueError(f"Refusing to overwrite existing {final}")
            os.rename(stage_dir, final)
            created.append((track, final))
            _fsync_dir(final.parent)
            _shared._invalidate_dir_cache(final)
            _write_created_attrs_at_path(final, description, cb, attrs)
        _pm_dbreload(root)
        if _db_is_indexed(root):
            for track in tracks:
                _index_created_track(track, remove_old=True)
    except BaseException:
        # BaseException: an interrupt must not leave half an import behind either
        for stage_dir in staging:
            if stage_dir.exists():
                _gdb_trash(stage_dir, async_unlink=True)
        for track, final in created:
            if not _gdb_trash(final, async_unlink=True) and final.exists():
                warnings.warn(
                    f"Track {track} post-create cleanup left residue at {final}; manual cleanup required",
                    PymishaWarning,
                    stacklevel=user_stacklevel(),
                )
        if created:
            try:
                _pm_dbreload(root)
            except Exception:
                _logger.warning("could not reload the database after removing %s", tracks, exc_info=True)
        raise

    # Only now, when the import can no longer fail
    _report_skipped_chroms(pd.DataFrame({"chrom": chrom_names}), unknown_files)


def gtrack_import_parquet(
    description: str,
    path: str,
    binsize: int,
    track_prefix: str | None = None,
    columns: str | list[str] | None = None,
    defval: float = np.nan,
) -> list[str]:
    """
    Create Dense tracks from the columns of Parquet files.

    Creates one Dense track per value column of one or more Parquet files.

    The files must have the interval columns ``chrom`` (string), ``start``
    and ``end`` (0-based, end exclusive; integers, or floating point numbers
    that are whole) and one or more value columns (integer or floating
    point), such as the files :func:`gextract` writes with
    ``file="x.parquet"`` or pandas' ``to_parquet``. *path* may contain
    wildcards, so that per-chromosome files import in one call; all the
    files must have the same value columns. Columns are matched by name,
    not by position.

    The track of column ``col`` is named ``{track_prefix}col``. All the
    track names are checked before anything is written: a name must start
    with a letter and hold only letters, digits, ``_`` and ``.`` (columns
    are never renamed; leave such a column out with *columns*). Existing
    tracks are not overwritten and no directories are created.

    Values are binned as :func:`gtrack_import` bins the same rows read from
    a tab-delimited file: each bin holds the mean of the values that overlap
    it, weighted by the overlap. Bases no row covers, and rows whose value
    is null or NaN, count as *defval* (unless it is NaN). The rows of a
    chromosome may be spread over several files and row groups in any
    order, but may not overlap. Chromosome names are matched through the
    chromosome alias chain as in :func:`gtrack_import`, with the same
    reporting of names the database does not have.

    The import is all or nothing: if it fails or is interrupted, none of
    the tracks is created.

    Supported: flat schemas; PLAIN, dictionary and BYTE_STREAM_SPLIT
    encodings; data pages v1 and v2; UNCOMPRESSED, SNAPPY and GZIP
    compression. Other compression codecs (e.g. ZSTD) and encodings (e.g.
    DELTA_BINARY_PACKED) are an error that names them. Unsigned integer
    columns are read as unsigned; decimal, date, time and timestamp columns
    are not numbers and are refused.

    Parameters
    ----------
    description : str
        Description stored as a track attribute of every created track.
    path : str
        Parquet file path (may contain wildcards).
    binsize : int
        Bin size of the new Dense tracks (> 0).
    track_prefix : str or None, optional
        Prefix of the track names.
    columns : str or list of str or None, optional
        Names of the value columns to import, in the order of the created
        tracks. With None every column other than ``chrom``, ``start`` and
        ``end`` is imported, and a column that is not numeric is an error.
    defval : float, default numpy.nan
        Default value of the bins no row covers.

    Returns
    -------
    list of str
        Names of the created tracks.

    Raises
    ------
    ValueError
        If an argument is invalid, no file matches *path*, the files do not
        have the columns, or a track name is invalid or exists.

    See Also
    --------
    gtrack_import : Import a single file into a track.
    gtrack_import_set : Batch-import multiple files into tracks.
    gextract : Write track expressions to a Parquet file.

    Examples
    --------
    >>> import os, tempfile
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> f = os.path.join(tempfile.mkdtemp(), "x.parquet")
    >>> pm.gextract(["dense_track", "dense_track * 2"], pm.gintervals("1", 0, 10000),
    ...             iterator=50, file=f, progress=False)
    >>> pm.gtrack_import_parquet("from parquet", f, 50, track_prefix="pq_", columns="dense_track")  # doctest: +SKIP
    ['pq_dense_track']
    >>> pm.gtrack_rm("pq_dense_track", force=True)  # doctest: +SKIP
    """
    _checkroot()
    if description is None:
        raise ValueError("description cannot be None")
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("path must be a single string (it may contain wildcards)")
    path = os.fspath(path)
    if (
        isinstance(binsize, bool)
        or not isinstance(binsize, numbers.Real)
        or not math.isfinite(binsize)
        or binsize < 0
        or binsize != int(binsize)
        or binsize >= 2**32
    ):
        raise ValueError("binsize must be a positive integer below 2^32")
    if binsize == 0:
        raise ValueError("Parquet import creates dense tracks; binsize must be > 0")
    if isinstance(defval, bool) or not isinstance(defval, numbers.Real):
        raise ValueError("defval must be a number")
    if track_prefix is None:
        track_prefix = ""
    if not isinstance(track_prefix, str):
        raise ValueError("track_prefix must be a string")

    files = sorted(f for f in glob.glob(path) if not os.path.isdir(f))
    if not files:
        if os.path.isdir(path):
            glob_path = os.path.join(path, "*.parquet")
            raise ValueError(f'{path} is a directory; to import the Parquet files in it use path = "{glob_path}"')
        raise ValueError(f"No files match {path}")

    if columns is None:
        info = [(n, t) for n, t in _pymisha.pm_parquet_columns(files[0], files[0]) if n not in _PARQUET_COORDS]
        bad = [n for n, t in info if t not in ("int32", "int64", "float", "double")]
        if bad:
            raise ValueError(
                f"Column(s) {', '.join(bad)} of {files[0]} are not numeric; choose the columns to import with 'columns'"
            )
        columns = [n for n, _ in info]
    elif isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if not columns or not all(isinstance(c, str) for c in columns):
        raise ValueError("No value columns to import")
    if any(c in _PARQUET_COORDS for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("columns must be distinct and cannot include chrom, start or end")

    # every file has the imported columns, and all have the same value columns (in any order)
    first_values = None
    for pq_file in files:
        file_cols = [n for n, _ in _pymisha.pm_parquet_columns(pq_file, pq_file)]
        absent = [c for c in (*_PARQUET_COORDS, *columns) if c not in file_cols]
        if absent:
            raise ValueError(f"File {pq_file} has no column(s) {', '.join(absent)}")
        file_values = sorted(set(file_cols) - set(_PARQUET_COORDS))
        if first_values is None:
            first_values = file_values
        elif file_values != first_values:
            raise ValueError(
                f"File {pq_file} has value columns {', '.join(file_values)}, but {files[0]} has "
                f"{', '.join(first_values)}: all files must have the same value columns"
            )

    tracks = [track_prefix + c for c in columns]
    # the rule gintervals_save applies to interval set names; dots are directories
    bad = [c for c, t in zip(columns, tracks, strict=True) if not re.fullmatch(r"[A-Za-z][\w.]*", t, re.ASCII)]
    if bad:
        raise ValueError(
            f"Column(s) {', '.join(f'{chr(34)}{c}{chr(34)}' for c in bad)} do not make valid track names with "
            f'track_prefix "{track_prefix}" (a letter, then letters, digits, _ or .). '
            "Leave them out with 'columns =' or rename them in the file."
        )
    for track in tracks:
        _validate_track_name(track)
        _ensure_track_absent(track)
        if not _track_dir_for_create(track).parent.is_dir():
            raise ValueError(f"Directory {os.path.dirname(track.replace('.', '/'))} does not exist")

    created_by = [
        f'gtrack.import_parquet(description, "{path}", {int(binsize)}, "{track_prefix}", "{c}", {float(defval):g})'
        for c in columns
    ]
    _import_parquet_tracks(tracks, columns, files, files, binsize, defval, description, created_by)
    return tracks


def _cleanup_empty_track_parents(track_dir: str | Path, db_root: str | None) -> None:
    """Remove the empty directories above a removed track up to its database's tracks/,
    and never the working directory or the tracks/ of another database kept below it (R's
    .cleanup_empty_dirs)."""
    tracks_root = Path(db_root) / "tracks" if db_root else None
    gwd = Path(_shared._GWD or "")
    cur = Path(track_dir).parent
    while (cur not in (tracks_root, gwd, cur.parent) and cur.exists()
           and not (cur.name == "tracks" and (cur.parent / "chrom_sizes.txt").exists())):
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def gtrack_mv(src: str, dest: str) -> None:
    """
    Rename or move a track within the same database.

    Renames a track or moves it to a different namespace (directory)
    within its source database. The track cannot be moved across
    databases; use `gtrack_copy` followed by `gtrack_rm` for that.

    Parameters
    ----------
    src : str
        Current track name.
    dest : str
        New track name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If source and destination are identical, the source track does
        not exist, or the destination track already exists.

    See Also
    --------
    gtrack_copy : Copy a track (possibly across databases).
    gtrack_rm : Delete a track.
    gtrack_exists : Test whether a track exists.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_mv("old_name", "new_name")
    """
    _checkroot()
    _validate_track_name(src)
    _validate_track_name(dest)
    if src == dest:
        raise ValueError("Source and destination track names are the same")
    if not _track_exists(src):
        raise ValueError(f"Track '{src}' does not exist")
    if _track_exists(dest):
        raise ValueError(f"Track '{dest}' already exists")

    src_dir = Path(_pymisha.pm_track_path(src))
    src_db = gtrack_dataset(src)
    # the destination is in the same database, by a name relative to where the source's is
    # (the working directory in a subdirectory)
    dest_dir = _track_dir(dest, None if _shared._gwd_is_subdir() else src_db)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        raise ValueError(f"Destination track directory already exists: {dest_dir}")

    try:
        src_dir.rename(dest_dir)
    except OSError:
        shutil.move(str(src_dir), str(dest_dir))

    # Both keys go stale: the source path no longer holds a track, and the
    # destination path now holds a different one than anything cached for it.
    _shared._invalidate_dir_cache(src_dir, dest_dir)

    _cleanup_empty_track_parents(src_dir, src_db)
    _pm_dbreload(src_db)


_TRACK_INTERNAL_FILES = frozenset(
    {"track.idx", "track.dat", ".attributes", "vars", ".meta"}
)


def _db_is_indexed_at(root: str | Path) -> bool:
    """R's .gdb.is_indexed_at: whether the database at root is in the indexed format. A
    loaded dataset with no sequence of its own (its seq/ link gone, or an empty folder)
    is in the loaded database's format."""
    real = os.path.realpath(str(root))
    groot = _shared._GROOT
    if groot is not None and real in {os.path.realpath(p) for p in [groot, *_shared._GDATASETS]}:
        first_chrom = str(_pymisha.pm_intervals_all()[1][0])
        if not _shared._has_own_sequence(real, first_chrom):
            root = groot
    return _db_is_indexed(str(root))


def _resolve_dest_db(db: str | None) -> Path:
    """Resolve a destination DB root for cross-database track copy.

    None means the database the working directory is in (UROOT for the working
    database's tracks/ when one is set). Otherwise the
    given path must either be one of the currently-loaded roots (GROOT
    or a loaded dataset) or look like a valid misha DB (chrom_sizes.txt
    and tracks/ both present).
    """
    if db is None:
        # in a subdirectory (gdir_cd), the database whose names the session lists
        if _shared._gwd_is_subdir() and _shared._gwd_db() is None:
            raise ValueError("The working directory is in no loaded database; give the destination with db.")
        root = _target_root()
        if not root:
            raise ValueError("Database not initialized. Call gdb_init() first.")
        return Path(root)
    resolved = Path(db).expanduser().resolve(strict=True)
    groot = _shared._GROOT
    gdatasets = list(_shared._GDATASETS or [])
    loaded = [str(Path(p).resolve()) for p in [groot] + gdatasets if p]
    if str(resolved) in loaded:
        return resolved
    if not (resolved / "chrom_sizes.txt").exists():
        raise ValueError(
            f"Destination db {resolved} does not contain a chrom_sizes.txt file."
        )
    if not (resolved / "tracks").is_dir():
        raise ValueError(
            f"Destination db {resolved} does not contain a tracks/ directory."
        )
    return resolved


def _dest_db_loaded(dest_db: Path) -> bool:
    groot = _shared._GROOT
    gdatasets = list(_shared._GDATASETS or [])
    loaded = [str(Path(p).resolve()) for p in [groot] + gdatasets if p]
    return str(dest_db.resolve()) in loaded


def _copy_track_dir(src_dir: Path, dest_dir: Path) -> None:
    """Copy the full source track directory into dest_dir.

    Tolerates a pre-existing empty dest_dir (used so cross-db conversion
    helpers can register the destination before writing into it).
    """
    if dest_dir.exists():
        for entry in src_dir.iterdir():
            target = dest_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
    else:
        shutil.copytree(src_dir, dest_dir)


def _match_chrom_alias(filename: str, dest_chroms: list[str]) -> str | None:
    """Return dest's canonical name for a per-chrom file, tolerating chr prefix variation."""
    if filename in dest_chroms:
        return filename
    if filename.startswith("chr"):
        stripped = filename[3:]
        if stripped in dest_chroms:
            return stripped
    else:
        prefixed = "chr" + filename
        if prefixed in dest_chroms:
            return prefixed
    return None


def _canonicalize_copied_chrom_files(
    src_dir: Path, dest_dir: Path, dest_chroms: list[str], destname: str
) -> None:
    """In a copy of a per-chromosome 1D track, the files the source is read from
    (GenomeTrack::ChromFiles) take their chromosomes' canonical names, and the files
    they hid go (R's .gtrack.copy.pipeline). A file whose own name the destination has,
    and its chromosome's name not (as "M" for chrMT into a database with chrM), keeps
    its name; of the files of a chromosome with the same bytes, one whose own name the
    destination has is taken where neither the used one's name nor its chromosome's is
    there."""
    files = _pymisha.pm_chrom_files(str(src_dir), False)
    rows = [list(r) for r in zip(files["chrom"], files["file"], files["used"], strict=True)]
    dest_set = set(dest_chroms)

    def in_dest(name: str) -> bool:
        return name in dest_set or (name[3:] if name.startswith("chr") else f"chr{name}") in dest_set

    others = [j for j, (_, f, used) in enumerate(rows) if not used and in_dest(f)]
    for i, (chrom, f, used) in enumerate(rows):
        if not (used and not in_dest(chrom) and not in_dest(f)):
            continue
        for j in others:
            if rows[j][0] == chrom and _same_bytes(src_dir / f, src_dir / rows[j][1]):
                rows[i][2], rows[j][2] = False, True
                break
    for _, f, used in rows:
        if not used:
            (dest_dir / f).unlink(missing_ok=True)
    for chrom, f, used in rows:
        if used and f != chrom and (in_dest(chrom) or not in_dest(f)):
            try:
                (dest_dir / f).rename(dest_dir / chrom)
            except OSError as exc:
                raise ValueError(
                    f"Failed to rename the chromosome files of {destname} in {dest_dir}"
                ) from exc


def _same_bytes(path1: Path, path2: Path) -> bool:
    return path1.stat().st_size == path2.stat().st_size and path1.read_bytes() == path2.read_bytes()


def _gtrack_copy_pipeline(
    src_dir: Path,
    dest_dir: Path,
    src_chroms: list[str],
    dest_chroms: list[str],
    src_indexed: bool,
    dest_indexed: bool,
    track_type: str,
    destname: str,
    dest_db: Path,
) -> None:
    """Pipeline: copy dir -> [decode if src indexed] -> [drop unmapped] -> [encode if dest indexed]."""
    same_order = src_chroms == dest_chroms

    if same_order and src_indexed == dest_indexed:
        _copy_track_dir(src_dir, dest_dir)
        return

    if track_type in ("rectangles", "points"):
        # _gtrack_copy_one already stopped for differing chrom orders and for the format
        # changes not supported (indexed into per-chromosome, or into a database that is
        # not the active one)
        _copy_track_dir(src_dir, dest_dir)
        if dest_indexed and not src_indexed:
            gtrack_2d_convert_to_indexed(destname, remove_old=True)
        return

    # 1D pipeline.
    _copy_track_dir(src_dir, dest_dir)

    if src_indexed:
        _pymisha.pm_track_split_indexed_to_per_chrom(
            str(dest_dir), list(src_chroms), True
        )
    else:
        _canonicalize_copied_chrom_files(src_dir, dest_dir, dest_chroms, destname)

    files_in_dir = [p.name for p in dest_dir.iterdir()]
    dest_with_variants = set(dest_chroms)
    for c in dest_chroms:
        if c.startswith("chr"):
            dest_with_variants.add(c[3:])
        else:
            dest_with_variants.add("chr" + c)
    candidates_for_drop = [f for f in files_in_dir if f not in _TRACK_INTERNAL_FILES]
    dropped = [f for f in candidates_for_drop if f not in dest_with_variants]
    if dropped:
        warnings.warn(
            f"gtrack_copy({destname!r}): dropped chromosomes not present in destination: "
            + ", ".join(dropped),
            stacklevel=3,
        )
        for f in dropped:
            (dest_dir / f).unlink()

    remaining = [
        p.name for p in dest_dir.iterdir() if p.name not in _TRACK_INTERNAL_FILES
    ]
    if candidates_for_drop and not remaining:
        _gdb_trash(dest_dir, async_unlink=True)
        raise ValueError(
            f"gtrack_copy({destname!r}): no chromosomes from source database are present in destination; "
            "refusing to create empty track."
        )

    # Canonicalize remaining per-chrom filenames to dest's preferred form.
    for f in list(remaining):
        canonical = _match_chrom_alias(f, dest_chroms)
        if canonical is not None and canonical != f:
            (dest_dir / f).rename(dest_dir / canonical)

    if dest_indexed:
        if track_type not in ("dense", "sparse", "array"):
            raise NotImplementedError(
                f"Cross-db copy of {track_type} track {destname!r} with format "
                "conversion to a non-active dataset is not yet supported."
            )
        _pymisha.pm_track_pack_per_chrom_to_indexed(
            str(dest_dir), list(dest_chroms), track_type
        )


def _gtrack_copy_one(
    srcname: str, destname: str, dest_info: dict[str, Any], overwrite: bool
) -> str:
    dest_db: Path = dest_info["db"]
    if not _track_exists(srcname):
        raise ValueError(f"Track '{srcname}' does not exist")

    src_db_raw = gtrack_dataset(srcname)
    src_db = Path(src_db_raw) if src_db_raw else Path(_shared._GROOT or "")

    src_dir = Path(_pymisha.pm_track_path(srcname))
    dest_dir = _track_dir(destname, str(dest_db))

    # the same name in the same database, or the same directory by two names (a linked
    # directory, another path to the database)
    if ((srcname == destname and src_db.resolve() == dest_db.resolve())
            or os.path.realpath(src_dir) == os.path.realpath(dest_dir)):
        raise ValueError(
            f"Source and destination are the same track: {srcname}"
        )

    if dest_dir.exists() and not overwrite:
        raise ValueError(
            f"Track {destname!r} already exists in {dest_db}; use overwrite=True to replace."
        )

    src_indexed = (src_dir / "track.idx").exists()
    dest_indexed = dest_info["indexed"] if src_indexed else dest_info["seq_indexed"]
    info = gtrack_info(srcname)
    track_type = info.get("type", "")
    is_2d = track_type in ("rectangles", "points")

    # An indexed track (track.idx) is keyed by chrom ids: reading one takes the source's
    # order, writing one the destination's, and a 2D track is copied in its own format,
    # so an indexed one needs both. Per-chromosome files are copied by name, by the
    # chrom_sizes.txt names, and need no order. Per-pair files are named by the names
    # gdb_init gives, which the two databases have to share, in any order; a destination
    # that cannot show them (no sequence) is compared by its chrom_sizes.txt names.
    # The destination's names are read once, when a track first needs them.
    if (is_2d or dest_indexed) and "chroms" not in dest_info:
        try:
            dest_info["chroms"] = _shared._gdb_chrom_names_at(str(dest_db))
        except Exception as exc:
            dest_info["chroms"] = exc
    order_known = (is_2d or dest_indexed) and not isinstance(dest_info.get("chroms"), Exception)
    if (src_indexed if is_2d else dest_indexed) and not order_known:
        raise dest_info["chroms"]
    if src_indexed or (is_2d and order_known):
        src_chroms = _shared._gdb_chrom_names_at(str(src_db))
    else:
        src_chroms = _shared._chrom_sizes_names_at(src_db)
    dest_chroms = dest_info["chroms"] if order_known else _shared._chrom_sizes_names_at(dest_db)

    # Every check that can stop the copy comes before the destination is touched
    # (overwrite=True sets the track there aside, and puts it back if the copy fails)
    def strip(names: list[str]) -> set[str]:
        return {n[3:] if n.startswith("chr") else n for n in names}

    if is_2d and src_indexed and src_chroms != dest_chroms:
        raise ValueError(
            f"Cross-db copy of 2D track {srcname!r} requires identical chromosome order in source and destination."
        )
    # (a loaded dataset's recorded order has its chrom_sizes.txt names, which may lack "chr")
    if is_2d and not src_indexed and strip(src_chroms) != strip(dest_chroms):
        raise ValueError(
            f"Cross-db copy of 2D track {srcname!r} requires the same chromosome names in source and destination."
        )
    if is_2d and src_indexed and not dest_indexed:
        raise NotImplementedError(
            f"Cross-db copy of indexed 2D track {destname!r} into a per-chromosome database is not yet supported."
        )
    if is_2d and dest_indexed and not src_indexed and dest_db.resolve() != Path(_shared._GROOT or "").resolve():
        raise NotImplementedError(
            f"Cross-db copy of 2D track {destname!r} with format conversion "
            "to a non-active dataset is not yet supported."
        )
    if _shared._gwd_is_subdir():
        gwd_db = _shared._gwd_db()
        if _shared._same_path(gwd_db, dest_db) and not _shared._same_path(_shared._path_db(dest_dir), dest_db):
            # a name that leads into another database kept below the working directory
            raise ValueError(f"Track {destname} would be written inside another database under {_shared._GWD}.")
        if is_2d and dest_indexed and not src_indexed and not _shared._same_path(gwd_db, dest_db):
            # converted by its name, which the session lists only under the working directory
            raise NotImplementedError(
                f"Copy of 2D track {destname} with format conversion into {dest_db} is not supported "
                f"from a directory of another database; gdir_cd() into {dest_db / 'tracks'} first."
            )
    if not is_2d:
        # the per-chromosome files the copy would have (an indexed track splits into one
        # per contig), by their chromosomes' names or their own
        if src_indexed:
            src_files = list(src_chroms)
        else:
            listed = _pymisha.pm_chrom_files(str(src_dir), False)
            src_files = list(listed["chrom"]) + list(listed["file"])
            # a computed track's pair files, or none named by a chromosome: its files
            if track_type == "computed" or not src_files:
                src_files = [f for f in os.listdir(src_dir) if f not in _TRACK_INTERNAL_FILES]
        dest_set = set(dest_chroms)
        if src_files and not any(
            f in dest_set or (f[3:] if f.startswith("chr") else f"chr{f}") in dest_set for f in src_files
        ):
            raise ValueError(
                f"gtrack_copy({destname!r}): no chromosomes from source database are present in destination; "
                "refusing to create empty track."
            )

    dest_db_loaded = _dest_db_loaded(dest_db)
    with _shared._with_umask():
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

    # The existing track is set aside (hidden, as a staging directory, swept as one if
    # left) until the copy has succeeded, and comes back if any step of it fails
    aside: Path | None = None
    if dest_dir.exists():
        aside = dest_dir.parent / f".{dest_dir.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}"
        try:
            os.rename(dest_dir, aside)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to set the existing destination {dest_dir} aside; copy aborted."
            ) from exc
        # dated now: a sweep of staging directories goes by their age
        with contextlib.suppress(OSError):
            os.utime(aside)
        _shared._invalidate_dir_cache(dest_dir)

    try:
        _gtrack_copy_pipeline(
            src_dir,
            dest_dir,
            src_chroms,
            dest_chroms,
            src_indexed,
            dest_indexed,
            track_type,
            destname,
            dest_db,
        )
    except BaseException:
        # the failed copy is moved away first, then the existing track comes back
        if dest_dir.exists():
            _gdb_trash(dest_dir, async_unlink=True)
        if aside is not None:
            try:
                os.rename(aside, dest_dir)
            except OSError:
                warnings.warn(
                    f"The copy over {dest_dir} failed; the track that was there is kept at {aside}",
                    stacklevel=user_stacklevel(),
                )
        _shared._invalidate_dir_cache(dest_dir)
        if dest_db_loaded:
            _pm_dbreload(str(dest_db))
        raise

    if aside is not None:
        _gdb_trash(aside, async_unlink=True)

    # The destination path now holds a different track than anything cached for
    # it (the copy pipeline rewrites track.idx in place at :_gtrack_copy_pipeline).
    _shared._invalidate_dir_cache(dest_dir)

    if dest_db_loaded:
        # dest_db can be the primary GROOT or a loaded secondary dataset;
        # pass it explicitly so the right one's caches get refreshed.
        _pm_dbreload(str(dest_db))
    else:
        # dest_db isn't loaded at all in this session, so _pm_dbreload()
        # (which rescans/clears caches for the *current* session) would be
        # both pointless and wrong here - just mark the sentinel directly.
        _shared._touch_db_cache_dirty(str(dest_db))

    return destname


@_shared._with_umask()   # database writes carry misha's permissions
def gtrack_copy(
    src: str | Iterable[str],
    dest: str | None = None,
    db: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    """
    Copy one or more tracks, optionally to a different database.

    Transparently handles format mismatches (per-chromosome vs indexed)
    and chromosome-order differences between source and destination
    databases. Chromosomes that exist in the source database but not in
    the destination are dropped with a warning.

    For 2D tracks (rectangles, points), cross-database copy requires
    identical chromosome order in source and destination.

    Parameters
    ----------
    src : str or iterable of str
        Source track name(s). Either a single name or an iterable of names.
    dest : str, optional
        Destination name. For a single source, this is the destination
        track name (defaults to *src*). For a sequence of sources, this
        is interpreted as a namespace prefix (e.g. ``"ns"`` produces
        ``"ns.track1"``, ``"ns.track2"``, ...). ``None`` keeps each
        track's original name.
    db : str, optional
        Destination database root. Must be the active GROOT, a member of
        the loaded datasets, or a path that looks like a valid misha
        database (contains ``chrom_sizes.txt`` and ``tracks/``). ``None``
        means the current writable database.
    overwrite : bool, default False
        If True, replace an existing destination track. Every check runs
        first, and the existing track is kept until the copy has succeeded.

    Returns
    -------
    list[str]
        The created destination track names.

    Raises
    ------
    ValueError
        Source track does not exist; destination already exists with
        ``overwrite=False``; 2D track copy with mismatched chromosome
        order; or destination database invalid.

    See Also
    --------
    gtrack_mv : Rename / move a track within the same database.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_copy("dense_track", "dense_track_copy")  # doctest: +SKIP
    >>> pm.gtrack_exists("dense_track_copy")  # doctest: +SKIP
    True
    >>> pm.gtrack_rm("dense_track_copy", force=True)  # doctest: +SKIP
    """
    _checkroot()

    if isinstance(src, str):
        srcnames = [src]
        single = True
    else:
        srcnames = list(src)
        single = False

    if not srcnames:
        return []

    for s in srcnames:
        _validate_track_name(s)

    if dest is None:
        destnames = list(srcnames)
    elif single:
        if not isinstance(dest, str):
            raise ValueError(
                "When copying a single track, 'dest' must be a single name or None."
            )
        destnames = [dest]
    else:
        if not isinstance(dest, str):
            raise ValueError(
                "When copying multiple tracks, 'dest' must be a single namespace prefix or None."
            )
        prefix = dest.rstrip(".")
        destnames = [f"{prefix}.{s}" for s in srcnames]

    for d in destnames:
        _validate_track_name(d)

    dest_db = _resolve_dest_db(db)
    tracks_dir = dest_db / "tracks"
    if not tracks_dir.exists():
        tracks_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(str(tracks_dir), os.W_OK):
        raise PermissionError(f"No write permission to copy track to {tracks_dir}")

    # The destination, once for all tracks: its format (an indexed track is copied in it,
    # any other track by its own seq/), and its chrom order, read when a track first needs
    # it. A loaded dataset that numbers the chromosomes differently from the working
    # database is read with the session's chrom ids but its own as a working database:
    # it gets files named by chromosome, which read right in both.
    dest_info: dict[str, Any] = {
        "db": dest_db,
        "indexed": _db_is_indexed_at(dest_db),
        "seq_indexed": _db_is_indexed(str(dest_db)),
    }
    if os.path.realpath(dest_db) in _shared._GDATASET_CHROMS:
        dest_info["indexed"] = dest_info["seq_indexed"] = False

    created: list[str] = []
    for s, d in zip(srcnames, destnames, strict=True):
        created.append(_gtrack_copy_one(s, d, dest_info, overwrite))
    return created


def gtrack_rm(track: str, force: bool = False, db: str | None = None) -> None:
    """
    Remove a track from disk.

    Permanently deletes the track directory and all associated files
    (per-chromosome data, attributes, variables). Empty parent
    directories are cleaned up automatically.

    Parameters
    ----------
    track : str
        Name of the track to remove.
    force : bool, default False
        If True, suppress errors when the track does not exist and
        allow deletion without confirmation. If False, raises
        ``ValueError`` when the track is missing.
    db : str or None, optional
        Explicit database root path. If None, the track is located in
        the currently initialized databases.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist (when *force* is False) or if
        *force* is False (safety guard).

    See Also
    --------
    gtrack_ls : List available tracks.
    gtrack_exists : Test whether a track exists.
    gtrack_mv : Rename or move a track.
    gtrack_copy : Copy a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_rm("my_track", force=True)
    """
    _checkroot()
    _validate_track_name(track)

    db_root: str | None
    if db is None:
        path = _pymisha.pm_track_path(track)
        if not path:
            if force:
                return
            raise ValueError(f"Track '{track}' does not exist")
        db_root = gtrack_dataset(track)
        track_dir = Path(path)
    else:
        db_root = str(Path(db))
        # after gdir_cd, db = the database the working directory is in means the names there
        track_dir = _track_dir(track, db_root)
        if not track_dir.exists():
            if force:
                return
            raise ValueError(f"Track '{track}' does not exist in database '{db_root}'")

    if not force:
        raise ValueError("Set force=True to delete a track")

    if not _gdb_trash(track_dir, async_unlink=True):
        raise RuntimeError(
            f"failed to remove track directory: {track_dir}"
        )
    _cleanup_empty_track_parents(track_dir, db_root)
    _pm_dbreload(db_root)


_DEFAULT_MAPPEDSEQ_COLS = (9, 11, 13, 14)


def _r_num(x: float) -> str:
    """A number as R's as.character() writes it: fixed or scientific, whichever is shorter."""
    x = float(x)
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    fixed = f"{x:.15g}"
    if "e" in fixed:
        return fixed
    mant, exp = f"{x:.14e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    sci = f"{mant}e{int(exp):+03d}"
    return sci if len(sci) < len(fixed) else fixed


def gtrack_import_mappedseq(
    track: str,
    description: str,
    file: str,
    pileup: int = 0,
    binsize: int = -1,
    cols_order: tuple[int, int, int, int] | None = _DEFAULT_MAPPEDSEQ_COLS,
    remove_dups: bool = True,
    paired: bool = False,
    min_mapq: int = 0,
    max_fraglen: float = 2000,
    one_based: bool = False,
) -> dict[str, Any]:
    """
    Import mapped sequences from SAM, BAM or tab-delimited text into a track.

    Reads aligned sequence data and creates either a Sparse (per-read) or a
    Dense (pileup or paired-end fragment coverage) track. The file can be in
    SAM format, in a tab-delimited format where each line describes a single
    read, in gzipped variants of either (``.sam.gz``, ``.tsv.gz``), or in BAM
    format (detected from its content; requires ``samtools`` on ``PATH``).
    Without *paired*, any bgzipped file is read as SAM. *file* may also be a
    named pipe (FIFO) carrying SAM or text, plain or gzipped, but not BAM. A
    pipe is not inspected, so the BAM and bgzip rules above do not apply to
    it: pass ``cols_order=None`` for SAM. An import that brings in no read at
    all gives a warning.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    file : str
        Path to a SAM, BAM, tab-delimited or fragment file, or a named pipe.
    pileup : int, default 0
        If 0, create a Sparse track with one interval per mapped read.
        If positive, create a Dense pileup track where each bin stores
        the mean number of reads covering its bases. Each read is extended
        to this length from its 5' end, downstream.
    binsize : int, default -1
        Bin size for Dense tracks. Required when *pileup* > 0 or *paired*.
        Must be -1 when *pileup* is 0 and *paired* is False.
    cols_order : tuple of int or None, default (9, 11, 13, 14)
        Column indices (1-based) for sequence, chromosome, coordinate,
        and strand in a tab-delimited file. Set to None for SAM. For BAM
        input the default is treated as SAM; any other value is an error.
    remove_dups : bool, default True
        If True, duplicated coordinates (fragments with the same start and
        end, with *paired*) are counted once.
    paired : bool, default False
        Import paired-end fragments (see Notes).
    min_mapq : int, default 0
        Minimal MAPQ of an imported SAM / BAM record.
    max_fraglen : int or float, default 2000
        Maximal fragment length when *paired* is True (``math.inf`` for no
        limit).
    one_based : bool, default False
        If True the coordinates of a tab-delimited file are 1-based (see
        Notes). Not used for SAM / BAM, which are always 1-based.

    Returns
    -------
    dict
        Dictionary with keys ``"total"`` (dict with ``"total"``,
        ``"total.mapped"``, ``"total.unmapped"``, ``"total.dups"``,
        ``"total.filtered"``) and ``"chrom"`` (pandas.DataFrame with
        per-chromosome mapping stats).

    Raises
    ------
    ValueError
        If the track already exists, *file* is None, column indices are
        invalid, or the arguments do not fit together.

    See Also
    --------
    gtrack_import : Import from WIG/BedGraph/BED/BigWig files.
    gtrack_create_sparse : Create a Sparse track from intervals.
    gtrack_create_dense : Create a Dense track from intervals.
    gtrack_rm : Delete a track.

    Notes
    -----
    Coordinates. SAM / BAM ``POS`` is 1-based and is converted to the 0-based
    leftmost aligned base. The 5' end of a reverse-strand read is its
    rightmost aligned base, from the CIGAR. A mapped SAM record without a
    CIGAR (``*``), or with a malformed one, is counted as unmapped, as htslib
    does. In a tab-delimited file the coordinate is the leftmost base of the
    read, and the 5' end of a reverse read is found from the length of the
    sequence. With ``one_based=True`` the coordinate is read as 1-based and
    reads are placed as for SAM; Illumina export files (the default
    *cols_order* layout) are 1-based. With the default ``one_based=False`` a
    tab-delimited file keeps the placement of earlier versions: the
    coordinate is used as is, which places dense coverage correctly for a
    0-based file, but in a sparse track a reverse read is recorded one base
    past its 5' end. Valid strand values are ``+`` or ``F`` for the forward
    strand and ``-`` or ``R`` for the reverse strand.

    For SAM and BAM input, unmapped, secondary, QC-failed and supplementary
    records (FLAG 0x4, 0x100, 0x200, 0x800) are never imported, and
    *min_mapq* drops records with a lower MAPQ.

    If *paired* is True each fragment covers its own span instead of being
    expanded by *pileup* (which must be 0), and the track is dense with bins
    of *binsize*. Each bin holds the mean number of fragments covering its
    bases (the last, partial bin of a chromosome is also divided by
    *binsize*). For SAM / BAM input (``cols_order=None``, or any BAM) the
    fragment is read from each proper pair (FLAG 0x1 and 0x2 set, 0x8 unset)
    by its first mate, as [min(POS, PNEXT), min(POS, PNEXT) + abs(TLEN)), the
    way MACS3 does with ``-f BAMPE``: records of second mates (FLAG 0x80) are
    skipped. The input may be sorted in any order. *min_mapq* is tested on
    the first mate only. With *remove_dups* fragments with the same start and
    end are counted once. Any other input (not a BAM, and *cols_order* not
    None) is read as a fragment file: tab-delimited, 0-based half-open
    chromosome, start and end in columns 1-3 (BED, or a 10x
    ``fragments.tsv.gz``), lines starting with ``#`` skipped. Fragment files
    are taken as already deduplicated, so *remove_dups* does not apply to
    them, and *cols_order* must not be set. Fragments longer than
    *max_fraglen* are skipped in both cases. With *paired* the records of
    second mates are not counted in the returned statistics.

    The statistics report ``total`` records, ``total.mapped`` (imported,
    duplicates included), ``total.unmapped`` (unmapped, unparsable, with an
    unusable CIGAR, or with its 5' end outside the chromosomes of the
    database), ``total.dups`` and ``total.filtered`` (left out by the flag,
    MAPQ, proper-pair or fragment length filters).

    R's ``missing(cols.order)`` is read here as *cols_order* equal to its
    default: passing ``(9, 11, 13, 14)`` explicitly counts as not passing it.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_import_mappedseq("reads", "Test", "reads.sam", cols_order=None)
    >>> # pm.gtrack_import_mappedseq("frags", "ATAC", "fragments.tsv.gz", binsize=20, paired=True)
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)
    if file is None:
        raise ValueError("file cannot be None")

    pileup_i = int(pileup)
    binsize_i = int(binsize)
    paired = bool(paired)
    if pileup_i < 0:
        raise ValueError("pileup cannot be negative")
    if not paired:
        if pileup_i == 0 and binsize_i >= 0:
            raise ValueError("For pileup=0 (sparse), binsize must be -1")
        if pileup_i > 0 and binsize_i <= 0:
            raise ValueError("For pileup>0 (dense), binsize must be > 0")

    cols_given = cols_order is not None and tuple(cols_order) != _DEFAULT_MAPPEDSEQ_COLS
    if cols_order is None:
        cols_arg: Any = None
    else:
        if len(cols_order) != 4:
            raise ValueError("cols_order must have 4 entries: sequence, chromosome, coordinate, strand")
        cols_tuple = tuple(int(x) for x in cols_order)
        if min(cols_tuple) <= 0:
            raise ValueError("cols_order indices are 1-based and must be positive")
        if len(set(cols_tuple)) != 4:
            raise ValueError("cols_order entries must be unique")
        cols_arg = cols_tuple

    path = str(file)
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    # Any bgzip file is read as SAM (BAM, or a bgzipped SAM) - except with paired, where a
    # bgzipped text file is a fragment file (e.g. 10x fragments.tsv.gz).
    sam_input = _is_bam_file(path, payload=paired)

    # paired fragment files have a fixed layout (chrom, start, end in columns 1-3)
    if paired and cols_given and not sam_input:
        raise ValueError(
            "cols_order is not used for fragment files: columns 1-3 must be chrom, start and end. "
            "Pass cols_order=None for a SAM file."
        )

    # BAM: samtools view emits SAM, so the default cols_order is switched to None and any
    # other one is an error.
    if sam_input:
        if cols_given:
            raise ValueError(
                "BAM input forces SAM column layout (without paired, any bgzipped file is read as SAM). "
                "Pass cols_order=None or omit the argument; for a bgzipped fragment file pass paired=True."
            )
        cols_arg = None

    max_fraglen_f = float(max_fraglen)
    if math.isnan(max_fraglen_f) or (not math.isinf(max_fraglen_f) and max_fraglen_f != int(max_fraglen_f)):
        raise ValueError("max_fraglen argument is not an integer")
    max_fraglen_i = 2**62 if max_fraglen_f == math.inf else int(max_fraglen_f)
    if int(min_mapq) != min_mapq:
        raise ValueError("min_mapq argument is not an integer")

    created_by = (
        f'gtrack.import_mappedseq("{track}", description, "{path}", '
        f"pileup={pileup_i}, binsize={binsize_i}, remove.dups={bool(remove_dups)}"
    )
    if paired or min_mapq > 0:
        created_by += (
            f", paired={'TRUE' if paired else 'FALSE'}, min.mapq={_r_num(min_mapq)}, "
            f"max.fraglen={_r_num(max_fraglen_f)}"
        )
    if one_based:
        created_by += ", one.based=TRUE"
    created_by += ")"

    dense = pileup_i > 0 or paired
    with _atomic_track_create(track) as tmp_dir:
        res = _pymisha.pm_import_mappedseq(
            str(tmp_dir), path,
            pileup_i, binsize_i, cols_arg, bool(remove_dups),
            paired, int(min_mapq), max_fraglen_i, bool(one_based),
        )
        if dense:
            _write_created_attrs_at_path(
                tmp_dir, description, created_by,
                {"type": "dense", "binsize": str(binsize_i)},
            )
        else:
            _write_created_attrs_at_path(tmp_dir, description, created_by)

    _pm_dbreload(_target_root())

    if res["total"]["total.mapped"] == 0:
        if not paired:
            msg = ("No read was imported. A SAM file needs cols_order=None, a fragment file paired=True, "
                   "and a tab-delimited file a matching cols_order.")
        elif cols_arg is None:
            msg = "No fragment was imported: no record was the first mate of a proper pair that passed the filters."
        else:
            msg = ("No fragment was imported. A SAM file needs cols_order=None; other files are read as "
                   "fragments (chrom, start, end).")
        warnings.warn(msg, stacklevel=2)

    cs = res["chrom_stats"]
    chrom_stat = pd.DataFrame({
        "chrom": list(cs["chrom"]),
        "mapped": np.asarray(cs["mapped"], dtype=float),
        "dups": np.asarray(cs["dups"], dtype=float),
    })
    return {"total": dict(res["total"]), "chrom": chrom_stat}


def gtrack_exists(track: str) -> bool:
    """
    Test for track existence in the Genomic Database.

    Parameters
    ----------
    track : str
        Track name to check.

    Returns
    -------
    bool
        True if the track exists, False otherwise.

    Raises
    ------
    ValueError
        If track is None.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_exists("dense_track")
    True
    >>> pm.gtrack_exists("nonexistent_track")
    False

    See Also
    --------
    gtrack_ls : List available tracks.
    gtrack_info : Get metadata for a track.
    gtrack_rm : Delete a track.
    """
    if track is None:
        raise ValueError("track cannot be None")

    _checkroot()

    if track == "":
        return False

    return _track_exists(track)


def gtrack_path(track: str) -> str:
    """
    Return the filesystem path of a track's directory.

    Parameters
    ----------
    track : str
        Track name (e.g. ``"dense_track"`` or ``"subdir.my_track"``).

    Returns
    -------
    str
        Absolute path to the track directory on disk.

    Raises
    ------
    ValueError
        If *track* is ``None`` or the track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_path("dense_track")  # doctest: +ELLIPSIS
    '...dense_track.track'

    See Also
    --------
    gtrack_exists : Check whether a track exists.
    gtrack_info : Get track metadata.
    gtrack_dataset : Get dataset root for a track.
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()
    path: str | None = _pymisha.pm_track_path(track)
    if path is None:
        raise ValueError(f"Track '{track}' does not exist")
    return path


def gtrack_attr_get(track: str, attr: str) -> str:
    """
    Get a single track attribute value.

    Parameters
    ----------
    track : str
        Track name.
    attr : str
        Attribute name.

    Returns
    -------
    str
        Attribute value, or empty string if attribute doesn't exist.

    Raises
    ------
    ValueError
        If track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_attr_get("sparse_track", "created.by")  # doctest: +SKIP
    '...'

    See Also
    --------
    gtrack_attr_set : Set a track attribute.
    gtrack_attr_export : Export attributes for multiple tracks.
    gtrack_attr_import : Batch-import attributes from a table.
    """
    if track is None:
        raise ValueError("track cannot be None")
    if attr is None:
        raise ValueError("attr cannot be None")

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    attrs = _load_track_attributes(track)
    return attrs.get(attr, "")


def gtrack_convert_to_indexed(track: str, remove_old: bool = False) -> None:
    """
    Convert a per-chromosome track to indexed format.

    Reads the per-chromosome binary files and writes a unified
    ``track.idx`` / ``track.dat`` pair. Optionally removes the original
    per-chromosome files after conversion.

    Parameters
    ----------
    track : str
        Name of the track to convert.
    remove_old : bool, default False
        If True, remove the original per-chromosome files after
        successful conversion.

    Returns
    -------
    None

    See Also
    --------
    gtrack_create_empty_indexed : Create empty indexed files.
    gdb_convert_to_indexed : Convert an entire database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_convert_to_indexed("my_track", remove_old=True)
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()

    # If this is a 2D track, dispatch to gtrack_2d_convert_to_indexed
    if _track_exists(track):
        info = gtrack_info(track)
        track_type = info.get("type")
        if track_type in ("rectangles", "points"):
            return gtrack_2d_convert_to_indexed(
                track, remove_old=remove_old, force=False
            )
        _check_convertible_track(track)
        if os.path.exists(os.path.join(_pymisha.pm_track_path(track), "track.idx")):
            return None

    _pymisha.pm_track_convert_to_indexed(track, bool(remove_old))
    # The layout under this path just changed; a cached "not indexed" is never
    # stored, but a cached index from a previous conversion would be.
    _shared._invalidate_dir_cache(_pymisha.pm_track_path(track))
    return None


def _index_created_track(track: str, remove_old: bool) -> None:
    """Write a track this session has just created, in an indexed working database, in the
    indexed format. A 1D one is written in whichever loaded database's tracks/ it was created
    in: R misha writes it indexed directly, in the session's chrom ids, without
    gtrack.convert_to_indexed's check of the database it is in. A 2D one is converted as
    gtrack_convert_to_indexed converts it."""
    if gtrack_info(track).get("type") in ("rectangles", "points"):
        gtrack_2d_convert_to_indexed(track, remove_old=remove_old, force=False)
        return
    track_dir = _pymisha.pm_track_path(track)
    if os.path.exists(os.path.join(track_dir, "track.idx")):
        return
    _pymisha.pm_track_convert_to_indexed(track, bool(remove_old))
    _shared._invalidate_dir_cache(track_dir)


def _check_convertible_track(track: str) -> None:
    """A track is written in the indexed format, keyed by the session's chrom ids, only in
    the working database (R's gtrack.convert_to_indexed): not in a dataset loaded with
    an order of its own, nor in another loaded dataset, nor, after gdir_cd, in a directory
    of none (the session may not know their chromosome order)."""
    track_db = gtrack_dataset(track)
    _shared._refuse_dataset_order("Track", track, track_db)
    groot = _shared._GROOT or ""
    if _shared._gwd_is_subdir():
        track_db = _shared._path_db(_pymisha.pm_track_path(track))
    elif track_db is None:
        track_db = groot
    if track_db is None or os.path.realpath(track_db) != os.path.realpath(groot):
        raise ValueError(
            f"Track {track} is not in the working database; convert it with gsetroot() on "
            "the database it is in."
        )


def gtrack_create_empty_indexed(track: str) -> None:
    """
    Create empty indexed files for an existing track directory.

    Writes an empty ``track.idx`` and ``track.dat`` pair in the track
    directory. Useful when the track has no data yet but indexed format
    is required by the database.

    Parameters
    ----------
    track : str
        Name of an existing track whose directory should receive the
        indexed files.

    Returns
    -------
    None

    See Also
    --------
    gtrack_convert_to_indexed : Convert per-chromosome files to indexed.
    gdb_convert_to_indexed : Convert an entire database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_create_empty_indexed("my_track")
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()
    _pymisha.pm_track_create_empty_indexed(track)
    _shared._invalidate_dir_cache(_pymisha.pm_track_path(track))


def gtrack_2d_convert_to_indexed(track: str, remove_old: bool = True, force: bool = False) -> None:
    """
    Convert a 2D track to indexed format (track.dat + track.idx).

    Consolidates per-chromosome-pair files into a single indexed format,
    reducing file descriptor usage from O(N^2) to O(1).

    Parameters
    ----------
    track : str
        Track name.
    remove_old : bool, default True
        If True, remove old per-pair files after conversion.
    force : bool, default False
        If True, re-convert even if already in indexed format.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist, is not a 2D track, or conversion
        fails.

    See Also
    --------
    gtrack_convert_to_indexed : Convert a 1D track to indexed format.
    gdb_convert_to_indexed : Convert an entire database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_2d_convert_to_indexed("my_2d_track")
    """
    from ._quadtree import clear_indexed_2d_cache

    if track is None:
        raise ValueError("track cannot be None")

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    info = gtrack_info(track)
    track_type = info.get("type")
    if track_type not in ("rectangles", "points"):
        raise ValueError(
            f"Track '{track}' is not a 2D track (type={track_type!r}). "
            f"Use gtrack_convert_to_indexed() for 1D tracks."
        )

    _shared._refuse_dataset_order("Track", track, gtrack_dataset(track))
    track_path = _pymisha.pm_track_path(track)
    idx_path = os.path.join(track_path, "track.idx")

    if os.path.exists(idx_path) and not force:
        return
    _check_convertible_track(track)

    # Clear any cached mmap handles for this track before conversion
    clear_indexed_2d_cache()

    track_type_int = 1 if track_type == "points" else 0
    _pymisha.pm_track2d_convert_to_indexed(track_path, track_type_int, bool(remove_old), bool(force))


def gtrack_attr_set(track: str, attr: str, value: str) -> None:
    """
    Set a track attribute value.

    Parameters
    ----------
    track : str
        Track name.
    attr : str
        Attribute name.
    value : str
        Attribute value. Set to empty string "" to remove the attribute.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If track does not exist or the attribute is read-only.

    See Also
    --------
    gtrack_attr_get : Read a single track attribute.
    gtrack_attr_export : Export attributes for multiple tracks.
    gtrack_attr_import : Batch-import attributes from a table.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_attr_set("sparse_track", "test_attr", "test_value")  # doctest: +SKIP
    >>> pm.gtrack_attr_get("sparse_track", "test_attr")  # doctest: +SKIP
    'test_value'
    >>> pm.gtrack_attr_set("sparse_track", "test_attr", "")  # doctest: +SKIP
    """
    if track is None:
        raise ValueError("track cannot be None")
    if attr is None:
        raise ValueError("attr cannot be None")
    if value is None:
        raise ValueError("value cannot be None")

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    from .db_attrs import gdb_get_readonly_attrs
    readonly_attrs = set(gdb_get_readonly_attrs() or [])
    if attr in readonly_attrs:
        raise ValueError(f"Attribute '{attr}' is read-only")

    # Load existing attributes
    attrs = _load_track_attributes(track)

    # Set or remove attribute
    if value == "":
        if attr in attrs:
            del attrs[attr]
    else:
        attrs[attr] = str(value)

    # Save back
    _save_track_attributes(track, attrs)


def gtrack_attr_import(table: pd.DataFrame, remove_others: bool = False) -> None:
    """
    Bulk import track attributes from a DataFrame.

    Parameters
    ----------
    table : DataFrame
        DataFrame with track names as index and attribute names as columns.
        Values are converted to strings. Empty string values are skipped
        (attribute not set for that track).
    remove_others : bool, default False
        If True, remove all non-readonly attributes not present in the table
        for tracks listed in the table.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If table is empty, any track in the index does not exist, or
        any attribute is read-only.

    See Also
    --------
    gtrack_attr_export : Export attributes to a DataFrame.
    gtrack_attr_get : Read a single attribute.
    gtrack_attr_set : Set a single attribute.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> tbl = pd.DataFrame({"description": ["test"]}, index=["dense_track"])
    >>> pm.gtrack_attr_import(tbl)  # doctest: +SKIP
    """
    import pandas as pd

    _checkroot()

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Invalid format of attributes table")

    tracks = list(table.index)
    attrs = list(table.columns)

    if not tracks or not attrs:
        raise ValueError("Invalid format of attributes table")

    seen_tracks = set()
    for track in tracks:
        if track in seen_tracks:
            raise ValueError(f"Track '{track}' appears more than once")
        seen_tracks.add(track)

    seen_attrs = set()
    for attr in attrs:
        if attr in seen_attrs:
            raise ValueError(f"Attribute '{attr}' appears more than once")
        seen_attrs.add(attr)

    if any((not isinstance(attr, str)) or attr == "" for attr in attrs):
        raise ValueError("Invalid format of attributes table")

    # Validate all tracks exist
    for track in tracks:
        if not _track_exists(track):
            raise ValueError(f"Track '{track}' does not exist")

    from .db_attrs import gdb_get_readonly_attrs
    readonly_attrs = set(gdb_get_readonly_attrs() or [])
    for attr in attrs:
        if attr in readonly_attrs:
            raise ValueError(f"Attribute '{attr}' is read-only")

    # Convert all values to strings
    table = table.astype(str)

    for track in tracks:
        existing_attrs = _load_track_attributes(track)

        if remove_others:
            # Remove attrs not in table columns, but keep readonly attributes.
            new_attrs = {k: v for k, v in existing_attrs.items() if k in readonly_attrs}
        else:
            new_attrs = dict(existing_attrs)

        for attr in attrs:
            val = table.at[track, attr]
            if val != "" and val != "nan":
                new_attrs[attr] = val

        _save_track_attributes(track, new_attrs)


def gtrack_attr_export(tracks: list[str] | None = None, attrs: list[str] | None = None) -> pd.DataFrame:
    """
    Export track attributes as a DataFrame.

    Parameters
    ----------
    tracks : list of str, optional
        List of track names. If None, all tracks.
    attrs : list of str, optional
        List of attribute names to include. If None, all attributes.

    Returns
    -------
    DataFrame
        DataFrame with tracks as rows and attributes as columns.

    Raises
    ------
    ValueError
        If any specified track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_attr_export()  # doctest: +SKIP
    >>> pm.gtrack_attr_export(tracks=["sparse_track", "dense_track"])  # doctest: +SKIP
    >>> pm.gtrack_attr_export(attrs=["created.by"])  # doctest: +SKIP

    See Also
    --------
    gtrack_attr_import : Batch-import attributes from a DataFrame.
    gtrack_attr_get : Read a single attribute.
    gtrack_attr_set : Set a single attribute.
    """
    import pandas as pd

    _checkroot()

    # Get list of tracks
    if tracks is None:
        tracks = gtrack_ls()
        if tracks is None:
            tracks = []
    else:
        # Validate tracks exist
        for track in tracks:
            if not _track_exists(track):
                raise ValueError(f"Track '{track}' does not exist")

    # Collect all attributes
    all_attrs: dict[str, dict[str, str]] = {}  # track -> {attr: value}
    all_attr_names: set[str] = set()

    for track in tracks:
        track_attrs = _load_track_attributes(track)
        all_attrs[track] = track_attrs
        all_attr_names.update(track_attrs.keys())

    # Filter attributes if specified
    if attrs is not None:
        all_attr_names = set(attrs)

    # Sort attribute names by popularity (number of tracks having this attr)
    attr_counts = {}
    for attr_name in all_attr_names:
        count = sum(1 for t in tracks if attr_name in all_attrs.get(t, {}))
        attr_counts[attr_name] = count

    sorted_attrs = sorted(all_attr_names, key=lambda a: (-attr_counts[a], a))

    # Build DataFrame
    data = {}
    for attr_name in sorted_attrs:
        data[attr_name] = [all_attrs.get(t, {}).get(attr_name, "") for t in tracks]

    return pd.DataFrame(data, index=tracks)


# ---------------------------------------------------------------------------
#  Track variables  (gtrack.var.* in R)
# ---------------------------------------------------------------------------

def _track_var_dir(track_name: str) -> str:
    """Return the path to the vars/ directory for a track, creating it if needed."""
    track_path = _pymisha.pm_track_path(track_name)
    if not track_path:
        raise ValueError(f"Track '{track_name}' does not exist")
    return os.path.join(track_path, "vars")


def gtrack_var_ls(track: str, pattern: str = "") -> list[str]:
    """
    List track variables.

    Parameters
    ----------
    track : str
        Track name.
    pattern : str, optional
        Regex pattern to filter variable names. Default ``""`` matches all.

    Returns
    -------
    list of str
        Sorted list of variable names matching the pattern.

    Raises
    ------
    ValueError
        If track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_var_ls("dense_track")
    []

    See Also
    --------
    gtrack_var_get : Read a variable's value.
    gtrack_var_set : Store a variable.
    gtrack_var_rm : Delete a variable.
    """
    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    var_dir = _track_var_dir(track)
    if not os.path.isdir(var_dir):
        return []

    files = os.listdir(var_dir)
    if not files:
        return []

    if pattern:
        files = [f for f in files if re.search(pattern, f)]

    return sorted(files)


def gtrack_var_get(track: str, var: str) -> Any:
    """
    Get the value of a track variable.

    Parameters
    ----------
    track : str
        Track name.
    var : str
        Variable name.

    Returns
    -------
    object
        The stored Python object.

    Raises
    ------
    ValueError
        If the track or variable does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_var_get("dense_track", "my_var")

    See Also
    --------
    gtrack_var_set : Store a variable.
    gtrack_var_ls : List variables for a track.
    gtrack_var_rm : Delete a variable.
    """
    import pickle

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    _validate_track_var_name(var)

    var_dir = _track_var_dir(track)
    filepath = os.path.join(var_dir, var)

    if not os.path.exists(filepath):
        raise ValueError(
            f"Variable '{var}' does not exist for track '{track}'"
        )

    with open(filepath, "rb") as f:
        header = f.read(2)
        f.seek(0)

        # Detect R serialization formats. R serialize v2 ASCII starts
        # with b'A\n', XDR binary with b'X\n', v3 with b'B\n', and
        # gzip-compressed RDS with the gzip magic number b'\x1f\x8b'.
        # PyMisha reads XDR-binary and gzipped XDR natively; the ASCII
        # variants are uncommon and not supported.
        if header in (b"X\n", b"\x1f\x8b"):
            from ._r_serialize import read as _r_read
            try:
                return _r_read(filepath)
            except NotImplementedError as exc:
                raise ValueError(
                    f"Track variable '{var}' on track '{track}' was written "
                    f"by R misha but contains an R-serialize feature pymisha "
                    f"does not decode yet: {exc}. Open an issue with a sample."
                ) from exc

        if header in (b"A\n", b"B\n"):
            raise ValueError(
                f"Track variable '{var}' on track '{track}' is in R's ASCII "
                f"serialize format, which pymisha does not read. Re-serialize "
                f"in R with `serialize(value, con, ascii=FALSE)` (or use "
                f"`saveRDS`) and try again."
            )

        try:
            return restricted_load(f)
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError):
            pass

    raise ValueError(
        f"Cannot read variable '{var}' for track '{track}': "
        f"unknown or unsafe format"
    )


@_shared._with_umask()   # database writes carry misha's permissions
def gtrack_var_set(track: str, var: str, value: Any) -> None:
    """
    Set the value of a track variable.

    Parameters
    ----------
    track : str
        Track name.
    var : str
        Variable name.
    value : object
        Value to store. Can be any pickle-able Python object.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist.

    See Also
    --------
    gtrack_var_get : Read a variable's value.
    gtrack_var_ls : List variables for a track.
    gtrack_var_rm : Delete a variable.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_var_set("dense_track", "my_var", [1, 2, 3])
    """
    import pickle

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    _validate_track_var_name(var)

    var_dir = _track_var_dir(track)
    os.makedirs(var_dir, exist_ok=True)

    filepath = os.path.join(var_dir, var)
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise TypeError("value is not serializable") from exc
    try:
        restricted_loads(payload)
    except pickle.UnpicklingError as exc:
        raise TypeError(
            "value contains unsupported objects for secure track-variable serialization"
        ) from exc
    with _shared._write_file(filepath) as f:
        f.write(payload)


def gtrack_var_rm(track: str, var: str) -> None:
    """
    Remove a track variable.

    Parameters
    ----------
    track : str
        Track name.
    var : str
        Variable name to remove.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist.

    See Also
    --------
    gtrack_var_set : Store a variable.
    gtrack_var_get : Read a variable's value.
    gtrack_var_ls : List variables for a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_var_rm("dense_track", "my_var")
    """
    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    _validate_track_var_name(var)

    var_dir = _track_var_dir(track)
    filepath = os.path.join(var_dir, var)

    if os.path.exists(filepath):
        os.remove(filepath)


# ---------------------------------------------------------------------------
# 2D track creation
# ---------------------------------------------------------------------------


def _normalize_2d_intervals(intervals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate and normalize a 2D intervals DataFrame.

    Returns a copy with columns: chrom1, start1, end1, chrom2, start2, end2,
    and ``_row``, the input row of each interval. Chromosome names are
    normalized via pm_normalize_chroms; intervals on a chromosome the database
    does not have are dropped.
    """
    if not isinstance(intervals, pd.DataFrame):
        raise ValueError("intervals must be a DataFrame")
    required = {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}
    if not required.issubset(intervals.columns):
        raise ValueError(f"intervals must contain columns: {', '.join(sorted(required))}")

    out = intervals[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
    out["_row"] = np.arange(len(out))
    out["chrom1"] = out["chrom1"].astype(str)
    out["chrom2"] = out["chrom2"].astype(str)
    for col in ("start1", "end1", "start2", "end2"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.int64)

    # Normalize chromosome names
    all_chroms = list(set(out["chrom1"].tolist() + out["chrom2"].tolist()))
    try:
        norm = _pymisha.pm_normalize_chroms(all_chroms)
        cmap = dict(zip(all_chroms, norm, strict=False))
    except _pymisha.error:
        # Falling back to the identity map can drop every row as "unknown
        # chromosome", so this one is worth a warning rather than a debug line.
        _logger.warning("chromosome normalization failed; using the names as given", exc_info=True)
        cmap = {c: c for c in all_chroms}
    out["chrom1"] = out["chrom1"].map(cmap)
    out["chrom2"] = out["chrom2"].map(cmap)

    # Filter to known chroms
    from .intervals import gintervals_all
    chrom_sizes_df = gintervals_all()
    known = set(chrom_sizes_df["chrom"].astype(str).tolist())
    mask = out["chrom1"].isin(known) & out["chrom2"].isin(known)
    out = out[mask].reset_index(drop=True)
    return out, chrom_sizes_df


def _chrom_ids_2d(intervals: pd.DataFrame, chrom_sizes_df: pd.DataFrame) -> tuple[Any, Any, list[str], list[int]]:
    """chrom1 / chrom2 of normalized 2D intervals as ids into the database's
    chromosomes, and those chromosomes and their sizes, in id order."""
    chroms = chrom_sizes_df["chrom"].astype(str).tolist()
    sizes = [int(e) for e in chrom_sizes_df["end"]]
    chrom2id = {c: i for i, c in enumerate(chroms)}
    cid1 = intervals["chrom1"].map(chrom2id).to_numpy(dtype=np.int32)
    cid2 = intervals["chrom2"].map(chrom2id).to_numpy(dtype=np.int32)
    return cid1, cid2, chroms, sizes


@_shared._with_umask()   # database writes carry misha's permissions
def gtrack_2d_create(track: str, description: str, intervals: pd.DataFrame, values: Any) -> None:
    """
    Create a 2D track from intervals and values.

    Parameters
    ----------
    track : str
        Track name (dot-separated namespace).
    description : str
        Track description.
    intervals : DataFrame
        2D intervals with columns: chrom1, start1, end1, chrom2, start2, end2.
    values : array-like
        Numeric values, one per interval.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, values length does not match
        intervals, no valid intervals remain after normalization, or
        overlapping intervals are detected within the same chromosome pair.

    See Also
    --------
    gtrack_2d_import : Create a 2D track from a file.
    gtrack_2d_import_contacts : Import HiC contact data as a 2D track.
    gtrack_create_sparse : Create a 1D Sparse track.
    gtrack_rm : Delete a track.

    Notes
    -----
    Creates a RECTS track, even when every interval is 1 x 1, as R misha
    does; :func:`gtrack_2d_import` and :func:`gtrack_2d_import_contacts`
    create POINTS tracks. The files are the bytes R writes for the same
    intervals and values. Overlapping intervals within the same chromosome
    pair raise an error.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> ivs = pd.DataFrame({
    ...     "chrom1": ["1"], "start1": [0], "end1": [100],
    ...     "chrom2": ["1"], "start2": [200], "end2": [300],
    ... })
    >>> pm.gtrack_2d_create("test_2d", "Test", ivs, [1.0])  # doctest: +SKIP
    >>> pm.gtrack_rm("test_2d", force=True)  # doctest: +SKIP
    """
    from ._quadtree import verify_no_overlaps_2d

    _checkroot()
    _validate_track_name(track)

    # R parity: validate coordinates before _normalize_2d_intervals's
    # to_numeric(...).astype(np.int64) cast, which silently turns a NaN
    # coordinate into a platform-dependent garbage int64 the same way the
    # pre-v0.9.1 1D path did -- and before a negative/inverted/past-boundary
    # rectangle is written straight into the track's binary quad-tree file.
    if not isinstance(intervals, pd.DataFrame):
        raise ValueError("intervals must be a DataFrame")
    from .intervals import _validate_2d_intervals, _verify_2d_intervals
    _validate_2d_intervals(intervals, "intervals")
    _verify_2d_intervals(intervals)

    values_arr = np.asarray(values, dtype=np.float64).ravel()
    if len(values_arr) != len(intervals):
        raise ValueError(
            f"Number of values ({len(values_arr)}) must match number of "
            f"intervals ({len(intervals)})"
        )
    intervals_df, chrom_sizes_df = _normalize_2d_intervals(intervals)
    if len(intervals_df) == 0:
        raise ValueError("No valid intervals after normalization")
    # R reads each value into a float, as the track stores it
    values_f32 = values_arr[intervals_df["_row"].to_numpy()].astype(np.float32)

    # Overlapping intervals are refused, pair by pair
    for _, group in intervals_df.groupby(["chrom1", "chrom2"], sort=False):
        verify_no_overlaps_2d(list(zip(group["start1"].to_numpy(dtype=int), group["start2"].to_numpy(dtype=int),
                                       group["end1"].to_numpy(dtype=int), group["end2"].to_numpy(dtype=int),
                                       strict=True)))

    cid1, cid2, chroms, sizes = _chrom_ids_2d(intervals_df, chrom_sizes_df)
    with _atomic_track_create(track) as track_dir:
        track_dir.mkdir(parents=True, exist_ok=True)
        # As R's gtrack.2d.create: a RECTS track whatever the size of the
        # intervals, the intervals inserted in R's order, written in R's chunks
        _pymisha.pm_track_2d_create(
            str(track_dir), chroms, sizes,
            cid1, intervals_df["start1"].to_numpy(dtype=np.int64), intervals_df["end1"].to_numpy(dtype=np.int64),
            cid2, intervals_df["start2"].to_numpy(dtype=np.int64), intervals_df["end2"].to_numpy(dtype=np.int64),
            values_f32,
        )

    try:
        _pm_dbreload(_target_root())
        _set_created_attrs(
            track,
            description,
            f'gtrack.2d.create("{track}", description, intervals, values)',
        )
        if _db_is_indexed(_shared._GROOT):
            gtrack_2d_convert_to_indexed(track, remove_old=True, force=False)
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=2,
        )
        raise


def _import_2d_limits() -> tuple[int, int]:
    """CONFIG['max_data_size'] and CONFIG['max_mem_usage'] (KB) as the 2D imports read them."""
    max_data_size = int(cast("int", _shared.CONFIG.get("max_data_size", 10000000)))
    max_mem_usage = _shared.CONFIG.get("max_mem_usage")
    return max_data_size, -1 if max_mem_usage is None else int(cast("int", max_mem_usage))


@_shared._with_umask()   # database writes carry misha's permissions
def gtrack_2d_import(track: str, description: str, file: str | list[str]) -> None:
    """
    Import a 2D track from one or more tab-delimited files.

    Parameters
    ----------
    track : str
        Track name.
    description : str
        Track description.
    file : str or list of str
        Path(s) to tab-delimited file(s) with header:
        chrom1, start1, end1, chrom2, start2, end2, <value_column>.
        When multiple files are given, all are read, in order.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, any file is not found, the file list
        is empty, or a file is malformed (the message names the file and
        line).

    See Also
    --------
    gtrack_2d_create : Create a 2D track from a DataFrame.
    gtrack_2d_import_contacts : Import HiC contacts as a 2D track.
    gtrack_rm : Delete a track.

    Notes
    -----
    The first 6 columns must be named chrom1, start1, end1, chrom2, start2
    and end2; the value is the 7th column. A POINTS track if every interval
    is 1 x 1, a RECTS track otherwise, as in R misha. Intervals on a
    chromosome the database does not have are skipped. Each chromosome pair
    is written as R writes it - its records in input order, a pair of at
    least 4 x ``CONFIG['max_data_size']`` records in parts (R's
    gmax.data.size) - so the files are the bytes R writes.

    If ``CONFIG['multitasking']`` is on (the default), the input files are read
    by several processes in parallel, each process reading a contiguous subset
    of the files, and the chromosome pairs are then written in parallel, one
    pair per process at a time. A single input file is read by one process, so
    split a large input into several files to read it in parallel. The number
    of processes is bounded by ``CONFIG['max_processes']``. The resulting track
    is identical to the one created with multitasking off. A pair is started
    only when its estimated memory, added to that of the pairs being written,
    stays within ``CONFIG['max_mem_usage']`` (KB). Larger pairs are taken
    first, but a smaller pair that fits may start while a larger one waits, and
    a pair estimated above ``CONFIG['max_mem_usage']`` is written alone.
    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_2d_import("test_2d", "Test", "contacts.tsv")
    >>> # pm.gtrack_2d_import("test_2d", "Test", ["a.tsv", "b.tsv"])
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    # Accept a single file path or a list of file paths
    files = [file] if isinstance(file, str) else list(file)

    if not files:
        raise ValueError(
            "At least one file must be provided. "
            "Usage: gtrack_2d_import(track, description, file)"
        )

    # Validate all files exist before reading any
    for f in files:
        if not os.path.exists(f):
            raise ValueError(f"File not found: {f}")

    max_data_size, max_mem_usage = _import_2d_limits()
    with _atomic_track_create(track) as track_dir:
        track_dir.mkdir(parents=True, exist_ok=True)
        try:
            _pymisha.pm_track_2d_import_files(str(track_dir), [str(f) for f in files], max_data_size, max_mem_usage)
        except _pymisha.error as exc:
            raise ValueError(str(exc)) from exc

    try:
        _pm_dbreload(_target_root())
        files_str = '", "'.join(files)
        _set_created_attrs(track, description, f'gtrack.2d.import("{track}", description, c("{files_str}"))')
        if _db_is_indexed(_shared._GROOT):
            gtrack_2d_convert_to_indexed(track, remove_old=True, force=False)
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=2,
        )
        raise


@_shared._with_umask()   # database writes carry misha's permissions
def gtrack_2d_import_contacts(
    track: str,
    description: str,
    contacts: str | list[str],
    fends: str | None = None,
    allow_duplicates: bool = True,
) -> None:
    """
    Create a 2D Points track from inter-genomic contacts.

    Parameters
    ----------
    track : str
        Track name (dot-separated namespace).
    description : str
        Track description.
    contacts : str or list of str
        Path(s) to contact files. If ``fends`` is None the files must be in
        "intervals-value" tab-separated format (header: chrom1, start1, end1,
        chrom2, start2, end2, then the value column). Otherwise they must be
        in "fends-value" format (columns: fend1, fend2, count).
    fends : str or None
        Path to a fragment-ends file with columns: fend, chr, coord.
    allow_duplicates : bool, default True
        If True, duplicate contacts (same midpoint pair) are summed.
        If False, duplicates raise ``ValueError``.

    Notes
    -----
    * Intervals are converted to midpoints: X = (start1+end1)//2,
      Y = (start2+end2)//2.
    * Contacts are canonically ordered: if chrom2 < chrom1 (or same chrom
      and coord2 < coord1) the two sides are swapped.
    * Cis contacts (same chromosome) are mirrored: both (X,Y) and (Y,X)
      are stored unless X == Y.
    * Trans contacts (different chromosomes) are written in both directions:
      a chrA-chrB file and a chrB-chrA file (with swapped coordinates) are
      created so that queries work regardless of chromosome pair order.
    * Contacts on a chromosome the database does not have, or on a fend the
      fends file does not define, are skipped.
    * Each chromosome pair is written as R misha writes it - the contacts
      summed and inserted in R's order, a pair of at least
      4 x ``CONFIG['max_data_size']`` records in parts (R's gmax.data.size;
      2 x for a chromosome with itself) - so the files are the bytes R
      writes for the same input.

    If ``CONFIG['multitasking']`` is on (the default), the input files are read
    by several processes in parallel, each process reading a contiguous subset
    of the files, and the chromosome pairs are then written in parallel, one
    pair per process at a time. A single input file is read by one process, so
    split a large input into several files to read it in parallel. The number
    of processes is bounded by ``CONFIG['max_processes']``. The resulting track
    is identical to the one created with multitasking off. A pair is started
    only when its estimated memory, added to that of the pairs being written,
    stays within ``CONFIG['max_mem_usage']`` (KB). Larger pairs are taken
    first, but a smaller pair that fits may start while a larger one waits, and
    a pair estimated above ``CONFIG['max_mem_usage']`` is written alone.
    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, no contact files are provided, a file is
        malformed, or duplicates are found when *allow_duplicates* is False.

    See Also
    --------
    gtrack_2d_create : Create a 2D track from a DataFrame.
    gtrack_2d_import : Import a 2D track from a tab-delimited file.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_2d_import_contacts("hic", "HiC", ["contacts.tsv"])
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    # Accept a single file path as a convenience
    if isinstance(contacts, str):
        contacts = [contacts]

    if not contacts:
        raise ValueError(
            "Usage: gtrack_2d_import_contacts(track, description, contacts, "
            "fends=None, allow_duplicates=True)"
        )

    if fends is not None and not os.path.exists(fends):
        raise ValueError(f"Fends file not found: {fends}")
    for cfile in contacts:
        if not os.path.exists(cfile):
            raise ValueError(f"Contacts file not found: {cfile}")

    max_data_size, max_mem_usage = _import_2d_limits()
    with _atomic_track_create(track) as track_dir:
        track_dir.mkdir(parents=True, exist_ok=True)
        try:
            _pymisha.pm_track_2d_import_contacts_files(
                str(track_dir), [str(f) for f in contacts], None if fends is None else str(fends),
                bool(allow_duplicates), max_data_size, max_mem_usage,
            )
        except _pymisha.error as exc:
            raise ValueError(str(exc)) from exc

    try:
        _pm_dbreload(_target_root())

        contacts_str = '", "'.join(contacts)
        fends_str = f'"{fends}"' if fends else "NULL"
        _set_created_attrs(
            track,
            description,
            f'gtrack.2d.import_contacts("{track}", description, '
            f'c("{contacts_str}"), {fends_str}, {allow_duplicates})',
        )
        if _db_is_indexed(_shared._GROOT):
            gtrack_2d_convert_to_indexed(track, remove_old=True, force=False)
    except Exception as exc:
        warnings.warn(
            f"post-create steps failed for track '{track}': {exc}; "
            "the track was created but may have incomplete attributes",
            stacklevel=2,
        )
        raise



def gtrack_2d_get_insu_doms(
    insu_track: str,
    thresh: float,
    iterator: int | float = 500,
) -> pd.DataFrame:
    """Extract TAD-style domains from a 1D insulation track.

    Domains are intervals where the insulation score is *missing* or
    *above* ``thresh`` (lax/inside-domain bins; mirrors R misha's
    ``gtrack.2d.get_insu_doms``).

    Parameters
    ----------
    insu_track : str
        Name of a 1D track of per-bin insulation values.
    thresh : float
        Threshold; bins with value > ``thresh`` (or ``NaN``) are kept.
    iterator : int or float, default 500
        Bin size passed to :func:`gscreen`.

    Returns
    -------
    pandas.DataFrame
        ``chrom``, ``start``, ``end`` of the domain intervals.

    See Also
    --------
    gtrack_2d_get_insu_borders : The complementary borders-of-domains
        extraction.
    gscreen : Underlying interval-screening engine.
    """
    from .extract import gscreen
    # In pymisha track expressions, R's `is.na(x)` is `np.isnan(x)` (see
    # tests/r_parity/test_db.py for the mapping convention).
    expr = f"np.isnan({insu_track}) | {insu_track} > {thresh}"
    return gscreen(expr, iterator=iterator)


def gtrack_2d_get_insu_borders(
    insu_track: str,
    thresh: float,
    iterator: int | float = 500,
) -> pd.DataFrame:
    """Extract TAD borders from a 1D insulation track.

    Borders are intervals where the insulation score is *present* and
    *below* ``thresh`` (strong-boundary bins; mirrors R misha's
    ``gtrack.2d.get_insu_borders``).

    Parameters
    ----------
    insu_track : str
        Name of a 1D track of per-bin insulation values.
    thresh : float
        Threshold; bins with value < ``thresh`` (and not ``NaN``) are kept.
    iterator : int or float, default 500
        Bin size passed to :func:`gscreen`.

    Returns
    -------
    pandas.DataFrame
        ``chrom``, ``start``, ``end`` of the border intervals.

    See Also
    --------
    gtrack_2d_get_insu_doms : The complementary domains extraction.
    gscreen : Underlying interval-screening engine.
    """
    from .extract import gscreen
    expr = f"~np.isnan({insu_track}) & {insu_track} < {thresh}"
    return gscreen(expr, iterator=iterator)


# ---------------------------------------------------------------------------
# Array tracks (R parity for gtrack.array.*)
# ---------------------------------------------------------------------------

def gtrack_array_create(
    track: str,
    description: str,
    intervals: pd.DataFrame,
    values: Any,
    colnames: list[str],
) -> None:
    """Create an ``array`` track from intervals + a per-position matrix.

    Lower-level analogue of R's ``gtrack.array.import`` for the
    in-memory case (R's import shells out to a C parser for files; here
    you supply intervals + values directly).

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description (stored as a track attribute).
    intervals : pandas.DataFrame
        ``chrom, start, end`` rows. Must be sorted within each chromosome
        and non-overlapping (R's array-track invariant).
    values : array_like of shape (n_intervals, n_cols)
        Float values. ``NaN`` is treated as "missing" and stored sparsely
        (matching the R format - NaN cells are not written).
    colnames : list of str
        Column names; ``len(colnames) == values.shape[1]``.

    Examples
    --------
    >>> import pymisha as pm
    >>> import numpy as np, pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> ivs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    >>> pm.gtrack_array_create("my_arr", "demo", ivs, np.array([[1.0, 2.0]]),
    ...                        ["a", "b"])  # doctest: +SKIP
    """
    from ._array_track import build_value_blocks, write_chrom_file, write_colnames

    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    if not isinstance(intervals, pd.DataFrame):
        raise ValueError("intervals must be a pandas DataFrame")
    for col in ("chrom", "start", "end"):
        if col not in intervals.columns:
            raise ValueError(f"intervals must have a '{col}' column")

    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 2:
        raise ValueError("values must be a 2-D array (n_intervals x n_cols)")
    if values_arr.shape[0] != len(intervals):
        raise ValueError(
            "values.shape[0] must match len(intervals); got "
            f"{values_arr.shape[0]} vs {len(intervals)}"
        )
    if values_arr.shape[1] != len(colnames):
        raise ValueError(
            "values.shape[1] must match len(colnames); got "
            f"{values_arr.shape[1]} vs {len(colnames)}"
        )
    if len(set(colnames)) != len(colnames):
        raise ValueError("colnames must be unique")

    ivs = _normalize_intervals_df(intervals).copy().reset_index(drop=True)
    ivs = _canonicalize_known_chroms(ivs)
    if len(ivs) == 0:
        raise ValueError("No intervals map to known chromosomes")
    if len(ivs) != len(intervals):
        raise ValueError(
            "intervals contains rows mapping to unknown chromosomes"
        )

    # Sort within chromosome.
    ivs["_row"] = np.arange(len(ivs))
    ivs_sorted = ivs.sort_values(["chrom", "start"], kind="mergesort").reset_index(drop=True)
    row_order = ivs_sorted["_row"].to_numpy()
    values_sorted = values_arr[row_order]

    with _atomic_track_create(track) as track_dir:
        track_dir.mkdir(parents=True, exist_ok=True)
        # vars subdir is part of the standard track layout
        (track_dir / "vars").mkdir(exist_ok=True)
        write_colnames(track_dir, list(colnames))
        for chrom, group in ivs_sorted.groupby("chrom", sort=False):
            chrom_path = track_dir / str(chrom)
            starts = group["start"].to_numpy(dtype=np.int64)
            ends = group["end"].to_numpy(dtype=np.int64)
            block_rows = values_sorted[group.index.to_numpy()]
            blocks = build_value_blocks(block_rows)
            write_chrom_file(chrom_path, starts, ends, blocks)

    # Reload the database so the new track is visible, then set attrs.
    _pymisha.pm_dbreload()
    _set_created_attrs(
        track,
        description,
        (
            f'gtrack_array_create("{track}", description, intervals, '
            f"values, colnames=[{', '.join(repr(n) for n in colnames)}])"
        ),
    )
    _pymisha.pm_dbreload()
    # gtrack_array_create predates _pm_dbreload() and still calls the raw
    # C++ rescan directly, so the sentinel has to be touched explicitly
    # here. It writes through _atomic_track_create(), i.e. _target_root()
    # (_UROOT if set, else _GROOT) - not necessarily _GROOT.
    _shared._touch_db_cache_dirty(_target_root())


def gtrack_array_import(
    track: str,
    description: str,
    *srcs: str,
) -> None:
    """Create an array track by merging one or more sources.

    R parity for ``gtrack.array.import``. Each ``src`` is either an
    existing array-track name or a path to a tab-separated file with
    header ``chrom\\tstart\\tend\\t<col1>\\t<col2>...`` (the format
    written by :func:`gtrack_array_extract` with ``file=``).

    Sources are merged interval-wise. If two sources share an identical
    ``(start, end)`` interval, their non-NaN cells are combined into a
    single output row; identical column names across sources are treated
    as a single output column (consistency-checked, error on mismatch).
    Partial overlaps across sources raise an error.

    Parameters
    ----------
    track : str
        Name of the array track to create.
    description : str
        Description string written to the track's ``.attributes`` file.
    *srcs : str
        One or more source paths (TSV files) or existing array-track names.

    Returns
    -------
    None

    See Also
    --------
    gtrack_array_extract : Read array-track values (and write to TSV via ``file=``).
    gtrack_array_create : Create an array track directly from a values matrix.
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)
    if not srcs:
        raise ValueError("gtrack_array_import: at least one source is required")

    from ._array_track import (
        _CSVSource,
        _import_sources,
        _TrackSource,
        write_colnames,
    )
    from .intervals import gintervals_all

    # Resolve each src to a CSV or track source. R: not-a-track -> CSV.
    sources: list = []
    for src in srcs:
        if _track_exists(src):
            sources.append(_TrackSource(src))
        else:
            sources.append(_CSVSource(src))

    chrom_order = [str(c) for c in gintervals_all()["chrom"].tolist()]

    # Atomic create: directory layout mirrors gtrack_array_create -- per-chrom
    # binary files at <track_dir>/<chrom>, <track_dir>/.colnames, and an empty
    # <track_dir>/vars/ subdir. No separate "signature" file; the array-track
    # reader detects the layout.
    with _atomic_track_create(track) as track_dir:
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "vars").mkdir(exist_ok=True)
        colnames = _import_sources(track_dir, sources, chrom_order)
        write_colnames(track_dir, colnames)

    _pymisha.pm_dbreload()
    created_by = (
        f'gtrack_array_import("{track}", description, src = c("'
        + '", "'.join(srcs)
        + '"))'
    )
    _set_created_attrs(track, description, created_by)
    _pymisha.pm_dbreload()
    # gtrack_array_import predates _pm_dbreload() and still calls the raw
    # C++ rescan directly, so the sentinel has to be touched explicitly
    # here. It writes through _atomic_track_create(), i.e. _target_root()
    # (_UROOT if set, else _GROOT) - not necessarily _GROOT.
    _shared._touch_db_cache_dirty(_target_root())


def gtrack_array_get_colnames(track: str) -> list[str]:
    """Return the column names of an ``array`` track.

    R parity for ``gtrack.array.get_colnames``. The names live in
    ``<track_dir>/.colnames`` as an R-serialized named integer vector.
    """
    _checkroot()
    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    info = gtrack_info(track)
    if info.get("type") != "array":
        raise ValueError(
            f"gtrack_array_get_colnames: '{track}' is not an array track "
            f"(type={info.get('type')!r})"
        )
    from ._array_track import read_colnames
    track_path = Path(_pymisha.pm_track_path(track))
    return read_colnames(track_path)


def gtrack_array_set_colnames(track: str, colnames: list[str]) -> None:
    """Set the column names of an ``array`` track.

    R parity for ``gtrack.array.set_colnames``. Writes
    ``<track_dir>/.colnames`` in a format both R misha and pymisha read.
    """
    _checkroot()
    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    info = gtrack_info(track)
    if info.get("type") != "array":
        raise ValueError(
            f"gtrack_array_set_colnames: '{track}' is not an array track "
            f"(type={info.get('type')!r})"
        )
    from ._array_track import write_colnames
    track_path = Path(_pymisha.pm_track_path(track))
    write_colnames(track_path, list(colnames))


def gtrack_array_extract(
    track: str,
    slice: list[str] | list[int] | None = None,
    intervals: pd.DataFrame | str | None = None,
    file: str | None = None,
) -> pd.DataFrame | None:
    """Extract per-position array values from an ``array`` track.

    R parity for ``gtrack.array.extract``. Returns a DataFrame with
    ``chrom, start, end, <selected colnames>, intervalID``. One output
    row per overlapping track interval (clipped to the query); columns
    where the track has no value are ``NaN``.

    Parameters
    ----------
    track : str
        Array track name.
    slice : list of str or int, optional
        Column subset to return. Strings are matched against the track
        colnames; integers are 0-based column indices. ``None`` returns
        all columns.
    intervals : DataFrame, optional
        Query intervals (``chrom, start, end``). Defaults to all genome.
    file : str, optional
        If given, write the result as a tab-separated table to this path
        and return ``None`` (R parity for ``gtrack.array.extract(file=)``).
        The output is the same format consumed by ``gtrack_array_import``.

    Returns
    -------
    DataFrame or None
        The extracted DataFrame, or ``None`` when ``file=`` is given.
    """
    _checkroot()
    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    info = gtrack_info(track)
    if info.get("type") != "array":
        raise ValueError(
            f"gtrack_array_extract: '{track}' is not an array track "
            f"(type={info.get('type')!r})"
        )

    from ._array_track import extract_array, read_colnames
    from .extract import _maybe_load_intervals_set

    track_path = Path(_pymisha.pm_track_path(track))
    colnames = read_colnames(track_path)

    if intervals is None:
        from .intervals import gintervals_all
        intervals = gintervals_all()
    intervals = _maybe_load_intervals_set(intervals)
    if not isinstance(intervals, pd.DataFrame):
        raise ValueError("intervals must be a DataFrame or interval-set name")

    if slice is None:
        slice_idx: list[int] | None = None
    else:
        slice_list = list(slice)
        if all(isinstance(s, str) for s in slice_list):
            cn_idx = {name: i for i, name in enumerate(colnames)}
            try:
                slice_idx = [cn_idx[str(s)] for s in slice_list]
            except KeyError as exc:
                raise ValueError(
                    f"{exc.args[0]!r} is not a column of track '{track}'"
                ) from exc
        elif all(isinstance(s, (int, np.integer)) for s in slice_list):
            int_slice: list[int] = []
            for s in slice_list:
                s_int = int(s)  # type: ignore[call-overload]
                if s_int < 0 or s_int >= len(colnames):
                    raise ValueError(
                        f"slice index {s_int} out of range [0, {len(colnames)})"
                    )
                int_slice.append(s_int)
            slice_idx = int_slice
        else:
            raise ValueError("slice must be a list of strings or ints")

    # Genome chrom-key order (used to map a chromosome to its track.idx contig
    # id when the track is in indexed format).
    from .intervals import gintervals_all
    chrom_order = gintervals_all()["chrom"].astype(str).tolist()

    df = extract_array(track_path, intervals, slice_idx, colnames, chrom_order=chrom_order)
    if file is not None:
        # R parity: gtrack.array.extract(file=) writes only the genomic + value
        # columns; intervalID is an in-memory-only attribute (see the
        # GAP_ARRAY_FILE_DUMP note in tests/r_parity/test_gtrack_array.py), with
        # the file mode R's C++ writer gives it.
        with _shared._with_cpp_umask():
            df.drop(columns=["intervalID"], errors="ignore").to_csv(file, sep="\t", index=False)
        return None
    return df
