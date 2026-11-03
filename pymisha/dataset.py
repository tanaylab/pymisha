"""Dataset management helpers."""

from __future__ import annotations

import datetime as _datetime
import getpass as _getpass
import hashlib
import os
import shutil
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import _shared
from ._log import get_logger
from ._shared import _checkroot, _pymisha, _with_umask

_logger = get_logger(__name__)


def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _as_list(values: str | Iterable[str] | None, name: str) -> list[str]:
    if values is None:
        return []
    out = [values] if isinstance(values, str) else list(values)
    if not out:
        return []
    for item in out:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must contain non-empty strings")
    return out


def _resource_path(root: str, name: str, suffix: str) -> Path:
    return Path(root) / "tracks" / f"{name.replace('.', '/')}{suffix}"


def _chrom_sizes_hash(root: str) -> str:
    chrom_path = Path(root) / "chrom_sizes.txt"
    h = hashlib.sha256()
    with open(chrom_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan_tracks(root: str) -> set[str]:
    tracks_dir = Path(root) / "tracks"
    if not tracks_dir.exists():
        return set()

    track_names: set[str] = set()

    for cur_dir, dirs, _ in os.walk(tracks_dir):
        # Skip hidden dirs and non-track dotted dirs
        pruned = []
        for d in dirs:
            if d.startswith("."):
                continue
            if d.endswith(".track"):
                rel = Path(cur_dir).relative_to(tracks_dir) / d
                parts = list(rel.parts)
                parts[-1] = parts[-1][:-len(".track")]
                track_names.add(".".join(parts))
                continue
            if "." in d:
                continue
            pruned.append(d)
        dirs[:] = pruned

    return track_names


def _scan_intervals(root: str) -> set[str]:
    tracks_dir = Path(root) / "tracks"
    if not tracks_dir.exists():
        return set()

    names: set[str] = set()
    suffixes = (".interv", ".interv2d")
    # os.walk (not rglob): it ignores scandir errors, so a misha `.trash.*`
    # removal-target that vanishes mid-walk doesn't raise FileNotFoundError.
    # Prune dotted dirs (incl. `.trash.*`) and don't descend into set dirs.
    for cur_dir, dirs, files in os.walk(tracks_dir):
        rel_dir = Path(cur_dir).relative_to(tracks_dir)
        # An interval set is either a `.interv` file (small) or dir (bigset).
        for entry in dirs + files:
            for suffix in suffixes:
                if entry.endswith(suffix):
                    name = str(rel_dir / entry)[:-len(suffix)].replace("/", ".").replace("\\", ".")
                    names.add(name)
                    break
        dirs[:] = [d for d in dirs if not d.startswith(".") and not d.endswith(suffixes)]
    return names


_DATASET_SCAN_CACHE: dict[str, tuple[set[str], set[str]]] = {}


def _clear_dataset_scan_cache() -> None:
    _DATASET_SCAN_CACHE.clear()


def _sync_dataset_scan_cache() -> None:
    keep = set(_shared._GDATASETS)
    if _shared._GROOT:
        keep.add(_normalize_path(_shared._GROOT))
    if _shared._UROOT:
        keep.add(_normalize_path(_shared._UROOT))
    stale = [p for p in _DATASET_SCAN_CACHE if p not in keep]
    for path in stale:
        _DATASET_SCAN_CACHE.pop(path, None)


def _set_datasets() -> None:
    """Hand the loaded datasets to the C++ side, which rescans the track list (the names a
    working directory that is a subdirectory lists may change too), and drop the Python views
    of the old list."""
    from .tracks import _clear_computed_track_cache

    _pymisha.pm_dbsetdatasets(_shared._GDATASETS)
    _shared._clear_track_names_cache()
    _clear_computed_track_cache()


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    text = text.replace("'", "''")
    return f"'{text}'"


@_with_umask()   # database writes carry misha's permissions
def _write_dataset_metadata(path: Path, metadata: dict[str, Any]) -> None:
    # Keep schema flat and scalar so the file can be parsed without pyyaml.
    lines = [f"{key}: {_yaml_scalar(value)}" for key, value in metadata.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_dataset_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError:
        _logger.debug("PyYAML is not installed; parsing %s as plain key: value", path, exc_info=True)
    except Exception:
        # A broken PyYAML install used to fall through to the plain parser too.
        _logger.warning("PyYAML could not be imported; parsing %s as plain key: value", path,
                        exc_info=True)
    else:
        try:
            yaml_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            _logger.warning("could not parse %s as YAML; falling back to plain key: value", path,
                            exc_info=True)
        else:
            if isinstance(yaml_data, dict):
                return yaml_data

    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if raw_val.lower() in {"null", "~", ""}:
            data[key] = None
            continue
        if raw_val.lower() in {"true", "false"}:
            data[key] = raw_val.lower() == "true"
            continue
        if (raw_val.startswith("'") and raw_val.endswith("'")) or (
            raw_val.startswith('"') and raw_val.endswith('"')
        ):
            raw_val = raw_val[1:-1]
            raw_val = raw_val.replace("''", "'")
            data[key] = raw_val
            continue
        try:
            data[key] = int(raw_val)
        except ValueError:
            data[key] = raw_val
    return data


@_with_umask()   # database writes carry misha's permissions
def _copy_or_link(src: Path, dst: Path, symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlinks:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _strip_chr(name: str) -> str:
    return name[3:] if name.startswith("chr") else name


def _session_chroms() -> list[str]:
    """The working database's chromosome names by chrom id."""
    return [str(c) for c in _pymisha.pm_intervals_all()[1]]


def _dataset_chrom_order(
    path_norm: str, groot: str, cs_path: Path, dataset_hash: str
) -> tuple[list[str] | None, bool, str | None]:
    """The chromosome order of a dataset with the working database's chrom_sizes.txt,
    when it differs from the session's (R's gdataset.load): (names by chrom id or None,
    whether its seq/genome.idx does not match its chrom_sizes.txt, the error of an
    indexed seq/ of its own whose index cannot be read or None).

    The same chrom_sizes.txt can number the chromosomes differently: by the sorted
    names in a per-chromosome database whose names lack "chr" and whose .seq files have
    it, in file order otherwise. The dataset's order, from its chrom_sizes.txt and its
    seq/ as gdb_init would read them, is compared with the session's. They are equal by
    construction when the dataset's seq/ is the working database's, or when both are
    indexed and the dataset's seq/ is a directory of its own, or a link into a database
    whose chrom_sizes.txt is the dataset's. A dataset whose order cannot be read (no
    sequence, the unreadable index of a linked seq/) is taken to be in the session's
    order; one with an indexed seq/ of its own whose index cannot be read, in
    chrom_sizes.txt order, as an indexed database numbers its chromosomes; one whose
    index does not match its chrom_sizes.txt (a seq/ linked to a database converted
    since), in chrom_sizes.txt order too.
    """
    dataset_seq = os.path.join(path_norm, "seq")
    groot_seq = os.path.join(groot, "seq")
    genome_files = ("genome.idx", "genome.seq")
    seq_owner_cs = Path(os.path.dirname(os.path.realpath(dataset_seq))) / "chrom_sizes.txt"
    both_indexed = (
        all(os.path.exists(os.path.join(dataset_seq, f)) for f in genome_files)
        and all(os.path.exists(os.path.join(groot_seq, f)) for f in genome_files)
        and (
            not os.path.islink(dataset_seq)
            or (seq_owner_cs.exists() and seq_owner_cs.stat().st_size == cs_path.stat().st_size
                and _chrom_sizes_hash(str(seq_owner_cs.parent)) == dataset_hash)
        )
    )
    if os.path.realpath(dataset_seq) == os.path.realpath(groot_seq) or both_indexed:
        return None, False, None

    error: str | None = None
    dataset_chroms: list[str] | None = None
    try:
        # without the warning gdb_init gives for index names other than chrom_sizes.txt's
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset_chroms = _shared._gdb_chrom_names_at(path_norm, from_disk=True)
    except Exception as exc:
        error = str(exc)
    index_mismatch = error is not None and "genome.idx does not match chrom_sizes.txt" in error
    index_unreadable = None
    if (error is not None and not index_mismatch and not os.path.islink(dataset_seq)
            and all(os.path.exists(os.path.join(dataset_seq, f)) for f in genome_files)):
        index_unreadable = error
    if index_mismatch or index_unreadable is not None:
        dataset_chroms = _shared._chrom_sizes_names_at(path_norm)
    if not index_mismatch and (
        dataset_chroms is None
        or [_strip_chr(c) for c in dataset_chroms] == [_strip_chr(c) for c in _session_chroms()]
    ):
        dataset_chroms = None
    return dataset_chroms, index_mismatch, index_unreadable


def gdataset_ls() -> list[str]:
    """
    List currently loaded datasets.

    Returns the normalized absolute paths of all datasets that have been
    loaded into the current session via `gdataset_load`.

    Returns
    -------
    list[str]
        Normalized absolute paths of loaded datasets. Empty list if no
        datasets are loaded.

    See Also
    --------
    gdataset_load : Load a dataset into the namespace.
    gdataset_info : Return metadata for a dataset.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdataset_ls()
    []
    """
    _checkroot()
    return list(_shared._GDATASETS)


def gdataset_load(path: str, force: bool = False, verbose: bool = False):
    """
    Load a dataset into the namespace.

    Loads tracks and intervals from a dataset directory, making them
    available for analysis alongside the working database. If the dataset
    contains tracks or intervals whose names collide with objects in the
    working database or previously loaded datasets, an error is raised
    unless ``force=True``. When collisions are forced, the working
    database always wins; for dataset-to-dataset collisions, the
    later-loaded dataset overrides earlier ones.

    Parameters
    ----------
    path : str
        Path to a dataset or misha database directory.
    force : bool, default False
        If True, ignore name collisions (working db wins; later datasets override earlier).
    verbose : bool, default False
        Print loaded track/interval counts.

    Returns
    -------
    dict
        Dictionary with keys ``"tracks"``, ``"intervals"``,
        ``"shadowed_tracks"``, and ``"shadowed_intervals"`` indicating
        how many objects were loaded and how many were shadowed by
        collisions.

    Raises
    ------
    ValueError
        If the dataset path does not exist, lacks a ``tracks/`` directory,
        has a mismatched genome, or has collisions without ``force=True``;
        or if it numbers its chromosomes differently from the working
        database (the same chrom_sizes.txt can, by its seq/) and holds
        indexed tracks or interval sets, which would read other
        chromosomes. Without those it loads, and keeps its own order.

    See Also
    --------
    gdataset_unload : Unload a dataset from the namespace.
    gdataset_save : Save tracks/intervals as a dataset.
    gdataset_ls : List loaded datasets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gdataset_load("/path/to/dataset")  # doctest: +SKIP
    >>> result["tracks"]  # doctest: +SKIP
    5
    """
    _checkroot()
    assert _shared._GROOT is not None

    if not os.path.isdir(path):
        raise ValueError(f"Dataset path '{path}' does not exist")

    path_norm = _normalize_path(path)
    groot = _shared._GROOT
    if path_norm == _normalize_path(groot):
        return {"tracks": 0, "intervals": 0, "shadowed_tracks": 0, "shadowed_intervals": 0}

    if path_norm in _shared._GDATASETS:
        gdataset_unload(path_norm, validate=False)

    tracks_dir = Path(path_norm) / "tracks"
    if not tracks_dir.exists():
        raise ValueError(f"Path '{path}' does not contain a 'tracks' directory")

    cs_path = Path(path_norm) / "chrom_sizes.txt"
    if not cs_path.exists():
        raise ValueError(f"Path '{path}' does not contain a chrom_sizes.txt file")

    dataset_hash = _chrom_sizes_hash(path_norm)
    if dataset_hash != _chrom_sizes_hash(groot):
        raise ValueError(f"Cannot load dataset '{path}': genome does not match working database")

    dataset_chroms, index_mismatch, index_unreadable = _dataset_chrom_order(
        path_norm, groot, cs_path, dataset_hash
    )

    dataset_tracks = _scan_tracks(path_norm)
    dataset_intervals = _scan_intervals(path_norm)

    # In another order: files keyed by chrom id, which would read other chromosomes
    if dataset_chroms is not None:
        keyed = [tracks_dir / f"{t.replace('.', '/')}.track" / "track.idx" for t in dataset_tracks]
        for name in dataset_intervals:
            keyed += [tracks_dir / f"{name.replace('.', '/')}.interv" / f for f in ("intervals.idx", "intervals2d.idx")]
        if any(p.exists() for p in keyed):
            if index_mismatch:
                raise ValueError(
                    f"Cannot load dataset '{path}': its seq/genome.idx does not match its chrom_sizes.txt, "
                    "so the chrom ids of its indexed tracks and interval sets cannot be checked"
                )
            groot_chroms = _session_chroms()
            i = next(
                k for k in range(max(len(dataset_chroms), len(groot_chroms)))
                if k >= len(dataset_chroms) or k >= len(groot_chroms)
                or _strip_chr(dataset_chroms[k]) != _strip_chr(groot_chroms[k])
            )
            ds_name = dataset_chroms[i] if i < len(dataset_chroms) else "NA"
            groot_name = groot_chroms[i] if i < len(groot_chroms) else "NA"
            if index_unreadable is not None:
                raise ValueError(
                    f"Cannot load dataset '{path}': its seq/genome.idx could not be read ({index_unreadable}), "
                    "so the chrom ids of its indexed tracks and interval sets cannot be checked; an indexed "
                    "database numbers its chromosomes in chrom_sizes.txt order, which differs from the working "
                    f"database's (chrom id {i} is {ds_name} in the dataset and {groot_name} in the working database)"
                )
            raise ValueError(
                f"Cannot load dataset '{path}': it numbers the chromosomes differently from the working "
                f"database, though its chrom_sizes.txt is the same (chrom id {i} is {ds_name} in the dataset "
                f"and {groot_name} in the working database), so its indexed tracks and interval sets would "
                "read other chromosomes"
            )

    # Use C++ track cache for existing tracks (avoids redundant Python
    # filesystem scans of the working db, user root, and loaded datasets).
    existing_tracks = set(_pymisha.pm_track_names())

    # For intervals, use cached scans (C++ backend doesn't cache intervals).
    existing_intervals: set[str] = set()
    _sync_dataset_scan_cache()
    for db in [groot] + ([_shared._UROOT] if _shared._UROOT else []) + list(_shared._GDATASETS):
        cached = _DATASET_SCAN_CACHE.get(db)
        if cached is None:
            cached = (_scan_tracks(db), _scan_intervals(db))
            _DATASET_SCAN_CACHE[db] = cached
        existing_intervals |= cached[1]

    track_collisions = dataset_tracks & existing_tracks
    interval_collisions = dataset_intervals & existing_intervals

    if (track_collisions or interval_collisions) and not force:
        msgs = []
        if track_collisions:
            msgs.append(f"tracks already exist: {sorted(track_collisions)[:5]}")
        if interval_collisions:
            msgs.append(f"interval sets already exist: {sorted(interval_collisions)[:5]}")
        msg = "Cannot load dataset with collisions. Use force=True to override.\n" + "\n".join(msgs)
        raise ValueError(msg)

    # Update datasets list (load order)
    _shared._GDATASETS.append(path_norm)
    if dataset_chroms is not None:
        _shared._GDATASET_CHROMS[path_norm] = dataset_chroms
    _set_datasets()
    _DATASET_SCAN_CACHE[path_norm] = (dataset_tracks, dataset_intervals)

    # Visible/shadowed counts (working db always wins)
    shadowed_tracks = len(dataset_tracks & existing_tracks)
    shadowed_intervals = len(dataset_intervals & existing_intervals)
    visible_tracks = len(dataset_tracks) - shadowed_tracks
    visible_intervals = len(dataset_intervals) - shadowed_intervals

    if verbose:
        print(f"Loaded dataset '{path_norm}':")
        print(f"  Tracks: {visible_tracks} visible, {shadowed_tracks} shadowed")
        print(f"  Intervals: {visible_intervals} visible, {shadowed_intervals} shadowed")

    return {
        "tracks": visible_tracks,
        "intervals": visible_intervals,
        "shadowed_tracks": shadowed_tracks,
        "shadowed_intervals": shadowed_intervals,
    }


def gdataset_unload(path: str, validate: bool = False):
    """
    Unload a dataset from the namespace.

    Removes all tracks and intervals from a previously loaded dataset.
    If a dataset track was shadowing another, the shadowed track becomes
    visible again.

    Parameters
    ----------
    path : str
        Path to a previously loaded dataset.
    validate : bool, default False
        If True, raise an error if the path is not currently loaded.
        Otherwise silently no-op.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``validate`` is True and the dataset is not currently loaded.

    See Also
    --------
    gdataset_load : Load a dataset into the namespace.
    gdataset_ls : List loaded datasets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdataset_load("/path/to/dataset")  # doctest: +SKIP
    >>> pm.gdataset_unload("/path/to/dataset", validate=True)  # doctest: +SKIP
    """
    _checkroot()

    path_norm = _normalize_path(path)
    if path_norm not in _shared._GDATASETS:
        if validate:
            raise ValueError(f"Dataset '{path}' is not loaded")
        return

    _shared._GDATASETS = [p for p in _shared._GDATASETS if p != path_norm]
    _shared._GDATASET_CHROMS.pop(path_norm, None)
    _set_datasets()
    _DATASET_SCAN_CACHE.pop(path_norm, None)
    return


@_with_umask()   # database writes carry misha's permissions
def gdataset_save(
    path: str,
    description: str,
    tracks: str | Iterable[str] | None = None,
    intervals: str | Iterable[str] | None = None,
    symlinks: bool = False,
    copy_seq: bool = False,
) -> str:
    """
    Save selected tracks/intervals into a standalone dataset directory.

    Parameters
    ----------
    path : str
        Destination directory. Must not exist.
    description : str
        Dataset description stored in ``misha.yaml``.
    tracks : str | Iterable[str] | None
        Track names to include.
    intervals : str | Iterable[str] | None
        Interval set names to include.
    symlinks : bool, default False
        If True, link track/interval resources instead of copying.
    copy_seq : bool, default False
        If True, copy ``seq/``. Otherwise create a symlink to the working DB ``seq/``.

    Returns
    -------
    str
        Absolute path of the created dataset directory.

    Raises
    ------
    ValueError
        If neither ``tracks`` nor ``intervals`` is specified, the path
        already exists, or a requested track/interval does not exist.

    See Also
    --------
    gdataset_load : Load a dataset into the namespace.
    gdataset_info : Return metadata for a dataset.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdataset_save(  # doctest: +SKIP
    ...     "/tmp/my_dataset",
    ...     description="Example dataset",
    ...     tracks=["dense_track"],
    ...     intervals=["my_intervals"],
    ... )
    """
    _checkroot()
    assert _shared._GROOT is not None

    track_names = _as_list(tracks, "tracks")
    interval_names = _as_list(intervals, "intervals")

    if not track_names and not interval_names:
        raise ValueError("At least one of 'tracks' or 'intervals' must be specified")

    out_path = Path(path).expanduser().resolve()
    if out_path.exists():
        raise ValueError(f"Path '{path}' already exists")

    from .intervals import _interv_file

    track_pairs: list[tuple[Path, Path]] = []
    interval_pairs: list[tuple[Path, Path]] = []

    for track in track_names:
        # where the session finds it (under the working directory after gdir_cd)
        track_path = _pymisha.pm_track_path(track)
        if not track_path:
            raise ValueError(f"Track '{track}' does not exist")
        src = Path(track_path)
        if not src.exists():
            raise ValueError(f"Track '{track}' path does not exist: {src}")
        dst = _resource_path(str(out_path), track, ".track")
        track_pairs.append((src, dst))

    for interval in interval_names:
        interv_file = _interv_file(interval)
        if interv_file is None:
            raise ValueError(f"Interval set '{interval}' does not exist")
        src = interv_file
        suffix = ".interv2d" if src.name.endswith(".interv2d") else ".interv"
        dst = _resource_path(str(out_path), interval, suffix)
        interval_pairs.append((src, dst))

    out_path.mkdir(parents=True)
    (out_path / "tracks").mkdir()

    root = Path(_shared._GROOT)
    shutil.copy2(root / "chrom_sizes.txt", out_path / "chrom_sizes.txt")

    if copy_seq:
        shutil.copytree(root / "seq", out_path / "seq")
    else:
        os.symlink(root / "seq", out_path / "seq", target_is_directory=True)

    for src, dst in track_pairs:
        _copy_or_link(src, dst, symlinks=symlinks)

    for src, dst in interval_pairs:
        _copy_or_link(src, dst, symlinks=symlinks)

    metadata = {
        "description": description,
        "author": _getpass.getuser(),
        "created": _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "original_db": str(root),
        "misha_version": "pymisha",
        "track_count": len(track_names),
        "interval_count": len(interval_names),
        "genome": _chrom_sizes_hash(str(root)),
    }
    _write_dataset_metadata(out_path / "misha.yaml", metadata)

    return str(out_path)


def gdataset_info(path: str) -> dict[str, Any]:
    """
    Return metadata and contents summary for a dataset path.

    Reads the ``misha.yaml`` metadata file and scans the dataset for
    tracks and intervals. The dataset does not need to be loaded.

    Parameters
    ----------
    path : str
        Path to a dataset directory (loaded or not).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``"description"``, ``"author"``,
        ``"created"``, ``"original_db"``, ``"misha_version"``,
        ``"track_count"``, ``"interval_count"``, ``"genome"``, and
        ``"is_loaded"``.

    See Also
    --------
    gdataset_ls : List loaded datasets.
    gdataset_load : Load a dataset into the namespace.

    Examples
    --------
    >>> import pymisha as pm
    >>> info = pm.gdataset_info("/path/to/dataset")  # doctest: +SKIP
    >>> info["track_count"]  # doctest: +SKIP
    3
    """
    ds_path = Path(path).expanduser().resolve(strict=True)

    metadata = _parse_dataset_metadata(ds_path / "misha.yaml")
    tracks = _scan_tracks(str(ds_path))
    intervals = _scan_intervals(str(ds_path))

    cs_path = ds_path / "chrom_sizes.txt"
    genome = _chrom_sizes_hash(str(ds_path)) if cs_path.exists() else None

    path_norm = _normalize_path(str(ds_path))
    return {
        "description": metadata.get("description"),
        "author": metadata.get("author"),
        "created": metadata.get("created"),
        "original_db": metadata.get("original_db"),
        "misha_version": metadata.get("misha_version"),
        "track_count": len(tracks),
        "interval_count": len(intervals),
        "genome": genome,
        "is_loaded": path_norm in _shared._GDATASETS,
    }


def gdataset_example_path() -> str:
    """
    Create and return the path to an example dataset.

    Creates a small dataset in a temporary directory using the built-in
    example database.  This mirrors R misha's ``gdataset.example_path()``
    and is intended for use in examples and tests.

    The function has side effects: it calls
    :func:`~pymisha.gdb_init_examples` (which resets the working
    database), creates temporary tracks and intervals in that database,
    saves them into a dataset, and then removes the temporary objects.

    Returns
    -------
    str
        Absolute path to the created dataset directory (in a temporary
        location).

    See Also
    --------
    gdataset_save : Save tracks/intervals as a dataset.
    gdataset_load : Load a dataset into the namespace.

    Examples
    --------
    >>> import pymisha as pm
    >>> dataset_path = pm.gdataset_example_path()
    >>> pm.gdataset_load(dataset_path)['tracks']
    1
    >>> pm.gdataset_unload(dataset_path)
    """
    import tempfile

    from .db import gdb_init_examples
    from .intervals import gintervals, gintervals_rm, gintervals_save
    from .tracks import gtrack_create, gtrack_rm

    gdb_init_examples()

    _TRACK_NAME = "example_dataset_track"
    _INTERV_NAME = "example_dataset_intervals"

    # Remove leftovers from a previous call (idempotent). Kept as two independent
    # handlers: a failure removing the track must not skip removing the intervals.
    try:
        gtrack_rm(_TRACK_NAME, force=True)
    except (_pymisha.error, OSError, RuntimeError, ValueError):
        _logger.debug("no leftover track %r to remove", _TRACK_NAME, exc_info=True)
    try:
        gintervals_rm(_INTERV_NAME, force=True)
    except (_pymisha.error, OSError, RuntimeError, ValueError):
        _logger.debug("no leftover interval set %r to remove", _INTERV_NAME, exc_info=True)

    # Create a small interval set and a derived track
    example_intervs = gintervals("1", 0, 10000)
    gintervals_save(example_intervs, _INTERV_NAME)
    gtrack_create(
        _TRACK_NAME,
        "Example dataset track",
        "dense_track",
        iterator=10000,
    )

    # Save as a dataset in a temp dir.  gdataset_save requires a
    # non-existent path, so we create the parent and let it make the leaf.
    parent = tempfile.mkdtemp(prefix="pymisha_dataset_")
    dataset_path = os.path.join(parent, "dataset")
    gdataset_save(
        path=dataset_path,
        description="Example dataset created on the fly",
        tracks=[_TRACK_NAME],
        intervals=[_INTERV_NAME],
    )

    # Clean up the temporary objects from the example DB
    gtrack_rm(_TRACK_NAME, force=True)
    gintervals_rm(_INTERV_NAME, force=True)

    return dataset_path
