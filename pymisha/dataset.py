"""Dataset management helpers."""

from __future__ import annotations

import datetime as _datetime
import getpass as _getpass
import hashlib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import _shared
from ._shared import _checkroot, _pymisha


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
    for suffix in (".interv", ".interv2d"):
        for path in tracks_dir.rglob(f"*{suffix}"):
            rel = path.relative_to(tracks_dir)
            name = str(rel)[:-len(suffix)].replace("/", ".").replace("\\", ".")
            names.add(name)
    return names


_DATASET_SCAN_CACHE: dict[str, tuple[set[str], set[str]]] = {}


def _sync_dataset_scan_cache() -> None:
    loaded = set(_shared._GDATASETS)
    stale = [p for p in _DATASET_SCAN_CACHE if p not in loaded]
    for path in stale:
        _DATASET_SCAN_CACHE.pop(path, None)


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


def _write_dataset_metadata(path: Path, metadata: dict[str, Any]) -> None:
    # Keep schema flat and scalar so the file can be parsed without pyyaml.
    lines = [f"{key}: {_yaml_scalar(value)}" for key, value in metadata.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_dataset_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore

        yaml_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(yaml_data, dict):
            return yaml_data
    except Exception:
        pass

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


def _copy_or_link(src: Path, dst: Path, symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlinks:
        os.symlink(src, dst, target_is_directory=src.is_dir())
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


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
        has a mismatched genome, or has collisions without ``force=True``.

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

    if _chrom_sizes_hash(path_norm) != _chrom_sizes_hash(groot):
        raise ValueError(f"Cannot load dataset '{path}': genome does not match working database")

    dataset_tracks = _scan_tracks(path_norm)
    dataset_intervals = _scan_intervals(path_norm)

    existing_tracks = _scan_tracks(groot)
    existing_intervals = _scan_intervals(groot)
    if _shared._UROOT:
        existing_tracks |= _scan_tracks(_shared._UROOT)
        existing_intervals |= _scan_intervals(_shared._UROOT)
    _sync_dataset_scan_cache()
    for ds in _shared._GDATASETS:
        cached = _DATASET_SCAN_CACHE.get(ds)
        if cached is None:
            cached = (_scan_tracks(ds), _scan_intervals(ds))
            _DATASET_SCAN_CACHE[ds] = cached
        existing_tracks |= cached[0]
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
    _pymisha.pm_dbsetdatasets(_shared._GDATASETS)
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
    _pymisha.pm_dbsetdatasets(_shared._GDATASETS)
    _DATASET_SCAN_CACHE.pop(path_norm, None)
    return


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

    track_names = _as_list(tracks, "tracks")
    interval_names = _as_list(intervals, "intervals")

    if not track_names and not interval_names:
        raise ValueError("At least one of 'tracks' or 'intervals' must be specified")

    out_path = Path(path).expanduser().resolve()
    if out_path.exists():
        raise ValueError(f"Path '{path}' already exists")

    from .intervals import gintervals_dataset
    from .tracks import gtrack_dataset

    track_pairs: list[tuple[Path, Path]] = []
    interval_pairs: list[tuple[Path, Path]] = []

    for track in track_names:
        source_db = gtrack_dataset(track)
        if not source_db:
            raise ValueError(f"Track '{track}' does not exist")
        src = _resource_path(source_db, track, ".track")
        if not src.exists():
            raise ValueError(f"Track '{track}' path does not exist: {src}")
        dst = _resource_path(str(out_path), track, ".track")
        track_pairs.append((src, dst))

    for interval in interval_names:
        source_db = gintervals_dataset(interval)
        if not source_db:
            raise ValueError(f"Interval set '{interval}' does not exist")
        src = _resource_path(source_db, interval, ".interv")
        if not src.exists():
            src = _resource_path(source_db, interval, ".interv2d")
        if not src.exists():
            raise ValueError(f"Interval set '{interval}' path does not exist")
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
