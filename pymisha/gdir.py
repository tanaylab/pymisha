"""Database directory management functions (gdir.*)."""

import shutil
from pathlib import Path

from . import _shared
from ._name_validation import validate_dotted_name
from ._shared import _checkroot


def _tracks_root():
    return (Path(_shared._GROOT) / "tracks").resolve()


def _resolve_within_tracks(base, relpath):
    target = (base / relpath).resolve()
    tracks_root = _tracks_root()
    try:
        target.relative_to(tracks_root)
    except ValueError as exc:
        raise ValueError(
            f"Path escapes tracks tree: {relpath}"
        ) from exc
    return target, tracks_root


def gdir_cwd():
    """
    Return the current working directory in the genomic database.

    Returns the absolute path of the current working directory in the
    genomic database. This is not the shell's current working directory
    but the directory within the misha tracks tree used for resolving
    track and interval set names.

    Returns
    -------
    str
        Absolute path of the current working directory within the database.

    See Also
    --------
    gdir_cd : Change the current working directory.
    gdir_create : Create a new directory in the database.
    gdir_rm : Delete a directory from the database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdir_cwd()  # doctest: +ELLIPSIS
    '...tracks'
    """
    _checkroot()
    return _shared._GWD


def gdir_cd(dir):
    """
    Change the current working directory in the genomic database.

    Changes the directory used for resolving track and interval set names.
    The list of database objects (tracks, intervals) is rescanned
    recursively under the new directory. Object names are updated relative
    to the new working directory. For example, a track named
    ``subdir.dense`` becomes ``dense`` once the working directory is set
    to ``subdir``. All virtual tracks are cleared.

    Parameters
    ----------
    dir : str
        Directory path (relative to current working directory, or "..").

    Returns
    -------
    None

    See Also
    --------
    gdir_cwd : Return the current working directory.
    gdir_create : Create a new directory in the database.
    gdir_rm : Delete a directory from the database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdir_cd("subdir")
    >>> pm.gdir_cd("..")
    """
    if dir is None:
        raise ValueError("dir argument is required")
    _checkroot()

    old_gwd = _shared._GWD

    base = Path(old_gwd).resolve()
    target, tracks_root = _resolve_within_tracks(base, dir)

    if not target.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {target}")

    _shared._GWD = str(target)

    # Clear all virtual tracks (R parity)
    _shared._VTRACKS.clear()

    # Reload the database to rescan under new working directory
    try:
        from .db import gdb_reload
        gdb_reload()
    except Exception:
        # Rollback on failure
        _shared._GWD = old_gwd
        raise


def gdir_create(dir, show_warnings=True):
    """
    Create a new directory in the genomic database.

    Creates a single directory level under the current working directory.
    Only the last element in the specified path is created; recursive
    directory creation is not supported. A new directory cannot be created
    within an existing ``.track`` directory.

    Parameters
    ----------
    dir : str
        Directory path relative to the current working directory.
    show_warnings : bool, default True
        If True, show warnings (currently unused; kept for R parity).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the parent directory does not exist.
    ValueError
        If the target is inside a ``.track`` directory or the name ends
        with ``.track``.

    See Also
    --------
    gdir_rm : Delete a directory from the database.
    gdir_cd : Change the current working directory.
    gdir_cwd : Return the current working directory.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdir_create("my_subdir")  # doctest: +SKIP
    """
    _checkroot()

    base = Path(_shared._GWD).resolve()
    target, tracks_root = _resolve_within_tracks(base, dir)

    # Check parent exists (no recursive creation)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Path {parent} does not exist.\n"
            "Note: recursive directory creation is forbidden."
        )

    # Cannot create within a .track directory
    _check_not_inside_track(str(parent.relative_to(tracks_root)))

    # Cannot create .track directories
    if target.name.endswith(".track"):
        raise ValueError("gdir_create cannot create track directories")

    target.mkdir(exist_ok=False)


def gdir_rm(dir, recursive=False, force=False):
    """
    Delete a directory from the genomic database.

    If ``recursive`` is True, the directory is deleted with all files and
    subdirectories it contains. Cannot delete ``.track`` directories
    directly; use track-removal functions instead.

    Parameters
    ----------
    dir : str
        Directory path relative to the current working directory.
    recursive : bool, default False
        If True, delete the directory and all its contents.
    force : bool, default False
        If True, suppress errors for non-existent directories.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the directory does not exist and ``force`` is False.
    ValueError
        If the target is a ``.track`` directory.
    OSError
        If the directory is not empty and ``recursive`` is False.

    See Also
    --------
    gdir_create : Create a new directory in the database.
    gdir_cd : Change the current working directory.
    gdir_cwd : Return the current working directory.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdir_create("temp_dir")  # doctest: +SKIP
    >>> pm.gdir_rm("temp_dir")  # doctest: +SKIP
    """
    _checkroot()

    base = Path(_shared._GWD).resolve()
    target, tracks_root = _resolve_within_tracks(base, dir)

    if target == tracks_root:
        raise ValueError("Cannot remove the tracks root directory")

    if not target.exists():
        if force:
            return
        raise FileNotFoundError(f"Directory {dir} does not exist")

    if not target.is_dir():
        raise ValueError(f"{dir} is not a directory")

    # Cannot delete a .track directory via gdir_rm
    _check_not_inside_track(str(target.relative_to(tracks_root)))

    if not recursive:
        # Try to remove empty directory
        try:
            target.rmdir()
        except OSError:
            raise OSError(
                f"Directory {dir} is not empty. Use recursive=True to delete."
            ) from None
    else:
        shutil.rmtree(target)

    # Reload the database
    from .db import gdb_reload
    gdb_reload()


def gtrack_create_dirs(track, mode="0777"):
    """
    Create the directory hierarchy needed for a dotted track name.

    For example, ``gtrack_create_dirs("proj.sample.my_track")`` creates
    the directories ``proj`` and ``proj/sample`` under the current
    working directory. Use this function with caution -- a long track
    name may create a deep directory structure.

    Parameters
    ----------
    track : str
        Track name with dot-separated namespace.
    mode : str, default "0777"
        Directory permissions (currently passed to os.mkdir).

    Returns
    -------
    None

    See Also
    --------
    gdir_create : Create a single directory in the database.
    gdir_cwd : Return the current working directory.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_create_dirs("proj.sample.my_track")  # doctest: +SKIP
    """
    _checkroot()
    validate_dotted_name(track, "track name")

    parts = track.split(".")
    if len(parts) <= 1:
        # Simple track name, no namespace dirs to create
        return

    # All parts except the last are namespace directories
    namespace_parts = parts[:-1]
    base = Path(_shared._GWD).resolve()
    tracks_root = _tracks_root()

    current = base
    for part in namespace_parts:
        current = current / part
        _resolve_within_tracks(base, current.relative_to(base))
        _check_not_inside_track(str(current.relative_to(tracks_root)))
        if not current.exists():
            current.mkdir()


def _check_not_inside_track(relpath):
    """Raise if relpath is inside a .track directory."""
    parts = Path(relpath).parts
    for part in parts:
        if part.endswith(".track"):
            raise ValueError(
                f"Cannot operate within track directory {part}"
            )
