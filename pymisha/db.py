"""Database initialization and example DB helpers."""

import atexit
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from . import _shared
from ._shared import CONFIG, _checkroot, _pymisha

_EXAMPLE_TMP_DIRS = []


def _cleanup_example_tmpdirs():
    for tmp in _EXAMPLE_TMP_DIRS:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp, ignore_errors=True)


atexit.register(_cleanup_example_tmpdirs)


def gdb_init(path: str, userpath: str = None):
    """
    Initialize connection to a misha genomic database.

    Loads the genome database at the given path and makes it available for
    all subsequent genomic operations. Must be called before any other
    pymisha function that accesses track data.

    Parameters
    ----------
    path : str
        Path to the root directory of the genome database.
    userpath : str, optional
        Path to a user-writable database root. New tracks and interval
        sets will be created here. If None, defaults to ``path``.

    Returns
    -------
    None

    See Also
    --------
    gdb_reload : Refresh track lists after external changes.
    gdb_unload : Disconnect from the database and clear all state.
    gdb_info : Return metadata about the database.
    gsetroot : Alternative entry point with directory validation.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()  # initializes a real test DB
    """
    db_path = Path(path).expanduser()
    if not db_path.exists():
        raise FileNotFoundError(f"Database path does not exist: {path}")
    _shared._GROOT = str(db_path)
    _shared._UROOT = userpath
    _shared._GWD = str(db_path / "tracks")
    _shared._GDATASETS = []
    _shared._VTRACKS = {}

    _pymisha.pm_dbinit(str(db_path), userpath or "", CONFIG)
    _pymisha.pm_dbsetdatasets([])


def gdb_reload():
    """
    Reload the database, refreshing track lists and metadata.

    Re-scans the database root directories for newly created or removed
    tracks and interval sets. Call this after external modifications to
    the database on disk (e.g., tracks created by R misha or another
    process).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If no database is currently initialized.

    See Also
    --------
    gdb_init : Initialize a database connection.
    gdb_unload : Disconnect from the database entirely.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdb_reload()
    """
    _checkroot()
    _pymisha.pm_dbreload()


def gdb_unload():
    """
    Unload the database, clearing all state.

    Disconnects from the currently active genome database and resets all
    internal state including the database root paths, working directory,
    datasets, and virtual tracks. After calling this function, a new
    :func:`gdb_init` call is required before any genomic operations.

    Returns
    -------
    None

    See Also
    --------
    gdb_init : Initialize a new database connection.
    gdb_reload : Refresh without disconnecting.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdb_unload()
    """
    _pymisha.pm_dbunload()
    _shared._GROOT = None
    _shared._UROOT = None
    _shared._GWD = None
    _shared._GDATASETS = []
    _shared._VTRACKS = {}


def gdb_examples_path():
    """
    Return the path to the example database if available.

    Checks the following locations in order:
    1) PYMISHA_EXAMPLES_DB environment variable
    2) pymisha/examples/trackdb/test (if packaged)
    3) tests/testdb/trackdb/test (repo checkout)

    Returns
    -------
    str
        Absolute path to the example database root directory.

    Raises
    ------
    FileNotFoundError
        If the example database cannot be located in any of the
        searched locations.

    See Also
    --------
    gdb_init_examples : Initialize the example database.
    gdb_init : Initialize a custom database.

    Examples
    --------
    >>> import pymisha as pm
    >>> path = pm.gdb_examples_path()  # doctest: +ELLIPSIS
    >>> path  # doctest: +SKIP
    """
    env = os.environ.get("PYMISHA_EXAMPLES_DB")
    if env:
        return env

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "examples" / "trackdb" / "test",
        here.parents[1] / "tests" / "testdb" / "trackdb" / "test",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Example database not found. Set PYMISHA_EXAMPLES_DB to a trackdb/test directory."
    )


def gsetroot(groot, subdir=None, rescan=False, **kwargs):
    """
    Set the database root directory with validation.

    Connects to a genome database after verifying that the directory
    exists and contains the required ``tracks/`` and ``seq/``
    subdirectories. This matches the R ``gsetroot()`` interface and is
    the recommended entry point when working interactively, since it
    provides clear error messages for invalid database paths.

    Parameters
    ----------
    groot : str
        Path to the genome database root directory.
    subdir : str, optional
        Sub-directory within ``tracks/`` to use as working directory
        after initialization.
    dir : str, optional
        Backward-compatible alias for ``subdir``.
    rescan : bool, default False
        If True, force a rescan of the database after initialization.
        Equivalent to calling :func:`gdb_reload` after :func:`gdb_init`.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If ``groot`` does not exist, or is missing the required
        ``tracks/`` or ``seq/`` subdirectories.

    See Also
    --------
    gdb_init : Lower-level initializer without directory validation.
    gdb_reload : Refresh track lists without re-initializing.

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.gsetroot(pm.gdb_examples_path())
    """
    if "dir" in kwargs:
        if subdir is not None:
            raise ValueError("Specify only one of 'subdir' or 'dir'")
        subdir = kwargs.pop("dir")
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {bad}")

    p = Path(groot)
    if not p.exists():
        raise FileNotFoundError(f"Database directory does not exist: {groot}")
    if not (p / "tracks").exists():
        raise FileNotFoundError(
            f"Database directory '{groot}' does not contain a 'tracks' subdirectory. "
            "This does not appear to be a valid misha database."
        )
    if not (p / "seq").exists():
        raise FileNotFoundError(
            f"Database directory '{groot}' does not contain a 'seq' subdirectory. "
            "This does not appear to be a valid misha database."
        )

    gdb_init(str(p.resolve()))

    if subdir is not None:
        from .gdir import gdir_cd

        gdir_cd(subdir)

    if rescan:
        gdb_reload()


def gdb_init_examples(copy=True):
    """
    Initialize the example database (mirrors R's gdb.init_examples).

    Parameters
    ----------
    copy : bool, default True
        If True, copy the example DB into a temp dir before initializing.
        This avoids mutating the repo data when running examples.

    Returns
    -------
    str
        Path to the initialized example DB.

    See Also
    --------
    gdb_examples_path : Get the path to the example database.
    gdb_init : Initialize a custom database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_ls()  # doctest: +NORMALIZE_WHITESPACE
    ['array_track', 'dense_track', 'rects_track', 'sparse_track', 'subdir.dense_track2']
    """
    src = Path(gdb_examples_path())
    if copy:
        tmpdir = Path(tempfile.mkdtemp(prefix="pymisha-example-"))
        _EXAMPLE_TMP_DIRS.append(tmpdir)
        dst = tmpdir / "trackdb" / "test"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        gdb_init(str(dst))
        return str(dst)

    gdb_init(str(src))
    return str(src)


def gdb_info(groot: str = None):
    """
    Return high-level information about a misha database.

    Inspects a genome database directory and returns metadata including
    the storage format, number of chromosomes, total genome size, and a
    table of per-chromosome sizes. Can be used to validate a database
    path without fully initializing a connection.

    Parameters
    ----------
    groot : str, optional
        Path to a database root directory. If ``None``, uses the
        currently initialized database.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``path`` (str) -- Resolved absolute path to the database.
        - ``is_db`` (bool) -- Whether the path is a valid misha database.
        - ``error`` (str) -- Present only when ``is_db`` is False;
          describes why validation failed.
        - ``format`` (str) -- ``"indexed"`` or ``"per-chromosome"``.
          Present only when ``is_db`` is True.
        - ``num_chromosomes`` (int) -- Number of chromosomes.
          Present only when ``is_db`` is True.
        - ``genome_size`` (int) -- Sum of all chromosome sizes.
          Present only when ``is_db`` is True.
        - ``chromosomes`` (pandas.DataFrame) -- Two-column table with
          ``chrom`` and ``size``. Present only when ``is_db`` is True.

    Raises
    ------
    ValueError
        If ``groot`` is None and no database is currently initialized.

    See Also
    --------
    gdb_init : Initialize a database connection.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> info = pm.gdb_info()
    >>> info["num_chromosomes"]
    3
    >>> info["genome_size"]
    1000000
    """
    if groot is None:
        if _shared._GROOT is None:
            raise ValueError(
                "No database is currently active. Call gdb_init() or pass groot."
            )
        groot = _shared._GROOT

    db_path = str(Path(groot).expanduser().resolve(strict=False))

    if not os.path.isdir(db_path):
        return {
            "path": db_path,
            "is_db": False,
            "error": "Directory does not exist",
        }

    chrom_sizes_path = os.path.join(db_path, "chrom_sizes.txt")
    if not os.path.exists(chrom_sizes_path):
        return {
            "path": db_path,
            "is_db": False,
            "error": "Not a misha database (chrom_sizes.txt not found)",
        }

    try:
        chrom_sizes = pd.read_csv(
            chrom_sizes_path,
            sep="\t",
            header=None,
            names=["chrom", "size"],
            dtype={"chrom": str, "size": "int64"},
        )
    except Exception:
        return {
            "path": db_path,
            "is_db": False,
            "error": "Invalid chrom_sizes.txt format",
        }

    if chrom_sizes.empty:
        return {
            "path": db_path,
            "is_db": False,
            "error": "Invalid chrom_sizes.txt format",
        }

    is_indexed = (
        os.path.exists(os.path.join(db_path, "seq", "genome.idx"))
        and os.path.exists(os.path.join(db_path, "seq", "genome.seq"))
    )

    return {
        "path": db_path,
        "is_db": True,
        "format": "indexed" if is_indexed else "per-chromosome",
        "num_chromosomes": int(len(chrom_sizes)),
        "genome_size": int(chrom_sizes["size"].sum()),
        "chromosomes": chrom_sizes,
    }
