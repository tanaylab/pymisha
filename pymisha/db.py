"""Database initialization and example DB helpers."""

import atexit
import copy
import contextlib
import os
import shutil
import struct
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


def _read_chrom_sizes_rows(chrom_sizes_path):
    rows = []
    with open(chrom_sizes_path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid chrom_sizes.txt format at line {line_num}: expected 2 columns"
                )
            chrom = parts[0].strip()
            if not chrom:
                raise ValueError(
                    f"Invalid chrom_sizes.txt format at line {line_num}: empty chromosome name"
                )
            try:
                size = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid chrom_sizes.txt format at line {line_num}: invalid size"
                ) from exc
            if size < 0:
                raise ValueError(
                    f"Invalid chrom_sizes.txt format at line {line_num}: size must be non-negative"
                )
            rows.append((chrom, size))
    return rows


def _resolve_seq_file(seq_dir, chrom):
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")

    for name in candidates:
        path = seq_dir / f"{name}.seq"
        if path.exists():
            return path

    tried = ", ".join(f"{candidate}.seq" for candidate in candidates)
    raise FileNotFoundError(
        f"Missing sequence file for chromosome '{chrom}' (tried: {tried})"
    )


def _iter_genome_idx_entries(index_path):
    with open(index_path, "rb") as fh:
        header = fh.read(24)
        if len(header) != 24:
            raise ValueError("Invalid genome.idx header")
        magic, version, num_entries, _checksum = struct.unpack("<8sIIQ", header)
        if magic != b"MISHAIDX":
            raise ValueError("Invalid genome.idx magic")
        if version != 1:
            raise ValueError(f"Unsupported genome.idx version {version}")
        if num_entries > 20000000:
            raise ValueError("Invalid genome.idx entry count")

        entries = []
        for _ in range(num_entries):
            id_and_len = fh.read(6)
            if len(id_and_len) != 6:
                raise ValueError("Truncated genome.idx entry table")
            chrom_id, name_len = struct.unpack("<IH", id_and_len)

            name_bytes = fh.read(name_len)
            if len(name_bytes) != name_len:
                raise ValueError("Truncated genome.idx contig name")

            tail = fh.read(24)
            if len(tail) != 24:
                raise ValueError("Truncated genome.idx entry table")
            offset, length, _reserved = struct.unpack("<QQQ", tail)

            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Invalid UTF-8 contig name in genome.idx") from exc
            entries.append((chrom_id, name, offset, length))

    entries.sort(key=lambda item: item[0])
    return entries


def _stream_wrapped_sequence(out_fh, seq_fh, chrom, length, line_width, chunk_size):
    pending = b""
    remaining = int(length)

    while remaining > 0:
        to_read = min(chunk_size, remaining)
        chunk = seq_fh.read(to_read)
        if len(chunk) != to_read:
            raise IOError(
                f"Failed reading sequence for chromosome {chrom}: "
                f"expected {to_read} bytes, got {len(chunk)}"
            )

        merged = pending + chunk
        full_len = (len(merged) // line_width) * line_width
        if full_len:
            block = merged[:full_len]
            out_fh.write(
                b"\n".join(
                    block[i: i + line_width] for i in range(0, full_len, line_width)
                )
            )
            out_fh.write(b"\n")
        pending = merged[full_len:]
        remaining -= to_read

    if pending:
        out_fh.write(pending)
        out_fh.write(b"\n")


@contextlib.contextmanager
def _temporary_db_root(groot):
    if groot is None:
        _checkroot()
        yield
        return

    root_path = Path(groot).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Database directory does not exist: {groot}")
    target_root = str(root_path.resolve(strict=False))

    old_root = _shared._GROOT
    old_user = _shared._UROOT
    old_gwd = _shared._GWD
    old_datasets = list(_shared._GDATASETS)
    old_vtracks = copy.deepcopy(_shared._VTRACKS)

    old_root_resolved = (
        str(Path(old_root).expanduser().resolve(strict=False))
        if old_root is not None
        else None
    )
    if old_root_resolved == target_root:
        yield
        return

    gdb_init(target_root)
    try:
        yield
    finally:
        if old_root is None:
            gdb_unload()
        else:
            gdb_init(old_root, old_user)
            _shared._GDATASETS = list(old_datasets)
            _shared._pymisha.pm_dbsetdatasets(old_datasets)
            _shared._VTRACKS = old_vtracks
            _shared._GWD = old_gwd


def gdb_export_fasta(
    file=None,
    groot=None,
    line_width=80,
    chunk_size=1000000,
    overwrite=False,
    verbose=False,
):
    """
    Export database genome sequence to a multi-FASTA file.

    Parameters
    ----------
    file : str or os.PathLike
        Output FASTA file path.
    groot : str, optional
        Database root to export. If None, exports the currently active DB.
    line_width : int, default 80
        Number of bases per FASTA line.
    chunk_size : int, default 1000000
        Number of bases to read per I/O chunk.
    overwrite : bool, default False
        If True, replace an existing output file.
    verbose : bool, default False
        If True, print per-chromosome progress.

    Returns
    -------
    str
        Output FASTA path.
    """
    usage = (
        "Usage: gdb_export_fasta(file, groot=None, line_width=80, "
        "chunk_size=1000000, overwrite=False, verbose=False)"
    )
    if file is None or not isinstance(file, (str, os.PathLike)):
        raise ValueError(usage)
    file_str = os.fspath(file)
    if file_str == "":
        raise ValueError(usage)

    try:
        line_width = int(line_width)
    except (TypeError, ValueError) as exc:
        raise ValueError("line_width must be a positive integer") from exc
    if line_width < 1:
        raise ValueError("line_width must be a positive integer")

    try:
        chunk_size = int(chunk_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_size must be a positive integer") from exc
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    out_path = Path(file_str).expanduser()
    out_dir = out_path.parent
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_path}. Use overwrite=True to replace it."
        )

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix="pymisha_export_",
        suffix=".fasta",
        dir=str(out_dir),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with _temporary_db_root(groot):
            db_root = Path(_shared._GROOT).expanduser().resolve(strict=False)
            seq_dir = db_root / "seq"
            if not seq_dir.is_dir():
                raise FileNotFoundError(f"seq directory does not exist: {seq_dir}")

            genome_idx = seq_dir / "genome.idx"
            genome_seq = seq_dir / "genome.seq"
            chrom_sizes_path = db_root / "chrom_sizes.txt"
            if not chrom_sizes_path.exists():
                raise FileNotFoundError(
                    f"chrom_sizes.txt not found: {chrom_sizes_path}"
                )

            with open(tmp_path, "wb") as out_fh:
                if genome_idx.exists() and genome_seq.exists():
                    entries = _iter_genome_idx_entries(genome_idx)
                    if not entries:
                        raise ValueError("No chromosomes found in the database")
                    with open(genome_seq, "rb") as seq_fh:
                        for _chrom_id, chrom, offset, length in entries:
                            if verbose:
                                print(f"Exporting {chrom} ({length} bp)")
                            out_fh.write(f">{chrom}\n".encode("utf-8"))
                            seq_fh.seek(offset)
                            _stream_wrapped_sequence(
                                out_fh,
                                seq_fh,
                                chrom,
                                length,
                                line_width,
                                chunk_size,
                            )
                else:
                    chrom_rows = _read_chrom_sizes_rows(chrom_sizes_path)
                    if not chrom_rows:
                        raise ValueError("No chromosomes found in the database")
                    for chrom, length in chrom_rows:
                        seq_path = _resolve_seq_file(seq_dir, chrom)
                        if verbose:
                            print(f"Exporting {chrom} ({length} bp)")
                        out_fh.write(f">{chrom}\n".encode("utf-8"))
                        with open(seq_path, "rb") as seq_fh:
                            _stream_wrapped_sequence(
                                out_fh,
                                seq_fh,
                                chrom,
                                length,
                                line_width,
                                chunk_size,
                            )

        os.replace(tmp_path, out_path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise

    return str(out_path)
