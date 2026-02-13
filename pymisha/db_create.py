"""Database creation from FASTA files (gdb_create)."""

import copy
import gzip
import hashlib
import os
import re
import shutil
import struct
import tarfile
import tempfile
import warnings
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib import request as _urlrequest
from urllib.parse import urlparse as _urlparse

import pandas as pd

from ._crc64 import (
    crc64_finalize as _crc64_finalize,
)
from ._crc64 import (
    crc64_incremental as _crc64_compute_incremental,
)
from ._crc64 import (
    crc64_init as _crc64_init,
)


def _sanitize_fasta_header(header):
    """Sanitize FASTA header to extract clean contig name.

    Mirrors the C++ sanitize_fasta_header logic from GenomeSeqMultiImport.cpp.
    """
    clean = header
    if clean.startswith(">"):
        clean = clean[1:]

    clean = clean.strip()

    # Extract first token (before whitespace)
    parts = clean.split(None, 1)
    clean = parts[0] if parts else ""

    # Handle pipe-delimited headers (e.g., "gi|12345|ref|NC_000001.1|")
    if "|" in clean:
        segments = clean.split("|")
        # Filter out empty segments
        segments = [s for s in segments if s]
        if segments:
            # Use last non-empty segment
            last = segments[-1]
            clean = last

    # Replace problematic characters with underscore
    clean = re.sub(r"[^a-zA-Z0-9_.\-]", "_", clean)

    if not clean:
        clean = "contig"

    return clean


def _compute_index_checksum(entries):
    """Compute CRC64 checksum for index entries (matches C++ compute_index_checksum)."""
    crc = _crc64_init()

    for entry in entries:
        chromid_bytes = struct.pack("<I", entry["chromid"])
        crc = _crc64_compute_incremental(crc, chromid_bytes)

        if entry["name"]:
            crc = _crc64_compute_incremental(crc, entry["name"].encode("utf-8"))

        offset_bytes = struct.pack("<Q", entry["offset"])
        crc = _crc64_compute_incremental(crc, offset_bytes)

        length_bytes = struct.pack("<Q", entry["length"])
        crc = _crc64_compute_incremental(crc, length_bytes)

    return _crc64_finalize(crc)


def _write_index_file(index_path, entries):
    """Write genome.idx file in MISHAIDX format."""
    checksum = _compute_index_checksum(entries)

    with open(index_path, "wb") as f:
        # Magic header
        f.write(b"MISHAIDX")

        # Version
        f.write(struct.pack("<I", 1))

        # Number of contigs
        f.write(struct.pack("<I", len(entries)))

        # Checksum
        f.write(struct.pack("<Q", checksum))

        # Contig entries
        for entry in entries:
            f.write(struct.pack("<I", entry["chromid"]))

            name_bytes = entry["name"].encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)

            f.write(struct.pack("<Q", entry["offset"]))
            f.write(struct.pack("<Q", entry["length"]))
            f.write(struct.pack("<Q", 0))  # reserved


def _open_fasta(path):
    """Open a FASTA file, handling gzip compression."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _download_file(url, dst_path):
    """Download a URL to a local path."""
    parsed = _urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Unsupported download URL scheme: {parsed.scheme!r}")

    max_bytes = 5 * 1024 * 1024 * 1024  # 5GB safety cap
    total = 0
    req = _urlrequest.Request(url, method="GET")
    with _urlrequest.urlopen(req, timeout=60) as resp, open(dst_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Download exceeds safety cap ({max_bytes} bytes): {url}")
            out.write(chunk)


def _download_text(url):
    parsed = _urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Unsupported download URL scheme: {parsed.scheme!r}")
    req = _urlrequest.Request(url, method="GET")
    with _urlrequest.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sha256_text(text):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        if re.fullmatch(r"[0-9a-fA-F]{64}", token):
            return token.lower()
    raise ValueError("Could not parse SHA256 checksum text")


def _safe_extract_tar(archive_path, dest_dir):
    """Extract a tar archive while preventing path traversal."""
    dest = Path(dest_dir).resolve(strict=False)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"Unsupported tar member type: {member.name}")
            member_path = (dest / member.name).resolve(strict=False)
            if dest != member_path and dest not in member_path.parents:
                raise ValueError(f"Unsafe path in tar archive: {member.name}")
        try:
            tf.extractall(path=dest, filter="data")
        except TypeError:
            for member in tf.getmembers():
                tf.extract(member, path=dest)


def _parse_fasta_files(fasta_paths):
    """Parse one or more FASTA files and return contig entries with sequences.

    Each entry contains: chromid, name, offset, length, _seq.
    Offset is temporary and recalculated after contig sorting.
    """
    entries = []
    current_offset = 0
    contig_index = -1
    in_sequence = False
    cur_chunks = []
    cur_length = 0

    def finalize_contig():
        nonlocal cur_chunks, cur_length
        if in_sequence and contig_index >= 0:
            entries[-1]["length"] = cur_length
            entries[-1]["_seq"] = b"".join(cur_chunks)
        cur_chunks = []
        cur_length = 0

    for fasta_path in fasta_paths:
        with _open_fasta(fasta_path) as f:
            for line in f:
                line = line.rstrip("\n\r")
                if not line:
                    continue

                if line.startswith(">"):
                    # Finalize previous contig.
                    finalize_contig()

                    # Start new contig
                    contig_index += 1
                    name = _sanitize_fasta_header(line)

                    entries.append({
                        "chromid": contig_index,
                        "name": name,
                        "offset": current_offset,
                        "length": 0,
                        "_seq": b"",
                    })
                    in_sequence = True

                elif line.startswith(";"):
                    continue

                elif in_sequence:
                    # Extract only alphabetic chars and dashes
                    seq = bytearray()
                    for ch in line:
                        if ch.isalpha() or ch == "-":
                            seq.append(ord(ch))
                        elif not ch.isspace():
                            raise ValueError(
                                f"Invalid character '{ch}' in FASTA sequence"
                            )
                    if seq:
                        seq_bytes = bytes(seq)
                        cur_chunks.append(seq_bytes)
                        cur_length += len(seq_bytes)
                        current_offset += len(seq)

    finalize_contig()
    return entries


def _read_chrom_sizes_rows(chrom_sizes_path):
    """Read chrom_sizes.txt preserving row order."""
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
    if not rows:
        raise ValueError("chrom_sizes.txt is empty")
    return rows


def _resolve_seq_file(seq_dir, chrom):
    """Resolve a per-chromosome .seq file path with chr-prefix fallback."""
    candidates = [chrom]
    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")

    for name in candidates:
        path = seq_dir / f"{name}.seq"
        if path.exists():
            return path, name

    raise FileNotFoundError(
        f"Missing sequence file for chromosome '{chrom}' "
        f"(tried: {', '.join([f'{c}.seq' for c in candidates])})"
    )


def _copy_binary_file(src_path, dst_fh, chunk_size):
    copied = 0
    with open(src_path, "rb") as src_fh:
        while True:
            chunk = src_fh.read(chunk_size)
            if not chunk:
                break
            dst_fh.write(chunk)
            copied += len(chunk)
    return copied


def _validate_indexed_genome(entries, seq_files, genome_seq_path, chunk_size):
    expected_total = sum(entry["length"] for entry in entries)
    actual_total = genome_seq_path.stat().st_size
    if expected_total != actual_total:
        raise ValueError(
            "Validation failed: genome.seq size mismatch "
            f"(expected {expected_total}, got {actual_total})"
        )

    with open(genome_seq_path, "rb") as merged_fh:
        for entry, seq_file in zip(entries, seq_files, strict=False):
            merged_fh.seek(entry["offset"])
            remaining = entry["length"]
            with open(seq_file, "rb") as src_fh:
                while remaining > 0:
                    n = min(chunk_size, remaining)
                    merged_chunk = merged_fh.read(n)
                    src_chunk = src_fh.read(n)
                    if merged_chunk != src_chunk:
                        raise ValueError(
                            f"Validation failed for chromosome '{entry['name']}'"
                        )
                    remaining -= n


def _interval_set_is_2d(intervals_dir):
    for entry in intervals_dir.iterdir():
        if entry.name in {
            ".meta",
            "intervals.idx",
            "intervals.dat",
            "intervals2d.idx",
            "intervals2d.dat",
        }:
            continue
        if entry.is_dir():
            continue
        if "-" in entry.name:
            return True
    return False


def _convert_all_tracks_to_indexed(groot, verbose=False):
    from .tracks import gtrack_convert_to_indexed, gtrack_info, gtrack_ls

    converted = 0
    failed = 0
    tracks = gtrack_ls() or []

    for track in tracks:
        track_dir = Path(groot) / "tracks" / f"{track.replace('.', '/')}.track"
        if not track_dir.is_dir():
            continue
        if (track_dir / "track.idx").exists():
            continue

        try:
            info = gtrack_info(track)
        except Exception as exc:
            warnings.warn(f"Skipping track '{track}': failed to read info ({exc})", stacklevel=2)
            failed += 1
            continue

        track_type = info.get("type")
        if track_type not in {"dense", "sparse", "array"}:
            continue

        try:
            gtrack_convert_to_indexed(track, remove_old=False)
            converted += 1
        except Exception as exc:
            warnings.warn(f"Failed to convert track '{track}': {exc}", stacklevel=2)
            failed += 1

    if verbose:
        print(f"Converted tracks to indexed format: {converted} converted, {failed} failed")


def _convert_all_intervals_to_indexed(groot, remove_old_files=False, verbose=False):
    from .intervals import (
        gintervals_2d_convert_to_indexed,
        gintervals_convert_to_indexed,
        gintervals_ls,
    )

    converted = 0
    failed = 0
    interval_sets = gintervals_ls() or []

    for intervals_set in interval_sets:
        intervals_dir = Path(groot) / "tracks" / f"{intervals_set.replace('.', '/')}.interv"
        if not intervals_dir.is_dir():
            continue

        try:
            if _interval_set_is_2d(intervals_dir):
                if (intervals_dir / "intervals2d.idx").exists():
                    continue
                gintervals_2d_convert_to_indexed(
                    intervals_set,
                    remove_old=bool(remove_old_files),
                    force=False,
                )
            else:
                if (intervals_dir / "intervals.idx").exists():
                    continue
                gintervals_convert_to_indexed(
                    intervals_set,
                    remove_old=bool(remove_old_files),
                    force=False,
                )
            converted += 1
        except Exception as exc:
            warnings.warn(f"Failed to convert intervals set '{intervals_set}': {exc}", stacklevel=2)
            failed += 1

    if verbose:
        print(f"Converted interval sets to indexed format: {converted} converted, {failed} failed")


@contextmanager
def _db_conversion_context(groot):
    from . import _shared
    from .db import gdb_init, gdb_reload, gdb_unload

    target_root = str(Path(groot).expanduser().resolve(strict=False))
    old_root = _shared._GROOT
    old_user = _shared._UROOT
    old_datasets = list(_shared._GDATASETS)
    old_vtracks = copy.deepcopy(_shared._VTRACKS)
    old_root_resolved = (
        str(Path(old_root).expanduser().resolve(strict=False))
        if old_root is not None else None
    )

    switched = old_root_resolved != target_root
    if switched:
        gdb_init(target_root)
    else:
        gdb_reload()

    try:
        yield
    finally:
        if switched:
            if old_root is None:
                gdb_unload()
            else:
                gdb_init(old_root, old_user)
                _shared._GDATASETS = list(old_datasets)
                _shared._pymisha.pm_dbsetdatasets(old_datasets)
                _shared._VTRACKS = old_vtracks


def gdb_create_linked(path, parent):
    """
    Create a linked database that reuses sequence data from a parent DB.

    Creates a new DB root with a writable ``tracks/`` directory and symlinks
    to the parent's ``seq/`` directory and ``chrom_sizes.txt`` file.

    Parameters
    ----------
    path : str
        Path for the new linked DB.
    parent : str
        Path to parent DB root.

    Returns
    -------
    bool
        ``True`` on success.

    Raises
    ------
    FileNotFoundError
        If the parent database directory does not exist or is missing
        required files (``chrom_sizes.txt``, ``seq/``).
    FileExistsError
        If the target path already exists.

    See Also
    --------
    gdb_create : Create a new database from FASTA files.
    gdataset_load : Load a dataset into the namespace.
    gdataset_ls : List loaded datasets.

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.gdb_create_linked("~/my_tracks", parent="/shared/genomics/hg38")  # doctest: +SKIP
    True
    """
    if path is None or parent is None:
        raise ValueError("Usage: gdb_create_linked(path, parent)")

    parent_path = Path(parent).expanduser().resolve(strict=False)
    target_path = Path(path).expanduser()

    if not parent_path.exists():
        raise FileNotFoundError(f"Parent database directory does not exist: {parent}")

    parent_chrom_sizes = parent_path / "chrom_sizes.txt"
    parent_seq = parent_path / "seq"
    if not parent_chrom_sizes.exists():
        raise FileNotFoundError("Parent database missing chrom_sizes.txt")
    if not parent_seq.is_dir():
        raise FileNotFoundError("Parent database missing seq/ directory")

    if target_path.exists():
        raise FileExistsError(f"Directory already exists: {target_path}")

    try:
        target_path.mkdir(parents=True)
        (target_path / "tracks").mkdir()

        os.symlink(str(parent_chrom_sizes), str(target_path / "chrom_sizes.txt"))
        os.symlink(str(parent_seq), str(target_path / "seq"))

        parent_pssms = parent_path / "pssms"
        if parent_pssms.exists():
            os.symlink(str(parent_pssms), str(target_path / "pssms"))
        else:
            (target_path / "pssms").mkdir()
    except Exception:
        shutil.rmtree(target_path, ignore_errors=True)
        raise

    return True


def gdb_convert_to_indexed(
    groot=None,
    remove_old_files=False,
    force=False,
    validate=True,
    convert_tracks=False,
    convert_intervals=False,
    verbose=False,
    chunk_size=104857600,
):
    """
    Convert a per-chromosome database to indexed genome format.

    Parameters
    ----------
    groot : str, optional
        Database root. If None, uses currently active DB.
    remove_old_files : bool, default False
        If True, remove old per-chromosome ``*.seq`` files after conversion.
    force : bool, default False
        Kept for parity with R API. Ignored in non-interactive Python flow.
    validate : bool, default True
        If True, validates converted ``genome.seq`` against source files.
    convert_tracks : bool, default False
        If True, converts all eligible tracks to indexed format.
    convert_intervals : bool, default False
        If True, converts all eligible interval sets to indexed format.
    verbose : bool, default False
        If True, prints conversion progress.
    chunk_size : int, default 104857600
        I/O chunk size for reading sequence files.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``chunk_size`` is not positive, or no database is active and
        ``groot`` is not specified.
    FileNotFoundError
        If the database directory, ``seq/`` directory, or
        ``chrom_sizes.txt`` does not exist.

    See Also
    --------
    gdb_create : Create a new database from FASTA files.
    gdb_init : Initialize a database connection.

    Examples
    --------
    Convert the currently active database to indexed format:

    >>> import pymisha as pm
    >>> pm.gdb_convert_to_indexed(groot="/path/to/mydb")  # doctest: +SKIP

    Convert a specific database with full options:

    >>> pm.gdb_convert_to_indexed(  # doctest: +SKIP
    ...     groot="/path/to/mydb",
    ...     convert_tracks=True,
    ...     convert_intervals=True,
    ...     remove_old_files=True,
    ...     verbose=True,
    ... )
    """
    from . import _shared

    del force  # Non-interactive Python API always proceeds.

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if groot is None:
        if _shared._GROOT is None:
            raise ValueError(
                "No database is currently active. Call gdb_init() or pass groot."
            )
        groot = _shared._GROOT

    db_root = Path(groot).expanduser().resolve(strict=False)
    if not db_root.exists():
        raise FileNotFoundError(f"Database directory does not exist: {db_root}")

    seq_dir = db_root / "seq"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"seq directory does not exist: {seq_dir}")

    chrom_sizes_path = db_root / "chrom_sizes.txt"
    if not chrom_sizes_path.exists():
        raise FileNotFoundError(f"chrom_sizes.txt not found: {chrom_sizes_path}")

    genome_idx = seq_dir / "genome.idx"
    genome_seq = seq_dir / "genome.seq"
    already_indexed = genome_idx.exists() and genome_seq.exists()

    if already_indexed and not convert_tracks and not convert_intervals:
        if verbose:
            print("Database is already in indexed format.")
        return

    if not already_indexed:
        if verbose:
            print(f"Converting database to indexed format: {db_root}")

        chrom_rows = _read_chrom_sizes_rows(chrom_sizes_path)
        seq_tmp = seq_dir / "genome.seq.tmp"
        idx_tmp = seq_dir / "genome.idx.tmp"
        chrom_sizes_tmp = db_root / "chrom_sizes.txt.tmp"
        entries = []
        source_seq_files = []
        offset = 0

        try:
            with open(seq_tmp, "wb") as merged_fh:
                for chrom, expected_size in chrom_rows:
                    seq_path, resolved_name = _resolve_seq_file(seq_dir, chrom)
                    seq_len = _copy_binary_file(seq_path, merged_fh, chunk_size)

                    if seq_len != expected_size:
                        warnings.warn(
                            f"Size mismatch for {chrom}: chrom_sizes={expected_size}, "
                            f"seq_file={seq_len}. Using sequence file size.",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                    entries.append({
                        "chromid": len(entries),
                        "name": resolved_name,
                        "offset": offset,
                        "length": seq_len,
                    })
                    source_seq_files.append(seq_path)
                    offset += seq_len

            _write_index_file(str(idx_tmp), entries)

            if validate:
                if verbose:
                    print("Validating conversion...")
                _validate_indexed_genome(entries, source_seq_files, seq_tmp, chunk_size)

            with open(chrom_sizes_tmp, "w", encoding="utf-8") as out_fh:
                for entry in entries:
                    out_fh.write(f"{entry['name']}\t{entry['length']}\n")

            os.replace(seq_tmp, genome_seq)
            os.replace(idx_tmp, genome_idx)
            os.replace(chrom_sizes_tmp, chrom_sizes_path)
        except Exception:
            for path in (seq_tmp, idx_tmp, chrom_sizes_tmp):
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
            raise

        if remove_old_files:
            for seq_path in dict.fromkeys(source_seq_files):
                with suppress(FileNotFoundError):
                    seq_path.unlink()

    if convert_tracks or convert_intervals:
        if verbose:
            print("Converting tracks/intervals to indexed format...")
        with _db_conversion_context(db_root):
            if convert_tracks:
                _convert_all_tracks_to_indexed(db_root, verbose=verbose)
            if convert_intervals:
                _convert_all_intervals_to_indexed(
                    db_root,
                    remove_old_files=remove_old_files,
                    verbose=verbose,
                )

    return


def gdb_create(groot, fasta, genes_file=None, annots_file=None,
               annots_names=None, db_format="indexed", verbose=False, **kwargs):
    """
    Create a new Genomic Database from FASTA file(s).

    Creates the directory structure, imports sequences, and writes
    the chromosome sizes file. Two formats are supported:

    - ``"indexed"`` (default): Single ``genome.seq`` + ``genome.idx``.
      Recommended for genomes with many contigs.
    - ``"per-chromosome"``: Separate ``.seq`` file per contig in the
      ``seq/`` directory.

    Parameters
    ----------
    groot : str
        Path for the new database root directory.
    fasta : str or list of str
        Path(s) to FASTA file(s). Gzipped files (.fa.gz) are supported.
    genes_file : str, optional
        Path to genes annotation file. Not yet implemented.
    annots_file : str, optional
        Path to annotations file. Not yet implemented.
    annots_names : list of str, optional
        Names for annotations. Not yet implemented.
    db_format : str, default "indexed"
        Database format: ``"indexed"`` or ``"per-chromosome"``.
    format : str, optional
        Backward-compatible alias for ``db_format``.
    verbose : bool, default False
        If True, print progress messages.

    Returns
    -------
    DataFrame
        DataFrame with columns ``name`` (contig name) and ``size``
        (contig length in bases).

    Raises
    ------
    FileExistsError
        If the target directory already exists.
    FileNotFoundError
        If a FASTA file does not exist.
    ValueError
        If no contigs are found, duplicate contig names are detected,
        or an unsupported format is specified.

    See Also
    --------
    gdb_init : Initialize a database connection.
    gdb_reload : Reload the current database.
    gdb_create_genome : Download and initialize a prebuilt genome.
    gdb_convert_to_indexed : Convert per-chromosome format to indexed.

    Examples
    --------
    Create a database from a single FASTA file:

    >>> import pymisha as pm
    >>> contigs = pm.gdb_create("/tmp/mydb", "genome.fa.gz")  # doctest: +SKIP

    Create from multiple FASTA files:

    >>> pm.gdb_create("/tmp/mydb", ["chr1.fa", "chr2.fa"], verbose=True)  # doctest: +SKIP

    Create a per-chromosome database:

    >>> pm.gdb_create("/tmp/mydb", "genome.fa", db_format="per-chromosome")  # doctest: +SKIP
    """
    if "format" in kwargs:
        if db_format != "indexed":
            raise ValueError("Specify only one of 'db_format' or 'format'")
        db_format = kwargs.pop("format")
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {bad}")

    groot = Path(groot)

    if groot.exists():
        raise FileExistsError(f"Directory already exists: {groot}")

    if db_format not in ("indexed", "per-chromosome"):
        raise ValueError(
            f"db_format must be 'indexed' or 'per-chromosome', got '{db_format}'"
        )

    # Normalize fasta argument
    if isinstance(fasta, str):
        fasta_paths = [fasta]
    elif isinstance(fasta, list | tuple):
        fasta_paths = list(fasta)
    else:
        raise TypeError(f"fasta must be a string or list, got {type(fasta)}")

    # Validate FASTA files exist
    for fp in fasta_paths:
        if not Path(fp).exists():
            raise FileNotFoundError(f"FASTA file not found: {fp}")

    # Parse FASTA
    entries = _parse_fasta_files(fasta_paths)

    if not entries:
        raise ValueError("No contigs found in FASTA file(s)")

    # Check for duplicate names
    names = [e["name"] for e in entries]
    seen = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        first = next(iter(dupes))
        raise ValueError(
            f"Duplicate contig name '{first}' after sanitization "
            f"({dupes[first]} occurrences)"
        )

    # Sort entries alphabetically by name (default behavior, matches C++)
    sorted_entries = sorted(entries, key=lambda e: e["name"])
    # Reassign chromids to match sorted order
    for i, entry in enumerate(sorted_entries):
        entry["chromid"] = i

    # Create directory structure
    groot.mkdir(parents=True)
    (groot / "seq").mkdir()
    (groot / "tracks").mkdir()
    (groot / "pssms").mkdir()

    if db_format == "indexed":
        # Indexed format: single genome.seq + genome.idx
        # Recalculate offsets and write genome.seq in sorted order without
        # materializing an additional full-genome byte string in memory.
        genome_seq_path = groot / "seq" / "genome.seq"
        offset = 0
        with open(genome_seq_path, "wb") as seq_fh:
            for entry in sorted_entries:
                entry["offset"] = offset
                seq_bytes = entry.pop("_seq", b"")
                seq_fh.write(seq_bytes)
                offset += entry["length"]

        # Write genome.idx
        _write_index_file(str(groot / "seq" / "genome.idx"), sorted_entries)
    else:
        # Per-chromosome format: one .seq file per contig
        offset = 0
        for entry in sorted_entries:
            entry["offset"] = offset
            seq_bytes = entry.pop("_seq", b"")
            seq_path = groot / "seq" / f"{entry['name']}.seq"
            with open(seq_path, "wb") as seq_fh:
                seq_fh.write(seq_bytes)
            offset += entry["length"]

    # Write chrom_sizes.txt
    with open(groot / "chrom_sizes.txt", "w") as f:
        for entry in sorted_entries:
            f.write(f"{entry['name']}\t{entry['length']}\n")

    if verbose:
        print(f"Created database at {groot}")
        print(f"  {len(sorted_entries)} contigs, {offset} total bases")

    # Return contig info
    return pd.DataFrame({
        "name": [e["name"] for e in sorted_entries],
        "size": [e["length"] for e in sorted_entries],
    })


def gdb_create_genome(genome, path=None, tmpdir=None, verify_checksum=True):
    """
    Download and initialize a prebuilt genome database.

    Parameters
    ----------
    genome : str
        Genome identifier. Supported values: ``mm9``, ``mm10``, ``mm39``,
        ``hg19``, ``hg38``.
    path : str, optional
        Directory to extract into. Defaults to current working directory.
    tmpdir : str, optional
        Directory to store the temporary downloaded archive. Defaults to
        ``tempfile.gettempdir()``.
    verify_checksum : bool, default True
        If True, download and verify the archive SHA256 checksum from
        ``<archive_url>.sha256`` before extraction.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the genome identifier is not supported.
    FileNotFoundError
        If the downloaded archive does not contain the expected directory.

    See Also
    --------
    gdb_create : Create a database from local FASTA files.
    gdb_init : Initialize a database connection.

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.gdb_create_genome("hg38", path="/tmp")  # doctest: +SKIP
    """
    supported_genomes = {"mm9", "mm10", "mm39", "hg19", "hg38"}
    if genome not in supported_genomes:
        available = ", ".join(sorted(supported_genomes))
        raise ValueError(
            f"The genome {genome} is not available yet. "
            f"Available genomes are: {available}"
        )

    base_dir = Path(path if path is not None else os.getcwd()).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tmpdir if tmpdir is not None else tempfile.gettempdir()).expanduser()
    tmp_base.mkdir(parents=True, exist_ok=True)

    fd, archive_name = tempfile.mkstemp(suffix=".tar.gz", dir=str(tmp_base))
    os.close(fd)
    archive_path = Path(archive_name)
    archive_url = f"https://misha-genome.s3.eu-west-1.amazonaws.com/{genome}.tar.gz"

    try:
        _download_file(archive_url, archive_path)
        if verify_checksum:
            checksum_text = _download_text(archive_url + ".sha256")
            expected_sha256 = _parse_sha256_text(checksum_text)
            actual_sha256 = _sha256_file(archive_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Checksum mismatch for {archive_url}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        _safe_extract_tar(archive_path, base_dir)
    finally:
        with suppress(OSError):
            archive_path.unlink(missing_ok=True)

    extracted_root = base_dir / genome
    if not extracted_root.exists():
        raise FileNotFoundError(
            f"Downloaded archive did not contain expected directory: {extracted_root}"
        )

    from .db import gdb_init, gdb_reload
    gdb_init(str(extracted_root))
    gdb_reload()

    return
