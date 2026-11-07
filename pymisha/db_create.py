"""Database creation from FASTA files (gdb_create)."""

from __future__ import annotations

import contextlib
import copy
import gzip
import hashlib
import os
import re
import shutil
import stat
import struct
import tarfile
import tempfile
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any
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
from ._log import get_logger
from ._shared import _with_umask

_logger = get_logger(__name__)


def _sanitize_fasta_header(header: str) -> str:
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


def _compute_index_checksum(entries: list[dict[str, Any]]) -> int:
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


def _write_index_file(index_path: str, entries: list[dict[str, Any]]) -> None:
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


def _open_fasta(path: str | Path) -> IO[str]:
    """Open a FASTA file, handling gzip compression."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _download_file(url: str, dst_path: str | Path) -> None:
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


def _download_text(url: str) -> str:
    parsed = _urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError(f"Unsupported download URL scheme: {parsed.scheme!r}")
    req = _urlrequest.Request(url, method="GET")
    with _urlrequest.urlopen(req, timeout=30) as resp:
        data: bytes = resp.read()
        return data.decode("utf-8", errors="replace")


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sha256_text(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        if re.fullmatch(r"[0-9a-fA-F]{64}", token):
            return token.lower()
    raise ValueError("Could not parse SHA256 checksum text")


def _safe_extract_tar(archive_path: str | Path, dest_dir: str | Path) -> None:
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


def _parse_fasta_files(fasta_paths: list[str]) -> list[dict[str, Any]]:
    """Parse one or more FASTA files and return contig entries with sequences.

    Each entry contains: chromid, name, offset, length, _seq.
    Offset is temporary and recalculated after contig sorting.
    """
    entries: list[dict[str, Any]] = []
    current_offset = 0
    contig_index = -1
    in_sequence = False
    cur_chunks: list[bytes] = []
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


def _read_chrom_sizes_rows(chrom_sizes_path: str | Path) -> list[tuple[str, int]]:
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


def _resolve_seq_file(seq_dir: Path, chrom: str) -> tuple[Path, str]:
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


def _copy_binary_file(src_path: str | Path, dst_fh: IO[bytes], chunk_size: int) -> int:
    copied = 0
    with open(src_path, "rb") as src_fh:
        while True:
            chunk = src_fh.read(chunk_size)
            if not chunk:
                break
            dst_fh.write(chunk)
            copied += len(chunk)
    return copied


def _validate_indexed_genome(
    entries: list[dict[str, Any]],
    seq_files: list[Path],
    genome_seq_path: Path,
    chunk_size: int,
) -> None:
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


def _convert_genome_setup(db_root: Path) -> dict[str, Any]:
    """The chromosomes of a per-chromosome database to convert, as R's
    gdb.convert_to_indexed sets them up: names and sizes by chrom id, as gdb_init gives
    them (the database keeps them, so its indexed tracks and interval sets stay keyed
    right), and their .seq files. Stops for a seq/ or chrom_sizes.txt of another
    database, a chromosome whose .seq file is seq/genome.seq, and a .seq file whose
    length differs from chrom_sizes.txt, before anything is written."""
    from ._shared import _pymisha

    seq_dir = db_root / "seq"
    chrom_sizes_path = db_root / "chrom_sizes.txt"
    # The conversion writes seq/ and rewrites chrom_sizes.txt. In a dataset saved without
    # copy_seq, or a database made by gdb_create_linked(), they are links into another
    # database (a directory with tracks/, seq/ or chrom_sizes.txt besides them), which the
    # conversion would change for every database that shares it. A seq/ or chrom_sizes.txt
    # linked to storage of its own elsewhere is converted there.
    for path in (seq_dir, chrom_sizes_path):
        real = Path(os.path.realpath(path))
        owner = real.parent
        if owner != db_root and any(
            (owner / name).exists() for name in ("tracks", "seq", "chrom_sizes.txt") if owner / name != real
        ):
            raise ValueError(f"{path} is in the database {owner}: convert that database instead.")

    _read_chrom_sizes_rows(chrom_sizes_path)  # validates the file
    names, sizes, _, _ = _pymisha.pm_gdb_chrom_order(str(db_root))
    seq_files = [_resolve_seq_file(seq_dir, name)[0] for name in names]

    # The indexed format's sequence file is seq/genome.seq, so a chromosome whose own file
    # has that name (in any case: macOS file systems ignore it) would lose its sequence
    for name, seq_file in zip(names, seq_files, strict=True):
        if seq_file.name.lower() == "genome.seq":
            raise ValueError(
                f"The sequence file of chromosome {name} is seq/genome.seq, the name of the "
                "indexed format's sequence file; rename the chromosome before converting."
            )
    for seq_file, size in zip(seq_files, sizes, strict=True):
        length = seq_file.stat().st_size
        if length != size:
            raise ValueError(
                f"seq/{seq_file.name} has {length} bytes and chrom_sizes.txt says {size}; "
                "they must agree before converting"
            )
    return {"names": list(names), "sizes": [int(x) for x in sizes], "seq_files": seq_files}


def _keep_copy(target: Path) -> Path:
    """A copy of target under a new name next to it, to put back if the conversion fails
    after replacing it: a hard link, or where that is refused (a file of another user
    under fs.protected_hardlinks) a copy with its group, mode and bytes."""
    fd, name = tempfile.mkstemp(prefix="chrom_sizes.txt.", dir=target.parent)
    os.close(fd)
    backup = Path(name)
    backup.unlink()
    try:
        os.link(target, backup)
        return backup
    except OSError:
        pass
    try:
        shutil.copyfile(target, backup)
        st = target.stat()
        with contextlib.suppress(OSError):
            os.chown(backup, -1, st.st_gid)
        os.chmod(backup, stat.S_IMODE(st.st_mode))
        if backup.read_bytes() != target.read_bytes():
            raise OSError("the copy differs")
    except BaseException as exc:
        backup.unlink(missing_ok=True)
        if isinstance(exc, OSError):
            raise ValueError(f"Failed to keep a copy of {target}") from None
        raise
    return backup


def _convert_genome_to_indexed(
    db_root: Path, setup: dict[str, Any], *, validate: bool, chunk_size: int
) -> None:
    """Writes seq/genome.seq and seq/genome.idx from the .seq files and rewrites
    chrom_sizes.txt in chrom id order (R's gdb.convert_to_indexed). genome.idx is moved
    into place last, after chrom_sizes.txt holds the new order: a database with
    genome.idx and genome.seq counts as converted, so a conversion stopped before that
    leaves a per-chromosome database that the next run converts again. A failure or
    interrupt after chrom_sizes.txt was replaced puts the original back."""
    seq_dir = db_root / "seq"
    genome_seq = seq_dir / "genome.seq"
    genome_idx = seq_dir / "genome.idx"
    seq_tmp = seq_dir / "genome.seq.tmp"
    idx_tmp = seq_dir / "genome.idx.tmp"
    # through a link, where chrom_sizes.txt is stored
    cs_target = Path(os.path.realpath(db_root / "chrom_sizes.txt"))
    names, sizes, seq_files = setup["names"], setup["sizes"], setup["seq_files"]

    # Files of an interrupted conversion: a genome.idx left next to the genome.seq written
    # below would be read with it, so it goes first
    for path in (genome_idx, idx_tmp, genome_seq):
        with contextlib.suppress(OSError):
            path.unlink()

    original = cs_target.stat()
    backup: Path | None = _keep_copy(cs_target)
    cs_tmp: Path | None = None
    replaced = False
    try:
        entries: list[dict[str, Any]] = []
        offset = 0
        with open(seq_tmp, "wb") as merged_fh:
            for chrom, seq_path in zip(names, seq_files, strict=True):
                length = _copy_binary_file(seq_path, merged_fh, chunk_size)
                entries.append({"chromid": len(entries), "name": chrom, "offset": offset, "length": length})
                offset += length
        if seq_tmp.stat().st_size != sum(sizes):
            raise ValueError(
                f"genome.seq has {seq_tmp.stat().st_size} bytes, not the {sum(sizes)} of chrom_sizes.txt"
            )
        _write_index_file(str(idx_tmp), entries)

        if validate:
            _validate_indexed_genome(entries, seq_files, seq_tmp, chunk_size)

        # chrom_sizes.txt in chrom id order, with the original's group and mode
        fd, name = tempfile.mkstemp(prefix="chrom_sizes.txt.", dir=cs_target.parent)
        cs_tmp = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as out_fh:
            for chrom, size in zip(names, sizes, strict=True):
                out_fh.write(f"{chrom}\t{size}\n")
        with contextlib.suppress(OSError):
            os.chown(cs_tmp, -1, original.st_gid)
        os.chmod(cs_tmp, stat.S_IMODE(original.st_mode))

        os.replace(seq_tmp, genome_seq)
        os.replace(cs_tmp, cs_target)
        cs_tmp = None
        replaced = True
        try:
            os.replace(idx_tmp, genome_idx)
        except OSError as exc:
            raise ValueError(f"Failed to move {genome_idx} into place: {exc}") from exc
    except BaseException as exc:
        # genome.idx before genome.seq, since the two count as indexed, and the original
        # chrom_sizes.txt back if it was replaced
        for leftover in (genome_idx, idx_tmp, genome_seq, seq_tmp, cs_tmp):
            if leftover is not None:
                with contextlib.suppress(OSError):
                    leftover.unlink()
        if replaced and backup is not None:
            try:
                os.replace(backup, cs_target)
                backup = None
            except OSError:
                raise ValueError(
                    f"Conversion failed: {exc}; and putting back chrom_sizes.txt failed too: "
                    f"the original is {backup}"
                ) from exc
        if isinstance(exc, Exception):
            raise ValueError(f"Conversion failed: {exc}") from exc
        raise
    finally:
        # kept only when putting the original back failed (the error names it)
        if backup is not None and (not replaced or genome_idx.exists()):
            with contextlib.suppress(OSError):
                backup.unlink()


def _interval_set_is_2d(intervals_dir: Path) -> bool:
    """Whether a big interval set is 2D, by the columns of its .meta (a contig's name may
    have a dash); by its file names (a pair's has a dash) when the .meta cannot be read."""
    from .intervals import _decode_intervals_meta

    try:
        _, zeroline = _decode_intervals_meta(intervals_dir / ".meta")
        return "chrom1" in zeroline.columns
    except Exception:
        _logger.debug("could not read %s; deciding by its file names", intervals_dir / ".meta",
                      exc_info=True)
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


def _convert_one_track_worker(
    args: tuple[str, str],
) -> tuple[str, str, str | None]:
    """Pool worker: convert a single track to indexed format.

    Re-inits the DB inside the worker so fork inheritance doesn't alias
    parent C++ singletons. Returns (track, status, error_msg|None).
    """
    track, groot = args
    try:
        from .db import gdb_init
        from .tracks import (
            gtrack_2d_convert_to_indexed,
            gtrack_convert_to_indexed,
            gtrack_info,
        )

        gdb_init(groot)
        track_dir = Path(groot) / "tracks" / f"{track.replace('.', '/')}.track"
        if not track_dir.is_dir():
            return (track, "skipped", None)
        if (track_dir / "track.idx").exists():
            return (track, "skipped", None)
        try:
            info = gtrack_info(track)
        except Exception as exc:
            # Runs in a forked pool worker: the parent only gets str(exc) back
            # (and warns with it), so the traceback is logged here or nowhere.
            _logger.warning("could not read info for track %r", track, exc_info=True)
            return (track, "failed", f"read info: {exc}")
        track_type = info.get("type")
        if track_type in {"dense", "sparse", "array"}:
            gtrack_convert_to_indexed(track, remove_old=False)
            return (track, "converted_1d", None)
        if track_type in {"rectangles", "points"}:
            gtrack_2d_convert_to_indexed(track, remove_old=True, force=False)
            return (track, "converted_2d", None)
        return (track, "skipped", None)
    except Exception as exc:
        _logger.warning("could not convert track %r to indexed format", track, exc_info=True)
        return (track, "failed", str(exc))


def _convert_all_tracks_to_indexed(
    groot: str | Path, verbose: bool = False, threads: int = 1
) -> tuple[int, int]:
    """Converts the database's tracks; returns the number of tracks it set out to
    rewrite (a failed one may have been written) and of those that failed."""
    import multiprocessing as mp
    import sys

    from .tracks import gtrack_ls

    tracks = gtrack_ls() or []
    if not tracks:
        if verbose:
            print("No tracks to convert.")
        return 0, 0

    args_list = [(t, str(groot)) for t in tracks]
    if threads <= 1 or len(args_list) <= 1 or sys.platform == "win32":
        results = [_convert_one_track_worker(a) for a in args_list]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(threads) as pool:
            results = pool.map(_convert_one_track_worker, args_list)

    # Whole-database layout flip: the workers converted in forked children, so the
    # parent still holds a parsed index for the per-chrom layout of every track it
    # had read. Drop them all rather than reconstruct which ones changed.
    from . import _shared

    _shared._pymisha.pm_clear_dir_caches()

    converted_1d = sum(1 for _, s, _ in results if s == "converted_1d")
    converted_2d = sum(1 for _, s, _ in results if s == "converted_2d")
    failures = [(t, e) for t, s, e in results if s == "failed"]
    for t, e in failures:
        warnings.warn(f"Failed to convert track '{t}': {e}", stacklevel=2)
    if verbose:
        print(
            f"Converted tracks to indexed format: "
            f"{converted_1d} 1D + {converted_2d} 2D converted, "
            f"{len(failures)} failed"
        )
    return converted_1d + converted_2d + len(failures), len(failures)


def _convert_one_intervals_worker(
    args: tuple[str, str, bool],
) -> tuple[str, str, str | None]:
    """Pool worker: convert a single intervals set to indexed format.

    Re-inits the DB inside the worker. Returns (intervals_set, status, error_msg|None).
    """
    intervals_set, groot, remove_old_files = args
    try:
        from .db import gdb_init
        from .intervals import (
            gintervals_2d_convert_to_indexed,
            gintervals_convert_to_indexed,
            gintervals_ls,
        )

        gdb_init(groot)
        # Defensive: re-verify the set exists post-fork.
        if intervals_set not in (gintervals_ls() or []):
            return (intervals_set, "skipped", None)
        intervals_dir = (
            Path(groot) / "tracks" / f"{intervals_set.replace('.', '/')}.interv"
        )
        if not intervals_dir.is_dir():
            return (intervals_set, "skipped", None)
        if _interval_set_is_2d(intervals_dir):
            if (intervals_dir / "intervals2d.idx").exists():
                return (intervals_set, "skipped", None)
            gintervals_2d_convert_to_indexed(
                intervals_set,
                remove_old=remove_old_files,
                force=False,
            )
        else:
            if (intervals_dir / "intervals.idx").exists():
                return (intervals_set, "skipped", None)
            gintervals_convert_to_indexed(
                intervals_set,
                remove_old=remove_old_files,
                force=False,
            )
        return (intervals_set, "converted", None)
    except Exception as exc:
        _logger.warning("could not convert intervals set %r to indexed format", intervals_set,
                        exc_info=True)
        return (intervals_set, "failed", str(exc))


def _convert_all_intervals_to_indexed(
    groot: str | Path,
    remove_old_files: bool = False,
    verbose: bool = False,
    threads: int = 1,
) -> tuple[int, int]:
    """Converts the database's interval sets; returns the number of sets it set out to
    rewrite and of those that failed."""
    import multiprocessing as mp
    import sys

    from .intervals import gintervals_ls

    interval_sets = gintervals_ls() or []
    if not interval_sets:
        if verbose:
            print("No interval sets to convert.")
        return 0, 0

    args_list = [
        (iset, str(groot), bool(remove_old_files)) for iset in interval_sets
    ]
    if threads <= 1 or len(args_list) <= 1 or sys.platform == "win32":
        results = [_convert_one_intervals_worker(a) for a in args_list]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(threads) as pool:
            results = pool.map(_convert_one_intervals_worker, args_list)

    converted = sum(1 for _, s, _ in results if s == "converted")
    failures = [(t, e) for t, s, e in results if s == "failed"]
    for t, e in failures:
        warnings.warn(
            f"Failed to convert intervals set '{t}': {e}", stacklevel=2
        )
    if verbose:
        print(
            f"Converted interval sets to indexed format: "
            f"{converted} converted, {len(failures)} failed"
        )
    return converted + len(failures), len(failures)


@contextmanager
def _db_conversion_context(
    groot: str | Path, state: dict[str, Any] | None = None
) -> Generator[None, None, None]:
    """Loads the database at groot for the track and interval steps of a conversion
    (their workers load it too), and puts the session loaded before back after them:
    its database, datasets, working directory and virtual tracks. A database that no
    longer loads is unloaded, and its error kept in state["restore_error"]."""
    from . import _shared
    from .db import _set_gwd, gdb_init, gdb_unload

    target_root = str(Path(groot).expanduser().resolve(strict=False))
    old_root = _shared._GROOT
    old_user = _shared._UROOT
    old_gwd = _shared._GWD
    old_datasets = list(_shared._GDATASETS)
    old_dataset_chroms = dict(_shared._GDATASET_CHROMS)
    old_vtracks = copy.deepcopy(_shared._VTRACKS)
    old_gvtracks = copy.deepcopy(_shared._GVTRACKS)

    # loaded afresh even when it is the session's database, as the workers load it: its
    # names counted from its tracks/, whatever the working directory, and no datasets (R's
    # gdb.convert_to_indexed calls gdb.init)
    gdb_init(target_root)

    try:
        yield
    finally:
        if old_root is None:
            gdb_unload()
        else:
            try:
                # its warnings are given once the session is back
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gdb_init(old_root, old_user)
            except Exception as exc:
                gdb_unload()
                if state is not None:
                    state["restore_error"] = str(exc)
            else:
                _shared._GDATASETS = list(old_datasets)
                _shared._GDATASET_CHROMS = dict(old_dataset_chroms)
                _shared._pymisha.pm_dbsetdatasets(old_datasets)
                assert old_gwd is not None
                _set_gwd(old_gwd)
                _shared._VTRACKS = old_vtracks
                _shared._GVTRACKS = old_gvtracks


@_with_umask()   # database writes carry misha's permissions
def gdb_create_linked(path: str, parent: str) -> bool:
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


@_with_umask()   # database writes carry misha's permissions
def gdb_convert_to_indexed(
    groot: str | None = None,
    remove_old_files: bool = False,
    force: bool = False,
    validate: bool = True,
    convert_tracks: bool = False,
    convert_intervals: bool = False,
    verbose: bool = False,
    chunk_size: int = 104857600,
    *,
    threads: int | None = None,
) -> None:
    """
    Convert a per-chromosome database to indexed genome format.

    The converted database keeps the chromosome names and order that
    :func:`gdb_init` gives it, whether or not it is the loaded database, so its
    indexed tracks and interval sets stay valid. In a per-chromosome database
    whose chrom_sizes.txt names lack the "chr" prefix of its .seq files, that
    order is the sorted names, and chrom_sizes.txt is rewritten in it. The
    conversion stops, leaving the database as it was, when a .seq file's length
    differs from chrom_sizes.txt, when a chromosome's .seq file is
    seq/genome.seq, and when seq/ or chrom_sizes.txt is in another database
    (a :func:`gdb_create_linked` database, a dataset saved without
    ``copy_seq``): convert that database. A failure or interrupt leaves a
    database the next run converts. The session loaded before is put back; a
    loaded dataset that the conversion changed is unloaded, with a warning.

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
    threads : int or None, optional
        Number of worker processes for per-track / per-intervals
        conversion. ``None`` (default) resolves to
        ``min(os.cpu_count() or 1, 8)``. ``1`` keeps the serial path.
        Values > 1 fork a `multiprocessing.Pool`; per-task failures
        are emitted as warnings rather than aborting. Falls back to
        serial on Windows.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``chunk_size`` is not positive, ``threads`` is not a positive
        integer, or no database is active and ``groot`` is not specified.
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

    if threads is None:
        threads = min(os.cpu_count() or 1, 8)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError(
            f"threads must be a positive integer, got {threads!r}"
        )

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

    # The session loaded before is put back as it was at the end (the track and interval
    # steps load the converted database). A loaded dataset that the conversion changed is
    # unloaded, as its files may be keyed by its own chrom ids now, and a loaded database
    # whose sequence the conversion changed (its seq/ is the converted one, with a
    # chrom_sizes.txt of its own) is unloaded when its chrom ids no longer match the index.
    old_root = _shared._GROOT
    converted = [d for d in _shared._GDATASETS if os.path.realpath(d) == str(db_root)]
    chrom_sizes_md5 = _file_md5(chrom_sizes_path) if converted else None
    state: dict[str, Any] = {"changed": False, "finished": False, "restore_error": None}
    try:
        _gdb_convert_to_indexed_steps(
            db_root, already_indexed, state, remove_old_files=remove_old_files,
            validate=validate, convert_tracks=convert_tracks,
            convert_intervals=convert_intervals, verbose=verbose,
            chunk_size=chunk_size, threads=threads,
        )
    finally:
        if old_root:
            _restore_session_after_conversion(
                old_root, converted, chrom_sizes_path, chrom_sizes_md5, state
            )


def _file_md5(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _restore_session_after_conversion(
    old_root: str, converted: list[str], chrom_sizes_path: Path,
    chrom_sizes_md5: str | None, state: dict[str, Any],
) -> None:
    """R's gdb.convert_to_indexed on.exit: the session loaded before the conversion,
    without the converted database if it was a loaded dataset that the conversion
    changed, and unloaded if it no longer reads its own sequence."""
    from . import _shared
    from .dataset import gdataset_unload
    from .db import gdb_unload

    _shared._pymisha.pm_clear_dir_caches()
    unload = bool(converted) and (
        state["changed"] or not chrom_sizes_path.exists()
        or _file_md5(chrom_sizes_path) != chrom_sizes_md5
    )
    if unload and _shared._GROOT is not None:
        gdataset_unload(converted[0])
    problem: str | None = state.get("restore_error")
    reads_own = _shared._GROOT is not None
    if reads_own:
        genome = _shared._pymisha.pm_intervals_all()
        try:
            _shared._pymisha.pm_gdb_check_genome_idx(
                old_root, [str(c) for c in genome[1]], [int(e) for e in genome[3]]
            )
        except Exception as exc:
            reads_own = False
            problem = str(exc)
    if not reads_own:
        gdb_unload()
    else:
        # as gdb_init checks it now
        try:
            _shared._pymisha.pm_gdb_check_genome_idx(old_root)
        except Exception as exc:
            problem = str(exc)
    # warnings last, once the session is in place
    if unload:
        if state["finished"]:
            warnings.warn(f"{converted[0]} was converted, so it is no longer loaded as a dataset.",
                          stacklevel=3)
        else:
            warnings.warn(
                f"The conversion of {converted[0]} failed and may have changed it, so it is no "
                "longer loaded as a dataset.",
                stacklevel=3,
            )
    if not reads_own:
        warnings.warn(
            f"{old_root}, loaded before the conversion, no longer reads its own sequence: "
            f"{problem} No database is loaded.",
            stacklevel=3,
        )
    elif problem is not None:
        warnings.warn(
            f"{old_root} is loaded as before the conversion, but gsetroot() would not load "
            f"it now: {problem}",
            stacklevel=3,
        )


def _gdb_convert_to_indexed_steps(
    db_root: Path, already_indexed: bool, state: dict[str, Any], *,
    remove_old_files: bool, validate: bool, convert_tracks: bool,
    convert_intervals: bool, verbose: bool, chunk_size: int, threads: int,
) -> None:
    if not already_indexed:
        if verbose:
            print(f"Converting database to indexed format: {db_root}")
        setup = _convert_genome_setup(db_root)
        _convert_genome_to_indexed(db_root, setup, validate=validate, chunk_size=chunk_size)
        state["changed"] = True
        if remove_old_files:
            for seq_path in dict.fromkeys(setup["seq_files"]):
                try:
                    seq_path.unlink()
                except FileNotFoundError:
                    _logger.debug("old sequence file %s was already gone", seq_path, exc_info=True)

    # A step counts as a change from its start, as an error or interrupt partway leaves
    # what it converted; one that had nothing to convert does not.
    failed = 0
    if convert_tracks or convert_intervals:
        if verbose:
            print("Converting tracks/intervals to indexed format...")
        with _db_conversion_context(db_root, state):
            if convert_tracks:
                changed_before = state["changed"]
                state["changed"] = True
                convertible, n_failed = _convert_all_tracks_to_indexed(
                    db_root, verbose=verbose, threads=threads,
                )
                failed += n_failed
                if convertible == 0:
                    state["changed"] = changed_before
            if convert_intervals:
                changed_before = state["changed"]
                state["changed"] = True
                convertible, n_failed = _convert_all_intervals_to_indexed(
                    db_root,
                    remove_old_files=remove_old_files,
                    verbose=verbose,
                    threads=threads,
                )
                failed += n_failed
                if convertible == 0:
                    state["changed"] = changed_before

    state["finished"] = failed == 0


@_with_umask()   # database writes carry misha's permissions
def gdb_create(
    groot: str,
    fasta: str | list[str],
    genes_file: str | None = None,
    annots_file: str | None = None,
    annots_names: list[str] | None = None,
    db_format: str = "indexed",
    verbose: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
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

    groot_path = Path(groot)

    if groot_path.exists():
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
    seen: dict[str, int] = {}
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
    groot_path.mkdir(parents=True)
    (groot_path / "seq").mkdir()
    (groot_path / "tracks").mkdir()
    (groot_path / "pssms").mkdir()

    if db_format == "indexed":
        # Indexed format: single genome.seq + genome.idx
        # Recalculate offsets and write genome.seq in sorted order without
        # materializing an additional full-genome byte string in memory.
        genome_seq_path = groot_path / "seq" / "genome.seq"
        offset = 0
        with open(genome_seq_path, "wb") as seq_fh:
            for entry in sorted_entries:
                entry["offset"] = offset
                seq_bytes = entry.pop("_seq", b"")
                seq_fh.write(seq_bytes)
                offset += entry["length"]

        # Write genome.idx
        _write_index_file(str(groot_path / "seq" / "genome.idx"), sorted_entries)
    else:
        # Per-chromosome format: one .seq file per contig
        offset = 0
        for entry in sorted_entries:
            entry["offset"] = offset
            seq_bytes = entry.pop("_seq", b"")
            seq_path = groot_path / "seq" / f"{entry['name']}.seq"
            with open(seq_path, "wb") as seq_fh:
                seq_fh.write(seq_bytes)
            offset += entry["length"]

    # Write chrom_sizes.txt
    with open(groot_path / "chrom_sizes.txt", "w") as f:
        for entry in sorted_entries:
            f.write(f"{entry['name']}\t{entry['length']}\n")

    if verbose:
        print(f"Created database at {groot_path}")
        print(f"  {len(sorted_entries)} contigs, {offset} total bases")

    # Return contig info
    return pd.DataFrame({
        "name": [e["name"] for e in sorted_entries],
        "size": [e["length"] for e in sorted_entries],
    })


def gdb_create_genome(
    genome: str,
    path: str | None = None,
    tmpdir: str | None = None,
    verify_checksum: bool = True,
) -> None:
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

    dest_dir = str(Path(path if path is not None else os.getcwd()).expanduser())
    _gdb_create_genome_from_s3(
        genome,
        dest_dir,
        verify_checksum=verify_checksum,
        tmpdir=tmpdir,
    )

    from .db import gdb_init, gdb_reload
    gdb_init(str(Path(dest_dir) / genome))
    gdb_reload()
    return


@_with_umask()   # database writes carry misha's permissions
def _gdb_create_genome_from_s3(
    name: str,
    dest_dir: str,
    verbose: bool = False,
    tmpdir: str | None = None,
    verify_checksum: bool = True,
) -> None:
    """Download ``<name>.tar.gz`` from the misha-genome S3 bucket and extract.

    Internal helper shared by :func:`gdb_create_genome` and the
    ``s3`` backend used by ``gdb_build_genome``. Downloads
    ``https://misha-genome.s3.eu-west-1.amazonaws.com/<name>.tar.gz`` to a
    temp file, optionally verifies its SHA256, extracts safely into
    ``dest_dir``, and confirms ``<dest_dir>/<name>`` exists. Does NOT
    initialize misha against the extracted directory; callers are
    responsible for that (or for moving the directory before init).

    Parameters
    ----------
    name : str
        Genome identifier and S3 object key (e.g. ``"hg38"``).
    dest_dir : str
        Directory to extract into. Must already exist or be creatable.
    verbose : bool, default False
        Reserved for future progress output. Currently unused.
    tmpdir : str, optional
        Directory for the temporary archive. Defaults to
        ``tempfile.gettempdir()``.
    verify_checksum : bool, default True
        If True, fetch and verify the archive SHA256 from
        ``<archive_url>.sha256``.
    """
    dest_path = Path(dest_dir).expanduser()
    dest_path.mkdir(parents=True, exist_ok=True)
    tmp_base = Path(tmpdir if tmpdir is not None else tempfile.gettempdir()).expanduser()
    tmp_base.mkdir(parents=True, exist_ok=True)

    fd, archive_name = tempfile.mkstemp(suffix=".tar.gz", dir=str(tmp_base))
    os.close(fd)
    archive_path = Path(archive_name)
    archive_url = f"https://misha-genome.s3.eu-west-1.amazonaws.com/{name}.tar.gz"

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
        _safe_extract_tar(archive_path, dest_path)
    finally:
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            _logger.warning("could not remove the downloaded archive %s", archive_path, exc_info=True)

    extracted_root = dest_path / name
    if not extracted_root.exists():
        raise FileNotFoundError(
            f"Downloaded archive did not contain expected directory: {extracted_root}"
        )
    return
