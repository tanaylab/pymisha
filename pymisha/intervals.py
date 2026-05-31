"""Interval creation and operations."""

from __future__ import annotations

import contextlib as _contextlib
import gzip
import os
import re
import shutil
import struct
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, cast

import pandas as pd

from ._crc64 import (
    crc64_finalize as _crc64_finalize,
)
from ._crc64 import (
    crc64_incremental as _crc64_incremental,
)
from ._crc64 import (
    crc64_init as _crc64_init,
)
from ._db_trash import _gdb_trash
from ._name_validation import validate_dotted_name
from ._shared import (
    CONFIG,
    _checkroot,
    _config_no_mt,
    _df2pymisha,
    _numpy,
    _pandas,
    _preprocess_intervals_iterator,
    _progress_context,
    _pymisha,
    _pymisha2df,
    _remap_interval_ids,
)


def _normalize_chroms(chroms: Any) -> Any:
    if chroms is None:
        return chroms
    if isinstance(chroms, (str, int)):
        chroms = [chroms]
    return list(_pymisha.pm_normalize_chroms(list(chroms)))


def _resolve_intervals(intervals: pd.DataFrame | str) -> pd.DataFrame | str:
    """Transparently load a named interval set if *intervals* is a string."""
    if isinstance(intervals, str) and gintervals_exists(intervals):
        return gintervals_load(intervals)
    return intervals


def gintervals_all() -> pd.DataFrame:
    """
    Return all chromosome intervals (ALLGENOME).

    Returns a DataFrame with one row per chromosome, covering the full
    extent of each chromosome in the current genome database as defined
    by ``chrom_sizes.txt``.

    Returns
    -------
    DataFrame
        Intervals with columns: chrom, start, end.

    See Also
    --------
    gintervals : Create a custom set of 1D intervals.
    gintervals_2d_all : Return 2D intervals covering the whole genome.
    gintervals_from_tuples : Create intervals from a list of tuples.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_all()  # doctest: +SKIP
    """
    _checkroot()
    result = _pymisha.pm_intervals_all()
    return _pymisha2df(result)


def _intervset_path(intervals_set: str) -> Path:
    root = gintervals_dataset(intervals_set)
    if root is None:
        raise ValueError(f"Intervals set '{intervals_set}' does not exist")
    path_part = intervals_set.replace(".", "/")
    return Path(root) / "tracks" / f"{path_part}.interv"


def _decode_r_obj_to_bytes(obj_path: str | Path) -> pd.DataFrame:
    """Decode a single R-serialized object on disk into a DataFrame.

    Used to read legacy bigset chromosome shards. Native reader (no R
    or pyreadr at runtime).
    """
    from ._r_serialize import read as _r_read

    obj = _r_read(obj_path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        # Some shards are wrapped in a single-element dict
        first = next(iter(obj.values()))
        if isinstance(first, pd.DataFrame):
            return first
    raise ValueError(
        f"expected a data.frame in {obj_path}, got {type(obj).__name__}"
    )


def _decode_intervals_meta(meta_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode a legacy bigset ``.meta`` file (an R-serialized
    ``list(stats=..., zeroline=...)``) into two DataFrames.

    Uses :mod:`pymisha._r_serialize` so no Rscript dependency is needed
    at runtime.
    """
    from ._r_serialize import read as _r_read

    obj = _r_read(meta_path)
    if not isinstance(obj, dict) or "stats" not in obj or "zeroline" not in obj:
        raise ValueError(
            f"{meta_path}: expected an R-serialized list with 'stats' and "
            f"'zeroline' entries, got {type(obj).__name__}"
        )
    stats = obj["stats"]
    zeroline = obj["zeroline"]
    if not isinstance(stats, pd.DataFrame) or not isinstance(zeroline, pd.DataFrame):
        raise ValueError(
            f"{meta_path}: 'stats' and 'zeroline' must be data.frames; got "
            f"{type(stats).__name__} / {type(zeroline).__name__}"
        )
    return stats, zeroline


def _intervset_is_bigset(intervals_set: str) -> bool:
    path = _intervset_path(intervals_set)
    return path.exists() and path.is_dir()


def _intervset_is_indexed(path: Path, allow_updates: bool = True) -> bool:
    idx_1d = path / "intervals.idx"
    idx_2d = path / "intervals2d.idx"
    if not idx_1d.exists() and not idx_2d.exists():
        return False
    if not allow_updates:
        return True
    files = {p.name for p in path.iterdir()}
    reserved = {
        "intervals.idx",
        "intervals.dat",
        "intervals2d.idx",
        "intervals2d.dat",
        ".meta",
    }
    return len(files - reserved) == 0


def _intervset_index_paths(path: Path) -> dict[str, Path]:
    return {
        "idx1d": path / "intervals.idx",
        "dat1d": path / "intervals.dat",
        "idx2d": path / "intervals2d.idx",
        "dat2d": path / "intervals2d.dat",
    }


def _load_index_entries_1d(idx_path: Path) -> list[tuple[int, int, int]]:
    with open(idx_path, "rb") as fh:
        header = fh.read(36)
        if len(header) != 36:
            raise ValueError("Invalid intervals.idx header")
        magic, version, num_entries, flags, checksum, reserved = struct.unpack(
            "<8sIIQQI", header
        )
        if magic != b"MISHAI1D":
            raise ValueError("Invalid intervals.idx magic")
        if version != 1:
            raise ValueError(f"Unsupported intervals.idx version {version}")
        if num_entries > 20000000:
            raise ValueError("Invalid intervals.idx entry count")
        if (flags & 0x01) == 0:
            raise ValueError("Unsupported intervals.idx endianness")
        entries = []
        crc = _crc64_init()
        for _ in range(num_entries):
            entry_bytes = fh.read(24)
            if len(entry_bytes) != 24:
                raise ValueError("Truncated intervals.idx entry table")
            chrom_id, offset, length, _reserved = struct.unpack("<IQQI", entry_bytes)
            entries.append((chrom_id, offset, length))
            crc = _crc64_incremental(crc, entry_bytes[:4])
            crc = _crc64_incremental(crc, entry_bytes[4:12])
            crc = _crc64_incremental(crc, entry_bytes[12:20])
        crc = _crc64_finalize(crc)
        if crc != checksum:
            raise ValueError("intervals.idx checksum mismatch")
    return entries


def _load_index_entries_2d(idx_path: Path) -> list[tuple[int, int, int, int]]:
    with open(idx_path, "rb") as fh:
        header = fh.read(40)
        if len(header) != 40:
            raise ValueError("Invalid intervals2d.idx header")
        magic, version, num_entries, flags, checksum, reserved = struct.unpack(
            "<8sIIQQQ", header
        )
        if magic != b"MISHAI2D":
            raise ValueError("Invalid intervals2d.idx magic")
        if version != 1:
            raise ValueError(f"Unsupported intervals2d.idx version {version}")
        if (flags & 0x01) == 0:
            raise ValueError("Unsupported intervals2d.idx endianness")
        entries = []
        crc = _crc64_init()
        for _ in range(num_entries):
            entry_bytes = fh.read(28)
            if len(entry_bytes) != 28:
                raise ValueError("Truncated intervals2d.idx entry table")
            chrom1_id, chrom2_id, offset, length, _reserved = struct.unpack(
                "<IIQQI", entry_bytes
            )
            entries.append((chrom1_id, chrom2_id, offset, length))
            crc = _crc64_incremental(crc, entry_bytes[:4])
            crc = _crc64_incremental(crc, entry_bytes[4:8])
            crc = _crc64_incremental(crc, entry_bytes[8:16])
            crc = _crc64_incremental(crc, entry_bytes[16:24])
        crc = _crc64_finalize(crc)
        if crc != checksum:
            raise ValueError("intervals2d.idx checksum mismatch")
    return entries


_STRAND_CHAR_MAP = {
    "+": 1,
    "-": -1,
    ".": 0,
    "*": 0,
    "": 0,
}


def _normalize_strand_value(value: Any) -> int:
    """Normalize a single strand value to ``-1``, ``0``, or ``1``.

    Accepts numeric ``-1``/``0``/``1`` and the BED-style character
    encodings ``"+"``/``"-"``/``"."``/``"*"``/``""``. Anything else
    raises ``ValueError``.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid strand value {value!r}")
    if isinstance(value, (int, _numpy.integer)):
        if value in (-1, 0, 1):
            return int(value)
        raise ValueError(f"Invalid strand value {value!r}: must be -1, 0, or 1")
    if isinstance(value, float) and not _numpy.isnan(value):
        ivalue = int(value)
        if float(ivalue) == value and ivalue in (-1, 0, 1):
            return ivalue
        raise ValueError(f"Invalid strand value {value!r}: must be -1, 0, or 1")
    if isinstance(value, str):
        if value in _STRAND_CHAR_MAP:
            return _STRAND_CHAR_MAP[value]
        # Allow numeric strings like "1", "-1", "0".
        try:
            ivalue = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid strand value {value!r}: expected '+'/'-'/'.'"
                f"/'*'/'' or -1/0/1"
            ) from exc
        if ivalue in (-1, 0, 1):
            return ivalue
        raise ValueError(f"Invalid strand value {value!r}: must be -1, 0, or 1")
    raise ValueError(f"Invalid strand value {value!r}")


def _normalize_strand_column(values: Any) -> list[int]:
    """Normalize a sequence of strand values to numeric -1/0/1.

    Accepts a list/Series/array of mixed input (numeric, strings,
    pandas Categorical). Returns a list of ``int``.
    """
    if isinstance(values, _pandas.Series):
        if isinstance(values.dtype, _pandas.CategoricalDtype):
            values = values.astype(object)
        values = values.tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)
    return [_normalize_strand_value(v) for v in values]


def gintervals(
    chroms: str | int | list[Any],
    starts: int | list[int] = 0,
    ends: int | list[int] = -1,
    strand: int | str | list[int | str] | None = None,
) -> pd.DataFrame:
    """
    Create a 1D intervals DataFrame.

    Constructs an intervals DataFrame from parallel arrays of chromosome
    names, start coordinates, and end coordinates. Scalar arguments are
    broadcast to match the longest array.

    Parameters
    ----------
    chroms : str, int, or list
        Chromosome names. Can be strings like ``"chr1"`` or integers like ``1``.
    starts : int or list of int, default 0
        Start coordinates (0-based, inclusive).
    ends : int or list of int, default -1
        End coordinates (0-based, exclusive). ``-1`` means full chromosome
        length.
    strand : int, str, or list, optional
        Strand information. Accepts numeric ``-1``/``0``/``1`` or the
        BED-style characters ``"+"``/``"-"``/``"."``/``"*"``/``""``
        (``"."``/``"*"``/``""`` map to ``0``). Output is always numeric.
        Note: this interval convention differs from liftover chain tables,
        where strand fields are encoded as ``0`` (``+``) or ``1`` (``-``).

    Returns
    -------
    DataFrame
        Sorted intervals with columns: chrom, start, end (and optionally
        strand).

    See Also
    --------
    gintervals_all : Return full-chromosome intervals for every chromosome.
    gintervals_2d : Create 2D intervals.
    gintervals_from_tuples : Create intervals from a list of tuples.
    gintervals_from_strings : Create intervals from region strings.
    gintervals_from_bed : Create intervals from a BED file.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    The following calls produce equivalent results:

    >>> pm.gintervals(1)  # doctest: +SKIP
    >>> pm.gintervals("1")  # doctest: +SKIP
    >>> pm.gintervals("chrX")  # doctest: +SKIP

    Specify start coordinates:

    >>> pm.gintervals(1, 1000)  # doctest: +SKIP

    Multiple intervals with broadcast:

    >>> pm.gintervals(["chr2", "chrX"], 10, [3000, 5000])  # doctest: +SKIP
    """
    result_chroms, result_starts, result_ends = _make_1d_intervals(chroms, starts, ends)

    result_strands = None
    if strand is not None:
        if isinstance(strand, (int, str, _numpy.integer)):
            strand = [strand]
        strand = list(strand)
        n = len(result_chroms)
        if len(strand) == 1:
            strand = strand * n
        if len(strand) != n:
            raise ValueError("strand must have the same length as other arguments")

        result_strands = _normalize_strand_column(strand)

    df = _pandas.DataFrame({
        'chrom': result_chroms,
        'start': result_starts,
        'end': result_ends
    })

    if result_strands is not None:
        df['strand'] = result_strands

    return df.sort_values(['chrom', 'start']).reset_index(drop=True)


def _make_1d_intervals(
    chroms: Any,
    starts: Any,
    ends: Any,
) -> tuple[list[str], list[int], list[int]]:
    """Shared helper: validate and expand 1D interval args, return lists."""
    _checkroot()

    if isinstance(chroms, (str, int)):
        chroms = [chroms]
    if isinstance(starts, (int, float)):
        starts = [starts]
    if isinstance(ends, (int, float)):
        ends = [ends]

    chroms = list(chroms)
    starts = [int(s) for s in starts]
    ends = [int(e) for e in ends]

    n = max(len(chroms), len(starts), len(ends))

    # Recycle each argument to the common length, as R does (a shorter vector is
    # repeated when its length divides the longest).
    def _recycle(vec: list, label: str) -> list:
        if len(vec) == n:
            return vec
        if len(vec) and n % len(vec) == 0:
            return vec * (n // len(vec))
        raise ValueError("chroms, starts, and ends must have the same length")

    chroms = _recycle(chroms, "chroms")
    starts = _recycle(starts, "starts")
    ends = _recycle(ends, "ends")

    chroms = _normalize_chroms(chroms)

    all_intervals = gintervals_all()
    chrom_sizes = dict(
        zip(
            all_intervals["chrom"].astype(str).tolist(),
            all_intervals["end"].astype(int).tolist(), strict=False,
        )
    )

    result_chroms = []
    result_starts = []
    result_ends = []

    for i in range(n):
        chrom = chroms[i]
        start = starts[i]
        end = ends[i]

        if chrom not in chrom_sizes:
            raise ValueError(f"Unknown chromosome: {chrom}")

        chrom_size = chrom_sizes[chrom]
        if end == -1:
            end = chrom_size
        if start < 0:
            raise ValueError(f"Invalid interval ({chrom}, {start}, {end}): start must be >= 0")
        if start >= end:
            raise ValueError(f"Invalid interval ({chrom}, {start}, {end}): start must be < end")
        if end > chrom_size:
            raise ValueError(f"Invalid interval ({chrom}, {start}, {end}): end exceeds chromosome size ({chrom_size})")

        result_chroms.append(chrom)
        result_starts.append(start)
        result_ends.append(end)

    return result_chroms, result_starts, result_ends


def gintervals_2d(
    chroms1: str | int | list[Any],
    starts1: int | list[int] = 0,
    ends1: int | list[int] = -1,
    chroms2: str | int | list[Any] | None = None,
    starts2: int | list[int] = 0,
    ends2: int | list[int] = -1,
) -> pd.DataFrame:
    """
    Create a set of 2D genomic intervals.

    Parameters
    ----------
    chroms1 : str, int, or list
        Chromosome name(s) for first dimension.
    starts1 : int or list, default 0
        Start coordinate(s) for first dimension.
    ends1 : int or list, default -1
        End coordinate(s) for first dimension. -1 means full chromosome length.
    chroms2 : str, int, list, or None
        Chromosome name(s) for second dimension. Defaults to chroms1.
    starts2 : int or list, default 0
        Start coordinate(s) for second dimension.
    ends2 : int or list, default -1
        End coordinate(s) for second dimension. -1 means full chromosome length.

    Returns
    -------
    DataFrame
        Sorted 2D intervals with columns: chrom1, start1, end1, chrom2, start2, end2.

    See Also
    --------
    gintervals : Create 1D intervals.
    gintervals_2d_all : Return 2D intervals covering the whole genome.
    gintervals_2d_band_intersect : Intersect 2D intervals with a diagonal band.
    gintervals_force_range : Clamp intervals to chromosome boundaries.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    The following calls produce equivalent results:

    >>> pm.gintervals_2d(1)  # doctest: +SKIP
    >>> pm.gintervals_2d("1")  # doctest: +SKIP
    >>> pm.gintervals_2d("chrX")  # doctest: +SKIP

    Explicit coordinates on both dimensions:

    >>> pm.gintervals_2d(1, 1000, 2000, "chrX", 400, 800)  # doctest: +SKIP

    Multiple intervals with broadcast:

    >>> pm.gintervals_2d(["chr2", "chrX"], 10, [3000, 5000], 1)  # doctest: +SKIP
    """
    if chroms2 is None:
        chroms2 = chroms1

    c1, s1, e1 = _make_1d_intervals(chroms1, starts1, ends1)
    c2, s2, e2 = _make_1d_intervals(chroms2, starts2, ends2)

    # R parity: ``gintervals.2d`` builds the result with a ``data.frame(...)`` of
    # both sides, which recycles a length-1 side against a length-N side. The
    # common case is ``gintervals.2d(c, s, e)`` with no axis2 args -- chroms2
    # defaults to chroms1 (length N), starts2/ends2 are scalars (length 1, =>
    # the per-row chrom's full extent), and the result is N rows of
    # ``(c[i], s[i], e[i]) x (c[i], 0, chrom_size[c[i]])``. Mirror that here:
    # broadcast a length-1 axis against a longer axis. (We only broadcast the
    # 1<->N case to keep accidental length-mismatches loud.)
    if len(c1) == 1 and len(c2) > 1:
        c1 = c1 * len(c2)
        s1 = s1 * len(c2)
        e1 = e1 * len(c2)
    elif len(c2) == 1 and len(c1) > 1:
        c2 = c2 * len(c1)
        s2 = s2 * len(c1)
        e2 = e2 * len(c1)

    if len(c1) != len(c2):
        raise ValueError("chroms1 and chroms2 must produce the same number of intervals")

    df = _pandas.DataFrame({
        'chrom1': c1, 'start1': s1, 'end1': e1,
        'chrom2': c2, 'start2': s2, 'end2': e2,
    })

    return df.sort_values(['chrom1', 'start1', 'chrom2', 'start2']).reset_index(drop=True)


def gintervals_2d_all(mode: str = "diagonal") -> pd.DataFrame:
    """
    Return 2D intervals covering the whole genome.

    Parameters
    ----------
    mode : str, default "diagonal"
        "diagonal" returns only intra-chromosomal pairs (chrom1 == chrom2).
        "full" returns all NxN chromosome pairs.

    Returns
    -------
    DataFrame
        2D intervals with columns: chrom1, start1, end1, chrom2, start2, end2.

    See Also
    --------
    gintervals_2d : Create a custom set of 2D intervals.
    gintervals_all : Return 1D intervals covering the whole genome.
    gintervals_2d_band_intersect : Intersect 2D intervals with a diagonal band.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Diagonal mode (intra-chromosomal pairs only):

    >>> pm.gintervals_2d_all()  # doctest: +SKIP

    Full NxN chromosome pairs:

    >>> pm.gintervals_2d_all(mode="full")  # doctest: +SKIP
    """
    _checkroot()

    if mode not in ("diagonal", "full"):
        raise ValueError(f"Unknown mode: {mode}. Must be 'diagonal' or 'full'")

    intervals = gintervals_all()

    if mode == "diagonal":
        df = _pandas.DataFrame({
            'chrom1': intervals['chrom'].to_numpy(),
            'start1': intervals['start'].values,
            'end1': intervals['end'].values,
            'chrom2': intervals['chrom'].to_numpy(),
            'start2': intervals['start'].values,
            'end2': intervals['end'].values,
        })
    else:
        # Full cartesian product (vectorized)
        chrom = intervals["chrom"].to_numpy()
        start = intervals["start"].to_numpy(copy=False)
        end = intervals["end"].to_numpy(copy=False)
        n = len(intervals)
        df = _pandas.DataFrame({
            "chrom1": _numpy.repeat(chrom, n),
            "start1": _numpy.repeat(start, n),
            "end1": _numpy.repeat(end, n),
            "chrom2": _numpy.tile(chrom, n),
            "start2": _numpy.tile(start, n),
            "end2": _numpy.tile(end, n),
        })

    return df.sort_values(['chrom1', 'start1', 'chrom2', 'start2']).reset_index(drop=True)


def gintervals_2d_band_intersect(
    intervals: pd.DataFrame,
    band: tuple[int, int] | None,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Intersect 2D intervals with a diagonal band.

    Each 2D interval is intersected with the band defined by two distances
    d1 and d2 from the main diagonal (where x == y). The band captures the
    region where d1 <= (start1 - start2) < d2. If the intersection is non-empty,
    the interval is shrunk to the minimal bounding rectangle of the intersection.

    Only cis (same-chromosome) intervals can intersect a band; trans intervals
    are removed.

    Parameters
    ----------
    intervals : DataFrame
        2D intervals with columns chrom1, start1, end1, chrom2, start2, end2.
    band : tuple of (int, int)
        Pair (d1, d2) defining the diagonal band. d1 must be < d2.
    intervals_set_out : str, optional
        If provided, save result as intervals set and return None.

    Returns
    -------
    DataFrame or None
        Intersected 2D intervals, or None if intervals_set_out is specified.

    See Also
    --------
    gintervals_2d : Create 2D intervals.
    gintervals_2d_all : Return 2D intervals covering the whole genome.
    gintervals_intersect : Intersect two 1D interval sets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals_2d(1)
    >>> pm.gintervals_2d_band_intersect(intervs, (10000, 20000))  # doctest: +SKIP
    """
    np = _numpy

    if band is None:
        raise ValueError("band cannot be None")
    if len(band) != 2:
        raise ValueError("band must be a pair (d1, d2)")
    d1, d2 = int(band[0]), int(band[1])
    if d1 >= d2:
        raise ValueError(f"band d1 ({d1}) must be less than d2 ({d2})")

    if len(intervals) == 0:
        result = intervals.copy()
        if intervals_set_out is not None:
            gintervals_save(result, intervals_set_out)
            return None
        return result

    # Only cis intervals can intersect a band
    cis_mask = intervals['chrom1'] == intervals['chrom2']

    x1 = intervals['start1'].values.astype(np.int64)
    y1 = intervals['start2'].values.astype(np.int64)
    x2 = intervals['end1'].values.astype(np.int64)
    y2 = intervals['end2'].values.astype(np.int64)

    # Intersection test: x2 - y1 > d1 and x1 - y2 + 1 < d2
    intersects = (x2 - y1 > d1) & (x1 - y2 + 1 < d2)
    keep = cis_mask.values & intersects

    if not np.any(keep):
        result = intervals.iloc[:0].copy()
        if intervals_set_out is not None:
            gintervals_save(result, intervals_set_out)
            return None
        return result.reset_index(drop=True)

    result = intervals.loc[keep].copy()
    rx1 = result['start1'].values.astype(np.int64).copy()
    ry1 = result['start2'].values.astype(np.int64).copy()
    rx2 = result['end1'].values.astype(np.int64).copy()
    ry2 = result['end2'].values.astype(np.int64).copy()

    # Containment check: x1 - y2 + 1 >= d1 and x2 - y1 <= d2
    contained = (rx1 - ry2 + 1 >= d1) & (rx2 - ry1 <= d2)
    need_shrink = ~contained

    if np.any(need_shrink):
        sx1 = rx1[need_shrink]
        sy1 = ry1[need_shrink]
        sx2 = rx2[need_shrink]
        sy2 = ry2[need_shrink]

        # Mirror misha C++ DiagonalBand::shrink2intersected exactly.
        delta1 = sx1 - sy1
        sx1 = np.where(delta1 < d1, sy1 + d1, sx1)
        sy1 = np.where(delta1 > d2, sx1 - d2, sy1)

        delta2 = sx2 - sy2
        sy2 = np.where(delta2 < d1, sx2 - d1, sy2)
        sx2 = np.where(delta2 > d2, sy2 + d2, sx2)

        rx1[need_shrink] = sx1
        ry1[need_shrink] = sy1
        rx2[need_shrink] = sx2
        ry2[need_shrink] = sy2

    result['start1'] = rx1
    result['start2'] = ry1
    result['end1'] = rx2
    result['end2'] = ry2
    result = result.reset_index(drop=True)

    if intervals_set_out is not None:
        gintervals_save(result, intervals_set_out)
        return None
    return result


def _sort_2d_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Sort 2D intervals by (chrom1, start1, chrom2, start2) and reset index."""
    return df.sort_values(
        ['chrom1', 'start1', 'chrom2', 'start2']
    ).reset_index(drop=True)


def _validate_2d_intervals(intervals: pd.DataFrame, name: str = "intervals") -> None:
    """Validate that a DataFrame has the required 2D interval columns."""
    required = {'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'}
    if not required.issubset(intervals.columns):
        missing = required - set(intervals.columns)
        raise ValueError(
            f"{name} is missing required 2D columns: {missing}"
        )


def _intersect_2d_rects(
    a: pd.DataFrame, b: pd.DataFrame, *, return_b_index: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
    """Clipped pairwise intersection of two 2D rectangle sets, scalably.

    Groups both sets by ``(chrom1, chrom2)``; for each shared chrom-pair builds
    an in-memory quadtree over the smaller side and queries it with each rect of
    the larger side, emitting the clipped (strict-overlap) intersection of every
    overlapping pair.  This mirrors R misha's quadtree-based ``gintervintersect``
    and avoids the ``O(n1*n2)`` broadcast that OOMs on the 10^5-rect 2D screens.

    Returns an *unsorted* DataFrame with columns ``chrom1, start1, end1,
    chrom2, start2, end2``.  Each overlapping ``(a, b)`` rectangle pair yields
    one row (same multiplicity as a brute-force broadcast).

    When ``return_b_index`` is True, also returns a parallel int64 array giving,
    for each output row, the 0-based positional index into ``b`` of the source
    rectangle (used to attribute each intersection back to its scope interval).
    """
    from ._quadtree import QuadTree

    np = _numpy
    cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        empty = _pandas.DataFrame(columns=cols)
        return (empty, np.empty(0, dtype=np.int64)) if return_b_index else empty

    ga = a.groupby(["chrom1", "chrom2"], observed=True)
    gb = b.groupby(["chrom1", "chrom2"], observed=True)
    common = set(ga.groups.keys()) & set(gb.groups.keys())

    out_c1: list[Any] = []
    out_c2: list[Any] = []
    out_s1: list[Any] = []
    out_e1: list[Any] = []
    out_s2: list[Any] = []
    out_e2: list[Any] = []
    out_bidx: list[Any] = []

    for key in common:
        da = ga.get_group(key)
        db = gb.get_group(key)
        # Positions of this group's rows in the original `b` (get_group and
        # .indices share the original-row order, so they align element-wise).
        db_pos = gb.indices[key]
        # Build the tree over the smaller side; query with the larger side.
        b_is_tree = len(db) <= len(da)
        tree_df, query_df = (db, da) if b_is_tree else (da, db)

        ts1 = tree_df["start1"].to_numpy(np.int64)
        te1 = tree_df["end1"].to_numpy(np.int64)
        ts2 = tree_df["start2"].to_numpy(np.int64)
        te2 = tree_df["end2"].to_numpy(np.int64)
        qs1 = query_df["start1"].to_numpy(np.int64)
        qe1 = query_df["end1"].to_numpy(np.int64)
        qs2 = query_df["start2"].to_numpy(np.int64)
        qe2 = query_df["end2"].to_numpy(np.int64)

        bound1 = int(max(int(te1.max()), int(qe1.max()))) + 1
        bound2 = int(max(int(te2.max()), int(qe2.max()))) + 1

        qt = QuadTree(0, 0, bound1, bound2, is_points=False)
        for i in range(len(ts1)):
            qt.insert((int(ts1[i]), int(ts2[i]), int(te1[i]), int(te2[i]), 0.0))

        q_idx_list: list[int] = []
        o_idx_list: list[int] = []
        for j in range(len(qs1)):
            cand = qt.query(int(qs1[j]), int(qs2[j]), int(qe1[j]), int(qe2[j]))
            if cand:
                q_idx_list.extend([j] * len(cand))
                o_idx_list.extend(cand)
        if not o_idx_list:
            continue

        qi = np.asarray(q_idx_list, dtype=np.int64)
        oi = np.asarray(o_idx_list, dtype=np.int64)
        ix1 = np.maximum(qs1[qi], ts1[oi])
        iy1 = np.maximum(qs2[qi], ts2[oi])
        ix2 = np.minimum(qe1[qi], te1[oi])
        iy2 = np.minimum(qe2[qi], te2[oi])
        # query() already guarantees strict overlap; the mask is a safety net.
        m = (ix1 < ix2) & (iy1 < iy2)
        if not np.any(m):
            continue
        c1, c2 = key
        n = int(m.sum())
        out_c1.extend([c1] * n)
        out_c2.extend([c2] * n)
        out_s1.append(ix1[m])
        out_e1.append(ix2[m])
        out_s2.append(iy1[m])
        out_e2.append(iy2[m])
        if return_b_index:
            # b's local row index for each pair = tree-object index when b is
            # the tree, else the query index.
            b_local = oi if b_is_tree else qi
            out_bidx.append(db_pos[b_local][m])

    if not out_c1:
        empty = _pandas.DataFrame(columns=cols)
        return (empty, np.empty(0, dtype=np.int64)) if return_b_index else empty
    result = _pandas.DataFrame({
        "chrom1": out_c1,
        "start1": np.concatenate(out_s1),
        "end1": np.concatenate(out_e1),
        "chrom2": out_c2,
        "start2": np.concatenate(out_s2),
        "end2": np.concatenate(out_e2),
    })
    if return_b_index:
        return result, np.concatenate(out_bidx).astype(np.int64)
    return result


def gintervals_2d_intersect(intervals1: pd.DataFrame, intervals2: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compute the intersection of two 2D interval sets.

    Returns rectangles representing the overlapping regions between pairs
    of intervals from *intervals1* and *intervals2*. Each result rectangle
    is the pairwise intersection of one rectangle from each input set:

    - ``new_start1 = max(rect1.start1, rect2.start1)``
    - ``new_end1   = min(rect1.end1,   rect2.end1)``
    - ``new_start2 = max(rect1.start2, rect2.start2)``
    - ``new_end2   = min(rect1.end2,   rect2.end2)``

    An intersection rectangle is emitted only when
    ``new_start1 < new_end1`` **and** ``new_start2 < new_end2``.

    Parameters
    ----------
    intervals1 : DataFrame
        First set of 2D intervals (chrom1, start1, end1, chrom2, start2, end2).
    intervals2 : DataFrame
        Second set of 2D intervals (chrom1, start1, end1, chrom2, start2, end2).

    Returns
    -------
    DataFrame or None
        2D intervals representing pairwise intersections, sorted by
        (chrom1, start1, chrom2, start2), or ``None`` if no intersections
        exist.

    See Also
    --------
    gintervals_intersect : Intersection of two 1D interval sets.
    gintervals_2d_union : Union of two 2D interval sets.
    gintervals_2d_band_intersect : Intersect 2D intervals with a diagonal band.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> iv1 = pm.gintervals_2d("1", 0, 1000, "1", 0, 1000)
    >>> iv2 = pm.gintervals_2d("1", 500, 1500, "1", 500, 1500)
    >>> pm.gintervals_2d_intersect(iv1, iv2)  # doctest: +SKIP
    """
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")

    _validate_2d_intervals(intervals1, "intervals1")
    _validate_2d_intervals(intervals2, "intervals2")

    if len(intervals1) == 0 or len(intervals2) == 0:
        return None

    result = _intersect_2d_rects(intervals1, intervals2)
    if len(result) == 0:
        return None

    return _sort_2d_intervals(result)


def gintervals_2d_union(intervals1: pd.DataFrame, intervals2: pd.DataFrame) -> pd.DataFrame | None:
    """
    Compute the union of two 2D interval sets.

    Concatenates the two interval sets and sorts the result by
    (chrom1, start1, chrom2, start2). Since merging overlapping 2D
    rectangles is not well-defined in general (the union of two rectangles
    is not necessarily a rectangle), this function simply returns the
    combined sorted set.

    Parameters
    ----------
    intervals1 : DataFrame
        First set of 2D intervals (chrom1, start1, end1, chrom2, start2, end2).
    intervals2 : DataFrame
        Second set of 2D intervals (chrom1, start1, end1, chrom2, start2, end2).

    Returns
    -------
    DataFrame or None
        Combined 2D intervals sorted by (chrom1, start1, chrom2, start2),
        or ``None`` if both inputs are empty.

    See Also
    --------
    gintervals_union : Union of two 1D interval sets.
    gintervals_2d_intersect : Intersection of two 2D interval sets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> iv1 = pm.gintervals_2d("1", 0, 1000, "1", 0, 1000)
    >>> iv2 = pm.gintervals_2d("1", 500, 1500, "1", 500, 1500)
    >>> pm.gintervals_2d_union(iv1, iv2)  # doctest: +SKIP
    """
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")

    _validate_2d_intervals(intervals1, "intervals1")
    _validate_2d_intervals(intervals2, "intervals2")

    cols = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

    if len(intervals1) == 0 and len(intervals2) == 0:
        return None
    if len(intervals1) == 0:
        return _sort_2d_intervals(intervals2[cols].copy())
    if len(intervals2) == 0:
        return _sort_2d_intervals(intervals1[cols].copy())

    combined = _pandas.concat(
        [intervals1[cols], intervals2[cols]],
        ignore_index=True,
    )
    return _sort_2d_intervals(combined)


def gintervals_from_tuples(
    rows: list[tuple[Any, ...]] | list[dict[str, Any]] | None,
    strand: int | list[int] | None = None,
) -> pd.DataFrame | None:
    """
    Create intervals from a list of tuples or dicts.

    Each tuple should be ``(chrom, start, end)`` or
    ``(chrom, start, end, strand)``. Alternatively, each element can be a
    dict with the corresponding keys.

    Parameters
    ----------
    rows : list of tuple or list of dict
        Interval specifications. Tuples must have 3 or 4 elements.
    strand : int or list of int, optional
        Strand values to assign when the tuples do not include strand.

    Returns
    -------
    DataFrame or None
        Sorted intervals with columns: chrom, start, end (and optionally
        strand). Returns ``None`` if *rows* is ``None``.

    See Also
    --------
    gintervals : Create intervals from parallel arrays.
    gintervals_from_strings : Create intervals from region strings.
    gintervals_from_bed : Create intervals from a BED file.
    gintervals_all : Return full-chromosome intervals.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_from_tuples([("1", 100, 200), ("1", 250, 300)])  # doctest: +SKIP
    """
    if rows is None:
        return None
    if len(rows) == 0:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])

    first = rows[0]
    if isinstance(first, dict):
        df = _pandas.DataFrame(rows)
    else:
        if len(first) == 3:
            df = _pandas.DataFrame(rows, columns=["chrom", "start", "end"])
        elif len(first) == 4:
            df = _pandas.DataFrame(rows, columns=["chrom", "start", "end", "strand"])
        else:
            raise ValueError("Tuples must have 3 or 4 elements")

    if strand is not None and "strand" not in df.columns:
        df["strand"] = strand

    return gintervals(df["chrom"], df["start"], df["end"], df.get("strand"))


def gintervals_from_strings(regions: str | list[str]) -> pd.DataFrame:
    """
    Create intervals from region strings.

    Parses strings of the form ``"chr1:100-200"`` or ``"chr1:100-200:+"``
    into an intervals DataFrame. If only a chromosome name is given
    (e.g. ``"chr1"``), the full chromosome extent is used.

    Parameters
    ----------
    regions : str or list of str
        One or more region strings. Accepted formats:

        - ``"chrom"`` -- full chromosome
        - ``"chrom:start-end"`` -- region without strand
        - ``"chrom:start-end:+"`` or ``"chrom:start-end:-"`` -- with strand

    Returns
    -------
    DataFrame
        Sorted intervals with columns: chrom, start, end (and optionally
        strand).

    Raises
    ------
    ValueError
        If a region string cannot be parsed.

    See Also
    --------
    gintervals : Create intervals from parallel arrays.
    gintervals_from_tuples : Create intervals from a list of tuples.
    gintervals_from_bed : Create intervals from a BED file.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_from_strings(["1:100-200", "1:300-400:-"])  # doctest: +SKIP
    """
    if isinstance(regions, str):
        regions = [regions]

    chroms, starts, ends, strands = [], [], [], []
    has_strand = False

    for spec in regions:
        m = re.match(r'^(?P<chrom>[^:]+)(?::(?P<start>\d+)(?:[-\.]{1,2})(?P<end>\d+))?(?::(?P<strand>[+-]))?$', spec)
        if not m:
            raise ValueError(f"Invalid interval string: {spec}")
        chrom = m.group("chrom")
        start = m.group("start")
        end = m.group("end")
        strand = m.group("strand")

        if start is None:
            start = 0
            end = -1
        else:
            start = int(start)
            end = int(end)

        chroms.append(chrom)
        starts.append(start)
        ends.append(end)
        if strand is not None:
            has_strand = True
            strands.append(1 if strand == "+" else -1)
        else:
            strands.append(0)

    return gintervals(
        chroms, starts, ends,
        cast(list[int | str], strands) if has_strand else None,
    )


def gintervals_from_bed(path: str | Path, has_strand: bool = False) -> pd.DataFrame | None:
    """
    Create intervals from a BED-like file.

    Reads a tab- or space-delimited file with at least three columns
    (chrom, start, end) and returns a sorted intervals DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to BED file (chrom, start, end[, ...]).
    has_strand : bool, default False
        If True, use column 6 for strand when present.

    Returns
    -------
    DataFrame or None
        Sorted intervals with columns: chrom, start, end (and optionally
        strand). Returns ``None`` if the file contains no intervals.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    See Also
    --------
    gintervals : Create intervals from parallel arrays.
    gintervals_from_tuples : Create intervals from a list of tuples.
    gintervals_from_strings : Create intervals from region strings.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_from_bed("example.bed")  # doctest: +SKIP
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[Any] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            strand = 0
            if has_strand and len(parts) >= 6:
                if parts[5] == "+":
                    strand = 1
                elif parts[5] == "-":
                    strand = -1
            if has_strand:
                rows.append((chrom, start, end, strand))
            else:
                rows.append((chrom, start, end))

    if not rows:
        return None

    return gintervals_from_tuples(rows)


def _open_text_for_import(path: Path) -> IO[str]:
    """Open a BED/GFF/VCF file as text, transparently handling .gz."""
    if str(path).endswith(".gz"):
        return cast(IO[str], gzip.open(path, "rt"))
    return path.open("rt")


def _read_table_filtered(path: Path, header_pattern: re.Pattern[str]) -> pd.DataFrame:
    """Read a tab-delimited file, skipping header lines that match
    *header_pattern* on column 1. Returns a DataFrame with all-string
    columns; later importers coerce as needed.
    """
    rows: list[list[str]] = []
    max_cols = 0
    with _open_text_for_import(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            first = line.split("\t", 1)[0].strip()
            if header_pattern.match(first):
                continue
            parts = line.split("\t")
            rows.append(parts)
            if len(parts) > max_cols:
                max_cols = len(parts)
    if not rows:
        return _pandas.DataFrame()
    for r in rows:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))
    cols = [f"V{i + 1}" for i in range(max_cols)]
    return _pandas.DataFrame(rows, columns=cols, dtype=object)


def _sort_intervals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort an intervals DataFrame by (chromid, start, end) without
    stripping extra columns, mirroring R's ``.gsort_intervals_df``.
    """
    if df.empty:
        return df.reset_index(drop=True)
    allchroms = gintervals_all()["chrom"].astype(str).tolist()
    chrom_order = {c: i for i, c in enumerate(allchroms)}
    chromid = df["chrom"].astype(str).map(
        lambda c: chrom_order.get(c, len(chrom_order))
    )
    df = df.assign(_chromid=chromid)
    df = df.sort_values(by=["_chromid", "start", "end"], kind="mergesort")
    return df.drop(columns=["_chromid"]).reset_index(drop=True)


_BED_HEADER_RE = re.compile(r"^(track|browser|#|$)")
_GFF_HEADER_RE = re.compile(r"^#")
_VCF_HEADER_RE = re.compile(r"^#")


def gintervals_import_bed(
    file: str | Path,
    *,
    name: bool = True,
    score: bool = True,
    strand: bool = True,
) -> pd.DataFrame:
    """
    Import intervals from a BED file.

    Reads a BED (or BED.gz) file and returns a misha 1D intervals
    DataFrame. ``track``/``browser``/``#`` header lines are skipped
    automatically. Chromosome names are normalised through the active
    database's chromosome-alias mechanism, so ``chr1`` <-> ``1`` works
    transparently.

    BED is already 0-based half-open, so coordinates are kept as-is.

    Parameters
    ----------
    file : str or Path
        Path to a BED file (``.bed`` or ``.bed.gz``).
    name : bool, default True
        If True and a 4th column exists, include it as ``name``.
    score : bool, default True
        If True and a 5th column exists, include it as ``score``
        (numeric).
    strand : bool, default True
        If True and a 6th column exists, include it as ``strand``
        (mapped to ``1``/``-1``/``0``).

    Returns
    -------
    DataFrame
        Sorted intervals with columns ``chrom``, ``start``, ``end`` and
        any of the optional metadata columns described above.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file has fewer than three columns or contains no records.

    See Also
    --------
    gintervals_import_gff : Import GFF/GTF.
    gintervals_import_vcf : Import VCF.
    gintervals_from_bed : Older 3-column BED reader.
    """
    _checkroot()
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"BED file {path} does not exist")

    table = _read_table_filtered(path, _BED_HEADER_RE)
    if table.empty:
        raise ValueError(
            f"BED file {path} appears to be empty or contains no data intervals"
        )
    if table.shape[1] < 3:
        raise ValueError(
            f"BED file {path} appears to be malformed (less than 3 columns)"
        )

    n_cols = table.shape[1]
    starts = _pandas.to_numeric(table.iloc[:, 1], errors="coerce")
    ends = _pandas.to_numeric(table.iloc[:, 2], errors="coerce")
    if starts.isna().any() or ends.isna().any():
        raise ValueError(f"Non-numeric coordinates detected in BED file {path}")

    out = _pandas.DataFrame({
        "chrom": _normalize_chroms(table.iloc[:, 0].astype(str).tolist()),
        "start": starts.astype("int64"),
        "end": ends.astype("int64"),
    })

    if n_cols >= 6 and strand:
        out["strand"] = _normalize_strand_column(table.iloc[:, 5])
    if n_cols >= 4 and name:
        out["name"] = table.iloc[:, 3].astype(str).values
    if n_cols >= 5 and score:
        out["score"] = _pandas.to_numeric(
            table.iloc[:, 4].astype(str), errors="coerce"
        )

    return _sort_intervals_df(out)


def gintervals_import_gff(
    file: str | Path,
    *,
    feature: str | list[str] | None = None,
    strand: bool = True,
    attrs: bool = True,
) -> pd.DataFrame:
    """
    Import intervals from a GFF/GTF file.

    GFF/GTF coordinates are 1-based and inclusive on both ends. The
    importer converts to 0-based half-open by subtracting 1 from
    ``start`` and leaving ``end`` as-is. Chromosome names are normalised
    through the active database's chromosome-alias mechanism.

    Parameters
    ----------
    file : str or Path
        Path to a GFF/GTF file (``.gff``, ``.gtf``, or ``.gz``).
    feature : str or list of str, optional
        If given, keep only records whose feature type (column 3) is in
        *feature*.
    strand : bool, default True
        If True, include the ``strand`` column (numeric).
    attrs : bool, default True
        If True, include the raw attribute string as ``attrs`` (not
        parsed).

    Returns
    -------
    DataFrame
        Sorted intervals with columns ``chrom``, ``start``, ``end``, and
        optionally ``strand``, ``source``, ``type``, ``score``,
        ``attrs``.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file is empty, malformed, or no records of the requested
        feature type are found.
    """
    _checkroot()
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"GFF file {path} does not exist")

    table = _read_table_filtered(path, _GFF_HEADER_RE)
    if table.empty:
        raise ValueError(
            f"GFF file {path} appears to be empty or contains no records"
        )
    if table.shape[1] < 8:
        raise ValueError(
            f"GFF file {path} appears to be malformed (expected at least 8 "
            f"tab-separated columns, got {table.shape[1]})"
        )

    if feature is not None:
        feature_list = [feature] if isinstance(feature, str) else list(feature)
        keep = table.iloc[:, 2].isin(feature_list)
        table = table.loc[keep].reset_index(drop=True)
        if table.empty:
            raise ValueError(
                f"No records of feature type(s) {', '.join(feature_list)} "
                f"found in GFF file {path}"
            )

    starts1 = _pandas.to_numeric(table.iloc[:, 3], errors="coerce")
    ends1 = _pandas.to_numeric(table.iloc[:, 4], errors="coerce")
    if starts1.isna().any() or ends1.isna().any():
        raise ValueError(f"Non-numeric coordinates detected in GFF file {path}")

    out = _pandas.DataFrame({
        "chrom": _normalize_chroms(table.iloc[:, 0].astype(str).tolist()),
        "start": (starts1 - 1).astype("int64"),
        "end": ends1.astype("int64"),
    })
    if strand:
        out["strand"] = _normalize_strand_column(table.iloc[:, 6])
    out["source"] = table.iloc[:, 1].astype(str).values
    out["type"] = table.iloc[:, 2].astype(str).values
    out["score"] = _pandas.to_numeric(
        table.iloc[:, 5].astype(str), errors="coerce"
    )
    if attrs and table.shape[1] >= 9:
        out["attrs"] = table.iloc[:, 8].astype(str).values

    return _sort_intervals_df(out)


def gintervals_import_vcf(
    file: str | Path,
    *,
    info: bool = True,
) -> pd.DataFrame:
    """
    Import intervals from a VCF file.

    VCF is 1-based; this importer sets ``start = POS - 1`` and
    ``end = POS - 1 + len(REF)``, yielding a 0-based half-open span
    covering the reference allele. Multi-allelic records are kept on a
    single row; the ``alt`` column contains the original
    comma-separated string.

    Chromosome names are normalised through the active database's
    chromosome-alias mechanism.

    Parameters
    ----------
    file : str or Path
        Path to a VCF file (``.vcf`` or ``.vcf.gz``).
    info : bool, default True
        If True, include the raw INFO column as ``info`` (not parsed).

    Returns
    -------
    DataFrame
        Sorted intervals with columns ``chrom``, ``start``, ``end``,
        ``id``, ``ref``, ``alt``, ``qual``, ``filter`` and optionally
        ``info``.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file is empty, malformed, or contains an empty REF allele.
    """
    _checkroot()
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"VCF file {path} does not exist")

    table = _read_table_filtered(path, _VCF_HEADER_RE)
    if table.empty:
        raise ValueError(
            f"VCF file {path} appears to be empty or contains no records"
        )
    if table.shape[1] < 5:
        raise ValueError(
            f"VCF file {path} appears to be malformed (expected at least 5 "
            f"tab-separated columns, got {table.shape[1]})"
        )

    pos = _pandas.to_numeric(table.iloc[:, 1], errors="coerce")
    if pos.isna().any():
        raise ValueError(f"Non-numeric POS detected in VCF file {path}")
    ref = table.iloc[:, 3].astype(str)
    ref_len = ref.str.len()
    if (ref_len < 1).any():
        raise ValueError(f"Empty REF allele detected in VCF file {path}")

    out = _pandas.DataFrame({
        "chrom": _normalize_chroms(table.iloc[:, 0].astype(str).tolist()),
        "start": (pos - 1).astype("int64"),
        "end": (pos - 1 + ref_len).astype("int64"),
    })
    out["id"] = table.iloc[:, 2].astype(str).values
    out["ref"] = ref.values
    out["alt"] = table.iloc[:, 4].astype(str).values
    if table.shape[1] >= 6:
        out["qual"] = _pandas.to_numeric(
            table.iloc[:, 5].astype(str), errors="coerce"
        )
    if table.shape[1] >= 7:
        out["filter"] = table.iloc[:, 6].astype(str).values
    if info and table.shape[1] >= 8:
        out["info"] = table.iloc[:, 7].astype(str).values

    return _sort_intervals_df(out)


def gintervals_window(
    chroms: str | int | list[Any],
    centers: int | list[int],
    half_width: int,
) -> pd.DataFrame:
    """
    Create intervals centered on positions with fixed half-width.

    Constructs intervals of width ``2 * half_width`` centered on each
    position in *centers*.

    Parameters
    ----------
    chroms : str, int, or list
        Chromosome name(s). Scalar is broadcast to match *centers*.
    centers : int or list of int
        Center positions. Scalar is broadcast to match *chroms*.
    half_width : int
        Half the desired interval width.

    Returns
    -------
    DataFrame
        Sorted intervals with columns: chrom, start, end.

    See Also
    --------
    gintervals : Create intervals from explicit start/end coordinates.
    gintervals_normalize : Resize intervals by centering.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_window("1", [1000, 2000], half_width=50)  # doctest: +SKIP
    """
    if isinstance(chroms, (str, int)):
        chroms = [chroms]
    if isinstance(centers, int):
        centers = [centers]

    if len(chroms) == 1 and len(centers) > 1:
        chroms = chroms * len(centers)
    if len(centers) == 1 and len(chroms) > 1:
        centers = centers * len(chroms)

    starts = [c - half_width for c in centers]
    ends = [c + half_width for c in centers]
    return gintervals(chroms, starts, ends)


def gintervals_force_range(
    intervals: pd.DataFrame | str,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Force intervals into valid chromosome ranges.

    Enforces intervals to lie within [0, chrom_length) by clamping their
    boundaries. Intervals that fall entirely outside chromosome ranges
    are removed.

    Parameters
    ----------
    intervals : DataFrame
        1D intervals with columns: chrom, start, end.

    Returns
    -------
    DataFrame or None
        Clamped intervals, or ``None`` if all intervals are out of range
        or the input is empty.

    Raises
    ------
    ValueError
        If *intervals* is ``None``.

    See Also
    --------
    gintervals : Create a set of 1D intervals.
    gintervals_2d : Create a set of 2D intervals.
    gintervals_canonic : Merge overlapping intervals.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import pandas as pd
    >>> intervs = pd.DataFrame({
    ...     "chrom": ["1", "1", "1", "1"],
    ...     "start": [11000, -100, 10000, 10500],
    ...     "end":   [12000, 200, 1300000, 10600],
    ... })
    >>> pm.gintervals_force_range(intervs)  # doctest: +SKIP
    """
    intervals = _resolve_intervals(intervals)
    _checkroot()

    if intervals is None:
        raise ValueError("intervals cannot be None")
    assert isinstance(intervals, pd.DataFrame)

    if len(intervals) == 0:
        return None

    all_intervals = gintervals_all()
    chrom_sizes = dict(
        zip(
            all_intervals["chrom"].astype(str).tolist(),
            all_intervals["end"].astype(int).tolist(), strict=False,
        )
    )

    def _force_axis(chrom_vals, starts, ends):
        out_chrom = []
        out_start = []
        out_end = []
        with _contextlib.suppress(Exception):
            chrom_vals = _normalize_chroms(chrom_vals)

        for chrom, start, end in zip(chrom_vals, starts, ends, strict=False):
            if chrom not in chrom_sizes:
                out_chrom.append(None)
                out_start.append(None)
                out_end.append(None)
                continue
            chrom_size = chrom_sizes[chrom]
            start = max(0, int(start))
            end = min(chrom_size, int(end))
            if start < end:
                out_chrom.append(chrom)
                out_start.append(start)
                out_end.append(end)
            else:
                out_chrom.append(None)
                out_start.append(None)
                out_end.append(None)
        return out_chrom, out_start, out_end

    # 2D intervals
    if {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}.issubset(intervals.columns):
        c1, s1, e1 = _force_axis(
            intervals["chrom1"].astype(str).tolist(),
            intervals["start1"].tolist(),
            intervals["end1"].tolist(),
        )
        c2, s2, e2 = _force_axis(
            intervals["chrom2"].astype(str).tolist(),
            intervals["start2"].tolist(),
            intervals["end2"].tolist(),
        )

        keep = [
            i for i in range(len(c1))
            if c1[i] is not None and c2[i] is not None
        ]
        if not keep:
            return None
        result = intervals.iloc[keep].copy()
        result["chrom1"] = [c1[i] for i in keep]
        result["start1"] = [s1[i] for i in keep]
        result["end1"] = [e1[i] for i in keep]
        result["chrom2"] = [c2[i] for i in keep]
        result["start2"] = [s2[i] for i in keep]
        result["end2"] = [e2[i] for i in keep]
        result = result.reset_index(drop=True)
        if intervals_set_out is not None:
            gintervals_save(result, intervals_set_out)
            return None
        return result

    # 1D intervals
    chrom_vals, starts, ends = _force_axis(
        intervals["chrom"].astype(str).tolist(),
        intervals["start"].tolist(),
        intervals["end"].tolist(),
    )
    keep = [i for i in range(len(chrom_vals)) if chrom_vals[i] is not None]
    if not keep:
        return None
    result = intervals.iloc[keep].copy()
    result["chrom"] = [chrom_vals[i] for i in keep]
    result["start"] = [starts[i] for i in keep]
    result["end"] = [ends[i] for i in keep]
    result = result.reset_index(drop=True)
    if intervals_set_out is not None:
        gintervals_save(result, intervals_set_out)
        return None
    return result


def gintervals_is_bigset(intervals_set: str) -> bool:
    """Return whether a saved interval set uses directory ("bigset") storage."""
    _checkroot()
    if not isinstance(intervals_set, str) or not intervals_set.strip():
        raise ValueError("intervals_set must be a non-empty string")
    dataset = gintervals_dataset(intervals_set)
    if dataset is None:
        return False
    path_part = intervals_set.replace(".", "/")
    for suffix in (".interv", ".interv2d"):
        p = Path(dataset) / "tracks" / f"{path_part}{suffix}"
        if p.exists():
            return p.is_dir()
    return False


def _sort_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    return intervals.sort_values(['chrom', 'start', 'end']).reset_index(drop=True)



def _intervals_to_cpp(intervals: pd.DataFrame) -> Any:
    """Prepare intervals for C++ processing (chrom/start/end columns).

    Returns the pymisha internal list-of-arrays format directly to skip the
    DataFrame.copy() + per-column iloc that _df2pymisha goes through.
    """
    chrom_series = intervals['chrom']
    if isinstance(chrom_series.dtype, pd.CategoricalDtype):
        chrom_arr = chrom_series.astype(str).to_numpy()
    else:
        chrom_arr = chrom_series.to_numpy()
    start_arr = intervals['start'].to_numpy()
    end_arr = intervals['end'].to_numpy()
    _validate_interval_coords(chrom_arr, start_arr, end_arr)
    return [
        _numpy.array(['chrom', 'start', 'end'], dtype=object),
        chrom_arr,
        start_arr,
        end_arr,
    ]


def _validate_interval_coords(
    chrom_arr: Any, start_arr: Any, end_arr: Any
) -> None:
    """Validate 1D interval coordinates before a C++ set operation.

    R misha raises on any interval with ``start >= end`` or ``start < 0``
    (e.g. in gintervals.canonic / intersect / union / diff). Without this
    check pymisha silently dropped zero-width intervals and passed inverted
    (start > end) intervals straight through, corrupting downstream results.
    Vectorized so it adds negligible cost on large inputs.
    """
    bad = start_arr >= end_arr
    if bad.any():
        i = int(_numpy.argmax(bad))
        raise ValueError(
            f"Invalid interval ({chrom_arr[i]}, {start_arr[i]}, {end_arr[i]}): "
            "start must be < end"
        )
    neg = start_arr < 0
    if neg.any():
        i = int(_numpy.argmax(neg))
        raise ValueError(
            f"Invalid interval ({chrom_arr[i]}, {start_arr[i]}, {end_arr[i]}): "
            "start must be >= 0"
        )


def gintervals_union(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Calculate the union of two sets of intervals.

    Returns intervals representing the genomic space covered by either
    ``intervals1`` or ``intervals2``. Overlapping and adjacent regions
    are merged in the result.

    Parameters
    ----------
    intervals1 : DataFrame
        First set of 1D intervals (chrom, start, end).
    intervals2 : DataFrame
        Second set of 1D intervals (chrom, start, end).

    Returns
    -------
    DataFrame or None
        Union intervals sorted by chrom and start, or ``None`` if both
        inputs are empty.

    See Also
    --------
    gintervals_intersect : Intersection of two interval sets.
    gintervals_diff : Difference of two interval sets.
    gintervals_canonic : Merge overlapping intervals within one set.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs1 = pm.gintervals("1", [0, 500], [300, 800])
    >>> intervs2 = pm.gintervals("1", [200, 700], [400, 900])
    >>> pm.gintervals_union(intervs1, intervs2)  # doctest: +SKIP
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")
    assert isinstance(intervals1, pd.DataFrame)
    assert isinstance(intervals2, pd.DataFrame)

    if len(intervals1) == 0 and len(intervals2) == 0:
        return None
    if len(intervals1) == 0:
        return _sort_intervals(intervals2[['chrom', 'start', 'end']].copy())
    if len(intervals2) == 0:
        return _sort_intervals(intervals1[['chrom', 'start', 'end']].copy())

    _checkroot()
    result = _pymisha.pm_intervals_union(
        _intervals_to_cpp(intervals1),
        _intervals_to_cpp(intervals2)
    )

    if result is None or len(result['chrom']) == 0:
        return None

    out = _pandas.DataFrame(result)
    if intervals_set_out is not None:
        gintervals_save(out, intervals_set_out)
        return None
    return out


def gintervals_intersect(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Calculate the intersection of two sets of intervals.

    Returns intervals representing the genomic space covered by both
    ``intervals1`` and ``intervals2``.

    Parameters
    ----------
    intervals1 : DataFrame
        First set of 1D intervals (chrom, start, end).
    intervals2 : DataFrame
        Second set of 1D intervals (chrom, start, end).

    Returns
    -------
    DataFrame or None
        Intersection intervals sorted by chrom and start, or ``None``
        if the intersection is empty.

    See Also
    --------
    gintervals_union : Union of two interval sets.
    gintervals_diff : Difference of two interval sets.
    gintervals_2d_band_intersect : Intersect 2D intervals with a diagonal band.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs1 = pm.gintervals("1", 0, 500)
    >>> intervs2 = pm.gintervals("1", 300, 800)
    >>> pm.gintervals_intersect(intervs1, intervs2)  # doctest: +SKIP
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")

    if len(intervals1) == 0 or len(intervals2) == 0:
        return None

    _checkroot()
    result = _pymisha.pm_intervals_intersect(
        _intervals_to_cpp(intervals1),
        _intervals_to_cpp(intervals2)
    )

    if result is None or len(result['chrom']) == 0:
        return None

    out = _pandas.DataFrame(result)
    if intervals_set_out is not None:
        gintervals_save(out, intervals_set_out)
        return None
    return out


def gintervals_diff(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Calculate the difference of two interval sets.

    Returns genomic space covered by ``intervals1`` but not by
    ``intervals2``.

    Parameters
    ----------
    intervals1 : DataFrame
        First set of 1D intervals (chrom, start, end).
    intervals2 : DataFrame
        Second set of 1D intervals (chrom, start, end).

    Returns
    -------
    DataFrame or None
        Difference intervals sorted by chrom and start, or ``None``
        if the result is empty.

    See Also
    --------
    gintervals_union : Union of two interval sets.
    gintervals_intersect : Intersection of two interval sets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs1 = pm.gintervals("1", 0, 500)
    >>> intervs2 = pm.gintervals("1", 200, 300)
    >>> pm.gintervals_diff(intervs1, intervs2)  # doctest: +SKIP
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")
    assert isinstance(intervals1, pd.DataFrame)
    assert isinstance(intervals2, pd.DataFrame)

    if len(intervals1) == 0:
        return None

    # R's gintervdiff sorts and unifies overlaps (incl. touching) on both
    # operands before the difference; mirror that so non-canonic (e.g.
    # rbind-ed) inputs give the same result as in R.
    canon1 = gintervals_canonic(intervals1[['chrom', 'start', 'end']].copy())
    assert isinstance(canon1, pd.DataFrame)
    intervals1 = canon1[['chrom', 'start', 'end']]

    if len(intervals2) == 0:
        return _sort_intervals(intervals1.copy())

    canon2 = gintervals_canonic(intervals2[['chrom', 'start', 'end']].copy())
    assert isinstance(canon2, pd.DataFrame)
    intervals2 = canon2[['chrom', 'start', 'end']]

    _checkroot()
    result = _pymisha.pm_intervals_diff(
        _intervals_to_cpp(intervals1),
        _intervals_to_cpp(intervals2)
    )

    if result is None or len(result['chrom']) == 0:
        return None

    out = _pandas.DataFrame(result)
    if intervals_set_out is not None:
        gintervals_save(out, intervals_set_out)
        return None
    return out


def gintervals_canonic(
    intervals: pd.DataFrame | str,
    unify_touching_intervals: bool = True,
) -> pd.DataFrame | None:
    """
    Convert intervals to canonical form.

    Sorts intervals and merges overlapping ones. If
    ``unify_touching_intervals`` is True, adjacent intervals (where one's
    end equals another's start) are also merged. The result has no overlaps
    and is properly sorted.

    A ``mapping`` attribute is attached to the result DataFrame mapping
    each original interval index to the canonical interval index:
    ``result.attrs['mapping']``.

    Parameters
    ----------
    intervals : DataFrame
        Intervals to canonicalize (chrom, start, end).
    unify_touching_intervals : bool, default True
        Whether to merge touching (end == start) intervals.

    Returns
    -------
    DataFrame or None
        Canonical intervals with ``mapping`` attribute, or ``None`` if
        input is empty.

    See Also
    --------
    gintervals_union : Union of two interval sets (implicitly canonicalizes).
    gintervals_intersect : Intersection of two interval sets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals("1", [0, 200, 100], [150, 300, 250])
    >>> result = pm.gintervals_canonic(intervs)
    >>> result  # doctest: +SKIP
    >>> result.attrs['mapping']  # doctest: +SKIP
    """
    intervals = _resolve_intervals(intervals)
    if intervals is None:
        raise ValueError("intervals cannot be None")
    assert isinstance(intervals, pd.DataFrame)
    if len(intervals) == 0:
        return None

    _checkroot()

    # Use C++ for the heavy lifting (sort + merge + mapping)
    cpp_result = _pymisha.pm_intervals_canonic(
        _intervals_to_cpp(intervals),
        unify_touching_intervals
    )

    if cpp_result is None:
        return None

    result_dict, mapping = cpp_result

    if len(result_dict['chrom']) == 0:
        return None

    result = _pandas.DataFrame(result_dict)
    result.attrs['mapping'] = mapping
    return result


def gintervals_covered_bp(intervals: pd.DataFrame | str, src: pd.DataFrame | str | None = None) -> int:
    """
    Compute total basepairs covered by intervals.

    Overlapping intervals are merged before counting to avoid double-counting.
    When *src* is provided, only the portion of *intervals* that overlaps
    *src* is counted.

    Parameters
    ----------
    intervals : DataFrame or str
        Interval set with columns: chrom, start, end.  A string is
        interpreted as a saved interval-set name.
    src : DataFrame, str, or None, default None
        If provided, restrict counting to the intersection of *intervals*
        with *src*.

    Returns
    -------
    int
        Total number of basepairs covered

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals("1", [0, 200], [300, 600])
    >>> pm.gintervals_covered_bp(intervs)  # 0-300 + 200-600 = 600 (overlaps merged)
    600

    See Also
    --------
    gintervals_coverage_fraction : Fraction of genomic space covered.
    gintervals_canonic : Merge overlapping intervals.
    gintervals : Create a set of 1D intervals.
    """
    intervals = _resolve_intervals(intervals)
    if src is not None:
        src = _resolve_intervals(src)
        intervals = gintervals_intersect(intervals, src)
        if intervals is None:
            return 0
    if intervals is None:
        raise ValueError("intervals cannot be None")
    if len(intervals) == 0:
        return 0

    _checkroot()
    return int(_pymisha.pm_intervals_covered_bp(
        _intervals_to_cpp(intervals)
    ))


def gintervals_coverage_fraction(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str | None = None,
) -> float:
    """
    Calculate the fraction of genomic space covered by intervals.

    Returns the fraction of *intervals2* (or the entire genome when
    *intervals2* is ``None``) that is covered by *intervals1*. Overlapping
    intervals in either set are unified before calculation.

    Parameters
    ----------
    intervals1 : DataFrame
        The covering set of 1D intervals (chrom, start, end).
    intervals2 : DataFrame or None, default None
        The reference space to measure against. ``None`` means the
        entire genome.

    Returns
    -------
    float
        A value between 0.0 and 1.0 representing the fraction of
        *intervals2* (or the genome) covered by *intervals1*.

    See Also
    --------
    gintervals_covered_bp : Total base pairs covered by intervals.
    gintervals_intersect : Intersection of two interval sets.
    gintervals_all : Return full-genome intervals.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs1 = pm.gintervals("1", 0, 100000)
    >>> intervs2 = pm.gintervals(["1", "2"], 0, [100000, 100000])
    >>> pm.gintervals_coverage_fraction(intervs1, intervs2)  # doctest: +SKIP
    >>> pm.gintervals_coverage_fraction(intervs1)  # doctest: +SKIP
    """
    intervals1 = _resolve_intervals(intervals1)
    if intervals2 is not None:
        intervals2 = _resolve_intervals(intervals2)
    if intervals1 is None:
        raise ValueError("intervals1 cannot be None")
    if len(intervals1) == 0:
        return 0.0
    if intervals2 is None:
        intervals2 = gintervals_all()
    if len(intervals2) == 0:
        return 0.0

    total_bp = gintervals_covered_bp(intervals2)
    if total_bp == 0:
        return 0.0

    intersection = gintervals_intersect(intervals1, intervals2)
    if intersection is None or len(intersection) == 0:
        return 0.0

    covered_bp = gintervals_covered_bp(intersection)
    return covered_bp / total_bp


_NEIGHBORS_2D_BRUTE_LIMIT = 100_000_000


def _neighbors_2d_via_cpp_nn(
    i1: pd.DataFrame,
    i2: pd.DataFrame,
    maxneighbors: int,
    mindist1: int,
    maxdist1: int,
    mindist2: int,
    maxdist2: int,
    na_if_notfound: bool,
) -> pd.DataFrame | None:
    """Unbounded / huge-input 2D NN: build a C++ MemQuadTree per chrom-pair
    and call the NN iterator binding (R parity).

    The bounded path uses _quadtree.QuadTree's expanded-rect query, which
    is faster for tiny inputs.  This path scales to multi-million rect
    inputs that would blow the brute-force budget; it is also correct
    for bounded windows (we pass mindist/maxdist through and the NN
    iterator early-breaks at the Manhattan cutoff).
    """
    np = _numpy

    qx1 = i1["start1"].to_numpy(np.int64)
    qy1 = i1["start2"].to_numpy(np.int64)
    qx2 = i1["end1"].to_numpy(np.int64)
    qy2 = i1["end2"].to_numpy(np.int64)
    tx1 = i2["start1"].to_numpy(np.int64)
    ty1 = i2["start2"].to_numpy(np.int64)
    tx2 = i2["end1"].to_numpy(np.int64)
    ty2 = i2["end2"].to_numpy(np.int64)

    g1_idx = i1.groupby(["chrom1", "chrom2"], observed=True).indices
    g2_idx = i2.groupby(["chrom1", "chrom2"], observed=True).indices

    out_q: list[int] = []
    out_t: list[int] = []
    out_d1: list[float] = []
    out_d2: list[float] = []

    queries_with_any_match: set[int] = set()

    for key, q_pos in g1_idx.items():
        t_pos = g2_idx.get(key)
        if t_pos is None or len(t_pos) == 0:
            continue
        q_pos_arr = np.asarray(q_pos, dtype=np.int64)
        t_pos_arr = np.asarray(t_pos, dtype=np.int64)
        res = _pymisha.pm_neighbors_2d(
            qx1[q_pos_arr], qy1[q_pos_arr], qx2[q_pos_arr], qy2[q_pos_arr],
            tx1[t_pos_arr], ty1[t_pos_arr], tx2[t_pos_arr], ty2[t_pos_arr],
            int(maxneighbors),
            int(mindist1), int(maxdist1),
            int(mindist2), int(maxdist2),
        )
        if len(res["q_local_idx"]) == 0:
            continue
        q_local = res["q_local_idx"]
        t_local = res["t_local_idx"]
        d1_arr = res["dist1"]
        d2_arr = res["dist2"]
        q_global = q_pos_arr[q_local]
        t_global = t_pos_arr[t_local]
        for j in range(len(q_global)):
            out_q.append(int(q_global[j]))
            out_t.append(int(t_global[j]))
            out_d1.append(float(d1_arr[j]))
            out_d2.append(float(d2_arr[j]))
            queries_with_any_match.add(int(q_global[j]))

    # NA-if-notfound: every query in i1 that produced zero matches.
    if na_if_notfound:
        for j in range(len(i1)):
            if j not in queries_with_any_match:
                out_q.append(j)
                out_t.append(-1)
                out_d1.append(float("nan"))
                out_d2.append(float("nan"))

    if not out_q:
        return None

    # Sort by R's IntervNeighbor2D order: (id1, |dist1+dist2|, id2).
    order = sorted(
        range(len(out_q)),
        key=lambda i: (
            out_q[i],
            abs((out_d1[i] + out_d2[i]) if out_t[i] >= 0 else 0.0),
            out_t[i],
        ),
    )
    id1_arr = [out_q[i] for i in order]
    id2_arr = [out_t[i] for i in order]
    d1_arr = [out_d1[i] for i in order]
    d2_arr = [out_d2[i] for i in order]

    # Build the output mirroring the bounded path (same column-rename and
    # final DataFrame shape).
    left = i1.iloc[id1_arr].reset_index(drop=True)
    used = set(i1.columns)
    rename2: dict[str, str] = {}
    for col in i2.columns:
        new = col
        while new in used:
            new = new + "1"
        rename2[col] = new
        used.add(new)
    # For NA rows, take the first i2 row as a template and overwrite to NA.
    if i2.shape[0] > 0:
        right = i2.iloc[[max(k, 0) for k in id2_arr]].reset_index(drop=True)
        right.rename(columns=rename2, inplace=True)
        for col in rename2.values():
            mask = [k < 0 for k in id2_arr]
            if any(mask):
                if pd.api.types.is_numeric_dtype(right[col]):
                    right.loc[mask, col] = float("nan")
                else:
                    right.loc[mask, col] = pd.NA
    else:
        right = pd.DataFrame()
    res_df = pd.concat([left, right], axis=1)
    res_df["dist1"] = d1_arr
    res_df["dist2"] = d2_arr
    return res_df


def _neighbors_2d(
    i1: pd.DataFrame,
    i2: pd.DataFrame,
    maxneighbors: int,
    mindist1: float,
    maxdist1: float,
    mindist2: float,
    maxdist2: float,
    na_if_notfound: bool,
) -> pd.DataFrame | None:
    """2D nearest-neighbor search between two 2D-interval sets (R parity).

    For each rectangle of *i1*, finds up to *maxneighbors* rectangles of *i2* on
    the SAME chrom-pair whose per-axis unsigned gaps ``(dist1, dist2)`` lie in
    ``[mindist1, maxdist1] x [mindist2, maxdist2]``, ordered by Manhattan
    distance ``dist1 + dist2`` (then by target index).  Mirrors R's
    ``gfind_neighbors`` 2D branch (``StatQuadTree::NNIterator``); the result is
    sorted by ``(query index, dist1 + dist2, target index)``.

    A bounded distance window uses the in-memory quadtree (expanded-rectangle
    query); an unbounded window (the default ``maxdist = 1e9`` sentinel) falls
    back to a per-chrom-pair brute force and raises ``NotImplementedError`` when
    that would exceed a safety budget (a scalable quadtree NN iterator for huge
    unbounded sets is not yet ported).
    """
    from ._quadtree import QuadTree

    # R parity: callers can hand 2D inputs whose chrom names are mixed (one
    # side carries a ``chr`` prefix the DB doesn't store, or vice versa).  The
    # downstream per-chrom-pair grouping is a literal string compare, so
    # normalise both sides first (mirrors `gintervals_neighbors` 1D behaviour
    # via `_df2pymisha`'s implicit normalisation).
    i1 = _normalize_interval_df(i1.copy())
    i2 = _normalize_interval_df(i2.copy())

    np = _numpy
    _BIG = 1e9
    bounded = maxdist1 < _BIG and maxdist2 < _BIG

    def _axis_gap(qa1: int, qa2: int, ta1: int, ta2: int) -> int:
        if qa1 >= ta2:
            return qa1 - ta2
        if qa2 <= ta1:
            return ta1 - qa2
        return 0

    c1q = i1["chrom1"].astype(str).to_numpy()
    c2q = i1["chrom2"].astype(str).to_numpy()
    qx1 = i1["start1"].to_numpy(np.int64)
    qx2 = i1["end1"].to_numpy(np.int64)
    qy1 = i1["start2"].to_numpy(np.int64)
    qy2 = i1["end2"].to_numpy(np.int64)

    tx1 = i2["start1"].to_numpy(np.int64)
    tx2 = i2["end1"].to_numpy(np.int64)
    ty1 = i2["start2"].to_numpy(np.int64)
    ty2 = i2["end2"].to_numpy(np.int64)
    g2_indices = i2.groupby(["chrom1", "chrom2"], observed=True).indices

    # Unbounded windows (and large bounded ones): use the C++ NN iterator
    # per chrom-pair (Phase NN of KICKOFF-5 deferred backlog).
    if not bounded:
        return _neighbors_2d_via_cpp_nn(
            i1, i2,
            int(maxneighbors),
            int(mindist1), int(maxdist1),
            int(mindist2), int(maxdist2),
            na_if_notfound,
        )

    pair_tree: dict[tuple, Any] = {}

    def _candidates(key: tuple, j: int) -> Any:
        t_pos = g2_indices.get(key)
        if t_pos is None:
            return None
        if not bounded:
            return t_pos
        tree = pair_tree.get(key)
        if tree is None:
            bx = int(max(int(tx2[t_pos].max()), int(qx2.max()))) + 1
            by = int(max(int(ty2[t_pos].max()), int(qy2.max()))) + 1
            tree = QuadTree(0, 0, bx, by, is_points=False)
            for k in t_pos:
                tree.insert((int(tx1[k]), int(ty1[k]), int(tx2[k]), int(ty2[k]), 0.0))
            # Map local quadtree object index -> global i2 index.
            pair_tree[key] = (tree, np.asarray(t_pos, dtype=np.int64))
            tree, t_pos_arr = pair_tree[key]
        else:
            tree, t_pos_arr = tree
        ex1 = int(qx1[j]) - int(maxdist1) - 1
        ey1 = int(qy1[j]) - int(maxdist2) - 1
        ex2 = int(qx2[j]) + int(maxdist1) + 1
        ey2 = int(qy2[j]) + int(maxdist2) + 1
        local = tree.query(ex1, ey1, ex2, ey2)
        return t_pos_arr[local] if len(local) else np.empty(0, dtype=np.int64)

    out_id1: list[int] = []
    out_id2: list[int] = []
    out_d1: list[float] = []
    out_d2: list[float] = []

    for j in range(len(i1)):
        key = (c1q[j], c2q[j])
        cands = _candidates(key, j)
        kept: list[tuple[int, int, int, int]] = []
        if cands is not None:
            for k in cands:
                k = int(k)
                d1 = _axis_gap(int(qx1[j]), int(qx2[j]), int(tx1[k]), int(tx2[k]))
                d2 = _axis_gap(int(qy1[j]), int(qy2[j]), int(ty1[k]), int(ty2[k]))
                if mindist1 <= d1 <= maxdist1 and mindist2 <= d2 <= maxdist2:
                    kept.append((d1 + d2, k, d1, d2))
        kept.sort(key=lambda t: (t[0], t[1]))
        if kept:
            for (_m, k, d1, d2) in kept[:maxneighbors]:
                out_id1.append(j)
                out_id2.append(k)
                out_d1.append(float(d1))
                out_d2.append(float(d2))
        elif na_if_notfound:
            out_id1.append(j)
            out_id2.append(-1)
            out_d1.append(float("nan"))
            out_d2.append(float("nan"))

    if not out_id1:
        return None

    # Sort by R's IntervNeighbor2D order: (id1, |dist1+dist2|, id2).
    order = sorted(
        range(len(out_id1)),
        key=lambda i: (
            out_id1[i],
            abs((out_d1[i] + out_d2[i]) if out_id2[i] >= 0 else 0.0),
            out_id2[i],
        ),
    )
    id1_arr = [out_id1[i] for i in order]
    id2_arr = [out_id2[i] for i in order]
    d1_arr = [out_d1[i] for i in order]
    d2_arr = [out_d2[i] for i in order]

    # Build the output: i1 columns, then i2 columns (collisions get a "1"
    # suffix, R make.unique style), then dist1, dist2.
    left = i1.iloc[id1_arr].reset_index(drop=True)
    used = set(i1.columns)
    rename2: dict[str, str] = {}
    for col in i2.columns:
        new = col
        while new in used:
            new = new + "1"
        rename2[col] = new
        used.add(new)
    has_na = any(t < 0 for t in id2_arr)
    safe_id2 = [t if t >= 0 else 0 for t in id2_arr]
    right = i2.iloc[safe_id2].reset_index(drop=True).rename(columns=rename2)
    out = _pandas.concat([left, right], axis=1)
    if has_na:
        na_mask = _numpy.array([t < 0 for t in id2_arr])
        for col in rename2.values():
            if out[col].dtype.kind in "iu":
                out[col] = out[col].astype(float)
            out.loc[na_mask, col] = _numpy.nan
    out["dist1"] = d1_arr
    out["dist2"] = d2_arr
    return out.reset_index(drop=True)


def gintervals_neighbors(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    maxneighbors: int = 1,
    mindist: float = -1e9,
    maxdist: float = 1e9,
    mindist1: float = -1e9,
    maxdist1: float = 1e9,
    mindist2: float = -1e9,
    maxdist2: float = 1e9,
    na_if_notfound: bool = False,
    use_intervals1_strand: bool = False,
    warn_ignored_strand: bool = True,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Find nearest neighbors between two sets of intervals.

    For each interval in intervals1, finds the closest intervals from intervals2.
    Distance directionality can be determined by either the strand of the target
    intervals (intervals2, default) or the query intervals (intervals1).

    Parameters
    ----------
    intervals1 : DataFrame
        Query intervals with columns 'chrom', 'start', 'end' (and optionally 'strand').
    intervals2 : DataFrame
        Target intervals to search for neighbors.
    maxneighbors : int, default 1
        Maximum number of neighbors to return per query interval.
    mindist : float, default -1e9
        Minimum 1D distance (negative means target is upstream/left of query).
    maxdist : float, default 1e9
        Maximum 1D distance (positive means target is downstream/right of query).
    mindist1, maxdist1, mindist2, maxdist2 : float, optional
        Per-dimension distance ranges for **2D** intervals. Accepted for
        R-misha API parity. PyMisha currently does not implement 2D neighbor
        search, so any 2D input raises ``NotImplementedError``.
    na_if_notfound : bool, default False
        If True, include queries with no neighbors (with NA values).
    use_intervals1_strand : bool, default False
        If True, use intervals1 strand column for distance directionality
        instead of intervals2 strand. This is useful for TSS analysis where
        you want upstream/downstream distances relative to gene direction.
        When True:
        - + strand queries: negative distance = upstream, positive = downstream
        - - strand queries: negative distance = downstream, positive = upstream
    warn_ignored_strand : bool, default True
        Emit a warning when ``intervals1`` carries a ``strand`` column and
        ``use_intervals1_strand=False`` (the strand would be silently
        ignored otherwise).
    intervals_set_out : str, optional
        Name of an interval set to write the result into. When supplied,
        the result is saved with ``gintervals_save(...)`` and the function
        returns ``None``.

    Returns
    -------
    DataFrame or None
        DataFrame with query and neighbor coordinates plus distance column.
        ``None`` if no neighbors are found, or if *intervals_set_out* was
        provided.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> query = pm.gintervals("1", [5000], [5100])
    >>> targets = pm.gintervals("1", [3000, 7000], [3100, 7100])
    >>> pm.gintervals_neighbors(query, targets)  # doctest: +SKIP

    See Also
    --------
    gintervals_neighbors_upstream : Find upstream neighbors only.
    gintervals_neighbors_downstream : Find downstream neighbors only.
    gintervals_neighbors_directional : Find both upstream and downstream.
    gintervals_annotate : Annotate intervals with nearest-neighbor columns.
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    # 2D track names (e.g. "test.generated_2d_5"): _resolve_intervals only
    # handles named intervals SETS; for 2D track names we materialize their
    # rectangles via gextract over the 2D ALLGENOME(full) scope.  R parity:
    # gintervals.neighbors(track_2d, ...) loads the track's rectangles.
    def _resolve_2d_track_name(x):
        if not isinstance(x, str):
            return x
        from .tracks import gtrack_exists, gtrack_info
        if not gtrack_exists(x):
            return x
        info = gtrack_info(x)
        if int(info.get("dimensions", 1) or 1) != 2:
            return x
        from .extract import gextract
        scope = gintervals_2d_all(mode="full")
        df = gextract(x, scope)
        if df is None:
            return x
        return df[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()

    intervals1 = _resolve_2d_track_name(intervals1)
    intervals2 = _resolve_2d_track_name(intervals2)
    _checkroot()

    if intervals1 is None:
        raise ValueError("intervals1 cannot be None")
    if intervals2 is None:
        raise ValueError("intervals2 cannot be None")

    if _is_2d_intervals_df(intervals1) or _is_2d_intervals_df(intervals2):
        if not (_is_2d_intervals_df(intervals1) and _is_2d_intervals_df(intervals2)):
            raise ValueError("Cannot intermix 1D and 2D intervals")
        if maxneighbors < 1:
            raise ValueError("maxneighbors must be >= 1")
        if mindist1 > maxdist1 or mindist2 > maxdist2:
            raise ValueError("mindist exceeds maxdist")
        # R returns NULL when an upper bound is negative.
        if maxdist1 < 0 or maxdist2 < 0 or len(intervals1) == 0:
            return None
        df = _neighbors_2d(
            intervals1, intervals2, int(maxneighbors),
            float(mindist1), float(maxdist1), float(mindist2), float(maxdist2),
            na_if_notfound,
        )
        if intervals_set_out is not None:
            if df is not None and len(df) > 0:
                gintervals_save(df, intervals_set_out)
            return None
        return df

    if mindist1 != -1e9 or maxdist1 != 1e9 or mindist2 != -1e9 or maxdist2 != 1e9:
        # R behaviour: for 1D input these are accepted but unused; we mirror
        # that rather than erroring so R-script ports work.
        pass

    if maxneighbors < 1:
        raise ValueError("maxneighbors must be >= 1")

    if mindist > maxdist:
        raise ValueError("mindist must be <= maxdist")

    if (
        warn_ignored_strand
        and not use_intervals1_strand
        and isinstance(intervals1, _pandas.DataFrame)
        and "strand" in intervals1.columns
    ):
        import warnings as _warnings
        _warnings.warn(
            "intervals1 contains a 'strand' column that will be ignored for "
            "distance calculation. Set use_intervals1_strand=True to use it, "
            "or warn_ignored_strand=False to suppress this warning.",
            stacklevel=2,
        )

    if len(intervals1) == 0:
        if intervals_set_out is not None:
            return None
        return None
    if len(intervals2) == 0 and not na_if_notfound:
        if intervals_set_out is not None:
            return None
        return None

    result = _pymisha.pm_find_neighbors(
        _df2pymisha(intervals1),
        _df2pymisha(intervals2),
        int(maxneighbors),
        float(mindist),
        float(maxdist),
        int(na_if_notfound),
        int(use_intervals1_strand)
    )

    df = _pymisha2df(result)

    if (
        na_if_notfound
        and df is not None
        and len(df) > 0
        and isinstance(intervals1, _pandas.DataFrame)
    ):
        # The C++ layer stores a -1 long sentinel for the target start/end of
        # rows with no neighbor (it can't put NaN in an int column). R misha
        # returns NaN there, so coerce those coordinates to NaN to match.
        # Not-found rows are exactly those whose target chrom became NaN.
        n_left = len(intervals1.columns)
        target_cols = list(df.columns[n_left:-1])  # intervals2 cols (excl. dist)
        if target_cols:
            notfound = df[target_cols[0]].isna()
            if notfound.any():
                for col in target_cols:
                    if df[col].dtype.kind in "iu":
                        df[col] = df[col].astype(float)
                    df.loc[notfound, col] = _numpy.nan

    if intervals_set_out is not None:
        if df is not None and len(df) > 0:
            gintervals_save(df, intervals_set_out)
        return None
    return df


def _is_2d_intervals_df(intervals: Any) -> bool:
    """True if *intervals* is a 2D-shaped DataFrame (has ``chrom1``)."""
    return (
        isinstance(intervals, _pandas.DataFrame)
        and "chrom1" in intervals.columns
        and "chrom2" in intervals.columns
    )


def gintervals_neighbors_upstream(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    maxneighbors: int = 1,
    maxdist: float = 1e9,
    na_if_notfound: bool = False,
) -> pd.DataFrame | None:
    """
    Find upstream neighbors of query intervals using strand directionality.

    Upstream neighbors are those located in the 5' direction relative to the
    query strand: left (negative distance) for + strand queries, right (positive
    distance) for - strand queries.

    Parameters
    ----------
    intervals1 : DataFrame
        Query intervals. If 'strand' column is present, it determines direction.
        Missing or strand=0 is treated as + strand.
    intervals2 : DataFrame
        Target intervals to search for neighbors.
    maxneighbors : int, default 1
        Maximum number of upstream neighbors to return per query.
    maxdist : float, default 1e9
        Maximum distance to search for neighbors (in bp).
    na_if_notfound : bool, default False
        If True, include queries with no neighbors (with NA values).

    Returns
    -------
    DataFrame or None
        DataFrame with query and neighbor coordinates plus distance column.
        Distance values are always <= 0 (upstream direction).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> query = pm.gintervals("1", [5000], [5100])
    >>> query["strand"] = 1  # + strand
    >>> targets = pm.gintervals("1", [3000, 7000], [3100, 7100])
    >>> pm.gintervals_neighbors_upstream(query, targets)  # doctest: +SKIP

    See Also
    --------
    gintervals_neighbors : General neighbor finding.
    gintervals_neighbors_downstream : Find downstream neighbors.
    gintervals_neighbors_directional : Find both upstream and downstream.
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    # Upstream: mindist=-maxdist, maxdist=0, use_intervals1_strand=True
    return gintervals_neighbors(
        intervals1, intervals2,
        maxneighbors=maxneighbors,
        mindist=-maxdist, maxdist=0,
        na_if_notfound=na_if_notfound,
        use_intervals1_strand=True,
        warn_ignored_strand=False,
    )


def gintervals_neighbors_downstream(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    maxneighbors: int = 1,
    maxdist: float = 1e9,
    na_if_notfound: bool = False,
) -> pd.DataFrame | None:
    """
    Find downstream neighbors of query intervals using strand directionality.

    Downstream neighbors are those located in the 3' direction relative to the
    query strand: right (positive distance) for + strand queries, left (negative
    distance) for - strand queries.

    Parameters
    ----------
    intervals1 : DataFrame
        Query intervals. If 'strand' column is present, it determines direction.
        Missing or strand=0 is treated as + strand.
    intervals2 : DataFrame
        Target intervals to search for neighbors.
    maxneighbors : int, default 1
        Maximum number of downstream neighbors to return per query.
    maxdist : float, default 1e9
        Maximum distance to search for neighbors (in bp).
    na_if_notfound : bool, default False
        If True, include queries with no neighbors (with NA values).

    Returns
    -------
    DataFrame or None
        DataFrame with query and neighbor coordinates plus distance column.
        Distance values are always >= 0 (downstream direction).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> query = pm.gintervals("1", [5000], [5100])
    >>> query["strand"] = 1  # + strand
    >>> targets = pm.gintervals("1", [3000, 7000], [3100, 7100])
    >>> pm.gintervals_neighbors_downstream(query, targets)  # doctest: +SKIP

    See Also
    --------
    gintervals_neighbors : General neighbor finding.
    gintervals_neighbors_upstream : Find upstream neighbors.
    gintervals_neighbors_directional : Find both upstream and downstream.
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    # Downstream: mindist=0, maxdist=maxdist, use_intervals1_strand=True
    return gintervals_neighbors(
        intervals1, intervals2,
        maxneighbors=maxneighbors,
        mindist=0, maxdist=maxdist,
        na_if_notfound=na_if_notfound,
        use_intervals1_strand=True,
        warn_ignored_strand=False,
    )


def gintervals_neighbors_directional(
    intervals1: pd.DataFrame | str,
    intervals2: pd.DataFrame | str,
    maxneighbors_upstream: int = 1,
    maxneighbors_downstream: int = 1,
    maxdist: float = 1e9,
    na_if_notfound: bool = False,
) -> dict[str, pd.DataFrame | None]:
    """
    Find both upstream and downstream neighbors of query intervals.

    Convenience function that returns both upstream and downstream neighbors
    in a single call.

    Parameters
    ----------
    intervals1 : DataFrame
        Query intervals. If 'strand' column is present, it determines direction.
        Missing or strand=0 is treated as + strand.
    intervals2 : DataFrame
        Target intervals to search for neighbors.
    maxneighbors_upstream : int, default 1
        Maximum number of upstream neighbors to return per query.
    maxneighbors_downstream : int, default 1
        Maximum number of downstream neighbors to return per query.
    maxdist : float, default 1e9
        Maximum distance to search for neighbors (in bp).
    na_if_notfound : bool, default False
        If True, include queries with no neighbors (with NA values).

    Returns
    -------
    dict
        Dictionary with keys 'upstream' and 'downstream', each containing
        a DataFrame (or None) with neighbor results.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> query = pm.gintervals("1", [5000], [5100])
    >>> query["strand"] = 1
    >>> targets = pm.gintervals("1", [3000, 7000], [3100, 7100])
    >>> result = pm.gintervals_neighbors_directional(query, targets)
    >>> result["upstream"]   # doctest: +SKIP
    >>> result["downstream"]  # doctest: +SKIP

    See Also
    --------
    gintervals_neighbors : General neighbor finding.
    gintervals_neighbors_upstream : Find upstream neighbors only.
    gintervals_neighbors_downstream : Find downstream neighbors only.
    """
    intervals1 = _resolve_intervals(intervals1)
    intervals2 = _resolve_intervals(intervals2)
    upstream = gintervals_neighbors_upstream(
        intervals1, intervals2,
        maxneighbors=maxneighbors_upstream,
        maxdist=maxdist,
        na_if_notfound=na_if_notfound
    )

    downstream = gintervals_neighbors_downstream(
        intervals1, intervals2,
        maxneighbors=maxneighbors_downstream,
        maxdist=maxdist,
        na_if_notfound=na_if_notfound
    )

    return {"upstream": upstream, "downstream": downstream}


def gintervals_ls(pattern: str = "", ignore_case: bool = False) -> list[str]:
    """
    List named interval sets in the database.

    Parameters
    ----------
    pattern : str, default ""
        Regular expression pattern to filter interval set names.
        Empty string matches all sets.
    ignore_case : bool, default False
        If True, pattern matching is case-insensitive.

    Returns
    -------
    list of str
        Names of interval sets matching the pattern.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_ls()  # doctest: +SKIP
    >>> pm.gintervals_ls("annot.*")  # doctest: +SKIP

    See Also
    --------
    gintervals_exists : Check if a named interval set exists.
    gintervals_load : Load a named interval set.
    gintervals_save : Save intervals as a named set.
    gintervals_rm : Remove a named interval set.
    """
    _checkroot()

    # C++ caches interval-set names alongside tracks during gdb_init / gdb_reload.
    # Falls back to a filesystem walk for the few callers that may run before
    # full db init (e.g. dataset bootstrap tests).
    try:
        interval_set_names = _pymisha.pm_interv_names()
    except Exception:
        from . import _shared
        assert _shared._GROOT is not None

        roots: list[str] = []
        if _shared._UROOT:
            roots.append(_shared._UROOT)
        roots.append(_shared._GROOT)
        roots.extend(_shared._GDATASETS)

        seen: set[str] = set()
        for root in roots:
            tracks_dir = Path(root) / "tracks"
            if not tracks_dir.exists():
                continue
            for suffix in (".interv", ".interv2d"):
                for interv_file in tracks_dir.rglob(f"*{suffix}"):
                    rel_path = interv_file.relative_to(tracks_dir)
                    name = str(rel_path)[:-len(suffix)].replace("/", ".").replace("\\", ".")
                    seen.add(name)
        interval_set_names = list(seen)

    interval_sets: list[str] = sorted(set(interval_set_names))

    # Apply pattern filter
    if pattern:
        flags = re.IGNORECASE if ignore_case else 0
        interval_sets = [s for s in interval_sets if re.search(pattern, s, flags)]

    return interval_sets


def gintervals_dbs(
    intervals: str | list[str],
    dataframe: bool = False,
) -> dict[str, list[str]] | pd.DataFrame:
    """
    Return database root(s) containing the given interval set(s).

    For each interval set name, searches the current database root and
    all loaded dataset roots to find which databases contain the set.

    Parameters
    ----------
    intervals : str or list of str
        Interval set name(s) to look up.
    dataframe : bool, default False
        If True, return a DataFrame with columns ``intervals`` and ``db``.
        If False, return a dict mapping set names to lists of database paths.

    Returns
    -------
    dict or DataFrame
        If *dataframe* is False, a dict ``{set_name: [db_path, ...]}``.
        If *dataframe* is True, a DataFrame with columns ``intervals`` and ``db``.

    See Also
    --------
    gintervals_ls : List available interval sets.
    gintervals_exists : Check if a named interval set exists.
    gtrack_dbs : Same for tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_dbs("annotations1")  # doctest: +SKIP
    {'annotations1': ['/path/to/db']}
    """
    _checkroot()
    from . import _shared
    assert _shared._GROOT is not None

    names = [intervals] if isinstance(intervals, str) else list(intervals)

    all_dbs: list[str] = [_shared._GROOT] + list(_shared._GDATASETS)

    result = {}
    for name in names:
        rel_base = os.path.join("tracks", name.replace(".", os.sep))
        dbs = []
        for db in all_dbs:
            if (os.path.exists(os.path.join(db, rel_base + ".interv"))
                    or os.path.exists(os.path.join(db, rel_base + ".interv2d"))):
                dbs.append(db)
        result[name] = dbs

    if dataframe:
        rows_name = []
        rows_db = []
        for n, dbs in result.items():
            for db in dbs:
                rows_name.append(n)
                rows_db.append(db)
        import pandas as pd
        return pd.DataFrame({"intervals": rows_name, "db": rows_db})

    return result


def gintervals_exists(name: str) -> bool:
    """
    Check if a named interval set exists.

    Parameters
    ----------
    name : str
        Name of the interval set to check.

    Returns
    -------
    bool
        True if the interval set exists, False otherwise.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_exists("annotations")
    True

    See Also
    --------
    gintervals_ls : List named interval sets.
    gintervals_load : Load a named interval set.
    gintervals_save : Save intervals as a named set.
    gintervals_rm : Remove a named interval set.
    """
    _checkroot()
    return gintervals_dataset(name) is not None


def gintervals_path(name: str) -> str:
    """
    Return the filesystem path of a named interval set's directory.

    Parameters
    ----------
    name : str
        Name of the interval set (e.g. ``"annotations"``).

    Returns
    -------
    str
        Absolute path to the interval set on disk (the ``.interv``
        or ``.interv2d`` file/directory).

    Raises
    ------
    ValueError
        If *name* is ``None`` or the interval set does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_path("annotations")  # doctest: +ELLIPSIS
    '...annotations.interv'

    See Also
    --------
    gintervals_exists : Check whether an interval set exists.
    gintervals_dataset : Get dataset root for an interval set.
    gintervals_load : Load a named interval set.
    """
    if name is None:
        raise ValueError("name cannot be None")
    _checkroot()
    root = gintervals_dataset(name)
    if root is None:
        raise ValueError(f"Interval set '{name}' does not exist")
    path_part = name.replace(".", "/")
    for suffix in (".interv", ".interv2d"):
        candidate = os.path.join(root, "tracks", f"{path_part}{suffix}")
        if os.path.exists(candidate):
            return candidate
    # Should not happen if gintervals_dataset found it, but be safe
    raise ValueError(f"Interval set '{name}' does not exist")


def gintervals_dataset(intervals: str | None = None) -> str | None:
    """
    Return the database/dataset root path for a named interval set.

    Searches the user root, genome root, and all linked datasets for
    the given interval set name.

    Parameters
    ----------
    intervals : str
        Name of the interval set (e.g. ``"annotations"``).

    Returns
    -------
    str or None
        The root path of the database/dataset containing the interval
        set, or ``None`` if the set is not found.

    Raises
    ------
    ValueError
        If *intervals* is ``None``.

    See Also
    --------
    gintervals_exists : Check if a named interval set exists.
    gintervals_ls : List named interval sets.
    gintervals_load : Load a named interval set.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_dataset("annotations")  # doctest: +ELLIPSIS
    '...trackdb/test'
    """
    if intervals is None:
        raise ValueError("intervals cannot be None")
    _checkroot()
    from . import _shared
    assert _shared._GROOT is not None

    roots: list[str] = []
    if _shared._UROOT:
        roots.append(_shared._UROOT)
    roots.append(_shared._GROOT)
    roots.extend(reversed(_shared._GDATASETS))

    path_part = intervals.replace(".", "/")
    for root in roots:
        for suffix in (".interv", ".interv2d"):
            if (Path(root) / "tracks" / f"{path_part}{suffix}").exists():
                return root
    return None


def gintervals_chrom_sizes(intervals: pd.DataFrame | str) -> pd.DataFrame:
    """
    Count intervals per chromosome (1D) or chromosome pair (2D).

    Parameters
    ----------
    intervals : DataFrame or str
        Intervals (with a ``chrom`` or ``chrom1``/``chrom2`` column), or the
        name of an interval set or track (resolved to its intervals).

    Returns
    -------
    DataFrame
        DataFrame with 'chrom' column containing unique chromosomes present
        in the input intervals.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervals = pm.gintervals(["1", "2"], [0, 0], [10000, 20000])
    >>> pm.gintervals_chrom_sizes(intervals)  # doctest: +SKIP

    See Also
    --------
    gintervals_load : Load a named interval set.
    gintervals_exists : Check if a named interval set exists.
    gintervals_ls : List named interval sets.
    """
    if isinstance(intervals, str):
        from .extract import _maybe_load_intervals_set

        resolved = _maybe_load_intervals_set(intervals)
        if isinstance(resolved, str):
            raise ValueError(f"Interval set or track '{intervals}' does not exist")
        intervals = resolved

    if intervals is None or len(intervals) == 0:
        return _pandas.DataFrame(columns=["chrom"])

    # Count intervals per chromosome (1D) or chromosome pair (2D), as R does.
    if "chrom" in intervals.columns:
        return (
            intervals.assign(chrom=intervals["chrom"].astype(str))
            .groupby("chrom", observed=True)
            .size()
            .reset_index(name="size")
            .sort_values("chrom")
            .reset_index(drop=True)
        )
    if "chrom1" in intervals.columns:
        return (
            intervals.assign(
                chrom1=intervals["chrom1"].astype(str),
                chrom2=intervals["chrom2"].astype(str),
            )
            .groupby(["chrom1", "chrom2"], observed=True)
            .size()
            .reset_index(name="size")
            .sort_values(["chrom1", "chrom2"])
            .reset_index(drop=True)
        )
    raise ValueError("intervals must have 'chrom' or 'chrom1'/'chrom2' columns")


def _read_serialized_dataframe(payload: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".rds") as tmp:
        tmp.write(payload)
        tmp.flush()
        return _decode_r_obj_to_bytes(tmp.name)


def _load_serialized_dataframe(path: str | Path) -> pd.DataFrame:
    return _decode_r_obj_to_bytes(path)


def _resolve_chrom_file(path: Path, chrom: str) -> Path | None:
    candidate = path / chrom
    if candidate.exists():
        return candidate
    if chrom.startswith("chr"):
        candidate = path / chrom[3:]
        if candidate.exists():
            return candidate
    else:
        candidate = path / f"chr{chrom}"
        if candidate.exists():
            return candidate
    return None


def _resolve_pair_file(path: Path, chrom1: str, chrom2: str) -> Path | None:
    candidate = path / f"{chrom1}-{chrom2}"
    if candidate.exists():
        return candidate
    chrom1_alt = chrom1[3:] if chrom1.startswith("chr") else f"chr{chrom1}"
    chrom2_alt = chrom2[3:] if chrom2.startswith("chr") else f"chr{chrom2}"
    candidate = path / f"{chrom1_alt}-{chrom2}"
    if candidate.exists():
        return candidate
    candidate = path / f"{chrom1}-{chrom2_alt}"
    if candidate.exists():
        return candidate
    candidate = path / f"{chrom1_alt}-{chrom2_alt}"
    if candidate.exists():
        return candidate
    return None


def _chrom_id_map() -> dict[str, int]:
    chroms = gintervals_all()["chrom"].tolist()
    return {chrom: idx for idx, chrom in enumerate(chroms)}


def _chrom_id_lookup(chrom_map: dict[str, int], chrom_name: str) -> int | None:
    if chrom_name in chrom_map:
        return chrom_map[chrom_name]
    alt = chrom_name[3:] if chrom_name.startswith("chr") else f"chr{chrom_name}"
    return chrom_map.get(alt)


def _indexed_entries_by_chrom(
    entries: list[tuple[int, int, int]],
) -> dict[int, tuple[int, int]]:
    return {chrom_id: (offset, length) for chrom_id, offset, length in entries}


def _indexed_entries_by_pair(
    entries: list[tuple[int, int, int, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    return {
        (chrom1_id, chrom2_id): (offset, length)
        for chrom1_id, chrom2_id, offset, length in entries
    }


def _read_indexed_entry(dat_path: Path, offset: int, length: int) -> pd.DataFrame | None:
    if length == 0:
        return None
    with open(dat_path, "rb") as fh:
        fh.seek(offset)
        payload = fh.read(length)
    return _read_serialized_dataframe(payload)


def _intervset_loadable(
    stats: pd.DataFrame | None,
    max_size: int | None,
    label: str,
    chrom: str | None = None,
    chrom1: str | None = None,
    chrom2: str | None = None,
) -> tuple[bool, str | None]:
    if max_size is None:
        return True, None
    if stats is None or len(stats) == 0 or "size" not in stats.columns:
        return True, None
    total = int(stats["size"].sum())
    if total <= max_size:
        return True, None
    if chrom is not None:
        return False, (
            f"Cannot load chromosome {chrom} of an intervals set {label}: its size "
            f"({total}) exceeds the limit ({max_size}) controlled by max_data_size."
        )
    if chrom1 is not None and chrom2 is not None:
        return False, (
            f"Cannot load chromosome pair ({chrom1}, {chrom2}) of an intervals set {label}: "
            f"its size ({total}) exceeds the limit ({max_size}) controlled by max_data_size."
        )
    if chrom1 is not None:
        return False, (
            f"Cannot load chromosome {chrom1} of an intervals set {label}: its size "
            f"({total}) exceeds the limit ({max_size}) controlled by max_data_size."
        )
    if chrom2 is not None:
        return False, (
            f"Cannot load chromosome {chrom2} of an intervals set {label}: its size "
            f"({total}) exceeds the limit ({max_size}) controlled by max_data_size."
        )
    return False, (
        f"Cannot load a big intervals set {label}: its size ({total}) exceeds the limit ({max_size}) "
        "controlled by max_data_size. For big intervals sets only one chromosome pair can be loaded at a time."
    )


def _normalize_chrom_column(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = _normalize_chroms(df[col].astype(str).tolist())
        df[col] = _pandas.Series(df[col])


def _normalize_interval_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return df
    if "chrom" in df.columns:
        _normalize_chrom_column(df, "chrom")
    if "chrom1" in df.columns:
        _normalize_chrom_column(df, "chrom1")
    if "chrom2" in df.columns:
        _normalize_chrom_column(df, "chrom2")
    for col in ("start", "end", "start1", "end1", "start2", "end2"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "strand" in df.columns:
        df["strand"] = df["strand"].astype(int)
    return df


def gintervals_load(
    intervals_set: str | pd.DataFrame,
    chrom: str | list[str] | None = None,
    chrom1: str | list[str] | None = None,
    chrom2: str | list[str] | None = None,
    progress: bool = False,
) -> pd.DataFrame | None:
    """
    Load a named interval set from the database.

    Parameters
    ----------
    intervals_set : str
        Name of the interval set to load (e.g., "annotations", "genes.coding").
    chrom : str, optional
        If specified, only load intervals from this chromosome.
    chrom1 : str, optional
        If specified, load only intervals for this chromosome (2D only).
    chrom2 : str, optional
        If specified, load only intervals for this chromosome (2D only).

    Returns
    -------
    DataFrame or None
        DataFrame with columns 'chrom', 'start', 'end' plus any additional columns
        stored in the interval set. Returns None if no intervals match.

    Raises
    ------
    ValueError
        If the interval set does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervals = pm.gintervals_load("annotations")
    >>> intervals = pm.gintervals_load("annotations", chrom="1")

    See Also
    --------
    gintervals_save : Save intervals as a named set.
    gintervals_update : Update a chromosome in an existing set.
    gintervals_exists : Check if a named interval set exists.
    gintervals_ls : List named interval sets.
    gintervals_rm : Remove a named interval set.
    """
    _checkroot()
    if chrom is not None and (chrom1 is not None or chrom2 is not None):
        raise ValueError("Cannot use chrom with chrom1/chrom2 in the same call")
    if chrom is not None and isinstance(chrom, (list, tuple)):
        if len(chrom) != 1:
            raise ValueError("chrom parameter should mark only one chromosome")
        chrom = chrom[0]
    if chrom1 is not None and isinstance(chrom1, (list, tuple)):
        if len(chrom1) != 1:
            raise ValueError("chrom1 parameter should mark only one chromosome")
        chrom1 = chrom1[0]
    if chrom2 is not None and isinstance(chrom2, (list, tuple)):
        if len(chrom2) != 1:
            raise ValueError("chrom2 parameter should mark only one chromosome")
        chrom2 = chrom2[0]

    if not isinstance(intervals_set, str):
        df = intervals_set
        if df is None:
            return None
        df = df.copy()
        df = _normalize_interval_df(df)
        if df is None:
            return None
        if chrom is not None:
            chrom_norm = _normalize_chroms([chrom])[0]
            if "chrom" not in df.columns:
                raise ValueError("chrom parameter can be applied only to 1D intervals")
            df = df[df["chrom"] == chrom_norm]
        if chrom1 is not None or chrom2 is not None:
            if "chrom1" not in df.columns or "chrom2" not in df.columns:
                raise ValueError("chrom1/chrom2 parameters can be applied only to 2D intervals")
            if chrom1 is not None:
                chrom1_norm = _normalize_chroms([chrom1])[0]
                df = df[df["chrom1"] == chrom1_norm]
            if chrom2 is not None:
                chrom2_norm = _normalize_chroms([chrom2])[0]
                df = df[df["chrom2"] == chrom2_norm]
        if len(df) == 0:
            return None
        return df.reset_index(drop=True)

    interv_path = _intervset_path(intervals_set)

    if interv_path.is_dir():
        stats, zeroline = _decode_intervals_meta(interv_path / ".meta")
        stats = stats.copy()
        if "chrom" in stats.columns:
            stats["chrom"] = _normalize_chroms(stats["chrom"].astype(str).tolist())
        if "chrom1" in stats.columns:
            stats["chrom1"] = _normalize_chroms(stats["chrom1"].astype(str).tolist())
        if "chrom2" in stats.columns:
            stats["chrom2"] = _normalize_chroms(stats["chrom2"].astype(str).tolist())
        max_size = cast("int | None", CONFIG.get("max_data_size"))
        if "chrom" in stats.columns:
            if chrom1 is not None or chrom2 is not None:
                raise ValueError(f"{intervals_set} is a 1D big intervals set. chrom1/chrom2 are for 2D only.")
            if chrom is not None:
                chrom = _normalize_chroms([chrom])[0]
                stats = stats[stats["chrom"].astype(str) == chrom]
            ok, err = _intervset_loadable(stats, max_size, intervals_set, chrom=cast("str | None", chrom))
            if not ok:
                raise ValueError(err)
            if len(stats) == 0:
                return _normalize_interval_df(zeroline)

            paths = _intervset_index_paths(interv_path)
            indexed_fast = chrom is None and _intervset_is_indexed(interv_path)
            if indexed_fast:
                idx_entries = _load_index_entries_1d(paths["idx1d"])
                dfs = []
                with _progress_context(progress, total=len(idx_entries), desc="Loading intervals") as cb:
                    for idx, (_chrom_id, offset, length) in enumerate(idx_entries):
                        if length == 0:
                            continue
                        df = _read_indexed_entry(paths["dat1d"], offset, length)
                        if df is not None:
                            dfs.append(df)
                        if cb:
                            done = idx + 1
                            pct = int(100 * done / len(idx_entries))
                            cb(done, len(idx_entries), pct)
                if not dfs:
                    return _normalize_interval_df(zeroline)
                df = _pandas.concat(dfs, ignore_index=True)
                return _normalize_interval_df(df)

            idx_entries_map = None
            if chrom is not None and paths["idx1d"].exists():
                idx_entries_map = _indexed_entries_by_chrom(_load_index_entries_1d(paths["idx1d"]))
            dfs = []
            with _progress_context(progress, total=len(stats), desc="Loading intervals") as cb:
                for idx, chrom_name in enumerate(stats["chrom"].tolist()):
                    chrom_file = _resolve_chrom_file(interv_path, chrom_name)
                    if chrom_file and chrom_file.exists():
                        dfs.append(_load_serialized_dataframe(chrom_file))
                    elif chrom is not None and idx_entries_map is not None:
                        chrom_map = _chrom_id_map()
                        chrom_id = _chrom_id_lookup(chrom_map, chrom_name)
                        if chrom_id is not None:
                            entry = idx_entries_map.get(chrom_id)
                            if entry:
                                offset, length = entry
                                df = _read_indexed_entry(paths["dat1d"], offset, length)
                                if df is not None:
                                    dfs.append(df)
                    done = idx + 1
                    if cb:
                        pct = int(100 * done / len(stats))
                        cb(done, len(stats), pct)
            if not dfs:
                return _normalize_interval_df(zeroline)
            df = _pandas.concat(dfs, ignore_index=True)
            return _normalize_interval_df(df)

        if chrom is not None:
            raise ValueError(f"{intervals_set} is a 2D big intervals set. chrom is for 1D only.")
        if chrom1 is not None:
            chrom1 = _normalize_chroms([chrom1])[0]
            stats = stats[stats["chrom1"].astype(str) == chrom1]
        if chrom2 is not None:
            chrom2 = _normalize_chroms([chrom2])[0]
            stats = stats[stats["chrom2"].astype(str) == chrom2]
        ok, err = _intervset_loadable(
            stats, max_size, intervals_set,
            chrom1=cast("str | None", chrom1),
            chrom2=cast("str | None", chrom2),
        )
        if not ok:
            raise ValueError(err)
        if len(stats) == 0:
            return _normalize_interval_df(zeroline)

        paths = _intervset_index_paths(interv_path)
        indexed_fast = chrom1 is None and chrom2 is None and _intervset_is_indexed(interv_path)
        if indexed_fast:
            idx_entries_2d = _load_index_entries_2d(paths["idx2d"])
            dfs = []
            with _progress_context(progress, total=len(idx_entries_2d), desc="Loading intervals") as cb:
                for idx, (_chrom1_id, _chrom2_id, offset, length) in enumerate(idx_entries_2d):
                    if length == 0:
                        continue
                    df = _read_indexed_entry(paths["dat2d"], offset, length)
                    if df is not None:
                        dfs.append(df)
                    if cb:
                        done = idx + 1
                        pct = int(100 * done / len(idx_entries_2d))
                        cb(done, len(idx_entries_2d), pct)
            if not dfs:
                return _normalize_interval_df(zeroline)
            df = _pandas.concat(dfs, ignore_index=True)
            return _normalize_interval_df(df)

        idx_entries_map_2d: dict[tuple[int, int], tuple[int, int]] | None = None
        if chrom1 is not None and chrom2 is not None and paths["idx2d"].exists():
            idx_entries_map_2d = _indexed_entries_by_pair(_load_index_entries_2d(paths["idx2d"]))
        dfs = []
        with _progress_context(progress, total=len(stats), desc="Loading intervals") as cb:
            for idx, row in enumerate(stats.itertuples(index=False)):
                chrom1_name = row.chrom1
                chrom2_name = row.chrom2
                pair_file = _resolve_pair_file(interv_path, chrom1_name, chrom2_name)
                if pair_file and pair_file.exists():
                    dfs.append(_load_serialized_dataframe(pair_file))
                elif idx_entries_map_2d is not None:
                    chrom_map = _chrom_id_map()
                    chrom1_id = _chrom_id_lookup(chrom_map, chrom1_name)
                    chrom2_id = _chrom_id_lookup(chrom_map, chrom2_name)
                    if chrom1_id is not None and chrom2_id is not None:
                        entry = idx_entries_map_2d.get((chrom1_id, chrom2_id))
                        if entry:
                            offset, length = entry
                            df = _read_indexed_entry(paths["dat2d"], offset, length)
                            if df is not None:
                                dfs.append(df)
                done = idx + 1
                if cb:
                    pct = int(100 * done / len(stats))
                    cb(done, len(stats), pct)
        if not dfs:
            return _normalize_interval_df(zeroline)
        df = _pandas.concat(dfs, ignore_index=True)
        return _normalize_interval_df(df)

    # Try loading with pyreadr (R's RDS format)
    df = _load_serialized_dataframe(interv_path)

    if df is None or len(df) == 0:
        return None

    # Convert column types
    df = _normalize_interval_df(df)
    if df is None:
        return None

    # Apply chromosome filter if specified
    if chrom is not None:
        if "chrom" not in df.columns:
            raise ValueError("chrom parameter can be applied only to 1D intervals")
        chrom_norm = _normalize_chroms([chrom])[0]
        df = df[df["chrom"] == chrom_norm]
        if len(df) == 0:
            return None
        df = df.reset_index(drop=True)
    if chrom1 is not None or chrom2 is not None:
        if "chrom1" not in df.columns or "chrom2" not in df.columns:
            raise ValueError("chrom1/chrom2 parameters can be applied only to 2D intervals")
        if chrom1 is not None:
            chrom1_norm = _normalize_chroms([chrom1])[0]
            df = df[df["chrom1"] == chrom1_norm]
        if chrom2 is not None:
            chrom2_norm = _normalize_chroms([chrom2])[0]
            df = df[df["chrom2"] == chrom2_norm]
        if len(df) == 0:
            return None
        df = df.reset_index(drop=True)

    return df


def gintervals_save(intervals: pd.DataFrame, intervals_set: str) -> None:
    """
    Save intervals to the database as a named interval set.

    Parameters
    ----------
    intervals : DataFrame
        Intervals to save. Must have either 'chrom', 'start', 'end' columns
        (1D) or 'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2'
        columns (2D).
    intervals_set : str
        Name for the interval set. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.

    Raises
    ------
    ValueError
        If the interval set name is invalid or already exists.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])
    >>> pm.gintervals_save(intervals, "my_intervals")  # doctest: +SKIP

    Returns
    -------
    None

    See Also
    --------
    gintervals_load : Load a named interval set.
    gintervals_update : Update a chromosome in an existing set.
    gintervals_exists : Check if a named interval set exists.
    gintervals_ls : List named interval sets.
    gintervals_rm : Remove a named interval set.
    """
    _checkroot()
    from . import _shared
    assert _shared._GROOT is not None

    # Validate name
    validate_dotted_name(intervals_set, "interval set name")

    # Check if already exists
    if gintervals_exists(intervals_set):
        raise ValueError(f"Intervals set '{intervals_set}' already exists")

    groot = _shared._GROOT
    path_part = intervals_set.replace(".", "/")
    interv_path = Path(groot) / "tracks" / f"{path_part}.interv"

    # Ensure parent directory exists
    interv_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate intervals
    if intervals is None or len(intervals) == 0:
        raise ValueError("Cannot save empty intervals")

    # Detect 1D vs 2D
    is_2d = {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}.issubset(
        intervals.columns
    )
    is_1d = {"chrom", "start", "end"}.issubset(intervals.columns)

    if not is_1d and not is_2d:
        raise ValueError(
            "Intervals must have 'chrom', 'start', 'end' columns (1D) "
            "or 'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2' columns (2D)"
        )

    # Prepare DataFrame for saving
    df = intervals.copy()

    if is_2d:
        # Normalize chromosome names
        df["chrom1"] = _normalize_chroms(df["chrom1"].astype(str).tolist())
        df["chrom2"] = _normalize_chroms(df["chrom2"].astype(str).tolist())
        # Sort by (chrom1, chrom2)
        df = df.sort_values(["chrom1", "chrom2"]).reset_index(drop=True)
        # Convert chrom to categorical (R factor style)
        df["chrom1"] = _pandas.Categorical(df["chrom1"])
        df["chrom2"] = _pandas.Categorical(df["chrom2"])
        # Ensure start/end are float (R numeric)
        for col in ("start1", "end1", "start2", "end2"):
            df[col] = df[col].astype(float)
    else:
        # Normalize chromosome names
        df["chrom"] = _normalize_chroms(df["chrom"].astype(str).tolist())
        # Sort by chrom
        df = df.sort_values(["chrom"]).reset_index(drop=True)
        # Convert chrom to categorical (R factor style)
        df["chrom"] = _pandas.Categorical(df["chrom"])
        # Ensure start/end are float (R numeric)
        df["start"] = df["start"].astype(float)
        df["end"] = df["end"].astype(float)

    # Save using the native R-serialize writer (drops the pyreadr/librdata
    # dependency at runtime and is ~50x faster on million-row data frames
    # because we avoid the Python <-> librdata bridge per row).
    from ._r_serialize import write_dataframe
    write_dataframe(str(interv_path), df)

    # Register the new interval-set name in the C++ cache so it shows up
    # in gintervals_ls() / gintervals_exists() without paying a full
    # pm_dbreload rescan of the tracks/ tree.
    with _contextlib.suppress(Exception):
        _pymisha.pm_interv_register(intervals_set)


def gintervals_update(
    intervals_set: str,
    intervals: pd.DataFrame | None,
    chrom: str | None = None,
) -> None:
    """
    Update intervals for a specific chromosome in an existing intervals set.

    Replaces all intervals for the given chromosome with the new intervals.
    Pass intervals=None to delete all intervals for that chromosome.

    Parameters
    ----------
    intervals_set : str
        Name of the existing intervals set.
    intervals : DataFrame or None
        New intervals for the chromosome, or None to delete.
    chrom : str
        Chromosome to update. Required.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If intervals set does not exist or chrom is not specified.

    See Also
    --------
    gintervals_save : Save a new interval set.
    gintervals_load : Load a named interval set.
    gintervals_exists : Check if a named interval set exists.
    gintervals_ls : List named interval sets.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals(["1", "2"], [0, 0], [10000, 10000])
    >>> pm.gintervals_save(intervs, "testintervs")  # doctest: +SKIP
    >>> pm.gintervals_update("testintervs", pm.gintervals("2", 500, 5000), chrom="2")  # doctest: +SKIP
    >>> pm.gintervals_rm("testintervs", force=True)  # doctest: +SKIP
    """
    _checkroot()

    if chrom is None:
        raise ValueError("Chromosome must be specified in chrom parameter")

    if not gintervals_exists(intervals_set):
        raise ValueError(f"Intervals set '{intervals_set}' does not exist")

    # Normalize chrom
    chrom = _normalize_chroms([str(chrom)])[0]

    # Load existing intervals
    existing = gintervals_load(intervals_set)
    if existing is None:
        existing = _pandas.DataFrame(columns=["chrom", "start", "end"])

    # Remove intervals for the target chrom
    mask = existing["chrom"] != chrom
    kept = existing[mask].copy()

    if intervals is not None:
        # Normalize new intervals
        new_df = intervals.copy()
        if "chrom" in new_df.columns:
            new_df["chrom"] = _normalize_chroms(new_df["chrom"].astype(str).tolist())

        # Combine
        if len(kept) > 0 and len(new_df) > 0:
            kept = _pandas.concat([kept, new_df], ignore_index=True)
        elif len(new_df) > 0:
            kept = new_df

    if len(kept) == 0:
        raise ValueError("Cannot save empty intervals")

    # Remove and re-save
    gintervals_rm(intervals_set, force=True)
    gintervals_save(kept, intervals_set)


def gintervals_mapply(
    func: Callable[..., Any],
    *exprs: str,
    intervals: pd.DataFrame | str | None = None,
    iterator: Any = None,
    intervals_set_out: str | None = None,
    colnames: str = "value",
    enable_gapply_intervals: bool = False,
    band: tuple[int, int] | tuple[float, float] | None = None,
) -> pd.DataFrame | None:
    """
    Apply a function to track expression values for each interval.

    Evaluates track expressions for each interval and passes the resulting
    value arrays to *func*. The return value of *func* becomes a new column
    in the output.

    Parameters
    ----------
    func : callable
        Function to apply. Receives one numpy array per track expression.
        If *enable_gapply_intervals* is True, also receives a keyword
        argument ``gapply_intervals`` containing the current iterator
        interval (as a dict with chrom/start/end keys, plus the 2D
        analogues when applicable).
    *exprs : str
        Track expressions to evaluate.
    intervals : DataFrame
        Intervals to process.
    iterator : optional
        Track expression iterator.
    intervals_set_out : str, optional
        If given, save result as an intervals set and return None.
    colnames : str, default "value"
        Name of the result column.
    enable_gapply_intervals : bool, default False
        R parity. When True, ``func`` is called with an additional
        ``gapply_intervals`` keyword argument carrying the current
        iterator interval. ``func`` must accept ``**kwargs`` or an
        explicit ``gapply_intervals=None`` parameter.
    band : (int, int), optional
        2D-band filter for 2D track expressions. Forwarded to
        :func:`gextract`.

    Returns
    -------
    DataFrame or None
        Intervals with an additional column containing func results,
        or None if intervals_set_out is specified.

    See Also
    --------
    giterator_intervals : Inspect iterator bin boundaries.

    Examples
    --------
    >>> import pymisha as pm
    >>> import numpy as np
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_mapply(
    ...     np.max, "dense_track",
    ...     intervals=pm.gintervals(["1", "2"], 0, 10000),
    ... )  # doctest: +SKIP
    """
    from .extract import _maybe_load_intervals_set, gextract

    _checkroot()

    if intervals is None:
        raise ValueError("intervals parameter is required")

    intervals = _maybe_load_intervals_set(intervals)

    expr_list = list(exprs)
    if not expr_list:
        raise ValueError("At least one track expression is required")

    np = _numpy

    # Determine the intervals to iterate over
    if iterator is not None:
        # With explicit iterator, use giterator_intervals to split
        iter_intervals = giterator_intervals(
            expr=expr_list[0], intervals=intervals, iterator=iterator
        )
        if iter_intervals is None or len(iter_intervals) == 0:
            return None
        work_intervals = iter_intervals
    else:
        # Without iterator, iterate over original intervals directly
        work_intervals = intervals

    # --- Batch extraction: one gextract call per expression for ALL intervals ---
    # This avoids N separate C++ extraction calls (one per interval).
    query_intervals = work_intervals[["chrom", "start", "end"]].copy()
    query_intervals = query_intervals.reset_index(drop=True)

    # Check strand reversal per interval
    has_strand = "strand" in work_intervals.columns
    reverse_flags = work_intervals["strand"].to_numpy() == -1 if has_strand else None

    # Extract each expression once for all intervals
    extracted = []
    for expr in expr_list:
        ext = gextract(expr, intervals=query_intervals, band=band)
        extracted.append(ext)

    n_intervals = len(query_intervals)

    # Pre-group extracted data by intervalID for O(1) lookup per interval.
    # gextract returns 1-based intervalIDs.
    grouped_data: list[tuple[dict[Any, Any] | None, Any]] = []
    for ext in extracted:
        if ext is not None and len(ext) > 0:
            iid = ext["intervalID"].to_numpy()
            val_cols = [c for c in ext.columns if c not in ("chrom", "start", "end", "intervalID")]
            val_col = val_cols[0] if val_cols else None
            vals = ext[val_col].to_numpy(dtype=float) if val_col else None
            # Build dict: intervalID -> (start_idx, end_idx) for contiguous groups
            grp_dict = {}
            if len(iid) > 0:
                breaks = np.flatnonzero(np.diff(iid) != 0)
                starts = np.concatenate([[0], breaks + 1])
                ends = np.concatenate([breaks + 1, [len(iid)]])
                for s, e in zip(starts, ends, strict=False):
                    grp_dict[iid[s]] = (s, e)
            grouped_data.append((grp_dict, vals))
        else:
            grouped_data.append((None, None))

    # Apply func per interval
    results = []
    iv_rows = (
        query_intervals.to_dict("records")
        if enable_gapply_intervals
        else None
    )
    for i in range(n_intervals):
        reverse = reverse_flags[i] if reverse_flags is not None else False
        interval_id = i + 1  # gextract uses 1-based intervalIDs

        arrays = []
        for g_dict, g_vals in grouped_data:
            if g_dict is not None and g_vals is not None and interval_id in g_dict:
                s, e = g_dict[interval_id]
                arr = g_vals[s:e]
            else:
                arr = np.array([])
            if reverse:
                arr = arr[::-1].copy()
            arrays.append(arr)

        if enable_gapply_intervals:
            assert iv_rows is not None
            val = func(*arrays, gapply_intervals=iv_rows[i])
        else:
            val = func(*arrays)
        results.append(val)

    # Build result DataFrame
    result_df = query_intervals.copy()
    result_df[colnames] = results

    if intervals_set_out is not None:
        if gintervals_exists(intervals_set_out):
            gintervals_rm(intervals_set_out, force=True)
        gintervals_save(result_df, intervals_set_out)
        return None

    return result_df


def _copy_file_contents(src_path: Path, dest_fh: IO[bytes], buffer_size: int = 1024 * 1024) -> int:
    total = 0
    with open(src_path, "rb") as src:
        while True:
            chunk = src.read(buffer_size)
            if not chunk:
                break
            dest_fh.write(chunk)
            total += len(chunk)
    return total


def _write_index_header_1d(fp: IO[bytes], num_entries: int, checksum: int) -> None:
    magic = b"MISHAI1D"
    version = 1
    flags = 0x01
    reserved = 0
    fp.write(struct.pack("<8sIIQQI", magic, version, num_entries, flags, checksum, reserved))


def _write_index_header_2d(fp: IO[bytes], num_entries: int, checksum: int) -> None:
    magic = b"MISHAI2D"
    version = 1
    flags = 0x01
    reserved = 0
    fp.write(struct.pack("<8sIIQQQ", magic, version, num_entries, flags, checksum, reserved))


def gintervals_convert_to_indexed(
    set_name: str,
    remove_old: bool = False,
    force: bool = False,
) -> None:
    """
    Convert a 1D big interval set to indexed format.

    Converts per-chromosome interval files into a single
    ``intervals.dat`` + ``intervals.idx`` pair, reducing file-descriptor
    usage from N files to 2. The indexed format is backward-compatible
    with all misha interval functions.

    Parameters
    ----------
    set_name : str
        Name of the 1D interval set to convert.
    remove_old : bool, default False
        If True, remove the old per-chromosome files after conversion.
    force : bool, default False
        If True, re-convert even if the set is already indexed.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If *set_name* is empty or the interval set does not exist.

    See Also
    --------
    gintervals_2d_convert_to_indexed : Convert a 2D interval set to indexed format.
    gintervals_is_indexed : Check if a set is already indexed.
    gintervals_save : Save intervals as a named set.
    gintervals_load : Load a named interval set.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_convert_to_indexed("my_intervals")  # doctest: +SKIP
    >>> pm.gintervals_convert_to_indexed("my_intervals", remove_old=True)  # doctest: +SKIP
    """
    if not isinstance(set_name, str) or not set_name:
        raise ValueError("set_name must be a non-empty string")
    _checkroot()

    interv_path = _intervset_path(set_name)
    if not interv_path.exists():
        raise ValueError(f"Intervals set '{set_name}' does not exist")
    if not interv_path.is_dir():
        return

    idx_path = interv_path / "intervals.idx"
    dat_path = interv_path / "intervals.dat"
    if idx_path.exists() and not force:
        return

    dat_tmp = interv_path / "intervals.dat.tmp"
    idx_tmp = interv_path / "intervals.idx.tmp"

    chroms = gintervals_all()["chrom"].tolist()
    chrom_map = _chrom_id_map()

    entries = []
    crc = _crc64_init()
    current_offset = 0
    files_to_remove = []

    with open(dat_tmp, "wb") as dat_fh, open(idx_tmp, "wb") as idx_fh:
        _write_index_header_1d(idx_fh, len(chroms), 0)
        for chrom in chroms:
            chrom_file = _resolve_chrom_file(interv_path, chrom)
            length = 0
            if chrom_file and chrom_file.exists():
                length = _copy_file_contents(chrom_file, dat_fh)
                if length > 0:
                    files_to_remove.append(chrom_file)
            chrom_id = chrom_map[chrom]
            entry = (chrom_id, current_offset, length)
            entries.append(entry)
            idx_fh.write(struct.pack("<IQQI", chrom_id, current_offset, length, 0))
            crc = _crc64_incremental(crc, struct.pack("<I", chrom_id))
            crc = _crc64_incremental(crc, struct.pack("<Q", current_offset))
            crc = _crc64_incremental(crc, struct.pack("<Q", length))
            current_offset += length

        checksum = _crc64_finalize(crc)
        idx_fh.flush()
        idx_fh.seek(8 + 4 + 4 + 8)
        idx_fh.write(struct.pack("<Q", checksum))
        idx_fh.flush()
        os.fsync(idx_fh.fileno())
        dat_fh.flush()
        os.fsync(dat_fh.fileno())

    os.replace(dat_tmp, dat_path)
    os.replace(idx_tmp, idx_path)

    if remove_old:
        for chrom_file in files_to_remove:
            with _contextlib.suppress(FileNotFoundError):
                chrom_file.unlink()
    return


def gintervals_2d_convert_to_indexed(
    set_name: str,
    remove_old: bool = False,
    force: bool = False,
) -> None:
    """
    Convert a 2D big interval set to indexed format.

    Converts per-chromosome-pair interval files into a single
    ``intervals2d.dat`` + ``intervals2d.idx`` pair.  This dramatically
    reduces file-descriptor usage, especially for genomes with many
    chromosomes (from N*(N-1)/2 files to 2).

    Parameters
    ----------
    set_name : str
        Name of the 2D interval set to convert.
    remove_old : bool, default False
        If True, remove the old per-pair files after conversion.
    force : bool, default False
        If True, re-convert even if the set is already indexed.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If *set_name* is empty or the interval set does not exist.

    See Also
    --------
    gintervals_convert_to_indexed : Convert a 1D interval set to indexed format.
    gintervals_is_indexed : Check if a set is already indexed.
    gintervals_save : Save intervals as a named set.
    gintervals_load : Load a named interval set.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_2d_convert_to_indexed("my_2d_intervals")  # doctest: +SKIP
    >>> pm.gintervals_2d_convert_to_indexed("my_2d_intervals", remove_old=True)  # doctest: +SKIP
    """
    if not isinstance(set_name, str) or not set_name:
        raise ValueError("set_name must be a non-empty string")
    _checkroot()

    interv_path = _intervset_path(set_name)
    if not interv_path.exists():
        raise ValueError(f"Intervals set '{set_name}' does not exist")
    if not interv_path.is_dir():
        return

    idx_path = interv_path / "intervals2d.idx"
    dat_path = interv_path / "intervals2d.dat"
    if idx_path.exists() and not force:
        return

    dat_tmp = interv_path / "intervals2d.dat.tmp"
    idx_tmp = interv_path / "intervals2d.idx.tmp"

    chrom_map = _chrom_id_map()

    pair_files = []
    for entry in interv_path.iterdir():
        if entry.name in {"intervals.idx", "intervals.dat", "intervals2d.idx", "intervals2d.dat", ".meta"}:
            continue
        if entry.is_dir():
            continue
        if "-" not in entry.name:
            continue
        chrom1_name, chrom2_name = entry.name.split("-", 1)
        chrom1_id = _chrom_id_lookup(chrom_map, chrom1_name)
        chrom2_id = _chrom_id_lookup(chrom_map, chrom2_name)
        if chrom1_id is None or chrom2_id is None:
            continue
        pair_files.append((chrom1_id, chrom2_id, entry))

    pair_files.sort(key=lambda x: (x[0], x[1]))

    crc = _crc64_init()
    current_offset = 0
    files_to_remove = []

    with open(dat_tmp, "wb") as dat_fh, open(idx_tmp, "wb") as idx_fh:
        _write_index_header_2d(idx_fh, len(pair_files), 0)
        for chrom1_id, chrom2_id, path in pair_files:
            length = _copy_file_contents(path, dat_fh)
            if length > 0:
                files_to_remove.append(path)
            idx_fh.write(struct.pack("<IIQQI", chrom1_id, chrom2_id, current_offset, length, 0))
            crc = _crc64_incremental(crc, struct.pack("<I", chrom1_id))
            crc = _crc64_incremental(crc, struct.pack("<I", chrom2_id))
            crc = _crc64_incremental(crc, struct.pack("<Q", current_offset))
            crc = _crc64_incremental(crc, struct.pack("<Q", length))
            current_offset += length

        checksum = _crc64_finalize(crc)
        idx_fh.flush()
        idx_fh.seek(8 + 4 + 4 + 8)
        idx_fh.write(struct.pack("<Q", checksum))
        idx_fh.flush()
        os.fsync(idx_fh.fileno())
        dat_fh.flush()
        os.fsync(dat_fh.fileno())

    os.replace(dat_tmp, dat_path)
    os.replace(idx_tmp, idx_path)

    if remove_old:
        for path in files_to_remove:
            with _contextlib.suppress(FileNotFoundError):
                path.unlink()
    return


def gintervals_is_indexed(intervals_set: str) -> bool:
    """
    Check whether a big interval set is stored in indexed format.

    Indexed format means the set uses ``intervals.idx``/``intervals.dat``
    (1D) or ``intervals2d.idx``/``intervals2d.dat`` (2D) files instead
    of per-chromosome files.

    Parameters
    ----------
    intervals_set : str
        Name of the interval set to check.

    Returns
    -------
    bool
        ``True`` if the set is a big (directory-based) interval set
        stored in indexed format, ``False`` otherwise (including
        non-directory sets).

    See Also
    --------
    gintervals_convert_to_indexed : Convert a 1D set to indexed format.
    gintervals_2d_convert_to_indexed : Convert a 2D set to indexed format.
    gintervals_exists : Check if a named interval set exists.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_is_indexed("annotations")
    False
    """
    if not isinstance(intervals_set, str):
        raise ValueError("intervals_set must be a string")
    path = _intervset_path(intervals_set)
    if not path.exists() or not path.is_dir():
        return False
    return _intervset_is_indexed(path, allow_updates=False)


def giterator_cartesian_grid(
    intervals1: pd.DataFrame,
    expansion1: Any,
    intervals2: pd.DataFrame | None = None,
    expansion2: Any | None = None,
    min_band_idx: int | None = None,
    max_band_idx: int | None = None,
    *,
    stream: bool = False,
) -> pd.DataFrame | Any:
    """
    Create a 2D cartesian-grid iterator as 2D intervals.

    When ``stream=True``, returns a :class:`~pymisha._iterator_policy.CartesianGridSpec`
    that can be passed to :func:`gextract` as the ``iterator=`` argument.
    The C++ scanner will generate the cartesian-product cells on the fly
    during extraction, without materializing the full grid first.

    When ``stream=False`` (default), materializes all cells as a DataFrame.

    The grid is built from 1D interval centers and expansion breakpoints.
    For each center ``C`` and consecutive expansion pair ``(E[i], E[i+1])``,
    one 1D window ``[C + E[i], C + E[i+1])`` is created (clipped to chromosome
    bounds). The final result is the cartesian product of windows from
    ``intervals1`` and ``intervals2``.

    Parameters
    ----------
    intervals1 : DataFrame
        1D intervals with columns ``chrom``, ``start``, ``end``.
    expansion1 : sequence of int
        Expansion breakpoints around centers of ``intervals1``.
        Must contain at least two unique values.
    intervals2 : DataFrame, optional
        Second 1D interval source. If ``None``, ``intervals1`` is reused.
    expansion2 : sequence of int, optional
        Expansion breakpoints for ``intervals2``. If ``None``, ``expansion1``
        is reused.
    min_band_idx : int, optional
        Lower bound for center-index delta filtering (``idx1 - idx2``).
        Can be used only when ``intervals2`` is ``None``.
    max_band_idx : int, optional
        Upper bound for center-index delta filtering. Can be used only when
        ``intervals2`` is ``None``.

    Returns
    -------
    DataFrame
        2D intervals with columns:
        ``chrom1``, ``start1``, ``end1``, ``chrom2``, ``start2``, ``end2``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if stream:
        from ._iterator_policy import CartesianGridSpec

        return CartesianGridSpec(
            intervals1=intervals1,
            expansion1=tuple(int(x) for x in expansion1),
            intervals2=intervals2,
            expansion2=tuple(int(x) for x in expansion2) if expansion2 is not None else None,
            min_band_idx=min_band_idx,
            max_band_idx=max_band_idx,
        )

    _checkroot()

    if intervals1 is None or expansion1 is None:
        raise ValueError(
            "Usage: giterator_cartesian_grid(intervals1, expansion1, "
            "intervals2=None, expansion2=None, min_band_idx=None, max_band_idx=None)"
        )

    use_band_idx = (min_band_idx is not None) or (max_band_idx is not None)
    if use_band_idx:
        if min_band_idx is None or max_band_idx is None:
            raise ValueError("Both min_band_idx and max_band_idx must be provided")
        if intervals2 is not None:
            raise ValueError("band.idx limit can only be used when intervals2 is None")
        min_band_idx = int(min_band_idx)
        max_band_idx = int(max_band_idx)
        if min_band_idx > max_band_idx:
            raise ValueError("min_band_idx exceeds max_band_idx")
    else:
        min_band_idx = 0
        max_band_idx = 0

    def _normalize_input_intervals(df, name):
        if not isinstance(df, _pandas.DataFrame):
            raise ValueError(f"{name} must be a DataFrame")
        if not {"chrom", "start", "end"}.issubset(df.columns):
            raise ValueError(f"{name} must contain chrom, start, end columns")
        out = df[["chrom", "start", "end"]].copy()
        out["chrom"] = _normalize_chroms(out["chrom"].astype(str).tolist())
        out["start"] = _pandas.to_numeric(out["start"], errors="coerce").astype("Int64")
        out["end"] = _pandas.to_numeric(out["end"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["start", "end"]).copy()
        if len(out) == 0:
            return _pandas.DataFrame(columns=["chrom", "start", "end", "center_idx"])
        out["start"] = out["start"].astype(_numpy.int64)
        out["end"] = out["end"].astype(_numpy.int64)
        out = out[out["end"] > out["start"]]
        out = out.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        out["center_idx"] = _numpy.arange(len(out), dtype=_numpy.int64)
        return out

    def _normalize_expansion(expansion, name):
        arr = _numpy.asarray(list(expansion), dtype=_numpy.int64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D sequence of integers")
        if arr.size < 2:
            raise ValueError(f"{name} must contain at least 2 values")
        unique_vals = _numpy.unique(arr)
        if unique_vals.size != arr.size:
            raise ValueError(f"{name} values must be unique")
        unique_vals.sort()
        return unique_vals

    i1 = _normalize_input_intervals(intervals1, "intervals1")
    i2 = i1 if intervals2 is None else _normalize_input_intervals(intervals2, "intervals2")

    e1 = _normalize_expansion(expansion1, "expansion1")
    e2 = e1 if expansion2 is None else _normalize_expansion(expansion2, "expansion2")

    chrom_sizes_df = gintervals_all()
    chrom_sizes = {
        str(chrom): int(end)
        for chrom, end in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)
    }

    def _build_windows(intervals_df, expansion):
        rows = []
        for row in intervals_df.itertuples(index=False):
            chrom = str(row.chrom)
            chrom_size = chrom_sizes.get(chrom)
            if chrom_size is None:
                continue
            center = (int(row.start) + int(row.end)) // 2
            for left, right in zip(expansion[:-1], expansion[1:], strict=False):
                start = center + int(left)
                end = center + int(right)
                if start < 0:
                    start = 0
                if end > chrom_size:
                    end = chrom_size
                if end <= start:
                    continue
                rows.append((int(row.center_idx), chrom, int(start), int(end)))
        return _pandas.DataFrame(rows, columns=["center_idx", "chrom", "start", "end"])

    w1 = _build_windows(i1, e1)
    w2 = _build_windows(i2, e2)
    if len(w1) == 0 or len(w2) == 0:
        return _pandas.DataFrame(
            columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        )

    out_rows = []
    for r1 in w1.itertuples(index=False):
        for r2 in w2.itertuples(index=False):
            if use_band_idx:
                if r1.chrom != r2.chrom:
                    continue
                delta = int(r1.center_idx) - int(r2.center_idx)
                if delta < min_band_idx or delta > max_band_idx:
                    continue
            out_rows.append(
                (
                    r1.chrom,
                    int(r1.start),
                    int(r1.end),
                    r2.chrom,
                    int(r2.start),
                    int(r2.end),
                )
            )

    if not out_rows:
        return _pandas.DataFrame(
            columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        )

    result = _pandas.DataFrame(
        out_rows,
        columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2"],
    )
    result = result.drop_duplicates()
    return result.sort_values(
        ["chrom1", "start1", "chrom2", "start2", "end1", "end2"]
    ).reset_index(drop=True)


def _resolve_2d_scope(intervals: pd.DataFrame | str | None) -> pd.DataFrame:
    """Resolve a scope argument to a 2D-intervals DataFrame for a 2D iterator.

    - ``None`` -> the whole 2D genome (all chromosome pairs, full rectangles),
      matching R's use of ``.misha$ALLGENOME`` as a 2D-iterator scope.
    - a 2D-track name -> the track's rectangles.
    - a named 2D interval set -> its rectangles.
    - a 2D DataFrame -> returned as-is.
    """
    from .extract import gextract
    from .tracks import gtrack_exists, gtrack_info

    if intervals is None:
        return gintervals_2d_all(mode="full")

    if isinstance(intervals, str):
        if gtrack_exists(intervals):
            if int(gtrack_info(intervals).get("dimensions", 1) or 1) != 2:
                raise ValueError(
                    f"Track '{intervals}' is not 2D; a cartesian-grid iterator "
                    "requires a 2D scope"
                )
            res = gextract(intervals, gintervals_2d_all(mode="full"))
            if res is None or len(res) == 0:
                return _pandas.DataFrame(
                    columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
                )
            return res[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
        loaded = gintervals_load(intervals)
        if loaded is None or "chrom1" not in getattr(loaded, "columns", []):
            raise ValueError(
                f"Scope '{intervals}' is not a 2D interval set / 2D track"
            )
        return loaded

    if isinstance(intervals, _pandas.DataFrame) and "chrom1" in intervals.columns:
        return intervals

    raise ValueError(
        "A cartesian-grid iterator requires a 2D scope (2D DataFrame, 2D track "
        "name, or None for the whole 2D genome)"
    )


def _enumerate_cartesian_grid_cells(
    spec: Any,
    intervals: pd.DataFrame | str | None,
    band: tuple[int, int] | tuple[float, float] | None,
    intervals_set_out: str | None,
) -> pd.DataFrame | None:
    """Enumerate the 2D cells of a CartesianGrid iterator over a 2D scope.

    Mirrors R's ``giterator.intervals(expr, scope, iterator = cartesian_grid)``:
    builds the grid (centers de-duplicated, adjacent-center expansions clipped
    at midpoints), intersects each cell with the scope rectangles, and applies
    the optional diagonal ``band``. Delegates to the C++ iterator port.
    """
    from .extract import _validate_band

    scope_df = _resolve_2d_scope(intervals)

    # chrom name <-> chromid (gintervals_all order == the C++ chromkey order).
    allg = gintervals_all()
    chrom2id = {str(c): i for i, c in enumerate(allg["chrom"].tolist())}
    id2chrom = {i: c for c, i in chrom2id.items()}

    def _chromids(names: Any) -> _numpy.ndarray:
        return _numpy.array(
            [chrom2id[str(c)] for c in names], dtype=_numpy.int32
        )

    def _to_1d_dict(df: pd.DataFrame) -> dict[str, Any]:
        d = df.copy()
        d["chrom"] = _normalize_chroms(d["chrom"].astype(str).tolist())
        return {
            "chrom": _chromids(d["chrom"]),
            "start": d["start"].to_numpy(_numpy.int64),
            "end": d["end"].to_numpy(_numpy.int64),
        }

    empty_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    if len(scope_df) == 0 or len(spec.intervals1) == 0:
        result = _pandas.DataFrame(columns=empty_cols)
        if intervals_set_out is not None:
            return None
        return result

    s = scope_df.copy()
    s["chrom1"] = _normalize_chroms(s["chrom1"].astype(str).tolist())
    s["chrom2"] = _normalize_chroms(s["chrom2"].astype(str).tolist())
    scope_dict = {
        "chrom1": _chromids(s["chrom1"]),
        "start1": s["start1"].to_numpy(_numpy.int64),
        "end1": s["end1"].to_numpy(_numpy.int64),
        "chrom2": _chromids(s["chrom2"]),
        "start2": s["start2"].to_numpy(_numpy.int64),
        "end2": s["end2"].to_numpy(_numpy.int64),
    }

    i1 = _to_1d_dict(spec.intervals1)
    i2 = None if spec.intervals2 is None else _to_1d_dict(spec.intervals2)
    e1 = _numpy.asarray(spec.expansion1, dtype=_numpy.int64)
    e2 = (
        None if spec.expansion2 is None
        else _numpy.asarray(spec.expansion2, dtype=_numpy.int64)
    )
    band_idx = (
        None if spec.min_band_idx is None
        else (int(spec.min_band_idx), int(spec.max_band_idx))
    )
    band_t = _validate_band(band)
    band_arg = None if band_t is None else (int(band_t[0]), int(band_t[1]))

    out = _pymisha.pm_cartesian_grid_intervals(
        i1, e1, i2, e2, band_idx, scope_dict, band_arg
    )

    result = _pandas.DataFrame(
        {
            "chrom1": [id2chrom[i] for i in out["chrom1"]],
            "start1": out["start1"],
            "end1": out["end1"],
            "chrom2": [id2chrom[i] for i in out["chrom2"]],
            "start2": out["start2"],
            "end2": out["end2"],
        }
    )

    if intervals_set_out is not None:
        if len(result) == 0:
            raise ValueError("Cannot save empty intervals")
        gintervals_save(result, intervals_set_out)
        return None
    return result


def _run_2d_iterator_coords(
    policy_dict: dict[str, Any],
    scope_df: pd.DataFrame,
    band: tuple[int, int] | tuple[float, float] | None,
    intervals_set_out: str | None,
) -> pd.DataFrame | None:
    """Enumerate a 2D iterator policy's cells over a 2D scope (coords only).

    Runs the C++ 2D scanner (``pm_extract_2d_scanner``) with an empty var list
    so it emits only the iterator cell coordinates, then maps chromids back to
    names. ``policy_dict`` is any policy the scanner accepts (``fixed_rect``,
    ``track_rects``, ``cartesian_grid``, ``intervals``).
    """
    from .extract import _validate_band

    # chrom name <-> chromid (gintervals_all order == the C++ chromkey order).
    allg = gintervals_all()
    chrom2id = {str(c): i for i, c in enumerate(allg["chrom"].tolist())}
    id2chrom = {i: c for c, i in chrom2id.items()}

    def _chromids(names: Any) -> _numpy.ndarray:
        return _numpy.array([chrom2id[str(c)] for c in names], dtype=_numpy.int32)

    empty_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    if len(scope_df) == 0:
        if intervals_set_out is not None:
            return None
        return _pandas.DataFrame(columns=empty_cols)

    s = scope_df.copy()
    s["chrom1"] = _normalize_chroms(s["chrom1"].astype(str).tolist())
    s["chrom2"] = _normalize_chroms(s["chrom2"].astype(str).tolist())
    scope_dict = {
        "chrom1": _chromids(s["chrom1"]),
        "start1": s["start1"].to_numpy(_numpy.int64),
        "end1": s["end1"].to_numpy(_numpy.int64),
        "chrom2": _chromids(s["chrom2"]),
        "start2": s["start2"].to_numpy(_numpy.int64),
        "end2": s["end2"].to_numpy(_numpy.int64),
    }

    band_t = _validate_band(band)
    band_arg = None if band_t is None else (int(band_t[0]), int(band_t[1]))

    out = _pymisha.pm_extract_2d_scanner(policy_dict, scope_dict, [], [], band_arg)

    result = _pandas.DataFrame(
        {
            "chrom1": [id2chrom[i] for i in out["_chrom1"]],
            "start1": out["_start1"],
            "end1": out["_end1"],
            "chrom2": [id2chrom[i] for i in out["_chrom2"]],
            "start2": out["_start2"],
            "end2": out["_end2"],
        }
    )

    if intervals_set_out is not None:
        if len(result) == 0:
            raise ValueError("Cannot save empty intervals")
        gintervals_save(result, intervals_set_out)
        return None
    return result


def _enumerate_2d_fixedrect_cells(
    iterator: tuple[Any, Any] | list,
    intervals: pd.DataFrame | str | None,
    band: tuple[int, int] | tuple[float, float] | None,
    intervals_set_out: str | None,
) -> pd.DataFrame | None:
    """Enumerate the cells of a fixed-size 2D iterator over a 2D scope.

    Mirrors R's ``giterator.intervals(expr, scope, iterator = c(width, height))``:
    subdivides each scope rectangle into a fixed ``width x height`` grid, clips
    cells at the scope boundaries, applies the optional diagonal ``band``, and
    returns the cell coordinates (no track values). Delegates to the same C++
    FixedRect iterator the 2D scanner uses, run with an empty var list so it
    emits coordinates only.
    """
    width = int(float(iterator[0]))
    height = int(float(iterator[1]))
    if width <= 0 or height <= 0:
        raise ValueError(
            "A 2D fixed-bin iterator requires two positive bin sizes"
        )

    scope_df = _resolve_2d_scope(intervals)
    return _run_2d_iterator_coords(
        {"kind": "fixed_rect", "width": width, "height": height},
        scope_df, band, intervals_set_out,
    )


def _enumerate_2d_iterator_intervals(
    iterator: Any,
    intervals: pd.DataFrame | str | None,
    band: tuple[int, int] | tuple[float, float] | None,
) -> pd.DataFrame | None:
    """Enumerate the iteration cells of a 2D iterator over a 2D scope (coords only).

    Returns the 2D-interval coordinates an explicit 2D iterator would visit:
      * a numeric ``(width, height)`` tuple -> a fixed-rect grid,
      * a 2D rectangles/points track name -> the track's rects within the scope,
      * a :class:`CartesianGridSpec` -> the cartesian grid cells.
    Returns ``None`` for any iterator that is not a recognised 2D iterator
    (the caller then keeps its existing behaviour).
    """
    from ._iterator_policy import CartesianGridSpec
    from .tracks import gtrack_exists, gtrack_info

    if isinstance(iterator, CartesianGridSpec):
        return _enumerate_cartesian_grid_cells(iterator, intervals, band, None)

    if (
        isinstance(iterator, (tuple, list))
        and len(iterator) == 2
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in iterator
        )
    ):
        return _enumerate_2d_fixedrect_cells(iterator, intervals, band, None)

    if (
        isinstance(iterator, str)
        and gtrack_exists(iterator)
        and int(gtrack_info(iterator).get("dimensions", 1) or 1) == 2
    ):
        scope_df = _resolve_2d_scope(intervals)
        return _run_2d_iterator_coords(
            {"kind": "track_rects", "track_name": iterator},
            scope_df, band, None,
        )

    return None


def giterator_intervals(
    expr: str | None = None,
    intervals: pd.DataFrame | str | None = None,
    iterator: Any = None,
    band: tuple[int, int] | tuple[float, float] | None = None,
    intervals_set_out: str | None = None,
    interval_relative: bool = False,
    partial_bins: str = "clip",
) -> pd.DataFrame | None:
    """
    Return the iterator intervals grid without evaluating track expressions.

    This is useful for inspecting the bin boundaries that would be produced
    by a given iterator/interval combination before running a full extraction.

    Parameters
    ----------
    expr : str, optional
        Track expression (used to determine the implicit iterator when
        *iterator* is ``None``).  Pass ``None`` when an explicit numeric
        *iterator* is supplied.
    intervals : DataFrame, optional
        Genomic scope.  Defaults to :func:`gintervals_all` (whole genome).
    iterator : int or str, optional
        Numeric bin size or track name that defines the iterator.
    band : tuple of (int, int), optional
        Diagonal band ``(d1, d2)`` restricting a 2D iterator to rectangles
        whose offset from the diagonal lies within the band (as in
        :func:`gextract`). Ignored for 1D iterators.
    intervals_set_out : str, optional
        If given, save the resulting iterator intervals as a named interval
        set under this name and return ``None`` instead of a DataFrame
        (mirrors R's ``intervals.set.out``).
    interval_relative : bool, default False
        When ``True``, bins are aligned to each input interval's start
        rather than to chromosome position 0.  Requires a numeric
        *iterator*.
    partial_bins : str, default ``"clip"``
        How to handle bins that do not fit entirely within an interval.

        * ``"clip"`` — truncate the last bin at the interval boundary
          (default, current behavior).
        * ``"drop"`` — discard bins whose size is smaller than the full
          bin size.
        * ``"exact"`` — same as ``"drop"``.

    Returns
    -------
    DataFrame
        DataFrame with columns ``chrom``, ``start``, ``end``, ``intervalID``.

    Raises
    ------
    ValueError
        If neither *expr* nor *iterator* is provided.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.giterator_intervals(intervals=pm.gintervals("1", 0, 200), iterator=50)  # doctest: +SKIP
    >>> pm.giterator_intervals("dense_track", pm.gintervals("1", 0, 1000))  # doctest: +SKIP

    See Also
    --------
    gintervals_mapply : Apply a function to track values per interval.
    """
    _valid_partial_bins = ("clip", "drop", "exact")
    if partial_bins not in _valid_partial_bins:
        raise ValueError(
            f"partial_bins must be one of {_valid_partial_bins}, "
            f"got {partial_bins!r}"
        )

    if expr is None and iterator is None:
        raise ValueError(
            "At least one of 'expr' or 'iterator' must be provided."
        )
    _checkroot()

    # CartesianGrid 2D iterator: enumerate the grid cells over the 2D scope
    # (R's giterator.intervals(expr, scope, iterator = cartesian_grid[, band])).
    from ._iterator_policy import CartesianGridSpec
    if isinstance(iterator, CartesianGridSpec):
        return _enumerate_cartesian_grid_cells(
            iterator, intervals, band=band, intervals_set_out=intervals_set_out
        )

    # Numeric 2D iterator c(width, height): a fixed-rect grid over a 2D scope
    # (R's giterator.intervals(expr, scope, iterator = c(width, height))). The
    # expr only matters when the iterator is implicit, so it is ignored here.
    if (
        isinstance(iterator, (tuple, list))
        and len(iterator) == 2
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in iterator
        )
    ):
        return _enumerate_2d_fixedrect_cells(
            iterator, intervals, band=band, intervals_set_out=intervals_set_out
        )

    # Determine iterator policy
    itr = iterator
    if itr is None and expr is not None:
        # Try to resolve track bin size from expression (track name)
        from .tracks import gtrack_exists, gtrack_info
        if isinstance(expr, str) and gtrack_exists(expr):
            info = gtrack_info(expr)
            bin_size = info.get("bin_size") or info.get("bin.size")
            # Dense track -> its fixed bins. Sparse / array / 2D track -> iterate
            # over the track's own intervals: pass the track name through, to be
            # resolved by the 2D branch below (dims==2) or by
            # _preprocess_intervals_iterator (1D sparse/array -> its intervals).
            itr = int(float(bin_size)) if bin_size is not None else expr
        if itr is None:
            raise ValueError(
                "Could not determine iterator from expression. "
                "Pass an explicit numeric iterator."
            )

    if interval_relative and isinstance(itr, str):
        raise ValueError("interval_relative=True requires a numeric iterator.")

    # Support string iterators for tracks.
    if isinstance(itr, str):
        from .tracks import gtrack_exists, gtrack_info

        if gtrack_exists(itr):
            info = gtrack_info(itr)
            dims = int(info.get("dimensions", 1) or 1)
            if dims == 2:
                from .extract import gextract

                if intervals is None:
                    # R's ALLGENOME used as the scope for a 2D iterator covers
                    # all chrom pairs (full mode), so a bare 2D-track iterator
                    # visits every rectangle of the track - not just the
                    # intra-chromosomal (diagonal) ones.
                    intervals = gintervals_2d_all(mode="full")
                elif isinstance(intervals, str):
                    # The scope may be an interval-set name *or* a 2D track name
                    # (its rectangles); _maybe_load_intervals_set handles both,
                    # whereas gintervals_load only knows interval sets.
                    from .extract import _maybe_load_intervals_set

                    intervals = _maybe_load_intervals_set(intervals)

                if intervals is None or len(intervals) == 0:
                    return None
                if not isinstance(intervals, _pandas.DataFrame) or "chrom1" not in intervals.columns:
                    raise ValueError("2D track iterator requires 2D intervals")

                res = gextract(itr, intervals=intervals, iterator=itr, band=band)
                if res is None or len(res) == 0:
                    return None
                cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"]
                df2d = res[cols].drop_duplicates().reset_index(drop=True)
                if intervals_set_out is not None:
                    gintervals_save(df2d, intervals_set_out)
                    return None
                return df2d

            bin_size = info.get("bin_size") or info.get("bin.size")
            if bin_size is not None:
                itr = int(float(bin_size))

    if intervals is None:
        intervals = gintervals_all()
    elif isinstance(intervals, str):
        # Resolve a named intervals set or a track name used as the scope
        # (1D track -> its intervals, 2D track -> its rectangles) - R parity.
        from .extract import _maybe_load_intervals_set

        intervals = _maybe_load_intervals_set(intervals)
        if isinstance(intervals, str):
            raise ValueError(f"Intervals set '{intervals}' does not exist")
        if intervals is None:
            return None

    if len(intervals) == 0:
        return None

    # A 2D interval-set name used as the iterator (e.g. "test.bigintervs_2d_5")
    # is loaded to its rectangles so the DataFrame branch below routes it through
    # the scalable intersect, matching R's intervals 2D iterator.  (2D *track*
    # iterators are handled by the TrackRects path above.)
    if isinstance(itr, str) and gintervals_exists(itr):
        from .extract import _maybe_load_intervals_set

        _loaded_itr = _maybe_load_intervals_set(itr)
        if isinstance(_loaded_itr, _pandas.DataFrame) and _is_2d_intervals_df(_loaded_itr):
            itr = _loaded_itr

    # 2D intervals DataFrame as iterator: the iteration cells are the clipped
    # intersections of the iterator rects with the 2D scope (R's
    # TrackExpressionIntervals2DIterator builds a quadtree over the scope and
    # walks the iterator rects).  Coordinates only - no track values.
    if (
        isinstance(itr, _pandas.DataFrame)
        and _is_2d_intervals_df(itr)
        and isinstance(intervals, _pandas.DataFrame)
        and _is_2d_intervals_df(intervals)
    ):
        units = _intersect_2d_rects(itr, intervals)
        if len(units) == 0:
            return None
        units = _sort_2d_intervals(units).reset_index(drop=True)
        units["intervalID"] = _numpy.arange(len(units), dtype=int)
        if intervals_set_out is not None:
            gintervals_save(units, intervals_set_out)
            return None
        return units

    # Handle DataFrame-as-iterator
    intervals, itr, _itr_id_map = _preprocess_intervals_iterator(intervals, itr)
    if isinstance(intervals, _pandas.DataFrame) and len(intervals) == 0:
        return None

    with _config_no_mt(_itr_id_map) as _cfg:
        if interval_relative:
            if isinstance(itr, bool) or not isinstance(itr, (int, float)):
                raise ValueError(
                    "interval_relative=True requires a numeric iterator."
                )
            cfg = dict(_cfg)
            cfg["interval_relative"] = True
        else:
            cfg = _cfg

        result = _pymisha.pm_iterate(_df2pymisha(intervals), itr, cfg)
    df = _pymisha2df(result)
    df = _remap_interval_ids(df, _itr_id_map)

    if (
        partial_bins in ("drop", "exact")
        and df is not None
        and len(df) > 0
        and isinstance(itr, (int, float))
    ):
        bin_size = int(itr)
        sizes = df["end"] - df["start"]
        df = df[sizes >= bin_size].reset_index(drop=True)

    if intervals_set_out is not None:
        if df is None or len(df) == 0:
            raise ValueError("Cannot save empty intervals")
        gintervals_save(df, intervals_set_out)
        return None

    return df


def gintervals_rbind(*intervals: pd.DataFrame | str, intervals_set_out: str | None = None) -> pd.DataFrame | None:
    """
    Concatenate interval sets (DataFrames and/or named interval-set strings).

    Parameters
    ----------
    *intervals : DataFrame or str
        One or more interval sets. Each argument can be a DataFrame or a
        named interval set (loaded via :func:`gintervals_load`).
    intervals_set_out : str, optional
        If provided, save the concatenated intervals via
        :func:`gintervals_save` and return ``None``.

    Returns
    -------
    DataFrame or None
        Concatenated intervals when *intervals_set_out* is ``None``.
        Otherwise returns ``None`` after saving.

    Raises
    ------
    ValueError
        If no interval arguments are provided, if an interval set does not
        exist, or if columns do not match exactly.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> i1 = pm.gextract("sparse_track", pm.gintervals(["1", "2"], 1000, 4000))  # doctest: +SKIP
    >>> i2 = pm.gextract("sparse_track", pm.gintervals(["2", "X"], 2000, 5000))  # doctest: +SKIP
    >>> pm.gintervals_save(i2, "tmp_intervs")  # doctest: +SKIP
    >>> pm.gintervals_rbind(i1, "tmp_intervs")  # doctest: +SKIP
    >>> pm.gintervals_rm("tmp_intervs", force=True)  # doctest: +SKIP

    See Also
    --------
    gintervals_load : Load a named interval set.
    gintervals_save : Save intervals as a named set.
    gintervals_canonic : Merge overlapping intervals within one set.
    """
    if not intervals:
        raise ValueError("Usage: gintervals_rbind([intervals]+, intervals_set_out=None)")

    _checkroot()

    loaded = []
    expected_cols = None
    for idx, item in enumerate(intervals):
        if isinstance(item, str):
            if not gintervals_exists(item):
                raise ValueError(f"Intervals set '{item}' does not exist")
            df = gintervals_load(item)
        elif isinstance(item, _pandas.DataFrame):
            df = item
        else:
            raise TypeError(
                f"intervals argument {idx + 1} must be DataFrame or interval set name"
            )

        if df is None or len(df) == 0:
            continue

        cols = list(df.columns)
        if expected_cols is None:
            expected_cols = cols
        elif cols != expected_cols:
            raise ValueError("Cannot rbind interval sets: columns differ")

        loaded.append(df)

    if not loaded:
        return None

    result = _pandas.concat(loaded, ignore_index=True, sort=False)
    if intervals_set_out is not None:
        gintervals_save(result, intervals_set_out)
        return None
    return result


def gintervals_mark_overlaps(
    intervals: pd.DataFrame | str,
    group_col: str = "overlap_group",
    unify_touching_intervals: bool = True,
) -> pd.DataFrame:
    """
    Mark groups of overlapping intervals with a shared group ID.

    Each interval in the input is assigned an integer group identifier.
    Intervals that overlap (or touch, when *unify_touching_intervals* is
    ``True``) share the same group ID.

    Parameters
    ----------
    intervals : DataFrame
        1D intervals with columns ``chrom``, ``start``, ``end`` and
        any additional data columns.
    group_col : str, default ``"overlap_group"``
        Name of the column to store group IDs.
    unify_touching_intervals : bool, default True
        Whether touching intervals (``end == start``) are considered
        overlapping.

    Returns
    -------
    DataFrame
        The original *intervals* with an added *group_col* column.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import pandas as pd
    >>> intervs = pd.DataFrame({
    ...     "chrom": ["1", "1", "1", "1"],
    ...     "start": [11000, 100, 10000, 10500],
    ...     "end":   [12000, 200, 13000, 10600],
    ...     "data":  [10, 20, 30, 40],
    ... })
    >>> pm.gintervals_mark_overlaps(intervs)  # doctest: +SKIP

    See Also
    --------
    gintervals_canonic : Merge overlapping intervals.
    gintervals_intersect : Intersection of two interval sets.
    gintervals_annotate : Annotate intervals with nearest-neighbor columns.
    """
    intervals = _resolve_intervals(intervals)
    if intervals is None or len(intervals) == 0:
        raise ValueError("intervals cannot be None or empty")
    assert isinstance(intervals, pd.DataFrame)

    _checkroot()

    canon = gintervals_canonic(intervals, unify_touching_intervals)
    if canon is None:
        result = intervals.copy()
        result[group_col] = 0
        return result

    mapping = canon.attrs.get("mapping")
    if mapping is None:
        raise RuntimeError("gintervals_canonic did not return a mapping attribute")

    result = intervals.copy()
    # mapping is indexed by sorted order; we need to map back to original order
    # gintervals_canonic sorts by (chrom, start), so recreate that sort order
    sort_idx = intervals[["chrom", "start"]].copy()
    sort_idx["_orig_idx"] = _numpy.arange(len(intervals))
    sort_idx = sort_idx.sort_values(["chrom", "start"]).reset_index(drop=True)

    # mapping[i] corresponds to sorted interval i -> canonical interval index
    # Distribute back to original order
    group_ids = _numpy.empty(len(intervals), dtype=_numpy.int64)
    group_ids[sort_idx["_orig_idx"].values] = mapping

    result[group_col] = group_ids
    return result


def gintervals_annotate(
    intervals: pd.DataFrame | str,
    annotation_intervals: pd.DataFrame | str,
    annotation_columns: list[str] | None = None,
    column_names: list[str] | None = None,
    dist_column: str | None = "dist",
    max_dist: float = float("inf"),
    na_value: Any = _numpy.nan,
    maxneighbors: int = 1,
    tie_method: str = "first",
    overwrite: bool = False,
    keep_order: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Annotate intervals with columns from the nearest annotation intervals.

    For each interval in *intervals*, the nearest neighbor in
    *annotation_intervals* is found (via :func:`gintervals_neighbors`),
    and the specified annotation columns are copied over.

    Parameters
    ----------
    intervals : DataFrame
        1D query intervals.
    annotation_intervals : DataFrame
        Source intervals containing annotation data.
    annotation_columns : list of str, optional
        Columns to copy from *annotation_intervals*.  ``None`` means all
        non-coordinate columns.
    column_names : list of str, optional
        Output names for the annotation columns (must match length of
        *annotation_columns*).
    dist_column : str or None, default ``"dist"``
        Name for the distance column.  ``None`` to omit.
    max_dist : float, default ``inf``
        Maximum absolute distance.  Annotations farther away are replaced
        with *na_value*.
    na_value : scalar or dict, default ``NaN``
        Fill value for annotations beyond *max_dist* or when no neighbor
        is found.  Can be a dict mapping column names to individual fill
        values.
    maxneighbors : int, default 1
        Number of nearest neighbors to consider.
    tie_method : str, default ``"first"``
        Tie-breaking strategy when multiple neighbors are equidistant.
        Only applies when ``maxneighbors > 1``.

        - ``"first"`` -- arbitrary but stable order (default).
        - ``"min.start"`` -- prefer the neighbor with the smaller start
          coordinate.
        - ``"min.end"`` -- prefer the neighbor with the smaller end
          coordinate.
    overwrite : bool, default False
        If ``True``, allow annotation columns to overwrite existing columns
        in *intervals*.
    keep_order : bool, default True
        Preserve original row order.
    **kwargs
        Additional keyword arguments passed to
        :func:`gintervals_neighbors` (e.g. ``mindist``, ``maxdist``).

    Returns
    -------
    DataFrame
        The input *intervals* with added annotation and distance columns.

    Raises
    ------
    ValueError
        If annotation columns conflict with existing columns and
        *overwrite* is ``False``.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals("1", [1000, 5000], [1100, 5050])
    >>> ann = pm.gintervals("1", [900, 5400], [950, 5500])
    >>> ann["remark"] = ["a", "b"]
    >>> ann["score"] = [10.0, 20.0]
    >>> pm.gintervals_annotate(intervs, ann)  # doctest: +SKIP
    >>> pm.gintervals_annotate(intervs, ann,
    ...     annotation_columns=["remark"],
    ...     column_names=["ann_remark"],
    ...     dist_column="ann_dist")  # doctest: +SKIP
    >>> pm.gintervals_annotate(intervs, ann,
    ...     annotation_columns=["remark"],
    ...     max_dist=200, na_value="no_ann")  # doctest: +SKIP
    >>> pm.gintervals_annotate(intervs, ann,
    ...     annotation_columns=["remark"],
    ...     maxneighbors=2,
    ...     tie_method="min.start")  # doctest: +SKIP

    See Also
    --------
    gintervals_neighbors : Find nearest neighbors between interval sets.
    gintervals_mark_overlaps : Mark groups of overlapping intervals.
    """
    intervals = _resolve_intervals(intervals)
    annotation_intervals = _resolve_intervals(annotation_intervals)
    if intervals is None or annotation_intervals is None:
        raise ValueError("intervals and annotation_intervals must not be None")
    assert isinstance(intervals, pd.DataFrame)
    assert isinstance(annotation_intervals, pd.DataFrame)

    _valid_tie_methods = ("first", "min.start", "min.end")
    if tie_method not in _valid_tie_methods:
        raise ValueError(
            f"tie_method must be one of {_valid_tie_methods}, got '{tie_method}'"
        )

    _checkroot()

    intervals = intervals.copy()
    annotation_intervals = annotation_intervals.copy()

    # Track original order
    if keep_order:
        intervals["_orig_order"] = _numpy.arange(len(intervals))

    # Determine annotation columns
    basic_cols = {"chrom", "start", "end", "chrom1", "start1", "end1",
                  "chrom2", "start2", "end2", "strand"}
    if annotation_columns is None:
        annotation_columns = [
            c for c in annotation_intervals.columns if c not in basic_cols
        ]

    # Validate annotation columns exist
    missing = [c for c in annotation_columns if c not in annotation_intervals.columns]
    if missing:
        raise ValueError(
            f"Annotation columns not found in annotation_intervals: "
            f"{', '.join(missing)}"
        )

    # Set up output column names
    if column_names is None:
        column_names = list(annotation_columns)
    elif len(column_names) != len(annotation_columns):
        raise ValueError(
            "column_names must have same length as annotation_columns"
        )

    # Check for column conflicts
    if not overwrite:
        existing_cols = set(intervals.columns)
        if dist_column is not None and dist_column in existing_cols:
            raise ValueError(
                f"Distance column '{dist_column}' already exists in intervals. "
                "Use overwrite=True or choose a different name."
            )
        conflicts = [c for c in column_names if c in existing_cols]
        if conflicts:
            raise ValueError(
                f"Annotation columns would overwrite existing columns: "
                f"{', '.join(conflicts)}. Use overwrite=True or provide "
                f"different column_names."
            )

    # Find neighbors
    nbrs = gintervals_neighbors(
        intervals, annotation_intervals,
        maxneighbors=maxneighbors,
        na_if_notfound=True,
        **kwargs
    )

    # Handle empty result
    if nbrs is None or len(nbrs) == 0:
        result = intervals.copy()
        n = len(result)
        for _i, col_name in enumerate(column_names):
            fill = na_value[col_name] if isinstance(na_value, dict) and col_name in na_value else na_value
            result[col_name] = [fill] * n
        if dist_column is not None:
            result[dist_column] = _numpy.nan
        if keep_order and "_orig_order" in result.columns:
            result = result.drop(columns=["_orig_order"])
        return result

    # Apply tie-breaking when maxneighbors > 1
    if tie_method != "first" and maxneighbors > 1 and "_orig_order" in nbrs.columns:
        # Determine neighbor coordinate column name
        # C++ appends "1" suffix when column names collide with query columns
        if tie_method == "min.start":
            tie_col = "start1" if "start1" in nbrs.columns else "start"
        else:  # min.end
            tie_col = "end1" if "end1" in nbrs.columns else "end"
        nbrs = nbrs.sort_values(
            ["_orig_order", "dist", tie_col],
            na_position="last",
        ).reset_index(drop=True)

    # Map annotation columns from neighbor result to output
    # The neighbor result has columns from both intervals1 and intervals2
    # Annotation columns from intervals2 may appear with "1" suffix if name conflicts
    result = nbrs.copy()

    # Build the annotation column mapping: src_col in result -> output name
    ann_col_map: dict[str, str | None] = {}  # output_name -> actual column in result
    for i, src_col in enumerate(annotation_columns):
        # gintervals_neighbors appends "1" suffix when columns conflict
        actual_col = src_col
        if src_col in intervals.columns:
            # The annotation column has been suffixed by neighbors
            candidate = src_col + "1"
            if candidate in result.columns:
                actual_col = candidate
        if actual_col not in result.columns:
            actual_col = src_col + "1"
        ann_col_map[column_names[i]] = actual_col if actual_col in result.columns else None

    # Build final output: start with original interval columns
    out_cols = [c for c in intervals.columns if c != "_orig_order"]
    if overwrite:
        out_cols = [c for c in out_cols if c not in column_names]

    output = _pandas.DataFrame()
    for col in out_cols:
        if col in result.columns:
            output[col] = result[col].to_numpy()

    # Add annotation columns with proper names
    for out_name, mapped_col in ann_col_map.items():
        if mapped_col is not None:
            output[out_name] = result[mapped_col].to_numpy()
        else:
            if isinstance(na_value, dict) and out_name in na_value:
                output[out_name] = na_value[out_name]
            else:
                output[out_name] = na_value

    # Add distance column
    if dist_column is not None and "dist" in result.columns:
        output[dist_column] = result["dist"].values

    # Apply distance threshold
    if max_dist < float("inf") and dist_column is not None and dist_column in output.columns:
        beyond = output[dist_column].abs() > max_dist
        for out_name in ann_col_map:
            if out_name in output.columns:
                fill = na_value[out_name] if isinstance(na_value, dict) and out_name in na_value else na_value
                output.loc[beyond, out_name] = fill

    # Restore original order
    if keep_order and "_orig_order" in result.columns:
        output["_orig_order"] = result["_orig_order"].values
        output = output.sort_values("_orig_order").reset_index(drop=True)
        output = output.drop(columns=["_orig_order"])
    elif "_orig_order" in output.columns:
        output = output.drop(columns=["_orig_order"])

    return output


def gintervals_normalize(
    intervals: pd.DataFrame,
    size: int | list[int] | Any,
    intervals_set_out: str | None = None,
) -> pd.DataFrame | None:
    """
    Normalize intervals to a specified size by centering.

    Each interval is resized to the target *size* while keeping its center
    position.  Results are clamped to chromosome boundaries.

    Parameters
    ----------
    intervals : DataFrame
        1D intervals with columns ``chrom``, ``start``, ``end``.
    size : int or array-like
        Target interval size(s) in basepairs.  Can be:

        - A single positive integer: all intervals get this size.
        - A vector matching the number of intervals: each interval gets its
          own target size.
        - A vector with ``len(intervals) == 1``: the single interval is
          replicated once per size (one-to-many expansion).

    Returns
    -------
    DataFrame
        Normalized intervals.

    Raises
    ------
    ValueError
        If *size* contains non-positive values or if vector length does not
        match the number of intervals.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals("1", [1000, 5000], [2000, 6000])
    >>> pm.gintervals_normalize(intervs, 500)  # doctest: +SKIP
    >>> pm.gintervals_normalize(intervs, [500, 1000])  # doctest: +SKIP
    >>> pm.gintervals_normalize(pm.gintervals("1", 1000, 2000), [500, 1000, 1500])  # doctest: +SKIP

    See Also
    --------
    gintervals_force_range : Clamp intervals to chromosome boundaries.
    gintervals_window : Create intervals centered on positions.
    """
    if intervals is None or len(intervals) == 0:
        raise ValueError("intervals cannot be None or empty")

    _checkroot()

    # Validate 2D intervals
    if "chrom1" in intervals.columns:
        raise ValueError("gintervals_normalize does not support 2D intervals")

    # Normalize size to numpy array
    size = _numpy.asarray(size, dtype=_numpy.int64).ravel()
    if _numpy.any(size <= 0):
        raise ValueError("size must contain only positive values")

    n_intervals = len(intervals)
    n_sizes = len(size)

    # Handle one-to-many: single interval, multiple sizes
    if n_sizes > 1 and n_intervals == 1:
        intervals = _pandas.concat([intervals] * n_sizes, ignore_index=True)
        n_intervals = n_sizes
    elif n_sizes > 1 and n_sizes != n_intervals:
        raise ValueError(
            f"Length of size vector ({n_sizes}) must match number of "
            f"intervals ({n_intervals}) or intervals must have exactly "
            f"1 row for one-to-many expansion"
        )

    # Broadcast scalar
    if len(size) == 1:
        size = _numpy.full(n_intervals, size[0], dtype=_numpy.int64)

    # Get chromosome sizes
    all_intervals = gintervals_all()
    chrom_sizes = dict(
        zip(
            all_intervals["chrom"].astype(str).tolist(),
            all_intervals["end"].astype(int).tolist(), strict=False,
        )
    )

    # Compute new intervals
    starts = intervals["start"].values.astype(_numpy.int64)
    ends = intervals["end"].values.astype(_numpy.int64)
    chroms = intervals["chrom"].to_numpy()

    centers = (starts + ends) / 2.0
    half = size / 2.0

    new_starts = _numpy.floor(centers - half).astype(_numpy.int64)
    new_ends = new_starts + size

    # Clamp to chromosome boundaries
    for i in range(n_intervals):
        chrom = str(chroms[i])
        chrom_sz = chrom_sizes.get(chrom, 0)
        if new_starts[i] < 0:
            new_starts[i] = 0
            new_ends[i] = min(size[i], chrom_sz)
        if new_ends[i] > chrom_sz:
            new_ends[i] = chrom_sz
            new_starts[i] = max(0, chrom_sz - size[i])

    # Build result preserving extra columns
    result = _pandas.DataFrame({
        "chrom": chroms,
        "start": new_starts,
        "end": new_ends,
    })

    # Preserve extra columns
    basic_cols = {"chrom", "start", "end"}
    for col in intervals.columns:
        if col not in basic_cols:
            result[col] = intervals[col].to_numpy()

    if intervals_set_out is not None:
        gintervals_save(result[["chrom", "start", "end"]], intervals_set_out)
        return None

    return result


# Routing threshold for gintervals_random: dispatch to the C++ fast path when
# the genome has many contigs OR is large. Below this, the Python path is
# competitive and lets users keep numpy-seeded reproducibility.
_GINTERVALS_RANDOM_CPP_MIN_CHROMS = 1000
_GINTERVALS_RANDOM_CPP_MIN_BP = 10_000_000


def _gintervals_random_python(
    size: int,
    n: int,
    dist_from_edge: float,
    all_genome: pd.DataFrame,
    mask: pd.DataFrame | None,
) -> pd.DataFrame:
    """Reference Python implementation (uses numpy's global RNG).

    Kept as the small-genome path. Output is deterministic against
    ``numpy.random.seed`` set by the caller.
    """
    segments = []  # list of (chrom_name, seg_start, seg_end) where seg_end is exclusive

    for row in all_genome.itertuples(index=False):
        chrom = row.chrom
        chrom_size = int(row.end)
        lo = int(dist_from_edge)
        hi = chrom_size - int(dist_from_edge) - size

        if hi < lo:
            continue  # chromosome too short

        if mask is None:
            segments.append((chrom, lo, hi + 1))
        else:
            chrom_filter = mask[mask["chrom"] == chrom]
            if len(chrom_filter) == 0:
                segments.append((chrom, lo, hi + 1))
                continue

            cur_lo = lo
            for frow in chrom_filter.itertuples(index=False):
                fs = int(frow.start)
                fe = int(frow.end)
                excl_lo = max(cur_lo, fs - size + 1)
                excl_hi = min(hi, fe - 1)
                if excl_lo > cur_lo:
                    seg_end = min(excl_lo, hi + 1)
                    if seg_end > cur_lo:
                        segments.append((chrom, cur_lo, seg_end))
                if excl_hi >= cur_lo:
                    cur_lo = excl_hi + 1
            if cur_lo <= hi:
                segments.append((chrom, cur_lo, hi + 1))

    if not segments:
        raise ValueError(
            f"No valid genomic positions for intervals of size {size} "
            f"with dist_from_edge {dist_from_edge}"
        )

    seg_chroms = [s[0] for s in segments]
    seg_starts = _numpy.array([s[1] for s in segments], dtype=_numpy.int64)
    seg_lengths = _numpy.array([s[2] - s[1] for s in segments], dtype=_numpy.int64)

    total_length = seg_lengths.sum()
    if total_length == 0:
        raise ValueError("No valid genomic positions for random intervals")

    cum_lengths = _numpy.cumsum(seg_lengths)
    rand_positions = _numpy.random.randint(0, total_length, size=n)

    seg_indices = _numpy.searchsorted(cum_lengths, rand_positions, side="right")
    offsets = rand_positions - _numpy.concatenate([[0], cum_lengths[:-1]])[seg_indices]

    result_chroms = [seg_chroms[i] for i in seg_indices]
    result_starts = seg_starts[seg_indices] + offsets
    result_ends = result_starts + size

    return _pandas.DataFrame({
        "chrom": result_chroms,
        "start": result_starts,
        "end": result_ends,
    })


def _gintervals_random_cpp(
    size: int,
    n: int,
    dist_from_edge: float,
    all_genome: pd.DataFrame,
    mask: pd.DataFrame | None,
    seed: int,
) -> pd.DataFrame:
    """C++ fast path. RNG is std::mt19937_64 seeded with ``seed``.

    The output is NOT bit-identical to the Python path (different RNG).
    For statistical purposes (mean position, per-chrom counts) the two
    paths are equivalent.
    """
    chrom_pm = _df2pymisha(all_genome[["chrom", "start", "end"]])
    filter_pm = (
        _df2pymisha(mask[["chrom", "start", "end"]])
        if mask is not None and len(mask) > 0
        else None
    )
    result = _pymisha.pm_intervals_random(
        int(size),
        int(n),
        float(dist_from_edge),
        chrom_pm,
        filter_pm,
        int(seed),
    )
    if result is None:
        raise ValueError("pm_intervals_random returned no result")
    return _pandas.DataFrame(result)


def gintervals_random(
    size: int,
    n: int,
    dist_from_edge: float = 3_000_000,
    chromosomes: list[str] | None = None,
    mask: pd.DataFrame | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate random genomic intervals.

    Intervals are sampled uniformly from the genome (after excluding
    chromosome edges and optional filter regions).  Each interval is
    exactly *size* basepairs.

    Routing
    -------
    For million-contig genomes the implementation dispatches to a C++
    fast path that avoids per-chrom Python overhead. The heuristic is
    "> 1000 contigs OR > 10M bp total". The C++ path uses
    ``std::mt19937_64`` seeded with *seed* (default 60427) and is NOT
    bit-identical to the Python path, which uses numpy's global RNG.
    Set *seed* explicitly to make C++ runs reproducible. To force the
    Python path (e.g. for numpy-seed reproducibility on a large genome),
    leave *seed* as ``None`` and the function falls back to the Python
    implementation only when the genome is small.

    Parameters
    ----------
    size : int
        Interval size in basepairs (must be positive).
    n : int
        Number of intervals to generate (must be positive).
    dist_from_edge : float, default 3_000_000
        Minimum distance from chromosome boundaries.
    chromosomes : list of str, optional
        Restrict sampling to these chromosomes.
    mask : DataFrame, optional
        Intervals to exclude from sampling (columns ``chrom``, ``start``,
        ``end``). Sampled intervals are guaranteed not to overlap any
        masked region.
    seed : int, optional
        Seed for the C++ RNG. Defaults to 60427 (project convention).
        Ignored when the Python fallback path is taken; set
        ``numpy.random.seed`` for that path.
    filter : DataFrame, optional
        Backward-compatible alias for ``mask``.

    Returns
    -------
    DataFrame
        DataFrame with columns ``chrom``, ``start``, ``end``.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_random(100, 1000)  # doctest: +SKIP
    >>> pm.gintervals_random(100, 1000, chromosomes=["1"])  # doctest: +SKIP
    >>> import numpy as np; np.random.seed(60427)
    >>> pm.gintervals_random(100, 50)  # doctest: +SKIP

    See Also
    --------
    gintervals : Create intervals manually.
    gintervals_all : Return full-genome intervals.
    """
    _checkroot()

    if "filter" in kwargs:
        if mask is not None:
            raise ValueError("Specify only one of 'mask' or 'filter'")
        mask = kwargs.pop("filter")
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {bad}")

    if not isinstance(size, (int, _numpy.integer)) or size <= 0:
        raise ValueError("size must be a positive integer")
    if not isinstance(n, (int, _numpy.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if dist_from_edge < 0:
        raise ValueError("dist_from_edge must be non-negative")

    size = int(size)
    n = int(n)
    dist_from_edge = float(dist_from_edge)

    all_genome = gintervals_all()

    if chromosomes is not None:
        if not isinstance(chromosomes, (list, tuple)):
            raise ValueError("chromosomes must be a list of strings")
        chromosomes = list(_normalize_chroms(chromosomes))
        all_genome = all_genome[all_genome["chrom"].isin(chromosomes)]
        if len(all_genome) == 0:
            raise ValueError(
                f"No chromosomes named {', '.join(chromosomes)} found in the genome"
            )

    if mask is not None:
        if not isinstance(mask, _pandas.DataFrame):
            raise ValueError("mask must be a DataFrame")
        if not {"chrom", "start", "end"}.issubset(mask.columns):
            raise ValueError("mask must have columns: chrom, start, end")
        if len(mask) > 0:
            mask = mask.copy()
            mask["chrom"] = _normalize_chroms(mask["chrom"].astype(str).tolist())
            if chromosomes is not None:
                mask = mask[mask["chrom"].isin(chromosomes)]
            mask = gintervals_canonic(mask)
        if mask is None or len(mask) == 0:
            mask = None

    # Heuristic: dispatch to C++ for million-contig or very large genomes.
    total_chroms = len(all_genome)
    total_bp = int(all_genome["end"].sum()) if total_chroms > 0 else 0
    use_cpp = (
        total_chroms > _GINTERVALS_RANDOM_CPP_MIN_CHROMS
        or total_bp > _GINTERVALS_RANDOM_CPP_MIN_BP
    )

    if use_cpp:
        return _gintervals_random_cpp(
            size, n, dist_from_edge, all_genome, mask,
            seed if seed is not None else 60427,
        )
    return _gintervals_random_python(size, n, dist_from_edge, all_genome, mask)


def gintervals_rm(intervals_set: str, force: bool = False) -> None:
    """
    Remove a named interval set from the database.

    Parameters
    ----------
    intervals_set : str
        Name of the interval set to remove.
    force : bool, default False
        If True, do not raise an error if the interval set does not exist.

    Raises
    ------
    ValueError
        If the interval set does not exist and force is False.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_rm("my_intervals")  # doctest: +SKIP

    Returns
    -------
    None

    See Also
    --------
    gintervals_save : Save intervals as a named set.
    gintervals_load : Load a named interval set.
    gintervals_exists : Check if a named interval set exists.
    gintervals_ls : List named interval sets.
    """
    _checkroot()

    from . import _shared
    assert _shared._GROOT is not None

    groot = _shared._GROOT
    path_part = intervals_set.replace(".", "/")
    interv_path = Path(groot) / "tracks" / f"{path_part}.interv"

    if not interv_path.exists():
        if force:
            return
        raise ValueError(f"Intervals set '{intervals_set}' does not exist")

    # Remove the file (or directory for big interval sets)
    if interv_path.is_dir():
        if not _gdb_trash(interv_path, async_unlink=True):
            raise RuntimeError(
                f"failed to remove intervals directory: {interv_path}"
            )
    else:
        interv_path.unlink()
        # Also remove companion .iattr file for small interval sets
        iattr_path = interv_path.with_suffix(".iattr")
        if iattr_path.exists():
            iattr_path.unlink()

    # Drop from C++ cache so gintervals_ls() / gintervals_exists() see
    # the removal immediately (no pm_dbreload required).
    with _contextlib.suppress(Exception):
        _pymisha.pm_interv_unregister(intervals_set)


def _open_genes_file(path_or_url: str) -> tuple[IO[str], str | None]:
    """Open a genes/annotations file, handling URLs, .gz, and plain files.

    Returns an open text-mode file handle (caller must close).
    Also returns a temporary directory path that must be cleaned up, or None.
    """
    tmpdir = None
    filepath = path_or_url

    # Download if URL
    if path_or_url.startswith(("ftp://", "http://", "https://")):
        tmpdir = tempfile.mkdtemp(prefix="pymisha_genes_")
        local_name = os.path.basename(path_or_url)
        filepath = os.path.join(tmpdir, local_name)
        urllib.request.urlretrieve(path_or_url, filepath)

    # Decompress if gzipped
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt"), tmpdir
    return open(filepath), tmpdir  # noqa: SIM115


def _parse_annots_file(annots_file: IO[str], num_annots: int) -> dict[str, list[str]]:
    """Parse annotation file. Returns dict mapping gene_id -> list of annotation values.

    Each line is tab-separated. The first column is the gene ID, followed by
    annotation columns. The total number of fields per line must equal
    ``num_annots``.
    """
    id2annots = {}
    lineno = 0
    for raw_line in annots_file:
        lineno += 1
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        fields = line.split("\t", num_annots)
        if len(fields) < 1 or not fields[0]:
            raise ValueError(
                f"Annotation file, line {lineno}: invalid format"
            )
        if len(fields) != num_annots:
            raise ValueError(
                f"Annotation file, line {lineno}: number of annotation "
                f"columns ({len(fields)}) does not match annots_names "
                f"length ({num_annots})"
            )
        gene_id = fields[0]
        if gene_id in id2annots:
            raise ValueError(
                f"Annotation file: id {gene_id} appears more than once"
            )
        id2annots[gene_id] = fields
    return id2annots


def _parse_genes_file(
    genes_file: IO[str],
    id2annots: dict[str, list[str]],
    known_chroms: set[str],
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    """Parse a UCSC knownGene-format file and return raw interval lists.

    Parameters
    ----------
    genes_file : file-like
        Open text file in knownGene format (12 tab-separated columns).
    id2annots : dict
        Mapping from gene ID to annotation list (or empty dict).
    known_chroms : set
        Set of normalized chromosome names in the current database.

    Returns
    -------
    tuple of four lists
        (tss_records, exon_records, utr3_records, utr5_records).
        Each record is (chrom, start, end, strand, annots_or_None).
    """
    ID, CHROM, STRAND, TXSTART, TXEND = 0, 1, 2, 3, 4
    _CDSSTART, _CDSEND, EXONCOUNT, EXONSTARTS, EXONENDS = 5, 6, 7, 8, 9
    # PROTEINID=10, ALIGNID=11
    NUM_COLS = 12

    tss = []
    exons = []
    utr3 = []
    utr5 = []

    lineno = 0
    for raw_line in genes_file:
        lineno += 1
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != NUM_COLS:
            raise ValueError(
                f"Genes file, line {lineno}: expected {NUM_COLS} columns, "
                f"got {len(fields)}"
            )

        gene_id = fields[ID]
        chrom_raw = fields[CHROM]
        strand_str = fields[STRAND]

        if not gene_id or not chrom_raw or not strand_str:
            raise ValueError(
                f"Genes file, line {lineno}: invalid file format"
            )

        # Normalize chromosome name
        try:
            chrom_norm = _normalize_chroms([chrom_raw])[0]
        except Exception:
            chrom_norm = chrom_raw

        # Skip chromosomes not in the database
        if chrom_norm not in known_chroms:
            continue

        # Parse strand
        if strand_str == "+":
            strand = 1
        elif strand_str == "-":
            strand = -1
        else:
            raise ValueError(
                f"Genes file, line {lineno}: invalid strand value "
                f"'{strand_str}'"
            )

        # Parse coordinates
        try:
            txstart = int(fields[TXSTART])
            txend = int(fields[TXEND])
        except ValueError:
            raise ValueError(
                f"Genes file, line {lineno}: invalid txStart/txEnd value"
            ) from None

        try:
            exoncount = int(fields[EXONCOUNT])
        except ValueError:
            raise ValueError(
                f"Genes file, line {lineno}: invalid exonCount value"
            ) from None
        if exoncount < 0:
            raise ValueError(
                f"Genes file, line {lineno}: invalid exonCount value"
            )

        # Parse exon starts (comma-separated, trailing comma)
        exon_starts_str = fields[EXONSTARTS]
        exon_ends_str = fields[EXONENDS]
        try:
            exon_starts = [
                int(x)
                for x in exon_starts_str.rstrip(",").split(",")
                if x
            ]
            exon_ends_list = [
                int(x)
                for x in exon_ends_str.rstrip(",").split(",")
                if x
            ]
        except ValueError:
            raise ValueError(
                f"Genes file, line {lineno}: invalid exonStarts/exonEnds "
                f"value"
            ) from None

        if len(exon_starts) != exoncount:
            raise ValueError(
                f"Genes file, line {lineno}: number of exonStarts values "
                f"does not match exonCount"
            )
        if len(exon_ends_list) != exoncount:
            raise ValueError(
                f"Genes file, line {lineno}: number of exonEnds values "
                f"does not match exonCount"
            )

        # Get annotations for this gene
        annots = id2annots.get(gene_id)

        # TSS
        if strand == 1:
            tss_start = txstart
            tss_end = txstart + 1
        else:
            tss_start = txend - 1
            tss_end = txend
        tss.append((chrom_norm, tss_start, tss_end, strand, annots))

        # Exons
        for i in range(exoncount):
            exons.append(
                (chrom_norm, exon_starts[i], exon_ends_list[i], strand, annots)
            )

        # UTR3
        if txend >= 0 and exoncount > 0:
            if strand == 1:
                utr3_start = exon_ends_list[exoncount - 1] - 1
                utr3_end = txend
            else:
                utr3_start = txstart
                utr3_end = exon_starts[0] + 1
            utr3.append((chrom_norm, utr3_start, utr3_end, strand, annots))

        # UTR5
        if txstart >= 0 and exoncount > 0:
            if strand == 1:
                utr5_start = txstart
                utr5_end = exon_starts[0] + 1
            else:
                utr5_start = exon_ends_list[exoncount - 1] - 1
                utr5_end = txend
            utr5.append((chrom_norm, utr5_start, utr5_end, strand, annots))

    return tss, exons, utr3, utr5


def _unify_intervals(
    records: list[Any],
    annots_names: list[str] | None,
) -> pd.DataFrame | None:
    """Unify (merge) overlapping intervals, combining strand and annotations.

    Follows R misha behaviour: overlapping intervals on the same chromosome are
    merged. If strands differ, strand is set to 0. Annotations from
    overlapping intervals are concatenated with semicolons (duplicates removed).

    Parameters
    ----------
    records : list of tuple
        Each tuple is (chrom, start, end, strand, annots_or_None).
    annots_names : list of str or None
        Annotation column names.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns chrom, start, end, strand, plus annotation
        columns. Returns None if records is empty.
    """
    if not records:
        return None

    num_annots = len(annots_names) if annots_names else 0

    # Sort by (chrom, start, end) -- same as R GIntervals::sort()
    records.sort(key=lambda r: (r[0], r[1], r[2]))

    # Merge overlapping intervals
    merged_chroms = []
    merged_starts = []
    merged_ends = []
    merged_strands = []
    merged_annots: list[list[str]] = [[] for _ in range(num_annots)]  # list of lists

    cur_chrom, cur_start, cur_end, cur_strand, cur_annot = records[0]
    # Annotation accumulator: sets of unique values per column
    annot_sets: list[set[str]] = [set() for _ in range(num_annots)]
    if cur_annot and num_annots > 0:
        for j in range(num_annots):
            if j < len(cur_annot) and cur_annot[j]:
                annot_sets[j].add(cur_annot[j])

    def _flush():
        merged_chroms.append(cur_chrom)
        merged_starts.append(cur_start)
        merged_ends.append(cur_end)
        merged_strands.append(cur_strand)
        for j in range(num_annots):
            # Concatenate sorted unique annotations with semicolons
            merged_annots[j].append(";".join(sorted(annot_sets[j])))

    for i in range(1, len(records)):
        chrom, start, end, strand, annot = records[i]
        if chrom != cur_chrom or start >= cur_end:
            # No overlap: flush current interval
            _flush()
            cur_chrom, cur_start, cur_end, cur_strand = (
                chrom,
                start,
                end,
                strand,
            )
            annot_sets = [set() for _ in range(num_annots)]
            if annot and num_annots > 0:
                for j in range(num_annots):
                    if j < len(annot) and annot[j]:
                        annot_sets[j].add(annot[j])
        else:
            # Overlap: extend
            if cur_strand != strand:
                cur_strand = 0
            if end > cur_end:
                cur_end = end
            if annot and num_annots > 0:
                for j in range(num_annots):
                    if j < len(annot) and annot[j]:
                        annot_sets[j].add(annot[j])

    # Flush last interval
    _flush()

    # Build DataFrame
    pd = _pandas
    df = pd.DataFrame(
        {
            "chrom": merged_chroms,
            "start": merged_starts,
            "end": merged_ends,
            "strand": merged_strands,
        }
    )
    if annots_names is not None:
        for j in range(num_annots):
            df[annots_names[j]] = merged_annots[j]

    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["strand"] = df["strand"].astype(float)

    return df


def gintervals_import_genes(
    genes_file: str,
    annots_file: str | None = None,
    annots_names: list[str] | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Import gene annotations from a UCSC knownGene-format file.

    Reads gene definitions from ``genes_file`` and produces four sets of
    intervals: TSS, exons, 3'UTR, and 5'UTR.  A ``strand`` column is
    included (``1`` for "+", ``-1`` for "-").

    If ``annots_file`` is provided, annotations are attached to the
    intervals. ``annots_names`` must be supplied when ``annots_file`` is
    given.

    Both ``genes_file`` and ``annots_file`` may be local file paths or URLs
    (http, https, ftp). Gzipped files (``.gz``) are handled automatically.

    Overlapping intervals within each set are unified (merged). When two
    overlapping intervals have different strands, the merged strand is set
    to ``0``. Annotations from overlapping intervals are concatenated with
    semicolons; duplicate annotation values are removed.

    Parameters
    ----------
    genes_file : str
        Path or URL to a knownGene-format file (12 tab-separated columns).
    annots_file : str, optional
        Path or URL to an annotation file. The first column is the gene ID
        (matching ``genes_file``), followed by annotation columns.
    annots_names : list of str, optional
        Names for the annotation columns. Required when ``annots_file`` is
        given. The length must match the number of columns in the annotation
        file.

    Returns
    -------
    dict
        Dictionary with keys ``"tss"``, ``"exons"``, ``"utr3"``, ``"utr5"``.
        Each value is a :class:`~pandas.DataFrame` with columns ``chrom``,
        ``start``, ``end``, ``strand`` (and any annotation columns), or
        ``None`` if the corresponding set is empty.

    Raises
    ------
    ValueError
        If ``genes_file`` is None, or ``annots_file`` is given without
        ``annots_names``, or file parsing fails.

    See Also
    --------
    gintervals : Create a custom set of 1D intervals.
    gintervals_save : Save intervals to the database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gintervals_import_genes("genes.txt")  # doctest: +SKIP
    >>> sorted(result.keys())  # doctest: +SKIP
    ['exons', 'tss', 'utr3', 'utr5']
    """
    _checkroot()

    if genes_file is None:
        raise ValueError(
            "Usage: gintervals_import_genes(genes_file, annots_file=None, "
            "annots_names=None)"
        )

    if annots_file is not None and annots_names is None:
        raise ValueError(
            "annots_names argument cannot be None if annots_file is specified"
        )

    if annots_names is not None and not isinstance(annots_names, (list, tuple)):
        raise ValueError("annots_names argument must be a list of strings")

    # Get known chromosomes from the current database
    all_intervs = gintervals_all()
    known_chroms = set(all_intervs["chrom"].tolist())

    # Parse annotations file if provided
    id2annots: dict[str, list[str]] = {}
    annots_tmpdir = None
    if annots_file is not None:
        assert annots_names is not None
        num_annots = len(annots_names)
        fh, annots_tmpdir = _open_genes_file(annots_file)
        try:
            id2annots = _parse_annots_file(fh, num_annots)
        finally:
            fh.close()
            if annots_tmpdir:
                shutil.rmtree(annots_tmpdir, ignore_errors=True)

    # Parse genes file
    fh, genes_tmpdir = _open_genes_file(genes_file)
    try:
        tss_records, exon_records, utr3_records, utr5_records = (
            _parse_genes_file(fh, id2annots, known_chroms)
        )
    finally:
        fh.close()
        if genes_tmpdir:
            shutil.rmtree(genes_tmpdir, ignore_errors=True)

    # Unify overlapping intervals for each set
    effective_annots = annots_names if annots_names else None
    return {
        "tss": _unify_intervals(tss_records, effective_annots),
        "exons": _unify_intervals(exon_records, effective_annots),
        "utr3": _unify_intervals(utr3_records, effective_annots),
        "utr5": _unify_intervals(utr5_records, effective_annots),
    }
