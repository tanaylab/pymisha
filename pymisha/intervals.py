"""Interval creation and operations."""

import contextlib as _contextlib
import gzip
import os
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from ._crc64 import (
    crc64_finalize as _crc64_finalize,
)
from ._crc64 import (
    crc64_incremental as _crc64_incremental,
)
from ._crc64 import (
    crc64_init as _crc64_init,
)
from ._name_validation import validate_dotted_name
from ._shared import (
    CONFIG,
    _checkroot,
    _df2pymisha,
    _numpy,
    _pandas,
    _progress_context,
    _pymisha,
    _pymisha2df,
)


def _normalize_chroms(chroms):
    if chroms is None:
        return chroms
    if isinstance(chroms, (str, int)):
        chroms = [chroms]
    return list(_pymisha.pm_normalize_chroms(list(chroms)))


def gintervals_all():
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


def _intervset_path(intervals_set):
    root = gintervals_dataset(intervals_set)
    if root is None:
        raise ValueError(f"Intervals set '{intervals_set}' does not exist")
    path_part = intervals_set.replace(".", "/")
    return Path(root) / "tracks" / f"{path_part}.interv"


def _decode_r_obj_to_bytes(obj_path):
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required to load interval sets. "
            "Install with: pip install pyreadr"
        ) from exc

    result = pyreadr.read_r(str(obj_path))
    if not result:
        raise ValueError(f"Could not read serialized object from {obj_path}")
    # pyreadr returns an OrderedDict; take the first value
    return list(result.values())[0]


def _decode_intervals_meta(meta_path):
    # .meta is a serialized list(stats=..., zeroline=...)
    # pyreadr cannot decode non-dataframe objects, so use Rscript.
    with tempfile.TemporaryDirectory(prefix="pymisha-meta-") as tmpdir:
        stats_path = Path(tmpdir) / "stats.rds"
        zero_path = Path(tmpdir) / "zeroline.rds"
        r_cmd = (
            "f<-commandArgs(TRUE)[1]; out_stats<-commandArgs(TRUE)[2]; out_zero<-commandArgs(TRUE)[3]; "
            "con<-file(f,'rb'); obj<-unserialize(con); close(con); "
            "saveRDS(obj$stats, out_stats); saveRDS(obj$zeroline, out_zero)"
        )
        rscript = shutil.which("Rscript")
        if not rscript:
            raise RuntimeError(
                "Rscript is required to load legacy intervals metadata (.meta). "
                "Install Rscript or convert the intervals set to indexed format."
            )
        try:
            subprocess.run(
                [rscript, "-e", r_cmd, str(meta_path), str(stats_path), str(zero_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(
                f"Failed to decode intervals metadata via Rscript: {stderr or exc}"
            ) from exc
        stats = _decode_r_obj_to_bytes(stats_path)
        zeroline = _decode_r_obj_to_bytes(zero_path)
    return stats, zeroline


def _intervset_is_bigset(intervals_set):
    path = _intervset_path(intervals_set)
    return path.exists() and path.is_dir()


def _intervset_is_indexed(path, allow_updates=True):
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


def _intervset_index_paths(path):
    return {
        "idx1d": path / "intervals.idx",
        "dat1d": path / "intervals.dat",
        "idx2d": path / "intervals2d.idx",
        "dat2d": path / "intervals2d.dat",
    }


def _load_index_entries_1d(idx_path):
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


def _load_index_entries_2d(idx_path):
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


def gintervals(chroms, starts=0, ends=-1, strand=None):
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
    strand : int or list of int, optional
        Strand information (``-1``, ``0``, or ``1``).
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
        if isinstance(strand, int):
            strand = [strand]
        strand = list(strand)
        n = len(result_chroms)
        if len(strand) == 1:
            strand = strand * n
        if len(strand) != n:
            raise ValueError("strand must have the same length as other arguments")

        result_strands = []
        for s in strand:
            if s not in (-1, 0, 1):
                raise ValueError(f"Invalid strand value {s}: must be -1, 0, or 1")
            result_strands.append(s)

    df = _pandas.DataFrame({
        'chrom': result_chroms,
        'start': result_starts,
        'end': result_ends
    })

    if result_strands is not None:
        df['strand'] = result_strands

    return df.sort_values(['chrom', 'start']).reset_index(drop=True)


def _make_1d_intervals(chroms, starts, ends):
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
    if len(chroms) == 1:
        chroms = chroms * n
    if len(starts) == 1:
        starts = starts * n
    if len(ends) == 1:
        ends = ends * n

    if not (len(chroms) == len(starts) == len(ends)):
        raise ValueError("chroms, starts, and ends must have the same length")

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


def gintervals_2d(chroms1, starts1=0, ends1=-1, chroms2=None, starts2=0, ends2=-1):
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

    if len(c1) != len(c2):
        raise ValueError("chroms1 and chroms2 must produce the same number of intervals")

    df = _pandas.DataFrame({
        'chrom1': c1, 'start1': s1, 'end1': e1,
        'chrom2': c2, 'start2': s2, 'end2': e2,
    })

    return df.sort_values(['chrom1', 'start1', 'chrom2', 'start2']).reset_index(drop=True)


def gintervals_2d_all(mode="diagonal"):
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
            'chrom1': intervals['chrom'].values,
            'start1': intervals['start'].values,
            'end1': intervals['end'].values,
            'chrom2': intervals['chrom'].values,
            'start2': intervals['start'].values,
            'end2': intervals['end'].values,
        })
    else:
        # Full cartesian product (vectorized)
        chrom = intervals["chrom"].to_numpy(copy=False)
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


def gintervals_2d_band_intersect(intervals, band, intervals_set_out=None):
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


def gintervals_from_tuples(rows, strand=None):
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


def gintervals_from_strings(regions):
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

    return gintervals(chroms, starts, ends, strands if has_strand else None)


def gintervals_from_bed(path, has_strand=False):
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

    rows = []
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


def gintervals_window(chroms, centers, half_width):
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


def gintervals_force_range(intervals):
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
    _checkroot()

    if intervals is None:
        raise ValueError("intervals cannot be None")

    if len(intervals) == 0:
        return None

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

    chrom_vals = intervals["chrom"].astype(str).tolist()
    with _contextlib.suppress(Exception):
        chrom_vals = _normalize_chroms(chrom_vals)

    for chrom, start, end in zip(
        chrom_vals,
        intervals["start"].tolist(),
        intervals["end"].tolist(), strict=False,
    ):

        if chrom not in chrom_sizes:
            continue

        chrom_size = chrom_sizes[chrom]
        start = max(0, start)
        end = min(chrom_size, end)

        if start < end:
            result_chroms.append(chrom)
            result_starts.append(start)
            result_ends.append(end)

    if not result_chroms:
        return None

    return _pandas.DataFrame({
        'chrom': result_chroms,
        'start': result_starts,
        'end': result_ends
    })


def _sort_intervals(intervals):
    return intervals.sort_values(['chrom', 'start', 'end']).reset_index(drop=True)


def _unify_overlaps(intervals, unify_touching=True):
    if intervals is None or len(intervals) == 0:
        return None

    intervals = _sort_intervals(intervals[["chrom", "start", "end"]].copy())

    result_chroms = []
    result_starts = []
    result_ends = []

    cur_chrom = intervals.iloc[0]['chrom']
    cur_start = intervals.iloc[0]['start']
    cur_end = intervals.iloc[0]['end']

    for i in range(1, len(intervals)):
        row = intervals.iloc[i]
        chrom = row['chrom']
        start = row['start']
        end = row['end']

        if chrom == cur_chrom:
            if unify_touching:
                if cur_end >= start:
                    cur_end = max(cur_end, end)
                    continue
            else:
                if cur_end > start:
                    cur_end = max(cur_end, end)
                    continue

        result_chroms.append(cur_chrom)
        result_starts.append(cur_start)
        result_ends.append(cur_end)

        cur_chrom = chrom
        cur_start = start
        cur_end = end

    result_chroms.append(cur_chrom)
    result_starts.append(cur_start)
    result_ends.append(cur_end)

    return _pandas.DataFrame({
        'chrom': result_chroms,
        'start': result_starts,
        'end': result_ends
    })


def _intervals_to_cpp(intervals):
    """Prepare intervals for C++ processing (convert Categorical chrom to string)."""
    df = intervals[['chrom', 'start', 'end']].copy()
    # Ensure chrom is string, not Categorical (C++ doesn't handle Categorical)
    if hasattr(df['chrom'], 'cat'):
        df['chrom'] = df['chrom'].astype(str)
    return _df2pymisha(df)


def gintervals_union(intervals1, intervals2):
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
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")

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

    return _pandas.DataFrame(result)


def gintervals_intersect(intervals1, intervals2):
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

    return _pandas.DataFrame(result)


def gintervals_diff(intervals1, intervals2):
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
    if intervals1 is None or intervals2 is None:
        raise ValueError("intervals1 and intervals2 cannot be None")

    if len(intervals1) == 0:
        return None
    if len(intervals2) == 0:
        return _sort_intervals(intervals1[['chrom', 'start', 'end']].copy())

    _checkroot()
    result = _pymisha.pm_intervals_diff(
        _intervals_to_cpp(intervals1),
        _intervals_to_cpp(intervals2)
    )

    if result is None or len(result['chrom']) == 0:
        return None

    return _pandas.DataFrame(result)


def gintervals_canonic(intervals, unify_touching_intervals=True):
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
    if intervals is None:
        raise ValueError("intervals cannot be None")
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


def gintervals_covered_bp(intervals):
    """
    Compute total basepairs covered by intervals.

    Overlapping intervals are merged before counting to avoid double-counting.

    Parameters
    ----------
    intervals : DataFrame
        Interval set with columns: chrom, start, end

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
    if intervals is None:
        raise ValueError("intervals cannot be None")
    if len(intervals) == 0:
        return 0

    _checkroot()
    return _pymisha.pm_intervals_covered_bp(
        _intervals_to_cpp(intervals)
    )


def gintervals_coverage_fraction(intervals1, intervals2=None):
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


def gintervals_neighbors(intervals1, intervals2, maxneighbors=1,
                         mindist=-1e9, maxdist=1e9, na_if_notfound=False,
                         use_intervals1_strand=False):
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
        Minimum distance (negative means target is upstream/left of query).
    maxdist : float, default 1e9
        Maximum distance (positive means target is downstream/right of query).
    na_if_notfound : bool, default False
        If True, include queries with no neighbors (with NA values).
    use_intervals1_strand : bool, default False
        If True, use intervals1 strand column for distance directionality
        instead of intervals2 strand. This is useful for TSS analysis where
        you want upstream/downstream distances relative to gene direction.
        When True:
        - + strand queries: negative distance = upstream, positive = downstream
        - - strand queries: negative distance = downstream, positive = upstream

    Returns
    -------
    DataFrame or None
        DataFrame with query and neighbor coordinates plus distance column.

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
    _checkroot()

    if intervals1 is None:
        raise ValueError("intervals1 cannot be None")
    if intervals2 is None:
        raise ValueError("intervals2 cannot be None")

    if maxneighbors < 1:
        raise ValueError("maxneighbors must be >= 1")

    if mindist > maxdist:
        raise ValueError("mindist must be <= maxdist")

    if len(intervals1) == 0:
        return None
    if len(intervals2) == 0 and not na_if_notfound:
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

    return _pymisha2df(result)


def gintervals_neighbors_upstream(intervals1, intervals2, maxneighbors=1,
                                   maxdist=1e9, na_if_notfound=False):
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
    # Upstream: mindist=-maxdist, maxdist=0, use_intervals1_strand=True
    return gintervals_neighbors(
        intervals1, intervals2,
        maxneighbors=maxneighbors,
        mindist=-maxdist, maxdist=0,
        na_if_notfound=na_if_notfound,
        use_intervals1_strand=True
    )


def gintervals_neighbors_downstream(intervals1, intervals2, maxneighbors=1,
                                     maxdist=1e9, na_if_notfound=False):
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
    # Downstream: mindist=0, maxdist=maxdist, use_intervals1_strand=True
    return gintervals_neighbors(
        intervals1, intervals2,
        maxneighbors=maxneighbors,
        mindist=0, maxdist=maxdist,
        na_if_notfound=na_if_notfound,
        use_intervals1_strand=True
    )


def gintervals_neighbors_directional(intervals1, intervals2,
                                      maxneighbors_upstream=1,
                                      maxneighbors_downstream=1,
                                      maxdist=1e9, na_if_notfound=False):
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


def gintervals_ls(pattern="", ignore_case=False):
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
    from . import _shared

    roots = []
    if _shared._UROOT:
        roots.append(_shared._UROOT)
    roots.append(_shared._GROOT)
    roots.extend(_shared._GDATASETS)

    interval_sets = set()
    for root in roots:
        tracks_dir = Path(root) / "tracks"
        if not tracks_dir.exists():
            continue
        for suffix in (".interv", ".interv2d"):
            for interv_file in tracks_dir.rglob(f"*{suffix}"):
                rel_path = interv_file.relative_to(tracks_dir)
                name = str(rel_path)[:-len(suffix)].replace("/", ".").replace("\\", ".")
                interval_sets.add(name)

    interval_sets = sorted(interval_sets)

    # Apply pattern filter
    if pattern:
        flags = re.IGNORECASE if ignore_case else 0
        interval_sets = [s for s in interval_sets if re.search(pattern, s, flags)]

    return interval_sets


def gintervals_exists(name):
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


def gintervals_dataset(intervals=None):
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

    roots = []
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


def gintervals_chrom_sizes(intervals):
    """
    Get chromosome sizes for intervals.

    Parameters
    ----------
    intervals : DataFrame
        Intervals with 'chrom' column.

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
    if intervals is None or len(intervals) == 0:
        return _pandas.DataFrame(columns=["chrom"])

    # Get unique chromosomes
    if "chrom" in intervals.columns:
        chroms = intervals["chrom"].unique()
    elif "chrom1" in intervals.columns:
        # 2D intervals
        chroms1 = intervals["chrom1"].unique()
        chroms2 = intervals["chrom2"].unique()
        chroms = _numpy.union1d(chroms1, chroms2)
    else:
        raise ValueError("intervals must have 'chrom' or 'chrom1'/'chrom2' columns")

    return _pandas.DataFrame({"chrom": sorted(chroms)})


def _read_serialized_dataframe(payload):
    with tempfile.NamedTemporaryFile(suffix=".rds") as tmp:
        tmp.write(payload)
        tmp.flush()
        return _decode_r_obj_to_bytes(tmp.name)


def _load_serialized_dataframe(path):
    return _decode_r_obj_to_bytes(path)


def _resolve_chrom_file(path, chrom):
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


def _resolve_pair_file(path, chrom1, chrom2):
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


def _chrom_id_map():
    chroms = gintervals_all()["chrom"].tolist()
    return {chrom: idx for idx, chrom in enumerate(chroms)}


def _chrom_id_lookup(chrom_map, chrom_name):
    if chrom_name in chrom_map:
        return chrom_map[chrom_name]
    alt = chrom_name[3:] if chrom_name.startswith("chr") else f"chr{chrom_name}"
    return chrom_map.get(alt)


def _indexed_entries_by_chrom(entries):
    return {chrom_id: (offset, length) for chrom_id, offset, length in entries}


def _indexed_entries_by_pair(entries):
    return {
        (chrom1_id, chrom2_id): (offset, length)
        for chrom1_id, chrom2_id, offset, length in entries
    }


def _read_indexed_entry(dat_path, offset, length):
    if length == 0:
        return None
    with open(dat_path, "rb") as fh:
        fh.seek(offset)
        payload = fh.read(length)
    return _read_serialized_dataframe(payload)


def _intervset_loadable(stats, max_size, label, chrom=None, chrom1=None, chrom2=None):
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


def _normalize_chrom_column(df, col):
    if col in df.columns:
        df[col] = _normalize_chroms(df[col].astype(str).tolist())
        df[col] = _pandas.Series(df[col])


def _normalize_interval_df(df):
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


def gintervals_load(intervals_set, chrom=None, chrom1=None, chrom2=None, progress=False):
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
        max_size = CONFIG.get("max_data_size")
        if "chrom" in stats.columns:
            if chrom1 is not None or chrom2 is not None:
                raise ValueError(f"{intervals_set} is a 1D big intervals set. chrom1/chrom2 are for 2D only.")
            if chrom is not None:
                chrom = _normalize_chroms([chrom])[0]
                stats = stats[stats["chrom"].astype(str) == chrom]
            ok, err = _intervset_loadable(stats, max_size, intervals_set, chrom=chrom)
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
        ok, err = _intervset_loadable(stats, max_size, intervals_set, chrom1=chrom1, chrom2=chrom2)
        if not ok:
            raise ValueError(err)
        if len(stats) == 0:
            return _normalize_interval_df(zeroline)

        paths = _intervset_index_paths(interv_path)
        indexed_fast = chrom1 is None and chrom2 is None and _intervset_is_indexed(interv_path)
        if indexed_fast:
            idx_entries = _load_index_entries_2d(paths["idx2d"])
            dfs = []
            with _progress_context(progress, total=len(idx_entries), desc="Loading intervals") as cb:
                for idx, (_chrom1_id, _chrom2_id, offset, length) in enumerate(idx_entries):
                    if length == 0:
                        continue
                    df = _read_indexed_entry(paths["dat2d"], offset, length)
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
        if chrom1 is not None and chrom2 is not None and paths["idx2d"].exists():
            idx_entries_map = _indexed_entries_by_pair(_load_index_entries_2d(paths["idx2d"]))
        dfs = []
        with _progress_context(progress, total=len(stats), desc="Loading intervals") as cb:
            for idx, row in enumerate(stats.itertuples(index=False)):
                chrom1_name = row.chrom1
                chrom2_name = row.chrom2
                pair_file = _resolve_pair_file(interv_path, chrom1_name, chrom2_name)
                if pair_file and pair_file.exists():
                    dfs.append(_load_serialized_dataframe(pair_file))
                elif idx_entries_map is not None:
                    chrom_map = _chrom_id_map()
                    chrom1_id = _chrom_id_lookup(chrom_map, chrom1_name)
                    chrom2_id = _chrom_id_lookup(chrom_map, chrom2_name)
                    if chrom1_id is not None and chrom2_id is not None:
                        entry = idx_entries_map.get((chrom1_id, chrom2_id))
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


def gintervals_save(intervals, intervals_set):
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

    # Save using pyreadr
    try:
        import pyreadr
        pyreadr.write_rds(str(interv_path), df)
    except ImportError:
        raise ImportError(
            "pyreadr is required to save interval sets. "
            "Install with: pip install pyreadr"
        ) from None


def gintervals_update(intervals_set, intervals, chrom=None):
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


def gintervals_mapply(func, *exprs, intervals=None, iterator=None,
                      intervals_set_out=None, colnames="value"):
    """
    Apply a function to track expression values for each interval.

    Evaluates track expressions for each interval and passes the resulting
    value arrays to *func*. The return value of *func* becomes a new column
    in the output.

    Parameters
    ----------
    func : callable
        Function to apply. Receives one numpy array per track expression.
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
    from .extract import gextract

    _checkroot()

    if intervals is None:
        raise ValueError("intervals parameter is required")

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

    results = []
    coord_rows = []

    for _idx, irow in work_intervals.iterrows():
        single = _pandas.DataFrame([{
            "chrom": irow["chrom"],
            "start": irow["start"],
            "end": irow["end"],
        }])

        # Check strand reversal
        reverse = False
        if "strand" in irow.index and irow.get("strand") == -1:
            reverse = True

        # Extract each expression for this single interval (at track resolution)
        arrays = []
        for expr in expr_list:
            ext = gextract(expr, intervals=single)
            if ext is not None and len(ext) > 0:
                val_cols = [c for c in ext.columns if c not in ("chrom", "start", "end", "intervalID")]
                arr = ext[val_cols[0]].to_numpy(dtype=float) if val_cols else np.array([])
            else:
                arr = np.array([])
            if reverse:
                arr = arr[::-1].copy()
            arrays.append(arr)

        val = func(*arrays)
        results.append(val)
        coord_rows.append({"chrom": irow["chrom"], "start": irow["start"], "end": irow["end"]})

    # Build result DataFrame
    result_df = _pandas.DataFrame(coord_rows)
    result_df[colnames] = results

    if intervals_set_out is not None:
        if gintervals_exists(intervals_set_out):
            gintervals_rm(intervals_set_out, force=True)
        gintervals_save(result_df, intervals_set_out)
        return None

    return result_df


def _copy_file_contents(src_path, dest_fh, buffer_size=1024 * 1024):
    total = 0
    with open(src_path, "rb") as src:
        while True:
            chunk = src.read(buffer_size)
            if not chunk:
                break
            dest_fh.write(chunk)
            total += len(chunk)
    return total


def _write_index_header_1d(fp, num_entries, checksum):
    magic = b"MISHAI1D"
    version = 1
    flags = 0x01
    reserved = 0
    fp.write(struct.pack("<8sIIQQI", magic, version, num_entries, flags, checksum, reserved))


def _write_index_header_2d(fp, num_entries, checksum):
    magic = b"MISHAI2D"
    version = 1
    flags = 0x01
    reserved = 0
    fp.write(struct.pack("<8sIIQQQ", magic, version, num_entries, flags, checksum, reserved))


def gintervals_convert_to_indexed(set_name, remove_old=False, force=False):
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


def gintervals_2d_convert_to_indexed(set_name, remove_old=False, force=False):
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


def gintervals_is_indexed(intervals_set):
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
    intervals1,
    expansion1,
    intervals2=None,
    expansion2=None,
    min_band_idx=None,
    max_band_idx=None,
):
    """
    Create a 2D cartesian-grid iterator as 2D intervals.

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


def giterator_intervals(expr=None, intervals=None, iterator=None,
                        interval_relative=False):
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
    interval_relative : bool, default False
        When ``True``, bins are aligned to each input interval's start
        rather than to chromosome position 0.  Requires a numeric
        *iterator*.

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
    if expr is None and iterator is None:
        raise ValueError(
            "At least one of 'expr' or 'iterator' must be provided."
        )
    _checkroot()

    if intervals is None:
        intervals = gintervals_all()

    if len(intervals) == 0:
        return None

    # Determine iterator policy
    itr = iterator
    if itr is None and expr is not None:
        # Try to resolve track bin size from expression (track name)
        from .tracks import gtrack_exists, gtrack_info
        if isinstance(expr, str) and gtrack_exists(expr):
            info = gtrack_info(expr)
            bin_size = info.get("bin_size") or info.get("bin.size")
            if bin_size is not None:
                itr = int(float(bin_size))
        if itr is None:
            raise ValueError(
                "Could not determine iterator from expression. "
                "Pass an explicit numeric iterator."
            )

    if interval_relative:
        if isinstance(itr, bool) or not isinstance(itr, (int, float)):
            raise ValueError(
                "interval_relative=True requires a numeric iterator."
            )
        cfg = dict(CONFIG)
        cfg["interval_relative"] = True
    else:
        cfg = CONFIG

    result = _pymisha.pm_iterate(_df2pymisha(intervals), itr, cfg)
    return _pymisha2df(result)


def gintervals_rbind(*intervals, intervals_set_out=None):
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

    result = _pandas.concat(loaded, ignore_index=True, sort=False, copy=False)
    if intervals_set_out is not None:
        gintervals_save(result, intervals_set_out)
        return None
    return result


def gintervals_mark_overlaps(intervals, group_col="overlap_group",
                              unify_touching_intervals=True):
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
    if intervals is None or len(intervals) == 0:
        raise ValueError("intervals cannot be None or empty")

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


def gintervals_annotate(intervals, annotation_intervals,
                         annotation_columns=None, column_names=None,
                         dist_column="dist", max_dist=float("inf"),
                         na_value=_numpy.nan, maxneighbors=1,
                         overwrite=False, keep_order=True, **kwargs):
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

    See Also
    --------
    gintervals_neighbors : Find nearest neighbors between interval sets.
    gintervals_mark_overlaps : Mark groups of overlapping intervals.
    """
    if intervals is None or annotation_intervals is None:
        raise ValueError("intervals and annotation_intervals must not be None")

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

    # Map annotation columns from neighbor result to output
    # The neighbor result has columns from both intervals1 and intervals2
    # Annotation columns from intervals2 may appear with "1" suffix if name conflicts
    result = nbrs.copy()

    # Build the annotation column mapping: src_col in result -> output name
    ann_col_map = {}  # output_name -> actual column in result
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
            output[col] = result[col].values

    # Add annotation columns with proper names
    for out_name, actual_col in ann_col_map.items():
        if actual_col is not None:
            output[out_name] = result[actual_col].values
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


def gintervals_normalize(intervals, size):
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
    chroms = intervals["chrom"].values

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
            result[col] = intervals[col].values

    return result


def gintervals_random(size, n, dist_from_edge=3_000_000,
                      chromosomes=None, mask=None, **kwargs):
    """
    Generate random genomic intervals.

    Intervals are sampled uniformly from the genome (after excluding
    chromosome edges and optional filter regions).  Each interval is
    exactly *size* basepairs.

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
        ``end``).
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
    >>> import numpy as np; np.random.seed(42)
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

    # Validate inputs
    if not isinstance(size, (int, _numpy.integer)) or size <= 0:
        raise ValueError("size must be a positive integer")
    if not isinstance(n, (int, _numpy.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if dist_from_edge < 0:
        raise ValueError("dist_from_edge must be non-negative")

    size = int(size)
    n = int(n)
    dist_from_edge = float(dist_from_edge)

    # Get genome intervals
    all_genome = gintervals_all()

    # Filter by chromosomes
    if chromosomes is not None:
        if not isinstance(chromosomes, (list, tuple)):
            raise ValueError("chromosomes must be a list of strings")
        chromosomes = list(_normalize_chroms(chromosomes))
        all_genome = all_genome[all_genome["chrom"].isin(chromosomes)]
        if len(all_genome) == 0:
            raise ValueError(
                f"No chromosomes named {', '.join(chromosomes)} found in the genome"
            )

    # Validate and canonicalize filter intervals
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
            # Canonicalize to merge overlaps
            mask = gintervals_canonic(mask)
        if mask is None or len(mask) == 0:
            mask = None

    # Build valid segments per chromosome
    # Each segment is a (chrom, seg_start, seg_end) where intervals can be placed
    # An interval of 'size' starting at position p must satisfy:
    #   p >= dist_from_edge  AND  p + size <= chrom_size - dist_from_edge
    #   AND  [p, p+size) does not overlap any filter region
    segments = []  # list of (chrom_name, seg_start, seg_end) where seg_end is exclusive

    for row in all_genome.itertuples(index=False):
        chrom = row.chrom
        chrom_size = int(row.end)
        lo = int(dist_from_edge)
        hi = chrom_size - int(dist_from_edge) - size

        if hi < lo:
            continue  # chromosome too short

        if mask is None:
            segments.append((chrom, lo, hi + 1))  # +1 because hi is inclusive start position
        else:
            # Subtract filter regions from [lo, hi]
            chrom_filter = mask[mask["chrom"] == chrom]
            if len(chrom_filter) == 0:
                segments.append((chrom, lo, hi + 1))
                continue

            # For each filter region [fs, fe), an interval starting at p
            # with size 'size' overlaps if p < fe and p + size > fs,
            # i.e. p in [fs - size + 1, fe - 1], so we exclude
            # start positions in [max(lo, fs - size + 1), min(hi, fe - 1)]
            cur_lo = lo
            for frow in chrom_filter.itertuples(index=False):
                fs = int(frow.start)
                fe = int(frow.end)
                # Excluded start positions: [fs - size + 1, fe - 1]
                excl_lo = max(cur_lo, fs - size + 1)
                excl_hi = min(hi, fe - 1)
                if excl_lo > cur_lo:
                    # Valid segment before this exclusion
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

    # Compute segment lengths and cumulative weights
    seg_chroms = [s[0] for s in segments]
    seg_starts = _numpy.array([s[1] for s in segments], dtype=_numpy.int64)
    seg_lengths = _numpy.array([s[2] - s[1] for s in segments], dtype=_numpy.int64)

    total_length = seg_lengths.sum()
    if total_length == 0:
        raise ValueError("No valid genomic positions for random intervals")

    cum_lengths = _numpy.cumsum(seg_lengths)

    # Sample n random positions
    rand_positions = _numpy.random.randint(0, total_length, size=n)

    # Map random positions to segments
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


def gintervals_rm(intervals_set, force=False):
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
    import shutil

    from . import _shared

    groot = _shared._GROOT
    path_part = intervals_set.replace(".", "/")
    interv_path = Path(groot) / "tracks" / f"{path_part}.interv"

    if not interv_path.exists():
        if force:
            return
        raise ValueError(f"Intervals set '{intervals_set}' does not exist")

    # Remove the file (or directory for big interval sets)
    if interv_path.is_dir():
        shutil.rmtree(interv_path)
    else:
        interv_path.unlink()


def _open_genes_file(path_or_url):
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


def _parse_annots_file(annots_file, num_annots):
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


def _parse_genes_file(genes_file, id2annots, known_chroms):
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


def _unify_intervals(records, annots_names):
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
    merged_annots = [[] for _ in range(num_annots)]  # list of lists

    cur_chrom, cur_start, cur_end, cur_strand, cur_annot = records[0]
    # Annotation accumulator: sets of unique values per column
    annot_sets = [set() for _ in range(num_annots)]
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
    for j in range(num_annots):
        df[annots_names[j]] = merged_annots[j]

    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["strand"] = df["strand"].astype(float)

    return df


def gintervals_import_genes(genes_file, annots_file=None, annots_names=None):
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
    id2annots = {}
    annots_tmpdir = None
    if annots_file is not None:
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

