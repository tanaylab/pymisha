"""Import and export of figwig files (``.fw``), the multi-track format of the figwig package
(https://github.com/jmschrei/figwig). figwig is an optional dependency:
``pip install "pymisha[figwig]"``."""

from __future__ import annotations

import json
import math
import numbers
import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _shared
from ._db_trash import _gdb_staging_name, _gdb_trash, _gdb_trash_sweep_old
from ._log import PymishaWarning, get_logger, user_stacklevel
from ._shared import _checkroot, _pm_dbreload

_logger = get_logger(__name__)

_FIGWIG_HINT = (
    'Reading and writing figwig files needs the figwig package: pip install "pymisha[figwig]" '
    "(or pip install figwig)"
)

# Keys of a figwig track record that are not attributes of the track
_FIGWIG_RESERVED = ("name", "dtype", "group")
# Attributes an import does not take from the file: those it sets itself, and those that describe how
# a misha track was stored (as gtrack_import_parquet)
_UNRESTORED_ATTRS = ("created.by", "created.date", "created.user", "type", "binsize")
# Values decoded at a time: tracks x bases of a chunk, as float64
_CHUNK_VALUES = 1 << 23
# Tracks written at a time (each holds an open file per chromosome)
_MAX_OPEN_TRACKS = 128
# Bases of a chromosome extracted at a time by gtrack_export_figwig
_EXPORT_CHUNK = 1 << 22

# The largest magnitude of the values of a figwig integer dtype
_INT_DTYPE_MAX = {"uint8": 255, "uint16": 65535, "uint32": 2**32 - 1, "int8": 128, "int16": 32768, "int32": 2**31}

_SPARSE_RECORD = np.dtype([("start", "<i8"), ("end", "<i8"), ("value", "<f4")])  # packed: 20 bytes
_SPARSE_SIGNATURE = np.int32(-1).tobytes()


def _figwig() -> Any:
    try:
        import figwig
    except ImportError as exc:
        raise ImportError(_FIGWIG_HINT) from exc
    return figwig


def _attr_text(value: Any) -> str | None:
    """A figwig attribute value as the text of a misha attribute: a string as it is, null as None
    (no attribute), anything else (number, boolean, list, object) as its JSON text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


_JSON_NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z")


def _number_text(v: float) -> str:
    """The canonical text of a number: a whole number below 1e15 as an integer, anything else
    as "%.17g" (C printf), so that 2, 2.0 and "2.0" compare equal. R misha's .figwig_number_text."""
    if math.isfinite(v) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.17g}"


def _match_text(value: Any) -> str | None:
    """The text a value of *select*, or a track's attribute value, is compared with: a number,
    or a string that is a JSON number, as _number_text; a boolean as "true" / "false"; any other
    string as it is; anything else as its JSON text; null as None (matches nothing)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral) and abs(int(value)) < 10**15:
        return str(int(value))
    if isinstance(value, numbers.Real):
        return _number_text(float(value))
    if isinstance(value, str):
        return _number_text(float(value)) if _JSON_NUMBER.match(value) else value
    return json.dumps(value)


def _sanitize_name(name: str, has_prefix: bool) -> str:
    """The misha track name a figwig track name becomes: a trailing "+" or "-" (a strand) is
    spelled "plus" or "minus", each run of characters other than ASCII letters, digits and "_"
    (dots included) becomes one "_", and a name that does not start with a letter gets an "X" in
    front, unless a prefix comes before it."""
    if name.endswith("+"):
        name = name[:-1] + "plus"
    elif name.endswith("-"):
        name = name[:-1] + "minus"
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    if not has_prefix and not re.match(r"[A-Za-z]", name):
        name = "X" + name
    return name


def _select_tracks(records: list[dict], tracks: Any, select: dict | None, file: str) -> list[int]:
    """The indices of the figwig tracks to import: those named by *tracks* (names, or 0-based
    indices), in that order, or all, in the file's order; then those whose attributes match all
    of *select*."""
    names = [str(r["name"]) for r in records]
    if tracks is None:
        idx = list(range(len(records)))
    else:
        if isinstance(tracks, (str, numbers.Integral)):
            tracks = [tracks]
        idx = []
        missing = []
        for t in tracks:
            if isinstance(t, (bool, np.bool_)):
                raise ValueError("tracks must be figwig track names or 0-based indices")
            if isinstance(t, numbers.Integral):
                if not 0 <= int(t) < len(records):
                    raise ValueError(f"Track index {int(t)} is out of range: {file} holds {len(records)} track(s)")
                idx.append(int(t))
            elif isinstance(t, str):
                if t in names:
                    idx.append(names.index(t))
                else:
                    missing.append(t)
            else:
                raise ValueError("tracks must be figwig track names or 0-based indices")
        if missing:
            raise ValueError(
                f"{file} has no track(s) named {', '.join(repr(m) for m in missing)}; its tracks: "
                f"{', '.join(names[:20])}{' ...' if len(names) > 20 else ''}"
            )
        if len(set(idx)) != len(idx):
            raise ValueError("tracks names a figwig track more than once")
        if not idx:
            raise ValueError("No figwig tracks to import")

    if select:
        if not isinstance(select, dict):
            raise ValueError("select must be a dict of attribute name -> value (or list of values)")
        keys = {k for r in records for k in r}
        for key, want in select.items():
            if key not in keys:
                raise ValueError(
                    f"No track of {file} has the attribute {key!r}; its attributes: {', '.join(sorted(keys))}"
                )
            wanted = (
                {_match_text(w) for w in want}
                if isinstance(want, (list, tuple, set, np.ndarray, pd.Series))
                else {_match_text(want)}
            )
            wanted.discard(None)
            idx = [i for i in idx if _match_text(records[i].get(key)) in wanted]
        if not idx:
            raise ValueError(f"No track of {file} matches select = {select!r}")
    return idx


def _base_names(records: list[dict], idx: list[int], names: Any, has_prefix: bool) -> list[str]:
    """The names (before the prefix) of the tracks made from records[idx]: *names* where it gives
    one (a list, one per track, or a dict from figwig name to name), else the sanitized figwig
    name."""
    fw_names = [str(records[i]["name"]) for i in idx]
    if names is None:
        return [_sanitize_name(n, has_prefix) for n in fw_names]
    if isinstance(names, dict):
        unknown = [k for k in names if k not in fw_names]
        if unknown:
            raise ValueError(f"names maps figwig track(s) that are not imported: {', '.join(map(repr, unknown))}")
        return [str(names[n]) if n in names else _sanitize_name(n, has_prefix) for n in fw_names]
    if isinstance(names, str):
        names = [names]
    names = list(names)
    if len(names) != len(idx):
        raise ValueError(f"names has {len(names)} element(s) but {len(idx)} track(s) are imported")
    if not all(isinstance(n, str) for n in names):
        raise ValueError("names must be strings")
    return [str(n) for n in names]


def _restored_attrs(record: dict, description: str, attrs: dict | None, file: str) -> dict[str, str]:
    """The attributes a track takes from its figwig record: all but name, dtype and group, those
    the import sets (created.by, created.date, created.user), type and binsize, description unless
    the *description* given is empty, and the names of *attrs*. Values are text (see _attr_text);
    a null or empty one is left out, and one holding a NUL byte is left out with a warning."""
    skip = set(_FIGWIG_RESERVED) | set(_UNRESTORED_ATTRS) | set(attrs or {})
    if str(description) != "":
        skip.add("description")
    res: dict[str, str] = {}
    for key, value in record.items():
        if not key or key in skip:
            continue
        text = _attr_text(value)
        if text is None or text == "":
            continue
        if "\x00" in key or "\x00" in text:
            warnings.warn(
                f"Attribute {key!r} of figwig track {record['name']!r} of {file} was left out: it holds a "
                "NUL byte, which an attribute cannot",
                stacklevel=user_stacklevel(),
            )
            continue
        res[key] = text
    return res


def _runs(x: np.ndarray, breaks: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The starts and ends of the maximal runs of equal values of *x* (NaN equals nothing, so each
    NaN is a run of its own), also cut every *breaks* elements when *breaks* > 0."""
    n = len(x)
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(x[1:], x[:-1], out=change[1:])
    if breaks:
        change[::breaks] = True
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    return starts, ends


class _DenseWriter:
    """Writes one chromosome of a dense track from per-base values given a chunk at a time, chunks
    starting at multiples of the bin size. A bin is misha's import mean (gtrack_import's): the
    sum, over the maximal runs of equal values cut at bin boundaries, of value x run length, divided
    by the bases with a value; a NaN base has none, and with a *defval* that is not NaN every base
    without a value counts as *defval*. A bin with no value is NaN."""

    def __init__(self, path: Path, binsize: int, chrom_size: int, defval: float, exact: bool = False):
        self.binsize = binsize
        self.chrom_size = chrom_size
        self.defval = defval
        # integers whose sums over a bin are exact in float64: summed base by base, the same sums
        self.exact = exact
        self.num_bins = -(-chrom_size // binsize)
        self.next_bin = 0
        self.fh = open(path, "wb")  # noqa: SIM115 - closed by close()
        self.fh.write(np.uint32(binsize).tobytes())

    def _write_bins(self, first: int, sums: np.ndarray, covered: np.ndarray) -> None:
        b = self.binsize
        n = len(sums)
        lens = np.minimum(b, self.chrom_size - (first + np.arange(n, dtype=np.int64)) * b).astype(np.float64)
        if not math.isnan(self.defval):
            part = covered < lens
            sums[part] = sums[part] + self.defval * (lens[part] - covered[part])
            covered[part] = lens[part]
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(covered > 0, sums / np.where(covered > 0, covered, 1), np.nan).astype(np.float32)
        vals.tofile(self.fh)
        self.next_bin = first + n

    def add(self, start: int, x: np.ndarray) -> None:
        b = self.binsize
        first = start // b
        if first > self.next_bin:
            self._fill(first)
        if self.exact:
            edges = np.arange(0, len(x), b)
            sums = np.add.reduceat(x, edges).astype(np.float64)
            covered = np.minimum(b, len(x) - edges).astype(np.float64)
            self._write_bins(first, sums, covered)
            return
        starts, ends = _runs(x, b)
        v = x[starts]
        ok = ~np.isnan(v)
        starts, ends, v = starts[ok], ends[ok], v[ok]
        lens = ends - starts
        nb = -(-len(x) // b)
        bins = starts // b
        # bincount with weights is float64; asarray says so to the type checker (no copy)
        sums = np.asarray(np.bincount(bins, weights=v * lens, minlength=nb), dtype=np.float64)
        covered = np.asarray(np.bincount(bins, weights=lens.astype(np.float64), minlength=nb), dtype=np.float64)
        self._write_bins(first, sums, covered)

    def _fill(self, upto: int) -> None:
        while self.next_bin < upto:
            n = min(upto - self.next_bin, 1 << 20)
            self._write_bins(self.next_bin, np.zeros(n), np.zeros(n))

    def close(self) -> None:
        try:
            self._fill(self.num_bins)
        finally:
            self.fh.close()


class _SparseWriter:
    """Writes one chromosome of a sparse track from per-base values given a chunk at a time, in
    order: one interval per maximal run of equal (float32) values that are neither 0 nor NaN."""

    def __init__(self, path: Path):
        self.fh = open(path, "wb")  # noqa: SIM115 - closed by close()
        self.fh.write(_SPARSE_SIGNATURE)
        self.pending: tuple[int, int, np.float32] | None = None

    def add(self, start: int, x: np.ndarray) -> None:
        x32 = x.astype(np.float32)
        starts, ends = _runs(x32)
        v = x32[starts]
        keep = (v != 0) & ~np.isnan(v)
        starts, ends, v = starts[keep] + start, ends[keep] + start, v[keep]
        if not len(starts):
            return
        if self.pending is not None:
            ps, pe, pv = self.pending
            if pe == starts[0] and pv == v[0]:
                starts[0] = ps
            else:
                self._write(np.array([ps]), np.array([pe]), np.array([pv], dtype=np.float32))
        self._write(starts[:-1], ends[:-1], v[:-1])
        self.pending = (int(starts[-1]), int(ends[-1]), v[-1])

    def _write(self, starts: np.ndarray, ends: np.ndarray, vals: np.ndarray) -> None:
        if len(starts):
            rec = np.empty(len(starts), dtype=_SPARSE_RECORD)
            rec["start"], rec["end"], rec["value"] = starts, ends, vals
            rec.tofile(self.fh)

    def close(self) -> None:
        try:
            if self.pending is not None:
                ps, pe, pv = self.pending
                self._write(np.array([ps]), np.array([pe]), np.array([pv], dtype=np.float32))
        finally:
            self.fh.close()


def _chrom_plan(reader: Any, file: str) -> tuple[list[tuple[str, int, str | None, int]], dict[str, Any], int]:
    """For each database chromosome, in database order: its name and size, the figwig chromosome
    it reads from (None if the file has none) and the bases with values (the smaller size).
    Also the _unknown_chroms of the file's other chromosomes and the number matched."""
    from .intervals import gintervals_all
    from .tracks import _resolve_import_chroms, _verify_chroms_matched

    fw_sizes = {str(c): int(s) for c, s in reader.chrom_sizes.items()}
    fw_names = list(fw_sizes)
    canon, unknown, n_matched = _resolve_import_chroms(fw_names)
    _verify_chroms_matched(unknown, n_matched, file)
    db_to_fw: dict[str, str] = {}
    for fw_name, db_name in zip(fw_names, canon, strict=True):
        if db_name is None:
            continue
        if db_name in db_to_fw:
            raise ValueError(
                f"Chromosomes {db_to_fw[db_name]} and {fw_name} of {file} both name chromosome {db_name} of the "
                "database"
            )
        db_to_fw[db_name] = fw_name
    allg = gintervals_all()
    plan: list[tuple[str, int, str | None, int]] = []
    mismatched = []
    for chrom, size in zip(allg["chrom"].astype(str), allg["end"].astype(np.int64), strict=True):
        source = db_to_fw.get(chrom)
        valid = 0
        if source is not None:
            valid = min(int(size), fw_sizes[source])
            if fw_sizes[source] != int(size):
                mismatched.append((chrom, fw_sizes[source], int(size)))
        plan.append((chrom, int(size), source, valid))
    if mismatched:
        c, fs, ds = mismatched[0]
        warnings.warn(
            f"{len(mismatched)} chromosome(s) of {file} differ in length from the database (the first: {c}, "
            f"{fs} bp in the file, {ds} bp in the database): bases past the database's end were dropped, and "
            "bases past the file's end have no value",
            stacklevel=user_stacklevel(),
        )
    return plan, unknown, n_matched


def _write_tracks(reader: Any, idx: list[int], stage_dirs: list[Path], plan: list, binsize: int,
                  defval: float) -> None:
    """Write the chromosome files of tracks idx of the figwig file into stage_dirs."""
    for b0 in range(0, len(idx), _MAX_OPEN_TRACKS):
        batch = idx[b0:b0 + _MAX_OPEN_TRACKS]
        dirs = stage_dirs[b0:b0 + _MAX_OPEN_TRACKS]
        step = max(1, binsize)
        width = max(step, (_CHUNK_VALUES // len(batch)) // step * step)
        exact = [binsize * _INT_DTYPE_MAX.get(str(reader.tracks[i]["dtype"]), math.inf) < 2**53 for i in batch]
        for chrom, size, fw_chrom, valid in plan:
            if binsize:
                writers: list[Any] = [_DenseWriter(d / chrom, binsize, size, defval, e)
                                      for d, e in zip(dirs, exact, strict=True)]
            else:
                writers = [_SparseWriter(d / chrom) for d in dirs]
            try:
                for start in range(0, valid, width):
                    w = min(width, valid - start)
                    vals = reader.read([fw_chrom], [start], w, tracks=list(batch), dtype="float64")
                    vals = np.asarray(vals).reshape(len(batch), w)
                    for i, writer in enumerate(writers):
                        writer.add(start, vals[i])
            finally:
                for writer in writers:
                    writer.close()


def gtrack_import_figwig(
    file: str,
    binsize: int | None = None,
    track_prefix: str | None = None,
    tracks: str | int | Sequence[str | int] | None = None,
    select: dict[str, Any] | None = None,
    names: Sequence[str] | dict[str, str] | None = None,
    defval: float = np.nan,
    description: str = "",
    attrs: dict[str, str] | None = None,
) -> list[str]:
    """
    Create tracks from the tracks of a figwig file.

    Creates one misha track per selected track of a figwig file (``.fw``, the
    multi-track format of the `figwig <https://github.com/jmschrei/figwig>`_
    package), and needs that package (``pip install "pymisha[figwig]"``).

    With *binsize* ``None`` or 0 each track is Sparse: one interval per
    maximal run of bases with the same value, leaving out the bases whose
    value is 0 or NaN. That suits tracks that are mostly zero, such as counts
    or cut sites; a dense float track makes one interval per base. With a
    positive *binsize* each track is Dense, its bins computed as
    :func:`gtrack_import` computes the bins of a bigWig: the mean of the
    values of the bin's bases, each run of equal values weighted by its
    length. A figwig file has no "no data": a base that was never written is
    0, as is one written as 0 (figwig's ``bw2fw`` writes a bigWig's uncovered
    bases as 0 unless given ``--missing nan``). NaN bases, bases past the end
    of the file's chromosome and chromosomes the file does not have have no
    value: they count as *defval* unless it is NaN, and a bin with no value
    is NaN (*defval*, when it is not NaN).

    Tracks are chosen with *tracks* (figwig track names, or 0-based indices),
    in the order given, or all of them in the file's order, and then with
    *select*, a dict of attribute name to value (or list of values, any of
    which matches): a track is imported when every attribute given matches.
    Numbers are compared by value, whether given as numbers or as text, so
    ``{"rep": 1}``, ``{"rep": "1"}`` and ``{"rep": "1.0"}`` all match a
    ``"rep": 1`` or ``"rep": 1.0`` record; other values are compared as text.
    ``name``, ``dtype`` and ``group`` can be selected on too.

    The track made from figwig track ``n`` is named ``{track_prefix}{n'}``,
    where ``n'`` is *names*' entry for it if given (a list, one per imported
    track, or a dict from figwig name to name) and otherwise ``n``
    sanitized: a trailing ``+`` or ``-`` (a strand) is spelled ``plus`` or
    ``minus``, each run of characters other than ASCII letters, digits and
    ``_`` becomes one ``_``, and without a prefix a name that does not start
    with a letter gets an ``X`` in front (``atac.+`` becomes ``atac_plus``,
    ``H3K27ac-rep1`` ``H3K27ac_rep1``). Dots in *track_prefix* are
    directories, which must exist. All the names are checked before anything
    is written, and existing tracks are not overwritten.

    Each figwig track's attributes become attributes of its track, except
    ``name``, ``dtype`` and ``group``; ``created.by``, ``created.date`` and
    ``created.user``, which the import sets; ``type`` and ``binsize``; and
    ``description``, unless the *description* given is empty (the default).
    A string attribute is kept as it is, a null one is left out, and a
    number, boolean, list or object becomes its JSON text. *attrs* win over
    them.

    Chromosome names are matched through the chromosome alias chain as in
    :func:`gtrack_import`, with the same reporting of names the database
    does not have; a chromosome whose length differs from the database's is
    warned about. The file is read a chunk at a time, never a whole
    chromosome of every track at once. The import is all or nothing: if it
    fails or is interrupted, none of the tracks is created.

    Parameters
    ----------
    file : str
        Path of the figwig file.
    binsize : int or None, optional
        Bin size of Dense tracks; None or 0 makes Sparse tracks.
    track_prefix : str or None, optional
        Prefix of the track names.
    tracks : str, int or list, optional
        Figwig track names or 0-based indices to import, in the order of the
        created tracks. None imports every track.
    select : dict or None, optional
        Attribute filter (see above).
    names : list or dict, optional
        Names of the created tracks (before the prefix), instead of the
        sanitized figwig names.
    defval : float, default numpy.nan
        Value of the bases with no value in Dense tracks (see above).
    description : str, default ""
        Description of every created track; when empty, a figwig
        ``description`` attribute is kept.
    attrs : dict or None, optional
        Attributes set on every created track, over those of the file.

    Returns
    -------
    list of str
        Names of the created tracks.

    Raises
    ------
    ImportError
        If the figwig package is not installed.
    ValueError
        If an argument is invalid, no track is selected, a track name is
        invalid, repeated or exists, or none of the file's chromosomes is in
        the database.

    See Also
    --------
    gtrack_export_figwig : Write tracks or track expressions to a figwig file.
    gtrack_import : Import a single track from a file.
    gtrack_import_parquet : Import the value columns of Parquet files.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_export_figwig(["dense_track", "dense_track * 2"], "/tmp/x.fw",
    ...                         names=["a", "b"])  # doctest: +SKIP
    >>> pm.gtrack_import_figwig("/tmp/x.fw", binsize=50, track_prefix="fw_")  # doctest: +SKIP
    ['fw_a', 'fw_b']
    """
    _checkroot()
    from .tracks import (
        _db_is_indexed,
        _ensure_track_absent,
        _fsync_dir,
        _fsync_tree,
        _index_created_track,
        _report_unknown_chroms,
        _target_root,
        _track_dir_for_create,
        _validate_track_name,
        _write_created_attrs_at_path,
    )

    if not isinstance(file, (str, os.PathLike)):
        raise ValueError("file must be a path")
    file = os.fspath(file)
    if binsize is None:
        binsize = 0
    if (
        isinstance(binsize, bool)
        or not isinstance(binsize, numbers.Real)
        or not math.isfinite(binsize)
        or binsize < 0
        or binsize != int(binsize)
        or binsize >= 2**32
    ):
        raise ValueError("binsize must be None, 0 or a positive integer below 2^32")
    binsize = int(binsize)
    if isinstance(defval, bool) or not isinstance(defval, numbers.Real):
        raise ValueError("defval must be a number")
    defval = float(defval)
    if track_prefix is None:
        track_prefix = ""
    if not isinstance(track_prefix, str):
        raise ValueError("track_prefix must be a string")
    if description is None or not isinstance(description, str):
        raise ValueError("description must be a string")
    if attrs is not None and not isinstance(attrs, dict):
        raise ValueError("attrs must be a dict of attribute name -> value")
    fw = _figwig()
    if not os.path.isfile(file):
        raise ValueError(f"File {file} does not exist")
    if not fw.is_figwig(file):
        raise ValueError(f"{file} is not a figwig file")
    reader = fw.FigWigReader(file)
    records = [dict(r) for r in reader.tracks]
    idx = _select_tracks(records, tracks, select, file)
    base = _base_names(records, idx, names, track_prefix != "")
    new_tracks = [track_prefix + b for b in base]

    bad = [t for t in new_tracks if not re.fullmatch(r"[A-Za-z][\w.]*", t, re.ASCII)]
    if bad:
        raise ValueError(
            f"Invalid track name(s): {', '.join(map(repr, bad))} (a letter, then letters, digits, _ or .); "
            "choose other names with 'names' or 'track_prefix'"
        )
    dups = sorted({t for t in new_tracks if new_tracks.count(t) > 1})
    if dups:
        clash = {t: [records[i]["name"] for i, n in zip(idx, new_tracks, strict=True) if n == t] for t in dups}
        raise ValueError(
            "Figwig tracks would make the same track name: "
            + "; ".join(f"{t} from {', '.join(map(repr, src))}" for t, src in clash.items())
            + ". Name them with 'names'."
        )
    for track in new_tracks:
        _validate_track_name(track)
        _ensure_track_absent(track)
        if not _track_dir_for_create(track).parent.is_dir():
            raise ValueError(f"Directory {os.path.dirname(track.replace('.', '/'))} does not exist")

    plan, unknown, n_matched = _chrom_plan(reader, file)
    restored = [_restored_attrs(records[i], description, attrs, file) for i in idx]
    created_by = [
        f'gtrack.import_figwig("{file}", {binsize}, "{track_prefix}", "{records[i]["name"]}", {defval:g})'
        for i in idx
    ]

    root = _target_root()
    finals = [_track_dir_for_create(t) for t in new_tracks]
    staging = [f.parent / _gdb_staging_name(f.name) for f in finals]
    created: list[tuple[str, Path]] = []
    try:
        for parent in dict.fromkeys(f.parent for f in finals):
            _gdb_trash_sweep_old(parent)
        with _shared._with_umask():
            for stage_dir in staging:
                stage_dir.mkdir()
            _write_tracks(reader, idx, staging, plan, binsize, defval)

        for track, stage_dir, final, cb, md in zip(new_tracks, staging, finals, created_by, restored, strict=True):
            _fsync_tree(stage_dir)
            if final.exists():
                raise ValueError(f"Refusing to overwrite existing {final}")
            os.rename(stage_dir, final)
            created.append((track, final))
            _fsync_dir(final.parent)
            _shared._invalidate_dir_cache(final)
            _write_created_attrs_at_path(final, description, cb, {**md, **(attrs or {})})
        _pm_dbreload(root, tracks=new_tracks)
        if _db_is_indexed(root):
            for track in new_tracks:
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
                _pm_dbreload(root, tracks=[track for track, _ in created])
            except Exception:
                _logger.warning("could not reload the database after removing %s", new_tracks, exc_info=True)
        raise

    _report_unknown_chroms(unknown, n_matched, file)
    return new_tracks


def _export_scopes(intervals: Any) -> list[pd.DataFrame]:
    """The scope of an export, one canonic DataFrame per chromosome, in database order."""
    from .intervals import (
        gintervals_all,
        gintervals_canonic,
        gintervals_chrom_sizes,
        gintervals_is_bigset,
        gintervals_load,
    )

    if intervals is None:
        allg = gintervals_all()
        return [allg.iloc[[i]].reset_index(drop=True) for i in range(len(allg))]
    if isinstance(intervals, str) and gintervals_is_bigset(intervals):
        parts = [gintervals_load(intervals, chrom=str(c)) for c in gintervals_chrom_sizes(intervals)["chrom"]]
    else:
        if isinstance(intervals, str):
            intervals = gintervals_load(intervals)
        parts = [intervals]
    order = {c: i for i, c in enumerate(gintervals_all()["chrom"].astype(str))}
    scopes = []
    for part in parts:
        if part is None or len(part) == 0:
            continue
        canonic = gintervals_canonic(part[["chrom", "start", "end"]], unify_touching_intervals=False)
        if canonic is None or len(canonic) == 0:
            continue
        for chrom, grp in canonic.groupby(canonic["chrom"].astype(str), sort=False):
            scopes.append((order.get(chrom, len(order)), grp.reset_index(drop=True)))
    scopes.sort(key=lambda s: s[0])
    return [s for _, s in scopes]


def _split_scope(scope: pd.DataFrame, chunk: int | None) -> list[pd.DataFrame]:
    """*scope* (one chromosome, sorted, not overlapping) cut at multiples of *chunk*, in pieces of
    at most one chunk each (whole when *chunk* is None)."""
    if chunk is None:
        return [scope]
    chrom = scope["chrom"].iloc[0]
    starts = scope["start"].to_numpy(dtype=np.int64)
    ends = scope["end"].to_numpy(dtype=np.int64)
    pieces = []
    k = int(starts.min()) // chunk
    last = int(ends.max())
    while k * chunk < last:
        lo, hi = k * chunk, (k + 1) * chunk
        s = np.maximum(starts, lo)
        e = np.minimum(ends, hi)
        keep = s < e
        if keep.any():
            pieces.append(pd.DataFrame({"chrom": chrom, "start": s[keep], "end": e[keep]}))
        k += 1
    return pieces


def gtrack_export_figwig(
    tracks: str | Sequence[str],
    file: str,
    intervals: pd.DataFrame | str | None = None,
    iterator: Any = None,
    names: Sequence[str] | None = None,
    genome: str | None = None,
    missing: float = np.nan,
    dtype: str | None = None,
) -> None:
    """
    Write tracks or track expressions as the tracks of a figwig file.

    Writes each of *tracks* (track names or track expressions) as one track
    of a figwig file (``.fw``, the multi-track format of the `figwig
    <https://github.com/jmschrei/figwig>`_ package), which needs that package
    (``pip install "pymisha[figwig]"``). The values are those
    :func:`gextract` gives over *intervals* (the whole genome by default)
    with *iterator*: each iterator interval's value is written to all of its
    bases, and bases outside every interval (and outside a sparse track's
    intervals) are 0, figwig's value of a base that is not written. Overlapping intervals are merged first. The file
    holds every chromosome of the database, at its length, and is read in
    the database's chromosome order.

    A bin whose value is NaN (no data) is written as *missing*: NaN by
    default, which figwig keeps (in a float32 track) and
    :func:`gtrack_import_figwig` reads back as no value; ``missing=0``
    writes what figwig's ``bw2fw`` writes for a bigWig's gaps. Each figwig
    track takes the smallest type that holds its values (``uint8`` for small
    counts, ``float32`` for fractions) unless *dtype* is given, such as
    ``"float64"`` to keep the full precision of an expression.

    The figwig track of a track name carries its misha attributes, but
    ``name``, ``dtype`` and ``group``, which are not attributes in figwig and
    are left out with a warning; that of an expression carries the
    attribute ``expression``. :func:`gtrack_import_figwig` turns them back
    into misha attributes.

    Parameters
    ----------
    tracks : str or list of str
        Track names or track expressions.
    file : str
        Path of the figwig file to write (replaced if it exists).
    intervals : DataFrame, str or None, optional
        Intervals to write, or the name of an interval set. None writes the
        whole genome.
    iterator : int, str or DataFrame, optional
        Iterator of :func:`gextract`, shared by all the expressions. None
        takes the one each expression implies: the bin size of its dense
        tracks, or the intervals of its sparse track, so that tracks of
        different kinds or bin sizes are each written at their own
        resolution (a base where a sparse track has no interval is 0).
    names : list of str, optional
        Names of the figwig tracks, distinct and not empty; *tracks* by
        default.
    genome : str or None, optional
        Name of the reference genome recorded in the file.
    missing : float, default numpy.nan
        Value written for NaN values.
    dtype : str or None, optional
        figwig dtype of every track (``"uint8"``, ``"uint16"``,
        ``"uint32"``, ``"int8"``, ``"int16"``, ``"int32"``, ``"float32"``,
        ``"float64"``); None lets figwig choose per track.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If the figwig package is not installed.
    ValueError
        If an argument is invalid.

    See Also
    --------
    gtrack_import_figwig : Create tracks from a figwig file.
    gtrack_export_bigwig : Export a track to a bigWig file.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_export_figwig(["dense_track", "sparse_track"], "/tmp/x.fw", iterator=50)  # doctest: +SKIP
    """
    _checkroot()
    from .extract import gextract
    from .gsynth import _implicit_binsize
    from .intervals import gintervals_all
    from .tracks import _load_track_attributes, _track_exists

    if isinstance(tracks, str):
        tracks = [tracks]
    tracks = list(tracks)
    if not tracks or not all(isinstance(t, str) and t for t in tracks):
        raise ValueError("tracks must be one or more track names or expressions")
    if not isinstance(file, (str, os.PathLike)):
        raise ValueError("file must be a path")
    file = os.fspath(file)
    out_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Cannot write to '{file}': directory does not exist")
    if names is None:
        names = list(tracks)
    elif isinstance(names, str):
        names = [names]
    names = list(names)
    if len(names) != len(tracks):
        raise ValueError(f"names has {len(names)} element(s) but there are {len(tracks)} track(s)")
    if not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
        raise ValueError("names must be distinct, non-empty strings")
    if isinstance(missing, bool) or not isinstance(missing, numbers.Real):
        raise ValueError("missing must be a number")
    missing = float(missing)
    if math.isinf(missing):
        raise ValueError("missing cannot be infinite: figwig holds finite values and NaN")

    fw = _figwig()
    records = []
    for expr, name in zip(tracks, names, strict=True):
        rec: dict[str, Any] = {"name": name}
        if dtype is not None:
            rec["dtype"] = dtype
        if _track_exists(expr):
            attrs = _load_track_attributes(expr)
            dropped = [k for k in attrs if k in _FIGWIG_RESERVED]
            if dropped:
                warnings.warn(
                    f"Attribute(s) {', '.join(dropped)} of track {expr} were not written: figwig reserves them",
                    stacklevel=user_stacklevel(),
                )
            rec.update({k: v for k, v in attrs.items() if k not in _FIGWIG_RESERVED})
        else:
            rec["expression"] = expr
        records.append(rec)

    # Each expression is extracted on its own when no iterator is given and the expressions do not
    # share a bin size (sparse tracks, or dense tracks of different bin sizes), so that each is
    # written at its own resolution, over its own intervals.
    separate = False
    if iterator is None:
        bs = _implicit_binsize(tracks)
        if bs is None and len(tracks) > 1:
            separate = True
            sizes = [_implicit_binsize([t]) for t in tracks]
            bs = math.lcm(*[s for s in sizes if s]) if any(sizes) else None
        chunk: int | None = bs * max(1, _EXPORT_CHUNK // bs) if bs else _EXPORT_CHUNK
    elif isinstance(iterator, numbers.Integral) and not isinstance(iterator, bool):
        it = int(iterator)
        if it <= 0:
            raise ValueError("iterator must be a positive integer")
        chunk = it * max(1, _EXPORT_CHUNK // it)
    else:
        chunk = None  # an iterator of intervals: each chromosome in one piece

    allg = gintervals_all()
    chrom_sizes = {str(c): int(e) for c, e in zip(allg["chrom"], allg["end"], strict=True)}
    kwargs: dict[str, Any] = {"progress": False}
    if iterator is not None:
        kwargs["iterator"] = iterator

    with fw.FigWigWriter(file, chrom_sizes, records, genome=genome) as writer:
        for scope in _export_scopes(intervals):
            chrom = str(scope["chrom"].iloc[0])
            for piece in _split_scope(scope, chunk):
                if separate:
                    got = _extract_separately(gextract, tracks, piece, kwargs)
                else:
                    got = _extract_jointly(gextract, tracks, piece, kwargs)
                if got is None:
                    continue
                starts, ends, vals = got
                if np.isinf(vals).any():
                    raise ValueError(
                        "A value is infinite, which a figwig file cannot hold: clip the expression first"
                    )
                if not math.isnan(missing):
                    vals = np.where(np.isnan(vals), missing, vals)
                writer.write(np.full(len(starts), chrom), starts, vals, ends=ends)


def _extract_jointly(gextract: Any, tracks: list[str], piece: pd.DataFrame,
                     kwargs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The iterator intervals of gextract of *tracks* over *piece*, sorted, and their values."""
    colnames = [f"v{i}" for i in range(len(tracks))]
    data = gextract(tracks, intervals=piece, colnames=colnames, **kwargs)
    if data is None or len(data) == 0:
        return None
    data = data.sort_values("start", kind="mergesort")
    return (data["start"].to_numpy(dtype=np.int64), data["end"].to_numpy(dtype=np.int64),
            data[colnames].to_numpy(dtype=np.float64))


def _extract_separately(gextract: Any, tracks: list[str], piece: pd.DataFrame,
                        kwargs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """gextract of each of *tracks* over *piece* with its own iterator, on the segments between the
    boundaries of all their intervals that at least one covers; a track has the value of its
    interval over a segment, or 0 (figwig's value of a base not written) where it has none."""
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for track in tracks:
        data = gextract(track, intervals=piece, colnames=["v"], **kwargs)
        if data is None or len(data) == 0:
            parts.append(None)
            continue
        data = data.sort_values("start", kind="mergesort")
        parts.append((data["start"].to_numpy(dtype=np.int64), data["end"].to_numpy(dtype=np.int64),
                      data["v"].to_numpy(dtype=np.float64)))
    present = [p for p in parts if p is not None]
    if not present:
        return None
    bounds = np.unique(np.concatenate([np.concatenate([p[0], p[1]]) for p in present]))
    seg_starts, seg_ends = bounds[:-1], bounds[1:]
    vals = np.zeros((len(seg_starts), len(tracks)))
    covered = np.zeros(len(seg_starts), dtype=bool)
    for i, p in enumerate(parts):
        if p is None:
            continue
        starts, ends, values = p
        k = np.searchsorted(starts, seg_starts, side="right") - 1
        inside = (k >= 0) & (ends[np.maximum(k, 0)] >= seg_ends)
        vals[inside, i] = values[k[inside]]
        covered |= inside
    return seg_starts[covered], seg_ends[covered], vals[covered]
