"""Reader for misha ``array`` tracks (1D multi-column dense / sparse).

Read-only Python implementation. Binary format mirrors R's
``GenomeTrackArrays`` (``src/GenomeTrackArrays.{cpp,h}``):

Per-chromosome file (little-endian throughout)
-----------------------------------------------
- ``int32`` format signature = ``-8`` (``FORMAT_SIGNATURES[ARRAYS]``)
- ``int64`` ``intervals_pos`` - file offset to the interval table
- value blocks at varying offsets (one per interval):
    - ``uint32`` ``num_vals``
    - ``num_vals`` records of ``(float val, uint32 idx)``  (8 bytes each)
- at ``intervals_pos``:
    - ``uint64`` ``num_intervals``
    - ``num_intervals`` records of ``(int64 start, int64 end, int64 vals_pos)``
      (24 bytes each on 64-bit Linux; ``long`` is 8 bytes there)

Idx is 0-based and indexes into the column-names list. The ``.colnames``
file in the track directory is an R-serialized named integer vector
(value = 1..N, names = column names). We only need its names.
"""

from __future__ import annotations

import io
import struct
from collections import defaultdict
from pathlib import Path

import numpy as _numpy
import pandas as _pd

from ._r_serialize import read as _r_read

_ARRAYS_SIGNATURE = -8
_RECORD_SIZE = 24  # 2*int64 + int64

# Indexed single-file storage (track.idx / track.dat). See src/TrackIndex.h:
# 36-byte header (magic, version, track_type, num_contigs, flags, checksum),
# then num_contigs * 24-byte entries (chrom_id, offset, length, reserved). Each
# contig's payload in track.dat is the verbatim per-chrom block (offsets inside
# it are block-relative), so it parses exactly like a standalone per-chrom file.
_INDEX_MAGIC = b"MISHATDX"
_INDEX_HEADER_FMT = "<8sIIIQQ"
_INDEX_HEADER_SIZE = 36
_INDEX_ENTRY_FMT = "<IQQI"
_INDEX_ENTRY_SIZE = 24


def read_colnames(track_dir: Path | str) -> list[str]:
    """Return the column names of an array track.

    Reads ``<track_dir>/.colnames`` which is an R-serialized named
    integer vector (``c(col0=1, col1=2, ...)``).
    """
    cn_path = Path(track_dir) / ".colnames"
    if not cn_path.exists():
        raise FileNotFoundError(
            f"missing .colnames file at {cn_path} - track is not an array track"
        )
    obj = _r_read(cn_path)
    names = getattr(obj, "names", None)
    if isinstance(names, list):
        return list(names)
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return list(obj)
    raise ValueError(
        f"{cn_path} does not contain a named vector; got {type(obj).__name__}"
    )


def write_colnames(track_dir: Path | str, names: list[str]) -> None:
    """Write a column-names file in a format both R and pymisha can read.

    R's ``gtrack.array.set_colnames`` writes
    ``setNames(seq_along(names), names)`` via ``serialize(..., ascii=FALSE)``.
    We replicate that bytewise.
    """
    if not all(isinstance(n, str) and n for n in names):
        raise ValueError("column names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("column names must be unique")

    cn_path = Path(track_dir) / ".colnames"
    payload = _serialize_named_int_vector(names)
    with open(cn_path, "wb") as fh:
        fh.write(payload)


def _serialize_named_int_vector(names: list[str]) -> bytes:
    """Produce the exact bytes R's `serialize(setNames(1:N, names), con, ascii=FALSE)`
    would write. Header + INTSXP (with attributes) + names STRSXP + NULL.
    """
    import struct as _struct

    out = bytearray()
    # XDR header: "X\n", version=2, writer=R 4.0.0, reader=R 3.5.0.
    # Versions don't affect downstream readers.
    out += b"X\n"
    out += _struct.pack(">iii", 2, (4 << 16) | (0 << 8) | 0, (3 << 16) | (5 << 8) | 0)

    n = len(names)

    # INTSXP with attributes: low byte = 13, bit 9 (has_attr) = 1 -> 0x0000020d
    out += _struct.pack(">I", 0x0000020d)
    out += _struct.pack(">i", n)
    for i in range(1, n + 1):
        out += _struct.pack(">i", i)

    # Attributes pairlist: a single LISTSXP cell with tag = symbol "names"
    # and value = STRSXP. Then NIL terminator.
    # LISTSXP with has_tag = 1 -> 0x00000402
    out += _struct.pack(">I", 0x00000402)
    # Tag = SYMSXP wrapping CHARSXP "names"
    out += _struct.pack(">I", 0x00000001)  # SYMSXP
    # CHARSXP, level UTF-8 (bit 18). 0x00040009 = ASCII default; works with R 4.x.
    out += _struct.pack(">I", 0x00040009)
    out += _struct.pack(">i", 5)
    out += b"names"
    # Value = STRSXP of column names
    out += _struct.pack(">I", 0x00000010)
    out += _struct.pack(">i", n)
    for name in names:
        out += _struct.pack(">I", 0x00040009)  # CHARSXP UTF-8
        b = name.encode("utf-8")
        out += _struct.pack(">i", len(b))
        out += b
    # NIL terminator for the pairlist
    out += _struct.pack(">I", _NIL_FLAGS)
    return bytes(out)


_NIL_FLAGS = 0x000000FE  # NILVALUE_SXP = 254


def _resolve_chrom_file(track_dir: Path, chrom: str) -> Path | None:
    """Return the per-chromosome data file for *chrom*, or ``None``.

    Tries ``<chrom>`` and ``chr<chrom>`` to match the misha convention.
    """
    candidate = track_dir / chrom
    if candidate.exists():
        return candidate
    candidate = track_dir / f"chr{chrom}"
    if candidate.exists():
        return candidate
    if chrom.startswith("chr"):
        candidate = track_dir / chrom[3:]
        if candidate.exists():
            return candidate
    return None


def _read_intervals_from_fh(fh) -> tuple[
    _numpy.ndarray, _numpy.ndarray, _numpy.ndarray
]:
    """Parse the interval table from an open array-track block (file or BytesIO).

    Offsets are read relative to the start of ``fh`` (block-relative), so this
    works for both a standalone per-chrom file and an indexed ``track.dat`` slice.
    Returns ``(starts, ends, vals_pos)`` as int64 numpy arrays.
    """
    fh.seek(0)
    signature = struct.unpack("<i", fh.read(4))[0]
    if signature != _ARRAYS_SIGNATURE:
        raise ValueError(
            f"expected array-track signature -8, got {signature}"
        )
    intervals_pos = struct.unpack("<q", fh.read(8))[0]
    fh.seek(intervals_pos)
    num_intervals = struct.unpack("<Q", fh.read(8))[0]
    raw = fh.read(num_intervals * _RECORD_SIZE)
    arr = _numpy.frombuffer(raw, dtype=_numpy.int64).reshape(num_intervals, 3)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()


def read_chrom_intervals(filepath: Path) -> tuple[
    _numpy.ndarray, _numpy.ndarray, _numpy.ndarray
]:
    """Parse the interval table of an array-track per-chromosome file."""
    with open(filepath, "rb") as fh:
        return _read_intervals_from_fh(fh)


def _read_track_index(track_dir: Path) -> dict[int, tuple[int, int]] | None:
    """Parse ``track.idx``; return ``{chrom_id: (offset, length)}`` or ``None``.

    ``None`` means the track is in legacy per-chromosome format (no ``track.idx``).
    """
    idx_path = Path(track_dir) / "track.idx"
    if not idx_path.exists():
        return None
    with open(idx_path, "rb") as fh:
        header = fh.read(_INDEX_HEADER_SIZE)
        if len(header) != _INDEX_HEADER_SIZE or header[:8] != _INDEX_MAGIC:
            return None
        num_contigs = struct.unpack(_INDEX_HEADER_FMT, header)[3]
        entries: dict[int, tuple[int, int]] = {}
        for _ in range(num_contigs):
            rec = fh.read(_INDEX_ENTRY_SIZE)
            if len(rec) != _INDEX_ENTRY_SIZE:
                break
            chrom_id, offset, length, _reserved = struct.unpack(_INDEX_ENTRY_FMT, rec)
            entries[chrom_id] = (offset, length)
    return entries


def _groot_from_track_dir(track_dir: Path) -> Path | None:
    """Walk up from a ``.track`` dir to the DB root (parent of ``tracks/``)."""
    track_dir = Path(track_dir)
    for parent in track_dir.parents:
        if parent.name == "tracks":
            return parent.parent
    return None


def _add_chrom_aliases(mapping: dict[str, int], name: str, chrom_id: int) -> None:
    """Register ``name`` and its chr-prefixed/stripped aliases -> ``chrom_id``."""
    stripped = name[3:] if name.startswith("chr") else name
    mapping[name] = chrom_id
    mapping.setdefault(stripped, chrom_id)
    mapping.setdefault(f"chr{stripped}", chrom_id)


def chrom_id_map_from_order(chrom_order: list[str]) -> dict[str, int]:
    """Map chromosome name -> chrom_id from the genome's chrom-key order.

    The indexed ``track.idx`` keys contigs by the genome chrom-key id, which is
    the position of each chromosome in ``gintervals_all()`` order. Passing that
    order here yields the authoritative mapping, independent of whether the
    *genome* (vs just the track) is in indexed format.
    """
    mapping: dict[str, int] = {}
    for chrom_id, name in enumerate(chrom_order):
        _add_chrom_aliases(mapping, str(name), chrom_id)
    return mapping


def _chrom_name_to_id(track_dir: Path) -> dict[str, int]:
    """Fallback chrom name -> chrom_id from ``seq/genome.idx`` (indexed genome).

    Used only when the caller does not supply the genome chrom order. Returns
    ``{}`` if the genome index is unavailable (e.g. a per-chromosome genome whose
    track alone was converted to indexed).
    """
    groot = _groot_from_track_dir(track_dir)
    if groot is None:
        return {}
    genome_idx = groot / "seq" / "genome.idx"
    if not genome_idx.exists():
        return {}
    from .db import _iter_genome_idx_entries

    mapping: dict[str, int] = {}
    for chrom_id, name, _offset, _length in _iter_genome_idx_entries(str(genome_idx)):
        _add_chrom_aliases(mapping, name, chrom_id)
    return mapping


def _read_one_interval_values(
    fh, vals_pos: int, num_cols: int
) -> _numpy.ndarray:
    """Read one interval's value block into a length-``num_cols`` float64
    array (NaN for missing columns).
    """
    fh.seek(vals_pos)
    num_vals = struct.unpack("<I", fh.read(4))[0]
    out = _numpy.full(num_cols, _numpy.nan, dtype=_numpy.float64)
    if num_vals == 0:
        return out
    raw = fh.read(num_vals * 8)
    block = _numpy.frombuffer(raw, dtype=[("val", "<f4"), ("idx", "<u4")])
    # Defensive: drop entries whose idx is out of range.
    keep = block["idx"] < num_cols
    out[block["idx"][keep]] = block["val"][keep]
    return out


def write_chrom_file(
    filepath: Path,
    starts: _numpy.ndarray,
    ends: _numpy.ndarray,
    value_blocks: list[_numpy.ndarray],
) -> None:
    """Write a single chromosome's array-track binary file.

    *value_blocks* is a list (one per interval) of ndarrays with dtype
    ``[("val", "<f4"), ("idx", "<u4")]``. NaN values must already be
    filtered out by the caller. Intervals must be sorted by start and
    must not overlap (parallels R's `GenomeTrackArrays::write_next_interval`).
    """
    n = len(starts)
    if n != len(ends) or n != len(value_blocks):
        raise ValueError(
            "starts, ends, and value_blocks must have equal lengths"
        )
    starts = _numpy.asarray(starts, dtype=_numpy.int64)
    ends = _numpy.asarray(ends, dtype=_numpy.int64)
    if n > 0 and ((ends <= starts).any() or (starts[1:] < ends[:-1]).any()):
        raise ValueError(
            "array-track intervals must be sorted, non-overlapping, and "
            "have end > start"
        )

    with open(filepath, "wb") as fh:
        fh.write(struct.pack("<i", _ARRAYS_SIGNATURE))
        intervals_pos_offset = fh.tell()
        fh.write(struct.pack("<q", 0))  # placeholder, rewritten at end
        vals_positions: list[int] = []
        for block in value_blocks:
            vals_positions.append(fh.tell())
            if block is None or len(block) == 0:
                fh.write(struct.pack("<I", 0))
                continue
            fh.write(struct.pack("<I", len(block)))
            fh.write(block.tobytes())
        intervals_pos = fh.tell()
        fh.write(struct.pack("<Q", n))
        for s, e, vp in zip(starts.tolist(), ends.tolist(),
                            vals_positions, strict=True):
            fh.write(struct.pack("<qqq", int(s), int(e), int(vp)))
        fh.seek(intervals_pos_offset)
        fh.write(struct.pack("<q", intervals_pos))


def build_value_blocks(
    values: _numpy.ndarray,
) -> list[_numpy.ndarray]:
    """Convert a 2D (n_intervals, n_cols) float matrix into per-interval
    sparse blocks. NaN entries are dropped (matching the R format
    invariant that NaN positions are absent from the value list).
    """
    if values.ndim != 2:
        raise ValueError("values must be a 2-D matrix")
    n_intervals, n_cols = values.shape
    block_dtype = _numpy.dtype([("val", "<f4"), ("idx", "<u4")])
    out: list[_numpy.ndarray] = []
    for i in range(n_intervals):
        row = values[i].astype(_numpy.float32, copy=False)
        idx = _numpy.where(~_numpy.isnan(row))[0].astype(_numpy.uint32)
        block = _numpy.zeros(idx.size, dtype=block_dtype)
        block["idx"] = idx
        block["val"] = row[idx]
        out.append(block)
    return out


def extract_array(
    track_dir: Path,
    intervals: _pd.DataFrame,
    slice_cols: list[int] | None,
    colnames: list[str],
    chrom_order: list[str] | None = None,
) -> _pd.DataFrame:
    """Extract per-interval array values for *intervals*.

    Returns a DataFrame with ``chrom, start, end, intervalID`` plus one
    column per slice column. ``slice_cols`` are 0-based column indices
    (``None`` = all columns). ``chrom_order`` is the genome chrom-key order
    (``gintervals_all()`` chroms) used to map a chromosome to its ``track.idx``
    contig id for indexed-format tracks; if omitted it is read from
    ``seq/genome.idx`` when present.
    """
    if slice_cols is None:
        slice_cols = list(range(len(colnames)))
    sel_names = [colnames[i] for i in slice_cols]
    sel_idx = _numpy.asarray(slice_cols, dtype=_numpy.int64)

    chroms_out: list[str] = []
    starts_out: list[int] = []
    ends_out: list[int] = []
    iid_out: list[int] = []
    vals_out: list[_numpy.ndarray] = []
    num_cols = len(colnames)

    # Indexed single-file storage has no per-chrom files: data lives in
    # track.dat keyed by track.idx. Detect once and prepare the chrom-id map.
    track_index = _read_track_index(track_dir)
    if track_index is not None:
        chrom_id_map = (
            chrom_id_map_from_order(chrom_order)
            if chrom_order is not None
            else _chrom_name_to_id(track_dir)
        )
    else:
        chrom_id_map = {}
    dat_path = Path(track_dir) / "track.dat"

    # Group intervals by chrom for efficient per-chrom processing.
    iv = intervals[["chrom", "start", "end"]].reset_index(drop=True)
    iv["__iid__"] = _numpy.arange(1, len(iv) + 1, dtype=_numpy.int64)
    for chrom, group in iv.groupby("chrom", sort=False):
        # Obtain a seekable block for this chrom: a per-chrom file, or the
        # chrom's slice of the indexed track.dat read into memory.
        filepath = _resolve_chrom_file(track_dir, str(chrom))
        if filepath is not None:
            # Closed in the finally below; uniform handling with the BytesIO path.
            fh: io.IOBase = open(filepath, "rb")  # noqa: SIM115
        elif track_index is not None:
            chrom_id = chrom_id_map.get(str(chrom))
            if chrom_id is None:
                continue
            entry = track_index.get(chrom_id)
            if entry is None or entry[1] == 0:
                continue
            offset, length = entry
            with open(dat_path, "rb") as dfh:
                dfh.seek(offset)
                fh = io.BytesIO(dfh.read(length))
        else:
            continue

        try:
            ivs_start, ivs_end, ivs_pos = _read_intervals_from_fh(fh)
            if ivs_start.size == 0:
                continue
            starts = group["start"].to_numpy(dtype=_numpy.int64)
            ends = group["end"].to_numpy(dtype=_numpy.int64)
            iids = group["__iid__"].to_numpy(dtype=_numpy.int64)

            # For each query interval, emit a row for every track interval that
            # overlaps it. Sweep with searchsorted for O((N+M)*log) lookup.
            for q_start, q_end, q_iid in zip(starts, ends, iids, strict=False):
                lo = int(_numpy.searchsorted(ivs_end, q_start, side="right"))
                hi = int(_numpy.searchsorted(ivs_start, q_end, side="left"))
                if hi <= lo:
                    continue
                for j in range(lo, hi):
                    block = _read_one_interval_values(fh, int(ivs_pos[j]), num_cols)
                    chroms_out.append(str(chrom))
                    starts_out.append(max(int(ivs_start[j]), int(q_start)))
                    ends_out.append(min(int(ivs_end[j]), int(q_end)))
                    iid_out.append(int(q_iid))
                    vals_out.append(block[sel_idx])
        finally:
            fh.close()

    if not chroms_out:
        return _pd.DataFrame(
            columns=["chrom", "start", "end", *sel_names, "intervalID"]
        )

    out = _pd.DataFrame({
        "chrom": chroms_out,
        "start": starts_out,
        "end": ends_out,
    })
    val_matrix = _numpy.vstack(vals_out)
    for i, name in enumerate(sel_names):
        out[name] = val_matrix[:, i]
    out["intervalID"] = iid_out
    return out


# Valid per-row reducer functions (matches R's SliceFunctions enum).
ARRAY_REDUCE_FUNCS = {"avg", "min", "max", "sum", "stddev", "stdev"}


def reduce_array_extract(
    extracted: _pd.DataFrame,
    val_cols: list[str],
    func: str,
    n_intervals: int,
) -> _numpy.ndarray:
    """Reduce *extracted* (output of ``extract_array``) to one scalar per interval.

    Groups extracted rows by ``intervalID`` and applies *func* over all
    non-NaN values in all selected columns (matching R's
    ``GenomeTrackArrays::get_sliced_val`` cross-position / cross-column
    aggregation semantics).  Rows for interval IDs absent from *extracted*
    get NaN.
    """
    func = func.lower()
    if func not in ARRAY_REDUCE_FUNCS:
        raise ValueError(
            f"Array track aggregation function must be one of "
            f"{sorted(ARRAY_REDUCE_FUNCS)!r}, got {func!r}"
        )

    out = _numpy.full(n_intervals, _numpy.nan, dtype=_numpy.float64)

    if extracted.empty or not val_cols:
        return out

    vals = extracted[val_cols].to_numpy(dtype=_numpy.float64)  # (nrows, ncols)
    iids = extracted["intervalID"].to_numpy(dtype=_numpy.int64)  # 1-based

    # Group rows by intervalID
    groups: dict[int, list] = defaultdict(list)
    for row_i, iid in enumerate(iids):
        idx = int(iid) - 1  # 0-based
        if 0 <= idx < n_intervals:
            groups[idx].append(vals[row_i])

    for idx, rows in groups.items():
        # Concatenate all values across selected columns and all rows
        all_vals = _numpy.concatenate(list(rows))
        finite = all_vals[~_numpy.isnan(all_vals)]
        if finite.size == 0:
            continue
        if func in ("avg", "mean"):
            out[idx] = float(_numpy.mean(finite))
        elif func == "min":
            out[idx] = float(_numpy.min(finite))
        elif func == "max":
            out[idx] = float(_numpy.max(finite))
        elif func == "sum":
            out[idx] = float(_numpy.sum(finite))
        elif func in ("stddev", "stdev"):
            out[idx] = float(_numpy.std(finite, ddof=1)) if finite.size > 1 else _numpy.nan

    return out
