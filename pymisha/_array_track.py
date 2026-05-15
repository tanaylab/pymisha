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

import struct
from pathlib import Path

import numpy as _numpy
import pandas as _pd

from ._r_serialize import read as _r_read

_ARRAYS_SIGNATURE = -8
_RECORD_SIZE = 24  # 2*int64 + int64


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


def read_chrom_intervals(filepath: Path) -> tuple[
    _numpy.ndarray, _numpy.ndarray, _numpy.ndarray
]:
    """Parse the interval table of an array-track chromosome file.

    Returns ``(starts, ends, vals_pos)`` as int64 numpy arrays.
    """
    with open(filepath, "rb") as fh:
        signature = struct.unpack("<i", fh.read(4))[0]
        if signature != _ARRAYS_SIGNATURE:
            raise ValueError(
                f"{filepath}: expected array-track signature -8, got {signature}"
            )
        intervals_pos = struct.unpack("<q", fh.read(8))[0]
        fh.seek(intervals_pos)
        num_intervals = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(num_intervals * _RECORD_SIZE)
    arr = _numpy.frombuffer(raw, dtype=_numpy.int64).reshape(num_intervals, 3)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy()


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
) -> _pd.DataFrame:
    """Extract per-interval array values for *intervals*.

    Returns a DataFrame with ``chrom, start, end, intervalID`` plus one
    column per slice column. ``slice_cols`` are 0-based column indices
    (``None`` = all columns).
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

    # Group intervals by chrom for efficient per-chrom processing.
    iv = intervals[["chrom", "start", "end"]].reset_index(drop=True)
    iv["__iid__"] = _numpy.arange(1, len(iv) + 1, dtype=_numpy.int64)
    for chrom, group in iv.groupby("chrom", sort=False):
        filepath = _resolve_chrom_file(track_dir, str(chrom))
        if filepath is None:
            continue
        ivs_start, ivs_end, ivs_pos = read_chrom_intervals(filepath)
        if ivs_start.size == 0:
            continue
        starts = group["start"].to_numpy(dtype=_numpy.int64)
        ends = group["end"].to_numpy(dtype=_numpy.int64)
        iids = group["__iid__"].to_numpy(dtype=_numpy.int64)

        # For each query interval, emit a row for every track interval that
        # overlaps it. Use a sweep with searchsorted for O((N+M)*log) lookup.
        with open(filepath, "rb") as fh:
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
