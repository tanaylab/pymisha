"""Minimal R serialize-format reader for the data shapes we need.

Removes the runtime ``Rscript`` dependency for legacy intervals ``.meta``
files and lets PyMisha read array-track ``.colnames`` files that are
written as named-integer vectors by R misha.

Supported types (sufficient for: data frames, named lists, character /
integer / numeric / logical vectors, NULL, raw bytes, named atomic
vectors): NILVALUE, REFSXP (back-references), CHARSXP, LGLSXP, INTSXP,
REALSXP, STRSXP, VECSXP (generic list, with names/class attributes),
LISTSXP (pairlist - used only for attribute chains), RAWSXP.

Out of scope: closures, environments, S4 objects, byte-code, weakref,
complex (CPLXSXP) - none of which appear in the data we load.

Format references:
- R source: src/main/serialize.c (R_Unserialize, ReadItem, ReadLevels)
- The 'XDR' header is the only mode we read (R writes XDR by default).

Returned shapes mirror what callers expect:
- character vector  -> ``list[str]`` (with ``.names`` attribute set when present)
- integer / numeric -> ``numpy.ndarray`` (with ``.names`` attribute when present)
- logical           -> ``numpy.ndarray[bool]``
- NULL              -> ``None``
- named list / data.frame -> ``dict[str, value]`` (data.frame is also
  reconstructed as a ``pandas.DataFrame`` if the class attribute matches)
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as _numpy

# Type codes (subset of R's SEXP type codes).
_NILVALUE_SXP = 254
_NILSXP = 0
_REFSXP = 255
_SYMSXP = 1
_LISTSXP = 2
_CHARSXP = 9
_LGLSXP = 10
_INTSXP = 13
_REALSXP = 14
_STRSXP = 16
_VECSXP = 19
_RAWSXP = 24
_ALTREP_SXP = 238  # added in R 3.5 for compact sequences / wrappers

_NA_INT = -2147483648  # R's NA_INTEGER sentinel
_NA_LOGICAL = -2147483648


class _NamedList(list):
    """A Python list that also carries a ``names`` attribute."""

    names: list[str] | None = None


def _read_int(fh: BinaryIO) -> int:
    """Read a big-endian int32 from the stream."""
    buf = fh.read(4)
    if len(buf) != 4:
        raise EOFError("unexpected end of R-serialize stream")
    return int(struct.unpack(">i", buf)[0])


def _read_double(fh: BinaryIO) -> float:
    return float(struct.unpack(">d", fh.read(8))[0])


def _read_bytes(fh: BinaryIO, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise EOFError(f"unexpected EOF reading {n} bytes")
    return buf


def _read_header(fh: BinaryIO) -> None:
    """Consume the XDR header. Only the binary XDR mode is supported."""
    magic = fh.read(2)
    if magic != b"X\n":
        raise ValueError(
            f"only R-serialize XDR-binary format is supported (got header {magic!r})"
        )
    version = _read_int(fh)
    if version not in (2, 3):
        raise ValueError(f"unsupported R-serialize version {version}")
    _read_int(fh)  # writer R version
    _read_int(fh)  # min reader R version
    if version >= 3:
        encoding_len = _read_int(fh)
        if encoding_len > 0:
            _read_bytes(fh, encoding_len)


def _read_item(fh: BinaryIO, ref_table: list[Any]) -> Any:
    flags = _read_int(fh)
    type_code = flags & 0xFF
    has_attr = bool(flags & (1 << 9))
    has_tag = bool(flags & (1 << 10))

    if type_code in (_NILVALUE_SXP, _NILSXP):
        return None

    if type_code == _REFSXP:
        idx = (flags >> 8) - 1
        if idx < 0:
            idx = _read_int(fh) - 1
        return ref_table[idx]

    if type_code == _SYMSXP:
        # R registers the symbol in the ref-table BEFORE reading its
        # printname so back-references work; we mimic that here.
        ref_idx = len(ref_table)
        ref_table.append(None)
        name = _read_item(fh, ref_table)
        ref_table[ref_idx] = name
        return name

    if type_code == _CHARSXP:
        length = _read_int(fh)
        if length < 0:
            return None  # NA_STRING
        raw = _read_bytes(fh, length)
        # Levels (bits 12-13) carry the encoding hint; UTF-8 is the only one we honour.
        encoding = "utf-8"
        if (flags >> 12) & 0b11 == 0b00 and length > 0:
            encoding = "utf-8"
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    if type_code in (_LGLSXP, _INTSXP):
        length = _read_int(fh)
        raw = _read_bytes(fh, 4 * length)
        int_arr: _numpy.ndarray = _numpy.frombuffer(raw, dtype=">i4").astype(
            _numpy.int32, copy=True
        )
        # LGLSXP: 0 = FALSE, 1 = TRUE, NA = -INT_MAX (treated as masked False).
        int_out: _numpy.ndarray = (
            int_arr.astype(bool, copy=True) if type_code == _LGLSXP else int_arr
        )
        return _wrap_with_attrs(fh, ref_table, int_out, has_attr, has_tag)

    if type_code == _REALSXP:
        length = _read_int(fh)
        raw = _read_bytes(fh, 8 * length)
        real_arr: _numpy.ndarray = _numpy.frombuffer(raw, dtype=">f8").astype(
            _numpy.float64, copy=True
        )
        return _wrap_with_attrs(fh, ref_table, real_arr, has_attr, has_tag)

    if type_code == _STRSXP:
        length = _read_int(fh)
        str_items: list[Any] = [_read_item(fh, ref_table) for _ in range(length)]
        return _wrap_with_attrs(fh, ref_table, str_items, has_attr, has_tag)

    if type_code == _VECSXP:
        length = _read_int(fh)
        items = [_read_item(fh, ref_table) for _ in range(length)]
        return _wrap_with_attrs(fh, ref_table, items, has_attr, has_tag)

    if type_code == _RAWSXP:
        length = _read_int(fh)
        return _read_bytes(fh, length)

    if type_code == _LISTSXP:
        return _read_pairlist_any(fh, ref_table, flags)

    if type_code == _ALTREP_SXP:
        return _read_altrep(fh, ref_table)

    raise NotImplementedError(
        f"R-serialize type {type_code} is not supported by pymisha's reader. "
        "Open an issue with a sample file if you hit this."
    )


def _read_altrep(fh: BinaryIO, ref_table: list[Any]) -> Any:
    """Decode an ALTREP cell.

    An ALTREP cell is serialized as three items:
      1. info: a pairlist with the ALTREP class symbol + package symbol +
         numeric type code
      2. state: class-specific state (often a numeric/integer vector or
         pairlist)
      3. attributes: a pairlist of attributes (often NULL)

    We support the common cases:
    - ``compact_intseq`` / ``compact_realseq``: state is ``c(length, start, step)``
    - ``wrap_integer`` / ``wrap_real`` / ``wrap_logical`` / ``wrap_string``:
      state is ``list(data, ...)`` - we take the first element as the
      underlying vector.
    - ``deferred_string``: state is ``list(payload, sep)``; we coerce
      payload to strings.

    Other ALTREP classes raise ``NotImplementedError`` with the class
    name so the user knows what to file a bug for.
    """
    info = _read_item(fh, ref_table)
    state = _read_item(fh, ref_table)
    attrs = _read_item(fh, ref_table)

    # info is a pairlist; the keys depend on R version. The class name is
    # usually the first element. ReadItem returned it via _read_pairlist
    # as a dict OR if version=1 of ALTREP serialization it's an unnamed list
    # accessed positionally. Be defensive.
    cls_name: str | None = None
    if isinstance(info, dict):
        # The class name is typically the first value when iterated
        # insertion-order; or under a known key.
        for v in info.values():
            if isinstance(v, str):
                cls_name = v
                break
    elif isinstance(info, list) and info:
        first = info[0]
        if isinstance(first, str):
            cls_name = first

    if cls_name is None:
        raise NotImplementedError(
            f"could not identify ALTREP class from info={info!r}"
        )

    decoded = _decode_altrep_state(cls_name, state)
    return _apply_attributes(decoded, attrs) if isinstance(attrs, dict) else decoded


def _decode_altrep_state(cls_name: str, state: Any) -> Any:
    if cls_name in ("compact_intseq", "compact_realseq"):
        # state is c(length, start, step)
        if isinstance(state, _numpy.ndarray) and state.size == 3:
            length = int(state[0])
            if cls_name == "compact_intseq":
                start_i = int(state[1])
                step_i = int(state[2])
                return _numpy.arange(length, dtype=_numpy.int32) * step_i + start_i
            start_f = float(state[1])
            step_f = float(state[2])
            return (
                _numpy.arange(length, dtype=_numpy.float64) * step_f + start_f
            )
        raise NotImplementedError(
            f"{cls_name} state was {type(state).__name__}: {state!r}"
        )
    if cls_name.startswith("wrap_"):
        # state is a list whose first element is the underlying vector
        if isinstance(state, list) and state:
            return state[0]
        return state
    if cls_name == "deferred_string" and isinstance(state, list) and state:
        payload = state[0]
        if isinstance(payload, _numpy.ndarray):
            return [str(v) for v in payload.tolist()]
        return payload
    raise NotImplementedError(
        f"ALTREP class {cls_name!r} is not supported by pymisha's reader. "
        "Open an issue with a sample file if you hit this."
    )


def _read_pairlist(
    fh: BinaryIO, ref_table: list[Any], flags: int
) -> dict[str, Any]:
    """Read a LISTSXP chain into a dict keyed by tag name.

    Used for attribute lists, where every cell is tagged with the
    attribute name. Drops untagged cells.
    """
    cells = _read_pairlist_cells(fh, ref_table, flags)
    return {tag: val for tag, val in cells if tag is not None}


def _read_pairlist_any(
    fh: BinaryIO, ref_table: list[Any], flags: int
) -> Any:
    """Read a LISTSXP chain and return either a dict (if every cell is
    tagged) or a positional list (otherwise)."""
    cells = _read_pairlist_cells(fh, ref_table, flags)
    if cells and all(tag is not None for tag, _ in cells):
        return dict(cells)  # type: ignore[arg-type]
    return [val for _, val in cells]


def _read_pairlist_cells(
    fh: BinaryIO, ref_table: list[Any], flags: int
) -> list[tuple[str | None, Any]]:
    cells: list[tuple[str | None, Any]] = []
    while True:
        type_code = flags & 0xFF
        if type_code == _NILVALUE_SXP:
            return cells
        if type_code != _LISTSXP:
            raise ValueError(f"expected LISTSXP in pairlist, got {type_code}")
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if has_attr:
            _read_item(fh, ref_table)
        tag_name: str | None = None
        if has_tag:
            tag = _read_item(fh, ref_table)
            tag_name = tag if isinstance(tag, str) else None
        value = _read_item(fh, ref_table)
        cells.append((tag_name, value))
        flags = _read_int(fh)


def _wrap_with_attrs(
    fh: BinaryIO,
    ref_table: list[Any],
    value: Any,
    has_attr: bool,
    _has_tag: bool,
) -> Any:
    if not has_attr:
        return value
    attr_flags = _read_int(fh)
    attrs = _read_pairlist(fh, ref_table, attr_flags)
    return _apply_attributes(value, attrs)


def _apply_attributes(value: Any, attrs: dict[str, Any]) -> Any:
    """Apply R attributes to a Python value.

    - ``names`` on an atomic vector -> set ``.names`` (a Python list[str]).
    - ``names`` on a list -> dict (zip(names, list)).
    - ``class == "data.frame"`` on a list -> ``pandas.DataFrame``.
    """
    names = attrs.get("names")
    cls = attrs.get("class")

    if isinstance(value, list):
        # A VECSXP. If names are present, materialize a dict.
        if isinstance(names, list):
            named = dict(zip(names, value, strict=False))
            if cls and isinstance(cls, list) and "data.frame" in cls:
                try:
                    import pandas as _pd
                    return _pd.DataFrame(
                        {k: _ndarray_or_passthrough(v) for k, v in named.items()}
                    )
                except ImportError:
                    return named
            return named
        return value

    if isinstance(value, _numpy.ndarray) and isinstance(names, list):
        wrapped = value.view(_NamedArray)
        wrapped.names = list(names)
        return wrapped

    return value


def _ndarray_or_passthrough(v: Any) -> Any:
    if isinstance(v, list) and all(isinstance(x, str) or x is None for x in v):
        return v
    return v


class _NamedArray(_numpy.ndarray):
    """A numpy ndarray that carries a ``names`` attribute."""

    names: list[str] | None = None

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.names = getattr(obj, "names", None)


def read(path: str | Path) -> Any:
    """Read an R-serialize file at *path* and return a Python object.

    Supports the XDR (binary) mode that R uses by default, including
    gzip-compressed RDS files (``saveRDS()`` output). Returned types are
    documented at the module level.
    """
    import gzip
    import io

    with open(path, "rb") as fh:
        head = fh.read(2)
        fh.seek(0)
        if head == b"\x1f\x8b":
            data = gzip.decompress(fh.read())
            return read_stream(io.BytesIO(data))
        return read_stream(fh)


def read_stream(fh: BinaryIO) -> Any:
    """Read an R-serialize stream from an already-open binary file handle."""
    _read_header(fh)
    ref_table: list[Any] = []
    return _read_item(fh, ref_table)


def read_named_vector(path: str | Path) -> list[str]:
    """Convenience: read a named character or integer vector and return
    its names. Used for misha ``.colnames`` files.
    """
    obj = read(path)
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return list(obj)
    names = getattr(obj, "names", None)
    if isinstance(names, list):
        return list(names)
    raise ValueError(
        f"expected a named vector in {path}, got {type(obj).__name__}"
    )
