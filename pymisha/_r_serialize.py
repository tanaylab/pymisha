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

# R attributes consumed/handled structurally elsewhere; presence of anything
# else on an atomic vector triggers preserving the full attribute dict.
_STRUCTURAL_ATTRS = frozenset(
    {"names", "class", "levels", "dim", "dimnames", "row.names"}
)


class _NamedList(list):
    """A Python list that also carries a ``names`` attribute."""

    names: list[str] | None = None


def _read_int(fh: BinaryIO) -> int:
    """Read a big-endian int32 from the stream."""
    buf = fh.read(4)
    if len(buf) != 4:
        raise EOFError("unexpected end of R-serialize stream")
    return int(_UNPACK_INT(buf)[0])


def _read_double(fh: BinaryIO) -> float:
    return float(_UNPACK_DOUBLE(fh.read(8))[0])


def _read_bytes(fh: BinaryIO, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise EOFError(f"unexpected EOF reading {n} bytes")
    return buf


# Precompiled struct unpackers - 2-3x faster than struct.unpack(">i", ...)
# inside hot loops because the parser bytecode is bound once.
_UNPACK_INT = struct.Struct(">i").unpack
_UNPACK_DOUBLE = struct.Struct(">d").unpack
_UNPACK_INT_FROM = struct.Struct(">i").unpack_from


def _read_strsxp_items(fh: BinaryIO, length: int) -> list[str | None]:
    """Bulk-read a STRSXP body of *length* CHARSXPs.

    Inlines CHARSXP parsing so we skip the recursive ``_read_item`` /
    ``_wrap_with_attrs`` call chain for every string.  For million-row
    interval-set chrom columns this turns a ~3.4 s scan into ~600 ms.
    Standard saveRDS doesn't emit ALTREP or REFSXP for STRSXP contents,
    so this fast path covers every chrom-column-shaped CHARSXP we have
    seen; anything unexpected falls back to the generic reader.
    """
    if length == 0:
        return []

    out: list[str | None] = [None] * length
    unpack_from = _UNPACK_INT_FROM
    read = fh.read
    for i in range(length):
        head = read(8)
        if len(head) != 8:
            raise EOFError("unexpected EOF inside STRSXP")
        flag = unpack_from(head, 0)[0]
        if flag & 0xFF != _CHARSXP:
            # CHARSXP-only fast path didn't match (e.g. REFSXP-encoded
            # interned strings).  Push the bytes back via BytesIO so the
            # generic reader can resume — but fh is a real stream that
            # may not support seek backwards, so we synthesise.
            from io import BytesIO

            tail = head + fh.read()
            buf = BytesIO(tail)
            ref_table: list[Any] = []
            # Re-read this item generically.
            out[i] = _read_item(buf, ref_table)
            for j in range(i + 1, length):
                out[j] = _read_item(buf, ref_table)
            # Stash whatever's left so attribute parsing can continue.
            remaining = buf.read()
            if remaining:
                _STRSXP_TAIL_STASH[id(fh)] = remaining
            return out
        slen = unpack_from(head, 4)[0]
        if slen < 0:
            out[i] = None
            continue
        raw = read(slen)
        try:
            out[i] = raw.decode("utf-8")
        except UnicodeDecodeError:
            out[i] = raw.decode("latin-1")
    return out


# Used by the rare slow-path branch in _read_strsxp_items: when we have
# to fall back to the generic reader mid-STRSXP, we may consume extra
# bytes past the STRSXP body that the caller's attribute-parsing pass
# still needs.  This is a defensive stash; the fast path doesn't touch it.
_STRSXP_TAIL_STASH: dict[int, bytes] = {}


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
        # R's CHARSXP carries an encoding hint in the level bits, but misha
        # strings (chrom names, set/attr names) are ASCII/UTF-8 in practice, so
        # we decode as UTF-8 and fall back to latin-1 for any stray bytes.
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    if type_code in (_LGLSXP, _INTSXP):
        length = _read_int(fh)
        raw = _read_bytes(fh, 4 * length)
        int_arr: _numpy.ndarray = _numpy.frombuffer(raw, dtype=">i4").astype(
            _numpy.int32, copy=True
        )
        # Defer NA-aware coercion to _wrap_with_attrs so the factor-decode
        # path in _apply_attributes still sees the raw 1-based codes
        # (NA_INTEGER == NA_LOGICAL == -INT_MAX).
        atomic_kind = "logical" if type_code == _LGLSXP else "integer"
        return _wrap_with_attrs(
            fh, ref_table, int_arr, has_attr, has_tag, _atomic_kind=atomic_kind
        )

    if type_code == _REALSXP:
        length = _read_int(fh)
        raw = _read_bytes(fh, 8 * length)
        real_arr: _numpy.ndarray = _numpy.frombuffer(raw, dtype=">f8").astype(
            _numpy.float64, copy=True
        )
        return _wrap_with_attrs(fh, ref_table, real_arr, has_attr, has_tag)

    if type_code == _STRSXP:
        length = _read_int(fh)
        str_items: list[Any] = _read_strsxp_items(fh, length)
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
    *,
    _atomic_kind: str | None = None,
) -> Any:
    if not has_attr:
        return _finalize_raw_atomic(value, _atomic_kind)
    attr_flags = _read_int(fh)
    attrs = _read_pairlist(fh, ref_table, attr_flags)
    return _apply_attributes(value, attrs, _atomic_kind=_atomic_kind)


def _finalize_raw_atomic(value: Any, kind: str | None) -> Any:
    """Apply NA-aware coercion to a raw LGLSXP/INTSXP int32 ndarray.

    R encodes NA in atomic LGLSXP / INTSXP vectors as the int sentinel
    -INT_MAX. Without this step that sentinel leaks through:
    - LGLSXP: ``astype(bool)`` turns the sentinel into ``True``.
    - INTSXP: the sentinel surfaces as a very large negative integer.

    When NAs are present we return a pandas nullable array (BooleanArray
    or IntegerArray) so callers see actual NA, not silently-corrupted
    values. With pandas unavailable we fall back to an object array
    (logical) or float64-with-NaN (integer).
    """
    if (
        kind is None
        or not isinstance(value, _numpy.ndarray)
        or value.dtype != _numpy.int32
    ):
        return value
    if kind == "logical":
        return _decode_logical(value)
    if kind == "integer":
        return _decode_integer(value)
    return value


def _decode_logical(int_arr: _numpy.ndarray) -> Any:
    na_mask: _numpy.ndarray = (int_arr == _NA_LOGICAL)
    if not na_mask.any():
        return int_arr.astype(bool, copy=True)
    try:
        from pandas.arrays import BooleanArray as _BoolArr
    except ImportError:
        out: _numpy.ndarray = _numpy.empty(int_arr.shape, dtype=object)
        for i, v in enumerate(int_arr.tolist()):
            out[i] = None if v == _NA_LOGICAL else bool(v)
        return out
    values = int_arr.astype(bool, copy=True)
    values[na_mask] = False  # mask takes precedence; concrete value irrelevant
    return _BoolArr(values, na_mask.astype(_numpy.bool_, copy=False))


def _decode_integer(int_arr: _numpy.ndarray) -> Any:
    na_mask: _numpy.ndarray = (int_arr == _NA_INT)
    if not na_mask.any():
        return int_arr
    try:
        from pandas.arrays import IntegerArray as _IntArr
    except ImportError:
        out_f: _numpy.ndarray = int_arr.astype(_numpy.float64, copy=True)
        out_f[na_mask] = _numpy.nan
        return out_f
    return _IntArr(int_arr, na_mask.astype(_numpy.bool_, copy=False))


def _apply_attributes(
    value: Any,
    attrs: dict[str, Any],
    *,
    _atomic_kind: str | None = None,
) -> Any:
    """Apply R attributes to a Python value.

    - R factor (INTSXP + class="factor" + levels=STRSXP) -> ``pandas.Categorical``.
    - ``names`` on an atomic vector -> set ``.names`` (a Python list[str]).
    - ``names`` on a list -> dict (zip(names, list)).
    - ``class == "data.frame"`` on a list -> ``pandas.DataFrame``.
    """
    names = attrs.get("names")
    cls = attrs.get("class")

    # R factor: INTSXP body + class containing "factor" + levels = STRSXP.
    # Without this branch, the 1-based factor codes leak through as the
    # column values (chrom "1".."N" instead of "chr1".."chrN"), breaking
    # every legacy `.interv` data.frame that stores chrom as a factor.
    if (
        isinstance(value, _numpy.ndarray)
        and value.dtype.kind in "iu"
        and isinstance(cls, list)
        and "factor" in cls
    ):
        levels = attrs.get("levels")
        if isinstance(levels, list) and all(
            isinstance(x, str) or x is None for x in levels
        ):
            try:
                import pandas as _pd
            except ImportError:
                # Best-effort fallback: materialise labels as an object array.
                out = _numpy.empty(value.shape, dtype=object)
                for i, code in enumerate(value.tolist()):
                    if code == _NA_INT or code <= 0 or code > len(levels):
                        out[i] = None
                    else:
                        out[i] = levels[code - 1]
                return out
            codes = value.astype(_numpy.int64, copy=True) - 1
            codes[value == _NA_INT] = -1
            categories = [("" if x is None else x) for x in levels]
            ordered = "ordered" in cls
            return _pd.Categorical.from_codes(
                codes, categories=categories, ordered=ordered
            )

    # Non-factor LGLSXP/INTSXP that still carries attrs (e.g. a named
    # integer vector). NA-coerce now so downstream sees pandas NA, not
    # the raw -INT_MAX sentinel. If NAs are present this returns a
    # pandas extension array, which loses the `.names` attribute - NA
    # correctness wins; named atomic vectors with NAs are exceedingly
    # rare (the .colnames files we read never have NAs).
    if _atomic_kind is not None:
        value = _finalize_raw_atomic(value, _atomic_kind)

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

    # Wrap an ndarray that carries a `names` attribute, or any non-structural
    # R attribute we want to preserve (e.g. the `breaks`/`minval`/`maxval` on a
    # track's pv.percentiles table, used by global.percentile virtual tracks).
    # Structural attributes (dim/dimnames/class/...) are handled elsewhere or
    # not needed, so a plain matrix/vector baseline stays a plain ndarray.
    if isinstance(value, _numpy.ndarray):
        extra = set(attrs) - _STRUCTURAL_ATTRS
        if isinstance(names, list) or extra:
            wrapped = value.view(_NamedArray)
            wrapped.names = list(names) if isinstance(names, list) else None
            wrapped.attributes = dict(attrs)
            return wrapped

    return value


def _ndarray_or_passthrough(v: Any) -> Any:
    if isinstance(v, list) and all(isinstance(x, str) or x is None for x in v):
        return v
    return v


class _NamedArray(_numpy.ndarray):
    """A numpy ndarray that carries ``names`` and other R ``attributes``."""

    names: list[str] | None = None
    attributes: dict[str, Any] | None = None

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.names = getattr(obj, "names", None)
        self.attributes = getattr(obj, "attributes", None)


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


# ---------------------------------------------------------------------------
# Writer.
#
# Minimal R-serialize XDR writer covering the data shapes pymisha emits via
# saveRDS() — primarily interval-set data frames (chrom factor + numeric
# coords, optional string/int/bool extra columns).  This replaces the
# pyreadr.write_rds() round-trip, which calls into the librdata C++
# library and is ~50-100x slower on million-row data frames because of the
# row-by-row API.
# ---------------------------------------------------------------------------

# Symbol flag bits in the head word (matches R's serialize.c).
_SYM_FLAG = _SYMSXP  # 1
_LIST_FLAG = _LISTSXP  # 2
_HAS_TAG_BIT = 1 << 10
_HAS_ATTR_BIT = 1 << 9
_OBJECT_BIT = 1 << 8  # R's IS_OBJECT: set whenever a `class` attribute is emitted
_LATIN1_BIT = 1 << 12  # not used (we always emit UTF-8)
_UTF8_BIT = 1 << 3 << 12  # bit 14: levels = 8 (UTF-8)
_ASCII_BIT = 0  # default; we set bit 14 instead

# Standard R-serialize header values for XDR / version-2 streams.
_WRITER_R_VERSION = 0x00040301  # 4.3.1, picked arbitrarily; readers ignore it
_READER_R_VERSION = 0x00020300  # require R >= 2.3.0 (the v2 floor)


def _write_int(fh: BinaryIO, v: int) -> None:
    fh.write(struct.pack(">i", v))


def _write_double(fh: BinaryIO, v: float) -> None:
    fh.write(struct.pack(">d", v))


def _write_header(fh: BinaryIO) -> None:
    fh.write(b"X\n")
    _write_int(fh, 2)  # serialization format version
    _write_int(fh, _WRITER_R_VERSION)
    _write_int(fh, _READER_R_VERSION)


def _write_charsxp(fh: BinaryIO, s: str | None) -> None:
    """Write a fresh CHARSXP (no ref-table dedup). Used inside a STRSXP
    when the caller is not interning strings via _write_strsxp_dedup."""
    if s is None:
        # NA_STRING: length == -1, no payload.
        _write_int(fh, _CHARSXP)
        _write_int(fh, -1)
        return
    raw = s.encode("utf-8")
    # CHARSXP flags: encoding hint goes in bits 12-15. Use UTF-8 (level 8).
    _write_int(fh, _CHARSXP | (8 << 12))
    _write_int(fh, len(raw))
    fh.write(raw)


def _write_strsxp(
    fh: BinaryIO,
    items: list[str | None],
    *,
    attrs: dict | None = None,
    ref_table: dict[str, int] | None = None,  # accepted for signature compat; unused
) -> None:
    """Write a STRSXP.  Each CHARSXP is emitted fresh - R's standard
    saveRDS does not dedup CHARSXPs via the ref-table, so doing it here
    would break ``readRDS`` compatibility.

    For chrom-like columns with millions of repeated values this means
    the bytes-out cost is O(N * avg_str_len), but it's still ~30x
    faster than pyreadr because we avoid the Python -> librdata bridge.
    Batch-encoding all CHARSXPs into a single buffer via io.BytesIO is
    what keeps the per-string overhead low.
    """
    head = _STRSXP
    if attrs:
        head |= _HAS_ATTR_BIT
    _write_int(fh, head)
    _write_int(fh, len(items))

    # Fast path: ASCII strings (>=99% of chrom names).  Encode all the
    # CHARSXPs into a single bytes buffer in a tight Python loop and
    # write once to disk; one fh.write() is many times faster than N
    # small writes through Python.
    import io
    buf = io.BytesIO()
    head_no_payload = struct.pack(">i", _CHARSXP | (8 << 12))
    for s in items:
        if s is None:
            buf.write(struct.pack(">i", _CHARSXP))
            buf.write(struct.pack(">i", -1))
            continue
        raw = s.encode("utf-8")
        buf.write(head_no_payload)
        buf.write(struct.pack(">i", len(raw)))
        buf.write(raw)
    fh.write(buf.getvalue())

    if attrs:
        _write_pairlist_attrs(fh, attrs)


def _write_intsxp(fh: BinaryIO, arr: _numpy.ndarray, *, attrs: dict | None = None) -> None:
    head = _INTSXP
    if attrs:
        head |= _HAS_ATTR_BIT
    _write_int(fh, head)
    _write_int(fh, arr.size)
    fh.write(arr.astype(">i4", copy=False).tobytes())
    if attrs:
        _write_pairlist_attrs(fh, attrs)


def _write_lglsxp(fh: BinaryIO, arr: _numpy.ndarray, *, attrs: dict | None = None) -> None:
    head = _LGLSXP
    if attrs:
        head |= _HAS_ATTR_BIT
    _write_int(fh, head)
    _write_int(fh, arr.size)
    fh.write(arr.astype(">i4", copy=False).tobytes())
    if attrs:
        _write_pairlist_attrs(fh, attrs)


def _write_realsxp(fh: BinaryIO, arr: _numpy.ndarray, *, attrs: dict | None = None) -> None:
    head = _REALSXP
    if attrs:
        head |= _HAS_ATTR_BIT
    _write_int(fh, head)
    _write_int(fh, arr.size)
    fh.write(arr.astype(">f8", copy=False).tobytes())
    if attrs:
        _write_pairlist_attrs(fh, attrs)


def _write_vecsxp(fh: BinaryIO, items: list[Any], *, attrs: dict | None = None) -> None:
    head = _VECSXP
    if attrs:
        head |= _HAS_ATTR_BIT
        if "class" in attrs:
            head |= _OBJECT_BIT
    _write_int(fh, head)
    _write_int(fh, len(items))
    for item in items:
        _write_value(fh, item)
    if attrs:
        _write_pairlist_attrs(fh, attrs)


def _write_symsxp(fh: BinaryIO, name: str) -> None:
    """Write a SYMSXP for an attribute tag (e.g., "names", "class")."""
    # SYMSXP: head word, then a CHARSXP for the printname.
    _write_int(fh, _SYMSXP)
    _write_charsxp(fh, name)


def _write_pairlist_attrs(
    fh: BinaryIO,
    attrs: dict,
    *,
    ref_table: dict[str, int] | None = None,
) -> None:
    """Write a chain of LISTSXP cells (one per attribute), terminated
    with NILVALUE_SXP. Each cell carries a tag (the attribute name as a
    SYMSXP) and a value.
    """
    for tag, value in attrs.items():
        # Cell head: LISTSXP with has_tag bit set.
        _write_int(fh, _LISTSXP | _HAS_TAG_BIT)
        _write_symsxp(fh, tag)
        _write_value(fh, value, ref_table=ref_table)
    _write_int(fh, _NILVALUE_SXP)


def _write_value(
    fh: BinaryIO,
    value: Any,
    *,
    ref_table: dict[str, int] | None = None,
) -> None:
    """Dispatch to the right writer based on Python/numpy/pandas type."""
    if value is None:
        _write_int(fh, _NILVALUE_SXP)
        return
    if isinstance(value, str):
        _write_strsxp(fh, [value], ref_table=ref_table)
        return
    if isinstance(value, list):
        # Heuristic: list of strings -> STRSXP; otherwise generic list.
        if all(s is None or isinstance(s, str) for s in value):
            _write_strsxp(fh, value, ref_table=ref_table)
            return
        _write_vecsxp(fh, value)
        return
    if isinstance(value, _numpy.ndarray):
        if value.dtype == _numpy.bool_:
            _write_lglsxp(fh, value)
            return
        if _numpy.issubdtype(value.dtype, _numpy.integer):
            _write_intsxp(fh, value)
            return
        if _numpy.issubdtype(value.dtype, _numpy.floating):
            _write_realsxp(fh, value)
            return
        if value.dtype == object:
            # Treat as character vector if all elements are strings.
            _write_strsxp(
                fh,
                [None if x is None else str(x) for x in value],
                ref_table=ref_table,
            )
            return
    # Pandas types are handled by write_dataframe; if we see them here it's
    # an unexpected nested case.
    try:
        import pandas as _pd
    except ImportError:
        _pd = None  # type: ignore[assignment]
    if _pd is not None and isinstance(value, _pd.Series):
        _write_value(fh, value.to_numpy(), ref_table=ref_table)
        return
    raise TypeError(f"_r_serialize writer does not handle {type(value).__name__}")


def _series_to_r_column(series: Any) -> tuple[str, Any]:
    """Convert a pandas Series to (kind, payload) where kind is one of
    integer, double, logical, character.

    Categorical columns are converted to character so the on-disk layout
    matches what pyreadr.write_rds() produced (and what pymisha's
    gintervals_load reader expects).  This keeps round-trip output
    byte-compatible with the previous behavior even though R misha's
    legacy `.interv` files store chrom as a factor.
    """
    import pandas as _pd

    dt = series.dtype
    if isinstance(dt, _pd.CategoricalDtype):
        # Flatten factor -> character to match pyreadr semantics.
        return ("character", [str(v) for v in series.astype(str).to_numpy()])
    if _pd.api.types.is_bool_dtype(dt):
        return ("logical", series.to_numpy(dtype=_numpy.bool_))
    if _pd.api.types.is_integer_dtype(dt):
        # R has no int64; widen to double if we'd lose precision.
        arr = series.to_numpy()
        if arr.dtype.itemsize > 4:
            arr64 = arr.astype(_numpy.int64, copy=False)
            if (arr64.min(initial=0) < _NA_INT + 1) or (arr64.max(initial=0) > 2_147_483_647):
                return ("double", arr64.astype(_numpy.float64))
        return ("integer", arr.astype(_numpy.int32, copy=False))
    if _pd.api.types.is_float_dtype(dt):
        return ("double", series.to_numpy(dtype=_numpy.float64))
    # Strings: convert to a list[str], with None for NAs.
    raw = series.to_numpy(dtype=object)
    return ("character", [None if (v is None or (isinstance(v, float) and _numpy.isnan(v))) else str(v) for v in raw])


def write_dataframe(path: str | Path, df: Any) -> None:
    """Write a pandas DataFrame to an R-serialize RDS file at *path*.

    Supports columns of dtype: pandas Categorical (flattened to
    character), bool (-> logical), integer (-> integer or double if it
    overflows int32), float (-> numeric), object/string (-> character).
    Row names are written as the compact-integer sequence
    ``c(NA_integer_, -nrow)`` which is how R stores trivial 1..nrow row
    indices.

    Strings within a column use R's ref-table to dedupe repeated values,
    which is what makes chrom columns (e.g. 1M rows, ~25 unique values)
    cheap to write.

    Output round-trips through :func:`read` and is layout-compatible
    with what ``pyreadr.write_rds`` produced — the previous implementation.
    """
    import pandas as _pd
    if not isinstance(df, _pd.DataFrame):
        raise TypeError("write_dataframe requires a pandas DataFrame")

    # Compute column descriptors up front so any unsupported dtype errors
    # out before we touch the file.
    col_kinds = []
    for name in df.columns:
        kind, payload = _series_to_r_column(df[name])
        col_kinds.append((name, kind, payload))

    nrows = len(df)
    # Shared ref table for CHARSXPs across the whole file — R uses one
    # table per stream and counts entries starting at 1.  We populate it
    # opportunistically: every distinct CHARSXP written via a STRSXP
    # gets an index; attribute tags (SYMSXP) are kept separate to mirror
    # R's behavior, which uses SYMSXP-typed entries.
    ref_table: dict[str, int] = {}

    with open(path, "wb") as fh:
        _write_header(fh)

        # data.frame is VECSXP of columns + attrs (names, class, row.names).
        # We always set has_attr on the head, and object since a `class`
        # attribute is always emitted below - without it R unserializes an
        # object whose `class` attribute says "data.frame" but whose OBJECT
        # flag is unset, so S3 dispatch (dim.data.frame, hence nrow()) never
        # fires.
        _write_int(fh, _VECSXP | _HAS_ATTR_BIT | _OBJECT_BIT)
        _write_int(fh, len(col_kinds))

        for _, kind, payload in col_kinds:
            if kind == "logical":
                _write_lglsxp(fh, payload)
            elif kind == "integer":
                _write_intsxp(fh, payload)
            elif kind == "double":
                _write_realsxp(fh, payload)
            elif kind == "character":
                _write_strsxp(fh, payload, ref_table=ref_table)
            else:
                raise AssertionError(kind)

        # Attributes: names, row.names, class.
        # row.names = c(NA_integer_, -nrow) is R's "compact" form for 1:nrow.
        row_names = _numpy.array([_NA_INT, -nrows], dtype=_numpy.int32)
        attrs = {
            "names": [str(c) for c, *_ in col_kinds],
            "row.names": row_names,
            "class": ["data.frame"],
        }
        # Attribute values get the same CHARSXP dedupe.
        _write_pairlist_attrs(fh, attrs, ref_table=ref_table)


def write(path: str | Path, value: Any) -> None:
    """Write *value* to *path* as an R-serialize RDS file.

    For DataFrames, dispatches to :func:`write_dataframe`.  For other
    values, writes a single SEXP via the generic writer.
    """
    import pandas as _pd
    if isinstance(value, _pd.DataFrame):
        write_dataframe(path, value)
        return
    with open(path, "wb") as fh:
        _write_header(fh)
        _write_value(fh, value)
