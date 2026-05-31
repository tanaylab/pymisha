"""Python port of R misha's Computer2D framework (KICKOFF-8 G3).

Currently supports CT2_AREA (= 0) + CT2_TEST (= 3) - the trivial
computers used by AreaComputer2D + TestComputer2D in R's HiCComputers.
CT2_POTENTIAL (= 1) + CT2_TECHNICAL (= 2) are deferred (no test fixture
exercises them; only real Hi-C normalisation workflows do).

R sources:
- src/Computer2D.{h,cpp} (factory + serialize dispatcher)
- src/HiCComputers.{h,cpp} (AreaComputer2D + TestComputer2D)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

# Mirrors Computer2D::Computer2DType in R's Computer2D.h.
CT2_AREA = 0
CT2_POTENTIAL = 1
CT2_TECHNICAL = 2
CT2_TEST = 3

_SUPPORTED_TYPES = {CT2_AREA, CT2_TEST}


def skip_computer2d_header(data: Any, offset: int) -> int:
    """Advance past the Computer2D header in a COMPUTED 2D track file.

    Returns the byte offset where the StatQuadTreeCached payload begins
    (i.e. the offset of ``num_objs``).  ``data`` may be any buffer-like
    object accepted by ``struct.unpack_from`` (mmap, bytes, bytearray, ...).
    """
    ct_type = int(struct.unpack_from("<i", data, offset)[0])
    offset += 4
    if ct_type in _SUPPORTED_TYPES:
        # CT2_AREA / CT2_TEST have no extra per-instance state on disk.
        return offset
    raise NotImplementedError(
        f"COMPUTED 2D track uses unsupported computer type {ct_type} "
        f"(supported: CT2_AREA={CT2_AREA}, CT2_TEST={CT2_TEST}). "
        "PotentialComputer2D / TechnicalComputer2D are deferred."
    )


def read_computer2d_type(data: Any, offset: int = 4) -> int:
    """Return the Computer2DType byte from a COMPUTED 2D track file.

    The caller is responsible for verifying the leading signature is -11.
    """
    return int(struct.unpack_from("<i", data, offset)[0])


# --------------------------------------------------------------------------- #
# Rectangle / DiagonalBand value types (R parity with src/Rectangle.h and
# src/DiagonalBand.h)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rectangle:
    """Mirrors R's ``Rectangle`` (axis-aligned, half-open on the upper edge)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class DiagonalBand:
    """Half-open diagonal band: cells where ``d1 <= (x - y) < d2``.

    Mirrors R's ``DiagonalBand`` (src/DiagonalBand.h).  ``active`` is True
    when the band is non-trivial (R distinguishes a default-constructed
    inactive band from a real one).
    """

    d1: int
    d2: int

    @property
    def active(self) -> bool:
        return self.d1 != 0 or self.d2 != 0

    def do_intersect(self, r: Rectangle) -> bool:
        # R: rect.x2 - rect.y1 > d1 && rect.x1 - rect.y2 + 1 < d2
        return (r.x2 - r.y1 > self.d1) and (r.x1 - r.y2 + 1 < self.d2)

    def do_contain(self, r: Rectangle) -> bool:
        # R: rect.x1 - rect.y2 + 1 >= d1 && rect.x2 - rect.y1 <= d2
        return (r.x1 - r.y2 + 1 >= self.d1) and (r.x2 - r.y1 <= self.d2)

    # R's diagonal-band parameterisation helpers (DiagonalBand.h:30-33).
    def x1_at(self, y1: int) -> int:
        return y1 + self.d1

    def y1_at(self, x1: int) -> int:
        return x1 - self.d1

    def x2_at(self, y2: int) -> int:
        return y2 + self.d2

    def y2_at(self, x2: int) -> int:
        return x2 - self.d2

    def shrink2intersected(self, r: Rectangle) -> Rectangle:
        """Return the minimal rect bounding the original ∩ band.

        Mirrors R's ``DiagonalBand::shrink2intersected`` (in-place there;
        we return a new Rectangle for immutability).
        """
        x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
        if x1 - y1 < self.d1:
            x1 = self.x1_at(y1)
        elif x1 - y1 > self.d2:
            y1 = self.y2_at(x1)

        if x2 - y2 < self.d1:
            y2 = self.y1_at(x2)
        elif x2 - y2 > self.d2:
            x2 = self.x2_at(y2)
        return Rectangle(x1, y1, x2, y2)


def intersected_area(r: Rectangle, band: DiagonalBand) -> int:
    """Area of ``r`` falling inside the half-open band ``[d1, d2)``.

    Mirrors R's ``DiagonalBand::intersected_area`` (which assumes ``r`` is
    already shrunk to its band-intersection bounding box, then subtracts
    the two off-band corner triangles).
    """
    area = r.area
    # Subtract the triangle above d1 (cells with x - y < d1).
    if r.x1 - r.y2 + 1 < band.d1:
        n = band.x1_at(r.y2) - r.x1
        area -= (n * n - n) >> 1
    # Subtract the triangle below d2 (cells with x - y > d2 - 1).
    if r.x2 - r.y1 > band.d2:
        n = r.x2 - band.x2_at(r.y1)
        area -= (n * n + n) >> 1
    return area


# --------------------------------------------------------------------------- #
# Computer2D implementations
# --------------------------------------------------------------------------- #


class Computer2D:
    """Abstract base: per-rectangle value lookup (R parity)."""

    def compute(self, r: Rectangle, band: DiagonalBand | None = None) -> float:
        raise NotImplementedError


class AreaComputer2D(Computer2D):
    """Constant 1.0; with a band, returns the band-intersection area fraction.

    Mirrors R's ``AreaComputer2D::compute``.
    """

    def compute(self, r: Rectangle, band: DiagonalBand | None = None) -> float:
        if band is None or not band.active:
            return 1.0
        if not band.do_intersect(r):
            return 0.0
        if band.do_contain(r):
            return 1.0
        return intersected_area(r, band) / r.area


class TestComputer2D(Computer2D):
    """``(x1+x2+y1+y2[+d1+d2]) % 10_000_000`` - R-side test fixture.

    Mirrors R's ``TestComputer2D::compute`` exactly, including the C-style
    ``%`` semantics (truncation toward zero, sign of the dividend - matters
    when band-shrinking produces negative coordinates).
    """

    _MOD = 10_000_000

    def compute(self, r: Rectangle, band: DiagonalBand | None = None) -> float:
        s = r.x1 + r.x2 + r.y1 + r.y2
        if band is not None and band.active:
            s += band.d1 + band.d2
        # C-style integer mod (truncation toward zero, sign of dividend):
        # ``s - trunc(s / mod) * mod``.  Python ``//`` floor-divides, so
        # negate-and-negate to get truncation for negatives.
        if s >= 0:
            return float(s - (s // self._MOD) * self._MOD)
        return float(s - (-(-s // self._MOD)) * self._MOD)


def create_computer2d(ct_type: int) -> Computer2D:
    """Factory mirroring ``Computer2D::unserializeComputer2D``.

    Returns a freshly-constructed computer of the requested type. Raises
    ``NotImplementedError`` for CT2_POTENTIAL / CT2_TECHNICAL (deferred)
    and ``ValueError`` for any unknown type byte.
    """
    if ct_type == CT2_AREA:
        return AreaComputer2D()
    if ct_type == CT2_TEST:
        return TestComputer2D()
    if ct_type == CT2_POTENTIAL:
        raise NotImplementedError(
            "CT2_POTENTIAL (PotentialComputer2D) not yet ported"
        )
    if ct_type == CT2_TECHNICAL:
        raise NotImplementedError(
            "CT2_TECHNICAL (TechnicalComputer2D) not yet ported"
        )
    raise ValueError(f"Unknown Computer2DType: {ct_type}")


def load_computer_from_header(data: Any) -> Computer2D:
    """Construct the Computer2D instance encoded in a COMPUTED 2D track header.

    Verifies the leading signature is -11 and dispatches the next 4 bytes
    through :func:`create_computer2d`.  CT2_AREA / CT2_TEST have no extra
    per-instance state so this is just signature + type-byte parsing.
    """
    sig = struct.unpack_from("<i", data, 0)[0]
    if sig != -11:
        raise ValueError(f"Expected COMPUTED signature -11, got {sig}")
    ct_type = read_computer2d_type(data, offset=4)
    return create_computer2d(ct_type)


def recompute_or_cached(
    stored: Rectangle,
    cached_v: float,
    query: Rectangle,
    computer: Computer2D,
    band: DiagonalBand | None,
) -> float:
    """Mirror R's ``Computed_val<T>::val(rect[, band])``.

    Returns the cached value when ``stored == query`` (and the band, if any,
    fully contains the rect); otherwise calls ``computer.compute(query, band)``.
    """
    coords_match = (
        stored.x1 == query.x1
        and stored.x2 == query.x2
        and stored.y1 == query.y1
        and stored.y2 == query.y2
    )
    if band is None or not band.active:
        if coords_match:
            return cached_v
        return computer.compute(query)
    if coords_match and band.do_contain(query):
        return cached_v
    return computer.compute(query, band)
