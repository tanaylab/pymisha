"""Iterator-policy parser for 2D gextract.

Translates the polymorphic `iterator=` arg into a tagged Python object
that the C++ binding can dispatch on. See dev/notes/2026-05-19-spec-a-k-iter-scanner-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class IteratorPolicy:
    """Base class. C++ binding does isinstance dispatch."""


@dataclass(frozen=True)
class FixedRectPolicy(IteratorPolicy):
    width: int
    height: int
    kind: str = "fixed_rect"


@dataclass(frozen=True)
class TrackRectsPolicy(IteratorPolicy):
    track_name: str
    kind: str = "track_rects"


@dataclass(frozen=True)
class IntervalsPolicy(IteratorPolicy):
    """The intervals scope drives iteration (no separate iterator)."""
    kind: str = "intervals"


@dataclass(frozen=True)
class CartesianGridSpec(IteratorPolicy):
    """Streaming CartesianGrid 2D iterator spec.

    intervals1, intervals2: 1D intervals DataFrames (chrom, start, end).
                            intervals2=None means reuse intervals1.
    expansion1, expansion2: sequences of int offsets defining the windows
                            around each interval center. Must contain at
                            least 2 unique values. expansion2=None means
                            reuse expansion1.
    min_band_idx, max_band_idx: optional center-index delta filter; only
                                allowed when intervals2 is None.
    """
    # Use object as the type for DataFrames so the dataclass module stays
    # portable in tests that don't import pandas at decorate time.
    intervals1: object
    expansion1: tuple
    intervals2: object = None
    expansion2: object = None
    min_band_idx: object = None
    max_band_idx: object = None
    kind: str = "cartesian_grid"

    def __post_init__(self):
        # Normalize/validate expansion sequences.
        e1 = self._normalize_expansion(self.expansion1, "expansion1")
        object.__setattr__(self, "expansion1", e1)

        if self.expansion2 is None:
            object.__setattr__(self, "expansion2", e1)
        else:
            e2 = self._normalize_expansion(self.expansion2, "expansion2")
            object.__setattr__(self, "expansion2", e2)

        # Band-idx validation: both or neither, and intervals2 must be None.
        if self.min_band_idx is not None or self.max_band_idx is not None:
            if self.min_band_idx is None or self.max_band_idx is None:
                raise ValueError(
                    "CartesianGridSpec: both min_band_idx and max_band_idx "
                    "must be provided")
            if self.intervals2 is not None:
                raise ValueError(
                    "CartesianGridSpec: min_band_idx/max_band_idx can only "
                    "be used when intervals2 is None")
            if int(self.min_band_idx) > int(self.max_band_idx):
                raise ValueError(
                    "CartesianGridSpec: min_band_idx exceeds max_band_idx")

    @staticmethod
    def _normalize_expansion(value, name) -> tuple:
        try:
            arr = tuple(int(x) for x in value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"CartesianGridSpec: {name} must be a sequence of integers"
            ) from e
        unique_sorted = tuple(sorted(set(arr)))
        if len(unique_sorted) < 2:
            raise ValueError(
                f"CartesianGridSpec: {name} must contain at least 2 unique values"
            )
        return unique_sorted


def _is_int_like(x) -> bool:
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, np.integer))


def parse_iterator_policy(value: object, *, intervals_is_2d: bool) -> IteratorPolicy:
    """Normalize the `iterator=` arg of gextract into a tagged policy.

    Only FixedRect is implemented in this release. Other branches are
    added in releases 2-4.
    """
    # tuple/list of 2 ints -> FixedRect
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                "iterator: a 2D iterator tuple/list must have exactly two "
                f"elements (got {len(value)})"
            )
        if not intervals_is_2d:
            raise ValueError(
                "iterator: a 2D iterator (tuple of bin sizes) requires "
                "2D intervals scope"
            )
        if not (_is_int_like(value[0]) and _is_int_like(value[1])):
            raise ValueError(
                "iterator: FixedRect bin sizes must be integer "
                f"(got {value!r})"
            )
        w, h = int(value[0]), int(value[1])
        if w <= 0 or h <= 0:
            raise ValueError(
                f"iterator: FixedRect bin sizes must be positive (got {w}, {h})"
            )
        return FixedRectPolicy(width=w, height=h)

    # str -> TrackRects (caller is responsible for verifying the track exists
    # and is 2D; parser only normalizes shape).
    if isinstance(value, str):
        if not intervals_is_2d:
            raise ValueError(
                "iterator: a 2D track iterator (str track name) requires "
                "2D intervals scope"
            )
        return TrackRectsPolicy(track_name=value)

    # CartesianGridSpec passes through unchanged.
    if isinstance(value, CartesianGridSpec):
        if not intervals_is_2d:
            raise ValueError(
                "iterator: a CartesianGrid iterator requires 2D intervals scope"
            )
        return value

    raise ValueError(
        f"iterator: unsupported policy {value!r}. "
        "Supported: tuple/list of 2 positive ints (FixedRect), "
        "or str track name (TrackRects)."
    )
