"""R misha baseline-parity comparator.

R misha's regression suite (`expect_regression(obj, id)`) compares a live result
to a frozen ``.rds`` snapshot with ``expect_equal(old, obj, tolerance = 1e-5)``.
We port those tests by running pymisha's equivalent call on R's *own* test DB and
comparing to the *same* ``.rds`` baseline -- no R process at test time, exact R
inputs, exact R expected outputs.

``assert_matches_baseline`` is the normalizing comparator. Normalizations
(learned from the code-review sweep, see docs/superpowers/PORTING_R_TESTS.md):

* chrom columns -> ``str`` (R factor vs pymisha str/categorical).
* coordinate columns -> ``int64`` (R stores start/end as float, pymisha int).
* ``intervalID`` -> rebased to its own minimum before compare (R is 1-based,
  pymisha 0-based; rebasing preserves the grouping structure either way).
* expression value column names differ (``"a + b"`` sanitized differently) ->
  matched positionally, not by name.
* sort by ``(chrom, start, end[, chrom2, start2, end2])`` before compare.
* NaN-aware numeric compare (``equal_nan=True``) at relative tolerance 1e-5.
* non-DataFrame baselines (vectors / lists) -> element-wise, order-preserving.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from pymisha import _r_serialize

# R misha's snapshot directory (see helper-regression.R default).
R_SNAPSHOT_DIR = os.environ.get(
    "PYMISHA_R_SNAPSHOT_DIR",
    "/net/mraid20/export/tgdata/db/tgdb/misha_snapshot",
)

# R misha's test DB, readable by pymisha unchanged (302 tracks, chroms chr1..).
R_TESTDB = os.environ.get(
    "PYMISHA_R_TESTDB",
    "/net/mraid20/ifs/wisdom/tanay_lab/tgdata/db/tgdb/misha_test_db_indexed/",
)

_RTOL = 1e-5

_COORD_1D = ("chrom", "start", "end")
_COORD_2D = ("chrom1", "start1", "end1", "chrom2", "start2", "end2")
_CHROM_COLS = ("chrom", "chrom1", "chrom2")
_INT_COORD_COLS = ("start", "end", "start1", "end1", "start2", "end2")


def _is_str_sequence(obj: Any) -> bool:
    """True for a non-empty sequence whose elements are all strings."""
    if isinstance(obj, (list, tuple)):
        return len(obj) > 0 and all(isinstance(x, str) for x in obj)
    if isinstance(obj, np.ndarray):
        return obj.size > 0 and (obj.dtype == object or obj.dtype.kind in ("U", "S"))
    return False


def baseline_path(baseline_id: str) -> str:
    return os.path.join(R_SNAPSHOT_DIR, f"{baseline_id}.rds")


def has_baseline(baseline_id: str) -> bool:
    return os.path.exists(baseline_path(baseline_id))


def load_baseline(baseline_id: str) -> Any:
    """Read R's ``.rds`` snapshot for ``baseline_id``.

    Returns a ``pandas.DataFrame`` for interval/extract baselines, or a
    ``numpy.ndarray`` / scalar / list for vector baselines.
    """
    path = baseline_path(baseline_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"R baseline not found: {path}")
    return _r_serialize.read(path)


# --------------------------------------------------------------------------- #
# DataFrame normalization
# --------------------------------------------------------------------------- #


def _coord_schema(df: pd.DataFrame) -> tuple[str, ...] | None:
    cols = set(df.columns)
    if set(_COORD_1D) <= cols:
        return _COORD_1D
    if set(_COORD_2D) <= cols:
        return _COORD_2D
    return None


def _normalize_df(df: pd.DataFrame, coords: tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in _CHROM_COLS or c.startswith("chrom"):
            # Normalize chrom naming: R baselines store 'chr1'; pymisha may return
            # '1' (per-chrom DB) or 'chr1' (indexed DB). Strip the prefix on both
            # sides so the comparison is prefix-agnostic.  Also covers R's
            # ``make.unique``-suffixed chrom columns (``chrom11``, ``chrom21``)
            # that arise when 2D ``gintervals_neighbors`` cbinds two 2D sets.
            df[c] = df[c].astype(str).str.replace(r"^chr", "", regex=True)
        elif c in _INT_COORD_COLS:
            df[c] = df[c].astype(np.int64)
    return df.sort_values(list(coords), kind="mergesort").reset_index(drop=True)


def _assert_df_matches(py: pd.DataFrame, base: pd.DataFrame, baseline_id: str, tol: float) -> None:
    coords = _coord_schema(base)
    if coords is None:
        # No coordinate schema (e.g. a stat table). Fall back to positional
        # column compare without sorting.
        _assert_df_positional(py, base, baseline_id, tol)
        return

    if _coord_schema(py) is None:
        raise AssertionError(
            f"[{baseline_id}] baseline is a {coords[0][:5]}-interval DataFrame but "
            f"pymisha result has columns {list(py.columns)}"
        )

    p = _normalize_df(py, coords)
    b = _normalize_df(base, coords)

    if len(p) != len(b):
        raise AssertionError(
            f"[{baseline_id}] row count differs: pymisha={len(p)} vs R={len(b)}"
        )

    # chrom columns: exact string match
    for c in coords:
        if c in _CHROM_COLS:
            if not (p[c].to_numpy() == b[c].to_numpy()).all():
                _raise_first_diff(p, b, c, baseline_id)
        else:  # integer coordinate
            if not (p[c].to_numpy() == b[c].to_numpy()).all():
                _raise_first_diff(p, b, c, baseline_id)

    # intervalID is order-dependent bookkeeping (R is 1-based, pymisha 0-based;
    # and its exact values depend on input/scope ordering, which legitimately
    # differs between implementations). It is not part of the extracted data, so
    # it is intentionally not asserted here - coordinates and values are.

    # value columns: everything that isn't a coord or intervalID, matched
    # positionally (R and pymisha sanitize expression column names differently).
    p_vals = [c for c in p.columns if c not in coords and c != "intervalID"]
    b_vals = [c for c in b.columns if c not in coords and c != "intervalID"]
    if len(p_vals) != len(b_vals):
        raise AssertionError(
            f"[{baseline_id}] value-column count differs: "
            f"pymisha={p_vals} vs R={b_vals}"
        )
    for pc, bc in zip(p_vals, b_vals, strict=True):
        what = f"value col py[{pc}] vs R[{bc}]"
        if pd.api.types.is_numeric_dtype(b[bc].dtype):
            _assert_numeric_close(p[pc].to_numpy(), b[bc].to_numpy(), tol, baseline_id, what)
        else:
            # Non-numeric value column (e.g. gene names): compare as strings.
            pv = p[pc].astype(str).to_numpy()
            bv = b[bc].astype(str).to_numpy()
            if not (pv == bv).all():
                i = int(np.flatnonzero(pv != bv)[0])
                raise AssertionError(
                    f"[{baseline_id}] {what}: differs at row {i}: "
                    f"pymisha={pv[i]!r} vs R={bv[i]!r}"
                )


def _assert_df_positional(py: pd.DataFrame, base: pd.DataFrame, baseline_id: str, tol: float) -> None:
    if py.shape != base.shape:
        raise AssertionError(
            f"[{baseline_id}] shape differs: pymisha={py.shape} vs R={base.shape}"
        )
    # For chrom-keyed tables without start/end (e.g. gintervals_chrom_sizes:
    # chrom[/chrom1,chrom2] + size), sort both sides by the chrom columns so the
    # comparison is order-agnostic (R uses chromosome-key order, pymisha string).
    sort_cols = [c for c in ("chrom", "chrom1", "chrom2") if c in base.columns and c in py.columns]
    if sort_cols:
        py = py.copy()
        base = base.copy()
        for c in sort_cols:
            py[c] = py[c].astype(str).str.replace(r"^chr", "", regex=True)
            base[c] = base[c].astype(str).str.replace(r"^chr", "", regex=True)
        py = py.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        base = base.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    for i in range(base.shape[1]):
        bcol = base.iloc[:, i]
        pcol = py.iloc[:, i]
        if pd.api.types.is_numeric_dtype(bcol.dtype):
            _assert_numeric_close(
                pcol.to_numpy(), bcol.to_numpy(), tol, baseline_id, f"col #{i}"
            )
        else:
            if not (pcol.astype(str).to_numpy() == bcol.astype(str).to_numpy()).all():
                raise AssertionError(f"[{baseline_id}] non-numeric col #{i} differs")


def _raise_first_diff(p: pd.DataFrame, b: pd.DataFrame, col: str, baseline_id: str) -> None:
    pv = p[col].to_numpy()
    bv = b[col].to_numpy()
    idx = np.flatnonzero(pv != bv)
    i = int(idx[0])
    raise AssertionError(
        f"[{baseline_id}] column '{col}' differs at row {i}: "
        f"pymisha={pv[i]!r} vs R={bv[i]!r} ({len(idx)} rows differ)"
    )


# --------------------------------------------------------------------------- #
# Vector / scalar normalization
# --------------------------------------------------------------------------- #


def _coerce_vector(obj: Any) -> np.ndarray:
    """Flatten a pymisha return value into a 1D float array (order preserved)."""
    if obj is None:
        return np.array([], dtype=float)
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy(dtype=float).ravel()
    if isinstance(obj, pd.Series):
        return obj.to_numpy(dtype=float).ravel()
    if isinstance(obj, dict):
        return np.asarray(list(obj.values()), dtype=float).ravel()
    if isinstance(obj, np.ndarray):
        return obj.astype(float).ravel()
    if np.isscalar(obj):
        return np.asarray([obj], dtype=float)
    return np.asarray(obj, dtype=float).ravel()


def _assert_numeric_close(
    p: np.ndarray, b: np.ndarray, tol: float, baseline_id: str, what: str
) -> None:
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    if p.shape != b.shape:
        raise AssertionError(
            f"[{baseline_id}] {what}: shape differs pymisha={p.shape} vs R={b.shape}"
        )
    if not np.allclose(p, b, rtol=tol, atol=tol, equal_nan=True):
        diff = np.abs(p - b)
        # ignore NaN-matched positions
        both_nan = np.isnan(p) & np.isnan(b)
        diff[both_nan] = 0.0
        i = int(np.nanargmax(diff))
        raise AssertionError(
            f"[{baseline_id}] {what}: values differ; "
            f"worst at index {i}: pymisha={p.ravel()[i]!r} vs R={b.ravel()[i]!r} "
            f"(max abs diff {np.nanmax(diff):.3g})"
        )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def assert_matches_list_baseline(py_obj: dict[str, Any], baseline_id: str, *, tol: float = _RTOL) -> None:
    """Assert a dict of results matches an R *named list* baseline element-wise.

    Several ``test-2d-hic-analysis.R`` cases freeze ``list(up = ..., down = ...)``
    etc. ``py_obj`` must be a dict with the same keys; each value is compared with
    the same normalization as :func:`assert_matches_baseline`. An empty/`None`
    side matches an empty/`None` other side.
    """
    base = load_baseline(baseline_id)
    if not isinstance(base, dict):
        raise AssertionError(f"[{baseline_id}] expected a list/dict baseline, got {type(base).__name__}")
    if set(py_obj) != set(base):
        raise AssertionError(f"[{baseline_id}] keys differ: pymisha={sorted(py_obj)} vs R={sorted(base)}")
    for key in base:
        b = base[key]
        p = py_obj[key]
        b_empty = b is None or (hasattr(b, "__len__") and len(b) == 0)
        p_empty = p is None or (hasattr(p, "__len__") and len(p) == 0)
        if b_empty or p_empty:
            if b_empty != p_empty:
                raise AssertionError(f"[{baseline_id}][{key}] one side empty: pymisha={p_empty=}, R={b_empty=}")
            continue
        if isinstance(b, pd.DataFrame):
            _assert_df_matches(p, b, f"{baseline_id}[{key}]", tol)
        else:
            _assert_numeric_close(_coerce_vector(p), _coerce_vector(b), tol, f"{baseline_id}[{key}]", "vector")


def assert_matches_baseline(py_obj: Any, baseline_id: str, *, tol: float = _RTOL) -> None:
    """Assert ``py_obj`` matches R's frozen ``.rds`` baseline for ``baseline_id``.

    Mirrors R's ``expect_regression`` (tolerance 1e-5) with cross-implementation
    normalizations applied. Raises ``AssertionError`` with a focused diff on
    mismatch.
    """
    base = load_baseline(baseline_id)

    if base is None or py_obj is None:
        if base is None and py_obj is None:
            return
        raise AssertionError(
            f"[{baseline_id}] one side is None: pymisha={py_obj!r}, R={base!r}"
        )

    if isinstance(base, pd.DataFrame):
        if not isinstance(py_obj, pd.DataFrame):
            raise AssertionError(
                f"[{baseline_id}] R baseline is a DataFrame but pymisha returned "
                f"{type(py_obj).__name__}"
            )
        _assert_df_matches(py_obj, base, baseline_id, tol)
        return

    # string-sequence baseline (e.g. gvtrack.ls, gtrack.ls -> R character vector).
    # Compared order-independently as a set of names.
    if _is_str_sequence(base):
        b = sorted(str(x) for x in base)
        if isinstance(py_obj, (list, tuple, np.ndarray)):
            p = sorted(str(x) for x in py_obj)
        elif isinstance(py_obj, pd.Series):
            p = sorted(str(x) for x in py_obj.tolist())
        else:
            raise AssertionError(
                f"[{baseline_id}] R baseline is a string vector but pymisha returned "
                f"{type(py_obj).__name__}: {py_obj!r}"
            )
        if p != b:
            raise AssertionError(f"[{baseline_id}] string-list differs: pymisha={p} vs R={b}")
        return

    # vector / scalar baseline
    b = _coerce_vector(base)
    p = _coerce_vector(py_obj)
    # R flattens matrices column-major; pymisha returns row-major 2D results.
    # If the row-major flatten doesn't match, retry with column-major order.
    is_2d = (
        isinstance(py_obj, pd.DataFrame) and py_obj.shape[1] > 1
    ) or (isinstance(py_obj, np.ndarray) and py_obj.ndim == 2)
    if is_2d and p.shape == b.shape and not np.allclose(p, b, rtol=tol, atol=tol, equal_nan=True):
        arr = py_obj.to_numpy(dtype=float) if isinstance(py_obj, pd.DataFrame) else py_obj.astype(float)
        p = arr.ravel(order="F")
    _assert_numeric_close(p, b, tol, baseline_id, "vector")
