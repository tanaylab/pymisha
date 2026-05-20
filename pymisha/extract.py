"""gextract and gscreen implementations."""

from __future__ import annotations

import collections.abc
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ._iterator_policy import FixedRectPolicy

import pandas as pd

from . import _shared
from ._safe_eval import UnsafeExpressionError, compile_safe_expression
from ._shared import (
    CONFIG,
    _bound_colname,
    _checkroot,
    _chunk_slices,
    _config_no_mt,
    _df2pymisha,
    _iterated_intervals,
    _numpy,
    _pandas,
    _preprocess_intervals_iterator,
    _progress_context,
    _pymisha,
    _pymisha2df,
    _remap_interval_ids,
    _track_names_set,
)
from ._types import Iterator
from .expr import _caller_namespace, _expr_safe_name, _parse_expr_vars, _resolve_user_vars
from .vtracks import _compute_vtrack_values


def _scanner_for_intervals_enabled() -> bool:
    """Return True when the env-var opt-in for the intervals scanner path is set."""
    return os.environ.get("PYMISHA_USE_SCANNER_FOR_INTERVALS", "").lower() in (
        "1", "true", "yes", "on"
    )


def _resolve_exprs_for_scanner(exprs: list[str]) -> list[tuple[str, str, object, int, int, int, int]] | None:
    """Try to resolve every expression in ``exprs`` for the C++ 2D scanner.

    Each expression must be either:
    - A bare physical 2D track name (not a vtrack), OR
    - A supported reducing 2D vtrack (see ``_resolve_2d_vtrack_var``).

    Returns a list of ``(physical_track, func, params, ss1, es1, ss2, es2)``
    tuples — one per expression — or ``None`` if any expression cannot be
    routed through the scanner (compound expression, unsupported vtrack,
    1D vtrack, etc.).
    """
    track_names_now = _track_names_set()
    vtrack_names_now = set(_shared._VTRACKS.keys())

    resolved: list[tuple[str, str, object, int, int, int, int]] = []
    for expr in exprs:
        if expr in track_names_now and expr not in vtrack_names_now:
            # Bare physical track: default aggregation is "avg", no shifts.
            resolved.append((expr, "avg", None, 0, 0, 0, 0))
        else:
            r = _resolve_2d_vtrack_var(expr)
            if r is None:
                return None  # compound expr, unsupported vtrack, etc.
            resolved.append(r)
    return resolved


def _resolve_2d_compound_for_scanner(
    exprs: list[str],
) -> tuple[list[tuple[str, dict[str, str]]], list[tuple[str, tuple[str, str, object, int, int, int, int]]]] | None:
    """Resolve compound 2D expressions referencing multiple 2D vtracks/tracks.

    For each user expression, identify every referenced symbol (vtrack name
    or bare 2D physical-track name), resolve each through
    :func:`_resolve_2d_vtrack_var` (vtracks) or the bare-track defaulting
    used by :func:`_resolve_exprs_for_scanner` (bare tracks). The resulting
    set of unique symbols is what the C++ scanner will be asked to compute;
    per-rectangle arithmetic is then evaluated in Python over the returned
    per-symbol arrays.

    Returns
    -------
    None
        If any symbol is unresolvable, refers to a 1D track, an unsupported
        vtrack, or if the expression contains no resolvable symbols at all.
    (eval_specs, var_specs)
        - ``eval_specs`` is a list (one entry per user expression) of
          ``(expr_with_safe_names, safe_to_orig_map)`` tuples. The safe
          expression uses ``__pmv_<hex>`` identifiers for symbols and is
          ready for :func:`compile_safe_expression`.
        - ``var_specs`` is a deduped list of
          ``(safe_name, (track, func, params, ss1, es1, ss2, es2))`` tuples
          driving one C++ scanner var per entry.
    """
    track_names_now = _track_names_set()
    vtrack_names_now = set(_shared._VTRACKS.keys())

    # safe_name -> (track, func, params, ss1, es1, ss2, es2)
    seen_vars: dict[str, tuple[str, str, object, int, int, int, int]] = {}
    eval_specs: list[tuple[str, dict[str, str]]] = []

    for expr in exprs:
        # Parse the expression into safe identifiers, collecting which tracks
        # and vtracks were referenced.
        new_expr, used_tracks, used_vtracks, var_map = _parse_expr_vars(
            expr, track_names_now, vtrack_names_now
        )
        # var_map: __pmv_<hex> -> original-name
        # used_tracks / used_vtracks: original names actually present

        if not used_tracks and not used_vtracks:
            # No resolvable symbols at all (e.g. "1 + 2"). Not for the scanner.
            return None

        for safe_name, orig_name in var_map.items():
            if safe_name in seen_vars:
                continue  # already resolved on a prior expression
            if orig_name in vtrack_names_now:
                r = _resolve_2d_vtrack_var(orig_name)
                if r is None:
                    return None  # 1D source, unsupported func, etc.
                seen_vars[safe_name] = r
            elif orig_name in track_names_now:
                from .tracks import gtrack_info
                try:
                    info = gtrack_info(orig_name)
                except Exception:
                    return None
                if int(info.get("dimensions", 1) or 1) != 2:
                    return None  # 1D bare-track ref in a 2D expression
                seen_vars[safe_name] = (orig_name, "avg", None, 0, 0, 0, 0)
            else:
                return None  # unknown identifier; fall through

        # Validate that the rewritten expression has no stray identifiers
        # beyond our safe names + the always-allowed numeric helpers. Any
        # other identifier likely refers to a caller-namespace variable; in
        # that case let the legacy path handle it.
        try:
            compile_safe_expression(new_expr, set(var_map.keys()))
        except UnsafeExpressionError:
            return None

        eval_specs.append((new_expr, dict(var_map)))

    var_specs = list(seen_vars.items())
    return eval_specs, var_specs


def _group_intervals_by_chrom_pair(
    intervals: pd.DataFrame,
) -> dict[tuple[str, str], list[tuple[int, int, int, int, int]]]:
    """Group 2D intervals by (chrom1, chrom2) using column arrays.

    Returns dict mapping (c1, c2) -> [(interval_idx, s1, e1, s2, e2), ...].
    """
    c1_arr = intervals["chrom1"].astype(str).values
    c2_arr = intervals["chrom2"].astype(str).values
    s1_arr = intervals["start1"].values
    e1_arr = intervals["end1"].values
    s2_arr = intervals["start2"].values
    e2_arr = intervals["end2"].values
    chrom_pair_intervals: dict[tuple[str, str], list[tuple[int, int, int, int, int]]] = {}
    for i in range(len(intervals)):
        key = (c1_arr[i], c2_arr[i])
        if key not in chrom_pair_intervals:
            chrom_pair_intervals[key] = []
        chrom_pair_intervals[key].append((i, int(s1_arr[i]), int(e1_arr[i]), int(s2_arr[i]), int(e2_arr[i])))
    return chrom_pair_intervals


# Functions eligible for C++ inline vtrack evaluation (no filter required)
_CPP_SEQ_FUNCS = {
    "pwm",
    "pwm.max",
    "pwm.max.pos",
    "pwm.count",
    "pwm.edit_distance",
    "pwm.edit_distance.pos",
    "pwm.max.edit_distance",
    "pwm.edit_distance.lse",
    "pwm.edit_distance.lse.pos",
    "kmer.count",
    "kmer.frac",
    "masked.count",
    "masked.frac",
}
_CPP_VALUE_FUNCS = {
    "avg",
    "mean",
    "sum",
    "min",
    "max",
    "first",
    "last",
    "size",
    "exists",
    "stddev",
    "std",
    "quantile",
    "sample",
    "nearest",
    "lse",
    "first.pos.abs",
    "first.pos.relative",
    "last.pos.abs",
    "last.pos.relative",
    "min.pos.abs",
    "min.pos.relative",
    "max.pos.abs",
    "max.pos.relative",
    "sample.pos.abs",
    "sample.pos.relative",
}


def _can_vtracks_use_cpp(vtrack_names: set[str] | list[str]) -> bool:
    """Check whether all listed vtracks can be evaluated in the C++ scanner.

    Returns True if every vtrack is eligible for inline C++ evaluation.
    A vtrack is eligible when:
      - It has no filter attached.
      - Its source is None (sequence-based) or a string (physical track name).
      - Its function is in the supported set.
    """
    for name in vtrack_names:
        cfg = _shared._VTRACKS.get(name)
        if cfg is None:
            return False
        # Array-slice vtracks must go through Python
        if cfg.get("kind") == "array_slice":
            return False
        # Filter vtracks must go through Python
        filt = cfg.get("filter")
        if filt is not None and not (isinstance(filt, _pandas.DataFrame) and len(filt) == 0):
            return False
        src = cfg.get("src")
        func = str(cfg.get("func", "avg")).lower()
        if src is None:
            # Sequence-based
            if func not in _CPP_SEQ_FUNCS:
                return False
        elif isinstance(src, str):
            # Physical track source
            if func not in _CPP_VALUE_FUNCS:
                return False
        elif isinstance(src, _pandas.DataFrame):
            # DataFrame source — must go to Python for now
            return False
        else:
            return False
    return True


def _build_vtracks_dict(vtrack_names: set[str] | list[str]) -> dict[str, dict[str, Any]]:
    """Build a Python dict of vtrack specs to pass to C++.

    The dict maps vtrack_name -> spec dict.  PSSM DataFrames are converted
    to numpy arrays so the C++ side can parse them.
    """
    result = {}
    for name in vtrack_names:
        cfg = _shared._VTRACKS.get(name)
        if cfg is None:
            continue
        spec = dict(cfg)
        # Ensure pssm is a numpy array
        pssm = spec.get("pssm")
        if pssm is not None and isinstance(pssm, _pandas.DataFrame):
            spec["pssm"] = pssm.to_numpy(dtype=float, copy=False)
        # Strip filter-related keys (shouldn't be present but be safe)
        spec.pop("filter", None)
        spec.pop("filter_key", None)
        spec.pop("filter_stats", None)
        # Strip DataFrame src (shouldn't happen on this path)
        src = spec.get("src")
        if isinstance(src, _pandas.DataFrame):
            spec.pop("src_df", None)
        result[name] = spec
    return result


def _is_2d_intervals(intervals: pd.DataFrame | Any) -> bool:
    """Check if intervals DataFrame has 2D columns."""
    return isinstance(intervals, _pandas.DataFrame) and "chrom1" in intervals.columns


def _maybe_load_intervals_set(intervals: pd.DataFrame | str) -> pd.DataFrame | str:
    """Transparently load a named interval set (including bigsets).

    If *intervals* is a string, attempt to load it via
    :func:`gintervals_load`.  Returns the loaded DataFrame on success, or
    the original string if the name does not correspond to a saved interval
    set (so that downstream code can produce its own error message).

    Non-string values are returned unchanged.
    """
    if not isinstance(intervals, str):
        return intervals

    from .intervals import gintervals_exists, gintervals_load

    if gintervals_exists(intervals):
        return gintervals_load(intervals)
    return intervals


def _find_2d_track_file(track_path: str, c1: str, c2: str) -> str | None:
    """Find a 2D track per-chrom-pair file, trying multiple naming conventions."""
    # Try: c1-c2 (pymisha convention)
    path = os.path.join(track_path, f"{c1}-{c2}")
    if os.path.exists(path):
        return path
    # Try: chrc1-chrc2 (R misha convention)
    path = os.path.join(track_path, f"chr{c1}-chr{c2}")
    if os.path.exists(path):
        return path
    return None


def _validate_band(band: tuple[int, int] | tuple[float, float] | list[int] | None) -> tuple[int, int] | None:
    """Validate band parameter. Returns (d1, d2) tuple or None."""
    if band is None:
        return None
    if not hasattr(band, "__len__") or len(band) != 2:
        raise ValueError("band must be a sequence of length 2: (d1, d2)")
    d1, d2 = int(band[0]), int(band[1])
    if d1 >= d2:
        raise ValueError(f"band d1 ({d1}) must be less than d2 ({d2})")
    return (d1, d2)


def _obj_in_band(obj: tuple[int, ...], is_points: bool, band: tuple[int, int]) -> bool:
    """Check if a 2D object intersects a diagonal band (d1, d2).

    Band condition: d1 <= (x - y) < d2 for any point in the object.
    For RECTS (x1, y1, x2, y2): intersects if x2 - y1 > d1 AND x1 - y2 + 1 < d2
    For POINTS (x, y): d1 <= x - y < d2
    """
    d1, d2 = band
    if is_points:
        ox, oy, _ = obj
        delta = ox - oy
        return d1 <= delta < d2
    ox1, oy1, ox2, oy2, _ = obj
    return (ox2 - oy1 > d1) and (ox1 - oy2 + 1 < d2)


def _gextract_2d_single_python(
    track: str, col_name: str, intervals: pd.DataFrame, band: tuple[int, int] | None
) -> pd.DataFrame | None:
    """Pure-Python reference implementation of 2D bare-track extract.

    Kept for parity tests against the C++ fast path. Not used in production.
    """
    from ._quadtree import open_2d_pair, query_2d_track_opened
    from .tracks import gtrack_info

    track_path = _pymisha.pm_track_path(track)
    info = gtrack_info(track)
    is_points = info.get("type") == "points"

    # Group intervals by (chrom1, chrom2) to open each file only once.
    chrom_pair_intervals = _group_intervals_by_chrom_pair(intervals)

    rows = []
    for (c1, c2), interval_list in chrom_pair_intervals.items():
        pair = open_2d_pair(track_path, c1, c2)
        if pair is None:
            continue

        file_is_points, num_objs, data, root_chunk_fpos, close_fn = pair
        try:
            if num_objs == 0:
                continue

            for interval_idx, s1, e1, s2, e2 in interval_list:
                objs = query_2d_track_opened(
                    data,
                    file_is_points,
                    num_objs,
                    root_chunk_fpos,
                    s1,
                    s2,
                    e1,
                    e2,
                    band=band,
                )
                for obj in objs:
                    if is_points:
                        ox, oy, val = obj
                        rows.append((c1, ox, ox + 1, c2, oy, oy + 1, float(val), interval_idx))
                    else:
                        ox1, oy1, ox2, oy2, val = obj
                        rows.append((c1, ox1, ox2, c2, oy1, oy2, float(val), interval_idx))
        finally:
            close_fn()

    if not rows:
        return None

    result = _pandas.DataFrame(
        rows,
        columns=[
            "chrom1",
            "start1",
            "end1",
            "chrom2",
            "start2",
            "end2",
            col_name,
            "intervalID",
        ],
    )
    return result.sort_values(["chrom1", "start1", "chrom2", "start2", "intervalID"]).reset_index(drop=True)


def _gextract_2d_single(
    track: str, col_name: str, intervals: pd.DataFrame, band: tuple[int, int] | None
) -> pd.DataFrame | None:
    """Extract objects from a 2D track via the native C++ binding.

    Returns a DataFrame with chrom1/start1/end1/chrom2/start2/end2/<col_name>/
    intervalID columns, sorted by (chrom1, start1, chrom2, start2, intervalID).
    Returns None if no objects intersect any interval.
    """
    from .intervals import _chrom_id_map

    n = len(intervals)
    if n == 0:
        return None

    cmap = _chrom_id_map()
    chrom1_ids = (
        _pandas.Series(intervals["chrom1"].astype(str).to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )
    chrom2_ids = (
        _pandas.Series(intervals["chrom2"].astype(str).to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )

    intervals_dict = {
        "chrom1": chrom1_ids,
        "start1": intervals["start1"].to_numpy(dtype=_numpy.int64),
        "end1":   intervals["end1"].to_numpy(dtype=_numpy.int64),
        "chrom2": chrom2_ids,
        "start2": intervals["start2"].to_numpy(dtype=_numpy.int64),
        "end2":   intervals["end2"].to_numpy(dtype=_numpy.int64),
    }

    result = _pymisha.pm_extract_2d(track, intervals_dict, band)
    if result is None:
        return None

    # Vectorized reverse chromid -> name lookup for output columns.
    id2name = {v: k for k, v in cmap.items()}
    c1_series = _pandas.Series(result["chrom1"]).astype("int64")
    c2_series = _pandas.Series(result["chrom2"]).astype("int64")
    chrom1_names = c1_series.map(id2name).fillna(c1_series.astype(str)).to_numpy()
    chrom2_names = c2_series.map(id2name).fillna(c2_series.astype(str)).to_numpy()

    out = _pandas.DataFrame({
        "chrom1": chrom1_names,
        "start1": result["start1"],
        "end1":   result["end1"],
        "chrom2": chrom2_names,
        "start2": result["start2"],
        "end2":   result["end2"],
        col_name: result["value"],
        "intervalID": result["intervalID"],
    })
    return out.sort_values(
        ["chrom1", "start1", "chrom2", "start2", "intervalID"]
    ).reset_index(drop=True)


_2D_VTRACK_FUNCS = {
    "avg",
    "mean",
    "area",
    "weighted.sum",
    "min",
    "max",
    "exists",
    "size",
    "first",
    "last",
    "sample",
    "global.percentile",
}
_2D_AGG_FUNCS = {"area", "weighted.sum", "min", "max", "avg"}
_2D_OBJECT_FUNCS = {"exists", "size", "first", "last", "sample"}
_2D_PERCENTILE_FUNCS = {"global.percentile"}

# Funcs supported by PMTrackExpression2DVars::add_var (C++ scanner).
# Verified by reading src/PMTrackExpression2DVars.cpp::parse_func.
# "mean" is an alias for "avg" and is normalised below.
# Object-level funcs (exists/size/first/last/sample) use query_objects per cell
# inside set_vars_batch; global.percentile is deferred.
_SCANNER_2D_FUNCS = {
    "area", "weighted.sum", "min", "max", "avg", "mean",
    "exists", "size", "first", "last", "sample",
}


def _resolve_2d_vtrack_var(expr: str) -> tuple[str, str, object, int, int, int, int] | None:
    """Resolve a 2D vtrack expression to (physical_track, func, params, ss1, es1, ss2, es2).

    Returns None when the expression is not a vtrack, or is a vtrack that
    cannot be routed through the C++ scanner in this release (unsupported func,
    compound source, non-string source, dim-projected 1D vtrack, or source
    that is not a 2D track).

    Supported: single-source 2D reducing vtracks with func in
    ``_SCANNER_2D_FUNCS`` (area/weighted.sum/min/max/avg + exists/size/first/last/sample),
    with optional per-var 2D shifts (sshift1/eshift1/sshift2/eshift2).
    ``global.percentile`` is not scanner-supported and returns None.
    """
    cfg = _shared._VTRACKS.get(expr)
    if cfg is None:
        return None  # not a vtrack at all

    # Dim-projected vtracks (dim=1 or dim=2) are 1D tracks viewed in 2D context.
    # They are not backed by a 2D source track and cannot use the 2D scanner.
    if cfg.get("dim") is not None:
        return None

    func = str(cfg.get("func", "avg")).lower()
    # Normalise "mean" -> "avg" to match C++ scanner expectations.
    if func == "mean":
        func = "avg"

    if func not in _SCANNER_2D_FUNCS:
        return None  # global.percentile not scanner-supported (deferred)

    src = cfg.get("src")
    if not isinstance(src, str):
        return None  # DataFrame source or compound — not a single named 2D track

    # Verify the source is actually a 2D track (not 1D dense/fixedbin/etc.).
    from .tracks import gtrack_info

    try:
        info = gtrack_info(src)
    except Exception:
        return None
    if int(info.get("dimensions", 1) or 1) != 2:
        return None  # 1D source track; can't use 2D scanner

    params = cfg.get("params")
    sshift1 = int(cfg.get("sshift1", 0) or 0)
    eshift1 = int(cfg.get("eshift1", 0) or 0)
    sshift2 = int(cfg.get("sshift2", 0) or 0)
    eshift2 = int(cfg.get("eshift2", 0) or 0)
    return (src, func, params, sshift1, eshift1, sshift2, eshift2)


def _resolve_2d_vtrack_source(vtrack_name: str) -> tuple[str, dict[str, int], str]:
    """Resolve a 2D-capable virtual track to its backing 2D physical track.

    Returns
    -------
    tuple of (str, dict, str)
        The physical track name, a dict with 2D shift values
        ``{"sshift1": int, "eshift1": int, "sshift2": int, "eshift2": int}``,
        and the aggregation function name (``"mean"`` is normalized to ``"avg"``).
    """
    from .tracks import gtrack_info

    cfg = _shared._VTRACKS.get(vtrack_name)
    if cfg is None:
        raise ValueError(f"Unknown virtual track '{vtrack_name}'")

    src = cfg.get("src")
    if not isinstance(src, str):
        raise ValueError(f"2D extraction for virtual track '{vtrack_name}' requires a physical 2D track source")

    info = gtrack_info(src)
    if int(info.get("dimensions", 1) or 1) != 2:
        raise ValueError(f"Virtual track '{vtrack_name}' does not reference a 2D track source")

    func = str(cfg.get("func", "avg")).lower()
    params = cfg.get("params")
    if func not in _2D_VTRACK_FUNCS:
        raise ValueError(
            f"2D extraction for virtual track '{vtrack_name}': "
            f"unsupported function '{func}' (supported: {sorted(_2D_VTRACK_FUNCS)})"
        )
    if params is not None:
        raise ValueError(f"2D extraction for virtual track '{vtrack_name}' does not support params")
    if int(cfg.get("sshift", 0) or 0) != 0 or int(cfg.get("eshift", 0) or 0) != 0:
        raise ValueError(f"2D extraction for virtual track '{vtrack_name}' does not support 1D iterator shifts")
    if cfg.get("filter") is not None:
        raise ValueError(f"2D extraction for virtual track '{vtrack_name}' does not support filters")

    # Normalize "mean" → "avg"
    if func == "mean":
        func = "avg"

    shifts = {
        "sshift1": int(cfg.get("sshift1", 0) or 0),
        "eshift1": int(cfg.get("eshift1", 0) or 0),
        "sshift2": int(cfg.get("sshift2", 0) or 0),
        "eshift2": int(cfg.get("eshift2", 0) or 0),
    }
    return src, shifts, func


def _maybe_load_2d_intervals_set(
    intervals: pd.DataFrame | str,
    exprs: list[str],
    iterator: Iterator | str,
    band: tuple[int, int] | tuple[float, float] | None,
) -> pd.DataFrame | str:
    """Load named interval sets only when we likely need a 2D scope."""
    if not isinstance(intervals, str):
        return intervals

    should_try = band is not None
    if not should_try and isinstance(iterator, str):
        from .tracks import gtrack_exists, gtrack_info

        if gtrack_exists(iterator):
            info = gtrack_info(iterator)
            should_try = int(info.get("dimensions", 1) or 1) == 2

    if not should_try:
        return intervals

    from .intervals import gintervals_load

    try:
        loaded = gintervals_load(intervals)
    except Exception:
        return intervals
    if _is_2d_intervals(loaded):
        return loaded
    return intervals


def _apply_2d_shifts(intervals: pd.DataFrame, sshift1: int, eshift1: int, sshift2: int, eshift2: int) -> pd.DataFrame:
    """Apply 2D iterator shifts to interval coordinates."""
    if sshift1 == 0 and eshift1 == 0 and sshift2 == 0 and eshift2 == 0:
        return intervals
    shifted = intervals.copy()
    shifted["start1"] = shifted["start1"] + sshift1
    shifted["end1"] = shifted["end1"] + eshift1
    shifted["start2"] = shifted["start2"] + sshift2
    shifted["end2"] = shifted["end2"] + eshift2
    return shifted


def _gextract_2d_vtrack_agg(
    track: str, col_name: str, intervals: pd.DataFrame, band: tuple[int, int] | None, func: str
) -> pd.DataFrame:
    """Extract aggregated stats from a 2D track for 2D intervals.

    Returns one row per query interval with the aggregated value.

    Parameters
    ----------
    track : str
        Physical 2D track name.
    col_name : str
        Column name for the aggregated value in the output DataFrame.
    intervals : DataFrame
        2D intervals with chrom1/start1/end1/chrom2/start2/end2 columns.
    band : tuple of (int, int) or None
        Diagonal band filter ``(d1, d2)``.
    func : str
        Aggregation function: ``"area"``, ``"weighted.sum"``, ``"min"``,
        ``"max"``, or ``"avg"``.

    Returns
    -------
    DataFrame
        One row per query interval with columns: chrom1, start1, end1,
        chrom2, start2, end2, <col_name>, intervalID.
    """
    from ._quadtree import open_2d_pair, query_2d_track_stats_batch

    track_path = _pymisha.pm_track_path(track)

    n = len(intervals)
    values = _numpy.full(n, _numpy.nan, dtype=float)

    # Group intervals by (chrom1, chrom2) to open each file only once.
    chrom_pair_intervals = _group_intervals_by_chrom_pair(intervals)

    for (c1, c2), interval_list in chrom_pair_intervals.items():
        pair = open_2d_pair(track_path, c1, c2)
        if pair is None:
            # No data for this chrom pair — values stay NaN.
            continue

        file_is_points, num_objs, data, root_chunk_fpos, close_fn = pair
        try:
            if num_objs == 0:
                continue

            # Build batch query rectangles: (N, 4) int64 array
            # Query rect coords: (s1, s2, e1, e2) maps to (qx1, qy1, qx2, qy2)
            m = len(interval_list)
            rects = _numpy.empty((m, 4), dtype=_numpy.int64)
            indices = _numpy.empty(m, dtype=int)
            for j, (interval_idx, s1, e1, s2, e2) in enumerate(interval_list):
                rects[j, 0] = s1
                rects[j, 1] = s2
                rects[j, 2] = e1
                rects[j, 3] = e2
                indices[j] = interval_idx

            batch = query_2d_track_stats_batch(
                data,
                file_is_points,
                num_objs,
                root_chunk_fpos,
                rects,
                band=band,
            )

            occ = batch["occupied_area"]
            for j in range(m):
                if occ[j] == 0:
                    continue
                idx = indices[j]
                if func == "area":
                    values[idx] = float(occ[j])
                elif func == "weighted.sum":
                    values[idx] = float(batch["weighted_sum"][j])
                elif func == "min":
                    values[idx] = float(batch["min_val"][j])
                elif func == "max":
                    values[idx] = float(batch["max_val"][j])
                elif func == "avg":
                    values[idx] = float(batch["weighted_sum"][j]) / float(occ[j])
        finally:
            close_fn()

    return _pandas.DataFrame(
        {
            "chrom1": intervals["chrom1"].to_numpy(),
            "start1": intervals["start1"].values,
            "end1": intervals["end1"].values,
            "chrom2": intervals["chrom2"].to_numpy(),
            "start2": intervals["start2"].values,
            "end2": intervals["end2"].values,
            col_name: values,
            "intervalID": _numpy.arange(n, dtype=int),
        }
    )


def _gextract_2d_vtrack_objects_python(
    track: str, col_name: str, intervals: pd.DataFrame, band: tuple[int, int] | None, func: str
) -> pd.DataFrame:
    """Pure-Python reference implementation of 2D object-level reduction.

    Kept for parity tests against the C++ fast path. Not used in production.
    """
    import random

    from ._quadtree import open_2d_pair, query_2d_track_opened
    from .tracks import gtrack_info

    track_path = _pymisha.pm_track_path(track)
    info = gtrack_info(track)
    is_points = info.get("type") == "points"

    n = len(intervals)
    # exists and size default to 0 (no objects = definite answer),
    # while first/last/sample default to NaN (no objects = undefined).
    values = _numpy.zeros(n, dtype=float) if func in ("exists", "size") else _numpy.full(n, _numpy.nan, dtype=float)

    # Group intervals by (chrom1, chrom2) to open each file only once.
    chrom_pair_intervals = _group_intervals_by_chrom_pair(intervals)

    for (c1, c2), interval_list in chrom_pair_intervals.items():
        pair = open_2d_pair(track_path, c1, c2)
        if pair is None:
            continue

        file_is_points, num_objs, data, root_chunk_fpos, close_fn = pair
        try:
            if num_objs == 0:
                continue

            for interval_idx, s1, e1, s2, e2 in interval_list:
                objs = query_2d_track_opened(
                    data,
                    file_is_points,
                    num_objs,
                    root_chunk_fpos,
                    s1,
                    s2,
                    e1,
                    e2,
                    band=band,
                )

                if func == "exists":
                    values[interval_idx] = 1.0 if len(objs) > 0 else 0.0
                elif func == "size":
                    values[interval_idx] = float(len(objs))
                elif len(objs) == 0:
                    # first, last, sample: NaN when no objects
                    pass
                elif func == "first":
                    val = objs[0][2] if is_points else objs[0][4]
                    values[interval_idx] = float(val)
                elif func == "last":
                    val = objs[-1][2] if is_points else objs[-1][4]
                    values[interval_idx] = float(val)
                elif func == "sample":
                    chosen = random.choice(objs)
                    val = chosen[2] if is_points else chosen[4]
                    values[interval_idx] = float(val)
        finally:
            close_fn()

    return _pandas.DataFrame(
        {
            "chrom1": intervals["chrom1"].to_numpy(),
            "start1": intervals["start1"].values,
            "end1": intervals["end1"].values,
            "chrom2": intervals["chrom2"].to_numpy(),
            "start2": intervals["start2"].values,
            "end2": intervals["end2"].values,
            col_name: values,
            "intervalID": _numpy.arange(n, dtype=int),
        }
    )


def _gextract_2d_vtrack_objects(
    track: str, col_name: str, intervals: pd.DataFrame, band: tuple[int, int] | None, func: str
) -> pd.DataFrame:
    """Reduce a 2D track to a per-interval scalar via object-level funcs.

    Calls the native C++ binding ``pm_extract_2d_objects``. Returns one row
    per input interval with the computed value. Defaults: exists/size -> 0.0,
    first/last/sample -> NaN.
    """
    from .intervals import _chrom_id_map

    if func not in _2D_OBJECT_FUNCS:
        raise ValueError(
            f"_gextract_2d_vtrack_objects: unknown func '{func}' "
            f"(expected one of {sorted(_2D_OBJECT_FUNCS)})"
        )

    n = len(intervals)
    if n == 0:
        return _pandas.DataFrame(
            {
                "chrom1": intervals["chrom1"].to_numpy(),
                "start1": intervals["start1"].values,
                "end1": intervals["end1"].values,
                "chrom2": intervals["chrom2"].to_numpy(),
                "start2": intervals["start2"].values,
                "end2": intervals["end2"].values,
                col_name: _numpy.empty(0, dtype=float),
                "intervalID": _numpy.empty(0, dtype=int),
            }
        )

    cmap = _chrom_id_map()
    chrom1_ids = (
        _pandas.Series(intervals["chrom1"].astype(str).to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )
    chrom2_ids = (
        _pandas.Series(intervals["chrom2"].astype(str).to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )

    intervals_dict = {
        "chrom1": chrom1_ids,
        "start1": intervals["start1"].to_numpy(dtype=_numpy.int64),
        "end1": intervals["end1"].to_numpy(dtype=_numpy.int64),
        "chrom2": chrom2_ids,
        "start2": intervals["start2"].to_numpy(dtype=_numpy.int64),
        "end2": intervals["end2"].to_numpy(dtype=_numpy.int64),
    }

    result = _pymisha.pm_extract_2d_objects(track, intervals_dict, band, func, 60427)

    return _pandas.DataFrame(
        {
            "chrom1": intervals["chrom1"].to_numpy(),
            "start1": intervals["start1"].values,
            "end1": intervals["end1"].values,
            "chrom2": intervals["chrom2"].to_numpy(),
            "start2": intervals["start2"].values,
            "end2": intervals["end2"].values,
            col_name: result["value"],
            "intervalID": _numpy.arange(n, dtype=int),
        }
    )


def _gextract_2d_vtrack_global_percentile(
    track: str, col_name: str, intervals: pd.DataFrame, band: tuple[int, int] | None
) -> pd.DataFrame:
    """Extract global percentile ranks from a 2D track for 2D intervals.

    Two-pass approach:
    1. Extract aggregated values (avg) for each query interval.
    2. Compute percentile rank of each value relative to the global
       distribution of all non-NaN values.

    Returns one row per query interval with the percentile rank (0-1).

    Parameters
    ----------
    track : str
        Physical 2D track name.
    col_name : str
        Column name for the percentile value in the output DataFrame.
    intervals : DataFrame
        2D intervals with chrom1/start1/end1/chrom2/start2/end2 columns.
    band : tuple of (int, int) or None
        Diagonal band filter ``(d1, d2)``.

    Returns
    -------
    DataFrame
        One row per query interval.
    """
    # Pass 1: get raw aggregated values (avg = weighted_sum / area) for each interval.
    agg_df = _gextract_2d_vtrack_agg(track, col_name, intervals, band, "avg")
    raw_values = agg_df[col_name].to_numpy(dtype=float, copy=False)

    # Pass 2: compute percentile rank among all non-NaN values.
    n = len(raw_values)
    result_values = _numpy.full(n, _numpy.nan, dtype=float)
    valid_mask = ~_numpy.isnan(raw_values)
    valid_vals = raw_values[valid_mask]

    if len(valid_vals) > 0:
        # For each valid value, percentile = fraction of valid values that are
        # strictly less than this value.
        sorted_vals = _numpy.sort(valid_vals)
        for i in range(n):
            if valid_mask[i]:
                v = raw_values[i]
                # Number of values strictly less than v.
                n_less = int(_numpy.searchsorted(sorted_vals, v, side="left"))
                result_values[i] = n_less / len(sorted_vals)

    agg_df[col_name] = result_values
    return agg_df


def _gextract_2d_via_scanner(
    exprs: list[str],
    intervals: pd.DataFrame,
    policy: FixedRectPolicy | Any,
    *,
    colnames: list[str] | None,
    band: tuple[int, int] | None,
    resolved_vars: list[tuple[str, str, object, int, int, int, int]] | None = None,
) -> pd.DataFrame:
    """Run a 2D extract through pm_extract_2d_scanner (FixedRect, TrackRects, or CartesianGrid).

    Supports bare 2D track names as expressions and, when ``resolved_vars`` is
    provided, reducing 2D vtracks.  Each bare track defaults to "avg".

    Parameters
    ----------
    resolved_vars : list of (physical_track, func, params, ss1, es1, ss2, es2) or None
        When provided, one entry per expression.  The physical_track, func, and
        shifts are passed directly to the C++ scanner; params is currently unused.
        When None, all ``exprs`` are treated as bare physical track names with
        "avg" aggregation and zero shifts.

    Returns a DataFrame with chrom1, start1, end1, chrom2, start2, end2,
    <colname>, intervalID columns - matching the shape returned by the
    regular _gextract_2d path.
    """
    from ._iterator_policy import CartesianGridSpec, FixedRectPolicy, IntervalsPolicy, TrackRectsPolicy
    from .intervals import _chrom_id_map

    exprs_list = list(exprs)

    # Build (track, func, ss1, es1, ss2, es2) tuples for the C++ scanner.
    if resolved_vars is not None:
        # Caller supplied (physical_track, func, params, ss1, es1, ss2, es2).
        vars_list = [
            (track, func, ss1, es1, ss2, es2)
            for (track, func, _params, ss1, es1, ss2, es2) in resolved_vars
        ]
    else:
        # Bare physical track names: default aggregation is "avg", no shifts.
        vars_list = [(expr, "avg", 0, 0, 0, 0) for expr in exprs_list]

    if colnames is None:
        colnames_list = list(exprs_list)
    elif isinstance(colnames, str):
        colnames_list = [colnames]
    else:
        colnames_list = list(colnames)

    # Build the policy dict for the C++ binding.
    policy_dict: dict[str, Any]
    if isinstance(policy, IntervalsPolicy):
        policy_dict = {"kind": "intervals"}
    elif isinstance(policy, TrackRectsPolicy):
        policy_dict = {
            "kind": "track_rects",
            "track_name": policy.track_name,
        }
    elif isinstance(policy, FixedRectPolicy):
        policy_dict = {
            "kind": "fixed_rect",
            "width": int(policy.width),
            "height": int(policy.height),
        }
    elif isinstance(policy, CartesianGridSpec):
        cmap = _chrom_id_map()

        def _df_to_intervals_dict(df: pd.DataFrame | None) -> dict[str, Any] | None:
            if df is None:
                return None
            chrom_strs = df["chrom"].astype(str).tolist()
            chromids = _numpy.array(
                [cmap.get(c, -1) for c in chrom_strs], dtype=_numpy.int32
            )
            return {
                "chrom": chromids,
                "start": df["start"].to_numpy(dtype=_numpy.int64),
                "end":   df["end"].to_numpy(dtype=_numpy.int64),
            }

        # `expansion2` is typed `object` on the dataclass; `__post_init__`
        # always normalizes it to a tuple (mirroring `expansion1`), so it is
        # safe to feed straight to numpy here.
        expansion2_tuple = cast(tuple, policy.expansion2)
        policy_dict = {
            "kind":         "cartesian_grid",
            "intervals1":   _df_to_intervals_dict(cast("pd.DataFrame | None", policy.intervals1)),
            "expansion1":   _numpy.array(list(policy.expansion1), dtype=_numpy.int64),
            "intervals2":   _df_to_intervals_dict(cast("pd.DataFrame | None", policy.intervals2)),
            "expansion2":   (
                _numpy.array(list(expansion2_tuple), dtype=_numpy.int64)
                if policy.intervals2 is not None
                else None
            ),
            "min_band_idx": policy.min_band_idx,
            "max_band_idx": policy.max_band_idx,
        }
    else:
        raise TypeError(f"Unsupported iterator policy: {type(policy)!r}")

    # Convert chrom name strings to chromid integers for the scope dict.
    cmap = _chrom_id_map()
    chrom1_ids = (
        _pandas.Series(intervals["chrom1"].astype(str).to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )
    chrom2_ids = (
        _pandas.Series(intervals["chrom2"].astype(str).to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )
    scope_dict = {
        "chrom1": chrom1_ids,
        "start1": intervals["start1"].to_numpy(dtype=_numpy.int64),
        "end1":   intervals["end1"].to_numpy(dtype=_numpy.int64),
        "chrom2": chrom2_ids,
        "start2": intervals["start2"].to_numpy(dtype=_numpy.int64),
        "end2":   intervals["end2"].to_numpy(dtype=_numpy.int64),
    }

    band_arg = (int(band[0]), int(band[1])) if band is not None else None

    result = _pymisha.pm_extract_2d_scanner(
        policy_dict, scope_dict, vars_list, colnames_list, band_arg
    )

    # Convert chromid integers back to chrom name strings.
    id2name = {v: k for k, v in cmap.items()}
    c1_arr = _numpy.asarray(result["_chrom1"], dtype=_numpy.int64)
    c2_arr = _numpy.asarray(result["_chrom2"], dtype=_numpy.int64)
    c1_series = _pandas.Series(c1_arr).astype("int64")
    chrom1_names = c1_series.map(id2name).fillna(c1_series.astype(str)).to_numpy()
    c2_series = _pandas.Series(c2_arr).astype("int64")
    chrom2_names = c2_series.map(id2name).fillna(c2_series.astype(str)).to_numpy()

    n = len(chrom1_names)
    out = _pandas.DataFrame({
        "chrom1": chrom1_names,
        "start1": result["_start1"],
        "end1":   result["_end1"],
        "chrom2": chrom2_names,
        "start2": result["_start2"],
        "end2":   result["_end2"],
    })
    for name in colnames_list:
        out[name] = result[name]
    out["intervalID"] = _numpy.arange(n, dtype=int)
    return out.sort_values(["chrom1", "start1", "chrom2", "start2", "intervalID"]).reset_index(drop=True)


def _gextract_2d_compound_via_scanner(
    exprs: list[str],
    intervals: pd.DataFrame,
    policy: object,
    *,
    colnames: list[str] | None,
    band: tuple[int, int] | None,
    eval_specs: list[tuple[str, dict[str, str]]],
    var_specs: list[tuple[str, tuple[str, str, object, int, int, int, int]]],
) -> pd.DataFrame:
    """Run compound 2D expressions through the C++ scanner.

    Asks the scanner for one column per unique symbol (under safe internal
    names), then evaluates each user expression over the resulting per-symbol
    arrays. ``eval_specs`` carries the rewritten expressions and the safe-name
    map produced by :func:`_resolve_2d_compound_for_scanner`; ``var_specs``
    drives the scanner with the deduped var list.
    """
    safe_names = [safe for safe, _ in var_specs]
    resolved_vars = [vt for _, vt in var_specs]

    scanner_out = _gextract_2d_via_scanner(
        # Use safe names so the binding does not collide with user-facing names.
        safe_names, intervals, policy,
        colnames=safe_names, band=band, resolved_vars=resolved_vars,
    )

    if colnames is None:
        out_names = list(exprs)
    elif isinstance(colnames, str):
        out_names = [colnames]
    else:
        out_names = list(colnames)
    if len(out_names) != len(exprs):
        raise ValueError(
            f"colnames length ({len(out_names)}) does not match exprs length ({len(exprs)})"
        )

    out = scanner_out[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
    for (safe_expr, var_map), out_name in zip(eval_specs, out_names, strict=True):
        ns: dict[str, Any] = {
            "np": _numpy, "numpy": _numpy,
            "abs": abs, "min": min, "max": max,
            "round": round, "float": float, "int": int, "bool": bool,
        }
        for safe in var_map:
            ns[safe] = scanner_out[safe].to_numpy()
        try:
            code = compile_safe_expression(safe_expr, set(var_map.keys()))
        except UnsafeExpressionError as exc:
            raise ValueError(
                f"2D compound expression {ns_orig(var_map, safe_expr)!r} is not safe to evaluate: {exc}"
            ) from exc
        out[out_name] = eval(code, ns)  # noqa: S307 - compiled+validated AST
    out["intervalID"] = scanner_out["intervalID"].to_numpy()
    return out


def ns_orig(var_map: dict[str, str], safe_expr: str) -> str:
    """Best-effort restoration of original symbol names in a safe expression for error messages."""
    s = safe_expr
    for safe, orig in var_map.items():
        s = s.replace(safe, orig)
    return s


def _gextract_2d(
    exprs: list[str],
    intervals: pd.DataFrame,
    iterator: Iterator | str = None,
    colnames: list[str] | None = None,
    band: tuple[int, int] | tuple[float, float] | None = None,
    caller_ns: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """
    Extract values from 2D tracks for 2D intervals.

    For each expression (must be a simple track name), queries the per-chrom-pair
    binary files using the quad-tree reader.

    Parameters
    ----------
    band : tuple of (d1, d2), optional
        Diagonal band filter. Only objects where d1 <= (x - y) < d2 are returned.

    Returns DataFrame with columns: chrom1, start1, end1, chrom2, start2, end2,
    [expr_columns...], intervalID.
    """
    from .tracks import gtrack_info

    band = _validate_band(band)

    # ── FixedRect via C++ scanner (new in v0.1.75) ────────────────────────
    # A tuple/list of two positive integers means FixedRect binning.
    # Route through pm_extract_2d_scanner when expressions are bare physical
    # track names (no vtracks, no complex expressions).  If the check below
    # finds vtracks, we fall through to the regular path which will raise a
    # helpful error (vtracks not supported with iterator=(N,M) for now).
    if (
        iterator is not None
        and not isinstance(iterator, str)
        and isinstance(iterator, (tuple, list))
        and len(iterator) == 2
    ):
        from ._iterator_policy import FixedRectPolicy, parse_iterator_policy

        policy = parse_iterator_policy(iterator, intervals_is_2d=True)
        if isinstance(policy, FixedRectPolicy):
            # Route if all expressions are bare physical track names or
            # supported reducing 2D vtracks (func in _SCANNER_2D_FUNCS).
            resolved = _resolve_exprs_for_scanner(exprs)
            if resolved is not None:
                return _gextract_2d_via_scanner(
                    exprs, intervals, policy, colnames=colnames, band=band,
                    resolved_vars=resolved,
                )
            compound = _resolve_2d_compound_for_scanner(exprs)
            if compound is not None:
                eval_specs, var_specs = compound
                return _gextract_2d_compound_via_scanner(
                    exprs, intervals, policy, colnames=colnames, band=band,
                    eval_specs=eval_specs, var_specs=var_specs,
                )
            raise NotImplementedError(
                "iterator=(N, M) FixedRect binning is not supported for "
                "this expression. Supported: bare physical 2D track names, "
                "single-source reducing vtracks "
                f"(func in {sorted(_SCANNER_2D_FUNCS - {'mean'})}), or "
                "compound expressions over those (e.g. \"v_a + v_b\")."
            )

    # ── TrackRects via C++ scanner (new in v0.1.75) ───────────────────────
    # String iterator: validate then dispatch.
    # - 2D rectangles/points track + all-bare exprs -> TrackRects C++ scanner
    # - 2D rectangles/points track + vtracks -> fall through to legacy path
    # - 1D track -> raise ValueError (R parity: R errors here)
    # - saved interval set name -> fall through to legacy path (valid misha use)
    # - unknown name (not a track, not an interval set) -> raise ValueError
    if iterator is not None and isinstance(iterator, str):
        from .tracks import gtrack_exists, gtrack_info

        try:
            _iter_exists = gtrack_exists(iterator)
        except Exception:
            _iter_exists = False

        if not _iter_exists:
            # Not a track: check if it's a saved interval set (valid legacy use).
            from .intervals import gintervals_exists

            try:
                _is_iset = gintervals_exists(iterator)
            except Exception:
                _is_iset = False

            if not _is_iset:
                raise ValueError(
                    f"Invalid iterator: {iterator!r} is not a known track name."
                )
            # It's a saved interval set: fall through to the legacy 2D dispatch.
        else:
            _iter_info = gtrack_info(iterator)
            _iter_type = (
                _iter_info.get("type") if isinstance(_iter_info, dict)
                else getattr(_iter_info, "type", None)
            )

            if _iter_type not in ("rectangles", "points"):
                raise ValueError(
                    f"Invalid iterator: {iterator!r} is a 1D track (type={_iter_type!r}). "
                    "A 2D rectangles or points track is required when using a track "
                    "name as iterator for 2D extraction."
                )

            # 2D rects/points track confirmed.
            # Route if all expressions are bare physical tracks or supported
            # reducing 2D vtracks; otherwise fall through to the legacy path.
            resolved = _resolve_exprs_for_scanner(exprs)
            if resolved is not None:
                from ._iterator_policy import TrackRectsPolicy

                policy = TrackRectsPolicy(track_name=iterator)
                return _gextract_2d_via_scanner(
                    exprs, intervals, policy, colnames=colnames, band=band,
                    resolved_vars=resolved,
                )
            compound = _resolve_2d_compound_for_scanner(exprs)
            if compound is not None:
                from ._iterator_policy import TrackRectsPolicy

                eval_specs, var_specs = compound
                policy = TrackRectsPolicy(track_name=iterator)
                return _gextract_2d_compound_via_scanner(
                    exprs, intervals, policy, colnames=colnames, band=band,
                    eval_specs=eval_specs, var_specs=var_specs,
                )
            # Unsupported vtracks or non-routable compound: fall through to
            # the legacy path which resolves vtracks via the original flow.

    # ── CartesianGridSpec via C++ scanner (new in v0.1.76) ───────────────────
    # CartesianGridSpec passed as iterator= routes through the C++ scanner
    # when all expressions are bare physical track names.
    if iterator is not None:
        from ._iterator_policy import CartesianGridSpec

        if isinstance(iterator, CartesianGridSpec):
            resolved = _resolve_exprs_for_scanner(exprs)
            if resolved is not None:
                return _gextract_2d_via_scanner(
                    exprs, intervals, iterator, colnames=colnames, band=band,
                    resolved_vars=resolved,
                )
            compound = _resolve_2d_compound_for_scanner(exprs)
            if compound is not None:
                eval_specs, var_specs = compound
                return _gextract_2d_compound_via_scanner(
                    exprs, intervals, iterator, colnames=colnames, band=band,
                    eval_specs=eval_specs, var_specs=var_specs,
                )
            raise NotImplementedError(
                "iterator=CartesianGridSpec(...) is not supported for this "
                "expression. Supported: bare physical 2D track names, "
                "single-source reducing vtracks "
                f"(func in {sorted(_SCANNER_2D_FUNCS - {'mean'})}), or "
                "compound expressions over those (e.g. \"v_a + v_b\")."
            )

    # ── Intervals iterator via C++ scanner (opt-in, release 4) ───────────────
    # When PYMISHA_USE_SCANNER_FOR_INTERVALS=1 and there is no explicit iterator
    # (iterator is None), route all-bare-track extracts through the scanner
    # instead of the legacy object-enumeration bypass.  vtracks and any case
    # with a distinct iterator fall through to the legacy path unchanged.
    if _scanner_for_intervals_enabled() and iterator is None:
        from ._iterator_policy import IntervalsPolicy

        resolved = _resolve_exprs_for_scanner(exprs)
        if resolved is not None:
            return _gextract_2d_via_scanner(
                exprs, intervals, IntervalsPolicy(),
                colnames=colnames, band=band,
                resolved_vars=resolved,
            )
        compound = _resolve_2d_compound_for_scanner(exprs)
        if compound is not None:
            eval_specs, var_specs = compound
            return _gextract_2d_compound_via_scanner(
                exprs, intervals, IntervalsPolicy(),
                colnames=colnames, band=band,
                eval_specs=eval_specs, var_specs=var_specs,
            )
        # Unsupported vtracks or non-routable compound: fall through to legacy.

    track_names = _track_names_set()
    vtrack_names = set(_shared._VTRACKS.keys())

    parsed = []
    used_tracks = set()
    used_vtracks = set()
    for e in exprs:
        new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(e, track_names, vtrack_names)
        parsed.append((e, new_expr, expr_tracks, expr_vtracks))
        used_tracks.update(expr_tracks)
        used_vtracks.update(expr_vtracks)

    if not used_tracks and not used_vtracks:
        if len(exprs) == 1:
            raise ValueError(
                "Cannot implicitly determine iterator policy:\n"
                f'track expression "{exprs[0]}" does not contain any tracks.'
            )
        raise ValueError("Cannot implicitly determine iterator policy: track expressions do not contain any tracks.")

    for tname in used_tracks:
        info = gtrack_info(tname)
        if int(info.get("dimensions", 1) or 1) != 2:
            raise ValueError(f"Track '{tname}' is not a 2D track (type: {info.get('type')})")

    # Separate vtracks into dim-projected (1D source + dim set) vs 2D vtracks.
    dim_vtracks = set()  # vtracks with dim=1 or dim=2 (1D projection)
    twod_vtracks = set()  # vtracks backed by a 2D source track

    for vt_name in used_vtracks:
        cfg = _shared._VTRACKS.get(vt_name, {})
        dim_val = cfg.get("dim")
        if dim_val is not None and dim_val != 0:
            dim_vtracks.add(vt_name)
        else:
            twod_vtracks.add(vt_name)

    vtrack_to_track = {}
    vtrack_shifts = {}
    vtrack_funcs = {}
    for vt_name in twod_vtracks:
        src, shifts, func = _resolve_2d_vtrack_source(vt_name)
        vtrack_to_track[vt_name] = src
        vtrack_shifts[vt_name] = shifts
        vtrack_funcs[vt_name] = func

    # Classify 2D vtracks: aggregation / object / percentile vs alias (passthrough).
    _ONE_ROW_FUNCS = _2D_AGG_FUNCS | _2D_OBJECT_FUNCS | _2D_PERCENTILE_FUNCS
    onerow_vtracks = {vt for vt in twod_vtracks if vtrack_funcs.get(vt, "") in _ONE_ROW_FUNCS}
    agg_vtracks = {vt for vt in twod_vtracks if vtrack_funcs.get(vt, "") in _2D_AGG_FUNCS}
    obj_vtracks = {vt for vt in twod_vtracks if vtrack_funcs.get(vt, "") in _2D_OBJECT_FUNCS}
    pct_vtracks = {vt for vt in twod_vtracks if vtrack_funcs.get(vt, "") in _2D_PERCENTILE_FUNCS}
    alias_vtracks = twod_vtracks - onerow_vtracks

    has_raw = bool(used_tracks)
    has_alias = bool(alias_vtracks)
    has_onerow = bool(onerow_vtracks)
    has_dim = bool(dim_vtracks)

    # ── Pure one-row-per-interval / dim-projected path ────────────────
    # Aggregation, object, percentile, and dim-projected vtracks all produce
    # one value per interval.  This path is used when we have no raw tracks
    # or alias vtracks.
    if (has_onerow or has_dim) and not has_raw and not has_alias:
        key_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"]
        coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]

        n = len(intervals)

        # Build the base result DataFrame from interval coordinates.
        result = _pandas.DataFrame(
            {
                "chrom1": intervals["chrom1"].to_numpy(),
                "start1": intervals["start1"].values,
                "end1": intervals["end1"].values,
                "chrom2": intervals["chrom2"].to_numpy(),
                "start2": intervals["start2"].values,
                "end2": intervals["end2"].values,
                "intervalID": _numpy.arange(n, dtype=int),
            }
        )

        # Compute 2D aggregation vtracks (area, weighted.sum, min, max, avg).
        for vt_name in agg_vtracks:
            src_track = vtrack_to_track[vt_name]
            s = vtrack_shifts[vt_name]
            shifted = _apply_2d_shifts(intervals, s["sshift1"], s["eshift1"], s["sshift2"], s["eshift2"])
            safe_col = _expr_safe_name(vt_name)
            agg_df = _gextract_2d_vtrack_agg(src_track, safe_col, shifted, band, vtrack_funcs[vt_name])
            result[safe_col] = agg_df[safe_col].to_numpy(dtype=float, copy=False)

        # Compute 2D object-level vtracks (exists, size, first, last, sample).
        for vt_name in obj_vtracks:
            src_track = vtrack_to_track[vt_name]
            s = vtrack_shifts[vt_name]
            shifted = _apply_2d_shifts(intervals, s["sshift1"], s["eshift1"], s["sshift2"], s["eshift2"])
            safe_col = _expr_safe_name(vt_name)
            obj_df = _gextract_2d_vtrack_objects(src_track, safe_col, shifted, band, vtrack_funcs[vt_name])
            result[safe_col] = obj_df[safe_col].to_numpy(dtype=float, copy=False)

        # Compute 2D global.percentile vtracks.
        for vt_name in pct_vtracks:
            src_track = vtrack_to_track[vt_name]
            s = vtrack_shifts[vt_name]
            shifted = _apply_2d_shifts(intervals, s["sshift1"], s["eshift1"], s["sshift2"], s["eshift2"])
            safe_col = _expr_safe_name(vt_name)
            pct_df = _gextract_2d_vtrack_global_percentile(src_track, safe_col, shifted, band)
            result[safe_col] = pct_df[safe_col].to_numpy(dtype=float, copy=False)

        # Compute dim-projected vtracks (1D source, dim=1 or dim=2).
        for vt_name in dim_vtracks:
            safe_col = _expr_safe_name(vt_name)
            vals = _compute_vtrack_values(vt_name, intervals)
            result[safe_col] = _numpy.asarray(vals, dtype=float)

        # Evaluate expressions.
        out_cols = colnames if colnames is not None else exprs
        out_data = {}
        for out_col, (orig_expr, expr_eval, _expr_tracks, expr_vtracks) in zip(out_cols, parsed, strict=False):
            user_vars = _resolve_user_vars(expr_eval, caller_ns) if caller_ns else {}
            allowed_names = {
                "np",
                "numpy",
                *(_expr_safe_name(vt) for vt in expr_vtracks),
                *user_vars.keys(),
            }
            try:
                code_obj = compile_safe_expression(expr_eval, allowed_names)
            except UnsafeExpressionError as exc:
                raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc

            local_ns = {"np": _numpy, "numpy": _numpy}
            local_ns.update(user_vars)
            for vt_name in expr_vtracks:
                safe_col = _expr_safe_name(vt_name)
                local_ns[safe_col] = result[safe_col].to_numpy(dtype=float, copy=False)

            vals = eval(code_obj, {"__builtins__": {}}, local_ns)
            if _numpy.isscalar(vals):
                vals = _numpy.full(len(result), vals, dtype=float)
            out_data[out_col] = _numpy.asarray(vals, dtype=float)

        out_df = result[coord_cols].copy()
        for out_col in out_cols:
            out_df[out_col] = out_data[out_col]
        out_df["intervalID"] = result["intervalID"].to_numpy(dtype=int, copy=False)
        return out_df.sort_values(["chrom1", "start1", "chrom2", "start2", "intervalID"]).reset_index(drop=True)

    # When one-row-per-interval vtracks are mixed with raw tracks or alias vtracks,
    # treat them as alias (one row per object) so that the expression can
    # combine them in a row-aligned manner.
    if has_onerow:
        alias_vtracks = alias_vtracks | onerow_vtracks
        onerow_vtracks = set()
        agg_vtracks = set()
        obj_vtracks = set()
        pct_vtracks = set()
        has_alias = bool(alias_vtracks)
        has_onerow = False

    # ── Raw / alias path (existing behaviour) ─────────────────────────
    required_tracks = []

    def _add_required(track_name: str) -> None:
        if track_name not in required_tracks:
            required_tracks.append(track_name)

    for _orig_expr, _expr_eval, expr_tracks, expr_vtracks in parsed:
        for tname in expr_tracks:
            _add_required(tname)
        for vt_name in expr_vtracks:
            if vt_name not in dim_vtracks:
                _add_required(vtrack_to_track[vt_name])

    if len(required_tracks) > 1 and iterator is None:
        raise ValueError(
            "Cannot implicitly determine iterator policy: track expressions contain more than one 2D track."
        )

    if not required_tracks and not dim_vtracks:
        raise ValueError("Cannot implicitly determine iterator policy: track expressions do not contain any tracks.")

    key_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"]
    coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]

    if required_tracks:
        anchor_track = required_tracks[0]
        if isinstance(iterator, str):
            if iterator in required_tracks:
                anchor_track = iterator
            elif iterator in vtrack_to_track:
                anchor_track = vtrack_to_track[iterator]

        track_cols = {tname: _expr_safe_name(tname) for tname in required_tracks}

        def _get_shifted_intervals(track_name: str) -> pd.DataFrame:
            """Return intervals with shifts applied if track is accessed via a shifted vtrack."""
            for vt_name, src in vtrack_to_track.items():
                if src == track_name:
                    s = vtrack_shifts[vt_name]
                    return _apply_2d_shifts(intervals, s["sshift1"], s["eshift1"], s["sshift2"], s["eshift2"])
            return intervals

        result = _gextract_2d_single(anchor_track, track_cols[anchor_track], _get_shifted_intervals(anchor_track), band)
        if result is None:
            return None

        for tname in required_tracks:
            if tname == anchor_track:
                continue
            cur = _gextract_2d_single(tname, track_cols[tname], _get_shifted_intervals(tname), band)
            if cur is None:
                result[track_cols[tname]] = _numpy.nan
                continue
            cur = cur[key_cols + [track_cols[tname]]]
            if cur.duplicated(key_cols).any():
                cur = cur.drop_duplicates(key_cols, keep="first")
            result = result.merge(cur, on=key_cols, how="left")
    else:
        track_cols = {}
        # No 2D raw tracks — only dim-projected vtracks mixed with something
        # that forced us into the raw/alias path. Build a base result from
        # the input intervals.
        n = len(intervals)
        result = _pandas.DataFrame(
            {
                "chrom1": intervals["chrom1"].to_numpy(),
                "start1": intervals["start1"].values,
                "end1": intervals["end1"].values,
                "chrom2": intervals["chrom2"].to_numpy(),
                "start2": intervals["start2"].values,
                "end2": intervals["end2"].values,
                "intervalID": _numpy.arange(n, dtype=int),
            }
        )

    # Compute dim-projected vtracks for the raw/alias path result rows.
    # For each result row, we map back to the original 2D interval via
    # intervalID and then project to 1D for the vtrack computation.
    dim_vtrack_arrays = {}
    if dim_vtracks:
        # Pre-compute vtrack values per original interval.
        dim_vals_per_interval = {}
        for vt_name in dim_vtracks:
            vals = _compute_vtrack_values(vt_name, intervals)
            dim_vals_per_interval[vt_name] = _numpy.asarray(vals, dtype=float)

        # Map result rows to original intervals via intervalID.
        interval_ids = result["intervalID"].to_numpy(dtype=int, copy=False)
        for vt_name in dim_vtracks:
            dim_vtrack_arrays[vt_name] = dim_vals_per_interval[vt_name][interval_ids]

    out_cols = colnames if colnames is not None else exprs
    out_data = {}
    for out_col, (orig_expr, expr_eval, expr_tracks, expr_vtracks) in zip(out_cols, parsed, strict=False):
        user_vars = _resolve_user_vars(expr_eval, caller_ns) if caller_ns else {}
        allowed_names = {
            "np",
            "numpy",
            *(_expr_safe_name(t) for t in expr_tracks),
            *(_expr_safe_name(vt) for vt in expr_vtracks),
            *user_vars.keys(),
        }
        try:
            code_obj = compile_safe_expression(expr_eval, allowed_names)
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc

        local_ns = {"np": _numpy, "numpy": _numpy}
        local_ns.update(user_vars)
        for tname in expr_tracks:
            local_ns[_expr_safe_name(tname)] = result[track_cols[tname]].to_numpy(dtype=float, copy=False)
        for vt_name in expr_vtracks:
            if vt_name in dim_vtracks:
                local_ns[_expr_safe_name(vt_name)] = dim_vtrack_arrays[vt_name]
            else:
                src_track = vtrack_to_track[vt_name]
                local_ns[_expr_safe_name(vt_name)] = result[track_cols[src_track]].to_numpy(dtype=float, copy=False)

        vals = eval(code_obj, {"__builtins__": {}}, local_ns)
        if _numpy.isscalar(vals):
            vals = _numpy.full(len(result), vals, dtype=float)
        out_data[out_col] = _numpy.asarray(vals, dtype=float)

    out_df = result[coord_cols].copy()
    for out_col in out_cols:
        out_df[out_col] = out_data[out_col]
    out_df["intervalID"] = result["intervalID"].to_numpy(dtype=int, copy=False)
    return out_df.sort_values(["chrom1", "start1", "chrom2", "start2", "intervalID"]).reset_index(drop=True)


def giterator_intervals_2d(
    expr: str | list[str],
    intervals: pd.DataFrame | str | None = None,
    iterator: Iterator | str = None,
    colnames: list[str] | None = None,
    band: tuple[int, int] | None = None,
) -> collections.abc.Generator[pd.DataFrame, None, None]:
    """Iterate over 2D intervals, yielding extracted data one interval at a time.

    This is a streaming interface for 2D track extraction.  Instead of
    returning one large DataFrame for all intervals (as :func:`gextract` does),
    it yields one DataFrame per input interval, keeping peak memory low.

    Each yielded DataFrame has the same column layout as the corresponding
    :func:`gextract` result (``chrom1, start1, end1, chrom2, start2, end2,
    <expr_columns...>, intervalID``).  The ``intervalID`` reflects the
    position of the interval in the original *intervals* DataFrame
    (0-based).

    Parameters
    ----------
    expr : str or list of str
        One or more track expressions to evaluate.
    intervals : DataFrame or str, optional
        2D genomic scope (``chrom1/start1/end1/chrom2/start2/end2``
        DataFrame, or a named interval-set string).  If ``None``, defaults
        to :func:`gintervals_2d_all`.
    iterator : int or str, optional
        Track expression iterator.  Passed through to the underlying
        extraction engine.
    colnames : list of str, optional
        Column names for expression values.  Must match the number of
        expressions.
    band : tuple of (int, int), optional
        Diagonal band filter ``(d1, d2)``.

    Yields
    ------
    DataFrame
        One DataFrame per input interval that produces at least one result
        row.  Intervals that match no data are silently skipped.

    See Also
    --------
    gextract : Bulk extraction (returns one DataFrame for all intervals).
    giterator_intervals : 1D iterator grid (no expression evaluation).

    Examples
    --------
    >>> import pymisha as pm                              # doctest: +SKIP
    >>> _ = pm.gdb_init_examples()                        # doctest: +SKIP
    >>> intervals = pm.gintervals_2d("1", 0, 5000, "1", 0, 5000)  # doctest: +SKIP
    >>> for chunk in pm.giterator_intervals_2d("dense_track_2d", intervals):
    ...     print(chunk.shape)                            # doctest: +SKIP
    """
    _checkroot()

    exprs = [expr] if isinstance(expr, str) else list(expr)

    if intervals is None:
        from .intervals import gintervals_2d_all

        intervals = gintervals_2d_all()

    intervals = _maybe_load_intervals_set(intervals)
    intervals = _maybe_load_2d_intervals_set(intervals, exprs, iterator, band)

    if not _is_2d_intervals(intervals):
        raise ValueError(
            "giterator_intervals_2d requires 2D intervals (columns chrom1/start1/end1/chrom2/start2/end2)."
        )

    if colnames is not None and len(colnames) != len(exprs):
        raise ValueError(f"colnames length ({len(colnames)}) must match number of expressions ({len(exprs)})")

    if len(intervals) == 0:
        return

    assert isinstance(intervals, pd.DataFrame)
    for idx in range(len(intervals)):
        single = intervals.iloc[[idx]].reset_index(drop=True)
        chunk = _gextract_2d(exprs, single, iterator=iterator, colnames=colnames, band=band)
        if chunk is not None and len(chunk) > 0:
            # Stamp the original interval index so callers can correlate
            # results back to the input DataFrame.
            chunk = chunk.copy()
            chunk["intervalID"] = idx
            yield chunk


def _is_attachable_dtype(s: pd.Series) -> tuple[bool, str]:
    """Return (ok, reason) for whether a column is supported by intervals_join='intervals'.

    Supported: int, float, bool, string/object (assumed str), category, pandas StringDtype.
    Rejected: datetime, timedelta, period, complex, lists/dicts in object columns.
    """
    # Categorical is allowed.
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True, ""
    # pandas StringDtype is allowed.
    if pd.api.types.is_string_dtype(s) and not pd.api.types.is_object_dtype(s):
        return True, ""
    # Numeric + bool via kind char.
    if s.dtype.kind in {"i", "u", "f", "b"}:
        return True, ""
    # Object dtype: only allow if every non-null entry is a str.
    if s.dtype == object:
        non_null = s.dropna()
        if len(non_null) == 0 or non_null.map(lambda v: isinstance(v, str)).all():
            return True, ""
        return False, "object column contains non-string values"
    return False, f"unsupported dtype {s.dtype}"


def _apply_intervals_join(
    df: pd.DataFrame | None,
    input_intervals: pd.DataFrame | None,
    intervals_join: str,
    is_2d: bool = False,
) -> pd.DataFrame | None:
    """Post-process gextract result according to intervals_join mode.

    - 'id': no change (intervalID kept).
    - 'none': drop intervalID.
    - 'intervals': drop intervalID, attach all columns of input_intervals
      (mapped via 1-indexed intervalID), suffix conflicts with '1'.
    """
    if df is None or intervals_join == "id":
        return df
    if "intervalID" not in df.columns:
        return df  # Already processed or no intervalID emitted.
    if intervals_join == "none":
        return df.drop(columns=["intervalID"])
    if intervals_join != "intervals":
        raise ValueError(f"intervals_join must be one of 'id', 'intervals', 'none'; got {intervals_join!r}")

    if input_intervals is None or not isinstance(input_intervals, pd.DataFrame):
        # No DataFrame to attach (e.g. ALLGENOME default). Treat as 'none'.
        return df.drop(columns=["intervalID"])

    # Validate attach-supported dtypes BEFORE doing the join.
    for col in input_intervals.columns:
        ok, reason = _is_attachable_dtype(input_intervals[col])
        if not ok:
            raise TypeError(
                f"intervals_join='intervals': column {col!r} has unsupported type "
                f"for attach ({reason})"
            )

    # Conflict resolution: output columns that collide with input get suffix '1'.
    output_cols = set(df.columns) - {"intervalID"}
    rename_map = {c: c + "1" for c in input_intervals.columns if c in output_cols}
    attach_src = input_intervals.rename(columns=rename_map).reset_index(drop=True)

    # 1-indexed intervalID -> 0-indexed positional lookup.
    ids = df["intervalID"].to_numpy(dtype=int) - 1
    attached = attach_src.iloc[ids].reset_index(drop=True)

    return pd.concat(
        [df.drop(columns=["intervalID"]).reset_index(drop=True), attached],
        axis=1,
    )


def _apply_extract_output(
    df: pd.DataFrame | None,
    file: str | None,
    intervals_set_out: str | None,
    *,
    is_2d: bool = False,
    intervals_join: str = "id",
    input_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Apply file-writing, intervals_set_out, and intervals_join post-processing to an extraction result.

    Parameters
    ----------
    df : DataFrame or None
        The extraction result.
    file : str or None
        If given, write *df* to this path as tab-separated values and return
        ``None`` instead of the DataFrame.
    intervals_set_out : str or None
        If given, save the coordinate columns of *df* as a named interval
        set via :func:`gintervals_save`.
    is_2d : bool
        Whether the extraction was 2D (affects which coordinate columns are
        used for ``intervals_set_out``).
    intervals_join : {"id", "intervals", "none"}, default "id"
        Post-processing mode for the intervalID column.
    input_intervals : DataFrame or None
        Resolved input intervals used for the ``intervals_join="intervals"``
        attach (not used here for ``"id"``/``"none"``).

    Returns
    -------
    DataFrame or None
    """
    if df is None:
        # Nothing to write; still honour the file contract (return None).
        if file is not None:
            return None
        return None

    # Apply intervals_join FIRST, before file/intervals_set_out which read
    # from the post-processed result.
    df = _apply_intervals_join(df, input_intervals, intervals_join, is_2d=is_2d)
    assert df is not None  # _apply_intervals_join preserves non-None input

    # -- intervals_set_out: save coordinate columns as a named interval set --
    if intervals_set_out is not None:
        from .intervals import gintervals_save

        coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"] if is_2d else ["chrom", "start", "end"]
        coords = df[coord_cols].drop_duplicates().reset_index(drop=True)
        gintervals_save(coords, intervals_set_out)

    # -- file: write TSV and return None --
    if file is not None:
        df.to_csv(file, sep="\t", index=False)
        return None

    return df


def _worker_extract_chunk(args: tuple[Any, ...]) -> pd.DataFrame | None:
    """Worker function for parallel gextract (runs in forked subprocess)."""
    if len(args) == 5:
        (chunk_dict, exprs, iterator_val, config_dict, vtracks_dict) = args
    else:
        (chunk_dict, exprs, iterator_val, config_dict) = args
        vtracks_dict = None
    chunk_intervals = _pandas.DataFrame(chunk_dict)
    result = _pymisha.pm_extract(
        exprs,
        _df2pymisha(chunk_intervals),
        iterator_val,
        config_dict,
        vtracks_dict,
    )
    return _pymisha2df(result)


_VALID_STRATEGIES = ("auto", "tracks", "tiles")
# Track count at or above which "auto" picks track-parallel.
# Matches R misha 5.6.18 threshold (>= 8 tracks + non-streaming iterator).
_AUTO_TRACKS_MIN_EXPRS = 8


def _workload_too_small_for_fork(
    intervals: pd.DataFrame, max_procs: int, config: dict
) -> bool:
    """Return True if the workload is below the multitask thresholds.

    Mirrors the C++ ``choose_num_kids`` gating (min_intervs4process,
    min_scope4process) so the Python-level worker-pool fork is also
    skipped when serial is expected to win.
    """
    if max_procs < 2:
        return True
    n = len(intervals)
    min_intervs = int(config.get("min_intervs4process", 250_000))
    if min_intervs > 0 and n < min_intervs:
        return True
    min_scope = int(config.get("min_scope4process", 1_000_000_000))
    if min_scope > 0 and "start" in intervals.columns and "end" in intervals.columns:
        # Sum can overflow Python int? No - Python ints are arbitrary precision.
        scope = int((intervals["end"].to_numpy() - intervals["start"].to_numpy()).sum())
        if scope // max_procs < min_scope:
            return True
    return False


def _validate_strategy(strategy: str) -> str:
    """Raise ValueError on unknown strategy; return lowercased value."""
    s = str(strategy).lower()
    if s not in _VALID_STRATEGIES:
        raise ValueError(
            f"multitasking_strategy must be one of {_VALID_STRATEGIES}; got {strategy!r}"
        )
    return s


def _resolve_parallel_strategy(strategy: str, n_exprs: int, iterator: Any) -> str:
    """Decide between 'tracks' and 'tiles' given the configured strategy.

    'auto' picks 'tracks' when there are enough expressions to amortize fork
    overhead AND the iterator is non-streaming (fixed bin size or a concrete
    interval set - i.e., output size is predictable per worker). Otherwise
    'tiles' (the legacy chrom-parallel path).
    """
    s = _validate_strategy(strategy)
    if s != "auto":
        return s
    if n_exprs < _AUTO_TRACKS_MIN_EXPRS:
        return "tiles"
    # Heuristic for "non-streaming" iterator: an int (fixed binsize) or a
    # concrete DataFrame is predictable; named-track/string iterators may
    # produce variable output and are kept on the tiles path.
    if isinstance(iterator, (int, _pandas.DataFrame)):
        return "tracks"
    return "tiles"


def _worker_extract_tracks(args: tuple[Any, ...]) -> pd.DataFrame | None:
    """Worker for track-parallel gextract: process a subset of expressions
    over the full interval set."""
    (intervals_dict, expr_subset, iterator_val, config_dict, vtracks_dict) = args
    intervals_df = _pandas.DataFrame(intervals_dict)
    result = _pymisha.pm_extract(
        expr_subset,
        _df2pymisha(intervals_df),
        iterator_val,
        config_dict,
        vtracks_dict,
    )
    return _pymisha2df(result)


def _split_exprs(exprs: list[str], n: int) -> list[list[str]]:
    """Split *exprs* into at most *n* contiguous, non-empty chunks."""
    n = max(1, min(n, len(exprs)))
    base = len(exprs) // n
    rem = len(exprs) % n
    out = []
    i = 0
    for k in range(n):
        size = base + (1 if k < rem else 0)
        out.append(exprs[i:i + size])
        i += size
    return out


def _parallel_extract_tracks(
    exprs: list[str],
    intervals: pd.DataFrame,
    iterator: Iterator | str,
    config: dict[str, Any],
    vtracks_dict: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame | None:
    """Run gextract with each worker handling a subset of expressions across
    the full interval set, then column-bind the expression results.

    Returns the merged DataFrame, or ``None`` to fall back to the legacy path
    when the inputs aren't amenable to track-parallel.
    """
    import multiprocessing

    max_procs = int(config.get("max_processes", 1))
    if max_procs < 2 or len(exprs) < 2:
        return None

    n_workers = min(max_procs, len(exprs))
    expr_chunks = _split_exprs(exprs, n_workers)

    worker_config = dict(config)
    worker_config["progress"] = False

    intervals_dict = intervals.to_dict(orient="list")
    worker_args = [
        (intervals_dict, chunk, iterator, worker_config, vtracks_dict)
        for chunk in expr_chunks
    ]

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_extract_tracks, worker_args)

    non_empty = [(chunk, df) for chunk, df in zip(expr_chunks, results, strict=True)
                 if df is not None and len(df) > 0]
    if not non_empty:
        return _pandas.DataFrame()

    # Workers ran on identical (intervals, iterator) so the coordinate /
    # intervalID rows match. Sort each by intervalID to be defensive, then
    # take coords from the first chunk, column-bind expression values from
    # all chunks, and append intervalID last to match serial output column
    # order (coords -> exprs -> intervalID).
    sort_key = "intervalID"
    first_chunk_exprs, first_df = non_empty[0]
    if sort_key in first_df.columns:
        first_df = first_df.sort_values(sort_key).reset_index(drop=True)

    coord_cols = [c for c in ("chrom", "start", "end", "chrom1", "start1", "end1",
                               "chrom2", "start2", "end2")
                  if c in first_df.columns]
    meta_cols = coord_cols + ([sort_key] if sort_key in first_df.columns else [])

    def _value_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if c not in meta_cols]]

    parts = [first_df[coord_cols].copy(), _value_columns(first_df)]
    for _chunk, df in non_empty[1:]:
        if sort_key in df.columns:
            df = df.sort_values(sort_key).reset_index(drop=True)
        parts.append(_value_columns(df))
    if sort_key in first_df.columns:
        parts.append(first_df[[sort_key]].copy())

    return _pandas.concat(parts, axis=1)


def _parallel_extract(
    exprs: list[str],
    intervals: pd.DataFrame,
    iterator: Iterator | str,
    config: dict[str, Any],
    vtracks_dict: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame | None:
    """Split intervals by chromosome and extract in parallel.

    Returns a merged DataFrame with globally consistent intervalIDs that
    match what a serial extraction over the same *intervals* would produce,
    or ``None`` to signal the caller to use the normal serial path.
    """
    import multiprocessing

    max_procs = int(config.get("max_processes", 1))
    if max_procs < 2:
        return None  # signal caller to use normal path

    # Group intervals by chromosome
    chroms = intervals["chrom"].unique()
    if len(chroms) < 2:
        return None  # not worth parallelizing a single chromosome

    n_workers = min(max_procs, len(chroms))
    # Suppress progress in workers (parent handles it)
    worker_config = dict(config)
    worker_config["progress"] = False

    # Build per-chromosome chunks, tracking original interval indices so we
    # can remap intervalID after the parallel extraction.
    chunks = []
    original_indices = []  # list of arrays: original 0-based positions
    for chrom in sorted(chroms):
        mask = intervals["chrom"] == chrom
        chunk = intervals[mask]
        orig_idx = _numpy.where(mask)[0]  # 0-based positions in parent
        chunks.append(chunk.to_dict(orient="list"))
        original_indices.append(orig_idx)

    worker_args = [(chunk_dict, exprs, iterator, worker_config, vtracks_dict) for chunk_dict in chunks]

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_extract_chunk, worker_args)

    # Collect and concatenate non-None results, remapping intervalID
    dfs = []
    for df, orig_idx in zip(results, original_indices, strict=False):
        if df is not None and len(df) > 0:
            if "intervalID" in df.columns:
                # C++ intervalID is 1-based within the chunk.
                # Remap: chunk_local_id -> original_global_id (1-based).
                local_ids = df["intervalID"].to_numpy()
                # local_ids are 1-based; orig_idx is 0-based
                global_ids = orig_idx[local_ids - 1] + 1
                df = df.copy()
                df["intervalID"] = global_ids
            dfs.append(df)
    if not dfs:
        return _pandas.DataFrame()
    return _pandas.concat(dfs, ignore_index=True)


def gextract(
    expr: str | list[str],
    intervals: pd.DataFrame | str | None = None,
    iterator: Iterator | str = None,
    colnames: list[str] | None = None,
    band: tuple[int, int] | tuple[float, float] | None = None,
    vars: dict[str, Any] | None = None,
    intervals_join: str = "id",
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Return evaluated track expression values for each iterator interval.

    For each interval in the iterator, evaluates one or more track expressions
    and returns the results as a DataFrame with interval coordinates and
    expression values. An ``intervalID`` column maps each output row back to
    the input interval.

    If input intervals overlap, overlapped coordinates appear multiple times.
    The order of results may differ from input interval order; use
    ``intervalID`` to match rows to original intervals.

    Parameters
    ----------
    expr : str or list of str
        One or more track expressions to evaluate.
    intervals : DataFrame or str, optional
        Genomic scope (chrom/start/end DataFrame or intervals set name).
        If None, uses ALLGENOME. For 2D tracks, pass 2D intervals (with
        chrom1/start1/end1/chrom2/start2/end2 columns).
    colnames : list of str, optional
        Column names for expression values. Must match the number of
        expressions. If None, uses expression strings.
    iterator : int or str, optional
        Track expression iterator. If None, determined from expressions.
        For multi-expression 2D extraction, pass an explicit iterator.
    band : tuple of (int, int), optional
        Diagonal band for 2D track extraction as ``(d1, d2)``. Only
        applicable with 2D intervals.
    vars : dict, optional
        Explicit variable bindings for the expression.  When provided,
        these are used instead of auto-capturing the caller's namespace.
    intervals_join : {"id", "intervals", "none"}, default "id"
        How to relate output rows back to the input *intervals*.

        - ``"id"`` (default): append an ``intervalID`` column (1-indexed)
          to each output row, mapping it to the input intervals row.
        - ``"intervals"``: drop ``intervalID`` and attach every column
          of the input *intervals* DataFrame to each output row.
          Conflicting names get a ``"1"`` suffix (``chrom`` -> ``chrom1``).
          Not supported with ``file=`` or ``intervals_set_out=``.
          Supported attach dtypes: numeric, bool, string, category.
        - ``"none"``: drop ``intervalID``, attach nothing.
    **kwargs
        Additional keyword arguments:

        - **file** (*str, optional*) -- Path to write extraction results as
          a tab-separated file. When provided, the result is written to the
          file and ``None`` is returned instead of a DataFrame.
        - **intervals_set_out** (*str, optional*) -- Name of an interval set
          to save the result coordinate columns to. The interval set is
          created via :func:`gintervals_save`.
        - **progress** (*bool or str, optional*) -- Whether to show a
          progress bar.
        - **progress_desc** (*str, optional*) -- Description for the
          progress bar (default ``'gextract'``).

    Returns
    -------
    DataFrame or None
        DataFrame with columns: chrom, start, end, <expr1>, ..., intervalID.
        Returns None if the iterator produces no intervals, or if *file* is
        specified.

    See Also
    --------
    gsummary : Summarize track expression over intervals.
    gquantiles : Compute quantiles of track expression over intervals.
    gdist : Compute distribution of track expression over intervals.
    glookup : Look up track values at specific positions.
    gscreen : Find intervals where a logical expression is True.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gextract("dense_track", intervals=pm.gintervals("1", 0, 1000),
    ...                      iterator=200, progress=False)
    >>> result.columns.tolist()
    ['chrom', 'start', 'end', 'dense_track', 'intervalID']
    >>> len(result)
    5
    """
    _checkroot()

    # Capture caller namespace for user variable resolution
    caller_ns = dict(vars) if vars is not None else _caller_namespace(depth=1)

    exprs = [expr] if isinstance(expr, str) else list(expr)

    if intervals_join not in ("id", "intervals", "none"):
        raise ValueError(
            f"intervals_join must be one of 'id', 'intervals', 'none'; got {intervals_join!r}"
        )

    if intervals_join == "intervals":
        if kwargs.get("file") is not None:
            raise ValueError(
                "intervals_join='intervals' is not supported with file= output "
                "(TSV writer cannot safely round-trip arbitrary column types)"
            )
        if kwargs.get("intervals_set_out") is not None:
            raise ValueError(
                "intervals_join='intervals' is not supported with intervals_set_out= "
                "(intervals sets only carry numeric coordinate columns)"
            )

    from .tracks import _check_computed_tracks

    _check_computed_tracks(exprs)

    file = kwargs.get("file")
    intervals_set_out = kwargs.get("intervals_set_out")

    if intervals is None:
        from .intervals import gintervals_all

        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)
    intervals = _maybe_load_2d_intervals_set(intervals, exprs, iterator, band)

    # Capture for intervals_join post-processing. _preprocess_intervals_iterator
    # may mutate intervals, but the row order / column set we want for attach
    # is the post-loader, pre-iterator-preprocess state.
    input_intervals_for_join = intervals.copy() if isinstance(intervals, pd.DataFrame) else None

    # Handle DataFrame-as-iterator: intersect scope with iterator intervals
    intervals, iterator, _itr_id_map = _preprocess_intervals_iterator(intervals, iterator)

    # Route to 2D extraction if intervals are 2D
    if _is_2d_intervals(intervals):
        if colnames is not None and len(colnames) != len(exprs):
            raise ValueError(f"colnames length ({len(colnames)}) must match number of expressions ({len(exprs)})")
        df = _gextract_2d(
            exprs,
            intervals,
            iterator=iterator,
            colnames=colnames,
            band=band,
            caller_ns=caller_ns,
        )
        return _apply_extract_output(
            df, file, intervals_set_out, is_2d=True,
            intervals_join=intervals_join,
            input_intervals=input_intervals_for_join,
        )

    if band is not None:
        raise ValueError("band parameter is only supported with 2D intervals")

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gextract")

    if colnames is not None and len(colnames) != len(exprs):
        raise ValueError(f"colnames length ({len(colnames)}) must match number of expressions ({len(exprs)})")

    track_names = _track_names_set()
    vtrack_names = set(_shared._VTRACKS.keys())

    parsed = []
    used_tracks = set()
    used_vtracks = set()
    all_user_vars = {}
    for e in exprs:
        new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(e, track_names, vtrack_names)
        used_tracks.update(expr_tracks)
        used_vtracks.update(expr_vtracks)
        parsed.append((e, new_expr, expr_tracks, expr_vtracks))
        # Resolve user variables from the caller's namespace
        user_vars = _resolve_user_vars(new_expr, caller_ns)
        all_user_vars.update(user_vars)

    for orig_expr, expr_eval, expr_tracks, expr_vtracks in parsed:
        allowed_names = {
            "np",
            "numpy",
            "CHROM",
            "START",
            "END",
            *(_expr_safe_name(t) for t in expr_tracks),
            *(_expr_safe_name(vt) for vt in expr_vtracks),
        }
        allowed_names |= set(all_user_vars.keys())
        try:
            compile_safe_expression(expr_eval, allowed_names)
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc

    # Check if vtracks can go through C++ path
    cpp_vtracks_extract = used_vtracks and _can_vtracks_use_cpp(used_vtracks) and not all_user_vars

    if (not used_vtracks and not all_user_vars) or cpp_vtracks_extract:
        vtracks_dict = _build_vtracks_dict(used_vtracks) if cpp_vtracks_extract else None
        # Try parallel extraction if max_processes > 1.
        # Skip when a custom progress callback is provided (not compatible
        # with forked workers) or when file output is requested.
        with _config_no_mt(_itr_id_map) as _cfg:
            df = None
            _max_procs = int(_cfg.get("max_processes", 1))
            use_parallel = (
                _cfg.get("multitasking")
                and _max_procs > 1
                and not callable(progress)
                and file is None
                and not _workload_too_small_for_fork(intervals, _max_procs, _cfg)
            )
            if use_parallel:
                strategy = _resolve_parallel_strategy(
                    _cfg.get("multitasking_strategy", "auto"),
                    n_exprs=len(exprs),
                    iterator=iterator,
                )
                if strategy == "tracks":
                    df = _parallel_extract_tracks(
                        exprs, intervals, iterator, _cfg, vtracks_dict=vtracks_dict,
                    )
                if df is None:
                    df = _parallel_extract(exprs, intervals, iterator, _cfg, vtracks_dict=vtracks_dict)

            if df is None:
                with _progress_context(progress, desc=progress_desc):
                    result = _pymisha.pm_extract(
                        exprs,
                        _df2pymisha(intervals),
                        iterator,
                        _cfg,
                        vtracks_dict,
                    )
                df = _pymisha2df(result)
        df = _remap_interval_ids(df, _itr_id_map)
        if colnames is not None and df is not None and isinstance(df, _pandas.DataFrame):
            # Build rename map: old expression columns -> new names
            # The C++ path names columns after the expression strings
            non_meta = [c for c in df.columns if c not in ("chrom", "start", "end", "intervalID")]
            if len(non_meta) == len(colnames):
                rename_map = dict(zip(non_meta, colnames, strict=False))
                df = df.rename(columns=rename_map)
        return _apply_extract_output(
            df, file, intervals_set_out, is_2d=False,
            intervals_join=intervals_join,
            input_intervals=input_intervals_for_join,
        )

    track_arrays = {}
    base_df = None
    iter_df = None

    if used_tracks:
        track_exprs = list(used_tracks)
        with _config_no_mt(_itr_id_map) as _cfg:
            base_result = _pymisha.pm_extract(track_exprs, _df2pymisha(intervals), iterator, _cfg)
        base_df = _pymisha2df(base_result)
        if base_df is None:
            raise RuntimeError("Failed to extract physical track values for mixed expression")
        base_df = _remap_interval_ids(base_df, _itr_id_map)
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            if col not in base_df.columns:
                raise KeyError(f"Track column not found for '{tname}'")
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)

        iter_df = base_df[["chrom", "start", "end", "intervalID"]]
    else:
        if iterator is None:
            # Array-slice vtracks can determine their own iterator from the
            # query intervals (one output row per input interval).
            all_array_slice = used_vtracks and all(
                _shared._VTRACKS.get(vt, {}).get("kind") == "array_slice"
                for vt in used_vtracks
            )
            if all_array_slice:
                # Use intervals as-is, assigning intervalID 1..N
                # By this point intervals is guaranteed to be a DataFrame
                # (preprocessing above resolves named sets); the cast is purely
                # to narrow the static type away from the input str alternative.
                assert isinstance(intervals, _pandas.DataFrame)
                iter_df = intervals.copy()
                iter_df["intervalID"] = _numpy.arange(1, len(iter_df) + 1, dtype=int)
            elif len(exprs) == 1:
                raise ValueError(
                    f"Cannot implicitly determine iterator policy:\n"
                    f'track expression "{exprs[0]}" does not contain any tracks.'
                )
            else:
                raise ValueError(
                    "Cannot implicitly determine iterator policy: track expressions do not contain any tracks."
                )
        else:
            iter_df = _iterated_intervals(intervals, iterator)

    if iter_df is None or len(iter_df) == 0:
        return _apply_extract_output(
            None, file, intervals_set_out, is_2d=False,
            intervals_join=intervals_join,
            input_intervals=input_intervals_for_join,
        )

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)  # type: ignore[call-overload]
    compiled = []
    result_cols = []
    for i, (orig_expr, expr_eval, expr_tracks, expr_vtracks) in enumerate(parsed):
        colname = colnames[i] if colnames is not None else _bound_colname(orig_expr, 40)
        allowed_names = {
            "np",
            "numpy",
            "CHROM",
            "START",
            "END",
            *(_expr_safe_name(t) for t in expr_tracks),
            *(_expr_safe_name(vt) for vt in expr_vtracks),
        }
        allowed_names |= set(all_user_vars.keys())
        try:
            code_obj = compile_safe_expression(expr_eval, allowed_names)
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc
        compiled.append((colname, code_obj, expr_tracks, expr_vtracks))
        result_cols.append(colname)

    result_arrays = {col: _numpy.empty(n_rows, dtype=float) for col in result_cols}

    chrom_vals = iter_df["chrom"].to_numpy()
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    # Pre-compute vtrack values once for all intervals (avoids per-chunk recomputation)
    precomputed_vtracks = {}
    if used_vtracks:
        all_intervals = iter_df[["chrom", "start", "end"]]
        for vt in used_vtracks:
            precomputed_vtracks[vt] = _compute_vtrack_values(vt, all_intervals)

    with _progress_context(progress, total=n_rows, desc=progress_desc) as progress_cb:
        for start_idx, end_idx in _chunk_slices(n_rows, chunk_size):
            sl = slice(start_idx, end_idx)

            local_ns = {
                "np": _numpy,
                "numpy": _numpy,
                "CHROM": chrom_vals[sl],
                "START": start_vals[sl],
                "END": end_vals[sl],
            }

            for tname, arr in track_arrays.items():
                local_ns[_expr_safe_name(tname)] = arr[sl]

            for vt, arr in precomputed_vtracks.items():
                local_ns[_expr_safe_name(vt)] = arr[sl]

            local_ns.update(all_user_vars)

            for colname, code_obj, _expr_tracks, _expr_vtracks in compiled:
                result_values = eval(code_obj, {"__builtins__": {}}, local_ns)
                if _numpy.isscalar(result_values):
                    result_values = _numpy.full(end_idx - start_idx, result_values)
                result_arrays[colname][sl] = _numpy.asarray(result_values, dtype=float)

            if progress_cb:
                total = n_rows
                done = end_idx
                pct = int(done * 100.0 / total) if total else 100
                progress_cb(done, total, pct)

        if progress_cb:
            progress_cb(n_rows, n_rows, 100)

    result_df = _pandas.DataFrame(
        {
            "chrom": chrom_vals,
            "start": start_vals,
            "end": end_vals,
        }
    )
    for col in result_cols:
        result_df[col] = result_arrays[col]
    result_df["intervalID"] = iter_df["intervalID"].to_numpy(dtype=int, copy=False)
    return _apply_extract_output(
        result_df, file, intervals_set_out, is_2d=False,
        intervals_join=intervals_join,
        input_intervals=input_intervals_for_join,
    )


def gscreen(
    expr: str, intervals: pd.DataFrame | str | None = None, vars: dict[str, Any] | None = None, **kwargs: Any
) -> pd.DataFrame | None:
    """Find intervals where a logical track expression is True.

    Evaluates a logical track expression and returns all intervals where
    the expression value is True (non-zero). Adjacent True intervals on the
    same chromosome are merged into a single interval.

    Parameters
    ----------
    expr : str
        Logical track expression.
    intervals : DataFrame or str, optional
        Genomic scope (chrom/start/end DataFrame or intervals set name).
        If None, uses ALLGENOME.
    vars : dict, optional
        Explicit variable bindings for the expression.
    **kwargs
        Additional keyword arguments:

        - **iterator** (*int or str, optional*) -- Track expression iterator.
          If None, determined from expression.
        - **progress** (*bool or str, optional*) -- Whether to show a
          progress bar.
        - **progress_desc** (*str, optional*) -- Description for the
          progress bar (default ``'gscreen'``).

    Returns
    -------
    DataFrame or None
        DataFrame with columns: chrom, start, end. Returns None if no
        intervals match the expression.

    See Also
    --------
    gextract : Extract track expression values for each interval.
    gsegment : Segment genome by track expression values.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gscreen("dense_track > 0.2", intervals=pm.gintervals("1", 0, 10000),
    ...                     progress=False)
    >>> "chrom" in result.columns
    True
    """
    _checkroot()

    # Capture caller namespace for user variable resolution
    caller_ns = dict(vars) if vars is not None else _caller_namespace(depth=1)

    from .tracks import _check_computed_tracks

    _check_computed_tracks(expr)

    if intervals is None:
        from .intervals import gintervals_all

        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gscreen")
    iterator = kwargs.get("iterator")
    band = kwargs.get("band")

    intervals = _maybe_load_2d_intervals_set(intervals, [expr], iterator, band)
    if _is_2d_intervals(intervals):
        extracted = gextract(
            expr,
            intervals=intervals,
            iterator=iterator,
            band=band,
            vars=caller_ns,
            progress=progress,
            progress_desc=progress_desc,
        )
        if extracted is None or len(extracted) == 0:
            return None
        data_cols = [
            c
            for c in extracted.columns
            if c not in {"chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"}
        ]
        if not data_cols:
            return None
        vals = extracted[data_cols[0]].to_numpy(dtype=float, copy=False)
        mask = (~_numpy.isnan(vals)) & (vals != 0.0)
        if not mask.any():
            return None
        result_2d = extracted.loc[mask, ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].reset_index(drop=True)
        intervals_set_out = kwargs.get("intervals_set_out")
        if intervals_set_out is not None:
            from .intervals import gintervals_save

            gintervals_save(result_2d, intervals_set_out)
            return None
        return result_2d

    # Handle DataFrame-as-iterator for 1D gscreen
    intervals, iterator, _scr_id_map = _preprocess_intervals_iterator(intervals, iterator)

    track_names = _track_names_set()
    vtrack_names = set(_shared._VTRACKS.keys())
    expr_eval, expr_tracks, expr_vtracks, _ = _parse_expr_vars(expr, track_names, vtrack_names)

    # Resolve user variables from the caller's namespace
    user_vars = _resolve_user_vars(expr_eval, caller_ns)

    allowed_names = {
        "np",
        "numpy",
        "CHROM",
        "START",
        "END",
        *(_expr_safe_name(t) for t in expr_tracks),
        *(_expr_safe_name(vt) for vt in expr_vtracks),
    }
    allowed_names |= set(user_vars.keys())
    try:
        compile_safe_expression(expr_eval, allowed_names)
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    # Check if vtracks can go through C++ path
    cpp_vtracks = expr_vtracks and _can_vtracks_use_cpp(expr_vtracks) and not user_vars

    if (not expr_vtracks and not user_vars) or cpp_vtracks:
        vtracks_dict = _build_vtracks_dict(expr_vtracks) if cpp_vtracks else None
        with _config_no_mt(_scr_id_map) as _cfg, _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_screen(
                expr,
                _df2pymisha(intervals),
                iterator,
                _cfg,
                vtracks_dict,
            )
        df = _pymisha2df(result)
        intervals_set_out = kwargs.get("intervals_set_out")
        if df is not None and intervals_set_out is not None:
            from .intervals import gintervals_save

            gintervals_save(df[["chrom", "start", "end"]], intervals_set_out)
            return None
        return df

    track_arrays = {}
    base_df = None
    iter_df = None

    if expr_tracks:
        track_exprs = list(expr_tracks)
        with _config_no_mt(_scr_id_map) as _cfg:
            base_result = _pymisha.pm_extract(track_exprs, _df2pymisha(intervals), iterator, _cfg)
        base_df = _pymisha2df(base_result)
        if base_df is None:
            raise RuntimeError("Failed to extract physical track values for mixed expression")
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            if col not in base_df.columns:
                raise KeyError(f"Track column not found for '{tname}'")
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)
        iter_df = base_df[["chrom", "start", "end", "intervalID"]]
    else:
        if iterator is None:
            raise ValueError(
                f'Cannot implicitly determine iterator policy:\ntrack expression "{expr}" does not contain any tracks.'
            )
        iter_df = _iterated_intervals(intervals, iterator)

    if iter_df is None or len(iter_df) == 0:
        return None

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)  # type: ignore[call-overload]
    mask = _numpy.zeros(n_rows, dtype=bool)

    chrom_vals = iter_df["chrom"].to_numpy()
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    code_obj = compile_safe_expression(expr_eval, allowed_names)

    with _progress_context(progress, total=n_rows, desc=progress_desc) as progress_cb:
        for start_idx, end_idx in _chunk_slices(n_rows, chunk_size):
            sl = slice(start_idx, end_idx)
            local_ns = {
                "np": _numpy,
                "numpy": _numpy,
                "CHROM": chrom_vals[sl],
                "START": start_vals[sl],
                "END": end_vals[sl],
            }
            for tname, arr in track_arrays.items():
                local_ns[_expr_safe_name(tname)] = arr[sl]

            vtrack_arrays = {}
            if expr_vtracks:
                chunk_intervals = iter_df.iloc[start_idx:end_idx][["chrom", "start", "end"]]
                for vt in expr_vtracks:
                    vtrack_arrays[vt] = _compute_vtrack_values(vt, chunk_intervals)
                    local_ns[_expr_safe_name(vt)] = vtrack_arrays[vt]

            local_ns.update(user_vars)

            chunk_mask = eval(code_obj, {"__builtins__": {}}, local_ns)
            chunk_mask = _numpy.asarray(chunk_mask, dtype=bool)

            for vt in expr_vtracks:
                chunk_mask = _numpy.where(_numpy.isnan(vtrack_arrays[vt]), False, chunk_mask)

            mask[sl] = chunk_mask

            if progress_cb:
                total = n_rows
                done = end_idx
                pct = int(done * 100.0 / total) if total else 100
                progress_cb(done, total, pct)

        if progress_cb:
            progress_cb(n_rows, n_rows, 100)

    if not mask.any():
        return None

    out_rows = []
    prev_chrom = None
    prev_start = None
    prev_end = None

    for idx in _numpy.where(mask)[0]:
        chrom = chrom_vals[idx]
        start = start_vals[idx]
        end = end_vals[idx]
        if prev_chrom is not None and chrom == prev_chrom and prev_end == start:
            prev_end = end
        else:
            if prev_chrom is not None:
                out_rows.append((prev_chrom, prev_start, prev_end))
            prev_chrom, prev_start, prev_end = chrom, start, end

    if prev_chrom is not None:
        out_rows.append((prev_chrom, prev_start, prev_end))

    if not out_rows:
        return None

    filtered = _pandas.DataFrame(out_rows, columns=["chrom", "start", "end"])
    filtered = filtered.reset_index(drop=True)
    intervals_set_out = kwargs.get("intervals_set_out")
    if intervals_set_out is not None:
        from .intervals import gintervals_save

        gintervals_save(filtered, intervals_set_out)
        return None
    return filtered
