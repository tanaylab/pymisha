"""gextract and gscreen implementations."""

from __future__ import annotations

import collections.abc
import os
from typing import Any

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


def _apply_extract_output(
    df: pd.DataFrame | None, file: str | None, intervals_set_out: str | None, *, is_2d: bool = False
) -> pd.DataFrame | None:
    """Apply file-writing and intervals_set_out post-processing to an extraction result.

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

    Returns
    -------
    DataFrame or None
    """
    if df is None:
        # Nothing to write; still honour the file contract (return None).
        if file is not None:
            return None
        return None

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

    from .tracks import _check_computed_tracks

    _check_computed_tracks(exprs)

    file = kwargs.get("file")
    intervals_set_out = kwargs.get("intervals_set_out")

    if intervals is None:
        from .intervals import gintervals_all

        intervals = gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)
    intervals = _maybe_load_2d_intervals_set(intervals, exprs, iterator, band)

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
        return _apply_extract_output(df, file, intervals_set_out, is_2d=True)

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
        return _apply_extract_output(df, file, intervals_set_out, is_2d=False)

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
            if len(exprs) == 1:
                raise ValueError(
                    f"Cannot implicitly determine iterator policy:\n"
                    f'track expression "{exprs[0]}" does not contain any tracks.'
                )
            raise ValueError(
                "Cannot implicitly determine iterator policy: track expressions do not contain any tracks."
            )
        iter_df = _iterated_intervals(intervals, iterator)

    if iter_df is None or len(iter_df) == 0:
        return _apply_extract_output(None, file, intervals_set_out, is_2d=False)

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
    return _apply_extract_output(result_df, file, intervals_set_out, is_2d=False)


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
