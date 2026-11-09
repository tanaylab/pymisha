"""gextract and gscreen implementations."""

from __future__ import annotations

import collections.abc
import contextlib
import functools
import math
import numbers
import os
import re
import socket
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ._iterator_policy import FixedRectPolicy

import pandas as pd

from . import _shared
from ._log import get_logger
from ._safe_eval import UnsafeExpressionError, compile_safe_expression
from ._shared import (
    CONFIG,
    _checkroot,
    _chrom_names,
    _chunk_slices,
    _config_multitasking,
    _config_no_mt,
    _dead_worker_message,
    _df2pymisha,
    _iterated_intervals,
    _numpy,
    _one_random_stream,
    _pandas,
    _preprocess_intervals_iterator,
    _progress_context,
    _pymisha,
    _pymisha2df,
    _remap_interval_ids,
    _run_forked,
    _sort_by_chrom,
    _track_names_set,
    _with_canonical_chroms,
)
from ._types import Iterator
from .expr import _caller_namespace, _expr_safe_name, _parse_expr_vars, _resolve_user_vars
from .vtracks import _array_slice_spec, _compute_vtrack_values

_logger = get_logger(__name__)


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
    1D vtrack, COMPUTED 2D track, etc.).
    """
    from .tracks import gtrack_info

    track_names_now = _track_names_set()
    vtrack_names_now = set(_shared._VTRACKS.keys())

    resolved: list[tuple[str, str, object, int, int, int, int]] = []
    for expr in exprs:
        if expr in track_names_now and expr not in vtrack_names_now:
            # COMPUTED 2D tracks need the Python read path (the C++ scanner
            # has no Computer2D port yet); a 1D track is refused there.
            try:
                info = gtrack_info(expr)
                if info.get("type") == "computed" or int(info.get("dimensions", 1) or 1) != 2:
                    return None
            except (_pymisha.error, ValueError):
                _logger.debug("no track info for %r; leaving it to the legacy path", expr,
                              exc_info=True)
                return None
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
                except (_pymisha.error, ValueError):
                    _logger.debug("no track info for %r; leaving it to the legacy path",
                                  orig_name, exc_info=True)
                    return None
                if int(info.get("dimensions", 1) or 1) != 2:
                    return None  # 1D bare-track ref in a 2D expression
                if info.get("type") == "computed":
                    return None  # C++ scanner has no Computer2D port yet
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
    c1_arr = _chrom_names(intervals["chrom1"], "chrom1").to_numpy()
    c2_arr = _chrom_names(intervals["chrom2"], "chrom2").to_numpy()
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


def _check_seq_vtrack_iterator(exprs, used_vtracks) -> None:
    """R's error for a sequence-based vtrack (no source track) with no iterator:
    it has no bin size to iterate by."""
    for vt in used_vtracks or ():
        cfg = _shared._VTRACKS.get(vt)
        if cfg is not None and cfg.get("src") is None:
            expr = next((e for e in exprs if vt in str(e)), vt)
            raise ValueError(
                "Cannot implicitly determine iterator policy:\n"
                f'track expression "{expr}" contains a sequence-based virtual track '
                "(pwm, potts, kmer or masked) or tracks in different formats."
            )


def _check_dense_bin_sizes(exprs, used_vtracks) -> None:
    """R's error for an implicit iterator over dense tracks of different bin sizes: those in
    the expressions and the sources of their virtual tracks (an explicit iterator reads them)."""
    from .expr import _parse_expr_vars
    from .tracks import gtrack_exists, gtrack_info

    track_names = _shared._track_names_set()
    sources: set[str] = set()
    for e in exprs:
        _, expr_tracks, _, _ = _parse_expr_vars(str(e), track_names, set(_shared._VTRACKS))
        sources.update(expr_tracks)
    for vt in used_vtracks or ():
        src = (_shared._VTRACKS.get(vt) or {}).get("src")
        if isinstance(src, str):
            sources.add(src)
    sizes = set()
    for src in sources:
        if not gtrack_exists(src):
            continue
        info = gtrack_info(src)
        bs = info.get("bin_size") or info.get("bin.size")
        if info.get("type") == "dense" and bs is not None:
            sizes.add(int(float(bs)))
    if len(sizes) > 1:
        if len(exprs) == 1:
            raise ValueError("Cannot implicitly determine iterator policy:\n"
                             f'track expression "{exprs[0]}" contains tracks with different bin sizes.')
        raise ValueError("Cannot implicitly determine iterator policy: track expressions contain tracks with "
                         "different bin sizes.")


def _default_vtrack_iterator(intervals, iterator, exprs, used_vtracks):
    """R's default iterator for an expression whose only sources are vtracks:
    a sequence vtrack is an error, a value vtrack iterates its source track's
    bins (or array bins). Returns (intervals, iterator); intervals change only
    for an array source, whose bins become the scope."""
    if iterator is not None or not used_vtracks:
        return intervals, iterator
    _check_seq_vtrack_iterator(exprs, used_vtracks)
    _check_dense_bin_sizes(exprs, used_vtracks)
    it = _infer_iterator_from_vtracks(used_vtracks)
    if isinstance(it, str):
        intervals, it, _ = _preprocess_intervals_iterator(intervals, it)
    return intervals, it


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
    "pwm.n_mutations",
    "potts",
    "potts.max",
    "potts.max.pos",
    "potts.count",
    "kmer.count",
    "kmer.frac",
    "masked.count",
    "masked.frac",
}
_CPP_VALUE_FUNCS = {
    "avg",
    "mean",
    "sum",
    "weighted.sum",
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

# Column-reduction functions supported by an array-slice virtual track in C++
# (GenomeTrackArray::SliceFunctions).
_CPP_ARRAY_SLICE_FUNCS = {"avg", "min", "max", "sum", "stddev", "quantile"}


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
        # Array-slice vtracks evaluate in C++: the source array track is read by
        # the scanner with the configured column slice + reduction, and the
        # vtrack's own function as any other.
        if cfg.get("kind") == "array_slice":
            sfunc = str((cfg.get("slice") or {}).get("func", "avg")).lower()
            if sfunc not in _CPP_ARRAY_SLICE_FUNCS:
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


def _infer_iterator_from_vtracks(vtrack_names: set[str] | list[str]) -> int | str | None:
    """Infer the implicit iterator from value-based vtrack sources.

    R determines the implicit iterator of a virtual-track expression from the
    vtrack's *source* track. For value-based vtracks over dense 1D tracks this
    is the source track's native bin size; for an array source it is the array
    track itself (iterate its bins); for a 2D source it is the 2D track itself
    (iterate its rects). Returns:

    - an ``int`` bin size when every vtrack is a value vtrack over a dense 1D
      track sharing one bin size,
    - the source track **name** (``str``) when a single array-source or
      2D-source vtrack drives the expression - the caller resolves it to the
      array's bins / the 2D track's rects,
    - ``None`` otherwise (sparse / sequence / mixed sources unchanged).
    """
    from .tracks import gtrack_exists, gtrack_info

    bin_sizes: set[int] = set()
    array_srcs: set[str] = set()
    for name in vtrack_names:
        cfg = _shared._VTRACKS.get(name)
        if cfg is None:
            return None
        src = cfg.get("src")
        if not isinstance(src, str) or not gtrack_exists(src):
            return None
        info = gtrack_info(src)
        dims = int(info.get("dimensions", 1) or 1)
        if dims == 2:
            # A 2D source is resolved by _gextract_2d instead, which iterates
            # the source's objects as R does (_expr_2d_track_sources).
            return None
        if dims != 1:
            return None
        if info.get("type") == "array":
            array_srcs.add(src)
            continue
        bs = info.get("bin_size") or info.get("bin.size")
        if bs is None:
            return None  # sparse source: native iterator deferred
        bin_sizes.add(int(float(bs)))
    # An array source iterates its own bins; only resolvable when it is the
    # sole source (a single array track name).
    if array_srcs:
        if len(array_srcs) == 1 and not bin_sizes:
            return array_srcs.pop()
        return None
    if len(bin_sizes) == 1:
        return bin_sizes.pop()
    return None


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
        # An array-slice vtrack: the slice as the C++ spec keys
        # (configure_array_slice); the vtrack's func, shifts and params as any other
        if spec.pop("kind", None) == "array_slice":
            spec.update(_array_slice_spec(cfg))
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

    # R parity: a track name used as the scope means "the intervals over which
    # the track is defined" (1D: its intervals; 2D: its rectangles/points).
    from .tracks import gtrack_exists, gtrack_info

    if gtrack_exists(intervals):
        info = gtrack_info(intervals)
        if int(info.get("dimensions", 1) or 1) == 2:
            from .intervals import _gintervals_2d_all

            res = gextract(intervals, _gintervals_2d_all("full"))
            if res is None or len(res) == 0:
                return intervals
            cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
            return res[cols].drop_duplicates().reset_index(drop=True)

        from .intervals import gintervals_all

        res = gextract(intervals, gintervals_all())
        if res is None or len(res) == 0:
            return intervals
        return res[["chrom", "start", "end"]].drop_duplicates().reset_index(drop=True)

    return intervals


def _find_2d_track_file(track_path: str, c1: str, c2: str) -> str | None:
    """The file of a 2D track's chromosome pair, or None, as every reader finds it
    (GenomeTrack::ChromFiles): named by the canonical names, as R misha names it, or by
    aliases, as pymisha 0.14 and earlier named it in a per-chromosome database ("1-2"
    for chr1-chr2)."""
    try:
        n1, n2 = _pymisha.pm_normalize_chroms([c1, c2])   # c1, c2 may be aliases
    except _pymisha.error:
        return None
    path = os.path.join(track_path, f"{n1}-{n2}")
    if os.path.isfile(path):
        return path
    files = _pymisha.pm_chrom_files(track_path, True)
    for f1, f2, name, used in zip(files["chrom1"], files["chrom2"], files["file"], files["used"], strict=True):
        if used and (f1, f2) == (n1, n2):
            return os.path.join(track_path, name)
    return None


def _validate_band(band: tuple[int, int] | tuple[float, float] | list[int] | None) -> tuple[int, int] | None:
    """Validate band parameter. Returns (d1, d2) tuple or None.

    As R: each distance a finite number within +-1e18, an integer as is and a
    fraction rounded half away from zero."""
    if band is None:
        return None
    if isinstance(band, (str, bytes)) or not hasattr(band, "__len__") or len(band) != 2:
        raise ValueError("band must be a sequence of length 2: (d1, d2)")
    dists: list[int] = []
    for v in band:
        if v is None or v is _pandas.NA:
            v = float("nan")
        if isinstance(v, (bool, _numpy.bool_)) or not isinstance(v, numbers.Real):
            raise ValueError("Invalid format of band argument")
        if isinstance(v, numbers.Integral):
            ok = abs(int(v)) <= 10**18
        else:
            v = float(v)
            ok = math.isfinite(v) and abs(v) <= 1e18
        if not ok:
            raise ValueError("Invalid band argument: band distances must be finite numbers between -1e18 and 1e18")
        dists.append(int(v) if isinstance(v, numbers.Integral) else int(v + 0.5) if v > 0 else int(v - 0.5))
    d1, d2 = dists
    if d1 >= d2:
        raise ValueError("Invalid band argument: distance1 exceeds distance2")
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

    COMPUTED tracks (R's ``GenomeTrackComputed``) hit a different per-rect
    emit path: each stored ``Computed_val<float>`` is clipped to the
    intersection of the query rect and (if present) the band, then the
    cached value is *recomputed* via the Computer2D mirroring R's
    ``Computed_val::val(rect, [band])``.  Pymisha emits the clipped coords
    and the recomputed value to match R.
    """
    from ._computer2d import (
        DiagonalBand as _DBand,
    )
    from ._computer2d import (
        Rectangle as _Rect,
    )
    from ._computer2d import (
        load_computer_from_header,
        recompute_or_cached,
    )
    from ._quadtree import (
        SIGNATURE_COMPUTED,
        open_2d_pair,
        query_2d_track_opened,
    )
    from .tracks import gtrack_info

    intervals = _with_canonical_chroms(intervals)   # the database's names, as the C++ path gives
    track_path = _pymisha.pm_track_path(track)
    info = gtrack_info(track)
    is_points = info.get("type") == "points"
    is_computed = info.get("type") == "computed"
    band_obj = _DBand(int(band[0]), int(band[1])) if band is not None else None

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

            # COMPUTED: parse the Computer2D header once per chrom-pair so the
            # per-rect recompute can call computer.compute(clipped_rect, band).
            computer = None
            if is_computed:
                import struct as _struct
                if _struct.unpack_from("<i", data, 0)[0] == SIGNATURE_COMPUTED:
                    computer = load_computer_from_header(data)

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
                        if computer is not None:
                            # Clip the stored rect to the query rect, then to
                            # the diagonal band (R's shrink2intersected); recompute
                            # via the Computer2D mirroring R's val(rect, band).
                            cx1 = max(int(ox1), int(s1))
                            cy1 = max(int(oy1), int(s2))
                            cx2 = min(int(ox2), int(e1))
                            cy2 = min(int(oy2), int(e2))
                            query_rect = _Rect(cx1, cy1, cx2, cy2)
                            if band_obj is not None and band_obj.active:
                                query_rect = band_obj.shrink2intersected(query_rect)
                            stored_rect = _Rect(int(ox1), int(oy1), int(ox2), int(oy2))
                            new_val = recompute_or_cached(
                                stored_rect, float(val), query_rect, computer, band_obj,
                            )
                            rows.append(
                                (c1, query_rect.x1, query_rect.x2,
                                 c2, query_rect.y1, query_rect.y2,
                                 float(new_val), interval_idx)
                            )
                        else:
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
    return _sort_by_chrom(result, ["chrom1", "start1", "chrom2", "start2", "intervalID"])


def _gextract_2d_single(
    track: str,
    col_name: str,
    intervals: pd.DataFrame,
    band: tuple[int, int] | None,
    *,
    _verified: bool = False,
) -> pd.DataFrame | None:
    """Extract objects from a 2D track via the native C++ binding.

    Returns a DataFrame with chrom1/start1/end1/chrom2/start2/end2/<col_name>/
    intervalID columns, sorted by (chrom1, start1, chrom2, start2, intervalID).
    Returns None if no objects intersect any interval.

    ``_verified=True`` says the caller already ran ``_verify_2d_intervals`` on
    this exact frame, or built it so that it cannot fail (``_apply_2d_shifts``
    clamps).  Internal only.  Validation costs ~0.27 ms per call regardless of
    row count, and a streamed job pays it once per row on top of the caller's
    own check.
    """
    from .intervals import _chrom_id_map, _verify_2d_intervals
    from .tracks import gtrack_info

    n = len(intervals)
    if n == 0:
        return None

    # R parity: validate here too (not just in the caller, _gextract_2d) so
    # direct callers of this function -- it is exercised directly by
    # test_extract_2d_fast.py and test_2d_band_cpp.py -- are covered as well.
    # _gextract_2d passes _verified=True: it validates the scope itself, and
    # the shifted frames it derives are clamped by construction, so they pass
    # this check rather than being rejected by it (R clamps a shifted iterator
    # rectangle; it does not reject one).
    if not _verified:
        _verify_2d_intervals(intervals)

    # COMPUTED 2D tracks: the C++ pm_extract_2d has no Computer2D dispatcher
    # port yet.  Read the per-rect cached value via the pure-Python path
    # which uses _quadtree.open_2d_pair (signature -11 aware).  Per-rect
    # recompute fallback (when a query doesn't exactly match a stored rect)
    # is handled inside _gextract_2d_single_python via the same data
    # structures that already read the cached `Computed_val.v`.
    # Only the probe is guarded: `track` may be an expression rather than a
    # track name.  The read itself must not be, or a genuine failure falls
    # through to a C++ path that cannot serve COMPUTED tracks at all.
    try:
        _is_computed = gtrack_info(track).get("type") == "computed"
    except (_pymisha.error, ValueError):
        # *track* is an expression rather than a track name.
        _logger.debug("no track info for %r; not a COMPUTED track", track, exc_info=True)
        _is_computed = False
    if _is_computed:
        return _gextract_2d_single_python(track, col_name, intervals, band)

    # chromosome ids by name, an alias (such as "1" for chr1) included
    intervals = _with_canonical_chroms(intervals)
    cmap = _chrom_id_map()
    chrom1_ids = (
        _pandas.Series(_chrom_names(intervals["chrom1"], "chrom1").to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )
    chrom2_ids = (
        _pandas.Series(_chrom_names(intervals["chrom2"], "chrom2").to_numpy())
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
    return _sort_by_chrom(out, ["chrom1", "start1", "chrom2", "start2", "intervalID"])


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
    except (_pymisha.error, ValueError):
        _logger.debug("no track info for vtrack source %r; not scanner-eligible", src,
                      exc_info=True)
        return None
    if int(info.get("dimensions", 1) or 1) != 2:
        return None  # 1D source track; can't use 2D scanner
    if info.get("type") == "computed":
        return None  # C++ scanner has no Computer2D port yet

    # 1D iterator shifts (sshift/eshift, no axis suffix) are invalid on a
    # 2D-source vtrack -- the legacy path rejects them with the same message;
    # mirror that here so the scanner path also raises instead of silently
    # ignoring the shifts.
    if int(cfg.get("sshift", 0) or 0) != 0 or int(cfg.get("eshift", 0) or 0) != 0:
        raise ValueError(
            f"2D extraction for virtual track '{expr}' does not support "
            "1D iterator shifts"
        )

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

    # R's messages: sequence, interval and value based vtracks and 1D track sources need dim=
    src = cfg.get("src")
    if not isinstance(src, str):
        raise ValueError(f"Virtual track {vtrack_name}: 2D iterator is used without explicit dimension projection")

    info = gtrack_info(src)
    if int(info.get("dimensions", 1) or 1) != 2:
        raise ValueError(
            f"Virtual track {vtrack_name}: 2D iterator is applied to a 1D track {src} "
            "without explicit dimension projection"
        )

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
    except (_pymisha.error, ValueError):
        # Not a saved interval-set name: pass the value through untouched.
        _logger.debug("could not load %r as an interval set", intervals, exc_info=True)
        return intervals
    if _is_2d_intervals(loaded):
        return loaded
    return intervals


def _apply_2d_shifts(
    intervals: pd.DataFrame, sshift1: int, eshift1: int, sshift2: int, eshift2: int
) -> tuple[pd.DataFrame, Any]:
    """Apply 2D iterator shifts to interval coordinates, clamped to the chromosome.

    R parity: ``TrackExpressionVars::Iterator_modifier2D::transform``
    (misha/src/TrackExpressionVars.h) computes
    ``max(start + sshift, 0)`` / ``min(end + eshift, chrom_size)`` and only flags
    the rectangle ``out_of_range`` when the clamp collapses it
    (``start >= end``), in which case the variable's value is NaN
    (misha/src/TrackVarProcessor.cpp).  A shift that runs off the end of a
    chromosome is never an error in R.

    Returns ``(shifted, kept)``: ``kept`` is the positional index into
    ``intervals`` of the rows that survived the clamp, or ``None`` when every
    row survived (the common case, and the only one with no shifts at all).
    Callers must map per-row results back through ``kept`` - see
    ``_scatter_shifted_values``.
    """
    if sshift1 == 0 and eshift1 == 0 and sshift2 == 0 and eshift2 == 0:
        return intervals, None

    from .intervals import _chrom_sizes_for_2d_verify, _verify_2d_intervals

    np = _numpy
    start1 = _pandas.to_numeric(intervals["start1"], errors="coerce").to_numpy(dtype="float64") + sshift1
    end1 = _pandas.to_numeric(intervals["end1"], errors="coerce").to_numpy(dtype="float64") + eshift1
    start2 = _pandas.to_numeric(intervals["start2"], errors="coerce").to_numpy(dtype="float64") + sshift2
    end2 = _pandas.to_numeric(intervals["end2"], errors="coerce").to_numpy(dtype="float64") + eshift2

    if np.isnan(start1).any() or np.isnan(end1).any() or np.isnan(start2).any() or np.isnan(end2).any():
        # Defensive: every caller validates the scope before shifting it, so a
        # NaN here means the frame was never checked. Raise the normal message.
        _verify_2d_intervals(intervals)

    chrom1 = _chrom_names(intervals["chrom1"], "chrom1").to_numpy()
    chrom2 = _chrom_names(intervals["chrom2"], "chrom2").to_numpy()

    np.maximum(start1, 0.0, out=start1)
    np.maximum(start2, 0.0, out=start2)

    chrom_sizes = _chrom_sizes_for_2d_verify()
    if chrom_sizes:
        # Unknown chrom label -> no upper bound, matching the boundary check in
        # _verify_2d_intervals, which is best-effort for the same reason.
        size1 = np.fromiter((chrom_sizes.get(c, np.inf) for c in chrom1), dtype="float64", count=len(chrom1))
        size2 = np.fromiter((chrom_sizes.get(c, np.inf) for c in chrom2), dtype="float64", count=len(chrom2))
        np.minimum(end1, size1, out=end1)
        np.minimum(end2, size2, out=end2)

    shifted = intervals.copy()
    shifted["start1"] = start1.astype("int64")
    shifted["end1"] = end1.astype("int64")
    shifted["start2"] = start2.astype("int64")
    shifted["end2"] = end2.astype("int64")

    keep = (start1 < end1) & (start2 < end2)
    if keep.all():
        return shifted, None
    kept = np.flatnonzero(keep)
    return shifted.iloc[kept].reset_index(drop=True), kept


def _scatter_shifted_values(values: Any, kept: Any, n: int) -> Any:
    """Realign per-row values computed on a clamped shifted frame to length ``n``.

    Rows dropped by ``_apply_2d_shifts`` (the clamp collapsed the rectangle)
    get NaN, which is what R's ``out_of_range`` produces for the same input.
    """
    arr = _numpy.asarray(values, dtype=float)
    if kept is None:
        return arr
    out = _numpy.full(n, _numpy.nan, dtype=float)
    out[kept] = arr
    return out


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
    # A rect that meets no cell, or a pair the track has no file for, has area 0
    # and NaN for the rest, as in R (StatQuadTreeCached::get_stat and
    # GenomeTrackRects::read_interval).
    values = _numpy.full(n, 0.0 if func == "area" else _numpy.nan, dtype=float)

    # Group intervals by (chrom1, chrom2) to open each file only once.
    chrom_pair_intervals = _group_intervals_by_chrom_pair(intervals)

    for (c1, c2), interval_list in chrom_pair_intervals.items():
        pair = open_2d_pair(track_path, c1, c2)
        if pair is None:
            # No data for this chrom pair.
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

    from ._quadtree import open_2d_pair, query_2d_track_opened
    from .tracks import gtrack_info

    intervals = _with_canonical_chroms(intervals)   # the database's names, as the C++ path gives
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
                    chosen = objs[int(_numpy.random.randint(len(objs)))]  # numpy's global generator
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

    # chromosome ids by name, an alias (such as "1" for chr1) included
    intervals = _with_canonical_chroms(intervals)
    cmap = _chrom_id_map()
    chrom1_ids = (
        _pandas.Series(_chrom_names(intervals["chrom1"], "chrom1").to_numpy())
        .map(cmap)
        .fillna(-1)
        .astype(_numpy.int32)
        .to_numpy()
    )
    chrom2_ids = (
        _pandas.Series(_chrom_names(intervals["chrom2"], "chrom2").to_numpy())
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

    # "sample" draws as every other random draw of the C++ layer (seeded from numpy's generator)
    result = _pymisha.pm_extract_2d_objects(track, intervals_dict, band, func, 0)

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


def _gextract_2d_via_scanner(
    exprs: list[str],
    intervals: pd.DataFrame,
    policy: FixedRectPolicy | Any,
    *,
    colnames: list[str] | None,
    band: tuple[int, int] | None,
    resolved_vars: list[tuple[str, str, object, int, int, int, int]] | None = None,
    iterator_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run a 2D extract through pm_extract_2d_scanner (Intervals, FixedRect, TrackRects, or CartesianGrid).

    Supports bare 2D track names as expressions and, when ``resolved_vars`` is
    provided, reducing 2D vtracks.  Each bare track defaults to "avg".

    Parameters
    ----------
    resolved_vars : list of (physical_track, func, params, ss1, es1, ss2, es2) or None
        When provided, one entry per expression.  The physical_track, func, and
        shifts are passed directly to the C++ scanner; params is currently unused.
        When None, all ``exprs`` are treated as bare physical track names with
        "avg" aggregation and zero shifts.
    iterator_intervals : DataFrame or None
        The 2D intervals of an intervals iterator (``IntervalsPolicy``), cut by
        the scope as R cuts them. None makes the scope itself the iterator.

    Returns a DataFrame with chrom1, start1, end1, chrom2, start2, end2,
    <colname>, intervalID columns - matching the shape returned by the
    regular _gextract_2d path. Rows come in the order R's 2D iterator walks
    them, and intervalID is the 1-based row of the scope interval a row comes
    from, as in R.
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
        if iterator_intervals is not None:
            policy_dict["intervals"] = _2d_frame_to_scanner_dict(iterator_intervals)
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
        from .intervals import _normalize_chroms

        cmap = _chrom_id_map()

        def _df_to_intervals_dict(df: pd.DataFrame | None) -> dict[str, Any] | None:
            if df is None:
                return None
            # Normalize chr-prefix to match the DB's storage convention; see the
            # rationale on the IntervalsPolicy / TrackRectsPolicy chrom mapping
            # block below.
            chrom_strs = _normalize_chroms(_chrom_names(df["chrom"]).tolist())
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
    # ``_normalize_chroms`` first maps the input to the DB's storage convention
    # (e.g. ``chr1 -> 1`` or vice versa), so callers passing the wrong prefix
    # don't end up with ``chromid=-1`` -- which combined with a registered
    # vtrack would crash the C++ scanner with ``std::bad_alloc``.
    cmap = _chrom_id_map()
    scope_dict = _2d_frame_to_scanner_dict(intervals)

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
    out["intervalID"] = _numpy.asarray(result["_scope_idx"], dtype=int) + 1
    return out


def _2d_frame_to_scanner_dict(intervals: pd.DataFrame) -> dict[str, Any]:
    """2D intervals as the six arrays pm_extract_2d_scanner takes (chrom ids, -1 if unknown)."""
    from .intervals import _chrom_id_map, _normalize_chroms

    cmap = _chrom_id_map()

    def _ids(col: str) -> Any:
        return (
            _pandas.Series(_normalize_chroms(_chrom_names(intervals[col], col).tolist()))
            .map(cmap)
            .fillna(-1)
            .astype(_numpy.int32)
            .to_numpy()
        )

    return {
        "chrom1": _ids("chrom1"),
        "start1": intervals["start1"].to_numpy(dtype=_numpy.int64),
        "end1": intervals["end1"].to_numpy(dtype=_numpy.int64),
        "chrom2": _ids("chrom2"),
        "start2": intervals["start2"].to_numpy(dtype=_numpy.int64),
        "end2": intervals["end2"].to_numpy(dtype=_numpy.int64),
    }


def _gextract_2d_compound_via_scanner(
    exprs: list[str],
    intervals: pd.DataFrame,
    policy: object,
    *,
    colnames: list[str] | None,
    band: tuple[int, int] | None,
    eval_specs: list[tuple[str, dict[str, str]]],
    var_specs: list[tuple[str, tuple[str, str, object, int, int, int, int]]],
    iterator_intervals: pd.DataFrame | None = None,
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
        iterator_intervals=iterator_intervals,
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


def _expr_2d_track_sources(exprs: list[str]) -> set[str]:
    """The 2D tracks the expressions read: bare 2D tracks and the sources of 2D vtracks."""
    from .tracks import gtrack_exists, gtrack_info

    track_names = _track_names_set()
    vtrack_names = set(_shared._VTRACKS.keys())
    candidates: set[str] = set()
    for e in exprs:
        _, tracks, vtracks, _ = _parse_expr_vars(e, track_names, vtrack_names)
        candidates.update(tracks)
        for vt in vtracks:
            cfg = _shared._VTRACKS.get(vt, {})
            src = cfg.get("src")
            if cfg.get("dim") is None and isinstance(src, str):
                candidates.add(src)
    sources = set()
    for name in candidates:
        try:
            if gtrack_exists(name) and int(gtrack_info(name).get("dimensions", 1) or 1) == 2:
                sources.add(name)
        except (_pymisha.error, ValueError):
            _logger.debug("no track info for %r; not a 2D track", name, exc_info=True)
    return sources


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
    from .intervals import _verify_2d_intervals
    from .tracks import gtrack_info

    band = _validate_band(band)
    explicit_iterator = iterator is not None

    # R parity: R's IntervalConverter runs GInterval2D::verify() on every
    # converted interval; validate the scope here so every dispatch branch
    # below (scanner, vtrack aggregation, and the legacy per-track path) is
    # covered by one check instead of repeating it in each branch.
    _verify_2d_intervals(intervals)

    # R parity: with no iterator R iterates the rects of the one 2D track the
    # expressions read - a bare track or the source of a 2D virtual track
    # (TrackExprScanner::create_expr_iterator): each object cut by the scope.
    # More than one such track is an error, raised by the legacy path below.
    if iterator is None and not _scanner_for_intervals_enabled():
        _sources = _expr_2d_track_sources(exprs)
        if len(_sources) == 1:
            iterator = next(iter(_sources))

    # A 2D interval-set name used as the iterator (not a track) is loaded to its
    # rectangles so the DataFrame branch below routes it through the scalable
    # intersect (R's intervals 2D iterator).  2D *track* iterators keep the
    # TrackRects path further down.
    if isinstance(iterator, str):
        from .tracks import gtrack_exists

        if not gtrack_exists(iterator):
            from .intervals import gintervals_exists

            if gintervals_exists(iterator):
                _loaded_iter = _maybe_load_intervals_set(iterator)
                if isinstance(_loaded_iter, pd.DataFrame) and _is_2d_intervals(_loaded_iter):
                    iterator = _loaded_iter

    # ── 2D intervals DataFrame iterator (K_INTERVALS over iterator ∩ scope) ─
    # R's TrackExpressionIntervals2DIterator walks the iterator rects in R's order
    # and cuts each by a quadtree of the scope, evaluating the expression on each
    # piece; the C++ scanner's intervals policy does the same.
    if (
        iterator is not None
        and isinstance(iterator, pd.DataFrame)
        and _is_2d_intervals(iterator)
    ):
        from ._iterator_policy import IntervalsPolicy

        # A raw 2D DataFrame iterator (as opposed to a saved interval set or
        # track name) reaches the scanner un-validated otherwise.
        _verify_2d_intervals(iterator)

        resolved = _resolve_exprs_for_scanner(exprs)
        if resolved is not None:
            df = _gextract_2d_via_scanner(
                exprs, intervals, IntervalsPolicy(),
                colnames=colnames, band=band, resolved_vars=resolved,
                iterator_intervals=iterator,
            )
            return df if len(df) else None
        compound = _resolve_2d_compound_for_scanner(exprs)
        if compound is not None:
            eval_specs, var_specs = compound
            df = _gextract_2d_compound_via_scanner(
                exprs, intervals, IntervalsPolicy(),
                colnames=colnames, band=band,
                eval_specs=eval_specs, var_specs=var_specs,
                iterator_intervals=iterator,
            )
            return df if len(df) else None
        # vtracks / non-routable expression: fall through to the legacy path.

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
            # Other expressions: the legacy path below, over the iterator's cells.

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
        except (_pymisha.error, ValueError):
            _logger.debug("could not test whether %r is a track", iterator, exc_info=True)
            _iter_exists = False

        if not _iter_exists:
            # Not a track: check if it's a saved interval set (valid legacy use).
            from .intervals import gintervals_exists

            try:
                _is_iset = gintervals_exists(iterator)
            except (_pymisha.error, ValueError):
                _logger.debug("could not test whether %r is an interval set", iterator,
                              exc_info=True)
                _is_iset = False

            if not _is_iset:
                raise _unknown_iterator_error(iterator)
            # It's a saved interval set: fall through to the legacy 2D dispatch.
        else:
            _iter_info = gtrack_info(iterator)
            _iter_type = (
                _iter_info.get("type") if isinstance(_iter_info, dict)
                else getattr(_iter_info, "type", None)
            )

            if _iter_type not in ("rectangles", "points", "computed"):
                raise ValueError("1D iterator is used along with 2D intervals")

            # 2D rects/points/computed track confirmed. Route if all expressions
            # are bare physical tracks or supported reducing 2D vtracks (a
            # computed track is read in Python, so it is not among them);
            # otherwise fall through to the legacy path.
            from ._iterator_policy import TrackRectsPolicy

            resolved = _resolve_exprs_for_scanner(exprs)
            if resolved is not None:
                policy = TrackRectsPolicy(track_name=iterator)
                return _gextract_2d_via_scanner(
                    exprs, intervals, policy, colnames=colnames, band=band,
                    resolved_vars=resolved,
                )
            compound = _resolve_2d_compound_for_scanner(exprs)
            if compound is not None:
                eval_specs, var_specs = compound
                policy = TrackRectsPolicy(track_name=iterator)
                return _gextract_2d_compound_via_scanner(
                    exprs, intervals, policy, colnames=colnames, band=band,
                    eval_specs=eval_specs, var_specs=var_specs,
                )
            # Unsupported vtracks or non-routable compound: fall through to
            # the legacy path, which evaluates them over the iterator's cells.

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
            # Other expressions: the legacy path below, over the iterator's cells.

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
            if explicit_iterator:
                raise ValueError(f"2D iterator is applied to a 1D track {tname} without explicit dimension projection")
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

    # Classify 2D vtracks: aggregation / object / percentile; any other is the
    # source's value, read as the avg over the cell.
    agg_vtracks = {vt for vt in twod_vtracks if vtrack_funcs.get(vt, "") in _2D_AGG_FUNCS}
    obj_vtracks = {vt for vt in twod_vtracks if vtrack_funcs.get(vt, "") in _2D_OBJECT_FUNCS}

    if iterator is None and len(set(used_tracks) | {vtrack_to_track[vt] for vt in twod_vtracks}) > 1:
        raise ValueError(
            "Cannot implicitly determine iterator policy: track expressions contain more than one 2D track."
        )

    coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]

    # The iterator cells, in the order R's 2D iterator walks them, each with the
    # 1-based row of the scope interval it comes from. R evaluates every track
    # and virtual track of the expressions over each cell. With no 2D track to
    # iterate (only dim-projected 1D vtracks) the cells are the scope rows.
    if iterator is not None:
        units = _gextract_2d_units(intervals, iterator, band)
        if units is None or len(units) == 0:
            return None
    else:
        units = intervals[coord_cols].reset_index(drop=True).copy()
        units["intervalID"] = _numpy.arange(1, len(units) + 1, dtype=int)
    cells = units[coord_cols]
    n = len(units)
    values: dict[str, Any] = {}

    def _shifted_cells(shifts: dict[str, int] | None) -> tuple[pd.DataFrame, Any]:
        if not shifts:
            return cells, None
        return _apply_2d_shifts(cells, shifts["sshift1"], shifts["eshift1"], shifts["sshift2"], shifts["eshift2"])

    # Bare 2D tracks: the track's value over the cell, its avg, as R reads it.
    for tname in used_tracks:
        agg_df = _gextract_2d_vtrack_agg(tname, "v", cells, band, "avg")
        values[_expr_safe_name(tname)] = agg_df["v"].to_numpy(dtype=float)

    for vt_name in twod_vtracks:
        src_track = vtrack_to_track[vt_name]
        shifted, kept = _shifted_cells(vtrack_shifts[vt_name])
        func = vtrack_funcs[vt_name]
        if vt_name in agg_vtracks:
            vdf = _gextract_2d_vtrack_agg(src_track, "v", shifted, band, func)
        elif vt_name in obj_vtracks:
            vdf = _gextract_2d_vtrack_objects(src_track, "v", shifted, band, func)
        else:
            vdf = _gextract_2d_vtrack_agg(src_track, "v", shifted, band, "avg")
        values[_expr_safe_name(vt_name)] = _numpy.asarray(_scatter_shifted_values(vdf["v"], kept, n), dtype=float)

    # Dim-projected vtracks (1D source, dim=1 or dim=2): each cell projected onto the dimension.
    for vt_name in dim_vtracks:
        values[_expr_safe_name(vt_name)] = _numpy.asarray(_compute_vtrack_values(vt_name, cells), dtype=float)

    out_cols = colnames if colnames is not None else exprs
    out_df = cells.copy()
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

        local_ns: dict[str, Any] = {"np": _numpy, "numpy": _numpy}
        local_ns.update(user_vars)
        for name in (*expr_tracks, *expr_vtracks):
            local_ns[_expr_safe_name(name)] = values[_expr_safe_name(name)]

        vals = eval(code_obj, {"__builtins__": {}}, local_ns)
        if _numpy.isscalar(vals):
            vals = _numpy.full(n, vals, dtype=float)
        out_df[out_col] = _numpy.asarray(vals, dtype=float)

    out_df["intervalID"] = units["intervalID"].to_numpy(dtype=int)
    return out_df.reset_index(drop=True)


def _gextract_2d_units(
    intervals: pd.DataFrame,
    iterator: Any,
    band: tuple[int, int] | None,
) -> pd.DataFrame | None:
    """The cells of a 2D iterator over a 2D scope, as R's 2D iterator walks them.

    Returns the cell coordinates and, as ``intervalID``, the 1-based row of the
    scope interval each cell comes from. ``iterator`` is a 2D track name, a 2D
    interval-set name, a 2D intervals DataFrame, a ``(width, height)`` pair or
    a :class:`CartesianGridSpec`.
    """
    from ._iterator_policy import CartesianGridSpec, FixedRectPolicy, IntervalsPolicy, TrackRectsPolicy
    from .tracks import gtrack_exists

    policy: Any
    iterator_intervals = None
    if isinstance(iterator, CartesianGridSpec):
        policy = iterator
    elif isinstance(iterator, (tuple, list)) and len(iterator) == 2:
        policy = FixedRectPolicy(width=int(float(iterator[0])), height=int(float(iterator[1])))
    elif isinstance(iterator, pd.DataFrame):
        policy = IntervalsPolicy()
        iterator_intervals = iterator
    elif isinstance(iterator, str) and gtrack_exists(iterator):
        policy = TrackRectsPolicy(track_name=iterator)
    elif isinstance(iterator, str):
        loaded = _maybe_load_intervals_set(iterator)
        if not isinstance(loaded, pd.DataFrame):
            raise _unknown_iterator_error(iterator)
        if not _is_2d_intervals(loaded):
            raise ValueError(f"Invalid iterator: {iterator!r} is not a 2D track or a 2D intervals set.")
        policy = IntervalsPolicy()
        iterator_intervals = loaded
    else:
        raise ValueError(f"Invalid 2D iterator: {iterator!r}")
    return _gextract_2d_via_scanner(
        [], intervals, policy, colnames=None, band=band, resolved_vars=[],
        iterator_intervals=iterator_intervals,
    )


def giterator_intervals_2d(
    expr: str | list[str],
    intervals: pd.DataFrame | str | None = None,
    iterator: Iterator | str = None,
    colnames: str | list[str] | None = None,
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
        to the intra-chromosomal pairs,
        ``gintervals_2d_all(mode="diagonal")``.
    iterator : int or str, optional
        Track expression iterator.  Passed through to the underlying
        extraction engine.
    colnames : str or list of str, optional
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
    if isinstance(colnames, str):
        colnames = [colnames]

    if intervals is None:
        from .intervals import _gintervals_2d_all

        # Intra-chromosomal pairs, as when gintervals_2d_all() defaulted to
        # them. This function has no R counterpart to follow.
        intervals = _gintervals_2d_all("diagonal")

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


def _genome_order(df: pd.DataFrame, scope: pd.DataFrame | None) -> pd.DataFrame:
    """Rows in R's order: R's gextract sorts the scope intervals by chromosome
    and start and scans them in turn, so rows follow their scope interval's
    genome position (ties by intervalID), each interval's rows in scan order."""
    if (scope is None or "intervalID" not in df.columns or "start" not in scope.columns
            or len(df) < 2 or df["intervalID"].isna().any()):
        return df
    from .intervals import _chrom_id_map

    ids = df["intervalID"].to_numpy(dtype=_numpy.int64)
    if ids.min() < 1 or ids.max() > len(scope):
        return df
    cmap = _chrom_id_map()
    scope_chrom = _numpy.asarray(
        _shared._with_canonical_chroms(scope[["chrom", "start", "end"]])["chrom"].astype(str).map(cmap)
        .fillna(len(cmap)), dtype=_numpy.int64)
    scope_start = scope["start"].to_numpy(dtype=_numpy.int64)
    # the usual case - a scope in genome order, rows by interval - needs no sort
    scope_sorted = bool(_numpy.all((_numpy.diff(scope_chrom) > 0)
                                   | ((_numpy.diff(scope_chrom) == 0) & (_numpy.diff(scope_start) >= 0))))
    if scope_sorted and bool(_numpy.all(_numpy.diff(ids) >= 0)):
        return df
    chrom_key = scope_chrom[ids - 1]
    start_key = scope_start[ids - 1]
    order = _numpy.lexsort((ids, start_key, chrom_key))  # stable: scan order within an interval
    if (order == _numpy.arange(len(order))).all():
        return df
    return df.iloc[order].reset_index(drop=True)


def _apply_extract_output(
    df: pd.DataFrame | None,
    intervals_set_out: str | None,
    *,
    is_2d: bool = False,
    intervals_join: str = "id",
    input_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Apply intervals_set_out and intervals_join post-processing to an extraction result held in
    memory (a file= output is written as it is computed: see _extract_to_file).

    Parameters
    ----------
    df : DataFrame or None
        The extraction result.
    intervals_set_out : str or None
        If given, save *df* as a named interval set via :func:`gintervals_save`
        and return ``None`` instead of the DataFrame.
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
        return None

    if not is_2d:
        df = _genome_order(df, input_intervals)

    # Apply intervals_join FIRST, before intervals_set_out, which reads from the post-processed
    # result.
    df = _apply_intervals_join(df, input_intervals, intervals_join, is_2d=is_2d)
    assert df is not None  # _apply_intervals_join preserves non-None input

    # -- intervals_set_out: save coords AND value columns as a named interval set --
    # R parity: the C++ writer behind C_gextract stores the full extraction frame
    # (minus the bookkeeping intervalID column), not a deduplicated coords-only
    # slice. Several R baselines (gintervals.1/.2/.3) depend on this.
    if intervals_set_out is not None:
        from .intervals import gintervals_save

        to_save = df.drop(columns=["intervalID"], errors="ignore").reset_index(drop=True)
        gintervals_save(to_save, intervals_set_out)
        # R parity: with intervals.set.out R misha returns no result (an invisible 0)
        return None
    return df


def _unknown_iterator_error(name: str) -> ValueError:
    """R's error for an iterator that names no track and no intervals set (as the C++
    parse_iterator_policy says it): a name with a character R's names cannot hold is an invalid
    intervals set name."""
    if not all(c.isascii() and (c.isalnum() or c in "_.") for c in name):
        return ValueError(f"Invalid interval name {name}. Only alphanumeric characters and _ are allowed in the name.")
    return ValueError(f"Invalid iterator: {name} is neither a name of a track nor a name of an intervals set")


def _is_parquet_path(file: Any) -> bool:
    return file is not None and str(file).lower().endswith(".parquet")


def _parquet_value_metadata(exprs: list[str]) -> list[list[tuple[str, str]]]:
    """The field metadata of each Parquet value column, as R misha's gextract writes it: the
    attributes of the track, sorted by name, for an expression that is exactly the name of a
    track, none for any other expression."""
    from .tracks import _load_track_attributes

    tracks = _shared._track_names_set()
    return [
        sorted((str(k), str(v)) for k, v in _load_track_attributes(e).items() if str(v) != "")
        if isinstance(e, str) and e in tracks else []
        for e in exprs
    ]


def _parquet_int64_coords(scope: Any) -> bool:
    """Whether start and end are INT64: only if a chromosome of the scope does not fit INT32."""
    from .intervals import gintervals_all

    genome = gintervals_all()
    big = set(genome.loc[genome["end"] > 2**31 - 1, "chrom"].astype(str))
    if not big or not isinstance(scope, pd.DataFrame) or "chrom" not in scope.columns:
        return False
    chroms = list(pd.unique(_chrom_names(scope["chrom"]).to_numpy()))
    try:
        chroms = _pymisha.pm_normalize_chroms(chroms)
    except _pymisha.error:
        _logger.debug("scope chromosomes do not all normalize; comparing them as given", exc_info=True)
    return any(c in big for c in chroms)


def _host_tag() -> str:
    """This host as temporary names carry it, as R misha's .gdb.host_tag() writes it."""
    return re.sub(r"[^A-Za-z0-9_]", "_", socket.gethostname().split(".", 1)[0]) or "unknown"


def _resolve_symlinks(filename: str) -> str:
    """The file a write to *filename* replaces: *filename*, or where its chain of symlinks
    leads, whether that file exists or not."""
    path = filename
    for _ in range(40):
        if not os.path.islink(path):
            return path
        text = os.readlink(path)
        # a relative link is relative to the directory of the link
        path = text if text.startswith("/") else path[: path.rfind("/") + 1] + text
    if os.path.islink(path):
        raise OSError(f"Failed to write {filename}: too many levels of symbolic links")
    return path


def _is_decimal(s: str) -> bool:
    """A number as str() writes it, small enough for a pid."""
    return 0 < len(s) <= 9 and s.isascii() and s.isdigit() and (s[0] != "0" or len(s) == 1)


@contextlib.contextmanager
def _parquet_output(file: Any) -> collections.abc.Iterator[tuple[str, str]]:
    """Where a Parquet output is written, as R misha's gextract(file = "x.parquet") writes it:
    yields (name, tmp). name is what messages call the file; the rows go to tmp, a hidden
    temporary file next to the target (where its symlinks lead), named for this host and
    process, which is renamed onto the target once the block completes: an error or an
    interrupt leaves an existing target as it was. A replaced target passes on its permission
    bits. Leftovers of killed writers of the same target on this host are removed first."""
    name = str(file)
    target = _resolve_symlinks(name)
    directory = target[: target.rfind("/") + 1]
    prefix = f".{target[len(directory):]}.tmp.{_host_tag()}."
    tmp = f"{directory}{prefix}{os.getpid()}"

    try:
        entries = os.listdir(directory or ".")
    except OSError:
        entries = []  # writing tmp fails with the reason
    for entry in entries:
        if not entry.startswith(prefix):
            continue
        # <pid>, or <pid>.<kid>: the parts R misha's multitasking writers add
        pid, dot, kid = entry[len(prefix):].partition(".")
        if not _is_decimal(pid) or (dot and not _is_decimal(kid)):
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            try:
                os.unlink(directory + entry)
            except OSError:
                _logger.debug("could not remove %s", directory + entry, exc_info=True)
        except OSError:
            pass  # alive, as another user's process

    try:
        yield name, tmp
        with contextlib.suppress(OSError):  # a new target keeps the permissions tmp was created with
            os.chmod(tmp, os.stat(target).st_mode & 0o777)
        os.rename(tmp, target)
    except BaseException:
        # BaseException: an interrupt must not leave the temporary file behind either
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _parquet_rows(df: pd.DataFrame | None, value_names: list[str]) -> tuple[Any, Any, Any, Any]:
    """The chroms, starts, ends and values of the rows of a gextract result that its Parquet
    file holds."""
    if df is None:
        return [], _numpy.empty(0, dtype=_numpy.int64), _numpy.empty(0, dtype=_numpy.int64), \
            _numpy.empty((0, len(value_names)))
    # by position: chrom, start, end, then the value columns, whose names may repeat
    # (an error the writer reports) or be one of the coordinates'
    return (
        df.iloc[:, 0].astype(str).to_numpy(dtype=object),
        df.iloc[:, 1].to_numpy(dtype=_numpy.int64),
        df.iloc[:, 2].to_numpy(dtype=_numpy.int64),
        _numpy.ascontiguousarray(df.iloc[:, 3 : 3 + len(value_names)].to_numpy(dtype=_numpy.float64)),
    )


def _parquet_created_by() -> str:
    from . import __version__

    return f"pymisha version {__version__}"


# A gextract(file=) is computed and written a part of the scope at a time (see _file_parts): about
# this many rows a part, as estimated before the extraction
_FILE_PART_ROWS = 1 << 21
# The estimate for an iterator whose rows are not known in advance (other than fixed bins or the
# intervals of a DataFrame iterator): a row for this many bases
_FILE_GUESS_BP_PER_ROW = 64


def _file_parts(
    scope: pd.DataFrame, iterator: Any, binsize: int | None, base_id_map: Any, orig_scope: pd.DataFrame
) -> list[tuple[pd.DataFrame, Any]]:
    """The parts of *scope* (a scope after gextract's iterator preprocessing) that a
    gextract(file=) computes and writes in turn, with the intervalIDs of their rows: each a
    DataFrame of whole scope intervals, or of pieces of a scope interval cut at the bins of a
    fixed-bin iterator (*binsize*), which gives the same bins. Their rows, put in R's order (by
    the scope interval of *orig_scope*, the caller's scope, they come from) and appended in
    turn, are the rows of the whole extraction in that order. *base_id_map* maps the rows of
    *scope* to those of *orig_scope* (None: the same rows)."""
    n = len(scope)
    if not n:
        return []
    gid = (_numpy.arange(1, n + 1, dtype=_numpy.int64) if base_id_map is None
           else _numpy.asarray(base_id_map, dtype=_numpy.int64))
    whole = [(scope, gid)]
    try:
        from .intervals import _chrom_id_map

        starts = scope["start"].to_numpy()
        ends = scope["end"].to_numpy()
        s = starts.astype(_numpy.int64)
        e = ends.astype(_numpy.int64)
        if not (_numpy.array_equal(s, starts) and _numpy.array_equal(e, ends)) or len(gid) != n:
            return whole  # the extraction reports what is wrong with the scope
        cmap = _chrom_id_map()
        orig_chrom = _numpy.asarray(
            _shared._with_canonical_chroms(orig_scope[["chrom", "start", "end"]])["chrom"].astype(str).map(cmap)
            .fillna(len(cmap)), dtype=_numpy.int64)
        orig_start = orig_scope["start"].to_numpy(dtype=_numpy.int64)
        if gid.min() < 1 or gid.max() > len(orig_scope):
            return whole
    except (ValueError, TypeError, KeyError, OverflowError):
        _logger.debug("gextract(file=): the scope is computed in one part", exc_info=True)
        return whole

    if isinstance(iterator, (int, _numpy.integer)) and int(iterator) == -1:
        est = _numpy.ones(n, dtype=_numpy.int64)  # an interval a row
    elif binsize:
        est = _numpy.maximum(-(-e // binsize) - s // binsize, 1)
    else:
        est = _numpy.maximum(-(-(e - s) // _FILE_GUESS_BP_PER_ROW), 1)
    if int(est.sum()) <= _FILE_PART_ROWS:
        return whole

    # R's order: by the caller's scope interval (chromosome, start, row), then as scanned
    order = _numpy.lexsort((e, s, gid, orig_start[gid - 1], orig_chrom[gid - 1]))
    pieces = _numpy.ones(n, dtype=_numpy.int64)
    big = _numpy.flatnonzero(est[order] > _FILE_PART_ROWS) if binsize else _numpy.empty(0, dtype=_numpy.int64)
    # an interval of more rows is cut at the multiples of a whole number of bins
    span = _FILE_PART_ROWS * (binsize or 1)
    cuts = [_numpy.arange((s[r] // span + 1) * span, e[r], span, dtype=_numpy.int64) for r in order[big]]
    pieces[big] += _numpy.array([len(c) for c in cuts], dtype=_numpy.int64)
    unit_rows = _numpy.repeat(order, pieces)
    unit_s = _numpy.repeat(s[order], pieces)
    unit_e = _numpy.repeat(e[order], pieces)
    first = _numpy.cumsum(pieces) - pieces
    for j, c in zip(big, cuts, strict=True):
        unit_s[first[j] + 1 : first[j] + 1 + len(c)] = c
        unit_e[first[j] : first[j] + len(c)] = c
    unit_est = (_numpy.maximum(-(-unit_e // binsize) - unit_s // binsize, 1) if binsize else est[unit_rows])
    unit_gid = gid[unit_rows]

    # Parts of about _FILE_PART_ROWS rows. The rows of one caller's interval that start at the same
    # place stay in one part: the scan orders them, as it does in the whole extraction.
    cum = _numpy.cumsum(unit_est)
    part = (cum - unit_est) // _FILE_PART_ROWS
    split = _numpy.r_[True, (part[1:] != part[:-1])
                      & ~((unit_gid[1:] == unit_gid[:-1]) & (unit_s[1:] == unit_s[:-1]))]
    bounds = _numpy.r_[_numpy.flatnonzero(split), len(unit_rows)]
    parts = []
    for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
        sel = slice(int(b0), int(b1))
        chunk = scope.iloc[unit_rows[sel]].reset_index(drop=True)
        chunk["start"] = unit_s[sel]
        chunk["end"] = unit_e[sel]
        parts.append((chunk, unit_gid[sel]))
    return parts


@contextlib.contextmanager
def _file_extraction(progress: Any, progress_desc: str, num_parts: int) -> collections.abc.Iterator[Any]:
    """The setting of a gextract(file=) computed a part at a time: CONFIG['max_data_size'] does not
    limit the parts, and their extractions report no progress of their own; yields the progress
    callback of the parts (None if none)."""
    cb, close = _shared._make_progress_callback(progress, total=num_parts, desc=progress_desc)
    saved = {k: CONFIG[k] for k in ("max_data_size", "_progress_cb") if k in CONFIG}
    CONFIG["max_data_size"] = None
    CONFIG.pop("_progress_cb", None)
    try:
        yield cb
    finally:
        for k in ("max_data_size", "_progress_cb"):
            if k in saved:
                CONFIG[k] = saved[k]
            else:
                CONFIG.pop(k, None)
        if close:
            close()


def _write_tsv_parts(file: Any, dfs: collections.abc.Iterable[pd.DataFrame], columns: list[str]) -> None:
    """Writes the rows of gextract results to a tab-delimited file as they come, as R misha's
    gextract(file=) writes them: a header, then the rows of each DataFrame, values as R's
    "%.15g", no intervalID. With no rows the file holds the header, *columns*."""
    from pandas.io.common import get_handle

    handles = None
    try:
        for df in dfs:
            header = handles is None
            if handles is None:
                # the file to_csv(file) opens, with the file mode R's C++ writer gives it
                with _shared._with_cpp_umask():
                    handles = get_handle(file, "w", encoding="utf-8", compression="infer")
            df.drop(columns=["intervalID"], errors="ignore").to_csv(
                handles.handle, sep="\t", index=False, header=header, float_format="%.15g", na_rep="nan")
            handles.handle.flush()  # before the next part forks
        if handles is None:
            with _shared._with_cpp_umask():
                handles = get_handle(file, "w", encoding="utf-8", compression="infer")
            pd.DataFrame(columns=columns).to_csv(handles.handle, sep="\t", index=False)
    finally:
        if handles is not None:
            handles.close()


def _extract_to_file(
    frame: Any,
    scope: pd.DataFrame,
    iterator: Any,
    binsize: int | None,
    base_id_map: Any,
    orig_scope: pd.DataFrame,
    file: Any,
    value_names: list[str],
    parquet_kw: dict[str, Any],
    progress: Any,
    progress_desc: str,
) -> None:
    """gextract(file=): computes the rows a part of the scope at a time (see _file_parts) with
    frame(part, id_map), the extraction of gextract, and appends them to the file as they come,
    as R misha writes them; CONFIG['max_data_size'] does not limit them. The file is a
    tab-delimited file (see _write_tsv_parts; *value_names* name the value columns of its
    header), or a Parquet file (see _parquet_output) of the columns of *parquet_kw*."""
    parts = _file_parts(scope, iterator, binsize, base_id_map, orig_scope)
    # the parts draw one random stream, as the extraction of the whole scope does
    with (_one_random_stream() if len(parts) > 1 else contextlib.nullcontext()), \
            _file_extraction(progress, progress_desc, len(parts)) as cb:
        def part_rows():
            for i, (part, id_map) in enumerate(parts):
                df = frame(part, id_map)
                if cb:
                    cb(i + 1, len(parts), int((i + 1) * 100 / len(parts)))
                if df is not None and len(df):
                    yield _genome_order(df, orig_scope)

        if parquet_kw:
            names = parquet_kw["value_names"]
            with _parquet_output(file) as (name, tmp):
                writer = _pymisha.pm_parquet_writer_open(tmp, name, list(names), parquet_kw["int64_coords"],
                                                         _parquet_created_by(), parquet_kw["value_metadata"])
                for df in part_rows():
                    _pymisha.pm_parquet_writer_add(writer, *_parquet_rows(df, names))
                _pymisha.pm_parquet_writer_close(writer)
        else:
            _write_tsv_parts(file, part_rows(), ["chrom", "start", "end", *value_names])


def _extract_2d_to_file(
    extract: Any, scope: pd.DataFrame, file: Any, value_names: list[str], progress: Any, progress_desc: str
) -> None:
    """2D gextract(file=): computes the rows a chromosome pair of the scope at a time with
    extract(part), the 2D extraction of gextract, and appends them to the tab-delimited file as
    they come (see _write_tsv_parts). The pairs, and the intervals of each, are taken in the
    order R misha's scan takes them (r_2d_sort_order), in which the extraction also takes them,
    so that the rows are those of the whole extraction. The scope is checked as R checks it
    before anything is written."""
    from .intervals import _normalize_chroms, _verify_2d_intervals

    _verify_2d_intervals(scope)
    cids = []
    for k in ("1", "2"):
        chroms = _normalize_chroms(_chrom_names(scope[f"chrom{k}"], f"chrom{k}").tolist())
        codes, uniques = _pandas.factorize(_numpy.asarray(chroms, dtype=object))
        cids.append(_numpy.asarray(_pymisha.pm_chrom_ids(list(uniques)), dtype=_numpy.int64)[codes])
    order = _numpy.asarray(_pymisha.pm_r_2d_sort_order(cids[0], cids[1]), dtype=_numpy.int64)
    c1, c2 = cids[0][order], cids[1][order]
    coords = {c: scope[c].to_numpy(dtype=_numpy.int64)[order] for c in ("start1", "end1", "start2", "end2")}
    _pymisha.pm_verify_no_overlaps_2d(c1, coords["start1"], coords["end1"], c2, coords["start2"], coords["end2"])
    bounds = _numpy.r_[0, _numpy.flatnonzero((c1[1:] != c1[:-1]) | (c2[1:] != c2[:-1])) + 1, len(order)]
    pairs = [order[b0:b1] for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True)]
    # the pairs draw one random stream, as the extraction of the whole scope does
    with (_one_random_stream() if len(pairs) > 1 else contextlib.nullcontext()), \
            _file_extraction(progress, progress_desc, len(pairs)) as cb:
        def part_rows():
            for i, rows in enumerate(pairs):
                df = extract(scope.iloc[rows].reset_index(drop=True))
                if cb:
                    cb(i + 1, len(pairs), int((i + 1) * 100 / len(pairs)))
                if df is not None and len(df):
                    yield df

        _write_tsv_parts(file, part_rows(), ["chrom1", "start1", "end1", "chrom2", "start2", "end2", *value_names])


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


def _effective_max_procs(config: dict) -> int:
    """Worker count the C++ side would pick, mirroring ``choose_num_kids``.

    The C++ scanner clamps to ``hardware_concurrency`` before applying
    ``max_processes``; the Python fork paths used ``max_processes`` raw, so on
    a machine with fewer cores than that they oversubscribed (e.g. 56
    expressions on a 4-core laptop forked one worker per expression).
    """
    hw = os.cpu_count() or 1
    lo = int(config.get("min_processes", 1) or 1)
    hi = int(config.get("max_processes", 1) or 1)
    return max(1, min(max(hw, lo), hi))


def _tracks_workload_too_small_for_fork(
    intervals: pd.DataFrame, n_exprs: int, config: dict
) -> bool:
    """Return True if a track-parallel fork is not worth it.

    The ``_workload_too_small_for_fork`` floors are expressed in base-pairs
    of scope because they were calibrated for the *tiles* strategy, where
    every worker walks its own slice of the genome.  Track-parallel splits
    the *expressions* instead, so the work scales with intervals x
    expressions and a bp-scope floor vetoes exactly the workloads it helps
    most (many tracks over a modest peak set).  Reuse ``min_intervs4process``
    as the floor, counting per-expression interval visits.
    """
    min_intervs = int(config.get("min_intervs4process", 250_000))
    if min_intervs <= 0:
        return False
    return len(intervals) * max(1, n_exprs) < min_intervs


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


# Every track-parallel worker scans the *same* interval set, so passing it
# through pool.map would pickle and pipe one full copy per worker - ~10 MB
# per worker for a 345k-row peak set, growing linearly with max_processes.
# The pool is forked, so children inherit this instead, at no copy cost.
_FORK_SHARED_INTERVALS: pd.DataFrame | None = None


def _worker_extract_tracks(args: tuple[Any, ...]) -> pd.DataFrame | None:
    """Worker for track-parallel gextract: process a subset of expressions
    over the full interval set."""
    (expr_subset, iterator_val, config_dict, vtracks_dict) = args
    intervals_df = _FORK_SHARED_INTERVALS
    if intervals_df is None:  # pragma: no cover - fork inheritance failed
        raise RuntimeError(
            "track-parallel worker started without the shared interval set"
        )
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
    if len(exprs) < 2:
        return None
    # Repeated expressions are deduped into distinct column names ('expr',
    # 'expr_', 'expr__') by the C++ writer, which only sees its own chunk -
    # so the same expression in two chunks would come back twice as 'expr'.
    # Rare enough that falling back beats threading a global name map.
    if len(set(exprs)) != len(exprs):
        return None

    n_workers = min(_effective_max_procs(config), len(exprs))
    if n_workers < 2:
        return None
    expr_chunks = _split_exprs(exprs, n_workers)

    worker_config = dict(config)
    worker_config["progress"] = False
    worker_config["multitasking"] = False  # workers must not fork again

    worker_args = [
        (chunk, iterator, worker_config, vtracks_dict) for chunk in expr_chunks
    ]

    # Publish the shared scope before the workers fork; they inherit it.
    global _FORK_SHARED_INTERVALS
    _FORK_SHARED_INTERVALS = intervals
    try:
        results, dead = _run_forked(_worker_extract_tracks, worker_args)
    finally:
        _FORK_SHARED_INTERVALS = None
    # A worker that died would leave its columns out of the result
    if dead:
        raise RuntimeError(
            f"track-parallel gextract: {len(dead)} of {len(worker_args)} worker processes ended without returning a "
            "result (killed, e.g. by the out-of-memory killer, or crashed)"
        )

    non_empty = [(chunk, df) for chunk, df in zip(expr_chunks, results, strict=True)
                 if df is not None and len(df) > 0]
    if not non_empty:
        return _pandas.DataFrame()

    # Workers ran on identical (intervals, iterator) so the coordinate /
    # intervalID rows match, in scan order (no sort: rows of one interval tie
    # on intervalID, and numpy's default sort may reorder ties). Take coords
    # from the first chunk, column-bind expression values from all chunks,
    # and append intervalID last to match serial output column order
    # (coords -> exprs -> intervalID).
    sort_key = "intervalID"
    first_chunk_exprs, first_df = non_empty[0]

    coord_cols = [c for c in ("chrom", "start", "end", "chrom1", "start1", "end1",
                               "chrom2", "start2", "end2")
                  if c in first_df.columns]
    meta_cols = coord_cols + ([sort_key] if sort_key in first_df.columns else [])

    def _value_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if c not in meta_cols]]

    parts = [first_df[coord_cols].copy(), _value_columns(first_df)]
    for _chunk, df in non_empty[1:]:
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
    """Split the scope into bp-balanced range chunks and extract in parallel.

    Each worker handles a contiguous run of (possibly range-split) sub-intervals
    spanning roughly equal base-pairs, so a single huge interval (e.g. a whole
    chromosome in ALLGENOME) is spread across workers instead of bounding wall
    time on one kid. For a fixed-bin (int) iterator, splits are aligned to bin
    boundaries so the emitted genome-aligned bins are byte-identical to a serial
    extraction; for other iterator types intervals are kept whole (a mid-interval
    split could change which iterator intervals are emitted at the boundary).

    Returns a merged DataFrame with intervalIDs remapped to the original scope
    rows (matching a serial extraction over *intervals*), or ``None`` to signal
    the caller to fall back to the serial path.
    """
    max_procs = _effective_max_procs(config)
    if max_procs < 2:
        return None

    chrom_col = intervals["chrom"].to_numpy()
    start_col = intervals["start"].to_numpy()
    end_col = intervals["end"].to_numpy()
    total_bp = int(_numpy.maximum(end_col - start_col, 0).sum())
    if total_bp <= 0:
        return None

    bin_size = (
        int(iterator)
        if isinstance(iterator, (int, _numpy.integer)) and int(iterator) > 0
        else None
    )

    # ceil(total_bp / max_procs): target bp per worker.
    target_bp = max(1, -(-total_bp // max_procs))

    # Tile the scope into sub-intervals in original row order, each tagged with
    # its parent (original) row index.
    subs: list[tuple[Any, int, int, int]] = []
    for i in range(len(chrom_col)):
        c = chrom_col[i]
        s = int(start_col[i])
        e = int(end_col[i])
        if e <= s or bin_size is None or (e - s) <= target_bp:
            subs.append((c, s, e, i))
            continue
        # Split [s, e) into bin-aligned pieces of ~target_bp.
        step = max(bin_size, (target_bp // bin_size) * bin_size)
        cur = s
        while cur < e:
            nxt = cur + step
            if nxt < e:
                nxt = (nxt // bin_size) * bin_size  # align down to a bin boundary
                if nxt <= cur:
                    nxt = cur + step
            else:
                nxt = e
            subs.append((c, cur, min(nxt, e), i))
            cur = nxt

    if len(subs) < 2:
        return None

    # Partition the ordered sub-intervals into <= max_procs contiguous,
    # bp-balanced runs (preserves original order on concatenation).
    n_workers = min(max_procs, len(subs))
    runs: list[list[tuple[Any, int, int, int]]] = [[] for _ in range(n_workers)]
    run_bp = [0] * n_workers
    w = 0
    for sub in subs:
        runs[w].append(sub)
        run_bp[w] += max(0, sub[2] - sub[1])
        if run_bp[w] >= target_bp and w < n_workers - 1:
            w += 1
    runs = [r for r in runs if r]

    if len(runs) < 2:
        return None  # everything landed in one run -> serial is just as fast

    worker_config = dict(config)
    worker_config["progress"] = False
    worker_config["multitasking"] = False  # workers must not fork again

    worker_args = []
    run_parent_maps = []
    for r in runs:
        chunk_dict = {
            "chrom": [x[0] for x in r],
            "start": [x[1] for x in r],
            "end": [x[2] for x in r],
        }
        worker_args.append((chunk_dict, exprs, iterator, worker_config, vtracks_dict))
        run_parent_maps.append(_numpy.array([x[3] for x in r], dtype=_numpy.int64))

    results, dead = _run_forked(_worker_extract_chunk, worker_args)
    # A worker that died would leave its rows out of the result: as the C++ scanner reports a kid
    if dead:
        raise RuntimeError(_dead_worker_message(dead))

    dfs = []
    for df, parent_map in zip(results, run_parent_maps, strict=False):
        if df is not None and len(df) > 0:
            if "intervalID" in df.columns:
                # C++ intervalID is 1-based within the run -> the run's k-th
                # sub-interval -> its parent row -> original global 1-based id.
                local_ids = df["intervalID"].to_numpy()
                df = df.copy()
                df["intervalID"] = parent_map[local_ids - 1] + 1
            dfs.append(df)
    if not dfs:
        return _pandas.DataFrame()
    return _pandas.concat(dfs, ignore_index=True)


# The first suggestion of gextract's error for a result over CONFIG['max_data_size'], as R misha's
# GEXTRACT_FILE_SUGGESTION: a file output is not limited
_GEXTRACT_FILE_SUGGESTION = (
    "  - Write the result to a file, which CONFIG['max_data_size'] does not limit: "
    'gextract(..., file="x.tsv") or file="x.parquet"\n'
)


def _suggest_file_output(func: collections.abc.Callable[..., Any]) -> collections.abc.Callable[..., Any]:
    """gextract's error for a result over CONFIG['max_data_size'] lists first the suggestion to
    write the result to a file (the error comes from the C++ check shared with other functions).
    The wrapped function finds its caller's variables two frames up."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except _pymisha.error as exc:
            msg = str(exc)
            if msg.startswith("Result size (at least ") and "\nSuggestions:\n" in msg \
                    and _GEXTRACT_FILE_SUGGESTION not in msg:
                msg = msg.replace("\nSuggestions:\n", "\nSuggestions:\n" + _GEXTRACT_FILE_SUGGESTION, 1)
                raise _pymisha.error(msg) from None
            raise
    return wrapper


@_suggest_file_output
def gextract(
    expr: str | list[str],
    intervals: pd.DataFrame | str | None = None,
    iterator: Iterator | str = None,
    colnames: str | list[str] | None = None,
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
    colnames : str or list of str, optional
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

        - **file** (*str, optional*) -- Path to write extraction results
          to. When provided, the result is written to the file and ``None``
          is returned instead of a DataFrame. The rows are computed and
          written a part of the scope at a time (a 2D extraction a
          chromosome pair at a time), as R misha writes them, so the whole
          result is never held in memory, and ``CONFIG['max_data_size']``
          does not limit it. A name ending in ``.parquet``
          (in any case) writes a GZIP-compressed Parquet file: columns
          ``chrom``, ``start`` and ``end``, then one double column per
          expression, named as in the DataFrame, all nullable (NaN values
          are written as NaN). ``start`` and ``end`` are 32-bit integers
          unless a chromosome of the scope is 2^31 bp or longer. Only 1D
          intervals are supported, and the column names must be unique and
          other than ``chrom``, ``start`` and ``end``. The file is written
          under a temporary name in its directory (a symbolic link is
          followed, and the file it points to is replaced) and renamed when
          complete, so an error or an interrupt leaves an existing file as
          it was. A column whose expression is exactly the name of a track
          carries the track's attributes as its field metadata (Arrow field
          metadata: in pyarrow, ``pq.read_schema(file).field(name).metadata``);
          :func:`gtrack_import_parquet` restores them. Any other name writes
          a tab-separated file, in place: an error part way through leaves
          the rows written so far, as in R misha; with no rows it holds the
          header. A new file of
          either kind is created with mode 0660 (umask 007), whatever the
          caller's umask; a file it replaces keeps its permission bits.
        - **intervals_set_out** (*str, optional*) -- Name of an interval set
          to save the result to (created via :func:`gintervals_save`). When
          provided, ``None`` is returned instead of a DataFrame.
        - **progress** (*bool or str, optional*) -- Whether to show a
          progress bar.
        - **progress_desc** (*str, optional*) -- Description for the
          progress bar (default ``'gextract'``).

    Returns
    -------
    DataFrame or None
        DataFrame with columns: chrom, start, end, <expr1>, ..., intervalID.
        Returns None if the iterator produces no intervals, or if *file* or
        *intervals_set_out* is specified.

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

    # Capture caller namespace for user variable resolution (the caller of _suggest_file_output's
    # wrapper)
    caller_ns = dict(vars) if vars is not None else _caller_namespace(depth=2)

    exprs = [expr] if isinstance(expr, str) else list(expr)
    if isinstance(colnames, str):
        colnames = [colnames]

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

    file = kwargs.get("file")
    intervals_set_out = kwargs.get("intervals_set_out")
    if file is not None and intervals_set_out is not None:
        raise ValueError("Cannot use both file and intervals_set_out arguments")

    from .tracks import _check_computed_tracks

    _check_computed_tracks(exprs)

    if intervals is None:
        from .expr import _expr_is_2d
        from .intervals import _whole_genome_2d, gintervals_all

        # the whole genome in the dimension of the iterator, as R's .misha$ALLGENOME
        intervals = _whole_genome_2d() if _expr_is_2d(exprs, iterator) else gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)
    intervals = _maybe_load_2d_intervals_set(intervals, exprs, iterator, band)

    parquet_kw: dict[str, Any] = {}
    if _is_parquet_path(file):
        if _is_2d_intervals(intervals):
            raise ValueError("Parquet output does not support 2D iterators")
        parquet_kw = {
            "value_names": list(colnames) if colnames is not None else exprs,
            "int64_coords": _parquet_int64_coords(intervals),
            "value_metadata": _parquet_value_metadata(exprs),
        }

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
        if file is not None:
            # written as the rows come, a chromosome pair of the scope at a time, as R misha writes it
            _extract_2d_to_file(
                lambda part: _gextract_2d(exprs, part, iterator=iterator, colnames=colnames, band=band,
                                          caller_ns=caller_ns),
                intervals, file, list(colnames) if colnames is not None else exprs, kwargs.get("progress"),
                kwargs.get("progress_desc", "gextract"),
            )
            return None
        df = _gextract_2d(
            exprs,
            intervals,
            iterator=iterator,
            colnames=colnames,
            band=band,
            caller_ns=caller_ns,
        )
        if df is not None and len(df) == 0 and intervals_set_out is None:
            df = None  # R returns NULL when the iterator gives no interval
        return _apply_extract_output(
            df, intervals_set_out, is_2d=True,
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

    # When no explicit iterator is given, a value-based vtrack defaults to its
    # source track's native iterator (R parity), not the whole interval.
    if iterator is None and used_vtracks and not all_user_vars:
        _check_dense_bin_sizes(exprs, used_vtracks)
        _inferred_it = _infer_iterator_from_vtracks(used_vtracks)
        if _inferred_it is not None:
            iterator = _inferred_it
            # An array source resolves to its track name; expand it to the array
            # bins (one iterator interval per bin) via the same path a string
            # track iterator uses, and refresh the intervalID remap.
            if isinstance(iterator, str):
                intervals, iterator, _itr_id_map = _preprocess_intervals_iterator(
                    intervals, iterator
                )

    if iterator is None and not used_tracks:
        _check_seq_vtrack_iterator(exprs, used_vtracks)  # a physical track gives the iterator

    # Check if vtracks can go through C++ path
    cpp_vtracks_extract = used_vtracks and _can_vtracks_use_cpp(used_vtracks) and not all_user_vars

    def extract_frame(scope, id_map, mt_map, prog, allow_parallel):
        """The rows of the expressions over *scope* (a scope after the iterator preprocessing
        above), in the order of the scan, their intervalID mapped through *id_map*, or None if
        there are none. *mt_map* is the id map of a DataFrame iterator, which keeps the scan in
        one process; *allow_parallel*: whether the Python fork paths may be used."""

        if (not used_vtracks and not all_user_vars) or cpp_vtracks_extract:
            vtracks_dict = _build_vtracks_dict(used_vtracks) if cpp_vtracks_extract else None
            # Try parallel extraction if max_processes > 1.
            # Skip when a custom progress callback is provided (not compatible
            # with forked workers) or when file output is requested.
            # Resolve the strategy before entering _config_no_mt: track-parallel is
            # worth forking even when the iterator came in as a DataFrame, which
            # otherwise pins the whole extraction to one process.
            strategy = _resolve_parallel_strategy(
                str(CONFIG.get("multitasking_strategy", "auto")),
                n_exprs=len(exprs),
                iterator=iterator,
            )
            track_parallel = strategy == "tracks" and not _tracks_workload_too_small_for_fork(
                scope, len(exprs), CONFIG
            )
            with _config_no_mt(mt_map, keep=track_parallel) as _cfg:
                df = None
                # Mirror the C++ gate, which divides the scope by the same
                # core-clamped worker count it would actually fork.
                _max_procs = _effective_max_procs(_cfg)
                use_parallel = (
                    _config_multitasking(_cfg)
                    and _max_procs > 1
                    and not callable(prog)
                    and allow_parallel
                    and (
                        track_parallel
                        or not _workload_too_small_for_fork(scope, _max_procs, _cfg)
                    )
                )
                if use_parallel:
                    if track_parallel:
                        df = _parallel_extract_tracks(
                            exprs, scope, iterator, _cfg, vtracks_dict=vtracks_dict,
                        )
                    if df is None:
                        df = _parallel_extract(exprs, scope, iterator, _cfg, vtracks_dict=vtracks_dict)

                if df is None:
                    with _progress_context(prog, desc=progress_desc):
                        # The C++ scanner cannot fork under a DataFrame-derived
                        # iterator: intervalIDs are sequential indices into the
                        # intersected scope and _remap_interval_ids relies on that.
                        _serial_cfg = _cfg
                        if mt_map is not None and _config_multitasking(_cfg):
                            _serial_cfg = dict(_cfg)
                            _serial_cfg["multitasking"] = False
                        result = _pymisha.pm_extract(
                            exprs,
                            _df2pymisha(scope),
                            iterator,
                            _serial_cfg,
                            vtracks_dict,
                        )
                    df = _pymisha2df(result)
            df = _remap_interval_ids(df, id_map)
            if colnames is not None and df is not None and isinstance(df, _pandas.DataFrame):
                # Build rename map: old expression columns -> new names
                # The C++ path names columns after the expression strings
                non_meta = [c for c in df.columns if c not in ("chrom", "start", "end", "intervalID")]
                if len(non_meta) == len(colnames):
                    rename_map = dict(zip(non_meta, colnames, strict=False))
                    df = df.rename(columns=rename_map)
            return df

        track_arrays = {}
        base_df = None
        iter_df = None

        if used_tracks:
            if iterator is None and used_vtracks:
                _check_dense_bin_sizes(exprs, used_vtracks)
            track_exprs = list(used_tracks)
            with _config_no_mt(mt_map) as _cfg:
                base_result = _pymisha.pm_extract(track_exprs, _df2pymisha(scope), iterator, _cfg)
            base_df = _pymisha2df(base_result)
            if base_df is None:
                raise RuntimeError("Failed to extract physical track values for mixed expression")
            base_df = _remap_interval_ids(base_df, id_map)
            for tname in track_exprs:
                col = tname
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
                    assert isinstance(scope, _pandas.DataFrame)
                    iter_df = scope.copy()
                    iter_df["intervalID"] = _numpy.arange(1, len(iter_df) + 1, dtype=int)
                    iter_df = _remap_interval_ids(iter_df, id_map)
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
                # intervalID as the branch above: the caller's scope row, not the
                # row of the intersected scope a DataFrame/track iterator made
                iter_df = _remap_interval_ids(_iterated_intervals(scope, iterator), id_map)

        if iter_df is None or len(iter_df) == 0:
            return None

        n_rows = len(iter_df)
        chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)  # type: ignore[call-overload]
        compiled = []
        result_cols = []
        # Match the C++ PMDataFrame dedup: when two expressions (or user-supplied
        # colnames) resolve to the same name, append '_' until unique so each keeps
        # its own column instead of silently overwriting an earlier one.
        used_colnames = {"chrom", "start", "end"}
        for i, (orig_expr, expr_eval, expr_tracks, expr_vtracks) in enumerate(parsed):
            colname = colnames[i] if colnames is not None else orig_expr
            while colname in used_colnames:
                colname += "_"
            used_colnames.add(colname)
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

        with _progress_context(prog, total=n_rows, desc=progress_desc) as progress_cb:
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
        return result_df


    # file= output is written as the rows come, a part of the scope at a time, as R misha writes
    # it, and the size of the result is not limited
    if file is not None:
        binsize = None
        if isinstance(iterator, (int, _numpy.integer)) and not isinstance(iterator, bool) and int(iterator) > 0:
            binsize = int(iterator)
        elif iterator is None:
            from .gsynth import _implicit_binsize

            binsize = _implicit_binsize(exprs)
        _extract_to_file(
            # as before, a part is not split between Python workers; the C++ scan may fork
            lambda scope, id_map: extract_frame(scope, id_map, _itr_id_map, False, False),
            intervals, iterator, binsize, _itr_id_map,
            input_intervals_for_join if isinstance(input_intervals_for_join, pd.DataFrame) else None, file,
            list(colnames) if colnames is not None else exprs, parquet_kw, progress, progress_desc,
        )
        return None

    df = extract_frame(intervals, _itr_id_map, _itr_id_map, progress, True)
    return _apply_extract_output(
        df, intervals_set_out, is_2d=False,
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

    if kwargs.get("intervals_set_out") is not None:
        # as R: the hits are saved, an empty set when there are none
        from .intervals import gintervals_save

        set_out = kwargs.pop("intervals_set_out")
        df = gscreen(expr, intervals, vars=caller_ns, **kwargs)
        if df is None or len(df) == 0:
            from .expr import _expr_is_2d

            is_2d = _expr_is_2d(expr, kwargs.get("iterator")) or _is_2d_intervals(intervals)
            df = pd.DataFrame({c: [] for c in (
                ["chrom1", "start1", "end1", "chrom2", "start2", "end2"] if is_2d else ["chrom", "start", "end"])})
        gintervals_save(df, set_out)
        return None

    from .tracks import _check_computed_tracks

    _check_computed_tracks(expr)

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gscreen")
    iterator = kwargs.get("iterator")
    band = kwargs.get("band")

    if intervals is None:
        from .expr import _expr_is_2d
        from .intervals import _whole_genome_2d, gintervals_all

        intervals = _whole_genome_2d() if _expr_is_2d(expr, iterator) else gintervals_all()

    intervals = _maybe_load_intervals_set(intervals)

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
        return extracted.loc[mask, ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].reset_index(drop=True)

    # Handle DataFrame-as-iterator for 1D gscreen
    intervals, iterator, _scr_id_map = _preprocess_intervals_iterator(intervals, iterator, canonic_scope=True)

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

    if not expr_tracks:
        intervals, iterator = _default_vtrack_iterator(intervals, iterator, [expr], expr_vtracks)

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
        return _pymisha2df(result)

    track_arrays = {}
    base_df = None
    iter_df = None

    if expr_tracks:
        if iterator is None and expr_vtracks:
            _check_dense_bin_sizes([expr], expr_vtracks)
        track_exprs = list(expr_tracks)
        with _config_no_mt(_scr_id_map) as _cfg:
            base_result = _pymisha.pm_extract(track_exprs, _df2pymisha(intervals), iterator, _cfg)
        base_df = _pymisha2df(base_result)
        if base_df is None:
            raise RuntimeError("Failed to extract physical track values for mixed expression")
        for tname in track_exprs:
            col = tname
            if col not in base_df.columns:
                raise KeyError(f"Track column not found for '{tname}'")
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)
        iter_df = base_df[["chrom", "start", "end", "intervalID"]]
    else:
        intervals, iterator = _default_vtrack_iterator(intervals, iterator, [expr], expr_vtracks)
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

    return _pandas.DataFrame(out_rows, columns=["chrom", "start", "end"])
