"""Virtual track utilities and API."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from . import _shared
from ._shared import (
    CONFIG,
    _checkroot,
    _df2pymisha,
    _numpy,
    _pandas,
    _pymisha,
)

_FILTER_PASSTHROUGH_FUNCS = {"distance", "distance.center", "distance.edge"}
_FILTER_WEIGHTED_FUNCS = {"avg", "mean", "coverage", "kmer.frac", "masked.frac"}
_FILTER_ADDITIVE_FUNCS = {"sum", "kmer.count", "masked.count", "pwm.count", "neighbor.count"}
_FILTER_MIN_FUNCS = {"min"}
_FILTER_MAX_FUNCS = {"max", "pwm.max"}
_FILTER_STDDEV_FUNCS = {"stddev", "std"}
_FILTER_QUANTILE_FUNCS = {"quantile"}
_FILTER_NEAREST_FUNCS = {"nearest"}
_FILTER_EXISTS_FUNCS = {"exists"}
_FILTER_SIZE_FUNCS = {"size"}
_FILTER_FIRST_FUNCS = {"first"}
_FILTER_LAST_FUNCS = {"last"}
_FILTER_SAMPLE_FUNCS = {"sample"}
_FILTER_FIRST_POS_ABS_FUNCS = {"first.pos.abs"}
_FILTER_FIRST_POS_REL_FUNCS = {"first.pos.relative"}
_FILTER_LAST_POS_ABS_FUNCS = {"last.pos.abs"}
_FILTER_LAST_POS_REL_FUNCS = {"last.pos.relative"}
_FILTER_MAX_POS_ABS_FUNCS = {"max.pos.abs"}
_FILTER_MAX_POS_REL_FUNCS = {"max.pos.relative"}
_FILTER_MIN_POS_ABS_FUNCS = {"min.pos.abs"}
_FILTER_MIN_POS_REL_FUNCS = {"min.pos.relative"}
_FILTER_SAMPLE_POS_ABS_FUNCS = {"sample.pos.abs"}
_FILTER_SAMPLE_POS_REL_FUNCS = {"sample.pos.relative"}
_FILTER_LOGSUMEXP_FUNCS = {"pwm", "lse"}
_FILTER_GLOBAL_PERCENTILE_FUNCS = {
    "global.percentile",
    "global.percentile.min",
    "global.percentile.max",
}
_FILTER_PWM_MAX_POS_FUNCS = {"pwm.max.pos"}
_FILTER_EDIT_DISTANCE_FUNCS = {
    "pwm.edit_distance",
    "pwm.edit_distance.pos",
    "pwm.max.edit_distance",
    "pwm.edit_distance.lse",
    "pwm.edit_distance.lse.pos",
}
_DF_INTERVAL_FUNCS = {"distance", "distance.center", "distance.edge", "coverage", "neighbor.count"}
# Columns that never count as a "value" column when inferring a DataFrame source's
# default function (mirrors R's TrackExpressionVars value-column detection).
_NON_VALUE_COLS = {"chrom", "start", "end", "strand", "intervalID", "intervalID1", "intervalID2"}


def _infer_default_vtrack_func(src: pd.DataFrame | str | None) -> str:
    """Infer the default ``func`` when none is given (R parity).

    R defaults a virtual track's function to ``"distance"`` when its source
    resolves to an intervals set (no value column) and to ``"avg"`` when the
    source is a track or a value-bearing intervals set.
    """
    if isinstance(src, _pandas.DataFrame):
        has_value = any(
            c not in _NON_VALUE_COLS and _pandas.api.types.is_numeric_dtype(src[c])
            for c in src.columns
        )
        return "avg" if has_value else "distance"
    if isinstance(src, str):
        from .tracks import _track_exists

        if _track_exists(src):
            return "avg"
        from .intervals import gintervals_exists

        if gintervals_exists(src):
            return "distance"
    return "avg"
_VALUE_DF_PY_FUNCS = {
    "avg", "mean", "sum", "min", "max", "first", "last", "size", "exists",
    "stddev", "std", "quantile", "nearest",
    "first.pos.abs", "first.pos.relative",
    "last.pos.abs", "last.pos.relative",
    "min.pos.abs", "min.pos.relative",
    "max.pos.abs", "max.pos.relative",
}
_FILTER_SUPPORTED_FUNCS = (
    _FILTER_PASSTHROUGH_FUNCS
    | _FILTER_WEIGHTED_FUNCS
    | _FILTER_ADDITIVE_FUNCS
    | _FILTER_MIN_FUNCS
    | _FILTER_MAX_FUNCS
    | _FILTER_STDDEV_FUNCS
    | _FILTER_QUANTILE_FUNCS
    | _FILTER_NEAREST_FUNCS
    | _FILTER_EXISTS_FUNCS
    | _FILTER_SIZE_FUNCS
    | _FILTER_FIRST_FUNCS
    | _FILTER_LAST_FUNCS
    | _FILTER_SAMPLE_FUNCS
    | _FILTER_FIRST_POS_ABS_FUNCS
    | _FILTER_FIRST_POS_REL_FUNCS
    | _FILTER_LAST_POS_ABS_FUNCS
    | _FILTER_LAST_POS_REL_FUNCS
    | _FILTER_MAX_POS_ABS_FUNCS
    | _FILTER_MAX_POS_REL_FUNCS
    | _FILTER_MIN_POS_ABS_FUNCS
    | _FILTER_MIN_POS_REL_FUNCS
    | _FILTER_SAMPLE_POS_ABS_FUNCS
    | _FILTER_SAMPLE_POS_REL_FUNCS
    | _FILTER_LOGSUMEXP_FUNCS
    | _FILTER_GLOBAL_PERCENTILE_FUNCS
    | _FILTER_PWM_MAX_POS_FUNCS
    | _FILTER_EDIT_DISTANCE_FUNCS
)

_GLOBAL_PERCENTILE_CACHE: dict[tuple[str, str, int], np.ndarray] = {}
# Cache of a track's frozen pv.percentiles table: src -> (bins, breaks) or None.
_PV_TABLE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] = {}


def _canonicalize_filter_df(df: pd.DataFrame) -> pd.DataFrame:
    from .intervals import _normalize_chroms, gintervals_canonic

    if not isinstance(df, _pandas.DataFrame):
        raise ValueError("filter must be a DataFrame")
    if not {"chrom", "start", "end"}.issubset(df.columns):
        raise ValueError("filter must have columns: chrom, start, end")

    filt = df[["chrom", "start", "end"]].copy()
    filt["chrom"] = _normalize_chroms(filt["chrom"].astype(str).tolist())
    filt["start"] = _pandas.to_numeric(filt["start"], errors="coerce").astype("Int64")
    filt["end"] = _pandas.to_numeric(filt["end"], errors="coerce").astype("Int64")
    filt = filt.dropna(subset=["start", "end"]).copy()
    if len(filt) == 0:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])
    filt["start"] = filt["start"].astype(_numpy.int64)
    filt["end"] = filt["end"].astype(_numpy.int64)
    filt = filt[filt["end"] > filt["start"]].reset_index(drop=True)
    if len(filt) == 0:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])
    can = gintervals_canonic(filt, unify_touching_intervals=True)
    if can is None:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])
    return can[["chrom", "start", "end"]].reset_index(drop=True)


def _resolve_filter_sources(filter_obj: pd.DataFrame | str | list[Any] | tuple[Any, ...] | None) -> pd.DataFrame:
    from .extract import gextract
    from .intervals import gintervals_all, gintervals_load, gintervals_ls
    from .tracks import gtrack_info

    if filter_obj is None:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])

    if isinstance(filter_obj, _pandas.DataFrame):
        return _canonicalize_filter_df(filter_obj)

    if isinstance(filter_obj, list | tuple):
        if len(filter_obj) == 0:
            return _pandas.DataFrame(columns=["chrom", "start", "end"])
        # Collect all resolved parts, then union once via concat + single union
        parts = []
        for part in filter_obj:
            part_df = _resolve_filter_sources(part)
            if part_df is not None and len(part_df) > 0:
                parts.append(part_df[["chrom", "start", "end"]])
        if not parts:
            return _pandas.DataFrame(columns=["chrom", "start", "end"])
        if len(parts) == 1:
            return _canonicalize_filter_df(parts[0])
        # Concat all parts and canonicalize (which internally unions/merges)
        merged = _pandas.concat(parts, ignore_index=True)
        return _canonicalize_filter_df(merged)

    if isinstance(filter_obj, str):
        names = gintervals_ls()
        if names and filter_obj in names:
            loaded = gintervals_load(filter_obj)
            if loaded is None:
                return _pandas.DataFrame(columns=["chrom", "start", "end"])
            return _canonicalize_filter_df(loaded)

        track_path = _pymisha.pm_track_path(filter_obj)
        if track_path:
            info = gtrack_info(filter_obj)
            track_type = str(info.get("type", "")).lower()
            if track_type not in {"sparse", "intervals"}:
                raise ValueError(f"Track '{filter_obj}' is not an intervals-type track")
            extracted = gextract(filter_obj, gintervals_all())
            if extracted is None:
                return _pandas.DataFrame(columns=["chrom", "start", "end"])
            return _canonicalize_filter_df(extracted)

        raise ValueError(f"Unknown filter source '{filter_obj}'")

    raise ValueError("filter must be a DataFrame, string, list/tuple, or None")


def _filter_key(filter_df: pd.DataFrame | None) -> str | None:
    if filter_df is None or len(filter_df) == 0:
        return None
    # Vectorized hash: build byte string from arrays
    chroms = filter_df["chrom"].astype(str).values
    starts = filter_df["start"].astype(int).values
    ends = filter_df["end"].astype(int).values
    lines = [f"{c}\t{s}\t{e}\n" for c, s, e in zip(chroms, starts, ends, strict=False)]
    return hashlib.sha1("".join(lines).encode()).hexdigest()


def _filter_stats(filter_df: pd.DataFrame | None) -> dict[str, int] | None:
    if filter_df is None or len(filter_df) == 0:
        return None
    total = int((filter_df["end"] - filter_df["start"]).sum())
    chroms = int(filter_df["chrom"].nunique())
    return {"num_chroms": chroms, "total_bases": total}


def _subtract_masks(start: int, end: int, masks: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    if not masks:
        return [(start, end)]
    segs = [(start, end)]
    for mstart, mend in masks:
        if mend <= start:
            continue
        if mstart >= end:
            break
        updated = []
        for s, e in segs:
            if mend <= s or mstart >= e:
                updated.append((s, e))
                continue
            if mstart > s:
                updated.append((s, mstart))
            if mend < e:
                updated.append((mend, e))
        segs = updated
        if not segs:
            break
    return segs


def _build_unmasked_segments(
    intervals: pd.DataFrame,
    payload: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    from .intervals import gintervals_all

    sshift = int(payload.get("sshift", 0) or 0)
    eshift = int(payload.get("eshift", 0) or 0)
    chrom_sizes_df = gintervals_all()
    chrom_sizes = {
        str(chrom): int(end)
        for chrom, end in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)
    }

    has_filter = filter_df is not None and len(filter_df) > 0

    # --- Vectorized path (no mask) ---
    if not has_filter:
        chroms = intervals["chrom"].astype(str).values
        starts = intervals["start"].to_numpy(dtype=_numpy.int64) + sshift
        ends = intervals["end"].to_numpy(dtype=_numpy.int64) + eshift

        # Map chroms to sizes
        chrom_bound = _numpy.array(
            [chrom_sizes.get(str(c), -1) for c in chroms], dtype=_numpy.int64
        )
        known = chrom_bound >= 0
        starts = _numpy.maximum(starts, 0)
        ends = _numpy.where(known, _numpy.minimum(ends, chrom_bound), ends)
        valid = known & (ends > starts)

        if not valid.any():
            return None

        idx = _numpy.flatnonzero(valid)
        v_chroms = chroms[idx]
        v_starts = starts[idx]
        v_ends = ends[idx]
        v_lens = v_ends - v_starts

        return _pandas.DataFrame({
            "chrom": v_chroms,
            "start": v_starts,
            "end": v_ends,
            "orig_idx": idx,
            "seg_len": v_lens,
            "base_start": v_starts,
        })

    # --- Per-row path (with mask subtraction) ---
    assert filter_df is not None
    mask_map: dict[str, list[tuple[int, int]]] = {}
    _f_chroms = filter_df["chrom"].astype(str).values
    _f_starts = filter_df["start"].values
    _f_ends = filter_df["end"].values
    for _fi in range(len(filter_df)):
        _fc = _f_chroms[_fi]
        if _fc not in mask_map:
            mask_map[_fc] = []
        mask_map[_fc].append((int(_f_starts[_fi]), int(_f_ends[_fi])))
    for chrom in list(mask_map.keys()):
        mask_map[chrom].sort()

    seg_rows = []
    for row_idx, row in enumerate(intervals.itertuples(index=False)):
        chrom = str(row.chrom)
        start = int(row.start) + sshift
        end = int(row.end) + eshift
        chrom_size = chrom_sizes.get(chrom)
        if chrom_size is None:
            continue
        if start < 0:
            start = 0
        if end > chrom_size:
            end = chrom_size
        if end <= start:
            continue
        segments = _subtract_masks(start, end, mask_map.get(chrom))
        for s, e in segments:
            if e > s:
                seg_rows.append((chrom, int(s), int(e), row_idx, int(e - s), int(start)))
    if not seg_rows:
        return None
    return _pandas.DataFrame(
        seg_rows,
        columns=["chrom", "start", "end", "orig_idx", "seg_len", "base_start"],
    )


def _compute_value_df_vtrack(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """Python fallback evaluator for value-based virtual tracks."""
    func = str(payload_eval.get("func", "avg")).lower()
    has_filter = filter_df is not None and len(filter_df) > 0
    src_df = payload_eval.get("src_df")
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if src_df is None or len(src_df) == 0:
        if func == "exists":
            return _numpy.zeros(len(intervals), dtype=float)
        return out

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    if seg_df is None or len(seg_df) == 0:
        if func == "exists":
            return _numpy.zeros(len(intervals), dtype=float)
        return out

    src_by_chrom = {
        chrom: grp.reset_index(drop=True)
        for chrom, grp in src_df.groupby("chrom", sort=False)
    }

    for orig_idx, seg_group in seg_df.groupby("orig_idx", sort=False):
        oi = int(orig_idx)
        chrom = str(seg_group.iloc[0]["chrom"])
        sgrp = src_by_chrom.get(chrom)
        if sgrp is None or len(sgrp) == 0:
            if func == "exists":
                out[oi] = 0.0
            continue

        segments = [
            (int(r.start), int(r.end), int(r.base_start), int(r.seg_len))
            for r in seg_group.itertuples(index=False)
        ]
        base_start = segments[0][2]
        unmasked_len = int(seg_group["seg_len"].sum())

        if has_filter and func == "nearest":
            # nearest under filter uses only the first unmasked segment.
            segments = [segments[0]]

        # Vectorized overlap matching: compute overlaps between all source
        # intervals and all query segments using numpy broadcasting.
        src_starts = sgrp["start"].to_numpy(dtype=_numpy.int64)
        src_ends = sgrp["end"].to_numpy(dtype=_numpy.int64)
        src_vals = sgrp["value"].to_numpy(dtype=float)
        n_src = len(src_starts)

        total_ov = _numpy.zeros(n_src, dtype=_numpy.int64)
        overlaps_for_cov = []
        for ss, se, _bs, _sl in segments:
            ov_s = _numpy.maximum(src_starts, ss)
            ov_e = _numpy.minimum(src_ends, se)
            ov_len = _numpy.maximum(ov_e - ov_s, 0)
            total_ov += ov_len
            if func == "coverage":
                for k in _numpy.flatnonzero(ov_len > 0):
                    overlaps_for_cov.append((int(ov_s[k]), int(ov_e[k])))

        has_ov = (total_ov > 0) & ~_numpy.isnan(src_vals)
        m_idx = _numpy.flatnonzero(has_ov)
        n_matched = len(m_idx)

        if func == "exists":
            out[oi] = 1.0 if n_matched > 0 else 0.0
            continue
        if func == "coverage":
            if unmasked_len <= 0:
                continue
            if not overlaps_for_cov:
                out[oi] = 0.0
                continue
            overlaps_for_cov.sort()
            cov = 0
            cs, ce = overlaps_for_cov[0]
            for s, e in overlaps_for_cov[1:]:
                if s < ce:
                    ce = max(ce, e)
                else:
                    cov += ce - cs
                    cs, ce = s, e
            cov += ce - cs
            out[oi] = float(cov) / float(unmasked_len)
            continue

        if n_matched == 0:
            if func == "size":
                out[oi] = 0.0
            continue

        # Extract matched arrays directly via advanced indexing (no list-of-tuples)
        m_starts = src_starts[m_idx]
        m_vals = src_vals[m_idx]
        m_overlaps = total_ov[m_idx]

        if func in {"avg", "mean"}:
            if has_filter:
                weights = m_overlaps.astype(float)
                wsum = float(weights.sum())
                if wsum > 0:
                    out[oi] = float((m_vals * weights).sum() / wsum)
            else:
                out[oi] = float(m_vals.mean())
        elif func == "nearest":
            # If no overlap, nearest falls back to minimum distance.
            out[oi] = float(m_vals.mean())
        elif func == "sum":
            out[oi] = float(m_vals.sum())
        elif func == "min":
            out[oi] = float(m_vals.min())
        elif func == "max":
            out[oi] = float(m_vals.max())
        elif func == "size":
            out[oi] = float(n_matched)
        elif func in {"stddev", "std"}:
            if n_matched >= 2:
                out[oi] = float(_numpy.std(m_vals, ddof=1))
        elif func == "quantile":
            q = float(payload_eval.get("params", 0.5) or 0.5)
            out[oi] = float(_numpy.quantile(m_vals, q))
        elif func == "first":
            # min by (start, index) — use lexsort (sorts by last key first)
            order = _numpy.lexsort((m_idx, m_starts))
            out[oi] = float(m_vals[order[0]])
        elif func == "last":
            order = _numpy.lexsort((m_idx, m_starts))
            out[oi] = float(m_vals[order[-1]])
        elif func in {"first.pos.abs", "first.pos.relative"}:
            order = _numpy.lexsort((m_idx, m_starts))
            pos = float(m_starts[order[0]])
            out[oi] = pos - float(base_start) if func.endswith(".relative") else pos
        elif func in {"last.pos.abs", "last.pos.relative"}:
            order = _numpy.lexsort((m_idx, m_starts))
            pos = float(m_starts[order[-1]])
            out[oi] = pos - float(base_start) if func.endswith(".relative") else pos
        elif func in {"min.pos.abs", "min.pos.relative"}:
            # min by (value, start, index)
            order = _numpy.lexsort((m_idx, m_starts, m_vals))
            pos = float(m_starts[order[0]])
            out[oi] = pos - float(base_start) if func.endswith(".relative") else pos
        elif func in {"max.pos.abs", "max.pos.relative"}:
            # max by (value, -start, -index) — negate start/index for descending
            order = _numpy.lexsort((-m_idx, -m_starts, m_vals))
            pos = float(m_starts[order[-1]])
            out[oi] = pos - float(base_start) if func.endswith(".relative") else pos

    # nearest fallback for non-overlap cases — vectorized distance computation
    if func == "nearest":
        for orig_idx, seg_group in seg_df.groupby("orig_idx", sort=False):
            oi = int(orig_idx)
            if not _numpy.isnan(out[oi]):
                continue
            chrom = str(seg_group.iloc[0]["chrom"])
            sgrp = src_by_chrom.get(chrom)
            if sgrp is None or len(sgrp) == 0:
                continue
            ss = int(seg_group.iloc[0]["start"])
            se = int(seg_group.iloc[0]["end"])

            src_s = sgrp["start"].to_numpy(dtype=_numpy.int64)
            src_e = sgrp["end"].to_numpy(dtype=_numpy.int64)
            src_v = sgrp["value"].to_numpy(dtype=float)
            valid = ~_numpy.isnan(src_v)
            if not valid.any():
                continue

            # Vectorized distance: max(0, ss - src_e) for left, max(0, src_s - se) for right
            d_left = _numpy.maximum(ss - src_e, 0)
            d_right = _numpy.maximum(src_s - se, 0)
            dists = d_left + d_right  # one of them is 0 when overlapping

            dists_valid = dists[valid]
            vals_valid = src_v[valid]
            dmin = dists_valid.min()
            cand = vals_valid[dists_valid == dmin]
            if cand.size > 0:
                out[oi] = float(cand.mean())

    return out


def _compute_filtered_nearest(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """Nearest under filter: use only the first unmasked segment per interval."""
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    # Keep only the first segment per original interval
    first_segs = seg_df.groupby("orig_idx", sort=False).first().reset_index()
    payload_first = dict(payload_eval)
    payload_first["sshift"] = 0
    payload_first["eshift"] = 0
    seg_vals = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            payload_first,
            _df2pymisha(first_segs[["chrom", "start", "end"]]),
            CONFIG,
        ),
        dtype=float,
    )
    for orig_idx, val in zip(first_segs["orig_idx"], seg_vals, strict=False):
        out[int(orig_idx)] = val
    return out


def _extract_raw_unmasked_values(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> tuple[dict[int, list[float]], np.ndarray]:
    """Extract raw bin values from unmasked segments, grouped by original interval.

    Returns (groups dict {orig_idx: list[float]}, out_array) where out_array is
    pre-filled with NaN.  If there are no unmasked segments the groups dict is empty.
    """
    from .extract import gextract
    from .tracks import gtrack_info

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return {}, out

    src = payload_eval.get("src")
    if src is None:
        return {}, out

    # Determine bin size for raw extraction
    if isinstance(src, str):
        info = gtrack_info(src)
        bin_size = info.get("bin_size")
    else:
        bin_size = None

    src_expr = src

    iterator = int(bin_size) if bin_size else -1
    extracted = gextract(
        src_expr,
        seg_df[["chrom", "start", "end"]],
        iterator=iterator,
    )
    if extracted is None or len(extracted) == 0:
        return {}, out

    data_cols = [c for c in extracted.columns if c not in {"chrom", "start", "end", "intervalID"}]
    if not data_cols:
        return {}, out
    col = data_cols[0]

    seg_interval_ids = extracted["intervalID"].to_numpy(dtype=int, copy=False) - 1
    seg_orig_idx = seg_df["orig_idx"].to_numpy(dtype=int, copy=False)
    valid_ids = (seg_interval_ids >= 0) & (seg_interval_ids < len(seg_orig_idx))

    vals = extracted[col].to_numpy(dtype=float, copy=False)
    valid_vals = valid_ids & ~_numpy.isnan(vals)
    if not valid_vals.any():
        return {}, out

    mapped_orig = seg_orig_idx[seg_interval_ids[valid_vals]]
    mapped_vals = vals[valid_vals]
    groups: dict[int, list[float]] = {}
    for oi, v in zip(mapped_orig, mapped_vals, strict=False):
        groups.setdefault(int(oi), []).append(float(v))

    return groups, out


def _extract_raw_unmasked_values_with_positions(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> tuple[dict[int, list[tuple[float, int]]], dict[int, int], np.ndarray]:
    """Extract raw bin values *and* their genomic start positions from unmasked segments.

    Returns (groups dict {orig_idx: list[(float_val, int_start)]}, base_starts dict,
    out_array) where out_array is pre-filled with NaN.
    base_starts maps orig_idx -> the shifted interval start (for relative coordinate
    computation).
    """
    from .extract import gextract
    from .tracks import gtrack_info

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return {}, {}, out

    src = payload_eval.get("src")
    if src is None:
        return {}, {}, out

    # Determine bin size for raw extraction
    if isinstance(src, str):
        info = gtrack_info(src)
        bin_size = info.get("bin_size")
    else:
        bin_size = None

    iterator = int(bin_size) if bin_size else -1
    extracted = gextract(
        src,
        seg_df[["chrom", "start", "end"]],
        iterator=iterator,
    )
    if extracted is None or len(extracted) == 0:
        return {}, {}, out

    data_cols = [c for c in extracted.columns if c not in {"chrom", "start", "end", "intervalID"}]
    if not data_cols:
        return {}, {}, out
    col = data_cols[0]

    seg_interval_ids = extracted["intervalID"].to_numpy(dtype=int, copy=False) - 1
    seg_orig_idx = seg_df["orig_idx"].to_numpy(dtype=int, copy=False)
    valid_ids = (seg_interval_ids >= 0) & (seg_interval_ids < len(seg_orig_idx))

    vals = extracted[col].to_numpy(dtype=float, copy=False)
    starts = extracted["start"].to_numpy(dtype=int, copy=False)
    valid_vals = valid_ids & ~_numpy.isnan(vals)
    if not valid_vals.any():
        return {}, {}, out

    mapped_orig = seg_orig_idx[seg_interval_ids[valid_vals]]
    mapped_vals = vals[valid_vals]
    mapped_starts = starts[valid_vals]

    # Group (value, start) pairs by original interval index
    # Sort by orig_idx for efficient grouping via numpy
    order = _numpy.argsort(mapped_orig)
    sorted_orig = mapped_orig[order]
    sorted_vals = mapped_vals[order]
    sorted_starts = mapped_starts[order]

    groups: dict[int, list[tuple[float, int]]] = {}
    uniq, first_idx, counts = _numpy.unique(
        sorted_orig, return_index=True, return_counts=True
    )
    for k in range(len(uniq)):
        oi = int(uniq[k])
        fi = int(first_idx[k])
        c = int(counts[k])
        groups[oi] = list(zip(
            sorted_vals[fi:fi + c].tolist(),
            sorted_starts[fi:fi + c].tolist(),
            strict=False,
        ))

    # Collect base_start (shifted interval start) per original interval
    # Use groupby first() equivalent: take first occurrence per orig_idx
    seg_oi = seg_df["orig_idx"].to_numpy(dtype=int)
    seg_bs = seg_df["base_start"].to_numpy(dtype=int)
    _, first_seg_idx = _numpy.unique(seg_oi, return_index=True)
    base_starts = dict(zip(
        seg_oi[first_seg_idx].tolist(), seg_bs[first_seg_idx].tolist(), strict=False
    ))

    return groups, base_starts, out


def _compute_filtered_extremum_pos(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    mode: str,
    relative: bool,
) -> np.ndarray:
    """Compute max.pos.* or min.pos.* under filter.

    For track sources (string), extracts raw bin values with positions and finds
    the position of the global extremum across all unmasked segments.

    For value-based sources (DataFrame), delegates to C++ per-segment computation:
    computes both the extremum value (``max``/``min``) and its position
    (``max.pos.abs``/``min.pos.abs``) on each unmasked segment, then selects
    the segment with the overall best value.

    Parameters
    ----------
    mode : str
        ``"max"`` or ``"min"`` — selects the extremum.
    relative : bool
        If True, return position relative to the shifted interval start.
        If False, return the absolute genomic coordinate.
    """
    src = payload_eval.get("src")

    # For track sources (string), use raw bin extraction with positions
    if isinstance(src, str):
        groups, base_starts, out = _extract_raw_unmasked_values_with_positions(
            intervals, payload_eval, filter_df
        )
        for orig_idx, val_pos_pairs in groups.items():
            if not val_pos_pairs:
                continue
            if mode == "max":
                best_val, best_pos = val_pos_pairs[0]
                for v, s in val_pos_pairs[1:]:
                    if v > best_val:
                        best_val = v
                        best_pos = s
            else:  # min
                best_val, best_pos = val_pos_pairs[0]
                for v, s in val_pos_pairs[1:]:
                    if v < best_val:
                        best_val = v
                        best_pos = s
            if relative:
                bs = base_starts.get(orig_idx, 0)
                out[orig_idx] = float(best_pos - bs)
            else:
                out[orig_idx] = float(best_pos)
        return out

    # For value-based (DataFrame) sources, filter in Python.
    # The C++ backend does not support max.pos.*/min.pos.* for value-based vtracks.
    # We intersect source intervals with unmasked segments and find the extremum.
    from .intervals import _normalize_chroms

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    # Recover the original DataFrame source from the already-converted payload.
    # At this point ``src`` is the _df2pymisha output (list of arrays).
    # Reconstruct the relevant columns.
    assert src is not None
    src_arr: list[Any] = src  # list: [colnames, chrom_arr, start_arr, end_arr, val_arr, ...]
    src_chroms_raw = [str(c) for c in src_arr[1]]
    src_chroms = _normalize_chroms(src_chroms_raw)
    src_starts = _numpy.asarray(src_arr[2], dtype=int)
    src_ends = _numpy.asarray(src_arr[3], dtype=int)
    src_vals = _numpy.asarray(src_arr[4], dtype=float)

    best_extremum_val: dict[int, float] = {}
    for row in seg_df.itertuples(index=False):
        seg_chrom = str(row.chrom)
        seg_start = int(row.start)
        seg_end = int(row.end)
        oi = int(row.orig_idx)
        bs = int(row.base_start)

        for sc, ss, se, sv in zip(src_chroms, src_starts, src_ends, src_vals, strict=False):
            if sc != seg_chrom:
                continue
            # Check overlap between source interval and unmasked segment
            ov_start = max(int(ss), seg_start)
            ov_end = min(int(se), seg_end)
            if ov_end <= ov_start:
                continue
            if _numpy.isnan(sv):
                continue
            is_better = (
                oi not in best_extremum_val
                or (mode == "max" and float(sv) > best_extremum_val[oi])
                or (mode == "min" and float(sv) < best_extremum_val[oi])
            )
            if is_better:
                best_extremum_val[oi] = float(sv)
                pos = float(int(ss))
                if relative:
                    out[oi] = pos - float(bs)
                else:
                    out[oi] = pos

    return out


def _compute_filtered_stddev(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """Stddev under filter: extract raw bin values and compute exact stddev."""
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups.items():
        if len(raw_vals) >= 2:
            out[orig_idx] = float(_numpy.std(raw_vals, ddof=1))
    return out


def _compute_filtered_quantile(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """Quantile under filter: extract raw bin values and compute exact quantile."""
    percentile = float(payload_eval.get("params", 0.5) or 0.5)
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups.items():
        out[orig_idx] = float(_numpy.quantile(raw_vals, percentile))
    return out


def _logsumexp(values: list[float] | np.ndarray) -> float:
    arr = _numpy.asarray(values, dtype=float)
    arr = arr[~_numpy.isnan(arr)]
    if arr.size == 0:
        return _numpy.nan
    m = float(arr.max())
    if _numpy.isneginf(m):
        return m
    return float(m + _numpy.log(_numpy.exp(arr - m).sum()))


def _compute_filtered_lse(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """LSE under filter from raw unmasked source values."""
    if not isinstance(payload_eval.get("src"), str):
        raise NotImplementedError("lse under filter currently requires a track source")
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups.items():
        out[orig_idx] = _logsumexp(raw_vals)
    return out


def _compute_filtered_segment_logsumexp(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """Log-sum-exp composition over independently scored unmasked segments."""
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    payload_seg = dict(payload_eval)
    payload_seg["sshift"] = 0
    payload_seg["eshift"] = 0
    seg_vals = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            payload_seg,
            _df2pymisha(seg_df[["chrom", "start", "end"]]),
            CONFIG,
        ),
        dtype=float,
    )

    per_vals: list[list[float]] = [[] for _ in range(len(intervals))]
    for orig_idx, seg_val in zip(seg_df["orig_idx"], seg_vals, strict=False):
        per_vals[int(orig_idx)].append(float(seg_val))

    for i, vals in enumerate(per_vals):
        if vals:
            out[i] = _logsumexp(vals)
    return out


def _compute_filtered_pwm_max_pos(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """pwm.max.pos under filter: select position from segment with best pwm.max score."""
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    base_payload = dict(payload_eval)
    base_payload["sshift"] = 0
    base_payload["eshift"] = 0
    payload_score = dict(base_payload)
    payload_score["func"] = "pwm.max"
    payload_pos = dict(base_payload)
    payload_pos["func"] = "pwm.max.pos"

    seg_intervals = _df2pymisha(seg_df[["chrom", "start", "end"]])
    seg_scores = _numpy.asarray(_pymisha.pm_vtrack_compute(payload_score, seg_intervals, CONFIG), dtype=float)
    seg_pos = _numpy.asarray(_pymisha.pm_vtrack_compute(payload_pos, seg_intervals, CONFIG), dtype=float)

    best_score = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    for orig_idx, seg_start, base_start, score, pos in zip(
        seg_df["orig_idx"],
        seg_df["start"],
        seg_df["base_start"],
        seg_scores,
        seg_pos, strict=False,
    ):
        if _numpy.isnan(score) or _numpy.isnan(pos):
            continue
        i = int(orig_idx)
        if _numpy.isnan(best_score[i]) or score > best_score[i]:
            offset = int(seg_start) - int(base_start)
            sign = -1.0 if pos < 0 else 1.0
            mapped_pos = sign * (abs(float(pos)) + float(offset))
            out[i] = mapped_pos
            best_score[i] = float(score)

    return out


def _global_percentile_reference_values(src: str, bin_size: int) -> np.ndarray:
    key = (str(_shared._GROOT), str(src), int(bin_size))
    cached = _GLOBAL_PERCENTILE_CACHE.get(key)
    if cached is not None:
        return cached

    from .extract import gextract
    from .intervals import gintervals_all

    extracted = gextract(src, gintervals_all(), iterator=int(bin_size))
    if extracted is None or len(extracted) == 0:
        ref = _numpy.array([], dtype=float)
    else:
        data_cols = [c for c in extracted.columns if c not in {"chrom", "start", "end", "intervalID"}]
        if not data_cols:
            ref = _numpy.array([], dtype=float)
        else:
            ref = extracted[data_cols[0]].to_numpy(dtype=float, copy=False)
            ref = ref[~_numpy.isnan(ref)]
            ref.sort()
    _GLOBAL_PERCENTILE_CACHE[key] = ref
    return ref


def _percentile_from_reference(values: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    out = _numpy.full(values.shape, _numpy.nan, dtype=float)
    if ref_sorted.size == 0:
        return out
    valid = ~_numpy.isnan(values)
    if not valid.any():
        return out
    ranks = _numpy.searchsorted(ref_sorted, values[valid], side="right")
    out[valid] = ranks.astype(float) / float(ref_sorted.size)
    return out


def _load_pv_table(src: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read R's frozen ``vars/pv.percentiles`` binned quantile table for *src*.

    Returns ``(bins, breaks)`` - the percentile values and their track-value
    thresholds (same length) - or ``None`` if the track has no such file (a
    pymisha-created track, or one never prepared for percentile queries).
    """
    import os

    key = (str(_shared._GROOT), str(src))
    if key in _PV_TABLE_CACHE:
        return _PV_TABLE_CACHE[key]

    result: tuple[np.ndarray, np.ndarray] | None = None
    try:
        track_path = _pymisha.pm_track_path(src)
    except Exception:
        track_path = None
    if track_path:
        fpath = os.path.join(track_path, "vars", "pv.percentiles")
        if os.path.exists(fpath):
            from ._r_serialize import read as _r_read

            obj = _r_read(fpath)
            attrs = getattr(obj, "attributes", None)
            breaks = None if attrs is None else attrs.get("breaks")
            if breaks is not None:
                bins = _numpy.asarray(obj, dtype=float).ravel()
                br = _numpy.asarray(breaks, dtype=float).ravel()
                if bins.size == br.size and bins.size >= 2:
                    result = (bins, br)
    _PV_TABLE_CACHE[key] = result
    return result


def _percentile_from_pv_table(
    values: np.ndarray, bins: np.ndarray, breaks: np.ndarray
) -> np.ndarray:
    """Map per-bin statistics through R's binned pv.percentiles table.

    Mirrors R ``TrackVarProcessor`` + ``BinFinder::val2bin`` (right-closed
    bins): ``bin = val2bin(val)``; ``bins[bin]`` in range; for out-of-range,
    ``bins[0]`` when ``val <= breaks[0]`` else ``1.0``. NaN stays NaN.
    """
    from .summary import _bin_values

    out = _numpy.full(values.shape, _numpy.nan, dtype=float)
    valid = ~_numpy.isnan(values)
    if not valid.any():
        return out
    vals = values[valid]
    bin_idx = _bin_values(vals, breaks, include_lowest=False)
    res = _numpy.where(vals <= breaks[0], bins[0], 1.0)
    in_range = bin_idx >= 0
    res[in_range] = bins[bin_idx[in_range]]
    out[valid] = res
    return out


def _compute_filtered_global_percentile(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    func: str,
) -> np.ndarray:
    """global.percentile* under filter using raw unmasked bins and global dense reference."""
    from .tracks import gtrack_info

    src = payload_eval.get("src")
    if not isinstance(src, str):
        raise NotImplementedError("global.percentile* under filter requires a dense track source")

    info = gtrack_info(src)
    track_type = str(info.get("type", "")).lower()
    bin_size = info.get("bin_size") or info.get("bin.size")
    if track_type != "dense" or bin_size is None:
        raise NotImplementedError("global.percentile* under filter requires a dense track source")

    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    from collections.abc import Callable
    stat_fn: Callable[..., Any]
    if func == "global.percentile":
        stat_fn = _numpy.mean
    elif func == "global.percentile.min":
        stat_fn = _numpy.min
    else:
        stat_fn = _numpy.max

    stats = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    for orig_idx, raw_vals in groups.items():
        if raw_vals:
            stats[orig_idx] = float(stat_fn(raw_vals))

    table = _load_pv_table(src)
    if table is not None:
        return _percentile_from_pv_table(stats, table[0], table[1])
    ref = _global_percentile_reference_values(src, int(bin_size))
    return _percentile_from_reference(stats, ref)


def _compute_global_percentile_unfiltered(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    func: str,
) -> np.ndarray:
    """global.percentile* without filter using C++ per-interval stats + Python reference CDF."""
    from .tracks import gtrack_info

    src = payload_eval.get("src")
    if not isinstance(src, str):
        raise NotImplementedError("global.percentile* requires a dense track source")

    info = gtrack_info(src)
    track_type = str(info.get("type", "")).lower()
    bin_size = info.get("bin_size") or info.get("bin.size")
    if track_type != "dense" or bin_size is None:
        raise NotImplementedError("global.percentile* requires a dense track source")

    stat_payload = dict(payload_eval)
    if func == "global.percentile":
        stat_payload["func"] = "avg"
    elif func == "global.percentile.min":
        stat_payload["func"] = "min"
    else:
        stat_payload["func"] = "max"

    stats = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            stat_payload,
            _df2pymisha(intervals),
            CONFIG,
        ),
        dtype=float,
    )

    # Path A (R parity): map through R's frozen binned pv.percentiles table.
    # Path B (fallback): the exact empirical CDF over native bins, used when
    # the track has no pv.percentiles file (e.g. pymisha-created tracks).
    table = _load_pv_table(src)
    if table is not None:
        return _percentile_from_pv_table(stats, table[0], table[1])
    ref = _global_percentile_reference_values(src, int(bin_size))
    return _percentile_from_reference(stats, ref)


def _project_intervals_by_dim(intervals: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Project 2D intervals to 1D by selecting a dimension.

    Parameters
    ----------
    intervals : DataFrame
        Intervals DataFrame, potentially with 2D columns
        (chrom1/start1/end1/chrom2/start2/end2).
    dim : int
        Dimension to project onto: 1 for (chrom1, start1, end1),
        2 for (chrom2, start2, end2).

    Returns
    -------
    DataFrame
        1D intervals with columns (chrom, start, end).
    """
    if dim == 1:
        return _pandas.DataFrame({
            "chrom": intervals["chrom1"].values,
            "start": intervals["start1"].values,
            "end": intervals["end1"].values,
        })
    if dim == 2:
        return _pandas.DataFrame({
            "chrom": intervals["chrom2"].values,
            "start": intervals["start2"].values,
            "end": intervals["end2"].values,
        })
    raise ValueError(f"dim must be 1 or 2, got {dim}")


def _compute_array_slice_vtrack(
    vtrack_config: dict,
    intervals: _pandas.DataFrame,
) -> _numpy.ndarray:
    """Evaluate an ``array_slice`` virtual track for *intervals*.

    Returns a 1-D float64 array (one value per row of *intervals*).
    """
    from pathlib import Path

    from ._array_track import extract_array, read_colnames, reduce_array_extract

    src = vtrack_config["src"]
    slice_cols = vtrack_config.get("slice_cols")  # None = all columns
    func = vtrack_config.get("func", "avg")

    track_path = Path(_pymisha.pm_track_path(src))
    colnames = read_colnames(track_path)

    if slice_cols is None:
        sel_idx = None
        val_cols = colnames
    else:
        sel_idx = slice_cols
        val_cols = [colnames[i] for i in sel_idx]

    n_intervals = len(intervals)

    # extract_array returns one row per overlapping track interval.
    # intervalID is 1-based, matching the row order of *intervals*.
    extracted = extract_array(track_path, intervals, sel_idx, colnames)

    return reduce_array_extract(extracted, val_cols, func, n_intervals)


def _compute_vtrack_values(vtrack_name: str, intervals: pd.DataFrame) -> Any:
    """
    Compute values for a virtual track.

    Virtual tracks are evaluated by:
    1. Creating shifted intervals (sshift, eshift)
    2. Extracting source track values for shifted intervals
    3. Applying aggregation function
    4. Mapping results back to original intervals

    Returns numpy array of computed values (one per interval).
    """
    vtrack_config = _shared._VTRACKS.get(vtrack_name)
    if vtrack_config is None:
        return None

    # Array-slice vtracks are handled entirely in Python.
    if vtrack_config.get("kind") == "array_slice":
        return _compute_array_slice_vtrack(vtrack_config, intervals)

    # Handle dim parameter: project 2D intervals to 1D
    dim = vtrack_config.get("dim")
    if dim is not None and dim != 0 and "chrom1" in intervals.columns:
        intervals = _project_intervals_by_dim(intervals, dim)

    payload = dict(vtrack_config)
    src = payload.get('src')
    if isinstance(src, _pandas.DataFrame):
        payload['src_df'] = src.copy()
        payload['src'] = _df2pymisha(src)

    # Ensure pssm is passed as a numpy array with correct ordering
    if 'pssm' in payload and isinstance(payload['pssm'], _pandas.DataFrame):
        payload['pssm'] = payload['pssm'].to_numpy(dtype=float, copy=False)

    func = str(payload.get("func", "avg")).lower()

    # Functions that the C++ value-based vtrack path handles efficiently
    _VALUE_CPP_FUNCS = {
        "avg", "mean", "sum", "min", "max", "first", "last", "size",
        "exists", "stddev", "std", "quantile", "sample", "lse",
    }

    filter_df = payload.get("filter")
    if filter_df is None or (isinstance(filter_df, _pandas.DataFrame) and len(filter_df) == 0):
        if payload.get("src_df") is not None and func in _VALUE_DF_PY_FUNCS and func not in _VALUE_CPP_FUNCS:
            return _compute_value_df_vtrack(
                intervals,
                payload,
                _pandas.DataFrame(columns=["chrom", "start", "end"]),
            )
        # C++ backend does not support global.percentile* yet.
        if func in _FILTER_GLOBAL_PERCENTILE_FUNCS:
            payload_eval = dict(payload)
            payload_eval.pop("filter", None)
            payload_eval.pop("filter_key", None)
            payload_eval.pop("filter_stats", None)
            return _compute_global_percentile_unfiltered(intervals, payload_eval, func)
        return _pymisha.pm_vtrack_compute(
            payload,
            _df2pymisha(intervals),
            CONFIG
        )

    payload_eval = dict(payload)
    payload_eval.pop("filter", None)
    payload_eval.pop("filter_key", None)
    payload_eval.pop("filter_stats", None)

    if payload.get("src_df") is not None and func in _VALUE_DF_PY_FUNCS:
        return _compute_value_df_vtrack(intervals, payload_eval, filter_df)

    if func in _FILTER_PASSTHROUGH_FUNCS:
        return _pymisha.pm_vtrack_compute(payload_eval, _df2pymisha(intervals), CONFIG)

    if func not in _FILTER_SUPPORTED_FUNCS:
        raise NotImplementedError(
            f"gvtrack.filter for function '{func}' is not yet supported in PyMisha"
        )

    # --- nearest: first-unmasked-segment semantics ---
    if func in _FILTER_NEAREST_FUNCS:
        return _compute_filtered_nearest(intervals, payload_eval, filter_df)

    # --- quantile: raw-value extraction + numpy quantile ---
    if func in _FILTER_QUANTILE_FUNCS:
        return _compute_filtered_quantile(intervals, payload_eval, filter_df)

    # --- stddev: raw-value extraction + numpy stddev ---
    if func in _FILTER_STDDEV_FUNCS:
        return _compute_filtered_stddev(intervals, payload_eval, filter_df)

    # --- lse: raw-value extraction + logsumexp ---
    if func == "lse":
        return _compute_filtered_lse(intervals, payload_eval, filter_df)

    # --- global.percentile*: percentile of filtered per-interval statistic ---
    if func in _FILTER_GLOBAL_PERCENTILE_FUNCS:
        return _compute_filtered_global_percentile(intervals, payload_eval, filter_df, func)

    # --- pwm: combine segment scores by logsumexp ---
    if func == "pwm":
        return _compute_filtered_segment_logsumexp(intervals, payload_eval, filter_df)

    # --- pwm.max.pos: pick the position from the segment with highest pwm.max ---
    if func in _FILTER_PWM_MAX_POS_FUNCS:
        return _compute_filtered_pwm_max_pos(intervals, payload_eval, filter_df)

    # --- max.pos.* / min.pos.*: find position of extremum across unmasked bins ---
    if func in _FILTER_MAX_POS_ABS_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "max", False)
    if func in _FILTER_MAX_POS_REL_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "max", True)
    if func in _FILTER_MIN_POS_ABS_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "min", False)
    if func in _FILTER_MIN_POS_REL_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "min", True)

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    payload_eval["sshift"] = 0
    payload_eval["eshift"] = 0
    seg_vals = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            payload_eval,
            _df2pymisha(seg_df[["chrom", "start", "end"]]),
            CONFIG,
        ),
        dtype=float,
    )

    per_vals: list[list[float]] = [[] for _ in range(len(intervals))]
    per_lens: list[list[int]] = [[] for _ in range(len(intervals))]
    per_starts: list[list[int]] = [[] for _ in range(len(intervals))]
    per_base_starts: list[int | None] = [None for _ in range(len(intervals))]
    for orig_idx, seg_len, seg_start, base_start, seg_val in zip(
        seg_df["orig_idx"],
        seg_df["seg_len"],
        seg_df["start"],
        seg_df["base_start"],
        seg_vals, strict=False,
    ):
        i = int(orig_idx)
        per_vals[i].append(float(seg_val))
        per_lens[i].append(int(seg_len))
        per_starts[i].append(int(seg_start))
        if per_base_starts[i] is None:
            per_base_starts[i] = int(base_start)

    for i in range(len(intervals)):
        vals = per_vals[i]
        if not vals:
            continue
        arr = _numpy.asarray(vals, dtype=float)
        valid = ~_numpy.isnan(arr)
        if func in _FILTER_WEIGHTED_FUNCS:
            if not valid.any():
                continue
            lens = _numpy.asarray(per_lens[i], dtype=float)[valid]
            arr_valid = arr[valid]
            out[i] = float((arr_valid * lens).sum() / lens.sum())
        elif func in _FILTER_ADDITIVE_FUNCS:
            if not valid.any():
                continue
            out[i] = float(arr[valid].sum())
        elif func in _FILTER_MIN_FUNCS:
            if not valid.any():
                continue
            out[i] = float(arr[valid].min())
        elif func in _FILTER_MAX_FUNCS:
            if not valid.any():
                continue
            out[i] = float(arr[valid].max())
        elif func in _FILTER_EXISTS_FUNCS:
            out[i] = 1.0 if _numpy.any(arr == 1.0) else 0.0
        elif func in _FILTER_SIZE_FUNCS:
            out[i] = float(_numpy.nansum(arr))
        elif func in _FILTER_FIRST_FUNCS:
            out[i] = float(arr[0])
        elif func in _FILTER_LAST_FUNCS:
            out[i] = float(arr[-1])
        elif func in _FILTER_SAMPLE_FUNCS:
            candidates = arr[valid]
            if candidates.size == 0:
                continue
            idx = int(_numpy.random.randint(candidates.size))
            out[i] = float(candidates[idx])
        elif func in _FILTER_FIRST_POS_ABS_FUNCS:
            out[i] = float(arr[0])
        elif func in _FILTER_FIRST_POS_REL_FUNCS:
            if _numpy.isnan(arr[0]):
                continue
            out[i] = float(arr[0] + per_starts[i][0] - (per_base_starts[i] or 0))
        elif func in _FILTER_LAST_POS_ABS_FUNCS:
            out[i] = float(arr[-1])
        elif func in _FILTER_LAST_POS_REL_FUNCS:
            if _numpy.isnan(arr[-1]):
                continue
            out[i] = float(arr[-1] + per_starts[i][-1] - (per_base_starts[i] or 0))
        elif func in _FILTER_SAMPLE_POS_ABS_FUNCS:
            candidates = arr[valid]
            if candidates.size == 0:
                continue
            idx = int(_numpy.random.randint(candidates.size))
            out[i] = float(candidates[idx])
        elif func in _FILTER_SAMPLE_POS_REL_FUNCS:
            if not valid.any():
                continue
            seg_starts = _numpy.asarray(per_starts[i], dtype=float)[valid]
            abs_candidates = arr[valid] + seg_starts
            idx = int(_numpy.random.randint(abs_candidates.size))
            out[i] = float(abs_candidates[idx] - (per_base_starts[i] or 0))
    return out


def gvtrack_create(
    vtrack_name: str,
    src: pd.DataFrame | str | None,
    func: str | None = None,
    params: float | str | None = None,
    sshift: int = 0,
    eshift: int = 0,
    **kwargs: Any,
) -> None:
    """
    Create a virtual track.

    A virtual track evaluates an aggregation function over a source track,
    intervals set, or genomic sequence within each iterator interval. Virtual
    tracks can be referenced by name anywhere a track expression is accepted
    (e.g., in ``gextract``, ``gsummary``, ``gdist``). The virtual track
    persists in memory for the duration of the current session.

    Parameters
    ----------
    vtrack_name : str
        Name for the virtual track. If a virtual track with this name
        already exists, it is silently overwritten.
    src : str, pandas.DataFrame, or None
        Source for the virtual track. Can be:

        - A track name (str) -- any track in the database (dense, sparse,
          array, or 2D).
        - An intervals set name (str) -- used with interval-based functions
          like ``'distance'``, ``'coverage'``.
        - A DataFrame with columns ``chrom``, ``start``, ``end`` and one
          numeric value column -- acts as an in-memory sparse (value-based)
          track. Intervals must not overlap.
        - ``None`` -- for sequence-based functions (``'pwm'``, ``'pwm.max'``,
          ``'pwm.count'``, ``'kmer.count'``, ``'kmer.frac'``,
          ``'masked.count'``, ``'masked.frac'``).
    func : str, optional
        Aggregation function to apply. When omitted, defaults to ``'avg'``
        for a track or value-bearing source and to ``'distance'`` for an
        intervals-set source (R parity). Supported functions include:

        - **Track-based**: ``'avg'``, ``'sum'``, ``'min'``, ``'max'``,
          ``'stddev'``, ``'nearest'``, ``'quantile'``, ``'coverage'``,
          ``'exists'``, ``'size'``, ``'first'``, ``'last'``, ``'sample'``,
          ``'lse'``, ``'global.percentile'``
        - **Distance-based** (intervals source): ``'distance'``,
          ``'distance.center'``, ``'distance.edge'``, ``'neighbor.count'``
        - **Position-based**: ``'first.pos.abs'``, ``'first.pos.relative'``,
          ``'last.pos.abs'``, ``'last.pos.relative'``,
          ``'min.pos.abs'``, ``'min.pos.relative'``,
          ``'max.pos.abs'``, ``'max.pos.relative'``,
          ``'sample.pos.abs'``, ``'sample.pos.relative'``
        - **2D track**: ``'area'``, ``'weighted.sum'``, ``'exists'``,
          ``'size'``, ``'first'``, ``'last'``, ``'sample'``,
          ``'global.percentile'``
        - **Motif/PWM** (src=None): ``'pwm'``, ``'pwm.max'``,
          ``'pwm.max.pos'``, ``'pwm.count'``
        - **Edit distance** (src=None): ``'pwm.edit_distance'``,
          ``'pwm.edit_distance.pos'``, ``'pwm.max.edit_distance'``,
          ``'pwm.edit_distance.lse'``, ``'pwm.edit_distance.lse.pos'``
        - **K-mer** (src=None): ``'kmer.count'``, ``'kmer.frac'``
        - **Masked sequence** (src=None): ``'masked.count'``,
          ``'masked.frac'``
    params : float, str, or None, optional
        Function-specific parameter. For example, a percentile in [0, 1]
        for ``'quantile'``, a max-distance integer for ``'neighbor.count'``,
        or a score threshold for ``'pwm.count'``.
    sshift : int, default 0
        Shift added to the start coordinate of each iterator interval
        before the virtual track function is evaluated.
    eshift : int, default 0
        Shift added to the end coordinate of each iterator interval
        before the virtual track function is evaluated.
    **kwargs
        Additional keyword arguments, depending on ``func``:

        - ``pssm`` (numpy.ndarray or pandas.DataFrame) -- Position-specific
          scoring matrix with 4 columns (A, C, G, T) for PWM functions.
        - ``prior`` (float) -- Pseudocount added to PSSM frequencies
          (default 0.01 for PWM functions).
        - ``bidirect`` (bool) -- If True, score both DNA strands (PWM).
        - ``extend`` (bool) -- If True (default), extend the scanned
          sequence so boundary-anchored motifs retain full context.
        - ``score_thresh`` (float) -- Score threshold for ``'pwm.count'``
          and edit distance functions.
        - ``max_edits`` (int or None) -- Maximum number of edits for edit
          distance functions. None (default) uses exact computation.
        - ``max_indels`` (int) -- Maximum insertions+deletions for
          ``'pwm.edit_distance'``, ``'pwm.edit_distance.pos'``,
          ``'pwm.max.edit_distance'``. Default 0 (substitutions only).
        - ``direction`` (str) -- Score direction for edit distance
          functions: ``'above'`` (default) finds minimum edits to raise
          score above threshold; ``'below'`` finds minimum edits to
          lower score below threshold.
        - ``score_min`` (float or None) -- Minimum PWM score filter for
          edit distance functions. Windows below this are skipped.
        - ``score_max`` (float or None) -- Maximum PWM score filter for
          edit distance functions. Windows above this are skipped.
        - ``strand`` (int) -- Strand selection: 1 (forward), -1 (reverse),
          0 (both). Used by kmer and single-strand PWM modes.
        - ``kmer`` (str) -- DNA k-mer sequence for kmer functions.
        - ``spat_factor`` (list of float) -- Spatial weighting factors
          for PWM functions.
        - ``spat_bin`` (int) -- Bin width for spatial weighting.
        - ``spat_min`` (int) -- Minimum scan position (1-based).
        - ``spat_max`` (int) -- Maximum scan position (1-based).
        - ``filter`` (pandas.DataFrame, str, list, or None) -- Genomic mask
          filter. See ``gvtrack_filter`` for details.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the filter source is invalid or refers to a non-intervals-type
        track.

    See Also
    --------
    gvtrack_info : Retrieve the configuration of a virtual track.
    gvtrack_iterator : Override iterator shifts for a virtual track.
    gvtrack_iterator_2d : Set 2D iterator shifts for a virtual track.
    gvtrack_filter : Attach or clear a genomic mask filter.
    gvtrack_rm : Remove a single virtual track.
    gvtrack_ls : List all virtual tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Create a virtual track with a max aggregation:

    >>> pm.gvtrack_create("vt_max", "dense_track", func="max")
    >>> pm.gextract("vt_max", pm.gintervals(["1"], [0], [10000]), iterator=1000)  # doctest: +SKIP

    Create a quantile virtual track with a median (0.5) parameter:

    >>> pm.gvtrack_create("vt_q50", "dense_track", func="quantile", params=0.5)

    Create a distance virtual track from an intervals source:

    >>> pm.gvtrack_create("vt_dist", "annotations", func="distance")

    Create a PWM virtual track scanning both strands:

    >>> import numpy as np
    >>> pssm = np.array([[0.7, 0.1, 0.1, 0.1],
    ...                  [0.1, 0.7, 0.1, 0.1],
    ...                  [0.1, 0.1, 0.7, 0.1],
    ...                  [0.1, 0.1, 0.1, 0.7]])
    >>> pm.gvtrack_create("motif", None, func="pwm",
    ...                   pssm=pssm, bidirect=True, prior=0.01)

    Create a k-mer counting virtual track:

    >>> pm.gvtrack_create("cg_count", None, func="kmer.count",
    ...                   kmer="CG", strand=1)
    """
    _checkroot()
    if func is None:
        func = _infer_default_vtrack_func(src)
    func_lc = str(func).lower()

    if isinstance(src, _pandas.DataFrame):
        from .intervals import _normalize_chroms

        req = {"chrom", "start", "end"}
        if not req.issubset(src.columns):
            raise ValueError("DataFrame source must include columns: chrom, start, end")

        is_interval_func = func_lc in _DF_INTERVAL_FUNCS
        value_cols = [
            c
            for c in src.columns
            if c not in {"chrom", "start", "end", "intervalID", "intervalID1", "intervalID2"}
        ]
        if not is_interval_func and not value_cols:
            raise ValueError("DataFrame source must include one value column")

        if is_interval_func:
            src_df = src.copy()
        else:
            value_col = value_cols[0]
            src_df = src[["chrom", "start", "end", value_col]].copy()
            src_df.columns = ["chrom", "start", "end", "value"]

        src_df["chrom"] = _normalize_chroms(src_df["chrom"].astype(str).tolist())
        src_df["start"] = _pandas.to_numeric(src_df["start"], errors="coerce").astype("Int64")
        src_df["end"] = _pandas.to_numeric(src_df["end"], errors="coerce").astype("Int64")
        if not is_interval_func:
            src_df["value"] = _pandas.to_numeric(src_df["value"], errors="coerce")
            src_df = src_df.dropna(subset=["start", "end", "value"]).copy()
        else:
            src_df = src_df.dropna(subset=["start", "end"]).copy()
        src_df["start"] = src_df["start"].astype(int)
        src_df["end"] = src_df["end"].astype(int)
        src_df = src_df[src_df["end"] > src_df["start"]]
        src_df = src_df.sort_values(["chrom", "start", "end"], kind="mergesort").reset_index(drop=True)

        prev_end = src_df.groupby("chrom")["end"].shift(1)
        has_overlaps = bool((src_df["start"] < prev_end).fillna(False).any())
        if has_overlaps and (func_lc == "distance.center" or not is_interval_func):
            raise ValueError("overlapping intervals in DataFrame source are not allowed for this function")

        src = src_df

    config = {
        'src': src,
        'func': func,
        'params': params,
        'sshift': sshift,
        'eshift': eshift,
    }
    config.update(kwargs)

    if str(config.get("func", "")).startswith("pwm"):
        spat_factor = config.get("spat_factor")
        if spat_factor is not None:
            try:
                spat_vals = _numpy.asarray(spat_factor, dtype=float).ravel()
            except Exception as exc:
                raise ValueError("spat_factor must be a numeric array-like") from exc
            if spat_vals.size == 0:
                raise ValueError("spat_factor must be non-empty")
            if not _numpy.isfinite(spat_vals).all():
                raise ValueError("spat_factor must contain only finite values")
            if _numpy.any(spat_vals <= 0):
                raise ValueError("spat_factor must contain only positive values")
            config["spat_factor"] = spat_vals.tolist()

        if "spat_bin" in config and config.get("spat_bin") is not None:
            try:
                spat_bin = int(config["spat_bin"])  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError("spat_bin must be a positive integer") from exc
            if spat_bin <= 0:
                raise ValueError("spat_bin must be a positive integer")
            config["spat_bin"] = spat_bin

    # Validate direction parameter for edit distance functions
    if func_lc in _FILTER_EDIT_DISTANCE_FUNCS:
        direction = config.get("direction", "above")
        if direction is not None:
            direction = str(direction).lower()
            if direction not in ("above", "below"):
                raise ValueError("direction must be 'above' or 'below'")
            config["direction"] = direction

    # For PWM virtual tracks, if pssm is a DataFrame, ensure column order A, C, G, T
    if str(config.get('func', '')).startswith('pwm'):
        pssm = config.get('pssm')
        if isinstance(pssm, _pandas.DataFrame):
            # Check if we have A, C, G, T columns (case-insensitive)
            cols = [c.upper() for c in pssm.columns]
            if {'A', 'C', 'G', 'T'}.issubset(set(cols)):
                # Reorder to standard ACGT
                col_map = {c.upper(): c for c in pssm.columns}
                pssm = pssm[[col_map['A'], col_map['C'], col_map['G'], col_map['T']]]
                config['pssm'] = pssm

    filt = config.get("filter")
    if filt is not None:
        filt_df = _resolve_filter_sources(filt)
        config["filter"] = filt_df if len(filt_df) > 0 else None
        config["filter_key"] = _filter_key(filt_df)
        config["filter_stats"] = _filter_stats(filt_df)
    else:
        config["filter"] = None
        config["filter_key"] = None
        config["filter_stats"] = None

    _shared._VTRACKS[vtrack_name] = config


def gvtrack_ls() -> list[str]:
    """
    List all currently defined virtual tracks.

    Returns the names of all virtual tracks that have been created in the
    current session via ``gvtrack_create``. Unlike the R counterpart, this
    function does not support pattern filtering; use standard Python list
    comprehensions to filter the result if needed.

    Returns
    -------
    list of str
        Names of all virtual tracks in the current session. Returns an
        empty list if no virtual tracks have been created.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_info : Retrieve configuration of a virtual track.
    gvtrack_rm : Remove a single virtual track.
    gvtrack_clear : Remove all virtual tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_ls()
    []

    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> pm.gvtrack_create("vt2", "dense_track", func="max")
    >>> pm.gvtrack_ls()
    ['vt1', 'vt2']

    Filter with a list comprehension:

    >>> [v for v in pm.gvtrack_ls() if "2" in v]
    ['vt2']
    """
    return list(_shared._VTRACKS.keys())


def gvtrack_info(vtrack_name: str) -> dict[str, Any]:
    """
    Return the definition of a virtual track.

    Retrieves the full internal configuration dictionary for a previously
    created virtual track. This is useful for inspecting or programmatically
    modifying virtual track settings.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.

    Returns
    -------
    dict
        A copy of the virtual track configuration dictionary. Keys always
        include ``'src'``, ``'func'``, ``'params'``, ``'sshift'``,
        ``'eshift'``, ``'filter'``, ``'filter_key'``, and
        ``'filter_stats'``. Additional keys (e.g., ``'pssm'``,
        ``'bidirect'``, ``'kmer'``, ``'dim'``) are present when supplied
        at creation time or via ``gvtrack_iterator`` /
        ``gvtrack_iterator_2d``.

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_ls : List all virtual tracks.
    gvtrack_filter : Attach or clear a genomic mask filter.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_create("vt1", "dense_track", func="max")
    >>> info = pm.gvtrack_info("vt1")
    >>> info["func"]
    'max'
    >>> info["src"]
    'dense_track'
    >>> info["sshift"]
    0
    """
    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")
    return dict(_shared._VTRACKS[vtrack_name])


def gvtrack_iterator(vtrack_name: str, dim: int | None = None, sshift: int = 0, eshift: int = 0) -> None:
    """
    Define modification rules for the 1D iterator of a virtual track.

    By default a virtual track is evaluated over the same iterator intervals
    as the calling function (e.g., ``gextract``, ``gsummary``). This function
    allows independent control of the genomic window the virtual track sees
    by applying custom start/end shifts. It can also project a 2D iterator
    down to one of its 1D dimensions.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.
    dim : int or None, optional
        Dimension projection for 2D iterators:

        - ``None`` or ``0`` -- no conversion; shifts apply to the 1D
          iterator directly.
        - ``1`` -- convert a 2D iterator interval ``(chrom1, start1, end1,
          chrom2, start2, end2)`` to ``(chrom1, start1, end1)`` before
          applying shifts.
        - ``2`` -- convert to ``(chrom2, start2, end2)`` before applying
          shifts.
    sshift : int, default 0
        Value added to the start coordinate of each iterator interval.
        Negative values expand the window upstream.
    eshift : int, default 0
        Value added to the end coordinate of each iterator interval.
        Positive values expand the window downstream.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_iterator_2d : Set 2D iterator shifts for a virtual track.
    gvtrack_filter : Attach a genomic mask filter.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Shift the evaluation window 200 bp downstream:

    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> pm.gvtrack_iterator("vt1", sshift=200, eshift=200)
    >>> pm.gextract("dense_track", "vt1",  # doctest: +SKIP
    ...             pm.gintervals(["1"], [0], [500]))

    Expand the window symmetrically by 500 bp in each direction:

    >>> pm.gvtrack_create("vt2", "dense_track", func="sum")
    >>> pm.gvtrack_iterator("vt2", sshift=-500, eshift=500)

    Project dimension 1 of a 2D iterator for a 1D virtual track:

    >>> pm.gvtrack_create("vt3", "dense_track", func="avg")
    >>> pm.gvtrack_iterator("vt3", dim=1)
    """
    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")

    cfg = _shared._VTRACKS[vtrack_name]
    if dim is not None:
        cfg['dim'] = dim
    cfg['sshift'] = sshift
    cfg['eshift'] = eshift
    _shared._VTRACKS[vtrack_name] = cfg


def gvtrack_iterator_2d(
    vtrack_name: str,
    sshift1: int = 0,
    eshift1: int = 0,
    sshift2: int = 0,
    eshift2: int = 0,
) -> None:
    """
    Define modification rules for the 2D iterator of a virtual track.

    Sets independent start/end shifts for both dimensions of a 2D iterator
    interval. The shifts are added to the coordinates of each 2D iterator
    interval before the virtual track function is evaluated.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.
    sshift1 : int, default 0
        Value added to the ``start1`` coordinate of each 2D iterator
        interval.
    eshift1 : int, default 0
        Value added to the ``end1`` coordinate of each 2D iterator
        interval.
    sshift2 : int, default 0
        Value added to the ``start2`` coordinate of each 2D iterator
        interval.
    eshift2 : int, default 0
        Value added to the ``end2`` coordinate of each 2D iterator
        interval.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_iterator : Set 1D iterator shifts or project a 2D dimension.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_create("vt_2d", "rects_track", func="area")
    >>> pm.gvtrack_iterator_2d("vt_2d", sshift1=1000, eshift1=2000)
    >>> pm.gextract("rects_track", "vt_2d",  # doctest: +SKIP
    ...             pm.gintervals_2d(["1"], [0], [5000], ["2"], [0], [5000]))
    """
    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")

    cfg = _shared._VTRACKS[vtrack_name]
    cfg['itr_type'] = '2d'
    cfg['sshift1'] = sshift1
    cfg['eshift1'] = eshift1
    cfg['sshift2'] = sshift2
    cfg['eshift2'] = eshift2
    _shared._VTRACKS[vtrack_name] = cfg


def gvtrack_filter(vtrack_name: str, mask: pd.DataFrame | str | list[Any] | None = None, **kwargs: Any) -> None:
    """
    Attach or clear a genomic mask filter on a virtual track.

    When a filter is attached, the virtual track function is evaluated only
    over the *unmasked* regions -- that is, regions NOT covered by the filter
    intervals. Masked positions are excluded from aggregation, and an
    iterator interval that is entirely masked returns NaN. The filter
    persists on the virtual track until explicitly cleared.

    Filters are applied *after* iterator modifiers (``sshift``/``eshift``/
    ``dim``). The order of operations is: (1) apply iterator shifts,
    (2) subtract mask from the shifted intervals, (3) evaluate the virtual
    track function over the remaining unmasked segments.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.
    mask : pandas.DataFrame, str, list, or None
        The genomic mask to apply. Accepted forms:

        - A ``pandas.DataFrame`` with columns ``chrom``, ``start``,
          ``end`` -- intervals to mask.
        - A ``str`` naming an intervals set in the database.
        - A ``str`` naming an intervals-type (sparse) track.
        - A ``list`` or ``tuple`` of any combination of the above; all
          sources are unified into a single mask.
        - ``None`` -- clears any existing filter from the virtual track.
    filter : pandas.DataFrame, str, list, or None
        Backward-compatible alias for ``mask``.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.
    ValueError
        If a string filter source is not a recognized intervals set or
        intervals-type track, or if a DataFrame is missing required
        columns.

    See Also
    --------
    gvtrack_create : Create a virtual track (filter can also be set at
        creation time via the ``filter`` keyword argument).
    gvtrack_info : Inspect a virtual track's configuration including its
        filter.
    gvtrack_iterator : Set iterator shifts (applied before the filter).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Attach a filter to exclude specific regions:

    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> mask = pm.gintervals(["1", "1"], [100, 500], [200, 600])
    >>> pm.gvtrack_filter("vt1", filter=mask)
    >>> pm.gvtrack_info("vt1")["filter"] is not None
    True

    Clear the filter:

    >>> pm.gvtrack_filter("vt1", filter=None)
    >>> pm.gvtrack_info("vt1")["filter"] is None
    True

    Use multiple filter sources (automatically unified):

    >>> mask1 = pm.gintervals(["1"], [100], [200])
    >>> mask2 = pm.gintervals(["1"], [500], [600])
    >>> pm.gvtrack_filter("vt1", filter=[mask1, mask2])
    """
    if "filter" in kwargs:
        if mask is not None:
            raise ValueError("Specify only one of 'mask' or 'filter'")
        mask = kwargs.pop("filter")
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {bad}")

    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")

    cfg = dict(_shared._VTRACKS[vtrack_name])
    if mask is None:
        cfg["filter"] = None
        cfg["filter_key"] = None
        cfg["filter_stats"] = None
        _shared._VTRACKS[vtrack_name] = cfg
        return

    filter_df = _resolve_filter_sources(mask)
    cfg["filter"] = filter_df if len(filter_df) > 0 else None
    cfg["filter_key"] = _filter_key(filter_df)
    cfg["filter_stats"] = _filter_stats(filter_df)
    _shared._VTRACKS[vtrack_name] = cfg
    return


def gvtrack_rm(vtrack_name: str) -> None:
    """
    Remove a virtual track.

    Deletes a single virtual track from the current session. If the named
    virtual track does not exist, the call is silently ignored (no error is
    raised).

    Parameters
    ----------
    vtrack_name : str
        Name of the virtual track to remove.

    Returns
    -------
    None

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_clear : Remove all virtual tracks at once.
    gvtrack_ls : List all virtual tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_create("vt1", "dense_track", func="max")
    >>> pm.gvtrack_create("vt2", "dense_track", func="min")
    >>> pm.gvtrack_ls()
    ['vt1', 'vt2']
    >>> pm.gvtrack_rm("vt1")
    >>> pm.gvtrack_ls()
    ['vt2']

    Removing a non-existent track is a no-op:

    >>> pm.gvtrack_rm("does_not_exist")
    """
    if vtrack_name in _shared._VTRACKS:
        del _shared._VTRACKS[vtrack_name]


def gvtrack_clear() -> None:
    """
    Remove all virtual tracks.

    Clears the entire virtual track registry for the current session.
    After this call, ``gvtrack_ls()`` returns an empty list. This is
    useful for resetting state between analyses or in test fixtures.

    Returns
    -------
    None

    See Also
    --------
    gvtrack_rm : Remove a single virtual track by name.
    gvtrack_ls : List all virtual tracks.
    gvtrack_create : Create a new virtual track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> pm.gvtrack_create("vt2", "dense_track", func="max")
    >>> len(pm.gvtrack_ls())
    2
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_ls()
    []
    """
    _shared._VTRACKS.clear()


def gvtrack_array_slice(
    vtrack: str,
    slice: list[str] | list[int] | None = None,
    func: str = "avg",
    params=None,
) -> None:
    """Configure an existing virtual track as an array-slice aggregator.

    Mirrors R ``gvtrack.array.slice``. The vtrack must already exist (created
    via ``gvtrack_create(name, src=<array_track>)``). This function mutates
    the vtrack in place, setting the column selection and aggregation function.
    The vtrack can then be referenced in ``gextract``, ``gsummary``, etc.
    and returns one value per iterator interval.

    Parameters
    ----------
    vtrack : str
        Name of an existing virtual track. The vtrack must have been created
        with an array track as its source. Raises ``ValueError`` if the vtrack
        does not exist or its source is not an array track.
    slice : list of str or int, optional
        Column subset to use. Strings are matched against track colnames;
        integers are 0-based column indices. ``None`` uses all columns.
    func : str, default ``'avg'``
        Aggregation function applied across all non-NaN values in the
        selected columns for each iterator interval.  Supported values:
        ``'avg'``, ``'min'``, ``'max'``, ``'sum'``, ``'stddev'``
        (R's ``stdev`` is also accepted).
    params : None
        Reserved for future use. Must be ``None``; raises
        ``NotImplementedError`` if provided.

    See Also
    --------
    gvtrack_create : Create a virtual track from any source.
    gvtrack_rm : Remove a virtual track.
    gtrack_array_extract : Extract per-position values from an array track.
    gtrack_array_get_colnames : List column names of an array track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_create("v_avg", src="array_track")
    >>> pm.gvtrack_array_slice("v_avg", func="avg")
    >>> pm.gvtrack_create("v_col0", src="array_track")
    >>> pm.gvtrack_array_slice("v_col0", slice=["col0"], func="avg")
    >>> pm.gvtrack_ls()
    ['v_avg', 'v_col0']
    >>> pm.gvtrack_clear()
    """
    _shared._checkroot()

    # Vtrack must already exist.
    if vtrack not in _shared._VTRACKS:
        raise ValueError(f"gvtrack_array_slice: no such vtrack '{vtrack}'")

    existing = _shared._VTRACKS[vtrack]
    src = existing.get("src")

    # src must be a string (track name) pointing to an array track.
    if not isinstance(src, str):
        raise ValueError(
            f"gvtrack_array_slice: vtrack '{vtrack}' is not an array track vtrack "
            f"(src is not a track name)"
        )

    from .tracks import gtrack_exists, gtrack_info

    if not gtrack_exists(src):
        raise ValueError(
            f"gvtrack_array_slice: vtrack '{vtrack}' source track '{src}' does not exist"
        )
    info = gtrack_info(src)
    if info.get("type") != "array":
        raise ValueError(
            f"gvtrack_array_slice: vtrack '{vtrack}' source track '{src}' is not an array track "
            f"(type={info.get('type')!r}). "
            f"Use gvtrack_create for dense/sparse tracks."
        )

    from ._array_track import ARRAY_REDUCE_FUNCS

    # "quantile" is supported via the C++ scanner (StreamPercentiler), in
    # addition to the reducers the pure-Python path implements.
    _ALLOWED_SLICE_FUNCS = ARRAY_REDUCE_FUNCS | {"quantile"}
    func_lc = func.lower()
    if func_lc not in _ALLOWED_SLICE_FUNCS:
        raise ValueError(
            f"gvtrack_array_slice: func must be one of "
            f"{sorted(_ALLOWED_SLICE_FUNCS)!r}, got {func!r}"
        )

    # quantile requires a percentile in params; other reducers take no params.
    if func_lc == "quantile":
        pval = params[0] if isinstance(params, (list, tuple)) else params
        if pval is None:
            raise ValueError(
                "gvtrack_array_slice: func='quantile' requires params=<percentile in [0,1]>"
            )
        pval = float(pval)
        if not (0.0 <= pval <= 1.0):
            raise ValueError(
                f"gvtrack_array_slice: quantile percentile must be in [0, 1], got {pval}"
            )
        params = pval
    elif params is not None:
        raise ValueError(
            f"gvtrack_array_slice: params is only used with func='quantile', got func={func!r}"
        )

    # Resolve slice to a list of 0-based int indices (or None = all columns)
    if slice is not None:
        from pathlib import Path

        from ._array_track import read_colnames

        track_path = Path(_pymisha.pm_track_path(src))
        colnames = read_colnames(track_path)
        slice_list: list[Any] = list(slice)
        if all(isinstance(s, str) for s in slice_list):
            str_list: list[str] = [str(s) for s in slice_list]
            cn_idx = {name: i for i, name in enumerate(colnames)}
            bad_names = [s for s in str_list if s not in cn_idx]
            if bad_names:
                raise ValueError(
                    f"gvtrack_array_slice: column(s) not found in '{src}': {bad_names!r}"
                )
            slice_cols: list[int] | None = [cn_idx[s] for s in str_list]
        elif all(isinstance(s, (int, _numpy.integer)) for s in slice_list):
            int_list: list[int] = [int(s) for s in slice_list]
            ncols = len(colnames)
            bad_idx = [s for s in int_list if s < 0 or s >= ncols]
            if bad_idx:
                raise ValueError(
                    f"gvtrack_array_slice: column indices out of range [0, {ncols}): {bad_idx!r}"
                )
            slice_cols = int_list
        else:
            raise ValueError(
                "gvtrack_array_slice: slice must be a list of strings (column names) "
                "or ints (0-based column indices)"
            )
    else:
        slice_cols = None

    # Mutate the existing vtrack config in place.
    _shared._VTRACKS[vtrack] = {
        "kind": "array_slice",
        "src": src,
        "slice_cols": slice_cols,  # None = all columns
        "func": func_lc,
        "params": params,  # percentile for func="quantile", else None
    }
