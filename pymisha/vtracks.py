"""Virtual track utilities and API."""

import hashlib

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
)

_GLOBAL_PERCENTILE_CACHE = {}


def _canonicalize_filter_df(df):
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


def _resolve_filter_sources(filter_obj):
    from .extract import gextract
    from .intervals import gintervals_all, gintervals_load, gintervals_ls, gintervals_union
    from .tracks import gtrack_info

    if filter_obj is None:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])

    if isinstance(filter_obj, _pandas.DataFrame):
        return _canonicalize_filter_df(filter_obj)

    if isinstance(filter_obj, list | tuple):
        if len(filter_obj) == 0:
            return _pandas.DataFrame(columns=["chrom", "start", "end"])
        merged = None
        for part in filter_obj:
            part_df = _resolve_filter_sources(part)
            if part_df is None or len(part_df) == 0:
                continue
            merged = part_df if merged is None else gintervals_union(merged, part_df)
        if merged is None:
            return _pandas.DataFrame(columns=["chrom", "start", "end"])
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


def _filter_key(filter_df):
    if filter_df is None or len(filter_df) == 0:
        return None
    h = hashlib.sha1()
    for row in filter_df.itertuples(index=False):
        h.update(f"{row.chrom}\t{int(row.start)}\t{int(row.end)}\n".encode())
    return h.hexdigest()


def _filter_stats(filter_df):
    if filter_df is None or len(filter_df) == 0:
        return None
    total = int((filter_df["end"] - filter_df["start"]).sum())
    chroms = int(filter_df["chrom"].nunique())
    return {"num_chroms": chroms, "total_bases": total}


def _subtract_masks(start, end, masks):
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


def _build_unmasked_segments(intervals, payload, filter_df):
    from .intervals import gintervals_all

    sshift = int(payload.get("sshift", 0) or 0)
    eshift = int(payload.get("eshift", 0) or 0)
    chrom_sizes_df = gintervals_all()
    chrom_sizes = {
        str(chrom): int(end)
        for chrom, end in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)
    }
    mask_map = {}
    if filter_df is not None and len(filter_df) > 0:
        for row in filter_df.itertuples(index=False):
            chrom = str(row.chrom)
            mask_map.setdefault(chrom, []).append((int(row.start), int(row.end)))
        for chrom in list(mask_map.keys()):
            mask_map[chrom].sort()

    seg_rows = []
    for idx, row in enumerate(intervals.itertuples(index=False)):
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
                seg_rows.append((chrom, int(s), int(e), idx, int(e - s), int(start)))
    if not seg_rows:
        return None
    return _pandas.DataFrame(
        seg_rows,
        columns=["chrom", "start", "end", "orig_idx", "seg_len", "base_start"],
    )


def _compute_filtered_nearest(intervals, payload_eval, filter_df):
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


def _extract_raw_unmasked_values(intervals, payload_eval, filter_df):
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
    groups = {}
    for oi, v in zip(mapped_orig, mapped_vals, strict=False):
        groups.setdefault(int(oi), []).append(float(v))

    return groups, out


def _extract_raw_unmasked_values_with_positions(intervals, payload_eval, filter_df):
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

    groups = {}
    for oi, v, s in zip(mapped_orig, mapped_vals, mapped_starts, strict=False):
        groups.setdefault(int(oi), []).append((float(v), int(s)))

    # Collect base_start (shifted interval start) per original interval
    base_starts = {}
    for row in seg_df.itertuples(index=False):
        oi = int(row.orig_idx)
        if oi not in base_starts:
            base_starts[oi] = int(row.base_start)

    return groups, base_starts, out


def _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, mode, relative):
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
    src_arr = src  # list: [colnames, chrom_arr, start_arr, end_arr, val_arr, ...]
    src_chroms_raw = [str(c) for c in src_arr[1]]
    src_chroms = _normalize_chroms(src_chroms_raw)
    src_starts = _numpy.asarray(src_arr[2], dtype=int)
    src_ends = _numpy.asarray(src_arr[3], dtype=int)
    src_vals = _numpy.asarray(src_arr[4], dtype=float)

    best_extremum_val = {}
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


def _compute_filtered_stddev(intervals, payload_eval, filter_df):
    """Stddev under filter: extract raw bin values and compute exact stddev."""
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups.items():
        if len(raw_vals) >= 2:
            out[orig_idx] = float(_numpy.std(raw_vals, ddof=1))
    return out


def _compute_filtered_quantile(intervals, payload_eval, filter_df):
    """Quantile under filter: extract raw bin values and compute exact quantile."""
    percentile = float(payload_eval.get("params", 0.5) or 0.5)
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups.items():
        out[orig_idx] = float(_numpy.quantile(raw_vals, percentile))
    return out


def _logsumexp(values):
    arr = _numpy.asarray(values, dtype=float)
    arr = arr[~_numpy.isnan(arr)]
    if arr.size == 0:
        return _numpy.nan
    m = float(arr.max())
    if _numpy.isneginf(m):
        return m
    return float(m + _numpy.log(_numpy.exp(arr - m).sum()))


def _compute_filtered_lse(intervals, payload_eval, filter_df):
    """LSE under filter from raw unmasked source values."""
    if not isinstance(payload_eval.get("src"), str):
        raise NotImplementedError("lse under filter currently requires a track source")
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups.items():
        out[orig_idx] = _logsumexp(raw_vals)
    return out


def _compute_filtered_segment_logsumexp(intervals, payload_eval, filter_df):
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

    per_vals = [[] for _ in range(len(intervals))]
    for orig_idx, seg_val in zip(seg_df["orig_idx"], seg_vals, strict=False):
        per_vals[int(orig_idx)].append(float(seg_val))

    for i, vals in enumerate(per_vals):
        if vals:
            out[i] = _logsumexp(vals)
    return out


def _compute_filtered_pwm_max_pos(intervals, payload_eval, filter_df):
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


def _global_percentile_reference_values(src, bin_size):
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


def _percentile_from_reference(values, ref_sorted):
    out = _numpy.full(values.shape, _numpy.nan, dtype=float)
    if ref_sorted.size == 0:
        return out
    valid = ~_numpy.isnan(values)
    if not valid.any():
        return out
    ranks = _numpy.searchsorted(ref_sorted, values[valid], side="right")
    out[valid] = ranks.astype(float) / float(ref_sorted.size)
    return out


def _compute_filtered_global_percentile(intervals, payload_eval, filter_df, func):
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

    ref = _global_percentile_reference_values(src, int(bin_size))
    return _percentile_from_reference(stats, ref)


def _compute_global_percentile_unfiltered(intervals, payload_eval, func):
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

    ref = _global_percentile_reference_values(src, int(bin_size))
    return _percentile_from_reference(stats, ref)


def _compute_vtrack_values(vtrack_name, intervals):
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

    payload = dict(vtrack_config)
    src = payload.get('src')
    if isinstance(src, _pandas.DataFrame):
        payload['src'] = _df2pymisha(src)

    # Ensure pssm is passed as a numpy array with correct ordering
    if 'pssm' in payload and isinstance(payload['pssm'], _pandas.DataFrame):
        payload['pssm'] = payload['pssm'].to_numpy(dtype=float, copy=True)

    func = str(payload.get("func", "avg")).lower()

    filter_df = payload.get("filter")
    if filter_df is None or (isinstance(filter_df, _pandas.DataFrame) and len(filter_df) == 0):
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

    per_vals = [[] for _ in range(len(intervals))]
    per_lens = [[] for _ in range(len(intervals))]
    per_starts = [[] for _ in range(len(intervals))]
    per_base_starts = [None for _ in range(len(intervals))]
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
            out[i] = float(arr[0] + per_starts[i][0] - per_base_starts[i])
        elif func in _FILTER_LAST_POS_ABS_FUNCS:
            out[i] = float(arr[-1])
        elif func in _FILTER_LAST_POS_REL_FUNCS:
            if _numpy.isnan(arr[-1]):
                continue
            out[i] = float(arr[-1] + per_starts[i][-1] - per_base_starts[i])
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
            out[i] = float(abs_candidates[idx] - per_base_starts[i])
    return out


def gvtrack_create(vtrack_name, src, func='avg', params=None, sshift=0, eshift=0, **kwargs):
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
    func : str, default 'avg'
        Aggregation function to apply. Supported functions include:

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
        - **2D track**: ``'area'``, ``'weighted.sum'``
        - **Motif/PWM** (src=None): ``'pwm'``, ``'pwm.max'``,
          ``'pwm.max.pos'``, ``'pwm.count'``
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
        - ``score_thresh`` (float) -- Score threshold for ``'pwm.count'``.
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

    config = {
        'src': src,
        'func': func,
        'params': params,
        'sshift': sshift,
        'eshift': eshift,
    }
    config.update(kwargs)

    # For PWM virtual tracks, if pssm is a DataFrame, ensure column order A, C, G, T
    if config.get('func', '').startswith('pwm'):
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


def gvtrack_ls():
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


def gvtrack_info(vtrack_name):
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


def gvtrack_iterator(vtrack_name, dim=None, sshift=0, eshift=0):
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


def gvtrack_iterator_2d(vtrack_name, sshift1=0, eshift1=0, sshift2=0, eshift2=0):
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


def gvtrack_filter(vtrack_name, mask=None, **kwargs):
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


def gvtrack_rm(vtrack_name):
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


def gvtrack_clear():
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
