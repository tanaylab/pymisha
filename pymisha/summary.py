"""Summary, quantile, and distribution APIs."""

from . import _shared
from ._safe_eval import UnsafeExpressionError, compile_safe_expression
from ._shared import (
    CONFIG,
    _bound_colname,
    _checkroot,
    _chunk_slices,
    _df2pymisha,
    _numpy,
    _pandas,
    _progress_context,
    _pymisha,
    _pymisha2df,
)
from .expr import _expr_safe_name, _find_vtracks_in_expr, _parse_expr_vars
from .extract import _is_2d_intervals, gextract
from .intervals import gintervals_all
from .vtracks import _compute_vtrack_values


def _validate_expr_security(expr, track_names=None, vtrack_names=None):
    if track_names is None:
        track_names = set(_pymisha.pm_track_names())
    if vtrack_names is None:
        vtrack_names = set(_shared._VTRACKS.keys())

    expr_eval, expr_tracks, expr_vtracks, _ = _parse_expr_vars(
        expr, track_names, vtrack_names
    )
    allowed_names = {
        "np",
        "numpy",
        "CHROM",
        "START",
        "END",
        *(_expr_safe_name(t) for t in expr_tracks),
        *(_expr_safe_name(vt) for vt in expr_vtracks),
    }
    compile_safe_expression(expr_eval, allowed_names)


def gdist(*args, intervals=None, include_lowest=False, iterator=None,
          band=None, dataframe=False, names=None, **kwargs):
    """
    Calculate distribution of track expressions over bins.

    This function calculates the distribution of values of numeric track
    expressions over the given set of bins using a memory-efficient C++
    streaming implementation.

    Parameters
    ----------
    *args : pairs of (expr, breaks)
        Alternating track expressions and their bin breaks.
        Example: gdist("track1", [0, 0.5, 1], "track2", [0, 1, 2])
    intervals : DataFrame, optional
        Genomic scope for which the function is applied.
        If None, uses all genomic intervals.
    include_lowest : bool, default False
        If True, the lowest value will be included in the first bin.
        Example: breaks=[0, 0.2, 0.5] creates (0, 0.2], (0.2, 0.5].
        With include_lowest=True: [0, 0.2], (0.2, 0.5].
    iterator : int or str, optional
        Track expression iterator. If None, determined implicitly.
    band : optional
        Track expression band (not yet supported).
    dataframe : bool, default False
        If True, return a DataFrame instead of an N-dimensional array.
    names : list of str, optional
        Column names for the expressions in the returned DataFrame
        (only relevant when dataframe=True).

    Returns
    -------
    ndarray or DataFrame
        If dataframe=False: N-dimensional array where N is the number of
        expr-breaks pairs. The shape is (n_bins_1, n_bins_2, ..., n_bins_N).
        If dataframe=True: DataFrame with columns for each track expression
        (bin labels) and an 'n' column with counts.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Calculate distribution of dense_track for bins (0, 0.2], (0.2, 0.5], (0.5, 1]:

    >>> pm.gdist("dense_track", [0, 0.2, 0.5, 1])  # doctest: +SKIP

    Calculate 2D distribution - dense_track vs sparse_track:

    >>> pm.gdist("dense_track", [0, 0.5, 1], "sparse_track", [0, 1, 2],
    ...          iterator=100)  # doctest: +SKIP

    Get result as DataFrame:

    >>> pm.gdist("dense_track", [0, 0.2, 0.5, 1], dataframe=True)  # doctest: +SKIP

    See Also
    --------
    gsummary, gquantiles, gpartition
    """
    # Parse arguments: (expr1, breaks1, expr2, breaks2, ...)
    if len(args) < 2:
        raise ValueError("Usage: gdist([expr, breaks]+, intervals=None, include_lowest=False, iterator=None)")

    if len(args) % 2 != 0:
        raise ValueError("gdist requires pairs of (expression, breaks) arguments")

    _checkroot()
    if intervals is None:
        intervals = gintervals_all()

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gdist")

    exprs = []
    breaks_list = []

    for i in range(0, len(args), 2):
        expr = args[i]
        breaks = args[i + 1]

        if not isinstance(expr, str):
            raise TypeError(f"Expression at position {i} must be a string")

        breaks = _numpy.asarray(breaks, dtype=float)
        if breaks.ndim != 1:
            raise ValueError(f"Breaks at position {i+1} must be 1D array")
        if len(breaks) < 2:
            raise ValueError(f"Breaks at position {i+1} must have at least 2 elements")

        # Validate breaks are strictly increasing
        if not _numpy.all(_numpy.diff(breaks) > 0):
            raise ValueError(f"Breaks at position {i+1} must be strictly increasing")

        exprs.append(expr)
        breaks_list.append(breaks.tolist())

    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())
    for expr in exprs:
        try:
            _validate_expr_security(expr, track_names=track_names, vtrack_names=vtrack_names)
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    # Calculate number of bins for each dimension
    n_bins = [len(b) - 1 for b in breaks_list]

    # Band or 2D intervals require extract-then-bin path
    if band is not None or _is_2d_intervals(intervals):
        result = _gdist_band_path(
            exprs, breaks_list, n_bins, intervals, include_lowest, band,
            progress=progress, progress_desc=progress_desc,
        )
        if dataframe:
            return _array_to_dataframe(result, breaks_list, exprs, names, include_lowest)
        return result

    # Check if any expression uses virtual tracks
    # If so, fall back to Python implementation (virtual tracks not supported in pm_dist yet)
    has_vtracks = any(_find_vtracks_in_expr(e) for e in exprs)

    if not has_vtracks:
        # Use C++ streaming implementation
        config = dict(CONFIG)
        if progress is not None:
            config['progress'] = progress
        if progress_desc:
            config['progress_desc'] = progress_desc

        result = _pymisha.pm_dist(
            exprs,
            breaks_list,
            _df2pymisha(intervals),
            iterator,
            include_lowest,
            config
        )

        if dataframe:
            return _array_to_dataframe(result, breaks_list, exprs, names, include_lowest)

        return result

    # Streaming vtrack path: extract and bin values in chunks to keep memory bounded.
    result = _gdist_vtrack_streaming(
        exprs, breaks_list, n_bins, intervals, include_lowest,
        iterator=iterator, progress=progress, progress_desc=progress_desc,
    )

    if dataframe:
        return _array_to_dataframe(result, breaks_list, exprs, names, include_lowest)

    return result


_INTERVAL_META_COLS = {"chrom", "start", "end", "chrom1", "start1", "end1",
                      "chrom2", "start2", "end2", "intervalID"}


def _gdist_band_path(exprs, breaks_list, n_bins, intervals, include_lowest, band,
                     *, progress=None, progress_desc=None):
    """Compute distribution from 2D extraction with band filter."""
    counts = _numpy.zeros(n_bins, dtype=_numpy.int64)

    for i, expr in enumerate(exprs):
        result = gextract(expr, intervals, band=band,
                          progress=progress, progress_desc=progress_desc)
        if result is None or len(result) == 0:
            continue

        data_cols = [c for c in result.columns if c not in _INTERVAL_META_COLS]
        if not data_cols:
            continue
        values = result[data_cols[0]].to_numpy(dtype=float, copy=False)
        bin_idx = _bin_values(values, _numpy.asarray(breaks_list[i], dtype=float), include_lowest)

        # Single-dim: accumulate directly
        if len(exprs) == 1:
            valid_idx = bin_idx[bin_idx >= 0]
            if valid_idx.size:
                counts += _numpy.bincount(valid_idx, minlength=counts.shape[0])
        else:
            # Multi-dim would require co-extraction which 2D doesn't support yet
            raise NotImplementedError(
                "Multi-expression gdist with band is not yet supported"
            )

    return counts


def _bin_values(values, breaks, include_lowest):
    """Assign values to bin indices following C++ BinFinder semantics.

    Bins are ``(breaks[i], breaks[i+1]]`` — open on left, closed on right.
    With *include_lowest*, the first bin becomes ``[breaks[0], breaks[1]]``.

    Returns an int32 array of bin indices (0-based).  Out-of-range or NaN
    values receive -1.
    """
    breaks = _numpy.asarray(breaks, dtype=float)
    n_bins = len(breaks) - 1
    indices = _numpy.searchsorted(breaks, values, side='right') - 1

    # Exclude values at or below breaks[0] (bins are open on the left)
    indices[values <= breaks[0]] = -1

    if include_lowest:
        # First bin is [breaks[0], breaks[1]]
        indices[values == breaks[0]] = 0

    # Exclude values above breaks[-1]
    indices[values > breaks[-1]] = -1

    # Exclude NaN values
    indices[_numpy.isnan(values)] = -1

    # Clamp any stray indices
    indices[(indices < 0) | (indices >= n_bins)] = -1

    return indices.astype(_numpy.int32)


def _gdist_vtrack_streaming(exprs, breaks_list, n_bins, intervals,
                            include_lowest, *, iterator=None,
                            progress=None, progress_desc=None):
    """Streaming gdist for expressions containing virtual tracks.

    Instead of materializing all expression values and then binning, this
    function extracts values in chunks and accumulates bin counts per chunk.
    This keeps peak memory proportional to chunk_size rather than total
    number of iterated intervals.
    """
    # Parse all expressions to find physical tracks and vtracks
    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())

    parsed = []
    used_tracks = set()
    used_vtracks = set()

    for e in exprs:
        new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(
            e, track_names, vtrack_names
        )
        used_tracks.update(expr_tracks)
        used_vtracks.update(expr_vtracks)
        parsed.append((e, new_expr, expr_tracks, expr_vtracks))

    # Get iterated intervals (physical tracks define the iterator if present)
    if used_tracks:
        track_exprs = list(used_tracks)
        base_result = _pymisha.pm_extract(
            track_exprs, _df2pymisha(intervals), iterator, CONFIG
        )
        base_df = _pymisha2df(base_result)
        if base_df is None or len(base_df) == 0:
            return _numpy.zeros(n_bins, dtype=int)

        track_arrays = {}
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)
        iter_df = base_df[["chrom", "start", "end", "intervalID"]].copy()
    else:
        from ._shared import _iterated_intervals
        iter_df = _iterated_intervals(intervals, iterator)
        track_arrays = {}

    if iter_df is None or len(iter_df) == 0:
        return _numpy.zeros(n_bins, dtype=int)

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)

    # Compile expressions
    compiled = []
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
        try:
            compiled.append(compile_safe_expression(expr_eval, allowed_names))
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc

    # Accumulate bin counts across chunks
    result = _numpy.zeros(n_bins, dtype=int)
    breaks_arrays = [_numpy.asarray(b, dtype=float) for b in breaks_list]

    chrom_vals = iter_df["chrom"].values
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    with _progress_context(progress, total=n_rows, desc=progress_desc or "gdist") as progress_cb:
        for start_idx, end_idx in _chunk_slices(n_rows, chunk_size):
            sl = slice(start_idx, end_idx)

            local_ns = {
                'np': _numpy,
                'numpy': _numpy,
                'CHROM': chrom_vals[sl],
                'START': start_vals[sl],
                'END': end_vals[sl],
            }

            for tname, arr in track_arrays.items():
                local_ns[_expr_safe_name(tname)] = arr[sl]

            if used_vtracks:
                chunk_intervals = iter_df.iloc[start_idx:end_idx][
                    ["chrom", "start", "end"]
                ]
                for vt in used_vtracks:
                    local_ns[_expr_safe_name(vt)] = _compute_vtrack_values(
                        vt, chunk_intervals
                    )

            # Evaluate each expression and bin
            chunk_indices = []
            for i, code_obj in enumerate(compiled):
                vals = eval(code_obj, {'__builtins__': {}}, local_ns)
                if _numpy.isscalar(vals):
                    vals = _numpy.full(end_idx - start_idx, vals)
                vals = _numpy.asarray(vals, dtype=float)
                chunk_indices.append(
                    _bin_values(vals, breaks_arrays[i], include_lowest)
                )

            # Accumulate valid bin combinations
            chunk_len = end_idx - start_idx
            valid = _numpy.ones(chunk_len, dtype=bool)
            for idx_arr in chunk_indices:
                valid &= idx_arr >= 0

            if valid.any():
                valid_idx = [idx[valid] for idx in chunk_indices]
                if len(exprs) == 1:
                    _numpy.add.at(result, valid_idx[0], 1)
                else:
                    flat = _numpy.ravel_multi_index(valid_idx, result.shape)
                    result += _numpy.bincount(flat, minlength=result.size).reshape(result.shape)

            if progress_cb:
                pct = int(end_idx * 100.0 / n_rows) if n_rows else 100
                progress_cb(end_idx, n_rows, pct)

        if progress_cb:
            progress_cb(n_rows, n_rows, 100)

    return result


def _array_to_dataframe(result, breaks_list, exprs, names, include_lowest=False):
    """Convert N-dimensional result array to DataFrame."""
    import itertools

    # Create bin labels for each dimension
    bin_labels_list = []
    for breaks in breaks_list:
        labels = []
        for i in range(len(breaks) - 1):
            # Format: "(low, high]" or "[low, high]" for first bin if include_lowest
            left = "[" if i == 0 and include_lowest else "("
            labels.append(f"{left}{breaks[i]}, {breaks[i+1]}]")
        bin_labels_list.append(labels)

    # Generate all combinations of bin labels
    if len(exprs) == 1:
        rows = [(label, count) for label, count in zip(bin_labels_list[0], result.flatten(), strict=False)]
        col_names = names if names else exprs
        df = _pandas.DataFrame(rows, columns=[col_names[0], 'n'])
    else:
        rows = []
        for indices in itertools.product(*[range(len(labels)) for labels in bin_labels_list]):
            row = [bin_labels_list[dim][idx] for dim, idx in enumerate(indices)]
            row.append(result[indices])
            rows.append(row)

        col_names = names if names else exprs
        df = _pandas.DataFrame(rows, columns=list(col_names) + ['n'])

    return df


def _gsummary_vtrack_streaming(expr, intervals, iterator=None, progress=None, progress_desc=None):
    """Streaming gsummary for expressions containing virtual tracks."""
    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())

    new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(
        expr, track_names, vtrack_names
    )

    # Get iterated intervals
    if expr_tracks:
        track_exprs = list(expr_tracks)
        base_result = _pymisha.pm_extract(
            track_exprs, _df2pymisha(intervals), iterator, CONFIG
        )
        base_df = _pymisha2df(base_result)
        if base_df is None or len(base_df) == 0:
            return _pandas.Series(
                [0.0, 0.0, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan],
                index=["Total intervals", "NaN intervals", "Min", "Max", "Sum", "Mean", "Std dev"],
            )

        track_arrays = {}
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)
        iter_df = base_df[["chrom", "start", "end", "intervalID"]].copy()
    else:
        from ._shared import _iterated_intervals
        iter_df = _iterated_intervals(intervals, iterator)
        track_arrays = {}

    if iter_df is None or len(iter_df) == 0:
        return _pandas.Series(
            [0.0, 0.0, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan],
            index=["Total intervals", "NaN intervals", "Min", "Max", "Sum", "Mean", "Std dev"],
        )

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)

    allowed_names = {
        "np",
        "numpy",
        "CHROM",
        "START",
        "END",
        *(_expr_safe_name(t) for t in expr_tracks),
        *(_expr_safe_name(vt) for vt in expr_vtracks),
    }
    try:
        code_obj = compile_safe_expression(new_expr, allowed_names)
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    # Accumulators
    total_count = 0.0
    nan_count = 0.0
    val_sum = 0.0
    running_mean = 0.0
    running_m2 = 0.0
    valid_count = 0
    val_min = _numpy.inf
    val_max = -_numpy.inf

    chrom_vals = iter_df["chrom"].values
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    with _progress_context(progress, total=n_rows, desc=progress_desc or "gsummary") as progress_cb:
        for start_idx, end_idx in _chunk_slices(n_rows, chunk_size):
            sl = slice(start_idx, end_idx)

            local_ns = {
                'np': _numpy,
                'numpy': _numpy,
                'CHROM': chrom_vals[sl],
                'START': start_vals[sl],
                'END': end_vals[sl],
            }

            for tname, arr in track_arrays.items():
                local_ns[_expr_safe_name(tname)] = arr[sl]

            if expr_vtracks:
                chunk_intervals = iter_df.iloc[start_idx:end_idx][
                    ["chrom", "start", "end"]
                ]
                for vt in expr_vtracks:
                    local_ns[_expr_safe_name(vt)] = _compute_vtrack_values(
                        vt, chunk_intervals
                    )

            vals = eval(code_obj, {'__builtins__': {}}, local_ns)
            if _numpy.isscalar(vals):
                vals = _numpy.full(end_idx - start_idx, vals)
            vals = _numpy.asarray(vals, dtype=float)

            # Update stats
            total_count += vals.size
            nan_mask = _numpy.isnan(vals)
            n_nans = _numpy.count_nonzero(nan_mask)
            nan_count += n_nans

            if n_nans < vals.size:
                valid = vals[~nan_mask]
                val_sum += _numpy.sum(valid)
                val_min = min(val_min, float(_numpy.min(valid)))
                val_max = max(val_max, float(_numpy.max(valid)))

                # Parallel Welford merge: combine chunk stats with global stats.
                chunk_count = int(valid.size)
                chunk_mean = float(_numpy.mean(valid))
                chunk_m2 = float(_numpy.sum((valid - chunk_mean) ** 2))
                if valid_count == 0:
                    valid_count = chunk_count
                    running_mean = chunk_mean
                    running_m2 = chunk_m2
                else:
                    total_count_non_nan = valid_count + chunk_count
                    delta = chunk_mean - running_mean
                    running_mean += delta * (chunk_count / total_count_non_nan)
                    running_m2 += chunk_m2 + (delta * delta) * (
                        valid_count * chunk_count / total_count_non_nan
                    )
                    valid_count = total_count_non_nan

            if progress_cb:
                pct = int(end_idx * 100.0 / n_rows) if n_rows else 100
                progress_cb(end_idx, n_rows, pct)

        if progress_cb:
            progress_cb(n_rows, n_rows, 100)

    # Compute final stats
    num_non_nan = total_count - nan_count

    if num_non_nan == 0:
        mean_val = _numpy.nan
        stdev_val = _numpy.nan
        val_min = _numpy.nan
        val_max = _numpy.nan
        val_sum = _numpy.nan
    else:
        mean_val = running_mean
        stdev_val = _numpy.sqrt(running_m2 / (valid_count - 1)) if valid_count > 1 else _numpy.nan

    return _pandas.Series(
        [total_count, nan_count, val_min, val_max, val_sum, mean_val, stdev_val],
        index=["Total intervals", "NaN intervals", "Min", "Max", "Sum", "Mean", "Std dev"],
    )


def _reservoir_merge_chunk(samples, sample_size, stream_count, chunk_values, sample_cap, rng):
    """Merge a chunk of values into a fixed-size uniform reservoir."""
    if chunk_values.size == 0:
        return sample_size, stream_count

    offset = 0
    if sample_size < sample_cap:
        take = min(sample_cap - sample_size, chunk_values.size)
        samples[sample_size:sample_size + take] = chunk_values[:take]
        sample_size += take
        stream_count += take
        offset = take

    remaining = chunk_values[offset:]
    if remaining.size == 0:
        return sample_size, stream_count

    old_stream = stream_count
    new_total = old_stream + remaining.size
    # Uniformly select the number of items contributed from each source.
    take_from_chunk = int(rng.hypergeometric(remaining.size, old_stream, sample_cap))
    if take_from_chunk == 0:
        return sample_size, new_total

    take_from_old = sample_cap - take_from_chunk
    merged = _numpy.empty(sample_cap, dtype=float)

    if take_from_old > 0:
        old_idx = rng.choice(sample_cap, size=take_from_old, replace=False)
        merged[:take_from_old] = samples[old_idx]
    chunk_idx = rng.choice(remaining.size, size=take_from_chunk, replace=False)
    merged[take_from_old:] = remaining[chunk_idx]
    samples[:sample_cap] = merged
    return sample_cap, new_total


def _gquantiles_vtrack_streaming(expr, pct, intervals, iterator=None, progress=None, progress_desc=None):
    """Streaming gquantiles for expressions containing virtual tracks.

    Uses chunked expression evaluation and bounded reservoir sampling to keep
    memory proportional to ``CONFIG['max_data_size']``.
    """
    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())

    new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(
        expr, track_names, vtrack_names
    )

    # Get iterated intervals
    if expr_tracks:
        track_exprs = list(expr_tracks)
        base_result = _pymisha.pm_extract(
            track_exprs, _df2pymisha(intervals), iterator, CONFIG
        )
        base_df = _pymisha2df(base_result)
        if base_df is None or len(base_df) == 0:
            return _numpy.full(pct.shape, _numpy.nan, dtype=float), False

        track_arrays = {}
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)
        iter_df = base_df[["chrom", "start", "end", "intervalID"]].copy()
    else:
        from ._shared import _iterated_intervals
        iter_df = _iterated_intervals(intervals, iterator)
        track_arrays = {}

    if iter_df is None or len(iter_df) == 0:
        return _numpy.full(pct.shape, _numpy.nan, dtype=float), False

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)
    sample_cap = int(CONFIG.get("max_data_size", 10000000) or 10000000)
    if sample_cap <= 0:
        sample_cap = 1

    allowed_names = {
        "np",
        "numpy",
        "CHROM",
        "START",
        "END",
        *(_expr_safe_name(t) for t in expr_tracks),
        *(_expr_safe_name(vt) for vt in expr_vtracks),
    }
    try:
        code_obj = compile_safe_expression(new_expr, allowed_names)
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    chrom_vals = iter_df["chrom"].values
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    samples = _numpy.empty(sample_cap, dtype=float)
    sample_size = 0
    stream_count = 0
    rng = _numpy.random.default_rng()

    with _progress_context(progress, total=n_rows, desc=progress_desc or "gquantiles") as progress_cb:
        for start_idx, end_idx in _chunk_slices(n_rows, chunk_size):
            sl = slice(start_idx, end_idx)

            local_ns = {
                'np': _numpy,
                'numpy': _numpy,
                'CHROM': chrom_vals[sl],
                'START': start_vals[sl],
                'END': end_vals[sl],
            }

            for tname, arr in track_arrays.items():
                local_ns[_expr_safe_name(tname)] = arr[sl]

            if expr_vtracks:
                chunk_intervals = iter_df.iloc[start_idx:end_idx][["chrom", "start", "end"]]
                for vt in expr_vtracks:
                    local_ns[_expr_safe_name(vt)] = _compute_vtrack_values(vt, chunk_intervals)

            vals = eval(code_obj, {'__builtins__': {}}, local_ns)
            if _numpy.isscalar(vals):
                vals = _numpy.full(end_idx - start_idx, vals)
            vals = _numpy.asarray(vals, dtype=float)

            valid = vals[~_numpy.isnan(vals)]
            sample_size, stream_count = _reservoir_merge_chunk(
                samples,
                sample_size,
                stream_count,
                valid,
                sample_cap,
                rng,
            )

            if progress_cb:
                pct_done = int(end_idx * 100.0 / n_rows) if n_rows else 100
                progress_cb(end_idx, n_rows, pct_done)

        if progress_cb:
            progress_cb(n_rows, n_rows, 100)

    if sample_size == 0:
        return _numpy.full(pct.shape, _numpy.nan, dtype=float), False

    quantiles = _numpy.quantile(samples[:sample_size], pct)
    estimated = stream_count > sample_cap
    return quantiles, estimated


def _extract_expr_values(expr, intervals, iterator=None, band=None,
                         progress=None, progress_desc=None):
    result = gextract(expr, intervals, iterator=iterator, band=band,
                      progress=progress, progress_desc=progress_desc)
    if result is None:
        return _numpy.array([], dtype=float)

    if isinstance(result, _pandas.DataFrame):
        if len(result) == 0:
            return _numpy.array([], dtype=float)

        data_cols = [c for c in result.columns if c not in _INTERVAL_META_COLS]
        if not data_cols:
            return _numpy.array([], dtype=float)
        if len(data_cols) > 1:
            raise ValueError("Expected a single expression column for summary/quantiles")
        return result[data_cols[0]].to_numpy(dtype=float, copy=False)

    if isinstance(result, _numpy.ndarray):
        return result.astype(float, copy=False)

    raise TypeError("Unexpected gextract result type for summary/quantiles")


def _format_percentile(value):
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def _gsummary_from_values(values):
    """Compute summary statistics from an array of values."""
    stats = _compute_summary_stats(values)
    return _pandas.Series(
        stats,
        index=["Total intervals", "NaN intervals", "Min", "Max", "Sum", "Mean", "Std dev"],
    )


def gsummary(expr, intervals=None, iterator=None, **kwargs):
    """
    Calculate summary statistics of a track expression.

    Returns summary statistics: total bins, NaN count, min, max, sum, mean,
    and standard deviation of the values.

    Parameters
    ----------
    expr : str
        Track expression.
    intervals : DataFrame or str, optional
        Genomic scope. If None, uses ALLGENOME.
    iterator : int or str, optional
        Track expression iterator. If None, determined from expression.
    band : tuple of (int, int), optional
        Diagonal band for 2D tracks as ``(d1, d2)``.

    Returns
    -------
    pandas.Series
        Series with index:
        ``["Total intervals", "NaN intervals", "Min", "Max", "Sum", "Mean", "Std dev"]``.

    See Also
    --------
    gintervals_summary, gbins_summary, gquantiles

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gsummary("dense_track")  # doctest: +SKIP
    """
    if expr is None:
        raise ValueError("Usage: gsummary(expr, intervals=None, iterator=None)")

    band = kwargs.get("band")

    _checkroot()
    if intervals is None:
        intervals = gintervals_all()

    try:
        _validate_expr_security(expr)
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gsummary")

    # Band or 2D intervals require extract-then-summarize path
    if band is not None or _is_2d_intervals(intervals):
        values = _extract_expr_values(expr, intervals, iterator=iterator, band=band,
                                      progress=progress, progress_desc=progress_desc)
        return _gsummary_from_values(values)

    vtracks_used = _find_vtracks_in_expr(expr)
    if not vtracks_used:
        with _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_summary(expr, _df2pymisha(intervals), iterator, CONFIG)
        if result is None:
            return _pandas.Series(
                [0.0, 0.0, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan],
                index=[
                    "Total intervals",
                    "NaN intervals",
                    "Min",
                    "Max",
                    "Sum",
                    "Mean",
                    "Std dev",
                ],
            )
        if isinstance(result, dict):
            return _pandas.Series(result)
        if isinstance(result, _pandas.Series):
            return result
        raise TypeError("Unexpected pm_summary result type")

    # Streaming path for vtracks
    return _gsummary_vtrack_streaming(
        expr, intervals, iterator=iterator,
        progress=progress, progress_desc=progress_desc
    )


def gquantiles(expr, percentiles=0.5, intervals=None, iterator=None, **kwargs):
    """
    Calculate quantiles of a track expression.

    Computes quantiles for the given percentiles. If data size exceeds the
    configured limit, data is randomly sampled to fit.

    Parameters
    ----------
    expr : str
        Track expression.
    percentiles : array-like
        Percentiles in [0, 1] range.
    intervals : DataFrame or str, optional
        Genomic scope. If None, uses ALLGENOME.
    iterator : int or str, optional
        Track expression iterator. If None, determined from expression.
    band : tuple of (int, int), optional
        Diagonal band for 2D tracks.

    Returns
    -------
    numpy.ndarray
        Array of quantile values corresponding to the given percentiles.

    See Also
    --------
    gintervals_quantiles, gbins_quantiles, gdist

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gquantiles("dense_track", [0.25, 0.5, 0.75])  # doctest: +SKIP
    """
    if expr is None:
        raise ValueError("Usage: gquantiles(expr, percentiles=0.5, intervals=None, iterator=None)")

    if "percentile" in kwargs and percentiles == 0.5:
        percentiles = kwargs["percentile"]

    band = kwargs.get("band")

    _checkroot()
    if intervals is None:
        intervals = gintervals_all()

    try:
        _validate_expr_security(expr)
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gquantiles")

    pct = _numpy.asarray(percentiles, dtype=float)
    if pct.ndim == 0:
        pct = pct.reshape(1)
    if _numpy.any((pct < 0) | (pct > 1)):
        raise ValueError("percentiles must be within [0, 1]")

    # Band or 2D intervals require extract-then-quantile path
    if band is not None or _is_2d_intervals(intervals):
        values = _extract_expr_values(expr, intervals, iterator=iterator, band=band,
                                      progress=progress, progress_desc=progress_desc)
        if values.size == 0 or _numpy.all(_numpy.isnan(values)):
            quantiles = _numpy.full(pct.shape, _numpy.nan, dtype=float)
        else:
            quantiles = _numpy.nanquantile(values, pct)
        return _pandas.Series(quantiles, index=pct)

    vtracks_used = _find_vtracks_in_expr(expr)
    if not vtracks_used:
        with _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_quantiles(expr, pct.tolist(), _df2pymisha(intervals), iterator, CONFIG)
        if result is None:
            quantiles = _numpy.full(pct.shape, _numpy.nan, dtype=float)
        else:
            quantiles = _numpy.asarray(result, dtype=float)
        return _pandas.Series(quantiles, index=pct)

    quantiles, estimated = _gquantiles_vtrack_streaming(
        expr, pct, intervals, iterator=iterator, progress=progress, progress_desc=progress_desc
    )
    if estimated:
        import warnings
        warnings.warn(
            "Data size exceeds the limit; quantiles are approximate. "
            "Adjust CONFIG['max_data_size'] to increase the limit.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _pandas.Series(quantiles, index=pct)


def gintervals_summary(expr, intervals, iterator=None, **kwargs):
    """
    Compute summary statistics for each interval.

    Parameters
    ----------
    intervals_set_out : str, optional
        When provided, saves the resulting intervals set via ``gintervals_save``
        and returns ``None``.

    Returns
    -------
    DataFrame or None
        DataFrame with the original interval columns (chrom, start, end) plus
        summary statistic columns: Total intervals, NaN intervals, Min, Max,
        Sum, Mean, Std dev. Returns ``None`` if ``intervals_set_out`` is
        provided (result is saved to disk instead).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals([1, 2], 0, 5000)
    >>> pm.gintervals_summary("dense_track", intervs)  # doctest: +SKIP

    See Also
    --------
    gsummary, gbins_summary
    """
    if expr is None:
        raise ValueError("Usage: gintervals_summary(expr, intervals, iterator=None)")

    if intervals is None:
        return None

    intervals_set_out = kwargs.get("intervals_set_out")
    band = kwargs.get("band")

    _checkroot()

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gintervals_summary")

    vtracks_used = _find_vtracks_in_expr(expr)
    use_extract_path = bool(vtracks_used) or band is not None or _is_2d_intervals(intervals)
    if not use_extract_path:
        with _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_intervals_summary(expr, _df2pymisha(intervals), iterator, CONFIG)
        if result is None:
            out = intervals[["chrom", "start", "end"]].copy()
            out["Total intervals"] = 0.0
            out["NaN intervals"] = 0.0
            out["Min"] = _numpy.nan
            out["Max"] = _numpy.nan
            out["Sum"] = _numpy.nan
            out["Mean"] = _numpy.nan
            out["Std dev"] = _numpy.nan
            out = out.reset_index(drop=True)
        else:
            out = _pymisha2df(result)
    else:
        result = gextract(expr, intervals, iterator=iterator, band=band,
                          progress=progress, progress_desc=progress_desc)
        if result is None or len(result) == 0:
            out = intervals[["chrom", "start", "end"]].copy()
            out["Total intervals"] = 0.0
            out["NaN intervals"] = 0.0
            out["Min"] = _numpy.nan
            out["Max"] = _numpy.nan
            out["Sum"] = _numpy.nan
            out["Mean"] = _numpy.nan
            out["Std dev"] = _numpy.nan
            out = out.reset_index(drop=True)
        else:
            data_cols = [c for c in result.columns if c not in {"chrom", "start", "end", "intervalID"}]
            if len(data_cols) != 1:
                raise ValueError("Expected a single expression column for gintervals_summary")

            col = data_cols[0]
            out = intervals[["chrom", "start", "end"]].copy()
            n = len(out)
            out["Total intervals"] = 0.0
            out["NaN intervals"] = 0.0
            out["Min"] = _numpy.nan
            out["Max"] = _numpy.nan
            out["Sum"] = _numpy.nan
            out["Mean"] = _numpy.nan
            out["Std dev"] = _numpy.nan

            if "intervalID" in result.columns and n > 0:
                id_arr = result["intervalID"].to_numpy(dtype=int, copy=False)
                valid_id = (id_arr >= 1) & (id_arr <= n)
                if valid_id.any():
                    grouped = result.loc[valid_id].groupby("intervalID")[col]
                    total_counts = grouped.size()
                    stats = grouped.agg(["min", "max", "sum", "mean", "std", "count"])
                    idx = total_counts.index.to_numpy(dtype=int) - 1
                    out.iloc[idx, out.columns.get_loc("Total intervals")] = total_counts.to_numpy(dtype=float)
                    out.iloc[idx, out.columns.get_loc("NaN intervals")] = (
                        total_counts.to_numpy(dtype=float) - stats["count"].to_numpy(dtype=float)
                    )
                    out.iloc[idx, out.columns.get_loc("Min")] = stats["min"].to_numpy(dtype=float)
                    out.iloc[idx, out.columns.get_loc("Max")] = stats["max"].to_numpy(dtype=float)
                    out.iloc[idx, out.columns.get_loc("Sum")] = stats["sum"].to_numpy(dtype=float)
                    out.iloc[idx, out.columns.get_loc("Mean")] = stats["mean"].to_numpy(dtype=float)
                    out.iloc[idx, out.columns.get_loc("Std dev")] = stats["std"].to_numpy(dtype=float)
            out = out.reset_index(drop=True)
    if intervals_set_out is not None:
        from .intervals import gintervals_save
        gintervals_save(out, intervals_set_out)
        return None
    return out


def gintervals_quantiles(expr, percentiles=0.5, intervals=None, iterator=None, **kwargs):
    """
    Compute quantiles for each interval.

    Parameters
    ----------
    intervals_set_out : str, optional
        When provided, saves the resulting intervals set via ``gintervals_save``
        and returns ``None``.

    Returns
    -------
    DataFrame or None
        DataFrame with the original interval columns (chrom, start, end) plus
        one column per requested percentile, named by the percentile value
        (e.g., ``"0.5"``). Returns ``None`` if ``intervals_set_out`` is
        provided (result is saved to disk instead).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals([1, 2], 0, 5000)
    >>> pm.gintervals_quantiles("dense_track", percentiles=[0.5, 0.3, 0.9],
    ...                         intervals=intervs)  # doctest: +SKIP

    See Also
    --------
    gquantiles, gbins_quantiles
    """
    if expr is None:
        raise ValueError("Usage: gintervals_quantiles(expr, percentiles=0.5, intervals, iterator=None)")

    if intervals is None:
        return None

    if "percentile" in kwargs and percentiles == 0.5:
        percentiles = kwargs["percentile"]

    intervals_set_out = kwargs.get("intervals_set_out")
    band = kwargs.get("band")

    _checkroot()

    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "gintervals_quantiles")

    pct = _numpy.asarray(percentiles, dtype=float)
    if pct.ndim == 0:
        pct = pct.reshape(1)
    if _numpy.any((pct < 0) | (pct > 1)):
        raise ValueError("percentiles must be within [0, 1]")

    vtracks_used = _find_vtracks_in_expr(expr)
    use_extract_path = bool(vtracks_used) or band is not None or _is_2d_intervals(intervals)
    if not use_extract_path:
        with _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_intervals_quantiles(expr, pct.tolist(), _df2pymisha(intervals), iterator, CONFIG)
        if result is None:
            out = intervals[["chrom", "start", "end"]].copy()
            for p in pct:
                out[_format_percentile(p)] = _numpy.nan
            out = out.reset_index(drop=True)
        else:
            out = _pymisha2df(result)
    else:
        result = gextract(expr, intervals, iterator=iterator, band=band,
                          progress=progress, progress_desc=progress_desc)
        if result is None or len(result) == 0:
            out = intervals[["chrom", "start", "end"]].copy()
            for p in pct:
                out[_format_percentile(p)] = _numpy.nan
            out = out.reset_index(drop=True)
        else:
            data_cols = [c for c in result.columns if c not in {"chrom", "start", "end", "intervalID"}]
            if len(data_cols) != 1:
                raise ValueError("Expected a single expression column for gintervals_quantiles")

            col = data_cols[0]

            # Group by intervalID and calculate quantiles
            # Note: intervalID is 1-based index
            grouped = result.groupby("intervalID")[col]

            # grouped.quantile(list) returns MultiIndex (intervalID, quantile)
            # unstack(-1) moves quantile to columns
            q_res = grouped.quantile(pct.tolist())

            if isinstance(q_res.index, _pandas.MultiIndex):
                q_df = q_res.unstack(-1)
            else:
                # Should not happen with list input, but fallback
                if isinstance(q_res, _pandas.Series):
                    q_df = q_res.to_frame()
                    if len(pct) == 1:
                        q_df.columns = pct
                else:
                    q_df = q_res

            num_intervals = len(intervals)
            all_ids = _pandas.Index(_numpy.arange(1, num_intervals + 1), name="intervalID")

            q_df = q_df.reindex(all_ids)

            out = intervals[["chrom", "start", "end"]].copy()
            for p in pct:
                if p in q_df.columns:
                    out[_format_percentile(p)] = q_df[p].values
                else:
                    out[_format_percentile(p)] = _numpy.nan
            out = out.reset_index(drop=True)

    if intervals_set_out is not None:
        from .intervals import gintervals_save
        gintervals_save(out, intervals_set_out)
        return None
    return out


def gpartition(expr, breaks, intervals=None, include_lowest=False, iterator=None, **kwargs):
    """
    Partition track expression values into bins and return corresponding intervals.

    Converts track expression values to 1-based bin indices according to 'breaks',
    then returns the intervals with their corresponding bin index. Adjacent intervals
    with the same bin value are merged.

    The range of bins is determined by 'breaks' argument. For example:
    breaks=[x1, x2, x3, x4] represents three bins: (x1, x2], (x2, x3], (x3, x4].

    If 'include_lowest' is True, the lowest value is included in the first bin:
    [x1, x2], (x2, x3], (x3, x4].

    Parameters
    ----------
    expr : str
        Track expression to evaluate.
    breaks : array-like
        Break points that determine the bins. Must have at least 2 elements
        and be strictly increasing.
    intervals : DataFrame, optional
        Genomic scope for which the function is applied.
        If None, uses all genomic intervals.
    include_lowest : bool, default False
        If True, the lowest value of the range is included in the first bin.
    iterator : int or str, optional
        Track expression iterator. If None, determined implicitly.
    band : optional
        Track expression band (not yet supported).

    Returns
    -------
    DataFrame or None
        DataFrame with columns 'chrom', 'start', 'end', 'bin' where 'bin' is
        the 1-based bin index. Returns None if no values fall within the breaks.
        Adjacent intervals with the same bin are merged.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Partition dense_track values into 4 bins:

    >>> breaks = [0, 0.05, 0.1, 0.15, 0.2]
    >>> pm.gpartition("dense_track", breaks, pm.gintervals("1", 0, 5000))  # doctest: +SKIP

    See Also
    --------
    gdist, gsummary, gextract

    Notes
    -----
    Values outside the break range are excluded from the result.
    NaN values are also excluded.
    """
    from .intervals import gintervals_all
    _checkroot()

    if breaks is None:
        raise ValueError("gpartition requires breaks argument")

    # Convert breaks to list if needed
    breaks_list = list(breaks)
    if len(breaks_list) < 2:
        raise ValueError("gpartition requires at least 2 break values (for 1 bin)")

    if intervals is None:
        intervals = gintervals_all()

    result = _pymisha.pm_partition(
        expr,
        breaks_list,
        _df2pymisha(intervals),
        iterator,
        include_lowest,
        CONFIG
    )

    if result is None:
        return None

    return _pandas.DataFrame(result)


def gsample(expr, n, intervals=None, iterator=None):
    """
    Sample values from a track expression using reservoir sampling.

    Randomly samples *n* values from the track expression over the given
    intervals.  The sampling is performed in a single streaming pass using a
    reservoir sampler, so it is memory-efficient regardless of the number of
    genomic positions scanned.

    Parameters
    ----------
    expr : str
        Track expression.
    n : int
        Number of samples to draw.
    intervals : DataFrame, optional
        Genomic scope.  If ``None``, uses ``gintervals_all()``.
    iterator : int or str, optional
        Iterator policy for binning the intervals.

    Returns
    -------
    numpy.ndarray
        1-D array of sampled values (float64).  Length may be less than *n*
        if fewer non-NaN data points exist.

    Raises
    ------
    ValueError
        If *n* < 1.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> samples = pm.gsample("dense_track", 100)
    >>> len(samples)
    100
    >>> samples = pm.gsample("dense_track", 50,
    ...                      pm.gintervals(1, 0, 10000))
    >>> len(samples)
    50

    See Also
    --------
    gextract, gsummary, gquantiles
    """
    _checkroot()

    if intervals is None:
        intervals = gintervals_all()

    result = _pymisha.pm_sample(
        expr,
        int(n),
        _df2pymisha(intervals),
        iterator,
        CONFIG,
    )

    if result is None:
        return _numpy.array([], dtype=_numpy.float64)

    return result


def gcor(*exprs, intervals=None, iterator=None, method="pearson", details=False, names=None):
    """
    Compute correlation between pairs of track expressions.

    Calculates correlation in a single streaming pass over the data, making
    it memory-efficient for genome-wide computations.  Supports multitasking
    via chromosome partitioning when enabled.

    Parameters
    ----------
    *exprs : str
        An even number of track expressions.  Each consecutive pair (expr1,
        expr2) defines one correlation to compute.
    intervals : DataFrame, optional
        Genomic scope.  If ``None``, uses ``gintervals_all()``.
    iterator : int or str, optional
        Iterator policy.
    method : {"pearson", "spearman", "spearman.exact"}, default "pearson"
        Correlation method. ``"pearson"`` computes Pearson correlation in a
        streaming pass. ``"spearman"`` computes an approximate, memory-bounded
        Spearman correlation using reservoir sampling. ``"spearman.exact"``
        computes exact Spearman correlation with average-rank ties (requires
        O(n) memory in number of non-NaN bins).
    details : bool, default False
        If ``True``, return a DataFrame with full statistics (cor, cov,
        mean1, mean2, sd1, sd2, n, n.na) for Pearson, or (n, n.na, cor) for
        Spearman methods, instead of just correlation values.
    names : list of str, optional
        Names for each correlation pair.  If ``None``, names are generated
        as ``"expr1~expr2"``.

    Returns
    -------
    numpy.ndarray or DataFrame
        If ``details=False``: 1-D array of correlation values, one per pair.
        If ``details=True``: DataFrame with rows per pair and columns for
        all statistics.

    Raises
    ------
    ValueError
        If an odd number of expressions is provided.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gcor("dense_track", "sparse_track",
    ...         intervals=pm.gintervals(1, 0, 10000), iterator=1000)  # doctest: +SKIP
    array([...])
    >>> pm.gcor("dense_track", "sparse_track",
    ...         intervals=pm.gintervals(1, 0, 10000), iterator=1000,
    ...         details=True)  # doctest: +SKIP
              cor       cov     mean1  ...
    0  ...
    >>> pm.gcor("dense_track", "sparse_track",
    ...         intervals=pm.gintervals(1, 0, 10000), iterator=1000,
    ...         method="spearman")  # doctest: +SKIP
    array([...])
    >>> pm.gcor("dense_track", "sparse_track",
    ...         intervals=pm.gintervals(1, 0, 10000), iterator=1000,
    ...         method="spearman.exact", details=True)  # doctest: +SKIP
              n  n.na       cor
    0  ...

    See Also
    --------
    gsummary, gextract
    """
    _checkroot()

    expr_list = list(exprs)
    if len(expr_list) < 2:
        raise ValueError("gcor requires at least two track expressions")
    if len(expr_list) % 2 != 0:
        raise ValueError("gcor requires an even number of track expressions (pairs)")
    if method not in {"pearson", "spearman", "spearman.exact"}:
        raise ValueError("method must be one of: 'pearson', 'spearman', 'spearman.exact'")

    if intervals is None:
        intervals = gintervals_all()

    result = _pymisha.pm_cor(
        expr_list,
        _df2pymisha(intervals),
        iterator,
        method,
        CONFIG,
    )

    if result is None:
        return None

    num_pairs = len(expr_list) // 2

    # Generate pair names
    if names is None:
        names = [
            f"{expr_list[i * 2]}~{expr_list[i * 2 + 1]}"
            for i in range(num_pairs)
        ]

    if details:
        rows = []
        for _i, d in enumerate(result):
            if method == "pearson":
                rows.append({
                    "cor": d["cor"],
                    "cov": d["cov"],
                    "mean1": d["mean1"],
                    "mean2": d["mean2"],
                    "sd1": d["sd1"],
                    "sd2": d["sd2"],
                    "n": d["n"],
                    "n.na": d["n.na"],
                })
            else:
                rows.append({
                    "n": d["n"],
                    "n.na": d["n.na"],
                    "cor": d["cor"],
                })
        return _pandas.DataFrame(rows, index=names)

    return _numpy.array([d["cor"] for d in result])


# ---------------------------------------------------------------------------
# gbins_summary / gbins_quantiles
# ---------------------------------------------------------------------------

def _parse_bin_args(args):
    """Parse variadic (bin_expr, breaks, ...) pairs.

    Returns (bin_exprs, breaks_list) with validated breaks.
    """
    if len(args) < 2 or len(args) % 2 != 0:
        raise ValueError(
            "Binning arguments must be pairs of (expression, breaks)"
        )

    bin_exprs = []
    breaks_list = []
    for i in range(0, len(args), 2):
        expr = args[i]
        if not isinstance(expr, str):
            raise TypeError(f"Bin expression at position {i} must be a string")
        brk = _numpy.asarray(args[i + 1], dtype=float)
        if brk.ndim != 1:
            raise ValueError(f"Breaks at position {i + 1} must be 1D")
        if len(brk) < 2:
            raise ValueError(f"Breaks at position {i + 1} must have at least 2 elements")
        if not _numpy.all(_numpy.diff(brk) > 0):
            raise ValueError(f"Breaks at position {i + 1} must be strictly increasing")
        bin_exprs.append(expr)
        breaks_list.append(brk)
    return bin_exprs, breaks_list


def _assign_bins(values, breaks, include_lowest):
    """Assign values to bins defined by breaks.

    Bins are (breaks[i], breaks[i+1]] (right-closed).
    With include_lowest the first bin becomes [breaks[0], breaks[1]].

    Returns array of bin indices (0-based), -1 for values outside all bins.
    """
    return _bin_values(values, breaks, include_lowest)


def _compute_summary_stats(values):
    """Compute the 7 summary stats for an array of values.

    Returns [total, nan_count, min, max, sum, mean, stddev].
    """
    total = float(len(values))
    if total == 0:
        return _numpy.array([0.0, 0.0, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan])

    nan_mask = _numpy.isnan(values)
    nan_count = float(_numpy.count_nonzero(nan_mask))
    valid = values[~nan_mask]
    n_valid = len(valid)

    if n_valid == 0:
        return _numpy.array([total, nan_count, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan, _numpy.nan])

    min_val = float(_numpy.min(valid))
    max_val = float(_numpy.max(valid))
    sum_val = float(_numpy.sum(valid))
    mean_val = sum_val / n_valid
    std_val = float(_numpy.std(valid, ddof=1)) if n_valid > 1 else _numpy.nan

    return _numpy.array([total, nan_count, min_val, max_val, sum_val, mean_val, std_val])


def gbins_summary(*args, expr=None, intervals=None, include_lowest=False,
                  iterator=None, band=None, **kwargs):
    """Compute summary statistics per bin.

    Parameters
    ----------
    *args : pairs of (bin_expr, breaks)
        Alternating track expressions and bin break vectors.
    expr : str, optional
        Track expression to summarize.  If None the first bin expression is
        used.
    intervals : DataFrame, optional
        Genomic scope.  Defaults to all intervals.
    include_lowest : bool
        Include the left edge of the first bin.
    iterator : int or str, optional
        Track expression iterator.
    band : tuple of (float, float), optional
        Diagonal band ``(d1, d2)`` for 2D interval filtering.

    Returns
    -------
    ndarray
        Shape ``(*n_bins, 7)`` where the last axis holds:
        [Total intervals, NaN intervals, Min, Max, Sum, Mean, Std dev].

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gbins_summary("dense_track", [0, 0.2, 0.4, 2], expr="sparse_track",
    ...                  intervals=pm.gintervals(1), iterator="dense_track")  # doctest: +SKIP

    See Also
    --------
    gsummary, gintervals_summary, gdist
    """
    _checkroot()

    bin_exprs, breaks_list = _parse_bin_args(args)

    if expr is None:
        expr = bin_exprs[0]

    if intervals is None:
        intervals = gintervals_all()

    # Extract each expression individually (handles mixed track types)
    all_exprs = list(dict.fromkeys(list(bin_exprs) + [expr]))
    expr_cache = {}
    for e in all_exprs:
        if e not in expr_cache:
            expr_cache[e] = _extract_expr_values(
                e, intervals, iterator=iterator, band=band,
            )

    n_bins = [len(b) - 1 for b in breaks_list]
    if len(expr_cache[all_exprs[0]]) == 0:
        return _numpy.full(n_bins + [7], _numpy.nan)

    # Assign bin indices for each dimension
    bin_idx_arrays = []
    for be, brk in zip(bin_exprs, breaks_list, strict=False):
        idx = _assign_bins(expr_cache[be], brk, include_lowest)
        bin_idx_arrays.append(idx)

    expr_values = expr_cache[expr]

    # Build valid mask: must be in-range for all bin dimensions
    valid = _numpy.ones(len(expr_values), dtype=bool)
    for idx_arr in bin_idx_arrays:
        valid &= idx_arr >= 0

    # Flat multi-dimensional index
    result = _numpy.full(n_bins + [7], _numpy.nan)

    if not _numpy.any(valid):
        # Set total/nan counts to 0 for empty result
        result[..., 0] = 0.0
        result[..., 1] = 0.0
        return result

    # Default empty-bin stats: zero counts, NaN moments.
    result[..., 0] = 0.0
    result[..., 1] = 0.0

    valid_bin_idx = [idx[valid].astype(_numpy.int64, copy=False) for idx in bin_idx_arrays]
    flat_idx = _numpy.ravel_multi_index(valid_bin_idx, n_bins)
    vals = expr_values[valid]

    sort_order = _numpy.argsort(flat_idx, kind="mergesort")
    flat_sorted = flat_idx[sort_order]
    vals_sorted = vals[sort_order]

    unique_bins, starts = _numpy.unique(flat_sorted, return_index=True)
    ends = _numpy.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(flat_sorted)

    result_flat = result.reshape(-1, 7)
    for ub, s, e in zip(unique_bins, starts, ends, strict=False):
        result_flat[int(ub)] = _compute_summary_stats(vals_sorted[s:e])

    return result


def gbins_quantiles(*args, expr=None, percentiles=0.5, intervals=None,
                    include_lowest=False, iterator=None, band=None, **kwargs):
    """Compute quantiles per bin.

    Parameters
    ----------
    *args : pairs of (bin_expr, breaks)
        Alternating track expressions and bin break vectors.
    expr : str, optional
        Track expression to compute quantiles for.  If None the first bin
        expression is used.
    percentiles : float or array-like
        Percentile(s) in [0, 1].
    intervals : DataFrame, optional
        Genomic scope.  Defaults to all intervals.
    include_lowest : bool
        Include the left edge of the first bin.
    iterator : int or str, optional
        Track expression iterator.
    band : tuple of (float, float), optional
        Diagonal band ``(d1, d2)`` for 2D interval filtering.

    Returns
    -------
    ndarray
        Shape ``(*n_bins, n_percentiles)``.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gbins_quantiles("dense_track", [0, 0.2, 0.4, 2],
    ...                    expr="sparse_track", percentiles=[0.2, 0.5],
    ...                    intervals=pm.gintervals(1), iterator="dense_track")  # doctest: +SKIP

    See Also
    --------
    gquantiles, gintervals_quantiles, gdist
    """
    _checkroot()

    bin_exprs, breaks_list = _parse_bin_args(args)

    if expr is None:
        expr = bin_exprs[0]

    pct = _numpy.asarray(percentiles, dtype=float)
    if pct.ndim == 0:
        pct = pct.reshape(1)
    if _numpy.any((pct < 0) | (pct > 1)):
        raise ValueError("percentiles must be within [0, 1]")

    if intervals is None:
        intervals = gintervals_all()

    all_exprs = list(dict.fromkeys(list(bin_exprs) + [expr]))
    expr_cache = {}
    for e in all_exprs:
        if e not in expr_cache:
            expr_cache[e] = _extract_expr_values(
                e, intervals, iterator=iterator, band=band,
            )

    n_bins = [len(b) - 1 for b in breaks_list]
    if len(expr_cache[all_exprs[0]]) == 0:
        return _numpy.full(n_bins + [len(pct)], _numpy.nan)

    bin_idx_arrays = []
    for be, brk in zip(bin_exprs, breaks_list, strict=False):
        idx = _assign_bins(expr_cache[be], brk, include_lowest)
        bin_idx_arrays.append(idx)

    expr_values = expr_cache[expr]

    valid = _numpy.ones(len(expr_values), dtype=bool)
    for idx_arr in bin_idx_arrays:
        valid &= idx_arr >= 0

    result = _numpy.full(n_bins + [len(pct)], _numpy.nan)

    if not _numpy.any(valid):
        return result

    import itertools
    for bin_combo in itertools.product(*[range(n) for n in n_bins]):
        mask = valid.copy()
        for dim, bi in enumerate(bin_combo):
            mask &= bin_idx_arrays[dim] == bi
        bin_values = expr_values[mask]
        bin_values = bin_values[~_numpy.isnan(bin_values)]
        if len(bin_values) > 0:
            result[bin_combo] = _numpy.quantile(bin_values, pct)

    return result
