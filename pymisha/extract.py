"""gextract and gscreen implementations."""

import os

from . import _shared
from ._safe_eval import UnsafeExpressionError, compile_safe_expression
from ._shared import (
    CONFIG,
    _bound_colname,
    _checkroot,
    _chunk_slices,
    _df2pymisha,
    _iterated_intervals,
    _numpy,
    _pandas,
    _progress_context,
    _pymisha,
    _pymisha2df,
)
from .expr import _expr_safe_name, _parse_expr_vars
from .vtracks import _compute_vtrack_values


def _is_2d_intervals(intervals):
    """Check if intervals DataFrame has 2D columns."""
    return isinstance(intervals, _pandas.DataFrame) and "chrom1" in intervals.columns


def _find_2d_track_file(track_path, c1, c2):
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


def _validate_band(band):
    """Validate band parameter. Returns (d1, d2) tuple or None."""
    if band is None:
        return None
    if not hasattr(band, '__len__') or len(band) != 2:
        raise ValueError("band must be a sequence of length 2: (d1, d2)")
    d1, d2 = int(band[0]), int(band[1])
    if d1 >= d2:
        raise ValueError(f"band d1 ({d1}) must be less than d2 ({d2})")
    return (d1, d2)


def _obj_in_band(obj, is_points, band):
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


def _gextract_2d_single(track, col_name, intervals, band):
    """Extract a single 2D track over 2D intervals."""
    from ._quadtree import query_2d_track_objects
    from .tracks import gtrack_info

    track_path = _pymisha.pm_track_path(track)
    info = gtrack_info(track)
    is_points = info.get("type") == "points"

    rows = []
    for interval_idx, qrow in enumerate(intervals.itertuples(index=False)):
        c1 = str(qrow.chrom1)
        s1 = int(qrow.start1)
        e1 = int(qrow.end1)
        c2 = str(qrow.chrom2)
        s2 = int(qrow.start2)
        e2 = int(qrow.end2)

        filepath = _find_2d_track_file(track_path, c1, c2)
        if filepath is None:
            continue

        objs = query_2d_track_objects(filepath, s1, s2, e1, e2)
        for obj in objs:
            if band is not None and not _obj_in_band(obj, is_points, band):
                continue
            if is_points:
                ox, oy, val = obj
                rows.append((c1, ox, ox + 1, c2, oy, oy + 1, float(val), interval_idx))
            else:
                ox1, oy1, ox2, oy2, val = obj
                rows.append((c1, ox1, ox2, c2, oy1, oy2, float(val), interval_idx))

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
    return result.sort_values(
        ["chrom1", "start1", "chrom2", "start2", "intervalID"]
    ).reset_index(drop=True)


def _resolve_2d_vtrack_source(vtrack_name):
    """Resolve a 2D-capable virtual track to its backing 2D physical track."""
    from .tracks import gtrack_info

    cfg = _shared._VTRACKS.get(vtrack_name)
    if cfg is None:
        raise ValueError(f"Unknown virtual track '{vtrack_name}'")

    src = cfg.get("src")
    if not isinstance(src, str):
        raise ValueError(
            f"2D extraction for virtual track '{vtrack_name}' requires a physical 2D track source"
        )

    info = gtrack_info(src)
    if int(info.get("dimensions", 1) or 1) != 2:
        raise ValueError(
            f"Virtual track '{vtrack_name}' does not reference a 2D track source"
        )

    func = str(cfg.get("func", "avg")).lower()
    params = cfg.get("params")
    if func not in {"avg", "mean"} or params is not None:
        raise ValueError(
            f"2D extraction for virtual track '{vtrack_name}' supports only direct aliases"
        )
    if int(cfg.get("sshift", 0) or 0) != 0 or int(cfg.get("eshift", 0) or 0) != 0:
        raise ValueError(
            f"2D extraction for virtual track '{vtrack_name}' does not support iterator shifts"
        )
    if cfg.get("filter") is not None:
        raise ValueError(
            f"2D extraction for virtual track '{vtrack_name}' does not support filters"
        )

    return src


def _maybe_load_2d_intervals_set(intervals, exprs, iterator, band):
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


def _gextract_2d(exprs, intervals, iterator=None, colnames=None, band=None):
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
    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())

    parsed = []
    used_tracks = set()
    used_vtracks = set()
    for e in exprs:
        new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(
            e, track_names, vtrack_names
        )
        parsed.append((e, new_expr, expr_tracks, expr_vtracks))
        used_tracks.update(expr_tracks)
        used_vtracks.update(expr_vtracks)

    if not used_tracks and not used_vtracks:
        if len(exprs) == 1:
            raise ValueError(
                "Cannot implicitly determine iterator policy:\n"
                f"track expression \"{exprs[0]}\" does not contain any tracks."
            )
        raise ValueError(
            "Cannot implicitly determine iterator policy: "
            "track expressions do not contain any tracks."
        )

    for tname in used_tracks:
        info = gtrack_info(tname)
        if int(info.get("dimensions", 1) or 1) != 2:
            raise ValueError(
                f"Track '{tname}' is not a 2D track (type: {info.get('type')})"
            )

    vtrack_to_track = {}
    for vt_name in used_vtracks:
        vtrack_to_track[vt_name] = _resolve_2d_vtrack_source(vt_name)

    required_tracks = []

    def _add_required(track_name):
        if track_name not in required_tracks:
            required_tracks.append(track_name)

    for _orig_expr, _expr_eval, expr_tracks, expr_vtracks in parsed:
        for tname in expr_tracks:
            _add_required(tname)
        for vt_name in expr_vtracks:
            _add_required(vtrack_to_track[vt_name])

    if len(required_tracks) > 1 and iterator is None:
        raise ValueError(
            "Cannot implicitly determine iterator policy: "
            "track expressions contain more than one 2D track."
        )

    anchor_track = required_tracks[0]
    if isinstance(iterator, str):
        if iterator in required_tracks:
            anchor_track = iterator
        elif iterator in vtrack_to_track:
            anchor_track = vtrack_to_track[iterator]

    key_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"]
    coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]

    track_cols = {tname: _expr_safe_name(tname) for tname in required_tracks}
    result = _gextract_2d_single(anchor_track, track_cols[anchor_track], intervals, band)
    if result is None:
        return None

    for tname in required_tracks:
        if tname == anchor_track:
            continue
        cur = _gextract_2d_single(tname, track_cols[tname], intervals, band)
        if cur is None:
            result[track_cols[tname]] = _numpy.nan
            continue
        cur = cur[key_cols + [track_cols[tname]]]
        if cur.duplicated(key_cols).any():
            cur = cur.drop_duplicates(key_cols, keep="first")
        result = result.merge(cur, on=key_cols, how="left")

    out_cols = colnames if colnames is not None else exprs
    out_data = {}
    for out_col, (orig_expr, expr_eval, expr_tracks, expr_vtracks) in zip(
        out_cols, parsed, strict=False
    ):
        allowed_names = {
            "np",
            "numpy",
            *(_expr_safe_name(t) for t in expr_tracks),
            *(_expr_safe_name(vt) for vt in expr_vtracks),
        }
        try:
            code_obj = compile_safe_expression(expr_eval, allowed_names)
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc

        local_ns = {"np": _numpy, "numpy": _numpy}
        for tname in expr_tracks:
            local_ns[_expr_safe_name(tname)] = result[track_cols[tname]].to_numpy(
                dtype=float, copy=False
            )
        for vt_name in expr_vtracks:
            src_track = vtrack_to_track[vt_name]
            local_ns[_expr_safe_name(vt_name)] = result[track_cols[src_track]].to_numpy(
                dtype=float, copy=False
            )

        vals = eval(code_obj, {"__builtins__": {}}, local_ns)
        if _numpy.isscalar(vals):
            vals = _numpy.full(len(result), vals, dtype=float)
        out_data[out_col] = _numpy.asarray(vals, dtype=float)

    out_df = result[coord_cols].copy()
    for out_col in out_cols:
        out_df[out_col] = out_data[out_col]
    out_df["intervalID"] = result["intervalID"].to_numpy(dtype=int, copy=False)
    return out_df.sort_values(
        ["chrom1", "start1", "chrom2", "start2", "intervalID"]
    ).reset_index(drop=True)


def gextract(expr, intervals=None, iterator=None, colnames=None, band=None, **kwargs):
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
    **kwargs
        Additional keyword arguments:

        - **progress** (*bool or str, optional*) -- Whether to show a
          progress bar.
        - **progress_desc** (*str, optional*) -- Description for the
          progress bar (default ``'gextract'``).

    Returns
    -------
    DataFrame or None
        DataFrame with columns: chrom, start, end, <expr1>, ..., intervalID.
        Returns None if the iterator produces no intervals.

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
    exprs = [expr] if isinstance(expr, str) else list(expr)

    if intervals is None:
        from .intervals import gintervals_all

        intervals = gintervals_all()

    intervals = _maybe_load_2d_intervals_set(intervals, exprs, iterator, band)

    # Route to 2D extraction if intervals are 2D
    if _is_2d_intervals(intervals):
        if colnames is not None and len(colnames) != len(exprs):
            raise ValueError(
                f"colnames length ({len(colnames)}) must match number of "
                f"expressions ({len(exprs)})"
            )
        return _gextract_2d(
            exprs, intervals, iterator=iterator, colnames=colnames, band=band
        )

    if band is not None:
        raise ValueError("band parameter is only supported with 2D intervals")

    progress = kwargs.get('progress')
    progress_desc = kwargs.get('progress_desc', 'gextract')

    if colnames is not None and len(colnames) != len(exprs):
        raise ValueError(
            f"colnames length ({len(colnames)}) must match number of "
            f"expressions ({len(exprs)})"
        )

    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())

    parsed = []
    used_tracks = set()
    used_vtracks = set()
    for e in exprs:
        new_expr, expr_tracks, expr_vtracks, _ = _parse_expr_vars(e, track_names, vtrack_names)
        used_tracks.update(expr_tracks)
        used_vtracks.update(expr_vtracks)
        parsed.append((e, new_expr, expr_tracks, expr_vtracks))

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
            compile_safe_expression(expr_eval, allowed_names)
        except UnsafeExpressionError as exc:
            raise ValueError(f"Unsafe expression '{orig_expr}': {exc}") from exc

    if not used_vtracks:
        with _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_extract(
                exprs,
                _df2pymisha(intervals),
                iterator,
                CONFIG
            )
        df = _pymisha2df(result)
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
        track_exprs = list(used_tracks)
        base_result = _pymisha.pm_extract(
            track_exprs,
            _df2pymisha(intervals),
            iterator,
            CONFIG
        )
        base_df = _pymisha2df(base_result)
        if base_df is None:
            raise RuntimeError("Failed to extract physical track values for mixed expression")
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            if col not in base_df.columns:
                raise KeyError(f"Track column not found for '{tname}'")
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)

        iter_df = base_df[["chrom", "start", "end", "intervalID"]].copy()
    else:
        if iterator is None:
            if len(exprs) == 1:
                raise ValueError(
                    f"Cannot implicitly determine iterator policy:\n"
                    f"track expression \"{exprs[0]}\" does not contain any tracks."
                )
            raise ValueError(
                "Cannot implicitly determine iterator policy: "
                "track expressions do not contain any tracks."
            )
        iter_df = _iterated_intervals(intervals, iterator)

    if iter_df is None or len(iter_df) == 0:
        return None

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)
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
        try:
            code_obj = compile_safe_expression(expr_eval, allowed_names)
        except UnsafeExpressionError as exc:
            raise ValueError(
                f"Unsafe expression '{orig_expr}': {exc}"
            ) from exc
        compiled.append((colname, code_obj, expr_tracks, expr_vtracks))
        result_cols.append(colname)

    result_arrays = {col: _numpy.empty(n_rows, dtype=float) for col in result_cols}

    chrom_vals = iter_df["chrom"].values
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    with _progress_context(progress, total=n_rows, desc=progress_desc) as progress_cb:
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

            vtrack_arrays = {}
            if used_vtracks:
                chunk_intervals = iter_df.iloc[start_idx:end_idx][["chrom", "start", "end"]]
                for vt in used_vtracks:
                    vtrack_arrays[vt] = _compute_vtrack_values(vt, chunk_intervals)
                    local_ns[_expr_safe_name(vt)] = vtrack_arrays[vt]

            for colname, code_obj, _expr_tracks, _expr_vtracks in compiled:
                result_values = eval(code_obj, {'__builtins__': {}}, local_ns)
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

    result_df = _pandas.DataFrame({
        'chrom': chrom_vals,
        'start': start_vals,
        'end': end_vals,
    })
    for col in result_cols:
        result_df[col] = result_arrays[col]
    result_df['intervalID'] = iter_df['intervalID'].to_numpy(dtype=int, copy=False)
    return result_df


def gscreen(expr, intervals=None, **kwargs):
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
    if intervals is None:
        from .intervals import gintervals_all

        intervals = gintervals_all()

    progress = kwargs.get('progress')
    progress_desc = kwargs.get('progress_desc', 'gscreen')
    iterator = kwargs.get("iterator")
    band = kwargs.get("band")

    intervals = _maybe_load_2d_intervals_set(intervals, [expr], iterator, band)
    if _is_2d_intervals(intervals):
        extracted = gextract(
            expr,
            intervals=intervals,
            iterator=iterator,
            band=band,
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
        return (
            extracted.loc[mask, ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]]
            .reset_index(drop=True)
        )

    track_names = set(_pymisha.pm_track_names())
    vtrack_names = set(_shared._VTRACKS.keys())
    expr_eval, expr_tracks, expr_vtracks, _ = _parse_expr_vars(expr, track_names, vtrack_names)

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
        compile_safe_expression(expr_eval, allowed_names)
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsafe expression '{expr}': {exc}") from exc

    if not expr_vtracks:
        with _progress_context(progress, desc=progress_desc):
            result = _pymisha.pm_screen(
                expr,
                _df2pymisha(intervals),
                kwargs.get('iterator'),
                CONFIG
            )
        return _pymisha2df(result)

    track_arrays = {}
    base_df = None
    iter_df = None

    if expr_tracks:
        track_exprs = list(expr_tracks)
        base_result = _pymisha.pm_extract(
            track_exprs,
            _df2pymisha(intervals),
            kwargs.get('iterator'),
            CONFIG
        )
        base_df = _pymisha2df(base_result)
        if base_df is None:
            raise RuntimeError("Failed to extract physical track values for mixed expression")
        for tname in track_exprs:
            col = tname if tname in base_df.columns else _bound_colname(tname, 40)
            if col not in base_df.columns:
                raise KeyError(f"Track column not found for '{tname}'")
            track_arrays[tname] = base_df[col].to_numpy(dtype=float, copy=False)
        iter_df = base_df[["chrom", "start", "end", "intervalID"]].copy()
    else:
        if kwargs.get('iterator') is None:
            raise ValueError(
                f"Cannot implicitly determine iterator policy:\n"
                f"track expression \"{expr}\" does not contain any tracks."
            )
        iter_df = _iterated_intervals(intervals, kwargs.get('iterator'))

    if iter_df is None or len(iter_df) == 0:
        return None

    n_rows = len(iter_df)
    chunk_size = int(CONFIG.get("eval_buf_size", 1000) or 1000)
    mask = _numpy.zeros(n_rows, dtype=bool)

    chrom_vals = iter_df["chrom"].values
    start_vals = iter_df["start"].to_numpy(dtype=int, copy=False)
    end_vals = iter_df["end"].to_numpy(dtype=int, copy=False)

    code_obj = compile_safe_expression(expr_eval, allowed_names)

    with _progress_context(progress, total=n_rows, desc=progress_desc) as progress_cb:
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

            vtrack_arrays = {}
            if expr_vtracks:
                chunk_intervals = iter_df.iloc[start_idx:end_idx][["chrom", "start", "end"]]
                for vt in expr_vtracks:
                    vtrack_arrays[vt] = _compute_vtrack_values(vt, chunk_intervals)
                    local_ns[_expr_safe_name(vt)] = vtrack_arrays[vt]

            chunk_mask = eval(code_obj, {'__builtins__': {}}, local_ns)
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
    return filtered.reset_index(drop=True)
