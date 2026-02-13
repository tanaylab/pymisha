"""Lookup and bin transform functions."""

from . import _shared
from ._shared import (
    CONFIG,
    _checkroot,
    _df2pymisha,
    _numpy,
    _pymisha,
    _pymisha2df,
)
from .expr import _find_vtracks_in_expr, _parse_expr_vars
from .extract import _is_2d_intervals, gextract
from .intervals import gintervals_2d_all, gintervals_all


def glookup(lookup_table, *args, intervals=None, include_lowest=False,
            force_binning=True, iterator=None, band=None, **kwargs):
    """
    Look up values in an N-dimensional lookup table indexed by track expressions.

    For each iterator interval, evaluates one or more track expressions and
    uses the resulting values to index into a lookup table. Returns the table
    value for each interval.

    Uses a memory-efficient C++ streaming implementation when expressions
    do not contain virtual tracks. Falls back to Python (memory-resident)
    when virtual tracks, a band filter, or 2D intervals are present.

    Parameters
    ----------
    lookup_table : numpy.ndarray
        N-dimensional lookup table. The shape must match the number of bins
        in each dimension. For 1D lookup, shape is ``(n_bins,)``. For
        multi-dimensional lookup, shape is ``(n_bins_1, n_bins_2, ...)``.
    *args : pairs of (str, array-like)
        Alternating track expressions and break arrays defining bins.
        Same format as ``gdist``.
        Example: ``glookup(table, "track1", breaks1, "track2", breaks2, ...)``.
    intervals : DataFrame or str
        Genomic scope for which the function is applied. Required.
    include_lowest : bool, default False
        If True, the lowest break value is included in the first bin.
        Example: ``breaks=[0, 0.2, 0.5]`` creates ``(0, 0.2], (0.2, 0.5]``.
        With ``include_lowest=True``: ``[0, 0.2], (0.2, 0.5]``.
    force_binning : bool, default True
        If True, clamp out-of-range values to the nearest bin instead of
        NaN. If False, out-of-range values produce NaN.
    iterator : int or str, optional
        Track expression iterator. If None, determined implicitly.
    band : tuple of (int, int), optional
        Diagonal band for 2D tracks. Triggers Python fallback path.

    Returns
    -------
    DataFrame or None
        Intervals with columns: ``chrom``, ``start``, ``end``, ``value``,
        ``intervalID``. Returns None if *intervals* is empty.

    See Also
    --------
    gdist : Compute distribution over binned track expressions.
    gtrack_lookup : Create a track from an N-dimensional lookup table.

    Examples
    --------
    >>> import pymisha as pm
    >>> import numpy as np
    >>> _ = pm.gdb_init_examples()

    One-dimensional lookup:

    >>> breaks = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    >>> result = pm.glookup([10, 20, 30, 40, 50], "dense_track", breaks,
    ...                     intervals=pm.gintervals("1", 0, 500))
    >>> "value" in result.columns
    True
    """
    if lookup_table is None:
        raise ValueError("lookup_table cannot be None")

    if intervals is None:
        raise ValueError("intervals cannot be None")

    # Parse arguments: (expr1, breaks1, expr2, breaks2, ...)
    if len(args) < 2:
        raise ValueError(
            "Usage: glookup(lookup_table, [expr, breaks]+, intervals=intervals, ...)"
        )

    if len(args) % 2 != 0:
        raise ValueError("glookup requires pairs of (expression, breaks) arguments")

    _checkroot()

    if len(intervals) == 0:
        return None

    lookup_table = _numpy.asarray(lookup_table, dtype=float)
    progress = kwargs.get("progress")
    progress_desc = kwargs.get("progress_desc", "glookup")

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
        breaks_list.append(breaks)

    # Validate lookup_table dimensions match number of expr-breaks pairs
    expected_dims = len(exprs)
    if lookup_table.ndim != expected_dims:
        raise ValueError(
            f"lookup_table has {lookup_table.ndim} dimensions but "
            f"{expected_dims} expr-breaks pairs were provided"
        )

    # Validate lookup_table shape matches number of bins
    n_bins = [len(b) - 1 for b in breaks_list]
    for i, (n, expected) in enumerate(zip(lookup_table.shape, n_bins, strict=False)):
        if n != expected:
            raise ValueError(
                f"lookup_table dimension {i} has size {n} but {expected} bins expected"
            )

    # Check if we need the Python path (vtracks, band, or 2D intervals)
    has_vtracks = any(_find_vtracks_in_expr(e) for e in exprs)
    use_python = has_vtracks or band is not None or _is_2d_intervals(intervals)

    if not use_python:
        # Use C++ streaming implementation
        config = dict(CONFIG)
        if progress is not None:
            config['progress'] = progress
        if progress_desc:
            config['progress_desc'] = progress_desc

        # Flatten lookup table in Fortran (column-major) order to match
        # BinsManager::vals2idx flat index convention
        flat_table = _numpy.ascontiguousarray(lookup_table.ravel(order='F'))

        result = _pymisha.pm_lookup(
            exprs,
            [b.tolist() for b in breaks_list],
            flat_table,
            _df2pymisha(intervals),
            iterator,
            include_lowest,
            force_binning,
            config,
        )

        if result is None:
            return None

        return _pymisha2df(result)

    # Fall back to Python implementation for virtual tracks / band / 2D
    return _glookup_python(
        lookup_table, exprs, breaks_list, intervals,
        include_lowest=include_lowest, force_binning=force_binning,
        iterator=iterator, band=band,
        progress=progress, progress_desc=progress_desc,
    )


def _glookup_python(lookup_table, exprs, breaks_list, intervals,
                    include_lowest=False, force_binning=True,
                    iterator=None, band=None,
                    progress=None, progress_desc=None):
    """Pure-Python fallback for glookup (used when vtracks, band, or 2D)."""
    all_values = []
    extract_result = None
    coord_cols = None

    for i, expr in enumerate(exprs):
        result = gextract(expr, intervals, iterator=iterator, band=band,
                         progress=progress, progress_desc=progress_desc)
        if result is None or len(result) == 0:
            return None

        coord_cols = {
            "chrom", "start", "end",
            "chrom1", "start1", "end1", "chrom2", "start2", "end2",
            "intervalID",
        }
        data_cols = [c for c in result.columns if c not in coord_cols]
        if not data_cols:
            return None

        values = result[data_cols[0]].to_numpy(dtype=float, copy=False)
        all_values.append(values)

        if i == 0:
            extract_result = result
            if {"chrom", "start", "end", "intervalID"}.issubset(result.columns):
                coord_cols = ["chrom", "start", "end", "intervalID"]
            elif {
                "chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"
            }.issubset(result.columns):
                coord_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "intervalID"]
            else:
                raise ValueError("Unsupported extract result schema returned by gextract")
        else:
            if len(result) != len(extract_result):
                raise ValueError(
                    "glookup expression extraction produced mismatched row counts; "
                    "cannot align multi-expression lookup safely"
                )
            for c in coord_cols:
                if not _numpy.array_equal(
                    result[c].to_numpy(copy=False),
                    extract_result[c].to_numpy(copy=False),
                ):
                    raise ValueError(
                        f"glookup expression extraction produced misaligned rows (column '{c}')"
                    )

    n_values = len(all_values[0])

    # Bin each expression's values
    bin_indices = []
    for values, breaks in zip(all_values, breaks_list, strict=False):
        n_bin = len(breaks) - 1

        indices = _numpy.searchsorted(breaks, values, side='right') - 1

        if include_lowest:
            at_lowest = values == breaks[0]
            indices[at_lowest] = 0

        at_max = values == breaks[-1]
        indices[at_max] = n_bin - 1

        below_min = values < breaks[0]
        if not include_lowest:
            at_min = values == breaks[0]
            below_min = below_min | at_min

        above_max = values > breaks[-1]

        if force_binning:
            indices[below_min] = 0
            indices[above_max] = n_bin - 1
        else:
            indices[below_min] = -1
            indices[above_max] = -1

        nan_mask = _numpy.isnan(values)
        indices[nan_mask] = -1

        bin_indices.append(indices)

    # Look up values from table
    output_values = _numpy.full(n_values, _numpy.nan, dtype=float)

    valid = _numpy.ones(n_values, dtype=bool)
    for indices in bin_indices:
        valid &= (indices >= 0)

    if valid.any():
        if len(exprs) == 1:
            valid_indices = bin_indices[0][valid]
            output_values[valid] = lookup_table[valid_indices]
        else:
            valid_idx = [idx[valid] for idx in bin_indices]
            flat = _numpy.ravel_multi_index(valid_idx, lookup_table.shape)
            output_values[valid] = lookup_table.reshape(-1)[flat]

    result = extract_result[coord_cols].copy()
    result["value"] = output_values

    return result


def _resolve_lookup_dimensions(exprs, iterator):
    """Infer whether lookup output should be 1D or 2D from used physical tracks."""
    track_names = set(_pymisha.pm_track_names())
    if not track_names:
        return set()

    used_tracks = set()
    vtrack_names = _shared._VTRACKS

    for expr in exprs:
        _, parsed_tracks, _, _ = _parse_expr_vars(expr, track_names, vtrack_names)
        used_tracks.update(parsed_tracks)

    if isinstance(iterator, str) and iterator in track_names:
        used_tracks.add(iterator)

    if not used_tracks:
        return set()

    from .tracks import gtrack_info

    dims = set()
    for track in used_tracks:
        dims.add(int(gtrack_info(track).get("dimensions", 1)))
    return dims


def gtrack_lookup(track, description, lookup_table, *args, iterator=None,
                  include_lowest=False, force_binning=True, band=None):
    """
    Create a track from an N-dimensional lookup table.

    Evaluates track expressions genome-wide, looks up values in the table,
    and creates a new track from the results. Dense or sparse output is
    determined by the iterator type.

    This is the track-creation counterpart of :func:`glookup`, which returns
    values in-memory without creating a track.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Track description.
    lookup_table : numpy.ndarray
        N-dimensional lookup table. Shape must match the number of bins
        defined by each ``(expr, breaks)`` pair. For 1D: ``(n_bins,)``.
        For multi-dimensional: ``(n_bins_1, n_bins_2, ...)``.
    *args : pairs of (str, array-like)
        Alternating track expressions and break arrays defining bins.
    iterator : int or str, optional
        Track expression iterator. Integer values create dense tracks
        with that bin size. Intervals-based iterators create sparse tracks.
    include_lowest : bool, default False
        If True, the lowest break value is included in the first bin.
    force_binning : bool, default True
        If True, clamp out-of-range values to the nearest bin. If False,
        out-of-range values produce NaN in the track.
    band : tuple of (int, int), optional
        Diagonal band for 2D tracks. Passed through to :func:`glookup`.

    Returns
    -------
    None

    See Also
    --------
    glookup : Look up values without creating a track.
    gtrack_create : Create a track from an expression.
    gdist : Compute distribution over binned track expressions.

    Examples
    --------
    >>> import pymisha as pm
    >>> import numpy as np
    >>> _ = pm.gdb_init_examples()

    Create a dense track from 1D lookup:

    >>> pm.gtrack_lookup("my_track", "lookup track",  # doctest: +SKIP
    ...     np.array([10.0, 20.0, 30.0, 40.0]),
    ...     "dense_track", [0, 0.05, 0.1, 0.15, 0.2],
    ...     iterator=100)
    """
    if lookup_table is None:
        raise ValueError("lookup_table cannot be None")

    # Parse arguments: (expr1, breaks1, expr2, breaks2, ...)
    if len(args) < 2:
        raise ValueError(
            "Usage: gtrack_lookup(track, description, lookup_table, "
            "[expr, breaks]+, iterator=..., ...)"
        )

    if len(args) % 2 != 0:
        raise ValueError("gtrack_lookup requires pairs of (expression, breaks) arguments")

    _checkroot()

    lookup_table = _numpy.asarray(lookup_table, dtype=float)

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
        if not _numpy.all(_numpy.diff(breaks) > 0):
            raise ValueError(f"Breaks at position {i+1} must be strictly increasing")

        exprs.append(expr)
        breaks_list.append(breaks)

    # Validate lookup_table dimensions
    expected_dims = len(exprs)
    if lookup_table.ndim != expected_dims and not (lookup_table.ndim == 1 and expected_dims == 1):
        raise ValueError(
            f"lookup_table has {lookup_table.ndim} dimensions but "
                f"{expected_dims} expr-breaks pairs were provided"
            )

    n_bins = [len(b) - 1 for b in breaks_list]
    for i, (n, expected) in enumerate(zip(lookup_table.shape, n_bins, strict=False)):
        if n != expected:
            raise ValueError(
                f"lookup_table dimension {i} has size {n} but {expected} bins expected"
            )

    # Lazy imports to avoid circular dependency
    import shutil

    from .tracks import (
        _ensure_track_absent,
        _track_dir_for_create,
        _validate_track_name,
        gtrack_create_dense,
        gtrack_create_sparse,
    )

    _validate_track_name(track)
    _ensure_track_absent(track)

    dims = _resolve_lookup_dimensions(exprs, iterator)
    if 1 in dims and 2 in dims:
        raise ValueError("Cannot mix 1D and 2D tracks in gtrack_lookup expressions/iterator")

    use_2d_scope = (band is not None) or (2 in dims)
    scope_intervals = gintervals_2d_all() if use_2d_scope else gintervals_all()

    # Compute lookup values using glookup over the whole genome
    result = glookup(
        lookup_table, *args,
        intervals=scope_intervals,
        include_lowest=include_lowest,
        force_binning=force_binning,
        iterator=iterator,
        band=band,
    )

    if result is None or len(result) == 0:
        raise ValueError("Lookup produced no values; cannot create track")

    is_2d_result = {
        "chrom1", "start1", "end1", "chrom2", "start2", "end2"
    }.issubset(result.columns)
    if is_2d_result:
        intervals_df = result[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
    else:
        intervals_df = result[["chrom", "start", "end"]].copy()
    values = result["value"].to_numpy(dtype=float)

    # Determine track type from iterator for 1D outputs.
    # If iterator is a positive integer, create dense track; otherwise sparse.
    is_dense = (
        (not is_2d_result)
        and isinstance(iterator, (int, float, _numpy.integer, _numpy.floating))
        and int(iterator) > 0
    )

    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()
    try:
        if is_2d_result:
            from .tracks import gtrack_2d_create
            gtrack_2d_create(track, description, intervals_df, values)
        elif is_dense:
            binsize = int(iterator)
            gtrack_create_dense(track, description, intervals_df, values,
                                binsize, defval=_numpy.nan)
        else:
            gtrack_create_sparse(track, description, intervals_df, values)

        # Build created.by string matching R convention
        exprs_str = ", ".join(
            f'"{e}", c({", ".join(str(v) for v in b)})'
            for e, b in zip(exprs, breaks_list, strict=False)
        )
        created_by = (
            f'gtrack.lookup("{track}", description, lookup_table, '
            f'{exprs_str}, iterator={iterator!r})'
        )
        # Bypass readonly check for internal track creation
        from .tracks import _load_track_attributes, _save_track_attributes
        attrs = _load_track_attributes(track)
        attrs["created.by"] = created_by
        _save_track_attributes(track, attrs)

    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
            _pymisha.pm_dbreload()
        raise
