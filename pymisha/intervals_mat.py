"""gintervals_to_mat / gintervals_from_mat: pivot intervals + values to a
DataFrame indexed by intervals.

Mirrors R misha PR #120 (gintervals.to_mat / gintervals.from_mat).
"""

from __future__ import annotations

import pandas as pd

_COORD_COLS = ("chrom", "start", "end")


def gintervals_to_mat(
    df: pd.DataFrame,
    id_col: str | None = None,
    value_cols: list[str] | None = None,
    labels: bool = True,
) -> pd.DataFrame:
    """Pivot intervals + value columns into a DataFrame indexed by intervals.

    Parameters
    ----------
    df : DataFrame
        Must contain ``chrom``, ``start``, ``end`` columns. May contain
        an ``intervalID`` column (from :func:`gextract`); it is kept out of
        the value columns.
    id_col : str, optional
        Column name whose values become the leading row index level. If
        ``None`` (default), the index is a 3-level MultiIndex of
        ``(chrom, start, end)``.
    value_cols : list of str, optional
        Columns to keep as values. If ``None`` (default), auto-detect: all
        non-coord, non-``intervalID`` numeric columns. Non-numeric
        auto-detect columns raise ``TypeError``; pass *value_cols*
        explicitly to override.
    labels : bool, default True
        If False, return with a default :class:`pandas.RangeIndex` instead
        of building the MultiIndex. Useful for million-row pipelines that
        don't need row labels.

    Returns
    -------
    DataFrame
        Rows indexed by intervals (MultiIndex if ``labels=True``), columns
        are *value_cols*.

    Raises
    ------
    ValueError
        If required coord columns are missing or *id_col* is not in *df*.
    TypeError
        If any selected value column is non-numeric.

    See Also
    --------
    gintervals_from_mat : Inverse operation.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    missing = [c for c in _COORD_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"df is missing required interval column(s): {', '.join(missing)}"
        )

    has_interval_id = "intervalID" in df.columns
    identity_cols = list(_COORD_COLS) + (["intervalID"] if has_interval_id else [])

    if value_cols is None:
        value_cols = [c for c in df.columns if c not in identity_cols]
        non_numeric = [c for c in value_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(
                f"Non-numeric value column(s): {', '.join(non_numeric)}. "
                "Use value_cols= to select numeric columns explicitly."
            )
    else:
        missing_vals = [c for c in value_cols if c not in df.columns]
        if missing_vals:
            raise ValueError(
                f"value_cols not found in df: {', '.join(missing_vals)}"
            )
        non_numeric = [c for c in value_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(
                f"Non-numeric value column(s): {', '.join(non_numeric)}. "
                "value_cols must select numeric columns only."
            )

    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"id_col not found in df: {id_col}")

    values = df[value_cols].reset_index(drop=True).copy()

    if not labels:
        return values

    if id_col is None:
        index = pd.MultiIndex.from_arrays(
            [df["chrom"].values, df["start"].values, df["end"].values],
            names=("chrom", "start", "end"),
        )
    else:
        index = pd.MultiIndex.from_arrays(
            [df[id_col].values, df["chrom"].values, df["start"].values, df["end"].values],
            names=(id_col, "chrom", "start", "end"),
        )
    values.index = index
    return values


def gintervals_from_mat(mat: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`gintervals_to_mat`.

    Requires *mat* to have a :class:`pandas.MultiIndex` with at least
    ``(chrom, start, end)`` as the last three levels. Drops any leading
    id level. Returns a flat DataFrame with ``chrom``, ``start``, ``end``
    as columns plus all value columns.

    Parameters
    ----------
    mat : DataFrame
        Output of :func:`gintervals_to_mat` (with ``labels=True``).

    Returns
    -------
    DataFrame

    Raises
    ------
    ValueError
        If the index is not a MultiIndex with the expected coord levels.
    """
    idx = mat.index
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError(
            "gintervals_from_mat requires a MultiIndex with (chrom, start, end) levels; "
            "did you call gintervals_to_mat with labels=False?"
        )
    if tuple(idx.names[-3:]) != _COORD_COLS:
        raise ValueError(
            f"MultiIndex last three levels must be {_COORD_COLS}; "
            f"got {tuple(idx.names)}"
        )

    out = mat.reset_index()
    # Drop any leading id level: keep only chrom, start, end, *value_cols.
    leading = list(idx.names[:-3])
    if leading:
        out = out.drop(columns=leading)
    value_cols = list(mat.columns)
    out = out[["chrom", "start", "end", *value_cols]]
    out["start"] = pd.to_numeric(out["start"], errors="raise").astype(int)
    out["end"] = pd.to_numeric(out["end"], errors="raise").astype(int)
    out["chrom"] = out["chrom"].astype(str)
    return out.reset_index(drop=True)
