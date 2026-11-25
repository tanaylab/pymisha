"""Fixed-width windows as arrays for machine learning (gseq_onehot, gextract_windows)."""

from __future__ import annotations

import math
import numbers
from typing import Any, NamedTuple

import numpy as _numpy
import pandas as _pandas

from ._shared import _check_strand_column, _checkroot, _pymisha, _str2strand_values
from .expr import _caller_namespace
from .extract import _gextract_matrix, _maybe_load_intervals_set

_CHANNELS = ("A", "C", "G", "T")
_ONEHOT_DTYPES = frozenset(_numpy.dtype(t) for t in (bool, "u1", "i1", "i2", "i4", "i8", "f4", "f8"))


def _onehot_dtype(dtype: Any) -> _numpy.dtype:
    try:
        dt = _numpy.dtype(dtype)
    except TypeError as e:
        raise ValueError(f"Unsupported dtype {dtype!r}") from e
    if dt not in _ONEHOT_DTYPES:
        raise ValueError(f"Unsupported dtype {dt}: use bool, uint8, int8, int16, int32, int64, float32 or float64")
    # the canonical type number for its size (e.g. int64 rather than longlong), as the C++ side
    # dispatches on it
    canonical: _numpy.dtype = _numpy.dtype(dt.str)
    return canonical


def _minus_strand(intervals: _pandas.DataFrame) -> _numpy.ndarray:
    """True for the rows with strand -1 or "-". The strand column is read and checked as the
    other interval functions read it: labels "+", "-", ".", "*", "", or numbers -1, 0, 1, in any
    dtype (categorical and object columns included); anything else is an error."""
    if "strand" not in intervals.columns:
        return _numpy.zeros(len(intervals), dtype=bool)
    col = intervals["strand"]
    _check_strand_column(col)
    values = _str2strand_values(col)
    if values is None:
        values = col.to_numpy(dtype=_numpy.float64)
    return _numpy.asarray(values == -1, dtype=bool)


def _whole_coords(intervals: _pandas.DataFrame) -> tuple[_numpy.ndarray, _numpy.ndarray]:
    starts = _pandas.to_numeric(intervals["start"], errors="coerce").to_numpy(dtype=_numpy.float64)
    ends = _pandas.to_numeric(intervals["end"], errors="coerce").to_numpy(dtype=_numpy.float64)
    if not (_numpy.isfinite(starts).all() and _numpy.isfinite(ends).all()):
        raise ValueError("Interval coordinates must be finite numbers")
    if (starts != _numpy.round(starts)).any() or (ends != _numpy.round(ends)).any():
        raise ValueError("Interval coordinates must be whole numbers")
    return starts.astype(_numpy.int64), ends.astype(_numpy.int64)


def _chrom_codes(chroms: _pandas.Series) -> tuple[list[str], _numpy.ndarray]:
    """The canonical names of the distinct chromosomes and each row's index into them."""
    from .intervals import _normalize_chroms

    if isinstance(chroms.dtype, _pandas.CategoricalDtype):
        chroms = chroms.astype(object)
    uniq, codes = _numpy.unique(chroms.astype(str).to_numpy(), return_inverse=True)
    names = [str(c) for c in _normalize_chroms(list(uniq))]
    return names, codes.astype(_numpy.int32).reshape(-1)


def _chrom_sizes(names: list[str]) -> _numpy.ndarray:
    from .intervals import gintervals_all

    allg = gintervals_all()
    size_of = dict(zip(allg["chrom"].astype(str), allg["end"].to_numpy(dtype=_numpy.int64), strict=True))
    return _numpy.array([size_of[c] for c in names], dtype=_numpy.int64)


def _load_1d(iv: Any, what: str = "intervals") -> _pandas.DataFrame:
    iv = _maybe_load_intervals_set(iv)
    if not isinstance(iv, _pandas.DataFrame) or list(iv.columns[:3]) != ["chrom", "start", "end"]:
        raise ValueError(f"{what} must be 1D intervals")
    return iv


def gseq_onehot(
    x: _pandas.DataFrame | str | list[str],
    pad: bool = False,
    dtype: Any = _numpy.uint8,
) -> _numpy.ndarray:
    """One-hot encode genomic intervals or DNA strings.

    Returns the one-hot encoding of intervals of equal width, read directly
    from the genome sequence files in C++ (no Python strings are made), or of
    strings of equal length.

    The result has shape ``(N, 4, L)``, i.e. ``(batch, channels, length)``:
    ``x[n, c, p]`` is 1 when position ``p`` of sequence ``n`` is base ``c``.
    The channels are A, C, G and T, in that order. Lowercase (soft-masked)
    bases are encoded like uppercase ones. Any other character (N, IUPAC
    ambiguity codes, gaps) and, with ``pad=True``, every position outside the
    chromosome has all four channels 0. The array is C-contiguous, so
    ``torch.from_numpy`` takes it without a copy.

    For intervals with a ``strand`` column, a row with strand -1 (or "-") is
    reverse complemented, as in :func:`gseq_extract`.

    Element ``[n, c, p]`` equals element ``[n, c, p]`` of R misha's
    ``gseq.onehot`` (whose array has dimensions ``c(N, 4, L)``).

    The array takes ``N * 4 * L * dtype.itemsize`` bytes.

    Parameters
    ----------
    x : DataFrame, str or list of str
        Intervals of equal width (a DataFrame or the name of an intervals
        set), or a list of sequences of equal length. A bare string is taken
        as the name of an intervals set; pass ``[seq]`` for a single sequence.
    pad : bool, default False
        For intervals: if True, an interval may run past either end of its
        chromosome and the positions outside it are all 0, as the "N" of
        ``gseq_extract(pad=True)``. If False, such an interval is an error.
    dtype : numpy dtype, default uint8
        One of bool, uint8, int8, int16, int32, int64, float32, float64.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 4, L)``.

    See Also
    --------
    gseq_extract, gextract_windows

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> x = pm.gseq_onehot(pm.gintervals(["1", "1"], [10000, 20000], [10010, 20010]))
    >>> x.shape
    (2, 4, 10)
    >>> pm.gseq_onehot(["ACGTN"])[0]
    array([[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0]], dtype=uint8)
    """
    if x is None:
        raise ValueError("Usage: gseq_onehot(x, pad=False, dtype=numpy.uint8)")
    if not isinstance(pad, (bool, _numpy.bool_)):
        raise ValueError("pad must be True or False")
    dt = _onehot_dtype(dtype)

    if isinstance(x, (list, tuple, _numpy.ndarray, _pandas.Series)):
        seqs = list(x)
        width = len(seqs[0]) if seqs else 0
        out = _numpy.empty((len(seqs), 4, width), dtype=dt)
        if seqs:
            _pymisha.pm_seq_onehot_strings(seqs, False, out)
        return out

    _checkroot()
    iv = _load_1d(x, "x")
    if not len(iv):
        raise ValueError("x is empty")
    starts, ends = _whole_coords(iv)
    width = int(ends[0] - starts[0])
    if width <= 0 or ((ends - starts) != width).any():
        raise ValueError("All intervals must have the same positive width")
    names, codes = _chrom_codes(iv["chrom"])
    out = _numpy.empty((len(iv), 4, width), dtype=dt)  # every element is written
    _pymisha.pm_seq_onehot(names, codes, starts, _minus_strand(iv).astype(_numpy.uint8), out, bool(pad))
    return out


class ExtractedWindows(NamedTuple):
    """The result of :func:`gextract_windows`."""

    x: _numpy.ndarray | None
    """One-hot sequences, shape ``(N, 4, width + 2 * jitter)``, or None."""
    y: _numpy.ndarray | None
    """Signals, shape ``(N, T, (signal_width + 2 * jitter) // binsize)``, or None."""
    intervals: _pandas.DataFrame
    """The kept rows, one per window, in their original order."""
    dropped: _pandas.DataFrame
    """The dropped rows: ``index`` and ``reason``."""


def _whole(v: Any, name: str, minimum: int) -> int:
    """v as an int: a whole real number (not a bool) of at least minimum."""
    f = float(v) if isinstance(v, numbers.Real) and not isinstance(v, (bool, _numpy.bool_)) else float("nan")
    if not math.isfinite(f) or f != int(f) or f < minimum:
        raise ValueError(f"{name} must be a whole number of at least {minimum}")
    return int(f)


def _windows_overlap(
    names: list[str], codes: _numpy.ndarray, lo: _numpy.ndarray, hi: _numpy.ndarray, exclude: _pandas.DataFrame
) -> _numpy.ndarray:
    """True for the windows [lo, hi) that overlap an interval of exclude by at least one base."""
    from .intervals import _normalize_chroms

    hit = _numpy.zeros(len(codes), dtype=bool)
    if not len(exclude):
        return hit
    ex_starts, ex_ends = _whole_coords(exclude)
    keep = ex_ends > ex_starts
    ex_chroms = exclude["chrom"]
    if isinstance(ex_chroms.dtype, _pandas.CategoricalDtype):
        ex_chroms = ex_chroms.astype(object)
    ex_chroms = ex_chroms.astype(str).to_numpy()
    # chromosomes the genome does not have are ignored
    canon: dict[str, str | None] = {}
    for c in _numpy.unique(ex_chroms):
        try:
            canon[c] = str(_normalize_chroms([str(c)])[0])
        except Exception:  # noqa: BLE001 - an unknown chromosome
            canon[c] = None
    ex_canon = _numpy.array([canon[c] for c in ex_chroms], dtype=object)

    for k, name in enumerate(names):
        sel = keep & (ex_canon == name)
        if not sel.any():
            continue
        order = _numpy.argsort(ex_starts[sel], kind="stable")
        e_start = ex_starts[sel][order]
        max_end = _numpy.maximum.accumulate(ex_ends[sel][order])
        w = _numpy.flatnonzero(codes == k)
        # the exclusion intervals that start before each window's end; one of them overlaps the
        # window iff the furthest end among them is past the window's start
        j = _numpy.searchsorted(e_start, hi[w], side="left")
        hit[w] = (j > 0) & (max_end[_numpy.maximum(j, 1) - 1] > lo[w])
    return hit


def gextract_windows(
    intervals: _pandas.DataFrame | str | list | dict,
    tracks: str | list[str] | dict[str, str] | None = None,
    width: int | None = None,
    signal_width: int | None = None,
    jitter: int = 0,
    binsize: int = 1,
    use_strand: bool = True,
    exclude: _pandas.DataFrame | str | None = None,
    min_counts: float | None = None,
    max_counts: float | None = None,
    count_track: int | str = 0,
    sequence: bool = True,
    dtype: Any = _numpy.uint8,
    signal_dtype: Any = _numpy.float32,
    vars: dict[str, Any] | None = None,
) -> ExtractedWindows:
    """Extract fixed-width windows as arrays for machine learning.

    Extracts fixed-width windows around a set of loci and returns their DNA
    as a one-hot array and the values of track expressions over them as a
    numeric array, ready to feed a sequence model. One call replaces
    :func:`gseq_extract`, a one-hot encoding of the strings and
    :func:`gextract_matrix`; the one-hot array is encoded in C++ straight
    from the sequence files. Both arrays are C-contiguous with the
    ``(batch, channels, length)`` layout, so ``torch.from_numpy`` takes them
    without a copy.

    Each locus (a row of *intervals*) is centred at
    ``mid = start + (end - start) // 2``. Its sequence window is
    ``[mid - width // 2 - jitter, mid - width // 2 + width + jitter)`` and
    its signal window is the same with *signal_width*: an odd width reaches
    one base further right than left. A positive *jitter* widens both windows
    by *jitter* on each side, so that a data loader can crop randomly shifted
    windows of the requested widths later without reading the genome again
    (as in tangermeme's ``extract_loci``).

    Rows are dropped, and listed in ``dropped`` with the first reason that
    applies, when

    - ``"off_chrom"``: the sequence window or (with tracks) the signal window
      runs past either end of the chromosome;
    - ``"excluded"``: either window overlaps an interval of *exclude* by at
      least one base (intervals of *exclude* on chromosomes the genome does
      not have are ignored);
    - ``"min_counts"`` / ``"max_counts"``: the sum of the values of track
      expression *count_track* over the signal window (as returned, i.e.
      after binning, NaN counted as 0) is below *min_counts* or above
      *max_counts*.

    With ``use_strand=True`` and a ``strand`` column, the rows with strand -1
    (or "-") are reverse complemented: their one-hot encoding is that of the
    reverse complement, and their signals are reversed along the length axis.

    The arrays hold the same values, at the same ``[n, c, p]`` and
    ``[n, t, b]``, as R misha's ``gextract.windows``.

    Memory: ``x`` takes ``N * 4 * (width + 2 * jitter)`` bytes as uint8, and
    ``y`` takes ``N * T * (signal_width + 2 * jitter) // binsize * 4`` bytes
    as float32 for T track expressions. The signals are extracted with
    :func:`gextract_matrix` in chunks of about 1e7 values, so the extraction
    itself adds little to that. For 100,000 windows of 2 kb, ``x`` takes
    800 MB; ``y`` of three tracks at 1 bp resolution takes 2.4 GB, or 75 MB
    in 32 bp bins.

    Parameters
    ----------
    intervals : DataFrame, str, list or dict
        The loci: 1D intervals (a DataFrame or the name of an intervals set),
        or a list or dict of such, e.g. ``{"pos": peaks, "neg": background}``,
        whose rows are concatenated and labelled by a ``set`` column (the
        keys of the dict, or the 0-based indices of the list).
    tracks : str, list of str or dict, optional
        Track expressions for ``y``; a dict maps names to expressions. None
        for sequence only.
    width : int
        Width of the sequence window.
    signal_width : int, optional
        Width of the signal window; defaults to *width*.
    jitter : int, default 0
        Extra margin added to each side of both windows.
    binsize : int, default 1
        Bin size of the signals: a positive integer that divides
        ``signal_width + 2 * jitter``. Each value is the one
        :func:`gextract_matrix` returns for the bin (e.g. the mean of a dense
        track; use a virtual track with ``func="sum"`` for sums).
    use_strand : bool, default True
        If True, rows with strand -1 are reverse complemented.
    exclude : DataFrame or str, optional
        Intervals whose overlapping windows are dropped.
    min_counts, max_counts : float, optional
        Bounds on the sum of *count_track* over the signal window
        (inclusive).
    count_track : int or str, default 0
        The track expression the count filters use: a 0-based index into
        *tracks*, or one of its names.
    sequence : bool, default True
        If False, ``x`` is not computed (None).
    dtype : numpy dtype, default uint8
        dtype of ``x`` (see :func:`gseq_onehot`).
    signal_dtype : numpy dtype, default float32
        dtype of ``y``.
    vars : dict, optional
        Explicit variable bindings for the track expressions.

    Returns
    -------
    ExtractedWindows
        A named tuple ``(x, y, intervals, dropped)``: ``x`` of shape
        ``(N, 4, width + 2 * jitter)``; ``y`` of shape
        ``(N, T, (signal_width + 2 * jitter) // binsize)`` or None;
        ``intervals``, the kept rows in their original order with ``chrom``,
        ``start`` and ``end`` of the sequence window (jitter included), the
        ``strand`` column if any, ``signal_start`` and ``signal_end`` with
        tracks, the other columns of *intervals*, ``index`` (the 0-based row
        in *intervals*) and, for a list or dict, ``set``; and ``dropped``,
        with ``index`` and ``reason``.

    See Also
    --------
    gseq_onehot, gextract_matrix, gseq_extract

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import pandas as pd
    >>> loci = pd.DataFrame({"chrom": "chr1", "start": [5000, 20000, 100], "end": [5001, 20001, 101]})
    >>> w = pm.gextract_windows(loci, tracks={"dense": "dense_track"}, width=500, binsize=50)
    >>> w.x.shape, w.y.shape
    ((2, 4, 500), (2, 1, 10))
    >>> w.dropped  # the third locus is too close to the chromosome start
       index     reason
    0      2  off_chrom
    """
    if intervals is None or width is None:
        raise ValueError("Usage: gextract_windows(intervals, tracks=None, width=..., ...)")
    _checkroot()
    caller_ns = dict(vars) if vars is not None else _caller_namespace(depth=1)
    dt = _onehot_dtype(dtype)
    sdt = _numpy.dtype(signal_dtype)
    if sdt.kind != "f":
        raise ValueError("signal_dtype must be a floating point dtype")

    width = _whole(width, "width", 1)
    jitter = _whole(jitter, "jitter", 0)
    binsize = _whole(binsize, "binsize", 1)
    for name, v in (("use_strand", use_strand), ("sequence", sequence)):
        if not isinstance(v, (bool, _numpy.bool_)):
            raise ValueError(f"{name} must be True or False")

    exprs: list[str] = []
    track_names: list[str] = []
    signal_width_ = signal_len = 0
    if tracks is not None:
        if isinstance(tracks, str):
            tracks = [tracks]
        if isinstance(tracks, dict):
            track_names = [str(k) for k in tracks]
            exprs = [str(v) for v in tracks.values()]
        else:
            exprs = [str(t) for t in tracks]
            track_names = list(exprs)
        if not exprs:
            raise ValueError("tracks must hold at least one track expression")
        signal_width_ = _whole(width if signal_width is None else signal_width, "signal_width", 1)
        signal_len = signal_width_ + 2 * jitter
        if signal_len % binsize:
            raise ValueError(f"binsize ({binsize}) does not divide signal_width + 2 * jitter ({signal_len})")

    ct = None
    if min_counts is not None or max_counts is not None:
        if not exprs:
            raise ValueError("min_counts and max_counts are measured on tracks, so tracks must be given")
        for bound in (min_counts, max_counts):
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, numbers.Real) or _numpy.isnan(bound)
            ):
                raise ValueError("min_counts and max_counts must be numbers or None")
        if isinstance(count_track, str):
            if count_track not in track_names:
                raise ValueError("count_track must be an index into tracks or one of their names")
            ct = track_names.index(count_track)
        else:
            ct = _whole(count_track, "count_track", 0)
            if ct >= len(exprs):
                raise ValueError("count_track must be an index into tracks or one of their names")

    # the loci
    set_labels = None
    if isinstance(intervals, (list, tuple, dict)):
        items = list(intervals.items()) if isinstance(intervals, dict) else list(enumerate(intervals))
        frames = [_load_1d(iv) for _, iv in items]
        cols = [c for c in frames[0].columns if all(c in f.columns for f in frames)]
        set_labels = _numpy.repeat(_numpy.array([k for k, _ in items], dtype=object), [len(f) for f in frames])
        iv = _pandas.concat([f[cols] for f in frames], ignore_index=True)
    else:
        iv = _load_1d(intervals).reset_index(drop=True)
    n0 = len(iv)
    starts, ends = _whole_coords(iv)
    if (ends <= starts).any():
        raise ValueError("Interval start must be lesser than end")
    names, codes = _chrom_codes(iv["chrom"]) if n0 else ([], _numpy.zeros(0, dtype=_numpy.int32))
    chrom_lens = _chrom_sizes(names)[codes] if n0 else _numpy.zeros(0, dtype=_numpy.int64)
    minus = _minus_strand(iv) if use_strand else _numpy.zeros(n0, dtype=bool)

    mids = starts + (ends - starts) // 2
    seq_len = width + 2 * jitter
    seq_start = mids - width // 2 - jitter
    lo = seq_start.copy()
    hi = seq_start + seq_len
    sig_start = mids - signal_width_ // 2 - jitter
    if exprs:
        lo = _numpy.minimum(lo, sig_start)
        hi = _numpy.maximum(hi, sig_start + signal_len)

    reason = _numpy.full(n0, None, dtype=object)
    reason[(lo < 0) | (hi > chrom_lens)] = "off_chrom"
    if exclude is not None:
        hit = _windows_overlap(names, codes, lo, hi, _load_1d(exclude, "exclude"))
        reason[_pandas.isna(reason) & hit] = "excluded"

    keep = _numpy.flatnonzero(_pandas.isna(reason))
    y = None
    if exprs:
        num_bins = signal_len // binsize
        y = _numpy.empty((len(keep), len(exprs), num_bins), dtype=sdt)
        if len(keep):
            sig_iv = _pandas.DataFrame(
                {
                    "chrom": _numpy.array(names, dtype=object)[codes[keep]],
                    "start": sig_start[keep],
                    "end": sig_start[keep] + signal_len,
                }
            )
            chunk = max(1, int(1e7 // (num_bins * len(exprs))))
            for first in range(0, len(keep), chunk):
                last = min(first + chunk, len(keep))
                m = _gextract_matrix(exprs, sig_iv.iloc[first:last], binsize, None, caller_ns)
                m = m.reshape(last - first, num_bins, len(exprs))
                y[first:last] = m.transpose(0, 2, 1)
                del m
            neg = _numpy.flatnonzero(minus[keep])
            if len(neg):
                y[neg] = y[neg, :, ::-1]

        if ct is not None:
            counts = _numpy.nansum(y[:, ct, :], axis=1, dtype=_numpy.float64)
            failed = _numpy.full(len(keep), None, dtype=object)
            if max_counts is not None:
                failed[counts > max_counts] = "max_counts"
            if min_counts is not None:
                failed[counts < min_counts] = "min_counts"
            reason[keep] = failed
            ok = _pandas.isna(failed)
            if not ok.all():
                y = _numpy.ascontiguousarray(y[ok])
            keep = keep[ok]

    x = None
    if sequence:
        x = _numpy.empty((len(keep), 4, seq_len), dtype=dt)
        if len(keep):
            _pymisha.pm_seq_onehot(names, codes[keep], seq_start[keep], minus[keep].astype(_numpy.uint8), x, False)

    out = _pandas.DataFrame(
        {
            "chrom": iv["chrom"].to_numpy()[keep],
            "start": seq_start[keep],
            "end": seq_start[keep] + seq_len,
        }
    )
    if "strand" in iv.columns:
        out["strand"] = iv["strand"].to_numpy()[keep]
    if exprs:
        out["signal_start"] = sig_start[keep]
        out["signal_end"] = sig_start[keep] + signal_len
    for col in iv.columns:
        if col not in ("chrom", "start", "end", "strand", "index", "set") and col not in out.columns:
            out[col] = iv[col].to_numpy()[keep]
    out["index"] = keep
    if set_labels is not None:
        out["set"] = set_labels[keep]
    dropped_idx = _numpy.flatnonzero(~_pandas.isna(reason))
    dropped = _pandas.DataFrame({"index": dropped_idx, "reason": reason[dropped_idx].astype(str)})

    return ExtractedWindows(x, y, out, dropped)
