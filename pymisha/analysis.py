"""gsegment, gwilcox, and gcis_decay implementations."""

import bisect
import math
import warnings

import numpy as _numpy

from ._shared import (
    CONFIG,
    _checkroot,
    _df2pymisha,
    _pymisha,
    _pymisha2df,
)

_APPROX_QNORM_WARNED = False


def _pval_to_zscore(pval):
    """Convert a p-value to z-score using the normal distribution PPF.

    Equivalent to R's qnorm(pval). Uses the inverse error function
    to avoid scipy dependency.
    """
    # erfinv approximation: for p in (0,1), qnorm(p) = sqrt(2) * erfinv(2*p - 1)
    # Use math.erfc and its inverse via a rational approximation
    # This matches R's qnorm to high precision for typical p-value ranges
    if pval <= 0:
        return float('-inf')
    if pval >= 1:
        return float('inf')
    if pval == 0.5:
        return 0.0

    # Use scipy if available, otherwise fall back to approximation
    try:
        from scipy.stats import norm
        return norm.ppf(pval)
    except ImportError:
        pass

    # Rational approximation of the inverse normal CDF (Abramowitz & Stegun 26.2.23)
    # Accurate to ~4.5e-4.
    global _APPROX_QNORM_WARNED
    if not _APPROX_QNORM_WARNED:
        warnings.warn(
            "scipy is not installed; using an approximate inverse normal CDF "
            "for p-value to z-score conversion (accuracy ~4.5e-4).",
            RuntimeWarning,
            stacklevel=2,
        )
        _APPROX_QNORM_WARNED = True

    if pval < 0.5:
        t = math.sqrt(-2.0 * math.log(pval))
        # Coefficients for the rational approximation
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
    t = math.sqrt(-2.0 * math.log(1.0 - pval))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def gsegment(expr, minsegment, maxpval=0.05, onetailed=True, intervals=None,
             iterator=None, intervals_set_out=None):
    """
    Divide track expression into segments using Wilcoxon test.

    Divides the values of a track expression into segments, where each
    segment size is at least ``minsegment`` and the P-value of comparing
    the segment with the first ``minsegment`` values from the next segment
    is at most ``maxpval``. Comparison is done using the Wilcoxon
    (Mann-Whitney) test.

    Parameters
    ----------
    expr : str
        Track expression.
    minsegment : int
        Minimal segment size in base pairs.
    maxpval : float, optional
        Maximal P-value that separates two adjacent segments. Default 0.05.
    onetailed : bool, optional
        If True, Wilcoxon test is one-tailed. Default True.
    intervals : DataFrame, optional
        Genomic scope. Defaults to all genome intervals.
    iterator : int, optional
        Fixed bin iterator size. If None, determined from track expression.
    intervals_set_out : str, optional
        If provided, save result as an intervals set and return None.

    Returns
    -------
    DataFrame or None
        Intervals where each row represents a segment (chrom, start, end).
        Returns None if intervals_set_out is provided, or if input is empty.

    See Also
    --------
    gwilcox : Sliding-window Wilcoxon test.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gsegment("dense_track", 5000, maxpval=0.0001)
    >>> result.columns.tolist()
    ['chrom', 'start', 'end']
    """
    _checkroot()

    if intervals is None:
        from .intervals import gintervals_all
        intervals = gintervals_all()

    if intervals is None or (hasattr(intervals, '__len__') and len(intervals) == 0):
        return None

    maxz = _pval_to_zscore(maxpval)

    result = _pymisha.pm_segment(
        str(expr),
        _df2pymisha(intervals),
        float(minsegment),
        float(maxz),
        int(bool(onetailed)),
        iterator,
        CONFIG,
    )

    df = _pymisha2df(result)

    if intervals_set_out is not None:
        if df is not None and len(df) > 0:
            from .intervals import gintervals_save
            gintervals_save(df, intervals_set_out)
        return None

    return df


def gwilcox(expr, winsize1, winsize2, maxpval=0.05, onetailed=True,
            what2find=1, intervals=None, iterator=None,
            intervals_set_out=None):
    """
    Sliding-window Wilcoxon test over track expression values.

    Runs a Wilcoxon test (Mann-Whitney) over the values of a track expression
    in two sliding windows with an identical center. Returns intervals where
    the smaller window tested against the larger window gives a P-value below
    ``maxpval``.

    Parameters
    ----------
    expr : str
        Track expression.
    winsize1 : int
        Size of the first sliding window in base pairs.
    winsize2 : int
        Size of the second sliding window in base pairs.
    maxpval : float, optional
        Maximal P-value threshold. Default 0.05.
    onetailed : bool, optional
        If True, Wilcoxon test is one-tailed. Default True.
    what2find : int, optional
        -1 for lows, 1 for peaks, 0 for both. Default 1.
    intervals : DataFrame, optional
        Genomic scope. Defaults to all genome intervals.
    iterator : int, optional
        Fixed bin iterator size. If None, determined from track expression.
    intervals_set_out : str, optional
        If provided, save result as an intervals set and return None.

    Returns
    -------
    DataFrame or None
        Intervals with ``pval`` column where P-value is below ``maxpval``.
        Returns None if no significant regions found, input is empty, or
        intervals_set_out is provided.

    See Also
    --------
    gsegment : Divide track expression into segments using Wilcoxon test.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gwilcox("dense_track", 100000, 1000, maxpval=0.01, what2find=1)
    >>> result is None or "chrom" in result.columns
    True
    """
    _checkroot()

    if intervals is None:
        from .intervals import gintervals_all
        intervals = gintervals_all()

    if intervals is None or (hasattr(intervals, '__len__') and len(intervals) == 0):
        return None

    effective_pval = maxpval
    if not onetailed:
        effective_pval = maxpval / 2.0

    maxz = _pval_to_zscore(effective_pval)

    result = _pymisha.pm_wilcox(
        str(expr),
        _df2pymisha(intervals),
        float(winsize1),
        float(winsize2),
        float(maxz),
        int(bool(onetailed)),
        int(what2find),
        iterator,
        CONFIG,
    )

    df = _pymisha2df(result)

    if intervals_set_out is not None:
        if df is not None and len(df) > 0:
            from .intervals import gintervals_save
            gintervals_save(df, intervals_set_out)
        return None

    return df


# ---------------------------------------------------------------------------
# gcis_decay helpers
# ---------------------------------------------------------------------------

def _val2bin(val, breaks, include_lowest):
    """Map *val* to a bin index given sorted *breaks*.

    Bins are half-open intervals ``(breaks[i], breaks[i+1]]``.
    When *include_lowest* is True the first bin becomes ``[breaks[0], breaks[1]]``.
    Returns -1 when *val* falls outside all bins or is NaN.

    This replicates the C++ ``BinFinder::val2bin`` logic (right=True).
    """
    if val != val:  # NaN check
        return -1
    n = len(breaks)
    if n < 2:
        return -1
    if include_lowest and val == breaks[0]:
        return 0
    if val <= breaks[0] or val > breaks[-1]:
        return -1
    # Binary search: find rightmost break <= val
    idx = bisect.bisect_left(breaks, val) - 1
    # bisect_left gives the insertion point; the bin is idx clamped to [0, n-2]
    if idx < 0:
        idx = 0
    if idx >= n - 1:
        idx = n - 2
    return idx


def _unify_overlaps_per_chrom(df):
    """Sort intervals, merge overlapping ones, return dict[chrom -> sorted list of (start, end)].

    The input DataFrame must have columns ``chrom``, ``start``, ``end``.
    """
    result = {}
    if df is None or len(df) == 0:
        return result
    for chrom, group in df.groupby("chrom"):
        starts = group["start"].values
        ends = group["end"].values
        # sort by start
        order = starts.argsort()
        starts = starts[order]
        ends = ends[order]
        merged = []
        cs, ce = int(starts[0]), int(ends[0])
        for i in range(1, len(starts)):
            s, e = int(starts[i]), int(ends[i])
            if s <= ce:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        result[str(chrom)] = merged
    return result


def _intervals_per_chrom(df):
    """Group non-overlapping intervals by chrom -> sorted list of (start, end).

    Used for domain intervals which must not overlap.
    """
    result = {}
    if df is None or len(df) == 0:
        return result
    for chrom, group in df.groupby("chrom"):
        starts = group["start"].values
        ends = group["end"].values
        order = starts.argsort()
        intervals = [(int(starts[i]), int(ends[i])) for i in order]
        result[str(chrom)] = intervals
    return result


def _containing_interval(intervals_sorted, start, end):
    """Return the index of the interval that fully contains [start, end), or -1.

    *intervals_sorted* is a sorted list of ``(istart, iend)`` tuples.
    Uses binary search for efficiency.
    """
    if not intervals_sorted:
        return -1
    # Binary search: find the last interval whose start <= start
    lo, hi = 0, len(intervals_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals_sorted[mid][0] <= start:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if idx < 0:
        return -1
    istart, iend = intervals_sorted[idx]
    if istart <= start and end <= iend:
        return idx
    return -1


def gcis_decay(expr, breaks, src, domain, intervals=None,
               include_lowest=False, iterator=None, band=None):
    """
    Calculate distribution of cis contact distances.

    For contacts where ``chrom1`` equals ``chrom2`` and the first interval
    (I1) is fully within ``src`` intervals, this function bins the distance
    between I1 and I2 separately for intra-domain and inter-domain contacts.

    A contact is *intra-domain* when both I1 and I2 are fully contained
    within the **same** domain interval. Otherwise it is *inter-domain*.

    The distance is ``abs((start1 + end1 - start2 - end2) / 2)`` (integer
    division), i.e. the absolute difference of the interval midpoints.

    Parameters
    ----------
    expr : str
        A 2D track expression (must be a simple 2D track name).
    breaks : array_like
        Sorted break points defining distance bins.
        Example: ``breaks=[x1, x2, x3]`` creates bins ``(x1, x2]`` and
        ``(x2, x3]``.
    src : DataFrame
        Source intervals (chrom, start, end). Only contacts whose I1 is
        fully within the unified source intervals are counted.
        Overlapping source intervals are allowed and will be merged.
    domain : DataFrame
        Domain intervals (chrom, start, end). Must be non-overlapping.
        Used to classify contacts as intra- or inter-domain.
    intervals : DataFrame, optional
        Genomic scope (1D intervals). Defaults to all genome intervals.
        Only cis contacts (chrom1 == chrom2) within these chromosomes
        are considered.
    include_lowest : bool, default False
        If True, the lowest break value is included in the first bin:
        ``[x1, x2]`` instead of ``(x1, x2]``.
    iterator : str, optional
        2D iterator specification. Currently unused (extraction uses the
        track's native resolution).
    band : tuple of (int, int), optional
        Diagonal band filter ``(d1, d2)``. Only contacts where the
        diagonal offset falls within the band are considered.

    Returns
    -------
    numpy.ndarray
        2D array of shape ``(n_bins, 2)`` where column 0 is *intra*-domain
        counts and column 1 is *inter*-domain counts. Row and column labels
        are stored as a ``breaks`` attribute on the array.

    See Also
    --------
    gdist : General distribution of track expressions.
    gextract : Extract track values over intervals.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import pandas as pd
    >>> src = pd.DataFrame({"chrom": ["1", "1"], "start": [0, 200000], "end": [100000, 400000]})
    >>> domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
    >>> breaks = [0, 100000, 200000, 300000, 400000, 500000]
    >>> result = pm.gcis_decay("rects_track", breaks, src, domain)
    >>> result.shape[1]
    2
    """
    _checkroot()

    if expr is None or breaks is None or src is None or domain is None:
        raise ValueError(
            "Usage: gcis_decay(expr, breaks, src, domain, "
            "intervals=None, include_lowest=False, iterator=None, band=None)"
        )

    from ._quadtree import query_2d_track_objects
    from .extract import _find_2d_track_file, _obj_in_band, _validate_band
    from .intervals import _normalize_chroms, gintervals_all
    from .tracks import gtrack_info

    breaks = [float(b) for b in breaks]
    if len(breaks) < 2:
        raise ValueError("breaks must have at least 2 elements")

    n_bins = len(breaks) - 1
    intra_dist = _numpy.zeros(n_bins, dtype=_numpy.float64)
    inter_dist = _numpy.zeros(n_bins, dtype=_numpy.float64)

    # Normalize chromosome names in src and domain
    src = src.copy()
    if "chrom" in src.columns:
        src["chrom"] = _normalize_chroms(src["chrom"].astype(str).tolist())
    domain = domain.copy()
    if "chrom" in domain.columns:
        domain["chrom"] = _normalize_chroms(domain["chrom"].astype(str).tolist())

    # Build per-chrom lookup structures
    src_per_chrom = _unify_overlaps_per_chrom(src)
    domain_per_chrom = _intervals_per_chrom(domain)

    # Validate band
    band = _validate_band(band)

    # Get track info
    info = gtrack_info(expr)
    if info.get("dimensions") != 2:
        raise ValueError(f"Track '{expr}' is not a 2D track")
    is_points = info.get("type") == "points"

    track_path = _pymisha.pm_track_path(expr)

    # Determine which chromosomes to iterate over
    if intervals is None:
        intervals = gintervals_all()

    if intervals is not None and "chrom" in intervals.columns:
        intervals = intervals.copy()
        intervals["chrom"] = _normalize_chroms(
            intervals["chrom"].astype(str).tolist()
        )

    # Build chrom -> max_end mapping from intervals (ALLGENOME gives full chrom sizes)
    chrom_sizes = {}
    if intervals is not None and len(intervals) > 0:
        for _, row in intervals.iterrows():
            c = str(row["chrom"])
            e = int(row["end"])
            chrom_sizes[c] = max(chrom_sizes.get(c, 0), e)

    chroms = list(chrom_sizes.keys())

    # Iterate over cis chromosome pairs
    for chrom in chroms:
        filepath = _find_2d_track_file(track_path, chrom, chrom)
        if filepath is None:
            continue

        src_intervals = src_per_chrom.get(str(chrom), [])
        if not src_intervals:
            continue

        domain_intervals = domain_per_chrom.get(str(chrom), [])

        csize = chrom_sizes[chrom]

        # Query all objects in this chrom-chrom file
        objs = query_2d_track_objects(filepath, 0, 0, csize, csize)

        for obj in objs:
            # Apply band filter if specified
            if band is not None and not _obj_in_band(obj, is_points, band):
                continue

            if is_points:
                x, y, val = obj
                s1, e1 = x, x + 1
                s2, e2 = y, y + 1
            else:
                x1, y1, x2, y2, val = obj
                s1, e1 = x1, x2
                s2, e2 = y1, y2

            # Check if I1 is fully within unified src intervals
            if _containing_interval(src_intervals, s1, e1) < 0:
                continue

            # Distance between interval centers (integer division as in C++)
            distance = abs((s1 + e1 - s2 - e2) // 2)

            idx = _val2bin(distance, breaks, include_lowest)
            if idx < 0:
                continue

            # Check domain containment for intra/inter classification
            d1_idx = _containing_interval(domain_intervals, s1, e1)
            d2_idx = _containing_interval(domain_intervals, s2, e2)

            if d1_idx >= 0 and d1_idx == d2_idx:
                intra_dist[idx] += 1
            else:
                inter_dist[idx] += 1

    # Build result: 2D array (n_bins x 2), column-major like R
    result = _numpy.column_stack([intra_dist, inter_dist])

    # Build bin labels matching R's BinsManager format: "(x1,x2]"
    bin_labels = []
    for i in range(n_bins):
        left = "[" if (include_lowest and i == 0) else "("
        bin_labels.append(f"{left}{breaks[i]:g},{breaks[i+1]:g}]")

    # Return as a structured result with attributes accessible via .breaks
    class CisDecayResult(_numpy.ndarray):
        """ndarray subclass with breaks and label metadata."""

        def __new__(cls, data, breaks_attr, bin_labels_attr):
            obj = _numpy.asarray(data).view(cls)
            obj.breaks = breaks_attr
            obj.bin_labels = bin_labels_attr
            obj.col_labels = ["intra", "inter"]
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.breaks = getattr(obj, "breaks", None)
            self.bin_labels = getattr(obj, "bin_labels", None)
            self.col_labels = getattr(obj, "col_labels", None)

        def __repr__(self):
            # Produce a readable table similar to R's print
            lines = []
            header = "         " + "  ".join(f"{c:>8s}" for c in self.col_labels)
            lines.append(header)
            for i, label in enumerate(self.bin_labels):
                row = f"{label:>9s}" + "  ".join(
                    f"{self[i, j]:>8.0f}" for j in range(self.shape[1])
                )
                lines.append(row)
            return "\n".join(lines)

    return CisDecayResult(result, breaks, bin_labels)
