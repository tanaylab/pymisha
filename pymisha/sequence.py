"""Sequence manipulation functions (gseq.*)."""

from __future__ import annotations

import collections.abc
import math as _math

import numpy as _numpy
import pandas as _pandas

from ._shared import (
    CONFIG,
    _check_intervals_strand,
    _check_spat_bounds,
    _checkroot,
    _coerce_pssm_matrix,
    _coerce_score_thresh,
    _df2pymisha,
    _is_na,
    _potts_params,
    _pymisha,
    _seq_bytes,
    _seq_upper,
)
from .extract import _maybe_load_intervals_set

# Complement mapping for DNA bases
_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_VALID_BASES = frozenset("ACGTacgt")
_BASE4_ALPHABET = "ACGT"
_DNA_CODE_TABLE = _numpy.full(256, -1, dtype=_numpy.int8)
for _base, _code in ((ord("A"), 0), (ord("C"), 1), (ord("G"), 2), (ord("T"), 3)):
    _DNA_CODE_TABLE[_base] = _code
    _DNA_CODE_TABLE[ord(chr(_base).lower())] = _code

# Pre-built k-mer string tables (lazily populated)
_KMER_STRING_CACHE: dict[int, _numpy.ndarray] = {}


def _kmer_strings(k: int) -> _numpy.ndarray:
    """Return array of all 4**k k-mer strings for a given k, cached."""
    if k in _KMER_STRING_CACHE:
        return _KMER_STRING_CACHE[k]
    num = 4**k
    # Build via cartesian product of base indices
    codes = _numpy.arange(num, dtype=_numpy.int64)
    chars = _numpy.empty((num, k), dtype="U1")
    for pos in range(k - 1, -1, -1):
        chars[:, pos] = _numpy.array(list(_BASE4_ALPHABET))[codes % 4]
        codes //= 4
    # Join each row into a single string
    # Use a view-based approach: build bytes then decode
    result = _numpy.array(["".join(row) for row in chars], dtype=f"U{k}")
    _KMER_STRING_CACHE[k] = result
    return result


# PWM scoring constants
_PWM_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_PWM_COMP_CODE = {"A": 3, "C": 2, "G": 1, "T": 0}  # complement base codes
_LOG_QUARTER = _math.log(0.25)


def gseq_extract(intervals: _pandas.DataFrame | str, pad: bool = False) -> list[str]:
    """
    Extract DNA sequences for given intervals.

    Returns an array of sequence strings for each interval from 'intervals'.
    If intervals contain an additional 'strand' column and its value is -1,
    the reverse-complementary sequence is returned.

    By default (``pad=False``) each interval must lie within its chromosome:
    an interval that starts before 0 or ends past the end of the chromosome
    is an error.

    With ``pad=True`` an interval may run past either end of its chromosome
    (``start`` below 0, ``end`` above the chromosome length). The positions
    outside the chromosome come back as "N", so each sequence is
    ``end - start`` long, and an interval wholly outside its chromosome comes
    back as all "N". ``start`` must still be below ``end``, and the chromosome
    must exist. For a row with strand -1 the padded sequence is reverse
    complemented as a whole, so the "N" run of the chromosome start comes
    last. The bases inside the chromosome are returned as stored, lowercase
    included. This gives fixed-width windows near chromosome ends, e.g. for
    sequence models (see the examples).

    Parameters
    ----------
    intervals : DataFrame
        Intervals for which DNA sequence is returned. Must have 'chrom',
        'start', and 'end' columns. Optional 'strand' column (-1 for reverse
        complement).
    pad : bool, default False
        If True, the part of an interval outside its chromosome is returned
        as "N". If False, an interval that runs past a chromosome end is an
        error.

    Returns
    -------
    list of str
        Array of character strings representing DNA sequence.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals(["1", "2"], [10000, 10000], [10020, 10020])
    >>> pm.gseq_extract(intervs)  # doctest: +ELLIPSIS
    [...]

    Fixed-width windows centred on positions. The second window starts 30 bp
    before the start of chr1, so its first 30 bases are N.

    >>> import pandas as pd
    >>> w = 100
    >>> centers = [250000, 20]
    >>> windows = pd.DataFrame({"chrom": "chr1", "start": [c - w // 2 for c in centers],
    ...                         "end": [c + w // 2 for c in centers]})
    >>> seqs = pm.gseq_extract(windows, pad=True)
    >>> [len(s) for s in seqs]
    [100, 100]
    >>> seqs[1][:30] == "N" * 30
    True

    See Also
    --------
    gseq_rev, gseq_comp, gseq_revcomp, gseq_kmer
    """
    if intervals is None:
        raise ValueError("Usage: gseq_extract(intervals, pad=False)")
    if not isinstance(pad, (bool, _numpy.bool_)):
        raise ValueError("pad must be True or False")

    _checkroot()

    intervals = _maybe_load_intervals_set(intervals)
    if (
        isinstance(intervals, _pandas.DataFrame)
        and "strand" in intervals.columns
        and isinstance(intervals["strand"].dtype, _pandas.CategoricalDtype)
    ):
        # a categorical strand is read by its labels, as R reads a factor
        intervals = intervals.assign(strand=intervals["strand"].astype(object))

    if pad:
        return _gseq_extract_padded(intervals)
    return list(_pymisha.pm_seq_extract(_df2pymisha(intervals), CONFIG))


def _gseq_extract_padded(intervals: _pandas.DataFrame | str) -> list[str]:
    """gseq_extract(pad=True), as R's .gseq_extract_padded.

    The rows that run past a chromosome end are clipped to the chromosome and
    read with all the other rows in one call, which validates them as usual;
    the N runs are added here. A row wholly outside its chromosome reads a 1 bp
    placeholder at the chromosome start, so that its chrom and strand are still
    validated, and is all N. Rows within their chromosome are read as they are,
    and when there are only such rows the intervals are read untouched.
    """

    def read(iv: _pandas.DataFrame | str) -> list[str]:
        return list(_pymisha.pm_seq_extract(_df2pymisha(iv), CONFIG))

    # anything but a 1D intervals frame is read (or rejected) as without pad
    if (
        not isinstance(intervals, _pandas.DataFrame)
        or not len(intervals)
        or not {"chrom", "start", "end"}.issubset(intervals.columns)
        or not _pandas.api.types.is_numeric_dtype(intervals["start"])
        or not _pandas.api.types.is_numeric_dtype(intervals["end"])
    ):
        return read(intervals)

    from .intervals import _normalize_chroms, gintervals_all

    chrom_col = intervals["chrom"]
    if isinstance(chrom_col.dtype, _pandas.CategoricalDtype):
        chrom_col = chrom_col.astype(object)
    starts = intervals["start"].to_numpy(dtype=_numpy.float64)
    ends = intervals["end"].to_numpy(dtype=_numpy.float64)
    uchroms, inverse = _numpy.unique(chrom_col.astype(str).to_numpy(), return_inverse=True)
    sizes = gintervals_all()
    size_of = dict(zip(sizes["chrom"].astype(str), sizes["end"].to_numpy(dtype=_numpy.float64), strict=True))
    # an unknown chromosome is the same error as without pad
    ulens = _numpy.array([size_of[c] for c in _normalize_chroms(list(uchroms))], dtype=_numpy.float64)
    chrom_lens = ulens[inverse]

    # rows with NaN or infinite coordinates are left to the read, which rejects them
    with _numpy.errstate(invalid="ignore"):
        idx = _numpy.flatnonzero(
            _numpy.isfinite(starts) & _numpy.isfinite(ends) & ((starts < 0) | (ends > chrom_lens))
        )
    if not idx.size:
        return read(intervals)

    s = starts[idx]
    e = ends[idx]
    lens = chrom_lens[idx]
    bad = _numpy.flatnonzero(s >= e)
    if bad.size:
        i = bad[0]
        raise ValueError(
            f"Interval ({chrom_col.iloc[idx[i]]}, {s[i]:.0f}, {e[i]:.0f}): "
            "start coordinate must be lesser than end coordinate"
        )

    cs = _numpy.maximum(s, 0)
    ce = _numpy.minimum(e, lens)
    outside = cs >= ce
    left = _numpy.where(outside, e - s, cs - s)
    right = _numpy.where(outside, 0, e - ce)
    cs[outside] = 0
    ce[outside] = 1

    clipped = _pandas.DataFrame({
        "chrom": intervals["chrom"].to_numpy(),
        "start": intervals["start"].to_numpy(copy=True),
        "end": intervals["end"].to_numpy(copy=True),
    })
    strand = intervals["strand"] if "strand" in intervals.columns else None
    if strand is not None:
        clipped["strand"] = strand.to_numpy()
    start_col = clipped.columns.get_loc("start")
    end_col = clipped.columns.get_loc("end")
    clipped.iloc[idx, start_col] = cs.astype(clipped["start"].dtype)
    clipped.iloc[idx, end_col] = ce.astype(clipped["end"].dtype)

    # an empty chromosome has nothing to read, not even the placeholder
    to_read = _numpy.ones(len(clipped), dtype=bool)
    to_read[idx[outside & (lens == 0)]] = False
    res = [""] * len(clipped)
    if to_read.any():
        seqs = read(clipped if to_read.all() else clipped[to_read].reset_index(drop=True))
        for i, seq in zip(_numpy.flatnonzero(to_read), seqs, strict=True):
            res[i] = seq

    # a strand -1 row comes back reverse complemented, so its N runs swap sides
    def is_minus(v: object) -> bool:
        if isinstance(v, str):
            return v == "-"
        return isinstance(v, (int, float, _numpy.number)) and not isinstance(v, (bool, _numpy.bool_)) and v == -1

    neg = (
        _numpy.zeros(idx.size, dtype=bool)
        if strand is None
        else _numpy.array([is_minus(v) for v in strand.to_numpy()[idx]], dtype=bool)
    )

    for j, i in enumerate(idx):
        core = "" if outside[j] else res[i]
        n_left = "N" * int(left[j])
        n_right = "N" * int(right[j])
        res[i] = n_right + core + n_left if neg[j] else n_left + core + n_right
    return res


def gseq_rev(seq: str | list[str]) -> str | list[str]:
    """
    Reverse DNA sequence(s).

    Parameters
    ----------
    seq : str or list of str
        DNA sequence(s) to reverse.

    Returns
    -------
    str or list of str
        Reversed sequence(s).

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.gseq_rev("ACGT")
    'TGCA'
    >>> pm.gseq_rev(["ACGT", "TATA"])
    ['TGCA', 'ATAT']

    See Also
    --------
    gseq_comp, gseq_revcomp, gseq_extract
    """
    if isinstance(seq, str):
        return seq[::-1]
    return [s[::-1] for s in seq]


def gseq_comp(seq: str | list[str]) -> str | list[str]:
    """
    Return complement of DNA sequence(s).

    Parameters
    ----------
    seq : str or list of str
        DNA sequence(s) to complement.

    Returns
    -------
    str or list of str
        Complemented sequence(s).

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.gseq_comp("ACGT")
    'TGCA'
    >>> pm.gseq_comp(["ACGT", "AAAA"])
    ['TGCA', 'TTTT']

    See Also
    --------
    gseq_rev, gseq_revcomp, gseq_extract
    """
    if isinstance(seq, str):
        return seq.translate(_COMPLEMENT)
    return [s.translate(_COMPLEMENT) for s in seq]


def gseq_revcomp(seq: str | list[str]) -> str | list[str]:
    """
    Return reverse complement of DNA sequence(s).

    Parameters
    ----------
    seq : str or list of str
        DNA sequence(s) to reverse complement.

    Returns
    -------
    str or list of str
        Reverse complemented sequence(s).

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.gseq_revcomp("AACG")
    'CGTT'
    >>> pm.gseq_revcomp(["AACG", "AAAA"])
    ['CGTT', 'TTTT']

    See Also
    --------
    gseq_rev, gseq_comp, gseq_extract
    """
    if isinstance(seq, str):
        return seq[::-1].translate(_COMPLEMENT)
    return [s[::-1].translate(_COMPLEMENT) for s in seq]


def grevcomp(seq: str | list[str]) -> str | list[str]:
    """
    Return reverse complement of a DNA string.

    Standalone reverse-complement function matching R misha's ``grevcomp``.
    Handles A/C/G/T and their lowercase equivalents; N and other IUPAC
    characters are complemented according to the standard complement table.

    Parameters
    ----------
    seq : str or list of str
        DNA sequence(s) to reverse complement.

    Returns
    -------
    str or list of str
        Reverse complemented sequence(s), same type as input.

    Examples
    --------
    >>> import pymisha as pm
    >>> pm.grevcomp("ACTG")
    'CAGT'
    >>> pm.grevcomp(["ACTG", "GGCC"])
    ['CAGT', 'GGCC']

    See Also
    --------
    gseq_revcomp, gseq_rev, gseq_comp
    """
    return gseq_revcomp(seq)


def _na_or_seq(x: object) -> bool:
    """R's NA, or more than one value."""
    return _is_na(x) or isinstance(x, (list, tuple, _numpy.ndarray, _pandas.Series))


def _seq_extend_arg(extend: object) -> bool | int:
    """``extend`` of gseq_pwm / gseq_kmer as R's .normalize_bounds takes it:
    True, False, or a non-negative number (truncated, as as.integer())."""
    if isinstance(extend, (bool, _numpy.bool_)):
        return bool(extend)
    if (isinstance(extend, (int, float, _numpy.integer, _numpy.floating))
            and not _is_na(extend) and extend >= 0):
        return int(extend)
    raise ValueError("extend must be False, True, or a non-negative integer")


def _seq_strand_arg(strand: object) -> int:
    """``strand`` of gseq_pwm / gseq_kmer / gseq_potts: -1, 0 or 1; R also takes
    TRUE and FALSE (1 and 0)."""
    if (_na_or_seq(strand) or not isinstance(strand, (int, float, _numpy.number, bool, _numpy.bool_))
            or strand not in (-1, 0, 1)):
        raise ValueError("strand must be -1, 0, or 1")
    return int(strand)  # type: ignore[call-overload]


def _skip_gaps_arg(skip_gaps: object) -> bool:
    """R reads skip_gaps as a logical: anything but NA."""
    if _is_na(skip_gaps):
        raise ValueError("skip_gaps must be True or False")
    return bool(skip_gaps)


def gseq_kmer(
    seqs: str | list[str],
    kmer: str,
    mode: str = "count",
    strand: int = 0,
    start_pos: int | list[int] | None = None,
    end_pos: int | list[int] | None = None,
    extend: bool | int = False,
    skip_gaps: bool = True,
    gap_chars: list[str] | None = None,
) -> _numpy.ndarray:
    """
    Count k-mer occurrences in DNA sequences.

    Scans every window start of the region of interest (ROI) and counts the
    starts where the k-mer matches. With ``strand=0`` a start counts once when
    the k-mer matches on either strand, so a palindromic k-mer is not counted
    twice. ``mode="frac"`` divides the count by the number of window starts
    scanned, whatever the strand.

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) to search (case-insensitive).
    kmer : str
        K-mer pattern to search for (only A, C, G, T characters).
    mode : str, default "count"
        "count" returns raw counts, "frac" returns the fraction of window
        starts that match.
    strand : int, default 0
        0 = either strand, 1 = forward only, -1 = reverse complement only.
    start_pos : int or list of int, optional
        1-based start of the ROI, per sequence (recycled); 1 by default.
    end_pos : int or list of int, optional
        1-based end of the ROI (inclusive), per sequence (recycled); the
        sequence end by default.
    extend : bool or int, default False
        How far window starts may reach past the ROI on each side: False for
        none, True for ``len(kmer) - 1``, or a non-negative number of bases.
    skip_gaps : bool, default True
        If True, gap characters are removed before scanning.
    gap_chars : list of str, optional
        Characters to treat as gaps. Default: ["-", "."].

    Returns
    -------
    numpy.ndarray
        Array of counts or fractions, one per input sequence.

    Examples
    --------
    >>> import pymisha as pm

    Count CG dinucleotides (a palindrome counts once per position):

    >>> pm.gseq_kmer(["CGCGCGCGCG", "ATATATATAT"], "CG")
    array([5., 0.])

    Get fraction instead of count:

    >>> pm.gseq_kmer(["CGCGCGCGCG"], "CG", mode="frac")  # doctest: +ELLIPSIS
    array([0.555...])

    Count in a specific region:

    >>> pm.gseq_kmer("ATCGATCG", "AT", start_pos=1, end_pos=4)
    array([1.])

    See Also
    --------
    gseq_kmer_dist, gseq_pwm, gseq_extract
    """
    if mode not in ("count", "frac"):
        raise ValueError("mode must be 'count' or 'frac'")
    if not isinstance(kmer, str):
        raise ValueError("kmer must be a single character string")
    kmer = kmer.upper()
    if not kmer or not all(c in "ACGT" for c in kmer):
        raise ValueError("kmer must contain only A, C, G, T characters")
    strand = _seq_strand_arg(strand)
    skip_gaps = _skip_gaps_arg(skip_gaps)
    if gap_chars is None:
        gap_chars = ["-", "."]
    if skip_gaps:
        if isinstance(gap_chars, str) or len(gap_chars) == 0:
            raise ValueError("gap_chars must be a non-empty list of characters")
        if any(not isinstance(g, str) or len(g) != 1 for g in gap_chars):
            raise ValueError("gap_chars must be single characters")
        if len(set(gap_chars)) != len(gap_chars):
            raise ValueError("gap_chars must be distinct")
    ext = _seq_extend_arg(extend)
    extend_val = ext if not isinstance(ext, bool) else len(kmer) - 1 if ext else 0

    if isinstance(seqs, str):
        seqs = [seqs]
    # one character per byte, as the C++ side reads them (Latin-1)
    seqs = [_seq_bytes(x) if isinstance(x, str) else x for x in seqs]
    n = len(seqs)
    lengths = _numpy.array([len(x) if isinstance(x, str) else 0 for x in seqs], dtype=_numpy.int64)
    roi_start = (_numpy.ones(n, dtype=_numpy.int64) if start_pos is None
                 else _numpy.resize(_numpy.asarray(start_pos, dtype=_numpy.int64), n))
    roi_end = (lengths if end_pos is None
               else _numpy.resize(_numpy.asarray(end_pos, dtype=_numpy.int64), n))

    result: _numpy.ndarray = _pymisha.pm_kmer_count_strings(
        seqs, kmer, mode, int(strand), roi_start, roi_end, extend_val, bool(skip_gaps),
        "".join(gap_chars) if skip_gaps else "",
    )
    return result


def _unify_overlaps(intervals: _pandas.DataFrame | str) -> _pandas.DataFrame | str:
    """*intervals* with overlapping intervals unified and touching ones kept apart, as R's
    GIntervals::unify_overlaps(false): a unified interval keeps the other columns (the strand) of
    its first interval by start. Intervals without overlaps are returned as they are."""
    if not isinstance(intervals, _pandas.DataFrame) or len(intervals) < 2 or not {
        "chrom", "start", "end"
    }.issubset(intervals.columns):
        return intervals
    from .intervals import _normalize_chroms

    df = intervals.assign(chrom=_normalize_chroms(intervals["chrom"].tolist()))
    df = df.sort_values(["chrom", "start"], kind="mergesort").reset_index(drop=True)
    chroms = df["chrom"].to_numpy()
    starts = df["start"].to_numpy(dtype=_numpy.float64)
    ends = df["end"].to_numpy(dtype=_numpy.float64)
    new_chrom = _numpy.ones(len(df), dtype=bool)
    new_chrom[1:] = chroms[1:] != chroms[:-1]
    # the furthest end of the rows before each one on its chromosome
    running_end = df.assign(end=ends).groupby("chrom", sort=False)["end"].cummax().to_numpy()
    prev_end = _numpy.empty(len(df))
    prev_end[1:] = running_end[:-1]
    prev_end[new_chrom] = -_numpy.inf
    starts_new = new_chrom | (starts >= prev_end)
    if starts_new.all():
        return intervals
    group = _numpy.cumsum(starts_new) - 1
    group_end = _numpy.full(group[-1] + 1, -_numpy.inf)
    _numpy.maximum.at(group_end, group, ends)
    out = df[starts_new].reset_index(drop=True)
    out["end"] = group_end.astype(df["end"].dtype)
    return out


def gseq_kmer_dist(
    intervals: _pandas.DataFrame | str, k: int = 6, mask: _pandas.DataFrame | None = None
) -> _pandas.DataFrame:
    """
    Compute k-mer distribution in genomic intervals.

    Counts all k-mers of size k within the specified genomic intervals,
    optionally excluding masked regions.

    Parameters
    ----------
    intervals : DataFrame
        Genomic intervals to analyze. Overlapping intervals are unified
        first, so each position is counted once; a k-mer that crosses from
        an interval into an adjacent one is not counted.
    k : int, default 6
        K-mer size (1-10).
    mask : DataFrame, optional
        Intervals to exclude: a k-mer with any base in one is not counted.

    Returns
    -------
    DataFrame
        DataFrame with columns 'kmer' (str) and 'count' (int).
        Only k-mers with count > 0 are included.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pm.gintervals(1, 0, 10000)
    >>> result = pm.gseq_kmer_dist(intervs, k=2)
    >>> list(result.columns)
    ['kmer', 'count']

    See Also
    --------
    gseq_kmer, gseq_pwm, gseq_extract
    """
    _checkroot()

    k = int(k)
    if k < 1 or k > 10:
        raise ValueError("k must be an integer between 1 and 10")

    if intervals is None:
        raise ValueError("intervals cannot be None")

    intervals = _maybe_load_intervals_set(intervals)
    # A position covered by several intervals is counted once. Touching intervals stay apart:
    # a k-mer that crosses from one into the next is not counted, as before.
    intervals = _unify_overlaps(intervals)

    # Extract sequences for the intervals
    seqs = gseq_extract(intervals)
    _check_intervals_strand(mask)

    # A k-mer is left out when any of its bases is masked, as in R: masked
    # positions per interval, from the mask intervals of its chromosome.
    masked_pos: list[_numpy.ndarray | None] = [None] * len(seqs)
    if mask is not None:
        from .intervals import _normalize_chroms

        assert isinstance(intervals, _pandas.DataFrame)  # loaded above

        m = _pandas.DataFrame({
            "chrom": _normalize_chroms(mask["chrom"].tolist()),
            "start": mask["start"].to_numpy(dtype=_numpy.int64),
            "end": mask["end"].to_numpy(dtype=_numpy.int64),
        }).sort_values("start", kind="mergesort")
        # Per chromosome: the mask starts, sorted, and the furthest end reached
        # by any mask starting at or before each of them.
        by_chrom = {
            c: (g["start"].to_numpy(), _numpy.maximum.accumulate(g["end"].to_numpy()))
            for c, g in m.groupby("chrom", sort=False)
        }
        iv_chrom = _normalize_chroms(intervals["chrom"].tolist())
        iv_start = intervals["start"].to_numpy(dtype=_numpy.int64)
        for i, seq in enumerate(seqs):
            if iv_chrom[i] not in by_chrom:
                continue
            ms, me = by_chrom[iv_chrom[i]]
            pos = iv_start[i] + _numpy.arange(len(seq), dtype=_numpy.int64)
            j = _numpy.searchsorted(ms, pos, side="right") - 1
            masked_pos[i] = (j >= 0) & (pos < me[_numpy.maximum(j, 0)])

    num_kmers = 4**k
    counts = _numpy.zeros(num_kmers, dtype=_numpy.int64)

    # Pre-compute powers of 4 for the rolling hash
    powers = _numpy.int64(4) ** _numpy.arange(k - 1, -1, -1, dtype=_numpy.int64)

    for seq, masked in zip(seqs, masked_pos, strict=True):
        if not seq or len(seq) < k:
            continue

        # one byte per character, as gseq_extract returns them (Latin-1)
        raw = _numpy.frombuffer(seq.encode("latin-1"), dtype=_numpy.uint8)
        codes = _DNA_CODE_TABLE[raw]
        # a masked base breaks the runs as an N does
        valid = codes >= 0 if masked is None else (codes >= 0) & ~masked
        if not valid.any():
            continue

        # Process only contiguous valid A/C/G/T runs to skip N/invalid chars.
        valid_i8 = valid.view(_numpy.int8)
        padded = _numpy.empty(len(valid_i8) + 2, dtype=_numpy.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = valid_i8
        transitions = _numpy.diff(padded)
        starts = _numpy.flatnonzero(transitions == 1)
        ends = _numpy.flatnonzero(transitions == -1)

        for start, end in zip(starts, ends, strict=False):
            segment = codes[start:end].astype(_numpy.int64, copy=False)
            seg_len = int(segment.size)
            n_windows = seg_len - k + 1
            if n_windows <= 0:
                continue

            # Rolling base-4 hash: use dot product with pre-computed powers.
            # For small k, this is faster than the multiply-accumulate loop.
            if k <= 8:
                # Sliding window via stride tricks
                strides = segment.strides
                window_view = _numpy.lib.stride_tricks.as_strided(
                    segment,
                    shape=(n_windows, k),
                    strides=(strides[0], strides[0]),
                )
                window_codes = window_view @ powers
            else:
                # For larger k, use the rolling multiply-accumulate
                window_codes = _numpy.zeros(n_windows, dtype=_numpy.int64)
                for j in range(k):
                    window_codes *= 4
                    window_codes += segment[j : j + n_windows]

            counts += _numpy.bincount(window_codes, minlength=num_kmers)

    nonzero_idx = _numpy.flatnonzero(counts)
    if nonzero_idx.size == 0:
        return _pandas.DataFrame({"kmer": _pandas.Series(dtype=str), "count": _pandas.Series(dtype=int)})

    # Use cached k-mer string table for fast decoding
    all_kmers = _kmer_strings(k)
    kmers = all_kmers[nonzero_idx]

    return _pandas.DataFrame(
        {"kmer": kmers, "count": counts[nonzero_idx].astype(_numpy.int64, copy=False)},
        columns=["kmer", "count"],
    )


# ---------------------------------------------------------------------------
# PWM scoring helpers
# ---------------------------------------------------------------------------

# Base encoding for vectorized PWM scoring
_PWM_FWD_CODE = _numpy.full(256, -1, dtype=_numpy.int8)
_PWM_RC_CODE = _numpy.full(256, -1, dtype=_numpy.int8)
for _b, _f, _r in ((ord("A"), 0, 3), (ord("C"), 1, 2), (ord("G"), 2, 1), (ord("T"), 3, 0)):
    _PWM_FWD_CODE[_b] = _f
    _PWM_FWD_CODE[_b + 32] = _f  # lowercase
    _PWM_RC_CODE[_b] = _r
    _PWM_RC_CODE[_b + 32] = _r


def _numpy_logsumexp(arr: _numpy.ndarray) -> float:
    """Numerically stable logsumexp over a 1-D array (finite values only)."""
    finite = arr[_numpy.isfinite(arr)]
    if finite.size == 0:
        return -_numpy.inf
    m = finite.max()
    return float(m + _numpy.log(_numpy.sum(_numpy.exp(finite - m))))


def _check_max_edits(max_edits: object) -> None:
    """R's check (gseq.pwm_edits and the pwm edit-distance vtracks)."""
    if max_edits is None:
        return
    if (isinstance(max_edits, bool) or not isinstance(max_edits, (int, float, _numpy.integer, _numpy.floating))
            or max_edits < 1 or max_edits != int(max_edits)):
        raise ValueError("max_edits must be NULL or a positive integer")


def _pos_strand_frame(pos: object, strand: object) -> _pandas.DataFrame:
    """gseq_pwm's (pos, strand) result; the strand is NA where pos is, as in R."""
    p = _numpy.asarray(pos, dtype=float)
    s = _pandas.Series(_numpy.asarray(strand, dtype=_numpy.int64), dtype="Int64")
    s[_numpy.isnan(p)] = _pandas.NA
    return _pandas.DataFrame({"pos": p, "strand": s})


def _pwm_score_batch_vectorized(
    seqs: list[str],
    log_pssm: _numpy.ndarray,
    avg_log: _numpy.ndarray,
    w: int,
    mode: str,
    scan_fwd: bool,
    scan_rev: bool,
    score_thresh: float,
    return_strand: bool,
    spat_log_factors: _numpy.ndarray | None = None,
    spat_bin: int = 1,
    neutral_policy: str = "average",
    neutral_set: frozenset[str] | None = None,
) -> _numpy.ndarray | tuple[_numpy.ndarray, _numpy.ndarray]:
    """Vectorized PWM scoring for a batch of sequences (no ROI/gaps).

    Replaces the per-window Python loop with numpy stride-tricks-based
    matrix indexing for ~10-50x speedup on typical workloads.
    """
    n = len(seqs)
    results = _numpy.full(n, _numpy.nan, dtype=float)
    strand_results = _numpy.zeros(n, dtype=_numpy.int64) if return_strand else None
    use_spat = spat_log_factors is not None and len(spat_log_factors) > 0
    n_spat = len(spat_log_factors) if spat_log_factors is not None else 0

    # Pre-compute reversed log_pssm for RC scoring
    # RC scoring at position j uses pssm[w-1-j, comp_code]
    # which equals reversed_pssm[j, comp_code]
    log_pssm_rev = log_pssm[::-1]  # reversed row order

    for i, seq in enumerate(seqs):
        seq_upper = _seq_upper(seq)
        L = len(seq_upper)
        if w > L:
            if mode == "count":
                results[i] = 0
            continue

        # Encode sequence to base codes
        raw = _numpy.frombuffer(seq_upper.encode("latin-1"), dtype=_numpy.uint8)
        fwd_codes = _PWM_FWD_CODE[raw]  # A=0,C=1,G=2,T=3, other=-1
        rc_codes = _PWM_RC_CODE[raw]  # A=3,C=2,G=1,T=0, other=-1

        # A code of -1 says "not ACGT"; it does NOT say "neutral". R charges
        # the neutral rate only for characters the caller listed and returns
        # -Inf for anything else, so a window holding an unlisted character
        # must score -Inf rather than the column mean. Without this the two
        # packages disagreed on a plain "X": -9.77 here against R's -Inf.
        if neutral_set is None:
            bad_byte = _numpy.zeros(256, dtype=bool)   # every non-ACGT is neutral
        else:
            bad_byte = _numpy.ones(256, dtype=bool)
            bad_byte[_numpy.frombuffer(b"ACGTacgt", dtype=_numpy.uint8)] = False
            for ch in neutral_set:  # upper-cased as the sequence (_seq_upper)
                bb = ch.encode("latin-1", "ignore")
                if len(bb) == 1:
                    bad_byte[bb[0]] = False
        is_bad = bad_byte[raw]

        n_windows = L - w + 1

        # Create sliding window views of codes: shape (n_windows, w)
        fwd_windows = _numpy.lib.stride_tricks.as_strided(
            fwd_codes,
            shape=(n_windows, w),
            strides=(fwd_codes.strides[0], fwd_codes.strides[0]),
        )
        rc_windows = _numpy.lib.stride_tricks.as_strided(
            rc_codes,
            shape=(n_windows, w),
            strides=(rc_codes.strides[0], rc_codes.strides[0]),
        )

        # Position indices for PSSM columns: [0, 1, ..., w-1]
        pos_idx = _numpy.arange(w)

        all_scores = []

        # Windows holding a character that is not ACGT and not listed neutral
        bad_windows = (
            _numpy.lib.stride_tricks.as_strided(
                is_bad, shape=(n_windows, w),
                strides=(is_bad.strides[0], is_bad.strides[0]),
            ).any(axis=1)
            if is_bad.any()
            else _numpy.zeros(n_windows, dtype=bool)
        )

        if scan_fwd:
            # Check for bases with code == -1 (neutral chars like N, *)
            has_unknown = _numpy.any(fwd_windows == -1, axis=1) & ~bad_windows

            # Score fully-valid windows via vectorized fancy indexing
            fwd_scores = _numpy.full(n_windows, -_numpy.inf)
            # A bad window keeps the -inf it was filled with. It must not fall
            # into `valid`: a -1 code there would index log_pssm's LAST column
            # and score a confident wrong number instead of -inf.
            valid = ~has_unknown & ~bad_windows
            if valid.any():
                valid_wins = fwd_windows[valid]
                fwd_scores[valid] = log_pssm[pos_idx, valid_wins].sum(axis=1)

            # Handle windows with neutral chars (code -1) using avg_log
            # In the fast path (no gaps), -1 codes come from neutral chars
            if has_unknown.any() and neutral_policy == "average":
                for wi in _numpy.flatnonzero(has_unknown):
                    win = fwd_windows[wi]
                    s = 0.0
                    for j in range(w):
                        if win[j] == -1:
                            s += avg_log[j]
                        else:
                            s += log_pssm[j, win[j]]
                    fwd_scores[wi] = s
            elif has_unknown.any() and neutral_policy == "log_quarter":
                for wi in _numpy.flatnonzero(has_unknown):
                    win = fwd_windows[wi]
                    s = 0.0
                    for j in range(w):
                        if win[j] == -1:
                            s += _LOG_QUARTER
                        else:
                            s += log_pssm[j, win[j]]
                    fwd_scores[wi] = s

            if use_spat:
                assert spat_log_factors is not None
                offsets = _numpy.arange(n_windows)
                bins = _numpy.clip(offsets // spat_bin, 0, n_spat - 1)
                spat = spat_log_factors[bins]
                finite_mask = _numpy.isfinite(fwd_scores)
                fwd_scores[finite_mask] += spat[finite_mask]

            all_scores.append((_numpy.ones(n_windows, dtype=_numpy.int8), fwd_scores))

        if scan_rev:
            has_unknown_rc = _numpy.any(rc_windows == -1, axis=1) & ~bad_windows
            rc_scores = _numpy.full(n_windows, -_numpy.inf)
            valid_rc = ~has_unknown_rc & ~bad_windows
            if valid_rc.any():
                valid_wins_rc = rc_windows[valid_rc]
                rc_scores[valid_rc] = log_pssm_rev[pos_idx, valid_wins_rc].sum(axis=1)

            if has_unknown_rc.any() and neutral_policy == "average":
                for wi in _numpy.flatnonzero(has_unknown_rc):
                    win = rc_windows[wi]
                    s = 0.0
                    for j in range(w):
                        if win[j] == -1:
                            s += avg_log[w - 1 - j]
                        else:
                            s += log_pssm_rev[j, win[j]]
                    rc_scores[wi] = s
            elif has_unknown_rc.any() and neutral_policy == "log_quarter":
                for wi in _numpy.flatnonzero(has_unknown_rc):
                    win = rc_windows[wi]
                    s = 0.0
                    for j in range(w):
                        if win[j] == -1:
                            s += _LOG_QUARTER
                        else:
                            s += log_pssm_rev[j, win[j]]
                    rc_scores[wi] = s

            if use_spat:
                assert spat_log_factors is not None
                offsets = _numpy.arange(n_windows)
                bins = _numpy.clip(offsets // spat_bin, 0, n_spat - 1)
                spat = spat_log_factors[bins]
                finite_mask_rc = _numpy.isfinite(rc_scores)
                rc_scores[finite_mask_rc] += spat[finite_mask_rc]

            all_scores.append((-_numpy.ones(n_windows, dtype=_numpy.int8), rc_scores))

        if not all_scores:
            if mode == "count":
                results[i] = 0
            continue

        # Under bidirect the forward and reverse-complement matches are two
        # readings of the SAME window, so every mode but "pos" combines them
        # per window by log-sum-exp before reducing over windows - the union
        # the pwm, pwm.max and pwm.count virtual tracks take, and the one the
        # C++ fast path above already takes. "pos" still maximises across the
        # strands, because it has to name one; that asymmetry is the pwm
        # family's own (pwm.max combines, pwm.max.pos does not).
        union = (_numpy.logaddexp(all_scores[0][1], all_scores[1][1])
                 if len(all_scores) == 2 else all_scores[0][1])

        # Aggregate scores by mode
        if mode == "lse":
            results[i] = _numpy_logsumexp(union)

        elif mode == "max":
            non_nan = union[~_numpy.isnan(union)]
            if non_nan.size > 0:
                results[i] = non_nan.max()

        elif mode == "pos":
            # R's gseq.pwm (GseqString.cpp): the higher score, then the
            # earlier start, then the forward strand.
            best = None
            for dirs, scores in all_scores:
                finite_mask = _numpy.isfinite(scores)
                if not finite_mask.any():
                    continue
                idx = int(_numpy.argmax(scores))  # earliest best window on this strand
                key = (-scores[idx], idx, -int(dirs[0]))
                if best is None or key < best:
                    best = key
            best_pos = _numpy.nan if best is None else float(best[1] + 1)  # 1-based
            best_strand = 0 if best is None else -best[2]
            results[i] = best_pos
            if strand_results is not None:
                strand_results[i] = best_strand

        elif mode == "count":
            # One window is at most one hit: the strands are combined before
            # the threshold, not thresholded apart, so a window whose reverse
            # complement also matches is not counted twice.
            results[i] = float(_numpy.sum(union >= score_thresh))

    if return_strand and mode == "pos":
        assert strand_results is not None
        return results, strand_results
    return results


def _pwm_log_sum_exp(scores: list[float]) -> float:
    """Numerically stable log-sum-exp matching C++/R iterative algorithm."""
    finite = [s for s in scores if _numpy.isfinite(s)]
    if not finite:
        return -_numpy.inf
    if len(finite) == 1:
        return finite[0]
    finite.sort(reverse=True)
    s = finite[0]
    for i in range(1, len(finite)):
        if s > finite[i]:
            s = s + _math.log1p(_math.exp(finite[i] - s))
        else:
            s = finite[i] + _math.log1p(_math.exp(s - finite[i]))
    return s


def _pwm_score_window(
    bases: str,
    log_pssm: _numpy.ndarray,
    avg_log: _numpy.ndarray,
    w: int,
    neutral_set: frozenset[str],
    neutral_policy: str,
    reverse: bool,
) -> float:
    """Score a single w-base window against the PSSM.

    Returns float score, -inf for unknown bases, or NaN for neutral+na policy.
    """
    score = 0.0
    for j in range(w):
        base = bases[j]
        if base in neutral_set:
            pssm_idx = (w - 1 - j) if reverse else j
            if neutral_policy == "average":
                score += avg_log[pssm_idx]
            elif neutral_policy == "log_quarter":
                score += _LOG_QUARTER
            else:  # "na"
                return float("nan")
        else:
            if reverse:
                code = _PWM_COMP_CODE.get(base, -1)
                pssm_idx = w - 1 - j
            else:
                code = _PWM_BASE_CODE.get(base, -1)
                pssm_idx = j
            if code < 0:
                return -_numpy.inf
            score += log_pssm[pssm_idx, code]
    return score


def _pwm_iter_windows(
    seq_upper: str, start_min: int, start_max: int, w: int, skip_gaps: bool, gap_set: frozenset[str]
) -> collections.abc.Generator[tuple[int, str], None, None]:
    """Yield (physical_pos_0based, window_str) for PWM scanning."""
    if skip_gaps and gap_set:
        # Build compacted sequence and logical-to-physical mapping
        comp = []
        log_to_phys = []
        for i in range(len(seq_upper)):
            if seq_upper[i] not in gap_set:
                comp.append(seq_upper[i])
                log_to_phys.append(i)
        n_comp = len(comp)
        phys_end_bound = start_max + w - 1
        for log_i in range(n_comp - w + 1):
            phys_start = log_to_phys[log_i]
            if phys_start < start_min:
                continue
            if phys_start > start_max:
                break  # monotonically increasing
            phys_end = log_to_phys[log_i + w - 1]
            if phys_end > phys_end_bound:
                continue
            yield phys_start, "".join(comp[log_i : log_i + w])
    else:
        for pos in range(start_min, start_max + 1):
            yield pos, seq_upper[pos : pos + w]


def _pwm_score_sequence(
    seq_upper: str,
    L: int,
    log_pssm: _numpy.ndarray,
    avg_log: _numpy.ndarray,
    w: int,
    mode: str,
    scan_fwd: bool,
    scan_rev: bool,
    score_thresh: float,
    start_pos: int | None,
    end_pos: int | None,
    extend: bool | int,
    skip_gaps: bool,
    gap_set: frozenset[str],
    neutral_set: frozenset[str],
    neutral_policy: str,
    return_strand: bool,
    spat_log_factors: _numpy.ndarray | None = None,
    spat_bin: int = 1,
    spat_min: int | None = None,
    spat_max: int | None = None,
) -> float | int | tuple[float, int]:
    """Score a single uppercased sequence with PWM.

    Parameters
    ----------
    spat_log_factors : numpy.ndarray or None
        Pre-computed log(spat_factor) per spatial bin.  ``None`` means
        no spatial weighting.
    spat_bin : int
        Bin size for spatial weighting (positions per bin).
    spat_min, spat_max : int or None
        1-based window counted from the ROI start; as R's C_gseq_pwm.
    """
    # ROI (1-based inclusive)
    roi_start = start_pos if start_pos is not None else 1
    roi_end = end_pos if end_pos is not None else L

    # Extension
    if extend is True:
        E = w - 1
    elif extend is False or extend is None:
        E = 0
    else:
        E = int(extend)

    # 0-based allowed window start range
    start_min = max(0, roi_start - 1 - E)
    start_max = min(L - w, roi_end - w + E)
    bin_origin = roi_start
    if spat_min is not None:
        start_min = max(start_min, roi_start + spat_min - 2)
        bin_origin = roi_start + spat_min - 1
    if spat_max is not None:
        start_max = min(start_max, roi_start + spat_max - 1 - w)

    if start_max < start_min or w > L:
        if mode == "count":
            return 0
        if return_strand and mode == "pos":
            return (float("nan"), 0)
        return float("nan")

    use_spat = spat_log_factors is not None and len(spat_log_factors) > 0
    n_spat = len(spat_log_factors) if spat_log_factors is not None else 0

    # Iterate windows and aggregate
    all_scores = []  # for lse
    best_score = -_numpy.inf
    best_pos = -1
    best_strand = 0
    count = 0
    has_valid = False
    na_seq = False

    for phys_pos, window in _pwm_iter_windows(seq_upper, start_min, start_max, w, skip_gaps, gap_set):
        # Spatial log factor for this window position.
        # offset = (phys_pos + 1) - bin_origin, as C++ (s0 + 1) - roi_start1, where
        # roi_start1 is the spat window start when spat_min is given
        if use_spat:
            assert spat_log_factors is not None
            offset = (phys_pos + 1) - bin_origin
            sb = offset // spat_bin
            if sb < 0:
                sb = 0
            elif sb >= n_spat:
                sb = n_spat - 1
            spat_log = float(spat_log_factors[sb])
        else:
            spat_log = 0.0

        win_scores: list[float] = []
        win_dirs: list[int] = []
        seq_is_na = False
        for is_rev, strand_dir in ((False, 1), (True, -1)):
            if is_rev and not scan_rev:
                continue
            if not is_rev and not scan_fwd:
                continue

            s = _pwm_score_window(window, log_pssm, avg_log, w, neutral_set, neutral_policy, is_rev)
            if _numpy.isnan(s):
                # neutral_chars_policy = "na" means the SEQUENCE is unanswerable,
                # not that this window is skipped: R returns NA for the whole
                # sequence as soon as one evaluated window is NA, and both
                # packages document the policy that way. Dropping the window
                # instead reduced over the rest and returned a confident number
                # from the windows that happened to be clean.
                seq_is_na = True
                break

            # Apply spatial weight in log space
            if use_spat and s > -_numpy.inf:
                s += spat_log

            win_scores.append(s)
            win_dirs.append(strand_dir)

        if seq_is_na:
            na_seq = True
            break
        if not win_scores:
            continue
        has_valid = True

        if mode == "lse":
            # log-sum-exp is associative, so appending both strands is the
            # same number as appending their union.
            all_scores.extend(win_scores)
        elif mode == "pos":
            # "pos" has to name a strand, so it maximises across the two
            # rather than combining them - what pwm.max.pos does.
            for s, strand_dir in zip(win_scores, win_dirs, strict=False):
                if s > best_score:
                    best_score = s
                    best_pos = phys_pos + 1  # 1-based
                    best_strand = strand_dir
        else:
            # "max" and "count": the two strands are two readings of the same
            # window, so they are combined by log-sum-exp before the reduction
            # over windows - the union the pwm virtual tracks take. A window
            # whose reverse complement also matches is one hit, not two.
            u = (win_scores[0] if len(win_scores) == 1
                 else float(_numpy.logaddexp(win_scores[0], win_scores[1])))
            if mode == "max":
                if u > best_score:
                    best_score = u
            elif u >= score_thresh:
                count += 1

    # Aggregate final result
    if na_seq:
        # neutral_chars_policy = "na": one unanswerable window makes the whole
        # sequence unanswerable, in every mode. `count` included - R returns NA
        # there too, rather than a count of the windows that happened to be
        # clean.
        if mode == "pos" and return_strand:
            return (float("nan"), 0)
        return float("nan")

    if mode == "lse":
        if not has_valid:
            return float("nan")
        return _pwm_log_sum_exp(all_scores)
    if mode == "max":
        if not has_valid:
            return float("nan")
        return best_score
    if mode == "pos":
        if not has_valid or best_pos < 0:
            if return_strand:
                return (float("nan"), 0)
            return float("nan")
        if return_strand:
            return (float(best_pos), best_strand)
        return float(best_pos)
    # count
    return count


def gseq_pwm(
    seqs: str | list[str],
    pssm: _numpy.ndarray | _pandas.DataFrame,
    mode: str = "lse",
    bidirect: bool = True,
    strand: int = 1,
    score_thresh: float | None = None,
    start_pos: int | None = None,
    end_pos: int | None = None,
    extend: bool | int = False,
    spat_factor: _numpy.ndarray | list[float] | None = None,
    spat_bin: int = 1,
    spat_min: float | None = None,
    spat_max: float | None = None,
    return_strand: bool = False,
    skip_gaps: bool = True,
    gap_chars: list[str] | None = None,
    neutral_chars: list[str] | None = None,
    neutral_chars_policy: str = "average",
    prior: float = 0.01,
) -> _numpy.ndarray | _pandas.DataFrame:
    """
    Score DNA sequences with a position weight matrix (PWM/PSSM).

    Scans each input sequence with a sliding window of PSSM width and
    computes per-position log-probability scores, then aggregates them
    according to *mode*.

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) to score.
    pssm : numpy.ndarray or pandas.DataFrame
        Position-specific scoring matrix. If ndarray, shape ``(w, 4)`` with
        columns ordered [A, C, G, T]. If DataFrame, must contain columns
        ``A``, ``C``, ``G``, ``T`` (extra columns are ignored).
    mode : str, default ``"lse"``
        Scoring mode:

        - ``"lse"``: log-sum-exp of all per-position scores.
        - ``"max"``: maximum per-position score.
        - ``"pos"``: 1-based position of best match.
        - ``"count"``: number of positions with score >= *score_thresh*.
    bidirect : bool, default True
        If True, score both forward and reverse complement strands.
        Overrides *strand*.
    strand : int, default 1
        Used only when ``bidirect=False``, and required there: ``1`` for the
        forward strand, ``-1`` for the reverse complement. ``0`` is refused,
        since exactly one strand is read and ``0`` cannot name it. Under
        ``bidirect=True`` both strands are read and this is ignored.
    score_thresh : float, optional
        Threshold for ``"count"`` mode, where it is required: PWM scores are
        log-likelihoods, so no default suits every PSSM. Ignored otherwise.
    start_pos : int or None
        1-based inclusive start of region of interest.
    end_pos : int or None
        1-based inclusive end of region of interest.
    extend : bool or int, default False
        Allow motif window to start before ROI.  ``True`` = ``w-1``,
        ``False`` = 0, or an explicit integer extension.
    spat_factor : array-like or None
        Spatial weighting factors, one per spatial bin.  When provided,
        the score at each window position is shifted in log-space by
        ``log(spat_factor[bin])`` where ``bin = offset // spat_bin``
        and *offset* is the 0-based position of the window start
        relative to the ROI start.  Values must be non-negative.
    spat_bin : int, default 1
        Number of consecutive positions that share the same spatial
        weight.  Position *offset* maps to bin ``offset // spat_bin``.
    spat_min : float or None
        1-based position, counted from ``start_pos``, where the scanning
        window starts (>= 1); spatial bins are counted from it.
    spat_max : float or None
        1-based position, counted from ``start_pos``, where the scanning
        window ends: only windows ending at or before it are scored (>= 1).
        A ``spat_min``..``spat_max`` window shorter than the PSSM is an error.
    return_strand : bool, default False
        For ``mode="pos"`` with bidirectional scoring, return a DataFrame
        with ``pos`` and ``strand`` columns instead of a plain array.
    skip_gaps : bool, default True
        Skip gap characters when scanning.
    gap_chars : list of str or None
        Characters treated as gaps. Default ``["-", "."]``.
    neutral_chars : list of str or None
        Characters to score as unknown bases, by ``neutral_chars_policy``,
        instead of leaving their windows out. Default: none, as in R misha
        5.15, so a window holding N or any other non-ACGT base is not scored
        (an interval with no scorable window gives -inf, or NaN for
        ``mode="pos"``). Pass ``["N", "n", "*"]`` for the scores of earlier
        versions.
    neutral_chars_policy : str, default ``"average"``
        How to score neutral characters:

        - ``"average"``: mean log-probability of the PSSM column.
        - ``"log_quarter"``: ``log(0.25)``.
        - ``"na"``: window is invalid (NaN).
    prior : float, default 0.01
        Pseudocount added to PSSM before normalization, between 0 and 1.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Array of scores (one per input sequence). For ``mode="pos"`` with
        ``return_strand=True``, returns a DataFrame with columns ``pos``
        and ``strand``.

    Examples
    --------
    >>> import pymisha as pm
    >>> import numpy as np

    Create a simple PSSM (frequency matrix):

    >>> pssm = np.array([
    ...     [0.7, 0.1, 0.1, 0.1],
    ...     [0.1, 0.7, 0.1, 0.1],
    ...     [0.1, 0.1, 0.7, 0.1],
    ...     [0.1, 0.1, 0.1, 0.7],
    ... ])

    Score sequences using log-sum-exp (default mode):

    >>> pm.gseq_pwm(["ACGTACGT", "GGGGGGGG"], pssm, mode="lse")
    array([...])

    Find position of best match:

    >>> pm.gseq_pwm(["ACGTACGT"], pssm, mode="pos")
    array([...])

    Count matches above a threshold:

    >>> pm.gseq_pwm(["ACGTACGT"], pssm, mode="count", score_thresh=-5.0)
    array([...])

    See Also
    --------
    gseq_kmer, gseq_kmer_dist, gseq_extract
    """
    # --- Validation ---
    if mode not in ("lse", "max", "pos", "count"):
        raise ValueError(f"mode must be 'lse', 'max', 'pos', or 'count', got '{mode}'")
    if mode == "count":
        if score_thresh is None:
            raise ValueError(
                "gseq_pwm(mode='count') requires a 'score_thresh' argument. PWM "
                "scores are log-likelihoods, so there is no default that suits "
                "every PSSM - pick a threshold from the score distribution of your "
                "own matrix, e.g. with mode='max'."
            )
        score_thresh = _coerce_score_thresh(score_thresh)
    elif score_thresh is None:
        score_thresh = 0.0
    if _na_or_seq(strand) or isinstance(strand, str) or strand not in (-1, 0, 1):
        raise ValueError(f"strand must be -1, 0, or 1, got {strand}")
    strand = int(strand)
    # R reads bidirect as a logical: TRUE, FALSE or a number; not NA or a string
    if _na_or_seq(bidirect) or not isinstance(bidirect, (bool, _numpy.bool_, int, float, _numpy.number)):
        raise ValueError("bidirect must be True or False")
    bidirect = bool(bidirect)
    skip_gaps = _skip_gaps_arg(skip_gaps)
    if (isinstance(prior, (bool, _numpy.bool_)) or not isinstance(prior, (int, float, _numpy.number))
            or not 0 <= prior <= 1):
        raise ValueError("prior must be a number between 0 and 1")
    if neutral_chars_policy not in ("average", "log_quarter", "na"):
        raise ValueError("neutral_chars_policy must be 'average', 'log_quarter', or 'na'")
    # An NA extend is an error, as in R (it used to read as TRUE there).
    extend = _seq_extend_arg(extend)
    pssm = _coerce_pssm_matrix(pssm)

    # --- Single-string convenience ---
    if isinstance(seqs, str):
        seqs = [seqs]
    # one character per byte, as R reads a string (Latin-1)
    seqs = [_seq_bytes(x) if isinstance(x, str) else x for x in seqs]

    # spat_min / spat_max: 1-based positions counted from start_pos, as in the
    # pwm virtual tracks with the ROI in the interval's place. Windows must lie
    # inside them and spatial bins count from spat_min. R truncates them with
    # as.integer() after the same checks.
    if spat_min is not None or spat_max is not None:
        smin_f, smax_f = _check_spat_bounds(spat_min, spat_max, len(pssm))
        spat_min = None if smin_f is None else int(smin_f)
        spat_max = None if smax_f is None else int(smax_f)

    # --- C++ fast path ---
    # Dispatch to the C++ PWMScorer when no Python-only features are needed.
    # The C++ path handles prior/log conversion internally and is significantly
    # faster than both the vectorized numpy path and the per-sequence Python path.
    # The C++ scorer follows no neutral-base policy exactly (see the N check
    # below), so it only takes sequences without one, where every policy gives
    # the same answer. Only "log_quarter" is routed to it, as before: the
    # default stays on the Python paths, which compute in double as R's
    # gseq.pwm does, where the C++ scorer sums in float.
    # The minus strand alone stays in Python too: the C++ scorer reverse
    # complements the string and so counts spatial bins from its 3' end and
    # breaks "pos" ties from that end, where R's gseq.pwm counts both from the
    # 5' end (up to 0.2 nats and 117 bp apart on a 200-mer).
    _use_cpp = (
        gap_chars is None
        and neutral_chars is None
        and neutral_chars_policy == "log_quarter"
        and mode != "pos"  # its tie rule is the pwm.max.pos vtrack's, not gseq.pwm's
        and not extend
        and (bidirect or strand != -1)
    )
    # The C++ scorer doesn't handle gap characters; fall back to Python if
    # any sequence contains default gap chars ("-", "."). Nor does it charge a
    # neutral base (N, n, *) log(1/4) on both strands: its lse charges the
    # reverse strand the column mean and its max both strands, so any sequence
    # holding one goes to Python too (0.04 nats out on a 10-mer with two Ns).
    if _use_cpp:
        _py_chars = frozenset("Nn*-." if skip_gaps else "Nn*")
        for seq in seqs:
            if _py_chars.intersection(seq):
                _use_cpp = False
                break
    if _use_cpp:
        # Extract raw PSSM as (W, 4) numpy array — C++ applies prior itself.
        if isinstance(pssm, _pandas.DataFrame):
            for col in ("A", "C", "G", "T"):
                if col not in pssm.columns:
                    raise KeyError(f"PSSM DataFrame missing column '{col}'")
            cpp_pssm = pssm[["A", "C", "G", "T"]].values.astype(float)
        elif isinstance(pssm, _numpy.ndarray):
            if pssm.ndim != 2 or pssm.shape[1] != 4:
                raise ValueError("PSSM array must have shape (w, 4)")
            cpp_pssm = _numpy.ascontiguousarray(pssm, dtype=float)
        else:
            raise ValueError("PSSM must be a numpy array or pandas DataFrame")

        # Validate spatial factor early (same checks as Python path).
        cpp_spat: _numpy.ndarray | None = None
        if spat_factor is not None:
            cpp_spat = _numpy.asarray(spat_factor, dtype=float)
            if cpp_spat.ndim != 1 or cpp_spat.size == 0:
                raise ValueError("spat_factor must be a non-empty 1-D array")
            if _numpy.any(cpp_spat < 0):
                raise ValueError("spat_factor values must be non-negative")

        # Apply ROI slicing in Python before passing to C++.
        cpp_seqs = list(seqs)
        _roi_offset = 0
        if start_pos is not None or end_pos is not None or spat_min is not None or spat_max is not None:
            s = (start_pos - 1) if start_pos is not None else 0
            e = end_pos if end_pos is not None else None
            # The spat window, counted from the ROI start, narrows the slice;
            # its start is also where the spatial bins begin.
            if spat_max is not None:
                e = s + spat_max if e is None else min(e, s + spat_max)
            if spat_min is not None:
                s += spat_min - 1
            cpp_seqs = [seq[s:e] for seq in cpp_seqs]
            # The slice is invisible to C++, so a position it reports counts
            # from the slice. Positions are documented - in both packages - as
            # being on the FULL input sequence, so the offset goes back on
            # below. max/lse/count are unaffected: the window set is the same.
            _roi_offset = s

        # Map strand for C++: in Python, strand=0 with bidirect=False
        # means "forward only"; the C++ scorer treats strand=0 as
        # "no strand" (neither fwd nor rev), so remap to 1.
        cpp_strand = int(strand)
        if not bidirect and cpp_strand == 0:
            cpp_strand = 1

        try:
            _res = _pymisha.pm_pwm_score_strings(
                cpp_seqs,
                cpp_pssm,
                float(prior),
                mode,
                int(bidirect),
                cpp_strand,
                float(score_thresh),
                0,  # extend=0 (extend cases already excluded above)
                cpp_spat,
                int(spat_bin),
            )
            if mode == "count":
                # score_string() answers NaN for a string shorter than the
                # motif; R counts no window there as 0, as the Python paths do.
                _res = _numpy.nan_to_num(_numpy.asarray(_res, dtype=float), nan=0.0)
            if mode == "pos" and _roi_offset:
                # Back onto the full sequence's coordinates; a NaN (nothing
                # scorable) stays NaN rather than becoming the offset.
                _res = _numpy.asarray(_res, dtype=float)
                _res = _numpy.where(_numpy.isnan(_res), _res, _res + _roi_offset)
            return _res
        except AttributeError:
            # _pymisha not built with pm_pwm_score_strings — fall through.
            pass

    # --- Spatial weighting ---
    spat_log_factors = None
    if spat_factor is not None:
        spat_arr = _numpy.asarray(spat_factor, dtype=float)
        if spat_arr.ndim != 1 or spat_arr.size == 0:
            raise ValueError("spat_factor must be a non-empty 1-D array")
        if _numpy.any(spat_arr < 0):
            raise ValueError("spat_factor values must be non-negative")
        # Clamp zeros to tiny positive value before taking log (matches C++)
        spat_arr = _numpy.maximum(spat_arr, 1e-30)
        spat_log_factors = _numpy.log(spat_arr)
        spat_bin = max(1, int(spat_bin))

    # --- Normalize PSSM ---
    if isinstance(pssm, _pandas.DataFrame):
        for col in ("A", "C", "G", "T"):
            if col not in pssm.columns:
                raise KeyError(f"PSSM DataFrame missing column '{col}'")
        pssm_arr = pssm[["A", "C", "G", "T"]].values.astype(float)
    elif isinstance(pssm, _numpy.ndarray):
        if pssm.ndim != 2 or pssm.shape[1] != 4:
            raise ValueError("PSSM array must have shape (w, 4)")
        pssm_arr = pssm.astype(float).copy()
    else:
        raise ValueError("PSSM must be a numpy array or pandas DataFrame")

    w = pssm_arr.shape[0]

    # Normalise FIRST, then add the prior, then re-normalise. R's DnaProbVec
    # constructor normalises before the prior is applied and pymisha's own C++
    # path does the same; adding the prior to raw values instead makes it mean
    # something different for every matrix scale. On a JASPAR-style count
    # matrix - the ordinary way a motif arrives - the old order was 0.07 nats
    # out at prior = 0.01 and far worse at prior = 1.
    row_sums = pssm_arr.sum(axis=1, keepdims=True)
    if _numpy.any(row_sums <= 0):
        raise ValueError("Each PSSM row must have a positive total weight")
    pssm_arr = pssm_arr / row_sums
    if prior > 0:
        pssm_arr = pssm_arr + prior
        pssm_arr = pssm_arr / pssm_arr.sum(axis=1, keepdims=True)

    # Log-probabilities
    log_pssm = _numpy.full_like(pssm_arr, -_numpy.inf)
    pos_mask = pssm_arr > 0
    log_pssm[pos_mask] = _numpy.log(pssm_arr[pos_mask])

    # Per-position average log-prob (for neutral "average" policy)
    avg_log = log_pssm.mean(axis=1)

    # --- Strand config ---
    if bidirect:
        scan_fwd, scan_rev = True, True
    else:
        # Exactly one strand is read here, so strand = 0 cannot name it and is
        # refused - as gseq_potts() and the pwm/potts virtual tracks already do,
        # and as R's gseq.pwm() now does. It used to be the DEFAULT, and R
        # honoured it three different ways: forward for lse/pos/count, BOTH
        # (hard max) for max, and "0 = both strands" in the documentation. The
        # default is 1 now, so an ordinary gseq_pwm(bidirect=False) call keeps
        # working and reads forward.
        if strand == 0:
            raise ValueError(
                "gseq_pwm(bidirect=False) needs an explicit strand: "
                "1 for the forward strand, -1 for the reverse."
            )
        scan_fwd = strand == 1
        scan_rev = strand == -1

    # --- Defaults ---
    if gap_chars is None:
        gap_chars = ["-", "."]
    if neutral_chars is None:
        neutral_chars = []

    gap_set = frozenset(gap_chars)
    # Uppercase neutral chars so they match uppercased sequence
    neutral_set = frozenset(_seq_upper(_seq_bytes(c)) for c in neutral_chars)

    # --- Vectorized fast path ---
    # Use numpy-vectorized scoring when no ROI, no gaps, and neutral policy
    # is "average" (the common case).  Gap characters virtually never appear
    # in DNA sequences; we only fall back if they're actually present.
    has_roi = start_pos is not None or end_pos is not None
    if not has_roi and not extend and neutral_chars_policy != "na":
        need_slow = False
        if skip_gaps and gap_set:
            for seq in seqs:
                if gap_set.intersection(seq):
                    need_slow = True
                    break
        if not need_slow:
            # With no ROI, the spat window is a slice of each sequence, and
            # its start is where positions and spatial bins count from.
            lo = 0 if spat_min is None else spat_min - 1
            r = _pwm_score_batch_vectorized(
                seqs if lo == 0 and spat_max is None else [seq[lo:spat_max] for seq in seqs],
                log_pssm,
                avg_log,
                w,
                mode,
                scan_fwd,
                scan_rev,
                score_thresh,
                return_strand,
                spat_log_factors=spat_log_factors,
                spat_bin=spat_bin,
                neutral_policy=neutral_chars_policy,
                neutral_set=neutral_set,
            )
            if mode == "pos" and lo:   # NaN stays NaN
                r = (r[0] + lo, r[1]) if isinstance(r, tuple) else r + lo
            if return_strand and mode == "pos":
                return _pos_strand_frame(r[0], r[1])
            return r

    # --- Slow path: per-sequence scoring ---
    results: list[float | int | tuple[float, int]] = []
    strand_results: list[int] = []

    for seq in seqs:
        seq_upper = _seq_upper(seq)
        sr = _pwm_score_sequence(
            seq_upper,
            len(seq_upper),
            log_pssm,
            avg_log,
            w,
            mode,
            scan_fwd,
            scan_rev,
            score_thresh,
            start_pos,
            end_pos,
            extend,
            skip_gaps,
            gap_set,
            neutral_set,
            neutral_chars_policy,
            return_strand,
            spat_log_factors=spat_log_factors,
            spat_bin=spat_bin,
            spat_min=spat_min,
            spat_max=spat_max,
        )
        if return_strand and mode == "pos":
            assert isinstance(sr, tuple)
            results.append(sr[0])
            strand_results.append(sr[1])
        else:
            results.append(sr)

    if return_strand and mode == "pos":
        return _pos_strand_frame(results, strand_results)

    return _numpy.array(results, dtype=float)


def gseq_potts(
    seqs: str | list[str],
    model: dict,
    mode: str = "lse",
    bidirect: bool = True,
    strand: int = 0,
    score_thresh: float | None = None,
    extend: bool = False,
) -> _numpy.ndarray:
    """
    Score DNA sequences under a Potts (pairwise energy) model.

    The sequence-level entry point of the ``potts`` family, the way
    :func:`gseq_pwm` is the ``pwm`` family's. The score of one width-``W``
    window is ``intercept + sum_i e[i, x_i] + sum_k J_k[x_i, x_j]``, and a
    sequence longer than ``W`` is reduced over every window by *mode*.

    The parameterisation is not unique: adding a constant to a position's four
    ``e`` entries, or a row or column shift to a ``J`` block, changes no score,
    so ``e`` is comparable across positions only in the zero-sum gauge, where
    each row of ``e`` sums to 0 and every ``J`` block has zero row and column
    sums.

    A Potts carries energies, not probabilities, so there is no ``prior`` and
    no fallback for an ambiguous base: a window containing any non-ACGT base is
    not scored and is left out of the reduction. A sequence that has windows
    but none of them scorable returns ``nan``, and ``0`` for ``mode="count"``:
    it counted, and found none. A sequence shorter than the model returns
    ``nan`` for every mode including ``"count"`` - there was no window to count.

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) to score (case-insensitive).
    model : dict
        The energy model, with keys:

        - ``e`` -- a ``(W, 4)`` numeric array or DataFrame with columns
          ``A``, ``C``, ``G``, ``T``: the per-position energies.
        - ``J`` -- a list of ``4x4`` arrays, or an ``(npair, 16)`` array: the
          coupling tables, one per row of ``pairs``, indexed with the lower
          position on the rows.
        - ``pairs`` -- an ``(npair, 2)`` array of 1-based position pairs, lower
          position first.
        - ``intercept`` -- a single number.

        Extra keys named ``width``, ``pair_strength``, ``attr`` or ``link`` are
        accepted and ignored, so a fitted model can be passed verbatim;
        ``width`` is cross-checked against the row count of ``e``.
    mode : str, default ``"lse"``
        ``"lse"`` for the log-sum-exp over windows, ``"max"`` for the best
        window, ``"pos"`` for the 1-based position of the best window (signed
        by strand when ``bidirect=True``), ``"count"`` for the number of
        windows scoring at least *score_thresh*.
    bidirect : bool, default True
        If True both strands are read. The two strands are combined at each
        window by log-sum-exp for ``"lse"``, ``"max"`` and ``"count"``, and by
        the maximum for ``"pos"``, which has to name a strand. :func:`gseq_pwm`
        splits the two the same way.
    strand : int, default 0
        Used only when ``bidirect=False``, and required there: ``1`` for the
        forward strand, ``-1`` for the reverse. The default ``0`` is accepted
        only under ``bidirect=True``, where both strands are read and it is
        ignored.
    score_thresh : float, optional
        Required for ``mode="count"`` and ignored otherwise. A Potts score is
        an energy whose usable range depends entirely on the model, so there is
        no default; read one off ``mode="max"`` over your own sequences.
    extend : bool, default False
        Accepted for signature parity with :func:`gseq_pwm` and ignored - a
        bare sequence has no genome to extend into. True or False, as in R.

    Returns
    -------
    numpy.ndarray
        One value per input sequence.

    Examples
    --------
    >>> import numpy as np
    >>> import pymisha as pm

    A ``W=2`` model with a real coupling. On the per-position energies alone
    the model prefers ``"AC"``, but the pair term penalises A-then-C and
    rewards G-then-T, so ``"GT"`` wins instead:

    >>> e = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    >>> J = [np.array([[0, -3.0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3.0], [0, 0, 0, 0]])]
    >>> model = {"e": e, "J": J, "pairs": np.array([[1, 2]]), "intercept": 0.0}
    >>> pm.gseq_potts(["AC", "GT"], model, mode="max", bidirect=False, strand=1)
    array([-1.,  3.])

    See Also
    --------
    gseq_pwm : the same reduction for a position weight matrix.
    gvtrack_create : the ``potts`` virtual track functions.
    """
    if mode not in ("lse", "max", "pos", "count"):
        raise ValueError(f"mode must be 'lse', 'max', 'pos', or 'count', got '{mode}'")

    if mode == "count":
        if score_thresh is None:
            raise ValueError(
                "gseq_potts(mode='count') requires a 'score_thresh' argument. A "
                "Potts score is an energy whose usable range depends entirely on "
                "the model, so there is no default that suits every one - read a "
                "threshold off mode='max' over your own sequences."
            )
        score_thresh = _coerce_score_thresh(score_thresh)
    elif score_thresh is None:
        score_thresh = 0.0

    if not isinstance(bidirect, (bool, _numpy.bool_)):
        raise ValueError("bidirect must be True or False")
    if not isinstance(extend, (bool, _numpy.bool_)):
        raise ValueError("gseq_potts: extend must be True or False")
    strand = _seq_strand_arg(strand)
    if bidirect:
        strand = 0
    elif strand == 0:
        # 0 is the signature default, kept for parity with gseq_pwm. Under
        # bidirect=False it cannot mean anything: exactly one strand is read,
        # and silently picking the forward one hands half an answer to a caller
        # carrying gseq_pwm's "0 = both strands" reading over.
        raise ValueError(
            "gseq_potts(bidirect=False) needs an explicit strand: 1 for the "
            "forward strand, -1 for the reverse."
        )

    if isinstance(seqs, str):
        seqs = [seqs]
    # one character per byte, as R reads a string (Latin-1)
    seqs = [_seq_bytes(x) if isinstance(x, str) else x for x in seqs]

    # The same dict the potts virtual tracks build, parsed by the same C++, so
    # gseq_potts() and a potts vtrack cannot disagree about what a model means.
    # strand = 0 never reaches strand_mode: bidirect already implies both.
    params = _potts_params(
        model,
        bidirect=bool(bidirect),
        extend=bool(extend),
        strand=1 if strand == 0 else int(strand),
        score_thresh=float(score_thresh),
        what="gseq_potts",
    )

    out: _numpy.ndarray = _pymisha.pm_gseq_potts(list(seqs), params, mode)
    return out


def gseq_pwm_edits(
    seqs: str | list[str] | _pandas.DataFrame,
    pssm: _numpy.ndarray | _pandas.DataFrame,
    score_thresh: float,
    *,
    max_edits: int | None = None,
    max_indels: int | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    score_min: float | None = None,
    score_max: float | None = None,
    extend: bool | int = True,
    strand: int = 1,
    direction: str = "above",
) -> _pandas.DataFrame:
    """
    Show optimal edits to reach a PWM score threshold.

    For each input sequence (or genomic interval), finds the optimal motif
    window and the specific base changes needed to reach *score_thresh*.
    Returns a long-format DataFrame with one row per edit.

    This is an investigation tool: use it on a small set of sequences to
    see what mutations would create or strengthen binding sites.

    Parameters
    ----------
    seqs : list of str, or pandas.DataFrame
        DNA sequences to analyse.  If a DataFrame with columns ``chrom``,
        ``start``, ``end`` is provided, sequences are automatically
        extracted via ``gseq_extract``, and the windows scanned are those
        the ``pwm.edit_distance`` virtual track scans for the same
        interval.  A ``strand`` column is ignored, as by the virtual track:
        the strand scanned is set by *bidirect* and *strand*.
    pssm : numpy.ndarray or pandas.DataFrame
        Position-specific scoring matrix.  If ndarray, shape ``(w, 4)``
        with columns [A, C, G, T].  If DataFrame, must have columns
        ``A``, ``C``, ``G``, ``T``.  Each row is a motif position and holds
        the probabilities or counts of the four bases; log-odds matrices,
        NaN values and rows summing to 0 are rejected.
    score_thresh : float
        Target PWM log-likelihood score to reach.
    max_edits : int or None, default None
        Maximum number of edits to consider.  ``None`` means no cap.
    max_indels : int or None, default None
        Maximum number of insertions/deletions to allow.  ``None`` or
        ``0`` means substitutions only.  When > 0, a banded
        Needleman-Wunsch DP is used to find the optimal alignment
        between the motif and each candidate window, allowing up to
        *max_indels* insertions or deletions.  The ``edit_type`` column
        distinguishes ``"sub"``, ``"ins"``, and ``"del"`` edits.  An
        inserted base is the best its motif column allows (the column
        maximum for ``"above"``, the minimum for ``"below"``) and is
        scored as such.
    bidirect : bool, default True
        Scan both DNA strands.
    prior : float, default 0.01
        Pseudocount added to PSSM frequencies, between 0 and 1.
    score_min : float or None, default None
        Skip windows whose PWM score is below this value.  The score is the
        window's ungapped score (the PSSM-length window from its start),
        also with ``max_indels > 0``; a window without one (a non-ACGT base
        among its first PSSM-length bases) is skipped when a filter is set.
    score_max : float or None, default None
        Skip windows whose PWM score is above this value.  Same score as
        *score_min*.
    extend : bool or int, default True
        Extend sequence for boundary motifs.  ``True`` = ``w - 1``,
        ``False`` = 0, or an explicit integer.  For interval input it means
        what it means for the ``pwm.edit_distance`` virtual track: ``True``
        or a positive integer lets a window start at every position of the
        interval and read bases past its end; ``False`` or 0 keeps every
        window (deletions included) inside the interval.
    strand : int, default 1
        When ``bidirect=False``: ``1`` = forward, ``-1`` = reverse.
    direction : str, default "above"
        Score direction for edit distance computation:

        - ``"above"``: find minimum edits to raise score **above**
          *score_thresh* (create/strengthen binding site).
        - ``"below"``: find minimum edits to lower score **below**
          *score_thresh* (disrupt/weaken binding site).

        With ``"below"`` and ``bidirect=True``, as in the
        ``pwm.edit_distance`` virtual track, both strand readings of a
        window must end up below the threshold, so the strand needing more
        edits decides ``n_edits`` (a strand that cannot be brought below
        makes the window unreachable); the reported edits and ``strand``
        are those of that strand's reading.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with one row per edit:

        - ``seq_idx``: 1-based index into input sequences.
        - ``strand``: +1 (forward) or -1 (reverse).
        - ``window_start``: 1-based position of optimal window.
        - ``score_before``: PWM score before edits.
        - ``score_after``: PWM score after all edits.
        - ``n_edits``: total edits needed.
        - ``edit_num``: which edit this row represents.
        - ``motif_col``: 1-based position within the motif.
        - ``ref``: current base (``None`` for insertions).
        - ``alt``: suggested replacement base (``None`` for deletions).
        - ``gain``: score improvement from this edit.
        - ``edit_type``: ``"sub"`` (substitution), ``"ins"`` (insertion),
          or ``"del"`` (deletion).  ``None`` for rows with ``n_edits == 0``.
        - ``window_seq``: motif-length sequence at the window.
        - ``mutated_seq``: sequence with all edits applied.

        When intervals are provided, additional columns ``chrom``,
        ``start``, ``end`` are included.

    See Also
    --------
    gseq_pwm : Score sequences with a PWM.
    gvtrack_create : Create virtual tracks with edit distance functions.
    """
    # Validate PSSM (before any sequence is extracted, as R)
    pssm = _coerce_pssm_matrix(pssm)
    if isinstance(pssm, _pandas.DataFrame):
        for col in ("A", "C", "G", "T"):
            if col not in pssm.columns:
                raise KeyError(f"PSSM DataFrame missing column '{col}'")
        pssm_arr = pssm[["A", "C", "G", "T"]].values.astype(_numpy.float64)
    elif isinstance(pssm, _numpy.ndarray):
        if pssm.ndim != 2 or pssm.shape[1] != 4:
            raise ValueError("PSSM array must have shape (w, 4)")
        pssm_arr = pssm.astype(_numpy.float64).copy()
    else:
        raise ValueError("PSSM must be a numpy array or pandas DataFrame")
    # Ensure contiguous C order for C++
    pssm_arr = _numpy.ascontiguousarray(pssm_arr, dtype=_numpy.float64)
    w = pssm_arr.shape[0]

    score_thresh = _coerce_score_thresh(score_thresh)
    _check_max_edits(max_edits)
    if max_indels is not None:
        max_indels = int(max_indels)
        if max_indels < 0:
            raise ValueError("max_indels must be >= 0")
    else:
        max_indels = 0
    if not isinstance(bidirect, (bool, _numpy.bool_)):
        raise ValueError("bidirect must be True or False")
    # R misha #226: an NA extend acted as True on sequences
    if _na_or_seq(extend):
        raise ValueError("extend must be False, True, or a non-negative integer")
    if (isinstance(prior, (bool, _numpy.bool_)) or not isinstance(prior, (int, float, _numpy.number))
            or not 0 <= prior <= 1):
        raise ValueError("prior must be between 0 and 1")
    # R compares with 1 and -1, so TRUE (1) passes
    if _na_or_seq(strand) or isinstance(strand, str) or strand not in (1, -1):
        raise ValueError("strand must be 1 or -1")
    direction = str(direction).lower()
    if direction not in ("above", "below"):
        raise ValueError("direction must be 'above' or 'below'")

    # Handle interval input: the windows scanned are the pwm.edit_distance
    # vtrack's for the same interval and extend value. The vtrack reads
    # extend=True or a positive integer as True: the sequence is padded at the
    # END by (motif length + max_indels - 1), clipped at the chromosome end, and
    # a window starts at every interval position. With extend=False or 0 nothing
    # is padded and a window must fit in the interval (last start: interval
    # length - motif length).
    is_intervals = isinstance(seqs, _pandas.DataFrame) and {"chrom", "start", "end"}.issubset(seqs.columns)
    intervals_df = None
    roi_start = roi_end = None
    if is_intervals:
        assert isinstance(seqs, _pandas.DataFrame)
        intervals_df = seqs
        ext_flag = (isinstance(extend, (bool, _numpy.bool_)) and bool(extend)) or (
            isinstance(extend, (int, float, _numpy.integer, _numpy.floating))
            and not isinstance(extend, (bool, _numpy.bool_))
            and not _is_na(extend) and extend > 0
        )
        ext = w - 1 + max_indels if ext_flag else 0
        # A strand column is ignored, as the vtrack ignores it: gseq_extract()
        # would return strand = -1 rows reverse-complemented, and the scan
        # would then run over the other strand.
        extended = intervals_df.drop(columns="strand", errors="ignore").copy()
        extended["end"] = extended["end"] + ext
        from .intervals import gintervals_all

        chrom_sizes = gintervals_all()
        cs_map = dict(zip(chrom_sizes["chrom"].astype(str), chrom_sizes["end"], strict=False))
        max_end = extended["chrom"].astype(str).map(cs_map)
        keep = max_end.notna().to_numpy()
        ends = extended["end"].to_numpy(copy=True)
        ends[keep] = _numpy.minimum(ends[keep], max_end.to_numpy()[keep])
        extended["end"] = ends
        seqs = gseq_extract(extended)
        # Window starts are limited through the ROI (1-based; C++ lets a window
        # start up to roi_end - w, and no later than where fewer than the
        # shortest window's bases remain). Padded: roi_end = length + w - 1, so
        # the last start is length - 1. Not padded: roi_end = length.
        lengths = (intervals_df["end"] - intervals_df["start"]).to_numpy(dtype=_numpy.int64)
        roi_start = _numpy.ones(len(lengths), dtype=_numpy.int64)
        roi_end = lengths + (w - 1) if ext_flag else lengths
        c_extend: bool | int = False
    else:
        if isinstance(seqs, str):
            seqs = [seqs]
        # The C++ side reads a bool or an int (R's logical or as.integer())
        if isinstance(extend, (bool, _numpy.bool_)):
            c_extend = bool(extend)
        elif isinstance(extend, (int, float, _numpy.integer, _numpy.floating)) and not _is_na(extend):
            c_extend = int(extend)
        else:
            c_extend = extend

    # Ensure seqs is a list of strings, one character per byte (Latin-1)
    seqs = [_seq_bytes(str(s)) if s is not None else "" for s in seqs]

    # Resolve strand mode
    strand_mode = 0 if bidirect else int(strand)

    # Call C++ binding
    result = _pymisha.pm_gseq_pwm_edits(
        seqs,
        pssm_arr,
        float(score_thresh),
        max_edits,
        int(bidirect),
        strand_mode,
        float(prior),
        c_extend,
        score_min,
        score_max,
        max_indels,
        direction,
        roi_start,
        roi_end,
    )

    df = _pandas.DataFrame(result)

    # Add interval columns if input was intervals
    if is_intervals and len(df) > 0:
        assert isinstance(intervals_df, _pandas.DataFrame)
        df["chrom"] = intervals_df["chrom"].values[_numpy.array(df["seq_idx"]) - 1]
        df["start"] = intervals_df["start"].values[_numpy.array(df["seq_idx"]) - 1]
        df["end"] = intervals_df["end"].values[_numpy.array(df["seq_idx"]) - 1]
        # Reorder columns
        cols = ["seq_idx", "chrom", "start", "end"] + [
            c for c in df.columns if c not in ("seq_idx", "chrom", "start", "end")
        ]
        df = df[cols]

    return df
