"""Sequence manipulation functions (gseq.*)."""

import math as _math

import numpy as _numpy
import pandas as _pandas

from ._shared import CONFIG, _checkroot, _df2pymisha, _pymisha
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
    num = 4 ** k
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
_PWM_BASE_CODE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_PWM_COMP_CODE = {'A': 3, 'C': 2, 'G': 1, 'T': 0}  # complement base codes
_LOG_QUARTER = _math.log(0.25)


def gseq_extract(intervals):
    """
    Extract DNA sequences for given intervals.

    Returns an array of sequence strings for each interval from 'intervals'.
    If intervals contain an additional 'strand' column and its value is -1,
    the reverse-complementary sequence is returned.

    Parameters
    ----------
    intervals : DataFrame
        Intervals for which DNA sequence is returned. Must have 'chrom',
        'start', and 'end' columns. Optional 'strand' column (-1 for reverse
        complement).

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

    See Also
    --------
    gseq_rev, gseq_comp, gseq_revcomp, gseq_kmer
    """
    if intervals is None:
        raise ValueError("Usage: gseq_extract(intervals)")

    _checkroot()

    intervals = _maybe_load_intervals_set(intervals)

    return _pymisha.pm_seq_extract(_df2pymisha(intervals), CONFIG)


def gseq_rev(seq):
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


def gseq_comp(seq):
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


def gseq_revcomp(seq):
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


def grevcomp(seq):
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


def _count_str_occurrences(haystack: str, needle: str) -> int:
    """Count overlapping occurrences of needle in haystack using str.find."""
    count = 0
    pos = 0
    while True:
        pos = haystack.find(needle, pos)
        if pos == -1:
            break
        count += 1
        pos += 1
    return count


def _count_kmer_in_seq(seq, kmer, strand, start_pos, end_pos, extend,
                       skip_gaps, gap_chars):
    """Count occurrences of a k-mer in a single sequence.

    Parameters follow R misha conventions:
    - start_pos/end_pos are 1-based inclusive
    - extend: if True, allow k-mer to start outside ROI
    - skip_gaps: skip gap characters when scanning
    """
    seq_upper = seq.upper()
    k = len(kmer)
    seq_len = len(seq_upper)

    if seq_len == 0:
        return 0

    # Determine ROI bounds (convert 1-based inclusive to 0-based)
    roi_start = max(0, start_pos - 1) if start_pos is not None else 0
    roi_end = min(seq_len, end_pos) if end_pos is not None else seq_len

    if roi_end <= roi_start:
        return 0

    # Determine scan window
    if extend:
        scan_start = max(0, roi_start - k + 1)
        scan_end = min(seq_len, roi_end + k - 1)
    else:
        scan_start = roi_start
        scan_end = roi_end

    # Remove gap characters if skip_gaps
    if skip_gaps and gap_chars:
        gap_set = frozenset(gap_chars)
        chars = []
        for i in range(scan_start, scan_end):
            if seq_upper[i] not in gap_set:
                chars.append(seq_upper[i])
        clean_seq = "".join(chars)
    else:
        clean_seq = seq_upper[scan_start:scan_end]

    count = 0
    kmer_rc = kmer[::-1].translate(_COMPLEMENT).upper() if strand != 1 else None

    # Count forward strand
    if strand in (0, 1):
        count += _count_str_occurrences(clean_seq, kmer)

    # Count reverse strand
    if strand in (0, -1) and kmer_rc is not None:
        count += _count_str_occurrences(clean_seq, kmer_rc)

    return count


def _numpy_count_kmer(seq_bytes, kmer_bytes, k):
    """Count overlapping occurrences of kmer_bytes in seq_bytes using numpy.

    Parameters
    ----------
    seq_bytes : numpy.ndarray of uint8
        Upper-cased ASCII byte array of the sequence.
    kmer_bytes : numpy.ndarray of uint8
        Upper-cased ASCII byte array of the k-mer.
    k : int
        Length of the k-mer.

    Returns
    -------
    int
        Number of overlapping occurrences.
    """
    n = len(seq_bytes)
    if n < k:
        return 0
    n_windows = n - k + 1
    match = _numpy.ones(n_windows, dtype=bool)
    for j in range(k):
        match &= seq_bytes[j:j + n_windows] == kmer_bytes[j]
    return int(match.sum())


def _numpy_count_kmer_per_seq(seq_bytes, boundaries, kmer_bytes, k):
    """Count k-mer occurrences per sequence in a concatenated byte array.

    Sequences are concatenated with N*k separators so k-mer matches cannot
    span sequence boundaries.

    Parameters
    ----------
    seq_bytes : numpy.ndarray of uint8
        Concatenated upper-cased ASCII byte array with separators.
    boundaries : numpy.ndarray of int64
        Cumulative boundaries: seq i occupies bytes
        [boundaries[i], boundaries[i+1] - k_sep) where k_sep = k (separator len).
    kmer_bytes : numpy.ndarray of uint8
        Upper-cased ASCII byte array of the k-mer.
    k : int
        Length of the k-mer.

    Returns
    -------
    numpy.ndarray of int64
        Count per sequence.
    """
    n_total = len(seq_bytes)
    n_seqs = len(boundaries) - 1
    counts = _numpy.zeros(n_seqs, dtype=_numpy.int64)

    if n_total < k:
        return counts

    # Find all match positions in the concatenated array
    n_windows = n_total - k + 1
    match = _numpy.ones(n_windows, dtype=bool)
    for j in range(k):
        match &= seq_bytes[j:j + n_windows] == kmer_bytes[j]

    match_positions = _numpy.flatnonzero(match)
    if match_positions.size == 0:
        return counts

    # Assign each match to its sequence using searchsorted on boundaries
    seq_idx = _numpy.searchsorted(boundaries[1:], match_positions, side="right")
    return _numpy.bincount(seq_idx, minlength=n_seqs)


def _gseq_kmer_fast(seqs, kmer, mode, strand, k):
    """Fast path for gseq_kmer: no ROI, no gaps, no extend.

    Uses numpy byte-level matching for vectorized k-mer counting.
    """
    n = len(seqs)
    if n == 0:
        return _numpy.zeros(0, dtype=float)

    kmer_bytes = _numpy.frombuffer(kmer.encode("ascii"), dtype=_numpy.uint8)
    kmer_rc = kmer[::-1].translate(_COMPLEMENT).upper() if strand != 1 else None
    kmer_rc_bytes = (
        _numpy.frombuffer(kmer_rc.encode("ascii"), dtype=_numpy.uint8)
        if kmer_rc is not None
        else None
    )

    # Determine which patterns to search for
    search_fwd = strand in (0, 1)
    search_rev = strand in (0, -1) and kmer_rc is not None

    # For small batches of short sequences, str.find is faster than numpy overhead
    total_bytes = sum(len(s) for s in seqs)
    if total_bytes < 500:
        results = _numpy.zeros(n, dtype=float)
        for i in range(n):
            seq = seqs[i]
            seq_len = len(seq)
            if seq_len < k:
                continue
            seq_upper = seq.upper()
            count = 0
            if search_fwd:
                count += _count_str_occurrences(seq_upper, kmer)
            if search_rev:
                count += _count_str_occurrences(seq_upper, kmer_rc)
            if mode == "frac":
                possible = max(0, seq_len - k + 1)
                if strand == 0:
                    possible *= 2
                results[i] = count / possible if possible > 0 else 0.0
            else:
                results[i] = count
        return results

    # For a single large sequence, avoid concatenation overhead
    if n == 1:
        seq = seqs[0]
        seq_len = len(seq)
        if seq_len < k:
            return _numpy.zeros(1, dtype=float)

        sb = _numpy.frombuffer(seq.upper().encode("ascii"), dtype=_numpy.uint8)
        count = 0
        if search_fwd:
            count += _numpy_count_kmer(sb, kmer_bytes, k)
        if search_rev:
            count += _numpy_count_kmer(sb, kmer_rc_bytes, k)

        if mode == "frac":
            possible = max(0, seq_len - k + 1)
            if strand == 0:
                possible *= 2
            return _numpy.array(
                [count / possible if possible > 0 else 0.0], dtype=float
            )
        return _numpy.array([float(count)], dtype=float)

    # Multiple sequences: concatenate with N*k separators for batch processing.
    # N (0x4E) cannot match any ACGT kmer byte, so matches won't cross boundaries.
    sep = bytes([0x4E]) * k  # b'NNN...' with length k
    seq_lens = _numpy.array([len(s) for s in seqs], dtype=_numpy.int64)
    # Each seq contributes len(s) bytes + k separator bytes (except last)
    # But we add separator after every seq for simplicity, extra sep at end is harmless
    parts = []
    boundaries = _numpy.empty(n + 1, dtype=_numpy.int64)
    boundaries[0] = 0
    for i, seq in enumerate(seqs):
        parts.append(seq.upper().encode("ascii", "ignore"))
        parts.append(sep)
        boundaries[i + 1] = boundaries[i] + len(parts[-2]) + k

    all_bytes = _numpy.frombuffer(b"".join(parts), dtype=_numpy.uint8)

    fwd_counts = _numpy.zeros(n, dtype=_numpy.int64)
    rev_counts = _numpy.zeros(n, dtype=_numpy.int64)

    if search_fwd:
        fwd_counts = _numpy_count_kmer_per_seq(
            all_bytes, boundaries, kmer_bytes, k
        )
    if search_rev:
        rev_counts = _numpy_count_kmer_per_seq(
            all_bytes, boundaries, kmer_rc_bytes, k
        )

    total_counts = fwd_counts + rev_counts

    if mode == "frac":
        possible = _numpy.maximum(0, seq_lens - k + 1).astype(float)
        if strand == 0:
            possible *= 2
        with _numpy.errstate(divide="ignore", invalid="ignore"):
            return _numpy.where(possible > 0, total_counts / possible, 0.0)

    return total_counts.astype(float)


def gseq_kmer(seqs, kmer, mode="count", strand=0, start_pos=None,
              end_pos=None, extend=False, skip_gaps=True,
              gap_chars=None):
    """
    Count k-mer occurrences in DNA sequences.

    Parameters
    ----------
    seqs : str or list of str
        DNA sequence(s) to search.
    kmer : str
        K-mer pattern to search for (only A, C, G, T characters).
    mode : str, default "count"
        "count" returns raw counts, "frac" returns fraction of possible positions.
    strand : int, default 0
        0 = both strands, 1 = forward only, -1 = reverse complement only.
    start_pos : int, optional
        1-based start position of region of interest.
    end_pos : int, optional
        1-based end position of region of interest (inclusive).
    extend : bool, default False
        If True, allow k-mer to extend beyond ROI boundaries.
    skip_gaps : bool, default True
        If True, skip gap characters when scanning.
    gap_chars : list of str, optional
        Characters to treat as gaps. Default: ["-", "."].

    Returns
    -------
    numpy.ndarray
        Array of counts or fractions, one per input sequence.

    Examples
    --------
    >>> import pymisha as pm

    Count CG dinucleotides on both strands:

    >>> pm.gseq_kmer(["CGCGCGCGCG", "ATATATATAT"], "CG")
    array([10.,  0.])

    Get fraction instead of count:

    >>> pm.gseq_kmer(["CGCGCGCGCG"], "CG", mode="frac")  # doctest: +ELLIPSIS
    array([0.555...])

    Forward strand only:

    >>> pm.gseq_kmer(["CGCGCGCGCG"], "CG", strand=1)
    array([5.])

    Count in a specific region:

    >>> pm.gseq_kmer("ATCGATCG", "AT", start_pos=1, end_pos=4)
    array([2.])

    See Also
    --------
    gseq_kmer_dist, gseq_pwm, gseq_extract
    """
    if gap_chars is None:
        gap_chars = ["-", "."]

    if isinstance(seqs, str):
        seqs = [seqs]

    kmer = kmer.upper()
    if not all(c in "ACGT" for c in kmer):
        raise ValueError("kmer must contain only A, C, G, T characters")
    if len(kmer) == 0:
        raise ValueError("kmer must be non-empty")

    if strand not in (-1, 0, 1):
        raise ValueError("strand must be -1, 0, or 1")

    k = len(kmer)

    # Fast path: no ROI constraints, no extend.
    # Gap characters ("-", ".") are virtually never present in DNA sequences;
    # even when skip_gaps is True, if no gap chars actually exist in the data
    # the slow path and fast path produce identical results.  We check cheaply.
    has_roi = start_pos is not None or end_pos is not None
    if not has_roi and not extend:
        # Quick gap scan: only fall back to slow path if gap chars are actually
        # present in any sequence.
        need_slow = False
        if skip_gaps and gap_chars:
            gap_set = frozenset(gap_chars)
            for seq in seqs:
                if gap_set.intersection(seq):
                    need_slow = True
                    break
        if not need_slow:
            return _gseq_kmer_fast(seqs, kmer, mode, strand, k)

    # Slow path: handles ROI, gaps, extend
    results = _numpy.zeros(len(seqs), dtype=float)

    for i, seq in enumerate(seqs):
        count = _count_kmer_in_seq(seq, kmer, strand, start_pos, end_pos,
                                   extend, skip_gaps, gap_chars)
        if mode == "frac":
            seq_len = len(seq)
            if start_pos is not None and end_pos is not None:
                roi_len = min(seq_len, end_pos) - max(0, start_pos - 1)
            elif start_pos is not None:
                roi_len = seq_len - max(0, start_pos - 1)
            elif end_pos is not None:
                roi_len = min(seq_len, end_pos)
            else:
                roi_len = seq_len
            possible = max(0, roi_len - k + 1)
            if strand == 0:
                possible *= 2
            results[i] = count / possible if possible > 0 else 0.0
        else:
            results[i] = count

    return results


def gseq_kmer_dist(intervals, k=6, mask=None):
    """
    Compute k-mer distribution in genomic intervals.

    Counts all k-mers of size k within the specified genomic intervals,
    optionally excluding masked regions.

    Parameters
    ----------
    intervals : DataFrame
        Genomic intervals to analyze.
    k : int, default 6
        K-mer size (1-10).
    mask : DataFrame, optional
        Intervals to exclude from counting.

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

    # Extract sequences for the intervals
    seqs = gseq_extract(intervals)

    # If mask is provided, extract mask intervals and subtract
    if mask is not None:
        from .intervals import gintervals_diff
        intervals_clean = gintervals_diff(intervals, mask)
        if intervals_clean is None or len(intervals_clean) == 0:
            return _pandas.DataFrame({"kmer": [], "count": []}).astype(
                {"kmer": str, "count": int}
            )
        seqs = gseq_extract(intervals_clean)

    num_kmers = 4 ** k
    counts = _numpy.zeros(num_kmers, dtype=_numpy.int64)

    # Pre-compute powers of 4 for the rolling hash
    powers = _numpy.int64(4) ** _numpy.arange(k - 1, -1, -1, dtype=_numpy.int64)

    for seq in seqs:
        if not seq or len(seq) < k:
            continue

        raw = _numpy.frombuffer(seq.encode("ascii", "ignore"), dtype=_numpy.uint8)
        codes = _DNA_CODE_TABLE[raw]
        valid = codes >= 0
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
                    window_codes += segment[j:j + n_windows]

            counts += _numpy.bincount(window_codes, minlength=num_kmers)

    nonzero_idx = _numpy.flatnonzero(counts)
    if nonzero_idx.size == 0:
        return _pandas.DataFrame({"kmer": _pandas.Series(dtype=str),
                                  "count": _pandas.Series(dtype=int)})

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
for _b, _f, _r in ((ord('A'), 0, 3), (ord('C'), 1, 2),
                    (ord('G'), 2, 1), (ord('T'), 3, 0)):
    _PWM_FWD_CODE[_b] = _f
    _PWM_FWD_CODE[_b + 32] = _f  # lowercase
    _PWM_RC_CODE[_b] = _r
    _PWM_RC_CODE[_b + 32] = _r


def _numpy_logsumexp(arr):
    """Numerically stable logsumexp over a 1-D array (finite values only)."""
    finite = arr[_numpy.isfinite(arr)]
    if finite.size == 0:
        return -_numpy.inf
    m = finite.max()
    return m + _numpy.log(_numpy.sum(_numpy.exp(finite - m)))


def _pwm_score_batch_vectorized(seqs, log_pssm, avg_log, w,
                                mode, scan_fwd, scan_rev,
                                score_thresh, return_strand,
                                spat_log_factors=None, spat_bin=1,
                                neutral_policy="average"):
    """Vectorized PWM scoring for a batch of sequences (no ROI/gaps).

    Replaces the per-window Python loop with numpy stride-tricks-based
    matrix indexing for ~10-50x speedup on typical workloads.
    """
    n = len(seqs)
    results = _numpy.full(n, _numpy.nan, dtype=float)
    strand_results = _numpy.zeros(n, dtype=_numpy.int64) if return_strand else None
    use_spat = spat_log_factors is not None and len(spat_log_factors) > 0
    n_spat = len(spat_log_factors) if use_spat else 0

    # Pre-compute reversed log_pssm for RC scoring
    # RC scoring at position j uses pssm[w-1-j, comp_code]
    # which equals reversed_pssm[j, comp_code]
    log_pssm_rev = log_pssm[::-1]  # reversed row order

    for i, seq in enumerate(seqs):
        seq_upper = seq.upper()
        L = len(seq_upper)
        if w > L:
            if mode == "count":
                results[i] = 0
            continue

        # Encode sequence to base codes
        raw = _numpy.frombuffer(seq_upper.encode('ascii'), dtype=_numpy.uint8)
        fwd_codes = _PWM_FWD_CODE[raw]   # A=0,C=1,G=2,T=3, other=-1
        rc_codes = _PWM_RC_CODE[raw]      # A=3,C=2,G=1,T=0, other=-1

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

        if scan_fwd:
            # Check for bases with code == -1 (neutral chars like N, *)
            has_unknown = _numpy.any(fwd_windows == -1, axis=1)

            # Score fully-valid windows via vectorized fancy indexing
            fwd_scores = _numpy.full(n_windows, -_numpy.inf)
            valid = ~has_unknown
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
                offsets = _numpy.arange(n_windows)
                bins = _numpy.clip(offsets // spat_bin, 0, n_spat - 1)
                spat = spat_log_factors[bins]
                finite_mask = _numpy.isfinite(fwd_scores)
                fwd_scores[finite_mask] += spat[finite_mask]

            all_scores.append((_numpy.ones(n_windows, dtype=_numpy.int8), fwd_scores))

        if scan_rev:
            has_unknown_rc = _numpy.any(rc_windows == -1, axis=1)
            rc_scores = _numpy.full(n_windows, -_numpy.inf)
            valid_rc = ~has_unknown_rc
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

        # Aggregate scores by mode
        if mode == "lse":
            combined = _numpy.concatenate([s for _, s in all_scores])
            results[i] = _numpy_logsumexp(combined)

        elif mode == "max":
            combined = _numpy.concatenate([s for _, s in all_scores])
            non_nan = combined[~_numpy.isnan(combined)]
            if non_nan.size > 0:
                results[i] = non_nan.max()

        elif mode == "pos":
            best_score = -_numpy.inf
            best_pos = _numpy.nan
            best_strand = 0
            for dirs, scores in all_scores:
                finite_mask = _numpy.isfinite(scores)
                if not finite_mask.any():
                    continue
                idx = _numpy.argmax(scores)
                if scores[idx] > best_score:
                    best_score = scores[idx]
                    best_pos = float(idx + 1)  # 1-based
                    best_strand = int(dirs[0])
            results[i] = best_pos
            if strand_results is not None:
                strand_results[i] = best_strand

        elif mode == "count":
            combined = _numpy.concatenate([s for _, s in all_scores])
            results[i] = float(_numpy.sum(combined >= score_thresh))

    if return_strand and mode == "pos":
        return results, strand_results
    return results


def _pwm_log_sum_exp(scores):
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


def _pwm_score_window(bases, log_pssm, avg_log, w, neutral_set,
                      neutral_policy, reverse):
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
                return float('nan')
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


def _pwm_iter_windows(seq_upper, start_min, start_max, w, skip_gaps, gap_set):
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
            yield phys_start, "".join(comp[log_i:log_i + w])
    else:
        for pos in range(start_min, start_max + 1):
            yield pos, seq_upper[pos:pos + w]


def _pwm_score_sequence(seq_upper, L, log_pssm, avg_log, w,
                        mode, scan_fwd, scan_rev, score_thresh,
                        start_pos, end_pos, extend,
                        skip_gaps, gap_set, neutral_set,
                        neutral_policy, return_strand,
                        spat_log_factors=None, spat_bin=1):
    """Score a single uppercased sequence with PWM.

    Parameters
    ----------
    spat_log_factors : numpy.ndarray or None
        Pre-computed log(spat_factor) per spatial bin.  ``None`` means
        no spatial weighting.
    spat_bin : int
        Bin size for spatial weighting (positions per bin).
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

    if start_max < start_min or w > L:
        if mode == "count":
            return 0
        if return_strand and mode == "pos":
            return (float('nan'), 0)
        return float('nan')

    use_spat = spat_log_factors is not None and len(spat_log_factors) > 0
    n_spat = len(spat_log_factors) if use_spat else 0

    # Iterate windows and aggregate
    all_scores = []  # for lse
    best_score = -_numpy.inf
    best_pos = -1
    best_strand = 0
    count = 0
    has_valid = False

    for phys_pos, window in _pwm_iter_windows(seq_upper, start_min,
                                              start_max, w,
                                              skip_gaps, gap_set):
        # Spatial log factor for this window position.
        # offset_from_roi = (phys_pos + 1) - roi_start  (matching C++: (s0+1) - roi_start1)
        if use_spat:
            offset = (phys_pos + 1) - roi_start
            sb = offset // spat_bin
            if sb < 0:
                sb = 0
            elif sb >= n_spat:
                sb = n_spat - 1
            spat_log = float(spat_log_factors[sb])
        else:
            spat_log = 0.0

        for is_rev, strand_dir in ((False, 1), (True, -1)):
            if is_rev and not scan_rev:
                continue
            if not is_rev and not scan_fwd:
                continue

            s = _pwm_score_window(window, log_pssm, avg_log, w,
                                  neutral_set, neutral_policy, is_rev)
            if _numpy.isnan(s):
                continue
            has_valid = True

            # Apply spatial weight in log space
            if use_spat and s > -_numpy.inf:
                s += spat_log

            if mode == "lse":
                all_scores.append(s)
            elif mode in ("max", "pos"):
                if s > best_score:
                    best_score = s
                    best_pos = phys_pos + 1  # 1-based
                    best_strand = strand_dir
            elif mode == "count" and s >= score_thresh:
                count += 1

    # Aggregate final result
    if mode == "lse":
        if not has_valid:
            return float('nan')
        return _pwm_log_sum_exp(all_scores)
    if mode == "max":
        if not has_valid:
            return float('nan')
        return best_score
    if mode == "pos":
        if not has_valid or best_pos < 0:
            if return_strand:
                return (float('nan'), 0)
            return float('nan')
        if return_strand:
            return (float(best_pos), best_strand)
        return float(best_pos)
    # count
    return count


def gseq_pwm(seqs, pssm, mode="lse", bidirect=True, strand=0,
             score_thresh=0, start_pos=None, end_pos=None, extend=False,
             spat_factor=None, spat_bin=1, spat_min=None, spat_max=None,
             return_strand=False, skip_gaps=True, gap_chars=None,
             neutral_chars=None, neutral_chars_policy="average",
             prior=0.01):
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
    strand : int, default 0
        When ``bidirect=False``: ``1`` = forward only, ``-1`` = reverse
        complement only, ``0`` = forward only (default).
    score_thresh : float, default 0
        Threshold for ``"count"`` mode.
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
        Reserved for virtual-track context (not used in string scoring).
    spat_max : float or None
        Reserved for virtual-track context (not used in string scoring).
    return_strand : bool, default False
        For ``mode="pos"`` with bidirectional scoring, return a DataFrame
        with ``pos`` and ``strand`` columns instead of a plain array.
    skip_gaps : bool, default True
        Skip gap characters when scanning.
    gap_chars : list of str or None
        Characters treated as gaps. Default ``["-", "."]``.
    neutral_chars : list of str or None
        Characters treated as unknown/ambiguous bases.
        Default ``["N", "n", "*"]``.
    neutral_chars_policy : str, default ``"average"``
        How to score neutral characters:

        - ``"average"``: mean log-probability of the PSSM column.
        - ``"log_quarter"``: ``log(0.25)``.
        - ``"na"``: window is invalid (NaN).
    prior : float, default 0.01
        Pseudocount added to PSSM before normalization. Must be >= 0.

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
        raise ValueError(
            f"mode must be 'lse', 'max', 'pos', or 'count', got '{mode}'"
        )
    if strand not in (-1, 0, 1):
        raise ValueError(f"strand must be -1, 0, or 1, got {strand}")
    if prior < 0:
        raise ValueError(f"prior must be non-negative, got {prior}")
    if neutral_chars_policy not in ("average", "log_quarter", "na"):
        raise ValueError(
            "neutral_chars_policy must be 'average', 'log_quarter', or 'na'"
        )

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
        for col in ('A', 'C', 'G', 'T'):
            if col not in pssm.columns:
                raise KeyError(f"PSSM DataFrame missing column '{col}'")
        pssm_arr = pssm[['A', 'C', 'G', 'T']].values.astype(float)
    elif isinstance(pssm, _numpy.ndarray):
        if pssm.ndim != 2 or pssm.shape[1] != 4:
            raise ValueError("PSSM array must have shape (w, 4)")
        pssm_arr = pssm.astype(float).copy()
    else:
        raise ValueError("PSSM must be a numpy array or pandas DataFrame")

    w = pssm_arr.shape[0]

    # Apply prior and normalize rows into probabilities.
    if prior > 0:
        pssm_arr = pssm_arr + prior
    row_sums = pssm_arr.sum(axis=1, keepdims=True)
    if _numpy.any(row_sums <= 0):
        raise ValueError("Each PSSM row must have a positive total weight")
    pssm_arr = pssm_arr / row_sums

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
        scan_fwd = strand in (0, 1)
        scan_rev = strand == -1

    # --- Defaults ---
    if gap_chars is None:
        gap_chars = ["-", "."]
    if neutral_chars is None:
        neutral_chars = ["N", "n", "*"]

    gap_set = frozenset(gap_chars)
    # Uppercase neutral chars so they match uppercased sequence
    neutral_set = frozenset(c.upper() for c in neutral_chars)

    # --- Single-string convenience ---
    if isinstance(seqs, str):
        seqs = [seqs]

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
            r = _pwm_score_batch_vectorized(
                seqs, log_pssm, avg_log, w,
                mode, scan_fwd, scan_rev, score_thresh,
                return_strand,
                spat_log_factors=spat_log_factors, spat_bin=spat_bin,
                neutral_policy=neutral_chars_policy,
            )
            if return_strand and mode == "pos":
                return _pandas.DataFrame({"pos": r[0], "strand": r[1]})
            return r

    # --- Slow path: per-sequence scoring ---
    results = []
    strand_results = []

    for seq in seqs:
        seq_upper = seq.upper()
        r = _pwm_score_sequence(
            seq_upper, len(seq_upper), log_pssm, avg_log, w,
            mode, scan_fwd, scan_rev, score_thresh,
            start_pos, end_pos, extend,
            skip_gaps, gap_set, neutral_set, neutral_chars_policy,
            return_strand,
            spat_log_factors=spat_log_factors, spat_bin=spat_bin,
        )
        if return_strand and mode == "pos":
            results.append(r[0])
            strand_results.append(r[1])
        else:
            results.append(r)

    if return_strand and mode == "pos":
        return _pandas.DataFrame({"pos": results, "strand": strand_results})

    return _numpy.array(results, dtype=float)
