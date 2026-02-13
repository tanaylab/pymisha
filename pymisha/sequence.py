"""Sequence manipulation functions (gseq.*)."""

import math as _math

import numpy as _numpy
import pandas as _pandas

from ._shared import CONFIG, _checkroot, _df2pymisha, _pymisha

# Complement mapping for DNA bases
_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_VALID_BASES = frozenset("ACGTacgt")
_BASE4_ALPHABET = "ACGT"
_DNA_CODE_TABLE = _numpy.full(256, -1, dtype=_numpy.int8)
for _base, _code in ((ord("A"), 0), (ord("C"), 1), (ord("G"), 2), (ord("T"), 3)):
    _DNA_CODE_TABLE[_base] = _code
    _DNA_CODE_TABLE[ord(chr(_base).lower())] = _code

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
        pos = 0
        while True:
            pos = clean_seq.find(kmer, pos)
            if pos == -1:
                break
            count += 1
            pos += 1

    # Count reverse strand
    if strand in (0, -1) and kmer_rc is not None:
        pos = 0
        while True:
            pos = clean_seq.find(kmer_rc, pos)
            if pos == -1:
                break
            count += 1
            pos += 1

    return count


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

    for seq in seqs:
        if not seq or len(seq) < k:
            continue

        raw = _numpy.frombuffer(seq.encode("ascii", "ignore"), dtype=_numpy.uint8)
        codes = _DNA_CODE_TABLE[raw].astype(_numpy.int64, copy=False)
        valid = codes >= 0
        if not valid.any():
            continue

        # Process only contiguous valid A/C/G/T runs to skip N/invalid chars.
        transitions = _numpy.diff(
            _numpy.concatenate((
                _numpy.array([0], dtype=_numpy.int8),
                valid.astype(_numpy.int8),
                _numpy.array([0], dtype=_numpy.int8),
            ))
        )
        starts = _numpy.flatnonzero(transitions == 1)
        ends = _numpy.flatnonzero(transitions == -1)

        for start, end in zip(starts, ends, strict=False):
            segment = codes[start:end]
            seg_len = int(segment.size)
            n_windows = seg_len - k + 1
            if n_windows <= 0:
                continue

            # Rolling base-4 hash for each k-mer window.
            window_codes = _numpy.zeros(n_windows, dtype=_numpy.int64)
            for j in range(k):
                window_codes *= 4
                window_codes += segment[j:j + n_windows]
            counts += _numpy.bincount(window_codes, minlength=num_kmers)

    nonzero_idx = _numpy.flatnonzero(counts)
    if nonzero_idx.size == 0:
        return _pandas.DataFrame({"kmer": _pandas.Series(dtype=str),
                                  "count": _pandas.Series(dtype=int)})

    kmers = []
    for code in nonzero_idx:
        n = int(code)
        chars = ["A"] * k
        for pos in range(k - 1, -1, -1):
            n, rem = divmod(n, 4)
            chars[pos] = _BASE4_ALPHABET[rem]
        kmers.append("".join(chars))

    return _pandas.DataFrame(
        {"kmer": kmers, "count": counts[nonzero_idx].astype(_numpy.int64, copy=False)},
        columns=["kmer", "count"],
    )


# ---------------------------------------------------------------------------
# PWM scoring helpers
# ---------------------------------------------------------------------------


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
                        neutral_policy, return_strand):
    """Score a single uppercased sequence with PWM."""
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
        Spatial weighting factors per bin (not yet implemented).
    spat_bin : int, default 1
        Bin size for spatial weighting.
    spat_min : float or None
        Spatial scan minimum.
    spat_max : float or None
        Spatial scan maximum.
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
    if spat_factor is not None:
        raise NotImplementedError("Spatial weighting is not yet supported")

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

    # --- Score each sequence ---
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
        )
        if return_strand and mode == "pos":
            results.append(r[0])
            strand_results.append(r[1])
        else:
            results.append(r)

    if return_strand and mode == "pos":
        return _pandas.DataFrame({"pos": results, "strand": strand_results})

    return _numpy.array(results, dtype=float)
