"""Motif writers for the MEME, JASPAR and HOMER formats.

The counterparts of :mod:`pymisha.motif_import`: ``read(write(x))`` gives back
``x`` within the written precision. R misha's ``gseq.write_meme``,
``gseq.write_jaspar`` and ``gseq.write_homer`` (``R/motif-export.R``) write the
same bytes for the same input; keep the two in step.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence

import numpy as _numpy
import pandas as _pandas

_ACGT = ("A", "C", "G", "T")

# Whitespace as the writers see it: spelled out, so that R and Python agree.
_WS = re.compile(r"[ \t\r\n\f\v]+")

# IUPAC symbols indexed by A=1 + C=2 + G=4 + T=8
_IUPAC = ("", "A", "C", "M", "G", "R", "S", "V", "T", "W", "Y", "H", "K", "D", "B", "N")

__all__ = ["gseq_write_meme", "gseq_write_jaspar", "gseq_write_homer"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(_pandas.isna(x))
    except (TypeError, ValueError):
        return False


def _text(x) -> str | None:
    """Trim and collapse whitespace; None for missing, empty or non-scalar values."""
    if _is_missing(x) or isinstance(x, (list, tuple, dict, _numpy.ndarray)):
        return None
    s = _WS.sub(" ", str(x)).strip(" ")
    return s or None


def _number(x) -> float | None:
    """A finite float, or None."""
    if x is None or isinstance(x, (str, bytes)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt_prob(x: float) -> str:
    """Probabilities: 6 decimals, as MEME writes them."""
    return format(x, ".6f")


def _fmt_number(x: float) -> str:
    """Metadata numbers (nsites, E-values): 15 significant digits."""
    return format(x, ".15g")


def _fmt_count(x: float) -> str:
    """Counts: 6 decimals without trailing zeros, so integer counts print as integers."""
    s = format(x, ".6f").rstrip("0")
    return s[:-1] if s.endswith(".") else s


def _write_lines(lines: list[str], file) -> None:
    with open(file, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _check_file(file) -> None:
    if not isinstance(file, (str, bytes)) and not hasattr(file, "__fspath__"):
        raise TypeError("'file' must be a single file path string")
    if isinstance(file, str) and not file:
        raise ValueError("'file' must be a single file path string")


def _recycle(x, n: int, what: str) -> list[float]:
    vals = [x] if _numpy.isscalar(x) else list(x)
    try:
        vals = [float(v) for v in vals]
    except (TypeError, ValueError) as err:
        raise ValueError(f"'{what}' must be a finite number or one per motif") from err
    if len(vals) not in (1, n) or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"'{what}' must be a finite number or one per motif")
    return vals * n if len(vals) == 1 else vals


def _is_acgt(labels) -> bool:
    if labels is None:
        return False
    labs = [str(x).upper() for x in labels]
    return len(labs) == 4 and len(set(labs)) == 4 and set(labs) == set(_ACGT)


def _export_matrix(m, mid: str) -> _numpy.ndarray:
    """One motif as a positions x A,C,G,T probability matrix."""
    if isinstance(m, _pandas.DataFrame):
        cols, idx = list(m.columns), list(m.index)
        default_cols = isinstance(m.columns, _pandas.RangeIndex)
        if _is_acgt(cols):
            upper = [str(c).upper() for c in cols]
            arr = m.iloc[:, [upper.index(b) for b in _ACGT]].to_numpy()
        elif _is_acgt(idx):
            upper = [str(i).upper() for i in idx]
            arr = m.iloc[[upper.index(b) for b in _ACGT], :].to_numpy().T
        elif default_cols and m.shape[1] == 4:
            arr = m.to_numpy()
        else:
            raise ValueError(f"Motif '{mid}' must have 4 columns A, C, G, T (one row per position)")
    else:
        arr = _numpy.asarray(m)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"Motif '{mid}' must have 4 columns A, C, G, T (one row per position)")
    try:
        arr = _numpy.array(arr, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Motif '{mid}' must be a numeric matrix") from err
    if arr.shape[0] == 0:
        raise ValueError(f"Motif '{mid}' has no positions")
    if not _numpy.all(_numpy.isfinite(arr)):
        raise ValueError(f"Motif '{mid}' has missing or infinite values")
    if _numpy.any(arr < 0):
        raise ValueError(f"Motif '{mid}' has negative values; log-odds matrices cannot be written")
    arr[arr == 0] = 0.0  # no "-0.000000"
    # Summed in a fixed order, as R misha sums them, so the two agree to the last bit.
    rsums = ((arr[:, 0] + arr[:, 1]) + arr[:, 2]) + arr[:, 3]
    bad = _numpy.where(_numpy.abs(rsums - 1.0) > 1e-4)[0]
    if len(bad) > 0:
        warnings.warn(
            f"Row sums deviate from 1.0 in motif '{mid}'; re-normalizing {len(bad)} row(s)",
            stacklevel=4,
        )
        for i in bad:
            arr[i, :] = 0.25 if rsums[i] == 0 else arr[i, :] / rsums[i]
    return arr


def _attrs_of(m) -> dict:
    return dict(m.attrs) if isinstance(m, _pandas.DataFrame) else {}


def _prepare(motifs) -> list[tuple[str, _numpy.ndarray, dict]]:
    """The motifs to write: a list of (id, matrix, attrs)."""
    if isinstance(motifs, (_pandas.DataFrame, _numpy.ndarray)):
        items = [("", motifs)]
    elif isinstance(motifs, Mapping):
        items = [("" if k is None else k, v) for k, v in motifs.items()]
    elif isinstance(motifs, Sequence) and not isinstance(motifs, (str, bytes)):
        items = [("", v) for v in motifs]
    else:
        raise TypeError("'motifs' must be a matrix or a non-empty dict or list of matrices")
    if not items:
        raise ValueError("'motifs' must be a matrix or a non-empty dict or list of matrices")
    ids = [_text(k) for k, _ in items]
    if any(i is not None and " " in i for i in ids):
        warnings.warn("Motif IDs cannot contain whitespace; replacing it with '_'", stacklevel=3)
        ids = [None if i is None else i.replace(" ", "_") for i in ids]
    ids = [f"motif_{n + 1}" if i is None else i for n, i in enumerate(ids)]
    return [(mid, _export_matrix(v, mid), _attrs_of(v)) for mid, (_, v) in zip(ids, items, strict=True)]


def _background(bg) -> tuple[float, ...] | None:
    """A background as (A, C, G, T) summing to 1, or None if invalid."""
    if bg is None:
        return None
    if isinstance(bg, _pandas.Series):
        bg = bg.to_dict()
    if isinstance(bg, Mapping):
        keys = [str(k).upper() for k in bg]
        if len(keys) != 4 or set(keys) != set(_ACGT):
            return None
        upper = {str(k).upper(): v for k, v in bg.items()}
        vals = [upper[b] for b in _ACGT]
    elif isinstance(bg, (str, bytes)):
        return None
    else:
        try:
            vals = list(bg)
        except TypeError:
            return None
    if len(vals) != 4:
        return None
    try:
        vals = [float(v) for v in vals]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and v >= 0 for v in vals):
        return None
    vals = [0.0 if v == 0 else v for v in vals]
    s = ((vals[0] + vals[1]) + vals[2]) + vals[3]
    if s == 0:
        return None
    if abs(s - 1.0) > 1e-4:
        vals = [v / s for v in vals]
    return tuple(vals)


def _strands(x) -> str | None:
    """The strands line value ("+ -" or "+"), or None if invalid."""
    s = _text(x)
    if s is None:
        return None
    s = s.replace(" ", "")
    if s in ("+-", "-+"):
        return "+ -"
    if s == "+":
        return "+"
    return None


def _common_attr(ms, name: str, parse, what: str):
    """The value the motifs agree on for an attribute, after ``parse``; None if
    none or if they disagree (with a warning naming ``what``)."""
    vals = []
    for _, _, attrs in ms:
        v = attrs.get(name)
        if v is not None:
            p = parse(v)
            if p is not None:
                vals.append(p)
    if not vals:
        return None
    if any(v != vals[0] for v in vals[1:]):
        warnings.warn(f"Motifs carry different {what}; writing the default", stacklevel=3)
        return None
    return vals[0]


def _homer_consensus(mat: _numpy.ndarray) -> str:
    """IUPAC consensus by HOMER's rule: a position's symbol covers every base
    within 0.15 of its most probable base."""
    out = []
    for row in mat:
        best = max(float(v) for v in row)
        bits = sum(b for b, v in zip((1, 2, 4, 8), row, strict=True) if (best - float(v)) < 0.15)
        out.append(_IUPAC[bits])
    return "".join(out)


def _homer_max_score(mat: _numpy.ndarray) -> float:
    """The highest log-odds score against a uniform background (natural log, as
    HOMER scores). ``math.log`` is the C library's ``log``, which R calls too."""
    s = 0.0
    for row in mat:
        s += math.log(max(float(v) for v in row) / 0.25)
    return s


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------




def gseq_write_meme(motifs, file, background=None, strands=None) -> None:
    """Write motifs to a MEME minimal motif format file.

    Each motif is written as a ``MOTIF <id> [<name>]`` line, a
    ``letter-probability matrix:`` line with ``alength``, ``w`` and, when the
    motif has them, ``nsites`` and ``E``, the matrix rows and, when the motif
    has one, a ``URL`` line. The file header declares the DNA alphabet, the
    strands and the background letter frequencies.

    Probabilities are written with 6 decimals, as MEME writes them, so a written
    motif reads back within 5e-7 of the original. ``nsites`` and E-values are
    written with 15 significant digits. The output is byte-identical to R
    misha's ``gseq.write_meme`` for the same input.

    Rows that do not sum to 1 are re-normalized (with a warning), as
    :func:`gseq_read_meme` does. Whitespace in IDs is replaced by ``_``;
    whitespace in names is collapsed to single spaces (other tools may keep only
    the first word of a name). Motifs with the same ID (in a list) are written as
    they are; :func:`gseq_read_meme` adds suffixes to tell them apart.

    Parameters
    ----------
    motifs : dict, list, pandas.DataFrame or numpy.ndarray
        A dict of motif matrices, such as returned by :func:`gseq_read_meme`,
        :func:`gseq_read_jaspar` or :func:`gseq_read_homer`, a list of them, or a
        single matrix. Each matrix has one row per position and columns ``A``,
        ``C``, ``G``, ``T`` (a numpy array or a DataFrame with default integer
        columns is taken in that order; a DataFrame indexed by ``A, C, G, T`` is
        transposed). Dict keys are the motif IDs; motifs without one are called
        ``motif_<i>``. Metadata is read from each DataFrame's ``.attrs``.
        The attrs ``name``, ``nsites``, ``E``, ``url``, ``strand`` and
        ``background`` are written when present.
    file : str or path-like
        Path of the file to write.
    background : dict, sequence or None
        Background letter frequencies: 4 values keyed ``A, C, G, T`` (or a
        sequence in that order), normalized to sum to 1. ``None`` (the default)
        uses the ``background`` attr the motifs share (as :func:`gseq_read_meme`
        returns it), and uniform frequencies otherwise.
    strands : str or None
        ``"+ -"`` (both strands) or ``"+"``. ``None`` (the default) uses the
        ``strand`` attr the motifs share, and ``"+ -"`` otherwise.

    See Also
    --------
    gseq_read_meme, gseq_write_jaspar, gseq_write_homer

    Examples
    --------
    >>> import pandas as pd
    >>> import pymisha as pm
    >>> m = pd.DataFrame([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]],
    ...                  columns=["A", "C", "G", "T"])
    >>> pm.gseq_write_meme({"M1": m}, "motifs.meme")  # doctest: +SKIP
    """
    _check_file(file)
    ms = _prepare(motifs)

    if background is None:
        bg = _common_attr(ms, "background", _background, "backgrounds") or (0.25, 0.25, 0.25, 0.25)
    else:
        bg = _background(background)
        if bg is None:
            raise ValueError(
                "'background' must be 4 non-negative frequencies named A, C, G, T "
                "(or in that order), not all zero"
            )

    if strands is None:
        st = _common_attr(ms, "strand", _strands, "strands") or "+ -"
    else:
        st = _strands(strands)
        if st is None:
            raise ValueError("'strands' must be \"+ -\" or \"+\"")

    out = [
        "MEME version 4", "",
        "ALPHABET= ACGT", "",
        f"strands: {st}", "",
        "Background letter frequencies",
        " ".join(f"{b} {_fmt_prob(v)}" for b, v in zip(_ACGT, bg, strict=True)),
        "",
    ]
    for mid, mat, attrs in ms:
        name = _text(attrs.get("name"))
        out.append(f"MOTIF {mid}" if name is None else f"MOTIF {mid} {name}")
        hdr = f"letter-probability matrix: alength= 4 w= {mat.shape[0]}"
        nsites = _number(attrs.get("nsites"))
        if nsites is not None:
            hdr += f" nsites= {_fmt_number(nsites)}"
        e_val = _number(attrs.get("E"))
        if e_val is not None:
            hdr += f" E= {_fmt_number(e_val)}"
        out.append(hdr)
        out.extend(" ".join(_fmt_prob(v) for v in row) for row in mat)
        url = _text(attrs.get("url"))
        if url is not None:
            out.append(f"URL {url}")
        out.append("")
    _write_lines(out, file)


def gseq_write_jaspar(motifs, file, nsites=None, format="jaspar", round_counts=True) -> None:
    """Write motifs to a JASPAR format file.

    Writes motifs as count matrices in the JASPAR format that
    :func:`gseq_read_jaspar` reads: a ``>ID name`` header line and one
    ``A [ ... ]`` row of counts per base. ``format="simple"`` writes the four
    rows of counts only, without headers or labels.

    JASPAR matrices hold counts, so each motif's probabilities are multiplied by
    its number of sites: the ``nsites`` argument, else the motif's ``nsites``
    attr (as :func:`gseq_read_jaspar` and :func:`gseq_read_meme` return it),
    else 100. Every position then sums to that number. Counts are rounded to
    whole numbers (halves to even), as JASPAR files hold them and as tools such
    as the MEME Suite's ``jaspar2meme`` and universalmotif require, so
    probabilities read back within about 0.5 / ``nsites``; the integer counts of
    a JASPAR file are written back unchanged. With an ``nsites`` of a few, whole
    positions can round to zero, which readers take as uniform.
    ``round_counts=False`` writes counts with up to 6 decimals and without
    trailing zeros instead, so that probabilities read back within
    1.5e-6 / ``nsites`` of the original. The output is byte-identical to R
    misha's ``gseq.write_jaspar``.

    The header names a motif by its ``name`` attr, or by its ID when it has none
    (JASPAR headers always carry a name). Rows that do not sum to 1 are
    re-normalized (with a warning). Whitespace in IDs is replaced by ``_``;
    whitespace in names is collapsed to single spaces (other tools may keep only
    the first word of a name). A simple-format file has no IDs:
    :func:`gseq_read_jaspar` names its motifs after the file.

    Parameters
    ----------
    motifs : dict, list, pandas.DataFrame or numpy.ndarray
        A dict of motif matrices, such as returned by :func:`gseq_read_meme`,
        :func:`gseq_read_jaspar` or :func:`gseq_read_homer`, a list of them, or a
        single matrix. Each matrix has one row per position and columns ``A``,
        ``C``, ``G``, ``T`` (a numpy array or a DataFrame with default integer
        columns is taken in that order; a DataFrame indexed by ``A, C, G, T`` is
        transposed). Dict keys are the motif IDs; motifs without one are called
        ``motif_<i>``. Metadata is read from each DataFrame's ``.attrs``.
        The attrs ``name`` and ``nsites`` are used when present.
    file : str or path-like
        Path of the file to write.
    nsites : float, sequence of float, or None
        Number of sites to scale probabilities to counts: a positive number, or
        one per motif. ``None`` (the default) uses each motif's ``nsites`` attr,
        and 100 for motifs without one.
    format : {"jaspar", "simple"}
        ``"jaspar"`` (headers and labeled rows) or ``"simple"`` (four rows of
        counts per motif).
    round_counts : bool
        Round counts to whole numbers (the default), or write them with up to 6
        decimals (``False``).

    See Also
    --------
    gseq_read_jaspar, gseq_write_meme, gseq_write_homer
    """
    _check_file(file)
    if format not in ("jaspar", "simple"):
        raise ValueError("'format' must be \"jaspar\" or \"simple\"")
    if not isinstance(round_counts, (bool, _numpy.bool_)):
        raise ValueError("'round_counts' must be True or False")
    ms = _prepare(motifs)
    ns_arg = None
    if nsites is not None:
        ns_arg = _recycle(nsites, len(ms), "nsites")
        if any(v <= 0 for v in ns_arg):
            raise ValueError("'nsites' must be positive")

    out: list[str] = []
    for i, (mid, mat, attrs) in enumerate(ms):
        ns = ns_arg[i] if ns_arg is not None else _number(attrs.get("nsites"))
        if ns is None or ns <= 0:
            ns = 100.0
        scaled = mat * ns
        if round_counts:
            scaled = _numpy.round(scaled)  # half to even, as R's round()
        counts = [[_fmt_count(v) for v in col] for col in scaled.T]
        width = max(len(c) for col in counts for c in col)
        rows = [" ".join(c.rjust(width) for c in col) for col in counts]
        if i > 0:
            out.append("")
        if format == "jaspar":
            name = _text(attrs.get("name")) or mid
            out.append(f">{mid} {name}")
            out.extend(f"{b}  [ {r} ]" for b, r in zip(_ACGT, rows, strict=True))
        else:
            out.extend(rows)
    _write_lines(out, file)


def gseq_write_homer(motifs, file, threshold=None, threshold_frac=0.6) -> None:
    """Write motifs to a HOMER motif format file.

    Writes motifs in the HOMER ``.motif`` format that :func:`gseq_read_homer`
    and HOMER read: a header line
    ``>consensus<TAB>name<TAB>log-odds threshold[<TAB>log p-value]`` and one
    tab-separated row of A, C, G, T probabilities per position.

    The consensus is the motif's ``consensus`` attr (as :func:`gseq_read_homer`
    returns it) when it has one of the motif's width, and otherwise is computed
    by HOMER's rule: each position's IUPAC symbol covers every base whose
    probability is within 0.15 of the highest. The name is the motif's ``name``
    attr, or its ID when it has none. :func:`gseq_read_homer` names motifs by
    their consensus, so IDs that are not used as names are not kept.

    The log-odds threshold (natural log against a uniform background, as HOMER
    scores) is the ``threshold`` argument, else the motif's
    ``log_odds_threshold`` attr, else ``threshold_frac`` times the motif's
    highest possible score. The log p-value is written when the motif has a
    ``log_p_value`` attr.

    Probabilities, thresholds and log p-values are written with 6 decimals, so a
    written motif reads back within 5e-7 of the original. Rows that do not sum
    to 1 are re-normalized (with a warning). Tabs and other whitespace in names
    are collapsed to single spaces. The output is byte-identical to R misha's
    ``gseq.write_homer``.

    Parameters
    ----------
    motifs : dict, list, pandas.DataFrame or numpy.ndarray
        A dict of motif matrices, such as returned by :func:`gseq_read_meme`,
        :func:`gseq_read_jaspar` or :func:`gseq_read_homer`, a list of them, or a
        single matrix. Each matrix has one row per position and columns ``A``,
        ``C``, ``G``, ``T`` (a numpy array or a DataFrame with default integer
        columns is taken in that order; a DataFrame indexed by ``A, C, G, T`` is
        transposed). Dict keys are the motif IDs; motifs without one are called
        ``motif_<i>``. Metadata is read from each DataFrame's ``.attrs``.
        The attrs ``name``, ``consensus``, ``log_odds_threshold`` and
        ``log_p_value`` are used when present.
    file : str or path-like
        Path of the file to write.
    threshold : float, sequence of float, or None
        Log-odds detection threshold: a number, or one per motif. ``None`` (the
        default) uses each motif's ``log_odds_threshold`` attr, and
        ``threshold_frac`` of its highest score for motifs without one.
    threshold_frac : float
        Fraction of a motif's highest log-odds score used as its threshold when
        neither ``threshold`` nor the motif gives one.

    See Also
    --------
    gseq_read_homer, gseq_write_meme, gseq_write_jaspar
    """
    _check_file(file)
    ms = _prepare(motifs)
    thr_arg = None if threshold is None else _recycle(threshold, len(ms), "threshold")
    frac = _number(threshold_frac)
    if frac is None or isinstance(threshold_frac, bool):
        raise ValueError("'threshold_frac' must be a finite number")

    out: list[str] = []
    for i, (mid, mat, attrs) in enumerate(ms):
        cons = _text(attrs.get("consensus"))
        if cons is None or " " in cons or len(cons) != mat.shape[0]:
            cons = _homer_consensus(mat)
        name = _text(attrs.get("name")) or mid
        thr = thr_arg[i] if thr_arg is not None else _number(attrs.get("log_odds_threshold"))
        if thr is None:
            thr = frac * _homer_max_score(mat)
        hdr = f">{cons}\t{name}\t{_fmt_prob(thr)}"
        log_p = _number(attrs.get("log_p_value"))
        if log_p is not None:
            hdr += f"\t{_fmt_prob(log_p)}"
        out.append(hdr)
        out.extend("\t".join(_fmt_prob(v) for v in row) for row in mat)
    _write_lines(out, file)
