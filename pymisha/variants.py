"""gextract_variants: a track expression under single-base substitutions.

Port of R misha's ``R/compute-variants.R``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as _numpy
import pandas as _pandas

from . import _shared
from ._safe_eval import compile_safe_expression
from ._shared import CONFIG, _checkroot, _coerce_score_thresh, _df2pymisha, _pymisha
from .expr import _expr_safe_name, _parse_expr_vars
from .extract import _build_vtracks_dict, _can_vtracks_use_cpp, gextract
from .intervals import _normalize_chroms, gintervals_all
from .sequence import gseq_extract
from .vtracks import _compute_vtrack_values

_BASES = ("A", "C", "G", "T")
# The three alternative bases of each reference base, in _BASES order.
_ALTS = {"A": ("C", "G", "T"), "C": ("A", "G", "T"), "G": ("A", "C", "T"), "T": ("A", "C", "G")}


def _expr_vtracks(expr: str) -> set[str]:
    _, _, used_vtracks, _ = _parse_expr_vars(
        expr, _shared._track_names_set(), set(_shared._VTRACKS)
    )
    return used_vtracks


def _gvariants_reach(expr: str) -> _numpy.ndarray:
    """One row (lo, hi) per sequence vtrack in *expr*: a bin [s, e) of that
    vtrack reads [s + lo, e + hi)."""
    seq_vtracks = {vt for vt in _expr_vtracks(expr) if _shared._VTRACKS[vt].get("src") is None}
    x = _pymisha.pm_gvariants_reach(expr, _build_vtracks_dict(seq_vtracks))
    if not len(x):
        raise ValueError(
            f'Expression "{expr}" has no sequence virtual track, so no substitution can change it'
        )
    return _numpy.asarray(x, dtype=_numpy.int64).reshape(-1, 2)


def _gvariants_eval(
    expr: str,
    chroms: Any,
    starts: Any,
    ends: Any,
    pos: Any,
    base: Any,
) -> _numpy.ndarray:
    """The value of *expr* on each row, with base *base* (0..3 = A, C, G, T;
    -1 = none) at position *pos* while the row is computed. Rows sorted by
    chromosome and start.

    Evaluated the way gextract() evaluates *expr*: in the C++ scanner, or, when
    a virtual track needs pymisha's Python path (a filter, or a source other
    than the sequence or a track), variable by variable and then in Python.
    """
    rows = _pandas.DataFrame({"chrom": [str(c) for c in chroms], "start": starts, "end": ends})
    pos = _numpy.asarray(pos, dtype=_numpy.int64)
    base = _numpy.asarray(base, dtype=_numpy.int64)
    expr_eval, tracks, vtracks, _ = _parse_expr_vars(
        expr, _shared._track_names_set(), set(_shared._VTRACKS)
    )
    if not vtracks or _can_vtracks_use_cpp(vtracks):
        out: _numpy.ndarray = _pymisha.pm_gextract_variants(
            expr, _df2pymisha(rows), pos, base, _build_vtracks_dict(vtracks) if vtracks else None
        )
        return out

    ns: dict[str, Any] = {}
    no_sub = _numpy.full(len(rows), -1, dtype=_numpy.int64)
    for t in tracks:
        ns[_expr_safe_name(t)] = _pymisha.pm_gextract_variants(t, _df2pymisha(rows), pos, no_sub, None)
    for vt in vtracks:
        ns[_expr_safe_name(vt)] = _numpy.asarray(
            _compute_vtrack_values(vt, rows, subs=(pos, base)), dtype=float
        )
    code = compile_safe_expression(expr_eval, {"np", "numpy", "CHROM", "START", "END", *ns})
    coords = {
        "CHROM": rows["chrom"].to_numpy(dtype=object),
        "START": rows["start"].to_numpy(dtype=_numpy.int64),
        "END": rows["end"].to_numpy(dtype=_numpy.int64),
    }
    out = _numpy.empty(len(rows), dtype=float)
    step = int(CONFIG.get("eval_buf_size", 1000) or 1000)  # type: ignore[call-overload]
    for lo in range(0, len(rows), step):
        sl = slice(lo, lo + step)
        local_ns = {"np": _numpy, "numpy": _numpy, **{k: v[sl] for k, v in {**coords, **ns}.items()}}
        val = eval(code, {"__builtins__": {}}, local_ns)
        out[sl] = _numpy.full(len(out[sl]), val) if _numpy.isscalar(val) else _numpy.asarray(val, dtype=float)
    return out


def gextract_variants(
    expr: str,
    intervals: _pandas.DataFrame | str,
    variants: _pandas.DataFrame | None = None,
    iterator: Any = None,
    score_thresh: float | None = None,
    colnames: str | None = None,
) -> _pandas.DataFrame | None:
    """Evaluate a track expression under single-base substitutions.

    For each iterator bin, evaluates *expr* once on the reference genome and
    once for every single-base substitution the bin's sequence virtual tracks
    read: all three alternative bases at each base in reach (the default), or
    the variants listed in *variants*.

    A bin ``[s, e)`` of a sequence virtual track reads
    ``[s + sshift, e + eshift + W - 1)``, where the shifts come from
    :func:`gvtrack_iterator` and ``W - 1`` is added when the track extends
    (``extend=True``, motif or model width ``W``). ``pwm.edit_distance``
    with ``max_indels`` reads further: it ends ``max_indels`` bases later. A
    substitution is paired with every bin whose
    reach, the union over the expression's sequence virtual tracks, contains
    it. Tracks and non-sequence virtual tracks in *expr* are read as usual, so
    they contribute the same value to both columns. The substituted base takes
    the case of the reference base, so ``masked.*`` tracks see the same mask.
    Only single-base substitutions are supported.

    The substituted values are computed in one process. For genome-scale runs,
    split *intervals* by chromosome and run the pieces in parallel.

    Run genome-wide, every bin returns about three rows per base in reach. To
    find the bins one substitution can move across a threshold, screen first
    with a bound (for example from ``potts.max`` virtual tracks with
    ``max_edits=1``) and pass the result as *intervals*, with *score_thresh*.

    Parameters
    ----------
    expr : str
        A track expression with at least one sequence virtual track.
    intervals : DataFrame or str
        Genomic scope, as in :func:`gextract`.
    variants : DataFrame, optional
        None for every substitution in reach, or a DataFrame with columns
        ``chrom``, ``start`` (0-based), ``ref`` and ``alt``, such as the output
        of :func:`gintervals_import_vcf`. A variant whose ``ref`` does not
        match the genome gets a warning and NaN in ``<name>.alt``;
        ``<name>.ref`` keeps the reference value.
    iterator : int or str, optional
        Track expression iterator, as in :func:`gextract`.
    score_thresh : float, optional
        If given, keep only the rows where the reference value and the
        substituted value fall on different sides of it.
    colnames : str, optional
        Name used for the value columns, ``<name>.ref`` and ``<name>.alt``;
        the expression itself by default.

    Returns
    -------
    DataFrame or None
        One row per (bin, substitution): the bin's ``chrom``, ``start`` and
        ``end``; the substitution's ``pos`` (0-based), ``ref`` and ``alt``;
        ``<name>.ref`` and ``<name>.alt``; and ``intervalID`` as in
        :func:`gextract`. None when there is no row.

    See Also
    --------
    gextract, gvtrack_create, gintervals_import_vcf

    Examples
    --------
    >>> import numpy as np
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pssm = np.array([[0.7, 0.1, 0.1, 0.1],
    ...                  [0.1, 0.7, 0.1, 0.1],
    ...                  [0.1, 0.1, 0.7, 0.1]])
    >>> pm.gvtrack_create("motif", None, "pwm.max", pssm=pssm, bidirect=False)
    >>> res = pm.gextract_variants("motif", pm.gintervals("1", 100, 102), iterator=1)
    >>> res.columns.tolist()
    ['chrom', 'start', 'end', 'pos', 'ref', 'alt', 'motif.ref', 'motif.alt', 'intervalID']
    >>> pm.gvtrack_rm("motif")
    """
    _checkroot()
    return _gextract_variants_impl(expr, intervals, variants, iterator, score_thresh, colnames)


def _gextract_variants_impl(
    expr: Any,
    intervals: Any,
    variants: Any,
    iterator: Any,
    score_thresh: Any,
    colnames: Any,
    chunk: int | None = None,
) -> _pandas.DataFrame | None:
    if not isinstance(expr, str):
        raise ValueError("gextract_variants takes a single track expression")
    name = expr if colnames is None else colnames
    if not isinstance(name, str):
        raise ValueError("colnames must be a single string")
    if score_thresh is not None:
        score_thresh = _coerce_score_thresh(score_thresh)
    reach = _gvariants_reach(expr)
    if variants is not None:
        variants = _gvariants_check(variants)
        # Checked once, against the genome, whether or not any bin reads them.
        if len(variants):
            genome_ref = gseq_extract(_pandas.DataFrame({
                "chrom": variants["chrom"], "start": variants["pos"], "end": variants["pos"] + 1,
            }))
            variants["ok"] = _numpy.char.upper(_numpy.asarray(genome_ref, dtype=str)) == variants["ref"].to_numpy()
        else:
            variants["ok"] = _numpy.zeros(0, dtype=bool)
        n_bad = int((~variants["ok"]).sum())
        if n_bad:
            warnings.warn(
                f"{n_bad} of {len(variants)} variants have a ref that does not match the "
                "genome; their substituted values are NA",
                stacklevel=3,
            )

    # vars={}: the expression must not pick up this function's locals. No
    # progress bar: this is the short part of the run.
    bins = gextract(expr, intervals, iterator=iterator, colnames=["value"], vars={}, progress=False)
    if bins is None or not len(bins):
        return None
    bins = bins.reset_index(drop=True)
    if chunk is None:
        # About 3e6 (bin, substitution) rows per chunk, whatever the reach.
        span = int(reach[:, 1].max() - reach[:, 0].min() + (bins["end"] - bins["start"]).max())
        span = max(1, span)
        chunk = max(1, 3_000_000 // (3 * span))
    chrom_rank = {c: i for i, c in enumerate(gintervals_all()["chrom"].astype(str))}

    out = []
    for frm in range(0, len(bins), chunk):
        b = bins.iloc[frm:frm + chunk].reset_index(drop=True)
        b_chrom = b["chrom"].astype(str).to_numpy()
        p = _gvariants_bin_positions(b, reach, with_ref=variants is None)
        if variants is None:
            r = _gvariants_saturate(p)
            r["ok"] = True
        else:
            if not len(p):
                continue
            p["chrom"] = b_chrom[p["bin"].to_numpy()]
            # Only the variants this chunk can reach go into the join.
            v = variants[
                variants["chrom"].isin(set(p["chrom"]))
                & (variants["pos"] >= p["pos"].min())
                & (variants["pos"] <= p["pos"].max())
            ]
            r = p.merge(v, on=["chrom", "pos"])
            r = r.sort_values(["bin", "pos", "alt"], kind="mergesort")
            r = r[["bin", "pos", "ref", "alt", "ok"]].reset_index(drop=True)
        if not len(r):
            continue
        rbin = r["bin"].to_numpy()
        alt_value = _numpy.full(len(r), _numpy.nan)
        e = _numpy.flatnonzero(r["ok"].to_numpy(dtype=bool))
        if len(e):
            # The engine needs rows sorted by chromosome and start; gextract's
            # bins need not be (overlapping intervals repeat a stretch).
            eb = rbin[e]
            ranks = _numpy.array([chrom_rank[c] for c in b_chrom], dtype=_numpy.int64)
            e = e[_numpy.lexsort((b["start"].to_numpy()[eb], ranks[eb]))]
            eb = rbin[e]
            alt_codes = _pandas.Series(r["alt"].to_numpy()[e]).map({x: i for i, x in enumerate(_BASES)})
            alt_value[e] = _gvariants_eval(
                expr, b_chrom[eb], b["start"].to_numpy()[eb], b["end"].to_numpy()[eb],
                r["pos"].to_numpy()[e], alt_codes.to_numpy(),
            )
        sel = b.iloc[rbin].reset_index(drop=True)
        res = _pandas.DataFrame({
            "chrom": sel["chrom"],
            "start": sel["start"],
            "end": sel["end"],
            "pos": r["pos"].to_numpy(dtype=_numpy.int64),
            "ref": r["ref"].to_numpy(),
            "alt": r["alt"].to_numpy(),
            "ref_value": sel["value"],
            "alt_value": alt_value,
            "intervalID": sel["intervalID"],
        })
        if score_thresh is not None:
            rv = res["ref_value"].to_numpy(dtype=float)
            av = res["alt_value"].to_numpy(dtype=float)
            with _numpy.errstate(invalid="ignore"):
                keep = ~_numpy.isnan(rv) & ~_numpy.isnan(av) & ((rv >= score_thresh) != (av >= score_thresh))
            res = res[keep]
        out.append(res)

    if not out:
        return None
    res = _pandas.concat(out, ignore_index=True)
    if not len(res):
        return None
    return res.rename(columns={"ref_value": f"{name}.ref", "alt_value": f"{name}.alt"})


def _gvariants_bin_positions(
    bins: _pandas.DataFrame, reach: _numpy.ndarray, with_ref: bool = True
) -> _pandas.DataFrame:
    """One row per (bin, position) a bin reads, sorted, with the genome base
    (upper case) at it unless *with_ref* is False. *bins* has chrom, start,
    end; *reach* is _gvariants_reach()'s array."""
    sizes = gintervals_all()
    size_of = dict(zip(sizes["chrom"].astype(str), sizes["end"].astype(_numpy.int64), strict=True))
    chrom_size = _numpy.array([size_of[c] for c in bins["chrom"].astype(str)], dtype=_numpy.int64)
    starts = bins["start"].to_numpy(dtype=_numpy.int64)
    ends = bins["end"].to_numpy(dtype=_numpy.int64)

    all_bin, all_pos = [], []
    for lo, hi in reach:
        frm = _numpy.maximum(starts + lo, 0)
        n = _numpy.maximum(_numpy.minimum(ends + hi, chrom_size) - frm, 0)
        bin_idx = _numpy.repeat(_numpy.arange(len(bins)), n)
        offset = _numpy.arange(int(n.sum())) - _numpy.repeat(_numpy.cumsum(n) - n, n)
        all_bin.append(bin_idx)
        all_pos.append(frm[bin_idx] + offset)
    pbin = _numpy.concatenate(all_bin)
    ppos = _numpy.concatenate(all_pos)
    order = _numpy.lexsort((ppos, pbin))
    pbin, ppos = pbin[order], ppos[order]
    keep = _numpy.ones(len(pbin), dtype=bool)
    keep[1:] = (pbin[1:] != pbin[:-1]) | (ppos[1:] != ppos[:-1])
    p = _pandas.DataFrame({"bin": pbin[keep], "pos": ppos[keep]})
    if not with_ref:
        return p

    span_lo = _numpy.maximum(starts + reach[:, 0].min(), 0)
    span_hi = _numpy.minimum(ends + reach[:, 1].max(), chrom_size)
    has = _numpy.flatnonzero(span_hi > span_lo)
    seq_off = _numpy.zeros(len(bins), dtype=_numpy.int64)
    genome = b""
    if len(has):
        seqs = gseq_extract(_pandas.DataFrame({
            "chrom": bins["chrom"].astype(str).to_numpy()[has],
            "start": span_lo[has],
            "end": span_hi[has],
        }))
        lens = _numpy.array([len(s) for s in seqs], dtype=_numpy.int64)
        seq_off[has] = _numpy.cumsum(lens) - lens
        genome = "".join(seqs).upper().encode()
    pb = p["bin"].to_numpy()
    idx = seq_off[pb] + p["pos"].to_numpy() - span_lo[pb]
    p["ref"] = _numpy.frombuffer(genome, dtype="S1")[idx].astype(str) if len(idx) else _numpy.zeros(0, dtype=str)
    return p


def _gvariants_saturate(p: _pandas.DataFrame) -> _pandas.DataFrame:
    """Three rows per A/C/G/T position, one per alternative base."""
    p = p[p["ref"].isin(_BASES)]
    ref = p["ref"].to_numpy()
    return _pandas.DataFrame({
        "bin": _numpy.repeat(p["bin"].to_numpy(), 3),
        "pos": _numpy.repeat(p["pos"].to_numpy(), 3),
        "ref": _numpy.repeat(ref, 3),
        "alt": [a for x in ref for a in _ALTS[x]],
    })


def _gvariants_check(variants: Any) -> _pandas.DataFrame:
    """chrom (canonical), pos, ref, alt (upper case), one row per variant."""
    if not isinstance(variants, _pandas.DataFrame) or not {"chrom", "start", "ref", "alt"} <= set(
        variants.columns
    ):
        raise ValueError(
            "variants must be a DataFrame with columns chrom, start, ref and alt "
            "(gintervals_import_vcf() output works)"
        )
    ref = variants["ref"].astype(str).str.upper().to_numpy()
    alt = variants["alt"].astype(str).str.upper().to_numpy()
    bad = _numpy.flatnonzero(~_numpy.isin(ref, _BASES) | ~_numpy.isin(alt, _BASES))
    if len(bad):
        i = bad[0]
        raise ValueError(
            f"variants row {variants.index[i]}: only single-base substitutions are supported "
            f'(ref and alt each one of A, C, G, T); got ref "{variants["ref"].iloc[i]}", '
            f'alt "{variants["alt"].iloc[i]}"'
        )
    same = _numpy.flatnonzero(ref == alt)
    if len(same):
        raise ValueError(f"variants row {variants.index[same[0]]}: ref and alt are the same base")
    start = variants["start"]
    if (
        not _pandas.api.types.is_numeric_dtype(start)
        or _pandas.api.types.is_bool_dtype(start)
        or start.isna().any()
        or (start < 0).any()
        or (start != _numpy.floor(start)).any()
    ):
        raise ValueError("variants: start must be a non-negative whole number (0-based)")
    start_v = start.to_numpy(dtype=_numpy.int64)
    chrom = _numpy.asarray(_normalize_chroms(variants["chrom"].tolist()), dtype=object)
    sizes = gintervals_all()
    size_of = dict(zip(sizes["chrom"].astype(str), sizes["end"].astype(_numpy.int64), strict=True))
    past = _numpy.flatnonzero(start_v >= _numpy.array([size_of[c] for c in chrom], dtype=_numpy.int64))
    if len(past):
        i = past[0]
        raise ValueError(f"variants row {variants.index[i]}: start {start_v[i]} is past the end of {chrom[i]}")
    return _pandas.DataFrame({"chrom": chrom, "pos": start_v, "ref": ref, "alt": alt})
