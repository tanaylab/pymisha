"""Virtual track utilities and API."""

from __future__ import annotations

import hashlib
import math as _math
import numbers
import warnings as _warnings
from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from . import _shared
from ._log import get_logger
from ._shared import (
    CONFIG,
    _check_breaks_na,
    _check_intervals_strand,
    _check_spat_bounds,
    _checkroot,
    _chrom_names,
    _coerce_pssm_matrix,
    _coerce_score_thresh,
    _df2pymisha,
    _is_na,
    _numpy,
    _pandas,
    _potts_params,
    _pymisha,
)

_logger = get_logger(__name__)

_FILTER_PASSTHROUGH_FUNCS = {"distance", "distance.center", "distance.edge"}
_FILTER_WEIGHTED_FUNCS = {"avg", "mean", "coverage", "kmer.frac", "masked.frac"}
_FILTER_ADDITIVE_FUNCS = {"sum", "kmer.count", "masked.count", "pwm.count", "potts.count",
                          "neighbor.count"}
_FILTER_MIN_FUNCS = {"min"}
_FILTER_MAX_FUNCS = {"max", "pwm.max", "potts.max"}
_FILTER_STDDEV_FUNCS = {"stddev", "std"}
_FILTER_QUANTILE_FUNCS = {"quantile"}
_FILTER_NEAREST_FUNCS = {"nearest"}
_FILTER_EXISTS_FUNCS = {"exists"}
_FILTER_SIZE_FUNCS = {"size"}
_FILTER_FIRST_FUNCS = {"first"}
_FILTER_LAST_FUNCS = {"last"}
_FILTER_SAMPLE_FUNCS = {"sample"}
_FILTER_FIRST_POS_ABS_FUNCS = {"first.pos.abs"}
_FILTER_FIRST_POS_REL_FUNCS = {"first.pos.relative"}
_FILTER_LAST_POS_ABS_FUNCS = {"last.pos.abs"}
_FILTER_LAST_POS_REL_FUNCS = {"last.pos.relative"}
_FILTER_MAX_POS_ABS_FUNCS = {"max.pos.abs"}
_FILTER_MAX_POS_REL_FUNCS = {"max.pos.relative"}
_FILTER_MIN_POS_ABS_FUNCS = {"min.pos.abs"}
_FILTER_MIN_POS_REL_FUNCS = {"min.pos.relative"}
_FILTER_SAMPLE_POS_ABS_FUNCS = {"sample.pos.abs"}
_FILTER_SAMPLE_POS_REL_FUNCS = {"sample.pos.relative"}
_FILTER_LOGSUMEXP_FUNCS = {"pwm", "potts", "lse"}
_FILTER_WEIGHTED_SUM_FUNCS = {"weighted.sum"}
_FILTER_GLOBAL_PERCENTILE_FUNCS = {
    "global.percentile",
    "global.percentile.min",
    "global.percentile.max",
}
# Each of these picks the position from the unmasked part carrying the best
# score (see _compute_filtered_max_pos_by_score).
_FILTER_MAX_POS_BY_SCORE_FUNCS = {"pwm.max.pos", "potts.max.pos"}
_FILTER_EDIT_DISTANCE_FUNCS = {
    "pwm.edit_distance",
    "pwm.edit_distance.pos",
    "pwm.max.edit_distance",
    "pwm.edit_distance.lse",
    "pwm.edit_distance.lse.pos",
    "pwm.n_mutations",
}
# Keyword arguments each func accepts, mirroring R misha's .VTRACK_PARAM_HANDLERS
# (5.11.15). A func that is not listed accepts none: a misspelled key used to be
# dropped and silently defaulted, which is indistinguishable from the parameter
# having had no effect. `filter` is a pymisha addition accepted everywhere.
_PWM_PARAMS = {
    "pssm", "bidirect", "prior", "extend", "strand",
    "spat_factor", "spat_bin", "spat_min", "spat_max", "score_thresh",
}
_EDIT_DISTANCE_PARAMS = {
    "pssm", "score_thresh", "max_edits", "max_indels", "score_min", "score_max",
    "bidirect", "prior", "extend", "strand", "direction",
}
_LSE_EDIT_DISTANCE_PARAMS = _EDIT_DISTANCE_PARAMS - {"max_indels"}
_EDIT_DISTANCE_FUNCS_MAX_EDITS = {"pwm.edit_distance", "pwm.edit_distance.pos", "pwm.max.edit_distance",
                                  "pwm.edit_distance.lse", "pwm.edit_distance.lse.pos"}
_N_MUTATIONS_PARAMS = _EDIT_DISTANCE_PARAMS - {"max_edits", "max_indels"}
# The allowlist is deliberately wider than the parameters this family uses:
# `width`, `pair_strength`, `attr` and `link` are accepted and ignored so that a
# fitted Potts model - whose fields are exactly `e`, `J`, `pairs`,
# `pair_strength`, `width`, `intercept`, `attr`, `link` - can be handed over
# verbatim as `params`, with no subsetting at the call site.
# _coerce_potts_model() cross-checks `width` against the row count of `e`.
_POTTS_PARAMS = {
    "e", "J", "pairs", "intercept", "bidirect", "extend", "strand",
    "score_thresh", "width", "pair_strength", "attr", "link", "max_edits",
}
_POTTS_FUNCS = ("potts", "potts.max", "potts.max.pos", "potts.count")
_PWM_FUNCS = ("pwm", "pwm.max", "pwm.max.pos", "pwm.count")
_ACCEPTED_VTRACK_PARAMS = {
    "pwm": _PWM_PARAMS,
    "pwm.max": _PWM_PARAMS,
    "pwm.max.pos": _PWM_PARAMS,
    "pwm.count": _PWM_PARAMS,
    "potts": _POTTS_PARAMS,
    "potts.max": _POTTS_PARAMS,
    "potts.max.pos": _POTTS_PARAMS,
    "potts.count": _POTTS_PARAMS,
    "kmer.count": {"kmer", "extend", "strand"},
    "kmer.frac": {"kmer", "extend", "strand"},
    "pwm.edit_distance": _EDIT_DISTANCE_PARAMS,
    "pwm.edit_distance.pos": _EDIT_DISTANCE_PARAMS,
    "pwm.max.edit_distance": _EDIT_DISTANCE_PARAMS,
    "pwm.edit_distance.lse": _LSE_EDIT_DISTANCE_PARAMS,
    "pwm.edit_distance.lse.pos": _LSE_EDIT_DISTANCE_PARAMS,
    "pwm.n_mutations": _N_MUTATIONS_PARAMS,
}
# R spells these with a dot; pymisha reads either.
_DOTTED_PARAM_ALIASES = {"score.thresh": "score_thresh", "score.min": "score_min",
                         "score.max": "score_max"}
# masked.count / masked.frac warn rather than raise, as R misha does.
_WARN_ON_PARAMS_FUNCS = {"masked.count", "masked.frac"}


def _check_params_dots_collision(func: str, params: Any, kwargs: dict[str, Any]) -> None:
    """Port of R misha's ``.vtrack_check_params_dots_collision``: ``params``
    and keyword arguments are two spellings of the same parameter set, and
    the keyword arguments would be ignored, so a call may not use both."""
    dots = sorted(k for k in kwargs if k != "filter")
    if params is None or not dots:
        return
    raise ValueError(
        f"function '{func}': parameters supplied both via 'params' and as named "
        f"argument{'s' if len(dots) > 1 else ''} {', '.join(repr(k) for k in dots)}. These "
        "are two alternative ways of passing the same parameters - the named arguments "
        "would be ignored, so supply either 'params' or the named arguments, not both."
    )


def _kmer_params(name: str, func: str, params: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Port of R misha's ``.vtrack_params_kmer``: the ``kmer``, ``extend`` and
    ``strand`` of a kmer.count / kmer.frac vtrack, given as ``params`` (a dict,
    R's list) or as keyword arguments."""
    from .sequence import gseq_revcomp

    _check_params_dots_collision(func, params, kwargs)
    dots = {k: v for k, v in kwargs.items() if k != "filter"}
    if params is not None:
        if isinstance(params, str):
            # pymisha also takes the k-mer itself as params.
            params = {"kmer": params}
        if not isinstance(params, dict) or "kmer" not in params:
            raise ValueError("kmer function requires a dict with at least 'kmer' parameter")
        kp = dict(params)
    elif "kmer" in dots:
        kp = dots
    else:
        raise ValueError("kmer functions require a 'kmer' parameter")

    _check_vtrack_params(func, kp)
    # R: extend TRUE or FALSE only; strand -1, 0 or 1 (a number, not a bool)
    if kp.get("extend") is None:
        kp["extend"] = True
    elif not isinstance(kp["extend"], (bool, _numpy.bool_)):
        raise ValueError(f"Virtual track {name}: extend parameter must be True or False")
    kp["extend"] = bool(kp["extend"])
    if kp.get("strand") is None:
        kp["strand"] = 0
    elif not _is_num(kp["strand"]) or kp["strand"] not in (-1, 0, 1):
        raise ValueError(f"Virtual track {name}: strand parameter must be -1, 0, or 1")

    kmer = kp.get("kmer")
    if not isinstance(kmer, str):
        raise ValueError("kmer parameter must be a single string")
    if not kmer:
        raise ValueError("kmer sequence cannot be empty")
    if kmer == gseq_revcomp(kmer) and kp["strand"] == 0:
        _warnings.warn(
            f"kmer sequence '{kmer}' is palindromic, please set strand to 1 or -1 to "
            "avoid double counting",
            stacklevel=3,
        )
    return kp


def _quantile_param(payload: dict) -> float:
    # 0 is a valid percentile; only a missing param defaults to the median.
    q = payload.get("params")
    return 0.5 if q is None else float(q)


def _kmer_possible_positions(seg_df: _pandas.DataFrame, payload: dict) -> _numpy.ndarray:
    """R misha's ``KmerCounter::possible_positions`` for each part: the anchors in it whose
    k-mer fits in the sequence read for it (read k-1 bases past its end with extend, up to
    the chromosome end), on each strand scored."""
    from .intervals import gintervals_all

    k = len(payload["kmer"])
    all_df = gintervals_all()
    sizes = seg_df["chrom"].astype(str).map(dict(zip(all_df["chrom"].astype(str), all_df["end"], strict=True)))
    starts = seg_df["start"].to_numpy(dtype=_numpy.int64)
    ends = seg_df["end"].to_numpy(dtype=_numpy.int64)
    read_end = _numpy.minimum(ends + (k - 1 if payload.get("extend", True) else 0), sizes.to_numpy(dtype=_numpy.int64))
    per_strand: _numpy.ndarray = _numpy.maximum(_numpy.minimum(ends, read_end - k + 1) - starts, 0)
    return per_strand * 2 if int(payload.get("strand", 0)) == 0 else per_strand


def _check_vtrack_params(func: str | None, kwargs: dict) -> None:
    """Reject kwargs that ``func`` does not accept (R misha 5.11.15)."""
    keys = {k for k in kwargs if k != "filter"}
    if not keys:
        return

    func_lc = str(func).lower() if func is not None else None
    if func_lc in _WARN_ON_PARAMS_FUNCS:
        _warnings.warn(
            f"{func} does not accept parameters; ignoring {sorted(keys)}",
            stacklevel=3,
        )
        return

    # Annotated and None-guarded for mypy: func_lc is str | None while the table
    # is keyed by str, and a bare empty-set default gave mypy nothing to infer
    # the element type from. Behaviour is unchanged - no key is None, so the old
    # .get(None, ...) already returned the empty set - and `accepted` is only
    # ever tested for membership, sorted and truthiness, never mutated.
    accepted: set[str] = set()
    if func_lc is not None:
        accepted = _ACCEPTED_VTRACK_PARAMS.get(func_lc, set())
    unknown = sorted(k for k in keys
                     if _DOTTED_PARAM_ALIASES.get(k, k) not in accepted)
    if not unknown:
        return

    keylist = ", ".join(repr(k) for k in unknown)
    plural = "s" if len(unknown) > 1 else ""
    if not accepted:
        raise ValueError(
            f"function {func!r} does not accept parameter{plural} {keylist}; "
            "this function takes no additional keyword arguments"
        )
    raise ValueError(
        f"function {func!r} does not accept parameter{plural} {keylist}; "
        f"accepted parameters are: {', '.join(sorted(accepted))}"
    )


def _convert_spat_range(config: dict) -> None:
    """R misha's .vtrack_params_pwm conversion of spat_min/spat_max, in place.

    Both are 1-based positions in the scanned window (checked by
    _check_spat_bounds). The scorer takes the 0-based first and last anchor
    START, so spat_min becomes spat_min - 1 and spat_max becomes
    (spat_max - 1) - motif_length + 1: the last anchor whose motif still ends at
    spat_max. Either may be given alone - the C++ side then leaves the other end
    open - and neither needs spat_factor.
    """
    if config.get("spat_min") is None and config.get("spat_max") is None:
        return
    pssm = config.get("pssm")
    if pssm is None:
        raise ValueError("pwm function requires a 'pssm' matrix parameter")
    smin, smax = _check_spat_bounds(config.get("spat_min"), config.get("spat_max"), len(pssm))
    # R's arithmetic, in R's order, truncated toward zero like as.integer().
    if smin is not None:
        config["spat_min"] = int(smin - 1)
    if smax is not None:
        config["spat_max"] = int((smax - 1) - len(pssm) + 1)


_DF_INTERVAL_FUNCS = {"distance", "distance.center", "distance.edge", "coverage", "neighbor.count"}

# R's TrackExpressionVars::Track_var::FUNC_NAMES (functions of a track source),
# plus pymisha's aliases "mean" and "std" (same values as avg and stddev).
_R_TRACK_FUNCS = {
    "avg", "min", "max", "nearest", "stddev", "sum", "lse", "quantile",
    "global.percentile", "global.percentile.min", "global.percentile.max",
    "weighted.sum", "area", "pwm", "pwm.max", "pwm.max.pos", "pwm.count", "kmer.count", "kmer.frac",
    "masked.count", "masked.frac",
    "max.pos.abs", "max.pos.relative", "min.pos.abs", "min.pos.relative",
    "exists", "size", "sample", "sample.pos.abs", "sample.pos.relative",
    "first", "first.pos.abs", "first.pos.relative", "last", "last.pos.abs", "last.pos.relative",
    "pwm.edit_distance", "pwm.edit_distance.pos", "pwm.max.edit_distance",
    "pwm.edit_distance.lse", "pwm.edit_distance.lse.pos", "pwm.n_mutations",
    "potts", "potts.max", "potts.max.pos", "potts.count",
    "mean", "std",
}
# R's Value_var::FUNC_NAMES (a DataFrame source with a value column)
_R_VALUE_FUNCS = {
    "avg", "min", "max", "stddev", "sum", "lse", "quantile", "nearest", "exists", "size",
    "first", "last", "sample", "first.pos.abs", "first.pos.relative", "last.pos.abs",
    "last.pos.relative", "sample.pos.abs", "sample.pos.relative", "min.pos.abs",
    "min.pos.relative", "max.pos.abs", "max.pos.relative", "weighted.sum", "mean", "std",
}
_R_2D_TYPES = {"rectangles", "points", "computed"}


def _is_num(x: Any) -> bool:
    return not isinstance(x, (bool, _numpy.bool_)) and isinstance(x, (int, float, _numpy.integer, _numpy.floating))


def _check_edit_distance_args(kw: dict, func: str) -> None:
    """R's checks on the pwm edit-distance vtracks' arguments, R's messages."""
    for key, rname in (("score_min", "score.min"), ("score_max", "score.max")):
        v = kw.get(key)
        if v is not None and not _is_num(v):
            raise ValueError(f"{rname} must be NULL or a single numeric value")
    v = kw.get("max_indels")  # (the lse functions reject it as an unknown parameter)
    if (v is not None and "max_indels" in _ACCEPTED_VTRACK_PARAMS.get(func, ())
            and not (_is_num(v) and v >= 0 and v == int(v))):
        raise ValueError("max_indels must be NULL or a non-negative integer")
    v = kw.get("extend")  # (an NA or a list gets R's message from _check_pwm_flags)
    if (v is not None and not _is_na(v) and not isinstance(v, (bool, _numpy.bool_, list, tuple, _numpy.ndarray))
            and not (_is_num(v) and v >= 0 and v == int(v))):
        raise ValueError("extend must be True, False, or a non-negative integer")


def _is_flag(x: Any) -> bool:
    return isinstance(x, (bool, _numpy.bool_))


def _is_na_or_seq(x: Any) -> bool:
    """R's NA, or more than one value (R's C++ checks report both alike)."""
    return _is_na(x) or isinstance(x, (list, tuple, _numpy.ndarray, _pandas.Series))


def _check_pwm_flags(name: str, func_lc: str, config: dict) -> None:
    """R misha's checks on bidirect, extend, prior and strand of a pwm vtrack (the
    R handlers' and TrackExpressionParams' messages): bidirect True or False;
    extend True or False, or for the edit-distance functions also a
    non-negative whole number; prior a number in [0, 1]; strand 1 or -1 (a
    number, not a bool).
    A bool-like bidirect or extend is stored as a bool."""
    v = config.get("bidirect")
    if v is not None:
        if _is_na_or_seq(v):
            raise ValueError(f"Virtual track {name}: bidirect parameter must be True or False")
        if not _is_flag(v):
            raise ValueError("bidirect must be True or False")
        config["bidirect"] = bool(v)
    v = config.get("extend")
    if v is not None:
        if _is_na_or_seq(v):
            raise ValueError(f"Virtual track {name}: extend parameter must be True, False or a non-negative integer")
        if _is_flag(v):
            config["extend"] = bool(v)
        elif func_lc not in _FILTER_EDIT_DISTANCE_FUNCS:
            raise ValueError("extend must be True or False")
        elif not (_is_num(v) and v >= 0 and v == int(v)):
            raise ValueError("extend must be True, False, or a non-negative integer")
    v = config.get("prior")
    if v is not None and (not _is_num(v) or not 0 <= v <= 1):
        raise ValueError("prior must be a number between 0 and 1")
    v = config.get("strand")
    if v is not None:
        if _is_flag(v) and v:  # R's 1/-1 check lets TRUE through to the C++ type check
            raise ValueError(f"Virtual track {name}: strand parameter must be numeric")
        if not _is_num(v) or v not in (1, -1):
            raise ValueError("strand must be 1 or -1")


def _r_vtrack_checks(name: str, src: Any, func_lc: str, params: Any) -> None:
    """R's checks on a value / interval vtrack's function and params, made at
    gvtrack.create (R's TrackExpressionVars), with R's messages. Sequence
    vtracks (no source) have their own checks."""
    from .intervals import gintervals_exists
    from .tracks import gtrack_exists, gtrack_info

    if isinstance(params, (list, tuple, _numpy.ndarray)) and len(params) == 1:
        params = params[0]  # R's length-1 vector

    def no_params():
        if params is not None:
            raise ValueError(f"Virtual track {name}: function {func_lc} does not accept any parameters")

    if isinstance(src, str) and gtrack_exists(src):
        if func_lc not in _R_TRACK_FUNCS:
            raise ValueError(f"Virtual track {name}: invalid function {func_lc} used for a virtual track")
        ttype = str(gtrack_info(src).get("type", ""))
        is_2d = ttype in _R_2D_TYPES
        if ((not is_2d and func_lc == "area")
                or (is_2d and func_lc in ("nearest", "stddev", "std", "sum", "lse", "quantile"))
                or (ttype != "dense" and func_lc.startswith("global.percentile"))):
            raise ValueError(f"Virtual track {name}: function {func_lc} is not supported by {ttype} tracks")
        if func_lc == "quantile":
            if params is None:
                raise ValueError(f"Virtual track {name}: function quantile requires an additional "
                                 "parameter (percentile) to be specified")
            if isinstance(params, bool) or not isinstance(params, (int, float, _numpy.integer, _numpy.floating)):
                raise ValueError(f"Virtual track {name}: invalid parameters used for function quantile")
            if not 0 <= float(params) <= 1:
                raise ValueError(f"Virtual track {name}: parameter (percentile) used for function quantile "
                                 "is out of range")
        elif func_lc in ("avg", "mean", "min", "max", "nearest", "stddev", "std", "sum", "lse",
                         "global.percentile", "global.percentile.min", "global.percentile.max",
                         "weighted.sum", "area", "exists", "size", "first", "last", "sample",
                         "max.pos.abs", "max.pos.relative", "min.pos.abs", "min.pos.relative",
                         "sample.pos.abs", "sample.pos.relative", "first.pos.abs",
                         "first.pos.relative", "last.pos.abs", "last.pos.relative"):
            no_params()
        return

    is_df = isinstance(src, _pandas.DataFrame)
    is_set = isinstance(src, str) and gintervals_exists(src)
    if not (is_df or is_set):
        return
    has_values = is_df and any(c not in _NON_VALUE_COLS for c in src.columns)
    if func_lc in _DF_INTERVAL_FUNCS or not has_values:
        if func_lc not in _DF_INTERVAL_FUNCS:
            raise ValueError(f"Virtual track {name}: invalid function {func_lc} used with intervals")
        if func_lc in ("distance.center", "distance.edge", "coverage"):
            no_params()
        elif params is not None:
            if isinstance(params, bool) or not isinstance(params, (int, float, _numpy.integer, _numpy.floating)):
                if func_lc == "neighbor.count":
                    raise ValueError("neighbor.count requires 'params' to be a single numeric value")
                raise ValueError(f"Virtual track {name}: invalid parameters used for function {func_lc}")
            if float(params) < 0:
                what = "margin" if func_lc == "distance" else "distance"
                raise ValueError(f"Virtual track {name}, function {func_lc}: {what} cannot be a negative number")
        return
    if func_lc not in _R_VALUE_FUNCS:
        raise ValueError(f"Virtual track {name}: invalid function {func_lc} used with value-based intervals")
    if func_lc == "quantile":
        if params is None:
            raise ValueError(f"Virtual track {name}: function quantile requires percentile parameter")
        if not 0 <= float(params) <= 1:
            raise ValueError(f"Virtual track {name}: percentile must be between 0 and 1, got {params:g}")
    else:
        no_params()
# Columns that never count as a "value" column when inferring a DataFrame source's
# default function (mirrors R's TrackExpressionVars value-column detection).
_NON_VALUE_COLS = {"chrom", "start", "end", "strand", "intervalID", "intervalID1", "intervalID2"}


def _vtrack_source_rows(src: pd.DataFrame, error: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The chromosomes (the database's names) and coordinates of a vtrack's DataFrame source,
    checked as R misha's IntervalConverter checks them, row by row: a missing, infinite or
    out-of-range coordinate is "Invalid format of interval at index i" (from 1); then the
    chromosome: a missing string is the name "NA", a missing category or number is an invalid
    interval, and a name the database lacks is 'Chromosome "..." does not exist'; then the
    interval: start below 0, start not below end, end past the chromosome. A fraction is
    truncated, as R's as.integer. error(msg) gives the exception to raise."""
    from .intervals import _valid_coord, gintervals_all

    n = len(src)
    start = _pandas.to_numeric(src["start"], errors="coerce").to_numpy(dtype=float)
    end = _pandas.to_numeric(src["end"], errors="coerce").to_numpy(dtype=float)
    coord_bad = ~(_valid_coord(start) & _valid_coord(end))

    col = src["chrom"]
    name_bad = _numpy.zeros(n, dtype=bool)
    if isinstance(col.dtype, _pandas.CategoricalDtype):
        name_bad = col.cat.codes.to_numpy() < 0
        names = [str(v) if not bad else "" for v, bad in zip(col.astype(object), name_bad, strict=True)]
    elif _pandas.api.types.is_numeric_dtype(col.dtype) and not _pandas.api.types.is_bool_dtype(col.dtype):
        v = col.to_numpy(dtype=float, na_value=_numpy.nan)
        with _numpy.errstate(invalid="ignore"):
            name_bad = ~_numpy.isfinite(v) | (v != _numpy.floor(v)) | (_numpy.abs(v) > 1e15)
        names = [str(int(x)) if not bad else "" for x, bad in zip(v, name_bad, strict=True)]
    else:
        names = []
        for i, v in enumerate(col.to_numpy(dtype=object)):
            if isinstance(v, str):
                names.append(v)
            elif v is None or v is _pandas.NA or (isinstance(v, float) and _numpy.isnan(v)):
                names.append("NA")  # R's NA_character_ is read as the name "NA"
            elif ((isinstance(v, (int, _numpy.integer)) and not isinstance(v, (bool, _numpy.bool_)))
                  or (isinstance(v, (float, _numpy.floating)) and _numpy.isfinite(v) and v == _numpy.floor(v))):
                names.append(str(int(v)))
            else:
                name_bad[i] = True
                names.append("")

    canonical: dict[str, str | None] = {}
    for name in set(names) - {""}:
        try:
            canonical[name] = str(_pymisha.pm_normalize_chroms([name])[0])
        except Exception:
            canonical[name] = None
    chroms = [canonical.get(name) if name else None for name in names]
    unknown = _numpy.array([not bad and c is None for c, bad in zip(chroms, name_bad, strict=True)], dtype=bool)

    sizes = dict(zip(_chrom_names(gintervals_all()["chrom"]).tolist(), gintervals_all()["end"].tolist(), strict=True))
    with _numpy.errstate(invalid="ignore"):
        s_int = _numpy.where(coord_bad, 0, _numpy.trunc(start)).astype(_numpy.int64)
        e_int = _numpy.where(coord_bad, 0, _numpy.trunc(end)).astype(_numpy.int64)
    size = _numpy.array([sizes.get(c, -1) if c is not None else -1 for c in chroms], dtype=_numpy.int64)
    ok = ~coord_bad & ~name_bad & ~unknown
    neg = ok & (s_int < 0)
    inverted = ok & ~neg & (s_int >= e_int)
    past = ok & ~neg & ~inverted & (e_int > size)
    bad_row = coord_bad | name_bad | unknown | neg | inverted | past
    if bad_row.any():
        i = int(_numpy.argmax(bad_row))
        if coord_bad[i] or name_bad[i]:
            raise error(f"Invalid format of interval at index {i + 1}")
        if unknown[i]:
            raise error(f'Chromosome "{names[i]}" does not exist')
        where = f"Interval ({chroms[i]}, {s_int[i]}, {e_int[i]}): "
        if neg[i]:
            raise error(where + "start coordinate must be greater or equal than zero")
        if inverted[i]:
            raise error(where + "start coordinate must be lesser than end coordinate. misha uses 0-based, "
                        "half-open [start, end) coordinates; if your input is 1-based (GFF/GTF/VCF), "
                        "subtract 1 from start.")
        raise error(where + "end coordinate exceeds chromosome boundaries")
    return [str(c) for c in chroms], s_int, e_int


def _infer_default_vtrack_func(src: pd.DataFrame | str | None) -> str:
    """Infer the default ``func`` when none is given (R parity).

    R defaults a virtual track's function to ``"distance"`` when its source
    resolves to an intervals set (no value column) and to ``"avg"`` when the
    source is a track or a value-bearing intervals set.
    """
    if isinstance(src, _pandas.DataFrame):
        has_value = any(
            c not in _NON_VALUE_COLS and _pandas.api.types.is_numeric_dtype(src[c])
            for c in src.columns
        )
        return "avg" if has_value else "distance"
    if isinstance(src, str):
        from .tracks import _track_exists

        if _track_exists(src):
            return "avg"
        from .intervals import gintervals_exists

        if gintervals_exists(src):
            return "distance"
    return "avg"
_VALUE_DF_PY_FUNCS = {
    "avg", "mean", "sum", "min", "max", "first", "last", "size", "exists",
    "stddev", "std", "quantile", "nearest", "sample", "weighted.sum",
    "first.pos.abs", "first.pos.relative",
    "last.pos.abs", "last.pos.relative",
    "min.pos.abs", "min.pos.relative",
    "max.pos.abs", "max.pos.relative",
    "sample.pos.abs", "sample.pos.relative", "lse",
}
_FILTER_SUPPORTED_FUNCS = (
    _FILTER_PASSTHROUGH_FUNCS
    | _FILTER_WEIGHTED_FUNCS
    | _FILTER_ADDITIVE_FUNCS
    | _FILTER_MIN_FUNCS
    | _FILTER_MAX_FUNCS
    | _FILTER_STDDEV_FUNCS
    | _FILTER_QUANTILE_FUNCS
    | _FILTER_NEAREST_FUNCS
    | _FILTER_EXISTS_FUNCS
    | _FILTER_SIZE_FUNCS
    | _FILTER_FIRST_FUNCS
    | _FILTER_LAST_FUNCS
    | _FILTER_SAMPLE_FUNCS
    | _FILTER_FIRST_POS_ABS_FUNCS
    | _FILTER_FIRST_POS_REL_FUNCS
    | _FILTER_LAST_POS_ABS_FUNCS
    | _FILTER_LAST_POS_REL_FUNCS
    | _FILTER_MAX_POS_ABS_FUNCS
    | _FILTER_MAX_POS_REL_FUNCS
    | _FILTER_MIN_POS_ABS_FUNCS
    | _FILTER_MIN_POS_REL_FUNCS
    | _FILTER_SAMPLE_POS_ABS_FUNCS
    | _FILTER_SAMPLE_POS_REL_FUNCS
    | _FILTER_LOGSUMEXP_FUNCS
    | _FILTER_WEIGHTED_SUM_FUNCS
    | _FILTER_MAX_POS_BY_SCORE_FUNCS
    | _FILTER_GLOBAL_PERCENTILE_FUNCS
    | _FILTER_EDIT_DISTANCE_FUNCS
)

# Each value is kept with the stamp of the track it came from (_source_stamp), and read again
# once another session replaced the track.
_GLOBAL_PERCENTILE_CACHE: dict[tuple[str, str, int], tuple[Any, np.ndarray]] = {}
# Cache of a track's frozen pv.percentiles table: src -> (bins, breaks) or None.
_PV_TABLE_CACHE: dict[tuple[str, str], tuple[Any, tuple[np.ndarray, np.ndarray] | None]] = {}


def _source_stamp(src: str) -> Any:
    """The stamp of track src (_shared._track_stamp), with that of its vars/pv.percentiles;
    None for what is not a track."""
    try:
        path = _pymisha.pm_track_path(src)
    except _pymisha.error:
        return None
    if not path:
        return None
    return _shared._track_stamp(path), _shared._file_stamp(f"{path}/vars/pv.percentiles")


def _canonicalize_filter_df(df: pd.DataFrame) -> pd.DataFrame:
    from .intervals import _normalize_chroms, gintervals_canonic

    if not isinstance(df, _pandas.DataFrame):
        raise ValueError("filter must be a DataFrame")
    if not {"chrom", "start", "end"}.issubset(df.columns):
        raise ValueError("filter must have columns: chrom, start, end")

    filt = df[["chrom", "start", "end"]].copy()
    filt["chrom"] = _normalize_chroms(_chrom_names(filt["chrom"]).tolist())
    filt["start"] = _pandas.to_numeric(filt["start"], errors="coerce").astype("Int64")
    filt["end"] = _pandas.to_numeric(filt["end"], errors="coerce").astype("Int64")
    filt = filt.dropna(subset=["start", "end"]).copy()
    if len(filt) == 0:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])
    filt["start"] = filt["start"].astype(_numpy.int64)
    filt["end"] = filt["end"].astype(_numpy.int64)
    filt = filt[filt["end"] > filt["start"]].reset_index(drop=True)
    if len(filt) == 0:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])
    can = gintervals_canonic(filt, unify_touching_intervals=True)
    if can is None:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])
    return can[["chrom", "start", "end"]].reset_index(drop=True)


def _resolve_filter_sources(filter_obj: pd.DataFrame | str | list[Any] | tuple[Any, ...] | None) -> pd.DataFrame:
    from .extract import gextract
    from .intervals import gintervals_all, gintervals_load, gintervals_ls
    from .tracks import gtrack_info

    if filter_obj is None:
        return _pandas.DataFrame(columns=["chrom", "start", "end"])

    if isinstance(filter_obj, _pandas.DataFrame):
        return _canonicalize_filter_df(filter_obj)

    if isinstance(filter_obj, list | tuple):
        if len(filter_obj) == 0:
            return _pandas.DataFrame(columns=["chrom", "start", "end"])
        # Collect all resolved parts, then union once via concat + single union
        parts = []
        for part in filter_obj:
            part_df = _resolve_filter_sources(part)
            if part_df is not None and len(part_df) > 0:
                parts.append(part_df[["chrom", "start", "end"]])
        if not parts:
            return _pandas.DataFrame(columns=["chrom", "start", "end"])
        if len(parts) == 1:
            return _canonicalize_filter_df(parts[0])
        # Concat all parts and canonicalize (which internally unions/merges)
        merged = _pandas.concat(parts, ignore_index=True)
        return _canonicalize_filter_df(merged)

    if isinstance(filter_obj, str):
        names = gintervals_ls()
        if names and filter_obj in names:
            loaded = gintervals_load(filter_obj)
            if loaded is None:
                return _pandas.DataFrame(columns=["chrom", "start", "end"])
            return _canonicalize_filter_df(loaded)

        track_path = _pymisha.pm_track_path(filter_obj)
        if track_path:
            info = gtrack_info(filter_obj)
            track_type = str(info.get("type", "")).lower()
            if track_type not in {"sparse", "intervals"}:
                raise ValueError(f"Track '{filter_obj}' is not an intervals-type track")
            extracted = gextract(filter_obj, gintervals_all())
            if extracted is None:
                return _pandas.DataFrame(columns=["chrom", "start", "end"])
            return _canonicalize_filter_df(extracted)

        raise ValueError(f"Unknown filter source '{filter_obj}'")

    raise ValueError("filter must be a DataFrame, string, list/tuple, or None")


def _filter_key(filter_df: pd.DataFrame | None) -> str | None:
    if filter_df is None or len(filter_df) == 0:
        return None
    # Vectorized hash: build byte string from arrays
    chroms = filter_df["chrom"].astype(str).values
    starts = filter_df["start"].astype(int).values
    ends = filter_df["end"].astype(int).values
    lines = [f"{c}\t{s}\t{e}\n" for c, s, e in zip(chroms, starts, ends, strict=False)]
    return hashlib.sha1("".join(lines).encode()).hexdigest()


def _filter_stats(filter_df: pd.DataFrame | None) -> dict[str, int] | None:
    if filter_df is None or len(filter_df) == 0:
        return None
    total = int((filter_df["end"] - filter_df["start"]).sum())
    chroms = int(filter_df["chrom"].nunique())
    return {"num_chroms": chroms, "total_bases": total}


def _build_unmasked_segments(
    intervals: pd.DataFrame,
    payload: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    from .intervals import gintervals_all

    sshift = int(payload.get("sshift", 0) or 0)
    eshift = int(payload.get("eshift", 0) or 0)
    chrom_sizes_df = gintervals_all()
    chrom_sizes = {
        str(chrom): int(end)
        for chrom, end in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)
    }

    has_filter = filter_df is not None and len(filter_df) > 0

    # --- Vectorized path (no mask) ---
    if not has_filter:
        chroms = _chrom_names(intervals["chrom"]).to_numpy()
        starts = intervals["start"].to_numpy(dtype=_numpy.int64) + sshift
        ends = intervals["end"].to_numpy(dtype=_numpy.int64) + eshift

        # Map chroms to sizes
        chrom_bound = _numpy.array(
            [chrom_sizes.get(str(c), -1) for c in chroms], dtype=_numpy.int64
        )
        known = chrom_bound >= 0
        starts = _numpy.maximum(starts, 0)
        ends = _numpy.where(known, _numpy.minimum(ends, chrom_bound), ends)
        valid = known & (ends > starts)

        if not valid.any():
            return None

        idx = _numpy.flatnonzero(valid)
        v_chroms = chroms[idx]
        v_starts = starts[idx]
        v_ends = ends[idx]
        v_lens = v_ends - v_starts

        return _pandas.DataFrame({
            "chrom": v_chroms,
            "start": v_starts,
            "end": v_ends,
            "orig_idx": idx,
            "seg_len": v_lens,
            "base_start": v_starts,
        })

    # --- With a mask: each interval minus the mask. The mask is canonic (sorted, disjoint,
    # touching intervals unified), so the mask intervals an interval overlaps are consecutive
    # and its unmasked parts are the gaps between them, in order ---
    assert filter_df is not None
    names = _pandas.Index(list(chrom_sizes))
    sizes = _numpy.array(list(chrom_sizes.values()), dtype=_numpy.int64)
    chroms = intervals["chrom"].astype(str).to_numpy()
    cid = names.get_indexer(chroms)
    starts = _numpy.maximum(intervals["start"].to_numpy(dtype=_numpy.int64) + sshift, 0)
    ends = intervals["end"].to_numpy(dtype=_numpy.int64) + eshift
    known = cid >= 0
    ends = _numpy.where(known, _numpy.minimum(ends, sizes[cid]), ends)
    keep = _numpy.flatnonzero(known & (ends > starts))
    if keep.size == 0:
        return None
    cid, starts, ends = cid[keep], starts[keep], ends[keep]

    # positions keyed by chromosome, then coordinate
    key = int(sizes.max()) + 1
    m_cid = names.get_indexer(filter_df["chrom"].astype(str).to_numpy())
    m_start = filter_df["start"].to_numpy(dtype=_numpy.int64)[m_cid >= 0]
    m_end = filter_df["end"].to_numpy(dtype=_numpy.int64)[m_cid >= 0]
    m_cid = m_cid[m_cid >= 0]
    order = _numpy.lexsort((m_start, m_cid))
    m_cid, m_start, m_end = m_cid[order], m_start[order], m_end[order]
    # the first mask interval ending after the start, the first starting at or after the end
    k0 = _numpy.searchsorted(m_cid * key + m_end, cid * key + starts, side="right")
    k1 = _numpy.searchsorted(m_cid * key + m_start, cid * key + ends, side="left")
    # part t of an interval runs from the end of mask interval k0 + t - 1 (or the interval's
    # start) to the start of mask interval k0 + t (or the interval's end)
    n_parts = k1 - k0 + 1
    row = _numpy.repeat(_numpy.arange(len(keep)), n_parts)
    t = _numpy.arange(int(n_parts.sum())) - _numpy.repeat(_numpy.cumsum(n_parts) - n_parts, n_parts)
    k = k0[row] + t
    m_start = _numpy.append(m_start, 0)
    m_end = _numpy.append(m_end, 0)
    p_start = _numpy.where(t == 0, starts[row], m_end[k - 1])
    p_end = _numpy.where(k == k1[row], ends[row], m_start[k])
    nonempty = p_end > p_start
    row, p_start, p_end = row[nonempty], p_start[nonempty], p_end[nonempty]
    if row.size == 0:
        return None
    return _pandas.DataFrame({
        "chrom": chroms[keep][row],
        "start": p_start,
        "end": p_end,
        "orig_idx": keep[row],
        "seg_len": p_end - p_start,
        "base_start": starts[row],
    })


def _compute_value_df_vtrack(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """A filtered value-based vtrack (a DataFrame source), as R misha's ValueVarProcessor: the
    source is held as R's in-memory track holds it (values as floats), and C++ reads the unmasked
    parts of each interval (pm_vtrack_compute with part_owner). "nearest" without a value in the
    parts is the value whose unmasked stretch is nearest to them (_nearest_unmasked)."""
    func = str(payload_eval.get("func", "avg")).lower()
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    src_df = payload_eval.get("src_df")
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    if src_df is None or len(src_df) == 0 or seg_df is None or len(seg_df) == 0:
        return out  # every interval masked entirely (or empty after the shifts): NaN, as R

    oi = seg_df["orig_idx"].to_numpy(dtype=_numpy.int64)
    payload = {k: v for k, v in payload_eval.items() if k != "src_df"}
    payload.update(sshift=0, eshift=0, part_owner=oi, part_base=seg_df["base_start"].to_numpy(dtype=_numpy.int64))
    vals = _numpy.asarray(
        _pymisha.pm_vtrack_compute(payload, _df2pymisha(seg_df[["chrom", "start", "end"]]), CONFIG), dtype=float)
    first = _numpy.flatnonzero(_numpy.r_[True, oi[1:] != oi[:-1]])
    out[oi[first]] = vals[first]

    if func == "nearest":
        last = _numpy.r_[first[1:], len(oi)] - 1
        need = _numpy.isnan(vals[first])
        if need.any():
            masks = _mask_by_chrom(filter_df)
            chroms = seg_df["chrom"].astype(str).to_numpy()
            p_start = seg_df["start"].to_numpy(dtype=_numpy.int64)
            p_end = seg_df["end"].to_numpy(dtype=_numpy.int64)
            src_chroms = src_df["chrom"].astype(str).to_numpy()
            items: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for a, b in zip(first[need].tolist(), last[need].tolist(), strict=True):
                chrom = str(chroms[a])
                if chrom not in items:
                    grp = src_df[src_chroms == chrom].sort_values("start", kind="stable")
                    # each value as a float, as R's in-memory track holds it
                    items[chrom] = (grp["start"].to_numpy(dtype=_numpy.int64), grp["end"].to_numpy(dtype=_numpy.int64),
                                    grp["value"].to_numpy(dtype=_numpy.float32).astype(float))
                starts, ends, ivals = items[chrom]
                m_start, m_end = masks.get(chrom, _NO_MASK)
                out[oi[a]] = _nearest_unmasked(starts, ends, ivals, m_start, m_end, int(p_start[a]), int(p_end[b]))
    return out


def _filtered_item_reduce(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    func: str,
) -> np.ndarray:
    """sum, size or avg of a filtered track vtrack, each bin or track interval once, however many
    unmasked parts it overlaps (R misha 5.15)."""
    runs, out = _unmasked_items(intervals, payload_eval, filter_df)
    for u in runs:
        n = u.hi - u.lo
        if func == "size":
            first = _numpy.flatnonzero(_numpy.r_[True, u.oi[1:] != u.oi[:-1]])
            out[u.oi[first]] = _numpy.add.reduceat(n, first).astype(float)
            continue
        read = n > 0
        if not read.any():
            continue
        # as R: each part's sum or mean, which the track holds as a float, then the parts'
        # sums, or their means weighted by their value counts, added in order
        counts = n[read].astype(float)
        part_sums = _seq_range_sums(u.vals, u.lo[read], u.hi[read])
        if func == "sum":
            terms = part_sums.astype(_numpy.float32).astype(float)
        else:
            terms = (part_sums / counts).astype(_numpy.float32).astype(float) * counts
        oi = u.oi[read]
        first = _numpy.flatnonzero(_numpy.r_[True, oi[1:] != oi[:-1]])
        totals = _seq_range_sums(terms, first, _numpy.r_[first[1:], len(oi)])
        out[oi[first]] = totals if func == "sum" else totals / _numpy.add.reduceat(counts, first)
    return out


def _compute_filtered_nearest(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """nearest of a filtered track vtrack, as R misha 5.17: the average of the unmasked values,
    each value once (as filtered avg); without an unmasked value in the interval, the value of
    the sparse or array track nearest to its unmasked parts (_nearest_unmasked). A dense track
    gives NaN there, as its unfiltered nearest does not look outside the interval either."""
    out = _filtered_item_reduce(intervals, payload_eval, filter_df, "avg")
    src = payload_eval.get("src")
    if not _numpy.isnan(out).any() or not isinstance(src, str):
        return out
    try:
        if _pymisha.pm_track_info(src).get("bin_size"):
            return out  # dense
    except _pymisha.error:
        return out
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    if seg_df is None or len(seg_df) == 0:
        return out
    oi = seg_df["orig_idx"].to_numpy(dtype=_numpy.int64)
    first = _numpy.flatnonzero(_numpy.r_[True, oi[1:] != oi[:-1]])
    last = _numpy.r_[first[1:], len(oi)] - 1
    need = _numpy.isnan(out[oi[first]])
    if not need.any():
        return out
    chroms = seg_df["chrom"].astype(str).to_numpy()
    p_start = seg_df["start"].to_numpy(dtype=_numpy.int64)
    p_end = seg_df["end"].to_numpy(dtype=_numpy.int64)
    masks = _mask_by_chrom(filter_df)
    items: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for a, b in zip(first[need].tolist(), last[need].tolist(), strict=True):
        chrom = str(chroms[a])
        if chrom not in items:
            items[chrom] = _track_chrom_items(src, chrom, _slice_keys(payload_eval))
        starts, ends, vals = items[chrom]
        m_start, m_end = masks.get(chrom, _NO_MASK)
        out[oi[a]] = _nearest_unmasked(starts, ends, vals, m_start, m_end, int(p_start[a]), int(p_end[b]))
    return out


_NO_MASK = (_numpy.empty(0, dtype=_numpy.int64), _numpy.empty(0, dtype=_numpy.int64))


def _mask_by_chrom(filter_df: pd.DataFrame | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The starts and ends of a canonic mask (sorted, disjoint, touching intervals unified),
    by chromosome."""
    if filter_df is None or len(filter_df) == 0:
        return {}
    return {str(c): (g["start"].to_numpy(dtype=_numpy.int64), g["end"].to_numpy(dtype=_numpy.int64))
            for c, g in filter_df.groupby(_chrom_names(filter_df["chrom"]), sort=False)}


def _track_chrom_items(src: str, chrom: str,
                       slice_keys: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The starts, ends and values of the intervals of sparse or array track src on chrom,
    by start (NaN where an interval holds no value), after the slice of an array-slice vtrack."""
    from .intervals import gintervals_all

    all_iv = gintervals_all()
    size = int(all_iv["end"].to_numpy()[_chrom_names(all_iv["chrom"]).to_numpy() == chrom][0])
    res = _pymisha.pm_extract([src], _df2pymisha(_pandas.DataFrame(
        {"chrom": [chrom], "start": [0], "end": [size]})), None, CONFIG)
    if res is None:
        empty = _numpy.empty(0, dtype=_numpy.int64)
        return empty, empty, _numpy.empty(0, dtype=float)
    starts = _numpy.asarray(res[2], dtype=_numpy.int64)
    vals = _numpy.asarray(res[4], dtype=float)
    if slice_keys is not None and len(vals):
        vals = _sliced_values(src, slice_keys, _numpy.full(len(vals), chrom, dtype=object), res[2], res[3])
    order = _numpy.argsort(starts, kind="stable")
    return starts[order], _numpy.asarray(res[3], dtype=_numpy.int64)[order], vals[order]


def _nearest_unmasked(
    starts: np.ndarray, ends: np.ndarray, vals: np.ndarray,
    m_start: np.ndarray, m_end: np.ndarray, first_start: int, last_end: int,
) -> float:
    """nearest of a filtered vtrack whose unmasked parts, from first_start to last_end, hold no
    value (R misha's TrackVarProcessor::nearest_unmasked_value): the value of the interval
    whose unmasked stretch is nearest to the parts, measured from the first part for those on
    the left and from the last part for those on the right; a fully masked interval is never
    chosen, the left one wins at the same distance; NaN if there is none. The intervals
    (starts, ends, vals) do not overlap and are sorted; the mask (m_start, m_end) is canonic."""
    left_val = right_val = _numpy.nan
    left_dist = right_dist = -1
    for j in range(int(_numpy.searchsorted(starts, first_start, side="right")) - 1, -1, -1):
        if _numpy.isnan(vals[j]):
            continue
        a, b = int(starts[j]), min(int(ends[j]), first_start)
        # the end of the last unmasked bp of [a, b)
        k = int(_numpy.searchsorted(m_end, b - 1, side="right"))
        e = b if k == len(m_end) or m_start[k] > b - 1 else int(m_start[k]) if m_start[k] > a else None
        if e is not None:
            left_val, left_dist = float(vals[j]), first_start - e
            break
    for j in range(int(_numpy.searchsorted(ends, last_end, side="right")), len(starts)):
        if _numpy.isnan(vals[j]):
            continue
        a, b = max(int(starts[j]), last_end), int(ends[j])
        # the first unmasked bp of [a, b)
        k = int(_numpy.searchsorted(m_end, a, side="right"))
        st = a if k == len(m_end) or m_start[k] > a else int(m_end[k]) if m_end[k] < b else None
        if st is not None:
            right_val, right_dist = float(vals[j]), st - last_end
            break
    if left_dist >= 0 and (right_dist < 0 or left_dist <= right_dist):
        return left_val
    return right_val


def _compute_filtered_sample(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    func: str,
) -> np.ndarray:
    """sample, sample.pos.abs or sample.pos.relative of a filtered track vtrack: one value drawn
    with the same chance from the unmasked values, each bin or track interval once, however
    many unmasked parts it overlaps (R misha 5.17); its position clipped to the first part it
    overlaps."""
    runs, out = _unmasked_items(intervals, payload_eval, filter_df)
    for u in runs:
        base = dict(zip(u.oi.tolist(), u.base.tolist(), strict=True))
        for orig_idx, ivals, ipos in _unmasked_groups(u, positions=func != "sample"):
            k = int(_numpy.random.randint(len(ivals)))
            if func == "sample":
                out[orig_idx] = float(ivals[k])
            else:
                assert ipos is not None
                out[orig_idx] = float(ipos[k] - base[orig_idx]) if func.endswith(".relative") else float(ipos[k])
    return out


class _Unmasked(NamedTuple):
    """What a filtered track vtrack reads for a run of intervals (_read_unmasked_items).

    vals and pos are the items read (bins or track intervals, non-NaN values only) in
    genome order, pos being each item's start. The other arrays have one entry per
    unmasked part of the run's intervals, in the order of _build_unmasked_segments (by
    interval, then position): the interval it belongs to (oi), its start, its interval's
    shifted start (base) and the range [lo, hi) of the items it reads.
    """

    vals: np.ndarray
    pos: np.ndarray
    oi: np.ndarray
    start: np.ndarray
    base: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


# The bins a filtered dense track vtrack reads at a time, about (_read_unmasked_items)
_UNMASKED_READ_ROWS = 2_000_000


def _unmasked_items(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> tuple[Iterator[_Unmasked], np.ndarray]:
    """The values a filtered track vtrack reads, each bin or track interval once.

    As R misha 5.15 (parts_reading_items_once): a bin (dense) or interval (sparse,
    array) that the filter splits is read once, with the first unmasked part it
    overlaps, at its start clipped to that part (the *.pos functions clip to the part
    they read).

    Returns the items and parts read, a run of intervals at a time (_read_unmasked_items),
    and a NaN-filled output array.
    """
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    src = payload_eval.get("src")
    if seg_df is None or len(seg_df) == 0 or src is None:
        return iter(()), out
    return _read_unmasked_items(seg_df, src, _slice_keys(payload_eval)), out


def _slice_keys(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The slice keys of an array-slice vtrack's payload (_array_slice_spec), None for another."""
    if "slice_func" not in payload:
        return None
    return {k: payload[k] for k in ("slice", "slice_func", "slice_percentile") if k in payload}


def _sliced_values(src: str, slice_keys: dict[str, Any], chroms: Any, starts: Any, ends: Any) -> np.ndarray:
    """The value of each interval of array track src (one interval each, possibly clipped) after
    the slice of an array-slice vtrack."""
    payload = {"src": src, "func": "avg", "sshift": 0, "eshift": 0, **slice_keys}
    df = _pandas.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    return _numpy.asarray(_pymisha.pm_vtrack_compute(payload, _df2pymisha(df), CONFIG), dtype=float)


def _read_unmasked_items(seg_df: pd.DataFrame, src: str,
                         slice_keys: dict[str, Any] | None = None) -> Iterator[_Unmasked]:
    """The items each unmasked part of seg_df (_build_unmasked_segments) reads from track src.

    The track is read with its own iterator (a row per bin or track interval) over the union
    of the intervals' spans (first part's start to last part's end), in runs of intervals:
    those on a chromosome for a sparse or array track, whose reader loads the chromosome's
    intervals anyway; for a dense track, intervals whose union holds about
    _UNMASKED_READ_ROWS bins (or max_data_size, if lower). Its items do not overlap, so the
    items a part reads are consecutive: those that start before its end and not before the
    previous part's end, and end after its start.
    """
    from .intervals import gintervals_all

    oi = seg_df["orig_idx"].to_numpy(dtype=_numpy.int64)
    p_start = seg_df["start"].to_numpy(dtype=_numpy.int64)
    p_end = seg_df["end"].to_numpy(dtype=_numpy.int64)
    base = seg_df["base_start"].to_numpy(dtype=_numpy.int64)
    is_first = _numpy.r_[True, oi[1:] != oi[:-1]]
    first = _numpy.flatnonzero(is_first)
    last = _numpy.r_[first[1:], len(oi)] - 1
    prev_end = _numpy.r_[0, p_end[:-1]]
    # positions keyed by chromosome (in genome order, as the track's rows come), then coordinate
    names = _pandas.Index(gintervals_all()["chrom"].astype(str).to_numpy())
    key = int(p_end.max()) + 1
    p_key = names.get_indexer(seg_df["chrom"].astype(str).to_numpy()).astype(_numpy.int64) * key

    # the intervals in genome order of their spans, cut into runs
    span_s = p_key[first] + p_start[first]
    span_e = p_key[first] + p_end[last]
    order = _numpy.argsort(span_s, kind="stable")
    span_s, span_e = span_s[order], span_e[order]
    try:
        bin_size = _pymisha.pm_track_info(src).get("bin_size")
    except _pymisha.error:
        bin_size = None
    if bin_size:
        # the bases each span adds to the union of those before it
        run_end = _numpy.maximum.accumulate(span_e)
        added = _numpy.maximum(span_e - _numpy.maximum(span_s, _numpy.r_[span_s[:1], run_end[:-1]]), 0)
        budget = min(_shared._config_limit("max_data_size"), _UNMASKED_READ_ROWS) * int(bin_size)
        run = (_numpy.cumsum(added) - added) // budget
    else:
        run = span_s // key
    _, span_run = _numpy.unique(run, return_inverse=True)
    part_run: _numpy.ndarray = _numpy.empty(len(first), dtype=_numpy.int64)
    part_run[order] = span_run
    part_run = _numpy.repeat(part_run, _numpy.diff(_numpy.r_[first, len(oi)]))
    by_run = _numpy.argsort(part_run, kind="stable")
    n_runs = int(span_run[-1]) + 1
    part_bounds = _numpy.searchsorted(part_run[by_run], _numpy.arange(n_runs + 1))
    span_bounds = _numpy.searchsorted(span_run, _numpy.arange(n_runs + 1))

    for r in range(n_runs):
        sel = by_run[part_bounds[r]:part_bounds[r + 1]]
        s, e = span_s[span_bounds[r]:span_bounds[r + 1]], span_e[span_bounds[r]:span_bounds[r + 1]]
        # the run's spans, merged where they overlap or touch
        e_max = _numpy.maximum.accumulate(e)
        new = _numpy.flatnonzero(_numpy.r_[True, s[1:] > e_max[:-1]])
        m_s = s[new]
        m_e = e_max[_numpy.r_[new[1:], len(s)] - 1]
        m_key = m_s // key * key
        # gextract(src, spans) without building its data frame: columns chrom, start, end,
        # value, intervalID (None if no rows)
        res = _pymisha.pm_extract([src], _df2pymisha(_pandas.DataFrame({
            "chrom": names[m_s // key].to_numpy(), "start": m_s - m_key, "end": m_e - m_key})), None, CONFIG)
        if res is None:
            res = [None, None] + [_numpy.empty(0, dtype=_numpy.int64)] * 4
        vals = _numpy.asarray(res[4], dtype=float)
        if slice_keys is not None and len(vals):
            # an array-slice vtrack: each array interval's columns reduced by its slice
            span_chrom = names[m_s // key].to_numpy()
            vals = _sliced_values(src, slice_keys, span_chrom[_numpy.asarray(res[5], dtype=_numpy.int64) - 1],
                                  res[2], res[3])
        ok = ~_numpy.isnan(vals)
        vals = vals[ok]
        r_key = m_key[_numpy.asarray(res[5], dtype=_numpy.int64)[ok] - 1]
        pos = _numpy.asarray(res[2], dtype=_numpy.int64)[ok]
        r_s = r_key + pos
        r_e = r_key + _numpy.asarray(res[3], dtype=_numpy.int64)[ok]
        del res
        if (_numpy.diff(r_s) < 0).any():
            o = _numpy.argsort(r_s, kind="stable")
            vals, pos, r_s, r_e = vals[o], pos[o], r_s[o], r_e[o]
        if (_numpy.diff(r_e) < 0).any():
            raise RuntimeError(f"Track {src} holds overlapping intervals")

        k = p_key[sel]
        hi = _numpy.searchsorted(r_s, k + p_end[sel], side="left")
        lo = _numpy.searchsorted(r_e, k + p_start[sel], side="right")
        lo = _numpy.where(is_first[sel], lo,
                          _numpy.maximum(lo, _numpy.searchsorted(r_s, k + prev_end[sel], side="left")))
        yield _Unmasked(vals, pos, oi[sel], p_start[sel], base[sel], _numpy.minimum(lo, hi), hi)


def _unmasked_groups(u: _Unmasked, positions: bool = False) -> list[tuple[int, np.ndarray, np.ndarray | None]]:
    """(interval, values, positions) for each interval of u that reads at least one item: its
    values in genome order and, if asked, their positions (each item's start clipped to its
    part)."""
    read = _numpy.flatnonzero(u.hi > u.lo)
    oi = u.oi[read]
    bounds = _numpy.flatnonzero(_numpy.r_[True, oi[1:] != oi[:-1], True]).tolist() if len(read) else []
    lo, hi, start = u.lo.tolist(), u.hi.tolist(), u.start.tolist()
    groups: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        parts = read[a:b].tolist()
        vals = [u.vals[lo[j]:hi[j]] for j in parts]
        pos = [_numpy.maximum(u.pos[lo[j]:hi[j]], start[j]) for j in parts] if positions else None
        groups.append((int(oi[a]), vals[0] if len(parts) == 1 else _numpy.concatenate(vals),
                       None if pos is None else pos[0] if len(parts) == 1 else _numpy.concatenate(pos)))
    return groups


def _extract_raw_unmasked_values(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> tuple[Iterator[tuple[int, np.ndarray]], np.ndarray]:
    """The unmasked values of each interval that has any, each bin or track interval once
    (_unmasked_items)."""
    runs, out = _unmasked_items(intervals, payload_eval, filter_df)
    return ((o, v) for u in runs for o, v, _p in _unmasked_groups(u)), out


def _seq_range_sums(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The values in each range [lo, hi) of vals added one after the other, as np.cumsum
    adds them (-0.0 for an empty range); np.add.reduceat adds pairwise, which rounds
    differently.

    The ranges are summed together, a value of each per step, except the longest ones,
    summed on their own where that takes fewer steps.
    """
    n = hi - lo
    out = _numpy.full(len(n), -0.0)
    by_len = _numpy.argsort(-n, kind="stable")
    lens = n[by_len]
    # summing the m longest on their own leaves lens[m] steps
    m = int(_numpy.argmin(_numpy.r_[lens, 0] + _numpy.arange(len(lens) + 1)))
    for k in by_len[:m]:
        out[k] = _numpy.cumsum(vals[lo[k]:hi[k]])[-1]
    rest, starts, neg_lens = by_len[m:], lo[by_len[m:]], -lens[m:]
    acc = _numpy.full(len(rest), -0.0)  # -0.0 + v is v for every v
    for step in range(-int(neg_lens[0]) if len(rest) else 0):
        active = int(_numpy.searchsorted(neg_lens, -step, side="left"))  # the ranges longer than step
        acc[:active] += vals[starts[:active] + step]
    out[rest] = acc
    return out


def _compute_filtered_extremum_pos(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    mode: str,
    relative: bool,
) -> np.ndarray:
    """Compute max.pos.* or min.pos.* under filter.

    For track sources (string), extracts raw bin values with positions and finds
    the position of the global extremum across all unmasked segments.

    For value-based sources (DataFrame), delegates to C++ per-segment computation:
    computes both the extremum value (``max``/``min``) and its position
    (``max.pos.abs``/``min.pos.abs``) on each unmasked segment, then selects
    the segment with the overall best value.

    Parameters
    ----------
    mode : str
        ``"max"`` or ``"min"`` — selects the extremum.
    relative : bool
        If True, return position relative to the shifted interval start.
        If False, return the absolute genomic coordinate.
    """
    src = payload_eval.get("src")

    # For track sources (string), use raw bin extraction with positions
    if isinstance(src, str):
        runs, out = _unmasked_items(intervals, payload_eval, filter_df)
        for u in runs:
            base = dict(zip(u.oi.tolist(), u.base.tolist(), strict=True))
            for orig_idx, ivals, ipos in _unmasked_groups(u, positions=True):
                assert ipos is not None
                # the first item holding the extremum
                best_pos = int(ipos[int(_numpy.argmax(ivals) if mode == "max" else _numpy.argmin(ivals))])
                out[orig_idx] = float(best_pos - base[orig_idx]) if relative else float(best_pos)
        return out

    # For value-based (DataFrame) sources, filter in Python.
    # The C++ backend does not support max.pos.*/min.pos.* for value-based vtracks.
    # We intersect source intervals with unmasked segments and find the extremum.
    from .intervals import _normalize_chroms

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    # Recover the original DataFrame source from the already-converted payload.
    # At this point ``src`` is the _df2pymisha output (list of arrays).
    # Reconstruct the relevant columns.
    assert src is not None
    src_arr: list[Any] = src  # list: [colnames, chrom_arr, start_arr, end_arr, val_arr, ...]
    src_chroms_raw = [str(c) for c in src_arr[1]]
    src_chroms = _normalize_chroms(src_chroms_raw)
    src_starts = _numpy.asarray(src_arr[2], dtype=int)
    src_ends = _numpy.asarray(src_arr[3], dtype=int)
    src_vals = _numpy.asarray(src_arr[4], dtype=float)

    best_extremum_val: dict[int, float] = {}
    for row in seg_df.itertuples(index=False):
        seg_chrom = str(row.chrom)
        seg_start = int(row.start)
        seg_end = int(row.end)
        oi = int(row.orig_idx)
        bs = int(row.base_start)

        for sc, ss, se, sv in zip(src_chroms, src_starts, src_ends, src_vals, strict=False):
            if sc != seg_chrom:
                continue
            # Check overlap between source interval and unmasked segment
            ov_start = max(int(ss), seg_start)
            ov_end = min(int(se), seg_end)
            if ov_end <= ov_start:
                continue
            if _numpy.isnan(sv):
                continue
            is_better = (
                oi not in best_extremum_val
                or (mode == "max" and float(sv) > best_extremum_val[oi])
                or (mode == "min" and float(sv) < best_extremum_val[oi])
            )
            if is_better:
                best_extremum_val[oi] = float(sv)
                pos = float(int(ss))
                if relative:
                    out[oi] = pos - float(bs)
                else:
                    out[oi] = pos

    return out


def _compute_filtered_edit_distance(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    func: str,
) -> np.ndarray:
    """An edit-distance vtrack under a filter, as R misha's aggregate_edit_distance_parts:
    each unmasked part scored alone, then the fewest edits over the parts (the default);
    for the .pos functions the position in the part with the fewest edits, moved back to
    the interval's coordinates (sign kept: negative is the reverse strand); for
    pwm.max.edit_distance the edits of the part with the best PWM score; the most
    mutations for pwm.n_mutations."""
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    if seg_df is None or len(seg_df) == 0:
        return out
    payload_eval = _with_row_subs(payload_eval, seg_df)
    payload_eval["sshift"] = 0
    payload_eval["eshift"] = 0
    segs = _df2pymisha(seg_df[["chrom", "start", "end"]])

    def per_part(f: str) -> np.ndarray:
        return _numpy.asarray(_pymisha.pm_vtrack_compute(dict(payload_eval, func=f), segs, CONFIG), dtype=float)

    score = per_part(func)
    key = score  # the per-part value the parts are ranked by
    if func.endswith(".pos"):
        key = per_part(func[:-len(".pos")])  # the part's own edit count
    elif func == "pwm.max.edit_distance":
        key = per_part("pwm.max")  # the part's best PWM score
    oi = seg_df["orig_idx"].to_numpy(dtype=_numpy.int64)
    offset = (seg_df["start"].to_numpy(dtype=_numpy.int64)
              - seg_df["base_start"].to_numpy(dtype=_numpy.int64)).astype(float)
    bounds = _numpy.flatnonzero(_numpy.r_[True, oi[1:] != oi[:-1], True])  # the parts by interval
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
        o, idx = oi[a], range(a, b)
        best = _numpy.nan
        if func.endswith(".pos"):
            best_edits = _numpy.nan
            for k in idx:
                if _numpy.isnan(key[k]) or not (_numpy.isnan(best_edits) or key[k] < best_edits):
                    continue
                best_edits = key[k]
                sc = score[k]
                best = sc + offset[k] if sc > 0 else sc - offset[k] if sc < 0 else sc
        elif func == "pwm.max.edit_distance":
            best_logp = -_numpy.inf
            found = False
            for k in idx:
                if not found or key[k] > best_logp:
                    best_logp, best, found = key[k], score[k], True
        elif func == "pwm.n_mutations":
            for k in idx:
                if _numpy.isnan(best) or (not _numpy.isnan(score[k]) and score[k] > best):
                    best = score[k]
        else:
            for k in idx:
                if _numpy.isnan(best) or (not _numpy.isnan(score[k]) and score[k] < best):
                    best = score[k]
        out[o] = best
    return out


def _compute_filtered_stddev(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """stddev of a filtered track vtrack, as R misha's aggregate_stddev_with_filter: each unmasked
    part's count, mean and standard deviation (each value once, the part's mean and deviation
    held as floats, as the track gives them) merged pairwise (Chan et al.)."""
    runs, out = _unmasked_items(intervals, payload_eval, filter_df)
    for u in runs:
        lo, hi, oi = u.lo.tolist(), u.hi.tolist(), u.oi.tolist()
        vals = u.vals.tolist()
        j = 0
        while j < len(oi):
            o = oi[j]
            n = mean = m2 = 0.0
            while j < len(oi) and oi[j] == o:
                part = vals[lo[j]:hi[j]]
                j += 1
                if not part:
                    continue
                # the part's mean (summed in double, rounded to float) and its deviation (Welford)
                part_n = float(len(part))
                part_mean = float(_numpy.float32(_seq_sum(part) / part_n))
                part_m2 = 0.0
                if len(part) > 1:
                    w_mean = w_m2 = 0.0
                    for k, v in enumerate(part, 1):
                        delta = v - w_mean
                        w_mean += delta / k
                        w_m2 += delta * (v - w_mean)
                    part_sd = float(_numpy.float32(_math.sqrt(w_m2 / (len(part) - 1))))
                    part_m2 = part_sd * part_sd * (part_n - 1)
                delta = part_mean - mean
                total = n + part_n
                mean += delta * part_n / total
                m2 += part_m2 + delta * delta * n * part_n / total
                n = total
            if n > 1:
                out[o] = _math.sqrt(m2 / (n - 1))
    return out


def _seq_sum(vals: list[float]) -> float:
    """vals added one after the other, from 0, as C++ adds them in a double."""
    total = 0.0
    for v in vals:
        total += v
    return total


def _compute_filtered_quantile(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """quantile of a filtered track vtrack, as R misha's aggregate_quantile_with_filter: the
    unmasked values (each once, floats) sorted, interpolated in double and rounded to a float, as
    StreamPercentiler<float> does while they fit max_data_size."""
    percentile = min(max(_quantile_param(payload_eval), 0.0), 1.0)
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups:
        v = _numpy.sort(_numpy.asarray(raw_vals, dtype=_numpy.float32)).astype(float)
        index = (len(v) - 1) * percentile
        i1, i2 = _math.floor(index), _math.ceil(index)
        weight = index - i1
        out[orig_idx] = float(_numpy.float32(v[i1] * (1 - weight) + v[i2] * weight))
    return out


def _logsumexp(values: list[float] | np.ndarray) -> float:
    arr = _numpy.asarray(values, dtype=float)
    arr = arr[~_numpy.isnan(arr)]
    if arr.size == 0:
        return _numpy.nan
    m = float(arr.max())
    if _numpy.isneginf(m):
        return m
    return float(m + _numpy.log(_numpy.exp(arr - m).sum()))


def _compute_filtered_lse(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """LSE under filter from raw unmasked source values."""
    if not isinstance(payload_eval.get("src"), str):
        raise NotImplementedError("lse under filter currently requires a track source")
    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    for orig_idx, raw_vals in groups:
        out[orig_idx] = _logsumexp(raw_vals)
    return out


def _with_row_subs(payload: dict[str, Any], seg_df: pd.DataFrame) -> dict[str, Any]:
    """*payload* for scoring the parts in *seg_df*.

    gextract_variants hands a sequence vtrack one substitution per iterator
    interval (``row_subs``: positions and base codes); each unmasked part is
    scored with its interval's substitution.
    """
    row_subs = payload.get("row_subs")
    if row_subs is None:
        return payload
    out = {k: v for k, v in payload.items() if k != "row_subs"}
    idx = seg_df["orig_idx"].to_numpy(dtype=_numpy.int64)
    out["sub_pos"] = row_subs[0][idx]
    out["sub_base"] = row_subs[1][idx]
    return out


def _compute_filtered_segment_logsumexp(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """Log-sum-exp composition over independently scored unmasked segments."""
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    payload_seg = _with_row_subs(dict(payload_eval), seg_df)
    payload_seg["sshift"] = 0
    payload_seg["eshift"] = 0
    seg_vals = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            payload_seg,
            _df2pymisha(seg_df[["chrom", "start", "end"]]),
            CONFIG,
        ),
        dtype=float,
    )

    per_vals: list[list[float]] = [[] for _ in range(len(intervals))]
    for orig_idx, seg_val in zip(seg_df["orig_idx"], seg_vals, strict=False):
        per_vals[int(orig_idx)].append(float(seg_val))

    for i, vals in enumerate(per_vals):
        out[i] = _log_sum_parts(vals)
    return out


def _log_sum_parts(vals: list[float]) -> float:
    """R's SequenceVarProcessor reduction of a pwm / potts filter's parts.

    Pairwise log_sum_log() in part order, seeded from the first scorable
    (non-NaN) part, skipping -inf: the same steps R takes, so the same last
    bit. A vectorised log-sum-exp lands up to an ulp away. NaN when no part
    was scorable.
    """
    lse = _numpy.nan
    for v in vals:
        if _math.isnan(v):
            continue
        if _math.isnan(lse) or not _math.isfinite(lse):
            lse = v
        elif _math.isfinite(v):
            if lse > v:
                lse += _math.log(1 + _math.exp(v - lse))
            else:
                lse = v + _math.log(1 + _math.exp(lse - v))
    return lse


def _compute_filtered_max_pos_by_score(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
) -> np.ndarray:
    """A ``*.max.pos`` func under a filter, as R's SequenceVarProcessor does it.

    The position alone cannot be compared between two unmasked parts, so each
    part is also asked for the score its position was chosen by - the
    scorer's ``get_last_max_score()``, returned by ``pm_vtrack_compute`` under
    the spec key ``last_max_score`` - and the part with the highest one wins,
    the first on a tie. Its position is then offset into the iterator
    interval's own coordinate frame, re-signed so the offset moves it away from
    zero in whichever direction it already points. Serves ``pwm.max.pos`` and
    ``potts.max.pos``.

    That score includes the spatial weight. The ranking used to be rebuilt
    from two single-strand ``*.max`` scans, which is the same number without
    ``spat_factor``; with it, the reverse-strand scan read the reverse
    complement, whose spatial bins count from the 3' end, so the parts were
    ranked by weights the position was never chosen by.
    """
    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    payload_pos = _with_row_subs(dict(payload_eval), seg_df)
    payload_pos["sshift"] = 0
    payload_pos["eshift"] = 0
    seg_intervals = _df2pymisha(seg_df[["chrom", "start", "end"]])
    seg_pos = _numpy.asarray(_pymisha.pm_vtrack_compute(payload_pos, seg_intervals, CONFIG), dtype=float)
    seg_scores = _numpy.asarray(
        _pymisha.pm_vtrack_compute(dict(payload_pos, last_max_score=1), seg_intervals, CONFIG),
        dtype=float,
    )

    best_score = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    for orig_idx, seg_start, base_start, score, pos in zip(
        seg_df["orig_idx"],
        seg_df["start"],
        seg_df["base_start"],
        seg_scores,
        seg_pos, strict=False,
    ):
        if _numpy.isnan(score) or _numpy.isnan(pos):
            continue
        i = int(orig_idx)
        if _numpy.isnan(best_score[i]) or score > best_score[i]:
            offset = int(seg_start) - int(base_start)
            sign = -1.0 if pos < 0 else 1.0
            mapped_pos = sign * (abs(float(pos)) + float(offset))
            out[i] = mapped_pos
            best_score[i] = float(score)

    return out


def _global_percentile_reference_values(src: str, bin_size: int) -> np.ndarray:
    key = (str(_shared._GROOT), str(src), int(bin_size))
    stamp = _source_stamp(src)
    cached = _GLOBAL_PERCENTILE_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    from .extract import gextract
    from .intervals import gintervals_all

    extracted = gextract(src, gintervals_all(), iterator=int(bin_size))
    if extracted is None or len(extracted) == 0:
        ref = _numpy.array([], dtype=float)
    else:
        data_cols = [c for c in extracted.columns if c not in {"chrom", "start", "end", "intervalID"}]
        if not data_cols:
            ref = _numpy.array([], dtype=float)
        else:
            ref = extracted[data_cols[0]].to_numpy(dtype=float, copy=False)
            ref = ref[~_numpy.isnan(ref)]
            ref.sort()
    _GLOBAL_PERCENTILE_CACHE[key] = (stamp, ref)
    return ref


def _percentile_from_reference(values: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    out = _numpy.full(values.shape, _numpy.nan, dtype=float)
    if ref_sorted.size == 0:
        return out
    valid = ~_numpy.isnan(values)
    if not valid.any():
        return out
    ranks = _numpy.searchsorted(ref_sorted, values[valid], side="right")
    out[valid] = ranks.astype(float) / float(ref_sorted.size)
    return out


def _load_pv_table(src: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read R's frozen ``vars/pv.percentiles`` binned quantile table for *src*.

    Returns ``(bins, breaks)`` - the percentile values and their track-value
    thresholds (same length) - or ``None`` if the track has no such file (a
    pymisha-created track, or one never prepared for percentile queries).
    """
    import os

    key = (str(_shared._GROOT), str(src))
    stamp = _source_stamp(src)
    cached = _PV_TABLE_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    result: tuple[np.ndarray, np.ndarray] | None = None
    try:
        track_path = _pymisha.pm_track_path(src)
    except _pymisha.error:
        # *src* is an expression or a vtrack rather than a physical track:
        # there is no frozen percentile table to read.
        _logger.debug("no track path for %r; no frozen percentile table", src, exc_info=True)
        track_path = None
    if track_path:
        fpath = os.path.join(track_path, "vars", "pv.percentiles")
        if os.path.exists(fpath):
            from ._r_serialize import read as _r_read

            obj = _r_read(fpath)
            attrs = getattr(obj, "attributes", None)
            breaks = None if attrs is None else attrs.get("breaks")
            if breaks is not None:
                bins = _numpy.asarray(obj, dtype=float).ravel()
                br = _numpy.asarray(breaks, dtype=float).ravel()
                if br.size and _numpy.isnan(br).all():
                    # the table of a track with no values (c(1, 1), breaks c(NaN, NaN)):
                    # nothing to look up, every percentile is NaN
                    result = (_numpy.empty(0), _numpy.empty(0))
                elif bins.size == br.size and bins.size >= 2:
                    _check_breaks_na(br)
                    result = (bins, br)
    _PV_TABLE_CACHE[key] = (stamp, result)
    return result


def _percentile_from_pv_table(
    values: np.ndarray, bins: np.ndarray, breaks: np.ndarray
) -> np.ndarray:
    """Map per-bin statistics through R's binned pv.percentiles table.

    Mirrors R ``TrackVarProcessor`` + ``BinFinder::val2bin`` (right-closed
    bins): ``bin = val2bin(val)``; ``bins[bin]`` in range; for out-of-range,
    ``bins[0]`` when ``val <= breaks[0]`` else ``1.0``. NaN stays NaN, and an
    empty table (a track with no values) gives NaN.
    """
    from .summary import _bin_values

    out = _numpy.full(values.shape, _numpy.nan, dtype=float)
    valid = ~_numpy.isnan(values)
    if not valid.any() or not bins.size:
        return out
    vals = values[valid]
    bin_idx = _bin_values(vals, breaks, include_lowest=False)
    res = _numpy.where(vals <= breaks[0], bins[0], 1.0)
    in_range = bin_idx >= 0
    res[in_range] = bins[bin_idx[in_range]]
    out[valid] = res
    return out


def _compute_filtered_global_percentile(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    filter_df: pd.DataFrame | None,
    func: str,
) -> np.ndarray:
    """global.percentile* under filter using raw unmasked bins and global dense reference."""
    from .tracks import gtrack_info

    src = payload_eval.get("src")
    if not isinstance(src, str):
        raise NotImplementedError("global.percentile* under filter requires a dense track source")

    info = gtrack_info(src)
    track_type = str(info.get("type", "")).lower()
    bin_size = info.get("bin_size") or info.get("bin.size")
    if track_type != "dense" or bin_size is None:
        raise NotImplementedError("global.percentile* under filter requires a dense track source")

    groups, out = _extract_raw_unmasked_values(intervals, payload_eval, filter_df)
    from collections.abc import Callable
    stat_fn: Callable[..., Any]
    if func == "global.percentile":
        stat_fn = _numpy.mean
    elif func == "global.percentile.min":
        stat_fn = _numpy.min
    else:
        stat_fn = _numpy.max

    stats = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    for orig_idx, raw_vals in groups:
        stats[orig_idx] = float(stat_fn(raw_vals))

    table = _load_pv_table(src)
    if table is not None:
        return _percentile_from_pv_table(stats, table[0], table[1])
    ref = _global_percentile_reference_values(src, int(bin_size))
    return _percentile_from_reference(stats, ref)


def _compute_global_percentile_unfiltered(
    intervals: pd.DataFrame,
    payload_eval: dict[str, Any],
    func: str,
) -> np.ndarray:
    """global.percentile* without filter using C++ per-interval stats + Python reference CDF."""
    from .tracks import gtrack_info

    src = payload_eval.get("src")
    if not isinstance(src, str):
        raise NotImplementedError("global.percentile* requires a dense track source")

    info = gtrack_info(src)
    track_type = str(info.get("type", "")).lower()
    bin_size = info.get("bin_size") or info.get("bin.size")
    if track_type != "dense" or bin_size is None:
        raise NotImplementedError("global.percentile* requires a dense track source")

    stat_payload = dict(payload_eval)
    if func == "global.percentile":
        stat_payload["func"] = "avg"
    elif func == "global.percentile.min":
        stat_payload["func"] = "min"
    else:
        stat_payload["func"] = "max"

    stats = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            stat_payload,
            _df2pymisha(intervals),
            CONFIG,
        ),
        dtype=float,
    )

    # Path A (R parity): map through R's frozen binned pv.percentiles table.
    # Path B (fallback): the exact empirical CDF over native bins, used when
    # the track has no pv.percentiles file (e.g. pymisha-created tracks).
    table = _load_pv_table(src)
    if table is not None:
        return _percentile_from_pv_table(stats, table[0], table[1])
    ref = _global_percentile_reference_values(src, int(bin_size))
    return _percentile_from_reference(stats, ref)


def _project_intervals_by_dim(intervals: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Project 2D intervals to 1D by selecting a dimension.

    Parameters
    ----------
    intervals : DataFrame
        Intervals DataFrame, potentially with 2D columns
        (chrom1/start1/end1/chrom2/start2/end2).
    dim : int
        Dimension to project onto: 1 for (chrom1, start1, end1),
        2 for (chrom2, start2, end2).

    Returns
    -------
    DataFrame
        1D intervals with columns (chrom, start, end).
    """
    if dim == 1:
        return _pandas.DataFrame({
            "chrom": intervals["chrom1"].values,
            "start": intervals["start1"].values,
            "end": intervals["end1"].values,
        })
    if dim == 2:
        return _pandas.DataFrame({
            "chrom": intervals["chrom2"].values,
            "start": intervals["start2"].values,
            "end": intervals["end2"].values,
        })
    raise ValueError(f"dim must be 1 or 2, got {dim}")


def _array_slice_spec(cfg: dict) -> dict[str, Any]:
    """The slice of an array-slice vtrack (gvtrack_array_slice) as the C++ spec keys read by
    PMTrackExpressionVars::configure_array_slice: the column indices sorted and de-duplicated, as
    R; the slice function and its percentile."""
    sl = cfg.get("slice") or {}
    cols = sl.get("slice")
    spec: dict[str, Any] = {
        "slice": None if cols is None else sorted({int(c) for c in cols}),
        "slice_func": str(sl.get("func", "avg")).lower(),
    }
    params = sl.get("params")
    if params is not None:
        spec["slice_percentile"] = float(params[0] if isinstance(params, (list, tuple)) else params)
    return spec


def _compute_vtrack_values(
    vtrack_name: str,
    intervals: pd.DataFrame,
    subs: tuple[np.ndarray, np.ndarray] | None = None,
) -> Any:
    """
    Compute values for a virtual track.

    Virtual tracks are evaluated by:
    1. Creating shifted intervals (sshift, eshift)
    2. Extracting source track values for shifted intervals
    3. Applying aggregation function
    4. Mapping results back to original intervals

    *subs* (gextract_variants) gives one substitution per interval, as
    positions and base codes (0..3 = A, C, G, T; -1 = none), applied to the
    sequence while a sequence vtrack scores that interval. Other vtracks do
    not read the sequence and ignore it.

    Returns numpy array of computed values (one per interval).
    """
    vtrack_config = _shared._VTRACKS.get(vtrack_name)
    if vtrack_config is None:
        return None

    # Handle dim parameter: project 2D intervals to 1D
    dim = vtrack_config.get("dim")
    if dim is not None and dim != 0 and "chrom1" in intervals.columns:
        intervals = _project_intervals_by_dim(intervals, dim)

    payload = dict(vtrack_config)
    # an array-slice vtrack: the slice as the C++ spec keys; the vtrack's func, shifts and filter
    # apply as for any track source
    if payload.pop("kind", None) == "array_slice":
        payload.update(_array_slice_spec(vtrack_config))
    src = payload.get('src')
    if isinstance(src, str) and "chrom1" not in intervals.columns:
        from .tracks import gtrack_exists, gtrack_info

        if gtrack_exists(src) and int(gtrack_info(src).get("dimensions", 1) or 1) == 2:
            # R's TrackExpressionVars::init message
            raise ValueError(f"Virtual track {vtrack_name}: 1D iterator is applied to a 2D track {src}")
    if isinstance(src, _pandas.DataFrame):
        payload['src_df'] = src.copy()
        payload['src'] = _df2pymisha(src)

    # Ensure pssm is passed as a numpy array with correct ordering
    if 'pssm' in payload and isinstance(payload['pssm'], _pandas.DataFrame):
        payload['pssm'] = payload['pssm'].to_numpy(dtype=float, copy=False)

    func = str(payload.get("func", "avg")).lower()
    row_subs = None
    if subs is not None and payload.get("src") is None:
        row_subs = (_numpy.asarray(subs[0], dtype=_numpy.int64), _numpy.asarray(subs[1], dtype=_numpy.int64))

    filter_df = payload.get("filter")
    if filter_df is None or (isinstance(filter_df, _pandas.DataFrame) and len(filter_df) == 0):
        # C++ backend does not support global.percentile* yet.
        if func in _FILTER_GLOBAL_PERCENTILE_FUNCS:
            payload_eval = dict(payload)
            payload_eval.pop("filter", None)
            payload_eval.pop("filter_key", None)
            payload_eval.pop("filter_stats", None)
            return _compute_global_percentile_unfiltered(intervals, payload_eval, func)
        if row_subs is not None:
            payload["sub_pos"], payload["sub_base"] = row_subs
        return _pymisha.pm_vtrack_compute(
            payload,
            _df2pymisha(intervals),
            CONFIG
        )

    payload_eval = dict(payload)
    payload_eval.pop("filter", None)
    payload_eval.pop("filter_key", None)
    payload_eval.pop("filter_stats", None)
    if row_subs is not None:
        payload_eval["row_subs"] = row_subs

    if payload.get("src_df") is not None and func in _VALUE_DF_PY_FUNCS:
        return _compute_value_df_vtrack(intervals, payload_eval, filter_df)

    if func in _FILTER_PASSTHROUGH_FUNCS:
        return _pymisha.pm_vtrack_compute(payload_eval, _df2pymisha(intervals), CONFIG)

    if func not in _FILTER_SUPPORTED_FUNCS:
        raise NotImplementedError(
            f"gvtrack.filter for function '{func}' is not yet supported in PyMisha"
        )

    # --- nearest: first-unmasked-segment semantics ---
    if func in _FILTER_NEAREST_FUNCS:
        return _compute_filtered_nearest(intervals, payload_eval, filter_df)

    # --- quantile: raw-value extraction + numpy quantile ---
    if func in _FILTER_QUANTILE_FUNCS:
        return _compute_filtered_quantile(intervals, payload_eval, filter_df)

    # --- stddev: raw-value extraction + numpy stddev ---
    if func in _FILTER_STDDEV_FUNCS:
        return _compute_filtered_stddev(intervals, payload_eval, filter_df)

    # --- lse: raw-value extraction + logsumexp ---
    if func == "lse":
        return _compute_filtered_lse(intervals, payload_eval, filter_df)

    # --- global.percentile*: percentile of filtered per-interval statistic ---
    if func in _FILTER_GLOBAL_PERCENTILE_FUNCS:
        return _compute_filtered_global_percentile(intervals, payload_eval, filter_df, func)

    # --- pwm / potts: combine segment scores by logsumexp ---
    if func in ("pwm", "potts"):
        return _compute_filtered_segment_logsumexp(intervals, payload_eval, filter_df)

    # --- pwm.max.pos / potts.max.pos: the position from the segment holding
    # the highest score, in the interval's own coordinates ---
    if func in _FILTER_MAX_POS_BY_SCORE_FUNCS:
        return _compute_filtered_max_pos_by_score(intervals, payload_eval, filter_df)

    # --- max.pos.* / min.pos.*: find position of extremum across unmasked bins ---
    if func in _FILTER_MAX_POS_ABS_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "max", False)
    if func in _FILTER_MAX_POS_REL_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "max", True)
    if func in _FILTER_MIN_POS_ABS_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "min", False)
    if func in _FILTER_MIN_POS_REL_FUNCS:
        return _compute_filtered_extremum_pos(intervals, payload_eval, filter_df, "min", True)

    if func in _FILTER_EDIT_DISTANCE_FUNCS:
        return _compute_filtered_edit_distance(intervals, payload_eval, filter_df, func)

    # sum, size and avg of a track: each bin or track interval once, however many unmasked
    # parts it overlaps (R misha 5.15)
    if isinstance(payload_eval.get("src"), str) and func in ("sum", "size", "avg", "mean"):
        return _filtered_item_reduce(intervals, payload_eval, filter_df, func)

    # sample and sample.pos.* of a track: a uniform draw over the unmasked values, each value once
    # (R misha 5.17)
    if isinstance(payload_eval.get("src"), str) and func in (
            _FILTER_SAMPLE_FUNCS | _FILTER_SAMPLE_POS_ABS_FUNCS | _FILTER_SAMPLE_POS_REL_FUNCS):
        return _compute_filtered_sample(intervals, payload_eval, filter_df, func)

    seg_df = _build_unmasked_segments(intervals, payload_eval, filter_df)
    out = _numpy.full(len(intervals), _numpy.nan, dtype=float)
    if seg_df is None or len(seg_df) == 0:
        return out

    payload_eval = _with_row_subs(payload_eval, seg_df)
    payload_eval["sshift"] = 0
    payload_eval["eshift"] = 0
    seg_vals = _numpy.asarray(
        _pymisha.pm_vtrack_compute(
            payload_eval,
            _df2pymisha(seg_df[["chrom", "start", "end"]]),
            CONFIG,
        ),
        dtype=float,
    )

    # kmer.frac pools the parts by their possible positions, not their lengths (R misha 5.14.2)
    seg_weights = (_kmer_possible_positions(seg_df, payload_eval) if func == "kmer.frac"
                   else seg_df["seg_len"].to_numpy())

    per_vals: list[list[float]] = [[] for _ in range(len(intervals))]
    per_lens: list[list[int]] = [[] for _ in range(len(intervals))]
    per_starts: list[list[int]] = [[] for _ in range(len(intervals))]
    per_base_starts: list[int | None] = [None for _ in range(len(intervals))]
    for orig_idx, seg_len, seg_start, base_start, seg_val in zip(
        seg_df["orig_idx"],
        seg_weights,
        seg_df["start"],
        seg_df["base_start"],
        seg_vals, strict=False,
    ):
        i = int(orig_idx)
        per_vals[i].append(float(seg_val))
        per_lens[i].append(int(seg_len))
        per_starts[i].append(int(seg_start))
        if per_base_starts[i] is None:
            per_base_starts[i] = int(base_start)

    for i in range(len(intervals)):
        vals = per_vals[i]
        if not vals:
            continue
        arr = _numpy.asarray(vals, dtype=float)
        valid = ~_numpy.isnan(arr)
        if func == "kmer.frac":
            # R's SequenceVarProcessor: one part is scored alone; several pool their counts
            # (fraction times possible positions) over their possible positions, 0 if none
            if len(vals) == 1:
                out[i] = vals[0]
            else:
                total_positions = sum(per_lens[i])
                total_count = 0.0
                for v, n in zip(vals, per_lens[i], strict=True):
                    total_count += v * n
                out[i] = total_count / total_positions if total_positions > 0 else 0.0
        elif func in _FILTER_WEIGHTED_FUNCS:
            if not valid.any():
                continue
            lens = _numpy.asarray(per_lens[i], dtype=float)[valid]
            arr_valid = arr[valid]
            out[i] = float((arr_valid * lens).sum() / lens.sum())
        elif func in _FILTER_ADDITIVE_FUNCS:
            if not valid.any():
                continue
            out[i] = float(arr[valid].sum())
        elif func in _FILTER_WEIGHTED_SUM_FUNCS:
            # the parts' weighted sums added in order (each unmasked bp counted once)
            if valid.any():
                out[i] = float(_numpy.cumsum(_numpy.r_[0.0, arr[valid]])[-1])
        elif func in _FILTER_MIN_FUNCS:
            if not valid.any():
                continue
            out[i] = float(arr[valid].min())
        elif func in _FILTER_MAX_FUNCS:
            if not valid.any():
                continue
            out[i] = float(arr[valid].max())
        elif func in _FILTER_EXISTS_FUNCS:
            out[i] = 1.0 if _numpy.any(arr == 1.0) else 0.0
        elif func in _FILTER_SIZE_FUNCS:
            out[i] = float(_numpy.nansum(arr))
        elif func in (_FILTER_FIRST_FUNCS | _FILTER_FIRST_POS_ABS_FUNCS | _FILTER_FIRST_POS_REL_FUNCS
                      | _FILTER_LAST_FUNCS | _FILTER_LAST_POS_ABS_FUNCS | _FILTER_LAST_POS_REL_FUNCS):
            # the first (last) part that holds a value has the first (last) value (R misha 5.17)
            if not valid.any():
                continue
            j = int(_numpy.flatnonzero(valid)[0 if func.startswith("first") else -1])
            out[i] = float(arr[j])
            if func.endswith(".relative"):
                out[i] += per_starts[i][j] - (per_base_starts[i] or 0)
    return out


def gvtrack_create(
    vtrack_name: str,
    src: pd.DataFrame | str | None,
    func: str | None = None,
    params: float | str | dict[str, Any] | None = None,
    sshift: int = 0,
    eshift: int = 0,
    **kwargs: Any,
) -> None:
    """
    Create a virtual track.

    A virtual track evaluates an aggregation function over a source track,
    intervals set, or genomic sequence within each iterator interval. Virtual
    tracks can be referenced by name anywhere a track expression is accepted
    (e.g., in ``gextract``, ``gsummary``, ``gdist``). The virtual track
    persists in memory for the duration of the current session.

    Parameters
    ----------
    vtrack_name : str
        Name for the virtual track. If a virtual track with this name
        already exists, it is silently overwritten.
    src : str, pandas.DataFrame, or None
        Source for the virtual track. Can be:

        - A track name (str) -- any track in the database (dense, sparse,
          array, or 2D).
        - An intervals set name (str) -- used with interval-based functions
          like ``'distance'``, ``'coverage'``.
        - A DataFrame with columns ``chrom``, ``start``, ``end`` and one
          numeric value column -- acts as an in-memory sparse (value-based)
          track. Intervals must not overlap.
        - ``None`` -- for sequence-based functions (``'pwm'``, ``'pwm.max'``,
          ``'pwm.count'``, ``'potts'``, ``'potts.max'``, ``'potts.count'``,
          ``'kmer.count'``, ``'kmer.frac'``, ``'masked.count'``,
          ``'masked.frac'``).
    func : str, optional
        Aggregation function to apply. When omitted, defaults to ``'avg'``
        for a track or value-bearing source and to ``'distance'`` for an
        intervals-set source (R parity). Supported functions include:

        - **Track-based**: ``'avg'``, ``'sum'``, ``'weighted.sum'``, ``'min'``,
          ``'max'``, ``'stddev'``, ``'nearest'``, ``'quantile'``, ``'coverage'``,
          ``'exists'``, ``'size'``, ``'first'``, ``'last'``, ``'sample'``,
          ``'lse'``, ``'global.percentile'``. ``'sum'`` adds the value of every
          bin or interval the window touches, in full; ``'weighted.sum'`` adds
          each value times the number of its bp inside the window (after
          ``sshift``/``eshift``, clipped to the chromosome), skipping NaN, and is
          NaN when the window holds no value. A dense track of bin size 20
          under a 312 bp window gives ``'sum'`` of 16 or 17 bins by the
          window's phase and ``'weighted.sum'`` of 312 bp. Also for a value-based
          (DataFrame) source.
        - **Distance-based** (intervals source): ``'distance'``,
          ``'distance.center'``, ``'distance.edge'``, ``'neighbor.count'``
        - **Position-based**: ``'first.pos.abs'``, ``'first.pos.relative'``,
          ``'last.pos.abs'``, ``'last.pos.relative'``,
          ``'min.pos.abs'``, ``'min.pos.relative'``,
          ``'max.pos.abs'``, ``'max.pos.relative'``,
          ``'sample.pos.abs'``, ``'sample.pos.relative'``
        - **2D track**: ``'area'``, ``'weighted.sum'``, ``'exists'``,
          ``'size'``, ``'first'``, ``'last'``, ``'sample'``,
          ``'global.percentile'``. ``'avg'``, ``'min'``, ``'max'`` and
          ``'weighted.sum'`` skip NaN values, as the 1D summarizers do, and
          give NaN when no value overlaps the interval; ``'avg'`` is the mean
          over the area that has a value. ``'area'`` counts every rectangle,
          NaN ones included.
        - **Motif/PWM** (src=None): ``'pwm'``, ``'pwm.max'``,
          ``'pwm.max.pos'``, ``'pwm.count'``
        - **Potts / pairwise energy** (src=None): ``'potts'``,
          ``'potts.max'``, ``'potts.max.pos'``, ``'potts.count'``
        - **Edit distance** (src=None): ``'pwm.edit_distance'``,
          ``'pwm.edit_distance.pos'``, ``'pwm.max.edit_distance'``,
          ``'pwm.edit_distance.lse'``, ``'pwm.edit_distance.lse.pos'``,
          ``'pwm.n_mutations'``

        The ``*.pos`` funcs all report the 1-based position of the **first
        base** of the match in **forward-strand** orientation, on either
        strand and under either ``bidirect`` setting. Under
        ``bidirect=True`` the sign says which strand won, not which end is
        pointed at, so a span is ``pos .. pos + len(pssm) - 1`` in forward
        coordinates in every case. Ties break by scan order, which is
        strand-dependent: ``strand=-1`` scans a reverse-complemented buffer
        and so keeps the most 3' tied anchor. (This convention holds from
        pymisha 0.12.0 / misha 5.11.25; older builds returned an off-by-one
        position for ``strand=-1`` and reverse-complemented coordinates from
        ``pwm.edit_distance.pos``.)
        - **K-mer** (src=None): ``'kmer.count'``, ``'kmer.frac'``
        - **Masked sequence** (src=None): ``'masked.count'``,
          ``'masked.frac'``
    params : float, str, dict, or None, optional
        Function-specific parameter. For example, a percentile in [0, 1]
        for ``'quantile'`` or a max-distance integer for
        ``'neighbor.count'``. For the pwm, edit distance, potts and kmer
        functions, a dict of the keyword arguments below (R's ``params``
        list), as an alternative to passing them as keyword arguments; a
        call may not use both.
    sshift : int, default 0
        Shift added to the start coordinate of each iterator interval
        before the virtual track function is evaluated.
    eshift : int, default 0
        Shift added to the end coordinate of each iterator interval
        before the virtual track function is evaluated.
    **kwargs
        Additional keyword arguments, depending on ``func``:

        - ``pssm`` (numpy.ndarray or pandas.DataFrame) -- Position-specific
          scoring matrix with 4 columns (A, C, G, T) for PWM functions. Each
          row holds the probabilities or counts of the four bases; log-odds
          matrices, NaN values and rows summing to 0 are rejected.
        - ``e``, ``J``, ``pairs``, ``intercept`` -- the Potts model, for the
          ``'potts'`` family. The score of one window is
          ``intercept + sum_i e[i, x_i] + sum_k J_k[x_i, x_j]``: ``e`` is a
          ``(W, 4)`` matrix of per-position energies (columns A, C, G, T),
          and each row of ``pairs`` names a coupled pair of positions
          (1-based, lower position first) whose ``4x4`` coupling table is the
          matching entry of ``J``, indexed with the lower position on the
          rows. The whole model is normally passed at once as ``params``; a
          fitted model can go through verbatim, since the extra fields
          ``width``, ``pair_strength``, ``attr`` and ``link`` are accepted
          and ignored (``width`` is cross-checked against ``e``).
        - ``prior`` (float) -- Pseudocount added to PSSM frequencies
          (default 0.01 for PWM functions).
        - ``bidirect`` (bool) -- If True, score both DNA strands (PWM).
        - ``extend`` (bool) -- If True (default), extend the scanned
          sequence so boundary-anchored motifs retain full context.
        - ``score_thresh`` (float) -- Score threshold. Required for
          ``'pwm.count'`` (PWM scores are log-likelihoods, so no default
          suits every PSSM); also used by the edit distance functions.
        - ``max_edits`` (int or None) -- Maximum number of edits for edit
          distance functions. None (default) uses exact computation. For
          ``'potts.max'`` it is 0 (default) or 1: with 1, each anchor scores
          the best of its own window and every window one substitution away,
          and the vtrack returns the best anchor. Exact; costs about 4 to 30
          times a plain ``'potts.max'`` (more for wide models and with
          ``bidirect=True``), and is meant as a genome-wide bound before
          :func:`gextract_variants`. A window with a non-ACGT base is still
          not scored. Any other potts function rejects it. The search of
          ``'pwm.edit_distance.lse'`` for 3 or more edits is a heuristic
          (beam search): its count can exceed the minimum, and its NaN can
          mean "not found".
        - ``direction``/``score_min``/``score_max`` also apply to
          ``'pwm.n_mutations'``, which counts the single-base substitutions
          that each independently bring a window across ``score_thresh``
          (0 when the threshold is already met, NaN when no single edit
          suffices; MAX across the windows of an iterator interval, so an
          interval that already holds a match can return more than 0). With
          ``direction='below'`` and ``bidirect=True`` it counts the
          substitutions that bring both strands below.
        - ``max_indels`` (int) -- Maximum insertions+deletions for
          ``'pwm.edit_distance'``, ``'pwm.edit_distance.pos'``,
          ``'pwm.max.edit_distance'``. Default 0 (substitutions only). An
          inserted base scores the column maximum (``'above'``) or minimum
          (``'below'``).
        - ``direction`` (str) -- Score direction for edit distance
          functions: ``'above'`` (default) finds minimum edits to raise
          score above threshold; ``'below'`` finds minimum edits to
          lower score below threshold. With ``'below'`` and
          ``bidirect=True`` both strands must end up below, so the harder
          strand decides; a strand that cannot be brought below within
          ``max_edits`` makes the window NaN.
        - ``score_min`` (float or None) -- Minimum PWM score filter for
          edit distance functions. Windows below this are skipped. The
          score is the window's ungapped score (the PSSM-length window from
          its start), also with ``max_indels > 0``.
        - ``score_max`` (float or None) -- Maximum PWM score filter for
          edit distance functions. Windows above this are skipped. Same
          score as ``score_min``.
        - ``strand`` (int) -- Strand selection: 1 (forward), -1 (reverse),
          0 (both). Used by kmer and single-strand PWM modes.
        - ``kmer`` (str) -- DNA k-mer sequence for kmer functions. The
          ``'kmer.frac'`` denominator is the number of possible anchor
          positions in the interval: every anchor with ``extend=True``
          (minus the last k-1 at a chromosome end), the first L-k+1 of an
          interval of length L with ``extend=False``; with ``strand=0``
          positions on both strands are counted, so the fraction stays in
          [0, 1].
        - ``spat_factor`` (list of float) -- Spatial weighting factors
          for PWM functions.
        - ``spat_bin`` (int) -- Bin width for spatial weighting.
        - ``spat_min`` (int) -- First scanned position (1-based, >= 1) of
          the window: anchors start no earlier than this.
        - ``spat_max`` (int) -- Last scanned position (1-based) of the
          window: the last anchor is the one whose motif ends here. Either
          bound may be given alone, with or without ``spat_factor``.
        - ``filter`` (pandas.DataFrame, str, list, or None) -- Genomic mask
          filter. See ``gvtrack_filter`` for details.

    Notes
    -----
    **The potts family.** A Potts carries energies, not probabilities, so
    there is no ``prior`` and no spatial weighting: a window containing any
    non-ACGT base is not scored and is left out of the reduction.
    ``score_thresh`` is mandatory for ``'potts.count'`` - a Potts score is an
    energy whose usable range depends on the model, so there is no default -
    and ignored by the rest of the family. ``strand`` is used only when
    ``bidirect=False``; it is clamped to 1 under ``bidirect=True``, as in the
    pwm family.

    "Counted nothing" and "nothing to count" are different answers. An
    interval that has anchors but none of them scorable (for example, all
    ``N``) gives NaN for ``'potts'``, ``'potts.max'`` and ``'potts.max.pos'``,
    and 0 for ``'potts.count'``: it counted, and found none. An interval with
    no anchor at all gives NaN for all four, ``'potts.count'`` included - an
    interval narrower than the model with ``extend=False``, or a chromosome
    end with the default ``extend=True``, where the end padding is clipped at
    the contig boundary and the fetched sequence comes back shorter than the
    model.

    ``'potts'``, ``'potts.max'`` and ``'potts.count'`` combine the two strands
    at each anchor by log-sum-exp; ``'potts.max.pos'`` takes the maximum,
    because it has to name a strand. So ``'potts.max'`` and
    ``'potts.max.pos'`` can select **different anchors**. This matches the pwm
    family exactly and is inherited on purpose rather than fixed on one side.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the filter source is invalid or refers to a non-intervals-type
        track.

    See Also
    --------
    gvtrack_info : Retrieve the configuration of a virtual track.
    gvtrack_iterator : Override iterator shifts for a virtual track.
    gvtrack_iterator_2d : Set 2D iterator shifts for a virtual track.
    gvtrack_filter : Attach or clear a genomic mask filter.
    gvtrack_rm : Remove a single virtual track.
    gvtrack_ls : List all virtual tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Create a virtual track with a max aggregation:

    >>> pm.gvtrack_create("vt_max", "dense_track", func="max")
    >>> pm.gextract("vt_max", pm.gintervals(["1"], [0], [10000]), iterator=1000)  # doctest: +SKIP

    Create a quantile virtual track with a median (0.5) parameter:

    >>> pm.gvtrack_create("vt_q50", "dense_track", func="quantile", params=0.5)

    Create a distance virtual track from an intervals source:

    >>> pm.gvtrack_create("vt_dist", "annotations", func="distance")

    Create a PWM virtual track scanning both strands:

    >>> import numpy as np
    >>> pssm = np.array([[0.7, 0.1, 0.1, 0.1],
    ...                  [0.1, 0.7, 0.1, 0.1],
    ...                  [0.1, 0.1, 0.7, 0.1],
    ...                  [0.1, 0.1, 0.1, 0.7]])
    >>> pm.gvtrack_create("motif", None, func="pwm",
    ...                   pssm=pssm, bidirect=True, prior=0.01)

    Create a k-mer counting virtual track:

    >>> pm.gvtrack_create("cg_count", None, func="kmer.count",
    ...                   kmer="CG", strand=1)
    """
    _checkroot()
    from .intervals import gintervals_exists
    from .tracks import gtrack_exists

    if gtrack_exists(vtrack_name):
        raise ValueError(f"Cannot create virtual track: regular track named {vtrack_name} already exists")
    if gintervals_exists(vtrack_name):
        raise ValueError(f"Cannot create virtual track: intervals named {vtrack_name} already exists")
    if func is None:
        func = _infer_default_vtrack_func(src)
    func_lc = str(func).lower()
    _r_vtrack_checks(vtrack_name, src, func_lc, params)
    if func_lc in _EDIT_DISTANCE_FUNCS_MAX_EDITS or func_lc == "pwm.n_mutations":
        from .sequence import _check_max_edits

        _check_max_edits(kwargs.get("max_edits"))
        _check_edit_distance_args(kwargs, func_lc)
    sshift = _shift_arg(vtrack_name, "sshift", sshift)
    eshift = _shift_arg(vtrack_name, "eshift", eshift)
    if (sshift or eshift) and _vtrack_src_is_2d({"src": src}):
        raise ValueError(f"Virtual track {vtrack_name}: 1D iterator modifier cannot be used with source "
                         "that supports only 2D iterators")

    if isinstance(src, _pandas.DataFrame):
        from .intervals import gintervals_all

        req = {"chrom", "start", "end"}
        if not req.issubset(src.columns):
            raise ValueError("DataFrame source must include columns: chrom, start, end")
        try:
            _check_intervals_strand(src)
        except _pymisha.error as e:
            raise _pymisha.error(f"Source of virtual track {vtrack_name} was not recognized neither as a track "
                                 f'nor as intervals. ("{e}")') from None

        is_interval_func = func_lc in _DF_INTERVAL_FUNCS
        value_cols = [
            c
            for c in src.columns
            if c not in {"chrom", "start", "end", "intervalID", "intervalID1", "intervalID2"}
        ]
        if not is_interval_func and not value_cols:
            raise ValueError("DataFrame source must include one value column")

        if is_interval_func:
            src_df = src.copy()
        else:
            value_col = value_cols[0]
            src_df = src[["chrom", "start", "end", value_col]].copy()
            src_df.columns = ["chrom", "start", "end", "value"]

        def source_error(msg: str) -> Exception:
            # R wraps the errors of an intervals source (no value column used)
            if is_interval_func:
                err: Exception = _pymisha.error(f"Source of virtual track {vtrack_name} was not recognized neither "
                                                f'as a track nor as intervals. ("{msg}")')
                return err
            return ValueError(msg)

        chroms, starts, ends = _vtrack_source_rows(src_df, source_error)
        src_df["chrom"], src_df["start"], src_df["end"] = chroms, starts, ends
        if not is_interval_func:
            # a missing value is kept, as R's in-memory track keeps it (no function counts it)
            src_df["value"] = _pandas.to_numeric(src_df["value"], errors="coerce")
        # by chromosome (in the database's order) and start, as R sorts the source
        chrom_order = {c: k for k, c in enumerate(_chrom_names(gintervals_all()["chrom"]).tolist())}
        src_df["_cid"] = src_df["chrom"].map(chrom_order)
        src_df = src_df.sort_values(["_cid", "start"], kind="mergesort").drop(columns="_cid").reset_index(drop=True)

        # distance.center accepts overlapping sources (a bin center inside several intervals
        # resolves to the nearest center) - matching R misha 5.9.1. Value-bearing functions
        # reject overlaps, with R's message (the position is in the sorted source, from 0).
        if not is_interval_func and len(src_df) > 1:
            c, st, en = src_df["chrom"].to_numpy(), src_df["start"].to_numpy(), src_df["end"].to_numpy()
            ovl = _numpy.flatnonzero((c[1:] == c[:-1]) & (en[:-1] > st[1:]))
            if ovl.size:
                i = int(ovl[0]) + 1
                raise ValueError(
                    f"Virtual track {vtrack_name}: overlapping intervals detected. Interval at position {i} "
                    f"[{c[i]}:{st[i]}-{en[i]}] overlaps with previous interval [{c[i - 1]}:{st[i - 1]}-{en[i - 1]}]. "
                    "Value-based virtual tracks do not support overlapping intervals.")

        src = src_df

    config = {
        'src': src,
        'func': func,
        'params': params,
        'sshift': sshift,
        'eshift': eshift,
    }

    # The potts family takes its whole model through `params`, the way R misha
    # does, so a fitted model can be handed over verbatim; the individual
    # fields are also accepted as keyword arguments.
    if func_lc in _POTTS_FUNCS:
        # A whole model object handed over as `params` may carry any number of
        # fields its fit happened to produce - far more than the eight this
        # family reads - so unknown names are IGNORED there rather than
        # rejected. That is what lets a fitted model be passed verbatim, with
        # no subsetting at the call site and no list of another package's field
        # names hardcoded here to go stale. Keyword arguments are typed by
        # hand, where an unknown name is a typo worth catching - a misspelled
        # `bidirect` would silently score one strand - so those stay checked.
        dots = dict(kwargs)
        from_params = params is not None
        if from_params:
            if not isinstance(params, dict) or "e" not in params:
                raise ValueError(
                    "potts functions require a dict with at least an 'e' matrix "
                    "parameter"
                )
            # R misha lets `params` shadow the named arguments and ignores them.
            # pymisha refuses instead: a `bidirect=False` that vanishes because
            # a model was also passed is the same silent-default failure the
            # parameter allowlist exists to catch.
            extra = sorted(k for k in kwargs if k != "filter")
            if extra:
                raise ValueError(
                    f"function {func!r} was given both a model in 'params' and "
                    f"the keyword argument(s) {', '.join(repr(k) for k in extra)}; "
                    "the model in 'params' would silently win. Put them in the "
                    "'params' dict, or pass every field as a keyword argument."
                )
            dots = dict(params)
        else:
            _check_vtrack_params(func, dots)
        if "e" not in dots:
            raise ValueError("potts functions require an 'e' matrix parameter")

        # Same contract as pwm.count: a Potts score is an energy whose usable
        # range depends entirely on the model, so no default suits every one.
        score_thresh = dots.get("score_thresh", dots.get("score.thresh"))
        if func_lc == "potts.count":
            if score_thresh is None:
                raise ValueError(
                    "potts.count requires a 'score_thresh' parameter. A Potts "
                    "score is an energy, so there is no default that suits every "
                    "model - pick a threshold from the score distribution of your "
                    "own model, e.g. with a 'potts' or 'potts.max' virtual track."
                )
            score_thresh = _coerce_score_thresh(score_thresh)
        elif score_thresh is None:
            score_thresh = 0.0

        # _potts_params() is the same builder gseq_potts() uses, so the two
        # entry points cannot drift on validation or on the dict's shape.
        payload = _potts_params(
            dots,
            bidirect=dots.get("bidirect", True),
            extend=dots.get("extend", True),
            strand=dots.get("strand", 1),
            score_thresh=score_thresh,
            what=f"virtual track function '{func}'",
        )
        # Only potts.max: for a log-sum-exp, a count or a position the best
        # single substitution does not decompose by anchor. Left out of the
        # spec at 0, so existing potts vtracks keep their spec unchanged.
        max_edits = dots.get("max_edits")
        if max_edits is not None:
            if func_lc != "potts.max":
                raise ValueError(f"max_edits is supported by potts.max only, not by '{func}'")
            if (
                isinstance(max_edits, (bool, _numpy.bool_))
                or not isinstance(max_edits, (int, float, _numpy.number))
                or max_edits not in (0, 1)
            ):
                raise ValueError(f"virtual track function '{func}': max_edits must be 0 or 1")
            if max_edits == 1:
                payload["max_edits"] = 1
        config.update(payload)
        # The model now lives in the spec's own fields; leaving it under
        # `params` as well would hand the C++ side a dict where it reads a
        # number.
        config["params"] = None
        if "filter" in kwargs:
            config["filter"] = kwargs["filter"]
    elif func_lc in ("kmer.count", "kmer.frac"):
        config.update(_kmer_params(vtrack_name, func, params, kwargs))
        # The k-mer now lives in the spec's own fields.
        config["params"] = None
        if "filter" in kwargs:
            config["filter"] = kwargs["filter"]
    else:
        if params is not None and (func_lc in _PWM_FUNCS or func_lc in _FILTER_EDIT_DISTANCE_FUNCS):
            # R's pwm handlers take the parameters as one list, like the
            # keyword arguments.
            _check_params_dots_collision(func, params, kwargs)
            if not isinstance(params, dict) or "pssm" not in params:
                raise ValueError(f"{func} requires a dict with at least 'pssm' matrix parameter")
            kwargs = {**params, **kwargs}
            config["params"] = None
        _check_vtrack_params(func, kwargs)
        config.update(kwargs)

    if str(config.get("func", "")).startswith("pwm"):
        _check_pwm_flags(vtrack_name, func_lc, config)
        if config.get("pssm") is not None:
            config["pssm"] = _coerce_pssm_matrix(config["pssm"])

        # A missing `strand` under `bidirect=False` reached C++ as strand 0,
        # where `scan_fwd = bidirect || strand == 1` and
        # `scan_rev = bidirect || strand == -1` are BOTH false: every anchor
        # scored -inf, so pwm.count returned 0 and the rest returned -inf, with
        # no error. R defaults it to 1, and the potts family here already does
        # (see _potts_params above); the pwm family was the one that did not.
        if config.get("bidirect", True) is False and config.get("strand") is None:
            config["strand"] = 1

        # R misha requires an explicit threshold wherever one decides what is
        # counted or reached, and coerces it the same way at every site.
        if func_lc == "pwm.count" or func_lc in _FILTER_EDIT_DISTANCE_FUNCS:
            thresh = config.get("score_thresh", config.get("score.thresh"))
            if thresh is None:
                what = (
                    "pwm.count" if func_lc == "pwm.count"
                    else "pwm edit distance functions"
                )
                raise ValueError(
                    f"{what} require a 'score_thresh' parameter. PWM scores are "
                    "log-likelihoods, so there is no default that suits every "
                    "PSSM - pick a threshold from the score distribution of "
                    "your own matrix, e.g. with a 'pwm' or 'pwm.max' virtual "
                    "track."
                )
            config["score_thresh"] = _coerce_score_thresh(thresh)
            config.pop("score.thresh", None)

        spat_factor = config.get("spat_factor")
        if spat_factor is not None:
            try:
                spat_vals = _numpy.asarray(spat_factor, dtype=float).ravel()
            except Exception as exc:
                raise ValueError("spat_factor must be a numeric array-like") from exc
            if spat_vals.size == 0:
                raise ValueError("spat_factor must be non-empty")
            if not _numpy.isfinite(spat_vals).all():
                raise ValueError("spat_factor must contain only finite values")
            if _numpy.any(spat_vals <= 0):
                raise ValueError("spat_factor must contain only positive values")
            config["spat_factor"] = spat_vals.tolist()

        if "spat_bin" in config and config.get("spat_bin") is not None:
            try:
                spat_bin = int(config["spat_bin"])  # type: ignore[arg-type]
            except Exception as exc:
                raise ValueError("spat_bin must be a positive integer") from exc
            if spat_bin <= 0:
                raise ValueError("spat_bin must be a positive integer")
            config["spat_bin"] = spat_bin

        _convert_spat_range(config)

    # Validate direction parameter for edit distance functions
    if func_lc in _FILTER_EDIT_DISTANCE_FUNCS:
        direction = config.get("direction", "above")
        if direction is not None:
            direction = str(direction).lower()
            if direction not in ("above", "below"):
                raise ValueError("direction must be 'above' or 'below'")
            config["direction"] = direction

    # For PWM virtual tracks, if pssm is a DataFrame, ensure column order A, C, G, T
    if str(config.get('func', '')).startswith('pwm'):
        pssm = config.get('pssm')
        if isinstance(pssm, _pandas.DataFrame):
            # Check if we have A, C, G, T columns (case-insensitive)
            cols = [c.upper() for c in pssm.columns]
            if {'A', 'C', 'G', 'T'}.issubset(set(cols)):
                # Reorder to standard ACGT
                col_map = {c.upper(): c for c in pssm.columns}
                pssm = pssm[[col_map['A'], col_map['C'], col_map['G'], col_map['T']]]
                config['pssm'] = pssm

    filt = config.get("filter")
    if filt is not None:
        filt_df = _resolve_filter_sources(filt)
        config["filter"] = filt_df if len(filt_df) > 0 else None
        config["filter_key"] = _filter_key(filt_df)
        config["filter_stats"] = _filter_stats(filt_df)
    else:
        config["filter"] = None
        config["filter_key"] = None
        config["filter_stats"] = None

    _shared._VTRACKS[vtrack_name] = config


def gvtrack_ls() -> list[str]:
    """
    List all currently defined virtual tracks.

    Returns the names of all virtual tracks that have been created in the
    current session via ``gvtrack_create``. Unlike the R counterpart, this
    function does not support pattern filtering; use standard Python list
    comprehensions to filter the result if needed.

    Returns
    -------
    list of str
        Names of all virtual tracks in the current session. Returns an
        empty list if no virtual tracks have been created.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_info : Retrieve configuration of a virtual track.
    gvtrack_rm : Remove a single virtual track.
    gvtrack_clear : Remove all virtual tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_ls()
    []

    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> pm.gvtrack_create("vt2", "dense_track", func="max")
    >>> pm.gvtrack_ls()
    ['vt1', 'vt2']

    Filter with a list comprehension:

    >>> [v for v in pm.gvtrack_ls() if "2" in v]
    ['vt2']
    """
    return list(_shared._VTRACKS.keys())


def gvtrack_info(vtrack_name: str) -> dict[str, Any]:
    """
    Return the definition of a virtual track.

    Retrieves the full internal configuration dictionary for a previously
    created virtual track. This is useful for inspecting or programmatically
    modifying virtual track settings.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.

    Returns
    -------
    dict
        A copy of the virtual track configuration dictionary. Keys always
        include ``'src'``, ``'func'``, ``'params'``, ``'sshift'``,
        ``'eshift'``, ``'filter'``, ``'filter_key'``, and
        ``'filter_stats'``. Additional keys (e.g., ``'pssm'``,
        ``'bidirect'``, ``'kmer'``, ``'dim'``) are present when supplied
        at creation time or via ``gvtrack_iterator`` /
        ``gvtrack_iterator_2d``.

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_ls : List all virtual tracks.
    gvtrack_filter : Attach or clear a genomic mask filter.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_create("vt1", "dense_track", func="max")
    >>> info = pm.gvtrack_info("vt1")
    >>> info["func"]
    'max'
    >>> info["src"]
    'dense_track'
    >>> info["sshift"]
    0
    """
    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")
    return dict(_shared._VTRACKS[vtrack_name])


def _shift_arg(vtrack_name: str, name: str, value: Any) -> int:
    """An iterator shift as R takes it: a finite number, a fraction truncated."""
    if (value is None or value is _pandas.NA or isinstance(value, (bool, _numpy.bool_))
            or not isinstance(value, numbers.Real) or not _math.isfinite(value)):
        raise ValueError(f"Virtual track {vtrack_name}: {name} must be an integer")
    return int(float(value))


def gvtrack_iterator(vtrack_name: str, dim: int | None = None, sshift: int = 0, eshift: int = 0) -> None:
    """
    Define modification rules for the 1D iterator of a virtual track.

    By default a virtual track is evaluated over the same iterator intervals
    as the calling function (e.g., ``gextract``, ``gsummary``). This function
    allows independent control of the genomic window the virtual track sees
    by applying custom start/end shifts. It can also project a 2D iterator
    down to one of its 1D dimensions.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.
    dim : int or None, optional
        Dimension projection for 2D iterators:

        - ``None`` or ``0`` -- no conversion; shifts apply to the 1D
          iterator directly.
        - ``1`` -- convert a 2D iterator interval ``(chrom1, start1, end1,
          chrom2, start2, end2)`` to ``(chrom1, start1, end1)`` before
          applying shifts.
        - ``2`` -- convert to ``(chrom2, start2, end2)`` before applying
          shifts.
    sshift : int, default 0
        Value added to the start coordinate of each iterator interval.
        Negative values expand the window upstream.
    eshift : int, default 0
        Value added to the end coordinate of each iterator interval.
        Positive values expand the window downstream.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_iterator_2d : Set 2D iterator shifts for a virtual track.
    gvtrack_filter : Attach a genomic mask filter.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Shift the evaluation window 200 bp downstream:

    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> pm.gvtrack_iterator("vt1", sshift=200, eshift=200)
    >>> pm.gextract("dense_track", "vt1",  # doctest: +SKIP
    ...             pm.gintervals(["1"], [0], [500]))

    Expand the window symmetrically by 500 bp in each direction:

    >>> pm.gvtrack_create("vt2", "dense_track", func="sum")
    >>> pm.gvtrack_iterator("vt2", sshift=-500, eshift=500)

    Project dimension 1 of a 2D iterator for a 1D virtual track:

    >>> pm.gvtrack_create("vt3", "dense_track", func="avg")
    >>> pm.gvtrack_iterator("vt3", dim=1)
    """
    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")

    cfg = _shared._VTRACKS[vtrack_name]
    if _vtrack_src_is_2d(cfg):
        raise ValueError(f"Virtual track {vtrack_name}: 1D iterator modifier cannot be used with source "
                         "that supports only 2D iterators")
    sshift = _shift_arg(vtrack_name, "sshift", sshift)
    eshift = _shift_arg(vtrack_name, "eshift", eshift)
    if dim is not None:
        cfg['dim'] = dim
    cfg['sshift'] = sshift
    cfg['eshift'] = eshift
    _shared._VTRACKS[vtrack_name] = cfg


def _vtrack_src_is_2d(cfg: dict) -> bool:
    """Whether a vtrack's source is a 2D track (R's 2D-only iterator source)."""
    src = cfg.get("src")
    if not isinstance(src, str):
        return False
    from .tracks import gtrack_exists, gtrack_info

    return gtrack_exists(src) and str(gtrack_info(src).get("type", "")) in _R_2D_TYPES


def gvtrack_iterator_2d(
    vtrack_name: str,
    sshift1: int = 0,
    eshift1: int = 0,
    sshift2: int = 0,
    eshift2: int = 0,
) -> None:
    """
    Define modification rules for the 2D iterator of a virtual track.

    Sets independent start/end shifts for both dimensions of a 2D iterator
    interval. The shifts are added to the coordinates of each 2D iterator
    interval before the virtual track function is evaluated.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.
    sshift1 : int, default 0
        Value added to the ``start1`` coordinate of each 2D iterator
        interval.
    eshift1 : int, default 0
        Value added to the ``end1`` coordinate of each 2D iterator
        interval.
    sshift2 : int, default 0
        Value added to the ``start2`` coordinate of each 2D iterator
        interval.
    eshift2 : int, default 0
        Value added to the ``end2`` coordinate of each 2D iterator
        interval.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_iterator : Set 1D iterator shifts or project a 2D dimension.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_create("vt_2d", "rects_track", func="area")
    >>> pm.gvtrack_iterator_2d("vt_2d", sshift1=1000, eshift1=2000)
    >>> pm.gextract("rects_track", "vt_2d",  # doctest: +SKIP
    ...             pm.gintervals_2d(["1"], [0], [5000], ["2"], [0], [5000]))
    """
    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")

    cfg = _shared._VTRACKS[vtrack_name]
    if not _vtrack_src_is_2d(cfg):
        raise ValueError(f"Virtual track {vtrack_name}: 2D iterator modifier cannot be used with source "
                         "that supports only 1D iterators")
    shifts = {k: _shift_arg(vtrack_name, k, v) for k, v in
              (("sshift1", sshift1), ("eshift1", eshift1), ("sshift2", sshift2), ("eshift2", eshift2))}
    cfg['itr_type'] = '2d'
    cfg.update(shifts)
    _shared._VTRACKS[vtrack_name] = cfg


def gvtrack_filter(vtrack_name: str, mask: pd.DataFrame | str | list[Any] | None = None, **kwargs: Any) -> None:
    """
    Attach or clear a genomic mask filter on a virtual track.

    When a filter is attached, the virtual track function is evaluated only
    over the *unmasked* regions -- that is, regions NOT covered by the filter
    intervals. Masked positions are excluded from aggregation, and an
    iterator interval that is entirely masked returns NaN. The filter
    persists on the virtual track until explicitly cleared.

    Filters are applied *after* iterator modifiers (``sshift``/``eshift``/
    ``dim``). The order of operations is: (1) apply iterator shifts,
    (2) subtract mask from the shifted intervals, (3) evaluate the virtual
    track function over the remaining unmasked segments.

    Parameters
    ----------
    vtrack_name : str
        Name of an existing virtual track.
    mask : pandas.DataFrame, str, list, or None
        The genomic mask to apply. Accepted forms:

        - A ``pandas.DataFrame`` with columns ``chrom``, ``start``,
          ``end`` -- intervals to mask.
        - A ``str`` naming an intervals set in the database.
        - A ``str`` naming an intervals-type (sparse) track.
        - A ``list`` or ``tuple`` of any combination of the above; all
          sources are unified into a single mask.
        - ``None`` -- clears any existing filter from the virtual track.
    filter : pandas.DataFrame, str, list, or None
        Backward-compatible alias for ``mask``.

    Notes
    -----
    A bin or interval of the source that overlaps an unmasked region counts
    once, even when the mask splits it. ``'size'`` and ``'exists'`` give 0
    when the unmasked regions hold no value. ``'first'`` and ``'last'`` (and
    their positions) are the first and last unmasked values; ``'sample'``
    (and ``'sample.pos.*'``) draws one of the unmasked values, each with the
    same chance; ``'nearest'`` is the average of the unmasked values, and
    without any, for a sparse, array or value-based source, the value whose
    unmasked part is nearest (never one inside the mask; the left one at the
    same distance). ``'weighted.sum'`` counts the unmasked bp of each value.

    For the sequence-scoring families a masked base is a hard boundary: a
    match cannot span one. Each unmasked part of the iterator interval is
    scored on its own and the parts are combined - log-sum-exp for ``'pwm'``
    and ``'potts'``, maximum for ``'pwm.max'`` and ``'potts.max'``, sum for
    ``'pwm.count'`` and ``'potts.count'``, and for ``'pwm.max.pos'`` and
    ``'potts.max.pos'`` the position from whichever part carried the highest
    score, reported in the iterator interval's own coordinates.

    A fully masked interval, and one whose every unmasked part is narrower
    than the model, give NaN for all four funcs - the counting ones included,
    since neither held an anchor to count. A part that holds anchors and
    merely has none of them scorable still counts 0.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If no virtual track with the given name exists.
    ValueError
        If a string filter source is not a recognized intervals set or
        intervals-type track, or if a DataFrame is missing required
        columns.

    See Also
    --------
    gvtrack_create : Create a virtual track (filter can also be set at
        creation time via the ``filter`` keyword argument).
    gvtrack_info : Inspect a virtual track's configuration including its
        filter.
    gvtrack_iterator : Set iterator shifts (applied before the filter).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()

    Attach a filter to exclude specific regions:

    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> mask = pm.gintervals(["1", "1"], [100, 500], [200, 600])
    >>> pm.gvtrack_filter("vt1", filter=mask)
    >>> pm.gvtrack_info("vt1")["filter"] is not None
    True

    Clear the filter:

    >>> pm.gvtrack_filter("vt1", filter=None)
    >>> pm.gvtrack_info("vt1")["filter"] is None
    True

    Use multiple filter sources (automatically unified):

    >>> mask1 = pm.gintervals(["1"], [100], [200])
    >>> mask2 = pm.gintervals(["1"], [500], [600])
    >>> pm.gvtrack_filter("vt1", filter=[mask1, mask2])
    """
    if "filter" in kwargs:
        if mask is not None:
            raise ValueError("Specify only one of 'mask' or 'filter'")
        mask = kwargs.pop("filter")
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {bad}")

    if vtrack_name not in _shared._VTRACKS:
        raise KeyError(f"Virtual track not found: {vtrack_name}")

    cfg = dict(_shared._VTRACKS[vtrack_name])
    if mask is None:
        cfg["filter"] = None
        cfg["filter_key"] = None
        cfg["filter_stats"] = None
        _shared._VTRACKS[vtrack_name] = cfg
        return

    filter_df = _resolve_filter_sources(mask)
    cfg["filter"] = filter_df if len(filter_df) > 0 else None
    cfg["filter_key"] = _filter_key(filter_df)
    cfg["filter_stats"] = _filter_stats(filter_df)
    _shared._VTRACKS[vtrack_name] = cfg
    return


def gvtrack_rm(vtrack_name: str) -> None:
    """
    Remove a virtual track.

    Deletes a single virtual track from the current session. If the named
    virtual track does not exist, the call is silently ignored (no error is
    raised).

    Parameters
    ----------
    vtrack_name : str
        Name of the virtual track to remove.

    Returns
    -------
    None

    See Also
    --------
    gvtrack_create : Create a new virtual track.
    gvtrack_clear : Remove all virtual tracks at once.
    gvtrack_ls : List all virtual tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_create("vt1", "dense_track", func="max")
    >>> pm.gvtrack_create("vt2", "dense_track", func="min")
    >>> pm.gvtrack_ls()
    ['vt1', 'vt2']
    >>> pm.gvtrack_rm("vt1")
    >>> pm.gvtrack_ls()
    ['vt2']

    Removing a non-existent track is a no-op:

    >>> pm.gvtrack_rm("does_not_exist")
    """
    if vtrack_name in _shared._VTRACKS:
        del _shared._VTRACKS[vtrack_name]


def gvtrack_clear() -> None:
    """
    Remove all virtual tracks.

    Clears the entire virtual track registry for the current session.
    After this call, ``gvtrack_ls()`` returns an empty list. This is
    useful for resetting state between analyses or in test fixtures.

    Returns
    -------
    None

    See Also
    --------
    gvtrack_rm : Remove a single virtual track by name.
    gvtrack_ls : List all virtual tracks.
    gvtrack_create : Create a new virtual track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_create("vt1", "dense_track", func="avg")
    >>> pm.gvtrack_create("vt2", "dense_track", func="max")
    >>> len(pm.gvtrack_ls())
    2
    >>> pm.gvtrack_clear()
    >>> pm.gvtrack_ls()
    []
    """
    _shared._VTRACKS.clear()


def gvtrack_array_slice(
    vtrack: str,
    slice: str | int | list[str] | list[int] | None = None,
    func: str = "avg",
    params=None,
) -> None:
    """Configure an existing virtual track as an array-slice aggregator.

    Mirrors R ``gvtrack.array.slice``. The vtrack must already exist (created
    via ``gvtrack_create(name, src=<array_track>)``). This function sets the
    column selection and the function that reduces the selected columns of
    each array interval to one value; the vtrack keeps its own function,
    shifts and filter, which then apply to these values as to a sparse
    track's (``gvtrack_info(vtrack)["slice"]`` holds the selection).
    The vtrack can then be referenced in ``gextract``, ``gsummary``, etc.
    and returns one value per iterator interval.

    Parameters
    ----------
    vtrack : str
        Name of an existing virtual track. The vtrack must have been created
        with an array track as its source. Raises ``ValueError`` if the vtrack
        does not exist or its source is not an array track.
    slice : str, int, or list of str or int, optional
        Column subset to use. Strings are matched against track colnames;
        integers are 0-based column indices. ``None`` uses all columns.
    func : str, default ``'avg'``
        Function applied to the non-NaN values of the selected columns of
        each array interval.  Supported values: ``'avg'``, ``'min'``,
        ``'max'``, ``'sum'``, ``'stddev'``, ``'quantile'``.
    params : float, optional
        The percentile in [0, 1] for ``func='quantile'``; must be ``None``
        otherwise.

    See Also
    --------
    gvtrack_create : Create a virtual track from any source.
    gvtrack_rm : Remove a virtual track.
    gtrack_array_extract : Extract per-position values from an array track.
    gtrack_array_get_colnames : List column names of an array track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gvtrack_create("v_avg", src="array_track")
    >>> pm.gvtrack_array_slice("v_avg", func="avg")
    >>> pm.gvtrack_create("v_col0", src="array_track")
    >>> pm.gvtrack_array_slice("v_col0", slice=["col0"], func="avg")
    >>> pm.gvtrack_ls()
    ['v_avg', 'v_col0']
    >>> pm.gvtrack_clear()
    """
    _shared._checkroot()

    # Vtrack must already exist.
    if vtrack not in _shared._VTRACKS:
        raise ValueError(f"gvtrack_array_slice: no such vtrack '{vtrack}'")

    existing = _shared._VTRACKS[vtrack]
    src = existing.get("src")

    # src must be a string (track name) pointing to an array track.
    if not isinstance(src, str):
        raise ValueError(
            f"gvtrack_array_slice: vtrack '{vtrack}' is not an array track vtrack "
            f"(src is not a track name)"
        )

    from .tracks import gtrack_exists, gtrack_info

    if not gtrack_exists(src):
        raise ValueError(
            f"gvtrack_array_slice: vtrack '{vtrack}' source track '{src}' does not exist"
        )
    info = gtrack_info(src)
    if info.get("type") != "array":
        raise ValueError(
            f"gvtrack_array_slice: vtrack '{vtrack}' source track '{src}' is not an array track "
            f"(type={info.get('type')!r}). "
            f"Use gvtrack_create for dense/sparse tracks."
        )

    from ._array_track import ARRAY_REDUCE_FUNCS

    # "quantile" is supported via the C++ scanner (StreamPercentiler), in
    # addition to the reducers the pure-Python path implements.
    _ALLOWED_SLICE_FUNCS = ARRAY_REDUCE_FUNCS | {"quantile"}
    func_lc = func.lower()
    if func_lc not in _ALLOWED_SLICE_FUNCS:
        raise ValueError(f"Virtual track {vtrack}: invalid function {func_lc} used with a track")

    # quantile requires a percentile in params; other reducers take no params.
    if func_lc == "quantile":
        pval = params[0] if isinstance(params, (list, tuple)) else params
        if pval is None:
            raise ValueError(
                "gvtrack_array_slice: func='quantile' requires params=<percentile in [0,1]>"
            )
        pval = float(pval)
        if not (0.0 <= pval <= 1.0):
            raise ValueError(
                f"gvtrack_array_slice: quantile percentile must be in [0, 1], got {pval}"
            )
        params = pval
    elif params is not None:
        raise ValueError(
            f"gvtrack_array_slice: params is only used with func='quantile', got func={func!r}"
        )

    # Resolve slice to a list of 0-based int indices (or None = all columns)
    if slice is not None:
        from pathlib import Path

        from ._array_track import read_colnames

        track_path = Path(_pymisha.pm_track_path(src))
        colnames = read_colnames(track_path)
        slice_list: list[Any] = [slice] if isinstance(slice, (str, int, np.integer)) else list(slice)
        if all(isinstance(s, str) for s in slice_list):
            str_list: list[str] = [str(s) for s in slice_list]
            cn_idx = {name: i for i, name in enumerate(colnames)}
            bad_names = [s for s in str_list if s not in cn_idx]
            if bad_names:
                raise ValueError(
                    f"gvtrack_array_slice: column(s) not found in '{src}': {bad_names!r}"
                )
            slice_cols: list[int] | None = [cn_idx[s] for s in str_list]
        elif all(isinstance(s, (int, _numpy.integer)) for s in slice_list):
            int_list: list[int] = [int(s) for s in slice_list]
            ncols = len(colnames)
            bad_idx = [s for s in int_list if s < 0 or s >= ncols]
            if bad_idx:
                raise ValueError(
                    f"gvtrack_array_slice: column indices out of range [0, {ncols}): {bad_idx!r}"
                )
            slice_cols = int_list
        else:
            raise ValueError(
                "gvtrack_array_slice: slice must be a list of strings (column names) "
                "or ints (0-based column indices)"
            )
    else:
        slice_cols = None

    # The vtrack keeps its function, shifts and filter, as R: its function reduces the sliced value
    # of each array interval the (shifted) iterator interval holds. slice is R's var$slice, with
    # 0-based column indices (None = all columns).
    cfg = dict(existing)
    cfg["kind"] = "array_slice"
    cfg["slice"] = {"slice": slice_cols, "func": func_lc, "params": params}
    _shared._VTRACKS[vtrack] = cfg
