"""Liftover chain loading and interval/track coordinate conversion.

Implements gintervals_load_chain, gintervals_as_chain, gintervals_liftover,
and gtrack_liftover with parity to R misha.

UCSC terminology note: In UCSC chain format, 't' fields (tName, tStart, tEnd)
are "target/reference" and 'q' fields are "query". Misha reverses this:
UCSC target = misha source (chromsrc), UCSC query = misha target (chrom).

Strand note: a chain DataFrame (`strand`, `strandsrc`) uses R's {1,-1} for
'+'/'-'. Internally the chain code uses {0,1} (0='+', 1='-'): _chain_to_r /
_chain_from_r convert at the public boundary.
"""

from __future__ import annotations

import heapq
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import _shared
from ._crc64 import (
    crc64_finalize as _crc64_finalize,
)
from ._crc64 import (
    crc64_incremental as _crc64_incremental,
)
from ._crc64 import (
    crc64_init as _crc64_init,
)
from ._shared import _checkroot, _pm_dbreload, _pymisha, _sort_by_chrom, _with_canonical_chroms
from .intervals import gintervals_all

# ---------------------------------------------------------------------------
# Overlap policy constants
# ---------------------------------------------------------------------------

_SRC_POLICIES = {"error", "keep", "discard"}
_CLUSTER_TGT_POLICIES = (
    "best_source_cluster", "best_cluster_union",
    "best_cluster_sum", "best_cluster_max",
)
_TGT_POLICIES = {
    "error", "auto", "auto_first", "auto_longer", "auto_score",
    "discard", "keep", "agg", *_CLUSTER_TGT_POLICIES,
}

_EMPTY_CHAIN_COLS = [
    "chrom", "start", "end", "strand",
    "chromsrc", "startsrc", "endsrc", "strandsrc",
    "chain_id", "score",
]


def _empty_chain_df() -> pd.DataFrame:
    return pd.DataFrame({
        c: pd.Series(
            dtype="object" if c in ("chrom", "chromsrc") else "int64" if c in (
                "start", "end", "startsrc", "endsrc", "strand", "strandsrc", "chain_id"
            ) else "float64"
        ) for c in _EMPTY_CHAIN_COLS
    })


def _normalize_chrom(name: str) -> str | None:
    """Normalize a chromosome name using the C++ chromkey."""
    try:
        normalized = _pymisha.pm_normalize_chroms([name])
    except Exception as exc:
        msg = str(exc)
        if "does not exist" in msg or "Unknown chromosome" in msg:
            return None
        raise
    if not normalized:
        return None
    return str(normalized[0])


def _get_db_chrom_sizes() -> dict[str, int]:
    """Return {chrom_name: size} for the current database."""
    all_iv = gintervals_all()
    return dict(
        zip(
            all_iv["chrom"].astype(str).tolist(),
            all_iv["end"].astype(int).tolist(), strict=False,
        )
    )


# ===================================================================
# Chain file parser
# ===================================================================

def _parse_chain_file(
    path: str,
    db_chrom_sizes: dict[str, int],
    min_score: float | None = None,
    _force_pure_python: bool = False,
) -> dict[str, Any] | None:
    """Parse a UCSC chain file. Dispatches to C++ by default.

    Returns a dict whose values are numpy arrays on the C++ path and Python
    lists on the pure-Python path. Downstream callers consume both via
    ``pd.DataFrame(...)``, which infers the same dtypes either way.

    The `_force_pure_python` kwarg is used by the G1.P2 cross-validation tests
    in tests/test_chain_parser_cpp.py to compare the C++ and Python paths.
    Setting the env var PYMISHA_FORCE_PY_CHAIN_PARSER=1 has the same effect
    globally.
    """
    chain_path = Path(path)
    if not chain_path.exists():
        raise FileNotFoundError(f"Chain file does not exist: {path}")
    if not chain_path.is_file():
        raise ValueError(f"Chain path is not a regular file: {path}")

    use_py = _force_pure_python or os.environ.get(
        "PYMISHA_FORCE_PY_CHAIN_PARSER", ""
    ) == "1"
    if use_py:
        return _parse_chain_file_python(path, db_chrom_sizes, min_score=min_score)

    ms = float("nan") if min_score is None else float(min_score)
    return _pymisha.pm_parse_chain_file(str(chain_path), ms)  # type: ignore[no-any-return]


def _parse_chain_file_python(
    path: str,
    db_chrom_sizes: dict[str, int],
    min_score: float | None = None,
) -> dict[str, list[Any]] | None:
    """Parse a UCSC chain file and return list of chain block dicts.

    Each dict has: chrom, start, end, strand, chromsrc, startsrc, endsrc,
    strandsrc, chain_id, score.

    Blocks targeting chromosomes not in db_chrom_sizes are silently skipped.
    """
    chain_path = Path(path)
    if not chain_path.exists():
        raise FileNotFoundError(f"Chain file does not exist: {path}")
    if not chain_path.is_file():
        raise ValueError(f"Chain path is not a regular file: {path}")

    # Column-wise accumulators for parsed chain blocks
    b_chrom = []
    b_start = []
    b_end = []
    b_strand = []
    b_chromsrc = []
    b_startsrc = []
    b_endsrc = []
    b_strandsrc = []
    b_chain_id = []
    b_score = []
    src_chrom_sizes: dict[str, int] = {}  # track source chrom sizes for consistency validation

    with open(chain_path, encoding="utf-8") as f:
        lineno = 0
        # State for current chain
        in_chain = False
        skip_chain = False
        src_chrom = None
        src_size = 0
        src_strand = 0
        src_start = 0
        src_end = 0
        tgt_chrom = None
        tgt_size = 0
        tgt_strand = 0
        tgt_start = 0
        chain_id = 0
        chain_score = 0.0
        cur_src_pos = 0
        cur_tgt_pos = 0

        for raw_line in f:
            lineno += 1
            line = raw_line.strip()

            # Skip empty lines (chain separator)
            if not line:
                in_chain = False
                continue

            # Skip comments
            if line.startswith("#"):
                continue

            parts = line.split()

            # Chain header line
            if parts[0] == "chain":
                if len(parts) != 13:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: invalid file format"
                    )

                chain_score = float(parts[1])

                # min_score filtering
                if min_score is not None and chain_score < min_score:
                    skip_chain = True
                    in_chain = True
                    continue

                skip_chain = False

                # Source (UCSC target/reference) fields
                src_chrom = parts[2]
                src_size = int(parts[3])
                if src_size <= 0:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: invalid format of reference chrom size"
                    )

                # Validate source chrom size consistency
                if src_chrom in src_chrom_sizes:
                    if src_chrom_sizes[src_chrom] != src_size:
                        raise ValueError(
                            f"Chain file {path}, line {lineno}: reference chrom size "
                            f"({src_size}) differs from previous value ({src_chrom_sizes[src_chrom]})"
                        )
                else:
                    src_chrom_sizes[src_chrom] = src_size

                src_strand_str = parts[4]
                if src_strand_str == "+":
                    src_strand = 0
                elif src_strand_str == "-":
                    src_strand = 1
                else:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: invalid format of reference strand"
                    )

                src_start = int(parts[5])
                src_end = int(parts[6])
                if src_start < 0:
                    raise ValueError(f"Chain file {path}, line {lineno}: invalid value of reference start coordinate")
                if src_start >= src_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: reference start coordinate exceeds chromosome size")
                if src_end <= src_start:
                    raise ValueError(f"Chain file {path}, line {lineno}: reference end coordinate is less or "
                                     "equal than the start coordinate")
                if src_end > src_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: reference end coordinate exceeds chromosome size")

                # Target (UCSC query) fields — normalize chrom name
                tgt_chrom_raw = parts[7]
                tgt_chrom = _normalize_chrom(tgt_chrom_raw)
                tgt_size = int(parts[8])

                # Check if target chrom exists in DB
                if tgt_chrom is None or tgt_chrom not in db_chrom_sizes:
                    skip_chain = True
                    in_chain = True
                    cur_src_pos = src_start
                    cur_tgt_pos = tgt_start
                    continue

                # Validate target chrom size against DB
                db_size = db_chrom_sizes[tgt_chrom]
                if tgt_size != db_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: query chrom size "
                        f"({tgt_size}) differs from what have been defined in the database ({db_size})"
                    )

                tgt_strand_str = parts[9]
                if tgt_strand_str == "+":
                    tgt_strand = 0
                elif tgt_strand_str == "-":
                    tgt_strand = 1
                else:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: invalid format of query strand"
                    )

                tgt_start_raw = int(parts[10])
                tgt_end_raw = int(parts[11])
                if tgt_start_raw < 0:
                    raise ValueError(f"Chain file {path}, line {lineno}: invalid value of query start coordinate")
                if tgt_start_raw >= tgt_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: query start coordinate exceeds chromosome size")
                # R's wording, "reference" included, for the query end too
                if tgt_end_raw <= tgt_start_raw:
                    raise ValueError(f"Chain file {path}, line {lineno}: reference end coordinate is less or "
                                     "equal than the start coordinate")
                if tgt_end_raw > tgt_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: reference end coordinate exceeds chromosome size")
                tgt_start = tgt_start_raw

                chain_id = int(parts[12])
                cur_src_pos = src_start
                cur_tgt_pos = tgt_start
                in_chain = True
                continue

            # Alignment block line (1 or 3 fields)
            if not in_chain:
                raise ValueError(
                    f"Chain file {path}, line {lineno}: invalid file format"
                )

            if len(parts) not in (1, 3):
                raise ValueError(
                    f"Chain file {path}, line {lineno}: invalid file format"
                )

            if skip_chain:
                # Chain explicitly skipped (e.g. low score): ignore blocks.
                continue

            size = int(parts[0])
            if size <= 0:
                raise ValueError(
                    f"Chain file {path}, line {lineno}: invalid size"
                )

            # Compute source coordinates (handle negative strand)
            if src_strand == 0:
                block_src_start = cur_src_pos
                block_src_end = cur_src_pos + size
            else:
                block_src_start = src_size - cur_src_pos - size
                block_src_end = src_size - cur_src_pos

            # Compute target coordinates (handle negative strand)
            if tgt_strand == 0:
                block_tgt_start = cur_tgt_pos
                block_tgt_end = cur_tgt_pos + size
            else:
                block_tgt_start = tgt_size - cur_tgt_pos - size
                block_tgt_end = tgt_size - cur_tgt_pos

            b_chrom.append(tgt_chrom)
            b_start.append(block_tgt_start)
            b_end.append(block_tgt_end)
            b_strand.append(tgt_strand)
            b_chromsrc.append(src_chrom)
            b_startsrc.append(block_src_start)
            b_endsrc.append(block_src_end)
            b_strandsrc.append(src_strand)
            b_chain_id.append(chain_id)
            b_score.append(chain_score)

            # Advance positions
            if len(parts) == 3:
                dt = int(parts[1])
                dq = int(parts[2])
                if dt < 0 or dq < 0:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: invalid block gaps"
                    )
                cur_src_pos += size + dt
                cur_tgt_pos += size + dq
            else:
                cur_src_pos += size
                cur_tgt_pos += size

    if not b_chrom:
        return None
    return {
        "chrom": b_chrom,
        "start": b_start,
        "end": b_end,
        "strand": b_strand,
        "chromsrc": b_chromsrc,
        "startsrc": b_startsrc,
        "endsrc": b_endsrc,
        "strandsrc": b_strandsrc,
        "chain_id": b_chain_id,
        "score": b_score,
    }


# ===================================================================
# Overlap handling
# ===================================================================

def _handle_src_overlaps_python(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Handle source-side overlaps according to policy."""
    if df.empty or policy == "keep":
        return df

    # Sort by source coordinates
    df = df.sort_values(["chromsrc", "startsrc", "endsrc"]).reset_index(drop=True)

    if policy == "error":
        n = len(df)
        if n > 1:
            chroms = df["chromsrc"].to_numpy()
            starts = df["startsrc"].to_numpy(dtype=np.int64, copy=False)
            ends = df["endsrc"].to_numpy(dtype=np.int64, copy=False)
            same_chrom = chroms[1:] == chroms[:-1]
            overlaps = same_chrom & (starts[1:] < ends[:-1])
            idx = np.flatnonzero(overlaps)
            if idx.size > 0:
                i = idx[0] + 1
                raise ValueError(
                    f"Source overlap detected on {chroms[i]}: "
                    f"[{starts[i-1]}, {ends[i-1]}) overlaps "
                    f"[{starts[i]}, {ends[i]})"
                )
        return df

    if policy == "discard":
        # R parity (rdbinterval.cpp:820-841): mark each consecutive pair that
        # overlaps on the same chromsrc. Strictly weaker than whole-cluster
        # discard - a row nested inside a larger row with a gap to its prev
        # neighbor stays kept. Replaces the prior _discard_overlapping_intervals
        # whole-cluster call which over-discarded on nested-with-gap inputs.
        n = len(df)
        if n < 2:
            return df
        chroms = df["chromsrc"].to_numpy()
        starts = df["startsrc"].to_numpy(dtype=np.int64, copy=False)
        ends = df["endsrc"].to_numpy(dtype=np.int64, copy=False)
        pair_overlap = (chroms[1:] == chroms[:-1]) & (ends[:-1] > starts[1:])
        discard_mask = np.zeros(n, dtype=bool)
        if pair_overlap.any():
            idx = np.flatnonzero(pair_overlap)
            discard_mask[idx] = True
            discard_mask[idx + 1] = True
        if discard_mask.any():
            return df.loc[~discard_mask].reset_index(drop=True)
        return df

    raise ValueError(f"Unknown src_overlap_policy: {policy}")


def _handle_tgt_overlaps_python(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Handle target-side overlaps according to policy."""
    if df.empty or policy == "keep":
        return df

    # Sort by target coordinates
    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

    if policy == "error":
        n = len(df)
        if n > 1:
            chroms = df["chrom"].to_numpy()
            starts_arr = df["start"].to_numpy(dtype=np.int64, copy=False)
            ends_arr = df["end"].to_numpy(dtype=np.int64, copy=False)
            same_chrom = chroms[1:] == chroms[:-1]
            overlaps = same_chrom & (starts_arr[1:] < ends_arr[:-1])
            idx = np.flatnonzero(overlaps)
            if idx.size > 0:
                i = idx[0] + 1
                raise ValueError(
                    f"Target overlap detected on {chroms[i]}: "
                    f"[{starts_arr[i-1]}, {ends_arr[i-1]}) overlaps "
                    f"[{starts_arr[i]}, {ends_arr[i]})"
                )
        return df

    if policy == "discard":
        return _discard_overlapping_intervals(df, "chrom", "start", "end")

    if policy in ("auto_score", "auto_first", "auto_longer"):
        return _handle_tgt_overlaps_auto(df, policy)

    if policy == "agg":
        return _handle_tgt_overlaps_agg(df)

    if policy in ("best_source_cluster", "best_cluster_union",
                   "best_cluster_sum", "best_cluster_max"):
        # These are resolved during liftover, not during chain loading.
        # During loading, we keep all chains (like "keep").
        return df

    raise ValueError(f"Unknown tgt_overlap_policy: {policy}")


def _resolve_chain_overlaps(
    chain_dict: dict[str, Any],
    src_overlap_policy: str,
    tgt_overlap_policy: str,
    _force_pure_python: bool = False,
) -> dict[str, Any]:
    """Apply src + tgt overlap policies on a chain DataFrame dict.

    Dispatches to C++ ``_pymisha.pm_chain_intervals_resolve`` by default.
    Setting ``_force_pure_python=True`` or the env var
    ``PYMISHA_FORCE_PY_CHAIN_INTERVALS_RESOLVE=1`` falls back to the pure-Python
    pair ``_handle_src_overlaps_python`` + ``_handle_tgt_overlaps_python``.

    Cluster policies (``best_source_cluster`` etc.) are NOT resolved here; they
    are normalized to ``"keep"`` for the load-time pass and resolved later by
    ``_resolve_cluster_policy`` after interval mapping. This matches the
    existing ``effective_tgt_policy`` semantics in ``gintervals_load_chain``.
    """
    if src_overlap_policy not in _SRC_POLICIES:
        raise ValueError(
            f"src_overlap_policy must be one of {sorted(_SRC_POLICIES)}, "
            f"got '{src_overlap_policy}'"
        )
    if tgt_overlap_policy not in _TGT_POLICIES:
        raise ValueError(
            f"tgt_overlap_policy must be one of {sorted(_TGT_POLICIES)}, "
            f"got '{tgt_overlap_policy}'"
        )

    use_py = _force_pure_python or os.environ.get(
        "PYMISHA_FORCE_PY_CHAIN_INTERVALS_RESOLVE", ""
    ).lower() in ("1", "true", "yes")

    effective_tgt = tgt_overlap_policy
    if effective_tgt in (
        "best_source_cluster", "best_cluster_union",
        "best_cluster_sum", "best_cluster_max",
    ):
        effective_tgt = "keep"
    if effective_tgt == "auto":
        effective_tgt = "auto_score"

    if use_py:
        df = pd.DataFrame(chain_dict)[_EMPTY_CHAIN_COLS]
        df = _handle_src_overlaps_python(df, src_overlap_policy)
        df = _handle_tgt_overlaps_python(df, effective_tgt)
        return {c: df[c].to_numpy() for c in _EMPTY_CHAIN_COLS}

    return _pymisha.pm_chain_intervals_resolve(  # type: ignore[no-any-return]
        chain_dict, src_overlap_policy, effective_tgt
    )


# Deprecation aliases. test_liftover.py imports these by name. They now
# delegate to the pure-Python implementations - observable behavior unchanged.
# The dispatcher routes through _resolve_chain_overlaps for the production
# path used by gintervals_load_chain.
_handle_src_overlaps = _handle_src_overlaps_python
_handle_tgt_overlaps = _handle_tgt_overlaps_python


def _discard_overlapping_intervals(
    df: pd.DataFrame,
    chrom_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    """Drop all intervals that overlap any other interval on the same chrom."""
    n = len(df)
    if n < 2:
        return df

    chroms = df[chrom_col].to_numpy()
    starts = df[start_col].to_numpy(dtype=np.int64, copy=False)
    ends = df[end_col].to_numpy(dtype=np.int64, copy=False)

    # Find where chroms change — these are group boundaries
    chrom_change = np.empty(n, dtype=bool)
    chrom_change[0] = True
    chrom_change[1:] = chroms[1:] != chroms[:-1]

    # Compute running max of ends within each chrom group, resetting at
    # chrom boundaries. This is a prefix-max scan with resets — the data
    # dependency prevents full vectorization, but iterating over numpy
    # scalars (not pandas .loc) is fast.
    max_end = ends.copy()
    for i in range(1, n):
        if not chrom_change[i] and max_end[i - 1] > max_end[i]:
            max_end[i] = max_end[i - 1]

    # An interval at position i overlaps its predecessor's cluster if:
    # same chrom AND start[i] < max_end up to i-1
    # We detect overlap pairs: start[i] < max_end[i-1] and same chrom
    overlaps_prev = np.zeros(n, dtype=bool)
    overlaps_prev[1:] = (~chrom_change[1:]) & (starts[1:] < max_end[:-1])

    # Now we need to identify clusters of overlapping intervals and mark
    # entire clusters that contain at least one overlap.
    # A cluster boundary occurs where overlaps_prev is False.
    # Assign cluster IDs
    cluster_ids = np.cumsum(~overlaps_prev)

    # Find clusters that have more than one member OR contain an overlap
    # A cluster has an overlap if any overlaps_prev within it is True
    # Since cluster boundaries are at ~overlaps_prev, a cluster with overlap
    # means it has size > 1 (any overlaps_prev=True entry in it)
    # Actually, we need: clusters where at least one pair overlaps.
    # overlaps_prev[i] = True means interval i overlaps with something before it
    # in the same cluster.

    # Find which cluster IDs have any overlap
    # Use np.bincount to find if any overlaps_prev is True per cluster
    max_cluster = cluster_ids[-1]
    has_overlap = np.zeros(max_cluster + 1, dtype=bool)
    overlap_indices = np.flatnonzero(overlaps_prev)
    if overlap_indices.size > 0:
        has_overlap[cluster_ids[overlap_indices]] = True

    discard_mask = has_overlap[cluster_ids]

    if discard_mask.any():
        return df.loc[~discard_mask].reset_index(drop=True)
    return df


def _handle_tgt_overlaps_auto(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Segment overlapping target intervals and select winner per segment.

    auto_score: highest score wins (tiebreak: longer span, lower chain_id)
    auto_first: lowest chain_id wins
    auto_longer: longest span wins (tiebreak: higher score, lower chain_id)

    Uses vectorized numpy operations for breakpoint segmentation, winner
    selection, and adjacent merging.
    """
    if df.empty:
        return df

    result_parts = []

    # Process per chromosome
    for chrom, group in df.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"]).reset_index(drop=True)
        ng = len(group)
        if ng == 0:
            continue

        starts = group["start"].to_numpy(dtype=np.int64, copy=False)
        ends = group["end"].to_numpy(dtype=np.int64, copy=False)
        strands = group["strand"].to_numpy(dtype=np.int64, copy=False)
        src_starts = group["startsrc"].to_numpy(dtype=np.int64, copy=False)
        src_ends = group["endsrc"].to_numpy(dtype=np.int64, copy=False)
        chain_ids = group["chain_id"].to_numpy(dtype=np.int64, copy=False)
        scores = group["score"].to_numpy(dtype=np.float64, copy=False)
        chromsrc_vals = group["chromsrc"].to_numpy()
        strandsrc_vals = group["strandsrc"].to_numpy(dtype=np.int64, copy=False)
        spans = ends - starts

        points = np.unique(np.concatenate((starts, ends)))
        if points.size < 2:
            result_parts.append(group)
            continue

        n_segs = len(points) - 1
        seg_starts = points[:-1]  # (n_segs,)
        seg_ends = points[1:]     # (n_segs,)

        # For each segment, find which intervals cover it.
        # Interval j covers segment i iff starts[j] <= seg_starts[i]
        # and ends[j] >= seg_ends[i].
        # Use broadcasting: (n_segs, ng) boolean matrix.
        # For large groups this is memory-intensive; fall back to sweep-line
        # for very large groups.
        if n_segs * ng <= 500_000:
            # Vectorized: build coverage matrix
            # covers[i, j] = (starts[j] <= seg_starts[i]) & (ends[j] >= seg_ends[i])
            covers = (
                (starts[np.newaxis, :] <= seg_starts[:, np.newaxis])
                & (ends[np.newaxis, :] >= seg_ends[:, np.newaxis])
            )  # (n_segs, ng)

            # Build priority keys per interval for argsort
            idx_arr = np.arange(ng, dtype=np.int64)
            if policy == "auto_score":
                # prio: (-score, -span, chain_id, idx) — lower is better
                prio_keys = np.column_stack((
                    -scores, -spans.astype(np.float64), chain_ids.astype(np.float64),
                    idx_arr.astype(np.float64),
                ))
            elif policy == "auto_first":
                prio_keys = np.column_stack((
                    chain_ids.astype(np.float64), idx_arr.astype(np.float64),
                ))
            else:  # auto_longer
                prio_keys = np.column_stack((
                    -spans.astype(np.float64), -scores, chain_ids.astype(np.float64),
                    idx_arr.astype(np.float64),
                ))

            # For each segment, find the winner among covering intervals.
            # Sort interval indices by priority; for each segment, the winner
            # is the first interval in sorted order that covers the segment.
            # Pre-sort intervals by priority.
            if prio_keys.shape[1] == 2:
                sort_order = np.lexsort((prio_keys[:, 1], prio_keys[:, 0]))
            else:
                sort_order = np.lexsort(tuple(
                    prio_keys[:, k] for k in range(prio_keys.shape[1] - 1, -1, -1)
                ))

            # Reorder covers columns by priority
            covers_sorted = covers[:, sort_order]  # (n_segs, ng)

            # Winner for each segment = first True in sorted covers
            # argmax on axis=1 gives first True (or 0 if no True)
            first_true = covers_sorted.argmax(axis=1)  # (n_segs,)
            has_any = covers_sorted.any(axis=1)        # (n_segs,)

            # Map back to original indices
            winner_orig = sort_order[first_true]  # (n_segs,)

            # Filter segments with no covering interval
            valid = has_any
            if not valid.any():
                continue

            seg_starts_v = seg_starts[valid]
            seg_ends_v = seg_ends[valid]
            w = winner_orig[valid]
        else:
            # Fall back to sweep-line for very large groups to avoid OOM
            seg_starts_v, seg_ends_v, w = _sweep_line_winners(
                starts, ends, spans, chain_ids, scores, points, policy,
            )
            if len(w) == 0:
                continue

        # Compute source coordinates for each segment based on winner
        orig_tgt_starts = starts[w]
        orig_tgt_ends = ends[w]
        orig_src_starts = src_starts[w]
        orig_src_ends = src_ends[w]
        orig_tgt_lens = orig_tgt_ends - orig_tgt_starts
        w_strands = strands[w]

        # Vectorized source coordinate mapping
        seg_src_starts = np.empty_like(seg_starts_v)
        seg_src_ends = np.empty_like(seg_ends_v)

        pos_strand = w_strands == 0
        nonzero_len = orig_tgt_lens > 0
        # Positive strand, nonzero length
        mask_pn = pos_strand & nonzero_len
        if mask_pn.any():
            seg_src_starts[mask_pn] = (
                orig_src_starts[mask_pn] + (seg_starts_v[mask_pn] - orig_tgt_starts[mask_pn])
            )
            seg_src_ends[mask_pn] = (
                orig_src_starts[mask_pn] + (seg_ends_v[mask_pn] - orig_tgt_starts[mask_pn])
            )
        # Negative strand, nonzero length
        mask_nn = (~pos_strand) & nonzero_len
        if mask_nn.any():
            seg_src_starts[mask_nn] = (
                orig_src_ends[mask_nn] - (seg_ends_v[mask_nn] - orig_tgt_starts[mask_nn])
            )
            seg_src_ends[mask_nn] = (
                orig_src_ends[mask_nn] - (seg_starts_v[mask_nn] - orig_tgt_starts[mask_nn])
            )
        # Zero-length target
        mask_z = ~nonzero_len
        if mask_z.any():
            seg_src_starts[mask_z] = orig_src_starts[mask_z]
            seg_src_ends[mask_z] = orig_src_ends[mask_z]

        seg_chain_ids = chain_ids[w]
        seg_scores = scores[w]
        seg_strands = w_strands
        seg_strandsrc = strandsrc_vals[w]
        seg_chromsrc = chromsrc_vals[w]

        # Vectorized adjacent merging: merge consecutive segments with same
        # chain_id (all segments are same chrom within this loop).
        ns = len(seg_starts_v)
        if ns == 0:
            continue

        # A new group starts where chain_id changes, tgt is not adjacent,
        # OR src is not adjacent (R-parity: rdbinterval.cpp:889-902 requires
        # `prev.end_src == slice.start_src` for the merge to fire). The src
        # check matters only for negative-strand chains, whose slices have
        # reversed src coords across consecutive tgt segments.
        new_group = np.ones(ns, dtype=bool)
        if ns > 1:
            new_group[1:] = (
                (seg_chain_ids[1:] != seg_chain_ids[:-1])
                | (seg_starts_v[1:] != seg_ends_v[:-1])
                | (seg_src_starts[1:] != seg_src_ends[:-1])
            )

        group_ids = np.cumsum(new_group) - 1
        n_groups = group_ids[-1] + 1

        # For each merge group: start = first seg_start, end = last seg_end,
        # startsrc = min(seg_src_starts), endsrc = max(seg_src_ends),
        # other columns from the first segment in the group.
        # Use np.minimum/maximum.reduceat for src bounds.
        group_starts_idx = np.flatnonzero(new_group)

        m_start = seg_starts_v[group_starts_idx]
        # For end: last element of each group = element before next group start
        group_ends_idx = np.empty(n_groups, dtype=np.intp)
        group_ends_idx[:-1] = group_starts_idx[1:] - 1
        group_ends_idx[-1] = ns - 1
        m_end = seg_ends_v[group_ends_idx]

        m_startsrc = np.minimum.reduceat(seg_src_starts, group_starts_idx)
        m_endsrc = np.maximum.reduceat(seg_src_ends, group_starts_idx)

        # Other columns: take from first element of each group
        fi = group_starts_idx
        m_strand = seg_strands[fi]
        m_chain_id = seg_chain_ids[fi]
        m_score = seg_scores[fi]
        m_strandsrc = seg_strandsrc[fi]
        m_chromsrc = seg_chromsrc[fi]

        part = pd.DataFrame({
            "chrom": np.full(n_groups, chrom, dtype=object),
            "start": m_start,
            "end": m_end,
            "strand": m_strand,
            "chromsrc": m_chromsrc,
            "startsrc": m_startsrc,
            "endsrc": m_endsrc,
            "strandsrc": m_strandsrc,
            "chain_id": m_chain_id,
            "score": m_score,
        })
        result_parts.append(part)

    if not result_parts:
        return _empty_chain_df()
    return pd.concat(result_parts, ignore_index=True)[_EMPTY_CHAIN_COLS]


def _sweep_line_winners(
    starts: np.ndarray,
    ends: np.ndarray,
    spans: np.ndarray,
    chain_ids: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep-line fallback for large chrom groups in _handle_tgt_overlaps_auto.

    Returns (seg_starts, seg_ends, winner_indices) as numpy arrays.
    """
    from collections import defaultdict as _defaultdict

    n = len(starts)
    starts_at = _defaultdict(list)
    ends_at = _defaultdict(list)
    for idx in range(n):
        starts_at[int(starts[idx])].append(idx)
        ends_at[int(ends[idx])].append(idx)

    if policy == "auto_score":
        def prio(idx: int) -> Any:
            return (-float(scores[idx]), -int(spans[idx]), int(chain_ids[idx]), int(idx))
    elif policy == "auto_first":
        def prio(idx: int) -> Any:
            return (int(chain_ids[idx]), int(idx))
    else:  # auto_longer
        def prio(idx: int) -> Any:
            return (-int(spans[idx]), -float(scores[idx]), int(chain_ids[idx]), int(idx))

    active: set[int] = set()
    heap: list[tuple[Any, int]] = []
    r_seg_starts: list[int] = []
    r_seg_ends: list[int] = []
    r_winners: list[int] = []

    for i in range(len(points) - 1):
        coord = int(points[i])
        next_coord = int(points[i + 1])

        for idx in ends_at.get(coord, ()):
            active.discard(idx)
        for idx in starts_at.get(coord, ()):
            active.add(idx)
            heapq.heappush(heap, (prio(idx), idx))

        if next_coord <= coord or not active:
            continue

        while heap and heap[0][1] not in active:
            heapq.heappop(heap)
        if not heap:
            continue

        r_seg_starts.append(coord)
        r_seg_ends.append(next_coord)
        r_winners.append(heap[0][1])

    return (
        np.array(r_seg_starts, dtype=np.int64),
        np.array(r_seg_ends, dtype=np.int64),
        np.array(r_winners, dtype=np.intp),
    )


def _handle_tgt_overlaps_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Segment overlapping target regions, keeping all chains per segment.

    Uses vectorized numpy operations for breakpoint segmentation and
    interval-segment overlap computation.
    """
    if df.empty:
        return df

    result_parts = []

    for chrom, group in df.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"]).reset_index(drop=True)
        ng = len(group)
        if ng == 0:
            continue

        iv_starts = group["start"].to_numpy(dtype=np.int64, copy=False)
        iv_ends = group["end"].to_numpy(dtype=np.int64, copy=False)
        iv_strands = group["strand"].to_numpy(dtype=np.int64, copy=False)
        iv_src_starts = group["startsrc"].to_numpy(dtype=np.int64, copy=False)
        iv_src_ends = group["endsrc"].to_numpy(dtype=np.int64, copy=False)
        iv_strandsrc = group["strandsrc"].to_numpy(dtype=np.int64, copy=False)
        iv_chain_ids = group["chain_id"].to_numpy(dtype=np.int64, copy=False)
        iv_scores = group["score"].to_numpy(dtype=np.float64, copy=False)
        iv_chromsrc = group["chromsrc"].to_numpy()

        points = np.unique(np.concatenate((iv_starts, iv_ends)))
        if points.size < 2:
            result_parts.append(group[_EMPTY_CHAIN_COLS])
            continue

        n_segs = len(points) - 1
        seg_starts = points[:-1]
        seg_ends = points[1:]

        # Coverage: interval j covers segment i iff
        # iv_starts[j] < seg_ends[i] AND iv_ends[j] > seg_starts[i]
        if n_segs * ng <= 500_000:
            # Vectorized coverage matrix
            covers = (
                (iv_starts[np.newaxis, :] < seg_ends[:, np.newaxis])
                & (iv_ends[np.newaxis, :] > seg_starts[:, np.newaxis])
            )  # (n_segs, ng)

            seg_idx, iv_idx = np.nonzero(covers)
        else:
            # Fall back to per-segment check for very large groups
            seg_idx_list = []
            iv_idx_list = []
            for i in range(n_segs):
                mask = (iv_starts < seg_ends[i]) & (iv_ends > seg_starts[i])
                js = np.flatnonzero(mask)
                seg_idx_list.append(np.full(len(js), i, dtype=np.intp))
                iv_idx_list.append(js)
            if seg_idx_list:
                seg_idx = np.concatenate(seg_idx_list)
                iv_idx = np.concatenate(iv_idx_list)
            else:
                continue

        if len(seg_idx) == 0:
            continue

        # Compute source coordinates vectorized
        r_seg_starts = seg_starts[seg_idx]
        r_seg_ends = seg_ends[seg_idx]
        r_iv_strands = iv_strands[iv_idx]
        r_orig_tgt_starts = iv_starts[iv_idx]
        r_orig_src_starts = iv_src_starts[iv_idx]
        r_orig_src_ends = iv_src_ends[iv_idx]

        r_src_starts = np.empty_like(r_seg_starts)
        r_src_ends = np.empty_like(r_seg_ends)

        pos_mask = r_iv_strands == 0
        neg_mask = ~pos_mask

        if pos_mask.any():
            r_src_starts[pos_mask] = (
                r_orig_src_starts[pos_mask]
                + (r_seg_starts[pos_mask] - r_orig_tgt_starts[pos_mask])
            )
            r_src_ends[pos_mask] = (
                r_orig_src_starts[pos_mask]
                + (r_seg_ends[pos_mask] - r_orig_tgt_starts[pos_mask])
            )
        if neg_mask.any():
            r_src_starts[neg_mask] = (
                r_orig_src_ends[neg_mask]
                - (r_seg_ends[neg_mask] - r_orig_tgt_starts[neg_mask])
            )
            r_src_ends[neg_mask] = (
                r_orig_src_ends[neg_mask]
                - (r_seg_starts[neg_mask] - r_orig_tgt_starts[neg_mask])
            )

        part = pd.DataFrame({
            "chrom": np.full(len(seg_idx), chrom, dtype=object),
            "start": r_seg_starts,
            "end": r_seg_ends,
            "strand": r_iv_strands,
            "chromsrc": iv_chromsrc[iv_idx],
            "startsrc": r_src_starts,
            "endsrc": r_src_ends,
            "strandsrc": iv_strandsrc[iv_idx],
            "chain_id": iv_chain_ids[iv_idx],
            "score": iv_scores[iv_idx],
        })
        result_parts.append(part)

    if not result_parts:
        return _empty_chain_df()
    return pd.concat(result_parts, ignore_index=True)[_EMPTY_CHAIN_COLS]


def _interval_union_length(starts: np.ndarray, ends: np.ndarray) -> float:
    """Return total union length of half-open intervals."""
    if len(starts) == 0:
        return 0.0

    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = ends[order]

    # Vectorized union: propagate max end forward, then sum non-overlapping
    # cluster breaks where start >= running max end.
    n = len(starts)
    if n == 1:
        return float(ends[0] - starts[0])

    # Compute running max of ends
    max_ends = np.maximum.accumulate(ends)

    # A new cluster starts where starts[i] >= max_ends[i-1]
    new_cluster = np.ones(n, dtype=bool)
    new_cluster[1:] = starts[1:] >= max_ends[:-1]

    # Each cluster: start = min(starts) = starts[first_in_cluster],
    # end = max(ends) in cluster
    cluster_starts_idx = np.flatnonzero(new_cluster)
    cluster_starts = starts[cluster_starts_idx]

    # For cluster ends: use maximum.reduceat
    cluster_ends = np.maximum.reduceat(ends, cluster_starts_idx)

    return float(np.sum(cluster_ends - cluster_starts))


def _resolve_cluster_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Apply best_cluster_* policy on mapped rows per intervalID."""
    if df.empty:
        return df

    if "__src_start" not in df.columns or "__src_end" not in df.columns:
        return df

    if policy == "best_source_cluster":
        policy = "best_cluster_union"

    if policy not in ("best_cluster_union", "best_cluster_sum", "best_cluster_max"):
        return df

    kept = []
    for _interval_id, group in df.groupby("intervalID", sort=False):
        if len(group) <= 1:
            kept.append(group)
            continue

        ordered = group.sort_values(
            ["__src_start", "__src_end", "chain_id", "start", "end"],
            kind="mergesort",
        ).reset_index()

        starts = ordered["__src_start"].to_numpy(dtype=np.int64, copy=False)
        ends = ordered["__src_end"].to_numpy(dtype=np.int64, copy=False)

        # Connected components combining (a) chain_id equality and (b) source
        # overlap. Mirrors R IntervalsLiftover.cpp:226-258.
        n_ord = len(ordered)
        if n_ord == 1:
            ordered["__cluster_id"] = np.zeros(1, dtype=np.int64)
        else:
            # Union-find structure
            parent = np.arange(n_ord, dtype=np.int64)

            def _find(x: int, _p: np.ndarray = parent) -> int:
                while _p[x] != x:
                    _p[x] = _p[_p[x]]
                    x = int(_p[x])
                return x

            def _union(a: int, b: int, _p: np.ndarray = parent) -> None:
                ra, rb = _find(a, _p), _find(b, _p)
                if ra != rb:
                    _p[ra] = rb

            # (a) Union by chain_id
            chain_id_arr = ordered["chain_id"].to_numpy(dtype=np.int64, copy=False)
            first_for_chain: dict[int, int] = {}
            for i_pos in range(n_ord):
                cid = int(chain_id_arr[i_pos])
                if cid in first_for_chain:
                    _union(i_pos, first_for_chain[cid])
                else:
                    first_for_chain[cid] = i_pos

            # (b) Union by source overlap (sweep-line)
            sort_idx = np.argsort(starts, kind="mergesort")
            max_end = -1
            max_end_pos = -1
            for k in range(n_ord):
                pos = int(sort_idx[k])
                if max_end_pos >= 0 and starts[pos] < max_end:
                    _union(pos, max_end_pos)
                if ends[pos] > max_end:
                    max_end = int(ends[pos])
                    max_end_pos = pos

            roots = np.array([_find(i_pos, parent) for i_pos in range(n_ord)], dtype=np.int64)
            # Compact root ids → contiguous cluster ids
            _, cluster_ids = np.unique(roots, return_inverse=True)
            ordered["__cluster_id"] = cluster_ids.astype(np.int64, copy=False)

        best_cluster: Any = None
        best_score: float | None = None
        best_min_start: int | None = None

        for cid, cgrp in ordered.groupby("__cluster_id", sort=False):
            cstarts = cgrp["__src_start"].to_numpy(dtype=np.int64, copy=False)
            cends = cgrp["__src_end"].to_numpy(dtype=np.int64, copy=False)
            lens = cends - cstarts

            if policy == "best_cluster_union":
                score = _interval_union_length(cstarts, cends)
            elif policy == "best_cluster_sum":
                score = float(np.sum(lens))
            else:  # best_cluster_max
                score = float(np.max(lens))

            min_start = int(np.min(cstarts))
            if (
                best_score is None
                or score > best_score
                or (score == best_score and best_min_start is not None and min_start < best_min_start)
            ):
                best_score = score
                best_min_start = min_start
                best_cluster = cid

        chosen_idx = ordered.loc[ordered["__cluster_id"] == best_cluster, "index"].to_numpy()
        kept.append(group.loc[chosen_idx])

    if not kept:
        return df.iloc[0:0].copy()

    return pd.concat(kept, axis=0).sort_index().reset_index(drop=True)


# ===================================================================
# Public API: gintervals_load_chain
# ===================================================================

def _chain_to_r(chain: pd.DataFrame) -> pd.DataFrame:
    """Chain strands as R gives them: 1 for '+', -1 for '-' (internally 0 / 1)."""
    out = chain.copy()
    for c in ("strand", "strandsrc"):
        if c in out.columns:
            out[c] = np.where(out[c].to_numpy() == 1, -1, 1).astype(np.int64)
    return out


def _chain_from_r(chain: pd.DataFrame) -> pd.DataFrame:
    """A chain DataFrame's R strands (1 / -1) as the internal 0 / 1. 0 is read
    as '+' too, so a chain written with the old {0, 1} plus strand still works."""
    out = chain.copy()
    for c in ("strand", "strandsrc"):
        if c in out.columns:
            out[c] = (out[c].to_numpy() == -1).astype(np.int64)
    return out


def _check_chain_src(chain: pd.DataFrame, path: str, src_groot: str) -> None:
    """R's src_groot check: the chains' source chromosomes exist in that
    database and their source ends fit (the session's database is restored)."""
    import gzip

    from .db import _temporary_db_root
    from .intervals import gintervals_all

    with _temporary_db_root(src_groot):
        allc = gintervals_all()
        sizes = dict(zip(allc["chrom"].astype(str), allc["end"].astype(np.int64), strict=True))
        canon = {}
        for c in pd.unique(chain["chromsrc"].astype(str)):
            try:
                cc = _pymisha.pm_normalize_chroms([c])[0]
            except (_pymisha.error, IndexError):
                cc = None
            if cc not in sizes:
                raise ValueError(f"Chromosome {c} does not exist in the database. "
                                 f"Known chromosomes: {', '.join(sizes)}.")
            canon[c] = cc
    ends = chain.groupby("chain_id", sort=False)[["chromsrc", "endsrc"]].agg({"chromsrc": "first", "endsrc": "max"})
    past = [cid for cid, r in ends.iterrows() if r["endsrc"] > sizes[canon[str(r["chromsrc"])]]]
    if past:
        line = 0
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as fh:
            for i, ln in enumerate(fh, 1):
                if ln.startswith("chain") and ln.split()[-1] == str(past[0]):
                    line = i
                    break
        raise ValueError(f"Chain file {path}, line {line}: reference end coordinate exceeds chromosome size")


def gintervals_load_chain(
    file: str,
    src_overlap_policy: str = "error",
    tgt_overlap_policy: str = "auto",
    src_groot: str | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Load an assembly conversion table from a UCSC chain file.

    Reads a UCSC-format chain file and returns an assembly conversion table
    (DataFrame) that maps coordinates between a source genome and the current
    target genome. The resulting table can be used with
    ``gintervals_liftover`` and ``gtrack_liftover`` to convert intervals or
    tracks from the source assembly to the current one.

    Source overlaps occur when the same source genome position maps to
    multiple target positions. Target overlaps occur when multiple source
    positions map to overlapping regions in the target genome. Both types
    of overlaps are handled according to the specified policies.

    Parameters
    ----------
    file : str
        Path to the UCSC chain file. The file must follow the standard UCSC
        chain format specification. Chains whose target chromosomes are not
        present in the current database are silently skipped.
    src_overlap_policy : str, optional
        Policy for handling source-side overlaps. One of:

        - ``"error"`` (default) -- raise an error if source overlaps are
          detected.
        - ``"keep"`` -- allow one source interval to map to multiple target
          intervals.
        - ``"discard"`` -- remove all chain intervals involved in source
          overlaps.
    tgt_overlap_policy : str, optional
        Policy for handling target-side overlaps. One of:

        - ``"error"`` -- raise an error if target overlaps are detected.
        - ``"auto"`` (default) -- alias for ``"auto_score"``.
        - ``"auto_score"`` -- segment overlapping target regions and select
          the chain with the highest alignment score per segment.
          Tie-breakers: longest span, then lowest chain_id.
        - ``"auto_longer"`` -- segment and select the chain with the longest
          span per segment. Tie-breakers: highest score, then lowest
          chain_id.
        - ``"auto_first"`` -- segment and select the chain with the lowest
          chain_id per segment.
        - ``"keep"`` -- preserve all overlapping intervals.
        - ``"discard"`` -- remove all chain intervals involved in target
          overlaps.
        - ``"agg"`` -- segment overlaps into disjoint sub-regions, retaining
          all contributing chains per region for downstream aggregation.
        - ``"best_source_cluster"`` -- cluster chains by source overlap and
          keep the cluster with the largest total target length.
        - ``"best_cluster_union"`` -- best cluster union strategy.
        - ``"best_cluster_sum"`` -- best cluster sum strategy.
        - ``"best_cluster_max"`` -- best cluster max strategy.
    src_groot : str, optional
        Path to the source genome database root. When given, every chain's
        source chromosome must exist there and its source end must fit in it,
        as in misha.
    min_score : float, optional
        Minimum alignment score threshold. Chains with scores below this
        value are filtered out before overlap resolution.

    Returns
    -------
    pandas.DataFrame
        Assembly conversion table with the following columns:

        - ``chrom`` (str) -- target chromosome name (normalized).
        - ``start`` (int) -- target interval start (0-based, inclusive).
        - ``end`` (int) -- target interval end (0-based, exclusive).
        - ``strand`` (int) -- target strand (0 = +, 1 = -).
        - ``chromsrc`` (str) -- source chromosome name.
        - ``startsrc`` (int) -- source interval start.
        - ``endsrc`` (int) -- source interval end.
        - ``strandsrc`` (int) -- source strand.
        - ``chain_id`` (int) -- chain identifier from the chain file.
        - ``score`` (float) -- chain alignment score.

        The overlap policies are stored in ``DataFrame.attrs`` as
        ``"src_overlap_policy"`` and ``"tgt_overlap_policy"``.

    Raises
    ------
    ValueError
        If the chain file is malformed, contains inconsistent chromosome
        sizes, has coordinates out of range, or if overlap policies are
        invalid. Also raised when ``src_overlap_policy="error"`` and source
        overlaps are detected, or ``tgt_overlap_policy="error"`` and target
        overlaps are detected.

    See Also
    --------
    gintervals_as_chain : Convert an existing DataFrame to chain format.
    gintervals_liftover : Lift intervals from one assembly to another.
    gtrack_liftover : Import a track from another assembly via liftover.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import os
    >>> chainfile = os.path.join(pm._GROOT, "data", "test.chain")
    >>> chain = pm.gintervals_load_chain(
    ...     chainfile, src_overlap_policy="keep"
    ... )
    >>> list(chain.columns)  # doctest: +NORMALIZE_WHITESPACE
    ['chrom', 'start', 'end', 'strand', 'chromsrc', 'startsrc', 'endsrc', 'strandsrc', 'chain_id', 'score']
    """
    _checkroot()

    if src_overlap_policy not in _SRC_POLICIES:
        raise ValueError(
            f"src_overlap_policy must be one of {sorted(_SRC_POLICIES)}, "
            f"got '{src_overlap_policy}'"
        )
    if tgt_overlap_policy not in _TGT_POLICIES:
        raise ValueError(
            f"tgt_overlap_policy must be one of {sorted(_TGT_POLICIES)}, "
            f"got '{tgt_overlap_policy}'"
        )

    # Normalize "auto" alias
    if tgt_overlap_policy == "auto":
        tgt_overlap_policy = "auto_score"

    # Effective policy for loading: clustering policies load as "keep"
    effective_tgt_policy = tgt_overlap_policy
    if tgt_overlap_policy in ("best_source_cluster", "best_cluster_union",
                              "best_cluster_sum", "best_cluster_max"):
        effective_tgt_policy = "keep"

    db_chrom_sizes = _get_db_chrom_sizes()
    blocks = _parse_chain_file(file, db_chrom_sizes, min_score=min_score)

    chain = _empty_chain_df() if blocks is None else pd.DataFrame(blocks)[_EMPTY_CHAIN_COLS]
    if src_groot is not None:
        _check_chain_src(chain, file, src_groot)

    # Handle overlaps via dispatcher (C++ by default; pure-Python under
    # PYMISHA_FORCE_PY_CHAIN_INTERVALS_RESOLVE=1). Convert the current chain
    # DataFrame to a numpy dict for the call.
    chain_dict = {c: chain[c].to_numpy() for c in _EMPTY_CHAIN_COLS}
    resolved_dict = _resolve_chain_overlaps(
        chain_dict, src_overlap_policy, effective_tgt_policy,
    )
    chain = pd.DataFrame(resolved_dict)[_EMPTY_CHAIN_COLS]

    # Store policies as DataFrame attrs
    chain.attrs["src_overlap_policy"] = src_overlap_policy
    chain.attrs["tgt_overlap_policy"] = tgt_overlap_policy
    if min_score is not None:
        chain.attrs["min_score"] = min_score

    # rows in target order, as R returns them
    out = _sort_by_chrom(_chain_to_r(chain), ["chrom", "start", "end"]).reset_index(drop=True)
    out.attrs = dict(chain.attrs)
    return out


# ===================================================================
# Public API: gintervals_as_chain
# ===================================================================

def gintervals_as_chain(
    intervals: pd.DataFrame,
    src_overlap_policy: str = "error",
    tgt_overlap_policy: str = "auto",
    min_score: float | None = None,
) -> pd.DataFrame:
    """Convert a DataFrame to chain format by validating columns and setting attributes.

    Validates that the input DataFrame has all required chain columns and
    attaches overlap-policy metadata as DataFrame attributes. This is useful
    when you have manually constructed or modified chain data and need to
    mark it as a valid chain table for use with ``gintervals_liftover`` or
    ``gtrack_liftover``.

    Parameters
    ----------
    intervals : pandas.DataFrame
        A DataFrame that must contain all of the required chain columns:
        ``chrom``, ``start``, ``end``, ``strand``, ``chromsrc``,
        ``startsrc``, ``endsrc``, ``strandsrc``, ``chain_id``, ``score``.
    src_overlap_policy : str, optional
        Policy for handling source-side overlaps. One of ``"error"``
        (default), ``"keep"``, or ``"discard"``. This value is stored as a
        DataFrame attribute but does not trigger overlap resolution.
    tgt_overlap_policy : str, optional
        Policy for handling target-side overlaps. One of ``"error"``,
        ``"auto"`` (default, alias for ``"auto_score"``), ``"auto_score"``,
        ``"auto_longer"``, ``"auto_first"``, ``"keep"``, ``"discard"``,
        ``"agg"``, ``"best_source_cluster"``, ``"best_cluster_union"``,
        ``"best_cluster_sum"``, ``"best_cluster_max"``. Stored as a
        DataFrame attribute.
    min_score : float, optional
        Minimum alignment score threshold to record as a DataFrame attribute.
        Does not filter the data; the value is stored for informational use
        by downstream functions.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with overlap-policy attributes set in
        ``DataFrame.attrs``:

        - ``"src_overlap_policy"`` -- the source overlap policy.
        - ``"tgt_overlap_policy"`` -- the target overlap policy (``"auto"``
          is normalized to ``"auto_score"``).
        - ``"min_score"`` -- present only if *min_score* was provided.

    Raises
    ------
    TypeError
        If *intervals* is not a ``pandas.DataFrame``.
    ValueError
        If required columns are missing, or if either overlap policy string
        is not a recognized value.

    See Also
    --------
    gintervals_load_chain : Load a chain from a UCSC chain file.
    gintervals_liftover : Lift intervals from one assembly to another.
    gtrack_liftover : Import a track from another assembly via liftover.

    Examples
    --------
    >>> import pandas as pd
    >>> import pymisha as pm
    >>> chain_data = pd.DataFrame({
    ...     "chrom": ["1"], "start": [1000], "end": [2000], "strand": [1],
    ...     "chromsrc": ["chr25"], "startsrc": [5000], "endsrc": [6000],
    ...     "strandsrc": [1], "chain_id": [1], "score": [1000.0],
    ... })
    >>> chain = pm.gintervals_as_chain(chain_data)
    >>> chain.attrs["tgt_overlap_policy"]
    'auto_score'
    """
    if not isinstance(intervals, pd.DataFrame):
        raise TypeError("intervals must be a DataFrame")

    required = set(_EMPTY_CHAIN_COLS)
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if src_overlap_policy not in _SRC_POLICIES:
        raise ValueError(
            f"src_overlap_policy must be one of {sorted(_SRC_POLICIES)}, "
            f"got '{src_overlap_policy}'"
        )
    if tgt_overlap_policy not in _TGT_POLICIES:
        raise ValueError(
            f"tgt_overlap_policy must be one of {sorted(_TGT_POLICIES)}, "
            f"got '{tgt_overlap_policy}'"
        )

    if min_score is not None and not (isinstance(min_score, (int, float, np.integer, np.floating))
                                      and not isinstance(min_score, bool)):
        raise ValueError("min_score must be a single numeric value")
    # R's column checks (gintervals.as_chain)
    num = pd.api.types.is_numeric_dtype
    if not (num(intervals["start"]) and num(intervals["end"])):
        raise ValueError("start and end columns must be numeric")
    if not (num(intervals["startsrc"]) and num(intervals["endsrc"])):
        raise ValueError("startsrc and endsrc columns must be numeric")
    for col in ("strand", "strandsrc"):
        if not num(intervals[col]) or pd.api.types.is_bool_dtype(intervals[col]):
            raise ValueError(f"{col} column must be numeric")
    if not num(intervals["chain_id"]):
        raise ValueError("chain_id column must be integer or numeric")
    if not num(intervals["score"]):
        raise ValueError("score column must be numeric")
    for col in ("strand", "strandsrc"):
        v = intervals[col].to_numpy(dtype=float)
        if ((v != np.trunc(v)) | (v < -1) | (v > 1)).any():
            raise ValueError(f"{col} values must be -1, 0, or 1")

    if tgt_overlap_policy == "auto":
        tgt_overlap_policy = "auto_score"

    result = intervals.copy()
    result["chain_id"] = result["chain_id"].astype(np.int64)  # as R's as.integer
    result.attrs["src_overlap_policy"] = src_overlap_policy
    result.attrs["tgt_overlap_policy"] = tgt_overlap_policy
    if min_score is not None:
        result.attrs["min_score"] = min_score

    return result


# ===================================================================
# Vectorized coordinate mapping
# ===================================================================

def _map_intervals_vectorized(
    intervals: pd.DataFrame,
    chain_df: pd.DataFrame,
    include_metadata: bool,
    value_col: str | None,
    cluster_strategy: str = "",
    _force_pure_python: bool = False,
) -> pd.DataFrame:
    """Dispatch _map_intervals_vectorized to C++ by default; Python fallback.

    The C++ path is used unless ``_force_pure_python=True`` or the env var
    ``PYMISHA_FORCE_PY_MAP_INTERVALS=1`` is set. The Python path remains the
    reference implementation and is exercised by ``TestCrossValidatePython``.

    ``cluster_strategy``: when ``"union"``, ``"sum"``, or ``"max"``, per-src-interval
    cluster resolution is applied inside the C++ call. The Python fallback path
    returns un-resolved candidates and expects the caller to apply
    ``_resolve_cluster_policy`` separately.
    """
    use_py = _force_pure_python or os.environ.get(
        "PYMISHA_FORCE_PY_MAP_INTERVALS", ""
    ).lower() in ("1", "true", "yes")

    if use_py:
        return _map_intervals_vectorized_python(
            intervals, chain_df, include_metadata, value_col,
        )

    empty_cols = ["chrom", "start", "end", "intervalID", "chain_id"]
    if include_metadata:
        empty_cols.append("score")
    if value_col:
        empty_cols.append(value_col)

    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame({
            c: pd.Series(
                dtype="object" if c == "chrom" else "int64" if c in (
                    "start", "end", "intervalID", "chain_id"
                ) else "float64"
            ) for c in empty_cols
        })

    if chain_df.empty or len(intervals) == 0:
        return _empty_result()

    # Marshal to numpy dicts.
    src_dict = {
        "chrom": intervals["chrom"].to_numpy(),
        "start": intervals["start"].to_numpy(dtype=np.int64, copy=False),
        "end":   intervals["end"].to_numpy(dtype=np.int64, copy=False),
    }
    if value_col and value_col in intervals.columns:
        src_dict[value_col] = intervals[value_col].to_numpy(dtype=np.float64, copy=False)
        value_col_arg = value_col
    else:
        value_col_arg = ""

    chain_dict = {c: chain_df[c].to_numpy() for c in _EMPTY_CHAIN_COLS}

    result = _pymisha.pm_map_intervals(
        src_dict, chain_dict, value_col_arg, bool(include_metadata),
        cluster_strategy,
    )
    return pd.DataFrame(result)


def _map_intervals_vectorized_python(
    intervals: pd.DataFrame,
    chain_df: pd.DataFrame,
    include_metadata: bool,
    value_col: str | None,
) -> pd.DataFrame:
    """Map source intervals through chain blocks using vectorized numpy ops.

    For each source interval, finds overlapping chain blocks and computes
    target coordinates.  Returns a DataFrame with columns:
        chrom, start, end, intervalID, chain_id, __src_start, __src_end
    and optionally score (if include_metadata) and value_col.

    The chain blocks are sorted by (chromsrc, startsrc) per source chromosome.
    Overlap finding uses np.searchsorted on sorted arrays.  Coordinate
    transformation is fully vectorized over all overlapping pairs.
    """
    empty_cols = ["chrom", "start", "end", "intervalID", "chain_id"]
    if include_metadata:
        empty_cols.append("score")
    if value_col:
        empty_cols.append(value_col)

    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame({
            c: pd.Series(
                dtype="object" if c == "chrom" else "int64" if c in (
                    "start", "end", "intervalID", "chain_id"
                ) else "float64"
            ) for c in empty_cols
        })

    if chain_df.empty or len(intervals) == 0:
        return _empty_result()

    # Sort chain by (chromsrc, startsrc, endsrc) and extract numpy arrays
    chain_sorted = chain_df.sort_values(
        ["chromsrc", "startsrc", "endsrc"],
    ).reset_index(drop=True)

    ch_chromsrc = chain_sorted["chromsrc"].to_numpy()
    ch_startsrc = chain_sorted["startsrc"].to_numpy(dtype=np.int64, copy=False)
    ch_endsrc = chain_sorted["endsrc"].to_numpy(dtype=np.int64, copy=False)
    ch_chrom = chain_sorted["chrom"].to_numpy()
    ch_start = chain_sorted["start"].to_numpy(dtype=np.int64, copy=False)
    ch_end = chain_sorted["end"].to_numpy(dtype=np.int64, copy=False)
    ch_strand = chain_sorted["strand"].to_numpy(dtype=np.int64, copy=False)
    ch_chain_id = chain_sorted["chain_id"].to_numpy(dtype=np.int64, copy=False)
    ch_score = chain_sorted["score"].to_numpy(dtype=np.float64, copy=False)

    # Build per-chrom slice boundaries and prefix-max of endsrc
    # for efficient overlap search
    chrom_slices = {}  # chromsrc -> (first_idx, last_idx_excl)
    n_ch = len(chain_sorted)
    pmax_endsrc = ch_endsrc.copy()

    i = 0
    while i < n_ch:
        chrom = ch_chromsrc[i]
        first = i
        running_max = ch_endsrc[i]
        pmax_endsrc[i] = running_max
        i += 1
        while i < n_ch and ch_chromsrc[i] == chrom:
            running_max = max(running_max, ch_endsrc[i])
            pmax_endsrc[i] = running_max
            i += 1
        chrom_slices[chrom] = (first, i)

    # Extract source interval arrays
    iv_chroms = intervals["chrom"].to_numpy()
    iv_starts = intervals["start"].to_numpy(dtype=np.int64, copy=False)
    iv_ends = intervals["end"].to_numpy(dtype=np.int64, copy=False)
    n_iv = len(intervals)

    has_value_col = value_col and value_col in intervals.columns
    # Bound unconditionally so the use below is not merely *conditionally*
    # defined, mirroring how all_r_value is handled a few lines down.
    iv_values: np.ndarray | None = None
    if has_value_col:
        iv_values = intervals[value_col].to_numpy(dtype=np.float64, copy=False)

    # Collect result arrays — process per source chromosome for cache locality
    all_r_tgt_chrom = []
    all_r_tgt_start = []
    all_r_tgt_end = []
    all_r_interval_id = []
    all_r_chain_id = []
    all_r_src_start = []
    all_r_src_end = []
    all_r_score: list[np.ndarray] | None = [] if include_metadata else None
    all_r_value: list[np.ndarray] | None = [] if has_value_col else None

    # Group source intervals by chrom for batch processing
    # Use stable sort to preserve original ordering within each chrom
    iv_order = np.argsort(iv_chroms, kind="mergesort")
    iv_chroms_sorted = iv_chroms[iv_order]

    # Find chrom group boundaries
    if n_iv > 0:
        iv_chrom_breaks = np.flatnonzero(
            np.r_[True, iv_chroms_sorted[1:] != iv_chroms_sorted[:-1], True]
        )
    else:
        iv_chrom_breaks = np.array([0], dtype=np.intp)

    for g in range(len(iv_chrom_breaks) - 1):
        g_start = iv_chrom_breaks[g]
        g_end = iv_chrom_breaks[g + 1]
        src_chrom = iv_chroms_sorted[g_start]

        if src_chrom not in chrom_slices:
            continue

        ch_first, ch_last = chrom_slices[src_chrom]
        n_chain = ch_last - ch_first

        # Chain arrays for this source chrom (sliced views)
        c_startsrc = ch_startsrc[ch_first:ch_last]
        c_endsrc = ch_endsrc[ch_first:ch_last]
        c_pmax = pmax_endsrc[ch_first:ch_last]
        c_chrom = ch_chrom[ch_first:ch_last]
        c_start = ch_start[ch_first:ch_last]
        c_end = ch_end[ch_first:ch_last]
        c_strand = ch_strand[ch_first:ch_last]
        c_chain_id = ch_chain_id[ch_first:ch_last]
        c_score = ch_score[ch_first:ch_last]

        # Source interval indices and arrays for this chrom group
        g_indices = iv_order[g_start:g_end]  # original interval IDs
        g_starts = iv_starts[g_indices]
        g_ends = iv_ends[g_indices]
        n_src = len(g_indices)

        # For each source interval, find the range of potentially overlapping
        # chain blocks.
        #
        # Chain blocks are sorted by startsrc.  A chain block j overlaps
        # source interval i iff:
        #     c_startsrc[j] < g_ends[i]  AND  c_endsrc[j] > g_starts[i]
        #
        # Upper bound: first j where c_startsrc[j] >= g_ends[i]
        #   => np.searchsorted(c_startsrc, g_ends, side='left')
        #
        # Lower bound: we need the first j that could overlap.  Since blocks
        # are sorted by startsrc but endsrc can extend arbitrarily far, we
        # use the prefix-max of endsrc.  The first j with
        # pmax_endsrc[j] > g_starts[i] is our lower bound.
        #   => np.searchsorted(c_pmax, g_starts, side='right')
        #   (searchsorted 'right' gives first index where c_pmax > g_starts)
        upper = np.searchsorted(c_startsrc, g_ends, side="left")  # (n_src,)
        lower = np.searchsorted(c_pmax, g_starts, side="right")   # (n_src,)

        # Clip to valid range
        np.clip(upper, 0, n_chain, out=upper)
        np.clip(lower, 0, n_chain, out=lower)

        # Count candidate chain blocks per source interval
        counts = np.maximum(upper - lower, 0)  # (n_src,)
        total_candidates = int(counts.sum())

        if total_candidates == 0:
            continue

        # Expand: for each source interval, enumerate all candidate chain
        # block indices.  Build flat arrays of (src_idx, chain_idx) pairs.
        # Use np.repeat + arange trick for expansion.
        src_repeat = np.repeat(np.arange(n_src, dtype=np.intp), counts)
        # Compute flat chain indices: for source i, chain indices are
        # lower[i], lower[i]+1, ..., upper[i]-1
        offsets_within = np.arange(total_candidates, dtype=np.intp)
        group_offsets = np.repeat(np.cumsum(counts) - counts, counts)
        chain_idx = np.asarray(
            np.repeat(lower, counts) + (offsets_within - group_offsets),
            dtype=np.intp,
        )

        # Gather chain and source values for all candidate pairs
        p_src_start = g_starts[src_repeat]       # source interval starts
        p_src_end = g_ends[src_repeat]            # source interval ends
        p_ch_startsrc = c_startsrc[chain_idx]     # chain source starts
        p_ch_endsrc = c_endsrc[chain_idx]         # chain source ends

        # Compute overlap: common_start, common_end
        common_start = np.maximum(p_src_start, p_ch_startsrc)
        common_end = np.minimum(p_src_end, p_ch_endsrc)

        # Filter to actual overlaps (common_start < common_end)
        valid = common_start < common_end
        if not valid.any():
            continue

        # Apply filter
        common_start = common_start[valid]
        common_end = common_end[valid]
        v_chain_idx = chain_idx[valid]
        v_src_repeat = src_repeat[valid]

        # Gather chain target-side arrays
        v_ch_chrom = c_chrom[v_chain_idx]
        v_ch_start = c_start[v_chain_idx]
        v_ch_end = c_end[v_chain_idx]
        v_ch_strand = c_strand[v_chain_idx]
        v_ch_chain_id = c_chain_id[v_chain_idx]
        v_ch_startsrc = c_startsrc[v_chain_idx]

        # Vectorized coordinate transformation
        # offset from chain source start
        offset_start = common_start - v_ch_startsrc
        offset_end = common_end - v_ch_startsrc

        # Positive strand: tgt = ch_start + offset
        # Negative strand: tgt_start = ch_end - offset_end
        #                   tgt_end   = ch_end - offset_start
        pos_mask = v_ch_strand == 0

        tgt_start = np.empty_like(common_start)
        tgt_end = np.empty_like(common_end)

        if pos_mask.any():
            tgt_start[pos_mask] = v_ch_start[pos_mask] + offset_start[pos_mask]
            tgt_end[pos_mask] = v_ch_start[pos_mask] + offset_end[pos_mask]

        neg_mask = ~pos_mask
        if neg_mask.any():
            tgt_start[neg_mask] = v_ch_end[neg_mask] - offset_end[neg_mask]
            tgt_end[neg_mask] = v_ch_end[neg_mask] - offset_start[neg_mask]

        # Map src_repeat back to original interval IDs
        v_interval_ids = g_indices[v_src_repeat]

        all_r_tgt_chrom.append(v_ch_chrom)
        all_r_tgt_start.append(tgt_start)
        all_r_tgt_end.append(tgt_end)
        all_r_interval_id.append(v_interval_ids)
        all_r_chain_id.append(v_ch_chain_id)
        all_r_src_start.append(common_start)
        all_r_src_end.append(common_end)

        if include_metadata:
            assert all_r_score is not None
            all_r_score.append(c_score[v_chain_idx])

        if has_value_col:
            assert all_r_value is not None
            assert iv_values is not None
            all_r_value.append(iv_values[v_interval_ids])

    # Concatenate results from all chrom groups
    if not all_r_tgt_chrom:
        return _empty_result()

    result_data = {
        "chrom": np.concatenate(all_r_tgt_chrom),
        "start": np.concatenate(all_r_tgt_start),
        "end": np.concatenate(all_r_tgt_end),
        "intervalID": np.concatenate(all_r_interval_id),
        "chain_id": np.concatenate(all_r_chain_id),
        "__src_start": np.concatenate(all_r_src_start),
        "__src_end": np.concatenate(all_r_src_end),
    }
    if include_metadata:
        assert all_r_score is not None
        result_data["score"] = np.concatenate(all_r_score)
    if has_value_col and value_col is not None:
        assert all_r_value is not None
        result_data[value_col] = np.concatenate(all_r_value)

    return pd.DataFrame(result_data)


# ===================================================================
# Public API: gintervals_liftover
# ===================================================================

def gintervals_liftover(
    intervals: pd.DataFrame,
    chain: str | pd.DataFrame,
    src_overlap_policy: str = "error",
    tgt_overlap_policy: str = "auto",
    min_score: float | None = None,
    include_metadata: bool = False,
    canonic: bool = False,
    value_col: str | None = None,
    multi_target_agg: str = "mean",
    params: dict[str, Any] | int | None = None,
    na_rm: bool = True,
    min_n: int | None = None,
) -> pd.DataFrame:
    """Convert intervals from another assembly to the current one using a chain.

    Maps each source interval through the chain's alignment blocks to produce
    the corresponding target-genome coordinates. A single source interval may
    produce multiple target intervals when it spans chain gaps or maps through
    multiple chains. The ``intervalID`` column in the output links each result
    row back to the originating source interval (0-based positional index).

    When *chain* is a file path, it is loaded with the specified overlap
    policies. When it is a pre-loaded DataFrame (from ``gintervals_load_chain``
    or ``gintervals_as_chain``), the policies stored in its attributes are
    used and the policy arguments here are ignored.

    Parameters
    ----------
    intervals : pandas.DataFrame
        Source-assembly intervals. Must contain at least the columns
        ``chrom``, ``start``, and ``end``. Chromosome names should match the
        source side of the chain (``chromsrc``).
    chain : str or pandas.DataFrame
        Either a path to a UCSC chain file (loaded via
        ``gintervals_load_chain``) or a pre-loaded chain DataFrame.
    src_overlap_policy : str, optional
        Source overlap policy, used only when *chain* is a file path.
        One of ``"error"`` (default), ``"keep"``, or ``"discard"``.
    tgt_overlap_policy : str, optional
        Target overlap policy, used only when *chain* is a file path.
        One of ``"error"``, ``"auto"`` (default), ``"auto_score"``,
        ``"auto_longer"``, ``"auto_first"``, ``"keep"``, ``"discard"``,
        ``"agg"``, ``"best_source_cluster"``, ``"best_cluster_union"``,
        ``"best_cluster_sum"``, ``"best_cluster_max"``.
    min_score : float, optional
        Minimum chain alignment score, used only when *chain* is a file
        path. Chains scoring below this threshold are excluded.
    include_metadata : bool, optional
        If ``True``, a ``score`` column is added to the output containing
        the alignment score of the chain that produced each mapping.
        Default is ``False``.
    canonic : bool, optional
        If ``True``, adjacent target intervals originating from the same
        source interval (same ``intervalID``) and the same chain (same
        ``chain_id``) are merged into a single interval. Useful when a
        source interval maps to multiple adjacent target blocks separated
        by chain alignment gaps. Default is ``False``.
    value_col : str, optional
        Name of a numeric column in *intervals* whose values should be
        carried through the liftover. When specified, the output includes
        this column with its original name. Ignored if ``None``.
    multi_target_agg : str, optional
        Aggregation method applied to *value_col* when multiple source
        intervals map to the same target region. One of ``"mean"``
        (default), ``"median"``, ``"sum"``, ``"min"``, ``"max"``,
        ``"count"``, ``"first"``, ``"last"``. Ignored when *value_col* is
        ``None``.
    params : dict or int, optional
        Additional parameters for specific aggregation methods (e.g.,
        ``n`` for ``"nth"`` aggregation).
    na_rm : bool, optional
        If ``True`` (default), ``NaN`` values are removed before
        aggregation. If ``False``, any ``NaN`` in the group causes the
        aggregated result to be ``NaN``. Only used when *value_col* is
        specified.
    min_n : int, optional
        Minimum number of non-``NaN`` values required for aggregation. If
        fewer values are available, the result is ``NaN``. ``None``
        (default) means no minimum. Only used when *value_col* is specified.

    Returns
    -------
    pandas.DataFrame
        Lifted intervals sorted by target coordinates with the columns:

        - ``chrom`` (str) -- target chromosome.
        - ``start`` (int) -- target start (0-based, inclusive).
        - ``end`` (int) -- target end (0-based, exclusive).
        - ``intervalID`` (int) -- 0-based index of the source interval in
          the input *intervals* DataFrame.
        - ``chain_id`` (int) -- identifier of the chain that produced the
          mapping.
        - ``score`` (float) -- chain alignment score (only when
          *include_metadata* is ``True``).
        - *value_col* (float) -- carried-through values (only when
          *value_col* is specified).

    Raises
    ------
    ValueError
        If *intervals* or *chain* is ``None``, or if a file-path chain
        cannot be loaded.

    See Also
    --------
    gintervals_load_chain : Load a chain from a UCSC chain file.
    gintervals_as_chain : Convert a DataFrame to chain format.
    gtrack_liftover : Import a full track from another assembly.

    Examples
    --------
    >>> import pandas as pd
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import os
    >>> chainfile = os.path.join(pm._GROOT, "data", "test.chain")
    >>> intervs = pd.DataFrame({
    ...     "chrom": ["chr25", "chr25"],
    ...     "start": [0, 7000],
    ...     "end": [6000, 20000],
    ... })
    >>> lifted = pm.gintervals_liftover(
    ...     intervs, chainfile, src_overlap_policy="keep"
    ... )
    >>> list(lifted.columns)  # doctest: +NORMALIZE_WHITESPACE
    ['chrom', 'start', 'end', 'intervalID', 'chain_id']
    """
    _checkroot()

    if intervals is None or chain is None:
        raise ValueError("intervals and chain are required")

    nth = 0
    if value_col is not None:
        if not isinstance(value_col, str):
            raise ValueError("value_col must be a single character string specifying the column name")
        if value_col not in intervals.columns:
            raise ValueError(f"value_col '{value_col}' not found in intervals")
        if not isinstance(na_rm, (bool, np.bool_)):
            raise ValueError("na_rm must be a single non-NA logical value")
        if min_n is not None and (
            isinstance(min_n, bool) or not isinstance(min_n, (int, float, np.integer, np.floating))
            or not np.isfinite(min_n) or min_n < 0 or min_n != int(min_n)
        ):
            raise ValueError("min_n must be None or a non-negative integer")
        nth = _nth_param(multi_target_agg, params)

    # Load chain if file path
    if isinstance(chain, str):
        chain_df = gintervals_load_chain(
            chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            min_score=min_score,
        )
    else:
        # a target chromosome named by an alias (such as "1" for chr1) is the
        # database's chromosome, as in misha
        chain_df = _with_canonical_chroms(chain)
    chain_df = _chain_from_r(chain_df)

    if "chrom1" in intervals.columns:
        return _liftover_2d(intervals, chain_df)

    # as R (IntervalsLiftover.cpp): intervals none of which is on a chromosome of
    # the chain's source genome are an error, not an empty result
    if len(intervals) and not chain_df.empty and not set(intervals["chrom"].astype(str)) & set(
            chain_df["chromsrc"].astype(str)):
        raise ValueError("Source chromosome does not exist in the chain source genome")

    if chain_df.empty:
        cols = ["chrom", "start", "end", "intervalID", "chain_id"]
        if include_metadata:
            cols.append("score")
        if value_col:
            cols.append(value_col)
        return pd.DataFrame({
            c: pd.Series(
                dtype="object" if c == "chrom" else "int64" if c in (
                    "start", "end", "intervalID", "chain_id"
                ) else "float64"
            ) for c in cols
        })

    effective_tgt_policy = chain_df.attrs.get("tgt_overlap_policy", tgt_overlap_policy)
    if effective_tgt_policy == "auto":
        effective_tgt_policy = "auto_score"

    # Map cluster policy to the C++ strategy enum.
    _STRAT_MAP = {
        "best_source_cluster": "union",
        "best_cluster_union":  "union",
        "best_cluster_sum":    "sum",
        "best_cluster_max":    "max",
    }
    strat = _STRAT_MAP.get(effective_tgt_policy, "")

    result = _map_intervals_vectorized(
        intervals, chain_df, include_metadata, value_col,
        cluster_strategy=strat,
    )
    # When the C++ path runs with strat != "", cluster resolution is already
    # done. The Python fallback path returns un-resolved candidates; apply
    # _resolve_cluster_policy when the env var forces Python.
    if strat and os.environ.get(
        "PYMISHA_FORCE_PY_MAP_INTERVALS", ""
    ).lower() in ("1", "true", "yes"):
        result = _resolve_cluster_policy(result, effective_tgt_policy)

    helper_cols = [c for c in ("__src_start", "__src_end") if c in result.columns]
    if helper_cols:
        result = result.drop(columns=helper_cols)

    if value_col is not None:
        result = _aggregate_lifted(
            result, value_col, multi_target_agg, na_rm, min_n, nth, include_metadata,
            drop_ids=tgt_overlap_policy == "agg",
        )

    # Canonic merging: merge adjacent target blocks from same intervalID + chain_id
    if canonic:
        # R's canonic pass leaves the rows by intervalID, chain_id, target
        result = _canonic_merge(result, include_metadata, value_col)
    elif value_col is not None:
        # R's aggregation sweeps each target chromosome
        result = _sort_by_chrom(result, ["chrom", "start", "end"])
    else:
        result = _r_mapping_order(result, intervals)
    # R's intervalID is 1-based
    if "intervalID" in result.columns:
        result["intervalID"] = result["intervalID"] + 1
    return result.reset_index(drop=True)


def _liftover_2d(intervals: pd.DataFrame, chain_df: pd.DataFrame) -> pd.DataFrame:
    """R's 2D gintervals.liftover: each side mapped on its own, every target piece of
    side 1 paired with every one of side 2. Rows on a chromosome the chain does not
    map are dropped; the rest are walked in R's order (sorted by source chromosome
    pair) and each one's pieces in chain order. No aggregation, metadata or
    canonic merging, as in R."""
    cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    # R numbers the source chromosomes in the order the chain names them
    src_id = {c: i for i, c in enumerate(dict.fromkeys(chain_df["chromsrc"].astype(str)))}
    c1 = intervals["chrom1"].astype(str).map(src_id).to_numpy(dtype=float)
    c2 = intervals["chrom2"].astype(str).map(src_id).to_numpy(dtype=float)
    rows = np.flatnonzero(~np.isnan(c1) & ~np.isnan(c2))
    if len(intervals) and not len(rows):
        raise ValueError("Source chromosome does not exist in the chain source genome")
    rows = rows[_pymisha.pm_r_2d_sort_order(c1[rows].astype(np.int64), c2[rows].astype(np.int64))]
    n = len(rows)

    def side(k):
        src = pd.DataFrame({
            "chrom": intervals[f"chrom{k}"].astype(str).to_numpy()[rows],
            "start": intervals[f"start{k}"].to_numpy(np.int64)[rows],
            "end": intervals[f"end{k}"].to_numpy(np.int64)[rows],
        })
        m = _map_intervals_vectorized(src, chain_df, False, None) if n else src.iloc[:0].assign(intervalID=[])
        m = m.iloc[np.argsort(m["intervalID"].to_numpy(np.int64), kind="stable")]
        cnt = np.bincount(m["intervalID"].to_numpy(np.int64), minlength=n)
        return m, cnt, np.cumsum(cnt) - cnt

    m1, n1, off1 = side(1)
    m2, n2, off2 = side(2)
    k = n1 * n2
    src = np.repeat(np.arange(n), k)
    within = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    a = off1[src] + within // n2[src]
    b = off2[src] + within % n2[src]
    return pd.DataFrame({
        "chrom1": m1["chrom"].to_numpy()[a], "start1": m1["start"].to_numpy(np.int64)[a],
        "end1": m1["end"].to_numpy(np.int64)[a],
        "chrom2": m2["chrom"].to_numpy()[b], "start2": m2["start"].to_numpy(np.int64)[b],
        "end2": m2["end"].to_numpy(np.int64)[b],
        "intervalID": rows[src] + 1,  # R's intervalID is the 1-based input row
    }, columns=[*cols, "intervalID"])


def _r_mapping_order(result: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Rows in the order R's gintervals.liftover maps them: the source intervals
    sorted (source chromosome, then start), each one's pieces in chain order."""
    if len(result) < 2:
        return result
    iid = result["intervalID"].to_numpy(np.int64)
    src_chrom = intervals["chrom"].astype(str).to_numpy()[iid]
    src_start = intervals["start"].to_numpy(np.int64)[iid]
    src_chrom_rank = pd.Index(sorted(set(src_chrom))).get_indexer(src_chrom)
    # stable: within one source interval, the mapper already gives R's chain order
    order = np.lexsort((iid, src_start, src_chrom_rank))
    return result.iloc[order]


def _canonic_merge(
    df: pd.DataFrame,
    include_metadata: bool,
    value_col: str | None,
) -> pd.DataFrame:
    """Merge adjacent target blocks from same intervalID and chain_id.

    Uses vectorized numpy operations for group detection and aggregation.
    """
    if df.empty or "intervalID" not in df.columns or "chain_id" not in df.columns:
        return df

    # Sort by intervalID, chain_id, chrom, start
    df = df.sort_values(["intervalID", "chain_id", "chrom", "start"]).reset_index(drop=True)
    n = len(df)
    if n <= 1:
        return df

    interval_ids = df["intervalID"].to_numpy()
    chain_ids = df["chain_id"].to_numpy()
    chroms = df["chrom"].to_numpy()
    starts = df["start"].to_numpy(dtype=np.int64, copy=False)
    ends = df["end"].to_numpy(dtype=np.int64, copy=False)

    # A new merge group starts where any key changes or blocks aren't adjacent
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = (
        (interval_ids[1:] != interval_ids[:-1])
        | (chain_ids[1:] != chain_ids[:-1])
        | (chroms[1:] != chroms[:-1])
        | (starts[1:] != ends[:-1])
    )

    group_starts_idx = np.flatnonzero(new_group)
    n_groups = len(group_starts_idx)

    # For each group: start = first block's start, end = last block's end
    group_ends_idx = np.empty(n_groups, dtype=np.intp)
    group_ends_idx[:-1] = group_starts_idx[1:] - 1
    group_ends_idx[-1] = n - 1

    # Build merged DataFrame — take all columns from first row of each group,
    # then fix "end" from the last row.
    merged = df.iloc[group_starts_idx].copy()
    merged["end"] = ends[group_ends_idx]

    return merged.reset_index(drop=True)


def _aggregate_lifted(
    df: pd.DataFrame,
    value_col: str,
    agg: str,
    na_rm: bool,
    min_n: int | None,
    nth: int,
    include_metadata: bool,
    *,
    drop_ids: bool,
) -> pd.DataFrame:
    """R's value_col aggregation in gintervals.liftover (IntervalsLiftover.cpp).

    Per target chrom, the lifted intervals are cut at every start/end, each
    piece takes the aggregate of the intervals covering it, and adjacent pieces
    with the same value, intervalID and chain_id are merged. intervalID /
    chain_id (and score) are NA where the contributors disagree; a column that is
    all NA is dropped, and both are dropped under tgt_overlap_policy="agg".

    ponytail: a Python loop per piece, fine for typical interval sets; move it to
    C++ (as R's IntervalsLiftover.cpp) if million-interval lifts become common.
    """
    cols = ["chrom", "start", "end", "intervalID", "chain_id"] + (["score"] if include_metadata else []) + [value_col]
    if df.empty:
        return df[cols].drop(columns=["intervalID", "chain_id"] if drop_ids else [])
    nan = float("nan")
    out: dict[str, list] = {c: [] for c in cols}
    for chrom, g in df.groupby("chrom", sort=False, observed=True):
        starts = g["start"].to_numpy(np.int64)
        ends = g["end"].to_numpy(np.int64)
        vals = g[value_col].to_numpy(float)
        iids = g["intervalID"].to_numpy(np.int64)
        cids = g["chain_id"].to_numpy(np.int64)
        scores = g["score"].to_numpy(float) if include_metadata else np.full(len(g), nan)
        breaks = np.unique(np.concatenate([starts, ends]))
        covering: list[list[int]] = [[] for _ in range(len(breaks) - 1)]
        for i, k0 in enumerate(np.searchsorted(breaks, starts)):
            for k in range(k0, np.searchsorted(breaks, ends[i])):
                covering[k].append(i)
        for k, idx in enumerate(covering):
            if not idx:
                continue
            s, e = int(breaks[k]), int(breaks[k + 1])
            contribs = [
                {"value": vals[i], "overlap_len": float(e - s), "coverage_frac": 1.0, "start": s, "end": e,
                 "is_na": bool(np.isnan(vals[i])), "chain_id": (int(cids[i]) << 32) ^ int(iids[i])}
                for i in idx
            ]
            v = _aggregate_value_for_bin(contribs, agg, na_rm, min_n, nth)
            v = nan if np.isnan(v) else float(np.float32(v))  # R stores the value as float
            iid = int(iids[idx[0]]) if all(iids[i] == iids[idx[0]] for i in idx) else None
            cid = int(cids[idx[0]]) if all(cids[i] == cids[idx[0]] for i in idx) else None
            sc = nan
            if include_metadata:
                same = True
                for i in idx:
                    if np.isnan(sc):
                        sc = scores[i]
                    elif sc != scores[i]:
                        same = False
                sc = sc if same else nan
            if out["start"] and out["chrom"][-1] == chrom and out["end"][-1] == s \
                    and out["intervalID"][-1] == iid and out["chain_id"][-1] == cid \
                    and (out[value_col][-1] == v or (np.isnan(out[value_col][-1]) and np.isnan(v))) \
                    and (not include_metadata or out["score"][-1] == sc
                         or (np.isnan(out["score"][-1]) and np.isnan(sc))):
                out["end"][-1] = e
                continue
            for c, x in zip(cols, [chrom, s, e, iid, cid] + ([sc] if include_metadata else []) + [v], strict=True):
                out[c].append(x)
    res = pd.DataFrame(out)
    for c in ("intervalID", "chain_id"):
        ids = res[c]
        if drop_ids or ids.isna().all():
            res = res.drop(columns=c)
        else:
            res[c] = ids.astype("int64") if ids.notna().all() else ids.astype("Int64")
    return res


# ===================================================================
# Public API: gtrack_liftover
# ===================================================================

# The multi_target_agg values gtrack_liftover accepts, as R's gtrack.liftover
_TRACK_LIFTOVER_AGGS = (
    "mean", "median", "sum", "min", "max", "count", "first", "last", "nth",
    "max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac",
)


def _nth_param(multi_target_agg: str, params: dict[str, Any] | int | None) -> int:
    """The 1-based n of multi_target_agg="nth" (0 for any other), validated as R's gtrack.liftover.

    *params* is the n itself or a dict holding it as "n" (or as its only value).
    """
    if multi_target_agg not in _TRACK_LIFTOVER_AGGS:
        raise ValueError(
            f"Unsupported aggregation: {multi_target_agg}. "
            f"Supported: {', '.join(_TRACK_LIFTOVER_AGGS)}"
        )
    if multi_target_agg != "nth":
        if params is not None:
            raise ValueError(f"params is only supported for 'nth' aggregation, not '{multi_target_agg}'")
        return 0
    if params is None:
        raise ValueError("params must be supplied for 'nth' aggregation (e.g. params=2 or params={'n': 2})")
    n: Any = params
    if isinstance(params, dict):
        if not params:
            raise ValueError("params list must contain an element 'n' for 'nth'")
        if "n" in params:
            n = params["n"]
        elif len(params) == 1:
            n = next(iter(params.values()))
        else:
            raise ValueError("params must contain a single numeric value (or 'n') for 'nth'")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, float, np.integer, np.floating)) or np.isnan(n):
        raise ValueError("params for 'nth' must be a single positive integer")
    if int(n) <= 0:
        raise ValueError("params for 'nth' must be a positive integer")
    return int(n)


_TRACK_IDX_MAGIC = b"MISHATDX"
_TRACK_IDX_2D_MAGIC = b"MISHT2D\x00"  # track.idx of an indexed 2D track (TrackIndex2D.h)
_TRACK_IDX_VERSION = 1
_TRACK_IDX_FLAG_LITTLE_ENDIAN = 0x01
_TRACK_TYPE_DENSE = 0
_TRACK_TYPE_SPARSE = 1

# 2D quadtree signatures (R misha GenomeTrack.cpp).
_SIGNATURE_RECTS_2D = -9
_SIGNATURE_POINTS_2D = -10


def _refuse_cluster_policy(chain: pd.DataFrame, tgt_overlap_policy: str) -> None:
    """Refuse the cluster target-overlap policies, as R's gtrack.liftover does.

    They are implemented only for gintervals_liftover. The policy is the
    chain's own, or *tgt_overlap_policy* for a chain that carries none.
    """
    policy = chain.attrs.get("tgt_overlap_policy", tgt_overlap_policy)
    if policy in _CLUSTER_TGT_POLICIES:
        raise ValueError(
            f"tgt_overlap_policy = '{policy}' is only supported by gintervals.liftover(), "
            "not gtrack.liftover(). Use 'auto', 'agg', 'keep', or 'discard'."
        )


def _detect_source_track_2d(src_track_dir: str) -> bool:
    """Return True if the source-track directory holds a 2D track.

    Signatures: ``-9`` (RECTS) or ``-10`` (POINTS) in the first 4 bytes of any
    non-index data file. Any 1D signature (dense > 0 or sparse == -1) returns
    False immediately. With no per-pair file, an indexed 2D track is recognized
    by the magic of its ``track.idx``. Directories with neither are treated as
    1D-by-default so empty directories don't accidentally route through the 2D
    path.
    """
    src_track_dir = str(src_track_dir)
    if not os.path.isdir(src_track_dir):
        return False
    for fname in sorted(os.listdir(src_track_dir)):
        if fname.startswith(".") or fname in ("track.idx", "track.dat"):
            continue
        fpath = os.path.join(src_track_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "rb") as f:
            head = f.read(4)
        if len(head) < 4:
            continue
        sig = struct.unpack("<i", head)[0]
        if sig in (_SIGNATURE_RECTS_2D, _SIGNATURE_POINTS_2D):
            return True
        if sig > 0 or sig == -1:
            return False
    idx_path = os.path.join(src_track_dir, "track.idx")
    if os.path.isfile(idx_path):
        with open(idx_path, "rb") as f:
            return f.read(len(_TRACK_IDX_2D_MAGIC)) == _TRACK_IDX_2D_MAGIC
    return False


def _compute_track_idx_checksum(entries: list[tuple[int, int, int, int]]) -> int:
    crc = _crc64_init()
    for chrom_id, offset, length, _reserved in entries:
        crc = _crc64_incremental(crc, struct.pack("<I", chrom_id))
        crc = _crc64_incremental(crc, struct.pack("<Q", offset))
        crc = _crc64_incremental(crc, struct.pack("<Q", length))
    return int(_crc64_finalize(crc))


def _read_track_idx(idx_path: str) -> tuple[int, list[tuple[int, int, int, int]]]:
    with open(idx_path, "rb") as fh:
        if fh.read(8) != _TRACK_IDX_MAGIC:
            raise ValueError(f"Invalid track index header in {idx_path}")
        (version,) = struct.unpack("<I", fh.read(4))
        if version != _TRACK_IDX_VERSION:
            raise ValueError(f"Unsupported track index version {version} in {idx_path}")
        (track_type_raw,) = struct.unpack("<I", fh.read(4))
        (num_contigs,) = struct.unpack("<I", fh.read(4))
        (flags,) = struct.unpack("<Q", fh.read(8))
        if (flags & _TRACK_IDX_FLAG_LITTLE_ENDIAN) == 0:
            raise ValueError(f"Unsupported track index endianness in {idx_path}")
        (stored_checksum,) = struct.unpack("<Q", fh.read(8))

        entries = []
        for _ in range(num_contigs):
            rec = fh.read(24)
            if len(rec) != 24:
                raise ValueError(f"Truncated track index entries in {idx_path}")
            entries.append(struct.unpack("<IQQI", rec))

    checksum = _compute_track_idx_checksum(entries)
    if checksum != stored_checksum:
        raise ValueError(
            f"track.idx checksum mismatch in {idx_path} "
            f"(expected {stored_checksum:016X}, got {checksum:016X})"
        )
    return track_type_raw, entries


def _source_db_root_from_track_dir(src_track_dir: str) -> Path | None:
    """The database that holds the track directory: the parent of the nearest enclosing
    "tracks" directory that has a chrom_sizes.txt next to it, or None."""
    p = Path(src_track_dir).resolve()
    for parent in p.parents:
        if parent.name == "tracks" and (parent.parent / "chrom_sizes.txt").exists():
            return parent.parent
    return None


def _load_source_chrom_names(src_track_dir: str) -> dict[int, str]:
    # An indexed track is keyed by the chrom ids of its own database, as pymisha (and
    # misha) gives them when it opens that database.
    db_root = _source_db_root_from_track_dir(src_track_dir)
    if db_root is None:
        raise ValueError(
            f"Indexed source track {src_track_dir} is not in a database (a directory with "
            'chrom_sizes.txt whose "tracks" directory contains the track); an indexed track '
            "is read by the chromosome ids of its database"
        )

    chrom_sizes_path = db_root / "chrom_sizes.txt"
    if not chrom_sizes_path.exists():
        raise ValueError(
            f"Cannot resolve chromosome IDs for indexed source track: missing {chrom_sizes_path}"
        )
    # as R's .gdb.chrom_names_at: the session's order for a loaded database, and an
    # error for one whose order cannot be read
    try:
        chroms = _shared._gdb_chrom_names_at(str(db_root))
    except _pymisha.error as exc:
        raise ValueError(str(exc)) from exc
    return dict(enumerate(chroms))


def _indexed_source_db_chroms(src_track_dir: str) -> list[str] | None:
    """The chromosome names by chrom id of the database holding an indexed source track
    (``_load_source_chrom_names``), None for a per-chromosome one."""
    if not os.path.exists(os.path.join(src_track_dir, "track.idx")):
        return None
    return list(_load_source_chrom_names(src_track_dir).values())


def _parse_dense_payload(
    payload: bytes,
    chrom_name: str,
    source_label: str,
) -> list[tuple[str, int, int, float]]:
    if len(payload) < 4:
        return []
    if (len(payload) - 4) % 4 != 0:
        raise ValueError(f"Corrupt dense track payload for {chrom_name} in {source_label}")

    (bin_size,) = struct.unpack("<i", payload[:4])
    if bin_size <= 0:
        raise ValueError(f"Invalid dense bin size for {chrom_name} in {source_label}: {bin_size}")

    # every bin, an infinite value as NaN, as R's GenomeTrackFixedBin reads it
    vals = np.frombuffer(payload, dtype="<f4", offset=4).astype(np.float64)
    vals[np.isinf(vals)] = np.nan
    starts = np.arange(vals.size, dtype=np.int64) * int(bin_size)
    ends = starts + int(bin_size)
    chroms = np.full(vals.size, chrom_name, dtype=object)
    return list(zip(chroms.tolist(), starts.tolist(), ends.tolist(), vals.tolist(), strict=False))


def _parse_sparse_payload(
    payload: bytes,
    chrom_name: str,
    source_label: str,
) -> list[tuple[str, int, int, float]]:
    if len(payload) < 4:
        return []

    (sig,) = struct.unpack("<i", payload[:4])
    if sig != -1:
        raise ValueError(f"Invalid sparse signature for {chrom_name} in {source_label}: {sig}")

    body = payload[4:]
    if not body:
        return []

    dtype64 = np.dtype([("start", "<i8"), ("end", "<i8"), ("value", "<f4")])
    dtype32 = np.dtype([("start", "<i4"), ("end", "<i4"), ("value", "<f4")])

    can64 = len(body) % dtype64.itemsize == 0
    can32 = len(body) % dtype32.itemsize == 0
    if not can64 and not can32:
        raise ValueError(
            f"Corrupt sparse track payload length for {chrom_name} in {source_label}: "
            f"{len(body)} bytes"
        )

    def _decode_sparse_recs(dt: np.dtype) -> tuple[np.ndarray, bool]:
        recs = np.frombuffer(body, dtype=dt)
        if recs.size == 0:
            return recs, True
        starts = recs["start"].astype(np.int64, copy=False)
        ends = recs["end"].astype(np.int64, copy=False)
        valid = np.all(starts >= 0) and np.all(ends >= starts)
        return recs, bool(valid)

    recs = None
    if can64 and can32:
        recs64, ok64 = _decode_sparse_recs(dtype64)
        recs32, ok32 = _decode_sparse_recs(dtype32)
        if ok64 and not ok32:
            recs = recs64
        elif ok32 and not ok64:
            recs = recs32
        elif ok64 and ok32:
            # Ambiguous payload shape; prefer modern 64-bit sparse format.
            recs = recs64
        else:
            raise ValueError(
                f"Corrupt sparse track payload records for {chrom_name} in {source_label}"
            )
    elif can64:
        recs, ok = _decode_sparse_recs(dtype64)
        if not ok:
            raise ValueError(
                f"Invalid sparse 64-bit records for {chrom_name} in {source_label}"
            )
    else:
        recs, ok = _decode_sparse_recs(dtype32)
        if not ok:
            raise ValueError(
                f"Invalid sparse 32-bit records for {chrom_name} in {source_label}"
            )

    # every interval, an infinite value as NaN, as R's GenomeTrackSparse reads it
    # typed as arrays: numpy 2.2's stubs (the last for Python 3.10) type tolist() of a field as a scalar
    vals: np.ndarray = recs["value"].astype(np.float64)
    vals[np.isinf(vals)] = np.nan
    starts: np.ndarray = recs["start"].astype(np.int64, copy=False)
    ends: np.ndarray = recs["end"].astype(np.int64, copy=False)
    chroms = np.full(starts.size, chrom_name, dtype=object)
    return list(zip(chroms.tolist(), starts.tolist(), ends.tolist(), vals.tolist(), strict=False))


def _read_indexed_source_track(src_track_dir: str) -> tuple[str, pd.DataFrame]:
    idx_path = os.path.join(src_track_dir, "track.idx")
    dat_path = os.path.join(src_track_dir, "track.dat")
    if not os.path.exists(idx_path) or not os.path.exists(dat_path):
        raise ValueError(f"Indexed source track is missing track.idx/track.dat in {src_track_dir}")

    track_type_raw, entries = _read_track_idx(idx_path)
    if track_type_raw not in (_TRACK_TYPE_DENSE, _TRACK_TYPE_SPARSE):
        raise ValueError(f"Unsupported indexed source track type {track_type_raw} in {src_track_dir}")

    chrom_names = _load_source_chrom_names(src_track_dir)
    rows = []

    with open(dat_path, "rb") as dat_fh:
        for chrom_id, offset, length, _reserved in entries:
            if length == 0:
                continue
            chrom_name = chrom_names.get(chrom_id)
            if chrom_name is None:
                raise ValueError(
                    f"Indexed source track has chrom_id={chrom_id} not present in source chrom_sizes.txt"
                )

            dat_fh.seek(offset)
            payload = dat_fh.read(length)
            if len(payload) != length:
                raise ValueError(
                    f"Failed to read {length} bytes for chrom_id={chrom_id} from {dat_path}"
                )

            if track_type_raw == _TRACK_TYPE_DENSE:
                rows.extend(_parse_dense_payload(payload, chrom_name, "indexed source track"))
            else:
                rows.extend(_parse_sparse_payload(payload, chrom_name, "indexed source track"))

    track_type = "dense" if track_type_raw == _TRACK_TYPE_DENSE else "sparse"
    if not rows:
        return track_type, pd.DataFrame(columns=["chrom", "start", "end", "value"])
    return track_type, pd.DataFrame(rows, columns=["chrom", "start", "end", "value"])


def _detect_source_bin_size(src_track_dir: str) -> int:
    """Detect bin_size from a dense source-track directory.

    Returns the bin_size in bp for dense sources, 0 for sparse/empty.
    Raises ValueError if dense per-chrom files have mismatched bin_sizes
    (matches GTrackLiftover.cpp:528-530 binsize consistency check).
    """
    src_track_dir = str(src_track_dir)
    data_files = [
        f for f in sorted(os.listdir(src_track_dir))
        if not f.startswith(".")
        and f not in ("track.idx", "track.dat")
        and os.path.isfile(os.path.join(src_track_dir, f))
    ]
    prev_bin_size: int | None = None
    prev_file: str | None = None
    for fname in data_files:
        fpath = os.path.join(src_track_dir, fname)
        with open(fpath, "rb") as fh:
            head = fh.read(4)
        if len(head) < 4:
            continue
        sig = struct.unpack("<i", head)[0]
        if sig > 0:
            if prev_bin_size is not None and sig != prev_bin_size:
                raise ValueError(
                    f"Binsize of track file {fname} differs from the "
                    f"binsize of track file {prev_file} ({sig} vs. {prev_bin_size})"
                )
            prev_bin_size = sig
            prev_file = fname
    if prev_bin_size is not None:
        return prev_bin_size

    # No per-chrom dense files; check indexed format.
    # Indexed-format dense tracks store a single track-level type header; bin_size
    # is uniform by file invariant. No cross-chrom consistency check needed (unlike
    # per-chrom files where each file has its own header).
    idx_path = os.path.join(src_track_dir, "track.idx")
    dat_path = os.path.join(src_track_dir, "track.dat")
    if os.path.isfile(idx_path) and os.path.isfile(dat_path):
        track_type_raw, entries = _read_track_idx(idx_path)
        if track_type_raw == _TRACK_TYPE_DENSE:
            for _chrom_id, offset, length, _reserved in entries:
                if length == 0:
                    continue
                with open(dat_path, "rb") as fh:
                    fh.seek(offset)
                    head = fh.read(4)
                if len(head) < 4:
                    continue
                bs = struct.unpack("<i", head)[0]
                if bs > 0:
                    return int(bs)
    return 0


def _aggregate_value_for_bin(
    contribs: list[dict],
    agg_name: str,
    na_rm: bool,
    min_n: int | None,
    nth_index: int,
) -> float:
    """Per-bin reducer matching R's aggregate_values.

    Matches AggregationHelpers.h semantics:
    1. Merge contributions sharing the same chain_id (sum overlap_len).
    2. Apply na_rm / min_n filtering.
    3. Reduce via agg_name.
    """
    # Under na_rm=False, any NaN contribution makes the whole locus NaN (checked
    # on the raw contributions, before the per-chain merge). R 5.11.5.
    if not na_rm:
        for c in contribs:
            if c["is_na"]:
                return float("nan")

    # Step 1: merge contribs sharing chain_id. NaN pieces are dropped FIRST
    # (na_rm is True here) so a chain mapping both a finite and a NaN source bin
    # into this locus keeps its finite value instead of being discarded wholesale
    # by an is_na carried over the whole chain. R 5.11.5.
    merged: list[dict] = []
    for c in contribs:
        if c["is_na"]:
            continue  # na_rm is True here (na_rm=False already returned)
        found = False
        for mc in merged:
            if mc["chain_id"] == c["chain_id"]:
                mc["overlap_len"] += c["overlap_len"]
                mc["coverage_frac"] = mc.get("coverage_frac", 0.0) + c.get("coverage_frac", 0.0)
                mc["start"] = min(mc["start"], c["start"])
                mc["end"] = max(mc["end"], c["end"])
                found = True
                break
        if not found:
            merged.append(dict(c))

    # Step 2: all merged contributions are non-NA.
    valid = merged

    if min_n is not None and min_n >= 0 and len(valid) < min_n:
        return float("nan")

    if agg_name == "count":
        return float(len(valid))

    if not valid:
        return float("nan")

    vals = [c["value"] for c in valid]

    if agg_name == "mean":
        return float(sum(vals) / len(vals))
    if agg_name == "sum":
        return float(sum(vals))
    if agg_name == "min":
        return float(min(vals))
    if agg_name == "max":
        return float(max(vals))
    if agg_name == "median":
        vs = sorted(vals)
        n = len(vs)
        mid = n // 2
        return float((vs[mid - 1] + vs[mid]) / 2.0 if n % 2 == 0 else vs[mid])
    if agg_name in ("first", "last", "nth"):
        # Sort by (start asc, end asc, value desc) matching R ordering.
        sorted_v = sorted(valid, key=lambda c: (c["start"], c["end"], -c["value"]))
        if agg_name == "first":
            return float(sorted_v[0]["value"])
        if agg_name == "last":
            return float(sorted_v[-1]["value"])
        # nth
        if nth_index is None or nth_index <= 0 or nth_index > len(sorted_v):
            return float("nan")
        return float(sorted_v[nth_index - 1]["value"])
    if agg_name in ("max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac"):
        # The contributor covering the most (least) of the locus; a tie goes to the higher value.
        key = "overlap_len" if agg_name.endswith("_len") else "coverage_frac"
        sign = 1.0 if agg_name.startswith("max") else -1.0
        best = valid[0]
        for c in valid[1:]:
            cur, top = sign * c.get(key, 0.0), sign * best.get(key, 0.0)
            if cur > top or (cur == top and c["value"] > best["value"]):
                best = c
        return float(best["value"])
    raise ValueError(f"Unhandled agg_name: {agg_name}")


def _aggregate_per_bin_python(
    intervals_df: pd.DataFrame,
    bin_size: int,
    tgt_chrom_sizes: dict[str, int],
    *,
    agg_name: str,
    na_rm: bool = True,
    min_n: int | None = None,
    nth_index: int = 0,
) -> pd.DataFrame:
    """FIXED_BIN per-bin aggregation matching R GTrackLiftover.cpp:702-768.

    For each target chrom in tgt_chrom_sizes, iterate output bins. Per bin,
    collect (value, overlap_len, chain_id) contributions from every interval
    overlapping [bin_start, bin_end), in R's order (start, end, chain_id,
    value). Merge contributions sharing the same chain_id (sum overlap_len).
    Apply the agg function. Emit one row per bin: a bin with no contributions
    gets what R's aggregate_values gives it (NaN, or 0 for "count" without
    min_n), and a chrom with none at all is NaN throughout, as R writes it.

    intervals_df must have columns chrom, start, end, value, chain_id.
    """
    if agg_name not in _TRACK_LIFTOVER_AGGS:
        raise ValueError(f"Unsupported agg: {agg_name}")
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")

    # Group by chrom for fast lookup.
    by_chrom: dict[str, tuple] = {}
    for chrom, group in intervals_df.groupby("chrom", sort=False):
        g = group.sort_values(["start", "end", "chain_id", "value"], kind="mergesort")
        by_chrom[str(chrom)] = (
            g["start"].to_numpy(dtype=np.int64, copy=False),
            g["end"].to_numpy(dtype=np.int64, copy=False),
            g["value"].to_numpy(dtype=np.float64, copy=False),
            g["chain_id"].to_numpy(dtype=np.int64, copy=False),
        )

    out_rows: list[tuple] = []
    for chrom, chrom_size in tgt_chrom_sizes.items():
        if chrom in by_chrom:
            starts, ends, vals, cids = by_chrom[chrom]
        else:
            starts = ends = vals = cids = np.array([], dtype=np.int64)

        end_bin = (chrom_size + bin_size - 1) // bin_size
        # Cursor advances monotonically past intervals ending at or before the
        # bin start (sorted by start, those cannot overlap this or any later bin);
        # intervals ending later are re-scanned each bin. R 5.11.3: the old
        # `cursor = k` over-advanced past a boundary-spanning contribution whenever
        # an overlapping sibling shared its interval, dropping it from the next bin.
        cursor = 0
        n_iv = len(starts)
        for bin_idx in range(end_bin):
            bs = bin_idx * bin_size
            be = min((bin_idx + 1) * bin_size, chrom_size)

            # Advance the cursor past intervals that end at or before this bin start.
            while cursor < n_iv and int(ends[cursor]) <= bs:
                cursor += 1

            # Collect every contribution overlapping [bs, be). Sorted by start, so
            # stop as soon as an interval starts at or after the bin end.
            contribs: list[dict] = []
            k = cursor
            while k < n_iv:
                s_k = int(starts[k])
                if s_k >= be:
                    break
                e_k = int(ends[k])
                ovl_s = max(bs, s_k)
                ovl_e = min(be, e_k)
                if ovl_s < ovl_e:
                    v = float(vals[k])
                    contribs.append({
                        "value": v,
                        "overlap_len": float(ovl_e - ovl_s),
                        "coverage_frac": float(ovl_e - ovl_s) / float(be - bs),
                        "start": ovl_s,
                        "end": ovl_e,
                        "is_na": bool(np.isnan(v)),
                        "chain_id": int(cids[k]),
                    })
                k += 1

            v_out = (_aggregate_value_for_bin(contribs, agg_name, na_rm, min_n, nth_index)
                     if n_iv else float("nan"))
            out_rows.append((chrom, bs, be, v_out))

    return pd.DataFrame(out_rows, columns=["chrom", "start", "end", "value"])


def _read_source_track(
    src_track_dir: str,
    *,
    _force_pure_python: bool = False,
) -> tuple[str, pd.DataFrame]:
    """Read a source track directory and return (type, intervals_df).

    Dispatches to ``_pymisha.pm_read_source_track_1d`` by default. Falls back
    to the pure-Python implementation when ``_force_pure_python=True`` or when
    the env var ``PYMISHA_FORCE_PY_READ_SOURCE_TRACK=1`` is set. The fallback
    path is exercised by the G1.P3.A cross-validation tests in
    ``tests/test_source_track_cpp.py``.

    Returns a DataFrame with columns: chrom, start, end, value. Source chrom
    names are the raw file names (per-chrom case) or the names from
    ``chrom_sizes.txt`` (indexed case), not normalized to the target DB.
    """
    src_track_dir = str(src_track_dir)
    if not os.path.isdir(src_track_dir):
        raise ValueError(f"Source track directory does not exist: {src_track_dir}")

    use_py = _force_pure_python or os.environ.get(
        "PYMISHA_FORCE_PY_READ_SOURCE_TRACK", ""
    ) == "1"
    if use_py:
        return _read_source_track_python(src_track_dir)

    track_type, df_dict = _pymisha.pm_read_source_track_1d(src_track_dir)
    if len(df_dict["chrom"]) == 0:
        return track_type, pd.DataFrame(columns=["chrom", "start", "end", "value"])
    return track_type, pd.DataFrame({
        "chrom": df_dict["chrom"],
        "start": df_dict["start"],
        "end":   df_dict["end"],
        "value": df_dict["value"],
    })


def _read_source_track_python(src_track_dir: str) -> tuple[str, pd.DataFrame]:
    """Read a source track directory and return (type, intervals_df). (Python reference impl.)

    Returns a DataFrame with columns: chrom, start, end, value.
    Source chrom names are the raw file names (not normalized to target DB).
    For dense tracks, each bin becomes one row. NaN/inf bins are skipped.
    For sparse tracks, each stored interval becomes one row.
    """
    src_track_dir = str(src_track_dir)
    if not os.path.isdir(src_track_dir):
        raise ValueError(f"Source track directory does not exist: {src_track_dir}")

    data_files = [
        fname for fname in sorted(os.listdir(src_track_dir))
        if (
            not fname.startswith(".")
            and os.path.isfile(os.path.join(src_track_dir, fname))
        )
    ]
    per_chrom_files = [f for f in data_files if f not in ("track.idx", "track.dat")]
    has_indexed_files = "track.idx" in data_files and "track.dat" in data_files

    if not per_chrom_files and has_indexed_files:
        return _read_indexed_source_track(src_track_dir)

    rows = []
    track_type = None

    # Scan for track data files (skip hidden files like .attributes)
    for fname in per_chrom_files:
        fpath = os.path.join(src_track_dir, fname)
        if not os.path.isfile(fpath):
            continue

        with open(fpath, "rb") as f:
            payload = f.read()
        if len(payload) < 4:
            continue
        sig = struct.unpack("<i", payload[:4])[0]

        if sig > 0:
            if track_type is None:
                track_type = "dense"
            elif track_type != "dense":
                raise ValueError(f"Mixed dense/sparse source files in {src_track_dir}")
            rows.extend(_parse_dense_payload(payload, fname, "per-chrom source track"))
        elif sig == -1:
            if track_type is None:
                track_type = "sparse"
            elif track_type != "sparse":
                raise ValueError(f"Mixed dense/sparse source files in {src_track_dir}")
            rows.extend(_parse_sparse_payload(payload, fname, "per-chrom source track"))

    if not rows:
        return track_type or "sparse", pd.DataFrame(columns=["chrom", "start", "end", "value"])

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "value"])
    return track_type or "sparse", df


def _agg_key(lifted: pd.DataFrame) -> np.ndarray:
    """R's aggregation key of each lifted piece: (chain id, source interval index).

    Pieces sharing it, one source interval split by the chain, count once;
    different source intervals of one chain stay apart. R 5.11.6.
    """
    cid = lifted["chain_id"].to_numpy(dtype=np.int64)
    iid = lifted["intervalID"].to_numpy(dtype=np.int64)
    key: np.ndarray = (cid << np.int64(32)) ^ iid
    return key


def _aggregate_overlapping(
    intervals_df: pd.DataFrame,
    *,
    agg_name: str,
    na_rm: bool = True,
    min_n: int | None = None,
    nth_index: int = 0,
    _force_pure_python: bool = False,
) -> pd.DataFrame:
    """The intervals of a sparse lifted track, as R's gtrack.liftover writes them.

    *intervals_df* holds the lifted pieces: chrom, start, end, value and key
    (``_agg_key``). Per chromosome, in (start, end, key, value) order, every
    run of overlapping pieces becomes one interval, valued by R's
    aggregate_values over the run (``_aggregate_value_for_bin``): a piece
    counts its whole length, pieces sharing a key count once, and a NaN value
    is kept, so a run of NaN source values gives a NaN interval. Touching
    pieces are not merged.

    Runs through the C++ ``pm_liftover_aggregate``; ``_force_pure_python``
    runs the same steps in Python.
    """
    if not _force_pure_python:
        from _pymisha import pm_liftover_aggregate
        out = pm_liftover_aggregate(
            {
                "chrom": intervals_df["chrom"].astype(str).to_numpy(dtype=object),
                "start": intervals_df["start"].to_numpy(dtype=np.int64),
                "end": intervals_df["end"].to_numpy(dtype=np.int64),
                "value": intervals_df["value"].to_numpy(dtype=np.float64),
                "key": intervals_df["key"].to_numpy(dtype=np.int64),
            },
            agg_name, bool(na_rm), int(-1 if min_n is None else min_n), int(nth_index),
        )
        return pd.DataFrame({c: out[c] for c in ("chrom", "start", "end", "value")})

    df = intervals_df[intervals_df["start"] < intervals_df["end"]]
    df = df.assign(chrom=df["chrom"].astype(str)).sort_values(
        ["chrom", "start", "end", "key", "value"], kind="mergesort")
    rows: list[tuple[str, int, int, float]] = []
    for chrom, group in df.groupby("chrom", sort=True):
        starts = group["start"].to_numpy(dtype=np.int64)
        ends = group["end"].to_numpy(dtype=np.int64)
        vals = group["value"].to_numpy(dtype=np.float64)
        keys = group["key"].to_numpy(dtype=np.int64)
        i, n = 0, len(group)
        while i < n:
            run_start, run_end = int(starts[i]), int(ends[i])
            contribs = []
            while i < n and starts[i] < run_end:
                run_end = max(run_end, int(ends[i]))
                v = float(vals[i])
                contribs.append({
                    "value": v, "overlap_len": float(ends[i] - starts[i]), "coverage_frac": 1.0,
                    "start": int(starts[i]), "end": int(ends[i]), "is_na": math.isnan(v),
                    "chain_id": int(keys[i]),
                })
                i += 1
            rows.append((str(chrom), run_start, run_end,
                         _aggregate_value_for_bin(contribs, agg_name, na_rm, min_n, nth_index)))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "value"])


def _create_empty_lifted_track(
    track: str, description: str, created_by: str, track_type: str, bin_size: int,
) -> None:
    """The track R's gtrack.liftover writes when it lifts no value: for a dense source a
    dense track of NaN in its bins, otherwise a sparse track with no interval (an empty
    file per chromosome, or the indexed track of one in an indexed database)."""
    from .tracks import (
        _apply_create_parallel_writers_from_config,
        _atomic_track_create,
        _df2pymisha,
        _load_track_attributes,
        _save_track_attributes,
        _set_created_attrs,
        _target_root,
        gtrack_create_dense,
    )

    if track_type == "dense" and bin_size > 0:
        chrom = str(gintervals_all()["chrom"].iloc[0])
        gtrack_create_dense(track, description, pd.DataFrame({"chrom": [chrom], "start": [0], "end": [1]}),
                            [np.nan], binsize=bin_size)
        attrs = _load_track_attributes(track)
        attrs["created.by"] = created_by
        _save_track_attributes(track, attrs)
        return

    no_intervals = pd.DataFrame({
        "chrom": pd.Series([], dtype=object), "start": pd.Series([], dtype=np.int64),
        "end": pd.Series([], dtype=np.int64), "value": pd.Series([], dtype=np.float64),
    })
    with _atomic_track_create(track):
        _apply_create_parallel_writers_from_config()
        _pymisha.pm_track_create_sparse(track, _df2pymisha(no_intervals))
    _pm_dbreload(_target_root())
    _set_created_attrs(track, description, created_by)


def gtrack_liftover(
    track: str,
    description: str,
    src_track_dir: str,
    chain: str | pd.DataFrame,
    src_overlap_policy: str = "error",
    tgt_overlap_policy: str = "auto",
    multi_target_agg: str = "mean",
    params: dict[str, Any] | int | None = None,
    na_rm: bool = True,
    min_n: int | None = None,
    min_score: float | None = None,
    *,
    _force_pure_python: bool = False,
) -> None:
    """Import a track from another assembly via coordinate liftover.

    Dispatches to the C++ fast path (G1.P3.C) by default. Falls back to the
    pure-Python implementation when ``_force_pure_python=True`` or when
    ``PYMISHA_FORCE_PY_LIFTOVER_TRACK=1`` in the environment.

    See :func:`_gtrack_liftover_python` for the full parameter docstring.
    """
    # 2D source tracks route through the dedicated 2D path. Detection is by
    # quadtree file signature (R-parity: GTrackLiftover.cpp:843 dispatches on
    # GenomeTrack::RECTS / POINTS). multi_target_agg / na_rm / min_n / nth apply
    # to the 2D path too: overlapping mapped rectangles are aggregated into
    # disjoint cells before insertion (R 5.11.8).
    if _detect_source_track_2d(src_track_dir):
        return _gtrack_liftover_2d(
            track, description, src_track_dir, chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            multi_target_agg=multi_target_agg,
            params=params, na_rm=na_rm, min_n=min_n,
            min_score=min_score,
        )

    use_py = _force_pure_python or os.environ.get(
        "PYMISHA_FORCE_PY_LIFTOVER_TRACK", ""
    ) == "1"
    if use_py:
        return _gtrack_liftover_python(
            track, description, src_track_dir, chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            multi_target_agg=multi_target_agg,
            params=params, na_rm=na_rm, min_n=min_n,
            min_score=min_score,
        )

    # C++ path: pre-validate, build chain dict + tgt_chrom_sizes, call C++.
    from .tracks import (
        _load_track_attributes,
        _save_track_attributes,
        _track_exists,
        _validate_track_name,
        gtrack_create_dense,
        gtrack_create_sparse,
    )

    _checkroot()
    _validate_track_name(track)
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")
    nth_index = _nth_param(multi_target_agg, params)

    if isinstance(chain, str):
        chain = gintervals_load_chain(
            chain, src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            min_score=min_score,
        )
    elif not isinstance(chain, pd.DataFrame):
        raise TypeError("chain must be a file path string or a chain DataFrame")
    else:
        chain = _with_canonical_chroms(chain)   # an alias names the database's chromosome
    chain = _chain_from_r(chain)
    _refuse_cluster_policy(chain, tgt_overlap_policy)

    chain_dict = {
        "chrom":     chain["chrom"].to_numpy(dtype=object),
        "start":     chain["start"].to_numpy(dtype=np.int64),
        "end":       chain["end"].to_numpy(dtype=np.int64),
        "strand":    chain["strand"].to_numpy(dtype=np.int64),
        "chromsrc":  chain["chromsrc"].to_numpy(dtype=object),
        "startsrc":  chain["startsrc"].to_numpy(dtype=np.int64),
        "endsrc":    chain["endsrc"].to_numpy(dtype=np.int64),
        "strandsrc": chain["strandsrc"].to_numpy(dtype=np.int64),
        "chain_id":  chain["chain_id"].to_numpy(dtype=np.int64),
        "score":     chain["score"].to_numpy(dtype=np.float64),
    }
    tgt_chrom_sizes = _get_db_chrom_sizes()
    result = _pymisha.pm_liftover_track(
        str(src_track_dir), chain_dict, tgt_chrom_sizes,
        "", multi_target_agg, bool(na_rm),
        int(-1 if min_n is None else min_n),
        int(nth_index),
        _indexed_source_db_chroms(str(src_track_dir)),
    )

    created_by = f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)'
    track_type = result["track_type"]
    bin_size = int(result["bin_size"])
    if len(result["chrom"]) == 0:
        _create_empty_lifted_track(track, description, created_by, track_type, bin_size)
        return None

    target_df = pd.DataFrame({
        "chrom": result["chrom"],
        "start": result["start"],
        "end":   result["end"],
        "value": result["value"],
    })
    if track_type == "dense":
        # FIXED_BIN preservation: aggregate_per_bin_cpp pre-aggregated to one
        # row per bin per target chrom. Filter NaN bins (gtrack_create_dense
        # fills empty bins via defval=NaN). func="weighted.mean" of a single-
        # contribution bin returns that value as-is.
        target_df = target_df[~target_df["value"].isna()].reset_index(drop=True)
        if len(target_df) == 0:
            _create_empty_lifted_track(track, description, created_by, track_type, bin_size)
            return None
        gtrack_create_dense(
            track, description,
            target_df[["chrom", "start", "end"]],
            target_df["value"].to_numpy(),
            binsize=bin_size,
            func="weighted.mean",
        )
    else:
        gtrack_create_sparse(
            track, description,
            target_df[["chrom", "start", "end"]],
            target_df["value"].to_numpy(),
        )
    attrs = _load_track_attributes(track)
    attrs["created.by"] = created_by
    _save_track_attributes(track, attrs)
    return None


def _gtrack_liftover_python(
    track: str,
    description: str,
    src_track_dir: str,
    chain: str | pd.DataFrame,
    src_overlap_policy: str = "error",
    tgt_overlap_policy: str = "auto",
    multi_target_agg: str = "mean",
    params: dict[str, Any] | int | None = None,
    na_rm: bool = True,
    min_n: int | None = None,
    min_score: float | None = None,
) -> None:
    """Import a track from another assembly via coordinate liftover.

    Reads a source track from *src_track_dir* (a directory containing
    per-chromosome binary track files or an indexed ``track.idx``/``track.dat``
    pair), maps its intervals through *chain* to the current target genome,
    aggregates values when multiple source intervals land on the same target
    region, and creates a new track (sparse or dense, matching the source
    track type) in the current database.

    When *chain* is a file path it is loaded with the specified overlap
    policies. When it is a pre-loaded DataFrame the policies stored in its
    attributes are used and the policy arguments here are ignored.

    Parameters
    ----------
    track : str
        Name of the new track to create in the current database. The track
        must not already exist.
    description : str
        Human-readable description stored as a track attribute.
    src_track_dir : str
        Path to the source track directory. The directory may contain
        per-chromosome binary files (dense or sparse) or an indexed pair of
        ``track.idx`` and ``track.dat`` files.
    chain : str or pandas.DataFrame
        Either a path to a UCSC chain file or a pre-loaded chain DataFrame
        as returned by ``gintervals_load_chain``.
    src_overlap_policy : str, optional
        Source overlap policy, used only when *chain* is a file path.
        One of ``"error"`` (default), ``"keep"``, or ``"discard"``.
    tgt_overlap_policy : str, optional
        Target overlap policy, used only when *chain* is a file path.
        One of ``"error"``, ``"auto"`` (default), ``"auto_score"``,
        ``"auto_longer"``, ``"auto_first"``, ``"keep"``, ``"discard"``,
        ``"agg"``. The cluster policies (``"best_source_cluster"``,
        ``"best_cluster_union"``, ``"best_cluster_sum"``,
        ``"best_cluster_max"``) are refused, here or in a pre-loaded chain,
        as in R misha: they are for ``gintervals_liftover`` only.
    multi_target_agg : str, optional
        Aggregation function applied when multiple source values map to
        the same target locus. One of ``"mean"`` (default), ``"median"``,
        ``"sum"``, ``"min"``, ``"max"``, ``"count"``, ``"first"``,
        ``"last"``, ``"nth"``, ``"max.coverage_len"``,
        ``"min.coverage_len"``, ``"max.coverage_frac"``,
        ``"min.coverage_frac"``. The coverage aggregations take the value
        of the contributor that covers the most (least) of the locus, by
        length or by fraction of the locus; a tie goes to the higher value.
    params : int or dict, optional
        The ``n`` of ``"nth"`` aggregation (1-based), given as is or as
        ``{"n": n}``. Required for ``"nth"`` and not allowed otherwise.
    na_rm : bool, optional
        If ``True`` (default), ``NaN`` values are removed before
        aggregation. If ``False``, any ``NaN`` in the group causes the
        aggregated result to be ``NaN``.
    min_n : int, optional
        Minimum number of non-``NaN`` values required for aggregation. If
        fewer values are available the result is ``NaN``. ``None`` (default)
        means no minimum.
    min_score : float, optional
        Minimum chain alignment score. Chains scoring below this value are
        excluded during loading. Only used when *chain* is a file path.

    Returns
    -------
    None
        The function creates a new track (sparse or dense, matching the
        source track type) in the current database as a side effect and
        does not return a value.

    Raises
    ------
    ValueError
        If *track* already exists, if *src_track_dir* does not exist, if the
        aggregation function is unsupported, or if the chain file is invalid.
    TypeError
        If *chain* is neither a file path string nor a ``pandas.DataFrame``.

    See Also
    --------
    gintervals_load_chain : Load a chain from a UCSC chain file.
    gintervals_liftover : Lift intervals (without creating a track).
    gtrack_create_sparse : Create a sparse track from intervals and values.

    Notes
    -----
    UCSC chain format terminology is reversed from misha convention: UCSC
    "target" (``tName``, ``tStart``, ``tEnd``) corresponds to misha "source"
    (``chromsrc``, ``startsrc``, ``endsrc``), and UCSC "query" corresponds to
    misha "target" (``chrom``, ``start``, ``end``).

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> import os
    >>> chainfile = os.path.join(pm._GROOT, "data", "test.chain")
    >>> pm.gtrack_liftover(  # doctest: +SKIP
    ...     "lifted_track", "Track lifted from other assembly",
    ...     "/path/to/source/tracks/my_track.track", chainfile,
    ... )
    """
    from .tracks import (
        _checkroot,
        _load_track_attributes,
        _save_track_attributes,
        _track_exists,
        _validate_track_name,
        gtrack_create_dense,
        gtrack_create_sparse,
    )

    _checkroot()
    _validate_track_name(track)
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")

    # Validate aggregation function
    nth_index = _nth_param(multi_target_agg, params)

    # Load chain if path
    if isinstance(chain, str):
        chain = gintervals_load_chain(
            chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            min_score=min_score,
        )
    elif isinstance(chain, pd.DataFrame):
        # Pre-loaded chain — policies are already baked in. An alias names the
        # database's chromosome.
        chain = _with_canonical_chroms(chain)
    else:
        raise TypeError("chain must be a file path string or a chain DataFrame")
    chain = _chain_from_r(chain)
    _refuse_cluster_policy(chain, tgt_overlap_policy)

    # Read source track
    src_type, src_data = _read_source_track(src_track_dir)

    created_by = f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)'

    if len(src_data) == 0 or len(chain) == 0:
        _create_empty_lifted_track(track, description, created_by, src_type,
                                   _detect_source_bin_size(src_track_dir) if src_type == "dense" else 0)
        return

    # An indexed source is read by its database's chromosomes, which match a chain
    # chromosome by name, as is, then without or with the "chr" prefix; the files of a
    # per-chromosome source are found by the chain's names, as is or "chr" toggled
    # (R's gtrack.liftover).
    chain_names = set(chain["chromsrc"].astype(str))
    renamed = {}
    for c in src_data["chrom"].astype(str).unique():
        alt = c[3:] if c.startswith("chr") else f"chr{c}"
        renamed[c] = alt if c not in chain_names and alt in chain_names else c
    src_data = src_data.assign(chrom=src_data["chrom"].astype(str).map(renamed))

    # Liftover source intervals to target coordinates: the pieces R's
    # map_interval gives, not merged (R does not merge touching pieces)
    # No value_col: that would aggregate overlapping pieces, as gintervals.liftover
    # does; the track aggregation below needs every piece. Values follow by
    # intervalID, the 0-based row of src_data.
    # chain is in the internal strand encoding here; gintervals_liftover takes R's.
    # Its intervalID is 1-based; keep the 0-based row of src_data from here on.
    lifted = gintervals_liftover(src_data[["chrom", "start", "end"]].copy(), _chain_to_r(chain))
    if "intervalID" in lifted.columns:
        lifted["intervalID"] = lifted["intervalID"] - 1
    if "value" in src_data.columns and "intervalID" in lifted.columns:
        lifted = lifted.assign(
            value=src_data["value"].to_numpy()[lifted["intervalID"].to_numpy(np.int64)])

    if len(lifted) == 0:
        _create_empty_lifted_track(track, description, created_by, src_type,
                                   _detect_source_bin_size(src_track_dir) if src_type == "dense" else 0)
        return

    if src_type == "dense":
        # R-parity: FIXED_BIN source -> FIXED_BIN target (GTrackLiftover.cpp:702-768).
        _bs = _detect_source_bin_size(src_track_dir)
        # Fallback: dense type but no readable bin_size -> treat as sparse.
        bin_size: int | None = _bs if _bs > 0 else None

        if bin_size is not None:
            tgt_chrom_sizes = _get_db_chrom_sizes()
            dense_in = lifted[["chrom", "start", "end", "value"]].assign(chain_id=_agg_key(lifted))
            per_bin = _aggregate_per_bin_python(
                dense_in,
                bin_size,
                tgt_chrom_sizes,
                agg_name=multi_target_agg,
                na_rm=na_rm,
                min_n=min_n,
                nth_index=nth_index,
            )
            # Filter NaN bins; gtrack_create_dense fills uncovered bins with defval=NaN.
            per_bin_nonnan = per_bin[~per_bin["value"].isna()].reset_index(drop=True)
            if len(per_bin_nonnan) == 0:
                _create_empty_lifted_track(track, description, created_by, "dense", bin_size)
                return
            # Each interval is exactly 1 bin wide and we pre-aggregated via the
            # per-bin helper; func="weighted.mean" of a single-contribution bin
            # returns that pre-aggregated value as-is. (Any single-row reduction
            # would also work; "weighted.mean" is clearest in intent.)
            gtrack_create_dense(
                track, description,
                per_bin_nonnan[["chrom", "start", "end"]],
                per_bin_nonnan["value"].to_numpy(),
                binsize=bin_size,
                func="weighted.mean",
            )
            attrs = _load_track_attributes(track)
            attrs["created.by"] = created_by
            _save_track_attributes(track, attrs)
            return

    # SPARSE path (also used as fallback when dense bin_size cannot be detected).
    target_data = _aggregate_overlapping(
        lifted[["chrom", "start", "end", "value"]].assign(key=_agg_key(lifted)),
        agg_name=multi_target_agg, na_rm=na_rm, min_n=min_n, nth_index=nth_index,
        _force_pure_python=True,
    )

    if len(target_data) == 0:
        _create_empty_lifted_track(track, description, created_by, "sparse", 0)
        return

    gtrack_create_sparse(
        track, description,
        target_data[["chrom", "start", "end"]],
        target_data["value"].to_numpy(),
    )

    # Update the created.by attribute to reflect liftover (bypass readonly check).
    attrs = _load_track_attributes(track)
    attrs["created.by"] = created_by
    _save_track_attributes(track, attrs)


# ===================================================================
# 2D source-track liftover (G1.P3.D)
# ===================================================================

def _gtrack_liftover_2d(
    track: str,
    description: str,
    src_track_dir: str,
    chain: str | pd.DataFrame,
    *,
    src_overlap_policy: str = "error",
    tgt_overlap_policy: str = "auto",
    multi_target_agg: str = "mean",
    params: dict[str, Any] | int | None = None,
    na_rm: bool = True,
    min_n: int | None = None,
    min_score: float | None = None,
) -> None:
    """Lift a 2D source track (RECTS or POINTS) to the current target DB.

    Routed to from ``gtrack_liftover`` whenever the source-track directory
    holds a 2D track. The lift streams through C++ (``pm_liftover_track_2d``),
    as R misha 5.12.1's: both ends of every source object are mapped into a
    buffer file per target chromosome pair, and the pairs are then written one
    at a time. A POINTS source gives a POINTS track, points landing on one
    target point merged by ``multi_target_agg``, and a pair of at least
    4 x ``CONFIG['max_data_size']`` points is written in parts (subtrees);
    overlapping mapped rectangles are cut into disjoint cells aggregated by
    ``multi_target_agg``.
    """
    from .tracks import (
        _atomic_track_create,
        _db_is_indexed,
        _set_created_attrs,
        _target_root,
        _track_exists,
        _validate_track_name,
        gtrack_2d_convert_to_indexed,
        gtrack_create_empty_indexed,
    )

    _checkroot()
    _validate_track_name(track)
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")
    nth_index = _nth_param(multi_target_agg, params)

    if isinstance(chain, str):
        chain = gintervals_load_chain(
            chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            min_score=min_score,
        )
    elif not isinstance(chain, pd.DataFrame):
        raise TypeError("chain must be a file path string or a chain DataFrame")
    else:
        chain = _with_canonical_chroms(chain)   # an alias names the database's chromosome
    chain = _chain_from_r(chain)
    _refuse_cluster_policy(chain, tgt_overlap_policy)

    effective_tgt_policy = chain.attrs.get("tgt_overlap_policy", tgt_overlap_policy)
    chain_dict = {
        "chrom":     chain["chrom"].astype(str).to_numpy(dtype=object),
        "start":     chain["start"].to_numpy(dtype=np.int64),
        "end":       chain["end"].to_numpy(dtype=np.int64),
        "strand":    chain["strand"].to_numpy(dtype=np.int64),
        "chromsrc":  chain["chromsrc"].astype(str).to_numpy(dtype=object),
        "startsrc":  chain["startsrc"].to_numpy(dtype=np.int64),
        "endsrc":    chain["endsrc"].to_numpy(dtype=np.int64),
        "chain_id":  chain["chain_id"].to_numpy(dtype=np.int64),
        "score":     chain["score"].to_numpy(dtype=np.float64),
    }
    tgt_chrom_sizes = _get_db_chrom_sizes()
    src_db_chroms = _indexed_source_db_chroms(str(src_track_dir))

    with _atomic_track_create(track) as track_dir:
        with _shared._with_umask():
            track_dir.mkdir(parents=True, exist_ok=True)
        _pymisha.pm_liftover_track_2d(
            str(src_track_dir), chain_dict, str(track_dir),
            list(tgt_chrom_sizes.keys()), list(tgt_chrom_sizes.values()),
            str(effective_tgt_policy), multi_target_agg, bool(na_rm),
            int(-1 if min_n is None else min_n), int(nth_index),
            src_db_chroms,
        )
        has_pairs = any(not name.startswith(".") for name in os.listdir(track_dir))

    _pm_dbreload(_target_root())
    _set_created_attrs(
        track, description,
        f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)',
    )
    # In an indexed database the track is converted to indexed, as R does. A lift that
    # maps nothing leaves no pair files: R writes the empty (1D) indexed track then.
    if _db_is_indexed(_shared._GROOT):
        if has_pairs:
            gtrack_2d_convert_to_indexed(track, remove_old=True, force=False)
        else:
            gtrack_create_empty_indexed(track)
