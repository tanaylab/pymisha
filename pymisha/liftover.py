"""Liftover chain loading and interval/track coordinate conversion.

Implements gintervals_load_chain, gintervals_as_chain, gintervals_liftover,
and gtrack_liftover with parity to R misha.

UCSC terminology note: In UCSC chain format, 't' fields (tName, tStart, tEnd)
are "target/reference" and 'q' fields are "query". Misha reverses this:
UCSC target = misha source (chromsrc), UCSC query = misha target (chrom).

Strand note: intervals APIs use {-1,0,1}, while chain-derived columns
(`strand`, `strandsrc`) use {0,1} where 0='+' and 1='-'.
"""

import heapq
import os
import struct
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ._crc64 import (
    crc64_finalize as _crc64_finalize,
)
from ._crc64 import (
    crc64_incremental as _crc64_incremental,
)
from ._crc64 import (
    crc64_init as _crc64_init,
)
from ._shared import _checkroot, _pymisha
from .intervals import gintervals_all

# ---------------------------------------------------------------------------
# Overlap policy constants
# ---------------------------------------------------------------------------

_SRC_POLICIES = {"error", "keep", "discard"}
_TGT_POLICIES = {
    "error", "auto", "auto_first", "auto_longer", "auto_score",
    "discard", "keep", "agg",
    "best_source_cluster", "best_cluster_union",
    "best_cluster_sum", "best_cluster_max",
}

_EMPTY_CHAIN_COLS = [
    "chrom", "start", "end", "strand",
    "chromsrc", "startsrc", "endsrc", "strandsrc",
    "chain_id", "score",
]


def _empty_chain_df():
    return pd.DataFrame({
        c: pd.Series(
            dtype="object" if c in ("chrom", "chromsrc") else "int64" if c in (
                "start", "end", "startsrc", "endsrc", "strand", "strandsrc", "chain_id"
            ) else "float64"
        ) for c in _EMPTY_CHAIN_COLS
    })


def _normalize_chrom(name):
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
    return normalized[0]


def _get_db_chrom_sizes():
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

def _parse_chain_file(path, db_chrom_sizes, min_score=None):
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

    blocks = []
    src_chrom_sizes = {}  # track source chrom sizes for consistency validation

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
                        f"Chain file {path}, line {lineno}: expected 13 fields "
                        f"in chain header, got {len(parts)}"
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
                        f"Chain file {path}, line {lineno}: invalid source chrom size"
                    )

                # Validate source chrom size consistency
                if src_chrom in src_chrom_sizes:
                    if src_chrom_sizes[src_chrom] != src_size:
                        raise ValueError(
                            f"Chain file {path}, line {lineno}: source chrom size "
                            f"({src_size}) differs from previous ({src_chrom_sizes[src_chrom]})"
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
                        f"Chain file {path}, line {lineno}: invalid source strand '{src_strand_str}'"
                    )

                src_start = int(parts[5])
                src_end = int(parts[6])
                if src_start < 0 or src_start >= src_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: source start out of range"
                    )
                if src_end <= src_start or src_end > src_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: source end out of range"
                    )

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
                        f"Chain file {path}, line {lineno}: target chrom size "
                        f"({tgt_size}) differs from database ({db_size})"
                    )

                tgt_strand_str = parts[9]
                if tgt_strand_str == "+":
                    tgt_strand = 0
                elif tgt_strand_str == "-":
                    tgt_strand = 1
                else:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: invalid target strand '{tgt_strand_str}'"
                    )

                tgt_start_raw = int(parts[10])
                tgt_end_raw = int(parts[11])
                if tgt_start_raw < 0 or tgt_start_raw >= tgt_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: target start out of range"
                    )
                if tgt_end_raw <= tgt_start_raw or tgt_end_raw > tgt_size:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: target end out of range"
                    )
                tgt_start = tgt_start_raw

                chain_id = int(parts[12])
                cur_src_pos = src_start
                cur_tgt_pos = tgt_start
                in_chain = True
                continue

            # Alignment block line (1 or 3 fields)
            if not in_chain:
                raise ValueError(
                    f"Chain file {path}, line {lineno}: alignment block outside chain"
                )

            if len(parts) not in (1, 3):
                raise ValueError(
                    f"Chain file {path}, line {lineno}: expected 1 or 3 fields "
                    f"in block line, got {len(parts)}"
                )

            if skip_chain:
                # Chain explicitly skipped (e.g. low score): ignore blocks.
                continue

            size = int(parts[0])
            if size <= 0:
                raise ValueError(
                    f"Chain file {path}, line {lineno}: invalid block size"
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

            blocks.append({
                "chrom": tgt_chrom,
                "start": block_tgt_start,
                "end": block_tgt_end,
                "strand": tgt_strand,
                "chromsrc": src_chrom,
                "startsrc": block_src_start,
                "endsrc": block_src_end,
                "strandsrc": src_strand,
                "chain_id": chain_id,
                "score": chain_score,
            })

            # Advance positions
            if len(parts) == 3:
                dt = int(parts[1])
                dq = int(parts[2])
                if dt < 0 or dq < 0:
                    raise ValueError(
                        f"Chain file {path}, line {lineno}: negative gap values"
                    )
                cur_src_pos += size + dt
                cur_tgt_pos += size + dq
            else:
                cur_src_pos += size
                cur_tgt_pos += size

    return blocks


# ===================================================================
# Overlap handling
# ===================================================================

def _handle_src_overlaps(df, policy):
    """Handle source-side overlaps according to policy."""
    if df.empty or policy == "keep":
        return df

    # Sort by source coordinates
    df = df.sort_values(["chromsrc", "startsrc", "endsrc"]).reset_index(drop=True)

    if policy == "error":
        for i in range(1, len(df)):
            if (df.loc[i, "chromsrc"] == df.loc[i - 1, "chromsrc"] and
                    df.loc[i, "startsrc"] < df.loc[i - 1, "endsrc"]):
                raise ValueError(
                    f"Source overlap detected on {df.loc[i, 'chromsrc']}: "
                    f"[{df.loc[i-1, 'startsrc']}, {df.loc[i-1, 'endsrc']}) overlaps "
                    f"[{df.loc[i, 'startsrc']}, {df.loc[i, 'endsrc']})"
                )
        return df

    if policy == "discard":
        return _discard_overlapping_intervals(df, "chromsrc", "startsrc", "endsrc")

    raise ValueError(f"Unknown src_overlap_policy: {policy}")


def _handle_tgt_overlaps(df, policy):
    """Handle target-side overlaps according to policy."""
    if df.empty or policy == "keep":
        return df

    # Sort by target coordinates
    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

    if policy == "error":
        for i in range(1, len(df)):
            if (df.loc[i, "chrom"] == df.loc[i - 1, "chrom"] and
                    df.loc[i, "start"] < df.loc[i - 1, "end"]):
                raise ValueError(
                    f"Target overlap detected on {df.loc[i, 'chrom']}: "
                    f"[{df.loc[i-1, 'start']}, {df.loc[i-1, 'end']}) overlaps "
                    f"[{df.loc[i, 'start']}, {df.loc[i, 'end']})"
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


def _discard_overlapping_intervals(df, chrom_col, start_col, end_col):
    """Drop all intervals that overlap any other interval on the same chrom."""
    n = len(df)
    if n < 2:
        return df

    discard_mask = np.zeros(n, dtype=bool)
    cluster_start = 0
    cluster_end = int(df.loc[0, end_col])
    cluster_has_overlap = False

    for i in range(1, n):
        cur_end = int(df.loc[i, end_col])
        if df.loc[i, chrom_col] != df.loc[i - 1, chrom_col]:
            if cluster_has_overlap:
                discard_mask[cluster_start:i] = True
            cluster_start = i
            cluster_end = cur_end
            cluster_has_overlap = False
            continue

        cur_start = int(df.loc[i, start_col])
        if cur_start < cluster_end:
            cluster_has_overlap = True
            cluster_end = max(cluster_end, cur_end)
        else:
            if cluster_has_overlap:
                discard_mask[cluster_start:i] = True
            cluster_start = i
            cluster_end = cur_end
            cluster_has_overlap = False

    if cluster_has_overlap:
        discard_mask[cluster_start:n] = True

    if discard_mask.any():
        return df.loc[~discard_mask].reset_index(drop=True)
    return df


def _handle_tgt_overlaps_auto(df, policy):
    """Segment overlapping target intervals and select winner per segment.

    auto_score: highest score wins (tiebreak: longer span, lower chain_id)
    auto_first: lowest chain_id wins
    auto_longer: longest span wins (tiebreak: higher score, lower chain_id)
    """
    if df.empty:
        return df

    result_rows = []
    # Process per chromosome
    for chrom, group in df.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"]).reset_index(drop=True)
        if len(group) == 0:
            continue

        starts = group["start"].to_numpy(dtype=np.int64, copy=False)
        ends = group["end"].to_numpy(dtype=np.int64, copy=False)
        strands = group["strand"].to_numpy(dtype=np.int64, copy=False)
        src_starts = group["startsrc"].to_numpy(dtype=np.int64, copy=False)
        src_ends = group["endsrc"].to_numpy(dtype=np.int64, copy=False)
        chain_ids = group["chain_id"].to_numpy(dtype=np.int64, copy=False)
        scores = group["score"].to_numpy(dtype=float, copy=False)
        chromsrc_vals = group["chromsrc"].to_numpy(copy=False)
        strandsrc_vals = group["strandsrc"].to_numpy(dtype=np.int64, copy=False)
        spans = ends - starts

        points = np.unique(np.concatenate((starts, ends)))
        if points.size < 2:
            result_rows.extend(group.to_dict("records"))
            continue

        starts_at = defaultdict(list)
        ends_at = defaultdict(list)
        for idx in range(len(group)):
            starts_at[int(starts[idx])].append(idx)
            ends_at[int(ends[idx])].append(idx)

        if policy == "auto_score":
            def prio(idx, scores=scores, spans=spans, chain_ids=chain_ids):
                return (-float(scores[idx]), -int(spans[idx]), int(chain_ids[idx]), int(idx))
        elif policy == "auto_first":
            def prio(idx, chain_ids=chain_ids):
                return (int(chain_ids[idx]), int(idx))
        else:  # auto_longer
            def prio(idx, spans=spans, scores=scores, chain_ids=chain_ids):
                return (-int(spans[idx]), -float(scores[idx]), int(chain_ids[idx]), int(idx))

        active = set()
        heap = []
        segments = []

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

            winner_idx = heap[0][1]
            orig_tgt_start = int(starts[winner_idx])
            orig_tgt_end = int(ends[winner_idx])
            orig_src_start = int(src_starts[winner_idx])
            orig_src_end = int(src_ends[winner_idx])
            orig_tgt_len = orig_tgt_end - orig_tgt_start

            if orig_tgt_len > 0:
                if int(strands[winner_idx]) == 0:
                    src_start = orig_src_start + (coord - orig_tgt_start)
                    src_end = orig_src_start + (next_coord - orig_tgt_start)
                else:
                    src_start = orig_src_end - (next_coord - orig_tgt_start)
                    src_end = orig_src_end - (coord - orig_tgt_start)
            else:
                src_start = orig_src_start
                src_end = orig_src_end

            segments.append({
                "chrom": chrom,
                "start": coord,
                "end": next_coord,
                "strand": int(strands[winner_idx]),
                "chromsrc": chromsrc_vals[winner_idx],
                "startsrc": src_start,
                "endsrc": src_end,
                "strandsrc": int(strandsrc_vals[winner_idx]),
                "chain_id": int(chain_ids[winner_idx]),
                "score": float(scores[winner_idx]),
            })

        # Merge adjacent segments from the same chain.
        merged = []
        for seg in segments:
            if (
                merged
                and merged[-1]["chain_id"] == seg["chain_id"]
                and merged[-1]["chrom"] == seg["chrom"]
                and merged[-1]["end"] == seg["start"]
            ):
                merged[-1]["end"] = seg["end"]
                merged[-1]["startsrc"] = min(merged[-1]["startsrc"], seg["startsrc"])
                merged[-1]["endsrc"] = max(merged[-1]["endsrc"], seg["endsrc"])
            else:
                merged.append(seg)
        result_rows.extend(merged)

    if not result_rows:
        return _empty_chain_df()
    return pd.DataFrame(result_rows)[_EMPTY_CHAIN_COLS].reset_index(drop=True)


def _handle_tgt_overlaps_agg(df):
    """Segment overlapping target regions, keeping all chains per segment."""
    if df.empty:
        return df

    result_rows = []
    for chrom, group in df.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"]).reset_index(drop=True)

        breakpoints = set()
        for row in group.itertuples(index=False):
            breakpoints.add(row.start)
            breakpoints.add(row.end)
        breakpoints = sorted(breakpoints)

        for i in range(len(breakpoints) - 1):
            seg_start = breakpoints[i]
            seg_end = breakpoints[i + 1]

            for row in group.itertuples(index=False):
                if row.start < seg_end and row.end > seg_start:
                    orig_tgt_start = row.start
                    orig_src_start = row.startsrc
                    orig_src_end = row.endsrc

                    if row.strand == 0:
                        src_start = orig_src_start + (seg_start - orig_tgt_start)
                        src_end = orig_src_start + (seg_end - orig_tgt_start)
                    else:
                        src_start = orig_src_end - (seg_end - orig_tgt_start)
                        src_end = orig_src_end - (seg_start - orig_tgt_start)

                    result_rows.append({
                        "chrom": chrom,
                        "start": seg_start,
                        "end": seg_end,
                        "strand": row.strand,
                        "chromsrc": row.chromsrc,
                        "startsrc": src_start,
                        "endsrc": src_end,
                        "strandsrc": row.strandsrc,
                        "chain_id": row.chain_id,
                        "score": row.score,
                    })

    if not result_rows:
        return _empty_chain_df()
    return pd.DataFrame(result_rows)[_EMPTY_CHAIN_COLS].reset_index(drop=True)


# ===================================================================
# Public API: gintervals_load_chain
# ===================================================================

def gintervals_load_chain(file, src_overlap_policy="error",
                          tgt_overlap_policy="auto", src_groot=None,
                          min_score=None):
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
        Path to the source genome database root for validating source
        chromosomes and coordinates. Not yet implemented.
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

    chain = _empty_chain_df() if not blocks else pd.DataFrame(blocks)[_EMPTY_CHAIN_COLS]

    # Handle overlaps
    chain = _handle_src_overlaps(chain, src_overlap_policy)
    chain = _handle_tgt_overlaps(chain, effective_tgt_policy)

    # Store policies as DataFrame attrs
    chain.attrs["src_overlap_policy"] = src_overlap_policy
    chain.attrs["tgt_overlap_policy"] = tgt_overlap_policy
    if min_score is not None:
        chain.attrs["min_score"] = min_score

    return chain


# ===================================================================
# Public API: gintervals_as_chain
# ===================================================================

def gintervals_as_chain(intervals, src_overlap_policy="error",
                        tgt_overlap_policy="auto", min_score=None):
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
    ...     "chrom": ["1"], "start": [1000], "end": [2000], "strand": [0],
    ...     "chromsrc": ["chr25"], "startsrc": [5000], "endsrc": [6000],
    ...     "strandsrc": [0], "chain_id": [1], "score": [1000.0],
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

    if tgt_overlap_policy == "auto":
        tgt_overlap_policy = "auto_score"

    result = intervals.copy()
    result.attrs["src_overlap_policy"] = src_overlap_policy
    result.attrs["tgt_overlap_policy"] = tgt_overlap_policy
    if min_score is not None:
        result.attrs["min_score"] = min_score

    return result


# ===================================================================
# Public API: gintervals_liftover
# ===================================================================

def gintervals_liftover(intervals, chain, src_overlap_policy="error",
                        tgt_overlap_policy="auto", min_score=None,
                        include_metadata=False, canonic=False,
                        value_col=None, multi_target_agg="mean",
                        params=None, na_rm=True, min_n=None):
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

    # Load chain if file path
    if isinstance(chain, str):
        chain_df = gintervals_load_chain(
            chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            min_score=min_score,
        )
    else:
        chain_df = chain

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

    # Build source-side index for efficient lookup
    # Group chain blocks by source chrom
    chain_by_src_chrom = {}
    for _idx, row in chain_df.iterrows():
        src_chrom = row["chromsrc"]
        if src_chrom not in chain_by_src_chrom:
            chain_by_src_chrom[src_chrom] = []
        chain_by_src_chrom[src_chrom].append(row)

    # Sort each group by source start for binary search
    for chrom in chain_by_src_chrom:
        chain_by_src_chrom[chrom].sort(key=lambda r: (r["startsrc"], r["endsrc"]))
    chain_starts_by_src_chrom = {
        chrom: [int(r["startsrc"]) for r in rows]
        for chrom, rows in chain_by_src_chrom.items()
    }

    # Map each source interval
    result_rows = []
    for interval_id, (_, src_row) in enumerate(intervals.iterrows()):
        src_chrom = src_row["chrom"]
        src_start = src_row["start"]
        src_end = src_row["end"]

        if src_chrom not in chain_by_src_chrom:
            continue

        chain_blocks = chain_by_src_chrom[src_chrom]
        chain_starts = chain_starts_by_src_chrom[src_chrom]
        block_start = bisect_left(chain_starts, src_start)
        while block_start > 0 and int(chain_blocks[block_start - 1]["endsrc"]) > src_start:
            block_start -= 1
        block_end = bisect_left(chain_starts, src_end)
        if block_end <= block_start:
            block_end = min(len(chain_blocks), block_start + 1)

        for cb in chain_blocks[block_start:block_end]:
            cb_src_start = cb["startsrc"]
            cb_src_end = cb["endsrc"]

            # Check overlap
            common_start = max(src_start, cb_src_start)
            common_end = min(src_end, cb_src_end)
            if common_start >= common_end:
                continue

            # Map to target coordinates
            if cb["strand"] == 0:
                tgt_start = cb["start"] + (common_start - cb_src_start)
                tgt_end = cb["start"] + (common_end - cb_src_start)
            else:
                tgt_start = cb["end"] - (common_end - cb_src_start)
                tgt_end = cb["end"] - (common_start - cb_src_start)

            row_dict = {
                "chrom": cb["chrom"],
                "start": int(tgt_start),
                "end": int(tgt_end),
                "intervalID": interval_id,
                "chain_id": int(cb["chain_id"]),
            }

            if include_metadata:
                row_dict["score"] = cb["score"]

            if value_col and value_col in src_row.index:
                row_dict[value_col] = src_row[value_col]

            result_rows.append(row_dict)

    if not result_rows:
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

    result = pd.DataFrame(result_rows)

    # Canonic merging: merge adjacent target blocks from same intervalID + chain_id
    if canonic:
        result = _canonic_merge(result, include_metadata, value_col)

    # Sort by target coordinates
    return result.sort_values(["chrom", "start", "end"]).reset_index(drop=True)



def _canonic_merge(df, include_metadata, value_col):
    """Merge adjacent target blocks from same intervalID and chain_id."""
    if df.empty:
        return df

    # Sort by intervalID, chain_id, chrom, start
    df = df.sort_values(["intervalID", "chain_id", "chrom", "start"]).reset_index(drop=True)

    merged = []
    prev = None
    for _, row in df.iterrows():
        if (prev is not None and
                prev["intervalID"] == row["intervalID"] and
                prev["chain_id"] == row["chain_id"] and
                prev["chrom"] == row["chrom"] and
                prev["end"] == row["start"]):
            # Merge: extend previous
            prev["end"] = row["end"]
        else:
            if prev is not None:
                merged.append(prev)
            prev = dict(row)
    if prev is not None:
        merged.append(prev)

    return pd.DataFrame(merged)


# ===================================================================
# Public API: gtrack_liftover
# ===================================================================

# Supported aggregation functions for multi-target value merging
_AGG_FUNCS = {
    "mean": lambda v: np.nanmean(v),
    "median": lambda v: np.nanmedian(v),
    "sum": lambda v: np.nansum(v),
    "min": lambda v: np.nanmin(v),
    "max": lambda v: np.nanmax(v),
    "count": lambda v: np.sum(~np.isnan(v)),
    "first": lambda v: v[~np.isnan(v)][0] if np.any(~np.isnan(v)) else np.nan,
    "last": lambda v: v[~np.isnan(v)][-1] if np.any(~np.isnan(v)) else np.nan,
}


_TRACK_IDX_MAGIC = b"MISHATDX"
_TRACK_IDX_VERSION = 1
_TRACK_IDX_FLAG_LITTLE_ENDIAN = 0x01
_TRACK_TYPE_DENSE = 0
_TRACK_TYPE_SPARSE = 1


def _compute_track_idx_checksum(entries):
    crc = _crc64_init()
    for chrom_id, offset, length, _reserved in entries:
        crc = _crc64_incremental(crc, struct.pack("<I", chrom_id))
        crc = _crc64_incremental(crc, struct.pack("<Q", offset))
        crc = _crc64_incremental(crc, struct.pack("<Q", length))
    return _crc64_finalize(crc)


def _read_track_idx(idx_path):
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


def _source_db_root_from_track_dir(src_track_dir):
    p = Path(src_track_dir).resolve()
    for parent in p.parents:
        if parent.name == "tracks":
            return parent.parent
    return None


def _load_source_chrom_names(src_track_dir):
    db_root = _source_db_root_from_track_dir(src_track_dir)
    if db_root is None:
        raise ValueError(
            "Indexed source track path must be located under a database tracks directory "
            f"(got: {src_track_dir})"
        )

    chrom_sizes_path = db_root / "chrom_sizes.txt"
    if not chrom_sizes_path.exists():
        raise ValueError(
            f"Cannot resolve chromosome IDs for indexed source track: missing {chrom_sizes_path}"
        )

    chroms = []
    with open(chrom_sizes_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid line in {chrom_sizes_path}: {line!r}")
            chroms.append(parts[0])

    return dict(enumerate(chroms))


def _parse_dense_payload(payload, chrom_name, source_label):
    if len(payload) < 4:
        return []
    if (len(payload) - 4) % 4 != 0:
        raise ValueError(f"Corrupt dense track payload for {chrom_name} in {source_label}")

    (bin_size,) = struct.unpack("<i", payload[:4])
    if bin_size <= 0:
        raise ValueError(f"Invalid dense bin size for {chrom_name} in {source_label}: {bin_size}")

    data = np.frombuffer(payload, dtype="<f4", offset=4).astype(np.float64, copy=False)
    data[np.isinf(data)] = np.nan
    valid_idx = np.flatnonzero(~np.isnan(data))
    if valid_idx.size == 0:
        return []

    starts = (valid_idx * bin_size).astype(np.int64, copy=False)
    ends = starts + int(bin_size)
    vals = data[valid_idx]
    chroms = np.full(valid_idx.size, chrom_name, dtype=object)
    return list(zip(chroms.tolist(), starts.tolist(), ends.tolist(), vals.tolist(), strict=False))


def _parse_sparse_payload(payload, chrom_name, source_label):
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

    def _decode_sparse_recs(dt):
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

    vals = recs["value"].astype(np.float64, copy=False)
    vals[np.isinf(vals)] = np.nan
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return []

    starts = recs["start"].astype(np.int64, copy=False)[valid]
    ends = recs["end"].astype(np.int64, copy=False)[valid]
    vals = vals[valid]
    chroms = np.full(starts.size, chrom_name, dtype=object)
    return list(zip(chroms.tolist(), starts.tolist(), ends.tolist(), vals.tolist(), strict=False))


def _read_indexed_source_track(src_track_dir):
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


def _read_source_track(src_track_dir):
    """Read a source track directory and return (type, intervals_df).

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
        if not fname.startswith(".")
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


def _aggregate_overlapping(intervals_df, agg_func, na_rm=True, min_n=None):
    """Aggregate values for overlapping target intervals.

    Groups intervals by (chrom, start, end) and applies the aggregation
    function to the values in each group. Returns a DataFrame with unique
    (chrom, start, end, value) rows sorted by coordinates.
    """
    if len(intervals_df) == 0:
        return intervals_df

    def _agg(group):
        vals = group["value"].to_numpy()
        if not na_rm and np.any(np.isnan(vals)):
            return np.nan
        vals_clean = vals[~np.isnan(vals)]
        if min_n is not None and len(vals_clean) < min_n:
            return np.nan
        if len(vals_clean) == 0:
            return np.nan
        return agg_func(vals_clean if na_rm else vals)

    grouped = intervals_df.groupby(["chrom", "start", "end"], sort=False)
    result = grouped.apply(_agg, include_groups=False).reset_index()
    result.columns = ["chrom", "start", "end", "value"]
    result = result.dropna(subset=["value"])
    return result.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def gtrack_liftover(track, description, src_track_dir, chain,
                    src_overlap_policy="error", tgt_overlap_policy="auto",
                    multi_target_agg="mean", params=None, na_rm=True,
                    min_n=None, min_score=None):
    """Import a track from another assembly via coordinate liftover.

    Reads a source track from *src_track_dir* (a directory containing
    per-chromosome binary track files or an indexed ``track.idx``/``track.dat``
    pair), maps its intervals through *chain* to the current target genome,
    aggregates values when multiple source intervals land on the same target
    region, and creates a new sparse track in the current database.

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
        ``"agg"``, ``"best_source_cluster"``, ``"best_cluster_union"``,
        ``"best_cluster_sum"``, ``"best_cluster_max"``.
    multi_target_agg : str, optional
        Aggregation function applied when multiple source values map to
        the same target locus. One of ``"mean"`` (default), ``"median"``,
        ``"sum"``, ``"min"``, ``"max"``, ``"count"``, ``"first"``,
        ``"last"``.
    params : dict, optional
        Extra parameters for specific aggregation methods (e.g., ``n`` for
        ``"nth"`` aggregation).
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
        The function creates a new sparse track in the current database as
        a side effect and does not return a value.

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
        _set_created_attrs,
        _track_dir_for_create,
        _track_exists,
        _validate_track_name,
        gtrack_create_sparse,
    )

    _checkroot()
    _validate_track_name(track)
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")

    # Validate aggregation function
    if multi_target_agg not in _AGG_FUNCS:
        raise ValueError(
            f"Unsupported aggregation: {multi_target_agg}. "
            f"Supported: {', '.join(sorted(_AGG_FUNCS))}"
        )
    agg_func = _AGG_FUNCS[multi_target_agg]

    # Load chain if path
    if isinstance(chain, str):
        chain = gintervals_load_chain(
            chain,
            src_overlap_policy=src_overlap_policy,
            tgt_overlap_policy=tgt_overlap_policy,
            min_score=min_score,
        )
    elif isinstance(chain, pd.DataFrame):
        # Pre-loaded chain — policies are already baked in
        pass
    else:
        raise TypeError("chain must be a file path string or a chain DataFrame")

    # Read source track
    src_type, src_data = _read_source_track(src_track_dir)

    if len(src_data) == 0 or len(chain) == 0:
        # Create empty sparse track
        pd.DataFrame({"chrom": pd.Series(dtype=str),
                                       "start": pd.Series(dtype=int),
                                       "end": pd.Series(dtype=int)})
        # Need at least one interval for track creation, create an empty track dir
        track_dir = _track_dir_for_create(track)
        track_dir.mkdir(parents=True, exist_ok=True)
        # Write .attributes file
        _pymisha.pm_dbreload()
        _set_created_attrs(track, description,
                           f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)')
        return

    # Liftover source intervals to target coordinates
    lifted = gintervals_liftover(
        src_data[["chrom", "start", "end", "value"]].copy(),
        chain,
        value_col="value" if "value" in src_data.columns else None,
        canonic=True,
    )

    # If gintervals_liftover didn't carry values (shouldn't happen with value_col),
    # merge values back via intervalID
    if "value" not in lifted.columns and "intervalID" in lifted.columns:
        lifted = lifted.merge(
            src_data[["value"]].reset_index().rename(columns={"index": "intervalID"}),
            on="intervalID", how="left",
        )

    if len(lifted) == 0:
        # No intervals mapped — create empty track
        track_dir = _track_dir_for_create(track)
        track_dir.mkdir(parents=True, exist_ok=True)
        _pymisha.pm_dbreload()
        _set_created_attrs(track, description,
                           f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)')
        return

    # Aggregate overlapping target intervals
    target_data = lifted[["chrom", "start", "end", "value"]].copy()
    target_data = _aggregate_overlapping(target_data, agg_func, na_rm=na_rm, min_n=min_n)

    if len(target_data) == 0:
        track_dir = _track_dir_for_create(track)
        track_dir.mkdir(parents=True, exist_ok=True)
        _pymisha.pm_dbreload()
        _set_created_attrs(track, description,
                           f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)')
        return

    # Create sparse track with the lifted data
    gtrack_create_sparse(
        track, description,
        target_data[["chrom", "start", "end"]],
        target_data["value"].to_numpy(),
    )

    # Update the created.by attribute to reflect liftover (bypass readonly check)
    from .tracks import _load_track_attributes, _save_track_attributes
    attrs = _load_track_attributes(track)
    attrs["created.by"] = f'gtrack.liftover("{track}", description, "{src_track_dir}", chain)'
    _save_track_attributes(track, attrs)
