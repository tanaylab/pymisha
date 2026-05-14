"""ChromAlias resolution: pick a canonical column + build a translator.

The alias_df comes from `_<src>_fetch_assets`. Columns are arbitrary source
labels (e.g. "ucsc", "refseq", "genbank", "ensembl"); each row is one
sequence with its name in each convention. There is no single fixed canonical
column - we pick the column that best matches the groot's chromosomes,
weighted by bp.

C.2.5 shipped single-pass detection (match_by_length=False). C.3.1 adds the
four-pass rescue (match_by_length=True): length-fill missing canonical
values, length-override misnamed canonical values, name-override across-row
collisions, and synthesize rows for user-supplied target_chroms.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def _canonical_coverage(
    canonical_values: pd.Series,
    groot_chroms: list[str],
    groot_lengths: list[int],
) -> float:
    """bp-weighted coverage: fraction of groot bp whose chrom names appear in
    `canonical_values`.

    Returns 1.0 if groot_chroms is empty.
    """
    if not groot_chroms:
        return 1.0
    if len(groot_chroms) != len(groot_lengths):
        raise ValueError(
            f"groot_chroms ({len(groot_chroms)}) and groot_lengths ({len(groot_lengths)}) must align"
        )
    canon_set = set(canonical_values.dropna().astype(str)) - {""}
    total = sum(groot_lengths)
    if total == 0:
        return 0.0
    mapped_bp = sum(
        length for chrom, length in zip(groot_chroms, groot_lengths, strict=True)
        if chrom in canon_set
    )
    return mapped_bp / total


def _detect_alias_column(
    alias_df: pd.DataFrame,
    groot_chroms: list[str],
    groot_lengths: list[int],
    *,
    min_coverage: float = 1.0,
) -> tuple[str | None, dict[str, float]]:
    """Pick the alias column with highest bp-weighted coverage.

    Returns (picked_column_name | None, per_column_coverage_dict).
    Returns None for the picked column if no column reaches min_coverage.
    Examines every column in alias_df.
    """
    coverages: dict[str, float] = {}
    for col in alias_df.columns:
        coverages[col] = _canonical_coverage(alias_df[col], groot_chroms, groot_lengths)

    eligible = {c: v for c, v in coverages.items() if v >= min_coverage}
    if not eligible:
        return None, coverages
    best = max(eligible.items(), key=lambda kv: kv[1])
    return best[0], coverages


def _build_translator(alias_df: pd.DataFrame, canonical_col: str) -> Callable[[str], str | None]:
    """Build a translator: upstream_chrom_name -> canonical_chrom_name | None.

    The translator searches every column of alias_df. For each row, if the
    upstream name appears in any column, the translator returns the canonical
    column's value (or None if the canonical column is empty/NaN for that
    row).

    Implementation: precompute a dict {upstream_name: canonical_value} from
    a single sweep over rows.

    Returns identity-ish: an upstream chrom that already equals a canonical
    value also resolves (the canonical column is searched too).
    """
    lookup: dict[str, str] = {}
    for _, row in alias_df.iterrows():
        canon = row.get(canonical_col)
        if pd.isna(canon) or canon == "":
            continue
        canon_str = str(canon)
        for _col, val in row.items():
            if pd.isna(val) or val == "":
                continue
            val_str = str(val)
            # First-wins: don't overwrite an existing mapping. Means rows
            # earlier in alias_df dominate name collisions.
            lookup.setdefault(val_str, canon_str)

    def _tr(name: str) -> str | None:
        return lookup.get(name)

    return _tr


# ---------------------------------------------------------------------------
# C.3.1 multi-pass rescue helpers
# ---------------------------------------------------------------------------

def _length_column(alias_df: pd.DataFrame) -> str | None:
    """Return the per-row length column name (case-insensitive `length`), or
    None if absent. Only an exact lowercased match wins so columns like
    `seqLength` or `len` are ignored - keeps the heuristic predictable.
    """
    for col in alias_df.columns:
        if str(col).lower() == "length":
            return str(col)
    return None


def _alias_row_lengths(alias_df: pd.DataFrame) -> pd.Series | None:
    """Return numeric per-row lengths aligned with alias_df.index, or None
    if no length column exists. Non-numeric cells become NaN.
    """
    col = _length_column(alias_df)
    if col is None:
        return None
    return pd.to_numeric(alias_df[col], errors="coerce")


def _is_missing(value: object) -> bool:
    """True for NaN/None/empty-string canonical cells."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return bool(isinstance(value, str) and value == "")


def _unique_pair_lookup(
    alias_row_lengths: pd.Series,
    groot_chroms: list[str],
    groot_lengths: list[int],
) -> dict[int, str]:
    """Lengths that occur exactly once on BOTH sides -> the groot_chrom for
    that length. Empty if no length pairs uniquely.
    """
    if alias_row_lengths is None or len(groot_chroms) == 0:
        return {}
    a_counts: dict[int, int] = {}
    for v in alias_row_lengths.dropna().tolist():
        try:
            key = int(v)
        except (TypeError, ValueError):
            continue
        a_counts[key] = a_counts.get(key, 0) + 1
    g_counts: dict[int, int] = {}
    for L in groot_lengths:
        try:
            key = int(L)
        except (TypeError, ValueError):
            continue
        g_counts[key] = g_counts.get(key, 0) + 1
    pair_lengths = {k for k, v in a_counts.items() if v == 1} & {
        k for k, v in g_counts.items() if v == 1
    }
    if not pair_lengths:
        return {}
    return {
        int(L): chrom
        for chrom, L in zip(groot_chroms, groot_lengths, strict=True)
        if int(L) in pair_lengths
    }


def _length_fill(
    canonical: pd.Series,
    alias_row_lengths: pd.Series,
    groot_chroms: list[str],
    groot_lengths: list[int],
) -> pd.Series:
    """Pass 1: fill rows where canonical is missing using a length match
    that is unique on BOTH the alias side and the groot side.
    """
    if alias_row_lengths is None:
        return canonical
    out = canonical.copy()
    needs_idx = [i for i in out.index if _is_missing(out.loc[i])]
    if not needs_idx:
        return out
    lookup = _unique_pair_lookup(alias_row_lengths, groot_chroms, groot_lengths)
    if not lookup:
        return out
    for i in needs_idx:
        raw_len = alias_row_lengths.loc[i] if i in alias_row_lengths.index else None
        if raw_len is None or pd.isna(raw_len):
            continue
        try:
            key = int(raw_len)
        except (TypeError, ValueError):
            continue
        if key in lookup:
            out.loc[i] = lookup[key]
    return out


def _length_override(
    canonical: pd.Series,
    alias_row_lengths: pd.Series,
    groot_chroms: list[str],
    groot_lengths: list[int],
) -> pd.Series:
    """Pass 2: override canonical entries that exist but aren't in groot.
    Only replaces when the row's length matches uniquely on BOTH sides AND
    the target groot_chrom isn't already used by another canonical cell.
    """
    if alias_row_lengths is None:
        return canonical
    out = canonical.copy()
    misaligned = []
    groot_set = set(groot_chroms)
    for i in out.index:
        v = out.loc[i]
        if _is_missing(v):
            continue
        if str(v) not in groot_set:
            misaligned.append(i)
    if not misaligned:
        return out
    lookup = _unique_pair_lookup(alias_row_lengths, groot_chroms, groot_lengths)
    if not lookup:
        return out
    # Never reuse a groot_chrom already present in canonical (real placement).
    used = {str(v) for v in out.dropna().tolist() if not _is_missing(v)}
    available = {k: v for k, v in lookup.items() if v not in used}
    if not available:
        return out
    for i in misaligned:
        raw_len = alias_row_lengths.loc[i] if i in alias_row_lengths.index else None
        if raw_len is None or pd.isna(raw_len):
            continue
        try:
            key = int(raw_len)
        except (TypeError, ValueError):
            continue
        if key in available:
            out.loc[i] = available[key]
            # Move the freshly-placed chrom out of `available` so a later row
            # at the same length doesn't reuse it (defensive; lookup is
            # length-unique on both sides so this can't actually fire).
            del available[key]
    return out


def _name_override(
    canonical: pd.Series,
    alias_row_lengths: pd.Series,
    groot_chroms: list[str],
    groot_lengths: list[int],
) -> pd.Series:
    """Pass 3: when two rows collide on the same canonical name, use lengths
    to decide which row is the real owner. The loser's canonical is cleared
    (set to None) so downstream callers can either leave it unmapped or
    re-fill via other means.
    """
    if alias_row_lengths is None:
        return canonical
    out = canonical.copy()
    chrom_to_length: dict[str, int] = {}
    for chrom, L in zip(groot_chroms, groot_lengths, strict=True):
        try:
            chrom_to_length[chrom] = int(L)
        except (TypeError, ValueError):
            continue
    # Group rows by their current canonical value.
    groups: dict[str, list] = {}
    for i in out.index:
        v = out.loc[i]
        if _is_missing(v):
            continue
        groups.setdefault(str(v), []).append(i)
    for canon_name, idxs in groups.items():
        if len(idxs) < 2:
            continue
        # Need a known groot length for the collided name to arbitrate.
        if canon_name not in chrom_to_length:
            continue
        expected = chrom_to_length[canon_name]
        winners = []
        for i in idxs:
            raw_len = (
                alias_row_lengths.loc[i] if i in alias_row_lengths.index else None
            )
            if raw_len is None or pd.isna(raw_len):
                continue
            try:
                if int(raw_len) == expected:
                    winners.append(i)
            except (TypeError, ValueError):
                continue
        # Only resolve when exactly one row's length matches the groot's
        # length for that name. Any other shape is ambiguous - leave alone.
        if len(winners) != 1:
            continue
        winner = winners[0]
        for i in idxs:
            if i != winner:
                out.loc[i] = None
    return out


def _synthesize_target_chroms(
    alias_df: pd.DataFrame,
    canonical_col: str,
    target_chroms: list[str],
    target_lengths: list[int],
    groot_chroms: list[str],
    groot_lengths: list[int],
) -> pd.DataFrame:
    """Pass 4: append synthetic rows for target_chroms not yet present in
    `alias_df[canonical_col]`. The synthetic row carries the target name in
    the canonical column and the length in the length column (if any);
    other columns are empty so the translator returns the canonical name
    unchanged via the canonical-as-input pathway in `_build_translator`.

    Only target_chroms whose length matches some groot_length AND whose name
    is missing from the canonical column are appended.
    """
    if not target_chroms:
        return alias_df
    if not target_lengths or len(target_chroms) != len(target_lengths):
        raise ValueError(
            "target_chroms requires aligned target_lengths (same length)"
        )
    groot_length_set = {int(L) for L in groot_lengths}
    already = {
        str(v) for v in alias_df[canonical_col].dropna().tolist() if not _is_missing(v)
    }
    length_col = _length_column(alias_df)
    new_rows: list[dict] = []
    for name, L in zip(target_chroms, target_lengths, strict=True):
        if name in already:
            continue
        try:
            L_int = int(L)
        except (TypeError, ValueError):
            continue
        if L_int not in groot_length_set:
            continue
        row: dict = dict.fromkeys(alias_df.columns, "")
        row[canonical_col] = name
        if length_col is not None:
            row[length_col] = L_int
        new_rows.append(row)
    if not new_rows:
        return alias_df
    appended = pd.DataFrame(new_rows, columns=list(alias_df.columns))
    return pd.concat([alias_df, appended], ignore_index=True)


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

def _identity_translator(groot_chroms: list[str]) -> Callable[[str], str | None]:
    groot_set = set(groot_chroms)
    return lambda name: name if name in groot_set else None


def _resolve_chrom_alias_single_pass(
    alias_df: pd.DataFrame | None,
    groot_chroms: list[str],
    groot_lengths: list[int],
    *,
    min_coverage: float = 1.0,
) -> Callable[[str], str | None]:
    """Single-pass resolver: pick the column that hits min_coverage on its
    own and build a translator. Raises if no column qualifies.
    """
    if alias_df is None or alias_df.empty:
        return _identity_translator(groot_chroms)
    canonical, coverages = _detect_alias_column(
        alias_df, groot_chroms, groot_lengths, min_coverage=min_coverage
    )
    if canonical is None:
        raise ValueError(
            f"chromAlias has no column with >= {min_coverage:.0%} bp-weighted "
            f"coverage of groot chroms. Per-column coverages: {coverages}"
        )
    return _build_translator(alias_df, canonical)


def _resolve_chrom_alias(
    alias_df: pd.DataFrame | None,
    groot_chroms: list[str],
    groot_lengths: list[int],
    *,
    target_chroms: list[str] | None = None,
    target_lengths: list[int] | None = None,
    min_coverage: float = 1.0,
    match_by_length: bool = False,
) -> Callable[[str], str | None]:
    """Pick a canonical column and return a translator.

    When `match_by_length=False`, runs the single-pass resolver: pick the
    column that hits `min_coverage` on its own, or raise.

    When `match_by_length=True`, runs the four-pass rescue (R 5.6.16):
      1. Length-fill missing canonical cells via unique-length pairing.
      2. Length-override misnamed cells when length pairs uniquely AND the
         target groot_chrom isn't already used.
      3. Name-override: clear cross-row collisions when lengths arbitrate.
      4. Synthesize rows for any `target_chroms` still unmapped.
    After the passes, the post-rescue bp-weighted coverage is gated against
    `min_coverage`; failure raises with the pre-rescue per-column scores.

    If `alias_df` is None or empty:
      * With target_chroms / target_lengths supplied, build a synthetic
        alias frame containing just those rows.
      * Otherwise, return an identity translator.
    """
    if not match_by_length:
        return _resolve_chrom_alias_single_pass(
            alias_df, groot_chroms, groot_lengths, min_coverage=min_coverage
        )

    # Argument-shape checks shared by all match_by_length=True paths.
    if target_chroms and not target_lengths:
        raise ValueError(
            "target_chroms requires target_lengths (match_by_length=True)"
        )
    if target_lengths and not target_chroms:
        raise ValueError(
            "target_lengths requires target_chroms (match_by_length=True)"
        )
    # Once we get here, target_chroms and target_lengths are either both None /
    # empty or both populated. Materialize concrete lists for the populated case
    # so mypy can stop treating them as Optional.
    _t_chroms: list[str] = list(target_chroms) if target_chroms else []
    _t_lengths: list[int] = list(target_lengths) if target_lengths else []
    if _t_chroms and len(_t_chroms) != len(_t_lengths):
        raise ValueError(
            "target_chroms and target_lengths must have the same length"
        )

    # Empty alias_df: either synthesize from target_chroms or fall back to
    # identity.
    if alias_df is None or alias_df.empty:
        if _t_chroms:
            synth = pd.DataFrame(
                {"_target": _t_chroms, "length": _t_lengths}
            )
            final_coverage = _canonical_coverage(
                synth["_target"], groot_chroms, groot_lengths
            )
            if final_coverage < min_coverage:
                raise ValueError(
                    f"synthetic target_chroms cover {final_coverage:.2%} of "
                    f"groot bp; need >= {min_coverage:.2%}"
                )
            return _build_translator(synth, "_target")
        return _identity_translator(groot_chroms)

    # Pre-rescue canonical column pick. min_coverage=0 so we always get a
    # winner if any column has signal; the gate is applied after rescue.
    length_col = _length_column(alias_df)
    candidate_cols = [c for c in alias_df.columns if c != length_col]
    if not candidate_cols:
        # alias_df has nothing but a length column. Treat as empty for the
        # purposes of detection.
        if _t_chroms:
            synth = pd.DataFrame(
                {"_target": _t_chroms, "length": _t_lengths}
            )
            final_coverage = _canonical_coverage(
                synth["_target"], groot_chroms, groot_lengths
            )
            if final_coverage < min_coverage:
                raise ValueError(
                    f"synthetic target_chroms cover {final_coverage:.2%} of "
                    f"groot bp; need >= {min_coverage:.2%}"
                )
            return _build_translator(synth, "_target")
        raise ValueError("alias_df has no candidate columns for canonical pick")

    canonical_col, coverages = _detect_alias_column(
        alias_df[candidate_cols], groot_chroms, groot_lengths, min_coverage=0.0
    )
    if canonical_col is None:
        # _detect_alias_column with min_coverage=0 returns the best column
        # whenever there is at least one column; only a literal empty
        # candidate set would hit this branch. We've already gated above.
        canonical_col = candidate_cols[0]

    canonical = alias_df[canonical_col].astype("object").copy()
    alias_row_lengths = _alias_row_lengths(alias_df)

    if alias_row_lengths is not None:
        canonical = _length_fill(canonical, alias_row_lengths, groot_chroms, groot_lengths)
        canonical = _length_override(canonical, alias_row_lengths, groot_chroms, groot_lengths)
        canonical = _name_override(canonical, alias_row_lengths, groot_chroms, groot_lengths)

    rescued_df = alias_df.copy()
    rescued_df[canonical_col] = canonical

    if _t_chroms:
        rescued_df = _synthesize_target_chroms(
            rescued_df,
            canonical_col,
            _t_chroms,
            _t_lengths,
            groot_chroms,
            groot_lengths,
        )

    final_coverage = _canonical_coverage(
        rescued_df[canonical_col], groot_chroms, groot_lengths
    )
    if final_coverage < min_coverage:
        raise ValueError(
            f"chromAlias post-rescue coverage {final_coverage:.2%} < "
            f"min_coverage {min_coverage:.2%}. Per-column pre-rescue "
            f"coverages: {coverages}"
        )

    return _build_translator(rescued_df, canonical_col)
