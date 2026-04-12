"""Motif format parsers for MEME, JASPAR, and HOMER formats.

Each parser returns ``dict[str, pd.DataFrame]`` where keys are motif IDs and
values are position probability matrices with columns ``A``, ``C``, ``G``,
``T``.  The DataFrames are directly usable with :func:`gseq_pwm`.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as _numpy
import pandas as _pandas

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dedup_motif_ids(ids: list[str]) -> list[str]:
    """Append .1, .2, ... suffixes to duplicate motif IDs."""
    if not ids:
        return ids
    seen: dict[str, int] = {}
    for mid in ids:
        seen[mid] = seen.get(mid, 0) + 1
    dups = {k for k, v in seen.items() if v > 1}
    if not dups:
        return ids
    warnings.warn("Duplicate motif IDs found; appending numeric suffixes to disambiguate", stacklevel=2)
    counters: dict[str, int] = {}
    result: list[str] = []
    for mid in ids:
        if mid in dups:
            counters[mid] = counters.get(mid, 0) + 1
            result.append(f"{mid}.{counters[mid]}")
        else:
            result.append(mid)
    return result


def _renormalize_rows(mat: _numpy.ndarray, tol: float = 1e-4,
                      context: str = "") -> _numpy.ndarray:
    """Re-normalize matrix rows that do not sum to 1.0."""
    rsums = mat.sum(axis=1)
    bad = _numpy.where(_numpy.abs(rsums - 1.0) > tol)[0]
    if len(bad) > 0:
        warnings.warn(
            f"Row sums deviate from 1.0 in {context}; re-normalizing "
            f"{len(bad)} row(s)", stacklevel=2
        )
        for i in bad:
            if rsums[i] == 0:
                mat[i, :] = 0.25
            else:
                mat[i, :] = mat[i, :] / rsums[i]
    return mat


def _parse_meme_key(line: str, key: str) -> float | None:
    """Parse a ``key= value`` pair from a MEME matrix header line.

    Returns the numeric value, or ``None`` if the key is not found.
    """
    pattern = rf"\b{re.escape(key)}=\s*(\S+)"
    m = re.search(pattern, line)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# MEME parser
# ---------------------------------------------------------------------------

def gseq_read_meme(file: str) -> dict[str, _pandas.DataFrame]:
    """Read motifs from a MEME minimal motif format file.

    Parses a MEME minimal motif format file and returns a dict of position
    probability matrices (PPM).  Each DataFrame has columns ``A``, ``C``,
    ``G``, ``T`` with one row per motif position.  Metadata is stored in the
    DataFrame's ``.attrs`` dict.

    Parameters
    ----------
    file : str
        Path to a MEME format file (``.meme``, ``.txt``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are motif identifiers, values are DataFrames.  Each DataFrame
        has ``.attrs`` with keys: ``name``, ``alength``, ``w``, ``nsites``,
        ``E``, ``url``, ``strand``, ``background``.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file is malformed or contains unsupported content.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

    with open(file) as fh:
        lines = fh.read().splitlines()

    # Strip Windows line endings
    lines = [ln.rstrip("\r") for ln in lines]

    # --- Parse global header fields ---
    strand: str | None = None
    background: dict[str, float] | None = None

    # Strands
    for ln in lines:
        if re.match(r"^strands:", ln, re.IGNORECASE):
            strand = re.sub(r"^strands:\s*", "", ln, flags=re.IGNORECASE).strip()
            break

    # Alphabet validation
    for ln in lines:
        m = re.match(r"^ALPHABET\s*=\s*(.*)", ln, re.IGNORECASE)
        if m:
            alpha_val = m.group(1).strip()
            if alpha_val and not re.fullmatch(r"[ACGTacgt]+", alpha_val):
                raise ValueError("Only DNA alphabet (ACGT) is supported")
            break

    # Background letter frequencies
    for idx, ln in enumerate(lines):
        if re.match(r"^Background letter frequencies", ln, re.IGNORECASE):
            bg_text = ""
            for j in range(idx + 1, min(idx + 3, len(lines))):
                tok = lines[j].strip()
                if not tok or re.match(r"^MOTIF\b", tok, re.IGNORECASE):
                    break
                bg_text += " " + tok
            bg_tokens = bg_text.split()
            if len(bg_tokens) >= 8:
                bg_names = bg_tokens[0::2][:4]
                try:
                    bg_vals = [float(x) for x in bg_tokens[1::2][:4]]
                    background = dict(zip(bg_names, bg_vals, strict=False))
                except ValueError:
                    pass
            break

    # --- Find MOTIF blocks ---
    motif_starts = [i for i, ln in enumerate(lines) if re.match(r"^MOTIF\b", ln)]
    if not motif_starts:
        raise ValueError(f"No motifs found in {file}")

    results: dict[str, _pandas.DataFrame] = {}
    ids: list[str] = []

    for mi, start_line in enumerate(motif_starts):
        end_line = motif_starts[mi + 1] - 1 if mi + 1 < len(motif_starts) else len(lines) - 1
        block = lines[start_line:end_line + 1]

        # Parse MOTIF line: MOTIF <id> [<name>]
        motif_tokens = block[0].split()
        motif_id = motif_tokens[1] if len(motif_tokens) >= 2 else f"motif_{mi + 1}"
        motif_name: str | None = (
            " ".join(motif_tokens[2:]) if len(motif_tokens) >= 3 else None
        )

        # Find letter-probability matrix header
        lpm_idx = None
        for bi, bln in enumerate(block):
            if re.search(r"letter-probability matrix", bln, re.IGNORECASE):
                lpm_idx = bi
                break

        if lpm_idx is None:
            # Check for log-odds matrix
            for bln in block:
                if re.search(r"log-odds matrix", bln, re.IGNORECASE):
                    raise ValueError(
                        "Log-odds matrices not supported; use letter-probability "
                        "matrix format"
                    )
            warnings.warn(f"Motif '{motif_id}' has no matrix data; skipping", stacklevel=2)
            continue

        # Parse metadata from the matrix header line
        header_line = block[lpm_idx]
        alength = _parse_meme_key(header_line, "alength")
        w = _parse_meme_key(header_line, "w")
        nsites = _parse_meme_key(header_line, "nsites")
        E_val = _parse_meme_key(header_line, "E")

        # Read matrix rows starting after the header line
        mat_lines: list[str] = []
        for j in range(lpm_idx + 1, len(block)):
            ln = block[j].strip()
            if not ln:
                continue
            if re.match(r"^URL\b", ln, re.IGNORECASE):
                break
            if re.match(r"^MOTIF\b", ln, re.IGNORECASE):
                break
            # Check if this line looks numeric
            try:
                [float(x) for x in ln.split()]
            except ValueError:
                break
            mat_lines.append(ln)

        if not mat_lines:
            warnings.warn(f"Motif '{motif_id}' has no matrix data; skipping", stacklevel=2)
            continue

        # Check w consistency
        if w is not None and len(mat_lines) != int(w):
            raise ValueError(
                f"Expected {int(w)} rows but found {len(mat_lines)} for "
                f"motif '{motif_id}'"
            )

        # Parse matrix
        rows: list[list[float]] = []
        for i, ml in enumerate(mat_lines):
            toks: list[str] = ml.split()
            if len(toks) != 4:
                raise ValueError(
                    f"Expected 4 columns (A,C,G,T) at line "
                    f"{start_line + lpm_idx + i + 1} of motif '{motif_id}'"
                )
            try:
                row = [float(v) for v in toks]
            except ValueError as err:
                raise ValueError(
                    f"Non-numeric value in probability matrix for "
                    f"motif '{motif_id}'"
                ) from err
            rows.append(row)

        mat = _numpy.array(rows, dtype=float)

        # Check for log-odds (negative values)
        if _numpy.any(mat < 0):
            raise ValueError(
                "Log-odds matrices not supported; use letter-probability "
                "matrix format"
            )

        # Re-normalize rows if needed
        mat = _renormalize_rows(mat, context=f"motif '{motif_id}'")

        # Parse URL if present
        url_val: str | None = None
        for bln in block:
            if re.match(r"^URL\b", bln.strip(), re.IGNORECASE):
                url_val = re.sub(r"^URL\s+", "", bln.strip(), flags=re.IGNORECASE)
                break

        # Build DataFrame
        df = _pandas.DataFrame(mat, columns=["A", "C", "G", "T"])
        df.attrs["name"] = motif_name
        df.attrs["alength"] = int(alength) if alength is not None else 4
        df.attrs["w"] = int(mat.shape[0])
        df.attrs["nsites"] = float(nsites) if nsites is not None else None
        df.attrs["E"] = float(E_val) if E_val is not None else None
        df.attrs["url"] = url_val
        df.attrs["strand"] = strand
        df.attrs["background"] = background

        ids.append(motif_id)
        results[motif_id] = df  # temporary; will rebuild with deduped ids

    if not ids:
        raise ValueError(f"No motifs found in {file}")

    deduped = _dedup_motif_ids(ids)
    final: dict[str, _pandas.DataFrame] = {}
    for orig_id, new_id in zip(ids, deduped, strict=False):
        df = results[orig_id]
        final[new_id] = df
    return final


# ---------------------------------------------------------------------------
# JASPAR parser
# ---------------------------------------------------------------------------

def gseq_read_jaspar(file: str) -> dict[str, _pandas.DataFrame]:
    """Read motifs from a JASPAR PFM format file.

    Parses a JASPAR Position Frequency Matrix (PFM) file and returns a dict
    of position probability matrices (PPM).  Supports both the standard JASPAR
    header format (``>ID NAME`` followed by labeled A/C/G/T rows) and the
    simple 4-row PFM format.  Counts are converted to probabilities.

    Parameters
    ----------
    file : str
        Path to a JASPAR format file (``.jaspar``, ``.pfm``, ``.txt``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are motif identifiers, values are DataFrames with columns
        ``A``, ``C``, ``G``, ``T``.  Each DataFrame has ``.attrs`` with keys:
        ``name``, ``w``, ``nsites``, ``format``.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file is malformed.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

    with open(file) as fh:
        lines = fh.read().splitlines()

    lines = [ln.rstrip("\r") for ln in lines]
    nonempty = [ln for ln in lines if ln.strip()]

    if not nonempty:
        raise ValueError(f"No motifs found in {file}")

    has_header = any(ln.startswith(">") for ln in nonempty)

    if has_header:
        return _parse_jaspar_header(nonempty, file)
    return _parse_jaspar_simple(nonempty, file)


def _parse_jaspar_header(lines: list[str], file: str) -> dict[str, _pandas.DataFrame]:
    """Parse JASPAR header format (>ID NAME + labeled rows)."""
    header_idx = [i for i, ln in enumerate(lines) if ln.startswith(">")]
    if not header_idx:
        raise ValueError(f"No motifs found in {file}")

    results: dict[str, _pandas.DataFrame] = {}
    ids: list[str] = []

    for hi, start in enumerate(header_idx):
        end = header_idx[hi + 1] - 1 if hi + 1 < len(header_idx) else len(lines) - 1

        # Parse header: >ID NAME
        hdr = lines[start].lstrip(">").strip()
        hdr_tokens = hdr.split()
        motif_id = hdr_tokens[0]
        motif_name = " ".join(hdr_tokens[1:]) if len(hdr_tokens) >= 2 else None

        # Grab data rows
        data_lines = [ln for ln in lines[start + 1:end + 1] if ln.strip()]

        if len(data_lines) != 4:
            raise ValueError(
                f"Expected 4 rows (A/C/G/T) for motif '{motif_id}', "
                f"got {len(data_lines)}"
            )

        expected_bases = {"A", "C", "G", "T"}
        count_rows: dict[str, list[float]] = {}

        for _ri, dl in enumerate(data_lines):
            # Strip brackets
            dl = dl.replace("[", "").replace("]", "")
            parts = dl.split()
            parts = [p for p in parts if p]

            # Extract base label
            base_label = parts[0].rstrip(":").upper()
            if base_label not in expected_bases:
                raise ValueError(
                    f"Unexpected row label '{base_label}'; "
                    f"expected one of A, C, G, T"
                )

            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError as err:
                raise ValueError(
                    f"Non-numeric count value in motif '{motif_id}'"
                ) from err

            if any(v < 0 for v in vals):
                raise ValueError(
                    "Negative count values not allowed in JASPAR format"
                )

            if count_rows and len(vals) != len(next(iter(count_rows.values()))):
                raise ValueError(
                    f"Rows have different lengths for motif '{motif_id}'"
                )

            count_rows[base_label] = vals

        # Build count matrix in A, C, G, T order: shape (4, w)
        w = len(count_rows["A"])
        count_mat = _numpy.array(
            [count_rows["A"], count_rows["C"], count_rows["G"], count_rows["T"]],
            dtype=float,
        )

        # Convert counts to probabilities by column
        col_sums = count_mat.sum(axis=0)
        zero_cols = _numpy.where(col_sums == 0)[0]
        if len(zero_cols) > 0:
            warnings.warn(
                f"Position(s) {', '.join(str(c + 1) for c in zero_cols)} "
                f"have zero total counts; using uniform probability", stacklevel=2
            )

        prob_mat = count_mat.copy()
        for j in range(prob_mat.shape[1]):
            if col_sums[j] == 0:
                prob_mat[:, j] = 0.25
            else:
                prob_mat[:, j] = count_mat[:, j] / col_sums[j]

        # Transpose: bases-by-positions -> positions-by-bases
        mat = prob_mat.T  # shape (w, 4)

        nsites = float(col_sums[0])

        df = _pandas.DataFrame(mat, columns=["A", "C", "G", "T"])
        df.attrs["name"] = motif_name
        df.attrs["w"] = int(w)
        df.attrs["nsites"] = nsites
        df.attrs["format"] = "jaspar"

        ids.append(motif_id)
        results[motif_id] = df

    deduped = _dedup_motif_ids(ids)
    final: dict[str, _pandas.DataFrame] = {}
    for orig_id, new_id in zip(ids, deduped, strict=False):
        final[new_id] = results[orig_id]
    return final


def _parse_jaspar_simple(lines: list[str], file: str) -> dict[str, _pandas.DataFrame]:
    """Parse JASPAR simple PFM format (4 rows of counts, no header)."""
    if len(lines) % 4 != 0:
        raise ValueError(
            f"Simple PFM format expects a multiple of 4 non-empty lines, "
            f"got {len(lines)}"
        )

    n_motifs = len(lines) // 4
    results: dict[str, _pandas.DataFrame] = {}
    ids: list[str] = []

    base_id = os.path.splitext(os.path.basename(file))[0]

    for mi in range(n_motifs):
        row_start = mi * 4
        row_vals: list[list[float]] = []

        for ri in range(4):
            ln = lines[row_start + ri].strip()
            # Strip brackets if present
            ln = ln.replace("[", "").replace("]", "")
            try:
                vals = [float(x) for x in ln.split()]
            except ValueError as err:
                raise ValueError(
                    f"Non-numeric count value in motif at rows "
                    f"{row_start + 1}-{row_start + 4}"
                ) from err
            if any(v < 0 for v in vals):
                raise ValueError(
                    "Negative count values not allowed in JASPAR format"
                )

            if row_vals and len(vals) != len(row_vals[0]):
                raise ValueError(
                    "Rows have different lengths in simple PFM format"
                )

            row_vals.append(vals)

        count_mat = _numpy.array(row_vals, dtype=float)  # (4, w)

        # Convert counts to probabilities by column
        col_sums = count_mat.sum(axis=0)
        zero_cols = _numpy.where(col_sums == 0)[0]
        if len(zero_cols) > 0:
            warnings.warn(
                f"Position(s) {', '.join(str(c + 1) for c in zero_cols)} "
                f"have zero total counts; using uniform probability", stacklevel=2
            )

        prob_mat = count_mat.copy()
        for j in range(prob_mat.shape[1]):
            if col_sums[j] == 0:
                prob_mat[:, j] = 0.25
            else:
                prob_mat[:, j] = count_mat[:, j] / col_sums[j]

        # Transpose: bases-by-positions -> positions-by-bases
        mat = prob_mat.T

        motif_id = base_id if n_motifs == 1 else f"{base_id}.{mi + 1}"

        df = _pandas.DataFrame(mat, columns=["A", "C", "G", "T"])
        df.attrs["name"] = None
        df.attrs["w"] = int(mat.shape[0])
        df.attrs["nsites"] = None
        df.attrs["format"] = "simple"

        ids.append(motif_id)
        results[motif_id] = df

    deduped = _dedup_motif_ids(ids)
    final: dict[str, _pandas.DataFrame] = {}
    for orig_id, new_id in zip(ids, deduped, strict=False):
        final[new_id] = results[orig_id]
    return final


# ---------------------------------------------------------------------------
# HOMER parser
# ---------------------------------------------------------------------------

def gseq_read_homer(file: str) -> dict[str, _pandas.DataFrame]:
    """Read motifs from a HOMER motif format file.

    Parses a HOMER ``.motif`` format file and returns a dict of position
    probability matrices (PPM).  Each DataFrame has columns ``A``, ``C``,
    ``G``, ``T``.

    Parameters
    ----------
    file : str
        Path to a HOMER motif file (``.motif``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are derived from the consensus sequence.  Each DataFrame has
        ``.attrs`` with keys: ``name``, ``consensus``, ``log_odds_threshold``,
        ``log_p_value``, ``w``, ``source``.

    Raises
    ------
    FileNotFoundError
        If *file* does not exist.
    ValueError
        If the file is malformed.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f"File not found: {file}")

    with open(file) as fh:
        lines = fh.read().splitlines()

    lines = [ln.rstrip("\r") for ln in lines]

    header_idx = [i for i, ln in enumerate(lines) if ln.startswith(">")]
    if not header_idx:
        raise ValueError(f"No motifs found in HOMER file {file}")

    results: dict[str, _pandas.DataFrame] = {}
    ids: list[str] = []

    for hi, start in enumerate(header_idx):
        end = header_idx[hi + 1] - 1 if hi + 1 < len(header_idx) else len(lines) - 1

        # Parse header: tab-separated fields
        hdr = lines[start].lstrip(">")
        fields = hdr.split("\t")

        consensus = fields[0].strip() if len(fields) >= 1 else None
        motif_name = fields[1].strip() if len(fields) >= 2 else None

        try:
            log_odds_threshold = float(fields[2]) if len(fields) >= 3 else None
        except ValueError:
            log_odds_threshold = None

        try:
            log_p_value = float(fields[3]) if len(fields) >= 4 else None
        except ValueError:
            log_p_value = None

        if len(fields) < 2:
            warnings.warn(f"Incomplete HOMER header at line {start + 1}", stacklevel=2)

        # Parse matrix rows
        if start >= end:
            warnings.warn(f"Motif '{consensus}' has no matrix data; skipping", stacklevel=2)
            continue

        mat_lines = [
            ln for ln in lines[start + 1:end + 1] if ln.strip()
        ]

        if not mat_lines:
            warnings.warn(f"Motif '{consensus}' has no matrix data; skipping", stacklevel=2)
            continue

        rows: list[list[float]] = []
        for i, ml in enumerate(mat_lines):
            vals_str = ml.strip().split()
            if len(vals_str) != 4:
                raise ValueError(
                    f"Expected 4 columns at line {start + i + 2} of HOMER file"
                )
            try:
                row = [float(v) for v in vals_str]
            except ValueError as err:
                raise ValueError(
                    f"Non-numeric probability value at line {start + i + 2} "
                    f"of HOMER file"
                ) from err
            rows.append(row)

        mat = _numpy.array(rows, dtype=float)

        # Check for negative values
        if _numpy.any(mat < 0):
            raise ValueError("Negative probability values not allowed")

        # Re-normalize rows if needed
        mat = _renormalize_rows(mat, context=f"motif '{consensus}'")

        motif_id = consensus if consensus else f"motif_{hi + 1}"

        df = _pandas.DataFrame(mat, columns=["A", "C", "G", "T"])
        df.attrs["name"] = motif_name
        df.attrs["consensus"] = consensus
        df.attrs["log_odds_threshold"] = log_odds_threshold
        df.attrs["log_p_value"] = log_p_value
        df.attrs["w"] = int(mat.shape[0])
        df.attrs["source"] = "homer"

        ids.append(motif_id)
        results[motif_id] = df

    if not ids:
        raise ValueError(f"No motifs found in HOMER file {file}")

    deduped = _dedup_motif_ids(ids)
    final: dict[str, _pandas.DataFrame] = {}
    for orig_id, new_id in zip(ids, deduped, strict=False):
        final[new_id] = results[orig_id]
    return final
