"""Export misha tracks to bedGraph and BigWig formats."""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._log import get_logger, user_stacklevel
from ._shared import _checkroot, _pymisha
from .extract import gextract
from .intervals import (
    gintervals_all,
    gintervals_canonic,
    gintervals_chrom_sizes,
    gintervals_is_bigset,
    gintervals_load,
)

_logger = get_logger(__name__)


def gtrack_export_bedgraph(
    track: str,
    file: str,
    intervals: pd.DataFrame | str | None = None,
    iterator: int | None = None,
    name: str | None = None,
    header: bool = True,
) -> None:
    """Export a track or track expression to bedGraph format.

    Evaluates a track expression over the specified genomic intervals and
    writes the result in standard bedGraph format (4-column, tab-separated:
    chrom, start, end, value). NaN values are omitted from the output. The
    lines are those R misha writes: values as R's ``as.character()`` writes
    them (15 significant digits), coordinates as integers.

    If the output file path ends in ``.gz``, the output is gzip-compressed.

    Parameters
    ----------
    track : str
        Track name or track expression (e.g. ``"dense_track"`` or
        ``"dense_track * 2"``).
    file : str
        Output file path. If it ends in ``.gz``, output is gzip-compressed.
    intervals : DataFrame, str or None, optional
        Genomic intervals to export, or the name of an interval set. If
        ``None`` (default), the entire genome is used. Overlapping intervals
        are merged first, so each base is written once.
    iterator : int or None, optional
        Iterator bin size. If ``None`` (default), the iterator is
        determined automatically from the track expression.
    name : str or None, optional
        Track name for the bedGraph header line. If ``None`` (default),
        uses the ``track`` parameter value.
    header : bool, default True
        Write the ``track type=bedGraph`` header line. Set to ``False``
        for consumers that reject it (e.g. ``bedGraphToBigWig``).

    Returns
    -------
    None
        Called for its side effect of writing a file.

    Raises
    ------
    ValueError
        If the track does not exist or is a 2D track.
    FileNotFoundError
        If the output directory does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_export_bedgraph("dense_track", "/tmp/dense.bedgraph")  # doctest: +SKIP
    """
    _export_bedgraph(track, file, intervals, iterator, name, header)


# Rows formatted in one call to the C++ formatter
_BEDGRAPH_CHUNK_ROWS = 1 << 20


def _export_bedgraph(
    track: str,
    file: str,
    intervals: pd.DataFrame | str | None,
    iterator: int | None,
    name: str | None,
    header: bool,
    chroms: list[str] | None = None,
) -> None:
    """gtrack_export_bedgraph with the chromosomes written in the order of *chroms* (names), genome
    order if None (R misha's .gtrack_export_bedgraph)."""
    _checkroot()

    if not isinstance(track, str):
        raise TypeError("'track' must be a string (track name or expression)")

    if not isinstance(file, str):
        raise TypeError("'file' must be a string (output file path)")

    # Check for 2D tracks
    try:
        from .tracks import gtrack_info

        info = gtrack_info(track)
        if info.get("dimensions") == 2:
            raise ValueError("2D tracks are not supported by bedGraph export")
    except ValueError as e:
        if "2D tracks are not supported" in str(e):
            raise
        # Track expression or virtual track -- that's fine
    except _pymisha.error:
        # gtrack_info may fail for track expressions or virtual tracks
        _logger.debug("no track info for %r; treating it as an expression or virtual track",
                      track, exc_info=True)

    # Check output directory exists
    output_dir = os.path.dirname(file) or "."
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Cannot write to '{file}': directory does not exist"
        )

    if name is None:
        name = track

    # Extract the data. Overlapping intervals would extract the same bins twice, and bedGraph (and
    # bedGraphToBigWig) does not allow overlapping records, so they are merged first.
    kwargs: dict = {}
    if iterator is not None:
        kwargs["iterator"] = iterator

    def extract_canonic(intervals):
        canonic = gintervals_canonic(intervals, unify_touching_intervals=False)
        return None if canonic is None else gextract(track, intervals=canonic, **kwargs)

    if intervals is None:
        data = gextract(track, intervals=gintervals_all(), **kwargs)
    elif isinstance(intervals, str) and gintervals_is_bigset(intervals):
        # a big intervals set is read one chromosome at a time
        parts = [extract_canonic(gintervals_load(intervals, chrom=str(chrom)))
                 for chrom in gintervals_chrom_sizes(intervals)["chrom"]]
        parts = [p for p in parts if p is not None and len(p)]
        data = pd.concat(parts, ignore_index=True) if parts else None
    else:
        data = extract_canonic(intervals)

    # gextract returns None when the intervals hold no values (e.g. a sparse track)
    value_col = track
    if data is None or len(data) == 0:
        warnings.warn(
            "The track has no values in the intervals; output file will contain only the header",
            stacklevel=user_stacklevel(),
        )
        data = None
    else:
        # the value column is named after the track expression
        if value_col not in data.columns:
            value_col = next(c for c in data.columns if c not in {"chrom", "start", "end", "intervalID"})
        if data[value_col].isna().all():
            warnings.warn(
                "All values are NaN; output file will contain only the header",
                stacklevel=user_stacklevel(),
            )

    # Write output
    use_gz = file.endswith(".gz")
    opener = gzip.open if use_gz else open
    with opener(file, "wb") as f:
        if header:
            f.write(f'track type=bedGraph name="{name}"\n'.encode())
        if data is not None:
            _write_bedgraph_lines(f, data, value_col, chroms)


def _write_bedgraph_lines(f: Any, data: pd.DataFrame, value_col: str, chroms: list[str] | None) -> None:
    """Writes the bedGraph lines of the rows of *data* whose value is not NaN to the binary stream
    *f*, rows in genome order (chromosome as in the genome database, or as in *chroms*), then
    start. The lines are R misha's: paste(chrom, sprintf("%.0f", start), sprintf("%.0f", end),
    value, sep = "\t")."""
    genome_chroms = [str(c) for c in gintervals_all()["chrom"]] if chroms is None else chroms
    rank = {c: i for i, c in enumerate(genome_chroms)}
    chrom = pd.Categorical(data["chrom"].astype(str))
    # a chromosome the order does not name sorts last, as NA in R's order()
    chrom_rank = np.array([rank.get(c, len(rank)) for c in chrom.categories], dtype=np.int64)[chrom.codes]
    start = data["start"].to_numpy(dtype=np.int64)
    ord_ = np.lexsort((start, chrom_rank))
    codes = chrom.codes.astype(np.int32)[ord_]
    names = [str(c) for c in chrom.categories]
    start = start[ord_]
    end = data["end"].to_numpy(dtype=np.int64)[ord_]
    value = data[value_col].to_numpy(dtype=np.float64)[ord_]
    for b in range(0, len(value), _BEDGRAPH_CHUNK_ROWS):
        e = b + _BEDGRAPH_CHUNK_ROWS
        f.write(_pymisha.pm_bedgraph_lines(codes[b:e], names, start[b:e], end[b:e], value[b:e]))


def gtrack_export_bigwig(
    track: str,
    file: str,
    intervals: pd.DataFrame | None = None,
    iterator: int | None = None,
) -> None:
    """Export a track or track expression to BigWig format.

    Creates a temporary bedGraph file via :func:`gtrack_export_bedgraph` and
    then converts it to BigWig using the UCSC ``bedGraphToBigWig`` utility.

    Parameters
    ----------
    track : str
        Track name or track expression.
    file : str
        Output file path (typically ending in ``.bw`` or ``.bigwig``).
    intervals : DataFrame or None, optional
        Genomic intervals to export. If ``None`` (default), the entire
        genome is used.
    iterator : int or None, optional
        Iterator bin size. If ``None`` (default), the iterator is
        determined automatically from the track expression.

    Returns
    -------
    None
        Called for its side effect of writing a file.

    Raises
    ------
    RuntimeError
        If ``bedGraphToBigWig`` is not found on PATH or conversion fails.
    ValueError
        If the track is a 2D track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_export_bigwig("dense_track", "/tmp/dense.bw")  # doctest: +SKIP
    """
    _checkroot()

    if not isinstance(track, str):
        raise TypeError("'track' must be a string (track name or expression)")

    if not isinstance(file, str):
        raise TypeError("'file' must be a string (output file path)")

    # Check output directory exists
    output_dir = os.path.dirname(file) or "."
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Cannot write to '{file}': directory does not exist"
        )

    # Locate bedGraphToBigWig converter
    converter = shutil.which("bedGraphToBigWig")
    converter_name = "bedGraphToBigWig"

    if converter is None:
        # Try wigToBigWig as fallback
        converter = shutil.which("wigToBigWig")
        converter_name = "wigToBigWig"

    if converter is None:
        raise RuntimeError(
            "bedGraphToBigWig or wigToBigWig not found. "
            "Install from UCSC tools: https://hgdownload.cse.ucsc.edu/admin/exe/"
        )

    # Create temporary files
    tmp_bedgraph = None
    tmp_chromsizes = None
    try:
        # Write bedGraph to temp file (no header for bedGraphToBigWig)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bedgraph", delete=False
        ) as tmp_bg:
            tmp_bedgraph = tmp_bg.name

        # bedGraphToBigWig rejects the track header, so never write it. It wants the chromosomes
        # sorted by name in the C locale ("10" before "9"), which is not the genome order of a
        # database whose chromosomes are 1, 2, ..., 10.
        c_order = sorted((str(c) for c in gintervals_all()["chrom"]), key=lambda c: c.encode())
        _export_bedgraph(track, tmp_bedgraph, intervals, iterator, None, False, chroms=c_order)

        # Write chrom.sizes
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".chrom.sizes", delete=False
        ) as tmp_cs:
            tmp_chromsizes = tmp_cs.name

        genome_intervals = gintervals_all()
        with open(tmp_chromsizes, "w") as f:
            genome_intervals[["chrom", "end"]].astype({"end": "int64"}).to_csv(
                f, sep="\t", header=False, index=False
            )

        # Run conversion
        result = subprocess.run(
            [converter, tmp_bedgraph, tmp_chromsizes, file],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"{converter_name} failed (exit code {result.returncode}): {stderr}"
            )

    finally:
        # Clean up temp files
        if tmp_bedgraph is not None and os.path.exists(tmp_bedgraph):
            os.unlink(tmp_bedgraph)
        if tmp_chromsizes is not None and os.path.exists(tmp_chromsizes):
            os.unlink(tmp_chromsizes)
