"""Export misha tracks to bedGraph and BigWig formats."""

from __future__ import annotations

import gzip
import os
import warnings
from collections.abc import Callable
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

    # Extract the data. Overlapping intervals would extract the same bins twice, and bedGraph does
    # not allow overlapping records, so they are merged first.
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
            _write_bedgraph_lines(f, data, value_col)


def _write_bedgraph_lines(f: Any, data: pd.DataFrame, value_col: str) -> None:
    """Writes the bedGraph lines of the rows of *data* whose value is not NaN to the binary stream
    *f*, rows in genome order (chromosome as in the genome database), then start. The lines are R
    misha's: paste(chrom, sprintf("%.0f", start), sprintf("%.0f", end), value, sep = "\t")."""
    genome_chroms = [str(c) for c in gintervals_all()["chrom"]]
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
    intervals: pd.DataFrame | str | None = None,
    iterator: int | None = None,
) -> None:
    """Export a track or track expression to BigWig format.

    The track expression is evaluated over the intervals as by :func:`gextract`, one batch of
    chromosomes at a time, and the values are written to the file as they are extracted, so the
    whole genome's values are never held in memory. Overlapping intervals are merged first, so
    each base is written once. NaN values are not written. Values are stored as 32-bit floats, as
    in every bigWig file.

    The file is written by pymisha itself (the same C++ writer as R misha's); no external tool is
    needed. Runs of bins of equal size (as from a dense track or a numeric iterator) are stored as
    fixedStep sections, other values as bedGraph sections. The file has zoom levels (summaries at
    coarser resolutions), which genome browsers and UCSC's ``bigWigSummary`` use for zoomed-out
    views. Chromosomes are written in the order of their names (as the bigWig format requires),
    and every chromosome of the genome is listed in the file, also those without values.

    Parameters
    ----------
    track : str
        Track name or track expression.
    file : str
        Output file path (typically ending in ``.bw`` or ``.bigwig``).
    intervals : DataFrame, str or None, optional
        Genomic intervals to export, or the name of an interval set. If ``None`` (default), the
        entire genome is used.
    iterator : int or None, optional
        Iterator bin size. If ``None`` (default), the iterator is
        determined automatically from the track expression.

    Returns
    -------
    None
        Called for its side effect of writing a file.

    Raises
    ------
    ValueError
        If the track is a 2D track.
    FileNotFoundError
        If the output directory does not exist.

    Examples
    --------
    >>> import os, tempfile
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> f = os.path.join(tempfile.mkdtemp(), "dense.bw")
    >>> pm.gtrack_export_bigwig("dense_track", f)
    >>> pm.gintervals_import_bigwig(f, pm.gintervals("1", 0, 100))
      chrom  start  end     value
    0  chr1      0   50  0.177778
    1  chr1     50  100  0.160000
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

    # Check for 2D tracks
    try:
        from .tracks import gtrack_info

        info = gtrack_info(track)
        if info.get("dimensions") == 2:
            raise ValueError("2D tracks are not supported by BigWig export")
    except ValueError as e:
        if "2D tracks are not supported" in str(e):
            raise
    except _pymisha.error:
        _logger.debug("no track info for %r; treating it as an expression or virtual track",
                      track, exc_info=True)

    # The chromosomes in the order of the file: by name, byte by byte, as the bigWig format's
    # chromosome tree is keyed ("1", "10", "2"), which is not the genome order of a database whose
    # chromosomes are 1, 2, ..., 10. Chromosome id i of the file is file_chroms[i].
    genome = gintervals_all()
    genome_chroms = [str(c) for c in genome["chrom"]]
    order = sorted(range(len(genome_chroms)), key=lambda i: genome_chroms[i].encode())
    file_chroms = [genome_chroms[i] for i in order]
    file_sizes = genome["end"].to_numpy(dtype=np.int64)[order]
    genome = pd.DataFrame({"chrom": file_chroms, "start": 0, "end": file_sizes})

    groups = _bigwig_batches(intervals, iterator, genome, file_chroms)

    kwargs: dict = {}
    if iterator is not None:
        kwargs["iterator"] = iterator
    rank = {c: i for i, c in enumerate(file_chroms)}

    # Created before the try, so a failure before the file is opened (bad chromosome sizes, an
    # unwritable path) never deletes a file already at the path.
    state = _pymisha.pm_bigwig_create(file, file_chroms, file_sizes)
    success = False
    try:
        for group in groups:
            for intervs in group():
                data = gextract(track, intervals=intervs, colnames="value", **kwargs)
                if data is None or not len(data):
                    continue
                chrom = pd.Categorical(data["chrom"].astype(str))
                cat_ids = np.array([rank.get(c, -1) for c in chrom.categories], dtype=np.int64)
                ids = cat_ids[chrom.codes]
                start = data["start"].to_numpy(dtype=np.int64)
                rows = np.lexsort((start, ids))
                state = _pymisha.pm_bigwig_append(
                    file, state, ids[rows], start[rows], data["end"].to_numpy(dtype=np.int64)[rows],
                    data["value"].to_numpy(dtype=np.float64)[rows],
                )
        num_records = _pymisha.pm_bigwig_finish(file, state)
        success = True
    finally:
        if not success and os.path.exists(file):
            os.unlink(file)

    if num_records == 0:
        warnings.warn(
            "The track has no values in the intervals; the bigWig file holds no data",
            stacklevel=user_stacklevel(),
        )


def _bigwig_batches(
    intervals: pd.DataFrame | str | None,
    iterator: Any,
    genome: pd.DataFrame,
    file_chroms: list[str],
) -> list[Callable[[], list[pd.DataFrame]]]:
    """The scope of a bigWig export, in the order of the file (by the rank of the chromosome in
    *file_chroms*, then start), as a list of functions that each return a list of batches of
    intervals: one function for an intervals set read one chromosome at a time, one in all
    otherwise. Overlapping intervals are merged. A batch holds whole chromosomes of about 1e8
    bases in all (an interval is not cut); with a numeric iterator, about 2^22 bins, the intervals
    being cut at multiples of 2^22 bins (bins do not cross them, so the values are those of the
    whole interval). R misha's .gbigwig_batches."""
    batch_bases = int(1e8)
    cut = (isinstance(iterator, (int, float, np.integer, np.floating)) and not isinstance(iterator, bool)
           and np.isfinite(iterator) and iterator >= 1 and iterator == int(iterator))
    if cut:
        batch_bases = int(iterator) * (1 << 22)
    rank = {c: i for i, c in enumerate(file_chroms)}

    def split_batches(intervs: pd.DataFrame | None) -> list[pd.DataFrame]:
        if intervs is None or not len(intervs):
            return []
        chrom = intervs["chrom"].astype(str).to_numpy()
        start = intervs["start"].to_numpy(dtype=np.int64)
        end = intervs["end"].to_numpy(dtype=np.int64)
        ids = np.array([rank.get(c, len(rank)) for c in chrom], dtype=np.int64)
        o = np.lexsort((start, ids))
        chrom, start, end = chrom[o], start[o], end[o]
        if cut:
            first = start // batch_bases
            n = (end - 1) // batch_bases - first + 1
            if (n > 1).any():
                idx = np.repeat(np.arange(len(start)), n)
                k = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)
                piece = first[idx] + k
                chrom = chrom[idx]
                start = np.maximum(start[idx], piece * batch_bases)
                end = np.minimum(end[idx], (piece + 1) * batch_bases)
        widths = end - start
        batch = (np.cumsum(widths) - widths) // batch_bases
        bounds = np.flatnonzero(np.diff(batch)) + 1
        parts = zip(np.split(chrom, bounds), np.split(start, bounds), np.split(end, bounds), strict=True)
        return [pd.DataFrame({"chrom": c, "start": s, "end": e}) for c, s, e in parts]

    if intervals is None:
        return [lambda: split_batches(genome)]

    if isinstance(intervals, str) and gintervals_is_bigset(intervals):
        # a big intervals set is read one chromosome at a time
        chroms = [str(c) for c in gintervals_chrom_sizes(intervals)["chrom"]]
        chroms.sort(key=lambda c: rank.get(c, len(rank)))

        def chrom_batches(chrom: str) -> Callable[[], list[pd.DataFrame]]:
            return lambda: split_batches(gintervals_canonic(gintervals_load(intervals, chrom=chrom),
                                                            unify_touching_intervals=False))

        return [chrom_batches(chrom) for chrom in chroms]

    canonic = gintervals_canonic(intervals, unify_touching_intervals=False)
    return [lambda: split_batches(canonic)]
