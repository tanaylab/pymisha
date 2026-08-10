"""Export misha tracks to bedGraph and BigWig formats."""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile

import pandas as pd

from ._shared import _checkroot
from .extract import gextract
from .intervals import gintervals_all


def gtrack_export_bedgraph(
    track: str,
    file: str,
    intervals: pd.DataFrame | None = None,
    iterator: int | None = None,
    name: str | None = None,
    header: bool = True,
) -> None:
    """Export a track or track expression to bedGraph format.

    Evaluates a track expression over the specified genomic intervals and
    writes the result in standard bedGraph format (4-column, tab-separated:
    chrom, start, end, value). NaN values are omitted from the output.

    If the output file path ends in ``.gz``, the output is gzip-compressed.

    Parameters
    ----------
    track : str
        Track name or track expression (e.g. ``"dense_track"`` or
        ``"dense_track * 2"``).
    file : str
        Output file path. If it ends in ``.gz``, output is gzip-compressed.
    intervals : DataFrame or None, optional
        Genomic intervals to export. If ``None`` (default), the entire
        genome is used.
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
    >>> pm.gtrack_export_bedgraph("dense_track", "/tmp/dense.bedgraph")
    """
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
    except Exception:
        # gtrack_info may fail for track expressions or virtual tracks
        pass

    # Check output directory exists
    output_dir = os.path.dirname(file) or "."
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Cannot write to '{file}': directory does not exist"
        )

    if name is None:
        name = track

    # Set up intervals
    if intervals is None:
        intervals = gintervals_all()

    # Extract data
    kwargs: dict = {}
    if iterator is not None:
        kwargs["iterator"] = iterator
    data = gextract(track, intervals=intervals, **kwargs)

    if data is None or len(data) == 0:
        import warnings

        warnings.warn(
            "All values are NaN; output file will contain only the header",
            stacklevel=2,
        )
        data = pd.DataFrame(columns=["chrom", "start", "end", track])

    # Determine value column name (gextract uses the track expression as column name)
    value_col = track
    if value_col not in data.columns:
        # gextract may use a sanitized column name; find the data column
        data_cols = [
            c for c in data.columns if c not in {"chrom", "start", "end", "intervalID"}
        ]
        if data_cols:
            value_col = data_cols[0]
        else:
            # No data columns -- empty result
            import warnings

            warnings.warn(
                "All values are NaN; output file will contain only the header",
                stacklevel=2,
            )
            data = pd.DataFrame(columns=["chrom", "start", "end", track])
            value_col = track

    # Remove rows with NaN values
    data = data.dropna(subset=[value_col])

    if len(data) == 0:
        import warnings

        warnings.warn(
            "All values are NaN; output file will contain only the header",
            stacklevel=2,
        )

    # Sort by chromosome (genome order) then by start
    genome_intervals = gintervals_all()
    chrom_order = {
        chrom: i for i, chrom in enumerate(genome_intervals["chrom"].values)
    }
    data = data.copy()
    data["_chrom_order"] = data["chrom"].map(chrom_order)
    data = data.sort_values(["_chrom_order", "start"]).drop(columns=["_chrom_order"])

    # Write output
    use_gz = file.endswith(".gz")

    opener = gzip.open if use_gz else open
    with opener(file, "wt") as f:
        if header:
            f.write(f'track type=bedGraph name="{name}"\n')

        if len(data) > 0:
            out = data[["chrom", "start", "end", value_col]].astype(
                {"start": "int64", "end": "int64"}
            )
            out.to_csv(f, sep="\t", header=False, index=False)


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

        # bedGraphToBigWig rejects the track header, so never write it
        gtrack_export_bedgraph(
            track, tmp_bedgraph, intervals=intervals, iterator=iterator,
            header=False,
        )

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
