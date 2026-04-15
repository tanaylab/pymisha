"""Genome editing functions (ggenome.*)."""

from __future__ import annotations

import logging as _logging
import os as _os
import tempfile as _tempfile
from pathlib import Path as _Path

import pandas as _pd

from ._shared import _checkroot
from .db import _temporary_db_root
from .sequence import gseq_extract

_logger = _logging.getLogger(__name__)

_LINE_WIDTH = 80


def _read_fasta_chroms(fasta_path: str) -> dict[str, str]:
    """Read a FASTA file and return {chrom: sequence} dict.

    Not suitable for very large genomes — loads everything into memory.
    For mammalian genomes (~3 GB) this is fine on current hardware.
    """
    chroms: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current is not None:
                    chroms[current] = "".join(chunks)
                current = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())

    if current is not None:
        chroms[current] = "".join(chunks)

    return chroms


def ggenome_implant(
    intervals: _pd.DataFrame,
    donor: list[str] | str,
    output: str,
    *,
    genome_fasta: str | None = None,
    create_trackdb: bool = False,
    trackdb_path: str | None = None,
    line_width: int = _LINE_WIDTH,
    overwrite: bool = False,
) -> str:
    """Create a genome FASTA with specified regions replaced by donor sequences.

    Takes a reference genome (FASTA file or the current misha database),
    replaces the given intervals with donor sequences, and writes a new
    FASTA file.  Optionally creates a misha trackdb from the result.

    The core FASTA read/perturb/write loop runs in C++ with buffered I/O.

    Parameters
    ----------
    intervals : DataFrame
        Genomic intervals to replace.  Must have ``chrom``, ``start``, ``end``
        columns.
    donor : list of str or str
        Donor sequences to implant.  Either:

        - A list of DNA strings, one per row in *intervals* (same length).
        - A misha database root path — sequences will be extracted from that
          database at the same coordinates.
    output : str
        Path for the output FASTA file.
    genome_fasta : str, optional
        Path to the reference genome FASTA that will be edited.  If ``None``,
        the current misha database genome is exported to a temporary FASTA.
    create_trackdb : bool, default False
        If True, create a misha trackdb alongside the FASTA.
    trackdb_path : str, optional
        Path for the trackdb directory.  Defaults to ``<output_dir>/trackdb``.
    line_width : int, default 80
        Bases per line in the output FASTA.
    overwrite : bool, default False
        If True, overwrite an existing output file.

    Returns
    -------
    str
        Path to the output FASTA file.

    Examples
    --------
    >>> import pymisha as pm
    >>> import tempfile, os
    >>> _ = pm.gdb_init_examples()

    Export the example DB to a reference FASTA:

    >>> ref_fasta = tempfile.mktemp(suffix=".fa")
    >>> _ = pm.gdb_export_fasta(ref_fasta)

    Replace a region with a literal sequence:

    >>> intervs = pm.gintervals(["1"], [100], [110])
    >>> out = tempfile.mktemp(suffix=".fa")
    >>> _ = pm.ggenome_implant(intervs, ["T" * 10], out,
    ...                        genome_fasta=ref_fasta)

    Verify:

    >>> from pymisha.genome_edit import _read_fasta_chroms
    >>> seqs = _read_fasta_chroms(out)
    >>> seqs["1"][100:110]
    'TTTTTTTTTT'
    >>> os.unlink(ref_fasta); os.unlink(out); os.unlink(out + ".fai")

    See Also
    --------
    ggenome_transplant : Sugar for cross-genome sequence swaps.
    """
    # --- validate intervals ------------------------------------------------
    if intervals is None or not isinstance(intervals, _pd.DataFrame):
        raise ValueError("intervals must be a DataFrame with chrom, start, end")
    for col in ("chrom", "start", "end"):
        if col not in intervals.columns:
            raise ValueError(f"intervals must have a '{col}' column")
    if len(intervals) == 0:
        raise ValueError("intervals is empty")

    # --- resolve donor sequences -------------------------------------------
    if isinstance(donor, str):
        # donor is a misha root path — extract sequences
        donor_root = donor
        if genome_fasta is None:
            # We need an active DB to export the reference genome; fail early
            # rather than silently exporting the donor as the target.
            _checkroot()
        _logger.info("Extracting donor sequences from %s", donor_root)
        with _temporary_db_root(donor_root):
            donor_seqs: list[str] = [
                s.upper() for s in gseq_extract(intervals[["chrom", "start", "end"]])
            ]
    elif isinstance(donor, list):
        if len(donor) != len(intervals):
            raise ValueError(
                f"donor list length ({len(donor)}) must match "
                f"intervals length ({len(intervals)})"
            )
        donor_seqs = [s.upper() for s in donor]
    else:
        raise TypeError(
            "donor must be a list of sequence strings or a misha root path"
        )

    # --- validate donor/interval length consistency ------------------------
    for i, (_, row) in enumerate(intervals.iterrows()):
        expected_len = int(row["end"]) - int(row["start"])
        actual_len = len(donor_seqs[i])
        if actual_len != expected_len:
            raise ValueError(
                f"Donor sequence {i} length ({actual_len}) does not match "
                f"interval length ({expected_len}) at "
                f"{row['chrom']}:{row['start']}-{row['end']}. "
                f"Length-changing perturbations are not supported."
            )

    # --- resolve the reference genome --------------------------------------
    output_path = _Path(output).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use overwrite=True."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fasta: str | None = None
    if genome_fasta is None:
        _checkroot()
        from .db import gdb_export_fasta as _export_fasta

        tmp_fd, tmp_fasta = _tempfile.mkstemp(suffix=".fa", prefix="ggenome_ref_")
        _os.close(tmp_fd)
        _export_fasta(tmp_fasta, overwrite=True)
        ref_fasta_path = tmp_fasta
    else:
        fasta_path = _Path(genome_fasta).expanduser()
        if not fasta_path.exists():
            raise FileNotFoundError(f"Genome FASTA not found: {fasta_path}")
        ref_fasta_path = str(fasta_path)

    # --- C++ fast path: read FASTA, apply perturbations, write output + .fai
    try:
        import _pymisha

        fai_entries = _pymisha.pm_ggenome_implant(
            ref_fasta_path,
            str(output_path),
            [str(c) for c in intervals["chrom"]],
            [int(s) for s in intervals["start"]],
            [int(e) for e in intervals["end"]],
            donor_seqs,
            line_width,
        )
    finally:
        if tmp_fasta is not None:
            _os.unlink(tmp_fasta)

    # Validate all interval chromosomes were found
    fai_chroms = {e[0] for e in fai_entries}
    missing = {str(c) for c in intervals["chrom"].unique()} - fai_chroms
    if missing:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"Chromosome(s) not found in reference: {', '.join(sorted(missing))}"
        )

    _logger.info("Wrote %s", output_path)

    # --- optionally create trackdb -----------------------------------------
    if create_trackdb:
        from .db_create import gdb_create as _gdb_create

        if trackdb_path is None:
            trackdb_path = str(output_path.parent / "trackdb")
        tdb = _Path(trackdb_path)
        if tdb.exists():
            import shutil

            shutil.rmtree(tdb)
        _gdb_create(str(tdb), str(output_path))
        _logger.info("Created trackdb: %s", tdb)

    return str(output_path)


def ggenome_transplant(
    intervals: _pd.DataFrame,
    source_genome: str,
    output: str,
    *,
    target_genome: str | None = None,
    create_trackdb: bool = False,
    trackdb_path: str | None = None,
    line_width: int = _LINE_WIDTH,
    overwrite: bool = False,
) -> str:
    """Transplant sequences from one genome to another at given intervals.

    Extracts DNA from *source_genome* at the given *intervals* and implants
    it into *target_genome*, writing a new FASTA.

    This is convenience sugar around :func:`ggenome_implant`.

    Parameters
    ----------
    intervals : DataFrame
        Genomic intervals (chrom, start, end) to transplant.
    source_genome : str
        Misha database root **or** FASTA path to extract donor sequences from.
        If a misha root, sequences are extracted via ``gseq_extract``.
        If a FASTA file (ends with ``.fa`` or ``.fasta``), sequences are read
        directly.
    output : str
        Output FASTA path.
    target_genome : str, optional
        Path to the reference FASTA that will be edited.  If ``None``, the
        current misha database is used.
    create_trackdb : bool, default False
        Create a misha trackdb alongside the output FASTA.
    trackdb_path : str, optional
        Path for the trackdb directory.
    line_width : int, default 80
        Bases per line in the output FASTA.
    overwrite : bool, default False
        Overwrite existing output file.

    Returns
    -------
    str
        Path to the output FASTA file.

    Examples
    --------
    >>> import pymisha as pm
    >>> import tempfile, os
    >>> _ = pm.gdb_init_examples()

    Create a donor DB (all G's) and a reference FASTA:

    >>> donor_fa = tempfile.mktemp(suffix=".fa")
    >>> with open(donor_fa, "w") as f:
    ...     _ = f.write(">1\\n" + "G" * 500000 + "\\n")
    >>> donor_db = tempfile.mktemp()
    >>> _ = pm.gdb_create(donor_db, donor_fa)
    >>> ref_fasta = tempfile.mktemp(suffix=".fa")
    >>> _ = pm.gdb_export_fasta(ref_fasta)

    Transplant donor sequence into chr1:100-200:

    >>> intervs = pm.gintervals(["1"], [100], [200])
    >>> out = tempfile.mktemp(suffix=".fa")
    >>> _ = pm.ggenome_transplant(intervs, donor_db, out,
    ...                           target_genome=ref_fasta)

    Verify:

    >>> from pymisha.genome_edit import _read_fasta_chroms
    >>> seqs = _read_fasta_chroms(out)
    >>> seqs["1"][100:200] == "G" * 100
    True

    >>> import shutil
    >>> os.unlink(donor_fa); shutil.rmtree(donor_db)
    >>> os.unlink(ref_fasta); os.unlink(out); os.unlink(out + ".fai")

    See Also
    --------
    ggenome_implant : The lower-level function with more options.
    """
    source_path = _Path(source_genome).expanduser()

    # Determine if source_genome is a FASTA file or a misha root
    if source_path.is_file() and source_path.suffix in (".fa", ".fasta", ".fna"):
        # Source is a FASTA — read donor sequences directly
        _logger.info("Reading donor sequences from FASTA: %s", source_path)
        ref = _read_fasta_chroms(str(source_path))
        donor_seqs: list[str] = []
        for _, row in intervals.iterrows():
            chrom = str(row["chrom"])
            start = int(row["start"])
            end = int(row["end"])
            if chrom not in ref:
                raise ValueError(
                    f"Chromosome '{chrom}' not found in source FASTA"
                )
            donor_seqs.append(ref[chrom][start:end].upper())
        donor: list[str] | str = donor_seqs
    else:
        # Source is a misha root — ggenome_implant will extract via gseq_extract
        donor = str(source_path)

    return ggenome_implant(
        intervals,
        donor,
        output,
        genome_fasta=target_genome,
        create_trackdb=create_trackdb,
        trackdb_path=trackdb_path,
        line_width=line_width,
        overwrite=overwrite,
    )
