"""Backend dispatch for ``gdb_build_genome`` sequence assembly.

Given a normalized recipe dict, ``_build_seq`` routes to the appropriate
backend builder that materializes a misha groot (chrom_sizes.txt + seq/)
at the requested target path.

ucsc-hub builds preserve the original (hub-provided) chrom names. Users
wanting custom target naming can run ``gdb_install_intervals(...,
target_chroms=..., match_by_length=True)`` afterwards, which synthesizes
target_chroms at the alias-translator level.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path


def _build_seq(
    recipe: dict,
    groot: str | Path,
    format: str | None = None,
    verbose: bool = True,
) -> None:
    """Dispatch a recipe to the matching backend builder.

    Parameters
    ----------
    recipe : dict
        Normalized recipe with at least a ``source`` key.
    groot : str or Path
        Target misha groot directory to create.
    format : str, optional
        Database format hint forwarded to backends that build from FASTA
        (e.g. ``"indexed"`` or ``"per-chromosome"``).
    verbose : bool, default True
        If True, backends may emit progress output.
    """
    src = recipe["source"]
    dispatch = {
        "manual": _build_seq_manual,
        "local": _build_seq_local,
        "s3": _build_seq_s3,
        "ucsc-hub": _build_seq_ucsc_hub,
        "ncbi": _build_seq_ncbi,
    }
    if src not in dispatch:
        raise NotImplementedError(f"backend {src!r} not yet implemented")
    dispatch[src](recipe, Path(groot), format, verbose)


def _build_seq_manual(
    recipe: dict, groot: Path, fmt: str | None, verbose: bool
) -> None:
    """Build a groot from an inline FASTA blob in ``recipe['content']``."""
    content = recipe["content"]
    content_bytes = (
        content.encode("utf-8") if isinstance(content, str) else bytes(content)
    )

    # Write FASTA to a tmp file, then call gdb_create.
    # gdb_create creates the groot directory itself.
    with tempfile.NamedTemporaryFile(
        prefix="pymisha-manual-", suffix=".fa", delete=False
    ) as tf:
        tf.write(content_bytes)
        fasta = tf.name
    try:
        from pymisha.db_create import gdb_create
        gdb_create(
            str(groot),
            fasta=fasta,
            db_format=fmt or "indexed",
            verbose=verbose,
        )
    finally:
        with suppress(OSError):
            Path(fasta).unlink(missing_ok=True)


def _build_seq_local(
    recipe: dict, groot: Path, fmt: str | None, verbose: bool
) -> None:
    """Copy an existing on-disk misha groot to ``groot``."""
    src_path = Path(recipe["path"]).expanduser()
    if not (src_path / "chrom_sizes.txt").exists():
        raise ValueError(f"local path is not a misha groot: {src_path}")
    if groot.exists():
        raise FileExistsError(f"target groot already exists: {groot}")
    shutil.copytree(src_path, groot)


def _build_seq_s3(
    recipe: dict, groot: Path, fmt: str | None, verbose: bool
) -> None:
    """Materialize a groot from the misha-genome S3 bucket.

    Delegates to ``_gdb_create_genome_from_s3``. ``recipe['name']`` is the
    canonical S3 key (e.g. ``"hg38"``). The helper extracts into
    ``<dest_dir>/<name>``; if the caller wants the groot at a different
    path, we rename the extracted directory afterward. The helper performs
    download + extract + existence check only; misha session binding is
    the responsibility of the public ``gdb_build_genome`` wrapper.
    """
    from pymisha import db_create as _db_create

    name = recipe["name"]
    expected = groot.parent / name
    _db_create._gdb_create_genome_from_s3(
        name, str(groot.parent), verbose=verbose
    )
    # If user supplied a custom groot path different from <base>/<name>, rename.
    if expected != groot:
        if groot.exists():
            raise FileExistsError(f"target groot already exists: {groot}")
        expected.rename(groot)


def _build_seq_ucsc_hub(
    recipe: dict, groot: Path, fmt: str | None, verbose: bool
) -> None:
    """Build a groot from a UCSC mammal hub.

    Steps:
      1. Download ``<acc>.fa.gz`` from the hub directory.
      2. Decompress to a temporary FASTA file.
      3. Call :func:`gdb_create` to materialize the groot.

    The FASTA's original hub-provided chrom names are preserved. Users
    wanting alternate chrom naming (e.g. dropping ``chr`` prefixes) can
    run ``gdb_install_intervals(..., target_chroms=..., match_by_length=True)``
    afterwards, which synthesizes target_chroms at the alias-translator
    level.

    TODO(C.x): in-place FASTA header rewriting via chromAlias is in the
    design doc; out of scope here.
    """
    from pymisha.db_create import gdb_create

    from ._http import _open_url
    from ._hub import _hub_fasta_url

    accession = recipe["accession"]
    fa_bytes = _open_url(_hub_fasta_url(accession))
    # Decompress to a tmp FASTA file. (Sequence builders expect a plain or .gz path.)
    with tempfile.NamedTemporaryFile(
        prefix=f"pymisha-hub-{accession}-", suffix=".fa", delete=False
    ) as tf:
        tf.write(gzip.decompress(fa_bytes))
        fasta = tf.name
    try:
        gdb_create(
            str(groot),
            fasta=fasta,
            db_format=fmt or "indexed",
            verbose=verbose,
        )
    finally:
        with suppress(OSError):
            Path(fasta).unlink(missing_ok=True)


def _build_seq_ncbi(
    recipe: dict, groot: Path, fmt: str | None, verbose: bool
) -> None:
    """Build a groot from an NCBI Datasets accession.

    Fetches ``GENOME_FASTA`` (the full assembly) via the Datasets v2 API,
    decompresses to a tmp file, and calls :func:`gdb_create`.
    """
    import io
    import zipfile

    from pymisha.db_create import gdb_create

    from ._ncbi import _ncbi_post_download

    accession = recipe["accession"]
    zip_bytes = _ncbi_post_download(accession, ["GENOME_FASTA"])
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        fa_candidates = [
            n for n in zf.namelist()
            if n.endswith((".fna", ".fna.gz", ".fa", ".fa.gz"))
        ]
        if not fa_candidates:
            raise RuntimeError(
                f"NCBI Datasets payload contains no FASTA for {accession}"
            )
        body = zf.read(fa_candidates[0])
    if fa_candidates[0].endswith(".gz"):
        body = gzip.decompress(body)
    with tempfile.NamedTemporaryFile(
        prefix=f"pymisha-ncbi-{accession}-", suffix=".fa", delete=False
    ) as tf:
        tf.write(body)
        fasta = tf.name
    try:
        gdb_create(
            str(groot),
            fasta=fasta,
            db_format=fmt or "indexed",
            verbose=verbose,
        )
    finally:
        with suppress(OSError):
            Path(fasta).unlink(missing_ok=True)
