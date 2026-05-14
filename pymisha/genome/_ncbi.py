"""NCBI Datasets API + FTP fallback helpers.

This module is stdlib-only. The Datasets API v2 is used for the fast path
(SEQUENCE_REPORT + optional GENOME_GFF / GENOME_FASTA, packaged as a ZIP).
FTP fallback covers older accessions that the Datasets API has not indexed.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import urllib.error
import warnings
import zipfile
from collections.abc import Iterable

import pandas as pd

from ._http import _open_url

_NCBI_DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2"

_ACCESSION_RE = re.compile(r"^(GC[FA])_(\d{9})\.\d+$")

# Valid include values for the Datasets zip download.
NCBI_INCLUDE_VALUES = frozenset({
    "GENOME_FASTA", "GENOME_GFF", "GENOME_GBFF", "RNA_FASTA", "PROT_FASTA",
    "CDS_FASTA", "SEQUENCE_REPORT",
})


def _validate_accession(acc: str) -> None:
    if not _ACCESSION_RE.match(acc):
        raise ValueError(
            f"Invalid NCBI accession {acc!r}; expected GC[FA]_<9 digits>.<version>"
        )


def _datasets_zip_url(accession: str, include: Iterable[str]) -> str:
    _validate_accession(accession)
    inc_list = list(include)
    if not inc_list:
        raise ValueError("include must be non-empty")
    invalid = set(inc_list) - NCBI_INCLUDE_VALUES
    if invalid:
        raise ValueError(f"Invalid include values: {sorted(invalid)}")
    return (
        f"{_NCBI_DATASETS_API}/genome/accession/{accession}"
        f"/download?include_annotation_type={','.join(inc_list)}"
    )


def _datasets_report_url(accession: str) -> str:
    _validate_accession(accession)
    return f"{_NCBI_DATASETS_API}/genome/accession/{accession}/dataset_report"


def _ncbi_post_download(accession: str, include: Iterable[str]) -> bytes:
    """Download the NCBI Datasets ZIP for `accession` with the given includes.

    Returns the raw zip bytes. Raises HTTPError on Datasets failure (caller
    decides whether to fall back to FTP).
    """
    return _open_url(_datasets_zip_url(accession, include))


def _ncbi_dataset_report(accession: str) -> dict:
    """Fetch + parse the Datasets `dataset_report` JSON for `accession`.

    Returns the parsed top-level dict. Raises on network / parse errors.
    """
    body = _open_url(_datasets_report_url(accession))
    parsed: dict = json.loads(body.decode("utf-8"))
    return parsed


def _ncbi_extract_sequence_report(zip_bytes: bytes) -> list[dict]:
    """Extract sequence_report records from a Datasets zip.

    Returns a list of dicts (one per sequence). Returns [] if no sequence
    report is present.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        candidates = [
            n for n in zf.namelist()
            if n.endswith("sequence_report.jsonl")
        ]
        if not candidates:
            return []
        body = zf.read(candidates[0]).decode("utf-8")
    rows: list[dict] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _ncbi_sequence_report(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a Datasets zip into a 5-column chromAlias DataFrame.

    Columns (named to match other backends):
      - refseq:        sequenceAccessionNumber (RefSeq)
      - genbank:       genbankAccession
      - sequence_name: chrName (or empty for unplaced scaffolds)
      - chr_name:      friendly chr name (e.g. "1", "X") for assembled-molecule, else accession
      - ucsc:          UCSC-style ("chr1", "chrX", "chrUn_<acc>")
      - length:        sequence length
    """
    rows = _ncbi_extract_sequence_report(zip_bytes)
    out = []
    for r in rows:
        refseq = r.get("refseqAccession", "") or ""
        gb = r.get("genbankAccession", "") or ""
        seqname = r.get("chrName", "") or ""
        role = r.get("role", "") or ""
        length = r.get("length", 0) or 0
        # Friendly name: chrName for assembled molecules, refseq accession otherwise.
        chr_name = seqname if role == "assembled-molecule" else refseq
        # UCSC-style: chr<name> for assembled-molecule; chrUn_<gbacc_with_v_replaced> otherwise
        if role == "assembled-molecule" and seqname:
            ucsc = f"chr{seqname}"
        elif gb:
            # UCSC uses _v1 suffix for unplaced scaffolds, replacing `.` with `v`
            ucsc = "chrUn_" + gb.replace(".", "v")
        else:
            ucsc = ""
        out.append({
            "refseq": refseq,
            "genbank": gb,
            "sequence_name": seqname,
            "chr_name": chr_name,
            "ucsc": ucsc,
            "length": int(length),
        })
    if not out:
        return pd.DataFrame(columns=["refseq", "genbank", "sequence_name", "chr_name", "ucsc", "length"])
    return pd.DataFrame(out)


def _ncbi_ftp_assembly_dir(accession: str, assembly_name: str) -> str:
    """Build the NCBI FTP directory URL for a (accession, assembly_name) pair.

    Pattern (over HTTPS):
      https://ftp.ncbi.nlm.nih.gov/genomes/all/{GCF|GCA}/<NNN>/<NNN>/<NNN>/<acc>_<asm>
    """
    _validate_accession(accession)
    if not assembly_name:
        raise ValueError("assembly_name must be non-empty")
    prefix = accession[:3]
    digits = accession.split("_", 1)[1].split(".", 1)[0]
    a, b, c = digits[0:3], digits[3:6], digits[6:9]
    return (
        f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{prefix}/{a}/{b}/{c}/{accession}_{assembly_name}"
    )


def _ncbi_assembly_name_from_report(report: dict) -> str:
    """Pull `assembly_info.assembly_name` out of a `dataset_report` JSON."""
    reports = report.get("reports", [])
    if not reports:
        return ""
    return reports[0].get("assembly_info", {}).get("assembly_name", "") or ""


def _ncbi_has_annotation(report: dict) -> bool:
    """True iff the Datasets `dataset_report` says the assembly is annotated."""
    reports = report.get("reports", [])
    if not reports:
        return False
    return bool(reports[0].get("annotation_info", {}).get("provider", "") or "")


def _extract_gff_from_zip(zip_bytes: bytes) -> bytes | None:
    """Find a .gff(.gz) entry in a Datasets zip.

    Returns raw bytes (gunzipped when the entry is .gz), or ``None`` when no
    GFF entry is present in the archive.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        candidates = [
            n for n in zf.namelist()
            if n.endswith((".gff", ".gff.gz"))
        ]
        if not candidates:
            return None
        body = zf.read(candidates[0])
    if candidates[0].endswith(".gz"):
        return gzip.decompress(body)
    return body


def _ncbi_fetch_assets(
    recipe: dict,
    sets: tuple[str, ...],
    *,
    gtf_priority: tuple[str, ...] = (),  # unused for ncbi; kept for signature parity
) -> dict:
    """Fetch NCBI assets for the install path (R 5.6.30 ``409c235e`` parity).

    Datasets fast-path: fetch ``SEQUENCE_REPORT`` (always, for chromAlias) plus
    ``GENOME_GFF`` only when ``genes`` is requested. ``GENOME_FASTA`` is never
    fetched here - the full assembly download is the seq-builder's job
    (:func:`_build_seq_ncbi`). On a human accession this saves ~900 MB per
    install call.

    FTP fallback (R 5.6.30 ``d6cd6047``): when Datasets returns an empty zip
    (no GFF inside), try ``<ftp_dir>/<acc>_<asm>_genomic.gff.gz``. For
    ``rmsk``, NCBI Datasets does not ship a RepeatMasker product; fetch
    ``<ftp_dir>/<acc>_<asm>_rm.out.gz`` directly (404 -> warn, skip).

    ``cgi`` and ``cytoband`` have no NCBI source; the function emits a warning
    and returns ``None`` for those keys.

    Parameters
    ----------
    recipe : dict
        Normalized recipe; must contain ``accession``.
    sets : tuple of str
        Subset of ``{"genes","rmsk","cgi","cytoband"}`` to attempt.
    gtf_priority : tuple of str
        Unused for the NCBI backend; the parameter exists only to match the
        UCSC/UCSC-Hub fetcher signature so the orchestrator can call all three
        the same way.

    Returns
    -------
    dict
        Keys: ``chrom_alias`` (DataFrame or None), ``genes`` (bytes or None),
        ``genes_source`` (str or None), ``rmsk`` (bytes or None), ``cgi``
        (None), ``cytoband`` (None).
    """
    del gtf_priority  # accepted for signature parity; not used.

    accession = recipe["accession"]
    _validate_accession(accession)

    out: dict = {
        "chrom_alias": None,
        "genes": None,
        "genes_source": None,
        "rmsk": None,
        "cgi": None,
        "cytoband": None,
    }

    # Pull dataset_report once if we need assembly_name (for FTP fallback).
    needs_asm = ("genes" in sets) or ("rmsk" in sets)
    asm_name = ""
    if needs_asm:
        try:
            report = _ncbi_dataset_report(accession)
            asm_name = _ncbi_assembly_name_from_report(report)
        except Exception:
            # Tolerated; FTP fetches will warn-and-skip when asm_name unknown.
            asm_name = ""

    # Build the Datasets include list. Always SEQUENCE_REPORT (chromAlias),
    # plus GENOME_GFF when genes is requested. Never GENOME_FASTA here.
    include = ["SEQUENCE_REPORT"]
    if "genes" in sets:
        include.append("GENOME_GFF")

    try:
        zip_bytes = _ncbi_post_download(accession, include)
    except urllib.error.HTTPError as exc:
        # Hard fail: even SEQUENCE_REPORT could not be fetched.
        raise RuntimeError(
            f"NCBI Datasets download failed for {accession}: {exc}"
        ) from exc

    # SEQUENCE_REPORT -> chromAlias DataFrame.
    alias_df = _ncbi_sequence_report(zip_bytes)
    if not alias_df.empty:
        out["chrom_alias"] = alias_df

    # GENOME_GFF (from Datasets zip first, then FTP fallback).
    if "genes" in sets:
        gff_bytes = _extract_gff_from_zip(zip_bytes)
        if gff_bytes is not None:
            out["genes"] = gff_bytes
            out["genes_source"] = "RefSeq"
        elif asm_name:
            ftp_url = (
                f"{_ncbi_ftp_assembly_dir(accession, asm_name)}"
                f"/{accession}_{asm_name}_genomic.gff.gz"
            )
            try:
                # Bytes left gzipped; downstream installers gunzip on demand.
                out["genes"] = _open_url(ftp_url)
                out["genes_source"] = "RefSeq-FTP"
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    warnings.warn(
                        f"'genes' requested but no GFF in Datasets payload "
                        f"and FTP 404 for {accession}; skipping.",
                        stacklevel=2,
                    )
                else:
                    raise
        else:
            warnings.warn(
                f"'genes' requested but no GFF in Datasets payload for "
                f"{accession} and no assembly_name to try FTP; skipping.",
                stacklevel=2,
            )

    # rmsk: FTP only (NCBI Datasets does not ship a RepeatMasker product).
    if "rmsk" in sets:
        if not asm_name:
            warnings.warn(
                f"'rmsk' requested but no assembly_name available for "
                f"{accession}; skipping.",
                stacklevel=2,
            )
        else:
            ftp_url = (
                f"{_ncbi_ftp_assembly_dir(accession, asm_name)}"
                f"/{accession}_{asm_name}_rm.out.gz"
            )
            try:
                # Bytes left gzipped; downstream installers gunzip on demand.
                out["rmsk"] = _open_url(ftp_url)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    warnings.warn(
                        f"NCBI does not publish rm.out.gz for {accession}; "
                        f"skipping rmsk.",
                        stacklevel=2,
                    )
                else:
                    raise

    # cgi / cytoband: NCBI doesn't provide.
    if "cgi" in sets:
        warnings.warn(
            f"'cgi' is not available from NCBI for {accession}; skipping.",
            stacklevel=2,
        )
    if "cytoband" in sets:
        warnings.warn(
            f"'cytoband' is not available from NCBI for {accession}; skipping.",
            stacklevel=2,
        )

    return out
