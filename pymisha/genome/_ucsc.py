"""UCSC Golden Path URL helpers + _ucsc_fetch_assets.

NOTE on R parity: R's ``.ucsc_fetch_assets`` fetches genePred from
``<base>/database/ncbiRefSeq.txt.gz`` (no chromAlias from UCSC for this
backend; computed locally). We diverge: pymisha fetches GTF from
``<base>/bigZips/genes/<priority>.gtf.gz`` and chromAlias from
``<base>/bigZips/<assembly>.chromAlias.txt`` directly. This is simpler and
covers all assemblies UCSC publishes GTFs for (hg19, hg38, mm10, mm39, ...).
Older or niche assemblies without GTF may not work via this backend; users
can fall back to ``manual`` or ``ncbi``.

URL layout (all rooted at ``https://hgdownload.soe.ucsc.edu/goldenPath``):

- ``<asm>/bigZips/<asm>.fa.gz``                       FASTA
- ``<asm>/bigZips/<asm>.chromAlias.txt``              chromAlias (wide or long)
- ``<asm>/bigZips/genes/<priority>.gtf.gz``           gene models (GTF)
- ``<asm>/bigZips/<asm>.fa.out.gz``                   RepeatMasker .out
- ``<asm>/database/cpgIslandExt.txt.gz``              CpG islands (TSV dump)
- ``<asm>/database/cytoBandIdeo.txt.gz``              cytobands (TSV dump)

CGI and cytoband only live under ``database/`` on UCSC's mirror, hence the
asymmetry with the bigZips-rooted assets above.
"""
from __future__ import annotations

from collections.abc import Iterable
from io import StringIO

import pandas as pd

from ._http import _open_url, _read_url_text

_UCSC_BASE = "https://hgdownload.soe.ucsc.edu/goldenPath"


def _ucsc_fasta_url(assembly: str) -> str:
    return f"{_UCSC_BASE}/{assembly}/bigZips/{assembly}.fa.gz"


def _ucsc_chrom_alias_url(assembly: str) -> str:
    return f"{_UCSC_BASE}/{assembly}/bigZips/{assembly}.chromAlias.txt"


def _ucsc_gtf_url(assembly: str, priority: str) -> str:
    return f"{_UCSC_BASE}/{assembly}/bigZips/genes/{priority}.gtf.gz"


def _ucsc_rmsk_url(assembly: str) -> str:
    # RepeatMasker .out format (15-ish cols). R uses database/rmsk.txt.gz
    # (UCSC 17-col format) instead; downstream parser (C.2.4) must know
    # which format to expect.
    return f"{_UCSC_BASE}/{assembly}/bigZips/{assembly}.fa.out.gz"


def _ucsc_cgi_url(assembly: str) -> str:
    return f"{_UCSC_BASE}/{assembly}/database/cpgIslandExt.txt.gz"


def _ucsc_cytoband_url(assembly: str) -> str:
    return f"{_UCSC_BASE}/{assembly}/database/cytoBandIdeo.txt.gz"


_DEFAULT_GTF_PRIORITY: tuple[str, ...] = (
    "ncbiRefSeq",
    "refGene",
    "ensGene",
    "augustus",
    "xenoRefGene",
)


def _parse_chrom_alias_tsv(text: str) -> pd.DataFrame:
    """Parse UCSC chromAlias TSV into a wide DataFrame.

    Handles two on-disk formats:

    1. **Wide** (``bigZips/<asm>.chromAlias.txt``): one row per chrom, header
       prefixed with ``#``, e.g.::

           # sequenceName\\talias names\\tUCSC database: hg38
           chr1\\t1\\tCM000663.2\\tNC_000001.11

       The header's column names after ``sequenceName`` are descriptive
       ("alias names", "UCSC database: ..."); we assign generic
       ``alias_0``, ``alias_1``, ... since UCSC does not tag each column
       with its source in this format.

    2. **Long** (``database/chromAlias.txt.gz``): three cols (alias, chrom,
       source), no header. Pivoted to one row per chrom with one column
       per source.

    Detection: leading ``#`` => wide; else long.

    Returns a DataFrame keyed on ``chrom`` with one column per alias source.
    """
    if not text:
        return pd.DataFrame(columns=["chrom"])

    stripped = text.lstrip()
    if stripped.startswith("#"):
        # Wide format. The first line is the header; the column names after
        # the leading ``# sequenceName`` are descriptive labels, not source
        # tags, so we assign generic ``alias_0..N-1`` names.
        lines = stripped.splitlines()
        header = lines[0].lstrip("#").strip().split("\t")
        n_cols = len(header)
        names = ["chrom"] + [f"alias_{i}" for i in range(n_cols - 1)]
        body = "\n".join(lines[1:])
        if not body.strip():
            return pd.DataFrame(columns=names)
        return pd.read_csv(
            StringIO(body),
            sep="\t",
            header=None,
            names=names,
            dtype=str,
            comment=None,
        )

    # Long format: alias <TAB> chrom <TAB> source, no header.
    long = pd.read_csv(
        StringIO(text),
        sep="\t",
        header=None,
        names=["alias", "chrom", "source"],
        dtype=str,
        comment=None,
    )
    # Pivot to wide. Some rows have multiple aliases per (chrom, source) -
    # keep the first.
    wide = long.pivot_table(
        index="chrom", columns="source", values="alias", aggfunc="first"
    )
    return wide.reset_index().rename_axis(columns=None)


def _fetch_gtf_with_priority(
    assembly: str,
    priorities: Iterable[str],
) -> tuple[bytes, str]:
    """Try each GTF source in order; return ``(raw_bytes, source_name)`` on first success.

    Raises ``FileNotFoundError`` if every priority 404s (or otherwise fails).
    """
    last_exc: Exception | None = None
    priorities_list = list(priorities)
    for source in priorities_list:
        url = _ucsc_gtf_url(assembly, source)
        try:
            return _open_url(url), source
        except Exception as exc:
            last_exc = exc
            continue
    raise FileNotFoundError(
        f"No GTF found for {assembly} from priorities {priorities_list}: {last_exc}"
    )


def _ucsc_fetch_assets(
    recipe: dict,
    sets: tuple[str, ...],
    *,
    gtf_priority: tuple[str, ...] = _DEFAULT_GTF_PRIORITY,
) -> dict:
    """Fetch the requested UCSC assets for ``recipe['assembly']``.

    Parameters
    ----------
    recipe : dict
        Must contain ``assembly``.
    sets : tuple[str, ...]
        Which annotation sets to fetch. Recognized: ``"genes"``, ``"rmsk"``,
        ``"cgi"``, ``"cytoband"``. Unknown values are ignored silently.
    gtf_priority : tuple[str, ...]
        Order in which to try GTF sources under ``bigZips/genes/``.

    Returns
    -------
    dict with keys:
      - ``chrom_alias`` : pd.DataFrame | None
      - ``genes``       : bytes (gzipped GTF) | None
      - ``genes_source``: str | None  (which priority hit)
      - ``rmsk``        : bytes | None
      - ``cgi``         : bytes | None
      - ``cytoband``    : bytes | None

    ``chrom_alias`` is fetched best-effort regardless of ``sets`` since the
    translator needs it; a missing alias file is recorded as ``None`` rather
    than raised so the caller can still install with
    ``match_by_length=False``.
    """
    assembly = recipe["assembly"]
    out: dict = {
        "chrom_alias": None,
        "genes": None,
        "genes_source": None,
        "rmsk": None,
        "cgi": None,
        "cytoband": None,
    }

    try:
        text = _read_url_text(_ucsc_chrom_alias_url(assembly))
        out["chrom_alias"] = _parse_chrom_alias_tsv(text)
    except Exception:
        # chromAlias is optional at fetch time.
        out["chrom_alias"] = None

    if "genes" in sets:
        gtf_bytes, src = _fetch_gtf_with_priority(assembly, gtf_priority)
        out["genes"] = gtf_bytes
        out["genes_source"] = src

    if "rmsk" in sets:
        out["rmsk"] = _open_url(_ucsc_rmsk_url(assembly))

    if "cgi" in sets:
        out["cgi"] = _open_url(_ucsc_cgi_url(assembly))

    if "cytoband" in sets:
        out["cytoband"] = _open_url(_ucsc_cytoband_url(assembly))

    return out
