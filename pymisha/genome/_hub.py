"""UCSC Hub URL helpers + _hub_fetch_assets.

UCSC's mammal hubs serve flat directories keyed by GenBank/RefSeq accession:
  https://hgdownload.soe.ucsc.edu/hubs/<GCF|GCA>/<NNN>/<NNN>/<NNN>/<acc>/

Unlike R's `.hub_fetch_assets`, we do not parse directory listings. We probe
known filename conventions and treat 404 as "asset not available". This is
faster and avoids HTML scraping; the trade-off is that we can't pick up
non-standard filename variants UCSC may publish (e.g. version-suffixed
GTFs). Users wanting those can fall back to manual.
"""
from __future__ import annotations

import re
import urllib.error
from io import StringIO

import pandas as pd

from ._http import _open_url, _read_url_text

_ACCESSION_RE = re.compile(r"^(GC[FA])_(\d{3})(\d{3})(\d{3})\.\d+$")


def _hub_url_for(accession: str) -> str:
    """Return the hub directory URL for a GenBank/RefSeq accession.

    Parameters
    ----------
    accession : str
        GenBank or RefSeq accession of the form ``GC[FA]_NNNNNNNNN.N``
        (e.g. ``GCA_009914755.4``, ``GCF_000001635.27``).

    Returns
    -------
    str
        Hub directory URL with trailing slash.

    Raises
    ------
    ValueError
        If ``accession`` does not match the expected format.
    """
    m = _ACCESSION_RE.match(accession)
    if not m:
        raise ValueError(
            f"Invalid accession {accession!r}; expected GC[FA]_<9 digits>.<version>"
        )
    gc, a, b, c = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"https://hgdownload.soe.ucsc.edu/hubs/{gc}/{a}/{b}/{c}/{accession}/"


def _hub_chrom_alias_url(accession: str) -> str:
    return f"{_hub_url_for(accession)}{accession}.chromAlias.txt"


def _hub_chrom_sizes_url(accession: str) -> str:
    return f"{_hub_url_for(accession)}{accession}.chrom.sizes.txt"


def _hub_fasta_url(accession: str) -> str:
    return f"{_hub_url_for(accession)}{accession}.fa.gz"


def _hub_rmsk_url(accession: str) -> str:
    return f"{_hub_url_for(accession)}{accession}.repeatMasker.out.gz"


def _hub_cgi_url(accession: str) -> str:
    return f"{_hub_url_for(accession)}{accession}.cpgIslandExt.txt.gz"


def _hub_gtf_url(accession: str, source: str) -> str:
    return f"{_hub_url_for(accession)}genes/{accession}.{source}.gtf.gz"


def _is_http_404(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and exc.code == 404


def _try_fetch(url: str) -> bytes | None:
    """Fetch ``url``; return None on 404, re-raise on other errors."""
    try:
        return _open_url(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _try_fetch_text(url: str) -> str | None:
    """Fetch ``url`` as text; return None on 404, re-raise on other errors."""
    try:
        return _read_url_text(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _parse_hub_chrom_alias(text: str) -> pd.DataFrame:
    """Parse a hub-style chromAlias TSV into a wide DataFrame.

    Hub chromAlias.txt files use the wide format with a `# header` line
    followed by one row per chrom. Header columns are arbitrary (assembly,
    refseq, genbank, ucsc, length).

    If the leading ``#`` header is absent we fall back to the long-format
    parser in :mod:`pymisha.genome._ucsc`, which handles the
    ``alias <TAB> chrom <TAB> source`` shape.
    """
    s = text.lstrip()
    if not s.startswith("#"):
        # Fall back: long format (alias, chrom, source); pivot to wide.
        from ._ucsc import _parse_chrom_alias_tsv as _ucsc_parse
        return _ucsc_parse(text)
    # Wide format: first line is `# col1<TAB>col2<TAB>...`
    lines = s.splitlines()
    header = lines[0][1:].strip()  # strip leading "#"
    header_cols = header.split("\t")
    body = "\n".join(lines[1:])
    if not body.strip():
        return pd.DataFrame(columns=header_cols)
    return pd.read_csv(
        StringIO(body),
        sep="\t",
        header=None,
        names=header_cols,
        dtype=str,
        comment=None,
    )


def _parse_chrom_sizes(text: str) -> pd.DataFrame:
    """Parse a hub chrom.sizes.txt (2-col TSV ``name<TAB>length``)."""
    return pd.read_csv(
        StringIO(text),
        sep="\t",
        header=None,
        names=["name", "length"],
        dtype={"name": str, "length": int},
    )


_DEFAULT_GTF_PRIORITY: tuple[str, ...] = (
    "ncbiRefSeq",
    "refGene",
    "ensGene",
    "augustus",
    "xenoRefGene",
)


def _hub_fetch_assets(
    recipe: dict,
    sets: tuple[str, ...],
    *,
    gtf_priority: tuple[str, ...] = _DEFAULT_GTF_PRIORITY,
) -> dict:
    """Fetch the requested UCSC Hub assets.

    Parameters
    ----------
    recipe : dict
        Normalized recipe; must contain ``accession``.
    sets : tuple[str, ...]
        Which annotation sets to fetch. Recognized: ``"genes"``, ``"rmsk"``,
        ``"cgi"``, ``"cytoband"``. Hub directories do not publish cytoband,
        so ``"cytoband"`` is always ``None``.
    gtf_priority : tuple[str, ...]
        Order in which to try GTF sources under ``genes/``.

    Returns
    -------
    dict with keys:
      - ``chrom_alias`` : pd.DataFrame | None  (with a ``length`` column if available)
      - ``genes``       : bytes | None
      - ``genes_source``: str | None
      - ``rmsk``        : bytes | None
      - ``cgi``         : bytes | None
      - ``cytoband``    : None  (hubs don't ship cytoband)
    """
    accession = recipe["accession"]
    out: dict = {
        "chrom_alias": None,
        "genes": None,
        "genes_source": None,
        "rmsk": None,
        "cgi": None,
        "cytoband": None,
    }

    # chromAlias + chrom.sizes (always best-effort).
    alias_text = _try_fetch_text(_hub_chrom_alias_url(accession))
    sizes_text = _try_fetch_text(_hub_chrom_sizes_url(accession))
    if alias_text is not None:
        alias_df = _parse_hub_chrom_alias(alias_text)
        if sizes_text is not None and "length" not in alias_df.columns:
            sizes = _parse_chrom_sizes(sizes_text)
            # Merge length into alias_df keyed on the column whose values
            # intersect most with sizes["name"].
            best_col = None
            best_overlap = 0
            for col in alias_df.columns:
                overlap = alias_df[col].astype(str).isin(sizes["name"]).sum()
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = col
            if best_col is not None and best_overlap > 0:
                alias_df = alias_df.merge(
                    sizes.rename(columns={"name": best_col}),
                    on=best_col,
                    how="left",
                )
        out["chrom_alias"] = alias_df

    if "genes" in sets:
        for source in gtf_priority:
            url = _hub_gtf_url(accession, source)
            body = _try_fetch(url)
            if body is not None:
                out["genes"] = body
                out["genes_source"] = source
                break

    if "rmsk" in sets:
        out["rmsk"] = _try_fetch(_hub_rmsk_url(accession))

    if "cgi" in sets:
        out["cgi"] = _try_fetch(_hub_cgi_url(accession))

    # cytoband: hubs don't ship it; leave None.

    return out
