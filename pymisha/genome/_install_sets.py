"""Per-set installers for `gdb_install_intervals`.

Each installer assumes `gdb_init(groot)` has been called by the orchestrator
so `pymisha.gintervals_save` writes to the right groot. Installers do NOT
handle overwrite-vs-error themselves - the orchestrator deletes pre-existing
sets via `gintervals_rm` first when `overwrite=True`.

Translator contract: `translator: Callable[[str], str | None]`. Returns the
canonical (groot-side) chrom name, or None if the chrom can't be mapped.
"""
from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

import pandas as pd

from ._gtf import _gtf_to_dataframe
from ._http import _gunzip_bytes

_DEFAULT_GENE_SETS: dict[str, str] = {
    "tss": "tss",
    "exons": "exons",
    "utr3": "utr3",
    "utr5": "utr5",
}

# Map source-specific feature names to the four canonical roles we install.
# Different backends emit different feature column conventions:
#   - Ensembl/GENCODE GTF: transcript / exon / five_prime_utr / three_prime_utr
#   - UCSC ncbiRefSeq.gtf.gz: transcript / exon / 5UTR / 3UTR
#   - NCBI RefSeq GFF3: mRNA / exon / five_prime_UTR / three_prime_UTR
# Canonical role names (`transcript`, `exon`, `five_prime_utr`, `three_prime_utr`)
# are what _install_genes splits on internally.
_FEATURE_CANONICAL: dict[str, str] = {
    "transcript": "transcript",
    "mRNA": "transcript",
    "exon": "exon",
    "five_prime_utr": "five_prime_utr",
    "five_prime_UTR": "five_prime_utr",
    "5UTR": "five_prime_utr",
    "three_prime_utr": "three_prime_utr",
    "three_prime_UTR": "three_prime_utr",
    "3UTR": "three_prime_utr",
}


def _install_genes(
    gtf_bytes: bytes,
    translator: Callable[[str], str | None],
    *,
    gene_sets: dict[str, str] | None = None,
    prefix: str = "",
) -> dict[str, int]:
    """Parse a GTF and write up to 4 misha intervals sets.

    Returns dict of installed_set_name -> row_count.
    Caller must have done `gdb_init(groot)` first. Caller is responsible for
    pre-deleting set names if overwrite is desired.
    """
    import pymisha as pm

    gene_sets = gene_sets if gene_sets is not None else dict(_DEFAULT_GENE_SETS)

    df = _gtf_to_dataframe(
        gtf_bytes,
        feature_filter=tuple(_FEATURE_CANONICAL.keys()),
    )

    # Canonicalize feature names so downstream splits work regardless of
    # whether the source was Ensembl/GENCODE GTF, UCSC GTF, or NCBI GFF3.
    df["feature"] = df["feature"].map(_FEATURE_CANONICAL)
    df = df[df["feature"].notna()].reset_index(drop=True)

    # Translate chroms; drop rows we can't map.
    df["chrom"] = df["chrom"].map(translator)
    df = df[df["chrom"].notna()].reset_index(drop=True)

    installed: dict[str, int] = {}

    def _save(sub: pd.DataFrame, key: str) -> None:
        if key not in gene_sets:
            return
        if sub.empty:
            return
        name = f"{prefix}{gene_sets[key]}"
        # Keep only misha-required columns; ensure ordering.
        out = sub[["chrom", "start", "end"]].copy()
        pm.gintervals_save(out, name)
        installed[name] = len(out)

    # TSS: 1bp interval at transcript start (strand-aware)
    transcripts = df[df["feature"] == "transcript"].copy()
    if not transcripts.empty:
        pos_mask = transcripts["strand"] == "+"
        # Plus strand: start..start+1; minus: end-1..end (half-open)
        tss = transcripts.copy()
        tss.loc[pos_mask, "end"] = tss.loc[pos_mask, "start"] + 1
        tss.loc[~pos_mask, "start"] = tss.loc[~pos_mask, "end"] - 1
        _save(tss, "tss")

    _save(df[df["feature"] == "exon"], "exons")
    _save(df[df["feature"] == "five_prime_utr"], "utr5")
    _save(df[df["feature"] == "three_prime_utr"], "utr3")

    return installed


def _parse_rmsk_out(rmsk_bytes: bytes) -> pd.DataFrame:
    """Parse a RepeatMasker .out file (possibly gzipped) into a DataFrame.

    Returns cols: chrom, start, end, repeat_name, repeat_class.
    Coordinates converted to half-open 0-based.
    Skips the 3-line header and any blank lines.
    """
    data = _gunzip_bytes(rmsk_bytes)
    rows = []
    for raw in BytesIO(data):
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith(("SW", "score", "There")):
            # Header lines.
            continue
        # RepeatMasker .out is whitespace-separated, not tab-separated.
        parts = line.split()
        # Must have at least 11 columns to extract chrom, start, end, name, class.
        if len(parts) < 11:
            continue
        try:
            # 1-based positions in the .out: col 5 = chrom, col 6 = begin,
            # col 7 = end, col 10 = name, col 11 = class/family. Python list
            # indices are 4, 5, 6, 9, 10.
            chrom = parts[4]
            start_1 = int(parts[5])
            end_1 = int(parts[6])
            name = parts[9]
            cls = parts[10]
        except (ValueError, IndexError):
            continue
        rows.append({
            "chrom": chrom,
            "start": start_1 - 1,
            "end": end_1,
            "repeat_name": name,
            "repeat_class": cls,
        })
    if not rows:
        return pd.DataFrame(
            columns=["chrom", "start", "end", "repeat_name", "repeat_class"]
        )
    return pd.DataFrame(rows)


def _install_rmsk(
    rmsk_bytes: bytes,
    translator: Callable[[str], str | None],
    *,
    prefix: str = "",
) -> dict[str, int]:
    """Parse a RepeatMasker .out (.gz allowed) and write `<prefix>rmsk` plus
    per-class subsets `<prefix>rmsk_<class>` for the major classes:
    SINE, LINE, LTR, DNA, Simple_repeat, Low_complexity.

    Returns dict of installed_set_name -> row_count.
    """
    import pymisha as pm

    df = _parse_rmsk_out(rmsk_bytes)
    if df.empty:
        return {}

    df["chrom"] = df["chrom"].map(translator)
    df = df[df["chrom"].notna()].reset_index(drop=True)
    if df.empty:
        return {}

    installed: dict[str, int] = {}

    # Whole rmsk.
    whole_name = f"{prefix}rmsk"
    pm.gintervals_save(df[["chrom", "start", "end"]].copy(), whole_name)
    installed[whole_name] = len(df)

    # Per-class subsets. repeat_class values look like "LINE/L1", "SINE/Alu",
    # "Simple_repeat", "Low_complexity"; take the part before the first slash.
    df["_class"] = df["repeat_class"].str.split("/").str[0]
    for cls in ("SINE", "LINE", "LTR", "DNA", "Simple_repeat", "Low_complexity"):
        sub = df[df["_class"] == cls]
        if sub.empty:
            continue
        name = f"{prefix}rmsk_{cls}"
        pm.gintervals_save(sub[["chrom", "start", "end"]].copy(), name)
        installed[name] = len(sub)

    return installed


def _install_cgi(
    cgi_bytes: bytes,
    translator: Callable[[str], str | None],
    *,
    prefix: str = "",
) -> dict[str, int]:
    """Parse UCSC cpgIslandExt.txt.gz, translate, write `<prefix>cgi`."""
    import pymisha as pm

    data = _gunzip_bytes(cgi_bytes)
    # 11 cols: bin, chrom, chromStart, chromEnd, name, ...
    df = pd.read_csv(
        BytesIO(data),
        sep="\t",
        header=None,
        usecols=[1, 2, 3],
        names=["chrom", "start", "end"],
        dtype={"chrom": str, "start": int, "end": int},
    )
    df["chrom"] = df["chrom"].map(translator)
    df = df[df["chrom"].notna()].reset_index(drop=True)
    if df.empty:
        return {}
    name = f"{prefix}cgi"
    pm.gintervals_save(df, name)
    return {name: len(df)}


def _install_cytoband(
    cytoband_bytes: bytes,
    translator: Callable[[str], str | None],
    *,
    prefix: str = "",
) -> dict[str, int]:
    """Parse UCSC cytoBandIdeo.txt.gz, translate, write `<prefix>cytoband`.

    TODO: band_name and gieStain are dropped here; preserving them would
    require writing an iattr file alongside the intervals set. Deferred
    beyond v0.1.44.
    """
    import pymisha as pm

    data = _gunzip_bytes(cytoband_bytes)
    # 5 cols: chrom, chromStart, chromEnd, name, gieStain.
    df = pd.read_csv(
        BytesIO(data),
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "band_name", "stain"],
        dtype={"chrom": str, "start": int, "end": int, "band_name": str, "stain": str},
    )
    df["chrom"] = df["chrom"].map(translator)
    df = df[df["chrom"].notna()].reset_index(drop=True)
    if df.empty:
        return {}
    name = f"{prefix}cytoband"
    out = df[["chrom", "start", "end"]].copy()
    pm.gintervals_save(out, name)
    return {name: len(out)}
