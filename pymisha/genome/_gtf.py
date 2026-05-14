"""Streaming GTF parser.

GTF is tab-separated, 9-column:
  chrom  source  feature  start  end  score  strand  frame  attributes

`attributes` is `key "value"; key "value";` style.

Coordinates in GTF are 1-based, inclusive (`start` and `end` both inclusive).
We convert to half-open 0-based (start-1, end) on the way out so the output
DataFrame matches misha's interval convention.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from io import BytesIO

import pandas as pd

from ._http import _gunzip_bytes

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')


def _parse_attributes(s: str) -> dict[str, str]:
    """Parse a GTF attribute column. Returns dict of key -> value (first occurrence)."""
    out: dict[str, str] = {}
    for k, v in _ATTR_RE.findall(s):
        if k not in out:
            out[k] = v
    return out


def _iter_gtf_rows(gtf_bytes: bytes) -> Iterator[dict]:
    """Yield one dict per GTF row.

    Skips comment lines (`#`) and empty lines.
    Coordinates converted to half-open 0-based.
    """
    data = _gunzip_bytes(gtf_bytes)
    text_stream = BytesIO(data)
    for raw in text_stream:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        chrom, source, feature, start_s, end_s, _score, strand, _frame, attrs = parts[:9]
        try:
            start_1 = int(start_s)
            end_1 = int(end_s)
        except ValueError:
            continue
        yield {
            "chrom": chrom,
            "source": source,
            "feature": feature,
            "start": start_1 - 1,   # 0-based
            "end": end_1,            # half-open
            "strand": strand,
            "attributes": _parse_attributes(attrs),
        }


def _gtf_to_dataframe(
    gtf_bytes: bytes,
    *,
    feature_filter: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Parse a GTF into a DataFrame in one pass.

    `feature_filter`: if non-None, keep only rows whose `feature` is in the set.
    """
    keep = set(feature_filter) if feature_filter else None
    rows = []
    for r in _iter_gtf_rows(gtf_bytes):
        if keep is not None and r["feature"] not in keep:
            continue
        rows.append(r)
    if not rows:
        return pd.DataFrame(
            columns=["chrom", "source", "feature", "start", "end", "strand", "attributes"]
        )
    return pd.DataFrame(rows)
