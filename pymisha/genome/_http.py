"""Stdlib HTTP fetchers with retry + gunzip support.

Used by the genome backends. Deliberately small: just enough surface for
the UCSC / NCBI / Hub fetchers to retrieve assets. No new third-party
dependencies.
"""
from __future__ import annotations

import gzip
import time
import urllib.error
import urllib.request


def _open_url(url: str, *, timeout: float = 60.0, retries: int = 3) -> bytes:
    """Fetch ``url`` with exponential backoff.

    Parameters
    ----------
    url : str
    timeout : float
        Per-attempt timeout in seconds.
    retries : int
        Maximum number of attempts (including the first one). Must be >= 1.

    Returns
    -------
    bytes
        Raw response body. No decompression is applied here; see
        ``_gunzip_bytes`` and ``_read_url_text``.

    Raises
    ------
    URLError, HTTPError, OSError
        The exception from the final attempt is re-raised.
    """
    if retries < 1:
        raise ValueError(f"retries must be >= 1, got {retries}")
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body: bytes = resp.read()
                return body
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"unreachable: {last_exc}")  # pragma: no cover


def _gunzip_bytes(buf: bytes) -> bytes:
    """Decompress ``buf`` if it's gzip-framed; otherwise return as-is."""
    if len(buf) >= 2 and buf[:2] == b"\x1f\x8b":
        return gzip.decompress(buf)
    return buf


def _read_url_text(url: str, **kw) -> str:
    """Fetch + (optional) gunzip + decode as UTF-8.

    Convenience wrapper for text-ish responses (chromAlias TSV, cytoband, etc.).
    """
    return _gunzip_bytes(_open_url(url, **kw)).decode("utf-8", errors="replace")
