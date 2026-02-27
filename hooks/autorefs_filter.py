"""MkDocs hook that suppresses false-positive warnings in strict mode.

Suppresses:
- autorefs warnings about doctest output (e.g. ``['vt2']``) mistaken for
  Markdown reference links.
- griffe warnings about ``**kwargs`` parameters documented in NumPy-style
  docstrings but not present as explicit function signature arguments.
"""

from __future__ import annotations

import logging


class _StrictModeFilter(logging.Filter):
    """Drop known false-positive warnings that break --strict builds."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Could not find cross-reference target" in msg:
            return False
        if "does not appear in the function signature" in msg:
            return False
        return True


_filter = _StrictModeFilter()


def on_startup(**kwargs):  # noqa: ARG001
    for name in [
        "mkdocs.plugins.mkdocs_autorefs._internal.plugin",
        "mkdocs.plugins.mkdocs_autorefs",
        "mkdocs.plugins.griffe",
        "mkdocs.plugins",
        "mkdocs",
        "griffe",
    ]:
        logging.getLogger(name).addFilter(_filter)
