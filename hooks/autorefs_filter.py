"""MkDocs hook that suppresses false-positive autorefs warnings.

The mkdocs-autorefs plugin mistakes doctest output like ``['vt2']`` for
Markdown reference links and emits warnings that break ``--strict`` builds.
This hook installs a logging filter to silence those specific messages.
"""

from __future__ import annotations

import logging


class _AutorefsFilter(logging.Filter):
    """Drop autorefs warnings about doctest output references."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Could not find cross-reference target" in msg:
            return False
        return True


_filter = _AutorefsFilter()


def on_startup(**kwargs):  # noqa: ARG001
    # The autorefs plugin logger name follows the pattern:
    # mkdocs.plugins.mkdocs_autorefs._internal.plugin
    # Add filter to the parent loggers as well to catch all paths.
    for name in [
        "mkdocs.plugins.mkdocs_autorefs._internal.plugin",
        "mkdocs.plugins.mkdocs_autorefs",
        "mkdocs.plugins",
        "mkdocs",
    ]:
        logging.getLogger(name).addFilter(_filter)
