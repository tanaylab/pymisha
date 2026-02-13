"""Name validation helpers."""

from __future__ import annotations

import re

_DOTTED_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]*$")


def validate_dotted_name(name, kind):
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} must be a non-empty string")
    if not _DOTTED_NAME_RE.fullmatch(name):
        raise ValueError(
            f"Invalid {kind} '{name}'. Must start with a letter and contain "
            "only alphanumeric characters, underscores, and dots."
        )
    parts = name.split(".")
    if any(not part for part in parts):
        raise ValueError(
            f"Invalid {kind} '{name}'. Empty dot-separated components are not allowed."
        )
