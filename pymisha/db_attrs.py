"""Database-level read-only track attributes."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from . import _shared
from ._shared import _checkroot

_RO_ATTRS_FILE = ".ro_attributes"
_PY_RO_MAGIC = b"PYMISHA_ROATTRS_V1\0"


def _is_trusted_executable(path: str | Path) -> bool:
    try:
        st = os.stat(path)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    if not os.access(path, os.X_OK):
        return False
    # Reject world/group writable executables as PATH-hijack mitigation.
    if st.st_mode & stat.S_IWOTH:
        return False
    return not st.st_mode & stat.S_IWGRP


def _find_trusted_rscript() -> str | None:
    candidates = [
        Path("/usr/bin/Rscript"),
        Path("/bin/Rscript"),
    ]
    path_hit = shutil.which("Rscript")
    if path_hit:
        candidates.append(Path(path_hit))
    for candidate in candidates:
        candidate = candidate.resolve()
        if _is_trusted_executable(candidate):
            return str(candidate)
    return None


def _readonly_attrs_path() -> Path:
    _checkroot()
    assert _shared._GROOT is not None
    return Path(_shared._GROOT) / _RO_ATTRS_FILE


def _dedupe_and_validate_attrs(attrs: Any) -> list[str]:
    if isinstance(attrs, str):
        values: list[Any] = [attrs]
    else:
        try:
            values = list(attrs)
        except TypeError as exc:
            raise TypeError("attrs must be an iterable of attribute names or None") from exc

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        attr = str(value)
        if attr == "":
            raise ValueError("Attribute name cannot be an empty string")
        if attr in seen:
            raise ValueError(f"Attribute {attr} appears more than once")
        seen.add(attr)
        out.append(attr)

    return out


def _read_python_readonly_format(path: Path) -> list[str] | None:
    raw = path.read_bytes()
    if not raw.startswith(_PY_RO_MAGIC):
        return None

    payload = raw[len(_PY_RO_MAGIC):].decode("utf-8")
    if payload == "":
        return []
    return payload.split("\n")


def _coerce_readonly_obj(obj: Any, filename: str) -> list[str]:
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(
                f"Invalid format of read-only attributes file {filename}"
            )
        values: list[Any] = obj.iloc[:, 0].tolist()
    elif isinstance(obj, pd.Series):
        values = obj.tolist()
    elif isinstance(obj, list | tuple | set):
        values = list(obj)
    else:
        values = [obj]

    attrs: list[str] = []
    seen: set[str] = set()
    for val in values:
        if pd.isna(val):
            continue
        if not isinstance(val, str):
            raise ValueError(
                f"Invalid format of read-only attributes file {filename}"
            )
        if val == "":
            continue
        if val not in seen:
            seen.add(val)
            attrs.append(val)

    return attrs


def _read_r_readonly_format(path: Path) -> list[str]:
    """Read a R-written readonly-attrs file (XDR or gzip RDS character vector).

    Uses the native pymisha R-serialize reader (no Rscript or pyreadr
    dependency at runtime).
    """
    from ._r_serialize import read as _r_read

    filename = str(path)
    try:
        obj = _r_read(filename)
    except Exception as exc:
        raise ValueError(
            f"Invalid format of read-only attributes file {filename}"
        ) from exc

    if obj is None:
        return []
    return _coerce_readonly_obj(obj, filename)


def gdb_get_readonly_attrs() -> list[str] | None:
    """
    Return read-only track attributes for the current database.

    Returns the list of track attribute names that are protected from
    modification or deletion. If no attributes are marked as read-only,
    ``None`` is returned.

    Returns
    -------
    list[str] | None
        List of read-only attribute names, or ``None`` when no read-only
        attributes are configured.

    See Also
    --------
    gdb_set_readonly_attrs : Set the list of read-only attributes.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> result = pm.gdb_get_readonly_attrs()
    >>> result is None or isinstance(result, list)
    True
    """
    _checkroot()
    path = _readonly_attrs_path()
    if not path.exists():
        return None

    attrs = _read_python_readonly_format(path)
    if attrs is None:
        attrs = _read_r_readonly_format(path)

    return attrs or None


def gdb_set_readonly_attrs(attrs: list[str] | tuple[str, ...] | str | None) -> None:
    """
    Set the list of read-only track attributes for the current database.

    Parameters
    ----------
    attrs : list[str] | tuple[str] | str | None
        Attribute names to protect. Pass ``None`` to clear all read-only
        attributes.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an attribute name is empty or appears more than once.

    See Also
    --------
    gdb_get_readonly_attrs : Return the current read-only attributes.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gdb_set_readonly_attrs(["created_by", "creation_date"])  # doctest: +SKIP
    >>> pm.gdb_set_readonly_attrs(None)  # doctest: +SKIP
    """
    _checkroot()
    path = _readonly_attrs_path()

    if attrs is None:
        path.unlink(missing_ok=True)
        return

    attrs = _dedupe_and_validate_attrs(attrs)

    rscript = _find_trusted_rscript()
    if rscript is not None:
        cmd = [
            rscript,
            "-e",
            "args <- commandArgs(TRUE); "
            "f <- args[1]; "
            "attrs <- args[-1]; "
            "con <- file(f, 'wb'); "
            "serialize(attrs, con); "
            "close(con)",
            str(path),
            *attrs,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return

    warnings.warn(
        "Rscript is not available; writing read-only attributes in a "
        "PyMisha-specific fallback format.",
        RuntimeWarning,
        stacklevel=2,
    )
    payload = _PY_RO_MAGIC + "\n".join(attrs).encode("utf-8")
    path.write_bytes(payload)
    return
