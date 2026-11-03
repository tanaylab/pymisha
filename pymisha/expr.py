"""Expression parsing helpers."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Set as AbstractSet
from types import FrameType
from typing import Any

from . import _shared
from ._log import get_logger
from ._shared import _pymisha

_logger = get_logger(__name__)

_BUILTIN_EXPR_NAMES: set[str] = {"np", "numpy", "CHROM", "START", "END", "True", "False", "None"}
_ALWAYS_ALLOWED_NAMES: set[str] = _BUILTIN_EXPR_NAMES | {"abs", "min", "max", "round", "float", "int", "bool"}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_TOKEN_RE = re.compile(
    r"""
    \s+|
    [A-Za-z_][A-Za-z0-9_.]*|
    \d+(?:\.\d*)?(?:[eE][+-]?\d+)?|
    \.\d+(?:[eE][+-]?\d+)?|
    ==|!=|<=|>=|//|\*\*|
    \S
    """,
    re.VERBOSE,
)


def _expr_safe_name(name: str) -> str:
    """Return a collision-proof identifier used in expression eval namespaces."""
    return "__pmv_" + name.encode("utf-8").hex()


def _register_expr_name(
    name: str,
    track_names: AbstractSet[str],
    vtrack_names: AbstractSet[str],
    used_tracks: set[str],
    used_vtracks: set[str],
    var_map: dict[str, str],
) -> str:
    safe = _expr_safe_name(name)
    if name in track_names:
        used_tracks.add(name)
    if name in vtrack_names:
        used_vtracks.add(name)
    var_map[safe] = name
    return safe


def _replace_identifier_token(
    token: str,
    track_names: AbstractSet[str],
    vtrack_names: AbstractSet[str],
    used_tracks: set[str],
    used_vtracks: set[str],
    var_map: dict[str, str],
) -> str | None:
    if token in track_names or token in vtrack_names:
        return _register_expr_name(
            token, track_names, vtrack_names, used_tracks, used_vtracks, var_map
        )

    # Tokenizer keeps dotted names together. If the full token is unknown,
    # resolve the longest known dotted prefix and preserve the suffix.
    if "." in token:
        parts = token.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in track_names or prefix in vtrack_names:
                safe = _register_expr_name(
                    prefix, track_names, vtrack_names, used_tracks, used_vtracks, var_map
                )
                return safe + token[len(prefix):]

    return None


def _find_vtracks_in_expr(expr: str) -> list[str]:
    """Find virtual track names used in an expression."""
    if not _shared._VTRACKS:
        return []
    known_vtracks = set(_shared._VTRACKS)
    matched = set()

    for token in set(_IDENTIFIER_TOKEN_RE.findall(expr)):
        if token in known_vtracks:
            matched.add(token)
            continue
        if "." in token:
            parts = token.split(".")
            for i in range(len(parts) - 1, 0, -1):
                prefix = ".".join(parts[:i])
                if prefix in known_vtracks:
                    matched.add(prefix)
                    break

    return [name for name in _shared._VTRACKS if name in matched]


def _expr_is_2d(exprs: str | list[str], iterator: Any = None) -> bool:
    """True if the iterator R would use for the expression(s) is 2D.

    That is an explicit 2D iterator - a ``(width, height)`` pair, 2D intervals,
    a cartesian grid, a 2D track or 2D intervals-set name - or, without one,
    a 2D physical track among the identifiers, or the 2D source of a virtual
    track. Used to choose the implicit whole-genome scope (2D
    ``gintervals_2d_all`` vs 1D ``gintervals_all``) when no intervals are
    given: R passes both (.misha$ALLGENOME) and the iterator picks one.
    """
    from .tracks import gtrack_exists, gtrack_info

    def _is_2d_track(name: str) -> bool:
        try:
            return bool(gtrack_exists(name) and int(gtrack_info(name).get("dimensions", 1) or 1) == 2)
        except (_pymisha.error, ValueError):
            # The token is not a track name - the common case for any
            # identifier in an expression.
            _logger.debug("%r is not a resolvable track name", name, exc_info=True)
            return False

    if iterator is not None:
        import pandas as pd

        from ._iterator_policy import CartesianGridSpec

        if isinstance(iterator, CartesianGridSpec):
            return True
        if isinstance(iterator, (tuple, list)) and len(iterator) == 2:
            return True
        if isinstance(iterator, pd.DataFrame):
            return "chrom1" in iterator.columns
        if isinstance(iterator, str):
            if _is_2d_track(iterator):
                return True
            from .intervals import gintervals_exists, gintervals_load

            try:
                if gintervals_exists(iterator):
                    loaded = gintervals_load(iterator)
                    return isinstance(loaded, pd.DataFrame) and "chrom1" in loaded.columns
            except (_pymisha.error, ValueError):
                _logger.debug("%r is not a loadable intervals set", iterator, exc_info=True)
        return False

    names = exprs if isinstance(exprs, (list, tuple)) else [exprs]
    checked: set[str] = set()
    for e in names:
        for token in set(_IDENTIFIER_TOKEN_RE.findall(str(e))):
            candidates = [token]
            if "." in token:
                parts = token.split(".")
                candidates += [".".join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]
            for cand in candidates:
                if cand in checked:
                    continue
                checked.add(cand)
                vt = _shared._VTRACKS.get(cand)
                if vt is not None:
                    src = vt.get("src")
                    if vt.get("dim") is None and isinstance(src, str) and _is_2d_track(src):
                        return True
                    continue
                if _is_2d_track(cand):
                    return True
    return False


def _parse_expr_vars(
    expr: str,
    track_names: AbstractSet[str],
    vtrack_names: AbstractSet[str],
) -> tuple[str, set[str], set[str], dict[str, str]]:
    """
    Parse an expression and replace track/vtrack names with safe Python identifiers.

    Returns:
        new_expr, used_tracks, used_vtracks, var_map
    """
    tokens = _TOKEN_RE.findall(expr)

    used_tracks: set[str] = set()
    used_vtracks: set[str] = set()
    var_map: dict[str, str] = {}
    out: list[str] = []

    for token in tokens:
        if _IDENTIFIER_RE.fullmatch(token):
            replaced = _replace_identifier_token(
                token, track_names, vtrack_names, used_tracks, used_vtracks, var_map
            )
            if replaced is not None:
                out.append(replaced)
                continue
        out.append(token)

    return ''.join(out), used_tracks, used_vtracks, var_map


def _caller_namespace(depth: int = 1) -> dict[str, Any]:
    """Capture the caller's local and global variables.

    Starts at the caller's frame and walks up through frames that share
    the same ``f_globals`` (i.e. the same module).  This covers variables
    defined in enclosing functions — necessary because Python only creates
    closure cells for variables directly referenced in inner bytecode, not
    for variables that appear only inside string expressions.  Limiting the
    walk to same-module frames avoids leaking names from unrelated framework
    frames (pytest, etc.).
    """
    frame = sys._getframe(depth + 1)
    try:
        caller_globals = frame.f_globals
        ns = dict(caller_globals)
        # Walk frames within the same module (same f_globals).
        f: FrameType | None = frame
        chain = []
        while f is not None and f.f_globals is caller_globals:
            chain.append(f.f_locals)
            f = f.f_back
        # Apply outermost to innermost so inner scopes shadow outer ones.
        for locals_dict in reversed(chain):
            ns.update(locals_dict)
        return ns
    finally:
        del frame


def _resolve_user_vars(expr_eval: str, caller_ns: dict[str, Any]) -> dict[str, Any]:
    """Find non-track identifiers in the parsed expression and resolve from caller namespace."""
    tree = ast.parse(expr_eval, mode="eval")
    user_vars = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("__pmv_"):
                continue
            if name in _ALWAYS_ALLOWED_NAMES:
                continue
            if name in caller_ns:
                user_vars[name] = caller_ns[name]
    return user_vars
