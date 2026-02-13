"""Expression parsing helpers."""

import re

from . import _shared

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


def _expr_safe_name(name):
    """Return a collision-proof identifier used in expression eval namespaces."""
    return "__pmv_" + name.encode("utf-8").hex()


def _register_expr_name(name, track_names, vtrack_names, used_tracks, used_vtracks, var_map):
    safe = _expr_safe_name(name)
    if name in track_names:
        used_tracks.add(name)
    if name in vtrack_names:
        used_vtracks.add(name)
    var_map[safe] = name
    return safe


def _replace_identifier_token(token, track_names, vtrack_names, used_tracks, used_vtracks, var_map):
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


def _find_vtracks_in_expr(expr):
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


def _parse_expr_vars(expr, track_names, vtrack_names):
    """
    Parse an expression and replace track/vtrack names with safe Python identifiers.

    Returns:
        new_expr, used_tracks, used_vtracks, var_map
    """
    tokens = _TOKEN_RE.findall(expr)

    used_tracks = set()
    used_vtracks = set()
    var_map = {}
    out = []

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
