"""Restricted pickle loading helpers."""

from __future__ import annotations

import io
import pickle

_ALLOWED_GLOBALS = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "tuple"),
    ("builtins", "str"),
    ("builtins", "bytes"),
    ("builtins", "bytearray"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("builtins", "complex"),
    ("builtins", "slice"),
    ("builtins", "range"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "dtype"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file_obj, allowed_globals):
        super().__init__(file_obj)
        self._allowed_globals = allowed_globals

    def find_class(self, module, name):
        if (module, name) in self._allowed_globals:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Disallowed pickle global '{module}.{name}'"
        )


def restricted_load(file_obj, extra_allowed_globals=None):
    allowed = set(_ALLOWED_GLOBALS)
    if extra_allowed_globals:
        allowed.update(extra_allowed_globals)
    return _RestrictedUnpickler(file_obj, allowed).load()


def restricted_loads(data, extra_allowed_globals=None):
    return restricted_load(io.BytesIO(data), extra_allowed_globals=extra_allowed_globals)
