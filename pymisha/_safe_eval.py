"""AST validation for expression evaluation."""

from __future__ import annotations

import ast

from .expr import _ALWAYS_ALLOWED_NAMES


class UnsafeExpressionError(ValueError):
    """Raised when an expression fails security validation."""


_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Slice,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.keyword,
)

_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)
_ALLOWED_CMPOPS = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)
_ALLOWED_ROOT_ATTRS = {"np", "numpy"}
_ALLOWED_DIRECT_CALLS = {"abs", "min", "max", "round", "float", "int", "bool"}


def _get_attribute_root(node):
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return cur


def _validate_attribute(node):
    if node.attr.startswith("_"):
        raise UnsafeExpressionError("Attributes starting with '_' are not allowed")

    root = _get_attribute_root(node)
    if not isinstance(root, ast.Name) or root.id not in _ALLOWED_ROOT_ATTRS:
        raise UnsafeExpressionError(
            "Attribute access is only allowed on numpy module aliases (np/numpy)"
        )


def _validate_call(node):
    fn = node.func
    if isinstance(fn, ast.Name):
        if fn.id not in _ALLOWED_DIRECT_CALLS:
            raise UnsafeExpressionError(f"Function '{fn.id}' is not allowed")
        return

    if isinstance(fn, ast.Attribute):
        _validate_attribute(fn)
        return

    raise UnsafeExpressionError("Only direct function calls are allowed")


def validate_expression_ast(expr, allowed_names):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression syntax: {exc}") from exc

    allowed_names = set(allowed_names)

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeExpressionError(
                f"Unsupported expression construct: {type(node).__name__}"
            )

        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in _ALWAYS_ALLOWED_NAMES:
            raise UnsafeExpressionError(f"Unknown identifier '{node.id}' in expression")

        if isinstance(node, ast.Attribute):
            _validate_attribute(node)

        if isinstance(node, ast.Call):
            _validate_call(node)

        if isinstance(node, ast.BinOp) and not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError(
                f"Operator '{type(node.op).__name__}' is not allowed"
            )

        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError(
                f"Unary operator '{type(node.op).__name__}' is not allowed"
            )

        if isinstance(node, ast.BoolOp) and not isinstance(node.op, _ALLOWED_BOOLOPS):
            raise UnsafeExpressionError(
                f"Boolean operator '{type(node.op).__name__}' is not allowed"
            )

        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, _ALLOWED_CMPOPS):
                    raise UnsafeExpressionError(
                        f"Comparison operator '{type(op).__name__}' is not allowed"
                    )


def _split_top_level(s, op):
    """Split ``s`` on top-level occurrences of single-char operator ``op``.

    Occurrences inside parentheses/brackets/braces or string literals are not
    split.
    """
    parts = []
    depth = 0
    instr = None
    last = 0
    for i, c in enumerate(s):
        if instr is not None:
            if c == instr:
                instr = None
        elif c in "\"'":
            instr = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and c == op:
            parts.append(s[last:i])
            last = i + 1
    parts.append(s[last:])
    return parts


def _recurse_into_parens(atom):
    """Reassociate &/| inside top-level parenthesized groups of ``atom``.

    ``atom`` has no top-level ``&``/``|``; recurse so user-written parentheses
    like ``(a > x & b < y)`` are fixed too.
    """
    res = []
    instr = None
    i = 0
    n = len(atom)
    while i < n:
        c = atom[i]
        if instr is not None:
            res.append(c)
            if c == instr:
                instr = None
            i += 1
        elif c in "\"'":
            instr = c
            res.append(c)
            i += 1
        elif c == "(":
            depth = 1
            j = i + 1
            while j < n and depth:
                if atom[j] == "(":
                    depth += 1
                elif atom[j] == ")":
                    depth -= 1
                j += 1
            inner = atom[i + 1:j - 1]
            res.append("(" + _reassociate_logical(inner) + ")")
            i = j
        else:
            res.append(c)
            i += 1
    return "".join(res)


def _reassociate_logical(expr):
    out_or = []
    for or_part in _split_top_level(expr, "|"):
        and_parts = _split_top_level(or_part, "&")
        if len(and_parts) == 1:
            out_or.append(_recurse_into_parens(or_part.strip()))
        else:
            out_or.append(" & ".join(f"({_recurse_into_parens(p.strip())})" for p in and_parts))
    if len(out_or) == 1:
        return out_or[0]
    return " | ".join(f"({p})" for p in out_or)


def _normalize_logical_precedence(expr):
    """Reparenthesize ``&`` / ``|`` to R operator precedence.

    In R, ``&`` and ``|`` bind *looser* than comparisons (and ``&`` tighter than
    ``|``), so ``track > a & track < b`` means ``(track > a) & (track < b)``.
    Python binds ``&``/``|`` *tighter* than comparisons, which mis-parses such
    expressions (and raises a bitwise ufunc error on floats). Reassociate the
    operands of ``|`` then ``&`` (recursing into parentheses) to restore R
    precedence. Track expressions are R, so this is the intended semantics.
    """
    if "&" not in expr and "|" not in expr:
        return expr
    # R's scalar && / || are element-wise here; treat them as & / |.
    expr = expr.replace("&&", "&").replace("||", "|")
    return _reassociate_logical(expr)


def compile_safe_expression(expr, allowed_names):
    expr = _normalize_logical_precedence(expr)
    validate_expression_ast(expr, allowed_names)
    return compile(expr, "<string>", "eval")
