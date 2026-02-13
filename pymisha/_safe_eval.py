"""AST validation for expression evaluation."""

from __future__ import annotations

import ast


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

        if isinstance(node, ast.Name) and node.id not in allowed_names:
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


def compile_safe_expression(expr, allowed_names):
    validate_expression_ast(expr, allowed_names)
    return compile(expr, "<string>", "eval")
