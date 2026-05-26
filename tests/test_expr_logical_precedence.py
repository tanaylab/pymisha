"""Track expressions: R operator precedence for & and |.

In R, the logical operators ``&`` and ``|`` bind *looser* than comparisons, so
``track > a & track < b`` means ``(track > a) & (track < b)``. Python binds ``&``
*tighter* than ``>``/``<``, which mis-parses the expression (and raises a bitwise
ufunc error on floats). pymisha must follow R precedence for track expressions.
"""

import pymisha as pm


def test_and_precedence_matches_parenthesized():
    scope = pm.gintervals(["1", "2"])
    implicit = pm.gscreen("dense_track > 0.05 & dense_track < 0.2", scope)
    explicit = pm.gscreen("(dense_track > 0.05) & (dense_track < 0.2)", scope)
    assert (implicit is None) == (explicit is None)
    if implicit is not None:
        a = implicit.sort_values(["chrom", "start"]).reset_index(drop=True)
        b = explicit.sort_values(["chrom", "start"]).reset_index(drop=True)
        assert a.equals(b)


def test_or_precedence_matches_parenthesized():
    scope = pm.gintervals(["1"])
    implicit = pm.gscreen("dense_track < 0.05 | dense_track > 0.2", scope)
    explicit = pm.gscreen("(dense_track < 0.05) | (dense_track > 0.2)", scope)
    assert (implicit is None) == (explicit is None)
    if implicit is not None:
        a = implicit.sort_values(["chrom", "start"]).reset_index(drop=True)
        b = explicit.sort_values(["chrom", "start"]).reset_index(drop=True)
        assert a.equals(b)


def test_mixed_and_or_precedence():
    # R: & binds tighter than | -> a | b & c == a | (b & c)
    scope = pm.gintervals(["1"])
    implicit = pm.gscreen(
        "dense_track > 0.5 | dense_track > 0.05 & dense_track < 0.1", scope
    )
    explicit = pm.gscreen(
        "(dense_track > 0.5) | ((dense_track > 0.05) & (dense_track < 0.1))", scope
    )
    assert (implicit is None) == (explicit is None)
    if implicit is not None:
        a = implicit.sort_values(["chrom", "start"]).reset_index(drop=True)
        b = explicit.sort_values(["chrom", "start"]).reset_index(drop=True)
        assert a.equals(b)


def test_extract_with_and_expression():
    scope = pm.gintervals(["1"], [0], [200000])
    df = pm.gextract("dense_track > 0.05 & dense_track < 0.2", scope, iterator=50)
    assert df is not None
    vals = df[df.columns[3]].dropna().unique()
    assert set(vals).issubset({0.0, 1.0})
