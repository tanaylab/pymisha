"""Parity port of R misha ``test-giterator.cartesian_grid.R``.

R builds a 2D cartesian-grid *iterator* from 1D interval centers + expansion
breakpoints, then enumerates the resulting cells with
``giterator.intervals(expr, scope, iterator = itr[, band = ...])``.

All 6 cases pass. ``giterator_intervals`` consumes a ``CartesianGridSpec``
iterator and delegates to the C++ iterator port
(``PMTrackExpressionCartesianGridIterator``), which de-duplicates grid-point
centers, clips adjacent-center expansions at their midpoints, intersects each
cell with the 2D scope, and applies the diagonal ``band``. R's
``.misha$ALLGENOME`` 2D scope maps to ``gintervals_2d_all(mode="full")``.

Pass: 6 / xfail: 0.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

_EXP1 = [-100000, -50000, -10000, 20000, 700000]
_EXP2 = [-200000, -30000, -10000, 60000, 100000, 200000]


def _scope():
    return pm.gintervals([1, 2, 3])


def _grid(screen1, screen2=None, *, min_band_idx=None, max_band_idx=None):
    """Build the cartesian-grid iterator spec exactly as the R test does."""
    i1 = pm.gscreen(screen1, _scope())
    i2 = None if screen2 is None else pm.gscreen(screen2, _scope())
    e2 = None if screen2 is None else _EXP2
    return pm.giterator_cartesian_grid(
        i1, _EXP1, i2, e2,
        min_band_idx=min_band_idx, max_band_idx=max_band_idx,
        stream=True,
    )


def _enumerate(spec, *, scope_2d="ALLGENOME", band=None):
    """Mirror R's giterator.intervals('1', scope, iterator=itr[, band]).

    R's ``.misha$ALLGENOME`` used as a 2D-iterator scope is the whole 2D
    genome (all chromosome pairs, full rectangles) -> ``mode="full"``.
    """
    scope = pm.gintervals_2d_all(mode="full") if scope_2d == "ALLGENOME" else scope_2d
    kwargs = {} if band is None else {"band": band}
    return pm.giterator_intervals("1", scope, iterator=spec, **kwargs)


def _case_1():
    spec = _grid("test.sparse>1.5 & test.sparse<1.6", "test.sparse>1.55")
    return _enumerate(spec)


def _case_band():
    spec = _grid("test.sparse>1 & test.sparse<1.2", "test.sparse>1.1")
    return _enumerate(spec, band=(-20000, 30000))


def _case_band_1d():
    spec = _grid("test.sparse>1 & test.sparse<1.2")
    return _enumerate(spec, band=(-20000, 30000))


def _case_min_band_idx():
    spec = _grid("test.sparse>1 & test.sparse<1.2", min_band_idx=-1, max_band_idx=2)
    return _enumerate(spec)


def _case_min_band_idx_2():
    spec = _grid("test.sparse>1 & test.sparse<1.2", min_band_idx=-1, max_band_idx=2)
    return _enumerate(spec, scope_2d="test.generated_2d_5")


def _case_min_band_idx_3():
    spec = _grid("test.sparse>1 & test.sparse<1.2", min_band_idx=-1, max_band_idx=2)
    return _enumerate(spec, band=(-20000, 30000))


# id -> (callable, xfail_reason_or_None)
_CASES: dict[str, tuple] = {
    "giterator.cartesian_grid.1": (_case_1, None),
    "giterator.cartesian_grid.band": (_case_band, None),
    "giterator.cartesian_grid.band.1d": (_case_band_1d, None),
    "giterator.cartesian_grid.min.band.idx": (_case_min_band_idx, None),
    "giterator.cartesian_grid.min.band.idx.2": (_case_min_band_idx_2, None),
    "giterator.cartesian_grid.min.band.idx.3": (_case_min_band_idx_3, None),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_giterator_cartesian_grid(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
