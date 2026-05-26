"""Parity port of R misha ``test-giterator.cartesian_grid.R``.

R builds a 2D cartesian-grid *iterator* from 1D interval centers + expansion
breakpoints, then enumerates the resulting cells with
``giterator.intervals(expr, scope, iterator = itr[, band = ...])``.

pymisha cannot reproduce this R-parity call:

* :func:`giterator_intervals` does **not** accept a ``CartesianGridSpec`` scope
  (it requires 1D ``chrom``/``start``/``end`` intervals) and has **no**
  ``band=`` parameter -- so the diagonal-band cases cannot be expressed at all.
* The standalone :func:`giterator_cartesian_grid` (``stream=False``) materializes
  cells via a naive per-center expansion. It does **not** implement R's
  adjacent-center overlap resolution (the ``min_expansion`` / ``max_expansion``
  midpoint clipping in ``TrackExpressionCartesianGridIterator.cpp``) nor R's
  grid-point (center) de-duplication. As a result the cell set diverges
  massively from R -- e.g. the ``min.band.idx`` scope yields 228,800 pymisha
  rows vs 125,750 R rows (~1.82x), and case ``.1`` yields 79,300 vs 78,080.

All 6 cases are therefore xfail(strict). The two reason constants below split
the gap into "no diagonal band" (the grid-generation divergence is the live
issue -- see INVESTIGATE) and "diagonal band requested" (band filtering during
iteration is entirely unimplemented). ``min.band.idx.2`` additionally needs the
cells intersected against a 2D track's rectangles, which only the (absent)
iterator path performs.

Pass: 0 / xfail: 6.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

# Grid-generation divergence in giterator_cartesian_grid: missing adjacent-center
# expansion clipping + center de-dup; ~1.8x too many cells vs R. The R-parity
# call giterator.intervals(expr, scope, iterator=cartesian_grid) is also absent
# (giterator_intervals rejects a CartesianGridSpec scope).
GAP_GRID = (
    "INVESTIGATE: giterator_cartesian_grid omits R's adjacent-center expansion "
    "clipping + center de-dup -> ~1.8x too many cells "
    "(min.band.idx: py 228800 vs R 125750; case .1: py 79300 vs R 78080); "
    "and giterator_intervals cannot consume a cartesian-grid iterator scope"
)

# Diagonal-band filtering during iteration is unimplemented: giterator_intervals
# has no band= parameter, so band=c(...) cases cannot be expressed.
GAP_BAND = (
    "giterator_intervals has no band= parameter; diagonal-band filtering of a "
    "cartesian-grid iterator is not implemented"
)

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

    pymisha cannot consume a cartesian-grid iterator here, so this raises;
    under xfail(strict) the raise is the expected failure.
    """
    scope = pm.gintervals_2d_all() if scope_2d == "ALLGENOME" else scope_2d
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
    "giterator.cartesian_grid.1": (_case_1, GAP_GRID),
    "giterator.cartesian_grid.band": (_case_band, GAP_BAND),
    "giterator.cartesian_grid.band.1d": (_case_band_1d, GAP_BAND),
    "giterator.cartesian_grid.min.band.idx": (_case_min_band_idx, GAP_GRID),
    "giterator.cartesian_grid.min.band.idx.2": (_case_min_band_idx_2, GAP_GRID),
    "giterator.cartesian_grid.min.band.idx.3": (_case_min_band_idx_3, GAP_BAND),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_giterator_cartesian_grid(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)
