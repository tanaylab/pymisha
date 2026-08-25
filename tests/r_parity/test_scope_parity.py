"""Every aggregating function must return exactly what R misha returns.

Compares pymisha against R misha's frozen output for the same calls on misha's
own bundled example database - 8 scope shapes x 9 functions. No R process at
test time, so this runs in ordinary CI; regenerate the baseline with
``tools/generate_scope_parity_baseline.R`` when the cases change.

This exists because reading the code was not enough. The audit that produced
the scope-canonicalisation fix recorded ``gpartition`` as already correct; a
differential run against R showed it returning 74 partitions where misha
returned 37. Two identical scope intervals, doubled.

Scope shapes cover the ways an interval set can be non-canonical: overlapping,
nested, touching, unsorted, duplicated, and spread across chromosomes. Each runs
under three iterators, because the iterator axis hides its own class of
divergence: pymisha intersects a DataFrame iterator with the scope in Python and
hands C++ the resulting BINS, so anything that canonicalises there merges
adjacent bins. gdist with a two-bin touching iterator returned 1 bin where misha
returned 2, and had done so long before this test existed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

BASELINE = json.loads((Path(__file__).parent / "scope_parity_baseline.json").read_text())
_RTOL = 1e-5


def _df(chrom, start, end):
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end})


SCOPES = {
    "single":      _df(["chr1"], [0], [10000]),
    "overlapping": _df(["chr1"] * 2, [0, 5000], [10000, 15000]),
    "nested":      _df(["chr1"] * 2, [0, 2000], [10000, 4000]),
    "touching":    _df(["chr1"] * 2, [0, 10000], [10000, 20000]),
    "unsorted":    _df(["chr1"] * 2, [5000, 0], [15000, 10000]),
    "disjoint":    _df(["chr1"] * 2, [0, 20000], [10000, 30000]),
    "multichrom":  _df(["chr1", "chr2", "chr1"], [0, 0, 5000], [10000, 10000, 15000]),
    "dup":         _df(["chr1"] * 2, [0, 0], [10000, 10000]),
}

_BREAKS = [0, 0.25, 0.5, 0.75, 1.0]

ITERATORS = {
    "auto":  None,
    "fixed": 500,
    "df":    _df(["chr1"] * 2, [0, 1000], [1000, 2000]),
}


@pytest.fixture(scope="module", autouse=True)
def _examples():
    pm.gdb_init_examples()


def _n(x):
    return 0 if x is None else int(len(x))


def _calls(iv, it):
    kw = {} if it is None else {"iterator": it}

    def screen_coords():
        r = pm.gscreen("dense_track > 0.1", intervals=iv, **kw)
        return None if r is None or len(r) == 0 else list(r["start"]) + list(r["end"])

    return {
        "gsummary":     lambda: pm.gsummary("dense_track", intervals=iv, **kw).values,
        "gquantiles":   lambda: pm.gquantiles("dense_track", [0.1, 0.5, 0.9], intervals=iv, **kw).values,
        "gcor":         lambda: np.asarray(pm.gcor("dense_track", "dense_track*2", intervals=iv, **kw)),
        "gscreen_n":    lambda: _n(pm.gscreen("dense_track > 0.1", intervals=iv, **kw)),
        "gscreen":      screen_coords,
        # gextract is the control: it reports per-interval rows with an
        # intervalID, and misha does NOT canonicalise there. If a future change
        # "fixes" gextract to unify, this baseline catches it.
        "gextract_n":   lambda: _n(pm.gextract("dense_track", intervals=iv, **kw)),
        "gdist":        lambda: np.asarray(pm.gdist("dense_track", _BREAKS, intervals=iv, **kw)),
        "gsegment_n":   lambda: _n(pm.gsegment("dense_track", 500, 0.5, intervals=iv)),
        "gpartition_n": lambda: _n(pm.gpartition("dense_track", _BREAKS, intervals=iv, **kw)),
    }


def _flat(v):
    if v is None:
        return None
    if isinstance(v, str):
        return "ERR"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [float(v)]
    return [None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
            for x in np.asarray(v, dtype=float).ravel()]


# Divergences this harness found that PREDATE it. Recorded rather than hidden:
# strict, so whoever fixes one gets a failure telling them to delete the entry.
KNOWN_DIVERGENCE = {
    # pm_cor rejects a DataFrame iterator outright ("iterator must be an integer
    # or float"); gcor never routes through _preprocess_intervals_iterator the
    # way gsummary/gquantiles/gdist/gscreen do. R accepts it and returns a value.
    **{(f"{sc}|df", "gcor"): "gcor does not accept a DataFrame iterator (R does)"
       for sc in ("single", "overlapping", "nested", "touching", "unsorted",
                  "disjoint", "multichrom", "dup")},
    # A bin holding exactly 0.100 under "dense_track > 0.1": R promotes the
    # stored float32 to double (0.10000000149... > 0.1, kept), pymisha compares
    # in float32 (equal, dropped). Same result for a canonical single-interval
    # scope, so it is a comparison-width difference, not a scope question.
    ("touching|fixed", "gscreen"): "float32 vs double compare at an exact threshold",
    ("touching|fixed", "gscreen_n"): "float32 vs double compare at an exact threshold",
}


@pytest.mark.parametrize("case", sorted(BASELINE))
@pytest.mark.parametrize("fn", sorted(next(iter(BASELINE.values()))))
def test_matches_r_misha(case, fn, request):
    reason = KNOWN_DIVERGENCE.get((case, fn))
    if reason:
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))
    scope, _, itr = case.partition("|")
    expected = _flat(BASELINE[case][fn])
    if expected == "ERR":
        pytest.skip(f"R misha itself errors on {fn}/{case}")

    got = _flat(_calls(SCOPES[scope], ITERATORS[itr])[fn]())

    assert (got is None) == (expected is None), f"{fn}/{case}: {got} vs R {expected}"
    if expected is None:
        return
    assert len(got) == len(expected), f"{fn}/{case}: {len(got)} values vs R's {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected)):
        if g is None or e is None:
            assert g == e, f"{fn}/{case}[{i}]: {g} vs R {e}"
        else:
            assert math.isclose(g, e, rel_tol=_RTOL, abs_tol=1e-9), \
                f"{fn}/{case}[{i}]: {g} vs R {e}"
