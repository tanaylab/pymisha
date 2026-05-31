"""Parity port of R misha ``test-gintervals2.R`` (9 regressions).

Two cases (``rbind.1``, ``union.1``) are pure-compute and run unchanged on R's
read-only test DB. The rest exercise gintervals.save / .update / .ls and need
a writable copy of the DB; they use the ``overlay_db`` fixture, the same one
that the array-track and 2D-Hi-C suites use.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline, load_baseline


def _i1(*a, **k):
    return pm.gintervals(*a, **k)


# ----------------------------------------------------------------------------
# Pure-compute cases (R read-only DB is fine).
# ----------------------------------------------------------------------------

_CASES = {
    "gintervals.rbind.1": lambda: pm.gintervals_rbind(
        pm.gextract("test.fixedbin", _i1([1, 2], 1000, 4000)),
        pm.gextract("test.fixedbin", _i1([2, "X"], 2000, 5000)),
    ),
    "gscreen_and_gintervals.union.1": lambda: pm.gintervals_union(
        pm.gscreen("test.fixedbin > 0.1 & test.fixedbin < 0.3", _i1([1, 2, 4, 8, 9], 0, -1)),
        pm.gscreen("test.fixedbin < 0.2", _i1([1, 2, 4, 7, 9], 0, -1)),
    ),
}


@pytest.mark.parametrize("bid", list(_CASES))
def test_gintervals2(bid):
    assert_matches_baseline(_CASES[bid](), bid)


# ----------------------------------------------------------------------------
# Save/load/update/ls cases (need a writable DB overlay).
# ----------------------------------------------------------------------------


_TMPTRACK_RE = re.compile(r"^test\.tmptrack_")


def _filter_ls(names):
    """Drop temp/legacy interval names, then sort - matches the R test recipe."""
    return sorted(
        n for n in names
        if not _TMPTRACK_RE.match(str(n)) and n != "test.testintervs" and n != "testintervs"
    )


def _assert_list_baseline(items, baseline_id):
    """Compare a Python list-of-string-vectors to a saved R `list(...)` baseline.

    Both ``gintervals.rbind.2`` and ``gintervals.ls.1`` freeze a list of two
    char vectors. R's baseline is read back as a dict keyed by integer index.
    """
    base = load_baseline(baseline_id)
    # The R `list` shows up as a dict here; values may be ndarrays of objects.
    if isinstance(base, dict):
        base_vecs = [list(base[k]) for k in sorted(base, key=lambda k: (isinstance(k, str), k))]
    else:
        base_vecs = [list(v) for v in base]
    py_vecs = [list(v) for v in items]
    assert len(py_vecs) == len(base_vecs), (
        f"[{baseline_id}] list length differs: pymisha={len(py_vecs)} vs R={len(base_vecs)}"
    )
    for i, (p, b) in enumerate(zip(py_vecs, base_vecs, strict=True)):
        p_n = sorted(str(x) for x in p)
        b_n = sorted(str(x) for x in b)
        assert p_n == b_n, (
            f"[{baseline_id}][{i}] vectors differ:\n  pymisha={p_n}\n  R={b_n}"
        )


def test_gintervals_save_1(overlay_db, track_namer):
    """R: gintervals.save(temp, gintervals(c(1,2), 1000, 2000)); gextract(track, temp)."""
    name = track_namer("test.tmptrack")
    pm.gintervals_save(pm.gintervals([1, 2], 1000, 2000), name)
    result = pm.gextract("test.fixedbin", name)
    assert_matches_baseline(result, "gintervals.save.1")


def test_gintervals_rbind_2(overlay_db, track_namer):
    """R: save a temp intervs; compare two filtered gintervals.ls() snapshots."""
    name = track_namer("test.tmptrack")
    pm.gintervals_save(pm.gintervals([1, 2]), name)
    r1 = _filter_ls(pm.gintervals_ls())
    r2 = _filter_ls(pm.gintervals_ls())
    _assert_list_baseline([r1, r2], "gintervals.rbind.2")


def test_gintervals_ls_1(overlay_db, track_namer):
    """R: snapshot gintervals.ls() before and after a save; the filtered lists match."""
    name = track_namer("test.tmptrack")
    r1 = _filter_ls(pm.gintervals_ls())
    pm.gintervals_save(pm.gintervals([1, 2], 1000, 2000), name)
    r2 = _filter_ls(pm.gintervals_ls())
    _assert_list_baseline([r1, r2], "gintervals.ls.1")


def _save_big_set_under(temp_name, source_name):
    """Mimic R's `gintervals.save(temp, source_name)` for a 1D / 2D big set.

    The source name path needs a temporary bump of ``max_data_size`` so the
    whole set fits in memory for the load + save round-trip.
    """
    prev = pm.CONFIG.get("max_data_size")
    try:
        sizes = pm.gintervals_chrom_sizes(source_name)
        total = int(sizes["size"].sum()) + 100
        pm.CONFIG["max_data_size"] = max(int(prev or 0), total)
        df = pm.gintervals_load(source_name)
        pm.gintervals_save(df, temp_name)
    finally:
        pm.CONFIG["max_data_size"] = prev


def test_gintervals_update_3(overlay_db, track_namer):
    """R: gintervals.save(temp, "bigintervs1d"); update chrom=2 with rows [2,3]."""
    name = track_namer("test.tmptrack")
    _save_big_set_under(name, "bigintervs1d")
    r = pm.gintervals_load(name, chrom=2)
    pm.gintervals_update(name, r.iloc[[1, 2]], chrom=2)
    result = [pm.gintervals_load(name, chrom=2), pm.gintervals_chrom_sizes(name)]
    # The R baseline is a list of (loaded_chrom2, chrom_sizes) - compare each piece.
    base = load_baseline("gintervals.update.3")
    _compare_list_of_dfs(result, base, "gintervals.update.3")


def test_gintervals_update_4(overlay_db, track_namer):
    """R: same as .3, then delete chrom 2 (update with None)."""
    name = track_namer("test.tmptrack")
    _save_big_set_under(name, "bigintervs1d")
    pm.gintervals_update(name, None, chrom=2)
    # gintervals_load with chrom filter and no matches returns None - mirror that.
    loaded = pm.gintervals_load(name, chrom=2)
    result = [loaded, pm.gintervals_chrom_sizes(name)]
    base = load_baseline("gintervals.update.4")
    _compare_list_of_dfs(result, base, "gintervals.update.4")


def test_gintervals_update_2d_3(overlay_db, track_namer):
    """R: 2D variant — bigintervs2d, update chrom1=2,chrom2=2 with rows [2,3]."""
    name = track_namer("test.tmptrack")
    _save_big_set_under(name, "bigintervs2d")
    r = pm.gintervals_load(name, chrom1=2, chrom2=2)
    pm.gintervals_update(name, r.iloc[[1, 2]], chrom1=2, chrom2=2)
    result = [
        pm.gintervals_load(name, chrom1=2, chrom2=2),
        pm.gintervals_chrom_sizes(name),
    ]
    base = load_baseline("gintervals.update.2d.3")
    _compare_list_of_dfs(result, base, "gintervals.update.2d.3")


def test_gintervals_update_2d_4(overlay_db, track_namer):
    """R: 2D variant — delete chrom1=2,chrom2=2 cell."""
    name = track_namer("test.tmptrack")
    _save_big_set_under(name, "bigintervs2d")
    pm.gintervals_update(name, None, chrom1=2, chrom2=2)
    loaded = pm.gintervals_load(name, chrom1=2, chrom2=2)
    result = [loaded, pm.gintervals_chrom_sizes(name)]
    base = load_baseline("gintervals.update.2d.4")
    _compare_list_of_dfs(result, base, "gintervals.update.2d.4")


def _compare_list_of_dfs(py_items, base, baseline_id):
    """Compare a Python list of (DataFrame | None) to an R list baseline."""
    if isinstance(base, dict):
        base_items = [base[k] for k in sorted(base, key=lambda k: (isinstance(k, str), k))]
    else:
        base_items = list(base)
    assert len(py_items) == len(base_items), (
        f"[{baseline_id}] list length differs: pymisha={len(py_items)} vs R={len(base_items)}"
    )
    from .baseline import _RTOL, _assert_df_matches  # type: ignore[attr-defined]
    for i, (p, b) in enumerate(zip(py_items, base_items, strict=True)):
        sub_id = f"{baseline_id}[{i}]"
        p_empty = p is None or (hasattr(p, "__len__") and len(p) == 0)
        b_empty = b is None or (hasattr(b, "__len__") and len(b) == 0)
        if p_empty or b_empty:
            assert p_empty == b_empty, (
                f"[{sub_id}] one side empty: pymisha={p_empty}, R={b_empty}"
            )
            continue
        assert isinstance(p, pd.DataFrame) and isinstance(b, pd.DataFrame), (
            f"[{sub_id}] non-DataFrame item: pymisha={type(p).__name__}, R={type(b).__name__}"
        )
        _assert_df_matches(p, b, sub_id, _RTOL)
