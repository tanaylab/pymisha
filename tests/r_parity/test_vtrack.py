"""Parity port of R misha ``test-vtrack.R`` (virtual tracks).

Value-based vtracks over physical 1D tracks (avg/min/max/sum/stddev/nearest/
quantile, and the intervals-source ``distance``) match R. Findings surfaced by
this port:

* **Fixed:** ``global.percentile*`` vtracks read back as all-NaN on indexed-
  format databases. ``pm_vtrack_compute`` (the non-scanner C++ entry these funcs
  use) gated chromosome loading on a per-chrom file that indexed tracks don't
  have. Fixed in ``PMVTrack.cpp`` (see CHANGELOG).
* **Fixed:** ``global.percentile.min`` / ``.max`` now map each per-bin value
  through R's frozen ``vars/pv.percentiles`` binned table (instead of an exact
  empirical CDF), matching R bit-for-bit. The ``global.percentile`` (avg) variant
  remains off (``GAP_GLOBAL_PCT_AVG``).

Open gaps marked ``xfail(strict=True)``:

* ``GAP_ARRAY`` -- array-track extraction / ``gvtrack.array.slice`` extraction.
* ``GAP_COMPUTED`` -- COMPUTED 2D Hi-C tracks (``test.computed2d``).
* ``GAP_GLOBAL_PCT_AVG`` -- global.percentile (avg) per-bin value vs R (float32 ULP).
* ``GAP_VTRACK_ITER`` -- default iterator not inferred from a value-based
  vtrack's source track (+ ``gvtrack.iterator`` shifts).
* ``GAP_2D_ITER`` -- a 1D vtrack ``dim``-projected over a 2D iterator.
* ``GAP_STDDEV`` -- stddev one-pass cancellation: ~tiny diffs vs R two-pass.
* ``GAP_DIST_CENTER`` -- ``distance.center`` differs from R at a single boundary
  row (1 / ~920k).
* ``GAP_DEFAULT_FUNC`` -- ``gvtrack.create`` with an intervals source and no
  ``func`` should default to ``distance`` (R parity); pymisha defaults to avg.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline, load_baseline

GAP_ARRAY = "array-track gextract / gvtrack.array.slice extraction not supported"
GAP_COMPUTED = "COMPUTED 2D Hi-C tracks (test.computed2d) not supported"
# global.percentile.min/.max now read R's frozen vars/pv.percentiles binned table
# (exact native per-bin values -> exact bin). The AVG variant remains off: the
# weighted/averaged per-bin value differs from R by ~1 float32 ULP, so values
# that land exactly on a break (the breaks are the track's float32 quantiles)
# flip to an adjacent bin. Matching needs bit-exact AVG float32 accumulation order.
GAP_GLOBAL_PCT_AVG = "global.percentile (avg) per-bin value differs from R by ~1 float32 ULP at break boundaries"
GAP_VTRACK_ITER = "default iterator not inferred from value-based vtrack source (+ gvtrack.iterator shifts)"
GAP_2D_ITER = "1D vtrack dim-projection over a 2D iterator differs from R"
GAP_STDDEV = "stddev one-pass numerical cancellation vs R two-pass (tiny diffs)"
GAP_DIST_CENTER = "distance.center differs from R at one boundary row"
GAP_DEFAULT_FUNC = "gvtrack.create intervals source should default func='distance' (R parity)"


def _clear():
    for v in pm.gvtrack_ls():
        pm.gvtrack_rm(v)


def _iv12():
    return pm.gintervals([1, 2])


def _iv2d_a():
    return pm.gintervals_2d([1, 3], 3000000, -1, [1, 4], 3000000, -1)


def _iv2d_b():
    return pm.gintervals_2d([6, 1, 5], 3000000, -1, [8, 1, 9], 3000000, -1)


def _iv2d_rects():
    return pm.gintervals_2d([1, 3], 3000000, -1, [2, 4], 3000000, -1)


def _value(src, func, it, params=None):
    def f():
        _clear()
        pm.gvtrack_create("v1", src, func=func, params=params)
        return pm.gextract("v1", _iv12(), iterator=it)
    return f


def _value_2d(src, func):
    scope = _iv2d_a if src == "test.rects" else _iv2d_b

    def f():
        _clear()
        pm.gvtrack_create("v1", src, func=func)
        return pm.gextract("v1", scope(), iterator=[2000000, 3000000])
    return f


def _two_quantile(it):
    def f():
        _clear()
        pm.gvtrack_create("v1", "test.fixedbin", func="quantile", params=0.5)
        pm.gvtrack_create("v2", "test.fixedbin", func="quantile", params=0.9)
        return pm.gextract(["v1", "v2"], _iv12(), iterator=it)
    return f


def _basic(iterator):
    def f():
        _clear()
        pm.gvtrack_create("v1", "test.fixedbin")
        return pm.gextract("v1", _iv12(), iterator=iterator)
    return f


def _intervs_vtrack(func, it, strand=None):
    def f():
        _clear()
        intervs = pm.gscreen("test.fixedbin > 0.5", pm.gintervals([1, 3], 0, -1))
        if strand is not None:
            intervs = intervs.copy()
            intervs["strand"] = strand
        if func is None:
            pm.gvtrack_create("v1", intervs)
        else:
            pm.gvtrack_create("v1", intervs, func)
        return pm.gextract("v1", _iv12(), iterator=it)
    return f


def _slice(cols, func=None, params=None):
    def f():
        _clear()
        pm.gvtrack_create("v1", "test.array")
        if func is None:
            pm.gvtrack_array_slice("v1", cols)
        else:
            pm.gvtrack_array_slice("v1", cols, func, params)
        return pm.gextract("v1", _iv12())
    return f


def _iter_dim(dim=None, sshift=0, eshift=0, scope=_iv12, it=None):
    def f():
        _clear()
        pm.gvtrack_create("v1", "test.fixedbin")
        pm.gvtrack_iterator("v1", dim=dim, sshift=sshift, eshift=eshift)
        return pm.gextract("v1", scope(), iterator=it)
    return f


def _iter2d_custom(src, scope):
    def f():
        _clear()
        pm.gvtrack_create("v1", src)
        pm.gvtrack_iterator_2d("v1", sshift1=-1000000, eshift1=-500000, sshift2=2000000, eshift2=2800000)
        return pm.gextract("v1", scope(), iterator=src)
    return f


def _ls(*specs):
    def f():
        _clear()
        for name, src in specs:
            pm.gvtrack_create(name, src)
        return pm.gvtrack_ls()
    return f


# id -> (callable, xfail_reason_or_None)
_CASES: dict[str, tuple] = {
    "vtrack.sparse.basic": (_basic("test.sparse"), None),
    "vtrack.array.basic": (_basic("test.array"), GAP_ARRAY),
    "vtrack.sparse.avg": (_value("test.fixedbin", "avg", 233), None),
    "vtrack.sparse.avg.high": (_value("test.sparse", "avg", 10000), None),
    "vtrack.array.avg.high": (_value("test.array", "avg", 10000), GAP_ARRAY),
    "vtrack.rects.avg.2d": (_value_2d("test.rects", "avg"), None),
    "vtrack.computed2d.avg.2d": (_value_2d("test.computed2d", "avg"), GAP_COMPUTED),
    "vtrack.sparse.max": (_value("test.fixedbin", "max", 233), None),
    "vtrack.sparse.max.high": (_value("test.sparse", "max", 10000), None),
    "vtrack.array.max.high": (_value("test.array", "max", 10000), GAP_ARRAY),
    "vtrack.rects.max.2d": (_value_2d("test.rects", "max"), None),
    "vtrack.computed2d.max.2d": (_value_2d("test.computed2d", "max"), GAP_COMPUTED),
    "vtrack.fixedbin_min": (_value("test.fixedbin", "min", 233), None),
    "vtrack.sparse_min": (_value("test.sparse", "min", 10000), None),
    "vtrack.array_min": (_value("test.array", "min", 10000), GAP_ARRAY),
    "vtrack.rects_min": (_value_2d("test.rects", "min"), None),
    "vtrack.computed2d_min": (_value_2d("test.computed2d", "min"), GAP_COMPUTED),
    "vtrack.fixedbin_nearest": (_value("test.fixedbin", "nearest", 233), None),
    "vtrack.sparse_nearest": (_value("test.sparse", "nearest", 10000), None),
    "vtrack.array_nearest": (_value("test.array", "nearest", 10000), GAP_ARRAY),
    "vtrack.fixedbin_stddev": (_value("test.fixedbin", "stddev", 233), None),
    "vtrack.sparse_stddev": (_value("test.sparse", "stddev", 10000), None),
    "vtrack.array_stddev": (_value("test.array", "stddev", 10000), GAP_ARRAY),
    "vtrack.fixedbin_sum": (_value("test.fixedbin", "sum", 233), None),
    "vtrack.sparse_sum": (_value("test.sparse", "sum", 10000), None),
    "vtrack.array_sum": (_value("test.array", "sum", 10000), GAP_ARRAY),
    "vtrack.fixedbin_quantile": (_two_quantile(233), None),
    "vtrack.fixedbin.quantile_extraction_1": (_two_quantile(10000), None),
    "vtrack.fixedbin.global_percentile_extraction": (_value("test.fixedbin", "global.percentile", 233), GAP_GLOBAL_PCT_AVG),
    "vtrack.fixedbin.global_percentile_max_extraction": (_value("test.fixedbin", "global.percentile.max", 233), None),
    "vtrack.fixedbin.result": (_value("test.fixedbin", "global.percentile.min", 233), None),
    "vtrack.fixedbin.gscreen.result": (_intervs_vtrack(None, 533), None),
    "vtrack.fixedbin.positive.result": (_intervs_vtrack("distance", 533, strand=1), None),
    "vtrack.fixedbin.negative.result": (_intervs_vtrack("distance", 533, strand=-1), None),
    "vtrack.reg_fixedbin_distance_center": (_intervs_vtrack("distance.center", 533), None),
    "vtrack.reg_fixedbin_strand1_distance_center_without_param": (_intervs_vtrack("distance.center", 533, strand=1), None),
    "vtrack.iterator_533_regression": (_intervs_vtrack("distance.center", 533, strand=-1), None),
    "vtrack.iterator_test.sparse_regression": (_intervs_vtrack("distance.center", "test.sparse"), None),
    "vtrack.iterator_test.array_regression": (_intervs_vtrack("distance.center", "test.array"), GAP_ARRAY),
    "vtrack.tracktype_test.array_regression": (
        lambda: (_clear(), pm.gvtrack_create("v1", "test.array"), pm.gvtrack_array_slice("v1"), pm.gvtrack_ls())[-1],
        None,
    ),
    "vtrack.tracktype_test.array_slice_columns_regression": (_slice(["col1", "col1", "col3", "col5"]), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col135_regression": (_slice(["col1", "col3", "col5"]), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col135_avg_regression": (_slice(["col1", "col3", "col5"], "avg"), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col135_min_regression": (_slice(["col1", "col3", "col5"], "min"), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col135_max_regression": (_slice(["col1", "col3", "col5"], "max"), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col13568_stddev_regression": (_slice(["col1", "col3", "col5", "col6", "col8"], "stddev"), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col13568_sum_regression": (_slice(["col1", "col3", "col5", "col6", "col8"], "sum"), GAP_ARRAY),
    "vtrack.tracktype_test.array_slice_col13568_quantile_0.4_regression": (_slice(["col1", "col3", "col5", "col6", "col8"], "quantile", 0.4), GAP_ARRAY),
    "vtrack.tracktype_test.fixedbin_iterator_dim0_sshift-130_eshift224_regression": (_iter_dim(dim=0, sshift=-130, eshift=224), None),
    "vtrack.tracktype_test.fixedbin_iterator_sshift-130_eshift224_regression": (_iter_dim(sshift=-130, eshift=224), None),
    "vtrack.tracktype_test.fixedbin_iterator_dim1_gintervals2d_testrects_regression": (_iter_dim(dim=1, scope=_iv2d_rects, it="test.rects"), None),
    "vtrack.fixedbin_dim2_gintervals2d_testrects_regression": (_iter_dim(dim=2, scope=_iv2d_rects, it="test.rects"), None),
    "vtrack.fixedbin_dim1_shifts_gintervals2d_testrects_regression": (_iter_dim(dim=1, sshift=-130, eshift=224, scope=_iv2d_rects, it="test.rects"), None),
    "vtrack.fixedbin_dim1_gintervals2d_testcomputed2d_v1_regression": (_iter_dim(dim=1, scope=_iv2d_b, it="test.computed2d"), GAP_COMPUTED),
    "vtrack.fixedbin_dim2_gintervals2d_testcomputed2d_v1_regression": (_iter_dim(dim=2, scope=_iv2d_b, it="test.computed2d"), GAP_COMPUTED),
    "vtrack.fixedbin_dim1_shifts_gintervals2d_testcomputed2d_v1_regression": (_iter_dim(dim=1, sshift=-130, eshift=224, scope=_iv2d_b, it="test.computed2d"), GAP_COMPUTED),
    "vtrack.rects_iterator2d_customShifts_gintervals2d_testrects_regression": (_iter2d_custom("test.rects", _iv2d_rects), None),
    "vtrack.computed2d_iterator2d_customShifts_gintervals2d_testcomputed2d_regression": (_iter2d_custom("test.computed2d", _iv2d_b), GAP_COMPUTED),
    "vtrack.multipleCreation_gvtrackls_regression": (_ls(("v1", "test.rects"), ("v2", "test.sparse"), ("v3", "test.computed2d")), None),
    "vtrack.sparseCreation_v1Removal_gvtrackls_regression": (_ls(("v2", "test.sparse")), None),
}


@pytest.mark.parametrize("baseline_id", list(_CASES))
def test_vtrack(baseline_id, request):
    fn, xfail_reason = _CASES[baseline_id]
    if xfail_reason is not None:
        request.node.add_marker(pytest.mark.xfail(reason=xfail_reason, strict=True))
    assert_matches_baseline(fn(), baseline_id)


def test_vtrack_info_fixedbin_func_max():
    """gvtrack.info: R returns a list with src/func; compare those fields."""
    _clear()
    pm.gvtrack_create("v1", "test.fixedbin", func="max")
    info = pm.gvtrack_info("v1")
    base = load_baseline("vtrack.tracktype_test.fixedbin_func_max_regression")
    # R wraps scalars in length-1 vectors (lists); compare element 0.
    assert str(info["src"]) == str(base["src"][0])
    assert str(info["func"]) == str(base["func"][0])
