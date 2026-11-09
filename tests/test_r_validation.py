"""R misha 5.14.2's argument checks, with R's messages: test-vtrack.R error
blocks (26-450, 492-508, 793-869), test-directional-neighbors.R:98,
test-pwm-edit-distance.R max_edits, liftover's source-chromosome checks."""

import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

PSSM = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])


@pytest.fixture(autouse=True)
def _clear():
    yield
    pm.gvtrack_clear()


@pytest.mark.parametrize("src,func,params,msg", [
    ("dense_track", "avg", 10, "function avg does not accept any parameters"),
    ("dense_track", "max", 10, "function max does not accept any parameters"),
    ("dense_track", "min", 10, "function min does not accept any parameters"),
    ("dense_track", "nearest", 10, "function nearest does not accept any parameters"),
    ("dense_track", "stddev", 10, "function stddev does not accept any parameters"),
    ("dense_track", "sum", 10, "function sum does not accept any parameters"),
    ("dense_track", "global.percentile", 10, "function global.percentile does not accept any parameters"),
    ("dense_track", "blabla", None, "invalid function blabla used for a virtual track"),
    ("dense_track", "quantile", None, "requires an additional parameter \\(percentile\\)"),
    ("dense_track", "quantile", 1.5, "is out of range"),
    ("dense_track", "area", None, "function area is not supported by dense tracks"),
    ("sparse_track", "global.percentile", None, "not supported by sparse tracks"),
    ("array_track", "global.percentile.max", None, "not supported by array tracks"),
    ("rects_track", "global.percentile", None, "not supported by rectangles tracks"),
    ("rects_track", "nearest", None, "function nearest is not supported by rectangles tracks"),
    ("rects_track", "stddev", None, "not supported by rectangles tracks"),
    ("rects_track", "sum", None, "not supported by rectangles tracks"),
])
def test_track_source_checks(src, func, params, msg):
    with pytest.raises(ValueError, match=msg):
        pm.gvtrack_create("v1", src, func, params)


@pytest.mark.parametrize("func,params,msg", [
    ("distance.center", 200, "function distance.center does not accept any parameters"),
    ("distance.edge", 100, "function distance.edge does not accept any parameters"),
    ("coverage", 1, "function coverage does not accept any parameters"),
    ("distance", -5, "margin cannot be a negative number"),
    ("neighbor.count", "foo", "neighbor.count requires 'params' to be a single numeric value"),
    ("blabla", None, "invalid function blabla used with intervals"),
])
def test_intervals_source_checks(func, params, msg):
    iv = pm.gintervals("chr1", [0, 500], [100, 600])
    with pytest.raises(ValueError, match=msg):
        pm.gvtrack_create("v1", iv, func, params)


@pytest.mark.parametrize("func,params,msg", [
    ("avg", 10, "function avg does not accept any parameters"),
    ("quantile", None, "function quantile requires percentile parameter"),
    ("quantile", 2, "percentile must be between 0 and 1"),
    ("blabla", None, "invalid function blabla used with value-based intervals"),
])
def test_value_source_checks(func, params, msg):
    df = pd.DataFrame({"chrom": ["chr1"] * 2, "start": [0, 100], "end": [50, 150], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match=msg):
        pm.gvtrack_create("v1", df, func, params)


def test_valid_calls_still_work():
    pm.gvtrack_create("v1", "dense_track", "quantile", 0.5)
    pm.gvtrack_create("v2", "dense_track", "quantile", [0.9])  # R's length-1 vector
    pm.gvtrack_create("v3", "dense_track", "global.percentile")
    pm.gvtrack_create("v4", pm.gintervals("chr1", [0], [100]), "neighbor.count", 10)


def test_iterator_modifiers_follow_the_source_dimension():
    pm.gvtrack_create("v1", "dense_track")
    with pytest.raises(ValueError, match="2D iterator modifier cannot be used with source that supports only 1D"):
        pm.gvtrack_iterator_2d("v1")
    pm.gvtrack_create("v2", "rects_track")
    with pytest.raises(ValueError, match="1D iterator modifier cannot be used with source that supports only 2D"):
        pm.gvtrack_iterator("v2", sshift=10, eshift=10)
    with pytest.raises(ValueError, match="1D iterator modifier"):
        pm.gvtrack_iterator("v2", dim=1)
    pm.gvtrack_iterator("v1", dim=1)  # dim projects a 2D iterator onto a 1D source


@pytest.mark.parametrize("strand,msg", [
    (0, "Invalid strand value 0 at row 1. Strand must be -1 or 1."),
    # 2 is no strand at all: rejected when the intervals are read (R 5.16.1)
    (2, "Invalid strand value 2 at index 1")])
def test_directional_neighbors_reject_other_strands(strand, msg):
    q = pd.DataFrame({"chrom": ["chr1"], "start": [5000], "end": [5100], "strand": [strand]})
    tg = pm.gintervals("chr1", [1000, 9000], [1100, 9100])
    with pytest.raises(Exception, match=msg):
        pm.gintervals_neighbors(q, tg, use_intervals1_strand=True)


def test_max_edits_must_be_positive():
    with pytest.raises(ValueError, match="max_edits must be NULL or a positive integer"):
        pm.gvtrack_create("ve", None, "pwm.edit_distance", pssm=PSSM, score_thresh=-5, max_edits=0)
    with pytest.raises(ValueError, match="max_edits must be NULL or a positive integer"):
        pm.gseq_pwm_edits(["ACGT"], PSSM, score_thresh=-5, max_edits=0)


def _chain(tmp_path, header, size=100, name="c.chain"):
    p = os.path.join(str(tmp_path), name)
    with open(p, "w") as f:
        f.write(f"chain {header}\n{size}\n\n")
    return p


def test_liftover_with_no_source_chromosome_in_the_chain(tmp_path):
    chain = pm.gintervals_load_chain(_chain(tmp_path, "100 chrS 100000 + 0 100 chr2 300000 + 500 600 1"))
    with pytest.warns(UserWarning, match=r"None of the intervals' chromosomes is a source chromosome of the "
                                         r"chain \(for example chrS\); the result is empty"):
        got = pm.gintervals_liftover(pd.DataFrame({"chrom": ["chrZZ"], "start": [0], "end": [10]}), chain)
    assert got.empty


def test_load_chain_src_groot(tmp_path):
    from _dbpath import TESTDB_ROOT

    root = pm._shared._GROOT
    ok = _chain(tmp_path, "100 chr1 500000 + 0 100 chr2 300000 + 500 600 1", name="ok.chain")
    assert len(pm.gintervals_load_chain(ok, src_groot=str(TESTDB_ROOT))) == 1
    bad = _chain(tmp_path, name="bad.chain", header="100 chrQ 500000 + 0 100 chr2 300000 + 500 600 1")
    with pytest.raises(ValueError, match="Chromosome chrQ does not exist in the database"):
        pm.gintervals_load_chain(bad, src_groot=str(TESTDB_ROOT))
    far = _chain(tmp_path, name="far.chain", header="100 chr1 500000 + 499950 500050 chr2 300000 + 500 600 1")
    with pytest.raises(ValueError, match="line 1: reference end coordinate exceeds chromosome size"):
        pm.gintervals_load_chain(far, src_groot=str(TESTDB_ROOT))
    with pytest.raises(FileNotFoundError, match="Database directory does not exist"):
        pm.gintervals_load_chain(ok, src_groot="/nonexistent")
    assert pm._shared._GROOT == root


PSSM3 = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]])


@pytest.mark.parametrize("func", ["pwm.edit_distance", "pwm.edit_distance.lse"])
@pytest.mark.parametrize("kw,msg", [
    ({"score_min": "abc"}, "score.min must be NULL or a single numeric value"),
    ({"extend": -1}, "extend must be True, False, or a non-negative integer"),
    ({"extend": 2.5}, "extend must be True, False, or a non-negative integer"),
    ({"extend": "yes"}, "extend must be True, False, or a non-negative integer"),
])
def test_edit_distance_argument_checks(func, kw, msg):  # test-pwm-edit-distance.R:1558, 2133-2237
    with pytest.raises(ValueError, match=msg):
        pm.gvtrack_create("v", None, func, pssm=PSSM3, score_thresh=-3, **kw)


def test_edit_distance_max_indels_checks():  # test-pwm-edit-distance.R:822, test-vtrack-params-validation.R:131
    with pytest.raises(ValueError, match="max_indels must be NULL or a non-negative integer"):
        pm.gvtrack_create("v", None, "pwm.edit_distance", pssm=PSSM3, score_thresh=-3, max_indels=-1)
    for f in ("pwm.edit_distance.lse", "pwm.n_mutations"):
        with pytest.raises(ValueError, match=f"function '{f}' does not accept parameter 'max_indels'"):
            pm.gvtrack_create("v", None, f, pssm=PSSM3, score_thresh=-3, max_indels=1)


@pytest.mark.parametrize("func", ["pwm.edit_distance", "pwm.edit_distance.lse"])
def test_edit_distance_integer_extend(func):  # test-pwm-edit-distance.R:2133-2237
    """R 5.15.1: 0 is FALSE, a positive integer extends like TRUE. The lse version
    targets the strand-combined score (bidirect), so its counts differ."""
    pssm = np.array([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7]])
    iv = pm.gintervals(["chr1"] * 11, list(range(200, 301, 10)), list(range(205, 306, 10)))
    vals = {}
    for e in (True, False, 0, 1, 5):
        pm.gvtrack_create("v", None, func, pssm=pssm, score_thresh=-4, extend=e)
        vals[e] = pm.gextract("v", iv, iterator=iv)["v"].tolist()
    lse = func.endswith(".lse")
    assert vals[False] == [1, 1, 3, 1, 2, 1, 2, 2, 0 if lse else 1, 2, 2]
    assert vals[0] == vals[False]
    assert vals[True] == vals[1] == vals[5] == ([1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1] if lse else [1] * 11)


@pytest.mark.parametrize("func,want", [("pwm.edit_distance", 1.0), ("pwm.n_mutations", 1.0),
                                       ("pwm.edit_distance.lse", 0.0)])
def test_edit_distance_score_thresh_coercion(func, want):  # test-vtrack-params-validation.R:252, 260, 268
    iv = pm.gintervals("chr1", 200, 400)
    for th in ("-3", -3, [-3]):
        pm.gvtrack_create("v", None, func, pssm=PSSM3, score_thresh=th)
        assert pm.gextract("v", iv, iterator=iv)["v"][0] == want
    for bad, msg in ((True, "not a bool"), ("loose", "'loose' is not"), (float("nan"), "NaN is not")):
        with pytest.raises(ValueError, match=msg):
            pm.gvtrack_create("v", None, func, pssm=PSSM3, score_thresh=bad)


def test_lse_score_thresh_string_equals_number():  # test-vtrack-params-validation.R:268
    iv = pm.gintervals("chr1", 200, 400)
    vals = []
    for th in (0.5, "0.5"):
        pm.gvtrack_create("v", None, "pwm.edit_distance.lse", pssm=PSSM3, score_thresh=th)
        vals.append(pm.gextract("v", iv, iterator=iv)["v"][0])
    assert vals[0] == vals[1] == 0.0


def test_misspelled_key_inside_params():  # test-vtrack-params-validation.R:27
    with pytest.raises(ValueError, match="does not accept parameter 'bidrect'"):
        pm.gvtrack_create("v", None, "pwm", params={"pssm": PSSM3, "bidrect": False})
    iv = pm.gintervals("chr1", 200, 400)
    pm.gvtrack_create("vt", None, "pwm", params={"pssm": PSSM3, "bidirect": True})
    pm.gvtrack_create("vf", None, "pwm", params={"pssm": PSSM3, "bidirect": False})
    r = pm.gextract(["vt", "vf"], iv, iterator=iv)
    assert r["vt"][0] != r["vf"][0]


def test_neighbor_count_rejects_unknown_arguments():  # test-vtrack-params-validation.R:147
    with pytest.raises(ValueError, match="'neighbor.count'"):
        pm.gvtrack_create("v", pm.gintervals("chr1", 0, 100), "neighbor.count", 10, foo=1)
