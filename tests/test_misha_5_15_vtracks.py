"""R misha 5.15.0 virtual track fixes (test-vtrack-review-fixes.R, test-vtrack-values.R:383),
against the values R 5.15.1 gives for the same tracks on this test DB.

A filter that splits a bin or interval counts its value once (sum, size, lse, quantile; avg
is the plain mean, stddev the sample sd of the unmasked values); *.pos functions clip to the
iterator start (and to the unmasked part they read); a NaN dense bin is no value; edit
distances work under a filter. Where R 5.15.1 is wrong (array lse, value-based stddev at 1e9
in float) the test says so and checks the exact value."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha

TRACKS = ["r515_dn", "r515_sp", "r515_dn2", "r515_sp2", "r515_arr3", "r515_arr4", "r515_dn4", "r515_sp5"]
PSSM = np.array([
    [0.121683998590809, 0.0850085927520161, 0.500547218308171, 0.292760190349004],
    [0.392032408712155, 0.465192233767692, 0.100925341762383, 0.0418500157577697],
    [0.271908618585922, 0.221964804846898, 0.192211156788389, 0.313915419778791],
    [0.0857786908572187, 0.280728034850757, 0.435676478546413, 0.197816795745611],
    [0.285164974654435, 0.166981052616758, 0.295002167249298, 0.25285180547951],
    [0.605297366135969, 0.153265719826041, 0.144880959004694, 0.0965559550332962],
])


def gi(c, s, e):
    return pm.gintervals(c, s, e)


def _array(name, rows, tmp_path):
    f = tmp_path / f"{name}.tsv"
    f.write_text("chrom\tstart\tend\ta\n" + "".join(f"chr1\t{s}\t{e}\t{v}\n" for s, e, v in rows))
    pm.gtrack_array_import(name, "x", str(f))


@pytest.fixture(scope="module")
def tracks(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arr")
    for t in TRACKS:
        if pm.gtrack_exists(t):
            pm.gtrack_rm(t, force=True)
    pm.gtrack_create_dense("r515_dn", "x", gi("chr1", [0, 50, 100], [50, 100, 150]), [1, np.nan, 3], 50, np.nan)
    pm.gtrack_create_sparse("r515_sp", "x", gi("chr1", [313, 400], [332, 420]), [-1.5, 2])
    starts = np.arange(0, 1000, 50)
    pm.gtrack_create_dense("r515_dn2", "x", gi("chr1", starts, starts + 50), np.arange(1, 21, dtype=float), 50, np.nan)
    pm.gtrack_create_sparse("r515_sp2", "x", gi("chr1", [10, 100, 200, 300], [60, 120, 220, 320]), [4.0, 1, 7, 2])
    _array("r515_arr3", [(313, 332, 5), (400, 420, 1)], tmp)
    _array("r515_arr4", [(10, 60, 4), (100, 120, 1), (200, 220, 7), (300, 320, 2)], tmp)
    pm.gtrack_create_dense("r515_dn4", "x", gi(["chr1"] * 3 + ["chr2"] * 3, [0, 50, 100] * 2, [50, 100, 150] * 2),
                           [1.0, 2, 7, 4, 10, 6], 50, np.nan)
    pm.gtrack_convert_to_indexed("r515_dn4")
    pm.gtrack_create_sparse("r515_sp5", "x", gi("chr1", [313, 500], [340, 510]), [-1.5, 2])
    _pymisha.pm_dbreload()
    yield
    pm.gvtrack_clear()
    for t in TRACKS:
        if pm.gtrack_exists(t):
            pm.gtrack_rm(t, force=True)
    _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    yield
    pm.gvtrack_clear()


def vt(src, func, iv, params=None, iterator=None, **kw):
    pm.gvtrack_create("t", src, func, params, **kw)
    return pm.gextract("t", iv, iterator=iv if iterator is None else iterator)["t"].tolist()


MASK = pd.DataFrame(dict(chrom=["chr1", "chr1"], start=[20, 320], end=[30, 322]))
VV = pd.DataFrame(dict(chrom="chr1", start=[313, 400], end=[332, 420], score=[-1.5, 2]))
MASK2 = pd.DataFrame(dict(chrom=["chr1"] * 3, start=[20, 190, 500], end=[30, 230, 600]))
VV2 = pd.DataFrame(dict(chrom="chr1", start=[10, 100, 200, 300], end=[60, 120, 220, 320], score=[4.0, 1, 7, 2]))


@pytest.mark.parametrize("func,dense,sparse", [  # R test-vtrack-review-fixes.R:64-106
    ("sum", 4, 0.5), ("size", 2, 2), ("lse", 3.126928011, 2.029750418), ("avg", 2, 0.25),
    ("stddev", 1.414213562, 2.474873734), ("quantile", 2, 0.25),
])
def test_split_bin_or_interval_counts_once(tracks, func, dense, sparse):
    p = 0.5 if func == "quantile" else None
    np.testing.assert_allclose(vt("r515_dn", func, gi("chr1", 0, 150), p, filter=MASK), [dense], rtol=1e-6)
    np.testing.assert_allclose(vt("r515_sp", func, gi("chr1", 300, 450), p, filter=MASK), [sparse], rtol=1e-6)
    np.testing.assert_allclose(vt(VV, func, gi("chr1", 300, 450), p, filter=MASK), [sparse], rtol=1e-6)


@pytest.mark.parametrize("func,dense,others", [  # R test-vtrack-review-fixes.R:174-199, 428-452
    ("sum", 187, 7), ("size", 18, 3), ("lse", 20.4583851, 4.16984602), ("avg", 10.38888889, 2.333333333),
    ("stddev", 6.241595676, 1.527525232), ("quantile", 9.5, 2),
])
def test_several_masks_dense_sparse_value_array(tracks, func, dense, others):
    p = 0.5 if func == "quantile" else None
    iv = gi("chr1", 0, 1000)
    np.testing.assert_allclose(vt("r515_dn2", func, iv, p, filter=MASK2), [dense], rtol=1e-6)
    for src in ("r515_sp2", VV2, "r515_arr4"):
        # R 5.15.1 gives log(3) for the array lse, the lse of zeros (its array tracks never fill lse);
        # the values are 4, 1, 2, as for the sparse and value-based sources
        np.testing.assert_allclose(vt(src, func, iv, p, filter=MASK2), [others], rtol=1e-6)


def test_stddev_with_avg_of_the_same_track(tracks):
    pm.gvtrack_create("t", "r515_dn2", "stddev", filter=MASK2)
    pm.gvtrack_create("t2", "r515_dn2", "avg", filter=MASK2)
    iv = gi("chr1", 0, 1000)
    np.testing.assert_allclose(pm.gextract(["t", "t2"], iv, iterator=iv)["t"], [6.241595676], rtol=1e-6)


def test_value_based_quantile_uses_every_part(tracks):  # R :108-116
    vq = pd.DataFrame(dict(chrom="chr1", start=list(range(100, 601, 100)), end=list(range(150, 651, 100)),
                           score=[1.0, 9, 2, 8, 3, 7]))
    assert vt(vq, "quantile", gi("chr1", 0, 1000), 0.5, filter=gi("chr1", 360, 380)) == [5]


@pytest.mark.parametrize("func,want", [("sum", [10, 20]), ("size", [3, 3]), ("stddev", [3.214550253, 3.055050441])])
def test_indexed_dense_track(tracks, func, want):  # R :657-686
    iv4 = gi(["chr1", "chr2"], [0, 0], [150, 150])
    np.testing.assert_allclose(vt("r515_dn4", func, iv4, filter=gi(["chr1", "chr2"], [20, 20], [30, 30])), want,
                               rtol=1e-6)


def test_last_pos_reads_the_last_part(tracks):  # R :688-712
    vv5 = pd.DataFrame(dict(chrom="chr1", start=[313, 500], end=[340, 510], score=[-1.5, 2]))
    for src in ("r515_sp5", vv5):
        assert vt(src, "last.pos.abs", gi("chr1", 300, 450), filter=gi("chr1", 320, 322)) == [322]


def test_value_based_stddev_at_a_large_offset(tracks):
    starts = np.arange(100) * 10
    vals = 1e9 + 64 * (np.arange(1, 101) % 7)
    big = pd.DataFrame(dict(chrom="chr1", start=starts, end=starts + 5, score=vals))
    # the exact sample sd; R 5.15.1 gives 128.1330 from float per-part means
    np.testing.assert_allclose(vt(big, "stddev", gi("chr1", 0, 1000), filter=gi("chr1", 502, 504)),
                               [np.std(vals, ddof=1)], rtol=1e-9)


def test_value_first_pos_relative_under_a_filter(tracks):  # R test-vtrack-values.R:383-405
    src = pd.DataFrame(dict(chrom="chr1", start=[100], end=[150], score=[3.0]))
    assert vt(src, "first.pos.relative", gi("chr1", 50, 200), filter=gi("chr1", 50, 120)) == [70]


@pytest.mark.parametrize("src", ["r515_sp", "vv"])
@pytest.mark.parametrize("func,wide,narrow", [  # R :201-238; wide = [318, 450), narrow = [318, 330)
    ("first.pos.abs", 318, 318), ("first.pos.relative", 0, 0), ("last.pos.abs", 400, 318),
    ("last.pos.relative", 82, 0), ("min.pos.abs", 318, 318), ("min.pos.relative", 0, 0),
    ("max.pos.abs", 400, 318), ("max.pos.relative", 82, 0),
])
def test_pos_clips_to_the_iterator_start(tracks, src, func, wide, narrow):
    s = VV if src == "vv" else src
    assert vt(s, func, gi("chr1", 318, 450)) == [wide]
    assert vt(s, func, gi("chr1", 318, 330)) == [narrow]


@pytest.mark.parametrize("src", ["r515_sp", "vv"])
def test_sample_pos_clips_to_the_iterator_start(tracks, src):
    s = VV if src == "vv" else src
    assert vt(s, "sample.pos.abs", gi("chr1", 318, 330)) == [318]
    assert vt(s, "sample.pos.relative", gi("chr1", 318, 330)) == [0]
    assert vt(s, "sample.pos.abs", gi("chr1", 318, 450))[0] in (318, 400)


def test_array_pos_clips_to_the_iterator_start(tracks):
    iv = gi("chr1", 318, 450)
    assert vt("r515_arr3", "max.pos.abs", iv) == [318]
    assert vt("r515_arr3", "max.pos.relative", iv) == [0]
    assert vt("r515_arr3", "min.pos.abs", iv) == [400]
    assert vt("r515_arr3", "min.pos.relative", iv) == [82]


def test_array_lse(tracks):
    """pymisha's array lse was NaN; R 5.15.1's is 0 (a bug there). The log-sum-exp of the values."""
    iv = gi("chr1", 0, 1000)
    np.testing.assert_allclose(vt("r515_arr4", "lse", iv), [np.log(np.exp([4, 1, 7, 2]).sum())], rtol=1e-6)
    np.testing.assert_allclose(vt("r515_arr4", "lse", iv, iterator=100)[:4], [4, 1, 7, 2])


def test_nan_dense_bin_is_no_value(tracks):  # R :19-46
    pm.gvtrack_create("ex", "r515_dn", "exists")
    pm.gvtrack_create("sz", "r515_dn", "size")
    pm.gvtrack_create("mp", "r515_dn", "max.pos.abs")
    pm.gvtrack_create("fp", "r515_dn", "first.pos.relative")
    r = pm.gextract(["r515_dn", "ex", "sz", "mp", "fp"], gi("chr1", 0, 150), iterator=50)
    assert r["ex"].tolist() == [1, 0, 1] and r["sz"].tolist() == [1, 0, 1]
    np.testing.assert_array_equal(r["mp"], [0, np.nan, 100])
    np.testing.assert_array_equal(r["fp"], [0, np.nan, 0])
    assert np.isnan(pm.gextract("mp", gi("chr1", 50, 100), iterator=50)["mp"][0])
    r = pm.gextract(["ex", "sz", "mp"], gi("chr1", 40, 110), iterator=10)
    inside = (r["start"] >= 50) & (r["start"] < 100)
    assert (r["ex"][inside] == 0).all() and (r["sz"][inside] == 0).all() and r["mp"][inside].isna().all()
    assert r["mp"][~inside].tolist() == [40, 100]


def test_value_based_on_several_chromosomes(tracks):  # R :48-62
    vals = pd.DataFrame(dict(chrom=["chr1", "chr2"], start=[1000, 10], end=[1100, 20], score=[5.0, 7]))
    pm.gvtrack_create("v_avg", vals, "avg")
    pm.gvtrack_create("v_near", vals, "nearest")
    pm.gvtrack_create("v_ex", vals, "exists")
    q = gi(["chr1", "chr1", "chr2"], [1000, 3000, 10], [1100, 3100, 20])
    r = pm.gextract(["v_avg", "v_near", "v_ex"], q, iterator=q)
    np.testing.assert_array_equal(r["v_avg"], [5, np.nan, 7])
    assert r["v_near"].tolist() == [5, 5, 7]
    assert r["v_ex"].tolist() == [1, 0, 1]


@pytest.mark.parametrize("func,unfiltered,filtered", [  # R :118-141
    ("pwm.edit_distance.pos", 1, 11), ("pwm.edit_distance.lse.pos", -6, -16),
])
def test_edit_distance_positions_under_a_filter(tracks, func, unfiltered, filtered):
    part, whole = gi("chr1", 110, 140), gi("chr1", 100, 140)
    pm.gvtrack_create("pm", None, "pwm.max", pssm=PSSM)
    pm.gvtrack_create("p", None, "pwm", pssm=PSSM)
    sc = pm.gextract(["pm", "p"], part, iterator=part)
    thresh = float(sc["pm" if func == "pwm.edit_distance.pos" else "p"][0]) + 0.5
    assert vt(None, func, part, pssm=PSSM, score_thresh=thresh) == [unfiltered]
    pm.gvtrack_create("ed", None, func, pssm=PSSM, score_thresh=thresh)
    pm.gvtrack_filter("ed", gi("chr1", 100, 110))
    assert pm.gextract("ed", whole, iterator=whole)["ed"].tolist() == [filtered]


@pytest.mark.parametrize("start,want", [(10050, 15896), (10099, 15798)])
def test_gsynth_train_unaligned_scope(start, want):  # R test-review-2026-11-intervals.R:139-153
    assert pm.gsynth_train(intervals=gi("chr1", start, 18000), iterator=100, k=2).total_kmers == want
