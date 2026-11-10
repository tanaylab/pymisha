"""R misha 5.16.0 compute and virtual-track fixes (test-review-2026-11-compute.R, and the
gtrack.create_dense blocks of test-review-2026-11-import.R).

gcor and the summaries are accurate at a large offset and with infinite values; array tracks
give first/last/sample/lse (and .pos) as R; tied gseq_pwm_edits edits come in motif order;
gtrack_smooth keeps running sums; gtrack_create_dense gives defval the bases no interval covers
when intervals overlap, and does not slow down on a long interval."""

import shutil
import time

import numpy as np
import pandas as pd
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm
from pymisha._computer2d import AreaComputer2D, DiagonalBand, Rectangle

SEED = 60427


@pytest.fixture(scope="module")
def db_copy(tmp_path_factory):
    """A copy of the test DB, for the tests that create tracks."""
    dst = tmp_path_factory.mktemp("db516") / "test"
    shutil.copytree(TESTDB_ROOT, dst)
    return dst


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    yield
    pm.gvtrack_clear()


def _multitask(monkeypatch, on, max_processes=8):
    monkeypatch.setitem(pm.CONFIG, "multitasking", on)
    monkeypatch.setitem(pm.CONFIG, "min_scope4process", 1000)
    monkeypatch.setitem(pm.CONFIG, "max_processes", max_processes)


def _seq_db(path, seqs):
    """An indexed database of the given sequences ({chrom: sequence}), opened."""
    path.mkdir(parents=True)
    fasta = path / "genome.fa"
    fasta.write_text("".join(f">{c}\n{s}\n" for c, s in seqs.items()))
    pm.gdb_create(str(path / "db"), str(fasta), db_format="indexed")
    pm.gdb_init(str(path / "db"))
    return path / "db"


# --------------------------------------------------------------------------- gcor


@pytest.mark.parametrize("mt", [False, True])
def test_gcor_large_offset(monkeypatch, mt):
    iv = pm.gintervals(["chr1", "chr2"], 0, 200000)
    x = pm.gextract("dense_track", iv, iterator="dense_track")["dense_track"].to_numpy()
    x = x[~np.isnan(x)] + 1e8
    y = 2 * x - 1e8
    _multitask(monkeypatch, mt)
    r = pm.gcor("dense_track + 1e8", "dense_track * 2 + 1e8", intervals=iv, iterator="dense_track", details=True)
    np.testing.assert_allclose(r.iloc[0]["sd1"], np.std(x, ddof=1), rtol=1e-6)
    np.testing.assert_allclose(r.iloc[0]["sd2"], np.std(y, ddof=1), rtol=1e-6)
    np.testing.assert_allclose(r.iloc[0]["cov"], np.cov(x, y)[0, 1], rtol=1e-6)
    np.testing.assert_allclose(r.iloc[0]["cor"], 1, rtol=1e-9)


def test_gcor_large_offset_vtrack_and_spearman():
    iv = pm.gintervals(["chr1", "chr2"], 0, 200000)
    x = pm.gextract("dense_track", iv, iterator=50)["dense_track"].to_numpy()
    x = x[~np.isnan(x)] + 1e8
    pm.gvtrack_create("vd", "dense_track", "avg")
    r = pm.gcor("vd + 1e8", "vd * 2 + 1e8", intervals=iv, iterator=50, details=True)
    np.testing.assert_allclose(r.iloc[0]["sd1"], np.std(x, ddof=1), rtol=1e-6)
    np.testing.assert_allclose(r.iloc[0]["sd2"], 2 * np.std(x, ddof=1), rtol=1e-6)
    np.testing.assert_allclose(r.iloc[0]["cor"], 1, rtol=1e-9)
    for method in ("spearman", "spearman.exact"):
        assert pm.gcor("vd + 1e8", "vd * 2 + 1e8", intervals=iv, iterator=50, method=method)[0] == pytest.approx(1)
        assert pm.gcor("dense_track + 1e8", "dense_track * 2 + 1e8", intervals=iv, iterator=50,
                       method=method)[0] == pytest.approx(1)


def test_gcor_iterator_track_name_and_intervals():
    """R accepts a track name or intervals as the iterator."""
    iv = pm.gintervals(["chr1", "chr2"], 0, 200000)
    dense = pm.gcor("dense_track", "sparse_track", intervals=iv, iterator="dense_track", details=True)
    pd.testing.assert_frame_equal(
        dense.reset_index(drop=True),
        pm.gcor("dense_track", "sparse_track", intervals=iv, iterator=50, details=True).reset_index(drop=True))

    itr = pm.gintervals(["chr1", "chr1", "chr2"], [0, 1000, 500], [500, 3000, 9000])
    v = pm.gextract(["dense_track", "sparse_track"], itr, iterator=itr)
    ok = v["dense_track"].notna() & v["sparse_track"].notna()
    r = pm.gcor("dense_track", "sparse_track", intervals=iv, iterator=itr, details=True).iloc[0]
    assert r["n"] == 3
    assert r["n.na"] == 3 - ok.sum()
    np.testing.assert_allclose(r["mean1"], v["dense_track"][ok].mean(), rtol=1e-6)
    np.testing.assert_allclose(r["cov"], np.cov(v["dense_track"][ok], v["sparse_track"][ok])[0, 1], rtol=1e-5)

    sp = pm.gcor("dense_track", "dense_track * 3", intervals=iv, iterator="sparse_track", details=True).iloc[0]
    assert sp["n"] == len(pm.gextract("sparse_track", iv))
    np.testing.assert_allclose(sp["cor"], 1, rtol=1e-9)


# --------------------------------------------------------------------------- summaries


def test_summaries_nan_sd_with_infinite_value():
    expr = "np.where(dense_track > 0.15, np.inf, dense_track)"
    iv = pm.gintervals("chr1", 0, 1000)
    x = pm.gextract(expr, iv, iterator=100, colnames=["v"])["v"].to_numpy()
    assert np.isinf(x).any()

    assert np.isnan(pm.gsummary(expr, iv, iterator=100)["Std dev"])
    assert np.isnan(pm.gintervals_summary(expr, iv, iterator=100)["Std dev"].iloc[0])
    assert np.isnan(pm.gbins_summary("dense_track", [0, 0.5], expr=expr, intervals=iv, iterator=100)[0, 6])
    # a finite set still has a finite one
    v = pm.gextract("dense_track", iv, iterator=100)["dense_track"].to_numpy()
    assert pm.gsummary("dense_track", iv, iterator=100)["Std dev"] == pytest.approx(np.std(v, ddof=1), rel=1e-9)

    # the same through a virtual track
    pm.gvtrack_create("vd", "dense_track", "avg")
    vexpr = "np.where(vd > 0.15, np.inf, vd)"
    assert np.isnan(pm.gsummary(vexpr, iv, iterator=100)["Std dev"])
    assert np.isnan(pm.gintervals_summary(vexpr, iv, iterator=100)["Std dev"].iloc[0])


@pytest.mark.parametrize("mt", [False, True])
@pytest.mark.parametrize("inf", ["-np.inf", "np.inf"])
def test_summaries_infinite_min_max(monkeypatch, mt, inf):
    iv = pm.gintervals(["chr1", "chr2"], 0, 50000)
    expr = f"np.where(np.isnan(dense_track), np.nan, {inf})"
    x = pm.gextract(expr, iv, iterator=100, colnames=["v"])["v"].to_numpy()
    x = x[~np.isnan(x)]
    _multitask(monkeypatch, mt, max_processes=4)
    s = pm.gsummary(expr, iv, iterator=100)
    assert [s["Min"], s["Max"]] == [x.min(), x.max()]
    s = pm.gintervals_summary(expr, iv, iterator=100)
    assert s["Min"].tolist() + s["Max"].tolist() == [x.min()] * 2 + [x.max()] * 2
    s = pm.gbins_summary("dense_track", [0, 1], expr=expr, intervals=iv, iterator=100)
    assert [s[0, 2], s[0, 3]] == [x.min(), x.max()]

    pm.gvtrack_create("vd", "dense_track", "avg")
    vexpr = f"np.where(np.isnan(vd), np.nan, {inf})"
    s = pm.gsummary(vexpr, iv, iterator=100)
    assert [s["Min"], s["Max"]] == [x.min(), x.max()]
    s = pm.gintervals_summary(vexpr, iv, iterator=100)
    assert s["Min"].tolist() + s["Max"].tolist() == [x.min()] * 2 + [x.max()] * 2


def test_summary_without_values_has_nan_min_max():
    s = pm.gsummary("dense_track * np.nan", pm.gintervals(["chr1", "chr2"], 0, 50000), iterator=100)
    assert np.isnan(s["Min"]) and np.isnan(s["Max"])


# --------------------------------------------------------------------------- array tracks


def _array_track(name, rows, cols, tmp_path):
    f = tmp_path / f"{name}.tsv"
    pd.DataFrame(rows, columns=["chrom", "start", "end", *cols]).to_csv(f, sep="\t", index=False)
    pm.gtrack_array_import(name, "", str(f))


def test_array_first_last_sample_lse(db_copy, tmp_path):
    pm.gdb_init(str(db_copy))
    starts, ends = [100, 200, 300, 500, 700], [150, 260, 400, 600, 750]
    a, b = np.arange(1, 6), np.arange(11, 16)
    vals = (a + b) / 2
    _array_track("arr_fl", [("chr1", s, e, x, y) for s, e, x, y in zip(starts, ends, a, b, strict=True)],
                 ["a", "b"], tmp_path)
    q = pm.gintervals("chr1", [0, 120, 230, 410], [1000, 330, 240, 490])

    def ext(func):
        pm.gvtrack_create("arr_v", "arr_fl", func)
        return pm.gextract("arr_v", q, iterator=q, colnames=["v"])["v"].to_numpy()

    nan = np.nan
    np.testing.assert_array_equal(ext("first"), [vals[0], vals[0], vals[1], nan])
    np.testing.assert_array_equal(ext("first.pos.abs"), [100, 120, 230, nan])
    np.testing.assert_array_equal(ext("last"), [vals[4], vals[2], vals[1], nan])
    np.testing.assert_array_equal(ext("last.pos.abs"), [700, 300, 230, nan])
    np.testing.assert_array_equal(ext("last.pos.relative"), [700, 180, 0, nan])
    lse = [np.log(np.exp(vals).sum()), np.log(np.exp(vals[:3]).sum()), vals[1], nan]
    np.testing.assert_allclose(ext("lse"), lse, rtol=1e-6)
    for _ in range(5):
        s = ext("sample")
        assert s[0] in vals and s[1] in vals[:3]
        np.testing.assert_array_equal(s[2:], [vals[1], nan])
        sp = ext("sample.pos.abs")
        assert sp[0] in starts and sp[1] in (120, 200, 300)
        np.testing.assert_array_equal(sp[2:], [230, nan])


def test_array_extract_after_one_interval_chrom(db_copy, tmp_path):
    pm.gdb_init(str(db_copy))
    _array_track("arr_ext", [("chr1", 0, 10, 1, 2), ("chr2", 20, 30, 3, 4)], ["a", "b"], tmp_path)
    r = pm.gtrack_array_extract("arr_ext", None, pm.gintervals(["chr1", "chr2"], 0, 100))
    assert r["a"].tolist() == [1, 3]
    assert r["b"].tolist() == [2, 4]


def test_array_lse_with_and_without_filter(db_copy, tmp_path):
    pm.gdb_init(str(db_copy))
    _array_track("arr_lse", [("chr1", 10, 60, 4), ("chr1", 100, 120, 1), ("chr1", 200, 220, 7),
                             ("chr1", 300, 320, 2)], ["a"], tmp_path)
    pm.gvtrack_create("arr_lse_v", "arr_lse", "lse")
    q = pm.gintervals("chr1", 0, 1000)
    got = pm.gextract("arr_lse_v", q, iterator=q)["arr_lse_v"].iloc[0]
    assert got == pytest.approx(np.log(np.exp([4, 1, 7, 2]).sum()), rel=1e-6)
    # [10,60) is split by the filter and read once; [200,220) lies inside [190,230) and is masked
    pm.gvtrack_filter("arr_lse_v", pm.gintervals("chr1", [20, 190, 500], [30, 230, 600]))
    got = pm.gextract("arr_lse_v", q, iterator=q)["arr_lse_v"].iloc[0]
    assert got == pytest.approx(np.log(np.exp([4, 1, 2]).sum()), rel=1e-6)


# --------------------------------------------------------------------------- 2D


def test_bands_wider_than_2_31():
    sq = pm.gintervals_2d("chr1", 0, 100, "chr1", 0, 100)
    cols = ["start1", "end1", "start2", "end2"]
    r = pm.gintervals_2d_band_intersect(sq, band=(10, 3e9))
    assert r[cols].iloc[0].tolist() == [10, 100, 0, 90]
    r = pm.gintervals_2d_band_intersect(sq, band=(-1e10, 1e10))
    assert r[cols].iloc[0].tolist() == [0, 100, 0, 100]


def test_2d_intersect_many_rectangles():
    rng = np.random.default_rng(SEED)

    def mk(n):
        s1, s2 = rng.integers(0, 9001, n), rng.integers(0, 9001, n)
        return pm.gintervals_2d("chr1", s1, s1 + rng.integers(1, 301, n), "chr1", s2, s2 + rng.integers(1, 301, n))

    a, b = mk(1500), mk(1000)
    A, B = a[["start1", "end1", "start2", "end2"]].to_numpy(), b[["start1", "end1", "start2", "end2"]].to_numpy()
    hit = ((A[:, None, 0] < B[None, :, 1]) & (A[:, None, 1] > B[None, :, 0])
           & (A[:, None, 2] < B[None, :, 3]) & (A[:, None, 3] > B[None, :, 2]))
    ia, ib = np.nonzero(hit)
    expected = sorted(zip(np.maximum(A[ia, 0], B[ib, 0]), np.minimum(A[ia, 1], B[ib, 1]),
                          np.maximum(A[ia, 2], B[ib, 2]), np.minimum(A[ia, 3], B[ib, 3]), strict=True))
    r = pm.gintervals_2d_intersect(a, b)
    assert len(r) == len(expected)
    assert sorted(map(tuple, r[["start1", "end1", "start2", "end2"]].to_numpy())) == expected


def test_2d_iterator_needs_dimension_projection():
    pssm = pd.DataFrame(np.tile([0.7, 0.1, 0.1, 0.1], (4, 1)), columns=list("ACGT"))
    it2d = pm.gintervals_2d("chr1", 100, 200, "chr2", 10, 60)
    pm.gvtrack_create("pwm2d", None, "pwm.max", pssm=pssm)
    with pytest.raises(ValueError, match="without explicit dimension projection"):
        pm.gextract("pwm2d", it2d, iterator=it2d)
    pm.gvtrack_create("kmer2d", None, "kmer.count", kmer="ACG")
    with pytest.raises(ValueError, match="without explicit dimension projection"):
        pm.gextract("kmer2d", it2d, iterator=it2d)
    pm.gvtrack_create("val2d", pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [300], "score": [5.0]}), "avg")
    with pytest.raises(ValueError, match="without explicit dimension projection"):
        pm.gextract("val2d", it2d, iterator=it2d)
    pm.gvtrack_create("d1", "dense_track", "avg")
    with pytest.raises(ValueError, match="2D iterator is applied to a 1D track dense_track without explicit"):
        pm.gextract("d1", it2d, iterator=it2d)

    pm.gvtrack_iterator("val2d", dim=1)
    assert pm.gextract("val2d", it2d, iterator=it2d)["val2d"].tolist() == [5]


def test_area_computer_band_area_of_shrunk_rectangle():
    """R measures the band's area within the rectangle shrunk to the band."""
    r = Rectangle(0, 0, 100, 30)
    band = DiagonalBand(-10, 10)
    # cells with -10 <= x - y < 10: y + 10 of them for y < 10, 20 for each y in [10, 30)
    expected = sum(min(100, y + 10) - max(0, y - 10) for y in range(30))
    assert expected == 545
    assert AreaComputer2D().compute(r, band) == expected / r.area


# --------------------------------------------------------------------------- pwm


def test_pwm_na_at_chrom_end():
    pssm = pd.DataFrame(np.tile([0.7, 0.1, 0.1, 0.1], (6, 1)), columns=list("ACGT"))
    sizes = pm.gintervals_all()
    sz = int(sizes.loc[sizes["chrom"] == "chr1", "end"].iloc[0])
    iv = pm.gintervals("chr1", sz - 4, sz)

    def ext(func, **kw):
        pm.gvtrack_create("pwm_end", None, func, pssm=pssm, **kw)
        return pm.gextract("pwm_end", iv, iterator=iv)["pwm_end"].tolist()

    for fn in ("pwm", "pwm.max", "pwm.max.pos"):
        for kw in ({}, {"bidirect": False}, {"bidirect": False, "strand": -1}, {"spat_factor": [1, 2], "spat_bin": 2}):
            assert np.isnan(ext(fn, **kw)[0]), (fn, kw)
    # pwm.count too: nothing to count (R misha 5.18.0)
    assert np.isnan(ext("pwm.count", score_thresh=-100)[0])
    assert np.isnan(ext("pwm.count", score_thresh=-100, bidirect=False)[0])

    # 2 bp bins: the motif still fits from the first three
    pm.gvtrack_create("pwm_end", None, "pwm.max", pssm=pssm)
    got = pm.gextract("pwm_end", pm.gintervals("chr1", sz - 10, sz), iterator=2)["pwm_end"].to_numpy()
    np.testing.assert_array_equal(got, [-np.inf, -np.inf, -np.inf, np.nan, np.nan])


def test_pwm_edit_distance_long_motif(tmp_path):
    rng = np.random.default_rng(SEED)
    bases = np.array(list("ACGT"))
    motif = rng.choice(bases, 32)
    seqv = rng.choice(bases, 1000)

    def other(b):
        return bases[bases != b][0]

    # the motif with one substitution at 100 and with two at 500 (0-based starts)
    seqv[100:132] = motif
    seqv[109] = other(motif[9])
    seqv[500:532] = motif
    seqv[504], seqv[519] = other(motif[4]), other(motif[19])
    _seq_db(tmp_path / "ed", {"chr1": "".join(seqv)})

    pssm = pd.DataFrame([[0.97 if c == m else 0.01 for c in "ACGT"] for m in motif], columns=list("ACGT"))
    thr = float(np.log(pssm.max(axis=1)).sum()) - 0.1
    # 32 columns and max_edits = 1 make two prefilter blocks of 16 columns
    pm.gvtrack_create("ed32", None, "pwm.edit_distance", pssm=pssm, score_thresh=thr, max_edits=1,
                      bidirect=False, prior=0)
    q = pm.gintervals("chr1", [100, 500], [101, 501])
    np.testing.assert_array_equal(pm.gextract("ed32", q, iterator=q)["ed32"].to_numpy(), [1, np.nan])


@pytest.mark.parametrize("max_indels", [None, 1])
def test_pwm_edits_ties_in_motif_order(max_indels):
    pssm = pd.DataFrame(np.tile([0.7, 0.1, 0.1, 0.1], (20, 1)), columns=list("ACGT"))
    thr = 20 * np.log(0.1) + 5 * (np.log(0.7) - np.log(0.1)) - 0.01
    r = pm.gseq_pwm_edits("C" * 20, pssm, score_thresh=thr, bidirect=False, prior=0, max_indels=max_indels)
    assert r["motif_col"].tolist() == [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- intervals


def test_neighbors_ties_do_not_depend_on_input_order():
    rng = np.random.default_rng(SEED)
    starts = rng.choice(np.arange(1000, 100001, 1000), 200)
    i2 = pd.DataFrame({"chrom": "chr1", "start": starts, "end": starts + 10, "id": np.arange(1, 201)})
    s1 = np.arange(1500, 100501, 1000)
    i1 = pd.DataFrame({"chrom": "chr1", "start": s1, "end": s1 + 1})
    unsorted = pm.gintervals_neighbors(i1, i2, maxneighbors=1, mindist=0)
    presorted = pm.gintervals_neighbors(i1, i2.sort_values(["start", "id"], kind="stable"), maxneighbors=1, mindist=0)
    assert unsorted["id"].tolist() == presorted["id"].tolist()


def test_multitask_split_does_not_cut_an_iterator_bin(monkeypatch):
    # 7 bp intervals, each inside one 10 bp bin; the split points fall inside them
    sizes = pm.gintervals_all()
    parts = []
    for chrom, size in zip(sizes["chrom"], sizes["end"], strict=True):
        s = np.arange(0, size - 9, 10) + 1
        parts.append(pm.gintervals(chrom, s, s + 7))
    iv = pd.concat(parts, ignore_index=True)
    _multitask(monkeypatch, True, max_processes=4)
    for cut in (0, 3):
        sub = iv.iloc[: len(iv) - cut]
        assert pm.gsummary("dense_track", sub, iterator=10)["Total intervals"] == len(sub)


# --------------------------------------------------------------------------- gtrack_smooth


def _smooth_reference(v, aside, alg, weight_thr, smooth_nans):
    """The window average of each bin, recomputed over its whole window."""
    n = len(v)
    out = np.full(n, np.nan)
    thr = weight_thr * (aside + 1) if alg == "LINEAR_RAMP" else weight_thr
    for i in range(n):
        if not smooth_nans and np.isnan(v[i]):
            continue
        lo, hi = max(0, i - aside), min(n, i + aside + 1)
        w = aside + 1 - np.abs(np.arange(lo, hi) - i) if alg == "LINEAR_RAMP" else np.ones(hi - lo)
        ok = ~np.isnan(v[lo:hi])
        tw = w[ok].sum()
        if tw > 0 and tw >= thr:
            out[i] = (w[ok] * v[lo:hi][ok]).sum() / tw
    return out


@pytest.mark.parametrize("alg,winsize,weight_thr,smooth_nans", [
    ("LINEAR_RAMP", 500, 0, False), ("MEAN", 500, 0.5, True),
    ("LINEAR_RAMP", 1700, 0.3, True), ("MEAN", 1200, 0, False),
])
def test_smooth_matches_full_window_average(db_copy, alg, winsize, weight_thr, smooth_nans):
    pm.gdb_init(str(db_copy))
    nm = f"sm_{alg.lower()}_{winsize}"
    pm.gtrack_smooth(nm, "", "dense_track * 3 + 7", winsize, weight_thr=weight_thr, smooth_nans=smooth_nans,
                     alg=alg, iterator=50)
    aside = int(0.5 * winsize / 50 + 0.5)
    for chrom in ("chr1", "chrX"):
        iv = pm.gintervals(chrom, 0, 20000)
        src = pm.gextract("dense_track * 3 + 7", pm.gintervals(chrom, 0, 30000), iterator=50,
                          colnames=["v"])["v"].to_numpy()
        got = pm.gextract(nm, iv, iterator=50)[nm].to_numpy()
        np.testing.assert_allclose(got, _smooth_reference(src, aside, alg, weight_thr, smooth_nans)[:len(got)],
                                   rtol=1e-6)
    pm.gtrack_rm(nm, force=True)


def test_smooth_narrow_window_is_an_error(db_copy):
    pm.gdb_init(str(db_copy))
    with pytest.raises(Exception, match="Smoothing window is narrow"):
        pm.gtrack_smooth("sm_narrow", "", "dense_track", 20, iterator=50)
    assert not pm.gtrack_exists("sm_narrow")


def test_smooth_time_does_not_grow_with_the_window(db_copy):
    pm.gdb_init(str(db_copy))

    def elapsed(ws):
        t = time.perf_counter()
        pm.gtrack_smooth("sm_timed", "", "dense_track", ws, iterator=10)
        t = time.perf_counter() - t
        pm.gtrack_rm("sm_timed", force=True)
        return t

    small = min(elapsed(2000) for _ in range(2))
    # rescanning the window per bin made this about 80x the small window
    assert min(elapsed(200000) for _ in range(2)) < 5 * max(small, 0.05)


# --------------------------------------------------------------------------- gtrack_create_dense


@pytest.fixture(scope="module")
def small_db(tmp_path_factory):
    base = tmp_path_factory.mktemp("dense516")
    db = _seq_db(base / "small", {"chr1": "A" * 10000, "chr2": "A" * 10000})
    big = _seq_db(base / "big", {"chr1": "A" * 200000})
    return db, big


def test_create_dense_counts_overlapping_bases_once_for_defval(small_db, tmp_path):
    pm.gdb_init(str(small_db[0]))
    q = pm.gintervals("chr1", 0, 100)

    def one(name):
        return pm.gextract(name, q)[name].iloc[0]

    # two intervals over the same 50 bases of a 100 bp bin, the other 50 uncovered
    ov = pd.DataFrame({"chrom": "chr1", "start": [0, 0], "end": [50, 50]})
    pm.gtrack_create_dense("ov_mean", "", ov, [1, 3], 100, defval=0)
    assert one("ov_mean") == pytest.approx((50 * 1 + 50 * 3 + 50 * 0) / 150, rel=1e-6)
    pm.gtrack_create_dense("ov_min", "", ov, [1, 3], 100, defval=-1, func="min")
    assert one("ov_min") == -1
    pm.gtrack_create_dense("ov_wsum", "", ov, [1, 3], 100, defval=0.5, func="weighted.sum")
    assert one("ov_wsum") == 225
    pm.gtrack_create_dense("ov_cov", "", ov, [1, 3], 100, defval=0.5, func="coverage")
    assert one("ov_cov") == pytest.approx(2.25)
    pm.gtrack_create_dense("ov_cnt", "", ov, [1, 3], 100, defval=0.5, func="count")
    assert one("ov_cnt") == 2

    # the same through a BED file
    bed = tmp_path / "ov.bed"
    bed.write_text("chr1\t0\t50\tp1\t1\nchr1\t0\t50\tp2\t3\n")
    pm.gtrack_import("ov_bed", "", str(bed), 100, defval=0)
    assert one("ov_bed") == pytest.approx(4 / 3, rel=1e-6)

    # random overlapping intervals against a per-base computation
    rng = np.random.default_rng(SEED)
    n = 60
    starts = rng.integers(0, 9801, n)
    ends = np.minimum(starts + rng.integers(1, 501, n), 10000)
    vals = np.round(rng.uniform(0, 10, n), 2)
    defval = 0.5
    pm.gtrack_create_dense("ov_rand", "", pd.DataFrame({"chrom": "chr1", "start": starts, "end": ends}), vals, 100,
                           defval=defval)
    expected = []
    for bs in range(0, 10000, 100):
        overlap = np.maximum(0, np.minimum(ends, bs + 100) - np.maximum(starts, bs))
        covered = sum(bool(np.any((starts <= x) & (ends > x))) for x in range(bs, bs + 100))
        expected.append((np.sum(vals * overlap) + defval * (100 - covered)) / (np.sum(overlap) + 100 - covered))
    got = pm.gextract("ov_rand", pm.gintervals("chr1", 0, 10000))["ov_rand"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_create_dense_direct_matches_create_dense(small_db):
    """Documented as equivalent: the weighted mean over the covered bases, defval for the rest."""
    pm.gdb_init(str(small_db[0]))
    one = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [50]})
    for defval, expected in ((np.nan, 1.0), (2.0, 1.5)):
        pm.gtrack_create_dense_direct("dd_one", "", one, [1], 100, defval=defval)
        assert pm.gextract("dd_one", pm.gintervals("chr1", 0, 100))["dd_one"].iloc[0] == expected
        pm.gtrack_rm("dd_one", force=True)

    rng = np.random.default_rng(SEED)
    n = 400
    chroms = rng.choice(["chr1", "chr2"], n)
    starts = rng.integers(0, 9999, n)
    ends = np.minimum(starts + rng.integers(1, 700, n), 10000)
    vals = np.round(rng.normal(0, 3, n), 3)
    vals[rng.random(n) < 0.1] = np.nan
    iv = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    for defval in (np.nan, 0.5):
        for binsize in (7, 100, 333):
            pm.gtrack_create_dense("cd", "", iv, vals, binsize, defval=defval)
            pm.gtrack_create_dense_direct("dd", "", iv, vals, binsize, defval=defval)
            r = pm.gextract(["cd", "dd"], pm.gintervals_all(), iterator=binsize)
            np.testing.assert_array_equal(r["dd"].to_numpy(), r["cd"].to_numpy())
            pm.gtrack_rm("cd", force=True)
            pm.gtrack_rm("dd", force=True)


def test_create_dense_long_interval_does_not_slow_down(small_db):
    pm.gdb_init(str(small_db[1]))
    n = 100000
    short = pd.DataFrame({"chrom": "chr1", "start": np.arange(n) * 2, "end": np.arange(n) * 2 + 1})
    with_long = pd.concat([pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [200000]}), short],
                          ignore_index=True)

    def elapsed(iv):
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            pm.gtrack_create_dense("timed", "", iv, np.ones(len(iv)), 10)
            ts.append(time.perf_counter() - t)
            pm.gtrack_rm("timed", force=True)
        return min(ts)

    short_t = elapsed(short)
    # one more interval should cost about as much as one more interval, not a rescan per bin
    assert elapsed(with_long) < 5 * max(short_t, 0.005)
