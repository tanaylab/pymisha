"""Virtual tracks as R misha 5.17.0.

Ports the cases of R misha's tests added in 5.17.0 for virtual tracks:
test-vtrack-weighted-sum.R (func "weighted.sum" on 1D sources), test-gvtrack.filter.R (first,
last, sample, nearest, size and exists with a filter), test-vtrack-review-fixes.R (an indexed
track read under two shifts in one call) and test-sliding-sum-precision.R (windows of two sizes
slide and give what each window read afresh gives).

The databases are built here from random sequence and values (seed 60427), as the R tests build
theirs; the expected values come from the raw bins and intervals, as in R.
"""

import numpy as np
import pandas as pd
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm

SEED = 60427


@pytest.fixture(autouse=True)
def _clean_vtracks(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def _make_fasta_db(root, sizes, rng, db_format, seqs=None):
    """A database at root/db with chromosomes of the given sizes (random sequence, or seqs)."""
    root.mkdir(parents=True, exist_ok=True)
    fastas = []
    for chrom, size in sizes.items():
        seq = seqs[chrom] if seqs and chrom in seqs else "".join(rng.choice(list("ACGT"), size))
        fa = root / f"{chrom}.fa"
        fa.write_text(f">{chrom}\n{seq}\n")
        fastas.append(str(fa))
    db = str(root / "db")
    pm.gdb_create(db, fastas, db_format=db_format)
    pm.gdb_init(db)
    return db


# ---------------------------------------------------------------------------------------------
# weighted.sum (test-vtrack-weighted-sum.R)
#
# Every value is a multiple of 1/1024 under 4, so it is stored exactly as a float, and every sum
# of products with bp counts is exact in double: the references are compared without tolerance.
# ---------------------------------------------------------------------------------------------

WS_SIZES = {"chr1": 20013, "chr2": 9991}


def _ws_random_intervals(rng, chrom, length, nan_frac=0.1):
    """Random non-overlapping intervals with gaps, some values NaN."""
    starts, ends = [], []
    pos = int(rng.integers(0, 41))
    while True:
        size = int(rng.integers(1, 121))
        if pos + size > length:
            break
        starts.append(pos)
        ends.append(pos + size)
        pos += size + int(rng.integers(0, 81))
    vals = rng.integers(1, 4001, len(starts)) / 1024
    vals[rng.random(len(vals)) < nan_frac] = np.nan
    return pd.DataFrame({"chrom": chrom, "start": starts, "end": ends, "value": vals})


def _ws_make_db(root, indexed):
    """Dense tracks at bin sizes 1, 20, 32 and 50, a constant track, a sparse track and an array
    track. The chromosome sizes do not divide by the bin sizes, so the last bin is short."""
    rng = np.random.default_rng(SEED)
    db = _make_fasta_db(root, WS_SIZES, rng, "indexed" if indexed else "per-chromosome")
    for binsize in (1, 20, 32, 50):
        bins = pd.concat([pd.DataFrame({"chrom": c, "start": np.arange(0, n, binsize),
                                        "end": np.minimum(np.arange(0, n, binsize) + binsize, n)})
                          for c, n in WS_SIZES.items()], ignore_index=True)
        vals = rng.integers(1, 4001, len(bins)) / 1024
        # runs of bins without a value (defval NaN): a few long ones and scattered single bins
        keep = rng.random(len(bins)) > 0.05
        for run in rng.choice(len(bins) - 40, 4, replace=False):
            keep[run:run + 40] = False
        pm.gtrack_create_dense(f"d{binsize}", "", bins[keep].reset_index(drop=True), vals[keep], binsize,
                               defval=np.nan)
    const = pd.DataFrame({"chrom": list(WS_SIZES), "start": 0, "end": list(WS_SIZES.values())})
    pm.gtrack_create_dense("const20", "", const, np.ones(len(const)), 20, defval=np.nan)

    sp = pd.concat([_ws_random_intervals(rng, c, n) for c, n in WS_SIZES.items()], ignore_index=True)
    pm.gtrack_create_sparse("sp", "", sp[["chrom", "start", "end"]], sp["value"].to_numpy())

    arr = pd.concat([_ws_random_intervals(rng, c, n, nan_frac=0) for c, n in WS_SIZES.items()],
                    ignore_index=True)
    arr["col1"] = arr["value"]
    arr["col2"] = rng.integers(1, 4001, len(arr)) / 1024
    arr.loc[rng.random(len(arr)) < 0.3, "col2"] = np.nan
    arr["col3"] = rng.integers(1, 4001, len(arr)) / 1024
    arr_file = root / "arr.tsv"
    arr[["chrom", "start", "end", "col1", "col2", "col3"]].to_csv(arr_file, sep="\t", index=False, na_rep="")
    pm.gtrack_array_import("arr", "", str(arr_file))

    if indexed:
        for track in ("d1", "d20", "d32", "d50", "const20", "sp", "arr"):
            if not (root / "db" / "tracks" / f"{track}.track" / "track.idx").exists():
                pm.gtrack_convert_to_indexed(track)
        assert (root / "db" / "tracks" / "d20.track" / "track.idx").exists()
    return {"db": db, "sp": sp, "arr": arr}


@pytest.fixture(scope="module")
def ws_dbs(tmp_path_factory):
    root = tmp_path_factory.mktemp("wsum")
    return {indexed: _ws_make_db(root / ("indexed" if indexed else "per_chrom"), indexed)
            for indexed in (False, True)}


@pytest.fixture
def ws_db(ws_dbs):
    def use(indexed=False):
        info = ws_dbs[indexed]
        pm.gdb_init(info["db"])
        pm.gvtrack_clear()
        return info
    return use


def _bp_values(length, starts, ends, vals):
    """Per-bp values of a chromosome (NaN where no value)."""
    bp = np.full(length, np.nan)
    for s, e, v in zip(starts, ends, vals, strict=True):
        if not np.isnan(v):
            bp[s:e] = v
    return bp


def _bp_from_intervals(df, vals=None):
    vals = df["value"].to_numpy() if vals is None else vals
    return {c: _bp_values(n, df["start"].to_numpy()[df["chrom"] == c], df["end"].to_numpy()[df["chrom"] == c],
                          vals[(df["chrom"] == c).to_numpy()])
            for c, n in WS_SIZES.items()}


def _bp_from_track(track):
    """The raw bins of a dense track (or the items of a sparse track) as per-bp values."""
    raw = pm.gextract(track, pm.gintervals_all(), iterator=track, colnames=["v"])
    raw["chrom"] = raw["chrom"].astype(str)
    return _bp_from_intervals(raw, raw["v"].to_numpy(dtype=float))


def _ws_reference(bp, iv, sshift=0, eshift=0, mask=None):
    """Weighted sum of the windows [start + sshift, end + eshift) clipped to the chromosome,
    optionally with a mask whose bp do not count."""
    if mask is not None:
        bp = {c: v.copy() for c, v in bp.items()}
        for c, s, e in zip(mask["chrom"].astype(str), mask["start"], mask["end"], strict=True):
            bp[c][s:e] = np.nan
    out = np.full(len(iv), np.nan)
    chroms = iv["chrom"].astype(str).to_numpy()
    for c in WS_SIZES:
        sel = np.flatnonzero(chroms == c)
        s = np.maximum(0, iv["start"].to_numpy()[sel] + sshift)
        e = np.minimum(WS_SIZES[c], iv["end"].to_numpy()[sel] + eshift)
        csum = np.r_[0.0, np.cumsum(np.nan_to_num(bp[c]))]  # exact: multiples of 1/1024
        cnt = np.r_[0, np.cumsum(~np.isnan(bp[c]))]
        ok = e > s
        se, ee = np.where(ok, s, 0), np.where(ok, e, 0)
        has = ok & (cnt[ee] - cnt[se] > 0)
        out[sel[has]] = csum[ee[has]] - csum[se[has]]
    return out


def _ws_extract(vtrack, iterator):
    return pm.gextract(vtrack, pm.gintervals_all(), iterator=iterator, colnames=["v"])


def _identical(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


def test_weighted_sum_of_a_constant_track_is_the_window_length(ws_db):
    ws_db()
    pm.gvtrack_create("ws", "const20", func="weighted.sum", sshift=-140, eshift=140)
    pm.gvtrack_create("s", "const20", func="sum", sshift=-140, eshift=140)
    res = pm.gextract(["ws", "s"], pm.gintervals_all(), iterator=32)
    length = res["chrom"].astype(str).map(WS_SIZES).to_numpy()
    expected = np.minimum(length, res["end"] + 140) - np.maximum(0, res["start"] - 140)
    _identical(res["ws"], expected)
    inside = (res["start"] >= 140) & (res["end"] + 140 <= length)
    assert (res["ws"][inside] == 312).all()
    # the 312 bp window touches 16 or 17 of the 20 bp bins, by its phase
    assert set(res["s"][inside].unique()) == {16, 17}
    assert res["ws"].iloc[0] == 32 + 140


@pytest.mark.parametrize("indexed", [False, True])
def test_weighted_sum_of_dense_tracks_matches_the_raw_bins(ws_db, indexed):
    ws_db(indexed)
    for binsize in (1, 20, 32, 50):
        track = f"d{binsize}"
        bp = _bp_from_track(track)
        for sshift, eshift in ((-140, 140), (-7, 13), (0, 0), (25, 61)):
            pm.gvtrack_create("ws", track, func="weighted.sum", sshift=sshift, eshift=eshift)
            for iterator in (32, 25):
                res = _ws_extract("ws", iterator)
                _identical(res["v"], _ws_reference(bp, res, sshift, eshift))
        # windows inside a run of bins without a value are NaN
        assert _ws_extract("ws", 32)["v"].isna().any()


def test_weighted_sum_over_unaligned_intervals_of_varying_width(ws_db):
    ws_db()
    bp = _bp_from_track("d20")
    iv = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
                       "start": [0, 3, 1017, 9000, 9970], "end": [1, 1500, 1019, 9991, 9991]})
    pm.gvtrack_create("ws", "d20", func="weighted.sum")
    res = pm.gextract("ws", iv, iterator=iv, colnames=["v"])
    _identical(res["v"], _ws_reference(bp, res))


def test_weighted_sum_over_a_bin_aligned_window_is_sum_times_bin_size(ws_db):
    ws_db()
    for binsize in (20, 50):
        track = f"d{binsize}"
        pm.gvtrack_create("ws", track, func="weighted.sum", sshift=-2 * binsize, eshift=3 * binsize)
        pm.gvtrack_create("s", track, func="sum", sshift=-2 * binsize, eshift=3 * binsize)
        res = pm.gextract(["ws", "s"], pm.gintervals_all(), iterator=2 * binsize)
        # windows that end in the short last bin of a chromosome are not bin-aligned
        length = res["chrom"].astype(str).map(WS_SIZES).to_numpy()
        aligned = (res["end"] + 3 * binsize <= length // binsize * binsize).to_numpy()
        assert (aligned & res["s"].notna().to_numpy()).sum() > 100
        _identical(res["ws"][aligned], res["s"][aligned] * binsize)


def test_weighted_sum_of_a_sparse_track_and_of_a_value_based_vtrack(ws_db):
    info = ws_db()
    sp = info["sp"]
    bp = _bp_from_intervals(sp)
    for sshift, eshift in ((-140, 140), (0, 0), (-3, 50)):
        pm.gvtrack_create("ws_sparse", "sp", func="weighted.sum", sshift=sshift, eshift=eshift)
        pm.gvtrack_create("ws_values", sp, func="weighted.sum", sshift=sshift, eshift=eshift)
        for iterator in (32, 7):
            res = pm.gextract(["ws_sparse", "ws_values"], pm.gintervals_all(), iterator=iterator)
            ref = _ws_reference(bp, res, sshift, eshift)
            _identical(res["ws_sparse"], ref)
            _identical(res["ws_values"], ref)


def test_weighted_sum_of_an_array_track_weighs_each_interval(ws_db):
    info = ws_db()
    arr = info["arr"]
    cols = arr[["col1", "col2", "col3"]].to_numpy()
    pm.gvtrack_create("ws", "arr", func="weighted.sum", sshift=-140, eshift=140)
    res = _ws_extract("ws", 32)
    ref = _ws_reference(_bp_from_intervals(arr, np.nanmean(cols, axis=1)), res, -140, 140)
    np.testing.assert_allclose(res["v"].to_numpy(dtype=float), ref, rtol=1e-6)


def test_weighted_sum_of_an_array_track_slice(ws_db):
    info = ws_db()
    arr = info["arr"]
    pm.gvtrack_create("ws", "arr", func="weighted.sum", sshift=-140, eshift=140)
    pm.gvtrack_array_slice("ws", ["col1", "col3"], "sum")
    res = _ws_extract("ws", 32)
    ref = _ws_reference(_bp_from_intervals(arr, (arr["col1"] + arr["col3"]).to_numpy()), res, -140, 140)
    _identical(res["v"], ref)


def test_weighted_sum_with_a_filter_counts_each_unmasked_bp_once(ws_db):
    info = ws_db()
    mask = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chr2"], "start": [105, 333, 5000, 41],
                         "end": [117, 600, 5013, 9000]})
    sp = info["sp"]
    sp_bp = _bp_from_intervals(sp)
    for track in ("d20", "d50", "sp"):
        bp = sp_bp if track == "sp" else _bp_from_track(track)
        pm.gvtrack_create("ws", track, func="weighted.sum", sshift=-140, eshift=140, filter=mask)
        res = _ws_extract("ws", 32)
        _identical(res["v"], _ws_reference(bp, res, -140, 140, mask))
    pm.gvtrack_create("ws", sp, func="weighted.sum", sshift=-140, eshift=140, filter=mask)
    res = _ws_extract("ws", 32)
    _identical(res["v"], _ws_reference(sp_bp, res, -140, 140, mask))


def test_weighted_sum_is_the_same_with_multitasking_and_on_an_indexed_database(ws_db, monkeypatch):
    results = {}
    for indexed in (False, True):
        for multitask in (False, True):
            ws_db(indexed)
            monkeypatch.setitem(pm.CONFIG, "multitasking", multitask)
            monkeypatch.setitem(pm.CONFIG, "min_processes", 4)
            monkeypatch.setitem(pm.CONFIG, "max_processes", 4)
            monkeypatch.setitem(pm.CONFIG, "min_scope4process", 1)
            # the same track under two shifts and next to other functions
            pm.gvtrack_create("ws1", "d20", func="weighted.sum", sshift=-140, eshift=140)
            pm.gvtrack_create("ws2", "d20", func="weighted.sum", sshift=-60, eshift=10)
            pm.gvtrack_create("s1", "d20", func="sum", sshift=-140, eshift=140)
            pm.gvtrack_create("a2", "d20", func="avg", sshift=-60, eshift=10)
            pm.gvtrack_create("wsp", "sp", func="weighted.sum", sshift=-140, eshift=140)
            res = pm.gextract(["ws1", "ws2", "s1", "a2", "wsp"], pm.gintervals_all(), iterator=32)
            res = res.sort_values(["chrom", "start"]).reset_index(drop=True)
            results[(indexed, multitask)] = res[["chrom", "start", "end", "ws1", "ws2", "s1", "a2", "wsp"]]
    first = results[(False, False)]
    for key, res in results.items():
        pd.testing.assert_frame_equal(res, first, check_exact=True, obj=str(key))
    ws_db()
    bp = _bp_from_track("d20")
    _identical(first["ws1"], _ws_reference(bp, first, -140, 140))
    _identical(first["ws2"], _ws_reference(bp, first, -60, 10))


def test_weighted_sum_does_not_change_sum_and_accepts_no_parameters(ws_db):
    ws_db()
    pm.gvtrack_create("s", "d20", func="sum", sshift=-140, eshift=140)
    alone = _ws_extract("s", 32)
    pm.gvtrack_create("ws", "d20", func="weighted.sum", sshift=-140, eshift=140)
    both = pm.gextract(["s", "ws"], pm.gintervals_all(), iterator=32)
    _identical(both["s"], alone["v"])
    with pytest.raises(ValueError):
        pm.gvtrack_create("bad", "d20", func="weighted.sum", params=1)
    # area stays a 2D function
    with pytest.raises(ValueError, match="not supported by dense tracks"):
        pm.gvtrack_create("bad", "d20", func="area")


# ---------------------------------------------------------------------------------------------
# Filters (test-gvtrack.filter.R)
# ---------------------------------------------------------------------------------------------

def _vf_blocks(nblocks):
    """Blocks of 400 bp on chr1, each with the values 1, 2 and 6 at [0,50), [100,150) and
    [250,300) of the block."""
    off = np.repeat(np.arange(nblocks) * 400, 3)
    return pd.DataFrame({"chrom": "chr1", "start": off + np.tile([0, 100, 250], nblocks),
                         "end": off + np.tile([50, 150, 300], nblocks),
                         "score": np.tile([1.0, 2.0, 6.0], nblocks)})


@pytest.fixture
def vf_sources(tmp_path):
    """vf_sources(vv): a sparse, a dense (50 bp bins, the bins between the values hold none) and
    an array track of the values of vv, in a new per-chromosome database; returns the sources to
    compare, the value-based vtrack included, and a function converting the tracks to indexed."""
    def make(vv):
        rng = np.random.default_rng(SEED)
        _make_fasta_db(tmp_path / "vf", {"chr1": int(vv["end"].max()) + 1000, "chr2": 12}, rng,
                       "per-chromosome", seqs={"chr2": "ACGTACGTACGT"})
        pm.gtrack_create_sparse("vf_sp", "x", vv[["chrom", "start", "end"]], vv["score"].to_numpy())
        pm.gtrack_create_dense("vf_dn", "x", vv[["chrom", "start", "end"]], vv["score"].to_numpy(), 50, np.nan)
        f = tmp_path / "vf_arr.tsv"
        vv.rename(columns={"score": "a"}).to_csv(f, sep="\t", index=False)
        pm.gtrack_array_import("vf_arr", "x", str(f))

        def to_indexed():
            for track in ("vf_sp", "vf_dn", "vf_arr"):
                pm.gtrack_convert_to_indexed(track)
                assert (tmp_path / "vf" / "db" / "tracks" / f"{track}.track" / "track.idx").exists()
        return {"sp": "vf_sp", "dn": "vf_dn", "arr": "vf_arr", "vv": vv}, to_indexed
    return make


def _iv(chrom, starts, ends):
    starts, ends = np.atleast_1d(starts), np.atleast_1d(ends)
    return pd.DataFrame({"chrom": chrom, "start": starts, "end": ends})


def _vf_value(src, func, mask, iv):
    pm.gvtrack_create("vf", src, func, filter=mask)
    return pm.gextract("vf", iv, iterator=iv)["vf"].to_numpy(dtype=float)


def test_filter_first_and_last_skip_unmasked_parts_without_a_value(vf_sources):
    srcs, to_indexed = vf_sources(_vf_blocks(2))
    iv = _iv("chr1", 400, 800)
    # parts [460,470) and [480,800): the first part holds no value
    m_first = _iv("chr1", [400, 470], [460, 480])
    # parts [400,550), [560,720) and [730,800): the last part holds no value
    m_last = _iv("chr1", [550, 720], [560, 730])
    expected = {"first": (m_first, 2), "first.pos.abs": (m_first, 500), "first.pos.relative": (m_first, 100),
                "last": (m_last, 6), "last.pos.abs": (m_last, 650), "last.pos.relative": (m_last, 250)}

    def check(label):
        for func, (mask, want) in expected.items():
            for name, src in srcs.items():
                got = _vf_value(src, func, mask, iv)
                assert got.tolist() == [want], (label, name, func, got)

    check("per-chromosome")
    to_indexed()
    check("indexed")


@pytest.mark.parametrize("mask_start", [420, 460])
def test_filter_nearest_is_the_average_of_the_unmasked_values(vf_sources, mask_start):
    srcs, _ = vf_sources(_vf_blocks(2))
    iv = _iv("chr1", 400, 800)
    # [420,430) splits the first value; [460,470) falls between the first two
    mask = _iv("chr1", mask_start, mask_start + 10)
    for name, src in srcs.items():
        got = _vf_value(src, "nearest", mask, iv)
        np.testing.assert_allclose(got, [np.mean([1, 2, 6])], rtol=1e-6, err_msg=name)


def test_filter_nearest_without_an_unmasked_value_is_the_nearest_unmasked_value(vf_sources):
    vv = _vf_blocks(2)
    srcs, to_indexed = vf_sources(vv)
    # the values: 1 at [0,50), 2 at [100,150), 6 at [250,300), and the same again 400 bp later
    cases = [
        # the mask leaves [100,140) of the value 2, 20 bp from the interval; 6 is 10 bp away
        (_iv("chr1", 160, 240), _iv("chr1", 140, 155), 6),
        # the value 2 lies inside the mask, not a candidate
        (_iv("chr1", 80, 240), _iv("chr1", 80, 160), 6),
        # 1 and 2 both 10 bp away: the left one, as without a filter
        (_iv("chr1", 60, 90), _iv("chr1", 1500, 1510), 1),
        # distances are from the nearest unmasked part: [170,195) to 2 is 20 bp, [205,240) to 6 is 10 bp
        (_iv("chr1", 160, 240), _iv("chr1", [160, 195], [170, 205]), 6),
        # every value of the chromosome is masked
        (_iv("chr1", 160, 240), pd.concat([vv[["chrom", "start", "end"]], _iv("chr1", 170, 180)]), np.nan),
        # the chromosome holds no value
        (_iv("chr2", 0, 12), _iv("chr2", 3, 5), np.nan),
    ]

    def check(label):
        for k, (iv, mask, want) in enumerate(cases):
            for name, src in srcs.items():
                # nearest on a dense track does not look outside the interval, with or without a filter
                expected = np.nan if name == "dn" else want
                got = _vf_value(src, "nearest", mask, iv)
                np.testing.assert_array_equal(got, [expected], err_msg=f"{label} {name} case {k + 1}")

    check("per-chromosome")
    to_indexed()
    check("indexed")


def test_filter_sample_draws_each_unmasked_value_with_the_same_chance(vf_sources):
    nblocks = 1200
    srcs, _ = vf_sources(_vf_blocks(nblocks))
    starts = np.arange(nblocks) * 400
    iv = _iv("chr1", starts, starts + 400)
    np.random.seed(SEED)
    # [150,160) leaves two values in the first part and one in the second; [120,130) splits the
    # second value, which is still one value
    for cut in ((150, 160), (120, 130)):
        mask = _iv("chr1", starts + cut[0], starts + cut[1])
        for name, src in srcs.items():
            pm.gvtrack_create("vf", src, "sample", filter=mask)
            pm.gvtrack_create("vf_pos", src, "sample.pos.relative", filter=mask)
            r = pm.gextract(["vf", "vf_pos"], iv, iterator=iv)
            assert len(r) == nblocks
            # with 1200 draws the standard error of each share is about 0.014
            share = np.array([(r["vf"] == v).mean() for v in (1, 2, 6)])
            assert np.all(np.abs(share - 1 / 3) < 0.06), (name, cut, share)
            share = np.array([(r["vf_pos"] == p).mean() for p in (0, 100, 250)])
            assert np.all(np.abs(share - 1 / 3) < 0.06), (name, cut, "pos", share)


def test_filter_size_and_exists_are_0_when_the_unmasked_parts_hold_no_value(vf_sources):
    vv = pd.DataFrame({"chrom": "chr1", "start": [100, 500], "end": [110, 510], "score": [1.0, 2.0]})
    srcs, _ = vf_sources(vv)
    iv = _iv("chr1", 200, 300)
    # parts [200,250) and [260,300), neither holds a value
    mask = _iv("chr1", 250, 260)
    for name, src in srcs.items():
        for func in ("size", "exists", "avg", "sum"):
            expected = 0 if func in ("size", "exists") else np.nan
            np.testing.assert_array_equal(_vf_value(src, func, mask, iv), [expected], err_msg=f"{name} {func}")
        # an interval masked entirely gives NaN
        for func in ("size", "exists"):
            assert np.isnan(_vf_value(src, func, _iv("chr1", 150, 350), iv)).all(), (name, func)


# ---------------------------------------------------------------------------------------------
# An indexed track under two shifts in one call (test-vtrack-review-fixes.R)
# ---------------------------------------------------------------------------------------------

def test_an_indexed_track_read_under_different_shifts_keeps_each_shifts_values(tmp_path):
    _make_fasta_db(tmp_path / "shifts", {"chr1": 2000, "chr2": 500}, np.random.default_rng(SEED), "indexed")
    # bin k holds (k + 1)^2, so a window's sum is not the value of its middle bin
    v = np.arange(1, 21, dtype=float) ** 2
    pm.gtrack_create_dense("dn5", "x", _iv("chr1", np.arange(0, 951, 50), np.arange(50, 1001, 50)), v, 50, np.nan)
    if not (tmp_path / "shifts" / "db" / "tracks" / "dn5.track" / "track.idx").exists():
        pm.gtrack_convert_to_indexed("dn5")
    pm.gvtrack_create("d5_wide", "dn5", "sum", sshift=-100, eshift=100)
    pm.gvtrack_create("d5_narrow", "dn5", "sum")
    iv = _iv("chr1", [200, 500], [250, 550])
    # [200, 250) is bin 4 and [500, 550) bin 10; widened, [100, 350) holds bins 2..6 and [400, 650) bins 8..12
    narrow = v[[4, 10]].tolist()
    wide = [v[2:7].sum(), v[8:13].sum()]
    r = pm.gextract(["d5_wide", "d5_narrow"], iv, iterator=iv)
    assert r["d5_wide"].tolist() == wide
    assert r["d5_narrow"].tolist() == narrow
    # the track itself next to a shifted vtrack of it
    r = pm.gextract(["dn5", "d5_wide"], iv, iterator=iv)
    assert r["dn5"].tolist() == narrow
    assert r["d5_wide"].tolist() == wide


# ---------------------------------------------------------------------------------------------
# Sliding windows of two sizes (test-sliding-sum-precision.R)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shifts", [(-140, 173), (-1000, 1000), (7, 290)])
def test_windows_of_two_sizes_slide_and_give_what_a_fresh_read_gives(shifts):
    """A 60 bp iterator over dense_track's 50 bp bins gives windows of two sizes, which slide from
    one to the next (dropping and adding different numbers of bins) for sum, exists and size, and
    for sum with avg, while the sum stays exact; lse slides between windows of one size. Rows
    600 bp apart are each read afresh."""
    pm.gdb_init(str(TESTDB_ROOT))
    sets = [["sum"], ["lse"], ["exists"], ["size"], ["sum", "lse", "exists", "size"], ["sum", "avg"],
            ["exists", "size"]]
    end1 = int(pm.gintervals_all()["end"].iloc[0])
    query = pd.DataFrame({"chrom": "chr1", "start": [0, end1 - 30000], "end": [300000, end1]})
    for funcs in sets:
        pm.gvtrack_clear()
        vts = [f"vt_{f}" for f in funcs]
        for vt, func in zip(vts, funcs, strict=True):
            pm.gvtrack_create(vt, "dense_track", func=func)
            pm.gvtrack_iterator(vt, sshift=shifts[0], eshift=shifts[1])
        slid = pm.gextract(vts, query, iterator=60)
        rows = np.arange(0, len(slid), 10)
        far = slid.iloc[rows][["chrom", "start", "end"]].reset_index(drop=True)
        fresh = pm.gextract(vts, far, iterator=far)
        assert fresh[vts[0]].notna().sum() > 100
        for vt in vts:
            _identical(slid[vt].to_numpy()[rows], fresh[vt])


# ---------------------------------------------------------------------------------------------
# Dense tracks read in bulk (test-dense-bulk-read.R)
#
# A dense track read at its own bin size, with no virtual track in the expressions, is read a run
# of bins at a time instead of an iterator interval at a time. Each call below is run both ways -
# MISHA_DISABLE_DENSE_BULK_READ=1 forces the per-interval path - and the results must be the same
# to the bit, -0, a stored NaN with a sign or a payload and inf included. The calls that may not
# use the bulk path (other bin sizes, virtual tracks, other iterators) are checked too.
# ---------------------------------------------------------------------------------------------

BULK_SIZES = {"chr1": 100007, "chr2": 55555, "chr3": 20000, "chrX": 3}


def _bulk_create(track, binsize, rng, nan_chrom=None):
    bins = pd.concat([pd.DataFrame({"chrom": c, "start": np.arange(0, n, binsize),
                                    "end": np.minimum(np.arange(0, n, binsize) + binsize, n)})
                      for c, n in BULK_SIZES.items()], ignore_index=True)
    v = np.round(rng.normal(size=len(bins)), 3)
    for st in rng.choice(len(bins), 20, replace=False):
        v[st:st + int(rng.integers(1, 101)) + 1] = np.nan
    chroms = bins["chrom"].to_numpy()
    v[[np.flatnonzero(chroms == "chr1").max(), np.flatnonzero(chroms == "chr2").min()]] = np.nan
    if nan_chrom is not None:
        v[chroms == nan_chrom] = np.nan
    pm.gtrack_create_dense(track, "bulk", bins, v, binsize, np.nan)


@pytest.fixture(scope="module")
def bulk_db(tmp_path_factory):
    """Chromosome ends off the 50 bp grid and a chromosome shorter than a bin, per-chromosome
    and indexed copies of the tracks, NaN runs, a chromosome with no values, and a track whose
    files hold -0, -NaN, a NaN with a payload and infinities."""
    root = tmp_path_factory.mktemp("bulk")
    rng = np.random.default_rng(SEED)
    db = _make_fasta_db(root, BULK_SIZES, rng, "per-chromosome")
    _bulk_create("d50a", 50, rng)
    _bulk_create("d50b", 50, rng, nan_chrom="chr3")
    _bulk_create("d20", 20, rng)
    _bulk_create("d1", 1, rng)
    pm.gtrack_copy("d50a", "d50z")
    tdir = root / "db" / "tracks" / "d50z.track"

    def write_bins(chrom, bins, raw):
        with open(tdir / chrom, "r+b") as f:
            for b in bins:
                f.seek(4 + 4 * b)
                f.write(bytes(raw))

    write_bins("chr1", [0, 21, 22, 500, 2000], [0, 0, 0, 0x80])  # -0
    write_bins("chr1", [23, 501], [0, 0, 0xc0, 0xff])  # -NaN
    write_bins("chr1", [24, 502], [1, 0, 0xa0, 0x7f])  # NaN with a payload
    write_bins("chr1", [25], [0, 0, 0x80, 0x7f])  # inf
    write_bins("chr2", [0, 1, 1111], [0, 0, 0, 0x80])
    write_bins("chr2", [2], [0, 0, 0x80, 0xff])  # -inf
    write_bins("chrX", [0], [0, 0, 0, 0x80])
    pm.gdb_reload()
    for track in ("d50a", "d50b", "d20", "d1", "d50z"):
        indexed = "i" + track[1:]
        pm.gtrack_copy(track, indexed)
        pm.gtrack_convert_to_indexed(indexed)
        assert (root / "db" / "tracks" / f"{indexed}.track" / "track.idx").exists()
    pm.gdb_reload()
    return db


def _bulk_scope():
    return pd.DataFrame({"chrom": ["chr1", "chr1", "chr2", "chr3", "chrX"], "start": [1005, 7777, 3, 19990, 0],
                         "end": [5003, 50001, 55555, 20000, 3]})


def _assert_bits(a, b, what=""):
    if isinstance(a, pd.DataFrame):
        assert list(a.columns) == list(b.columns), what
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=what)
        for col in a.columns:
            if pd.api.types.is_float_dtype(a[col]):
                assert a[col].to_numpy().tobytes() == b[col].to_numpy().tobytes(), (what, col)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_exact=True, obj=what)
        assert np.asarray(a, dtype=float).tobytes() == np.asarray(b, dtype=float).tobytes(), what
    elif a is None:
        assert b is None, what
    else:
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape, what
        assert a.tobytes() == b.tobytes(), what


@pytest.fixture
def same_both_paths(bulk_db, monkeypatch):
    pm.gdb_init(bulk_db)
    monkeypatch.setitem(pm.CONFIG, "max_data_size", int(1e9))

    def check(fn, what=""):
        monkeypatch.setenv("MISHA_DISABLE_DENSE_BULK_READ", "1")
        per_interval = fn()
        monkeypatch.delenv("MISHA_DISABLE_DENSE_BULK_READ")
        bulk = fn()
        _assert_bits(bulk, per_interval, what)
        return bulk
    return check


def test_a_dense_track_at_its_own_bin_size_reads_the_same_in_bulk(same_both_paths, monkeypatch, tmp_path):
    check = same_both_paths
    scope = _bulk_scope()
    all_iv = pm.gintervals_all()
    check(lambda: pm.gextract("d50a", all_iv), "all")
    check(lambda: pm.gextract("d50a", scope, iterator=50), "scope 50")
    check(lambda: pm.gextract("i50a", scope, iterator="i50a"), "iterator track")
    check(lambda: pm.gextract(["d50a", "d50b"], all_iv, iterator="i50b"), "two tracks")
    check(lambda: pm.gextract(["d50a", "d50b", "i50a", "i50b"], scope), "four tracks")
    check(lambda: pm.gextract("d50a * 2 + np.log(np.abs(i50b) + 1)", scope, iterator=50), "expression")
    check(lambda: pm.gextract(["d50a", "np.where(np.isnan(d50b), 0, d50b)", "np.fmax(d50a, i50a)"], all_iv,
                              colnames=["a", "b", "c"]), "expressions")
    check(lambda: pm.gextract(["d20", "i20"], all_iv), "20 bp")
    # tracks of different bin sizes, read through an explicit iterator (not in bulk)
    check(lambda: pm.gextract(["d50a", "d20"], scope, iterator=50), "mixed bin sizes")
    check(lambda: pm.gextract("d1 - i1", pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [10, 99],
                                                       "end": [100007, 5000]}), iterator=1), "1 bp")
    res = check(lambda: pm.gextract(["d50z", "i50z", "1 / d50z", "1 / i50z"], all_iv), "signed zeros")
    assert not np.signbit(res["d50z"].to_numpy()[[0, 21]]).any()  # a stored -0 reads as +0, as R
    assert np.isnan(res["d50z"].to_numpy()[[23, 24, 25]]).all()
    # intervals inside one bin, and overlapping ones
    check(lambda: pm.gextract("d50a", pd.DataFrame({"chrom": "chr1", "start": [100, 120, 140, 1049],
                                                    "end": [110, 130, 160, 1051]}), iterator=50), "in one bin")
    check(lambda: pm.gextract("d50a", pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [100, 120, 500],
                                                    "end": [300, 251, 777]}), iterator=50), "overlapping")
    check(lambda: pm.gextract("d50a", scope, iterator=50, intervals_join="intervals"), "intervals_join")
    check(lambda: pm.gextract(["d50b", "i50b"], pm.gintervals("chr3")), "NaN chromosome")
    check(lambda: pm.gextract("d50a", pm.gintervals("chrX")), "3 bp chromosome")
    # the iterator intervals as an expression sees them, and an expression with no track
    check(lambda: pm.gextract(["d50a", "END - START", "START + 0"], scope, iterator=50,
                              colnames=["v", "len", "pos"]), "coordinates")
    check(lambda: pm.gextract("1 + 2", scope, iterator=50), "no track")
    # batches that end inside a scope interval
    monkeypatch.setitem(pm.CONFIG, "eval_buf_size", 7)
    check(lambda: pm.gextract(["d50a", "i50b"], scope), "buffer of 7")
    monkeypatch.setitem(pm.CONFIG, "eval_buf_size", 1000)

    check(lambda: pm.gsummary("d50z"), "gsummary")
    check(lambda: pm.gsummary("1 / i50z"), "gsummary 1/x")
    check(lambda: pm.gsummary("d1"), "gsummary 1 bp")
    check(lambda: pm.gquantiles("i50a", [0.1, 0.5, 0.9], scope), "gquantiles")
    check(lambda: pm.gscreen("(d50a > 0) & (i50b < 0)", scope), "gscreen")
    check(lambda: pm.gdist("d50a", np.arange(-3, 4), "i50b", [-2, 0, 2], intervals=scope), "gdist")
    check(lambda: pm.gcor("d1", "i1", intervals=scope), "gcor")
    check(lambda: pm.gintervals_summary("d50a", scope, iterator=50), "gintervals_summary")
    check(lambda: pm.gbins_summary("d50a", np.arange(-2, 3), expr="d50b"), "gbins_summary")
    check(lambda: pm.gpartition("d50a", [-1, 0, 1], all_iv), "gpartition")

    # (more than the values there are: all of them, in an order the draws decide)
    check(lambda: np.sort(pm.gsample("d50a", 10**7, scope)), "gsample")

    def created():
        if "bulk_created" in pm.gtrack_ls():
            pm.gtrack_rm("bulk_created", force=True)
        pm.gtrack_create("bulk_created", "", "d50a * 3 + i50b", iterator=50)
        res = pm.gextract("bulk_created", all_iv)
        pm.gtrack_rm("bulk_created", force=True)
        return res
    check(created, "gtrack_create")


def test_a_dense_track_reads_the_same_in_bulk_in_multitasking(same_both_paths, monkeypatch):
    check = same_both_paths
    monkeypatch.setitem(pm.CONFIG, "multitasking", True)
    monkeypatch.setitem(pm.CONFIG, "min_processes", 4)
    monkeypatch.setitem(pm.CONFIG, "max_processes", 4)
    monkeypatch.setitem(pm.CONFIG, "min_scope4process", 1000)
    scope = _bulk_scope()
    all_iv = pm.gintervals_all()
    check(lambda: pm.gextract(["d50a", "i50b", "d50z"], all_iv), "gextract")
    check(lambda: pm.gextract("d1 + i1", scope, iterator="d1"), "1 bp")
    check(lambda: pm.gsummary("d50a + d50b", scope), "gsummary")
    check(lambda: pm.gquantiles("d1", [0.01, 0.5, 0.99]), "gquantiles")
    check(lambda: pm.gscreen("d20 > 0.5"), "gscreen")
    check(lambda: pm.gdist("i50z", np.arange(-3, 3.5, 0.5)), "gdist")


def test_intervals_that_do_not_lie_in_one_bin_of_each_track_are_read_one_by_one(same_both_paths):
    check = same_both_paths
    scope = _bulk_scope()
    pm.gvtrack_create("v_shift", "d50a", "avg", sshift=-50, eshift=50)
    pm.gvtrack_create("v_exists", "d50a", "exists")
    pm.gvtrack_create("v_avg", "d50a", "avg")
    pm.gvtrack_create("v_kmer", None, "kmer.count", kmer="AA", strand=1)
    pm.gvtrack_create("v_dist", pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [1000, 7000],
                                              "end": [1500, 9000]}), "distance")
    pm.gvtrack_create("v_filter", "d50a", "avg", filter=pm.gintervals("chr1", 2000, 3000))
    pm.gvtrack_create("v_pv", "d50a", "global.percentile")
    # an iterator of another bin size
    check(lambda: pm.gextract("d50a", scope, iterator=100), "iterator 100")
    check(lambda: pm.gextract("d50a", scope, iterator=20), "iterator 20")
    check(lambda: pm.gsummary("i50a", iterator=100), "gsummary 100")
    # virtual tracks, other iterators
    check(lambda: pm.gextract("v_shift", scope, iterator=50), "shift")
    check(lambda: pm.gextract(["d50a", "v_exists"], scope, iterator=50), "exists")
    check(lambda: pm.gextract("v_avg", scope, iterator=50), "avg")
    check(lambda: pm.gextract(["d50a", "v_kmer"], scope, iterator=50), "kmer")
    check(lambda: pm.gextract(["d50a", "v_dist"], scope, iterator=50), "distance")
    check(lambda: pm.gextract(["d50a", "v_filter"], scope, iterator=50), "filter")
    check(lambda: pm.gextract(["d50a", "v_pv"], scope, iterator=50), "global.percentile")
    check(lambda: pm.gextract("d50a", scope, iterator=pd.DataFrame({"chrom": "chr1", "start": [100, 1049],
                                                                    "end": [110, 1051]})), "intervals iterator")
    check(lambda: pm.gscreen("v_shift > 0", scope, iterator=50), "gscreen shift")


def test_dense_tracks_of_different_bin_sizes_need_an_explicit_iterator(bulk_db):
    pm.gdb_init(bulk_db)
    scope = _bulk_scope()
    with pytest.raises(Exception, match="track expressions contain tracks with different bin sizes"):
        pm.gextract(["d50a", "d20"], scope)
    with pytest.raises(Exception, match='track expression "d50a \\+ d20" contains tracks with different bin sizes'):
        pm.gsummary("d50a + d20", scope)
    pm.gvtrack_create("v20", "d20", "avg")
    with pytest.raises(ValueError, match="different bin sizes"):
        pm.gextract(["v20", "d50a"], scope)
    with pytest.raises(ValueError, match="different bin sizes"):
        pm.gsummary("v20 + d50a", scope)
    res = pm.gextract(["d50a", "d20", "d50a - d20"], pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [200]}),
                      iterator=50)
    np.testing.assert_array_equal(res["d50a - d20"], res["d50a"] - res["d20"])


# ---------------------------------------------------------------------------------------------
# A stored -0, as R misha 5.17.0 reads it: the average of one value is (float)(0. + v), so the
# track itself, avg and nearest read +0; sum of one bin, min, max, first, last, quantile and lse
# keep the stored value; a sum over several bins starts from 0
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bulk", [False, True])
def test_a_stored_minus_zero_reads_as_plus_zero(bulk_db, monkeypatch, bulk):
    pm.gdb_init(bulk_db)
    if not bulk:
        monkeypatch.setenv("MISHA_DISABLE_DENSE_BULK_READ", "1")
    # d50z holds -0 in chr1 bins 21 and 22
    one = _iv("chr1", 1050, 1100)
    two = _iv("chr1", 1050, 1150)
    plus, minus = False, True
    for iv, want in ((one, {"avg": plus, "nearest": plus, "sum": minus, "min": minus, "max": minus, "first": minus,
                            "last": minus, "quantile": minus, "lse": minus, "weighted.sum": plus}),
                     (two, {"avg": plus, "nearest": plus, "sum": plus, "min": minus, "max": minus, "first": minus,
                            "last": minus, "quantile": minus, "weighted.sum": plus})):
        got = pm.gextract("d50z", iv, iterator=iv)["d50z"].to_numpy()
        assert got[0] == 0 and not np.signbit(got[0])
        for func, sign in want.items():
            pm.gvtrack_create("z", "d50z", func, 0.5 if func == "quantile" else None)
            got = pm.gextract("z", iv, iterator=iv)["z"].to_numpy()
            assert got[0] == 0 and np.signbit(got[0]) == sign, (iv["end"][0], func)
    res = pm.gextract("d50z", _iv("chr1", 1000, 1200), iterator=50)["d50z"].to_numpy()
    assert not np.signbit(res[[1, 2]]).any()


# ---------------------------------------------------------------------------------------------
# Array slices (gvtrack_array_slice) keep the vtrack's function, shifts and filter, as R. The
# expected values are R misha 5.17.0's on the test database.
# ---------------------------------------------------------------------------------------------

def test_an_array_slice_keeps_the_vtracks_function_shifts_and_filter():
    pm.gdb_init(str(TESTDB_ROOT))
    nan = np.nan
    pm.gvtrack_create("v", "array_track", "max", sshift=-300, eshift=300)
    pm.gvtrack_array_slice("v", ["col1", "col3"], "sum")
    pm.gvtrack_create("w", "array_track", "weighted.sum", sshift=-300, eshift=300)
    pm.gvtrack_array_slice("w", ["col1", "col3"], "max")
    pm.gvtrack_create("f", "array_track", "first", filter=pm.gintervals("chr1", 1000, 2500))
    pm.gvtrack_array_slice("f", ["col2"], "avg")
    pm.gvtrack_create("s", "array_track", "sum", filter=pm.gintervals("chr1", 1000, 2500))
    pm.gvtrack_array_slice("s", None, "max")
    assert pm.gvtrack_info("v")["func"] == "max"
    assert pm.gvtrack_info("v")["slice"] == {"slice": [1, 3], "func": "sum", "params": None}
    res = pm.gextract(["v", "w", "f", "s"], pm.gintervals("chr1", 0, 6000), iterator=1000)
    np.testing.assert_array_equal(res["v"], [404, 604, nan, nan, 401, 401])
    np.testing.assert_array_equal(res["w"], [10150, 15150, nan, nan, 20050, 20050])
    np.testing.assert_array_equal(res["f"], [2, nan, nan, nan, 402, nan])
    np.testing.assert_array_equal(res["s"], [325, nan, nan, nan, 409, nan])
    # with no iterator, the array's intervals
    pm.gvtrack_create("d", "array_track")
    pm.gvtrack_array_slice("d", ["col1", "col3"], "sum")
    res = pm.gextract("d", pm.gintervals("chr1", 0, 3000))
    assert res["start"].tolist() == [0, 100, 250, 1400]
    np.testing.assert_array_equal(res["d"], [nan, nan, 404, 604])
    # the Python path (a user variable in the expression) gives the same
    k = 1.0  # noqa: F841 (read by the expression)
    res2 = pm.gextract("v * k", pm.gintervals("chr1", 0, 6000), iterator=1000)
    np.testing.assert_array_equal(res2["v * k"], [404, 604, nan, nan, 401, 401])


# ---------------------------------------------------------------------------------------------
# Value-based vtracks hold their source as R's in-memory track does: values as floats, a missing
# value kept (no function counts it), a missing coordinate an error. Values are R misha 5.17.0's.
# ---------------------------------------------------------------------------------------------

def test_value_based_vtracks_hold_their_values_as_floats():
    pm.gdb_init(str(TESTDB_ROOT))
    df = pd.DataFrame({"chrom": "chr1", "start": [0, 10], "end": [5, 20], "v": [0.1, 0.2]})
    iv = _iv("chr1", 0, 30)
    want = {"sum": 0.30000001192092896, "avg": 0.15000000596046448, "first": 0.10000000149011612,
            "max": 0.20000000298023224}
    for func, value in want.items():
        pm.gvtrack_create("vb", df, func)
        assert pm.gextract("vb", iv, iterator=iv)["vb"].tolist() == [value], func
    # a missing value is kept: the nearest interval holds none
    df2 = pd.DataFrame({"chrom": "chr1", "start": [0, 15], "end": [10, 20], "v": [1.0, np.nan]})
    pm.gvtrack_create("vb", df2, "nearest")
    assert np.isnan(pm.gextract("vb", _iv("chr1", 22, 30), iterator=_iv("chr1", 22, 30))["vb"]).all()
    pm.gvtrack_create("vb", df2, "avg")
    assert pm.gextract("vb", iv, iterator=iv)["vb"].tolist() == [1.0]


def test_a_missing_coordinate_in_a_vtrack_source_is_an_invalid_interval():
    pm.gdb_init(str(TESTDB_ROOT))
    for start, end in (([0, np.nan], [10, 20]), ([0, 15], [10, np.inf]), ([0, None], [10, 20])):
        df = pd.DataFrame({"chrom": "chr1", "start": start, "end": end, "v": [1.0, 2.0]})
        with pytest.raises(ValueError, match="^Invalid format of interval at index 2$"):
            pm.gvtrack_create("e", df, "avg")
        with pytest.raises(Exception, match="not recognized neither as a track nor as intervals. "
                                            '\\("Invalid format of interval at index 2"\\)'):
            pm.gvtrack_create("e", df[["chrom", "start", "end"]], "distance")


# R misha 5.17.0's errors for a bad DataFrame source, checked row by row as its IntervalConverter
# reads it (an intervals source wraps them); the value-based ones also refuse overlaps
_INVERTED = ("start coordinate must be lesser than end coordinate. misha uses 0-based, half-open [start, end) "
             "coordinates; if your input is 1-based (GFF/GTF/VCF), subtract 1 from start.")
_NEGATIVE = "start coordinate must be greater or equal than zero"
SOURCE_ERRORS = [
    (["chr1", "chr1"], [0, 20], [10, 20], "Interval (chr1, 20, 20): " + _INVERTED),
    (["chr1", "chr1"], [0, 30], [10, 20], "Interval (chr1, 30, 20): " + _INVERTED),
    (["chr1", "chr1"], [0, -5], [10, 20], "Interval (chr1, -5, 20): " + _NEGATIVE),
    (["chr1", "chr1"], [0, 20], [10, 1e9], "Interval (chr1, 20, 1000000000): end coordinate exceeds chromosome "
                                           "boundaries"),
    (["chr1", None], [0, 15], [10, 20], 'Chromosome "NA" does not exist'),
    (["chr1", "chrZZ"], [0, 15], [10, 20], 'Chromosome "chrZZ" does not exist'),
    (pd.Categorical(["chr1", None]), [0, 15], [10, 20], "Invalid format of interval at index 2"),
    ([1, np.nan], [0, 15], [10, 20], "Invalid format of interval at index 2"),
    # row by row: the first bad row, and in a row its coordinates, then its chromosome
    (["chrZZ", "chr1"], [5, -15], [10, 20], 'Chromosome "chrZZ" does not exist'),
    (["chr1", "chrZZ"], [-5, 15], [10, 20], "Interval (chr1, -5, 10): " + _NEGATIVE),
    (["chr1", None], [0, np.nan], [10, 20], "Invalid format of interval at index 2"),
]


@pytest.mark.parametrize(("chrom", "start", "end", "msg"), SOURCE_ERRORS)
def test_a_bad_dataframe_source_gives_rs_error(chrom, start, end, msg):
    pm.gdb_init(str(TESTDB_ROOT))
    df = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "v": [1.0, 2.0]})
    with pytest.raises(Exception) as err:
        pm.gvtrack_create("e", df, "avg")
    assert str(err.value) == msg
    with pytest.raises(Exception) as err:
        pm.gvtrack_create("e", df[["chrom", "start", "end"]], "distance")
    assert str(err.value) == ("Source of virtual track e was not recognized neither as a track nor as intervals. "
                              f'("{msg}")')


def test_overlapping_values_in_a_dataframe_source_give_rs_error():
    pm.gdb_init(str(TESTDB_ROOT))
    df = pd.DataFrame({"chrom": ["chr2", "chr1", "chr1", "chr1"], "start": [0, 100, 0, 50], "end": [10, 200, 10, 150],
                       "v": [1.0, 2, 3, 4]})
    with pytest.raises(ValueError) as err:
        pm.gvtrack_create("e", df, "avg")
    assert str(err.value) == ("Virtual track e: overlapping intervals detected. Interval at position 2 [chr1:100-200] "
                              "overlaps with previous interval [chr1:50-150]. Value-based virtual tracks do not "
                              "support overlapping intervals.")
    # an intervals source may overlap (R: distance 0, 0, 50 at iterator 100)
    pm.gvtrack_create("e", df[["chrom", "start", "end"]], "distance")
    res = pm.gextract("e", pm.gintervals("chr1", 0, 300), iterator=100)
    assert res["e"].tolist() == [0, 0, 50]
