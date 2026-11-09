"""R misha 5.16.0 argument types (test-review-2026-11-boundary.R): a numeric chrom column is
read as chromosome names, a missing strand is an error, a chain data frame takes a numeric
chromsrc and numeric strands only, and CONFIG['max_data_size'], CONFIG['max_mem_usage'] and
CONFIG['multitasking'] are read the same way by the C++ and the Python code, as R reads
gmax.data.size, gmax.mem.usage and gmultitasking."""

import resource

import numpy as np
import pandas as pd
import pytest
from _chromdb_helpers import create_test_db
from _dbpath import TESTDB_ROOT

import pymisha as pm
from pymisha import _pymisha, _shared

DF = pd.DataFrame
SCOPE = DF(dict(chrom=["chr1"], start=[0], end=[20000]))
ERRORS = (ValueError, _pymisha.error)


@pytest.fixture()
def config():
    """pm.CONFIG, restored after the test."""
    saved = dict(pm.CONFIG)
    yield pm.CONFIG
    pm.CONFIG.clear()
    pm.CONFIG.update(saved)


@pytest.fixture()
def tracks():
    """Names of tracks the test creates, removed after it."""
    names = []
    yield names
    for name in names:
        if pm.gtrack_exists(name):
            pm.gtrack_rm(name, force=True)
    _pymisha.pm_dbreload()


def _vals(scope=SCOPE):
    return pm.gextract("dense_track", scope, progress=False)["dense_track"].to_numpy()


def _lookup_bin(vals, breaks):
    """The bin (1-based) glookup puts each value in: (b1, b2], ...; force_binning puts values
    outside the breaks into the first or the last bin (R's lookup_bin)."""
    return np.clip(np.searchsorted(breaks, vals, side="left"), 1, len(breaks) - 1)


def _lookup_expected(table, vals, breaks):
    out = np.asarray(table, dtype=float)[_lookup_bin(vals, breaks) - 1]
    out[np.isnan(vals)] = np.nan
    return out


# --- an integer or logical NA is stored and looked up as NaN (regression) ---

def test_integer_or_logical_na_is_stored_and_looked_up_as_nan(tracks):
    tracks += ["review516_na_sparse", "review516_na_2d", "review516_na_lookup"]
    iv = pm.gintervals(["chr1", "chr1", "chr2"], [10, 500, 100], [20, 600, 300])
    pm.gtrack_create_sparse("review516_na_sparse", "", iv, pd.array([1, pd.NA, 3], dtype="Int64"))
    got = pm.gextract("review516_na_sparse", iv)["review516_na_sparse"].to_numpy()
    np.testing.assert_array_equal(got, [1, np.nan, 3])

    iv2 = pm.gintervals_2d(["chr1", "chr2"], 0, 1000, ["chr1", "chr2"], 2000, 3000)
    pm.gtrack_2d_create("review516_na_2d", "", iv2, pd.array([5, pd.NA], dtype="Int64"))
    np.testing.assert_array_equal(pm.gextract("review516_na_2d", iv2)["review516_na_2d"].to_numpy(), [5, np.nan])

    vals = _vals()
    breaks = [0, 0.1, 0.2, 1]
    expected = _lookup_expected([1, np.nan, 3], vals, breaks)
    got = pm.glookup(pd.array([1, pd.NA, 3], dtype="Int64"), "dense_track", breaks, intervals=SCOPE)["value"]
    np.testing.assert_array_equal(got.to_numpy(), expected)
    got = pm.glookup(pd.array([True, pd.NA, False], dtype="boolean"), "dense_track", breaks, intervals=SCOPE)["value"]
    np.testing.assert_array_equal(got.to_numpy(), _lookup_expected([1, np.nan, 0], vals, breaks))

    pm.gtrack_lookup("review516_na_lookup", "", pd.array([1, pd.NA, 3], dtype="Int64"), "dense_track", breaks)
    np.testing.assert_array_equal(pm.gextract("review516_na_lookup", SCOPE)["review516_na_lookup"].to_numpy(), expected)


# --- a numeric chrom column is read as chromosome names ---

CHR = DF(dict(chrom=["chr1", "chr2"], start=[0, 0], end=[1000, 1000]))


def _chr2d():
    return DF(dict(chrom1=["chr1", "chr2"], start1=[0, 0], end1=[500000, 300000],
                   chrom2=["chr1", "chr2"], start2=[0, 0], end2=[500000, 300000]))


@pytest.mark.parametrize("num", [[1, 2], [1.0, 2.0], pd.array([1, 2], dtype="Int64"),
                                 np.array([1, 2], dtype=np.int32), pd.Categorical([1.0, 2.0])])
def test_numeric_chrom_column_is_read_as_chromosome_names(num):
    expected = pm.gextract("dense_track", CHR)
    pd.testing.assert_frame_equal(pm.gextract("dense_track", CHR.assign(chrom=num)), expected)


def test_numeric_chrom1_chrom2_are_read_as_chromosome_names():
    expected = pm.gextract("rects_track", _chr2d())
    assert len(expected) > 0
    got = pm.gextract("rects_track", _chr2d().assign(chrom1=[1, 2], chrom2=[1.0, 2.0]))
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize("bad,msg", [
    (DF(dict(chrom=[["chr1"]], start=[0], end=[100])), "Invalid format of interval"),
    (DF(dict(chrom=pd.array([pd.NA], dtype="Int64"), start=[0], end=[100])), "Invalid format of interval at index 1"),
    (DF(dict(chrom=[1.5], start=[0], end=[100])), "Invalid format of interval at index 1"),
    (DF(dict(chrom=["chr1", np.nan], start=[0, 0], end=[100, 100])), "Invalid format of interval at index 2"),
    (DF(dict(chrom=[None], start=[0], end=[100])), "Invalid format of interval at index 1"),
    (DF(dict(chrom=[True], start=[0], end=[100])), "column chrom must be"),
    (DF(dict(chrom=pd.Categorical(["chr1", None]), start=[0, 0], end=[100, 100])),
     "Invalid format of interval at index 2"),
    (["chr1", "chr2"], "Invalid format of interval"),
])
def test_other_chrom_types_are_rejected(bad, msg):
    with pytest.raises(ERRORS, match=msg):
        pm.gextract("dense_track", bad)


def test_bad_2d_chrom_and_too_few_2d_columns_are_rejected():
    iv2 = pm.gintervals_2d("chr1", 0, 100, "chr1", 0, 100)
    iv2["chrom2"] = [["chr1"]]
    with pytest.raises(ERRORS, match="Invalid format of intervals"):
        pm.gextract("rects_track", iv2)
    with pytest.raises(ERRORS, match="Invalid format of intervals: column names"):
        pm.gextract("rects_track", DF(dict(chrom1=["chr1"], start1=[0], end1=[100], chrom2=["chr1"], start2=[0])))


def test_fractional_or_bool_chrom_is_not_truncated_in_cpp():
    for bad in ([1.5], [True], np.array([1.0, 2.5])):
        df = DF(dict(chrom=bad, start=0, end=10))
        with pytest.raises(ERRORS, match="Invalid format of interval"):
            pm.gseq_extract(df)
        with pytest.raises(ERRORS, match="Invalid format of interval"):
            pm.gextract("dense_track", df)
    assert _pymisha.pm_normalize_chroms([np.int64(1), 2.0, "X"]) == ["chr1", "chr2", "chrX"]
    for bad in (1.5, True, float("nan"), None):
        with pytest.raises(_pymisha.error, match="Invalid format of interval"):
            _pymisha.pm_normalize_chroms([bad])


def test_numeric_chrom_in_the_python_paths(tracks):
    num = DF(dict(chrom=[1.0, 2.0], start=[0, 0], end=[1000, 1000]))
    named = num.assign(chrom=["chr1", "chr2"])
    other = DF(dict(chrom=["chr1"], start=[2000], end=[2100]))
    assert pm.gintervals_neighbors(num, other)["dist"].tolist() == \
        pm.gintervals_neighbors(named, other)["dist"].tolist()
    assert len(pm.gintervals_annotate(num, pm.gintervals("chr1", 0, 500))) == 2
    tracks.append("review516_num_sparse")
    pm.gtrack_create_sparse("review516_num_sparse", "", num, [1.0, 2.0])
    got = pm.gextract("review516_num_sparse", pm.gintervals_all())
    assert got["chrom"].astype(str).tolist() == ["chr1", "chr2"]
    try:
        pm.gintervals_save(num, "review516_num_set")
        assert pm.gintervals_load("review516_num_set")["chrom"].astype(str).tolist() == ["chr1", "chr2"]
    finally:
        if pm.gintervals_exists("review516_num_set"):
            pm.gintervals_rm("review516_num_set", force=True)
    domain = pm.gintervals("chr1", [0, 200000], [150000, 350000])
    src = pm.gextract("sparse_track", pm.gintervals("chr1"))
    pd.testing.assert_frame_equal(
        pd.DataFrame(pm.gcis_decay("rects_track", [0, 1000, 2000], src.assign(chrom=1.0), domain.assign(chrom=1))),
        pd.DataFrame(pm.gcis_decay("rects_track", [0, 1000, 2000], src, domain)))


def test_minus_zero_is_chromosome_0(tmp_path):
    db = create_test_db(tmp_path / "db0", [("chr0", 1000), ("chr1", 1000)])
    try:
        pm.gdb_init(db)
        for ch in ([0.0], [-0.0], [0.0, -0.0], [-0.0, 0.0], [np.float32(-0.0)]):
            df = DF(dict(chrom=ch, start=[0, 100][:len(ch)], end=[10, 110][:len(ch)]))
            res = pm.gintervals_canonic(df)
            assert res["chrom"].astype(str).tolist() == ["chr0"] * len(ch), ch
            assert pm.gseq_extract(df) == ["AAAAAAAAAA"] * len(ch)
    finally:
        pm.gdb_init(str(TESTDB_ROOT))


# --- a chain data frame with a numeric chromsrc; character strands are rejected ---

CHAIN = DF(dict(chrom=["chr1"], start=[1000], end=[2000], strand=[0], chromsrc=["1"], startsrc=[0],
                endsrc=[1000], strandsrc=[0], chain_id=[1], score=[1.0]))
SRC = DF(dict(chrom=["1"], start=[100], end=[200]))


def test_chain_with_numeric_chromsrc_is_read():
    expected = pm.gintervals_liftover(SRC, CHAIN)
    assert expected["start"].tolist() == [1100]
    for num in ([1], [1.0], pd.array([1], dtype="Int64")):
        pd.testing.assert_frame_equal(pm.gintervals_liftover(SRC, CHAIN.assign(chromsrc=num)), expected)
    # a numeric source chrom, and a numeric target chrom
    for num in ([1], [1.0]):
        pd.testing.assert_frame_equal(pm.gintervals_liftover(SRC.assign(chrom=num), CHAIN), expected)
        pd.testing.assert_frame_equal(pm.gintervals_liftover(SRC, CHAIN.assign(chrom=num)), expected)


def test_chain_with_character_or_missing_strands_is_rejected():
    for strand in ("+", "-"):
        with pytest.raises(ValueError, match="columns strand and strandsrc must be numeric"):
            pm.gintervals_liftover(SRC, CHAIN.assign(strand=[strand], strandsrc=[strand]))
    with pytest.raises(ValueError, match="Invalid format of interval at index 1"):
        pm.gintervals_liftover(SRC, CHAIN.assign(strand=[np.nan]))
    for bad in (pd.array([pd.NA], dtype="Int64"), [None], [np.nan], [1.5]):
        with pytest.raises(ValueError, match="Invalid format of interval at index 1"):
            pm.gintervals_liftover(SRC, CHAIN.assign(chromsrc=bad))


# --- a missing strand is not strand 0 ---

@pytest.mark.parametrize("strand,msg", [
    # a missing number is a bad interval, as R says of NA_real_ (5.16.1)
    (pd.array([1, pd.NA, -1], dtype="Int64"), r"Invalid strand value \(NA\) at index 2"),  # R's NA_integer_
    ([1, np.nan, -1], "Invalid format of interval at index 2"),
    ([1, None, -1], "Invalid format of interval at index 2"),
    (["+", None, "-"], r"Invalid strand value \(NA\) at index 2"),
    (pd.Categorical(["+", None, "-"]), r"Invalid strand value \(NA\) at index 2")])
def test_missing_strand_is_an_error(strand, msg):
    df = DF(dict(chrom="chr1", start=0, end=8, strand=strand))
    with pytest.raises(_pymisha.error, match=msg):
        pm.gseq_extract(df)
    with pytest.raises(_pymisha.error, match=msg):
        pm.gextract("dense_track", df)


def test_strands_without_missing_values_still_work():
    df = DF(dict(chrom="chr1", start=0, end=8, strand=[1, 1, -1]))
    seqs = pm.gseq_extract(df)
    assert seqs[0] == seqs[1] and seqs[2] != seqs[0]
    assert pm.gseq_extract(df.assign(strand=["+", "+", "-"])) == seqs
    assert len(pm.gextract("dense_track", df)) == 3


# --- CONFIG['max_data_size'] (gmax.data.size) ---

def test_max_data_size_inf_is_no_limit_below_1_counts_as_1_nan_or_negative_is_an_error(config):
    vals = _vals()
    for no_limit in (float("inf"), 2**64, None, 2**53):
        config["max_data_size"] = no_limit
        assert np.asarray(pm.gquantiles("dense_track", 1, SCOPE))[0] == pytest.approx(np.nanmax(vals), abs=1e-6)
        assert len(pm.gextract("dense_track", SCOPE)) == 400
    for limit in (400, 400.0, np.int64(400), np.float32(400)):
        config["max_data_size"] = limit
        assert len(pm.gextract("dense_track", SCOPE)) == 400
    for limit in (399, 399.9, np.int64(399)):
        config["max_data_size"] = limit
        with pytest.raises(_pymisha.error, match=r"maximum allowed \(399 rows"):
            pm.gextract("dense_track", SCOPE)
    for small in (0, 0.5):
        config["max_data_size"] = small
        with pytest.warns(RuntimeWarning, match="approximate"):
            # a sampling buffer of one value, not a crash
            assert len(pm.gquantiles("dense_track", [0.1, 0.5], SCOPE)) == 2
        with pytest.raises(_pymisha.error, match=r"maximum allowed \(1 rows"):
            pm.gextract("dense_track", SCOPE)
    for bad, msg in ((float("nan"), "must be a number, not NA"), (pd.NA, "must be a number, not NA"),
                     (-1, r"must not be negative \(got -1\)"), (float("-inf"), r"must not be negative \(got -inf\)"),
                     ("1e7", "must be a number"), (True, "must be a number")):
        config["max_data_size"] = bad
        with pytest.raises(ERRORS, match=r"CONFIG\['max_data_size'\] " + msg):
            pm.gquantiles("dense_track", [0.1, 0.5], SCOPE)
    # an invalid value leaves no Python error behind: the next call works
    config["max_data_size"] = 2**64
    assert len(pm.gextract("dense_track", SCOPE)) == 400


def test_max_data_size_absent_is_the_default(config):
    config.pop("max_data_size")
    assert _shared._config_limit("max_data_size") == 10_000_000
    assert len(pm.gextract("dense_track", SCOPE)) == 400


def test_max_data_size_inf_for_gbins_quantiles_2d_import_and_cartesian_grid(tracks, config, tmp_path):
    vals = _vals()
    breaks = [0, 0.1, 0.2, 1]
    b = np.searchsorted(breaks, vals, side="left")   # cut(): (0, 0.1], (0.1, 0.2], (0.2, 1]
    expected = [np.nanmax(vals[b == k]) for k in (1, 2, 3)]
    config["max_data_size"] = float("inf")
    res = pm.gbins_quantiles("dense_track", breaks, expr="dense_track", percentiles=1, intervals=SCOPE)
    np.testing.assert_allclose(np.asarray(res)[:, 0], expected, atol=1e-6)

    tracks.append("review516_2d_inf")
    rng = np.random.default_rng(60427)
    n = 5000
    s1 = rng.choice(100000, n, replace=False)
    s2 = rng.choice(100000, n, replace=False)
    pts = DF(dict(chrom1="chr1", start1=s1, end1=s1 + 1, chrom2="chr2", start2=s2, end2=s2 + 1,
                  value=np.arange(1, n + 1)))
    f = tmp_path / "pts.tsv"
    pts.to_csv(f, sep="\t", index=False)
    pm.gtrack_2d_import("review516_2d_inf", "", str(f))
    got = pm.gextract("review516_2d_inf", pm.gintervals_2d("chr1", 0, 100000, "chr2", 0, 100000))
    assert sorted(got["review516_2d_inf"].tolist()) == list(map(float, range(1, n + 1)))

    # a limit above 2**31
    config["max_data_size"] = 2**32
    itr = pm.giterator_cartesian_grid(pm.gintervals("chr1", [1000, 50000], [2000, 51000]), [-100, 100])
    assert len(pm.giterator_intervals(intervals=pm.gintervals_2d("chr1"), iterator=itr)) == 4


def test_max_data_size_inf_for_the_python_sampler(config):
    expected = pm.gquantiles("dense_track", [0.1, 0.5], SCOPE)
    pm.gvtrack_create("review516_vavg", "dense_track", "avg")
    try:
        for no_limit in (float("inf"), None):
            config["max_data_size"] = no_limit
            np.testing.assert_allclose(pm.gquantiles("review516_vavg", [0.1, 0.5], SCOPE), expected)
    finally:
        pm.gvtrack_rm("review516_vavg")


# --- multitasked calls with max_data_size of no limit give the serial results ---

S2 = DF(dict(chrom=["chr1", "chr2"], start=[0, 0], end=[500000, 300000]))


def _multitask(config):
    config.update(multitasking=True, max_processes=4, min_processes=2, min_scope4process=1, min_intervs4process=1)


def _child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def test_multitasked_calls_with_no_limit_give_the_serial_results(config):
    iv = pm.giterator_intervals(intervals=S2, iterator=50000)
    p = [0.1, 0.5, 0.9]

    def run():
        return [pm.gquantiles("dense_track", p, iv),
                pm.gintervals_quantiles("dense_track", p, iv),
                pm.gintervals_mapply(np.nanmax, "dense_track", intervals=iv),
                pm.glookup([1, 2], "dense_track", [0, 0.15, 1], intervals=iv),
                pm.gcor("dense_track", "sparse_track", intervals=S2, iterator=50, method="spearman")]

    config["multitasking"] = False
    expected = run()
    x = pm.gextract(["dense_track", "sparse_track"], S2, iterator=50)
    assert float(np.ravel(expected[4])[0]) == pytest.approx(
        x[["dense_track", "sparse_track"]].dropna().corr(method="spearman").iloc[0, 1], abs=1e-6)
    _multitask(config)
    for limit in (float("inf"), None, "absent"):
        if limit == "absent":
            config.pop("max_data_size")
        else:
            config["max_data_size"] = limit
        t0 = _child_cpu()
        got = run()
        assert _child_cpu() > t0   # the kids ran
        np.testing.assert_array_equal(got[0], expected[0])
        for a, b in zip(got[1:4], expected[1:4], strict=True):
            pd.testing.assert_frame_equal(a, b)
        np.testing.assert_allclose(got[4], expected[4], atol=1e-6)


# --- CONFIG['max_mem_usage'] (gmax.mem.usage) ---

def test_max_mem_usage_of_0_works_nan_or_negative_is_an_error(tracks, config, tmp_path):
    p = [0.1, 0.5, 0.9]
    v = _vals(S2)
    expected = np.quantile(v[~np.isnan(v)], p, method="inverted_cdf")
    _multitask(config)
    config.update(max_data_size=float("inf"), max_mem_usage=0)
    np.testing.assert_allclose(pm.gquantiles("dense_track", p, S2), expected, atol=1e-6)
    for bad, msg in ((float("nan"), "must be a number, not NA"), (pd.NA, "must be a number, not NA"),
                     (-1, r"must not be negative \(got -1\)"), (-5.0, r"must not be negative \(got -5\)")):
        config["max_mem_usage"] = bad
        with pytest.raises(ERRORS, match=r"CONFIG\['max_mem_usage'\] " + msg):
            pm.gquantiles("dense_track", p, S2)

    f = tmp_path / "pts.tsv"
    DF(dict(chrom1="chr1", start1=[10, 20], end1=[11, 21], chrom2="chr2", start2=[30, 40], end2=[31, 41],
            value=[1.0, 2.0])).to_csv(f, sep="\t", index=False)
    for limit in (0, 10.0, float("inf"), None, np.int64(5)):
        config["max_mem_usage"] = limit
        name = f"review516_mem_{len(tracks)}"
        tracks.append(name)
        pm.gtrack_2d_import(name, "", str(f))
        assert len(pm.gextract(name, pm.gintervals_2d("chr1", 0, 1000, "chr2", 0, 1000))) == 2
    config["max_mem_usage"] = float("nan")
    tracks.append("review516_mem_bad")
    with pytest.raises(ERRORS, match=r"CONFIG\['max_mem_usage'\] must be a number, not NA"):
        pm.gtrack_2d_import("review516_mem_bad", "", str(f))
    assert _shared._config_limit("max_mem_usage", {}) == _shared._NO_LIMIT


# --- CONFIG['multitasking'] (gmultitasking) ---

def test_multitasking_must_be_true_or_false(config):
    for ok, want in ((True, True), (False, False), (np.True_, True), (np.False_, False), (None, False),
                     (1, True), (0, False), (np.int64(1), True)):
        config["multitasking"] = ok
        assert _shared._config_multitasking() is want
        assert len(pm.gextract("dense_track", SCOPE)) == 400
    config.pop("multitasking")
    assert _shared._config_multitasking() is False
    for bad in (float("nan"), pd.NA, 2, -1, 1.0, "yes"):
        config["multitasking"] = bad
        with pytest.raises(ERRORS, match=r"CONFIG\['multitasking'\] must be True or False"):
            pm.gsummary("dense_track", SCOPE)


def test_out_of_range_integer_config_leaves_no_python_error(config):
    for key in ("min_processes", "max_processes", "min_scope4process", "min_intervs4process", "eval_buf_size"):
        old = config[key]
        config[key] = 2**70
        with pytest.raises(_pymisha.error, match=rf"CONFIG\['{key}'\] is out of range"):
            pm.gextract("dense_track", SCOPE)
        config[key] = old
        assert len(pm.gextract("dense_track", SCOPE)) == 400
