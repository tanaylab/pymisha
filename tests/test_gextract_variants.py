"""Tests for gextract_variants, ported from R misha's
tests/testthat/test-gextract-variants.R and helper-variants.R.

R's tests on the shared test database use chromosome 1 as ``1`` / ``chr1``;
pymisha's test database names it ``1``, and R's ``test.fixedbin`` dense track
is ``dense_track`` there. The tests that build their own one-chromosome
database from a FASTA keep R's ``chr1``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from _potts_oracle import potts_ref_model

import pymisha as pm
from pymisha.extract import _can_vtracks_use_cpp
from pymisha.variants import _gextract_variants_impl, _gvariants_eval, _gvariants_reach

VARIANT_BASES = ("A", "C", "G", "T")
VARIANT_VTRACKS = ("v_pwm", "v_potts", "v_kmer", "v_masked", "v_ed")


@pytest.fixture(autouse=True)
def _clean_vtracks():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def variant_test_pssm():
    return np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.7],
        [0.4, 0.1, 0.4, 0.1],
    ])


def define_variant_vtracks():
    """One vtrack of each sequence family, with shifts. Vtracks belong to the
    working database, so call this again after every root change."""
    pssm = variant_test_pssm()
    pm.gvtrack_create("v_pwm", None, "pwm.max", pssm=pssm, bidirect=True, extend=True, prior=0.01)
    pm.gvtrack_iterator("v_pwm", sshift=-5, eshift=3)
    m = potts_ref_model(6, npair_mode="full", seed=3)
    pm.gvtrack_create("v_potts", None, "potts.max", params=dict(m, bidirect=False, strand=-1))
    pm.gvtrack_iterator("v_potts", sshift=4, eshift=4)
    pm.gvtrack_create("v_kmer", None, "kmer.count", kmer="ACG", extend=True, strand=0)
    pm.gvtrack_iterator("v_kmer", sshift=-2, eshift=2)
    pm.gvtrack_create("v_masked", None, "masked.count")
    pm.gvtrack_iterator("v_masked", sshift=-3, eshift=3)
    pm.gvtrack_create("v_ed", None, "pwm.edit_distance", pssm=pssm, score_thresh=-2, max_indels=1)


def variant_test_seq(n=400, seed=7):
    """A random chromosome with some soft-masked (lowercase) bases."""
    rng = np.random.default_rng(seed)
    letters = list(VARIANT_BASES) + [b.lower() for b in VARIANT_BASES]
    return "".join(rng.choice(letters, size=n, p=[0.2] * 4 + [0.05] * 4))


@pytest.fixture
def make_seq_db(tmp_path):
    """make_seq_db(seq): a one-chromosome database ("chr1") built from seq;
    returns its root."""
    count = [0]

    def make(seq):
        count[0] += 1
        d = tmp_path / f"db{count[0]}"
        d.mkdir()
        fa = d / "g.fa"
        fa.write_text(f">chr1\n{seq}\n")
        pm.gdb_create(str(d / "db"), str(fa))
        return str(d / "db")

    return make


def substitute_bases(seq, pos, base):
    """seq with the 0-based positions pos replaced by base, each keeping the
    case of the base it replaces."""
    chars = list(seq)
    for p, b in zip(pos, base, strict=True):
        old = chars[p]
        chars[p] = b.lower() if old == old.lower() else b.upper()
    return "".join(chars)


def values_on_db(db, vtrack, bins, define=define_variant_vtracks):
    """The value of vtrack (or an expression) on 1 bp bins of chr1 in database db."""
    bins = np.asarray(bins)
    pm.gdb_init(db)
    define()
    lo, hi = int(bins.min()), int(bins.max())
    vals = pm.gextract(vtrack, pm.gintervals("chr1", lo, hi + 1), iterator=1, colnames=["x"])["x"].to_numpy()
    return vals[bins - lo]


def base_code(b):
    return VARIANT_BASES.index(b)


def test_reach_a_bin_of_each_sequence_vtrack_reads_lo_to_hi():
    define_variant_vtracks()
    pssm = variant_test_pssm()
    pm.gvtrack_create("r_noext", None, "pwm.max", pssm=pssm, bidirect=True, extend=False, prior=0.01)
    pm.gvtrack_create("r_kmer_fwd", None, "kmer.count", kmer="ACG", extend=True, strand=1)
    pm.gvtrack_create("r_kmer_noext", None, "kmer.count", kmer="ACG", extend=False, strand=0)
    assert _gvariants_reach("v_pwm").tolist() == [[-5, 3 + 3]]
    assert _gvariants_reach("r_noext").tolist() == [[0, 0]]
    assert _gvariants_reach("v_potts").tolist() == [[4, 4 + 5]]
    assert _gvariants_reach("v_kmer").tolist() == [[-2, 2 + 2]]
    assert _gvariants_reach("r_kmer_fwd").tolist() == [[0, 2]]
    assert _gvariants_reach("r_kmer_noext").tolist() == [[0, 0]]
    assert _gvariants_reach("v_masked").tolist() == [[-3, 3]]
    assert _gvariants_reach("v_ed").tolist() == [[0, 4 + 1 - 1]]
    assert len(_gvariants_reach("v_pwm + 2 * v_potts + dense_track")) == 2


def test_reach_an_expression_with_no_sequence_vtrack_is_an_error():
    with pytest.raises(ValueError, match="no sequence virtual track"):
        _gvariants_reach("dense_track * 2")


def test_engine_a_substituted_row_equals_the_same_bin_on_a_really_edited_genome(make_seq_db):
    seq = variant_test_seq()
    db = make_seq_db(seq)
    subs = [(p, b) for p, b in ((100, "A"), (101, "C"), (150, "G"), (151, "T"), (152, "A"), (200, "C"))
            if seq[p].upper() != b]
    edited = [make_seq_db(substitute_bases(seq, [p], [b])) for p, b in subs]
    bins = np.arange(90, 211)
    # Bin-major: consecutive rows share a bin and differ in the substitution,
    # which is where a cache that ignores the substitution would leak.
    row_bin = np.repeat(bins, len(subs))
    row_sub = np.tile(np.arange(len(subs)), len(bins))
    for vt in VARIANT_VTRACKS:
        pm.gdb_init(db)
        define_variant_vtracks()
        got = _gvariants_eval(
            vt, ["chr1"] * len(row_bin), row_bin, row_bin + 1,
            [subs[s][0] for s in row_sub], [base_code(subs[s][1]) for s in row_sub],
        )
        for s in range(len(subs)):
            np.testing.assert_allclose(
                got[row_sub == s], values_on_db(edited[s], vt, bins),
                rtol=1e-8, err_msg=f"{vt}, substitution {s + 1}",
            )


def test_engine_a_row_with_no_substitution_after_substituted_rows_reads_the_reference(make_seq_db):
    seq = variant_test_seq(seed=8)
    db = make_seq_db(seq)
    b = 120
    alt = next(x for x in VARIANT_BASES if x != seq[b].upper())
    for vt in VARIANT_VTRACKS:
        want = values_on_db(db, vt, [b])[0]
        got = _gvariants_eval(vt, ["chr1"] * 3, [b] * 3, [b + 1] * 3, [b] * 3, [base_code(alt), -1, -1])
        np.testing.assert_allclose(got[1:], [want, want], rtol=1e-8, err_msg=vt)


def test_engine_unsorted_rows_are_an_error(make_seq_db):
    db = make_seq_db(variant_test_seq())
    pm.gdb_init(db)
    define_variant_vtracks()
    with pytest.raises(pm.error, match="sorted"):
        _gvariants_eval("v_pwm", ["chr1", "chr1"], [50, 40], [51, 41], [50, 40], [0, 0])


def test_saturation_every_non_n_base_in_reach_three_alts_each():
    define_variant_vtracks()
    res = pm.gextract_variants("v_pwm", pm.gintervals(1, 1000, 1001), iterator=1, colnames="s")
    assert list(res.columns) == ["chrom", "start", "end", "pos", "ref", "alt", "s.ref", "s.alt", "intervalID"]
    genome = pm.gseq_extract(pm.gintervals(1, 995, 1007))[0].upper()
    acgt = [p for p in range(995, 1007) if genome[p - 995] in VARIANT_BASES]
    assert set(res["pos"]) == set(acgt)
    assert len(res) == 3 * len(acgt)
    assert list(res["ref"]) == [genome[p - 995] for p in res["pos"]]
    assert (res["alt"] != res["ref"]).all()
    want = pm.gextract("v_pwm", pm.gintervals(1, 1000, 1001), iterator=1)["v_pwm"]
    assert res["s.ref"].unique().tolist() == want.tolist()


def test_values_ref_is_the_reference_and_alt_the_really_edited_genome(make_seq_db):
    seq = variant_test_seq(seed=9)
    db = make_seq_db(seq)
    pm.gdb_init(db)
    define_variant_vtracks()
    res = pm.gextract_variants("v_pwm + v_potts", pm.gintervals("chr1", 100, 104), iterator=1, colnames="s")
    rng = np.random.default_rng(1)
    for r in rng.choice(len(res), 4, replace=False):
        b = int(res["start"].iloc[r])
        want_ref = values_on_db(db, "v_pwm", [b])[0] + values_on_db(db, "v_potts", [b])[0]
        assert res["s.ref"].iloc[r] == pytest.approx(want_ref, rel=1e-8)
        edb = make_seq_db(substitute_bases(seq, [int(res["pos"].iloc[r])], [res["alt"].iloc[r]]))
        want_alt = values_on_db(edb, "v_pwm", [b])[0] + values_on_db(edb, "v_potts", [b])[0]
        assert res["s.alt"].iloc[r] == pytest.approx(want_alt, rel=1e-8), r


def test_reach_is_complete_changing_every_base_outside_it_leaves_the_value_unchanged(make_seq_db):
    seq = variant_test_seq(seed=10)
    db = make_seq_db(seq)
    b = 200
    shift = {"A": "C", "C": "G", "G": "T", "T": "A"}

    def edit_outside(res):
        outside = sorted(set(range(170, 241)) - set(res["pos"]))
        return make_seq_db(substitute_bases(seq, outside, [shift[seq[p].upper()] for p in outside]))

    for vt in VARIANT_VTRACKS:
        pm.gdb_init(db)
        define_variant_vtracks()
        res = pm.gextract_variants(vt, pm.gintervals("chr1", b, b + 1), iterator=1)
        edb = edit_outside(res)
        np.testing.assert_allclose(values_on_db(edb, vt, [b]), values_on_db(db, vt, [b]), err_msg=vt)

    # misha takes its batched k-mer path only with 32+ k-mer vtracks on bins of
    # 100 bp or more, so R checks one such expression too. pymisha has no
    # batched path; checked all the same.
    pads = [f"v_pad{i}" for i in range(1, 32)]

    def define_vtracks():
        define_variant_vtracks()
        for p in pads:
            pm.gvtrack_create(p, None, "kmer.count", kmer="CGT", extend=True, strand=0)

    expr = " + ".join(["v_kmer", "0 * v_pwm", "0 * v_potts", "0 * v_masked", *(f"0 * {p}" for p in pads)])
    bin_iv = pm.gintervals("chr1", b, b + 100)

    def value_on(root):
        pm.gdb_init(root)
        define_vtracks()
        return pm.gextract(expr, bin_iv, iterator=100, colnames=["x"])["x"].to_numpy()

    pm.gdb_init(db)
    define_vtracks()
    res = pm.gextract_variants(expr, bin_iv, iterator=100)
    outside = sorted(set(range(150, 351)) - set(res["pos"]))
    edb = make_seq_db(substitute_bases(seq, outside, [shift[seq[p].upper()] for p in outside]))
    np.testing.assert_allclose(value_on(edb), value_on(db), err_msg="batched expression")


def test_chromosome_ends_clip_the_reach(make_seq_db):
    db = make_seq_db(variant_test_seq(seed=11))
    pm.gdb_init(db)
    define_variant_vtracks()
    res = pm.gextract_variants("v_pwm", pm.gintervals(["chr1", "chr1"], [0, 398], [2, 400]), iterator=1)
    assert ((res["pos"] >= 0) & (res["pos"] < 400)).all()
    assert set(res["pos"]) == set(range(8)) | set(range(393, 400))


def test_n_bases_are_not_substituted_and_a_listed_variant_on_an_n_is_a_ref_mismatch(make_seq_db):
    seq = variant_test_seq(seed=12)
    seq = seq[:100] + "NNN" + seq[103:]
    db = make_seq_db(seq)
    pm.gdb_init(db)
    define_variant_vtracks()
    res = pm.gextract_variants("v_pwm", pm.gintervals("chr1", 100, 101), iterator=1)
    assert not res["pos"].isin([100, 101, 102]).any()
    v = pd.DataFrame({"chrom": ["chr1"], "start": [100], "end": [101], "ref": ["A"], "alt": ["C"]})
    with pytest.warns(UserWarning, match="1 of 1 variants"):
        res2 = pm.gextract_variants("v_pwm", pm.gintervals("chr1", 100, 101), variants=v, iterator=1)
    assert len(res2) == 1
    assert res2["v_pwm.alt"].isna().all()


def test_a_listed_variant_pairs_with_every_bin_whose_reach_contains_it():
    define_variant_vtracks()
    g = pm.gseq_extract(pm.gintervals(1, 1000, 1030))[0].upper()

    def alt_at(i):
        return next(x for x in VARIANT_BASES if x != g[i - 1000])

    v = pd.DataFrame({
        "chrom": ["1"] * 3,
        "start": [1005, 1020, 1029],
        "end": [1006, 1021, 1030],
        "ref": [g[5], g[20], "C" if g[29] == "A" else "A"],
        "alt": [alt_at(1005), alt_at(1020), "G"],
    })
    v = v[v["ref"] != v["alt"]]
    with pytest.warns(UserWarning, match="1 of 3 variants"):
        res = pm.gextract_variants("v_pwm", pm.gintervals(1, 1000, 1010), variants=v, iterator=1)
    # v_pwm reads [s - 5, s + 7): 1005 is read by bins 999..1010 (clipped to
    # 1000..1009), 1020 by none; the third variant has a wrong ref: counted in
    # the warning although no bin reads it.
    assert set(res.loc[res["pos"] == 1005, "start"]) == set(range(1000, 1010))
    assert not res["pos"].isin([1020, 1029]).any()


def test_variants_are_validated():
    define_variant_vtracks()
    iv = pm.gintervals(1, 1000, 1001)

    def bad(**kw):
        return pd.DataFrame({"chrom": ["1"], "start": [1000], "end": [1001], **{k: [x] for k, x in kw.items()}})

    with pytest.raises(ValueError, match="single-base"):
        pm.gextract_variants("v_pwm", iv, variants=bad(ref="A", alt="AC"))
    with pytest.raises(ValueError, match="single-base"):
        pm.gextract_variants("v_pwm", iv, variants=bad(ref="A", alt="C,G"))
    with pytest.raises(ValueError, match="same"):
        pm.gextract_variants("v_pwm", iv, variants=bad(ref="A", alt="A"))
    with pytest.raises(ValueError, match="columns"):
        pm.gextract_variants("v_pwm", iv, variants=pd.DataFrame({"chrom": ["1"], "start": [1000]}))


def test_score_thresh_keeps_exactly_the_rows_that_cross_it():
    define_variant_vtracks()
    iv = pm.gintervals(1, 2000, 2050)
    all_rows = pm.gextract_variants("v_pwm", iv, iterator=1, colnames="s")
    ref, alt = all_rows["s.ref"].to_numpy(), all_rows["s.alt"].to_numpy()
    assert not np.isnan(ref).any() and not np.isnan(alt).any()
    # halfway across the largest change, so at least one row crosses
    i = int(np.argmax(np.abs(alt - ref)))
    threshold = (ref[i] + alt[i]) / 2
    crossed = all_rows[(ref >= threshold) != (alt >= threshold)].reset_index(drop=True)
    got = pm.gextract_variants("v_pwm", iv, iterator=1, colnames="s", score_thresh=threshold)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), crossed)


def test_a_non_sequence_track_contributes_the_same_value_to_ref_and_alt():
    define_variant_vtracks()
    iv = pm.gintervals(1, 3000, 3010)
    a = pm.gextract_variants("v_pwm", iv, iterator=1, colnames="s")
    b = pm.gextract_variants("v_pwm + dense_track", iv, iterator=1, colnames="s")
    keep = ~b["s.ref"].isna().to_numpy()
    assert keep.sum() > 0
    np.testing.assert_allclose(
        (b["s.alt"] - b["s.ref"]).to_numpy()[keep], (a["s.alt"] - a["s.ref"]).to_numpy()[keep],
        rtol=1e-6, atol=1e-6,
    )


def test_interval_id_matches_gextract_and_chunking_does_not_change_the_result():
    define_variant_vtracks()
    iv = pm.gintervals([1, 1, 1], [4000, 4003, 4100], [4006, 4010, 4103])
    ref = pm.gextract("v_pwm", iv, iterator=1)
    res = _gextract_variants_impl("v_pwm", iv, None, 1, None, "s", chunk=3)
    one = _gextract_variants_impl("v_pwm", iv, None, 1, None, "s", chunk=1_000_000)
    pd.testing.assert_frame_equal(res, one)
    key = set(zip(ref["start"], ref["intervalID"], strict=True))
    assert set(zip(res["start"], res["intervalID"], strict=True)) <= key


def test_no_rows_gives_none_and_an_expression_with_no_sequence_vtrack_is_an_error():
    define_variant_vtracks()
    v = pd.DataFrame({"chrom": ["1"], "start": [90000], "end": [90001], "ref": ["A"], "alt": ["C"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pm.gextract_variants("v_pwm", pm.gintervals(1, 1000, 1001), variants=v, iterator=1) is None
    with pytest.raises(ValueError, match="no sequence virtual track"):
        pm.gextract_variants("dense_track", pm.gintervals(1, 1000, 1001), iterator=1)


def test_intervals_on_two_chromosomes_sort_the_way_the_engine_needs():
    define_variant_vtracks()
    res = pm.gextract_variants("v_pwm", pm.gintervals([1, 2], [1000, 1000], [1003, 1003]), iterator=1)
    assert set(res["chrom"].astype(str)) == {"chr1", "chr2"}


def test_a_reach_with_zero_span_reads_nothing_and_gives_none():
    define_variant_vtracks()
    pm.gvtrack_create("v_empty", None, "pwm.max", pssm=variant_test_pssm(), extend=False)
    pm.gvtrack_iterator("v_empty", sshift=0, eshift=-1)
    assert pm.gextract_variants("v_empty", pm.gintervals(1, 1000, 1003), iterator=1) is None


# Not in R's suite. pymisha scores a vtrack with a filter, or with a source
# other than the sequence or a track, in Python (vtracks.py, pm_vtrack_compute)
# rather than in the C++ scanner, so the substitutions have to reach that path
# too. The filter cuts the reads of the bins below into several unmasked parts.
PY_PATH_VTRACKS = ("f_pwm", "f_lse", "f_pos", "f_cnt", "f_kmer", "f_masked", "f_potts", "f_ppos")


def define_python_path_vtracks():
    """One filtered vtrack per way vtracks.py combines a filter's parts, plus a
    distance and a DataFrame-source vtrack. Call again after every root change."""
    pssm = variant_test_pssm()
    filt = pm.gintervals(["chr1"] * 3, [118, 150, 185], [121, 152, 186])

    def make(name, func, sshift, eshift, **kw):
        pm.gvtrack_create(name, None, func, filter=filt, **kw)
        pm.gvtrack_iterator(name, sshift=sshift, eshift=eshift)

    pwm = {"pssm": pssm, "bidirect": True, "prior": 0.01}
    make("f_pwm", "pwm.max", -5, 3, **pwm)
    make("f_lse", "pwm", -5, 3, **pwm)
    make("f_pos", "pwm.max.pos", -5, 3, **pwm)
    make("f_cnt", "pwm.count", -5, 3, score_thresh=-3.5, **pwm)
    make("f_kmer", "kmer.count", -4, 4, kmer="ACG", extend=True, strand=0)
    make("f_masked", "masked.count", -3, 3)
    m = potts_ref_model(6, npair_mode="full", seed=3)
    make("f_potts", "potts.max", 4, 4, params=dict(m, bidirect=False, strand=-1))
    make("f_ppos", "potts.max.pos", 0, 0, params=dict(m, bidirect=True))
    pm.gvtrack_create("v_dist", pm.gintervals("chr1", 300, 301), func="distance")
    val = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [80, 140], "end": [130, 220], "value": [1.5, -2.0]})
    pm.gvtrack_create("v_val", val, func="avg")


def test_engine_python_path_vtracks_equal_the_same_bin_on_a_really_edited_genome(make_seq_db):
    seq = variant_test_seq(seed=13)
    db = make_seq_db(seq)
    # Every fifth base, the masked ones included, each changed to the next base.
    shift = {"A": "C", "C": "G", "G": "T", "T": "A"}
    subs = [(p, shift[seq[p].upper()]) for p in range(106, 192, 5)]
    edited = [make_seq_db(substitute_bases(seq, [p], [b])) for p, b in subs]
    bins = np.arange(105, 195)
    row_bin = np.repeat(bins, len(subs))
    row_sub = np.tile(np.arange(len(subs)), len(bins))
    pm.gdb_init(db)
    define_python_path_vtracks()
    assert not any(_can_vtracks_use_cpp([vt]) for vt in (*PY_PATH_VTRACKS, "v_dist", "v_val"))
    for vt in PY_PATH_VTRACKS:
        pm.gdb_init(db)
        define_python_path_vtracks()
        got = _gvariants_eval(
            vt, ["chr1"] * len(row_bin), row_bin, row_bin + 1,
            [subs[s][0] for s in row_sub], [base_code(subs[s][1]) for s in row_sub],
        )
        ref = values_on_db(db, vt, bins, define_python_path_vtracks)
        changed = False
        for s in range(len(subs)):
            want = values_on_db(edited[s], vt, bins, define_python_path_vtracks)
            np.testing.assert_allclose(got[row_sub == s], want, rtol=1e-6, err_msg=f"{vt}, substitution {s + 1}")
            changed |= not np.allclose(want, ref, equal_nan=True)
        # Not vacuous: some substitution changes the value. masked.count cannot
        # change, since a substitution keeps the case of the base it replaces.
        assert changed == (vt != "f_masked"), vt


def test_python_path_expression_ref_and_alt_equal_the_reference_and_a_really_edited_genome(make_seq_db):
    seq = variant_test_seq(seed=14)
    db = make_seq_db(seq)
    expr = "f_lse + 0.001 * v_dist + v_val + f_kmer"
    pm.gdb_init(db)
    define_python_path_vtracks()
    res = pm.gextract_variants(expr, pm.gintervals("chr1", 115, 120), iterator=1, colnames="s")
    assert (res["s.ref"] != res["s.alt"]).any()
    rng = np.random.default_rng(60427)
    for r in rng.choice(len(res), 4, replace=False):
        b = int(res["start"].iloc[r])
        want_ref = values_on_db(db, expr, [b], define_python_path_vtracks)[0]
        assert res["s.ref"].iloc[r] == pytest.approx(want_ref, rel=1e-6)
        edb = make_seq_db(substitute_bases(seq, [int(res["pos"].iloc[r])], [res["alt"].iloc[r]]))
        want_alt = values_on_db(edb, expr, [b], define_python_path_vtracks)[0]
        assert res["s.alt"].iloc[r] == pytest.approx(want_alt, rel=1e-6), r
