"""kmer.count / kmer.frac virtual tracks, ported from R misha's
tests/testthat/test-kmer.R (all 39 test_that blocks, in R's order).

R's ``gvtrack.create(name, NULL, func, list(kmer = ..., strand = ...))`` passes
the parameters as ``params``; those tests pass a dict as ``params`` here.

misha scores 4 or more sequence vtracks of one expression in batches, and its
batched kmer counts were wrong near bin edges until 5.14.1 (#200). pymisha has
no batched path, so it always counted what misha now counts; the last tests
keep it that way.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _clean_vtracks():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def count_kmers_manually(sequence, kmer, original, extended=None):
    """Occurrences of kmer whose start falls in original, in sequence fetched
    over extended (R's helper of the same name)."""
    kmer = kmer.upper()
    sequence = sequence.upper()
    o_start, o_end = original
    offset = o_start - extended[0] if extended is not None and extended[0] < o_start else 0
    k = len(kmer)
    count = sum(
        1 for i in range(offset, offset + o_end - o_start)
        if i + k <= len(sequence) and sequence[i:i + k] == kmer
    )
    possible = max(0, o_end - o_start - k + 1)
    return count, possible, (count / possible if possible > 0 else 0.0)


def _iv(start, end, chrom="chr1"):
    return pm.gintervals(chrom, start, end)


def _seq(start, end, chrom="chr1"):
    return pm.gseq_extract(_iv(start, end, chrom))[0].upper()


def _one(exprs, iv, **kw):
    """gextract of exprs with the interval itself as the iterator."""
    return pm.gextract(exprs, iv, iterator=iv, **kw)


def _vt(name, func, params=None, **kw):
    pm.gvtrack_create(name, None, func, params, **kw)


def test_kmer_count_and_kmer_frac_vtracks_work_with_basic_inputs():
    iv = _iv(200, 240)
    seq = _seq(200, 240)  # CCCTAACCCTAACCCTAACCCTAACCCTAACCCCTAACCC
    _vt("count_ta", "kmer.count", kmer="TA", strand=1)
    _vt("frac_ta", "kmer.frac", kmer="TA", strand=1)
    _vt("count_ccc", "kmer.count", kmer="CCC", strand=1)
    _vt("frac_ccc", "kmer.frac", kmer="CCC", strand=1)
    scores = _one(["count_ta", "frac_ta", "count_ccc", "frac_ccc"], iv)

    ext = (199, 241)
    ext_seq = _seq(*ext)
    ta = count_kmers_manually(ext_seq, "TA", (200, 240), ext)
    ccc = count_kmers_manually(ext_seq, "CCC", (200, 240), ext)
    assert scores["count_ta"].iloc[0] == ta[0]
    assert scores["frac_ta"].iloc[0] == pytest.approx(ta[2], rel=1e-5)
    assert scores["count_ccc"].iloc[0] == ccc[0]
    assert scores["frac_ccc"].iloc[0] == pytest.approx(ccc[2], rel=1e-5)
    # "CCCC" counts as two occurrences of "CCC"
    manual_ccc = sum(1 for i in range(len(seq) - 2) if seq[i:i + 3] == "CCC")
    assert scores["count_ccc"].iloc[0] == manual_ccc


def test_kmer_honors_gvtrack_iterator_shifts_for_count_and_frac():
    iv = _iv(2000, 2040)
    k = "TA"
    for name in ("count_small", "count_large"):
        _vt(name, "kmer.count", kmer=k, strand=1)
    for name in ("frac_small", "frac_large"):
        _vt(name, "kmer.frac", kmer=k, strand=1)
    for name in ("count_small", "frac_small"):
        pm.gvtrack_iterator(name, sshift=-10, eshift=10)
    for name in ("count_large", "frac_large"):
        pm.gvtrack_iterator(name, sshift=-1000, eshift=1000)
    scores = _one(["count_small", "count_large", "frac_small", "frac_large"], iv)

    def expected(sshift, eshift):
        # the per-vtrack interval after the shift; extend=True pads the end by k-1
        orig = (max(0, 2000 + sshift), 2040 + eshift)
        fetch = (orig[0], orig[1] + len(k) - 1)
        return count_kmers_manually(_seq(*fetch), k, orig, fetch)

    small = expected(-10, 10)
    large = expected(-1000, 1000)
    assert scores["count_small"].iloc[0] == small[0]
    assert scores["frac_small"].iloc[0] == pytest.approx(small[2], rel=1e-5)
    assert scores["count_large"].iloc[0] == large[0]
    assert scores["frac_large"].iloc[0] == pytest.approx(large[2], rel=1e-5)
    assert (scores["count_small"].iloc[0] != scores["count_large"].iloc[0]
            or abs(scores["frac_small"].iloc[0] - scores["frac_large"].iloc[0]) > 1e-12)

    # asymmetric shift
    _vt("count_asym", "kmer.count", kmer=k, strand=1)
    pm.gvtrack_iterator("count_asym", sshift=-25, eshift=5)
    asym = _one("count_asym", iv)
    assert asym["count_asym"].iloc[0] == expected(-25, 5)[0]


def test_kmer_count_and_kmer_frac_work_with_smaller_iterator_intervals():
    _vt("count_ta", "kmer.count", kmer="TA", strand=1)
    _vt("frac_ta", "kmer.frac", kmer="TA", strand=1)
    scores = pm.gextract(["count_ta", "frac_ta"], _iv(200, 240), iterator=10)
    counts, fracs = [], []
    for i in range(4):
        b = (200 + i * 10, 200 + (i + 1) * 10)
        ext = (b[0] - 1, b[1] + 1)
        c, _, f = count_kmers_manually(_seq(*ext), "TA", b, ext)
        counts.append(c)
        fracs.append(f)
    assert scores["count_ta"].tolist() == counts
    np.testing.assert_allclose(scores["frac_ta"], fracs, rtol=1e-5)


def test_kmer_count_and_kmer_frac_handle_case_sensitivity_correctly():
    iv = _iv(200, 240)
    _vt("count_ta_upper", "kmer.count", kmer="TA", strand=1)
    _vt("count_ta_lower", "kmer.count", kmer="ta", strand=1)
    _vt("count_ta_mixed", "kmer.count", kmer="Ta", strand=1)
    scores = _one(["count_ta_upper", "count_ta_lower", "count_ta_mixed"], iv)
    ext = (199, 241)
    want = count_kmers_manually(_seq(*ext), "TA", (200, 240), ext)[0]
    for c in ("count_ta_upper", "count_ta_lower", "count_ta_mixed"):
        assert scores[c].iloc[0] == want, c


def test_kmer_count_and_kmer_frac_handle_longer_kmers_correctly():
    iv = _iv(200, 240)
    _vt("count_taac", "kmer.count", kmer="TAAC", strand=1)
    _vt("frac_taac", "kmer.frac", kmer="TAAC", strand=1)
    _vt("count_ccctaa", "kmer.count", kmer="CCCTAA", strand=1)
    _vt("frac_ccctaa", "kmer.frac", kmer="CCCTAA", strand=1)
    scores = _one(["count_taac", "frac_taac", "count_ccctaa", "frac_ccctaa"], iv)
    for kmer, pad in (("TAAC", 3), ("CCCTAA", 5)):
        ext = (200 - pad, 240 + pad)
        c, _, f = count_kmers_manually(_seq(*ext), kmer, (200, 240), ext)
        assert scores[f"count_{kmer.lower()}"].iloc[0] == c
        assert scores[f"frac_{kmer.lower()}"].iloc[0] == pytest.approx(f, rel=1e-5)


def test_kmer_count_and_kmer_frac_handle_edge_cases_correctly():
    iv = _iv(200, 240)
    seq = _seq(200, 240)
    _vt("count_full", "kmer.count", kmer=seq, strand=1)
    _vt("frac_full", "kmer.frac", kmer=seq, strand=1)
    _vt("count_a", "kmer.count", kmer="A", strand=1)
    _vt("frac_a", "kmer.frac", kmer="A", strand=1)
    scores = _one(["count_full", "frac_full", "count_a", "frac_a"], iv)
    c, _, f = count_kmers_manually(_seq(200, 240), "A", (200, 240), (200, 240))
    assert scores["count_full"].iloc[0] == 1
    assert scores["frac_full"].iloc[0] == 1
    assert scores["count_a"].iloc[0] == c
    assert scores["frac_a"].iloc[0] == pytest.approx(f, rel=1e-5)


def test_kmer_count_and_kmer_frac_handle_no_matches_correctly():
    iv = _iv(200, 240)
    _vt("count_xyz", "kmer.count", kmer="XYZ", strand=1)
    _vt("frac_xyz", "kmer.frac", kmer="XYZ", strand=1)
    scores = _one(["count_xyz", "frac_xyz"], iv)
    assert scores["count_xyz"].iloc[0] == 0
    assert scores["frac_xyz"].iloc[0] == 0


def test_kmer_count_and_kmer_frac_work_with_different_iterator_sizes():
    iv = _iv(200, 240)
    _vt("count_ta", "kmer.count", kmer="TA", strand=1)
    _vt("frac_ta", "kmer.frac", kmer="TA", strand=1)
    s5, s10, s20 = (pm.gextract(["count_ta", "frac_ta"], iv, iterator=it) for it in (5, 10, 20))
    assert s5["count_ta"].sum() == s10["count_ta"].sum()
    assert s10["count_ta"].sum() == s20["count_ta"].sum()
    for s in (s5, s10, s20):
        assert ((s["frac_ta"] >= 0) & (s["frac_ta"] <= 1)).all()


def test_kmer_count_and_kmer_frac_work_with_overlapping_kmers():
    iv = _iv(200, 220)
    _vt("count_taa", "kmer.count", kmer="TAA", strand=1)
    _vt("frac_taa", "kmer.frac", kmer="TAA", strand=1)
    scores = _one(["count_taa", "frac_taa"], iv)
    ext_seq = _seq(198, 222)
    seq = _seq(200, 220)
    # R's loop, 1-based i from 2 to 2 + nchar(seq) - 3, over substr(ext_seq, i, i + 2)
    taa = sum(1 for i in range(2, 2 + len(seq) - 3 + 1) if ext_seq[i - 1:i + 2] == "TAA")
    possible = len(seq) - 3 + 1
    assert scores["count_taa"].iloc[0] == taa
    assert scores["frac_taa"].iloc[0] == pytest.approx(taa / possible, rel=1e-5)


def test_kmer_count_and_kmer_frac_handle_interval_boundaries_correctly():
    _vt("count_ta", "kmer.count", kmer="TA", strand=1)
    _vt("frac_ta", "kmer.frac", kmer="TA", strand=1)
    full = _one(["count_ta", "frac_ta"], _iv(200, 240))
    sub1 = _one(["count_ta", "frac_ta"], _iv(200, 220))
    sub2 = _one(["count_ta", "frac_ta"], _iv(220, 240))
    assert abs(full["count_ta"].iloc[0] - (sub1["count_ta"].iloc[0] + sub2["count_ta"].iloc[0])) <= 2


def test_kmer_count_and_kmer_frac_handle_multiple_chromosomes():
    _vt("count_ta", "kmer.count", kmer="TA", strand=1)
    _vt("frac_ta", "kmer.frac", kmer="TA", strand=1)
    for chrom in ("chr1", "chr2"):
        scores = _one(["count_ta", "frac_ta"], _iv(200, 220, chrom))
        assert np.issubdtype(scores["count_ta"].dtype, np.number)
        assert np.issubdtype(scores["frac_ta"].dtype, np.number)


def test_kmer_count_and_kmer_frac_return_correct_results_with_gdist():
    iv = _iv(200, 300)
    _vt("count_a", "kmer.count", kmer="A", strand=1)
    _vt("frac_a", "kmer.frac", kmer="A", strand=1)
    count_dist = np.asarray(pm.gdist("count_a", list(range(11)), intervals=iv, iterator=10))
    frac_dist = np.asarray(pm.gdist("frac_a", list(np.arange(0, 0.51, 0.1)), intervals=iv, iterator=10))
    assert (count_dist >= 0).all()
    assert (frac_dist >= 0).all()
    assert len(count_dist) == 10
    assert len(frac_dist) == 5


def test_kmer_count_and_kmer_frac_preserve_position_information():
    _vt("count_ccc", "kmer.count", kmer="CCC", strand=1)
    scores = pm.gextract("count_ccc", _iv(200, 300), iterator=10).sort_values("intervalID", kind="mergesort")
    assert {"chrom", "start", "end", "count_ccc"} <= set(scores.columns)
    assert scores["start"].iloc[0] == 200
    assert scores["end"].iloc[9] == 300
    assert len(scores) == 10


def test_kmer_count_and_kmer_frac_work_with_other_virtual_track_functions():
    _vt("count_a", "kmer.count", kmer="A", strand=1)
    _vt("frac_a", "kmer.frac", kmer="A", strand=1)
    pm.gvtrack_create("dist_track", _iv(220, 240), "distance")
    scores = pm.gextract(["count_a", "frac_a", "dist_track"], _iv(200, 300), iterator=20)
    assert {"count_a", "frac_a", "dist_track"} <= set(scores.columns)
    assert len(scores) == 5
    ok = scores[["count_a", "frac_a"]].dropna()
    assert np.corrcoef(ok["count_a"], ok["frac_a"])[0, 1] > 0.8


def test_kmer_count_and_kmer_frac_parameter_validation_works():
    with pytest.raises(ValueError):
        _vt("empty_kmer", "kmer.count", kmer="", strand=1)
    with pytest.raises(ValueError):
        _vt("numeric_kmer", "kmer.count", kmer=123, strand=1)
    with pytest.raises(ValueError):
        _vt("no_param", "kmer.count", strand=1)
    with pytest.raises(ValueError):
        _vt("multi_kmer", "kmer.count", kmer=["AA", "TT"], strand=1)


def test_kmer_count_and_kmer_frac_work_with_explicit_extension_parameters():
    iv = _iv(200, 220)
    _vt("count_default", "kmer.count", kmer="TA", strand=1)
    _vt("count_extend", "kmer.count", kmer="TA", extend=True, strand=1)
    _vt("count_no_extend", "kmer.count", kmer="TA", extend=False, strand=1)
    _vt("frac_default", "kmer.frac", kmer="TA", strand=1)
    _vt("frac_extend", "kmer.frac", kmer="TA", extend=True, strand=1)
    _vt("frac_no_extend", "kmer.frac", kmer="TA", extend=False, strand=1)
    r = _one(["count_default", "count_extend", "count_no_extend",
              "frac_default", "frac_extend", "frac_no_extend"], iv)
    assert r["count_default"].iloc[0] == r["count_extend"].iloc[0]
    assert r["frac_default"].iloc[0] == r["frac_extend"].iloc[0]
    assert np.issubdtype(r["count_no_extend"].dtype, np.number)
    assert np.issubdtype(r["frac_no_extend"].dtype, np.number)


def test_kmer_count_and_kmer_frac_extension_handles_interval_boundaries_properly():
    _vt("count_with_ext", "kmer.count", kmer="ACGT", extend=True, strand=1)
    _vt("count_no_ext", "kmer.count", kmer="ACGT", extend=False, strand=1)
    for iv in (_iv(0, 10), _iv(9990, 10000)):
        r = _one(["count_with_ext", "count_no_ext"], iv)
        assert np.issubdtype(r["count_with_ext"].dtype, np.number)
        assert np.issubdtype(r["count_no_ext"].dtype, np.number)
        assert not np.isnan(r["count_with_ext"].iloc[0])


def test_kmer_count_and_kmer_frac_performance_is_acceptable():
    iv = _iv(1, 10000)
    _vt("count_2mer", "kmer.count", kmer="AT", strand=1)
    _vt("count_5mer", "kmer.count", kmer="ATGCG", strand=1)
    _vt("count_10mer", "kmer.count", kmer="ATGCGCGTAA", strand=1)

    def elapsed(vt):
        # best of 3, so that one slow call on a loaded host does not decide it
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            pm.gextract(vt, iv, iterator=1000)
            best = min(best, time.perf_counter() - t)
        return best

    t2, t5, t10 = elapsed("count_2mer"), elapsed("count_5mer"), elapsed("count_10mer")
    assert t2 < 10 and t5 < 10 and t10 < 10
    assert t10 < t2 * 3


def test_kmer_count_and_kmer_frac_handle_overlapping_kmers_correctly():
    iv = _iv(200, 210)
    seq = _seq(200, 210)  # CCCTAACCCT
    _vt("count_ccc", "kmer.count", kmer="CCC", strand=1)
    scores = _one("count_ccc", iv)
    ext_seq = _seq(198, 212)
    # R's loop, 1-based i from 3 to 3 + nchar(seq) - 3
    manual = sum(1 for i in range(3, 3 + len(seq) - 3 + 1) if ext_seq[i - 1:i + 2] == "CCC")
    assert scores["count_ccc"].iloc[0] == manual


def test_kmer_count_and_kmer_frac_handle_strand_parameter_correctly():
    iv = _iv(200, 220)
    with pytest.warns(UserWarning, match="palindromic"):
        _vt("count_both", "kmer.count", {"kmer": "ACGT", "strand": 0})
    _vt("count_fwd", "kmer.count", {"kmer": "ACGT", "strand": 1})
    _vt("count_rev", "kmer.count", {"kmer": "ACGT", "strand": -1})
    _vt("count_aagt_both", "kmer.count", {"kmer": "AAGT", "strand": 0})
    _vt("count_aagt_fwd", "kmer.count", {"kmer": "AAGT", "strand": 1})
    _vt("count_aagt_rev", "kmer.count", {"kmer": "AAGT", "strand": -1})
    s = _one(["count_both", "count_fwd", "count_rev",
              "count_aagt_both", "count_aagt_fwd", "count_aagt_rev"], iv).iloc[0]
    assert s["count_fwd"] == s["count_rev"]
    assert s["count_both"] == 2 * s["count_fwd"]
    assert s["count_aagt_both"] == s["count_aagt_fwd"] + s["count_aagt_rev"]

    with pytest.warns(UserWarning, match="palindromic"):
        _vt("frac_both", "kmer.frac", {"kmer": "ACGT", "strand": 0})
    _vt("frac_fwd", "kmer.frac", {"kmer": "ACGT", "strand": 1})
    _vt("frac_rev", "kmer.frac", {"kmer": "ACGT", "strand": -1})
    f = _one(["frac_both", "frac_fwd", "frac_rev"], iv).iloc[0]
    assert f["frac_fwd"] == pytest.approx(f["frac_rev"], rel=1e-5)
    assert f["frac_both"] >= f["frac_fwd"]
    assert f["frac_both"] <= min(1.0, 2 * f["frac_fwd"])


def test_kmer_functions_properly_handle_strand_parameter_for_forward_strand():
    iv = _iv(200, 240)
    _vt("count_ta_fwd", "kmer.count", {"kmer": "TA", "strand": 1})
    _vt("frac_ta_fwd", "kmer.frac", {"kmer": "TA", "strand": 1})
    scores = _one(["count_ta_fwd", "frac_ta_fwd"], iv)
    assert scores["count_ta_fwd"].iloc[0] > 0
    with pytest.warns(UserWarning, match="palindromic"):
        _vt("count_ta_both", "kmer.count", {"kmer": "TA", "strand": 0})
    total = _one("count_ta_both", iv)["count_ta_both"].iloc[0]
    chunks = pm.gextract("count_ta_both", iv, iterator=10)["count_ta_both"].sum()
    assert total == chunks


def test_kmer_functions_properly_handle_strand_parameter_for_reverse_strand():
    iv = _iv(200, 240)
    seq = _seq(200, 240)
    revcomp_seq = pm.grevcomp(seq)
    _vt("count_ag_rev", "kmer.count", {"kmer": "AG", "strand": -1, "extend": False})
    _vt("frac_ag_rev", "kmer.frac", {"kmer": "AG", "strand": -1})
    scores = _one(["count_ag_rev", "frac_ag_rev"], iv)
    _vt("count_ct_fwd", "kmer.count", {"kmer": "CT", "strand": 1, "extend": False})
    ct_fwd = _one("count_ct_fwd", iv)["count_ct_fwd"].iloc[0]
    assert scores["count_ag_rev"].iloc[0] == ct_fwd
    ct_count = sum(1 for i in range(len(seq) - 1) if seq[i:i + 2] == "CT")
    ag_rev_count = sum(1 for i in range(len(revcomp_seq) - 1) if revcomp_seq[i:i + 2] == "AG")
    assert ct_count == ag_rev_count
    assert scores["count_ag_rev"].iloc[0] == ct_count


def test_kmer_functions_on_both_strands_equal_the_sum_of_forward_and_reverse_strands():
    iv = _iv(200, 240)
    kmer = "AGTC"
    for strand, suffix in ((0, "both"), (1, "fwd"), (-1, "rev")):
        _vt(f"count_{suffix}", "kmer.count", {"kmer": kmer, "strand": strand})
        _vt(f"frac_{suffix}", "kmer.frac", {"kmer": kmer, "strand": strand})
    s = _one(["count_both", "count_fwd", "count_rev", "frac_both", "frac_fwd", "frac_rev"], iv).iloc[0]
    assert s["count_both"] == s["count_fwd"] + s["count_rev"]
    possible = 40 - len(kmer) + 1
    expected = (s["count_fwd"] + s["count_rev"]) / (2 * possible)
    assert s["frac_both"] == pytest.approx(expected, rel=0.01)


def test_kmer_functions_handle_palindromic_sequences_correctly_with_strand_parameter():
    iv = _iv(200, 240)
    palindromic_kmers = ("AT", "TA", "GC", "CG", "ATAT", "CGCG", "ACGT")
    palindrome = "ACGT"
    with pytest.warns(UserWarning, match="palindromic"):
        _vt("count_pal_both", "kmer.count", {"kmer": palindrome, "strand": 0})
    _vt("count_pal_fwd", "kmer.count", {"kmer": palindrome, "strand": 1})
    _vt("count_pal_rev", "kmer.count", {"kmer": palindrome, "strand": -1})
    s = _one(["count_pal_fwd", "count_pal_rev"], iv).iloc[0]
    assert s["count_pal_fwd"] == s["count_pal_rev"]
    for pal in palindromic_kmers[2:5]:
        pm.gvtrack_rm("count_pal_fwd")
        pm.gvtrack_rm("count_pal_rev")
        _vt("count_pal_fwd", "kmer.count", {"kmer": pal, "strand": 1})
        _vt("count_pal_rev", "kmer.count", {"kmer": pal, "strand": -1})
        s = _one(["count_pal_fwd", "count_pal_rev"], iv).iloc[0]
        assert s["count_pal_fwd"] == s["count_pal_rev"], pal


def test_kmer_functions_correctly_handle_strand_with_overlapping_intervals():
    iv1, iv2, combined = _iv(200, 220), _iv(210, 230), _iv(200, 230)
    kmer = "AGTC"
    _vt("count_fwd", "kmer.count", {"kmer": kmer, "strand": 1})
    _vt("count_rev", "kmer.count", {"kmer": kmer, "strand": -1})
    _vt("count_both", "kmer.count", {"kmer": kmer, "strand": 0})
    for vt in ("count_fwd", "count_rev", "count_both"):
        a, b, c = (_one(vt, iv)[vt].iloc[0] for iv in (iv1, iv2, combined))
        assert c >= a and c >= b, vt
    s = _one(["count_fwd", "count_rev", "count_both"], combined).iloc[0]
    assert s["count_both"] == s["count_fwd"] + s["count_rev"]


def test_kmer_functions_handle_strand_parameter_with_different_kmer_lengths():
    iv = _iv(200, 240)
    for kmer in ("AG", "AGCT", "AGCTAG", "AGCTAGCT"):
        if kmer == pm.grevcomp(kmer):  # palindromic, skip
            continue
        _vt("count_fwd", "kmer.count", {"kmer": kmer, "strand": 1, "extend": False})
        _vt("count_rev", "kmer.count", {"kmer": kmer, "strand": -1, "extend": False})
        _vt("count_both", "kmer.count", {"kmer": kmer, "strand": 0, "extend": False})
        s = _one(["count_fwd", "count_rev", "count_both"], iv).iloc[0]
        assert s["count_both"] == s["count_fwd"] + s["count_rev"], kmer
        _vt("count_rc_fwd", "kmer.count", {"kmer": pm.grevcomp(kmer), "strand": 1, "extend": False})
        assert s["count_rev"] == _one("count_rc_fwd", iv)["count_rc_fwd"].iloc[0], kmer
        for vt in ("count_fwd", "count_rev", "count_both", "count_rc_fwd"):
            pm.gvtrack_rm(vt)


def test_kmer_functions_correctly_handle_negative_strand_with_extension_at_boundaries():
    kmer = "ACGT"
    _vt("count_fwd_ext", "kmer.count", {"kmer": kmer, "strand": 1, "extend": True})
    _vt("count_fwd_noext", "kmer.count", {"kmer": kmer, "strand": 1, "extend": False})
    _vt("count_rev_ext", "kmer.count", {"kmer": kmer, "strand": -1, "extend": True})
    _vt("count_rev_noext", "kmer.count", {"kmer": kmer, "strand": -1, "extend": False})
    cols = ["count_fwd_ext", "count_fwd_noext", "count_rev_ext", "count_rev_noext"]
    for iv in (_iv(0, 20), _iv(9980, 10000)):
        s = _one(cols, iv).iloc[0]
        assert not s[cols].isna().any()


def test_kmer_functions_handle_strand_with_mixed_case_sequences():
    iv = _iv(200, 240)
    for kmer, case in (("AG", "upper"), ("ag", "lower"), ("Ag", "mixed")):
        _vt(f"count_{case}_fwd", "kmer.count", {"kmer": kmer, "strand": 1, "extend": False})
        _vt(f"count_{case}_rev", "kmer.count", {"kmer": kmer, "strand": -1, "extend": False})
    f = _one(["count_upper_fwd", "count_lower_fwd", "count_mixed_fwd"], iv).iloc[0]
    r = _one(["count_upper_rev", "count_lower_rev", "count_mixed_rev"], iv).iloc[0]
    assert f["count_upper_fwd"] == f["count_lower_fwd"] == f["count_mixed_fwd"]
    assert r["count_upper_rev"] == r["count_lower_rev"] == r["count_mixed_rev"]
    _vt("count_ct_upper_fwd", "kmer.count", {"kmer": "CT", "strand": 1, "extend": False})
    _vt("count_ct_lower_fwd", "kmer.count", {"kmer": "ct", "strand": 1, "extend": False})
    ct = _one(["count_ct_upper_fwd", "count_ct_lower_fwd"], iv).iloc[0]
    assert ct["count_ct_upper_fwd"] == r["count_upper_rev"]
    assert ct["count_ct_lower_fwd"] == r["count_lower_rev"]


def test_sum_of_fraction_of_t_c_g_a_equals_1():
    iv = _iv(200, 240)
    for strand in (1, -1, 0):
        for b in "TCGA":
            _vt(f"frac_{b.lower()}", "kmer.frac", kmer=b, strand=strand)
        s = _one("frac_a + frac_c + frac_g + frac_t", iv, colnames=["s"])["s"]
        assert s.sum() == pytest.approx(1, rel=1e-5), strand


def test_kmer_frac_correctly_doubles_possible_positions_when_strand_0():
    iv = _iv(200, 210)
    seq = _seq(200, 210)  # CCCTAACCCT
    kmer, rc_kmer = "AG", "CT"
    _vt("frac_fwd", "kmer.frac", {"kmer": kmer, "strand": 1, "extend": False})
    _vt("frac_rev", "kmer.frac", {"kmer": kmer, "strand": -1, "extend": False})
    _vt("frac_both", "kmer.frac", {"kmer": kmer, "strand": 0, "extend": False})
    _vt("frac_rc_fwd", "kmer.frac", {"kmer": rc_kmer, "strand": 1, "extend": False})
    s = _one(["frac_fwd", "frac_rev", "frac_both", "frac_rc_fwd"], iv).iloc[0]
    ag = sum(1 for i in range(len(seq) - 1) if seq[i:i + 2] == kmer)
    ct = sum(1 for i in range(len(seq) - 1) if seq[i:i + 2] == rc_kmer)
    possible = len(seq) - len(kmer) + 1
    # R's expect_equal: tolerance 1.5e-8, relative
    assert s["frac_fwd"] == pytest.approx(ag / possible, rel=1.5e-8)
    assert s["frac_rev"] == pytest.approx(ct / possible, rel=1.5e-8)
    assert s["frac_rc_fwd"] == pytest.approx(ct / possible, rel=1.5e-8)
    assert s["frac_both"] == pytest.approx((ag + ct) / (possible * 2), rel=1.5e-8)
    assert s["frac_both"] * 2 * possible == pytest.approx(ag + ct, rel=1.5e-8)


def test_kmer_frac_handles_edge_cases_with_strand_0_correctly():
    short = _iv(200, 202)
    _vt("frac_1bp_both", "kmer.frac", {"kmer": "A", "strand": 0})
    _vt("frac_2bp_both", "kmer.frac", {"kmer": "AG", "strand": 0})
    s = _one(["frac_1bp_both", "frac_2bp_both"], short).iloc[0]
    assert 0 <= s["frac_1bp_both"] <= 1
    assert 0 <= s["frac_2bp_both"] <= 1

    pal_iv = _iv(200, 220)
    _vt("frac_at_fwd", "kmer.frac", {"kmer": "AT", "strand": 1})
    with pytest.warns(UserWarning, match="palindromic"):
        _vt("frac_at_both", "kmer.frac", {"kmer": "AT", "strand": 0})
    p = _one(["frac_at_fwd", "frac_at_both"], pal_iv).iloc[0]
    assert p["frac_at_both"] == pytest.approx(p["frac_at_fwd"] / 2, rel=1.5e-8, abs=1e-12)

    for seq_len in (10, 20, 50):
        for k_len in (1, 2, 4):
            if k_len <= seq_len:
                name = f"frac_{k_len}mer_{seq_len}_both"
                _vt(name, "kmer.frac", {"kmer": "A" * k_len, "strand": 0})
                v = _one(name, _iv(200, 200 + seq_len))[name].iloc[0]
                assert 0 <= v <= 1


def test_kmer_frac_with_strand_0_handles_boundary_extension_correctly():
    kmer = "AAG"
    _vt("frac_both_extend", "kmer.frac", {"kmer": kmer, "strand": 0, "extend": True})
    _vt("frac_both_no_extend", "kmer.frac", {"kmer": kmer, "strand": 0, "extend": False})
    s = _one(["frac_both_extend", "frac_both_no_extend"], _iv(200, 210)).iloc[0]
    assert s["frac_both_extend"] >= s["frac_both_no_extend"]
    small = _one(["frac_both_extend", "frac_both_no_extend"], _iv(200, 203)).iloc[0]
    assert small["frac_both_extend"] >= small["frac_both_no_extend"]
    assert 0 <= small["frac_both_no_extend"] <= 1


def test_sum_of_base_fractions_equals_1_when_strand_0():
    iv = _iv(200, 240)
    for b in "ACGT":
        _vt(f"frac_{b.lower()}", "kmer.frac", {"kmer": b, "strand": 0})
    total = _one("frac_a + frac_c + frac_g + frac_t", iv, colnames=["sum"])["sum"].iloc[0]
    assert total == pytest.approx(1, rel=1e-5)
    for b in "ACGT":
        _vt(f"frac_{b.lower()}_fwd", "kmer.frac", {"kmer": b, "strand": 1})
    total_fwd = _one("frac_a_fwd + frac_c_fwd + frac_g_fwd + frac_t_fwd", iv, colnames=["sum_fwd"])
    assert total_fwd["sum_fwd"].iloc[0] == pytest.approx(1, rel=1e-5)

    cols = [c for b in "acgt" for c in (f"frac_{b}", f"frac_{b}_fwd")]
    s = _one(cols, iv).iloc[0]
    seq = _seq(200, 240)
    rc = pm.grevcomp(seq)
    n = len(seq)
    for b in "ACGT":
        assert s[f"frac_{b.lower()}_fwd"] == pytest.approx(seq.count(b) / n, rel=1e-5)
        assert s[f"frac_{b.lower()}"] == pytest.approx((seq.count(b) + rc.count(b)) / (2 * n), rel=1e-5)


def test_kmer_honors_iterator_shifts_across_multiple_magnitudes_extend_true_fwd_strand():
    k = "TA"  # palindromic is fine for forward-only tests
    shifts = (0, 1, 5, 10, 25, 50, 100, 250, 1000)
    names = [f"shift_{s}" for s in shifts]
    scores = _shift_scores(names, 1, k, shifts)
    for name, s in zip(names, shifts, strict=True):
        orig, fetch, seq = _window(s, k)
        count, _, frac = count_kmers_manually(seq, k, orig, fetch)
        assert scores[f"count_{name}"].iloc[0] == count, s
        tol = 1e-5 if s >= 250 else 1e-6
        assert scores[f"frac_{name}"].iloc[0] == pytest.approx(frac, rel=tol), s


def test_kmer_honors_iterator_shifts_without_extension_extend_false_fwd_strand():
    k = "TA"
    shifts = (0, 5, 10, 50, 200)
    names = [f"noext_shift_{s}" for s in shifts]
    for name, s in zip(names, shifts, strict=True):
        for func in ("kmer.count", "kmer.frac"):
            vt = f"{func.split('.')[1]}_{name}"
            _vt(vt, func, kmer=k, strand=1, extend=False)
            pm.gvtrack_iterator(vt, sshift=-s, eshift=s)
    iv = _iv(2000, 2040)
    scores = _one([f"{f}_{n}" for f in ("count", "frac") for n in names], iv)
    for name, s in zip(names, shifts, strict=True):
        orig = (max(0, 2000 - s), 2040 + s)
        count, _, frac = count_kmers_manually(_seq(*orig), k, orig, orig)  # no extension at fetch
        assert scores[f"count_{name}"].iloc[0] == count, s
        tol = 1e-5 if s >= 50 else 1e-6
        assert scores[f"frac_{name}"].iloc[0] == pytest.approx(frac, rel=tol), s


def _shift_scores(names, strand, k, shifts):
    for name, s in zip(names, shifts, strict=True):
        for func in ("kmer.count", "kmer.frac"):
            vt = f"{func.split('.')[1]}_{name}"
            pm.gvtrack_create(vt, None, func, kmer=k, strand=strand)
            pm.gvtrack_iterator(vt, sshift=-s, eshift=s)
    iv = pm.gintervals("chr1", 2000, 2040)
    cols = [f"{f}_{n}" for f in ("count", "frac") for n in names]
    return pm.gextract(cols, iv, iterator=iv)


def _window(s, k):
    """The shifted iterator interval and its fetch: extend=True pads the end by
    k - 1, on either strand."""
    orig = (max(0, 2000 - s), 2040 + s)
    fetch = (orig[0], orig[1] + len(k) - 1)
    seq = pm.gseq_extract(pm.gintervals("chr1", *fetch))[0].upper()
    return orig, fetch, seq


def test_kmer_honors_iterator_shifts_for_reverse_strand_extend_true():
    k = "AG"  # non-palindromic; reverse complement is CT
    rc_k = pm.grevcomp(k)
    shifts = (0, 10, 200)
    names = [f"rev_shift_{s}" for s in shifts]
    scores = _shift_scores(names, -1, k, shifts)
    for name, s in zip(names, shifts, strict=True):
        orig, fetch, seq = _window(s, k)
        # Reverse-strand k-mers starting in the window = its reverse complement
        # on the forward strand
        count, _, frac = count_kmers_manually(seq, rc_k, orig, fetch)
        assert scores[f"count_{name}"].iloc[0] == count, s
        assert scores[f"frac_{name}"].iloc[0] == pytest.approx(frac, rel=1e-6), s


def test_kmer_honors_iterator_shifts_for_both_strands_extend_true():
    k = "AG"
    rc_k = pm.grevcomp(k)
    shifts = (0, 10, 200)
    names = [f"both_shift_{s}" for s in shifts]
    scores = _shift_scores(names, 0, k, shifts)
    for name, s in zip(names, shifts, strict=True):
        orig, fetch, seq = _window(s, k)
        fwd, _, _ = count_kmers_manually(seq, k, orig, fetch)
        rev, _, _ = count_kmers_manually(seq, rc_k, orig, fetch)
        possible = max(0, (orig[1] - orig[0]) - len(k) + 1)
        expected_frac = (fwd + rev) / (2 * possible) if possible > 0 else 0.0
        assert scores[f"count_{name}"].iloc[0] == fwd + rev, s
        assert scores[f"frac_{name}"].iloc[0] == pytest.approx(expected_frac, rel=1e-5), s


# With 4 or more sequence vtracks (and none filtered) misha scores kmer vtracks
# in batches. A batched kmer vtrack must count exactly the k-mers it counts when
# scored on its own: those whose 0-based genomic start lies in its (shifted)
# iterator interval, on either strand.
@pytest.fixture
def kmer_seq_db(tmp_path):
    """kmer_seq_db({name: seq}): make a database of these chromosomes and
    make it the working one."""

    def make(seqs):
        fasta = tmp_path / "genome.fa"
        fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in seqs.items()))
        pm.gdb_create(str(tmp_path / "db"), str(fasta))
        pm.gdb_init(str(tmp_path / "db"))

    return make


def _alone_and_batched(vt, others, intervals, iterator):
    expr = " + ".join([vt, *(f"0 * {o}" for o in others)])
    alone = pm.gextract(vt, intervals, iterator=iterator, colnames=["v"])["v"].to_numpy()
    batched = pm.gextract(expr, intervals, iterator=iterator, colnames=["v"])["v"].to_numpy()
    return alone, batched


def test_batched_kmer_vtracks_count_the_same_kmers_as_a_kmer_vtrack_scored_alone(kmer_seq_db):
    rng = np.random.default_rng(60417)

    def rand_seq(n):
        return rng.choice(list("ACGT"), size=n)

    soft = rand_seq(613)
    soft = np.where((np.arange(613) // 25) % 2 == 1, np.char.lower(soft), soft)
    kmer_seq_db({"chr1": "".join(rand_seq(601)), "chr2": "".join(soft)})

    # Whole chromosomes in 1 bp and 7 bp bins: every bin touching a chromosome
    # end is scored, including a short last bin.
    intervals = pm.gintervals_all()
    others = ("o1", "o2", "o3")
    for func in ("kmer.count", "kmer.frac"):
        for strand in (1, 0, -1):
            for extend in (True, False):
                # The other k-mers are longer than most tested ones, so under
                # extend=False a short k-mer shares its batch with a longer one.
                for o, kmer in zip(others, ("C", "GG", "GATTACA"), strict=True):
                    pm.gvtrack_create(o, None, "kmer.count", kmer=kmer, extend=extend, strand=strand)
                for kmer in ("A", "CA", "ACG", "TTAC"):
                    for shift in (None, (-3, 2), (4, 4)):
                        pm.gvtrack_create("k1", None, func, kmer=kmer, extend=extend, strand=strand)
                        if shift is not None:
                            pm.gvtrack_iterator("k1", sshift=shift[0], eshift=shift[1])
                        for iterator in (1, 7):
                            alone, batched = _alone_and_batched("k1", others, intervals, iterator)
                            np.testing.assert_allclose(
                                batched, alone, rtol=1.5e-8,
                                err_msg=f"{func} kmer={kmer} strand={strand} extend={extend} "
                                        f"shift={shift} iterator={iterator}",
                            )


def test_batched_kmer_count_does_not_count_a_reverse_strand_kmer_anchored_left_of_the_bin(kmer_seq_db):
    # chr2 differs from chr1 only at base 99, which completes CGT at [98, 101):
    # ACG on the reverse strand anchored at 98, outside the bin [100, 110).
    chr1 = ["A"] * 300
    chr1[98:101] = ["C", "A", "T"]
    chr2 = list(chr1)
    chr2[99] = "G"
    kmer_seq_db({"chr1": "".join(chr1), "chr2": "".join(chr2)})
    for v in ("k1", "k2", "k3", "k4"):
        pm.gvtrack_create(v, None, "kmer.count", kmer="ACG", extend=True, strand=0)
    bins = pm.gintervals(["chr1", "chr2"], [100, 100], [110, 110])
    alone = pm.gextract("k1", bins, iterator=10, colnames=["v"])["v"].tolist()
    batched = pm.gextract("k1 + 0 * k2 + 0 * k3 + 0 * k4", bins, iterator=10, colnames=["v"])["v"].tolist()
    assert alone == [0, 0]
    assert batched == [0, 0]
    # Not in R's test: chr2's bin before counts that k-mer, plus ACG on the
    # forward strand at 97.
    before = pm.gintervals(["chr1", "chr2"], [90, 90], [100, 100])
    assert pm.gextract("k1", before, iterator=10, colnames=["v"])["v"].tolist() == [0, 2]


# ---------------------------------------------------------------------------
# kmer blocks of other R test files
# ---------------------------------------------------------------------------


def test_masked_functions_work_in_expressions():
    # R test-masked.R, "masked functions work in expressions"
    iv = _iv(200, 400)
    pm.gvtrack_create("masked_frac", None, "masked.frac")
    _vt("g_frac", "kmer.frac", kmer="G")
    scores = pm.gextract("masked_frac + g_frac", iv, iterator=iv, colnames=["combined"])
    assert "combined" in scores.columns
    assert np.issubdtype(scores["combined"].dtype, np.number)
