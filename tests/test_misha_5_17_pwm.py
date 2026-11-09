"""R misha 5.17.0 (PRs #222, #223, #225): PWM edit distance, gseq_pwm_edits, PSSM
validation, NA logical arguments, motif ID de-duplication and the potts kernels.

Each test ports a test_that block R 5.17.0 added or changed, with the same inputs
and expectations: test-pwm.R, test-vtrack-params-validation.R, test-sequence.R,
test-motif-import.R, test-gseq-pwm-edits.R, test-gseq-pwm-edits-below.R,
test-pwm-edit-distance.R, test-pwm-edit-distance-below.R,
test-pwm-edit-distance-adversarial.R, test-potts.R and test-potts-vtrack.R. Where R
draws random inputs, the inputs here come from numpy with seed 60427 and the checks
are R's (agreement with an exhaustive search or with the virtual track), except for
the LSE search for 3+ edits, a heuristic, which is checked on R's own inputs.
"""

from __future__ import annotations

import itertools
import os
import warnings

import _pymisha
import numpy as np
import pandas as pd
import pytest
from _potts_oracle import potts_ref_model

import pymisha as pm
from pymisha._shared import _potts_params
from pymisha.motif_import import _dedup_motif_ids

BASES = "ACGT"
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
SEED = 60427


@pytest.fixture(autouse=True)
def _cleanup():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)
    yield
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def _df(rows):
    return pd.DataFrame(rows, columns=list(BASES), dtype=float)


def _tiny_genome(path, seq, chrom="chr1"):
    """A one-chromosome database holding `seq`, made the current one."""
    (path / "tracks").mkdir()
    (path / "seq").mkdir()
    (path / "chrom_sizes.txt").write_text(f"{chrom}\t{len(seq)}\n")
    (path / "seq" / f"{chrom}.seq").write_bytes(seq.encode())
    pm.gdb_init(str(path))


def _by_interval(df, col, n):
    """A vtrack column in input-interval order."""
    return df.sort_values("intervalID")[col].to_numpy(dtype=float)


def _n_edits_per_seq(res, n):
    out = np.full(n, np.nan)
    if len(res):
        first = res.drop_duplicates("seq_idx")
        out[first["seq_idx"].to_numpy() - 1] = first["n_edits"].to_numpy()
    return out


def _gamma_pssm(rng, L, shape):
    m = rng.gamma(shape, size=(L, 4))
    return _df(m / m.sum(axis=1, keepdims=True))


def _log_probs(pssm, prior):
    p = np.asarray(pssm, dtype=float) + prior
    return np.log(p / p.sum(axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# Brute-force references (R's test helpers)
# ---------------------------------------------------------------------------

def bruteforce_edit_distance_with_indels(seq, pssm, threshold, max_indels, max_edits=None,
                                         bidirect=False, prior=0.0, direction="above",
                                         starts=None):
    """R's bruteforce_pwm_edit_distance_with_indels (test-pwm-edit-distance-adversarial.R):
    enumerate every alignment of the L motif columns to the W bases of each window
    (W in [L - D, L + D]) with at most D indels. An aligned column scores its base, an
    inserted column scores the column maximum ("above") or minimum ("below"), and a
    deleted base scores nothing; then greedily count the substitutions the aligned columns
    need. A zero-probability base is a forced substitution ("above" only). `starts`
    restricts the window starts (1-based)."""
    pssm = np.asarray(pssm, dtype=float)
    L = pssm.shape[0]
    S = len(seq)
    D = max_indels
    below = direction == "below"
    if prior > 0:
        pssm = pssm + prior
        pssm = pssm / pssm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_pssm = np.log(pssm)
    col_t = log_pssm.min(axis=1) if below else log_pssm.max(axis=1)

    def alignments(W):
        res = []

        def rec(i, j, k, a):
            if i == L and j == W:
                res.append((a, k))
                return
            if i < L and j < W:
                rec(i + 1, j + 1, k, a[:i] + (j + 1,) + a[i + 1:])
            if k < D and i < L:
                rec(i + 1, j, k + 1, a[:i] + (0,) + a[i + 1:])
            if k < D and j < W:
                rec(i, j + 1, k + 1, a)

        rec(0, 0, 0, (0,) * L)
        return res

    def window_edits(sub, reverse):
        bases = [_COMP[b] for b in reversed(sub)] if reverse else list(sub)
        if any(b not in BASES for b in bases):
            return np.nan
        bi = [BASES.index(b) for b in bases]
        best = np.nan
        for a, k in alignments(len(sub)):
            matched = [i for i in range(L) if a[i] > 0]
            inserted = [i for i in range(L) if a[i] == 0]
            sc = np.array([log_pssm[i, bi[a[i] - 1]] for i in matched])
            mand = np.zeros(len(sc), dtype=bool) if below else ~np.isfinite(sc)
            score = sc[~mand].sum() + col_t[matched][mand].sum() + col_t[inserted].sum()
            edits = k + int(mand.sum())
            deficit = score - threshold if below else threshold - score
            if deficit > 1e-10:
                gains = np.sort(np.abs(col_t[matched] - sc)[~mand])[::-1]
                hit = np.flatnonzero(np.cumsum(gains) >= deficit - 1e-10)
                if len(hit) == 0:
                    continue
                edits += int(hit[0]) + 1
            if max_edits is not None and edits > max_edits:
                continue
            if np.isnan(best) or edits < best:
                best = edits
        return best

    best = np.nan
    for W in range(max(1, L - D), L + D + 1):
        if W > S:
            continue
        for p in range(1, S - W + 2):
            if starts is not None and p not in starts:
                continue
            sub = seq[p - 1:p - 1 + W]
            for rev in ((False, True) if bidirect else (False,)):
                r = window_edits(sub, rev)
                if not np.isnan(r) and (np.isnan(best) or r < best):
                    best = r
    return best


def lse_exact_min_edits(seq, pssm, n_starts, thr, direction, prior=0.01):
    """R's lse_exact_min_edits (test-pwm-edit-distance-below.R): the minimum number of
    substitutions that bring the strand-combined LSE of the windows starting at 1..n_starts
    past `thr`, by enumerating every sequence."""
    lp = _log_probs(pssm, prior)
    L = lp.shape[0]
    b = np.array([BASES.index(c) for c in seq])
    every = np.array(list(itertools.product(range(4), repeat=len(b))))
    cols = []
    for p in range(n_starts):
        f = sum(lp[i, every[:, p + i]] for i in range(L))
        r = sum(lp[L - 1 - i, 3 - every[:, p + i]] for i in range(L))
        cols += [f, r]
    sc = np.column_stack(cols)
    m = sc.max(axis=1)
    lse = m + np.log(np.exp(sc - m[:, None]).sum(axis=1))
    ok = lse >= thr if direction == "above" else lse <= thr
    if not ok.any():
        return np.nan
    return float((every != b).sum(axis=1)[ok].min())


# ---------------------------------------------------------------------------
# PSSM validation (test-pwm.R)
# ---------------------------------------------------------------------------

_PROBS = _df([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1],
              [0.25, 0.25, 0.25, 0.25], [0.1, 0.1, 0.1, 0.7]])
_IV2 = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 1000], "end": [100, 1100]})


def test_pssm_counts_are_accepted_log_odds_na_and_zero_rows_are_refused():
    counts = (_PROBS.to_numpy() * 20).astype(np.int64)
    assert np.issubdtype(counts.dtype, np.integer)
    pm.gvtrack_create("pssm_counts", None, "pwm.max", pssm=counts, prior=0.01)
    pm.gvtrack_create("pssm_probs", None, "pwm.max", pssm=_PROBS, prior=0.01)
    r = pm.gextract(["pssm_counts", "pssm_probs"], _IV2, iterator=_IV2)
    np.testing.assert_allclose(r["pssm_counts"], r["pssm_probs"], rtol=1e-5)
    pm.gvtrack_create("ed_counts", None, "pwm.edit_distance", pssm=counts, score_thresh=-3, prior=0.01)
    pm.gvtrack_create("ed_probs", None, "pwm.edit_distance", pssm=_PROBS, score_thresh=-3, prior=0.01)
    r = pm.gextract(["ed_counts", "ed_probs"], _IV2, iterator=_IV2)
    np.testing.assert_array_equal(r["ed_counts"], r["ed_probs"])
    np.testing.assert_allclose(pm.gseq_pwm("ACGTACGT", counts, mode="max", prior=0.01),
                               pm.gseq_pwm("ACGTACGT", _PROBS, mode="max", prior=0.01), rtol=1e-6)

    logodds = np.log2(_PROBS.to_numpy() / 0.25)
    na = _PROBS.to_numpy().copy()
    na[1, 0] = np.nan
    zero_row = np.vstack([_PROBS.to_numpy(), np.zeros(4)])
    bad = {"log-odds": (logodds, "negative values"), "NA": (na, "NA, NaN or infinite"),
           "zero row": (zero_row, "row 5 sums to 0")}
    for m, msg in bad.values():
        for func, kw in (("pwm", {}), ("pwm.count", {"score_thresh": -5}),
                         ("pwm.edit_distance", {"score_thresh": -5}),
                         ("pwm.edit_distance.lse", {"score_thresh": -5}),
                         ("pwm.n_mutations", {"score_thresh": -5})):
            with pytest.raises(ValueError, match=msg):
                pm.gvtrack_create("bad_pwm", None, func, pssm=m, **kw)
        for m_kind in (m, _df(m)):
            with pytest.raises(ValueError, match=msg):
                pm.gseq_pwm("ACGTACGT", m_kind, mode="max")
            with pytest.raises(ValueError, match=msg):
                pm.gseq_pwm_edits("ACGTACGT", m_kind, score_thresh=-5)
    with pytest.raises(ValueError, match="log-odds"):
        pm.gvtrack_create("bad_pwm", None, "pwm", pssm=logodds)


def test_pssm_noise_below_zero_is_zero_a_real_negative_is_refused():
    probs = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1],
                      [0.5, 0.5, 0, 0], [0.1, 0.1, 0.1, 0.7]])
    noisy = probs.copy()
    noisy[2, 2] = -1e-18  # e.g. left by 1 - sum(...)
    counts_noisy = probs * 50
    counts_noisy[2, 3] = -1e-8  # below 1e-9 of the row maximum (25)

    pm.gvtrack_create("pssm_clean", None, "pwm.max", pssm=probs, prior=0.01)
    pm.gvtrack_create("pssm_noisy", None, "pwm.max", pssm=noisy, prior=0.01)
    pm.gvtrack_create("pssm_cnoisy", None, "pwm.max", pssm=counts_noisy, prior=0.01)
    r = pm.gextract(["pssm_clean", "pssm_noisy", "pssm_cnoisy"], _IV2, iterator=_IV2)
    np.testing.assert_array_equal(r["pssm_noisy"], r["pssm_clean"])
    np.testing.assert_allclose(r["pssm_cnoisy"], r["pssm_clean"], rtol=1e-6)
    np.testing.assert_array_equal(pm.gseq_pwm("ACGTACGT", noisy, mode="max", prior=0.01),
                                  pm.gseq_pwm("ACGTACGT", probs, mode="max", prior=0.01))
    assert (list(pm.gseq_pwm_edits("ACGTACGT", noisy, score_thresh=-3)["n_edits"])
            == list(pm.gseq_pwm_edits("ACGTACGT", probs, score_thresh=-3)["n_edits"]))

    # A negative beyond the noise tolerance is still refused, small or not
    for v in (-1e-3, -1e-6, -0.2):
        bad = probs.copy()
        bad[2, 2] = v
        with pytest.raises(ValueError, match="negative values"):
            pm.gvtrack_create("bad_neg", None, "pwm", pssm=bad)
        with pytest.raises(ValueError, match="negative values"):
            pm.gseq_pwm("ACGTACGT", bad, mode="max")
    # and so is a row with no positive entry
    bad = probs.copy()
    bad[2] = [-1e-18, 0, 0, 0]
    with pytest.raises(ValueError, match="negative values"):
        pm.gvtrack_create("bad_neg", None, "pwm", pssm=bad)


# ---------------------------------------------------------------------------
# NA logical arguments (test-vtrack-params-validation.R, test-sequence.R,
# test-gseq-pwm-edits.R)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("na", [np.nan, pd.NA])
def test_an_na_bidirect_or_extend_of_a_sequence_vtrack_is_an_error(na):
    pssm = _df([[.7, .1, .1, .1], [.1, .7, .1, .1]])

    def na_vtrack(func, **kw):
        pm.gvtrack_create("v_na", None, func, **kw)
        return pm.gextract("v_na", pm.gintervals("1", 10000, 10100), iterator=100)

    with pytest.raises(ValueError, match="bidirect parameter must be True or False"):
        na_vtrack("pwm", pssm=pssm, bidirect=na)
    with pytest.raises(ValueError, match="extend parameter must be True, False"):
        na_vtrack("pwm", pssm=pssm, extend=na)
    with pytest.raises(ValueError, match="extend parameter must be True, False"):
        na_vtrack("pwm.max", pssm=pssm, bidirect=False, extend=na)
    with pytest.raises(ValueError, match="extend parameter must be True or False"):
        na_vtrack("kmer.count", kmer="AC", extend=na)
    with pytest.raises(ValueError, match="bidirect parameter must be True or False"):
        na_vtrack("pwm.edit_distance", pssm=pssm, score_thresh=-1, bidirect=na)
    with pytest.raises(ValueError, match="extend parameter must be True, False"):
        na_vtrack("pwm.n_mutations", pssm=pssm, score_thresh=-1, extend=na)


@pytest.mark.parametrize("na", [np.nan, pd.NA])
def test_an_na_extend_of_gseq_pwm_and_gseq_kmer_is_an_error(na):
    pssm = _df([[1, 0, 0, 0], [0, 1, 0, 0]])
    seq = "GGGGGGGACGGGGG"
    with pytest.raises(ValueError, match="extend must be"):
        pm.gseq_pwm(seq, pssm, mode="max", start_pos=9, end_pos=10, bidirect=False, strand=1, extend=na)
    with pytest.raises(ValueError, match="extend must be"):
        pm.gseq_kmer(seq, "AC", start_pos=10, end_pos=11, extend=na)
    np.testing.assert_array_equal(pm.gseq_kmer(seq, "AC", start_pos=10, end_pos=11, extend=True),
                                  pm.gseq_kmer(seq, "AC", start_pos=10, end_pos=11, extend=1))


def test_gseq_pwm_edits_refuses_a_bidirect_that_is_not_true_or_false():
    pssm = _df([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    for b in (np.nan, pd.NA, [True, False], [], "TRUE", 1, None):
        with pytest.raises(ValueError, match="bidirect must be True or False"):
            pm.gseq_pwm_edits("GTGT", pssm, score_thresh=-1, prior=0, bidirect=b)


# ---------------------------------------------------------------------------
# Motif ID de-duplication (test-motif-import.R)
# ---------------------------------------------------------------------------

def test_deduplicated_motif_ids_do_not_collide_with_the_other_ids(tmp_path):
    f = tmp_path / "dup.motif"
    f.write_text("\n".join([
        ">ACGT\tm1\t5.0", "0.97\t0.01\t0.01\t0.01",
        ">ACGT\tm2\t5.0", "0.01\t0.97\t0.01\t0.01",
        ">ACGT.1\tm3\t5.0", "0.01\t0.01\t0.97\t0.01",
    ]) + "\n")
    with pytest.warns(UserWarning, match="Duplicate motif IDs"):
        motifs = pm.gseq_read_homer(str(f))
    assert list(motifs) == ["ACGT.2", "ACGT.3", "ACGT.1"]
    # the motif read as ACGT.1 is the one with that ID in the file, and each copy of
    # the repeated ID keeps its own matrix
    np.testing.assert_array_equal(motifs["ACGT.1"].iloc[0].to_numpy(), [0.01, 0.01, 0.97, 0.01])
    np.testing.assert_array_equal(motifs["ACGT.2"].iloc[0].to_numpy(), [0.97, 0.01, 0.01, 0.01])
    np.testing.assert_array_equal(motifs["ACGT.3"].iloc[0].to_numpy(), [0.01, 0.97, 0.01, 0.01])

    with pytest.warns(UserWarning, match="Duplicate motif IDs"):
        ids = _dedup_motif_ids(["A", "A", "A.1", "A.1", "B"])
    assert len(set(ids)) == len(ids)
    assert ids[4] == "B"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _dedup_motif_ids(["x", "y"]) == ["x", "y"]


# ---------------------------------------------------------------------------
# gseq_pwm_edits (test-gseq-pwm-edits.R, test-gseq-pwm-edits-below.R)
# ---------------------------------------------------------------------------

def _random_intervals(rng, n, lo, hi, lengths, step=1):
    starts = np.sort(rng.choice(np.arange(lo, hi + 1, step), n, replace=False))
    return pm.gintervals("1", starts, starts + rng.choice(lengths, n))


def test_gseq_pwm_edits_ignores_a_strand_column_as_the_vtrack_does():
    pssm = _df([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]])
    thr = -3
    rng = np.random.default_rng(SEED)
    iv = _random_intervals(rng, 12, 1000, 400000, np.arange(1, 7))
    stranded = iv.copy()
    stranded["strand"] = -1

    # A strand = -1 row used to be scanned reverse-complemented, and with indels over
    # windows shifted by max_indels
    for cfg in ({"bidirect": False, "strand": 1, "max_indels": None},
                {"bidirect": False, "strand": -1, "max_indels": None},
                {"bidirect": True, "strand": 1, "max_indels": 1},
                {"bidirect": False, "strand": 1, "max_indels": 2}):
        args = {"score_thresh": thr, "prior": 0.01, "bidirect": cfg["bidirect"], "strand": cfg["strand"]}
        if cfg["max_indels"] is not None:
            args["max_indels"] = cfg["max_indels"]
        n_plain = _n_edits_per_seq(pm.gseq_pwm_edits(iv, pssm, **args), len(iv))
        n_strand = _n_edits_per_seq(pm.gseq_pwm_edits(stranded, pssm, **args), len(iv))
        np.testing.assert_array_equal(n_strand, n_plain)

        pm.gvtrack_create("ed_strandcol", None, "pwm.edit_distance", params={"pssm": pssm, **args})
        v = pm.gextract("ed_strandcol", stranded, iterator=stranded)
        np.testing.assert_array_equal(n_strand, _by_interval(v, "ed_strandcol", len(iv)))
        pm.gvtrack_rm("ed_strandcol")


def test_gseq_pwm_edits_on_intervals_scans_the_vtracks_windows_for_every_extend():
    # With extend=False and max_indels > 0 windows used to start up to
    # length - (L - max_indels), past the vtrack's last start (length - L), and an
    # integer extend was read as a number of extra bases where the vtrack reads any
    # positive value as True.
    rng = np.random.default_rng(SEED)
    pssm = _gamma_pssm(rng, 8, 0.4)
    iv = _random_intervals(rng, 120, 1000, 150000, np.arange(5, 41), step=50)
    for ext in (True, False, 3, 0):
        for d in range(4):
            for direction in ("above", "below"):
                thr = -8 if direction == "above" else -14
                va = {"pssm": pssm, "score_thresh": thr, "extend": ext, "direction": direction}
                if d > 0:
                    va["max_indels"] = d
                pm.gvtrack_create("ed_ext", None, "pwm.edit_distance", params=va)
                v = _by_interval(pm.gextract("ed_ext", iv, iterator=iv), "ed_ext", len(iv))
                g = pm.gseq_pwm_edits(iv, pssm, score_thresh=thr, max_indels=d if d > 0 else None,
                                      extend=ext, direction=direction)
                np.testing.assert_array_equal(_n_edits_per_seq(g, len(iv)), v,
                                              err_msg=f"extend={ext} D={d} {direction}")
                pm.gvtrack_rm("ed_ext")


def test_gseq_pwm_edits_below_bidirect_reports_the_harder_strand():
    pssm = _df([[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]])
    # "GT" forward scores ~-9.21 (already below -1.0 -> 0 edits), its reverse "AC"
    # ~-0.06 (needs edits). A genomic substitution changes both readings and both must
    # end up below, so, as in the pwm.edit_distance vtrack, the harder strand decides:
    # the reverse strand with 1 edit, and the edit is on its reading.
    res = pm.gseq_pwm_edits("GT", pssm, score_thresh=-1.0, score_min=-np.inf,
                            prior=0, bidirect=True, direction="below")
    assert len(res) == 1
    assert res["n_edits"].iloc[0] == 1
    assert res["strand"].iloc[0] == -1
    assert res["window_seq"].iloc[0] == "AC"
    assert res["score_after"].iloc[0] <= -1.0

    # "AC": forward scores ~-0.06 and its reverse "GT" ~-9.21, both already at or
    # below -0.01, so 0 edits (a tie goes to the forward strand)
    res2 = pm.gseq_pwm_edits("AC", pssm, score_thresh=-0.01, score_min=-np.inf,
                             prior=0, bidirect=True, direction="below")
    assert len(res2) == 1
    assert res2["n_edits"].iloc[0] == 0
    assert res2["strand"].iloc[0] == 1


def test_gseq_pwm_edits_below_bidirect_matches_the_pwm_edit_distance_vtrack():
    rng = np.random.default_rng(SEED)
    iv = _random_intervals(rng, 15, 1000, 400000, np.arange(1, 9))
    for k in range(1, 7):
        L = int(rng.integers(4, 7))
        pssm = _gamma_pssm(rng, L, 0.4)
        lp = _log_probs(pssm, 0.01)
        smax, smin = lp.max(axis=1).sum(), lp.min(axis=1).sum()
        thr = smin + rng.uniform(0.3, 0.7) * (smax - smin)
        args = {"score_thresh": thr, "prior": 0.01, "bidirect": True, "direction": "below"}
        if k % 3 == 1:
            args["max_edits"] = 1
        if k % 2 == 0:
            args["max_indels"] = 1
        if k == 5:
            args["score_min"] = smin + 0.5 * (smax - smin)
        pm.gvtrack_create("ed_below_bidi", None, "pwm.edit_distance", params={"pssm": pssm, **args})
        v = _by_interval(pm.gextract("ed_below_bidi", iv, iterator=iv), "ed_below_bidi", len(iv))
        g = pm.gseq_pwm_edits(iv, pssm, **args)
        np.testing.assert_array_equal(_n_edits_per_seq(g, len(iv)), v, err_msg=f"case {k}")
        pm.gvtrack_rm("ed_below_bidi")


# ---------------------------------------------------------------------------
# Edit distance with indels (test-pwm-edit-distance.R)
# ---------------------------------------------------------------------------

_ACGT_PSSM = _df([[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01],
                  [0.01, 0.01, 0.97, 0.01], [0.01, 0.01, 0.01, 0.97]])


def test_deletion_improves_alignment_one_indel_solver():
    # "ACAGT" is "ACGT" with an extra A inserted after C; the threshold is the
    # perfect-match score. ("ATCGT" used to be the sequence, but there one
    # substitution, TCGT -> ACGT, is just as good as the deletion.)
    seq = "ACAGT"
    threshold = float(np.sum(np.log([0.97] * 4)))
    no_indel = pm.gseq_pwm_edits(seq, _ACGT_PSSM, score_thresh=threshold, max_indels=0,
                                 prior=0, bidirect=False)
    assert len(no_indel) == 0
    with_indel = pm.gseq_pwm_edits(seq, _ACGT_PSSM, score_thresh=threshold, max_indels=1,
                                   prior=0, bidirect=False)
    assert len(with_indel) > 0
    assert with_indel["n_edits"].min() == 1
    edit_rows = with_indel[with_indel["edit_num"] > 0]
    assert (edit_rows["edit_type"] == "del").any()


def test_score_filter_applies_with_indels():
    full_seq = pm.gseq_extract(pm.gintervals("1", 0, 5000))[0].upper()
    pos = full_seq.find("ACGT")
    assert pos >= 0
    iv = pm.gintervals("1", pos, pos + 12)
    test_seq = pm.gseq_extract(iv)[0].upper()
    threshold = -3.0
    lp = np.log(_ACGT_PSSM.to_numpy())
    scores = [sum(lp[i, BASES.index(c)] for i, c in enumerate(test_seq[s:s + 4]))
              for s in range(len(test_seq) - 3)]
    strict = max(scores) + 0.5

    common = {"pssm": _ACGT_PSSM, "score_thresh": threshold, "bidirect": False, "extend": False, "prior": 0}
    pm.gvtrack_create("edist_no_indel_filt", None, "pwm.edit_distance", score_min=strict,
                      max_indels=0, **common)
    # With indels score_min filters on the same ungapped window score (it used to be
    # ignored silently when max_indels > 0)
    pm.gvtrack_create("edist_indel_filt", None, "pwm.edit_distance", score_min=strict,
                      max_indels=1, **common)
    r = pm.gextract(["edist_no_indel_filt", "edist_indel_filt"], iv, iterator=iv)
    assert np.isnan(r["edist_no_indel_filt"].iloc[0])
    assert np.isnan(r["edist_indel_filt"].iloc[0])

    # A score_min every window passes leaves the result unfiltered
    pm.gvtrack_create("edist_indel_nofilt", None, "pwm.edit_distance", score_min=-np.inf,
                      max_indels=1, **common)
    pm.gvtrack_create("edist_indel_plain", None, "pwm.edit_distance", max_indels=1, **common)
    r2 = pm.gextract(["edist_indel_nofilt", "edist_indel_plain"], iv, iterator=iv)
    assert not np.isnan(r2["edist_indel_plain"].iloc[0])
    assert r2["edist_indel_nofilt"].iloc[0] == r2["edist_indel_plain"].iloc[0]


def test_edit_distance_positions_past_2_24_are_exact():
    # Positions were kept in float, which holds consecutive integers only up to 2^24.
    # Reaching that needs an iterator interval longer than 16.7 Mb, so this runs only
    # where a real assembly is at hand.
    big = os.environ.get("MISHA_LARGE_GENOME", "")
    if not big or not os.path.isdir(big):
        pytest.skip("set MISHA_LARGE_GENOME to a groot with a chromosome over 16.7 Mb")
    pm.gdb_init(big)
    ci = pm.gintervals_all()
    ci = ci[ci["end"] > 16777218 + 1000]
    if len(ci) == 0:
        pytest.skip("no chromosome over 16.7 Mb in this genome")
    chrom = str(ci["chrom"].iloc[0])
    L = 16

    def one_hot(word):
        m = np.zeros((L, 4))
        m[np.arange(L), [BASES.index(c) for c in word]] = 1
        return _df(m)

    # 1-based positions must be ODD: floats represent only even integers between 2^24
    # and 2^25. off is 0-based, so it is even.
    found = False
    for off in 16777218 + 2 * np.array([50, 177, 301, 452, 613]):
        off = int(off)
        word = pm.gseq_extract(pm.gintervals(chrom, off, off + L))[0].upper()
        if any(c not in BASES for c in word):
            continue
        pssm = one_hot(word)
        # the planted window must be the first exact match: none before it
        pm.gvtrack_create("big_ed", None, "pwm.edit_distance", pssm=pssm, score_thresh=0,
                          prior=0, bidirect=False)
        before_iv = pm.gintervals(chrom, 0, off)
        before = pm.gextract("big_ed", before_iv, iterator=before_iv)["big_ed"].iloc[0]
        pm.gvtrack_rm("big_ed")
        if before > 0:
            found = True
            break
    if not found:
        pytest.skip("no first-occurrence 16-mer past 2^24 in this genome")

    pm.gvtrack_create("big_edpos", None, "pwm.edit_distance.pos", pssm=pssm, score_thresh=0,
                      prior=0, bidirect=False)
    iv = pm.gintervals(chrom, 0, off + 1)
    assert pm.gextract("big_edpos", iv, iterator=iv)["big_edpos"].iloc[0] == off + 1

    # LSE: a PSSM one substitution away from the window at `off` (column 2), so the one
    # edit that reaches the threshold is at 1-based off + 3 (odd).
    j = 2
    alt = [b for b in BASES if b != word[j]][0]
    lse_pssm = one_hot(word[:j] + alt + word[j + 1:])
    pm.gvtrack_create("big_near", None, "pwm.edit_distance", pssm=lse_pssm, score_thresh=0,
                      prior=0, bidirect=False)
    near = pm.gextract("big_near", before_iv, iterator=before_iv)["big_near"].iloc[0]
    if not near >= 2:
        pytest.skip("the near-match window is not unique")
    pm.gvtrack_create("big_lsepos", None, "pwm.edit_distance.lse.pos", pssm=lse_pssm,
                      score_thresh=-1, prior=0.01, bidirect=False)
    assert pm.gextract("big_lsepos", iv, iterator=iv)["big_lsepos"].iloc[0] == off + j + 1


# ---------------------------------------------------------------------------
# LSE edit distance: beam search for 3+ edits (test-pwm-edit-distance-below.R)
# ---------------------------------------------------------------------------

def test_lse_below_reachable_threshold_past_a_greedy_local_optimum(tmp_path):
    # mm10 chr19:15,466,149-15,466,159: the greedy search for 3+ edits stopped when no
    # single substitution lowered the LSE and returned NaN although 4 substitutions
    # reach the threshold.
    seq = "AAGTACTATA"
    _tiny_genome(tmp_path, seq)
    vals = [0.4233, 0.0003, 0.0537, 0.1266, 0.0359035903590359, 0.0292,
            0.5043, 0.0365, 0.7398, 0.0016, 0.0042004200420042, 0.3129,
            0.002, 0.8887, 0.0312, 0.5199, 0.694369436943694, 0.0326,
            0.0704, 0.0745, 0.1753, 0.3519, 0.265526552655266, 0.6253]
    pssm = _df(np.array(vals).reshape(4, 6).T)  # R's column-major dim = c(6, 4)
    iv = pm.gintervals("chr1", 0, 5)
    kw = {"pssm": pssm, "score_thresh": -11.2, "direction": "below", "bidirect": True, "prior": 0.01}
    pm.gvtrack_create("lse_ed", None, "pwm.edit_distance.lse", **kw)
    pm.gvtrack_create("lse_pos", None, "pwm.edit_distance.lse.pos", **kw)
    r = pm.gextract(["lse_ed", "lse_pos"], iv, iterator=iv)
    assert lse_exact_min_edits(seq, pssm, 5, -11.2, "below") == 4
    assert r["lse_ed"].iloc[0] == 4
    assert not np.isnan(r["lse_pos"].iloc[0])


# R's 24 cases (set.seed(4117) in test-pwm-edit-distance-below.R): the sequence and the
# 4 x 4 PSSM. The search for 3+ edits is a heuristic, so the check needs R's inputs:
# on other random inputs it can exceed the minimum (as R's own does).
_LSE_CASES = [
    ("AGCTCTGA", [
        [0.06861329058090282, 0.6096116422334128, 0.014936514729260521, 0.30683855245642383],
        [0.0007640983929807138, 0.07913228674151014, 0.9173590994595182, 0.002744515405990891],
        [0.6311098779126959, 0.3039583566891442, 0.06493165255477239, 1.1284338745231638e-07],
        [0.8081633172320472, 0.006406197639009221, 0.18488840654545094, 0.0005420785834926381],
    ]),
    ("TTTCCGCT", [
        [0.451077195590504, 1.0242242267386832e-05, 0.002014528734258441, 0.54689803343297],
        [0.054580671394618065, 0.8412670137083751, 0.00853952203425565, 0.09561279286275115],
        [0.32668668206757656, 0.008556015030235615, 0.5564882637720383, 0.10826903913014944],
        [0.04207151079688871, 0.9462266078837046, 0.005755974214593975, 0.005945907104812724],
    ]),
    ("GAGGCAAC", [
        [0.039667687796146656, 0.45641977192315997, 0.005506063298965831, 0.4984064769817276],
        [0.3527753587280645, 0.14017501453588888, 0.5056172646298903, 0.001432362106156383],
        [0.4556446087752217, 0.39862441475907096, 0.13890350769915386, 0.00682746876655345],
        [0.002681439052166201, 0.47131256090889295, 0.001515617202916303, 0.5244903828360246],
    ]),
    ("CACAGTCT", [
        [0.20837724183295672, 0.2750397932518776, 0.49067449348756387, 0.025908471427601765],
        [0.08405746820134827, 0.17849901142709448, 0.5789592777563809, 0.15848424261517638],
        [0.1053309316384735, 0.32476632219928603, 0.4399515051166389, 0.12995124104560155],
        [0.1800536237407322, 0.03557528830659756, 0.6688950718013459, 0.11547601615132444],
    ]),
    ("ACTCGAAT", [
        [0.39774257389347173, 0.09974994071719695, 0.2346445653559173, 0.26786292003341405],
        [0.10829340164877316, 0.00042098786380559183, 0.011278499826252944, 0.8800071106611683],
        [0.0022636862357603973, 0.2895818439838266, 0.1185730936073331, 0.5895813761730798],
        [0.24181409465491105, 0.027969905188300757, 0.5414607143726515, 0.18875528578413664],
    ]),
    ("GGCCAACC", [
        [0.7196277149370842, 0.0035542535167558636, 0.22975354198332693, 0.047064489562833],
        [0.12832230006534603, 0.5243331147796684, 0.194443596002256, 0.1529009891527296],
        [0.22271028093480133, 0.010792226926535486, 0.03400960362531097, 0.7324878885133521],
        [0.4433238579963257, 0.5248082996408058, 0.01037443635951465, 0.021493406003353914],
    ]),
    ("TACCGCCA", [
        [0.32543195278855863, 0.06035228970078477, 0.05466209063727616, 0.5595536668733805],
        [0.003791657235157983, 0.1362405375749681, 0.10523538069895665, 0.7547324244909173],
        [0.3125777428216432, 0.0002705490526544413, 0.6867761958288855, 0.0003755122968168702],
        [0.02125892015538636, 0.01411311189344535, 0.06192509218650047, 0.9027028757646678],
    ]),
    ("CCCACTCA", [
        [0.11779070462260845, 0.04186040562188443, 0.49206282709685184, 0.34828606265865536],
        [0.007494816820533382, 0.553511803221169, 0.046932635495397994, 0.3920607444628996],
        [0.29862501360561544, 0.17902423093631384, 0.2992177813061043, 0.22313297415196642],
        [0.5213833518180153, 0.13321881534619084, 0.34518898877394494, 0.0002088440618488406],
    ]),
    ("TCACTAAC", [
        [0.2574807337697786, 0.0059343503258613055, 0.0019176249626559755, 0.7346672909417041],
        [0.03300022687076128, 0.04866885032080559, 0.9181158702648646, 0.00021505254356853098],
        [0.014438459772437084, 0.000852578727862516, 0.9782148478499976, 0.006494113649702789],
        [0.02109556099197006, 0.14611599804620679, 0.4122307493996445, 0.4205576915621787],
    ]),
    ("GCGTTTTT", [
        [0.3804160297500695, 0.4749144518667449, 0.08200903389450942, 0.06266048448867614],
        [0.7660117409331747, 0.20743395476765067, 0.024747029011187333, 0.0018072752879873507],
        [0.004748511327044522, 0.09121578281313356, 0.442521082397306, 0.461514623462516],
        [0.5655424777791077, 0.22203261909352084, 0.1792206717896558, 0.03320423133771559],
    ]),
    ("GACGTCCT", [
        [0.20845391898020582, 0.1784600015581184, 0.023535745758218193, 0.5895503337034576],
        [1.5195485356054241e-05, 0.9527241581289562, 2.1060389534315067e-05, 0.047239585996153396],
        [0.0075873951028615815, 0.6999050613699611, 0.04127267895209978, 0.2512348645750774],
        [0.047989923443668106, 0.0024308588722251547, 0.6937708280347961, 0.25580838964931063],
    ]),
    ("CAAGTGTT", [
        [0.026913027640454213, 0.16218923685642925, 0.7776697446917215, 0.03322799081139508],
        [0.025006255018393632, 0.4714170895170154, 0.2757902510675475, 0.2277864043970435],
        [1.899881132231151e-05, 0.014437834838160055, 0.001778458689581247, 0.9837647076609364],
        [0.9347888808636946, 8.205035652667949e-06, 0.006179824825030439, 0.059023089275622366],
    ]),
    ("TGATCAGC", [
        [0.23560326455340658, 0.7306985508313865, 0.026629100661090407, 0.007069083954116591],
        [0.24477422960154988, 0.28034833288535904, 0.2529350158283631, 0.22194242168472803],
        [0.0078229596351065, 0.036434173813480034, 0.777329196126761, 0.1784136704246525],
        [0.01337238776811065, 0.594456532270661, 0.3803809782143866, 0.01179010174684174],
    ]),
    ("TCGATCAT", [
        [0.1286283484588416, 0.5273471540689544, 0.31822303536826274, 0.025801462103941245],
        [0.13351254335866414, 0.4736448534996886, 0.01377726668610414, 0.37906533645554313],
        [0.49786196431442425, 0.1932220288291256, 0.031735110644958056, 0.2771808962114921],
        [0.8781640720255524, 0.022661005616543377, 0.009636134460095097, 0.08953878789780928],
    ]),
    ("ATGTACCC", [
        [0.21264180719551917, 0.31145735166331595, 0.4736778507152232, 0.0022229904259416975],
        [0.01614384032316776, 0.1618165164883754, 0.4088580668184385, 0.41318157637001823],
        [0.946885696775371, 0.036139344347509546, 0.0007160299309260101, 0.01625892894619334],
        [0.06477462334091838, 0.19463180274946038, 0.6938887048702309, 0.046704869039390265],
    ]),
    ("ATACGCCC", [
        [0.10330556520352085, 0.05315355580293965, 0.8261119973596778, 0.017428881633861724],
        [0.016385736960538844, 0.21509012084286885, 0.023047553598123587, 0.7454765885984688],
        [0.3635837854124219, 0.5606885777436512, 0.058841843828095104, 0.016885793015831783],
        [0.08274063948727665, 0.623755754957649, 6.36618649332756e-05, 0.29343994369014104],
    ]),
    ("GTTTGAAG", [
        [0.6064971444197695, 0.19148359542546814, 0.20122235652068066, 0.0007969036340816921],
        [0.016070755708407703, 0.019860127583631506, 0.9570181183520333, 0.007050998355927467],
        [0.8242459075102859, 0.034600578883272406, 0.13585952893858363, 0.00529398466785808],
        [0.6044841078420187, 0.0006010929652183213, 1.4385880441942767e-06, 0.39491336060471877],
    ]),
    ("TTTCGCTT", [
        [0.03128180549904394, 0.14714970082742937, 0.1311641151305374, 0.6904043785429893],
        [0.10839089854841645, 0.8250192478398191, 0.0052380619840404, 0.061351791627724146],
        [0.31115067529589996, 0.2433903114318083, 0.00028232694145666036, 0.44517668633083507],
        [0.013397526712737121, 0.012907682592944399, 0.0037664937406139683, 0.9699282969537044],
    ]),
    ("CAGCCCGG", [
        [0.047222913541561294, 0.1681257058780362, 0.7697673222531457, 0.014884058327256838],
        [0.36637001599418456, 0.5989462097695389, 0.033586824159920675, 0.0010969500763558235],
        [0.25265643388875997, 0.10809715843827719, 0.6289411070381127, 0.010305300634850212],
        [0.052534840535169464, 0.2269525530155214, 0.38459230796145627, 0.33592029848785276],
    ]),
    ("AAACTCTG", [
        [0.0581051063249544, 0.8664143115037891, 0.0026828081141287044, 0.07279777405712787],
        [0.10197394429572351, 0.016585268269981537, 0.203813986576332, 0.677626800857963],
        [0.06245679059359249, 0.214754502752993, 0.15893450492243116, 0.5638542017309832],
        [1.2534014773543167e-06, 0.7369856377205597, 0.01680459356532148, 0.24620851531264143],
    ]),
    ("ATGCCAGG", [
        [0.11954962864318021, 0.7418921950840227, 0.030933216267523912, 0.10762496000527315],
        [0.0003183684773410654, 0.08139637026082806, 0.8014085393849218, 0.11687672187690909],
        [0.1971509037886634, 0.22533441301298315, 0.0018898693434640614, 0.5756248138548895],
        [0.6425697245429378, 0.21138353895625686, 0.03339560037262885, 0.11265113612817648],
    ]),
    ("ACGCGAGG", [
        [0.003908255545302942, 0.5578716981316089, 0.056357444192889974, 0.3818626021301983],
        [0.41631627491472856, 0.00017559657302013406, 0.34512128986269147, 0.2383868386495599],
        [0.012660177408943223, 0.4759093613746645, 0.2022053823184551, 0.3092250788979371],
        [0.010376015919133074, 0.5837079376595712, 0.015661045909496366, 0.3902550005117994],
    ]),
    ("TATTAACG", [
        [0.02451503985122885, 0.014333531447470748, 0.9478656035952473, 0.013285825106053153],
        [0.2777956339417469, 0.021033917781957234, 0.19112228284242458, 0.5100481654338712],
        [0.38769674836205864, 0.3186458761551557, 0.042553973255187914, 0.2511034022275978],
        [0.04757710556267404, 0.34118098998776614, 0.0015252919283067232, 0.6097166125212531],
    ]),
    ("GTGGTGAG", [
        [0.8774275381988343, 0.10355517949459413, 0.009547692158331494, 0.009469590148240014],
        [0.0014452069369119435, 0.8493829115190314, 0.001180742054725619, 0.14799113948933107],
        [0.06035976529687288, 0.16600568852278516, 0.008766061213907371, 0.7648684849664346],
        [0.34552466044647867, 0.3454014971626474, 0.0005009514577282108, 0.3085728909331457],
    ]),
]


def test_lse_matches_an_exhaustive_search_on_short_regions(tmp_path):
    for k, (seq, rows) in enumerate(_LSE_CASES, start=1):
        L = 4
        pssm = _df(rows)
        n_starts = 8 - L + 1
        direction = "above" if k % 2 else "below"
        db = tmp_path / f"lse{k}"
        db.mkdir()
        _tiny_genome(db, seq)
        iv = pm.gintervals("chr1", 0, n_starts)
        pm.gvtrack_create("lse_f", None, "pwm", pssm=pssm, bidirect=True, prior=0.01)
        f0 = pm.gextract("lse_f", iv, iterator=iv)["lse_f"].iloc[0]
        # a threshold three to four log units away: usually 3+ substitutions
        thr = f0 + 3.5 if direction == "above" else f0 - 3.5
        pm.gvtrack_create("lse_e", None, "pwm.edit_distance.lse", pssm=pssm, score_thresh=thr,
                          direction=direction, bidirect=True, prior=0.01)
        got = pm.gextract("lse_e", iv, iterator=iv)["lse_e"].iloc[0]
        expected = lse_exact_min_edits(seq, pssm, n_starts, thr, direction)
        np.testing.assert_array_equal(got, expected, err_msg=f"case {k}: {seq} {direction} thr {thr:.3f}")
        pm.gvtrack_rm("lse_f")
        pm.gvtrack_rm("lse_e")


# ---------------------------------------------------------------------------
# Indels: inserted columns score the inserted base; exact search
# (test-pwm-edit-distance-adversarial.R)
# ---------------------------------------------------------------------------

def test_indels_a_threshold_above_the_best_possible_score_is_unreachable():
    # An inserted column used to add 0 instead of the inserted base's score, so a
    # threshold above the sum of the column maxima was "reachable" with indels.
    pssm = _df([[.97, .01, .01, .01], [.97, .01, .01, .01], [.25, .25, .25, .25], [.97, .01, .01, .01]])
    thr = float(np.log(pssm.to_numpy().max(axis=1)).sum()) + 0.5
    iv = pm.gintervals("1", [0, 1000, 5000], [200, 1200, 5200])
    for d in (1, 2, 3):
        pm.gvtrack_create(f"unreach_ed{d}", None, "pwm.edit_distance", pssm=pssm, score_thresh=thr,
                          max_indels=d, prior=0, bidirect=False)
    r = pm.gextract([f"unreach_ed{d}" for d in (1, 2, 3)], iv, iterator=iv)
    assert np.isnan(r[[f"unreach_ed{d}" for d in (1, 2, 3)]].to_numpy(dtype=float)).all()
    for d in (1, 2, 3):
        res = pm.gseq_pwm_edits("TTTAAATTT", pssm, score_thresh=thr, prior=0, bidirect=False, max_indels=d)
        assert len(res) == 0


def test_indels_gseq_pwm_edits_reports_edits_whose_result_scores_score_after():
    rng = np.random.default_rng(SEED)
    for _ in range(12):
        L = int(rng.integers(4, 7))
        pssm = _gamma_pssm(rng, L, 0.5)
        lp = _log_probs(pssm, 0.01)
        smax, smin = lp.max(axis=1).sum(), lp.min(axis=1).sum()
        for direction in ("above", "below"):
            thr = smax - 0.25 * (smax - smin) if direction == "above" else smin + 0.25 * (smax - smin)
            seqs = ["".join(rng.choice(list(BASES), L + int(rng.integers(-1, 3)))) for _ in range(8)]
            for d in (1, 2):
                res = pm.gseq_pwm_edits(seqs, pssm, score_thresh=thr, max_indels=d, prior=0.01,
                                        bidirect=False, direction=direction)
                res = res.drop_duplicates("seq_idx")
                for _, row in res.iterrows():
                    mutated = row["mutated_seq"].replace("-", "")
                    assert len(mutated) == L
                    own = sum(lp[i, BASES.index(c)] for i, c in enumerate(mutated))
                    assert row["score_after"] == pytest.approx(own, abs=1e-4)
                    if direction == "above":
                        assert row["score_after"] >= thr - 1e-4
                    else:
                        assert row["score_after"] <= thr + 1e-4


def test_indels_gseq_pwm_edits_matches_an_exhaustive_alignment_search():
    rng = np.random.default_rng(SEED)
    mismatches = []
    for _ in range(10):
        L = int(rng.integers(4, 6))
        pssm = _gamma_pssm(rng, L, 0.5)
        lp = _log_probs(pssm, 0.01)
        smax, smin = lp.max(axis=1).sum(), lp.min(axis=1).sum()
        for direction in ("above", "below"):
            thr = smax - 0.3 * (smax - smin) if direction == "above" else smin + 0.3 * (smax - smin)
            for _ in range(4):
                s = "".join(rng.choice(list(BASES), L + int(rng.integers(0, 3))))
                for d in (1, 2, 3):
                    res = pm.gseq_pwm_edits(s, pssm, score_thresh=thr, max_indels=d, prior=0.01,
                                            bidirect=False, direction=direction)
                    got = float(res["n_edits"].min()) if len(res) else np.nan
                    ref = bruteforce_edit_distance_with_indels(s, pssm, thr, max_indels=d, prior=0.01,
                                                               direction=direction)
                    if not (got == ref or (np.isnan(got) and np.isnan(ref))):
                        mismatches.append(f"{s} {direction} D={d}: C++={got} ref={ref}")
    assert mismatches == []


# R's 7 x 4 PSSM of the chr19 cases, in R's column-major order (dim = c(7, 4))
_CHR19_PSSM = _df(np.array([
    0.289059949638304, 0.765800540168912, 0.00347585269750313,
    0.643266951333656, 0.166971128351294, 0.392771351852197, 0.956813518658636,
    0.00720825395649754, 0.125848970554678, 0.185711579486119, 0.245713777963034,
    0.620780527463697, 0.235169567229761, 0.00186895219631772, 0.131171669840629,
    0.00289069135768031, 0.650092955874321, 0.0102013770101287, 0.00149015116496921,
    0.333079810960369, 0.0329072533347471, 0.572560126564569, 0.10545979791873,
    0.160719611942056, 0.100817893693182, 0.21075819302004, 0.038979269957673,
    0.00841027581029925,
]).reshape(4, 7).T)
_CHR19_THR = -27.05969
_CHR19_CORE = "CTATATGATTTTCTTT"


def test_indels_max_indels_3_is_exact(tmp_path):
    # The generic DP for three or more indels kept only the best-scoring alignment per
    # window length and indel count, then added substitutions; an alignment scoring
    # lower can need fewer. On mm10 chr19 at 22,639,521 this gave 6 edits with
    # max_indels = 3 and 5 with 2.
    seq = _CHR19_CORE
    _tiny_genome(tmp_path, seq)
    for d in (2, 3):
        for st, tag in ((1, "f"), (-1, "r")):
            pm.gvtrack_create(f"ed{d}_{tag}", None, "pwm.edit_distance", pssm=_CHR19_PSSM,
                              score_thresh=_CHR19_THR, prior=0.01, bidirect=False, strand=st,
                              direction="below", max_indels=d)
    vt = ["ed2_f", "ed2_r", "ed3_f", "ed3_r"]
    starts = np.arange(0, len(seq) - 7 + 1)
    r = pm.gextract(vt, pm.gintervals("chr1", starts, starts + 1), iterator=1).sort_values("start")
    for _, row in r.iterrows():
        p = int(row["start"]) + 1
        sub = seq[p - 1:]
        for d in (2, 3):
            ref = bruteforce_edit_distance_with_indels(sub, _CHR19_PSSM, _CHR19_THR, max_indels=d, prior=0.01,
                                                       direction="below", starts=[1])
            assert row[f"ed{d}_f"] == ref, f"start {row['start']} D={d} fwd"
        assert row["ed3_f"] <= row["ed2_f"]
        assert row["ed3_r"] <= row["ed2_r"]


def test_indels_a_window_shorter_than_the_motif_at_a_chromosome_end_is_scanned(tmp_path):
    # With insertions a window may hold fewer than L bases. Before an N run such a
    # window was scanned; at the chromosome end the scan stopped at the last
    # full-length window, so the last starts were NaN.
    values = []
    for k, genome in enumerate((_CHR19_CORE, _CHR19_CORE + "N" * 10)):
        db = tmp_path / f"g{k}"
        db.mkdir()
        _tiny_genome(db, genome)
        for d in range(4):
            pm.gvtrack_create(f"ce{d}", None, "pwm.edit_distance", pssm=_CHR19_PSSM, score_thresh=_CHR19_THR,
                              prior=0.01, bidirect=False, direction="below", max_indels=d)
        iv = pm.gintervals("chr1", np.arange(6, 13), np.arange(7, 14))
        r = pm.gextract([f"ce{d}" for d in range(4)], iv, iterator=1).sort_values("start")
        values.append(r[[f"ce{d}" for d in range(4)]].to_numpy(dtype=float))

        for d in (1, 2, 3):
            g = pm.gseq_pwm_edits(iv, _CHR19_PSSM, score_thresh=_CHR19_THR, prior=0.01, bidirect=False,
                                  direction="below", max_indels=d)
            np.testing.assert_array_equal(_n_edits_per_seq(g, len(iv)), r[f"ce{d}"].to_numpy(),
                                          err_msg=f"gseq_pwm_edits D={d}")
        for d in range(4):
            pm.gvtrack_rm(f"ce{d}")
    # the same at the chromosome end as before the N run
    np.testing.assert_array_equal(values[0], values[1])
    r = values[0]
    # without indels nothing past the last full window
    assert np.isnan(r[4:7, 0]).all()
    # starts 10 and 11 (rows 5, 6): 5 edits with 2 or 3 indels, as an exhaustive search says
    for k in range(5, 8):
        sub = _CHR19_CORE[k + 5:]
        for d in (1, 2, 3):
            ref = bruteforce_edit_distance_with_indels(sub, _CHR19_PSSM, _CHR19_THR, max_indels=d, prior=0.01,
                                                       direction="below", starts=[1])
            np.testing.assert_array_equal(r[k - 1, d], ref, err_msg=f"start {k + 5} D={d}")
    np.testing.assert_array_equal(r[4:6, 2], [5, 5])
    np.testing.assert_array_equal(r[4:6, 3], [5, 5])


# ---------------------------------------------------------------------------
# Potts kernels (test-potts.R, test-potts-vtrack.R)
# ---------------------------------------------------------------------------

def _one_strand_params(m):
    return _potts_params(m, bidirect=False, extend=False, strand=1, score_thresh=0.0, what="best_delta test")


@pytest.mark.parametrize(("w", "npair_mode"), [
    (w, mode) for w in (1, 2, 7, 20, 21, 45) for mode in (("none",) if w == 1 else ("full", "sparse", "none"))
])
def test_best_delta_is_the_largest_delta_codes_bit_for_bit(w, npair_mode):
    # potts.max with max_edits = 1 on one strand adds PottsModel::best_delta() to the
    # anchor's score; it must be the largest delta_codes() over every position and base
    # bit for bit, and score_codes4() must equal score_codes() on four anchors.
    m = potts_ref_model(w, npair_mode=npair_mode, seed=7)
    codes = np.random.default_rng(8).integers(0, 4, size=(5000, w), dtype=np.int32)
    res = _pymisha.pm_potts_score_codes_cmp(_one_strand_params(m), codes)
    np.testing.assert_array_equal(res["best_delta"], res["delta_max"])
    assert np.isfinite(res["best_delta"]).all()
    assert res["score4_mismatch"] == 0


def test_best_delta_and_score_codes4_agree_on_a_model_the_naive_kernel_scores():
    # 100 couplings at W = 60, each in a block pair of its own: the gate in
    # build_blocked_tables() runs the naive kernel for it.
    rng = np.random.default_rng(9)
    W = 60
    bp = np.array(list(itertools.combinations(range(1, W // 2 + 1), 2)))
    bp = bp[np.sort(rng.choice(len(bp), 100, replace=False))]
    pairs = np.column_stack([2 * bp[:, 0] - 1, 2 * bp[:, 1] - 1])
    m = {"e": np.round(rng.normal(size=(W, 4)), 4),
         "J": [np.round(rng.normal(size=(4, 4)), 4) for _ in range(len(pairs))],
         "pairs": pairs, "intercept": 0.25}
    codes = rng.integers(0, 4, size=(3000, W), dtype=np.int32)
    res = _pymisha.pm_potts_score_codes_cmp(_one_strand_params(m), codes)
    assert not res["use_blocked"][0]
    np.testing.assert_array_equal(res["best_delta"], res["delta_max"])
    assert res["score4_mismatch"] == 0


def test_a_per_base_potts_scan_scores_from_the_encoded_chunk_as_from_one_fetch_per_bin():
    # Bins that step forward a few bases at a time read their bases from an encoded
    # stretch of the chromosome (PottsScorer::target_from_chunk); a bin extracted on its
    # own fetches its bases. Both orientations, odd and even widths, an iterator shift,
    # extend = False, max_edits = 1, across both edges of the N run at chr1 [167280, 217280).
    regions = pm.gintervals("1", [166900, 216900], [167700, 217700])
    probe = np.sort(np.concatenate([
        167280 + np.array([-60, -30, -21, -20, -19, -8, -1, 0, 3]),
        217280 + np.array([-25, -20, -10, -1, 0, 1, 2, 7, 19, 20, 21, 64]),
        [166950, 167333, 217500, 217601],
    ]))
    configs = [
        (20, "full", "potts.max", {"bidirect": False, "strand": 1}, (-3, -3)),
        (20, "full", "potts.max", {"bidirect": False, "strand": -1}, (4, 4)),
        (7, "sparse", "potts.max", {"bidirect": True}, (0, 0)),
        (21, "full", "potts.max", {"bidirect": False, "strand": -1, "extend": False}, (0, 20)),
        (6, "full", "potts.count", {"bidirect": True, "score_thresh": 0}, (0, 0)),
        (8, "sparse", "potts.max.pos", {"bidirect": True}, (0, 0)),
        (9, "full", "potts.max", {"bidirect": False, "strand": -1, "max_edits": 1}, (-2, -2)),
        (9, "full", "potts.max", {"bidirect": True, "max_edits": 1}, (0, 0)),
        (5, "none", "potts.max", {"bidirect": False, "strand": 1}, (0, 0)),
    ]
    for k, (w, mode, func, p, sh) in enumerate(configs, start=1):
        m = potts_ref_model(w, npair_mode=mode, seed=300 + k)
        pm.gvtrack_create("pchunk", None, func, params={**m, **p})
        pm.gvtrack_iterator("pchunk", sshift=sh[0], eshift=sh[1])
        dense = pm.gextract("pchunk", regions, iterator=1)
        got = dense.set_index("start").loc[probe, "pchunk"].to_numpy(dtype=float)
        # each probe in a call of its own: a fresh scorer, so a fetch
        one = np.array([pm.gextract("pchunk", pm.gintervals("1", int(s), int(s) + 1), iterator=1)["pchunk"].iloc[0]
                        for s in probe], dtype=float)
        np.testing.assert_array_equal(got, one, err_msg=f"config {k} ({func}, W = {w})")
        # NaN where an anchor reads an N (potts.count counts 0 there instead)
        if func != "potts.count":
            assert np.isnan(one).any() and (~np.isnan(one)).any(), f"config {k} reaches the N run"
        pm.gvtrack_rm("pchunk")


def test_a_windows_potts_max_and_count_are_those_of_its_anchors_one_by_one():
    # A window of many anchors is scored four anchors at a time
    # (PottsModel::score_codes4); a 1 bp bin scores its one anchor alone.
    regions = pm.gintervals("1", [167000, 217000], [167600, 217600])
    K = 30
    configs = [
        (20, "full", {"bidirect": False, "strand": 1}),
        (20, "full", {"bidirect": False, "strand": -1}),
        (41, "sparse", {"bidirect": True}),
        (7, "none", {"bidirect": False, "strand": -1}),
        (9, "sparse", {"bidirect": True}),
    ]
    for k, (w, mode, p) in enumerate(configs, start=1):
        m = potts_ref_model(w, npair_mode=mode, seed=400 + k)
        pm.gvtrack_create("pwin", None, "potts.max", params={**m, **p})
        pm.gvtrack_create("pwinc", None, "potts.count", params={**m, **p, "score_thresh": 0})
        win = pm.gextract(["pwin", "pwinc"], regions, iterator=K)
        one = pm.gextract(["pwin", "pwinc"], regions, iterator=1)
        # the K bp bin of each 1 bp bin: bins sit at multiples of K, the first one of a
        # region cut at its start
        r0 = np.where(one["start"].to_numpy() >= 217000, 217000, 167000)
        one["bin"] = np.maximum(r0, one["start"].to_numpy() // K * K)
        agg = one.groupby("bin").agg(
            mx=("pwin", lambda v: np.nan if v.isna().all() else v.max()),
            cnt=("pwinc", "sum"),
        )
        assert set(win["start"]) <= set(agg.index)
        np.testing.assert_array_equal(win["pwin"].to_numpy(dtype=float),
                                      agg.loc[win["start"], "mx"].to_numpy(dtype=float), err_msg=f"config {k}, max")
        np.testing.assert_array_equal(win["pwinc"].to_numpy(dtype=float),
                                      agg.loc[win["start"], "cnt"].to_numpy(dtype=float), err_msg=f"config {k}, count")
        assert win["pwin"].isna().any() and win["pwin"].notna().any(), f"config {k} reaches the N run"
        pm.gvtrack_rm("pwin")
        pm.gvtrack_rm("pwinc")


# ---------------------------------------------------------------------------
# pymisha: the filtered path scores its parts with the vtrack's direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", ["pwm.edit_distance", "pwm.edit_distance.pos", "pwm.max.edit_distance",
                                  "pwm.n_mutations", "pwm.edit_distance.lse", "pwm.edit_distance.lse.pos"])
def test_a_filtered_edit_distance_vtrack_keeps_direction_below(func):
    # The unmasked parts of a filtered vtrack are scored through pm_vtrack_compute, which
    # used to drop `direction` and score "below" as "above". A mask away from the
    # intervals leaves each interval one part, scored as without a filter.
    pssm = _df([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]])
    iv = pm.gintervals("1", [2000, 5000, 9000], [2040, 5030, 9050])
    # the LSE of a whole interval is near -1.0 to -2.2 here; one window scores -3 to -9
    thr = -1.5 if "lse" in func else -6
    kw = {"pssm": pssm, "score_thresh": thr, "direction": "below", "bidirect": True, "prior": 0.01}
    pm.gvtrack_create("plain", None, func, **kw)
    pm.gvtrack_create("filtered", None, func, **kw)
    pm.gvtrack_filter("filtered", pm.gintervals("1", 100000, 100500))
    r = pm.gextract(["plain", "filtered"], iv, iterator=iv)
    assert r["plain"].notna().any()
    np.testing.assert_array_equal(r["filtered"].to_numpy(), r["plain"].to_numpy())


# ---------------------------------------------------------------------------
# Flags as R misha 5.17.0 takes them, and #226 (gseq.pwm_edits NA extend)
# ---------------------------------------------------------------------------

def test_gseq_pwm_edits_refuses_an_na_extend_on_sequences():
    # R misha #226: an NA extend acted as True on sequences, as it did in gseq_pwm
    pssm = _df([[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]])
    for ext in (np.nan, pd.NA, [True, False]):
        with pytest.raises(ValueError, match="extend must be"):
            pm.gseq_pwm_edits("GTGT", pssm, score_thresh=-1, prior=0, extend=ext)
    with pytest.raises(ValueError, match="extend must be"):
        pm.gseq_pwm_edits(pm.gintervals("1", 1000, 1010), pssm, score_thresh=-1, extend=np.nan)


def test_pwm_and_kmer_vtrack_flags_take_what_r_takes():
    pssm = _df([[.7, .1, .1, .1], [.1, .7, .1, .1]])
    iv = pm.gintervals("1", 10000, 10100)

    def vt(func, **kw):
        pm.gvtrack_create("v", None, func, **kw)
        return pm.gextract("v", iv, iterator=50)

    for func, kw in (("pwm", {}), ("pwm.max", {}), ("pwm.count", {"score_thresh": -2})):
        for v in (1, 0, 2.0, "TRUE", np.int64(1)):
            with pytest.raises(ValueError, match="^bidirect must be True or False"):
                vt(func, pssm=pssm, bidirect=v, **kw)
            with pytest.raises(ValueError, match="^extend must be True or False"):
                vt(func, pssm=pssm, extend=v, **kw)
        vt(func, pssm=pssm, bidirect=np.False_, extend=np.True_, strand=1, **kw)
    # the edit-distance functions also take a non-negative whole number for extend
    for v in (1, 0, 2, 2.0):
        vt("pwm.edit_distance", pssm=pssm, score_thresh=-1, extend=v)
    for v in (2.5, -1, "TRUE"):
        with pytest.raises(ValueError, match="extend must be True, False, or a non-negative integer"):
            vt("pwm.edit_distance", pssm=pssm, score_thresh=-1, extend=v)
    with pytest.raises(ValueError, match="extend must be True, False, or a non-negative integer"):
        vt("pwm.edit_distance", params={"pssm": pssm, "score_thresh": -1, "extend": 2.5})
    # strand 1 or -1, also with bidirect=True; True is not a number
    for func, kw in (("pwm", {}), ("pwm.edit_distance", {"score_thresh": -1})):
        with pytest.raises(ValueError, match="strand parameter must be numeric"):
            vt(func, pssm=pssm, strand=True, **kw)
        for v in (0, 2, False, np.nan, "1"):
            with pytest.raises(ValueError, match="^strand must be 1 or -1"):
                vt(func, pssm=pssm, strand=v, **kw)
        vt(func, pssm=pssm, strand=-1.0, **kw)
    for v in (-0.1, 1.5, np.nan, True, "0.1"):
        with pytest.raises(ValueError, match="prior must be a number between 0 and 1"):
            vt("pwm", pssm=pssm, prior=v)
    # kmer: extend True or False only; strand -1, 0 or 1, not a bool
    for v in (1, 0, 2.0, "TRUE"):
        with pytest.raises(ValueError, match="extend parameter must be True or False"):
            vt("kmer.count", kmer="AC", extend=v)
    for v in (True, False, 2, np.nan, "1"):
        with pytest.raises(ValueError, match="strand parameter must be -1, 0, or 1"):
            vt("kmer.count", kmer="AC", strand=v)
    vt("kmer.count", kmer="AC", strand=-1.0, extend=np.False_)


def test_potts_vtrack_strand_nan_is_refused_with_its_message():
    m = potts_ref_model(3, npair_mode="full", seed=11)
    with pytest.raises(ValueError, match="strand must be 1 or -1"):
        pm.gvtrack_create("v", None, "potts.max", params={**m, "bidirect": False, "strand": np.nan})


def test_gseq_function_flags_take_what_r_takes():
    pssm = _df([[.7, .1, .1, .1], [.1, .7, .1, .1]])
    seq = "ACGTACGTAC"
    # gseq_pwm: bidirect is read as a logical (a number passes), not NA or a string
    np.testing.assert_array_equal(pm.gseq_pwm(seq, pssm, mode="max", bidirect=1),
                                  pm.gseq_pwm(seq, pssm, mode="max", bidirect=True))
    for v in (np.nan, pd.NA, "TRUE", [True, False]):
        with pytest.raises(ValueError, match="bidirect must be True or False"):
            pm.gseq_pwm(seq, pssm, mode="max", bidirect=v)
    for v in (np.nan, pd.NA):
        with pytest.raises(ValueError, match="skip_gaps must be True or False"):
            pm.gseq_pwm(seq, pssm, mode="max", skip_gaps=v)
        with pytest.raises(ValueError, match="skip_gaps must be True or False"):
            pm.gseq_kmer(seq, "AC", skip_gaps=v)
    for v in (-0.1, 2, np.nan, True):
        with pytest.raises(ValueError, match="prior must be a number between 0 and 1"):
            pm.gseq_pwm(seq, pssm, mode="max", prior=v)
    # gseq_kmer: extend a non-negative number (truncated); strand True/False are 1/0
    np.testing.assert_array_equal(pm.gseq_kmer(seq, "AC", extend=2.7), pm.gseq_kmer(seq, "AC", extend=2))
    np.testing.assert_array_equal(pm.gseq_kmer(seq, "AC", strand=True), pm.gseq_kmer(seq, "AC", strand=1))
    np.testing.assert_array_equal(pm.gseq_kmer(seq, "AC", strand=False), pm.gseq_kmer(seq, "AC", strand=0))
    for v in (np.nan, pd.NA, 2, "1"):
        with pytest.raises(ValueError, match="strand must be -1, 0, or 1"):
            pm.gseq_kmer(seq, "AC", strand=v)
    # gseq_potts: extend True or False only; strand True is 1
    m = potts_ref_model(2, npair_mode="full", seed=12)
    for v in (1, 0, np.nan, "TRUE"):
        with pytest.raises(ValueError, match="extend must be True or False"):
            pm.gseq_potts(seq, m, mode="max", extend=v)
    np.testing.assert_array_equal(pm.gseq_potts(seq, m, mode="max", bidirect=False, strand=True),
                                  pm.gseq_potts(seq, m, mode="max", bidirect=False, strand=1))
    # gseq_pwm_edits: strand is compared with 1 and -1, so True passes and False does not
    a = pm.gseq_pwm_edits(seq, pssm, -1, bidirect=False, strand=True)
    b = pm.gseq_pwm_edits(seq, pssm, -1, bidirect=False, strand=1)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="strand must be 1 or -1"):
        pm.gseq_pwm_edits(seq, pssm, -1, bidirect=False, strand=False)


# ---------------------------------------------------------------------------
# A sequence byte above 127: str input is read one byte per character (Latin-1),
# as gseq_extract returns such a byte and as R reads a native string
# ---------------------------------------------------------------------------

_HB = b"AC\xc8ACAC".decode("latin-1")  # R: rawToChar(as.raw(c(0x41, 0x43, 0xc8, 0x41, 0x43, 0x41, 0x43)))
_HB_PSSM = _df([[.7, .1, .1, .1], [.1, .7, .1, .1]])


def _hb_potts():
    return {"e": np.log(_HB_PSSM.to_numpy()), "J": [np.full((4, 4), 0.1)], "pairs": np.array([[1, 2]]),
            "intercept": 0.0}


def _hb_values(s):
    p = _HB_PSSM
    return {
        "pwm_lse": pm.gseq_pwm(s, p, mode="lse", bidirect=False, strand=1)[0],
        "pwm_max": pm.gseq_pwm(s, p, mode="max", bidirect=False, strand=1)[0],
        "pwm_pos": pm.gseq_pwm(s, p, mode="pos", bidirect=False, strand=1)[0],
        "pwm_count": pm.gseq_pwm(s, p, mode="count", score_thresh=-1, bidirect=False, strand=1)[0],
        "pwm_lse_log_quarter": pm.gseq_pwm(s, p, mode="lse", neutral_chars_policy="log_quarter")[0],
        "pwm_max_roi": pm.gseq_pwm(s, p, mode="max", start_pos=4, end_pos=7)[0],
        "kmer_count": pm.gseq_kmer(s, "AC", mode="count", strand=1)[0],
        "kmer_frac": pm.gseq_kmer(s, "AC", mode="frac", strand=1)[0],
        "kmer_roi": pm.gseq_kmer(s, "AC", mode="count", strand=1, start_pos=4, end_pos=7)[0],
        "potts_lse": pm.gseq_potts(s, _hb_potts(), mode="lse", bidirect=False, strand=1)[0],
        "potts_max": pm.gseq_potts(s, _hb_potts(), mode="max", bidirect=False, strand=1)[0],
        "potts_pos": pm.gseq_potts(s, _hb_potts(), mode="pos", bidirect=False, strand=1)[0],
    }


def test_a_byte_above_127_is_one_non_acgt_position_as_in_r():
    # the values R misha 5.17.0 gives for the same bytes
    expected = {"pwm_lse": 0.3431594347, "pwm_max": -0.7634220719, "pwm_pos": 1, "pwm_count": 3,
                "pwm_lse_log_quarter": 0.3744160099, "pwm_max_roi": -0.7397024497, "kmer_count": 3,
                "kmer_frac": 0.5, "kmer_roi": 2, "potts_lse": 0.4920420878, "potts_max": -0.6133498879,
                "potts_pos": 1}
    got = _hb_values(_HB)
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, abs=1e-6), k
    pos = pm.gseq_pwm(_HB, _HB_PSSM, mode="pos", bidirect=True, return_strand=True)
    assert pos.to_numpy().ravel().tolist() == [1, 1]
    # str.upper() would make U+00DF two characters ("SS"); the bytes stay one each
    lower = b"ac\xdfacac".decode("latin-1")
    assert pm.gseq_pwm(lower, _HB_PSSM, mode="pos", bidirect=False, strand=1, score_thresh=-1).tolist() == [1]
    np.testing.assert_allclose(pm.gseq_pwm(lower, _HB_PSSM, mode="lse"),
                               pm.gseq_pwm(b"AC\xdfACAC".decode("latin-1"), _HB_PSSM, mode="lse"))
    assert pm.grevcomp(_HB).encode("latin-1") == b"GTGT\xc8GT"
    assert pm.gseq_comp(_HB).encode("latin-1") == b"TG\xc8TGTG"


def test_a_character_above_u00ff_is_read_as_its_utf8_bytes():
    # R reads a UTF-8 string's bytes; pymisha does the same, one position a byte
    u8 = "AC中ACAC"
    as_bytes = u8.encode("utf-8").decode("latin-1")
    got, ref = _hb_values(u8), _hb_values(as_bytes)
    for k in got:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
    assert got["kmer_frac"] == 3 / 8


@pytest.mark.parametrize("max_indels", [None, 1])
def test_gseq_pwm_edits_counts_window_starts_in_bytes_as_r(max_indels):
    # R misha 5.17.0: seq_idx, window_start, n_edits, edit_type, window_seq, mutated_seq
    cases = {
        b"CC\xc8AAC".decode("latin-1"): (5, 0, None, "AC", "AC"),
        "CC中AAC": (7, 0, None, "AC", "AC"),
        b"CC\xc8AA".decode("latin-1"): (1, 1, "sub", "CC", "AC") if max_indels is None else (5, 1, "ins", "A-", "AC"),
    }
    for s, (start, n, etype, wseq, mseq) in cases.items():
        r = pm.gseq_pwm_edits([s, "GGAC"], _HB_PSSM, -1, bidirect=False, prior=0.01, max_indels=max_indels)
        first = r[r["seq_idx"] == 1].iloc[0]
        assert (first["window_start"], first["n_edits"]) == (start, n)
        assert (first["edit_type"] if n else None) == etype
        assert (first["window_seq"], first["mutated_seq"]) == (wseq, mseq)
        assert r[r["seq_idx"] == 2]["window_start"].iloc[0] == 3


def test_ggenome_implant_writes_a_donor_byte_above_127_as_one_byte(tmp_path):
    ref = tmp_path / "ref.fa"
    ref.write_bytes(b">chr1\nAAAAAAAAAA\n")
    out = tmp_path / "out.fa"
    donor = b"C\xc8G".decode("latin-1")
    intervals = pd.DataFrame({"chrom": ["chr1"], "start": [2], "end": [5]})
    pm.ggenome_implant(intervals, [donor], str(out), genome_fasta=str(ref))
    assert out.read_bytes().split(b"\n")[1] == b"AAC\xc8GAAAAA"
