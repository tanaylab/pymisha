"""R misha 5.15.0 PWM fixes, ported from test-vtrack-review-fixes.R (263-700). As there,
the expected values are computed by brute force, not by another pymisha code path, on small
databases whose sequences are drawn here.

No window holding N or another non-ACGT base is scored (pwm vtracks, gseq_pwm by default,
the edit distances, gseq_pwm_edits, gtrack_create_pwm_energy); on the minus strand the spatial
bins and spat_min/spat_max count from the 5' start; pwm.edit_distance.lse with bidirect answers
for the strand-combined LSE."""

import itertools
import os

import numpy as np
import pytest

import pymisha as pm

BASES = np.array(list("ACGT"))
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _pssm():
    m = np.random.default_rng(60427).random((6, 4))
    return m / m.sum(axis=1, keepdims=True)


PSSM = _pssm()
LP = np.log(PSSM)
L = len(PSSM)


def _db(tmp_path, chroms):
    fasta = tmp_path / "genome.fa"
    fasta.write_text("".join(f">{name}\n{''.join(seq)}\n" for name, seq in chroms.items()))
    root = tmp_path / "db"
    pm.gdb_create(str(root), str(fasta), db_format="indexed")
    pm.gdb_init(str(root))
    return root


def window_scores(bases, anchors, strand):
    """Scores of the windows starting at 0-based anchors, -inf for a window with a non-ACGT base."""
    out = []
    for a in anchors:
        w = list(bases[a:a + L])
        if len(w) < L or any(b not in COMP for b in w):
            out.append(-np.inf)
            continue
        if strand == -1:
            w = [COMP[b] for b in reversed(w)]
        out.append(sum(LP[i, "ACGT".index(b)] for i, b in enumerate(w)))
    return np.array(out)


def lse(x):
    x = np.asarray(x, dtype=float)
    m = x.max()
    return m if not np.isfinite(m) else m + np.log(np.exp(x - m).sum())


@pytest.fixture()
def fix_db(tmp_path):
    rng = np.random.default_rng(1)
    s1 = rng.choice(BASES, 5000)
    _db(tmp_path, {"chr1": s1, "chr2": rng.choice(BASES, 300)})
    yield s1
    pm.gvtrack_clear()


@pytest.fixture()
def n_db(tmp_path):
    """400 bp with N at 100-105, R at 200 and N at 299-329 (0-based)."""
    s = np.random.default_rng(5).choice(BASES, 400).astype("<U1")
    s[100:106] = "N"
    s[200] = "R"
    s[299:330] = "N"
    _db(tmp_path, {"chr1": s})
    yield s
    pm.gvtrack_clear()


def test_minus_strand_spatial_bins_count_from_the_5_prime_start(fix_db):  # R :263
    s = fix_db
    starts = 3100 + np.arange(0, 58, 3)
    ivs = pm.gintervals("chr1", starts, starts + 60)
    spat = [1, 50, 3]
    kw = dict(pssm=PSSM, prior=0, bidirect=False, strand=-1)
    pm.gvtrack_create("m_pwm", None, "pwm", spat_factor=spat, spat_bin=25, **kw)
    pm.gvtrack_create("m_max", None, "pwm.max", spat_factor=spat, spat_bin=25, **kw)
    pm.gvtrack_create("m_pos", None, "pwm.max.pos", spat_factor=spat, spat_bin=25, **kw)
    pm.gvtrack_create("m_rpos", None, "pwm.max.pos", spat_min=1, spat_max=20, **kw)
    pm.gvtrack_create("m_rmax", None, "pwm.max", spat_min=1, spat_max=20, **kw)
    r = pm.gextract(["m_pwm", "m_max", "m_pos", "m_rpos", "m_rmax"], ivs, iterator=ivs).sort_values("intervalID")
    anchors = np.arange(60)
    for k, st in enumerate(starts):
        sc = window_scores(s, st + anchors, -1)
        w = sc + np.log(np.asarray(spat, dtype=float)[np.minimum(anchors // 25, len(spat) - 1)])
        np.testing.assert_allclose(r["m_pwm"].iloc[k], lse(w), rtol=1e-4)
        np.testing.assert_allclose(r["m_max"].iloc[k], w.max(), rtol=1e-4)
        assert r["m_pos"].iloc[k] == np.argmax(w) + 1
        in_range = anchors <= 20 - L
        np.testing.assert_allclose(r["m_rmax"].iloc[k], sc[in_range].max(), rtol=1e-4)
        assert r["m_rpos"].iloc[k] == np.argmax(sc[in_range]) + 1


def test_pwm_vtracks_never_score_a_non_acgt_window(tmp_path):  # R :297
    s = np.random.default_rng(5).choice(BASES, 400).astype("<U1")
    s[100:106] = "N"
    s[200] = "R"
    s[249:255] = "Y"
    _db(tmp_path, {"chr1": s})
    try:
        pm.gvtrack_create("n_pwm", None, "pwm", pssm=PSSM, prior=0)
        pm.gvtrack_create("n_max", None, "pwm.max", pssm=PSSM, prior=0)
        pm.gvtrack_create("n_pos", None, "pwm.max.pos", pssm=PSSM, prior=0, bidirect=False)
        pm.gvtrack_create("n_fmax", None, "pwm.max", pssm=PSSM, prior=0, bidirect=False)
        pm.gvtrack_create("n_spw", None, "pwm.max", pssm=PSSM, prior=0, spat_factor=[1, 2], spat_bin=5)
        pm.gvtrack_create("n_sp", None, "pwm", pssm=PSSM, prior=0, spat_factor=[1, 2], spat_bin=5)
        spat_w = np.log(np.array([1.0, 2.0])[np.minimum(np.arange(10) // 5, 1)])

        def both_at(st):
            a = st + np.arange(10)
            return np.logaddexp(window_scores(s, a, 1), window_scores(s, a, -1)), window_scores(s, a, 1)

        for st in (90, 95, 100, 190, 196, 240, 245, 300):
            iv = pm.gintervals("chr1", st, st + 10)
            both, fwd = both_at(st)
            r = pm.gextract(["n_pwm", "n_max", "n_pos", "n_fmax", "n_spw", "n_sp"], iv, iterator=iv)
            np.testing.assert_allclose(r["n_pwm"][0], lse(both), rtol=1e-4, err_msg=str(st))
            np.testing.assert_allclose(r["n_max"][0], both.max(), rtol=1e-4, err_msg=str(st))
            np.testing.assert_allclose(r["n_fmax"][0], fwd.max(), rtol=1e-4, err_msg=str(st))
            if np.isfinite(fwd.max()):
                assert r["n_pos"][0] == np.argmax(fwd) + 1, st
            else:
                assert np.isnan(r["n_pos"][0]), st
            np.testing.assert_allclose(r["n_spw"][0], (both + spat_w).max(), rtol=1e-4, err_msg=str(st))
            np.testing.assert_allclose(r["n_sp"][0], lse(both + spat_w), rtol=1e-4, err_msg=str(st))
            # the spatial value of an interval does not depend on the interval scored before it
            two = pm.gintervals("chr1", [10, st], [20, st + 10])
            assert pm.gextract("n_spw", two, iterator=two)["n_spw"].iloc[1] == r["n_spw"][0]

        # overlapping windows in one call, where the sliding caches run
        ivs = pm.gintervals("chr1", np.arange(85, 121), np.arange(95, 131))
        r = pm.gextract(["n_pwm", "n_max", "n_spw", "n_sp"], ivs, iterator=ivs).sort_values("intervalID")
        for k, st in enumerate(range(85, 121)):
            both, _ = both_at(st)
            np.testing.assert_allclose(r["n_pwm"].iloc[k], lse(both), rtol=1e-4, err_msg=str(st))
            np.testing.assert_allclose(r["n_max"].iloc[k], both.max(), rtol=1e-4, err_msg=str(st))
            np.testing.assert_allclose(r["n_spw"].iloc[k], (both + spat_w).max(), rtol=1e-4, err_msg=str(st))
            np.testing.assert_allclose(r["n_sp"].iloc[k], lse(both + spat_w), rtol=1e-4, err_msg=str(st))
    finally:
        pm.gvtrack_clear()


def _reachable_lse(bases, length, editable, n_thresh):
    """The strand-combined LSE of `bases` and of every sequence 1 and 2 substitutions away
    (at the editable positions), and thresholds in the gaps between reachable scores."""
    def combined(b):
        a = np.arange(length)
        return lse(np.concatenate([window_scores(b, a, 1), window_scores(b, a, -1)]))

    f0 = combined(bases)
    one, one_pos = [], []
    for q in editable:
        for y in "ACGT":
            if y != bases[q]:
                b = bases.copy()
                b[q] = y
                one.append(combined(b))
                one_pos.append(q + 1)
    two = []
    for q1, q2 in itertools.combinations(editable, 2):
        for y1, y2 in itertools.product("ACGT", repeat=2):
            if y1 != bases[q1] and y2 != bases[q2]:
                b = bases.copy()
                b[q1], b[q2] = y1, y2
                two.append(combined(b))
    vals = np.unique(np.round(np.concatenate([[f0], one, two]), 6))
    gaps = np.flatnonzero(np.diff(vals) > 1e-3)
    pick = gaps[np.unique(np.round(np.linspace(0, len(gaps) - 1, n_thresh)).astype(int))]
    return f0, np.array(one), np.array(one_pos), np.array(two), (vals[pick] + vals[pick + 1]) / 2


def _expected_edits(thresh, above, f0, one, two):
    def hit(x):
        return np.any(np.asarray(x) >= thresh) if above else np.any(np.asarray(x) <= thresh)
    return 0 if hit([f0]) else 1 if hit(one) else 2 if hit(two) else np.nan


def test_lse_edit_distance_with_bidirect_targets_the_combined_score(fix_db):  # R :359
    s = fix_db
    length = 8
    iv = pm.gintervals("chr1", 3100, 3100 + length)
    bases = s[3100:3100 + length + L - 1].copy()
    f0, one, one_pos, two, thresholds = _reachable_lse(bases, length, range(len(bases)), 12)
    pm.gvtrack_create("e_pwm", None, "pwm", pssm=PSSM, prior=0)
    np.testing.assert_allclose(pm.gextract("e_pwm", iv, iterator=iv)["e_pwm"][0], f0, rtol=1e-4)
    for thresh in thresholds:
        for dirn in ("above", "below"):
            kw = dict(pssm=PSSM, prior=0, score_thresh=thresh, max_edits=2, direction=dirn)
            pm.gvtrack_create("e_ed", None, "pwm.edit_distance.lse", **kw)
            pm.gvtrack_create("e_pos", None, "pwm.edit_distance.lse.pos", **kw)
            r = pm.gextract(["e_ed", "e_pos"], iv, iterator=iv)
            want = _expected_edits(thresh, dirn == "above", f0, one, two)
            if np.isnan(want):
                assert np.isnan(r["e_ed"][0]), (thresh, dirn)
            else:
                assert r["e_ed"][0] == want, (thresh, dirn)
            if want == 1:  # the reported position is one where a single substitution reaches it
                at = one[one_pos == abs(r["e_pos"][0])]
                assert (at >= thresh).any() if dirn == "above" else (at <= thresh).any(), (thresh, dirn)


def test_gseq_pwm_leaves_out_n_windows_by_default(n_db):  # R :464
    for f in ("pwm", "pwm.max", "pwm.max.pos"):
        pm.gvtrack_create(f.replace(".", "_"), None, f, pssm=PSSM)
    pm.gvtrack_create("pwm_count", None, "pwm.count", pssm=PSSM, score_thresh=-7)
    for st in (95, 98, 195, 296):
        iv = pm.gintervals("chr1", st, st + 10)
        v = pm.gextract(["pwm", "pwm_max", "pwm_max_pos", "pwm_count"], iv, iterator=iv)
        seq = pm.gseq_extract(pm.gintervals("chr1", st, st + 15))[0]

        def g(mode, **kw):
            return pm.gseq_pwm([seq], PSSM, mode=mode, end_pos=10, extend=True, **kw)[0]
        np.testing.assert_allclose(g("lse"), v["pwm"][0], rtol=1e-5, err_msg=str(st))
        np.testing.assert_allclose(g("max"), v["pwm_max"][0], rtol=1e-5, err_msg=str(st))
        if np.isfinite(v["pwm_max"][0]):
            assert abs(g("pos")) == abs(v["pwm_max_pos"][0]), st
        else:
            assert np.isnan(g("pos")) and np.isnan(v["pwm_max_pos"][0]), st
        assert g("count", score_thresh=-7) == v["pwm_count"][0], st
    # nothing scorable
    assert pm.gseq_pwm(["NNNNNNNNNN"], PSSM, mode="lse")[0] == -np.inf
    assert pm.gseq_pwm(["NNNNNNNNNN"], PSSM, mode="max")[0] == -np.inf
    assert np.isnan(pm.gseq_pwm(["NNNNNNNNNN"], PSSM, mode="pos")[0])
    assert np.isnan(pm.gseq_pwm(["RRRRRRRRRR"], PSSM, mode="pos")[0])
    assert pm.gseq_pwm(["NNNNNNNNNN"], PSSM, mode="count", score_thresh=-100)[0] == 0
    # scoring N on request still charges the column mean
    lp = np.log((PSSM + 0.01) / (PSSM + 0.01).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(
        pm.gseq_pwm(["ACGNNN"], PSSM, mode="max", bidirect=False, neutral_chars=["N"])[0],
        lp[[0, 1, 2], [0, 1, 2]].sum() + lp[3:].mean(axis=1).sum(), rtol=1e-5)


def window_edits(w, thresh, above):
    """Fewest substitutions that bring one window to the threshold (NaN if none): the largest
    per-column gains first, which is exact for a position weight matrix."""
    cols = ["ACGT".index(b) for b in w]
    cur = LP[np.arange(L), cols]
    gains = np.sort(LP.max(axis=1) - cur if above else cur - LP.min(axis=1))[::-1]
    need = thresh - cur.sum() if above else cur.sum() - thresh
    if need <= 0:
        return 0
    k = np.flatnonzero(np.cumsum(gains) >= need - 1e-9)
    return np.nan if len(k) == 0 else k[0] + 1


def test_edit_distances_never_use_a_non_acgt_window(n_db):  # R :516
    s = n_db
    for st in (92, 97, 195):
        iv = pm.gintervals("chr1", st, st + 10)
        ws = [s[a:a + L] for a in st + np.arange(10)]
        ok = np.array([all(b in COMP for b in w) for w in ws])
        scores = np.array([LP[np.arange(L), ["ACGT".index(b) for b in w]].sum() if k else -np.inf
                           for w, k in zip(ws, ok, strict=True)])
        for dirn in ("above", "below"):
            thresh = scores.max() + 1.5 if dirn == "above" else scores[ok].max() - 1.5
            ref = np.array([window_edits(w, thresh, dirn == "above") if k else np.nan
                            for w, k in zip(ws, ok, strict=True)])
            for f in ("pwm.edit_distance", "pwm.edit_distance.pos", "pwm.max.edit_distance"):
                pm.gvtrack_create("ed", None, f, pssm=PSSM, prior=0, bidirect=False, score_thresh=thresh,
                                  direction=dirn)
                got = pm.gextract("ed", iv, iterator=iv)["ed"][0]
                if f == "pwm.edit_distance":
                    want = np.nanmin(ref) if not np.isnan(ref).all() else np.nan
                elif f == "pwm.edit_distance.pos":
                    want = np.nanargmin(ref) + 1 if not np.isnan(ref).all() else np.nan
                else:
                    want = ref[np.argmax(scores)]
                if np.isnan(want):
                    assert np.isnan(got), (st, dirn, f)
                else:
                    assert got == want, (st, dirn, f)
            if not np.isnan(ref).all() and dirn == "above":  # the string version agrees
                e = pm.gseq_pwm_edits(pm.gseq_extract(pm.gintervals("chr1", st, st + 10 + L - 1)), PSSM,
                                      score_thresh=thresh, bidirect=False, prior=0, extend=False)
                assert e["n_edits"].iloc[0] == np.nanmin(ref), (st, "gseq_pwm_edits")
    # an interval whose every window holds an N: NA for the whole family
    iv = pm.gintervals("chr1", 302, 312)
    for f in ("pwm.edit_distance", "pwm.edit_distance.pos", "pwm.max.edit_distance", "pwm.n_mutations",
              "pwm.edit_distance.lse", "pwm.edit_distance.lse.pos"):
        for dirn in ("above", "below"):
            pm.gvtrack_create("ed", None, f, pssm=PSSM, score_thresh=-5, direction=dirn)
            assert np.isnan(pm.gextract("ed", iv, iterator=iv)["ed"][0]), (f, dirn)


def test_gseq_pwm_edits_gives_no_row_for_a_non_acgt_sequence():  # R test-gseq-pwm-edits.R:379
    pssm = np.array([[0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]])
    assert len(pm.gseq_pwm_edits(["NC"], pssm, score_thresh=-1, prior=0)) == 0  # its one window holds N


def test_gseq_pwm_edits_integer_extend_on_a_sequence():  # R test-gseq-pwm-edits.R, misha 5.18.0
    """An integer extend on a sequence is how many bases a window may run past its end."""
    pssm = np.array([[.97, .01, .01, .01], [.01, .97, .01, .01], [.01, .01, .97, .01], [.01, .01, .01, .97]])
    thr = 4 * np.log(0.97) - 0.01

    def ed(extend):
        r = pm.gseq_pwm_edits("TTTTTTAC", pssm, score_thresh=thr, max_indels=2, extend=extend, prior=0,
                              bidirect=False)
        return r[["window_start", "n_edits"]].drop_duplicates().values.tolist()

    # the AC at 7-8 needs G and T inserted past the end: 2 bases
    assert ed(True) == [[7, 2]]
    assert ed(2) == ed(True)
    assert 7 not in [w for w, _ in ed(1)]
    assert 7 not in [w for w, _ in ed(False)]


def test_lse_edit_distance_leaves_out_n_windows(n_db):  # R :572
    s = n_db
    length, st = 8, 89  # windows 89..96: the N run at 100..105 cuts the last two
    iv = pm.gintervals("chr1", st, st + length)
    bases = s[st:st + length + L - 1].copy()
    editable = [q for q in range(len(bases)) if bases[q] in COMP]
    f0, one, _, two, thresholds = _reachable_lse(bases, length, editable, 10)
    assert np.isfinite(f0)
    pm.gvtrack_create("e_pwm", None, "pwm", pssm=PSSM, prior=0)
    np.testing.assert_allclose(pm.gextract("e_pwm", iv, iterator=iv)["e_pwm"][0], f0, rtol=1e-4)
    for thresh in thresholds:
        for dirn in ("above", "below"):
            pm.gvtrack_create("e_ed", None, "pwm.edit_distance.lse", pssm=PSSM, prior=0, score_thresh=thresh,
                              max_edits=2, direction=dirn)
            got = pm.gextract("e_ed", iv, iterator=iv)["e_ed"][0]
            want = _expected_edits(thresh, dirn == "above", f0, one, two)
            if np.isnan(want):
                assert np.isnan(got), (thresh, dirn)
            else:
                assert got == want, (thresh, dirn)


def test_create_pwm_energy_leaves_out_n_windows(n_db):  # R :627
    s = n_db
    pssms = pm._shared._GROOT + "/pssms"
    os.makedirs(pssms, exist_ok=True)
    with open(f"{pssms}/tset.key", "w") as f:
        f.write("0 x 0\n")  # whitespace-separated, as R reads it; bidirect 0
    with open(f"{pssms}/tset.data", "w") as f:
        f.writelines(f"0\t{i}\t{a:.10f}\t{c:.10f}\t{g:.10f}\t{t:.10f}\n" for i, (a, c, g, t) in enumerate(PSSM))
    pm.gtrack_create_pwm_energy("en", "x", "tset", 0, 0, iterator=10)
    r = pm.gextract("en", pm.gintervals("chr1", 60, 340))
    for _, row in r.iterrows():
        # windows starting in [start, end) of the bin, as in the pwm vtrack
        anchors = [a for a in range(int(row["start"]), int(row["end"])) if a + L <= 400]
        want = lse(window_scores(s, anchors, 1))
        if want == -np.inf:  # a track stores -inf as NaN
            assert np.isnan(row["en"]), row["start"]
        else:
            np.testing.assert_allclose(row["en"], want, rtol=1e-4, err_msg=str(row["start"]))
    assert r["en"].isna().any()
