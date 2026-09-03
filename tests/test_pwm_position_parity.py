"""PWM ``*.pos`` conventions and the spatially-weighted sliding window.

Ported from R misha 5.11.25 / 5.11.26:
- tests/testthat/test-pwm.R  (forward-strand ``*.pos``, strand under bidirect)
- tests/testthat/test-pwm-spatial.R  (sliding vs non-sliding agreement)

``*.pos`` reports the 1-based position of the first base of the match in
forward-strand orientation, on either strand and under either ``bidirect``
setting - the same convention ``gseq_pwm`` uses.
"""

import os

import numpy as np
import pytest

import pymisha as pm

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def _rc(word: str) -> str:
    return word.translate(_COMPLEMENT)[::-1]


def _consensus_pssm(word: str) -> np.ndarray:
    """A PSSM whose consensus is ``word`` and which matches nothing else."""
    m = np.full((len(word), 4), 0.001)
    for i, base in enumerate(word):
        m[i, "ACGT".index(base)] = 0.997
    return m


class TestPwmPositionStrand:
    """*.pos reports forward-strand coordinates on both strands."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    @pytest.fixture
    def fixture(self):
        """A motif planted so the correct answer is known without scanning."""
        iv = pm.gintervals("1", 500, 600)
        seq = pm.gseq_extract(iv)
        if isinstance(seq, (list, np.ndarray)):
            seq = seq[0]
        seq = str(seq).upper()
        motif_len, pos0 = 8, 40
        word = seq[pos0:pos0 + motif_len]
        rc_word = _rc(word)
        # Assert, do not skip: if the fixture stops being unique the test must
        # fail loudly rather than silently pass without checking anything.
        assert seq.count(word) == 1, "planted word is not unique"
        assert rc_word not in seq, "reverse complement also occurs on the plus strand"
        return iv, seq, word, rc_word, motif_len, pos0 + 1

    @pytest.mark.parametrize("extend", [True, False])
    def test_pos_is_forward_strand_start_for_minus_strand(self, fixture, extend):
        iv, _seq, word, rc_word, motif_len, expected = fixture
        # A PSSM whose consensus is the reverse complement of `word` matches the
        # MINUS strand at that site; one whose consensus is `word` the PLUS strand.
        pssm_minus = _consensus_pssm(rc_word)
        pssm_plus = _consensus_pssm(word)

        pm.gvtrack_create("p_minus", None, "pwm.max.pos", pssm=pssm_minus,
                          prior=0, bidirect=False, strand=-1, extend=extend)
        pm.gvtrack_create("p_plus", None, "pwm.max.pos", pssm=pssm_plus,
                          prior=0, bidirect=False, strand=1, extend=extend)
        pm.gvtrack_create("e_minus", None, "pwm.edit_distance.pos", pssm=pssm_minus,
                          prior=0, bidirect=False, strand=-1, extend=extend,
                          score_thresh=-1)
        r = pm.gextract(["p_minus", "p_plus", "e_minus"], intervals=iv, iterator=iv)

        # Both strands must report the same convention: the forward-strand start.
        assert r["p_minus"].iloc[0] == expected
        assert r["p_plus"].iloc[0] == expected
        # The edit-distance position must land inside the same window, not in
        # reverse-complemented target coordinates.
        assert expected <= r["e_minus"].iloc[0] <= expected + motif_len - 1

    def test_gseq_pwm_agrees(self, fixture):
        """gseq_pwm is an independent implementation; it must agree."""
        _iv, seq, _word, rc_word, _motif_len, expected = fixture
        got = pm.gseq_pwm(seq, _consensus_pssm(rc_word), mode="pos",
                          strand=-1, bidirect=False, prior=0)
        assert float(np.asarray(got).ravel()[0]) == expected

    def test_strand_is_ignored_when_bidirect(self, fixture):
        """Under bidirect the sign of *.pos records which strand matched, and
        must not depend on the (ignored) strand argument."""
        iv, _seq, word, rc_word, _motif_len, expected = fixture
        for pssm, sign in ((_consensus_pssm(word), 1), (_consensus_pssm(rc_word), -1)):
            vals = []
            for strand in (1, -1):
                pm.gvtrack_create("v", None, "pwm.max.pos", pssm=pssm, prior=0,
                                  bidirect=True, strand=strand)
                vals.append(pm.gextract("v", intervals=iv, iterator=iv)["v"].iloc[0])
            assert vals[0] == vals[1]
            assert vals[0] == sign * expected

    def test_pos_is_nan_when_sequence_shorter_than_pssm(self):
        """extend=True at a chromosome end can fetch fewer bases than the PSSM
        has positions. That has no window at all - NaN, not a fabricated
        position (misha 5.11.25)."""
        chrom_size = 500000
        pssm = _consensus_pssm("ACGTACGTACGTACGTACGT")  # 20bp
        pm.gvtrack_create("tail", None, "pwm.max.pos", pssm=pssm, prior=0,
                          bidirect=False, strand=1, extend=True)
        iv = pm.gintervals("1", chrom_size - 5, chrom_size)
        r = pm.gextract("tail", intervals=iv, iterator=iv)
        assert np.isnan(r["tail"].iloc[0])


class TestPwmLseEditDistancePos:
    """The LSE scorer's reverse pass indexes a reverse-complemented copy."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_lse_pos_points_at_a_real_edit_on_the_reverse_strand(self):
        # Getting the remap wrong mirrors the position about the interval, which
        # still looks like a plausible answer - so verify functionally, with
        # gseq_pwm as an independent oracle: applying the single best edit AT the
        # reported position must cross the threshold, at the mirrored one it must not.
        motif_len = 7
        rng = np.random.default_rng(60427)
        pssm = rng.uniform(0.05, 1.0, size=(motif_len, 4))
        pssm = pssm / pssm.sum(axis=1, keepdims=True)

        def lse_of(seq):
            return float(np.asarray(
                pm.gseq_pwm(seq, pssm, mode="lse", strand=-1, bidirect=False, prior=0)
            ).ravel()[0])

        found = 0
        for start in range(2000, 40000, 1000):
            iv = pm.gintervals("1", start, start + 40)
            target = pm.gseq_extract(pm.gintervals("1", start, start + 40 + motif_len - 1))
            if isinstance(target, (list, np.ndarray)):
                target = target[0]
            target = str(target).upper()
            if "N" in target:
                continue
            thresh = lse_of(target) + 0.35

            params = dict(pssm=pssm, prior=0, bidirect=False, strand=-1,
                          extend=True, score_thresh=thresh, direction="above")
            pm.gvtrack_create("k", None, "pwm.edit_distance.lse", **params)
            pm.gvtrack_create("q", None, "pwm.edit_distance.lse.pos", **params)
            r = pm.gextract(["k", "q"], intervals=iv, iterator=iv)
            # Only a window needing exactly one edit gives an unambiguous answer;
            # with two or more the optimum is a set and either member is legal.
            if np.isnan(r["k"].iloc[0]) or r["k"].iloc[0] != 1:
                continue

            pos = int(r["q"].iloc[0])
            n = len(target)

            def best_at(at):
                if at < 1 or at > n:
                    return -np.inf
                return max(lse_of(target[:at - 1] + b + target[at:]) for b in "ACGT")

            assert best_at(pos) >= thresh - 1e-6
            assert best_at(n + 1 - pos) < thresh - 1e-6
            found += 1
            if found >= 3:
                break
        # Assert rather than skip: no usable window means the test checked nothing.
        assert found >= 1


class TestPwmSpatialSlidingParity:
    """The sliding window is an optimization: it must equal the plain scan."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()
        os.environ.pop("MISHA_DISABLE_SPATIAL_SLIDING", None)

    def test_sliding_agrees_with_non_sliding(self):
        # Regression coverage for three defects that made them disagree
        # (misha 5.11.26):
        #   - the incoming anchor was read from a fixed offset, so a stride > 1
        #     slide (any fixed-size iterator) reused one base for every step;
        #   - the per-bin max rescan re-selected the element being evicted/moved;
        #   - the seed's hit count stopped at bins*B, missing the positions that
        #     clamp into the last bin when the profile is shorter than the window.
        rng = np.random.default_rng(60427)
        intervals = pm.gintervals("1", 2000, 2300)
        checked = 0
        ref = None
        for motif_len in (6, 9):
            pssm = rng.uniform(0.05, 1.0, size=(motif_len, 4))
            pssm = pssm / pssm.sum(axis=1, keepdims=True)
            for spat_bin in (1, 5):
                # A short profile relative to the window exercises the last-bin clamp.
                for spat_len in (10, 60):
                    spat = rng.uniform(0.2, 3.0, size=spat_len)
                    for func in ("pwm", "pwm.max", "pwm.max.pos", "pwm.count"):
                        for strand in (1, -1):
                            params = dict(pssm=pssm, prior=0, bidirect=False,
                                          strand=strand, extend=True,
                                          spat_factor=spat, spat_bin=spat_bin)
                            if func == "pwm.count":
                                params["score_thresh"] = -12
                            pm.gvtrack_create("spat_v", None, func, **params)

                            os.environ.pop("MISHA_DISABLE_SPATIAL_SLIDING", None)
                            slid = pm.gextract("spat_v", intervals=intervals, iterator=50)
                            os.environ["MISHA_DISABLE_SPATIAL_SLIDING"] = "1"
                            ref = pm.gextract("spat_v", intervals=intervals, iterator=50)

                            slid = slid.sort_values("start")
                            ref = ref.sort_values("start")
                            np.testing.assert_allclose(
                                slid["spat_v"].values, ref["spat_v"].values,
                                rtol=1e-5, atol=1e-5,
                                err_msg=(f"func={func} strand={strand} "
                                         f"spat_bin={spat_bin} spat_len={spat_len} "
                                         f"motif_len={motif_len}"),
                            )
                            checked += 1
        # More than one bin must be produced, or sliding never happens and the
        # test would pass without exercising anything.
        assert len(ref) > 1
        assert checked > 0
