"""Regression tests for the three scoring fixes ported from R misha.

Each one guards a defect that shipped silently: a wrong number with no warning.
They are written so that reverting the fix makes them fail - a test that passes
either way would be worse than no test, and two earlier attempts at the
position check did exactly that.
"""

import numpy as np
import pytest

import pymisha as pm


def _onehot_pssm(word):
    p = np.full((len(word), 4), 1e-9)
    for i, ch in enumerate(word):
        p[i, "ACGT".index(ch)] = 1 - 3e-9
    return p


def test_scan_is_not_capped_at_a_megabase():
    """DnaPSSM::m_max_range defaulted to 1000000, which is a cap of a million
    and not "no cap", so a scan longer than that answered from its first
    megabase. The assertion is self-consistency and needs no oracle: the
    maximum over an interval must equal the maximum over its parts."""
    W, word = 24, "ACGTACGTACGTACGTACGTACGT"
    # the match sits past the old cap, and nothing before it scores as well
    s = "G" * 1_200_000 + word + "G" * 100_000
    p = _onehot_pssm(word)
    whole = float(np.asarray(pm.gseq_pwm([s], p, mode="max", bidirect=False,
                                         strand=1, prior=1e-7)).ravel()[0])
    head = float(np.asarray(pm.gseq_pwm([s[:1_000_000]], p, mode="max", bidirect=False,
                                        strand=1, prior=1e-7)).ravel()[0])
    tail = float(np.asarray(pm.gseq_pwm([s[1_000_000:]], p, mode="max", bidirect=False,
                                        strand=1, prior=1e-7)).ravel()[0])
    assert whole == pytest.approx(max(head, tail), abs=1e-6)
    assert whole > head  # i.e. the tail really is where the maximum lives


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


@pytest.fixture(autouse=True)
def _clean_vtracks():
    _remove_all_vtracks()
    yield
    _remove_all_vtracks()


def test_running_logsumexp_survives_its_maximum_leaving():
    """pop_front() rescaled to the new max and then subtracted the departing
    term. Exact in real arithmetic; in floating point the sum is by then almost
    entirely that term, so once the gap passes ~37 nats the rest falls below
    the double epsilon, the subtraction returns 0, and the window never
    recovers - a silently wrong value, or a -Inf reading as "nothing scorable".

    The aggregator has no entry point of its own, so it is reached the only way
    a user can: a `pwm` vtrack over OVERLAPPING windows, which is what puts the
    running sum in play. gseq_pwm() scores each window from scratch and shares
    no state, so it is an independent oracle. The PSSM is deliberately peaked -
    a narrow dynamic range never opens the ~37 nat gap the defect needs."""
    W = 10
    word = "ACGTACGTAC"
    p = _onehot_pssm(word)          # exact match towers over everything else
    starts = list(range(2000, 2140))
    win = 26
    ivs = pm.gintervals(["1"] * len(starts), starts, [x + 1 for x in starts])
    pm.gvtrack_create("lse_slide", None, func="pwm", pssm=p, bidirect=False,
                      strand=1, prior=1e-7, extend=False)
    pm.gvtrack_iterator("lse_slide", sshift=0, eshift=win - 1)
    got = np.asarray(pm.gextract("lse_slide", ivs, iterator=ivs)["lse_slide"], float)

    seqs = [str(x) for x in pm.gseq_extract(
        pm.gintervals(["1"] * len(starts), starts, [x + win for x in starts]))]
    ref = np.asarray(pm.gseq_pwm(seqs, p, mode="lse", bidirect=False,
                                 strand=1, prior=1e-7, extend=False), float)
    assert np.isfinite(got).all(), "the sliding sum collapsed to -Inf"
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_reported_position_is_exact_past_2_24():
    """Positions were carried as float, and binary32 represents consecutive
    integers only to 2^24.

    The target position MUST be odd: an even one is exactly representable, so
    the check would pass against the defect. The anchor must also be unique -
    a repeat makes the scan report an earlier copy and the test says nothing.
    Both traps produced false "verified" results while this was being fixed."""
    word = "ACGTACGTACGTACGTACGTACGT"
    at = 16_777_524              # 1-based position 16777525: odd, past 2^24
    s = "G" * at + word          # 'G' runs cannot match, so the anchor is unique
    assert (at + 1) % 2 == 1, "the target position must be odd to have power"
    # pwm only: gseq_potts(mode="pos") builds its position as (double)(i + 1)
    # in PMGseqPotts and never carried the defect, so a potts case here would
    # pass whether or not the fix is present. The potts VTRACK path did carry
    # it and is covered in R misha's suite, which has a genome long enough to
    # reach an index past 2^24.
    got = pm.gseq_pwm([s], _onehot_pssm(word), mode="pos",
                      bidirect=False, strand=1, prior=1e-7)
    assert float(np.asarray(got).ravel()[0]) == at + 1
