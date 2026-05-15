"""Regression tests for the 0D `gsynth_sample` unaligned-interval bug.

Roadmap follow-up #1: pymisha passed `iter_size = INT64_MAX` as a
no-constraint sentinel for 0D models. The C++ bin-bounds check did
`bins[c].first + iter_size`, which overflowed signed int64 when the
sample interval did not start at a bin boundary; the lookup returned
-1 and the sampler fell through to uniform-random base selection.

Fixed by adding a `num_bins == 1` fast path that skips the bounds
check entirely, plus a saturating-add in the general path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import pymisha as pm

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gsynth_0d_aligned_v0_1_53.txt"


@pytest.fixture(scope="module")
def forbid_cg_model():
    """0D k=2 model with CG forbidden."""
    pm.gdb_init_examples()
    model = pm.gsynth_train(
        intervals=pm.gintervals("1", 0, 50_000),
        iterator=200,
        k=2,
    )
    return pm.gsynth_forbid_kmer(model, "CG", check=False)


@pytest.fixture(scope="module")
def plain_model():
    """0D k=2 model without forbid (control for CG occurrence count)."""
    pm.gdb_init_examples()
    return pm.gsynth_train(
        intervals=pm.gintervals("1", 0, 50_000),
        iterator=200,
        k=2,
    )


def _sample_seq(model, start: int, end: int) -> str:
    out = pm.gsynth_sample(
        model,
        intervals=pm.gintervals("1", start, end),
        iterator=200,
        seed=60427,
        output_format="vector",
    )
    return out[0]


def test_unaligned_forbid_zero_cg(forbid_cg_model):
    """Unaligned interval start: forbid_kmer must still zero CG occurrences.

    Pre-fix: ~5% CG rate (uniform-random fallback). Post-fix: 0.
    Skip the first k=2 bases (seeding caveat).
    """
    seq = _sample_seq(forbid_cg_model, 64, 2064)  # start mod 200 != 0
    assert len(seq) == 2000
    cg = seq[2:].count("CG")
    assert cg == 0, (
        f"Unaligned 0D forbid sampler should yield 0 CG, got {cg}. "
        "This means the INT64_MAX overflow in PMGsynth.cpp bin lookup has regressed."
    )


def test_unaligned_forbid_zero_cg_long(forbid_cg_model):
    """10kb unaligned sample - the fix must scale, not just luck out on 2kb."""
    seq = _sample_seq(forbid_cg_model, 64, 10064)
    assert len(seq) == 10000
    assert seq[2:].count("CG") == 0


def test_unaligned_no_forbid_has_cg(plain_model):
    """Discriminator: without forbid, the same unaligned interval has > 0 CG.

    Establishes that the forbid result on the matching unaligned interval
    is real signal (not coincidence on a CG-free sequence).
    """
    seq = _sample_seq(plain_model, 64, 10064)
    assert len(seq) == 10000
    assert seq[2:].count("CG") > 0, (
        "No-forbid 10kb unaligned sample produced 0 CG - the test fixture "
        "may have changed."
    )


def test_aligned_byte_identical(forbid_cg_model):
    """Aligned-interval output unchanged by the fix.

    Compares against the fixture pinned in DF.1 (sampled on the unfixed
    v0.1.53 build immediately before applying the bin-lookup change).
    """
    seq = _sample_seq(forbid_cg_model, 0, 2000)  # start mod 200 == 0
    pinned = FIXTURE.read_text()
    assert seq == pinned, "Aligned 0D sample drifted from pinned baseline."
