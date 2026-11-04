"""Ports of R misha's test-masked.R:79 (iterator shifts, checked against the
shifted window) and :129 (several chromosomes)."""

import pytest

import pymisha as pm


def _frac(seq):  # R's count_masked_manually: lower case is masked
    return sum(c.islower() for c in seq) / len(seq)


@pytest.fixture(autouse=True)
def _clear():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def test_masked_frac_follows_iterator_shifts():  # test-masked.R:79
    iv = pm.gintervals("chr1", 1000, 1100)
    pm.gvtrack_create("masked_full", None, "masked.frac")
    pm.gvtrack_create("masked_shift", None, "masked.frac")
    pm.gvtrack_iterator("masked_shift", sshift=10, eshift=-10)
    s = pm.gextract(["masked_full", "masked_shift"], iv, iterator=iv)
    assert s["masked_full"][0] == pytest.approx(_frac(pm.gseq_extract(iv)[0]), abs=1e-6)
    assert s["masked_shift"][0] == pytest.approx(_frac(pm.gseq_extract(pm.gintervals("chr1", 1010, 1090))[0]), abs=1e-6)


def test_masked_frac_on_several_chromosomes():  # test-masked.R:129
    iv = pm.gintervals(["chr1", "chr2"], [200, 200], [300, 300])
    pm.gvtrack_create("masked_frac", None, "masked.frac")
    s = pm.gextract("masked_frac", iv, iterator=iv)
    assert len(s) == 2
    for i, seq in enumerate(pm.gseq_extract(iv)):
        assert s["masked_frac"][i] == pytest.approx(_frac(seq), abs=1e-6)
