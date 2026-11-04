"""A sequence vtrack scored over several chromosomes in one scan must give the
same values as scanning each chromosome alone (scorer caches reset per chrom)."""

import itertools

import numpy as np
import pytest

import pymisha as pm

PSSM = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]])


@pytest.mark.parametrize(
    "func,bidirect,spat,iterator",
    list(itertools.product(["pwm", "pwm.max", "pwm.max.pos"], [True, False], [True, False], [None, 10])),
)
def test_pwm_vtrack_same_values_in_multi_chrom_scan(func, bidirect, spat, iterator):
    kw = dict(pssm=PSSM, prior=0.01, bidirect=bidirect, extend=True)
    if spat:
        kw.update(spat_factor=[0.5, 1, 2, 1, 0.5], spat_bin=50)
    pm.gvtrack_create("pwm_cc", None, func=func, **kw)
    try:
        both = pm.gintervals(["chr1", "chr2", "chrX"], [1000] * 3, [1300] * 3)
        a = pm.gextract("pwm_cc", both, iterator=both if iterator is None else iterator)
        for c in ["chr2", "chrX"]:
            one = pm.gintervals([c], [1000], [1300])
            b = pm.gextract("pwm_cc", one, iterator=one if iterator is None else iterator)
            got = a[a["chrom"].astype(str) == c]["pwm_cc"].to_numpy()
            np.testing.assert_allclose(got, b["pwm_cc"].to_numpy(), rtol=1e-6, equal_nan=True)
    finally:
        pm.gvtrack_rm("pwm_cc")
