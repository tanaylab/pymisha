"""gvtrack_iterator shifts that push the iterator interval off the chromosome,
ported from R misha's tests/testthat/test-vtrack-shift-out-of-range.R.

Partially: the window is clamped to the chromosome. Completely (or into an
empty window): every virtual track function returns NaN. R's chr1 scope at
3,000,000 is past the end of pymisha's test chr1 (500 kb), so the scope here is
chr1:100,000-101,000, and R's test.fixedbin is dense_track.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _clean_vtracks():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def shift_pssm():
    return pd.DataFrame({
        "A": [0.7, 0.1, 0.1, 0.1],
        "C": [0.1, 0.7, 0.1, 0.1],
        "G": [0.1, 0.1, 0.7, 0.1],
        "T": [0.1, 0.1, 0.1, 0.7],
    })


def shifted_values(sshift, eshift, scope):
    """Every vtrack flavor that has its own out-of-range path, extracted with
    the given shift."""
    pm.gvtrack_clear()
    src = pm.gintervals(["chr1", "chr1"], [100100, 100500], [100200, 100600])
    pm.gvtrack_create("s_pwm", None, func="pwm", pssm=shift_pssm(), prior=0.01)
    pm.gvtrack_create("s_pwm_max", None, func="pwm.max", pssm=shift_pssm(), prior=0.01)
    pm.gvtrack_create("s_kmer", None, func="kmer.count", kmer="ACGT", strand=1)
    pm.gvtrack_create("s_masked", None, func="masked.count")
    pm.gvtrack_create("s_coverage", src, func="coverage")
    pm.gvtrack_create("s_neighbor", src, func="neighbor.count")
    pm.gvtrack_create("s_distance", src, func="distance")
    pm.gvtrack_create("s_track", "dense_track", "avg")
    vtracks = ["s_pwm", "s_pwm_max", "s_kmer", "s_masked",
               "s_coverage", "s_neighbor", "s_distance", "s_track"]
    for v in vtracks:
        pm.gvtrack_iterator(v, sshift=sshift, eshift=eshift)
    # 4+ sequence vtracks in one call also exercises misha's batched sequence path
    r = pm.gextract(vtracks, scope, iterator=scope)
    return r.iloc[0][vtracks].to_numpy(dtype=float)


def test_a_shift_that_leaves_the_chromosome_yields_nan_for_every_vtrack_function():
    scope = pm.gintervals("chr1", 100000, 101000)
    assert not np.isnan(shifted_values(0, 0, scope)).any()
    # window pushed 5 Mb to the left of the chromosome start
    assert np.isnan(shifted_values(-5_000_000, -5_000_000, scope)).all()
    # window pushed past the chromosome end
    all_iv = pm.gintervals_all()
    chr1_end = int(all_iv.loc[all_iv["chrom"] == "chr1", "end"].iloc[0])
    tail_scope = pm.gintervals("chr1", chr1_end - 1000, chr1_end)
    assert np.isnan(shifted_values(10000, 10000, tail_scope)).all()
    # sshift/eshift that collapse the window to nothing
    assert np.isnan(shifted_values(500, -500, scope)).all()


def test_a_shift_that_only_partly_leaves_the_chromosome_clamps_the_window():
    scope = pm.gintervals(["chr1", "chr1"], [0, 1000], [1000, 2000])
    pm.gvtrack_create("clamped", "dense_track", "avg")
    unshifted = pm.gextract("clamped", scope, iterator=scope)["clamped"].to_numpy()
    pm.gvtrack_iterator("clamped", sshift=-1000, eshift=0)
    shifted = pm.gextract("clamped", scope, iterator=scope)["clamped"].to_numpy()
    # the first bin has nothing to its left, so it keeps its unshifted value;
    # the second one widens to [0, 2000) and changes
    assert shifted[0] == unshifted[0]
    assert not np.isclose(shifted[1], unshifted[1])
    assert not np.isnan(shifted).any()
