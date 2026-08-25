"""An overlapping 1D scope must be counted once, as R misha counts it.

misha canonicalises the scope in every aggregating entry point
(GenomeTrackSummary/Quantiles/Cor/Screener/Sampler/Partition/Distribution/
Segmentation/Wilcox). pymisha did it in gdist, gpartition and glookup only, so
gsummary over chr1:0-1000 + chr1:500-1500 reported 40 intervals and sum
2.557778 where misha reported 30 and 2.337778 - same call, same data, silently
different answer.

Expected values below were read off R misha 5.11.24 on the same example
database, not off pymisha.
"""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(scope="module")
def examples():
    pm.gdb_init_examples()


OVERLAPPING = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 500], "end": [1000, 1500]})
MERGED = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [1500]})


def test_gsummary_unifies_an_overlapping_scope(examples):
    ov = pm.gsummary("dense_track", intervals=OVERLAPPING)
    mg = pm.gsummary("dense_track", intervals=MERGED)
    assert ov["Total intervals"] == mg["Total intervals"] == 30  # R misha
    assert ov["Sum"] == pytest.approx(2.337778, abs=1e-6)        # R misha
    assert ov["Sum"] == pytest.approx(mg["Sum"])


def test_gquantiles_and_gscreen_unify(examples):
    assert pm.gquantiles("dense_track", [0.5], intervals=OVERLAPPING).equals(
        pm.gquantiles("dense_track", [0.5], intervals=MERGED)
    )
    assert len(pm.gscreen("dense_track > 0.1", intervals=OVERLAPPING)) == len(
        pm.gscreen("dense_track > 0.1", intervals=MERGED)
    ) == 3  # R misha


def test_gpartition_unifies(examples):
    """Two identical scope intervals gave 74 partitions where misha gave 37.

    Found by a differential run against R, not by reading the code - the audit
    behind this fix first recorded gpartition as already unifying, because the
    helper definitions further down the file fell inside the function span it
    was measuring.
    """
    breaks = [0, 0.25, 0.5, 0.75, 1.0]
    dup = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 0], "end": [10000, 10000]})
    single = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10000]})
    n_dup = len(pm.gpartition("dense_track", breaks, intervals=dup))
    n_single = len(pm.gpartition("dense_track", breaks, intervals=single))
    assert n_dup == n_single == 37  # R misha


def test_gextract_does_not_unify(examples):
    """The counterpart: per-interval output must keep the caller's intervals.

    misha does not unify in gextract - each scope interval gets its own rows and
    its own intervalID - so canonicalising here would be its own parity break.
    """
    assert len(pm.gextract("dense_track", intervals=OVERLAPPING)) == 40  # R misha
    assert len(pm.gextract("dense_track", intervals=MERGED)) == 30       # R misha
