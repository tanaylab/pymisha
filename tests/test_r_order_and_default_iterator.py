"""R misha 5.14.2 parity: gextract row order, and the default iterator of an
expression made only of virtual tracks."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture
def vt():
    pm.gvtrack_create("vt_dflt", "dense_track", func="avg")
    pm.gvtrack_create("vt_seq", None, func="kmer.count", kmer="CG", strand=1)
    yield
    pm.gvtrack_rm("vt_dflt")
    pm.gvtrack_rm("vt_seq")


def test_gextract_rows_in_genome_order():
    # R: chr1:100, chr1:2000, chr2, chrX; intervalIDs name the caller's rows
    a = pd.DataFrame({"chrom": ["X", "1", "2", "1"], "start": [1, 2000, 4, 100], "end": [10, 2020, 40, 120]})
    res = pm.gextract("dense_track", a)
    assert res["chrom"].astype(str).tolist() == ["chr1", "chr1", "chr2", "chrX"]
    assert res["start"].tolist() == [100, 2000, 4, 1]
    assert res["intervalID"].tolist() == [4, 2, 3, 1]
    binned = pm.gextract("dense_track", a, iterator=10)
    assert binned["intervalID"].tolist()[:4] == [4, 4, 2, 2]


def test_value_vtrack_iterates_its_source_bins(vt):
    iv = pm.gintervals("chr1", 0, 10000)
    s = pm.gsummary("vt_dflt", iv)
    assert s["Total intervals"] == 200          # R: dense_track's 50 bp bins
    assert s["Mean"] == pytest.approx(0.04798889, rel=1e-6)
    assert s["Std dev"] == pytest.approx(0.05005616, rel=1e-6)
    assert pm.gquantiles("vt_dflt", [0.5], iv).tolist() == pytest.approx([0.04])
    assert len(pm.gscreen("vt_dflt > 0.05", iv)) == 40
    np.testing.assert_array_equal(pm.gdist("vt_dflt", [0, 0.05, 1], intervals=iv), [81, 72])
    assert len(pm.glookup(np.array([1, 2]), "vt_dflt", [0, 0.05, 1], intervals=iv)) == 200
    assert len(pm.gpartition("vt_dflt", [0, 0.05, 1], iv)) == 87
    assert pm.gcor("vt_dflt", "vt_dflt * vt_dflt", intervals=iv)[0] == pytest.approx(0.9265149, rel=1e-6)


@pytest.mark.parametrize("call", [
    lambda iv: pm.gextract("vt_seq", iv),
    lambda iv: pm.gscreen("vt_seq > 0", iv),
    lambda iv: pm.gsummary("vt_seq", iv),
])
def test_sequence_vtrack_without_iterator_is_an_error(vt, call):
    with pytest.raises(Exception, match="Cannot implicitly determine iterator policy"):
        call(pm.gintervals("chr1", 0, 1000))


def test_sequence_vtrack_with_iterator_works(vt):
    res = pm.gextract("vt_seq", pm.gintervals("chr1", 0, 1000), iterator=500)
    assert len(res) == 2
