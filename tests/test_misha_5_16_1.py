"""R misha 5.16.1: tests of test-review-2026-11-followup.R (defects in 5.16.0 found by
comparing R misha with pymisha). The Parquet test is in test_gtrack_import_parquet.py, the
2D create test in test_misha_5_16_followups_iter2d.py."""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture
def kmer_vtrack():
    pm.gvtrack_create("followup_kmer", None, "kmer.count", kmer="ACG")
    yield "followup_kmer"
    pm.gvtrack_rm("followup_kmer")


def test_a_sequence_virtual_track_next_to_a_sparse_track_picks_its_iterator(kmer_vtrack):  # R :31
    intervs = pm.gintervals("chr1", 0, 100000)
    expected = pm.gextract("sparse_track + followup_kmer", intervs, iterator="sparse_track")
    for expr in ("sparse_track + followup_kmer", "followup_kmer + sparse_track"):
        res = pm.gextract(expr, intervs)
        assert res[expr].tolist() == expected["sparse_track + followup_kmer"].tolist()


def test_a_sequence_virtual_track_next_to_a_2d_track_is_an_error(kmer_vtrack):  # R :46
    for expr in ("rects_track + followup_kmer", "followup_kmer + rects_track"):
        with pytest.raises(Exception, match="2D iterator is used without explicit dimension projection"):
            pm.gextract(expr, pm.gintervals_2d_all())


def test_a_numeric_strand_other_than_minus_one_zero_or_one_is_an_error():  # R :62
    intervs = pd.DataFrame({"chrom": ["chr1"], "start": [200], "end": [210]})
    fwd = pm.gseq_extract(intervs.assign(strand=1.0))
    assert pm.gseq_extract(intervs.assign(strand=1)) == fwd
    assert pm.gseq_extract(intervs.assign(strand=0.0)) == fwd
    for s in (255.0, 511.0, 2.0, 0.5, -2.0, 255):
        with pytest.raises(Exception, match=f"Invalid strand value {s:g} at index 1"):
            pm.gseq_extract(intervs.assign(strand=s))
