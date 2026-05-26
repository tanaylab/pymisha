"""Parity port of R misha ``test-gtrack.import1.R`` (mapped-sequence import).

``gtrack.import_mappedseq`` from an Eland export file and a SAM file -- plain
coverage and with ``pileup``/``binsize`` -- followed by extraction over a
screened scope, all match R exactly.
"""

from __future__ import annotations

import pymisha as pm

from .baseline import assert_matches_baseline

_IF = "/net/mraid20/export/tgdata/db/tgdb/misha_snapshot/input_files"


def _scope():
    return pm.gscreen("test.fixedbin > 0.1", pm.gintervals([1, 2]))


def test_import_mappedseq_s_7_export(overlay_db, track_namer):
    t = track_namer()
    pm.gtrack_import_mappedseq(t, "", f"{_IF}/s_7_export.txt", remove_dups=False)
    assert_matches_baseline(
        pm.gextract(t, _scope(), colnames=[t]), "track.import_mappedseq.s_7_export"
    )


def test_import_mappedseq_sample_small_sam(overlay_db, track_namer):
    t = track_namer()
    pm.gtrack_import_mappedseq(t, "", f"{_IF}/sample-small.sam", cols_order=None, remove_dups=False)
    assert_matches_baseline(
        pm.gextract(t, _scope(), colnames=[t]), "track.import_mappedseq.sample_small_sam"
    )


def test_import_pileup_binsize(overlay_db, track_namer):
    t = track_namer()
    pm.gtrack_import_mappedseq(t, "", f"{_IF}/s_7_export.txt", remove_dups=False, pileup=180, binsize=50)
    assert_matches_baseline(
        pm.gextract(t, _scope(), colnames=[t]), "track.import_pileup_binsize"
    )
