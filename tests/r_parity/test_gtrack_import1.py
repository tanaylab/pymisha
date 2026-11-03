"""Parity port of R misha ``test-gtrack.import1.R`` (mapped-sequence import).

``gtrack.import_mappedseq`` from an Eland export file -- plain coverage and
with ``pileup``/``binsize`` -- followed by extraction over a screened scope
matches R's baselines. The SAM import is checked, as in R, against the 5'
bases computed independently from the file.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

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
    sam = f"{_IF}/sample-small.sam"
    t = track_namer()
    pm.gtrack_import_mappedseq(t, "", sam, cols_order=None, remove_dups=False)
    r = pm.gextract(t, pm.gintervals_all(), colnames=["v"])

    # Expected 5' bases, computed independently: POS is 1-based; a reverse read ends at POS - 1 + its CIGAR span.
    with open(sam) as fh:
        rows = [line.rstrip("\n").split("\t")[1:6] for line in fh if not line.startswith("@")]
    reads = pd.DataFrame({
        "flag": [int(x[0]) for x in rows], "chrom": [x[1] for x in rows],
        "pos": [int(x[2]) for x in rows], "cigar": [x[4] for x in rows],
    })
    reads = reads[(reads["flag"] & 0xB04) == 0]
    span = reads["cigar"].map(lambda c: sum(int(op[:-1]) for op in re.findall(r"[0-9]+[MDN=X]", c)))
    reverse = (reads["flag"] & 0x10) != 0
    reads = reads.assign(start=np.where(reverse, reads["pos"] - 1 + span - 1, reads["pos"] - 1))
    chroms = pm.gintervals_all()
    chrom_end = dict(zip(chroms["chrom"].astype(str), chroms["end"].astype(int), strict=True))
    reads = reads[reads["chrom"].isin(chrom_end)]
    reads = reads[reads["start"] < reads["chrom"].map(chrom_end)]
    expected = reads.groupby(["chrom", "start"]).size().reset_index(name="freq").sort_values(["chrom", "start"])

    got = pd.DataFrame({"chrom": r["chrom"].astype(str), "start": r["start"].astype(int), "v": r["v"].astype(float)})
    got = got.sort_values(["chrom", "start"])
    assert len(got) == len(expected)
    assert got["chrom"].tolist() == expected["chrom"].tolist()
    assert got["start"].tolist() == expected["start"].tolist()
    np.testing.assert_array_equal(got["v"].to_numpy(), expected["freq"].to_numpy(dtype=float))


def test_import_pileup_binsize(overlay_db, track_namer):
    t = track_namer()
    pm.gtrack_import_mappedseq(t, "", f"{_IF}/s_7_export.txt", remove_dups=False, pileup=180, binsize=50)
    assert_matches_baseline(
        pm.gextract(t, _scope(), colnames=[t]), "track.import_pileup_binsize"
    )
