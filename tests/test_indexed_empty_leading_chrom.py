"""Regression for the indexed-dense empty-leading-chrom bin_size=0 / SIGFPE bug.

Ported from R misha tests/testthat/test-indexed-empty-leading-chrom.R (5.10.1).

When a per-chrom dense track is packed into the indexed format with a chrom
order whose leading chrom has no per-chrom file (e.g. it was removed, or a
cross-db copy whose destination lists a leading chrom absent from the source),
the resulting track.idx has length=0 for that chrom. The old
GenomeTrackFixedBin::init_read returned early on the length=0 entry without
populating m_bin_size, leaving it at the constructor default 0. gtrack.info
(which probes the genome's first chrom) then reported bin_size=0, and subsequent
unguarded `interval.start / m_bin_size` divisions in the read paths triggered
SIGFPE.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import pymisha as pm
from pymisha import _pymisha


def _write_per_chrom_db(root: Path, chrom_rows):
    seq_dir = root / "seq"
    tracks_dir = root / "tracks"
    pssms_dir = root / "pssms"
    seq_dir.mkdir(parents=True)
    tracks_dir.mkdir()
    pssms_dir.mkdir()
    with (root / "chrom_sizes.txt").open("w") as fh:
        for chrom, seq in chrom_rows:
            fh.write(f"{chrom}\t{len(seq)}\n")
            (seq_dir / f"{chrom}.seq").write_bytes(seq.encode("ascii"))


@pytest.fixture
def restore_db():
    old_root = pm._shared._GROOT
    old_user = pm._shared._UROOT
    yield
    if old_root is None:
        pm.gdb_unload()
    else:
        pm.gdb_init(old_root, old_user)


def test_indexed_dense_empty_leading_chrom(tmp_path, restore_db):
    root = tmp_path / "db"
    _write_per_chrom_db(root, [("1", "A" * 10000), ("2", "A" * 8000)])
    pm.gdb_init(str(root))

    pm.gtrack_create_dense(
        "t",
        "x",
        pm.gintervals(["1", "2"], [0, 0], [10000, 8000]),
        [1.0, 1.0],
        binsize=20,
        defval=0.0,
        func="coverage",
    )

    # Remove chrom 1's per-chrom file so the pack writes a length=0 entry for it.
    track_dir = root / "tracks" / "t.track"
    (track_dir / "1").unlink()
    assert (track_dir / "2").exists()

    # Pack directly with both chroms in the dest order; chrom 1 gets length=0.
    # (gtrack_convert_to_indexed probes gtrack_info first, which would fail on
    # the missing per-chrom file before reaching the indexed read path, so we
    # call the low-level kernel like R's test does.)
    _pymisha.pm_track_pack_per_chrom_to_indexed(str(track_dir), ["1", "2"], "dense")
    assert (track_dir / "track.idx").exists()
    assert (track_dir / "track.dat").exists()

    # gtrack.info probes the first chrom (length=0). Before the fix it reported
    # bin_size=0; the fix back-fills from the first non-empty index entry.
    info = pm.gtrack_info("t")
    assert int(info["bin_size"]) == 20

    # Reading chrom 2 (the populated chrom) must not be poisoned by the earlier
    # empty-chrom open that left m_bin_size at 0 (was a SIGFPE).
    out = pm.gextract("t", pm.gintervals("2", 0, 100), iterator=20)
    assert len(out) == 5
    assert (out["t"] == 1).all()
