"""Per-chrom analog of the indexed empty-leading-chrom bug.

Ported from R misha tests/testthat/test-perchrom-empty-leading-chrom.R (commit
292aa286, companion to the indexed #133 fix).

A per-chromosome dense track may legitimately lack a per-chrom file for the
genome's first chromosome (an empty scaffold, or a partial track). `gtrack_info`
probes the first genome chrom for bin_size; the old code called `init_read` on
the resolved (but absent) file and errored with "No such file or directory".
`gtrack_copy` calls `gtrack_info` on the source, so copying such a track failed
too. Fix: probe the first chrom that actually has a per-chrom file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import pymisha as pm


def _write_per_chrom_db(root: Path, chrom_rows):
    seq_dir = root / "seq"
    (root / "tracks").mkdir(parents=True)
    (root / "pssms").mkdir()
    seq_dir.mkdir()
    with (root / "chrom_sizes.txt").open("w") as fh:
        for chrom, size in chrom_rows:
            fh.write(f"{chrom}\t{size}\n")
            (seq_dir / f"{chrom}.seq").write_bytes(b"A" * size)


@pytest.fixture
def restore_db():
    old_root = pm._shared._GROOT
    old_user = pm._shared._UROOT
    yield
    if old_root is None:
        pm.gdb_unload()
    else:
        pm.gdb_init(old_root, old_user)


def _make_track_missing_leading_chrom(root: Path, name: str) -> Path:
    _write_per_chrom_db(root, [("1", 10000), ("2", 8000)])
    pm.gdb_init(str(root))
    pm.gtrack_create_dense(
        name, "x",
        pm.gintervals(["1", "2"], [0, 0], [10000, 8000]),
        [1.0, 1.0], binsize=20, defval=0.0, func="coverage",
    )
    track_dir = root / "tracks" / f"{name}.track"
    (track_dir / "1").unlink()  # track now lacks the genome's first chrom
    assert (track_dir / "2").exists()
    assert not (track_dir / "track.idx").exists()  # still per-chrom
    return track_dir


def test_perchrom_missing_leading_chrom_reports_bin_size(tmp_path, restore_db):
    root = tmp_path / "db"
    _make_track_missing_leading_chrom(root, "t")
    info = pm.gtrack_info("t")
    assert int(info["bin_size"]) == 20
    assert info["format"] == "per-chromosome"


def test_gtrack_copy_of_perchrom_missing_leading_chrom(tmp_path, restore_db):
    root = tmp_path / "db"
    _make_track_missing_leading_chrom(root, "src")
    pm.gtrack_copy("src", "dst")
    assert int(pm.gtrack_info("dst")["bin_size"]) == 20
