"""Phase 4 of Group E: byte-identical regression for direct indexed write.

When the database is indexed (genome.seq + genome.idx present), the C++
writer for gtrack_create_sparse / gtrack_create_dense produces
track.dat + track.idx directly, skipping the per-chrom intermediates
and the post-create convert step.

These tests assert that the resulting on-disk bytes are identical to
what the legacy per-chrom + gtrack_convert_to_indexed pipeline produces
on the same input.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT


def _copy_db(tmp_path: Path, name: str = "test") -> Path:
    dst = tmp_path / name / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _make_indexed(root: Path) -> None:
    """Convert genome to indexed format (creates seq/genome.idx)."""
    pm.gdb_convert_to_indexed(groot=str(root), force=True, validate=False)


def _bytes_identical(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


@pytest.fixture
def restore_db():
    old_root = pm._shared._GROOT
    old_user = pm._shared._UROOT
    yield
    if old_root is None:
        pm.gdb_unload()
    else:
        pm.gdb_init(old_root, old_user)


def _build_sparse(root: Path, name: str, intervals: pd.DataFrame, values) -> Path:
    """Create a sparse track in the given DB and return its directory."""
    pm.gdb_init(str(root))
    pm.gtrack_create_sparse(name, "sparse", intervals, values)
    return root / "tracks" / f"{name}.track"


def _build_dense(root: Path, name: str, intervals, values, binsize, defval=np.nan) -> Path:
    pm.gdb_init(str(root))
    pm.gtrack_create_dense(name, "dense", intervals, values, binsize=binsize, defval=defval)
    return root / "tracks" / f"{name}.track"


def test_sparse_direct_indexed_bytes_match_convert_pipeline(tmp_path, restore_db):
    # Per-chrom + convert path: create on non-indexed DB then convert.
    root_a = _copy_db(tmp_path, "perchrom")
    intervals = pd.DataFrame(
        {
            "chrom": ["1", "1", "2", "X"],
            "start": [0, 100, 50, 1000],
            "end": [50, 200, 150, 1500],
        }
    )
    vals = [1.5, 2.5, 3.5, 4.5]
    dir_a = _build_sparse(root_a, "trk", intervals, vals)
    pm.gtrack_convert_to_indexed("trk", remove_old=False)

    # Direct indexed write path: convert DB to indexed first, then create.
    root_b = _copy_db(tmp_path, "direct")
    _make_indexed(root_b)
    dir_b = _build_sparse(root_b, "trk", intervals, vals)

    dat_a = dir_a / "track.dat"
    idx_a = dir_a / "track.idx"
    dat_b = dir_b / "track.dat"
    idx_b = dir_b / "track.idx"
    assert dat_a.is_file() and idx_a.is_file()
    assert dat_b.is_file() and idx_b.is_file()
    assert _bytes_identical(dat_a, dat_b), "track.dat differs between pipelines"
    assert _bytes_identical(idx_a, idx_b), "track.idx differs between pipelines"

    # Direct-write produced only track.dat + track.idx (plus .attributes); no per-chrom leftover files.
    leftovers = sorted(p.name for p in dir_b.iterdir())
    assert "track.dat" in leftovers
    assert "track.idx" in leftovers
    # No per-chrom files should exist on the direct path.
    per_chrom = [n for n in leftovers if n not in {"track.dat", "track.idx", ".attributes"}]
    assert per_chrom == [], f"unexpected per-chrom files: {per_chrom}"


def test_dense_direct_indexed_bytes_match_convert_pipeline(tmp_path, restore_db):
    intervals = pd.DataFrame(
        {
            "chrom": ["1", "1", "2"],
            "start": [0, 100, 50],
            "end": [50, 200, 150],
        }
    )
    vals = [1.0, 2.0, 3.0]
    binsize = 25
    defval = 0.5

    root_a = _copy_db(tmp_path, "perchrom")
    dir_a = _build_dense(root_a, "dn", intervals, vals, binsize, defval)
    pm.gtrack_convert_to_indexed("dn", remove_old=False)

    root_b = _copy_db(tmp_path, "direct")
    _make_indexed(root_b)
    dir_b = _build_dense(root_b, "dn", intervals, vals, binsize, defval)

    dat_a = dir_a / "track.dat"
    idx_a = dir_a / "track.idx"
    dat_b = dir_b / "track.dat"
    idx_b = dir_b / "track.idx"
    assert _bytes_identical(dat_a, dat_b), "track.dat differs between pipelines"
    assert _bytes_identical(idx_a, idx_b), "track.idx differs between pipelines"

    leftovers = sorted(p.name for p in dir_b.iterdir())
    per_chrom = [n for n in leftovers if n not in {"track.dat", "track.idx", ".attributes"}]
    assert per_chrom == [], f"unexpected per-chrom files: {per_chrom}"


def test_sparse_direct_indexed_empty_chroms(tmp_path, restore_db):
    """Chroms with no data contribute zero bytes (length=0 in idx).

    Mirrors R misha 5.6.30 94a6446d: only chromosomes with payload get
    bytes in track.dat; empty chroms have offset=prev/length=0 entries.
    """
    intervals = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [10]})
    vals = [42.0]

    root = _copy_db(tmp_path)
    _make_indexed(root)
    dir_b = _build_sparse(root, "lone", intervals, vals)

    dat = (dir_b / "track.dat").read_bytes()
    idx = (dir_b / "track.idx").read_bytes()

    # Sparse signature is -1 (FORMAT_SIGNATURES[SPARSE]); record = 8 + 8 + 4 = 20 bytes.
    # Only chrom 1 has data -> sig (4) + 1 record (20) = 24 bytes total.
    assert len(dat) == 24

    # idx = 36-byte header + 3 contig entries (24 bytes each) = 108 bytes.
    assert len(idx) == 108

    # Decode the per-chrom (offset, length) trio: only chrom 1 has non-zero length.
    import struct
    for i, expected_length in enumerate([24, 0, 0]):
        chrom_id, offset, length, _reserved = struct.unpack_from("<IQQI", idx, 36 + i * 24)
        assert chrom_id == i
        assert length == expected_length, f"chrom {i}: length={length} expected {expected_length}"


def test_sparse_direct_indexed_track_info_indexed(tmp_path, restore_db):
    """Direct-written sparse track is recognized as indexed format."""
    root = _copy_db(tmp_path)
    _make_indexed(root)
    pm.gdb_init(str(root))
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr2"], "start": [0, 20, 10], "end": [10, 30, 20]}
    )
    pm.gtrack_create_sparse("sp", "smoke", intervals, [1.0, 2.0, 3.0])
    track_dir = root / "tracks" / "sp.track"
    assert (track_dir / "track.dat").is_file()
    assert (track_dir / "track.idx").is_file()
    assert pm.gtrack_info("sp")["format"] == "indexed"


def test_dense_direct_indexed_track_info_indexed(tmp_path, restore_db):
    """Direct-written dense track is recognized as indexed format."""
    root = _copy_db(tmp_path)
    _make_indexed(root)
    pm.gdb_init(str(root))
    intervals = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0, 5], "end": [10, 15]}
    )
    pm.gtrack_create_dense(
        "dn", "smoke", intervals, [2.0, 4.0], binsize=10, defval=1.0
    )
    track_dir = root / "tracks" / "dn.track"
    assert (track_dir / "track.dat").is_file()
    assert (track_dir / "track.idx").is_file()
    assert pm.gtrack_info("dn")["format"] == "indexed"
