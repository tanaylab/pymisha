"""Regression tests for cache invalidation on rm + create on indexed DBs.

Ports R commit 4c3803b0 scenarios (test-index-cache-invalidation.R).

These tests confirm that gtrack_rm + gtrack_create_* and gintervals_rm +
gintervals_save on an indexed DB do not leave stale per-directory index
caches behind. R had a process-static cache bug here; pymisha uses
_pm_dbreload-based invalidation, which should already invalidate.
"""

import pandas as pd
import pytest

import pymisha as pm


def _write_per_chrom_db(root, chrom_rows):
    """Build a minimal per-chrom DB. Pattern from tests/test_gdb_convert_to_indexed.py."""
    seq_dir = root / "seq"
    tracks_dir = root / "tracks"
    pssms_dir = root / "pssms"
    seq_dir.mkdir(parents=True)
    tracks_dir.mkdir()
    pssms_dir.mkdir()
    with open(root / "chrom_sizes.txt", "w", encoding="utf-8") as fh:
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


@pytest.fixture
def indexed_test_db(tmp_path, restore_db):
    """Build a small indexed trackdb in tmp_path and init it. Yields the root."""
    root = tmp_path / "db"
    _write_per_chrom_db(
        root,
        [("chr1", "A" * 20000), ("chr2", "C" * 15000), ("chr3", "T" * 10000)],
    )
    pm.gdb_convert_to_indexed(groot=str(root), force=True, validate=False)
    pm.gdb_init(str(root))
    yield root


# ---------------------------------------------------------------------------
# Task 3.1: 1D dense rm + recreate (3 cycles)
# ---------------------------------------------------------------------------


def test_indexed_db_dense_rm_recreate_3cycles(indexed_test_db):
    """gtrack_create_dense -> gtrack_rm -> gtrack_create_dense should reflect
    new values each cycle. Before R fix 4c3803b0, rep 2 raised
    'Cannot open .../track.dat: No such file or directory' because the stale
    s_index_cache entry from rep 1 routed reads through the indexed path on a
    directory that had been deleted."""
    intervs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5000]})

    for i in range(1, 4):
        if pm.gtrack_exists("idxcache_dense"):
            pm.gtrack_rm("idxcache_dense", force=True)
        pm.gtrack_create_dense(
            "idxcache_dense",
            "test",
            intervs,
            [float(i)],
            binsize=200,
        )
        result = pm.gextract("idxcache_dense", intervals=intervs, iterator=200)
        assert len(result) > 0, f"rep {i}: gextract returned no rows"


# ---------------------------------------------------------------------------
# Task 3.2a: sparse rm + recreate (3 cycles)
# ---------------------------------------------------------------------------


def test_indexed_db_sparse_rm_recreate_3cycles(indexed_test_db):
    """gtrack_create_sparse -> gtrack_rm -> gtrack_create_sparse should succeed
    each cycle and expose correct track metadata. Before R fix 4c3803b0,
    rep 2's gtrack.create_sparse (and the subsequent read) errored because
    the stale s_index_cache entry from rep 1 routed I/O through the indexed
    path on a directory that had been deleted."""
    # Non-overlapping sparse intervals on chr1
    starts = list(range(0, 5000, 55))
    ends = [s + 50 for s in starts]
    sparse_ivs = pd.DataFrame({
        "chrom": ["chr1"] * len(starts),
        "start": starts,
        "end": ends,
    })

    for i in range(1, 4):
        if pm.gtrack_exists("idxcache_sparse"):
            pm.gtrack_rm("idxcache_sparse", force=True)
        vals = [float(i * 10 + j) for j in range(len(starts))]
        # The create itself would raise "Cannot open track.dat" if the
        # index cache still points at the just-deleted directory.
        pm.gtrack_create_sparse("idxcache_sparse", "test", sparse_ivs, vals)
        info = pm.gtrack_info("idxcache_sparse")
        assert info["type"] == "sparse", f"rep {i}: unexpected type {info['type']}"
        assert info["format"] == "indexed", f"rep {i}: expected indexed, got {info['format']}"


# ---------------------------------------------------------------------------
# Task 3.2b: intervals rm + resave (3 cycles)
# ---------------------------------------------------------------------------


def test_indexed_db_intervals_rm_resave_3cycles(indexed_test_db):
    """gintervals_rm + gintervals_save cycle should load fresh data each time,
    not stale per-dir index data from the first save."""
    for i in range(1, 4):
        if pm.gintervals_exists("idxcache_iv"):
            pm.gintervals_rm("idxcache_iv", force=True)
        iset = pd.DataFrame({
            "chrom": ["chr1"] * 5,
            "start": [i * 100 + j * 10 for j in range(5)],
            "end": [i * 100 + j * 10 + 9 for j in range(5)],
        })
        pm.gintervals_save(iset, "idxcache_iv")
        loaded = pm.gintervals_load("idxcache_iv")
        assert loaded is not None and len(loaded) > 0, (
            f"rep {i}: gintervals_load returned nothing"
        )
        assert int(loaded["start"].iloc[0]) == i * 100, (
            f"rep {i}: expected start={i * 100}, got {loaded['start'].iloc[0]}"
        )


# ---------------------------------------------------------------------------
# Task 3.2c: convert_to_indexed after warm read invalidates the reader cache
# ---------------------------------------------------------------------------


def test_convert_to_indexed_round_trip_with_cached_reader(tmp_path, restore_db):
    """convert_to_indexed after a warm per-chrom read should still be
    observable by the next read. Before R fix 4c3803b0 the nullptr cache
    entry (per-chrom => no index) persisted after conversion, causing
    readers to miss the newly created track.dat."""
    root = tmp_path / "db"
    _write_per_chrom_db(
        root,
        [("chr1", "A" * 20000), ("chr2", "C" * 15000)],
    )
    pm.gdb_init(str(root))  # per-chrom, not yet indexed

    intervs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5000]})
    pm.gtrack_create_dense("idxcache_conv", "test", intervs, [1.0], binsize=200)

    # Warm the reader cache on the per-chrom layout.
    _ = pm.gextract("idxcache_conv", intervals=intervs, iterator=200)

    # Convert in-place; reader cache MUST drop the prior nullptr entry.
    pm.gdb_convert_to_indexed(groot=str(root), force=True, validate=False)

    result = pm.gextract("idxcache_conv", intervals=intervs, iterator=200)
    assert len(result) > 0, "post-conversion gextract returned no rows"
