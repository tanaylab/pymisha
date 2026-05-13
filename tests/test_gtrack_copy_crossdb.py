"""Tests for gtrack_copy cross-db destination + overwrite + format conversion.

Ported from R misha tests/testthat/test-track-copy-crossdb.R (commit
062e80e7, release 5.6.28). Covers:
  - The new C++ split / pack kernels on their own
  - Cross-DB copy via db= argument
  - Format conversion (per-chrom <-> indexed) on the fly
  - Chromosome-order remap
  - chr-prefix variant handling on the destination side
  - Multi-track input with namespace prefix
  - overwrite=True replaces an existing destination
  - .attributes survive the cross-db split + pack round-trip
"""

import contextlib
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm
from pymisha import _pymisha

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _write_per_chrom_db(root: Path, chrom_rows):
    """Create a minimal per-chrom DB at root with the given (chrom, seq) rows."""
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


def _make_two_chrom_db(root: Path, chroms=("1", "2"), size: int = 10000):
    _write_per_chrom_db(root, [(c, "A" * size) for c in chroms])


@pytest.fixture
def restore_db():
    """Save/restore the working DB across each test."""
    old_root = pm._shared._GROOT
    old_user = pm._shared._UROOT
    yield
    for ds in list(pm.gdataset_ls()):
        with contextlib.suppress(Exception):
            pm.gdataset_unload(ds, validate=False)
    if old_root is None:
        pm.gdb_unload()
    else:
        pm.gdb_init(old_root, old_user)


# ---------------------------------------------------------------------------
# C++ kernel: split_indexed_to_per_chrom
# ---------------------------------------------------------------------------


class TestSplitIndexedKernel:
    def test_round_trip_byte_identical(self, tmp_path, restore_db):
        root = tmp_path / "db_a"
        _make_two_chrom_db(root)
        pm.gdb_init(str(root))
        pm.gtrack_create_sparse(
            "t1", "src", pm.gintervals("1", 0, 1000), [7.0]
        )
        track_dir = root / "tracks" / "t1.track"
        files_before = sorted(p.name for p in track_dir.iterdir())
        bytes_before = {
            f: (track_dir / f).read_bytes() for f in files_before
        }

        pm.gtrack_convert_to_indexed("t1")
        assert (track_dir / "track.idx").exists()

        _pymisha.pm_track_split_indexed_to_per_chrom(
            str(track_dir),
            pm.tracks._db_chrom_names_at(root),
            True,
        )
        assert not (track_dir / "track.idx").exists()
        assert not (track_dir / "track.dat").exists()
        after = sorted(p.name for p in track_dir.iterdir())
        assert after == files_before
        for f in files_before:
            assert (track_dir / f).read_bytes() == bytes_before[f]

    def test_multi_chrom_split_preserves_data(self, tmp_path, restore_db):
        root = tmp_path / "db_multi"
        _make_two_chrom_db(root)
        pm.gdb_init(str(root))
        intervs = pm.gintervals(["1", "2"], [0, 0], [500, 500])
        pm.gtrack_create_sparse("tm", "src", intervs, [5.0, 5.0])
        track_dir = root / "tracks" / "tm.track"
        files_before = sorted(p.name for p in track_dir.iterdir())
        bytes_before = {
            f: (track_dir / f).read_bytes() for f in files_before
        }

        pm.gtrack_convert_to_indexed("tm")
        _pymisha.pm_track_split_indexed_to_per_chrom(
            str(track_dir),
            pm.tracks._db_chrom_names_at(root),
            True,
        )

        after = sorted(p.name for p in track_dir.iterdir())
        assert after == files_before
        for f in files_before:
            assert (track_dir / f).read_bytes() == bytes_before[f]

        pm._pymisha.pm_dbreload()
        assert (
            pm.gextract("tm", pm.gintervals("1", 0, 100))["tm"].iloc[0] == 5.0
        )
        assert (
            pm.gextract("tm", pm.gintervals("2", 0, 100))["tm"].iloc[0] == 5.0
        )

    def test_remove_indexed_false_keeps_indexed_pair(self, tmp_path, restore_db):
        root = tmp_path / "db_keep"
        _make_two_chrom_db(root)
        pm.gdb_init(str(root))
        pm.gtrack_create_sparse(
            "tk", "src", pm.gintervals("1", 0, 1000), [3.0]
        )
        track_dir = root / "tracks" / "tk.track"

        pm.gtrack_convert_to_indexed("tk")
        _pymisha.pm_track_split_indexed_to_per_chrom(
            str(track_dir),
            pm.tracks._db_chrom_names_at(root),
            False,
        )
        assert (track_dir / "track.idx").exists()
        assert (track_dir / "track.dat").exists()
        # Per-chrom file present alongside.
        assert (track_dir / "1").exists()

    def test_chromid_out_of_range_errors(self, tmp_path, restore_db):
        root = tmp_path / "db_oor"
        _make_two_chrom_db(root)
        pm.gdb_init(str(root))
        intervs = pm.gintervals(["1", "2"], [0, 0], [500, 500])
        pm.gtrack_create_sparse("t_oor", "src", intervs, [1.0, 1.0])
        pm.gtrack_convert_to_indexed("t_oor")
        track_dir = root / "tracks" / "t_oor.track"
        assert (track_dir / "track.idx").exists()

        with pytest.raises((RuntimeError, ValueError, Exception)) as exc:
            _pymisha.pm_track_split_indexed_to_per_chrom(
                str(track_dir),
                ["only_one_chrom"],
                True,
            )
        assert "chrom_id" in str(exc.value).lower() or "internal" in str(exc.value).lower()

        # Indexed pair survives the failed split.
        assert (track_dir / "track.idx").exists()
        assert (track_dir / "track.dat").exists()
        # No .tmp leftovers.
        assert not list(track_dir.glob("*.tmp"))


# ---------------------------------------------------------------------------
# gtrack_copy: cross-DB destination + overwrite
# ---------------------------------------------------------------------------


class TestGtrackCopyCrossDb:
    def test_db_argument_lands_track_in_named_dataset(
        self, tmp_path, restore_db
    ):
        workdb = tmp_path / "workdb"
        otherdb = tmp_path / "otherdb"
        _make_two_chrom_db(workdb)
        _make_two_chrom_db(otherdb)
        pm.gdb_init(str(workdb))
        pm.gdataset_load(str(otherdb), force=True)
        pm.gtrack_create_sparse(
            "src_t", "src", pm.gintervals("1", 0, 1000), [9.0]
        )

        result = pm.gtrack_copy("src_t", "copied_t", db=str(otherdb))

        assert result == ["copied_t"]
        assert pm.gtrack_exists("copied_t")
        assert pm.gtrack_dataset("copied_t") == str(otherdb)
        assert pm.gtrack_dataset("src_t") == str(workdb)
        assert (
            pm.gextract("copied_t", pm.gintervals("1", 0, 500))["copied_t"].iloc[0]
            == 9.0
        )

    def test_overwrite_false_errors_overwrite_true_replaces(
        self, tmp_path, restore_db
    ):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _make_two_chrom_db(a)
        _make_two_chrom_db(b)
        pm.gdb_init(str(a))
        pm.gdataset_load(str(b), force=True)
        pm.gtrack_create_sparse(
            "x", "src", pm.gintervals("1", 0, 100), [1.0]
        )
        pm.gtrack_copy("x", "x_copy", db=str(b))

        with pytest.raises(ValueError, match="already exists"):
            pm.gtrack_copy("x", "x_copy", db=str(b))

        # Replace source with a different value, then copy with overwrite=True.
        pm.gtrack_rm("x", force=True)
        pm.gtrack_create_sparse(
            "x", "src", pm.gintervals("1", 0, 100), [99.0]
        )

        pm.gtrack_copy("x", "x_copy", db=str(b), overwrite=True)
        # Move root to b so we can read the overwritten copy through gextract.
        # (gtrack_dataset confirms placement.)
        assert pm.gtrack_dataset("x_copy") == str(b)
        assert (
            pm.gextract("x_copy", pm.gintervals("1", 0, 100))["x_copy"].iloc[0]
            == 99.0
        )

    def test_vector_src_with_prefix_dest(self, tmp_path, restore_db):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _make_two_chrom_db(a)
        _make_two_chrom_db(b)
        pm.gdb_init(str(a))
        pm.gdataset_load(str(b), force=True)
        pm.gtrack_create_sparse(
            "x", "x", pm.gintervals("1", 0, 100), [1.0]
        )
        pm.gtrack_create_sparse(
            "y", "y", pm.gintervals("1", 0, 100), [2.0]
        )

        out = pm.gtrack_copy(["x", "y"], dest="ns", db=str(b))

        assert set(out) == {"ns.x", "ns.y"}
        assert pm.gtrack_exists("ns.x")
        assert pm.gtrack_exists("ns.y")
        assert (
            pm.gextract("ns.x", pm.gintervals("1", 0, 100))["ns.x"].iloc[0] == 1.0
        )
        assert (
            pm.gextract("ns.y", pm.gintervals("1", 0, 100))["ns.y"].iloc[0] == 2.0
        )

    def test_vector_src_with_null_dest_keeps_names(self, tmp_path, restore_db):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _make_two_chrom_db(a)
        _make_two_chrom_db(b)
        pm.gdb_init(str(a))
        pm.gdataset_load(str(b), force=True)
        pm.gtrack_create_sparse(
            "x", "x", pm.gintervals("1", 0, 100), [1.0]
        )
        pm.gtrack_create_sparse(
            "y", "y", pm.gintervals("1", 0, 100), [2.0]
        )

        out = pm.gtrack_copy(["x", "y"], db=str(b))

        assert set(out) == {"x", "y"}
        # In a loaded dataset b: x and y exist; the working DB still has its own.
        assert pm.gtrack_dataset("x") == str(a)  # working db wins
        # But b also has copies, accessible via the dataset registry.
        # Verify by switching root to b.
        pm.gdb_init(str(b))
        assert pm.gtrack_exists("x")
        assert pm.gtrack_exists("y")
        assert (
            pm.gextract("x", pm.gintervals("1", 0, 100))["x"].iloc[0] == 1.0
        )
        assert (
            pm.gextract("y", pm.gintervals("1", 0, 100))["y"].iloc[0] == 2.0
        )

    def test_chr_prefix_variant_handled_via_rename(self, tmp_path, restore_db):
        src = tmp_path / "src_chrprefix"
        dst = tmp_path / "dest_noprefix"
        _write_per_chrom_db(src, [("chr1", "A" * 10000), ("chr2", "A" * 10000)])
        _write_per_chrom_db(dst, [("1", "A" * 10000), ("2", "A" * 10000)])

        pm.gdb_init(str(src))
        # Cross-genome destinations cannot be loaded via gdataset_load (genome
        # hash mismatch). The db= argument must accept a valid but unloaded
        # misha root.
        pm.gtrack_create_sparse(
            "t", "src", pm.gintervals("chr1", 0, 100), [7.0]
        )

        pm.gtrack_copy("t", "t_copy", db=str(dst))

        pm.gdb_init(str(dst))
        assert pm.gtrack_exists("t_copy")
        result = pm.gextract("t_copy", pm.gintervals("1", 0, 100))
        assert result["t_copy"].iloc[0] == 7.0

    def test_drops_chroms_not_in_destination_with_warning(
        self, tmp_path, restore_db
    ):
        src = tmp_path / "src3"
        dst = tmp_path / "dst2"
        _write_per_chrom_db(
            src,
            [("1", "A" * 10000), ("2", "A" * 10000), ("3", "A" * 10000)],
        )
        _make_two_chrom_db(dst, chroms=("1", "2"))

        pm.gdb_init(str(src))
        # Cross-genome destination (different chromosome set) cannot be
        # gdataset_loaded; the db= argument must accept a valid unloaded path.
        intervs = pm.gintervals(["1", "3"], [0, 0], [100, 100])
        pm.gtrack_create_sparse("t", "src", intervs, [5.0, 5.0])

        with pytest.warns(UserWarning, match="3"):
            pm.gtrack_copy("t", "t_copy", db=str(dst))

        pm.gdb_init(str(dst))
        assert pm.gtrack_exists("t_copy")
        result = pm.gextract("t_copy", pm.gintervals("1", 0, 100))
        assert result["t_copy"].iloc[0] == 5.0

    def test_attrs_survive_cross_db_copy(self, tmp_path, restore_db):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _make_two_chrom_db(a)
        _make_two_chrom_db(b)
        pm.gdb_init(str(a))
        pm.gdataset_load(str(b), force=True)
        pm.gtrack_create_sparse(
            "t_attr", "src", pm.gintervals("1", 0, 100), [5.0]
        )
        pm.gtrack_attr_set("t_attr", "experiment", "foo")

        pm.gtrack_copy("t_attr", "t_attr_copy", db=str(b))

        pm.gdb_init(str(b))
        assert pm.gtrack_exists("t_attr_copy")
        assert pm.gtrack_attr_get("t_attr_copy", "experiment") == "foo"

    def test_same_src_and_dest_in_same_db_errors(self, tmp_path, restore_db):
        a = tmp_path / "a"
        _make_two_chrom_db(a)
        pm.gdb_init(str(a))
        pm.gtrack_create_sparse(
            "t", "src", pm.gintervals("1", 0, 100), [1.0]
        )

        with pytest.raises(ValueError, match="same"):
            pm.gtrack_copy("t", "t")

    def test_empty_src_list_returns_empty(self, tmp_path, restore_db):
        a = tmp_path / "a"
        _make_two_chrom_db(a)
        pm.gdb_init(str(a))
        assert pm.gtrack_copy([]) == []

    def test_invalid_db_path_errors(self, tmp_path, restore_db):
        a = tmp_path / "a"
        bad = tmp_path / "not_a_db"
        bad.mkdir()
        _make_two_chrom_db(a)
        pm.gdb_init(str(a))
        pm.gtrack_create_sparse(
            "t", "src", pm.gintervals("1", 0, 100), [1.0]
        )

        with pytest.raises(ValueError, match="chrom_sizes"):
            pm.gtrack_copy("t", "t_copy", db=str(bad))


# ---------------------------------------------------------------------------
# Cross-DB format conversion
# ---------------------------------------------------------------------------


class TestGtrackCopyFormatConversion:
    def test_per_chrom_src_to_indexed_dest_packs_on_the_fly(
        self, tmp_path, restore_db
    ):
        src = tmp_path / "perchrom_src"
        dst = tmp_path / "indexed_dest"
        _make_two_chrom_db(src)
        _make_two_chrom_db(dst)
        # Make dst indexed BEFORE creating tracks in src.
        pm.gdb_init(str(dst))
        pm.gdb_convert_to_indexed(groot=str(dst), force=True, validate=False)
        pm.gdb_init(str(src))
        pm.gdataset_load(str(dst), force=True)
        pm.gtrack_create_sparse(
            "t", "src", pm.gintervals("1", 0, 1000), [11.0]
        )

        pm.gtrack_copy("t", "t_copy", db=str(dst))

        dest_dir = dst / "tracks" / "t_copy.track"
        assert (dest_dir / "track.idx").exists()
        assert (dest_dir / "track.dat").exists()
        # Per-chrom files coexist (pymisha's read path still needs them; the
        # pack mirrors pm_track_convert_to_indexed's remove_old=False default).
        # Values readable.
        pm.gdb_init(str(dst))
        assert (
            pm.gextract("t_copy", pm.gintervals("1", 0, 500))["t_copy"].iloc[0]
            == 11.0
        )

    def test_indexed_src_to_per_chrom_dest_splits_on_the_fly(
        self, tmp_path, restore_db
    ):
        src = tmp_path / "indexed_src"
        dst = tmp_path / "perchrom_dest"
        _make_two_chrom_db(src)
        _make_two_chrom_db(dst)
        # Make src indexed first.
        pm.gdb_init(str(src))
        pm.gdb_convert_to_indexed(groot=str(src), force=True, validate=False)
        pm.gtrack_create_sparse(
            "t", "src", pm.gintervals("1", 0, 1000), [22.0]
        )
        pm.gdb_init(str(dst))
        pm.gdataset_load(str(src), force=True)

        pm.gtrack_copy("t", "t_copy", db=str(dst))

        dest_dir = dst / "tracks" / "t_copy.track"
        assert not (dest_dir / "track.idx").exists()
        # Per-chrom files present.
        assert (dest_dir / "1").exists()
        pm.gdb_init(str(dst))
        assert (
            pm.gextract("t_copy", pm.gintervals("1", 0, 500))["t_copy"].iloc[0]
            == 22.0
        )

    def test_indexed_to_indexed_with_different_chrom_order(
        self, tmp_path, restore_db
    ):
        src = tmp_path / "src_idx"
        dst = tmp_path / "dst_idx"
        _write_per_chrom_db(src, [("1", "A" * 10000), ("2", "A" * 10000)])
        _write_per_chrom_db(dst, [("2", "A" * 10000), ("1", "A" * 10000)])

        pm.gdb_init(str(src))
        pm.gdb_convert_to_indexed(groot=str(src), force=True, validate=False)
        pm.gtrack_create_sparse(
            "t",
            "src",
            pm.gintervals(["1", "2"], [0, 0], [100, 100]),
            [13.0, 13.0],
        )
        pm.gdb_init(str(dst))
        pm.gdb_convert_to_indexed(groot=str(dst), force=True, validate=False)
        pm.gdb_init(str(src))

        pm.gtrack_copy("t", "t_copy", db=str(dst))

        pm.gdb_init(str(dst))
        assert pm.gtrack_exists("t_copy")
        assert (
            pm.gextract("t_copy", pm.gintervals("1", 0, 100))["t_copy"].iloc[0]
            == 13.0
        )
        assert (
            pm.gextract("t_copy", pm.gintervals("2", 0, 100))["t_copy"].iloc[0]
            == 13.0
        )

    def test_per_chrom_dense_src_to_indexed_dest(self, tmp_path, restore_db):
        src = tmp_path / "perchrom_dense"
        dst = tmp_path / "indexed_dense"
        _make_two_chrom_db(src)
        _make_two_chrom_db(dst)
        pm.gdb_init(str(dst))
        pm.gdb_convert_to_indexed(groot=str(dst), force=True, validate=False)
        pm.gdb_init(str(src))
        pm.gdataset_load(str(dst), force=True)

        # Build a dense fixed-bin track in src (one value per interval).
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_dense("d", "dense", intervs, [1.0], binsize=100)

        pm.gtrack_copy("d", "d_copy", db=str(dst))

        dest_dir = dst / "tracks" / "d_copy.track"
        assert (dest_dir / "track.idx").exists()
        assert (dest_dir / "track.dat").exists()
        pm.gdb_init(str(dst))
        result = pm.gextract("d_copy", pm.gintervals("1", 0, 500), iterator=100)
        assert np.allclose(result["d_copy"].to_numpy(dtype=float), 1.0)
