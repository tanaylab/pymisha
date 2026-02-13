"""Tests for multi-database support ported from R test-multi-db.R.

These tests verify that pymisha correctly handles multiple database
contexts via gdataset_load, including track operations, extraction,
virtual tracks, interval operations, and error handling across databases.
"""

from pathlib import Path

import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _create_linked_db(tmp_path, name):
    """Create a linked DB under tmp_path/<name> linked to TEST_DB.

    Returns the string path to the new DB.
    """
    db_path = tmp_path / name
    pm.gdb_create_linked(str(db_path), parent=str(TEST_DB))
    return str(db_path)


@pytest.fixture(autouse=True)
def _restore_root():
    """Ensure we always restore root to TEST_DB after each test."""
    yield
    # Unload all loaded datasets and restore root
    for ds in list(pm.gdataset_ls()):
        pm.gdataset_unload(ds, validate=False)
    pm.gdb_init(str(TEST_DB))


# ==============================================================================
# TestMultiDbSetup: migration and basic multi-db patterns
# ==============================================================================


class TestMultiDbSetup:
    """Tests for basic multi-DB setup patterns."""

    def test_multi_db_pattern_setroot_plus_dataset_load(self, tmp_path):
        """gsetroot(db1) + gdataset_load(db2) makes tracks from both visible."""
        db1 = _create_linked_db(tmp_path, "db1")
        pm.gsetroot(db1)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track1", "from db1", intervs, [1.0])

        db2 = _create_linked_db(tmp_path, "db2")
        pm.gsetroot(db2)
        pm.gtrack_create_sparse("track2", "from db2", intervs, [2.0])

        # New pattern: working db + loaded dataset
        pm.gsetroot(db1)
        pm.gdataset_load(db2)

        assert len(pm.gdataset_ls()) == 1  # 1 loaded dataset (db2)
        tracks = pm.gtrack_ls()
        assert "track1" in tracks
        assert "track2" in tracks

    def test_empty_dataset_in_chain(self, tmp_path):
        """An empty dataset in the chain does not break anything."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track1", "working", intervs, [1.0])

        # Create an empty dataset (no tracks)
        db_empty = _create_linked_db(tmp_path, "empty_dataset")

        db2 = _create_linked_db(tmp_path, "dataset2")
        pm.gsetroot(db2)
        pm.gtrack_create_sparse("track2", "ds2", intervs, [2.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_empty)
        pm.gdataset_load(db2)

        assert len(pm.gdataset_ls()) == 2
        tracks = pm.gtrack_ls()
        assert "track1" in tracks
        assert "track2" in tracks
        assert len([t for t in tracks if t in ("track1", "track2")]) == 2


# ==============================================================================
# TestMultiDbTrackOps: track queries across databases
# ==============================================================================


class TestMultiDbTrackOps:
    """Tests for track operations across databases."""

    def test_gtrack_info_across_databases(self, tmp_path):
        """gtrack_info works for tracks in different databases."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("working_track", "sparse track", intervs, [1.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "sparse track 2", intervs, [2.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        info1 = pm.gtrack_info("working_track")
        info2 = pm.gtrack_info("ds_track")

        assert info1["type"] == "sparse"
        assert info2["type"] == "sparse"

    def test_gtrack_exists_across_databases(self, tmp_path):
        """gtrack_exists works for tracks in loaded datasets."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("working_track", "working", intervs, [1.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "dataset", intervs, [2.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        assert pm.gtrack_exists("working_track") is True
        assert pm.gtrack_exists("ds_track") is True
        assert pm.gtrack_exists("nonexistent") is False

    def test_track_attributes_across_databases(self, tmp_path):
        """Track attributes work for tracks in different databases."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("working_track", "track in working db", intervs, [1.0])
        pm.gtrack_attr_set("working_track", "custom_attr", "working_value")

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "track in dataset", intervs, [2.0])
        pm.gtrack_attr_set("ds_track", "custom_attr", "ds_value")

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        assert pm.gtrack_attr_get("working_track", "custom_attr") == "working_value"
        assert pm.gtrack_attr_get("ds_track", "custom_attr") == "ds_value"


# ==============================================================================
# TestMultiDbTrackLifecycle: create, delete, move, copy
# ==============================================================================


class TestMultiDbTrackLifecycle:
    """Tests for track lifecycle operations across databases."""

    def test_gtrack_rm_only_affects_correct_database(self, tmp_path):
        """Deleting a track only removes it from the correct database."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track_to_keep", "keep", intervs, [1.0])
        pm.gtrack_create_sparse("track_to_delete", "delete", intervs, [2.0])

        pm.gtrack_rm("track_to_delete", force=True)

        assert pm.gtrack_exists("track_to_delete") is False
        assert pm.gtrack_exists("track_to_keep") is True

        # Verify persistence after reload
        pm.gdb_reload()
        assert pm.gtrack_exists("track_to_delete") is False
        assert pm.gtrack_exists("track_to_keep") is True

    def test_gtrack_mv_in_single_db(self, tmp_path):
        """gtrack_mv works within a single database context."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("original", "original track", intervs, [42.0])

        pm.gtrack_mv("original", "renamed")

        assert pm.gtrack_exists("original") is False
        assert pm.gtrack_exists("renamed") is True

        # Value should be preserved
        result = pm.gextract("renamed", pm.gintervals("1", 0, 500))
        assert result["renamed"].iloc[0] == 42.0

    def test_gtrack_copy_in_single_db(self, tmp_path):
        """gtrack_copy works within a single database context."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("source", "src", intervs, [123.0])

        pm.gtrack_copy("source", "dest")

        assert pm.gtrack_exists("source") is True
        assert pm.gtrack_exists("dest") is True

        # Values should match
        src_val = pm.gextract("source", pm.gintervals("1", 0, 500))["source"].iloc[0]
        dst_val = pm.gextract("dest", pm.gintervals("1", 0, 500))["dest"].iloc[0]
        assert src_val == dst_val

    def test_deletion_of_overriding_track_reveals_dataset_track(self, tmp_path):
        """Deleting a working-db track reveals the shadowed dataset track."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("shared", "from working db", intervs, [100.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("shared", "from dataset", intervs, [200.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds, force=True)

        # Working db wins, so value should be 100
        result = pm.gextract("shared", pm.gintervals("1", 0, 500))
        assert result["shared"].iloc[0] == 100.0

        # Delete from working db
        pm.gtrack_rm("shared", force=True)
        pm.gdb_reload()

        # Now dataset1 should be visible
        assert pm.gtrack_exists("shared") is True
        result2 = pm.gextract("shared", pm.gintervals("1", 0, 500))
        assert result2["shared"].iloc[0] == 200.0


# ==============================================================================
# TestMultiDbExtract: gextract, gscreen, gsummary across databases
# ==============================================================================


class TestMultiDbExtract:
    """Tests for extraction and computation across databases."""

    def test_gextract_across_databases(self, tmp_path):
        """gextract works with tracks from working db and loaded datasets."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 5000)
        pm.gtrack_create_sparse("working_track", "working", intervs, [10.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "dataset", intervs, [20.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        result = pm.gextract(
            ["working_track", "ds_track"],
            pm.gintervals("1", 0, 1000),
            iterator=100,
        )

        assert "working_track" in result.columns
        assert "ds_track" in result.columns
        assert result["working_track"].iloc[0] == 10.0
        assert result["ds_track"].iloc[0] == 20.0

    def test_track_expressions_across_databases(self, tmp_path):
        """Track expressions using tracks from different databases."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 5000)
        pm.gtrack_create_sparse("x", "working", intervs, [10.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("y", "dataset", intervs, [3.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        result = pm.gextract("x + y", pm.gintervals("1", 0, 1000), iterator=100)
        assert result["x + y"].iloc[0] == 13.0

    def test_gscreen_across_databases(self, tmp_path):
        """gscreen works with tracks from different databases."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 5000)
        pm.gtrack_create_sparse("working_track", "working", intervs, [10.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "dataset", intervs, [5.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        # Screen using track from working db
        result1 = pm.gscreen("working_track > 5", pm.gintervals("1", 0, 1000))
        assert len(result1) > 0

        # Screen using track from dataset
        result2 = pm.gscreen("ds_track > 3", pm.gintervals("1", 0, 1000))
        assert len(result2) > 0

        # Screen using expression combining both
        result3 = pm.gscreen(
            "working_track + ds_track > 10",
            pm.gintervals("1", 0, 1000),
            iterator=100,
        )
        assert len(result3) > 0

    def test_gsummary_across_databases(self, tmp_path):
        """gsummary works with tracks from different databases."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 5000)
        pm.gtrack_create_sparse("working_track", "working", intervs, [10.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "dataset", intervs, [5.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        sum1 = pm.gsummary("working_track", pm.gintervals("1", 0, 1000))
        assert sum1["Sum"] == 10.0

        sum2 = pm.gsummary("ds_track", pm.gintervals("1", 0, 1000))
        assert sum2["Sum"] == 5.0


# ==============================================================================
# TestMultiDbVirtualTracks: virtual tracks across databases
# ==============================================================================


class TestMultiDbVirtualTracks:
    """Tests for virtual tracks referencing tracks from multiple databases."""

    def test_virtual_tracks_reference_any_database(self, tmp_path):
        """Virtual tracks can reference tracks from any database."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 5000)
        pm.gtrack_create_sparse("working_track", "working", intervs, [10.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track", "dataset", intervs, [20.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        pm.gvtrack_create("vt_working", "working_track", "avg")
        pm.gvtrack_create("vt_ds", "ds_track", "avg")
        try:
            result1 = pm.gextract("vt_working", pm.gintervals("1", 0, 1000), iterator=1000)
            result2 = pm.gextract("vt_ds", pm.gintervals("1", 0, 1000), iterator=1000)

            assert result1["vt_working"].iloc[0] == 10.0
            assert result2["vt_ds"].iloc[0] == 20.0
        finally:
            pm.gvtrack_rm("vt_working")
            pm.gvtrack_rm("vt_ds")

    def test_virtual_tracks_survive_dataset_load(self, tmp_path):
        """Virtual tracks remain accessible after loading a dataset."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("base_track", "base", intervs, [1.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")

        pm.gvtrack_create("vtrack", "base_track", "avg")
        try:
            pm.gdataset_load(db_ds)
            assert "vtrack" in pm.gvtrack_ls()
        finally:
            pm.gvtrack_rm("vtrack")


# ==============================================================================
# TestMultiDbIntervals: interval operations across databases
# ==============================================================================


class TestMultiDbIntervals:
    """Tests for interval operations across databases."""

    def test_gintervals_visible_from_both_databases(self, tmp_path):
        """Interval sets from both databases should be visible."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        int1 = pm.gintervals("1", 0, 1000)
        pm.gintervals_save(int1, "working_intervals")

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        int2 = pm.gintervals("1", 500, 1500)
        pm.gintervals_save(int2, "ds_intervals")

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        all_intervals = pm.gintervals_ls()
        assert "working_intervals" in all_intervals
        assert "ds_intervals" in all_intervals

    def test_gintervals_dataset_resolves_correctly(self, tmp_path):
        """gintervals_dataset returns the correct DB path for each interval set."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        int1 = pm.gintervals("1", 0, 1000)
        pm.gintervals_save(int1, "working_intervals")

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        int2 = pm.gintervals("1", 500, 1500)
        pm.gintervals_save(int2, "ds_intervals")

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        assert pm.gintervals_dataset("working_intervals") == str(
            Path(db_work).resolve()
        )
        assert pm.gintervals_dataset("ds_intervals") == str(
            Path(db_ds).resolve()
        )


# ==============================================================================
# TestMultiDbReload: database reload behavior
# ==============================================================================


class TestMultiDbReload:
    """Tests for database reload with loaded datasets."""

    def test_gdb_reload_preserves_loaded_datasets(self, tmp_path):
        """gdb_reload correctly refreshes state and preserves loaded datasets."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track1", "t1", intervs, [1.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("ds_track1", "ds1", intervs, [2.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds)

        tracks_before = pm.gtrack_ls()
        assert "track1" in tracks_before
        assert "ds_track1" in tracks_before

        pm.gdb_reload()

        tracks_after = pm.gtrack_ls()
        assert "track1" in tracks_after
        assert "ds_track1" in tracks_after


# ==============================================================================
# TestMultiDbMultipleDatasets: three or more datasets
# ==============================================================================


class TestMultiDbMultipleDatasets:
    """Tests for loading multiple datasets."""

    def test_three_datasets_loaded(self, tmp_path):
        """Three or more datasets can be loaded simultaneously."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track_a", "from working db", intervs, [1.0])

        db_ds1 = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds1)
        pm.gtrack_create_sparse("track_b", "from ds1", intervs, [2.0])

        db_ds2 = _create_linked_db(tmp_path, "dataset2")
        pm.gsetroot(db_ds2)
        pm.gtrack_create_sparse("track_c", "from ds2", intervs, [3.0])

        db_ds3 = _create_linked_db(tmp_path, "dataset3")
        pm.gsetroot(db_ds3)
        pm.gtrack_create_sparse("track_d", "from ds3", intervs, [4.0])

        # Connect working db + three datasets
        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds1)
        pm.gdataset_load(db_ds2)
        pm.gdataset_load(db_ds3)

        assert len(pm.gdataset_ls()) == 3
        tracks = pm.gtrack_ls()
        assert "track_a" in tracks
        assert "track_b" in tracks
        assert "track_c" in tracks
        assert "track_d" in tracks

    def test_track_dataset_resolves_to_correct_db(self, tmp_path):
        """gtrack_dataset resolves each track to its source DB."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track_a", "from working db", intervs, [1.0])

        db_ds1 = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds1)
        pm.gtrack_create_sparse("track_b", "from ds1", intervs, [2.0])

        db_ds2 = _create_linked_db(tmp_path, "dataset2")
        pm.gsetroot(db_ds2)
        pm.gtrack_create_sparse("track_c", "from ds2", intervs, [3.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds1)
        pm.gdataset_load(db_ds2)

        assert pm.gtrack_dataset("track_a") == str(Path(db_work).resolve())
        assert pm.gtrack_dataset("track_b") == str(Path(db_ds1).resolve())
        assert pm.gtrack_dataset("track_c") == str(Path(db_ds2).resolve())


# ==============================================================================
# TestMultiDbCollisions: collision / shadowing behavior
# ==============================================================================


class TestMultiDbCollisions:
    """Tests for track name collision handling across databases."""

    def test_working_db_wins_on_collision(self, tmp_path):
        """Working DB track takes priority over colliding dataset track."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("shared", "from working db", intervs, [100.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("shared", "from dataset", intervs, [200.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds, force=True)

        result = pm.gextract("shared", pm.gintervals("1", 0, 500))
        assert result["shared"].iloc[0] == 100.0
        assert pm.gtrack_dataset("shared") == str(Path(db_work).resolve())

    def test_collision_counts_in_gdataset_ls(self, tmp_path):
        """Track collision counts are properly reflected."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("unique_working", "working", intervs, [1.0])
        pm.gtrack_create_sparse("shared", "shared from working", intervs, [10.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("unique_ds", "ds", intervs, [2.0])
        pm.gtrack_create_sparse("shared", "shared from ds", intervs, [20.0])

        pm.gsetroot(db_work)
        pm.gdataset_load(db_ds, force=True)

        # Total visible tracks should be 3 (unique_working, shared from working, unique_ds)
        tracks = pm.gtrack_ls()
        assert "unique_working" in tracks
        assert "unique_ds" in tracks
        assert "shared" in tracks
        assert len([t for t in tracks if t in ("unique_working", "unique_ds", "shared")]) == 3


# ==============================================================================
# TestMultiDbPaths: absolute and relative path handling
# ==============================================================================


class TestMultiDbPaths:
    """Tests for path handling in multi-DB context."""

    def test_absolute_paths_work(self, tmp_path):
        """Absolute paths work correctly for gdataset_load."""
        db_work = _create_linked_db(tmp_path, "working_db")
        pm.gsetroot(db_work)
        intervs = pm.gintervals("1", 0, 1000)
        pm.gtrack_create_sparse("track1", "t1", intervs, [1.0])

        db_ds = _create_linked_db(tmp_path, "dataset1")
        pm.gsetroot(db_ds)
        pm.gtrack_create_sparse("track2", "t2", intervs, [2.0])

        pm.gsetroot(db_work)

        # Load with absolute path
        abs_path = str(Path(db_ds).resolve())
        pm.gdataset_load(abs_path)

        tracks = pm.gtrack_ls()
        assert "track1" in tracks
        assert "track2" in tracks


# ==============================================================================
# TestMultiDbErrors: error handling
# ==============================================================================


class TestMultiDbErrors:
    """Tests for error handling in multi-DB operations."""

    def test_gsetroot_nonexistent_directory(self):
        """gsetroot gives a clear error when directory does not exist."""
        with pytest.raises((FileNotFoundError, Exception)):
            pm.gsetroot("/this/path/does/not/exist")

    def test_gsetroot_missing_tracks_dir(self, tmp_path):
        """gsetroot gives a clear error when tracks/ is missing."""
        bad_db = tmp_path / "not_a_db"
        bad_db.mkdir()
        (bad_db / "seq").mkdir()
        (bad_db / "chrom_sizes.txt").write_text("chr1\t1000\n")

        with pytest.raises((FileNotFoundError, Exception)):
            pm.gsetroot(str(bad_db))

    def test_gsetroot_missing_seq_dir(self, tmp_path):
        """gsetroot gives a clear error when seq/ is missing."""
        bad_db = tmp_path / "not_a_db"
        bad_db.mkdir()
        (bad_db / "tracks").mkdir()
        (bad_db / "chrom_sizes.txt").write_text("chr1\t1000\n")

        with pytest.raises((FileNotFoundError, Exception)):
            pm.gsetroot(str(bad_db))

    def test_gsetroot_missing_chrom_sizes(self, tmp_path):
        """gsetroot gives a clear error when chrom_sizes.txt is missing."""
        bad_db = tmp_path / "not_a_db"
        bad_db.mkdir()
        (bad_db / "tracks").mkdir()
        (bad_db / "seq").mkdir()

        with pytest.raises((FileNotFoundError, Exception)):
            pm.gsetroot(str(bad_db))
