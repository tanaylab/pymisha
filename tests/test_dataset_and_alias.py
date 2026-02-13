import hashlib
import shutil
from pathlib import Path

import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def test_chrom_alias_chr_prefix():
    intervals = pm.gintervals("chr1", 0, 100)
    assert intervals.iloc[0]["chrom"] == "1"


def test_dataset_load_force_and_resolution(tmp_path):
    dataset_root = _copy_db(tmp_path)

    # Add a unique track to dataset
    src = dataset_root / "tracks" / "dense_track.track"
    dst = dataset_root / "tracks" / "dataset_dense.track"
    if not dst.exists():
        shutil.copytree(src, dst)

    # Add a unique interval set to dataset
    interv_src = dataset_root / "tracks" / "annotations.interv"
    interv_dst = dataset_root / "tracks" / "dataset_annotations.interv"
    if not interv_dst.exists():
        shutil.copy2(interv_src, interv_dst)

    with pytest.raises(ValueError):
        pm.gdataset_load(str(dataset_root), force=False)

    pm.gdataset_load(str(dataset_root), force=True)

    tracks = pm.gtrack_ls()
    assert "dataset_dense" in tracks
    assert pm.gtrack_dataset("dataset_dense") == str(dataset_root)

    # Working db should win collisions
    assert pm.gtrack_dataset("dense_track") == str(TEST_DB)

    assert pm.gintervals_dataset("dataset_annotations") == str(dataset_root)
    assert pm.gintervals_dataset("annotations") == str(TEST_DB)

    pm.gdataset_unload(str(dataset_root), validate=True)
    assert "dataset_dense" not in pm.gtrack_ls()


def test_dataset_save_and_info_roundtrip(tmp_path):
    dataset_path = tmp_path / "saved_dataset"
    out = pm.gdataset_save(
        path=str(dataset_path),
        description="Dataset for unit tests",
        tracks=["dense_track"],
        intervals=["annotations"],
        symlinks=False,
        copy_seq=True,
    )
    assert out == str(dataset_path.resolve())

    assert (dataset_path / "tracks" / "dense_track.track").is_dir()
    assert (dataset_path / "tracks" / "annotations.interv").exists()
    assert (dataset_path / "seq").is_dir()
    assert not (dataset_path / "seq").is_symlink()
    assert (dataset_path / "misha.yaml").is_file()

    info = pm.gdataset_info(str(dataset_path))
    assert info["description"] == "Dataset for unit tests"
    assert info["track_count"] == 1
    assert info["interval_count"] == 1
    assert info["is_loaded"] is False

    with open(dataset_path / "chrom_sizes.txt", "rb") as f:
        expected_genome = hashlib.sha256(f.read()).hexdigest()
    assert info["genome"] == expected_genome

    pm.gdataset_load(str(dataset_path), force=True)
    assert pm.gdataset_info(str(dataset_path))["is_loaded"] is True
    pm.gdataset_unload(str(dataset_path), validate=True)
    assert pm.gdataset_info(str(dataset_path))["is_loaded"] is False


def test_dataset_save_validation(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        pm.gdataset_save(
            path=str(tmp_path / "x"),
            description="bad",
            tracks=None,
            intervals=None,
        )

    with pytest.raises(ValueError, match="Track 'no_such_track' does not exist"):
        pm.gdataset_save(
            path=str(tmp_path / "x2"),
            description="bad",
            tracks=["no_such_track"],
        )

    with pytest.raises(ValueError, match="Interval set 'no_such_intervals' does not exist"):
        pm.gdataset_save(
            path=str(tmp_path / "x3"),
            description="bad",
            intervals=["no_such_intervals"],
        )


# ==============================================================================
# Helpers for comprehensive dataset tests
# ==============================================================================


def _make_dataset(tmp_path: Path, name: str, track_names=None, interval_names=None):
    """Create a minimal dataset directory under tmp_path with given tracks/intervals.

    Tracks are copied from the test DB's dense_track.track (renamed).
    Intervals are copied from the test DB's annotations.interv (renamed).
    Returns the absolute path to the created dataset.
    """
    ds = tmp_path / name
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "tracks").mkdir(exist_ok=True)
    # Copy chrom_sizes.txt from the test DB
    shutil.copy2(TEST_DB / "chrom_sizes.txt", ds / "chrom_sizes.txt")
    # Symlink seq directory
    if not (ds / "seq").exists():
        (ds / "seq").symlink_to(TEST_DB / "seq")

    if track_names:
        for tname in track_names:
            src = TEST_DB / "tracks" / "dense_track.track"
            dst = ds / "tracks" / f"{tname}.track"
            if not dst.exists():
                shutil.copytree(src, dst)

    if interval_names:
        for iname in interval_names:
            src = TEST_DB / "tracks" / "annotations.interv"
            dst = ds / "tracks" / f"{iname}.interv"
            if not dst.exists():
                shutil.copy2(src, dst)

    return str(ds)


# ==============================================================================
# gdataset_load: basic loading
# ==============================================================================


def test_dataset_load_tracks_from_dataset(tmp_path):
    """gdataset_load loads tracks from a dataset and returns correct counts."""
    ds_path = _make_dataset(tmp_path, "ds1", track_names=["ds_only_track"])
    try:
        result = pm.gdataset_load(ds_path)
        assert "ds_only_track" in pm.gtrack_ls()
        assert result["tracks"] == 1
        assert result["intervals"] == 0
        assert result["shadowed_tracks"] == 0
        assert result["shadowed_intervals"] == 0
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_intervals_from_dataset(tmp_path):
    """gdataset_load loads intervals from a dataset."""
    ds_path = _make_dataset(tmp_path, "ds1", interval_names=["ds_only_intervals"])
    try:
        result = pm.gdataset_load(ds_path)
        assert "ds_only_intervals" in pm.gintervals_ls()
        assert result["tracks"] == 0
        assert result["intervals"] == 1
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_tracks_and_intervals(tmp_path):
    """gdataset_load loads both tracks and intervals from a dataset."""
    ds_path = _make_dataset(
        tmp_path, "ds1",
        track_names=["ds_trk"],
        interval_names=["ds_ivl"],
    )
    try:
        result = pm.gdataset_load(ds_path)
        assert "ds_trk" in pm.gtrack_ls()
        assert "ds_ivl" in pm.gintervals_ls()
        assert result["tracks"] == 1
        assert result["intervals"] == 1
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_reload_idempotent(tmp_path):
    """Loading a dataset that is already loaded unloads and reloads it."""
    ds_path = _make_dataset(tmp_path, "ds1", track_names=["reload_track"])
    try:
        pm.gdataset_load(ds_path)
        assert "reload_track" in pm.gtrack_ls()
        # Load again - should be idempotent
        pm.gdataset_load(ds_path)
        assert "reload_track" in pm.gtrack_ls()
        # Should appear only once in the dataset list
        assert pm.gdataset_ls().count(str(Path(ds_path).resolve())) == 1
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_working_db_is_noop(tmp_path):
    """Loading the working DB path as a dataset is a silent no-op."""
    result = pm.gdataset_load(str(TEST_DB))
    assert result["tracks"] == 0
    assert result["intervals"] == 0
    assert result["shadowed_tracks"] == 0
    assert result["shadowed_intervals"] == 0


def test_dataset_load_normalizes_paths(tmp_path):
    """Loading with relative vs absolute paths resolves to the same dataset."""
    ds_path = _make_dataset(tmp_path, "ds1", track_names=["norm_track"])
    try:
        pm.gdataset_load(ds_path)
        norm = str(Path(ds_path).resolve())
        assert norm in pm.gdataset_ls()
        # Reload with resolved path should not error
        pm.gdataset_load(norm)
        assert "norm_track" in pm.gtrack_ls()
    finally:
        pm.gdataset_unload(ds_path, validate=False)


# ==============================================================================
# gdataset_load: error cases
# ==============================================================================


def test_dataset_load_nonexistent_path_errors():
    """gdataset_load raises on non-existent path."""
    with pytest.raises(ValueError, match="does not exist"):
        pm.gdataset_load("/nonexistent/path/to/dataset")


def test_dataset_load_missing_tracks_dir_errors(tmp_path):
    """gdataset_load raises when path has no tracks/ directory."""
    ds = tmp_path / "not_a_db"
    ds.mkdir()
    with pytest.raises(ValueError, match="tracks"):
        pm.gdataset_load(str(ds))


def test_dataset_load_missing_chrom_sizes_errors(tmp_path):
    """gdataset_load raises when chrom_sizes.txt is missing."""
    ds = tmp_path / "no_chrom"
    ds.mkdir()
    (ds / "tracks").mkdir()
    with pytest.raises(ValueError, match="chrom_sizes"):
        pm.gdataset_load(str(ds))


def test_dataset_load_genome_mismatch_errors(tmp_path):
    """gdataset_load raises when genome (chrom_sizes.txt) does not match working DB."""
    ds = tmp_path / "bad_genome"
    ds.mkdir()
    (ds / "tracks").mkdir()
    # Write a different chrom_sizes.txt
    (ds / "chrom_sizes.txt").write_text("chrFAKE\t99999\n")
    with pytest.raises(ValueError, match="genome.*match"):
        pm.gdataset_load(str(ds))


# ==============================================================================
# Collision handling
# ==============================================================================


def test_dataset_load_collision_with_working_db_tracks(tmp_path):
    """gdataset_load detects track name collision with working DB."""
    # dense_track already exists in the working DB
    ds_path = _make_dataset(tmp_path, "ds_collision", track_names=["dense_track"])
    try:
        with pytest.raises(ValueError, match="[Cc]annot load|collision"):
            pm.gdataset_load(ds_path, force=False)
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_collision_with_working_db_intervals(tmp_path):
    """gdataset_load detects interval name collision with working DB."""
    # annotations already exists in the working DB
    ds_path = _make_dataset(tmp_path, "ds_collision_ivl", interval_names=["annotations"])
    try:
        with pytest.raises(ValueError, match="[Cc]annot load|collision"):
            pm.gdataset_load(ds_path, force=False)
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_force_working_db_wins_tracks(tmp_path):
    """With force=True, working DB wins for colliding tracks."""
    ds_path = _make_dataset(tmp_path, "ds_force", track_names=["dense_track", "ds_unique"])
    try:
        result = pm.gdataset_load(ds_path, force=True)
        # Working DB should win for dense_track
        assert pm.gtrack_dataset("dense_track") == str(TEST_DB)
        # Dataset unique track should be visible
        assert "ds_unique" in pm.gtrack_ls()
        assert result["shadowed_tracks"] >= 1
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_force_working_db_wins_intervals(tmp_path):
    """With force=True, working DB wins for colliding intervals."""
    ds_path = _make_dataset(
        tmp_path, "ds_force_ivl",
        interval_names=["annotations", "ds_unique_ivl"],
    )
    try:
        result = pm.gdataset_load(ds_path, force=True)
        # Working DB should win for annotations
        assert pm.gintervals_dataset("annotations") == str(TEST_DB)
        assert "ds_unique_ivl" in pm.gintervals_ls()
        assert result["shadowed_intervals"] >= 1
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_dataset_load_dataset_to_dataset_collision(tmp_path):
    """gdataset_load detects collision between two loaded datasets."""
    ds1_path = _make_dataset(tmp_path, "ds1", track_names=["shared_ds_track"])
    ds2_path = _make_dataset(tmp_path, "ds2", track_names=["shared_ds_track"])
    try:
        pm.gdataset_load(ds1_path)
        with pytest.raises(ValueError, match="[Cc]annot load|collision"):
            pm.gdataset_load(ds2_path, force=False)
    finally:
        pm.gdataset_unload(ds1_path, validate=False)
        pm.gdataset_unload(ds2_path, validate=False)


def test_dataset_load_force_later_dataset_wins(tmp_path):
    """With force=True, later-loaded dataset overrides earlier for shared names."""
    ds1_path = _make_dataset(tmp_path, "ds1", track_names=["shared_ds_trk2"])
    ds2_path = _make_dataset(tmp_path, "ds2", track_names=["shared_ds_trk2"])
    try:
        pm.gdataset_load(ds1_path)
        result = pm.gdataset_load(ds2_path, force=True)
        # ds2 should own the track now (later dataset wins via C++ search order)
        ds2_norm = str(Path(ds2_path).resolve())
        assert pm.gtrack_dataset("shared_ds_trk2") == ds2_norm
        # The load result counts the collision as "shadowed" even though ds2
        # actually wins at the C++ layer (datasets are searched last-to-first).
        # The important assertion is that ds2's track resolves correctly above.
        assert result["shadowed_tracks"] == 1
    finally:
        pm.gdataset_unload(ds1_path, validate=False)
        pm.gdataset_unload(ds2_path, validate=False)


# ==============================================================================
# gdataset_unload
# ==============================================================================


def test_dataset_unload_removes_tracks(tmp_path):
    """gdataset_unload removes the dataset's tracks from visibility."""
    ds_path = _make_dataset(tmp_path, "ds_unload", track_names=["unload_trk"])
    pm.gdataset_load(ds_path)
    assert "unload_trk" in pm.gtrack_ls()
    pm.gdataset_unload(ds_path)
    assert "unload_trk" not in pm.gtrack_ls()


def test_dataset_unload_removes_intervals(tmp_path):
    """gdataset_unload removes the dataset's intervals from visibility."""
    ds_path = _make_dataset(tmp_path, "ds_unload_ivl", interval_names=["unload_ivl"])
    pm.gdataset_load(ds_path)
    assert "unload_ivl" in pm.gintervals_ls()
    pm.gdataset_unload(ds_path)
    assert "unload_ivl" not in pm.gintervals_ls()


def test_dataset_unload_path_normalization(tmp_path):
    """gdataset_unload works with different path representations."""
    ds_path = _make_dataset(tmp_path, "ds_norm_unload", track_names=["norm_unload_trk"])
    pm.gdataset_load(ds_path)
    # Unload with the resolved absolute path
    abs_path = str(Path(ds_path).resolve())
    pm.gdataset_unload(abs_path)
    assert "norm_unload_trk" not in pm.gtrack_ls()


def test_dataset_unload_validate_false_ignores_nonloaded():
    """gdataset_unload with validate=False silently ignores non-loaded paths."""
    pm.gdataset_unload("/nonexistent/path", validate=False)


def test_dataset_unload_validate_true_errors_nonloaded():
    """gdataset_unload with validate=True raises if the dataset is not loaded."""
    with pytest.raises(ValueError, match="not loaded"):
        pm.gdataset_unload("/nonexistent/path", validate=True)


def test_dataset_unload_shadowed_by_working_db_still_visible(tmp_path):
    """After unloading a force-loaded dataset, working DB tracks remain visible."""
    ds_path = _make_dataset(tmp_path, "ds_shadow", track_names=["dense_track"])
    pm.gdataset_load(ds_path, force=True)
    # Working DB wins, so dense_track should still resolve to TEST_DB
    assert pm.gtrack_dataset("dense_track") == str(TEST_DB)
    pm.gdataset_unload(ds_path)
    # After unload, working DB track still visible
    assert "dense_track" in pm.gtrack_ls()
    assert pm.gtrack_dataset("dense_track") == str(TEST_DB)


def test_dataset_unload_restores_shadowed_in_load_order(tmp_path):
    """Unloading a later dataset restores the earlier dataset's track."""
    ds1_path = _make_dataset(tmp_path, "ds_order1", track_names=["order_shared"])
    ds2_path = _make_dataset(tmp_path, "ds_order2", track_names=["order_shared"])
    try:
        pm.gdataset_load(ds1_path)
        ds1_norm = str(Path(ds1_path).resolve())
        assert pm.gtrack_dataset("order_shared") == ds1_norm

        pm.gdataset_load(ds2_path, force=True)
        ds2_norm = str(Path(ds2_path).resolve())
        # ds2 wins
        assert pm.gtrack_dataset("order_shared") == ds2_norm

        # Unload ds2 -> ds1 should become visible
        pm.gdataset_unload(ds2_path)
        assert pm.gtrack_dataset("order_shared") == ds1_norm
    finally:
        pm.gdataset_unload(ds1_path, validate=False)
        pm.gdataset_unload(ds2_path, validate=False)


# ==============================================================================
# gdataset_ls
# ==============================================================================


def test_dataset_ls_empty_initially():
    """gdataset_ls returns empty list when no datasets are loaded."""
    assert pm.gdataset_ls() == []


def test_dataset_ls_tracks_loaded_datasets(tmp_path):
    """gdataset_ls shows loaded datasets."""
    ds1_path = _make_dataset(tmp_path, "ds_ls1", track_names=["ls_trk1"])
    ds2_path = _make_dataset(tmp_path, "ds_ls2", track_names=["ls_trk2"])
    try:
        pm.gdataset_load(ds1_path)
        loaded = pm.gdataset_ls()
        assert len(loaded) == 1
        assert str(Path(ds1_path).resolve()) in loaded

        pm.gdataset_load(ds2_path)
        loaded = pm.gdataset_ls()
        assert len(loaded) == 2
        assert str(Path(ds1_path).resolve()) in loaded
        assert str(Path(ds2_path).resolve()) in loaded
    finally:
        pm.gdataset_unload(ds1_path, validate=False)
        pm.gdataset_unload(ds2_path, validate=False)


def test_dataset_ls_after_unload(tmp_path):
    """gdataset_ls removes unloaded datasets from the list."""
    ds_path = _make_dataset(tmp_path, "ds_ls_unload", track_names=["ls_unload_trk"])
    try:
        pm.gdataset_load(ds_path)
        assert len(pm.gdataset_ls()) == 1
        pm.gdataset_unload(ds_path)
        assert len(pm.gdataset_ls()) == 0
    finally:
        pm.gdataset_unload(ds_path, validate=False)


# ==============================================================================
# gdataset_save: comprehensive
# ==============================================================================


def test_dataset_save_creates_directory_structure(tmp_path):
    """gdataset_save creates the full dataset directory structure."""
    ds_path = tmp_path / "save_struct"
    pm.gdataset_save(
        path=str(ds_path),
        description="Structure test",
        tracks=["dense_track"],
    )
    assert ds_path.is_dir()
    assert (ds_path / "tracks").is_dir()
    assert (ds_path / "chrom_sizes.txt").is_file()
    assert (ds_path / "seq").exists()
    assert (ds_path / "misha.yaml").is_file()
    assert (ds_path / "tracks" / "dense_track.track").is_dir()


def test_dataset_save_with_intervals_only(tmp_path):
    """gdataset_save works with only intervals specified."""
    ds_path = tmp_path / "save_ivl"
    pm.gdataset_save(
        path=str(ds_path),
        description="Intervals only",
        intervals=["annotations"],
    )
    assert (ds_path / "tracks" / "annotations.interv").exists()


def test_dataset_save_with_tracks_and_intervals(tmp_path):
    """gdataset_save includes both tracks and intervals."""
    ds_path = tmp_path / "save_both"
    pm.gdataset_save(
        path=str(ds_path),
        description="Both",
        tracks=["dense_track"],
        intervals=["annotations"],
    )
    assert (ds_path / "tracks" / "dense_track.track").is_dir()
    assert (ds_path / "tracks" / "annotations.interv").exists()


def test_dataset_save_with_symlinks(tmp_path):
    """gdataset_save with symlinks=True creates symlinks for tracks."""
    ds_path = tmp_path / "save_sym"
    pm.gdataset_save(
        path=str(ds_path),
        description="Symlink test",
        tracks=["dense_track"],
        symlinks=True,
    )
    track_path = ds_path / "tracks" / "dense_track.track"
    assert track_path.is_symlink()


def test_dataset_save_seq_symlink_by_default(tmp_path):
    """gdataset_save creates a seq symlink by default (copy_seq=False)."""
    ds_path = tmp_path / "save_seq_sym"
    pm.gdataset_save(
        path=str(ds_path),
        description="Seq symlink",
        tracks=["dense_track"],
        symlinks=False,
        copy_seq=False,
    )
    assert (ds_path / "seq").is_symlink()


def test_dataset_save_copy_seq(tmp_path):
    """gdataset_save with copy_seq=True copies the seq directory."""
    ds_path = tmp_path / "save_seq_copy"
    pm.gdataset_save(
        path=str(ds_path),
        description="Seq copy",
        tracks=["dense_track"],
        copy_seq=True,
    )
    assert (ds_path / "seq").is_dir()
    assert not (ds_path / "seq").is_symlink()


def test_dataset_save_misha_yaml_content(tmp_path):
    """gdataset_save writes correct misha.yaml metadata."""
    ds_path = tmp_path / "save_yaml"
    pm.gdataset_save(
        path=str(ds_path),
        description="YAML test dataset",
        tracks=["dense_track"],
        intervals=["annotations"],
    )
    info = pm.gdataset_info(str(ds_path))
    assert info["description"] == "YAML test dataset"
    assert info["track_count"] == 1
    assert info["interval_count"] == 1
    assert info["genome"] is not None
    assert info["created"] is not None


def test_dataset_save_errors_when_path_exists(tmp_path):
    """gdataset_save raises when path already exists."""
    existing = tmp_path / "existing_dir"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        pm.gdataset_save(
            path=str(existing),
            description="bad",
            tracks=["dense_track"],
        )


# ==============================================================================
# gdataset_info: comprehensive
# ==============================================================================


def test_dataset_info_without_misha_yaml(tmp_path):
    """gdataset_info works for a dataset without misha.yaml (scan-based)."""
    ds_path = _make_dataset(tmp_path, "ds_no_yaml", track_names=["info_trk"])
    # No misha.yaml was created by _make_dataset
    info = pm.gdataset_info(ds_path)
    assert info["description"] is None
    assert info["author"] is None
    assert info["track_count"] == 1
    assert info["is_loaded"] is False


def test_dataset_info_is_loaded_transitions(tmp_path):
    """gdataset_info reflects is_loaded changes during load/unload."""
    ds_path = _make_dataset(tmp_path, "ds_loaded_info", track_names=["loaded_info_trk"])
    assert pm.gdataset_info(ds_path)["is_loaded"] is False

    pm.gdataset_load(ds_path)
    try:
        assert pm.gdataset_info(ds_path)["is_loaded"] is True
    finally:
        pm.gdataset_unload(ds_path, validate=False)

    assert pm.gdataset_info(ds_path)["is_loaded"] is False


def test_dataset_info_track_and_interval_counts(tmp_path):
    """gdataset_info correctly counts tracks and intervals."""
    ds_path = _make_dataset(
        tmp_path, "ds_counts",
        track_names=["cnt_trk1", "cnt_trk2"],
        interval_names=["cnt_ivl1"],
    )
    info = pm.gdataset_info(ds_path)
    assert info["track_count"] == 2
    assert info["interval_count"] == 1


# ==============================================================================
# gtrack_dataset / gintervals_dataset edge cases
# ==============================================================================


def test_gtrack_dataset_returns_none_for_nonexistent():
    """gtrack_dataset returns None for a track that does not exist."""
    result = pm.gtrack_dataset("totally_nonexistent_track_xyz")
    assert result is None


def test_gintervals_dataset_returns_none_for_nonexistent():
    """gintervals_dataset returns None for a non-existent interval set."""
    assert pm.gintervals_dataset("nonexistent_ivl_set") is None


def test_gtrack_dataset_for_dataset_track(tmp_path):
    """gtrack_dataset returns the dataset path for dataset tracks."""
    ds_path = _make_dataset(tmp_path, "ds_gtrack", track_names=["gtrack_ds_trk"])
    try:
        pm.gdataset_load(ds_path)
        result = pm.gtrack_dataset("gtrack_ds_trk")
        assert result == str(Path(ds_path).resolve())
    finally:
        pm.gdataset_unload(ds_path, validate=False)


def test_gintervals_dataset_for_dataset_intervals(tmp_path):
    """gintervals_dataset returns the dataset path for dataset intervals."""
    ds_path = _make_dataset(
        tmp_path, "ds_givl", interval_names=["givl_ds_ivl"],
    )
    try:
        pm.gdataset_load(ds_path)
        result = pm.gintervals_dataset("givl_ds_ivl")
        assert result == str(Path(ds_path).resolve())
    finally:
        pm.gdataset_unload(ds_path, validate=False)


# ==============================================================================
# Multiple datasets loaded simultaneously
# ==============================================================================


def test_multiple_datasets_simultaneous(tmp_path):
    """Multiple datasets can be loaded and all their unique tracks are visible."""
    paths = []
    for i in range(3):
        ds_path = _make_dataset(
            tmp_path, f"multi_ds{i}",
            track_names=[f"multi_trk_{i}"],
        )
        paths.append(ds_path)
    try:
        for p in paths:
            pm.gdataset_load(p)
        loaded = pm.gdataset_ls()
        assert len(loaded) == 3
        tracks = pm.gtrack_ls()
        for i in range(3):
            assert f"multi_trk_{i}" in tracks
    finally:
        for p in paths:
            pm.gdataset_unload(p, validate=False)


# ==============================================================================
# Dataset save -> load -> use workflow
# ==============================================================================


def test_dataset_save_load_use_roundtrip(tmp_path):
    """Full workflow: save a dataset, load it, and use the tracks."""
    # Save dataset from working DB
    ds_path = tmp_path / "workflow_ds"
    pm.gdataset_save(
        path=str(ds_path),
        description="Workflow test",
        tracks=["dense_track"],
        intervals=["annotations"],
    )

    # Load the dataset (will collide with working DB, use force)
    pm.gdataset_load(str(ds_path), force=True)
    try:
        assert str(ds_path.resolve()) in pm.gdataset_ls()
        info = pm.gdataset_info(str(ds_path))
        assert info["is_loaded"] is True
        assert info["description"] == "Workflow test"
    finally:
        pm.gdataset_unload(str(ds_path), validate=True)


# ==============================================================================
# Verbose mode
# ==============================================================================


def test_dataset_load_verbose(tmp_path, capsys):
    """gdataset_load with verbose=True prints loading information."""
    ds_path = _make_dataset(tmp_path, "ds_verbose", track_names=["verbose_trk"])
    try:
        pm.gdataset_load(ds_path, verbose=True)
        captured = capsys.readouterr()
        assert "Loaded" in captured.out or "dataset" in captured.out.lower()
    finally:
        pm.gdataset_unload(ds_path, validate=False)
