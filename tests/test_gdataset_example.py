"""Tests for gdataset_example_path (GAP-035)."""

import os
from pathlib import Path

import pytest

import pymisha as pm


class TestGdatasetExamplePath:
    """Tests for gdataset_example_path."""

    def test_returns_string(self):
        """gdataset_example_path returns a non-empty string."""
        result = pm.gdataset_example_path()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_path_exists(self):
        """The returned path exists as a directory on disk."""
        result = pm.gdataset_example_path()
        assert os.path.isdir(result)

    def test_is_valid_dataset(self):
        """The returned path is a valid dataset with expected structure."""
        result = pm.gdataset_example_path()
        ds = Path(result)
        assert (ds / "tracks").is_dir()
        assert (ds / "chrom_sizes.txt").is_file()
        assert (ds / "misha.yaml").is_file()

    def test_contains_track(self):
        """The example dataset contains at least one track."""
        result = pm.gdataset_example_path()
        info = pm.gdataset_info(result)
        assert info["track_count"] >= 1

    def test_contains_intervals(self):
        """The example dataset contains at least one interval set."""
        result = pm.gdataset_example_path()
        info = pm.gdataset_info(result)
        assert info["interval_count"] >= 1

    def test_can_load_and_unload(self):
        """The example dataset can be loaded and unloaded."""
        result = pm.gdataset_example_path()
        load_result = pm.gdataset_load(result)
        assert load_result["tracks"] >= 1
        pm.gdataset_unload(result, validate=True)

    def test_description_set(self):
        """The dataset metadata includes a description."""
        result = pm.gdataset_example_path()
        info = pm.gdataset_info(result)
        assert info["description"] is not None
        assert len(info["description"]) > 0

    def test_genome_matches_example_db(self):
        """The dataset genome hash matches the example DB genome hash."""
        result = pm.gdataset_example_path()
        # After gdataset_example_path, gdb_init_examples has been called
        # so the working DB is the example DB — loading should succeed
        # (which it only does if genomes match)
        load_result = pm.gdataset_load(result)
        assert load_result["tracks"] >= 1
        pm.gdataset_unload(result)
