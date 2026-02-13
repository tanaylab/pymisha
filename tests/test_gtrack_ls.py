"""
Tests for gtrack_ls() with regex/pattern support.

These tests verify that gtrack_ls() can filter tracks by:
1. Pattern matching on track names (regex)
2. Attribute filtering (e.g., created_by="pattern")
3. Options like ignore_case
"""
import pytest

import pymisha as pm

# Path to test database
TESTDB = "tests/testdb/trackdb/test"


class TestGtrackLsBasic:
    """Basic tests for gtrack_ls()."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gtrack_ls_returns_all_tracks(self):
        """gtrack_ls() without arguments returns all tracks."""
        tracks = pm.gtrack_ls()
        assert tracks is not None
        assert isinstance(tracks, list)
        assert len(tracks) > 0
        # We know these tracks exist in test db
        assert "dense_track" in tracks
        assert "sparse_track" in tracks

    def test_gtrack_ls_with_pattern_filters_tracks(self):
        """gtrack_ls() with pattern filters track names."""
        tracks = pm.gtrack_ls("dense")
        assert tracks is not None
        assert isinstance(tracks, list)
        # Should only include tracks matching "dense"
        for t in tracks:
            assert "dense" in t.lower()

    def test_gtrack_ls_with_regex_pattern(self):
        """gtrack_ls() supports regex patterns."""
        # Use regex to match tracks starting with 'd' or 's'
        tracks = pm.gtrack_ls("^[ds]")
        assert tracks is not None
        for t in tracks:
            assert t[0] in ['d', 's'], f"Track '{t}' doesn't start with d or s"

    def test_gtrack_ls_with_no_matches_returns_empty(self):
        """gtrack_ls() with non-matching pattern returns empty list."""
        tracks = pm.gtrack_ls("nonexistent_pattern_xyz")
        # Should return empty list or None
        assert tracks is None or len(tracks) == 0

    def test_gtrack_ls_ignore_case(self):
        """gtrack_ls() respects ignore_case parameter."""
        # Case-sensitive search
        tracks_sensitive = pm.gtrack_ls("DENSE", ignore_case=False)
        # Should not find anything (tracks are lowercase)
        assert tracks_sensitive is None or len(tracks_sensitive) == 0

        # Case-insensitive search
        tracks_insensitive = pm.gtrack_ls("DENSE", ignore_case=True)
        # Should find dense tracks
        assert tracks_insensitive is not None
        assert len(tracks_insensitive) > 0


class TestGtrackLsAttributeFilter:
    """Tests for gtrack_ls() attribute filtering."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gtrack_ls_filter_by_attribute(self):
        """gtrack_ls() can filter by track attribute."""
        # Filter by created.by attribute
        tracks = pm.gtrack_ls(created_by="immaculate")
        assert tracks is not None
        # dense_track has created.by = "immaculate conception"
        assert "dense_track" in tracks

    def test_gtrack_ls_combined_name_and_attribute_filter(self):
        """gtrack_ls() can combine name pattern and attribute filter."""
        # Filter by both name pattern and attribute
        tracks = pm.gtrack_ls("dense", created_by="immaculate")
        assert tracks is not None
        assert "dense_track" in tracks

    def test_gtrack_ls_attribute_filter_no_match(self):
        """gtrack_ls() with non-matching attribute returns empty."""
        tracks = pm.gtrack_ls(created_by="nonexistent_creator_xyz")
        assert tracks is None or len(tracks) == 0


class TestGtrackInfo:
    """Tests for gtrack_info() attribute loading."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gtrack_info_loads_attributes(self):
        """gtrack_info() loads attributes from binary .attributes file."""
        info = pm.gtrack_info("dense_track")
        assert 'attributes' in info
        attrs = info['attributes']
        # We know dense_track has created.by attribute
        assert 'created.by' in attrs
        assert 'immaculate' in attrs['created.by']

    def test_gtrack_info_returns_type(self):
        """gtrack_info() returns track type."""
        info = pm.gtrack_info("dense_track")
        assert info['type'] == 'dense'

        info = pm.gtrack_info("sparse_track")
        assert info['type'] == 'sparse'


class TestGtrackLsGoldenMaster:
    """Golden-master tests comparing gtrack_ls with R misha."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize the database before each test."""
        pm.gdb_init(TESTDB)

    def test_gtrack_ls_matches_r_basic(self):
        """gtrack_ls() basic output matches R."""
        import os
        import subprocess
        import tempfile

        # Get pymisha result
        py_tracks = pm.gtrack_ls()

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
tracks <- gtrack.ls()
cat(paste(tracks, collapse="\\n"))
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            f.write(r_code)
            script_path = f.name

        try:
            result = subprocess.run(
                ['R', '--quiet', '--no-save', '-f', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            r_output = result.stdout.strip()
            # Parse R output - last lines should be the track names
            # Clean up R output artifacts like '>' and '[1]' prefixes
            r_tracks = []
            for line in r_output.split('\n'):
                line = line.strip()
                # Skip empty lines, R prompts, and bracket prefixes
                if not line or line.startswith(('>', '[')):
                    continue
                # Remove any trailing '>' from R prompt artifacts
                line = line.rstrip('>')
                if line:
                    r_tracks.append(line)
        finally:
            os.unlink(script_path)

        # Compare
        py_set = set(py_tracks) if py_tracks else set()
        r_set = set(r_tracks) if r_tracks else set()

        assert py_set == r_set, f"Track lists differ:\npymisha: {py_set}\nR: {r_set}"

    def test_gtrack_ls_with_pattern_matches_r(self):
        """gtrack_ls() with pattern matches R output."""
        import os
        import subprocess
        import tempfile

        pattern = "dense"

        # Get pymisha result
        py_tracks = pm.gtrack_ls(pattern)

        # Get R result
        r_code = f'''
library(misha)
gdb.init("{TESTDB}")
tracks <- gtrack.ls("{pattern}")
cat(paste(tracks, collapse="\\n"))
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            f.write(r_code)
            script_path = f.name

        try:
            result = subprocess.run(
                ['R', '--quiet', '--no-save', '-f', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            r_output = result.stdout.strip()
            # Parse R output - clean up R artifacts
            r_tracks = []
            for line in r_output.split('\n'):
                line = line.strip()
                if not line or line.startswith(('>', '[')):
                    continue
                line = line.rstrip('>')
                if line:
                    r_tracks.append(line)
        finally:
            os.unlink(script_path)

        # Compare
        py_set = set(py_tracks) if py_tracks else set()
        r_set = set(r_tracks) if r_tracks else set()

        assert py_set == r_set, f"Track lists differ for pattern '{pattern}':\npymisha: {py_set}\nR: {r_set}"
