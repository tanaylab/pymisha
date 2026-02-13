"""Tests for interval set management functions."""

import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize the example database for each test."""
    pm.gdb_init_examples()
    yield


class TestGintervalsLs:
    """Tests for gintervals_ls function."""

    def test_gintervals_ls_returns_list(self):
        """Test that gintervals_ls returns a list of interval sets."""
        result = pm.gintervals_ls()
        assert result is not None
        assert isinstance(result, list)

    def test_gintervals_ls_with_pattern(self):
        """Test gintervals_ls with a pattern filter."""
        all_sets = pm.gintervals_ls()

        if len(all_sets) > 0:
            # Get first character of first set
            first_char = all_sets[0][0]
            pattern = f"^{first_char}"
            filtered = pm.gintervals_ls(pattern)
            assert all(s.startswith(first_char) for s in filtered)


class TestGintervalsExists:
    """Tests for gintervals_exists function."""

    def test_gintervals_exists_returns_bool(self):
        """Test that gintervals_exists returns a boolean."""
        result = pm.gintervals_exists("nonexistent_set_12345")
        assert result is False

    def test_gintervals_exists_with_existing(self):
        """Test gintervals_exists with an existing set."""
        all_sets = pm.gintervals_ls()
        if len(all_sets) > 0:
            result = pm.gintervals_exists(all_sets[0])
            assert result is True


class TestGintervalsChromSizes:
    """Tests for gintervals_chrom_sizes function."""

    def test_gintervals_chrom_sizes_from_intervals(self):
        """Test getting chrom sizes from intervals."""
        intervals = pm.gintervals(["1", "2"], [0, 0], [10000, 20000])
        result = pm.gintervals_chrom_sizes(intervals)

        assert result is not None
        assert "chrom" in result.columns
        # Should have chromosomes 1 and 2
        assert len(result) == 2
