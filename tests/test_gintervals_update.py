"""Tests for gintervals_update."""

import pytest

import pymisha


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture()
def _test_iset():
    """Create a test intervals set with known content."""
    name = "test_update_iset"
    intervals = pymisha.gintervals(["1", "1", "2"], [0, 1000, 0], [500, 1500, 800])
    # Remove if it exists from a prior run
    if pymisha.gintervals_exists(name):
        pymisha.gintervals_rm(name)
    pymisha.gintervals_save(intervals, name)
    yield name
    if pymisha.gintervals_exists(name):
        pymisha.gintervals_rm(name)


class TestGintervalsUpdate:
    """Tests for gintervals_update."""

    def test_update_replaces_chrom_intervals(self, _test_iset):
        """Replacing intervals for a specific chrom."""
        new_intervals = pymisha.gintervals("1", 100, 200)
        pymisha.gintervals_update(_test_iset, new_intervals, chrom="1")
        result = pymisha.gintervals_load(_test_iset)
        # chrom 1 should now have only one interval [100, 200)
        chrom1 = result[result["chrom"] == "1"]
        assert len(chrom1) == 1
        assert chrom1.iloc[0]["start"] == 100.0
        assert chrom1.iloc[0]["end"] == 200.0
        # chrom 2 should be unchanged
        chrom2 = result[result["chrom"] == "2"]
        assert len(chrom2) == 1

    def test_delete_chrom_intervals(self, _test_iset):
        """Passing None removes all intervals for that chrom."""
        pymisha.gintervals_update(_test_iset, None, chrom="1")
        result = pymisha.gintervals_load(_test_iset)
        # Only chrom 2 should remain
        assert all(result["chrom"] == "2")

    def test_no_chrom_raises(self, _test_iset):
        """Must specify chrom parameter."""
        new_intervals = pymisha.gintervals("1", 100, 200)
        with pytest.raises(ValueError, match="[Cc]hrom"):
            pymisha.gintervals_update(_test_iset, new_intervals)

    def test_nonexistent_set_raises(self):
        """Non-existent intervals set raises error."""
        new_intervals = pymisha.gintervals("1", 100, 200)
        with pytest.raises(ValueError, match="does not exist"):
            pymisha.gintervals_update("no_such_set", new_intervals, chrom="1")

    def test_add_new_chrom(self, _test_iset):
        """Can add intervals for a chrom not previously in the set."""
        new_intervals = pymisha.gintervals("X", 0, 1000)
        pymisha.gintervals_update(_test_iset, new_intervals, chrom="X")
        result = pymisha.gintervals_load(_test_iset)
        chromx = result[result["chrom"] == "X"]
        assert len(chromx) == 1

    def test_preserves_extra_columns(self, _test_iset):
        """Extra columns beyond chrom/start/end are preserved."""
        new_intervals = pymisha.gintervals("1", 100, 200)
        new_intervals["score"] = 42.0
        pymisha.gintervals_update(_test_iset, new_intervals, chrom="1")
        result = pymisha.gintervals_load(_test_iset)
        chrom1 = result[result["chrom"] == "1"]
        # The new intervals should have the score column (though other chroms may have NaN)
        assert "score" in result.columns
        assert chrom1.iloc[0]["score"] == 42.0

    def test_chrom_normalization(self, _test_iset):
        """chr prefix is normalized."""
        new_intervals = pymisha.gintervals("chr1", 100, 200)
        pymisha.gintervals_update(_test_iset, new_intervals, chrom="chr1")
        result = pymisha.gintervals_load(_test_iset)
        chrom1 = result[result["chrom"] == "1"]
        assert len(chrom1) == 1
