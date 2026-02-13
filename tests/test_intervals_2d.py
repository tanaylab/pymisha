"""Tests for 2D interval constructors."""

import pytest

import pymisha


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    """Ensure test DB is initialized."""


class TestGintervals2d:
    """Tests for gintervals_2d."""

    def test_single_chrom_defaults(self):
        """Single chrom with default starts/ends gives full-chrom 2D interval."""
        result = pymisha.gintervals_2d("1")
        assert list(result.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        assert len(result) == 1
        assert result.iloc[0]["chrom1"] == "1"
        assert result.iloc[0]["chrom2"] == "1"
        assert result.iloc[0]["start1"] == 0
        assert result.iloc[0]["start2"] == 0
        # ends should be chrom size
        assert result.iloc[0]["end1"] == 500000
        assert result.iloc[0]["end2"] == 500000

    def test_two_different_chroms(self):
        """chrom1 != chrom2."""
        result = pymisha.gintervals_2d("1", 100, 200, "2", 300, 400)
        assert len(result) == 1
        assert result.iloc[0]["chrom1"] == "1"
        assert result.iloc[0]["start1"] == 100
        assert result.iloc[0]["end1"] == 200
        assert result.iloc[0]["chrom2"] == "2"
        assert result.iloc[0]["start2"] == 300
        assert result.iloc[0]["end2"] == 400

    def test_chroms2_defaults_to_chroms1(self):
        """When chroms2 is None, it defaults to chroms1."""
        result = pymisha.gintervals_2d("X", 10, 100)
        assert result.iloc[0]["chrom1"] == "X"
        assert result.iloc[0]["chrom2"] == "X"
        assert result.iloc[0]["start2"] == 0
        assert result.iloc[0]["end2"] == 200000

    def test_multiple_intervals(self):
        """Multiple intervals created from lists."""
        result = pymisha.gintervals_2d(
            ["1", "2"], [100, 200], [300, 400],
            ["X", "1"], [50, 60], [150, 160],
        )
        assert len(result) == 2

    def test_sorted_output(self):
        """Output is sorted by chrom1, start1, chrom2, start2."""
        result = pymisha.gintervals_2d(
            ["2", "1"], [100, 200], [300, 400],
            ["1", "X"], [50, 60], [150, 160],
        )
        # chrom1="1" should come before chrom1="2"
        assert result.iloc[0]["chrom1"] == "1"
        assert result.iloc[1]["chrom1"] == "2"

    def test_scalar_broadcast(self):
        """Scalar args broadcast to match list length."""
        result = pymisha.gintervals_2d(["1", "2"], 0, 100)
        assert len(result) == 2
        assert result.iloc[0]["start1"] == 0
        assert result.iloc[0]["end1"] == 100

    def test_end_minus1_means_full_chrom(self):
        """end=-1 means full chromosome length."""
        result = pymisha.gintervals_2d("2", 0, -1, "X", 0, -1)
        assert result.iloc[0]["end1"] == 300000
        assert result.iloc[0]["end2"] == 200000

    def test_invalid_chrom_raises(self):
        """Unknown chromosome raises ValueError or pymisha.error."""
        from pymisha import _pymisha
        with pytest.raises((ValueError, _pymisha.error)):
            pymisha.gintervals_2d("nonexistent")

    def test_bad_range_raises(self):
        """start >= end raises ValueError."""
        with pytest.raises(ValueError):
            pymisha.gintervals_2d("1", 200, 100)

    def test_chrom_normalization(self):
        """chr prefix is normalized away."""
        result = pymisha.gintervals_2d("chr1", 0, 100, "chrX", 0, 50)
        assert result.iloc[0]["chrom1"] == "1"
        assert result.iloc[0]["chrom2"] == "X"


class TestGintervals2dAll:
    """Tests for gintervals_2d_all."""

    def test_returns_2d_dataframe(self):
        """Returns DataFrame with 2D interval columns."""
        result = pymisha.gintervals_2d_all()
        assert list(result.columns) == ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]

    def test_diagonal_mode_default(self):
        """Default mode is diagonal: only chrom1 == chrom2 pairs."""
        result = pymisha.gintervals_2d_all()
        # Test DB has 3 chroms: 1, 2, X → 3 diagonal entries
        assert len(result) == 3
        for _, row in result.iterrows():
            assert row["chrom1"] == row["chrom2"]

    def test_diagonal_covers_full_chroms(self):
        """Each diagonal entry covers the full chromosome."""
        result = pymisha.gintervals_2d_all()
        all_intervals = pymisha.gintervals_all()
        # all_intervals has 'chrom', 'start', 'end' (where end is chrom size)

        for _, row in result.iterrows():
            expected_size = all_intervals[all_intervals["chrom"] == row["chrom1"]]["end"].iloc[0]
            assert row["start1"] == 0
            assert row["end1"] == expected_size
            assert row["start2"] == 0
            assert row["end2"] == expected_size

    def test_full_mode(self):
        """Full mode gives all NxN chromosome pairs."""
        result = pymisha.gintervals_2d_all(mode="full")
        # 3 chroms → 9 pairs
        assert len(result) == 9

    def test_full_mode_has_all_pairs(self):
        """Full mode contains cross-chromosome pairs."""
        result = pymisha.gintervals_2d_all(mode="full")
        pairs = set(zip(result["chrom1"], result["chrom2"], strict=False))
        chroms = ["1", "2", "X"]
        for c1 in chroms:
            for c2 in chroms:
                assert (c1, c2) in pairs

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            pymisha.gintervals_2d_all(mode="invalid")
