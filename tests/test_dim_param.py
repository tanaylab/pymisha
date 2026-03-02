"""Tests for the dim parameter in gvtrack_iterator.

The dim parameter projects 2D intervals to 1D before evaluating a virtual
track, allowing 1D source tracks to be used in 2D extraction contexts.

- dim=1: project onto (chrom1, start1, end1)
- dim=2: project onto (chrom2, start2, end2)
"""

import numpy as np
import pytest

import pymisha as pm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _clean_vtracks():
    """Clean up all vtracks after each test."""
    yield
    pm.gvtrack_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_avg(chrom, start, end):
    """Compute the reference avg of dense_track over a 1D interval."""
    s = pm.gsummary(
        "dense_track",
        pm.gintervals([chrom], [start], [end]),
        progress=False,
    )
    return s["Mean"]


def _reference_max(chrom, start, end):
    """Compute the reference max of dense_track over a 1D interval."""
    s = pm.gsummary(
        "dense_track",
        pm.gintervals([chrom], [start], [end]),
        progress=False,
    )
    return s["Max"]


def _reference_sum(chrom, start, end):
    """Compute the reference sum of dense_track over a 1D interval."""
    s = pm.gsummary(
        "dense_track",
        pm.gintervals([chrom], [start], [end]),
        progress=False,
    )
    return s["Sum"]


# ---------------------------------------------------------------------------
# Tests: dim=1 projection
# ---------------------------------------------------------------------------


class TestDim1Projection:
    """Tests for dim=1: project 2D intervals onto first dimension."""

    def test_dim1_avg_basic(self):
        """dim=1 with avg should use chrom1/start1/end1 coordinates."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [200000], [210000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        assert len(result) == 1
        expected = _reference_avg("1", 0, 10000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-5
        )

    def test_dim1_avg_multiple_intervals(self):
        """dim=1 with avg over multiple 2D intervals."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1", "1"],
            [0, 100000],
            [10000, 110000],
            ["1", "1"],
            [200000, 300000],
            [210000, 310000],
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        assert len(result) == 2

        expected_0 = _reference_avg("1", 0, 10000)
        expected_1 = _reference_avg("1", 100000, 110000)

        np.testing.assert_allclose(
            result["vt"].values[0], expected_0, rtol=1e-5
        )
        np.testing.assert_allclose(
            result["vt"].values[1], expected_1, rtol=1e-5
        )

    def test_dim1_max(self):
        """dim=1 with max function."""
        pm.gvtrack_create("vt", "dense_track", func="max")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [200000], [210000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        expected = _reference_max("1", 0, 10000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-5
        )

    def test_dim1_sum(self):
        """dim=1 with sum function."""
        pm.gvtrack_create("vt", "dense_track", func="sum")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [200000], [210000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        expected = _reference_sum("1", 0, 10000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-4
        )

    def test_dim1_ignores_dim2_coords(self):
        """dim=1 should ignore the second dimension entirely.

        Same dim1 coords, different dim2 coords => same result.
        """
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        intv_a = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [0], [10000]
        )
        intv_b = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [400000], [410000]
        )

        result_a = pm.gextract("vt", intervals=intv_a, progress=False)
        result_b = pm.gextract("vt", intervals=intv_b, progress=False)

        np.testing.assert_allclose(
            result_a["vt"].values[0],
            result_b["vt"].values[0],
            rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests: dim=2 projection
# ---------------------------------------------------------------------------


class TestDim2Projection:
    """Tests for dim=2: project 2D intervals onto second dimension."""

    def test_dim2_avg_basic(self):
        """dim=2 with avg should use chrom2/start2/end2 coordinates."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=2)

        intervals_2d = pm.gintervals_2d(
            ["1"], [200000], [210000], ["1"], [0], [10000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        assert len(result) == 1
        expected = _reference_avg("1", 0, 10000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-5
        )

    def test_dim2_cross_chrom(self):
        """dim=2 with chrom2 different from chrom1."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=2)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["2"], [0], [10000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        expected = _reference_avg("2", 0, 10000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-5
        )

    def test_dim2_ignores_dim1_coords(self):
        """dim=2 should ignore the first dimension entirely.

        Same dim2 coords, different dim1 coords => same result.
        """
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=2)

        intv_a = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [300000], [310000]
        )
        intv_b = pm.gintervals_2d(
            ["1"], [400000], [410000], ["1"], [300000], [310000]
        )

        result_a = pm.gextract("vt", intervals=intv_a, progress=False)
        result_b = pm.gextract("vt", intervals=intv_b, progress=False)

        np.testing.assert_allclose(
            result_a["vt"].values[0],
            result_b["vt"].values[0],
            rtol=1e-10,
        )

    def test_dim2_nan_propagation(self):
        """dim=2 should return NaN when the projected interval has no data."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=2)

        # dense_track has NaN from ~170000 onward on chrom 1
        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [200000], [210000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        assert np.isnan(result["vt"].values[0])


# ---------------------------------------------------------------------------
# Tests: dim with shifts
# ---------------------------------------------------------------------------


class TestDimWithShifts:
    """Tests for dim combined with sshift/eshift."""

    def test_dim1_with_shifts(self):
        """dim=1 with sshift/eshift: shifts apply after projection."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1, sshift=-5000, eshift=5000)

        intervals_2d = pm.gintervals_2d(
            ["1"], [10000], [20000], ["1"], [200000], [210000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        # Projected interval: (1, 10000, 20000)
        # After shifts: (1, 5000, 25000)
        expected = _reference_avg("1", 5000, 25000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-5
        )

    def test_dim2_with_shifts(self):
        """dim=2 with sshift/eshift: shifts apply to projected dim2."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=2, sshift=-5000, eshift=5000)

        intervals_2d = pm.gintervals_2d(
            ["1"], [200000], [210000], ["1"], [10000], [20000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        # Projected interval: (1, 10000, 20000)
        # After shifts: (1, 5000, 25000)
        expected = _reference_avg("1", 5000, 25000)
        np.testing.assert_allclose(
            result["vt"].values[0], expected, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Tests: dim stored in vtrack config
# ---------------------------------------------------------------------------


class TestDimConfig:
    """Tests for dim parameter storage and retrieval."""

    def test_dim_stored_in_info(self):
        """gvtrack_info should report the dim value."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        info = pm.gvtrack_info("vt")
        assert info["dim"] == 1

    def test_dim2_stored_in_info(self):
        """gvtrack_info should report dim=2."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=2)

        info = pm.gvtrack_info("vt")
        assert info["dim"] == 2

    def test_dim_none_not_stored(self):
        """dim=None should not add a dim key."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=None)

        info = pm.gvtrack_info("vt")
        assert "dim" not in info

    def test_dim_zero_not_active(self):
        """dim=0 means no projection — same as None."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=0)

        info = pm.gvtrack_info("vt")
        assert info.get("dim") == 0


# ---------------------------------------------------------------------------
# Tests: output DataFrame structure
# ---------------------------------------------------------------------------


class TestDimOutputStructure:
    """Tests for the structure of the output DataFrame with dim."""

    def test_output_has_2d_columns(self):
        """Even with dim, the output should retain 2D interval columns."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [200000], [210000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        expected_cols = {
            "chrom1", "start1", "end1",
            "chrom2", "start2", "end2",
            "vt", "intervalID",
        }
        assert set(result.columns) == expected_cols

    def test_output_preserves_interval_coords(self):
        """The output should preserve the original 2D interval coordinates."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1"], [50000], [60000], ["2"], [10000], [20000]
        )
        result = pm.gextract("vt", intervals=intervals_2d, progress=False)

        assert result is not None
        row = result.iloc[0]
        assert str(row["chrom1"]) == "1"
        assert int(row["start1"]) == 50000
        assert int(row["end1"]) == 60000
        assert str(row["chrom2"]) == "2"
        assert int(row["start2"]) == 10000
        assert int(row["end2"]) == 20000


# ---------------------------------------------------------------------------
# Tests: expressions with dim vtracks
# ---------------------------------------------------------------------------


class TestDimExpressions:
    """Tests for dim vtracks in compound expressions."""

    def test_expression_with_dim_vtrack(self):
        """A dim vtrack should work in arithmetic expressions."""
        pm.gvtrack_create("vt", "dense_track", func="avg")
        pm.gvtrack_iterator("vt", dim=1)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [200000], [210000]
        )
        result = pm.gextract(
            "vt * 2",
            intervals=intervals_2d,
            progress=False,
        )

        assert result is not None
        expected = _reference_avg("1", 0, 10000) * 2
        np.testing.assert_allclose(
            result.iloc[0, -2], expected, rtol=1e-5
        )

    def test_two_dim_vtracks(self):
        """Two dim vtracks (dim=1 and dim=2) in the same extraction."""
        pm.gvtrack_create("vt1", "dense_track", func="avg")
        pm.gvtrack_iterator("vt1", dim=1)

        pm.gvtrack_create("vt2", "dense_track", func="avg")
        pm.gvtrack_iterator("vt2", dim=2)

        intervals_2d = pm.gintervals_2d(
            ["1"], [0], [10000], ["1"], [300000], [310000]
        )

        result = pm.gextract(
            ["vt1", "vt2"],
            intervals=intervals_2d,
            progress=False,
        )

        assert result is not None
        assert len(result) == 1

        expected_dim1 = _reference_avg("1", 0, 10000)
        expected_dim2 = _reference_avg("1", 300000, 310000)

        np.testing.assert_allclose(
            result["vt1"].values[0], expected_dim1, rtol=1e-5
        )
        np.testing.assert_allclose(
            result["vt2"].values[0], expected_dim2, rtol=1e-5
        )
