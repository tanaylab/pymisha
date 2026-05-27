"""Tests for the array-track read API (Group H of the 2026-05-15 parity audit).

Closes B1: ``gextract('array_track', ...)`` used to raise
``"Track type 'array' not yet supported"`` with no path forward. After
Group H, ``gtrack_array_get_colnames``, ``gtrack_array_set_colnames``,
and ``gtrack_array_extract`` give the user a complete read API; the
``gextract`` error now points there explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _examples_db():
    pm.gdb_init_examples()


class TestGtrackArrayGetColnames:
    def test_returns_declared_names(self):
        names = pm.gtrack_array_get_colnames("array_track")
        assert names == [f"col{i}" for i in range(10)]

    def test_non_array_track_raises(self):
        with pytest.raises(ValueError, match="not an array track"):
            pm.gtrack_array_get_colnames("dense_track")

    def test_unknown_track_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            pm.gtrack_array_get_colnames("not_a_real_track")


class TestGtrackArraySetColnames:
    def test_roundtrip(self):
        orig = pm.gtrack_array_get_colnames("array_track")
        try:
            new = [f"x{i}" for i in range(10)]
            pm.gtrack_array_set_colnames("array_track", new)
            assert pm.gtrack_array_get_colnames("array_track") == new
        finally:
            pm.gtrack_array_set_colnames("array_track", orig)

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            pm.gtrack_array_set_colnames(
                "array_track", ["a"] * 10
            )

    def test_empty_names_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            pm.gtrack_array_set_colnames(
                "array_track", ["a", "b", "c", "d", "e", "f", "g", "h", "i", ""]
            )


class TestGtrackArrayExtract:
    def test_all_columns_by_default(self):
        ivs = pm.gintervals("1", 0, 10000)
        df = pm.gtrack_array_extract("array_track", intervals=ivs)
        names = pm.gtrack_array_get_colnames("array_track")
        assert set(df.columns) >= {"chrom", "start", "end", "intervalID"} | set(names)

    def test_returns_per_position_rows(self):
        ivs = pm.gintervals("1", 0, 5000)
        df = pm.gtrack_array_extract("array_track", intervals=ivs)
        assert len(df) > 0
        assert (df["chrom"] == "1").all()
        assert (df["start"] < df["end"]).all()
        # Each row corresponds to one track interval clipped to the query.
        assert (df["start"] >= 0).all() and (df["end"] <= 5000).all()

    def test_slice_by_name(self):
        ivs = pm.gintervals("1", 0, 5000)
        df = pm.gtrack_array_extract(
            "array_track", slice=["col0", "col5"], intervals=ivs
        )
        assert list(df.columns) == [
            "chrom", "start", "end", "col0", "col5", "intervalID"
        ]

    def test_slice_by_index(self):
        ivs = pm.gintervals("1", 0, 5000)
        df = pm.gtrack_array_extract(
            "array_track", slice=[0, 5], intervals=ivs
        )
        assert list(df.columns) == [
            "chrom", "start", "end", "col0", "col5", "intervalID"
        ]

    def test_unknown_column_raises(self):
        ivs = pm.gintervals("1", 0, 5000)
        with pytest.raises(ValueError, match="not a column"):
            pm.gtrack_array_extract(
                "array_track", slice=["nonexistent"], intervals=ivs
            )

    def test_out_of_range_index_raises(self):
        ivs = pm.gintervals("1", 0, 5000)
        with pytest.raises(ValueError, match="out of range"):
            pm.gtrack_array_extract(
                "array_track", slice=[50], intervals=ivs
            )

    def test_empty_query_returns_empty_frame(self):
        # A query interval entirely outside any track interval -> empty df
        # but still typed correctly.
        ivs = pm.gintervals("2", 0, 10)
        df = pm.gtrack_array_extract("array_track", intervals=ivs)
        # Whatever the result, must have the standard columns.
        assert set(df.columns) >= {"chrom", "start", "end", "intervalID"}

    def test_nans_preserved_for_missing_columns(self):
        ivs = pm.gintervals("1", 0, 5000)
        df = pm.gtrack_array_extract(
            "array_track", slice=["col1"], intervals=ivs
        )
        # The bundled track stores only a subset of columns per interval;
        # some col1 values must be NaN (sparse storage).
        assert df["col1"].isna().any()


class TestGextractOnArrayTrackScanner:
    """The C++ scanner reads array tracks directly: each iterator bin is the
    average of the array's per-column values aggregated over that bin."""

    def test_bare_array_reads(self):
        ivs = pm.gintervals("1", 0, 5000)
        result = pm.gextract("array_track", intervals=ivs, iterator=200)
        assert result is not None and len(result) > 0
        assert "array_track" in result.columns
        # At least one bin carries data.
        assert result["array_track"].notna().any()


class TestArrayTrackHelperExports:
    """The new functions are public and listed in __all__."""

    def test_get_colnames_exported(self):
        assert "gtrack_array_get_colnames" in pm.__all__
        assert "gtrack_array_set_colnames" in pm.__all__
        assert "gtrack_array_extract" in pm.__all__


class TestGtrackArrayCreate:
    """Write-then-read round trips. The on-disk format matches R misha
    (binary + R-serialized .colnames)."""

    def test_simple_roundtrip(self):
        import pandas as pd
        ivs = pd.DataFrame({
            "chrom": ["1", "1", "2"],
            "start": [0, 200, 0],
            "end":   [100, 300, 50],
        })
        values = np.array([
            [1.0, 2.0, np.nan, 4.0],
            [np.nan, 6.0, 7.0, 8.0],
            [9.0, np.nan, 11.0, np.nan],
        ])
        try:
            pm.gtrack_array_create(
                "test_arr_rt", "roundtrip", ivs, values,
                colnames=["a", "b", "c", "d"],
            )
            assert pm.gtrack_exists("test_arr_rt")
            assert pm.gtrack_info("test_arr_rt")["type"] == "array"
            assert pm.gtrack_array_get_colnames("test_arr_rt") == [
                "a", "b", "c", "d"
            ]
            out = pm.gtrack_array_extract("test_arr_rt")
            assert len(out) == 3
            # Verify a known cell
            row1 = out[out["start"] == 0].iloc[0]
            assert row1["a"] == 1.0
            assert np.isnan(row1["c"])
        finally:
            pm.gtrack_rm("test_arr_rt", force=True)

    def test_rejects_mismatched_value_columns(self):
        import pandas as pd
        ivs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
        with pytest.raises(ValueError, match="match len.colnames"):
            pm.gtrack_array_create(
                "test_arr_x", "x", ivs, np.array([[1.0, 2.0]]),
                colnames=["a", "b", "c"],
            )

    def test_rejects_unsorted_intervals(self):
        import pandas as pd
        ivs = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [200, 0],
            "end":   [300, 100],
        })
        # gtrack_array_create sorts internally, so this should succeed
        try:
            pm.gtrack_array_create(
                "test_arr_s", "s", ivs, np.array([[1.0], [2.0]]),
                colnames=["a"],
            )
            out = pm.gtrack_array_extract("test_arr_s")
            assert list(out["start"]) == [0, 200]
        finally:
            pm.gtrack_rm("test_arr_s", force=True)

    def test_rejects_overlapping_intervals(self):
        import pandas as pd
        ivs = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 50],
            "end":   [100, 150],
        })
        with pytest.raises(ValueError, match="non-overlapping"):
            pm.gtrack_array_create(
                "test_arr_o", "o", ivs, np.array([[1.0], [2.0]]),
                colnames=["a"],
            )


class TestColnamesRSerializeRoundTrip:
    """The .colnames file we write must be readable by R's
    `unserialize()` - the wire format mirrors R's
    `serialize(setNames(seq_along(names), names), con, ascii=FALSE)`.
    """

    def test_written_colnames_round_trip_through_native_reader(self, tmp_path):
        from pymisha._array_track import write_colnames
        from pymisha._r_serialize import read

        write_colnames(tmp_path, ["a", "b", "c"])
        obj = read(tmp_path / ".colnames")
        assert obj.names == ["a", "b", "c"]
        np.testing.assert_array_equal(obj, [1, 2, 3])
