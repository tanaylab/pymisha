"""Tests for gintervals_load and gintervals_save functions."""


import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


_RSCRIPT_AVAILABLE = shutil.which("Rscript") is not None


def _require_r_misha() -> None:
    """Skip the calling test unless the R ``misha`` package actually loads.

    Deliberately *not* probed at import time: pytest imports every test module
    during collection, so a module-level ``library(misha)`` probe spawns an
    Rscript subprocess (and pays R's package-load time) on every pytest
    invocation, whether or not this file was selected.
    """
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(misha)"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure means "no R misha"
        pytest.skip(f"R misha not usable: {exc}")
    if result.returncode != 0:
        pytest.skip("R misha not installed")


class TestGintervalsLoad:
    """Test gintervals_load function."""

    def test_load_existing_interval_set(self):
        """Load an existing interval set returns correct DataFrame."""
        result = pm.gintervals_load("annotations")
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "chrom" in result.columns
        assert "start" in result.columns
        assert "end" in result.columns
        # annotations has 8 intervals
        assert len(result) == 8

    def test_load_returns_correct_columns(self):
        """Loaded interval set has expected columns including extra ones."""
        result = pm.gintervals_load("annotations")
        # annotations has strand and remark columns
        assert "strand" in result.columns
        assert "remark" in result.columns

    def test_load_values_match_expected(self):
        """Loaded interval set has correct values."""
        result = pm.gintervals_load("annotations")
        # First interval should be chr1:20-2000
        first = result.iloc[0]
        assert str(first["chrom"]) == "1"
        assert first["start"] == 20
        assert first["end"] == 2000
        assert first["strand"] == 1

    def test_load_nonexistent_interval_set_raises(self):
        """Loading non-existent interval set raises appropriate error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_load("nonexistent_intervals")

    def test_load_with_chrom_filter(self):
        """Load interval set filtered by chromosome."""
        result = pm.gintervals_load("annotations", chrom="1")
        assert result is not None
        # All intervals should be on chr1
        assert all(str(c) == "1" for c in result["chrom"])

    def test_load_with_chrom_filter_no_match(self):
        """Load with chromosome filter that matches nothing returns empty."""
        # annotations has chr1, chr2 - chrX should return empty
        result = pm.gintervals_load("annotations", chrom="chrX")
        # Should return empty DataFrame or None
        assert result is None or len(result) == 0


class TestGintervalsSave:
    """Test gintervals_save function."""

    def test_save_basic(self):
        """Save a simple interval set to the database."""
        # Create test intervals (using chroms "1", "2" matching test DB)
        intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])

        # Save to a new name
        pm.gintervals_save(intervals, "test_save_basic")

        # Verify it was saved
        assert pm.gintervals_exists("test_save_basic")

        # Clean up
        pm.gintervals_rm("test_save_basic")

    def test_save_and_load_roundtrip(self):
        """Saved intervals can be loaded back correctly."""
        # Create intervals with specific values (using chroms "1", "2" matching test DB)
        intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])

        # Save
        pm.gintervals_save(intervals, "test_roundtrip")

        # Load back
        loaded = pm.gintervals_load("test_roundtrip")

        # Verify
        assert loaded is not None
        assert len(loaded) == 2
        # Check values (sorted by chrom, start)
        assert loaded.iloc[0]["chrom"] == "1" or str(loaded.iloc[0]["chrom"]) == "1"
        assert loaded.iloc[0]["start"] == 100
        assert loaded.iloc[0]["end"] == 1000

        # Clean up
        pm.gintervals_rm("test_roundtrip")

    def test_save_with_extra_columns(self):
        """Save intervals with additional columns preserves them."""
        # Create intervals with extra column (using chroms "1", "2" matching test DB)
        df = pd.DataFrame({
            "chrom": ["1", "2"],
            "start": [100, 200],
            "end": [1000, 2000],
            "score": [1.5, 2.5],
            "name": ["gene1", "gene2"]
        })

        pm.gintervals_save(df, "test_extra_cols")
        loaded = pm.gintervals_load("test_extra_cols")

        assert "score" in loaded.columns
        assert "name" in loaded.columns
        assert loaded.iloc[0]["score"] == 1.5 or np.isclose(loaded.iloc[0]["score"], 1.5)

        pm.gintervals_rm("test_extra_cols")

    def test_save_existing_raises(self):
        """Saving to an existing interval set name raises error."""
        intervals = pm.gintervals("1", 100, 1000)

        # First save should work
        pm.gintervals_save(intervals, "test_dup")

        # Second save to same name should fail
        with pytest.raises(ValueError, match="already exists"):
            pm.gintervals_save(intervals, "test_dup")

        pm.gintervals_rm("test_dup")

    def test_save_invalid_name_raises(self):
        """Invalid interval set names are rejected."""
        intervals = pm.gintervals("1", 100, 1000)

        with pytest.raises(ValueError):
            pm.gintervals_save(intervals, "123invalid")  # starts with number

        with pytest.raises(ValueError):
            pm.gintervals_save(intervals, "has spaces")  # contains space


class TestGintervalsRm:
    """Test gintervals_rm function."""

    def test_rm_existing(self):
        """Remove an existing interval set."""
        # Create and save
        intervals = pm.gintervals("1", 100, 1000)
        pm.gintervals_save(intervals, "test_rm")
        assert pm.gintervals_exists("test_rm")

        # Remove
        pm.gintervals_rm("test_rm")
        assert not pm.gintervals_exists("test_rm")

    def test_rm_nonexistent_raises(self):
        """Removing non-existent interval set raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_rm("nonexistent_set")

    def test_rm_with_force_nonexistent(self):
        """Remove with force=True on non-existent set doesn't raise."""
        # Should not raise
        pm.gintervals_rm("nonexistent_set", force=True)


class TestGintervalsLoadGoldenMaster:
    """Golden master tests comparing with R misha output."""

    def test_load_annotations_matches_r(self):
        """gintervals_load returns same data as R gintervals.load."""
        result = pm.gintervals_load("annotations")

        # Expected values from R misha (verified with the interv file)
        expected_chroms = ["1", "1", "2", "2", "2", "2", "2", "2"]
        expected_starts = [20, 2500, 20, 3000, 9000, 12000, 13000, 15000]
        expected_ends = [2000, 2600, 2000, 8000, 11000, 12001, 14000, 15500]

        assert len(result) == len(expected_chroms)
        for i, (chrom, start, end) in enumerate(zip(expected_chroms, expected_starts, expected_ends, strict=False)):
            assert str(result.iloc[i]["chrom"]) == chrom
            assert result.iloc[i]["start"] == start
            assert result.iloc[i]["end"] == end


class TestGintervalsSaveRParity:
    """An interval set written by pymisha must be loadable by R misha.

    Regression for the missing OBJECT bit in `_r_serialize.py`'s data.frame
    head word: without it, R unserializes an object whose `class` attribute
    says "data.frame" but whose internal OBJECT flag is unset, so S3 dispatch
    for `dim()`/`nrow()` never fires and `gintervals.load()` dies with
    "argument is of length zero" even though the file is otherwise well-formed.
    """

    def test_object_bit_is_written(self, tmp_path):
        """The head word of the written data.frame carries R's OBJECT bit.

        Unconditional counterpart to the round-trip test below, which can only
        run where R misha is installed. R's serialize.c ORs the OBJECT bit
        (1 << 8) into the head word whenever the object carries a `class`
        attribute; without it R's `dim`/`nrow` never dispatch. For a classed
        VECSXP the head word is VECSXP | HAS_ATTR | OBJECT = 19 | 512 | 256
        = 0x313.
        """
        import struct

        from pymisha._r_serialize import write_dataframe

        path = tmp_path / "object_bit.interv"
        write_dataframe(str(path), pd.DataFrame({"chrom": ["1"], "start": [1.0], "end": [2.0]}))

        raw = path.read_bytes()
        # XDR R serialization: "X\n" magic, then version / writer / reader
        # version ints, then the head word of the top-level object.
        assert raw[:2] == b"X\n"
        head = struct.unpack_from(">i", raw, 2 + 3 * 4)[0]
        assert head == 0x313, f"head word 0x{head:x}, expected 0x313 (OBJECT bit missing?)"

    @pytest.mark.skipif(
        not _RSCRIPT_AVAILABLE, reason="Rscript not on PATH"
    )
    def test_saved_intervals_load_in_r(self):
        _require_r_misha()
        groot = pm._shared._GROOT
        name = "test_r_object_bit_roundtrip"
        intervals = pm.gintervals(["1", "2"], [100, 200], [1000, 2000])
        pm.gintervals_save(intervals, name)
        try:
            r_script = (
                "library(misha); "
                f"gdb.init('{groot}', rescan=TRUE); "
                f"res <- gintervals.load('{name}'); "
                "cat('NROW=', nrow(res), '\\n', sep=''); "
                "stopifnot(is.object(res), nrow(res) == 2L)"
            )
            result = subprocess.run(
                ["Rscript", "-e", r_script],
                capture_output=True, text=True, timeout=60,
            )
            assert result.returncode == 0, (
                "R failed to load a pymisha-written interval set:\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
            assert "NROW=2" in result.stdout
        finally:
            pm.gintervals_rm(name)
