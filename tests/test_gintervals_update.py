"""Tests for gintervals_update."""

from pathlib import Path

import pandas as pd
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


# ──────────────────────────────────────────────────────────────────
# 2D updates: a validation failure must not destroy the existing set
# ──────────────────────────────────────────────────────────────────


def _iv2d(rows):
    """A 2D interval frame from (chrom1, start1, end1, chrom2, start2, end2) tuples."""
    return pd.DataFrame(
        [
            {
                "chrom1": r[0], "start1": float(r[1]), "end1": float(r[2]),
                "chrom2": r[3], "start2": float(r[4]), "end2": float(r[5]),
            }
            for r in rows
        ]
    )


def _save_2d_unvalidated(df, name):
    """Write a 2D interval set the way R misha's ``gintervals.save`` fast path
    does: serialize the frame straight to ``<groot>/tracks/<name>.interv`` with
    no coordinate check at all (misha/R/intervals-management.R, the
    ``!.gintervals.needs_bigset`` branch).

    Used so the "existing rows are out of bounds" test starts from a set that a
    real R session could have written, rather than from a frame pymisha's own
    validated writer would have refused.
    """
    import _pymisha

    from pymisha import _shared
    from pymisha._r_serialize import write_dataframe

    path = Path(_shared._GROOT) / "tracks" / (name.replace(".", "/") + ".interv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["chrom1"] = pd.Categorical(out["chrom1"].astype(str))
    out["chrom2"] = pd.Categorical(out["chrom2"].astype(str))
    for col in ("start1", "end1", "start2", "end2"):
        out[col] = out[col].astype(float)
    write_dataframe(str(path), out)
    _pymisha.pm_interv_register(name)


@pytest.fixture()
def _iset_name():
    """A scratch interval-set name, removed before and after the test."""
    name = "test_update_2d_scratch"
    if pymisha.gintervals_exists(name):
        pymisha.gintervals_rm(name, force=True)
    yield name
    if pymisha.gintervals_exists(name):
        pymisha.gintervals_rm(name, force=True)


class TestGintervalsUpdate2d:
    """chrom1/chrom2 updates, and the data-loss regression they exposed.

    ``gintervals_update`` used to ``gintervals_rm`` the set and only then
    ``gintervals_save`` the merged frame. Since the save validates 2D
    coordinates, any rejected frame left the user with *no* set at all - the
    update deleted data it then refused to write back.
    """

    def test_valid_2d_update_replaces_the_pair(self, _iset_name):
        """Baseline: a valid 2D update replaces only the named chrom pair."""
        pymisha.gintervals_save(
            _iv2d([("1", 0, 1000, "1", 0, 1000), ("1", 0, 1000, "2", 0, 1000)]),
            _iset_name,
        )
        pymisha.gintervals_update(
            _iset_name, _iv2d([("1", 5000, 6000, "2", 5000, 6000)]),
            chrom1="1", chrom2="2",
        )
        loaded = pymisha.gintervals_load(_iset_name)
        assert len(loaded) == 2
        pair = loaded[(loaded["chrom1"] == "1") & (loaded["chrom2"] == "2")]
        assert len(pair) == 1
        assert int(pair.iloc[0]["start1"]) == 5000
        untouched = loaded[(loaded["chrom1"] == "1") & (loaded["chrom2"] == "1")]
        assert len(untouched) == 1
        assert int(untouched.iloc[0]["end1"]) == 1000

    def test_bad_payload_leaves_the_set_intact(self, _iset_name):
        """Reachable path 1: the caller's own rectangle is out of bounds."""
        pymisha.gintervals_save(
            _iv2d([("1", 0, 1000, "1", 0, 1000)]), _iset_name
        )
        bad = _iv2d([("1", 0, 1_000_000_000, "2", 0, 1000)])
        with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
            pymisha.gintervals_update(_iset_name, bad, chrom1="1", chrom2="2")

        assert pymisha.gintervals_exists(_iset_name), "the failed update deleted the set"
        loaded = pymisha.gintervals_load(_iset_name)
        assert len(loaded) == 1
        assert int(loaded.iloc[0]["end1"]) == 1000

    def test_preexisting_out_of_range_row_leaves_the_set_intact(self, _iset_name):
        """Reachable path 2, the dangerous one: the *existing* rows are out of
        bounds. R's gintervals.save never verifies, so any R-written set can be
        in this state, and the update below touches none of the offending rows.
        """
        _save_2d_unvalidated(
            _iv2d([
                ("1", 0, 1000, "1", 0, 1000),
                ("2", 0, 1_000_000_000, "2", 0, 1000),   # R-written, unverifiable
            ]),
            _iset_name,
        )
        assert len(pymisha.gintervals_load(_iset_name)) == 2

        good = _iv2d([("1", 2000, 3000, "1", 2000, 3000)])
        with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
            pymisha.gintervals_update(_iset_name, good, chrom1="1", chrom2="1")

        assert pymisha.gintervals_exists(_iset_name), "the failed update deleted the set"
        loaded = pymisha.gintervals_load(_iset_name)
        assert len(loaded) == 2
        assert int(loaded[loaded["chrom1"] == "2"].iloc[0]["end1"]) == 1_000_000_000

    def test_successful_update_leaves_no_stray_temporary_set(self, _iset_name):
        """The temporary set the swap writes must not outlive the update."""
        pymisha.gintervals_save(_iv2d([("1", 0, 1000, "1", 0, 1000)]), _iset_name)
        before = set(pymisha.gintervals_ls())
        pymisha.gintervals_update(
            _iset_name, _iv2d([("1", 5000, 6000, "2", 5000, 6000)]),
            chrom1="1", chrom2="2",
        )
        assert set(pymisha.gintervals_ls()) == before

    def test_update_keeps_the_sets_attributes(self, _iset_name):
        """Attributes describe the set, not its rows. R's gintervals.update
        rewrites the file in place and keeps the .iattr companion; the old
        rm + save dropped it.
        """
        pymisha.gintervals_save(_iv2d([("1", 0, 1000, "1", 0, 1000)]), _iset_name)
        pymisha.gintervals_attr_set(_iset_name, "description", "kept across updates")
        pymisha.gintervals_update(
            _iset_name, _iv2d([("1", 5000, 6000, "2", 5000, 6000)]),
            chrom1="1", chrom2="2",
        )
        assert pymisha.gintervals_attr_get(_iset_name, "description") == "kept across updates"

    def test_failed_update_leaves_no_stray_temporary_set(self, _iset_name):
        """The rejected write must not leave a half-named leftover behind."""
        pymisha.gintervals_save(_iv2d([("1", 0, 1000, "1", 0, 1000)]), _iset_name)
        before = set(pymisha.gintervals_ls())
        with pytest.raises(ValueError):
            pymisha.gintervals_update(
                _iset_name, _iv2d([("1", 0, 1_000_000_000, "2", 0, 1000)]),
                chrom1="1", chrom2="2",
            )
        assert set(pymisha.gintervals_ls()) == before
