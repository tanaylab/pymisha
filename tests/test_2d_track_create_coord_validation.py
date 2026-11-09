"""Regression tests: 2D track-creation entry points validate coordinates.

Finding 1 of the coordinator's review of the task-4 fix: `gtrack_2d_create`,
`gtrack_2d_import`, and `gtrack_2d_import_contacts` wrote unvalidated 2D
coordinates straight into a new track's binary quad-tree files -- the same
bug class the task opens with (negative/NaN/inverted/past-chromosome
coordinates silently accepted), except worse: the bad coordinates are
persisted to disk instead of just mis-answering one query.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT

TRACK_DIR = os.path.join(str(TESTDB_ROOT), "tracks")

# chromosome "1" of the test DB is 500000 bp.


def _track_dir(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name: str) -> None:
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        import _pymisha
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def _2d(chrom1, start1, end1, chrom2=None, start2=None, end2=None):
    if chrom2 is None:
        chrom2, start2, end2 = chrom1, start1, end1
    return pd.DataFrame({
        "chrom1": [chrom1], "start1": [start1], "end1": [end1],
        "chrom2": [chrom2], "start2": [start2], "end2": [end2],
    })


class TestGtrack2dCreateCoordValidation:
    def test_negative_start_raises(self):
        tname = "test.tc_neg"
        _cleanup_track(tname)
        try:
            iv = _2d("1", -100, 1000, "1", 0, 1000)
            with pytest.raises(ValueError, match="start coordinate must be greater"):
                pm.gtrack_2d_create(tname, "d", iv, [1.0])
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_nan_raises(self):
        tname = "test.tc_nan"
        _cleanup_track(tname)
        try:
            iv = pd.DataFrame({
                "chrom1": ["1"], "start1": [np.nan], "end1": [1000.0],
                "chrom2": ["1"], "start2": [0.0], "end2": [1000.0],
            })
            with pytest.raises(ValueError, match="Invalid format of interval at index 1"):  # as R misha 5.17
                pm.gtrack_2d_create(tname, "d", iv, [1.0])
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_inverted_raises(self):
        tname = "test.tc_inv"
        _cleanup_track(tname)
        try:
            iv = _2d("1", 1000, 100, "1", 0, 1000)
            with pytest.raises(ValueError, match="lesser than end"):
                pm.gtrack_2d_create(tname, "d", iv, [1.0])
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_beyond_chromosome_raises(self):
        tname = "test.tc_oob"
        _cleanup_track(tname)
        try:
            iv = _2d("1", 0, 1_000_000_000, "1", 0, 1000)
            with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
                pm.gtrack_2d_create(tname, "d", iv, [1.0])
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_valid_intervals_still_work(self):
        """Non-regression: a valid create still succeeds."""
        tname = "test.tc_valid"
        _cleanup_track(tname)
        try:
            iv = _2d("1", 100, 200, "1", 300, 400)
            pm.gtrack_2d_create(tname, "d", iv, [1.0])
            assert pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)


class TestGtrack2dImportCoordValidation:
    def test_negative_start_in_file_raises(self, tmp_path):
        """gtrack_2d_import reads the file as R does and names the file, line and column."""
        tname = "test.ti_neg"
        _cleanup_track(tname)
        try:
            f = tmp_path / "bad.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tvalue\n"
                "1\t-100\t1000\t1\t0\t1000\t1.0\n"
            )
            with pytest.raises(ValueError, match="line 2: invalid format of start1 coordinate"):
                pm.gtrack_2d_import(tname, "d", str(f))
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)


class TestGtrack2dImportContactsCoordValidation:
    def test_inverted_interval_in_intervals_value_mode_raises(self, tmp_path):
        tname = "test.tic_inv"
        _cleanup_track(tname)
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t1000\t100\t1\t300\t400\t5.0\n"
            )
            with pytest.raises(ValueError, match="start1 coordinate exceeds or equals the end1 coordinate"):
                pm.gtrack_2d_import_contacts(tname, "d", [str(f)])
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_negative_start_in_intervals_value_mode_raises(self, tmp_path):
        tname = "test.tic_neg"
        _cleanup_track(tname)
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t-100\t100\t1\t300\t400\t5.0\n"
            )
            with pytest.raises(ValueError, match="invalid format of start1 coordinate"):
                pm.gtrack_2d_import_contacts(tname, "d", [str(f)])
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_out_of_range_fend_coord_raises(self, tmp_path):
        """A fend past the chromosome end is refused while the fends file is read."""
        tname = "test.tic_fend_oob"
        _cleanup_track(tname)
        try:
            fends_file = tmp_path / "fends.tsv"
            fends_file.write_text(
                "fend\tchr\tcoord\n"
                "1\t1\t150\n"
                "2\t1\t999999999\n"
            )
            contacts_file = tmp_path / "contacts.tsv"
            contacts_file.write_text(
                "fend1\tfend2\tcount\n"
                "1\t2\t10.0\n"
            )
            with pytest.raises(ValueError, match="line 3: coord is out of range"):
                pm.gtrack_2d_import_contacts(
                    tname, "d", [str(contacts_file)], fends=str(fends_file)
                )
            assert not pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)

    def test_valid_contacts_still_work(self, tmp_path):
        """Non-regression: both import modes still succeed for valid input."""
        tname = "test.tic_valid"
        _cleanup_track(tname)
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
            )
            pm.gtrack_2d_import_contacts(tname, "d", [str(f)])
            assert pm.gtrack_exists(tname)
        finally:
            _cleanup_track(tname)


def _write_2d_intervals_file(tmp_path, *row):
    src = tmp_path / "src.txt"
    cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "value"]
    pd.DataFrame([row], columns=cols).to_csv(src, sep="\t", index=False)
    return str(src)


class TestR2dImportStart2:
    """Port of the 2D import tests of R misha's test-gtrack.import3.R (chr1 / chr2 there)."""

    def test_gtrack_2d_import_rejects_a_negative_start2_rectangles(self, tmp_path):
        tname = "test.ti_neg_start2_rects"
        src = _write_2d_intervals_file(tmp_path, "1", 100, 200, "2", -5, 10, 1)
        with pytest.raises(ValueError, match="invalid format of start2 coordinate"):
            pm.gtrack_2d_import(tname, "negative start2", src)
        assert not pm.gtrack_exists(tname)

    def test_gtrack_2d_import_rejects_a_negative_start2_points(self, tmp_path):
        tname = "test.ti_neg_start2_points"
        src = _write_2d_intervals_file(tmp_path, "1", 100, 101, "2", -1, 0, 1)
        with pytest.raises(ValueError, match="invalid format of start2 coordinate"):
            pm.gtrack_2d_import(tname, "negative start2", src)
        assert not pm.gtrack_exists(tname)

    def test_gtrack_2d_import_contacts_rejects_a_negative_start2(self, tmp_path):
        tname = "test.tic_neg_start2"
        src = _write_2d_intervals_file(tmp_path, "1", 100, 101, "2", -5, 10, 1)
        with pytest.raises(ValueError, match="invalid format of start2 coordinate"):
            pm.gtrack_2d_import_contacts(tname, "negative start2", src)
        assert not pm.gtrack_exists(tname)

    def test_2d_importers_name_end2_when_start2_ge_end2(self, tmp_path):
        tname = "test.ti_start2_end2"
        src = _write_2d_intervals_file(tmp_path, "1", 100, 101, "2", 10, 10, 1)
        msg = "start2 coordinate exceeds or equals the end2 coordinate"
        with pytest.raises(ValueError, match=msg):
            pm.gtrack_2d_import(tname, "start2 >= end2", src)
        with pytest.raises(ValueError, match=msg):
            pm.gtrack_2d_import_contacts(tname, "start2 >= end2", src)
        assert not pm.gtrack_exists(tname)
