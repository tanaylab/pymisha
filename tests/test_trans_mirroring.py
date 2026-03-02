"""Tests for trans contact mirroring in gtrack_2d_import_contacts.

R misha writes both chrA-chrB and chrB-chrA files for trans contacts.
PyMisha must do the same so queries work in both directions.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import _pymisha
import pymisha as pm

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestTransContactMirroring:
    """Trans contacts must be queryable in both chrom-pair directions."""

    TRACK = "test.trans_mirror"

    def _cleanup(self):
        _cleanup_track(self.TRACK)

    def test_trans_both_files_exist(self, tmp_path):
        """Importing a trans contact creates both chrA-chrB and chrB-chrA files."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            # Trans contact: chrom 1 -> chrom 2
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t2\t5000\t6000\t7.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "trans test", [str(f)])

            tdir = _track_dir(self.TRACK)
            files = sorted(
                fn for fn in os.listdir(tdir) if not fn.startswith(".")
            )
            # Both 1-2 and 2-1 files must exist
            assert "1-2" in files, f"Expected 1-2 in {files}"
            assert "2-1" in files, f"Expected 2-1 in {files}"
        finally:
            self._cleanup()

    def test_trans_query_canonical_direction(self, tmp_path):
        """Querying trans contacts in canonical direction (chrA-chrB) works."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t2\t5000\t6000\t7.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "trans test", [str(f)])

            # Query in canonical direction: chrom1=1, chrom2=2
            # Midpoints: (150, 5500)
            intervals = pd.DataFrame({
                "chrom1": ["1"], "start1": [0], "end1": [500000],
                "chrom2": ["2"], "start2": [0], "end2": [300000],
            })
            result = pm.gextract(self.TRACK, intervals)
            assert result is not None and len(result) > 0, \
                "Expected contacts in canonical direction 1-2"
            assert float(result.iloc[0][self.TRACK]) == 7.0
        finally:
            self._cleanup()

    def test_trans_query_reverse_direction(self, tmp_path):
        """Querying trans contacts in reverse direction (chrB-chrA) works."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t2\t5000\t6000\t7.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "trans test", [str(f)])

            # Query in reverse direction: chrom1=2, chrom2=1
            # The mirrored contact should be at (5500, 150) in the 2-1 file
            intervals = pd.DataFrame({
                "chrom1": ["2"], "start1": [0], "end1": [300000],
                "chrom2": ["1"], "start2": [0], "end2": [500000],
            })
            result = pm.gextract(self.TRACK, intervals)
            assert result is not None and len(result) > 0, \
                "Expected contacts in reverse direction 2-1"
            assert float(result.iloc[0][self.TRACK]) == 7.0
        finally:
            self._cleanup()

    def test_trans_mirrored_coordinates_swapped(self, tmp_path):
        """Mirrored trans contact has swapped coordinates."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            # Midpoints: (150, 5500)
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t2\t5000\t6000\t7.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "trans test", [str(f)])

            # Query canonical (1-2): contact at x=150, y=5500
            ivs_canon = pd.DataFrame({
                "chrom1": ["1"], "start1": [0], "end1": [500000],
                "chrom2": ["2"], "start2": [0], "end2": [300000],
            })
            res_canon = pm.gextract(self.TRACK, ivs_canon)
            assert res_canon is not None and len(res_canon) > 0
            # start1 should be 150 (midpoint of 100-200)
            assert int(res_canon.iloc[0]["start1"]) == 150
            # start2 should be 5500 (midpoint of 5000-6000)
            assert int(res_canon.iloc[0]["start2"]) == 5500

            # Query reverse (2-1): contact at x=5500, y=150 (swapped)
            ivs_rev = pd.DataFrame({
                "chrom1": ["2"], "start1": [0], "end1": [300000],
                "chrom2": ["1"], "start2": [0], "end2": [500000],
            })
            res_rev = pm.gextract(self.TRACK, ivs_rev)
            assert res_rev is not None and len(res_rev) > 0
            # In the 2-1 file, x=5500, y=150 (swapped from original)
            assert int(res_rev.iloc[0]["start1"]) == 5500
            assert int(res_rev.iloc[0]["start2"]) == 150
        finally:
            self._cleanup()

    def test_cis_contacts_unaffected(self, tmp_path):
        """Cis contacts (same chrom) still work correctly -- no extra files."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "cis test", [str(f)])

            tdir = _track_dir(self.TRACK)
            files = sorted(
                fn for fn in os.listdir(tdir) if not fn.startswith(".")
            )
            # Only 1-1 file should exist (no mirroring for cis -- cis mirroring
            # is handled at the contact level, not at the file level)
            chrom_pair_files = [fn for fn in files if "-" in fn]
            assert chrom_pair_files == ["1-1"], \
                f"Expected only 1-1 for cis contacts, got {chrom_pair_files}"

            # Querying should still work
            intervals = pd.DataFrame({
                "chrom1": ["1"], "start1": [0], "end1": [500000],
                "chrom2": ["1"], "start2": [0], "end2": [500000],
            })
            result = pm.gextract(self.TRACK, intervals)
            assert result is not None and len(result) > 0
        finally:
            self._cleanup()

    def test_mixed_cis_and_trans(self, tmp_path):
        """Import with both cis and trans contacts creates correct files."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
                "1\t1000\t2000\t2\t5000\t6000\t3.0\n"
                "2\t1000\t2000\tX\t5000\t6000\t9.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "mixed test", [str(f)])

            tdir = _track_dir(self.TRACK)
            files = sorted(
                fn for fn in os.listdir(tdir) if not fn.startswith(".")
            )
            # Expect: 1-1 (cis), 1-2 + 2-1 (trans), 2-X + X-2 (trans)
            assert "1-1" in files
            assert "1-2" in files
            assert "2-1" in files
            assert "2-X" in files
            assert "X-2" in files
        finally:
            self._cleanup()

    def test_trans_reverse_input_order(self, tmp_path):
        """Trans contacts provided in reverse order (chrB < chrA) still produce both files."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            # Input is chrom2=1 -> chrom1=2 (reverse of canonical)
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "2\t5000\t6000\t1\t100\t200\t7.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "reverse test", [str(f)])

            tdir = _track_dir(self.TRACK)
            files = sorted(
                fn for fn in os.listdir(tdir) if not fn.startswith(".")
            )
            assert "1-2" in files, f"Expected 1-2 in {files}"
            assert "2-1" in files, f"Expected 2-1 in {files}"

            # Both queries should return results
            ivs_12 = pd.DataFrame({
                "chrom1": ["1"], "start1": [0], "end1": [500000],
                "chrom2": ["2"], "start2": [0], "end2": [300000],
            })
            ivs_21 = pd.DataFrame({
                "chrom1": ["2"], "start1": [0], "end1": [300000],
                "chrom2": ["1"], "start2": [0], "end2": [500000],
            })
            res_12 = pm.gextract(self.TRACK, ivs_12)
            res_21 = pm.gextract(self.TRACK, ivs_21)
            assert res_12 is not None and len(res_12) > 0
            assert res_21 is not None and len(res_21) > 0
            assert float(res_12.iloc[0][self.TRACK]) == 7.0
            assert float(res_21.iloc[0][self.TRACK]) == 7.0
        finally:
            self._cleanup()

    def test_trans_multiple_contacts_per_pair(self, tmp_path):
        """Multiple trans contacts on the same chrom pair are all mirrored."""
        self._cleanup()
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t2\t5000\t6000\t7.0\n"
                "1\t1000\t2000\t2\t10000\t11000\t3.0\n"
                "1\t2000\t3000\t2\t15000\t16000\t1.0\n"
            )
            pm.gtrack_2d_import_contacts(self.TRACK, "multi trans", [str(f)])

            # Query canonical direction
            ivs_12 = pd.DataFrame({
                "chrom1": ["1"], "start1": [0], "end1": [500000],
                "chrom2": ["2"], "start2": [0], "end2": [300000],
            })
            res_12 = pm.gextract(self.TRACK, ivs_12)
            assert res_12 is not None and len(res_12) == 3

            # Query reverse direction
            ivs_21 = pd.DataFrame({
                "chrom1": ["2"], "start1": [0], "end1": [300000],
                "chrom2": ["1"], "start2": [0], "end2": [500000],
            })
            res_21 = pm.gextract(self.TRACK, ivs_21)
            assert res_21 is not None and len(res_21) == 3

            # Values should match (same set of values in both directions)
            vals_12 = sorted(res_12[self.TRACK].to_numpy())
            vals_21 = sorted(res_21[self.TRACK].to_numpy())
            np.testing.assert_array_equal(vals_12, vals_21)
        finally:
            self._cleanup()
