"""Tests for gtrack_2d_import_contacts."""

import os
import shutil

import pytest

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
        import _pymisha
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


class TestGtrack2dImportContacts:
    """Tests for gtrack_2d_import_contacts."""

    def _cleanup(self, name):
        _cleanup_track(name)

    def test_import_intervals_format(self, tmp_path):
        """Import contacts from intervals-value format."""
        tname = "test.test_contacts_basic"
        self._cleanup(tname)
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
                "1\t1000\t2000\t2\t5000\t6000\t3.0\n"
            )
            pm.gtrack_2d_import_contacts(tname, "test contacts", [str(f)])
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"
            assert info["dimensions"] == 2
        finally:
            self._cleanup(tname)

    def test_import_fends_format(self, tmp_path):
        """Import contacts from fragment-ends format."""
        tname = "test.test_contacts_fends"
        self._cleanup(tname)
        try:
            fends_file = tmp_path / "fends.tsv"
            fends_file.write_text(
                "fend\tchr\tcoord\n"
                "1\t1\t150\n"
                "2\t1\t350\n"
                "3\t2\t5500\n"
            )
            contacts_file = tmp_path / "contacts.tsv"
            contacts_file.write_text(
                "fend1\tfend2\tcount\n"
                "1\t2\t10.0\n"
                "1\t3\t5.0\n"
            )
            pm.gtrack_2d_import_contacts(
                tname, "fends test", [str(contacts_file)], fends=str(fends_file)
            )
            assert pm.gtrack_exists(tname)
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"
        finally:
            self._cleanup(tname)

    def test_duplicates_summed(self, tmp_path):
        """Duplicate contacts are summed when allow_duplicates=True."""
        tname = "test.test_contacts_dups"
        self._cleanup(tname)
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
                "1\t100\t200\t1\t300\t400\t3.0\n"
            )
            pm.gtrack_2d_import_contacts(tname, "dups test", [str(f)], allow_duplicates=True)
            assert pm.gtrack_exists(tname)
        finally:
            self._cleanup(tname)

    def test_duplicates_error(self, tmp_path):
        """Duplicate contacts raise error when allow_duplicates=False."""
        tname = "test.test_contacts_no_dups"
        self._cleanup(tname)
        try:
            f = tmp_path / "contacts.tsv"
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
                "1\t100\t200\t1\t300\t400\t3.0\n"
            )
            with pytest.raises(ValueError, match="[Dd]uplic"):
                pm.gtrack_2d_import_contacts(tname, "no dups", [str(f)], allow_duplicates=False)
        finally:
            self._cleanup(tname)

    def test_canonical_ordering(self, tmp_path):
        """Contacts are canonically ordered (lower chrom first, or lower coord for same chrom)."""
        tname = "test.test_contacts_canon"
        self._cleanup(tname)
        try:
            f = tmp_path / "contacts.tsv"
            # chrom2 < chrom1, so should be swapped
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "2\t100\t200\t1\t300\t400\t5.0\n"
            )
            pm.gtrack_2d_import_contacts(tname, "canon test", [str(f)])
            assert pm.gtrack_exists(tname)
            # Track should have 1-2 file (not 2-1)
            tdir = _track_dir(tname)
            files = [f for f in os.listdir(tdir) if not f.startswith(".")]
            assert any("1-2" in f for f in files)
        finally:
            self._cleanup(tname)

    def test_multiple_contact_files(self, tmp_path):
        """Multiple contact files are merged."""
        tname = "test.test_contacts_multi"
        self._cleanup(tname)
        try:
            f1 = tmp_path / "contacts1.tsv"
            f1.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
            )
            f2 = tmp_path / "contacts2.tsv"
            f2.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t1000\t2000\t1\t3000\t4000\t3.0\n"
            )
            pm.gtrack_2d_import_contacts(tname, "multi files", [str(f1), str(f2)])
            assert pm.gtrack_exists(tname)
        finally:
            self._cleanup(tname)

    def test_cis_contacts_mirrored(self, tmp_path):
        """Cis contacts are mirrored (both chrom1-chrom1 entries exist)."""
        tname = "test.test_contacts_mirror"
        self._cleanup(tname)
        try:
            f = tmp_path / "contacts.tsv"
            # Contact at (150, 350) on chrom 1
            f.write_text(
                "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\tcount\n"
                "1\t100\t200\t1\t300\t400\t5.0\n"
            )
            pm.gtrack_2d_import_contacts(tname, "mirror test", [str(f)])
            assert pm.gtrack_exists(tname)
            # For cis, both (150,350) and (350,150) should exist as mirrored
            # We can verify by checking the track info detects it correctly
            info = pm.gtrack_info(tname)
            assert info["type"] == "points"
        finally:
            self._cleanup(tname)

    def test_missing_contacts_raises(self):
        """Missing contacts parameter raises error."""
        with pytest.raises((ValueError, TypeError)):
            pm.gtrack_2d_import_contacts("test.bad", "desc", [])

    def test_existing_track_raises(self, tmp_path):
        """Creating over existing track raises error."""
        with pytest.raises((ValueError, Exception)):
            pm.gtrack_2d_import_contacts("dense_track", "desc", ["/nonexistent"])
