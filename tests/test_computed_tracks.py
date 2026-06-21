"""Tests for unsupported COMPUTED computer types - clear errors only.

Since v0.8.4, COMPUTED tracks backed by ``CT2_AREA`` (=0) or ``CT2_TEST``
(=3) read end-to-end; this file only exercises the still-rejected types
``CT2_POTENTIAL`` (=1) and ``CT2_TECHNICAL`` (=2) which raise informative
``NotImplementedError``s pointing the user at R misha.
"""

import os
import shutil
import struct

import pytest

import pymisha as pm
import _pymisha

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT
TRACK_DIR = str(TEST_DB / "tracks")

COMPUTED_TRACK_NAME = "test.computed_stub"
COMPUTED_SIGNATURE = -11  # GenomeTrack::FORMAT_SIGNATURES[COMPUTED]
CT2_POTENTIAL = 1  # PotentialComputer2D - not yet ported, raises NotImplementedError.


def _track_path(name):
    """Get track directory path (dots become subdirectories)."""
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _create_computed_track():
    """Create a minimal COMPUTED track fixture.

    Writes a per-chrom-pair file with the COMPUTED format signature (-11)
    as its first 4 bytes, plus enough dummy data so the quad-tree reader
    does not crash when probing the file.  The file is just enough for
    ``GenomeTrack::get_type()`` to identify the track as COMPUTED.
    """
    tdir = _track_path(COMPUTED_TRACK_NAME)
    os.makedirs(tdir, exist_ok=True)

    # Write a minimal 2D file named "1-1" with the COMPUTED signature.
    # The file just needs the int32 signature so get_type() recognizes it.
    # We add some zero padding so buffered reads don't fail.
    filepath = os.path.join(tdir, "1-1")
    with open(filepath, "wb") as f:
        # Signature -11 + Computer2DType byte = 1 (CT2_POTENTIAL).
        # Since v0.8.4, CT2_AREA (=0) / CT2_TEST (=3) are SUPPORTED so we
        # need an unsupported computer type to keep the NotImplementedError
        # path in scope.
        f.write(struct.pack("<i", COMPUTED_SIGNATURE))
        f.write(struct.pack("<i", CT2_POTENTIAL))
        # Pad so buffered reads don't fail probing the file.
        f.write(b"\x00" * 256)

    _pymisha.pm_dbreload()


def _cleanup_computed_track():
    """Remove the COMPUTED track fixture and reload the DB."""
    tdir = _track_path(COMPUTED_TRACK_NAME)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
    _pymisha.pm_dbreload()


@pytest.fixture(scope="module", autouse=True)
def computed_track_fixture():
    """Create the COMPUTED track before the module, remove after.

    Earlier tests in the session may have left the global db pointed at a
    temp copy from ``gdb_init_examples()``.  Re-init the canonical test db
    here so the on-disk track file written below is the one ``pm_dbreload``
    actually sees.
    """
    pm.gdb_init(str(TEST_DB))
    _create_computed_track()
    yield
    _cleanup_computed_track()


class TestComputedTrackDetection:
    """Test that COMPUTED tracks are detected and raise clear errors."""

    def test_gtrack_info_identifies_computed_type(self):
        """gtrack_info must report type='computed' for a COMPUTED track."""
        info = pm.gtrack_info(COMPUTED_TRACK_NAME)
        assert info["type"] == "computed"
        assert info["dimensions"] == 2

    def test_gextract_raises_not_implemented(self):
        """gextract must raise NotImplementedError for COMPUTED tracks."""
        intervals = pm.gintervals_2d(
            chroms1=["1"], starts1=[0], ends1=[1000],
            chroms2=["1"], starts2=[0], ends2=[1000],
        )
        with pytest.raises(NotImplementedError, match="COMPUTED"):
            pm.gextract(COMPUTED_TRACK_NAME, intervals=intervals)

    def test_gsummary_raises_not_implemented(self):
        """gsummary must raise NotImplementedError for COMPUTED tracks."""
        with pytest.raises(NotImplementedError, match="COMPUTED"):
            pm.gsummary(COMPUTED_TRACK_NAME)

    def test_gquantiles_raises_not_implemented(self):
        """gquantiles must raise NotImplementedError for COMPUTED tracks."""
        with pytest.raises(NotImplementedError, match="COMPUTED"):
            pm.gquantiles(COMPUTED_TRACK_NAME)

    def test_gdist_raises_not_implemented(self):
        """gdist must raise NotImplementedError for COMPUTED tracks."""
        with pytest.raises(NotImplementedError, match="COMPUTED"):
            pm.gdist(COMPUTED_TRACK_NAME, [0, 1, 2])

    def test_gscreen_raises_not_implemented(self):
        """gscreen must raise NotImplementedError for COMPUTED tracks."""
        with pytest.raises(NotImplementedError, match="COMPUTED"):
            pm.gscreen(f"{COMPUTED_TRACK_NAME} > 0")

    def test_error_message_contains_track_name(self):
        """The error message must mention the specific track name."""
        with pytest.raises(NotImplementedError, match=COMPUTED_TRACK_NAME):
            pm.gextract(COMPUTED_TRACK_NAME, intervals=pm.gintervals("1", 0, 1000))

    def test_error_message_contains_guidance(self):
        """The error message should suggest using R misha."""
        with pytest.raises(NotImplementedError, match="R misha"):
            pm.gsummary(COMPUTED_TRACK_NAME)

    def test_computed_in_expression_detected(self):
        """COMPUTED tracks used in compound expressions are detected."""
        expr = f"{COMPUTED_TRACK_NAME} + 1"
        with pytest.raises(NotImplementedError, match="COMPUTED"):
            pm.gsummary(expr)

    def test_non_computed_tracks_pass(self):
        """Normal tracks must not be affected by the COMPUTED check."""
        # This should work without raising NotImplementedError
        result = pm.gsummary("dense_track")
        assert result is not None
