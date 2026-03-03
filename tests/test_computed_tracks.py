"""Tests for COMPUTED track detection and informative error messages."""

import os
import shutil
import struct

import pytest

import pymisha as pm
import _pymisha

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)

COMPUTED_TRACK_NAME = "test.computed_stub"
COMPUTED_SIGNATURE = -11  # GenomeTrack::FORMAT_SIGNATURES[COMPUTED]


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
        f.write(struct.pack("<i", COMPUTED_SIGNATURE))
        # Pad with zeros — enough for get_type() to succeed without
        # reading further into the file.
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
    """Create the COMPUTED track before the module, remove after."""
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
