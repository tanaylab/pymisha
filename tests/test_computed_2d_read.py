"""Read-side tests for COMPUTED 2D tracks (G3 KICKOFF-8)."""

from __future__ import annotations

from pathlib import Path

import pytest

import pymisha as pm
from pymisha._quadtree import (
    SIGNATURE_COMPUTED,
    SIGNATURE_POINTS,
    SIGNATURE_RECTS,
    _read_file_header,
)

R_TESTDB = "/net/mraid20/ifs/wisdom/tanay_lab/tgdata/db/tgdb/misha_test_db_indexed/"


def test_signatures_constants_present():
    assert SIGNATURE_RECTS == -9
    assert SIGNATURE_POINTS == -10
    assert SIGNATURE_COMPUTED == -11


def test_read_file_header_recognises_computed():
    chunk = Path(R_TESTDB) / "tracks/test/computed2d.track/chr10-chr8"
    if not chunk.exists():
        pytest.skip(f"R test DB fixture not present at {chunk}")
    is_points, num_objs, data = _read_file_header(str(chunk))
    try:
        # COMPUTED tracks share the 48-byte Obj layout with RECTS, so the
        # `_read_file_header` API surfaces them as ``is_points = False``.
        assert is_points is False
        assert num_objs > 0
    finally:
        data.close()


def test_file_track_kind_detects_computed():
    from pymisha._quadtree import _file_track_kind

    chunk = Path(R_TESTDB) / "tracks/test/computed2d.track/chr10-chr8"
    if not chunk.exists():
        pytest.skip(f"R test DB fixture not present at {chunk}")
    assert _file_track_kind(str(chunk)) == "COMPUTED"


# --------------------------------------------------------------------------- #
# Computer2D implementations
# --------------------------------------------------------------------------- #

from pymisha._computer2d import (  # noqa: E402
    CT2_AREA,
    CT2_POTENTIAL,
    CT2_TECHNICAL,
    CT2_TEST,
    AreaComputer2D,
    DiagonalBand,
    Rectangle,
    TestComputer2D,
    create_computer2d,
    intersected_area,
)


def test_test_computer_compute_no_band():
    c = TestComputer2D()
    r = Rectangle(100, 200, 300, 400)
    assert c.compute(r) == (100 + 300 + 200 + 400) % 10_000_000


def test_test_computer_compute_with_band():
    c = TestComputer2D()
    r = Rectangle(100, 200, 300, 400)
    band = DiagonalBand(d1=-50, d2=50)
    assert c.compute(r, band) == (100 + 300 + 200 + 400 - 50 + 50) % 10_000_000


def test_area_computer_no_band():
    c = AreaComputer2D()
    r = Rectangle(0, 0, 100, 100)
    assert c.compute(r) == 1.0


def test_area_computer_with_band_full_overlap():
    c = AreaComputer2D()
    r = Rectangle(0, 0, 100, 100)
    band = DiagonalBand(d1=-10000, d2=10000)
    assert c.compute(r, band) == 1.0


def test_area_computer_with_band_no_overlap():
    c = AreaComputer2D()
    # Rect well above the diagonal, narrow band hugging the diagonal.
    r = Rectangle(0, 1000, 100, 1100)
    band = DiagonalBand(d1=-10, d2=10)
    assert c.compute(r, band) == 0.0


def test_area_computer_with_band_partial_overlap_matches_helper():
    c = AreaComputer2D()
    r = Rectangle(0, 0, 100, 100)
    band = DiagonalBand(d1=-10, d2=10)
    expected = intersected_area(r, band) / (100 * 100)
    assert c.compute(r, band) == expected


def test_create_computer2d_dispatch():
    assert isinstance(create_computer2d(CT2_AREA), AreaComputer2D)
    assert isinstance(create_computer2d(CT2_TEST), TestComputer2D)
    with pytest.raises(NotImplementedError, match="CT2_POTENTIAL"):
        create_computer2d(CT2_POTENTIAL)
    with pytest.raises(NotImplementedError, match="CT2_TECHNICAL"):
        create_computer2d(CT2_TECHNICAL)
    with pytest.raises(ValueError, match="Unknown"):
        create_computer2d(99)
