"""R-parity / robustness: `_gextract_2d_via_scanner` must accept ``chr``-prefixed
chrom names against a DB that stores them without the prefix (and vice versa).

Previously the scanner did a raw ``cmap.get(c, -1)`` on the input chrom names,
so a DataFrame with ``chrom1='chr1'`` against a DB with chrom ``'1'`` produced
``chromid=-1``. Combined with a pre-registered vtrack (global state) this
crashed the C++ scanner with ``std::bad_alloc`` / ``basic_string::append``.

The fix routes the chrom column through ``_normalize_chroms`` (which already
handles the prefix swap, same way the 1D ``gextract`` and ``gintervals_load``
paths do) before mapping to chromid. The test pins both behaviors: a clean
result (not a crash, not all-NaN) AND no chrom round-trip drift.
"""

from __future__ import annotations

import os
import shutil

import _pymisha
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import write_2d_track_file

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup(name: str) -> None:
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture()
def simple_2d_rects(_init_db):
    """One 10kb x 10kb diagonal rect on chrom 1 x chrom 1, value 1.0."""
    tname = "test.scanner_chrom_prefix"
    _cleanup(tname)
    tdir = _track_dir(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")
    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [(10_000, 10_000, 20_000, 20_000, 1.0)],
        (0, 0, 500_000, 500_000),
        is_points=False,
    )
    _pymisha.pm_dbreload()
    yield tname
    _cleanup(tname)


@pytest.fixture(autouse=True)
def _clean_vtracks():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


# --------------------------------------------------------------------------- #
# Bug: chr-prefixed chrom + vtrack state + band crashes the C++ scanner.
# Fix pins: same call returns the rect and the chrom comes back unprefixed
# (matching the DB's actual chrom names).
# --------------------------------------------------------------------------- #


class TestScannerNormalizesChromPrefix:
    """`gextract` over a 2D scope passed with ``chr`` prefix must work against
    a DB whose chroms are stored without the prefix (the local test DB)."""

    def test_chr_prefixed_input_with_vtrack_and_band(self, simple_2d_rects):
        pm.gvtrack_create("vt", simple_2d_rects, "avg")
        # Note the ``chr`` prefix -- the local test DB stores chroms as ``1``.
        grid = pd.DataFrame({
            "chrom1": ["chr1"], "start1": [0], "end1": [50_000],
            "chrom2": ["chr1"], "start2": [0], "end2": [50_000],
        })
        r = pm.gextract("vt", grid, iterator=grid, band=(-100_000, -100))
        # Previously this raised std::bad_alloc (vtrack + chromid=-1 + band).
        assert r is not None
        assert len(r) == 1
        # The track has a single rect at (10k, 20k)^2 (val=1.0); with the band
        # shrinking it to a strip the avg is still 1.0.
        assert float(r["vt"].iloc[0]) == pytest.approx(1.0)
        # The returned chrom matches the DB's storage convention, not the input
        # prefix (callers can re-prefix downstream if needed).
        assert str(r["chrom1"].iloc[0]) == "1"
        assert str(r["chrom2"].iloc[0]) == "1"

    def test_chr_prefixed_input_no_band(self, simple_2d_rects):
        """Same as above without a band -- pins the normalization in isolation
        (the crash needed band + vtrack state, but normalization should always
        produce a real chromid)."""
        pm.gvtrack_create("vt", simple_2d_rects, "avg")
        grid = pd.DataFrame({
            "chrom1": ["chr1"], "start1": [0], "end1": [50_000],
            "chrom2": ["chr1"], "start2": [0], "end2": [50_000],
        })
        r = pm.gextract("vt", grid, iterator=grid)
        assert r is not None
        assert len(r) == 1
        assert float(r["vt"].iloc[0]) == pytest.approx(1.0)
