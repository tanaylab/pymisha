"""Tests of gtrack_liftover of a 2D source track (pm_liftover_track_2d)."""
from __future__ import annotations

import shutil
from pathlib import Path

import _pymisha
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import read_2d_track_objects, write_2d_track_file

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT


def _track_dir(name: str) -> Path:
    return TEST_DB / "tracks" / (name.replace(".", "/") + ".track")


def _cleanup_track(name: str):
    td = _track_dir(name)
    if td.exists():
        shutil.rmtree(td)
        _pymisha.pm_dbreload()


def _make_chain_df(rows):
    cols = ["chrom", "start", "end", "strand", "chromsrc", "startsrc", "endsrc",
            "strandsrc", "chain_id", "score"]
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["src_overlap_policy"] = "keep"
    df.attrs["tgt_overlap_policy"] = "keep"
    return df


def _identity_chain(src_chrom="chr1", tgt_chrom="1", length=1000, chain_id=1):
    """Identity chain: src [0, length) maps to tgt [0, length) on + strand."""
    return _make_chain_df([[tgt_chrom, 0, length, 1, src_chrom, 0, length, 1, chain_id, 1000.0]])


def _lift(tmp_path, pairs, chain, is_points=False):
    """Lift a source of the given {pair file: objects} with chain into the test DB;
    the lifted objects as sorted (chrom1, chrom2, x1, y1, x2, y2, v) rows."""
    src_dir = tmp_path / "src.track"
    src_dir.mkdir()
    for name, objs in pairs.items():
        write_2d_track_file(str(src_dir / name), objs, (0, 0, 1000, 1000), is_points=is_points)
    out_name = "test.lifted_2d"
    _cleanup_track(out_name)
    try:
        pm.gtrack_liftover(out_name, "x", str(src_dir), chain)
        rows = []
        for pair_file in sorted(_track_dir(out_name).iterdir()):
            if pair_file.name.startswith("."):
                continue
            c1, c2 = pair_file.name.split("-")
            file_is_points, objs = read_2d_track_objects(str(pair_file))
            assert file_is_points is is_points
            for o in objs:
                rows.append((c1, c2, o[0], o[1], o[0] + 1, o[1] + 1, o[2]) if is_points else (c1, c2, *o))
        return sorted(rows)
    finally:
        _cleanup_track(out_name)


@pytest.fixture(autouse=True)
def _reinit_db():
    # Re-init in case a previous test (e.g. test_path_functions.py via
    # gdb_init_examples) left the global GROOT pointing elsewhere.
    pm.gdb_init(str(TEST_DB))
    yield


class TestLiftover2DRectsIdentity:
    def test_single_rect_identity_chain(self, tmp_path):
        rows = _lift(tmp_path, {"chr1-chr1": [(10, 20, 30, 40, 1.5)]}, _identity_chain("chr1", "1", 1000))
        assert rows == [("chr1", "chr1", 10, 20, 30, 40, 1.5)]

    def test_multi_rect_identity_chain(self, tmp_path):
        rows = _lift(tmp_path, {"chr1-chr1": [(10, 20, 30, 40, 1.5), (100, 200, 150, 250, 2.5)]},
                     _identity_chain("chr1", "1", 1000))
        assert rows == sorted([
            ("chr1", "chr1", 10, 20, 30, 40, 1.5),
            ("chr1", "chr1", 100, 200, 150, 250, 2.5),
        ])

    def test_cross_chrom_pair(self, tmp_path):
        chain = _make_chain_df([
            ["1", 0, 1000, 0, "chr1", 0, 1000, 0, 1, 1000.0],
            ["2", 0, 1000, 0, "chr2", 0, 1000, 0, 2, 1000.0],
        ])
        rows = _lift(tmp_path, {"chr1-chr2": [(10, 20, 30, 40, 1.5)]}, chain)
        assert rows == [("chr1", "chr2", 10, 20, 30, 40, 1.5)]


class TestLiftover2DDropsRectsThatDontMatchChain:
    def test_no_chain_for_chrom1_drops(self, tmp_path):
        # Chain has entries only for source chr99 - matches nothing.
        rows = _lift(tmp_path, {"chr1-chr1": [(10, 20, 30, 40, 1.5)]}, _identity_chain("chr99", "1", 1000))
        assert rows == []

    def test_no_chain_for_chrom2_drops(self, tmp_path):
        # Chain matches chr1 (x-dim) but not chr2 (y-dim).
        rows = _lift(tmp_path, {"chr1-chr2": [(10, 20, 30, 40, 1.5)]}, _identity_chain("chr1", "1", 1000))
        assert rows == []


class TestLiftover2DMultiBlockSplit:
    def test_x_dim_chain_splits_rect_into_two(self, tmp_path):
        # X-dim chain has TWO blocks with a gap between target intervals:
        #   src [0, 50) -> tgt [0, 50)
        #   src [50, 100) -> tgt [100, 150)  (gap at tgt [50, 100))
        # The y-dim [20, 40) maps by the first block as is.
        # Source rect (10, 20, 80, 40, 1.5) spans both x-blocks: two target rects.
        chain = _make_chain_df([
            ["1", 0, 50, 0, "chr1", 0, 50, 0, 1, 1000.0],
            ["1", 100, 150, 0, "chr1", 50, 100, 0, 2, 1000.0],
        ])
        rows = _lift(tmp_path, {"chr1-chr1": [(10, 20, 80, 40, 1.5)]}, chain)
        assert rows == sorted([
            ("chr1", "chr1", 10, 20, 50, 40, 1.5),
            ("chr1", "chr1", 100, 20, 130, 40, 1.5),
        ])


class TestLiftover2DPoints:
    def test_single_point_identity(self, tmp_path):
        rows = _lift(tmp_path, {"chr1-chr1": [(10, 20, 1.5)]}, _identity_chain("chr1", "1", 1000), is_points=True)
        assert rows == [("chr1", "chr1", 10, 20, 11, 21, 1.5)]


class TestLiftover2DNegStrand:
    def test_neg_strand_on_x_dim_reflects_coords(self, tmp_path):
        # A negative-strand block src [0, 100) -> tgt [0, 100) reflects an interval:
        # src [10, 30) -> tgt [100 - 30, 100 - 10) = [70, 90), and the y-dim
        # src [20, 40) -> [60, 80).
        chain = _make_chain_df([["1", 0, 100, -1, "chr1", 0, 100, 1, 1, 1000.0]])  # R strands
        rows = _lift(tmp_path, {"chr1-chr1": [(10, 20, 30, 40, 1.5)]}, chain)
        assert rows == [("chr1", "chr1", 70, 60, 90, 80, 1.5)]


class TestLiftover2DAggregatesOverlaps:
    """Overlapping mapped rects are cut into disjoint cells, each holding the aggregate of
    the rects covering it (R 5.11.8)."""

    @staticmethod
    def _two_blocks_onto_one():
        # src [0, 100) and [500, 600) both land on tgt [0, 100)
        return _make_chain_df([
            ["1", 0, 100, 0, "chr1", 0, 100, 0, 1, 1000.0],
            ["1", 0, 100, 0, "chr1", 500, 600, 0, 2, 1000.0],
        ])

    def test_fully_overlapping_rects_aggregated(self, tmp_path):
        rows = _lift(tmp_path, {"chr1-chr1": [(0, 0, 10, 10, 4.0), (500, 500, 510, 510, 8.0)]},
                     self._two_blocks_onto_one())
        assert rows == [("chr1", "chr1", 0, 0, 10, 10, 6.0)]

    def test_partial_overlap_splits_into_disjoint_cells(self, tmp_path):
        # A: x[0,10] y[0,30]=2 ; B: x[0,10] y[10,20]=6 (overlap in y[10,20]).
        rows = _lift(tmp_path, {"chr1-chr1": [(0, 0, 10, 30, 2.0), (500, 510, 510, 520, 6.0)]},
                     self._two_blocks_onto_one())
        assert rows == [
            ("chr1", "chr1", 0, 0, 10, 10, 2.0),    # A only
            ("chr1", "chr1", 0, 10, 10, 20, 4.0),   # mean(A, B)
            ("chr1", "chr1", 0, 20, 10, 30, 2.0),   # A only
        ]


# ---------------------------------------------------------------------------
# End-to-end dispatcher tests via gtrack_liftover (Python public API)
# ---------------------------------------------------------------------------


class TestGtrackLiftoverDispatcher2D:
    """End-to-end via pm.gtrack_liftover (the Python public API).

    Requires the session-scoped test DB initialized by conftest.py.
    """

    @pytest.fixture(autouse=True)
    def _reinit_db(self):
        # Re-init in case a previous test (e.g. test_path_functions.py via
        # gdb_init_examples) left the global GROOT pointing elsewhere.
        pm.gdb_init(str(TEST_DB))
        yield

    def test_rects_end_to_end(self, tmp_path):
        # Build source 2D track outside the DB.
        src_dir = tmp_path / "src_2d_rects.track"
        src_dir.mkdir()
        write_2d_track_file(
            str(src_dir / "src1-src1"),
            [(10, 20, 30, 40, 1.5),
             (100, 200, 150, 250, 2.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )

        chain = _make_chain_df([
            ["1", 0, 1000, 0, "src1", 0, 1000, 0, 1, 1000.0],
        ])

        out_name = "test.lifted_2d_rects"
        _cleanup_track(out_name)
        try:
            pm.gtrack_liftover(out_name, "lifted 2d rects", str(src_dir), chain)
            # Verify the output via direct quadtree read.
            out_dir = _track_dir(out_name)
            pair_file = out_dir / "chr1-chr1"
            assert pair_file.exists(), f"expected 2D pair file at {pair_file}, found: {list(out_dir.iterdir())}"
            is_points, objs = read_2d_track_objects(str(pair_file))
            assert is_points is False
            got = sorted((x1, y1, x2, y2, v) for x1, y1, x2, y2, v in objs)
            assert got == sorted([(10, 20, 30, 40, 1.5),
                                  (100, 200, 150, 250, 2.5)])
        finally:
            _cleanup_track(out_name)

    def test_points_end_to_end(self, tmp_path):
        src_dir = tmp_path / "src_2d_points.track"
        src_dir.mkdir()
        write_2d_track_file(
            str(src_dir / "src1-src1"),
            [(10, 20, 1.5), (100, 200, 2.5)],
            (0, 0, 1000, 1000),
            is_points=True,
        )

        chain = _make_chain_df([
            ["1", 0, 1000, 0, "src1", 0, 1000, 0, 1, 1000.0],
        ])

        out_name = "test.lifted_2d_points"
        _cleanup_track(out_name)
        try:
            pm.gtrack_liftover(out_name, "lifted 2d points", str(src_dir), chain)
            out_dir = _track_dir(out_name)
            pair_file = out_dir / "chr1-chr1"
            assert pair_file.exists()
            is_points, objs = read_2d_track_objects(str(pair_file))
            assert is_points is True
            # POINTS objects from read_2d_track_objects are (x, y, val).
            got = sorted(objs)
            assert got == sorted([(10, 20, 1.5), (100, 200, 2.5)])
        finally:
            _cleanup_track(out_name)

    def test_no_chain_match_produces_empty_track(self, tmp_path):
        src_dir = tmp_path / "src_2d_nochain.track"
        src_dir.mkdir()
        write_2d_track_file(
            str(src_dir / "src1-src1"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        # Chain references chrom 'other' which doesn't match 'src1' -> no output.
        chain = _make_chain_df([
            ["1", 0, 1000, 0, "other", 0, 1000, 0, 1, 1000.0],
        ])

        out_name = "test.lifted_2d_empty"
        _cleanup_track(out_name)
        try:
            pm.gtrack_liftover(out_name, "empty 2d", str(src_dir), chain)
            out_dir = _track_dir(out_name)
            assert out_dir.exists()
            # No per-pair files - the track is empty but created.
            data_files = [
                p for p in out_dir.iterdir()
                if p.name not in (".attributes", "track.idx", "track.dat")
            ]
            assert len(data_files) == 0
        finally:
            _cleanup_track(out_name)


class TestDetectSourceTrack2d:
    """The dispatcher detection helper itself."""

    def test_detects_rects_pair_file(self, tmp_path):
        from pymisha.liftover import _detect_source_track_2d
        write_2d_track_file(
            str(tmp_path / "1-1"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        assert _detect_source_track_2d(str(tmp_path)) is True

    def test_detects_points_pair_file(self, tmp_path):
        from pymisha.liftover import _detect_source_track_2d
        write_2d_track_file(
            str(tmp_path / "1-1"),
            [(10, 20, 1.5)],
            (0, 0, 1000, 1000),
            is_points=True,
        )
        assert _detect_source_track_2d(str(tmp_path)) is True

    def test_returns_false_for_1d_dense(self, tmp_path):
        import struct as _s

        from pymisha.liftover import _detect_source_track_2d
        with open(tmp_path / "src1", "wb") as f:
            f.write(_s.pack("<i", 100))  # dense 1D signature (bin_size > 0)
            f.write(_s.pack("<f", 1.0))
        assert _detect_source_track_2d(str(tmp_path)) is False

    def test_returns_false_for_1d_sparse(self, tmp_path):
        import struct as _s

        from pymisha.liftover import _detect_source_track_2d
        with open(tmp_path / "src1", "wb") as f:
            f.write(_s.pack("<i", -1))  # sparse 1D signature
        assert _detect_source_track_2d(str(tmp_path)) is False

    def test_returns_false_for_empty_dir(self, tmp_path):
        from pymisha.liftover import _detect_source_track_2d
        assert _detect_source_track_2d(str(tmp_path)) is False

    def test_returns_false_for_missing_dir(self, tmp_path):
        from pymisha.liftover import _detect_source_track_2d
        assert _detect_source_track_2d(str(tmp_path / "nope")) is False
