"""Cross-validation tests for pm_liftover_track_2d (G1.P3.D)."""
from __future__ import annotations

import shutil
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha._quadtree import read_2d_track_objects, write_2d_track_file

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT


EMPTY_CHAIN = {
    "chrom":     np.array([], dtype=object),
    "start":     np.array([], dtype=np.int64),
    "end":       np.array([], dtype=np.int64),
    "strand":    np.array([], dtype=np.int64),
    "chromsrc":  np.array([], dtype=object),
    "startsrc":  np.array([], dtype=np.int64),
    "endsrc":    np.array([], dtype=np.int64),
    "strandsrc": np.array([], dtype=np.int64),
    "chain_id":  np.array([], dtype=np.int64),
    "score":     np.array([], dtype=np.float64),
}


def _chain(rows):
    """Build a chain dict from a list of dict rows."""
    cols = ("chrom", "start", "end", "strand", "chromsrc", "startsrc", "endsrc",
            "strandsrc", "chain_id", "score")
    if not rows:
        return EMPTY_CHAIN
    return {
        "chrom":     np.array([r["chrom"]     for r in rows], dtype=object),
        "start":     np.array([r["start"]     for r in rows], dtype=np.int64),
        "end":       np.array([r["end"]       for r in rows], dtype=np.int64),
        "strand":    np.array([r["strand"]    for r in rows], dtype=np.int64),
        "chromsrc":  np.array([r["chromsrc"]  for r in rows], dtype=object),
        "startsrc":  np.array([r["startsrc"]  for r in rows], dtype=np.int64),
        "endsrc":    np.array([r["endsrc"]    for r in rows], dtype=np.int64),
        "strandsrc": np.array([r["strandsrc"] for r in rows], dtype=np.int64),
        "chain_id":  np.array([r["chain_id"]  for r in rows], dtype=np.int64),
        "score":     np.array([r["score"]     for r in rows], dtype=np.float64),
    }


def _identity_chain(src_chrom="chr1", tgt_chrom="1", length=1000, chain_id=1):
    """Identity chain: src [0, length) maps to tgt [0, length) on + strand."""
    return _chain([{
        "chrom": tgt_chrom, "start": 0, "end": length, "strand": 0,
        "chromsrc": src_chrom, "startsrc": 0, "endsrc": length, "strandsrc": 0,
        "chain_id": chain_id, "score": 1000.0,
    }])


def _result_to_sorted_rows(r):
    return sorted(zip(
        r["chrom1"], r["chrom2"],
        r["x1"], r["y1"], r["x2"], r["y2"],
        r["value"],
    ))


class TestAggregate2DRects:
    """_aggregate_2d_rects turns possibly-overlapping mapped rects into disjoint
    cells before insertion into the quadtree (R 5.11.8)."""

    def _agg(self, x1, y1, x2, y2, v, agg="mean", na_rm=True, min_n=None, nth=0):
        from pymisha.liftover import _aggregate_2d_rects
        return sorted(_aggregate_2d_rects(
            np.array(x1), np.array(y1), np.array(x2), np.array(y2),
            np.array(v, dtype=float), agg, na_rm, min_n, nth))

    def test_disjoint_rects_pass_through(self):
        cells = self._agg([0, 20], [0, 20], [10, 30], [10, 30], [1.0, 2.0])
        assert cells == [(0, 0, 10, 10, 1.0), (20, 20, 30, 30, 2.0)]

    def test_fully_overlapping_rects_aggregated(self):
        # Two identical rects -> one cell holding the aggregate (mean).
        cells = self._agg([0, 0], [0, 0], [10, 10], [10, 10], [4.0, 8.0])
        assert cells == [(0, 0, 10, 10, 6.0)]

    def test_fully_overlapping_rects_max(self):
        cells = self._agg([0, 0], [0, 0], [10, 10], [10, 10], [4.0, 8.0], agg="max")
        assert cells == [(0, 0, 10, 10, 8.0)]

    def test_partial_overlap_splits_into_disjoint_cells(self):
        # A: x[0,10] y[0,30]=2 ; B: x[0,10] y[10,20]=6 (overlap in y[10,20]).
        cells = self._agg([0, 0], [0, 10], [10, 10], [30, 20], [2.0, 6.0])
        assert cells == [
            (0, 0, 10, 10, 2.0),    # A only
            (0, 10, 10, 20, 4.0),   # mean(A, B)
            (0, 20, 10, 30, 2.0),   # A only
        ]

    def test_output_rects_are_disjoint(self):
        # A random-ish overlap must never yield two cells sharing interior area.
        cells = self._agg([0, 5], [0, 5], [10, 15], [10, 15], [1.0, 3.0])
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                ax1, ay1, ax2, ay2, _ = cells[i]
                bx1, by1, bx2, by2, _ = cells[j]
                overlap_x = min(ax2, bx2) - max(ax1, bx1)
                overlap_y = min(ay2, by2) - max(ay1, by1)
                assert overlap_x <= 0 or overlap_y <= 0


class TestPmLiftoverTrack2dSkeleton:
    def test_returns_empty_on_empty_source(self, tmp_path):
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), EMPTY_CHAIN)
        assert len(r["chrom1"]) == 0
        assert r["is_points"] is False

    def test_rejects_non_dict_chain(self, tmp_path):
        with pytest.raises(TypeError):
            _pymisha.pm_liftover_track_2d(str(tmp_path), "not a dict")

    def test_missing_src_dir_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            _pymisha.pm_liftover_track_2d(str(tmp_path / "nope"), EMPTY_CHAIN)


class TestPmLiftoverTrack2dRectsIdentity:
    def test_single_rect_identity_chain(self, tmp_path):
        # Source has one rect on (chr1, chr1).
        write_2d_track_file(
            str(tmp_path / "chr1-chr1"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        chain = _identity_chain("chr1", "1", 1000)
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        assert _result_to_sorted_rows(r) == [("1", "1", 10, 20, 30, 40, 1.5)]
        assert r["is_points"] is False

    def test_multi_rect_identity_chain(self, tmp_path):
        write_2d_track_file(
            str(tmp_path / "chr1-chr1"),
            [
                (10, 20, 30, 40, 1.5),
                (100, 200, 150, 250, 2.5),
            ],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        chain = _identity_chain("chr1", "1", 1000)
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        assert _result_to_sorted_rows(r) == sorted([
            ("1", "1", 10, 20, 30, 40, 1.5),
            ("1", "1", 100, 200, 150, 250, 2.5),
        ])

    def test_cross_chrom_pair(self, tmp_path):
        # Source rect on (chr1, chr2).
        write_2d_track_file(
            str(tmp_path / "chr1-chr2"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        chain = _chain([
            {"chrom": "1", "start": 0, "end": 1000, "strand": 0,
             "chromsrc": "chr1", "startsrc": 0, "endsrc": 1000, "strandsrc": 0,
             "chain_id": 1, "score": 1000.0},
            {"chrom": "2", "start": 0, "end": 1000, "strand": 0,
             "chromsrc": "chr2", "startsrc": 0, "endsrc": 1000, "strandsrc": 0,
             "chain_id": 2, "score": 1000.0},
        ])
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        assert _result_to_sorted_rows(r) == [("1", "2", 10, 20, 30, 40, 1.5)]


class TestPmLiftoverTrack2dDropsRectsThatDontMatchChain:
    def test_no_chain_for_chrom1_drops(self, tmp_path):
        write_2d_track_file(
            str(tmp_path / "chr1-chr1"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        # Chain has entries only on chr99 - matches nothing.
        chain = _identity_chain("chr99", "99", 1000)
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        assert len(r["chrom1"]) == 0

    def test_no_chain_for_chrom2_drops(self, tmp_path):
        write_2d_track_file(
            str(tmp_path / "chr1-chr2"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        # Chain matches chr1 (x-dim) but not chr2 (y-dim).
        chain = _identity_chain("chr1", "1", 1000)
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        assert len(r["chrom1"]) == 0


class TestPmLiftoverTrack2dMultiBlockSplit:
    def test_x_dim_chain_splits_rect_into_two(self, tmp_path):
        # X-dim chain has TWO blocks with a gap between target intervals:
        #   src [0, 50) -> tgt [0, 50)
        #   src [50, 100) -> tgt [100, 150)  (gap at tgt [50, 100))
        # Y-dim has one identity block on (chr1, 1).
        # Source rect (10, 20, 80, 40, 1.5) spans both x-blocks.
        # Expected: x-dim mapping yields two intervals -> two target rects.
        write_2d_track_file(
            str(tmp_path / "chr1-chr1"),
            [(10, 20, 80, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        chain = _chain([
            {"chrom": "1", "start": 0,   "end": 50,  "strand": 0,
             "chromsrc": "chr1", "startsrc": 0,  "endsrc": 50,  "strandsrc": 0,
             "chain_id": 1, "score": 1000.0},
            {"chrom": "1", "start": 100, "end": 150, "strand": 0,
             "chromsrc": "chr1", "startsrc": 50, "endsrc": 100, "strandsrc": 0,
             "chain_id": 2, "score": 1000.0},
        ])
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        rows = _result_to_sorted_rows(r)
        # Source x-interval [10, 80) intersects chain block 1 -> [10, 50)
        # (target [10, 50)), and chain block 2 -> [50, 80) (target [100, 130)).
        # Y-dim [20, 40) maps identity -> [20, 40).
        # Cross-product = 2 target rects.
        assert rows == sorted([
            ("1", "1", 10, 20, 50, 40, 1.5),
            ("1", "1", 100, 20, 130, 40, 1.5),
        ])


class TestPmLiftoverTrack2dPoints:
    def test_single_point_identity(self, tmp_path):
        write_2d_track_file(
            str(tmp_path / "chr1-chr1"),
            [(10, 20, 1.5)],
            (0, 0, 1000, 1000),
            is_points=True,
        )
        chain = _identity_chain("chr1", "1", 1000)
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        # Points are emitted as (x, x+1, y, y+1, v).
        assert _result_to_sorted_rows(r) == [("1", "1", 10, 20, 11, 21, 1.5)]
        assert r["is_points"] is True


class TestPmLiftoverTrack2dNegStrand:
    def test_neg_strand_on_x_dim_reflects_coords(self, tmp_path):
        # Source rect (10, 20, 30, 40, 1.5) on chr1-chr1.
        # X-dim chain block is negative strand: src [0, 100) on chr1 maps
        # to tgt [0, 100) on '1' but with reflection.
        # For src [10, 30) on neg-strand: target = (chain_end_tgt - (src_pos - chain_start_src))
        # which is mirror-image around the chain block.
        # Specifically, R's map_interval on a neg-strand chain block with
        # src [10, 30) within [0, 100) on neg-strand maps tgt to
        # [100 - 30, 100 - 10) = [70, 90).
        # Y-dim is identity.
        write_2d_track_file(
            str(tmp_path / "chr1-chr1"),
            [(10, 20, 30, 40, 1.5)],
            (0, 0, 1000, 1000),
            is_points=False,
        )
        chain = _chain([
            # X-dim: chr1 src -> '1' tgt, src strand=0, tgt strand=1 (neg).
            {"chrom": "1", "start": 0, "end": 100, "strand": 1,
             "chromsrc": "chr1", "startsrc": 0, "endsrc": 100, "strandsrc": 0,
             "chain_id": 1, "score": 1000.0},
        ])
        r = _pymisha.pm_liftover_track_2d(str(tmp_path), chain)
        rows = _result_to_sorted_rows(r)
        # Both x-dim and y-dim use the same chain (same name 'chr1').
        # Y-interval [20, 40) on chr1 with neg-strand chain -> [60, 80).
        # X-interval [10, 30) -> [70, 90).
        # So output: (1, 1, 70, 60, 90, 80, 1.5).
        assert rows == [("1", "1", 70, 60, 90, 80, 1.5)]


# ---------------------------------------------------------------------------
# End-to-end dispatcher tests via gtrack_liftover (Python public API)
# ---------------------------------------------------------------------------


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
            pair_file = out_dir / "1-1"
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
            pair_file = out_dir / "1-1"
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
