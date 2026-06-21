"""Cross-validation tests for pm_liftover_track (G1.P3.C)."""

import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

import _pymisha
import pymisha as pm

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


class TestPmLiftoverTrackSkeleton:
    """Argument-validation and skeleton-behavior tests for pm_liftover_track."""

    def test_returns_empty_dict_skeleton(self, tmp_path):
        # Skeleton in Task 1 ignores src_track_dir; supply tmp_path to keep
        # the call legal.
        result = _pymisha.pm_liftover_track(
            str(tmp_path), EMPTY_CHAIN, {}, "", "mean", True, -1, 0,
        )
        assert set(result) == {"chrom", "start", "end", "value", "track_type", "bin_size"}
        for k in ("chrom", "start", "end", "value"):
            assert len(result[k]) == 0
        assert result["track_type"] == "sparse"
        assert result["bin_size"] == 0

    def test_rejects_non_dict_chain(self, tmp_path):
        with pytest.raises(TypeError):
            _pymisha.pm_liftover_track(
                str(tmp_path), "not a dict", {}, "", "mean", True, -1, 0,
            )

    def test_rejects_non_dict_tgt_chrom_sizes(self, tmp_path):
        with pytest.raises(TypeError):
            _pymisha.pm_liftover_track(
                str(tmp_path), EMPTY_CHAIN, "not a dict", "", "mean", True, -1, 0,
            )


class TestReadSourceTrack1dHelperRoundtrip:
    """The helper is C++-only; verify pm_read_source_track_1d (thin wrapper)
    still works after the refactor."""

    def test_empty_directory(self, tmp_path):
        import _pymisha
        track_type, df = _pymisha.pm_read_source_track_1d(str(tmp_path))
        assert track_type in ("sparse", "")
        assert len(df["chrom"]) == 0

    def test_bin_size_mismatch_raises(self, tmp_path):
        """R-parity: dense per-chrom files with different bin_sizes must
        produce a clear error matching R's message format."""
        import struct
        # Write two dense per-chrom files with different bin_sizes.
        with open(tmp_path / "srcA", "wb") as f:
            f.write(struct.pack("I", 100))  # bin_size 100
            f.write(struct.pack("f", 1.0))
        with open(tmp_path / "srcB", "wb") as f:
            f.write(struct.pack("I", 200))  # bin_size 200 (mismatched)
            f.write(struct.pack("f", 2.0))
        with pytest.raises(ValueError, match="Binsize"):
            _pymisha.pm_read_source_track_1d(str(tmp_path))


class TestMapIntervalsHelperRoundtrip:
    """The helper is C++-only; verify pm_map_intervals (thin wrapper)
    still produces identical output after the refactor."""

    def test_pm_map_intervals_empty(self):
        empty_src = {
            "chrom": np.array([], dtype=object),
            "start": np.array([], dtype=np.int64),
            "end":   np.array([], dtype=np.int64),
        }
        result = _pymisha.pm_map_intervals(empty_src, EMPTY_CHAIN, "", 0, "")
        assert set(result) >= {"chrom", "start", "end", "intervalID", "chain_id"}
        assert all(len(v) == 0 for v in result.values())


class TestAggregateHelperRoundtrip:
    """Verify pm_liftover_aggregate (thin wrapper) still works after the refactor."""

    def test_pm_liftover_aggregate_empty(self):
        import _pymisha
        empty = {
            "chrom": np.array([], dtype=object),
            "start": np.array([], dtype=np.int64),
            "end": np.array([], dtype=np.int64),
            "value": np.array([], dtype=np.float64),
        }
        result = _pymisha.pm_liftover_aggregate(empty, "mean", True, -1, 0)
        assert set(result) == {"chrom", "start", "end", "value"}
        assert all(len(v) == 0 for v in result.values())

    def test_pm_liftover_aggregate_mean_overlap(self):
        import _pymisha
        df = {
            "chrom": np.array(["chr1", "chr1"], dtype=object),
            "start": np.array([100, 150], dtype=np.int64),
            "end":   np.array([200, 250], dtype=np.int64),
            "value": np.array([1.0, 3.0], dtype=np.float64),
        }
        result = _pymisha.pm_liftover_aggregate(df, "mean", True, -1, 0)
        # Sweep: [100,150) -> 1, [150,200) -> 2 (mean(1,3)), [200,250) -> 3
        assert list(result["chrom"]) == ["chr1", "chr1", "chr1"]
        assert list(result["start"]) == [100, 150, 200]
        assert list(result["end"])   == [150, 200, 250]
        assert list(result["value"]) == [1.0, 2.0, 3.0]


class TestPmLiftoverTrackEndToEnd:
    """End-to-end orchestration tests against the real C++ implementation."""

    @staticmethod
    def _build_chain_dict(chain_df):
        return {
            "chrom":     chain_df["chrom"].to_numpy(dtype=object),
            "start":     chain_df["start"].to_numpy(dtype=np.int64),
            "end":       chain_df["end"].to_numpy(dtype=np.int64),
            "strand":    chain_df["strand"].to_numpy(dtype=np.int64),
            "chromsrc":  chain_df["chromsrc"].to_numpy(dtype=object),
            "startsrc":  chain_df["startsrc"].to_numpy(dtype=np.int64),
            "endsrc":    chain_df["endsrc"].to_numpy(dtype=np.int64),
            "strandsrc": chain_df["strandsrc"].to_numpy(dtype=np.int64),
            "chain_id":  chain_df["chain_id"].to_numpy(dtype=np.int64),
            "score":     chain_df["score"].to_numpy(dtype=np.float64),
        }

    def test_dense_source_1to1_chain(self, tmp_path):
        from test_track_liftover import (
            _create_dense_track_dir, _write_chain, _copy_db,
        )
        import pymisha as pm
        src_dir = tmp_path / "src.track"
        _create_dense_track_dir(src_dir, 100, {"srcA": [10.0, 20.0, 30.0]})
        chain_path = _write_chain(tmp_path, [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 300,
              "tgt_chrom": "1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1300,
              "chain_id": 1}, [(300,)]),
        ])
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            chain_df = pm.gintervals_load_chain(chain_path, tgt_overlap_policy="keep")
            chain_dict = self._build_chain_dict(chain_df)
            tgt_chrom_sizes = {"1": 500000, "2": 300000, "X": 200000}
            result = _pymisha.pm_liftover_track(
                str(src_dir), chain_dict, tgt_chrom_sizes,
                "", "mean", True, -1, 0,
            )
        finally:
            pm.gdb_init(str(pm._GROOT) if hasattr(pm, "_GROOT") else str(TESTDB_ROOT))
        # FIXED_BIN preservation
        assert result["track_type"] == "dense"
        assert int(result["bin_size"]) == 100
        chrom_arr = np.asarray(result["chrom"], dtype=object)
        start_arr = np.asarray(result["start"], dtype=np.int64)
        value_arr = np.asarray(result["value"], dtype=np.float64)
        mask = (chrom_arr == "1") & np.isin(start_arr, [1000, 1100, 1200])
        ordered = np.argsort(start_arr[mask])
        non_nan = value_arr[mask][ordered]
        assert list(non_nan) == [10.0, 20.0, 30.0]

    def test_2d_source_track_raises(self, tmp_path):
        """2D source-track directory must raise cleanly."""
        d = tmp_path / "src2d.track"
        d.mkdir()
        (d / "chr1-chr1").write_bytes(b"\xff\xff\xff\xff")
        empty_chain = {
            "chrom": np.array([], dtype=object),
            "start": np.array([], dtype=np.int64),
            "end": np.array([], dtype=np.int64),
            "strand": np.array([], dtype=np.int64),
            "chromsrc": np.array([], dtype=object),
            "startsrc": np.array([], dtype=np.int64),
            "endsrc": np.array([], dtype=np.int64),
            "strandsrc": np.array([], dtype=np.int64),
            "chain_id": np.array([], dtype=np.int64),
            "score": np.array([], dtype=np.float64),
        }
        # 2D source files like "chr1-chr1" are not per-chrom-1D files; the
        # reader may or may not raise, but the orchestrator must not silently
        # produce wrong output. ARRAYS detection is also not implemented yet.
        # Acceptable behavior: either raise OR return empty with track_type
        # detection. We just verify no crash.
        try:
            result = _pymisha.pm_liftover_track(
                str(d), empty_chain, {}, "", "mean", True, -1, 0,
            )
            # If no raise, the bogus file should not have been parsed as
            # data -- output should be empty.
            assert len(result["chrom"]) == 0
        except (ValueError, RuntimeError):
            pass  # Expected error path.


# ===================================================================
# Cross-validation: C++ path vs Python fallback path.
# ===================================================================


def _write_chain(tmpdir, entries):
    """Write a chain file from entries list."""
    path = os.path.join(str(tmpdir), "xval_test.chain")
    with open(path, "w") as f:
        for hdr, blocks in entries:
            f.write(
                f"chain {hdr['score']} "
                f"{hdr['src_chrom']} {hdr['src_size']} {hdr['src_strand']} "
                f"{hdr['src_start']} {hdr['src_end']} "
                f"{hdr['tgt_chrom']} {hdr['tgt_size']} {hdr['tgt_strand']} "
                f"{hdr['tgt_start']} {hdr['tgt_end']} "
                f"{hdr['chain_id']}\n"
            )
            for blk in blocks:
                if len(blk) == 3:
                    f.write(f"{blk[0]}\t{blk[1]}\t{blk[2]}\n")
                else:
                    f.write(f"{blk[0]}\n")
            f.write("\n")
    return path


def _create_dense_track_dir(path, bin_size, values_per_chrom):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for chrom, vals in values_per_chrom.items():
        with open(path / chrom, "wb") as f:
            f.write(struct.pack("I", bin_size))
            np.array(vals, dtype=np.float32).tofile(f)


def _create_sparse_track_dir(path, intervals_per_chrom):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for chrom, intervals in intervals_per_chrom.items():
        with open(path / chrom, "wb") as f:
            f.write(struct.pack("i", -1))  # sparse signature
            for start, end, val in intervals:
                f.write(struct.pack("iif", start, end, np.float32(val)))


class TestPmLiftoverTrackXval:
    """Per fixture: run both paths and assert byte-equal output via gextract.

    The two paths share the dispatcher; this harness specifically asserts
    that the C++ orchestrator's output matches the Python R-parity reference.
    """

    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path):
        """Per-test DB copy so the dispatcher's _track_exists check doesn't trip."""
        dst = tmp_path / "trackdb" / "test"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(TEST_DB, dst)
        pm.gdb_init(str(dst))
        self._db_root = dst
        self._tmp = tmp_path
        yield
        pm.gdb_init(str(TEST_DB))

    def _run_xval(self, src_dir, chain_path, agg, tgt_overlap_policy="keep"):
        """Run both paths; return (cpp_df, py_df) extracted over target region."""
        # 1) C++ path (default).
        os.environ.pop("PYMISHA_FORCE_PY_LIFTOVER_TRACK", None)
        pm.gtrack_liftover(
            "xval_cpp", "xval c++", str(src_dir), chain_path,
            multi_target_agg=agg, tgt_overlap_policy=tgt_overlap_policy,
        )
        cpp_info = pm.gtrack_info("xval_cpp")

        # 2) Python path.
        os.environ["PYMISHA_FORCE_PY_LIFTOVER_TRACK"] = "1"
        try:
            pm.gtrack_liftover(
                "xval_py", "xval py", str(src_dir), chain_path,
                multi_target_agg=agg, tgt_overlap_policy=tgt_overlap_policy,
            )
        finally:
            os.environ.pop("PYMISHA_FORCE_PY_LIFTOVER_TRACK", None)
        py_info = pm.gtrack_info("xval_py")

        assert cpp_info["type"] == py_info["type"], (
            f"{agg}/{tgt_overlap_policy}: type {cpp_info['type']} vs {py_info['type']}"
        )

        # Extract over the target region (chrom 1, positions 1000-1600 covers
        # all fixtures in this harness).
        query = pm.gintervals("1", 1000, 1600)
        cpp_df = pm.gextract("xval_cpp", query)
        py_df = pm.gextract("xval_py", query)
        return cpp_df, py_df

    @pytest.mark.parametrize("agg", [
        "mean", "median", "sum", "min", "max", "count", "first", "last",
    ])
    def test_xval_dense_aggregations(self, agg):
        """Dense src + 1:1 chain - both paths must produce identical dense track."""
        src_dir = self._tmp / f"src_{agg}.track"
        _create_dense_track_dir(src_dir, 100, {
            "srcA": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        chain_path = _write_chain(self._tmp, [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 500,
              "tgt_chrom": "1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1500,
              "chain_id": 1}, [(500,)]),
        ])
        cpp_df, py_df = self._run_xval(src_dir, chain_path, agg)

        cpp_vals = cpp_df["xval_cpp"].to_numpy()
        py_vals = py_df["xval_py"].to_numpy()

        assert np.array_equal(np.isnan(cpp_vals), np.isnan(py_vals)), (
            f"{agg}: NaN positions differ\nC++: {cpp_vals}\nPy:  {py_vals}"
        )
        non_nan = ~np.isnan(cpp_vals)
        np.testing.assert_allclose(
            cpp_vals[non_nan], py_vals[non_nan], rtol=1e-5,
            err_msg=f"{agg}: values differ",
        )

    @pytest.mark.parametrize("policy", ["keep", "auto_score"])
    def test_xval_tgt_overlap_policies_sparse(self, policy):
        """Sparse src + simple chain across overlap policies."""
        src_dir = self._tmp / f"src_{policy}.track"
        _create_sparse_track_dir(src_dir, {
            "srcA": [(0, 100, 1.0), (200, 300, 2.0), (400, 500, 3.0)],
        })
        chain_path = _write_chain(self._tmp, [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 500,
              "tgt_chrom": "1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1500,
              "chain_id": 1}, [(500,)]),
        ])
        cpp_df, py_df = self._run_xval(src_dir, chain_path, "mean",
                                       tgt_overlap_policy=policy)

        cpp_vals = cpp_df["xval_cpp"].to_numpy()
        py_vals = py_df["xval_py"].to_numpy()

        assert np.array_equal(np.isnan(cpp_vals), np.isnan(py_vals)), (
            f"{policy}: NaN positions differ\nC++: {cpp_vals}\nPy:  {py_vals}"
        )
        non_nan = ~np.isnan(cpp_vals)
        np.testing.assert_allclose(
            cpp_vals[non_nan], py_vals[non_nan], rtol=1e-5,
        )

    def test_xval_chain_with_internal_gap(self):
        """Chain block with internal gap - exercises spanning-interval logic.

        Two src bins map to non-adjacent target intervals via a chain with an
        internal gap. The per-bin cursor-advance in both paths must handle the
        gap identically (the bin spanning the gap gets a value only from the
        overlapping block side).
        """
        src_dir = self._tmp / "src_gap.track"
        _create_dense_track_dir(src_dir, 100, {"srcA": [10.0, 20.0]})
        chain_path = _write_chain(self._tmp, [
            ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
              "src_strand": "+", "src_start": 0, "src_end": 200,
              "tgt_chrom": "1", "tgt_size": 500000,
              "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1300,
              "chain_id": 1}, [(100, 50, 50), (100,)]),
        ])
        cpp_df, py_df = self._run_xval(src_dir, chain_path, "mean")

        cpp_vals = cpp_df["xval_cpp"].to_numpy()
        py_vals = py_df["xval_py"].to_numpy()

        assert np.array_equal(np.isnan(cpp_vals), np.isnan(py_vals)), (
            "NaN positions differ - cursor advance divergence between C++ and Python\n"
            f"C++: {cpp_vals}\nPy:  {py_vals}"
        )
        non_nan = ~np.isnan(cpp_vals)
        np.testing.assert_allclose(
            cpp_vals[non_nan], py_vals[non_nan], rtol=1e-5,
        )
