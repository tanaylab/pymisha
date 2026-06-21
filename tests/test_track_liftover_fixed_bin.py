"""R-parity tests for FIXED_BIN preservation in gtrack_liftover.

A FIXED_BIN (dense) source track must produce a FIXED_BIN (dense) target track
after liftover. This mirrors R misha's GTrackLiftover.cpp behavior.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _write_dense_track(path: Path, bin_size: int, values_per_chrom: dict) -> None:
    """Write a per-chrom dense track directory."""
    path.mkdir(parents=True, exist_ok=True)
    for chrom, vals in values_per_chrom.items():
        with open(path / chrom, "wb") as f:
            f.write(struct.pack("I", bin_size))
            np.array(vals, dtype=np.float32).tofile(f)


def _write_sparse_track(path: Path, intervals_per_chrom: dict) -> None:
    """Write a per-chrom sparse track directory.

    intervals_per_chrom: {chrom: [(start, end, value), ...]}
    """
    path.mkdir(parents=True, exist_ok=True)
    for chrom, ivs in intervals_per_chrom.items():
        with open(path / chrom, "wb") as f:
            f.write(struct.pack("<i", -1))  # sparse magic
            for start, end, val in ivs:
                f.write(struct.pack("<ii", start, end))
                f.write(struct.pack("<f", val))


def _write_chain(tmp: Path, entries: list) -> Path:
    """Write a UCSC chain file. Returns its path."""
    p = tmp / "test.chain"
    with open(p, "w") as f:
        for hdr, blocks in entries:
            f.write(
                f"chain {hdr['score']} "
                f"{hdr['src_chrom']} {hdr['src_size']} {hdr['src_strand']} "
                f"{hdr['src_start']} {hdr['src_end']} "
                f"{hdr['tgt_chrom']} {hdr['tgt_size']} {hdr['tgt_strand']} "
                f"{hdr['tgt_start']} {hdr['tgt_end']} {hdr['chain_id']}\n"
            )
            for blk in blocks:
                if len(blk) == 3:
                    f.write(f"{blk[0]}\t{blk[1]}\t{blk[2]}\n")
                else:
                    f.write(f"{blk[0]}\n")
            f.write("\n")
    return str(p)


# ===================================================================
# Core: dense -> dense preservation
# ===================================================================

class TestFixedBinPreservation:
    """A FIXED_BIN source must produce a FIXED_BIN target track (R-parity)."""

    def test_dense_source_produces_dense_target(self, tmp_path):
        """Simple liftover of a dense source yields a dense target track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_dense.track"
            # 3 bins of 100bp each: vals [10, 20, 30] covering srcA:[0,300)
            _write_dense_track(src_dir, 100, {"srcA": [10.0, 20.0, 30.0]})
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 300,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1300,
                  "chain_id": 1}, [(300,)]),
            ])
            pm.gtrack_liftover("fb_dense_target", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               _force_pure_python=True)
            info = pm.gtrack_info("fb_dense_target")
            assert info["type"] == "dense", (
                f"Expected 'dense', got '{info['type']}' — FIXED_BIN source must produce dense target"
            )
            assert info["bin_size"] == 100
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_sparse_source_still_produces_sparse(self, tmp_path):
        """Sparse source must keep producing sparse target (regression check)."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_sparse.track"
            _write_sparse_track(src_dir, {"srcA": [(0, 100, 5.0), (200, 300, 7.0)]})
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 500,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 2500,
                  "chain_id": 1}, [(500,)]),
            ])
            pm.gtrack_liftover("fb_sparse_target", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               _force_pure_python=True)
            info = pm.gtrack_info("fb_sparse_target")
            assert info["type"] == "sparse", (
                f"Expected 'sparse', got '{info['type']}' — sparse source must remain sparse"
            )
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_dense_bin_size_preserved(self, tmp_path):
        """Bin size of 50 bp is preserved in the target track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_bs50.track"
            _write_dense_track(src_dir, 50, {"srcA": [1.0, 2.0, 3.0, 4.0]})
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 200,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 200,
                  "chain_id": 1}, [(200,)]),
            ])
            pm.gtrack_liftover("fb_bs50", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               _force_pure_python=True)
            info = pm.gtrack_info("fb_bs50")
            assert info["type"] == "dense"
            assert info["bin_size"] == 50
        finally:
            pm.gdb_init(str(TEST_DB))


# ===================================================================
# Per-bin aggregation numerical correctness
# ===================================================================

class TestPerBinAggregationValues:
    """Verify _aggregate_per_bin_python produces R-correct numerical output."""

    def test_1to1_values_preserved(self, tmp_path):
        """1:1 liftover preserves bin values exactly."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_1to1.track"
            _write_dense_track(src_dir, 100, {"srcA": [10.0, 20.0, 30.0]})
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 300,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 300,
                  "chain_id": 1}, [(300,)]),
            ])
            pm.gtrack_liftover("fb_1to1", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               _force_pure_python=True)
            # Query at bin-exact intervals to read dense track values.
            out0 = pm.gextract("fb_1to1", pm.gintervals("chr1", 0, 100))
            out1 = pm.gextract("fb_1to1", pm.gintervals("chr1", 100, 200))
            out2 = pm.gextract("fb_1to1", pm.gintervals("chr1", 200, 300))
            v0 = out0["fb_1to1"].dropna().tolist()
            v1 = out1["fb_1to1"].dropna().tolist()
            v2 = out2["fb_1to1"].dropna().tolist()
            assert v0, "Bin 0 should have a value"
            assert v1, "Bin 1 should have a value"
            assert v2, "Bin 2 should have a value"
            np.testing.assert_allclose(v0[0], 10.0, rtol=1e-5)
            np.testing.assert_allclose(v1[0], 20.0, rtol=1e-5)
            np.testing.assert_allclose(v2[0], 30.0, rtol=1e-5)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_mean_aggregation_two_chains(self, tmp_path):
        """Two chains projecting different values into same bin -> mean.

        Bin [0, 100) on tgt chr1 receives:
          - chain 1: val=10 with full overlap (100bp)
          - chain 2: val=20 with full overlap (100bp)
        Expected mean (after chain_id merging): mean(10, 20) = 15.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_mean2.track"
            # srcA bin 0 = val 10, srcB bin 0 = val 20
            _write_dense_track(src_dir, 100, {
                "srcA": [10.0],
                "srcB": [20.0],
            })
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
                  "chain_id": 1}, [(100,)]),
                ({"score": 1000, "src_chrom": "srcB", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
                  "chain_id": 2}, [(100,)]),
            ])
            pm.gtrack_liftover("fb_mean2", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               multi_target_agg="mean",
                               _force_pure_python=True)
            out = pm.gextract("fb_mean2", pm.gintervals("chr1", 0, 100))
            v = out["fb_mean2"].dropna().tolist()
            assert v, "Bin [0,100) must have a value"
            np.testing.assert_allclose(v[0], 15.0, rtol=1e-5,
                                       err_msg="mean of 10 and 20 must be 15")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_unmapped_bins_are_nan(self, tmp_path):
        """Bins not covered by the chain are NaN in the target dense track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_gap_bins.track"
            # 4 bins of 50bp: [0,50), [50,100), [100,150), [150,200)
            # Chain only maps [0,50) -> [0,50) and [150,200) -> [200,250)
            _write_dense_track(src_dir, 50, {"srcA": [1.0, 2.0, 3.0, 4.0]})
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 50,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 50,
                  "chain_id": 1}, [(50,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 150, "src_end": 200,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 200, "tgt_end": 250,
                  "chain_id": 2}, [(50,)]),
            ])
            pm.gtrack_liftover("fb_gap_bins", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               _force_pure_python=True)
            info = pm.gtrack_info("fb_gap_bins")
            assert info["type"] == "dense"
            # Bin [0,50) -> val 1.0 (mapped)
            out0 = pm.gextract("fb_gap_bins", pm.gintervals("chr1", 0, 50))
            assert out0["fb_gap_bins"].dropna().tolist() == pytest.approx([1.0], rel=1e-5)
            # Bin [200,250) -> val 4.0 (mapped)
            out200 = pm.gextract("fb_gap_bins", pm.gintervals("chr1", 200, 250))
            assert out200["fb_gap_bins"].dropna().tolist() == pytest.approx([4.0], rel=1e-5)
            # Bins [50,200) are unmapped -> all NaN in target
            out_gap = pm.gextract("fb_gap_bins", pm.gintervals("chr1", 50, 200))
            vals_gap = out_gap["fb_gap_bins"].dropna().tolist()
            assert len(vals_gap) == 0, f"Unmapped bins should be NaN, got {vals_gap}"
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_nan_source_bins_skipped(self, tmp_path):
        """Source bins with NaN/inf are treated as NaN and skipped in output."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            src_dir = tmp_path / "src_nan_bins.track"
            # bin 0 = 1.0, bin 1 = inf (stored as inf, read back as NaN), bin 2 = 3.0
            _write_dense_track(src_dir, 100, {"srcA": [1.0, float("inf"), 3.0]})
            chain = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 300,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 300,
                  "chain_id": 1}, [(300,)]),
            ])
            pm.gtrack_liftover("fb_nan_bins", "", str(src_dir), chain,
                               tgt_overlap_policy="keep",
                               _force_pure_python=True)
            info = pm.gtrack_info("fb_nan_bins")
            assert info["type"] == "dense"
            # Bins 0 and 2 should be present
            out0 = pm.gextract("fb_nan_bins", pm.gintervals("chr1", 0, 100))
            assert out0["fb_nan_bins"].dropna().tolist() == pytest.approx([1.0], rel=1e-5)
            out2 = pm.gextract("fb_nan_bins", pm.gintervals("chr1", 200, 300))
            assert out2["fb_nan_bins"].dropna().tolist() == pytest.approx([3.0], rel=1e-5)
            # Bin 1 (inf -> NaN) should be absent
            out1 = pm.gextract("fb_nan_bins", pm.gintervals("chr1", 100, 200))
            assert len(out1["fb_nan_bins"].dropna()) == 0


        finally:
            pm.gdb_init(str(TEST_DB))


# ===================================================================
# Helper-level unit tests
# ===================================================================

class TestDetectSourceBinSize:
    """Unit tests for _detect_source_bin_size."""

    def test_detects_correct_bin_size(self, tmp_path):
        """Reads bin_size from a single-chrom dense track dir."""
        from pymisha.liftover import _detect_source_bin_size

        src_dir = tmp_path / "bs_detect.track"
        _write_dense_track(src_dir, 200, {"chr1": [1.0, 2.0]})
        assert _detect_source_bin_size(str(src_dir)) == 200

    def test_returns_zero_for_sparse(self, tmp_path):
        """Returns 0 for a sparse track directory."""
        from pymisha.liftover import _detect_source_bin_size

        src_dir = tmp_path / "sparse_detect.track"
        _write_sparse_track(src_dir, {"chr1": [(0, 100, 5.0)]})
        assert _detect_source_bin_size(str(src_dir)) == 0

    def test_raises_on_mismatched_bin_sizes(self, tmp_path):
        """Raises ValueError if two chrom files have different bin_sizes."""
        from pymisha.liftover import _detect_source_bin_size

        src_dir = tmp_path / "mismatch.track"
        src_dir.mkdir()
        # Write chr1 with bin_size=100
        with open(src_dir / "chr1", "wb") as f:
            f.write(struct.pack("I", 100))
            np.array([1.0], dtype=np.float32).tofile(f)
        # Write chr2 with bin_size=200
        with open(src_dir / "chr2", "wb") as f:
            f.write(struct.pack("I", 200))
            np.array([2.0], dtype=np.float32).tofile(f)
        with pytest.raises(ValueError, match="[Bb]insize"):
            _detect_source_bin_size(str(src_dir))


class TestAggregatePerBinPython:
    """Unit tests for _aggregate_per_bin_python."""

    def _make_lifted_df(self, rows):
        """Build a lifted DataFrame with chrom, start, end, value, chain_id."""
        import pandas as pd
        return pd.DataFrame(rows, columns=["chrom", "start", "end", "value", "chain_id"])

    def test_empty_input_produces_all_nan_bins(self):
        """No contributions -> all bins have NaN value."""
        from pymisha.liftover import _aggregate_per_bin_python

        df = self._make_lifted_df([])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 300},
            agg_name="mean",
        )
        assert len(result) == 3  # 3 bins of 100bp in 300bp chrom
        assert result["value"].isna().all()

    def test_single_contribution_mean(self):
        """Single contribution in a bin -> that value."""
        from pymisha.liftover import _aggregate_per_bin_python

        df = self._make_lifted_df([
            ("chr1", 0, 100, 5.0, 1),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 200},
            agg_name="mean",
        )
        # Bin [0,100): value 5.0; bin [100,200): NaN
        assert len(result) == 2
        r = result.sort_values("start").reset_index(drop=True)
        np.testing.assert_allclose(r.iloc[0]["value"], 5.0, rtol=1e-9)
        assert np.isnan(r.iloc[1]["value"])

    def test_two_chain_contributions_mean(self):
        """Two contributions with different chain_ids -> mean."""
        from pymisha.liftover import _aggregate_per_bin_python

        df = self._make_lifted_df([
            ("chr1", 0, 100, 10.0, 1),
            ("chr1", 0, 100, 20.0, 2),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 100},
            agg_name="mean",
        )
        assert len(result) == 1
        np.testing.assert_allclose(result.iloc[0]["value"], 15.0, rtol=1e-9)

    def test_same_chain_id_merged_before_aggregation(self):
        """Same chain_id contributions within a bin are merged first."""
        from pymisha.liftover import _aggregate_per_bin_python

        # chain_id=1 appears twice within [0,100): parts [0,50) and [50,100)
        # After merging: one entry chain_id=1 val=10 (value from first; overlap summed)
        # chain_id=2 val=20
        # mean(10, 20) = 15
        df = self._make_lifted_df([
            ("chr1", 0, 50, 10.0, 1),
            ("chr1", 50, 100, 10.0, 1),
            ("chr1", 0, 100, 20.0, 2),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 100},
            agg_name="mean",
        )
        assert len(result) == 1
        np.testing.assert_allclose(result.iloc[0]["value"], 15.0, rtol=1e-9)

    def test_sum_aggregation(self):
        """Sum aggregation returns sum of contributions."""
        from pymisha.liftover import _aggregate_per_bin_python

        df = self._make_lifted_df([
            ("chr1", 0, 100, 3.0, 1),
            ("chr1", 0, 100, 7.0, 2),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 100},
            agg_name="sum",
        )
        np.testing.assert_allclose(result.iloc[0]["value"], 10.0, rtol=1e-9)

    def test_max_aggregation(self):
        """Max aggregation returns max of contributions."""
        from pymisha.liftover import _aggregate_per_bin_python

        df = self._make_lifted_df([
            ("chr1", 0, 100, 3.0, 1),
            ("chr1", 0, 100, 7.0, 2),
            ("chr1", 0, 100, 5.0, 3),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 100},
            agg_name="max",
        )
        np.testing.assert_allclose(result.iloc[0]["value"], 7.0, rtol=1e-9)

    def test_na_rm_false_propagates_nan(self):
        """na_rm=False: a NaN contribution makes the whole bin NaN."""
        from pymisha.liftover import _aggregate_per_bin_python

        df = self._make_lifted_df([
            ("chr1", 0, 100, float("nan"), 1),
            ("chr1", 0, 100, 5.0, 2),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 100},
            agg_name="mean",
            na_rm=False,
        )
        assert np.isnan(result.iloc[0]["value"])

    def test_partial_overlap_across_bin_boundary(self):
        """A spanning interval that's the LAST in the list contributes to its
        FIRST overlapping bin only (R behavior - the --iter step-back logic in
        GTrackLiftover.cpp:719-751 fires only when a subsequent interval exists)."""
        from pymisha.liftover import _aggregate_per_bin_python

        # Interval [50, 150) overlaps bin [0,100) by 50bp and bin [100,200) by 50bp.
        df = self._make_lifted_df([
            ("chr1", 50, 150, 8.0, 1),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 200},
            agg_name="mean",
        )
        r = result.sort_values("start").reset_index(drop=True)
        # Bin [0, 100) gets val=8 (the interval's first overlap).
        np.testing.assert_allclose(r.iloc[0]["value"], 8.0, rtol=1e-9)
        # Bin [100, 200) gets NaN - interval was consumed in bin 0; no subsequent
        # interval triggers the --iter step-back. Matches R GTrackLiftover.cpp.
        assert np.isnan(r.iloc[1]["value"])

    def test_step_back_when_subsequent_interval_triggers(self):
        """A spanning interval contributes to multiple bins WHEN a subsequent
        non-overlapping interval triggers R's --iter step-back. This is the
        only path where a spanning interval can repeat across bins."""
        from pymisha.liftover import _aggregate_per_bin_python

        # Interval A: [50, 150) val=8 (spans bins 0 and 1).
        # Interval B: [180, 220) val=9 (lies in bin 1).
        df = self._make_lifted_df([
            ("chr1", 50, 150, 8.0, 1),
            ("chr1", 180, 220, 9.0, 2),
        ])
        result = _aggregate_per_bin_python(
            df, bin_size=100,
            tgt_chrom_sizes={"chr1": 300},
            agg_name="mean",
        )
        r = result.sort_values("start").reset_index(drop=True)
        # Bin [0, 100): only A contributes. mean = 8.
        np.testing.assert_allclose(r.iloc[0]["value"], 8.0, rtol=1e-9)
        # Bin [100, 200): A (re-encountered via --iter) and B both contribute.
        # mean = (8+9)/2 = 8.5.
        np.testing.assert_allclose(r.iloc[1]["value"], 8.5, rtol=1e-9)
        # Bin [200, 300): empty. NaN.
        assert np.isnan(r.iloc[2]["value"])
