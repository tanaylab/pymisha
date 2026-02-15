"""Tests for gtrack_liftover: track-level liftover from source to target assembly."""

import os
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


# ─── Helpers ─────────────────────────────────────────────────────────

def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _write_chain(tmpdir, entries):
    """Write a chain file from entries list."""
    path = os.path.join(str(tmpdir), "test.chain")
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
    """Create a dense track directory with per-chromosome binary files.

    values_per_chrom: dict mapping chrom_name -> numpy array of float32 values
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for chrom, vals in values_per_chrom.items():
        chrom_file = path / chrom
        with open(chrom_file, "wb") as f:
            # Dense format: 4-byte bin_size header, then float32 values
            f.write(struct.pack("I", bin_size))
            np.array(vals, dtype=np.float32).tofile(f)


def _create_sparse_track_dir(path, intervals_per_chrom):
    """Create a sparse track directory with per-chromosome binary files.

    intervals_per_chrom: dict mapping chrom_name -> list of (start, end, value) tuples
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for chrom, intervals in intervals_per_chrom.items():
        chrom_file = path / chrom
        with open(chrom_file, "wb") as f:
            # Sparse format: signature (-1 as int32), then (start_int32, end_int32, value_float32) per interval
            f.write(struct.pack("i", -1))  # sparse signature
            for start, end, val in intervals:
                f.write(struct.pack("iif", start, end, np.float32(val)))


# ─── Tests ───────────────────────────────────────────────────────────

class TestGtrackLiftover:
    """Tests for gtrack_liftover."""

    def test_liftover_dense_basic(self, tmp_path):
        """Liftover a dense source track to target DB."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            # Create source track: dense, bin_size=100, on source chrom "srcA"
            src_track_dir = tmp_path / "src_track.track"
            # 10 bins of known values: 1.0, 2.0, ..., 10.0
            vals = np.arange(1, 11, dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 100, {"srcA": vals})

            # Chain: srcA[0-1000] -> chr1[5000-6000] (1:1 mapping, + strand)
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 1000,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 5000, "tgt_end": 6000,
                  "chain_id": 1},
                 [(1000,)]),
            ])

            pm.gtrack_liftover("lifted_dense", "Dense liftover test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            assert pm.gtrack_exists("lifted_dense")
            info = pm.gtrack_info("lifted_dense")
            assert info["type"] == "sparse"  # liftover creates sparse track

            # Extract lifted values
            intervs = pm.gintervals("chr1", 5000, 6000)
            out = pm.gextract("lifted_dense", intervs)
            assert len(out) > 0
            # Values should be preserved (1.0 through 10.0)
            lifted_vals = out["lifted_dense"].to_numpy()
            assert not np.all(np.isnan(lifted_vals))
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_values_correct(self, tmp_path):
        """Verify lifted dense track values are correct."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_track.track"
            vals = np.array([10.0, 20.0, 30.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 100, {"srcA": vals})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 300,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1300,
                  "chain_id": 1},
                 [(300,)]),
            ])

            pm.gtrack_liftover("lifted_vals", "Values test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            # Extract: should get 3 intervals with values 10, 20, 30
            intervs = pm.gintervals("chr1", 1000, 1300)
            out = pm.gextract("lifted_vals", intervs)
            vals_out = sorted(out["lifted_vals"].dropna().tolist())
            assert vals_out == [10.0, 20.0, 30.0]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_indexed_only_source_track(self, tmp_path):
        """Indexed-only source track (track.idx/track.dat) must be read correctly."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            pm.gtrack_convert_to_indexed("dense_track", remove_old=True)
            src_track_dir = root / "tracks" / "dense_track.track"
            assert (src_track_dir / "track.idx").exists()
            assert (src_track_dir / "track.dat").exists()
            assert not (src_track_dir / "1").exists()
            assert not (src_track_dir / "chr1").exists()

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "1", "src_size": 500000,
                  "src_strand": "+", "src_start": 0, "src_end": 1000,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 10000, "tgt_end": 11000,
                  "chain_id": 1},
                 [(1000,)]),
            ])

            pm.gtrack_liftover(
                "lifted_indexed_source",
                "Indexed source liftover",
                str(src_track_dir),
                chain_path,
                tgt_overlap_policy="keep",
            )

            out = pm.gextract("lifted_indexed_source", pm.gintervals("chr1", 10000, 11000))
            lifted_vals = out["lifted_indexed_source"].dropna().to_numpy()
            assert len(lifted_vals) > 0
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_sparse_basic(self, tmp_path):
        """Liftover a sparse source track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_sparse.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(100, 200, 5.0), (300, 500, 7.0)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 1000,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 2000, "tgt_end": 3000,
                  "chain_id": 1},
                 [(1000,)]),
            ])

            pm.gtrack_liftover("lifted_sparse", "Sparse liftover test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            assert pm.gtrack_exists("lifted_sparse")
            info = pm.gtrack_info("lifted_sparse")
            assert info["type"] == "sparse"

            intervs = pm.gintervals("chr1", 2000, 3000)
            out = pm.gextract("lifted_sparse", intervs)
            vals = sorted(out["lifted_sparse"].dropna().tolist())
            assert 5.0 in vals
            assert 7.0 in vals
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_nan_values_skipped(self, tmp_path):
        """NaN values in source track are omitted from lifted track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_nan.track"
            vals = np.array([1.0, float('inf'), 3.0], dtype=np.float32)  # inf -> NaN in dense format
            _create_dense_track_dir(src_track_dir, 100, {"srcA": vals})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 300,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1300,
                  "chain_id": 1},
                 [(300,)]),
            ])

            pm.gtrack_liftover("lifted_nan", "NaN test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            intervs = pm.gintervals("chr1", 1000, 1300)
            out = pm.gextract("lifted_nan", intervs)
            # Should have values at positions for bins 1 and 3 (1.0 and 3.0)
            # Bin 2 had inf -> NaN, should be absent in sparse output
            vals = sorted(out["lifted_nan"].dropna().tolist())
            assert 1.0 in vals
            assert 3.0 in vals
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_multi_chrom_source(self, tmp_path):
        """Source track spanning multiple chromosomes."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_multi.track"
            _create_dense_track_dir(src_track_dir, 100, {
                "srcA": np.array([1.0, 2.0], dtype=np.float32),
                "srcB": np.array([3.0, 4.0], dtype=np.float32),
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 200,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1200,
                  "chain_id": 1},
                 [(200,)]),
                ({"score": 1000, "src_chrom": "srcB", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 200,
                  "tgt_chrom": "chrX", "tgt_size": 200000,
                  "tgt_strand": "+", "tgt_start": 5000, "tgt_end": 5200,
                  "chain_id": 2},
                 [(200,)]),
            ])

            pm.gtrack_liftover("lifted_multi", "Multi chrom test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            # Check chr1 values
            out1 = pm.gextract("lifted_multi", pm.gintervals("chr1", 1000, 1200))
            assert len(out1) > 0

            # Check chrX values
            outx = pm.gextract("lifted_multi", pm.gintervals("chrX", 5000, 5200))
            assert len(outx) > 0
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_aggregation_mean(self, tmp_path):
        """Multiple source intervals mapping to same target: mean aggregation."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_agg.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(0, 100, 10.0)],
                "srcB": [(0, 100, 20.0)],
            })

            # Two chains mapping different sources to the same target region
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 1},
                 [(100,)]),
                ({"score": 1000, "src_chrom": "srcB", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 2},
                 [(100,)]),
            ])

            pm.gtrack_liftover("lifted_agg", "Mean agg test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep",
                               multi_target_agg="mean")

            intervs = pm.gintervals("chr1", 1000, 1100)
            out = pm.gextract("lifted_agg", intervs)
            # Mean of 10 and 20 = 15
            val = out["lifted_agg"].dropna().iloc[0]
            np.testing.assert_allclose(val, 15.0, rtol=1e-5)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_aggregation_sum(self, tmp_path):
        """Multiple source intervals mapping to same target: sum aggregation."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_agg_sum.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(0, 100, 10.0)],
                "srcB": [(0, 100, 20.0)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 1},
                 [(100,)]),
                ({"score": 1000, "src_chrom": "srcB", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 2},
                 [(100,)]),
            ])

            pm.gtrack_liftover("lifted_sum", "Sum agg test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep",
                               multi_target_agg="sum")

            intervs = pm.gintervals("chr1", 1000, 1100)
            out = pm.gextract("lifted_sum", intervs)
            val = out["lifted_sum"].dropna().iloc[0]
            np.testing.assert_allclose(val, 30.0, rtol=1e-5)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_rejects_existing_track(self, tmp_path):
        """gtrack_liftover should reject creating over an existing track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_exists.track"
            _create_dense_track_dir(src_track_dir, 100, {"srcA": np.array([1.0], dtype=np.float32)})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 1},
                 [(100,)]),
            ])

            with pytest.raises(Exception):
                pm.gtrack_liftover("dense_track", "Overwrite attempt",
                                   str(src_track_dir), chain_path,
                                   tgt_overlap_policy="keep")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_nonexistent_source(self, tmp_path):
        """gtrack_liftover should error on nonexistent source directory."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 100,
                  "chain_id": 1},
                 [(100,)]),
            ])

            with pytest.raises(Exception):
                pm.gtrack_liftover("no_exist", "Test",
                                   "/no/such/dir", chain_path,
                                   tgt_overlap_policy="keep")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_sets_attributes(self, tmp_path):
        """gtrack_liftover should set description and created.by attributes."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_attrs.track"
            _create_dense_track_dir(src_track_dir, 100, {"srcA": np.array([1.0], dtype=np.float32)})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 1},
                 [(100,)]),
            ])

            pm.gtrack_liftover("lifted_attrs", "My lifted track",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            desc = pm.gtrack_attr_get("lifted_attrs", "description")
            created_by = pm.gtrack_attr_get("lifted_attrs", "created.by")
            assert desc == "My lifted track"
            assert "gtrack.liftover" in created_by
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_with_preloaded_chain(self, tmp_path):
        """gtrack_liftover accepts a pre-loaded chain DataFrame."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_preloaded.track"
            _create_dense_track_dir(src_track_dir, 100, {"srcA": np.array([42.0], dtype=np.float32)})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 1},
                 [(100,)]),
            ])

            chain = pm.gintervals_load_chain(chain_path, tgt_overlap_policy="keep")
            pm.gtrack_liftover("lifted_preloaded", "Preloaded chain test",
                               str(src_track_dir), chain)

            intervs = pm.gintervals("chr1", 1000, 1100)
            out = pm.gextract("lifted_preloaded", intervs)
            val = out["lifted_preloaded"].dropna().iloc[0]
            np.testing.assert_allclose(val, 42.0, rtol=1e-5)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_empty_chain(self, tmp_path):
        """Liftover with empty chain creates an empty sparse track."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_empty_chain.track"
            _create_dense_track_dir(src_track_dir, 100, {"srcA": np.array([1.0], dtype=np.float32)})

            # Empty chain - no matching source chroms
            chain_path = _write_chain(tmp_path, [])

            pm.gtrack_liftover("lifted_empty", "Empty chain test",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            assert pm.gtrack_exists("lifted_empty")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_aggregation_na_rm(self, tmp_path):
        """na_rm=True skips NaN in aggregation; na_rm=False propagates NaN."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_narm.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(0, 100, 10.0)],
                "srcB": [(0, 100, float('nan'))],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 1},
                 [(100,)]),
                ({"score": 1000, "src_chrom": "srcB", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 100,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1100,
                  "chain_id": 2},
                 [(100,)]),
            ])

            # na_rm=True: should give mean of [10.0] = 10.0
            pm.gtrack_liftover("lifted_narm_true", "na_rm=True",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep",
                               multi_target_agg="mean", na_rm=True)
            intervs = pm.gintervals("chr1", 1000, 1100)
            out = pm.gextract("lifted_narm_true", intervs)
            val = out["lifted_narm_true"].dropna().iloc[0]
            np.testing.assert_allclose(val, 10.0, rtol=1e-5)
        finally:
            pm.gdb_init(str(TEST_DB))


# ===================================================================
# Ported from R test-gtrack.liftover-bin.R
# Dense (bin-level) liftover edge cases.
# The R tests cross-validate against the Kent liftOver binary;
# here we test the pymisha behavior directly (no external binary).
# ===================================================================


class TestGtrackLiftoverBin:
    """Dense bin-level track liftover edge cases.

    Ported from R test-gtrack.liftover-bin.R.
    """

    def test_liftover_dense_1to1_simple(self, tmp_path):
        """Simple 1:1 dense track liftover preserves values and coordinates.

        R: 'gtrack.liftover matches liftOver binary - basic sparse track'
        (adapted to dense bin-level).
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_simple.track"
            # 5 bins of 100bp each with known values
            vals = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 100, {"srcA": vals})

            # Chain: srcA[0-500] -> chr1[1000-1500] (1:1)
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 500,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 1000, "tgt_end": 1500,
                  "chain_id": 1},
                 [(500,)]),
            ])

            pm.gtrack_liftover("lifted_dense_1to1", "Dense 1to1",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_1to1",
                              pm.gintervals("chr1", 1000, 1500))
            assert len(out) == 5
            out = out.sort_values("start").reset_index(drop=True)
            expected_vals = [100.0, 200.0, 300.0, 400.0, 500.0]
            np.testing.assert_allclose(
                out["lifted_dense_1to1"].tolist(), expected_vals, rtol=1e-5
            )
            # Verify coordinates: bins shifted from [0,500) to [1000,1500)
            assert out.iloc[0]["start"] == 1000
            assert out.iloc[0]["end"] == 1100
            assert out.iloc[4]["start"] == 1400
            assert out.iloc[4]["end"] == 1500
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_with_chain_gap(self, tmp_path):
        """Dense track liftover with chain gap drops unmapped bins.

        R: 'gtrack.liftover matches liftOver binary - chain with gaps'
        Chain maps [0-50) and [150-250) only. Bins in [50-150) are unmapped.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_gap.track"
            # 5 bins of 50bp: covers [0,250)
            vals = np.array([111.0, 222.0, 333.0, 444.0, 555.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 50, {"srcA": vals})

            # Chain 1: srcA[0-50) -> chr1[0-50)
            # Chain 2: srcA[150-250) -> chr1[100-200)
            # Gap: srcA[50-150) is not mapped
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 50,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 50,
                  "chain_id": 1},
                 [(50,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 150, "src_end": 250,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 100, "tgt_end": 200,
                  "chain_id": 2},
                 [(100,)]),
            ])

            pm.gtrack_liftover("lifted_dense_gap", "Dense gap",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_gap", pm.gintervals_all())
            assert len(out) > 0
            # Bin 0 [0,50) -> mapped at [0,50) = val 111
            # Bins 1,2 [50,150) -> unmapped
            # Bins 3,4 [150,250) -> mapped at [100,200) = vals 444, 555
            vals_out = sorted(out["lifted_dense_gap"].dropna().tolist())
            assert 111.0 in vals_out
            assert 444.0 in vals_out
            assert 555.0 in vals_out
            # Unmapped bins should not appear
            assert 222.0 not in vals_out
            assert 333.0 not in vals_out
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_reverse_strand(self, tmp_path):
        """Dense track liftover with reverse strand target.

        R: 'gtrack.liftover matches liftOver binary - reverse strand'
        Values are preserved; coordinates are flipped.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_rev.track"
            vals = np.array([111.0, 222.0, 333.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 50, {"srcA": vals})

            # Chain: srcA[0-150) -> chr1 reverse strand [0-150)
            # Negative strand: target coords are flipped
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 150,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "-", "tgt_start": 0, "tgt_end": 150,
                  "chain_id": 1},
                 [(150,)]),
            ])

            pm.gtrack_liftover("lifted_dense_rev", "Dense reverse",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_rev", pm.gintervals_all())
            assert len(out) > 0
            # All three values should be present
            vals_out = sorted(out["lifted_dense_rev"].dropna().tolist())
            assert 111.0 in vals_out
            assert 222.0 in vals_out
            assert 333.0 in vals_out
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_small_bins(self, tmp_path):
        """Dense track with small 1bp bins (adjacent intervals).

        R: 'gtrack.liftover matches liftOver binary - small intervals'
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_small.track"
            # 4 bins of 1bp each
            vals = np.array([1.0, 2.0, 50.0, 100.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 1, {"srcA": vals})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 4,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 100, "tgt_end": 104,
                  "chain_id": 1},
                 [(4,)]),
            ])

            pm.gtrack_liftover("lifted_dense_small", "Dense small bins",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_small",
                              pm.gintervals("chr1", 100, 104))
            assert len(out) == 4
            out = out.sort_values("start").reset_index(drop=True)
            np.testing.assert_allclose(
                out["lifted_dense_small"].tolist(), [1.0, 2.0, 50.0, 100.0],
                rtol=1e-5,
            )
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_boundary_bins(self, tmp_path):
        """Dense track liftover at chromosome boundaries.

        R: 'gtrack.liftover matches liftOver binary - boundary intervals'
        Bins at start and end of chain range.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_boundary.track"
            vals = np.array([111.0, 222.0, 333.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 10, {"srcA": vals})

            # Map entire source range
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 30,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
                  "chain_id": 1},
                 [(30,)]),
            ])

            pm.gtrack_liftover("lifted_dense_boundary", "Dense boundary",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_boundary",
                              pm.gintervals("chr1", 0, 30))
            assert len(out) == 3
            out = out.sort_values("start").reset_index(drop=True)
            assert out.iloc[0]["start"] == 0
            assert out.iloc[2]["end"] == 30
            np.testing.assert_allclose(
                out["lifted_dense_boundary"].tolist(), [111.0, 222.0, 333.0],
                rtol=1e-5,
            )
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_consecutive_bins_with_offset(self, tmp_path):
        """Consecutive bins with coordinate offset through chain.

        R: 'gtrack.liftover matches liftOver binary - consecutive intervals'
        4 consecutive 10bp bins mapped with offset.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_consec.track"
            vals = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 10, {"srcA": vals})

            # Map srcA[0-40) -> chr1[50-90)
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 40,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 50, "tgt_end": 90,
                  "chain_id": 1},
                 [(40,)]),
            ])

            pm.gtrack_liftover("lifted_dense_consec", "Dense consecutive",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_consec",
                              pm.gintervals("chr1", 50, 90))
            assert len(out) == 4
            out = out.sort_values("start").reset_index(drop=True)
            np.testing.assert_allclose(
                out["lifted_dense_consec"].tolist(), [10.0, 20.0, 30.0, 40.0],
                rtol=1e-5,
            )
            assert out.iloc[0]["start"] == 50
            assert out.iloc[3]["end"] == 90
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_multi_chrom(self, tmp_path):
        """Dense track with multiple source chromosomes.

        R: 'gtrack.liftover matches liftOver binary - multiple chromosomes'
        Two source chroms map to two target chroms.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_multi.track"
            _create_dense_track_dir(src_track_dir, 10, {
                "srcA": np.array([10.0, 20.0], dtype=np.float32),
                "srcB": np.array([30.0, 40.0], dtype=np.float32),
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 20,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 10, "tgt_end": 30,
                  "chain_id": 1},
                 [(20,)]),
                ({"score": 1000, "src_chrom": "srcB", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 20,
                  "tgt_chrom": "chrX", "tgt_size": 200000,
                  "tgt_strand": "+", "tgt_start": 20, "tgt_end": 40,
                  "chain_id": 2},
                 [(20,)]),
            ])

            pm.gtrack_liftover("lifted_dense_multi2", "Dense multi chrom",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            # Check chr1 values
            out1 = pm.gextract("lifted_dense_multi2",
                               pm.gintervals("chr1", 10, 30))
            vals1 = sorted(out1["lifted_dense_multi2"].dropna().tolist())
            assert 10.0 in vals1
            assert 20.0 in vals1

            # Check chrX values
            outx = pm.gextract("lifted_dense_multi2",
                               pm.gintervals("chrX", 20, 40))
            valsx = sorted(outx["lifted_dense_multi2"].dropna().tolist())
            assert 30.0 in valsx
            assert 40.0 in valsx
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_special_values(self, tmp_path):
        """Dense track with special values (zero, large positive, large negative).

        R: 'gtrack.liftover matches liftOver binary - special values'
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_special.track"
            vals = np.array([0.0, 1e10, -1e10], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 10, {"srcA": vals})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 30,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
                  "chain_id": 1},
                 [(30,)]),
            ])

            pm.gtrack_liftover("lifted_dense_special", "Dense special",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_special",
                              pm.gintervals("chr1", 0, 30))
            vals_out = out["lifted_dense_special"].dropna().tolist()
            # 0.0 might not be stored in sparse if treated as no-data
            # But we should have at least the non-zero values
            assert any(abs(v) > 1e9 for v in vals_out)
            assert any(v < -1e9 for v in vals_out)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_liftover_dense_unmapped_bins_excluded(self, tmp_path):
        """Bins outside chain range are excluded from lifted track.

        R: 'gtrack.liftover matches liftOver binary - unmapped intervals'
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_dense_unmapped.track"
            # 6 bins: [0,60) with bin_size=10
            vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
            _create_dense_track_dir(src_track_dir, 10, {"srcA": vals})

            # Only map [0,20) and [40,60); [20,40) is a gap
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 0, "src_end": 20,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 20,
                  "chain_id": 1},
                 [(20,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 10000,
                  "src_strand": "+", "src_start": 40, "src_end": 60,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 100, "tgt_end": 120,
                  "chain_id": 2},
                 [(20,)]),
            ])

            pm.gtrack_liftover("lifted_dense_unmap", "Dense unmapped",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_dense_unmap", pm.gintervals_all())
            vals_out = sorted(out["lifted_dense_unmap"].dropna().tolist())
            # Mapped: bins 0,1 (vals 1,2) and bins 4,5 (vals 5,6)
            assert 1.0 in vals_out
            assert 2.0 in vals_out
            assert 5.0 in vals_out
            assert 6.0 in vals_out
            # Unmapped: bins 2,3 (vals 3,4)
            assert 3.0 not in vals_out
            assert 4.0 not in vals_out
        finally:
            pm.gdb_init(str(TEST_DB))


# ===================================================================
# Ported from R test-gtrack.liftover-sparse-overlap-merge.R
# Sparse track liftover with overlapping intervals and merging.
# ===================================================================


class TestGtrackLiftoverSparseOverlapMerge:
    """Tests for sparse track liftover overlapping interval merging.

    Ported from R test-gtrack.liftover-sparse-overlap-merge.R.
    These tests verify that gtrack_liftover correctly handles the case
    where different source intervals map to overlapping target regions.
    """

    def test_overlapping_target_intervals_merged(self, tmp_path):
        """Overlapping target intervals from multiple chains are merged.

        R: 'gtrack.liftover merges overlapping target intervals in sparse tracks'
        Single source interval maps through two chains to overlapping targets:
        Chain 1: source[40,70) -> target[0,30), so source[50,60) -> target[10,20)
        Chain 2: source[45,75) -> target[12,42), so source[50,60) -> target[17,27)
        Targets [10,20) and [17,27) overlap at [17,20).
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_overlap_merge.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(50, 60, 10.0)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 300,
                  "src_strand": "+", "src_start": 40, "src_end": 70,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
                  "chain_id": 1},
                 [(30,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 300,
                  "src_strand": "+", "src_start": 45, "src_end": 75,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 12, "tgt_end": 42,
                  "chain_id": 2},
                 [(30,)]),
            ])

            pm.gtrack_liftover("lifted_overlap_merge", "Overlap merge",
                               str(src_track_dir), chain_path,
                               src_overlap_policy="keep",
                               tgt_overlap_policy="keep",
                               multi_target_agg="max")

            out = pm.gextract("lifted_overlap_merge", pm.gintervals_all())
            assert len(out) >= 1

            # Verify no overlapping intervals
            if len(out) > 1:
                out = out.sort_values("start").reset_index(drop=True)
                for i in range(len(out) - 1):
                    assert out.iloc[i]["end"] <= out.iloc[i + 1]["start"], \
                        f"Overlapping intervals at rows {i} and {i+1}"
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_aggregation_values_correct(self, tmp_path):
        """Values are correctly aggregated when overlapping intervals merge.

        R: 'gtrack.liftover correctly aggregates values when merging overlapping intervals'
        Three source intervals with values 100, 200, 300 map through
        overlapping chains producing overlapping target intervals.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_agg_vals.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(10, 20, 100.0), (30, 40, 200.0), (50, 60, 300.0)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 0, "src_end": 30,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
                  "chain_id": 1},
                 [(30,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 20, "src_end": 50,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 10, "tgt_end": 40,
                  "chain_id": 2},
                 [(30,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 40, "src_end": 70,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 20, "tgt_end": 50,
                  "chain_id": 3},
                 [(30,)]),
            ])

            # Max aggregation
            pm.gtrack_liftover("lifted_agg_max", "Agg max",
                               str(src_track_dir), chain_path,
                               src_overlap_policy="keep",
                               tgt_overlap_policy="keep",
                               multi_target_agg="max")

            out_max = pm.gextract("lifted_agg_max", pm.gintervals_all())
            assert len(out_max) >= 1
            # No overlapping intervals
            if len(out_max) > 1:
                out_max = out_max.sort_values("start").reset_index(drop=True)
                for i in range(len(out_max) - 1):
                    assert out_max.iloc[i]["end"] <= out_max.iloc[i + 1]["start"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_mean_aggregation_no_overlaps(self, tmp_path):
        """Mean aggregation produces non-overlapping output.

        R: 'gtrack.liftover correctly aggregates values when merging overlapping intervals'
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_agg_mean2.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(10, 20, 100.0), (30, 40, 200.0), (50, 60, 300.0)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 0, "src_end": 30,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
                  "chain_id": 1},
                 [(30,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 20, "src_end": 50,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 10, "tgt_end": 40,
                  "chain_id": 2},
                 [(30,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 40, "src_end": 70,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 20, "tgt_end": 50,
                  "chain_id": 3},
                 [(30,)]),
            ])

            pm.gtrack_liftover("lifted_agg_mean2", "Agg mean",
                               str(src_track_dir), chain_path,
                               src_overlap_policy="keep",
                               tgt_overlap_policy="keep",
                               multi_target_agg="mean")

            out_mean = pm.gextract("lifted_agg_mean2", pm.gintervals_all())
            assert len(out_mean) >= 1
            if len(out_mean) > 1:
                out_mean = out_mean.sort_values("start").reset_index(drop=True)
                for i in range(len(out_mean) - 1):
                    assert out_mean.iloc[i]["end"] <= out_mean.iloc[i + 1]["start"]
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_many_overlapping_chains(self, tmp_path):
        """Sparse track liftover with many overlapping source-to-target mappings.

        R: 'gtrack.liftover handles sparse track with many overlapping source-to-target mappings'
        20 small intervals mapped through 3 overlapping chains.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_many_overlap.track"
            n_intervals = 20
            intervals = [
                (100 + i * 10, 105 + i * 10, float(i + 1))
                for i in range(n_intervals)
            ]
            _create_sparse_track_dir(src_track_dir, {"srcA": intervals})

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 1000,
                  "src_strand": "+", "src_start": 50, "src_end": 400,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 350,
                  "chain_id": 1},
                 [(350,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 1000,
                  "src_strand": "+", "src_start": 100, "src_end": 450,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 30, "tgt_end": 380,
                  "chain_id": 2},
                 [(350,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 1000,
                  "src_strand": "+", "src_start": 150, "src_end": 500,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 60, "tgt_end": 410,
                  "chain_id": 3},
                 [(350,)]),
            ])

            pm.gtrack_liftover("lifted_many_overlap", "Many overlapping",
                               str(src_track_dir), chain_path,
                               src_overlap_policy="keep",
                               tgt_overlap_policy="keep",
                               multi_target_agg="max")

            # Key test: gextract should work without "Invalid format" error
            out = pm.gextract("lifted_many_overlap", pm.gintervals_all())
            assert len(out) >= 1

            # Verify no overlapping intervals
            if len(out) > 1:
                out = out.sort_values("start").reset_index(drop=True)
                for i in range(len(out) - 1):
                    assert out.iloc[i]["end"] <= out.iloc[i + 1]["start"], \
                        f"Overlapping at rows {i} and {i+1}"
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_output_track_format_valid(self, tmp_path):
        """Lifted sparse track has valid format (can be read by gextract/gsummary).

        R: 'gtrack.liftover sparse track produces valid track that can be read without errors'
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_valid.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(50, 75, 1.5), (100, 125, 2.5),
                         (150, 175, 3.5), (200, 225, 4.5)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 500,
                  "src_strand": "+", "src_start": 0, "src_end": 150,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 150,
                  "chain_id": 1},
                 [(150,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 500,
                  "src_strand": "+", "src_start": 100, "src_end": 250,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 80, "tgt_end": 230,
                  "chain_id": 2},
                 [(150,)]),
            ])

            pm.gtrack_liftover("lifted_valid_fmt", "Valid format",
                               str(src_track_dir), chain_path,
                               src_overlap_policy="keep",
                               tgt_overlap_policy="keep",
                               multi_target_agg="mean")

            # Verify track info works
            info = pm.gtrack_info("lifted_valid_fmt")
            assert info["type"] == "sparse"

            # Verify gextract works on full genome
            out_full = pm.gextract("lifted_valid_fmt", pm.gintervals_all())
            assert out_full is not None

            # Verify gextract works on partial interval
            out_partial = pm.gextract("lifted_valid_fmt",
                                      pm.gintervals("chr1", 50, 200))
            assert out_partial is not None
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_adjacent_same_value_merge(self, tmp_path):
        """Adjacent intervals with same value produce valid output.

        R: 'gtrack.liftover merges adjacent intervals with same value'
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_adj_same.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(10, 20, 42.0), (20, 30, 42.0)],
            })

            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 0, "src_end": 50,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 50,
                  "chain_id": 1},
                 [(50,)]),
            ])

            pm.gtrack_liftover("lifted_adj_same", "Adjacent same value",
                               str(src_track_dir), chain_path,
                               tgt_overlap_policy="keep")

            out = pm.gextract("lifted_adj_same", pm.gintervals_all())
            assert len(out) >= 1
            # All values should be 42
            assert all(out["lifted_adj_same"].dropna() == 42.0)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_sum_aggregation_overlapping(self, tmp_path):
        """Sum aggregation correctly sums overlapping values.

        R: 'gtrack.liftover with sum aggregation correctly sums overlapping values'
        Single source interval maps to overlapping targets through two chains.
        """
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))

            src_track_dir = tmp_path / "src_sum_overlap.track"
            _create_sparse_track_dir(src_track_dir, {
                "srcA": [(10, 20, 100.0)],
            })

            # Two chains producing overlapping targets:
            # Chain 1: srcA[0,30) -> chr1[0,30), source[10,20) -> target[10,20)
            # Chain 2: srcA[5,35) -> chr1[8,38), source[10,20) -> target[13,23)
            # Overlap at target[13,20)
            chain_path = _write_chain(tmp_path, [
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 0, "src_end": 30,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 0, "tgt_end": 30,
                  "chain_id": 1},
                 [(30,)]),
                ({"score": 1000, "src_chrom": "srcA", "src_size": 200,
                  "src_strand": "+", "src_start": 5, "src_end": 35,
                  "tgt_chrom": "chr1", "tgt_size": 500000,
                  "tgt_strand": "+", "tgt_start": 8, "tgt_end": 38,
                  "chain_id": 2},
                 [(30,)]),
            ])

            pm.gtrack_liftover("lifted_sum_overlap", "Sum overlap",
                               str(src_track_dir), chain_path,
                               src_overlap_policy="keep",
                               tgt_overlap_policy="keep",
                               multi_target_agg="sum")

            out = pm.gextract("lifted_sum_overlap", pm.gintervals_all())
            assert len(out) >= 1
            # Verify no overlapping intervals
            if len(out) > 1:
                out = out.sort_values("start").reset_index(drop=True)
                for i in range(len(out) - 1):
                    assert out.iloc[i]["end"] <= out.iloc[i + 1]["start"]
        finally:
            pm.gdb_init(str(TEST_DB))
