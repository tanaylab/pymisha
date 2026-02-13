import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def test_gtrack_create_sparse(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 20, 10],
                "end": [10, 30, 20],
            }
        )
        pm.gtrack_create_sparse("created_sparse", "sparse test", intervals, [1.0, 2.0, 3.0])
        assert pm.gtrack_exists("created_sparse")
        info = pm.gtrack_info("created_sparse")
        assert info["type"] == "sparse"

        q = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2"],
                "start": [0, 20, 10],
                "end": [10, 30, 20],
            }
        )
        out = pm.gextract("created_sparse", q)
        assert out is not None
        np.testing.assert_allclose(out["created_sparse"].to_numpy(), np.array([1.0, 2.0, 3.0]), equal_nan=True)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_create_dense_overlap_and_defval(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        intervals = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1"],
                "start": [0, 5],
                "end": [10, 15],
            }
        )
        pm.gtrack_create_dense("created_dense", "dense test", intervals, [2.0, 4.0], binsize=10, defval=1.0)
        out = pm.gextract(
            "created_dense",
            pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [10, 20]}),
            iterator=10,
        )
        assert out is not None
        got = out["created_dense"].to_numpy(dtype=float, copy=False)
        np.testing.assert_allclose(got, np.array([40.0 / 15.0, 2.5]), rtol=1e-6, atol=1e-6)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_import_bed_sparse_with_attrs(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        bed_path = tmp_path / "input.bed"
        bed_path.write_text(
            "chr1\t0\t10\ta\t5\n"
            "chr1\t10\t20\tb\t7\n",
            encoding="utf-8",
        )
        pm.gtrack_import(
            "imported_sparse",
            "import test",
            str(bed_path),
            binsize=0,
            attrs={"author": "tester"},
        )
        out = pm.gextract(
            "imported_sparse",
            pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [10, 20]}),
        )
        assert out is not None
        np.testing.assert_allclose(out["imported_sparse"].to_numpy(), np.array([5.0, 7.0]), equal_nan=True)
        assert pm.gtrack_attr_get("imported_sparse", "author") == "tester"
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_import_wig_fixedstep_dense(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        wig_path = tmp_path / "input.wig"
        wig_path.write_text(
            "track type=wiggle_0 name=test\n"
            "fixedStep chrom=chr1 start=1 step=10 span=10\n"
            "2\n"
            "4\n",
            encoding="utf-8",
        )
        pm.gtrack_import("imported_wig", "wig test", str(wig_path), binsize=10, defval=np.nan)
        out = pm.gextract(
            "imported_wig",
            pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [10, 20]}),
            iterator=10,
        )
        assert out is not None
        np.testing.assert_allclose(out["imported_wig"].to_numpy(dtype=float), np.array([2.0, 4.0]), equal_nan=True)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_import_bigwig_optional(tmp_path):
    pybw = pytest.importorskip("pyBigWig")
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        bw_path = tmp_path / "input.bw"
        bw = pybw.open(str(bw_path), "w")
        try:
            bw.addHeader([("chr1", 1000)])
            bw.addEntries(["chr1", "chr1"], [0, 10], ends=[10, 20], values=[5.0, 7.0])
        finally:
            bw.close()

        pm.gtrack_import("imported_bw", "bigwig test", str(bw_path), binsize=0)
        out = pm.gextract(
            "imported_bw",
            pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [10, 20]}),
        )
        assert out is not None
        np.testing.assert_allclose(out["imported_bw"].to_numpy(dtype=float), np.array([5.0, 7.0]), equal_nan=True)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_create_auto_indexed_db(tmp_path):
    root = _copy_db(tmp_path)
    try:
        seq_dir = root / "seq"
        (seq_dir / "genome.idx").write_bytes(b"")
        (seq_dir / "genome.seq").write_bytes(b"")
        pm.gdb_init(str(root))

        intervals = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})
        pm.gtrack_create_sparse("indexed_sparse", "indexed sparse", intervals, [1.0])

        tdir = root / "tracks" / "indexed_sparse.track"
        assert (tdir / "track.idx").exists()
        assert (tdir / "track.dat").exists()
        assert pm.gtrack_info("indexed_sparse")["format"] == "indexed"
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_create_expression_dense_streaming(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        pm.gtrack_create("created_expr_dense", "expr dense", "dense_track * 2", iterator=10)
        out = pm.gextract(
            "created_expr_dense",
            pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]}),
            iterator=10,
        )
        ref = pm.gextract(
            "dense_track * 2",
            pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]}),
            iterator=10,
        )
        assert out is not None and ref is not None
        np.testing.assert_allclose(
            out["created_expr_dense"].to_numpy(dtype=float),
            ref["dense_track * 2"].to_numpy(dtype=float),
            equal_nan=True,
        )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_copy_mv_rm(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        pm.gtrack_copy("dense_track", "dense_copy")
        assert pm.gtrack_exists("dense_copy")

        pm.gtrack_mv("dense_copy", "dense_renamed")
        assert not pm.gtrack_exists("dense_copy")
        assert pm.gtrack_exists("dense_renamed")

        with pytest.raises(ValueError):
            pm.gtrack_rm("dense_renamed", force=False)
        pm.gtrack_rm("dense_renamed", force=True)
        assert not pm.gtrack_exists("dense_renamed")
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_import_mappedseq_sparse_dedup(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text(
            "A\tchr1\t10\t+\n"
            "A\tchr1\t10\t+\n"
            "A\tchr1\t20\t-\n",
            encoding="utf-8",
        )
        stats = pm.gtrack_import_mappedseq(
            "mapped_sparse",
            "mapped sparse",
            str(inp),
            pileup=0,
            binsize=-1,
            cols_order=(1, 2, 3, 4),
            remove_dups=True,
        )
        assert "total" in stats and "chrom" in stats
        out = pm.gextract(
            "mapped_sparse",
            pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [10, 21], "end": [11, 22]}),
        )
        assert out is not None
        np.testing.assert_allclose(out["mapped_sparse"].to_numpy(dtype=float), np.array([1.0, 1.0]), equal_nan=True)
    finally:
        pm.gdb_init(str(TEST_DB))


def test_gtrack_import_mappedseq_dense(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped_dense.tsv"
        inp.write_text(
            "AAA\tchr1\t10\t+\n"
            "AAA\tchr1\t30\t+\n",
            encoding="utf-8",
        )
        pm.gtrack_import_mappedseq(
            "mapped_dense",
            "mapped dense",
            str(inp),
            pileup=5,
            binsize=5,
            cols_order=(1, 2, 3, 4),
            remove_dups=True,
        )
        info = pm.gtrack_info("mapped_dense")
        assert info["type"] == "dense"
        out = pm.gextract(
            "mapped_dense",
            pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [10, 30], "end": [15, 35]}),
            iterator=5,
        )
        assert out is not None
        vals = out["mapped_dense"].to_numpy(dtype=float)
        assert vals[0] > 0
        assert vals[1] > 0
    finally:
        pm.gdb_init(str(TEST_DB))


class TestGtrackImportRParity:
    """R misha parity tests ported from test-gtrack.import2/4/5.R."""

    def test_import_bed_dense_with_binsize(self, tmp_path):
        """Port of: import BED as dense track with binsize."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            bed = tmp_path / "dense_bed.bed"
            bed.write_text("chr1\t0\t10\tseg1\t2\nchr1\t10\t20\tseg2\t4\n")
            pm.gtrack_import("bed_dense", "BED dense track", str(bed), binsize=5, defval=0)
            info = pm.gtrack_info("bed_dense")
            assert info["type"] == "dense"
            assert int(info["bin_size"]) == 5
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_import_tsv_sparse(self, tmp_path):
        """Port of: import tab-delimited with header chrom/start/end/value (sparse)."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            tsv = tmp_path / "sparse.tsv"
            tsv.write_text("chrom\tstart\tend\tvalue\nchr1\t0\t3\t2.5\nchr1\t4\t6\t1.0\n")
            pm.gtrack_import("tsv_sparse", "TSV import track", str(tsv), binsize=0)
            assert pm.gtrack_exists("tsv_sparse")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_import_tsv_multiple_value_cols_error(self, tmp_path):
        """Port of: import tab-delimited with multiple value columns fails."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            tsv_bad = tmp_path / "bad.tsv"
            tsv_bad.write_text("chrom\tstart\tend\tv1\tv2\nchr1\t0\t3\t2.5\t1.1\n")
            with pytest.raises(ValueError, match="value column"):
                pm.gtrack_import("tsv_bad", "bad TSV", str(tsv_bad), binsize=0)
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_import_null_attrs(self, tmp_path):
        """Port of: import with attrs parameter - NULL attrs works."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            bed = tmp_path / "null_attr.bed"
            bed.write_text("chr1\t0\t10\ta\t5\n")
            pm.gtrack_import("null_attrs", "Test track", str(bed), binsize=0, attrs=None)
            assert pm.gtrack_exists("null_attrs")
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_import_custom_attrs_with_defaults(self, tmp_path):
        """Port of: attrs parameter - attributes don't interfere with default attributes."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            bed = tmp_path / "custom_attr.bed"
            bed.write_text("chr1\t0\t10\ta\t5\n")
            pm.gtrack_import("custom_attrs", "Test description", str(bed), binsize=0,
                             attrs={"author": "test_user", "custom_attr": "custom_value"})
            assert pm.gtrack_attr_get("custom_attrs", "author") == "test_user"
            assert pm.gtrack_attr_get("custom_attrs", "custom_attr") == "custom_value"
            assert pm.gtrack_attr_get("custom_attrs", "description") == "Test description"
            assert len(pm.gtrack_attr_get("custom_attrs", "created.by")) > 0
            assert len(pm.gtrack_attr_get("custom_attrs", "created.date")) > 0
        finally:
            pm.gdb_init(str(TEST_DB))

    def test_import_multiple_attrs(self, tmp_path):
        """Port of: import with attrs parameter - multiple attributes."""
        root = _copy_db(tmp_path)
        try:
            pm.gdb_init(str(root))
            bed = tmp_path / "multi_attr.bed"
            bed.write_text("chr1\t0\t10\ta\t5\n")
            pm.gtrack_import("multi_attrs", "Test track", str(bed), binsize=0,
                             attrs={"author": "test_user", "version": "1.0", "experiment": "test_exp"})
            assert pm.gtrack_attr_get("multi_attrs", "author") == "test_user"
            assert pm.gtrack_attr_get("multi_attrs", "version") == "1.0"
            assert pm.gtrack_attr_get("multi_attrs", "experiment") == "test_exp"
        finally:
            pm.gdb_init(str(TEST_DB))


def test_gtrack_import_set_mixed_success(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        good = tmp_path / "good.bed"
        bad = tmp_path / "bad.bed"
        good.write_text("chr1\t0\t10\ta\t3\n", encoding="utf-8")
        bad.write_text("chr1\tbad\t10\ta\t3\n", encoding="utf-8")

        res = pm.gtrack_import_set(
            description="batch import",
            path=str(tmp_path / "*.bed"),
            binsize=0,
            track_prefix="batch_",
            defval=np.nan,
        )
        assert "good.bed" in res.get("files_imported", [])
        assert "bad.bed" in res.get("files_failed", [])
        assert pm.gtrack_exists("batch_good")
        assert not pm.gtrack_exists("batch_bad")
    finally:
        pm.gdb_init(str(TEST_DB))
