"""Tests for gtrack_export_bedgraph and gtrack_export_bigwig."""

import gzip
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


class TestBedGraphExport:
    """Tests for gtrack_export_bedgraph."""

    def test_basic_export(self, tmp_path):
        """Export a dense track to bedGraph and verify format."""
        outfile = str(tmp_path / "dense.bedgraph")
        pm.gtrack_export_bedgraph("dense_track", outfile)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # First line should be the header
        assert lines[0].startswith("track type=bedGraph")
        assert 'name="dense_track"' in lines[0]

        # Data lines should have 4 tab-separated columns
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == 4, f"Expected 4 columns, got {len(parts)}: {line}"
            # chrom should be a non-empty string
            assert len(parts[0]) > 0
            # start and end should be integers
            int(parts[1])
            int(parts[2])
            # value should be numeric
            float(parts[3])

    def test_export_with_intervals(self, tmp_path):
        """Export with specific intervals produces data only for those intervals."""
        outfile = str(tmp_path / "chr1.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph("dense_track", outfile, intervals=intervals)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # Header + at least some data
        assert len(lines) >= 2

        # All data lines should be for chromosome 1 (name may or may not have "chr" prefix)
        genome_chroms = pm.gintervals_all()
        chr1_name = genome_chroms[genome_chroms["chrom"].str.contains("1")]["chrom"].iloc[0]
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert parts[0] == chr1_name

    def test_export_with_iterator(self, tmp_path):
        """Export with explicit iterator bin size."""
        outfile = str(tmp_path / "iter.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, iterator=200
        )

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # Should have header + 5 bins (1000/200)
        assert len(lines) == 6  # 1 header + 5 data lines

    def test_gzip_compression(self, tmp_path):
        """Export to .gz file produces valid gzip output."""
        outfile = str(tmp_path / "dense.bedgraph.gz")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, iterator=200
        )

        assert os.path.exists(outfile)

        # Verify it's a valid gzip file
        with gzip.open(outfile, "rt") as f:
            lines = f.readlines()

        assert lines[0].startswith("track type=bedGraph")
        assert len(lines) == 6  # header + 5 data lines

    def test_nan_exclusion(self, tmp_path):
        """NaN values are excluded from bedGraph output."""
        outfile = str(tmp_path / "sparse.bedgraph")
        # Use a region where sparse_track has no data (expect NaNs)
        intervals = pm.gintervals("X", 190000, 200000)
        pm.gtrack_export_bedgraph(
            "sparse_track", outfile, intervals=intervals, iterator=1000
        )

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # Data lines should not contain NaN
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert parts[3].lower() != "nan", f"NaN found in output: {line}"

    def test_custom_name(self, tmp_path):
        """Custom name appears in bedGraph header."""
        outfile = str(tmp_path / "custom.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, name="my_track"
        )

        with open(outfile) as f:
            header = f.readline()

        assert 'name="my_track"' in header

    def test_sorted_output(self, tmp_path):
        """Output should be sorted by chrom (genome order) then start."""
        outfile = str(tmp_path / "sorted.bedgraph")
        # Use intervals across multiple chromosomes
        intervals = pd.concat(
            [
                pm.gintervals("2", 0, 500),
                pm.gintervals("1", 0, 500),
            ],
            ignore_index=True,
        )
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, iterator=100
        )

        with open(outfile) as f:
            lines = f.readlines()

        data_lines = lines[1:]
        chroms = [line.split("\t")[0] for line in data_lines]
        starts = [int(line.split("\t")[1]) for line in data_lines]

        # chr1 should come before chr2
        chr1_indices = [i for i, c in enumerate(chroms) if c == "chr1"]
        chr2_indices = [i for i, c in enumerate(chroms) if c == "chr2"]
        if chr1_indices and chr2_indices:
            assert max(chr1_indices) < min(chr2_indices)

        # Within each chromosome, starts should be sorted
        for chrom in set(chroms):
            chrom_starts = [
                starts[i] for i, c in enumerate(chroms) if c == chrom
            ]
            assert chrom_starts == sorted(chrom_starts)

    def test_track_expression(self, tmp_path):
        """Export a track expression (not just a track name)."""
        outfile = str(tmp_path / "expr.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track * 2",
            outfile,
            intervals=intervals,
            iterator=200,
        )

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        assert len(lines) >= 2
        # Header should contain the expression
        assert 'name="dense_track * 2"' in lines[0]

    def test_nonexistent_directory_raises(self, tmp_path):
        """Writing to a nonexistent directory raises FileNotFoundError."""
        outfile = str(tmp_path / "nonexistent" / "dense.bedgraph")
        with pytest.raises(FileNotFoundError):
            pm.gtrack_export_bedgraph("dense_track", outfile)

    def test_sparse_track_export(self, tmp_path):
        """Export a sparse track to bedGraph."""
        outfile = str(tmp_path / "sparse.bedgraph")
        pm.gtrack_export_bedgraph("sparse_track", outfile)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        assert lines[0].startswith("track type=bedGraph")
        # Sparse track should still have some data
        assert len(lines) >= 2


class TestBigWigExport:
    """Tests for gtrack_export_bigwig."""

    @pytest.fixture(autouse=True)
    def _check_converter(self):
        """Skip BigWig tests if bedGraphToBigWig is not available."""
        if (
            shutil.which("bedGraphToBigWig") is None
            and shutil.which("wigToBigWig") is None
        ):
            pytest.skip("bedGraphToBigWig/wigToBigWig not available")

    def test_basic_bigwig_export(self, tmp_path):
        """Export a dense track to BigWig."""
        outfile = str(tmp_path / "dense.bw")
        pm.gtrack_export_bigwig("dense_track", outfile)

        assert os.path.exists(outfile)
        assert os.path.getsize(outfile) > 0

    def test_bigwig_with_intervals(self, tmp_path):
        """Export to BigWig with specific intervals."""
        outfile = str(tmp_path / "chr1.bw")
        intervals = pm.gintervals("1", 0, 10000)
        pm.gtrack_export_bigwig(
            "dense_track", outfile, intervals=intervals, iterator=100
        )

        assert os.path.exists(outfile)
        assert os.path.getsize(outfile) > 0

    def test_bigwig_nonexistent_directory_raises(self, tmp_path):
        """Writing BigWig to a nonexistent directory raises FileNotFoundError."""
        outfile = str(tmp_path / "nonexistent" / "dense.bw")
        with pytest.raises(FileNotFoundError):
            pm.gtrack_export_bigwig("dense_track", outfile)


class TestBigWigConverterMissing:
    """Tests for BigWig when no converter is available."""

    def test_missing_converter_raises(self, tmp_path, monkeypatch):
        """RuntimeError is raised when bedGraphToBigWig is not on PATH."""
        # Monkeypatch shutil.which to always return None
        monkeypatch.setattr(shutil, "which", lambda x: None)
        outfile = str(tmp_path / "dense.bw")
        with pytest.raises(RuntimeError, match="bedGraphToBigWig"):
            pm.gtrack_export_bigwig("dense_track", outfile)
