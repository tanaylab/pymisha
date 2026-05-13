"""Tests for .fai output of gsynth_sample and gsynth_random (FASTA mode)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _parse_fai(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["name", "length", "offset", "linebases", "linewidth"],
        dtype={"name": str, "length": int, "offset": int,
               "linebases": int, "linewidth": int},
    )


class TestSampleFai:
    """Mirrors R `gsynth.sample FASTA output writes a samtools-compatible .fai alongside`."""

    @pytest.fixture(scope="class")
    def model(self):
        pm.gdb_init_examples()
        return pm.gsynth_train(
            intervals=pm.gintervals("1", 0, 50_000), iterator=200, k=2,
        )

    def test_single_record_fai(self, model, tmp_path):
        out = str(tmp_path / "sample.fa")
        pm.gsynth_sample(
            model, output=out, output_format="fasta",
            intervals=pm.gintervals("1", 0, 5000), seed=60427,
        )
        fai_path = out + ".fai"
        assert os.path.exists(fai_path)
        fai = _parse_fai(fai_path)
        assert len(fai) == 1
        assert fai["length"].iloc[0] == 5000
        assert fai["linebases"].iloc[0] == 60
        assert fai["linewidth"].iloc[0] == 61
        # Single record: header at byte 0, offset == len(">name\n")
        assert fai["offset"].iloc[0] == 1 + len(fai["name"].iloc[0]) + 1
        # Seek to offset, read linebases bytes -> all ACGT or N.
        with open(out, "rb") as f:
            f.seek(int(fai["offset"].iloc[0]))
            buf = f.read(int(fai["linebases"].iloc[0])).decode("ascii")
        assert all(c in "ACGTN" for c in buf)

    def test_multi_record_fai(self, model, tmp_path):
        out = str(tmp_path / "multi.fa")
        intervals = pd.DataFrame({
            "chrom": ["1", "1", "1"],
            "start": [0, 10_000, 20_000],
            "end":   [2000, 12_000, 22_500],
        })
        pm.gsynth_sample(
            model, output=out, output_format="fasta",
            intervals=intervals, seed=60427,
        )
        fai = _parse_fai(out + ".fai")
        assert len(fai) == 3
        np.testing.assert_array_equal(
            fai["length"].to_numpy(),
            (intervals["end"] - intervals["start"]).to_numpy(),
        )
        # Per-record seek+read recovers exactly `length` bp after stripping \n.
        with open(out, "rb") as f:
            for _, row in fai.iterrows():
                length = int(row["length"])
                linebases = int(row["linebases"])
                linewidth = int(row["linewidth"])
                n_full = length // linebases if linebases else 0
                tail = length % linebases if linebases else 0
                bytes_to_read = n_full * linewidth + (tail + 1 if tail else 0)
                f.seek(int(row["offset"]))
                chunk = f.read(bytes_to_read).decode("ascii")
                assert len(chunk.replace("\n", "")) == length


class TestRandomFai:
    """Pymisha-only: gsynth_random with output_format='fasta' must also emit .fai."""

    def test_single_record(self, tmp_path):
        pm.gdb_init_examples()
        out = str(tmp_path / "rand.fa")
        pm.gsynth_random(
            output=out, output_format="fasta",
            intervals=pm.gintervals("1", 0, 3000), seed=60427,
        )
        fai = _parse_fai(out + ".fai")
        assert len(fai) == 1
        assert fai["length"].iloc[0] == 3000
        assert fai["linebases"].iloc[0] == 60
        assert fai["linewidth"].iloc[0] == 61

    def test_multi_record(self, tmp_path):
        pm.gdb_init_examples()
        out = str(tmp_path / "rand_multi.fa")
        intervals = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 5000],
            "end":   [1000, 7000],
        })
        pm.gsynth_random(
            output=out, output_format="fasta",
            intervals=intervals, seed=60427,
        )
        fai = _parse_fai(out + ".fai")
        assert len(fai) == 2
        np.testing.assert_array_equal(
            fai["length"].to_numpy(),
            (intervals["end"] - intervals["start"]).to_numpy(),
        )
