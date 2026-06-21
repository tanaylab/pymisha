"""Arg-validation + FSM parser tests for the C++ pm_import_mappedseq.

Full end-to-end semantics are tested via gtrack_import_mappedseq once
the Python dispatcher is wired up (Task 6 of the plan).
"""
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def _track_dir(root: Path, name: str) -> str:
    return str(root / "tracks" / f"{name}.track")


def test_pm_import_mappedseq_pileup_negative(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text("A\t1\t10\t+\n")
        with pytest.raises(Exception, match="pileup cannot be negative"):
            _pymisha.pm_import_mappedseq(
                _track_dir(root, "x"), str(inp),
                -1, -1, (1, 2, 3, 4), True,
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_sparse_requires_binsize_minus_one(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text("A\t1\t10\t+\n")
        with pytest.raises(Exception, match="binsize must be -1"):
            _pymisha.pm_import_mappedseq(
                _track_dir(root, "x"), str(inp),
                0, 5, (1, 2, 3, 4), True,
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_dense_requires_positive_binsize(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text("A\t1\t10\t+\n")
        with pytest.raises(Exception, match="binsize must be > 0"):
            _pymisha.pm_import_mappedseq(
                _track_dir(root, "x"), str(inp),
                5, 0, (1, 2, 3, 4), True,
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_cols_order_dupes(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text("A\t1\t10\t+\n")
        with pytest.raises(Exception, match="share index"):
            _pymisha.pm_import_mappedseq(
                _track_dir(root, "x"), str(inp),
                0, -1, (1, 1, 3, 4), True,
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_file_not_found(tmp_path):
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        with pytest.raises(Exception, match="File not found"):
            _pymisha.pm_import_mappedseq(
                _track_dir(root, "x"), str(tmp_path / "nonexistent.sam"),
                0, -1, (1, 2, 3, 4), True,
            )
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_returns_dict(tmp_path):
    """Parser wired: single mapped line -> total.mapped == 1."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text("A\t1\t10\t+\n")
        result = _pymisha.pm_import_mappedseq(
            _track_dir(root, "x"), str(inp),
            0, -1, (1, 2, 3, 4), True,
        )
        assert "total" in result and "chrom_stats" in result
        assert result["total"]["total.mapped"] == 1.0
        assert isinstance(result["chrom_stats"]["chrom"], list)
        assert len(result["chrom_stats"]["chrom"]) > 0  # at least one DB chrom
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_parser_counts_basic(tmp_path):
    """Tab-delimited fixture: chrom mismatches, bad coord, bad strand
    all count as unmapped (R-parity)."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv"
        inp.write_text(
            "A\t1\t10\t+\n"          # mapped
            "A\t1\t20\t-\n"          # mapped
            "A\tchrZZ\t30\t+\n"      # unknown chrom -> unmapped
            "A\t1\tNOTNUM\t+\n"      # bad coord -> unmapped
            "A\t1\t10\t?\n",         # bad strand -> unmapped
            encoding="utf-8",
        )
        res = _pymisha.pm_import_mappedseq(
            _track_dir(root, "x"), str(inp),
            0, -1, (1, 2, 3, 4), True,
        )
        assert res["total"]["total.mapped"] == 2.0
        assert res["total"]["total.unmapped"] == 3.0
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_parser_sam_mode(tmp_path):
    """SAM mode: @-header lines skipped; strand from flag bit 0x10."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "in.sam"
        inp.write_text(
            "@HD\tVN:1.6\n"
            "@SQ\tSN:1\tLN:500000\n"
            "r1\t0\t1\t100\t30\t5M\t*\t0\t0\tAAAAA\t*\n"   # flag=0 -> +
            "r2\t16\t1\t200\t30\t5M\t*\t0\t0\tAAAAA\t*\n"  # flag=16 -> -
            "r3\t4\tunknown\t1\t0\t*\t*\t0\t0\t*\t*\n",     # unknown chrom -> unmapped
            encoding="utf-8",
        )
        res = _pymisha.pm_import_mappedseq(
            _track_dir(root, "x"), str(inp),
            0, -1, None, True,
        )
        assert res["total"]["total.mapped"] == 2.0
        assert res["total"]["total.unmapped"] == 1.0
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_parser_gzip(tmp_path):
    """Gzip auto-detect via 0x1f 0x8b magic."""
    import gzip
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "mapped.tsv.gz"
        with gzip.open(inp, "wt") as fh:
            fh.write("A\t1\t10\t+\n")
            fh.write("A\t1\t20\t-\n")
        res = _pymisha.pm_import_mappedseq(
            _track_dir(root, "x"), str(inp),
            0, -1, (1, 2, 3, 4), True,
        )
        assert res["total"]["total.mapped"] == 2.0
        assert res["total"]["total.unmapped"] == 0.0
    finally:
        pm.gdb_init(str(TEST_DB))


def _pileup_via_cpp(tmp_path, lines, pileup, binsize, track_name="m"):
    root = _copy_db(tmp_path)
    pm.gdb_init(str(root))
    inp = tmp_path / "m.tsv"
    inp.write_text("".join(lines))
    track_dir = root / "tracks" / f"{track_name}.track"
    res = _pymisha.pm_import_mappedseq(
        str(track_dir), str(inp),
        pileup, binsize, (1, 2, 3, 4), True,
    )
    _pymisha.pm_dbreload()
    return res, root


def test_pm_import_mappedseq_dense_single_read(tmp_path):
    """Single +read with pileup=5, coord=100 -> covers [100, 105], single bin."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path, ["AAAAA\t1\t100\t+\n"], pileup=5, binsize=5,
        )
        assert res["total"]["total.mapped"] == 1.0
        out = pm.gextract(
            "m",
            pd.DataFrame({"chrom": ["1"], "start": [100], "end": [105]}),
            iterator=5,
        )
        np.testing.assert_allclose(out["m"].to_numpy(dtype=float), np.array([1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_dense_minus_strand_extension(tmp_path):
    """-strand read at coord=100, seq=AAAAA, pileup=5:
    R extends from (coord+seq_size - pileup) = 100+5-5 = 100 to (coord+seq_size) = 105.
    So the pileup window is [100, 105], same as +strand at coord=100."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path, ["AAAAA\t1\t100\t-\n"], pileup=5, binsize=5,
        )
        assert res["total"]["total.mapped"] == 1.0
        out = pm.gextract(
            "m",
            pd.DataFrame({"chrom": ["1"], "start": [100], "end": [105]}),
            iterator=5,
        )
        np.testing.assert_allclose(out["m"].to_numpy(dtype=float), np.array([1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_dense_dedup_counts(tmp_path):
    """Two identical reads with remove_dups=True -> 1 mapped, 1 dup, val=1.
    R parity: num_mapped counts raw reads (so total.mapped=2)."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path,
            ["AAAAA\t1\t100\t+\n", "AAAAA\t1\t100\t+\n"],
            pileup=5, binsize=5,
        )
        assert res["total"]["total.mapped"] == 2.0
        assert res["total"]["total.dups"] == 1.0
        out = pm.gextract(
            "m",
            pd.DataFrame({"chrom": ["1"], "start": [100], "end": [105]}),
            iterator=5,
        )
        # remove_dups=True -> the dup is dropped before pileup -> single read pileup -> 1.0
        np.testing.assert_allclose(out["m"].to_numpy(dtype=float), np.array([1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_sparse_dedup_value(tmp_path):
    """3 identical +reads, remove_dups=True -> val=1, num_dups=2."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path,
            ["AAAAA\t1\t100\t+\n", "AAAAA\t1\t100\t+\n", "AAAAA\t1\t100\t+\n"],
            pileup=0, binsize=-1, track_name="ms1",
        )
        assert res["total"]["total.mapped"] == 3.0  # raw count
        assert res["total"]["total.dups"] == 2.0
        out = pm.gextract(
            "ms1",
            pd.DataFrame({"chrom": ["1"], "start": [100], "end": [101]}),
        )
        np.testing.assert_allclose(out["ms1"].to_numpy(dtype=float), np.array([1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_sparse_keep_dups(tmp_path):
    """3 identical reads, remove_dups=False -> val=3, num_dups=2 (R parity)."""
    root = _copy_db(tmp_path)
    try:
        pm.gdb_init(str(root))
        inp = tmp_path / "m.tsv"
        inp.write_text(
            "A\t1\t100\t+\n"
            "A\t1\t100\t+\n"
            "A\t1\t100\t+\n",
        )
        track_dir = str(root / "tracks" / "ms2.track")
        res = _pymisha.pm_import_mappedseq(
            track_dir, str(inp), 0, -1, (1, 2, 3, 4), False,
        )
        _pymisha.pm_dbreload()
        assert res["total"]["total.mapped"] == 3.0
        assert res["total"]["total.dups"] == 2.0
        out = pm.gextract(
            "ms2",
            pd.DataFrame({"chrom": ["1"], "start": [100], "end": [101]}),
        )
        np.testing.assert_allclose(out["ms2"].to_numpy(dtype=float), np.array([3.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_sparse_strands_merge_same_coord(tmp_path):
    """+read at coord=100 (seq=AAAAA) maps to 100; -read at coord=95 (seq=AAAAA)
    maps to 95+5=100. Same final coord -> one merged interval. Both contribute
    to val per R lines 289-305."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path,
            ["AAAAA\t1\t100\t+\n", "AAAAA\t1\t95\t-\n"],
            pileup=0, binsize=-1, track_name="ms3",
        )
        assert res["total"]["total.mapped"] == 2.0
        out = pm.gextract(
            "ms3",
            pd.DataFrame({"chrom": ["1"], "start": [100], "end": [101]}),
        )
        # remove_dups=True: plus branch fires (minus[j]=100 >= plus[i]=100),
        # val = max(0 + 0, 1) = 1. Then minus branch fires (minus[j]==coord),
        # val = max(1 + 0, 1) = 1. Final val = 1.
        np.testing.assert_allclose(out["ms3"].to_numpy(dtype=float), np.array([1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_sparse_two_coords(tmp_path):
    """Two distinct +reads -> two sparse intervals, val=1 each."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path,
            ["AAAAA\t1\t100\t+\n", "AAAAA\t1\t200\t+\n"],
            pileup=0, binsize=-1, track_name="ms4",
        )
        assert res["total"]["total.mapped"] == 2.0
        assert res["total"]["total.dups"] == 0.0
        out = pm.gextract(
            "ms4",
            pd.DataFrame({"chrom": ["1", "1"], "start": [100, 200], "end": [101, 201]}),
        )
        np.testing.assert_allclose(out["ms4"].to_numpy(dtype=float), np.array([1.0, 1.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_dense_multibin(tmp_path):
    """Read at coord=10 with pileup=15, binsize=5 covers [10, 25] -> bins 2,3,4 fully."""
    try:
        res, root = _pileup_via_cpp(
            tmp_path, ["AAAAA\t1\t10\t+\n"], pileup=15, binsize=5,
        )
        assert res["total"]["total.mapped"] == 1.0
        out = pm.gextract(
            "m",
            pd.DataFrame({"chrom": ["1"], "start": [0], "end": [30]}),
            iterator=5,
        )
        vals = out["m"].to_numpy(dtype=float)
        # bins [0-5), [5-10), [10-15), [15-20), [20-25), [25-30)
        # pileup covers [10, 25): bins 2, 3, 4 each get 1.0
        # Bin 2 (10-15): from_coord=10, from_bin=2, to_coord=25, to_bin=4
        #   first_frac = (2+1) - 10/5 = 1.0
        #   last_frac = 25/5 - 4 = 1.0
        #   middle bin (idx 3): +1.0
        np.testing.assert_allclose(vals[2:5], np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(vals[0:2], np.array([0.0, 0.0]))
        np.testing.assert_allclose(vals[5:6], np.array([0.0]))
    finally:
        pm.gdb_init(str(TEST_DB))


def test_pm_import_mappedseq_stdin_dash(tmp_path):
    """file='-' reads from fd 0. Smoke test by redirecting stdin via
    a child Python process so we don't trample the pytest stdin."""
    root = _copy_db(tmp_path)
    sam = tmp_path / "reads.sam"
    sam.write_text("A\t1\t10\t+\nA\t1\t20\t-\n")
    track_dir = str(root / "tracks" / "stdin_test.track")
    repo_root = str(Path(__file__).resolve().parent.parent)

    code = textwrap.dedent(f"""
        import os, sys, json
        sys.path.insert(0, {repr(repo_root)})
        import pymisha as pm
        from pymisha import _pymisha
        from pathlib import Path
        pm.gdb_init({repr(str(root))})
        os.makedirs(Path({repr(track_dir)}).parent, exist_ok=True)
        res = _pymisha.pm_import_mappedseq(
            {repr(track_dir)}, "-", 0, -1, (1, 2, 3, 4), True,
        )
        print(json.dumps({{"mapped": res["total"]["total.mapped"]}}))
    """)
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        stdin=open(sam, "rb"),
    )
    res = json.loads(out.strip().splitlines()[-1])
    assert res["mapped"] == 2.0
