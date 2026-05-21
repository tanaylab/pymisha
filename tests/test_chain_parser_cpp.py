"""Direct unit tests for _pymisha.pm_parse_chain_file (G1.P2)."""

from pathlib import Path

import _pymisha
import numpy as np
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"
CHAIN_FILE = str(TEST_DB / "data" / "test.chain")


def _call(path, min_score=float("nan")):
    """Invoke the C++ parser and return the result dict (or None)."""
    return _pymisha.pm_parse_chain_file(path, float(min_score))


@pytest.fixture(autouse=True)
def _ensure_db():
    pm.gdb_init(str(TEST_DB))


class TestHappyPath:
    """Cross-validates against the canonical test.chain."""

    def test_returns_dict_with_ten_columns(self):
        out = _call(CHAIN_FILE)
        assert isinstance(out, dict)
        assert set(out.keys()) == {
            "chrom", "start", "end", "strand",
            "chromsrc", "startsrc", "endsrc", "strandsrc",
            "chain_id", "score",
        }

    def test_four_blocks_total(self):
        out = _call(CHAIN_FILE)
        assert len(out["chrom"]) == 4

    def test_dtypes(self):
        out = _call(CHAIN_FILE)
        assert out["chrom"].dtype == np.object_
        assert out["chromsrc"].dtype == np.object_
        for k in ("start", "end", "strand", "startsrc", "endsrc",
                  "strandsrc", "chain_id"):
            assert out[k].dtype == np.int64, f"{k} dtype {out[k].dtype}"
        assert out["score"].dtype == np.float64

    def test_chain1_block1_values(self):
        out = _call(CHAIN_FILE)
        assert out["chrom"][0] == "1"
        assert out["start"][0] == 12000
        assert out["end"][0] == 12500
        assert out["strand"][0] == 0
        assert out["chromsrc"][0] == "chr25"
        assert out["startsrc"][0] == 2000
        assert out["endsrc"][0] == 2500
        assert out["strandsrc"][0] == 0
        assert out["chain_id"][0] == 1
        assert out["score"][0] == 200000.0

    def test_chain2_block_values(self):
        out = _call(CHAIN_FILE)
        chrx_rows = np.where(out["chrom"] == "X")[0]
        assert len(chrx_rows) == 1
        i = chrx_rows[0]
        assert out["start"][i] == 5000
        assert out["end"][i] == 7000
        assert out["startsrc"][i] == 10000
        assert out["endsrc"][i] == 12000
        assert out["chain_id"][i] == 2
        assert out["score"][i] == 200000.0


class TestEdgeCases:
    """File-level edge cases that mirror existing test_liftover.py coverage."""

    def test_empty_file_returns_none(self, tmp_path):
        path = str(tmp_path / "empty.chain")
        Path(path).write_text("")
        assert _call(path) is None

    def test_blank_lines_only(self, tmp_path):
        path = str(tmp_path / "blanks.chain")
        Path(path).write_text("\n\n\n\n")
        assert _call(path) is None

    def test_comments_only(self, tmp_path):
        path = str(tmp_path / "comments.chain")
        Path(path).write_text("# comment 1\n# comment 2\n\n")
        assert _call(path) is None

    def test_comments_interleaved(self, tmp_path):
        path = str(tmp_path / "comm_inter.chain")
        Path(path).write_text(
            "# a leading comment\n"
            "chain 1000 chr25 100000 + 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
        )
        out = _call(path)
        assert out is not None
        assert len(out["chrom"]) == 1
        assert out["chrom"][0] == "1"
        assert out["start"][0] == 0
        assert out["end"][0] == 100

    def test_crlf_line_endings(self, tmp_path):
        path = str(tmp_path / "crlf.chain")
        Path(path).write_bytes(
            b"chain 1000 chr25 100000 + 0 100 1 500000 + 0 100 1\r\n"
            b"100\r\n\r\n"
        )
        out = _call(path)
        assert out is not None
        assert len(out["chrom"]) == 1
        assert out["start"][0] == 0
        assert out["end"][0] == 100

    def test_unknown_target_chrom_skipped(self, tmp_path):
        """Chains targeting chroms not in DB are silently dropped."""
        path = str(tmp_path / "unknown.chain")
        Path(path).write_text(
            "chain 1000 chr25 100000 + 0 100 chr_unknown 50000 + 0 100 1\n"
            "100\n\n"
            "chain 1000 chr25 100000 + 0 500 chr1 500000 + 0 500 2\n"
            "500\n\n"
        )
        out = _call(path)
        assert out is not None
        assert len(out["chrom"]) == 1
        assert out["chrom"][0] == "1"
        assert out["chain_id"][0] == 2

    def test_unknown_target_only_returns_none(self, tmp_path):
        path = str(tmp_path / "all_unknown.chain")
        Path(path).write_text(
            "chain 1000 chrA 10000 + 0 100 chr_unknown 50000 + 0 100 1\n"
            "100\n\n"
        )
        assert _call(path) is None

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _call(str(tmp_path / "no_such.chain"))


class TestMinScoreAndStrand:
    def test_min_score_drops_low(self, tmp_path):
        path = str(tmp_path / "minscore.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
            "chain 10000 chr25 100000 + 200 300 1 500000 + 200 300 2\n"
            "100\n\n"
        )
        # Filter at 500: chain 1 (score 100) drops, chain 2 (score 10000) keeps.
        out = _call(path, min_score=500.0)
        assert out is not None
        assert len(out["chrom"]) == 1
        assert out["chain_id"][0] == 2
        assert out["score"][0] == 10000.0

    def test_min_score_zero_keeps_all(self, tmp_path):
        path = str(tmp_path / "ms0.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
        )
        out = _call(path, min_score=0.0)
        assert out is not None
        assert len(out["chrom"]) == 1

    def test_min_score_nan_disables(self, tmp_path):
        """NaN min_score must not filter anything (Python passes None -> NaN)."""
        path = str(tmp_path / "msnan.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
        )
        out = _call(path, min_score=float("nan"))
        assert out is not None
        assert len(out["chrom"]) == 1

    def test_negative_target_strand(self, tmp_path):
        """- strand on target flips coords: start = size - pos - block_size."""
        path = str(tmp_path / "negstrand.chain")
        Path(path).write_text(
            "chain 1000 chr25 100000 + 0 100 1 500000 - 0 100 1\n"
            "100\n\n"
        )
        out = _call(path)
        assert out is not None
        assert len(out["chrom"]) == 1
        # Negative strand: start = 500000 - 0 - 100 = 499900, end = 500000 - 0 = 500000
        assert out["start"][0] == 499900
        assert out["end"][0] == 500000
        assert out["strand"][0] == 1

    def test_negative_source_strand(self, tmp_path):
        """- strand on source flips source coords."""
        path = str(tmp_path / "negsrc.chain")
        Path(path).write_text(
            "chain 1000 chr25 100000 - 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
        )
        out = _call(path)
        assert out is not None
        # src_strand=-: block_src_start = 100000 - 0 - 100 = 99900, end = 100000
        assert out["startsrc"][0] == 99900
        assert out["endsrc"][0] == 100000
        assert out["strandsrc"][0] == 1

    def test_multi_block_with_gaps(self, tmp_path):
        """3-block chain: verify cursors advance with dt/dq."""
        path = str(tmp_path / "multi.chain")
        # src: 0..500 then 700..1100 (dt=200) then 1500..2000 (dt=400)
        # tgt: 0..500 then 800..1200 (dq=300) then 1500..2000 (dq=300)
        Path(path).write_text(
            "chain 1000 chr25 100000 + 0 2000 1 500000 + 0 2000 1\n"
            "500\t200\t300\n"
            "400\t400\t300\n"
            "500\n\n"
        )
        out = _call(path)
        assert out is not None
        assert len(out["chrom"]) == 3
        assert list(out["start"])     == [0, 800, 1500]
        assert list(out["end"])       == [500, 1200, 2000]
        assert list(out["startsrc"])  == [0, 700, 1500]
        assert list(out["endsrc"])    == [500, 1100, 2000]


class TestMalformedInput:
    def test_bad_header_field_count(self, tmp_path):
        path = str(tmp_path / "bad_hdr.chain")
        Path(path).write_text("chain 100 chr25 +\n\n")
        with pytest.raises(ValueError, match="13 fields"):
            _call(path)

    def test_bad_source_size_zero(self, tmp_path):
        path = str(tmp_path / "bad_srcsize.chain")
        Path(path).write_text(
            "chain 100 chr25 0 + 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
        )
        with pytest.raises(ValueError, match="source chrom size"):
            _call(path)

    def test_inconsistent_source_size(self, tmp_path):
        path = str(tmp_path / "incons_src.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
            "chain 100 chr25 200000 + 0 100 1 500000 + 200 300 2\n"
            "100\n\n"
        )
        with pytest.raises(ValueError, match="differs from previous"):
            _call(path)

    def test_bad_source_strand(self, tmp_path):
        path = str(tmp_path / "bad_strand.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 ? 0 100 1 500000 + 0 100 1\n"
            "100\n\n"
        )
        with pytest.raises(ValueError, match="source strand"):
            _call(path)

    def test_target_size_mismatch_db(self, tmp_path):
        """Target chrom exists in DB but the declared size doesn't match."""
        path = str(tmp_path / "tgtsize.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 999999 + 0 100 1\n"
            "100\n\n"
        )
        with pytest.raises(ValueError, match="target chrom size"):
            _call(path)

    def test_target_start_out_of_range(self, tmp_path):
        path = str(tmp_path / "tgtstart.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 500000 + 999999 1000000 1\n"
            "100\n\n"
        )
        with pytest.raises(ValueError, match="target start"):
            _call(path)

    def test_block_outside_chain(self, tmp_path):
        path = str(tmp_path / "no_chain.chain")
        Path(path).write_text("500\n100\n")
        with pytest.raises(ValueError, match="outside chain"):
            _call(path)

    def test_bad_block_size_zero(self, tmp_path):
        path = str(tmp_path / "blk0.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 100 1 500000 + 0 100 1\n"
            "0\n\n"
        )
        with pytest.raises(ValueError, match="block size"):
            _call(path)

    def test_negative_dt_rejected(self, tmp_path):
        path = str(tmp_path / "neg_dt.chain")
        Path(path).write_text(
            "chain 100 chr25 100000 + 0 200 1 500000 + 0 200 1\n"
            "50\t-10\t0\n"
            "150\n\n"
        )
        with pytest.raises(ValueError, match="dt"):
            _call(path)


class TestCrossValidatePython:
    """C++ output must match the pure-Python _parse_chain_file row-for-row."""

    @staticmethod
    def _python_parse(path):
        """Direct call to the Python reference."""
        from pymisha.liftover import _get_db_chrom_sizes, _parse_chain_file
        sizes = _get_db_chrom_sizes()
        return _parse_chain_file(path, sizes, min_score=None,
                                 _force_pure_python=True)

    @staticmethod
    def _assert_same(cpp_out, py_out):
        if cpp_out is None or py_out is None:
            assert cpp_out is None and (py_out is None or len(py_out["chrom"]) == 0)
            return
        keys = {"chrom", "start", "end", "strand",
                "chromsrc", "startsrc", "endsrc", "strandsrc",
                "chain_id", "score"}
        assert set(cpp_out.keys()) == keys
        assert set(py_out.keys()) == keys
        n = len(cpp_out["chrom"])
        assert len(py_out["chrom"]) == n
        for k in keys:
            for i in range(n):
                a = cpp_out[k][i]
                b = py_out[k][i]
                if isinstance(a, np.generic):
                    a = a.item()
                assert a == b, f"col={k} row={i}: cpp={a!r} py={b!r}"

    def test_xval_test_chain_file(self):
        self._assert_same(_call(CHAIN_FILE), self._python_parse(CHAIN_FILE))

    @pytest.mark.parametrize("score,nblocks", [
        (1000, 3),
        (5000, 5),
        (10000, 1),
    ])
    def test_xval_synth_multi_chain(self, tmp_path, score, nblocks):
        body = []
        for cid in range(1, 4):
            body.append(
                f"chain {score + cid * 10} chr25 100000 + "
                f"{cid * 1000} {cid * 1000 + nblocks * 100} "
                f"1 500000 + {cid * 1000} {cid * 1000 + nblocks * 100} {cid}\n"
            )
            for b in range(nblocks - 1):
                body.append(f"{100}\t0\t0\n")
            body.append(f"{100}\n\n")
        path = str(tmp_path / "synth.chain")
        Path(path).write_text("".join(body))
        self._assert_same(_call(path), self._python_parse(path))

    def test_xval_negative_strand_chains(self, tmp_path):
        path = str(tmp_path / "negstrand.chain")
        Path(path).write_text(
            "chain 1000 chr25 100000 - 0 200 1 500000 + 0 200 1\n"
            "200\n\n"
            "chain 2000 chr25 100000 + 5000 5300 1 500000 - 5000 5300 2\n"
            "100\t50\t50\n"
            "150\n\n"
        )
        self._assert_same(_call(path), self._python_parse(path))

    def test_xval_empty(self, tmp_path):
        path = str(tmp_path / "empty.chain")
        Path(path).write_text("")
        self._assert_same(_call(path), self._python_parse(path))
