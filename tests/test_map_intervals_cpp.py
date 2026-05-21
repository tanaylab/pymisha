"""Tests for the C++ pm_map_intervals fast path (G1.P3.B.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import _pymisha
from pymisha import liftover as _liftover_mod


_EMPTY_CHAIN_COLS = list(_liftover_mod._EMPTY_CHAIN_COLS)


def _empty_chain_dict() -> dict:
    return {
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


def _empty_src_dict() -> dict:
    return {
        "chrom": np.array([], dtype=object),
        "start": np.array([], dtype=np.int64),
        "end": np.array([], dtype=np.int64),
    }


class TestArgValidation:
    def test_src_not_dict_raises(self):
        with pytest.raises(TypeError, match="src_df_dict must be a dict"):
            _pymisha.pm_map_intervals(
                "not a dict", _empty_chain_dict(), "", False, ""
            )

    def test_chain_not_dict_raises(self):
        with pytest.raises(TypeError, match="chain_df_dict must be a dict"):
            _pymisha.pm_map_intervals(
                _empty_src_dict(), "not a dict", "", False, ""
            )

    def test_bad_cluster_strategy_raises(self):
        with pytest.raises(ValueError, match="cluster_strategy"):
            _pymisha.pm_map_intervals(
                _empty_src_dict(), _empty_chain_dict(), "", False, "bogus"
            )

    def test_missing_src_chrom_key_raises(self):
        bad_src = {"start": np.array([0], dtype=np.int64),
                   "end": np.array([10], dtype=np.int64)}
        with pytest.raises(ValueError, match="missing required key 'chrom'"):
            _pymisha.pm_map_intervals(
                bad_src, _empty_chain_dict(), "", False, ""
            )

    def test_missing_chain_col_raises(self):
        bad_chain = dict(_empty_chain_dict())
        del bad_chain["score"]
        with pytest.raises(ValueError, match="missing required key 'score'"):
            _pymisha.pm_map_intervals(
                _empty_src_dict(), bad_chain, "", False, ""
            )

    def test_src_col_length_mismatch_raises(self):
        bad_src = {
            "chrom": np.array(["a", "b"], dtype=object),
            "start": np.array([0], dtype=np.int64),
            "end": np.array([10], dtype=np.int64),
        }
        with pytest.raises(ValueError, match="mismatched lengths"):
            _pymisha.pm_map_intervals(
                bad_src, _empty_chain_dict(), "", False, ""
            )

    def test_empty_src_returns_empty_result(self):
        out = _pymisha.pm_map_intervals(
            _empty_src_dict(), _empty_chain_dict(), "", False, ""
        )
        assert isinstance(out, dict)
        for k in ("chrom", "start", "end", "intervalID", "chain_id",
                  "__src_start", "__src_end"):
            assert k in out, f"missing key {k}"
            assert len(out[k]) == 0

    def test_score_present_when_include_metadata(self):
        out = _pymisha.pm_map_intervals(
            _empty_src_dict(), _empty_chain_dict(), "", True, ""
        )
        assert "score" in out
        assert len(out["score"]) == 0

    def test_score_absent_when_not_include_metadata(self):
        out = _pymisha.pm_map_intervals(
            _empty_src_dict(), _empty_chain_dict(), "", False, ""
        )
        assert "score" not in out

    def test_value_col_present_when_set(self):
        src = dict(_empty_src_dict())
        src["myval"] = np.array([], dtype=np.float64)
        out = _pymisha.pm_map_intervals(
            src, _empty_chain_dict(), "myval", False, ""
        )
        assert "myval" in out
        assert len(out["myval"]) == 0

    def test_value_col_missing_in_src_raises(self):
        with pytest.raises(ValueError, match="value_col 'myval' not found"):
            _pymisha.pm_map_intervals(
                _empty_src_dict(), _empty_chain_dict(), "myval", False, ""
            )


class TestSimplestMapping:
    """Smallest possible inputs that exercise build_src_aux + map_one_src_interval."""

    def _chain1(self, chromsrc="src", chrom="chr1", strand=0,
                src_start=0, src_end=100, tgt_start=0, tgt_end=100,
                chain_id=1, score=100.0) -> dict:
        return {
            "chrom":     np.array([chrom],     dtype=object),
            "start":     np.array([tgt_start], dtype=np.int64),
            "end":       np.array([tgt_end],   dtype=np.int64),
            "strand":    np.array([strand],    dtype=np.int64),
            "chromsrc":  np.array([chromsrc],  dtype=object),
            "startsrc":  np.array([src_start], dtype=np.int64),
            "endsrc":    np.array([src_end],   dtype=np.int64),
            "strandsrc": np.array([0],         dtype=np.int64),
            "chain_id":  np.array([chain_id],  dtype=np.int64),
            "score":     np.array([score],     dtype=np.float64),
        }

    def test_single_src_inside_single_chain(self):
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([10],    dtype=np.int64),
            "end":   np.array([50],    dtype=np.int64),
        }
        chain = self._chain1()
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        assert len(out["chrom"]) == 1
        assert out["chrom"][0] == "chr1"
        assert out["start"][0] == 10
        assert out["end"][0] == 50
        assert out["intervalID"][0] == 0
        assert out["chain_id"][0] == 1
        assert out["__src_start"][0] == 10
        assert out["__src_end"][0] == 50

    def test_negative_strand_mapping(self):
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([10],    dtype=np.int64),
            "end":   np.array([30],    dtype=np.int64),
        }
        # src[0,100) maps reverse-complement onto tgt[1000,1100).
        # src[10,30) → tgt[1100 - 30, 1100 - 10) = tgt[1070, 1090).
        chain = self._chain1(strand=1, tgt_start=1000, tgt_end=1100)
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        assert len(out["chrom"]) == 1
        assert out["start"][0] == 1070
        assert out["end"][0] == 1090

    def test_no_overlap_returns_empty(self):
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([200], dtype=np.int64),
            "end":   np.array([300], dtype=np.int64),
        }
        chain = self._chain1()  # chain covers src[0,100)
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        assert len(out["chrom"]) == 0

    def test_src_chrom_not_in_chain(self):
        src = {
            "chrom": np.array(["other"], dtype=object),
            "start": np.array([0], dtype=np.int64),
            "end":   np.array([50], dtype=np.int64),
        }
        chain = self._chain1()
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        assert len(out["chrom"]) == 0

    def test_zero_length_src_returns_empty(self):
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([10], dtype=np.int64),
            "end":   np.array([10], dtype=np.int64),
        }
        chain = self._chain1()
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        assert len(out["chrom"]) == 0


class TestMultiChain:
    def _chain_n(self, rows: list[dict]) -> dict:
        """rows: list of dicts with chromsrc, chrom, strand, src_start, src_end,
        tgt_start, tgt_end, chain_id, [score]."""
        cols = {c: [] for c in _EMPTY_CHAIN_COLS}
        for r in rows:
            cols["chrom"].append(r["chrom"])
            cols["start"].append(r["tgt_start"])
            cols["end"].append(r["tgt_end"])
            cols["strand"].append(r.get("strand", 0))
            cols["chromsrc"].append(r["chromsrc"])
            cols["startsrc"].append(r["src_start"])
            cols["endsrc"].append(r["src_end"])
            cols["strandsrc"].append(0)
            cols["chain_id"].append(r["chain_id"])
            cols["score"].append(r.get("score", 100.0))
        return {
            "chrom":     np.array(cols["chrom"],     dtype=object),
            "start":     np.array(cols["start"],     dtype=np.int64),
            "end":       np.array(cols["end"],       dtype=np.int64),
            "strand":    np.array(cols["strand"],    dtype=np.int64),
            "chromsrc":  np.array(cols["chromsrc"],  dtype=object),
            "startsrc":  np.array(cols["startsrc"],  dtype=np.int64),
            "endsrc":    np.array(cols["endsrc"],    dtype=np.int64),
            "strandsrc": np.array(cols["strandsrc"], dtype=np.int64),
            "chain_id":  np.array(cols["chain_id"],  dtype=np.int64),
            "score":     np.array(cols["score"],     dtype=np.float64),
        }

    def test_src_spans_two_adjacent_chains(self):
        chain = self._chain_n([
            {"chromsrc": "src", "chrom": "chr1",
             "src_start": 0, "src_end": 100, "tgt_start": 0, "tgt_end": 100,
             "chain_id": 1},
            {"chromsrc": "src", "chrom": "chr1",
             "src_start": 100, "src_end": 200, "tgt_start": 1000, "tgt_end": 1100,
             "chain_id": 2},
        ])
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([50],    dtype=np.int64),
            "end":   np.array([150],   dtype=np.int64),
        }
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        # Two rows (one per chain).
        assert len(out["chrom"]) == 2
        # Map into a dict keyed by chain_id for deterministic assertions.
        rows = {int(cid): (s, e) for cid, s, e in
                zip(out["chain_id"], out["start"], out["end"])}
        assert rows[1] == (50, 100)
        assert rows[2] == (1000, 1050)

    def test_overlapping_source_chains_emit_both(self):
        """Two chains with overlapping source ranges both produce mappings."""
        chain = self._chain_n([
            {"chromsrc": "src", "chrom": "chr1",
             "src_start": 0, "src_end": 100, "tgt_start": 0, "tgt_end": 100,
             "chain_id": 1},
            {"chromsrc": "src", "chrom": "chr2",
             "src_start": 0, "src_end": 100, "tgt_start": 500, "tgt_end": 600,
             "chain_id": 2},
        ])
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([10],    dtype=np.int64),
            "end":   np.array([50],    dtype=np.int64),
        }
        out = _pymisha.pm_map_intervals(src, chain, "", False, "")
        assert len(out["chrom"]) == 2
        chroms = set(out["chrom"].tolist())
        assert chroms == {"chr1", "chr2"}

    def test_include_metadata_adds_score(self):
        chain = self._chain_n([
            {"chromsrc": "src", "chrom": "chr1",
             "src_start": 0, "src_end": 100, "tgt_start": 0, "tgt_end": 100,
             "chain_id": 1, "score": 42.5},
        ])
        src = {
            "chrom": np.array(["src"], dtype=object),
            "start": np.array([10],    dtype=np.int64),
            "end":   np.array([50],    dtype=np.int64),
        }
        out = _pymisha.pm_map_intervals(src, chain, "", True, "")
        assert "score" in out
        assert out["score"][0] == pytest.approx(42.5)

    def test_value_col_carries_through(self):
        chain = self._chain_n([
            {"chromsrc": "src", "chrom": "chr1",
             "src_start": 0, "src_end": 100, "tgt_start": 0, "tgt_end": 100,
             "chain_id": 1},
        ])
        src = {
            "chrom":     np.array(["src", "src"], dtype=object),
            "start":     np.array([10, 60],       dtype=np.int64),
            "end":       np.array([50, 80],       dtype=np.int64),
            "weight":    np.array([0.5, 1.5],     dtype=np.float64),
        }
        out = _pymisha.pm_map_intervals(src, chain, "weight", False, "")
        assert "weight" in out
        # Rows order is implementation-defined; sort by intervalID for assertion.
        order = np.argsort(out["intervalID"])
        weights = out["weight"][order]
        assert weights.tolist() == [0.5, 1.5]


class TestCrossValidatePython:
    """Cross-validate pm_map_intervals output against _map_intervals_vectorized_python.

    Set semantics: rows may be emitted in any order. Compare as a sorted DataFrame.
    """

    def _xval(self, intervals_df, chain_df, include_metadata=False, value_col=None):
        py_out = _liftover_mod._map_intervals_vectorized_python(
            intervals_df, chain_df, include_metadata, value_col,
        )
        cpp_out = _liftover_mod._map_intervals_vectorized(
            intervals_df, chain_df, include_metadata, value_col,
        )
        # Normalize: sort by every column, reset index.
        sort_cols = [c for c in py_out.columns if c not in ("score",)]
        py_sorted = py_out.sort_values(sort_cols).reset_index(drop=True)
        cpp_sorted = cpp_out.sort_values(sort_cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            py_sorted, cpp_sorted, check_dtype=False, check_like=True,
        )

    def test_xval_single_chain_single_src(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1"], "start": [0],   "end": [100],
            "strand":    [0],
            "chromsrc":  ["src"],  "startsrc": [0], "endsrc": [100],
            "strandsrc": [0], "chain_id": [1], "score": [100.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src"], "start": [10], "end": [60],
        })
        self._xval(intervals, chain)

    def test_xval_negative_strand(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1"], "start": [1000], "end": [1100],
            "strand":    [1],
            "chromsrc":  ["src"],  "startsrc": [0], "endsrc": [100],
            "strandsrc": [0], "chain_id": [1], "score": [100.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src", "src"], "start": [10, 30], "end": [50, 70],
        })
        self._xval(intervals, chain)

    def test_xval_multiple_chains_per_src(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr2", "chr1"],
            "start":     [0,      500,    300],
            "end":       [100,    600,    400],
            "strand":    [0, 0, 0],
            "chromsrc":  ["src", "src", "src"],
            "startsrc":  [0,     50,    200],
            "endsrc":    [100,   150,   300],
            "strandsrc": [0, 0, 0],
            "chain_id":  [1, 2, 3],
            "score":     [100.0, 90.0, 80.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src"], "start": [0], "end": [400],
        })
        self._xval(intervals, chain)

    def test_xval_with_score(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1"], "start": [0],    "end": [100],
            "strand":    [0],
            "chromsrc":  ["src"],  "startsrc": [0], "endsrc": [100],
            "strandsrc": [0], "chain_id": [1], "score": [123.5],
        })
        intervals = pd.DataFrame({
            "chrom": ["src"], "start": [10], "end": [60],
        })
        self._xval(intervals, chain, include_metadata=True)

    def test_xval_with_value_col(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1"], "start": [0],    "end": [100],
            "strand":    [0],
            "chromsrc":  ["src"],  "startsrc": [0], "endsrc": [100],
            "strandsrc": [0], "chain_id": [1], "score": [100.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src", "src"], "start": [10, 30], "end": [50, 70],
            "weight": [0.5, 1.5],
        })
        self._xval(intervals, chain, value_col="weight")

    def test_xval_multi_chrom_intervals(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr2"],
            "start":     [0, 0],
            "end":       [100, 200],
            "strand":    [0, 0],
            "chromsrc":  ["srcA", "srcB"],
            "startsrc":  [0, 0],
            "endsrc":    [100, 200],
            "strandsrc": [0, 0],
            "chain_id":  [1, 2],
            "score":     [100.0, 200.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["srcA", "srcB", "srcA"],
            "start": [10, 50, 80],
            "end":   [50, 150, 95],
        })
        self._xval(intervals, chain)

    def test_xval_real_load_chain(self, tmp_path):
        """Use a real loaded chain to exercise resolve + map together."""
        # Minimal UCSC chain file with two chains on the same src.
        chain_text = (
            "chain 1000 chr1_tgt 200000 + 0 100 src1 1000 + 0 100 1\n"
            "100\n\n"
            "chain 800 chr2_tgt 300000 + 0 200 src1 1000 + 50 250 2\n"
            "200\n\n"
        )
        chain_path = tmp_path / "two.chain"
        chain_path.write_text(chain_text)
        # We cannot call gintervals_load_chain without a DB; build manually.
        chain = pd.DataFrame({
            "chrom":     ["chr1_tgt", "chr2_tgt"],
            "start":     [0, 0],
            "end":       [100, 200],
            "strand":    [0, 0],
            "chromsrc":  ["src1", "src1"],
            "startsrc":  [0, 50],
            "endsrc":    [100, 250],
            "strandsrc": [0, 0],
            "chain_id":  [1, 2],
            "score":     [1000.0, 800.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src1", "src1"], "start": [0, 100], "end": [80, 200],
        })
        self._xval(intervals, chain)

    def test_xval_force_py_env_var(self, monkeypatch):
        """Setting the env var falls back to the Python implementation."""
        chain = pd.DataFrame({
            "chrom":     ["chr1"], "start": [0],    "end": [100],
            "strand":    [0],
            "chromsrc":  ["src"],  "startsrc": [0], "endsrc": [100],
            "strandsrc": [0], "chain_id": [1], "score": [100.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src"], "start": [10], "end": [50],
        })
        monkeypatch.setenv("PYMISHA_FORCE_PY_MAP_INTERVALS", "1")
        out_py = _liftover_mod._map_intervals_vectorized(
            intervals, chain, False, None,
        )
        assert len(out_py) == 1
        assert out_py.iloc[0]["start"] == 10
        assert out_py.iloc[0]["end"] == 50


class TestClusterRParity:
    """Verify R-parity for _resolve_cluster_policy.

    R unions candidates by chain_id then by source-overlap; pymisha pre-fix
    Python only unions by source-overlap. This test exposes the divergence.
    """

    def test_chain_with_gap_clusters_as_one(self):
        """A single chain split by a gap produces two candidates with the same
        chain_id. R unions them via the chain_id step; Python (pre-fix) does not."""
        # Chain row A: src[0,50) -> chr1[1000,1050), chain_id=1
        # Chain row B: src[100,150) -> chr1[1100,1150), chain_id=1  (gap in chain)
        # Chain row C: src[60,90) -> chr2[2000,2030), chain_id=2  (disjoint, smaller mass)
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr1", "chr2"],
            "start":     [1000, 1100, 2000],
            "end":       [1050, 1150, 2030],
            "strand":    [0, 0, 0],
            "chromsrc":  ["src", "src", "src"],
            "startsrc":  [0, 100, 60],
            "endsrc":    [50, 150, 90],
            "strandsrc": [0, 0, 0],
            "chain_id":  [1, 1, 2],
            "score":     [1000.0, 1000.0, 1000.0],
        })
        intervals = pd.DataFrame({
            "chrom": ["src"], "start": [0], "end": [200],
        })

        result_py = _liftover_mod._map_intervals_vectorized(
            intervals, chain, False, None,
        )
        from pymisha.liftover import _resolve_cluster_policy
        resolved = _resolve_cluster_policy(result_py, "best_cluster_union")

        # R behavior: chain_id 1 has mass 50+50=100, chain_id 2 has mass 30.
        # chain_id 1 wins. Both chain_id=1 rows survive.
        # Python (pre-fix) behavior: clusters split by source-only union.
        # src=[0,50) and src=[100,150) are non-overlapping → two clusters of
        # mass 50 each. src=[60,90) is its own cluster of mass 30. The first
        # cluster (mass 50, lowest start) wins → only ONE row survives
        # (the src=[0,50) one).
        chain_ids = sorted(resolved["chain_id"].tolist())
        assert chain_ids == [1, 1], (
            f"Expected both chain_id=1 rows to survive, got {chain_ids}. "
            f"Python may have a chain_id-union divergence vs R."
        )


class TestClusterStrategies:
    """C++ cluster-resolution output matches the (R-parity-corrected) Python."""

    def _run(self, intervals, chain, strategy):
        return _pymisha.pm_map_intervals(
            {
                "chrom": intervals["chrom"].to_numpy(),
                "start": intervals["start"].to_numpy(dtype=np.int64),
                "end":   intervals["end"].to_numpy(dtype=np.int64),
            },
            {c: chain[c].to_numpy() for c in _EMPTY_CHAIN_COLS},
            "",
            False,
            strategy,
        )

    def _xval(self, intervals, chain, strategy):
        from pymisha.liftover import _resolve_cluster_policy

        full_strategy = {
            "union": "best_cluster_union",
            "sum":   "best_cluster_sum",
            "max":   "best_cluster_max",
        }[strategy]
        py_out = _liftover_mod._map_intervals_vectorized_python(
            intervals, chain, False, None,
        )
        py_resolved = _resolve_cluster_policy(py_out, full_strategy)
        cpp_resolved = pd.DataFrame(self._run(intervals, chain, strategy))

        # Drop helper cols for comparison
        cols = ["chrom", "start", "end", "intervalID", "chain_id"]
        py_sorted = py_resolved[cols].sort_values(cols).reset_index(drop=True)
        cpp_sorted = cpp_resolved[cols].sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(py_sorted, cpp_sorted, check_dtype=False)

    def test_xval_union_simple_disjoint(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr1"],
            "start":     [0, 200],
            "end":       [50, 300],
            "strand":    [0, 0],
            "chromsrc":  ["src", "src"],
            "startsrc":  [0, 100],
            "endsrc":    [50, 200],
            "strandsrc": [0, 0],
            "chain_id":  [1, 2],
            "score":     [1000.0, 500.0],
        })
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [200]})
        self._xval(intervals, chain, "union")

    def test_xval_sum_strategy(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr1", "chr2"],
            "start":     [0, 100, 500],
            "end":       [100, 200, 600],
            "strand":    [0, 0, 0],
            "chromsrc":  ["src", "src", "src"],
            "startsrc":  [0, 50, 300],
            "endsrc":    [100, 150, 400],
            "strandsrc": [0, 0, 0],
            "chain_id":  [1, 2, 3],
            "score":     [100.0, 90.0, 80.0],
        })
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [400]})
        self._xval(intervals, chain, "sum")

    def test_xval_max_strategy(self):
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr1", "chr2"],
            "start":     [0, 100, 500],
            "end":       [80,  120, 700],
            "strand":    [0, 0, 0],
            "chromsrc":  ["src", "src", "src"],
            "startsrc":  [0, 100, 300],
            "endsrc":    [80, 120, 500],
            "strandsrc": [0, 0, 0],
            "chain_id":  [1, 2, 3],
            "score":     [100.0, 90.0, 80.0],
        })
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [500]})
        self._xval(intervals, chain, "max")

    def test_xval_chain_with_gap(self):
        """Same chain_id, two non-overlapping src segments → one cluster."""
        chain = pd.DataFrame({
            "chrom":     ["chr1", "chr1", "chr2"],
            "start":     [1000, 1100, 2000],
            "end":       [1050, 1150, 2030],
            "strand":    [0, 0, 0],
            "chromsrc":  ["src", "src", "src"],
            "startsrc":  [0, 100, 60],
            "endsrc":    [50, 150, 90],
            "strandsrc": [0, 0, 0],
            "chain_id":  [1, 1, 2],
            "score":     [1000.0, 1000.0, 1000.0],
        })
        intervals = pd.DataFrame({"chrom": ["src"], "start": [0], "end": [200]})
        self._xval(intervals, chain, "union")


class TestXvalEveryFixture:
    """Cross-validate C++ vs Python on representative fixtures.

    Lighter-weight than running the full test_liftover.py through both paths;
    picks fixtures that exercise different code paths.
    """

    @pytest.fixture(scope="module")
    def hg38_chain(self):
        """Optional fixture: load the hg38 chain if available, else skip."""
        import os
        chain_path = os.path.expanduser("~/hg38/hg19ToHg38.over.chain")
        if not os.path.exists(chain_path):
            pytest.skip(f"hg38 chain not at {chain_path}")
        return chain_path

    @pytest.mark.parametrize("n_src", [10, 100, 1000])
    def test_xval_random_synthetic(self, n_src):
        """Generate n_src random intervals over a 100-row synthetic chain."""
        rng = np.random.default_rng(60427 + n_src)
        chain = pd.DataFrame({
            "chrom":     ["chr1"] * 100,
            "start":     np.arange(0, 100*200, 200, dtype=np.int64),
            "end":       np.arange(100, 100*200 + 100, 200, dtype=np.int64),
            "strand":    np.zeros(100, dtype=np.int64),
            "chromsrc":  ["src"] * 100,
            "startsrc":  np.arange(0, 100*150, 150, dtype=np.int64),
            "endsrc":    np.arange(100, 100*150 + 100, 150, dtype=np.int64),
            "strandsrc": np.zeros(100, dtype=np.int64),
            "chain_id":  np.arange(100, dtype=np.int64) + 1,
            "score":     rng.uniform(100, 1000, size=100).astype(np.float64),
        })
        intervals = pd.DataFrame({
            "chrom": ["src"] * n_src,
            "start": rng.integers(0, 100*150, size=n_src).astype(np.int64),
            "end":   np.zeros(n_src, dtype=np.int64),
        })
        intervals["end"] = intervals["start"] + rng.integers(50, 500, size=n_src).astype(np.int64)

        py_out = _liftover_mod._map_intervals_vectorized_python(intervals, chain, False, None)
        cpp_out = _liftover_mod._map_intervals_vectorized(intervals, chain, False, None)

        cols = ["chrom", "start", "end", "intervalID", "chain_id"]
        py_sorted = py_out[cols].sort_values(cols).reset_index(drop=True)
        cpp_sorted = cpp_out[cols].sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(py_sorted, cpp_sorted, check_dtype=False)
