"""Parity with R misha 5.12.1: gtrack_liftover of 2D tracks and of indexed source tracks.

Ports of the 5.12.1 cases in R's test-gtrack.liftover-2d-points.R and
test-gtrack.liftover.R.
"""

import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _mk_db(root, chroms, size, db_format="per-chromosome"):
    """Create a database of `size`-bp chromosomes under root, and open it."""
    root.mkdir(parents=True, exist_ok=True)
    if db_format == "indexed":
        fasta = root / "genome.fasta"
        fasta.write_text("".join(f">{c}\n{'A' * size}\n" for c in chroms))
        fastas = [str(fasta)]
    else:
        fastas = []
        for c in chroms:
            fasta = root / f"{c}.fasta"
            fasta.write_text(f">{c}\n{'A' * size}\n")
            fastas.append(str(fasta))
    db = root / "db"
    pm.gdb_create(str(db), fastas, db_format=db_format)
    pm.gdb_init(str(db))
    return db


def _import_points(tmp_path, track, chrom1, start1, chrom2, start2, v):
    """Import 1bp contacts as a POINTS track in the current database."""
    f = tmp_path / f"{track}.tsv"
    pd.DataFrame({
        "chrom1": chrom1, "start1": start1, "end1": np.asarray(start1) + 1,
        "chrom2": chrom2, "start2": start2, "end2": np.asarray(start2) + 1,
        "value": v,
    }).to_csv(f, sep="\t", index=False)
    pm.gtrack_2d_import_contacts(track, "x", str(f))


def _extract_points(track):
    cols = ["chrom1", "start1", "chrom2", "start2", "v"]
    x = pm.gextract(track, pm.gintervals_2d_all(mode="full"))
    if x is None:
        return pd.DataFrame(columns=cols)
    assert ((x["end1"] - x["start1"] == 1) & (x["end2"] - x["start2"] == 1)).all()
    out = pd.DataFrame({
        "chrom1": x["chrom1"].astype(str), "start1": x["start1"].astype(np.int64),
        "chrom2": x["chrom2"].astype(str), "start2": x["start2"].astype(np.int64),
        "v": x[track].astype(float),
    })
    return out.sort_values(cols).reset_index(drop=True)


def _extract_rects(track):
    x = pm.gextract(track, pm.gintervals_2d_all(mode="full"))
    cols = ["start1", "end1", "start2", "end2"]
    return sorted(
        (*(int(c) for c in row[:4]), float(row[4]))
        for row in x[[*cols, track]].itertuples(index=False)
    )


def _chain(path, entries):
    """Write a chain file of one block per chain. Each entry is (src, src_size, src_strand,
    src_start, src_end, tgt, tgt_size, tgt_strand, tgt_start, tgt_end, chain_id)."""
    with open(path, "w") as f:
        for (s, ss, sst, s0, s1, t, ts, tst, t0, t1, cid) in entries:
            f.write(f"chain 1000 {s} {ss} {sst} {s0} {s1} {t} {ts} {tst} {t0} {t1} {cid}\n"
                    f"{s1 - s0}\n\n")
    return str(path)


# chrS1 [0,100) and [500,600) both land on chrT1 [0,100), kept by tgt_overlap_policy="keep"
_TWO_BLOCKS_ONTO_ONE = [
    ("chrS1", 1000, "+", 0, 100, "chrT1", 1000, "+", 0, 100, 1),
    ("chrS1", 1000, "+", 500, 600, "chrT1", 1000, "+", 0, 100, 2),
]


# source chrS1 (1000bp): [0,300) -> chrT1 [100,400) +, [300,400) unmapped,
# [400,700) -> chrT1 [1500,1800) -, [700,1000) -> chrT2 [0,300) +;
# chrS2 (1000bp): [0,1000) -> chrT2 [1000,2000) +.
# The "-" block is given on the target's reverse strand: [1500,1800) is [200,500) there.
_POINTS_CHAIN = [
    ("chrS1", 1000, "+", 0, 300, "chrT1", 2000, "+", 100, 400, 1),
    ("chrS1", 1000, "+", 400, 700, "chrT1", 2000, "-", 200, 500, 2),
    ("chrS1", 1000, "+", 700, 1000, "chrT2", 2000, "+", 0, 300, 3),
    ("chrS2", 1000, "+", 0, 1000, "chrT2", 2000, "+", 1000, 2000, 4),
]


class TestLiftover2DPoints:
    """A POINTS source gives a POINTS track, each point lifted by both ends as a 1bp
    interval; a point with an unmapped end is dropped, and points landing on one target
    point are merged by multi_target_agg."""

    def test_lifted_end_by_end(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1", "chrS2"], 1000)
        rng = np.random.default_rng(17)
        n = 300
        chroms = np.array(["chrS1", "chrS2"])
        _import_points(tmp_path, "src",
                       chroms[(rng.random(n) < 0.25).astype(int)], rng.integers(0, 1000, n),
                       chroms[(rng.random(n) < 0.25).astype(int)], rng.integers(0, 1000, n),
                       rng.integers(1, 10, n))
        assert pm.gtrack_info("src")["type"] == "points"
        src = _extract_points("src")

        # an indexed target: the lifted track is converted to indexed
        tgt_db = _mk_db(tmp_path / "tgt", ["chrT1", "chrT2"], 2000, db_format="indexed")
        chain = _chain(tmp_path / "x.chain", _POINTS_CHAIN)
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"), chain,
                           multi_target_agg="sum")
        assert pm.gtrack_info("lifted")["type"] == "points"
        assert (tgt_db / "tracks" / "lifted.track" / "track.idx").exists()
        got = _extract_points("lifted")

        # expected: both ends lifted as 1bp intervals; the chain is 1:1, so nothing collides
        def lift_pos(chrom, start):
            lifted = pm.gintervals_liftover(
                pd.DataFrame({"chrom": chrom, "start": start, "end": start + 1}), chain)
            out_chrom = np.full(len(chrom), None, dtype=object)
            out_start = np.full(len(chrom), -1, dtype=np.int64)
            row = lifted["intervalID"].to_numpy() - 1  # 1-based, as R
            out_chrom[row] = lifted["chrom"].astype(str).to_numpy()
            out_start[row] = lifted["start"].to_numpy()
            return out_chrom, out_start

        c1, s1 = lift_pos(src.chrom1.to_numpy(), src.start1.to_numpy())
        c2, s2 = lift_pos(src.chrom2.to_numpy(), src.start2.to_numpy())
        keep = (s1 >= 0) & (s2 >= 0)
        exp = pd.DataFrame({"chrom1": c1[keep].astype(str), "start1": s1[keep],
                            "chrom2": c2[keep].astype(str), "start2": s2[keep],
                            "v": src.v.to_numpy()[keep]})
        exp = exp.sort_values(list(exp.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(got, exp)

        # the data exercises every case: an unmapped end, the inverted block, cis -> trans
        def in_gap(chrom, start):
            return (chrom == "chrS1") & (start >= 300) & (start < 400)

        dropped = in_gap(src.chrom1, src.start1) | in_gap(src.chrom2, src.start2)
        assert dropped.any()
        assert len(got) == (~dropped).sum()
        assert ((got.chrom1 == "chrT1") & (got.start1 >= 1500)).any()
        assert ((got.chrom1 == "chrT1") & (got.chrom2 == "chrT2") & (got.start2 < 300)).any()

        # mirrored cis source -> mirrored target, including the cis pairs that became trans
        mirror = pd.DataFrame({"chrom1": got.chrom2, "start1": got.start2,
                               "chrom2": got.chrom1, "start2": got.start1, "v": got.v})
        mirror = mirror.sort_values(list(mirror.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(mirror, got)

    def test_each_kind_of_block(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1", "chrS2"], 1000)
        _import_points(tmp_path, "src", ["chrS1"] * 6, [10, 450, 50, 20, 350, 55],
                       ["chrS1", "chrS1", "chrS1", "chrS2", "chrS1", "chrS1"],
                       [20, 460, 750, 5, 30, 55], [1, 2, 3, 4, 5, 6])
        _mk_db(tmp_path / "tgt", ["chrT1", "chrT2"], 2000)
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"),
                           _chain(tmp_path / "x.chain", _POINTS_CHAIN))
        # (10, 20) -> (110, 120) and its mirror; (450, 460) is in the inverted block,
        # x -> 1500 + (699 - x); (50, 750) has 750 in the block to chrT2, so the cis point
        # becomes the trans pair chrT1-chrT2 (150, 50) plus chrT2-chrT1 (50, 150); the trans
        # (chrS1 20, chrS2 5) gives both orientations; (350, 30) has an end in the unmapped
        # [300, 400) and is dropped; the diagonal (55, 55) stays a single point.
        expected = pd.DataFrame({
            "chrom1": ["chrT1"] * 7 + ["chrT2"] * 2,
            "start1": [110, 120, 120, 150, 155, 1739, 1749, 50, 1005],
            "chrom2": ["chrT1", "chrT1", "chrT2", "chrT2", "chrT1", "chrT1", "chrT1", "chrT1", "chrT1"],
            "start2": [120, 110, 1005, 50, 155, 1749, 1739, 150, 120],
            "v": [1.0, 1, 4, 3, 6, 2, 2, 3, 4],
        })
        pd.testing.assert_frame_equal(_extract_points("lifted"), expected)

    @pytest.mark.parametrize(("agg", "expected"), [
        ("sum", [7, 7, 5]), ("mean", [3.5, 7, 2.5]), ("count", [2, 1, 2]),
        ("max", [5, 7, 4]), ("min", [2, 7, 1]),
    ])
    def test_merged_by_multi_target_agg(self, tmp_path, agg, expected):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        # (10, 20) and (510, 520) land on (10, 20); diagonal (45, 45) and (545, 545) on (45, 45)
        _import_points(tmp_path, "src", ["chrS1"] * 5, [10, 510, 30, 45, 545],
                       ["chrS1"] * 5, [20, 520, 40, 45, 545], [2, 5, 7, 1, 4])
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"),
                           _chain(tmp_path / "x.chain", _TWO_BLOCKS_ONTO_ONE),
                           tgt_overlap_policy="keep", multi_target_agg=agg)
        assert pm.gtrack_info("lifted")["type"] == "points"
        got = _extract_points("lifted")
        assert list(zip(got.start1, got.start2)) == [(10, 20), (20, 10), (30, 40), (40, 30), (45, 45)]
        values = dict(zip(zip(got.start1, got.start2), got.v))
        assert [values[(10, 20)], values[(30, 40)], values[(45, 45)]] == expected

    def test_first_and_last_pick_different_contributors(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        _import_points(tmp_path, "src", ["chrS1"] * 2, [10, 510], ["chrS1"] * 2, [20, 520], [2, 5])
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        chain = _chain(tmp_path / "x.chain", _TWO_BLOCKS_ONTO_ONE)
        picked = []
        for agg in ("first", "last"):
            pm.gtrack_liftover(agg, "x", str(src_db / "tracks" / "src.track"), chain,
                               tgt_overlap_policy="keep", multi_target_agg=agg)
            got = _extract_points(agg)
            picked.append(got.v[(got.start1 == 10) & (got.start2 == 20)].item())
        assert sorted(picked) == [2, 5]


class TestLiftoverAggregations:
    """gtrack_liftover takes every multi_target_agg R's gtrack.liftover takes: "nth", whose n
    is given by params, and the coverage aggregations, which take the value of the contributor
    that covers the most (least) of the locus by length or by fraction, a tie going to the
    higher value. They were rejected as unsupported."""

    @staticmethod
    def _lift_points(tmp_path, name, agg, **kwargs):
        if not (tmp_path / "src").exists():
            _mk_db(tmp_path / "src", ["chrS1"], 1000)
            _import_points(tmp_path, "src", ["chrS1"] * 5, [10, 510, 30, 45, 545],
                           ["chrS1"] * 5, [20, 520, 40, 45, 545], [2, 5, 7, 1, 4])
            _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        pm.gtrack_liftover(name, "x", str(tmp_path / "src" / "db" / "tracks" / "src.track"),
                           _chain(tmp_path / "x.chain", _TWO_BLOCKS_ONTO_ONE),
                           tgt_overlap_policy="keep", multi_target_agg=agg, **kwargs)
        got = _extract_points(name)
        return dict(zip(zip(got.start1, got.start2, strict=True), got.v, strict=True))

    def test_points_nth(self, tmp_path):
        # R 5.12.1's nth case: (10, 20) and (510, 520) land on (10, 20)
        last = self._lift_points(tmp_path, "last", "last")[(10, 20)]
        got = self._lift_points(tmp_path, "nth2", "nth", params=2)
        assert got[(10, 20)] == last
        assert np.isnan(got[(30, 40)])
        first = self._lift_points(tmp_path, "first", "first")[(10, 20)]
        assert first != last
        assert self._lift_points(tmp_path, "nth1", "nth", params={"n": 1})[(10, 20)] == first

    @pytest.mark.parametrize("agg", ["max.coverage_len", "min.coverage_len",
                                     "max.coverage_frac", "min.coverage_frac"])
    def test_points_coverage(self, tmp_path, agg):
        # every point covers its whole target point, so the higher value wins
        got = self._lift_points(tmp_path, "lifted", agg)
        assert [got[(10, 20)], got[(30, 40)], got[(45, 45)]] == [5, 7, 4]

    def test_rects(self, tmp_path):
        # R's test-gtrack.liftover-agg.R overlap case: R1 = x[0,10) y[100,110) (10) and
        # R2 = x[100,110) y[0,10) (20) land overlapping on x[5,10) y[5,10)
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        pm.gtrack_2d_create("src", "x", pd.DataFrame({
            "chrom1": ["chrS1"] * 2, "start1": [0, 100], "end1": [10, 110],
            "chrom2": ["chrS1"] * 2, "start2": [100, 0], "end2": [110, 10],
        }), [10.0, 20.0])
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        chain = _chain(tmp_path / "x.chain", [
            ("chrS1", 1000, "+", 0, 10, "chrT1", 1000, "+", 0, 10, 1),
            ("chrS1", 1000, "+", 100, 110, "chrT1", 1000, "+", 5, 15, 2),
        ])

        def overlap(agg, **kwargs):
            name = agg.replace(".", "_") + "".join(f"_{v}" for v in kwargs.values())
            if not pm.gtrack_exists(name):
                pm.gtrack_liftover(name, "x", str(src_db / "tracks" / "src.track"), chain,
                                   tgt_overlap_policy="keep", multi_target_agg=agg, **kwargs)
            rects = [r for r in _extract_rects(name) if r[0] <= 7 < r[1] and r[2] <= 7 < r[3]]
            assert len(rects) == 1
            return rects[0][4]

        for agg in ("max.coverage_len", "min.coverage_len", "max.coverage_frac", "min.coverage_frac"):
            assert overlap(agg) == 20
        assert overlap("nth", params=1) == overlap("first")
        assert overlap("nth", params=2) == overlap("last")
        assert {overlap("first"), overlap("last")} == {10, 20}
        assert np.isnan(overlap("nth", params=3))

    @pytest.mark.parametrize(("agg", "params", "expected"), [
        ("nth", 2, [1, 20, 30, 40, np.nan]),
        ("max.coverage_len", None, [10, 20, 30, 40, 4]),
        ("min.coverage_len", None, [1, 1, 2, 3, 4]),
        ("max.coverage_frac", None, [10, 20, 30, 40, 4]),
        ("min.coverage_frac", None, [1, 1, 2, 3, 4]),
    ])
    def test_fixed_bin(self, tmp_path, agg, params, expected):
        # R 5.12.1's values. Two blocks land on chrT1, a 10bp-binned source cut by the first
        # at an offset of 3, so most target bins get a piece of 3, one of 7 and a whole bin
        # of the second block; the last bin (40-45) is cut short by the chromosome end.
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 200)
        pm.gtrack_create_dense("dn", "x", pm.gintervals(["chrS1"] * 20, list(range(0, 200, 10)),
                                                        list(range(10, 210, 10))),
                               [1, 2, 3, 4] + [np.nan] * 6 + [10, 20, 30, 40] + [np.nan] * 6, 10, np.nan)
        _mk_db(tmp_path / "tgt", ["chrT1"], 45)
        chain = _chain(tmp_path / "x.chain", [
            ("chrS1", 200, "+", 0, 40, "chrT1", 45, "+", 3, 43, 1),
            ("chrS1", 200, "+", 100, 140, "chrT1", 45, "+", 0, 40, 2),
        ])
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "dn.track"), chain,
                           tgt_overlap_policy="keep", multi_target_agg=agg, params=params)
        got = pm.gextract("lifted", pm.gintervals(["chrT1"], [0], [45]), iterator=10)["lifted"]
        np.testing.assert_array_equal(got.to_numpy(), np.array(expected, dtype=float))

    @pytest.mark.parametrize(("agg", "params"), [
        ("nth", 1), ("max.coverage_len", None), ("min.coverage_len", None),
        ("max.coverage_frac", None), ("min.coverage_frac", None),
    ])
    def test_sparse(self, tmp_path, agg, params):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 200)
        pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chrS1"] * 2, [0, 100], [10, 110]), [3.0, 7.0])
        _mk_db(tmp_path / "tgt", ["chrT1"], 45)
        chain = _chain(tmp_path / "x.chain", [
            ("chrS1", 200, "+", 0, 20, "chrT1", 45, "+", 0, 20, 1),
            ("chrS1", 200, "+", 100, 120, "chrT1", 45, "+", 25, 45, 2),
        ])
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "sp.track"), chain,
                           multi_target_agg=agg, params=params)
        got = pm.gextract("lifted", pm.gintervals_all())
        assert list(zip(got.start, got.end, got["lifted"], strict=True)) == [(0, 10, 3.0), (25, 35, 7.0)]

    def test_params(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 200)
        pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chrS1"], [0], [10]), [3.0])
        _import_points(tmp_path, "pts", ["chrS1"], [10], ["chrS1"], [20], [1])
        _mk_db(tmp_path / "tgt", ["chrT1"], 200)
        chain = _chain(tmp_path / "x.chain", [("chrS1", 200, "+", 0, 200, "chrT1", 200, "+", 0, 200, 1)])
        for tr in ("sp", "pts"):
            src = str(src_db / "tracks" / f"{tr}.track")
            with pytest.raises(ValueError, match="params must be supplied for 'nth'"):
                pm.gtrack_liftover("lifted", "x", src, chain, multi_target_agg="nth")
            with pytest.raises(ValueError, match="positive integer"):
                pm.gtrack_liftover("lifted", "x", src, chain, multi_target_agg="nth", params=0)
            with pytest.raises(ValueError, match="params is only supported for 'nth'"):
                pm.gtrack_liftover("lifted", "x", src, chain, multi_target_agg="mean", params=2)
            with pytest.raises(ValueError, match="Unsupported aggregation"):
                pm.gtrack_liftover("lifted", "x", src, chain, multi_target_agg="mode")
            assert not pm.gtrack_exists("lifted")


def _num_subtrees(path):
    """The number of subtrees a 2D track file was written in. A file written in k > 1
    subtrees (StatQuadTreeCachedSerializer) has the top tree over them, (k - 1) / 3 nodes
    and no leaf, alone in its root chunk."""
    data = path.read_bytes()
    root = struct.unpack_from("<q", data, 12)[0]           # after the signature and num_objs
    size = struct.unpack_from("<q", data, root)[0] - 16    # after the chunk header
    nodes = size // 104                                    # sizeof(StatQuadTreeCached::Node)
    if size % 104 or not nodes or any(data[root + 16 + 104 * i] for i in range(nodes)):
        return 1
    return 3 * nodes + 1


class TestLiftover2DSubtrees:
    """As R's gtrack.2d.import_contacts, a lifted pair of at least 4 x CONFIG['max_data_size']
    points is written in subtrees (a power of 4, rounded down), holding one subtree in memory
    at a time, and the track is the same."""

    @pytest.fixture()
    def src_and_chain(self, tmp_path):
        # R 5.12.1's split case
        src_db = _mk_db(tmp_path / "src", ["chrS1", "chrS2"], 1000)
        rng = np.random.default_rng(3)
        n = 500
        chroms = np.array(["chrS1", "chrS2"])
        _import_points(tmp_path, "src", chroms[rng.integers(0, 2, n)], rng.integers(0, 1000, n),
                       chroms[rng.integers(0, 2, n)], rng.integers(0, 1000, n), rng.random(n))
        tgt_db = _mk_db(tmp_path / "tgt", ["chrT1", "chrT2"], 2000)
        return str(src_db / "tracks" / "src.track"), _chain(tmp_path / "x.chain", _POINTS_CHAIN), tgt_db

    def test_split_gives_the_same_track(self, src_and_chain, monkeypatch):
        src, chain, tgt_db = src_and_chain
        pm.gtrack_liftover("whole", "x", src, chain)
        monkeypatch.setitem(pm.CONFIG, "max_data_size", 10)
        pm.gtrack_liftover("split", "x", src, chain)
        assert pm.gtrack_info("split")["type"] == "points"
        whole = _extract_points("whole")
        assert len(whole) > 400
        pd.testing.assert_frame_equal(_extract_points("split"), whole)

        # every pair was split, by its number of points (the chain is 1:1, so no point
        # merges with another): 4^floor(log4(n / 10)) subtrees
        whole_dir, split_dir = (tgt_db / "tracks" / f"{t}.track" for t in ("whole", "split"))
        pairs = whole.groupby(["chrom1", "chrom2"]).size()
        expected = {f"{c1}-{c2}": 4 ** int(np.log2(max(n // 10, 1)) // 2) for (c1, c2), n in pairs.items()}
        assert min(expected.values()) > 1
        assert {p.name: _num_subtrees(p) for p in split_dir.iterdir() if p.name[0] != "."} == expected
        assert {p.name: _num_subtrees(p) for p in whole_dir.iterdir() if p.name[0] != "."} == \
            dict.fromkeys(expected, 1)
        # no buffer or subtree files are left behind
        assert sorted(p.name for p in split_dir.iterdir()) == sorted(p.name for p in whole_dir.iterdir())

    def test_max_data_size_below_one_counts_as_one(self, src_and_chain, monkeypatch):
        # it divided by zero in R before 5.12.1
        src, chain, _ = src_and_chain
        pm.gtrack_liftover("whole", "x", src, chain)
        monkeypatch.setitem(pm.CONFIG, "max_data_size", 0)
        pm.gtrack_liftover("split", "x", src, chain)
        pd.testing.assert_frame_equal(_extract_points("split"), _extract_points("whole"))

    def test_pair_buffers_are_closed_before_the_pairs_are_written(self, tmp_path):
        # Every target pair has a buffer file while the source is read. As in R (5.12.1),
        # the buffers are closed before the pairs are written, as a pair written in subtrees
        # opens a file per subtree: with 256 pairs of 64 subtrees each, a limit of 256 + 32
        # files above those already open fits the read, and would not fit the buffers kept
        # open with the subtrees of a pair.
        n = 16
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000 * n)
        # 40 contacts per pair, each in both orientations: 80 points per pair
        i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(40), indexing="ij")
        x, y = (1000 * i + 25 * k).ravel(), (1000 * j + 25 * k + 1).ravel()
        _import_points(tmp_path, "src", ["chrS1"] * len(x), x, ["chrS1"] * len(y), y, np.ones(len(x)))
        tgt_chroms = [f"chrT{i}" for i in range(n)]
        tgt_db = _mk_db(tmp_path / "tgt", tgt_chroms, 1000)
        chain = _chain(tmp_path / "x.chain", [
            ("chrS1", 1000 * n, "+", 1000 * i, 1000 * (i + 1), c, 1000, "+", 0, 1000, i + 1)
            for i, c in enumerate(tgt_chroms)
        ])
        script = textwrap.dedent(f"""
            import os, resource
            import pymisha as pm
            pm.gdb_init({str(tgt_db)!r})
            pm.CONFIG["max_data_size"] = 1
            n_open = len(os.listdir("/dev/fd"))
            hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            resource.setrlimit(resource.RLIMIT_NOFILE, (n_open + {n * n} + 32, hard))
            pm.gtrack_liftover("lifted", "x", {str(src_db / "tracks" / "src.track")!r}, {chain!r})
        """)
        res = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        pairs = [p for p in (tgt_db / "tracks" / "lifted.track").iterdir() if p.name[0] != "."]
        assert len(pairs) == n * n
        assert {_num_subtrees(p) for p in pairs} == {64}


class TestLiftover2DEmpty:
    """A 2D lift that maps nothing gives the track R's gtrack.liftover gives: no pair files,
    which R and pymisha alike report as a sparse track, and in an indexed database the
    empty indexed track R writes. That one was missing."""

    @pytest.mark.parametrize("track", ["pts", "rct"])
    @pytest.mark.parametrize("db_format", ["per-chromosome", "indexed"])
    def test_maps_nothing(self, tmp_path, track, db_format):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        _import_points(tmp_path, "pts", ["chrS1"], [10], ["chrS1"], [20], [1])
        pm.gtrack_2d_create("rct", "x", pd.DataFrame({
            "chrom1": ["chrS1"], "start1": [10], "end1": [20],
            "chrom2": ["chrS1"], "start2": [30], "end2": [40]}), [2.0])
        tgt_db = _mk_db(tmp_path / "tgt", ["chrT1"], 1000, db_format=db_format)
        # the chain maps chrS1 [500, 600) only
        chain = _chain(tmp_path / "x.chain", [("chrS1", 1000, "+", 500, 600, "chrT1", 1000, "+", 0, 100, 1)])
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / f"{track}.track"), chain)
        info = pm.gtrack_info("lifted")
        assert info["type"] == "sparse"
        files = sorted(p.name for p in (tgt_db / "tracks" / "lifted.track").iterdir())
        if db_format == "indexed":
            assert files == [".attributes", "track.dat", "track.idx"]
            assert info["format"] == "indexed"
        else:
            assert files == [".attributes"]


class TestLiftover1DEmpty:
    """A 1D lift that lifts no value gives the track R's gtrack.liftover gives: a dense
    track of NaN for a dense source, a sparse track with an empty file per chromosome for a
    sparse one, indexed in an indexed database. It was an empty directory, which the
    database took for a sparse track, a dense source included."""

    @pytest.mark.parametrize("db_format", ["per-chromosome", "indexed"])
    @pytest.mark.parametrize("force_python", [False, True])
    def test_maps_nothing(self, tmp_path, db_format, force_python):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chrS1"], [10], [20]), [1.0])
        pm.gtrack_create_dense("dn", "x", pm.gintervals(["chrS1"], [0], [100]), [1.0], 10, np.nan)
        tgt_db = _mk_db(tmp_path / "tgt", ["chrT1", "chrT2"], 1000, db_format=db_format)
        # the chain maps chrS1 [500, 600), where neither track has a value
        chain = _chain(tmp_path / "x.chain", [("chrS1", 1000, "+", 500, 600, "chrT1", 1000, "+", 0, 100, 1)])
        for tr in ("sp", "dn"):
            pm.gtrack_liftover(tr, "x", str(src_db / "tracks" / f"{tr}.track"), chain,
                               _force_pure_python=force_python)
            files = sorted(p.name for p in (tgt_db / "tracks" / f"{tr}.track").iterdir() if p.name[0] != ".")
            info = pm.gtrack_info(tr)
            assert info["format"] == db_format
            assert files == (["track.dat", "track.idx"] if db_format == "indexed" else ["chrT1", "chrT2"])
        assert pm.gtrack_info("sp")["type"] == "sparse"
        assert pm.gextract("sp", pm.gintervals_all()) is None
        assert pm.gtrack_info("dn")["type"] == "dense"
        got = pm.gextract("dn", pm.gintervals_all())
        assert len(got) == 200
        assert got["dn"].isna().all()


class TestLiftover2DNanCells:
    """A target cell whose aggregate is NaN is kept with a NaN value, as R inserts it.
    It was dropped from the lifted track."""

    def test_point_below_min_n_is_kept_as_nan(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        # (10, 20) and (510, 520) land on (10, 20); (30, 40) has one contributor
        _import_points(tmp_path, "src", ["chrS1"] * 5, [10, 510, 30, 45, 545],
                       ["chrS1"] * 5, [20, 520, 40, 45, 545], [2, 5, 7, 1, 4])
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        chain = _chain(tmp_path / "x.chain", _TWO_BLOCKS_ONTO_ONE)
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"), chain,
                           tgt_overlap_policy="keep", multi_target_agg="mean", min_n=2)
        assert pm.gtrack_info("lifted")["type"] == "points"
        got = _extract_points("lifted")
        assert list(zip(got.start1, got.start2)) == [(10, 20), (20, 10), (30, 40), (40, 30), (45, 45)]
        assert got.v.tolist()[:2] == [3.5, 3.5]
        assert np.isnan(got.v.tolist()[2:4]).all()
        assert got.v.tolist()[4] == 2.5

    def test_rect_below_min_n_is_kept_as_nan(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        # the first two land on one target rect; the third has one contributor
        pm.gtrack_2d_create("src", "x", pd.DataFrame({
            "chrom1": ["chrS1"] * 3, "start1": [10, 510, 60], "end1": [20, 520, 70],
            "chrom2": ["chrS1"] * 3, "start2": [30, 530, 80], "end2": [40, 540, 90],
        }), [2.0, 4.0, 5.0])
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        chain = _chain(tmp_path / "x.chain", _TWO_BLOCKS_ONTO_ONE)
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"), chain,
                           tgt_overlap_policy="keep", multi_target_agg="mean", min_n=2)
        got = _extract_rects("lifted")
        assert [r[:4] for r in got] == [(10, 20, 30, 40), (60, 70, 80, 90)]
        assert got[0][4] == 3.0
        assert np.isnan(got[1][4])


class TestLiftover2DInfiniteValues:
    """An infinite source value becomes NaN, as R does for points and rects. It was
    lifted as is, so a sum over the merged points was infinite."""

    def test_points(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        _import_points(tmp_path, "src", ["chrS1"] * 3, [10, 30, 510],
                       ["chrS1"] * 3, [20, 40, 520], [np.inf, -np.inf, 3])
        assert _extract_points("src").v.tolist() == [np.inf, np.inf, -np.inf, -np.inf, 3, 3]
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        # (10, 20) and (510, 520) land on (10, 20)
        chain = _chain(tmp_path / "x.chain", _TWO_BLOCKS_ONTO_ONE)
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"), chain,
                           tgt_overlap_policy="keep", multi_target_agg="sum")
        got = _extract_points("lifted")
        assert got.start1.tolist() == [10, 20, 30, 40]
        # the NaN from Inf is dropped by na_rm; -Inf alone gives NaN
        assert got.v.tolist()[:2] == [3, 3]
        assert np.isnan(got.v.tolist()[2:]).all()

    def test_rects(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000)
        pm.gtrack_2d_create("src", "x", pd.DataFrame({
            "chrom1": ["chrS1"] * 2, "start1": [10, 200], "end1": [50, 260],
            "chrom2": ["chrS1"] * 2, "start2": [100, 300], "end2": [150, 350],
        }), [np.inf, 2.0])
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        chain = _chain(tmp_path / "x.chain", [("chrS1", 1000, "+", 0, 1000, "chrT1", 1000, "+", 0, 1000, 1)])
        pm.gtrack_liftover("lifted", "x", str(src_db / "tracks" / "src.track"), chain)
        got = _extract_rects("lifted")
        assert [r[:4] for r in got] == [(10, 50, 100, 150), (200, 260, 300, 350)]
        assert np.isnan(got[0][4])
        assert got[1][4] == 2.0


class TestLiftoverIndexedSourceDatabase:
    """An indexed source track is read by the chrom ids of the database that holds it,
    found from the resolved path at the nearest "tracks" directory with a
    chrom_sizes.txt next to it."""

    @pytest.fixture()
    def src_and_chain(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1", "chrS2"], 1000)
        pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chrS1", "chrS2"], 0, 100), [1.0, 2.0])
        pm.gtrack_create("dn", "x", "sp", iterator=100)
        pm.gtrack_convert_to_indexed("sp", remove_old=True)
        assert (src_db / "tracks" / "sp.track" / "track.idx").exists()
        _mk_db(tmp_path / "tgt", ["chrT1", "chrT2"], 1000)
        chain = _chain(tmp_path / "x.chain", [
            ("chrS1", 1000, "+", 0, 1000, "chrT1", 1000, "+", 0, 1000, 1),
            ("chrS2", 1000, "+", 0, 1000, "chrT2", 1000, "+", 0, 1000, 2),
        ])
        return src_db, chain

    @staticmethod
    def _lifted(track):
        return pm.gextract(track, pm.gintervals(["chrT1", "chrT2"], 0, 100))[track].tolist()

    def test_relative_path(self, src_and_chain, monkeypatch):
        src_db, chain = src_and_chain
        monkeypatch.chdir(src_db / "tracks")
        pm.gtrack_liftover("from_tracks_dir", "x", "sp.track", chain)
        assert self._lifted("from_tracks_dir") == [1.0, 2.0]
        monkeypatch.chdir(src_db)
        pm.gtrack_liftover("from_db_root", "x", "tracks/sp.track", chain)
        assert self._lifted("from_db_root") == [1.0, 2.0]
        monkeypatch.chdir(src_db / "tracks" / "sp.track")
        pm.gtrack_liftover("from_track_dir", "x", ".", chain)
        assert self._lifted("from_track_dir") == [1.0, 2.0]

    def test_tracks_dir_without_chrom_sizes_is_skipped(self, src_and_chain):
        # a directory named "tracks" inside the database, with no chrom_sizes.txt beside it
        src_db, chain = src_and_chain
        nested = src_db / "tracks" / "sub" / "tracks" / "sp.track"
        shutil.copytree(src_db / "tracks" / "sp.track", nested)
        pm.gtrack_liftover("lifted", "x", str(nested), chain)
        assert self._lifted("lifted") == [1.0, 2.0]

    def test_outside_a_database_is_an_error(self, src_and_chain, tmp_path):
        src_db, chain = src_and_chain
        loose = tmp_path / "loose"
        for tr in ("sp", "dn"):
            shutil.copytree(src_db / "tracks" / f"{tr}.track", loose / f"{tr}.track")
        # the chrom ids of an indexed track are unknown without its database
        with pytest.raises(ValueError, match="is not in a database"):
            pm.gtrack_liftover("lifted_sp", "x", str(loose / "sp.track"), chain)
        # a per-chromosome track is read by file name and needs none
        pm.gtrack_liftover("lifted_dn", "x", str(loose / "dn.track"), chain)
        assert self._lifted("lifted_dn") == [1.0, 2.0]


class TestLiftoverIndexed2DSource:
    """An indexed 2D source track (track.idx + track.dat) is lifted, each chromosome pair
    read by the chrom ids of the source database. It was refused with "Invalid track
    index header"."""

    def test_read_by_the_source_database_ids(self, tmp_path):
        # the same track as per-pair files and in an indexed DB, where the import converts
        # it to indexed
        contacts = (["chrS1", "chrS2"], [10, 40], ["chrS1", "chrS2"], [100, 300], [1, 4])
        src_db = _mk_db(tmp_path / "src", ["chrS1", "chrS2"], 1000)
        _import_points(tmp_path, "src", *contacts)
        assert not (src_db / "tracks" / "src.track" / "track.idx").exists()
        src_idx_db = _mk_db(tmp_path / "src_idx", ["chrS1", "chrS2"], 1000, db_format="indexed")
        _import_points(tmp_path, "src_idx", *contacts)
        assert (src_idx_db / "tracks" / "src_idx.track" / "track.idx").exists()

        _mk_db(tmp_path / "tgt", ["chrT2"], 2000)
        # the chain covers chrS2 only, so its chromosomes are not the source genome's
        chain = _chain(tmp_path / "x.chain", [("chrS2", 1000, "+", 0, 1000, "chrT2", 2000, "+", 0, 1000, 1)])
        pm.gtrack_liftover("from_files", "x", str(src_db / "tracks" / "src.track"), chain)
        pm.gtrack_liftover("from_idx", "x", str(src_idx_db / "tracks" / "src_idx.track"), chain)
        expected = _extract_points("from_files")
        assert expected.start1.tolist() == [40, 300]
        assert pm.gtrack_info("from_idx")["type"] == "points"
        pd.testing.assert_frame_equal(_extract_points("from_idx"), expected)

    def test_per_chromosome_db_with_unsorted_chrom_sizes(self, tmp_path):
        # R's setup_db_with_unprefixed_chrom_sizes: chrom_sizes.txt lists chromosome 2
        # before 1. pymisha, as R, names them chr1 and chr2 and sorts their ids.
        fastas = []
        for name, size in (("chr1", 1000), ("chr2", 2000)):
            fasta = tmp_path / f"{name}.fasta"
            fasta.write_text(f">{name}\n{'A' * size}\n")
            fastas.append(str(fasta))
        src_db = tmp_path / "src"
        pm.gdb_create(str(src_db), fastas, db_format="per-chromosome")
        (src_db / "chrom_sizes.txt").write_text("2\t2000\n1\t1000\n")
        pm.gdb_init(str(src_db))
        pm.gtrack_create_sparse("sp", "x", pm.gintervals(["1", "2"], 0, 100), [1.0, 2.0])
        pm.gtrack_create("dn", "x", "sp", iterator=100)
        _import_points(tmp_path, "pts", ["1", "2", "1"], [10, 20, 50], ["1", "2", "2"], [30, 40, 60], [1, 2, 3])

        tgt_db = _mk_db(tmp_path / "tgt", ["chrT1", "chrT2"], 1000)
        chain = _chain(tmp_path / "x.chain", [
            ("chr1", 1000, "+", 0, 1000, "chrT1", 1000, "+", 0, 1000, 1),
            ("chr2", 2000, "+", 0, 1000, "chrT2", 1000, "+", 0, 1000, 2),
        ])

        def lift_all(prefix):
            pm.gdb_init(str(tgt_db))
            for tr in ("sp", "dn", "pts"):
                pm.gtrack_liftover(prefix + tr, "x", str(src_db / "tracks" / f"{tr}.track"), chain)
            return (
                pm.gextract(prefix + "sp", pm.gintervals_all())[prefix + "sp"].tolist(),
                pm.gextract(prefix + "dn", pm.gintervals(["chrT1", "chrT2"], 0, 100))[prefix + "dn"].tolist(),
                _extract_points(prefix + "pts").values.tolist(),
            )

        from_files = lift_all("files_")
        assert from_files[0] == [1.0, 2.0]
        assert from_files[1] == [1.0, 2.0]
        assert from_files[2] == [
            ["chrT1", 10, "chrT1", 30, 1.0], ["chrT1", 30, "chrT1", 10, 1.0],
            ["chrT1", 50, "chrT2", 60, 3.0], ["chrT2", 20, "chrT2", 40, 2.0],
            ["chrT2", 40, "chrT2", 20, 2.0], ["chrT2", 60, "chrT1", 50, 3.0],
        ]

        pm.gdb_init(str(src_db))
        for tr in ("sp", "dn"):
            pm.gtrack_convert_to_indexed(tr, remove_old=True)
        pm.gtrack_2d_convert_to_indexed("pts", remove_old=True)
        for tr in ("sp", "dn", "pts"):
            assert (src_db / "tracks" / f"{tr}.track" / "track.idx").exists()
        assert lift_all("idx_") == from_files

    def test_outside_a_database_is_an_error(self, tmp_path):
        src_db = _mk_db(tmp_path / "src", ["chrS1"], 1000, db_format="indexed")
        _import_points(tmp_path, "src", ["chrS1"], [10], ["chrS1"], [100], [1])
        loose = tmp_path / "loose" / "src.track"
        shutil.copytree(src_db / "tracks" / "src.track", loose)
        assert (loose / "track.idx").exists()
        _mk_db(tmp_path / "tgt", ["chrT1"], 1000)
        chain = _chain(tmp_path / "x.chain", [("chrS1", 1000, "+", 0, 1000, "chrT1", 1000, "+", 0, 1000, 1)])
        with pytest.raises(ValueError, match="is not in a database"):
            pm.gtrack_liftover("lifted", "x", str(loose), chain)
