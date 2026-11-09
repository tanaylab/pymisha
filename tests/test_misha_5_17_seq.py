"""R misha 5.17.0 sequence changes: gseq_extract(pad), a byte above 127 on the reverse strand,
random draws in forked kids, gseq_kmer_dist and gsynth fixes, per-chromosome multi-FASTA
databases. The cases are those of R's tests (test-gseq.extract.R,
test-gseq-extract-parallel.R, test-sequence.R, test-gsample.R, test-gsynth*.R,
test-multifasta-import.R)."""

import math
import os
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest
import yaml

import pymisha as pm
from pymisha import _shared

SEED = 60427


def _all():
    return pm.gintervals_all()


def _revcomp(s):
    return s.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def _pad_ref(iv):
    """The part of each interval inside its chromosome read without pad, N runs added; strand
    -1 rows are the reverse complement of the whole padded forward sequence."""
    sizes = dict(zip(_all()["chrom"], _all()["end"], strict=True))
    lens = np.array([sizes[c] for c in iv["chrom"]])
    cs = np.maximum(iv["start"].to_numpy(), 0)
    ce = np.minimum(iv["end"].to_numpy(), lens)
    inside = cs < ce
    core = [""] * len(iv)
    if inside.any():
        got = pm.gseq_extract(pd.DataFrame({"chrom": iv["chrom"][inside].to_numpy(),
                                            "start": cs[inside], "end": ce[inside]}))
        for i, s in zip(np.flatnonzero(inside), got, strict=True):
            core[i] = s
    left = np.where(inside, cs - iv["start"].to_numpy(), iv["end"].to_numpy() - iv["start"].to_numpy())
    right = np.where(inside, iv["end"].to_numpy() - ce, 0)
    res = ["N" * int(a) + c + "N" * int(b) for a, c, b in zip(left, core, right, strict=True)]
    if "strand" in iv.columns:
        res = [_revcomp(s) if st in (-1, "-") else s for s, st in zip(res, iv["strand"], strict=True)]
    return res


# ---------------------------------------------------------------------------
# gseq_extract(pad=True)
# ---------------------------------------------------------------------------


class TestGseqExtractPad:
    def test_fills_the_part_past_either_end_with_n(self):
        len1, len2 = int(_all()["end"][0]), int(_all()["end"][1])
        c1, c2 = _all()["chrom"][0], _all()["chrom"][1]
        iv = pd.DataFrame({
            "chrom": [c1, c1, c2, c1, c1, c2, c1],
            "start": [-50, len1 - 30, -20, -300, len1 + 10, len2 - 1, 1000],
            "end": [70, len1 + 40, 120, -100, len1 + 60, len2 + 1, 1100],
        })
        seqs = pm.gseq_extract(iv, pad=True)
        assert [len(s) for s in seqs] == list(iv["end"] - iv["start"])
        assert seqs == _pad_ref(iv)
        # left end, right end, wholly outside on either side
        assert seqs[0][:50] == "N" * 50
        assert seqs[1][30:70] == "N" * 40
        assert seqs[3] == "N" * 200
        assert seqs[4] == "N" * 50
        # rows within their chromosome are read as without pad
        assert seqs[6] == pm.gseq_extract(iv.iloc[[6]])[0]
        assert seqs[0][50:120] == pm.gseq_extract(pm.gintervals(c1, 0, 70))[0]

    def test_window_longer_than_the_whole_chromosome(self):
        i = int(np.argmin(_all()["end"]))
        chrom, length = _all()["chrom"][i], int(_all()["end"][i])
        seqs = pm.gseq_extract(pd.DataFrame({"chrom": [chrom], "start": [-7], "end": [length + 9]}), pad=True)
        assert seqs == ["N" * 7 + pm.gseq_extract(pm.gintervals(chrom, 0, length))[0] + "N" * 9]

    def test_reverse_complements_strand_minus_rows_as_a_whole(self):
        len1, c1 = int(_all()["end"][0]), _all()["chrom"][0]
        iv = pd.DataFrame({
            "chrom": c1,
            "start": [-50, len1 - 30, -300, 1000, -5],
            "end": [70, len1 + 40, -100, 1100, 20],
            "strand": [-1, -1, -1, -1, 1],
        })
        seqs = pm.gseq_extract(iv, pad=True)
        assert seqs == _pad_ref(iv)
        # the N run of the chromosome start comes last on strand -1
        assert seqs[0][70:120] == "N" * 50
        assert seqs[1][:40] == "N" * 40
        assert seqs[0] == _revcomp(pm.gseq_extract(iv.iloc[[0], :3], pad=True)[0])
        assert seqs[3] == pm.gseq_extract(iv.iloc[[3]])[0]

        # strand as "+"/"-" text or categorical, and float coordinates, give the same
        iv_chr = iv.assign(strand=np.where(iv["strand"] == -1, "-", "+"))
        assert pm.gseq_extract(iv_chr, pad=True) == seqs
        assert pm.gseq_extract(iv_chr.assign(strand=iv_chr["strand"].astype("category")), pad=True) == seqs
        iv_float = iv.assign(start=iv["start"].astype(float), end=iv["end"].astype(float))
        assert pm.gseq_extract(iv_float, pad=True) == seqs

    def test_keeps_the_row_order_and_extra_columns_do_not_matter(self):
        rng = np.random.default_rng(SEED)
        sizes = _all()
        n, w = 300, 500
        i = rng.integers(0, len(sizes), n)
        ends = sizes["end"].to_numpy()[i]
        centers = np.floor(rng.random(n) * ends).astype(np.int64)
        # a third of the windows sit on a chromosome end
        edge = rng.random(n) < 1 / 3
        centers[edge] = np.where(rng.random(edge.sum()) < 0.5, 0, ends[edge])
        iv = pd.DataFrame({"chrom": sizes["chrom"].to_numpy()[i], "start": centers - w // 2,
                           "end": centers + w // 2, "strand": rng.choice([-1, 1], n),
                           "id": np.arange(n)})
        seqs = pm.gseq_extract(iv, pad=True)
        assert all(len(s) == w for s in seqs)
        assert seqs == _pad_ref(iv)

    def test_many_windows_some_wholly_outside(self):
        # R's "pad = TRUE multitasking matches sequential" case: pymisha reads in one process
        rng = np.random.default_rng(SEED)
        sizes = _all()
        n, w = 2000, 1000
        i = rng.integers(0, len(sizes), n)
        ends = sizes["end"].to_numpy()[i]
        centers = np.floor(rng.random(n) * ends).astype(np.int64)
        edge = rng.choice(n, 400, replace=False)
        centers[edge] = np.where(rng.random(400) < 0.5, 0, ends[edge])
        outside = rng.choice(np.setdiff1d(np.arange(n), edge), 20, replace=False)
        centers[outside] = np.where(rng.random(20) < 0.5, -5 * w, ends[outside] + 5 * w)
        iv = pd.DataFrame({"chrom": sizes["chrom"].to_numpy()[i], "start": centers - w // 2,
                           "end": centers + w // 2, "strand": rng.choice([-1, 1], n)})
        seqs = pm.gseq_extract(iv, pad=True)
        assert all(len(s) == w for s in seqs)
        assert [seqs[j] for j in outside] == ["N" * w] * 20
        assert seqs == _pad_ref(iv)

    def test_without_a_row_past_an_end_is_the_same_as_pad_false(self):
        iv = pm.gscreen("dense_track > 0.6", pm.gintervals(["chr1", "chr2", "chrX"]))
        iv = iv.assign(strand=np.resize([1, -1], len(iv)))
        assert pm.gseq_extract(iv, pad=True) == pm.gseq_extract(iv)
        assert pm.gseq_extract(pm.gintervals("chr1", 0, 10).iloc[0:0], pad=True) == []

    def test_pads_a_named_intervals_set(self):
        c1 = _all()["chrom"][0]
        iv = pd.DataFrame({"chrom": c1, "start": [-5, 100], "end": [10, 120]})
        name = "test_seq_pad_set"
        try:
            pm.gintervals_save(iv, name)
            assert pm.gseq_extract(name, pad=True) == _pad_ref(iv)
        finally:
            if pm.gintervals_exists(name):
                pm.gintervals_rm(name, force=True)

    def test_without_pad_still_rejects_intervals_past_an_end(self):
        len1, c1 = int(_all()["end"][0]), _all()["chrom"][0]
        with pytest.raises(Exception, match="start coordinate must be greater or equal than zero"):
            pm.gseq_extract(pd.DataFrame({"chrom": [c1], "start": [-5], "end": [10]}))
        with pytest.raises(Exception, match="end coordinate exceeds chromosome boundaries"):
            pm.gseq_extract(pd.DataFrame({"chrom": [c1], "start": [len1 - 5], "end": [len1 + 5]}))
        with pytest.raises(Exception, match="end coordinate exceeds chromosome boundaries"):
            pm.gseq_extract(pd.DataFrame({"chrom": [c1], "start": [len1 - 5], "end": [len1 + 5]}), pad=False)

    def test_still_validates_the_intervals(self):
        c1 = _all()["chrom"][0]

        def one(start, end, **kw):
            return pd.DataFrame({"chrom": [c1], "start": [start], "end": [end], **{k: [v] for k, v in kw.items()}})

        with pytest.raises(Exception, match="lesser than end"):
            pm.gseq_extract(one(-3, -5), pad=True)
        with pytest.raises(Exception, match="lesser than end"):
            pm.gseq_extract(one(10, 5), pad=True)
        with pytest.raises(Exception, match="does not exist"):
            pm.gseq_extract(pd.DataFrame({"chrom": ["chrNoSuch"], "start": [-3], "end": [5]}), pad=True)
        with pytest.raises(Exception, match="Invalid strand value"):
            pm.gseq_extract(one(-10, -5, strand=5), pad=True)
        with pytest.raises(Exception):
            pm.gseq_extract(pm.gintervals_2d(1, 10, 100, 2, 20, 300), pad=True)
        with pytest.raises(ValueError, match="pad must be True or False"):
            pm.gseq_extract(one(0, 5), pad=None)
        with pytest.raises(ValueError, match="pad must be True or False"):
            pm.gseq_extract(one(0, 5), pad="yes")

    def test_rejects_non_finite_coordinates_as_pad_false_does(self):
        c1 = _all()["chrom"][0]

        def msg(iv, pad):
            try:
                pm.gseq_extract(iv, pad=pad)
            except Exception as e:  # noqa: BLE001
                return str(e)
            return None

        def one(start, end):
            return pd.DataFrame({"chrom": [c1], "start": [start], "end": [end]})

        cases = [(-math.inf, 10), (10, math.inf), (-math.inf, math.inf), (math.nan, 10), (-5, math.nan)]
        with np.errstate(invalid="ignore"):
            # alone: the same error as without pad
            for start, end in cases:
                m = msg(one(start, end), True)
                assert m is not None
                assert m == msg(one(start, end), False)
            # next to a row that is padded: still an error, never a sequence holding "NA"
            for start, end in cases:
                with pytest.raises(Exception, match="coordinate|Invalid format of interval|NaN"):
                    pm.gseq_extract(pd.concat([one(-5, 10), one(start, end)], ignore_index=True), pad=True)

    @pytest.mark.parametrize("db_format", ["indexed", "per-chromosome"])
    def test_keeps_the_genomes_lowercase_and_n(self, tmp_path, db_format):
        # one FASTA per chromosome, named after it
        (tmp_path / "chrA.fa").write_text(">chrA\nNNacgtACGTnn\n")
        (tmp_path / "chrB.fa").write_text(">chrB\nACGTAC\n")
        groot = tmp_path / "db"
        pm.gdb_create(str(groot), [str(tmp_path / "chrA.fa"), str(tmp_path / "chrB.fa")], db_format=db_format)
        pm.gdb_init(str(groot))
        iv = pd.DataFrame({
            "chrom": ["chrA", "chrA", "chrA", "chrB", "chrB"],
            "start": [-3, 8, -3, 6, -4],
            "end": [4, 15, 15, 9, -1],
            "strand": [1, 1, -1, 1, -1],
        })
        assert pm.gseq_extract(iv, pad=True) == ["NNNNNac", "GTnnNNN", "NNNnnACGTacgtNNNNN", "NNN", "NNN"]

    def test_reads_an_interval_on_an_empty_contig_as_all_n(self, tmp_path):
        fasta = tmp_path / "g.fa"
        fasta.write_text(">contig1\nACTG\n>empty\n>contig2\nGGGG\n")
        groot = tmp_path / "db"
        pm.gdb_create(str(groot), str(fasta))
        pm.gdb_init(str(groot))
        sizes = pm.gintervals_all()
        assert int(sizes.loc[sizes["chrom"] == "empty", "end"].iloc[0]) == 0
        iv = pd.DataFrame({"chrom": ["contig1", "empty", "contig2"], "start": [-1, -2, 2], "end": [2, 3, 6]})
        assert pm.gseq_extract(iv, pad=True) == ["NAC", "NNNNN", "GGNN"]


def test_reverse_strand_read_leaves_a_byte_above_127_as_it_is(tmp_path):
    db = tmp_path / "highbyte"
    (db / "seq").mkdir(parents=True)
    (db / "tracks").mkdir()
    data = bytearray(b"ACGTTGCAAC" * 6)
    data[30] = 200
    (db / "seq" / "chrT.seq").write_bytes(bytes(data))
    (db / "chrom_sizes.txt").write_text(f"chrT\t{len(data)}\n")
    pm.gdb_init(str(db))

    # one character per byte (Latin-1): .encode("latin-1") gives the bytes R returns
    got = pm.gseq_extract(pd.DataFrame({"chrom": ["chrT"], "start": [25], "end": [37], "strand": [-1]}))[0]
    expected = bytes(data[25:37])[::-1]
    comp = {ord("A"): ord("T"), ord("C"): ord("G"), ord("G"): ord("C"), ord("T"): ord("A")}
    expected = bytes(comp.get(b, b) for b in expected)
    assert got.encode("latin-1") == expected
    assert "\xc8" in got
    # forward strand too, and the length is end - start
    got = pm.gseq_extract(pd.DataFrame({"chrom": ["chrT"], "start": [25], "end": [37]}))[0]
    assert got.encode("latin-1") == bytes(data[25:37])
    assert len(pm.gseq_extract(pm.gintervals("chrT", 0, 60))[0]) == 60

    # the other functions that read the sequence keep the byte the same way
    iv = pm.gintervals("chrT", 0, 60)
    replaced = pm.gsynth_replace_kmer("CG", "GC", intervals=iv, output_format="vector")[0]
    assert len(replaced) == 60 and replaced[30] == "\xc8"
    kd = pm.gseq_kmer_dist(iv, k=2)
    plain = bytes(data).decode("latin-1")
    expected_count = sum(1 for i in range(59) if set(plain[i:i + 2]) <= set("ACGT"))
    assert kd["count"].sum() == expected_count
    sampled = pm.gsynth_random(intervals=iv, output_format="vector", seed=1,
                               mask_copy=pm.gintervals("chrT", 28, 33))[0]
    assert sampled[28:33] == plain[28:33]


# ---------------------------------------------------------------------------
# random draws: numpy's global generator, as R's .Random.seed
# ---------------------------------------------------------------------------


def _same(a, b):
    if isinstance(a, pd.DataFrame):
        return a.equals(b)
    return np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))


def _seeded(fn, seed):
    np.random.seed(seed)
    return fn()


def _check_seed_reproduces(fn):
    """The same numpy seed gives the same result, another seed another one."""
    a = _seeded(fn, 1)
    assert _same(_seeded(fn, 1), a)
    assert not _same(_seeded(fn, 2), a)


def test_gsample_follows_numpys_seed_and_advances_it():
    iv = pm.gintervals("chr1", 0, 100000)
    _check_seed_reproduces(lambda: pm.gsample("dense_track", 50, iv))
    np.random.seed(4)
    s1 = pm.gsample("dense_track", 50, iv)
    s2 = pm.gsample("dense_track", 50, iv)
    # the next call does not repeat the numbers the first one used
    assert not np.array_equal(s1, s2)
    # a call that draws nothing leaves numpy's generator as it is; one that draws advances it
    np.random.seed(3)
    x = np.random.rand()
    np.random.seed(3)
    pm.gextract("dense_track", iv)
    assert np.random.rand() == x
    np.random.seed(3)
    pm.gsample("dense_track", 5, iv)
    assert np.random.rand() != x
    # the private pm_seed seeds the same generator
    _pymisha.pm_seed(4)
    assert np.array_equal(pm.gsample("dense_track", 50, iv), s1)


def test_every_random_draw_follows_numpys_seed(monkeypatch):
    iv = pm.gintervals("chr1", 0, 100000)
    st = np.arange(0, 100000, 10)
    src = pd.DataFrame({"chrom": "chr1", "start": st, "end": st + 5, "value": st.astype(float)})
    created = []
    try:
        for name, source, func in (("rs_dense", "dense_track", "sample"), ("rs_df", src, "sample"),
                                   ("rs_avg", src, "avg"), ("rs_2d", "rects_track", "sample")):
            pm.gvtrack_create(name, source, func)
            created.append(name)
        # sample vtracks: C++ (track source) and value-based
        _check_seed_reproduces(lambda: pm.gextract("rs_dense", iv, iterator=1000))
        _check_seed_reproduces(lambda: pm.gextract("rs_df", iv, iterator=1000))
        # gsample of a vtrack expression (sampled in Python)
        _check_seed_reproduces(lambda: pm.gsample("rs_avg", 20, iv, iterator=100))
        # the 2D sample vtrack
        iv2 = pm.gintervals_2d("chr1", 0, 500000, "chr1", 0, 500000)
        _check_seed_reproduces(lambda: pm.gextract("rs_2d", iv2, iterator=(50000, 50000)))
        # sampled quantiles, in C++ and in Python (a vtrack expression)
        monkeypatch.setitem(_shared.CONFIG, "max_data_size", 50)
        _check_seed_reproduces(lambda: pm.gquantiles("dense_track", [0.3, 0.5, 0.9], iv))
        _check_seed_reproduces(lambda: pm.gquantiles("rs_avg * 2", [0.3, 0.5, 0.9], iv, iterator=100))
    finally:
        for name in created:
            pm.gvtrack_rm(name)
    # gsynth without a seed, gintervals_random
    _check_seed_reproduces(lambda: pm.gsynth_random(intervals=pm.gintervals("chr1", 0, 2000), output_format="vector"))
    _check_seed_reproduces(lambda: pm.gintervals_random(100, 5, dist_from_edge=1000))


def test_gintervals_random_cpp_path_follows_numpys_seed(monkeypatch):
    from pymisha import intervals as _intervals

    monkeypatch.setattr(_intervals, "_GINTERVALS_RANDOM_CPP_MIN_BP", 0)
    _check_seed_reproduces(lambda: pm.gintervals_random(100, 5, dist_from_edge=1000))


def test_2d_object_sample_follows_numpys_seed(rects_track):
    from pymisha.extract import _gextract_2d_vtrack_objects, _gextract_2d_vtrack_objects_python

    # each window holds the two objects valued 1 and 2
    n = 40
    iv = pd.DataFrame({"chrom1": ["1"] * n, "start1": [0] * n, "end1": [1000] * n,
                       "chrom2": ["1"] * n, "start2": [0] * n, "end2": [1000] * n})
    for fn in (_gextract_2d_vtrack_objects, _gextract_2d_vtrack_objects_python):
        _check_seed_reproduces(lambda fn=fn: fn(rects_track, "v", iv, None, "sample")["v"].to_numpy())


@pytest.fixture
def kid_points():
    """Ten points per 1 kb window, valued 0-9, on three chromosomes: a "sample" vtrack
    returns the index of the point it drew."""
    chroms = ["chr1", "chr2", "chrX"]
    bins = np.arange(0, 20000, 1000)
    rows = [(c, b + j * 100, b + j * 100 + 1, float(j)) for c in chroms for b in bins for j in range(10)]
    src = pd.DataFrame(rows, columns=["chrom", "start", "end", "value"])
    name = "test_kid_points"
    if pm.gtrack_exists(name):
        pm.gtrack_rm(name, force=True)
    pm.gtrack_create_sparse(name, "points", src[["chrom", "start", "end"]], src["value"].to_numpy())
    pm.gvtrack_create("kid_sample", name, "sample")
    yield chroms
    pm.gvtrack_rm("kid_sample")
    pm.gtrack_rm(name, force=True)


@pytest.fixture
def multitask(monkeypatch):
    for key, val in {"multitasking": True, "max_processes": 3, "min_processes": 3,
                     "min_scope4process": 1, "min_intervs4process": 0, "progress": False}.items():
        monkeypatch.setitem(_shared.CONFIG, key, val)


def _no_repeated_run(values, run=8):
    """True unless a run of `run` values at the start recurs later, which is what kids that
    draw the same numbers give (each kid's part starts the same stream again)."""
    head = values[:run]
    return all(values[a:a + run] != head for a in range(1, len(values) - run + 1))


def test_pool_workers_draw_different_random_numbers(kid_points, multitask):
    # gextract's worker processes: one per part of the scope
    chroms = kid_points
    scope = pm.gintervals(chroms, 0, 20000)
    r = _seeded(lambda: pm.gextract("kid_sample", scope, iterator=1000), 5)
    per = {c: g["kid_sample"].tolist() for c, g in r.groupby("chrom", observed=True)}
    # kids that drew the same numbers would sample the same points on every chromosome
    assert per["chr1"] != per["chr2"]
    assert per["chr2"] != per["chrX"]
    assert _no_repeated_run(r["kid_sample"].tolist())
    _check_seed_reproduces(lambda: pm.gextract("kid_sample", scope, iterator=1000))


def test_cpp_kids_draw_different_random_numbers(kid_points, multitask):
    # gscreen splits the scope among the kids the C++ layer forks; the value drawn in each
    # window is read back through ten screens under the same seed
    chroms = kid_points
    starts = [s for _ in chroms for s in (0, 5000, 10000, 15000)]
    scope = pm.gintervals([c for c in chroms for _ in range(4)], starts, [s + 4000 for s in starts])
    drawn = {}
    for v in range(10):
        r = _seeded(lambda v=v: pm.gscreen(f"kid_sample == {v}", scope, iterator=1000), 5)
        if r is not None:
            drawn.update({(c, int(s)): v for c, s in zip(r["chrom"].astype(str), r["start"], strict=True)})
    windows = [(c, s) for c in chroms for s0 in (0, 5000, 10000, 15000) for s in range(s0, s0 + 4000, 1000)]
    values = [drawn.get(w, -1) for w in windows]
    assert sum(v >= 0 for v in values) > 40
    assert _no_repeated_run(values)
    _check_seed_reproduces(lambda: pm.gscreen("kid_sample > 4", scope, iterator=1000))


def test_multitask_draws_follow_numpys_seed_like_single_task(kid_points, multitask, monkeypatch):
    chroms = kid_points
    scope = pm.gintervals(chroms, 0, 20000)
    for mt in (True, False):
        monkeypatch.setitem(_shared.CONFIG, "multitasking", mt)
        _check_seed_reproduces(lambda: pm.gextract("kid_sample", scope, iterator=1000))


def test_a_worker_that_dies_is_an_error():
    from pymisha._shared import _map_forked

    assert _map_forked(lambda x: x * 10, [1, 2, 3], 2) == [10, 20, 30]
    with pytest.raises(RuntimeError, match="ended unexpectedly"):
        _map_forked(lambda x: os.kill(os.getpid(), 9) if x == 2 else x, [1, 2, 3], 3)
    # the random draws of the workers differ, and the parent's seed reproduces them
    draws = _seeded(lambda: _map_forked(lambda _x: np.random.rand(), [0, 1, 2], 3), 9)
    assert len(set(draws)) == 3
    assert _seeded(lambda: _map_forked(lambda _x: np.random.rand(), [0, 1, 2], 3), 9) == draws


# ---------------------------------------------------------------------------
# gseq_kmer_dist
# ---------------------------------------------------------------------------


def test_kmer_dist_counts_overlapping_intervals_once():
    iv = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 500], "end": [1000, 1500]})
    result = pm.gseq_kmer_dist(iv, k=2)
    # the k-mers of chr1:0-1500, 1500 - 1 = 1499 dinucleotides
    assert result["count"].sum() == 1499
    pd.testing.assert_frame_equal(result, pm.gseq_kmer_dist(pm.gintervals("chr1", 0, 1500), k=2))
    # touching intervals stay apart: the k-mer across them is not counted
    touching = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 750], "end": [750, 1500]})
    assert pm.gseq_kmer_dist(touching, k=2)["count"].sum() == 1498


# ---------------------------------------------------------------------------
# gdb_create(db_format="per-chromosome") of a multi-FASTA file
# ---------------------------------------------------------------------------


def test_per_chromosome_gives_each_record_of_a_multi_fasta_a_chromosome(tmp_path):
    multi = tmp_path / "multi.fasta"
    multi.write_text(">chrX1 first record\nACGTACGTAC\n>chrX2\nGGGG\nCCCC\n")
    groot = tmp_path / "db"
    pm.gdb_create(str(groot), str(multi), db_format="per-chromosome")
    sizes = pd.read_csv(groot / "chrom_sizes.txt", sep="\t", header=None, names=["chrom", "size"])
    assert sizes["chrom"].tolist() == ["chrX1", "chrX2"]
    assert sizes["size"].tolist() == [10, 8]
    assert sorted(os.listdir(groot / "seq")) == ["chrX1.seq", "chrX2.seq"]
    pm.gdb_init(str(groot))
    assert pm.gseq_extract(pm.gintervals(["chrX1", "chrX2"], 0, [10, 8])) == ["ACGTACGTAC", "GGGGCCCC"]


def test_per_chromosome_names_a_single_record_file_after_the_file(tmp_path):
    multi = tmp_path / "multi.fasta"
    multi.write_text(">chrX1 first record\nACGTACGTAC\n>chrX2\nGGGG\nCCCC\n")
    single = tmp_path / "chrS.fa"
    single.write_text(">a header the per-chromosome format does not use\nTTTTAAAA\n")
    groot = tmp_path / "db"
    pm.gdb_create(str(groot), [str(multi), str(single)], db_format="per-chromosome")
    sizes = pd.read_csv(groot / "chrom_sizes.txt", sep="\t", header=None, names=["chrom", "size"])
    assert sizes["chrom"].tolist() == ["chrX1", "chrX2", "chrS"]
    assert sizes["size"].tolist() == [10, 8, 8]
    assert sorted(os.listdir(groot / "seq")) == ["chrS.seq", "chrX1.seq", "chrX2.seq"]
    pm.gdb_init(str(groot))
    assert pm.gseq_extract(pm.gintervals(["chrX1", "chrX2", "chrS"], 0, [10, 8, 8])) == \
        ["ACGTACGTAC", "GGGGCCCC", "TTTTAAAA"]


def test_per_chromosome_refuses_repeated_record_names(tmp_path):
    dup = tmp_path / "dup.fasta"
    dup.write_text(">a\nACGT\n>b\nGG\n>a\nTTTT\n")
    groot = tmp_path / "db"
    with pytest.raises(ValueError, match="Duplicate contig name 'a'"):
        pm.gdb_create(str(groot), str(dup), db_format="per-chromosome")
    assert not groot.exists()


# ---------------------------------------------------------------------------
# gsynth
# ---------------------------------------------------------------------------


@pytest.fixture
def g_frac():
    names = []

    def make(name):
        if name in pm.gvtrack_ls():
            pm.gvtrack_rm(name)
        pm.gvtrack_create(name, None, "kmer.frac", kmer="G")
        names.append(name)
        return name

    yield make
    for n in names:
        if n in pm.gvtrack_ls():
            pm.gvtrack_rm(n)


@pytest.fixture
def scratch_tracks():
    names = []
    yield names
    for t in names:
        if pm.gtrack_exists(t):
            pm.gtrack_rm(t, force=True)


def _values(track, iv, iterator):
    return pm.gextract(track, iv, iterator=iterator)[track].to_numpy()


def test_gsynth_train_leaves_out_kmers_that_reach_into_the_mask():
    k = 2
    iv = pm.gintervals("chr1", 1000, 61000)
    mask = pm.gintervals("chr1", [5000, 20000], [5100, 20007])
    model = pm.gsynth_train(intervals=iv, mask=mask, iterator=1000, k=k)

    # the forward and reverse-complement (k+1)-mer counts gseq_kmer_dist gives
    kd = pm.gseq_kmer_dist(iv, k=k + 1, mask=mask)
    expected = np.zeros((4 ** k, 4))
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    for w, n in zip(kd["kmer"], kd["count"], strict=True):
        x = [code[c] for c in w]
        for y in (x, [3 - c for c in reversed(x)]):
            ctx = sum(c * 4 ** (k - 1 - i) for i, c in enumerate(y[:k]))
            expected[ctx, y[k]] += n
    np.testing.assert_array_equal(np.asarray(model.model_data["counts"][0]), expected)
    assert model.total_kmers == 2 * kd["count"].sum()


def test_gsynth_train_counts_overlapping_intervals_once(g_frac):
    one = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 4000), iterator=100, k=2)
    overlapping = pm.gsynth_train(intervals=pm.gintervals(["chr1", "chr1"], [0, 1000], [3000, 4000]),
                                  iterator=100, k=2)
    assert overlapping.total_kmers == one.total_kmers
    np.testing.assert_array_equal(np.asarray(overlapping.model_data["counts"]), np.asarray(one.model_data["counts"]))

    g = g_frac("g_frac_ov")
    spec = {"expr": g, "breaks": [0, 0.25, 1]}
    one = pm.gsynth_train(spec, intervals=pm.gintervals("chr1", 0, 4000), iterator=100, k=2)
    overlapping = pm.gsynth_train(spec, intervals=pm.gintervals(["chr1", "chr1"], [0, 1000], [3000, 4000]),
                                  iterator=100, k=2)
    np.testing.assert_array_equal(np.asarray(overlapping.per_bin_kmers), np.asarray(one.per_bin_kmers))
    np.testing.assert_array_equal(np.asarray(overlapping.model_data["counts"]), np.asarray(one.model_data["counts"]))


def test_gsynth_random_draws_every_base_from_nuc_probs():
    only_a = {"A": 1, "C": 0, "G": 0, "T": 0}
    seqs = pm.gsynth_random(intervals=pm.gintervals("chr1", np.arange(0, 9001, 1000), np.arange(20, 9021, 1000)),
                            output_format="vector", nuc_probs=only_a, seed=1)
    assert seqs == ["A" * 20] * 10

    # around the N run at chr1:167280-217280, which preserve_n keeps
    iv = pm.gintervals("chr1", 167200, 217400)
    s = pm.gsynth_random(intervals=iv, output_format="vector", nuc_probs=only_a, seed=1)[0]
    ref = pm.gseq_extract(iv)[0]
    assert s == "".join(c if c in "Nn" else "A" for c in ref)
    assert "N" in s


def test_gsynth_replace_kmer_refuses_a_replacement_equal_to_the_target_in_another_case():
    # R replaced every CG by CG forever
    with pytest.raises(ValueError, match="cannot be identical"):
        pm.gsynth_replace_kmer("cg", "CG", output_format="vector", intervals=pm.gintervals("chr1", 0, 200),
                               check_composition=False)
    # the composition check ignores case too
    iv = pm.gintervals("chr1", 0, 2000)
    assert pm.gsynth_replace_kmer("cg", "GC", output_format="vector", intervals=iv) == \
        pm.gsynth_replace_kmer("CG", "GC", output_format="vector", intervals=iv)


def test_gsynth_sample_takes_the_bins_of_an_interval_from_its_own_windows(g_frac):
    g = g_frac("g_frac_own")
    iv1 = pm.gintervals("chr1", 250, 1250)
    iv2 = pm.gintervals("chr1", 1100, 1400)
    # the window of iv1 at 1000-1250 and that of iv2 at 1100-1400 fall in different bins
    v1 = _values(g, pm.gintervals("chr1", 1000, 1250), 500)[0]
    v2 = _values(g, iv2, 500)[0]
    assert v1 != v2
    model = pm.gsynth_train({"expr": g, "breaks": [0, (v1 + v2) / 2, 1]},
                            intervals=pm.gintervals("chr1", 0, 50000), iterator=500, k=2)
    # bin 1 emits only A, bin 2 only C, so the output shows the bin of each position
    model.model_data["cdf"][0][:] = 1
    model.model_data["cdf"][1][:] = np.array([0, 1, 1, 1])
    alone = pm.gsynth_sample(model, output_format="vector", intervals=iv1, seed=1)
    # iv1 is sampled first, so it draws the same random numbers
    with_iv2 = pm.gsynth_sample(model, output_format="vector", intervals=pd.concat([iv2, iv1], ignore_index=True),
                                seed=1)
    assert with_iv2[0] == alone[0]


def test_gsynth_outputs_the_intervals_of_a_chromosome_sorted():
    # as R: chr1:0-2, chr1:250-255, chr1:1100-1110, chr2:0-3
    iv = pd.concat([pm.gintervals("chr1", 1100, 1110), pm.gintervals("chr1", 250, 255),
                    pm.gintervals("chr2", 0, 3), pm.gintervals("chr1", 0, 2)], ignore_index=True)
    assert [len(x) for x in pm.gsynth_random(intervals=iv, output_format="vector", seed=1)] == [2, 5, 10, 3]
    replaced = pm.gsynth_replace_kmer("CG", "GC", intervals=iv, output_format="vector")
    assert [len(x) for x in replaced] == [2, 5, 10, 3]
    model = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 20000), iterator=200, k=2)
    assert [len(x) for x in pm.gsynth_sample(model, output_format="vector", intervals=iv, seed=1)] == [2, 5, 10, 3]


def _read_fasta(path):
    lines = Path(path).read_text().splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith(">")]
    return [(lines[h][1:], "".join(lines[h + 1:(heads[k + 1] if k + 1 < len(heads) else len(lines))]))
            for k, h in enumerate(heads)]


@pytest.mark.parametrize("n_samples", [1, 2])
def test_parallel_gsynth_sample_fasta_is_laid_out_as_the_serial_one(tmp_path, n_samples):
    model = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 20000), iterator=200, k=2)
    # unsorted, one on a whole chromosome, one at a chromosome end
    iv = pd.DataFrame({"chrom": ["chr1", "chrX", "chr1", "chr2"], "start": [9000, 0, 0, 299000],
                       "end": [18000, 200000, 9000, 300000]})
    serial, parallel = tmp_path / "serial.fa", tmp_path / "parallel.fa"
    pm.gsynth_sample(model, str(serial), output_format="fasta", intervals=iv, seed=3, n_samples=n_samples,
                     allow_parallel=False)
    pm.gsynth_sample(model, str(parallel), output_format="fasta", intervals=iv, seed=3, n_samples=n_samples,
                     allow_parallel=True, num_cores=4, max_chunk_size=1000)
    # the same records, lengths and layout as the serial output: the same index
    assert Path(str(parallel) + ".fai").read_text() == Path(str(serial) + ".fai").read_text()
    s_rec, p_rec = _read_fasta(serial), _read_fasta(parallel)
    assert [h for h, _ in p_rec] == [h for h, _ in s_rec]
    assert [len(x) for _, x in p_rec] == [len(x) for _, x in s_rec]
    assert Path(parallel).read_text().count("\n") == Path(serial).read_text().count("\n")
    # the records are the sequences the vector output gives with the same seed
    vec = pm.gsynth_sample(model, output_format="vector", intervals=iv, seed=3, n_samples=n_samples,
                           allow_parallel=True, num_cores=4, max_chunk_size=1000)
    assert [x for _, x in p_rec] == vec
    # each offset points at the first base of its record
    raw = Path(parallel).read_bytes()
    for line, (_, x) in zip(Path(str(parallel) + ".fai").read_text().splitlines(), p_rec, strict=True):
        _, length, offset, linebases, _ = line.split("\t")
        assert raw[int(offset):int(offset) + int(linebases)].decode() == x[:int(linebases)]


def test_gsynth_score_scores_a_position_covered_by_overlapping_intervals_once(scratch_tracks):
    scratch_tracks += ["zz_ovl_a", "zz_ovl_b"]
    model = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 50000), iterator=200, k=2)
    iv = pm.gintervals("chr1", 1000, 1300)
    pm.gsynth_score(model, "zz_ovl_a", intervals=iv, resolution=1)
    pm.gsynth_score(model, "zz_ovl_b", intervals=pd.concat([iv, pm.gintervals("chr1", 1200, 1300)],
                                                          ignore_index=True), resolution=1)
    a = _values("zz_ovl_a", iv, 1)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(_values("zz_ovl_b", iv, 1), a)


def test_gsynth_score_of_an_interval_does_not_depend_on_the_others(g_frac, scratch_tracks):
    scratch_tracks += ["zz_nb_a", "zz_nb_b"]
    g = g_frac("g_frac_nb")
    iv = pm.gintervals("chr1", 1000, 1300)
    v1 = _values(g, iv, 500)[0]
    # A neighbor at 1400-1500 has, extended upstream by an iterator, a stratum window at
    # 1000-1500. Stratify so that this window falls in a higher bin than the window of iv at
    # 1000-1300: the two windows start at the same position.
    neighbor = pm.gintervals("chr1", 1400, 1500)
    v2 = _values(g, pm.gintervals("chr1", 1000, 1500), 500)[0]
    assert v1 != v2
    sgn = 1 if v2 > v1 else -1
    spec = {"expr": g if sgn > 0 else f"-{g}", "breaks": sorted(sgn * np.array([0, (v1 + v2) / 2, 1]))}
    model = pm.gsynth_train(spec, intervals=pm.gintervals("chr1", 0, 50000), iterator=500, k=2)
    pm.gsynth_score(model, "zz_nb_a", intervals=iv, resolution=1)
    pm.gsynth_score(model, "zz_nb_b", intervals=pd.concat([iv, neighbor], ignore_index=True), resolution=1)
    a = _values("zz_nb_a", iv, 1)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(_values("zz_nb_b", iv, 1), a)


def test_a_loaded_model_samples_and_scores_like_the_model_it_was_saved_from(g_frac, scratch_tracks, tmp_path):
    scratch_tracks += ["zz_gsm_score_a", "zz_gsm_score_b"]
    g = g_frac("g_frac_it")
    model = pm.gsynth_train({"expr": g, "breaks": [0, 0.2, 0.25, 0.3, 1]},
                            intervals=pm.gintervals("chr1", 0, 50000), iterator=500, k=3)
    for compress, path in ((False, tmp_path / "test_gsm_iter"), (True, tmp_path / "test_gsm_iter.zip")):
        pm.gsynth_save(model, str(path), compress=compress)
        loaded = pm.gsynth_load(str(path))
        assert loaded.iterator == 500
        iv = pm.gintervals("chr1", 1234, 9876)
        assert pm.gsynth_sample(loaded, output_format="vector", intervals=iv, seed=17) == \
            pm.gsynth_sample(model, output_format="vector", intervals=iv, seed=17)

    iv = pm.gintervals("chr1", 1000, 6000)
    pm.gsynth_score(model, "zz_gsm_score_a", intervals=iv, resolution=100)
    pm.gsynth_score(loaded, "zz_gsm_score_b", intervals=iv, resolution=100)
    a = _values("zz_gsm_score_a", iv, 100)
    assert np.isfinite(a).any()
    np.testing.assert_array_equal(_values("zz_gsm_score_b", iv, 100), a)

    meta = yaml.safe_load((tmp_path / "test_gsm_iter" / "metadata.yaml").read_text())
    assert meta["iterator"] == 500


def _old_save(model, path):
    """A .gsm written by a gsynth_save that did not store the iterator."""
    pm.gsynth_save(model, str(path))
    meta_path = Path(path) / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta.pop("iterator", None)
    meta_path.write_text(yaml.safe_dump(meta))


def test_a_model_saved_without_its_iterator_asks_for_it(g_frac, tmp_path):
    g = g_frac("g_frac_noit")
    model = pm.gsynth_train({"expr": g, "breaks": [0, 0.25, 1]}, intervals=pm.gintervals("chr1", 0, 20000),
                            iterator=200, k=2)
    _old_save(model, tmp_path / "noiter")
    loaded = pm.gsynth_load(str(tmp_path / "noiter"))
    assert loaded.iterator is None
    iv = pm.gintervals("chr1", 0, 1000)
    with pytest.raises(ValueError, match=r"model\.iterator = "):
        pm.gsynth_sample(loaded, output_format="vector", intervals=iv, seed=1)
    with pytest.raises(ValueError, match=r"model\.iterator = "):
        pm.gsynth_score(loaded, "zz_gsm_noiter", intervals=iv)
    assert not pm.gtrack_exists("zz_gsm_noiter")
    loaded.iterator = 200
    assert pm.gsynth_sample(loaded, output_format="vector", intervals=iv, seed=1) == \
        pm.gsynth_sample(model, output_format="vector", intervals=iv, seed=1)

    # every position of an unstratified model is in its single bin, whatever the iterator
    model0 = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 20000), iterator=200, k=2)
    _old_save(model0, tmp_path / "noiter0")
    loaded0 = pm.gsynth_load(str(tmp_path / "noiter0"))
    assert loaded0.iterator is None
    assert pm.gsynth_sample(loaded0, output_format="vector", intervals=iv, seed=1) == \
        pm.gsynth_sample(model0, output_format="vector", intervals=iv, seed=1)
    with pytest.raises(ValueError, match="or pass resolution"):
        pm.gsynth_score(loaded0, "zz_gsm_noiter0", intervals=iv)


def test_a_model_trained_with_the_implicit_iterator_of_a_dense_track(scratch_tracks, tmp_path):
    scratch_tracks += ["zz_impl_a", "zz_impl_b"]
    if "dense_avg_it" in pm.gvtrack_ls():
        pm.gvtrack_rm("dense_avg_it")
    pm.gvtrack_create("dense_avg_it", "dense_track", "avg")
    try:
        rng = pm.gsummary("dense_track", intervals=pm.gintervals("chr1", 0, 50000))
        breaks = np.linspace(float(rng["Min"]), float(rng["Max"]), 4).tolist()
        binsize = int(pm.gtrack_info("dense_track")["bin_size"])
        spec = {"expr": "dense_avg_it", "breaks": breaks}

        # no iterator: the bins are those of dense_track
        model = pm.gsynth_train(spec, intervals=pm.gintervals("chr1", 0, 50000), k=2)
        assert model.iterator == binsize
        explicit = pm.gsynth_train(spec, intervals=pm.gintervals("chr1", 0, 50000), iterator=binsize, k=2)
        np.testing.assert_array_equal(np.asarray(model.model_data["counts"]), np.asarray(explicit.model_data["counts"]))

        iv = pm.gintervals("chr1", 1000, 5000)
        s = pm.gsynth_sample(model, output_format="vector", intervals=iv, seed=2)
        assert s == pm.gsynth_sample(explicit, output_format="vector", intervals=iv, seed=2)

        pm.gsynth_save(model, str(tmp_path / "implicit"))
        loaded = pm.gsynth_load(str(tmp_path / "implicit"))
        assert loaded.iterator == binsize
        assert pm.gsynth_sample(loaded, output_format="vector", intervals=iv, seed=2) == s

        pm.gsynth_score(model, "zz_impl_a", intervals=iv)
        pm.gsynth_score(loaded, "zz_impl_b", intervals=iv)
        a = _values("zz_impl_a", iv, binsize)
        assert np.isfinite(a).any()
        np.testing.assert_array_equal(_values("zz_impl_b", iv, binsize), a)
    finally:
        pm.gvtrack_rm("dense_avg_it")


def test_a_gsm_without_a_prior_key_loads(tmp_path):
    # misha #226: R matched metadata$prior to prior_mode; pymisha wrote no prior key before 5.17
    m = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 20000), iterator=200, k=2)
    f = tmp_path / "m_noprior"
    pm.gsynth_save(m, str(f))
    meta_path = f / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    assert len(meta["prior"]) == m.total_bins  # written for R, as R writes it
    meta.pop("prior")
    meta["data"].pop("prior")
    meta_path.write_text(yaml.safe_dump(meta))
    loaded = pm.gsynth_load(str(f))
    assert loaded.k == m.k
    # a uniform prior is backfilled, as R does; the model saves again with it
    np.testing.assert_array_equal(loaded.prior_matrix, np.full((m.total_bins, 4), 0.25))
    assert loaded.prior_mode == m.prior_mode
    pm.gsynth_save(loaded, str(tmp_path / "resaved"))
    np.testing.assert_array_equal(pm.gsynth_load(str(tmp_path / "resaved")).prior_matrix, loaded.prior_matrix)


def test_a_legacy_model_without_a_prior_loads_with_a_uniform_prior_and_saves_again(tmp_path):
    # R's "an RDS model saved before priors existed" case, for pymisha's legacy pickle format
    import pickle

    m = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 20000), iterator=200, k=2)
    del m.__dict__["prior_mode"], m.__dict__["prior_matrix"]
    old = tmp_path / "old_model.pkl"
    old.write_bytes(pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL))
    loaded = pm.gsynth_load(str(old))
    assert loaded.prior_mode == "uniform"
    np.testing.assert_array_equal(loaded.prior_matrix, np.full((m.total_bins, 4), 0.25))
    pm.gsynth_save(loaded, str(tmp_path / "resaved"))
    # printing a model with a non-uniform prior_mode but no prior does not fail
    loaded.prior_mode, loaded.prior_matrix = "custom", None
    assert "Prior: custom" in repr(loaded)


def test_a_gsm_with_a_prior_of_the_wrong_shape_is_an_error(tmp_path):
    m = pm.gsynth_train(intervals=pm.gintervals("chr1", 0, 20000), iterator=200, k=2)
    f = tmp_path / "badprior"
    pm.gsynth_save(m, str(f))
    meta_path = f / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["data"].pop("prior")
    meta["prior"] = [[0.5, 0.5]]
    meta_path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="Invalid prior in .gsm: got 1 x 2, expected 1 x 4"):
        pm.gsynth_load(str(f))


def test_a_gsm_written_by_r_with_one_dimension_loads(g_frac, tmp_path):
    # R writes a vector of length 1 (dim_sizes of one dimension) as a YAML scalar, and the
    # prior in metadata.yaml only
    g = g_frac("g_frac_r")
    m = pm.gsynth_train({"expr": g, "breaks": [0, 0.25, 1]}, intervals=pm.gintervals("chr1", 0, 20000),
                        iterator=200, k=2)
    f = tmp_path / "r_like"
    pm.gsynth_save(m, str(f))
    meta_path = f / "metadata.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    meta["dim_sizes"] = meta["dim_sizes"][0]
    meta["data"].pop("prior")
    (f / "prior.bin").unlink()
    meta_path.write_text(yaml.safe_dump(meta))
    loaded = pm.gsynth_load(str(f))
    assert loaded.dim_sizes == [2]
    np.testing.assert_allclose(loaded.prior_matrix, m.prior_matrix)
    assert loaded.iterator == 200


def test_a_file_output_of_several_parts_draws_what_the_in_memory_extraction_draws(tmp_path, monkeypatch):
    # gextract(file=) runs a call per part of the scope; the parts draw one random stream, so a
    # seed gives the in-memory values, as R's single process does
    from pymisha import extract as _extract

    monkeypatch.setitem(_shared.CONFIG, "multitasking", False)
    monkeypatch.setitem(_shared.CONFIG, "progress", False)
    pm.gvtrack_create("vsamp_file", "dense_track", func="sample")
    try:
        whole = _seeded(lambda: pm.gextract("vsamp_file", pm.gintervals_all(), iterator=100), 60427)
        monkeypatch.setattr(_extract, "_FILE_PART_ROWS", 500)  # 20 parts
        _seeded(lambda: pm.gextract("vsamp_file", pm.gintervals_all(), iterator=100,
                                    file=str(tmp_path / "parts.tsv")), 60427)
        got = pd.read_csv(tmp_path / "parts.tsv", sep="\t")
        assert len(got) == len(whole)
        # the file holds the values as text, in fewer digits
        np.testing.assert_allclose(got["vsamp_file"].to_numpy(dtype=float), whole["vsamp_file"].to_numpy(dtype=float),
                                   rtol=1e-6)
        # a call after the file extraction seeds afresh from numpy's generator
        np.random.seed(5)
        a = pm.gsample("dense_track", 20, pm.gintervals("chr1", 0, 100000))
        np.random.seed(5)
        assert np.array_equal(pm.gsample("dense_track", 20, pm.gintervals("chr1", 0, 100000)), a)
    finally:
        pm.gvtrack_rm("vsamp_file")


def test_a_2d_file_output_of_several_pairs_draws_what_the_in_memory_extraction_draws(tmp_path, monkeypatch):
    monkeypatch.setitem(_shared.CONFIG, "multitasking", False)
    monkeypatch.setitem(_shared.CONFIG, "progress", False)
    pm.gvtrack_create("vsamp_2d", "rects_track", func="sample")
    try:
        scope = pm.gintervals_2d(["chr1"] * 3, [0] * 3, [500000] * 3, ["chr1", "chr2", "chrX"], [0] * 3,
                                 [500000, 300000, 200000])
        cols = ["vsamp_2d"]
        whole = _seeded(lambda: pm.gextract("vsamp_2d", scope, iterator=(20000, 20000)), 60427)
        _seeded(lambda: pm.gextract("vsamp_2d", scope, iterator=(20000, 20000), file=str(tmp_path / "pairs.tsv")),
                60427)
        got = pd.read_csv(tmp_path / "pairs.tsv", sep="\t")
        assert len(got) == len(whole) and len(got.groupby(["chrom1", "chrom2"])) == 3
        np.testing.assert_allclose(got[cols].to_numpy(dtype=float), whole[cols].to_numpy(dtype=float), rtol=1e-6)
    finally:
        pm.gvtrack_rm("vsamp_2d")


def test_kids_of_the_parts_of_a_file_output_draw_their_own_numbers(tmp_path, kid_points, multitask, monkeypatch):
    # within one random stream, the kids of each part's call draw numbers of their own, and the
    # seed reproduces the file
    from pymisha import extract as _extract

    chroms = kid_points
    monkeypatch.setattr(_extract, "_FILE_PART_ROWS", 20)
    scope = pm.gintervals(chroms, 0, 20000)
    for name in ("a.tsv", "b.tsv"):
        _seeded(lambda name=name: pm.gextract("kid_sample", scope, iterator=1000, file=str(tmp_path / name)), 5)
    assert (tmp_path / "a.tsv").read_bytes() == (tmp_path / "b.tsv").read_bytes()
    values = pd.read_csv(tmp_path / "a.tsv", sep="\t")["kid_sample"].astype(int).tolist()
    assert len(values) == 60
    assert _no_repeated_run(values)
