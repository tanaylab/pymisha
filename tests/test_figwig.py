"""gtrack_import_figwig and gtrack_export_figwig. The figwig files are written by the figwig package
itself, which these tests need; without it they are skipped (but for the error of a missing figwig)."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

SIZES = {"chr1": 1000, "chr2": 600}


@pytest.fixture()
def figwig():
    return pytest.importorskip("figwig")


def _mk_db(root: Path, sizes: dict[str, int], db_format: str = "per-chromosome") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fastas = []
    for chrom, size in sizes.items():
        fasta = root / f"{chrom}.fasta"
        fasta.write_text(f">{chrom}\n{'A' * size}\n")
        fastas.append(str(fasta))
    db = root / "db"
    pm.gdb_create(str(db), fastas, db_format=db_format)
    pm.gdb_init(str(db))
    return db


@pytest.fixture(params=["per-chromosome", "indexed"])
def db(request, tmp_path):
    return _mk_db(tmp_path / "fwdb", SIZES, request.param)


@pytest.fixture()
def db1(tmp_path):
    return _mk_db(tmp_path / "fwdb", SIZES)


# The tracks of the test files: name, figwig dtype, and a function of (rng, size) giving the values of
# a chromosome. Between them they make figwig store every element type (u1 u2 u4 i1 i2 i4 f4 f8 and
# the "one" code) and dense, sparse, runs and matrix chunks.
def _counts(rng, n):
    return rng.poisson(0.05, n).astype(np.float64)


def _runs_u16(rng, n):
    out = np.zeros(n)
    pos = 0
    while pos < n:
        length = int(rng.integers(5, 60))
        out[pos:pos + length] = rng.choice([0, 300, 1000, 65535])
        pos += length
    return out


def _sparse_u32(rng, n):
    out = np.zeros(n)
    idx = rng.choice(n, 6, replace=False)
    out[idx] = rng.integers(2**24 + 1, 2**32 - 1, 6)
    return out


def _sparse_i8(rng, n):
    out = np.zeros(n)
    idx = rng.choice(n, 30, replace=False)
    out[idx] = rng.integers(-128, 128, 30)
    return out


def _dense_i16(rng, n):
    return rng.integers(-32768, 32768, n).astype(np.float64)


def _dense_i32(rng, n):
    return rng.integers(-(2**31), 2**31, n).astype(np.float64)


def _f32_nan(rng, n):
    out = rng.random(n).astype(np.float32).astype(np.float64)
    out[n // 3:n // 3 + 40] = np.nan
    out[n // 2:n // 2 + 30] = 0
    return out


def _f64(rng, n):
    return rng.normal(0, 1e6, n)


def _ones(rng, n):
    out = np.zeros(n)
    out[rng.choice(n, 25, replace=False)] = 1
    return out


TRACKS = [
    ("cuts.+", "uint8", _counts),
    ("cuts.-", None, _ones),
    ("runs-rep1", "uint16", _runs_u16),
    ("big", "uint32", _sparse_u32),
    ("neg", "int8", _sparse_i8),
    ("i16", "int16", _dense_i16),
    ("i32", "int32", _dense_i32),
    ("f32", "float32", _f32_nan),
    ("f64", "float64", _f64),
]


def _values(seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return {name: {c: f(rng, n) for c, n in SIZES.items()} for name, _, f in TRACKS}


def _write_fw(figwig, path: Path, values, order="positions", groups=4, level=None, tile=16,
              chrom_names=None, sizes=None, attrs=None):
    sizes = sizes or SIZES
    chrom_names = chrom_names or {c: c for c in sizes}
    tracks = []
    for name, dtype, _ in TRACKS:
        rec = {"name": name, "assay": "ATAC" if name.startswith("cuts") else "other", "rep": 1}
        if dtype is not None:
            rec["dtype"] = dtype
        rec.update((attrs or {}).get(name, {}))
        tracks.append(rec)
    with figwig.FigWigWriter(str(path), {chrom_names[c]: s for c, s in sizes.items()}, tracks, tile=tile,
                             groups=groups, order=order, level=level, genome="test") as w:
        for c, n in sizes.items():
            vals = np.stack([values[name][c][:n] for name, _, _ in TRACKS])[None]
            w.write([chrom_names[c]], [0], vals)
    return path


def _expected_sparse(x: np.ndarray) -> list[tuple[int, int, float]]:
    """Runs of equal float32 values that are neither 0 nor NaN, base by base."""
    x32 = x.astype(np.float32)
    res = []
    i = 0
    while i < len(x32):
        j = i + 1
        while j < len(x32) and x32[j] == x32[i]:
            j += 1
        if x32[i] != 0 and not np.isnan(x32[i]):
            res.append((i, j, float(x32[i])))
        i = j
    return res


def _expected_dense(x: np.ndarray, size: int, binsize: int, defval: float) -> np.ndarray:
    """Per bin of a chromosome of *size*: the mean of the non-NaN values of x (the values of the
    first len(x) bases), the other bases counting as defval unless it is NaN."""
    nb = -(-size // binsize)
    out = np.empty(nb, dtype=np.float32)
    for k in range(nb):
        lo, hi = k * binsize, min((k + 1) * binsize, size)
        vals = list(x[lo:min(hi, len(x))])
        vals = [v for v in vals if not math.isnan(v)]
        if not math.isnan(defval):
            vals += [defval] * ((hi - lo) - len(vals))
        out[k] = math.fsum(vals) / len(vals) if vals else np.nan
    return out


def _sparse_of(track: str, chrom: str) -> list[tuple[int, int, float]]:
    df = pm.gextract(track, pm.gintervals(chrom, 0, SIZES[chrom]), progress=False)
    if df is None:
        return []
    return [(int(s), int(e), float(v)) for s, e, v in zip(df["start"], df["end"], df[track], strict=True)]


def _dense_of(track: str, chrom: str) -> np.ndarray:
    df = pm.gextract(track, pm.gintervals(chrom, 0, SIZES[chrom]), progress=False)
    return df[track].to_numpy(dtype=np.float32)


@pytest.mark.parametrize("order,groups,level", [("positions", 4, None), ("positions", 9, 0), ("positions", 1, 6),
                                                ("tracks", 1, None), ("tracks", 1, 0)])
def test_import_sparse_every_dtype_and_order(figwig, db, tmp_path, order, groups, level):
    values = _values()
    fw = _write_fw(figwig, tmp_path / "x.fw", values, order=order, groups=groups, level=level)
    created = pm.gtrack_import_figwig(str(fw))
    assert created == ["cuts_plus", "cuts_minus", "runs_rep1", "big", "neg", "i16", "i32", "f32", "f64"]
    for (name, _, _), track in zip(TRACKS, created, strict=True):
        assert pm.gtrack_info(track)["type"] == "sparse"
        for c in SIZES:
            assert _sparse_of(track, c) == _expected_sparse(values[name][c]), (track, c)


@pytest.mark.parametrize("binsize,defval", [(7, np.nan), (16, 0.5), (1, np.nan), (2000, -1.0)])
def test_import_dense(figwig, db, tmp_path, binsize, defval):
    values = _values(1)
    fw = _write_fw(figwig, tmp_path / "x.fw", values)
    created = pm.gtrack_import_figwig(str(fw), binsize=binsize, defval=defval, track_prefix="d_")
    for (name, _, _), track in zip(TRACKS, created, strict=True):
        info = pm.gtrack_info(track)
        assert info["type"] == "dense" and int(info["bin_size"]) == binsize
        for c, n in SIZES.items():
            np.testing.assert_allclose(_dense_of(track, c), _expected_dense(values[name][c], n, binsize, defval),
                                       rtol=2e-7, atol=0, equal_nan=True, err_msg=f"{track} {c}")


def test_import_dense_matches_gtrack_import_of_the_same_records(figwig, db1, tmp_path):
    # the bins are those gtrack_import makes from a bedGraph of the same per-base values (a NaN base
    # is a gap in the bedGraph)
    values = _values(2)
    fw = _write_fw(figwig, tmp_path / "x.fw", values)
    for defval in [0.0, np.nan]:
        prefix = "z" if defval == 0 else "n"
        pm.gtrack_import_figwig(str(fw), binsize=13, tracks=["f32", "i16"], defval=defval, track_prefix=prefix)
        for name in ["f32", "i16"]:
            rows = []
            for c in SIZES:
                x = values[name][c]
                rows += [f"{c}\t{i}\t{i + 1}\t{float(x[i])!r}" for i in range(len(x)) if not np.isnan(x[i])]
            bg = tmp_path / f"{name}.bedgraph"
            bg.write_text("\n".join(rows) + "\n")
            pm.gtrack_import(f"bg{prefix}_{name}", "", str(bg), binsize=13, defval=defval)
            for c in SIZES:
                np.testing.assert_allclose(_dense_of(f"{prefix}{name}", c), _dense_of(f"bg{prefix}_{name}", c),
                                           rtol=1e-7, equal_nan=True)


def test_selection_by_names_indices_and_attributes(figwig, db1, tmp_path):
    fw = _write_fw(figwig, tmp_path / "x.fw", _values(),
                   attrs={"f32": {"assay": "ATAC", "rep": 2}, "big": {"rep": "2"}})
    assert pm.gtrack_import_figwig(str(fw), tracks=["f64", "neg"]) == ["f64", "neg"]
    assert pm.gtrack_import_figwig(str(fw), tracks=[0, 3], track_prefix="i_") == ["i_cuts_plus", "i_big"]
    assert pm.gtrack_import_figwig(str(fw), select={"assay": "ATAC"}, track_prefix="s_") == \
        ["s_cuts_plus", "s_cuts_minus", "s_f32"]
    # compared as text: 2 matches 2 and "2"
    assert pm.gtrack_import_figwig(str(fw), select={"rep": 2}, track_prefix="r_") == ["r_big", "r_f32"]
    assert pm.gtrack_import_figwig(str(fw), select={"assay": "ATAC", "rep": [1, 3]}, track_prefix="m_") == \
        ["m_cuts_plus", "m_cuts_minus"]
    assert pm.gtrack_import_figwig(str(fw), tracks=["f32", "cuts.+", "i16"], select={"assay": "ATAC"},
                                   track_prefix="t_") == ["t_f32", "t_cuts_plus"]
    assert pm.gtrack_import_figwig(str(fw), select={"dtype": "int8"}, track_prefix="dt_") == ["dt_neg"]
    with pytest.raises(ValueError, match="no track"):
        pm.gtrack_import_figwig(str(fw), tracks=["nope"])
    with pytest.raises(ValueError, match="out of range"):
        pm.gtrack_import_figwig(str(fw), tracks=[9])
    with pytest.raises(ValueError, match="more than once"):
        pm.gtrack_import_figwig(str(fw), tracks=["f32", 7])
    with pytest.raises(ValueError, match="matches"):
        pm.gtrack_import_figwig(str(fw), select={"assay": "ChIP"})
    with pytest.raises(ValueError, match="has the attribute"):
        pm.gtrack_import_figwig(str(fw), select={"cell": "K562"})


def test_names_sanitized_overridden_and_checked(figwig, db1, tmp_path):
    fw = _write_fw(figwig, tmp_path / "x.fw", _values())
    assert pm.gtrack_import_figwig(str(fw), tracks=["cuts.+", "cuts.-", "runs-rep1"]) == \
        ["cuts_plus", "cuts_minus", "runs_rep1"]
    assert pm.gtrack_import_figwig(str(fw), tracks=["cuts.+", "f32"], names={"cuts.+": "plus"},
                                   track_prefix="n_") == ["n_plus", "n_f32"]
    assert pm.gtrack_import_figwig(str(fw), tracks=["neg", "big"], names=["a", "b"]) == ["a", "b"]
    # the track exists: nothing is created
    before = set(pm.gtrack_ls())
    with pytest.raises(ValueError, match="already exists|exists"):
        pm.gtrack_import_figwig(str(fw), tracks=["i16", "cuts.+"])
    assert set(pm.gtrack_ls()) == before
    with pytest.raises(ValueError, match="same track name"):
        pm.gtrack_import_figwig(str(fw), tracks=["i16", "i32"], names=["x", "x"])
    with pytest.raises(ValueError, match="Invalid track name"):
        pm.gtrack_import_figwig(str(fw), tracks=["i16"], names=["1x"])
    with pytest.raises(ValueError, match="does not exist"):
        pm.gtrack_import_figwig(str(fw), tracks=["i16"], track_prefix="nodir.")
    with pytest.raises(ValueError, match="element"):
        pm.gtrack_import_figwig(str(fw), tracks=["i16"], names=["a", "b"])
    assert set(pm.gtrack_ls()) == before


def test_sanitize_rules():
    from pymisha.figwig import _sanitize_name

    assert _sanitize_name("atac.+", False) == "atac_plus"
    assert _sanitize_name("atac.-", False) == "atac_minus"
    assert _sanitize_name("H3K27ac-rep1", False) == "H3K27ac_rep1"
    assert _sanitize_name("ENCFF001 (rep 2)", False) == "ENCFF001_rep_2_"
    assert _sanitize_name("1abc", False) == "X1abc"
    assert _sanitize_name("1abc", True) == "1abc"
    assert _sanitize_name("+", False) == "plus"
    assert _sanitize_name("café.x", False) == "caf_x"


def test_attributes_copied(figwig, db1, tmp_path):
    fw = _write_fw(figwig, tmp_path / "x.fw", _values(), attrs={
        "f32": {"description": "from the file", "nested": {"k": [1, 2.5], "s": "é"}, "flag": True,
                "none": None, "empty": "", "created.by": "someone", "binsize": 10, "type": "x"},
    })
    pm.gtrack_import_figwig(str(fw), tracks=["f32", "i16"])
    assert pm.gtrack_attr_get("f32", "description") == "from the file"
    assert pm.gtrack_attr_get("f32", "nested") == '{"k": [1, 2.5], "s": "\\u00e9"}'
    assert pm.gtrack_attr_get("f32", "flag") == "true"
    assert pm.gtrack_attr_get("f32", "rep") == "1"
    assert pm.gtrack_attr_get("f32", "assay") == "other"
    for absent in ["none", "empty", "type", "binsize", "name", "dtype", "group"]:
        assert pm.gtrack_attr_get("f32", absent) == ""
    assert pm.gtrack_attr_get("f32", "created.by").startswith("gtrack.import_figwig(")
    assert pm.gtrack_attr_get("i16", "description") == ""
    # a description given replaces the file's, and attrs win over the file's
    pm.gtrack_import_figwig(str(fw), tracks=["f32"], track_prefix="d_", description="mine",
                            attrs={"assay": "ATAC-seq", "lab": "x"})
    assert pm.gtrack_attr_get("d_f32", "description") == "mine"
    assert pm.gtrack_attr_get("d_f32", "assay") == "ATAC-seq"
    assert pm.gtrack_attr_get("d_f32", "lab") == "x"


def test_chromosomes_aliases_unknown_and_lengths(figwig, db1, tmp_path, caplog):
    values = _values()
    # "1" is chr1 by alias; chr2 is shorter in the file; chrUn_x is not in the database
    sizes = {"chr1": 1000, "chr2": 500, "chrUn_x": 100}
    vals = {k: {**v, "chrUn_x": np.ones(100)} for k, v in values.items()}
    fw = _write_fw(figwig, tmp_path / "x.fw", vals, sizes=sizes, chrom_names={"chr1": "1", "chr2": "chr2",
                                                                             "chrUn_x": "chrUn_x"})
    with caplog.at_level(logging.INFO, logger="pymisha"), pytest.warns(UserWarning, match="differ in length"):
        pm.gtrack_import_figwig(str(fw), tracks=["i16"], binsize=10, defval=7)
    assert "chrUn_x" in caplog.text
    np.testing.assert_allclose(_dense_of("i16", "chr1"), _expected_dense(values["i16"]["chr1"], 1000, 10, 7),
                               rtol=2e-7)
    np.testing.assert_allclose(_dense_of("i16", "chr2"), _expected_dense(values["i16"]["chr2"][:500], 600, 10, 7),
                               rtol=2e-7)
    # none of the file's chromosomes in the database
    fw2 = tmp_path / "y.fw"
    with figwig.FigWigWriter(str(fw2), {"chrZ": 100}, ["a"]) as w:
        w.write(["chrZ"], [0], np.ones((1, 1, 100)))
    with pytest.raises(ValueError, match="None of the chromosome names"):
        pm.gtrack_import_figwig(str(fw2))


def test_chromosome_absent_from_file(figwig, db1, tmp_path):
    fw = tmp_path / "x.fw"
    with figwig.FigWigWriter(str(fw), {"chr1": 1000}, ["a"]) as w:
        w.write(["chr1"], [0], np.full((1, 1, 1000), 2.0))
    pm.gtrack_import_figwig(str(fw), binsize=100, defval=5, track_prefix="d_")
    pm.gtrack_import_figwig(str(fw), track_prefix="s_")
    assert (_dense_of("d_a", "chr1") == 2).all()
    assert (_dense_of("d_a", "chr2") == 5).all()
    assert _sparse_of("s_a", "chr2") == []
    assert _sparse_of("s_a", "chr1") == [(0, 1000, 2.0)]


def test_not_a_figwig_file(figwig, db1, tmp_path):
    p = tmp_path / "x.fw"
    p.write_text("hello")
    with pytest.raises(ValueError, match="not a figwig file"):
        pm.gtrack_import_figwig(str(p))
    with pytest.raises(ValueError, match="does not exist"):
        pm.gtrack_import_figwig(str(tmp_path / "none.fw"))


def test_chunked_reads_give_the_same_tracks(figwig, db1, tmp_path, monkeypatch):
    import pymisha.figwig as pfw

    values = _values(3)
    fw = _write_fw(figwig, tmp_path / "x.fw", values)
    pm.gtrack_import_figwig(str(fw), binsize=7, track_prefix="a_")
    pm.gtrack_import_figwig(str(fw), track_prefix="sa_")
    monkeypatch.setattr(pfw, "_CHUNK_VALUES", 50)  # a few bins per read
    monkeypatch.setattr(pfw, "_MAX_OPEN_TRACKS", 2)
    pm.gtrack_import_figwig(str(fw), binsize=7, track_prefix="b_")
    pm.gtrack_import_figwig(str(fw), track_prefix="sb_")
    for name in ["cuts_plus", "runs_rep1", "f32", "f64"]:
        for c in SIZES:
            np.testing.assert_array_equal(_dense_of(f"a_{name}", c), _dense_of(f"b_{name}", c))
            assert _sparse_of(f"sa_{name}", c) == _sparse_of(f"sb_{name}", c)


def test_round_trip_misha_figwig_misha(figwig, db, tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for c, n in SIZES.items():
        starts = np.arange(0, n, 20)
        rows.append(pd.DataFrame({"chrom": c, "start": starts, "end": np.minimum(starts + 20, n)}))
    bins = pd.concat(rows, ignore_index=True)
    dense_vals = rng.random(len(bins)).astype(np.float32)
    dense_vals[5:9] = np.nan
    pm.gtrack_create_dense("dn", "dense", bins, dense_vals, 20)
    pm.gtrack_attr_set("dn", "assay", "ChIP")
    sp = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [10, 200, 0], "end": [30, 260, 600]})
    pm.gtrack_create_sparse("sp", "sparse", sp, [3.0, 250.0, 1.0])
    counts = pd.DataFrame({"chrom": ["chr1"] * 3, "start": [0, 1, 500], "end": [1, 2, 501]})
    pm.gtrack_create_sparse("cnt", "counts", counts, [2.0, 3.0, 1.0])

    out = tmp_path / "rt.fw"
    pm.gtrack_export_figwig(["dn"], str(out))
    # each sparse track over its own intervals: where one has an interval and the other none, the other is 0
    pm.gtrack_export_figwig(["sp", "cnt"], str(tmp_path / "sp.fw"), names=["sp", "cnt"])
    reader = figwig.FigWigReader(str(out))
    assert reader.tracks[0]["assay"] == "ChIP"
    assert reader.tracks[0]["dtype"] == "float32"
    assert reader.chrom_sizes == SIZES
    sp_reader = figwig.FigWigReader(str(tmp_path / "sp.fw"))
    assert [t["dtype"] for t in sp_reader.tracks] == ["uint8", "uint8"]

    pm.gtrack_import_figwig(str(out), binsize=20, track_prefix="rt_")
    for c in SIZES:
        np.testing.assert_array_equal(_dense_of("rt_dn", c), _dense_of("dn", c))
    assert pm.gtrack_attr_get("rt_dn", "assay") == "ChIP"
    assert pm.gtrack_attr_get("rt_dn", "description") == "dense"

    pm.gtrack_import_figwig(str(tmp_path / "sp.fw"), track_prefix="rt_")
    for c in SIZES:
        assert _sparse_of("rt_sp", c) == _sparse_of("sp", c)
        assert _sparse_of("rt_cnt", c) == ([(0, 1, 2.0), (1, 2, 3.0), (500, 501, 1.0)] if c == "chr1" else [])


def test_export_expressions_iterator_intervals_and_options(figwig, db1, tmp_path):
    bins = pd.DataFrame({"chrom": ["chr1"] * 50, "start": np.arange(0, 1000, 20), "end": np.arange(20, 1020, 20)})
    pm.gtrack_create_dense("dn", "dense", bins, np.arange(50, dtype=float) / 4, 20)
    pm.gtrack_attr_set("dn", "dtype", "x")  # reserved in figwig
    out = tmp_path / "e.fw"
    with pytest.warns(UserWarning, match="reserves"):
        pm.gtrack_export_figwig(["dn", "dn * 2"], str(out), intervals=pm.gintervals("chr1", [100, 150], [300, 200]),
                                iterator=10, names=["a", "b"], genome="g1", dtype="float64")
    r = figwig.FigWigReader(str(out))
    assert r.genome == "g1"
    assert [t["name"] for t in r.tracks] == ["a", "b"]
    assert r.tracks[1]["expression"] == "dn * 2"
    assert r.tracks[0]["dtype"] == "float64"
    y = r.read(["chr1"], [0], 1000, dtype="float64")[0]
    expected = np.zeros(1000)
    expected[100:300] = (np.arange(1000) // 20 / 4)[100:300]
    np.testing.assert_array_equal(y[0], expected)
    np.testing.assert_array_equal(y[1], 2 * expected)
    assert r.read(["chr2"], [0], 600, dtype="float64").sum() == 0

    # NaN as missing
    pm.gtrack_create_dense("nan_dn", "d", pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [40]}), [3.0], 20)
    pm.gtrack_export_figwig("nan_dn", str(tmp_path / "n.fw"))
    pm.gtrack_export_figwig("nan_dn", str(tmp_path / "z.fw"), missing=0)
    rn = figwig.FigWigReader(str(tmp_path / "n.fw"))
    rz = figwig.FigWigReader(str(tmp_path / "z.fw"))
    assert rn.tracks[0]["dtype"] == "float32" and rz.tracks[0]["dtype"] == "uint8"
    yn = rn.read(["chr1"], [0], 1000, dtype="float64")[0, 0]
    assert (yn[:40] == 3).all() and np.isnan(yn[40:]).all()
    yz = rz.read(["chr1"], [0], 1000)[0, 0]
    assert (yz[:40] == 3).all() and (yz[40:] == 0).all()

    with pytest.raises(ValueError, match="distinct"):
        pm.gtrack_export_figwig(["dn", "dn"], str(tmp_path / "x.fw"))
    with pytest.raises(ValueError, match="names has"):
        pm.gtrack_export_figwig(["dn"], str(tmp_path / "x.fw"), names=["a", "b"])
    with pytest.raises(FileNotFoundError):
        pm.gtrack_export_figwig(["dn"], str(tmp_path / "nodir" / "x.fw"))


def test_export_tracks_of_different_kinds(figwig, db1, tmp_path):
    bins = pd.DataFrame({"chrom": ["chr1"] * 100, "start": np.arange(0, 1000, 10), "end": np.arange(10, 1010, 10)})
    pm.gtrack_create_dense("d10", "d", bins, np.arange(100, dtype=float), 10)
    bins = pd.DataFrame({"chrom": ["chr1"] * 40, "start": np.arange(0, 1000, 25), "end": np.arange(25, 1025, 25)})
    pm.gtrack_create_dense("d25", "d", bins, np.arange(40, dtype=float) + 0.5, 25)
    pm.gtrack_create_sparse("sp", "s", pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [33, 7], "end": [44, 9]}),
                            [5.0, 6.0])
    pm.gtrack_export_figwig(["d10", "d25", "sp", "d10 + 1"], str(tmp_path / "m.fw"))
    y = figwig.FigWigReader(str(tmp_path / "m.fw")).read(["chr1", "chr2"], [0, 0], 600, dtype="float64")
    pos = np.arange(600)
    np.testing.assert_array_equal(y[0, 0], pos // 10)
    np.testing.assert_array_equal(y[0, 1], pos // 25 + 0.5)
    np.testing.assert_array_equal(y[0, 2], np.where((pos >= 33) & (pos < 44), 5, 0))
    np.testing.assert_array_equal(y[0, 3], pos // 10 + 1)
    np.testing.assert_array_equal(y[1, 2], np.where((pos >= 7) & (pos < 9), 6, 0))
    assert np.isnan(y[1, 0]).all()  # the bins of d10 on chr2 are NaN
    pm.gtrack_import_figwig(str(tmp_path / "m.fw"), tracks=["sp"], track_prefix="rt_")
    assert _sparse_of("rt_sp", "chr1") == [(33, 44, 5.0)]


def test_export_chunks_do_not_change_values(figwig, db1, tmp_path, monkeypatch):
    import pymisha.figwig as pfw

    rng = np.random.default_rng(9)
    bins = pd.DataFrame({"chrom": ["chr1"] * 200, "start": np.arange(0, 1000, 5), "end": np.arange(5, 1005, 5)})
    pm.gtrack_create_dense("dn5", "d", bins, rng.random(200), 5)
    sp = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [3, 90], "end": [80, 900]})
    pm.gtrack_create_sparse("sp", "s", sp, [1.5, 2.5])
    pm.gtrack_export_figwig(["dn5"], str(tmp_path / "a.fw"))
    pm.gtrack_export_figwig(["sp"], str(tmp_path / "sa.fw"))
    monkeypatch.setattr(pfw, "_EXPORT_CHUNK", 7)
    pm.gtrack_export_figwig(["dn5"], str(tmp_path / "b.fw"))
    pm.gtrack_export_figwig(["sp"], str(tmp_path / "sb.fw"))
    for a, b in [("a.fw", "b.fw"), ("sa.fw", "sb.fw")]:
        ya = figwig.FigWigReader(str(tmp_path / a)).read(["chr1"], [0], 1000, dtype="float64")
        yb = figwig.FigWigReader(str(tmp_path / b)).read(["chr1"], [0], 1000, dtype="float64")
        np.testing.assert_array_equal(ya, yb)


def test_missing_figwig_package(db1, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "figwig", None)
    with pytest.raises(ImportError, match=r"pymisha\[figwig\]"):
        pm.gtrack_import_figwig(str(tmp_path / "x.fw"))
    with pytest.raises(ImportError, match=r"pymisha\[figwig\]"):
        pm.gtrack_export_figwig("x", str(tmp_path / "x.fw"))
