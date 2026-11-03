"""Chromosome names and order on a per-chromosome database, against R misha.

The fixture (``chromdb/``, built by ``chromdb/make_chromdb.R``) is a per-chromosome
database whose chrom_sizes.txt names (2, 10, 1, X, 1_KI270706v1_random) lack the "chr"
prefix of its seq files. R names these chromosomes chr2, chr10, ... and numbers them in
the order of the sorted names: chr1, chr1_KI270706v1_random, chr10, chr2, chrX (R's
collation; Python's ``sorted`` and a C locale put chr10 second). ``chromdb/expected``
holds R's outputs on it.

The comparisons are exact, names and row order, with no sorting: pymisha must read what
R wrote and give R's rows in R's order, and what pymisha writes must be what R writes.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _r_serialize

_HERE = Path(__file__).resolve().parent / "chromdb"
_EXPECTED = _HERE / "expected"

# chrom_sizes.txt order and R's chrom id order
_FILE_NAMES = ["2", "10", "1", "X", "1_KI270706v1_random"]
_R_NAMES = ["chr1", "chr1_KI270706v1_random", "chr10", "chr2", "chrX"]

_CHROM_COLS = ("chrom", "chrom1", "chrom2")


def _expected(name: str):
    return _r_serialize.read(str(_EXPECTED / f"{name}.rds"))


def _copy_db(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    for sub in ("tracks", "pssms"):
        (dst / sub).mkdir(exist_ok=True)
    return dst


@pytest.fixture
def chromdb(tmp_path):
    """A writable copy of the fixture database, active for one test (the conftest
    restores the previous database afterwards)."""
    db = _copy_db(_HERE / "db", tmp_path / "db")
    pm.gdb_init(str(db))
    return db


def assert_frame_as_r(py: pd.DataFrame | None, r: pd.DataFrame, value_cols: dict[str, str] | None = None) -> None:
    """py has R's rows in R's order: chrom columns equal as strings, the other columns
    equal as numbers. value_cols maps an R column name to pymisha's where they differ."""
    value_cols = value_cols or {}
    assert py is not None, "pymisha returned no rows"
    py = py.reset_index(drop=True)
    assert len(py) == len(r), f"{len(py)} rows, R has {len(r)}\npymisha:\n{py}\nR:\n{r}"
    for col in r.columns:
        pycol = value_cols.get(col, col)
        assert pycol in py.columns, f"pymisha lacks column {pycol}: {list(py.columns)}"
        if col in _CHROM_COLS:
            assert py[pycol].astype(str).tolist() == r[col].astype(str).tolist(), (
                f"{col} differs\npymisha:\n{py}\nR:\n{r}"
            )
        else:
            np.testing.assert_allclose(
                py[pycol].to_numpy(dtype=float), r[col].to_numpy(dtype=float),
                rtol=1e-6, equal_nan=True, err_msg=f"{col} differs\npymisha:\n{py}\nR:\n{r}",
            )


# --------------------------------------------------------------------------------------
# names and chrom id order


def test_gintervals_all_names_and_order(chromdb):
    assert_frame_as_r(pm.gintervals_all(), _expected("allgenome"))
    assert pm.gintervals_all()["chrom"].tolist() == _R_NAMES


def test_gintervals_2d_all_order(chromdb):
    assert_frame_as_r(pm.gintervals_2d_all(), _expected("allgenome2d"))


def test_aliases_still_accepted(chromdb):
    for spelling in ("1", "chr1"):
        iv = pm.gintervals(spelling, 0, 100)
        assert iv["chrom"].tolist() == ["chr1"]
    iv = pm.gintervals(["X", "1_KI270706v1_random", "10"], 0, 10)
    assert iv["chrom"].tolist() == ["chr1_KI270706v1_random", "chr10", "chrX"]


# --------------------------------------------------------------------------------------
# reading what R wrote


@pytest.mark.parametrize("track", ["dense", "dense_idx"])
def test_extract_dense(chromdb, track):
    assert_frame_as_r(pm.gextract(track, pm.gintervals_all(), iterator=100), _expected(f"extract_{track}"))


@pytest.mark.parametrize("track", ["sparse", "sparse_idx", "array", "array_idx"])
def test_extract_by_track_intervals(chromdb, track):
    assert_frame_as_r(pm.gextract(track, pm.gintervals_all()), _expected(f"extract_{track}"))


@pytest.mark.parametrize("track", ["rects", "points", "rects_idx", "points_idx"])
def test_extract_2d(chromdb, track):
    assert_frame_as_r(pm.gextract(track, pm.gintervals_2d_all()), _expected(f"extract_{track}"))


def test_extract_with_interval_set_iterator(chromdb):
    assert_frame_as_r(
        pm.gextract("sparse", pm.gintervals_all(), iterator="iv1d"), _expected("extract_sparse_iter_set")
    )


def test_screen(chromdb):
    assert_frame_as_r(pm.gscreen("dense > 5", pm.gintervals_all()), _expected("screen_dense"))


def test_iterator_intervals(chromdb):
    assert_frame_as_r(
        pm.giterator_intervals(intervals=pm.gintervals_all(), iterator=1000), _expected("iterator_1000")
    )


@pytest.mark.parametrize("name", ["iv1d", "iv2d", "big1d", "big2d", "big1d_idx", "big2d_idx"])
def test_load_interval_sets(chromdb, name):
    assert_frame_as_r(pm.gintervals_load(name), _expected(f"load_{name}"))


def test_load_bigset_chromosome(chromdb):
    assert_frame_as_r(pm.gintervals_load("big1d", chrom="chr10"), _expected("load_big1d_chr10"))
    assert_frame_as_r(pm.gintervals_load("big1d", chrom="10"), _expected("load_big1d_chr10"))
    assert_frame_as_r(
        pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr2"), _expected("load_big2d_chr1_chr2")
    )


def _mixed():
    mixed = pm.gintervals(
        ["chrX", "chr10", "chr1_KI270706v1_random", "chr1", "chr2", "chr1"],
        [0, 100, 50, 900, 10, 950],
        [200, 300, 400, 1000, 20, 1100],
    )
    other = pm.gintervals(
        ["chr1", "chr10", "chr2", "chr1_KI270706v1_random"], [950, 0, 0, 0], [2000, 150, 15, 60]
    )
    return mixed, other


def test_interval_operations(chromdb):
    mixed, other = _mixed()
    assert_frame_as_r(pm.gintervals_canonic(mixed), _expected("canonic"))
    assert_frame_as_r(pm.gintervals_union(mixed, other), _expected("union"))
    assert_frame_as_r(pm.gintervals_intersect(pm.gintervals_canonic(mixed), other), _expected("intersect"))
    assert_frame_as_r(pm.gintervals_diff(mixed, other), _expected("diff"))


def _read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _assert_lines_as_r(py: list[str], r: list[str]) -> None:
    """The same lines, the coordinates as text and the values as numbers (R prints 9 for 9.0)."""
    assert len(py) == len(r), f"pymisha:\n{py}\nR:\n{r}"
    for py_line, r_line in zip(py, r, strict=True):
        py_fields, r_fields = py_line.split("\t"), r_line.split("\t")
        assert py_fields[:3] == r_fields[:3], f"pymisha: {py_line!r}, R: {r_line!r}"
        for a, b in zip(py_fields[3:], r_fields[3:], strict=True):
            assert a == b or float(a) == float(b), f"pymisha: {py_line!r}, R: {r_line!r}"


def test_written_text_files(chromdb, tmp_path):
    out = tmp_path / "extract.tsv"
    pm.gextract("sparse", pm.gintervals_all(), file=str(out))
    _assert_lines_as_r(_read_lines(out), _expected("extract_file_lines"))

    out = tmp_path / "sparse.bedGraph"
    pm.gtrack_export_bedgraph("sparse", str(out))
    _assert_lines_as_r(_read_lines(out), _expected("bedgraph_lines"))

    out = tmp_path / "genome.fa"
    pm.gdb_export_fasta(str(out))
    assert [line for line in _read_lines(out) if line.startswith(">")] == _expected("fasta_headers")


# --------------------------------------------------------------------------------------
# liftover: an indexed source track is keyed by its own database's chrom ids


@pytest.mark.parametrize(
    "track", ["sparse", "sparse_idx", "dense", "dense_idx", "rects", "rects_idx", "points", "points_idx"]
)
def test_liftover_from_per_chromosome_db(tmp_path, track):
    src = _copy_db(_HERE / "db", tmp_path / "db")
    tgt = _copy_db(_HERE / "tgt", tmp_path / "tgt")
    pm.gdb_init(str(tgt))
    chain = pm.gintervals_load_chain(str(_HERE / "chain.txt"))
    lifted = f"lifted_{track}"
    pm.gtrack_liftover(lifted, "", str(src / "tracks" / f"{track}.track"), chain)
    if track.startswith(("rects", "points")):
        res = pm.gextract(lifted, pm.gintervals_2d_all())
    elif track.startswith("dense"):
        res = pm.gextract(lifted, pm.gintervals_all(), iterator=100)
    else:
        res = pm.gextract(lifted, pm.gintervals_all())
    assert_frame_as_r(res, _expected(f"liftover_{track}"))


# --------------------------------------------------------------------------------------
# writing what R writes

_IV1D = pd.DataFrame({
    "chrom": [c for c in ["chr2", "chr10", "chr1", "chrX", "chr1_KI270706v1_random"] for _ in range(2)],
    "start": [100, 500] * 5,
    "end": [300, 700] * 5,
})
_VALS = np.arange(1, 11) * 1.5
_IV2D = pd.DataFrame({
    "chrom1": ["chr10", "chr1", "chr2", "chrX", "chr1_KI270706v1_random", "chr1"],
    "start1": [100, 200, 300, 400, 500, 600],
    "end1": [150, 260, 380, 470, 590, 650],
    "chrom2": ["chrX", "chr2", "chr2", "chr1", "chr10", "chr1_KI270706v1_random"],
    "start2": [50, 60, 70, 80, 90, 100],
    "end2": [90, 120, 110, 130, 150, 170],
})
_VALS2D = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def _data_files(track_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in track_dir.iterdir() if p.is_file() and not p.name.startswith(".")}


def _write_like_r(db: Path, tmp_path: Path) -> None:
    """Create, with pymisha, the tracks make_chromdb.R creates, under a py_ prefix."""
    pm.gtrack_create_sparse("py_sparse", "", _IV1D, _VALS)
    pm.gtrack_create_dense("py_dense", "", _IV1D, _VALS, 100)
    src = tmp_path / "array.tsv"
    arr = _IV1D.assign(c1=_VALS, c2=10 * _VALS)
    arr.to_csv(src, sep="\t", index=False)
    pm.gtrack_array_import("py_array", "", str(src))
    pm.gtrack_2d_create("py_rects", "", _IV2D, _VALS2D)
    points = _IV2D.assign(end1=_IV2D["start1"] + 1, end2=_IV2D["start2"] + 1, value=_VALS2D)
    src = tmp_path / "points.tsv"
    points.to_csv(src, sep="\t", index=False)
    pm.gtrack_2d_import("py_points", "", str(src))
    for t in ("sparse", "dense", "array"):
        pm.gtrack_copy(f"py_{t}", f"py_{t}_idx")
        pm.gtrack_convert_to_indexed(f"py_{t}_idx", remove_old=True)
    for t in ("rects", "points"):
        pm.gtrack_copy(f"py_{t}", f"py_{t}_idx")
        pm.gtrack_2d_convert_to_indexed(f"py_{t}_idx", remove_old=True)


@pytest.mark.parametrize(
    "track",
    ["sparse", "dense", "array", "rects", "points", "sparse_idx", "dense_idx", "array_idx", "rects_idx", "points_idx"],
)
def test_written_files_match_r(chromdb, tmp_path, track):
    """pymisha names a track's files as R does and writes R's bytes into them."""
    _write_like_r(chromdb, tmp_path)
    r_files = _data_files(chromdb / "tracks" / f"{track}.track")
    py_files = _data_files(chromdb / "tracks" / f"py_{track}.track")
    assert sorted(py_files) == sorted(r_files)
    for name, data in r_files.items():
        assert py_files[name] == data, f"{track}/{name} differs from R's"


def test_saved_interval_sets_named_and_ordered_as_r(chromdb):
    pm.gintervals_save(_IV1D, "py_iv1d")
    pm.gintervals_save(_IV2D, "py_iv2d")
    assert_frame_as_r(pm.gintervals_load("py_iv1d"), _expected("load_iv1d"))
    assert_frame_as_r(pm.gintervals_load("py_iv2d"), _expected("load_iv2d"))
    saved = _r_serialize.read(str(chromdb / "tracks" / "py_iv1d.interv"))
    assert saved["chrom"].astype(str).tolist() == _expected("load_iv1d")["chrom"].astype(str).tolist()


def _r_available() -> bool:
    """Whether Rscript loads misha (the default library; any misha names and orders the
    chromosomes of this database as 5.12.1 does)."""
    if shutil.which("Rscript") is None:
        return False
    res = subprocess.run(
        ["Rscript", "-e", "suppressMessages(library(misha))"], capture_output=True, text=True, timeout=120
    )
    return res.returncode == 0


def test_r_reads_what_pymisha_writes(chromdb, tmp_path):
    """R reads the tracks and interval sets pymisha wrote as it reads its own."""
    if not _r_available():
        pytest.skip("R with misha is not available")
    _write_like_r(chromdb, tmp_path)
    pm.gintervals_save(_IV1D, "py_iv1d")
    pm.gintervals_save(_IV2D, "py_iv2d")
    out = tmp_path / "r_out"
    out.mkdir()
    script = f"""
suppressMessages(library(misha))
options(gmax.processes = 1)
gsetroot("{chromdb}")
out <- "{out}"
for (t in c("dense", "dense_idx")) saveRDS(gextract(paste0("py_", t), gintervals.all(), iterator = 100), file.path(out, paste0(t, ".rds")))
for (t in c("sparse", "sparse_idx", "array", "array_idx")) saveRDS(gextract(paste0("py_", t), gintervals.all()), file.path(out, paste0(t, ".rds")))
for (t in c("rects", "points", "rects_idx", "points_idx")) saveRDS(gextract(paste0("py_", t), gintervals.2d.all()), file.path(out, paste0(t, ".rds")))
saveRDS(gintervals.load("py_iv1d"), file.path(out, "iv1d.rds"))
saveRDS(gintervals.load("py_iv2d"), file.path(out, "iv2d.rds"))
"""
    res = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr
    for t in ("dense", "dense_idx", "sparse", "sparse_idx", "array", "array_idx",
              "rects", "points", "rects_idx", "points_idx"):
        r_read = _r_serialize.read(str(out / f"{t}.rds"))
        expected = _expected(f"extract_{t}")
        assert_frame_as_r(r_read.rename(columns={f"py_{t}": t}), expected)
    for s in ("iv1d", "iv2d"):
        assert_frame_as_r(_r_serialize.read(str(out / f"{s}.rds")), _expected(f"load_{s}"))


# --------------------------------------------------------------------------------------
# files written by pymisha 0.14 and earlier, named by the chrom_sizes.txt names


def _rename_to_file_names(track_dir: Path) -> None:
    to_old = dict(zip(_R_NAMES, ["1", "1_KI270706v1_random", "10", "2", "X"], strict=True))
    for p in list(track_dir.iterdir()):
        if p.name in to_old:
            p.rename(track_dir / to_old[p.name])
        elif "-" in p.name and not p.name.startswith("."):
            c1, c2 = p.name.split("-", 1)
            if c1 in to_old and c2 in to_old:
                p.rename(track_dir / f"{to_old[c1]}-{to_old[c2]}")


@pytest.mark.parametrize("track", ["dense", "sparse", "array", "rects", "points"])
def test_reads_files_named_by_chrom_sizes_names(chromdb, track):
    track_dir = chromdb / "tracks" / f"{track}.track"
    _rename_to_file_names(track_dir)
    assert not any(p.name.startswith("chr") for p in track_dir.iterdir())
    pm.gdb_reload()
    if track in ("rects", "points"):
        res = pm.gextract(track, pm.gintervals_2d_all())
    elif track == "dense":
        res = pm.gextract(track, pm.gintervals_all(), iterator=100)
    else:
        res = pm.gextract(track, pm.gintervals_all())
    assert_frame_as_r(res, _expected(f"extract_{track}"))


@pytest.mark.parametrize("name", ["big1d", "big2d"])
def test_reads_bigset_files_named_by_chrom_sizes_names(chromdb, name):
    _rename_to_file_names(chromdb / "tracks" / f"{name}.interv")
    pm.gdb_reload()
    assert_frame_as_r(pm.gintervals_load(name), _expected(f"load_{name}"))


def _rekey_track_idx(track_dir: Path, new_id: dict[int, int]) -> None:
    """Rewrite a 1D track.idx with its chrom ids mapped by new_id, as a track converted
    with other chrom ids would have them."""
    from pymisha._crc64 import crc64_finalize, crc64_incremental, crc64_init

    idx = track_dir / "track.idx"
    data = bytearray(idx.read_bytes())
    header = 36
    n = struct.unpack_from("<I", data, 16)[0]
    entries = []
    for i in range(n):
        chrom_id, offset, length, reserved = struct.unpack_from("<IQQI", data, header + 24 * i)
        entries.append((new_id[chrom_id], offset, length, reserved))
    entries.sort()
    data[header:] = b"".join(struct.pack("<IQQI", *e) for e in entries)
    crc = crc64_init()   # over chrom_id, offset and length (TrackIndex::compute_checksum)
    for chrom_id, offset, length, _ in entries:
        crc = crc64_incremental(crc, struct.pack("<IQQ", chrom_id, offset, length))
    struct.pack_into("<Q", data, 28, crc64_finalize(crc))
    idx.write_bytes(bytes(data))


def test_dense_indexed_with_other_chrom_ids_errors_as_in_r(chromdb):
    """An indexed dense track keyed by chrom_sizes.txt order (as pymisha 0.14 and earlier
    converted one) does not fit the chromosome sizes: R refuses to read it, and so does
    pymisha, instead of returning another chromosome's values."""
    old_id = {r_id: _FILE_NAMES.index(name[3:]) for r_id, name in enumerate(_R_NAMES)}
    _rekey_track_idx(chromdb / "tracks" / "dense_idx.track", old_id)
    pm.gdb_reload()
    with pytest.raises(Exception, match="do not match the chromosome size"):
        pm.gextract("dense_idx", pm.gintervals_all(), iterator=100)


# --------------------------------------------------------------------------------------
# database-level operations keep R's chrom ids


def test_convert_db_to_indexed_keeps_chrom_ids(chromdb):
    pm.gdb_convert_to_indexed(groot=str(chromdb), convert_tracks=False, convert_intervals=False)
    names = [line.split("\t")[0] for line in (chromdb / "chrom_sizes.txt").read_text().splitlines()]
    assert names == _R_NAMES
    pm.gdb_init(str(chromdb))
    assert_frame_as_r(pm.gintervals_all(), _expected("allgenome"))
    for t in ("sparse_idx", "array_idx"):
        assert_frame_as_r(pm.gextract(t, pm.gintervals_all()), _expected(f"extract_{t}"))
    assert_frame_as_r(pm.gextract("dense_idx", pm.gintervals_all(), iterator=100), _expected("extract_dense_idx"))
    assert_frame_as_r(pm.gextract("rects_idx", pm.gintervals_2d_all()), _expected("extract_rects_idx"))


def test_copy_indexed_track_to_another_db(chromdb, tmp_path):
    """gtrack_copy decodes an indexed source by its database's chrom ids."""
    dest = _copy_db(_HERE / "db", tmp_path / "dest")
    shutil.rmtree(dest / "tracks")
    (dest / "tracks").mkdir()
    # the destination lists the chromosomes in another order
    (dest / "chrom_sizes.txt").write_text("".join(
        f"{c}\t{s}\n" for c, s in [("X", 1500), ("1", 4000), ("2", 3000), ("10", 2500),
                                    ("1_KI270706v1_random", 1200)]
    ))
    pm.gtrack_copy("sparse_idx", "copied", db=str(dest))
    pm.gdb_init(str(dest))
    res = pm.gextract("copied", pm.gintervals_all())
    assert_frame_as_r(res, _expected("extract_sparse_idx"), value_cols={"sparse_idx": "copied"})


@pytest.mark.parametrize("track", ["sparse", "sparse_idx"])
@pytest.mark.parametrize("pure_python", [False, True])
def test_liftover_with_chain_named_as_chrom_sizes(tmp_path, track, pure_python):
    """A chain naming the source chromosomes 1, 10, ... reads an indexed source by its
    database's chr1, chr10, ... (as R does); the files of a per-chromosome source are found
    by the chain's names or those names with "chr" toggled (R misha 5.13.3, PR #198), so
    both lift the same. liftover_unprefixed_chain_sparse.rds is R misha 5.14.0's output
    (5.12.1, which made the other files here, lifted nothing from the per-chromosome files)."""
    src = _copy_db(_HERE / "db", tmp_path / "db")
    tgt = _copy_db(_HERE / "tgt", tmp_path / "tgt")
    pm.gdb_init(str(tgt))
    chain = pm.gintervals_load_chain(str(_HERE / "chain_unprefixed.txt"))
    lifted = f"lifted_{track}"
    pm.gtrack_liftover(lifted, "", str(src / "tracks" / f"{track}.track"), chain,
                       _force_pure_python=pure_python)
    res = pm.gextract(lifted, pm.gintervals_all())
    expected = _expected(f"liftover_unprefixed_chain_{track}")
    if expected is None:
        assert res is None or len(res) == 0
    else:
        assert_frame_as_r(res, expected)
