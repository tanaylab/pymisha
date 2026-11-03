"""Per-chromosome files of tracks and big intervals sets named by an alias of their
chromosomes ("1" for chr1, "MT" for chrM; "1-2", "1-chr2" or "chr1-2" for chr1 and chr2),
as pymisha 0.14 and earlier named them in databases whose chrom_sizes.txt names lack "chr".
One listing of the directory (GenomeTrack::ChromFiles) maps them to their chromosomes for
every reader and converter. Ported from R misha 5.13.3 (PR #198,
tests/testthat/test-alias-chrom-files.R); pymisha writes no big intervals sets, so these use
the ones R wrote (tests/r_parity/chromdb)."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest
from _chromdb_helpers import copy_r_bigset, unsorted_db, write_index

import pymisha as pm


def alias_db(path: Path) -> str:
    """A per-chromosome database as unsorted_db(), with a chrM, which has an alias other than
    its name without "chr" (MT)."""
    chroms = ["2", "10", "1", "X", "M"]
    sizes = [2000, 1500, 1000, 1200, 500]
    (path / "tracks").mkdir(parents=True)
    (path / "seq").mkdir()
    for chrom, size in zip(chroms, sizes, strict=True):
        (path / "seq" / f"chr{chrom}.seq").write_bytes(b"A" * size)
    (path / "chrom_sizes.txt").write_text("".join(f"{c}\t{s}\n" for c, s in zip(chroms, sizes, strict=True)))
    return str(path.resolve())


def _mkdb(path: Path, chroms: list[str]) -> str:
    (path / "tracks").mkdir(parents=True)
    (path / "seq").mkdir()
    for chrom in chroms:
        (path / "seq" / f"chr{chrom}.seq").write_bytes(b"A" * 1000)
    (path / "chrom_sizes.txt").write_text("".join(f"{c}\t1000\n" for c in chroms))
    return str(path.resolve())


def alias_names(files: list[str]) -> list[str]:
    """chr1 becomes 1 and chrM MT; chr1-chr2 becomes 1-2, 1-chr2 or chr1-2 in turn."""
    out = []
    for i, f in enumerate(files):
        chroms = f.split("-")
        strip = [True] if len(chroms) == 1 else [(True, True), (True, False), (False, True)][i % 3]
        out.append("-".join(
            ("MT" if c == "chrM" else c[3:] if c.startswith("chr") else c) if s else c
            for c, s in zip(chroms, strip, strict=True)
        ))
    return out


def rename_to_aliases(obj_dir: Path, expect_min: int = 3) -> None:
    files = sorted(f for f in os.listdir(obj_dir) if f.startswith("chr"))
    assert len(files) >= expect_min
    for f, a in zip(files, alias_names(files), strict=True):
        (obj_dir / f).rename(obj_dir / a)
    (obj_dir / ".meta").unlink(missing_ok=True)


TRACKS = ["sp", "dn", "arr", "rects", "pts"]


def create_alias_objects(db: str, tmp_path: Path) -> None:
    """A sparse, a dense and an array track, a rectangles and a points track, all with files
    on chrM."""
    pm.gdb_init(db)
    iv = pm.gintervals(["chr1", "chr1", "chr2", "chrX", "chrM"], [0, 100, 0, 10, 5], [50, 150, 100, 20, 15])
    iv2d = pm.gintervals_2d(["chr1", "chr2", "chr1", "chrM"], [0, 0, 10, 0], [50, 100, 20, 30],
                            ["chr1", "chr2", "chrX", "chr1"], [0, 0, 10, 40], [50, 100, 20, 60])
    pm.gtrack_create_sparse("sp", "x", iv, list(range(1, 6)))
    pm.gtrack_create("dn", "x", "sp", iterator=10)
    arr = tmp_path / "arr.tsv"
    pd.concat([iv.reset_index(drop=True), pd.DataFrame({"a": range(1, 6), "b": range(6, 11)})], axis=1).to_csv(
        arr, sep="\t", index=False)
    pm.gtrack_array_import("arr", "x", str(arr))
    pm.gtrack_2d_create("rects", "x", iv2d, list(range(1, 5)))
    pts = iv2d.copy()
    pts["end1"] = pts["start1"] + 1
    pts["end2"] = pts["start2"] + 1
    pts["value"] = range(1, 5)
    contacts = tmp_path / "contacts.tsv"
    pts.to_csv(contacts, sep="\t", index=False)
    pm.gtrack_2d_import_contacts("pts", "x", str(contacts), fends=None)


def read_alias_objects() -> dict:
    """Every path that reads the objects."""
    scope = pm.gintervals_all()
    scope2d = pm.gintervals_2d_all()
    pm.gvtrack_create("v_dn", "dn", "avg")
    pm.gvtrack_create("v_sp1", "sp", "sum")
    pm.gvtrack_iterator("v_sp1", dim=1)
    try:
        return {
            "sp": pm.gextract("sp", scope),
            "sp_bins": pm.gextract("sp", scope, iterator=50),
            "dn": pm.gextract("dn", scope),
            "dn_vtrack": pm.gextract("v_dn * 2", scope, iterator=100),
            "arr": pm.gtrack_array_extract("arr", None, scope),
            "arr_cols": pm.gtrack_array_extract("arr", ["b"], scope),
            "screen": pm.gscreen("sp > 1", scope),
            "projected": pm.gextract("v_sp1", scope2d, iterator="rects"),
            "info": [{k: pm.gtrack_info(t)[k] for k in ("type", "dimensions")} for t in TRACKS],
            "sp_load": pm.gintervals_load("sp"),
            "sp_as_scope": pm.gextract("dn", "sp"),
            "rects": pm.gextract("rects", scope2d),
            "rects_bins": pm.gextract("rects", scope2d, iterator=scope2d),
            "pts": pm.gextract("pts", scope2d),
            "pts_bins": pm.gextract("pts", scope2d, iterator=scope2d),
            "rects_load": pm.gintervals_load("rects"),
            "pts_load": pm.gintervals_load("pts", chrom1="chrM", chrom2="chr1"),
            "rects_as_scope": pm.gextract("rects", "rects"),
            "pts_as_scope": pm.gextract("pts", "pts"),
        }
    finally:
        pm.gvtrack_rm("v_dn")
        pm.gvtrack_rm("v_sp1")


def assert_same(got: dict, expected: dict) -> None:
    assert got.keys() == expected.keys()
    for k in expected:
        if isinstance(expected[k], pd.DataFrame):
            pd.testing.assert_frame_equal(got[k].reset_index(drop=True), expected[k].reset_index(drop=True), obj=k)
        else:
            assert got[k] == expected[k], k


BIGSETS = ["big1d", "big2d"]


def read_bigsets() -> dict:
    """The big intervals sets R wrote, on unsorted_db()'s chromosomes, and as scopes and
    iterators of a track on them."""
    return {
        "big1d": pm.gintervals_load("big1d"),
        "big1d_chrom": pm.gintervals_load("big1d", chrom="chr10"),
        "big1d_as_scope": pm.gextract("t1", "big1d"),
        "big1d_as_iterator": pm.gextract("t1", pm.gintervals_all(), iterator="big1d"),
        "big2d": pm.gintervals_load("big2d"),
        "big2d_pair": pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr2"),
        "big2d_as_scope": pm.gextract("t2", "big2d"),
    }


def bigset_db(tmp_path: Path) -> str:
    """unsorted_db() with the chromosome sizes of the database R wrote the sets in."""
    db = unsorted_db(tmp_path / "bigdb")
    sizes = {"2": 3000, "10": 2500, "1": 4000, "X": 1500, "1_KI270706v1_random": 1200}
    (Path(db) / "chrom_sizes.txt").write_text("".join(f"{c}\t{s}\n" for c, s in sizes.items()))
    for c, s in sizes.items():
        (Path(db) / "seq" / f"chr{c}.seq").write_bytes(b"A" * s)
    for name in BIGSETS:
        copy_r_bigset(name, db)
    pm.gdb_init(db)
    pm.gtrack_create_sparse("t1", "x", pm.gintervals(["chr1", "chr10", "chr2", "chrX"], 0, 1000), [1, 10, 2, 23])
    pm.gtrack_2d_create("t2", "x", pm.gintervals_2d(["chr1", "chr10"], 0, 1000, ["chr2", "chrX"], 0, 1000), [12, 1023])
    return db


def test_files_named_by_aliases_read_as_with_canonical_names(tmp_path):
    """R: tracks and intervals sets whose files are named by chromosome aliases read as with
    canonical names."""
    db = alias_db(tmp_path / "db")
    create_alias_objects(db, tmp_path)
    expected = read_alias_objects()
    assert len(expected["sp"]) == 5
    assert len(expected["rects"]) == 4
    for t in TRACKS:
        rename_to_aliases(Path(db) / "tracks" / f"{t}.track")
    assert "MT" in os.listdir(Path(db) / "tracks" / "sp.track")
    pm.gdb_reload()
    assert_same(read_alias_objects(), expected)

    bdb = bigset_db(tmp_path)
    expected = read_bigsets()
    assert len(expected["big1d_chrom"]) > 0
    for name in BIGSETS:
        rename_to_aliases(Path(bdb) / "tracks" / f"{name}.interv")
        # the stats of the set (its .meta) are kept: they are not named by files
    for name in BIGSETS:
        shutil.copy(Path(__file__).parent / "r_parity" / "chromdb" / "db" / "tracks" / f"{name}.interv" / ".meta",
                    Path(bdb) / "tracks" / f"{name}.interv" / ".meta")
    assert_same(read_bigsets(), expected)


def test_files_named_by_aliases_convert_to_the_indexed_format(tmp_path):
    """R: tracks and intervals sets whose files are named by chromosome aliases convert to
    the indexed format."""
    db = alias_db(tmp_path / "db")
    create_alias_objects(db, tmp_path)
    expected = read_alias_objects()
    for t in TRACKS:
        rename_to_aliases(Path(db) / "tracks" / f"{t}.track")
    pm.gdb_reload()
    for t in ("sp", "dn", "arr"):
        pm.gtrack_convert_to_indexed(t, remove_old=True)
    for t in ("rects", "pts"):
        pm.gtrack_2d_convert_to_indexed(t, remove_old=True)
    for t in TRACKS:
        files = os.listdir(Path(db) / "tracks" / f"{t}.track")
        assert "track.idx" in files, t
        # no per-chromosome file is left
        assert [f for f in files if not f.startswith(".") and f not in ("track.idx", "track.dat", "vars")] == [], t
    assert_same(read_alias_objects(), expected)

    bdb = bigset_db(tmp_path)
    expected = read_bigsets()
    meta = {n: (Path(bdb) / "tracks" / f"{n}.interv" / ".meta").read_bytes() for n in BIGSETS}
    for name in BIGSETS:
        rename_to_aliases(Path(bdb) / "tracks" / f"{name}.interv")
        (Path(bdb) / "tracks" / f"{name}.interv" / ".meta").write_bytes(meta[name])
    pm.gintervals_convert_to_indexed("big1d", remove_old=True)
    pm.gintervals_2d_convert_to_indexed("big2d", remove_old=True)
    for name in BIGSETS:
        files = sorted(f for f in os.listdir(Path(bdb) / "tracks" / f"{name}.interv") if not f.startswith("."))
        assert all(f.startswith("intervals") for f in files), files
    assert_same(read_bigsets(), expected)


def test_files_named_by_aliases_are_copied_and_modified(tmp_path):
    """R: tracks whose files are named by chromosome aliases are copied and modified (the
    gintervals_update part: pymisha rewrites the whole set, which leaves no alias-named
    file)."""
    db = alias_db(tmp_path / "db")
    dest_perchrom = alias_db(tmp_path / "dest_perchrom")
    dest_indexed = tmp_path / "dest_indexed"
    (dest_indexed / "tracks").mkdir(parents=True)
    names = ["chr2", "chr10", "chr1", "chrX", "chrM"]
    write_index(dest_indexed, names, [2000, 1500, 1000, 1200, 500])
    (dest_indexed / "chrom_sizes.txt").write_text(
        "".join(f"{c}\t{s}\n" for c, s in zip(names, [2000, 1500, 1000, 1200, 500], strict=True)))
    create_alias_objects(db, tmp_path)
    expected = read_alias_objects()
    expected_arr = pm.gtrack_array_extract("arr", None, pm.gintervals_all())
    for t in TRACKS:
        rename_to_aliases(Path(db) / "tracks" / f"{t}.track")
    pm.gdb_reload()

    # tracks written from alias-named ones have files with canonical names
    pm.gtrack_create("sp_copy", "x", "sp", iterator="sp")
    pm.gtrack_array_import("arr_copy", "x", "arr")
    np.testing.assert_array_equal(pm.gextract("sp_copy", pm.gintervals_all())["sp_copy"], expected["sp"]["sp"])
    pd.testing.assert_frame_equal(pm.gtrack_array_extract("arr_copy", None, pm.gintervals_all()), expected_arr)
    chroms = set(pm.gintervals_all()["chrom"])
    for t in ("sp_copy", "arr_copy"):
        files = set(os.listdir(Path(db) / "tracks" / f"{t}.track"))
        assert {"chr1", "chrM"} <= files
        assert {f for f in files if not f.startswith(".")} <= chroms | {"vars"}

    # gtrack_modify changes the alias-named file
    pm.gtrack_modify("dn", "dn * 2", pm.gintervals_all())
    np.testing.assert_array_equal(pm.gextract("dn", pm.gintervals_all())["dn"], expected["dn"]["dn"] * 2)
    dn_files = os.listdir(Path(db) / "tracks" / "dn.track")
    assert "MT" in dn_files
    assert "chrM" not in dn_files

    # gtrack_copy into a per-chromosome and into an indexed database
    def by_chrom(res):
        return sorted(zip(res["chrom"].astype(str), res["start"], res["end"], res["sp"], strict=True))

    for dest in (dest_perchrom, str(dest_indexed.resolve())):
        pm.gdb_init(db)
        pm.gtrack_copy("sp", "sp", db=dest)
        pm.gdb_init(dest)
        assert by_chrom(pm.gextract("sp", pm.gintervals_all())) == by_chrom(expected["sp"]), dest


def test_a_canonically_named_file_wins_also_when_converted(tmp_path):
    """R: a canonically named file wins over a file named by an alias, also when converted,
    and the conversion's remove_old removes both."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals("chr1", 0, 100), [1])
    pm.gtrack_create_sparse("other", "x", pm.gintervals("chr1", 200, 300), [2])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d("chr1", 10, 15, "chr2", 30, 35), [1])
    pm.gtrack_2d_create("other2d", "x", pm.gintervals_2d("chr1", 100, 105, "chr2", 300, 305), [2])

    def read():
        return [pm.gextract("sp", pm.gintervals_all()), pm.gextract("rects", pm.gintervals_2d_all())]

    expected = read()
    tracks = Path(db) / "tracks"
    shutil.copy(tracks / "other.track" / "chr1", tracks / "sp.track" / "1")
    shutil.copy(tracks / "other2d.track" / "chr1-chr2", tracks / "rects.track" / "1-2")
    for got, exp in zip(read(), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)
    pm.gtrack_convert_to_indexed("sp", remove_old=True)
    pm.gtrack_2d_convert_to_indexed("rects", remove_old=True)
    assert not {"chr1", "1"} & set(os.listdir(tracks / "sp.track"))
    assert not {"chr1-chr2", "1-2"} & set(os.listdir(tracks / "rects.track"))
    for got, exp in zip(read(), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)

    # a big set R wrote: the canonical file wins over one named "1-chr2"
    bdb = bigset_db(tmp_path)
    expected = pm.gintervals_load("big2d")
    sdir = Path(bdb) / "tracks" / "big2d.interv"
    shutil.copy(sdir / "chr2-chr2", sdir / "1-chr2")
    pd.testing.assert_frame_equal(pm.gintervals_load("big2d"), expected)
    pm.gintervals_2d_convert_to_indexed("big2d", remove_old=True)
    assert not {"chr1-chr2", "1-chr2"} & set(os.listdir(sdir))
    pd.testing.assert_frame_equal(pm.gintervals_load("big2d"), expected)


def test_toggled_name_wins_and_two_other_aliases_conflict(tmp_path):
    """R: without a canonical name, the "chr"-toggled one wins, and two other alias-named
    files are an error (R's big set part needs a set on chrM, which no set R wrote here has)."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chrM"], 0, 100), [1, 2])
    pm.gtrack_create_sparse("other", "x", pm.gintervals("chrM", 200, 300), [3])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d("chr1", 10, 15, "chr2", 30, 35), [1])
    pm.gtrack_2d_create("other2d", "x", pm.gintervals_2d("chr1", 100, 105, "chr2", 300, 305), [2])
    expected = pm.gextract("sp", pm.gintervals_all())
    expected2d = pm.gextract("rects", pm.gintervals_2d_all())

    # chrM's chrom_sizes.txt name is "M": "M" wins over the other aliases ("MT", "chrMT")
    tracks = Path(db) / "tracks"
    sp_dir = tracks / "sp.track"
    (sp_dir / "chrM").rename(sp_dir / "M")
    shutil.copy(tracks / "other.track" / "chrM", sp_dir / "MT")
    pd.testing.assert_frame_equal(pm.gextract("sp", pm.gintervals_all()), expected)
    # without it, two other aliases
    (sp_dir / "M").rename(sp_dir / "chrMT")
    with pytest.raises(Exception, match="two files of chromosome chrM, MT and chrMT, and none named chrM or M"):
        pm.gextract("sp", pm.gintervals_all())
    with pytest.raises(Exception, match="two files of chromosome chrM"):
        pm.gtrack_convert_to_indexed("sp")
    # two names of one file are not two files
    (sp_dir / "MT").unlink()
    (sp_dir / "MT").symlink_to(sp_dir / "chrMT")
    pd.testing.assert_frame_equal(pm.gextract("sp", pm.gintervals_all()), expected)

    rects_dir = tracks / "rects.track"
    (rects_dir / "chr1-chr2").rename(rects_dir / "1-2")
    shutil.copy(tracks / "other2d.track" / "chr1-chr2", rects_dir / "chr1-2")
    pd.testing.assert_frame_equal(pm.gextract("rects", pm.gintervals_2d_all()), expected2d)
    (rects_dir / "1-2").rename(rects_dir / "1-chr2")
    with pytest.raises(Exception, match=r"two files of chromosome pair \(chr1, chr2\), 1-chr2 and chr1-2"):
        pm.gextract("rects", pm.gintervals_2d_all())
    with pytest.raises(Exception, match="two files of chromosome pair"):
        pm.gtrack_2d_convert_to_indexed("rects")



def test_a_conversion_stops_at_a_file_not_named_by_a_chromosome(tmp_path):
    """R: a conversion stops at a file whose name is not a chromosome's."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals("chr1", 0, 100), [1])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d("chr1", 10, 15, "chr2", 30, 35), [1])
    tracks = Path(db) / "tracks"
    (tracks / "sp.track" / "chr1.bak").write_text("notes")
    (tracks / "rects.track" / "chr1-chr7").write_text("notes")
    with pytest.raises(Exception, match="file chr1.bak is not named by a chromosome"):
        pm.gtrack_convert_to_indexed("sp")
    with pytest.raises(Exception, match="file chr1-chr7 is not named by a chromosome pair"):
        pm.gtrack_2d_convert_to_indexed("rects")

    bdb = bigset_db(tmp_path)
    for name in BIGSETS:
        (Path(bdb) / "tracks" / f"{name}.interv" / "notes").write_text("notes")
    with pytest.raises(ValueError, match="file notes is not named by a chromosome"):
        pm.gintervals_convert_to_indexed("big1d")
    with pytest.raises(ValueError, match="file notes is not named by a chromosome pair"):
        pm.gintervals_2d_convert_to_indexed("big2d")
    for obj in (tracks / "sp.track", tracks / "rects.track", *(Path(bdb) / "tracks" / f"{n}.interv" for n in BIGSETS)):
        assert not [f for f in os.listdir(obj) if f.endswith((".idx", ".dat"))], obj


def test_dangling_symlinks_are_ignored_and_symlinks_count_when_named_by_a_chromosome(tmp_path):
    """R: dangling symlinks are ignored, and a symlink counts only when named by a
    chromosome."""
    db = alias_db(tmp_path / "db")
    dest = alias_db(tmp_path / "dest")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d(["chr1", "chr2"], 10, 15, "chr2", 30, 35), [1, 2])

    def read():
        return [pm.gextract("sp", pm.gintervals_all()), pm.gextract("rects", pm.gintervals_2d_all())]

    expected = read()
    notes = tmp_path / "notes.txt"
    notes.write_text("Notes on this track")
    tracks = Path(db) / "tracks"
    for f in ("sp.track/chr1", "rects.track/chr1-chr2"):
        real = tmp_path / f.replace("/", "_")
        (tracks / f).rename(real)
        (tracks / f.replace("chr1", "1", 1)).symlink_to(real)
        (tracks / f).symlink_to(tmp_path / "missing")
        (tracks / Path(f).parent / "README").symlink_to(notes)
    pm.gdb_reload()
    assert pm.gtrack_info("sp")["type"] == "sparse"
    assert pm.gtrack_info("rects")["type"] == "rectangles"
    for got, exp in zip(read(), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)

    for t in ("sp.track", "rects.track"):
        (tracks / t / "README").unlink()
    # a copy stops at a dangling symlink (its target may come back)
    with pytest.raises(Exception):
        pm.gtrack_copy("rects", "rects", db=dest)
    for f in ("sp.track/chr1", "rects.track/chr1-chr2"):
        (tracks / f).unlink()
    pm.gtrack_copy("sp", "sp", db=dest)
    pm.gtrack_copy("rects", "rects", db=dest)
    pm.gtrack_convert_to_indexed("sp", remove_old=True)
    pm.gtrack_2d_convert_to_indexed("rects", remove_old=True)
    for got, exp in zip(read(), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)
    pm.gdb_init(dest)
    pd.testing.assert_frame_equal(pm.gextract("sp", pm.gintervals_all()), expected[0])
    pd.testing.assert_frame_equal(pm.gextract("rects", pm.gintervals_2d_all()), expected[1])


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_conversion_stops_at_a_file_it_cannot_read(tmp_path):
    """R: a conversion stops at a file it cannot read, and removes nothing."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d(["chr1", "chr2"], 10, 15, "chr2", 30, 35), [1, 2])
    bdb = bigset_db(tmp_path)
    tracks = Path(db) / "tracks"
    files = [tracks / "sp.track" / "chr2", tracks / "rects.track" / "chr2-chr2",
             Path(bdb) / "tracks" / "big1d.interv" / "chr2", Path(bdb) / "tracks" / "big2d.interv" / "chr2-chr2"]
    for f in files:
        os.chmod(f, 0)
    try:
        pm.gdb_init(db)
        with pytest.raises(Exception, match="Failed to open"):
            pm.gtrack_convert_to_indexed("sp", remove_old=True)
        with pytest.raises(Exception, match="Failed to open"):
            pm.gtrack_2d_convert_to_indexed("rects", remove_old=True)
        pm.gdb_init(bdb)
        with pytest.raises(OSError):
            pm.gintervals_convert_to_indexed("big1d", remove_old=True)
        with pytest.raises(OSError):
            pm.gintervals_2d_convert_to_indexed("big2d", remove_old=True)
        assert all(f.exists() for f in files)
        for obj in {f.parent for f in files}:
            assert not [f for f in os.listdir(obj) if f.endswith((".idx", ".dat", ".tmp"))], obj
    finally:
        for f in files:
            os.chmod(f, 0o644)


def test_a_forced_2d_conversion_keeps_its_index_when_the_pair_files_fail(tmp_path):
    """R: a forced conversion of an indexed 2D track keeps its index when the per-pair files
    fail the checks."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d(["chr1", "chr2"], 10, 15, ["chr1", "chrX"], 30, 35), [1, 2])
    pm.gtrack_2d_convert_to_indexed("rects", remove_old=True)
    expected = pm.gextract("rects", pm.gintervals_2d_all())
    d = Path(db) / "tracks" / "rects.track"
    (d / "README").write_text("notes")
    with pytest.raises(Exception, match="file README is not named by a chromosome pair"):
        pm.gtrack_2d_convert_to_indexed("rects", remove_old=True, force=True)
    assert all((d / f).exists() for f in ("track.idx", "track.dat"))
    pd.testing.assert_frame_equal(pm.gextract("rects", pm.gintervals_2d_all()), expected)

    # a pair file it cannot open stops it later: the old ones come back too
    (d / "README").unlink()
    if os.geteuid() != 0:
        (d / "chr1-chr2").write_text("x")
        os.chmod(d / "chr1-chr2", 0)
        with pytest.raises(Exception, match="Failed to open"):
            pm.gtrack_2d_convert_to_indexed("rects", remove_old=True, force=True)
        assert all((d / f).exists() for f in ("track.idx", "track.dat"))
        pd.testing.assert_frame_equal(pm.gextract("rects", pm.gintervals_2d_all()), expected)
        (d / "chr1-chr2").unlink()

    # once it succeeds nothing is left aside
    pm.gtrack_2d_convert_to_indexed("rects", remove_old=True, force=True)
    pd.testing.assert_frame_equal(pm.gextract("rects", pm.gintervals_2d_all()), expected)
    assert not [f for f in os.listdir(d) if f.endswith(".old")]
    assert {"track.idx", "track.dat"} <= set(os.listdir(d))


def test_toggled_names_convert_in_a_database_of_over_1000_contigs(tmp_path):
    """R: files named by the "chr" toggle convert in a database of over 1000 contigs
    without "chr" (whose aliases leave out the toggle)."""
    chroms = [f"s{i}" for i in range(1, 1002)]
    pc = tmp_path / "pc"
    (pc / "tracks").mkdir(parents=True)
    (pc / "seq").mkdir()
    for c in chroms:
        (pc / "seq" / f"chr{c}.seq").write_bytes(b"A" * 100)
    (pc / "chrom_sizes.txt").write_text("".join(f"{c}\t100\n" for c in chroms))
    ix = tmp_path / "ix"
    (ix / "tracks").mkdir(parents=True)
    write_index(ix, chroms, [100] * len(chroms))
    (ix / "chrom_sizes.txt").write_text("".join(f"{c}\t100\n" for c in chroms))

    pm.gdb_init(str(pc))
    iv = pm.gintervals(["chrs1", "chrs2", "chrs500"], [0, 0, 10], [50, 60, 20])
    pm.gtrack_create_sparse("sp", "x", iv, [1, 2, 3])
    assert (pc / "tracks" / "sp.track" / "chrs500").exists()
    shutil.copytree(pc / "tracks" / "sp.track", ix / "tracks" / "sp.track")

    pm.gdb_init(str(ix))
    pm.gtrack_convert_to_indexed("sp", remove_old=True)
    iv = pm.gintervals(["s1", "s2", "s500"], [0, 0, 10], [50, 60, 20])
    assert pm.gextract("sp", iv)["sp"].tolist() == [1, 2, 3]


def test_loading_one_chromosome_lists_only_without_its_canonical_file(tmp_path, monkeypatch):
    """R: loading one chromosome (pair) of a big set lists its directory only when the
    canonical file is missing."""
    bdb = bigset_db(tmp_path)
    listed = {"n": 0}
    chrom_files = _pymisha.pm_chrom_files

    def counting(*args):
        listed["n"] += 1
        return chrom_files(*args)

    monkeypatch.setattr(_pymisha, "pm_chrom_files", counting)
    assert len(pm.gintervals_load("big1d", chrom="chr2")) > 0
    assert len(pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr2")) > 0
    assert len(pm.gintervals_load("big2d", chrom2="chr2")) > 0
    assert listed["n"] == 0

    tracks = Path(bdb) / "tracks"
    (tracks / "big1d.interv" / "chr2").rename(tracks / "big1d.interv" / "2")
    (tracks / "big2d.interv" / "chr1-chr2").rename(tracks / "big2d.interv" / "1-2")
    assert len(pm.gintervals_load("big1d", chrom="chr2")) > 0
    assert len(pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr2")) > 0
    assert listed["n"] == 2


def test_alias_named_file_next_to_a_sets_index_is_ignored(tmp_path, monkeypatch):
    """R: next to an intervals set's index a file named by an alias is ignored, as older than
    the index; and such a set loads a chromosome without listing."""
    bdb = bigset_db(tmp_path)
    pm.gintervals_convert_to_indexed("big1d", remove_old=True)
    pm.gintervals_2d_convert_to_indexed("big2d", remove_old=True)
    expected = (pm.gintervals_load("big1d"), pm.gintervals_load("big2d"),
                pm.gintervals_load("big1d", chrom="chr1"), pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr2"))
    # as an earlier conversion left them: an older file named by an alias kept
    src = Path(__file__).parent / "r_parity" / "chromdb" / "db" / "tracks"
    tracks = Path(bdb) / "tracks"
    shutil.copy(src / "big1d.interv" / "chr2", tracks / "big1d.interv" / "1")
    shutil.copy(src / "big2d.interv" / "chr2-chr2", tracks / "big2d.interv" / "1-2")
    listed = {"n": 0}
    chrom_files = _pymisha.pm_chrom_files

    def counting(*args):
        listed["n"] += 1
        return chrom_files(*args)

    monkeypatch.setattr(_pymisha, "pm_chrom_files", counting)
    got = (pm.gintervals_load("big1d"), pm.gintervals_load("big2d"))
    for g, e in zip(got, expected[:2], strict=True):
        pd.testing.assert_frame_equal(g, e)
    listed["n"] = 0
    got = (pm.gintervals_load("big1d", chrom="chr1"), pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr2"))
    for g, e in zip(got, expected[2:], strict=True):
        pd.testing.assert_frame_equal(g, e)
    assert listed["n"] == 0
    assert len(pm.gextract("t1", intervals="big1d", iterator="big1d")) == len(expected[0])


def test_a_converted_2d_copy_that_stops_keeps_the_destination(tmp_path):
    """R: a converted 2D copy that stops at a file not named by a pair keeps the
    destination."""
    p = alias_db(tmp_path / "p")
    w = tmp_path / "w"
    (w / "tracks").mkdir(parents=True)
    write_index(w, ["chr1", "chr2"], [1000, 2000])
    (w / "chrom_sizes.txt").write_text("chr1\t1000\nchr2\t2000\n")
    pm.gdb_init(p)
    iv = pm.gintervals_2d(["chr1", "chr2"], 10, 15, "chr2", 30, 35)
    pm.gtrack_2d_create("src", "x", iv, [1, 2])
    pm.gtrack_2d_create("dst", "x", iv, [7, 8])
    for t in ("src.track", "dst.track"):
        shutil.copytree(Path(p) / "tracks" / t, w / "tracks" / t)
    (w / "tracks" / "src.track" / "README").write_text("notes")
    pm.gdb_init(str(w))
    pm.gtrack_2d_convert_to_indexed("dst", remove_old=True)
    expected = pm.gextract("dst", pm.gintervals_2d_all())

    with pytest.raises(Exception, match="file README is not named by a chromosome pair"):
        pm.gtrack_copy("src", "dst", overwrite=True)
    pd.testing.assert_frame_equal(pm.gextract("dst", pm.gintervals_2d_all()), expected)
    # the copy makes a file of a symlink, which the conversion stops at when it is not
    # named by a pair
    src_dir = w / "tracks" / "src.track"
    (src_dir / "README").unlink()
    (src_dir / "chr1").symlink_to(src_dir / "chr1-chr2")
    with pytest.raises(Exception, match="file chr1 is not named by a chromosome pair"):
        pm.gtrack_copy("src", "dst", overwrite=True)
    pd.testing.assert_frame_equal(pm.gextract("dst", pm.gintervals_2d_all()), expected)
    (src_dir / "chr1").unlink()
    # a file the conversion skips does not stop it
    (src_dir / "track.dat.tmp").write_text("x")
    pm.gtrack_copy("src", "dst", overwrite=True)
    assert pm.gextract("dst", pm.gintervals_2d_all())["dst"].tolist() == [1, 2]


def test_a_forced_reconversion_of_an_indexed_set_packs_its_alias_files(tmp_path):
    """R: a forced re-conversion of an indexed set packs its files named by aliases, as
    before (the update part of R's test needs per-chromosome writes of a big set, which
    pymisha does not do)."""
    bdb = bigset_db(tmp_path)
    expected = (pm.gintervals_load("big1d"), pm.gintervals_load("big2d"))
    for name in BIGSETS:
        sdir = Path(bdb) / "tracks" / f"{name}.interv"
        meta = (sdir / ".meta").read_bytes()
        rename_to_aliases(sdir)
        (sdir / ".meta").write_bytes(meta)
    # converted keeping the files
    pm.gintervals_convert_to_indexed("big1d")
    pm.gintervals_2d_convert_to_indexed("big2d")
    pm.gintervals_convert_to_indexed("big1d", remove_old=True, force=True)
    pm.gintervals_2d_convert_to_indexed("big2d", remove_old=True, force=True)
    for got, exp in zip((pm.gintervals_load("big1d"), pm.gintervals_load("big2d")), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)
    assert sorted(f for f in os.listdir(Path(bdb) / "tracks" / "big1d.interv") if not f.startswith(".")) == [
        "intervals.dat", "intervals.idx"]
    assert sorted(f for f in os.listdir(Path(bdb) / "tracks" / "big2d.interv") if not f.startswith(".")) == [
        "intervals2d.dat", "intervals2d.idx"]


def test_a_symlink_named_by_the_other_dimensions_names_is_ignored(tmp_path):
    """R: a symlink named only by the other dimension's names is ignored."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d(["chr1", "chr2"], 10, 15, "chr2", 30, 35), [1, 2])

    def read():
        return [pm.gextract("sp", pm.gintervals_all()), pm.gextract("rects", pm.gintervals_2d_all())]

    expected = read()
    tracks = Path(db) / "tracks"
    (tracks / "sp.track" / "chr1-chr2").symlink_to(tracks / "rects.track" / "chr1-chr2")
    (tracks / "rects.track" / "chr1").symlink_to(tracks / "sp.track" / "chr1")
    pm.gdb_reload()
    assert pm.gtrack_info("sp")["type"] == "sparse"
    assert pm.gtrack_info("rects")["type"] == "rectangles"
    for got, exp in zip(read(), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)
    pm.gtrack_convert_to_indexed("sp", remove_old=True)
    pm.gtrack_2d_convert_to_indexed("rects", remove_old=True)
    for got, exp in zip(read(), expected, strict=True):
        pd.testing.assert_frame_equal(got, exp)
    # the conversions' remove_old keeps them
    assert (tracks / "sp.track" / "chr1-chr2").is_symlink()
    assert (tracks / "rects.track" / "chr1").is_symlink()


def test_a_file_named_by_the_destinations_mitochondrial_name_is_copied_to_it(tmp_path):
    """R: a file named by the destination's mitochondrial name is copied to it across
    genomes."""
    src = _mkdb(tmp_path / "src", ["1", "MT"])
    dest = _mkdb(tmp_path / "dest", ["1", "M"])
    pm.gdb_init(src)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chrMT"], 0, 100), [1, 2])
    d = Path(src) / "tracks" / "sp.track"
    (d / "chrMT").rename(d / "M")
    pm.gdb_reload()
    assert pm.gextract("sp", pm.gintervals("chrMT", 0, 1000))["sp"].tolist() == [2]
    pm.gtrack_copy("sp", "sp", db=dest)
    pm.gdb_init(dest)
    assert pm.gextract("sp", pm.gintervals(["chr1", "chrM"], 0, 1000))["sp"].tolist() == [1, 2]


def test_a_copy_of_a_file_with_two_names_reads(tmp_path):
    """R: a copy of a file with two names (a symlink), neither canonical, reads; different
    bytes are two files."""
    db = alias_db(tmp_path / "db")
    dest = alias_db(tmp_path / "dest")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chrM"], 0, 100), [1, 2])
    d = Path(db) / "tracks" / "sp.track"
    (d / "chrM").rename(d / "MT")
    (d / "chrMT").symlink_to("MT")
    pm.gdb_reload()
    assert pm.gextract("sp", pm.gintervals_all())["sp"].tolist() == [1, 2]
    # each copy makes a file of each name: the same bytes are not two files of chrM
    pm.gtrack_copy("sp", "sp2")
    assert pm.gextract("sp2", pm.gintervals_all())["sp2"].tolist() == [1, 2]
    pm.gtrack_copy("sp", "sp", db=dest)
    pm.gdb_init(dest)
    assert pm.gextract("sp", pm.gintervals_all())["sp"].tolist() == [1, 2]
    # different bytes are
    pm.gdb_init(db)
    d2 = Path(db) / "tracks" / "sp2.track"
    if (d2 / "chrMT").exists() or (d2 / "chrMT").is_symlink():
        (d2 / "chrMT").unlink()
    if not (d2 / "MT").exists():
        pytest.skip("the copy took the canonical name")
    shutil.copy(d2 / "chr1", d2 / "chrMT")
    pm.gdb_reload()
    with pytest.raises(Exception, match="two files of chromosome chrM"):
        pm.gextract("sp2", pm.gintervals_all())


@pytest.mark.parametrize("how", ["symlink", "hardlink", "copy"])
def test_modify_of_a_chromosome_with_two_names_leaves_both_naming_the_new_data(tmp_path, how):
    """R: gtrack.modify of a chromosome with two names, neither canonical, leaves both
    naming the new data."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_dense("d", "x", pm.gintervals(["chr1", "chrM"], 0, 30), [1, 1], 10, 0)
    d = Path(db) / "tracks" / "d.track"
    (d / "chrM").rename(d / "MT")
    if how == "symlink":
        (d / "M").symlink_to("MT")
    elif how == "hardlink":
        os.link(d / "MT", d / "M")
    else:
        shutil.copy(d / "MT", d / "M")
    pm.gdb_reload()
    pm.gtrack_modify("d", "d * 7", pm.gintervals("chrM", 0, 30))
    assert pm.gextract("d", pm.gintervals("chrM", 0, 30))["d"].tolist() == [7, 7, 7]
    assert pm.gextract("d", pm.gintervals("chr1", 0, 30))["d"].tolist() == [1, 1, 1]
    # both name the modified file; no canonical name is added
    assert (d / "M").read_bytes() == (d / "MT").read_bytes()
    assert "chrM" not in os.listdir(d)


def test_a_copy_across_genomes_takes_the_name_the_destination_has(tmp_path):
    """R: a copy across genomes takes the name of a chromosome's file the destination has."""
    src = _mkdb(tmp_path / "src", ["1", "M"])
    dest = _mkdb(tmp_path / "dest", ["1", "MT"])
    pm.gdb_init(src)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chrM"], 0, 100), [1, 2])
    d = Path(src) / "tracks" / "sp.track"
    (d / "chrM").rename(d / "M")
    shutil.copy(d / "M", d / "MT")
    pm.gdb_reload()
    pm.gtrack_copy("sp", "sp", db=dest)
    pm.gdb_init(dest)
    assert pm.gextract("sp", pm.gintervals(["chr1", "chrMT"], 0, 1000))["sp"].tolist() == [1, 2]


def test_a_symlink_to_a_file_that_modify_replaced_reads_the_new_data(tmp_path):
    """R: a symlink to a file that gtrack.modify replaced reads the new data."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_dense("d", "x", pm.gintervals(["chr1", "chr2"], 0, 30), [1, 2], 10, 0)
    tracks = Path(db) / "tracks"
    (tracks / "d.track" / "chr1").rename(tracks / "d.track" / "1")
    # another track's chr1 is d's file
    (tracks / "e.track").mkdir()
    for f in os.listdir(tracks / "d.track"):
        if f != "1":
            src = tracks / "d.track" / f
            if src.is_dir():
                shutil.copytree(src, tracks / "e.track" / f)
            else:
                shutil.copy(src, tracks / "e.track" / f)
    (tracks / "e.track" / "chr1").symlink_to(tracks / "d.track" / "1")
    pm.gdb_reload()
    assert pm.gextract("e", pm.gintervals("chr1", 0, 30))["e"].tolist() == [1, 1, 1]
    pm.gtrack_modify("d", "d * 7", pm.gintervals("chr1", 0, 30))
    assert pm.gextract("d", pm.gintervals("chr1", 0, 30))["d"].tolist() == [7, 7, 7]
    assert pm.gextract("e", pm.gintervals("chr1", 0, 30))["e"].tolist() == [7, 7, 7]


def test_an_indexed_big_set_does_not_return_a_deleted_chromosome(tmp_path):
    """R: an indexed big set does not return a chromosome deleted with gintervals.update
    (NULL): R's update leaves the chromosome's entry in the index and drops it from .meta
    (big1d_without_chr1.meta, which R wrote so), and a load goes by .meta. pymisha's own
    gintervals_update rewrites the whole set."""
    bdb = bigset_db(tmp_path)
    pm.gintervals_convert_to_indexed("big1d", remove_old=True)
    before = pm.gintervals_load("big1d")
    assert "chr1" in set(before["chrom"])
    sdir = Path(bdb) / "tracks" / "big1d.interv"
    shutil.copy(Path(__file__).parent / "r_parity" / "chromdb" / "big1d_without_chr1.meta", sdir / ".meta")
    after = pm.gintervals_load("big1d")
    pd.testing.assert_frame_equal(after, before[before["chrom"] != "chr1"].reset_index(drop=True))
    # and through pymisha's update
    pm.gintervals_update("big1d", None, chrom="chr2")
    assert set(pm.gintervals_load("big1d")["chrom"]) == set(after["chrom"]) - {"chr2"}


def test_modify_keeps_a_file_named_by_an_alias(tmp_path):
    """R: gtrack.modify keeps a file named by an alias where earlier pymisha also writes it;
    and pymisha 0.14's write of "1" (renaming a new file over it) is read."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_dense("d", "x", pm.gintervals(["chr1", "chr2"], 0, 30), [1, 2], 10, 0)
    d = Path(db) / "tracks" / "d.track"
    files = sorted(f for f in os.listdir(d) if f.startswith("chr"))
    for f, a in zip(files, alias_names(files), strict=True):
        (d / f).rename(d / a)
    pm.gdb_reload()
    pm.gtrack_modify("d", "d + 100", pm.gintervals("chr1", 0, 30))
    assert sorted(f for f in os.listdir(d) if not f.startswith(".") and f != "vars") == sorted(alias_names(files))
    saved = tmp_path / "chr1_101"
    shutil.copy(d / "1", saved)
    pm.gtrack_modify("d", "d + 1000", pm.gintervals("chr1", 0, 30))
    assert pm.gextract("d", pm.gintervals("chr1", 0, 30))["d"].tolist() == [1101, 1101, 1101]
    os.replace(saved, d / "1")
    pm.gdb_reload()
    assert pm.gextract("d", pm.gintervals("chr1", 0, 30))["d"].tolist() == [101, 101, 101]


@pytest.mark.parametrize("how", ["symlink", "hardlink", "copy"])
def test_after_modify_a_write_of_the_toggled_name_is_read(tmp_path, how):
    """R: after gtrack.modify, earlier pymisha's write of the "chr"-toggled name is read."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_dense("d", "x", pm.gintervals(["chr1", "chrM"], 0, 30), [1, 5], 10, 0)
    d = Path(db) / "tracks" / "d.track"
    (d / "chrM").rename(d / "M")
    if how == "symlink":
        (d / "MT").symlink_to("M")
    elif how == "hardlink":
        os.link(d / "M", d / "MT")
    else:
        shutil.copy(d / "M", d / "MT")
    pm.gdb_reload()
    saved = tmp_path / "M_5"
    shutil.copy(d / "M", saved)
    pm.gtrack_modify("d", "d + 100", pm.gintervals("chrM", 0, 30))
    assert pm.gextract("d", pm.gintervals("chrM", 0, 30))["d"].tolist() == [105, 105, 105]
    # earlier pymisha writes chrM's data by renaming a new file over "M", its name for it
    os.replace(saved, d / "M")
    pm.gdb_reload()
    assert pm.gextract("d", pm.gintervals("chrM", 0, 30))["d"].tolist() == [5, 5, 5]
    assert pm.gextract("d", pm.gintervals("chr1", 0, 30))["d"].tolist() == [1, 1, 1]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_file_behind_an_unreachable_symlink_is_an_error(tmp_path):
    """R: a file read through a symlink that cannot be reached is an error, not a missing
    chromosome; a canonically named file wins over it."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d(["chr1", "chr2"], 10, 15, "chr2", 30, 35), [1, 2])
    expected = pm.gextract("sp", pm.gintervals_all())
    tracks = Path(db) / "tracks"
    closed = tmp_path / "closed"
    closed.mkdir()
    for f in ("sp.track/chr2", "rects.track/chr1-chr2"):
        target = closed / Path(f).name
        (tracks / f).rename(target)
        (tracks / f).symlink_to(target)
    os.chmod(closed, 0)
    try:
        pm.gdb_reload()
        with pytest.raises(Exception, match="Permission denied"):
            pm.gextract("sp", pm.gintervals_all())
        with pytest.raises(Exception, match="Permission denied"):
            pm.gextract("rects", pm.gintervals_2d_all())
        with pytest.raises(Exception):
            pm.gtrack_copy("rects", "rects2")
        assert not pm.gtrack_exists("rects2")
        # a canonically named file wins over it
        os.chmod(closed, 0o755)
        (tracks / "sp.track" / "chr2").unlink()
        shutil.copy(closed / "chr2", tracks / "sp.track" / "chr2")
        (tracks / "sp.track" / "2").symlink_to(closed / "chr2")
        os.chmod(closed, 0)
        pm.gdb_reload()
        pd.testing.assert_frame_equal(pm.gextract("sp", pm.gintervals_all()), expected)
        assert pm.gtrack_info("sp")["type"] == "sparse"
    finally:
        os.chmod(closed, 0o755)


def test_a_leftover_tmp_file_does_not_stop_a_2d_conversion(tmp_path):
    """R: a leftover *.tmp file does not stop a 2D conversion (the gintervals.update part
    needs per-chromosome writes of a big set)."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_2d_create("rects", "x", pm.gintervals_2d(["chr1", "chr2"], 10, 15, "chr2", 30, 35), [1, 2])
    expected = pm.gextract("rects", pm.gintervals_2d_all())
    (Path(db) / "tracks" / "rects.track" / "chr1-chr2.tmp").write_text("x")
    pm.gtrack_2d_convert_to_indexed("rects", remove_old=True)
    pd.testing.assert_frame_equal(pm.gextract("rects", pm.gintervals_2d_all()), expected)


def test_a_chromosome_ending_in_tmp_is_read_and_a_copy_matches_by_file_name(tmp_path):
    """R: a chromosome whose name ends in .tmp is read, and a copy matches a destination by
    a file's own name."""
    db = _mkdb(tmp_path / "db", ["1", "Un.tmp"])
    pm.gdb_init(db)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chrUn.tmp"], 0, 100), [1, 2])
    assert pm.gextract("sp", pm.gintervals_all())["sp"].tolist() == [1, 2]
    pm.gtrack_convert_to_indexed("sp", remove_old=True)
    assert pm.gextract("sp", pm.gintervals_all())["sp"].tolist() == [1, 2]

    # chrM's file named "MT", copied into a database whose only chromosome is chrMT
    src = _mkdb(tmp_path / "src", ["1", "M"])
    dest = _mkdb(tmp_path / "dest", ["MT"])
    pm.gdb_init(src)
    pm.gtrack_create_sparse("m", "x", pm.gintervals("chrM", 0, 100), [3])
    d = Path(src) / "tracks" / "m.track"
    (d / "chrM").rename(d / "MT")
    pm.gdb_reload()
    pm.gtrack_copy("m", "m", db=dest)
    pm.gdb_init(dest)
    assert pm.gextract("m", pm.gintervals_all())["m"].tolist() == [3]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_overwriting_copy_that_fails_keeps_the_existing_destination(tmp_path):
    """R: an overwriting gtrack.copy that fails keeps the existing destination."""
    db = alias_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("src", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_create_sparse("victim", "x", pm.gintervals("chr1", 500, 600), [42])
    tracks = Path(db) / "tracks"
    # a leftover the listing skips, which the copy cannot read
    unreadable = tracks / "src.track" / "notes.tmp"
    unreadable.write_text("x")
    os.chmod(unreadable, 0)
    try:
        with pytest.raises(Exception):
            pm.gtrack_copy("src", "victim", overwrite=True)
        assert pm.gextract("victim", pm.gintervals_all())["victim"].tolist() == [42]
        assert pm.gtrack_exists("victim")
        assert not [n for n in os.listdir(tracks) if n.startswith(".victim.track.tmp.")]
        # once the copy can succeed it replaces the destination and leaves nothing aside
        os.chmod(unreadable, 0o644)
        pm.gtrack_copy("src", "victim", overwrite=True)
        assert pm.gextract("victim", pm.gintervals_all())["victim"].tolist() == [1, 2]
        assert not [n for n in os.listdir(tracks) if n.startswith(".victim.track.tmp.")]
        # a database that cannot be written stops the copy before anything is touched
        pm.gtrack_create_sparse("other", "x", pm.gintervals("chr2", 0, 100), [5])
        os.chmod(tracks, 0o555)
        with pytest.raises(Exception):
            pm.gtrack_copy("other", "victim", overwrite=True)
        os.chmod(tracks, 0o755)
        assert pm.gextract("victim", pm.gintervals_all())["victim"].tolist() == [1, 2]
    finally:
        os.chmod(tracks, 0o755)
        os.chmod(unreadable, 0o644)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_failed_overwriting_copy_restores_the_destination_and_its_registration(tmp_path):
    """R: a failed overwriting copy restores the destination even when the copy cannot be
    removed, and its registration (into a dataset whose track the working database's
    shadows)."""
    w = alias_db(tmp_path / "w")
    d = alias_db(tmp_path / "d")
    pm.gdb_init(d)
    pm.gtrack_create_sparse("t", "x", pm.gintervals("chr1", 0, 100), [2])
    pm.gdb_init(w)
    pm.gtrack_create_sparse("t", "x", pm.gintervals("chr1", 0, 100), [1])
    pm.gtrack_create_sparse("src", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [5, 6])
    src_dir = Path(w) / "tracks" / "src.track"
    (src_dir / "zz.tmp").write_text("x")
    os.chmod(src_dir / "zz.tmp", 0)
    try:
        pm.gdataset_load(d, force=True)
        assert pm.gextract("t", pm.gintervals("chr1", 0, 100))["t"].tolist() == [1]
        with pytest.raises(Exception):
            pm.gtrack_copy("src", "t", db=d, overwrite=True)
        assert pm.gextract("t", pm.gintervals("chr1", 0, 100))["t"].tolist() == [1]
        assert pm.gtrack_dataset("t") == w
        assert (Path(d) / "tracks" / "t.track" / "chr1").exists()
        pm.gdb_reload()
        assert pm.gextract("t", pm.gintervals("chr1", 0, 100))["t"].tolist() == [1]
        pm.gdb_init(d)
        assert pm.gextract("t", pm.gintervals("chr1", 0, 100))["t"].tolist() == [2]
    finally:
        os.chmod(src_dir / "zz.tmp", stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.parametrize("pure_python", [False, True])
def test_liftover_finds_source_files_named_by_an_alias(tmp_path, pure_python):
    """R (test-gtrack.liftover.R): gtrack.liftover finds per-chromosome source files named
    by a chromosome alias ("1" for chr1, pairs such as "1-1", "1-chr2" or "chr2-1")."""
    src = tmp_path / "src"
    (src / "tracks").mkdir(parents=True)
    (src / "seq").mkdir()
    (src / "seq" / "chr1.seq").write_bytes(b"A" * 1000)
    (src / "seq" / "chr2.seq").write_bytes(b"C" * 2000)
    (src / "chrom_sizes.txt").write_text("2\t2000\n1\t1000\n")
    pm.gdb_init(str(src))
    assert pm.gintervals_all()["chrom"].tolist() == ["chr1", "chr2"]
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_create("dn", "x", "sp", iterator=100)
    contacts = tmp_path / "contacts.tsv"
    pd.DataFrame({
        "chrom1": ["chr1", "chr2", "chr1"], "start1": [10, 20, 50], "end1": [11, 21, 51],
        "chrom2": ["chr1", "chr2", "chr2"], "start2": [30, 40, 60], "end2": [31, 41, 61], "value": [1, 2, 3],
    }).to_csv(contacts, sep="\t", index=False)
    pm.gtrack_2d_import_contacts("pts", "x", str(contacts), fends=None)

    tgt = _mkdb(tmp_path / "tgt", ["T1", "T2"])
    (Path(tgt) / "seq" / "chrT2.seq").write_bytes(b"C" * 2000)
    (Path(tgt) / "chrom_sizes.txt").write_text("T1\t1000\nT2\t2000\n")
    chain = tmp_path / "chain.txt"
    chain.write_text(
        "chain 1000 chr1 1000 + 0 1000 chrT1 1000 + 0 1000 1\n1000\n\n"
        "chain 1000 chr2 2000 + 0 2000 chrT2 2000 + 0 2000 2\n2000\n\n"
    )
    pm.gdb_init(tgt)
    tracks = ["sp", "dn", "pts"]

    def lift_all(prefix):
        for t in tracks:
            pm.gtrack_liftover(f"{prefix}{t}", "x", str(src / "tracks" / f"{t}.track"), str(chain),
                               _force_pure_python=pure_python)

    def extract_all(prefix):
        return {
            "sp": pm.gextract(f"{prefix}sp", pm.gintervals_all(), colnames=["v"]),
            "dn": pm.gextract(f"{prefix}dn", pm.gintervals(["chrT1", "chrT2"], 0, 100), colnames=["v"]),
            "pts": pm.gextract(f"{prefix}pts", pm.gintervals_2d_all(), colnames=["v"]),
        }

    lift_all("canonical_")
    expected = extract_all("canonical_")
    assert expected["sp"]["v"].tolist() == [1, 2]
    assert len(expected["pts"]) == 6

    for t in tracks:
        rename_to_aliases(src / "tracks" / f"{t}.track", expect_min=2)
    lift_all("alias_")
    got = extract_all("alias_")
    for k in expected:
        pd.testing.assert_frame_equal(got[k], expected[k], obj=k)
