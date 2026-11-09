"""gtrack_copy reads and writes indexed tracks by each database's chrom ids, copies per-pair
2D tracks by chromosome name, and checks everything before it touches an existing
destination, which it sets aside until the copy has succeeded. Ported from R misha 5.13.1
and 5.13.3 (PRs #193 and #198, tests/testthat/test-track-copy-crossdb.R and
test-track-rm-fast.R)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from _chromdb_helpers import (
    SORTED_PREFIXED,
    create_test_db,
    unsorted_db,
    write_index,
)

import pymisha as pm
import pymisha.tracks as tracks_mod
from pymisha._shared import _gdb_chrom_names_at
from pymisha.tracks import _db_is_indexed_at


def _values_by_chrom(res, col):
    return dict(zip(res["chrom"].astype(str), res[col], strict=True))


def test_copy_splits_an_indexed_track_by_the_source_chrom_ids(tmp_path):
    """R: gtrack.copy splits an indexed track of a per-chromosome db by the chrom ids
    gsetroot gives it (into a per-chromosome and an indexed database)."""
    src = unsorted_db(tmp_path / "src")
    dest_perchrom = unsorted_db(tmp_path / "dest_perchrom")
    dest_indexed = create_test_db(tmp_path / "dest_indexed", [
        ("chr2", 2000), ("chr10", 1500), ("chr1", 1000), ("chrX", 1200), ("chr1_KI270706v1_random", 500),
    ])
    pm.gdb_init(dest_indexed)
    pm.gdb_convert_to_indexed()

    pm.gdb_init(src)
    iv = pm.gintervals_all()
    iv["end"] = 100
    pm.gtrack_create_sparse("sp", "x", iv, list(range(1, len(iv) + 1)))
    pm.gtrack_convert_to_indexed("sp")
    expected = _values_by_chrom(pm.gextract("sp", pm.gintervals_all()), "sp")

    for db in (dest_perchrom, dest_indexed):
        pm.gdb_init(src)
        pm.gtrack_copy("sp", "sp_copy", db=db)
        pm.gdb_init(db)
        res = _values_by_chrom(pm.gextract("sp_copy", pm.gintervals_all()), "sp_copy")
        assert {c: res[c] for c in expected} == expected, db


@pytest.mark.parametrize("seq_state", ["missing", "empty"])
def test_dataset_without_sequence_numbers_chromosomes_as_its_db(tmp_path, seq_state):
    """R: a dataset of a per-chromosome db with its seq/ missing or empty numbers
    chromosomes as the db does."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals(["chr1", "chr2", "chr10", "chrX"], 0, 100)
    pm.gtrack_create_sparse("t1", "x", iv, [1, 2, 10, 23])
    pm.gtrack_convert_to_indexed("t1")
    pm.gtrack_2d_create("r2", "x", pm.gintervals_2d("chr1", 10, 15, "chr2", 30, 35), [7])
    expected = pm.gextract("t1", pm.gintervals_all())
    expected2d = pm.gextract("r2", pm.gintervals_2d_all())

    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="t1")
    (ds / "seq").unlink()
    if seq_state == "empty":
        (ds / "seq").mkdir()
    pm.gtrack_rm("t1", force=True)
    pm.gdataset_load(str(ds))
    assert _gdb_chrom_names_at(str(ds)) == SORTED_PREFIXED

    # an indexed track copied out of the dataset keeps its values on their chromosomes
    pm.gtrack_copy("t1", "t1c")
    assert pm.gextract("t1c", pm.gintervals_all())["t1c"].tolist() == expected["t1"].tolist()
    # a 2D track copied into the dataset
    pm.gtrack_copy("r2", "r2c", db=str(ds))
    assert pm.gextract("r2c", pm.gintervals_2d_all())["r2c"].tolist() == expected2d["r2"].tolist()


def test_dataset_of_an_indexed_db_without_its_seq_is_indexed_as_the_db(tmp_path):
    """R: a dataset of an indexed db without its seq/ link is indexed as the db is; an
    indexed track is copied into it in the database's format, other tracks as they are."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    # tracks in per-chromosome and per-pair files, made before the database was converted
    pm.gtrack_create_sparse("v2", "x", pm.gintervals(["chr1", "chr2"], 0, 100), [7, 8])
    pm.gtrack_2d_create("r2", "x", pm.gintervals_2d("chr1", 10, 15, "chr2", 30, 35), [7])
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    pm.gdb_init(db)
    assert _db_is_indexed_at(db)

    iv = pm.gintervals(["chr1", "chr2"], 0, 100)
    pm.gtrack_create_sparse("t1", "x", iv, [1, 2])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="t1")
    (ds / "seq").unlink()
    pm.gtrack_rm("t1", force=True)
    pm.gdataset_load(str(ds))
    assert _db_is_indexed_at(str(ds))
    # nor does a genome.seq without genome.idx make a sequence of its own
    (ds / "seq").mkdir()
    (ds / "seq" / "genome.seq").write_bytes(b"")
    assert _db_is_indexed_at(str(ds))
    shutil.rmtree(ds / "seq")

    pm.gtrack_create_sparse("v1", "x", iv, [5, 6])
    pm.gtrack_convert_to_indexed("v1")
    pm.gtrack_copy(["v1", "v2", "r2"], "c", db=str(ds))
    assert pm.gtrack_info("c.v1")["format"] == "indexed"
    assert not (ds / "tracks" / "c" / "v2.track" / "track.idx").exists()
    assert not (ds / "tracks" / "c" / "r2.track" / "track.idx").exists()
    assert pm.gextract("c.v1", iv)["c.v1"].tolist() == [5, 6]
    assert pm.gextract("c.v2", iv)["c.v2"].tolist() == [7, 8]
    assert pm.gextract("c.r2", pm.gintervals_2d_all())["c.r2"].tolist() == [7]


def test_unloaded_db_without_sequence_gives_no_order(tmp_path):
    """R: an unloaded database whose seq/ holds no sequence gives no chrom order to guess
    from; copies into it go by chromosome name."""
    from pymisha.liftover import _load_source_chrom_names

    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals(["chr1", "chr2"], 0, 100)
    pm.gtrack_create_sparse("sp", "x", iv, [1, 2])
    pm.gtrack_create_sparse("spi", "x", iv, [3, 4])
    pm.gtrack_convert_to_indexed("spi")

    other = tmp_path / "other"
    (other / "tracks").mkdir(parents=True)
    shutil.copy(Path(db) / "chrom_sizes.txt", other / "chrom_sizes.txt")
    with pytest.raises(ValueError, match="cannot be told"):
        _gdb_chrom_names_at(str(other))

    # a liftover of an indexed source track out of it needs the order
    src = other / "tracks" / "lift.track"
    src.mkdir()
    with pytest.raises(ValueError, match="cannot be told"):
        _load_source_chrom_names(str(src))
    shutil.rmtree(src)
    # a genome.idx without genome.seq is no sequence either
    (other / "seq").mkdir()
    (other / "seq" / "genome.idx").write_bytes(bytes(16))
    with pytest.raises(ValueError, match="cannot be told"):
        _gdb_chrom_names_at(str(other))
    shutil.rmtree(other / "seq")

    # copies into it go by chromosome name: the source's order is known, as it is loaded
    pm.gtrack_copy("sp", "sp_copy", db=str(other))
    pm.gtrack_copy("spi", "spi_copy", db=str(other))
    pm.gdataset_load(str(other))
    assert pm.gextract("sp_copy", iv)["sp_copy"].tolist() == [1, 2]
    assert pm.gextract("spi_copy", iv)["spi_copy"].tolist() == [3, 4]


def test_copy_into_db_of_unknown_order_copies_per_pair_2d_by_name(tmp_path):
    """R: gtrack.copy into a database whose order cannot be told copies per-pair 2D tracks
    by name, and checks before overwriting."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    pairs = pm.gintervals_2d(["chr1", "chr10"], 10, 15, ["chr2", "chrX"], 30, 35)
    pm.gtrack_2d_create("r2", "x", pairs, [12, 1023])
    pm.gtrack_2d_create("r2i", "x", pairs, [5, 6])
    pm.gtrack_2d_convert_to_indexed("r2i")
    # a tracks-only copy of the database: no seq/, unprefixed names
    other = tmp_path / "other"
    (other / "tracks").mkdir(parents=True)
    shutil.copy(Path(db) / "chrom_sizes.txt", other / "chrom_sizes.txt")

    pm.gtrack_copy("r2", "r2c", db=str(other))
    old = sorted(os.listdir(other / "tracks" / "r2c.track"))
    # an indexed 2D track needs the order: it stops before the existing track is touched
    with pytest.raises(ValueError, match="cannot be told"):
        pm.gtrack_copy("r2i", "r2c", db=str(other), overwrite=True)
    assert sorted(os.listdir(other / "tracks" / "r2c.track")) == old
    pm.gdataset_load(str(other))
    assert pm.gextract("r2c", pm.gintervals_2d_all())["r2c"].tolist() == [12, 1023]

    # a chr-prefixed first name and a .seq file without the prefix still show the order
    pm.gdb_unload()
    third = tmp_path / "third"
    (third / "seq").mkdir(parents=True)
    (third / "chrom_sizes.txt").write_text("chrA\t10\nB\t10\nC\t10\nD\t10\nE\t10\n")
    (third / "seq" / "A.seq").write_bytes(b"A" * 10)
    assert _gdb_chrom_names_at(str(third))[0] == "chrA"


def test_copy_per_pair_2d_between_prefixed_and_unprefixed_names(tmp_path):
    """R: gtrack.copy copies a per-pair 2D track between chr-prefixed and unprefixed names
    of the same chromosomes."""
    unprefixed = unsorted_db(tmp_path / "unprefixed")
    prefixed = unsorted_db(tmp_path / "prefixed")
    cs = Path(prefixed) / "chrom_sizes.txt"
    cs.write_text("".join(f"chr{line}\n" for line in cs.read_text().splitlines()))
    for src, dest in ((prefixed, unprefixed), (unprefixed, prefixed)):
        pm.gdb_init(src)
        pairs = pm.gintervals_2d(["chr1", "chr10"], 10, 15, ["chr2", "chrX"], 30, 35)
        # (gintervals_2d orders the pairs by the database's chromosome ids, as R)
        pm.gtrack_2d_create("r2", "x", pairs, [12 if c == "chr1" else 1023 for c in pairs["chrom1"]])
        pm.gtrack_copy("r2", "r2c", db=dest)
        pm.gtrack_rm("r2", force=True)
        pm.gdb_init(dest)
        res = pm.gextract("r2c", pm.gintervals_2d_all())
        got = dict(zip(zip(res["chrom1"], res["chrom2"], strict=True), res["r2c"], strict=True))
        assert got == {("chr1", "chr2"): 12, ("chr10", "chrX"): 1023}, src
        pm.gtrack_rm("r2c", force=True)


def test_overwriting_copy_into_an_unloaded_db(tmp_path):
    """R: gtrack.copy with overwrite = TRUE into an unloaded database leaves the session's
    tracks out of its cache (pymisha keeps no .db.cache: the database lists only its own
    tracks)."""
    src = unsorted_db(tmp_path / "src")
    dest = unsorted_db(tmp_path / "dest")
    pm.gdb_init(src)
    pm.gtrack_create_sparse("sp", "x", pm.gintervals(["chr1", "chr2"], 0, 10), [1, 2])
    pm.gtrack_create_sparse("other", "x", pm.gintervals("chr1", 0, 10), [3])
    pm.gtrack_copy("sp", "sp_c", db=dest)
    pm.gtrack_copy("other", "sp_c", db=dest, overwrite=True)
    pm.gdb_init(dest)
    assert pm.gtrack_ls() == ["sp_c"]
    assert pm.gextract("sp_c", pm.gintervals("chr1", 0, 10))["sp_c"].tolist() == [3]
    # the replaced track is gone, not left hidden
    assert not [n for n in os.listdir(Path(dest) / "tracks") if n.startswith(".sp_c")]


def test_copy_of_a_track_onto_its_own_name_stops(tmp_path):
    """R: gtrack.copy of a track onto its own name stops, also in a subdirectory."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("t", "x", pm.gintervals("chr1", 0, 10), [1])
    with pytest.raises(ValueError, match="Source and destination are the same track"):
        pm.gtrack_copy("t", "t", overwrite=True)
    assert pm.gextract("t", pm.gintervals("chr1", 0, 10))["t"].tolist() == [1]
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("sub.t", "x", pm.gintervals("chr1", 0, 10), [2])
    pm.gdir_cd("sub")
    with pytest.raises(ValueError, match="Source and destination are the same track"):
        pm.gtrack_copy("t", "t", overwrite=True)
    pm.gdir_cd("..")
    assert pm.gextract("t", pm.gintervals("chr1", 0, 10))["t"].tolist() == [1]
    assert pm.gextract("sub.t", pm.gintervals("chr1", 0, 10))["sub.t"].tolist() == [2]


def test_overwriting_copy_stops_for_the_same_directory_under_two_names(tmp_path):
    """R: gtrack.copy with overwrite = TRUE stops for the same directory under two names."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("sub.t", "x", pm.gintervals("chr1", 0, 10), [1])
    # tracks/alias is a link to tracks/sub: alias.t is sub.t
    (Path(db) / "tracks" / "alias").symlink_to(Path(db) / "tracks" / "sub")
    pm.gdb_reload()
    assert {"alias.t", "sub.t"} <= set(pm.gtrack_ls())
    with pytest.raises(ValueError, match="Source and destination are the same track"):
        pm.gtrack_copy("sub.t", "alias.t", overwrite=True)
    assert pm.gextract("sub.t", pm.gintervals("chr1", 0, 10))["sub.t"].tolist() == [1]


def _mk_db(td: Path, name: str, chroms: list[str], indexed: bool = False) -> str:
    path = td / name
    (path / "tracks").mkdir(parents=True)
    (path / "seq").mkdir()
    if indexed:
        write_index(path, chroms, [1000] * len(chroms))
    else:
        for ch in chroms:
            (path / "seq" / f"{ch}.seq").write_bytes(b"A" * 1000)
    (path / "chrom_sizes.txt").write_text("".join(f"{ch}\t1000\n" for ch in chroms))
    return str(path.resolve())


def test_overwriting_copy_keeps_the_existing_track_when_it_stops(tmp_path):
    """R: gtrack.copy with overwrite = TRUE keeps the existing track when the copy stops."""
    src = _mk_db(tmp_path, "src", ["chr1", "chr2"])
    pm.gdb_init(src)
    pm.gtrack_create_sparse("s1", "x", pm.gintervals(["chr1", "chr2"], 0, 10), [1, 2])
    pair = pm.gintervals_2d("chr1", 0, 10, "chr2", 0, 10)
    pm.gtrack_2d_create("r2", "x", pair, [5])
    pm.gtrack_2d_create("r2i", "x", pair, [6])
    pm.gtrack_2d_convert_to_indexed("r2i")

    cases = [
        ("r2", _mk_db(tmp_path, "renamed", ["chrA", "chrB"]), "requires the same chromosome names"),
        ("r2i", _mk_db(tmp_path, "reversed", ["chr2", "chr1"], indexed=True), "requires identical chromosome order"),
        ("s1", _mk_db(tmp_path, "disjoint", ["chrZ"]), "no chromosomes from source database are present"),
        ("r2i", _mk_db(tmp_path, "per_chrom", ["chr1", "chr2"]), "indexed 2D track .* into a per-chromosome database"),
        ("r2", _mk_db(tmp_path, "indexed", ["chr1", "chr2"], indexed=True),
         "format conversion to a non-active dataset"),
    ]
    for track, db, error in cases:
        old = Path(db) / "tracks" / "t.track"
        old.mkdir()
        (old / "sentinel").write_text("keep me")
        with pytest.raises((ValueError, NotImplementedError), match=error):
            pm.gtrack_copy(track, "t", db=db, overwrite=True)
        assert os.listdir(old) == ["sentinel"], db


def test_unloaded_db_whose_linked_seq_was_converted_since(tmp_path):
    """R: an unloaded database whose linked seq/ was converted since does not give a stale
    order."""
    from pymisha.liftover import _load_source_chrom_names

    parent = unsorted_db(tmp_path / "parent")
    child = tmp_path / "child"
    (child / "tracks" / "t1.track").mkdir(parents=True)
    shutil.copy(Path(parent) / "chrom_sizes.txt", child / "chrom_sizes.txt")
    (child / "seq").symlink_to(Path(parent) / "seq")
    assert _gdb_chrom_names_at(str(child)) == SORTED_PREFIXED

    pm.gdb_convert_to_indexed(groot=parent, validate=False)
    pm.gdb_init(parent)
    with pytest.raises(Exception, match="does not match chrom_sizes.txt"):
        _gdb_chrom_names_at(str(child))
    with pytest.raises(Exception, match="does not match chrom_sizes.txt"):
        _load_source_chrom_names(str(child / "tracks" / "t1.track"))


def test_overwriting_copy_completes_when_removing_the_replaced_track_fails(tmp_path, monkeypatch):
    """R (test-track-rm-fast.R): gtrack.copy overwrite completes when removing the replaced
    destination fails; the replaced track is left hidden, for the sweep of staging
    directories."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals("chr1", 0, 1000)
    pm.gtrack_create_sparse("test_copy_src", "x", iv, [1])
    pm.gtrack_create_sparse("test_copy_dst", "x", iv, [2])
    monkeypatch.setattr(tracks_mod, "_gdb_trash", lambda *a, **k: False)
    pm.gtrack_copy("test_copy_src", "test_copy_dst", overwrite=True)
    assert pm.gextract("test_copy_dst", iv)["test_copy_dst"].tolist() == [1]
    aside = [n for n in os.listdir(Path(db) / "tracks") if n.startswith(".test_copy_dst.track.tmp.")]
    assert len(aside) == 1
