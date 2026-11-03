"""After gdir_cd, track and interval set names are relative to the working directory, as in R
misha: they are listed, read, written, copied, converted and removed there. Ported from R
misha's tests/testthat (test-db.R, test-dataset.R, test-track-copy-crossdb.R and
test-audit-2026-06-28-adversarial3.R; the test-dataset.R and test-track-copy-crossdb.R ones
are PR #197's)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _chromdb_helpers import create_test_db, indexed_like, unsorted_db

import pymisha as pm


def _values(track, iv):
    return pm.gextract(track, iv)[track].tolist()


def _files(*track_dirs):
    """The files of track directories (not the hidden ones), by path, with their bytes."""
    return {p: p.read_bytes() for d in track_dirs for p in sorted(Path(d).iterdir()) if not p.name.startswith(".")}


def test_gdir_cd_lists_the_tracks_of_the_subdirectory(tmp_path):
    """R test-db.R: gdir.cd works."""
    pm.gdb_init(create_test_db(tmp_path / "db"))
    pm.gdir_create("test", show_warnings=False)
    pm.gtrack_create_sparse("test.sparse", "Test sparse track",
                            pm.gintervals(["chr1", "chr1", "chr2"], [0, 6000, 0], [5000, 9000, 5000]), [1, 2, 3])
    pm.gtrack_create_dense("test.fixedbin", "Test fixedbin track",
                           pm.gintervals(["chr1", "chr2"], 0, 10000), [0.5, 0.7], 1000, 0)
    pm.gtrack_2d_create("test.rects", "Test rects track", pm.gintervals_2d("chr1", 0, 5000, "chr2", 0, 5000), [10])
    pm.gdir_cd("test")
    r = pm.gtrack_ls()
    pm.gdir_cd("..")
    assert {"sparse", "fixedbin", "rects"} <= set(r)


def test_gtrack_rm_from_a_subdirectory_marks_the_cache_dirty(tmp_path):
    """R test-audit-2026-06-28-adversarial3.R: ADV H8-rm."""
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals("chr1", 0, 10000)
    pm.gdir_create("rmsub", show_warnings=False)
    pm.gtrack_create_sparse("rmsub.tA", "a", iv, [1])
    pm.gtrack_create_sparse("rmsub.tB", "b", iv, [1])
    dirty = Path(db) / ".db.cache.dirty"
    dirty.unlink(missing_ok=True)
    pm.gdir_cd("rmsub")
    pm.gtrack_rm("tB", force=True)
    assert dirty.exists()
    assert pm.gtrack_ls() == ["tA"]


def test_with_a_dataset_loaded_names_in_a_subdirectory_are_found(tmp_path):
    """R test-dataset.R (PR #197): with a dataset loaded, tracks and interval sets in a
    subdirectory of the working directory are found."""
    db = unsorted_db(tmp_path / "db")
    other = unsorted_db(tmp_path / "other")
    pm.gdb_init(other)
    iv = pm.gintervals(["chr1", "chr2"], 0, 10)
    # a dataset whose track has the short name of a track in the subdirectory
    pm.gtrack_create_sparse("tsub", "x", iv, [1000, 2000])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="tsub")

    pm.gdb_init(db)
    pm.gtrack_create_sparse("u", "x", iv, [5, 5])
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("sub.u", "x", iv, [7, 7])
    pm.gtrack_create_sparse("sub.tsub", "x", iv, [3, 6])
    pm.gtrack_attr_set("sub.tsub", "note", "mine")
    pm.gintervals_save(iv, "sub.isub")
    pm.gintervals_attr_set("sub.isub", "note", "set")
    pm.gdataset_load(str(ds))
    assert _values("tsub", iv) == [1000, 2000]
    # a dataset's track is not converted in this session
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("tsub")
    assert not (ds / "tracks" / "tsub.track" / "track.idx").exists()

    pm.gdir_cd("sub")
    assert pm.gtrack_ls() == ["tsub", "u"]
    assert pm.gtrack_exists("tsub")
    assert pm.gtrack_info("tsub")["type"] == "sparse"
    assert _values("tsub", iv) == [3, 6]
    assert pm.gtrack_attr_get("tsub", "note") == "mine"
    assert pm.gintervals_ls() == ["isub"]
    assert len(pm.gintervals_load("isub")) == 2
    assert pm.gintervals_attr_get("isub", "note") == "set"
    # a track made here lands here
    pm.gtrack_create_sparse("made", "x", iv, [7, 8])
    assert (Path(db) / "tracks" / "sub" / "made.track").is_dir()
    assert _values("made", iv) == [7, 8]
    pm.gtrack_rm("made", force=True)
    assert not (Path(db) / "tracks" / "sub" / "made.track").exists()
    # an overwrite here replaces the track here, not the top-level one of that name
    pm.gtrack_copy("tsub", "u", overwrite=True)
    assert _values("u", iv) == [3, 6]
    # a dataset saved from here gets the tracks here
    ds2 = tmp_path / "ds2"
    pm.gdataset_save(str(ds2), "d2", tracks="tsub")
    sub_files = _files(Path(db) / "tracks" / "sub" / "tsub.track")
    assert {p.name: b for p, b in sub_files.items()} == {
        p.name: b for p, b in _files(ds2 / "tracks" / "tsub.track").items()
    }
    # removing a track here leaves the dataset's of that name
    pm.gtrack_rm("tsub", force=True)
    assert not (Path(db) / "tracks" / "sub" / "tsub.track").exists()
    assert (ds / "tracks" / "tsub.track").is_dir()
    assert not pm.gtrack_exists("tsub")

    pm.gdir_cd("..")
    assert _values("tsub", iv) == [1000, 2000]
    assert _values("u", iv) == [5, 5]
    assert _values("sub.u", iv) == [3, 6]


def test_outside_the_working_database_db_means_its_tracks_and_other_tracks_are_not_converted(tmp_path):
    """R test-dataset.R (PR #197): with the working directory outside the working database,
    db = it means its tracks/ and other tracks are not converted."""
    db = unsorted_db(tmp_path / "db")
    other = unsorted_db(tmp_path / "other")
    pm.gdb_init(other)
    iv = pm.gintervals(["chr1", "chr2"], 0, 10)
    pm.gtrack_create_sparse("x", "x", iv, [3, 3])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="x")
    ds = ds.resolve()
    # a database kept in the working database's directory
    nested = unsorted_db(Path(db) / "datasets" / "n")
    pm.gdb_init(nested)
    pm.gtrack_create_sparse("nt", "x", iv, [4, 4])
    pm.gdb_init(db)
    pm.gtrack_create_sparse("x", "x", iv, [1, 1])
    pm.gdataset_load(str(ds), force=True)

    # at the dataset's directory, names are under it
    pm.gdir_cd(str(ds))
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("tracks.x")
    assert not (ds / "tracks" / "x.track" / "track.idx").exists()
    pm.gtrack_copy("tracks.x", "y", db=db)
    assert (Path(db) / "tracks" / "y.track").is_dir()
    # in a dataset unloaded since, a copy needs its destination
    pm.gdir_cd(str(ds / "tracks"))
    pm.gdataset_unload(str(ds))
    with pytest.raises(ValueError, match="in no loaded database"):
        pm.gtrack_copy("x", "z")
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("x")
    pm.gdir_cd(str(Path(nested) / "tracks"))
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("nt")
    assert not (Path(nested) / "tracks" / "nt.track" / "track.idx").exists()
    # a dataset kept in the working database's tracks/: its subdirectories are its own
    inner = Path(db) / "tracks" / "inner"
    pm.gdir_cd(str(Path(db) / "tracks"))
    pm.gdataset_save(str(inner), "i", tracks="y")
    (inner / "tracks" / "sub").mkdir()
    os.rename(inner / "tracks" / "y.track", inner / "tracks" / "sub" / "y.track")
    pm.gdataset_load(str(inner), force=True)
    pm.gdir_cd(str(inner / "tracks" / "sub"))
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("y")
    assert not (inner / "tracks" / "sub" / "y.track" / "track.idx").exists()
    # nor is its directory the working database's
    pm.gdir_cd(str(inner))
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("tracks.sub.y")
    assert not (inner / "tracks" / "sub" / "y.track" / "track.idx").exists()
    pm.gdir_cd(str(inner / "tracks" / "sub"))
    assert _values("y", iv) == [3, 3]

    pm.gdir_cd(str(Path(db) / "tracks"))
    assert _values("x", iv) == [1, 1]
    assert _values("y", iv) == [3, 3]


def test_gtrack_create_works_in_a_loaded_datasets_tracks_when_both_are_indexed(tmp_path):
    """R test-dataset.R (PR #197): gtrack.create works in a loaded dataset's tracks/ when
    both databases are indexed."""
    a = unsorted_db(tmp_path / "a")
    b = unsorted_db(tmp_path / "b")
    for d in (a, b):
        pm.gdb_convert_to_indexed(groot=d, force=True, validate=False)
    pm.gdb_init(a)
    iv = pm.gintervals(["chr1", "chr2"], 0, 10)
    pm.gtrack_create_sparse("src", "x", iv, [1, 2])
    pm.gdataset_load(b)
    pm.gdir_cd(str(Path(b) / "tracks"))
    pm.gtrack_create("made", "x", "src", iterator=10)
    assert (Path(b) / "tracks" / "made.track" / "track.idx").exists()
    pm.gdb_init(b)
    assert _values("made", iv) == [1, 2]


def test_indexed_tracks_are_not_created_in_a_dataset_whose_order_cannot_be_read(tmp_path):
    """R test-dataset.R (PR #197): in a subdirectory of a dataset whose chromosome order
    cannot be read, indexed tracks are not created."""
    p0 = unsorted_db(tmp_path / "p0")
    # indexed, the same chrom_sizes.txt as p0, chrom ids in chrom_sizes.txt order
    i = indexed_like(p0, tmp_path / "i")
    pm.gdb_init(p0)
    iv = pm.gintervals(["1", "2"], 0, 10)
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("sub.sp", "x", iv, [1, 2])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="sub.sp")
    # the dataset's seq/ links to p0's, which is gone: its order (sorted names) cannot be read
    os.rename(Path(p0) / "seq", Path(p0) / "seq.away")
    pm.gdb_init(i)
    pm.gdataset_load(str(ds))
    pm.gdir_cd(str(ds / "tracks" / "sub"))
    with pytest.raises(ValueError, match="may number the chromosomes differently"):
        pm.gtrack_create("x", "x", "sp", iterator=10)
    assert not (ds / "tracks" / "sub" / "x.track").exists()


def test_gtrack_copy_writes_in_the_subdirectory_and_its_tracks_stay_readable(tmp_path):
    """R test-track-copy-crossdb.R (PR #197): after gdir.cd(), gtrack.copy writes in the
    subdirectory and its tracks stay readable."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals(["chr1", "chr2"], 0, 10)
    pm.gtrack_create_sparse("u", "x", iv, [5, 5])
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("sub.u", "x", iv, [7, 7])
    pm.gtrack_create_sparse("sub.t", "x", iv, [3, 6])
    pm.gdir_cd("sub")
    pm.gtrack_copy("t", "cp")
    assert (Path(db) / "tracks" / "sub" / "cp.track").is_dir()
    assert _values("cp", iv) == [3, 6]
    assert _values("u", iv) == [7, 7]
    pm.gtrack_copy("t", "u", overwrite=True)
    assert _values("u", iv) == [3, 6]
    # a name that leads through a link into another database is not written there
    other = unsorted_db(tmp_path / "other")
    (Path(db) / "tracks" / "sub" / "lnk").symlink_to(Path(other) / "tracks")
    with pytest.raises(ValueError, match="inside another database"):
        pm.gtrack_copy("t", "lnk.new")
    assert not (Path(other) / "tracks" / "new.track").exists()
    pm.gdir_cd("..")
    assert _values("u", iv) == [5, 5]
    assert _values("sub.u", iv) == [3, 6]


def test_in_a_subdirectory_of_an_indexed_database_tracks_are_converted_there(tmp_path):
    """R test-track-copy-crossdb.R (PR #197): in a subdirectory of an indexed database,
    tracks are converted there and the top-level ones of the same name kept."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_convert_to_indexed(groot=db, force=True, validate=False)
    pm.gdb_init(db)
    iv = pm.gintervals(["chr1", "chr2"], 0, 10)
    rect = pm.gintervals_2d("chr1", 0, 10, "chr1", 0, 10)
    pm.gtrack_create_sparse("sig", "x", iv, [1, 11])
    pm.gtrack_2d_create("p2", "x", rect, [9])
    top = [Path(db) / "tracks" / f"{t}.track" for t in ("sig", "p2")]
    top_files = _files(*top)
    pm.gdir_create("sub", show_warnings=False)
    pm.gdir_cd("sub")
    pm.gtrack_create_sparse("sig", "x", iv, [2, 2])
    pm.gtrack_2d_create("p2", "x", rect, [5])
    pm.gtrack_2d_create("q2", "x", rect, [7])
    pm.gtrack_copy("q2", "p2", overwrite=True)
    for t in ("sig", "p2"):
        assert (Path(db) / "tracks" / "sub" / f"{t}.track" / "track.idx").exists(), t
    assert _values("sig", iv) == [2, 2]
    assert _values("p2", rect) == [7]
    assert _files(*top) == top_files
    # nor created, already indexed, through a link into another database
    other = unsorted_db(tmp_path / "other")
    (Path(db) / "tracks" / "sub" / "lnk").symlink_to(Path(other) / "tracks")
    with pytest.raises(ValueError, match="may number the chromosomes differently"):
        pm.gtrack_create("lnk.cr", "x", "sig", iterator=10)
    assert os.listdir(Path(other) / "tracks") == []
    pm.gdir_cd("..")
    assert _values("sig", iv) == [1, 11]
    assert _values("p2", rect) == [9]


def test_virtual_tracks_belong_to_the_directory_they_were_created_in(tmp_path):
    """R keeps the virtual tracks by working directory (GVTRACKS[[GWD]]): gdir.cd hides
    them and the return to their directory brings them back, also when the database was
    loaded through a link."""
    os.symlink(create_test_db(tmp_path / "db"), tmp_path / "link")
    pm.gdb_init(str(tmp_path / "link"))
    iv = pm.gintervals("chr1", 0, 100)
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("t", "x", iv, [1])
    pm.gtrack_create_sparse("sub.t", "x", iv, [2])
    pm.gvtrack_create("v", "t", "avg")
    pm.gdir_cd("sub")
    assert pm.gvtrack_ls() == []
    pm.gvtrack_create("v", "t", "max")
    assert _values("v", iv) == [2]
    pm.gdir_cd("..")
    assert pm.gvtrack_ls() == ["v"]
    assert _values("v", iv) == [1]
    pm.gdir_cd("sub")
    assert _values("v", iv) == [2]


def test_gdir_cd_to_the_database_root_names_the_tracks_by_their_path(tmp_path):
    """R's gdir.cd goes anywhere outside a track: from tracks/, ".." is the database root,
    where a track is named by its path from there."""
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals("chr1", 0, 100)
    pm.gtrack_create_sparse("t", "x", iv, [1])
    pm.gdir_cd("..")
    assert pm.gdir_cwd() == str(Path(db).resolve())
    assert pm.gtrack_ls() == ["tracks.t"]
    assert _values("tracks.t", iv) == [1]
    with pytest.raises(ValueError, match="belongs to track|within track"):
        pm.gdir_cd("tracks/t.track")
    pm.gdir_cd("tracks")
    assert pm.gtrack_ls() == ["t"]


def test_db_of_the_working_database_means_the_names_of_the_subdirectory(tmp_path):
    """R misha 5.13.2 (PR #197): after gdir.cd() into a subdirectory, db = <working
    database> in gtrack.rm() and gtrack.copy() means the names listed there, as db = None
    does, not the top of its tracks/."""
    db = create_test_db(tmp_path / "db")
    pm.gdb_init(db)
    iv = pm.gintervals("chr1", 0, 100)
    pm.gtrack_create_sparse("t", "x", iv, [1])
    pm.gdir_create("sub", show_warnings=False)
    pm.gtrack_create_sparse("sub.t", "x", iv, [2])
    pm.gdir_cd("sub")
    pm.gtrack_copy("t", "c", db=db)
    assert (Path(db) / "tracks" / "sub" / "c.track").is_dir()
    assert not (Path(db) / "tracks" / "c.track").exists()
    pm.gtrack_rm("t", force=True, db=db)
    assert not (Path(db) / "tracks" / "sub" / "t.track").exists()
    pm.gdir_cd("..")
    assert _values("t", iv) == [1]
    assert _values("sub.c", iv) == [2]
