"""R misha 5.16.0 database-state fixes (test-review-2026-11-dbstate.R, and the forced 2D
re-conversion blocks of test-review-2026-11-import.R): interval sets of datasets, attributes
across gintervals_update, names that are both a track and an interval set, gtrack_copy built
in a staging directory, forced re-conversions that would drop data, and tracks other sessions
re-created.

The big interval sets come from tests/fixtures/misha_5_16_db (R 5.16 wrote them with make.R
there): pymisha writes no big sets.
"""

import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pandas as pd
import pytest
from _chromdb_helpers import create_test_db

import pymisha as pm

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "misha_5_16_db"
DF = pd.DataFrame


def review_db(dir, name="db"):
    """A small indexed database of its own (chr1, chr2, chrX, chrM), built from a FASTA, made
    the root."""
    rng = random.Random(60427)
    fa = Path(dir) / f"{name}.fa"
    fa.write_text("".join(f">{c}\n{''.join(rng.choice('ACGT') for _ in range(5000))}\n"
                          for c in ["chr1", "chr2", "chrX", "chrM"]))
    db = Path(dir) / name
    pm.gdb_create(str(db), str(fa), db_format="indexed")
    pm.gdb_init(str(db))
    return os.path.realpath(db)


def import_test_db(dir):
    """test-review-2026-11-import.R's database: chr1 10000, chr2 10000, chrX 5000, chrM 500."""
    sizes = {"chr1": 10000, "chr2": 10000, "chrX": 5000, "chrM": 500}
    fa = Path(dir) / "genome.fa"
    fa.write_text("".join(f">{c}\n{'A' * n}\n" for c, n in sizes.items()))
    db = Path(dir) / "db"
    pm.gdb_create(str(db), str(fa), db_format="indexed")
    pm.gdb_init(str(db))
    return os.path.realpath(db)


def add_fixture_set(db, name, as_name=None):
    """An interval set R wrote (FIXTURES/<name>.interv), as <as_name> in db."""
    shutil.copytree(FIXTURES / f"{name}.interv", Path(db) / "tracks" / f"{as_name or name}.interv")


def sparse(track, iv, values):
    pm.gtrack_create_sparse(track, "", iv, values)


def track_bytes(db, track):
    return (Path(db) / "tracks" / f"{track}.track" / "track.dat").read_bytes()


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def in_other_session(db, code):
    """Runs code in another Python process with db loaded: another session on the same
    database."""
    script = "import sys\nimport pandas as pd\nimport pymisha as pm\npm.CONFIG['multitasking'] = False\n" \
        "pm.gdb_init(sys.argv[1])\n" + textwrap.dedent(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    return subprocess.run([sys.executable, "-c", script, str(db)], env=env, capture_output=True, text=True)


# --- interval sets ---------------------------------------------------------------------------


def test_save_stores_alias_names_under_canonical_names(tmp_path):  # R :19 (the small set)
    review_db(tmp_path)
    d = DF(dict(chrom=["1", "X", "MT", "chr2"], start=[0, 10, 20, 30], end=[5, 15, 25, 35]))
    pm.gintervals_save(d, "al_small")
    res = pm.gintervals_load("al_small").sort_values("start")
    assert res["chrom"].astype(str).tolist() == ["chr1", "chrX", "chrM", "chr2"]
    assert res["end"].tolist() == [5, 15, 25, 35]
    with pytest.raises(Exception, match="chrNOPE"):
        pm.gintervals_save(DF(dict(chrom=["chrNOPE"], start=[0], end=[5])), "al_bad")


def test_update_takes_a_character_chrom_column_and_aliases(tmp_path):  # R :57
    review_db(tmp_path)
    pm.gintervals_save(pm.gintervals(["chr1", "chr2"], 0, 100), "up")
    pm.gintervals_update("up", DF(dict(chrom=["chr2"], start=[200], end=[300])), chrom="chr2")
    res = pm.gintervals_load("up")
    assert res.loc[res["chrom"].astype(str) == "chr2", "start"].tolist() == [200]

    # by an alias of the chromosome (R saves this one big; pymisha writes it small)
    pm.gintervals_save(pm.gintervals(["chr1", "chrX", "chr2"], 0, 100), "upb")
    pm.gintervals_update("upb", DF(dict(chrom=["X", "X"], start=[500, 700], end=[600, 800])), chrom="chrX")
    res = pm.gintervals_load("upb")
    assert res.loc[res["chrom"].astype(str) == "chrX", "start"].tolist() == [500, 700]
    assert len(res) == 4


def test_update_refuses_an_interval_set_of_a_loaded_dataset(tmp_path):  # R :76
    ds = review_db(tmp_path, "ds")
    pm.gintervals_save(pm.gintervals("chrX", 300, 400), "dsmall")
    add_fixture_set(ds, "u3", "dbig")
    work = review_db(tmp_path, "work")
    pm.gdataset_load(ds)

    with pytest.raises(ValueError, match="read-only dataset"):
        pm.gintervals_update("dsmall", pm.gintervals("chrX", 1000, 2000), chrom="chrX")
    with pytest.raises(ValueError, match="read-only dataset"):
        pm.gintervals_update("dbig", pm.gintervals("chrX", 1000, 2000), chrom="chrX")
    assert not os.path.exists(os.path.join(work, "tracks", "dsmall.interv"))
    assert not os.path.exists(os.path.join(work, "tracks", "dbig.interv"))
    assert pm.gintervals_load("dsmall")["start"].tolist() == [300]


def test_attributes_survive_update_between_small_and_big(tmp_path):  # R :94
    db = review_db(tmp_path)
    tracks = Path(db) / "tracks"
    pm.gintervals_save(pm.gintervals(["chr1"] * 10, range(0, 1000, 100), range(50, 1000, 100)), "at1")
    pm.gintervals_attr_set("at1", "source", "lab")
    # (R makes it big here; pymisha keeps it small)
    pm.gintervals_update("at1", pm.gintervals(["chrX"] * 30, range(0, 3000, 100), range(50, 3000, 100)),
                         chrom="chrX")
    assert pm.gintervals_attr_get("at1", "source") == "lab"
    pm.gintervals_rm("at1", force=True)

    # big to small: R's at1 after the update, its attributes inside its directory
    add_fixture_set(db, "at1")
    pm.gdb_reload()
    assert pm.gintervals_attr_get("at1", "source") == "lab"
    pm.gintervals_update("at1", None, chrom="chrX")
    assert not (tracks / "at1.interv").is_dir()
    assert pm.gintervals_attr_get("at1", "source") == "lab"
    assert len(pm.gintervals_load("at1")) == 10


def test_dataset_save_keeps_the_attributes_of_a_small_interval_set(tmp_path):  # R :237
    review_db(tmp_path, "src")
    pm.gintervals_save(pm.gintervals("chr1", 0, 100), "ii")
    pm.gintervals_attr_set("ii", "who", "DA")
    pm.gdataset_save(str(tmp_path / "DS"), "x", intervals="ii")
    assert (tmp_path / "DS" / "tracks" / "ii.iattr").exists()

    review_db(tmp_path, "other")
    pm.gdataset_load(str(tmp_path / "DS"))
    assert pm.gintervals_attr_get("ii", "who") == "DA"


def test_db_convert_to_indexed_converts_a_set_over_contigs_with_a_dash_as_1d(tmp_path):  # R :251
    db = create_test_db(tmp_path / "db", [("chr1", 10000), ("HLA-A", 5000), ("scaf-2", 5000)])
    add_fixture_set(db, "dash")
    pm.gdb_init(db)
    assert (Path(db) / "tracks" / "dash.interv" / "HLA-A").exists()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.gdb_convert_to_indexed(convert_intervals=True, force=True, remove_old_files=True)
    assert (Path(db) / "tracks" / "dash.interv" / "intervals.idx").exists()
    pm.gdb_init(db)
    assert len(pm.gintervals_load("dash", chrom="HLA-A")) == 5
    assert len(pm.gintervals_load("dash")) == 15


def test_forced_reconversion_refuses_to_drop_what_is_only_in_the_index(tmp_path):  # R :415
    db = review_db(tmp_path)
    for name in ("u3", "u3b", "u2d"):
        add_fixture_set(db, name)
    pm.gdb_reload()
    d = Path(db) / "tracks" / "u3.interv"
    before = [md5(d / f) for f in ("intervals.dat", "intervals.idx")]

    with pytest.raises(ValueError, match="Nothing was changed"):
        pm.gintervals_convert_to_indexed("u3", force=True)
    assert [md5(d / f) for f in ("intervals.dat", "intervals.idx")] == before
    assert len(pm.gintervals_load("u3")) == 80

    # one chromosome updated since (its own file), the others only in the index
    with pytest.raises(ValueError, match="chr1"):
        pm.gintervals_convert_to_indexed("u3b", force=True)
    assert len(pm.gintervals_load("u3b")) == 62

    # 2D
    with pytest.raises(ValueError, match="Nothing was changed"):
        pm.gintervals_2d_convert_to_indexed("u2d", force=True)
    assert len(pm.gintervals_load("u2d")) == 20


# --- names of tracks and interval sets -------------------------------------------------------


def test_mv_and_copy_refuse_the_name_of_an_interval_set(tmp_path):  # R :114
    db = review_db(tmp_path)
    sparse("t1", pm.gintervals("chr1", 0, 10), [1])
    pm.gintervals_save(pm.gintervals("chr1", 0, 100), "iset")

    with pytest.raises(ValueError, match="iset already exists"):
        pm.gtrack_mv("t1", "iset")
    assert os.path.isdir(os.path.join(db, "tracks", "t1.track"))
    assert not os.path.isdir(os.path.join(db, "tracks", "iset.track"))
    with pytest.raises(ValueError, match="iset already exists"):
        pm.gtrack_copy("t1", "iset")
    assert not os.path.isdir(os.path.join(db, "tracks", "iset.track"))
    with pytest.raises(ValueError, match="iset already exists"):
        sparse("iset", pm.gintervals("chr1", 0, 10), [1])
    assert not os.path.isdir(os.path.join(db, "tracks", "iset.track"))

    # the database still opens
    pm.gdb_init(db)
    assert "t1" in pm.gtrack_ls()


def test_copy_into_another_database_refuses_the_name_of_its_interval_set(tmp_path):
    other = review_db(tmp_path, "other")
    pm.gintervals_save(pm.gintervals("chr1", 0, 100), "iset")
    review_db(tmp_path, "work")
    sparse("t1", pm.gintervals("chr1", 0, 10), [1])
    with pytest.raises(ValueError, match="iset already exists"):
        pm.gtrack_copy("t1", "iset", db=other)
    assert not os.path.isdir(os.path.join(other, "tracks", "iset.track"))


def test_dataset_load_refuses_a_track_that_is_an_interval_set_elsewhere(tmp_path):  # R :131
    ds = review_db(tmp_path, "ds")
    sparse("x", pm.gintervals("chr1", 0, 10), [1])
    review_db(tmp_path, "work")
    pm.gintervals_save(pm.gintervals("chr1", 0, 100), "x")
    sparse("mine", pm.gintervals("chr1", 0, 10), [2])
    datasets = pm.gdataset_ls()

    for force in (False, True):
        with pytest.raises(ValueError, match="'x' would be both a track and an interval set"):
            pm.gdataset_load(ds, force=force)
    assert pm.gtrack_ls() == ["mine"]
    assert pm.gintervals_ls() == ["x"]
    assert pm.gdataset_ls() == datasets
    assert pm.gextract("mine", pm.gintervals("chr1", 0, 10))["mine"].tolist() == [2]


def test_dataset_load_in_a_subdirectory_does_not_check_names_across_databases(tmp_path):
    ds = review_db(tmp_path, "ds")
    sparse("x", pm.gintervals("chr1", 0, 10), [1])
    work = review_db(tmp_path, "work")
    pm.gintervals_save(pm.gintervals("chr1", 0, 100), "x")
    os.makedirs(os.path.join(work, "tracks", "sub"))
    pm.gdir_cd("sub")
    pm.gdataset_load(ds)
    assert ds in pm.gdataset_ls()


def test_genome_implant_leaves_the_session_as_it_was(tmp_path):  # R :148
    donor = review_db(tmp_path, "donor")
    ds = review_db(tmp_path, "ds")
    sparse("dtrack", pm.gintervals("chr1", 0, 10), [5])
    work = review_db(tmp_path, "work")
    pm.gdataset_load(ds)
    datasets = pm.gdataset_ls()
    assert ds in datasets
    fa = str(tmp_path / "ref.fa")
    pm.gdb_export_fasta(fa)

    from pymisha import _shared

    with pytest.raises(Exception):
        pm.ggenome_implant(DF(dict(chrom=["chrNOPE"], start=[0], end=[10])), donor, str(tmp_path / "o1.fa"),
                           genome_fasta=fa, create_trackdb=False)
    assert work == _shared._GROOT
    assert pm.gdataset_ls() == datasets

    pm.ggenome_implant(DF(dict(chrom=["chr1"], start=[0], end=[10])), donor, str(tmp_path / "o2.fa"),
                       genome_fasta=fa, create_trackdb=False)
    assert work == _shared._GROOT
    assert pm.gdataset_ls() == datasets
    assert pm.gextract("dtrack", pm.gintervals("chr1", 0, 10))["dtrack"].tolist() == [5]

    pm.gdb_export_fasta(str(tmp_path / "donor_export.fa"), groot=donor)
    assert work == _shared._GROOT
    assert pm.gdataset_ls() == datasets


def test_creation_over_the_name_of_a_dataset_track(tmp_path):  # R :178
    ds = review_db(tmp_path, "ds")
    sparse("dtrack", pm.gintervals("chr1", 0, 10), [5])
    work = review_db(tmp_path, "work")
    pm.gdataset_load(ds)

    # a failed one leaves the dataset's track readable
    with pytest.raises(Exception):
        pm.gtrack_create("dtrack", "", "no_such_track + 1")
    assert pm.gtrack_dataset("dtrack") == ds
    assert pm.gextract("dtrack", pm.gintervals("chr1", 0, 10))["dtrack"].tolist() == [5]
    assert not os.path.isdir(os.path.join(work, "tracks", "dtrack.track"))

    # a successful one writes into the working database, which then wins
    sparse("dtrack", pm.gintervals("chr1", 0, 10), [7])
    assert pm.gtrack_dataset("dtrack") == work
    assert pm.gextract("dtrack", pm.gintervals("chr1", 0, 10))["dtrack"].tolist() == [7]
    assert pm.gtrack_ls().count("dtrack") == 1
    # and the working database's track of that name is taken
    with pytest.raises(ValueError, match="already exists"):
        sparse("dtrack", pm.gintervals("chr1", 0, 10), [8])


def test_removing_a_working_interval_set_shows_the_last_loaded_datasets_set(tmp_path):  # R :198
    review_db(tmp_path, "d1")
    pm.gintervals_save(pm.gintervals("chr1", 100, 200), "shared")
    d1 = os.path.realpath(tmp_path / "d1")
    review_db(tmp_path, "d2")
    pm.gintervals_save(pm.gintervals("chr1", 300, 400), "shared")
    d2 = os.path.realpath(tmp_path / "d2")
    review_db(tmp_path, "work")
    pm.gintervals_save(pm.gintervals("chr1", 500, 600), "shared")
    pm.gdataset_load(d1, force=True)
    pm.gdataset_load(d2, force=True)
    assert pm.gintervals_load("shared")["start"].tolist() == [500]

    pm.gintervals_rm("shared", force=True)
    # the dataset loaded last wins, as gdb_reload() has it
    assert pm.gintervals_exists("shared")
    assert pm.gintervals_load("shared")["start"].tolist() == [300]
    pm.gdb_reload()
    assert pm.gintervals_load("shared")["start"].tolist() == [300]


def test_removing_a_working_track_shows_the_last_loaded_datasets_track(tmp_path):  # R :219
    d1 = review_db(tmp_path, "d1")
    sparse("tt", pm.gintervals("chr1", 0, 10), [1])
    d2 = review_db(tmp_path, "d2")
    sparse("tt", pm.gintervals("chr1", 0, 10), [2])
    review_db(tmp_path, "work")
    sparse("tt", pm.gintervals("chr1", 0, 10), [3])
    pm.gdataset_load(d1, force=True)
    pm.gdataset_load(d2, force=True)

    pm.gtrack_rm("tt", force=True)
    assert pm.gextract("tt", pm.gintervals("chr1", 0, 10))["tt"].tolist() == [2]
    pm.gdb_reload()
    assert pm.gextract("tt", pm.gintervals("chr1", 0, 10))["tt"].tolist() == [2]


# --- gtrack_copy(overwrite=True) -------------------------------------------------------------

# gtrack_copy(overwrite=True) SIGKILLed in another process, as an OOM kill or a job limit: no
# handler runs. Whatever point it dies at, the real name holds the old or the new track
# complete, or the old one is in a directory no later sweep removes.
_KILLED_COPY = """
import os
import shutil
import signal

kill_at = sys.argv[2]
real_rename = os.rename


def rename(src, dst, *args, **kwargs):
    # the rename out of the real name (the track set aside), or into it (the copy put in place)
    hit = (kill_at == "aside" and os.path.basename(str(src)) == "orig.track") or (
        kill_at == "swap" and os.path.basename(str(dst)) == "orig.track")
    res = real_rename(src, dst, *args, **kwargs)
    if hit:
        os.kill(os.getpid(), signal.SIGKILL)
    return res


os.rename = rename
if kill_at == "copy":
    real_copy2 = shutil.copy2

    def copy2(*args, **kwargs):
        res = real_copy2(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGKILL)
        return res

    shutil.copy2 = copy2
pm.gtrack_copy("src", "orig", overwrite=True)
"""


def copy_killed_at(tmp_path, kill_at):
    db = review_db(tmp_path)
    sparse("orig", pm.gintervals("chr1", 0, 10), [42])
    sparse("src", pm.gintervals("chr1", 100, 110), [7])
    tracks = Path(db) / "tracks"
    orig_bytes = track_bytes(db, "orig")

    script = ("import sys\nimport pymisha as pm\npm.CONFIG['multitasking'] = False\npm.gdb_init(sys.argv[1])\n"
              + _KILLED_COPY)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    res = subprocess.run([sys.executable, "-c", script, db, kill_at], env=env, capture_output=True, text=True)
    assert res.returncode == -9, res.stderr

    # a later write in the directory sweeps what the dead process left
    pm.gdb_init(db)
    sparse("another", pm.gintervals("chr1", 0, 10), [1])
    leftovers = sorted(set(os.listdir(tracks)) - {"orig.track", "src.track", "another.track"})
    return db, tracks, leftovers, orig_bytes


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_copy_killed_during_the_copy_leaves_the_old_track_in_place(tmp_path):  # R :312
    _, _, leftovers, _ = copy_killed_at(tmp_path, "copy")
    assert pm.gextract("orig", pm.gintervals("chr1", 0, 200))["orig"].tolist() == [42]
    assert leftovers == []


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_copy_killed_between_its_renames_keeps_the_old_track_where_no_sweep_removes_it(tmp_path):  # R :321
    _, tracks, leftovers, orig_bytes = copy_killed_at(tmp_path, "aside")
    assert not (tracks / "orig.track").exists()
    assert len(leftovers) == 1
    kept = tracks / leftovers[0]
    assert (kept / "track.dat").read_bytes() == orig_bytes
    # put back by hand, it is the track as it was
    os.rename(kept, tracks / "orig.track")
    pm.gdb_reload()
    assert pm.gextract("orig", pm.gintervals("chr1", 0, 200))["orig"].tolist() == [42]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_copy_killed_once_the_copy_is_in_place_leaves_the_new_track(tmp_path):  # R :336
    copy_killed_at(tmp_path, "swap")
    pm.gdb_reload()
    res = pm.gextract("orig", pm.gintervals("chr1", 0, 200))
    assert res["start"].tolist() == [100]
    assert res["orig"].tolist() == [7]


def test_failed_copy_keeps_what_another_session_put_at_the_name(tmp_path, monkeypatch):  # R :669
    db = review_db(tmp_path)
    tracks = Path(db) / "tracks"
    sparse("orig", pm.gintervals("chr1", 0, 10), [42])
    sparse("src", pm.gintervals("chr1", 100, 110), [7])
    sparse("other", pm.gintervals("chr1", 200, 210), [99])
    other_bytes = track_bytes(db, "other")
    orig_bytes = track_bytes(db, "orig")

    # another session's copy lands at the name between this one's two renames
    real_rename = os.rename

    def rename(src, dst, *args, **kwargs):
        if (os.path.basename(str(dst)) == "orig.track" and ".tmp." in os.path.basename(str(src))
                and not os.path.isdir(dst)):
            shutil.copytree(tracks / "other.track", dst)
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", rename)
    with pytest.warns(UserWarning, match="is kept at"), pytest.raises(RuntimeError, match="Failed to rename"):
        pm.gtrack_copy("src", "orig", overwrite=True)
    monkeypatch.undo()
    assert track_bytes(db, "orig") == other_bytes
    # the session still lists the name, which a track holds
    assert pm.gtrack_exists("orig")
    # the track that was there is kept, under its hidden name: what took the name may be
    # incomplete (an older misha writes in place) or go again (a copy that fails after it was
    # placed)
    kept = [p for p in tracks.iterdir() if p.name[0] == "." and all(c in "0123456789abcdef" for c in p.name[1:])]
    assert len(kept) == 1
    assert (kept[0] / "track.dat").read_bytes() == orig_bytes


def test_copy_stops_on_a_short_copy_instead_of_committing_it(tmp_path, monkeypatch):  # R :698
    db = review_db(tmp_path)
    sparse("orig", pm.gintervals("chr1", 0, 10), [42])
    sparse("src", pm.gintervals(["chr1", "chr2"], 100, 110), [7, 8])
    before = track_bytes(db, "orig")

    # the copy succeeds, but the data reached the disk short (as a close error it does not report)
    real_copy2 = shutil.copy2

    def copy2(src, dst, *args, **kwargs):
        res = real_copy2(src, dst, *args, **kwargs)
        if os.path.basename(str(src)) == "track.dat":
            dest = Path(dst) / "track.dat" if os.path.isdir(dst) else Path(dst)
            dest.write_bytes(dest.read_bytes()[:8])
        return res

    monkeypatch.setattr(shutil, "copy2", copy2)
    with pytest.raises(ValueError, match="incomplete"):
        pm.gtrack_copy("src", "orig", overwrite=True)
    monkeypatch.undo()
    assert track_bytes(db, "orig") == before
    assert pm.gextract("orig", pm.gintervals("chr1", 0, 200))["orig"].tolist() == [42]


def test_copy_over_a_track_replaces_it(tmp_path):
    db = review_db(tmp_path)
    tracks = Path(db) / "tracks"
    sparse("orig", pm.gintervals("chr1", 0, 10), [42])
    sparse("src", pm.gintervals("chr1", 100, 110), [7])
    pm.gtrack_copy("src", "orig", overwrite=True)
    assert pm.gextract("orig", pm.gintervals("chr1", 0, 200))["orig"].tolist() == [7]
    assert track_bytes(db, "orig") == track_bytes(db, "src")
    # nothing is left beside it but the trash its background removal is still working on
    assert {p.name for p in tracks.iterdir() if not p.name.startswith(".trash.")} == {"orig.track", "src.track"}


# --- what other sessions re-created ----------------------------------------------------------


def test_tracks_and_sets_another_session_recreated_with_another_layout_are_read_right(tmp_path):  # R :549
    db = review_db(tmp_path)
    sparse("st", pm.gintervals(["chr1", "chr2"], 0, 10), [1, 2])
    pm.gtrack_create("dt", "", "st", iterator=10)
    pm.gtrack_2d_create("rt", "", DF(dict(chrom1=["chr1"], start1=[0], end1=[10], chrom2=["chr2"], start2=[0],
                                           end2=[10])), [1])
    add_fixture_set(db, "bs1", "bs")
    pm.gdb_reload()
    for t in ("st", "dt", "rt"):
        if not (Path(db) / "tracks" / f"{t}.track" / "track.idx").exists():
            pm.gtrack_convert_to_indexed(t, remove_old=True)
    assert (Path(db) / "tracks" / "st.track" / "track.idx").exists()
    assert (Path(db) / "tracks" / "bs.interv" / "intervals.idx").exists()
    # this session caches their indexes
    pm.gextract("st", pm.gintervals_all())
    pm.gextract("dt", pm.gintervals_all())
    pm.gextract("rt", pm.gintervals_2d("chr1", 0, 100, "chr2", 0, 100))
    pm.gextract("st", intervals="bs")

    res = in_other_session(db, f"""
        import shutil
        pm.gtrack_rm("st", force=True)
        pm.gtrack_create_sparse("st", "", pm.gintervals(["chr1", "chr1", "chr1", "chr2"], [0, 20, 40, 0],
                                                        [10, 30, 50, 10]), [5, 6, 7, 8])
        pm.gtrack_rm("dt", force=True)
        pm.gtrack_create("dt", "", "st", iterator=7)
        pm.gtrack_rm("rt", force=True)
        pm.gtrack_2d_create("rt", "", pd.DataFrame(dict(chrom1=["chr1"] * 3, start1=[0, 20, 40], end1=[10, 30, 50],
                                                        chrom2=["chr1", "chr2", "chr2"], start2=[0, 0, 20],
                                                        end2=[10, 10, 30])), [3, 4, 5])
        for t in ("st", "dt", "rt"):
            if not os.path.exists(os.path.join(pm.gtrack_path(t), "track.idx")):
                pm.gtrack_convert_to_indexed(t, remove_old=True)
        pm.gintervals_rm("bs", force=True)
        shutil.copytree({str(FIXTURES / "bs2.interv")!r}, os.path.join(sys.argv[1], "tracks", "bs.interv"))
    """.replace("import shutil", "import os\n        import shutil", 1))
    assert res.returncode == 0, res.stderr

    assert pm.gextract("st", pm.gintervals_all())["st"].tolist() == [5, 6, 7, 8]
    assert pm.gextract("dt", pm.gintervals("chr1", 0, 14))["dt"].tolist() == [5, 5]
    assert pm.gextract("rt", pm.gintervals_2d("chr1", 0, 100, "chr2", 0, 100))["rt"].tolist() == [4, 5]
    assert pm.gextract("st", intervals="bs")["st"].tolist() == [5, 7, 8]


def test_an_array_track_another_session_recreated_with_another_layout_is_read_right(tmp_path):  # R :719
    db = review_db(tmp_path)

    def write_arr(f, iv):
        iv = iv.copy()
        iv["a"] = range(1, len(iv) + 1)
        iv["b"] = [10 * k for k in range(1, len(iv) + 1)]
        iv.to_csv(f, sep="\t", index=False)

    f1 = tmp_path / "a1.tsv"
    write_arr(f1, pm.gintervals(["chr1", "chr2"], 0, 10))
    pm.gtrack_array_import("arr", "", str(f1))
    pm.gtrack_convert_to_indexed("arr")
    assert (Path(db) / "tracks" / "arr.track" / "track.idx").exists()
    pm.gtrack_array_extract("arr", ["a", "b"], pm.gintervals_all())

    f2 = tmp_path / "a2.tsv"
    write_arr(f2, pm.gintervals(["chr1", "chr1", "chr1", "chr2"], [0, 20, 40, 0], [10, 30, 50, 10]))
    res = in_other_session(db, f"""
        pm.gtrack_rm("arr", force=True)
        pm.gtrack_array_import("arr", "", {str(f2)!r})
        pm.gtrack_convert_to_indexed("arr")
    """)
    assert res.returncode == 0, res.stderr
    res = pm.gtrack_array_extract("arr", ["a", "b"], pm.gintervals_all())
    assert res["a"].tolist() == [1, 2, 3, 4]
    assert res["b"].tolist() == [10, 20, 30, 40]


def test_cached_percentiles_and_2d_readers_follow_a_track_another_session_replaced(tmp_path):
    db = review_db(tmp_path)
    pm.gtrack_create_dense("dt", "", pm.gintervals(["chr1", "chr2"], 0, 10), [1, 2], 10, float("nan"))
    pm.gvtrack_create("gp", "dt", func="global.percentile")
    iv = pm.gintervals("chr2", 0, 10)
    assert pm.gextract("gp", iv, iterator=10)["gp"].tolist() == [1.0]

    res = in_other_session(db, """
        pm.gtrack_rm("dt", force=True)
        pm.gtrack_create_dense("dt", "", pm.gintervals(["chr1", "chr2"], 0, 10), [5, 3], 10, float("nan"))
    """)
    assert res.returncode == 0, res.stderr
    # chr2's 3 is now the lower of the two
    assert pm.gextract("gp", iv, iterator=10)["gp"].tolist() == [0.5]

    from pymisha._quadtree import _get_indexed_reader

    pm.gtrack_2d_create("rt", "", DF(dict(chrom1=["chr1"], start1=[0], end1=[10], chrom2=["chr2"], start2=[0],
                                           end2=[10])), [1])
    if not (Path(db) / "tracks" / "rt.track" / "track.idx").exists():
        pm.gtrack_2d_convert_to_indexed("rt")
    rt = str(Path(db) / "tracks" / "rt.track")
    reader = _get_indexed_reader(rt)
    assert _get_indexed_reader(rt) is reader
    res = in_other_session(db, """
        pm.gtrack_rm("rt", force=True)
        pm.gtrack_2d_create("rt", "", pd.DataFrame(dict(chrom1=["chr1"], start1=[0], end1=[10], chrom2=["chr2"],
                                                        start2=[0], end2=[10])), [2])
        if not os.path.exists(os.path.join(pm.gtrack_path("rt"), "track.idx")):
            pm.gtrack_2d_convert_to_indexed("rt")
    """.replace("pm.gtrack_rm", "import os\n        pm.gtrack_rm", 1))
    assert res.returncode == 0, res.stderr
    assert _get_indexed_reader(rt) is not reader


def test_gdb_reload_drops_the_track_indexes_read_so_far(tmp_path, monkeypatch):
    from pymisha import _pymisha

    review_db(tmp_path)
    called = []
    real = _pymisha.pm_clear_dir_caches
    monkeypatch.setattr(_pymisha, "pm_clear_dir_caches", lambda: called.append(1) or real())
    pm.gdb_reload()
    assert called


# --- forced 2D re-conversion of a track (test-review-2026-11-import.R) -----------------------


def test_forced_2d_reconversion_refuses_when_only_some_pairs_have_a_file(tmp_path):  # import R :411
    import_test_db(tmp_path)
    pm.gtrack_2d_create("partial", "", pm.gintervals_2d(["chr1", "chr2"], 0, 100, ["chr1", "chrX"], 0, 100), [1, 2])
    pm.gtrack_2d_convert_to_indexed("partial", remove_old=False, force=True)
    d = Path(pm.gtrack_path("partial"))
    pair_files = sorted(set(os.listdir(d)) - {"track.dat", "track.idx", ".attributes"})
    assert len(pair_files) == 2
    (d / pair_files[0]).unlink()
    dat_before = (d / "track.dat").read_bytes()

    with pytest.raises(Exception, match="per-pair files for only some"):
        pm.gtrack_2d_convert_to_indexed("partial", remove_old=True, force=True)
    assert (d / "track.dat").read_bytes() == dat_before
    assert (d / pair_files[1]).exists()
    assert pm.gextract("partial", pm.gintervals_2d_all())["partial"].tolist() == [1, 2]


def test_forced_2d_reconversion_finds_per_pair_files_named_by_aliases(tmp_path):  # import R :428
    import_test_db(tmp_path)
    pm.gtrack_2d_create("aliased", "", pm.gintervals_2d(["chr1", "chr2"], 0, 100, ["chr1", "chrX"], 0, 100), [1, 2])
    pm.gtrack_2d_convert_to_indexed("aliased", remove_old=False, force=True)
    d = Path(pm.gtrack_path("aliased"))
    os.rename(d / "chr2-chrX", d / "2-X")
    pm.gtrack_2d_convert_to_indexed("aliased", remove_old=True, force=True)
    assert pm.gextract("aliased", pm.gintervals_2d_all())["aliased"].tolist() == [1, 2]

    pm.gtrack_2d_convert_to_indexed("aliased", remove_old=False, force=True)
    (d / "chr1-chr1").unlink()
    with pytest.raises(Exception, match="Remove the per-pair files and run it again"):
        pm.gtrack_2d_convert_to_indexed("aliased", remove_old=True, force=True)


# --- the whole-genome 2D scope above CONFIG["multicontig_2d_threshold"] contigs --------------

MODE_MSG = re.escape("CONFIG['multicontig_2d_mode'] = \"full\"")
R2D = DF(dict(chrom1=["chr1", "chr1", "chr2", "chrX"], start1=[0, 0, 0, 0], end1=[100, 100, 100, 100],
              chrom2=["chr1", "chr2", "chrX", "chrM"], start2=[200, 0, 0, 0], end2=[300, 100, 100, 100]))


@pytest.fixture()
def multicontig_db(tmp_path, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_threshold", 2)
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", None)
    review_db(tmp_path)
    pm.gtrack_2d_create("mr", "", R2D, [10, 20, 30, 40])
    sparse("t1d", pm.gintervals(["chr1", "chr2", "chrX"], 0, 100), [1, 2, 3])
    return tmp_path


def test_2d_computations_over_the_whole_genome_scope_name_multicontig_2d_mode(multicontig_db, monkeypatch):  # R :445
    for mt in (True, False):
        monkeypatch.setitem(pm.CONFIG, "multitasking", mt)
        # gquantiles gave the median of the cis value only (10)
        for call in (lambda: pm.gquantiles("mr", 0.5), lambda: pm.gextract("mr"), lambda: pm.gsummary("mr"),
                     lambda: pm.gscreen("mr > 0"), lambda: pm.gdist("mr", [0, 100])):
            with pytest.raises(ValueError, match=MODE_MSG):
                call()

        # 1D computations over the whole-genome scope are not affected
        assert pm.gsummary("t1d")["Sum"] == 6
        assert len(pm.gscreen("t1d > 0")) == 3
        assert pm.gdist("t1d", [0, 2, 4]).sum() == 3
        assert float(pm.gquantiles("t1d", 0.5).iloc[0]) == 2
        assert len(pm.gextract("t1d")) == 3

    # gintervals_2d_all() itself gives the cis pairs, with a warning; what is made from them works
    with pytest.warns(UserWarning, match=MODE_MSG):
        all2d = pm.gintervals_2d_all()
    assert len(all2d) == 4
    with pytest.warns(UserWarning, match=MODE_MSG):
        assert pm.gextract("mr", pm.gintervals_2d_all())["mr"].tolist() == [10]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        some = pm.gintervals_2d_all().head(2)
    assert len(pm.gintervals_intersect(some, some)) == 2
    pm.gintervals_save(some, "some2d")
    assert len(pm.gintervals_load("some2d")) == 2
    pm.gtrack_2d_create("some_track", "", some, [1, 2])
    assert pm.gextract("some_track", some)["some_track"].tolist() == [1, 2]

    # track creation over the whole genome too
    with pytest.raises(ValueError, match=MODE_MSG):
        pm.gtrack_create("c2d", "", "mr")
    # and the default scope of gcis_decay
    with pytest.raises(ValueError, match=MODE_MSG):
        pm.gcis_decay("mr", [0, 1000], pm.gintervals_all(), pm.gintervals_all())

    # in a mode that was set, the whole-genome scope is built in it, and only for a 2D computation
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", "full")
    assert pm.gsummary("t1d")["Sum"] == 6
    assert len(pm.gextract("t1d")) == 3
    assert float(pm.gquantiles("t1d", 0.5).iloc[0]) == 2
    assert len(pm.gscreen("t1d > 0")) == 3
    pm.gtrack_create("c1d", "", "t1d", iterator=50)
    pm.gtrack_create("c2d", "", "mr")
    assert sorted(pm.gextract("c2d", R2D)["c2d"].tolist()) == [10, 20, 30, 40]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(pm.gextract("mr", pm.gintervals_2d_all())) == 4
        assert float(pm.gquantiles("mr", 0.5).iloc[0]) == 25
        assert pm.gsummary("mr")["Sum"] == 100
        assert len(pm.gscreen("mr > 0")) == 4
        assert pm.gdist("mr", [0, 100]).sum() == 4
        assert len(pm.gextract("mr")) == 4
        pm.gcis_decay("mr", [0, 1000], pm.gintervals_all(), pm.gintervals_all())

    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", "diagonal")
    assert pm.gextract("mr", pm.gintervals_2d_all())["mr"].tolist() == [10]
    assert pm.gsummary("mr")["Sum"] == 10

    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", "all")
    with pytest.raises(ValueError, match="Unknown 2D generation mode"):
        pm.gsummary("mr")


def test_gextract_of_a_2d_track_without_intervals_runs_over_the_2d_genome(tmp_path):
    review_db(tmp_path)
    pm.gtrack_2d_create("mr", "", R2D, [10, 20, 30, 40])
    res = pm.gextract("mr")
    assert sorted(res["mr"].tolist()) == [10, 20, 30, 40]
    assert pm.gsummary("mr")["Sum"] == 100


@pytest.mark.parametrize("mode", [None, "full", "diagonal"])
def test_1d_track_creation_works_above_the_threshold_in_every_2d_mode(tmp_path, monkeypatch, mode):  # R :526
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_threshold", 2)
    monkeypatch.setitem(pm.CONFIG, "multicontig_2d_mode", mode)
    review_db(tmp_path)
    bg = tmp_path / "a.bedgraph"
    bg.write_text("chr1\t0\t100\t1\nchr2\t0\t100\t2\n")
    tag = mode or "unset"
    sp = f"sp_{tag}"
    sparse(sp, pm.gintervals(["chr1", "chr2"], 0, 100), [1, 2])
    pm.gtrack_create(f"cr_{tag}", "", sp, iterator=50)
    assert pm.gextract(f"cr_{tag}", pm.gintervals("chr2", 0, 50))[f"cr_{tag}"].tolist() == [2]
    pm.gtrack_lookup(f"lk_{tag}", "", [10, 20], sp, [0, 1.5, 3])
    assert pm.gextract(f"lk_{tag}", pm.gintervals("chr2", 0, 50), iterator=50)[f"lk_{tag}"].tolist() == [20]
    pm.gtrack_smooth(f"sm_{tag}", "", f"cr_{tag}", 100, alg="MEAN")
    assert pm.gtrack_exists(f"sm_{tag}")
    pm.gtrack_import(f"im_{tag}", "", str(bg), 0)
    assert pm.gextract(f"im_{tag}", pm.gintervals_all())[f"im_{tag}"].tolist() == [1, 2]
