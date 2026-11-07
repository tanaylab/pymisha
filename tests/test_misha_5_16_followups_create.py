"""R misha 5.16.0 follow-ups on track creation: the error of a Parquet import that matches none
of the database's chromosomes, tracks created from no intervals, and the staging directories
the track writers build in, which carry the host and pid so that the leftover of a dead
process is swept at once.

R misha 5.16.0 is the spec: the expected values are what it gives on the same input. Each test
runs on a small database of its own (chr1, chr2, chrX, chrM; tests/fixtures/misha_5_16_import).
"""

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import tracks as _tracks
from pymisha._db_trash import _gdb_staging_name
from pymisha.extract import _host_tag

DF = pd.DataFrame
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "misha_5_16_import"
# gintervals_all() of the fixture databases
DB_CHROMS = {"indexed": "chr1, chr2, chrM, chrX", "per-chromosome": "chr1, chr2, chrX, chrM"}


@pytest.fixture()
def make_db(tmp_path):
    """Make a copy of the R-made database in the given format the root."""
    def make(fmt="indexed"):
        db = tmp_path / f"db_{fmt}"
        shutil.copytree(_FIXTURES / fmt, db)
        (db / "tracks").mkdir()
        pm.gdb_init(str(db))
        return db
    return make


def write_parquet(path, chroms, columns=("v",)):
    pytest.importorskip("pyarrow")
    n = len(chroms)
    df = DF({"chrom": chroms, "start": [0] * n, "end": [100] * n})
    for i, col in enumerate(columns):
        df[col] = np.arange(n, dtype=float) + i
    df.to_parquet(path, index=False)
    return str(path)


def staging_leftovers(directory):
    """Hidden staging directories in directory (.trash.* are being removed already)."""
    return [n for n in os.listdir(directory) if ".tmp." in n and not n.startswith(".trash.")]


def no_chroms_message(srcname, file_chroms, db_chroms):
    return (f"None of the chromosome names in {srcname} exist in the genome database: no data from the file "
            f"would reach the track.\nFile chromosomes: {file_chroms}.\nDatabase chromosomes: {db_chroms}.\n")


# Parquet import that matches no chromosome of the database


@pytest.mark.parametrize("fmt", ["indexed", "per-chromosome"])
def test_parquet_import_with_no_chromosome_of_the_database_gives_the_error_of_the_other_formats(
        make_db, tmp_path, fmt):
    db = make_db(fmt)
    f = write_parquet(tmp_path / "unknown.parquet", ["chr7", "chr8"])
    msg = no_chroms_message(f, "chr7, chr8", DB_CHROMS[fmt])
    with pytest.raises(ValueError, match="^" + re.escape(msg)):
        pm.gtrack_import("t_unk", "", f, 10)
    with pytest.raises(ValueError, match="^" + re.escape(msg)):
        pm.gtrack_import_parquet("", f, 10, track_prefix="pq_")
    assert not pm.gtrack_ls()
    assert staging_leftovers(db / "tracks") == []

    # at most 5 names, and how many more
    f = write_parquet(tmp_path / "many.parquet", [f"chrU{i}" for i in range(8)])
    msg = no_chroms_message(f, "chrU0, chrU1, chrU2, chrU3, chrU4 (and 3 more)", DB_CHROMS[fmt])
    with pytest.raises(ValueError, match="^" + re.escape(msg)):
        pm.gtrack_import("t_many", "", f, 10)


def test_parquet_import_of_files_with_no_chromosome_of_the_database_names_all_of_them(make_db, tmp_path):
    db = make_db()
    src = tmp_path / "src"
    src.mkdir()
    u1 = write_parquet(src / "u1.parquet", ["chr7", "chr8"])
    u2 = write_parquet(src / "u2.parquet", ["chr9", "chr7", "1_random"])
    msg = no_chroms_message(f"{u1}, {u2}", "chr7, chr8, chr9, 1_random", DB_CHROMS["indexed"])
    with pytest.raises(ValueError, match="^" + re.escape(msg)):
        pm.gtrack_import_parquet("", str(src / "u*.parquet"), 10, track_prefix="u_")
    assert not pm.gtrack_ls()
    assert staging_leftovers(db / "tracks") == []

    # a scaffold only: an error as well
    f = write_parquet(tmp_path / "scaf.parquet", ["scaffold_1"])
    with pytest.raises(ValueError, match="^" + re.escape(no_chroms_message(f, "scaffold_1", DB_CHROMS["indexed"]))):
        pm.gtrack_import("sc", "", f, 10)


def test_parquet_import_with_some_chromosomes_of_the_database_warns_of_the_others(make_db, tmp_path):
    make_db()
    f = write_parquet(tmp_path / "some.parquet", ["chr1", "chr7"])
    with pytest.warns(UserWarning, match=re.escape(
            f"1 chromosome name(s) in {f} do not exist in the genome database and were skipped, among them "
            "primary chromosome(s): chr7. Data for the remaining 1 chromosome(s) was imported.")):
        pm.gtrack_import("t_some", "", f, 10)
    res = pm.gextract("t_some", pm.gintervals("chr1", 0, 30))
    assert res["t_some"].tolist() == [0, 0, 0]


def test_parquet_input_with_no_rows_is_an_error_and_an_empty_file_named_parquet_an_empty_track(make_db, tmp_path):
    db = make_db()
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    empty = str(tmp_path / "empty.parquet")
    pq.write_table(pa.table({"chrom": pa.array([], pa.string()), "start": pa.array([], pa.int64()),
                             "end": pa.array([], pa.int64()), "v": pa.array([], pa.float64())}), empty)
    with pytest.raises(Exception, match=re.escape(f"{empty} has no rows")):
        pm.gtrack_import("t_empty", "", empty, 10)
    assert not pm.gtrack_exists("t_empty")
    assert staging_leftovers(db / "tracks") == []

    # not a Parquet file (no "PAR1"): read as the other formats are, an empty track
    zero = tmp_path / "zero.parquet"
    zero.write_bytes(b"")
    pm.gtrack_import("t_zero", "", str(zero), 10)
    info = pm.gtrack_info("t_zero")
    assert (info["type"], info["bin_size"]) == ("dense", 10)
    res = pm.gextract("t_zero", pm.gintervals(["chr1", "chrM"], 0, 30))
    assert res["start"].tolist() == [0, 10, 20, 0, 10, 20]
    assert np.isnan(res["t_zero"]).all()
    pm.gtrack_import("t_zero0", "", str(zero), 0)
    assert pm.gtrack_info("t_zero0")["type"] == "sparse"
    assert pm.gextract("t_zero0", pm.gintervals_all()) is None


# Tracks created from no intervals


@pytest.mark.parametrize("fmt", ["indexed", "per-chromosome"])
def test_create_sparse_with_no_intervals_creates_an_empty_track(make_db, fmt):
    db = make_db(fmt)
    none = pm.gintervals("chr1", 0, 100).iloc[:0]
    pm.gtrack_create_sparse("sp0", "d", none, np.array([], dtype=float))
    info = pm.gtrack_info("sp0")
    assert (info["type"], info["format"]) == ("sparse", fmt)
    assert pm.gtrack_attr_get("sp0", "created.by") == 'gtrack.create_sparse("sp0", description, intervals, values)'
    assert pm.gtrack_attr_get("sp0", "description") == "d"
    track_dir = db / "tracks" / "sp0.track"
    if fmt == "indexed":
        assert (track_dir / "track.dat").stat().st_size == 0
        assert (track_dir / "track.idx").stat().st_size == 132
    else:
        assert {p.name: p.stat().st_size for p in track_dir.iterdir() if p.name != ".attributes"} == \
            {"chr1": 4, "chr2": 4, "chrM": 4, "chrX": 4}

    assert pm.gextract("sp0", pm.gintervals(["chr1", "chrM"], 0, 30)) is None
    assert pm.gextract("sp0", pm.gintervals_all()) is None
    assert pm.gscreen("sp0 > 0") is None
    summary = pm.gsummary("sp0")
    assert summary["Total intervals"] == 0
    assert summary["NaN intervals"] == 0
    assert np.isnan(summary[["Min", "Max", "Sum", "Mean", "Std dev"]].astype(float)).all()

    # a frame of no rows with the values in its value column, and one of other dtypes
    pm.gtrack_create_sparse("sp1", "d", DF({"chrom": [], "start": [], "end": [], "value": []}))
    assert pm.gtrack_info("sp1")["type"] == "sparse"
    assert pm.gextract("sp1", pm.gintervals_all()) is None


def test_create_sparse_with_no_intervals_and_values_is_an_error(make_db):
    make_db()
    none = pm.gintervals("chr1", 0, 100).iloc[:0]
    with pytest.raises(ValueError, match=re.escape("Number of intervals (0) does not match the number of values (1)")):
        pm.gtrack_create_sparse("sp1", "d", none, [5])
    assert not pm.gtrack_exists("sp1")
    with pytest.raises(ValueError, match=re.escape("Number of intervals (1) does not match the number of values (2)")):
        pm.gtrack_create_sparse("sp1", "d", pm.gintervals("chr1", 0, 100), [5, 6])
    assert not pm.gtrack_exists("sp1")


@pytest.mark.parametrize("fmt", ["indexed", "per-chromosome"])
def test_create_dense_with_no_intervals_creates_a_track_of_defval(make_db, fmt):
    db = make_db(fmt)
    none = DF({"chrom": pd.Series([], dtype=str), "start": pd.Series([], dtype=float),
               "end": pd.Series([], dtype=float)})
    pm.gtrack_create_dense("de0", "d", none, [], 10, defval=3)
    info = pm.gtrack_info("de0")
    assert (info["type"], info["bin_size"], info["format"]) == ("dense", 10, fmt)
    assert pm.gtrack_attr_get("de0", "created.by") == \
        'gtrack.create_dense("de0", description, intervals, values, 10, 3)'
    res = pm.gextract("de0", pm.gintervals(["chr1", "chrM"], 0, 30))
    assert res["start"].tolist() == [0, 10, 20, 0, 10, 20]
    assert res["de0"].tolist() == [3] * 6
    track_dir = db / "tracks" / "de0.track"
    if fmt == "indexed":
        assert (track_dir / "track.dat").stat().st_size == 10216
    else:
        assert {p.name: p.stat().st_size for p in track_dir.iterdir() if p.name != ".attributes"} == \
            {"chr1": 4004, "chr2": 4004, "chrM": 204, "chrX": 2004}

    pm.gtrack_create_dense("de1", "d", pm.gintervals("chr1", 0, 100).iloc[:0], [], 10)
    assert np.isnan(pm.gextract("de1", pm.gintervals_all())["de1"]).all()
    pm.gtrack_create_dense("de2", "d", none, [], 10, func="max")
    assert np.isnan(pm.gextract("de2", pm.gintervals_all())["de2"]).all()


def test_create_dense_with_no_intervals_and_values_is_an_error(make_db):
    make_db()
    with pytest.raises(ValueError, match="^Length of values must match the number of intervals$"):
        pm.gtrack_create_dense("de2", "d", pm.gintervals("chr1", 0, 100).iloc[:0], [5], 10)
    assert not pm.gtrack_exists("de2")


@pytest.mark.parametrize("fmt", ["indexed", "per-chromosome"])
def test_create_over_an_iterator_of_no_intervals_creates_an_empty_sparse_track(make_db, fmt):
    db = make_db(fmt)
    pm.gtrack_create_sparse("sp0", "", pm.gintervals("chr1", 0, 100).iloc[:0], [])
    pm.gtrack_create_dense("dn", "", pm.gintervals("chr1", 0, 100), [1], 10)
    pm.gtrack_create("cr0", "d", "sp0", iterator="sp0")
    pm.gtrack_create("cr1", "d", "dn * 2", iterator="sp0")
    pm.gtrack_create("cr2", "d", "dn", iterator=pm.gintervals("chr1", 0, 100).iloc[:0])
    pm.gtrack_create("cr3", "d", "1", iterator="sp0")
    for track in ["cr0", "cr1", "cr2", "cr3"]:
        info = pm.gtrack_info(track)
        assert (info["type"], info["format"]) == ("sparse", fmt), track
        assert pm.gextract(track, pm.gintervals_all()) is None
        files = {p.name: p.stat().st_size for p in (db / "tracks" / f"{track}.track").iterdir()}
        if fmt == "indexed":
            assert (files["track.dat"], files["track.idx"]) == (16, 132)
        else:
            assert files == {".attributes": files[".attributes"], "chr1": 4, "chr2": 4, "chrM": 4, "chrX": 4}

    # the expression is still checked
    with pytest.raises(Exception, match="nosuch"):
        pm.gtrack_create("c1", "", "nosuch + 1", iterator="sp0")
    with pytest.raises(Exception, match="sp0 \\+"):
        pm.gtrack_create("c2", "", "sp0 +", iterator="sp0")
    assert not pm.gtrack_exists("c1") and not pm.gtrack_exists("c2")
    assert staging_leftovers(db / "tracks") == []


# gtrack_attr_export


def test_attr_export_takes_a_single_track_and_attribute_name(make_db):
    make_db()
    pm.gtrack_create_dense("dn", "dense one", pm.gintervals("chr1", 0, 100), [1], 10)
    pm.gtrack_create_sparse("sp", "sparse one", pm.gintervals("chr1", 0, 10), [1])
    pm.gtrack_attr_set("sp", "extra", "x")

    res = pm.gtrack_attr_export("dn")
    assert res.index.tolist() == ["dn"]
    assert [c for c in res.columns if c not in ("created.date", "created.user")] == \
        ["binsize", "created.by", "description", "type"]
    assert pm.gtrack_attr_export(["dn", "dn"]).index.tolist() == ["dn"]

    res = pm.gtrack_attr_export(attrs="description")
    assert res.columns.tolist() == ["description"]
    assert res.loc[["dn", "sp"], "description"].tolist() == ["dense one", "sparse one"]
    res = pm.gtrack_attr_export("dn", "description")
    assert res.index.tolist() == ["dn"] and res["description"].tolist() == ["dense one"]

    # the attributes asked for, in the order asked, once each; "" where a track has none
    res = pm.gtrack_attr_export(["sp", "dn"], ["type", "description", "description"])
    assert res.index.tolist() == ["sp", "dn"]
    assert res.columns.tolist() == ["type", "description"]
    assert res.values.tolist() == [["", "sparse one"], ["dense", "dense one"]]
    assert pm.gtrack_attr_export("dn", ["extra", "description"]).values.tolist() == [["", "dense one"]]
    assert pm.gtrack_attr_export("sp", "nosuchattr").values.tolist() == [[""]]

    with pytest.raises(ValueError, match="^Track nosuch does not exist$"):
        pm.gtrack_attr_export("nosuch")


# A single name where a list of names is taken


@pytest.fixture()
def array_db(make_db, tmp_path):
    db = make_db()
    pm.gtrack_create_dense("dn", "", pm.gintervals("chr1", 0, 100), [1], 10)
    f = tmp_path / "arr.tsv"
    f.write_text("chrom\tstart\tend\tab\tcd\nchr1\t0\t10\t1\t2\nchr1\t20\t30\t3\t4\n")
    pm.gtrack_array_import("arr", "", str(f))
    return db


def test_a_single_column_name_is_taken_as_one_name(array_db):
    iv = pm.gintervals("chr1", 0, 40)
    res = pm.gextract("dn", iv, colnames="xy")
    assert res.columns.tolist() == ["chrom", "start", "end", "xy", "intervalID"]
    assert res["xy"].tolist() == [1, 1, 1, 1]

    res = pm.gtrack_array_extract("arr", "ab", iv)
    assert res.columns.tolist() == ["chrom", "start", "end", "ab", "intervalID"]
    assert res["ab"].tolist() == [1, 3]
    # an index alone (0-based, as in a list)
    assert pm.gtrack_array_extract("arr", 1, iv)["cd"].tolist() == [2, 4]

    pm.gvtrack_create("v", "arr")
    pm.gvtrack_array_slice("v", "ab", "max")
    assert pm.gextract("v", iv, iterator="arr")["v"].tolist() == [1, 3]
    pm.gvtrack_create("v2", "arr")
    pm.gvtrack_array_slice("v2", 1, "max")
    assert pm.gextract("v2", iv, iterator="arr")["v2"].tolist() == [2, 4]

    mat = pm.gintervals_to_mat(pm.gextract("dn", iv), value_cols="dn")
    assert mat.columns.tolist() == ["dn"] and mat["dn"].tolist() == [1, 1, 1, 1]

    res = pm.gdist("dn", [0, 0.5, 2], intervals=iv, dataframe=True, names="xy")
    assert res.columns.tolist() == ["xy", "n"] and res["n"].tolist() == [0, 4]
    res = pm.gcor("dn", "dn", intervals=iv, details=True, names="xy")
    assert res.index.tolist() == ["xy"] and res["n"].tolist() == [4]


def test_a_single_chromosome_or_annotation_name_is_taken_as_one_name(array_db, tmp_path):
    res = pm.gintervals_random(10, 3, dist_from_edge=0, chromosomes="chr2")
    assert res["chrom"].unique().tolist() == ["chr2"]

    genes = tmp_path / "genes.txt"
    genes.write_text("\t".join(["g1", "chr1", "+", "1000", "2000", "1100", "1900", "1", "1100,", "1900,", "", ""])
                     + "\n")
    annots = tmp_path / "annots.txt"
    annots.write_text("g1\n")
    tss = pm.gintervals_import_genes(str(genes), str(annots), "gene_id")["tss"]
    assert tss["gene_id"].tolist() == ["g1"]
    assert tss[["start", "end"]].values.tolist() == [[1000, 1001]]


def test_array_colnames_take_a_single_name(array_db, tmp_path):
    pm.gtrack_array_create("one", "", pm.gintervals("chr1", 0, 10), np.array([[5.0]]), "q")
    assert pm.gtrack_array_get_colnames("one") == ["q"]
    assert pm.gtrack_array_extract("one", None, pm.gintervals("chr1", 0, 40))["q"].tolist() == [5]
    pm.gtrack_array_set_colnames("one", "zz")
    assert pm.gtrack_array_get_colnames("one") == ["zz"]
    # one name for a track of two columns: R's error, and the names stay
    with pytest.raises(ValueError, match=re.escape(
            "The number of columns in the track (2) does not match the number of column names (1)")):
        pm.gtrack_array_set_colnames("arr", "zz")
    assert pm.gtrack_array_get_colnames("arr") == ["ab", "cd"]


# Tracks created on an indexed database, and their created.by


def test_created_tracks_keep_only_the_indexed_files_on_an_indexed_database(array_db):
    db = array_db
    pm.gtrack_create_sparse("sp", "", pm.gintervals("chr1", 0, 10), [1])
    pm.gtrack_create("c_sp", "", "sp * 2", iterator="sp")
    pm.gtrack_create("c_10", "", "dn * 2", iterator=10)
    pm.gtrack_smooth("c_sm", "", "dn", 30)
    for track in ["c_sp", "c_10", "c_sm"]:
        assert sorted(os.listdir(db / "tracks" / f"{track}.track")) == [".attributes", "track.dat", "track.idx"]
    assert pm.gextract("c_sp", pm.gintervals_all())["c_sp"].tolist() == [2]
    assert pm.gextract("c_10", pm.gintervals("chr1", 0, 20))["c_10"].tolist() == [2, 2]
    # a modified track reads back the new values, whatever reads it
    pm.gtrack_modify("c_10", "c_10 * 10", pm.gintervals("chr1", 0, 20))
    assert pm.gextract("c_10", pm.gintervals("chr1", 0, 20))["c_10"].tolist() == [20, 20]


def test_created_by_records_the_iterator_as_r_writes_it(array_db):
    from _dbpath import TESTDB_ROOT

    db = array_db
    pm.gtrack_create_sparse("sp", "", pm.gintervals("chr1", 0, 10), [1])
    cases = {
        "c_sp": ("sp * 2", "sp", '"sp"'),
        "c_10": ("dn * 2", 10, "10"),
        "c_null": ("dn * 2", None, "NULL"),
        "c_big": ("dn * 2", 100000, "1e+05"),  # R's deparse(100000), with the default scipen
        "c_df": ("dn * 2", pm.gintervals("chr1", 0, 50), "intervals"),
    }
    for track, (expr, iterator, written) in cases.items():
        pm.gtrack_create(track, "", expr, iterator=iterator)
        assert pm.gtrack_attr_get(track, "created.by") == \
            f'gtrack.create("{track}", description, {expr!r}, iterator={written})'

    # a 2D expression
    pm.gtrack_2d_create("r", "", pm.gintervals_2d("chr1", 0, 100, "chr1", 0, 100), [1])
    pm.gtrack_create("r2", "", "r + 10")
    assert pm.gtrack_attr_get("r2", "created.by") == 'gtrack.create("r2", description, \'r + 10\', iterator=NULL)'

    shutil.copytree(Path(TESTDB_ROOT) / "pssms", db / "pssms")
    pm.gtrack_create_pwm_energy("c_pwm", "", "pssm", 0, 0.01, 50)
    assert pm.gtrack_attr_get("c_pwm", "created.by") == \
        'gtrack.create_pwm_energy("c_pwm", description, "pssm", 0, 0.01, iterator=50)'


# Staging directories


def dead_pid():
    """The pid of a process that has exited."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    assert not os.path.exists(f"/proc/{proc.pid}")
    return proc.pid


def leftover(parent, name):
    """A staging directory with a file in it, as a killed writer leaves it."""
    path = Path(parent) / name
    path.mkdir(parents=True)
    (path / "chr1").write_text("x")
    return path


def test_staging_name_carries_the_host_and_the_pid():
    name = _gdb_staging_name("x.track")
    assert re.fullmatch(rf"\.x\.track\.tmp\.{re.escape(_host_tag())}-{os.getpid()}\.[0-9a-f]+", name)


def test_track_writers_build_in_a_staging_directory_named_by_gdb_staging_name(make_db, tmp_path, monkeypatch):
    db = make_db()
    bases = []

    def spy(base):
        bases.append(base)
        return _gdb_staging_name(base)

    monkeypatch.setattr(_tracks, "_gdb_staging_name", spy)
    pm.gtrack_create_sparse("cs", "", pm.gintervals("chr1", 0, 10), [1])
    pm.gtrack_create_dense("cd", "", pm.gintervals("chr1", 0, 100), [1], 10)
    pm.gtrack_create("ce", "", "cd * 2", iterator=10)
    pm.gtrack_modify("cd", "cd * 2", pm.gintervals("chr1", 0, 100))
    pm.gtrack_create_sparse("sub.t", "", pm.gintervals("chr1", 0, 10), [1])
    pm.gtrack_import_parquet("", write_parquet(tmp_path / "ok.parquet", ["chr1"], ("a", "b")), 10, track_prefix="pq_")
    arr = tmp_path / "arr.tsv"
    arr.write_text("chrom\tstart\tend\ta\tb\nchr1\t0\t10\t1\t2\n")
    pm.gtrack_array_import("arr", "", str(arr))
    assert bases == ["cs.track", "cd.track", "ce.track", "cd.track", "t.track", "pq_a.track", "pq_b.track",
                     "arr.track"]
    assert staging_leftovers(db / "tracks") == []
    assert staging_leftovers(db / "tracks" / "sub") == []


def test_gdb_init_sweeps_the_staging_directory_of_a_dead_process_at_once(make_db):
    db = make_db()
    tracks = db / "tracks"
    host, dead = _host_tag(), dead_pid()
    gone = leftover(tracks, f".dead.track.tmp.{host}-{dead}.abc123")
    kept = [
        leftover(tracks, f".live.track.tmp.{host}-{os.getpid()}.abc123"),  # a live process
        leftover(tracks, f".oth.track.tmp.someotherhost-{dead}.abc123"),  # a pid of another host
        leftover(tracks, f".old.track.tmp.{dead}.abc123"),  # no host: left to the age rule
        leftover(tracks, f"sub/.dead.track.tmp.{host}-{dead}.abc123"),  # gdb_init sweeps tracks/ only
    ]
    pm.gdb_init(str(db))
    assert not gone.exists()
    assert all(p.exists() for p in kept)


def test_modify_and_parquet_import_sweep_their_directory_before_they_stage(make_db, tmp_path):
    """In a subdirectory of tracks/, which gdb_init does not sweep."""
    db = make_db()
    sub = db / "tracks" / "sub"
    host, dead = _host_tag(), dead_pid()
    pm.gtrack_create_dense("sub.d", "", pm.gintervals("chr1", 0, 100), [1], 10)
    live = leftover(sub, f".live.track.tmp.{host}-{os.getpid()}.abc123")

    gone = leftover(sub, f".dead.track.tmp.{host}-{dead}.abc123")
    pm.gtrack_modify("sub.d", "sub.d * 2", pm.gintervals("chr1", 0, 100))
    assert not gone.exists()

    gone = leftover(sub, f".dead2.track.tmp.{host}-{dead}.abc123")
    pm.gtrack_import_parquet("", write_parquet(tmp_path / "ok.parquet", ["chr1"]), 10, track_prefix="sub.pq_")
    assert not gone.exists()
    assert live.exists()


_KILLED_CREATE = """
import os, signal, sys
import pymisha as pm
from pymisha import _pymisha
pm.CONFIG["multitasking"] = False
pm.gdb_init(sys.argv[1])
real = _pymisha.pm_track_create_sparse

def create_and_die(*args):
    real(*args)
    os.kill(os.getpid(), signal.SIGKILL)

_pymisha.pm_track_create_sparse = create_and_die
pm.gtrack_create_sparse("victim", "", pm.gintervals("chr1", 0, 10), [1])
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL")
def test_the_staging_directory_of_a_killed_writer_is_swept_by_the_next_gdb_init(make_db):
    db = make_db()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    proc = subprocess.Popen([sys.executable, "-c", _KILLED_CREATE, str(db)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, err = proc.communicate()
    assert proc.returncode == -signal.SIGKILL, err
    left = staging_leftovers(db / "tracks")
    assert len(left) == 1
    assert re.fullmatch(rf"\.victim\.track\.tmp\.{re.escape(_host_tag())}-{proc.pid}\.[0-9a-f]+", left[0])

    pm.gdb_init(str(db))
    assert staging_leftovers(db / "tracks") == []
    assert not pm.gtrack_exists("victim")
