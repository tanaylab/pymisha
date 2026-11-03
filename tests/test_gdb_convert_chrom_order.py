"""gdb_convert_to_indexed keeps the chromosome names and ids gdb_init gives a database,
stops instead of converting inexactly, leaves the database as it was when it fails or is
interrupted, and puts the session loaded before back. Ported from R misha 5.13.1 (PR #193,
tests/testthat/test-db-format-conversion.R)."""

from __future__ import annotations

import grp
import multiprocessing
import os
import shutil
import signal
import stat
import warnings
from pathlib import Path

import pytest
from _chromdb_helpers import (
    SORTED_PREFIXED,
    UNSORTED_BASES,
    copy_r_bigset,
    first_bases,
    indexed_like,
    unsorted_db,
)

import pymisha as pm
import pymisha.db_create as db_create


def _write_db(path: Path, chrom_sizes: list[tuple[str, int]], seq_names: list[str]) -> str:
    (path / "seq").mkdir(parents=True)
    (path / "tracks").mkdir()
    (path / "chrom_sizes.txt").write_text("".join(f"{c}\t{s}\n" for c, s in chrom_sizes))
    for (_, size), name, base in zip(chrom_sizes, seq_names, "ACGTN", strict=False):
        (path / "seq" / f"{name}.seq").write_bytes(base.encode() * size)
    return str(path.resolve())


def _state(db: str) -> dict:
    return {
        "chrom_sizes": (Path(db) / "chrom_sizes.txt").read_bytes(),
        "seq": sorted(os.listdir(Path(db) / "seq")),
        "files": sorted(os.listdir(db)),
    }


def test_adds_chr_prefix_and_keeps_gdb_init_order(tmp_path):
    """R: adds the chr prefix and keeps the chrom order gsetroot gives the database."""
    db = _write_db(tmp_path / "db", [("15", 15), ("10", 10), ("17_random", 17), ("1", 20)],
                   ["chr15", "chr10", "chr17_random", "chr1"])
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    rows = [line.split("\t") for line in (Path(db) / "chrom_sizes.txt").read_text().splitlines()]
    assert [r[0] for r in rows] == ["chr1", "chr10", "chr15", "chr17_random"]
    assert [int(r[1]) for r in rows] == [20, 10, 15, 17]


def test_keeps_chrom_sizes_names_and_order_when_seq_files_lack_chr(tmp_path):
    """R: keeps the chrom_sizes.txt names and order when the .seq files lack the chr prefix."""
    chroms = [("chr15", 15), ("chr10", 10), ("chr17_random", 17), ("chr1", 20)]
    db = _write_db(tmp_path / "db", chroms, ["15", "10", "17_random", "1"])
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    rows = [line.split("\t") for line in (Path(db) / "chrom_sizes.txt").read_text().splitlines()]
    assert [(r[0], int(r[1])) for r in rows] == chroms


def test_keeps_chrom_ids_whether_or_not_loaded(tmp_path):
    """R: keeps the chrom ids of a per-chromosome database whether or not it is loaded."""
    loaded_db = unsorted_db(tmp_path / "loaded")
    unloaded_db = unsorted_db(tmp_path / "unloaded")
    other_db = unsorted_db(tmp_path / "other")

    def snapshot():
        return (
            pm.gintervals_all()["chrom"].tolist(),
            pm.gextract("sp", pm.gintervals_all())["sp"].tolist(),
            pm.gseq_extract(pm.gintervals(pm.gintervals_all()["chrom"].tolist(), 0, 5)),
        )

    before = {}
    for db in (loaded_db, unloaded_db):
        pm.gdb_init(db)
        as_written = [f"chr{c}" for c in ["2", "10", "1", "X", "1_KI270706v1_random"]]
        assert pm.gintervals_all()["chrom"].tolist() != as_written
        iv = pm.gintervals_all()
        iv["end"] = 100
        pm.gtrack_create_sparse("sp", "x", iv, list(range(1, len(iv) + 1)))
        pm.gtrack_convert_to_indexed("sp")
        before[db] = snapshot()

    pm.gdb_init(loaded_db)
    pm.gdb_convert_to_indexed()
    pm.gdb_init(other_db)
    pm.gdb_convert_to_indexed(groot=unloaded_db)

    for db in (loaded_db, unloaded_db):
        assert (Path(db) / "seq" / "genome.idx").exists()
        pm.gdb_init(db)
        assert snapshot() == before[db]
    # the same database whichever database was loaded
    for f in ("chrom_sizes.txt", "seq/genome.idx", "seq/genome.seq"):
        assert (Path(unloaded_db) / f).read_bytes() == (Path(loaded_db) / f).read_bytes(), f


def _convert_in_killed_child(db: str, kill_at: str) -> None:
    def child():
        if kill_at == "import":
            write_index = db_create._write_index_file

            def killing(*args, **kwargs):
                write_index(*args, **kwargs)
                os.kill(os.getpid(), signal.SIGKILL)

            db_create._write_index_file = killing
        else:
            # between replacing chrom_sizes.txt and moving genome.idx into place
            replace = os.replace

            def killing_replace(src, dst, *args, **kwargs):
                if os.path.basename(str(dst)) == "genome.idx":
                    os.kill(os.getpid(), signal.SIGKILL)
                return replace(src, dst, *args, **kwargs)

            os.replace = killing_replace
        pm.gdb_convert_to_indexed(groot=db, validate=False)

    proc = multiprocessing.get_context("fork").Process(target=child)
    proc.start()
    proc.join()
    assert proc.exitcode == -signal.SIGKILL


@pytest.mark.parametrize("kill_at", ["import", "rename"])
def test_conversion_killed_leaves_a_database_the_next_run_converts(tmp_path, kill_at):
    """R: a conversion killed at its import / rename leaves a database the next run
    converts (with an index left by an earlier, interrupted conversion)."""
    db = unsorted_db(tmp_path / "db")
    seq_dir = Path(db) / "seq"
    original = (Path(db) / "chrom_sizes.txt").read_text()
    (seq_dir / "genome.idx").write_bytes(bytes(range(1, 65)))

    _convert_in_killed_child(db, kill_at)

    assert "genome.idx.tmp" in os.listdir(seq_dir)
    assert not (seq_dir / "genome.idx").exists()
    chrom_sizes = (Path(db) / "chrom_sizes.txt").read_text()
    if kill_at == "import":
        assert chrom_sizes == original
    else:
        # chrom_sizes.txt is replaced before genome.idx is moved into place
        assert [line.split("\t")[0] for line in chrom_sizes.splitlines()] == SORTED_PREFIXED

    pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert (seq_dir / "genome.idx").exists()
    assert not (seq_dir / "genome.idx.tmp").exists()
    pm.gdb_init(db)
    assert first_bases() == UNSORTED_BASES


def test_stops_when_a_seq_file_does_not_match_chrom_sizes(tmp_path):
    """R: stops, leaving the database as it was, when the import does not match
    chrom_sizes.txt (a .seq file shorter than chrom_sizes.txt says; its .seq files are kept
    although asked to go)."""
    db = unsorted_db(tmp_path / "db")
    (Path(db) / "seq" / "chr1.seq").write_bytes(b"G" * 900)
    before = _state(db)
    with pytest.raises(ValueError, match="seq/chr1.seq has 900 bytes and chrom_sizes.txt says 1000; "
                                         "they must agree before converting"):
        pm.gdb_convert_to_indexed(groot=db, validate=False, remove_old_files=True)
    assert _state(db) == before


def test_converts_a_name_the_r_import_would_change_as_it_is(tmp_path):
    """R stops for a chromosome name its FASTA import changes ("HLA-A*01:01" becomes
    HLA-A_01_01); pymisha writes the index itself and keeps the name, so the conversion is
    exact and goes ahead."""
    db = tmp_path / "db2"
    (db / "seq").mkdir(parents=True)
    (db / "tracks").mkdir()
    (db / "seq" / "chr1.seq").write_bytes(b"A" * 100)
    (db / "seq" / "HLA-A*01:01.seq").write_bytes(b"C" * 100)
    (db / "chrom_sizes.txt").write_text("chr1\t100\nHLA-A*01:01\t100\n")
    pm.gdb_convert_to_indexed(groot=str(db), validate=True)
    pm.gdb_init(str(db))
    assert pm.gintervals_all()["chrom"].tolist() == ["chr1", "HLA-A*01:01"]
    assert pm.gseq_extract(pm.gintervals("HLA-A*01:01", 0, 3))[0].upper() == "CCC"


def test_refuses_a_database_whose_seq_is_another_databases(tmp_path):
    """R: refuses a database whose seq/ is another database's (gdb_create_linked, a dataset
    saved without copy_seq); a seq/ and chrom_sizes.txt linked to storage of their own are
    converted there."""
    parent = unsorted_db(tmp_path / "parent")

    def parent_state():
        return sorted(os.listdir(Path(parent) / "seq")), (Path(parent) / "chrom_sizes.txt").read_text()

    before = parent_state()
    refused = "is in the database .*: convert that database instead"

    linked = tmp_path / "linked"
    pm.gdb_create_linked(str(linked), parent)
    with pytest.raises(ValueError, match=refused):
        pm.gdb_convert_to_indexed(groot=str(linked), validate=False)
    pm.gdb_init(parent)
    with pytest.raises(ValueError, match=refused):
        pm.gdb_convert_to_indexed(groot=str(linked), validate=False)

    pm.gtrack_create_sparse("sp", "x", pm.gintervals("chr1", 0, 10), [1])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="sp")
    with pytest.raises(ValueError, match=refused):
        pm.gdb_convert_to_indexed(groot=str(ds), validate=False)
    assert parent_state() == before

    own = Path(unsorted_db(tmp_path / "own"))
    for f in ("seq", "chrom_sizes.txt"):
        storage = tmp_path / f"storage_{f}"
        storage.mkdir()
        (own / f).rename(storage / f)
        (own / f).symlink_to(storage / f)
    pm.gdb_convert_to_indexed(groot=str(own), validate=False)
    assert all((tmp_path / "storage_seq" / "seq" / f).exists() for f in ("genome.idx", "genome.seq"))
    assert (tmp_path / "storage_chrom_sizes.txt" / "chrom_sizes.txt").read_text().splitlines()[0] == "chr1\t1000"
    assert (own / "chrom_sizes.txt").is_symlink()
    pm.gdb_init(str(own))
    assert first_bases() == UNSORTED_BASES


@pytest.mark.parametrize("seq_name", ["Genome", "genome"])
def test_stops_for_a_chromosome_whose_sequence_file_is_genome_seq(tmp_path, seq_name):
    """R: stops for a chromosome whose sequence file is seq/genome.seq (or seq/Genome.seq,
    which a file system that ignores case takes for it)."""
    db = tmp_path / "db"
    (db / "seq").mkdir(parents=True)
    (db / "tracks").mkdir()
    (db / "seq" / "chr1.seq").write_bytes(b"A" * 100)
    (db / "seq" / f"{seq_name}.seq").write_bytes(b"C" * 50)
    (db / "chrom_sizes.txt").write_text(f"chr1\t100\n{seq_name}\t50\n")
    before = (db / "seq" / f"{seq_name}.seq").read_bytes()
    with pytest.raises(ValueError, match="seq/genome.seq, the name of the indexed format"):
        pm.gdb_convert_to_indexed(groot=str(db), validate=False, remove_old_files=True)
    assert (db / "seq" / f"{seq_name}.seq").read_bytes() == before
    assert not (db / "seq" / "genome.idx").exists()


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="no /dev/full")
def test_stops_on_a_failed_write(tmp_path, monkeypatch):
    """R: stops on a failed write, leaving the database as it was (genome.idx.tmp written
    as on a full disk)."""
    db = unsorted_db(tmp_path / "db")
    before = _state(db)
    write_index = db_create._write_index_file

    def full_disk(path, entries):
        os.symlink("/dev/full", path)
        write_index(path, entries)

    monkeypatch.setattr(db_create, "_write_index_file", full_disk)
    with pytest.raises(ValueError, match="No space left on device"):
        pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert _state(db) == before


def test_stops_when_writing_chrom_sizes_fails(tmp_path, monkeypatch):
    """R: stops when writing chrom_sizes.txt fails, leaving the database as it was."""
    db = unsorted_db(tmp_path / "db")
    before = _state(db)
    fdopen = os.fdopen

    class FullDisk:
        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.fh.close()

        def write(self, _):
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(db_create.os, "fdopen", lambda fd, *a, **k: FullDisk(fdopen(fd, *a, **k)))
    with pytest.raises(ValueError, match="Conversion failed: .*No space left on device"):
        pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert _state(db) == before
    assert before["files"] == ["chrom_sizes.txt", "seq", "tracks"]


def _refuse_hard_links(monkeypatch):
    def no_link(*_args, **_kwargs):
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(db_create.os, "link", no_link)


def test_stops_when_the_backup_copy_is_short(tmp_path, monkeypatch):
    """R: stops, with chrom_sizes.txt as it was, when its backup copy is short."""
    db = unsorted_db(tmp_path / "db")
    before = _state(db)
    _refuse_hard_links(monkeypatch)
    copyfile = shutil.copyfile

    def short_copy(src, dst, *args, **kwargs):
        copyfile(src, dst, *args, **kwargs)
        Path(dst).write_bytes(b"\x01")

    monkeypatch.setattr(db_create.shutil, "copyfile", short_copy)
    with pytest.raises(ValueError, match="Failed to keep a copy of"):
        pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert _state(db) == before


@pytest.mark.parametrize("interrupt_at", ["backup copy", "import", "chrom_sizes.txt rename"])
def test_interrupted_conversion_leaves_no_temporary_file(tmp_path, monkeypatch, interrupt_at):
    """R: gdb.convert_to_indexed interrupted (Ctrl-C) at the backup copy, the import or the
    chrom_sizes.txt rename leaves no temporary file next to chrom_sizes.txt."""
    db = unsorted_db(tmp_path / "db")
    before = (Path(db) / "chrom_sizes.txt").read_bytes()
    if interrupt_at == "backup copy":
        _refuse_hard_links(monkeypatch)
        copyfile = shutil.copyfile

        def interrupted_copy(src, dst, *args, **kwargs):
            copyfile(src, dst, *args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(db_create.shutil, "copyfile", interrupted_copy)
    elif interrupt_at == "import":
        write_index = db_create._write_index_file

        def interrupted_import(*args, **kwargs):
            write_index(*args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(db_create, "_write_index_file", interrupted_import)
    else:
        replace = os.replace

        def interrupted_replace(src, dst, *args, **kwargs):
            if os.path.basename(str(src)).startswith("chrom_sizes.txt."):
                raise KeyboardInterrupt
            return replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(db_create.os, "replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert sorted(os.listdir(db)) == ["chrom_sizes.txt", "seq", "tracks"]
    assert (Path(db) / "chrom_sizes.txt").read_bytes() == before


def test_a_warning_does_not_stop_the_conversion(tmp_path, monkeypatch):
    """R: a warning that is not from a write does not stop gdb.convert_to_indexed."""
    db = unsorted_db(tmp_path / "db")
    write_index = db_create._write_index_file

    def warning_first(*args, **kwargs):
        warnings.warn("closing unused connection 3", stacklevel=1)
        return write_index(*args, **kwargs)

    monkeypatch.setattr(db_create, "_write_index_file", warning_first)
    with pytest.warns(UserWarning, match="closing unused connection"):
        pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert all((Path(db) / "seq" / f).exists() for f in ("genome.idx", "genome.seq"))
    pm.gdb_init(db)
    assert first_bases() == UNSORTED_BASES


def test_stops_when_genome_seq_is_not_the_size_of_chrom_sizes(tmp_path, monkeypatch):
    """R: stops when genome.seq is not the size of chrom_sizes.txt (cut short)."""
    db = unsorted_db(tmp_path / "db")
    before = (Path(db) / "chrom_sizes.txt").read_bytes()
    copy = db_create._copy_binary_file

    def short(src, fh, chunk_size):
        n = copy(src, fh, chunk_size)
        fh.truncate(100)
        fh.seek(100)
        return n

    monkeypatch.setattr(db_create, "_copy_binary_file", short)
    with pytest.raises(ValueError, match="genome.seq has 100 bytes"):
        pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert (Path(db) / "chrom_sizes.txt").read_bytes() == before
    assert not {"genome.seq", "genome.idx", "genome.idx.tmp"} & set(os.listdir(Path(db) / "seq"))


@pytest.mark.parametrize("backup", ["hard link", "copy"])
def test_failure_after_replacing_chrom_sizes_puts_back_bytes_and_mode(tmp_path, monkeypatch, backup):
    """R: a conversion that fails after replacing chrom_sizes.txt puts back the same bytes and
    mode (genome.idx cannot be moved into place: a directory has its name)."""
    db = unsorted_db(tmp_path / "db")
    cs = Path(db) / "chrom_sizes.txt"
    # no newline at the end, and group-writable
    cs.write_bytes(b"2\t2000\n10\t1500\n1\t1000\nX\t1200\n1_KI270706v1_random\t500")
    os.chmod(cs, 0o664)
    before = (cs.read_bytes(), cs.stat().st_mode)
    if backup == "copy":
        _refuse_hard_links(monkeypatch)
    (Path(db) / "seq" / "genome.idx").mkdir()
    old_umask = os.umask(0o077)
    try:
        with pytest.raises(ValueError, match="genome.idx into place"):
            pm.gdb_convert_to_indexed(groot=db, validate=False)
    finally:
        os.umask(old_umask)
    assert (cs.read_bytes(), cs.stat().st_mode) == before
    assert sorted(os.listdir(db)) == ["chrom_sizes.txt", "seq", "tracks"]


def test_keeps_the_group_of_chrom_sizes(tmp_path, monkeypatch):
    """R: gdb.convert_to_indexed keeps the group of chrom_sizes.txt (also when a failed
    conversion puts back its copy)."""
    primary = os.getgid()
    others = [g for g in os.getgroups() if g != primary]
    if not others:
        pytest.skip("the test user belongs to no group besides the primary one")
    db = unsorted_db(tmp_path / "db")
    cs = Path(db) / "chrom_sizes.txt"
    try:
        os.chown(cs, -1, others[0])
    except OSError:
        pytest.skip("chgrp failed")
    gid = cs.stat().st_gid
    assert gid == others[0], grp.getgrgid(gid).gr_name
    data = cs.read_bytes()
    with monkeypatch.context() as m:
        _refuse_hard_links(m)
        (Path(db) / "seq" / "genome.idx").mkdir()
        with pytest.raises(ValueError, match="genome.idx into place"):
            pm.gdb_convert_to_indexed(groot=db, validate=False)
    (Path(db) / "seq" / "genome.idx").rmdir()
    assert cs.stat().st_gid == gid
    assert cs.read_bytes() == data
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert (Path(db) / "seq" / "genome.idx").exists()
    assert cs.stat().st_gid == gid


def test_leaves_the_users_files_next_to_chrom_sizes_alone(tmp_path):
    """R: gdb.convert_to_indexed leaves the user's files next to chrom_sizes.txt alone."""
    mine = {"chrom_sizes.txt.orig": "my backup", "chrom_sizes.txt.tmp": "my notes"}
    for fail in (True, False):
        db = unsorted_db(tmp_path / f"db{fail}")
        for f, text in mine.items():
            (Path(db) / f).write_text(text)
        if fail:
            (Path(db) / "seq" / "genome.idx").mkdir()
        try:
            pm.gdb_convert_to_indexed(groot=db, validate=False)
            failed = False
        except ValueError:
            failed = True
        assert failed == fail
        for f, text in mine.items():
            assert (Path(db) / f).read_text() == text
        assert sorted(os.listdir(db)) == sorted(["chrom_sizes.txt", *mine, "seq", "tracks"])


def _child_on_parents_seq(tmp_path: Path, parent: str, chrom_sizes: str | None = None) -> str:
    """A database made from parent with a chrom_sizes.txt of its own and a link to its seq/."""
    child = tmp_path / "child"
    (child / "tracks").mkdir(parents=True)
    text = chrom_sizes if chrom_sizes is not None else (Path(parent) / "chrom_sizes.txt").read_text()
    (child / "chrom_sizes.txt").write_text(text)
    (child / "seq").symlink_to(Path(parent) / "seq")
    return str(child.resolve())


def test_gsetroot_stops_for_a_child_whose_parent_was_converted(tmp_path):
    """R: gsetroot stops when seq/genome.idx does not match chrom_sizes.txt, keeping the
    loaded database (the parent converted with the child loaded: the child's session is
    kept, as its chrom ids are still those of the index)."""
    parent = unsorted_db(tmp_path / "parent")
    child = _child_on_parents_seq(tmp_path, parent)
    pm.gdb_init(child)
    with pytest.warns(UserWarning, match="child is loaded as before the conversion, but gsetroot\\(\\) would not "
                                         "load it now: .*does not match chrom_sizes.txt.*copy .*parent/"
                                         "chrom_sizes.txt into .*child"):
        pm.gdb_convert_to_indexed(groot=parent)
    assert child == pm._shared._GROOT
    assert first_bases() == UNSORTED_BASES
    pm.gtrack_create_sparse("written", "x", pm.gintervals("chr1", 0, 10), [1])
    assert (Path(child) / "tracks" / "written.track").is_dir()
    assert not (Path(parent) / "tracks" / "written.track").exists()

    with pytest.raises(Exception, match="does not match chrom_sizes.txt.*copy .*parent/chrom_sizes.txt into"):
        pm.gdb_init(child)
    assert child == pm._shared._GROOT

    # copying the converted parent's chrom_sizes.txt fixes it
    shutil.copy(Path(parent) / "chrom_sizes.txt", Path(child) / "chrom_sizes.txt")
    pm.gdb_init(child)
    assert first_bases() == UNSORTED_BASES


def test_a_database_that_no_longer_reads_its_own_sequence_is_unloaded(tmp_path):
    """R: a database that does not load after a conversion and no longer reads its own
    sequence is unloaded (prefixed names in chrom_sizes.txt order on the parent's seq/)."""
    parent = unsorted_db(tmp_path / "parent")
    prefixed = "".join(f"chr{line}\n" for line in (Path(parent) / "chrom_sizes.txt").read_text().splitlines())
    child = _child_on_parents_seq(tmp_path, parent, prefixed)
    pm.gdb_init(child)
    assert pm.gintervals_all()["chrom"].tolist()[0] == "chr2"
    with pytest.warns(UserWarning, match="child, loaded before the conversion, no longer reads its own "
                                         "sequence: .*No database is loaded"):
        pm.gdb_convert_to_indexed(groot=parent)
    assert pm._shared._GROOT is None


@pytest.mark.parametrize("how", ["converted", "failed in the genome step", "failed in the track step",
                                 "converted with warnings as errors"])
def test_conversion_of_a_loaded_dataset_keeps_the_rest_of_the_session(tmp_path, monkeypatch, how):
    """R: gdb.convert_to_indexed of a loaded dataset keeps the rest of the session; the
    dataset is unloaded when the conversion changed it."""
    x = unsorted_db(tmp_path / "X")
    y = unsorted_db(tmp_path / "Y")
    pm.gdb_init(y)
    iv = pm.gintervals(["chr1", "chr10", "chr2", "chrX"], 0, 10)
    pm.gtrack_create_sparse("yt", "x", iv, [1, 10, 2, 99])
    pm.gdb_init(x)
    pm.gtrack_create_sparse("zt", "x", pm.gintervals("chr1", 0, 10), [7])
    z = tmp_path / "Z"
    pm.gdataset_save(str(z), "z", tracks="zt")
    pm.gtrack_rm("zt", force=True)
    pm.gdataset_load(y)
    pm.gdataset_load(str(z))
    pm.gvtrack_create("v", "zt", "max")
    assert pm.gextract("yt", iv)["yt"].tolist() == [1, 10, 2, 99]

    def convert():
        pm.gdb_convert_to_indexed(groot=y, validate=False, convert_tracks=True)

    if how == "converted":
        with pytest.warns(UserWarning, match="Y was converted, so it is no longer loaded as a dataset\\.$"):
            convert()
    elif how == "failed in the genome step":
        (Path(y) / "seq" / "chrX.seq").write_bytes(b"A" * 10)
        with pytest.raises(ValueError, match="seq/chrX.seq has 10 bytes"):
            convert()
    elif how == "failed in the track step":
        def injected(*_args, **_kwargs):
            raise RuntimeError("injected")

        monkeypatch.setattr(db_create, "_convert_all_tracks_to_indexed", injected)
        with pytest.warns(UserWarning, match="The conversion of .*Y failed and may have changed it, so it is no "
                                             "longer loaded as a dataset"), \
                pytest.raises(RuntimeError, match="injected"):
            convert()
    else:
        # the warning is an error: the session is put back before it
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="Y was converted, so it is no longer loaded as a dataset"):
                convert()
    unloaded = how != "failed in the genome step"
    assert x == pm._shared._GROOT
    assert pm.gdataset_ls() == ([str(z.resolve())] if unloaded else [y, str(z.resolve())])
    assert pm.gvtrack_ls() == ["v"]
    assert pm.gextract("v", pm.gintervals("chr1", 0, 10))["v"].tolist() == [7]
    if unloaded:
        with pytest.raises(Exception):
            pm.gextract("yt", iv)
    else:
        assert pm.gextract("yt", iv)["yt"].tolist() == [1, 10, 2, 99]


def test_unloads_a_loaded_dataset_only_when_it_changed_it(tmp_path, monkeypatch):
    """R: gdb.convert_to_indexed unloads a loaded dataset only when it changed it."""
    p = unsorted_db(tmp_path / "P")
    x = indexed_like(p, tmp_path / "X")
    y = indexed_like(p, tmp_path / "Y")
    pm.gdb_init(x)
    pm.gdataset_load(y)
    # Y is indexed already: the conversion changes nothing, and Y stays
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.gdb_convert_to_indexed(groot=y)
        assert pm.gdataset_ls() == [y]
        # a track step with nothing to convert
        pm.gdb_convert_to_indexed(groot=y, convert_tracks=True, convert_intervals=True)
        assert pm.gdataset_ls() == [y]
    # a track step that stops: what it converted before stopping cannot be told
    with monkeypatch.context() as m:
        def injected(*_args, **_kwargs):
            raise RuntimeError("injected")

        m.setattr(db_create, "_convert_all_tracks_to_indexed", injected)
        with pytest.warns(UserWarning, match="failed and may have changed it"), \
                pytest.raises(RuntimeError, match="injected"):
            pm.gdb_convert_to_indexed(groot=y, convert_tracks=True)
    assert pm.gdataset_ls() == []
    pm.gdataset_load(y)

    # a track step that converts a track (the genome step is skipped: Y is indexed)
    pm.gdb_init(p)
    iv = pm.gintervals_all()
    iv["end"] = 10
    pm.gtrack_create_sparse("pt", "x", iv, list(range(1, len(iv) + 1)))
    shutil.copytree(Path(p) / "tracks" / "pt.track", Path(y) / "tracks" / "pt.track")
    pm.gtrack_rm("pt", force=True)
    pm.gdb_init(x)
    pm.gdataset_load(y)
    with pytest.warns(UserWarning, match="Y was converted, so it is no longer loaded as a dataset"):
        pm.gdb_convert_to_indexed(groot=y, convert_tracks=True)
    assert (Path(y) / "tracks" / "pt.track" / "track.idx").exists()
    assert pm.gdataset_ls() == []

    # an interval step that converts an interval set
    copy_r_bigset("big1d", y)
    pm.gdb_init(x)
    pm.gdataset_load(y)
    with pytest.warns(UserWarning, match="Y was converted, so it is no longer loaded as a dataset"):
        pm.gdb_convert_to_indexed(groot=y, convert_intervals=True)
    assert (Path(y) / "tracks" / "big1d.interv" / "intervals.idx").exists()
    assert pm.gdataset_ls() == []

    # a failed conversion that could not put chrom_sizes.txt back
    q = unsorted_db(tmp_path / "Q")
    r = unsorted_db(tmp_path / "R")
    pm.gdb_init(r)
    pm.gdataset_load(q)
    (Path(q) / "seq" / "genome.idx").mkdir()
    replace = os.replace
    calls = {"n": 0}

    def second_fails(src, dst, *args, **kwargs):
        if os.path.basename(str(src)).startswith("chrom_sizes.txt."):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError(2, "No such file or directory")
        return replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(db_create.os, "replace", second_fails)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="putting back chrom_sizes.txt failed too"):
            pm.gdb_convert_to_indexed(groot=q, validate=False)
    assert any("The conversion of" in str(w.message) and "Q failed and may have changed it" in str(w.message)
               for w in warned)
    assert pm.gdataset_ls() == []


def test_conversion_of_the_loaded_database_leaves_it_in_the_indexed_format(tmp_path):
    """R: gdb.convert_to_indexed of the loaded database leaves its session in the indexed
    format."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_init(db)
    pm.gdb_convert_to_indexed(convert_tracks=True)
    assert pm.tracks._db_is_indexed_at(db)
    assert first_bases() == UNSORTED_BASES


@pytest.mark.parametrize("steps", ["validate", "tracks and interval sets"])
def test_conversion_of_another_database_puts_the_session_back(tmp_path, steps):
    """R: gdb.convert_to_indexed of another database puts the session back as it was (its
    dataset, working directory and virtual tracks)."""
    db = unsorted_db(tmp_path / "db")
    other = unsorted_db(tmp_path / "other")
    pm.gdb_init(other)
    pm.gtrack_create_sparse("o", "x", pm.gintervals("chr1", 0, 10), [1])
    pm.gintervals_save(pm.gintervals("chr2", 0, 10), "oi")
    pm.gdb_init(db)
    pm.gtrack_create_sparse("t", "x", pm.gintervals("chr1", 0, 10), [1])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="t", copy_seq=True)
    pm.gtrack_rm("t", force=True)
    pm.gdataset_load(str(ds))
    pm.gvtrack_create("v", "t", "avg")
    pm.gdir_create("sub", show_warnings=False)
    pm.gdir_cd("sub")
    before = (pm._shared._GROOT, pm.gdir_cwd(), pm.gdataset_ls(), sorted(pm.gtrack_ls() or []), pm.gvtrack_ls())
    if steps == "validate":
        pm.gdb_convert_to_indexed(groot=other)
    else:
        pm.gdb_convert_to_indexed(groot=other, validate=False, convert_tracks=True, convert_intervals=True)
        assert (Path(other) / "tracks" / "o.track" / "track.idx").exists()
    assert (pm._shared._GROOT, pm.gdir_cwd(), pm.gdataset_ls(), sorted(pm.gtrack_ls() or []), pm.gvtrack_ls()) == before
    pm.gdir_cd("..")
    # the virtual track of the directory left comes back with it
    assert pm.gvtrack_ls() == ["v"]
    assert pm.gextract("t", pm.gintervals("chr1", 0, 10))["t"].tolist() == [1]
    assert first_bases() == UNSORTED_BASES


def test_mode_of_rewritten_chrom_sizes_is_kept(tmp_path):
    """chrom_sizes.txt keeps its mode through a conversion (R chmods the new file to the
    original's)."""
    db = unsorted_db(tmp_path / "db")
    cs = Path(db) / "chrom_sizes.txt"
    os.chmod(cs, 0o640)
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    assert stat.S_IMODE(cs.stat().st_mode) == 0o640
