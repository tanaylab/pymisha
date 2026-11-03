"""gdataset_load compares a dataset's chromosome order with the working database's: one
that numbers its chromosomes differently is refused when it holds files keyed by chrom id
(indexed tracks and interval sets), and otherwise loads and keeps its own order. Ported from
R misha 5.13.1 and 5.13.3 (PRs #193 and #198, tests/testthat/test-dataset.R)."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import pytest
from _chromdb_helpers import (
    copy_r_bigset,
    indexed_like,
    unsorted_db,
    with_indexed_track,
)

import pymisha as pm

DIFFERENTLY = "numbers the chromosomes differently from the working database"


def _strip(names):
    return [n[3:] if n.startswith("chr") else n for n in names]


def test_refuses_a_dataset_numbering_chromosomes_differently_with_indexed_files(tmp_path):
    """R: gdataset.load refuses a dataset that numbers chromosomes differently and holds
    indexed tracks."""
    # P: per-chromosome, chrom ids by the sorted names; I: indexed, in chrom_sizes.txt order
    p = unsorted_db(tmp_path / "P")
    i = with_indexed_track(indexed_like(p, tmp_path / "I"))
    pi = with_indexed_track(unsorted_db(tmp_path / "PI"))
    pm.gdb_init(p)
    with pytest.raises(ValueError, match=DIFFERENTLY):
        pm.gdataset_load(i)
    assert pm.gdataset_ls() == []
    pm.gdb_init(i)
    with pytest.raises(ValueError, match=DIFFERENTLY):
        pm.gdataset_load(pi)

    # files keyed by chrom id of each kind, alone: an indexed track in a subdirectory, an
    # indexed interval set, an indexed 2D interval set
    for kind in ("nested track", "interval set", "2D interval set"):
        k = unsorted_db(tmp_path / kind.replace(" ", "_"))
        pm.gdb_init(k)
        if kind == "nested track":
            a = pm.gintervals_all()
            a["end"] = 10
            pm.gtrack_create_sparse("sub.nt", "x", a, list(range(1, len(a) + 1)))
            pm.gtrack_convert_to_indexed("sub.nt")
            assert (Path(k) / "tracks" / "sub" / "nt.track" / "track.idx").exists()
        elif kind == "interval set":
            copy_r_bigset("big1d", k)
            pm.gdb_reload()
            pm.gintervals_convert_to_indexed("big1d")
            assert (Path(k) / "tracks" / "big1d.interv" / "intervals.idx").exists()
        else:
            copy_r_bigset("big2d", k)
            pm.gdb_reload()
            pm.gintervals_2d_convert_to_indexed("big2d")
            assert (Path(k) / "tracks" / "big2d.interv" / "intervals2d.idx").exists()
        pm.gdb_init(i)
        with pytest.raises(ValueError, match=DIFFERENTLY):
            pm.gdataset_load(k)

    # the same order with names that differ by "chr": an indexed database whose
    # chrom_sizes.txt is sorted, and a per-chromosome one with that file
    sorted_root = tmp_path / "sorted_root"
    (sorted_root / "tracks").mkdir(parents=True)
    (sorted_root / "seq").mkdir()
    pm.gdb_init(p)
    ordered = pm.gintervals_all()
    (sorted_root / "chrom_sizes.txt").write_text(
        "".join(f"{c[3:]}\t{e}\n" for c, e in zip(ordered["chrom"], ordered["end"], strict=True))
    )
    sorted_root_i = indexed_like(sorted_root, tmp_path / "sorted_root_i")
    sorted_ds = tmp_path / "sorted_ds"
    (sorted_ds / "tracks").mkdir(parents=True)
    shutil.copytree(Path(p) / "seq", sorted_ds / "seq")
    shutil.copy(Path(sorted_root_i) / "chrom_sizes.txt", sorted_ds / "chrom_sizes.txt")
    with_indexed_track(str(sorted_ds.resolve()))
    truth = pm.gextract("it", pm.gintervals_all())
    pm.gdb_init(sorted_root_i)
    assert pm.gintervals_all()["chrom"].tolist() == _strip(truth["chrom"].tolist())
    pm.gdataset_load(str(sorted_ds))
    assert pm.gextract("it", pm.gintervals_all())["it"].tolist() == truth["it"].tolist()

    # an indexed track or interval set added to a dataset after it was loaded is found
    for added in (Path(pi) / "tracks" / "it.track", tmp_path / "interval_set" / "tracks" / "big1d.interv"):
        late = unsorted_db(tmp_path / f"late_{added.name}")
        pm.gdb_init(i)
        pm.gdataset_load(late)
        shutil.copytree(added, Path(late) / "tracks" / added.name)
        pm.gdb_init(i)
        with pytest.raises(ValueError, match=DIFFERENTLY):
            pm.gdataset_load(late)

    # .seq files without the "chr" prefix give chrom_sizes.txt order
    u = unsorted_db(tmp_path / "U")
    for f in (Path(u) / "seq").iterdir():
        f.rename(f.with_name(f.name[3:]))
    with_indexed_track(u)
    pm.gdb_init(p)
    with pytest.raises(ValueError, match=DIFFERENTLY):
        pm.gdataset_load(u)

    # the working database's order is the session's, whatever its seq/ holds: sorted with
    # the first chromosome's .seq file missing, chrom_sizes.txt order with no sequence
    first_missing = tmp_path / "first_missing"
    (first_missing / "tracks").mkdir(parents=True)
    (first_missing / "seq").mkdir()
    chroms6 = ["2", "10", "X", "3", "4", "1"]
    for k, c in enumerate(chroms6, start=1):
        (first_missing / "seq" / f"chr{c}.seq").write_bytes(b"A" * (1000 + k))
    (first_missing / "chrom_sizes.txt").write_text(
        "".join(f"{c}\t{1000 + k}\n" for k, c in enumerate(chroms6, start=1)))
    i6 = with_indexed_track(indexed_like(first_missing, tmp_path / "I6"))
    (first_missing / "seq" / "chr1.seq").unlink()
    pm.gdb_init(str(first_missing))
    assert pm.gintervals_all()["chrom"].tolist()[0] == "chr1"
    with pytest.raises(ValueError, match=DIFFERENTLY):
        pm.gdataset_load(i6)
    for kind in ("dangling", "empty"):
        r = tmp_path / f"R_{kind}"
        (r / "tracks").mkdir(parents=True)
        (r / "seq").mkdir()
        shutil.copy(Path(p) / "chrom_sizes.txt", r / "chrom_sizes.txt")
        if kind == "dangling":
            for f in os.listdir(Path(p) / "seq"):
                (r / "seq" / f).symlink_to(tmp_path / "nowhere" / f)
        pm.gdb_init(str(r))
        assert pm.gintervals_all()["chrom"].tolist()[0] == "2", kind
        with pytest.raises(ValueError, match=DIFFERENTLY):
            pm.gdataset_load(pi)
        # one without files keyed by chrom id loads
        pm.gdataset_load(p)
        assert pm.gdataset_ls() == [p], kind

    # both indexed: chrom_sizes.txt order on both sides, even with other contig names
    d = indexed_like(p, tmp_path / "D", contig="contig")
    pm.gdb_init(i)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.gdataset_load(d)
    assert pm.gdataset_ls() == [d]

    # the same order loads: a dataset on the working database's seq/, a copy, one without
    # sequence
    pm.gdb_init(p)
    pm.gtrack_create_sparse("t", "x", pm.gintervals("chr1", 0, 10), [1])
    linked = tmp_path / "linked"
    pm.gdataset_save(str(linked), "d", tracks="t")
    copied = tmp_path / "copied"
    pm.gdataset_save(str(copied), "d", tracks="t", copy_seq=True)
    bare = tmp_path / "bare"
    pm.gdataset_save(str(bare), "d", tracks="t")
    (bare / "seq").unlink()
    pm.gtrack_rm("t", force=True)
    for ds in (linked, copied, bare):
        pm.gdataset_load(str(ds))
        assert pm.gextract("t", pm.gintervals("chr1", 0, 10))["t"].tolist() == [1], ds
        pm.gdataset_unload(str(ds))


def _indexed_with_cs_order(p: str, path: Path) -> str:
    """I: indexed, the chrom_sizes.txt of p, chrom ids in chrom_sizes.txt order, no tracks."""
    return indexed_like(p, path)


def test_a_dataset_in_another_order_without_indexed_files_loads_and_keeps_it(tmp_path):
    """R: a dataset numbering chromosomes differently without indexed tracks loads, and
    keeps its own order."""
    from pymisha._shared import _gdb_chrom_names_at

    p = unsorted_db(tmp_path / "P")
    i = _indexed_with_cs_order(p, tmp_path / "I")
    pm.gdb_init(p)
    iv = pm.gintervals(["1", "2", "10", "X"], 0, 10)
    pm.gtrack_create_sparse("pt", "x", iv, [1, 2, 10, 99])
    pm.gtrack_create_sparse("pi", "x", iv, [1, 2, 10, 99])
    pm.gtrack_convert_to_indexed("pi")
    pm.gtrack_2d_create("p2", "x", pm.gintervals_2d("chr1", 0, 10, "chr2", 0, 10), [5])
    pm.gintervals_save(iv, "pv")
    pm.gintervals_save(pm.gintervals_2d("chr1", 0, 10, "chr2", 0, 10), "pv2")

    def by_name(track):
        res = pm.gextract(track, iv)
        values = dict(zip(_strip(res["chrom"].astype(str).tolist()), res[track], strict=True))
        return [values[c] for c in ["1", "2", "10", "X"]]

    truth = by_name("pt")
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks=["pt", "p2"], intervals=["pv", "pv2"], copy_seq=True)
    ds = ds.resolve()
    pm.gdb_init(i)
    iv = pm.gintervals(["1", "2", "10", "X"], 0, 10)
    pm.gdataset_load(str(ds))
    assert by_name("pt") == truth
    assert _strip(_gdb_chrom_names_at(str(ds)))[0] == "1"
    # nothing of it is written in the indexed format in this session
    for convert in (lambda: pm.gtrack_convert_to_indexed("pt"), lambda: pm.gtrack_2d_convert_to_indexed("p2"),
                    lambda: pm.gintervals_convert_to_indexed("pv"), lambda: pm.gintervals_2d_convert_to_indexed("pv2")):
        with pytest.raises(ValueError, match="in a dataset that numbers the chromosomes differently"):
            convert()

    # the order kept is dropped with the dataset, by gdataset_unload, gdb_init and a
    # gdb_init that fails
    assert pm._shared._GDATASET_CHROMS.get(str(ds)) is not None
    pm.gdataset_unload(str(ds))
    assert str(ds) not in pm._shared._GDATASET_CHROMS
    pm.gdataset_load(str(ds))
    # loaded again while loaded: listed once
    pm.gdataset_load(str(ds))
    assert pm.gdataset_ls() == [str(ds)]
    pm.gdb_init(i)
    assert pm._shared._GDATASET_CHROMS == {}

    # copies into I loaded as a dataset of P, of an indexed and a per-chromosome track, by
    # db = and from inside its tracks/, are written in files named by chromosome: they read
    # right in this session and with I as the working database
    pm.gdb_init(p)
    iv = pm.gintervals(["1", "2", "10", "X"], 0, 10)
    pm.gdataset_load(i)
    pm.gtrack_copy("pi", "pic", db=i)
    pm.gtrack_copy("pt", "ptc", db=i)
    pm.gdir_cd(str(Path(i) / "tracks"))
    pm.gtrack_copy("pi", "pic2")
    pm.gdir_cd(str(Path(p) / "tracks"))
    for t in ("pic", "ptc", "pic2"):
        assert not (Path(i) / "tracks" / f"{t}.track" / "track.idx").exists(), t
        assert by_name(t) == truth, t
    pm.gdb_init(i)
    iv = pm.gintervals(["1", "2", "10", "X"], 0, 10)
    assert by_name("pic") == truth
    assert by_name("ptc") == truth
    assert by_name("pic2") == truth
    pm.gdb_init(p)
    pm.gdataset_load(i)
    assert pm.gdataset_ls() == [i]


def test_refuses_a_dataset_whose_linked_seq_was_converted_since(tmp_path):
    """R: gdataset.load refuses a dataset whose linked seq/ belongs to a database converted
    since."""
    p = unsorted_db(tmp_path / "P")
    i = _indexed_with_cs_order(p, tmp_path / "I")
    # D: an indexed track of P (sorted chrom ids), saved with copy_seq = False
    with_indexed_track(p, "t")
    d = tmp_path / "D"
    pm.gdataset_save(str(d), "d", tracks="t", copy_seq=False)
    pm.gtrack_rm("t", force=True)
    pm.gdb_init(i)
    with pytest.raises(ValueError, match="numbers the chromosomes differently"):
        pm.gdataset_load(str(d))
    # P converted since: D's seq/ is indexed in P's order, which D's chrom_sizes.txt does not match
    pm.gdb_convert_to_indexed(groot=p, validate=False)
    assert (d / "seq" / "genome.idx").exists()
    pm.gdb_init(i)
    with pytest.raises(ValueError, match="its seq/genome.idx does not match its chrom_sizes.txt"):
        pm.gdataset_load(str(d))


def test_order_of_a_linked_seq_of_the_same_chrom_sizes_is_not_read(tmp_path, monkeypatch):
    """R: gdataset.load does not read the order of a linked seq/ whose database has the same
    chrom_sizes.txt, and loads one whose index cannot be read."""
    import pymisha._shared as shared

    p = unsorted_db(tmp_path / "P")
    r = indexed_like(p, tmp_path / "R")
    s = indexed_like(p, tmp_path / "S")
    # D: an indexed track of S, saved with copy_seq = False (seq/ a link to S's)
    with_indexed_track(s, "t")
    a = pm.gintervals_all()
    a["end"] = 10
    d = tmp_path / "D"
    pm.gdataset_save(str(d), "d", tracks="t", copy_seq=False)
    pm.gtrack_rm("t", force=True)
    assert (d / "seq").is_symlink()
    looked = {"n": 0}
    chrom_names_at = shared._gdb_chrom_names_at

    def counting(*args, **kwargs):
        looked["n"] += 1
        return chrom_names_at(*args, **kwargs)

    monkeypatch.setattr(shared, "_gdb_chrom_names_at", counting)

    def load_d():
        looked["n"] = 0
        pm.gdb_init(r)
        pm.gdataset_load(str(d))
        assert pm.gextract("t", a)["t"].tolist() == list(range(1, len(a) + 1))
        return looked["n"]

    # S's chrom_sizes.txt is D's: the order is not read
    assert load_d() == 0
    # S's chrom_sizes.txt of the same size but not D's (two lines swapped): the order is read
    cs_s = (Path(s) / "chrom_sizes.txt").read_text().splitlines()
    (Path(s) / "chrom_sizes.txt").write_text("\n".join([cs_s[1], cs_s[0], *cs_s[2:]]) + "\n")
    assert load_d() == 1
    # seq/ in a directory of no database: the order is read
    store = tmp_path / "store"
    store.mkdir()
    (Path(s) / "seq").rename(store / "seq")
    (d / "seq").unlink()
    (d / "seq").symlink_to(store / "seq")
    assert load_d() == 1
    # an index that cannot be read (truncated) leaves the order unknown: D loads as before
    idx = store / "seq" / "genome.idx"
    idx.write_bytes(idx.read_bytes()[: idx.stat().st_size // 2])
    with pytest.raises(Exception):
        pm.gdb_init(str(d))
    assert load_d() == 1


@pytest.mark.parametrize("how", ["truncated", "unreadable"])
def test_refuses_an_unreadable_own_index_in_another_order(tmp_path, how):
    """R: gdataset.load refuses a dataset with an indexed seq/ of its own whose index cannot
    be read, holding indexed files in another order."""
    if how == "unreadable" and os.geteuid() == 0:
        pytest.skip("root reads any file")
    p = unsorted_db(tmp_path / "P")
    # Q: per-chromosome as P, with its chrom_sizes.txt in the order gdb_init gives P's
    q = unsorted_db(tmp_path / "Q")
    pm.gdb_init(p)
    ordered = pm.gintervals_all()
    (Path(q) / "chrom_sizes.txt").write_text(
        "".join(f"{c[3:]}\t{e}\n" for c, e in zip(ordered["chrom"], ordered["end"], strict=True))
    )
    # with an indexed track, P's chrom_sizes.txt (another order than P's ids) and Q's (the same)
    d = with_indexed_track(indexed_like(p, tmp_path / "D"), "t")
    f = with_indexed_track(indexed_like(q, tmp_path / "F"), "t")
    # with P's chrom_sizes.txt and only per-chromosome tracks, P's
    e = indexed_like(p, tmp_path / "E")
    pm.gdb_init(p)
    pm.gtrack_create_sparse("pt", "x", pm.gintervals("chr1", 0, 10), [1])
    pm.gtrack_2d_create("pr", "x", pm.gintervals_2d("chr1", 0, 10, "chr2", 0, 10), [1])
    for t in ("pt.track", "pr.track"):
        shutil.copytree(Path(p) / "tracks" / t, Path(e) / "tracks" / t)
    pm.gtrack_rm("pt", force=True)
    pm.gtrack_rm("pr", force=True)
    pm.gdb_init(f)
    truth = pm.gextract("t", pm.gintervals_all())
    pm.gdb_init(p)
    with pytest.raises(ValueError, match="numbers the chromosomes differently"):
        pm.gdataset_load(d)

    for ds in (d, e, f):
        idx = Path(ds) / "seq" / "genome.idx"
        shutil.copy(idx, f"{idx}.orig")
        if how == "truncated":
            idx.write_bytes(idx.read_bytes()[: idx.stat().st_size // 2])
        else:
            os.chmod(idx, 0)
        with pytest.raises(Exception):
            pm.gdb_init(ds)
    try:
        # in chrom_sizes.txt order, unlike P's chrom ids: refused for its indexed track
        pm.gdb_init(p)
        with pytest.raises(ValueError, match="seq/genome.idx could not be read .*cannot be checked"):
            pm.gdataset_load(d)
        assert pm.gdataset_ls() == []
        # without indexed files it loads, and keeps that order
        pm.gdataset_load(e)
        assert pm.gdataset_ls() == [e]
        assert pm.gextract("pt", pm.gintervals("chr1", 0, 10))["pt"].tolist() == [1]
        from pymisha._shared import _gdb_chrom_names_at
        assert _strip(_gdb_chrom_names_at(e)) == ["2", "10", "1", "X", "1_KI270706v1_random"]
        # its per-pair 2D track copies out, though its recorded names lack "chr"
        pm.gtrack_copy("pr", "prcopy", db=p)
        assert pm.gextract("prcopy", pm.gintervals_2d_all())["prcopy"].tolist() == [1]
        pm.gtrack_rm("prcopy", force=True)
        # in the order of the working database's chrom ids it loads and reads right
        pm.gdb_init(q)
        pm.gdataset_load(f)
        assert pm.gextract("t", pm.gintervals_all())["t"].tolist() == truth["t"].tolist()
    finally:
        for ds in (d, e, f):
            idx = Path(ds) / "seq" / "genome.idx"
            os.chmod(idx, 0o644)
            os.replace(f"{idx}.orig", idx)


def test_no_warning_for_a_datasets_index_names(tmp_path):
    """R: gdataset.load gives no warning for a dataset's index names (contigs named
    otherwise in genome.idx than in chrom_sizes.txt)."""
    p = unsorted_db(tmp_path / "P")
    d = indexed_like(p, tmp_path / "D", contig="contig")
    pm.gdb_init(p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm.gdataset_load(d)


def test_a_datasets_track_is_converted_only_in_its_own_database(tmp_path):
    """R (PR #197, the part without gdir_cd): gtrack.convert_to_indexed and
    gtrack.2d.convert_to_indexed convert only tracks of the working database; a loaded
    dataset's track, which the session may not number as its database does, is refused."""
    db = unsorted_db(tmp_path / "db")
    other = unsorted_db(tmp_path / "other")
    pm.gdb_init(other)
    iv = pm.gintervals(["chr1", "chr2"], 0, 10)
    pm.gtrack_create_sparse("x", "x", iv, [3, 3])
    pm.gtrack_2d_create("r", "x", pm.gintervals_2d("chr1", 0, 10, "chr2", 0, 10), [1])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks=["x", "r"])
    pm.gdb_init(db)
    pm.gdataset_load(str(ds))
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_convert_to_indexed("x")
    with pytest.raises(ValueError, match="is not in the working database"):
        pm.gtrack_2d_convert_to_indexed("r")
    assert not (ds / "tracks" / "x.track" / "track.idx").exists()
    assert not (ds / "tracks" / "r.track" / "track.idx").exists()
    # a copy into the working database converts there
    pm.gtrack_copy("x", "y")
    pm.gtrack_convert_to_indexed("y")
    assert pm.gextract("y", iv)["y"].tolist() == [3, 3]
