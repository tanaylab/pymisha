"""Port of R misha tests/testthat/test-gtrack.2d.import-parallel.R.

gtrack_2d_import and gtrack_2d_import_contacts read the input files and build the
per-chromosome-pair quad trees in forked kids when multitasking is on. Each test runs the
same import with multitasking off and on and requires byte-identical track files and
identical gextract output. The parallel runs also assert that kids were really forked
(their CPU time shows up in os.times()), so a silent fallback to the serial path cannot
pass for parallel.

R's expect_counters_reset() (its RdbInitializer / PROTECT lifetime counters) has no
pymisha counterpart and is not ported.
"""

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import _pymisha, _shared

IMPORT_CHROMS = {"chr1": 1_000_000, "chr2": 800_000, "chr3": 500_000}


def _create_test_db(path):
    """R's create_test_db(): chrom_sizes.txt and a sequence file per chromosome."""
    (path / "tracks").mkdir(parents=True)
    (path / "seq").mkdir()
    (path / "chrom_sizes.txt").write_text("".join(f"{c}\t{n}\n" for c, n in IMPORT_CHROMS.items()))
    for c, n in IMPORT_CHROMS.items():
        (path / "seq" / f"{c}.seq").write_text("A" * n + "\n")
    return path


@pytest.fixture
def import_db(tmp_path):
    keys = ("multitasking", "max_processes", "max_data_size", "max_mem_usage")
    saved = {k: _shared.CONFIG.get(k) for k in keys}
    pm.gdb_init(str(_create_test_db(tmp_path / "import_db")))
    yield tmp_path
    _shared.CONFIG.update(saved)


def random_intervals(n, seed, rects=False, unknown=5):
    """Random points (or non-overlapping rectangles on a 1kb grid when rects=True), unique per
    chromosome pair, plus a few rows on chromosomes the database does not have."""
    rng = np.random.default_rng([60427, seed])
    names = np.array(list(IMPORT_CHROMS))
    sizes = pd.Series(IMPORT_CHROMS)
    chrom1 = rng.choice(names, n)
    chrom2 = rng.choice(names, n)
    size1 = sizes[chrom1].to_numpy()
    size2 = sizes[chrom2].to_numpy()
    if rects:
        start1 = np.floor(rng.random(n) * (size1 / 1000 - 1)) * 1000
        start2 = np.floor(rng.random(n) * (size2 / 1000 - 1)) * 1000
        width = 500
    else:
        start1 = np.floor(rng.random(n) * (size1 - 1))
        start2 = np.floor(rng.random(n) * (size2 - 1))
        width = 1
    start1 = start1.astype(np.int64)
    start2 = start2.astype(np.int64)
    d = pd.DataFrame({
        "chrom1": chrom1, "start1": start1, "end1": start1 + width,
        "chrom2": chrom2, "start2": start2, "end2": start2 + width,
        "value": np.round(rng.random(n) * 100, 3),
    })
    d = d.drop_duplicates(["chrom1", "start1", "chrom2", "start2"])
    unk = d.iloc[:unknown].copy()
    unk["chrom1"] = "chrUn_1"
    return pd.concat([d, unk], ignore_index=True)


def write_slices(d, fracs, prefix, where):
    """Writes consecutive slices of d into len(fracs) files and returns their paths."""
    ends = np.round(np.cumsum(fracs) / sum(fracs) * len(d)).astype(int)
    starts = np.concatenate([[0], ends[:-1]])
    paths = []
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        path = where / f"{prefix}_{i + 1}.tsv"
        d.iloc[s:e].to_csv(path, sep="\t", index=False)
        paths.append(str(path))
    return paths


def _track_dir(track):
    return _pymisha.pm_track_path(track)


def track_files_md5(track):
    d = _track_dir(track)
    return {f: hashlib.md5(open(os.path.join(d, f), "rb").read()).hexdigest()
            for f in sorted(os.listdir(d)) if not f.startswith(".")}


def _set_options(mt, max_data_size=None, max_mem_usage=None):
    _shared.CONFIG["multitasking"] = mt
    _shared.CONFIG["max_processes"] = 4
    if max_data_size is not None:
        _shared.CONFIG["max_data_size"] = max_data_size
    if max_mem_usage is not None:
        _shared.CONFIG["max_mem_usage"] = max_mem_usage


def import_both_ways(do_import, max_data_size=None, max_mem_usage=None):
    """Runs do_import(track) with multitasking off and on. Returns, per mode, the md5 of the track
    files, the gextract over the whole genome, the leftover hidden files and the kids' CPU time."""
    saved = {k: _shared.CONFIG.get(k) for k in ("multitasking", "max_processes", "max_data_size", "max_mem_usage")}
    res = {}
    try:
        for mt in (False, True):
            track = "imp_parallel" if mt else "imp_serial"
            _set_options(mt, max_data_size, max_mem_usage)
            if pm.gtrack_exists(track):
                pm.gtrack_rm(track, force=True)
            t0 = os.times()
            do_import(track)
            t1 = os.times()
            hidden = [f for f in os.listdir(_track_dir(track)) if f.startswith(".") and f != ".attributes"]
            res["parallel" if mt else "serial"] = {
                "md5": track_files_md5(track),
                "data": pm.gextract(track, pm.gintervals_2d_all(), colnames=["v"]),
                "hidden": hidden,
                "kids_cpu": (t1.children_user - t0.children_user) + (t1.children_system - t0.children_system),
            }
    finally:
        _shared.CONFIG.update(saved)
    return res


def expect_same_both_ways(res):
    assert len(res["serial"]["md5"]) > 0
    assert res["parallel"]["md5"] == res["serial"]["md5"]
    pd.testing.assert_frame_equal(res["parallel"]["data"], res["serial"]["data"])
    assert res["serial"]["hidden"] == []
    assert res["parallel"]["hidden"] == []
    assert res["parallel"]["kids_cpu"] > 0


def sort_2d(d):
    d = d.copy()
    d["chrom1"] = d["chrom1"].astype(str)
    d["chrom2"] = d["chrom2"].astype(str)
    return d.sort_values(["chrom1", "start1", "chrom2", "start2"]).reset_index(drop=True)


def expected_contacts(d):
    """What gtrack_2d_import_contacts should produce: contacts at the interval centers, duplicates
    summed, every contact also stored mirrored."""
    d = d[d["chrom1"].isin(IMPORT_CHROMS) & d["chrom2"].isin(IMPORT_CHROMS)]
    x1 = (d["start1"] + d["end1"]) // 2
    x2 = (d["start2"] + d["end2"]) // 2
    fwd = pd.DataFrame({"chrom1": d["chrom1"], "start1": x1, "chrom2": d["chrom2"], "start2": x2, "v": d["value"]})
    self_ = (d["chrom1"] == d["chrom2"]) & (x1 == x2)
    rev = pd.DataFrame({"chrom1": d["chrom2"], "start1": x2, "chrom2": d["chrom1"], "start2": x1, "v": d["value"]})
    rev = rev[~self_]
    agg = pd.concat([fwd, rev]).groupby(["chrom1", "start1", "chrom2", "start2"], as_index=False)["v"].sum()
    return sort_2d(agg)


def test_gtrack_2d_import_of_points_from_several_files_is_identical_in_parallel(import_db):
    d = random_intervals(60000, seed=1)
    files = write_slices(d, [5, 1, 3, 2], "points", import_db)
    res = import_both_ways(lambda track: pm.gtrack_2d_import(track, "test", files))
    expect_same_both_ways(res)

    assert pm.gtrack_info("imp_parallel")["type"] == "points"
    known = d[d["chrom1"].isin(IMPORT_CHROMS)]
    got = sort_2d(res["parallel"]["data"])
    want = sort_2d(known)
    assert len(got) == len(want)
    np.testing.assert_array_equal(got["start1"].to_numpy(), want["start1"].to_numpy())
    np.testing.assert_array_equal(got["start2"].to_numpy(), want["start2"].to_numpy())
    np.testing.assert_allclose(got["v"].to_numpy(), want["value"].to_numpy(), rtol=1e-6)


def test_gtrack_2d_import_of_rectangles_is_identical_in_parallel_with_a_points_only_file(import_db):
    """R: gtrack.2d.import of rectangles is identical in parallel, also when one file holds only points"""
    d = random_intervals(30000, seed=2, rects=True)
    files = write_slices(d, [2, 3, 1], "rects", import_db)
    # one file of points only: its kid sees nothing but points, the others do not
    pts = random_intervals(200, seed=3, unknown=0)
    pts["start1"] = pts["start1"] // 1000 * 1000 + 700
    pts["end1"] = pts["start1"] + 1
    pts = pts.drop_duplicates(["chrom1", "start1", "chrom2", "start2"])
    points_file = import_db / "only_points.tsv"
    pts.to_csv(points_file, sep="\t", index=False)
    files = [str(points_file), *files]

    res = import_both_ways(lambda track: pm.gtrack_2d_import(track, "test", files))
    expect_same_both_ways(res)
    assert pm.gtrack_info("imp_parallel")["type"] == "rectangles"
    assert len(res["parallel"]["data"]) == d["chrom1"].isin(IMPORT_CHROMS).sum() + len(pts)


def test_gtrack_2d_import_with_several_subtrees_per_pair_is_identical_in_parallel(import_db):
    d = random_intervals(40000, seed=4)
    files = write_slices(d, [1, 1], "subtrees", import_db)
    # well below the records of the cis pairs, so they are split into subtrees
    res = import_both_ways(lambda track: pm.gtrack_2d_import(track, "test", files), max_data_size=1000)
    expect_same_both_ways(res)
    assert len(res["parallel"]["data"]) == d["chrom1"].isin(IMPORT_CHROMS).sum()


def test_gtrack_2d_import_contacts_sums_duplicates_across_files_and_mirrors_cis_contacts_in_parallel(import_db):
    d = random_intervals(50000, seed=5)
    # the same contacts again, in both orientations, in other files: summed across kids
    rng = np.random.default_rng([60427, 105])
    dup = d.iloc[rng.choice(len(d), 3000, replace=False)]
    dup_rev = dup.iloc[:1500][["chrom2", "start2", "end2", "chrom1", "start1", "end1", "value"]]
    dup_rev.columns = d.columns
    all_ = pd.concat([d, dup, dup_rev], ignore_index=True)
    files = write_slices(d, [3, 1, 2], "contacts", import_db) + \
        write_slices(pd.concat([dup, dup_rev], ignore_index=True), [1, 1], "dups", import_db)

    res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files))
    expect_same_both_ways(res)

    got = sort_2d(res["parallel"]["data"][["chrom1", "start1", "chrom2", "start2", "v"]])
    want = expected_contacts(all_)
    assert len(got) == len(want)
    assert got["chrom1"].tolist() == want["chrom1"].tolist()
    assert got["chrom2"].tolist() == want["chrom2"].tolist()
    np.testing.assert_array_equal(got["start1"].to_numpy(), want["start1"].to_numpy())
    np.testing.assert_array_equal(got["start2"].to_numpy(), want["start2"].to_numpy())
    np.testing.assert_allclose(got["v"].to_numpy(), want["v"].to_numpy(), rtol=1e-5)


def test_gtrack_2d_import_contacts_with_several_subtrees_per_pair_is_identical_in_parallel(import_db):
    d = random_intervals(40000, seed=6)
    d = pd.concat([d, d.iloc[:2000]], ignore_index=True)
    files = write_slices(d, [1, 2, 1], "contacts", import_db)
    res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files), max_data_size=1000)
    expect_same_both_ways(res)
    assert len(res["parallel"]["data"]) == len(expected_contacts(d))


# gmax.data.size below 1 reads as 0 in C++, which the number of subtrees of a pair was divided by
# (SIGFPE in R). It now means one record per subtree.
def test_2d_imports_with_max_data_size_below_1_do_not_crash(import_db):
    files = write_slices(random_intervals(200, seed=17), [1, 1], "tiny", import_db)
    for do_import in (
        lambda track: pm.gtrack_2d_import(track, "test", files),
        lambda track: pm.gtrack_2d_import_contacts(track, "test", files),
    ):
        res = import_both_ways(do_import, max_data_size=0.5)
        assert len(res["serial"]["md5"]) > 0
        assert res["parallel"]["md5"] == res["serial"]["md5"]
        pd.testing.assert_frame_equal(res["parallel"]["data"], res["serial"]["data"])


# max_mem_usage (KB) caps the estimated memory of the pairs built at once: a fixed 4 MiB per pair
# plus its records. A 1 MB budget is below every pair, so the pairs are built one at a time. The
# larger budgets fit two of these pairs (contacts pairs are larger), and the other kids wait.
def test_gtrack_2d_import_under_a_memory_budget_is_identical_in_parallel(import_db):
    points = write_slices(random_intervals(40000, seed=11), [2, 1, 1], "points", import_db)
    rects = write_slices(random_intervals(20000, seed=12, rects=True), [1, 1], "rects", import_db)
    for budget in (1000, 10000):
        for files in (points, rects):
            res = import_both_ways(lambda track, files=files: pm.gtrack_2d_import(track, "test", files),
                                   max_mem_usage=budget)
            expect_same_both_ways(res)


def test_gtrack_2d_import_contacts_under_a_memory_budget_is_identical_in_parallel(import_db):
    d = random_intervals(40000, seed=13)
    d2 = pd.concat([d, d.iloc[:3000]], ignore_index=True)
    files = write_slices(d2, [1, 2, 1], "contacts", import_db)
    for budget in (1000, 13000):
        res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files), max_mem_usage=budget)
        expect_same_both_ways(res)
        assert len(res["parallel"]["data"]) == len(expected_contacts(d2))
    # with subtrees too
    res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files),
                           max_data_size=1000, max_mem_usage=1000)
    expect_same_both_ways(res)


def _error_both_ways(do_import, track_of_mode):
    saved = {k: _shared.CONFIG.get(k) for k in ("multitasking", "max_processes")}
    errs = []
    try:
        for mt in (False, True):
            _set_options(mt)
            track = track_of_mode(mt)
            with pytest.raises(ValueError) as err:
                do_import(track)
            assert not pm.gtrack_exists(track)
            errs.append(str(err.value))
    finally:
        _shared.CONFIG.update(saved)
    return errs


def test_gtrack_2d_import_contacts_with_allow_duplicates_false_errors_the_same_way_in_parallel(import_db):
    d = random_intervals(20000, seed=7, unknown=0)
    files = write_slices(d, [1, 1, 1], "nodup", import_db)

    # no duplicates: imports
    res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files, allow_duplicates=False))
    expect_same_both_ways(res)

    # one contact repeated in another file, the second time with its ends swapped
    dup = d.iloc[[99]][["chrom2", "start2", "end2", "chrom1", "start1", "end1", "value"]]
    dup.columns = d.columns
    dup_file = import_db / "dup.tsv"
    dup.to_csv(dup_file, sep="\t", index=False)

    errs = _error_both_ways(
        lambda track: pm.gtrack_2d_import_contacts(track, "test", [*files, str(dup_file)], allow_duplicates=False),
        lambda mt: f"dup_{mt}")
    assert "Duplicated contact" in errs[0]
    assert errs[1] == errs[0]


def test_gtrack_2d_import_contacts_with_fends_is_identical_in_parallel(import_db):
    rng = np.random.default_rng([60427, 8])
    nfends = 5000
    names = np.array(list(IMPORT_CHROMS))
    chr_ = rng.choice(names, nfends)
    fends = pd.DataFrame({
        "fend": np.arange(1, nfends + 1), "chr": chr_,
        "coord": np.floor(rng.random(nfends) * (pd.Series(IMPORT_CHROMS)[chr_].to_numpy() - 1)).astype(np.int64),
    })
    fends_file = import_db / "redb.fends"
    fends.to_csv(fends_file, sep="\t", index=False)

    n = 40000
    contacts = pd.DataFrame({
        "fend1": rng.integers(1, nfends + 51, n),  # fends above nfends are undefined: skipped
        "fend2": rng.integers(1, nfends + 51, n),
        "count": rng.integers(1, 6, n),
    })
    files = write_slices(contacts, [2, 1, 1], "fends_contacts", import_db)

    res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files, str(fends_file)))
    expect_same_both_ways(res)

    ok = (contacts["fend1"] <= nfends) & (contacts["fend2"] <= nfends)
    f1 = fends.iloc[contacts["fend1"][ok].to_numpy() - 1]
    f2 = fends.iloc[contacts["fend2"][ok].to_numpy() - 1]
    as_intervals = pd.DataFrame({
        "chrom1": f1["chr"].to_numpy(), "start1": f1["coord"].to_numpy(), "end1": f1["coord"].to_numpy() + 1,
        "chrom2": f2["chr"].to_numpy(), "start2": f2["coord"].to_numpy(), "end2": f2["coord"].to_numpy() + 1,
        "value": contacts["count"][ok].to_numpy(),
    })
    assert res["parallel"]["data"]["v"].sum() == pytest.approx(expected_contacts(as_intervals)["v"].sum(), rel=1e-6)


def test_malformed_input_gives_the_serial_error_message_in_parallel(import_db):
    d = random_intervals(9000, seed=9, unknown=0)
    files = write_slices(d, [1, 1, 1], "bad", import_db)
    with open(files[1]) as fh:
        lines = fh.read().splitlines()
    lines[499] = lines[499].rsplit("\t", 1)[0] + "\tnot_a_number"
    with open(files[1], "w") as fh:
        fh.write("".join(line + "\n" for line in lines))

    for do_import in (pm.gtrack_2d_import, pm.gtrack_2d_import_contacts):
        errs = _error_both_ways(lambda track, do_import=do_import: do_import(track, "test", files),
                                lambda mt: "bad_track")
        assert f"File {files[1]}, line 500: invalid value" in errs[0]
        assert errs[1] == errs[0]


# An input file that stat() sees but fopen() cannot open was read through a NULL FILE* in R (segfault).
def test_2d_imports_report_an_input_file_they_cannot_open(import_db):
    files = write_slices(random_intervals(100, seed=18), [1, 1], "locked", import_db)
    os.chmod(files[1], 0)
    try:
        if os.access(files[1], os.R_OK):
            pytest.skip("running as a user who can read a mode 000 file")
        for do_import in (pm.gtrack_2d_import, pm.gtrack_2d_import_contacts):
            errs = _error_both_ways(lambda track, do_import=do_import: do_import(track, "test", files),
                                    lambda mt: "imp_locked")
            for err in errs:
                assert f"Failed to open file {files[1]}" in err
    finally:
        os.chmod(files[1], 0o644)


def test_parallel_gtrack_2d_import_contacts_is_converted_in_an_indexed_database(import_db):
    root = import_db / "indexed"
    root.mkdir()
    fasta = root / "genome.fasta"
    fasta.write_text("".join(f">{c}\n{'A' * n}\n" for c, n in IMPORT_CHROMS.items()))
    db_path = root / "testdb"
    pm.gdb_create(str(db_path), str(fasta), db_format="indexed")
    pm.gdb_init(str(db_path))

    d = random_intervals(20000, seed=10)
    files = write_slices(d, [1, 1], "contacts", root)
    res = import_both_ways(lambda track: pm.gtrack_2d_import_contacts(track, "test", files))
    assert os.path.exists(os.path.join(_track_dir("imp_parallel"), "track.idx"))
    expect_same_both_ways(res)
