"""R misha 5.17.0 import and export fixes (#224) without a better home: gtrack_array_import
clips records past the chromosome end (test-review-2026-11-import.R), gtrack_2d_import_contacts
takes memory by the number of fends, and a dead worker of a track-parallel gextract is an error
(test-gmultitasking-strategy.R).

The database is R misha's import_test_db() (chr1, chr2, chrX, chrM), as in test_misha_5_16_import.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from test_misha_5_16_import import import_db, write_file  # noqa: F401 - pytest fixtures

import pymisha as pm


def test_array_import_clips_records_past_the_chromosome_end_with_one_warning(import_db, write_file):  # noqa: F811
    import_db("indexed")
    # chrM is 500 bp: the second record is clipped, the third dropped
    f = write_file("chrom\tstart\tend\ta\tb\nchrM\t0\t400\t1\t2\nchrM\t400\t600\t3\t4\nchrM\t700\t800\t5\t6\n", ".tsv")
    with pytest.warns(UserWarning) as caught:
        pm.gtrack_array_import("arr_clip", "", f)
    assert [str(w.message) for w in caught] == [
        f"File {f}: 1 record(s) running past the end of their chromosome were clipped to it, and 1 record(s) "
        "starting at or past the chromosome end were dropped; the first: line 3: (chrM, 400, 600), chromosome size 500"
    ]
    res = pm.gtrack_array_extract("arr_clip", None, pm.gintervals_all())
    assert res["start"].tolist() == [0, 400]
    assert res["end"].tolist() == [400, 500]
    assert res["a"].tolist() == [1, 3]
    assert res["b"].tolist() == [2, 4]


def test_array_import_of_a_file_whose_every_record_starts_past_the_chromosome_end_is_an_error(
        import_db, write_file):  # noqa: F811
    # as for BED: nothing would be left to import; the records are dropped, so the track would be empty
    import_db("indexed")
    chr1_end = int(pm.gintervals_all().set_index("chrom").loc["chr1", "end"])
    f = write_file("chrom\tstart\tend\ta\n" + "".join(f"chr1\t{chr1_end + d}\t{chr1_end + d + 10}\t{i}\n"
                                                      for i, d in enumerate((10, 100), 1)), ".tsv")
    with pytest.raises(ValueError, match="nothing to import"):
        pm.gtrack_array_import("past_end_arr", "x", f)
    assert not pm.gtrack_exists("past_end_arr")


def test_import_contacts_takes_memory_by_the_number_of_fends_not_by_the_largest_fend_id(
        import_db, write_file, tmp_path):  # noqa: F811
    if not sys.platform.startswith("linux"):
        pytest.skip("uses ulimit -v of a Linux shell")
    db = import_db("indexed")

    big_ids = [2147483000, 2147483001, 1000000000, 5]

    def fends_text(ids):
        chroms, coords = ["chr1", "chr1", "chr2", "chrX"], [1000, 2000, 300, 40]
        return "fend\tchr\tcoord\n" + "".join(f"{i}\t{c}\t{x}\n" for i, c, x in zip(ids, chroms, coords, strict=True))

    def contacts_text(ids):
        pairs = [(1, 2, 3), (1, 3, 4), (2, 4, 1), (3, 4, 2)]
        return "fend1\tfend2\tcount\n" + "".join(f"{ids[a - 1]}\t{ids[b - 1]}\t{n}\n" for a, b, n in pairs)

    pm.gtrack_2d_import_contacts("small_ids", "", write_file(contacts_text([1, 2, 3, 4]), ".tsv"),
                                 write_file(fends_text([1, 2, 3, 4]), ".tsv"))

    # in a process limited to 3 GB of address space: arrays indexed by the fend id would take 24 GB
    fends = write_file(fends_text(big_ids), ".tsv")
    contacts = write_file(contacts_text(big_ids), ".tsv")
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import pymisha as pm
        pm.gdb_init({str(db)!r})
        pm.CONFIG["multitasking"] = False
        pm.gtrack_2d_import_contacts("big_ids", "", {contacts!r}, {fends!r})
    """))
    import _pymisha

    # the pymisha under test, not an installed one
    path = [os.path.dirname(os.path.dirname(pm.__file__)), os.path.dirname(_pymisha.__file__)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([*path, os.environ.get("PYTHONPATH", "")]))
    res = subprocess.run(["sh", "-c", f'ulimit -v 3000000 && exec "{sys.executable}" "{script}"'],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr

    pm.gdb_reload()
    big = pm.gextract("big_ids", pm.gintervals_2d_all(), colnames=["v"])
    small = pm.gextract("small_ids", pm.gintervals_2d_all(), colnames=["v"])
    pd.testing.assert_frame_equal(big, small)


# ---- gextract: a dead worker of the track-parallel strategy ------------------------------------


def _restore(config, saved):
    config.clear()
    config.update(saved)


def test_track_parallel_gextract_fails_when_a_worker_dies_instead_of_dropping_its_columns(monkeypatch):
    from pymisha import extract as _extract

    parent_pid = os.getpid()
    orig = _extract._worker_extract_tracks

    # stands in for the out-of-memory killer
    def die_in_worker(args):
        if os.getpid() != parent_pid and "dense_track * 3" in args[0]:
            os.kill(os.getpid(), 9)
        return orig(args)

    monkeypatch.setattr(_extract, "_worker_extract_tracks", die_in_worker)
    saved = dict(pm.CONFIG)
    try:
        pm.CONFIG.update({"multitasking": True, "multitasking_strategy": "tracks", "min_processes": 3,
                          "max_processes": 3, "min_intervs4process": 0, "min_scope4process": 0})
        exprs = ["dense_track", "dense_track * 3", "dense_track * 2"]
        iv = pm.gintervals("1", 0, 100000)
        with pytest.raises(RuntimeError, match="1 of 3 worker processes ended without returning a result"):
            pm.gextract(exprs, iv, iterator=100, colnames=["a", "b", "c"])
        # the workers that live give the serial result
        monkeypatch.setattr(_extract, "_worker_extract_tracks", orig)
        got = pm.gextract(exprs, iv, iterator=100, colnames=["a", "b", "c"])
        pm.CONFIG["multitasking"] = False
        pd.testing.assert_frame_equal(got, pm.gextract(exprs, iv, iterator=100, colnames=["a", "b", "c"]))
    finally:
        _restore(pm.CONFIG, saved)


def test_tiled_gextract_fails_when_a_worker_dies(monkeypatch):
    from pymisha import extract as _extract

    parent_pid = os.getpid()
    orig = _extract._worker_extract_chunk

    def die_in_worker(args):
        if os.getpid() != parent_pid and str(args[0]["chrom"][0]) in ("2", "chr2"):
            os.kill(os.getpid(), 9)
        return orig(args)

    monkeypatch.setattr(_extract, "_worker_extract_chunk", die_in_worker)
    saved = dict(pm.CONFIG)
    try:
        pm.CONFIG.update({"multitasking": True, "multitasking_strategy": "tiles", "min_processes": 4,
                          "max_processes": 4, "min_intervs4process": 0, "min_scope4process": 0})
        with pytest.raises(RuntimeError, match=r"Child process \d+ ended unexpectedly \(signal 9\)"):
            pm.gextract("dense_track", pm.gintervals(["1", "2"], 0, 100000), iterator=100)
    finally:
        _restore(pm.CONFIG, saved)


def test_a_worker_error_is_raised_as_it_was():
    from pymisha import extract as _extract

    def fail(args):
        if args == 2:
            raise KeyError("no such thing")
        return args * 10

    with pytest.raises(KeyError, match="no such thing"):
        _extract._run_forked(fail, [1, 2, 3])
    assert _extract._run_forked(fail, [1, 3]) == ([10, 30], [])
    # a large result comes through the pipe whole
    got, dead = _extract._run_forked(lambda n: np.arange(n), [3_000_000, 5])
    assert not dead
    np.testing.assert_array_equal(got[0], np.arange(3_000_000))


# ---- gextract(file=) writes the rows as they come, as R misha does --------------------------------


def _whole_result_files(tmp_path, exprs, scope, iterator, colnames, **kw):
    """The tab-delimited and Parquet files gextract wrote from the whole result, before it wrote
    them a part at a time: the in-memory result through the same writers."""
    import _pymisha

    from pymisha import extract as _extract

    df = pm.gextract(exprs, scope, iterator=iterator, colnames=colnames, **kw)
    tsv, pq = tmp_path / "whole.tsv", tmp_path / "whole.parquet"
    if df is not None:
        df.drop(columns=["intervalID"]).to_csv(tsv, sep="\t", index=False, float_format="%.15g", na_rep="nan")
    with _extract._parquet_output(str(pq)) as (name, tmp):
        _pymisha.pm_parquet_write(tmp, name, *_extract._parquet_rows(df, list(colnames or exprs)),
                                  list(colnames or exprs), False, _extract._parquet_created_by(),
                                  _extract._parquet_value_metadata(exprs))
    return (tsv.read_bytes() if df is not None else None), pq.read_bytes(), 0 if df is None else len(df)


_SCOPE_OVERLAP = pd.DataFrame({"chrom": ["1", "1", "2", "1", "1", "1"], "start": [5000, 0, 100, 0, 7000, 7000],
                               "end": [9000, 20000, 30000, 300, 7100, 7100]})
_ITERATOR_DF = pd.DataFrame({"chrom": ["1", "1", "1", "2"], "start": [10, 6000, 8000, 50],
                             "end": [5500, 7050, 12000, 40000]})


@pytest.mark.parametrize(("exprs", "scope", "iterator"), [
    (["dense_track", "dense_track * 2"], None, 50),                 # fixed bins: intervals cut in pieces
    (["dense_track"], None, None),                                  # the bins of the track
    (["dense_track", "sparse_track"], _SCOPE_OVERLAP, 70),          # overlapping and repeated scope rows
    (["dense_track"], None, "sparse_track"),                        # a track's intervals
    (["dense_track"], _SCOPE_OVERLAP, _ITERATOR_DF),                # a DataFrame iterator
    (["sparse_track"], None, None),                                 # a sparse track's own intervals
    (["vmax", "dense_track * k"], None, 50),                        # a virtual track, a user variable
    (["START", "CHROM == '1'"], _SCOPE_OVERLAP, 100),
])
def test_file_output_written_a_part_at_a_time_is_the_whole_result(tmp_path, monkeypatch, exprs, scope, iterator):
    from pymisha import extract as _extract

    scope = pm.gintervals_all() if scope is None else scope
    pm.gvtrack_create("vmax", "dense_track", func="max")
    pm.gvtrack_iterator("vmax", sshift=-100, eshift=100)
    saved = dict(pm.CONFIG)
    try:
        for multitask in (False, True):
            pm.CONFIG["multitasking"] = multitask
            colnames = [f"c{i}" for i in range(len(exprs))]
            want_tsv, want_pq, n = _whole_result_files(tmp_path, exprs, scope, iterator, colnames, vars={"k": 3.5})
            assert n > 0
            # a few rows a part: many parts, and intervals cut in pieces
            for part_rows in (300, 1 << 21):
                monkeypatch.setattr(_extract, "_FILE_PART_ROWS", part_rows)
                label = f"multitask {multitask}, {part_rows} rows a part"
                got_tsv, got_pq = tmp_path / "x.tsv", tmp_path / "x.parquet"
                for f in (got_tsv, got_pq):
                    assert pm.gextract(exprs, scope, iterator=iterator, colnames=colnames, file=str(f),
                                       vars={"k": 3.5}) is None
                assert got_tsv.read_bytes() == want_tsv, label
                assert got_pq.read_bytes() == want_pq, label
    finally:
        _restore(pm.CONFIG, saved)
        pm.gvtrack_rm("vmax")


def test_file_output_is_not_limited_by_max_data_size(tmp_path):
    saved = dict(pm.CONFIG)
    try:
        pm.CONFIG.update({"max_data_size": 100, "multitasking": False})
        with pytest.raises(Exception, match="exceeded the maximum allowed"):
            pm.gextract("dense_track", pm.gintervals_all(), iterator=50)
        for name in ("x.tsv", "x.parquet"):
            f = tmp_path / name
            assert pm.gextract("dense_track", pm.gintervals_all(), iterator=50, file=str(f)) is None
        assert pm.CONFIG["max_data_size"] == 100
        pm.CONFIG["max_data_size"] = None
        whole = pm.gextract("dense_track", pm.gintervals_all(), iterator=50)
        assert len(pd.read_csv(tmp_path / "x.tsv", sep="\t")) == len(whole) > 100
    finally:
        _restore(pm.CONFIG, saved)


def test_file_output_with_no_rows_is_the_header(tmp_path):
    # as R misha 5.17.0 writes them (checked on this database); a Parquet file of one empty row group
    empty = pm.gintervals("1", 60, 90)
    assert pm.gextract("sparse_track", empty) is None
    for multitask in (False, True):
        saved = dict(pm.CONFIG)
        try:
            pm.CONFIG["multitasking"] = multitask
            assert pm.gextract("sparse_track", empty, file=str(tmp_path / "x.tsv")) is None
            assert (tmp_path / "x.tsv").read_text() == "chrom\tstart\tend\tsparse_track\n"
            pm.gextract(["sparse_track", "sparse_track * 2"], empty, colnames=["a", "b"], file=str(tmp_path / "x.tsv"))
            assert (tmp_path / "x.tsv").read_text() == "chrom\tstart\tend\ta\tb\n"
            pm.gextract("rects_track", pm.gintervals_2d("1", 0, 10, "1", 0, 10), file=str(tmp_path / "x2d.tsv"))
            assert (tmp_path / "x2d.tsv").read_text() == "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\trects_track\n"
            pm.gextract("sparse_track", empty, file=str(tmp_path / "x.parquet"))
            assert (tmp_path / "x.parquet").read_bytes()[:4] == b"PAR1"
        finally:
            _restore(pm.CONFIG, saved)


def _grid_2d_scope():
    """Rectangles that tile every chromosome pair of the database, shuffled: more than 16 of a
    pair, so that R's sort of the scope by chromosome pair (an unstable std::sort) reorders the
    rectangles of a pair."""
    sizes = pm.gintervals_all().set_index("chrom")["end"].to_dict()
    rows = []
    for c1, n1 in sizes.items():
        for c2, n2 in sizes.items():
            e1 = np.linspace(0, n1, 7).astype(int)
            e2 = np.linspace(0, n2, 6).astype(int)
            rows += [(c1, e1[i], e1[i + 1], c2, e2[j], e2[j + 1]) for i in range(6) for j in range(5)]
    scope = pd.DataFrame(rows, columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2"])
    return scope.sample(frac=1, random_state=60427).reset_index(drop=True)


@pytest.mark.parametrize(("exprs", "kw"), [
    (["rects_track"], {}),
    (["rects_track"], {"iterator": (30000, 30000)}),
    (["rects_track", "rects_track * 2"], {"iterator": "rects_track"}),
    (["v2"], {"iterator": (100000, 100000)}),
    (["v2 + rects_track"], {}),
    (["rects_track"], {"band": (-100000, 100000)}),
])
def test_2d_file_output_written_a_chromosome_pair_at_a_time_is_the_whole_result(tmp_path, exprs, kw):
    pm.gvtrack_create("v2", "rects_track", func="max")
    saved = dict(pm.CONFIG)
    try:
        for scope in (pm.gintervals_2d_all(), _grid_2d_scope()):
            for multitask in (False, True):
                pm.CONFIG["multitasking"] = multitask
                df = pm.gextract(exprs, scope, **kw)
                assert df is not None
                df.drop(columns=["intervalID"]).to_csv(tmp_path / "whole.tsv", sep="\t", index=False,
                                                       float_format="%.15g", na_rep="nan")
                assert pm.gextract(exprs, scope, file=str(tmp_path / "x.tsv"), **kw) is None
                assert (tmp_path / "x.tsv").read_bytes() == (tmp_path / "whole.tsv").read_bytes()
    finally:
        _restore(pm.CONFIG, saved)
        pm.gvtrack_rm("v2")


def test_2d_file_output_checks_the_scope_before_writing(tmp_path):
    scope = _grid_2d_scope()
    overlapping = pd.concat([scope, scope.iloc[[len(scope) - 1]]], ignore_index=True)
    with pytest.raises(Exception, match="overlap"):
        pm.gextract("rects_track", overlapping, file=str(tmp_path / "x.tsv"))
    assert not (tmp_path / "x.tsv").exists()


def test_random_draws_of_a_file_output_follow_the_seed(tmp_path, monkeypatch):
    # Each part is a call of its own, whose draws are seeded from numpy's generator: a seed
    # reproduces the file, and a scope of one part draws what the in-memory call draws
    from pymisha import extract as _extract

    pm.gvtrack_create("vsamp", "dense_track", func="sample")
    saved = dict(pm.CONFIG)
    try:
        pm.CONFIG["multitasking"] = False
        np.random.seed(60427)
        df = pm.gextract("vsamp", pm.gintervals_all(), iterator=100)
        df.drop(columns=["intervalID"]).to_csv(tmp_path / "whole.tsv", sep="\t", index=False, float_format="%.15g",
                                               na_rep="nan")
        np.random.seed(60427)
        pm.gextract("vsamp", pm.gintervals_all(), iterator=100, file=str(tmp_path / "one.tsv"))
        assert (tmp_path / "one.tsv").read_bytes() == (tmp_path / "whole.tsv").read_bytes()

        monkeypatch.setattr(_extract, "_FILE_PART_ROWS", 50)
        for name in ("a.tsv", "b.tsv"):
            np.random.seed(60427)
            pm.gextract("vsamp", pm.gintervals_all(), iterator=100, file=str(tmp_path / name))
        assert (tmp_path / "a.tsv").read_bytes() == (tmp_path / "b.tsv").read_bytes()
    finally:
        _restore(pm.CONFIG, saved)
        pm.gvtrack_rm("vsamp")


def test_a_file_output_reads_the_callers_variables(tmp_path, monkeypatch):
    from pymisha import extract as _extract

    monkeypatch.setattr(_extract, "_FILE_PART_ROWS", 300)
    k = 3.5
    whole = pm.gextract("dense_track * k", pm.gintervals_all(), iterator=50)
    pm.gextract("dense_track * k", pm.gintervals_all(), iterator=50, file=str(tmp_path / "x.tsv"))
    got = pd.read_csv(tmp_path / "x.tsv", sep="\t")
    np.testing.assert_allclose(got["dense_track * k"], whole["dense_track * k"], rtol=1e-14)
    assert k == 3.5
