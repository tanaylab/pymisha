"""gtrack_import_parquet and gtrack_import of a Parquet file, ported from R misha's
test-gtrack.import_parquet.R. The fixtures are R misha's, written by pyarrow with
fixtures/parquet/make_fixtures.py; see that script for what each one holds."""

from __future__ import annotations

import gzip
import logging
import os
import re
import shutil
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha import tracks as _tracks

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "parquet"


def pq_fixture(name: str) -> str:
    return str(FIXTURES / name)


def _mk_db(root: Path, sizes: dict[str, int]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fastas = []
    for chrom, size in sizes.items():
        fasta = root / f"{chrom}.fasta"
        fasta.write_text(f">{chrom}\n{'A' * size}\n")
        fastas.append(str(fasta))
    db = root / "db"
    pm.gdb_create(str(db), fastas, db_format="per-chromosome")
    pm.gdb_init(str(db))
    return db


@pytest.fixture(autouse=True)
def _config():
    saved = dict(pm.CONFIG)
    yield
    pm.CONFIG.clear()
    pm.CONFIG.update(saved)


@pytest.fixture()
def db(tmp_path):
    """chr1 (1000 bp) and chr2 (600 bp): 32 + 19 bins of 32 bp."""
    return _mk_db(tmp_path / "pqdb", {"chr1": 1000, "chr2": 600})


BIN_K = np.r_[np.arange(32), np.arange(19)]
BIN_CHROM_INDEX = np.repeat([0, 1], [32, 19])


def expected_values(value_col: str) -> np.ndarray:
    """The fixtures' closed form: a = k + 0.25, b = chrom_index * 1000 + k (k = start / 32)."""
    if value_col == "a":
        return BIN_K + 0.25
    return (BIN_CHROM_INDEX * 1000 + BIN_K).astype(np.float32).astype(np.float64)


def track_values(track: str) -> np.ndarray:
    return pm.gextract(track, pm.gintervals_all(), iterator=32)[track].to_numpy()


def assert_identical(got, expected, msg=""):
    np.testing.assert_array_equal(np.asarray(got, dtype=float), np.asarray(expected, dtype=float), err_msg=msg)


def staging_leftovers(db: Path) -> list[str]:
    """Hidden staging dirs left in the tracks dir (.trash.* entries are being removed already)."""
    return [e for e in os.listdir(db / "tracks") if ".tmp." in e and not e.startswith(".trash.")]


def expect_no_tracks_left(db: Path, tracks):
    """Nothing of tracks is left: not listed, no track dir, no staging dir."""
    assert not pm.gtrack_ls()  # None when there are none
    assert not any((db / "tracks" / f"{t}.track").exists() for t in tracks)
    assert staging_leftovers(db) == []


def local_fixture_dir(tmp_path: Path, fixtures: dict[str, str], name: str = "src") -> Path:
    """Copies fixtures (new name -> fixture) into a fresh directory."""
    src = tmp_path / name
    src.mkdir()
    for new, fx in fixtures.items():
        shutil.copy(pq_fixture(fx), src / new)
    return src


def test_imports_every_fixture_with_exact_values(db):
    fixtures = [
        "pandas_default", "plain_uncompressed", "gzip_v2", "mixed_rowgroups",
        "dict_fallback", "byte_stream_split", "unknown_fields", "shard_chr*",
    ]
    for fx in fixtures:
        prefix = f"t_{fx.replace('*', '')}_"
        created = pm.gtrack_import_parquet("test", pq_fixture(f"{fx}.parquet"), 32, track_prefix=prefix)
        assert created == [f"{prefix}a", f"{prefix}b"], fx
        for col in ("a", "b"):
            assert_identical(track_values(prefix + col), expected_values(col), f"{fx} {col}")


def test_nulls_become_nan(db):
    pm.gtrack_import_parquet("test", pq_fixture("nulls.parquet"), 32, track_prefix="n_")
    expected = expected_values("a")
    expected[BIN_K % 7 == 0] = np.nan
    assert_identical(track_values("n_a"), expected)
    assert_identical(track_values("n_b"), expected_values("b"))


def test_a_column_chunk_that_is_entirely_null_takes_defval(db):
    for fx in ("allnull_dict", "allnull_plain"):
        for defval in (np.nan, -7.0):
            prefix = f"{fx}_{'nan' if np.isnan(defval) else 'num'}_"
            pm.gtrack_import_parquet("test", pq_fixture(f"{fx}.parquet"), 32, track_prefix=prefix, defval=defval)
            expected = expected_values("a")
            expected[BIN_CHROM_INDEX == 1] = defval
            assert_identical(track_values(prefix + "a"), expected, prefix)
            assert_identical(track_values(prefix + "b"), expected_values("b"), prefix)


def test_more_values_than_the_row_group_has_is_refused_up_front(db):
    with pytest.raises(Exception, match="column chrom: .*number of values differs from the number of rows"):
        pm.gtrack_import_parquet("test", pq_fixture("huge_num_values.parquet"), 32, track_prefix="h_")
    expect_no_tracks_left(db, ["h_a"])


def test_a_definition_level_the_bit_width_cannot_hold_is_corrupt_data(db):
    with pytest.raises(
        Exception,
        match=r"bad_def_level.parquet, row group 0, column a: corrupt data \(RLE run value exceeds the bit width\)",
    ):
        pm.gtrack_import_parquet("test", pq_fixture("bad_def_level.parquet"), 32, track_prefix="d_")
    expect_no_tracks_left(db, ["d_a"])


def test_more_rows_than_the_genome_has_bases_is_refused_before_it_is_read(db):
    with pytest.raises(
        Exception, match="more_rows_than_bases.parquet, row group 0: 1601 rows, more than the 1600 bases of the genome"
    ):
        pm.gtrack_import_parquet("test", pq_fixture("more_rows_than_bases.parquet"), 32, track_prefix="m_")
    expect_no_tracks_left(db, ["m_a"])


def test_description_is_validated_up_front(db):
    with pytest.raises(ValueError, match="description"):
        pm.gtrack_import_parquet(None, pq_fixture("single.parquet"), 32)
    expect_no_tracks_left(db, ["a"])


def test_columns_selects_and_orders_tracks(db):
    created = pm.gtrack_import_parquet(
        "test", pq_fixture("pandas_default.parquet"), 32, track_prefix="s_", columns=["b", "a"]
    )
    assert created == ["s_b", "s_a"]
    assert_identical(track_values("s_b"), expected_values("b"))
    assert_identical(track_values("s_a"), expected_values("a"))

    created = pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="o_", columns="b")
    assert created == ["o_b"]
    assert not pm.gtrack_exists("o_a")


def test_columns_none_with_a_string_column_errors_and_creates_nothing(db):
    with pytest.raises(ValueError, match="columns"):
        pm.gtrack_import_parquet("test", pq_fixture("with_name.parquet"), 32, track_prefix="w_")
    assert not pm.gtrack_ls()  # None when there are none
    assert staging_leftovers(db) == []

    created = pm.gtrack_import_parquet(
        "test", pq_fixture("with_name.parquet"), 32, track_prefix="w_", columns=["a", "b"]
    )
    assert created == ["w_a", "w_b"]
    assert_identical(track_values("w_a"), expected_values("a"))


def _info_messages(caplog, fn):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pymisha"):
        fn()
    return [r.getMessage() for r in caplog.records if "chromosome name(s)" in r.getMessage()]


def test_aliases_resolve_unknown_chromosomes_are_reported(db, tmp_path, caplog):
    pm.gtrack_import_parquet("test", pq_fixture("alias.parquet"), 32, track_prefix="al_")
    assert_identical(track_values("al_a"), expected_values("a"))
    assert_identical(track_values("al_b"), expected_values("b"))

    pq_file = pq_fixture("unknown_chrom.parquet")
    pq_msgs = _info_messages(
        caplog, lambda: pm.gtrack_import_parquet("test", pq_file, 32, track_prefix="u_", columns="a")
    )
    assert len(pq_msgs) == 1 and "scaffold_9" in pq_msgs[0]

    # the message gtrack_import gives for the same rows in a TSV, but for the file name
    tsv = tmp_path / "rows.tsv"
    starts = np.arange(0, 1000, 32)
    pd.DataFrame({
        "chrom": ["chr1"] * 32 + ["scaffold_9"] * 3,
        "start": np.r_[starts, [0, 10, 20]],
        "end": np.r_[np.minimum(starts + 32, 1000), [10, 20, 30]],
        "value": np.r_[np.arange(32) + 0.25, [1, 1, 1]],
    }).to_csv(tsv, sep="\t", index=False)
    tsv_msgs = _info_messages(caplog, lambda: pm.gtrack_import("u_tsv", "test", str(tsv), 32))
    assert [m.replace(pq_file, "FILE") for m in pq_msgs] == [m.replace(str(tsv), "FILE") for m in tsv_msgs]

    # the message was raised after the import completed
    pm.gtrack_import_parquet("test", pq_file, 32, track_prefix="u2_")
    assert_identical(track_values("u2_a")[:32], expected_values("a")[:32])
    assert np.isnan(track_values("u2_a")[32:]).all()


def test_unsupported_files_error_and_create_nothing(db):
    cases = [
        ("zstd.parquet", 32, "ZSTD"),
        ("delta.parquet", 32, "DELTA_BINARY_PACKED"),
        ("overlap.parquet", 32, "overlap"),
        ("pandas_default.parquet", 0, "Parquet import creates dense tracks; binsize must be > 0"),
    ]
    for file, binsize, msg in cases:
        with pytest.raises(Exception, match=re.escape(msg)):
            pm.gtrack_import_parquet("test", pq_fixture(file), binsize, track_prefix="x_")
        assert not pm.gtrack_ls(), file
        assert not (db / "tracks" / "x_a.track").exists()
        assert not (db / "tracks" / "x_b.track").exists()
        assert staging_leftovers(db) == []
    with pytest.raises(ValueError, match="No files"):
        pm.gtrack_import_parquet("test", pq_fixture("no_such_*.parquet"), 32)


def _collect_warnings(fn):
    """The messages of the warnings fn() raises."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught]


def test_a_row_past_the_chromosome_end_is_clipped_with_a_warning_naming_the_row(db):
    # the last row of chr2 is (chr2, 576, 640); chr2 is 600 bp
    with pytest.warns(UserWarning, match=re.escape(
        "1 row(s) running past the end of their chromosome were clipped to it, and 0 row(s) starting at or past "
        "the chromosome end were dropped; the first is row 50 (row group 0) of "
    )):
        pm.gtrack_import_parquet("test", pq_fixture("beyond_end.parquet"), 32, track_prefix="x_")
    for value_col in ("a", "b"):
        assert_identical(track_values(f"x_{value_col}"), expected_values(value_col), value_col)


def test_rows_past_the_chromosome_end_are_clipped_or_dropped_with_one_warning_serial_and_in_kids(db, tmp_path):
    # 32 bp rows of a genome whose chromosomes are longer, written by gextract: 20 columns, so that
    # the import takes two batches of columns
    src = tmp_path / "src"
    long_db = _mk_db(src / "long", {"chr1": 1100, "chr2": 700})
    intervs = pm.giterator_intervals(intervals=pm.gintervals_all(), iterator=32)
    pm.gtrack_create_dense("v", "", intervs, intervs["start"] / 32 + 0.25, 32)
    f = str(src / "rows.parquet")
    pm.gextract([f"v + {k}" for k in range(1, 21)], pm.gintervals_all(), iterator=32,
                colnames=[f"c{k}" for k in range(1, 21)], file=f)
    assert long_db != db
    pm.gdb_init(str(db))

    # (chr1, 992, 1024) and (chr2, 576, 608) are clipped, the 3 rows of each after them dropped
    msg = ("2 row(s) running past the end of their chromosome were clipped to it, and 6 row(s) starting at or past "
           f"the chromosome end were dropped; the first is row 31 (row group 0) of {f}: (chr1, 992, 1024), "
           "chromosome size 1000")
    for multitask in (False, True):
        pm.CONFIG.update({"multitasking": multitask, "max_processes": 4, "min_processes": 4})
        prefix = "mt_" if multitask else "s_"
        warns = _collect_warnings(lambda prefix=prefix: pm.gtrack_import_parquet("test", f, 32, track_prefix=prefix))
        assert warns == [msg], prefix
        for k in (1, 20):
            assert_identical(track_values(f"{prefix}c{k}"), BIN_K + 0.25 + k, f"{prefix} {k}")


def test_a_value_near_the_float_limit_survives_binning(db, tmp_path):
    f = str(tmp_path / "big.parquet")
    pm.gtrack_create_dense("one", "", pm.gintervals_all(), [1.0, 1.0], 32)
    pm.gextract("one * 3.4e38", pm.gintervals_all(), iterator=32, colnames=["v"], file=f)
    pm.gtrack_import_parquet("test", f, 32, track_prefix="big_")
    # a float product value * 32 would overflow to Inf, which reads back as NaN
    assert_identical(track_values("big_v"), np.full(51, np.float64(np.float32(3.4e38))))


def test_field_metadata_holding_a_nul_byte_is_left_out_with_a_warning_and_the_track_imports(db):
    # column a's field metadata: model = "m1", nul = "a\0b" (make_fixtures.py)
    with pytest.warns(UserWarning, match=r'Field metadata "nul" of column "a" of .*nul_metadata.parquet was left out: '
                                         r"it holds a NUL byte"):
        pm.gtrack_import_parquet("test", pq_fixture("nul_metadata.parquet"), 32, track_prefix="n_")
    assert_identical(track_values("n_a"), expected_values("a"))
    assert pm.gtrack_attr_get("n_a", "model") == "m1"
    assert pm.gtrack_attr_get("n_a", "nul") == ""
    # only the metadata of the columns imported is read
    assert _collect_warnings(lambda: pm.gtrack_import_parquet(
        "test", pq_fixture("nul_metadata.parquet"), 32, track_prefix="b_", columns="b")) == []
    # gtrack_import of a one-column file
    with pytest.warns(UserWarning, match='Field metadata "nul" of column "a"'):
        pm.gtrack_import("one", "test", pq_fixture("nul_metadata_single.parquet"), 32)
    assert_identical(track_values("one"), expected_values("a"))
    assert pm.gtrack_attr_get("one", "model") == "m1"


def test_arguments_are_checked_before_anything_is_read(db):
    pq_path = pq_fixture("pandas_default.parquet")
    with pytest.raises(ValueError, match="path must be a single string"):
        pm.gtrack_import_parquet("test", [pq_path, pq_path], 32)
    for binsize in (np.inf, np.nan, None, -32, 1.5, 2**32, "32", [32, 64]):
        with pytest.raises(ValueError, match="binsize must be a positive integer"):
            pm.gtrack_import_parquet("test", pq_path, binsize)
    with pytest.raises(ValueError, match="defval must be a number"):
        pm.gtrack_import_parquet("test", pq_path, 32, defval="x")
    expect_no_tracks_left(db, ["a", "b"])


def test_integer_annotations_are_honoured_decimals_dates_timestamps_refused(db):
    pq_path = pq_fixture("annotated.parquet")
    info = dict(_pymisha.pm_parquet_columns(pq_path, pq_path))
    assert info == {
        "chrom": "string", "start": "int64", "end": "int64", "u32": "int32", "u64": "int64", "i8": "int32",
        "dec": "other", "ts": "other", "d": "other",
    }

    # columns=None: the decimal, timestamp and date columns are not numbers
    with pytest.raises(ValueError, match="dec, ts, d .*columns"):
        pm.gtrack_import_parquet("test", pq_path, 32, track_prefix="an_")
    expect_no_tracks_left(db, ["an_u32", "an_u64", "an_i8"])

    # unsigned values past the signed range; k = start / 32
    created = pm.gtrack_import_parquet("test", pq_path, 32, track_prefix="an_", columns=["u32", "u64", "i8"])
    assert created == ["an_u32", "an_u64", "an_i8"]
    assert_identical(track_values("an_u32"), 3e9 + BIN_K * 256)
    assert_identical(track_values("an_u64"), 2.0**63 + BIN_K * 2.0**40)
    assert_identical(track_values("an_i8"), -BIN_K.astype(float))

    for col, what in {"dec": "DECIMAL", "ts": "TIMESTAMP", "d": "DATE"}.items():
        with pytest.raises(Exception, match=what):
            pm.gtrack_import_parquet("test", pq_path, 32, track_prefix="bad_", columns=col)
        assert not pm.gtrack_exists(f"bad_{col}")
    assert staging_leftovers(db) == []


def test_floating_point_start_and_end_import_when_they_hold_whole_numbers(db):
    pm.gtrack_import_parquet("test", pq_fixture("float_coords.parquet"), 32, track_prefix="fc_")
    pm.gtrack_import_parquet("test", pq_fixture("single.parquet"), 32, track_prefix="ic_")
    assert_identical(track_values("fc_a"), track_values("ic_a"))
    assert_identical(track_values("fc_a"), expected_values("a"))

    with pytest.raises(
        Exception, match="float_coords_fraction.parquet, row group 0, column start: value 160.5 is not an integer"
    ):
        pm.gtrack_import_parquet("test", pq_fixture("float_coords_fraction.parquet"), 32, track_prefix="ff_")
    with pytest.raises(
        Exception, match="float_coords_nan.parquet, row group 0, column start: value -?nan is not an integer"
    ):
        pm.gtrack_import_parquet("test", pq_fixture("float_coords_nan.parquet"), 32, track_prefix="fn_")
    assert not pm.gtrack_exists("ff_a") and not pm.gtrack_exists("fn_a")
    assert staging_leftovers(db) == []


def test_a_chromosome_name_longer_than_1024_bytes_is_an_error(db):
    with pytest.raises(
        Exception,
        match="long_chrom.parquet, row group 0, column chrom: a string of 1025 bytes; "
        "strings longer than 1024 bytes are not supported",
    ):
        pm.gtrack_import_parquet("test", pq_fixture("long_chrom.parquet"), 32, track_prefix="lc_")
    expect_no_tracks_left(db, ["lc_a"])


def test_all_files_must_have_the_same_value_columns(db, tmp_path):
    src = local_fixture_dir(
        tmp_path, {"shard_1.parquet": "shard_chr1.parquet", "shard_2.parquet": "extra_column.parquet"}
    )
    with pytest.raises(ValueError, match="shard_2.parquet has value columns a, b, c"):
        pm.gtrack_import_parquet("test", str(src / "shard_*.parquet"), 32, track_prefix="x_")
    with pytest.raises(ValueError, match="shard_2.parquet has value columns a, b, c"):
        pm.gtrack_import_parquet("test", str(src / "shard_*.parquet"), 32, track_prefix="x_", columns=["a", "b"])
    expect_no_tracks_left(db, ["x_a", "x_b", "x_c"])


def test_error_messages_name_the_file_and_the_row_group(db):
    with pytest.raises(Exception, match="overlap.parquet, row group 0"):
        pm.gtrack_import_parquet("test", pq_fixture("overlap.parquet"), 32)
    with pytest.raises(Exception, match="zstd.parquet, row group 0, column chrom"):
        pm.gtrack_import_parquet("test", pq_fixture("zstd.parquet"), 32)


def test_existing_or_invalid_track_names_error_before_anything_is_written(db):
    pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="t_x_", columns="a")
    with pytest.raises(ValueError, match="already exists"):
        pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="t_x_")
    assert not pm.gtrack_exists("t_x_b")

    with pytest.raises(ValueError, match="does not exist"):
        pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="nodir.")
    assert pm.gtrack_ls() == ["t_x_a"]
    assert not (db / "tracks" / "nodir").exists()
    assert staging_leftovers(db) == []


def test_a_column_that_is_not_a_valid_track_name_is_an_error_never_renamed(db):
    with pytest.raises(ValueError) as e:
        pm.gtrack_import_parquet("test", pq_fixture("bad_name.parquet"), 32, track_prefix="bn_")
    assert '"x * 2"' in str(e.value)
    assert "columns =" in str(e.value)
    assert not pm.gtrack_ls()  # None when there are none
    assert staging_leftovers(db) == []

    created = pm.gtrack_import_parquet(
        "test", pq_fixture("bad_name.parquet"), 32, track_prefix="bn_", columns=["a", "b"]
    )
    assert created == ["bn_a", "bn_b"]
    assert_identical(track_values("bn_b"), expected_values("b"))


def test_all_or_nothing_when_the_second_file_fails(db, tmp_path):
    # a truncated file: refused before anything is staged
    src = local_fixture_dir(tmp_path, {"shard_1.parquet": "shard_chr1.parquet"}, "src1")
    data = Path(pq_fixture("shard_chr2.parquet")).read_bytes()
    (src / "shard_2.parquet").write_bytes(data[: len(data) // 2])
    with pytest.raises(Exception, match="is not a Parquet file, or is truncated"):
        pm.gtrack_import_parquet("test", str(src / "shard_*.parquet"), 32, track_prefix="x_")
    expect_no_tracks_left(db, ["x_a", "x_b"])

    # a corrupt page in the second file: fails inside the import, after chr1 was written
    src = local_fixture_dir(
        tmp_path, {"shard_1.parquet": "shard_chr1.parquet", "shard_2.parquet": "corrupt_page.parquet"}, "src2"
    )
    with pytest.raises(Exception, match="shard_2.parquet, row group 0, column a: corrupt"):
        pm.gtrack_import_parquet("test", str(src / "shard_*.parquet"), 32, track_prefix="x_")
    expect_no_tracks_left(db, ["x_a", "x_b"])


def test_all_or_nothing_when_a_track_fails_after_it_was_renamed_into_place(db, monkeypatch):
    write_attrs = _tracks._write_created_attrs_at_path

    def failing(track_dir, *args, **kwargs):
        if Path(track_dir).name == "x_b.track":
            raise RuntimeError("mocked attribute failure")
        return write_attrs(track_dir, *args, **kwargs)

    monkeypatch.setattr(_tracks, "_write_created_attrs_at_path", failing)
    with pytest.raises(RuntimeError, match="mocked attribute failure"):
        pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="x_")
    expect_no_tracks_left(db, ["x_a", "x_b"])


def test_a_track_the_cleanup_cannot_remove_is_reported(db, monkeypatch):
    def failing(*args, **kwargs):
        raise RuntimeError("mocked attribute failure")

    monkeypatch.setattr(_tracks, "_write_created_attrs_at_path", failing)
    monkeypatch.setattr(_tracks, "_gdb_trash", lambda *args, **kwargs: False)
    with pytest.warns(Warning, match="post-create cleanup left residue"), pytest.raises(
        RuntimeError, match="mocked attribute failure"
    ):
        pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="x_", columns="a")


def test_binning_equals_gtrack_import_of_the_same_rows(db, tmp_path):
    # rows that are not bins (the general path), with a gap, a NaN value and short rows, so
    # that some bins are only partly covered
    for defval in (np.nan, 0.1):
        prefix = "un_" if np.isnan(defval) else "und_"
        pm.gtrack_import_parquet("test", pq_fixture("unaligned.parquet"), 32, track_prefix=prefix, defval=defval)
        pm.gtrack_import(f"{prefix}tsv", "test", pq_fixture("unaligned.tsv"), 32, defval=defval)
        assert_identical(track_values(f"{prefix}value"), track_values(f"{prefix}tsv"), prefix)

    # rows that are exactly the bins (the fast path), with NaN values and a default value
    starts = np.r_[np.arange(0, 1000, 32), np.arange(0, 600, 32)]
    rows = pd.DataFrame({
        "chrom": ["chr1"] * 32 + ["chr2"] * 19,
        "start": starts,
        "end": np.r_[np.minimum(np.arange(32, 1025, 32), 1000), np.minimum(np.arange(32, 609, 32), 600)],
    })
    for i, defval in enumerate((np.nan, 0.1, 3.0), 1):
        for fx in ("pandas_default", "nulls"):
            prefix = f"f{i}_{fx}_"
            pm.gtrack_import_parquet(
                "test", pq_fixture(f"{fx}.parquet"), 32, track_prefix=prefix, columns="a", defval=defval
            )
            value = expected_values("a")
            if fx == "nulls":
                value[BIN_K % 7 == 0] = np.nan
            tsv = tmp_path / f"{prefix}.tsv"
            rows.assign(value=value).to_csv(tsv, sep="\t", index=False, na_rep="NaN")
            pm.gtrack_import(f"{prefix}tsv", "test", str(tsv), 32, defval=defval)
            assert_identical(track_values(f"{prefix}a"), track_values(f"{prefix}tsv"), prefix)


def test_many_chromosomes_per_row_group_serial_and_in_kids(tmp_path):
    _mk_db(tmp_path / "manydb", {f"chr{i}": 100 + 7 * i for i in range(1, 42)})
    pm.gtrack_import("tsv", "test", pq_fixture("many_chroms.tsv"), 32)
    expected = track_values("tsv")

    for fx in ("many_chroms", "many_chroms_shuffled"):
        for multitasking in (False, True):
            prefix = f"{fx}{'_par_' if multitasking else '_ser_'}"
            pm.CONFIG.update({"multitasking": multitasking, "max_processes": 4})
            pm.gtrack_import_parquet("test", pq_fixture(f"{fx}.parquet"), 32, track_prefix=prefix)
            assert_identical(track_values(f"{prefix}value"), expected, prefix)


def test_attributes_and_indexed_database(db):
    pm.gtrack_import_parquet("my description", pq_fixture("pandas_default.parquet"), 32, track_prefix="at_")
    assert pm.gtrack_attr_get("at_a", "description") == "my description"
    assert "gtrack.import_parquet" in pm.gtrack_attr_get("at_b", "created.by")

    pm.gdb_convert_to_indexed(force=True, verbose=False)
    pm.gtrack_import_parquet("test", pq_fixture("pandas_default.parquet"), 32, track_prefix="ix_")
    assert (db / "tracks" / "ix_a.track" / "track.idx").exists()
    assert_identical(track_values("ix_a"), expected_values("a"))
    assert_identical(track_values("ix_b"), expected_values("b"))


PQ6_EXPRS = [f"src * {i} + {i}" for i in range(1, 21)]
PQ6_COLS = [f"c{i}" for i in range(1, 21)]


@pytest.fixture()
def pq6_db(tmp_path):
    """Six chromosomes of 5000 bp, a dense track src binned at 50 bp. With max_processes = 4
    the import of 20 columns runs in kids, 6 x 2 units of work."""
    db = _mk_db(tmp_path / "pq6db", {f"chr{i}": 5000 for i in range(1, 7)})
    bins = pd.DataFrame({
        "chrom": np.repeat([f"chr{i}" for i in range(1, 7)], 100),
        "start": np.tile(np.arange(0, 5000, 50), 6),
    })
    bins["end"] = bins["start"] + 50
    pm.gtrack_create_dense("src", "src", bins, np.sin(np.arange(1, len(bins) + 1)) * 100, 50)
    return db


def test_parallel_import_equals_serial_import(pq6_db, tmp_path):
    pq_file = str(tmp_path / "x.parquet")
    pm.gextract(PQ6_EXPRS, pm.gintervals_all(), iterator=50, colnames=PQ6_COLS, file=pq_file)

    pm.CONFIG["multitasking"] = False
    pm.gtrack_import_parquet("test", pq_file, 50, track_prefix="ser_")
    pm.CONFIG.update({"multitasking": True, "max_processes": 4})
    t0 = os.times()
    pm.gtrack_import_parquet("test", pq_file, 50, track_prefix="par_")
    t1 = os.times()

    # kids were forked: their CPU time shows up once they are reaped
    assert (t1.children_user + t1.children_system) - (t0.children_user + t0.children_system) > 0
    for col in PQ6_COLS:
        ser = pm.gextract(f"ser_{col}", pm.gintervals_all(), iterator=50, colnames=["v"])
        par = pm.gextract(f"par_{col}", pm.gintervals_all(), iterator=50, colnames=["v"])
        pd.testing.assert_frame_equal(par, ser, obj=col)
    # and the values are those of the expressions, not NaN in both
    np.testing.assert_allclose(
        pm.gextract("par_c3", pm.gintervals_all(), iterator=50)["par_c3"].to_numpy(),
        pm.gextract("src * 3 + 3", pm.gintervals_all(), iterator=50)["src * 3 + 3"].to_numpy(),
        rtol=1e-6,
    )


def test_a_kid_error_removes_everything(pq6_db, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # written by one process, so that shard_1.parquet has one row group per chromosome
    pm.CONFIG["multitasking"] = False
    pm.gextract(PQ6_EXPRS, pm.gintervals_all(), iterator=50, colnames=PQ6_COLS, file=str(src / "shard_1.parquet"))
    # one row on chr5 again: the rows of chr5 overlap, the other chromosomes import fine
    pm.gextract(
        PQ6_EXPRS, pm.gintervals("chr5", 1000, 1050), iterator=50, colnames=PQ6_COLS,
        file=str(src / "shard_2.parquet"),
    )

    pm.CONFIG.update({"multitasking": True, "max_processes": 4})
    with pytest.raises(
        Exception, match="shard_1.parquet, row group 4 and file .*shard_2.parquet, row group 0: rows overlap"
    ):
        pm.gtrack_import_parquet("test", str(src / "shard_*.parquet"), 50, track_prefix="k_")
    assert pm.gtrack_ls() == ["src"]
    assert not any((pq6_db / "tracks" / f"k_{c}.track").exists() for c in PQ6_COLS)
    assert staging_leftovers(pq6_db) == []


def test_gtrack_import_reads_a_one_column_parquet_file(db):
    pm.gtrack_import("pq_one", "test", pq_fixture("single.parquet"), binsize=32)
    assert_identical(track_values("pq_one"), expected_values("a"))
    assert pm.gtrack_attr_get("pq_one", "created.by")


def test_gtrack_import_refuses_a_multi_column_parquet_file(db):
    with pytest.raises(ValueError, match="gtrack_import_parquet"):
        pm.gtrack_import("pq_two", "test", pq_fixture("pandas_default.parquet"), binsize=32)
    expect_no_tracks_left(db, ["pq_two"])


def test_gtrack_import_needs_binsize_above_0_for_parquet(db):
    with pytest.raises(ValueError, match="dense tracks"):
        pm.gtrack_import("pq_one", "test", pq_fixture("single.parquet"), binsize=0)
    with pytest.raises(ValueError, match="dense tracks"):
        pm.gtrack_import("pq_one", "test", pq_fixture("single.parquet"))


def test_a_parquet_name_is_not_needed(db, tmp_path):
    src = local_fixture_dir(tmp_path, {"x.dat": "single.parquet"})
    pm.gtrack_import("pq_dat", "test", str(src / "x.dat"), binsize=32)
    assert_identical(track_values("pq_dat"), expected_values("a"))


def test_messages_about_a_gzipped_parquet_file_name_it(db, tmp_path, caplog):
    def gzip_copy(src, dst):
        with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return str(dst)

    # an error of the reader (decoding a column) and one of the import (before decoding)
    gz = gzip_copy(pq_fixture("bad_def_level.parquet"), tmp_path / "bad_def_level.parquet.gz")
    with pytest.raises(Exception, match=re.escape(f"File {gz}, row group 0, column a: corrupt data")):
        pm.gtrack_import("g1", "test", gz, binsize=32)
    gz = gzip_copy(pq_fixture("more_rows_than_bases.parquet"), tmp_path / "more_rows_than_bases.parquet.gz")
    with pytest.raises(Exception, match=re.escape(f"File {gz}, row group 0: 1601 rows")):
        pm.gtrack_import("g2", "test", gz, binsize=32)
    expect_no_tracks_left(db, ["g1", "g2"])

    # the unknown chromosome report: a file of another database, with a chromosome this one lacks
    _mk_db(tmp_path / "otherdb", {"chr1": 1000, "scaffold_9": 100})
    pq_file = tmp_path / "x.parquet"
    # a track of 1s: pymisha's gextract does not evaluate the constant expression "1" R misha writes
    pm.gtrack_create_dense("ones", "ones", pm.gintervals_all(), [1.0, 1.0], 32)
    pm.gextract("ones", pm.gintervals_all(), iterator=32, colnames=["a"], file=str(pq_file))
    pm.gdb_init(str(db))
    gz = gzip_copy(pq_file, tmp_path / "x.parquet.gz")
    msgs = _info_messages(caplog, lambda: pm.gtrack_import("g3", "test", gz, binsize=32))
    assert any(f"chromosome name(s) in {gz} do not exist" in m for m in msgs), msgs
    assert_identical(track_values("g3")[:32], np.ones(32))


def test_a_parquet_input_with_no_rows_is_an_error_naming_the_file(db, tmp_path):
    empty = pq_fixture("no_rowgroups.parquet")
    with pytest.raises(Exception, match=re.escape(f"{empty} has no rows")):
        pm.gtrack_import_parquet("test", empty, 32, track_prefix="e_")
    expect_no_tracks_left(db, ["e_a"])
    with pytest.raises(Exception, match=re.escape(f"{empty} has no rows")):
        pm.gtrack_import("e_one", "test", empty, binsize=32)
    expect_no_tracks_left(db, ["e_one"])

    # misha's own output for an empty scope: one row group of zero rows
    own = str(tmp_path / "own.parquet")
    pm.gtrack_create_sparse("dense_src", "src", pm.gintervals("chr1", 0, 10), [1.0])
    pm.gextract("dense_src", pm.gintervals("chr1", 0, 100), iterator=pm.gintervals("chr2", 0, 100), file=own)
    with pytest.raises(Exception, match=re.escape(f"{own} has no rows")):
        pm.gtrack_import_parquet("test", own, 32, track_prefix="o_")
    assert staging_leftovers(db) == []
    assert pm.gtrack_ls() == ["dense_src"]


def test_an_empty_shard_among_non_empty_shards_imports(db, tmp_path):
    src = local_fixture_dir(
        tmp_path, {"shard_1.parquet": "no_rowgroups.parquet", "shard_2.parquet": "single.parquet"}, "src1"
    )
    pm.gtrack_import_parquet("test", str(src / "shard_*.parquet"), 32, track_prefix="x_")
    assert_identical(track_values("x_a"), expected_values("a"))
    # all shards empty: an error naming all of them
    src = local_fixture_dir(
        tmp_path, {"s1.parquet": "no_rowgroups.parquet", "s2.parquet": "no_rowgroups.parquet"}, "src2"
    )
    with pytest.raises(Exception, match="s1.parquet, .*s2.parquet have no rows"):
        pm.gtrack_import_parquet("test", str(src / "s*.parquet"), 32, track_prefix="y_")
    assert staging_leftovers(db) == []


def test_a_directory_as_path_suggests_the_glob(db, tmp_path):
    src = local_fixture_dir(tmp_path, {"a.parquet": "single.parquet"})
    msg = f'is a directory; to import the Parquet files in it use path = "{src / "*.parquet"}"'
    with pytest.raises(ValueError, match=re.escape(msg)):
        pm.gtrack_import_parquet("test", str(src), 32, track_prefix="d_")
    expect_no_tracks_left(db, ["d_a"])
    with pytest.raises(ValueError, match="No files match"):
        pm.gtrack_import_parquet("test", str(src / "nothing"), 32)


def test_empty_shards_written_by_pyarrow_and_pandas(db, tmp_path):  # R test-review-2026-11-followup.R :73
    # (make_fixtures.py: one row group of no rows, column chunks with data_page_offset 0;
    # pandas also types the chrom column null)
    pm.gtrack_import_parquet("", pq_fixture("shard_chr1.parquet"), 32, track_prefix="one_")
    for empty in ("empty_pyarrow.parquet", "empty_pandas.parquet"):
        src = local_fixture_dir(tmp_path, {"shard_chr1.parquet": "shard_chr1.parquet", empty: empty}, empty)
        prefix = empty.removesuffix(".parquet") + "_"
        pm.gtrack_import_parquet("", str(src / "*.parquet"), 32, track_prefix=prefix)
        assert_identical(track_values(prefix + "a"), track_values("one_a"))
    assert pm.gextract("one_a", pm.gintervals("chr1", 0, 64))["one_a"].tolist() == [0.25, 1.25]
