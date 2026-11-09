"""gextract(file="*.parquet"), ported from R misha's test-gextract-parquet.R.

The files are checked against pyarrow, the reader they are written for; those tests are
skipped without it, as R misha skips them. R misha writes the parts of a multitasking
extraction in its kids and merges them; pymisha writes the file in one process from the
(possibly multitasked) result, so the tests of the parts check here that the result and the
temporary files are the same with multitasking on and off.
"""

from __future__ import annotations

import io
import os
import re
import socket
import subprocess
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest

import pymisha as pm

pa = None
pq = None


def _pyarrow():
    global pa, pq
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")


def _mk_db(root: Path, sizes: dict[str, int]) -> Path:
    """A database of the given chromosomes under root, opened."""
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


def _bins_of(chrom, size, binsize):
    start = np.arange(0, size, binsize)
    return pd.DataFrame({"chrom": chrom, "start": start, "end": np.minimum(start + binsize, size)})


@pytest.fixture()
def pq_track_db(tmp_path):
    """chr1 = 1000 bp, chr2 = 600 bp; dense track pq (binsize 32) = start / 32 + 0.25, + 1000 on chr2."""
    db = _mk_db(tmp_path / "pqdb", {"chr1": 1000, "chr2": 600})
    iv = pd.concat([_bins_of("chr1", 1000, 32), _bins_of("chr2", 600, 32)], ignore_index=True)
    pm.gtrack_create_dense("pq", "pq", iv, iv["start"] / 32 + 0.25 + np.where(iv["chrom"] == "chr2", 1000, 0), 32)
    return db


def _six_chrom_db(tmp_path, sizes=(200000,) * 6):
    """chr1..chr6; dense track six (binsize 50) = start / 50 + chrom index / 8."""
    chroms = [f"chr{i}" for i in range(1, 7)]
    db = _mk_db(tmp_path / "sixdb", dict(zip(chroms, sizes, strict=True)))
    iv = pd.concat([_bins_of(c, s, 50) for c, s in zip(chroms, sizes, strict=True)], ignore_index=True)
    pm.gtrack_create_dense("six", "six", iv, iv["start"] / 50 + (iv["chrom"].map(chroms.index) + 1) / 8, 50)
    pm.CONFIG.update({"max_processes": 4, "min_processes": 4, "min_scope4process": 1000, "min_intervs4process": 0})
    return db


def _read_pq(path, coord_type="int32"):
    """The table, the arrow schema (with the field metadata of the columns that have some),
    per-row-group metadata and how the file compares with pyarrow's own write of the same table
    (coord_type: its start and end; the values double)."""
    _pyarrow()
    f = pq.ParquetFile(path)
    md = f.metadata
    t = f.read()
    coord = pa.int64() if coord_type == "int64" else pa.int32()
    ref_schema = pa.schema(
        [("chrom", pa.string()), ("start", coord), ("end", coord)] + [(n, pa.float64()) for n in t.column_names[3:]]
    )
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pydict({n: t.column(n).to_pylist() for n in t.column_names}, schema=ref_schema), buf)
    ref = pq.read_table(io.BytesIO(buf.getvalue()))
    try:
        pa.concat_tables([t, ref])
        concat_ok = True
    except Exception:
        concat_ok = False
    row_groups = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        cols = {}
        for j in range(rg.num_columns):
            c = rg.column(j)
            col = {"encodings": list(c.encodings), "compression": c.compression}
            s = c.statistics
            if s is not None and s.has_min_max:
                col["min"], col["max"] = s.min, s.max
            if s is not None and s.has_null_count:
                col["null_count"] = s.null_count
            cols[c.path_in_schema] = col
        row_groups.append({"num_rows": rg.num_rows, "columns": cols})
    return {
        "table": t,
        "names": [x.name for x in f.schema_arrow],
        "types": [str(x.type) for x in f.schema_arrow],
        "nullable": [x.nullable for x in f.schema_arrow],
        "row_groups": row_groups,
        "iter_batches_rows": sum(b.num_rows for b in f.iter_batches()),
        "read_table_rows": pq.read_table(path).num_rows,
        "schema_equals_reference": t.schema.equals(ref.schema),
        "concat_ok": concat_ok,
        "field_metadata": {x.name: {k.decode(): v.decode() for k, v in x.metadata.items()}
                           for x in f.schema_arrow if x.metadata},
    }


def _expect_arrow_like(p):
    """Every column nullable and every column chunk GZIP, as arrow writes them."""
    assert all(p["nullable"])
    assert p["schema_equals_reference"]
    assert p["concat_ok"]
    codecs = [c["compression"] for rg in p["row_groups"] for c in rg["columns"].values()]
    assert codecs and all(c == "GZIP" for c in codecs), set(codecs)


def _expect_same_rows(table, inmem, value_cols):
    assert table.column("chrom").to_pylist() == inmem["chrom"].astype(str).tolist()
    np.testing.assert_array_equal(table.column("start").to_numpy(), inmem["start"].to_numpy())
    np.testing.assert_array_equal(table.column("end").to_numpy(), inmem["end"].to_numpy())
    for k in value_cols:
        got = table.column(k).to_numpy(zero_copy_only=False)
        assert got.dtype == np.float64, k
        np.testing.assert_array_equal(got, inmem[k].to_numpy(dtype=np.float64), err_msg=k)


def _dir_entries(d):
    return sorted(os.listdir(d))


def _host_tag():
    # R misha's .gdb.host_tag(): the node name up to the first dot, [^A-Za-z0-9_] -> "_"
    return re.sub(r"[^A-Za-z0-9_]", "_", socket.gethostname().split(".")[0]) or "unknown"


def test_parquet_output_equals_in_memory_gextract(pq_track_db, tmp_path):
    f = str(tmp_path / "x.parquet")
    exprs = ["pq", "pq * 2", "pq + 0.5"]
    cols = ["a", "a.b c", "test_pq * 2"]

    inmem = pm.gextract(exprs, pm.gintervals_all(), iterator=32, colnames=cols)
    assert pm.gextract(exprs, pm.gintervals_all(), iterator=32, colnames=cols, file=f) is None
    p = _read_pq(f)

    assert p["names"] == ["chrom", "start", "end", *cols]
    assert p["types"] == ["string", "int32", "int32", "double", "double", "double"]
    _expect_arrow_like(p)
    assert p["table"].num_rows == 32 + 19
    _expect_same_rows(p["table"], inmem, cols)

    assert len(p["row_groups"]) == 2
    for rg, chrom in zip(p["row_groups"], ["chr1", "chr2"], strict=True):
        st = rg["columns"]["chrom"]
        assert st["min"] == chrom and st["max"] == chrom and st["null_count"] == 0
        start_st = rg["columns"]["start"]
        assert start_st["min"] == inmem.loc[inmem["chrom"] == chrom, "start"].min()
        assert start_st["max"] == inmem.loc[inmem["chrom"] == chrom, "start"].max()
        assert "min" not in rg["columns"]["a"]

    # the extension is matched case-insensitively
    f2 = tmp_path / "X.PARQUET"
    pm.gextract("pq", pm.gintervals_all(), iterator=32, file=str(f2))
    assert f2.read_bytes()[:4] == b"PAR1"

    # created_by is "<application> version <version>", the form parquet-mr parses
    assert f"pymisha version {pm.__version__}".encode() in Path(f).read_bytes()
    assert pq.ParquetFile(f).metadata.created_by == f"pymisha version {pm.__version__}"


def test_many_columns_values_kept_as_doubles(pq_track_db, tmp_path):
    # Thrift lists of 15 elements or more; pq / 3 and the like are not float32 values, so this
    # checks that they are not rounded to float
    f = str(tmp_path / "x.parquet")
    exprs = [f"pq / {i}" for i in range(1, 21)]

    inmem = pm.gextract(exprs, pm.gintervals_all(), iterator=32)
    pm.gextract(exprs, pm.gintervals_all(), iterator=32, file=f)
    p = _read_pq(f)

    assert p["names"] == ["chrom", "start", "end", *exprs]
    _expect_arrow_like(p)
    v = inmem["pq / 3"].to_numpy()
    assert not np.array_equal(v.astype(np.float32).astype(np.float64), v)
    _expect_same_rows(p["table"], inmem, exprs)


def test_coordinates_past_2_24_are_written_exactly_position_vtracks(tmp_path):
    _big_chrom_db(tmp_path, {"chr1": 200000000})
    pos = 190000000 + np.arange(1, 2000, 7)
    rng = np.random.default_rng(60427)
    pm.gtrack_create_sparse("sp", "sp", pm.gintervals("chr1", pos, pos + 1), rng.random(len(pos)))
    pm.gvtrack_create("mp", "sp", func="max.pos.abs")
    try:
        f = str(tmp_path / "x.parquet")
        scope = pm.gintervals("chr1", 190000000, 190002000)

        inmem = pm.gextract("mp", scope, iterator=100)
        pm.gextract("mp", scope, iterator=100, file=f)
        p = _read_pq(f)
    finally:
        pm.gvtrack_rm("mp")

    # float32 holds these only to a multiple of 16
    v = inmem["mp"].to_numpy()
    assert not np.array_equal(v.astype(np.float32).astype(np.float64), v)
    _expect_same_rows(p["table"], inmem, ["mp"])


def _track_attrs(track):
    """The attributes of a track that have a value, sorted by name."""
    return dict(sorted((k, v) for k, v in pm.gtrack_attr_export(track).iloc[0].items() if isinstance(v, str) and v))


def test_track_attributes_travel_as_field_metadata_and_import_restores_them(pq_track_db, tmp_path):
    from pymisha import tracks as _tracks

    model = "mod\u00e8le \u03b1"
    pm.gtrack_attr_set("pq", "value.scale", "log2(x+1)")
    pm.gtrack_attr_set("pq", "model", model)
    # a track without any attribute
    pm.gtrack_create_dense("bare", "", pm.gintervals_all(), [1.0, 2.0], 32)
    _tracks._save_track_attributes("bare", {})
    assert _track_attrs("bare") == {}
    f = str(tmp_path / "x.parquet")

    pm.gextract(["pq", "bare", "pq * 2"], pm.gintervals_all(), iterator=32, colnames=["pq", "bare", "twice"], file=f)
    p = _read_pq(f)

    # only the track column has field metadata: the attributes of the track
    assert list(p["field_metadata"]) == ["pq"]
    assert dict(sorted(p["field_metadata"]["pq"].items())) == _track_attrs("pq")
    assert p["field_metadata"]["pq"]["model"] == model
    _expect_arrow_like(p)

    # an empty description: the track's description comes from the file
    pm.gtrack_import_parquet("", f, 32, track_prefix="back_")
    back = _track_attrs("back_pq")
    assert back["value.scale"] == "log2(x+1)"
    assert back["model"] == model
    assert back["description"] == "pq"
    assert "gtrack.import_parquet" in back["created.by"]
    # type and binsize describe how pq is stored (gtrack_create_dense sets them): not carried over
    assert {"type", "binsize"} <= set(_track_attrs("pq"))
    assert not {"type", "binsize"} & set(back)
    assert sorted(back) == sorted(set(_track_attrs("pq")) - {"type", "binsize"})
    for track in ("back_bare", "back_twice"):
        assert sorted(_track_attrs(track)) == ["created.by", "created.date", "created.user"], track

    # a description given wins
    pm.gtrack_import_parquet("mine", f, 32, track_prefix="given_", columns="pq")
    assert _track_attrs("given_pq")["description"] == "mine"
    assert _track_attrs("given_pq")["value.scale"] == "log2(x+1)"


def test_field_metadata_survives_a_multitask_extraction(tmp_path):
    # R misha merges the parts its kids write; pymisha writes the multitasked result in one process
    _six_chrom_db(tmp_path, (1300000,) * 6)
    pm.gtrack_attr_set("six", "heldout.chroms", "chr3,chr5")
    f = str(tmp_path / "x.parquet")
    pm.gextract("six", pm.gintervals_all(), iterator=50, file=f)
    pm.gtrack_import_parquet("", f, 50, track_prefix="back_")
    assert pm.gtrack_attr_get("back_six", "heldout.chroms") == "chr3,chr5"
    assert pm.gtrack_attr_get("back_six", "description") == "six"


def test_gtrack_import_of_a_one_column_parquet_file_restores_the_attributes_its_own_attrs_first(
        pq_track_db, tmp_path):
    pm.gtrack_attr_set("pq", "value.scale", "linear")
    pm.gtrack_attr_set("pq", "checkpoint", "epoch 7")
    f = str(tmp_path / "x.parquet")
    pm.gextract("pq", pm.gintervals_all(), iterator=32, file=f)

    pm.gtrack_import("one", "", f, 64, attrs={"checkpoint": "mine"})
    assert pm.gtrack_attr_get("one", "value.scale") == "linear"
    assert pm.gtrack_attr_get("one", "checkpoint") == "mine"
    assert pm.gtrack_attr_get("one", "description") == "pq"
    assert "gtrack.import" in pm.gtrack_attr_get("one", "created.by")
    # pq's binsize is 32; the track imported at 64 does not carry it
    assert pm.gtrack_attr_get("pq", "binsize") == "32"
    assert pm.gtrack_attr_get("one", "binsize") == ""


def test_empty_result_is_one_empty_row_group(pq_track_db, tmp_path):
    # an iterator that does not meet the scope yields no rows
    f = str(tmp_path / "x.parquet")
    scope = pm.gintervals("chr1", 0, 100)
    it = pm.gintervals("chr2", 0, 100)

    assert pm.gextract(["pq", "pq * 2"], scope, iterator=it) is None
    pm.gextract(["pq", "pq * 2"], scope, iterator=it, file=f)
    p = _read_pq(f)

    assert p["names"] == ["chrom", "start", "end", "pq", "pq * 2"]
    assert p["types"] == ["string", "int32", "int32", "double", "double"]
    _expect_arrow_like(p)
    assert p["table"].num_rows == 0
    assert p["iter_batches_rows"] == 0
    assert p["read_table_rows"] == 0
    assert [rg["num_rows"] for rg in p["row_groups"]] == [0]


def test_row_groups_split_at_the_cap_and_at_chromosome_changes(tmp_path):
    _mk_db(tmp_path / "capdb", {"chr1": 1100000, "chr2": 1000})
    pm.CONFIG["multitasking"] = False
    iv = pd.concat([_bins_of("chr1", 1100000, 1), _bins_of("chr2", 1000, 1)], ignore_index=True)
    pm.gtrack_create_dense("pos", "pos", iv, iv["start"].astype(float), 1)
    f = str(tmp_path / "x.parquet")
    exprs = ["pos", "pos * 0"]

    inmem = pm.gextract(exprs, pm.gintervals_all(), iterator=1)
    pm.gextract(exprs, pm.gintervals_all(), iterator=1, file=f)
    p = _read_pq(f)

    assert [rg["num_rows"] for rg in p["row_groups"]] == [1048576, 51424, 1000]
    # a full row group: one data page per column chunk, its definition levels one RLE run
    _expect_arrow_like(p)
    assert "RLE_DICTIONARY" not in p["row_groups"][0]["columns"]["pos"]["encodings"]
    for rg in p["row_groups"]:
        assert "RLE_DICTIONARY" in rg["columns"]["pos * 0"]["encodings"]
    _expect_same_rows(p["table"], inmem, exprs)


def test_nan_column_stays_dictionary_encoded(tmp_path):
    _mk_db(tmp_path / "nandb", {"chr1": 1100000, "chr2": 1000})
    iv = pd.concat([_bins_of("chr1", 1100000, 1000), _bins_of("chr2", 1000, 1000)], ignore_index=True)
    iv["end"] = iv["start"] + 1
    pm.gtrack_create_dense("sparse_vals", "sparse", iv, iv["start"].astype(float), 1, defval=np.nan)
    f = str(tmp_path / "x.parquet")

    inmem = pm.gextract("sparse_vals", pm.gintervals_all(), iterator=1)
    pm.gextract("sparse_vals", pm.gintervals_all(), iterator=1, file=f)
    p = _read_pq(f)

    for rg in p["row_groups"]:
        assert "RLE_DICTIONARY" in rg["columns"]["sparse_vals"]["encodings"]
    got = p["table"].column("sparse_vals").to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(np.isnan(got), inmem["sparse_vals"].isna().to_numpy())
    assert inmem["sparse_vals"].notna().sum() > 1000
    _expect_same_rows(p["table"], inmem, ["sparse_vals"])


def _big_chrom_db(tmp_path, sizes):
    db = _mk_db(tmp_path / "bigdb", dict.fromkeys(sizes, 10))
    # the sizes misha reads; no sequence of that length is needed for this
    (db / "chrom_sizes.txt").write_text("".join(f"{c}\t{s}\n" for c, s in sizes.items()))
    pm.gdb_init(str(db))
    return db


def test_a_chromosome_of_2_31_bp_or_more_gets_int64_coordinates(tmp_path):
    _big_chrom_db(tmp_path, {"chr1": 2147483648})
    pm.gtrack_create_dense("big", "big", pm.gintervals("chr1", 2147480000, 2147483648), [5.0], 1000000)
    f = str(tmp_path / "x.parquet")
    scope = pm.gintervals("chr1", 2140000000, 2147483648)

    inmem = pm.gextract("big", scope, iterator=1000000)
    pm.gextract("big", scope, iterator=1000000, file=f)
    p = _read_pq(f, "int64")

    assert p["types"] == ["string", "int64", "int64", "double"]
    _expect_arrow_like(p)
    assert max(p["table"].column("end").to_pylist()) == 2147483648
    assert p["row_groups"][0]["columns"]["end"]["max"] == 2147483648
    _expect_same_rows(p["table"], inmem, ["big"])


# fails when an evaluation buffer holds a value of chr2 (>= 1010); pq otherwise
_FAIL_ON_CHR2 = "pq + np.take(np.zeros(1), np.int64(pq >= 1010))"


def test_errors_leave_no_file_behind(pq_track_db, tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    f = str(d / "x.parquet")

    # (a) duplicate column names, (b) a value column named like a coordinate column
    with pytest.raises(Exception, match='two columns named "v"'):
        pm.gextract(["pq", "pq * 2"], pm.gintervals_all(), iterator=32, colnames=["v", "v"], file=f)
    assert not os.path.exists(f)
    with pytest.raises(Exception, match='value column named "start"'):
        pm.gextract("pq", pm.gintervals_all(), iterator=32, colnames=["start"], file=f)
    assert not os.path.exists(f)

    # (c) an error in the middle of the scan, after chr1 was evaluated
    pm.CONFIG.update({"multitasking": False, "eval_buf_size": 8})
    with pytest.raises(Exception, match="out of bounds"):
        pm.gextract(_FAIL_ON_CHR2, pm.gintervals_all(), iterator=32, file=f)
    assert _dir_entries(d) == []

    # (d) the same error does not touch an existing target
    Path(f).write_text("existing content\n")
    before = Path(f).read_bytes()
    with pytest.raises(Exception, match="out of bounds"):
        pm.gextract(_FAIL_ON_CHR2, pm.gintervals_all(), iterator=32, file=f)
    assert Path(f).read_bytes() == before
    assert _dir_entries(d) == ["x.parquet"]


def test_tsv_output_unchanged(pq_track_db, tmp_path):
    f = tmp_path / "x.tsv"
    pm.gextract(["pq", "pq * 2"], pm.gintervals_all(), iterator=32, colnames=["a", "b"], file=str(f))
    lines = f.read_text().splitlines()
    assert lines[:2] == ["chrom\tstart\tend\ta\tb", "chr1\t0\t32\t0.25\t0.5"]
    assert len(lines) == 1 + 32 + 19


def test_multitask_output_equals_serial_output(tmp_path):
    _six_chrom_db(tmp_path, (1300000,) * 6)
    d = tmp_path / "out"
    d.mkdir()
    f = str(d / "x.parquet")
    exprs = ["six", "six * 2", "six + 0.5"]
    cols = ["a", "b", "c"]

    for it in (None, 50):
        pm.CONFIG["multitasking"] = False
        serial = pm.gextract(exprs, pm.gintervals_all(), iterator=it, colnames=cols)
        pm.CONFIG["multitasking"] = True
        inmem = pm.gextract(exprs, pm.gintervals_all(), iterator=it, colnames=cols)
        pd.testing.assert_frame_equal(inmem, serial)
        pm.gextract(exprs, pm.gintervals_all(), iterator=it, colnames=cols, file=f)
        p = _read_pq(f)

        assert p["table"].num_rows == 6 * 26000
        _expect_arrow_like(p)
        _expect_same_rows(p["table"], inmem, cols)
        # one chromosome per row group: the file is written in one process
        rg_chroms = [rg["columns"]["chrom"]["min"] for rg in p["row_groups"]]
        assert rg_chroms == [rg["columns"]["chrom"]["max"] for rg in p["row_groups"]]
        assert rg_chroms == [f"chr{i}" for i in range(1, 7)]
        assert _dir_entries(d) == ["x.parquet"]

    # misha reads it back, as the float values a track holds
    pm.gtrack_import_parquet("back", f, 50, track_prefix="back_")
    for k in cols:
        got = pm.gextract(f"back_{k}", pm.gintervals_all(), iterator=50)[f"back_{k}"].to_numpy()
        np.testing.assert_array_equal(got, inmem[k].to_numpy().astype(np.float32).astype(np.float64), err_msg=k)


def test_a_kid_with_no_rows_still_merges(tmp_path):
    _six_chrom_db(tmp_path, tuple(s * 100000 for s in (2, 2, 4, 2, 2, 2)))
    # no intervals on chr3
    it = pm.gintervals([f"chr{i}" for i in (1, 2, 4, 5, 6)], 1000, 1100)
    pm.gtrack_create_sparse("sp", "sp", it, np.arange(1, len(it) + 1, dtype=float))
    f = str(tmp_path / "x.parquet")

    inmem = pm.gextract(["six", "sp"], pm.gintervals_all(), iterator="sp")
    pm.gextract(["six", "sp"], pm.gintervals_all(), iterator="sp", file=f)
    p = _read_pq(f)

    _expect_same_rows(p["table"], inmem, ["six", "sp"])
    _expect_arrow_like(p)
    assert [rg["num_rows"] for rg in p["row_groups"]] == [1] * 5


# fails on chr5 (fraction 5 / 8); six otherwise
_FAIL_ON_CHR5 = "six + np.take(np.zeros(1), np.int64(six % 1 == 0.625))"


def test_a_failing_kid_leaves_nothing_behind(tmp_path):
    _six_chrom_db(tmp_path)
    d = tmp_path / "out"
    d.mkdir()
    f = str(d / "x.parquet")

    with pytest.raises(Exception, match="out of bounds"):
        pm.gextract(_FAIL_ON_CHR5, pm.gintervals_all(), file=f)
    assert _dir_entries(d) == []

    # an existing target is left as it was
    Path(f).write_text("existing content\n")
    with pytest.raises(Exception, match="out of bounds"):
        pm.gextract(_FAIL_ON_CHR5, pm.gintervals_all(), file=f)
    assert Path(f).read_text() == "existing content\n"
    assert _dir_entries(d) == ["x.parquet"]


def test_a_target_in_a_missing_directory_names_the_temporary_files_directory(tmp_path):
    _six_chrom_db(tmp_path)
    f = str(tmp_path / "nodir" / "x.parquet")
    msg = f"Failed to write {f}: cannot create a temporary file in {os.path.dirname(f)}: No such file or directory"
    for multitask in (False, True):
        pm.CONFIG["multitasking"] = multitask
        with pytest.raises(Exception, match=re.escape(msg)):
            pm.gextract("six", pm.gintervals_all(), file=f)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into any directory")
def test_a_writable_target_in_a_directory_that_is_not_writable(tmp_path):
    _six_chrom_db(tmp_path)
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    f = ro_dir / "x.parquet"
    f.touch()
    ro_dir.chmod(0o555)
    try:
        msg = f"Failed to write {f}: cannot create a temporary file in {ro_dir}: Permission denied"
        for multitask in (False, True):
            pm.CONFIG["multitasking"] = multitask
            with pytest.raises(Exception, match=re.escape(msg)):
                pm.gextract("six", pm.gintervals_all(), file=str(f))
    finally:
        ro_dir.chmod(0o755)
    assert f.stat().st_size == 0


def test_a_replaced_target_keeps_its_permission_bits(tmp_path):
    _six_chrom_db(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    for multitask in (False, True):
        f = out / f"x{multitask}.parquet"
        f.touch()
        os.chmod(f, 0o604)  # no umask turns 0666 into this
        pm.CONFIG["multitasking"] = multitask
        pm.gextract("six", pm.gintervals_all(), file=str(f))
        assert f.stat().st_mode & 0o777 == 0o604, multitask
        assert f.stat().st_size > 0
    # a new target gets the permissions a new text file gets (misha's own umask, not the caller's)
    old = os.umask(0o022)
    try:
        f = out / "new.parquet"
        pm.gextract("six", pm.gintervals_all(), file=str(f))
        tsv = out / "new.tsv"
        pm.gextract("six", pm.gintervals_all(), file=str(tsv))
    finally:
        os.umask(old)
    assert f.stat().st_mode & 0o777 == tsv.stat().st_mode & 0o777 == 0o660


def test_int64_coordinates_when_one_chromosome_of_the_scope_needs_them(tmp_path):
    _big_chrom_db(tmp_path, {"chr1": 2147483648, "chr2": 10000})
    pm.gtrack_create_dense(
        "big", "big", pm.gintervals(["chr1", "chr2"], [2147480000, 0], [2147483648, 10000]), [5.0, 7.0], 1000000
    )
    pm.CONFIG.update({"max_processes": 4, "min_processes": 4, "min_scope4process": 1000, "min_intervs4process": 0})
    f = str(tmp_path / "x.parquet")
    # the chr2 rows alone would fit INT32
    scope = pm.gintervals(["chr1", "chr2"], [2147000000, 0], [2147483648, 10000])

    inmem = pm.gextract("big", scope)
    pm.gextract("big", scope, file=f)
    p = _read_pq(f, "int64")

    assert p["types"] == ["string", "int64", "int64", "double"]
    _expect_arrow_like(p)
    _expect_same_rows(p["table"], inmem, ["big"])


def test_a_symlinked_target_is_written_through_and_the_link_kept(tmp_path):
    _pyarrow()
    _six_chrom_db(tmp_path)
    inmem = pm.gextract("six", pm.gintervals_all())

    for multitask in (False, True):
        parent = tmp_path / f"links{multitask}"
        link_dir = parent / "links"
        real_dir = parent / "real"
        link_dir.mkdir(parents=True)
        real_dir.mkdir()
        for name in ("abs.parquet", "rel.parquet"):
            (real_dir / name).write_text("old content\n")
        links = {
            "abs": str(real_dir / "abs.parquet"),  # to an existing file
            "rel": os.path.join("..", "real", "rel.parquet"),  # relative, to an existing file
            "dangling": "chain.parquet",  # relative, to a link to a file that does not exist yet
            "chain": str(real_dir / "new.parquet"),
        }
        for k, text in links.items():
            os.symlink(text, link_dir / f"{k}.parquet")

        pm.CONFIG["multitasking"] = multitask
        for k in ("abs", "rel", "dangling"):
            pm.gextract("six", pm.gintervals_all(), file=str(link_dir / f"{k}.parquet"))

        assert {k: os.readlink(link_dir / f"{k}.parquet") for k in links} == links
        assert _dir_entries(link_dir) == sorted(f"{k}.parquet" for k in links)
        assert _dir_entries(real_dir) == ["abs.parquet", "new.parquet", "rel.parquet"]
        for name in _dir_entries(real_dir):
            _expect_same_rows(_read_pq(str(real_dir / name))["table"], inmem, ["six"])


def test_temporary_files_are_named_for_this_host_and_process(tmp_path, monkeypatch):
    _six_chrom_db(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    f = out / "x.parquet"
    seen = []
    close = _pymisha.pm_parquet_writer_close

    def peek(*args):
        close(*args)
        seen.extend(_dir_entries(out))  # the finished temporary file, before the rename

    monkeypatch.setattr(_pymisha, "pm_parquet_writer_close", peek)
    tmp = f".x.parquet.tmp.{_host_tag()}.{os.getpid()}"
    for multitask in (False, True):
        seen.clear()
        pm.CONFIG["multitasking"] = multitask
        pm.gextract("six", pm.gintervals_all(), file=str(f))
        assert seen == [tmp] or seen == [tmp, "x.parquet"], seen
        assert _dir_entries(out) == ["x.parquet"]
        f.unlink()


def test_leftovers_of_killed_writers_on_this_host_are_removed_and_nothing_else(tmp_path):
    _six_chrom_db(tmp_path)
    host = _host_tag()
    dead = 4194303  # the largest pid Linux allows; running only if the pids wrapped around to it
    if os.path.exists(f"/proc/{dead}"):
        pytest.skip("pid 4194303 is running")
    sleeper = subprocess.Popen(["sleep", "600"])
    try:
        for multitask in (False, True):
            out = tmp_path / f"out{multitask}"
            out.mkdir()
            stale = [f".x.parquet.tmp.{host}.{dead}{s}" for s in ("", ".0", ".11")]
            kept = [
                f".x.parquet.tmp.someotherhost.{dead}",
                *(f".x.parquet.tmp.{host}.{sleeper.pid}{s}" for s in ("", ".2")),
                # not exactly .<target>.tmp.<host>.<pid>[.<kid>]
                f".x.parquet.tmp.{host}.0{dead}",
                *(f".x.parquet.tmp.{host}.{dead}{s}" for s in (".x", ".1.2", ".")),
                f".x.parquet.tmp.{dead}",
                f".y.parquet.tmp.{host}.{dead}",
            ]
            for name in stale + kept:
                (out / name).touch()

            pm.CONFIG["multitasking"] = multitask
            pm.gextract("six", pm.gintervals_all(), file=str(out / "x.parquet"))
            assert _dir_entries(out) == sorted(["x.parquet", *kept])
    finally:
        sleeper.kill()
        sleeper.wait()
