import shutil
import subprocess

import pandas as pd
import pytest

import pymisha as pm


def _require_rscript():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")


def _write_r_serialized_df(path, r_df_expr):
    r_cmd = (
        "out<-commandArgs(TRUE)[1]; "
        f"df<-{r_df_expr}; "
        "con<-file(out,'wb'); serialize(df, con); close(con)"
    )
    subprocess.check_call(["Rscript", "-e", r_cmd, str(path)])


def _write_meta(path, stats_expr, zeroline_expr):
    r_cmd = (
        "out<-commandArgs(TRUE)[1]; "
        f"stats<-{stats_expr}; "
        f"zeroline<-{zeroline_expr}; "
        "con<-file(out,'wb'); serialize(list(stats=stats, zeroline=zeroline), con); close(con)"
    )
    subprocess.check_call(["Rscript", "-e", r_cmd, str(path)])


def _create_db_root(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    chrom_sizes = root / "chrom_sizes.txt"
    chrom_sizes.write_text("1\t100\n2\t100\n")
    tracks = root / "tracks"
    tracks.mkdir()
    return root


@pytest.fixture
def restore_db():
    current_root = pm._shared._GROOT
    yield
    if current_root:
        pm.gdb_init(current_root)

def test_indexed_bigset_1d_roundtrip(tmp_path, restore_db):
    _require_rscript()
    root = _create_db_root(tmp_path)
    interv_dir = root / "tracks" / "big1.interv"
    interv_dir.mkdir()

    _write_r_serialized_df(
        interv_dir / "chr1",
        "data.frame(chrom=factor(c('chr1','chr1')), start=c(0,10), end=c(5,20))",
    )
    _write_r_serialized_df(
        interv_dir / "chr2",
        "data.frame(chrom=factor(c('chr2')), start=c(3), end=c(8))",
    )
    _write_meta(
        interv_dir / ".meta",
        "data.frame(chrom=factor(c('chr1','chr2')), size=c(2,1))",
        "data.frame(chrom=factor(character()), start=integer(), end=integer())",
    )

    pm.gdb_init(str(root))
    df = pm.gintervals_load("big1")
    assert df is not None
    assert len(df) == 3

    pm.gintervals_convert_to_indexed("big1", remove_old=True, force=True)
    assert pm.gintervals_is_indexed("big1")

    df2 = pm.gintervals_load("big1")
    assert len(df2) == 3
    pd.testing.assert_frame_equal(
        df.sort_values(["chrom", "start"]).reset_index(drop=True),
        df2.sort_values(["chrom", "start"]).reset_index(drop=True),
        check_dtype=False,
    )

    df_chr1 = pm.gintervals_load("big1", chrom="chr1")
    assert len(df_chr1) == 2


def test_indexed_bigset_2d_roundtrip(tmp_path, restore_db):
    _require_rscript()
    root = _create_db_root(tmp_path)
    interv_dir = root / "tracks" / "big2d.interv"
    interv_dir.mkdir()

    _write_r_serialized_df(
        interv_dir / "chr1-chr1",
        "data.frame(chrom1=factor(c('chr1')), start1=c(0), end1=c(10), "
        "chrom2=factor(c('chr1')), start2=c(5), end2=c(15))",
    )
    _write_meta(
        interv_dir / ".meta",
        "data.frame(chrom1=factor(c('chr1')), chrom2=factor(c('chr1')), size=c(1))",
        "data.frame(chrom1=factor(character()), start1=integer(), end1=integer(), "
        "chrom2=factor(character()), start2=integer(), end2=integer())",
    )

    pm.gdb_init(str(root))
    df = pm.gintervals_load("big2d")
    assert df is not None
    assert len(df) == 1

    pm.gintervals_2d_convert_to_indexed("big2d", remove_old=True, force=True)
    assert pm.gintervals_is_indexed("big2d")

    df2 = pm.gintervals_load("big2d")
    assert len(df2) == 1
    pd.testing.assert_frame_equal(
        df.sort_values(["chrom1", "start1"]).reset_index(drop=True),
        df2.sort_values(["chrom1", "start1"]).reset_index(drop=True),
        check_dtype=False,
    )

    df_pair = pm.gintervals_load("big2d", chrom1="chr1", chrom2="chr1")
    assert len(df_pair) == 1
