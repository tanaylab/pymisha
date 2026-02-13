import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pymisha as pm

TESTDB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _run_r_table(r_code):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as script_fd:
        script_path = script_fd.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as out_fd:
        out_path = out_fd.name

    r_script = f"""
library(misha)
gdb.init(\"{TESTDB}\")
{r_code}
names(df) <- gsub(\" \", \"_\", names(df))
write.table(df, file=\"{out_path}\", sep='\t', row.names=FALSE, quote=FALSE)
"""

    try:
        with open(script_path, "w") as handle:
            handle.write(r_script)

        result = subprocess.run(
            ["R", "--quiet", "--no-save", "-f", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"R failed: {result.stderr}\n{result.stdout}")

        return pd.read_csv(out_path, sep="\t")
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def _normalize_chrom(chrom):
    chrom = str(chrom)
    return chrom[3:] if chrom.startswith("chr") else chrom


def _format_percentile(value):
    return f"{float(value):g}"


def test_gintervals_summary_matches_r():
    intervs = pm.gintervals("1", [0, 100], [200, 400])
    py = pm.gintervals_summary("dense_track", intervs)

    r_code = """
intervs <- gintervals("1", c(0, 100), c(200, 400))
df <- gintervals.summary("dense_track", intervs)
"""
    r_df = _run_r_table(r_code)

    py = py.rename(columns={
        "Total intervals": "Total_intervals",
        "NaN intervals": "NaN_intervals",
        "Std dev": "Std_dev",
    })

    assert [_normalize_chrom(c) for c in py["chrom"]] == [
        _normalize_chrom(c) for c in r_df["chrom"]
    ]
    assert list(py["start"]) == list(r_df["start"])
    assert list(py["end"]) == list(r_df["end"])

    for col in [
        "Total_intervals",
        "NaN_intervals",
        "Min",
        "Max",
        "Sum",
        "Mean",
        "Std_dev",
    ]:
        np.testing.assert_allclose(py[col].to_numpy(dtype=float), r_df[col].to_numpy(dtype=float), rtol=1e-6, atol=1e-9)


def test_gintervals_quantiles_matches_r():
    intervs = pm.gintervals("1", [0, 100], [200, 400])
    percentiles = [0.25, 0.5, 0.9]
    py = pm.gintervals_quantiles("dense_track", percentiles=percentiles, intervals=intervs)

    r_code = """
intervs <- gintervals("1", c(0, 100), c(200, 400))
df <- gintervals.quantiles("dense_track", percentiles=c(0.25, 0.5, 0.9), intervals=intervs)
"""
    r_df = _run_r_table(r_code)

    assert [_normalize_chrom(c) for c in py["chrom"]] == [
        _normalize_chrom(c) for c in r_df["chrom"]
    ]
    assert list(py["start"]) == list(r_df["start"])
    assert list(py["end"]) == list(r_df["end"])

    for p in percentiles:
        col = _format_percentile(p)
        np.testing.assert_allclose(py[col].to_numpy(dtype=float), r_df[col].to_numpy(dtype=float), rtol=1e-6, atol=1e-9)


def test_gintervals_summary_intervals_set_out_roundtrip():
    intervs = pm.gintervals("1", [0, 100], [200, 400])
    set_name = "test_summary_set_out"

    if pm.gintervals_exists(set_name):
        pm.gintervals_rm(set_name, force=True)

    expected = pm.gintervals_summary("dense_track", intervs)
    ret = pm.gintervals_summary("dense_track", intervs, intervals_set_out=set_name)
    assert ret is None
    assert pm.gintervals_exists(set_name)

    loaded = pm.gintervals_load(set_name)
    assert loaded is not None
    assert list(loaded.columns) == list(expected.columns)

    for col in loaded.columns:
        if col == "chrom":
            assert [_normalize_chrom(v) for v in loaded[col]] == [_normalize_chrom(v) for v in expected[col]]
        else:
            np.testing.assert_allclose(
                loaded[col].to_numpy(dtype=float),
                expected[col].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
            )

    pm.gintervals_rm(set_name, force=True)


def test_gintervals_quantiles_intervals_set_out_roundtrip():
    intervs = pm.gintervals("1", [0, 100], [200, 400])
    percentiles = [0.25, 0.5, 0.9]
    set_name = "test_quantiles_set_out"

    if pm.gintervals_exists(set_name):
        pm.gintervals_rm(set_name, force=True)

    expected = pm.gintervals_quantiles("dense_track", percentiles=percentiles, intervals=intervs)
    ret = pm.gintervals_quantiles(
        "dense_track",
        percentiles=percentiles,
        intervals=intervs,
        intervals_set_out=set_name,
    )
    assert ret is None
    assert pm.gintervals_exists(set_name)

    loaded = pm.gintervals_load(set_name)
    assert loaded is not None
    assert list(loaded.columns) == list(expected.columns)

    for col in loaded.columns:
        if col == "chrom":
            assert [_normalize_chrom(v) for v in loaded[col]] == [_normalize_chrom(v) for v in expected[col]]
        else:
            np.testing.assert_allclose(
                loaded[col].to_numpy(dtype=float),
                expected[col].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
            )

    pm.gintervals_rm(set_name, force=True)
