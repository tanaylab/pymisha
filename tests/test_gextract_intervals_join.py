"""Port of R tests/testthat/test-gextract-intervals-join.R (PR #124)."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _make_1d_intervals_with_meta():
    """Build a small 1D intervals data frame with metadata of every supported type."""
    intervs = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2"]))
    n = len(intervs)
    intervs = intervs.copy()
    intervs["gene_id"] = [f"g{i:04d}" for i in range(n)]
    intervs["score"] = np.arange(1, n + 1, dtype=float) / 10.0
    intervs["rank"] = np.arange(1, n + 1, dtype=int)
    intervs["is_top"] = (np.arange(n) % 2 == 1)
    intervs["category"] = pd.Categorical(
        [["a", "b", "c"][i % 3] for i in range(n)]
    )
    return intervs


def test_intervals_join_invalid_value_raises():
    intervs = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2"]))
    with pytest.raises(ValueError, match="intervals_join"):
        pm.gextract("dense_track", intervs, intervals_join="bogus")


def test_intervals_join_id_is_default():
    intervs = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2"]))
    a = pm.gextract("dense_track", intervs)
    b = pm.gextract("dense_track", intervs, intervals_join="id")
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
    )
    assert "intervalID" in a.columns


def test_intervals_join_none_drops_intervalID():
    intervs = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2"]))
    base = pm.gextract("dense_track", intervs)
    none = pm.gextract("dense_track", intervs, intervals_join="none")
    assert "intervalID" not in none.columns
    assert len(base) == len(none)
    cols_no_id = [c for c in base.columns if c != "intervalID"]
    pd.testing.assert_frame_equal(
        base[cols_no_id].reset_index(drop=True),
        none.reset_index(drop=True),
    )


def test_intervals_join_intervals_1d_all_dtypes():
    intervs = _make_1d_intervals_with_meta()
    res = pm.gextract("dense_track", intervs, intervals_join="intervals")

    assert "intervalID" not in res.columns
    # Iterator coords stay as-is; input coords get "1" suffix
    expected_cols = {
        "chrom", "start", "end", "dense_track",
        "chrom1", "start1", "end1",
        "gene_id", "score", "rank", "is_top", "category",
    }
    assert expected_cols.issubset(set(res.columns))

    # Reference: build the join in Python via positional indexing on intervalID.
    base = pm.gextract("dense_track", intervs)
    ref_intervs = intervs.copy()
    conflict = [c for c in ref_intervs.columns if c in base.columns]
    ref_intervs = ref_intervs.rename(columns={c: c + "1" for c in conflict})
    ref_intervs = ref_intervs.reset_index(drop=True)
    base_sorted = base.sort_values("intervalID").reset_index(drop=True)
    # intervalID is 1-indexed -> subtract 1 for positional lookup
    attached = ref_intervs.iloc[base_sorted["intervalID"].values - 1].reset_index(drop=True)
    ref = pd.concat(
        [base_sorted.drop(columns=["intervalID"]).reset_index(drop=True), attached],
        axis=1,
    )
    res_sorted = res.sort_values(["chrom", "start"]).reset_index(drop=True)
    ref_sorted = ref.sort_values(["chrom", "start"]).reset_index(drop=True)

    for col in ("gene_id", "score", "rank", "is_top"):
        assert (res_sorted[col].values == ref_sorted[col].values).all(), col
    assert (res_sorted["chrom1"].astype(str).values == ref_sorted["chrom1"].astype(str).values).all()
    assert (res_sorted["start1"].values == ref_sorted["start1"].values).all()
    assert (res_sorted["end1"].values == ref_sorted["end1"].values).all()
    # Categorical compared as string
    assert (res_sorted["category"].astype(str).values == ref_sorted["category"].astype(str).values).all()


def test_intervals_join_intervals_rejects_unsupported_dtype():
    intervs = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2"]))
    intervs = intervs.copy()
    # List-of-int per row -> object dtype with non-str values
    intervs["bad"] = [[1] for _ in range(len(intervs))]
    with pytest.raises(TypeError, match="bad"):
        pm.gextract("dense_track", intervs, intervals_join="intervals")


def test_intervals_join_intervals_rejects_file_arg(tmp_path):
    intervs = _make_1d_intervals_with_meta()
    out = tmp_path / "out.tsv"
    with pytest.raises(ValueError, match="intervals_join"):
        pm.gextract("dense_track", intervs, intervals_join="intervals", file=str(out))


def test_intervals_join_intervals_rejects_intervals_set_out():
    intervs = _make_1d_intervals_with_meta()
    with pytest.raises(ValueError, match="intervals_join"):
        pm.gextract(
            "dense_track", intervs,
            intervals_join="intervals",
            intervals_set_out="tmpset_join",
        )


def test_intervals_join_intervals_2d_scope():
    """intervals_join='intervals' works on a 2D scope (rects_track)."""
    # rects_track is the 2D track present in the test DB.
    TRACK_NAME = "rects_track"

    # Build a plain 2D intervals frame directly - avoids going through
    # gscreen/gextract before we test the feature under test.
    intervs = pm.gintervals_2d(
        chroms1=["1", "1"],
        starts1=[0, 100000],
        ends1=[250000, 500000],
        chroms2=["1", "1"],
        starts2=[0, 200000],
        ends2=[300000, 500000],
    )
    if intervs is None or len(intervs) == 0:
        pytest.skip("gintervals_2d returned no intervals")

    intervs = intervs.copy()
    n = len(intervs)
    intervs["tag"] = [f"t{i:03d}" for i in range(n)]
    intervs["weight"] = np.arange(1, n + 1, dtype=float)

    try:
        res = pm.gextract(TRACK_NAME, intervs, intervals_join="intervals")
    except AttributeError as exc:
        if "pm_extract_2d" in str(exc):
            pytest.skip(f"2D C extension not available: {exc}")
        raise

    if res is None or len(res) == 0:
        pytest.skip("no 2D test data returned for this query")

    assert "intervalID" not in res.columns
    # All 6 2D coord columns conflict with output -> input copies get "1" suffix.
    expected = {
        "chrom1", "start1", "end1", "chrom2", "start2", "end2",
        "chrom11", "start11", "end11", "chrom21", "start21", "end21",
        "tag", "weight",
    }
    assert expected.issubset(set(res.columns))


def test_intervals_join_intervals_multitask_vs_serial():
    """Serial and multitask paths produce identical results with intervals_join='intervals'."""
    intervs = _make_1d_intervals_with_meta()

    saved_mt = pm.CONFIG["multitasking"]
    try:
        pm.CONFIG["multitasking"] = False
        res_serial = pm.gextract("dense_track", intervs, intervals_join="intervals")
        res_serial = res_serial.sort_values(["chrom", "start"]).reset_index(drop=True)
    finally:
        pm.CONFIG["multitasking"] = saved_mt

    try:
        pm.CONFIG["multitasking"] = True
        res_mt = pm.gextract("dense_track", intervs, intervals_join="intervals")
        res_mt = res_mt.sort_values(["chrom", "start"]).reset_index(drop=True)
    finally:
        pm.CONFIG["multitasking"] = saved_mt

    assert list(res_serial.columns) == list(res_mt.columns)
    pd.testing.assert_frame_equal(res_serial, res_mt)


def test_intervals_join_none_allows_track_parallel_strategy():
    """intervals_join='none' is compatible with the track-parallel multitasking strategy."""
    intervs = pm.gscreen("dense_track > 0.2", pm.gintervals(["1", "2"]))

    saved_mt = pm.CONFIG["multitasking"]
    saved_strat = pm.CONFIG.get("multitasking_strategy", "auto")
    try:
        pm.CONFIG["multitasking"] = True
        pm.CONFIG["multitasking_strategy"] = "tracks"
        # 9 distinct expressions (above the tracks-strategy threshold).
        # pymisha deduplicates identical expressions on the C++ side, so use
        # arithmetic variants to keep each expression unique.
        exprs = [f"dense_track + {i}" for i in range(9)]
        cn = [f"v{i}" for i in range(1, len(exprs) + 1)]
        res = pm.gextract(exprs, intervals=intervs, intervals_join="none", colnames=cn)
    finally:
        pm.CONFIG["multitasking"] = saved_mt
        pm.CONFIG["multitasking_strategy"] = saved_strat

    assert "intervalID" not in res.columns
    assert {"chrom", "start", "end", *cn}.issubset(set(res.columns))
