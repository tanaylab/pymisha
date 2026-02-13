import pandas as pd
import pandas.testing as pdt

import pymisha as pm


def _make_intervals(num_per_chrom=20, width=100):
    all_intervals = pm.gintervals_all()
    rows = []
    for _, row in all_intervals.iterrows():
        chrom = row["chrom"]
        chrom_end = int(row["end"])
        step = max(width, chrom_end // (num_per_chrom + 1))
        start = 0
        for _ in range(num_per_chrom):
            if start + width > chrom_end:
                break
            rows.append({"chrom": chrom, "start": start, "end": start + width})
            start += step
    return pd.DataFrame(rows)


def test_gextract_multitask_matches_single():
    intervals = _make_intervals()
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "min_processes": 2, "max_processes": 2})
        single = pm.gextract("dense_track", intervals)

        config.update({"multitasking": True, "min_processes": 2, "max_processes": 2})
        multi = pm.gextract("dense_track", intervals)
    finally:
        config.update(saved)

    pdt.assert_frame_equal(single.reset_index(drop=True), multi.reset_index(drop=True))


def test_gscreen_multitask_matches_single():
    intervals = _make_intervals()
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "min_processes": 2, "max_processes": 2})
        single = pm.gscreen("dense_track > 0.2", intervals)

        config.update({"multitasking": True, "min_processes": 2, "max_processes": 2})
        multi = pm.gscreen("dense_track > 0.2", intervals)
    finally:
        config.update(saved)

    if single is None or multi is None:
        assert single is None and multi is None
        return

    pdt.assert_frame_equal(single.reset_index(drop=True), multi.reset_index(drop=True))


def test_gextract_multitask_large_output_matches_single():
    intervals = _make_intervals(num_per_chrom=50, width=10000)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({"multitasking": False, "min_processes": 3, "max_processes": 3})
        single = pm.gextract("dense_track", intervals, iterator=100)

        config.update({"multitasking": True, "min_processes": 3, "max_processes": 3})
        multi = pm.gextract("dense_track", intervals, iterator=100)
    finally:
        config.update(saved)

    pdt.assert_frame_equal(single.reset_index(drop=True), multi.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Single-chrom multitask edge cases (ported from R test-gextract-single-chrom-multitask.R)
# ---------------------------------------------------------------------------


def _sort_df(df):
    """Sort a DataFrame by chrom/start/end/intervalID and reset index."""
    cols = ["chrom", "start", "end"]
    if "intervalID" in df.columns:
        cols.append("intervalID")
    return df.sort_values(cols).reset_index(drop=True)


def test_single_chrom_fixedbin_multitask_matches_serial():
    """Single-chrom fixed-bin multitask split matches serial output.

    Ported from R: test-gextract-single-chrom-multitask.R (line 3).
    When intervals span only a single chromosome, the multitask path should
    still produce exactly the same output as the serial path.
    """
    # Use only chromosome 1 (the largest in the test DB: 500000 bp)
    intervals = pm.gintervals(["1"], 0, 100000)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({
            "multitasking": False,
            "min_processes": 2,
            "max_processes": 8,
        })
        serial = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])

        config.update({
            "multitasking": True,
            "min_processes": 2,
            "max_processes": 8,
        })
        multi = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])
    finally:
        config.update(saved)

    pdt.assert_frame_equal(_sort_df(serial), _sort_df(multi))


def test_single_chrom_vtrack_shifted_multitask_matches_serial():
    """Single-chrom shifted vtrack extraction matches serial output.

    Ported from R: test-gextract-single-chrom-multitask.R (line 28).
    A virtual track with shifted windows on a single chromosome should
    produce identical results in serial and parallel modes.
    """
    vt_name = "test_mt_single_chrom_sum"
    # Clean up any pre-existing vtrack
    if vt_name in (pm.gvtrack_ls() or []):
        pm.gvtrack_rm(vt_name)

    try:
        pm.gvtrack_create(vt_name, src="dense_track", func="sum")
        pm.gvtrack_iterator(vt_name, sshift=-100, eshift=100)

        intervals = pm.gintervals(["1"], 0, 100000)
        config = pm.CONFIG
        saved = config.copy()

        try:
            config.update({
                "multitasking": False,
                "min_processes": 2,
                "max_processes": 8,
            })
            serial = pm.gextract(vt_name, intervals, iterator=20)

            config.update({
                "multitasking": True,
                "min_processes": 2,
                "max_processes": 8,
            })
            multi = pm.gextract(vt_name, intervals, iterator=20)
        finally:
            config.update(saved)

        serial_s = _sort_df(serial)
        multi_s = _sort_df(multi)

        pdt.assert_frame_equal(
            serial_s[["chrom", "start", "end", "intervalID"]],
            multi_s[["chrom", "start", "end", "intervalID"]],
        )
        pd.testing.assert_series_equal(
            serial_s[vt_name], multi_s[vt_name], rtol=1e-6
        )
    finally:
        if vt_name in (pm.gvtrack_ls() or []):
            pm.gvtrack_rm(vt_name)


def test_small_multichrom_multitask_matches_serial():
    """Small multi-chrom extraction still enters multitask path and matches serial.

    Ported from R: test-gextract-single-chrom-multitask.R (line 91).
    Even a tiny region spanning two chromosomes should produce correct results
    with multitasking enabled.
    """
    intervals = pm.gintervals(["1", "2"], 0, 1000)
    config = pm.CONFIG
    saved = config.copy()

    try:
        config.update({
            "multitasking": False,
            "min_processes": 2,
            "max_processes": 8,
        })
        serial = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])

        config.update({
            "multitasking": True,
            "min_processes": 2,
            "max_processes": 8,
        })
        multi = pm.gextract("dense_track", intervals, iterator=20, colnames=["value"])
    finally:
        config.update(saved)

    pdt.assert_frame_equal(_sort_df(serial), _sort_df(multi))
