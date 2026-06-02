"""Output-parity guard for the bp-balanced, range-splitting _parallel_extract.

The parallel (tiles) extract must produce results identical to a serial
extraction over the same scope - same rows, values, and intervalIDs - including
when a single large interval (e.g. a whole chromosome in ALLGENOME) is
range-split across workers. We force gextract down the parallel path on the
small test DB by zeroing the multitask floor.
"""
import pandas as pd
import pandas.testing as pdt

import pymisha as pm

COLS = ["chrom", "start", "end", "dense_track", "intervalID"]
_FORCE_PARALLEL = {
    "multitasking": True, "max_processes": 4,
    "min_intervs4process": 0, "min_scope4process": 0, "progress": False,
}


def _gextract(intervals, iterator, **cfg_over):
    saved = {k: pm.CONFIG.get(k) for k in cfg_over}
    pm.CONFIG.update(cfg_over)
    try:
        return pm.gextract("dense_track", intervals, iterator=iterator)
    finally:
        for k, v in saved.items():
            if v is None:
                pm.CONFIG.pop(k, None)
            else:
                pm.CONFIG[k] = v


def _assert_same(intervals, iterator):
    serial = _gextract(intervals, iterator, multitasking=False)
    par = _gextract(intervals, iterator, **_FORCE_PARALLEL)
    key = ["intervalID", "chrom", "start"]
    pdt.assert_frame_equal(
        serial.sort_values(key).reset_index(drop=True)[COLS],
        par.sort_values(key).reset_index(drop=True)[COLS],
        check_dtype=False,
    )


def test_allgenome_bin_iterator_range_split():
    # One interval per chromosome; chr1/chr2 exceed target_bp -> range-split.
    _assert_same(pm.gintervals_all(), 50)


def test_single_large_chrom_range_split():
    _assert_same(pm.gintervals("1", 0, 500000), 50)


def test_non_bin_aligned_scope_start():
    _assert_same(pm.gintervals([1, 2], [137, 49], [499337, 299999]), 50)


def test_many_small_intervals():
    starts = list(range(0, 400000, 4000))
    iv = pd.DataFrame({"chrom": ["1"] * len(starts),
                       "start": starts,
                       "end": [s + 4000 for s in starts]})
    _assert_same(iv, 50)


def test_track_name_iterator():
    # Non-int iterator: intervals kept whole (no range split); must match serial.
    _assert_same(pm.gintervals_all(), "dense_track")


def test_dataframe_iterator():
    it = pd.DataFrame({"chrom": ["1", "2"], "start": [0, 0], "end": [500000, 300000]})
    _assert_same(pm.gintervals_all(), it)
