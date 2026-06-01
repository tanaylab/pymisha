"""Parity port of R misha ``test-gtrack.array.R`` (array tracks).

Column-name get/set round-trips and direct ``gtrack_array_extract`` (full, a
column slice, and over screened/"sampled" intervals) match R exactly.

The indexed-array read was a real bug found by this port and fixed: on an
indexed-format DB ``gtrack_array_extract`` returned 0 rows because the reader
only knew the legacy per-chromosome files and indexed tracks keep their data in
``track.dat``/``track.idx``. The reader now reads the indexed block too (see
CHANGELOG; ``_array_track.py``).

Open gaps marked ``xfail(strict=True)``:

* ``GAP_ARRAY_FILE_DUMP`` -- the ``tmpresfile`` baselines are
  ``read.table(file, nrows=1000)`` snapshots of misha's *file* output: the
  header line is read as the first data row (so every column is a string) and
  there is no ``intervalID``. That raw text dump isn't a faithful target for the
  typed DataFrame the pandas API returns (the underlying data is already covered
  by the cases above).

The set-colnames case writes ``.colnames``; the overlay copies track *dotfiles*
(see overlay.py) so this never touches the shared source DB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY_FILE_DUMP = "tmpresfile baseline is a read.table dump of file output (header-as-data, stringified, no intervalID)"

_COLS = ["col1", "col3", "col5"]


def _iv12():
    return pm.gintervals([1, 2])


# ---- column-name metadata ------------------------------------------------ #


def test_gtrack_array_get_colnames(overlay_db):
    assert_matches_baseline(pm.gtrack_array_get_colnames("test.array"), "gtrack_array_colnames_array")


def test_gtrack_array_set_colnames(overlay_db):
    cols = pm.gtrack_array_get_colnames("test.array")
    try:
        pm.gtrack_array_set_colnames("test.array", [c + "blabla" for c in cols])
        assert_matches_baseline(pm.gtrack_array_get_colnames("test.array"), "gtrack_array_set_colnames")
    finally:
        pm.gtrack_array_set_colnames("test.array", cols)


# ---- array data extraction (indexed reader fixed) ------------------------ #


def test_gtrack_array_extract_array_intervals(overlay_db):
    assert_matches_baseline(
        pm.gtrack_array_extract("test.array", None, _iv12()), "gtrack_array_extract_array_intervals"
    )


def test_gtrack_array_extract_array_cols_intervals(overlay_db):
    assert_matches_baseline(
        pm.gtrack_array_extract("test.array", _COLS, _iv12()), "gtrack_array_extract_array_cols_intervals"
    )


def test_gtrack_array_extract_sampled_intervals(overlay_db):
    # R shuffles the screened intervals (seed 60427); the result is order-
    # independent (comparator sorts), so the shuffle is omitted.
    intervs = pm.gscreen("test.fixedbin>0.2", pm.gintervals([2, 4, 5, 10]))
    assert_matches_baseline(
        pm.gtrack_array_extract("test.array", _COLS, intervs), "gtrack_array_extract_sampled_intervals"
    )


@pytest.mark.parametrize(
    "baseline_id, slice_cols",
    [("gtrack_array_extract_tmpresfile", None), ("gtrack_array_extract_tmpresfile_cols", _COLS)],
)
def test_gtrack_array_extract_tmpresfile(baseline_id, slice_cols, overlay_db):
    """R's ``tmpresfile`` baseline is ``read.table(file, nrows=1000)``: row 0 is the
    header read as data, every cell is a string, and there is no intervalID column.
    Compare semantically: peel the header row, parse types, take the first 1000
    pymisha rows after sorting, then compare values column-by-column.
    """
    from .baseline import load_baseline
    base_raw = load_baseline(baseline_id)
    # Row 0 carries the column names; remaining rows are the data dump.
    header = list(base_raw.iloc[0])
    base = base_raw.iloc[1:].copy().reset_index(drop=True)
    base.columns = header

    # Cast types: chrom -> str (already), start/end -> int, value cols -> float.
    base["start"] = base["start"].astype(int)
    base["end"] = base["end"].astype(int)
    value_cols = [c for c in base.columns if c not in ("chrom", "start", "end")]
    for c in value_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    py = pm.gtrack_array_extract("test.array", slice_cols, _iv12())
    # Match R's chrom strings ('chr1') against pymisha's possibly bare-name format.
    py = py.copy()
    py["chrom"] = py["chrom"].astype(str).str.replace(r"^chr", "", regex=True)
    base["chrom"] = base["chrom"].astype(str).str.replace(r"^chr", "", regex=True)

    # Sort both, then truncate to the same row count R wrote.
    sort_keys = ["chrom", "start", "end"]
    py = py.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    base = base.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    n = len(base)
    py = py.head(n).reset_index(drop=True)

    assert (py["chrom"].to_numpy() == base["chrom"].to_numpy()).all(), "chrom mismatch"
    assert (py["start"].to_numpy() == base["start"].to_numpy()).all(), "start mismatch"
    assert (py["end"].to_numpy() == base["end"].to_numpy()).all(), "end mismatch"
    for c in value_cols:
        pv = py[c].to_numpy(dtype=float)
        bv = base[c].to_numpy(dtype=float)
        assert np.allclose(pv, bv, rtol=1e-5, atol=1e-5, equal_nan=True), (
            f"value column {c!r} differs"
        )


def test_gtrack_array_import_extract(overlay_db):
    import contextlib
    import tempfile
    from pathlib import Path

    from .baseline import assert_matches_list_baseline

    files = [Path(tempfile.mkstemp(suffix=".tsv")[1]) for _ in range(3)]
    f1, f2, f3 = files
    try:
        pm.gextract("test.sparse", pm.gintervals([1, 2]), file=str(f1))
        pm.gtrack_array_extract(
            "test.array", ["col2", "col3", "col4"], pm.gintervals([1, 2]), file=str(f2),
        )
        pm.gtrack_array_extract(
            "test.array", ["col1", "col3"], pm.gintervals([1, 2]), file=str(f3),
        )
        pm.gtrack_array_import("test_track1", "", str(f1), str(f2))
        r1 = pm.gtrack_array_extract("test_track1", None, pm.gintervals_all())
        pm.gtrack_array_import("test_track2", "", "test_track1", str(f3))
        r2 = pm.gtrack_array_extract("test_track2", None, pm.gintervals_all())
        assert_matches_list_baseline({"r1": r1, "r2": r2}, "gtrack_array_import_extract")
    finally:
        for f in files:
            f.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            pm.gtrack_rm("test_track1", force=True)
        with contextlib.suppress(Exception):
            pm.gtrack_rm("test_track2", force=True)
