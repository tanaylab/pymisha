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
* ``GAP_ARRAY_IMPORT`` -- ``gtrack.array.import`` has no pymisha equivalent.

The set-colnames case writes ``.colnames``; the overlay copies track *dotfiles*
(see overlay.py) so this never touches the shared source DB.
"""

from __future__ import annotations

import pytest

import pymisha as pm

from .baseline import assert_matches_baseline

GAP_ARRAY_FILE_DUMP = "tmpresfile baseline is a read.table dump of file output (header-as-data, stringified, no intervalID)"
GAP_ARRAY_IMPORT = "gtrack.array.import not implemented in pymisha"

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
@pytest.mark.xfail(reason=GAP_ARRAY_FILE_DUMP, strict=True)
def test_gtrack_array_extract_tmpresfile(baseline_id, slice_cols, overlay_db):
    assert_matches_baseline(pm.gtrack_array_extract("test.array", slice_cols, _iv12()), baseline_id)


@pytest.mark.xfail(reason=GAP_ARRAY_IMPORT, strict=True)
def test_gtrack_array_import_extract(overlay_db):
    assert hasattr(pm, "gtrack_array_import"), "gtrack_array_import not implemented"
