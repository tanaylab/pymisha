"""Percentile column naming in gintervals_quantiles.

A fixed ``%g`` (6 significant digits) collapsed nearby percentiles to the same
column name, silently dropping a column - the same failure mode as the gextract
40-char colname cap. Names now use the shortest decimal that round-trips, so
distinct percentiles always get distinct columns; genuine duplicates are deduped
with '_' (the C++ PMDataFrame convention).
"""

import pymisha as pm
from pymisha.summary import _format_percentile, _percentile_colnames


def test_format_percentile_clean_common_names():
    """Common percentiles keep clean, short names (no precision noise)."""
    assert _format_percentile(0.0) == "0"
    assert _format_percentile(1.0) == "1"
    assert _format_percentile(0.5) == "0.5"
    assert _format_percentile(0.95) == "0.95"
    assert _format_percentile(0.975) == "0.975"


def test_format_percentile_round_trips_close_values():
    """Nearby percentiles that old %g merged now get distinct names."""
    a, b = 0.123456789, 0.1234567891
    assert f"{a:g}" == f"{b:g}"  # old behaviour collided
    assert _format_percentile(a) != _format_percentile(b)
    assert float(_format_percentile(a)) == a
    assert float(_format_percentile(b)) == b


def test_percentile_colnames_dedup():
    """The same percentile listed twice keeps both columns (second gets '_')."""
    assert _percentile_colnames([0.5, 0.5]) == ["0.5", "0.5_"]


def test_close_percentiles_not_dropped_cpp_path():
    """C++ path: close percentiles must yield distinct columns, not silently drop."""
    ivs = pm.gintervals("1", 0, 100000)
    pcts = [0.123456789, 0.1234567891, 0.5, 0.95, 1.0, 0.0]
    df = pm.gintervals_quantiles("dense_track", percentiles=pcts, intervals=ivs, iterator=20000)
    valcols = [c for c in df.columns if c not in ("chrom", "start", "end", "intervalID")]
    assert valcols == ["0.123456789", "0.1234567891", "0.5", "0.95", "1", "0"]


def test_cpp_and_python_paths_name_alike():
    """Extract (vtrack) path names percentile columns exactly like the C++ path."""
    ivs = pm.gintervals("1", 0, 100000)
    pcts = [0.123456789, 0.1234567891, 0.5]
    cpp = pm.gintervals_quantiles("dense_track", percentiles=pcts, intervals=ivs, iterator=20000)
    pm.gvtrack_create("_qcoltest", "dense_track", func="avg")
    try:
        py = pm.gintervals_quantiles("_qcoltest", percentiles=pcts, intervals=ivs, iterator=20000)
    finally:
        pm.gvtrack_rm("_qcoltest")
    def valcols(d):
        return [c for c in d.columns if c not in ("chrom", "start", "end", "intervalID")]

    assert valcols(cpp) == valcols(py) == ["0.123456789", "0.1234567891", "0.5"]
