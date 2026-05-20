"""Tests for the CartesianGrid 2D iterator via pm_test_cartesian_grid_iterator binding."""
import os
import shutil

import numpy as np
import pymisha as pm
import pytest
import _pymisha

from pymisha._quadtree import write_2d_track_file

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)
TEST_DB = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test"
)


def _track_dir_cgi(name: str) -> str:
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track_cgi(name: str) -> None:
    tdir = _track_dir_cgi(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        _pymisha.pm_dbreload()


@pytest.fixture()
def cgi_rects_track(_init_db):
    """2D rects track for CartesianGrid scanner integration tests."""
    # Re-initialize to the test DB in case a previous test left GROOT elsewhere.
    pm.gdb_init(TEST_DB)

    tname = "test.cgi_scanner_rects"
    _cleanup_track_cgi(tname)

    tdir = _track_dir_cgi(tname)
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, ".attributes"), "w") as f:
        f.write("type=rectangles\ndimensions=2\n")

    # chrom-pair 1-1 (chromids 0-0): two objects
    write_2d_track_file(
        os.path.join(tdir, "1-1"),
        [
            (100, 200, 300, 400, 1.0),
            (50_000, 60_000, 150_000, 160_000, 2.0),
        ],
        (0, 0, 500_000, 500_000),
        is_points=False,
    )

    _pymisha.pm_dbreload()
    yield tname
    _cleanup_track_cgi(tname)


def _1d_intervals(rows):
    """rows: list of (chromid, start, end)."""
    return {
        "chrom": np.array([r[0] for r in rows], dtype=np.int32),
        "start": np.array([r[1] for r in rows], dtype=np.int64),
        "end":   np.array([r[2] for r in rows], dtype=np.int64),
    }


def _scope(rects):
    return {
        "chrom1": np.array([r[0] for r in rects], dtype=np.int32),
        "start1": np.array([r[1] for r in rects], dtype=np.int64),
        "end1":   np.array([r[2] for r in rects], dtype=np.int64),
        "chrom2": np.array([r[3] for r in rects], dtype=np.int32),
        "start2": np.array([r[4] for r in rects], dtype=np.int64),
        "end2":   np.array([r[5] for r in rects], dtype=np.int64),
    }


def test_two_centers_full_expansion(_init_db):
    # Two interval centers on chrom 0: 25000 and 125000. Expansion [-10000, 10000]
    # = single 20kb window per center.
    # No overlap between centers (distance=100k >> 20k window).
    # Pairwise: 2x2 = 4 cells.
    intervals = _1d_intervals([(0, 0, 50_000), (0, 100_000, 150_000)])
    expansion = np.array([-10_000, 10_000], dtype=np.int64)
    scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    assert len(out["start1"]) == 4
    # Centers are 25000 and 125000; windows are [15000, 35000) and [115000, 135000).
    expected_starts = sorted([(15_000, 15_000), (15_000, 115_000),
                              (115_000, 15_000), (115_000, 115_000)])
    actual_starts = sorted(zip(out["start1"].tolist(), out["start2"].tolist()))
    assert actual_starts == expected_starts


def test_three_expansion_values(_init_db):
    # Three expansion values -> 2 pairs per axis. 1 center, 2x2 = 4 cells.
    intervals = _1d_intervals([(0, 0, 50_000)])
    expansion = np.array([-10_000, 0, 10_000], dtype=np.int64)
    scope = _scope([(0, 0, 100_000, 0, 0, 100_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    # 2 expansion pairs per axis, 1 center each axis -> 4 cells.
    assert len(out["start1"]) == 4


def test_band_idx_filter(_init_db):
    # 3 centers; band_idx [0, 1] means delta in {0, 1} only.
    # delta = idx0 - idx1.
    # delta=0: (0,0), (1,1), (2,2) = 3 cells.
    # delta=1: (1,0), (2,1) = 2 cells.
    # Total 5.
    intervals = _1d_intervals([(0, 0, 10_000),
                                (0, 100_000, 110_000),
                                (0, 200_000, 210_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, (0, 1), scope, None)
    assert len(out["start1"]) == 5


def test_scope_filters_cells(_init_db):
    # Narrow scope drops most cells.
    intervals = _1d_intervals([(0, 0, 10_000),
                                (0, 100_000, 110_000),
                                (0, 200_000, 210_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    # Scope only covers the first center's window [4000,6000) x [4000,6000).
    scope = _scope([(0, 0, 50_000, 0, 0, 50_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    # Only the (gp0, gp0) cell at (4000-6000, 4000-6000) intersects the scope.
    # gp1 center is 105000 with window [104000,106000) -- outside scope.
    # gp2 center is 205000 with window [204000,206000) -- outside scope.
    # So only (gp0, gp0) survives.
    assert len(out["start1"]) == 1


def test_empty_scope_yields_nothing(_init_db):
    intervals = _1d_intervals([(0, 0, 10_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, _scope([]), None)
    assert len(out["start1"]) == 0


def test_empty_intervals_yields_nothing(_init_db):
    intervals = _1d_intervals([])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    scope = _scope([(0, 0, 100_000, 0, 0, 100_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    assert len(out["start1"]) == 0


def test_band_active_drops_inter_chrom(_init_db):
    # With band active, inter-chrom pairs should be skipped.
    # One center on chrom 0, one on chrom 1.
    intervals = _1d_intervals([(0, 0, 10_000), (1, 0, 10_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    scope = _scope([
        (0, 0, 50_000, 0, 0, 50_000),
        (1, 0, 50_000, 1, 0, 50_000),
        (0, 0, 50_000, 1, 0, 50_000),
        (1, 0, 50_000, 0, 0, 50_000),
    ])
    out_no_band = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    out_band = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, (-50, 50))
    # No band: all 4 pairs (chrom0xchrom0, chrom1xchrom1, chrom0xchrom1, chrom1xchrom0)
    # each contribute 1 cell.
    assert len(out_no_band["start1"]) == 4
    # With band: only same-chrom pairs -> 2 cells.
    assert len(out_band["start1"]) == 2


def test_separate_axes_intervals(_init_db):
    # Axis-0 center at 5000 (chrom 0), axis-1 center at 205000 (chrom 0).
    intervals1 = _1d_intervals([(0, 0, 10_000)])
    intervals2 = _1d_intervals([(0, 200_000, 210_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals1, expansion, intervals2, None, None, scope, None)
    # 1 cell: (4000-6000, 204000-206000).
    assert len(out["start1"]) == 1
    assert out["start1"][0] == 4_000 and out["end1"][0] == 6_000
    assert out["start2"][0] == 204_000 and out["end2"][0] == 206_000


def test_separate_expansions(_init_db):
    # Same center on both axes but different expansions.
    intervals = _1d_intervals([(0, 0, 10_000)])
    e1 = np.array([-1_000, 1_000], dtype=np.int64)
    e2 = np.array([-500, 500], dtype=np.int64)
    scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, e1, intervals, e2, None, scope, None)
    # Single center, one expansion pair per axis -> 1 cell.
    assert len(out["start1"]) == 1
    # Center is 5000.
    # Axis-0 window: [5000-1000, 5000+1000) = [4000, 6000).
    # Axis-1 window: [5000-500, 5000+500) = [4500, 5500).
    assert out["start1"][0] == 4_000 and out["end1"][0] == 6_000
    assert out["start2"][0] == 4_500 and out["end2"][0] == 5_500


def test_overlap_correction(_init_db):
    # Two centers very close: centers at 1000 and 1500.
    # Expansion [-1000, 1000]: maximal windows are [0, 2000) and [500, 2500).
    # They overlap, so R adjusts. mid_coord = (1000 + 1500) / 2 = 1250.
    # Both windows extend past the midpoint (1000+(-1000)=0 < 1250; 1000+1000=2000 > 1250).
    # => prev.max_expansion = 1250 - 1000 = 250; gp.min_expansion = 1250 - 1500 = -250.
    # Corrected windows: [0, 1250) and [1250, 2500).
    intervals = _1d_intervals([(0, 0, 2_000), (0, 1_000, 2_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    scope = _scope([(0, 0, 10_000, 0, 0, 10_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    # 2 axis-0 centers * 2 axis-1 centers = 4 cells.
    assert len(out["start1"]) == 4
    # Cell (0, 0): axis0 window [0, 1250), axis1 window [0, 1250).
    points = sorted(zip(out["start1"].tolist(), out["end1"].tolist(),
                        out["start2"].tolist(), out["end2"].tolist()))
    assert points[0] == (0, 1250, 0, 1250)
    assert points[1] == (0, 1250, 1250, 2500)
    assert points[2] == (1250, 2500, 0, 1250)
    assert points[3] == (1250, 2500, 1250, 2500)


def test_expansion_too_few_values_rejected(_init_db):
    intervals = _1d_intervals([(0, 0, 10_000)])
    expansion = np.array([100], dtype=np.int64)  # only 1 value
    scope = _scope([(0, 0, 100_000, 0, 0, 100_000)])
    with pytest.raises(RuntimeError, match="at least 2"):
        _pymisha.pm_test_cartesian_grid_iterator(
            intervals, expansion, None, None, None, scope, None)


def test_chrom_size_caps_expansion(_init_db):
    # Test DB chrom "1" is 500k (chromid=0). Center at 99999 or 100000 with
    # expansion [-1000000, 1000000] should be capped to [0, 500000).
    # center = (100000 - 1 + 100000) / 2 = 99999 (integer division).
    # Actually: (99999 + 100000) / 2 = 99999 (integer div).
    # max_expansion = 500000 - 99999 = 400001. So window is [0, 500000).
    intervals = _1d_intervals([(0, 99_999, 100_000)])
    expansion = np.array([-1_000_000, 1_000_000], dtype=np.int64)
    scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, None, scope, None)
    assert len(out["start1"]) == 1
    assert out["start1"][0] == 0
    assert out["end1"][0] == 500_000


def test_band_idx_diagonal_only(_init_db):
    # band_idx (0, 0): only diagonal pairs (idx0 == idx1).
    intervals = _1d_intervals([(0, 0, 10_000),
                                (0, 100_000, 110_000),
                                (0, 200_000, 210_000)])
    expansion = np.array([-1_000, 1_000], dtype=np.int64)
    scope = _scope([(0, 0, 500_000, 0, 0, 500_000)])
    out = _pymisha.pm_test_cartesian_grid_iterator(
        intervals, expansion, None, None, (0, 0), scope, None)
    # Only (0,0), (1,1), (2,2) -> 3 cells.
    assert len(out["start1"]) == 3
    # All cells are on the diagonal: start1 == start2.
    assert all(s1 == s2 for s1, s2 in zip(out["start1"].tolist(), out["start2"].tolist()))


# ---------------------------------------------------------------------------
# pm_extract_2d_scanner + cartesian_grid integration tests (T3.4)
# ---------------------------------------------------------------------------

def _intervals1_dict(rows):
    """rows: list of (chromid, start, end) for the cartesian_grid intervals1/2 dicts."""
    return {
        "chrom": np.array([r[0] for r in rows], dtype=np.int32),
        "start": np.array([r[1] for r in rows], dtype=np.int64),
        "end":   np.array([r[2] for r in rows], dtype=np.int64),
    }


def _scope_dict_cgi(rows):
    """rows: list of (c1, s1, e1, c2, s2, e2)."""
    return {
        "chrom1": np.array([r[0] for r in rows], dtype=np.int32),
        "start1": np.array([r[1] for r in rows], dtype=np.int64),
        "end1":   np.array([r[2] for r in rows], dtype=np.int64),
        "chrom2": np.array([r[3] for r in rows], dtype=np.int32),
        "start2": np.array([r[4] for r in rows], dtype=np.int64),
        "end2":   np.array([r[5] for r in rows], dtype=np.int64),
    }


def test_pm_extract_2d_scanner_cartesian_grid_basic(cgi_rects_track):
    """Smoke test: pm_extract_2d_scanner with kind='cartesian_grid' runs and returns
    the expected coord + value arrays."""
    # Two interval centers on chrom 0 (chromid 0): 25_000 and 125_000.
    # Expansion [-30_000, 30_000] = one 60kb window per center.
    # 2x2 = 4 cells, all within the full-chrom scope.
    ivd1 = _intervals1_dict([(0, 0, 50_000), (0, 100_000, 150_000)])
    scope = _scope_dict_cgi([(0, 0, 500_000, 0, 0, 500_000)])
    policy = {
        "kind":       "cartesian_grid",
        "intervals1": ivd1,
        "expansion1": np.array([-30_000, 30_000], dtype=np.int64),
        "intervals2": None,
        "expansion2": None,
        "min_band_idx": None,
        "max_band_idx": None,
    }
    vars_list = [(cgi_rects_track, "area")]
    colnames  = ["val"]

    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)

    # Coord arrays must be present with correct names.
    for key in ("_chrom1", "_start1", "_end1", "_chrom2", "_start2", "_end2"):
        assert key in result, f"missing coord key: {key}"

    # Value array must be present, float64.
    assert "val" in result
    assert result["val"].dtype == np.float64

    # 4 cells (2 centers x 2 centers, each with one expansion pair).
    assert len(result["val"]) == 4
    assert len(result["_chrom1"]) == 4


def test_pm_extract_2d_scanner_cartesian_grid_with_band_idx(cgi_rects_track):
    """band_idx filter restricts to diagonal (delta_idx == 0)."""
    # 2 centers -> only 2 diagonal cells when band_idx=[0,0].
    ivd1 = _intervals1_dict([(0, 0, 50_000), (0, 100_000, 150_000)])
    scope = _scope_dict_cgi([(0, 0, 500_000, 0, 0, 500_000)])
    policy = {
        "kind":         "cartesian_grid",
        "intervals1":   ivd1,
        "expansion1":   np.array([-30_000, 30_000], dtype=np.int64),
        "intervals2":   None,
        "expansion2":   None,
        "min_band_idx": 0,
        "max_band_idx": 0,
    }
    vars_list = [(cgi_rects_track, "area")]
    colnames  = ["val"]

    result = _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)

    # Only 2 diagonal cells.
    assert len(result["val"]) == 2
    # Diagonal cells: start1 == start2 for each row.
    assert all(s1 == s2 for s1, s2 in zip(result["_start1"].tolist(), result["_start2"].tolist()))


def test_pm_extract_2d_scanner_cartesian_grid_band_idx_with_intervals2_raises(cgi_rects_track):
    """Providing both intervals2 and band_idx must raise RuntimeError (R semantics)."""
    ivd1 = _intervals1_dict([(0, 0, 50_000)])
    ivd2 = _intervals1_dict([(0, 100_000, 150_000)])
    scope = _scope_dict_cgi([(0, 0, 500_000, 0, 0, 500_000)])
    policy = {
        "kind":         "cartesian_grid",
        "intervals1":   ivd1,
        "expansion1":   np.array([-30_000, 30_000], dtype=np.int64),
        "intervals2":   ivd2,   # explicitly set
        "expansion2":   None,
        "min_band_idx": 0,
        "max_band_idx": 1,
    }
    vars_list = [(cgi_rects_track, "area")]
    colnames  = ["val"]

    with pytest.raises(RuntimeError, match="band_idx filter requires intervals2=None"):
        _pymisha.pm_extract_2d_scanner(policy, scope, vars_list, colnames, None)
