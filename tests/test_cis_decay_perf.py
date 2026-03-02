"""Tests for gcis_decay vectorized optimization correctness.

These tests verify that the optimized (numpy-vectorized) gcis_decay produces
identical results to the original per-object Python loop.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

TRACK_DIR = os.path.join(
    os.path.dirname(__file__), "testdb", "trackdb", "test", "tracks"
)


def _track_dir(name):
    return os.path.join(TRACK_DIR, name.replace(".", "/") + ".track")


def _cleanup_track(name):
    tdir = _track_dir(name)
    if os.path.exists(tdir):
        shutil.rmtree(tdir)
        import _pymisha
        _pymisha.pm_dbreload()


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


# ---------------------------------------------------------------------------
# Reference: slow per-object Python implementation
# ---------------------------------------------------------------------------

def _reference_gcis_decay_slow(track, breaks, src, domain, include_lowest=False, band=None):
    """Per-object Python reference matching the original un-optimized loop."""
    from pymisha._quadtree import _read_file_header, query_2d_track_opened
    from pymisha.analysis import (
        _containing_interval,
        _intervals_per_chrom,
        _unify_overlaps_per_chrom,
        _val2bin,
    )
    from pymisha.extract import _find_2d_track_file, _obj_in_band, _validate_band
    from pymisha.intervals import _normalize_chroms
    from pymisha.tracks import gtrack_info

    import _pymisha

    breaks = [float(b) for b in breaks]
    n_bins = len(breaks) - 1
    intra = np.zeros(n_bins, dtype=np.float64)
    inter = np.zeros(n_bins, dtype=np.float64)

    src = src.copy()
    if "chrom" in src.columns:
        src["chrom"] = _normalize_chroms(src["chrom"].astype(str).tolist())
    domain = domain.copy()
    if "chrom" in domain.columns:
        domain["chrom"] = _normalize_chroms(domain["chrom"].astype(str).tolist())

    src_per_chrom = _unify_overlaps_per_chrom(src)
    domain_per_chrom = _intervals_per_chrom(domain)
    band = _validate_band(band)

    info = gtrack_info(track)
    is_points = info.get("type") == "points"
    track_path = _pymisha.pm_track_path(track)

    all_genome = pm.gintervals_all()
    chrom_sizes = {}
    for _, row in all_genome.iterrows():
        chrom_sizes[str(row["chrom"])] = int(row["end"])

    for chrom, csize in chrom_sizes.items():
        filepath = _find_2d_track_file(track_path, chrom, chrom)
        if filepath is None:
            continue
        src_ivs = src_per_chrom.get(str(chrom), [])
        if not src_ivs:
            continue
        domain_ivs = domain_per_chrom.get(str(chrom), [])

        is_pts, num_objs, data = _read_file_header(filepath)
        try:
            import struct
            if num_objs == 0:
                continue
            root_chunk_fpos = struct.unpack_from("<q", data, 12)[0]
            objs = query_2d_track_opened(data, is_pts, num_objs, root_chunk_fpos,
                                          0, 0, csize, csize, band=band)
        finally:
            data.close()

        for obj in objs:
            if is_points:
                x, y, val = obj
                s1, e1 = x, x + 1
                s2, e2 = y, y + 1
            else:
                x1, y1, x2, y2, val = obj
                s1, e1 = x1, x2
                s2, e2 = y1, y2

            if _containing_interval(src_ivs, s1, e1) < 0:
                continue

            distance = abs((s1 + e1 - s2 - e2) // 2)
            idx = _val2bin(distance, breaks, include_lowest)
            if idx < 0:
                continue

            d1_idx = _containing_interval(domain_ivs, s1, e1)
            d2_idx = _containing_interval(domain_ivs, s2, e2)

            if d1_idx >= 0 and d1_idx == d2_idx:
                intra[idx] += 1
            else:
                inter[idx] += 1

    return np.column_stack([intra, inter])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectorizedParity:
    """Verify vectorized gcis_decay matches reference per-object implementation."""

    def test_basic_parity(self):
        """Basic call: full-chrom src and single-domain."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_two_domains(self):
        """Two non-overlapping domains produce correct intra/inter split."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 250000],
            "end": [250000, 500000],
        })
        breaks = [0, 50000, 100000, 200000, 300000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_include_lowest(self):
        """Parity with include_lowest=True."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain, include_lowest=True)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain, include_lowest=True)
        np.testing.assert_array_equal(result, ref)

    def test_parity_partial_src(self):
        """Parity with src covering only part of the chromosome."""
        src = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 200000],
            "end": [80000, 350000],
        })
        domain = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 250000],
            "end": [250000, 500000],
        })
        breaks = [0, 50000, 100000, 200000, 300000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_multi_chrom(self):
        """Parity across multiple chromosomes."""
        src = pd.DataFrame({
            "chrom": ["1", "2"],
            "start": [0, 0],
            "end": [500000, 300000],
        })
        domain = pd.DataFrame({
            "chrom": ["1", "2"],
            "start": [0, 0],
            "end": [500000, 300000],
        })
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_empty_src(self):
        """Empty src produces all zeros."""
        src = pd.DataFrame({"chrom": pd.Series([], dtype=str),
                            "start": pd.Series([], dtype=int),
                            "end": pd.Series([], dtype=int)})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_empty_domain(self):
        """Empty domain: all contacts become inter-domain."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": pd.Series([], dtype=str),
                               "start": pd.Series([], dtype=int),
                               "end": pd.Series([], dtype=int)})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_many_small_domains(self):
        """Many small domains to stress domain lookup vectorization."""
        domain_rows = []
        for i in range(0, 500000, 10000):
            domain_rows.append({"chrom": "1", "start": i, "end": i + 10000})
        domain = pd.DataFrame(domain_rows)

        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 50000, 100000, 200000, 300000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_fine_breaks(self):
        """Fine-grained breaks to stress binning vectorization."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = list(range(0, 500001, 5000))

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)

    def test_parity_r_example(self):
        """Parity for the R documentation example pattern."""
        src = pd.DataFrame({
            "chrom": ["1", "1", "1", "1", "1", "1", "1", "2"],
            "start": [10, 200, 400, 600, 7000, 9000, 30000, 1130],
            "end": [100, 300, 500, 700, 9100, 18000, 31000, 15000],
        })
        domain = pd.DataFrame({
            "chrom": ["1", "2"],
            "start": [0, 0],
            "end": [483000, 300000],
        })
        breaks = [50000 * i for i in range(1, 11)]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        ref = _reference_gcis_decay_slow("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, ref)
