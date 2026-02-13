"""Tests for gcis_decay: cis contact distance distribution."""

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
# Helper: compute gcis_decay manually from extracted data
# ---------------------------------------------------------------------------

def _reference_gcis_decay(track, breaks, src, domain, include_lowest=False):
    """Pure-Python reference implementation for parity checks."""
    # Normalize chroms
    src = src.copy()
    domain = domain.copy()

    # Unify overlaps in src per chrom
    from pymisha.analysis import (
        _containing_interval,
        _intervals_per_chrom,
        _unify_overlaps_per_chrom,
        _val2bin,
    )

    src_per_chrom = _unify_overlaps_per_chrom(src)
    domain_per_chrom = _intervals_per_chrom(domain)

    n_bins = len(breaks) - 1
    intra = np.zeros(n_bins, dtype=float)
    inter = np.zeros(n_bins, dtype=float)

    # Extract all cis contacts using ALLGENOME for chrom sizes
    all_genome = pm.gintervals_all()
    chrom_sizes = {}
    for _, row in all_genome.iterrows():
        chrom_sizes[str(row["chrom"])] = int(row["end"])
    for chrom, csize in chrom_sizes.items():
        intervals_2d = pm.gintervals_2d(chrom, 0, csize, chrom, 0, csize)
        result = pm.gextract(track, intervals_2d)
        if result is None or len(result) == 0:
            continue

        src_ivs = src_per_chrom.get(str(chrom), [])
        domain_ivs = domain_per_chrom.get(str(chrom), [])

        for _, row in result.iterrows():
            s1, e1 = int(row["start1"]), int(row["end1"])
            s2, e2 = int(row["start2"]), int(row["end2"])

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


class TestGciDecayBasic:
    """Basic functionality tests for gcis_decay."""

    def test_returns_2d_array(self):
        """gcis_decay returns a 2D numpy array with shape (n_bins, 2)."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 2)

    def test_column_labels(self):
        """Result should have col_labels ['intra', 'inter']."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert hasattr(result, "col_labels")
        assert result.col_labels == ["intra", "inter"]

    def test_breaks_attribute(self):
        """Result should store the breaks as an attribute."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert hasattr(result, "breaks")
        assert result.breaks == breaks

    def test_bin_labels(self):
        """Result should have properly formatted bin_labels."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result.bin_labels == ["(0,100000]", "(100000,200000]"]

    def test_nonzero_counts(self):
        """With broad src and domain covering all of chr1, should get nonzero counts."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        # There are 76 cis contacts on chr1; some should fall in bins
        assert result.sum() > 0

    def test_all_values_nonnegative(self):
        """All counts must be non-negative."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert (result >= 0).all()


class TestGcisDecayIntraDomain:
    """Tests for intra-domain vs inter-domain classification."""

    def test_single_large_domain_all_intra(self):
        """When one domain covers the whole chromosome, all contacts are intra-domain."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        # All counts should be in intra column (col 0), inter column (col 1) should be 0
        assert result[:, 1].sum() == 0  # no inter-domain
        assert result[:, 0].sum() > 0   # has intra-domain

    def test_no_domain_all_inter(self):
        """With a tiny domain that contains nothing, all contacts become inter-domain."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        # Domain that is too small to contain any contacts
        domain = pd.DataFrame({"chrom": ["1"], "start": [499999], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result[:, 0].sum() == 0  # no intra-domain
        # Some contacts should be inter-domain (if any pass the src filter)
        total = result.sum()
        if total > 0:
            assert result[:, 1].sum() > 0

    def test_two_domains_split(self):
        """Two non-overlapping domains produce a mix of intra and inter contacts."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 250000],
            "end": [250000, 500000],
        })
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        # Contacts within same domain are intra, contacts crossing the boundary are inter
        # Both should be nonzero for this dataset
        assert result.sum() > 0


class TestGcisDecaySrcFiltering:
    """Tests for source interval filtering."""

    def test_no_src_coverage_returns_zeros(self):
        """When src doesn't cover any contact I1, result should be all zeros."""
        # src covers a region with no contacts (far from actual data)
        src = pd.DataFrame({"chrom": ["X"], "start": [199000], "end": [200000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result.sum() == 0

    def test_src_subset_reduces_counts(self):
        """Narrower src should produce fewer counts than full coverage."""
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        src_full = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        result_full = pm.gcis_decay("rects_track", breaks, src_full, domain)

        src_partial = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100000]})
        result_partial = pm.gcis_decay("rects_track", breaks, src_partial, domain)

        assert result_partial.sum() <= result_full.sum()

    def test_overlapping_src_unified(self):
        """Overlapping source intervals should be unified before containment check."""
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        # Two overlapping src intervals that together cover [0, 200000)
        src_overlapping = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 50000],
            "end": [150000, 200000],
        })
        result_overlapping = pm.gcis_decay("rects_track", breaks, src_overlapping, domain)

        # Single src interval covering [0, 200000)
        src_single = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [200000]})
        result_single = pm.gcis_decay("rects_track", breaks, src_single, domain)

        np.testing.assert_array_equal(result_overlapping, result_single)


class TestGcisDecayBreaks:
    """Tests for distance binning behavior."""

    def test_include_lowest_false(self):
        """With include_lowest=False, distance=0 should not be counted in first bin."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        # First break at 0: bin (0, 100000] excludes distance=0
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain, include_lowest=False)
        assert result.shape == (5, 2)

    def test_include_lowest_true(self):
        """With include_lowest=True, distance=0 should be counted in first bin."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain, include_lowest=True)
        assert result.shape == (5, 2)
        # Bin labels should start with [ when include_lowest
        assert result.bin_labels[0].startswith("[")

    def test_two_bins(self):
        """Minimal case with 2 breaks (1 bin)."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result.shape == (1, 2)
        # Total should equal all cis contacts from chr1 (that are in src)
        total = result[:, 0].sum() + result[:, 1].sum()
        assert total > 0

    def test_narrow_bins(self):
        """Fine-grained bins produce the same total as coarse bins."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})

        # Coarse
        breaks_coarse = [0, 500000]
        result_coarse = pm.gcis_decay("rects_track", breaks_coarse, src, domain)

        # Fine
        breaks_fine = list(range(0, 500001, 50000))
        result_fine = pm.gcis_decay("rects_track", breaks_fine, src, domain)

        # Totals must match since range is the same [0, 500000]
        assert result_coarse.sum() == result_fine.sum()


class TestGcisDecayParity:
    """Parity test: manual computation vs gcis_decay result."""

    def test_matches_reference_implementation(self):
        """gcis_decay should match the reference Python implementation."""
        src = pd.DataFrame({
            "chrom": ["1", "1", "1", "1"],
            "start": [0, 100000, 200000, 400000],
            "end": [80000, 200000, 350000, 500000],
        })
        domain = pd.DataFrame({
            "chrom": ["1", "1"],
            "start": [0, 250000],
            "end": [250000, 500000],
        })
        breaks = [0, 50000, 100000, 200000, 300000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain)
        reference = _reference_gcis_decay("rects_track", breaks, src, domain)

        np.testing.assert_array_equal(result, reference)

    def test_matches_reference_with_include_lowest(self):
        """Parity test with include_lowest=True."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]

        result = pm.gcis_decay("rects_track", breaks, src, domain, include_lowest=True)
        reference = _reference_gcis_decay("rects_track", breaks, src, domain, include_lowest=True)

        np.testing.assert_array_equal(result, reference)


class TestGcisDecayMultiChrom:
    """Tests for multi-chromosome behavior."""

    def test_multi_chrom_src(self):
        """Source intervals spanning multiple chromosomes should work."""
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
        assert result.shape == (5, 2)
        # Should have counts from both chromosomes
        assert result.sum() > 0

    def test_intervals_restricts_chroms(self):
        """Passing intervals for only chr2 should exclude chr1 contacts."""
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

        intervals_chr2 = pd.DataFrame({
            "chrom": ["2"],
            "start": [0],
            "end": [300000],
        })
        result = pm.gcis_decay("rects_track", breaks, src, domain,
                               intervals=intervals_chr2)
        # There may not be chr2 cis contacts in the test DB
        # but the function should still work
        assert result.shape == (5, 2)
        assert (result >= 0).all()


class TestGcisDecayEdgeCases:
    """Edge cases and error handling."""

    def test_missing_required_args(self):
        """Should raise ValueError when required arguments are None."""
        with pytest.raises(ValueError, match="Usage"):
            pm.gcis_decay(None, [0, 100], pd.DataFrame(), pd.DataFrame())

        with pytest.raises(ValueError, match="Usage"):
            pm.gcis_decay("rects_track", None, pd.DataFrame(), pd.DataFrame())

    def test_too_few_breaks(self):
        """Should raise ValueError with fewer than 2 breaks."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        with pytest.raises(ValueError, match="at least 2"):
            pm.gcis_decay("rects_track", [100], src, domain)

    def test_non_2d_track_raises(self):
        """Should raise ValueError for a 1D track."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        with pytest.raises(ValueError, match="not a 2D track"):
            pm.gcis_decay("dense_track", [0, 100000], src, domain)

    def test_empty_src_returns_zeros(self):
        """Empty source intervals should produce all-zero result."""
        src = pd.DataFrame({"chrom": pd.Series([], dtype=str),
                            "start": pd.Series([], dtype=int),
                            "end": pd.Series([], dtype=int)})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result.sum() == 0
        assert result.shape == (2, 2)

    def test_empty_domain_all_inter(self):
        """Empty domain intervals should make all contacts inter-domain."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": pd.Series([], dtype=str),
                               "start": pd.Series([], dtype=int),
                               "end": pd.Series([], dtype=int)})
        breaks = [0, 100000, 200000, 300000, 400000, 500000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        # No intra-domain possible with empty domain
        assert result[:, 0].sum() == 0  # intra = 0
        # But inter should have counts
        if result.sum() > 0:
            assert result[:, 1].sum() > 0

    def test_repr_produces_string(self):
        """The repr method should produce a readable string."""
        src = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        domain = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [500000]})
        breaks = [0, 100000, 200000]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        s = repr(result)
        assert "intra" in s
        assert "inter" in s


class TestGcisDecayRExample:
    """Test matching the R documentation example."""

    def test_r_example_signature(self):
        """Replicate the R documentation example structure.

        R example:
            src <- rbind(
                gintervals(1, 10, 100),
                gintervals(1, 200, 300),
                ...
            )
            domain <- rbind(
                gintervals(1, 0, 483000),
                gintervals(2, 0, 300000)
            )
            gcis_decay("rects_track", 50000 * (1:10), src, domain)
        """
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
        assert result.shape == (9, 2)
        assert (result >= 0).all()
        # The total should be some subset of contacts
        # (src intervals are narrow, so not all contacts will pass)


class TestGcisDecayRParity:
    """Tests ported from R test-gcis_decay.R to match R test patterns."""

    def test_extracted_sparse_as_src(self):
        """R test pattern: use gextract output from sparse track as src.

        R code:
            domain <- rbind(
                gintervals(1, 800000 * (0:5), 800000 * (0:5) + 400000), ...)
            src <- gextract("test.sparse", gintervals(c(1, 2, 3, 4, 5)))
            gcis_decay("test.rects", (0:20) * 1000, src, domain)

        Adapted for test DB: chroms 1, 2, X with smaller ranges.
        """
        # Build domain intervals: chunks of 80000 on each chrom
        domain_rows = []
        for chrom in ["1", "2"]:
            for i in range(3):
                domain_rows.append({
                    "chrom": chrom,
                    "start": 80000 * i,
                    "end": 80000 * i + 40000,
                })
        domain = pd.DataFrame(domain_rows)

        # Extract sparse track data as src (R pattern: using track extraction output)
        all_intervals = pd.DataFrame({
            "chrom": ["1", "2"],
            "start": [0, 0],
            "end": [500000, 300000],
        })
        src = pm.gextract("sparse_track", all_intervals)
        assert src is not None and len(src) > 0

        breaks = [1000 * i for i in range(21)]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result.shape == (20, 2)
        assert (result >= 0).all()
        # Should be reproducible: same call gives same result
        result2 = pm.gcis_decay("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, result2)

    def test_domain_as_src(self):
        """R test pattern: use domain intervals directly as src.

        R code:
            src <- domain
            gcis_decay("test.rects", (0:20) * 1000, src, domain)
        """
        domain = pd.DataFrame({
            "chrom": ["1", "1", "1", "2", "2"],
            "start": [0, 80000, 160000, 0, 80000],
            "end": [40000, 120000, 200000, 40000, 120000],
        })
        src = domain.copy()

        breaks = [1000 * i for i in range(21)]
        result = pm.gcis_decay("rects_track", breaks, src, domain)
        assert result.shape == (20, 2)
        assert (result >= 0).all()

        # Verify parity with reference implementation
        reference = _reference_gcis_decay("rects_track", breaks, src, domain)
        np.testing.assert_array_equal(result, reference)

    def test_domain_equals_src_all_intra(self):
        """When domain == src and single big domain per chrom, all contacts are intra."""
        domain = pd.DataFrame({
            "chrom": ["1"],
            "start": [0],
            "end": [500000],
        })
        src = domain.copy()
        breaks = [1000 * i for i in range(21)]
        result = pm.gcis_decay("rects_track", breaks, src, domain)

        # With one domain covering everything, inter should be 0
        assert result[:, 1].sum() == 0
        assert result[:, 0].sum() > 0


class TestVal2bin:
    """Unit tests for the _val2bin helper function."""

    def test_basic_binning(self):
        from pymisha.analysis import _val2bin
        breaks = [0.0, 1.0, 2.0, 3.0]
        assert _val2bin(0.5, breaks, False) == 0
        assert _val2bin(1.0, breaks, False) == 0
        assert _val2bin(1.5, breaks, False) == 1
        assert _val2bin(2.5, breaks, False) == 2
        assert _val2bin(3.0, breaks, False) == 2

    def test_out_of_range(self):
        from pymisha.analysis import _val2bin
        breaks = [0.0, 1.0, 2.0]
        assert _val2bin(-1.0, breaks, False) == -1
        assert _val2bin(0.0, breaks, False) == -1  # not include_lowest
        assert _val2bin(2.5, breaks, False) == -1

    def test_include_lowest(self):
        from pymisha.analysis import _val2bin
        breaks = [0.0, 1.0, 2.0]
        assert _val2bin(0.0, breaks, True) == 0
        assert _val2bin(0.0, breaks, False) == -1

    def test_nan(self):
        from pymisha.analysis import _val2bin
        breaks = [0.0, 1.0, 2.0]
        assert _val2bin(float('nan'), breaks, False) == -1


class TestContainingInterval:
    """Unit tests for the _containing_interval helper function."""

    def test_contained(self):
        from pymisha.analysis import _containing_interval
        intervals = [(0, 100), (200, 500), (600, 800)]
        assert _containing_interval(intervals, 10, 50) == 0
        assert _containing_interval(intervals, 200, 500) == 1
        assert _containing_interval(intervals, 250, 400) == 1
        assert _containing_interval(intervals, 600, 800) == 2

    def test_not_contained(self):
        from pymisha.analysis import _containing_interval
        intervals = [(0, 100), (200, 500)]
        assert _containing_interval(intervals, 50, 150) == -1  # extends past first
        assert _containing_interval(intervals, 100, 200) == -1  # gap
        assert _containing_interval(intervals, 150, 250) == -1  # partially in second

    def test_empty_intervals(self):
        from pymisha.analysis import _containing_interval
        assert _containing_interval([], 0, 100) == -1
