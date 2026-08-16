"""Regression tests for 2D interval coordinate validation.

R misha validates every 2D interval it reads via ``GInterval2D::verify``
(misha/src/GInterval2D.h, called from misha/src/IntervalConverter.cpp:418).
pymisha's C++ extension never calls the equivalent check for 2D data: the
raw-DataFrame path built its C++ argument dict straight from the columns
with no validation at all. Negative, NaN, and past-the-chromosome
coordinates all passed through silently, and an inverted rectangle
(``start1 > end1``) returned ``None`` -- indistinguishable from "no
contacts in this region" to a caller.

Mirrors the 1D coordinate-validation regression suite in
``test_interval_coord_validation.py``, extended to the 2D entry points:
``gextract``/``_gextract_2d_single`` (via ``rects_track``), ``gscreen``'s
2D branch, and the ``gintervals_2d_*`` set operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pymisha as pm
from pymisha.extract import _gextract_2d_single

# chromosome "1" of the test DB is 500000 bp; "2" is 300000 bp.


def _2d(chrom1, start1, end1, chrom2=None, start2=None, end2=None):
    """A single-row 2D interval DataFrame. Defaults chrom2/start2/end2 to
    the axis-1 values (a square on the diagonal) when omitted."""
    if chrom2 is None:
        chrom2, start2, end2 = chrom1, start1, end1
    return pd.DataFrame({
        "chrom1": [chrom1], "start1": [start1], "end1": [end1],
        "chrom2": [chrom2], "start2": [start2], "end2": [end2],
    })


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


# ──────────────────────────────────────────────────────────────────
# gextract / _gextract_2d_single (the raw-DataFrame extraction path)
# ──────────────────────────────────────────────────────────────────

class TestGextract2dCoordValidation:
    def test_negative_start_raises(self, rects_track):
        iv = _2d("1", -100_000, 200_000, "1", 0, 200_000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gextract(rects_track, intervals=iv)

    def test_nan_start_raises(self, rects_track):
        iv = pd.DataFrame({
            "chrom1": ["1"], "start1": [np.nan], "end1": [200_000.0],
            "chrom2": ["1"], "start2": [0.0], "end2": [200_000.0],
        })
        with pytest.raises(ValueError, match="missing"):
            pm.gextract(rects_track, intervals=iv)

    def test_inverted_raises_instead_of_silent_none(self, rects_track):
        """start1 > end1 used to return None -- indistinguishable from
        'no contacts in this region'. It must raise instead."""
        iv = _2d("1", 400_000, 1_000, "1", 0, 500_000)
        with pytest.raises(ValueError, match="start coordinate must be lesser than end"):
            pm.gextract(rects_track, intervals=iv)

    def test_beyond_chromosome_raises(self, rects_track):
        iv = _2d("1", 0, 1_000_000_000, "1", 0, 1_000_000_000)
        with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
            pm.gextract(rects_track, intervals=iv)

    def test_direct_call_to_gextract_2d_single_validates(self, rects_track):
        """_gextract_2d_single is called directly elsewhere in the test
        suite (test_extract_2d_fast.py), bypassing gextract/_gextract_2d.
        It must validate on its own."""
        iv = _2d("1", -100_000, 200_000, "1", 0, 200_000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            _gextract_2d_single(rects_track, "v", iv, None)

    def test_valid_query_still_works(self, rects_track):
        """Non-regression: an ordinary in-bounds query still extracts rows."""
        iv = _2d("1", 0, 500_000, "1", 0, 500_000)
        result = pm.gextract(rects_track, intervals=iv)
        assert result is not None
        assert len(result) == 3  # R1, R2, R3 on the (1, 1) chrom pair

    def test_scope_is_validated_once_per_extraction(self, rects_track, monkeypatch):
        """The scope must not be re-validated by every track in the expression.

        _gextract_2d validates the scope and then hands the same object to
        _gextract_2d_single per required track, which used to validate it
        again. The check costs ~0.27 ms per call regardless of row count, and
        a streamed 2D job runs it once per input row.
        """
        import pymisha.intervals as pmi

        seen = []
        orig = pmi._verify_2d_intervals

        def _counting(df):
            seen.append(df)
            return orig(df)

        monkeypatch.setattr(pmi, "_verify_2d_intervals", _counting)
        iv = _2d("1", 0, 500_000, "1", 0, 500_000)
        assert pm.gextract(rects_track, intervals=iv) is not None
        assert len(seen) == 1, f"scope validated {len(seen)} times, expected once"

    def test_raw_dataframe_iterator_validates(self, rects_track):
        """A raw 2D DataFrame passed as ``iterator=`` (as opposed to a saved
        interval set or track name) is intersected against the scope without
        going through _gextract_2d_single; it needs its own check."""
        scope = _2d("1", 0, 500_000, "1", 0, 500_000)
        bad_iterator = _2d("1", -100_000, 200_000, "1", 0, 200_000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gextract(rects_track, intervals=scope, iterator=bad_iterator)


# ──────────────────────────────────────────────────────────────────
# Exception type: the 1D and 2D checks do not agree
# ──────────────────────────────────────────────────────────────────

class TestCoordValidationExceptionType:
    """The 1D checks live in C++ and raise ``pymisha.error``; the 2D checks
    live in Python and raise ``ValueError``. ``pymisha.error`` is a bare
    ``Exception`` subclass (src/pymisha_init.cpp), so it is *not* a
    ``ValueError`` and ``except ValueError`` will not catch the 1D case.

    Pinned deliberately: unifying the hierarchy is an API decision, not part
    of this fix. This test exists so the split is visible rather than
    accidental.
    """

    def test_2d_raises_value_error(self, rects_track):
        iv = _2d("1", -100_000, 200_000, "1", 0, 200_000)
        with pytest.raises(ValueError):
            pm.gextract(rects_track, intervals=iv)

    def test_1d_raises_pymisha_error_not_value_error(self):
        import _pymisha

        iv = pd.DataFrame({"chrom": ["1"], "start": [-100], "end": [1000]})
        with pytest.raises(_pymisha.error) as exc:
            pm.gextract("dense_track", intervals=iv)
        assert not isinstance(exc.value, ValueError)


# ──────────────────────────────────────────────────────────────────
# gscreen's 2D branch
# ──────────────────────────────────────────────────────────────────

class TestGscreen2dCoordValidation:
    def test_negative_coords_raise(self, rects_track):
        iv = _2d("1", -100_000, 200_000, "1", 0, 200_000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gscreen(f"{rects_track} > 0", intervals=iv)


# ──────────────────────────────────────────────────────────────────
# gintervals_save: the only 2D writer, and the one that persists
# ──────────────────────────────────────────────────────────────────

class TestGintervalsSave2dCoordValidation:
    """A bad query raises where the mistake was made; a bad *save* wrote the
    bad coordinates to disk, where the next reader (possibly in another
    process, possibly R) inherits them. gintervals_save is the only 2D
    interval-set writer, and gscreen(..., intervals_set_out=) reaches it.
    """

    def test_negative_start_rejected(self):
        iv = _2d("1", -100, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gintervals_save(iv, "test_save_2d_neg")
        assert not pm.gintervals_exists("test_save_2d_neg")

    def test_inverted_rectangle_rejected(self):
        iv = _2d("1", 400_000, 1_000, "1", 0, 1000)
        with pytest.raises(ValueError, match="start coordinate must be lesser than end"):
            pm.gintervals_save(iv, "test_save_2d_inv")
        assert not pm.gintervals_exists("test_save_2d_inv")

    def test_zero_width_rectangle_rejected(self):
        iv = _2d("1", 1000, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="start coordinate must be lesser than end"):
            pm.gintervals_save(iv, "test_save_2d_zero")
        assert not pm.gintervals_exists("test_save_2d_zero")

    def test_beyond_chromosome_rejected(self):
        iv = _2d("1", 0, 1_000_000_000, "1", 0, 1000)
        with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
            pm.gintervals_save(iv, "test_save_2d_far")
        assert not pm.gintervals_exists("test_save_2d_far")

    def test_nan_rejected(self):
        iv = pd.DataFrame({
            "chrom1": ["1"], "start1": [np.nan], "end1": [1000.0],
            "chrom2": ["1"], "start2": [0.0], "end2": [1000.0],
        })
        with pytest.raises(ValueError, match="missing"):
            pm.gintervals_save(iv, "test_save_2d_nan")
        assert not pm.gintervals_exists("test_save_2d_nan")

    def test_valid_2d_set_still_saves(self):
        """Non-regression: an in-bounds 2D set round-trips as before."""
        iv = _2d("1", 100, 1000, "1", 200, 2000)
        pm.gintervals_save(iv, "test_save_2d_ok")
        try:
            loaded = pm.gintervals_load("test_save_2d_ok")
            assert len(loaded) == 1
            assert int(loaded["start1"].iloc[0]) == 100
            assert int(loaded["end2"].iloc[0]) == 2000
        finally:
            pm.gintervals_rm("test_save_2d_ok", force=True)

    def test_gscreen_intervals_set_out_still_works(self, rects_track):
        """The reachable-from-gscreen path is unaffected for valid results."""
        scope = _2d("1", 0, 500_000, "1", 0, 500_000)
        pm.gscreen(f"{rects_track} > 0", intervals=scope, intervals_set_out="test_save_2d_screen")
        try:
            loaded = pm.gintervals_load("test_save_2d_screen")
            assert loaded is not None and len(loaded) > 0
        finally:
            pm.gintervals_rm("test_save_2d_screen", force=True)


# ──────────────────────────────────────────────────────────────────
# gintervals_2d_* set operations
# ──────────────────────────────────────────────────────────────────

class TestSetOps2dCoordValidation:
    def test_union_rejects_negative(self):
        iv1 = _2d("1", -100, 1000, "1", 0, 1000)
        iv2 = _2d("1", 0, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gintervals_2d_union(iv1, iv2)

    def test_union_skips_boundary_check_for_unresolvable_chrom(self):
        """Best-effort design: gintervals_2d_intersect/_union/_band_intersect
        are chrom-agnostic (no _checkroot() call, see test_2d_set_ops.py's
        own use of a "chr1" label the test DB doesn't register). An
        out-of-db chrom label must not make the boundary check raise --
        only NaN/negative/inverted are unconditional."""
        iv1 = _2d("not_a_real_chrom", 0, 1_000_000_000, "not_a_real_chrom", 0, 1_000_000_000)
        iv2 = _2d("not_a_real_chrom", 500, 1_500, "not_a_real_chrom", 500, 1_500)
        result = pm.gintervals_2d_union(iv1, iv2)
        assert result is not None
        assert len(result) == 2

    def test_union_of_valid_frames_still_works(self):
        """Non-regression sanity: a valid union still works."""
        iv1 = _2d("1", 0, 1000, "1", 0, 1000)
        iv2 = _2d("1", 500, 1500, "1", 500, 1500)
        result = pm.gintervals_2d_union(iv1, iv2)
        assert result is not None
        assert len(result) == 2

    def test_intersect_rejects_negative(self):
        iv1 = _2d("1", -100, 1000, "1", 0, 1000)
        iv2 = _2d("1", 0, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gintervals_2d_intersect(iv1, iv2)

    def test_intersect_rejects_inverted(self):
        iv1 = _2d("1", 1000, 100, "1", 0, 1000)
        iv2 = _2d("1", 0, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="start coordinate must be lesser than end"):
            pm.gintervals_2d_intersect(iv1, iv2)

    def test_intersect_rejects_nan(self):
        iv1 = pd.DataFrame({
            "chrom1": ["1"], "start1": [np.nan], "end1": [1000.0],
            "chrom2": ["1"], "start2": [0.0], "end2": [1000.0],
        })
        iv2 = _2d("1", 0, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="missing"):
            pm.gintervals_2d_intersect(iv1, iv2)

    def test_band_intersect_rejects_negative(self):
        iv = _2d("1", -100, 1000, "1", 0, 1000)
        with pytest.raises(ValueError, match="must be greater or equal than zero"):
            pm.gintervals_2d_band_intersect(iv, (0, 100))


# ──────────────────────────────────────────────────────────────────
# Chrom-size cache: must be invalidated on a genome-root switch
# ──────────────────────────────────────────────────────────────────

class TestChromSizesCacheInvalidation:
    """_verify_2d_intervals's past-the-chromosome check memoizes the active
    root's chrom-size dict for performance (a streaming job calls it once
    per input interval). That memo must not survive a switch to a
    different root with different chrom sizes."""

    def test_boundary_check_follows_a_root_switch(self, tmp_path):
        import shutil

        from _dbpath import TESTDB_ROOT

        # A copy of the test DB with chrom "1" widened from 500000 to
        # 999000, so a query the original DB rejects is valid here.
        alt_root = tmp_path / "alt_test_db"
        shutil.copytree(str(TESTDB_ROOT), alt_root)
        cs_path = alt_root / "chrom_sizes.txt"
        cs_path.write_text(cs_path.read_text().replace("1\t500000", "1\t999000"))

        iv = _2d("1", 0, 600_000, "1", 0, 1000)  # beyond 500000, within 999000

        pm.gdb_init(str(TESTDB_ROOT))
        with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
            pm.gintervals_2d_union(iv, iv)

        pm.gdb_init(str(alt_root))
        # Must NOT raise: the cache must have picked up the new chrom size,
        # not served the previous root's stale 500000 boundary.
        result = pm.gintervals_2d_union(iv, iv)
        assert result is not None

        pm.gdb_init(str(TESTDB_ROOT))
        # And switching back must invalidate again, not keep serving the
        # alt root's 999000 boundary.
        with pytest.raises(ValueError, match="exceeds chromosome boundaries"):
            pm.gintervals_2d_union(iv, iv)
