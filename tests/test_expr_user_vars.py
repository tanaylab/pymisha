"""TDD tests for user variable resolution in pymisha string expressions.

These tests verify that Python variables from the caller's namespace can be
used inside pymisha expression strings.  For example::

    threshold = 0.5
    pm.gextract("dense_track > threshold", intervals, iterator=100)

Currently this fails because ``threshold`` is not recognized as a track or
coordinate name.  After implementation, ``sys._getframe()`` will capture the
caller's locals/globals at API entry, resolve non-track identifiers from the
caller namespace, and inject the values into the eval sandbox.

Priority order: coordinates > tracks > user vars > numpy.

An optional ``vars=`` parameter provides explicit control.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INTERVALS = pd.DataFrame(
    {"chrom": ["1", "1", "1"], "start": [0, 1000, 2000], "end": [1000, 2000, 3000]}
)

# Module-level variable used by test_gextract_with_module_level_var
MODULE_SCALE = 2.0


def _data_cols(df):
    """Return column names that are not coordinate/interval metadata."""
    return [c for c in df.columns if c not in {"chrom", "start", "end", "intervalID"}]


# ---------------------------------------------------------------------------
# gextract tests
# ---------------------------------------------------------------------------


class TestGextractUserVars:
    """User variable resolution in gextract expressions."""

    def test_gextract_with_module_level_var(self):
        """A module-level variable should be resolvable inside an expression."""
        result_ref = pm.gextract("dense_track * 2", INTERVALS, iterator=1000)
        result = pm.gextract("dense_track * MODULE_SCALE", INTERVALS, iterator=1000)
        assert result is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        np.testing.assert_array_equal(result[res_col].values, result_ref[ref_col].values)

    def test_gextract_with_function_local_var(self):
        """A variable defined in the calling function's local scope must be
        resolved.  This is the key bug case: locals are not visible unless
        ``sys._getframe()`` captures the caller's frame."""
        local_offset = 10.0
        result_ref = pm.gextract("dense_track + 10", INTERVALS, iterator=1000)
        result = pm.gextract("dense_track + local_offset", INTERVALS, iterator=1000)
        assert result is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        np.testing.assert_array_equal(result[res_col].values, result_ref[ref_col].values)

    def test_gextract_with_closure_var(self):
        """A variable from an enclosing (non-local, non-global) scope should
        be resolvable — i.e. closure / free variables."""
        factor = 3.0

        def _inner():
            result_ref = pm.gextract("dense_track * 3", INTERVALS, iterator=1000)
            result = pm.gextract("dense_track * factor", INTERVALS, iterator=1000)
            assert result is not None
            ref_col = _data_cols(result_ref)[0]
            res_col = _data_cols(result)[0]
            np.testing.assert_array_equal(
                result[res_col].values, result_ref[ref_col].values
            )

        _inner()

    def test_gextract_with_explicit_vars(self):
        """When ``vars=`` is provided, variables should be taken from that
        dict instead of (or in addition to) the caller's namespace."""
        result_ref = pm.gextract("dense_track + 42", INTERVALS, iterator=1000)
        result = pm.gextract(
            "dense_track + magic", INTERVALS, iterator=1000, vars={"magic": 42}
        )
        assert result is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        np.testing.assert_array_equal(result[res_col].values, result_ref[ref_col].values)

    def test_gextract_user_var_numeric_constant(self):
        """A numeric threshold variable should work in a comparison expression."""
        threshold = 0.5
        result = pm.gextract("dense_track > threshold", INTERVALS, iterator=1000)
        result_ref = pm.gextract("dense_track > 0.5", INTERVALS, iterator=1000)
        assert result is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        np.testing.assert_array_equal(result[res_col].values, result_ref[ref_col].values)

    def test_gextract_user_var_arithmetic(self):
        """Multiple user variables in a single arithmetic expression."""
        scale = 2.0
        offset = 5.0
        result = pm.gextract(
            "dense_track * scale + offset", INTERVALS, iterator=1000
        )
        result_ref = pm.gextract("dense_track * 2 + 5", INTERVALS, iterator=1000)
        assert result is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        np.testing.assert_array_equal(result[res_col].values, result_ref[ref_col].values)


# ---------------------------------------------------------------------------
# gscreen tests
# ---------------------------------------------------------------------------


class TestGscreenUserVars:
    """User variable resolution in gscreen expressions."""

    def test_gscreen_with_user_var(self):
        """A user variable should work inside a gscreen filter expression."""
        cutoff = 0.5
        result = pm.gscreen(
            "dense_track > cutoff", intervals=INTERVALS, iterator=1000
        )
        result_ref = pm.gscreen(
            "dense_track > 0.5", intervals=INTERVALS, iterator=1000
        )
        # Both should return a DataFrame (or both None)
        if result_ref is None:
            assert result is None
        else:
            assert result is not None
            pd.testing.assert_frame_equal(result, result_ref)


# ---------------------------------------------------------------------------
# gdist tests
# ---------------------------------------------------------------------------


class TestGdistUserVars:
    """User variable resolution in gdist expressions."""

    def test_gdist_with_user_var(self):
        """A user variable in a gdist expression should be resolved from the
        caller's namespace."""
        scale = 2.0
        result = pm.gdist(
            "dense_track * scale", [0, 0.5, 1.0, 2.0],
            intervals=INTERVALS, iterator=1000,
        )
        result_ref = pm.gdist(
            "dense_track * 2", [0, 0.5, 1.0, 2.0],
            intervals=INTERVALS, iterator=1000,
        )
        np.testing.assert_array_equal(result, result_ref)


# ---------------------------------------------------------------------------
# gsummary tests
# ---------------------------------------------------------------------------


class TestGsummaryUserVars:
    """User variable resolution in gsummary expressions."""

    def test_gsummary_with_user_var(self):
        """A user variable in a gsummary expression should produce the same
        result as an inlined literal."""
        factor = 3.0
        result = pm.gsummary(
            "dense_track * factor", intervals=INTERVALS, iterator=1000,
        )
        result_ref = pm.gsummary(
            "dense_track * 3", intervals=INTERVALS, iterator=1000,
        )
        assert result is not None and result_ref is not None
        # gsummary returns a pandas Series; compare all entries by index
        for key in result_ref.index:
            assert key in result.index, f"Missing key {key!r} in result"
            if isinstance(result_ref[key], float):
                np.testing.assert_allclose(
                    result[key], result_ref[key], rtol=1e-10,
                    err_msg=f"Mismatch for key {key!r}",
                )
            else:
                assert result[key] == result_ref[key], f"Mismatch for key {key!r}"


# ---------------------------------------------------------------------------
# gquantiles tests
# ---------------------------------------------------------------------------


class TestGquantilesUserVars:
    """User variable resolution in gquantiles expressions."""

    def test_gquantiles_with_user_var(self):
        """A user variable in a gquantiles expression should be resolved."""
        multiplier = 2.0
        result = pm.gquantiles(
            "dense_track * multiplier", percentiles=[0.25, 0.5, 0.75],
            intervals=INTERVALS, iterator=1000,
        )
        result_ref = pm.gquantiles(
            "dense_track * 2", percentiles=[0.25, 0.5, 0.75],
            intervals=INTERVALS, iterator=1000,
        )
        np.testing.assert_allclose(result, result_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# Virtual track + user variable tests
# ---------------------------------------------------------------------------


class TestUserVarWithVtracks:
    """User variables combined with virtual tracks."""

    def test_gextract_user_var_with_vtrack(self):
        """A user variable should work in an expression alongside a vtrack."""
        pm.gvtrack_create("vt_avg", "dense_track", func="avg")
        try:
            scale = 2.0
            result = pm.gextract(
                "vt_avg * scale", INTERVALS, iterator=1000
            )
            result_ref = pm.gextract("vt_avg * 2", INTERVALS, iterator=1000)
            assert result is not None
            ref_col = _data_cols(result_ref)[0]
            res_col = _data_cols(result)[0]
            np.testing.assert_array_equal(
                result[res_col].values, result_ref[ref_col].values
            )
        finally:
            pm.gvtrack_rm("vt_avg")


# ---------------------------------------------------------------------------
# Priority / shadowing tests
# ---------------------------------------------------------------------------


class TestUserVarPriority:
    """Verify that user variables do not shadow tracks or coordinates."""

    def test_user_var_does_not_shadow_track(self):
        """If a user variable has the same name as a track, the track must
        win (track names take priority over user variables)."""
        dense_track = 999.0  # noqa: F841  -- deliberately shadows
        result = pm.gextract("dense_track", INTERVALS, iterator=1000)
        assert result is not None
        col = _data_cols(result)[0]
        # If the track name won, we should get actual track data, not 999.0
        # (at least some values should differ from 999.0 or be NaN)
        vals = result[col].dropna().values
        if len(vals) > 0:
            assert not np.all(vals == 999.0), (
                "User variable shadowed the track name — track should take priority"
            )

    def test_user_var_does_not_shadow_coordinates(self):
        """CHROM, START, END are built-in coordinate variables and must not be
        overridden by user variables of the same name."""
        START = 12345.0  # noqa: F841
        END = 99999.0  # noqa: F841
        result = pm.gextract("START", INTERVALS, iterator=1000)
        assert result is not None
        col = _data_cols(result)[0]
        # START should reflect actual interval start values, not 12345.0
        starts = result[col].values
        # The first interval starts at 0, not 12345
        assert starts[0] != 12345.0, (
            "User variable shadowed the START coordinate"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestUserVarErrors:
    """Error handling for unresolvable variables."""

    def test_unknown_var_raises_error(self):
        """An expression containing an identifier that is not a track,
        coordinate, numpy name, or user variable should raise a clear error."""
        with pytest.raises((ValueError, NameError)):
            pm.gextract(
                "dense_track + completely_unknown_xyz_42",
                INTERVALS,
                iterator=1000,
            )


# ---------------------------------------------------------------------------
# Numpy integration
# ---------------------------------------------------------------------------


class TestUserVarNumpy:
    """User variables that are numpy arrays or used with numpy operations."""

    def test_user_var_numpy_array(self):
        """A numpy array user variable should be injectable into an expression."""
        weights = np.array([1.0, 2.0, 3.0])
        # This tests that a numpy array can be passed via vars= and used in an
        # expression.  The exact semantics depend on broadcasting.
        result = pm.gextract(
            "dense_track * weights",
            INTERVALS,
            iterator=1000,
            vars={"weights": weights},
        )
        result_ref = pm.gextract("dense_track", INTERVALS, iterator=1000)
        assert result is not None and result_ref is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        ref_vals = result_ref[ref_col].values
        res_vals = result[res_col].values
        expected = ref_vals * weights
        np.testing.assert_array_equal(res_vals, expected)

    def test_user_var_with_numpy_ops(self):
        """User variables combined with numpy functions (np.log, np.abs)
        should work in expressions."""
        base_val = 1.0
        result = pm.gextract(
            "np.abs(dense_track) + base_val",
            INTERVALS,
            iterator=1000,
        )
        result_ref = pm.gextract(
            "np.abs(dense_track) + 1",
            INTERVALS,
            iterator=1000,
        )
        assert result is not None
        ref_col = _data_cols(result_ref)[0]
        res_col = _data_cols(result)[0]
        np.testing.assert_array_equal(result[res_col].values, result_ref[ref_col].values)
