"""Tests for PWM spatial weighting and bidirectional spatial sliding.

Ported from R misha tests:
- tests/testthat/test-pwm-spatial.R (308 lines, ~9 tests)
- tests/testthat/test-pwm-count-spatial-bidirect.R (37 lines, 1 test)

Validates spatial PWM parameters (spat_factor, spat_bin, spat_min, spat_max)
across all PWM modes (pwm, pwm.max, pwm.max.pos, pwm.count) including
bidirectional counting with spatial sliding.
"""

import numpy as np
import pytest

import pymisha as pm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remove_all_vtracks():
    """Remove all virtual tracks."""
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def _create_test_pssm():
    """Create a simple 2-position PSSM matching 'AC' exactly (matches R helper)."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # Only A
        [0.0, 1.0, 0.0, 0.0],  # Only C
    ])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPwmSpatialBasic:
    """Basic spatial PWM tests for each mode."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_pwm_with_spatial_parameters(self):
        """pwm with spatial parameters (higher weight in middle) returns valid score."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        spat_factors = [0.5, 1.0, 2.0, 1.0, 0.5]
        spat_bin = 20

        pm.gvtrack_create(
            "pwm_spatial", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=spat_factors, spat_bin=spat_bin,
        )
        scores = pm.gextract("pwm_spatial", test_interval, iterator=-1)

        assert scores["pwm_spatial"].dtype == np.float64
        assert not np.isnan(scores["pwm_spatial"].iloc[0])
        assert np.isfinite(scores["pwm_spatial"].iloc[0])

    def test_pwm_max_with_spatial_parameters(self):
        """pwm.max with spatial parameters returns valid score."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        pm.gvtrack_create(
            "pwm_max_spatial", None, func="pwm.max",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5], spat_bin=20,
        )
        scores = pm.gextract("pwm_max_spatial", test_interval, iterator=-1)

        assert not np.isnan(scores["pwm_max_spatial"].iloc[0])

    def test_pwm_max_pos_with_spatial_parameters(self):
        """pwm.max.pos with spatial parameters returns a nonzero position."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        pm.gvtrack_create(
            "pwm_maxpos_spatial", None, func="pwm.max.pos",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=[0.5, 1.0, 2.0, 1.0, 0.5], spat_bin=20,
        )
        scores = pm.gextract("pwm_maxpos_spatial", test_interval, iterator=-1)

        assert abs(scores["pwm_maxpos_spatial"].iloc[0]) > 0


class TestPwmSpatialBackwardCompat:
    """Backward compatibility: no spatial params equals explicit no-spatial."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_no_spatial_backward_compat(self):
        """Two identical vtracks without spatial produce identical results."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 240)

        pm.gvtrack_create(
            "pwm_old", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
        )
        pm.gvtrack_create(
            "pwm_new", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
        )

        scores = pm.gextract(
            ["pwm_old", "pwm_new"], test_interval, iterator=-1,
        )
        np.testing.assert_allclose(
            scores["pwm_old"].values,
            scores["pwm_new"].values,
            atol=1e-10,
        )


class TestPwmSpatialUniformWeights:
    """Uniform spatial factors (all 1.0) should match no-spatial baseline."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_uniform_spatial_equals_no_spatial(self):
        """Uniform spatial factors (all 1.0) give same result as no spatial."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 250)

        pm.gvtrack_create(
            "pwm_nospatial", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
        )
        pm.gvtrack_create(
            "pwm_uniform", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=[1.0] * 10, spat_bin=10,
        )

        scores = pm.gextract(
            ["pwm_nospatial", "pwm_uniform"], test_interval, iterator=-1,
        )
        np.testing.assert_allclose(
            scores["pwm_nospatial"].values,
            scores["pwm_uniform"].values,
            atol=1e-5,
        )


class TestPwmSpatialRange:
    """Spatial range parameters (spat_min, spat_max)."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_spatial_range_parameters(self):
        """spat_min and spat_max restrict spatial weighting range.

        spat_min = 1: R's test used 0 until R PR #206, which made anything
        below 1 an error (see test_spat_bounds_below_1_are_refused)."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 400)

        pm.gvtrack_create(
            "pwm_spatial_range", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=[0.5, 1.0, 2.0], spat_bin=50,
            spat_min=1, spat_max=100,
        )
        scores = pm.gextract("pwm_spatial_range", test_interval, iterator=test_interval)

        assert scores["pwm_spatial_range"].dtype == np.float64
        assert not np.isnan(scores["pwm_spatial_range"].iloc[0])

    @pytest.mark.parametrize("key", ["spat_min", "spat_max"])
    @pytest.mark.parametrize("bad", [0, -5, 0.5, "30", float("nan"), [30, 40], True])
    def test_a_bound_that_is_not_one_position_is_refused(self, key, bad):
        # R's check, in both gvtrack.create and gseq.pwm: a single number >= 1.
        # spat_min = 0 used to start the scan one base before the sequence.
        msg = f"{key} must be a single 1-based position"
        with pytest.raises(ValueError, match=msg):
            pm.gvtrack_create("v", None, "pwm", pssm=_create_test_pssm(), **{key: bad})
        with pytest.raises(ValueError, match=msg):
            pm.gseq_pwm("ACGTACGT", _create_test_pssm(), **{key: bad})

    @pytest.mark.parametrize("key", ["spat_min", "spat_max"])
    def test_a_bound_past_the_32_bit_range_is_refused(self, key):
        # R's as.integer() makes it NA, with a warning, and scores on.
        with pytest.raises(ValueError, match=f"{key} is outside the 32-bit"):
            pm.gvtrack_create("v", None, "pwm", pssm=_create_test_pssm(), **{key: 1e10})

    def test_the_bounds_are_stored_converted_as_in_r(self):
        # R's .vtrack_params_pwm: spat_min - 1, and (spat_max - 1) - nrow + 1,
        # truncated like as.integer(). gvtrack.info() shows the same numbers.
        pssm = np.full((8, 4), 0.25)
        pm.gvtrack_create("v", None, "pwm.max", pssm=pssm, spat_min=30.7, spat_max=470.5)
        info = pm.gvtrack_info("v")
        assert (info["spat_min"], info["spat_max"]) == (29, 462)

    def test_edit_distance_funcs_take_no_spatial_bounds(self):
        # Not in R's key set for that family either.
        with pytest.raises(ValueError, match="spat_min"):
            pm.gvtrack_create("v", None, "pwm.edit_distance", pssm=_create_test_pssm(),
                              score_thresh=-3.0, spat_min=2)

    @pytest.mark.parametrize("spatial", [False, True])
    @pytest.mark.parametrize("bounds", ["both", "min", "max"])
    @pytest.mark.parametrize("strand", ["bidirect", "plus", "minus"])
    def test_a_range_scores_like_the_window_it_selects(self, strand, bounds, spatial):
        """spat_min/spat_max keep the anchors whose motif lies in positions
        spat_min..spat_max (1-based) of the interval, with or without
        spat_factor and with either bound alone: the same numbers as scoring
        that window on its own. On the minus strand the scan runs over the
        reverse complement, so the positions count from the interval's 3' end,
        as in R. *.pos reports from the interval start, so it is shifted."""
        rng = np.random.RandomState(60427)
        pssm = rng.random((8, 4))
        pssm /= pssm.sum(axis=1, keepdims=True)
        s, e, lo, hi = 10000, 10500, 30, 470
        kw = {"pssm": pssm, "prior": 0.01, "extend": False}
        kw.update({"bidirect": {"bidirect": True},
                   "plus": {"bidirect": False, "strand": 1},
                   "minus": {"bidirect": False, "strand": -1}}[strand])
        if spatial:
            kw.update(spat_factor=[0.5, 1.0, 2.0, 2.5, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 2.0, 1.5],
                      spat_bin=40)
        rkw = {"both": {"spat_min": lo, "spat_max": hi}, "min": {"spat_min": lo},
               "max": {"spat_max": hi}}[bounds]
        first = rkw.get("spat_min", 1)
        last = rkw.get("spat_max", e - s)
        window = (e - last, e - first + 1) if strand == "minus" else (s + first - 1, s + last)

        iv = pm.gintervals("1", s, e)
        sub = pm.gintervals("1", *window)
        assert window != (s, e), "the range must actually trim something"
        for func in ("pwm", "pwm.max", "pwm.max.pos", "pwm.count"):
            fkw = dict(kw, score_thresh=-9.0) if func == "pwm.count" else kw
            pm.gvtrack_create("ranged", None, func, **fkw, **rkw)
            pm.gvtrack_create("plain", None, func, **fkw)
            got = pm.gextract("ranged", iv, iterator=iv)["ranged"].iloc[0]
            want = pm.gextract("plain", sub, iterator=sub)["plain"].iloc[0]
            if func == "pwm.max.pos":
                want = np.sign(want) * (abs(want) + (window[0] - s))
            assert np.isfinite(want), func
            assert got == want, func


def _gap_thresh(pssm, ivs):
    """A pwm.count threshold in the widest gap between per-anchor scores near
    the top 10%, so no anchor sits on it."""
    pm.gvtrack_create("anchor_scores", None, "pwm.max", pssm=pssm, prior=0.01)
    v = pm.gextract("anchor_scores", ivs, iterator=1)["anchor_scores"].to_numpy()
    pm.gvtrack_rm("anchor_scores")
    v = np.unique(v[np.isfinite(v)])
    q = int(np.searchsorted(v, np.quantile(v, 0.9)))
    k = q - 50 + int(np.argmax(np.diff(v[q - 50:q + 50])))
    assert v[k + 1] - v[k] > 1e-3
    return float((v[k] + v[k + 1]) / 2)


def _rand_pssm(nrow=8):
    rng = np.random.RandomState(60427)
    m = rng.random((nrow, 4))
    return m / m.sum(axis=1, keepdims=True)


class TestPwmSpatialRangeEdges:
    """R misha PR #206 (test-pwm-spatial.R): what a spat_min/spat_max window
    does at its edges, in the virtual tracks and in gseq_pwm."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    @pytest.mark.parametrize("spatial", [False, True])
    @pytest.mark.parametrize("strand", ["bidirect", "plus", "minus"])
    def test_pwm_max_pos_with_spat_factor_is_nan_when_no_anchor_can_be_scored(self, strand, spatial):
        # prior = 0 makes every anchor that is not this exact word score -inf
        word = "GATTACAGATTACA"
        pssm = np.zeros((len(word), 4))
        for i, base in enumerate(word):
            pssm[i, "ACGT".index(base)] = 1.0
        iv = pm.gintervals("1", 1000, 3000)
        seq = pm.gseq_extract(iv)[0].upper()
        assert "GATTACAGATTACA" not in seq and "TGTAATCTGTAATC" not in seq
        kw = {"bidirect": {"bidirect": True}, "plus": {"bidirect": False, "strand": 1},
              "minus": {"bidirect": False, "strand": -1}}[strand]
        if spatial:
            kw.update(spat_factor=[1.0, 2.0], spat_bin=500)
        pm.gvtrack_create("nopos", None, "pwm.max.pos", pssm=pssm, prior=0, **kw)
        # 500 bp bins: the first is seeded by the sliding cache, the rest are
        # too far apart to slide and take the full spatial scan
        r = pm.gextract("nopos", intervals=iv, iterator=500)
        assert len(r) == 4
        assert np.isnan(r["nopos"].to_numpy(dtype=float)).all()

    def test_spat_min_and_spat_max_below_1_are_an_error(self):
        pssm = _create_test_pssm()
        for key in ("spat_min", "spat_max"):
            with pytest.raises(ValueError, match=key):
                pm.gvtrack_create("v", None, "pwm", pssm=pssm, spat_factor=[1.0], **{key: 0})
            with pytest.raises(ValueError, match=key):
                pm.gseq_pwm("ACGTACGT", pssm, **{key: 0})

    def test_gseq_pwm_spat_bounds_match_the_pwm_virtual_tracks(self):
        pssm = _create_test_pssm()
        w = pssm.shape[0]
        starts = [1000, 5000, 9000]
        ivs = pm.gintervals(["1", "1", "1"], starts, [s + 300 for s in starts])
        # extend = True pads the end by w - 1
        seqs = [s.upper() for s in pm.gseq_extract(
            pm.gintervals(["1", "1", "1"], starts, [s + 300 + w - 1 for s in starts]))]
        sp = {"spat_factor": [0.5, 1, 2, 1, 0.5], "spat_bin": 40, "spat_min": 30, "spat_max": 270}
        for func, mode in (("pwm", "lse"), ("pwm.max", "max"), ("pwm.max.pos", "pos")):
            pm.gvtrack_create("v", None, func, pssm=pssm, bidirect=True, prior=0.01, **sp)
            vt = pm.gextract("v", ivs, iterator=ivs).sort_values("start")["v"].to_numpy()
            gs = pm.gseq_pwm(seqs, pssm, mode=mode, bidirect=True, prior=0.01, end_pos=300,
                             extend=True, **sp)
            if func == "pwm.max.pos":
                # the vtrack reports the anchor's 1-based offset from the interval start
                np.testing.assert_array_equal(np.abs(vt), gs)
            else:
                np.testing.assert_allclose(vt, gs, rtol=1e-5)
        # spat_min past the last window start: no window, with or without gap skipping
        # (R misha eecf6959; R crashed with skip_gaps = TRUE)
        seq = "ACGTACGTACGTACGTACGT"
        for g in (True, False):
            for m in ("lse", "max", "pos"):
                assert np.isnan(pm.gseq_pwm(seq, pssm, mode=m, spat_min=25, skip_gaps=g)).all(), (m, g)
            np.testing.assert_array_equal(
                pm.gseq_pwm(seq, pssm, mode="count", score_thresh=-1, spat_min=25, skip_gaps=g), [0])
            assert np.isnan(pm.gseq_pwm(seq, pssm, start_pos=5, spat_min=18, skip_gaps=g)).all(), g
        # the window [spat_min, spat_max] counts from start_pos
        s = seqs[0]
        kw = {"spat_factor": [1, 3], "spat_bin": 50}
        np.testing.assert_allclose(
            pm.gseq_pwm(s, pssm, mode="lse", start_pos=11, spat_min=20, spat_max=200, **kw),
            pm.gseq_pwm(s[29:210], pssm, mode="lse", **kw),
            rtol=1e-12,
        )

    def test_a_spat_window_shorter_than_the_motif_is_an_error(self):
        pssm8 = np.vstack([_create_test_pssm()] * 4)
        msg = "shorter than the motif"
        for kw in ({"spat_min": 300, "spat_max": 200}, {"spat_max": 7},
                   {"spat_min": 10, "spat_max": 16}):
            with pytest.raises(ValueError, match=msg):
                pm.gvtrack_create("v", None, "pwm", pssm=pssm8, **kw)
        pm.gvtrack_create("v", None, "pwm", pssm=pssm8, spat_min=10, spat_max=17)
        with pytest.raises(ValueError, match=msg):
            pm.gseq_pwm("ACGTACGTACGT", pssm8, spat_max=7)
        with pytest.raises(ValueError, match=msg):
            pm.gseq_pwm("ACGTACGTACGT", pssm8, spat_min=5, spat_max=4)

    @pytest.mark.parametrize("spatial", [False, True])
    def test_pwm_vtracks_with_spat_bounds_score_exactly_the_windows_inside_them(self, spatial):
        # R uses set.seed(1); any graded PSSM shows the same thing.
        pssm = _rand_pssm()
        iv = pm.gintervals("1", 1000, 3000)
        # [101, 300] of each 500 bp bin = the anchors of [s + 100, s + 300)
        # without extension
        inner = pm.gintervals(["1"] * 4, [1100, 1600, 2100, 2600], [1300, 1800, 2300, 2800])
        thresh = _gap_thresh(pssm, iv)
        for func in ("pwm", "pwm.max", "pwm.max.pos", "pwm.count"):
            a = {"pssm": pssm, "prior": 0.01, "spat_min": 101, "spat_max": 300}
            b = {"pssm": pssm, "prior": 0.01, "extend": False}
            # flat weights, so the spatial path must give the plain values
            if spatial:
                a.update(spat_factor=[1.0, 1.0], spat_bin=100)
            if func == "pwm.count":
                a["score_thresh"] = b["score_thresh"] = thresh
            pm.gvtrack_create("e", None, func, **a)
            pm.gvtrack_create("r", None, func, **b)
            # 500 bp bins: the first is seeded, the rest take the full scan
            x = pm.gextract("e", iv, iterator=500).sort_values("start")["e"].to_numpy()
            y = pm.gextract("r", inner, iterator=inner).sort_values("start")["r"].to_numpy()
            if func == "pwm.max.pos":
                y = np.sign(y) * (np.abs(y) + 100)
            if func == "pwm.count":
                assert y.sum() > 0, "the threshold must count something"
            np.testing.assert_allclose(x, y, rtol=1e-5, err_msg=func)

    @pytest.mark.parametrize("spatial", [False, True])
    def test_a_spat_min_past_the_end_of_a_bin_gives_nan(self, spatial):
        pssm = _rand_pssm()
        iv = pm.gintervals("1", 1000, 3000)
        for func in ("pwm", "pwm.max", "pwm.max.pos", "pwm.count"):
            a = {"pssm": pssm, "prior": 0.01, "spat_min": 600}
            if spatial:
                a.update(spat_factor=[1.0, 2.0], spat_bin=100)
            if func == "pwm.count":
                a["score_thresh"] = -12
            pm.gvtrack_create("e", None, func, **a)
            x = pm.gextract("e", iv, iterator=500)["e"].to_numpy(dtype=float)
            assert len(x) == 4 and np.isnan(x).all(), func


# gseq_pwm's three implementations, forced on sequences with no N (where the
# C++ path's log(1/4) neutral charge and the Python paths' column average
# cannot differ): the C++ PWMScorer takes only neutral_chars_policy =
# "log_quarter" (and never the minus strand alone), the vectorized path
# anything without a region of interest, and start_pos = 1 - the whole
# sequence - leaves only the per-sequence path.
_GSEQ_PATHS = {
    "cpp": ({"neutral_chars_policy": "log_quarter"}, "pm_pwm_score_strings"),
    "vectorized": ({}, "_pwm_score_batch_vectorized"),
    "python": ({"start_pos": 1}, "_pwm_score_sequence"),
}


@pytest.mark.parametrize("bounds", [{"spat_min": 30}, {"spat_max": 150},
                                    {"spat_min": 30, "spat_max": 150}])
@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("strand", ["bidirect", "plus", "minus"])
def test_gseq_pwm_spat_bounds_agree_on_all_three_paths(strand, spatial, bounds, monkeypatch):
    """Each path restricts the scan to the spat window and counts spatial bins
    and positions from its start; all three must give the same answer, which
    is gseq_pwm's answer on that window sliced out."""
    import pymisha.sequence as seqmod

    pssm = _rand_pssm(6)
    seqs = [s.upper() for s in pm.gseq_extract(
        pm.gintervals(["1"] * 4, [1000, 5000, 9000, 20000], [1200, 5200, 9200, 20200]))]
    assert not any("N" in s for s in seqs)
    kw = {"bidirect": {"bidirect": True}, "plus": {"bidirect": False, "strand": 1},
          "minus": {"bidirect": False, "strand": -1}}[strand]
    if spatial:
        kw.update(spat_factor=[0.5, 1.0, 2.0, 1.0], spat_bin=40)
    lo = bounds.get("spat_min", 1) - 1
    hi = bounds.get("spat_max", 200)
    for mode in ("lse", "max", "pos", "count"):
        mkw = dict(kw, score_thresh=-8.0) if mode == "count" else kw
        got = {}
        for path, (pkw, fn) in _GSEQ_PATHS.items():
            calls = []
            mod = pm._shared._pymisha if path == "cpp" else seqmod
            orig = getattr(mod, fn)
            monkeypatch.setattr(mod, fn, lambda *a, _o=orig, _c=calls, **k: _c.append(1) or _o(*a, **k))
            got[path] = np.asarray(pm.gseq_pwm(seqs, pssm, mode=mode, **mkw, **pkw, **bounds),
                                   dtype=float)
            monkeypatch.setattr(mod, fn, orig)
            # the minus strand alone is kept off the C++ path, which counts
            # spatial bins from the reverse complement's start
            assert bool(calls) == (path != "cpp" or strand != "minus"), path
        want = np.asarray(pm.gseq_pwm([s[lo:hi] for s in seqs], pssm, mode=mode, **mkw),
                          dtype=float)
        if mode == "pos":
            want = want + lo
        for path, v in got.items():
            np.testing.assert_allclose(v, want, rtol=1e-5, err_msg=f"{path} {mode}")


class TestPwmSpatialRangeMatchesPrego:
    """Port of "misha PWM with spat_min/spat_max exactly matches prego
    gextract_pwm_old" (R test-pwm-prego-regression.R). prego is not available
    here, so its numbers - and R misha 5.14.0's - were computed once on this
    test database and are frozen below. The PSSM is the R test's
    set.seed(42) runif matrix, row-normalised."""

    _PSSM = np.array([
        [0.25769236728546996, 0.26479336179916252, 0.20777383598864824, 0.26974043492671929],
        [0.25928234714902371, 0.27066855110674104, 0.22441340121735018, 0.24563570052688508],
        [0.19985803382107456, 0.082060639109707414, 0.27107948704512369, 0.44700184002409432],
        [0.28040685457477565, 0.16038631864536637, 0.23135268575804485, 0.32785414102181309],
        [0.35166743033885528, 0.30705438714490685, 0.0021636335859413739, 0.33911454893029652],
        [0.20046411939454326, 0.34911822421903771, 0.32165496347507488, 0.12876269291134415],
        [0.59915387342423643, 0.11282928689415694, 0.0059657238988901932, 0.28205111578271647],
        [0.077855342671073621, 0.57171196123538381, 0.12005471898413546, 0.23037797710940711],
        [0.19939342083092396, 0.2873084205696137, 0.27514836729729802, 0.23814979130216432],
        [0.49023516641614184, 0.057319257646816953, 0.42537283363284301, 0.027072742304198226],
        [0.21794029038928195, 0.24482682468106537, 0.18071597436546979, 0.35651691056418294],
        [0.3235798493486306, 0.1755803475895242, 0.19608469121905114, 0.30475511184279408],
        [0.45613665918104024, 0.4420162964984074, 0.01826700892383748, 0.083580035396714866],
        [0.13186646994934753, 0.23075041426652532, 0.50259508540240472, 0.13478803038172238],
        [0.20597050594178937, 0.37247435392462069, 0.19236297610264649, 0.22919216403094345],
    ])
    _SPAT = [0.02, 0.03, 0.10, 0.10, 0.11, 0.11, 0.11, 0.10, 0.10, 0.03, 0.02]
    _R_MISHA = {
        "spatial": [-16.785676956176758, -16.855157852172852, -16.977926254272461],
        "nospatial": [-14.202239990234375, -14.28255558013916, -14.303085327148438],
    }
    _PREGO = {
        "spatial": [-16.785684585571289, -16.855157852172852, -16.977926254272461],
        "nospatial": [-14.202250480651855, -14.282564163208008, -14.303102493286133],
    }
    _TOL = {"spatial": 0.01, "nospatial": 1e-6}   # the R test's tolerances

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    @pytest.mark.parametrize("case", ["spatial", "nospatial"])
    def test_spat_min_max_matches_prego(self, case):
        ivs = pm.gintervals(["1", "1", "1"], [10000, 20000, 30000], [10500, 20500, 30500])
        kw = {"spat_factor": self._SPAT, "spat_bin": 40} if case == "spatial" else {}
        pm.gvtrack_create("v", None, "pwm", pssm=self._PSSM, bidirect=True, prior=0.01,
                          extend=False, spat_min=30, spat_max=470, **kw)
        got = pm.gextract("v", ivs, iterator=ivs).sort_values("start")["v"].to_numpy()
        prego = np.array(self._PREGO[case])
        # all.equal()'s mean relative difference, as the R test uses
        assert np.mean(np.abs(got - prego)) / np.mean(np.abs(prego)) < self._TOL[case]
        np.testing.assert_array_equal(got, self._R_MISHA[case])


class TestPwmSpatialBidirectional:
    """Bidirectional spatial: bidi score >= forward-only score."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_bidi_geq_forward_with_spatial(self):
        """Bidirectional spatial PWM >= forward-only spatial PWM."""
        pssm = _create_test_pssm()
        test_interval = pm.gintervals("1", 200, 300)

        pm.gvtrack_create(
            "pwm_spatial_bidi", None, func="pwm",
            pssm=pssm, bidirect=True, extend=True, prior=0.01,
            spat_factor=[1.0, 2.0, 1.0], spat_bin=30,
        )
        pm.gvtrack_create(
            "pwm_spatial_fwd", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=[1.0, 2.0, 1.0], spat_bin=30,
        )

        scores = pm.gextract(
            ["pwm_spatial_bidi", "pwm_spatial_fwd"],
            test_interval, iterator=-1,
        )
        assert scores["pwm_spatial_bidi"].iloc[0] >= scores["pwm_spatial_fwd"].iloc[0]


class TestPwmSpatialIteratorShifts:
    """Spatial PWM with iterator shifts (sshift/eshift)."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_spatial_honors_iterator_shifts(self):
        """Spatial PWM with different shift magnitudes returns valid scores."""
        pssm = _create_test_pssm()
        base = pm.gintervals("1", 2000, 2100)

        spat_factors = [0.5, 1.0, 2.0, 1.0, 0.5]
        spat_bin = 20

        pm.gvtrack_create(
            "pwm_spat_small", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=spat_factors, spat_bin=spat_bin,
        )
        pm.gvtrack_create(
            "pwm_spat_large", None, func="pwm",
            pssm=pssm, bidirect=False, extend=True, prior=0.01, strand=1,
            spat_factor=spat_factors, spat_bin=spat_bin,
        )

        pm.gvtrack_iterator("pwm_spat_small", sshift=-10, eshift=10)
        pm.gvtrack_iterator("pwm_spat_large", sshift=-50, eshift=50)

        scores = pm.gextract(
            ["pwm_spat_small", "pwm_spat_large"], base, iterator=-1,
        )
        assert not np.isnan(scores["pwm_spat_small"].iloc[0])
        assert not np.isnan(scores["pwm_spat_large"].iloc[0])


class TestPwmSpatialErrorHandling:
    """Error handling for invalid spatial parameters."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_negative_spatial_factor_rejected(self):
        """Negative spatial factors should be rejected."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "pwm_bad_spat", None, func="pwm",
                pssm=pssm, prior=0.01,
                spat_factor=[-1, 1, 1], spat_bin=10,
            )

    def test_non_positive_bin_size_rejected(self):
        """Non-positive bin size should be rejected."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "pwm_bad_bin", None, func="pwm",
                pssm=pssm, prior=0.01,
                spat_factor=[1, 1, 1], spat_bin=0,
            )

    def test_zero_spatial_factor_rejected(self):
        """Zero spatial factors should be rejected."""
        pssm = _create_test_pssm()
        with pytest.raises((ValueError, Exception)):
            pm.gvtrack_create(
                "pwm_zero_spat", None, func="pwm",
                pssm=pssm, prior=0.01,
                spat_factor=[0, 1, 1], spat_bin=10,
            )


class TestPwmCountSpatialBidirectSliding:
    """pwm.count spatial sliding with bidirectional scanning.

    Ported from test-pwm-count-spatial-bidirect.R.
    Validates that spatial sliding counts bidirectional hits consistently
    with a per-interval (non-sliding) reference.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_count_spatial_sliding_bidirect(self):
        """pwm.count spatial sliding counts bidirectional hits once per position.

        Uses uniform spatial weights (all 1.0) so the sliding path produces
        the same counts as a non-sliding reference computed per-interval.
        """
        pssm = _create_test_pssm()

        params = {
            "pssm": pssm,
            "bidirect": True,
            "strand": 1,
            "extend": True,
            "prior": 0.01,
            "score_thresh": -25,
            "spat_factor": [1.0] * 6,
            "spat_bin": 15,
        }

        pm.gvtrack_create("pwm_count_spat_slide", None, func="pwm.count", **params)

        n = 30
        starts = [2400 + i for i in range(n)]
        ends = [s + 55 for s in starts]
        ivs = pm.gintervals(["1"] * n, starts, ends)

        # Sliding result
        result_slide = pm.gextract("pwm_count_spat_slide", ivs, iterator=-1)

        # Per-interval reference (no sliding possible with single-interval extraction)
        per_interval_vals = []
        for idx in range(len(ivs)):
            single = ivs.iloc[[idx]].reset_index(drop=True)
            res = pm.gextract("pwm_count_spat_slide", single, iterator=-1)
            per_interval_vals.append(res["pwm_count_spat_slide"].iloc[0])

        np.testing.assert_allclose(
            result_slide["pwm_count_spat_slide"].values,
            np.array(per_interval_vals),
            atol=1e-6,
        )


class TestPwmSpatialSingleStrandSeed:
    """Regression: the spatial sliding-window seed must score a single strand
    for a non-bidirectional PSSM.

    PWMScorer::compute_motif_at decided which strands to score from m_strand
    using `m_strand != +1` for the reverse strand. With the default unstranded
    m_strand==0 that test was true, so a non-bidirectional PSSM had BOTH strands
    summed into the spatial TOTAL/MAX aggregation (the seed path), inflating the
    score to equal the bidirectional result.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_nonbidirect_spatial_is_single_strand(self):
        # Non-palindromic PSSM so the reverse strand contributes a different,
        # finite amount (otherwise fwd==rev and the bug would be invisible).
        pssm = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ])
        iv = pm.gintervals("1", 200, 300)
        spat = dict(spat_factor=[0.5, 1.0, 2.0, 2.5, 2.0, 1.0, 0.5], spat_bin=20)

        pm.gvtrack_create("v_bi", None, func="pwm", pssm=pssm, bidirect=True,
                          extend=False, prior=0.01, **spat)
        pm.gvtrack_create("v_nb", None, func="pwm", pssm=pssm, bidirect=False,
                          extend=False, prior=0.01, **spat)

        bi = pm.gextract("v_bi", iv, iterator=iv)["v_bi"].iloc[0]
        nb = pm.gextract("v_nb", iv, iterator=iv)["v_nb"].iloc[0]

        # Both strands (bidirect) is a log-sum-exp union over fwd+rev, so it must
        # be strictly greater than the single forward strand. Before the fix the
        # non-bidirect seed counted both strands too, making nb == bi.
        assert np.isfinite(bi) and np.isfinite(nb)
        assert nb < bi - 1e-4, f"non-bidirect spatial ({nb}) should be < bidirect ({bi})"
