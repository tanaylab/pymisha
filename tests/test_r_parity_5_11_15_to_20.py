"""Parity with R misha 5.11.15 / 5.11.18 / 5.11.20.

- ``gvtrack_create`` rejects parameters the func does not accept (5.11.15).
- Overlapping intervals used as a 1D iterator warn that they were merged (5.11.18).
- ``gtrack_import`` reports the chromosome names it skipped (5.11.18).
- ``max_processes`` auto-configures to at most 32 workers (5.11.20).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _remove_all_vtracks():
    for vt in pm.gvtrack_ls():
        pm.gvtrack_rm(vt)


def _pssm():
    return np.array([[0.7, 0.1, 0.1, 0.1],
                     [0.1, 0.7, 0.1, 0.1]])


class TestVtrackParamValidation:
    """5.11.15: a misspelled parameter is an error, not a silent default."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        _remove_all_vtracks()
        yield
        _remove_all_vtracks()

    def test_misspelled_pwm_param_raises(self):
        with pytest.raises(ValueError, match="does not accept parameter"):
            pm.gvtrack_create("v", None, "pwm.max", pssm=_pssm(), bidrect=False)

    def test_error_lists_the_accepted_parameters(self):
        with pytest.raises(ValueError, match="accepted parameters are:.*spat_bin"):
            pm.gvtrack_create("v", None, "pwm", pssm=_pssm(), spat_binn=5)

    def test_edit_distance_param_rejected_on_plain_pwm(self):
        # max_indels belongs to the edit-distance family, not to pwm.max.
        with pytest.raises(ValueError, match="max_indels"):
            pm.gvtrack_create("v", None, "pwm.max", pssm=_pssm(), max_indels=1)

    def test_max_indels_rejected_on_lse_edit_distance(self):
        # The LSE scorer has no indel support; R misha's handler omits it too.
        with pytest.raises(ValueError, match="max_indels"):
            pm.gvtrack_create("v", None, "pwm.edit_distance.lse", pssm=_pssm(),
                              score_thresh=-3.0, max_indels=1)

    def test_func_with_no_params_rejects_any_keyword(self):
        with pytest.raises(ValueError, match="takes no additional keyword arguments"):
            pm.gvtrack_create("v", "dense_track", "avg", sshfit=10)

    def test_accepted_params_still_work(self):
        pm.gvtrack_create("v", None, "pwm.max", pssm=_pssm(), prior=0,
                          bidirect=False, strand=1, extend=True)
        r = pm.gextract("v", intervals=pm.gintervals("1", 200, 240), iterator=40)
        assert np.isfinite(r["v"].iloc[0])

    def test_dotted_r_spelling_is_accepted(self):
        # score.thresh, R's spelling, must not read as an unknown parameter.
        pm.gvtrack_create("v", None, "pwm.count", pssm=_pssm(), prior=0,
                          **{"score.thresh": -3.0})
        assert "v" in pm.gvtrack_ls()
        pm.gvtrack_create("v2", None, "pwm.count", pssm=_pssm(), prior=0,
                          score_thresh=-3.0)
        assert "v2" in pm.gvtrack_ls()

    def test_masked_funcs_warn_rather_than_raise(self):
        with pytest.warns(UserWarning, match="does not accept parameters"):
            pm.gvtrack_create("v", None, "masked.count", bogus=1)

    def test_filter_is_accepted_everywhere(self):
        # pymisha extension, not an R parameter - must survive the check.
        pm.gvtrack_create("v", "dense_track", "avg", filter=None)
        assert "v" in pm.gvtrack_ls()


class TestOverlappingIteratorWarning:
    """5.11.18: overlapping iterator intervals are merged - say so."""

    SCOPE = None

    def setup_method(self):
        self.SCOPE = pm.gintervals("1", 0, 5000)

    def _extract(self, iterator):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = pm.gextract("dense_track", intervals=self.SCOPE, iterator=iterator)
        msgs = [str(w.message) for w in caught
                if "Overlapping intervals were used as an iterator" in str(w.message)]
        return res, msgs

    def test_overlapping_intervals_warn(self):
        it = pd.DataFrame({"chrom": ["1"] * 3, "start": [1000, 1500, 3000],
                           "end": [2000, 2500, 3100]})
        res, msgs = self._extract(it)
        assert len(res) == 2, "the overlapping pair must have merged"
        assert len(msgs) == 1
        # Names the pair and the row it became, as R misha does.
        assert "1 1000-2000 and 1 1500-2500" in msgs[0]
        assert "single row 1 1000-2500" in msgs[0]
        assert "3 intervals were merged into 2" in msgs[0]
        assert "intervalID" in msgs[0]

    def test_nested_interval_warns(self):
        # It widens no row, so it disappears entirely - the case that used to be
        # completely silent.
        it = pd.DataFrame({"chrom": ["1"] * 2, "start": [1000, 1200],
                           "end": [2000, 1500]})
        res, msgs = self._extract(it)
        assert len(res) == 1
        assert len(msgs) == 1

    def test_exact_duplicates_stay_silent(self):
        # Both copies come back as the same row, so no interval is missing.
        it = pd.DataFrame({"chrom": ["1"] * 2, "start": [1000, 1000],
                           "end": [2000, 2000]})
        _res, msgs = self._extract(it)
        assert msgs == []

    def test_disjoint_intervals_stay_silent(self):
        it = pd.DataFrame({"chrom": ["1"] * 2, "start": [1000, 3000],
                           "end": [2000, 3100]})
        res, msgs = self._extract(it)
        assert len(res) == 2
        assert msgs == []

    def test_touching_intervals_stay_silent(self):
        it = pd.DataFrame({"chrom": ["1"] * 2, "start": [1000, 2000],
                           "end": [2000, 3000]})
        res, msgs = self._extract(it)
        assert len(res) == 2
        assert msgs == []

    def test_same_coordinates_on_different_chromosomes_stay_silent(self):
        it = pd.DataFrame({"chrom": ["1", "2"], "start": [1000, 1000],
                           "end": [2000, 2000]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pm.gextract("dense_track", intervals=pm.gintervals(["1", "2"], [0, 0],
                                                               [5000, 5000]),
                        iterator=it)
        assert not [w for w in caught
                    if "Overlapping intervals" in str(w.message)]


class TestGtrackImportSkippedChroms:
    """5.11.18: report the chromosome names an import dropped."""

    @staticmethod
    def _write(tmp_path, name, rows):
        p = tmp_path / f"{name}.bedgraph"
        p.write_text("".join(f"{c}\t{s}\t{e}\t{v}\n" for c, s, e, v in rows))
        return str(p)

    @staticmethod
    def _clean(track):
        if pm.gtrack_exists(track):
            pm.gtrack_rm(track, force=True)

    def test_no_matching_chromosome_errors(self, tmp_path):
        f = self._write(tmp_path, "none", [("chr99", 100, 200, 1.5)])
        self._clean("tmp_skip_none")
        with pytest.raises(ValueError, match="No intervals map to known chromosomes"):
            pm.gtrack_import("tmp_skip_none", "t", f, binsize=10)
        self._clean("tmp_skip_none")

    def test_skipped_primary_chromosome_warns(self, tmp_path):
        f = self._write(tmp_path, "primary",
                        [("1", 100, 200, 1.5), ("chr7", 100, 200, 3.5)])
        self._clean("tmp_skip_primary")
        try:
            with pytest.warns(UserWarning, match="primary chromosome"):
                pm.gtrack_import("tmp_skip_primary", "t", f, binsize=10)
        finally:
            self._clean("tmp_skip_primary")

    def test_skipped_scaffold_does_not_warn(self, tmp_path):
        # A whole-genome bigWig against a primary-only database looks like this;
        # it is a message, not a warning.
        f = self._write(tmp_path, "scaffold",
                        [("1", 100, 200, 1.5), ("chrUn_xx", 100, 200, 2.5),
                         ("GL000220.1", 100, 200, 1.0)])
        self._clean("tmp_skip_scaffold")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pm.gtrack_import("tmp_skip_scaffold", "t", f, binsize=10)
            assert not [w for w in caught if "were skipped" in str(w.message)]
        finally:
            self._clean("tmp_skip_scaffold")

    def test_fully_matching_import_is_silent(self, tmp_path):
        f = self._write(tmp_path, "clean",
                        [("1", 100, 200, 1.5), ("2", 100, 200, 2.5)])
        self._clean("tmp_skip_clean")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pm.gtrack_import("tmp_skip_clean", "t", f, binsize=10)
            assert not [w for w in caught if "were skipped" in str(w.message)]
        finally:
            self._clean("tmp_skip_clean")

    @pytest.mark.parametrize(("name", "primary"), [
        ("chr7", True), ("7", True), ("X", True), ("chrX", True), ("22", True),
        ("chrM", False), ("chrUn_gl000220", False), ("GL000220.1", False),
        ("chr1_random", False), ("scaffold_12", False),
    ])
    def test_primary_chromosome_classifier(self, name, primary):
        from pymisha.tracks import _is_primary_chrom_name
        assert _is_primary_chrom_name(name) is primary


def test_max_processes_is_capped_at_32():
    """5.11.20: the auto-configured default caps at 32; past that the
    per-worker cost outweighs the parallelism. Still settable by hand."""
    import os

    from pymisha._shared import CONFIG

    cores = os.cpu_count() or 1
    assert CONFIG["max_processes"] == max(4, min(32, int(cores * 0.7)))
    assert CONFIG["max_processes"] <= 32

    prev = CONFIG["max_processes"]
    try:
        CONFIG["max_processes"] = 64
        assert CONFIG["max_processes"] == 64, "the cap must not block a manual override"
    finally:
        CONFIG["max_processes"] = prev
