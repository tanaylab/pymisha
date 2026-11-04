"""Ports of R misha's test-pwm-count.R:566 (vtrack score.thresh coercion) and
:628 (a threshold that discriminates), which pymisha only had for gseq_pwm."""

import numpy as np
import pytest

import pymisha as pm

AC = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


@pytest.fixture(autouse=True)
def _clear():
    pm.gvtrack_clear()
    yield
    pm.gvtrack_clear()


def test_vtrack_score_thresh_coercion():  # test-pwm-count.R:566
    iv = pm.gintervals("chr1", 200, 300)
    kw = dict(pssm=AC, bidirect=False, extend=True, prior=0.01)
    pm.gvtrack_create("count_chr", None, "pwm.count", score_thresh="-5", **kw)
    pm.gvtrack_create("count_num", None, "pwm.count", score_thresh=-5, **kw)
    res = pm.gextract(["count_chr", "count_num"], iv, iterator=iv)
    assert res["count_num"][0] > 0
    assert res["count_chr"][0] == res["count_num"][0]
    with pytest.raises(ValueError, match="score.thresh must be a single value"):
        pm.gvtrack_create("bad", None, "pwm.count", pssm=AC, score_thresh=[-10, -5])
    for bad in ("loose", True, False):
        with pytest.raises(ValueError, match="score.thresh must be a single number"):
            pm.gvtrack_create("bad", None, "pwm.count", pssm=AC, score_thresh=bad)


def test_workable_score_thresh_is_unchanged():  # test-pwm-count.R:628
    iv = pm.gintervals("chr1", 200, 300)
    pm.gvtrack_create("pwm_ref", None, "pwm", pssm=AC, bidirect=False, extend=True, prior=0.01, strand=1)
    per_base = pm.gextract("pwm_ref", iv, iterator=1)["pwm_ref"].to_numpy()
    expected = int(np.sum(per_base >= -5))
    assert 0 < expected < len(per_base)
    pm.gvtrack_create("count_ref", None, "pwm.count", pssm=AC, bidirect=False, extend=True, prior=0.01,
                      strand=1, score_thresh=-5)
    assert pm.gextract("count_ref", iv, iterator=iv)["count_ref"][0] == expected
