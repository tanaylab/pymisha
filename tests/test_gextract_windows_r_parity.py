"""gseq_onehot and gextract_windows return the arrays R misha's gseq.onehot and
gextract.windows return (tests/test_data/windows, written by
tools/generate_windows_baseline.R on the example database)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

BASELINE = Path(__file__).parent / "test_data" / "windows" / "baseline.json"


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text())


def frame(d):
    return pd.DataFrame(d) if isinstance(d, dict) else pd.DataFrame.from_records(d)


@pytest.fixture(autouse=True)
def _no_progress(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "progress", False)


def test_windows_match_r(baseline):
    loci = frame(baseline["loci"])
    exclude = frame(baseline["exclude"])
    assert baseline["cases"]
    for case in baseline["cases"]:
        a = case["args"]
        w = pm.gextract_windows(
            loci,
            tracks=a.get("tracks"),
            width=a["width"],
            signal_width=a.get("signal_width"),
            jitter=a.get("jitter", 0),
            binsize=a.get("binsize", 1),
            use_strand=a.get("use_strand", True),
            exclude=exclude if a.get("exclude") else None,
            min_counts=a.get("min_counts"),
            count_track=a.get("count_track", 1) - 1,
        )
        assert list(w.x.shape) == case["x_dim"], case["name"]
        assert hashlib.sha256(w.x.tobytes()).hexdigest() == case["x_sha256"], case["name"]
        np.testing.assert_array_equal(w.intervals["index"], case["index"])
        np.testing.assert_array_equal(w.intervals["start"], case["start"])
        np.testing.assert_array_equal(w.dropped["index"], case["dropped_index"])
        assert list(w.dropped["reason"]) == list(case["dropped_reason"])
        if case.get("y_dim") is None:
            assert w.y is None
        else:
            assert list(w.y.shape) == case["y_dim"]
            ry = np.array([np.nan if v is None else v for v in case["y"]], dtype=np.float64)
            np.testing.assert_array_equal(w.y.ravel(), ry.astype(np.float32))


def test_onehot_pad_matches_r(baseline):
    p = baseline["pad"]
    x = pm.gseq_onehot(frame(p["intervals"]), pad=True)
    assert list(x.shape) == p["x_dim"]
    assert hashlib.sha256(x.tobytes()).hexdigest() == p["x_sha256"]
