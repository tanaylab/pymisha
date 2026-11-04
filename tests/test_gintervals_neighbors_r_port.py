"""R misha test-gintervals.neighbors.R blocks 144-560, against R 5.14.2's own
output for the same fixtures (tests/test_data/neighbors_r_5_14_2.json; R's
chr3 is chrX here). Rows, row order, columns and distances must all match."""

import json
import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

with open(os.path.join(os.path.dirname(__file__), "test_data", "neighbors_r_5_14_2.json")) as fh:
    CASES = json.load(fh)


@pytest.mark.parametrize("name", list(CASES))
def test_neighbors_matches_r(name):
    c = CASES[name]
    got = pm.gintervals_neighbors(pd.DataFrame(c["q"]), pd.DataFrame(c["t"]), **c["kw"])
    want = pd.DataFrame(c["r"])
    if len(want) == 0:
        assert got is None or len(got) == 0
        return
    assert got is not None
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    for col in want.columns:
        g, w = got[col].to_numpy(), want[col].to_numpy()
        if col.startswith("chrom") or w.dtype == object:
            assert [None if pd.isna(x) else str(x) for x in g] == [None if pd.isna(x) else str(x) for x in w], col
        else:
            np.testing.assert_array_equal(g.astype(float), w.astype(float), err_msg=col)
