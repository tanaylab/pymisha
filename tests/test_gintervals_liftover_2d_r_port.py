"""R misha test-liftover.R:379 (2D gintervals.liftover) and more 2D cases, against
R 5.14.2's output for the same chains (tests/test_data/liftover_2d_r_5_14_2.json;
targets are this DB's chr1/chr2/chrX)."""

import json
import os

import pandas as pd
import pytest

import pymisha as pm

with open(os.path.join(os.path.dirname(__file__), "test_data", "liftover_2d_r_5_14_2.json")) as fh:
    CASES = json.load(fh)


@pytest.mark.parametrize("name", list(CASES))
def test_liftover_2d_matches_r(name, tmp_path):
    c = CASES[name]
    path = tmp_path / "c.chain"
    path.write_text(c["chain"])
    chain = pm.gintervals_load_chain(str(path), **c["load"])
    q = pd.DataFrame(c["q"])
    want = pd.DataFrame(c["r"])
    if "error" in want.columns:
        with pytest.raises(ValueError, match=want["error"][0]):
            pm.gintervals_liftover(q, chain, **c["lift"])
        return
    got = pm.gintervals_liftover(q, chain, **c["lift"])
    if want.empty:  # R returns NULL
        assert got.empty
        return
    assert list(got.columns) == list(want.columns)
    for col in want.columns:
        if col.startswith("chrom"):
            assert got[col].astype(str).tolist() == want[col].tolist(), col
        else:
            assert got[col].astype(float).tolist() == want[col].astype(float).tolist(), col
