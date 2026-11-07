"""R misha test-liftover.R blocks 1377-1700 and 2169-2300 (chain search,
ordering, edges, duplications), against R 5.14.2's output for the same chains
(tests/test_data/liftover_r_5_14_2.json; targets are this DB's chr1/chr2/chrX).
A case with a "warning" is R 5.16.0's: NULL with that warning."""

import json
import os
import re

import pandas as pd
import pytest

import pymisha as pm

with open(os.path.join(os.path.dirname(__file__), "test_data", "liftover_r_5_14_2.json")) as fh:
    CASES = json.load(fh)


@pytest.mark.parametrize("name", list(CASES))
def test_liftover_matches_r(name, tmp_path):
    c = CASES[name]
    path = tmp_path / "c.chain"
    path.write_text(c["chain"])
    want = pd.DataFrame(c["r"])
    q = pd.DataFrame(c["q"])
    if "warning" in c:
        chain = pm.gintervals_load_chain(str(path), **c["load"])
        with pytest.warns(UserWarning, match=re.escape(c["warning"])):
            assert pm.gintervals_liftover(q, chain, **c["lift"]).empty
        return
    got = pm.gintervals_liftover(q, pm.gintervals_load_chain(str(path), **c["load"]), **c["lift"])
    assert list(got.columns) == list(want.columns)
    assert got["chrom"].astype(str).tolist() == want["chrom"].astype(str).tolist()
    for col in want.columns.drop("chrom"):
        assert got[col].astype(float).tolist() == want[col].astype(float).tolist(), col
