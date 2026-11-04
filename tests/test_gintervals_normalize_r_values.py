"""gintervals_normalize against R misha 5.14.2 output on the test DB.

Covers an alias chrom ("1"), odd sizes, and clipping at both chromosome ends.
"""

import pandas as pd
import pytest

import pymisha as pm

IV = pd.DataFrame({"chrom": ["1", "chr1", "chr2", "chrX", "chr1"],
                   "start": [0, 10, 100, 0, 0], "end": [10, 21, 200, 5, 1]})


@pytest.mark.parametrize("size,starts,ends", [
    (3, [4, 14, 149, 1, 0], [7, 17, 152, 4, 2]),
    (1000, [0, 0, 0, 0, 0], [505, 515, 650, 502, 500]),
])
def test_normalize_matches_r(size, starts, ends):
    res = pm.gintervals_normalize(IV, size)
    assert res["chrom"].astype(str).tolist() == ["chr1", "chr1", "chr2", "chrX", "chr1"]
    assert res["start"].tolist() == starts
    assert res["end"].tolist() == ends


def test_normalize_clips_at_chrom_end_like_r():
    res = pm.gintervals_normalize(pd.DataFrame({"chrom": ["chr2"], "start": [299990], "end": [300000]}), 101)
    assert (res["start"].item(), res["end"].item()) == (299945, 300000)
