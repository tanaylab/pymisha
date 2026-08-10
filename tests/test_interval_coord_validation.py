"""Interval coordinate handling at the Python -> C++ conversion boundary.

Two things are checked here:

* Bad coordinates raise instead of corrupting or crashing.  A negative
  ``start`` used to reach ``GenomeTrackFixedBin::read_bins_bulk`` as a
  negative bin index, which memcpy'd from before the mmap: a segfault, or
  silently wrong values when the read happened to stay inside the same page.
* Ordinary pandas dtypes are accepted.  Float coordinates (a BED read with
  any blank field, or a non-matching merge) and categorical ``chrom``
  columns used to be rejected outright.

R misha's ``IntervalConverter`` runs ``GInterval::verify()`` on every
converted interval and coerces coordinates; so do we now.
"""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


def _iv(start, end, chrom="1"):
    return pd.DataFrame({"chrom": [chrom], "start": [start], "end": [end]})


# chromosome "1" of the example DB is 500000 bp
BAD_COORDS = {
    "negative_start": (_iv(-100, 100), "start coordinate must be"),
    "inverted": (_iv(300, 100), "start coordinate must be"),
    "zero_width": (_iv(100, 100), "start coordinate must be"),
    "past_chrom_end": (_iv(499900, 500100), "exceeds chromosome boundaries"),
    "nan_start": (_iv(np.nan, 100.0), "missing .NaN. value"),
    "nan_end": (_iv(0.0, np.nan), "missing .NaN. value"),
}

READERS = {
    "gextract": lambda iv: pm.gextract("dense_track", iv, iterator=100),
    "gsummary": lambda iv: pm.gsummary("dense_track", iv),
    "gscreen": lambda iv: pm.gscreen("dense_track > 0", iv, iterator=100),
    "gquantiles": lambda iv: pm.gquantiles("dense_track", [0.5], iv),
    "gdist": lambda iv: pm.gdist("dense_track", [0, 1, 2], intervals=iv),
    "gextract_sparse": lambda iv: pm.gextract("sparse_track", iv),
    "gseq_extract": lambda iv: pm.gseq_extract(iv),
}


@pytest.mark.parametrize("case", list(BAD_COORDS), ids=list(BAD_COORDS))
@pytest.mark.parametrize("reader", list(READERS), ids=list(READERS))
def test_bad_coords_raise_instead_of_crashing(reader, case):
    intervals, message = BAD_COORDS[case]
    with pytest.raises(Exception, match=message):
        READERS[reader](intervals)


@pytest.mark.parametrize(
    "intervals",
    [
        pytest.param(_iv(0, 1000), id="int64"),
        pytest.param(_iv(0.0, 1000.0), id="float64"),
        pytest.param(
            pd.DataFrame(
                {
                    "chrom": ["1"],
                    "start": np.array([0], dtype=np.int32),
                    "end": np.array([1000], dtype=np.int32),
                }
            ),
            id="int32",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "chrom": ["1"],
                    "start": pd.array([0], dtype="Int64"),
                    "end": pd.array([1000], dtype="Int64"),
                }
            ),
            id="nullable_Int64",
        ),
        pytest.param(
            pd.DataFrame(
                {"chrom": pd.Categorical(["1"]), "start": [0], "end": [1000]}
            ),
            id="categorical_chrom",
        ),
    ],
)
def test_ordinary_dtypes_are_accepted(intervals):
    result = pm.gextract("dense_track", intervals, iterator=100)
    assert result is not None
    assert len(result) == 10


def test_categorical_chrom_matches_plain_string_chrom():
    """Guards against the [categories, codes] pair being read positionally."""
    coords = {"start": [0, 0, 1000], "end": [100, 100, 1100]}
    chroms = ["X", "1", "X"]

    categorical = pm.gextract(
        "dense_track", pd.DataFrame({"chrom": pd.Categorical(chroms), **coords}),
        iterator=100,
    )
    plain = pm.gextract(
        "dense_track", pd.DataFrame({"chrom": chroms, **coords}), iterator=100
    )
    pd.testing.assert_frame_equal(categorical, plain)
