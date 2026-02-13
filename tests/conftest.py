import os
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"
os.environ.setdefault("PYMISHA_EXAMPLES_DB", str(TEST_DB))


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    pm.gdb_init(str(TEST_DB))
    yield
    pm.gdb_unload()


def extract_values(expr, intervals, iterator=None):
    df = pm.gextract(expr, intervals, iterator=iterator)
    if df is None or len(df) == 0:
        return np.array([], dtype=float)

    data_cols = [c for c in df.columns if c not in {"chrom", "start", "end", "intervalID"}]
    assert len(data_cols) == 1
    return df[data_cols[0]].to_numpy(dtype=float, copy=False)
