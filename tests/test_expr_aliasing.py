import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import pymisha as pm
from pymisha.expr import _expr_safe_name, _parse_expr_vars

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _copy_db(tmp_path: Path) -> Path:
    dst = tmp_path / "trackdb" / "test"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DB, dst)
    return dst


def test_parse_expr_vars_distinguishes_dot_and_underscore_names():
    new_expr, used_tracks, used_vtracks, var_map = _parse_expr_vars(
        "foo.bar + foo_bar",
        {"foo.bar", "foo_bar"},
        set(),
    )

    dot_alias = _expr_safe_name("foo.bar")
    underscore_alias = _expr_safe_name("foo_bar")

    assert dot_alias != underscore_alias
    assert var_map[dot_alias] == "foo.bar"
    assert var_map[underscore_alias] == "foo_bar"
    assert used_tracks == {"foo.bar", "foo_bar"}
    assert used_vtracks == set()
    assert new_expr == f"{dot_alias} + {underscore_alias}"


def test_parse_expr_vars_preserves_keyword_whitespace():
    new_expr, used_tracks, used_vtracks, var_map = _parse_expr_vars(
        "not track1 and a is None",
        {"track1"},
        set(),
    )

    track_alias = _expr_safe_name("track1")
    assert new_expr == f"not {track_alias} and a is None"
    assert used_tracks == {"track1"}
    assert used_vtracks == set()
    assert var_map[track_alias] == "track1"


def test_parse_expr_vars_preserves_multichar_operators():
    new_expr, used_tracks, used_vtracks, var_map = _parse_expr_vars(
        "track1<=10 and track1!=0 and track1**2//2",
        {"track1"},
        set(),
    )

    track_alias = _expr_safe_name("track1")
    assert new_expr == f"{track_alias}<=10 and {track_alias}!=0 and {track_alias}**2//2"
    assert used_tracks == {"track1"}
    assert used_vtracks == set()
    assert var_map[track_alias] == "track1"


def test_parse_expr_vars_resolves_dotted_prefix_when_full_name_is_missing():
    new_expr, used_tracks, used_vtracks, var_map = _parse_expr_vars(
        "foo.bar + 1",
        {"foo"},
        set(),
    )

    foo_alias = _expr_safe_name("foo")
    assert new_expr == f"{foo_alias}.bar + 1"
    assert used_tracks == {"foo"}
    assert used_vtracks == set()
    assert var_map[foo_alias] == "foo"


def test_gextract_keeps_dot_and_underscore_tracks_distinct(tmp_path):
    root = _copy_db(tmp_path)
    intervals = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 10, 20],
            "end": [10, 20, 30],
        }
    )

    try:
        pm.gdb_init(str(root))
        pm.gtrack_create_dense("foo.bar", "dot track", intervals, [1.0, 2.0, 3.0], binsize=10, defval=np.nan)
        pm.gvtrack_create("foo_bar", "sparse_track", func="exists")

        out = pm.gextract("foo.bar + foo_bar * 0", intervals, iterator=10)
        assert out is not None
        np.testing.assert_allclose(
            out["foo.bar + foo_bar * 0"].to_numpy(dtype=float),
            np.array([1.0, 2.0, 3.0]),
            equal_nan=True,
        )
    finally:
        pm.gdb_init(str(TEST_DB))
