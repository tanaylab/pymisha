"""misha 5.16.0: a track that another session re-created is read with its current index.

The C++ caches each indexed track's track.idx. Before 5.16 a track that another
process removed and created again with another layout was read through the old
index until the database was opened again: errors, or silently wrong values.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

import pymisha as pm
from _dbpath import TESTDB_ROOT

_MAKE = """
import sys
import pandas as pd
import pymisha as pm

pm.CONFIG["multitasking"] = False
pm.gdb_init(sys.argv[1])
version = sys.argv[2]
for t in ("st", "dt", "rt"):
    if pm.gtrack_exists(t):
        pm.gtrack_rm(t, force=True)
if version == "1":
    pm.gtrack_create_sparse("st", "", pm.gintervals(["chr1", "chr2"], 0, 10), [1, 2])
    pm.gtrack_create("dt", "", "st", iterator=10)
    rects = pd.DataFrame({"chrom1": ["chr1"], "start1": [0], "end1": [10],
                          "chrom2": ["chr2"], "start2": [0], "end2": [10]})
    pm.gtrack_2d_create("rt", "", rects, [1])
else:
    pm.gtrack_create_sparse("st", "", pm.gintervals(["chr1", "chr1", "chr1", "chr2"],
                                                    [0, 20, 40, 0], [10, 30, 50, 10]), [5, 6, 7, 8])
    pm.gtrack_create("dt", "", "st", iterator=7)
    rects = pd.DataFrame({"chrom1": ["chr1"] * 3, "start1": [0, 20, 40], "end1": [10, 30, 50],
                          "chrom2": ["chr1", "chr2", "chr2"], "start2": [0, 0, 20], "end2": [10, 10, 30]})
    pm.gtrack_2d_create("rt", "", rects, [3, 4, 5])
for t in ("st", "dt", "rt"):
    if not os.path.exists(os.path.join(pm.gtrack_path(t), "track.idx")):
        pm.gtrack_convert_to_indexed(t, remove_old=True)
"""


def _make(db: Path, version: str, script: Path) -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, str(script), str(db), version], check=True, env=env)


def _values():
    st = pm.gextract("st", pm.gintervals_all())["st"].tolist()
    dt = pm.gextract("dt", pm.gintervals("chr1", 0, 14))["dt"].tolist()
    rt = pm.gextract("rt", pm.gintervals_2d("chr1", 0, 100, "chr2", 0, 100))["rt"].tolist()
    return st, dt, rt


def test_tracks_recreated_by_another_session_are_read_with_their_current_index(tmp_path):
    db = tmp_path / "trackdb" / "test"
    db.parent.mkdir(parents=True)
    shutil.copytree(TESTDB_ROOT, db)
    script = tmp_path / "make.py"
    script.write_text("import os\n" + textwrap.dedent(_MAKE))
    try:
        _make(db, "1", script)
        pm.gdb_init(str(db))
        # reading caches each track's index
        st, dt, rt = _values()
        assert (st, dt[0], len(dt), rt) == ([1.0, 2.0], 1.0, 2, [1.0])
        _make(db, "2", script)
        # the same session, no reload: misha 5.16 gives the new values
        assert _values() == ([5.0, 6.0, 7.0, 8.0], [5.0, 5.0], [4.0, 5.0])
    finally:
        pm.gdb_init(str(TESTDB_ROOT))
