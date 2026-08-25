"""Database artefacts carry misha's permissions, whatever the caller's umask.

pymisha's C++ layer already set umask(07) around its own writes; the Python
write sites did not, so under the lab's default umask 022 a track directory
written from C++ was 0770 while the namespace directory containing it, written
from Python, was 0755 - and a .interv, .attributes or .db.cache.dirty saved
from Python was 0644, which a colleague could not overwrite from either
package. A 0644 .db.cache.dirty in particular blocks R's own
.gdb.cache_mark_dirty, which is the staleness misha 5.11.14 went to trouble to
fix.

Expected modes are R misha's, from tests/testthat/test-permissions-umask.R.
"""

import os
import stat
import subprocess
import sys
import textwrap

import pytest

EXPECTED = {
    "groot": 0o770,
    "tracks": 0o770,
    "seq": 0o770,
    "chrom_sizes.txt": 0o660,
    "namespace_dir": 0o770,
    "track_dir": 0o770,
    "nested_track_dir": 0o770,
    "attributes": 0o660,
    "interv": 0o660,
    "db_cache_dirty": 0o660,
}

_SCRIPT = textwrap.dedent(
    """
    import json, os, stat, sys, warnings
    warnings.filterwarnings("ignore")
    import pandas as pd, pymisha as pm

    os.umask(0o022)                      # the lab default: what we must override
    root, fa = sys.argv[1], sys.argv[2]
    with open(fa, "w") as fh:
        fh.write(">chr1\\n" + "A" * 20000 + "\\n")
    if os.environ.get("PYMISHA_NO_UMASK"):
        pm.CONFIG["permissions_umask"] = None
    pm.gdb_create(groot=root, fasta=fa)
    pm.gdb_init(root)
    iv = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5000]})
    pm.gtrack_create_dense("t1", "test", iv, [1.0], binsize=100)
    pm.gtrack_create_dirs("ns.sub.t")
    pm.gtrack_create_dense("ns.sub.t", "test", iv, [2.0], binsize=100)
    pm.gintervals_save(iv, "myiv")

    paths = {
        "groot": root,
        "tracks": root + "/tracks",
        "seq": root + "/seq",
        "chrom_sizes.txt": root + "/chrom_sizes.txt",
        "namespace_dir": root + "/tracks/ns",
        "track_dir": root + "/tracks/t1.track",
        "nested_track_dir": root + "/tracks/ns/sub/t.track",
        "attributes": root + "/tracks/t1.track/.attributes",
        "interv": root + "/tracks/myiv.interv",
        "db_cache_dirty": root + "/.db.cache.dirty",
    }
    print(json.dumps({k: stat.S_IMODE(os.stat(v).st_mode) for k, v in paths.items()}))
    """
)


def _modes(tmp_path, disabled=False):
    """A subprocess, because umask is process-global and pytest-xdist is not."""
    import json

    env = dict(os.environ)
    if disabled:
        env["PYMISHA_NO_UMASK"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path / "db"), str(tmp_path / "g.fa")],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_database_artefacts_are_group_writable_not_world_readable(tmp_path):
    modes = _modes(tmp_path)
    wrong = {k: (oct(modes[k]), oct(v)) for k, v in EXPECTED.items() if modes[k] != v}
    assert not wrong, f"got/expected: {wrong}"


def test_the_umask_is_what_makes_the_difference(tmp_path):
    """Without it, Python-written artefacts fall back to the caller's 022.

    Pinning the failure keeps the first test from passing on a machine whose
    umask happened to be 007 already.
    """
    modes = _modes(tmp_path, disabled=True)
    assert modes["namespace_dir"] == 0o755   # Python wrote it
    assert modes["track_dir"] == 0o770       # C++ wrote it - already correct
    assert modes["interv"] == 0o644
    assert modes["db_cache_dirty"] == 0o644


def test_none_leaves_the_process_umask_alone(tmp_path):
    """CONFIG["permissions_umask"] = None must not touch the umask at all."""
    from pymisha._shared import CONFIG, _with_umask

    old = CONFIG["permissions_umask"]
    CONFIG["permissions_umask"] = None
    try:
        before = os.umask(0o022)
        os.umask(before)
        with _with_umask():
            inside = os.umask(0o022)
            os.umask(inside)
        assert inside == before
    finally:
        CONFIG["permissions_umask"] = old
