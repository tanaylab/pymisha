"""On-disk safety: index-cache invalidation, gtrack_modify staging, fsync.

Three defects that share an acceptance shape - what does the database hold after
something goes wrong, or after a path stops being what it was.

P3: GenomeTrack::get_track_index memoised a parsed index per track directory and
nothing ever dropped it, so a path that changed layout kept being read through
the old one. tests/test_indexed_cache_invalidation.py is green for the wrong
reason: its rm/recreate cycles rebuild tracks with identical interval shapes, so
a stale index is byte-identical and the test passes either way. The discriminating
case is a swap - two tracks exchanging directories.

P4: gtrack_modify opened the live track "rb+" and rewrote floats in place. It is
now staged like every other writer.
"""

import os
import time
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

import pymisha as pm

from test_indexed_cache_invalidation import (  # noqa: F401
    _write_per_chrom_db,
    indexed_test_db,
    restore_db,
)


def _dense(name, value, binsize=200):
    intervs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [5000]})
    pm.gtrack_create_dense(name, "test", intervs, [float(value)], binsize=binsize)
    return intervs


def test_index_cache_survives_a_track_directory_swap(indexed_test_db):
    """dense -> x, sparse -> dense, then read 'dense'.

    Mirrors misha's test-index-cache-invalidation.R:222-256. Without
    invalidation the second read routes through the index parsed for the dense
    track that used to live at that path.
    """
    intervs = _dense("swap_dense", 7)

    starts = list(range(0, 5000, 55))
    pm.gtrack_create_sparse(
        "swap_sparse",
        "test",
        pd.DataFrame(
            {"chrom": ["chr1"] * len(starts), "start": starts, "end": [s + 50 for s in starts]}
        ),
        [float(i) for i in range(len(starts))],
    )

    # Warm the cache for both directories.
    assert len(pm.gextract("swap_dense", intervals=intervs, iterator=200)) > 0
    assert len(pm.gextract("swap_sparse", intervals=intervs)) > 0

    pm.gtrack_mv("swap_dense", "swap_parked")
    pm.gtrack_mv("swap_sparse", "swap_dense")

    # "swap_dense" is now the sparse track. Reading it through a stale index is
    # either an error or the wrong values.
    assert pm.gtrack_info("swap_dense")["type"] == "sparse"
    got = pm.gextract("swap_dense", intervals=intervs)
    assert len(got) == len(starts)


def test_gtrack_modify_still_modifies(indexed_test_db):
    """The staging must not change the result, only where it is written."""
    intervs = _dense("mod_ok", 3)
    before = pm.gextract("mod_ok", intervals=intervs, iterator=200)
    pm.gtrack_modify("mod_ok", "mod_ok * 2", intervals=intervs)
    after = pm.gextract("mod_ok", intervals=intervs, iterator=200)

    assert (after["mod_ok"] == before["mod_ok"] * 2).all()
    # No staging directory left behind.
    track_dir = pm._shared._os.path.dirname(pm._pymisha.pm_track_path("mod_ok"))
    leftovers = [n for n in os.listdir(track_dir) if ".tmp." in n and n.startswith(".")]
    assert leftovers == [], leftovers


def test_killed_gtrack_modify_never_leaves_a_mixed_track(tmp_path, restore_db):
    """SIGKILL mid-modify: the track must be all-old or all-new, never mixed.

    That invariant, not "unchanged", is what staging buys and it is what makes
    the test non-flaky: whether the kill lands before or after the commit is a
    race, but a half-old/half-new track is a failure either way. Before staging
    the writer rewrote floats in place, so a kill mid-loop left exactly that -
    structurally valid, with nothing to mark it as damaged.
    """
    root = tmp_path / "db"
    _write_per_chrom_db(root, [("chr1", "A" * 4_000_000)])
    pm.gdb_convert_to_indexed(groot=str(root), force=True, validate=False)
    pm.gdb_init(str(root))

    scope = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [4_000_000]})
    pm.gtrack_create_dense("mod_kill", "test", scope, [5.0], binsize=1)

    marker = tmp_path / "started"
    script = textwrap.dedent(
        f"""
        import pandas as pd, pymisha as pm
        pm.gdb_init({str(root)!r})
        ivs = pd.DataFrame({{"chrom": ["chr1"], "start": [0], "end": [4_000_000]}})
        open({str(marker)!r}, "w").close()
        pm.gtrack_modify("mod_kill", "mod_kill * 100", intervals=ivs)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    deadline = time.time() + 60
    while not marker.exists() and proc.poll() is None and time.time() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=60)

    pm.gdb_reload()
    got = pm.gextract("mod_kill", intervals=scope, iterator=1)["mod_kill"]
    distinct = set(got.unique())
    assert distinct in ({5.0}, {500.0}), f"half-old/half-new track: {sorted(distinct)}"


def test_fsync_tree_syncs_files_and_directories(tmp_path):
    """_fsync_tree walks depth-first and must not choke on nesting or symlinks."""
    from pymisha.tracks import _fsync_dir, _fsync_tree

    (tmp_path / "sub").mkdir()
    (tmp_path / "a").write_text("a")
    (tmp_path / "sub" / "b").write_text("b")
    (tmp_path / "link").symlink_to(tmp_path / "a")

    _fsync_tree(tmp_path)
    _fsync_dir(tmp_path)

    with pytest.raises(OSError):
        _fsync_dir(tmp_path / "does-not-exist")
