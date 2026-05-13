"""Atomic-rename + async-unlink helper for fast removal of large directories.

Mirrors R misha's `.gdb.trash` (R commits f46e4463, 3ff59742, a4a1e6bd, 6e3dab3d).

`os.rename` to a hidden sibling completes in microseconds even for directories
with millions of files. The actual cleanup runs in a detached background
process so it does not tie up the Python session. The same module also
sweeps stale trash entries on `gdb_init` to bound on-disk garbage.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path


def _gdb_trash(path: str | os.PathLike[str], async_unlink: bool = True) -> bool:
    """Rename ``path`` to a hidden sibling, then unlink (background by default).

    Parameters
    ----------
    path : str or PathLike
        Target directory or file to remove.
    async_unlink : bool, default True
        When True, the actual unlink runs in a detached background process
        (POSIX `rm -rf`). When False, runs `shutil.rmtree` synchronously.

    Returns
    -------
    bool
        True if ``path`` is gone after the call (rename succeeded, or sync
        fallback cleared it). False if neither rename nor fallback unlink
        could clear ``path``.
    """
    p = Path(path)
    if not p.exists() and not p.is_symlink():
        return False

    parent = p.parent
    rand = secrets.token_hex(4)
    trash_path = parent / f".trash.{p.name}.{os.getpid()}.{rand}"

    try:
        os.rename(p, trash_path)
    except OSError:
        # Cross-filesystem or other rename failure - fall back to sync rmtree.
        shutil.rmtree(p, ignore_errors=True)
        return not p.exists()

    if async_unlink:
        try:
            subprocess.Popen(
                ["rm", "-rf", "--", str(trash_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            shutil.rmtree(trash_path, ignore_errors=True)
    else:
        shutil.rmtree(trash_path, ignore_errors=True)
    return True


def _gdb_trash_sweep_old(
    parent: str | os.PathLike[str],
    max_age_hours: float = 24.0,
) -> int:
    """Remove stale trash + tmp siblings in ``parent`` older than ``max_age_hours``.

    Sweeps two patterns left behind by interrupted operations:

    * ``.trash.*`` - atomic-rename targets from `_gdb_trash` whose detached
      background `rm` was killed before completion.
    * ``.<name>.tmp.<pid>.<rand>`` - tmp directories from interrupted atomic
      `gtrack_create_*` calls that never reached the final rename.

    Called by `gdb_init` to bound on-disk garbage from prior sessions.
    Entries whose mtime cannot be read (e.g. permission denied) are
    intentionally skipped so we do not loop on inaccessible junk.

    Returns
    -------
    int
        Number of stale entries removed.
    """
    parent_p = Path(parent)
    if not os.path.isdir(parent_p):
        return 0
    try:
        names = os.listdir(parent_p)
    except OSError:
        return 0
    cutoff = time.time() - max_age_hours * 3600.0
    count = 0
    for name in names:
        # `.trash.*` from _gdb_trash, plus `.<name>.tmp.<pid>.<rand>` from
        # interrupted atomic creates. Sweep only fires under <groot>/tracks/
        # and the 24h cutoff guards short-lived legitimate tmp files.
        if not (
            name.startswith(".trash.")
            or (name.startswith(".") and ".tmp." in name)
        ):
            continue
        entry = parent_p / name
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            count += 1
    return count
